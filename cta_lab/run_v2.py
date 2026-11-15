# -*- coding: utf-8 -*-
"""
V2 主入口：按报告第七节的五层迭代顺序组装，并与 V1 在【同一成本口径】下对比。

  执行层：分品种流动性成本、非线性冲击（已校准至与 V1 可比）、单日换手上限、保证金约束
  策略层：协整价差、受控反转、变色止损趋势、市场中性代理；并按样本内表现做策略池筛选
  组合层：ML walk-forward 动态权重（仅限微调）+ 差分进化调参（maximin + 正则）
          + 动态风险预算 + 回撤刹车（只降不升）
  验证层：bootstrap、参数敏感性、分年度子样本、ML 增量诊断

样本内 {IN} 用于策略标定、策略池筛选、ML 训练与进化调参；
样本外 {OOS} 不参与任何拟合、筛选与参数选择。
注意：ML 模型按 walk-forward 规则在样本外期间持续用【截至当日的历史数据】在线更新，
      这是标准的在线学习，不使用任何未来信息；超参与策略池则在样本外全程冻结。
"""
import os, json, warnings
import numpy as np
import pandas as pd

from config import (DATA, OUT, IN_START, IN_END, OOS_START, OOS_END,
                    TARGET_VOL, TRADING_DAYS, PAIRS)
from data_prep import build
from strategies import (build_all, cointegrated_pairs, spread_arb_v2,
                        trend_mix_v2, trend_lf_v2, style_neutral)
from execution import build_market_profile, strategy_cost_rate, margin_cap
from engine import (backtest_exec, apply_risk_overlay, metrics, nav_from,
                    erc_weights)
from ml_portfolio import (build_panel, walkforward_predict,
                          build_risk_panel, walkforward_risk,
                          weights_from_pred, combine, evaluate, evolve)

warnings.filterwarnings('ignore')

HORIZON = 20
REBAL_BPS = 2.0
SCREEN_SHARPE = -0.50      # 样本内夏普低于此阈值的策略不进入组合池


def calibrate(W, ret, exec_kw, base_bps, liq, cols, in_start=IN_START, in_end=IN_END):
    """样本内标定固定杠杆（迭代 2 次以吸收非线性冲击成本），样本外沿用。"""
    cr = strategy_cost_rate(base_bps, liq, cols)
    scale = 1.0
    for _ in range(2):
        raw = backtest_exec(W * scale, ret, cost_rate=cr, **exec_kw)
        v = raw.loc[in_start:in_end].std() * np.sqrt(TRADING_DAYS)
        if v < 1e-6:
            break
        scale = float(np.clip(scale * TARGET_VOL / v, 0.2, 3.0))
    pnl = backtest_exec(W * scale, ret, cost_rate=cr, **exec_kw)
    return pnl, round(scale, 3)


def block_bootstrap(pnl, n=1200, block=10, seed=7):
    rng = np.random.default_rng(seed)
    x = pnl.values
    L = len(x)
    shp, mdd = [], []
    for _ in range(n):
        idx = []
        while len(idx) < L:
            s = rng.integers(0, max(1, L - block))
            idx.extend(range(s, min(s + block, L)))
        y = x[idx[:L]]
        mu, sd = y.mean(), y.std()
        shp.append(mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0)
        nav = np.cumprod(1 + y)
        mdd.append((nav / np.maximum.accumulate(nav) - 1).min())
    return {'sharpe_ci': [round(float(np.percentile(shp, 2.5)), 3),
                          round(float(np.percentile(shp, 97.5)), 3)],
            'mdd_ci': [round(float(np.percentile(mdd, 2.5)), 4),
                       round(float(np.percentile(mdd, 97.5)), 4)]}


def const_weights(cfg, index, columns):
    w = pd.DataFrame(0.0, index=index, columns=columns)
    for k, v in cfg.items():
        if k in w.columns:
            w[k] = v
    return w.div(w.sum(axis=1), axis=0)


def main():
    bundle = build()
    px, ret = bundle['px'], bundle['ret']
    liq, adv, amplitude = build_market_profile(bundle)
    cols = list(ret.columns)
    mg = margin_cap(liq, cols)
    exec_kw = dict(adv=adv, impact_k=0.0001, turnover_cap=0.25,
                   max_gross=mg, amplitude=amplitude)
    print('执行层: 保证金敞口上限', round(mg, 2), '| 冲击系数 0.0001（已校准）')

    specs_v1, _ = build_all(bundle)
    good_pairs, coin_stats = cointegrated_pairs(px, PAIRS)

    POOL = {
        'beta':          (specs_v1['beta'][0],      1.0, '第一层·股票贝塔'),
        'trend_lf':      (specs_v1['trend_lf'][0],  3.0, '第一层·趋势ER过滤'),
        'trend_raw':     (specs_v1['trend_raw'][0], 3.0, '第一层·快速趋势'),
        'xs_mom':        (specs_v1['xs_mom'][0],    5.0, '第二层·截面多空'),
        'macro_rp':      (specs_v1['macro_rp'][0],  2.0, '第三层·宏观配置'),
        'trend_v2':      (trend_lf_v2(px, ret, er_low=0.12, er_high=0.30, stop_k=1.2), 3.0, 'V2·变色止损趋势'),
        'trend_fast_v2': (trend_lf_v2(px, ret, windows=(10, 20, 40, 60, 120),
                                      smooth=0, quant=0, stop_k=1.5), 3.0, 'V2·快窗口止损趋势'),
        'spread_v2':     (spread_arb_v2(px, ret, good_pairs, z_entry=2.0, z_exit=0.6), 3.0, 'V2·协整价差'),
        'trend_mix_v2':  (trend_mix_v2(px, ret),    3.5, 'V2·受控反转防守'),
        'style_neutral': (style_neutral(px, ret),   2.0, 'V2·市场中性代理'),
        'bond':          (specs_v1['bond'][0],      1.0, '防御·国债期货'),
    }

    S, meta = {}, {}
    for name, (W, bps, layer) in POOL.items():
        p, sc = calibrate(W, ret, exec_kw, bps, liq, cols)
        S[name] = p
        meta[name] = {'layer': layer, 'cost_bps': bps, 'scale': sc,
                      'metrics_in': metrics(p.loc[IN_START:IN_END]),
                      'metrics_oos': metrics(p.loc[OOS_START:OOS_END])}
    S = pd.DataFrame(S)

    # 策略池筛选：只用样本内夏普，样本外全程不参与
    screened_out = {k: v['metrics_in']['sharpe'] for k, v in meta.items()
                    if v['metrics_in']['sharpe'] < SCREEN_SHARPE}
    core = [k for k in POOL if k not in screened_out]
    print(f'策略池筛选: 保留 {len(core)} / {len(POOL)}，剔除 '
          f'{ {k: round(v, 2) for k, v in screened_out.items()} }')

    # ---------------- 基准 ----------------
    v1 = pd.read_csv(os.path.join(OUT, 'portfolio_returns.csv'),
                     parse_dates=['date']).set_index('date')
    v1_pnl = v1.get('P4 三层完整（文章框架）', v1.iloc[:, 4])

    PRIOR_V1 = {'beta': 0.22, 'trend_lf': 0.20, 'trend_raw': 0.12,
                'xs_mom': 0.10, 'macro_rp': 0.36}
    PRIOR_V2 = {'beta': 0.20, 'trend_lf': 0.10, 'trend_v2': 0.08,
                'trend_raw': 0.08, 'trend_fast_v2': 0.14, 'xs_mom': 0.08,
                'macro_rp': 0.27, 'bond': 0.05}
    w_v1 = const_weights(PRIOR_V1, S.index, S.columns)
    w_v2 = const_weights(PRIOR_V2, S.index, S.columns)

    cov = S[core].loc[IN_START:IN_END].cov().values * TRADING_DAYS
    erc = {k: round(float(v), 4) for k, v in zip(core, erc_weights(cov))}

    # ---------------- ML ----------------
    X, y, feat_names = build_panel(bundle, S[core], core, horizon=HORIZON)
    pred, fold_log = walkforward_predict(X, y, core, train_window=300,
                                          step=10, horizon=HORIZON,
                                          ridge_alpha=30.0, blend=0.5)
    print(f'ML: {len(feat_names)} 特征 / {len(fold_log)} folds / {len(core)} 策略')

    anchor = pd.Series(PRIOR_V2).reindex(core).fillna(0.0)
    anchor = anchor / anchor.sum()

    # ML 路径 2：风险状态预测（walk-forward 训练，仅用历史）
    p_prior, _ = combine(S, w_v2, REBAL_BPS)
    feat_r, fvol, fdd = build_risk_panel(bundle, p_prior, horizon=HORIZON)
    pred_vol, pred_risk = walkforward_risk(feat_r, fvol, fdd,
                                            train_window=300, step=10,
                                            horizon=HORIZON)
    print(f'ML风险层: 预测未来波动样本外均值 '
          f'{pred_vol.loc[OOS_START:OOS_END].mean():.4f}，'
          f'高回撤概率均值 {pred_risk.loc[OOS_START:OOS_END].mean():.3f}')

    best, fit, nfev = evolve(S[core], pred, anchor, IN_START, IN_END,
                             maxiter=14, popsize=6, seed=42,
                             rebal_bps=REBAL_BPS, pred_vol=pred_vol)
    names = ['temperature', 'lambda_ml', 'w_cap', 'target_vol',
             'vol_window', 'dd_threshold', 'dd_floor', 'mu_ml']
    best_params = {k: round(float(v), 4) for k, v in zip(names, best)}
    print('进化最优参数:', best_params, '适应度', round(fit, 4), '评估次数', nfev)

    # ---------------- 变体 ----------------
    variants = {}
    def add(name, p, note):
        variants[name] = {'pnl': p, 'note': note,
                          'metrics_all': metrics(p.loc[IN_START:OOS_END]),
                          'metrics_in': metrics(p.loc[IN_START:IN_END]),
                          'metrics_oos': metrics(p.loc[OOS_START:OOS_END])}

    RISK_COMMON = dict(target_vol=0.10, vol_window=40,
                       dd_threshold=0.05, dd_floor=0.40, vol_cap=(0.35, 1.0))

    add('B0 V1原报告P4（旧成本参照）', v1_pnl.loc[IN_START:OOS_END],
        'V1 原始结果，仅作历史参照，成本口径不同')
    pB1, _ = combine(S, w_v1, REBAL_BPS)
    add('B1 公平基线（新成本+V1权重）', pB1,
        '与所有 V2 变体同一执行成本口径，真正的对比基准')
    pA, _ = combine(S, w_v2, REBAL_BPS)
    add('V2-A 策略层改进', pA, '策略池筛选 + 低回撤趋势替换 + 先验权重')
    pB, _, _ = apply_risk_overlay(pA, **RISK_COMMON)
    add('V2-B 叠加动态风控', pB, 'A + 波动率目标与回撤刹车（只降不升）')
    wC = weights_from_pred(pred, 0.40, 0.20, 0.45, w_anchor=anchor)
    pC, _ = combine(S[core], wC, REBAL_BPS)
    pC, _, _ = apply_risk_overlay(pC, **RISK_COMMON)
    add('V2-C 叠加ML微调', pC, 'B + ML walk-forward 小幅动态权重（未调参）')

    wD = weights_from_pred(pred, best[0], best[1], best[2], w_anchor=anchor)
    pD, _, _, lev_full = evaluate(S[core], pred, anchor, best,
                                  IN_START, OOS_END, REBAL_BPS,
                                  pred_vol=pred_vol)
    add('V2-D 完整（ML+进化+风控）', pD,
        '样本内进化调参冻结后，样本外一次前向推断；含 ML 收益微调 + ML 风险杠杆')

    # 完全无 ML 对照：同时关闭 ML 权重与 ML 风险杠杆
    p_no_ml = list(best)
    p_no_ml[1], p_no_ml[7] = 0.0, 0.0
    pE, _, _, _ = evaluate(S[core], pred, anchor, p_no_ml,
                           IN_START, OOS_END, REBAL_BPS, pred_vol=pred_vol)
    add('V2-E 完全无ML对照', pE, '检验两条 ML 路径的真实增量')

    for k, v in variants.items():
        m = v['metrics_oos']
        print(f"{k:30s} IN shp={v['metrics_in']['sharpe']:5.2f} | "
              f"OOS 年化={m['ann_ret']:7.4f} 夏普={m['sharpe']:6.2f} "
              f"回撤={m['mdd']:7.4f} Calmar={m['calmar']}")

    # ---------------- 验证层 ----------------
    oosD = variants['V2-D 完整（ML+进化+风控）']['pnl'].loc[OOS_START:OOS_END]
    oosB1 = variants['B1 公平基线（新成本+V1权重）']['pnl'].loc[OOS_START:OOS_END]
    oosV1 = variants['B0 V1原报告P4（旧成本参照）']['pnl'].loc[OOS_START:OOS_END]
    boot = {'B0_V1': block_bootstrap(oosV1), 'B1_公平基线': block_bootstrap(oosB1),
            'V2-D': block_bootstrap(oosD)}

    sens = []
    for i, nm in enumerate(names):
        for f in [0.7, 1.0, 1.3]:
            p = list(best)
            p[i] = p[i] * f
            if nm == 'vol_window':
                p[i] = max(10, int(round(p[i])))
            _, m, _, _ = evaluate(S[core], pred, anchor, p,
                                  OOS_START, OOS_END, REBAL_BPS,
                                  pred_vol=pred_vol)
            sens.append({'param': nm, 'factor': f, 'value': round(float(p[i]), 4),
                         'sharpe': m['sharpe'], 'mdd': m['mdd']})

    periods = {'2023H2': ('2023-07-01', '2023-12-31'), '2024': ('2024-01-01', '2024-12-31'),
               '2025H1': ('2025-01-01', '2025-06-30'), '2025H2': ('2025-07-01', '2025-12-31'),
               '2026': ('2026-01-01', '2026-11-13')}
    subsample = {k: {'B1': metrics(variants['B1 公平基线（新成本+V1权重）']['pnl'].loc[a:b]),
                     'V2-D': metrics(variants['V2-D 完整（ML+进化+风控）']['pnl'].loc[a:b])}
                 for k, (a, b) in periods.items()}

    d = variants['V2-D 完整（ML+进化+风控）']
    ml_gain = {
        'oos_sharpe_b0': variants['B0 V1原报告P4（旧成本参照）']['metrics_oos']['sharpe'],
        'oos_sharpe_b1': variants['B1 公平基线（新成本+V1权重）']['metrics_oos']['sharpe'],
        'oos_sharpe_v2d': d['metrics_oos']['sharpe'],
        'oos_sharpe_no_ml': variants['V2-E 完全无ML对照']['metrics_oos']['sharpe'],
        'in_vs_oos_gap': round(d['metrics_in']['sharpe'] - d['metrics_oos']['sharpe'], 3),
        'oos_mdd_b1': variants['B1 公平基线（新成本+V1权重）']['metrics_oos']['mdd'],
        'oos_mdd_v2d': d['metrics_oos']['mdd'],
    }

    # ---------------- 落盘 ----------------
    idx = variants['B1 公平基线（新成本+V1权重）']['pnl'].index
    curves = {'dates': [d_.strftime('%Y-%m-%d') for d_ in idx],
              'sample_split_date': OOS_START, 'nav': {}, 'dd': {}}
    for k, v in variants.items():
        p = v['pnl'].reindex(idx).fillna(0.0)
        nav = nav_from(p)
        curves['nav'][k] = [round(float(x), 5) for x in nav]
        curves['dd'][k] = [round(float(x), 5) for x in (nav / nav.cummax() - 1)]
    curves['leverage'] = [round(float(x), 4) for x in lev_full.reindex(idx).fillna(1.0)]
    curves['pred_vol'] = [round(float(x), 4) for x in pred_vol.reindex(idx).fillna(0.1)]

    result = {
        'meta': {'in': [IN_START, IN_END], 'oos': [OOS_START, OOS_END],
                 'n_pool': len(POOL), 'n_core': len(core),
                 'n_features': len(feat_names), 'n_folds': len(fold_log),
                 'horizon': HORIZON, 'rebal_bps': REBAL_BPS,
                 'screen_sharpe': SCREEN_SHARPE,
                 'margin_cap': round(mg, 3),
                 'exec': {'impact_k': 0.0001, 'turnover_cap': 0.25,
                          'illiq_sensitivity': 0.3},
                 'data_range': [str(px.index.min().date()), str(px.index.max().date())]},
        'strategy_meta': meta, 'screened_out': screened_out, 'core': core,
        'cointegration': {'passed': good_pairs, 'pvalue_threshold': 0.10, 'all': coin_stats},
        'erc': erc, 'prior_v1': PRIOR_V1, 'prior_v2': PRIOR_V2,
        'best_params': best_params,
        'evolution': {'fitness': round(fit, 4), 'nfev': nfev,
                      'constraints': 'maximin 3 折 + 正则 + 收紧搜索空间 + 杠杆只降不升'},
        'ml': {'features': feat_names, 'folds': fold_log[-6:],
               'risk_features': list(feat_r.columns),
               'pred_vol_oos_mean': round(float(pred_vol.loc[OOS_START:OOS_END].mean()), 4),
               'pred_risk_oos_mean': round(float(pred_risk.loc[OOS_START:OOS_END].mean()), 4),
               'weight_curve': {c: [round(float(x), 4) for x in wD[c].reindex(idx).ffill().fillna(0)]
                                for c in core}},
        'variants': {k: {kk: vv for kk, vv in v.items() if kk != 'pnl'}
                     for k, v in variants.items()},
        'validation': {'bootstrap': boot, 'sensitivity': sens,
                       'subsample': subsample, 'ml_gain': ml_gain},
        'curves': curves,
    }
    with open(os.path.join(OUT, 'results_v2.json'), 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    pd.DataFrame({k: v['pnl'] for k, v in variants.items()}).to_csv(
        os.path.join(OUT, 'v2_variant_returns.csv'))
    S.loc[IN_START:OOS_END].to_csv(os.path.join(OUT, 'v2_strategy_returns.csv'))

    print('\nBootstrap V2-D 夏普95%CI', boot['V2-D']['sharpe_ci'],
          '回撤95%CI', boot['V2-D']['mdd_ci'])
    print('ML增量诊断', ml_gain)
    print('结果已写入', OUT)
    return result


if __name__ == '__main__':
    main()
