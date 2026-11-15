# -*- coding: utf-8 -*-
"""
主运行入口：策略选型 → 固定杠杆标定 → 回测 → 组合构建 → 结果落盘。

防前视约定：
  1) 每个策略先跑「gross=1」的原始权重，仅用样本内(2023-07~2025-06)收益
     估计一个固定杠杆倍数 scale，使样本内年化波动对齐到 TARGET_VOL；
  2) 样本外(2025-07~2026-11) 原样沿用该 scale，不重新估计、不调参；
  3) 组合权重或来自文章框架先验，或来自样本内协方差的风险平价解，均固定。

诚实性约定：
  - 中高频 CTA 的日线代理、无协整检验的价差套利在回测中失效，
    保留为「可行性诊断项」，不进入主组合，避免用失真结果污染组合结论；
  - 「第二层缺口」用解析推演（显式假设零相关）而非编造历史净值。
"""
import os, json
import numpy as np
import pandas as pd

from config import (DATA, OUT, IN_START, IN_END, OOS_START, OOS_END,
                    TARGET_VOL, TRADING_DAYS, NAME_MAP, COMMODITY)
from data_prep import build
from strategies import build_all
from engine import backtest, metrics, nav_from, erc_weights


def estimate_scale(pnl_in, target=TARGET_VOL, cap=3.0):
    v = pnl_in.std() * np.sqrt(TRADING_DAYS)
    if v < 1e-6:
        return 1.0
    return float(np.clip(target / v, 0.2, cap))


def cost_breakdown(W0, ret, cost_bps, scale):
    """毛收益 / 换手成本 / 日均换手分解，用于判断策略是「无 alpha」还是「被成本吃掉」。"""
    w = W0.reindex(index=ret.index, columns=ret.columns).fillna(0.0) * scale
    to = w.diff().abs().sum(axis=1)
    to.iloc[0] = w.iloc[0].abs().sum()
    gross = (w.shift(1) * ret).sum(axis=1)
    c = to * cost_bps / 1e4
    return {
        'gross_ann': round(float(gross.mean() * TRADING_DAYS), 4),
        'cost_ann': round(float(c.mean() * TRADING_DAYS), 4),
        'net_ann': round(float((gross - c).mean() * TRADING_DAYS), 4),
        'turnover_daily': round(float(to.mean()), 3),
    }


def second_layer_gap(port_pnl_oos, sigma_a=TARGET_VOL, weight=0.15):
    """
    解析推演（非回测）：若第二层「稀缺阿尔法」为与现有组合零相关的独立收益源，
    给定不同假设夏普，组合夏普能改善多少。用于回答「缺失的一层值多少钱」。
    """
    p = port_pnl_oos.dropna()
    sigma_p = p.std() * np.sqrt(TRADING_DAYS)
    ret_p = p.mean() * TRADING_DAYS
    out = {'base_ann_ret': round(float(ret_p), 4),
           'base_ann_vol': round(float(sigma_p), 4),
           'base_sharpe': round(float(ret_p / sigma_p), 3),
           'weight': weight, 'assumed_alpha_vol': sigma_a,
           'scenarios': []}
    for sa in [0.0, 0.5, 1.0, 1.5]:
        ret_a = sigma_a * sa
        ret_new = (1 - weight) * ret_p + weight * ret_a
        vol_new = np.sqrt((1 - weight) ** 2 * sigma_p ** 2 + weight ** 2 * sigma_a ** 2)
        out['scenarios'].append({
            'assumed_alpha_sharpe': sa,
            'new_ann_ret': round(float(ret_new), 4),
            'new_ann_vol': round(float(vol_new), 4),
            'new_sharpe': round(float(ret_new / vol_new), 3),
        })
    return out


def main():
    bundle = build()
    ret = bundle['ret']
    specs, n_pairs = build_all(bundle)

    # ---------- 单策略：样本内标定固定杠杆，全期回测 ----------
    strategy_pnl, strategy_meta = {}, {}
    for name, (W0, cost, layer, desc) in specs.items():
        raw = backtest(W0, ret, cost_bps=cost, target_vol=None, max_gross=None)
        scale = estimate_scale(raw.loc[IN_START:IN_END])
        pnl = raw * scale
        strategy_pnl[name] = pnl
        strategy_meta[name] = {
            'layer': layer, 'desc': desc, 'cost_bps': cost,
            'scale': round(scale, 3),
            'in_vol_raw': round(float(raw.loc[IN_START:IN_END].std() * np.sqrt(TRADING_DAYS)), 4),
            'cost_breakdown': cost_breakdown(W0, ret, cost, scale),
            'metrics_all': metrics(pnl.loc[IN_START:OOS_END]),
            'metrics_in': metrics(pnl.loc[IN_START:IN_END]),
            'metrics_oos': metrics(pnl.loc[OOS_START:OOS_END]),
        }
        m = strategy_meta[name]['metrics_oos']
        print(f'{name:12s} scale={scale:5.2f}  OOS 年化={m.get("ann_ret")} '
              f'夏普={m.get("sharpe")} 回撤={m.get("mdd")}')

    S = pd.DataFrame(strategy_pnl)

    # ---------- 中高频代理成本敏感性 ----------
    hf_W = specs['hf_mr'][0]
    hf_scale = strategy_meta['hf_mr']['scale']
    hf_sens = []
    for bp in [0, 1, 2, 4, 8]:
        p = backtest(hf_W * hf_scale, ret, cost_bps=bp, target_vol=None, max_gross=None)
        hf_sens.append({'cost_bps': bp,
                        'metrics_all': metrics(p.loc[IN_START:OOS_END])})

    # ---------- 组合定义（主组合只纳入可回测策略） ----------
    core = ['beta', 'trend_lf', 'trend_raw', 'xs_mom', 'macro_rp']
    R_in = S[core].loc[IN_START:IN_END]
    cov = R_in.cov().values * TRADING_DAYS
    erc_w = erc_weights(cov)
    erc_dict = {k: round(float(v), 4) for k, v in zip(core, erc_w)}

    PORTFOLIOS = {
        'P0 单一股票贝塔（对照）': {'beta': 1.00},
        'P1 第一层·贝塔基石': {'beta': 0.50, 'trend_lf': 0.50},
        'P2 第一层·趋势纯度增强': {'beta': 0.45, 'trend_lf': 0.35, 'trend_raw': 0.20},
        'P3 两层·可回测阿尔法': {'beta': 0.35, 'trend_lf': 0.25, 'trend_raw': 0.15,
                            'xs_mom': 0.15, 'macro_rp': 0.10},
        'P4 三层完整（文章框架）': {'beta': 0.22, 'trend_lf': 0.20, 'trend_raw': 0.12,
                             'xs_mom': 0.10, 'macro_rp': 0.36},
        'P5 三层·样本内风险平价': erc_dict,
        'P6 传统 60/40 股债（对照）': {'beta': 0.60, 'bond': 0.40},
        'P7 纯 CTA 多策略（无股票）': {'trend_lf': 0.40, 'trend_raw': 0.30, 'xs_mom': 0.30},
        'P8 三层·波动率过滤（对照）': {'beta': 0.22, 'trend_vol': 0.20, 'trend_raw': 0.12,
                              'xs_mom': 0.10, 'macro_rp': 0.36},
    }
    # 敏感性对照：强行纳入失效的日线中高频代理（仅用于展示失真，不参与推荐）
    PORTFOLIOS['P9 三层·强行纳入日线高频代理（失真对照）'] = {
        'beta': 0.22, 'trend_lf': 0.20, 'trend_raw': 0.09,
        'xs_mom': 0.08, 'macro_rp': 0.28, 'hf_mr': 0.13}

    port_pnl, port_meta = {}, {}
    for pname, weights in PORTFOLIOS.items():
        p = sum(S[k] * w for k, w in weights.items())
        port_pnl[pname] = p
        port_meta[pname] = {
            'weights': weights,
            'metrics_all': metrics(p.loc[IN_START:OOS_END]),
            'metrics_in': metrics(p.loc[IN_START:IN_END]),
            'metrics_oos': metrics(p.loc[OOS_START:OOS_END]),
        }
        m = port_meta[pname]['metrics_oos']
        print(f'{pname:30s} OOS 年化={m.get("ann_ret")} 夏普={m.get("sharpe")} '
              f'回撤={m.get("mdd")} Calmar={m.get("calmar")}')

    P = pd.DataFrame(port_pnl)
    main_ports = [k for k in PORTFOLIOS if not k.startswith('P9')]

    # ---------- 落盘 ----------
    S.loc[IN_START:OOS_END].to_csv(os.path.join(OUT, 'strategy_returns.csv'))
    P.loc[IN_START:OOS_END].to_csv(os.path.join(OUT, 'portfolio_returns.csv'))

    with open(os.path.join(DATA, 'eda_report.json')) as f:
        eda = json.load(f)
    with open(os.path.join(DATA, 'fetch_report.json')) as f:
        fetch = json.load(f)

    idx = P.loc[IN_START:OOS_END].index
    nav_p = nav_from(P.loc[IN_START:OOS_END])
    nav_s = nav_from(S.loc[IN_START:OOS_END])
    dd_p = nav_p / nav_p.cummax() - 1

    result = {
        'meta': {
            'data_source': fetch['source'], 'data_range': fetch['merged_range'],
            'n_assets': len(ret.columns), 'n_commodity': len(COMMODITY),
            'n_pairs': n_pairs, 'cleaned_outliers': bundle['n_bad'],
            'in': [IN_START, IN_END], 'oos': [OOS_START, OOS_END],
            'target_vol': TARGET_VOL, 'trading_days': TRADING_DAYS,
            'name_map': NAME_MAP,
            'price_convention': '商品=动态结算价；HS300=收盘价',
        },
        'eda': eda,
        'strategies': strategy_meta,
        'portfolios': port_meta,
        'main_portfolios': main_ports,
        'erc_weights': erc_dict,
        'core_strategies': core,
        'hf_cost_sensitivity': hf_sens,
        'second_layer_gap': second_layer_gap(
            P['P4 三层完整（文章框架）'].loc[OOS_START:OOS_END]),
        'corr_oos': S.loc[OOS_START:OOS_END][core].corr().round(3).to_dict(),
        'curves': {
            'dates': [d.strftime('%Y-%m-%d') for d in idx],
            'port_nav': {c: [round(float(x), 5) for x in nav_p[c]] for c in main_ports},
            'port_dd': {c: [round(float(x), 5) for x in dd_p[c]] for c in main_ports},
            'strat_nav': {c: [round(float(x), 5) for x in nav_s[c]] for c in core},
            'sample_split_date': OOS_START,
        },
    }
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n样本内风险平价权重:', erc_dict)
    print('第二层缺口解析:', json.dumps(result['second_layer_gap'], ensure_ascii=False))
    print('结果已写入', OUT)
    return result


if __name__ == '__main__':
    main()
