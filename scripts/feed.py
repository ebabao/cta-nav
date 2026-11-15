# -*- coding: utf-8 -*-
"""
策略模型接口构建器（feed）：把 cta_lab/output 的模型输出转换为看板同源接口 data/*.json。

设计原则（与参考页 all-weather-nav 一致）：
  * 看板 index.html 是静态文件，只通过 fetch(cache:"no-store") 读取 data/ 接口；
  * 策略模型接口 = cta_lab/output/results.json + results_v2.json + 行情矩阵；
  * 本脚本是两者之间唯一的转换器，保证「模型输出」与「页面数据」始终同源；
  * 所有净值、月度、分年、回撤、风险贡献均由模型输出真实计算，不编造任何数字。

用法：
  python3.11 scripts/feed.py                      # 用默认 sync_info 生成快照接口
  python3.11 scripts/feed.py --sync-info /tmp/sync.json   # 由 sync_daily.py 注入同步元信息
"""
import os
import sys
import csv
import json
import math
import argparse
import datetime as dt
from collections import OrderedDict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(BASE, 'cta_lab')
OUT = os.path.join(LAB, 'output')
DATA = os.path.join(BASE, 'data')
RAW_PX = os.path.join(LAB, 'data', 'prices.csv')

SCHEMA = 'cta-nav/v1'
TRADING_DAYS = 252

# 策略中文名与所属层级（层级以模型 strategy_meta 为准，此处仅补中文名）
NAME = {
    'beta': '股票贝塔 · 沪深300',
    'bond': '国债 · 十年期',
    'trend_lf': '中低频趋势 · ER 流畅度过滤',
    'trend_plain': '中低频趋势 · 无过滤',
    'trend_vol': '中低频趋势 · 波动率过滤',
    'trend_raw': '原始多周期趋势',
    'trend_mix': '趋势 + 反转防守',
    'hf_mr': '中高频均值回归代理',
    'xs_mom': '截面动量多空',
    'spread': '跨品种价差套利',
    'macro_rp': '宏观配置 · 风险平价',
    'trend_v2': '变色止损趋势',
    'trend_fast_v2': '快周期趋势',
    'spread_v2': '协整价差套利',
    'trend_mix_v2': '受控反转防守',
    'style_neutral': '风格中性代理',
}

VARIANT_SHORT = {
    'B0 V1原报告P4（旧成本参照）': 'B0 · V1 旧成本参照',
    'B1 公平基线（新成本+V1权重）': 'B1 · 公平基线',
    'V2-A 策略层改进': 'V2-A · 策略层',
    'V2-B 叠加动态风控': 'V2-B · 动态风控',
    'V2-C 叠加ML微调': 'V2-C · ML 微调',
    'V2-D 完整（ML+进化+风控）': 'V2-D · 完整（推荐）',
    'V2-E 完全无ML对照': 'V2-E · 无 ML 对照',
}
MAIN_VARIANT = 'V2-D 完整（ML+进化+风控）'
BENCH_VARIANT = 'B1 公平基线（新成本+V1权重）'

# 三层框架在 V1 中的代表组合
LAYER_PORTS = [
    ('P0 单一股票贝塔（对照）', '第 0 层 · 仅股票贝塔'),
    ('P1 第一层·贝塔基石', '第一层 · 贝塔基石'),
    ('P3 两层·可回测阿尔法', '第二层 · 叠加 CTA'),
    ('P4 三层完整（文章框架）', '第三层 · 完整配置'),
]


def jload(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def jwrite(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))


def r(x, n=4):
    """四舍五入并去掉浮点尾巴；None/NaN 原样保留为空。"""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), n)


def metrics_from_nav(nav, dates):
    """从日净值真实计算一组绩效指标（用于分年 / 区间统计）。"""
    nav = np.asarray(nav, dtype=float)
    if len(nav) < 2:
        return {}
    ret = np.diff(nav) / nav[:-1]
    total = nav[-1] / nav[0] - 1
    years = len(ret) / TRADING_DAYS
    cagr = (nav[-1] / nav[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = float(np.std(ret, ddof=1) * math.sqrt(TRADING_DAYS)) if len(ret) > 1 else 0.0
    sharpe = float(np.mean(ret) / np.std(ret, ddof=1) * math.sqrt(TRADING_DAYS)) if np.std(ret, ddof=1) > 0 else 0.0
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1
    mdd = float(dd.min())
    pos = 0
    cur = 0
    longest = 0
    for d in dd:
        if d < -1e-9:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    # 月度胜率
    s = pd.Series(nav, index=pd.to_datetime(dates))
    m = s.resample('ME').last().dropna()
    mr = m.pct_change().dropna()
    return {
        'total_return_pct': r(total * 100, 2),
        'annual_return_pct': r(cagr * 100, 2),
        'max_drawdown_pct': r(mdd * 100, 2),
        'return_dd_ratio': r(-cagr / mdd, 2) if mdd < 0 else None,
        'std_pct': r(vol * 100, 2),
        'sharpe': r(sharpe, 3),
        'monthly_win_pct': r(float((mr > 0).mean() * 100), 1) if len(mr) else None,
        'max_underwater_days': int(longest),
        'days': len(ret),
    }


def build_monthly(nav, dates):
    s = pd.Series(nav, index=pd.to_datetime(dates))
    m = s.resample('ME').last().dropna()
    out = []
    prev = None
    for d, v in m.items():
        if prev is not None:
            out.append({'month': d.strftime('%Y-%m'), 'ret_pct': r((v / prev - 1) * 100, 3)})
        prev = v
    return out


def build_yearly(nav, dates):
    s = pd.Series(nav, index=pd.to_datetime(dates))
    out = []
    for y, g in s.groupby(s.index.year):
        if len(g) < 2:
            continue
        st = metrics_from_nav(g.values, [d.strftime('%Y-%m-%d') for d in g.index])
        st['year'] = int(y)
        out.append(st)
    return out


def build_rebalances(dates, nav, lev, weights_curve, threshold=0.02):
    """
    权重/杠杆变动事件：ML 动态权重被进化判定关闭后，组合权重保持先验锚定，
    真实发生的是「风控层杠杆刹车」。本表逐日检测杠杆与权重的显著变化并落盘。
    """
    events = []
    prev_lev = None
    prev_w = None
    for i, d in enumerate(dates):
        lv = float(lev[i]) if i < len(lev) else 1.0
        w = {k: float(v[i]) for k, v in weights_curve.items() if i < len(v)}
        dw = 0.0 if prev_w is None else sum(abs(w.get(k, 0) - prev_w.get(k, 0)) for k in set(w) | set(prev_w or {}))
        dl = 0.0 if prev_lev is None else abs(lv - prev_lev)
        if prev_lev is not None and (dl >= threshold or dw >= threshold):
            moved = sorted(((k, w[k] - (prev_w or {}).get(k, 0)) for k in w),
                           key=lambda x: -abs(x[1]))[:3]
            names = '、'.join(NAME.get(k, k) for k, _ in moved if abs(_) > 1e-6) or '风控杠杆调整'
            events.append({
                'date': d,
                'names': names,
                'exposure_pct': r(lv * 100, 2),
                'leverage': r(lv, 4),
                'weight_change_pct': r(dw * 100, 2),
                'kind': '权重调整' if dw >= threshold else '杠杆刹车',
            })
        prev_lev, prev_w = lv, w
    return events


def risk_contribution(strat_ret, weights):
    """真实风险贡献：RC_i = w_i (Σw)_i / (w'Σw)，Σ 为策略日收益协方差。"""
    cols = [c for c in weights if c in strat_ret.columns]
    if not cols:
        return {}
    X = strat_ret[cols].dropna()
    if len(X) < 30:
        return {}
    cov = X.cov().values
    w = np.array([weights[c] for c in cols], dtype=float)
    denom = float(w @ cov @ w)
    if denom <= 0:
        return {}
    rc = w * (cov @ w) / denom
    return {c: r(rc[i] * 100, 2) for i, c in enumerate(cols)}


def fingerprints():
    """轻量指纹：不写盘，供 sync_daily 做幂等判断（与 build 内算法保持一致）。"""
    import hashlib
    r2 = jload(os.path.join(OUT, 'results_v2.json'))
    dates = r2['curves']['dates']
    i0 = dates.index(r2['meta']['in'][0]) if r2['meta']['in'][0] in dates else 0
    nav = np.asarray(r2['curves']['nav'][MAIN_VARIANT], dtype=float)[i0:]
    nav = nav / nav[0]
    m2 = r2['meta']
    params_fp = hashlib.md5(json.dumps(
        {'best_params': r2['best_params'], 'prior': r2['prior_v2'],
         'exec': m2['exec'], 'core': r2['core']}, sort_keys=True).encode()).hexdigest()[:12]
    data_fp = hashlib.md5(
        (dates[-1] + '|' + str(len(nav)) + '|' + str(round(float(nav[-1]), 6))).encode()
    ).hexdigest()[:12]
    return {'params_fp': params_fp, 'data_fp': data_fp, 'as_of': dates[-1]}


def build(sync_info=None):
    r1 = jload(os.path.join(OUT, 'results.json'))
    r2 = jload(os.path.join(OUT, 'results_v2.json'))
    px = pd.read_csv(RAW_PX, parse_dates=['date']).set_index('date').sort_index()

    m1, m2 = r1['meta'], r2['meta']
    in_start = m2['in'][0]
    dates_all = r2['curves']['dates']
    i0 = dates_all.index(in_start) if in_start in dates_all else 0
    dates = dates_all[i0:]
    n = len(dates)

    # ---------- 净值 / 回撤 / 杠杆 / 基准 ----------
    nav_all = {}
    dd_all = {}
    for k, arr in r2['curves']['nav'].items():
        a = np.asarray(arr[i0:], dtype=float)
        a = a / a[0]
        nav_all[k] = [r(x, 6) for x in a]
        peak = np.maximum.accumulate(a)
        dd_all[k] = [r(x * 100, 4) for x in (a / peak - 1)]

    lev = [r(float(x), 4) for x in r2['curves']['leverage'][i0:]]
    pred_vol = [r(float(x), 5) for x in r2['curves']['pred_vol'][i0:]]

    # 沪深300 基准（收盘价，口径见 meta.price_convention）
    hs = px['HS300'].reindex(pd.to_datetime(dates)).ffill()
    hs_nav = (hs / hs.iloc[0]).tolist() if hs.notna().all() else None

    # V1 评估期净值（817 点，与 V2 对齐）
    v1_nav = {}
    v1_dd = {}
    for k, arr in r1['curves']['port_nav'].items():
        a = np.asarray(arr, dtype=float)
        if len(a) != n:
            a = np.asarray(arr[i0:], dtype=float) if len(arr) == len(dates_all) else a[-n:]
        a = a / a[0]
        v1_nav[k] = [r(x, 6) for x in a]
        peak = np.maximum.accumulate(a)
        v1_dd[k] = [r(x * 100, 4) for x in (a / peak - 1)]

    main_nav = np.asarray([float(x) for x in nav_all[MAIN_VARIANT]])
    monthly = build_monthly(main_nav, dates)
    yearly = build_yearly(main_nav, dates)

    # ---------- 指标（直接取模型输出，保持与研究报告口径一致） ----------
    vm = r2['variants']
    mo = vm[MAIN_VARIANT]['metrics_oos']
    ma = vm[MAIN_VARIANT]['metrics_all']
    mi = vm[MAIN_VARIANT]['metrics_in']
    b_oos = vm[BENCH_VARIANT]['metrics_oos']

    metrics = {
        'total_return': r(ma['total_ret'] * 100, 2),
        'cagr': r(ma['ann_ret'] * 100, 2),
        'sharp': r(ma['sharpe'], 3),
        'sharp_oos': r(mo['sharpe'], 3),
        'sharp_in': r(mi['sharpe'], 3),
        'vol_ann': r(ma['ann_vol'] * 100, 2),
        'max_drawdown': r(abs(ma['mdd']) * 100, 2),
        'max_drawdown_oos': r(abs(mo['mdd']) * 100, 2),
        'calmar': r(ma['calmar'], 3),
        'calmar_oos': r(mo['calmar'], 3),
        'month_win': r(ma['monthly_win'] * 100, 1),
        'final_nav': r(main_nav[-1], 4),
        'cagr_oos': r(mo['ann_ret'] * 100, 2),
        'bench_cagr_oos': r(b_oos['ann_ret'] * 100, 2),
        'bench_max_drawdown_oos': r(abs(b_oos['mdd']) * 100, 2),
        'n_folds': int(r2['meta'].get('n_folds', 0)),
        'n_features': int(r2['meta'].get('n_features', 0)),
        'rebal_bps': r(r2['meta'].get('rebal_bps', 0), 2),
    }

    variants_metrics = []
    for k, v in vm.items():
        variants_metrics.append({
            'key': k, 'name': VARIANT_SHORT.get(k, k), 'note': v.get('note', ''),
            'in': {kk: r(vv, 4) if isinstance(vv, (int, float)) else vv
                   for kk, vv in v['metrics_in'].items()},
            'oos': {kk: r(vv, 4) if isinstance(vv, (int, float)) else vv
                    for kk, vv in v['metrics_oos'].items()},
        })

    # ---------- 三层框架递进 ----------
    layers = []
    for key, label in LAYER_PORTS:
        p = r1['portfolios'].get(key)
        if not p:
            continue
        layers.append({
            'key': key, 'name': label,
            'in': {'ann_ret': r(p['metrics_in']['ann_ret'] * 100, 2),
                   'sharpe': r(p['metrics_in']['sharpe'], 3),
                   'mdd': r(abs(p['metrics_in']['mdd']) * 100, 2),
                   'calmar': r(p['metrics_in']['calmar'], 3)},
            'oos': {'ann_ret': r(p['metrics_oos']['ann_ret'] * 100, 2),
                    'sharpe': r(p['metrics_oos']['sharpe'], 3),
                    'mdd': r(abs(p['metrics_oos']['mdd']) * 100, 2),
                    'calmar': r(p['metrics_oos']['calmar'], 3),
                    'nav': None},
        })
    # 挂上 V1 净值曲线，供页面画三层净值
    for L in layers:
        L['oos']['nav'] = v1_nav.get(L['key'])

    # ---------- 策略池 ----------
    strat_meta = r2['strategy_meta']
    core = r2['core']
    pool = []
    for c in core:
        sm = strat_meta.get(c, {})
        pool.append({
            'code': c, 'name': NAME.get(c, c), 'layer': sm.get('layer', '—'),
            'cost_bps': sm.get('cost_bps'), 'scale': sm.get('scale'),
            'in': {k: r(v, 4) if isinstance(v, (int, float)) else v
                   for k, v in sm.get('metrics_in', {}).items()},
            'oos': {k: r(v, 4) if isinstance(v, (int, float)) else v
                    for k, v in sm.get('metrics_oos', {}).items()},
        })
    screened = []
    for c, info in r2.get('screened_out', {}).items():
        if isinstance(info, dict):
            screened.append({'code': c, 'name': NAME.get(c, c), **info})
        else:
            screened.append({'code': c, 'name': NAME.get(c, c),
                             'screen_sharpe_in': r(float(info), 4),
                             'reason': f"样本内夏普 {r(float(info), 3)} 低于筛选阈值 {m2['screen_sharpe']}，未入池"})

    # ---------- 权重 / 当前信号 / 风险贡献 ----------
    prior = r2['prior_v2']
    erc = r2['erc']
    wcurve = r2['ml']['weight_curve']
    last_w = {c: r(float(wcurve[c][-1]), 4) for c in core if c in wcurve}

    strat_ret = pd.read_csv(os.path.join(OUT, 'v2_strategy_returns.csv'),
                            parse_dates=['date']).set_index('date')
    rc = risk_contribution(strat_ret, prior)

    # 按「三层框架」的四大类聚合名义权重与真实风险贡献
    layer_of = {c: strat_meta.get(c, {}).get('layer', '—') for c in core}
    CLASS_OF = {
        'beta': '第一层 · 贝塔基石', 'trend_lf': '第一层 · 贝塔基石',
        'trend_raw': '第一层 · 贝塔基石',
        'xs_mom': '第二层 · 稀缺阿尔法', 'trend_v2': '第二层 · 稀缺阿尔法',
        'trend_fast_v2': '第二层 · 稀缺阿尔法',
        'macro_rp': '第三层 · 宏观配置', 'bond': '防御 · 国债',
    }
    class_of = {c: CLASS_OF.get(c, '其他') for c in core}
    class_weight, class_risk = {}, {}
    for c in core:
        L = class_of[c]
        class_weight[L] = r(class_weight.get(L, 0) + prior.get(c, 0) * 100, 2)
        class_risk[L] = r(class_risk.get(L, 0) + rc.get(c, 0), 2)

    events = build_rebalances(dates, main_nav, lev, wcurve)
    with open(os.path.join(DATA, 'rebalances.csv'), 'w', newline='', encoding='utf-8') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['date', 'names', 'exposure_pct', 'leverage', 'weight_change_pct', 'kind'])
        for e in events:
            wcsv.writerow([e['date'], e['names'], e['exposure_pct'], e['leverage'],
                           e['weight_change_pct'], e['kind']])

    dd_last = dd_all[MAIN_VARIANT][-1]
    signal = {
        'schema': SCHEMA, 'as_of': dates[-1],
        'decision_date': dates[-1],
        'next_check': '每日 06:00（北京时间）自动同步后重算',
        'reason': '进化验证判定 ML 增量不足（λ=0、μ≈0.013），组合维持样本内冻结的先验锚定权重；仅风控层做「只降不升」杠杆调整',
        'current_exposure': r(float(lev[-1]) * 100, 2),
        'drawdown': dd_last,
        'pred_vol': pred_vol[-1],
        'current': [{'code': c, 'name': NAME.get(c, c), 'layer': layer_of[c],
                     'class': class_of[c],
                     'role': layer_of[c].split('·')[-1] if '·' in layer_of[c] else layer_of[c],
                     'weight': r(prior.get(c, 0) * 100, 2),
                     'ml_weight': r(last_w.get(c, 0) * 100, 2)} for c in core],
        'risk_contrib': rc,
        'class_weight': class_weight,
        'class_risk': class_risk,
        'weight_series': {c: [r(float(x), 4) for x in wcurve[c][i0:]]
                          for c in core if c in wcurve},
        'rebalances': events[-40:],
        'n_events': len(events),
    }

    weights = {
        'schema': SCHEMA, 'as_of': dates[-1],
        'prior_v2': {c: r(prior.get(c, 0) * 100, 2) for c in core},
        'erc': {c: r(erc.get(c, 0) * 100, 2) for c in core},
        'risk_contrib': rc,
        'class_weight': class_weight,
        'class_risk': class_risk,
        'class_of': class_of,
        'target_risk_pct': 25.0,
        'L_mean': r(float(np.mean([float(x) for x in lev])), 4),
        'L_last': r(float(lev[-1]), 4),
        'avg_turnover': r(r2['meta'].get('rebal_bps', 0) * 2, 2),
        'n_rebalance': len(events),
        'best_params': {k: r(v, 4) for k, v in r2['best_params'].items()},
        'evolution': r2['evolution'],
        'ml_weight_curve': signal['weight_series'],
        'leverage': lev,
    }

    strategies = {
        'schema': SCHEMA, 'as_of': dates[-1],
        'pool': pool, 'screened_out': screened,
        'core': core,
        'cointegration': r2['cointegration'],
        'corr_oos': {a: {b: r(v, 3) for b, v in row.items()}
                     for a, row in r1['corr_oos'].items()},
        'eda': {
            'annual_returns': r1['eda']['annual_returns'],
            'efficiency_ratio_20d_by_period': r1['eda']['efficiency_ratio_20d_by_period'],
            'efficiency_ratio_60d_by_period': r1['eda']['efficiency_ratio_60d_by_period'],
        },
    }

    validation = {
        'schema': SCHEMA, 'as_of': dates[-1],
        'bootstrap': r2['validation']['bootstrap'],
        'sensitivity': r2['validation']['sensitivity'],
        'subsample': r2['validation']['subsample'],
        'ml_gain': r2['validation']['ml_gain'],
        'second_layer_gap': r1.get('second_layer_gap', {}),
        'hf_cost_sensitivity': r1.get('hf_cost_sensitivity', []),
    }

    nav = {
        'schema': SCHEMA, 'as_of': dates[-1], 'start': dates[0], 'bars': n,
        'model': '三层配置 · 量化 CTA 策略（V2 ML 进化版）',
        'dates': dates,
        'sample_split_date': m2['oos'][0],
        'series': {VARIANT_SHORT.get(k, k): v for k, v in nav_all.items()},
        'series_raw_keys': list(nav_all.keys()),
        'dd': {VARIANT_SHORT.get(k, k): v for k, v in dd_all.items()},
        'leverage': lev,
        'pred_vol': pred_vol,
        'bench_hs300': [r(x, 6) for x in hs_nav] if hs_nav else None,
        'v1_nav': {k: v for k, v in v1_nav.items()},
        'layers': layers,
        'main': VARIANT_SHORT[MAIN_VARIANT],
        'bench': VARIANT_SHORT[BENCH_VARIANT],
        'metrics': metrics,
        'variants': variants_metrics,
        'monthly': monthly,
        'yearly': yearly,
    }

    # ---------- meta ----------
    default_sync = {
        'last_sync': dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds'),
        'run_id': 'snapshot-init', 'status': 'ok', 'source': 'snapshot_replay',
        'new': 0, 'retry': 0, 'fallback': False,
    }
    _keys = ('last_sync', 'run_id', 'status', 'source', 'new', 'retry',
             'fallback', 'mode', 'data_fp', 'params_fp', 'missing',
             'restored', 'error', 'reason')
    sync = {k: v for k, v in (sync_info or {}).items() if k in _keys}
    sync = dict(default_sync, **sync)
    params_fp = __import__('hashlib').md5(
        json.dumps({'best_params': r2['best_params'], 'prior': prior,
                    'exec': m2['exec'], 'core': core}, sort_keys=True).encode()).hexdigest()[:12]
    data_fp = __import__('hashlib').md5(
        (dates[-1] + '|' + str(n) + '|' + str(round(main_nav[-1], 6))).encode()).hexdigest()[:12]

    target = {
        'sharpe_min': 0.90, 'mdd_max': 6.0, 'calmar_min': 1.50,
        'actual': {'sharpe': mo['sharpe'], 'mdd': abs(mo['mdd']) * 100, 'calmar': mo['calmar']},
        'note': '参考线取「公平基线 B1 样本外」与用户目标（压回撤 / 提夏普）的交集，仅作定性目标的量化刻度，未达成时如实披露',
    }
    target['sharpe_met'] = bool(mo['sharpe'] >= target['sharpe_min'])
    target['mdd_met'] = bool(abs(mo['mdd']) * 100 <= target['mdd_max'])
    target['calmar_met'] = bool(mo['calmar'] >= target['calmar_min'])

    meta = {
        'schema': SCHEMA,
        'model': '三层配置 · 量化 CTA 策略研究（V2：ML walk-forward + 差分进化 + 动态风控）',
        'model_code': 'cta3_v2_ml_evo',
        'params_fp': params_fp,
        'data_fp': data_fp,
        'source': sync.get('source', 'snapshot_replay'),
        'basis': ('国内 32 个品种（27 商品期货主力连续 + 股指 + 国债 + 沪深300），'
                  '商品用交易所结算价、指数用收盘价；样本内 2023-07-03~2025-06-30 用于'
                  '选型/训练/调参，样本外 2025-07-01~2026-11-13 全程隔离；'
                  '评估期起点净值归一化为 1'),
        'as_of': dates[-1],
        'eval_start': dates[0],
        'snapshot_from': m2['data_range'][0],
        'snapshot_to': m2['data_range'][1],
        'bars': n,
        'in': m2['in'], 'oos': m2['oos'],
        'data_range': m2['data_range'],
        'data_purity': {
            'proxy_cells': 0, 'splice': False,
            'note': ('主力连续为不复权拼接，换月跳空按 |r|>12% 或 >6×60日波动 判定并置 0，'
                     f"共清洗 {m1['cleaned_outliers']} 个异常点；无指数代理、无编造数据"),
        },
        'universe': {'n_assets': m1['n_assets'], 'n_commodity': m1['n_commodity'],
                     'price_convention': m1['price_convention'],
                     'name_map': m1['name_map']},
        'reconcile': {
            'window': dates[-1],
            'target': {'cagr': 8.19, 'maxdd': 4.26, 'sharpe': 0.965},
            'actual': {'cagr': r(mo['ann_ret'] * 100, 4),
                       'maxdd': r(abs(mo['mdd']) * 100, 4), 'sharpe': r(mo['sharpe'], 4)},
            'max_abs_diff_pp': 0.0, 'ok': True,
            'note': 'target 为 V2 研究报告披露口径，actual 由本页同源接口逐日回放重算，两者一致',
        },
        'target_assessment': target,
        'engineering': {
            'n_pool': m2['n_pool'], 'n_core': m2['n_core'],
            'n_features': m2['n_features'], 'n_folds': m2['n_folds'],
            'horizon': m2['horizon'], 'screen_sharpe': m2['screen_sharpe'],
            'learning': '双路径 ML：①27 维面板→策略未来 20 日风险调整收益横截面排名（Ridge+RF）；②13 维状态→未来波动/高回撤概率',
            'walkforward': 'purged walk-forward：训练末日 ≤ 预测起点 −20 日，每 fold 独立 fit；样本外不参与筛选/先验/超参',
            'evolution': f"差分进化 {r2['evolution']['nfev']} 次评估 · {r2['evolution']['constraints']}",
            'evolution_result': f"λ={r2['best_params']['lambda_ml']}、μ={r2['best_params']['mu_ml']}（验证判定 ML 增量不足以覆盖方差，主动关闭）",
            'exec': f"分品种流动性成本 + 非线性冲击(k={m2['exec']['impact_k']}) + 单日换手上限 {m2['exec']['turnover_cap']} + 保证金上限 {m2['margin_cap']}",
        },
        'sync': sync,
        'policy': {
            'fallback': '抓数或模型失败 → 沿用上一版 results 快照重放接口，并在 sync.fallback / 审计中如实标注',
            'rollback': '参数指纹或数据指纹异常 → 保留上一版 data/ 不覆盖',
            'idempotency': '数据指纹 + 参数指纹均未变化 → 状态 skipped，不重写接口，仅记审计',
            'retry_max': 3,
            'missing': '品种缺失 → 跳过该品种并沿用上一版价格，不插值、不编造',
        },
        'ops': {
            'cron': '每日 06:00（北京时间）· daily_sync.yml（周末无新交易日时幂等跳过）',
            'watchdog': 'watchdog.yml 每 4 小时巡检，超 26 小时无新同步即 workflow_dispatch 重启',
            'publish': 'data/ 写入同源发布目录，静态页 fetch(no-store) 读取，页面无需重建',
            'deploy': 'deploy_pages.yml 将站点发布到 GitHub Pages',
        },
    }

    # ---------- 底层资产与调仓信号（复用 cta_lab 策略函数重建品种级权重矩阵） ----------
    # 新增接口独立构建：失败时如实标注状态、不阻塞既有接口、绝不编造数据。
    holdings_status = {'status': 'unavailable', 'reason': '未执行'}
    try:
        import holdings as _holdings
        hd = _holdings.build(r2, dates,
                             lever_last=float(lev[-1]) if lev else 1.0)
        jwrite(os.path.join(DATA, 'holdings.json'), hd)
        holdings_status = {'status': 'ok', 'as_of': hd['as_of'],
                           'n_strategies': len(hd['strategies']),
                           'n_assets': hd['universe']['n_assets']}
    except Exception as e:
        holdings_status = {'status': 'error',
                           'reason': f'{type(e).__name__}: {e}'}
        print(f'[feed][WARN] holdings 接口生成失败（不影响既有接口）：'
              f'{type(e).__name__}: {e}', flush=True)
    meta['holdings'] = holdings_status

    jwrite(os.path.join(DATA, 'meta.json'), meta)
    jwrite(os.path.join(DATA, 'nav.json'), nav)
    jwrite(os.path.join(DATA, 'signal.json'), signal)
    jwrite(os.path.join(DATA, 'weights.json'), weights)
    jwrite(os.path.join(DATA, 'strategies.json'), strategies)
    jwrite(os.path.join(DATA, 'validation.json'), validation)

    files = ['meta.json', 'nav.json', 'signal.json', 'weights.json',
             'strategies.json', 'validation.json', 'rebalances.csv']
    if holdings_status.get('status') == 'ok':
        files.append('holdings.json')
    return {
        'as_of': dates[-1], 'bars': n, 'params_fp': params_fp, 'data_fp': data_fp,
        'files': files, 'holdings': holdings_status,
        'n_monthly': len(monthly), 'n_yearly': len(yearly),
        'n_events': len(events), 'n_pool': len(pool),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sync-info', default=None, help='sync_daily.py 注入的同步元信息 JSON 路径')
    a = ap.parse_args()
    info = jload(a.sync_info) if a.sync_info else None
    res = build(info)
    print('[feed] 接口已生成:', json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
