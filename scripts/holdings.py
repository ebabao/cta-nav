# -*- coding: utf-8 -*-
"""
底层资产与调仓信号接口构建器（holdings.json）。

设计原则（延续 feed.py / 用户 SOP）：
  * 只复用 cta_lab 现有策略函数重建「日期 × 品种」权重矩阵 W，
    不重写策略、不改回测逻辑、不改任何模型结果；
  * W 与 run_v2.py POOL 用同一组函数、同一组参数，保证与回测完全同源；
  * 底层资产、方向、权重、换手、调仓事件、ER 流畅度、动量读数
    全部由 W / 行情矩阵真实计算，不编造任何数字；
  * 信号定义（公式、参数、触发规则）是对 strategies.py 源码的如实转述；
  * 品种缺失 → 跳过该品种，不插值、不臆造。

由 feed.build() 调用，输出 data/holdings.json；云端 daily_sync 的
live 模式回放模型后会自动重建，因此本接口与看板数据每日同步。
"""
import os
import sys
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(BASE, 'cta_lab')
if LAB not in sys.path:
    sys.path.insert(0, LAB)

from config import NAME_MAP, COMMODITY, FINANCIAL          # noqa: E402
from data_prep import build as build_bundle                # noqa: E402
from strategies import (build_all, trend_lf_v2,            # noqa: E402
                        efficiency_ratio, _trend_signal)

SCHEMA = 'cta-nav/v1'

# ---------------------------------------------------------------- 品种元数据
# 板块为公开交易所行业分类（元数据标注，非行情数据）；口径与 data_prep 一致：
# 商品用交易所结算价、金融（指数/国债期货）用收盘价。
SECTOR_OF = {
    'AU0': '贵金属', 'AG0': '贵金属',
    'CU0': '有色', 'AL0': '有色', 'NI0': '有色',
    'RB0': '黑色', 'HC0': '黑色', 'I0': '黑色', 'J0': '黑色', 'JM0': '黑色',
    'SC0': '能化', 'TA0': '能化', 'MA0': '能化', 'L0': '能化', 'PP0': '能化',
    'EG0': '能化', 'SA0': '能化', 'RU0': '能化', 'FU0': '能化',
    'M0': '农产品', 'Y0': '农产品', 'P0': '农产品', 'C0': '农产品',
    'SR0': '农产品', 'CF0': '农产品', 'OI0': '农产品', 'RM0': '农产品',
    'HS300': '金融', 'IF0': '金融', 'IC0': '金融', 'T0': '金融', 'TF0': '金融',
}
SECTORS = ['贵金属', '有色', '黑色', '能化', '农产品', '金融']

# ---------------------------------------------------------------- 信号定义
# 以下文字/参数逐条对应 cta_lab/strategies.py 的实现，供页面如实展示「靠什么信号调仓」。
SIGNAL_META = {
    'beta': dict(
        family='买入持有 · 股票贝塔',
        formula='w(HS300) = 1，恒定满仓，无择时',
        params={},
        rules=['不产生调仓信号：低成本获取市场平均收益，仅在组合层再平衡时被动调整'],
        holding='长期持有', freq='无（恒定）'),
    'bond': dict(
        family='买入持有 · 债券贝塔',
        formula='w(T0) = 1，恒定满仓，无择时',
        params={},
        rules=['不产生调仓信号：十年国债期货买入持有，仅用于 60/40 对照与防御'],
        holding='长期持有', freq='无（恒定）'),
    'trend_lf': dict(
        family='时序动量 · 中低频趋势（ER 流畅度过滤）',
        formula='sig = mean_w[ tanh( (P/P_{−w}−1) / (σ_w·√w) ) ]，w ∈ {20,40,60,120}；'
                'w_i ∝ sig × ER过滤 × 逆波动率',
        params={'动量窗口': '20 / 40 / 60 / 120 日', '平滑': '5 日',
                '离散档位': '0.25（折线变色才调仓）',
                'ER 过滤': '((ER20−0.10)/0.20).clip(0.25, 1)',
                '全市场降敞口': '((mean ER20−0.15)/0.10).clip(0.5, 1)，滞后 1 日'},
        rules=['多周期趋势同向 → 加仓；折线变色（信号跨越 0.25 档）→ 调仓',
               '行情流畅度 ER 低 → 主动压缩该品种敞口（文章核心：要「高波动且方向流畅」）',
               '全市场流畅度过低 → 整体降趋势敞口',
               '逆波动率配置：波动大的品种给小权重，统一风险贡献'],
        holding='数周 ~ 数月', freq='信号驱动（非固定周期）'),
    'trend_raw': dict(
        family='时序动量 · 快速趋势（对照组）',
        formula='同 trend_lf，但窗口含 10 日、不做 5 日平滑、不做档位离散',
        params={'动量窗口': '10 / 20 / 40 / 60 / 120 日', '平滑': '无',
                '离散档位': '无（逐日连续）'},
        rules=['信号过度灵敏、折线频繁变色 → 换手显著高于 trend_lf',
               '用于验证「频繁变色 = 磨损期」，不作主力配置'],
        holding='数日 ~ 数周', freq='逐日微调'),
    'xs_mom': dict(
        family='截面动量 · 横截面多空',
        formula='rank = pct_rank(P/P_{−60}−1)；s = (rank−0.5)×2，去均值；w ∝ s × 逆波动率',
        params={'回看窗口': '60 日', '中性化': '横截面去均值（净敞口≈0）'},
        rules=['横向对比 27 个商品相对强弱：做多强势、做空弱势',
               '不预判商品单边方向，市场中性'],
        holding='数周', freq='日频再平衡'),
    'macro_rp': dict(
        family='宏观配置 · 风险平价 + 动量择时',
        formula='股(HS300)/债(T0)/商(等权商品指数) 按 60 日逆波动率定基准配比；'
                '120 日动量 > 0 → 满配，否则 ×0.30；总敞口 0.95',
        params={'波动窗口': '60 日', '择时窗口': '120 日',
                '防御系数': '0.30', '总敞口上限': '0.95'},
        rules=['风险平价定「配比」，动量择时做「防御性降仓」：跌时控得住',
               '商品腿按逆波动率分配到 27 个品种，构成实际底层持仓'],
        holding='月度级别', freq='月度 / 动量翻转时'),
    'trend_v2': dict(
        family='时序动量 · 变色止损趋势（V2）',
        formula='在 trend_lf 基础上叠加：ER 过滤阈值 (0.12, 0.30)；'
                '若 sign(sig)×近5日收益 < −1.2×σ5 → 该品种清仓',
        params={'动量窗口': '20 / 40 / 60 / 120 日', 'ER 区间': '0.12 ~ 0.30',
                '止损系数': '1.2 × 5日波动', '平滑': '5 日', '离散档位': '0.25'},
        rules=['趋势反转且反向波动过大 → 先平仓，不硬扛',
               '回撤较原趋势策略明显收窄（工程验证结论）'],
        holding='数周 ~ 数月', freq='信号驱动 + 止损触发'),
    'trend_fast_v2': dict(
        family='时序动量 · 快窗口止损趋势（V2）',
        formula='窗口含 10 日、不平滑不离散；ER (0.10, 0.30)；止损 1.5×σ5',
        params={'动量窗口': '10 / 20 / 40 / 60 / 120 日', 'ER 区间': '0.10 ~ 0.30',
                '止损系数': '1.5 × 5日波动', '平滑': '无', '离散档位': '无'},
        rules=['更快的趋势响应 + 变色止损，样本内夏普最高（0.923）',
               '换手偏高，靠执行层换手上限约束'],
        holding='数日 ~ 数周', freq='逐日微调 + 止损触发'),
}


def _r(x, n=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), n)


def _weight_matrices(bundle, px, ret):
    """复用现有策略函数，按 run_v2.py POOL 的同一参数重建 8 个入池策略的 W。"""
    specs, _ = build_all(bundle)   # 传入完整 bundle（feats/cidx 真实），不臆造
    W = {
        'beta': specs['beta'][0],
        'trend_lf': specs['trend_lf'][0],
        'trend_raw': specs['trend_raw'][0],
        'xs_mom': specs['xs_mom'][0],
        'macro_rp': specs['macro_rp'][0],
        'trend_v2': trend_lf_v2(px, ret, er_low=0.12, er_high=0.30, stop_k=1.2),
        'trend_fast_v2': trend_lf_v2(px, ret, windows=(10, 20, 40, 60, 120),
                                     smooth=0, quant=0, stop_k=1.5),
        'bond': specs['bond'][0],
    }
    return W


def _universe(px):
    """品种元数据：代码、中文名、板块、价格口径。缺失品种跳过。"""
    assets = []
    by_sector = {s: [] for s in SECTORS}
    for c in px.columns:
        if c not in SECTOR_OF or c not in NAME_MAP:
            continue
        item = {'code': c, 'name': NAME_MAP[c], 'sector': SECTOR_OF[c],
                'kind': '金融' if c in FINANCIAL else '商品',
                'price': '收盘价' if c in FINANCIAL else '交易所结算价'}
        assets.append(item)
        by_sector[item['sector']].append(c)
    return {'n_assets': len(assets), 'assets': assets, 'by_sector': by_sector,
            'sectors': SECTORS,
            'note': '商品用交易所结算价（趋势延续性真实）、指数/国债用收盘价；与回测口径一致'}


def _market_state(px, ret, dates_eval):
    """真实市场状态读数：全市场 ER20 流畅度、60 日动量宽度（上涨品种占比）。"""
    comm = [c for c in COMMODITY if c in px.columns]
    er = efficiency_ratio(px[comm], 20).mean(axis=1)
    mom60 = px[comm] / px[comm].shift(60) - 1.0
    breadth = (mom60 > 0).sum(axis=1) / max(len(comm), 1)
    # 评估期内按周采样，控制接口体积
    idx = px.index
    keep = [d for d in idx if d.strftime('%Y-%m-%d') in set(dates_eval)]
    # 用每 5 个交易日采样一次
    sample = list(keep[::5])
    if keep and sample[-1] != keep[-1]:
        sample.append(keep[-1])
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in sample],
        'er20': [_r(er.get(d, np.nan), 4) for d in sample],
        'breadth': [_r(breadth.get(d, np.nan), 4) for d in sample],
        'er20_last': _r(er.iloc[-1], 4),
        'breadth_last': _r(breadth.iloc[-1], 4),
        'note': 'ER20 = 20 日净位移/路径长度（越高越流畅）；宽度 = 60 日动量为正的品种占比',
    }


def _strategy_block(code, W, px, ret, prior, scale, lever_last, smeta,
                    dates_eval, i0):
    """单个策略：底层资产、当前持仓、信号读数、换手统计、调仓事件、信号时序。"""
    W = W.reindex(index=px.index, columns=px.columns).fillna(0.0)
    w_last = W.iloc[-1]
    held = w_last[w_last.abs() > 1e-6]

    # 组合层有效敞口：策略内权重 × 标定杠杆 × 组合先验权重 × 当前风控杠杆
    eff_scale = float(scale or 1.0) * float(prior or 0.0) * float(lever_last or 1.0)

    comm = [c for c in COMMODITY if c in px.columns]
    mom60 = (px / px.shift(60) - 1.0)
    er20 = efficiency_ratio(px[comm], 20)

    positions = []
    for c, w in held.sort_values(key=lambda s: s.abs(), ascending=False).items():
        if c not in SECTOR_OF or c not in NAME_MAP:
            continue
        positions.append({
            'code': c, 'name': NAME_MAP.get(c, c), 'sector': SECTOR_OF.get(c, '—'),
            'w': _r(w, 4),                       # 策略内归一化权重
            'w_eff_pct': _r(w * eff_scale * 100, 3),  # 组合层有效敞口(%)
            'side': '多' if w > 0 else '空',
            'mom60': _r(mom60[c].iloc[-1], 4),
            'er20': _r(er20[c].iloc[-1], 4) if c in er20.columns else None,
        })

    # 板块暴露（按策略内权重聚合）
    sec = {}
    for p in positions:
        s = p['sector']
        d = sec.setdefault(s, {'long': 0.0, 'short': 0.0})
        d['long' if p['w'] > 0 else 'short'] += p['w']
    sector_exposure = [{'sector': s,
                        'long': _r(sec[s]['long'], 4),
                        'short': _r(sec[s]['short'], 4),
                        'net': _r(sec[s]['long'] + sec[s]['short'], 4)}
                       for s in SECTORS if s in sec]

    # 换手与调仓事件（真实从 |Δw| 计算）
    to = W.diff().abs().sum(axis=1)
    to.iloc[0] = W.iloc[0].abs().sum()
    sign_flip = ((np.sign(W) != np.sign(W.shift(1))) & (W.abs() > 1e-6)).sum(axis=1)
    # 排除首日建仓（所有品种从 0 到 w 属初始建仓，非「调仓」）
    ev_dates = to.index[(to > 0.15) & (to.index > to.index[0])]
    events = []
    for d in ev_dates[-24:]:
        dw = (W.loc[d] - W.shift(1).loc[d])
        top = dw[dw.abs() > 1e-6].sort_values(key=lambda s: s.abs(), ascending=False).head(3)
        events.append({
            'date': d.strftime('%Y-%m-%d'),
            'turnover': _r(to.loc[d], 4),
            'flips': int(sign_flip.loc[d]),
            'top': [{'code': c, 'name': NAME_MAP.get(c, c), 'dw': _r(v, 4)}
                    for c, v in top.items() if c in NAME_MAP],
        })
    last_rebal = ev_dates[-1].strftime('%Y-%m-%d') if len(ev_dates) else None

    # 信号时序（评估期，每 2 日采样以控制体积）
    idx = list(W.index[i0:])
    samp = idx[::2]
    if samp and samp[-1] != idx[-1]:
        samp.append(idx[-1])
    net = W.sum(axis=1)
    n_held = (W.abs() > 1e-6).sum(axis=1).replace(0, np.nan)
    long_frac = (W > 1e-6).sum(axis=1) / n_held
    gross = W.abs().sum(axis=1)
    history = {
        'dates': [d.strftime('%Y-%m-%d') for d in samp],
        'net': [_r(net.get(d, np.nan), 4) for d in samp],
        'long_frac': [_r(long_frac.get(d, np.nan), 4) for d in samp],
        'gross': [_r(gross.get(d, np.nan), 4) for d in samp],
        'turnover': [_r(to.get(d, np.nan), 4) for d in samp],
    }

    # ---------- 当前信号状态（真实读数，不臆造） ----------
    comm_cols = [c for c in COMMODITY if c in px.columns]
    er_mean_now = float(efficiency_ratio(px[comm_cols], 20).mean(axis=1).iloc[-1])
    # 与 strategies.trend_lf / trend_lf_v2 的 global_filt 同口径
    global_filt = float(np.clip((er_mean_now - 0.15) / 0.10, 0.5, 1.0))
    breadth_now = float((px[comm_cols] / px[comm_cols].shift(60) - 1 > 0).sum(axis=1).iloc[-1]) / max(len(comm_cols), 1)
    is_trend = 'trend' in code
    state = {
        'er20_market': _r(er_mean_now, 4),
        'breadth60': _r(breadth_now, 4),
        'regime': ('趋势流畅' if er_mean_now >= 0.25 else
                   '方向偏弱 / 磨损期' if er_mean_now < 0.18 else '中性过渡'),
        'global_exposure_factor': _r(global_filt, 4) if is_trend else None,
        'stance': ('净空' if w_last.sum() < -0.15 else
                   '净多' if w_last.sum() > 0.15 else '中性 / 对冲'),
        'breadth_positions': f"{int((w_last > 1e-6).sum())} 多 / {int((w_last < -1e-6).sum())} 空 / {len(held)} 持仓",
        'notes': [],
    }
    if is_trend:
        state['notes'].append(
            f"全市场 ER20={_r(er_mean_now, 3)} → 降敞口系数 {_r(global_filt, 3)}"
            + ('（已触发主动压缩趋势敞口）' if global_filt < 0.99 else '（满敞口）'))
        if 'v2' in code:
            state['notes'].append('变色止损激活：单品种近 5 日反向波动越阈值即清仓')
    if code == 'macro_rp':
        eq_mom = float((px['HS300'] / px['HS300'].shift(120) - 1).iloc[-1])
        bd_mom = float((px['T0'] / px['T0'].shift(120) - 1).iloc[-1])
        state['notes'].append(
            f"120 日动量择时：股 {_r(eq_mom, 3)} → {'满配' if eq_mom > 0 else '×0.30 防御'}；"
            f"债 {_r(bd_mom, 3)} → {'满配' if bd_mom > 0 else '×0.30 防御'}")
        state['eq_mom120'] = _r(eq_mom, 4)
        state['bond_mom120'] = _r(bd_mom, 4)
    if code == 'xs_mom':
        state['notes'].append('横截面去均值，净敞口≈0：不预判单边方向，只做相对强弱')
    if code in ('beta', 'bond'):
        state['notes'].append('无调仓信号：恒定满仓，敞口变化仅来自组合层再平衡/风控杠杆')

    meta = SIGNAL_META.get(code, {})
    return {
        'code': code,
        'signal': meta,
        'state': state,
        'n_underlying': int((W.abs() > 1e-6).any(axis=0).sum()),
        'underlying_codes': [c for c in W.columns if (W[c].abs() > 1e-6).any() and c in NAME_MAP],
        'current': {
            'date': px.index[-1].strftime('%Y-%m-%d'),
            'gross': _r(w_last.abs().sum(), 4),
            'net': _r(w_last.sum(), 4),
            'n_long': int((w_last > 1e-6).sum()),
            'n_short': int((w_last < -1e-6).sum()),
            'positions': positions,
            'sector_exposure': sector_exposure,
        },
        'turnover': {
            'daily_avg': _r(to.mean(), 4),
            'annual_est': _r(to.mean() * 252, 2),
            'zero_frac': _r((to < 1e-6).mean(), 4),
            'n_events': int(len(ev_dates)),
            'last_rebalance': last_rebal,
            'build_only': len(ev_dates) == 0,  # 恒定权重策略：仅首日建仓，无调仓信号
        },
        'events': events,
        'history': history,
        'scale': _r(scale, 3),
        'prior_pct': _r(float(prior or 0) * 100, 2),
        'cost_bps': _r(smeta.get('cost_bps'), 1),
        'layer': smeta.get('layer', '—'),
    }


def build(r2, dates_eval, lever_last=1.0):
    """生成 holdings.json 的完整内容。r2 = results_v2.json，dates_eval = 评估期日期串列表。"""
    bundle = build_bundle()
    px, ret = bundle['px'], bundle['ret']
    W = _weight_matrices(bundle, px, ret)

    core = r2['core']
    strat_meta = r2['strategy_meta']
    prior = r2['prior_v2']
    i0 = int(px.index.get_indexer([pd.Timestamp(dates_eval[0])], method='bfill')[0])

    strategies = {}
    for code in core:
        if code not in W:
            continue  # 缺失跳过，不编造
        strategies[code] = _strategy_block(
            code, W[code], px, ret, prior.get(code, 0),
            strat_meta.get(code, {}).get('scale'), lever_last,
            strat_meta.get(code, {}), dates_eval, i0)

    # 资产 × 策略 覆盖矩阵（一眼看清每个策略交易哪些品种）
    universe = _universe(px)
    coverage = []
    for a in universe['assets']:
        c = a['code']
        row = {'code': c, 'name': a['name'], 'sector': a['sector'], 'used_by': []}
        for code in core:
            w = W.get(code)
            if w is not None and c in w.columns and (w[c].abs() > 1e-6).any():
                row['used_by'].append(code)
        coverage.append(row)

    market = _market_state(px, ret, dates_eval)

    return {
        'schema': SCHEMA,
        'as_of': px.index[-1].strftime('%Y-%m-%d'),
        'universe': universe,
        'core': core,
        'coverage': coverage,
        'market': market,
        'strategies': strategies,
        'notes': [
            '权重矩阵由 cta_lab 现有策略函数按 run_v2.py 的同一参数重建，与回测完全同源；未重写策略、未改动模型结果。',
            'w 为策略内归一化权重（总敞口通常 ≤1）；w_eff_pct 为组合层有效敞口 = w × 标定杠杆 × 先验权重 × 当前风控杠杆。',
            '调仓事件阈值 |Δw| 之和 > 0.15，直接由权重矩阵逐日差分计算；flips 为方向翻转品种数。',
            '信号定义（公式/参数/触发规则）逐条对应 strategies.py 源码，属如实转述；信号读数 ER20 / mom60 由行情真实计算。',
            '品种缺失时跳过，不插值、不臆造；所有数字是研究回测口径，不构成投资建议。',
        ],
    }
