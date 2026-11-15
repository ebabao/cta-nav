# -*- coding: utf-8 -*-
"""
V2 执行层：分品种流动性分级、非线性冲击成本、滑点振幅、保证金约束输入。

所有「结构性」参数（流动性等级）只用样本内区间估计并冻结到样本外，
避免任何形式的前视；当日振幅是收盘时可观测的行情数据。
"""
import os
import numpy as np
import pandas as pd
from config import DATA, IN_END, COMMODITY


def build_market_profile(bundle):
    """返回 (liq, adv, amplitude)。liq/adv 为每品种相对流动性；amplitude 为日序列。"""
    px = bundle['px']
    liq, amps = {}, []
    for sym in px.columns:
        f = os.path.join(DATA, 'raw', f'{sym}.csv')
        if not os.path.exists(f):
            liq[sym] = np.nan
            continue
        d = pd.read_csv(f, parse_dates=['date']).set_index('date').reindex(px.index)
        turnover = (d['close'] * d['volume']).loc[:IN_END]
        v = turnover[turnover > 0].median()
        liq[sym] = float(v) if np.isfinite(v) and v > 0 else np.nan
        amps.append(((d['high'] - d['low']) / d['close'].shift(1)).replace([np.inf, -np.inf], np.nan))

    s = pd.Series(liq, dtype=float)
    # 指数成交额口径与期货不同，用同标的股指期货的流动性代替
    if np.isnan(s.get('HS300', np.nan)) and np.isfinite(s.get('IF0', np.nan)):
        s['HS300'] = s['IF0']
    med = s[COMMODITY].median()
    liq = (s / med).clip(0.3, 3.0).fillna(1.0)
    amplitude = pd.concat(amps, axis=1).mean(axis=1).fillna(0.0) if amps else pd.Series(0.0, index=px.index)
    return liq, liq.copy(), amplitude


def strategy_cost_rate(base_bps, liq, cols, illiq_sensitivity=0.3):
    """把策略级成本拆成分品种成本：流动性越差，单位换手成本越高。"""
    adj = (1.0 / liq.reindex(cols).fillna(1.0) - 1.0).clip(-0.5, 2.0)
    return (base_bps * (1.0 + illiq_sensitivity * adj)).clip(lower=0.5)


def margin_cap(liq, cols, base=2.5, sensitivity=0.25):
    """保证金约束的简化代理：低流动性品种占用保证金更高，总敞口上限相应收缩。"""
    pen = (1.0 / liq.reindex(cols).fillna(1.0) - 1.0).clip(0, 2.0).mean()
    return float(base / (1.0 + sensitivity * pen))
