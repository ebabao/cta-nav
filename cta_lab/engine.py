# -*- coding: utf-8 -*-
"""向量化回测引擎与绩效评估。"""
import numpy as np
import pandas as pd
from config import TRADING_DAYS, RF, TARGET_VOL, MAX_GROSS, VOL_WINDOW


def backtest(weights, ret, cost_bps=0.0, target_vol=TARGET_VOL,
             max_gross=MAX_GROSS, vol_window=VOL_WINDOW):
    """
    weights: DataFrame(index=date, columns=asset)，t 日收盘形成的目标权重（t+1 生效）
    ret    : 同结构的日收益率矩阵
    返回   : 日收益序列 pnl（已扣交易成本、已做波动率目标化）
    """
    cols = ret.columns
    w = weights.reindex(index=ret.index, columns=cols).fillna(0.0)
    if max_gross is not None:
        gross = w.abs().sum(axis=1).replace(0, np.nan)
        w = w.mul((max_gross / gross).clip(upper=1.0).fillna(0.0), axis=0)

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()

    gross_pnl = (w.shift(1) * ret).sum(axis=1)
    cost = turnover * cost_bps / 1e4
    pnl = gross_pnl - cost

    if target_vol:
        realized = pnl.rolling(vol_window, min_periods=20).std() * np.sqrt(TRADING_DAYS)
        scale = (target_vol / realized.replace(0, np.nan)).clip(0.0, 2.0).shift(1)
        pnl = pnl * scale.fillna(1.0)
    return pnl.fillna(0.0)


def nav_from(pnl):
    return (1.0 + pnl).cumprod()


def metrics(pnl, rf=RF):
    """年化收益/波动/夏普/Sortino/最大回撤/Calmar/最长回撤修复/月度胜率。"""
    p = pnl.dropna()
    if len(p) < 20:
        return {}
    nav = nav_from(p)
    n = len(p)
    ann_ret = nav.iloc[-1] ** (TRADING_DAYS / n) - 1
    ann_vol = p.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    downside = p[p < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (ann_ret - rf) / downside if downside and downside > 0 else np.nan

    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan

    # 回撤修复时长（文章特别强调「回撤修复时间短」）
    under = (dd < -1e-9)
    grp = (under != under.shift()).cumsum()
    runs = under.groupby(grp).sum()
    max_recovery = int(runs[runs > 0].max()) if (runs > 0).any() else 0
    current_under = 0
    if under.iloc[-1]:
        g = grp.iloc[-1]
        current_under = int((grp == g).sum())

    monthly = (1 + p).resample('ME').prod() - 1
    return {
        'ann_ret': round(float(ann_ret), 4),
        'ann_vol': round(float(ann_vol), 4),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3) if not np.isnan(sortino) else None,
        'mdd': round(float(mdd), 4),
        'calmar': round(float(calmar), 3) if not np.isnan(calmar) else None,
        'max_recovery_days': max_recovery,
        'current_underwater_days': current_under,
        'monthly_win': round(float((monthly > 0).mean()), 4),
        'total_ret': round(float(nav.iloc[-1] - 1), 4),
        'days': n,
    }


def slice_period(pnl, start, end):
    return pnl.loc[start:end]


def erc_weights(cov, max_iter=2000, tol=1e-10):
    """等风险贡献（风险平价）权重，简单乘性迭代求解。"""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        rc = w * (cov @ w)
        target = rc.mean()
        if target <= 0:
            break
        w_new = w * (target / np.maximum(rc, 1e-12)) ** 0.5
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w / w.sum()


# ============================================================ V2 执行层 / 风控层
def backtest_exec(weights, ret, cost_rate=0.0, adv=None, impact_k=0.0,
                  turnover_cap=None, max_gross=None, amplitude=None):
    """
    更贴近实盘的回测：
      - cost_rate: 标量或每品种 Series（bp），按流动性分级
      - adv      : 每品种相对流动性（用于非线性冲击成本）
      - impact_k : 冲击成本系数，冲击 ∝ (换手/流动性)^1.5
      - turnover_cap: 单日单品种最大权重变动（成交约束，逐日递推，产生渐进建仓）
      - max_gross: 保证金约束下的总绝对敞口上限
      - amplitude: 当日平均振幅，用于放大滑点（可选）
    """
    cols = ret.columns
    w0 = weights.reindex(index=ret.index, columns=cols).fillna(0.0)
    if max_gross is not None:
        g = w0.abs().sum(axis=1).replace(0, np.nan)
        w0 = w0.mul((max_gross / g).clip(upper=1.0).fillna(0.0), axis=0)

    if turnover_cap is not None:
        arr = w0.values.astype(float)
        out = np.zeros_like(arr)
        prev = np.zeros(arr.shape[1])
        for t in range(len(arr)):
            d = np.clip(arr[t] - prev, -turnover_cap, turnover_cap)
            prev = prev + d
            out[t] = prev
        w = pd.DataFrame(out, index=w0.index, columns=w0.columns)
    else:
        w = w0

    to = w.diff().abs()
    to.iloc[0] = w.iloc[0].abs()

    if np.isscalar(cost_rate):
        cr = pd.Series(float(cost_rate), index=cols)
    else:
        cr = pd.Series(cost_rate).reindex(cols).fillna(0.0)
    linear = (to * cr).sum(axis=1) / 1e4

    if amplitude is not None:
        amp = pd.Series(amplitude).reindex(ret.index).fillna(0.0)
        mult = (1.0 + 0.4 * (amp / (amp.rolling(60, min_periods=20).mean() + 1e-9) - 1)).clip(0.7, 2.0)
        linear = linear * mult

    impact = 0.0
    if adv is not None and impact_k > 0:
        a = pd.Series(adv).reindex(cols).fillna(1.0).clip(0.2, 5.0)
        impact = impact_k * ((to.div(a, axis=1)) ** 1.5).sum(axis=1)

    gross_pnl = (w.shift(1) * ret).sum(axis=1)
    return (gross_pnl - linear - impact).fillna(0.0)


def apply_risk_overlay(pnl, target_vol=0.10, vol_window=40,
                       dd_threshold=0.05, dd_floor=0.35, vol_cap=(0.35, 1.8)):
    """
    组合层动态风险预算：
      1) 波动率目标：实现波动上升时降杠杆、下降时加杠杆；
      2) 回撤刹车：水下深度超过阈值时线性降杠杆，修复后自动渐进恢复。
    两者都用 shift(1)，信号在 t 日收盘形成、t+1 生效，无前视。
    """
    rv = pnl.rolling(vol_window, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    lev_v = (target_vol / rv.replace(0, np.nan)).clip(*vol_cap).shift(1).fillna(1.0)

    nav = (1.0 + pnl).cumprod()
    dd = nav / nav.cummax() - 1
    depth = (-dd).clip(lower=0)
    lev_d = (1.0 - (depth - dd_threshold).clip(lower=0) / (dd_threshold + 1e-9) * 0.8)
    lev_d = lev_d.clip(dd_floor, 1.0).shift(1).fillna(1.0)

    lev = (lev_v * lev_d).clip(0.2, 2.0)
    return pnl * lev, lev, pd.DataFrame({'lev_vol': lev_v, 'lev_dd': lev_d})
