# -*- coding: utf-8 -*-
"""
策略库：完全按两篇文章的论述工程化。

层级映射
  第一层 贝塔基石        : equity_beta（宽基指数）、trend_lf（中低频趋势贝塔）
  第二层 稀缺阿尔法      : hf_mr（中高频CTA代理）、xs_mom（截面多空）、
                          spread_arb（跨品种套利）、trend_mix（趋势+反转防守）
  第三层 宏观配置        : macro_rp（股/债/商风险平价 + 动量择时）

所有策略返回「t 日收盘形成、t+1 生效」的目标权重矩阵（gross 归一到 1），
再由 engine 施加样本内估计的固定杠杆倍数，避免滚动波动率目标把主动降仓抵消掉。
"""
import numpy as np
import pandas as pd
from config import COMMODITY, TRADING_DAYS, PAIRS


# ---------------------------------------------------------------- 工具函数
def efficiency_ratio(px, n=20):
    """Kaufman 效率比：|P_t-P_{t-n}| / Σ|ΔP|，度量「行情流畅度」。"""
    d = px.diff().abs()
    return (px - px.shift(n)).abs() / (d.rolling(n, min_periods=n // 2).sum() + 1e-12)


def _inv_vol(ret, cols, window=60):
    v = ret[cols].rolling(window, min_periods=20).std() * np.sqrt(TRADING_DAYS) + 1e-6
    iv = 1.0 / v
    return iv.div(iv.sum(axis=1), axis=0).fillna(0.0)


def _normalize(w):
    g = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(g, axis=0).fillna(0.0)


def _cross_z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-3, 3)


def _neutralize(w):
    """横截面净敞口归零（多空对冲），再按绝对敞口归一化。"""
    w = w.sub(w.mean(axis=1), axis=0)
    return _normalize(w)


def _trend_signal(px, ret, cols, windows=(20, 40, 60, 120), smooth=5, quant=0.25):
    """
    多周期叠加的时序动量：对应文章「五日折线 + 多周期叠加」的量化实现。

    工程要点（对应文章「折线频繁变色 = 磨损期」）：
      - 中低频 CTA 跟踪数周~数月趋势，故窗口取 20/40/60/120 日，剔除 10 日噪声；
      - smooth 日平滑 + quant 档位离散化，只有信号实质「变色」才调仓，
        把日均换手从 ~0.25 压到 ~0.14，避免交易成本吞噬薄利。
    """
    sig = pd.DataFrame(0.0, index=px.index, columns=cols)
    p = px[cols]
    r = ret[cols]
    for w in windows:
        mom = p / p.shift(w) - 1.0
        vol = r.rolling(w, min_periods=max(5, w // 2)).std() * np.sqrt(w) + 1e-6
        sig = sig + np.tanh(mom / vol).fillna(0.0)
    sig = sig / len(windows)
    if smooth:
        sig = sig.rolling(smooth, min_periods=1).mean()
    if quant:
        sig = (sig / quant).round() * quant
    return sig.clip(-1, 1)


# ------------------------------------------------- 第一层：贝塔基石
def equity_beta(px, ret):
    """宽基指数（股票贝塔）：最基础的经济增长红利，低成本获取市场平均收益。"""
    w = pd.DataFrame(0.0, index=ret.index, columns=['HS300'])
    w['HS300'] = 1.0
    return w


def bond_beta(px, ret):
    """债券贝塔代理：十年国债期货买入持有（仅用于 60/40 对照组）。"""
    w = pd.DataFrame(0.0, index=ret.index, columns=['T0'])
    w['T0'] = 1.0
    return w


def trend_lf(px, ret, mode='er', style='discrete'):
    """
    中低频 CTA（商品/趋势贝塔）：多品种趋势跟踪，逆波动率配置。
    mode='er'    : 用「行情流畅度」(ER) 调制信号 —— 文章核心观点
    mode='plain' : 纯多周期动量，不做过滤（对照基准）
    mode='vol'   : 用「波动率高低」调制信号（对照组，验证『波动率不是关键』）
    style='raw'  : 对照组，窗口含 10 日、不平滑不离散（信号过度灵敏、换手过高）
    另外叠加全市场 ER 过低时的主动降敞口（对应文章『主动降低趋势敞口』）。
    """
    cols = [c for c in COMMODITY if c in px.columns]
    if style == 'raw':
        sig = _trend_signal(px, ret, cols, windows=(10, 20, 40, 60, 120),
                            smooth=0, quant=0)
    else:
        sig = _trend_signal(px, ret, cols)
    iv = _inv_vol(ret, cols)

    if mode == 'er':
        e = efficiency_ratio(px[cols], 20)
        filt = ((e - 0.10) / 0.20).clip(0.25, 1.0).fillna(1.0)
        eff = sig * filt
    elif mode == 'vol':
        v = ret[cols].rolling(60, min_periods=20).std()
        vref = v.expanding(min_periods=60).mean()
        filt = (v / vref).clip(0.25, 1.0).fillna(1.0)
        eff = sig * filt
    else:
        eff = sig

    w = _normalize(eff * iv)

    if mode == 'er':
        er_mean = efficiency_ratio(px[cols], 20).mean(axis=1)
        global_filt = ((er_mean - 0.15) / 0.10).clip(0.5, 1.0).fillna(1.0)
        w = w.mul(global_filt.shift(1).fillna(1.0), axis=0)
    return w


# ------------------------------------------------- 第二层：稀缺阿尔法
def trend_mix(px, ret):
    """
    趋势跟踪 + 反转防守：文章指出「反转不作为主力独立存在，趋势持续太久或
    盈利丰厚时逐步提升反转权重，对冲趋势突然逆转带来的回撤」。
    这里用趋势信号强度的持续性作为「趋势太久了」的代理，动态分配反转权重。
    """
    cols = [c for c in COMMODITY if c in px.columns]
    sig = _trend_signal(px, ret, cols)
    persistence = sig.abs().rolling(20, min_periods=5).mean()
    alpha_rev = ((persistence - 0.5) / 0.5).clip(0.0, 0.5).fillna(0.0)

    r3 = ret[cols].rolling(3, min_periods=2).sum()
    rev = -_cross_z(r3)
    eff = sig * (1 - alpha_rev) + rev * alpha_rev
    return _normalize(eff * _inv_vol(ret, cols))


def hf_mr(px, ret, feats):
    """
    中高频 CTA 代理（独立于任何资产的绝对收益阿尔法）。
    用日线 OHLC 构造微结构特征：当日收益横截面反转、日内区间位置反转，
    做多短期超跌、做空短期超涨，横截面净敞口为零。

    ⚠ 重要局限：真实中高频 CTA 交易分钟/微秒级定价偏差，需 tick 数据、
      低延迟执行与专门硬件；此处仅为日线级「低相关性代理」，用于检验
      它在组合中的分散化作用，绝不代表真实中高频策略的可实现业绩。
    """
    cols = [c for c in COMMODITY if c in px.columns and c in feats]
    rp = pd.DataFrame({c: feats[c]['range_pos'] for c in cols})
    gap = pd.DataFrame({c: feats[c]['gap'] for c in cols})

    s = -(0.6 * _cross_z(ret[cols]) + 0.3 * _cross_z(rp) + 0.1 * _cross_z(gap))
    w = s * _inv_vol(ret, cols)
    return _neutralize(w)


def xs_mom(px, ret, lookback=60):
    """截面多空：横向对比全品种相对强弱，做多强势、做空弱势，不预判单边方向。"""
    cols = [c for c in COMMODITY if c in px.columns]
    r = px[cols] / px[cols].shift(lookback) - 1.0
    rank = r.rank(axis=1, pct=True)
    s = (rank - 0.5) * 2.0
    s = s.sub(s.mean(axis=1), axis=0)
    return _normalize(s * _inv_vol(ret, cols))


def spread_arb(px, ret, pairs=PAIRS, window=60, z_entry=1.5):
    """
    跨品种价差套利代理：同一产业链关联品种的价差均值回归，不博取商品单边 Beta。
    文章原述包含「跨期套利」，但主力连续合约无法还原同一品种不同交割月价差，
    因此这里以跨品种价差替代，并在报告中明确标注。
    ⚠ 未做协整检验，属简化研究实现，实盘需逐对检验协整与滚动稳定性。
    """
    cols = [c for c in COMMODITY if c in px.columns]
    w = pd.DataFrame(0.0, index=px.index, columns=cols)
    used = 0
    for a, b in pairs:
        if a not in px.columns or b not in px.columns:
            continue
        pa = px[a].replace(0, np.nan).clip(lower=1e-6)
        pb = px[b].replace(0, np.nan).clip(lower=1e-6)
        sp = np.log(pa / pb).replace([np.inf, -np.inf], np.nan).ffill()
        z = (sp - sp.rolling(window, min_periods=window // 2).mean()) / \
            (sp.rolling(window, min_periods=window // 2).std() + 1e-9)
        s = -np.tanh(z / z_entry).fillna(0.0)
        w[a] = w[a] + s / 2.0
        w[b] = w[b] - s / 2.0
        used += 1
    return _normalize(w), used


# ------------------------------------------------- 第三层：宏观配置
def macro_rp(px, ret, cidx):
    """
    宏观配置：以股/债/商三类贝塔资产为「食材」，用风险平价定「配比」，
    用 120 日动量做动态择时（前瞻性降低高风险资产仓位 = 防御性配置）。
    文章强调其「涨时跟得住、跌时控得住」的非对称收益结构。
    """
    cols = [c for c in COMMODITY if c in px.columns]
    r = pd.DataFrame({'eq': ret['HS300'], 'bond': ret['T0'], 'cmd': cidx})

    vol = r.rolling(60, min_periods=20).std() * np.sqrt(TRADING_DAYS) + 1e-6
    base = (1.0 / vol).div((1.0 / vol).sum(axis=1), axis=0)

    mom = r.rolling(120, min_periods=60).sum()
    timing = pd.DataFrame(np.where(mom > 0, 1.0, 0.30), index=r.index, columns=r.columns)
    w3 = (base * timing).div((base * timing).sum(axis=1), axis=0) * 0.95
    w3 = w3.fillna(0.0)

    iv = _inv_vol(ret, cols)
    W = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    W['HS300'] = w3['eq']
    W['T0'] = w3['bond']
    for c in cols:
        W[c] = w3['cmd'] * iv[c]
    return W


def build_all(bundle, cost_keys=None):
    """生成所有策略的原始权重，返回 {策略名: (权重, 成本bp, 层级, 说明)}"""
    from config import COST_BPS
    px, ret, feats, cidx = bundle['px'], bundle['ret'], bundle['feats'], bundle['cidx']
    spread_w, n_pairs = spread_arb(px, ret)
    return {
        'beta':        (equity_beta(px, ret), COST_BPS['beta'], '第一层·股票贝塔',
                        '买入持有沪深300，低成本获取市场平均收益'),
        'bond':        (bond_beta(px, ret), COST_BPS['bond'], '对照组·债券',
                        '十年国债期货买入持有，仅用于构建传统 60/40 对照'),
        'trend_lf':    (trend_lf(px, ret, 'er'), COST_BPS['trend_lf'], '第一层·趋势贝塔',
                        '多周期时序动量+流畅度(ER)过滤+逆波动率配置，震荡期主动降敞口'),
        'trend_plain': (trend_lf(px, ret, 'plain'), COST_BPS['trend_lf_plain'], '对照组',
                        '纯多周期时序动量，无过滤'),
        'trend_vol':   (trend_lf(px, ret, 'vol'), COST_BPS['trend_lf_vol'], '对照组',
                        '按波动率高低调制仓位，用于检验「波动率不是关键」'),
        'trend_raw':   (trend_lf(px, ret, 'plain', 'raw'), COST_BPS['trend_raw'], '对照组',
                        '含10日窗口、不平滑不离散，换手过高（验证「频繁变色=磨损」）'),
        'trend_mix':   (trend_mix(px, ret), COST_BPS['trend_mix'], '第二层·趋势增强',
                        '趋势为主、反转防守为辅，趋势持久化时动态提升反转权重'),
        'hf_mr':       (hf_mr(px, ret, feats), COST_BPS['hf_mr'], '第二层·纯阿尔法代理',
                        '日内微结构横截面反转，多空对冲（日线代理，非真实高频）'),
        'xs_mom':      (xs_mom(px, ret), COST_BPS['xs_mom'], '第二层·截面多空',
                        '60日相对强弱排名，多强空弱，市场中性'),
        'spread':      (spread_w, COST_BPS['spread'], '第二层·套利',
                        f'{n_pairs} 组产业链价差均值回归，与趋势天然互补'),
        'macro_rp':    (macro_rp(px, ret, cidx), COST_BPS['macro_rp'], '第三层·宏观配置',
                        '股/债/商风险平价 + 120日动量择时，动态风险预算'),
    }, n_pairs


# =============================================================== V2 策略层改进
def cointegrated_pairs(px, pairs, end=None, pvalue=0.10):
    """
    Engle-Granger 协整筛选：只在训练期（默认样本内末尾）检验，
    通过的配对列表冻结后用于全期回测，样本外不重新筛选 → 无前视。
    """
    from statsmodels.tsa.stattools import coint
    from config import IN_END
    end = end or IN_END
    good, all_stats = [], []
    for a, b in pairs:
        if a not in px.columns or b not in px.columns:
            continue
        s = pd.concat([np.log(px[a].replace(0, np.nan)),
                       np.log(px[b].replace(0, np.nan))], axis=1).loc[:end].dropna()
        if len(s) < 200:
            continue
        try:
            _, p, _ = coint(s.iloc[:, 0], s.iloc[:, 1], trend='c', maxlag=1)
        except Exception:
            continue
        all_stats.append({'a': a, 'b': b, 'pvalue': round(float(p), 4)})
        if p < pvalue:
            good.append((a, b))
    return good, all_stats


def spread_arb_v2(px, ret, pairs, window=60, hedge_window=120,
                  z_entry=1.5, z_exit=0.4):
    """
    协整版跨品种价差套利（修正 V1 失效项）：
      - 只交易通过协整检验的配对；
      - 滚动 OLS 动态对冲比（不使用未来数据）；
      - 带滞回：|z| > z_entry 开仓，|z| < z_exit 平仓，减少震荡期反复进出。
    """
    cols = [c for c in COMMODITY if c in px.columns]
    w = pd.DataFrame(0.0, index=px.index, columns=cols)
    for a, b in pairs:
        la = np.log(px[a].replace(0, np.nan).clip(lower=1e-6))
        lb = np.log(px[b].replace(0, np.nan).clip(lower=1e-6))
        beta = (la.rolling(hedge_window, min_periods=60).cov(lb) /
                (lb.rolling(hedge_window, min_periods=60).var() + 1e-9)).clip(0.2, 3.0)
        spread = la - beta * lb
        z = ((spread - spread.rolling(window, min_periods=window // 2).mean()) /
             (spread.rolling(window, min_periods=window // 2).std() + 1e-9)).fillna(0)
        # 滞回：离开死区才建仓，回落进死区即平仓
        mag = ((z.abs() - z_exit) / (z_entry - z_exit)).clip(0, 1)
        s = -np.sign(z) * mag
        w[a] = w[a] + s
        w[b] = w[b] - s * beta
    return _normalize(w)


def trend_mix_v2(px, ret, rev_cap=0.20, er_gate=0.26):
    """
    修正 V1 反转防守失效：反转权重上限 0.20，且只在「行情不流畅」时启用。
    文章原意是「趋势持续太久/盈利丰厚时提升反转权重做防守」，
    但 V1 在趋势市里也被反复打脸，故加流畅度状态开关。
    """
    cols = [c for c in COMMODITY if c in px.columns]
    sig = _trend_signal(px, ret, cols)
    persistence = sig.abs().rolling(20, min_periods=5).mean()
    alpha_rev = ((persistence - 0.5) / 0.5).clip(0.0, rev_cap).fillna(0.0)

    er_mean = efficiency_ratio(px[cols], 20).mean(axis=1)
    gate = ((er_gate - er_mean) / 0.06).clip(0.0, 1.0).fillna(0.0).shift(1).fillna(0.0)
    alpha_rev = alpha_rev.mul(gate, axis=0)   # 必须按行广播，否则日期索引被当成列名

    r3 = ret[cols].rolling(3, min_periods=2).sum()
    rev = -_cross_z(r3).fillna(0.0)          # 避免 0 × NaN 污染整行权重
    eff = sig * (1 - alpha_rev) + rev * alpha_rev
    return _normalize(eff * _inv_vol(ret, cols))


def trend_lf_v2(px, ret, er_low=0.10, er_high=0.30, stop_k=1.5,
                global_filt_on=True, windows=(20, 40, 60, 120), smooth=5, quant=0.25):
    """
    参数化趋势策略（供进化调参选择变体）：
      - ER 流畅度过滤阈值可学习；
      - 变色止损：单品种近期反向波动超过 stop_k 倍波动时先平仓，避免趋势反转硬扛；
      - 全市场流畅度低时主动降敞口。
    """
    cols = [c for c in COMMODITY if c in px.columns]
    sig = _trend_signal(px, ret, cols, windows=windows, smooth=smooth, quant=quant)
    er = efficiency_ratio(px[cols], 20)
    filt = ((er - er_low) / max(er_high - er_low, 1e-6)).clip(0.25, 1.0).fillna(1.0)

    # 变色止损：持仓方向与近 5 日累计收益相反且幅度过大 → 该品种清仓
    r5 = ret[cols].rolling(5, min_periods=3).sum()
    v5 = ret[cols].rolling(60, min_periods=20).std() * np.sqrt(5) + 1e-6
    stop = (np.sign(sig) * r5) < (-stop_k * v5)
    sig = sig.mask(stop, 0.0)

    w = _normalize(sig * filt * _inv_vol(ret, cols))
    if global_filt_on:
        er_mean = er.mean(axis=1)
        g = ((er_mean - 0.15) / 0.10).clip(0.5, 1.0).fillna(1.0).shift(1).fillna(1.0)
        w = w.mul(g, axis=0)
    return w


def style_neutral(px, ret, lookback=60, hedge_window=120):
    """
    市场中性代理：多中证500期货 / 空沪深300期货，用滚动 beta 对冲风格暴露。
    对应文章「市场中性 + CTA 混合更理想」的组合协同思路；
    注意这是风格价差贝塔，不是选股阿尔法（报告中明确标注）。
    """
    w = pd.DataFrame(0.0, index=ret.index, columns=['IF0', 'IC0'])
    ri, rc = ret['IF0'], ret['IC0']
    beta = (rc.rolling(hedge_window, min_periods=60).cov(ri) /
            (ri.rolling(hedge_window, min_periods=60).var() + 1e-9)).clip(0.3, 3.0)
    rel = (px['IC0'] / px['IC0'].shift(lookback) -
           px['IF0'] / px['IF0'].shift(lookback)).fillna(0.0)
    s = np.tanh(rel / (rel.rolling(120, min_periods=60).std() + 1e-6)).fillna(0.0)
    w['IC0'] = s
    w['IF0'] = -s * beta
    return _normalize(w)
