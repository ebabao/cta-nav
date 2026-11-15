# -*- coding: utf-8 -*-
"""
V2 组合层：机器学习训练 + 进化调参 + 动态风险预算。

防过拟合 / 防前视设计（本模块最核心的约束）：
  1) 标签是「未来 horizon 日各策略风险调整收益的横截面百分位排名」，
     学的是相对吸引力而非绝对收益预测，天然规避市场整体涨跌噪声；
  2) purged walk-forward：训练样本最后一个日期必须 ≤ 预测起点 - horizon，
     保证标签在预测时点之前完全实现，绝不使用未来信息；
  3) 每个 fold 只在历史窗口内 fit（含 StandardScaler），滚动前向推断；
  4) 进化调参的适应度只在【样本内】区间计算，最优参数冻结后
     样本外只做一次前向推断，样本外不参与任何拟合/选择；
  5) ML 模型配置固定，进化只优化组合层超参，避免嵌套训练导致的
     评估爆炸与「优化器过拟合」。
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import TRADING_DAYS, IN_START, IN_END, COMMODITY
from strategies import efficiency_ratio
from engine import metrics, apply_risk_overlay


# ---------------------------------------------------------------- 特征工程
def build_panel(bundle, S, core, horizon=20):
    """构造 (date, strategy) 面板特征 X 与排名标签 y。所有输入均为 t 日收盘可得。"""
    px, ret, cidx = bundle['px'], bundle['ret'], bundle['cidx']
    comm = [c for c in COMMODITY if c in px.columns]
    eps = 1e-9

    er20 = efficiency_ratio(px[comm], 20).mean(axis=1)
    er60 = efficiency_ratio(px[comm], 60).mean(axis=1)
    vol20 = ret[comm].rolling(20, min_periods=10).std().mean(axis=1) * np.sqrt(TRADING_DAYS)
    vol60 = ret[comm].rolling(60, min_periods=30).std().mean(axis=1) * np.sqrt(TRADING_DAYS)
    disp = ret[comm].std(axis=1).rolling(20, min_periods=10).mean()
    eq_vol = ret['HS300'].rolling(20, min_periods=10).std() * np.sqrt(TRADING_DAYS)
    cmd_mom20, cmd_mom60 = cidx.rolling(20).sum(), cidx.rolling(60).sum()
    eq_mom20, eq_mom60 = ret['HS300'].rolling(20).sum(), ret['HS300'].rolling(60).sum()
    bd_mom20, bd_mom60 = ret['T0'].rolling(20).sum(), ret['T0'].rolling(60).sum()

    basket = S[core].mean(axis=1)
    bnav = (1 + basket).cumprod()
    basket_dd = bnav / bnav.cummax() - 1

    market = pd.concat({
        'm_er20': er20, 'm_er60': er60, 'm_vol20': vol20, 'm_vol60': vol60,
        'm_disp': disp, 'm_eqvol': eq_vol, 'm_cmd20': cmd_mom20, 'm_cmd60': cmd_mom60,
        'm_eq20': eq_mom20, 'm_eq60': eq_mom60, 'm_bd20': bd_mom20, 'm_bd60': bd_mom60,
        'm_basketdd': basket_dd,
    }, axis=1)

    strat_feat = {}
    for s in core:
        p = S[s]
        nav = (1 + p).cumprod()
        dd = nav / nav.cummax() - 1
        strat_feat[s] = pd.DataFrame({
            's_ret20': p.rolling(20, min_periods=10).sum(),
            's_ret60': p.rolling(60, min_periods=30).sum(),
            's_shp20': p.rolling(20, min_periods=10).mean() /
                       (p.rolling(20, min_periods=10).std() + eps) * np.sqrt(TRADING_DAYS),
            's_vol20': p.rolling(20, min_periods=10).std() * np.sqrt(TRADING_DAYS),
            's_dd': dd,
            's_corr': p.rolling(60, min_periods=30).corr(basket),
        }, index=S.index)

    dates, n_s = S.index, len(core)
    full_idx = pd.MultiIndex.from_product([dates, core], names=['date', 'strategy'])
    parts = []
    for s in core:
        f = strat_feat[s].copy()
        for c in market.columns:
            f[c] = market[c]
        for s2 in core:
            f['id_' + s2] = 1.0 if s == s2 else 0.0
        f['_s'] = s
        parts.append(f)
    X = pd.concat(parts, axis=0).set_index('_s', append=True)
    X.index = X.index.set_names(['date', 'strategy'])
    X = X.reindex(full_idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 标签：未来 horizon 日累计收益 / 当前波动 → 横截面百分位排名
    fwd = pd.DataFrame(0.0, index=dates, columns=core)
    for h in range(1, horizon + 1):
        fwd = fwd + S[core].shift(-h).fillna(0.0)
    scale = S[core].rolling(60, min_periods=30).std() * np.sqrt(horizon) + eps
    y_adj = (fwd / scale).where(fwd.notnull().all(axis=1), np.nan)
    y_rank = y_adj.rank(axis=1, pct=True)
    y = y_rank.stack().reindex(full_idx).astype(float)
    return X, y, list(X.columns)


# ------------------------------------------------- ML 路径 2：风险状态预测
def build_risk_panel(bundle, p_prior, horizon=20):
    """
    日频市场状态特征 → 预测「未来 horizon 日组合实现波动」与「高回撤概率」。
    波动率的持续性远高于收益，因此比直接预测收益更可学习，且能直接用于压回撤。
    """
    from sklearn.linear_model import LogisticRegression
    px, ret, cidx = bundle['px'], bundle['ret'], bundle['cidx']
    comm = [c for c in COMMODITY if c in px.columns]
    idx = p_prior.index
    er20 = efficiency_ratio(px[comm], 20).mean(axis=1).reindex(idx)
    er60 = efficiency_ratio(px[comm], 60).mean(axis=1).reindex(idx)
    v20 = ret[comm].rolling(20).std().mean(axis=1).reindex(idx) * np.sqrt(TRADING_DAYS)
    v60 = ret[comm].rolling(60).std().mean(axis=1).reindex(idx) * np.sqrt(TRADING_DAYS)
    feat = pd.concat({
        'er20': er20, 'er60': er60, 'v20': v20, 'v60': v60,
        'disp': ret[comm].std(axis=1).rolling(20).mean().reindex(idx),
        'eqv': ret['HS300'].rolling(20).std().reindex(idx) * np.sqrt(TRADING_DAYS),
        'cmd20': cidx.rolling(20).sum().reindex(idx),
        'cmd60': cidx.rolling(60).sum().reindex(idx),
        'eq20': ret['HS300'].rolling(20).sum().reindex(idx),
        'eq60': ret['HS300'].rolling(60).sum().reindex(idx),
        'bd20': ret['T0'].rolling(20).sum().reindex(idx),
        'pvol20': p_prior.rolling(20).std() * np.sqrt(TRADING_DAYS),
        'pret20': p_prior.rolling(20).sum(),
    }, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n = len(p_prior)
    fvol = np.full(n, np.nan)
    fdd = np.zeros(n)
    pv = p_prior.values
    for t in range(n - horizon):
        seg = pv[t + 1:t + horizon + 1]
        nav = np.cumprod(1 + seg)
        fvol[t] = seg.std() * np.sqrt(TRADING_DAYS)
        fdd[t] = float((nav / np.maximum.accumulate(nav) - 1).min() < -0.025)
    return feat, fvol, fdd


def walkforward_risk(feat, fvol, fdd, train_window=300, step=10,
                     horizon=20, start_offset=140):
    from sklearn.linear_model import LogisticRegression
    X = feat.values
    n = len(feat)
    pred_vol = np.full(n, np.nan)
    pred_risk = np.full(n, np.nan)
    v_fallback = feat['v60'].values
    for T in range(start_offset, n, step):
        lo, hi = max(0, T - train_window - horizon), T - horizon
        if hi - lo < 80:
            continue
        m = np.isfinite(fvol[lo:hi])
        if m.sum() < 80:
            continue
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
        ridge.fit(X[lo:hi][m], fvol[lo:hi][m])
        end = min(T + step, n)
        pred_vol[T:end] = ridge.predict(X[T:end])
        # 高回撤是稀疏事件：某些窗口可能只有一个类别，此时降级为「无风险信号」
        y_clf = fdd[lo:hi][m]
        if len(np.unique(y_clf)) >= 2:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.1, max_iter=500))
            clf.fit(X[lo:hi][m], y_clf)
            pred_risk[T:end] = clf.predict_proba(X[T:end])[:, 1]
        else:
            pred_risk[T:end] = 0.0
    pred_vol = pd.Series(np.where(np.isfinite(pred_vol), pred_vol, v_fallback),
                         index=feat.index)
    pred_risk = pd.Series(np.where(np.isfinite(pred_risk), pred_risk, 0.5),
                          index=feat.index)
    return pred_vol, pred_risk


# ---------------------------------------------------------------- walk-forward 训练
def walkforward_predict(X, y, core, train_window=300, step=10, horizon=20,
                        ridge_alpha=30.0, blend=0.5, start_offset=140):
    """返回每日各策略的「相对吸引力」预测（0~1），滚动前向、样本外可做一次推断。"""
    dates = X.index.get_level_values(0).unique()
    n_d, n_s, n_f = len(dates), len(core), X.shape[1]
    Xv = X.values.reshape(n_d, n_s, n_f)
    yv = y.values.reshape(n_d, n_s)
    pred = np.full((n_d, n_s), np.nan)
    purge = horizon
    fold_log = []

    for T in range(start_offset, n_d, step):
        lo, hi = max(0, T - train_window - purge), T - purge
        if hi - lo < 80:
            continue
        Xtr, ytr = Xv[lo:hi].reshape(-1, n_f), yv[lo:hi].reshape(-1)
        m = np.isfinite(ytr) & np.isfinite(Xtr).all(axis=1)
        if m.sum() < 100:
            continue
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        rf = RandomForestRegressor(n_estimators=60, max_depth=4,
                                   min_samples_leaf=50, random_state=42, n_jobs=1)
        ridge.fit(Xtr[m], ytr[m])
        rf.fit(Xtr[m], ytr[m])
        end = min(T + step, n_d)
        Xte = Xv[T:end].reshape(-1, n_f)
        p = blend * ridge.predict(Xte) + (1 - blend) * rf.predict(Xte)
        pred[T:end] = p.reshape(-1, n_s)
        fold_log.append({'predict_start': str(dates[T].date()),
                         'train_rows': int(m.sum()),
                         'train_end': str(dates[hi - 1].date())})

    pred = np.where(np.isfinite(pred), pred, 0.5)
    return pd.DataFrame(pred, index=dates, columns=core), fold_log


# ---------------------------------------------------------------- 权重与组合
def weights_from_pred(pred, temperature=0.40, lambda_ml=0.20, w_cap=0.45,
                      w_anchor=None):
    """
    ML 只做「小幅微调」：以有效先验权重为锚，混入 softmax 动态权重。
    这是对小样本的关键约束——不允许模型推翻已被验证的三层框架配置。
    """
    z = (pred - 0.5) / max(temperature, 1e-3)
    e = np.exp(z - z.max(axis=1).values.reshape(-1, 1))
    w_ml = pd.DataFrame(e / e.sum(axis=1).values.reshape(-1, 1),
                        index=pred.index, columns=pred.columns)
    a = pd.Series(w_anchor).reindex(pred.columns).fillna(0.0) if w_anchor is not None \
        else pd.Series(1.0 / len(pred.columns), index=pred.columns)
    a = a / a.sum()
    w = w_ml.mul(lambda_ml).add((1 - lambda_ml) * a, axis=1)
    w = w.clip(upper=w_cap)
    return w.div(w.sum(axis=1), axis=0)


def combine(S, weights, rebal_bps=2.0):
    """按动态权重叠加策略收益，扣组合层再平衡成本；权重 t 日形成、t+1 生效。"""
    w = weights.reindex(index=S.index, columns=S.columns).ffill().fillna(0.0)
    p = (w.shift(1) * S).sum(axis=1)
    cost = w.diff().abs().sum(axis=1) * rebal_bps / 1e4
    cost.iloc[0] = w.iloc[0].abs().sum() * rebal_bps / 1e4
    return (p - cost).fillna(0.0), cost


# ---------------------------------------------------------------- 进化调参
def evaluate(S, pred, w_anchor, params, start, end, rebal_bps=2.0, pred_vol=None):
    """给定超参，返回区间内的 (pnl, metrics, 平均再平衡成本, 杠杆序列)。"""
    (temperature, lambda_ml, w_cap, target_vol,
     vol_window, dd_thr, dd_floor, mu_ml) = params[:8]
    w = weights_from_pred(pred, temperature, lambda_ml, w_cap, w_anchor=w_anchor)
    p, cost = combine(S, w, rebal_bps)
    # 只降不升：波动率目标不允许加杠杆，避免平静期加仓放大尾部风险
    p2, lev, detail = apply_risk_overlay(p, target_vol=target_vol,
                                         vol_window=int(round(vol_window)),
                                         dd_threshold=dd_thr, dd_floor=dd_floor,
                                         vol_cap=(0.35, 1.0))
    # ML 路径 2：用「预测的未来波动」替代部分「滚动已实现波动」来定杠杆
    if pred_vol is not None and mu_ml > 0:
        ml_v = (target_vol / pred_vol.reindex(p.index).replace(0, np.nan)) \
            .clip(0.4, 1.0).shift(1).fillna(1.0)
        lev_v = (1 - mu_ml) * detail['lev_vol'] + mu_ml * ml_v
        lev = (lev_v * detail['lev_dd']).clip(0.2, 1.0)
        p2 = p * lev
    seg = p2.loc[start:end]
    m = metrics(seg)
    return seg, m, float(cost.loc[start:end].mean() * TRADING_DAYS), lev


def evolve(S, pred, w_anchor, start=IN_START, end=IN_END,
           maxiter=14, popsize=6, seed=42, rebal_bps=2.0, pred_vol=None):
    """
    差分进化：只在样本内区间优化组合层超参。

    抗过拟合三重约束（V2 初版在此翻车：优化器把 λ 推到 0.94、温度压到 0.068）：
      1) 收紧搜索空间：ML 占比 ≤ 0.35、温度 ≥ 0.25、杠杆只降不升；
      2) maximin 适应度：取样本内 3 个子区间中【最差】的夏普为主，
         避免只拟合某一段行情；
      3) 显式正则：惩罚过高的 ML 占比与过尖的 softmax。
    """
    from scipy.optimize import differential_evolution
    bounds = [
        (0.25, 0.80),    # temperature：禁止尖锐押注
        (0.00, 0.35),    # lambda_ml：ML 仅限微调
        (0.30, 0.55),    # w_cap：单策略上限
        (0.08, 0.11),    # target_vol
        (30, 70),        # vol_window
        (0.03, 0.08),    # dd_threshold
        (0.35, 0.80),    # dd_floor
        (0.00, 0.80),    # mu_ml：ML 预测波动在杠杆中的占比
    ]
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 3
    mid2 = pd.Timestamp(start) + 2 * (pd.Timestamp(end) - pd.Timestamp(start)) / 3
    folds = [(start, str(mid.date())), (str(mid.date()), str(mid2.date())),
             (str(mid2.date()), end)]

    def obj(x):
        seg, m, c, _ = evaluate(S, pred, w_anchor, x, start, end, rebal_bps,
                                pred_vol=pred_vol)
        if not m:
            return 1e6
        sh = [metrics(seg.loc[a:b])['sharpe'] for a, b in folds]
        sh = [s for s in sh if s is not None]
        if not sh:
            return 1e6
        worst, avg = min(sh), float(np.mean(sh))
        reg = 0.25 * x[1] + 0.15 * max(0.0, 0.35 - x[0])
        return -(0.4 * avg + 0.6 * worst - 2.0 * abs(m['mdd']) - 0.5 * c - reg)

    res = differential_evolution(obj, bounds, maxiter=maxiter, popsize=popsize,
                                 seed=seed, polish=True, tol=1e-4,
                                 updating='immediate', workers=1)
    return res.x, float(-res.fun), int(res.nfev)
