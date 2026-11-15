# -*- coding: utf-8 -*-
"""
探索性分析：数据质量检查 + 「行情流畅度」实证
用 Kaufman 效率比 ER = |P_t - P_{t-n}| / Σ|P_i - P_{i-1}| 度量趋势流畅度：
ER 越接近 1 越流畅（趋势策略友好），越接近 0 越震荡（磨损期）。
输出 data/eda_report.json 供报告引用。
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
from data_prep import load_prices, outlier_mask
px = load_prices()                      # 与回测保持同一价格口径（商品=结算价）
ret = px.pct_change(fill_method=None)

COMMODITY = [c for c in px.columns if c not in ('HS300', 'IF0', 'IC0', 'T0', 'TF0')]

def efficiency_ratio(p, n=20):
    diff = p.diff().abs()
    return (p - p.shift(n)).abs() / (diff.rolling(n).sum() + 1e-12)

er20 = px[COMMODITY].apply(lambda s: efficiency_ratio(s, 20))
er60 = px[COMMODITY].apply(lambda s: efficiency_ratio(s, 60))

# 异常收益（主力连续换月跳空）检查：与回测共用同一清洗口径
abs_ret = ret[COMMODITY].abs()
bad = outlier_mask(px)[COMMODITY]
quality = {
    'n_days': int(len(px)),
    'n_assets': int(px.shape[1]),
    'range': [str(px.index.min().date()), str(px.index.max().date())],
    'missing_pct': float(px.isna().mean().mean()),
    'cleaned_outlier_count': int(bad.sum().sum()),
    'cleaned_outlier_ratio': float(bad.sum().sum() / abs_ret.size),
    'per_asset_annual_vol': {c: round(float(ret[c].std() * np.sqrt(252)), 4) for c in px.columns},
}

# 分年度 / 半年度流畅度（全商品等权平均）
def period_er(er, start, end):
    s = er.loc[start:end].mean().mean()
    return round(float(s), 4)

periods = {
    '2023H1': ('2023-01-01', '2023-06-30'), '2023H2': ('2023-07-01', '2023-12-31'),
    '2024H1': ('2024-01-01', '2024-06-30'), '2024H2': ('2024-07-01', '2024-12-31'),
    '2025H1': ('2025-01-01', '2025-06-30'), '2025H2': ('2025-07-01', '2025-12-31'),
    '2026Q1': ('2026-01-01', '2026-03-31'), '2026Q2': ('2026-04-01', '2026-06-30'),
    '2026H2': ('2026-07-01', '2026-11-13'),
}
quality['efficiency_ratio_20d_by_period'] = {k: period_er(er20, *v) for k, v in periods.items()}
quality['efficiency_ratio_60d_by_period'] = {k: period_er(er60, *v) for k, v in periods.items()}
quality['efficiency_ratio_20d_overall'] = round(float(er20.mean().mean()), 4)

# 各商品年度收益（用于对照文章对黄金/铜/原油的描述）
quality['annual_returns'] = {}
for y in ['2023', '2024', '2025', '2026']:
    sub = px.loc[f'{y}-01-01':f'{y}-12-31']
    if len(sub) > 5:
        r = (sub.iloc[-1] / sub.iloc[0] - 1).round(4)
        quality['annual_returns'][y] = {c: float(r[c]) for c in ['AU0', 'CU0', 'SC0', 'RB0', 'M0']}

with open(os.path.join(BASE, 'data', 'eda_report.json'), 'w') as f:
    json.dump(quality, f, ensure_ascii=False, indent=2)

print('缺失率:', round(quality['missing_pct'] * 100, 3), '%')
print('清洗异常点:', quality['cleaned_outlier_count'],
      '占比', round(quality['cleaned_outlier_ratio'] * 100, 3), '%')
print('\n20日流畅度(ER) 分期：')
for k, v in quality['efficiency_ratio_20d_by_period'].items():
    print(f'  {k}: {v}')
print('\n60日流畅度(ER) 分期：')
for k, v in quality['efficiency_ratio_60d_by_period'].items():
    print(f'  {k}: {v}')
print('\n关键品种年度收益：')
for y, d in quality['annual_returns'].items():
    print(' ', y, {k: f'{v*100:.1f}%' for k, v in d.items()})
