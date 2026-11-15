# -*- coding: utf-8 -*-
"""数据清洗：主力连续换月跳空处理、收益率矩阵、OHLC 微结构特征。"""
import os
import numpy as np
import pandas as pd
from config import DATA, COMMODITY, FINANCIAL, TRADING_DAYS


def load_prices():
    """
    价格矩阵口径选择（重要工程决策）：
    新浪「主力连续」的『收盘价』字段混合了不同口径，实测 27 个商品品种的
    收益率平均 1 阶自相关 = -0.0217（日内噪声/买卖价跳动），会系统性压制
    趋势策略、虚增反转策略；而交易所公布的『动态结算价』自相关 = +0.181，
    具备真实的趋势延续特征。CTA 实务亦以结算价估值，故商品统一改用结算价。
    宽基指数（HS300）无结算价，沿用收盘价。
    """
    px = pd.read_csv(os.path.join(DATA, 'prices.csv'), parse_dates=['date']).set_index('date')
    px = px.sort_index()
    for sym in COMMODITY:
        f = os.path.join(DATA, 'raw', f'{sym}.csv')
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, parse_dates=['date']).set_index('date')
        px[sym] = d['settle'].reindex(px.index)
    return px


def outlier_mask(px):
    """
    主力连续合约在换月拼接处会产生非真实的价格跳空，不能计为可交易损益。
    判定：|r| > 12% 或 |r| > 6 倍 60 日滚动波动 → 视为拼接异常。
    EDA 与回测共用本函数，保证统计口径一致。
    """
    ret = px.pct_change(fill_method=None)
    vol60 = ret.rolling(60, min_periods=20).std().fillna(0.01)
    thr = np.maximum(0.12, 6.0 * vol60)
    return ret.abs() > thr


def clean_returns(px):
    bad = outlier_mask(px)
    ret = px.pct_change(fill_method=None)
    n_bad = int(bad.sum().sum())
    ret = ret.mask(bad, 0.0).fillna(0.0)
    return ret, n_bad


def load_ohlc_features(px):
    """从原始 OHLC 构造日内微结构特征，作为中高频 CTA 的日线代理输入。"""
    feats = {}
    for sym in px.columns:
        f = os.path.join(DATA, 'raw', f'{sym}.csv')
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, parse_dates=['date']).set_index('date').reindex(px.index)
        o, h, l, c = d['open'], d['high'], d['low'], d['close']
        prev_c = c.shift(1)
        feats[sym] = pd.DataFrame({
            'gap': o / prev_c - 1,                       # 隔夜跳空
            'intraday': c / o - 1,                       # 日内动量
            'range_pos': (c - l) / (h - l + 1e-9) - 0.5,  # 收盘在当日区间的位置
            'range': (h - l) / (prev_c + 1e-9),         # 日内振幅
        })
    return feats


def commodity_index(ret):
    """商品等权指数收益，作为宏观配置层的「商品」资产代理。"""
    r = ret[COMMODITY].copy()
    iv = 1.0 / (r.rolling(60, min_periods=20).std() * np.sqrt(TRADING_DAYS) + 1e-6)
    w = iv.div(iv.sum(axis=1), axis=0)
    return (w.shift(1) * r).sum(axis=1)


def build():
    px = load_prices()
    ret, n_bad = clean_returns(px)
    feats = load_ohlc_features(px)
    cidx = commodity_index(ret)
    return dict(px=px, ret=ret, feats=feats, cidx=cidx, n_bad=n_bad)


if __name__ == '__main__':
    b = build()
    print('价格矩阵', b['px'].shape, '清洗异常点', b['n_bad'])
    print('商品指数', b['cidx'].describe().round(4).to_dict())
