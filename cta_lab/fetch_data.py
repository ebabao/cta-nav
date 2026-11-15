# -*- coding: utf-8 -*-
"""
数据层：抓取国内商品期货主力连续日线 + 宽基指数 + 国债期货
数据源：akshare / 新浪财经（futures_main_sina, stock_zh_index_daily）
输出：data/raw/{symbol}.csv  以及合并后的 data/prices.csv

说明：
- 主力连续合约由新浪拼接，换月处存在价格跳空，回测中统一使用「收益率」而非绝对价，
  并在合成信号时做换月清洗（见 clean.py）。
- 不使用任何模拟/编造数据；抓取失败的品种直接剔除并记录到 data/fetch_report.json。
"""
import os
import json
import time
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
import akshare as ak

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw')
os.makedirs(RAW, exist_ok=True)

START, END = '20230101', '20261114'

# 品种池：覆盖文章所述「多品种趋势贝塔」与宏观配置所需的股/债/商三类资产
UNIVERSE = {
    # 贵金属 / 有色
    'CU0': ('沪铜', '有色'), 'AL0': ('沪铝', '有色'), 'NI0': ('沪镍', '有色'),
    'AU0': ('沪金', '贵金属'), 'AG0': ('沪银', '贵金属'),
    # 黑色
    'RB0': ('螺纹钢', '黑色'), 'HC0': ('热卷', '黑色'), 'I0': ('铁矿石', '黑色'),
    'J0': ('焦炭', '黑色'), 'JM0': ('焦煤', '黑色'),
    # 能源化工
    'SC0': ('原油', '能化'), 'TA0': ('PTA', '能化'), 'MA0': ('甲醇', '能化'),
    'L0': ('塑料', '能化'), 'PP0': ('聚丙烯', '能化'), 'EG0': ('乙二醇', '能化'),
    'SA0': ('纯碱', '能化'), 'RU0': ('天然橡胶', '能化'), 'FU0': ('燃油', '能化'),
    # 农产品
    'M0': ('豆粕', '农产品'), 'Y0': ('豆油', '农产品'), 'P0': ('棕榈油', '农产品'),
    'C0': ('玉米', '农产品'), 'SR0': ('白糖', '农产品'), 'CF0': ('棉花', '农产品'),
    'OI0': ('菜油', '农产品'), 'RM0': ('菜粕', '农产品'),
    # 金融期货（宏观配置层 / 股票贝塔代理）
    'IF0': ('沪深300期货', '金融'), 'IC0': ('中证500期货', '金融'),
    'T0': ('十年国债期货', '金融'), 'TF0': ('五年国债期货', '金融'),
}


def fetch_one(symbol, retries=3):
    for i in range(retries):
        try:
            df = ak.futures_main_sina(symbol=symbol, start_date=START, end_date=END)
            if df is None or len(df) < 200:
                raise ValueError(f'数据过少 {0 if df is None else len(df)} 行')
            df = df.rename(columns={'日期': 'date', '开盘价': 'open', '最高价': 'high',
                                   '最低价': 'low', '收盘价': 'close', '成交量': 'volume',
                                   '持仓量': 'oi', '动态结算价': 'settle'})
            df['date'] = pd.to_datetime(df['date'])
            for c in ['open', 'high', 'low', 'close', 'volume', 'oi', 'settle']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'oi', 'settle']]
            df = df.dropna(subset=['close']).sort_values('date').drop_duplicates('date')
            df.to_csv(os.path.join(RAW, f'{symbol}.csv'), index=False)
            return {'status': 'ok', 'rows': len(df),
                    'start': str(df.date.min().date()), 'end': str(df.date.max().date())}
        except Exception as e:
            if i == retries - 1:
                return {'status': 'fail', 'error': f'{type(e).__name__}: {str(e)[:120]}'}
            time.sleep(1.0 + i)
    return {'status': 'fail', 'error': 'unknown'}


def fetch_index():
    """沪深300指数，作为第一层「宽基指数（股票贝塔）」的可投资代理"""
    try:
        df = ak.stock_zh_index_daily(symbol='sh000300')
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date')
        df = df[df.date >= START].drop_duplicates('date')
        df.to_csv(os.path.join(RAW, 'HS300.csv'), index=False)
        return {'status': 'ok', 'rows': len(df),
                'start': str(df.date.min().date()), 'end': str(df.date.max().date())}
    except Exception as e:
        return {'status': 'fail', 'error': f'{type(e).__name__}: {str(e)[:120]}'}


def main():
    report = {'source': 'akshare / 新浪财经', 'start': START, 'end': END, 'symbols': {}}
    frames = {}
    for sym, (name, sector) in UNIVERSE.items():
        r = fetch_one(sym)
        report['symbols'][sym] = dict(r, name=name, sector=sector)
        print(f'{sym:5s} {name:8s} {r.get("status")} {r.get("rows", r.get("error", ""))}')
        if r['status'] == 'ok':
            d = pd.read_csv(os.path.join(RAW, f'{sym}.csv'), parse_dates=['date'])
            frames[sym] = d.set_index('date')['close']
        time.sleep(0.35)

    ri = fetch_index()
    report['symbols']['HS300'] = dict(ri, name='沪深300指数', sector='股票指数')
    print(f'HS300 沪深300指数  {ri.get("status")} {ri.get("rows", ri.get("error", ""))}')
    if ri['status'] == 'ok':
        d = pd.read_csv(os.path.join(RAW, 'HS300.csv'), parse_dates=['date'])
        frames['HS300'] = d.set_index('date')['close']

    px = pd.DataFrame(frames).sort_index()
    px.to_csv(os.path.join(BASE, 'data', 'prices.csv'))
    report['merged_shape'] = list(px.shape)
    report['merged_range'] = [str(px.index.min().date()), str(px.index.max().date())]
    with open(os.path.join(BASE, 'data', 'fetch_report.json'), 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('\n合并价格矩阵:', px.shape, report['merged_range'])
    print('失败品种:', [k for k, v in report['symbols'].items() if v['status'] != 'ok'])


if __name__ == '__main__':
    main()
