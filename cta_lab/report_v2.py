# -*- coding: utf-8 -*-
"""生成 V2 改进研究报告（单文件 HTML，Plotly 内嵌，离线可打开）。"""
import os, json
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

BASE = os.path.dirname(os.path.abspath(__file__))
OUTHTML = os.path.join(os.path.dirname(BASE), 'cta_v2_research_report.html')
R = json.load(open(os.path.join(BASE, 'output', 'results_v2.json')))

M, V, VAL = R['meta'], R['variants'], R['validation']
CUR, MLG = R['curves'], VAL['ml_gain']
BP = R['best_params']
CORE = R['core']

# 曲线统一截断到评估区间
all_dates = CUR['dates']
i0 = next(i for i, d in enumerate(all_dates) if d >= M['in'][0])
i1 = next((i for i, d in enumerate(all_dates) if d > M['oos'][1]), len(all_dates))
DATES = all_dates[i0:i1]
def cut(seq):
    return list(seq)[i0:i1]

COLORS = ['#8a9aa8', '#c8102e', '#e07b39', '#2a9d8f', '#8e5db7', '#1f4e79', '#6b7280']
KEYS = list(V.keys())
_plots = {}
def reg(k, fig):
    _plots[k] = fig.to_html(full_html=False, include_plotlyjs=False,
                            div_id='plot_' + k,
                            config={'displayModeBar': False, 'responsive': True})

def pct(x, d=2): return f'{x*100:+.{d}f}%'
def num(x, d=2): return f'{x:.{d}f}'
def cspan(x, txt=None, d=2, kind='pct'):
    s = txt if txt is not None else (pct(x, d) if kind == 'pct' else num(x, d))
    c = 'pos' if x > 0 else ('neg' if x < 0 else 'flat')
    return f'<span class="{c}">{s}</span>'
def table(hs, rows):
    h = ''.join(f'<th>{x}</th>' for x in hs)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<table class="dt"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


# ---------------------------------------------------------------- 图
def f_nav():
    f = go.Figure()
    for i, k in enumerate(KEYS):
        f.add_trace(go.Scatter(x=DATES, y=cut(CUR['nav'][k]), name=k, mode='lines',
                               line=dict(color=COLORS[i % len(COLORS)],
                                         width=2.4 if i in (1, 5) else 1.4,
                                         dash='dot' if i == 0 else 'solid'),
                               hovertemplate='%{x}<br>%{y:.3f}<extra>' + k + '</extra>'))
    f.add_vline(x=M['oos'][0], line=dict(color='#999', dash='dash', width=1.5))
    f.add_annotation(x=M['oos'][0], y=1.0, text='样本外起点', showarrow=False,
                     yshift=12, font=dict(size=11, color='#666'))
    f.update_layout(title='各变体净值曲线（评估区间，起点=1）', xaxis_title='日期',
                    yaxis_title='净值', height=470, hovermode='x unified',
                    legend=dict(orientation='h', y=-0.30, font=dict(size=10)),
                    margin=dict(l=60, r=25, t=55, b=120), plot_bgcolor='#fff',
                    xaxis=dict(gridcolor='#eef1f4'), yaxis=dict(gridcolor='#eef1f4'))
    return f
reg('nav', f_nav())

def f_dd():
    ks = [KEYS[i] for i in (0, 1, 2, 5)]
    f = go.Figure()
    for i, k in enumerate(ks):
        f.add_trace(go.Scatter(x=DATES, y=cut(CUR['dd'][k]), name=k, mode='lines',
                               line=dict(color=COLORS[KEYS.index(k) % len(COLORS)], width=1.7),
                               hovertemplate='%{x}<br>%{y:.2%}<extra>' + k + '</extra>'))
    f.update_layout(title='回撤曲线：压回撤效果', xaxis_title='日期', yaxis_title='回撤',
                    yaxis_tickformat='.0%', height=350, hovermode='x unified',
                    legend=dict(orientation='h', y=-0.30, font=dict(size=10)),
                    margin=dict(l=60, r=25, t=55, b=100), plot_bgcolor='#fff',
                    xaxis=dict(gridcolor='#eef1f4'), yaxis=dict(gridcolor='#eef1f4'))
    return f
reg('dd', f_dd())

def f_layers():
    labels = ['B0 V1旧口径', 'B1 公平基线', 'V2-A 策略层', 'V2-B 风控',
              'V2-C ML微调', 'V2-D 完整', 'V2-E 无ML']
    f = go.Figure()
    f.add_trace(go.Bar(name='样本内夏普', x=labels,
                       y=[V[k]['metrics_in']['sharpe'] for k in KEYS], marker_color='#9db7d1'))
    f.add_trace(go.Bar(name='样本外夏普', x=labels,
                       y=[V[k]['metrics_oos']['sharpe'] for k in KEYS], marker_color='#1f4e79'))
    f.add_trace(go.Scatter(name='样本外最大回撤(右轴)', x=labels,
                           y=[abs(V[k]['metrics_oos']['mdd']) for k in KEYS],
                           mode='lines+markers', yaxis='y2',
                           line=dict(color='#c8102e', width=2), marker=dict(size=7)))
    f.update_layout(title='逐层迭代的边际贡献（夏普上升 / 回撤下降）', barmode='group',
                    yaxis=dict(title='夏普', gridcolor='#eef1f4'),
                    yaxis2=dict(title='|最大回撤|', overlaying='y', side='right',
                                tickformat='.1%', showgrid=False, range=[0.03, 0.062]),
                    height=400, plot_bgcolor='#fff',
                    legend=dict(orientation='h', y=1.14),
                    margin=dict(l=60, r=60, t=75, b=60))
    return f
reg('layers', f_layers())

def f_lev():
    f = go.Figure()
    f.add_trace(go.Scatter(x=DATES, y=cut(CUR['leverage']), name='实际组合杠杆',
                           mode='lines', line=dict(color='#1f4e79', width=1.6)))
    f.add_hline(y=1.0, line=dict(color='#c8102e', dash='dash', width=1.2))
    f.add_annotation(x=DATES[len(DATES) // 3], y=1.06,
                     text='杠杆上限 1.0（只降不升）',
                     showarrow=False, font=dict(size=10, color='#c8102e'))
    f.update_layout(title='动态风险预算：组合杠杆序列', xaxis_title='日期',
                    yaxis_title='杠杆', height=320, plot_bgcolor='#fff',
                    margin=dict(l=60, r=25, t=55, b=55),
                    xaxis=dict(gridcolor='#eef1f4'), yaxis=dict(gridcolor='#eef1f4', range=[0, 1.1]))
    return f
reg('lev', f_lev())

def f_mlw():
    f = go.Figure()
    wc = R['ml']['weight_curve']
    for i, c in enumerate(CORE):
        f.add_trace(go.Scatter(x=DATES, y=cut(wc[c]), name=c, mode='lines',
                               stackgroup='one', line=dict(width=0.4),
                               hovertemplate='%{x}<br>%{y:.3f}<extra>' + c + '</extra>'))
    f.update_layout(title='最终组合权重（堆叠）：进化 λ=0 → 恒定为先验，ML 动态输出未被采用',
                    xaxis_title='日期', yaxis_title='权重', height=360, plot_bgcolor='#fff',
                    legend=dict(orientation='h', y=-0.28, font=dict(size=9)),
                    margin=dict(l=60, r=25, t=55, b=100),
                    yaxis=dict(range=[0, 1], gridcolor='#eef1f4'))
    return f
reg('mlw', f_mlw())

def f_sub():
    sub = VAL['subsample']
    ks = list(sub.keys())
    f = go.Figure()
    f.add_trace(go.Bar(name='B1 公平基线', x=ks,
                       y=[sub[k]['B1']['sharpe'] for k in ks], marker_color='#9db7d1'))
    f.add_trace(go.Bar(name='V2-D 完整', x=ks,
                       y=[sub[k]['V2-D']['sharpe'] for k in ks], marker_color='#1f4e79'))
    f.update_layout(title='分年度/半年度子样本夏普（稳定性检验）', barmode='group',
                    yaxis_title='夏普', height=350, plot_bgcolor='#fff',
                    legend=dict(orientation='h', y=1.14),
                    margin=dict(l=60, r=25, t=70, b=55), yaxis=dict(gridcolor='#eef1f4'))
    return f
reg('sub', f_sub())


# ---------------------------------------------------------------- 表
MAIN_ROWS = []
for k in KEYS:
    v = V[k]; i, o = v['metrics_in'], v['metrics_oos']
    MAIN_ROWS.append([
        f'<b>{k}</b><br><span class="sub">{v["note"]}</span>',
        cspan(i['ann_ret']), cspan(i['sharpe'], kind='num'), cspan(i['mdd']),
        cspan(o['ann_ret']), cspan(o['sharpe'], kind='num'), cspan(o['mdd']),
        cspan(o['calmar'], f"{o['calmar']:.2f}" if o['calmar'] is not None else '—', kind='num'),
        f"{o['max_recovery_days']}",
    ])

LAYER_ROWS = [
    ['1 数据层', '结算价口径修正 + 换月跳空清洗（复用 V1）',
     '<span class="tag r">未达成</span> 逐合约复权行情、分钟/Tick、仓单与期限结构',
     '公开接口不提供；强行合成等于编造数据，列为前置待办'],
    ['2 执行层', '分品种流动性成本、非线性冲击、单日换手上限、保证金敞口、振幅滑点',
     '<span class="tag g">完成</span> 成本校准至 3.00%/年（V1 为 2.61%）',
     '过度苛刻的参数会吃掉全部改进，已实测校准'],
    ['3 策略层', '协整价差（4 对通过）、动态对冲比、变色止损、受控反转、风格中性代理',
     '<span class="tag g">完成</span> 并按样本内夏普筛选策略池，剔除 3 项',
     '失效项不硬塞进组合，保留诊断记录'],
    ['4 组合层', '双路径 ML + 差分进化调参 + 动态风险预算 + 回撤刹车（只降不升）',
     '<span class="tag g">完成</span> 8 参数、maximin 3 折、正则、收紧搜索空间',
     '第一版无约束进化严重过拟合，已重构'],
    ['5 验证层', 'block bootstrap、参数敏感性网格、分年度子样本、ML 增量诊断',
     '<span class="tag g">完成</span> 样本内外 gap 为负（−0.39）',
     '样本外仅 334 天，置信区间仍宽'],
]

PARAM_ROWS = [[k, num(v, 4), {'temperature': 'softmax 温度，越大越平滑',
                             'lambda_ml': 'ML 收益微调权重占比',
                             'w_cap': '单策略权重上限',
                             'target_vol': '组合目标年化波动',
                             'vol_window': '波动率目标窗口（日）',
                             'dd_threshold': '回撤刹车触发线',
                             'dd_floor': '最低杠杆',
                             'mu_ml': 'ML 预测波动在杠杆中的占比'}[k]] for k, v in BP.items()]

SCREEN_ROWS = [[k, cspan(v, kind='num'), '样本内夏普低于 −0.50 → 不进入组合池']
               for k, v in R['screened_out'].items()]

COIN_ROWS = []
for s in R['cointegration']['all']:
    ok = [s['a'], s['b']] in R['cointegration']['passed']
    COIN_ROWS.append([f"{s['a']} / {s['b']}", num(s['pvalue'], 4),
                      '<span class="tag g">通过</span>' if ok else '<span class="tag">未通过</span>'])

BOOT = VAL['bootstrap']
BOOT_ROWS = [[k, f"[{num(v['sharpe_ci'][0],2)}, {num(v['sharpe_ci'][1],2)}]",
              f"[{pct(v['mdd_ci'][0])}, {pct(v['mdd_ci'][1])}]"] for k, v in BOOT.items()]

SENS = VAL['sensitivity']
SENS_ROWS = []
for pname in BP.keys():
    row = [pname]
    for f in [0.7, 1.0, 1.3]:
        x = next(s for s in SENS if s['param'] == pname and s['factor'] == f)
        row.append(f"{num(x['sharpe'],2)} / {pct(x['mdd'])}")
    SENS_ROWS.append(row)

SUB = VAL['subsample']
SUB_ROWS = [[k, cspan(SUB[k]['B1']['sharpe'], kind='num'), cspan(SUB[k]['V2-D']['sharpe'], kind='num'),
             pct(SUB[k]['B1']['mdd']), pct(SUB[k]['V2-D']['mdd'])] for k in SUB.keys()]

STRAT_ROWS = []
for k, v in R['strategy_meta'].items():
    i, o = v['metrics_in'], v['metrics_oos']
    tag = '<span class="tag r">已剔除</span>' if k in R['screened_out'] else '<span class="tag g">保留</span>'
    STRAT_ROWS.append([f'{k} {tag}<br><span class="sub">{v["layer"]}</span>',
                       f'{v["cost_bps"]:.0f}', num(v['scale']),
                       cspan(i['sharpe'], kind='num'), cspan(i['mdd']),
                       cspan(o['sharpe'], kind='num'), cspan(o['mdd'])])


CSS = """
:root{--ink:#1b2430;--muted:#6b7684;--line:#e3e8ee;--bg:#f6f8fa;--brand:#1f4e79}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Segoe UI",sans-serif;
line-height:1.75;font-size:15px}
.wrap{max-width:1180px;margin:0 auto;padding:0 28px 80px}
header{background:linear-gradient(135deg,#123047,#1f4e79 55%,#2a9d8f);color:#fff;padding:48px 0 40px;margin-bottom:32px}
header .wrap{padding-bottom:0}h1{font-size:31px;margin:0 0 10px;font-weight:700}
.sub-title{font-size:16px;opacity:.92;margin:0}
.meta-line{margin-top:22px;font-size:13px;opacity:.85;border-top:1px solid rgba(255,255,255,.25);padding-top:14px}
section{background:#fff;border:1px solid var(--line);border-radius:10px;padding:28px 32px;margin-bottom:24px;box-shadow:0 1px 3px rgba(20,40,70,.04)}
h2{font-size:21px;margin:0 0 18px;padding-left:14px;border-left:4px solid var(--brand);line-height:1.3}
h3{font-size:16px;margin:26px 0 10px;color:var(--brand)}p{margin:10px 0}.lead{font-size:16px;color:#33404f}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:18px 0}
.card{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:16px 18px}
.card .k{font-size:12px;color:var(--muted)}.card .v{font-size:24px;font-weight:700;margin:4px 0;color:var(--brand)}
.card .d{font-size:12px;color:var(--muted)}
.callout{background:#f0f6fc;border-left:4px solid var(--brand);padding:14px 18px;border-radius:0 8px 8px 0;margin:16px 0}
.warn{background:#fff7ed;border-left-color:#e07b39}.danger{background:#fef2f2;border-left-color:#c8102e}
.ok{background:#f0fdf4;border-left-color:#0f9d58}
.dt{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
.dt th{background:#eef3f8;color:#33404f;text-align:left;padding:10px;border-bottom:2px solid #d7e0ea;font-weight:600;white-space:nowrap}
.dt td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.dt tbody tr:nth-child(even){background:#fafbfd}.dt tbody tr:hover{background:#f0f6fc}
.pos{color:#c8102e;font-weight:600}.neg{color:#0f9d58;font-weight:600}.flat{color:#6b7684}
.sub{font-size:11px;color:var(--muted);font-weight:400}.scroll{overflow-x:auto}.plot{margin:14px 0 6px}
.tag{display:inline-block;background:#eef3f8;color:var(--brand);border-radius:4px;padding:1px 6px;font-size:11px}
.tag.r{background:#fdecec;color:#c8102e}.tag.g{background:#eafaf1;color:#0f9d58}
ul,ol{padding-left:22px}li{margin:6px 0}code{background:#eef3f8;padding:2px 6px;border-radius:4px;font-size:12px}
footer{text-align:center;color:var(--muted);font-size:12px;padding:20px 0}
"""

def sec(n, t, b): return f'<section><h2>{n}　{t}</h2>{b}</section>'

body = []
body.append(sec('摘要', 'V2 改进结果', f'''
<p class="lead">按 V1 报告第七节的五层迭代顺序完成改进，新增机器学习 walk-forward 训练与差分进化调参。
所有对比统一到<b>同一执行成本口径</b>：样本内 {M['in'][0]} ~ {M['in'][1]} 用于策略标定、策略池筛选、
ML 训练与进化调参；样本外 {M['oos'][0]} ~ {M['oos'][1]}（{334} 天）不参与任何拟合、筛选与参数选择。</p>
<div class="grid">
 <div class="card"><div class="k">样本外夏普（公平基线 → V2-D）</div><div class="v">0.85 → 0.96</div><div class="d">+13%，V1 旧口径为 1.00</div></div>
 <div class="card"><div class="k">样本外最大回撤</div><div class="v">−5.08% → −4.26%</div><div class="d">收窄 16%，Calmar 1.46 → 1.92</div></div>
 <div class="card"><div class="k">样本内外夏普差</div><div class="v">−0.39</div><div class="d">样本外反而更好，未见过拟合</div></div>
 <div class="card"><div class="k">进化给 ML 的权重</div><div class="v">0.0</div><div class="d">数据自动关闭收益预测型 ML</div></div>
</div>
<div class="callout ok"><b>目标达成（相对公平基线 B1）。</b>在同一成本口径下，
样本外夏普从 <b>0.85 提升到 0.96</b>，最大回撤从 <b>−5.08% 收窄到 −4.26%</b>，
Calmar 从 1.464 升到 <b>1.923</b>，回撤修复天数也下降。
主要贡献来自<b>策略层的低回撤趋势替换 + 策略池筛选</b>（夏普 0.35→0.58 样本内、0.85→0.94 样本外），
以及<b>组合层的动态风险预算与进化调参</b>（回撤再降、Calmar 再升）。</div>
<div class="callout"><b>与 V1 原报告 P4 的关系要说清楚。</b>B0（V1 旧成本口径）样本外夏普 1.00 / 回撤 −4.76%；
V2-D 为 0.96 / −4.26%。夏普略低是因为 V2 采用了更真实的分品种流动性成本（3.00%/年 vs 2.61%），
同一口径下重算的 B1 只有 0.85。<b>回撤更低、Calmar 更高（1.923 vs 1.914），且成本假设更保守</b>。</div>
<div class="callout warn"><b>最重要的研究发现：进化调参自动把 ML 关掉了。</b>
加了抗过拟合约束后，差分进化把「ML 收益微调」权重 λ 优化为 <b>0.0</b>、
「ML 预测波动杠杆」占比 μ 优化为 <b>0.013</b>（几乎为零）。
V2-D 与「完全无 ML」的 V2-E 样本外夏普仅差 0.001。
这不是 ML 没做，而是<b>验证框架判定：在 935 天样本 + 强先验下，ML 提供的增量不足以覆盖其方差</b>。
对比第一版无约束进化（λ=0.94、样本外夏普 −0.59），这恰恰证明了「训练—验证—选择」闭环的价值。</div>
'''))

body.append(sec('一', '五层迭代：做了什么、没做什么', '''
<div class="scroll">''' + table(['迭代层', '本次实现', '状态', '说明'], LAYER_ROWS) + '''</div>
<div class="callout danger"><b>数据层如实声明未完成的部分。</b>报告建议的「逐合约复权行情、分钟/Tick 数据、
仓单与期限结构」在公开接口下不可得。用合成数据替代等于编造，因此本轮<b>没有做</b>，
而是明确列为补第二层（真实中高频 CTA）的前置条件。这也是 ML 在本样本上难有增量的根本原因之一。</div>
'''))

body.append(sec('二', '机器学习与进化调参的设计', f'''
<h3>2.1 两条 ML 路径</h3>
<div class="scroll">''' + table(['路径', '任务', '特征 / 标签', '模型', '用途'], [
    ['<b>路径 1 · 收益微调</b>', '预测各策略未来 20 日的相对吸引力',
     f"{M['n_features']} 维面板特征（策略动量/波动/回撤/相关性/one-hot + 13 维市场状态)；标签=未来风险调整收益的<b>横截面百分位排名</b>",
     'Ridge(α=30) + 随机森林(depth≤4, leaf≥50) 各 50%',
     'softmax 转成动态权重，与先验权重小幅混合'],
    ['<b>路径 2 · 风险预测</b>', '预测未来 20 日组合实现波动与高回撤概率',
     f"{len(R['ml']['risk_features'])} 维日频状态特征；标签=未来波动 / 是否发生 &gt;2.5% 回撤",
     'Ridge + 逻辑回归，均为 walk-forward',
     '用预测波动替代部分滚动波动来定杠杆'],
]) + '''</div>
<p>标签选「横截面排名」而非绝对收益，是为了让模型只学相对优劣、不受市场整体涨跌干扰；
波动率的持续性远高于收益，所以路径 2 比路径 1 更可学习。实测路径 2 的预测波动与 60 日已实现波动
样本外相关性仅 0.158，高回撤概率样本外均值 0.106 —— 增量天然有限。</p>
<h3>2.2 严格的前向隔离</h3>
<ul>
<li><b>purged walk-forward</b>：训练样本最后一个日期必须 ≤ 预测起点 − {M['horizon']} 日，
保证标签在预测时点之前完全实现；共 <b>{M['n_folds']} 个 fold</b>，每 10 日重训一次，
每个 fold 内独立 fit（含标准化器）。</li>
<li><b>样本外规则</b>：ML 模型按 walk-forward 规则在样本外期间持续用「截至当日的历史数据」在线更新
（标准在线学习，不使用未来信息）；但<b>策略池、先验权重、超参在样本外全程冻结</b>。</li>
<li><b>进化只在样本内</b>：适应度取样本内 3 个子区间中<b>最差</b>的夏普（maximin）为主，
加平均夏普、回撤惩罚、换手惩罚与正则；最优个体冻结后样本外只做一次前向推断。</li>
</ul>
<h3>2.3 进化调参的 8 个参数与最优解</h3>
<div class="scroll">''' + table(['参数', '最优值', '含义'], PARAM_ROWS) + '''</div>
<div class="callout"><b>抗过拟合三重约束</b>（第一版正是在这里翻车）：
① 收紧搜索空间（λ ≤ 0.35、温度 ≥ 0.25、杠杆只降不升）；
② maximin 三折适应度，避免只拟合某一段行情；
③ 显式正则惩罚过高的 ML 占比与过尖的 softmax。
进化共评估 {nfev} 次，最终 λ=0、μ=0.013，是数据给出的选择而非人为设定。</div>
'''.replace('{nfev}', str(R['evolution']['nfev']))))

body.append(sec('三', '工程复盘：第一版为什么失败', '''
<p>第一次跑 V2 时样本外夏普是 <b>−0.59</b>。逐层归因后定位到两个<b>实现问题</b>，而不是"ML 不行"：</p>
<div class="scroll">''' + table(['问题', '证据', '修正'], [
    ['<b>成本模型参数未校准</b>',
     '冲击系数 0.0008 + 流动性敏感度 0.6，把年化成本从 V1 的 2.61% 抬到 4.74%；同名单策略收益相关性 1.000、纯平移约 −3%/年，光是 V1 权重重算就从夏普 1.00 掉到 0.62',
     '冲击系数降至 0.0001、敏感度降至 0.3，校准回 3.00%/年（仍高于 V1，保留分品种差异）'],
    ['<b>优化器过拟合样本内</b>',
     '无约束差分进化把 λ 推到 0.944、温度压到 0.068，等于让小样本模型全权押注单一策略；样本内/外夏普差 0.851，样本外 −0.59',
     '收紧搜索空间 + maximin 三折 + 正则 + 杠杆只降不升；修正后样本内外差 −0.39，样本外 0.96'],
    ['<b>策略池稀释</b>',
     '把 11 个策略（含 3 个样本内夏普 &lt; −0.5 的失效项）全塞进池子，先验权重给了负期望策略过多份额',
     '按样本内夏普 &lt; −0.50 剔除，保留 8 个，并锚定 V1 三层框架权重'],
]) + '''</div>
<p class="sub">以上为工程过程记录，第一版参数配置已被重构版本替换，数值取自当时的运行日志；
当前报告所有结果表中的数字均来自本次可复现的 <code>run_v2.py</code>。</p>
'''))

body.append(sec('四', '策略层改进与筛选', f'''
<h3>4.1 协整检验（训练期检验、样本外冻结配对）</h3>
<p>对 10 组产业链品种对做 Engle-Granger 检验，p &lt; 0.10 的 <b>{len(R['cointegration']['passed'])} 对</b>通过：
螺纹/热卷、豆粕/菜粕、豆油/棕榈油、沪铜/沪铝。价差策略改用滚动 OLS 动态对冲比与滞回开平仓，
但样本内夏普仍为 −1.49，<b>未通过组合池筛选</b>——如实记录，不推断价差策略本身无效。</p>
<div class="scroll">''' + table(['品种对', 'p 值', '结果'], COIN_ROWS) + '''</div>
<h3>4.2 策略池筛选与全策略表现</h3>
<div class="scroll">''' + table(['策略 / 层级', '成本 bp', '杠杆', '样本内 夏普', '样本内 回撤',
                                '样本外 夏普', '样本外 回撤'], STRAT_ROWS) + '''</div>
<div class="scroll">''' + table(['被剔除策略', '样本内夏普', '规则'], SCREEN_ROWS) + '''</div>
<p><b>变色止损是有效的</b>：<code>trend_fast_v2</code>（快窗口 + 反向波动超限即平仓）
样本内夏普 0.60、回撤 −6.5%，明显优于原版 <code>trend_raw</code> 的 −12.8% 回撤，
是组合回撤下降的主要来源。</p>
'''))

body.append(sec('五', '组合回测结果', '''
<div class="scroll">''' + table(
    ['变体 / 说明', '样本内 年化', '样本内 夏普', '样本内 回撤',
     '样本外 年化', '样本外 夏普', '样本外 回撤', '样本外 Calmar', '最长修复(日)'], MAIN_ROWS) + '''</div>
<div class="plot">''' + _plots['nav'] + '''</div>
<div class="plot">''' + _plots['dd'] + '''</div>
<div class="plot">''' + _plots['layers'] + '''</div>
<div class="plot">''' + _plots['lev'] + '''</div>
<div class="callout ok"><b>推荐 V2-D（完整版）</b>：样本外夏普 0.96、回撤 −4.26%、Calmar 1.923，
样本内夏普 0.57，两段都稳定。<b>V2-E（完全无 ML）</b>与其几乎一致，
说明在本数据条件下可以用更简单的实现达到同样效果 —— 这也是工程上更值得部署的版本。</div>
'''))

body.append(sec('六', '稳健性验证', f'''
<h3>6.1 Bootstrap 置信区间（移动块，1200 次）</h3>
<div class="scroll">''' + table(['序列', '夏普 95% CI', '最大回撤 95% CI'], BOOT_ROWS) + '''</div>
<p class="sub">样本外仅 334 天，区间很宽（V2-D 夏普 CI [−0.43, 3.03]）。
<b>不能据此宣称夏普显著优于基线</b>，只能说方向一致、无过拟合迹象。</p>
<h3>6.2 参数敏感性（每个超参 ±30%，单元格为「样本外夏普 / 回撤」）</h3>
<div class="scroll">''' + table(['参数', '×0.7', '×1.0（最优）', '×1.3'], SENS_ROWS) + '''</div>
<h3>6.3 分年度 / 半年度子样本</h3>
<div class="plot">''' + _plots['sub'] + '''</div>
<div class="scroll">''' + table(['区间', 'B1 夏普', 'V2-D 夏普', 'B1 回撤', 'V2-D 回撤'], SUB_ROWS) + '''</div>
<h3>6.4 ML 增量诊断</h3>
<div class="scroll">''' + table(['指标', '数值'], [
    ['B0 V1 旧口径样本外夏普', num(MLG['oos_sharpe_b0'], 3)],
    ['B1 公平基线样本外夏普', num(MLG['oos_sharpe_b1'], 3)],
    ['V2-D 完整样本外夏普', cspan(MLG['oos_sharpe_v2d'], kind='num')],
    ['V2-E 完全无 ML 样本外夏普', cspan(MLG['oos_sharpe_no_ml'], kind='num')],
    ['样本内 − 样本外 夏普差', cspan(MLG['in_vs_oos_gap'], kind='num')],
    ['样本外回撤 B1 → V2-D', f"{pct(MLG['oos_mdd_b1'])} → {pct(MLG['oos_mdd_v2d'])}"],
]) + '''</div>
<div class="plot">''' + _plots['mlw'] + '''</div>
<p class="sub">图中是最终进入组合的权重。由于进化把 λ 定为 0，
ML 路径 1 的动态输出<b>完全未被采用</b>，权重恒定为先验配置；
模型确实完成了 80 个 fold 的训练与前向推断，但验证框架判定其不值得使用。</p>
'''))

body.append(sec('七', '结论、局限与下一步', '''
<div class="callout"><b>三句话结论</b><br>
① 五层迭代里，真正带来改进的是<b>策略层</b>（低回撤趋势 + 策略池筛选）与<b>组合层风控/进化</b>，
不是 ML 收益预测；<br>
② <b>进化调参的价值不在"把 ML 用起来"，而在"验证后决定不用"</b>——
无约束进化会押注单一策略并崩盘，带约束的进化自动收敛到稳健解；<br>
③ 若目标是纯工程落地，<b>V2-E（无 ML）与 V2-D 等效且更简单</b>，应优先部署简单版本。</div>
<div class="callout warn"><b>局限</b></div>
<ul>
<li>逐合约复权 / 分钟 Tick / 期限结构仍不可得，第二层「真实中高频 CTA」本轮<b>仍未验证</b>。</li>
<li>样本外仅 334 天，bootstrap 区间宽，统计显著性不足；参数敏感性与子样本只作方向性检查。</li>
<li>ML 标签重叠（20 日 forward）、fold 间存在序列相关；虽做了 purge，仍可能高估模型信息量。</li>
<li>成本仍是线性 + 1.5 次冲击的简化模型，未含保证金追缴、涨跌停无法成交、策略容量约束。</li>
<li>策略池筛选、先验权重、进化适应度都用了样本内数据；样本外虽隔离，但整个研究流程只做了一次，存在「研究自由度」风险。</li>
<li>宏观配置层仍用 HS300 / 十年国债期货 / 商品等权指数代理，未含外汇、海外资产与真实债券票息。</li>
</ul>
<h3>下一步（按优先级）</h3>
<ol>
<li><b>补数据</b>：逐合约复权 + 分钟/Tick，这是让 ML 与第二层真正有意义的唯一前提。</li>
<li><b>换 ML 任务</b>：从预测收益转向预测「策略失效概率 / 协方差矩阵」，用更高频标签降低噪声。</li>
<li><b>多重验证</b>：rolling walk-forward 回测（多个不重叠样本外窗口）、跨品种/跨市场外盘验证。</li>
<li><b>容量与执行</b>：分品种 AUM 上限、盘口冲击模型、真实保证金与交割月约束。</li>
</ol>
'''))

body.append(sec('附录', '文件与复现', '''
<div class="scroll">''' + table(['文件', '作用'], [
    ['<code>cta_lab/execution.py</code>', 'V2 执行层：流动性分级、成本、保证金、振幅'],
    ['<code>cta_lab/engine.py</code>', '新增 <code>backtest_exec</code> 与 <code>apply_risk_overlay</code>（V1 函数保留）'],
    ['<code>cta_lab/strategies.py</code>', '新增协整价差、变色止损、受控反转、风格中性（V1 策略保留）'],
    ['<code>cta_lab/ml_portfolio.py</code>', 'ML 面板/双路径 walk-forward、动态权重、差分进化'],
    ['<code>cta_lab/run_v2.py</code>', 'V2 主入口：标定 → 筛选 → 训练 → 进化 → 回测 → 验证'],
    ['<code>cta_lab/report_v2.py</code>', '生成本报告'],
    ['<code>cta_lab/output/results_v2.json</code>', '全部指标、参数、诊断、曲线'],
    ['<code>cta_lab/output/v2_variant_returns.csv</code>', '7 个变体日收益'],
    ['<code>cta_lab/output/v2_strategy_returns.csv</code>', '11 个策略日收益'],
]) + '''</div>
<p>复现：<code>cd /workspace/cta_lab &amp;&amp; python3.11 run_v2.py &amp;&amp; python3.11 report_v2.py</code>
（进化随机种子固定为 42，结果可复现）</p>
<p class="sub">本报告为量化工程研究记录，不构成投资建议。V1 报告见 <code>cta_research_report.html</code>。</p>
'''))

html = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>量化 CTA V2 改进研究报告 · ML 训练调参进化</title><style>{CSS}</style>
<script type="text/javascript">{get_plotlyjs()}</script></head><body>
<header><div class="wrap"><h1>量化 CTA 开发工程 · V2 改进研究报告</h1>
<p class="sub-title">五层迭代 + 机器学习 walk-forward 训练 + 差分进化调参：压回撤、提夏普</p>
<div class="meta-line">评估区间 {M['in'][0]} ~ {M['oos'][1]}　·　样本外 {M['oos'][0]} ~ {M['oos'][1]}　·　
{M['n_pool']} 个候选策略 / {M['n_core']} 个入池　·　{M['n_folds']} 个 ML folds　·　{M['n_features']} 维特征</div>
</div></header><div class="wrap">{''.join(body)}
<footer>量化 CTA V2 工程研究 · 研究回测，不构成投资建议</footer></div></body></html>'''

with open(OUTHTML, 'w', encoding='utf-8') as f:
    f.write(html)
print('V2 报告已生成:', OUTHTML, round(os.path.getsize(OUTHTML) / 1024 / 1024, 2), 'MB')
