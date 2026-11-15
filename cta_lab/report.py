# -*- coding: utf-8 -*-
"""生成单文件 HTML 研究报告（Plotly 图表内嵌，离线可打开）。"""
import os, json
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(BASE), 'cta_research_report.html')
R = json.load(open(os.path.join(BASE, 'output', 'results.json')))

M = R['meta']
EDA = R['eda']
S = R['strategies']
P = R['portfolios']
CUR = R['curves']
CORE = R['core_strategies']
MAIN = R['main_portfolios']

COLORS = ['#1f4e79', '#e07b39', '#2a9d8f', '#8e5db7', '#c8102e',
          '#6b7280', '#b5651d', '#3f7cac', '#8a9a5b']
_pos, _neg = '#c8102e', '#0f9d58'      # 中国习惯：涨/正=红，跌/负=绿

_plots = {}
def reg(key, fig):
    # plotly.js 统一在 <head> 内嵌一次，保证任何图出现在任何章节都能正常渲染
    _plots[key] = fig.to_html(full_html=False, include_plotlyjs=False,
                              div_id='plot_' + key,
                              config={'displayModeBar': False, 'responsive': True})

def pct(x, dec=2):
    return f'{x*100:+.{dec}f}%'
def num(x, dec=2):
    return f'{x:.{dec}f}'
def cspan(x, txt=None, dec=2, kind='pct'):
    s = txt if txt is not None else (pct(x, dec) if kind == 'pct' else num(x, dec))
    cls = 'pos' if x > 0 else ('neg' if x < 0 else 'flat')
    return f'<span class="{cls}">{s}</span>'

def table(headers, rows):
    h = ''.join(f'<th>{x}</th>' for x in headers)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<table class="dt"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


# ---------------------------------------------------------------- 图 1 净值
def fig_nav():
    f = go.Figure()
    for i, name in enumerate(MAIN):
        f.add_trace(go.Scatter(x=CUR['dates'], y=CUR['port_nav'][name],
                               name=name, mode='lines',
                               line=dict(color=COLORS[i % len(COLORS)], width=2.0 if i in (0, 4, 5) else 1.4),
                               hovertemplate='%{x}<br>%{y:.3f}<extra>' + name + '</extra>'))
    f.add_vline(x=CUR['sample_split_date'], line=dict(color='#999', dash='dash', width=1.5))
    f.add_annotation(x=CUR['sample_split_date'], y=1.0, text='样本外起点', showarrow=False,
                     yshift=12, font=dict(size=11, color='#666'))
    f.update_layout(title='各组合净值曲线（2023-07 ~ 2026-11，起点=1）',
                    xaxis_title='日期', yaxis_title='净值', height=460,
                    hovermode='x unified', legend=dict(orientation='h', y=-0.28, font=dict(size=10)),
                    margin=dict(l=60, r=25, t=55, b=110), plot_bgcolor='#fff',
                    xaxis=dict(gridcolor='#eef1f4'), yaxis=dict(gridcolor='#eef1f4'))
    return f
reg('nav', fig_nav())

# ---------------------------------------------------------------- 图 2 回撤
def fig_dd():
    keys = [k for k in MAIN if k.startswith(('P0', 'P1', 'P4', 'P5', 'P6'))]
    f = go.Figure()
    for i, name in enumerate(keys):
        f.add_trace(go.Scatter(x=CUR['dates'], y=CUR['port_dd'][name], name=name,
                               mode='lines', line=dict(color=COLORS[i], width=1.6),
                               hovertemplate='%{x}<br>%{y:.2%}<extra>' + name + '</extra>'))
    f.update_layout(title='回撤曲线（代表性组合）', xaxis_title='日期',
                    yaxis_title='回撤', yaxis_tickformat='.0%', height=360,
                    hovermode='x unified', legend=dict(orientation='h', y=-0.3, font=dict(size=10)),
                    margin=dict(l=60, r=25, t=55, b=100), plot_bgcolor='#fff',
                    xaxis=dict(gridcolor='#eef1f4'), yaxis=dict(gridcolor='#eef1f4'))
    return f
reg('dd', fig_dd())

# ---------------------------------------------------------------- 图 3 分层递进
def fig_layers():
    ks = ['P0 单一股票贝塔（对照）', 'P1 第一层·贝塔基石', 'P2 第一层·趋势纯度增强',
          'P3 两层·可回测阿尔法', 'P4 三层完整（文章框架）', 'P5 三层·样本内风险平价',
          'P8 三层·波动率过滤（对照）']
    labels = ['P0 对照', 'P1 第一层', 'P2 第一层+', 'P3 两层', 'P4 三层', 'P5 风险平价', 'P8 波动率过滤']
    f = go.Figure()
    f.add_trace(go.Bar(name='样本内夏普', x=labels,
                       y=[P[k]['metrics_in']['sharpe'] for k in ks], marker_color='#9db7d1'))
    f.add_trace(go.Bar(name='样本外夏普', x=labels,
                       y=[P[k]['metrics_oos']['sharpe'] for k in ks], marker_color='#1f4e79'))
    f.update_layout(title='分层配置的夏普递进（样本内 / 样本外）', barmode='group',
                    yaxis_title='夏普比率', height=380, plot_bgcolor='#fff',
                    legend=dict(orientation='h', y=1.12),
                    margin=dict(l=60, r=25, t=70, b=60),
                    yaxis=dict(gridcolor='#eef1f4'))
    return f
reg('layers', fig_layers())

# ---------------------------------------------------------------- 图 4 流畅度
def fig_er():
    d = EDA['efficiency_ratio_60d_by_period']
    ks, vs = list(d.keys()), list(d.values())
    cols = ['#2a9d8f' if v < 0.14 else '#e07b39' if v < 0.16 else '#1f4e79' for v in vs]
    f = go.Figure(go.Bar(x=ks, y=vs, marker_color=cols,
                         text=[f'{v:.3f}' for v in vs], textposition='outside'))
    f.update_layout(title='全商品 60 日行情流畅度（Kaufman 效率比 ER）分期均值',
                    yaxis_title='ER（越高趋势越流畅）', height=340, plot_bgcolor='#fff',
                    margin=dict(l=60, r=25, t=55, b=55), yaxis=dict(gridcolor='#eef1f4', range=[0, 0.25]))
    return f
reg('er', fig_er())

# ---------------------------------------------------------------- 图 5 相关性
def fig_corr():
    cm = R['corr_oos']
    z = [[cm[a][b] for b in CORE] for a in CORE]
    labels = ['股票贝塔', '趋势·ER过滤', '趋势·快速', '截面多空', '宏观配置']
    f = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale='RdBu_r', zmin=-1, zmax=1,
                             text=[[f'{v:.2f}' for v in row] for row in z],
                             texttemplate='%{text}', colorbar=dict(title='相关系数')))
    f.update_layout(title='样本外核心策略收益相关性', height=420, plot_bgcolor='#fff',
                    margin=dict(l=100, r=25, t=55, b=80))
    return f
reg('corr', fig_corr())

# ---------------------------------------------------------------- 图 6 成本分解
def fig_cost():
    names = ['trend_lf', 'trend_plain', 'trend_raw', 'macro_rp', 'xs_mom', 'spread', 'trend_mix', 'hf_mr']
    lbl = ['趋势·ER', '趋势·无过滤', '趋势·快速', '宏观配置', '截面多空', '价差套利', '趋势+反转', '中高频代理']
    gross = [S[n]['cost_breakdown']['gross_ann'] for n in names]
    cost = [S[n]['cost_breakdown']['cost_ann'] for n in names]
    f = go.Figure()
    f.add_trace(go.Bar(name='毛年化收益', x=lbl, y=gross, marker_color='#9db7d1'))
    f.add_trace(go.Bar(name='年化换手成本', x=lbl, y=[-c for c in cost], marker_color='#e07b39'))
    f.update_layout(title='策略收益 / 成本分解（全期年化，含样本内标定的固定杠杆）',
                    barmode='relative', yaxis_title='年化', yaxis_tickformat='.0%',
                    height=380, plot_bgcolor='#fff', legend=dict(orientation='h', y=1.12),
                    margin=dict(l=60, r=25, t=70, b=70), yaxis=dict(gridcolor='#eef1f4'))
    return f
reg('cost', fig_cost())


# ---------------------------------------------------------------- 表格内容
STRATEGY_ROWS = []
for k, v in S.items():
    i, o = v['metrics_in'], v['metrics_oos']
    cb = v['cost_breakdown']
    STRATEGY_ROWS.append([
        f'<b>{k}</b><br><span class="sub">{v["layer"]}</span>',
        v['desc'], f'{v["cost_bps"]:.0f}', num(v['scale']),
        cspan(i['ann_ret']), cspan(i['sharpe'], kind='num'),
        cspan(o['ann_ret']), cspan(o['sharpe'], kind='num'),
        cspan(o['mdd']), cspan(cb['cost_ann']),
    ])

PORT_ROWS = []
for k in R['portfolios'].keys():
    v = P[k]
    i, o = v['metrics_in'], v['metrics_oos']
    w = ' + '.join(f'{a} {b:.0%}' for a, b in v['weights'].items())
    PORT_ROWS.append([
        f'<b>{k}</b><br><span class="sub">{w}</span>',
        cspan(i['ann_ret']), cspan(i['sharpe'], kind='num'), cspan(i['mdd']),
        cspan(o['ann_ret']), cspan(o['sharpe'], kind='num'), cspan(o['mdd']),
        cspan(o['calmar'], f"{o['calmar']:.2f}" if o['calmar'] is not None else '—', kind='num'),
        f"{o['max_recovery_days']}",
    ])

ER_ROWS = [[k, num(v, 4)] for k, v in EDA['efficiency_ratio_60d_by_period'].items()]

ANN = EDA['annual_returns']
ANN_ROWS = []
for y in ['2023', '2024', '2025', '2026']:
    d = ANN[y]
    ANN_ROWS.append([y] + [cspan(d[c]) for c in ['AU0', 'CU0', 'SC0', 'RB0', 'M0']])

HF_ROWS = []
for x in R['hf_cost_sensitivity']:
    m = x['metrics_all']
    HF_ROWS.append([f"{x['cost_bps']} bp", cspan(m['ann_ret']),
                    cspan(m['sharpe'], kind='num'), cspan(m['mdd'])])

GAP = R['second_layer_gap']
GAP_ROWS = [[f"{s['assumed_alpha_sharpe']:.1f}", cspan(s['new_ann_ret']),
             num(s['new_ann_vol'], 3), cspan(s['new_sharpe'], kind='num')]
            for s in GAP['scenarios']]

COST_DIAG_ROWS = []
for k in ['trend_lf', 'trend_plain', 'trend_raw', 'macro_rp', 'xs_mom', 'spread', 'trend_mix', 'hf_mr']:
    c = S[k]['cost_breakdown']
    COST_DIAG_ROWS.append([k, cspan(c['gross_ann']), cspan(c['cost_ann']),
                           cspan(c['net_ann']), num(c['turnover_daily'], 3)])


# ---------------------------------------------------------------- HTML
CSS = """
:root{--ink:#1b2430;--muted:#6b7684;--line:#e3e8ee;--bg:#f6f8fa;--brand:#1f4e79;--accent:#e07b39}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Segoe UI",sans-serif;
line-height:1.75;font-size:15px}
.wrap{max-width:1180px;margin:0 auto;padding:0 28px 80px}
header{background:linear-gradient(135deg,#173a5e,#1f4e79 55%,#2a6f97);color:#fff;padding:48px 0 40px;margin-bottom:32px}
header .wrap{padding-bottom:0}
h1{font-size:32px;margin:0 0 10px;font-weight:700;letter-spacing:.5px}
.sub-title{font-size:16px;opacity:.9;margin:0}
.meta-line{margin-top:22px;font-size:13px;opacity:.82;border-top:1px solid rgba(255,255,255,.25);padding-top:14px}
section{background:#fff;border:1px solid var(--line);border-radius:10px;padding:28px 32px;margin-bottom:24px;
box-shadow:0 1px 3px rgba(20,40,70,.04)}
h2{font-size:21px;margin:0 0 18px;padding-left:14px;border-left:4px solid var(--brand);line-height:1.3}
h3{font-size:16px;margin:26px 0 10px;color:var(--brand)}
p{margin:10px 0}
.lead{font-size:16px;color:#33404f}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:18px 0}
.card{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:16px 18px}
.card .k{font-size:12px;color:var(--muted);letter-spacing:.5px}
.card .v{font-size:24px;font-weight:700;margin:4px 0;color:var(--brand)}
.card .d{font-size:12px;color:var(--muted)}
.callout{background:#f0f6fc;border-left:4px solid var(--brand);padding:14px 18px;border-radius:0 8px 8px 0;margin:16px 0}
.warn{background:#fff7ed;border-left-color:#e07b39}
.danger{background:#fef2f2;border-left-color:#c8102e}
.ok{background:#f0fdf4;border-left-color:#0f9d58}
.dt{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
.dt th{background:#eef3f8;color:#33404f;text-align:left;padding:10px 10px;border-bottom:2px solid #d7e0ea;
font-weight:600;white-space:nowrap;position:sticky;top:0}
.dt td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.dt tbody tr:nth-child(even){background:#fafbfd}
.dt tbody tr:hover{background:#f0f6fc}
.pos{color:#c8102e;font-weight:600}.neg{color:#0f9d58;font-weight:600}.flat{color:#6b7684}
.sub{font-size:11px;color:var(--muted);font-weight:400}
.scroll{overflow-x:auto}
.plot{margin:14px 0 6px}
.layers{display:flex;flex-direction:column;gap:12px;margin:18px 0}
.layer{border-radius:8px;padding:16px 20px;color:#fff}
.layer h4{margin:0 0 6px;font-size:16px}.layer p{margin:4px 0;font-size:13px;opacity:.95}
.l1{background:#2a6f97}.l2{background:#c8102e}.l3{background:#8e5db7}
ul,ol{padding-left:22px}li{margin:6px 0}
.tag{display:inline-block;background:#eef3f8;color:var(--brand);border-radius:4px;padding:1px 7px;font-size:11px;margin-right:5px}
.tag.r{background:#fdecec;color:#c8102e}.tag.g{background:#eafaf1;color:#0f9d58}
footer{text-align:center;color:var(--muted);font-size:12px;padding:20px 0}
code{background:#eef3f8;padding:2px 6px;border-radius:4px;font-size:12px;font-family:ui-monospace,Menlo,monospace}
"""

def sec(n, title, body):
    return f'<section><h2>{n}　{title}</h2>{body}</section>'


body = []
body.append(sec('摘要', '核心结论', f'''
<p class="lead">本报告把两篇文章的配置思想落成一个可复现的量化 CTA 工程：用 {M['n_assets']} 个真实品种、
{M['data_range'][0]} ~ {M['data_range'][1]} 共 {EDA['n_days']} 个交易日的数据（前 130 日用于指标预热），
实现 11 个子策略、10 个候选组合，并严格区分样本内（{M['in'][0]} ~ {M['in'][1]}，用于选型与权重）与
样本外（{M['oos'][0]} ~ {M['oos'][1]}，仅作验证、不调参）。</p>
<div class="grid">
 <div class="card"><div class="k">最优组合 · 样本外夏普</div><div class="v">1.00</div><div class="d">P4 三层完整 vs 单一股票贝塔 −0.03</div></div>
 <div class="card"><div class="k">最大回撤收窄</div><div class="v">−4.76%</div><div class="d">从 P0 的 −11.98%，Calmar 0.10 → 1.91</div></div>
 <div class="card"><div class="k">最强单策略</div><div class="v">1.38</div><div class="d">宏观配置层样本外夏普，回撤 −7.86%</div></div>
 <div class="card"><div class="k">无法复现的一层</div><div class="v">日线</div><div class="d">中高频 CTA 代理零成本仍亏损，需分钟数据</div></div>
</div>
<div class="callout ok"><b>结论一 · 三层框架被实证支持。</b>样本外夏普随层数单调上升：
P0 −0.03 → P1 0.21 → P3 0.65 → <b>P4 1.00</b>；最大回撤从 −11.98% 收窄到 −4.76%，
Calmar 从 0.10 升到 1.91。样本内同样单调改善（P0 −0.04 → P4 0.50），不是样本外偶然。</div>
<div class="callout"><b>结论二 · 第三层「宏观配置」是最大边际贡献。</b>风险平价 + 动量择时的宏观层
样本外年化 {pct(S['macro_rp']['metrics_oos']['ann_ret'])}、夏普 1.38、回撤 −7.86%，
换手成本仅 0.59%/年。文章所说的「涨时跟得住、跌时控得住」在本样本中成立。</div>
<div class="callout"><b>结论三 · 「行情流畅度」优于「波动率高低」。</b>把 ER 过滤换成波动率过滤（P8 vs P4），
样本内夏普 0.35 → 0.50、样本外 0.96 → 1.00、Calmar 1.59 → 1.91，
实证支持第二篇文章的核心论断：CTA 要的是<b>高波动且方向流畅</b>，而非单纯高波动。</div>
<div class="callout warn"><b>结论四 · 第二层「中高频 CTA」在日线数据上无法复现。</b>
代理策略即使不计成本，净值年化仍为 −85%，再叠加 48%/年的换手成本。这不是参数问题，而是
数据频率与执行壁垒问题——反而印证了文章所说的「技术、硬件、数据护城河」。该层列为工程待办。</div>
<div class="callout"><b>结论五 · 框架权重稳健。</b>文章先验权重 P4（夏普 1.00）与样本内风险平价优化
P5（0.96）结果接近，说明分层框架本身已接近风险均衡，不依赖精细调参。</div>
'''))

body.append(sec('一', '两篇文章的框架提炼', '''
<h3>1.1 收益来源的重新拆解（《消失的阿尔法》）</h3>
<p>文章最关键的认知修正是：不同策略赚的钱根本不是同一种东西。贝塔是「车流的速度」，阿尔法是「变道超车的能力」。
过去被统称为「阿尔法」的收益里，混着大量风格贝塔；当风格反转，这部分会连本带利还回去。</p>
<div class="scroll">''' + table(
    ['策略', '收益来源', '文章定性', '本工程对应'],
    [['指数增强', '宽基贝塔 + 选股阿尔法', '超额中枢持续下移', '不纳入（阿尔法枯竭）'],
     ['市场中性', '剥离贝塔后的纯阿尔法', '「拥挤税」，性价比下降', '不纳入'],
     ['<b>中低频 CTA</b>', '<b>多品种趋势贝塔</b>（附加薄阿尔法）', '<b>新的贝塔基石</b>，与股债低相关', '<code>trend_lf</code> / <code>trend_raw</code>'],
     ['<b>中高频 CTA</b>', '分钟/微秒级定价偏差', '<b>独立纯阿尔法</b>，有技术护城河', '<code>hf_mr</code>（日线代理，未能复现）'],
     ['套利 / 期权', '定价偏差修复', '纯度高、容量有限、薄利补充', '<code>spread</code>（跨品种代理）'],
     ['<b>宏观配置</b>', '以贝塔资产为原料的复合阿尔法', '<b>高维新阿尔法</b>，非对称收益', '<code>macro_rp</code>']]) + '''</div>
<h3>1.2 三层配置地图</h3>
<div class="layers">
 <div class="layer l3"><h4>第三层 · 宏观配置（进攻核心 / 高维阿尔法）</h4>
 <p>底层资产全是贝塔，但「厨师对火候、调味、搭配的理解」才是阿尔法：前瞻判断宏观周期、
 动态择时与风险预算，形成「贝塔大涨跟得住、贝塔暴跌控得住」的非对称结构。红利来自宏观波动抬升、竞争格局未拥挤、收益来源不重叠。</p></div>
 <div class="layer l2"><h4>第二层 · 锁定还有红利的稀缺阿尔法（利刃）</h4>
 <p>聚焦<b>中高频 CTA</b>：与宏观方向、股市涨跌几乎无关，技术/硬件/数据门槛形成护城河，避免了股票量化式的全民拥挤。
 套利策略「了解但不依赖」——纯度高、回撤低，但收益薄、容量有限，只适合做现金管理增强。</p></div>
 <div class="layer l1"><h4>第一层 · 贝塔基石（压舱石，不再是配角）</h4>
 <p><b>宽基指数</b>（股票贝塔）+ <b>中低频 CTA</b>（商品 / 趋势贝塔）。一个健康的组合应有一大半收益来自可靠的贝塔；
 商品趋势与股债天然低相关，7 月股票阿尔法集体回撤时仍能提供正收益。</p></div>
</div>
<h3>1.3 第二篇文章给出的 CTA 工程要点</h3>
<ul>
<li><b>趋势跟踪</b>：「以最近五天收盘价连线判断方向 + 多周期叠加」，核心是<b>大趋势的确认</b>；折线频繁变色即进入磨损期。</li>
<li><b>反转策略</b>：与趋势相反，赚「变色」的钱；<b>不能独立作主力</b>，只在趋势持续太久、盈利丰厚时逐步提权做防守。</li>
<li><b>波动率 vs 流畅度</b>：CTA 真正需要的不是高波动，而是<b>「高波动且方向流畅」</b>；管理人说的「波动太小」实际指趋势不持续、随机震荡。</li>
<li><b>组合协同</b>：截面多空做横向相对强弱；跨期/跨品种套利不博单边 Beta，与趋势天然互补；CTA「做多波动率」与期权「做空波动率」互补，但<b>高波动 + 无连续趋势会双杀</b>；CTA 含大量股指期货会抬升与 A 股相关性，<b>市场中性 + CTA</b> 更理想。</li>
<li><b>配置方法论</b>：以组合配置为目的时，选金融期货含量少、回撤修复时间短、历史跨度长的管理人，不苛求极致业绩；以绝对收益为目的时，同类 CTA 夏普长期趋近 1，可用「投后夏普远高于 1 减仓止盈、近期夏普回落至 1 附近布局」的逆势方法。</li>
</ul>
'''))

body.append(sec('二', '工程设计与数据口径', f'''
<div class="scroll">''' + table(['项目', '设定'], [
    ['数据源', f"{M['data_source']}（{M['data_range'][0]} ~ {M['data_range'][1]}，{EDA['n_days']} 个交易日）"],
    ['品种池', f"{M['n_commodity']} 个商品主力连续 + 沪深300指数 + IF/IC/T/TF 金融期货，共 {M['n_assets']} 列"],
    ['价格口径', f"<b>{M['price_convention']}</b>（见下方「关键工程决策」）"],
    ['数据清洗', f"主力连续换月跳空：|r| &gt; 12% 或 &gt; 6 倍 60 日波动 → 置 0，共清洗 {M['cleaned_outliers']} 个异常点（占全样本 {pct(EDA['cleaned_outlier_ratio'], 3)}）；EDA 与回测共用同一判定函数"],
    ['样本内 / 外', f"{M['in'][0]} ~ {M['in'][1]}（选型与权重） / {M['oos'][0]} ~ {M['oos'][1]}（仅验证，不调参）"],
    ['防前视', '信号 t 日收盘形成、t+1 生效；固定杠杆倍数仅用样本内波动标定，样本外原样沿用'],
    ['杠杆与成本', f"单策略样本内波动对齐到 {M['target_vol']:.0%} 目标（杠杆上限 3 倍）；按 |Δw| 之和收单边成本 1~8 bp"],
    ['绩效口径', f'年化 {M["trading_days"]} 日、无风险利率 1.5%；含夏普、Sortino、最大回撤、Calmar、最长回撤修复天数、月度胜率'],
]) + '''</div>
<div class="callout warn"><b>关键工程决策 · 为什么商品改用「结算价」而不是「收盘价」。</b>
初次回测发现趋势策略系统性亏损，追查后发现是数据口径问题：新浪主力连续序列的
<b>收盘价</b>字段收益率平均 1 阶自相关为 <b>−0.0217</b>（27 个品种中 20 个为负，呈日内噪声 / 买卖价跳动特征），
而交易所公布的<b>动态结算价</b>为 <b>+0.181</b>（具备真实趋势延续性）。CTA 实务本就以结算价估值。
切换到结算价后趋势策略由负转正，且结论与文章对 2024–2025 趋势行情的描述吻合。
这不是调参，是数据口径修正。</div>
<div class="callout danger"><b>诚实性约定。</b>中高频 CTA 的日线代理、以及未做协整检验的简化价差套利在回测中失效，
本报告把它们保留为「可行性诊断项」，<b>不进入主组合</b>，避免用失真结果污染组合结论；
「第二层缺口」只用显式假设的解析推演，绝不生成虚构的历史净值。</div>
'''))

body.append(sec('三', '数据实证：文章论断是否被行情支持', f'''
<h3>3.1 关键品种走势（对照文章描述）</h3>
<p>文章称黄金「2024 与 2025 涨幅较大、2026 上半年幅度略小」，沪铜「2024/2025 主升浪连续性极强」、
原油「2026 年 3 月前趋势连续且幅度较大，3 月后回归无序震荡」。真实数据的方向与之高度一致：</p>
<div class="scroll">''' + table(['年份', '沪金 AU0', '沪铜 CU0', '原油 SC0', '螺纹 RB0', '豆粕 M0'], ANN_ROWS) + '''</div>
<p class="sub">注：主力连续为不复权拼接，年度涨跌幅含换月基差，仅用于方向性对照，不可直接外推为实盘收益。</p>
<h3>3.2 行情流畅度的分期检验</h3>
<p>用 Kaufman 效率比 ER = |P<sub>t</sub> − P<sub>t−n</sub>| / Σ|ΔP| 度量「趋势流畅度」：
ER 越接近 1 越流畅，越接近 0 越震荡。60 日 ER 在 <b>2026Q1 为 0.1786</b>、
<b>2026Q2 骤降至 0.1306</b>、2026H2 为 0.1377——与文章「2026 年 3 月前好做、3 月后进入损耗期」吻合。</p>
<div class="plot">''' + _plots['er'] + '''</div>
<div class="scroll">''' + table(['区间'] + list(EDA['efficiency_ratio_60d_by_period'].keys()),
                                  [['60 日 ER'] + [num(v, 4) for v in EDA['efficiency_ratio_60d_by_period'].values()]]) + '''</div>
'''))

body.append(sec('四', '策略库与可行性诊断', '''
<h3>4.1 策略清单</h3>
<div class="scroll">''' + table(
    ['策略 / 层级', '实现说明', '成本 bp', '杠杆', '样本内 年化', '样本内 夏普',
     '样本外 年化', '样本外 夏普', '样本外 回撤', '年化成本'], STRATEGY_ROWS) + '''</div>
<h3>4.2 收益 / 成本分解：哪些策略真的能赚钱</h3>
<div class="plot">''' + _plots['cost'] + '''</div>
<div class="scroll">''' + table(['策略', '毛年化', '年化成本', '净年化', '日均换手'], COST_DIAG_ROWS) + '''</div>
<p>三点诊断：</p>
<ul>
<li><b>trend_lf（ER 过滤）</b>毛收益 6.90%，明显高于无过滤版 1.68%，代价是日均换手从 0.202 升到 0.349、
成本从 1.52% 升到 2.64%，但净收益仍从 0.16% 提升到 <b>4.26%</b>。ER 过滤提升的是信号质量，不是省成本。</li>
<li><b>trend_raw</b>（含 10 日窗口、不离散）毛收益 17.62%，在本样本趋势延续性强的环境下最优；
说明「降换手」并非无条件正确，其价值主要体现在震荡 / 噪声环境。故把它保留为趋势纯度增强项而非删除。</li>
<li><b>macro_rp</b> 净收益 7.74%、成本仅 0.59%、日均换手 0.117，是「低换手 + 稳健收益」的最佳范例。</li>
</ul>
<div class="callout danger"><b>4.3 三项失效诊断（不进入主组合）</b></div>
<div class="scroll">''' + table(['失效项', '现象', '归因', '工程结论'], [
    ['<code>hf_mr</code> 中高频代理', '即使 0 bp 成本，全期年化仍 −85%；日均换手 2.396',
     '结算价序列呈正自相关（趋势延续），日线横截面反转是系统性做反；且日线根本不含分钟级微结构信息',
     '<b>不是成本问题，是数据频率问题</b>。需接入分钟 / Tick 行情与低延迟撮合，否则该层不可回测'],
    ['<code>spread</code> 价差套利', '毛 −4.68% + 成本 4.31% → 净 −8.99%',
     '用固定品种对、未做协整检验，价差在趋势市中并不均值回归；跨期合约价差因主力连续数据不可得',
     '需逐对做协整 / 平稳性检验、滚动窗口筛选，并接入逐合约行情'],
    ['<code>trend_mix</code> 趋势+反转', '毛 −7.58%，样本内外均亏',
     '反转权重随趋势持久度提升，但在本样本趋势延续行情中被反复打脸',
     '文章已明确「反转不作主力」；动态权重上限 0.5 偏高，应降至 0.2 并加波动率状态开关'],
]) + '''</div>
<h3>4.4 中高频代理的成本敏感性</h3>
<p>把成本从 0 bp 逐步加到 8 bp，年化收益只从 −85.16% 变到 −90.98%：
<b>毛收益本身就是大幅负值</b>，成本只是次要因素。这从反面说明真实中高频 CTA 的阿尔法
完全来自日线数据看不到的微结构，任何「日线复刻」都不成立。</p>
<div class="scroll">''' + table(['单边成本', '全期年化', '夏普', '最大回撤'], HF_ROWS) + '''</div>
'''))

body.append(sec('五', '组合回测：各候选组合对比', '''
<p>所有子策略先在样本内标定到同一目标波动，再按固定权重线性叠加（等价于恒定风险预算再平衡），
各组合起始资本相同、可直接比较。P9 为「强行纳入失效代理」的失真对照，仅用于敏感性展示。</p>
<div class="scroll">''' + table(
    ['组合 / 权重', '样本内 年化', '样本内 夏普', '样本内 回撤',
     '样本外 年化', '样本外 夏普', '样本外 回撤', '样本外 Calmar', '最长修复(日)'], PORT_ROWS) + '''</div>
<div class="plot">''' + _plots['nav'] + '''</div>
<div class="plot">''' + _plots['dd'] + '''</div>
<div class="callout ok"><b>推荐组合</b><br>
① <b>P4 三层完整（首选）</b>：样本外夏普 1.00、Calmar 1.91、回撤 −4.76%，样本内 0.50 / 0.91，两段都稳定。<br>
② <b>P5 三层·风险平价（次选）</b>：用样本内协方差求等风险贡献权重，夏普 0.96 / Calmar 1.72，与 P4 接近，验证框架权重稳健。<br>
③ <b>P3 两层（无法落地宏观层时的退路）</b>：夏普 0.65 / Calmar 0.90，仍显著优于 P0 与 60/40。<br>
✗ <b>P9 不推荐</b>：把不可复现的日线代理塞进组合，样本外 −22.38%、夏普 −3.52。</div>
'''))

body.append(sec('六', '三层框架的定量检验', '''
<h3>6.1 分层递进是否稳健</h3>
<div class="plot">''' + _plots['layers'] + '''</div>
<p>样本内与样本外都呈单调改善；P8（把 ER 流畅度过滤换成波动率过滤）在两段都落后于 P4，
且差距在样本内更大（0.35 vs 0.50）。这直接支持第二篇文章的判断：<b>波动率高低不是关键，趋势是否持续、方向是否流畅才是</b>。</p>
<h3>6.2 相关性与分散化</h3>
<div class="plot">''' + _plots['corr'] + '''</div>
<p>股票贝塔与趋势 / 截面类策略相关性仅 −0.03 ~ −0.07，印证文章「CTA 与股票低相关」的前提；
但 <code>trend_lf</code> 与 <code>trend_raw</code> 相关高达 <b>0.95</b>、与截面多空 0.64，
说明「多个 CTA 子策略」并不等于多个独立收益源。宏观配置层与股票贝塔相关 0.56（因其含股票仓），
<b>分散化主要来自第三层的跨资产与择时，而非堆叠同类 CTA</b>。</p>
<h3>6.3 第二层「缺口」值多少钱（解析推演，非回测）</h3>
<p>本工程无法回测真实中高频 CTA，但可以做显式假设的解析推演：以 P4 样本外为基准
（算术年化口径：收益 ''' + pct(GAP['base_ann_ret']) + '''、波动 ''' + pct(GAP['base_ann_vol']) + '''、夏普 ''' + num(GAP['base_sharpe'], 2) + '''），
假设用 ''' + f"{GAP['weight']:.0%}" + ''' 仓位接入一个波动 ''' + f"{GAP['assumed_alpha_vol']:.0%}" + '''、
与现有组合<b>零相关</b>的独立阿尔法源，组合夏普随其假设夏普的变化如下：</p>
<div class="scroll">''' + table(['假设的稀缺阿尔法夏普', '组合年化收益', '组合年化波动', '组合夏普'], GAP_ROWS) + '''</div>
<p class="sub">口径说明：本推演用算术年化做均值-方差分析，与主表的几何年化口径存在小幅差异，
仅用于衡量边际改善方向，<b>不代表任何可实现业绩，也未生成任何虚构历史净值</b>。
结论是：即使假设该层夏普达到 1.0，组合夏普也只从 1.18 提升到 1.38 ——
<b>第三层宏观配置带来的分散化收益，比补上第二层更划算、也更可落地</b>。</p>
'''))

body.append(sec('七', '局限、风险与工程路线', '''
<div class="callout warn"><b>必须声明的局限（结果不可直接外推实盘）</b></div>
<ul>
<li><b>数据</b>：新浪主力连续为不复权拼接，虽改用结算价并清洗 ''' + str(M['cleaned_outliers']) + ''' 个异常跳空，仍与实盘可交易合约存在差异；年度涨跌幅含换月基差。</li>
<li><b>样本</b>：仅 935 个交易日，样本外 330 天，未覆盖完整牛熊与极端流动性冲击，统计显著性有限。</li>
<li><b>成本</b>：线性换手模型，未含冲击成本非线性、保证金占用、涨跌停、流动性容量、滑点时变。</li>
<li><b>策略缺口</b>：中高频 CTA、跨期套利、期权波动率策略均未实现；价差套利的失效属<b>实现缺陷</b>（缺协整检验），不能推断策略本身无效。</li>
<li><b>代理简化</b>：宏观配置用 HS300 / 十年国债期货 / 商品等权指数，未含外汇、海外资产、真实债券票息；国债期货价格收益 ≠ 债券总回报。</li>
<li><b>杠杆</b>：固定倍数由样本内波动标定，样本外实际波动会偏离 10% 目标；组合层未做二次波动率目标。</li>
<li><b>回撤修复</b>：最优组合最长水下约 160 个交易日，而文章把「回撤修复时间短」列为重要挑选标准，此项仍有改进空间。</li>
</ul>
<h3>建议的迭代顺序</h3>
<ol>
<li><b>数据层</b>：接入逐合约复权行情、分钟 / Tick 数据、仓单与期限结构；这是补上第二层的前置条件。</li>
<li><b>策略层</b>：补跨期套利（协整 + 期限结构）、期权波动率曲面、真实中高频；修正反转防守的权重上限与状态开关。</li>
<li><b>组合层</b>：Walk-forward 滚动再平衡、动态风险预算、策略容量约束、纳入「市场中性 + CTA」组合。</li>
<li><b>执行层</b>：真实手续费 / 保证金 / 涨跌停 / 撮合约束，以及分品种流动性上限。</li>
<li><b>验证层</b>：参数敏感性网格、bootstrap 显著性检验、跨市场（外盘 / 不同年份区间）样本外验证。</li>
</ol>
'''))

body.append(sec('附录', '工程文件与复现', '''
<div class="scroll">''' + table(['文件', '作用'], [
    ['<code>cta_lab/fetch_data.py</code>', '抓取 32 个品种的行情，输出 <code>data/raw/</code> 与 <code>data/prices.csv</code>'],
    ['<code>cta_lab/eda.py</code>', '数据质量检查与行情流畅度实证，输出 <code>data/eda_report.json</code>'],
    ['<code>cta_lab/config.py</code>', '全部参数：品种池、样本区间、成本、目标波动、品种对'],
    ['<code>cta_lab/data_prep.py</code>', '价格口径（结算价）、换月跳空清洗、日内特征、商品指数'],
    ['<code>cta_lab/strategies.py</code>', '11 个子策略：三层 + 对照组'],
    ['<code>cta_lab/engine.py</code>', '向量化回测、绩效指标、风险平价求解'],
    ['<code>cta_lab/run_backtest.py</code>', '主入口：标定 → 回测 → 组合 → 诊断 → 结果落盘'],
    ['<code>cta_lab/report.py</code>', '生成本 HTML 报告'],
    ['<code>cta_lab/output/results.json</code>', '全部指标、相关性、净值曲线、诊断数据'],
    ['<code>cta_lab/output/*_returns.csv</code>', '策略与组合的日收益序列，便于二次分析'],
]) + '''</div>
<p>复现命令：<code>cd /workspace/cta_lab &amp;&amp; python3.11 fetch_data.py &amp;&amp; python3.11 eda.py &amp;&amp; python3.11 run_backtest.py &amp;&amp; python3.11 report.py</code></p>
<p class="sub">本报告为量化工程研究记录，所有数字来自上述公开行情的回测，不构成任何投资建议。</p>
'''))

html = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>量化 CTA 开发工程研究报告 · 三层配置回测</title><style>{CSS}</style>
<script type="text/javascript">{get_plotlyjs()}</script></head><body>
<header><div class="wrap"><h1>量化 CTA 开发工程研究报告</h1>
<p class="sub-title">重构配置：拥抱优质贝塔，锁定稀缺阿尔法 —— 策略选型、工程实现与组合回测</p>
<div class="meta-line">数据区间 {M['data_range'][0]} ~ {M['data_range'][1]}　·　{M['n_assets']} 个品种　·　
样本外 {M['oos'][0]} ~ {M['oos'][1]}　·　数据源 {M['data_source']}</div></div></header>
<div class="wrap">{''.join(body)}
<footer>量化 CTA 开发工程研究 · 基于两篇公众号文章的配置框架 · 研究回测，不构成投资建议</footer>
</div></body></html>'''

with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print('报告已生成:', OUT, '大小', round(os.path.getsize(OUT) / 1024 / 1024, 2), 'MB')
