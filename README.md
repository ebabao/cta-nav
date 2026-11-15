# 三层配置 · 量化 CTA 策略看板（自动同步版）

参考 [all-weather-nav](https://ebabao.github.io/all-weather-nav/) 的「静态页 + 同源 JSON 接口」架构，
把本仓库的量化 CTA 策略模型（V1 三层配置 + V2 机器学习/进化改进）接入一个**每日自动同步、自动部署**的看板。

> 本轮交付已完成工程、接口、同步脚本、工作流与本地预览，**按你的要求没有执行任何 git 提交或推送**。
> 上线步骤见文末「五、首次上线（待你确认后执行）」。

---

## 一、架构：模型接口与页面数据始终同源

```
cta_lab/  （策略模型工程，复用既有脚本，不重写）
  data/prices.csv  ← fetch_data.py（32 品种行情快照）
  output/results.json        ← run_backtest.py（V1 三层配置）
  output/results_v2.json     ← run_v2.py（V2 ML + 进化 + 动态风控）
                │
                │  scripts/feed.py  唯一的转换器
                ▼
data/  （对外发布的同源接口，静态页直接 fetch）
  meta.json / nav.json / signal.json / weights.json
  strategies.json / holdings.json / validation.json / sync_audit.json / rebalances.csv
                │
                │  fetch(..., {cache:"no-store"})
                ▼
index.html （静态看板，零外部依赖，页面无需重建即可每日更新）
```

**关键点**：`index.html` 不含任何硬编码数据，也不引用外部 CDN；所有数字来自 `data/` 接口。
模型跑完写入 `data/` 后，页面刷新即生效（写入即线上生效），因此「策略模型接口」与「看板页面数据」永远同源。

---

## 二、目录说明

| 路径 | 作用 |
|---|---|
| `index.html` | 看板静态页（9 章节、原生 SVG 图表、周期/区间切换、策略切换、导航高亮） |
| `data/*.json` | 同源策略数据接口（由 `feed.py` 生成） |
| `scripts/feed.py` | 模型输出 → 接口 的转换器；净值/月度/分年/回撤/风险贡献全部真实计算 |
| `scripts/holdings.py` | 复用 `cta_lab` 现有策略函数重建品种级权重矩阵 → 底层资产与调仓信号接口（不重写策略、不臆造） |
| `scripts/sync_daily.py` | 每日同步 7 步编排（抓数/校验/回放/接口/幂等/审计/回退） |
| `scripts/check_freshness.py` | 停更巡检与一行中文汇报，可下发 `workflow_dispatch` 重启 |
| `cta_lab/` | 完整策略模型工程（代码 + 行情快照 + 结果快照），保证仓库自包含、可离线回退 |
| `.github/workflows/` | `daily_sync.yml` / `watchdog.yml` / `deploy_pages.yml` |
| `requirements.txt` | 同步链路依赖 |

---

## 三、数据接口契约

> **底层资产与调仓信号（holdings.json）**：由 `scripts/holdings.py` 复用 `cta_lab` 现有策略函数
> （`build_all` / `trend_lf_v2` 等，参数与 `run_v2.py` 的 POOL 完全一致）重建「日期 × 品种」权重矩阵 W，
> 从中真实计算每个子策略的底层资产、当前多空方向与权重、板块净暴露、逐日换手与调仓事件，
> 并从行情真实计算 ER 流畅度、60 日动量等信号读数。信号定义（公式/参数/触发规则）是对
> `strategies.py` 源码的如实转述。**不重写策略、不改回测结果、不编造数据**；品种缺失即跳过。
> `sync_daily.py` 复用 `feed.build()`，故云端每日回放模型后本接口自动同步，无需额外工作流。

| 接口 | 主要内容 |
|---|---|
| `meta.json` | 模型/参数指纹/数据指纹、口径说明、数据纯度、目标考核、工程指标、同步状态、回退策略 |
| `nav.json` | 817 个交易日 7 个组合净值、回撤、杠杆、ML 预测波动、沪深300 基准、三层递进净值、月度收益、分年统计、各变体样本内外指标 |
| `signal.json` | 当前生效权重（先验锚定）、当前回撤/敞口、ML 动态权重序列、风控杠杆事件 |
| `weights.json` | 先验权重 / 风险平价权重 / 真实风险贡献、四大类聚合、进化最优参数、杠杆序列 |
| `strategies.json` | 8 个入池策略 + 3 个筛除策略绩效、协整检验、样本外相关性、行情流畅度实证 |
| `holdings.json` | **底层资产与调仓信号**：32 品种元数据与 6 大板块、资产×策略覆盖矩阵、8 个策略各自的信号定义（公式/参数/触发规则）、当前信号状态（ER/宽度/降敞口系数/立场）、当前品种级持仓（方向/权重/组合有效敞口/动量/ER）、板块净暴露、信号时序、真实调仓事件、全市场信号环境 |
| `validation.json` | Bootstrap 置信区间、参数敏感性、分年子样本、ML 增量诊断、第二层缺口推演 |
| `sync_audit.json` | 最近 60 次同步审计（run_id / 状态 / 指纹 / 新增 / 重试 / 回退 / 步骤） |
| `rebalances.csv` | 权重与杠杆显著变动事件明细 |

**口径**：商品期货用交易所**结算价**、指数用收盘价；主力连续不复权，换月跳空按 `|r|>12%` 或 `>6×60日波动` 判定并清洗（60 个异常点）。样本内 `2023-07-03~2025-06-30` 用于选型/训练/调参，样本外 `2025-07-01~2026-11-13` 全程隔离。

---

## 四、每日自动同步与自动部署方案

### 4.1 同步链路（`daily_sync.yml`，每日 06:00 北京时间 + 手动触发）

| 步骤 | 动作 | 约束 |
|---|---|---|
| 1/7 | 依赖与结果快照检查 | 缺快照即终止（无回退基线不硬跑） |
| 2/7 | 复用 `fetch_data.py` 抓 32 品种 | 整体重试上限 **3**；单品种失败 → **跳过并沿用上一版价格**，不插值、不编造 |
| 3/7 | 数据质量校验 | 行数 ≥200、列数 ≥20、缺失率 ≤5%，否则回退 |
| 4/7 | 复用 `eda.py` → `run_backtest.py` → `run_v2.py` 回放 | 子进程超时上限 1200s |
| 5/7 | `feed.py` 生成 `data/` 接口 | 注入本次 run_id / 指纹 / 同步状态 |
| 6/7 | 指纹比对 | 数据指纹 + 参数指纹均未变 → `skipped`，**接口字节保持不变** |
| 7/7 | 写 `sync_audit.json` | 保留最近 60 条 |

**失败回退**：抓数或模型任一步失败 → 恢复回放前的 `output/` 快照 → 用上一版结果重放接口，
`sync.fallback=true` 且审计 `status=fallback`，**如实披露而不静默沿用**。

**幂等**：`data_fp = md5(最新交易日|评估期天数|期末净值)`、`params_fp = md5(最优参数|先验|执行参数|策略池)`；
周末/节假日无新交易日时自动跳过，`git commit` 也不会发生。

### 4.2 停更巡检（`watchdog.yml`，每 4 小时）

沿用你既有的时间线巡检习惯：检查 `meta.json` 的 `sync.last_sync`，**超 26 小时无新同步判定云端链停摆**，
自动 `gh workflow run daily_sync.yml` 重启，并把一行中文汇报写进 Actions 摘要：

```
同步状态：停摆 · 接口 8/8 个 · 数据截至 2026-11-13 · 交易日 817 · 距上次同步 30.2 小时（1812 分钟）· 重启指令 已下发 · 超阈值 26h
```

手动触发：`python3.11 scripts/check_freshness.py --threshold 26 --restart`

### 4.3 自动部署（`deploy_pages.yml`）

只发布 `index.html` + `data/`（不发布策略源码与行情快照）。由 `daily_sync.yml` 在提交变更后
**显式 `gh workflow run deploy_pages.yml` 触发**，绕开「`GITHUB_TOKEN` 的 push 不触发其它工作流」这个常见坑；
同时保留 `push` 触发，方便手动改页面后立即生效。

### 4.4 本地验证记录（本轮已实跑）

- `--dry-run` / `--mode snapshot` 幂等 skipped / `--force` 强制重写 / 停更巡检：四条路径全部通过，审计步骤 `[1..7]` 完整。
- **隔离副本跑通 `live` 全链路**（94 秒）：32 个品种抓数全部成功、无失败品种；`eda / run_backtest / run_v2` 全部 OK；
  V2 的 Bootstrap CI `[-0.432, 3.029]` 与 ML 增量诊断 `oos_sharpe_v2d=0.965` 与既有基线**逐项一致**，证明链路可复现。
  该测试在临时副本中进行，**未改动交付工程的数据基线**。

---

## 五、首次上线（待你确认后执行）

> 本轮**没有**执行 `git init` / `git commit` / `git push`，以下为后续操作清单。

1. 在 GitHub 新建仓库（例如 `cta-nav`），Settings → Pages → **Build and deployment → Source 选 `GitHub Actions`**。
2. Settings → Actions → General → Workflow permissions 选 **Read and write permissions**（`daily_sync.yml` 需要 `contents: write`、`actions: write`）。
3. 推送工程（把下面命令里的地址换成你的仓库）：
   ```bash
   cd /workspace/cta-dashboard
   git init && git add . && git commit -m "feat: CTA 策略看板 + 每日同步 + 自动部署"
   git branch -M main
   git remote add origin git@github.com:<你的账号>/cta-nav.git
   git push -u origin main
   ```
4. 首次手动跑一次：Actions → **每日同步 · CTA 策略模型接口** → Run workflow（mode=`snapshot` 先验证接口与部署，再跑 `live`）。
5. Actions → **部署看板 · GitHub Pages** → Run workflow，得到 `https://<账号>.github.io/cta-nav/`。
6. 之后每天 06:00 自动同步；`watchdog.yml` 每 4 小时巡检，停摆自动重启。

> 如果仓库有分支保护导致 `git push` 被拒，改用 PAT：把 `secrets.SYNC_TOKEN` 配到仓库，并在
> `daily_sync.yml` 的 checkout 步骤加 `with: token: ${{ secrets.SYNC_TOKEN }}`。

---

## 六、本地预览

```bash
cd /workspace/cta-dashboard
python3.11 -m http.server 8848
# 浏览器打开 http://localhost:8848/
```

**必须通过 HTTP 访问**：`file://` 直接打开无法 `fetch data/`（浏览器同源策略），
页面会显示「请通过 HTTP 访问本页」提示。

常用命令：

```bash
python3.11 scripts/sync_daily.py --mode snapshot           # 重建接口（零网络依赖）
python3.11 scripts/sync_daily.py --mode live --dry-run     # 只看计划不写文件
python3.11 scripts/sync_daily.py --mode live --force       # 完整抓数+回放+强制重写
python3.11 scripts/check_freshness.py --threshold 26       # 停更巡检
python3.11 scripts/feed.py                                 # 仅重建 data/ 接口
```

复现研究报告（需额外 `pip install plotly`）：
`cd cta_lab && python3.11 run_backtest.py && python3.11 run_v2.py && python3.11 report.py && python3.11 report_v2.py`

---

## 七、必须声明的局限

- 样本外仅 334 天，Bootstrap 夏普 95% CI 为 `[-0.43, 3.03]`，**统计显著性不足**。
- 主力连续为不复权拼接；成本为线性换手 + 校准后的非线性冲击模型，未含冲击/保证金实时约束/涨跌停。
- 第二层「真实中高频 CTA」因缺分钟/Tick 数据**本轮仍未验证**；看板中 ML 权重章节如实标注「进化判定 λ=0，ML 未实际进入组合」。
- 宏观层用股/债/商代理，未含外汇与债券票息。
- **所有数字是研究回测，不构成投资建议。**

数据与页面同源生成，任何一处改动都会同时反映在接口与看板上；若接口异常，页面会显示加载失败提示而非旧数据伪装。
