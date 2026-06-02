# 2026-06-01 至 2026-06-12 实盘流水线

本文记录比赛 10 个交易日的实盘流水线入口、输入合同、强断言、成交确认，以及收盘估值口径。

## 一键入口

立即运行完整盘前三阶段：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601
```

按真实时间轴等待 08:30、09:00、09:15 后运行：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601 -WaitForSchedule
```

PowerShell 入口会强制设置：

```text
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
CONDA_NO_PLUGINS=true
conda run --no-capture-output -n dl_env ...
```

这样可以避免 Windows GBK 控制台与 conda stdout 捕获导致的中文输出乱码。

## 时间轴

| 时间 | 阶段 | 脚本 | 输出 |
| --- | --- | --- | --- |
| 08:30-09:00 | 盘前数据校验与 Alpha 推理 | `scripts/live/01_live_inference.py` | `outputs/live_predictions/predictions_YYYYMMDD.parquet` |
| 09:00-09:15 | CVXPY live optimizer | `scripts/live/02_live_optimization.py` | `outputs/live_targets/target_weights_YYYYMMDD.csv` |
| 09:15-09:25 | 目标调仓差分明细 | `scripts/live/03_generate_target_orders.py` | `outputs/live_orders/orders_YYYYMMDD.csv` |
| 成交确认后 | 手工录入真实成交价和成交股数 | `scripts/live/05_interactive_execution.py` | `outputs/live/orders/execution_YYYYMMDD.json`、`outputs/live/portfolio_state.json` |
| 收盘后 | 读取 raw daily 当日 close 并计算实际收益 | `scripts/live/06_close_valuation.py` | `outputs/live/valuations/valuation_YYYYMMDD.*` |

## 配置入口

```text
configs/live/live_trading.yaml
```

该配置集中管理：

- 比赛交易日：20260601 至 20260612。
- 冻结模型路径：epoch 12 checkpoint。
- 18 个 live 特征的严格顺序。
- CVXPY 生产 optimizer 参数。
- 时间步动态短缺惩罚参数。
- 创业板全量 universe：2026-06-02 起仅允许 `300*.SZ`、`301*.SZ`。
- live 输入文件模板。
- 输出目录。
- 数据覆盖率、持仓继承、最小订单金额和整手规则。

## 输入数据合同

### Live Feature Panel

默认路径：

```text
data/live/features/features_{trade_date}.parquet
```

最低列要求：

```text
trade_date
ts_code 或 code
lag1_net_mf_strength_20d_mean
lag1_net_mf_strength_60d_mean
lag1_close_position
lag1_excess_ret_10d_mean
lag1_excess_ret_1d
lag1_excess_ret_5d_mean
lag1_industry_neutral_ret_1d
lag1_ret_1d
lag1_ret_20d
lag1_ret_5d_mean
lag1_bollinger_z_20d
lag1_ma_ratio_20_60
lag1_macd_hist
lag1_turnover_cost_proxy__resid_style
lag1_turnover_20d_std__resid_style
lag1_turnover_60d_std__resid_style
lag1_amount_rank_pct__resid_style
lag1_amount_log__resid_style
amount 或 next_amount
```

每只股票至少需要 60 个交易日的历史行，用于构造 `[N, 60, 18]` live tensor。若盘前已经提前构造好 NPZ，也可提供：

```text
data/live/features/live_sequence_{trade_date}.npz
```

NPZ 必须包含：

```text
X
ts_code
feature_names
```

### 持仓文件

当前开盘前持仓：

```text
data/live/account/positions_{trade_date}.csv
```

上一交易日真实收盘后成交持仓：

```text
data/live/account/close_positions_{prev_trade_date}.csv
```

最低列要求：

```text
ts_code 或 code
weight
volume
```

若没有 `weight`，可提供 `market_value`，脚本会按总市值派生权重。若没有 `volume`，脚本无法安全生成卖出股数，因此卖单会被限制为 0 股；正式比赛必须提供真实可卖股数。

### 价格快照

默认路径：

```text
data/live/market/quotes_{trade_date}_0920.csv
```

最低列要求：

```text
ts_code 或 code
price 或 last_price 或 open 或 pre_close 或 close
```

该价格只用于把目标权重转换成订单股数，以及作为成交交互的默认参考价。成交成本以 `05_interactive_execution.py` 中录入的真实成交价为准。

## 强断言

### 数据缺失断言

2026-06-02 起，实盘股票池按全创业板代码池执行，即所有 `300*.SZ`、`301*.SZ` 且能构造 60 日 live 序列的股票都进入阶段 01 推理。2026-06-01 的已成交实盘记录保持当日 Top100 候选池事实，不追溯重写。

```text
universe.name=chinext_full
universe.effective_date=20260602
universe.feature_source_after_effective_date=raw_daily
universe.allowed_prefixes=300,301
universe.allowed_suffix=.SZ
guards.expected_universe_size=1200
guards.min_market_coverage_ratio=0.8
```

因此，阶段 00/01/02 都会先过滤到 `300*.SZ`、`301*.SZ`；若 2026-06-02 起过滤后可用股票数低于 960，脚本会立即终止并发出终端警报，不生成后续预测、目标权重或订单。

三道防线分别是：

- 阶段 00：2026-06-02 起强制从 raw daily 构造全创业板 live feature panel，写入 `data/live/features/features_YYYYMMDD.parquet` 前先过滤 universe。
- 阶段 01：即使手工指定了旧的全市场 feature/NPZ，也会在推理前过滤，并在输出 predictions 前断言。
- 阶段 02：预测、流动性、当前持仓、上一日持仓均必须属于创业板 universe，否则优化阶段终止。

本修复以 2026-06-02 作为全创业板开始日，只影响 2026-06-02 及后续交易日策略；不能覆盖已经发生的 2026-06-01 实盘成交流水。

### 持仓继承检查

阶段二会强制比较：

```text
positions_{trade_date}.csv
close_positions_{prev_trade_date}.csv
```

若 `weight` 或 `volume` 与上一交易日真实收盘持仓不一致，脚本立即终止，防止 `old_w` 时间轴断裂。

### 模型输入合同

阶段一会检查：

- 特征列是否完整。
- 特征顺序是否与冻结 L60 模型一致。
- 每只股票是否有 60 日 lookback。
- tensor 中是否存在 NaN/Inf。
- 股票覆盖率是否达标。

## 时间步动态短缺惩罚

阶段二不会把 `min_invested>=0.8` 写死成不可达硬约束，而是注入动态 shortfall penalty：

```text
penalty = base + (max - base) * progress^2
```

默认：

```text
base = 500
max = 5000
progress = 当前比赛日序号 / 9
```

这样做的目的：

- 6 月 1 日比赛首日，如果旧持仓为空或太低，A 股 3% participation cap 可能导致 80% 仓位硬约束不可达。
- soft shortfall 变量避免优化器死锁。
- 随着比赛推进，惩罚逐步抬升，推动仓位尽快回到 80% 规则内。
- 如果最终仍低于 80%，脚本会在终端发出仓位警报，提示优先执行买单。

## 订单输出合同

订单文件：

```text
outputs/live_orders/orders_YYYYMMDD.csv
```

核心列：

```text
trade_date
code
action
price_ref
target_value
target_volume
delta_weight
```

交易股数按 `guards.lot_size=100` 向下取整，低于 `guards.min_order_value=1000` 的碎单会自动过滤。

## 成交交互

`05_interactive_execution.py` 只负责记录真实成交，不负责计算当日收益。

运行示例：

```powershell
python scripts/live/05_interactive_execution.py --trade-date 20260601 --orders-csv outputs/live_orders/orders_20260601.csv --price-snapshot data/live/market/quotes_20260601_0920.csv --no-push
```

交互端录入的 `actual_price` 是真实成交价，会写入 `avg_cost` 作为持仓成本。它不是收盘价，也不能直接作为当日盯市收益价格。

## 收盘估值

收盘后，`A股数据/daily/YYYYMMDD.csv` 会更新当日 raw daily 文件。阶段六只读取该文件中的真实 `close`，并据此计算组合盯市收益；不再手工逐只录入收盘价，也不再使用盘前 quote snapshot。

```powershell
python scripts/live/06_close_valuation.py --trade-date 20260601 --daily-csv "A股数据/daily/20260601.csv" --write-close-positions
```

若使用默认路径，可省略 `--daily-csv`：

```powershell
python scripts/live/06_close_valuation.py --trade-date 20260601 --write-close-positions
```

阶段六默认会查找 `outputs/live/orders/execution_YYYYMMDD.json`。若该日志存在且当天是初始建仓日，脚本会用 `status in {filled, partial}`、`actual_shares > 0`、`actual_price > 0` 的成交记录重建一份理论持仓，并与 `portfolio_state.json` 比对；`skipped`、`failed` 和 `actual_shares=0` 的记录不会计入。若发现 state 与 execution 不一致，脚本会终止，避免用错误持仓计算收盘收益。

如需显式按 execution 修复 state 后再估值，使用：

```powershell
python scripts/live/06_close_valuation.py --trade-date 20260601 --daily-csv "A股数据/daily/20260601.csv" --rebuild-state-from-execution --write-close-positions
```

Raw daily CSV 最低列要求：

```text
ts_code 或 code
trade_date
close
```

输出：

```text
outputs/live/valuations/valuation_YYYYMMDD.csv
outputs/live/valuations/valuation_YYYYMMDD.json
data/live/account/close_positions_YYYYMMDD.csv  # 仅在 --write-close-positions 时生成
```

收益计算口径：

```text
position_value = sum(shares * close_price)
NAV = cash + position_value
daily_pnl = NAV - previous_nav
daily_return = daily_pnl / previous_nav
unrealized_pnl_vs_cost = position_value - sum(shares * avg_cost)
```

若未提供 `--previous-nav`，首日默认以前一基准 NAV `state.initial_nav` 为收益基准；后续交易日优先使用上一条 `last_valuation.nav`。

## 2026-06-01 建仓日口径修正

2026-06-01 的实盘记录按“下午初始建仓”处理。当天组合在交易前为空仓，交互端录入的 `actual_price` 均为真实买入成交价，用于形成持仓成本；这些价格不是官方收盘价，也不是用于当日盯市收益计算的估值价格。

因此，2026-06-01 不确认日内 PnL，也不报告当日收益率。`outputs/live/orders/execution_20260601.json` 已合并首批建仓单和补仓单，并标注 `day_classification=initial_build`、`same_day_return_applicable=false`。若需要确认 2026-06-01 的真实交易收益，必须在收盘后通过 `06_close_valuation.py` 输入官方收盘价，生成独立估值记录。

2026-06-01 的真实成交事实不追溯重写。2026-06-02 起全创业板修复只能产生新的理论信号或后续交易日信号，不能用新的理论结果覆盖 `execution_20260601.json`、`portfolio_state.json` 或 `valuation_20260601.*`。

## 最终开盘前命令

2026 年 6 月 1 日开盘前：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601
```

若希望脚本严格等待时间轴：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601 -WaitForSchedule
```
