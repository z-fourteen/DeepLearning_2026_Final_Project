# 2026-06-01 至 2026-06-12 实盘流水线

本文记录比赛 10 个交易日的实盘流水线入口、输入合同、强断言和盘中执行辅助逻辑。

## 一键入口

立即运行完整盘前三阶段：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601
```

按真实时间轴等待 08:30、09:00、09:15 后运行：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601 -WaitForSchedule
```

盘前完成后继续进入盘中未成交残股监控：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601 -WaitForSchedule -RunIntradayMonitor
```

PowerShell 入口会强制设置：

```text
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
CONDA_NO_PLUGINS=true
conda run --no-capture-output -n dl_env ...
```

这样可以避免 Windows GBK 控制台与 conda stdout 捕获导致的中文输出崩溃。

## 时间轴

| 时间 | 阶段 | 脚本 | 输出 |
| --- | --- | --- | --- |
| 08:30-09:00 | 盘前数据校验与 Alpha 推理 | `scripts/live/01_live_inference.py` | `outputs/live_predictions/predictions_YYYYMMDD.parquet` |
| 09:00-09:15 | CVXPY live optimizer | `scripts/live/02_live_optimization.py` | `outputs/live_targets/target_weights_YYYYMMDD.csv` |
| 09:15-09:25 | 目标调仓差分明细 | `scripts/live/03_generate_target_orders.py` | `outputs/live_orders/orders_YYYYMMDD.csv` |
| 09:30-15:00 | 盘中残股监控与撤单重报建议 | `scripts/live/04_intraday_execution_monitor.py` | `outputs/live_monitor/intraday_advice_YYYYMMDD.csv` |

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
- live 输入文件模板。
- 输出目录。
- 数据覆盖率、持仓继承、最小订单金额和整手规则。

## 输入数据合同

### live feature panel

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

上一交易日真实收盘成交持仓：

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

## 强断言

### 数据缺失断言

`guards.expected_universe_size=5000`，`guards.min_market_coverage_ratio=0.8`。因此 live 数据中可用股票数低于 4000 时，脚本会立即终止并发出终端警报，不生成预测和订单。

### 持仓继承检查

阶段二会强制比较：

```text
positions_{trade_date}.csv
close_positions_{prev_trade_date}.csv
```

若 `weight` 或 `volume` 与上一交易日真实收盘成交持仓不一致，脚本立即终止，防止 old_w 时间轴断裂。

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

终端会打印：

```text
【买入调仓看板】
code action target_value target_volume

【卖出调仓看板】
code action target_value target_volume
```

交易股数按 `guards.lot_size=100` 向下取整，低于 `guards.min_order_value=1000` 的碎单会自动过滤。

## 盘中执行辅助

单次检查：

```powershell
conda run --no-capture-output -n dl_env python scripts/live/04_intraday_execution_monitor.py --trade-date 20260601
```

循环检查：

```powershell
conda run --no-capture-output -n dl_env python scripts/live/04_intraday_execution_monitor.py --trade-date 20260601 --loop --interval-minutes 5 --ticks 2
```

broker status 文件默认路径：

```text
data/live/broker/order_status_{trade_date}.csv
```

最低列要求：

```text
code
action
submitted_volume
filled_volume
order_price
best_bid
best_ask
tick_size
```

盘中监控逻辑：

- 每次读取未成交残股 `unfilled_volume = submitted_volume - filled_volume`。
- BUY 单建议价格：`best_ask + ticks * tick_size`。
- SELL 单建议价格：`best_bid - ticks * tick_size`。
- 终端打印撤单重报建议。
- 建议落盘到 `outputs/live_monitor/intraday_advice_YYYYMMDD.csv`。

TWAP/VWAP 执行原则：

- 09:30-10:00 完成 20%-30% 目标量，避免开盘冲击。
- 10:00-14:30 按成交量曲线匀速追踪，连续 N 分钟未成交则撤单重报。
- 14:30 后优先保证 `min_invested` 和卖出风险释放，可更积极向五档盘口让价。
- 14:50 后仍有未成交买单且仓位低于 80%，必须人工确认是否以更激进价格追单。

## 最终开盘前命令

2026 年 6 月 1 日开盘前：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601
```

若希望脚本严格等待时间轴：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601 -WaitForSchedule
```
