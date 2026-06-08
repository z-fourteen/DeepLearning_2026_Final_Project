# Live Workflow Scripts

## English

The live workflow is a post-experiment extension for the 2026-06-01 to 2026-06-12 trading window. It is not required for reproducing the assignment results.

### One-Command Entry

Windows PowerShell:

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601
```

Strict scheduled run:

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601 -WaitForSchedule
```

The PowerShell wrapper sets UTF-8 and conda execution environment variables to reduce Windows console encoding issues.

### Stage Map

| Stage | Script | Output |
| --- | --- | --- |
| Live inference | `scripts/live/01_live_inference.py` | `outputs/live_predictions/predictions_YYYYMMDD.parquet` |
| Live optimizer | `scripts/live/02_live_optimization.py` | `outputs/live_targets/target_weights_YYYYMMDD.csv` |
| Order generation | `scripts/live/03_generate_target_orders.py` | `outputs/live_orders/orders_YYYYMMDD.csv` |
| Fill recording | `scripts/live/05_interactive_execution.py` | `outputs/live/orders/execution_YYYYMMDD.json` |
| Close valuation | `scripts/live/06_close_valuation.py` | `outputs/live/valuations/valuation_YYYYMMDD.*` |

### Config

```text
configs/live/live_trading.yaml
```

### Input Contracts

Live feature panel:

```text
data/live/features/features_{trade_date}.parquet
```

Positions:

```text
data/live/account/positions_{trade_date}.csv
data/live/account/close_positions_{prev_trade_date}.csv
```

Price snapshot:

```text
data/live/market/quotes_{trade_date}_0920.csv
```

Live inputs and outputs are local runtime artifacts and are ignored by Git.

## 中文

实盘流程是离线实验后的扩展，覆盖 2026-06-01 到 2026-06-12 的交易窗口。它不属于课程作业结果复现的必需流程。

### 一键入口

Windows PowerShell：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601
```

严格按计划时间运行：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601 -WaitForSchedule
```

PowerShell 封装会设置 UTF-8 和 conda 执行环境变量，以减少 Windows 控制台编码问题。

### 阶段映射

| 阶段 | 脚本 | 输出 |
| --- | --- | --- |
| 实盘推理 | `scripts/live/01_live_inference.py` | `outputs/live_predictions/predictions_YYYYMMDD.parquet` |
| 实盘优化 | `scripts/live/02_live_optimization.py` | `outputs/live_targets/target_weights_YYYYMMDD.csv` |
| 订单生成 | `scripts/live/03_generate_target_orders.py` | `outputs/live_orders/orders_YYYYMMDD.csv` |
| 成交记录 | `scripts/live/05_interactive_execution.py` | `outputs/live/orders/execution_YYYYMMDD.json` |
| 收盘估值 | `scripts/live/06_close_valuation.py` | `outputs/live/valuations/valuation_YYYYMMDD.*` |

### 配置

```text
configs/live/live_trading.yaml
```

### 输入契约

实盘特征面板：

```text
data/live/features/features_{trade_date}.parquet
```

持仓：

```text
data/live/account/positions_{trade_date}.csv
data/live/account/close_positions_{prev_trade_date}.csv
```

价格快照：

```text
data/live/market/quotes_{trade_date}_0920.csv
```

实盘输入和输出均为本地运行产物，已被 Git 忽略。
