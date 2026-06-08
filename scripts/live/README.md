# Live Workflow Scripts

The live workflow is a post-experiment extension for the 2026-06-01 to 2026-06-12 trading window. It is not required for reproducing the assignment results.

## One-Command Entry

Windows PowerShell:

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601
```

Strict scheduled run:

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601 -WaitForSchedule
```

The PowerShell wrapper sets UTF-8 and conda execution environment variables to reduce Windows console encoding issues.

## Stage Map

| Stage | Script | Output |
| --- | --- | --- |
| Live inference | `scripts/live/01_live_inference.py` | `outputs/live_predictions/predictions_YYYYMMDD.parquet` |
| Live optimizer | `scripts/live/02_live_optimization.py` | `outputs/live_targets/target_weights_YYYYMMDD.csv` |
| Order generation | `scripts/live/03_generate_target_orders.py` | `outputs/live_orders/orders_YYYYMMDD.csv` |
| Fill recording | `scripts/live/05_interactive_execution.py` | `outputs/live/orders/execution_YYYYMMDD.json` |
| Close valuation | `scripts/live/06_close_valuation.py` | `outputs/live/valuations/valuation_YYYYMMDD.*` |

## Config

```text
configs/live/live_trading.yaml
```

## Input Contracts

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
