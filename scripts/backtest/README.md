# Backtest Scripts

## English

This directory contains T+1 execution and fixed-mainline backtest entry points.

### Frozen T+1 Mainline

```bash
python scripts/backtest/run_clean_resid_mainline.py
```

This command uses:

```text
configs/backtest/clean_resid_t1_top20_keep2.yaml
```

### Generic T+1 Fill Simulation

```bash
python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet --output-dir outputs/backtest/t1_fill_sim/<run_name>
```

Common settings:

```bash
python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet --k 10,20,30 --keep-multiplier 1,1.5,2,3 --portfolio-nav 10000000 --participation-cap 0.03 --rebalance-stride 5
```

The simulator evaluates ranking signals under open-price execution, participation limits, transaction cost, slippage, turnover, and executable-universe benchmarks.

Generated backtest outputs are ignored by Git.

## Transformer report runs

Run the six Transformer variants with the unified 5-day T+1 fill simulator:

```powershell
.\scripts\backtest\run_transformer_t1_5d_backtests.ps1 -Python python
```

Outputs:

```text
..\Final-YXR\outputs\backtest\unified_t1_5d_transformer\<run_name>\t1_fill_periods.csv
..\Final-YXR\outputs\backtest\unified_t1_5d_transformer\<run_name>\t1_fill_metrics.json
```

Run the same six Transformer variants with daily T+1 open rebalancing:

```powershell
.\scripts\backtest\run_transformer_daily_rebalance_backtests.ps1 -Python python
```

Outputs:

```text
..\Final-YXR\outputs\backtest\unified_daily_rebalance_transformer\<run_name>\full_topk\daily_rebalance_periods.csv
..\Final-YXR\outputs\backtest\unified_daily_rebalance_transformer\<run_name>\full_topk\daily_rebalance_positions.csv
..\Final-YXR\outputs\backtest\unified_daily_rebalance_transformer\<run_name>\full_topk\daily_rebalance_summary.csv
..\Final-YXR\outputs\backtest\unified_daily_rebalance_transformer\<run_name>\full_topk\daily_rebalance_metrics.json
```

## 中文

本目录包含 T+1 执行和固定主线回测入口。

### 冻结的 T+1 主线

```bash
python scripts/backtest/run_clean_resid_mainline.py
```

该命令使用：

```text
configs/backtest/clean_resid_t1_top20_keep2.yaml
```

### 通用 T+1 成交仿真

```bash
python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet --output-dir outputs/backtest/t1_fill_sim/<run_name>
```

常用设置：

```bash
python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet --k 10,20,30 --keep-multiplier 1,1.5,2,3 --portfolio-nav 10000000 --participation-cap 0.03 --rebalance-stride 5
```

仿真器会在开盘价执行、参与率约束、交易成本、滑点、换手和可执行股票池基准下评估排序信号。

生成的回测输出已被 Git 忽略。
