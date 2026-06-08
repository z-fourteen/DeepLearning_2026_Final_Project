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
