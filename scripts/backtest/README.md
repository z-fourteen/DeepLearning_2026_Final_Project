# Backtest Scripts

This directory contains T+1 execution and fixed-mainline backtest entry points.

## Frozen T+1 Mainline

```bash
python scripts/backtest/run_clean_resid_mainline.py
```

This command uses:

```text
configs/backtest/clean_resid_t1_top20_keep2.yaml
```

## Generic T+1 Fill Simulation

```bash
python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet --output-dir outputs/backtest/t1_fill_sim/<run_name>
```

Common settings:

```bash
python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet --k 10,20,30 --keep-multiplier 1,1.5,2,3 --portfolio-nav 10000000 --participation-cap 0.03 --rebalance-stride 5
```

The simulator evaluates ranking signals under open-price execution, participation limits, transaction cost, slippage, turnover, and executable-universe benchmarks.

Generated backtest outputs are ignored by Git.
