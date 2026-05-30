# Backtest And Execution

Primary execution stack:

```powershell
python scripts/backtest/run_clean_dataset_execution_stack.py --only-existing
```

Focused clean-resid mainline:

```powershell
python scripts/backtest/run_clean_resid_mainline.py
```

The core execution artifact is the T+1 open-fill simulation:

```text
scripts/backtest/backtest_t1_fill_sim.py
```

It evaluates buy and sell executability, partial capacity constraints,
transaction costs, slippage, turnover, and benchmark-relative returns.
