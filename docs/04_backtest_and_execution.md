# 回测与执行

主执行栈：

```powershell
python scripts/backtest/run_clean_dataset_execution_stack.py --only-existing
```

聚焦 clean-resid 主线：

```powershell
python scripts/backtest/run_clean_resid_mainline.py
```

核心执行产物是 T+1 开盘成交仿真：

```text
scripts/backtest/backtest_t1_fill_sim.py
```

该仿真会评估买入和卖出的可执行性、部分容量约束、交易成本、滑点、换手率以及相对基准收益。
