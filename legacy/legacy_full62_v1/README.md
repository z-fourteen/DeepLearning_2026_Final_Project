# Legacy Full62 V1 Archive

## English

This directory is a read-only research archive for the historical `full62` mainline.

It contains old GRU baseline configs, Top-K and turnover experiments, soft-penalty portfolio reranking, greedy hard-constraint selectors, LP feasibility trials, and corresponding audit or backtest outputs.

The current production-facing project mainline is `clean_dataset` only:

```text
configs/features/advanced_sequence_clean_v1.yaml
scripts/modeling/build_clean_model_datasets.py
scripts/modeling/train_sequence.py
scripts/backtest/backtest_t1_fill_sim.py
scripts/backtest/run_clean_dataset_execution_stack.py
```

Do not import code from this archive in clean pipeline scripts. If a historical result is needed for comparison, cite the archived output path directly.

## 中文

本目录是历史 `full62` 主线的只读研究归档。

其中包含旧版 GRU 基线配置、Top-K 与换手实验、soft penalty 组合重排序、贪心硬约束选择器、LP 可行性试验，以及对应的审计或回测输出。

当前面向最终提交的项目主线仅为 `clean_dataset`：

```text
configs/features/advanced_sequence_clean_v1.yaml
scripts/modeling/build_clean_model_datasets.py
scripts/modeling/train_sequence.py
scripts/backtest/backtest_t1_fill_sim.py
scripts/backtest/run_clean_dataset_execution_stack.py
```

不要在 clean 流水线脚本中导入本归档目录的代码。如需在报告中比较历史结果，应直接引用归档输出路径。
