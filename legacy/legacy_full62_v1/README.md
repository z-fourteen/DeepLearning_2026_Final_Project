# Legacy Full62 V1 Archive

This directory is a read-only research archive for the historical `full62` mainline.

It contains the old GRU baseline configs, Top-K/turnover experiments, soft penalty
portfolio reranker, greedy hard-constraint selectors, LP feasibility trials, and
the corresponding audit/backtest outputs.

The current production-facing project mainline is `clean_dataset` only:

- `configs/features/advanced_sequence_clean_v1.yaml`
- `scripts/modeling/build_clean_model_datasets.py`
- `scripts/modeling/train_sequence.py`
- `scripts/backtest/backtest_t1_fill_sim.py`
- `scripts/backtest/run_clean_dataset_execution_stack.py`

Do not import code from this archive in clean pipeline scripts. If a historical
result is needed for comparison, cite the archived output path directly.
