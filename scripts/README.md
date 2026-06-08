# Scripts Index

This directory contains command-line entry points. The root `README.md` is the single reproduction guide; use this file only as a script map when maintaining or extending the project.

Run all commands from the repository root.

## Data Pipeline

| Script | Purpose |
| --- | --- |
| `scripts/run_daily_dag.py` | Runs the full data DAG: ingest, pool, market state, validation, and mart build. |
| `scripts/data/run_ingest_raw.py` | Ingests raw source files into `data/lake/raw/`. |
| `scripts/data/run_build_pool.py` | Builds the ChiNext pool SCD2 table. |
| `scripts/data/run_build_market_state.py` | Builds tradability, listing, suspension, and limit-state features. |
| `scripts/data/validate_market_state_coverage.py` | Validates date and universe coverage. |
| `scripts/data/run_build_mart.py` | Builds daily features, labels, and core mart datasets. |
| `scripts/data/build_execution_labels.py` | Builds execution labels used by T+1 backtests and optimizer evaluation. |
| `scripts/data/build_canonical_labels.py` | Builds canonical labels for audit variants. |
| `scripts/data/query_market_state.py` | Queries generated market-state records. |
| `scripts/data/validate_ingest_schema.py` | Checks source schema compatibility. |

## Feature And Modeling

| Script | Purpose |
| --- | --- |
| `scripts/features/validate_clean_feature_set.py` | Validates the clean feature contract against mart fields. |
| `scripts/features/generate_feature_role_tags.py` | Regenerates feature role/tag metadata. |
| `scripts/modeling/build_clean_model_datasets.py` | Builds clean sequence tensors under `data/mart/datasets/clean_purged_wf/`. |
| `scripts/modeling/train_sequence.py` | Trains GRU-family sequence models from YAML configs. |

## Backtest And Portfolio

| Script | Purpose |
| --- | --- |
| `scripts/backtest/backtest_t1_fill_sim.py` | Runs T+1 fill simulation from prediction parquet files. |
| `scripts/backtest/run_clean_resid_mainline.py` | Runs the frozen T+1 mainline config. |
| `scripts/backtest/run_clean_dataset_execution_stack.py` | Batch-runs fixed clean-dataset execution stacks. |
| `scripts/portfolio/optimize_feasible_cash_buffer.py` | Runs the lower-level feasible-cash optimizer. |
| `scripts/portfolio/run_soft_optimizer_grid.py` | Runs soft-optimizer parameter grids. |
| `scripts/portfolio/run_final_mainline_optimizer.py` | Runs the frozen final optimizer. |
| `scripts/portfolio/run_capacity_participation_matrix.py` | Runs capacity and participation-rate sensitivity checks. |

## Audit And Analysis

| Script | Purpose |
| --- | --- |
| `scripts/audit/audit_point_in_time.py` | Audits point-in-time feature and label safety. |
| `scripts/audit/audit_barra_lite_residual_alpha.py` | Audits residual alpha after Barra-lite controls. |
| `scripts/audit/audit_clean_resid_mainline.py` | Runs a deeper audit of the clean residual mainline. |
| `scripts/analysis/summarize_model_closed_loop.py` | Summarizes training, prediction diagnostics, T+1, and optimizer evidence. |
| `scripts/analysis/analyze_optimizer_validation_attribution.py` | Analyzes optimizer validation attribution and weak periods. |

## Live Workflow

The live workflow is an extension of the offline experiment and is not required for reproducing the assignment results.

| Script | Purpose |
| --- | --- |
| `scripts/live/live_daily.py` | Live daily orchestration entry point. |
| `scripts/live/00_prepare_live_inputs.py` | Prepares live features, positions, and price snapshots. |
| `scripts/live/01_live_inference.py` | Runs live inference. |
| `scripts/live/02_live_optimization.py` | Runs live target-weight optimization. |
| `scripts/live/03_generate_target_orders.py` | Generates target orders. |
| `scripts/live/05_interactive_execution.py` | Records manual fills. |
| `scripts/live/06_close_valuation.py` | Performs close valuation and state updates. |

Use `python <script> --help` for argument details.
