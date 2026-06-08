# GRU Clean Dataset Stock Selection

This repository contains the source code for a deep-learning stock selection experiment on the ChiNext universe. The final pipeline builds a point-in-time clean dataset, trains a GRU-family sequence model, runs T+1 execution evaluation, and evaluates a frozen portfolio optimizer.

The final mainline is:

```text
raw A-share market data
-> clean_dataset v20260526
-> L60 alpha + residual-style tensor
-> feature-style interaction GRU
-> checkpoint_score epoch 12
-> frozen optimizer: risk_control=none, k=10, min_invested=0.8
```

Data files, generated model tensors, predictions, and model weights are intentionally not committed. The experiment report must describe the data source and date range; this repository provides the code and configuration needed to reproduce the results after the data is placed locally.

## Repository Structure

```text
configs/        Reproducible YAML configs for data, features, models, backtests, and optimizer
data/           Local data workspace; only .gitkeep placeholders are tracked
docs/           Experiment design and final analysis notes
legacy/         Historical full62 experiments kept for archive only
meta/           Schema registry and generated metadata locations
outputs/        Generated runs, predictions, audits, and reports; only .gitkeep is tracked
pipelines/      Core data and mart pipeline implementations
scripts/        Command-line entry points for data, training, evaluation, audit, and live workflow
src/            Model, dataset, and training library code
```

The root `README.md` is the single reproduction entry point. Files under `docs/` are analysis and background material, not required operating instructions.

## Environment Setup

Use Python 3.10 or a compatible Python 3.x environment. A CUDA-capable GPU is recommended for the final model training, but the code can run on CPU for dry-runs and small smoke checks.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quick syntax check:

```bash
python -m compileall scripts src pipelines
```

## Data Preparation

The code expects local A-share source data under the directory configured by `configs/data/data.yaml`:

```yaml
source:
  root_dir: "A股数据"
```

Place the raw data directory at the repository root, or edit `configs/data/data.yaml` so `source.root_dir` points to your local data location.

Expected raw datasets:

```text
A股数据/
  basic.csv
  trade_cal.csv
  daily/
  stock_st/
  index_weight/
  metric/
  moneyflow/
  market/
```

Important data assumptions:

- Main universe: ChiNext index, `399006.SZ`.
- Main data version: `v20260526`.
- Main historical range used by the frozen experiment: `20160104` to `20260525`.
- Data files are not included in the source submission.
- Generated parquet, npz, model weights, and predictions are local artifacts and are ignored by Git.

## Build Data Pipeline

Run commands from the repository root.

Full offline DAG:

```bash
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260525
```

This runs raw ingestion, ChiNext pool construction, market-state construction, coverage validation, and mart feature/label generation.

Equivalent step-by-step commands:

```bash
python scripts/data/run_ingest_raw.py --data-version v20260526
python scripts/data/run_build_pool.py --data-version v20260526
python scripts/data/run_build_market_state.py --data-version v20260526 --incremental
python scripts/data/validate_market_state_coverage.py --data-version v20260526 --start-date 20160104 --end-date 20260525 --strict
python scripts/data/run_build_mart.py --data-version v20260526 --start-date 20160104 --end-date 20260525
```

Key generated files:

```text
data/lake/raw/
data/lake/core/chinext_pool/chinext_pool_scd2.parquet
data/lake/state/security_daily_state.parquet
data/mart/features_daily/features_daily_v20260526.parquet
data/mart/labels/labels_v20260526.parquet
data/mart/datasets/core/dataset_v20260526.parquet
```

## Feature Engineering

Validate the clean feature contract:

```bash
python scripts/features/validate_clean_feature_set.py
```

The active feature contract is:

```text
configs/features/advanced_sequence_clean_v1.yaml
```

It defines the clean alpha and residual-style feature groups used by the final GRU model.

## Build Model Tensors

Build the tensors used by the final model:

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

The final model uses the L60 tensor:

```text
data/mart/datasets/clean_purged_wf/dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

Optional baseline tensor:

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
```

## Model Training

First run a CPU dry-run to verify config, paths, tensor shapes, and model construction:

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --dry-run --device cpu
```

Train the final model on GPU:

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
```

CPU fallback:

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cpu
```

Expected training output:

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/
  config.yaml
  metrics.json
  model.pt
  predictions.parquet
```

The frozen report uses epoch 12 selected by `checkpoint_score`. Training is seeded with `seed: 42` in the model config, but exact GPU reproducibility can still vary slightly across hardware, CUDA, BLAS, and PyTorch builds.

## Evaluation

Run the frozen T+1 execution route:

```bash
python scripts/backtest/run_clean_resid_mainline.py
```

Run the frozen final optimizer:

```bash
python scripts/portfolio/run_final_mainline_optimizer.py
```

The optimizer config is:

```text
configs/portfolio/final_mainline_optimizer.yaml
```

It expects:

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet
data/mart/datasets/core/dataset_v20260526.parquet
data/mart/labels/execution_labels_v20260526.parquet
```

If `execution_labels_v20260526.parquet` is missing, generate labels with:

```bash
python scripts/data/build_execution_labels.py --help
```

Then rerun with the arguments shown by the script help for your local label configuration.

## Audit And Analysis

Point-in-time leakage audit:

```bash
python scripts/audit/audit_point_in_time.py --labels data/mart/labels/labels_v20260526.parquet --out-dir outputs/audit/point_in_time
```

Residual alpha audit:

```bash
python scripts/audit/audit_barra_lite_residual_alpha.py
```

Closed-loop summary:

```bash
python scripts/analysis/summarize_model_closed_loop.py
```

Optimizer validation attribution:

```bash
python scripts/analysis/analyze_optimizer_validation_attribution.py --periods outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_periods.csv --summary outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_summary.csv --output-dir outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution --split validation --top-n 6
```

The attribution command requires the optimizer grid outputs. To rebuild a grid for a run:

```bash
python scripts/portfolio/run_soft_optimizer_grid.py --predictions outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet --output-dir outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80
```

## Reproduce Results

Use this complete sequence for a fresh local reproduction:

```bash
python -m compileall scripts src pipelines
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260525
python scripts/features/validate_clean_feature_set.py
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --dry-run --device cpu
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
python scripts/backtest/run_clean_resid_mainline.py
python scripts/portfolio/run_soft_optimizer_grid.py --predictions outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet --output-dir outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80
python scripts/portfolio/run_final_mainline_optimizer.py
python scripts/audit/audit_point_in_time.py --labels data/mart/labels/labels_v20260526.parquet --out-dir outputs/audit/point_in_time
python scripts/audit/audit_barra_lite_residual_alpha.py
python scripts/analysis/summarize_model_closed_loop.py
```

Expected final result directory:

```text
outputs/backtest/optimizer/final_mainline_ckptscore_e12/
  final_optimizer_periods.csv
  final_optimizer_summary.csv
  manifest.json
```

Expected headline result from the frozen run:

| split | net_ann | net_ir | excess_benchmark_ann | excess_exec_universe_ann | avg_invested_weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | -0.059635 | -0.013411 | 0.047074 | -0.028238 | 0.839216 |
| test | 0.268252 | 0.189517 | -0.258701 | -0.035435 | 0.893154 |

Interpretation: the final optimizer completes a reproducible research loop, but it should not be described as a stable production alpha because validation and test do not both show positive excess versus the executable universe.

## Output Description

Generated artifacts are ignored by Git:

```text
data/       local raw, lake, mart, tensor, and label artifacts
outputs/    training runs, predictions, weights, audits, and evaluation outputs
logs/       local audit and runtime logs
```

For source submission, keep generated data and model weights out of Git. Include the commands above and the data description in the experiment report.

## Documentation

The project can be reproduced from this README alone. Additional documents are optional context:

```text
docs/experiment_report.md                      Consolidated final experiment report
docs/01_data_and_label_protocol.md             Data and label design notes
docs/02_feature_engineering_clean_v1.md        Clean feature design notes
docs/06_results_and_limitations.md             Legacy result summary kept for traceability
docs/07_clean_dataset_v20260526_dictionary.md  Dataset dictionary and interpretation notes
docs/08_feature_style_interaction_wide30_closed_loop.md  Historical candidate analysis
docs/09_final_mainline_freeze.md               Frozen mainline record
```
