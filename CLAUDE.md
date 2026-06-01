# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A GRU-based time-series stock selection system for Chinese ChiNext (创业板) stocks. The pipeline ingests raw CSV data from Tushare, builds point-in-time clean feature tensors, trains sequence models, and evaluates portfolio-level execution.

The **frozen final mainline** is:
- Feature set: `advanced_sequence_clean_v1` (13 alpha + 5 residualized style features = 18 total)
- Model: `FeatureStyleInteractionGRUStockModel` (L60 lookback, alpha-resid-style, FiLM gating)
- Checkpoint: epoch 12 selected by `checkpoint_score`
- Optimizer: `risk_control=none, k=10, style_penalty=0.1, min_invested=0.8`

Historical `full62` experiments are frozen in `legacy/legacy_full62_v1/` — do not modify them.

## Environment

Conda environment: `dl_env` (Python 3.10, PyTorch 2.5.1). Activate before running anything:
```bash
conda activate dl_env
```

Large NPZ dataset artifacts are tracked with Git LFS. After cloning:
```bash
git lfs install
git lfs pull
```

## Data Pipeline (DAG)

Convenience wrapper (runs all steps in order):
```bash
python scripts/run_daily_dag.py --data-version v20260526
```

Or run sequentially from raw data to training-ready tensors:

1. **Ingest raw CSVs** (from `A股数据/` → `data/lake/raw/`):
   ```bash
   python scripts/data/run_ingest_raw.py --data-version v20260526
   ```

2. **Build pool table** (ChiNext constituent list with SCD2):
   ```bash
   python scripts/data/run_build_pool.py
   ```

3. **Build market state** (tradability, ST, price limits, suspension):
   ```bash
   python scripts/data/run_build_market_state.py
   ```

4. **Build data mart** (feature daily parquet):
   ```bash
   python scripts/data/run_build_mart.py --data-version v20260526
   ```

5. **Build clean model datasets** (NPZ tensors for training):
   ```bash
   python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
   python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20,60
   ```

6. **Train model**:
   ```bash
   python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
   ```

All relative paths resolve from the project root (`scripts/modeling/train_sequence.py` inserts it into `sys.path`).

## Architecture

### Models (`src/models/`)
- `BaseStockModel` — abstract interface: `forward([B,T,F]) → [B]`
- `GRUStockModel` — plain GRU baseline (FeatureProjection → GRU → LayerNorm → PredictionHead)
- `FeatureStyleInteractionGRUStockModel` — alpha-only GRU backbone + FiLM-style gating from residualized style features. Splits input into alpha (first `alpha_feature_count` features) and style (rest), encodes each separately, then modulates alpha context with style-derived gamma/beta before the prediction head.
- `RegimeGatedGRUStockModel` — regime-conditional GRU variant

### Training (`src/training/`)
- `Trainer` — epoch loop with date-aware validation, `checkpoint_score` early stopping, and collapse detection
- Loss functions: `PearsonICLoss`, `MSEICLoss`, `TopKMarginICLoss`, `TopKBandMarginICLoss` (two-band ranking loss aligned with Top-10 long-only portfolios — the final mainline loss)
- `summarize_daily_ic` — daily cross-section IC/RankIC/ICIR plus diagnostics (prediction_collapse, target_collapse)

### Data (`src/data/`)
- `SequenceNPZDataset` — loads `.npz` files with keys `X`, `y`, `trade_date`, `ts_code`, `split`, `feature_names`. Filters by split on init.
- `DateBatchSampler` — yields batches that keep each trade_date cross-section together (critical for the date-aware loss functions)

### Pipeline (`pipelines/`)
- `pipelines/ingest/agent.py` — incremental CSV→Parquet ingestion with schema validation, append-only raw layer, MD5-based change detection, and file registry
- `pipelines/mart/clean_dataset.py` — builds sequence tensors: reads mart parquet, applies strict tradable mask (ST, suspended, price limits, liquidity, microcap), computes residualized style features (OLS against industry + style exposures), constructs rolling lookback windows, splits by date, outputs NPZ + sidecar + manifest
- `pipelines/pool/agent.py` — ChiNext pool SCD2 table builder
- `pipelines/state/agent.py` — daily security state (tradability flags, limit rules)

### Configs (`configs/`)
- `configs/data/data.yaml` — all data paths, schema definitions, pool/state configs, ingestion specs
- `configs/data/splits.yaml` — purged walk-forward splits (active: `final_2025_2026`: train 2016-2022, val 2023-2024, test 2025-2026)
- `configs/features/advanced_sequence_clean_v1.yaml` — feature contract: 13 alpha features, residualized style candidates, risk/tradability controls, strict tradable mask policy
- `configs/models/` — per-model training configs (model architecture, loss, optimizer, scheduler, early stopping)
- `configs/portfolio/final_mainline_optimizer.yaml` — frozen optimizer params and evidence paths
- `configs/live/live_trading.yaml` — live competition config: trading calendar, frozen model checkpoint, optimizer hyperparams, input/output file paths, safety guards

### Scripts (`scripts/`)
- `scripts/modeling/train_sequence.py` — unified training entrypoint. `--dry-run` builds objects and prints summary without training. `--max-epochs N` for smoke tests.
- `scripts/modeling/build_clean_model_datasets.py` — NPZ dataset builder entrypoint
- `scripts/features/validate_clean_feature_set.py` — validates feature contract
- `scripts/audit/` — point-in-time leakage audit, Barra-lite residual alpha audit
- `scripts/backtest/` — T+1 fill simulation
- `scripts/portfolio/` — CVXPY portfolio optimization (CLARABEL solver)
- `scripts/live/` — live inference pipeline (4-step: inference → optimization → target orders → intraday execution monitor)

### Outputs (`outputs/`)
- `outputs/runs/{run_name}/` — `model.pt`, `predictions.parquet`, `metrics.json`, `config.yaml`
- `outputs/backtest/` — backtest results
- `outputs/analysis/` — attribution analysis

## Live Trading Pipeline

The live pipeline (`scripts/live/01_live_inference.py` → `02_live_optimization.py` → `03_generate_target_orders.py` → `04_intraday_execution_monitor.py`) reads from `configs/live/live_trading.yaml` and uses shared utilities in `scripts/live/common.py`.

Key safety guards in `common.py`:
- `assert_position_inheritance()` — current day weights must match previous day close positions exactly (tolerance 1e-6)
- `assert_market_coverage()` — feature panel must cover >80% of expected universe
- `competition_progress()` / `dynamic_shortfall_penalty()` — penalty for under-investing increases quadratically toward end of competition
- `die()` — hard-fails with audible alarm (BEL char) to prevent silent live-trading failures

## New Model Onboarding

See `docs/03a_new_model_clean_dataset_onboarding.md` for the complete checklist. In short:
1. Create a YAML config in `configs/models/` with `num_features` and `lookback` matching the NPZ
2. Dry-run: `python scripts/modeling/train_sequence.py --config <path> --dry-run --device cpu`
3. PIT audit: `python scripts/audit/audit_point_in_time.py`
4. Train, then run T+1 backtest: `python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<name>/predictions.parquet`

## Key Concepts

- **Point-in-time correctness**: All features are lagged by 1 day (`lag1_` prefix). The strict tradable mask filters out non-executable samples (ST, suspended, limit-up/down locked, low liquidity, microcap).
- **Residualized style features**: OLS residuals of style/liquidity features against industry dummies + size/volatility/momentum exposures, computed per-trade-date. Removes style leakage while preserving alpha signal.
- **Date-aware loss**: Loss functions compute per-cross-section metrics (IC, top-k margin). The batch sampler keeps dates together so gradients are computed over complete cross-sections.
- **Checkpoint scoring**: Composite of RankIC mean + top-k proxy mean + prediction dispersion ratio. Model selection uses validation `checkpoint_score`, not val_loss.
- **No test suite**: This project has no `tests/` directory. Validation is done via audit scripts and by observing training metrics.