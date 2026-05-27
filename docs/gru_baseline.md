# GRU Baseline - Date-Aware MSE+IC

## Status
- Baseline version: current
- Model: GRU Baseline
- Config: `configs/sequence_gru_baseline.yaml`
- Reference run: `outputs/runs/e02_gru_l20_mse_ic_02/`
- Recorded date: 2026-05-27
- Upload policy: local run artifacts are excluded from Git by `.gitignore`.

## Data Interface
- Dataset: sequence NPZ
- Lookback: 20
- Input shape: `[B, T, F] = [B, 20, 62]`
- Train samples: 133,403
- Validation samples: 43,803
- Test samples: 28,415
- DataLoader output:
  - `x`: float tensor `[B, 20, 62]`
  - `y`: float tensor `[B]`
  - `trade_date`: list-like batch metadata
  - `ts_code`: list-like batch metadata

## Model
- Backbone: single-direction GRU
- Input projection: `FeatureProjection`
- Sequence pooling: last hidden state
- Output head: linear regression score
- Output activation: none

## Training Objective
- Loss: `mse_ic`
- Formula: `(1 - alpha) * MSE - alpha * PearsonIC`
- `ic_loss_alpha`: 0.2
- IC loss scope: same-`trade_date` daily cross-section
- Train batch mode: `date`
- Validation/test batch mode: sample batch

## Training Configuration
- Device: CUDA in reference run
- Batch size: 256
- Train steps per epoch: 1,650
- Validation steps per epoch: 172
- Test steps: 111
- Max epochs configured: 80
- Optimizer: AdamW
- Learning rate: 0.0003
- Weight decay: 0.00001
- Scheduler: cosine
- Early-stop metric: validation `rank_ic_mean`
- Early-stop patience: 10
- Minimum best-checkpoint daily coverage: 0.8
- Collapse stop patience: 2

## Reference Result
- Best epoch: 13
- Best validation RankIC mean: 0.0299640
- Best validation IC mean: 0.0208654
- Validation RankIC IR: 0.2013793
- Stop reason: `metric_early_stop:rank_ic_mean`
- Prediction-collapse epochs: 0
- Best daily count: 484 / 484
- Best constant prediction days: 0 / 484

## Prediction Export
- Export file: `outputs/runs/e02_gru_l20_mse_ic_02/predictions.parquet`
- Rows: 72,218
  - validation: 43,803
  - test: 28,415
- Export columns:
  - `trade_date`
  - `ts_code`
  - `pred_score`
  - `label_rel_return`
  - `split`
  - `model_name`

## Exported Checkpoint Diagnostics
- Validation prediction std: approximately 0.01698
- Test prediction std: approximately 0.01975
- Validation constant prediction days: 0 / 484
- Test constant prediction days: 0 / 329
- Recomputed daily RankIC mean:
  - validation: approximately 0.02996 on 484 days
  - test: approximately 0.04880 on 329 days

## Assessment
This is the current GRU baseline. It replaces the earlier pure-regression E01 and stable variants. The date-aware MSE+IC objective aligns training with the daily cross-section RankIC evaluation target, avoids prediction collapse, and keeps full validation/test daily coverage.

## Run Command
```bash
conda activate dl_env
python scripts/train_sequence.py --config configs/sequence_gru_baseline.yaml --device cuda
```

## Next Actions
1. Add portfolio-style validation metrics, including top/bottom decile spread and long-short proxy.
2. Consider a small `ic_loss_alpha` sweep after locking this baseline as the reference.
