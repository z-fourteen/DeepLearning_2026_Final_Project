# Baseline v1 - E01 GRU L20

## Status
- Baseline version: v1
- Model: GRU Baseline
- Run name: `e01_gru_l20_baseline`
- Result source: `outputs/runs/e01_gru_l20_baseline/`
- Recorded date: 2026-05-27
- Upload policy: local record only; run artifacts are excluded from Git by `.gitignore`.

## Data Interface
- Dataset: sequence NPZ
- Lookback: 20
- Input shape: `[B, T, F] = [B, 20, 62]`
- Train samples: 133,403
- Validation samples: 43,803
- Test samples: 28,415

## Training Environment
- Device: CUDA
- Batch size: 256
- Train steps per epoch: 522
- Validation steps per epoch: 172
- Test steps: 111
- Max epochs configured: 80
- Actual stopped epoch: 12

## Best Validation Result
- Best epoch: 2
- Best metric: validation `rank_ic_mean`
- Best validation RankIC mean: 0.0142805
- Best validation IC mean: 0.0095784
- Validation RankIC IR: 0.0964835
- Validation ICIR: 0.0641040
- Validation loss: 0.00147711

## Prediction Export
- Export file: `outputs/runs/e01_gru_l20_baseline/predictions.parquet`
- Rows: 72,218
  - validation: 43,803
  - test: 28,415
- Exported checkpoint: best epoch checkpoint, epoch 2
- Export columns:
  - `trade_date`
  - `ts_code`
  - `pred_score`
  - `label_rel_return`
  - `split`
  - `model_name`

## Validation Diagnostics
- Epochs 1-4 had valid daily cross-sections.
- From epoch 5 onward, validation entered prediction collapse:
  - `daily_status`: `prediction_collapse`
  - `pred_constant_daily_count`: 484 / 484
  - `pred_std`: 0 or near-zero
- The collapse was caused by constant model predictions, not by missing labels or invalid daily cross-sections.

## Exported Best Checkpoint Diagnostics
- Validation prediction std: approximately 0.000013
- Test prediction std: approximately 0.000014
- Validation constant prediction days: 0 / 484
- Test constant prediction days: 0 / 329
- Recomputed daily RankIC mean from exported predictions:
  - validation: approximately 0.01428
  - test: approximately 0.03069

## Assessment
This run is accepted as the first GRU baseline. It shows weak but positive ranking signal and confirms that the sequence data interface, GRU model, trainer, metric pipeline, checkpoint reload, and prediction export are connected end to end.

The main risk is training instability. After early epochs, the model can collapse to near-constant predictions. Future GRU experiments should add collapse-aware early stopping and test more conservative optimization settings before scaling model complexity.

## Next Actions
1. Add collapse-aware early stopping in `Trainer`.
2. Create a stable GRU config with lower learning rate, lower weight decay, and lighter dropout.
3. Add portfolio-style validation metrics, including top/bottom decile spread and long-short proxy.
