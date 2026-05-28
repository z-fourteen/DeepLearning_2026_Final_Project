# GRU Baseline - Date-Aware MSE+IC

## Status
- Baseline version: frozen current baseline
- Model: GRU Baseline
- Config: `configs/sequence_gru_baseline.yaml`
- Reference run: `outputs/runs/e02_gru_l20_mse_ic_02/`
- Recorded date: 2026-05-27
- Upload policy: local run artifacts are excluded from Git by `.gitignore`.
- Baseline decision: `e02_gru_l20_mse_ic_02` is the main GRU baseline for the current stage.

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

## Stable Variant
- Config: `configs/sequence_gru_baseline_stable.yaml`
- Purpose: conservative ablation/stability check on top of the current date-aware MSE+IC baseline.
- Status: not the main baseline; it is kept only as a stability comparison run.
- It keeps the same data interface, GRU architecture, date-aware batch mode, and `mse_ic` objective.
- Conservative changes:
  - Learning rate: 0.0003 -> 0.0002
  - Weight decay: 0.00001 -> 0.000001
  - `ic_loss_alpha`: 0.2 -> 0.15
  - Input dropout: 0.1 -> 0.05
  - GRU dropout: 0.2 -> 0.1
  - Head dropout: 0.3 -> 0.1
  - Max grad norm: 1.0 -> 0.75
  - Early-stop patience: 10 -> 12
  - Minimum best-checkpoint daily coverage: 0.8 -> 0.9
- Output directory: `outputs/runs/gru_l20_date_aware_mse_ic_baseline_stable/`

## Reference Result
- Best epoch: 13
- Best validation RankIC mean: 0.0299640
- Best validation IC mean: 0.0208654
- Validation RankIC IR: 0.2013793
- Recomputed test RankIC mean: 0.0487976
- Recomputed test IC mean: 0.0259337
- Recomputed test RankIC IR: 0.3148049
- Recomputed test ICIR: 0.1630962
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

## Top-K Proxy Evaluation
- Evaluation script: `scripts/evaluate_topk.py`
- Input file: `outputs/runs/e02_gru_l20_mse_ic_02/predictions.parquet`
- Output files:
  - `outputs/runs/e02_gru_l20_mse_ic_02/topk_metrics.json`
  - `outputs/runs/e02_gru_l20_mse_ic_02/topk_daily.csv`
  - `outputs/runs/e02_gru_l20_mse_ic_02/topk_quantiles.csv`
- Scope: prediction-only portfolio proxy using `label_rel_return`; this is not yet a transaction-cost-aware backtest.

| Split | K | Top mean | Bottom mean | Top-Bottom spread | Spread IR | Spread positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 10 | -0.000116 | -0.002515 | 0.002399 | 0.0719 | 0.5599 |
| validation | 20 | -0.000314 | -0.002098 | 0.001784 | 0.0752 | 0.5558 |
| validation | 30 | 0.000195 | -0.002680 | 0.002875 | 0.1620 | 0.5909 |
| test | 10 | -0.005884 | -0.004612 | -0.001272 | -0.0374 | 0.5106 |
| test | 20 | -0.004043 | -0.005081 | 0.001039 | 0.0406 | 0.5714 |
| test | 30 | -0.002721 | -0.004404 | 0.001684 | 0.0804 | 0.5441 |

Top-K proxy interpretation:
- Validation split shows positive long-short spread across K=10/20/30, with the cleanest result at K=30.
- Test split keeps positive spread at K=20/30, but the magnitude and IR are weak; K=10 is negative.
- Decile monotonicity is not clean, so the current signal is useful as a model-level ranking baseline but not yet strong enough to claim robust tradable portfolio value.

## Assessment
This is the frozen current GRU baseline. It replaces the earlier pure-regression E01 as the main GRU result, while the stable variant is retained only as an ablation. The date-aware MSE+IC objective aligns training with the daily cross-section RankIC evaluation target, avoids checkpoint-level prediction collapse, and keeps full validation/test daily coverage. The signal is positive on both validation and test, with stronger test RankIC than validation RankIC. Top-K proxy results confirm some ranking value, especially around K=20/30, but the weak test spread and non-monotonic deciles mean portfolio-level validation remains the next gate before claiming tradable strategy value.

## Run Command
```bash
conda activate dl_env
python scripts/train_sequence.py --config configs/sequence_gru_baseline.yaml --device cuda
```

Stable run:
```bash
conda activate dl_env
python scripts/train_sequence.py --config configs/sequence_gru_baseline_stable.yaml --device cuda
```

## Next Actions
1. Run a transaction-cost-aware Top-K backtest with t+1 execution and 5-day holding alignment.
2. Compare Top 10/20/30 against benchmark and equal-weight universe returns.
3. Consider a small `ic_loss_alpha` sweep only after backtest diagnostics are available.
