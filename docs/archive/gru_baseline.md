# GRU Mainline - Strict Mask K20 Keep2

## Status
- Baseline version: promoted tradable GRU mainline
- Model: GRU Baseline
- Mainline score run: `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/`
- Mainline strategy run: `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_strictmask_overlay/`
- Mainline strategy config: `configs/gru_mainline_strategy.yaml`
- Mainline config: `configs/sequence_gru_l20_mse_ic_frozen_head.yaml`
- Recorded date: 2026-05-29
- Upload policy: local run artifacts are excluded from Git by `.gitignore`.
- Baseline decision: old GRU baseline scores plus strict tradable mask plus K20 keep=2x turnover buffer is the GRU mainline strategy.
- Historical versions kept only for traceback: `e02_gru_l20_mse_ic_02`, `gru_l20_mse_ic_gelu_head`, `gru_l20_mse_ic_leaky_head_001`, and stable variants.
- Historical versions should not be used for new experiments unless explicitly doing retrospective comparison.

## Mainline Strategy
- Score source: `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005/predictions.parquet`
- Execution universe: strict tradable mask from `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026_filter_log.csv`
- Overlay predictions: `outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_strictmask_overlay/predictions.parquet`
- Portfolio rule: equal-weight Top20 long portfolio.
- Turnover buffer: keep existing names while their current rank remains within Top40, because `K=20` and `keep_multiplier=2.0`.
- Rebalance: every 5 signal dates.
- Holding window: non-overlapping 5-day proxy.
- Main cost setting: 10 bps one-way transaction cost.
- Mainline analysis output: `outputs/analysis/gru_mainline_strategy/`

Promotion rationale:
- Strict mask overlay keeps the old score's test RankIC intact: old baseline RankIC `0.0544`, overlay RankIC `0.0564`.
- Retraining the 62-feature model on strict-mask samples was rejected as a failed experiment because test RankIC fell to about `0.0171`.
- K20 keep=2x is the stable compromise: it materially lowers turnover while keeping positive excess return.

Mainline test result at 10 bps:

| Strategy | K | Keep | Top annualized | Excess vs universe annualized | Long-short annualized | Top turnover | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strictmask overlay | 20 | 2.0x | 0.5985 | 0.0684 | 0.1457 | 0.6485 | -0.1818 |

Comparison with no turnover buffer:

| Strategy | K | Keep | Top annualized | Excess vs universe annualized | Top turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| strictmask overlay | 20 | 1.0x | 0.7277 | 0.1494 | 1.1152 |
| strictmask overlay | 20 | 2.0x | 0.5985 | 0.0684 | 0.6485 |

Interpretation:
- K20 keep=2x gives up some raw annualized return in exchange for a roughly 42% reduction in top-leg turnover versus the same strict-mask overlay without a buffer.
- This is now the default robust GRU strategy for reporting and downstream comparison.
- K30 keep=2x and K30 keep=3x remain sensitivity checks, not the promoted mainline.

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
- Mainline head: LeakyReLU hidden head with `negative_slope=0.005`, followed by a linear regression score
- Final output activation: none

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

## Historical Variants

Historical variants are retained only for traceback and report comparison. They are not active baselines for future experiments.

ReLU reference:
- Config: `configs/sequence_gru_baseline.yaml`
- Run: `outputs/runs/e02_gru_l20_mse_ic_02/`
- Status: historical version only; not used as the mainline after head saturation diagnosis.

GELU head:
- Config: `configs/sequence_gru_l20_mse_ic_gelu_head.yaml`
- Run: `outputs/runs/gru_l20_mse_ic_gelu_head/`
- Status: historical ablation only; saturation was removed but alpha weakened.

Parent LeakyReLU head:
- Config: `configs/sequence_gru_l20_mse_ic_leaky_head_001.yaml`
- Run: `outputs/runs/gru_l20_mse_ic_leaky_head_001/`
- Status: historical ablation only; kept as a fallback comparison to slope 0.005.

Stable variants:
- Config: `configs/sequence_gru_baseline_stable.yaml`
- Purpose: conservative ablation/stability check on top of the current date-aware MSE+IC baseline.
- Status: historical stability comparison only.
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

## Historical ReLU Reference Result
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

## Top-K Backtest
- Evaluation script: `scripts/backtest_topk.py`
- Input files:
  - `outputs/runs/e02_gru_l20_mse_ic_02/predictions.parquet`
  - `data/mart/labels/labels_v20260526.parquet`
- Output files:
  - `outputs/runs/e02_gru_l20_mse_ic_02/backtest_metrics.json`
  - `outputs/runs/e02_gru_l20_mse_ic_02/backtest_periods.csv`
- Method: non-overlapping 5-day holding-period proxy using label-side `future_return`.
- Rebalance stride: 5 signal dates.
- Cost model: equal-weight Top-K turnover multiplied by one-way cost bps.
- Main comparison below uses 10 bps one-way cost.

| Split | K | Periods | Top-K annualized | Top-K cumulative | Max drawdown | Excess vs benchmark annualized | Excess vs universe annualized | Long-short annualized | Avg turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 10 | 97 | -0.0910 | -0.1678 | -0.4999 | -0.0184 | 0.0512 | 0.0195 | 0.7732 |
| validation | 20 | 97 | -0.1498 | -0.2683 | -0.5120 | -0.0810 | -0.0157 | 0.0019 | 0.7577 |
| validation | 30 | 97 | -0.1268 | -0.2296 | -0.4603 | -0.0555 | 0.0117 | 0.0818 | 0.7409 |
| test | 10 | 66 | 0.2537 | 0.3446 | -0.2455 | -0.2475 | -0.1420 | -0.1728 | 0.8394 |
| test | 20 | 66 | 0.3344 | 0.4590 | -0.2122 | -0.1957 | -0.0851 | 0.0930 | 0.7818 |
| test | 30 | 66 | 0.4591 | 0.6401 | -0.2080 | -0.1177 | 0.0020 | 0.1501 | 0.7152 |

Backtest interpretation:
- Test Top30 is the strongest setting: annualized return is approximately 45.9% after 10 bps cost, roughly flat versus the equal-weight universe, and long-short annualized return is positive.
- The strategy still underperforms the benchmark on test, because the 2025-2026 test window is a strong benchmark period.
- Validation is weak in absolute return, but Top10/Top30 are slightly positive versus the equal-weight universe.
- The current GRU signal is therefore a valid model-level baseline with some portfolio value, but it is not yet a robust standalone trading strategy.

## Long-Short Backtest
- Evaluation script: `scripts/backtest_topk.py`
- Method: buy Top-K and short Bottom-K using the same non-overlapping 5-day holding windows.
- Cost model: one-way cost applied to both long and short turnover.
- Main fields: `long_short_net`, `average_long_short_turnover`, `average_long_short_transaction_cost`.

| Split | K | Cost bps | Periods | Long-short annualized | Long-short cumulative | Max drawdown | Win rate | Sharpe-like | Avg LS turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 10 | 10 | 97 | -0.0734 | -0.1365 | -0.3371 | 0.5155 | -0.3259 | 1.8928 |
| validation | 20 | 10 | 97 | -0.0829 | -0.1535 | -0.2882 | 0.5052 | -0.4841 | 1.7526 |
| validation | 30 | 10 | 97 | -0.0021 | -0.0041 | -0.1772 | 0.5258 | -0.0171 | 1.6027 |
| test | 10 | 10 | 66 | -0.2518 | -0.3160 | -0.3357 | 0.4545 | -0.8865 | 1.9788 |
| test | 20 | 10 | 66 | 0.0008 | 0.0011 | -0.1568 | 0.5000 | 0.0040 | 1.7500 |
| test | 30 | 10 | 66 | 0.0652 | 0.0862 | -0.1354 | 0.5000 | 0.4079 | 1.5263 |

Long-short interpretation:
- Before costs, K=30 has positive long-short performance on both validation and test.
- After 10 bps one-way cost, validation K=30 is approximately flat and test K=30 remains mildly positive.
- K=10 fails on test even before costs, confirming that the model's highest-conviction head is unstable.
- Long-short turnover is very high, roughly 1.5 to 2.0 per rebalance, so cost control is now a first-order issue.

## Top10 Failure Diagnosis
- Evaluation script: `scripts/diagnose_topk.py`
- Input files:
  - `outputs/runs/e02_gru_l20_mse_ic_02/predictions.parquet`
  - `data/mart/datasets/dataset_v20260526.parquet`
  - `data/lake/raw/basic/basic_b5ea92fdf45d.parquet`
- Output files:
  - `outputs/runs/e02_gru_l20_mse_ic_02/topk_diagnosis_test_groups.csv`
  - `outputs/runs/e02_gru_l20_mse_ic_02/topk_diagnosis_test_industry.csv`
  - `outputs/runs/e02_gru_l20_mse_ic_02/topk_diagnosis_test_worst.csv`
  - `outputs/runs/e02_gru_l20_mse_ic_02/topk_diagnosis_test_spread.csv`

Key test-split findings:
- Top10 is not obviously lower-quality than Top30 by liquidity, size, or volatility. Top10 has slightly smaller mean future return than Top30, but similar size and slightly lower 20-day volatility.
- The main anomaly is prediction-score saturation. In the test split, 16,435 / 28,415 rows, or approximately 57.8%, are exactly at the maximum prediction score `0.008304736`.
- On 324 / 329 test dates, at least 10 names share the maximum score; on 267 / 329 test dates, at least 30 names share the maximum score.
- This means Top10 is often selected from a large tied-score plateau instead of a truly ordered head. Top30 is more stable mainly because it diversifies across the plateau.
- Worst Top10 contributors repeatedly include the same max-score names across adjacent dates, for example `300033.SZ` in April 2026 and several March 2025 names. This suggests score saturation plus repeated holding-window exposure, not only a simple liquidity or small-cap issue.

Top10 failure interpretation:
- The immediate root cause is not a clean low-liquidity or high-volatility contamination pattern.
- The higher-priority issue is output saturation / poor head ranking resolution.
- Liquidity and volatility filters are still useful, but they should be treated as secondary ablations after fixing or bypassing the tied-score head.

## Assessment
`gru_l20_mse_ic_leaky_head_slope_0005` remains the canonical GRU score model. The promoted GRU mainline strategy is now the score model plus strict tradable mask overlay plus K20 keep=2x turnover buffer. The earlier ReLU, GELU, parent LeakyReLU, stable runs, clean alpha-only run, and clean alpha + residual style run are not active mainline strategies unless explicitly used for ablation or retrospective comparison.

## Run Command
Mainline run:
```bash
conda activate dl_env
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml --device cuda
```

Rebuild and export promoted mainline strategy:
```bash
conda activate dl_env
python scripts/build_strictmask_prediction_overlay.py
python scripts/run_turnover_control_eval.py --run-dir outputs/runs/gru_l20_mse_ic_leaky_head_slope_0005_strictmask_overlay
python scripts/export_gru_mainline_strategy.py
```

Historical ReLU reference run:
```bash
conda activate dl_env
python scripts/train_sequence.py --config configs/sequence_gru_baseline.yaml --device cuda
```

GELU head ablation run:
```bash
conda activate dl_env
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_gelu_head.yaml --device cuda
```

LeakyReLU head ablation run:
```bash
conda activate dl_env
python scripts/train_sequence.py --config configs/sequence_gru_l20_mse_ic_leaky_head_001.yaml --device cuda
```

LeakyReLU robustness grid:
```bash
conda activate dl_env
python scripts/run_leaky_head_grid.py --device cuda --evaluate
```
The grid has been reduced to the selected slope-0.005 variant after freezing the active head.

Stable run:
```bash
conda activate dl_env
python scripts/train_sequence.py --config configs/sequence_gru_baseline_stable.yaml --device cuda
```

## Next Actions
1. Treat `configs/gru_mainline_strategy.yaml` as the canonical GRU strategy configuration.
2. Treat `configs/sequence_gru_l20_mse_ic_frozen_head.yaml` as the canonical GRU score-model training configuration.
3. Use K20 keep=2x as the default reporting portfolio for GRU mainline comparisons.
4. Keep K30 keep=2x and K30 keep=3x as turnover sensitivity checks only.
5. Do not promote clean-v1 retraining variants unless they beat the promoted strategy after strict mask and turnover buffer.
6. Run GRU lookback=60 only after the current mainline strategy is frozen for reporting.
