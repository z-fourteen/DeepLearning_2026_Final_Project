# Experiment Report

## Overview

This project studies GRU-based stock selection on the ChiNext universe. The final submitted pipeline builds a point-in-time clean dataset, trains a feature-style interaction GRU, evaluates T+1 execution, and freezes a portfolio optimizer selected on validation evidence.

The root `README.md` is the only reproduction guide. This report records the experiment design and final findings.

## Data And Universe

| Item | Setting |
| --- | --- |
| Universe | ChiNext index universe, `399006.SZ` |
| Data version | `v20260526` |
| Historical range | `20160104` to `20260525` |
| Split protocol | Purged walk-forward with train, validation, and test splits |
| Label horizon | 5 trading days |
| Main dataset | `clean_dataset v20260526` |

Data files are not submitted with the source code. The report assumes the same local A-share data schema described in the root README.

## Clean Dataset

The clean dataset separates alpha features from execution and risk-control fields. The final model uses the `alpha_plus_residual_style` tensor:

```text
data/mart/datasets/clean_purged_wf/dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

Tensor contract:

| Key | Meaning |
| --- | --- |
| `X` | `[N, lookback, num_features]` model input tensor |
| `y` | supervised label |
| `trade_date` | signal date |
| `ts_code` | stock code |
| `split` | `train`, `validation`, or `test` |
| `feature_names` | ordered feature contract |

Final model feature shape:

| Item | Value |
| --- | ---: |
| lookback | 60 |
| alpha features | 13 |
| residual-style features | 5 |
| total features | 18 |

## Final Model

```text
model: feature_style_interaction_gru
run: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
best_metric: 0.05160820540933453
training objective: topk_band_margin_ic
```

Model scale:

| Item | Value |
| --- | ---: |
| train samples | 114,946 |
| validation samples | 40,624 |
| test samples | 26,678 |
| validation dates | 468 |
| prediction rows | 67,302 |
| batch size | 384 |
| batch mode | date |

Epoch 12 was selected because it maximized validation `checkpoint_score`. Later epochs continued to reduce training loss but degraded ranking evidence, so they were treated as overfitting risk.

## Prediction Diagnostics

| split | rows | dates | score_mean | score_std | label_mean | label_std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 40,624 | 468 | 0.000809 | 0.010875 | -0.001073 | 0.054722 |
| test | 26,678 | 311 | -0.002766 | 0.012977 | -0.002935 | 0.067726 |

The prediction score dispersion is much smaller than the label dispersion. The model output should therefore be interpreted as a cross-sectional ranking score, not as a direct return-magnitude forecast.

## T+1 Execution Evidence

The raw T+1 fill simulation is not the final optimizer, but it checks whether model rankings survive executable trading assumptions.

Better validation rows remained negative:

| validation setting | net_ann | net_ir | max_drawdown | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: |
| top_10_keep_1.5x | -0.147331 | -0.073154 | -0.364982 | -0.013408 |
| top_10_keep_2x | -0.147857 | -0.071794 | -0.368843 | -0.012154 |
| top_30_keep_1.5x | -0.152696 | -0.061664 | -0.386409 | -0.003156 |

Test T+1 showed stronger absolute returns:

| test setting | net_ann | net_ir | max_drawdown | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: |
| top_20_keep_1.5x | 0.549093 | 0.263183 | -0.170358 | 0.052450 |
| top_30_keep_1x | 0.508440 | 0.272589 | -0.184987 | 0.021582 |
| top_10_keep_3x | 0.410864 | 0.181112 | -0.200435 | -0.036879 |

This indicates test-period stock-selection value, but part of the absolute return is regime-assisted and does not consistently beat the benchmark.

## Final Optimizer

Frozen optimizer settings:

```text
risk_control: none
k: 10
style_penalty: 0.1
turnover_penalty: 0.0
min_invested: 0.8
participation_cap: 0.03
portfolio_nav: 10000000
cost_bps: 10
slippage_bps: 5
```

Final optimizer result:

| split | periods | net_ann | net_ir | net_max_drawdown | excess_benchmark_ann | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 94 | -0.059635 | -0.013411 | -0.313069 | 0.047074 | -0.028238 |
| test | 63 | 0.268252 | 0.189517 | -0.145907 | -0.258701 | -0.035435 |

Constraint health:

| split | avg_cash_weight | avg_invested_weight | avg_filled_turnover | avg_position_count |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.160784 | 0.839216 | 0.153775 | 65.77 |
| test | 0.106846 | 0.893154 | 0.344442 | 55.54 |

Solver health:

| split | optimal_rate | feasible_rate | solver_error_rate | fallback_rate | min_invested_rule_pass_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.978723 |
| test | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.984127 |

The optimizer is stable from a solver and constraint perspective, but it does not produce positive excess versus the executable universe on both validation and test.

## Validation Decision

Epoch-12 optimizer grid validation ranking supported:

| rank | risk_control | k | style_penalty | turnover_penalty | net_ann | excess_exec_universe_ann |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | none | 10 | 0.0 | 0.0 | -0.059635 | -0.028238 |
| 2 | none | 10 | 0.1 | 0.0 | -0.059635 | -0.028238 |
| 3 | none | 10 | 0.0 | 0.02 | -0.061213 | -0.030015 |
| 4 | none | 10 | 0.1 | 0.02 | -0.061213 | -0.030015 |

The final freeze used `style_penalty=0.1`, but because `risk_control=none`, `style_penalty=0.1` and `style_penalty=0.0` are equivalent in this final optimizer configuration.

## Limitations

- The final optimizer does not show positive excess versus the executable universe in both validation and test.
- The test period has strong absolute returns, but benchmark-relative performance remains weak.
- Test-favorable alternatives such as larger `k` values are post-hoc observations and should not replace validation-selected parameters.
- The model score is ranking-oriented, not calibrated as return magnitude.
- Exact numerical reproduction can vary across PyTorch, CUDA, and hardware environments.

## Conclusion

The final mainline is a complete and reproducible research loop:

```text
clean data -> model selection -> prediction diagnostics -> T+1 execution -> optimizer -> audit evidence
```

It should be described as a reproducible final research submission, not as a production-ready stable alpha strategy.
