# feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean Closed-Loop Analysis

## Executive Readout

- Training selected epoch 12 with validation checkpoint_score 0.051608; rank IC mean was 0.032550.
- The executable loop is complete: predictions, T+1 fill simulation, soft optimizer grid, and this summary are all materialized.
- The key risk is validation/test divergence: validation execution metrics are negative, while test absolute returns are strong. Treat this as research evidence, not a promotion signal.
- Test-period gains remain mostly beta/market-regime assisted: T+1 test absolute returns are positive, but excess versus benchmark is still negative for the best absolute-return rows.

## Training

| run_name | best_epoch | selection_metric | best_metric | best_ic_mean | best_rank_ic_mean | best_rank_icir | best_val_loss | best_pred_std | stop_reason | train_samples | validation_samples | test_samples | prediction_rows | lookback | num_features | model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | 12 | checkpoint_score | 0.051608 | 0.030695 | 0.032550 | 0.231031 | 0.078966 | 0.010875 | metric_early_stop:checkpoint_score | 114946 | 40624 | 26678 | 67302 | 60 | 18 | feature_style_interaction_gru |

## Prediction Diagnostics

| split | rows | dates | score_mean | score_std | score_min | score_max | label_mean | label_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 26678 | 311 | -0.002766 | 0.012977 | -0.172432 | 0.067353 | -0.002935 | 0.067726 |
| validation | 40624 | 468 | 0.000809 | 0.010875 | -0.194632 | 0.077087 | -0.001073 | 0.054722 |

## T+1 Fill Simulation: Best Rows By Split

| run_name | split | setting | net_ann | net_ir | net_mdd | win_rate | excess_benchmark_ann | excess_exec_universe_ann | avg_desired_turnover | avg_filled_turnover | avg_transaction_cost | avg_position_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | top_10_keep_1.5x | -0.147331 | -0.073154 | -0.364982 | 0.382979 | -0.057730 | -0.013408 | 1.551805 | 0.038142 | 0.000057 | 24.425532 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | top_10_keep_2x | -0.147857 | -0.071794 | -0.368843 | 0.382979 | -0.056733 | -0.012154 | 1.403213 | 0.037626 | 0.000056 | 22.904255 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | top_10_keep_3x | -0.151575 | -0.076710 | -0.356095 | 0.404255 | -0.062614 | -0.018146 | 1.163795 | 0.024872 | 0.000037 | 20.255319 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | top_30_keep_1.5x | -0.152696 | -0.061664 | -0.386409 | 0.404255 | -0.052141 | -0.003156 | 0.927280 | 0.064006 | 0.000096 | 55.265957 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | top_30_keep_2x | -0.154448 | -0.062753 | -0.369610 | 0.393617 | -0.053986 | -0.005430 | 0.624894 | 0.046403 | 0.000070 | 44.297872 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | top_30_keep_3x | -0.156741 | -0.065769 | -0.362483 | 0.414894 | -0.057728 | -0.010037 | 0.216033 | 0.018278 | 0.000027 | 33.425532 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | top_10_keep_1x | -0.157041 | -0.078218 | -0.376839 | 0.382979 | -0.067761 | -0.023967 | 1.670005 | 0.044830 | 0.000067 | 25.968085 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | top_30_keep_1x | -0.160055 | -0.064931 | -0.397586 | 0.414894 | -0.059649 | -0.010970 | 1.240611 | 0.090322 | 0.000135 | 68.180851 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | top_20_keep_1.5x | 0.549093 | 0.263183 | -0.170358 | 0.555556 | -0.087406 | 0.052450 | 1.348946 | 0.128154 | 0.000192 | 44.380952 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | top_30_keep_1x | 0.508440 | 0.272589 | -0.184987 | 0.571429 | -0.111191 | 0.021582 | 1.259412 | 0.197853 | 0.000297 | 64.047619 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | top_30_keep_2x | 0.442820 | 0.238290 | -0.172159 | 0.587302 | -0.151241 | -0.021316 | 0.696518 | 0.091922 | 0.000138 | 45.285714 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | top_10_keep_3x | 0.410864 | 0.181112 | -0.200435 | 0.555556 | -0.166395 | -0.036879 | 1.433430 | 0.077081 | 0.000116 | 24.031746 |

## Soft Optimizer Core80: Best Rows By Split

| run_name | split | risk_control | k | style_penalty | turnover_penalty | periods | net_ann | net_ir | net_max_drawdown | excess_benchmark_ann | excess_exec_universe_ann | avg_desired_turnover | avg_filled_turnover | avg_filled_desired_ratio | avg_position_count | avg_cash_weight | avg_invested_weight | optimal_rate | feasible_rate | solver_error_rate | fallback_rate | avg_max_abs_exposure_z | avg_abs_exposure_z | avg_max_exposure_slack | avg_exposure_slack | avg_total_exposure_slack | avg_buy_capacity_slack | avg_min_invested_shortfall | min_invested_rule_pass_rate | grid_run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | none | 10 | 0.000000 | 0.000000 | 94 | -0.059635 | -0.013411 | -0.313069 | 0.047074 | -0.028238 | 0.123194 | 0.153775 | 1.261727 | 65.765957 | 0.160784 | 0.839216 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003168 | 0.011731 | 0.978723 | 1 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | none | 10 | 0.100000 | 0.000000 | 94 | -0.059635 | -0.013411 | -0.313069 | 0.047074 | -0.028238 | 0.123194 | 0.153775 | 1.261727 | 65.765957 | 0.160784 | 0.839216 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003168 | 0.011731 | 0.978723 | 3 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | none | 10 | 0.000000 | 0.020000 | 94 | -0.061213 | -0.014482 | -0.317115 | 0.045135 | -0.030015 | 0.119979 | 0.119979 | 1.000003 | 65.627660 | 0.161567 | 0.838433 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003168 | 0.011731 | 0.978723 | 2 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | none | 10 | 0.100000 | 0.020000 | 94 | -0.061213 | -0.014482 | -0.317115 | 0.045135 | -0.030015 | 0.119979 | 0.119979 | 1.000003 | 65.627660 | 0.161567 | 0.838433 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003168 | 0.011731 | 0.978723 | 4 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | none | 20 | 0.000000 | 0.020000 | 94 | -0.080171 | -0.020182 | -0.357903 | 0.030489 | -0.044047 | 0.123105 | 0.123105 | 1.000003 | 67.829787 | 0.094690 | 0.905310 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003168 | 0.011731 | 0.978723 | 6 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | none | 20 | 0.100000 | 0.020000 | 94 | -0.080171 | -0.020182 | -0.357903 | 0.030489 | -0.044047 | 0.123105 | 0.123105 | 1.000003 | 67.829787 | 0.094690 | 0.905310 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003168 | 0.011731 | 0.978723 | 8 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | none | 30 | 0.100000 | 0.020000 | 94 | -0.080171 | -0.020182 | -0.357903 | 0.030489 | -0.044047 | 0.123105 | 0.123105 | 1.000003 | 67.829787 | 0.094690 | 0.905310 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003168 | 0.011731 | 0.978723 | 12 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | validation | none | 30 | 0.000000 | 0.020000 | 94 | -0.080171 | -0.020182 | -0.357903 | 0.030489 | -0.044047 | 0.123105 | 0.123105 | 1.000003 | 67.829787 | 0.094690 | 0.905310 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003168 | 0.011731 | 0.978723 | 10 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | none | 20 | 0.100000 | 0.020000 | 63 | 0.358337 | 0.226526 | -0.147384 | -0.202818 | 0.035382 | 0.275862 | 0.275572 | 0.998216 | 58.206349 | 0.039980 | 0.960020 | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000892 | 0.006580 | 0.984127 | 8 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | none | 20 | 0.000000 | 0.020000 | 63 | 0.358337 | 0.226526 | -0.147384 | -0.202818 | 0.035382 | 0.275862 | 0.275572 | 0.998216 | 58.206349 | 0.039980 | 0.960020 | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000892 | 0.006580 | 0.984127 | 6 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | none | 30 | 0.100000 | 0.020000 | 63 | 0.358337 | 0.226526 | -0.147384 | -0.202818 | 0.035382 | 0.275862 | 0.275572 | 0.998216 | 58.206349 | 0.039980 | 0.960020 | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000892 | 0.006580 | 0.984127 | 12 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | none | 30 | 0.000000 | 0.020000 | 63 | 0.358337 | 0.226526 | -0.147384 | -0.202818 | 0.035382 | 0.275862 | 0.275572 | 0.998216 | 58.206349 | 0.039980 | 0.960020 | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000892 | 0.006580 | 0.984127 | 10 |

## T+1 Comparator Snapshot

| run_name | split | setting | net_ann | net_ir | net_mdd | win_rate | excess_benchmark_ann | excess_exec_universe_ann | avg_desired_turnover | avg_filled_turnover | avg_transaction_cost | avg_position_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | top_20_keep_1.5x | 0.549093 | 0.263183 | -0.170358 | 0.555556 | -0.087406 | 0.052450 | 1.348946 | 0.128154 | 0.000192 | 44.380952 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | test | top_20_keep_1x | 0.517426 | 0.185555 | -0.230151 | 0.555556 | -0.098827 | 0.042323 | 1.557832 | 0.158189 | 0.000237 | 46.428571 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | top_30_keep_1x | 0.508440 | 0.272589 | -0.184987 | 0.571429 | -0.111191 | 0.021582 | 1.259412 | 0.197853 | 0.000297 | 64.047619 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | top_30_keep_2x | 0.442820 | 0.238290 | -0.172159 | 0.587302 | -0.151241 | -0.021316 | 0.696518 | 0.091922 | 0.000138 | 45.285714 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | test | top_30_keep_3x | 0.419965 | 0.218337 | -0.193181 | 0.619048 | -0.163007 | -0.039055 | 0.347642 | 0.037810 | 0.000057 | 35.539683 |
| feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean | test | top_10_keep_3x | 0.410864 | 0.181112 | -0.200435 | 0.555556 | -0.166395 | -0.036879 | 1.433430 | 0.077081 | 0.000116 | 24.031746 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_10_keep_1x | -0.021363 | 0.008417 | -0.317593 | 0.478723 | 0.089230 | 0.146947 | 1.742582 | 0.041163 | 0.000062 | 25.712766 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_10_keep_1.5x | -0.022683 | 0.007755 | -0.316418 | 0.446809 | 0.088060 | 0.145787 | 1.611252 | 0.038055 | 0.000057 | 24.840426 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_10_keep_2x | -0.039490 | -0.001072 | -0.320080 | 0.446809 | 0.069252 | 0.126089 | 1.505345 | 0.031274 | 0.000047 | 23.712766 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_10_keep_3x | -0.057975 | -0.010657 | -0.325067 | 0.446809 | 0.049046 | 0.105156 | 1.278383 | 0.026397 | 0.000040 | 21.457447 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_20_keep_1x | -0.082060 | -0.019902 | -0.364615 | 0.436170 | 0.026289 | 0.081775 | 1.558939 | 0.068839 | 0.000103 | 48.968085 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_20_keep_1.5x | -0.127722 | -0.044702 | -0.377893 | 0.393617 | -0.025057 | 0.028197 | 1.280030 | 0.055299 | 0.000083 | 43.521277 |

## Decision

- Do not promote as production mainline on the current evidence.
- Keep as a candidate architecture because the TopK wide-band loss improves test T+1 absolute return and Top10 robustness, but require a validation-positive rerun or rolling-window confirmation.
- Next useful experiment: keep architecture fixed and test stronger validation alignment, for example lower model capacity or add explicit excess-return/benchmark-relative selection in the loss/evaluation gate.