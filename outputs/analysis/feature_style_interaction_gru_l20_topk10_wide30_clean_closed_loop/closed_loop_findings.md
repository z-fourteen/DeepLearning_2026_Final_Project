# feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean Closed-Loop Analysis

## Executive Readout

- Training selected epoch 10 with validation rank IC 0.040001 and rank ICIR 0.261220.
- The executable loop is complete: predictions, T+1 fill simulation, soft optimizer grid, and this summary are all materialized.
- The key risk is validation/test divergence: validation execution metrics are negative, while test absolute returns are strong. Treat this as research evidence, not a promotion signal.
- Test-period gains remain mostly beta/market-regime assisted: T+1 test absolute returns are positive, but excess versus benchmark is still negative for the best absolute-return rows.

## Training

| run_name | best_epoch | best_rank_ic_mean | best_ic_mean | best_rank_icir | best_val_loss | best_pred_std | stop_reason | train_samples | validation_samples | test_samples | prediction_rows | lookback | num_features | model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | 10 | 0.040001 | 0.020936 | 0.261220 | 0.078885 | 0.004039 | metric_early_stop:rank_ic_mean | 124527 | 41393 | 27288 | 68681 | 20 | 18 | feature_style_interaction_gru |

## Prediction Diagnostics

| split | rows | dates | score_mean | score_std | score_min | score_max | label_mean | label_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 27288 | 311 | -0.000667 | 0.010110 | -0.370216 | 0.130065 | -0.002885 | 0.067923 |
| validation | 41393 | 468 | 0.001045 | 0.004039 | -0.079967 | 0.098263 | -0.001072 | 0.054933 |

## T+1 Fill Simulation: Best Rows By Split

| run_name | split | setting | net_ann | net_ir | net_mdd | win_rate | excess_benchmark_ann | excess_exec_universe_ann | avg_desired_turnover | avg_filled_turnover | avg_transaction_cost | avg_position_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | top_10_keep_1x | -0.112992 | -0.037093 | -0.380032 | 0.425532 | -0.010952 | 0.042897 | 1.693223 | 0.045700 | 0.000069 | 24.457447 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | top_10_keep_1.5x | -0.117272 | -0.039577 | -0.379075 | 0.425532 | -0.015293 | 0.038162 | 1.595016 | 0.043186 | 0.000065 | 23.606383 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | top_10_keep_2x | -0.125454 | -0.043988 | -0.379879 | 0.425532 | -0.024055 | 0.028872 | 1.498274 | 0.039912 | 0.000060 | 22.287234 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | top_10_keep_3x | -0.132092 | -0.046383 | -0.378609 | 0.436170 | -0.030105 | 0.022584 | 1.277264 | 0.032567 | 0.000049 | 19.489362 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | top_20_keep_1x | -0.163369 | -0.062679 | -0.409223 | 0.446809 | -0.062072 | -0.012227 | 1.473349 | 0.072229 | 0.000108 | 44.978723 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | top_30_keep_1x | -0.166754 | -0.064702 | -0.405505 | 0.425532 | -0.064603 | -0.014210 | 1.287288 | 0.087878 | 0.000132 | 64.127660 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | top_20_keep_1.5x | -0.168374 | -0.064055 | -0.411575 | 0.446809 | -0.066316 | -0.016340 | 1.246982 | 0.062241 | 0.000093 | 42.191489 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | top_20_keep_3x | -0.185819 | -0.073103 | -0.423058 | 0.425532 | -0.086000 | -0.036129 | 0.640873 | 0.035355 | 0.000053 | 29.819149 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | top_20_keep_1x | 0.547856 | 0.211174 | -0.216987 | 0.571429 | -0.080590 | 0.058175 | 1.532392 | 0.162937 | 0.000244 | 46.603175 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | top_20_keep_1.5x | 0.462410 | 0.225052 | -0.213150 | 0.587302 | -0.136875 | -0.009369 | 1.304556 | 0.135235 | 0.000203 | 40.190476 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | top_10_keep_3x | 0.431157 | 0.195363 | -0.199353 | 0.587302 | -0.151944 | -0.030800 | 1.446234 | 0.088877 | 0.000133 | 20.857143 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | top_10_keep_2x | 0.395717 | 0.174149 | -0.198465 | 0.555556 | -0.172455 | -0.053080 | 1.557181 | 0.103131 | 0.000155 | 23.190476 |

## Soft Optimizer Core80: Best Rows By Split

| run_name | split | risk_control | k | style_penalty | turnover_penalty | periods | net_ann | net_ir | net_max_drawdown | excess_benchmark_ann | excess_exec_universe_ann | avg_desired_turnover | avg_filled_turnover | avg_filled_desired_ratio | avg_position_count | avg_cash_weight | avg_invested_weight | optimal_rate | feasible_rate | solver_error_rate | fallback_rate | avg_max_abs_exposure_z | avg_abs_exposure_z | avg_max_exposure_slack | avg_exposure_slack | avg_total_exposure_slack | avg_buy_capacity_slack | avg_min_invested_shortfall | min_invested_rule_pass_rate | grid_run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | industry_size_liquidity_vol_mom | 20 | 0.100000 | 0.000000 | 94 | -0.162305 | -0.056938 | -0.450332 | -0.058332 | -0.115352 | 0.111985 | 0.206193 | 1.927331 | 63.787234 | 0.041445 | 0.958555 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 1.354834 | 0.390477 | 1.204834 | 0.254925 | 4.078805 | 0.030085 | 0.011694 | 0.978723 | 19 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | industry_size_liquidity_vol_mom | 30 | 0.100000 | 0.000000 | 94 | -0.162305 | -0.056938 | -0.450332 | -0.058332 | -0.115352 | 0.111985 | 0.206193 | 1.927331 | 63.787234 | 0.041445 | 0.958555 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 1.354834 | 0.390477 | 1.204834 | 0.254925 | 4.078805 | 0.030085 | 0.011694 | 0.978723 | 23 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | industry_size_liquidity_vol_mom | 10 | 0.100000 | 0.020000 | 94 | -0.163986 | -0.055449 | -0.453785 | -0.058197 | -0.114636 | 0.107608 | 0.107831 | 1.002726 | 61.361702 | 0.026403 | 0.973597 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 1.317802 | 0.385614 | 1.167802 | 0.249700 | 3.995194 | 0.030327 | 0.011694 | 0.978723 | 16 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | industry_size_liquidity_vol_mom | 30 | 0.100000 | 0.020000 | 94 | -0.168777 | -0.060876 | -0.447501 | -0.065985 | -0.122633 | 0.113536 | 0.113804 | 1.002923 | 65.031915 | 0.042089 | 0.957911 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 1.296541 | 0.374520 | 1.146541 | 0.238737 | 3.819786 | 0.030026 | 0.011694 | 0.978723 | 24 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | industry_size_liquidity_vol_mom | 20 | 0.100000 | 0.020000 | 94 | -0.168777 | -0.060876 | -0.447501 | -0.065985 | -0.122633 | 0.113536 | 0.113804 | 1.002923 | 65.031915 | 0.042089 | 0.957911 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 1.296541 | 0.374520 | 1.146541 | 0.238737 | 3.819786 | 0.030026 | 0.011694 | 0.978723 | 20 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | industry_size_liquidity_vol_mom | 10 | 0.100000 | 0.000000 | 94 | -0.175212 | -0.061014 | -0.455123 | -0.070632 | -0.126167 | 0.108580 | 0.158502 | 1.500766 | 61.521277 | 0.023638 | 0.976362 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 1.363415 | 0.399557 | 1.213415 | 0.262738 | 4.203812 | 0.031353 | 0.011694 | 0.978723 | 15 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | industry_size_liquidity_vol_mom | 30 | 0.000000 | 0.000000 | 94 | -0.184335 | -0.065075 | -0.465827 | -0.080166 | -0.136019 | 0.114388 | 0.207374 | 1.888196 | 65.148936 | 0.029296 | 0.970704 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 1.349040 | 0.382683 | 1.199040 | 0.245876 | 3.934013 | 0.031174 | 0.011694 | 0.978723 | 21 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | industry_size_liquidity_vol_mom | 20 | 0.000000 | 0.000000 | 94 | -0.184335 | -0.065075 | -0.465827 | -0.080166 | -0.136019 | 0.114388 | 0.207374 | 1.888196 | 65.148936 | 0.029296 | 0.970704 | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 1.349040 | 0.382683 | 1.199040 | 0.245876 | 3.934013 | 0.031174 | 0.011694 | 0.978723 | 17 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | none | 10 | 0.000000 | 0.020000 | 63 | 0.516370 | 0.269538 | -0.170611 | -0.105942 | 0.158297 | 0.227402 | 0.227074 | 0.997994 | 60.873016 | 0.030697 | 0.969303 | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000877 | 0.006575 | 0.984127 | 2 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | none | 10 | 0.100000 | 0.020000 | 63 | 0.516370 | 0.269538 | -0.170611 | -0.105942 | 0.158297 | 0.227402 | 0.227074 | 0.997994 | 60.873016 | 0.030697 | 0.969303 | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000877 | 0.006575 | 0.984127 | 4 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | none | 10 | 0.000000 | 0.000000 | 63 | 0.506509 | 0.264126 | -0.169664 | -0.111908 | 0.151003 | 0.237010 | 0.325798 | 1.406177 | 60.714286 | 0.033829 | 0.966171 | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000877 | 0.006575 | 0.984127 | 1 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | none | 10 | 0.100000 | 0.000000 | 63 | 0.506509 | 0.264126 | -0.169664 | -0.111908 | 0.151003 | 0.237010 | 0.325798 | 1.406177 | 60.714286 | 0.033829 | 0.966171 | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000877 | 0.006575 | 0.984127 | 3 |

## T+1 Comparator Snapshot

| run_name | split | setting | net_ann | net_ir | net_mdd | win_rate | excess_benchmark_ann | excess_exec_universe_ann | avg_desired_turnover | avg_filled_turnover | avg_transaction_cost | avg_position_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | top_20_keep_1x | 0.547856 | 0.211174 | -0.216987 | 0.571429 | -0.080590 | 0.058175 | 1.532392 | 0.162937 | 0.000244 | 46.603175 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | test | top_20_keep_1x | 0.517426 | 0.185555 | -0.230151 | 0.555556 | -0.098827 | 0.042323 | 1.557832 | 0.158189 | 0.000237 | 46.428571 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | top_20_keep_1.5x | 0.462410 | 0.225052 | -0.213150 | 0.587302 | -0.136875 | -0.009369 | 1.304556 | 0.135235 | 0.000203 | 40.190476 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | test | top_10_keep_3x | 0.431157 | 0.195363 | -0.199353 | 0.587302 | -0.151944 | -0.030800 | 1.446234 | 0.088877 | 0.000133 | 20.857143 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | test | top_30_keep_3x | 0.419965 | 0.218337 | -0.193181 | 0.619048 | -0.163007 | -0.039055 | 0.347642 | 0.037810 | 0.000057 | 35.539683 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | test | top_20_keep_1.5x | 0.404619 | 0.187749 | -0.221771 | 0.555556 | -0.170765 | -0.044352 | 1.277010 | 0.127975 | 0.000192 | 40.428571 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_10_keep_1x | -0.021363 | 0.008417 | -0.317593 | 0.478723 | 0.089230 | 0.146947 | 1.742582 | 0.041163 | 0.000062 | 25.712766 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_10_keep_1.5x | -0.022683 | 0.007755 | -0.316418 | 0.446809 | 0.088060 | 0.145787 | 1.611252 | 0.038055 | 0.000057 | 24.840426 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_10_keep_2x | -0.039490 | -0.001072 | -0.320080 | 0.446809 | 0.069252 | 0.126089 | 1.505345 | 0.031274 | 0.000047 | 23.712766 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_10_keep_3x | -0.057975 | -0.010657 | -0.325067 | 0.446809 | 0.049046 | 0.105156 | 1.278383 | 0.026397 | 0.000040 | 21.457447 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_clean | validation | top_20_keep_1x | -0.082060 | -0.019902 | -0.364615 | 0.436170 | 0.026289 | 0.081775 | 1.558939 | 0.068839 | 0.000103 | 48.968085 |
| feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean | validation | top_10_keep_1x | -0.112992 | -0.037093 | -0.380032 | 0.425532 | -0.010952 | 0.042897 | 1.693223 | 0.045700 | 0.000069 | 24.457447 |

## Decision

- Do not promote as production mainline on the current evidence.
- Keep as a candidate architecture because the TopK wide-band loss improves test T+1 absolute return and Top10 robustness, but require a validation-positive rerun or rolling-window confirmation.
- Next useful experiment: keep architecture fixed and test stronger validation alignment, for example lower model capacity or add explicit excess-return/benchmark-relative selection in the loss/evaluation gate.