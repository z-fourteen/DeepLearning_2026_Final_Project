# Full62 Attribution Audit

Scope: old full62 GRU score model with T+1 fill-simulation candidates.

## Execution Attribution

| candidate | split | periods | gross_mean | net_mean | cost_drag_mean | net_ir | excess_benchmark_mean | excess_executable_universe_mean | desired_turnover | filled_turnover | fill_ratio | avg_buy_reject | avg_sell_reject | avg_partial_fill | avg_positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k10_keep1.5 | test | 66 | 0.011938 | 0.011724 | 0.000214 | 0.267490 | 0.002019 | 0.003974 | 1.662486 | 0.142826 | 0.085911 | 0.015152 | 1.500000 | 23.242424 | 25.000000 |
| k30_keep3 | test | 66 | 0.010632 | 0.010570 | 0.000062 | 0.304255 | 0.000865 | 0.002820 | 0.305066 | 0.041279 | 0.135311 | 0.015152 | 4.409091 | 11.954545 | 34.590909 |
| k30_keep1 | test | 66 | 0.009628 | 0.009298 | 0.000330 | 0.255150 | -0.000407 | 0.001549 | 1.343890 | 0.220053 | 0.163744 | 0.030303 | 7.833333 | 53.318182 | 64.590909 |
| k30_keep3 | validation | 97 | -0.002353 | -0.002387 | 0.000035 | -0.047226 | -0.003294 | -0.001717 | 0.250957 | 0.023238 | 0.092596 | 0.020619 | 2.618557 | 12.783505 | 33.123711 |
| k30_keep1 | validation | 97 | -0.003007 | -0.003164 | 0.000156 | -0.059568 | -0.004071 | -0.002493 | 1.415092 | 0.104329 | 0.073726 | 0.020619 | 9.082474 | 59.371134 | 70.340206 |
| k10_keep1.5 | validation | 97 | -0.004348 | -0.004418 | 0.000070 | -0.083882 | -0.005325 | -0.003747 | 1.679360 | 0.046814 | 0.027876 | 0.010309 | 1.525773 | 23.195876 | 24.865979 |

## Largest Style And Liquidity Exposures

| candidate | split | feature | selected_mean | universe_mean | selected_minus_universe | diff_ir |
| --- | --- | --- | --- | --- | --- | --- |
| k10_keep1.5 | test | lag1_amount_60d_mean | 2294756.797423 | 1705618.839180 | 589137.958242 | 0.395533 |
| k10_keep1.5 | test | lag1_amount_20d_mean | 2193328.895442 | 1763026.307717 | 430302.587725 | 0.274121 |
| k10_keep1.5 | test | lag1_turnover_rate_f | 3.295576 | 4.368047 | -1.072471 | -1.020463 |
| k10_keep1.5 | test | lag1_turnover_rate | 2.305554 | 3.138565 | -0.833012 | -1.039043 |
| k10_keep1.5 | test | lag1_turnover_20d_mean | 2.350381 | 3.059068 | -0.708688 | -1.093996 |
| k10_keep1.5 | test | lag1_turnover_60d_mean | 2.614811 | 3.071993 | -0.457182 | -0.706232 |
| k10_keep1.5 | test | lag1_log_circ_mv | 15.290464 | 15.068688 | 0.221776 | 0.577485 |
| k10_keep1.5 | test | lag1_log_total_mv | 15.538055 | 15.323296 | 0.214760 | 0.581597 |
| k30_keep1 | test | lag1_amount_60d_mean | 1977071.873710 | 1705618.839180 | 271453.034530 | 0.453185 |
| k30_keep1 | test | lag1_amount_20d_mean | 1884315.740987 | 1763026.307717 | 121289.433270 | 0.191064 |
| k30_keep1 | test | lag1_turnover_rate_f | 3.553249 | 4.368047 | -0.814797 | -1.145524 |
| k30_keep1 | test | lag1_turnover_rate | 2.537938 | 3.138565 | -0.600628 | -1.070151 |
| k30_keep1 | test | lag1_turnover_20d_mean | 2.615883 | 3.059068 | -0.443185 | -1.101905 |
| k30_keep1 | test | lag1_turnover_60d_mean | 2.872823 | 3.071993 | -0.199170 | -0.493290 |
| k30_keep1 | test | lag1_vol_log | 11.926848 | 12.082918 | -0.156070 | -0.882354 |
| k30_keep1 | test | lag1_amount_log | 13.462667 | 13.565054 | -0.102387 | -0.430928 |
| k30_keep3 | test | lag1_amount_60d_mean | 2227004.372135 | 1705618.839180 | 521385.532954 | 2.185088 |
| k30_keep3 | test | lag1_amount_20d_mean | 2280493.095721 | 1763026.307717 | 517466.788004 | 1.634014 |
| k30_keep3 | test | lag1_turnover_rate_f | 3.764544 | 4.368047 | -0.603502 | -1.131396 |
| k30_keep3 | test | lag1_turnover_20d_mean | 2.711005 | 3.059068 | -0.348063 | -1.131642 |
| k30_keep3 | test | lag1_turnover_60d_mean | 2.737149 | 3.071993 | -0.334844 | -1.476012 |
| k30_keep3 | test | lag1_turnover_rate | 2.828616 | 3.138565 | -0.309949 | -0.856459 |
| k30_keep3 | test | lag1_log_circ_mv | 15.284045 | 15.068688 | 0.215357 | 3.588014 |
| k30_keep3 | test | lag1_log_total_mv | 15.489458 | 15.323296 | 0.166162 | 3.123096 |
| k10_keep1.5 | validation | lag1_amount_60d_mean | 700334.650405 | 621728.567396 | 78606.083009 | 0.219958 |
| k10_keep1.5 | validation | lag1_amount_20d_mean | 703853.950677 | 651399.084483 | 52454.866195 | 0.106056 |
| k10_keep1.5 | validation | lag1_turnover_rate_f | 2.167865 | 2.581893 | -0.414028 | -0.576419 |
| k10_keep1.5 | validation | lag1_turnover_rate | 1.564647 | 1.873543 | -0.308896 | -0.576865 |
| k10_keep1.5 | validation | lag1_turnover_20d_mean | 1.590376 | 1.861805 | -0.271429 | -0.541979 |
| k10_keep1.5 | validation | lag1_turnover_60d_mean | 1.660385 | 1.805004 | -0.144619 | -0.307648 |
| k10_keep1.5 | validation | lag1_log_circ_mv | 14.831507 | 14.693519 | 0.137988 | 0.412940 |
| k10_keep1.5 | validation | lag1_log_total_mv | 15.130140 | 14.995892 | 0.134248 | 0.422005 |
| k30_keep1 | validation | lag1_amount_60d_mean | 645725.028102 | 621728.567396 | 23996.460707 | 0.137619 |
| k30_keep1 | validation | lag1_amount_20d_mean | 633112.991358 | 651399.084483 | -18286.093125 | -0.072686 |
| k30_keep1 | validation | lag1_turnover_rate_f | 2.161647 | 2.581893 | -0.420246 | -0.874024 |
| k30_keep1 | validation | lag1_turnover_rate | 1.563703 | 1.873543 | -0.309840 | -0.833814 |
| k30_keep1 | validation | lag1_turnover_20d_mean | 1.593606 | 1.861805 | -0.268199 | -0.914933 |
| k30_keep1 | validation | lag1_turnover_60d_mean | 1.666107 | 1.805004 | -0.138897 | -0.546799 |
| k30_keep1 | validation | lag1_vol_log | 11.272862 | 11.387130 | -0.114268 | -0.598890 |
| k30_keep1 | validation | lag1_amount_log | 12.586791 | 12.663006 | -0.076216 | -0.352814 |
| k30_keep3 | validation | lag1_amount_60d_mean | 741490.254546 | 621728.567396 | 119761.687151 | 1.722810 |
| k30_keep3 | validation | lag1_amount_20d_mean | 751925.863725 | 651399.084483 | 100526.779242 | 0.936942 |
| k30_keep3 | validation | lag1_turnover_rate_f | 2.141191 | 2.581893 | -0.440703 | -1.163136 |
| k30_keep3 | validation | lag1_turnover_20d_mean | 1.463859 | 1.861805 | -0.397946 | -1.441268 |
| k30_keep3 | validation | lag1_turnover_rate | 1.482277 | 1.873543 | -0.391266 | -1.241212 |
| k30_keep3 | validation | lag1_turnover_60d_mean | 1.431765 | 1.805004 | -0.373239 | -2.094570 |
| k30_keep3 | validation | lag1_log_circ_mv | 15.030876 | 14.693519 | 0.337358 | 4.750898 |
| k30_keep3 | validation | lag1_log_total_mv | 15.275449 | 14.995892 | 0.279557 | 4.151966 |

## Liquidity Bucket Returns

| split | bucket_feature | bucket | rows | mean_exec_return | return_ir | buy_executable_rate | limit_up_rate | limit_down_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | lag1_amount_20d_mean | low | 9504 | 0.002826 | 0.118951 | 0.999474 | 0.000526 | 0.000000 |
| test | lag1_amount_20d_mean | mid | 9365 | 0.009575 | 0.312764 | 0.998505 | 0.001495 | 0.000214 |
| test | lag1_amount_20d_mean | high | 9546 | 0.012447 | 0.285445 | 0.996962 | 0.003038 | 0.001048 |
| test | lag1_amount_rank_pct | low | 9489 | 0.003409 | 0.143135 | 0.999473 | 0.000527 | 0.000000 |
| test | lag1_amount_rank_pct | mid | 9379 | 0.008542 | 0.269677 | 0.999040 | 0.000960 | 0.000640 |
| test | lag1_amount_rank_pct | high | 9547 | 0.012871 | 0.301525 | 0.996439 | 0.003561 | 0.000628 |
| test | lag1_log_circ_mv | low | 9547 | 0.005663 | 0.222710 | 0.999057 | 0.000943 | 0.000105 |
| test | lag1_log_circ_mv | mid | 9378 | 0.009174 | 0.269051 | 0.998294 | 0.001706 | 0.000427 |
| test | lag1_log_circ_mv | high | 9490 | 0.010036 | 0.268684 | 0.997576 | 0.002424 | 0.000738 |
| test | lag1_turnover_rate_f | low | 9512 | 0.003134 | 0.134397 | 0.999579 | 0.000421 | 0.000105 |
| test | lag1_turnover_rate_f | mid | 9369 | 0.009072 | 0.288932 | 0.999039 | 0.000961 | 0.000320 |
| test | lag1_turnover_rate_f | high | 9534 | 0.012643 | 0.297109 | 0.996329 | 0.003671 | 0.000839 |
| test | lag1_ret_20d_std | low | 9524 | 0.003883 | 0.166029 | 0.999160 | 0.000840 | 0.000105 |
| test | lag1_ret_20d_std | mid | 9358 | 0.007617 | 0.241755 | 0.998825 | 0.001175 | 0.000321 |
| test | lag1_ret_20d_std | high | 9533 | 0.013331 | 0.308810 | 0.996958 | 0.003042 | 0.000839 |
| validation | lag1_amount_20d_mean | low | 14654 | -0.001156 | -0.026142 | 0.999181 | 0.000819 | 0.000068 |
| validation | lag1_amount_20d_mean | mid | 14437 | -0.001651 | -0.034965 | 0.997991 | 0.002009 | 0.000000 |
| validation | lag1_amount_20d_mean | high | 14712 | -0.000407 | -0.009187 | 0.996262 | 0.003738 | 0.000272 |
| validation | lag1_amount_rank_pct | low | 14583 | -0.001286 | -0.028388 | 0.999246 | 0.000754 | 0.000069 |
| validation | lag1_amount_rank_pct | mid | 14485 | -0.001615 | -0.033425 | 0.998343 | 0.001657 | 0.000069 |
| validation | lag1_amount_rank_pct | high | 14735 | -0.000314 | -0.008471 | 0.995860 | 0.004140 | 0.000204 |
| validation | lag1_log_circ_mv | low | 14735 | -0.000720 | -0.016793 | 0.998575 | 0.001425 | 0.000136 |
| validation | lag1_log_circ_mv | mid | 14479 | -0.000610 | -0.014377 | 0.997928 | 0.002072 | 0.000069 |
| validation | lag1_log_circ_mv | high | 14589 | -0.001873 | -0.036450 | 0.996915 | 0.003085 | 0.000137 |
| validation | lag1_turnover_rate_f | low | 14656 | -0.002243 | -0.052468 | 0.999318 | 0.000682 | 0.000000 |
| validation | lag1_turnover_rate_f | mid | 14456 | -0.001764 | -0.032957 | 0.998409 | 0.001591 | 0.000208 |
| validation | lag1_turnover_rate_f | high | 14691 | 0.000790 | 0.008897 | 0.995712 | 0.004288 | 0.000136 |
| validation | lag1_ret_20d_std | low | 14697 | -0.001108 | -0.025681 | 0.999252 | 0.000748 | 0.000068 |
| validation | lag1_ret_20d_std | mid | 14429 | -0.001334 | -0.026619 | 0.997852 | 0.002148 | 0.000000 |
| validation | lag1_ret_20d_std | high | 14677 | -0.000766 | -0.016337 | 0.996321 | 0.003679 | 0.000273 |