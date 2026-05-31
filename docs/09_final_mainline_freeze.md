# 最终主线冻结记录与结果分析

冻结日期：2026-05-31

本文记录最终提交主线的冻结口径、复现入口、证据路径和完整结果分析。该主线应被表述为“可复核研究闭环与最终提交口径”，不应被描述为已经生产就绪的稳定 alpha。

## 冻结结论

最终主线模型：

```text
feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
```

模型选择：

```text
checkpoint: epoch 12
selection metric: checkpoint_score
best_metric: 0.05160820540933453
stop_reason: metric_early_stop:checkpoint_score
```

最终 optimizer：

```text
risk_control: none
k: 10
style_penalty: 0.1
turnover_penalty: 0.0
min_invested: 0.8
```

`style_penalty=0.1` 与 `style_penalty=0.0` 在 epoch-12 validation summary 中并列；冻结时选择 `0.1` 作为最终口径。旧的 `feature_style_interaction_gru_l60_wide30_validation_attribution` 指向非 epoch-12 的 `core80` 产物，不再作为最终选择证据。

## 单一入口

```text
configs/portfolio/final_mainline_optimizer.yaml
scripts/portfolio/run_final_mainline_optimizer.py
```

复现命令：

```powershell
conda run -n dl_env python scripts/portfolio/run_final_mainline_optimizer.py
```

## 证据路径

训练证据：

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/config.yaml
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/metrics.json
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/model.pt
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet
```

optimizer grid：

```text
outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/
```

validation attribution：

```text
outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution/
```

final wrapper 输出：

```text
outputs/backtest/optimizer/final_mainline_ckptscore_e12/
```

closed-loop 摘要：

```text
outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_closed_loop/
```

## 数据与模型规模

| 项 | 数值 |
| --- | ---: |
| lookback | 60 |
| num_features | 18 |
| train samples | 114,946 |
| validation samples | 40,624 |
| test samples | 26,678 |
| validation dates | 468 |
| batch size | 384 |
| batch mode | date |
| prediction rows | 67,302 |
| device | cuda |
| model | feature_style_interaction_gru |

该模型使用 L60 clean alpha-resid-style 序列张量，训练目标为 `topk_band_margin_ic`，checkpoint 选择使用 `checkpoint_score`，而不是单纯使用 `val_loss` 或 `rank_ic_mean`。

## 训练结果

最终选中 epoch 12：

| 指标 | 数值 |
| --- | ---: |
| train_loss | 0.064264 |
| val_loss | 0.078966 |
| ic_mean | 0.030695 |
| icir | 0.218859 |
| rank_ic_mean | 0.032550 |
| rank_icir | 0.231031 |
| checkpoint_score | 0.051608 |
| topk_proxy_mean | -0.000471 |
| pred_std | 0.010875 |
| daily_coverage_ratio | 1.000000 |
| checkpoint_eligible | 1 |

训练曲线的主要特征：

- epoch 1 到 8：模型已经有排序能力，但 score dispersion 较低，`pred_std` 约在 0.0015 到 0.0027。
- epoch 9 开始：`dispersion_floor_ratio=1.0`，预测分布打开，模型不再是窄幅打分器。
- epoch 12：`checkpoint_score=0.051608`，为整个训练过程最高，因此冻结为最终 checkpoint。
- epoch 13 以后：train loss 继续下降，但 rank IC 和 checkpoint score 明显退化，继续训练更像过拟合，而不是泛化提升。
- epoch 16 的 `val_loss=0.075897` 更低，但 `checkpoint_score=0.034651`，因此按执行相关排序目标不应选择 epoch 16。

训练层面的结论是：模型在 validation 上学到了中等强度的横截面排序信号，但不是压倒性的强 alpha。

## 预测分布

| split | rows | dates | score_mean | score_std | score_min | score_max | label_mean | label_std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 40,624 | 468 | 0.000809 | 0.010875 | -0.194632 | 0.077087 | -0.001073 | 0.054722 |
| test | 26,678 | 311 | -0.002766 | 0.012977 | -0.172432 | 0.067353 | -0.002935 | 0.067726 |

解读：

- score 标准差显著小于 label 标准差，因此模型输出更适合做排序，不应解释为收益幅度预测。
- validation 和 test 的 label 均值都为负，说明样本环境本身不是轻松赚钱环境。
- test 的 label 波动更大，`label_std=0.0677` 高于 validation 的 `0.0547`，会放大组合收益的 regime 差异。
- `valid_pred_ratio=1.0`，`pred_constant_daily_count=0`，说明预测输出合同健康，没有出现 collapse。

## 裸 T+1 TopK 执行结果

裸 T+1 仿真不是最终 optimizer，但可以观察模型排序信号在交易约束下的原始表现。

validation 上所有 TopK/keep 设置的净年化均为负：

| validation 较优设置 | net_ann | net_ir | max_drawdown | win_rate | excess_benchmark_ann | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_10_keep_1.5x | -0.147331 | -0.073154 | -0.364982 | 0.382979 | -0.057730 | -0.013408 |
| top_10_keep_2x | -0.147857 | -0.071794 | -0.368843 | 0.382979 | -0.056733 | -0.012154 |
| top_30_keep_1.5x | -0.152696 | -0.061664 | -0.386409 | 0.404255 | -0.052141 | -0.003156 |

validation 的重点不是绝对收益差，而是相对可执行域也没有稳定转正。最接近转正的是 `top_30_keep_1.5x`，相对可执行域超额为 -0.3156%，但净年化仍是 -15.27%。

test 上裸 T+1 较强：

| test 较优设置 | net_ann | net_ir | max_drawdown | win_rate | excess_benchmark_ann | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_20_keep_1.5x | 0.549093 | 0.263183 | -0.170358 | 0.555556 | -0.087406 | 0.052450 |
| top_30_keep_1x | 0.508440 | 0.272589 | -0.184987 | 0.571429 | -0.111191 | 0.021582 |
| top_10_keep_3x | 0.410864 | 0.181112 | -0.200435 | 0.555556 | -0.166395 | -0.036879 |

test 期绝对收益强，但很多设置仍然跑输 benchmark。真正相对可执行域为正的主要是 `top_20_keep_1.5x` 和 `top_30_keep_1x`。这说明 test 期模型有选股价值，但一部分绝对收益来自市场环境，而不是纯 alpha。

## 最终 optimizer 结果

最终冻结 optimizer 结果：

| split | periods | net_ann | net_ir | net_max_drawdown | excess_benchmark_ann | excess_exec_universe_ann |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 94 | -0.059635 | -0.013411 | -0.313069 | 0.047074 | -0.028238 |
| test | 63 | 0.268252 | 0.189517 | -0.145907 | -0.258701 | -0.035435 |

分层结论：

- 相比裸 T+1，optimizer 明显改善 validation 净收益，从约 -14% 到 -18% 年化改善到 -5.96%。
- optimizer 仍未实现相对可执行域正超额：validation 为 -2.82%，test 为 -3.54%。
- test 净年化为 +26.83%，但 benchmark excess 为 -25.87%，说明 test 期基准很强，最终组合没有跑赢 benchmark。

## optimizer 约束健康度

| split | avg_cash_weight | avg_invested_weight | avg_desired_turnover | avg_filled_turnover | avg_position_count |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.160784 | 0.839216 | 0.123194 | 0.153775 | 65.77 |
| test | 0.106846 | 0.893154 | 0.275793 | 0.344442 | 55.54 |

| split | optimal_rate | feasible_rate | solver_error_rate | fallback_rate | min_invested_rule_pass_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.978723 | 0.978723 | 0.000000 | 0.021277 | 0.978723 |
| test | 0.984127 | 0.984127 | 0.000000 | 0.015873 | 0.984127 |

| split | avg_buy_capacity_slack | avg_min_invested_shortfall | avg_max_abs_exposure_z | avg_abs_exposure_z | avg_total_exposure_slack |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.003168 | 0.011731 | 0.000000 | 0.000000 | 0.000000 |
| test | 0.000892 | 0.006580 | 0.000000 | 0.000000 | 0.000000 |

解读：

- solver 稳定，没有 solver error。
- validation 有 2 个 period 触发 `MinInvestedUnreachable` fallback，test 有 1 个。
- 最低仓位 shortfall 不是大面积问题，而是少数 period 拉高均值。
- `risk_control=none` 下，风格暴露和 exposure slack 指标为 0 是预期结果，不代表风格风险消失，只代表最终 optimizer 没启用风格约束。
- 因为 `risk_control=none`，`style_penalty=0.0` 与 `style_penalty=0.1` 在最终组合中等价。

## validation 选择依据

epoch-12 optimizer grid 的 validation 前列为：

| rank | risk_control | k | style_penalty | turnover_penalty | net_ann | excess_exec_universe_ann |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | none | 10 | 0.0 | 0.0 | -0.059635 | -0.028238 |
| 2 | none | 10 | 0.1 | 0.0 | -0.059635 | -0.028238 |
| 3 | none | 10 | 0.0 | 0.02 | -0.061213 | -0.030015 |
| 4 | none | 10 | 0.1 | 0.02 | -0.061213 | -0.030015 |

因此：

- `turnover_penalty=0.0` 比 `0.02` 略优。
- `k=10` 比 `k=20/30` 在 validation 上更优。
- `industry_size_liquidity_vol_mom` 风险控制在 validation 上显著更差。
- `style_penalty=0.1` 与 `0.0` 并列，因为最终风险控制为 `none`。

test 上存在更高收益的设置，例如 `none, k=20/30, turnover_penalty=0.02` 可达到约 35.83% 净年化，并有约 +3.54% 相对可执行域超额。但这是 test 后验观察，不能替代 validation 选择。

## 月度与 period 结构

validation 月度表现不是平滑亏损，而是少数月份冲击明显。

较差 validation 月份：

| month | net_mean | excess_exec_mean | invested_mean |
| --- | ---: | ---: | ---: |
| 202304 | -0.027754 | -0.016638 | 0.808347 |
| 202412 | -0.027163 | 0.007640 | 0.977749 |
| 202401 | -0.021225 | 0.047833 | 0.804921 |
| 202406 | -0.018194 | -0.001649 | 0.817990 |

较好 validation 月份：

| month | net_mean | excess_exec_mean | invested_mean |
| --- | ---: | ---: | ---: |
| 202409 | 0.058925 | 0.025980 | 0.885608 |
| 202402 | 0.051277 | -0.031119 | 0.800000 |
| 202410 | 0.014186 | -0.009174 | 0.975873 |
| 202411 | 0.012164 | -0.009365 | 0.991955 |

test 月份：

| month | net_mean | excess_exec_mean |
| --- | ---: | ---: |
| 202508 | 0.030504 | 0.026413 |
| 202604 | 0.032926 | 0.012914 |
| 202507 | 0.015649 | -0.000991 |
| 202503 | -0.012532 | -0.010099 |
| 202605 | -0.012082 | 0.006776 |

最差 period：

| split | trade_date | net_return | excess_vs_benchmark | excess_vs_executable_universe |
| --- | ---: | ---: | ---: | ---: |
| validation | 20230420 | -0.086136 | -0.050234 | -0.043951 |
| validation | 20241227 | -0.080930 | 0.003057 | 0.006116 |
| test | 20250402 | -0.073001 | 0.012469 | -0.010727 |
| validation | 20230620 | -0.061094 | -0.025508 | -0.052469 |

最佳 period：

| split | trade_date | net_return | excess_vs_benchmark | excess_vs_executable_universe |
| --- | ---: | ---: | ---: | ---: |
| validation | 20240920 | 0.189450 | -0.036442 | 0.046722 |
| validation | 20240205 | 0.103748 | -0.026493 | -0.055221 |
| validation | 20241101 | 0.096151 | 0.002600 | 0.004060 |
| test | 20260429 | 0.081734 | 0.021027 | 0.029055 |

收益分布具有明显尾部驱动：少数大涨和大跌 period 对总结果影响很大。有些大幅正收益 period 仍跑输 benchmark，说明基准弹性很强。

## 暴露与约束相关性

最终 `risk_control=none`，因此风格暴露相关性为空。可读的约束相关性为：

| 指标 | 与 excess_exec_universe 的相关 |
| --- | ---: |
| cash_weight | -0.056862 |
| buy_capacity_slack | -0.075799 |
| filled_turnover | -0.049165 |
| position_count | -0.054362 |

这些相关性都很弱，说明 validation 的相对可执行域亏损不能简单归因于现金过多、换手过高、持仓数量过多或容量 slack。主要问题仍然是选股收益在 validation 上不够强。

## 可写结论

可以写：

- 模型层面：epoch 12 在 validation `checkpoint_score` 上最优，rank IC mean 为 0.03255，rank ICIR 为 0.23103，说明存在中等强度排序信号。
- 执行层面：裸 T+1 在 test 上表现强，最优 test T+1 设置达到 54.91% 净年化，并有 +5.25% 相对可执行域超额。
- 冻结 optimizer 层面：按 validation 选择后，最终方案 validation 净年化 -5.96%，test 净年化 26.83%，但相对可执行域分别为 -2.82% 和 -3.54%。
- 工程层面：证据链完整，覆盖训练 metrics、prediction diagnostics、T+1 execution、soft optimizer grid、final wrapper、validation attribution、约束健康度。

不能写：

- 不能说它是稳定生产 alpha。
- 不能说它稳定跑赢 benchmark。
- 不能用 test 上更高收益的 k=20/30 设置替代最终参数，除非明确说明那是 test 后验观察。
- 不能把 `style_penalty=0.1` 解释成实际发挥了风格惩罚作用，因为 `risk_control=none` 时它与 `0.0` 等价。

## 总评

这条主线是一个训练选择合理、执行链完整、test 期有明显收益潜力，但 validation 和相对可执行域证据仍偏弱的最终提交方案。它适合写成可复核研究闭环，不适合包装成已经生产就绪的稳定选股策略。
