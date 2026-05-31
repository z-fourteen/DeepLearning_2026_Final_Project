# 结果与局限

最终复核产物集中在：

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/
outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/
outputs/backtest/optimizer/final_mainline_ckptscore_e12/
outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution/
outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_closed_loop/
outputs/audit/barra_lite_residual_alpha/
outputs/audit/point_in_time_canonical_labels/
```

最终主线：

```text
model: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
optimizer: risk_control=none, k=10, style_penalty=0.1, turnover_penalty=0.0, min_invested=0.8
```

关键读数：

| split | net_ann | net_ir | excess_benchmark_ann | excess_exec_universe_ann | avg_invested_weight | min_invested_rule_pass_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | -0.059635 | -0.013411 | 0.047074 | -0.028238 | 0.839216 | 0.978723 |
| test | 0.268252 | 0.189517 | -0.258701 | -0.035435 | 0.893154 | 0.984127 |

局限：最终 optimizer 在 validation/test 上均未取得相对可执行域正超额；它的价值主要在于完成了 clean 数据、模型选择、T+1 约束、容量规则和报告证据链的闭环。报告中不应把它描述为稳定生产 alpha，只能描述为最终提交主线和可复核研究结果。

旧 L20 wide-band feature-style interaction GRU 记录仍保留在：

```text
docs/08_feature_style_interaction_wide30_closed_loop.md
outputs/analysis/feature_style_interaction_gru_l20_topk10_wide30_clean_closed_loop/
```

已知局限：

历史 `full62` score 在测试期产生过较强结果，但其风格 regime 依赖和 residual-alpha 稳定性不足以支撑最终面向生产的主线。因此它被归档，而不是删除。

最终冻结记录见：

```text
docs/09_final_mainline_freeze.md
```
