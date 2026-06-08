# Results And Limitations / 结果与局限

## English

This document summarizes the final frozen result and its limitations. It is a report note; the root `README.md` remains the reproduction entry point.

### Final Mainline

```text
model: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
optimizer: risk_control=none, k=10, style_penalty=0.1, turnover_penalty=0.0, min_invested=0.8
```

### Main Evidence Paths

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/
outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/
outputs/backtest/optimizer/final_mainline_ckptscore_e12/
outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution/
outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_closed_loop/
outputs/audit/barra_lite_residual_alpha/
outputs/audit/point_in_time_canonical_labels/
```

These are generated local artifacts and are not committed to the repository.

### Headline Metrics

| split | net_ann | net_ir | excess_benchmark_ann | excess_exec_universe_ann | avg_invested_weight | min_invested_rule_pass_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | -0.059635 | -0.013411 | 0.047074 | -0.028238 | 0.839216 | 0.978723 |
| test | 0.268252 | 0.189517 | -0.258701 | -0.035435 | 0.893154 | 0.984127 |

### Limitations

- The final optimizer does not produce positive excess versus the executable universe in both validation and test.
- Test absolute return is strong, but benchmark-relative performance is weak.
- The final strategy should not be described as a production-ready stable alpha.
- The value of the final mainline is the complete reproducible research loop: clean data, model selection, T+1 constraints, capacity rules, optimizer evidence, and audits.

## 中文

本文总结最终冻结结果和局限。它是报告说明文档；根目录 `README.md` 仍是复现入口。

### 最终主线

```text
model: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
optimizer: risk_control=none, k=10, style_penalty=0.1, turnover_penalty=0.0, min_invested=0.8
```

### 主要证据路径

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/
outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/
outputs/backtest/optimizer/final_mainline_ckptscore_e12/
outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution/
outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_closed_loop/
outputs/audit/barra_lite_residual_alpha/
outputs/audit/point_in_time_canonical_labels/
```

这些都是本地生成物，不提交到仓库。

### 核心指标

| split | net_ann | net_ir | excess_benchmark_ann | excess_exec_universe_ann | avg_invested_weight | min_invested_rule_pass_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | -0.059635 | -0.013411 | 0.047074 | -0.028238 | 0.839216 | 0.978723 |
| test | 0.268252 | 0.189517 | -0.258701 | -0.035435 | 0.893154 | 0.984127 |

### 局限

- 最终 optimizer 没有在 validation 和 test 上同时取得相对可执行域正超额。
- test 绝对收益较强，但相对 benchmark 表现偏弱。
- 不应将最终策略描述为生产就绪的稳定 alpha。
- 最终主线的价值在于完成可复现研究闭环：clean data、模型选择、T+1 约束、容量规则、optimizer 证据和审计。
