# Feature-Style Interaction Wide30 Closed Loop / Feature-Style Interaction Wide30 闭环

## English

This document records a historical L20 wide30 candidate. It is useful for understanding model evolution, but it is not the final frozen mainline.

### Candidate

```text
feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean
```

### Evidence

| Component | Path |
| --- | --- |
| training output | `outputs/runs/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean/` |
| T+1 output | `outputs/backtest/t1_fill_sim/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean/` |
| optimizer output | `outputs/backtest/optimizer/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean_core80/` |
| closed-loop analysis | `outputs/analysis/feature_style_interaction_gru_l20_topk10_wide30_clean_closed_loop/` |

Generated evidence paths are local artifacts and are not committed.

### Key Readout

| Metric | Value |
| --- | ---: |
| selected validation rank IC mean | 0.040001 |
| selected validation rank ICIR | 0.261220 |
| selected validation IC mean | 0.020936 |
| prediction rows | 68,681 |

Best T+1 rows:

| split | setting | net_ann | net_ir | excess_exec_universe_ann |
| --- | --- | ---: | ---: | ---: |
| validation | top_10_keep_1x | -0.112992 | -0.037093 | 0.042897 |
| test | top_20_keep_1x | 0.547856 | 0.211174 | 0.058175 |

The candidate had promising test-period behavior, but validation/test divergence remained too large. It was retained as a useful research branch and not promoted as the final mainline.

## 中文

本文记录一个历史 L20 wide30 候选模型。它有助于理解模型演进，但不是最终冻结主线。

### 候选模型

```text
feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean
```

### 证据路径

| 组件 | 路径 |
| --- | --- |
| 训练输出 | `outputs/runs/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean/` |
| T+1 输出 | `outputs/backtest/t1_fill_sim/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean/` |
| optimizer 输出 | `outputs/backtest/optimizer/feature_style_interaction_gru_l20_clean_alpha_resid_style_topk10_wide30_clean_core80/` |
| 闭环分析 | `outputs/analysis/feature_style_interaction_gru_l20_topk10_wide30_clean_closed_loop/` |

这些证据路径均为本地生成物，不提交到仓库。

### 关键读数

| 指标 | 数值 |
| --- | ---: |
| selected validation rank IC mean | 0.040001 |
| selected validation rank ICIR | 0.261220 |
| selected validation IC mean | 0.020936 |
| prediction rows | 68,681 |

较优 T+1 行：

| split | setting | net_ann | net_ir | excess_exec_universe_ann |
| --- | --- | ---: | ---: | ---: |
| validation | top_10_keep_1x | -0.112992 | -0.037093 | 0.042897 |
| test | top_20_keep_1x | 0.547856 | 0.211174 | 0.058175 |

该候选在 test 期表现有潜力，但 validation/test 分化仍然明显，因此保留为研究分支，没有提升为最终主线。
