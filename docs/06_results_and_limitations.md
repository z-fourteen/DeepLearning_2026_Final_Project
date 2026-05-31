# 结果与局限

最终复核产物集中在：

```text
outputs/backtest/clean_dataset_execution_stack/
outputs/audit/barra_lite_residual_alpha/
outputs/audit/point_in_time_canonical_labels/
```

wide-band feature-style interaction GRU 候选模型的额外闭环分析记录在：

```text
docs/08_feature_style_interaction_wide30_closed_loop.md
outputs/analysis/feature_style_interaction_gru_l20_topk10_wide30_clean_closed_loop/
```

该 run 仅作为研究候选。它在测试期的可执行收益较强，但验证期执行结果为负，因此在没有验证期转正的重跑结果或滚动窗口确认之前，不应提升为生产主线。

已知局限：

历史 `full62` score 在测试期产生过较强结果，但其风格 regime 依赖和 residual-alpha 稳定性不足以支撑最终面向生产的主线。因此它被归档，而不是删除。
