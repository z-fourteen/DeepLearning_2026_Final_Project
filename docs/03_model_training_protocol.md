# 模型训练协议

构建 clean tensor：

```powershell
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20,60
```

最终主线训练配置：

```powershell
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
```

最终模型选择规则：

```text
run: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
selection metric: checkpoint_score
checkpoint: epoch 12
best_metric: 0.05160820540933453
stop_reason: metric_early_stop:checkpoint_score
```

clean alpha-only 和 L20 residual-style 模型保留为基准和历史对照，不再作为主线入口。

所有主线配置都消费重新生成的 `chinext_purged_walk_forward` clean tensors，并与已归档的 single-holdout runs 保持隔离。

如需添加新模型，请遵循以下文档中的数据集 schema、配置检查清单、预测输出合同和 smoke tests：

```text
docs/03a_new_model_clean_dataset_onboarding.md
```
