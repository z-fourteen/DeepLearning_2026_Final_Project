# 模型训练协议

构建 clean tensor：

```powershell
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20
```

训练 GRU：

```powershell
python scripts/modeling/train_sequence.py --config configs/models/gru_l20_clean_alpha_only.yaml --device cuda
python scripts/modeling/train_sequence.py --config configs/models/gru_l20_clean_alpha_resid_style.yaml --device cuda
```

clean alpha-only 模型是纯 13-alpha 基准。residual-style 版本作为受控研究扩展保留。

两个配置都消费重新生成的 `chinext_purged_walk_forward` clean tensors，并写入 `purgedwf` run 目录，从而与已归档的 single-holdout runs 保持隔离。

如需添加新模型，请遵循以下文档中的数据集 schema、配置检查清单、预测输出合同和 smoke tests：

```text
docs/03a_new_model_clean_dataset_onboarding.md
```
