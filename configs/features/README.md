# Feature Configs

## English

The active clean feature contract is:

```text
advanced_sequence_clean_v1.yaml
```

It defines the feature groups used to build clean model tensors from the mart dataset.

### Final Model Feature Contract

The final mainline uses the `alpha_plus_residual_style` build mode:

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

The final L60 tensor has:

| Field | Value |
| --- | ---: |
| lookback | 60 |
| alpha features | 13 |
| residual-style features | 5 |
| total features | 18 |

### Validation

Before building model tensors, validate that the feature config matches the generated mart columns:

```bash
python scripts/features/validate_clean_feature_set.py
```

The model input tensor must not include execution masks, raw risk-control columns, or future-looking labels.

## 中文

当前启用的 clean 特征契约为：

```text
advanced_sequence_clean_v1.yaml
```

该配置定义了从 mart 数据集构建 clean 模型张量时使用的特征分组。

### 最终模型特征契约

最终主线使用 `alpha_plus_residual_style` 构建模式：

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

最终 L60 张量包含：

| 字段 | 数值 |
| --- | ---: |
| lookback | 60 |
| alpha 特征 | 13 |
| residual-style 特征 | 5 |
| 特征总数 | 18 |

### 校验

在构建模型张量前，先校验特征配置是否与已生成的 mart 字段一致：

```bash
python scripts/features/validate_clean_feature_set.py
```

模型输入张量不得包含执行掩码、原始风控字段或任何未来信息标签。
