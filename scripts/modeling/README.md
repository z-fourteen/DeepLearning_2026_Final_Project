# Modeling Scripts

## English

This directory contains the model tensor builder wrapper and sequence-model training entry point. Run commands from the repository root.

### Build Clean Model Tensors

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

The builder reads the core mart dataset, clean feature config, split config, security daily state, and ChiNext pool. It writes NPZ tensors, sidecar parquet files, filter logs, and manifests under:

```text
data/mart/datasets/clean_purged_wf/
```

### Train The Final Model

Dry-run first:

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --dry-run --device cpu
```

GPU training:

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
```

CPU fallback:

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cpu
```

### Final Model Selection

```text
run: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
selection metric: checkpoint_score
checkpoint: epoch 12
best_metric: 0.05160820540933453
```

### New Model Checklist

- Create a config under `configs/models/`.
- Set `data.npz_path` to the intended clean tensor.
- Match `model.num_features` to `len(feature_names)`.
- Match `model.lookback` to `X.shape[1]`.
- Keep execution masks and risk-control fields out of `X`.
- Use validation for model selection.
- Treat test as the final locked holdout.
- Confirm output is written to `outputs/runs/<run_name>/`.

### Prediction Output Contract

Every trained model should write:

```text
outputs/runs/<run_name>/predictions.parquet
```

Required columns:

| Column | Meaning |
| --- | --- |
| `trade_date` | Signal date. |
| `ts_code` | Stock code. |
| `pred_score` | Cross-sectional ranking score. |
| `label_rel_return` | Label retained for diagnostics. |
| `split` | `train`, `validation`, or `test`. |
| `model_name` | Model identifier. |

## 中文

本目录包含模型张量构建封装和序列模型训练入口。命令应从仓库根目录运行。

### 构建 Clean 模型张量

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

构建器会读取核心 mart 数据集、clean 特征配置、切分配置、证券日状态和创业板股票池，并将 NPZ 张量、旁路 parquet、过滤日志和 manifest 写入：

```text
data/mart/datasets/clean_purged_wf/
```

### 训练最终模型

先执行 dry-run：

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --dry-run --device cpu
```

GPU 训练：

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
```

CPU 兜底：

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cpu
```

### 最终模型选择

```text
run: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
selection metric: checkpoint_score
checkpoint: epoch 12
best_metric: 0.05160820540933453
```

### 新模型检查清单

- 在 `configs/models/` 下创建配置。
- 将 `data.npz_path` 指向目标 clean 张量。
- 保证 `model.num_features` 与 `len(feature_names)` 一致。
- 保证 `model.lookback` 与 `X.shape[1]` 一致。
- 不要将执行掩码和风控字段放入 `X`。
- 使用验证集进行模型选择。
- 将测试集视为最终锁定 holdout。
- 确认输出写入 `outputs/runs/<run_name>/`。

### 预测输出契约

每个训练完成的模型都应写出：

```text
outputs/runs/<run_name>/predictions.parquet
```

必要字段：

| 字段 | 含义 |
| --- | --- |
| `trade_date` | 信号日期。 |
| `ts_code` | 股票代码。 |
| `pred_score` | 横截面排序分数。 |
| `label_rel_return` | 保留用于诊断的标签。 |
| `split` | `train`、`validation` 或 `test`。 |
| `model_name` | 模型标识。 |
