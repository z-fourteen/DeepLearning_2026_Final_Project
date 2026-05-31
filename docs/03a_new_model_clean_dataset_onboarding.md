# 新模型接入 clean_dataset 指南

本指南是仓库迁移到 `clean_dataset` 主线后，新增模型必须遵循的接口合同。

## 1. 数据集合同

当前 clean tensors 位于：

```text
data/mart/datasets/clean_purged_wf/
```

大型 tensor、sidecar 和 filter-log 文件属于本地产物。Git 仅通过 Git LFS 跟踪当前活跃的 clean artifacts；legacy artifacts 只保留 manifest。

克隆仓库后，使用以下命令拉取 clean dataset artifacts：

```powershell
git lfs install
git lfs pull
```

每个 clean tensor 都是压缩 NPZ，并且必须包含以下 key：

| Key | Shape | 含义 |
| --- | --- | --- |
| `X` | `[N, lookback, num_features]` | 模型输入 tensor |
| `y` | `[N]` | `label_rel_return` 目标 |
| `trade_date` | `[N]` | 信号日 |
| `ts_code` | `[N]` | 股票代码 |
| `split` | `[N]` | `train`、`validation` 或 `test` |
| `feature_names` | `[num_features]` | 有序特征合同 |

当前生产 tensors 为：

```text
data/mart/datasets/clean_purged_wf/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_purged_walk_forward.npz
data/mart/datasets/clean_purged_wf/dataset_seq_l20_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

clean 13-alpha 基准使用 `alpha_only`。只有当实验明确测试 residual style 信息时，才使用 `alpha_plus_residual_style`。

## 2. 重新生成 Clean Tensors

在仓库根目录中，使用项目 conda 环境运行：

```powershell
conda run -n dl_env python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
conda run -n dl_env python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20
```

builder 会读取：

```text
data/mart/datasets/core/dataset_v20260526.parquet
configs/features/advanced_sequence_clean_v1.yaml
configs/data/splits.yaml
data/lake/state/security_daily_state.parquet
data/lake/core/chinext_pool/chinext_pool_scd2.parquet
```

并将 NPZ tensor、sidecar parquet、filter log 和 manifest 写入 `data/mart/datasets/clean_purged_wf/`。

## 3. 新模型配置检查清单

在 `configs/models/` 下创建配置，并确保以下字段与所选 tensor 对齐：

```yaml
run:
  name: "your_model_clean_alpha_only_purgedwf"
  output_dir: "outputs/runs/your_model_clean_alpha_only_purgedwf"
  seed: 42

data:
  npz_path: "data/mart/datasets/clean_purged_wf/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_purged_walk_forward.npz"
  train_split: "train"
  validation_split: "validation"
  test_split: "test"

model:
  name: "your_model_name"
  num_features: 13
  lookback: 20
```

规则：

- `model.num_features` 必须等于 NPZ 中 `len(feature_names)`。
- `model.lookback` 必须等于 `X.shape[1]`。
- 不要把风险控制变量或可交易 mask 追加进 `X`。
- 使用 validation 做模型选择。test 必须作为锁定的最终 holdout。

## 4. Data Loader 接口

PyTorch 模型使用：

```python
from src.data import SequenceNPZDataset

train_dataset = SequenceNPZDataset(npz_path, "train")
validation_dataset = SequenceNPZDataset(npz_path, "validation")
test_dataset = SequenceNPZDataset(npz_path, "test")
```

每个样本返回：

```text
x, y, trade_date, ts_code
```

当 loss 需要稳定的日度截面 IC 估计时，可以使用 `DateBatchSampler` 进行 date-batched training。

## 5. 预测输出合同

任何新模型都必须写出：

```text
outputs/runs/<run_name>/predictions.parquet
```

并包含以下列：

| 列名 | 含义 |
| --- | --- |
| `trade_date` | 信号日 |
| `ts_code` | 股票代码 |
| `pred_score` | 用于截面排序的模型分数 |
| `label_rel_return` | 随预测结果一起保留的监督标签，用于诊断 |
| `split` | train、validation 或 test |
| `model_name` | 模型标识 |

T+1 成交仿真器和审计流程会直接消费这一 schema。

## 6. 全量训练前的 Smoke Tests

运行 loader dry-run：

```powershell
conda run -n dl_env python scripts/modeling/train_sequence.py --config configs/models/gru_l20_clean_alpha_only.yaml --dry-run --device cpu
```

运行 point-in-time pathcheck：

```powershell
conda run -n dl_env python scripts/audit/audit_point_in_time.py --out-dir outputs/audit/_pathcheck_clean_model
```

只有满足以下条件时，新模型才可以进入全量训练：

- dataset dry-run 能加载全部三个 split
- `num_features` 和 `lookback` 与 NPZ 匹配
- PIT audit 报告零 blockers
- 输出配置写入新的 `outputs/runs/<run_name>/` 目录

## 7. 执行与审计交接

训练完成后，将 `predictions.parquet` 接入：

```powershell
conda run -n dl_env python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet
```

对于当前主线 residual-style 候选模型，使用：

```powershell
conda run -n dl_env python scripts/backtest/run_clean_dataset_execution_stack.py --only-existing
conda run -n dl_env python scripts/audit/audit_barra_lite_residual_alpha.py
conda run -n dl_env python scripts/audit/audit_clean_resid_mainline.py
```

只有在 validation 表现、执行指标和审计结论都完成记录之后，才能在读取最终 test holdout 前推进新模型结果。
