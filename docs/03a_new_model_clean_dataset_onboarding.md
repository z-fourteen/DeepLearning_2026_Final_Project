# New Model Clean Dataset Onboarding

This guide is the required interface contract for adding a new model after the
repository moved to the `clean_dataset` mainline.

## 1. Dataset Contract

Active clean tensors live under:

```text
data/mart/datasets/clean_purged_wf/
```

Large tensor, sidecar, and filter-log files are local artifacts. Git tracks only
the manifest JSON files.

Each clean tensor is a compressed NPZ with these required keys:

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | `[N, lookback, num_features]` | model input tensor |
| `y` | `[N]` | `label_rel_return` target |
| `trade_date` | `[N]` | signal date |
| `ts_code` | `[N]` | stock code |
| `split` | `[N]` | `train`, `validation`, or `test` |
| `feature_names` | `[num_features]` | ordered feature contract |

The current production tensors are:

```text
data/mart/datasets/clean_purged_wf/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_purged_walk_forward.npz
data/mart/datasets/clean_purged_wf/dataset_seq_l20_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

Use `alpha_only` for the clean 13-alpha benchmark. Use
`alpha_plus_residual_style` only when the experiment explicitly tests residual
style information.

## 2. Regenerate Clean Tensors

Run from the repository root with the project conda environment:

```powershell
conda run -n dl_env python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
conda run -n dl_env python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20
```

The builder reads:

```text
data/mart/datasets/core/dataset_v20260526.parquet
configs/features/advanced_sequence_clean_v1.yaml
configs/data/splits.yaml
data/lake/state/security_daily_state.parquet
data/lake/core/chinext_pool/chinext_pool_scd2.parquet
```

It writes the NPZ tensor, sidecar parquet, filter log, and manifest to
`data/mart/datasets/clean_purged_wf/`.

## 3. New Model Config Checklist

Create a config under `configs/models/` and keep these fields aligned with the
selected tensor:

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

Rules:

- `model.num_features` must equal `len(feature_names)` in the NPZ.
- `model.lookback` must equal `X.shape[1]`.
- Do not append risk controls or tradability masks into `X`.
- Use validation for model selection. Treat test as the locked final holdout.

## 4. Data Loader Interface

For PyTorch models, use:

```python
from src.data import SequenceNPZDataset

train_dataset = SequenceNPZDataset(npz_path, "train")
validation_dataset = SequenceNPZDataset(npz_path, "validation")
test_dataset = SequenceNPZDataset(npz_path, "test")
```

Each sample returns:

```text
x, y, trade_date, ts_code
```

Date-batched training can use `DateBatchSampler` when the loss needs stable
cross-sectional daily IC estimates.

## 5. Prediction Output Contract

Any new model must write:

```text
outputs/runs/<run_name>/predictions.parquet
```

with these columns:

| Column | Meaning |
| --- | --- |
| `trade_date` | signal date |
| `ts_code` | stock code |
| `pred_score` | model score used for cross-sectional ranking |
| `label_rel_return` | supervised label carried through for diagnostics |
| `split` | train, validation, or test |
| `model_name` | model identifier |

The T+1 fill simulator and audits consume this schema directly.

## 6. Smoke Tests Before Full Training

Run the loader dry-run:

```powershell
conda run -n dl_env python scripts/modeling/train_sequence.py --config configs/models/gru_l20_clean_alpha_only.yaml --dry-run --device cpu
```

Run the point-in-time pathcheck:

```powershell
conda run -n dl_env python scripts/audit/audit_point_in_time.py --out-dir outputs/audit/_pathcheck_clean_model
```

A new model is ready for full training only when:

- dataset dry-run loads all three splits
- `num_features` and `lookback` match the NPZ
- PIT audit reports zero blockers
- output config writes to a new `outputs/runs/<run_name>/` directory

## 7. Execution And Audit Handoff

After training, route `predictions.parquet` through:

```powershell
conda run -n dl_env python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet
```

For the current mainline residual-style candidate, use:

```powershell
conda run -n dl_env python scripts/backtest/run_clean_dataset_execution_stack.py --only-existing
conda run -n dl_env python scripts/audit/audit_barra_lite_residual_alpha.py
conda run -n dl_env python scripts/audit/audit_clean_resid_mainline.py
```

New model results should be promoted only if validation performance, execution
metrics, and audit findings are all documented before reading the final test
holdout.
