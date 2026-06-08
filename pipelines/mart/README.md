# Mart And Clean Dataset Pipeline

The mart pipeline converts raw and lake data into model-ready features, labels, and clean sequence tensors.

## Main Inputs

```text
data/lake/raw/
data/lake/core/chinext_pool/chinext_pool_scd2.parquet
data/lake/state/security_daily_state.parquet
configs/features/advanced_sequence_clean_v1.yaml
configs/data/splits.yaml
```

## Main Outputs

```text
data/mart/features_daily/features_daily_v20260526.parquet
data/mart/labels/labels_v20260526.parquet
data/mart/datasets/core/dataset_v20260526.parquet
data/mart/datasets/clean_purged_wf/
```

These files are generated artifacts and are ignored by Git.

## Clean Tensor Contract

Clean tensors are compressed NPZ files under:

```text
data/mart/datasets/clean_purged_wf/
```

Each NPZ must contain:

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | `[N, lookback, num_features]` | Model input tensor. |
| `y` | `[N]` | Supervised target, usually `label_rel_return`. |
| `trade_date` | `[N]` | Signal date. |
| `ts_code` | `[N]` | Stock code. |
| `split` | `[N]` | `train`, `validation`, or `test`. |
| `feature_names` | `[num_features]` | Ordered feature contract. |

## Build Commands

Build the final clean tensors:

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

Optional alpha-only baseline:

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
```

The final submitted model consumes:

```text
data/mart/datasets/clean_purged_wf/dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```
