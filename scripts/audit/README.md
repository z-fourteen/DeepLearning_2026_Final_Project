# Audit Scripts

Audit scripts validate point-in-time safety, residual alpha behavior, and final-mainline evidence.

## Point-In-Time Audit

```bash
python scripts/audit/audit_point_in_time.py --labels data/mart/labels/labels_v20260526.parquet --out-dir outputs/audit/point_in_time
```

This checks feature columns, label fields, suspicious shifts, and possible leakage patterns.

## Residual Alpha Audit

```bash
python scripts/audit/audit_barra_lite_residual_alpha.py
```

Default inputs:

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet
data/mart/datasets/core/dataset_v20260526.parquet
data/mart/labels/labels_canonical_v20260526.parquet
```

## Mainline Audit

```bash
python scripts/audit/audit_clean_resid_mainline.py
```

Audit outputs are written under `outputs/audit/` and are ignored by Git.
