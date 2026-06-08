# Audit Scripts

## English

Audit scripts validate point-in-time safety, residual alpha behavior, and final-mainline evidence.

### Point-In-Time Audit

```bash
python scripts/audit/audit_point_in_time.py --labels data/mart/labels/labels_v20260526.parquet --out-dir outputs/audit/point_in_time
```

This checks feature columns, label fields, suspicious shifts, and possible leakage patterns.

### Residual Alpha Audit

```bash
python scripts/audit/audit_barra_lite_residual_alpha.py
```

Default inputs:

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet
data/mart/datasets/core/dataset_v20260526.parquet
data/mart/labels/labels_canonical_v20260526.parquet
```

### Mainline Audit

```bash
python scripts/audit/audit_clean_resid_mainline.py
```

Audit outputs are written under `outputs/audit/` and are ignored by Git.

## 中文

审计脚本用于验证 point-in-time 安全性、残差 alpha 行为和最终主线证据。

### Point-In-Time 审计

```bash
python scripts/audit/audit_point_in_time.py --labels data/mart/labels/labels_v20260526.parquet --out-dir outputs/audit/point_in_time
```

该脚本会检查特征字段、标签字段、可疑位移和潜在泄漏模式。

### 残差 Alpha 审计

```bash
python scripts/audit/audit_barra_lite_residual_alpha.py
```

默认输入：

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet
data/mart/datasets/core/dataset_v20260526.parquet
data/mart/labels/labels_canonical_v20260526.parquet
```

### 主线审计

```bash
python scripts/audit/audit_clean_resid_mainline.py
```

审计输出写入 `outputs/audit/`，并已被 Git 忽略。
