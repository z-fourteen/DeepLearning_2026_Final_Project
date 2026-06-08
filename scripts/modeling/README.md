# Modeling Scripts

This directory contains the model tensor builder wrapper and sequence-model training entry point. Run commands from the repository root.

## Build Clean Model Tensors

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

The builder reads the core mart dataset, clean feature config, split config, security daily state, and ChiNext pool. It writes NPZ tensors, sidecar parquet files, filter logs, and manifests under:

```text
data/mart/datasets/clean_purged_wf/
```

## Train The Final Model

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

## Final Model Selection

```text
run: feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
selection metric: checkpoint_score
checkpoint: epoch 12
best_metric: 0.05160820540933453
```

## New Model Checklist

- Create a config under `configs/models/`.
- Set `data.npz_path` to the intended clean tensor.
- Match `model.num_features` to `len(feature_names)`.
- Match `model.lookback` to `X.shape[1]`.
- Keep execution masks and risk-control fields out of `X`.
- Use validation for model selection.
- Treat test as the final locked holdout.
- Confirm output is written to `outputs/runs/<run_name>/`.

## Prediction Output Contract

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
