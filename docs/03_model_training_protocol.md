# Model Training Protocol

Clean tensor build:

```powershell
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20
```

GRU training:

```powershell
python scripts/modeling/train_sequence.py --config configs/models/gru_l20_clean_alpha_only.yaml --device cuda
python scripts/modeling/train_sequence.py --config configs/models/gru_l20_clean_alpha_resid_style.yaml --device cuda
```

The clean alpha-only model is the pure 13-alpha benchmark. The residual-style
variant is kept as a controlled research extension.

Both configs consume the regenerated `chinext_purged_walk_forward` clean tensors
and write to `purgedwf` run directories, keeping them separate from archived
single-holdout runs.
