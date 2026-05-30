# GRU Clean Dataset Stock Selection

This repository contains a GRU-based time-series stock selection project built around a production-facing `clean_dataset` pipeline.

The final mainline is:

```text
advanced_sequence_clean_v1
-> clean sequence tensor
-> GRU L20 score model
-> T+1 open-fill simulation
-> residual-alpha and capacity audits
```

Historical `full62` experiments are frozen under `legacy/legacy_full62_v1/`. They remain as reproducibility evidence, but the active pipeline does not depend on them.

## Core Commands

Validate the clean feature contract:

```powershell
python scripts/features/validate_clean_feature_set.py
```

Build clean tensors:

```powershell
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20
```

Train GRU models:

```powershell
python scripts/modeling/train_sequence.py --config configs/models/gru_l20_clean_alpha_only.yaml --device cuda
python scripts/modeling/train_sequence.py --config configs/models/gru_l20_clean_alpha_resid_style.yaml --device cuda
```

Run execution evaluation:

```powershell
python scripts/backtest/run_clean_dataset_execution_stack.py --only-existing
python scripts/backtest/run_clean_resid_mainline.py
```

Run audits:

```powershell
python scripts/audit/audit_point_in_time.py
python scripts/audit/audit_barra_lite_residual_alpha.py
python scripts/audit/audit_clean_resid_mainline.py
```

## Repository Map

```text
configs/features/advanced_sequence_clean_v1.yaml   # clean alpha contract
pipelines/mart/clean_dataset.py                    # clean tensor builder
scripts/modeling/                                  # tensor build and GRU training
scripts/backtest/                                  # T+1 fill simulation and clean execution stack
scripts/audit/                                     # PIT, residual-alpha, and mainline audits
docs/                                              # reviewer-facing documentation
legacy/legacy_full62_v1/                           # read-only historical full62 archive
```

Start with `docs/00_project_overview.md` for the review narrative.
Use `docs/03a_new_model_clean_dataset_onboarding.md` when wiring a new model
into the clean tensors.
