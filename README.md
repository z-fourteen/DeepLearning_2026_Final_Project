# GRU Clean Dataset Stock Selection

This repository contains a GRU-based time-series stock selection project built around a production-facing `clean_dataset` pipeline.

The final mainline is:

```text
advanced_sequence_clean_v1
-> L60 clean alpha-resid-style tensor
-> feature-style interaction GRU, checkpoint_score epoch 12
-> frozen soft optimizer: risk_control=none, k=10, style_penalty=0.1, turnover_penalty=0.0, min_invested=0.8
-> T+1 execution, residual-alpha, capacity, and validation-attribution evidence
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
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20,60
```

Train or rerun the frozen final model:

```powershell
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
```

Run execution evaluation:

```powershell
conda run -n dl_env python scripts/portfolio/run_final_mainline_optimizer.py
```

Run audits:

```powershell
python scripts/audit/audit_point_in_time.py
python scripts/audit/audit_barra_lite_residual_alpha.py
python scripts/analysis/analyze_optimizer_validation_attribution.py --periods outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_periods.csv --summary outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_summary.csv --output-dir outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution --split validation --top-n 6
```

## Repository Map

```text
configs/features/advanced_sequence_clean_v1.yaml   # clean alpha contract
configs/portfolio/final_mainline_optimizer.yaml    # frozen final optimizer and evidence paths
pipelines/mart/clean_dataset.py                    # clean tensor builder
scripts/modeling/                                  # tensor build and GRU training
scripts/portfolio/run_final_mainline_optimizer.py  # frozen optimizer entrypoint
scripts/backtest/                                  # T+1 fill simulation and execution utilities
scripts/audit/                                     # PIT, residual-alpha, and mainline audits
docs/                                              # reviewer-facing documentation
legacy/legacy_full62_v1/                           # read-only historical full62 archive
```

Start with `docs/09_final_mainline_freeze.md` for the frozen final evidence chain,
then use `docs/00_project_overview.md` for the review narrative.
Use `docs/03a_new_model_clean_dataset_onboarding.md` when wiring a new model
into the clean tensors.
