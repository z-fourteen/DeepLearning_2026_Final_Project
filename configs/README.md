# Configs

This directory contains the reproducible YAML contracts used by the project. The root `README.md` explains the full reproduction path; this file is a local map for maintainers.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/data/` | Data source, mart, label, and split settings. |
| `configs/features.yaml` | Top-level feature generation and validation settings. |
| `configs/features/` | Clean feature contracts. |
| `configs/models/` | Model training configs. Each config defines input tensor path, model shape, training objective, seed, and output directory. |
| `configs/backtest/` | T+1 execution and backtest settings. |
| `configs/portfolio/` | Portfolio optimizer settings. |
| `configs/live/` | Live-trading extension settings. Not required for assignment reproduction. |

## Final Mainline Configs

```text
configs/features/advanced_sequence_clean_v1.yaml
configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml
configs/portfolio/final_mainline_optimizer.yaml
```

Generated data, model weights, predictions, and optimizer outputs are not stored here. Configs should only contain reproducible parameters and paths.
