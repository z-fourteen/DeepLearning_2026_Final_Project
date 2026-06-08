# Clean v1 Feature Engineering

The active clean feature contract is:

```text
configs/features/advanced_sequence_clean_v1.yaml
```

This configuration keeps point-in-time alpha features and residual-style features for the model tensor. It does not put execution masks, tradability controls, or raw risk-control columns directly into `X`.

The final model uses:

```text
build_mode: alpha_plus_residual_style
lookback: 60
num_features: 18
```

For operating commands, use the root `README.md`. This file is retained only as a feature-design note.
