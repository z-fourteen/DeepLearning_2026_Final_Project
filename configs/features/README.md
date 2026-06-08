# Feature Configs

The active clean feature contract is:

```text
advanced_sequence_clean_v1.yaml
```

It defines the feature groups used to build clean model tensors from the mart dataset.

## Final Model Feature Contract

The final mainline uses the `alpha_plus_residual_style` build mode:

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

The final L60 tensor has:

| Field | Value |
| --- | ---: |
| lookback | 60 |
| alpha features | 13 |
| residual-style features | 5 |
| total features | 18 |

## Validation

Before building model tensors, validate that the feature config matches the generated mart columns:

```bash
python scripts/features/validate_clean_feature_set.py
```

The model input tensor must not include execution masks, raw risk-control columns, or future-looking labels.
