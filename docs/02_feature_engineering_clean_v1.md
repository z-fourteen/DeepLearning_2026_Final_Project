# Feature Engineering Clean V1

`advanced_sequence_clean_v1` is the current feature contract.

Primary file:

```text
configs/features/advanced_sequence_clean_v1.yaml
```

The feature set keeps the model tensor focused on the cleaned alpha pool while
placing style, liquidity, and tradability variables into explicit control roles.
This avoids silently mixing risk controls into alpha inputs.

Historical development notes are archived in:

```text
docs/archive/feature_engineering_progress_20260529.md
docs/archive/final_feature_pool_report.md
```
