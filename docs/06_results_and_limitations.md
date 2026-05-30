# Results And Limitations

Final review artifacts are concentrated under:

```text
outputs/backtest/clean_dataset_execution_stack/
outputs/audit/barra_lite_residual_alpha/
outputs/audit/point_in_time_canonical_labels/
```

Known limitation:

The historical `full62` score produced strong test-period results, but its
style-regime dependence and residual-alpha stability were not strong enough for
the final production-facing mainline. It is therefore archived, not deleted.
