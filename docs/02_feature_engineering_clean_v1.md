# 特征工程 Clean V1

`advanced_sequence_clean_v1` 是当前特征合同。

主配置文件：

```text
configs/features/advanced_sequence_clean_v1.yaml
```

该特征集让模型 tensor 聚焦在清洗后的 alpha pool 上，同时把风格、流动性和可交易性变量放入明确的控制角色中。这可以避免将风险控制变量静默混入 alpha 输入。

历史开发记录已归档在：

```text
docs/archive/feature_engineering_progress_20260529.md
docs/archive/final_feature_pool_report.md
```
