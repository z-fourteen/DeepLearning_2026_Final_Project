# Clean v1 Feature Engineering / Clean v1 特征工程

## English

The active clean feature contract is:

```text
configs/features/advanced_sequence_clean_v1.yaml
```

This configuration keeps point-in-time alpha features and residual-style features for the model tensor. It does not put execution masks, tradability controls, or raw risk-control columns directly into `X`.

Final model feature setup:

```text
build_mode: alpha_plus_residual_style
lookback: 60
num_features: 18
```

Feature roles:

| Role | Meaning |
| --- | --- |
| alpha | Candidate model input features. |
| residual_style | Residualized style-carry features allowed into the final tensor. |
| risk_control | Controls used outside the tensor for audit or optimization. |
| tradability_control | Execution and tradability controls used outside the tensor. |
| exclude | Features removed from the model contract. |

Use the root `README.md` for commands.

## 中文

当前 clean feature 合同为：

```text
configs/features/advanced_sequence_clean_v1.yaml
```

该配置只将 point-in-time alpha 特征和 residual-style 特征放入模型 tensor，不把执行 mask、可交易性控制或原始风控列直接放入 `X`。

最终模型特征设置：

```text
build_mode: alpha_plus_residual_style
lookback: 60
num_features: 18
```

特征角色：

| 角色 | 含义 |
| --- | --- |
| alpha | 模型输入候选特征。 |
| residual_style | 允许进入最终 tensor 的残差风格信息。 |
| risk_control | 在 tensor 外用于审计或优化的风控变量。 |
| tradability_control | 在 tensor 外使用的执行和可交易性控制。 |
| exclude | 从模型合同中移除的特征。 |

运行命令请以根目录 `README.md` 为准。
