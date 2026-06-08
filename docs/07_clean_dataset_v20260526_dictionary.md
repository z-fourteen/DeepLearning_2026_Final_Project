# Clean Dataset v20260526 Dictionary / Clean Dataset v20260526 字典

## English

This document explains the clean dataset used by the final model. It complements the root `README.md` and `pipelines/mart/README.md`.

### Dataset Purpose

`clean_dataset v20260526` converts the earlier full-feature research dataset into an auditable modeling asset. Its goals are:

- enforce point-in-time feature availability;
- separate alpha inputs from controls;
- preserve execution and risk information in sidecar data;
- use purged walk-forward split metadata;
- support T+1 execution evaluation after prediction.

### Tensor Contract

Clean tensors are compressed NPZ files under:

```text
data/mart/datasets/clean_purged_wf/
```

Each tensor contains:

| Key | Meaning |
| --- | --- |
| `X` | `[N, lookback, num_features]` model input tensor. |
| `y` | Supervised label. |
| `trade_date` | Signal date. |
| `ts_code` | Stock code. |
| `split` | `train`, `validation`, or `test`. |
| `feature_names` | Ordered model feature contract. |

### Final Tensor

```text
dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

| Item | Value |
| --- | ---: |
| lookback | 60 |
| alpha features | 13 |
| residual-style features | 5 |
| total features | 18 |

### Feature Interpretation

The 13 alpha features are intended to represent point-in-time residual alpha candidates. The 5 residual-style features bring back limited style-regime information after residualization, without directly feeding raw size, liquidity, volatility, or tradability controls into `X`.

### Execution Domain

The executable universe is defined by three layers:

```text
purged walk-forward split
strict tradable mask
T+1 execution labels and capacity constraints
```

The model learns only from the clean tensor. Execution feasibility is evaluated later by backtest and optimizer modules.

## 中文

本文说明最终模型使用的 clean dataset。它是根目录 `README.md` 和 `pipelines/mart/README.md` 的补充说明。

### 数据集目标

`clean_dataset v20260526` 将早期 full-feature 研究数据集转化为可审计的建模资产。目标包括：

- 强制 point-in-time 特征可得性；
- 分离 alpha 输入和控制变量；
- 将执行和风控信息保留在 sidecar 数据中；
- 使用 purged walk-forward split 元数据；
- 在预测之后支持 T+1 执行评估。

### Tensor 合同

clean tensor 是压缩 NPZ，位于：

```text
data/mart/datasets/clean_purged_wf/
```

每个 tensor 包含：

| Key | 含义 |
| --- | --- |
| `X` | `[N, lookback, num_features]` 模型输入张量。 |
| `y` | 监督标签。 |
| `trade_date` | 信号日期。 |
| `ts_code` | 股票代码。 |
| `split` | `train`、`validation` 或 `test`。 |
| `feature_names` | 有序模型特征合同。 |

### 最终 Tensor

```text
dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

| 项目 | 数值 |
| --- | ---: |
| lookback | 60 |
| alpha features | 13 |
| residual-style features | 5 |
| total features | 18 |

### 特征解释

13 个 alpha 特征用于表达 point-in-time residual alpha 候选。5 个 residual-style 特征在残差化后带回有限的风格 regime 信息，同时避免将原始规模、流动性、波动率或可交易性控制直接输入 `X`。

### 可执行域

可执行域由三层共同定义：

```text
purged walk-forward split
strict tradable mask
T+1 execution labels and capacity constraints
```

模型只学习 clean tensor；交易可行性在后续回测和 optimizer 模块中评估。
