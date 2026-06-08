# Data And Label Protocol / 数据与标签协议

## English

This document describes the data and label design behind the final project. Use the root `README.md` for reproduction commands.

### Data Layers

| Layer | Purpose |
| --- | --- |
| `data/lake/raw/` | Parsed raw source files. |
| `data/lake/core/` | Core derived tables such as the ChiNext pool. |
| `data/lake/state/` | Daily security state, tradability, listing, suspension, and price-limit information. |
| `data/mart/` | Feature, label, dataset, and clean tensor artifacts. |

These files are generated artifacts and are not submitted with the source code.

### Main Data Contract

| Item | Setting |
| --- | --- |
| data version | `v20260526` |
| historical range | `20160104` to `20260525` |
| universe | ChiNext index universe, `399006.SZ` |

### Label Types

The clean pipeline separates three concepts:

| Concept | Purpose |
| --- | --- |
| model target label | Supervised target used by GRU training. |
| strict tradable mask | Conservative sample filter used during clean tensor construction. |
| T+1 execution label | Execution-aware label used after prediction for fill simulation and optimizer evaluation. |

The model input must satisfy point-in-time requirements. Trading feasibility is evaluated after prediction through sidecar labels and T+1 execution simulation.

### Split Protocol

The main split contract is:

```text
configs/data/splits.yaml
```

| Item | Setting |
| --- | --- |
| label horizon | 5 trading days |
| purge | 5 trading days |
| embargo | 20 trading days |

Regenerated tensors record split and fold metadata in their manifests.

### Point-In-Time Rule

- Model features use lagged `lag1_` inputs.
- Labels are not included in `X`.
- Execution masks and tradability controls are not silently mixed into alpha tensors.
- Audit scripts check feature columns and suspicious shifts.

## 中文

本文说明最终项目的数据和标签设计。复现命令请以根目录 `README.md` 为准。

### 数据层

| 层级 | 用途 |
| --- | --- |
| `data/lake/raw/` | 解析后的原始数据。 |
| `data/lake/core/` | 创业板股票池等核心派生表。 |
| `data/lake/state/` | 日度证券状态、可交易性、上市状态、停牌和涨跌停信息。 |
| `data/mart/` | 特征、标签、数据集和 clean tensor artifact。 |

这些文件都是生成物，不随源代码提交。

### 主数据合同

| 项目 | 设置 |
| --- | --- |
| 数据版本 | `v20260526` |
| 历史区间 | `20160104` 至 `20260525` |
| 股票池 | 创业板指数成分，`399006.SZ` |

### 标签类型

clean pipeline 明确分离三类概念：

| 概念 | 用途 |
| --- | --- |
| 模型监督标签 | GRU 训练使用的监督目标。 |
| strict tradable mask | clean tensor 构建阶段使用的保守样本过滤。 |
| T+1 execution label | 预测后用于成交仿真和 optimizer 评估的执行标签。 |

模型输入必须满足 point-in-time 要求；交易可行性在预测之后通过 sidecar labels 和 T+1 成交仿真评估。

### 切分协议

主切分合同为：

```text
configs/data/splits.yaml
```

| 项目 | 设置 |
| --- | --- |
| 标签 horizon | 5 个交易日 |
| purge | 5 个交易日 |
| embargo | 20 个交易日 |

重新生成的 tensor 会在 manifest 中记录 split 和 fold 元数据。

### Point-In-Time 规则

- 模型特征使用 `lag1_` 滞后输入。
- 标签不进入 `X`。
- 执行 mask 和可交易性控制不静默混入 alpha tensor。
- 审计脚本会检查特征列和可疑 shift。
