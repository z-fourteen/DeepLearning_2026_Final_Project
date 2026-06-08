# Configs

## English

This directory contains the reproducible YAML contracts used by the project. The root `README.md` remains the single entry point for reproduction; this file is a local map for maintainers.

### Layout

| Path | Purpose |
| --- | --- |
| `configs/data/` | Data source, mart, label, and split settings. |
| `configs/features.yaml` | Top-level feature generation and validation settings. |
| `configs/features/` | Clean feature contracts. |
| `configs/models/` | Model training configs. Each config defines the input tensor path, model shape, training objective, seed, and output directory. |
| `configs/backtest/` | T+1 execution and backtest settings. |
| `configs/portfolio/` | Portfolio optimizer settings. |
| `configs/live/` | Live-trading extension settings. Not required for assignment reproduction. |

### Final Mainline Configs

```text
configs/features/advanced_sequence_clean_v1.yaml
configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml
configs/portfolio/final_mainline_optimizer.yaml
```

Generated data, model weights, predictions, and optimizer outputs are not stored here. Config files should only contain reproducible parameters and paths.

## 中文

本目录保存项目可复现流程所需的 YAML 配置契约。根目录 `README.md` 仍然是唯一复现入口；本文件仅作为维护者使用的局部索引。

### 目录说明

| 路径 | 作用 |
| --- | --- |
| `configs/data/` | 数据源、mart、标签与切分配置。 |
| `configs/features.yaml` | 顶层特征生成与校验配置。 |
| `configs/features/` | clean 特征契约。 |
| `configs/models/` | 模型训练配置。每个配置定义输入张量路径、模型结构、训练目标、随机种子和输出目录。 |
| `configs/backtest/` | T+1 执行与回测配置。 |
| `configs/portfolio/` | 组合优化器配置。 |
| `configs/live/` | 实盘扩展配置，不属于课程作业复现必需流程。 |

### 最终主线配置

```text
configs/features/advanced_sequence_clean_v1.yaml
configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml
configs/portfolio/final_mainline_optimizer.yaml
```

生成的数据、模型权重、预测结果和优化器输出不存放在本目录中。配置文件只应保留可复现的参数和路径。
