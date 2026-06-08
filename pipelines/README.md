# Pipelines

## English

This directory contains core pipeline implementations called by scripts. Users should run the command-line entry points under `scripts/`; pipeline modules are maintained as library code.

### Layout

| Path | Purpose |
| --- | --- |
| `pipelines/ingest/` | Raw data ingestion into the local lake. |
| `pipelines/pool/` | ChiNext universe and pool construction. |
| `pipelines/state/` | Security daily state, tradability, listing, suspension, and price-limit logic. |
| `pipelines/mart/` | Feature, label, clean dataset, and tensor builders. |

The root `README.md` documents the end-to-end reproduction commands.

## 中文

本目录保存由脚本调用的核心流水线实现。用户应通过 `scripts/` 下的命令行入口运行流程；`pipelines/` 中的模块按库代码维护。

### 目录说明

| 路径 | 作用 |
| --- | --- |
| `pipelines/ingest/` | 将原始数据接入本地数据湖。 |
| `pipelines/pool/` | 构建创业板股票池与成分历史表。 |
| `pipelines/state/` | 构建证券日状态、可交易性、上市、停牌与涨跌停逻辑。 |
| `pipelines/mart/` | 构建特征、标签、clean 数据集和模型张量。 |

端到端复现命令统一写在根目录 `README.md` 中。
