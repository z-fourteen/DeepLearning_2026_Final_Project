# GRU Clean Dataset Stock Selection

## English

This repository contains the source code for a deep-learning stock selection experiment on the ChiNext universe. The final pipeline builds a point-in-time clean dataset, trains a GRU-family sequence model, runs T+1 execution evaluation, and evaluates a frozen portfolio optimizer.

The final mainline is:

```text
raw A-share market data
-> clean_dataset v20260526
-> L60 alpha + residual-style tensor
-> feature-style interaction GRU
-> checkpoint_score epoch 12
-> frozen optimizer: risk_control=none, k=10, min_invested=0.8
```

Data files, generated model tensors, predictions, and model weights are intentionally not committed. The experiment report should describe the data source and date range; this repository provides the code and configuration needed to reproduce the results after the data is placed locally.

### Repository Structure

```text
configs/        Reproducible YAML configs for data, features, models, backtests, and optimizer
data/           Local data workspace; only .gitkeep placeholders are tracked
docs/           Experiment design and final analysis notes
legacy/         Historical full62 experiments kept for archive only
meta/           Schema registry and generated metadata locations
outputs/        Generated runs, predictions, audits, and reports; only .gitkeep is tracked
pipelines/      Core data and mart pipeline implementations
scripts/        Command-line entry points for data, training, evaluation, audit, and live workflow
src/            Model, dataset, and training library code
```

The root `README.md` is the single reproduction entry point. Files under `docs/` are analysis and background material, not required operating instructions.

### Environment Setup

Use Python 3.10 or a compatible Python 3.x environment. A CUDA-capable GPU is recommended for final model training, but the code can run on CPU for dry-runs and smoke checks.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quick syntax check:

```bash
python -m compileall scripts src pipelines
```

### Data Preparation

The code expects local A-share source data under the directory configured by `configs/data/data.yaml`:

```yaml
source:
  root_dir: "A股数据"
```

Place the raw data directory at the repository root, or edit `configs/data/data.yaml` so `source.root_dir` points to your local data location.

Expected raw datasets:

```text
A股数据/
  basic.csv
  trade_cal.csv
  daily/
  stock_st/
  index_weight/
  metric/
  moneyflow/
  market/
```

Important data assumptions:

- Main universe: ChiNext index, `399006.SZ`.
- Main data version: `v20260526`.
- Main historical range used by the frozen experiment: `20160104` to `20260525`.
- Data files are not included in the source submission.
- Generated parquet, npz, model weights, and predictions are local artifacts and are ignored by Git.

### Build Data Pipeline

Run commands from the repository root.

Full offline DAG:

```bash
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260525
```

Equivalent step-by-step commands:

```bash
python scripts/data/run_ingest_raw.py --data-version v20260526
python scripts/data/run_build_pool.py --data-version v20260526
python scripts/data/run_build_market_state.py --data-version v20260526 --incremental
python scripts/data/validate_market_state_coverage.py --data-version v20260526 --start-date 20160104 --end-date 20260525 --strict
python scripts/data/run_build_mart.py --data-version v20260526 --start-date 20160104 --end-date 20260525
```

Key generated files:

```text
data/lake/raw/
data/lake/core/chinext_pool/chinext_pool_scd2.parquet
data/lake/state/security_daily_state.parquet
data/mart/features_daily/features_daily_v20260526.parquet
data/mart/labels/labels_v20260526.parquet
data/mart/datasets/core/dataset_v20260526.parquet
```

### Feature Engineering

Validate the clean feature contract:

```bash
python scripts/features/validate_clean_feature_set.py
```

Active feature contract:

```text
configs/features/advanced_sequence_clean_v1.yaml
```

### Build Model Tensors

Build the tensors used by the final model:

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

The final model uses:

```text
data/mart/datasets/clean_purged_wf/dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

Optional alpha-only baseline tensor:

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
```

### Model Training

Dry-run:

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --dry-run --device cpu
```

GPU training:

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
```

CPU fallback:

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cpu
```

Expected training output:

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/
  config.yaml
  metrics.json
  model.pt
  predictions.parquet
```

The frozen report uses epoch 12 selected by `checkpoint_score`. Training is seeded with `seed: 42`, but exact GPU reproducibility can vary across hardware, CUDA, BLAS, and PyTorch builds.

### Evaluation

Run the frozen T+1 execution route:

```bash
python scripts/backtest/run_clean_resid_mainline.py
```

Run the frozen final optimizer:

```bash
python scripts/portfolio/run_final_mainline_optimizer.py
```

Optimizer config:

```text
configs/portfolio/final_mainline_optimizer.yaml
```

Required inputs:

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet
data/mart/datasets/core/dataset_v20260526.parquet
data/mart/labels/execution_labels_v20260526.parquet
```

If `execution_labels_v20260526.parquet` is missing, inspect the label builder:

```bash
python scripts/data/build_execution_labels.py --help
```

### Audit And Analysis

Point-in-time leakage audit:

```bash
python scripts/audit/audit_point_in_time.py --labels data/mart/labels/labels_v20260526.parquet --out-dir outputs/audit/point_in_time
```

Residual alpha audit:

```bash
python scripts/audit/audit_barra_lite_residual_alpha.py
```

Closed-loop summary:

```bash
python scripts/analysis/summarize_model_closed_loop.py
```

Optimizer validation attribution:

```bash
python scripts/analysis/analyze_optimizer_validation_attribution.py --periods outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_periods.csv --summary outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_summary.csv --output-dir outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution --split validation --top-n 6
```

Rebuild optimizer grid if needed:

```bash
python scripts/portfolio/run_soft_optimizer_grid.py --predictions outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet --output-dir outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80
```

### Reproduce Results

Complete sequence for a fresh local reproduction:

```bash
python -m compileall scripts src pipelines
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260525
python scripts/features/validate_clean_feature_set.py
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --dry-run --device cpu
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
python scripts/backtest/run_clean_resid_mainline.py
python scripts/portfolio/run_soft_optimizer_grid.py --predictions outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet --output-dir outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80
python scripts/portfolio/run_final_mainline_optimizer.py
python scripts/audit/audit_point_in_time.py --labels data/mart/labels/labels_v20260526.parquet --out-dir outputs/audit/point_in_time
python scripts/audit/audit_barra_lite_residual_alpha.py
python scripts/analysis/summarize_model_closed_loop.py
```

Expected final result directory:

```text
outputs/backtest/optimizer/final_mainline_ckptscore_e12/
  final_optimizer_periods.csv
  final_optimizer_summary.csv
  manifest.json
```

Expected headline result from the frozen run:

| split | net_ann | net_ir | excess_benchmark_ann | excess_exec_universe_ann | avg_invested_weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | -0.059635 | -0.013411 | 0.047074 | -0.028238 | 0.839216 |
| test | 0.268252 | 0.189517 | -0.258701 | -0.035435 | 0.893154 |

Interpretation: the final optimizer completes a reproducible research loop, but it should not be described as a stable production alpha because validation and test do not both show positive excess versus the executable universe.

### Documentation

The project can be reproduced from this README alone. Additional documents are optional context:

```text
docs/00_project_evolution.md
docs/01_data_and_label_protocol.md
docs/02_feature_engineering_clean_v1.md
docs/03_model_baseline_evolution.md
docs/06_results_and_limitations.md
docs/07_clean_dataset_v20260526_dictionary.md
docs/08_feature_style_interaction_wide30_closed_loop.md
docs/09_final_mainline_freeze.md
docs/experiment_report.md
```

## 中文

本仓库包含一个面向创业板股票池的深度学习选股实验源代码。最终流程会构建 point-in-time clean dataset，训练 GRU 系列序列模型，进行 T+1 执行评估，并运行冻结版组合优化器。

最终主线为：

```text
A 股原始行情数据
-> clean_dataset v20260526
-> L60 alpha + residual-style 张量
-> feature-style interaction GRU
-> checkpoint_score 选择 epoch 12
-> 冻结 optimizer: risk_control=none, k=10, min_invested=0.8
```

数据文件、模型张量、预测结果和模型权重不会提交到仓库。实验报告中应说明使用的数据来源和时间范围；本仓库提供复现实验所需的代码与配置。

### 仓库结构

```text
configs/        数据、特征、模型、回测和 optimizer 的可复现 YAML 配置
data/           本地数据工作区；仓库只跟踪 .gitkeep
docs/           实验设计和最终分析文档
legacy/         历史 full62 实验归档
meta/           schema registry 和元数据输出位置
outputs/        训练、预测、审计和评估输出；仓库只跟踪 .gitkeep
pipelines/      数据和 mart 的核心 pipeline 实现
scripts/        数据、训练、评估、审计和 live 流程的命令行入口
src/            模型、数据集和训练库代码
```

根目录 `README.md` 是唯一复现入口。`docs/` 中的文件只提供分析和背景材料，不是运行项目前必须阅读的说明。

### 环境配置

建议使用 Python 3.10 或兼容的 Python 3.x 环境。最终模型训练建议使用 CUDA GPU；CPU 可用于 dry-run 和 smoke check。

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

快速语法检查：

```bash
python -m compileall scripts src pipelines
```

### 数据准备

代码默认从 `configs/data/data.yaml` 指定的目录读取本地 A 股数据：

```yaml
source:
  root_dir: "A股数据"
```

可以将原始数据目录放在仓库根目录，也可以修改 `configs/data/data.yaml` 中的 `source.root_dir`。

期望的原始数据结构：

```text
A股数据/
  basic.csv
  trade_cal.csv
  daily/
  stock_st/
  index_weight/
  metric/
  moneyflow/
  market/
```

关键数据假设：

- 股票池：创业板指数成分，`399006.SZ`。
- 数据版本：`v20260526`。
- 冻结实验历史区间：`20160104` 至 `20260525`。
- 源数据不随代码提交。
- 生成的 parquet、npz、模型权重和预测文件均为本地 artifact，并被 Git 忽略。

### 构建数据流水线

所有命令均从仓库根目录执行。

完整离线 DAG：

```bash
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260525
```

等价拆分命令：

```bash
python scripts/data/run_ingest_raw.py --data-version v20260526
python scripts/data/run_build_pool.py --data-version v20260526
python scripts/data/run_build_market_state.py --data-version v20260526 --incremental
python scripts/data/validate_market_state_coverage.py --data-version v20260526 --start-date 20160104 --end-date 20260525 --strict
python scripts/data/run_build_mart.py --data-version v20260526 --start-date 20160104 --end-date 20260525
```

关键生成文件：

```text
data/lake/raw/
data/lake/core/chinext_pool/chinext_pool_scd2.parquet
data/lake/state/security_daily_state.parquet
data/mart/features_daily/features_daily_v20260526.parquet
data/mart/labels/labels_v20260526.parquet
data/mart/datasets/core/dataset_v20260526.parquet
```

### 特征工程

校验 clean feature 合同：

```bash
python scripts/features/validate_clean_feature_set.py
```

当前特征合同：

```text
configs/features/advanced_sequence_clean_v1.yaml
```

### 构建模型张量

构建最终模型使用的张量：

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
```

最终模型使用：

```text
data/mart/datasets/clean_purged_wf/dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
```

可选 alpha-only baseline 张量：

```bash
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
```

### 模型训练

先运行 dry-run：

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --dry-run --device cpu
```

GPU 训练：

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
```

CPU 兜底：

```bash
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cpu
```

训练输出：

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/
  config.yaml
  metrics.json
  model.pt
  predictions.parquet
```

冻结报告使用 `checkpoint_score` 选择的 epoch 12。配置中设置了 `seed: 42`，但不同 GPU、CUDA、BLAS 和 PyTorch 构建仍可能造成轻微数值差异。

### 评估

运行冻结 T+1 执行路线：

```bash
python scripts/backtest/run_clean_resid_mainline.py
```

运行冻结最终 optimizer：

```bash
python scripts/portfolio/run_final_mainline_optimizer.py
```

optimizer 配置：

```text
configs/portfolio/final_mainline_optimizer.yaml
```

必要输入：

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet
data/mart/datasets/core/dataset_v20260526.parquet
data/mart/labels/execution_labels_v20260526.parquet
```

如果缺少 `execution_labels_v20260526.parquet`，先查看标签构建脚本参数：

```bash
python scripts/data/build_execution_labels.py --help
```

### 审计与分析

point-in-time 泄漏审计：

```bash
python scripts/audit/audit_point_in_time.py --labels data/mart/labels/labels_v20260526.parquet --out-dir outputs/audit/point_in_time
```

残差 alpha 审计：

```bash
python scripts/audit/audit_barra_lite_residual_alpha.py
```

闭环摘要：

```bash
python scripts/analysis/summarize_model_closed_loop.py
```

optimizer validation 归因：

```bash
python scripts/analysis/analyze_optimizer_validation_attribution.py --periods outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_periods.csv --summary outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80/soft_optimizer_grid_summary.csv --output-dir outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution --split validation --top-n 6
```

如需重建 optimizer grid：

```bash
python scripts/portfolio/run_soft_optimizer_grid.py --predictions outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet --output-dir outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80
```

### 完整复现流程

```bash
python -m compileall scripts src pipelines
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260525
python scripts/features/validate_clean_feature_set.py
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20 60
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --dry-run --device cpu
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
python scripts/backtest/run_clean_resid_mainline.py
python scripts/portfolio/run_soft_optimizer_grid.py --predictions outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet --output-dir outputs/backtest/optimizer/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80
python scripts/portfolio/run_final_mainline_optimizer.py
python scripts/audit/audit_point_in_time.py --labels data/mart/labels/labels_v20260526.parquet --out-dir outputs/audit/point_in_time
python scripts/audit/audit_barra_lite_residual_alpha.py
python scripts/analysis/summarize_model_closed_loop.py
```

预期最终结果目录：

```text
outputs/backtest/optimizer/final_mainline_ckptscore_e12/
  final_optimizer_periods.csv
  final_optimizer_summary.csv
  manifest.json
```

冻结结果核心读数：

| split | net_ann | net_ir | excess_benchmark_ann | excess_exec_universe_ann | avg_invested_weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | -0.059635 | -0.013411 | 0.047074 | -0.028238 | 0.839216 |
| test | 0.268252 | 0.189517 | -0.258701 | -0.035435 | 0.893154 |

解释：最终 optimizer 完成了可复现研究闭环，但 validation 和 test 均未稳定取得相对可执行域正超额，因此不能描述为稳定生产 alpha。

### 文档

项目可仅通过本 README 复现。其他文档提供可选背景：

```text
docs/00_project_evolution.md
docs/01_data_and_label_protocol.md
docs/02_feature_engineering_clean_v1.md
docs/03_model_baseline_evolution.md
docs/06_results_and_limitations.md
docs/07_clean_dataset_v20260526_dictionary.md
docs/08_feature_style_interaction_wide30_closed_loop.md
docs/09_final_mainline_freeze.md
docs/experiment_report.md
```
