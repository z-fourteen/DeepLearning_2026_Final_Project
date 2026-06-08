# Scripts Index

## English

This directory contains command-line entry points. The root `README.md` is the single reproduction guide; use this file only as a script map when maintaining or extending the project.

Run all commands from the repository root.

### Data Pipeline

| Script | Purpose |
| --- | --- |
| `scripts/run_daily_dag.py` | Runs the full data DAG: ingest, pool, market state, validation, and mart build. |
| `scripts/data/run_ingest_raw.py` | Ingests raw source files into `data/lake/raw/`. |
| `scripts/data/run_build_pool.py` | Builds the ChiNext pool SCD2 table. |
| `scripts/data/run_build_market_state.py` | Builds tradability, listing, suspension, and limit-state features. |
| `scripts/data/validate_market_state_coverage.py` | Validates date and universe coverage. |
| `scripts/data/run_build_mart.py` | Builds daily features, labels, and core mart datasets. |
| `scripts/data/build_execution_labels.py` | Builds execution labels used by T+1 backtests and optimizer evaluation. |
| `scripts/data/build_canonical_labels.py` | Builds canonical labels for audit variants. |
| `scripts/data/query_market_state.py` | Queries generated market-state records. |
| `scripts/data/validate_ingest_schema.py` | Checks source schema compatibility. |

### Feature And Modeling

| Script | Purpose |
| --- | --- |
| `scripts/features/validate_clean_feature_set.py` | Validates the clean feature contract against mart fields. |
| `scripts/features/generate_feature_role_tags.py` | Regenerates feature role/tag metadata. |
| `scripts/modeling/build_clean_model_datasets.py` | Builds clean sequence tensors under `data/mart/datasets/clean_purged_wf/`. |
| `scripts/modeling/train_sequence.py` | Trains GRU-family sequence models from YAML configs. |

### Backtest And Portfolio

| Script | Purpose |
| --- | --- |
| `scripts/backtest/backtest_t1_fill_sim.py` | Runs T+1 fill simulation from prediction parquet files. |
| `scripts/backtest/run_clean_resid_mainline.py` | Runs the frozen T+1 mainline config. |
| `scripts/backtest/run_clean_dataset_execution_stack.py` | Batch-runs fixed clean-dataset execution stacks. |
| `scripts/portfolio/optimize_feasible_cash_buffer.py` | Runs the lower-level feasible-cash optimizer. |
| `scripts/portfolio/run_soft_optimizer_grid.py` | Runs soft-optimizer parameter grids. |
| `scripts/portfolio/run_final_mainline_optimizer.py` | Runs the frozen final optimizer. |
| `scripts/portfolio/run_capacity_participation_matrix.py` | Runs capacity and participation-rate sensitivity checks. |

### Audit And Analysis

| Script | Purpose |
| --- | --- |
| `scripts/audit/audit_point_in_time.py` | Audits point-in-time feature and label safety. |
| `scripts/audit/audit_barra_lite_residual_alpha.py` | Audits residual alpha after Barra-lite controls. |
| `scripts/audit/audit_clean_resid_mainline.py` | Runs a deeper audit of the clean residual mainline. |
| `scripts/analysis/summarize_model_closed_loop.py` | Summarizes training, prediction diagnostics, T+1, and optimizer evidence. |
| `scripts/analysis/analyze_optimizer_validation_attribution.py` | Analyzes optimizer validation attribution and weak periods. |

### Live Workflow

The live workflow is an extension of the offline experiment and is not required for reproducing the assignment results.

| Script | Purpose |
| --- | --- |
| `scripts/live/live_daily.py` | Live daily orchestration entry point. |
| `scripts/live/00_prepare_live_inputs.py` | Prepares live features, positions, and price snapshots. |
| `scripts/live/01_live_inference.py` | Runs live inference. |
| `scripts/live/02_live_optimization.py` | Runs live target-weight optimization. |
| `scripts/live/03_generate_target_orders.py` | Generates target orders. |
| `scripts/live/05_interactive_execution.py` | Records manual fills. |
| `scripts/live/06_close_valuation.py` | Performs close valuation and state updates. |

Use `python <script> --help` for argument details.

## 中文

本目录保存命令行入口。根目录 `README.md` 是唯一复现指南；本文件仅作为维护和扩展项目时使用的脚本索引。

所有命令都应从仓库根目录运行。

### 数据流水线

| 脚本 | 作用 |
| --- | --- |
| `scripts/run_daily_dag.py` | 运行完整数据 DAG：接入、股票池、市场状态、校验和 mart 构建。 |
| `scripts/data/run_ingest_raw.py` | 将原始数据接入 `data/lake/raw/`。 |
| `scripts/data/run_build_pool.py` | 构建创业板股票池 SCD2 表。 |
| `scripts/data/run_build_market_state.py` | 构建可交易性、上市、停牌和涨跌停状态特征。 |
| `scripts/data/validate_market_state_coverage.py` | 校验日期和股票池覆盖率。 |
| `scripts/data/run_build_mart.py` | 构建日频特征、标签和核心 mart 数据集。 |
| `scripts/data/build_execution_labels.py` | 构建 T+1 回测和优化器评估使用的执行标签。 |
| `scripts/data/build_canonical_labels.py` | 构建审计变体使用的标准标签。 |
| `scripts/data/query_market_state.py` | 查询已生成的市场状态记录。 |
| `scripts/data/validate_ingest_schema.py` | 检查源数据 schema 兼容性。 |

### 特征与建模

| 脚本 | 作用 |
| --- | --- |
| `scripts/features/validate_clean_feature_set.py` | 校验 clean 特征契约是否与 mart 字段一致。 |
| `scripts/features/generate_feature_role_tags.py` | 重新生成特征角色和标签元数据。 |
| `scripts/modeling/build_clean_model_datasets.py` | 在 `data/mart/datasets/clean_purged_wf/` 下构建 clean 序列张量。 |
| `scripts/modeling/train_sequence.py` | 基于 YAML 配置训练 GRU 系列序列模型。 |

### 回测与组合

| 脚本 | 作用 |
| --- | --- |
| `scripts/backtest/backtest_t1_fill_sim.py` | 基于预测 parquet 文件运行 T+1 成交仿真。 |
| `scripts/backtest/run_clean_resid_mainline.py` | 运行冻结的 T+1 主线配置。 |
| `scripts/backtest/run_clean_dataset_execution_stack.py` | 批量运行固定的 clean-dataset 执行栈。 |
| `scripts/portfolio/optimize_feasible_cash_buffer.py` | 运行底层可行现金缓冲优化器。 |
| `scripts/portfolio/run_soft_optimizer_grid.py` | 运行 soft optimizer 参数网格。 |
| `scripts/portfolio/run_final_mainline_optimizer.py` | 运行冻结的最终优化器。 |
| `scripts/portfolio/run_capacity_participation_matrix.py` | 运行容量和参与率敏感性检查。 |

### 审计与分析

| 脚本 | 作用 |
| --- | --- |
| `scripts/audit/audit_point_in_time.py` | 审计特征和标签的 point-in-time 安全性。 |
| `scripts/audit/audit_barra_lite_residual_alpha.py` | 审计 Barra-lite 控制后的残差 alpha。 |
| `scripts/audit/audit_clean_resid_mainline.py` | 对 clean residual 主线执行更深入的审计。 |
| `scripts/analysis/summarize_model_closed_loop.py` | 汇总训练、预测诊断、T+1 和优化器证据。 |
| `scripts/analysis/analyze_optimizer_validation_attribution.py` | 分析优化器验证集归因和弱势区间。 |

### 实盘扩展流程

实盘流程是离线实验之后的扩展，不属于课程作业结果复现的必需部分。

| 脚本 | 作用 |
| --- | --- |
| `scripts/live/live_daily.py` | 实盘日度编排入口。 |
| `scripts/live/00_prepare_live_inputs.py` | 准备实盘特征、持仓和价格快照。 |
| `scripts/live/01_live_inference.py` | 执行实盘推理。 |
| `scripts/live/02_live_optimization.py` | 执行实盘目标权重优化。 |
| `scripts/live/03_generate_target_orders.py` | 生成目标订单。 |
| `scripts/live/05_interactive_execution.py` | 记录人工成交。 |
| `scripts/live/06_close_valuation.py` | 执行收盘估值和状态更新。 |

参数细节可使用 `python <script> --help` 查看。
