# scripts 使用手册

本文是仓库中所有 `scripts/` 命令行入口的统一说明。第一次拿到仓库时，只需要阅读这份文档即可知道如何运行数据、训练、回测、优化、审计和实盘流程。

`scripts/` 只承担“外部可执行入口”职责；核心实现位于 `pipelines/` 和 `src/`。你可以把这里的脚本理解为一组面向用户的对象：

| 对象 | 目录 | 负责什么 |
|---|---|---|
| `DataPipeline` | `scripts/data/`、`scripts/run_daily_dag.py` | 从原始行情构建 lake、pool、market state、mart、label |
| `FeatureModelingPipeline` | `scripts/features/`、`scripts/modeling/` | 校验特征合同、构建训练张量、训练模型 |
| `BacktestPipeline` | `scripts/backtest/` | T+1 成交仿真和主线回测 |
| `PortfolioPipeline` | `scripts/portfolio/` | 组合优化、参数网格、容量敏感性 |
| `AuditAnalysisPipeline` | `scripts/audit/`、`scripts/analysis/` | 泄漏审计、残差 alpha 审计、闭环报告、归因分析 |
| `LivePipeline` | `scripts/live/` | 盘前输入、推理、优化、订单、手工成交记录、收盘估值 |

## 运行前准备

所有命令都从仓库根目录执行。

```powershell
conda activate dl_env
```

如果不想激活环境，也可以使用：

```powershell
conda run -n dl_env python <script> <args>
```

推荐先做一次语法检查：

```powershell
python -m compileall scripts
```

常用数据版本为：

```text
v20260526
```

## 最短完整离线流程

第一次运行项目，建议按下面顺序执行。

### 1. 构建离线数据 DAG

一条命令跑完整数据链路：

```powershell
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260525
```

日常增量更新：

```powershell
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20260526 --end-date 20260529 --incremental
```

如果需要拆开执行：

```powershell
python scripts/data/run_ingest_raw.py --data-version v20260526
python scripts/data/run_build_pool.py --data-version v20260526
python scripts/data/run_build_market_state.py --data-version v20260526 --incremental
python scripts/data/validate_market_state_coverage.py --data-version v20260526 --start-date 20160104 --end-date 20260525 --strict
python scripts/data/run_build_mart.py --data-version v20260526 --start-date 20160104 --end-date 20260525
```

辅助数据脚本：

```powershell
python scripts/data/validate_ingest_schema.py
python scripts/data/query_market_state.py --help
python scripts/data/build_execution_labels.py --help
python scripts/data/build_canonical_labels.py --help
```

## 特征与训练

### 2. 校验 clean feature 合同

```powershell
python scripts/features/validate_clean_feature_set.py
```

如需重新生成特征角色表：

```powershell
python scripts/features/generate_feature_role_tags.py
```

### 3. 构建模型训练张量

构建 alpha-only 数据集：

```powershell
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_only --lookbacks 20
```

构建最终主线使用的 alpha + residual style 数据集：

```powershell
python scripts/modeling/build_clean_model_datasets.py --data-version v20260526 --build-mode alpha_plus_residual_style --lookbacks 20,60
```

### 4. 训练模型

先 dry-run，确认配置、数据和模型能正确创建：

```powershell
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --dry-run --device cpu
```

正式训练：

```powershell
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cuda
```

没有 GPU 时：

```powershell
python scripts/modeling/train_sequence.py --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml --device cpu
```

训练输出通常在：

```text
outputs/runs/<run_name>/
```

其中最重要的是：

```text
predictions.parquet
metrics.json
model.pt
config.yaml
```

## 回测与组合优化

### 5. T+1 成交仿真

对一个训练 run 的预测结果做 T+1 回测：

```powershell
python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet
```

指定输出目录：

```powershell
python scripts/backtest/backtest_t1_fill_sim.py --predictions outputs/runs/<run_name>/predictions.parquet --output-dir outputs/backtest/<run_name>_t1
```

运行冻结主线 T+1 配置：

```powershell
python scripts/backtest/run_clean_resid_mainline.py
```

批量评估已存在的 clean dataset runs：

```powershell
python scripts/backtest/run_clean_dataset_execution_stack.py --only-existing
```

### 6. 组合优化

运行最终冻结 optimizer：

```powershell
python scripts/portfolio/run_final_mainline_optimizer.py
```

运行 soft optimizer 参数网格：

```powershell
python scripts/portfolio/run_soft_optimizer_grid.py --predictions outputs/runs/<run_name>/predictions.parquet --output-dir outputs/backtest/optimizer/<run_name>_grid
```

运行容量/participation 敏感性矩阵：

```powershell
python scripts/portfolio/run_capacity_participation_matrix.py --predictions outputs/runs/<run_name>/predictions.parquet --output-dir outputs/backtest/capacity/<run_name>
```

底层 optimizer 也可以直接运行：

```powershell
python scripts/portfolio/optimize_feasible_cash_buffer.py --predictions outputs/runs/<run_name>/predictions.parquet --output-dir outputs/backtest/optimizer/<run_name>
```

## 审计与分析

### 7. 生产就绪审计

点时间泄漏审计：

```powershell
python scripts/audit/audit_point_in_time.py
```

Barra-lite 残差 alpha 审计：

```powershell
python scripts/audit/audit_barra_lite_residual_alpha.py
```

冻结主线深度审计：

```powershell
python scripts/audit/audit_clean_resid_mainline.py
```

### 8. 闭环分析和 optimizer 归因

生成模型训练到回测、优化的闭环摘要：

```powershell
python scripts/analysis/summarize_model_closed_loop.py
```

分析 optimizer validation 归因：

```powershell
python scripts/analysis/analyze_optimizer_validation_attribution.py --periods outputs/backtest/optimizer/<grid_dir>/soft_optimizer_grid_periods.csv --summary outputs/backtest/optimizer/<grid_dir>/soft_optimizer_grid_summary.csv --output-dir outputs/analysis/<analysis_name> --split validation --top-n 6
```

## 实盘流程

实盘脚本默认读取：

```text
configs/live/live_trading.yaml
```

当前实盘主线是手工交易方案：脚本生成订单，人工完成买卖，再用脚本录入真实成交并做收盘估值。

### 9. 一条命令跑盘前流程

使用已有数据输入：

```powershell
python scripts/live/live_daily.py --skip-dag --trade-date 20260601 --feature-date 20260529
```

先跑数据增量，再跑盘前推理、优化和订单：

```powershell
python scripts/live/live_daily.py --run-dag --data-version v20260526 --end-date 20260529 --trade-date 20260601
```

全量 DAG + 实盘盘前流程：

```powershell
python scripts/live/live_daily.py --run-dag --full-dag --data-version v20260526 --end-date 20260529 --trade-date 20260601
```

生成订单后立即进入手工成交录入：

```powershell
python scripts/live/live_daily.py --skip-dag --trade-date 20260601 --feature-date 20260529 --execute --no-push
```

Windows PowerShell 顶层入口等价于 `live_daily.py`：

```powershell
.\run_live_trading_pipeline.ps1 -SkipDag -TradeDate 20260601 -FeatureDate 20260529
```

### 10. 拆开执行实盘阶段

准备 live feature panel、持仓输入和价格快照：

```powershell
python scripts/live/00_prepare_live_inputs.py --config configs/live/live_trading.yaml --data-version v20260526 --trade-date 20260601 --feature-date 20260529
```

只准备特征：

```powershell
python scripts/live/00_prepare_live_inputs.py --trade-date 20260601 --feature-date 20260529 --skip-prepare-account-inputs
```

只准备账户输入，复用已有 feature parquet：

```powershell
python scripts/live/00_prepare_live_inputs.py --trade-date 20260601 --feature-date 20260529 --features-parquet data/live/features/features_20260529.parquet --skip-prepare-features
```

模型推理：

```powershell
python scripts/live/01_live_inference.py --config configs/live/live_trading.yaml --trade-date 20260601 --feature-date 20260529
```

组合优化：

```powershell
python scripts/live/02_live_optimization.py --config configs/live/live_trading.yaml --trade-date 20260601 --feature-date 20260529
```

生成目标订单：

```powershell
python scripts/live/03_generate_target_orders.py --config configs/live/live_trading.yaml --trade-date 20260601
```

手工录入真实成交：

```powershell
python scripts/live/05_interactive_execution.py --config configs/live/live_trading.yaml --trade-date 20260601 --no-push
```

收盘估值：

```powershell
python scripts/live/06_close_valuation.py --config configs/live/live_trading.yaml --trade-date 20260601 --write-close-positions
```

如果需要在收盘估值前按成交记录重建状态：

```powershell
python scripts/live/06_close_valuation.py --config configs/live/live_trading.yaml --trade-date 20260601 --rebuild-state-from-execution --write-close-positions
```

## 当前脚本索引

### DataPipeline

| 脚本 | 何时使用 |
|---|---|
| `scripts/run_daily_dag.py` | 推荐的数据总入口。 |
| `scripts/data/run_ingest_raw.py` | 单独执行 raw ingestion。 |
| `scripts/data/run_build_pool.py` | 单独构建股票池 SCD2。 |
| `scripts/data/run_build_market_state.py` | 单独构建 market state。 |
| `scripts/data/run_build_mart.py` | 单独构建 mart/features。 |
| `scripts/data/query_market_state.py` | 查询 market state。 |
| `scripts/data/validate_ingest_schema.py` | ingest 前 schema 检查。 |
| `scripts/data/validate_market_state_coverage.py` | market state 覆盖检查。 |
| `scripts/data/build_execution_labels.py` | 重新生成 execution labels。 |
| `scripts/data/build_canonical_labels.py` | 重新生成 canonical labels。 |

### FeatureModelingPipeline

| 脚本 | 何时使用 |
|---|---|
| `scripts/features/generate_feature_role_tags.py` | 重建 feature role/tag 表。 |
| `scripts/features/validate_clean_feature_set.py` | 检查 clean feature 配置和 mart 字段。 |
| `scripts/modeling/build_clean_model_datasets.py` | 构建训练 NPZ。 |
| `scripts/modeling/train_sequence.py` | 训练 GRU-family 模型。 |

### BacktestPipeline

| 脚本 | 何时使用 |
|---|---|
| `scripts/backtest/backtest_t1_fill_sim.py` | 对预测结果做 T+1 回测。 |
| `scripts/backtest/run_clean_resid_mainline.py` | 跑冻结主线 T+1 配置。 |
| `scripts/backtest/run_clean_dataset_execution_stack.py` | 批量评估多个固定 run。 |

### PortfolioPipeline

| 脚本 | 何时使用 |
|---|---|
| `scripts/portfolio/optimize_feasible_cash_buffer.py` | 直接运行 LP optimizer。 |
| `scripts/portfolio/run_soft_optimizer_grid.py` | 参数网格搜索。 |
| `scripts/portfolio/run_final_mainline_optimizer.py` | 最终冻结 optimizer。 |
| `scripts/portfolio/run_capacity_participation_matrix.py` | 容量/参与率敏感性。 |

### AuditAnalysisPipeline

| 脚本 | 何时使用 |
|---|---|
| `scripts/audit/audit_point_in_time.py` | 点时间/泄漏审计。 |
| `scripts/audit/audit_barra_lite_residual_alpha.py` | 残差 alpha 审计。 |
| `scripts/audit/audit_clean_resid_mainline.py` | 冻结主线深度审计。 |
| `scripts/analysis/summarize_model_closed_loop.py` | 训练-回测-优化闭环摘要。 |
| `scripts/analysis/analyze_optimizer_validation_attribution.py` | optimizer validation 归因。 |

### LivePipeline

| 脚本 | 何时使用 |
|---|---|
| `scripts/live/live_daily.py` | 推荐的实盘盘前总入口。 |
| `scripts/live/00_prepare_live_inputs.py` | 准备实盘特征和账户输入。 |
| `scripts/live/00_prepare_live_features.py` | 单独准备实盘特征。通常由 `00_prepare_live_inputs.py` 调用。 |
| `scripts/live/00_prepare_live_account_inputs.py` | 单独准备持仓和价格快照。通常由 `00_prepare_live_inputs.py` 调用。 |
| `scripts/live/01_live_inference.py` | 实盘推理。 |
| `scripts/live/02_live_optimization.py` | 实盘目标权重优化。 |
| `scripts/live/03_generate_target_orders.py` | 生成手工交易订单。 |
| `scripts/live/05_interactive_execution.py` | 录入真实成交。 |
| `scripts/live/06_close_valuation.py` | 收盘估值。 |
| `scripts/live/common.py` | live 公共库，不直接运行。 |

## 排错建议

查看任意脚本参数：

```powershell
python <script> --help
```

确认 live 脚本语法：

```powershell
python -m compileall scripts/live
```

确认当前 Git 改动：

```powershell
git status --short
```

常见问题：

| 现象 | 处理 |
|---|---|
| 找不到输入 parquet/csv | 先运行 `scripts/run_daily_dag.py` 或检查 `configs/data/data.yaml`、`configs/live/live_trading.yaml` 中的路径。 |
| 训练脚本找不到 NPZ | 先运行 `scripts/modeling/build_clean_model_datasets.py`。 |
| 实盘推理找不到 feature panel | 先运行 `scripts/live/00_prepare_live_inputs.py`。 |
| optimizer 找不到 positions 或 price snapshot | 先运行 `scripts/live/00_prepare_live_inputs.py`，或显式传入 `--positions`、`--price-snapshot`。 |
| CUDA 不可用 | 把 `--device cuda` 改成 `--device cpu`。 |

