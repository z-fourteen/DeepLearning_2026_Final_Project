# 深度学习股票趋势预测与模拟交易复现说明

本仓库对应课程大作业《基于深度学习的股票趋势预测与模拟交易》。项目围绕创业板股票池，完成了从 A 股日频历史数据、特征工程、序列模型训练、历史回测、组合优化到同花顺模拟交易复盘的一条完整研究流程。

报告中的两条模型线都已经合并在 `Final/` 内：

| 模型线 | 定位 | 冻结版本 |
| --- | --- | --- |
| GRU | 实盘主力模型 | `feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean` |
| Transformer | 结构对照与消融模型 | `StdTF-l60-cls-mse-f13` |

数据、模型权重、序列张量、预测结果和回测输出不随代码提交。它们是本地运行产物，可按本文命令重新生成。

## 1. 仓库结构

```text
Final/
  configs/        数据、切分、特征、模型、回测、组合优化和 live 配置
  data/           本地数据工作区，保存原始数据映射和生成的 parquet/npz
  docs/           实验记录、报告源码和最终 PDF
  legacy/         旧版 full62 特征池与早期实验归档
  meta/           schema registry、数据版本和元数据
  outputs/        训练、预测、回测、审计和 live 输出
  pipelines/      ingest、pool、state、mart、clean dataset 等流水线实现
  scripts/        所有命令行入口
  src/            数据集、模型、训练器、损失函数和评价指标
```

最常用入口如下：

| 流程 | 入口 |
| --- | --- |
| 原始数据接入、股票池、交易状态、mart 构建 | `scripts/run_daily_dag.py` |
| clean 特征契约校验 | `scripts/features/validate_clean_feature_set.py` |
| 序列张量构建 | `scripts/modeling/build_clean_model_datasets.py` |
| GRU / Transformer 统一训练 | `scripts/modeling/train_sequence.py` |
| GRU 主线 T+1 回测 | `scripts/backtest/run_clean_resid_mainline.py` |
| Transformer 5 日调仓回测 | `scripts/backtest/run_transformer_t1_5d_backtests.ps1` |
| Transformer 每日调仓回测 | `scripts/backtest/run_transformer_daily_rebalance_backtests.ps1` |
| 最终组合优化 | `scripts/portfolio/run_final_mainline_optimizer.py` |
| 模拟交易日度流程 | `run_live_trading_pipeline.ps1` |

## 2. 环境配置

推荐 Python 3.10 或兼容版本。模型训练建议使用 CUDA GPU；CPU 可用于 dry-run、语法检查和小规模流程检查。

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m compileall scripts src pipelines
```

也可以根据 `environment.yml` 创建 Conda 环境。`run_live_trading_pipeline.ps1`
默认通过 `conda run -n dl_env python ...` 调用实盘流程；如果本机环境名不同，
需要修改脚本中的环境名，或直接运行各阶段 Python 脚本。

## 3. 数据准备

默认原始数据根目录在 `configs/data/data.yaml` 中配置：

```yaml
source:
  root_dir: "A股数据"
```

将课程提供的数据放到仓库根目录下的 `A股数据/`，或修改上述配置为本机实际路径。期望结构如下：

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

项目的关键数据约定：

| 项目 | 设定 |
| --- | --- |
| 股票池 | 创业板指数成分股，`399006.SZ` |
| 数据版本 | `v20260526` |
| 历史构建区间 | `20160104` 至 `20260525` |
| 预测标签 | 未来 5 个交易日个股收益减创业板基准收益 |
| 数据切分 | `configs/data/splits.yaml` 中的 `final_2025_2026` |
| 训练集 | `20160104` 至 `20221231` |
| 验证集 | `20230201` 至 `20241231` |
| 测试集 | `20250201` 至 `20260525` |

切分使用 purged walk-forward 口径，标签 horizon 为 5 日，并设置 purge 与 embargo，避免相邻区间之间的标签和窗口重叠泄露。

## 4. 历史数据流水线

从仓库根目录运行完整离线 DAG：

```powershell
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260525
```

该命令依次完成：

1. `run_ingest_raw.py`：读取 `A股数据/`，生成 `data/lake/raw/`。
2. `run_build_pool.py`：根据创业板指数权重和基础信息生成股票池 SCD2 表。
3. `run_build_market_state.py`：生成 ST、停牌、涨跌停、价格有效性、成交量有效性等交易状态。
4. `validate_market_state_coverage.py`：检查历史日期和股票池覆盖。
5. `run_build_mart.py`：构建日频特征、标签和核心建模表。

主要产物：

```text
data/lake/raw/
data/lake/core/chinext_pool/chinext_pool_scd2.parquet
data/lake/state/security_daily_state.parquet
data/mart/features_daily/features_daily_v20260526.parquet
data/mart/labels/labels_v20260526.parquet
data/mart/labels/execution_labels_v20260526.parquet
data/mart/datasets/core/dataset_v20260526.parquet
```

若只做日度增量更新，可使用：

```powershell
python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260601 --incremental
```

增量模式会避免误触发大范围 mart 重建；如确实需要在增量模式下重建 mart，需要显式加 `--rebuild-mart`。

## 5. 特征契约与序列张量

先校验 clean 特征契约：

```powershell
python scripts/features/validate_clean_feature_set.py
```

当前特征配置为：

```text
configs/features/advanced_sequence_clean_v1.yaml
```

该配置把字段分为 alpha 特征、risk controls、tradability controls 和 residual-style
候选特征。模型输入只使用 point-in-time 特征；交易状态和执行约束保留给过滤、
回测和诊断。

构建 GRU 最终模型使用的 18 维张量，包含 13 个 alpha 特征和 5 个 residual-style 特征：

```powershell
python scripts/modeling/build_clean_model_datasets.py `
  --data-version v20260526 `
  --build-mode alpha_plus_residual_style `
  --lookbacks 20 60
```

构建 Transformer 最终模型使用的 13 维 alpha-only 张量：

```powershell
python scripts/modeling/build_clean_model_datasets.py `
  --data-version v20260526 `
  --build-mode alpha_only `
  --lookbacks 20 60
```

输出目录：

```text
data/mart/datasets/clean_purged_wf/
```

关键文件示例：

```text
dataset_seq_l60_adv_clean_v1_alpha_resid_style_chinext_purged_walk_forward.npz
dataset_seq_l60_adv_clean_v1_alpha_only_chinext_purged_walk_forward.npz
*_sidecar.parquet
*_manifest.json
*_filter_log.csv
```

其中 `.npz` 保存 `X`、`y`、`trade_date`、`ts_code`、`split` 和 `feature_names`；sidecar 保存控制字段和过滤信息；manifest 记录数据版本、切分、特征列表和样本统计。

## 6. 模型训练

统一训练入口是：

```text
scripts/modeling/train_sequence.py
```

训练脚本根据配置中的 `model.name` 自动选择模型：

| `model.name` | 模型类 |
| --- | --- |
| `gru_baseline` | `GRUStockModel` |
| `feature_style_interaction_gru` | `FeatureStyleInteractionGRUStockModel` |
| `regime_gated_gru` | `RegimeGatedGRUStockModel` |
| `transformer_encoder` | `TransformerStockModel` |
| `transformer_enhanced` | `EnhancedTransformerModel` |

### 6.1 训练 GRU 主线

先 dry-run，确认配置、张量维度和样本数量：

```powershell
python scripts/modeling/train_sequence.py `
  --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml `
  --dry-run `
  --device cpu
```

正式训练：

```powershell
python scripts/modeling/train_sequence.py `
  --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml `
  --device cuda
```

报告采用 `checkpoint_score` 选择的 epoch 12 作为 GRU 冻结版本。训练输出：

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/
  config.yaml
  metrics.json
  model.pt
  predictions.parquet
```

### 6.2 训练 Transformer 冻结版本

```powershell
python scripts/modeling/train_sequence.py --config configs/models/StdTF-l60-cls-mse-f13.yaml --dry-run --device cpu
python scripts/modeling/train_sequence.py --config configs/models/StdTF-l60-cls-mse-f13.yaml --device cuda
```

用于报告消融的主要配置：

```text
configs/models/StdTF-l20-cls-mse-f13.yaml
configs/models/StdTF-l60-cls-mse-f13.yaml
configs/models/StdTF-l60-cls-mse-f18.yaml
configs/models/StdTF-l60-cls-huber-f13.yaml
configs/models/StdTF-l60-attn-mse-f13.yaml
configs/models/EnhancedTF-l60-cls-mse-f13.yaml
```

训练完成后，每个模型都应在 `outputs/runs/<run_name>/` 下生成 `model.pt`、`metrics.json` 和 `predictions.parquet`。

## 7. 历史回测与组合优化

### 7.1 GRU T+1 执行回测

```powershell
python scripts/backtest/run_clean_resid_mainline.py
```

该命令使用：

```text
configs/backtest/clean_resid_t1_top20_keep2.yaml
```

通用 T+1 回测器为：

```powershell
python scripts/backtest/backtest_t1_fill_sim.py `
  --predictions outputs/runs/<run_name>/predictions.parquet `
  --output-dir outputs/backtest/t1_fill_sim/<run_name>
```

### 7.2 GRU 最终组合优化

```powershell
python scripts/portfolio/run_final_mainline_optimizer.py
```

配置文件：

```text
configs/portfolio/final_mainline_optimizer.yaml
```

主要输入：

```text
outputs/runs/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean/predictions.parquet
data/mart/datasets/core/dataset_v20260526.parquet
data/mart/labels/execution_labels_v20260526.parquet
```

主要输出：

```text
outputs/backtest/optimizer/final_mainline_ckptscore_e12/
```

### 7.3 Transformer 回测

5 日调仓 T+1 回测：

```powershell
.\scripts\backtest\run_transformer_t1_5d_backtests.ps1 -Python python
```

每日调仓回测：

```powershell
.\scripts\backtest\run_transformer_daily_rebalance_backtests.ps1 -Python python
```

这两个脚本会批量读取报告中的 Transformer 消融模型预测文件，并把结果写入 `outputs/backtest/`。

## 8. 模拟交易与 live 流程

比赛窗口为 `2026-06-01` 至 `2026-06-12`。日度流程的一键入口：

```powershell
.\run_live_trading_pipeline.ps1 -TradeDate 20260601
```

如果需要先刷新历史数据并准备当日特征：

```powershell
.\run_live_trading_pipeline.ps1 -RunDag -DataVersion v20260526 -EndDate 20260601 -TradeDate 20260601
```

主要阶段：

| 阶段 | 脚本 | 产物 |
| --- | --- | --- |
| 准备 live 输入 | `scripts/live/00_prepare_live_inputs.py` | `data/live/features/`、`data/live/account/` |
| 推理 | `scripts/live/01_live_inference.py` | `outputs/live_predictions/predictions_YYYYMMDD.*` |
| 组合优化 | `scripts/live/02_live_optimization.py` | `outputs/live_targets/target_weights_YYYYMMDD.csv` |
| 订单生成 | `scripts/live/03_generate_target_orders.py` | `outputs/live_orders/orders_YYYYMMDD.csv` |
| 人工成交记录 | `scripts/live/05_interactive_execution.py` | `outputs/live/orders/execution_YYYYMMDD.json` |
| 收盘估值 | `scripts/live/06_close_valuation.py` | `outputs/live/valuations/valuation_YYYYMMDD.*` |

若要复现实盘窗口内的 GRU 理论表现，可运行：

```powershell
python scripts/backtest/backtest_gru_live_strategy_replay.py
python scripts/backtest/backtest_live_same_day_rebalance.py
```

若要补齐 Transformer 在比赛窗口内的 live 预测：

```powershell
python scripts/live/backfill_transformer_live_predictions.py `
  --yxr-root . `
  --daily-root ..\Final-OXX2\A股数据\daily `
  --trade-dates 20260605 20260608 20260609 20260610 20260611 20260612 `
  --overwrite
```

## 9. 审计与结果汇总

Point-in-time 审计：

```powershell
python scripts/audit/audit_point_in_time.py `
  --labels data/mart/labels/labels_v20260526.parquet `
  --out-dir outputs/audit/point_in_time
```

残差 alpha 审计：

```powershell
python scripts/audit/audit_barra_lite_residual_alpha.py
```

clean residual 主线审计：

```powershell
python scripts/audit/audit_clean_resid_mainline.py
```

闭环汇总：

```powershell
python scripts/analysis/summarize_model_closed_loop.py
```

优化器验证集归因：

```powershell
$OptDir = "outputs/backtest/optimizer/" +
  "feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean_ckptscore_e12_core80"

python scripts/analysis/analyze_optimizer_validation_attribution.py `
  --periods "$OptDir/soft_optimizer_grid_periods.csv" `
  --summary "$OptDir/soft_optimizer_grid_summary.csv" `
  --output-dir outputs/analysis/feature_style_interaction_gru_l60_ckptscore_e12_validation_attribution `
  --split validation `
  --top-n 6
```

## 10. 从零复现推荐顺序

按报告流程完整复现时，建议按以下顺序执行：

```powershell
python -m compileall scripts src pipelines

python scripts/run_daily_dag.py --data-version v20260526 --start-date 20160104 --end-date 20260525
python scripts/features/validate_clean_feature_set.py

python scripts/modeling/build_clean_model_datasets.py `
  --data-version v20260526 `
  --build-mode alpha_plus_residual_style `
  --lookbacks 20 60

python scripts/modeling/build_clean_model_datasets.py `
  --data-version v20260526 `
  --build-mode alpha_only `
  --lookbacks 20 60

python scripts/modeling/train_sequence.py `
  --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml `
  --dry-run `
  --device cpu

python scripts/modeling/train_sequence.py `
  --config configs/models/feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean.yaml `
  --device cuda

python scripts/modeling/train_sequence.py --config configs/models/StdTF-l60-cls-mse-f13.yaml --dry-run --device cpu
python scripts/modeling/train_sequence.py --config configs/models/StdTF-l60-cls-mse-f13.yaml --device cuda

python scripts/backtest/run_clean_resid_mainline.py
python scripts/portfolio/run_final_mainline_optimizer.py

.\scripts\backtest\run_transformer_t1_5d_backtests.ps1 -Python python
.\scripts\backtest\run_transformer_daily_rebalance_backtests.ps1 -Python python

python scripts/audit/audit_point_in_time.py `
  --labels data/mart/labels/labels_v20260526.parquet `
  --out-dir outputs/audit/point_in_time
python scripts/audit/audit_barra_lite_residual_alpha.py
python scripts/analysis/summarize_model_closed_loop.py
```

实盘流程按交易日单独运行：

```powershell
.\run_live_trading_pipeline.ps1 -RunDag -DataVersion v20260526 -EndDate 20260601 -TradeDate 20260601
.\run_live_trading_pipeline.ps1 -RunDag -DataVersion v20260526 -EndDate 20260602 -TradeDate 20260602
```

## 11. 提交边界

建议提交：

```text
configs/
docs/
legacy/
meta/
pipelines/
scripts/
src/
README.md
requirements.txt
environment.yml
run_live_trading_pipeline.ps1
```

不建议提交：

```text
A股数据/
data/lake/
data/mart/
data/live/
outputs/
logs/
*.pt
*.parquet
*.npz
```

这些文件体积大，且可由数据流水线、训练脚本、回测脚本和 live 脚本重新生成。
