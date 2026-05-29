```markdown
# Run Baselines

> Status: 2026-05-29  
> Scope: 创业板动态成分股日频预测，复现旧版 Benchmark，并在新版 Clean Dataset 上启动可对照评测。  
> Important: 本文档中的正式 Benchmark 数值全部来自旧版数据集 `advanced_sequence_fixed`。新版 Clean Dataset 目前只完成 dry-run / smoke check，尚未冻结正式结果。

## 1. Introduction

本项目的 Baseline 目标是建立一个可复现、可对照、可继续演进的量化预测基线：

```text
股票池: 创业板指 399006.SZ 历史动态成分股
频率: 日频
标签: 未来 5 个交易日相对创业板指数的超额收益
目标列: label_rel_return
模型输出: pred_score
下游评估: daily IC / RankIC, Top-K proxy, Top-K backtest, long-short backtest
```

新旧数据集定位如下：

| Dataset line | Feature set | Tensor shape | Role |
| --- | --- | --- | --- |
| Legacy baseline | `advanced_sequence_fixed` | `[B, 20, 62]` | 复现当前所有正式 GRU Benchmark 结果 |
| Legacy extended | `advanced_sequence_fixed` | `[B, 60, 62]` | 长窗口序列模型候选，尚非主线 |
| Clean alpha-only | `advanced_sequence_clean_v1` | `[B, 20, 13]` | 去风格、去冗余后的纯 alpha 输入 |
| Clean alpha + residual style | `advanced_sequence_clean_v1` | `[B, 20, 18]` | 13 个 alpha 特征 + 5 个残差化风格特征 |

## 2. Environment & Dry Run

推荐环境：

```bash
conda activate dl_env
```

也可以使用：

```bash
conda run -n dl_env python <script>
```

在正式训练前，先用 `--dry-run` 验证配置、数据路径、样本切分、特征维度和 DataLoader 是否正常。

Legacy GRU baseline dry-run:

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml \
  --device cpu \
  --dry-run
```

Clean alpha-only dry-run:

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_clean_alpha_only_strictmask_leaky0005.yaml \
  --device cpu \
  --dry-run
```

Clean alpha + residual style dry-run:

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_clean_alpha_resid_style_strictmask_leaky0005.yaml \
  --device cpu \
  --dry-run
```

短训练 smoke test：

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_clean_alpha_only_strictmask_leaky0005.yaml \
  --device cpu \
  --max-epochs 1 \
  --max-train-batches 5 \
  --max-val-batches 5 \
  --max-test-batches 5
```

## 3. Dataset Profiles

### 3.1 固定时间切分

配置文件：

```text
configs/splits.yaml
```

当前固定 split：

| Split | Date range |
| --- | --- |
| train | `20160104` - `20221231` |
| validation | `20230101` - `20241231` |
| test | `20250101` - `20260525` |

规则：

- 禁止随机切分。
- 禁止使用 test 调参。
- 所有正式对比必须使用同一 split。
- `trade_date` 和 `ts_code` 只用于 metadata / prediction export，不输入模型。

### 3.2 Legacy Dataset

Legacy 数据集是当前正式 Benchmark 的唯一来源。

Manifest:

```text
data/mart/datasets/dataset_manifest_v20260526.json
```

LightGBM flat dataset:

| Item | Value |
| --- | --- |
| Path | `data/mart/datasets/dataset_lgbm_baseline_lightgbm_fixed_chinext_2016_2026_v1_v20260526.parquet` |
| Feature set | `baseline_lightgbm_fixed` |
| Rows | `241643` |
| Features | `40` |
| Train | `161578` |
| Validation | `47628` |
| Test | `32437` |

Sequence L20 dataset:

| Item | Value |
| --- | --- |
| Path | `data/mart/datasets/dataset_sequence_l20_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz` |
| Feature set | `advanced_sequence_fixed` |
| Shape | `(205621, 20, 62)` |
| Train | `133403` |
| Validation | `43803` |
| Test | `28415` |

Sequence L60 dataset:

| Item | Value |
| --- | --- |
| Path | `data/mart/datasets/dataset_sequence_l60_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz` |
| Feature set | `advanced_sequence_fixed` |
| Shape | `(192014, 60, 62)` |
| Train | `121913` |
| Validation | `42464` |
| Test | `27637` |

Legacy 特征池特点：

- 全部模型输入特征为 `lag1_`，遵守 T+1 执行约束。
- `advanced_sequence_fixed` 保留 62 个序列特征，覆盖收益、反转、趋势、成交、换手、资金流、规模、估值、行业相对状态、涨跌停状态。
- 旧版 GRU 结果存在风格暴露、换手暴露、输出头饱和、Top-K narrow head 不稳定等已知问题。
- 正式历史结果仍然必须绑定到该数据集，不能迁移标注到新版 Clean Dataset。

### 3.3 Clean Dataset

Clean Dataset 是 2026-05-29 后的新评测入口，用于检验残差 alpha、严格可交易样本过滤和特征治理是否改善模型质量。

Clean alpha-only:

| Item | Value |
| --- | --- |
| Path | `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026.npz` |
| Sidecar | `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026_sidecar.parquet` |
| Filter log | `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_only_chinext_2016_2026_filter_log.csv` |
| Feature set | `advanced_sequence_clean_v1` |
| Build mode | `alpha_only` |
| Shape | `(196138, 20, 13)` |
| Train | `124527` |
| Validation | `42759` |
| Test | `28852` |

Clean alpha + residual style:

| Item | Value |
| --- | --- |
| Path | `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_resid_style_chinext_2016_2026.npz` |
| Sidecar | `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_resid_style_chinext_2016_2026_sidecar.parquet` |
| Filter log | `data/mart/datasets/dataset_seq_l20_adv_clean_v1_alpha_resid_style_chinext_2016_2026_filter_log.csv` |
| Feature set | `advanced_sequence_clean_v1` |
| Build mode | `alpha_plus_residual_style` |
| Shape | `(196138, 20, 18)` |
| Train | `124527` |
| Validation | `42759` |
| Test | `28852` |

Clean Dataset 核心变化：

- 将模型输入拆成 alpha tensor 与 sidecar controls。
- 默认不把风险控制、流动性控制、涨跌停控制直接喂入 GRU。
- 使用 strict tradable mask 过滤样本。
- 输出 filter log，便于审计每类样本剔除原因。
- 新版数据集目前没有正式 Benchmark 数值，不能与旧版结果混写。

Clean alpha-only 的 13 个模型特征：

```text
lag1_net_mf_strength_20d_mean
lag1_net_mf_strength_60d_mean
lag1_close_position
lag1_excess_ret_10d_mean
lag1_excess_ret_1d
lag1_excess_ret_5d_mean
lag1_industry_neutral_ret_1d
lag1_ret_1d
lag1_ret_20d
lag1_ret_5d_mean
lag1_bollinger_z_20d
lag1_ma_ratio_20_60
lag1_macd_hist
```

Clean alpha + residual style 额外加入 5 个残差化风格特征：

```text
lag1_turnover_cost_proxy__resid_style
lag1_turnover_20d_std__resid_style
lag1_turnover_60d_std__resid_style
lag1_amount_rank_pct__resid_style
lag1_amount_log__resid_style
```

Strict tradable mask 当前过滤结果：

| Item | Value |
| --- | ---: |
| Input panel rows | `241643` |
| Kept rows | `211978` |
| Dropped rows | `29665` |
| Drop rate | `12.28%` |
| Locked limit rows | `2023` |
| Low amount rows | `20253` |
| Microcap rows | `12827` |

## 4. Running Baselines

### 4.1 Canonical Legacy GRU Baseline

当前主线 GRU 配置：

```text
configs/sequence_gru_l20_mse_ic_frozen_head.yaml
```

运行：

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml \
  --device cuda
```

核心设置：

| Item | Value |
| --- | --- |
| Dataset | `dataset_sequence_l20_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz` |
| Input shape | `[B, 20, 62]` |
| Backbone | 2-layer GRU |
| Pooling | `last_hidden` |
| Head | LeakyReLU |
| `head_negative_slope` | `0.005` |
| Loss | `mse_ic` |
| `ic_loss_alpha` | `0.2` |
| Batch mode | `date` |
| Early stop metric | `rank_ic_mean` |

### 4.2 Historical Legacy Runs

Historical runs are kept for traceback only. Do not use them as the active baseline unless doing retrospective comparison.

ReLU reference:

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_baseline.yaml \
  --device cuda
```

GELU head ablation:

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_mse_ic_gelu_head.yaml \
  --device cuda
```

Parent LeakyReLU ablation:

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_mse_ic_leaky_head_001.yaml \
  --device cuda
```

Stable variant:

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_baseline_stable.yaml \
  --device cuda
```

### 4.3 Clean Alpha-Only GRU

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_clean_alpha_only_strictmask_leaky0005.yaml \
  --device cuda
```

Use this run to answer:

```text
在去除风格/流动性/可交易性控制列后，纯 alpha tensor 是否仍有稳定 RankIC？
```

### 4.4 Clean Alpha + Residual Style GRU

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_clean_alpha_resid_style_strictmask_leaky0005.yaml \
  --device cuda
```

Use this run to answer:

```text
残差化后的 turnover / amount style 信息，是否能在不恢复原始风格暴露的情况下提升排序能力？
```

### 4.5 Rebuild Clean Datasets

Validate clean feature set:

```bash
python scripts/validate_clean_feature_set.py
```

Build alpha-only dataset:

```bash
python scripts/build_clean_model_datasets.py \
  --data-version v20260526 \
  --build-mode alpha_only \
  --lookbacks 20
```

Build alpha + residual style dataset:

```bash
python scripts/build_clean_model_datasets.py \
  --data-version v20260526 \
  --build-mode alpha_plus_residual_style \
  --lookbacks 20
```

## 5. Legacy Benchmark Results

The following results are legacy-only. They were produced on:

```text
data/mart/datasets/dataset_sequence_l20_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz
```

Do not report these numbers as Clean Dataset results.

### 5.1 Historical ReLU Reference Result

Run:

```text
outputs/runs/e02_gru_l20_mse_ic_02/
```

| Metric | Value |
| --- | ---: |
| Best epoch | `13` |
| Best validation RankIC mean | `0.0299640` |
| Best validation IC mean | `0.0208654` |
| Validation RankIC IR | `0.2013793` |
| Recomputed test RankIC mean | `0.0487976` |
| Recomputed test IC mean | `0.0259337` |
| Recomputed test RankIC IR | `0.3148049` |
| Recomputed test ICIR | `0.1630962` |
| Stop reason | `metric_early_stop:rank_ic_mean` |
| Best daily count | `484 / 484` |
| Best constant prediction days | `0 / 484` |

### 5.2 Top-K Proxy

Input:

```text
outputs/runs/e02_gru_l20_mse_ic_02/predictions.parquet
```

Script:

```bash
python scripts/evaluate_topk.py
```

| Split | K | Top mean | Bottom mean | Top-Bottom spread | Spread IR | Spread positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 10 | `-0.000116` | `-0.002515` | `0.002399` | `0.0719` | `0.5599` |
| validation | 20 | `-0.000314` | `-0.002098` | `0.001784` | `0.0752` | `0.5558` |
| validation | 30 | `0.000195` | `-0.002680` | `0.002875` | `0.1620` | `0.5909` |
| test | 10 | `-0.005884` | `-0.004612` | `-0.001272` | `-0.0374` | `0.5106` |
| test | 20 | `-0.004043` | `-0.005081` | `0.001039` | `0.0406` | `0.5714` |
| test | 30 | `-0.002721` | `-0.004404` | `0.001684` | `0.0804` | `0.5441` |

Interpretation:

- Validation split 在 K=10/20/30 都有正 long-short spread。
- Test split 在 K=20/30 保持正 spread，但强度有限。
- K=10 不稳定，不能作为主判断口径。
- K=30 是当前更稳妥的诊断宽度。

### 5.3 Top-K Backtest

Script:

```bash
python scripts/backtest_topk.py
```

Method:

```text
非重叠 5 日持有期
rebalance stride = 5 signal dates
equal-weight Top-K
one-way cost = 10 bps
```

| Split | K | Periods | Top-K annualized | Top-K cumulative | Max drawdown | Excess vs benchmark annualized | Excess vs universe annualized | Long-short annualized | Avg turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 10 | `97` | `-0.0910` | `-0.1678` | `-0.4999` | `-0.0184` | `0.0512` | `0.0195` | `0.7732` |
| validation | 20 | `97` | `-0.1498` | `-0.2683` | `-0.5120` | `-0.0810` | `-0.0157` | `0.0019` | `0.7577` |
| validation | 30 | `97` | `-0.1268` | `-0.2296` | `-0.4603` | `-0.0555` | `0.0117` | `0.0818` | `0.7409` |
| test | 10 | `66` | `0.2537` | `0.3446` | `-0.2455` | `-0.2475` | `-0.1420` | `-0.1728` | `0.8394` |
| test | 20 | `66` | `0.3344` | `0.4590` | `-0.2122` | `-0.1957` | `-0.0851` | `0.0930` | `0.7818` |
| test | 30 | `66` | `0.4591` | `0.6401` | `-0.2080` | `-0.1177` | `0.0020` | `0.1501` | `0.7152` |

Interpretation:

- Test Top30 是旧版结果中最强的 Top-K 设置。
- Test Top30 年化约 `45.9%`，但相对 benchmark 仍落后。
- 策略信号有模型级排序价值，但还不是最终可交易策略。

### 5.4 Long-Short Backtest

| Split | K | Cost bps | Periods | Long-short annualized | Long-short cumulative | Max drawdown | Win rate | Sharpe-like | Avg LS turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 10 | `10` | `97` | `-0.0734` | `-0.1365` | `-0.3371` | `0.5155` | `-0.3259` | `1.8928` |
| validation | 20 | `10` | `97` | `-0.0829` | `-0.1535` | `-0.2882` | `0.5052` | `-0.4841` | `1.7526` |
| validation | 30 | `10` | `97` | `-0.0021` | `-0.0041` | `-0.1772` | `0.5258` | `-0.0171` | `1.6027` |
| test | 10 | `10` | `66` | `-0.2518` | `-0.3160` | `-0.3357` | `0.4545` | `-0.8865` | `1.9788` |
| test | 20 | `10` | `66` | `0.0008` | `0.0011` | `-0.1568` | `0.5000` | `0.0040` | `1.7500` |
| test | 30 | `10` | `66` | `0.0652` | `0.0862` | `-0.1354` | `0.5000` | `0.4079` | `1.5263` |

Interpretation:

- K=30 在 test 上仍有轻微正 long-short 表现。
- 换手率非常高，约 `1.5` 到 `2.0` per rebalance。
- 下一阶段必须优先处理 turnover control、strict tradability 和 head resolution。

## 6. Arguments & Configurations

### 6.1 `scripts/train_sequence.py`

| Argument | Required | Description |
| --- | --- | --- |
| `--config` | yes | YAML config path |
| `--device` | no | Override device: `auto`, `cpu`, `cuda` |
| `--output-dir` | no | Override `run.output_dir` |
| `--dry-run` | no | Build dataset/model/loss/optimizer and print summary without training |
| `--max-epochs` | no | Override training epochs for smoke tests |
| `--max-train-batches` | no | Limit train batches |
| `--max-val-batches` | no | Limit validation batches |
| `--max-test-batches` | no | Limit test prediction batches |

### 6.2 Main Configs

| Purpose | Config |
| --- | --- |
| Current legacy mainline | `configs/sequence_gru_l20_mse_ic_frozen_head.yaml` |
| Historical ReLU baseline | `configs/sequence_gru_baseline.yaml` |
| Historical stable variant | `configs/sequence_gru_baseline_stable.yaml` |
| Historical GELU ablation | `configs/sequence_gru_l20_mse_ic_gelu_head.yaml` |
| Historical LeakyReLU ablation | `configs/sequence_gru_l20_mse_ic_leaky_head_001.yaml` |
| Clean alpha-only | `configs/sequence_gru_l20_clean_alpha_only_strictmask_leaky0005.yaml` |
| Clean alpha + residual style | `configs/sequence_gru_l20_clean_alpha_resid_style_strictmask_leaky0005.yaml` |

### 6.3 Canonical GRU Hyperparameters

| Field | Value |
| --- | --- |
| `d_model` | `64` |
| `rnn_hidden_dim` | `128` |
| `rnn_num_layers` | `2` |
| `rnn_dropout` | `0.2` |
| `pooling` | `last_hidden` |
| `head_hidden_dim` | `64` |
| `head_dropout` | `0.3` |
| `head_activation` | `leaky_relu` |
| `head_negative_slope` | `0.005` |
| `optimizer` | `adamw` |
| `learning_rate` | `0.0003` |
| `weight_decay` | `0.00001` |
| `loss_fn` | `mse_ic` |
| `ic_loss_alpha` | `0.2` |
| `batch_mode` | `date` |
| `scheduler` | `cosine` |
| `early_stop_metric` | `rank_ic_mean` |
| `early_stop_patience` | `10` |

## 7. Prediction Export Contract

All baseline runs should export:

```text
outputs/runs/{run_name}/predictions.parquet
outputs/runs/{run_name}/metrics.json
outputs/runs/{run_name}/config.yaml
outputs/runs/{run_name}/model.pt
```

`predictions.parquet` must contain:

```text
trade_date
ts_code
pred_score
label_rel_return
split
model_name
```

Required rules:

- `trade_date` and `ts_code` must be preserved.
- `pred_score` is the only ranking column consumed by Top-K evaluation.
- `label_rel_return` is the future 5-day relative return label.
- `split` must remain one of `train`, `validation`, `test`.
- Do not export predictions without split metadata.

## 8. Evaluation Protocol

Minimum evaluation stack:

```text
1. validation/test MSE
2. daily IC
3. daily RankIC
4. ICIR / RankIC IR
5. Top-K proxy, K = 10 / 20 / 30
6. Top-K 5-day non-overlap backtest
7. long-short Top-K vs Bottom-K backtest
8. turnover and transaction-cost sensitivity
```

For Clean Dataset runs, compare at least:

| Run | Dataset | Purpose |
| --- | --- | --- |
| Legacy frozen GRU | `advanced_sequence_fixed`, 62 features | Existing benchmark anchor |
| Clean alpha-only GRU | `advanced_sequence_clean_v1`, 13 features | Test pure alpha survival |
| Clean alpha + residual style GRU | `advanced_sequence_clean_v1`, 18 features | Test residual style value |

Reporting rule:

```text
旧版结果和新版结果必须分表展示。
不得把旧版 benchmark 数字标注为 clean-v1 结果。
```

## 9. Common Pitfalls

- `label_rel_return` 是未来 5 日相对创业板指数收益，不是单日收益。
- 所有输入特征已经是 `lag1_`，不要再次整体 shift。
- 序列 NPZ 已经按单股票历史生成窗口，不要跨股票拼接。
- Clean Dataset 的 sidecar controls 不属于默认 GRU input tensor。
- Strict tradable mask 是样本过滤和执行可行性控制，不是模型输入特征。
- Test 只能用于最终评估，不能用于调参、特征选择或 early stopping。
- 新版 clean-v1 当前只有 dry-run / smoke evidence，正式 benchmark 需要重新训练后单独登记。

## 10. Recommended Next Runs

First, reproduce legacy anchor:

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_mse_ic_frozen_head.yaml \
  --device cuda
```

Then train clean datasets:

```bash
python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_clean_alpha_only_strictmask_leaky0005.yaml \
  --device cuda

python scripts/train_sequence.py \
  --config configs/sequence_gru_l20_clean_alpha_resid_style_strictmask_leaky0005.yaml \
  --device cuda
```

After both clean runs finish:

```text
1. Recompute IC / RankIC on validation and test.
2. Evaluate Top-K proxy at K=10/20/30.
3. Run 5-day non-overlap Top-K backtest with 10 bps and 20 bps one-way cost.
4. Run long-short Top-K vs Bottom-K.
5. Compare against legacy frozen GRU in a separate clean-vs-legacy table.
6. Promote a new baseline only if clean-v1 improves robustness without hiding turnover or tradability costs.
```
```