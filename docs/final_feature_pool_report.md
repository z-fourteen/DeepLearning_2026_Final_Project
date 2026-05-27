# 模型构建交接文档：创业板全历史数据集与固定特征池

本文档面向下一位合作者：模型构建者。目标是让模型同学无需回溯 Agent 1-4 的全部工程细节，即可直接读取数据、理解字段、按固定时间切分训练 LightGBM / GRU / Transformer，并输出统一预测文件供 Top-K 回测使用。

## 1. 任务定义

本项目当前任务是：

```text
股票池：创业板指 399006.SZ 历史动态成分股
频率：日频
预测目标：未来 5 个交易日相对创业板指数的超额收益
模型输出：每日每只股票的 pred_score
投资动作：按 pred_score 做横截面排序，进入 Top-K 回测
```

核心标签列：

```text
label_rel_return
```

该标签已经在 mart 数据集中生成。模型训练阶段不需要重新构造标签。

## 2. 环境与入口

推荐 Python 环境：

```powershell
conda activate dl_env
```

或使用：

```powershell
conda run -n dl_env python <script>
```

模型数据集构建脚本：

```text
scripts/build_model_datasets.py
```

如需重新生成本交接数据：

```powershell
conda run -n dl_env python scripts/build_model_datasets.py --data-version v20260526 --split-name chinext_2016_2026_v1 --lookbacks 20 60
```

## 3. 固定时间切分

配置文件：

```text
configs/splits.yaml
```

当前固定 split：

```text
split_name: chinext_2016_2026_v1
train:      20160104 - 20221231
validation: 20230101 - 20241231
test:       20250101 - 20260525
```

重要约束：

- 禁止随机切分。
- 禁止用 test 调参。
- 特征池已经按训练期验证结果固化，test 只能用于最终评估。

## 4. 模型数据集清单

Manifest：

```text
data/mart/datasets/dataset_manifest_v20260526.json
```

### 4.1 LightGBM 平铺数据集

路径：

```text
data/mart/datasets/dataset_lgbm_baseline_lightgbm_fixed_chinext_2016_2026_v1_v20260526.parquet
```

规模：

```text
shape: (241643, 44)
features: 40
train: 161578
validation: 47628
test: 32437
```

字段结构：

```text
trade_date
ts_code
label_rel_return
40 个 lag1_ 特征
split
```

推荐用于：

```text
LightGBM baseline
Ridge / Linear baseline
RandomForest / XGBoost 对照
```

读取示例：

```python
import pandas as pd

path = "data/mart/datasets/dataset_lgbm_baseline_lightgbm_fixed_chinext_2016_2026_v1_v20260526.parquet"
df = pd.read_parquet(path)

label_col = "label_rel_return"
feature_cols = [c for c in df.columns if c.startswith("lag1_")]

train = df[df["split"] == "train"]
valid = df[df["split"] == "validation"]
test = df[df["split"] == "test"]

X_train, y_train = train[feature_cols], train[label_col]
X_valid, y_valid = valid[feature_cols], valid[label_col]
X_test, y_test = test[feature_cols], test[label_col]
```

### 4.2 GRU / Transformer 序列数据集 lookback=20

路径：

```text
data/mart/datasets/dataset_sequence_l20_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz
```

规模：

```text
X shape: (205621, 20, 62)
y shape: (205621,)
train: 133403
validation: 43803
test: 28415
```

### 4.3 GRU / Transformer 序列数据集 lookback=60

路径：

```text
data/mart/datasets/dataset_sequence_l60_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz
```

规模：

```text
X shape: (192014, 60, 62)
y shape: (192014,)
train: 121913
validation: 42464
test: 27637
```

NPZ 字段：

```text
X              float32, [sample, lookback, feature]
y              float32, [sample]
trade_date     str, sample 对应信号日期
ts_code        str, sample 对应股票代码
split          str, train / validation / test
feature_names  str, 62 个序列特征名
```

读取示例：

```python
import numpy as np

path = "data/mart/datasets/dataset_sequence_l20_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz"
data = np.load(path, allow_pickle=True)

X = data["X"]
y = data["y"]
split = data["split"]
trade_date = data["trade_date"]
ts_code = data["ts_code"]
feature_names = data["feature_names"].tolist()

train_mask = split == "train"
valid_mask = split == "validation"
test_mask = split == "test"

X_train, y_train = X[train_mask], y[train_mask]
X_valid, y_valid = X[valid_mask], y[valid_mask]
X_test, y_test = X[test_mask], y[test_mask]
```

## 5. 固定特征集

配置文件：

```text
configs/features.yaml
```

### 5.1 LightGBM 特征集

特征集名称：

```text
baseline_lightgbm_fixed
```

规模：

```text
40 features
```

定位：

```text
低泄露风险、低冗余、适合横截面排序的 baseline 特征池。
```

主要覆盖：

- 换手与流动性：`turnover_*`, `amount_*`, `vol_log`
- 波动率：`ret_*_std`, `amplitude`, `max_drawdown_20d`
- 动量/反转：`ret_*_mean`, `excess_ret_*`
- 资金流：`net_mf_*`
- 行业相对：`industry_*_rank`, `industry_neutral_ret_*`
- 规模与估值：`log_circ_mv`, `pb_winsor`
- 创业板状态：`listed_trading_days`, `close_position`

### 5.2 序列模型特征集

特征集名称：

```text
advanced_sequence_fixed
```

规模：

```text
62 features
```

定位：

```text
保留更多连续状态、趋势路径、技术指标、资金流路径和创业板涨跌停边界状态，供 GRU / Transformer 学习时间依赖。
```

注意：

- 序列数据已经剔除窗口内含 NaN 的样本。
- 不需要模型代码再手工滚动拼窗口。
- `X[:, -1, :]` 对应当前信号日可用的 lag1 特征状态。

## 6. 防泄露约定

请严格遵守：

1. 不要重新随机切分数据。
2. 不要用 test 选择特征、调参或 early stopping。
3. 不要把 `trade_date`、`ts_code` 当作普通数值特征输入模型。
4. 所有模型输出必须保留 `trade_date` 和 `ts_code`，否则无法接入回测。
5. 当前所有特征均为 `lag1_`，表示已经按 T+1 执行逻辑滞后处理，不要再额外整体 shift 一次。
6. 序列模型不要跨股票拼接窗口；现有 NPZ 已经按单股票历史生成窗口。

## 7. 模型输出格式

请所有模型统一输出：

```text
trade_date
ts_code
pred_score
label_rel_return
split
model_name
```

建议路径：

```text
outputs/runs/{run_id}/predictions.parquet
outputs/runs/{run_id}/metrics.json
outputs/runs/{run_id}/config.yaml
```

示例：

```python
pred = test[["trade_date", "ts_code", "label_rel_return", "split"]].copy()
pred["pred_score"] = model.predict(X_test)
pred["model_name"] = "lightgbm_baseline"
pred.to_parquet("outputs/runs/20260527_lgbm_baseline/predictions.parquet", index=False)
```

## 8. 推荐建模顺序

建议模型构建者按以下顺序推进：

1. LightGBM regression baseline
2. Ridge / Linear baseline
3. GRU lookback=20
4. GRU lookback=60
5. Transformer Encoder lookback=20
6. 统一 predictions 后接入 Top-K 回测

LightGBM 建议先做，不要一开始直接做 Transformer。原因是 LightGBM 更容易校验数据、标签、split 和回测接口是否正常。

## 9. LightGBM 最小训练配置

可从 `configs/experiment.yaml` 读取，也可以先用以下参数：

```text
objective: regression
metric: l2
learning_rate: 0.03
num_leaves: 31
feature_fraction: 0.8
bagging_fraction: 0.8
bagging_freq: 1
min_data_in_leaf: 80
num_boost_round: 2000
early_stopping_rounds: 100
```

评估指标至少输出：

```text
MSE
MAE
daily IC
daily RankIC
ICIR
validation/test 分开统计
```

## 10. Top-K 回测接口预期

回测器预计读取 `predictions.parquet`，按每日 `pred_score` 排序。

建议规则：

```text
信号日：t 日收盘后
买入日：t+1
持有期：5 个交易日
Top-K：10 / 20 / 30
权重：等权
交易成本：单边 0.1% 与 0.2% 两档
benchmark：399006.SZ
过滤：ST、停牌、不可交易、涨跌停不可买入
```

当前阶段尚未实现正式回测器，模型同学只需要保证预测文件格式正确。

## 11. 已完成的上游验证

训练期候选池验证命令：

```powershell
conda run -n dl_env python scripts/run_factor_validation.py --data-version v20260526 --stage all --resume --train-end-date 20221231 --eval-start-date 20230101 --skip-neutralized --skip-extended-quantile
```

训练期验证输出：

```text
outputs/factor_validation/v20260526/all_lag1/label_rel_return_q5_cs30_corr0p85_train_20221231_eval_20230101_quantile_on_ext_off_neutral_off/
```

全历史诊断输出：

```text
outputs/factor_validation/v20260526/all_lag1/label_rel_return_q5_cs30_corr0p85_train_all_eval_all_quantile_on_ext_on_neutral_on/
```

全历史诊断只用于理解因子，不用于最终测试集调参。

## 12. 常见坑

- `label_rel_return` 是未来 5 日相对收益，不是明日收益。
- 平铺数据集中的 `split` 是字符串列，不要输入模型。
- 序列 NPZ 中 `trade_date`、`ts_code` 只用于回写预测，不输入模型。
- 序列样本数少于平铺样本是正常现象，因为 lookback 窗口和 NaN 过滤会丢样本。
- 大文件位于 `data/mart/datasets/`，通常不提交 Git。

## 13. 下一步交付物

模型构建者下一步应交付：

```text
outputs/runs/{run_id}/config.yaml
outputs/runs/{run_id}/model.*
outputs/runs/{run_id}/predictions.parquet
outputs/runs/{run_id}/metrics.json
```

完成 LightGBM 后，再进入真实 Top-K 回测与 GRU / Transformer 对比。
