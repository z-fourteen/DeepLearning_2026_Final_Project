# 模型架构设计文档 v2.0

本文档用于第二阶段模型构建。它以当前已落盘的 mart 数据集为唯一数据接口基准，优先推进 **GRU Baseline** 与 **Transformer Encoder 进阶模型**；LSTM 暂不进入当前主线，仅保留为后置任务。

## 1. 当前结论

当前任务是对创业板指 `399006.SZ` 历史动态成分股做日频横截面选股：

```text
输入：单只股票过去 lookback 个交易日的序列特征
输出：该股票信号日对应的未来 5 个交易日相对创业板指数超额收益预测分数
标签：label_rel_return
用途：每日按 pred_score 横截面排序，接入 Top-K 回测
```

第二阶段优先级：

| 优先级 | 模型 | 当前定位 |
| --- | --- | --- |
| P0 | GRU Baseline | 第一批实现，验证深度序列建模闭环 |
| P0 | Transformer Encoder | 第二批实现，作为进阶对比模型 |
| P2 | LSTM | 后置任务/未来规划，不占用当前核心实现时间 |

所有模型必须共享同一数据版本、同一 split、同一标签、同一评估与预测输出格式。当前固定数据版本为 `v20260526`，固定 split 为 `chinext_2016_2026_v1`。

## 2. 数据接口

### 2.1 固定数据源

模型训练直接读取已构建好的序列 NPZ，不再在训练代码里重新滚动拼窗口。

| lookback | 路径 | X shape | y shape | 特征数 | 样本数 |
| --- | --- | --- | --- | --- | --- |
| 20 | `data/mart/datasets/dataset_sequence_l20_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz` | `(205621, 20, 62)` | `(205621,)` | 62 | 205621 |
| 60 | `data/mart/datasets/dataset_sequence_l60_advanced_sequence_fixed_chinext_2016_2026_v1_v20260526.npz` | `(192014, 60, 62)` | `(192014,)` | 62 | 192014 |

NPZ 内部字段：

| key | dtype | shape | 用途 |
| --- | --- | --- | --- |
| `X` | `float32` | `[N, T, F]` | 模型输入特征序列 |
| `y` | `float32` | `[N]` | `label_rel_return` |
| `trade_date` | str | `[N]` | 信号日期，回写预测与按日评估使用 |
| `ts_code` | str | `[N]` | 股票代码，回写预测与回测使用 |
| `split` | str | `[N]` | `train` / `validation` / `test` |
| `feature_names` | str | `[F]` | 62 个输入特征名称 |

split 计数：

| lookback | train | validation | test |
| --- | ---: | ---: | ---: |
| 20 | 133403 | 43803 | 28415 |
| 60 | 121913 | 42464 | 27637 |

固定时间切分来自 `configs/splits.yaml`：

```text
train:      20160104 - 20221231
validation: 20230101 - 20241231
test:       20250101 - 20260525
```

### 2.2 特征拼接方式

序列数据集已经由 `pipelines/mart/dataset.py` 按单只股票历史构造：

```text
X[i] = stock s 在 [t-lookback+1, ..., t] 的 62 维 lag1 特征
y[i] = stock s 在信号日 t 的 label_rel_return
trade_date[i] = t
ts_code[i] = s
```

因此模型端只做 tensor 读取，不做以下操作：

- 不重新按 `ts_code` 滚动拼窗口。
- 不跨股票拼接序列。
- 不额外整体 `shift(1)`。
- 不把 `trade_date`、`ts_code`、`split` 输入模型。

所有 62 个特征来自 `advanced_sequence_fixed`，均为 `lag1_` 前缀，表示已经按 T+1 执行逻辑做过滞后处理。`X[:, -1, :]` 是信号日 t 可用的最新特征状态，不是未来信息。

### 2.3 DataLoader 输出格式

建议实现一个直接读取 NPZ 的 Dataset：

```python
class SequenceNPZDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, split: str):
        data = np.load(npz_path, allow_pickle=True)
        mask = data["split"] == split
        self.X = data["X"][mask].astype("float32")          # [N_split, T, 62]
        self.y = data["y"][mask].astype("float32")          # [N_split]
        self.trade_date = data["trade_date"][mask]          # [N_split]
        self.ts_code = data["ts_code"][mask]                # [N_split]
        self.feature_names = data["feature_names"].tolist()

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.X[idx]),             # [T, 62], float32
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "trade_date": str(self.trade_date[idx]),
            "ts_code": str(self.ts_code[idx]),
        }
```

默认 PyTorch `DataLoader` collate 后，batch 格式应为：

```text
batch["x"]          FloatTensor [B, T, 62]
batch["y"]          FloatTensor [B]
batch["trade_date"] list[str] or tuple[str], length B
batch["ts_code"]    list[str] or tuple[str], length B
```

模型 forward 接口只接收：

```python
pred = model(batch["x"])  # [B] 或 [B, 1]
```

训练损失计算前统一 reshape：

```python
pred = pred.view(-1)
target = batch["y"].view(-1)
loss = loss_fn(pred, target)
```

验证和测试阶段必须保留 `trade_date` 与 `ts_code`，用于每日 IC/RankIC 与预测文件回写。

## 3. 统一模型接口

所有深度模型统一遵守：

```python
class BaseStockModel(nn.Module):
    def __init__(self, num_features: int = 62, config: dict | None = None):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor [B, T, 62]
        Returns:
            pred_score: FloatTensor [B]
        """
        raise NotImplementedError
```

推荐模型输出 `[B]`，如果内部 head 输出 `[B, 1]`，在 `forward` 末尾使用 `squeeze(-1)`。这样训练、评估、预测落盘都能使用同一套代码。

公共组件：

```text
FeatureProjection:
  Linear(62, d_model)
  LayerNorm(d_model)
  Dropout(input_dropout)

PredictionHead:
  Linear(context_dim, head_hidden_dim)
  activation
  Dropout(head_dropout)
  Linear(head_hidden_dim, 1)
  squeeze(-1)
```

## 4. GRU Baseline

### 4.1 架构

GRU 是当前主 baseline，目标是先跑通稳定、低复杂度、可解释的深度序列模型。

```text
Input x: [B, T, 62]
  |
  v
FeatureProjection
  Linear(62, d_model=64)
  LayerNorm
  Dropout(0.1)
  -> [B, T, 64]
  |
  v
GRU Encoder
  input_size=64
  hidden_size=128
  num_layers=2
  batch_first=True
  dropout=0.2
  bidirectional=False
  -> all_hidden [B, T, 128]
  -> h_n [2, B, 128]
  |
  v
Pooling
  default: h_n[-1] -> [B, 128]
  optional: attention_pool(all_hidden) -> [B, 128]
  |
  v
PredictionHead
  Linear(128, 64) + ReLU + Dropout(0.3)
  Linear(64, 1)
  squeeze(-1)
  |
  v
pred_score: [B]
```

### 4.2 推荐配置

```yaml
model:
  name: gru_baseline
  num_features: 62
  lookback: 20
  d_model: 64
  input_dropout: 0.1
  rnn_hidden_dim: 128
  rnn_num_layers: 2
  rnn_dropout: 0.2
  bidirectional: false
  pooling: last_hidden
  head_hidden_dim: 64
  head_dropout: 0.3

training:
  batch_size: 256
  max_epochs: 80
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  loss_fn: huber
  max_grad_norm: 1.0
  scheduler: cosine
  early_stop_metric: validation_rank_ic
  early_stop_patience: 10
```

首轮必须先跑 `lookback=20`。`lookback=60` 是第二个 GRU 实验，用来比较较长历史是否带来稳定提升。

### 4.3 Forward 伪代码

```python
def forward(self, x):
    # x: [B, T, 62]
    z = self.input_proj(x)              # [B, T, 64]
    all_hidden, h_n = self.gru(z)       # [B, T, 128], [L, B, 128]
    context = h_n[-1]                   # [B, 128]
    pred = self.head(context)           # [B, 1]
    return pred.squeeze(-1)             # [B]
```

### 4.4 Attention Pooling 作为 GRU 消融

该模块不作为第一版 GRU 的必要功能，建议在 GRU baseline 跑通后作为消融实验：

```text
all_hidden: [B, T, 128]
score = Linear(128, 1)(all_hidden)      -> [B, T, 1]
weight = softmax(score, dim=1)          -> [B, T, 1]
context = sum(weight * all_hidden, dim=1) -> [B, 128]
```

可配置项：

```yaml
model:
  pooling: attention
```

注意：当前 NPZ 没有 padding，正常情况下不需要 attention mask。

## 5. Transformer Encoder

### 5.1 架构

Transformer 是当前进阶模型，用于检验全局时间步注意力是否优于 GRU 的递归压缩。

```text
Input x: [B, T, 62]
  |
  v
FeatureProjection
  Linear(62, d_model=64)
  LayerNorm
  Dropout(0.1)
  -> [B, T, 64]
  |
  v
Position Encoding
  sinusoidal or learnable
  -> [B, T, 64]
  |
  v
Optional CLS Token
  if use_cls_token: [B, T, 64] -> [B, T+1, 64]
  |
  v
TransformerEncoder x 2
  MultiHeadSelfAttention(d_model=64, nhead=4, dropout=0.1)
  FeedForward(64 -> 128 -> 64)
  GELU
  LayerNorm + Residual
  -> [B, T(+1), 64]
  |
  v
Pooling
  default: cls token -> [B, 64]
  fallback: last time step -> [B, 64]
  |
  v
PredictionHead
  Linear(64, 64) + GELU + Dropout(0.3)
  Linear(64, 1)
  squeeze(-1)
  |
  v
pred_score: [B]
```

### 5.2 推荐配置

```yaml
model:
  name: transformer_encoder
  num_features: 62
  lookback: 20
  d_model: 64
  input_dropout: 0.1
  num_encoder_layers: 2
  num_heads: 4
  dim_feedforward: 128
  attn_dropout: 0.1
  ff_dropout: 0.1
  activation: gelu
  norm_first: true
  positional_encoding: sinusoidal
  use_cls_token: true
  head_hidden_dim: 64
  head_dropout: 0.3

training:
  batch_size: 256
  max_epochs: 80
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.001
  loss_fn: huber
  max_grad_norm: 1.0
  scheduler: warmup_cosine
  warmup_ratio: 0.1
  early_stop_metric: validation_rank_ic
  early_stop_patience: 8
```

首轮 Transformer 只跑 `lookback=20`。在验证集稳定后，再考虑 `lookback=60`，否则容易把时间花在显存和过拟合问题上。

### 5.3 Forward 伪代码

```python
def forward(self, x):
    # x: [B, T, 62]
    z = self.input_proj(x)                  # [B, T, 64]
    z = self.pos_encoder(z)                 # [B, T, 64]
    if self.use_cls_token:
        cls = self.cls_token.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1)      # [B, T+1, 64]
    encoded = self.encoder(z)               # [B, T(+1), 64]
    context = encoded[:, 0] if self.use_cls_token else encoded[:, -1]
    pred = self.head(context)
    return pred.squeeze(-1)
```

### 5.4 位置编码策略

默认使用 sinusoidal positional encoding，因为：

- 当前训练样本虽然不少，但金融噪声高，先减少可学习参数更稳。
- `lookback=20` 较短，固定位置编码已经足够表达顺序。

可学习位置编码仅作为后续消融：

```yaml
model:
  positional_encoding: learnable
```

## 6. LSTM Backlog

LSTM 从当前主线降级为后置任务。原因是时间紧迫时，LSTM 相比 GRU 的边际信息有限，但参数更多、训练更慢；当前更值得把资源投向 GRU 稳定 baseline 与 Transformer 对比。

保留设计如下：

```text
Input [B, T, 62]
  -> FeatureProjection [B, T, 64]
  -> LSTM(hidden_size=128, num_layers=2, dropout=0.2)
  -> h_n[-1] or attention_pool
  -> PredictionHead
  -> pred_score [B]
```

进入条件：

- GRU lookback=20 与 lookback=60 已完成。
- Transformer lookback=20 已完成。
- 仍有时间做门控机制对比。

## 7. 训练与评估

### 7.1 损失函数

首选 `HuberLoss`，备选 `MSELoss`：

```python
loss_fn = torch.nn.HuberLoss(delta=1.0)
```

金融标签有极端值，Huber 通常比 MSE 更稳。模型选择不看训练 loss，优先看验证集 RankIC，其次 IC。

### 7.2 日频指标

验证/测试时收集：

```text
trade_date, ts_code, pred_score, label_rel_return, split, model_name
```

按 `trade_date` 分组计算：

| 指标 | 计算方式 |
| --- | --- |
| IC | Pearson(pred_score, label_rel_return) |
| RankIC | Spearman(pred_score, label_rel_return) |
| ICIR | mean(daily_ic) / std(daily_ic) |
| RankICIR | mean(daily_rank_ic) / std(daily_rank_ic) |
| MSE/MAE | 辅助观察，不作为主选择指标 |

每日有效股票数太少时应跳过该日，建议阈值 `min_daily_count=20`。

### 7.3 预测输出格式

所有模型统一输出：

```text
outputs/runs/{run_id}/predictions.parquet
```

字段必须为：

```text
trade_date
ts_code
pred_score
label_rel_return
split
model_name
```

其中 `pred_score` 是模型原始连续输出，不做每日 rank 替换。回测器负责按日排序。

## 8. 实验矩阵

当前阶段按以下顺序推进：

| 编号 | 优先级 | 模型 | 数据 | 关键变量 | 目标 |
| --- | --- | --- | --- | --- | --- |
| E01 | P0 | GRU | lookback=20 | last_hidden | 主 baseline |
| E02 | P0 | GRU | lookback=60 | last_hidden | 检验更长历史 |
| E03 | P0 | Transformer | lookback=20 | CLS + sinusoidal PE | 进阶模型 |
| E04 | P1 | GRU | lookback=20 | attention pooling | 池化消融 |
| E05 | P1 | Transformer | lookback=20 | last-step pooling | 池化消融 |
| E06 | P1 | Transformer | lookback=20 | learnable PE | 位置编码消融 |
| B01 | P2 | LSTM | lookback=20 | last_hidden | Backlog |

每个 run 至少保存：

```text
outputs/runs/{run_id}/config.yaml
outputs/runs/{run_id}/model.pt
outputs/runs/{run_id}/predictions.parquet
outputs/runs/{run_id}/metrics.json
```

`metrics.json` 至少包含 train/validation/test 分开的 MSE、MAE、IC、RankIC、ICIR、RankICIR。

## 9. 实现验收清单

第一批代码完成后，必须通过以下检查：

- `SequenceNPZDataset` 读取 lookback=20 后，`batch["x"].shape == [B, 20, 62]`。
- `SequenceNPZDataset` 读取 lookback=60 后，`batch["x"].shape == [B, 60, 62]`。
- GRU 随机 batch forward 输出 `[B]`。
- Transformer 随机 batch forward 输出 `[B]`。
- 训练循环不会把 `trade_date`、`ts_code`、`split` 输入模型。
- 验证阶段可以生成包含 6 个必需字段的 predictions DataFrame。
- validation 指标用于 early stopping，test 只在最终评估时使用。

## 10. 防泄露约束

必须严格遵守：

1. 只使用 NPZ 中的 `X` 作为模型输入。
2. 不重新随机划分 train/validation/test。
3. 不用 test 调参、选特征、选择 checkpoint。
4. 不额外 shift `lag1_` 特征。
5. 不使用 `trade_date` 或 `ts_code` embedding；当前阶段不做股票 ID 记忆。
6. 不跨股票拼接序列窗口。
7. 所有模型输出保留原始 `trade_date` 和 `ts_code`，否则无法进入回测。

## 11. 建议目录与文件结构

第二阶段采用“按实现步骤逐步生成”的工程结构。原则是：只创建已经实现或即将实现的 Python 文件，不提前生成空壳文件；LSTM、回测器、DailyBatchSampler 等后置任务不占用当前主线目录。

推荐目标结构如下：

```text
src/
  __init__.py

  data/
    __init__.py
    sequence_npz_dataset.py        # 已实现：读取 NPZ，按 split 过滤

  models/
    __init__.py                    # 实现 base.py 时创建
    base.py                        # FeatureProjection / PredictionHead / BaseStockModel
    gru_model.py                   # GRU Baseline，进入第 3 步时创建并实现
    transformer_model.py           # Transformer Encoder，进入第 6 步时创建并实现
    attention_pool.py              # GRU attention 消融时创建并实现

  training/
    __init__.py                    # 实现训练模块时创建
    metrics.py                     # daily IC / RankIC / ICIR
    trainer.py                     # 统一训练循环、early stopping、checkpoint

  evaluation/
    __init__.py                    # 正式评估/回测阶段创建
    prediction_io.py               # 可选：predictions.parquet 输出规范
    backtest.py                    # 后续 Top-K 回测器落地时创建

scripts/
  train_sequence.py                # 序列模型统一训练入口，训练闭环阶段创建

configs/
  sequence_gru_baseline.yaml       # GRU lookback=20 date-aware MSE+IC baseline
  sequence_gru_l60.yaml            # GRU lookback=60 对比
  sequence_transformer.yaml        # Transformer lookback=20
```

暂不建议创建：

```text
src/models/lstm_model.py           # LSTM 是 Backlog，当前不进入主线
src/data/dataset.py                # 旧版手工滑窗 Dataset，不再作为主接口
src/data/sampler.py                # 当前默认 DataLoader 足够，DailyBatchSampler 后置
scripts/backtest.py                # 正式回测器尚未进入当前最小闭环
requirements.txt                   # 当前默认使用 conda dl_env，先不重复维护
```

目录职责边界：

| 目录 | 职责 | 当前阶段 |
| --- | --- | --- |
| `src/data/` | 只负责读取已构建好的模型数据集，不重新做特征工程或滑窗 | 已启动 |
| `src/models/` | 模型组件与模型主体，先 GRU，再 Transformer | 即将启动 |
| `src/training/` | 训练循环、指标、checkpoint、early stopping | GRU 可运行前启动 |
| `src/evaluation/` | 预测文件整理与后续回测评估 | 后置 |
| `scripts/` | 命令行入口，不承载核心业务逻辑 | 训练闭环阶段启动 |
| `configs/` | 每个实验的可复现配置快照来源 | 与训练脚本同步启动 |

## 12. 下一步开发顺序

建议按以下工程顺序实现：

1. `src/data/sequence_npz_dataset.py`：直接读取 NPZ，并按 split 过滤。
2. `src/models/base.py`：实现 `FeatureProjection` 与 `PredictionHead`。
3. `src/models/gru_model.py`：完成 GRU baseline。
4. `src/training/metrics.py`：实现 daily IC / RankIC。
5. `src/training/trainer.py`：跑通 E01。
6. `src/models/transformer_model.py`：完成 Transformer Encoder。
7. `scripts/train_sequence.py`：用 YAML 配置统一启动 GRU/Transformer。

第二阶段的最小闭环是：E01 GRU lookback=20 训练完成，并输出可被回测器读取的 `predictions.parquet`。
