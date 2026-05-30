# 模型构建手册

## 第一部分：模型构建分析

### 1. 任务与模型路线

**任务定义**：输入创业板成分股过去 `lookback` 个交易日的特征序列 `[B, T, F]`，输出每只股票未来 5 日超额收益的预测分数 `[B, 1]`，用于横截面排序选股。

**模型路线**（由浅入深，确保每一步都可独立验证）：

```
GRU (主 Baseline) → LSTM (Baseline 对比) → GRU+AttnPool (进阶变体) → Transformer (进阶模型)
```

**所有模型共享**：相同的输入特征、标签 (`y_excess_5d`)、训练/验证/测试划分、评估指标（IC、RankIC、Top-K 回测）。仅对比模型架构差异。

---

### 2. 统一接口设计

所有模型继承同一基类，保证输入输出一致：

```python
class BaseStockModel(nn.Module):
    def __init__(self, num_features: int, config: dict):
        """
        Args:
            num_features: 输入特征维度 F (由特征工程决定，约 30~50)
            config: 模型超参数字典
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, F]  过去 lookback 个交易日的特征序列
        Returns:
            [B, 1]  预测分数 (未来 5 日超额收益的排序分数)
        """
        raise NotImplementedError
```

**统一预测输出格式**：

```text
predictions.parquet:
  trade_date | ts_code | pred_score | model_name
```

接入同一个回测器进行公平对比。

---

### 3. GRU 模型（主 Baseline）

#### 架构

```text
输入: [B, T, F]
  │
  ▼
FeatureProjection: Linear(num_features, embed_dim)
  │                                          F → embed_dim
  ▼
GRU(input_size=embed_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers, dropout=rnn_dropout)
  │
  ▼
Pooling: 取最后时刻隐状态  →  [B, rnn_hidden_dim]
  │     (可选替换为 Attention Pooling，见第 5 节)
  ▼
PredictionHead:
  Linear(rnn_hidden_dim, head_hidden_dim) + ReLU + Dropout
  Linear(head_hidden_dim, 1)
  │
  ▼
输出: [B, 1]
```

#### 超参数

| 参数名 | 推荐值 | 含义 |
|--------|--------|------|
| `embed_dim` | 64 | 特征映射后的统一维度 |
| `rnn_hidden_dim` | 128 | GRU 隐状态维度 |
| `rnn_num_layers` | 2 | GRU 层数 |
| `rnn_dropout` | 0.2 | GRU 层间 dropout |
| `head_hidden_dim` | 64 | 预测头中间层维度 |
| `head_dropout` | 0.3 | 预测头 dropout |

#### 参数量估算（以 F=50 为例）

| 组件 | 计算 | 参数量 |
|------|------|--------|
| FeatureProjection | 50 x 64 | 3,200 |
| GRU Layer 1 | (64+128) x 128 x 3 | 73,728 |
| GRU Layer 2 | (128+128) x 128 x 3 | 98,304 |
| PredictionHead | 128x64 + 64x1 | 8,256 |
| **总计** | | **~183K** |

#### 选择 GRU 作为主 Baseline 的理由

- 参数量比 LSTM 少 ~25%，训练更快
- 金融短序列 (T=20) 下门控更少的 GRU 表现稳定
- 足以验证时序建模基本能力，不易过拟合

---

### 4. LSTM 模型（Baseline 对比）

#### 架构

与 GRU 完全对称，仅将 GRU 替换为 LSTM：

```text
输入: [B, T, F]
  ▼
FeatureProjection: Linear(num_features, embed_dim)
  ▼
LSTM(input_size=embed_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers, dropout=rnn_dropout)
  ▼
Pooling: 取最后时刻隐状态 (h_n[-1])  →  [B, rnn_hidden_dim]
  ▼
PredictionHead (与 GRU 共享结构)
  ▼
输出: [B, 1]
```

#### 与 GRU 的差异

| 维度 | GRU | LSTM |
|------|-----|------|
| 门控数 | 2 (reset, update) | 3 (input, forget, output) |
| 参数量 | ~183K | ~245K |
| 归纳偏置 | 更简洁，短序列友好 | 门控更细粒度，理论上长依赖更强 |

**对比实验控制变量**：输入特征、lookback、训练超参数、评估指标全部相同，仅对比模型结构。

---

### 5. 注意力池化（RNN 进阶变体）

GRU 和 LSTM 均可挂载此模块，将"取最后隐状态"替换为自适应加权聚合。

```text
RNN 输出所有时间步隐状态: all_hidden [B, T, rnn_hidden_dim]
  │
  ▼
Attention Pooling:
  attn_score = Linear(rnn_hidden_dim, 1)(all_hidden)   # [B, T, 1]
  attn_weight = softmax(attn_score, dim=time_dim)       # [B, T, 1]
  pooled = sum(attn_weight * all_hidden)                 # [B, rnn_hidden_dim]
  │
  ▼
PredictionHead → 输出
```

**作用**：让模型学习"历史中哪些时间步更重要"，而非默认只看最后一步。

---

### 6. Transformer 模型（进阶）

#### 架构

```text
输入: [B, T, F]
  │
  ▼
FeatureProjection + PositionalEncoding:
  x = Linear(num_features, embed_dim)(x)       # [B, T, embed_dim]
  x = x + positional_encoding                   # 正弦 (默认) 或可学习
  │
  ▼
(可选) 拼接 CLS Token:  x → [B, T+1, embed_dim]
  │
  ▼
Transformer Encoder (num_encoder_layers 层):
  每层包含:
    MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
      + LayerNorm + Residual
    FeedForward(embed_dim → ff_hidden_dim → embed_dim)
      + GELU + ff_dropout + LayerNorm + Residual
  │
  ▼
Pooling: 取 CLS Token (x[:, 0]) 或最后一时间步  →  [B, embed_dim]
  │
  ▼
PredictionHead:
  Linear(embed_dim, head_hidden_dim) + GELU + Dropout
  Linear(head_hidden_dim, 1)
  │
  ▼
输出: [B, 1]
```

#### 超参数

| 参数名 | 推荐值 | 含义 |
|--------|--------|------|
| `embed_dim` | 64 | 特征映射维度（也是 Transformer 的 d_model） |
| `num_encoder_layers` | 2 | Encoder 层数 |
| `num_heads` | 4 | 多头注意力头数 |
| `ff_hidden_dim` | 128 | FFN 中间层维度 |
| `attn_dropout` | 0.1 | 注意力 dropout |
| `ff_dropout` | 0.1 | FFN dropout |
| `use_cls_token` | True | 是否使用 CLS Token 池化 |
| `learnable_pe` | False | True=可学习位置编码，False=正弦位置编码 |

#### 参数量估算（以 F=50 为例）

| 组件 | 参数量 |
|------|--------|
| FeatureProjection | 3,200 |
| PositionalEncoding | 1,280 |
| Attention x2 | ~99K |
| FeedForward x2 | ~33K |
| LayerNorm x4 | 256 |
| PredictionHead | 2,112 |
| **总计** | **~140K** |

#### 与 RNN 的对比

| 维度 | GRU/LSTM | Transformer |
|------|----------|-------------|
| 时序建模 | 顺序递归，逐步压缩 | 全局注意力，任意两步直达 |
| 并行训练 | 串行 | 全并行 |
| 归纳偏置 | 强时序先验（适合短序列） | 弱先验（需要更多数据） |
| 过拟合风险 | 中等 | 较高（金融小数据下） |

#### 注意事项

- T=20 极短，使用标准全注意力即可，无需稀疏/窗口变体
- 正则化必须从严：权重衰减比 RNN 高一个数量级（见第 8 节）
- 推荐使用 Warmup + Cosine 学习率调度

---

### 7. 数据加载设计

#### StockSequenceDataset

按股票分组，对每只股票滑窗构造序列样本：

```python
class StockSequenceDataset(Dataset):
    """
    每个样本:
        X = 股票 i 在 [t-lookback+1, t] 的特征序列  →  [lookback, F]
        y = 股票 i 在 [t+1, t+5] 的超额收益           →  标量
    返回: (X_tensor, y_scalar, trade_date, ts_code)
    """
    def __init__(self, features_df, labels_df, lookback=20):
        ...
```

#### DailyBatchSampler

按 `trade_date` 组织 batch，保证同一 batch 来自同一天，便于计算每日 IC：

```python
class DailyBatchSampler:
    """
    将样本按 trade_date 分组，每次 yield 一个交易日的所有样本。
    用于验证/测试阶段计算日频 IC。
    训练阶段可混日打乱，也可按日组织（取决于实验需求）。
    """
    ...
```

---

### 8. 训练与防过拟合策略

#### 统一训练配置

| 参数名 | GRU/LSTM 推荐值 | Transformer 推荐值 | 含义 |
|--------|-----------------|---------------------|------|
| `optimizer` | AdamW | AdamW | 优化器 |
| `weight_decay` | 1e-4 | 1e-3 | 权重衰减（Transformer 更大） |
| `peak_lr` | 1e-3 | 1e-3 | 最大学习率 |
| `lr_scheduler` | CosineAnnealing | Warmup(10% steps) + CosineAnnealing | 学习率调度 |
| `batch_size` | 256~512 | 256~512 | 批大小 |
| `loss_fn` | MSE 或 Huber | MSE 或 Huber | 损失函数 |
| `max_grad_norm` | 1.0 | 1.0 | 梯度裁剪 |
| `early_stop_patience` | 15 | 10 | 早停耐心值（监控验证集 IC） |

#### 防过拟合清单

| 策略 | 说明 |
|------|------|
| 时间划分 | train → val → test，禁止随机划分 |
| Early Stopping | 监控验证集 IC，连续 N 个 epoch 不提升则停止 |
| Dropout | RNN 层间 0.2~0.3，MLP 头 0.3，Transformer attention/ff 0.1 |
| 权重衰减 | GRU/LSTM 1e-4，Transformer 1e-3 |
| 梯度裁剪 | max_norm=1.0，防止梯度爆炸 |
| 标签裁剪 | 横截面 1%/99% 分位 winsorize（由数据处理侧完成） |
| 特征标准化 | 横截面 z-score，只用训练期参数（由数据处理侧完成） |

---

### 9. 评估指标

#### 预测质量（日频）

| 指标 | 公式 | 含义 |
|------|------|------|
| IC | Pearson(pred_score, y_excess_5d)，每日计算 | 预测分数与实际收益的线性相关 |
| RankIC | Spearman(pred_score, y_excess_5d)，每日计算 | 预测排名与实际排名的序相关 |
| ICIR | mean(IC) / std(IC) | IC 的稳定性 |

#### 投资绩效（Top-K 组合回测）

| 指标 | 含义 |
|------|------|
| 年化收益率 | Top-K 等权组合的年化收益 |
| 超额收益 | 策略收益 - 创业板指收益 |
| Sharpe Ratio | 超额收益 / 超额波动 x sqrt(252) |
| Max Drawdown | 累计净值的最大回撤 |

---

## 第二部分：具体执行步骤

### 步骤 1：搭建项目骨架

创建以下目录和文件结构：

```text
src/
  __init__.py
  models/
    __init__.py
    base.py              # BaseStockModel 基类
    gru_model.py         # GRUModel
    lstm_model.py        # LSTMModel
    transformer_model.py # TransformerModel
    attention_pool.py    # AttentionPooling 模块
  data/
    __init__.py
    dataset.py           # StockSequenceDataset
    sampler.py           # DailyBatchSampler
  training/
    __init__.py
    trainer.py           # 统一训练循环
    utils.py             # IC 计算、早停、日志等工具函数
  evaluation/
    __init__.py
    metrics.py           # IC, RankIC, ICIR
    backtest.py          # Top-K 回测
configs/
    gru_baseline.yaml    # GRU 实验配置
    lstm_baseline.yaml   # LSTM 实验配置
    transformer.yaml     # Transformer 实验配置
scripts/
    train.py             # 训练入口脚本
    backtest.py          # 回测入口脚本
requirements.txt
```

**产出**：空的项目骨架，所有文件可 import。

---

### 步骤 2：实现 BaseStockModel 和 FeatureProjection

在 `src/models/base.py` 中：

1. 定义 `BaseStockModel(nn.Module)` 基类，包含：
   - `__init__(num_features, config)` 接口
   - `forward(x) -> Tensor` 抽象方法
   - 公共的 `FeatureProjection` 层：`Linear(num_features, embed_dim)`
2. 定义 `PredictionHead` 模块（含 dropout、激活函数、输出层），供所有模型复用。

**验收**：基类可实例化（抽象方法由子类实现），`FeatureProjection` 和 `PredictionHead` 单独可运行。

---

### 步骤 3：实现 GRUModel

在 `src/models/gru_model.py` 中：

1. 继承 `BaseStockModel`
2. `__init__` 中创建：FeatureProjection → GRU → Pooling → PredictionHead
3. `forward` 中：投影 → GRU 编码 → 取最后隐状态 → 预测头 → 输出 `[B, 1]`
4. 超参数通过 `config` 字典传入，使用步骤 3 中的意图化参数名

**验收**：随机输入 `[B, T, F]` 张量，输出 `[B, 1]`，无报错；`print(model)` 可查看参数量。

---

### 步骤 4：实现 AttentionPooling 模块

在 `src/models/attention_pool.py` 中：

1. 实现 `AttentionPooling(hidden_dim)` 模块
2. 输入 `[B, T, hidden_dim]`，输出 `[B, hidden_dim]`
3. 内部：`Linear(hidden_dim, 1)` → `softmax(dim=1)` → 加权求和

**验收**：随机输入，输出形状正确；可视化一组 attention 权重确认分布合理。

---

### 步骤 5：实现 LSTMModel

在 `src/models/lstm_model.py` 中：

1. 继承 `BaseStockModel`，结构与 GRU 对称
2. Pooling 层支持切换：默认取最后隐状态，可选挂载 `AttentionPooling`
3. 通过 `config["use_attention_pool"]` 控制是否启用注意力池化

**验收**：与 GRUModel 使用相同的随机输入，输出形状一致。

---

### 步骤 6：实现 TransformerModel

在 `src/models/transformer_model.py` 中：

1. 继承 `BaseStockModel`
2. 实现正弦位置编码 `SinusoidalPositionalEncoding(lookback, embed_dim)`
3. 可选 CLS Token：`nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)`
4. 使用 `nn.TransformerEncoder` 构建编码器
5. Pooling：CLS Token 或最后时间步，通过 `config["use_cls_token"]` 控制
6. PredictionHead 激活函数使用 GELU（与 GRU/LSTM 的 ReLU 区分）

**验收**：随机输入输出正确；无 CLS Token 和有 CLS Token 两种模式均可运行。

---

### 步骤 7：实现 StockSequenceDataset

在 `src/data/dataset.py` 中：

1. 输入：`features_df`（含 `trade_date, ts_code, feat_1...`）和 `labels_df`（含 `trade_date, ts_code, y_excess_5d`）
2. 按 `ts_code` 分组，按 `trade_date` 排序
3. 对每只股票滑窗构造样本：窗口长度 `lookback`，标签对齐到窗口末日的未来收益
4. 历史不足 `lookback` 的股票跳过
5. `__getitem__` 返回 `(X_tensor [lookback, F], y_scalar, trade_date, ts_code)`

**验收**：使用 mock DataFrame 构造小数据集，验证滑窗逻辑和返回形状。

---

### 步骤 8：实现 DailyBatchSampler

在 `src/data/sampler.py` 中：

1. 将 dataset 中的样本按 `trade_date` 分组
2. 每次 yield 一个交易日的所有样本索引
3. 支持训练模式（混日打乱）和评估模式（按日顺序）

**验收**：与 `StockSequenceDataset` 配合使用，验证同一 batch 内日期一致。

---

### 步骤 9：实现训练循环

在 `src/training/trainer.py` 中实现统一 `Trainer` 类：

1. **初始化**：接收 model、train_loader、val_loader、optimizer、scheduler、loss_fn、config
2. **train_epoch**：
   - 前向 → 计算 loss → 反向 → 梯度裁剪 (`clip_grad_norm_`) → optimizer.step
3. **validate**：
   - 关闭梯度，逐 batch 收集 `(pred_score, y_true, trade_date)`
   - 按日计算 IC、RankIC，返回均值
4. **fit**：
   - epoch 循环，每个 epoch 后 validate
   - Early Stopping：验证集 IC 连续 `early_stop_patience` 个 epoch 不提升则停止
   - 保存最优模型（按验证集 IC）
   - 记录训练日志（loss、IC、学习率等）

在 `src/training/utils.py` 中实现：
- `compute_daily_ic(pred, y, dates)`：按日分组计算 Pearson IC
- `compute_daily_rank_ic(pred, y, dates)`：按日分组计算 Spearman RankIC
- `compute_icir(daily_ics)`：IC 均值 / IC 标准差

**验收**：用 mock 数据跑通完整 train → validate → early stop 流程。

---

### 步骤 10：实现配置管理

在 `configs/` 下创建 YAML 配置文件，示例 `gru_baseline.yaml`：

```yaml
model:
  name: "gru"
  embed_dim: 64
  rnn_hidden_dim: 128
  rnn_num_layers: 2
  rnn_dropout: 0.2
  head_hidden_dim: 64
  head_dropout: 0.3
  use_attention_pool: false

training:
  peak_lr: 1e-3
  weight_decay: 1e-4
  lr_scheduler: "cosine"
  batch_size: 256
  loss_fn: "mse"
  max_epochs: 200
  max_grad_norm: 1.0
  early_stop_patience: 15

data:
  lookback: 20
  label_col: "y_excess_5d"
```

在 `scripts/train.py` 中实现入口：
- 读取配置文件
- 构造 Dataset、DataLoader
- 根据配置实例化模型
- 调用 Trainer.fit()
- 保存最优模型和预测结果

**验收**：`python scripts/train.py --config configs/gru_baseline.yaml` 可运行（需数据就绪后端到端验证）。

---

### 步骤 11：实现评估与回测

在 `src/evaluation/metrics.py` 中：
- 汇总各模型在验证集和测试集上的 IC、RankIC、ICIR
- 分年度计算 IC，观察稳定性

在 `src/evaluation/backtest.py` 中：
- 每日按 `pred_score` 降序排列股票
- 选取 Top-K（K=10/20/30）等权持有
- 计算组合日收益、累计净值
- 计算年化收益、超额收益（相对创业板指）、Sharpe、Max Drawdown
- 扣除交易成本（单边 0.1%~0.2%）

**验收**：用 mock 预测结果跑通回测，输出净值曲线和绩效指标。

---

### 步骤 12：实验矩阵执行

按以下顺序跑实验，**每次只改一个变量**：

| 编号 | 模型 | 变量 | 对比目标 |
|------|------|------|----------|
| E01 | GRU | lookback=20, last_hidden | 主 Baseline 基准 |
| E03 | LSTM | lookback=20, last_hidden | 与 E01 对比门控机制 |
| E05 | GRU | lookback=20, attention_pool | 与 E01 对比池化方式 |
| E04 | LSTM | lookback=20, attention_pool | 与 E03 对比池化方式 |
| E02 | GRU | lookback=60, last_hidden | 与 E01 对比序列长度 |
| E06 | Transformer | CLS + sinusoidal_pe | 进阶模型 |
| E07 | Transformer | last_step + sinusoidal_pe | 与 E06 消融池化方式 |
| E08 | Transformer | CLS + learnable_pe | 与 E06 消融位置编码 |

每个实验产出：
- `outputs/runs/{run_id}/config.yaml` — 实验配置快照
- `outputs/runs/{run_id}/metrics.json` — IC、RankIC、ICIR、回测指标
- `outputs/runs/{run_id}/predictions.parquet` — 每日每股票预测分数
- `outputs/runs/{run_id}/figures/` — 训练曲线、IC 时序、净值曲线

---

### 步骤 13：对接数据与联调

当数据处理侧（同伴）产出以下文件后：

```text
data/processed/features_daily.parquet   # [trade_date, ts_code, feat_1, ..., feat_F]
data/processed/labels_daily.parquet     # [trade_date, ts_code, y_excess_5d]
data/processed/split_config.yaml        # train/val/test 时间划分
```

执行联调：
1. 确认 `features_daily.parquet` 中的特征列数和名称
2. 更新 `configs/*.yaml` 中的 `num_features`
3. 按时间划分构造 Dataset
4. 端到端运行 E01（GRU Baseline），确认数据流通畅
5. 修复联调中发现的问题后，依次执行实验矩阵
