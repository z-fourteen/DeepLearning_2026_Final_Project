# Transformer Stock Model 架构文档

> 模型名称: `TransformerStockModel` (E03 配置: l20 + CLS Pooling)
>
> 文件位置: `src/models/transformer.py`
>
> 特征集: `advanced_sequence_clean_v1` (13 alpha features)
>
> 输入: `[B, T, 13]` → 输出: `[B]` (pred_score)

---

## 一、整体架构概览

```
Input x [B, 20, 13]     — 13维 clean alpha 特征, 20天窗口
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 1: FeatureProjection                                │
│   Linear(13→64) + LayerNorm(64) + Dropout(0.1)           │
│   参数量: 952                                            │
└──────────────────────┬──────────────────────────────────┘
                     │ [B, 20, 64]
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 2: SinusoidalPositionalEncoding                    │
│   固定正弦位置编码 (buffer, 不可训练)                      │
│   参数量: 0 (precomputed)                                │
└──────────────────────┬──────────────────────────────────┘
                     │ [B, 20, 64] (x + pe)
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 3-4: TransformerEncoder x2 (Pre-LN)                │
│   L=2 层, H=4 头, d_ff=128, GELU                         │
│   参数量: 67,072 (86.3%)                                 │
│                                                           │
│   ┌─ EncoderLayer 1 ──────────────────────────────┐     │
│   │ Self-Attention(4头) + FFN(64→128→64) + 2xLN    │     │
│   │ 参数量: 33,472                                  │     │
│   ├───────────────────────────────────────────────┤     │
│   │ EncoderLayer 2 (同上结构)                       │     │
│   │ 参数量: 33,472                                  │     │
│   ├───────────────────────────────────────────────┤     │
│   │ Final LayerNorm (norm_first=True 时追加)       │     │
│   │ 参数量: 128                                     │     │
│   └───────────────────────────────────────────────┘     │
└──────────────────────┬──────────────────────────────────┘
                     │ [B, 21, 64] (含 CLS token)
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 5: CLS Token Extraction                             │
│   取 [:, 0] 作为全局表示                                   │
│   参数量: 64 (CLS token 可学习向量)                       │
└──────────────────────┬──────────────────────────────────┘
                     │ [B, 64]
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 6: Context LayerNorm                               │
│   LayerNorm(64)                                          │
│   参数量: 128 (0.2%)                                     │
└──────────────────────┬──────────────────────────────────┘
                     │ [B, 64]
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 7: PredictionHead                                  │
│   Linear(64→64) + GELU + Dropout(0.3) + Linear(64→1)     │
│   squeeze(-1) → [B]                                      │
│   参数量: 4,225 (5.4%)                                   │
└──────────────────────┬──────────────────────────────────┘
                     │
                     ▼
              pred_score [B]
```

---

## 二、特征集说明 (advanced_sequence_clean_v1)

### Alpha Features (13 维) — 模型输入

| # | 特征名 | 类别 | 说明 |
|---|--------|------|------|
| 1 | lag1_net_mf_strength_20d_mean | 资金流 | 20日净资金流强度均值 |
| 2 | lag1_net_mf_strength_60d_mean | 资金流 | 60日净资金流强度均值 |
| 3 | lag1_close_position | 价格位置 | 收盘价相对高低位 |
| 4 | lag1_excess_ret_10d_mean | 超额收益 | 10日超额收益均值 |
| 5 | lag1_excess_ret_1d | 超额收益 | 1日超额收益 |
| 6 | lag1_excess_ret_5d_mean | 超额收益 | 5日超额收益均值 |
| 7 | lag1_industry_neutral_ret_1d | 中性收益 | 行业中性化1日收益 |
| 8 | lag1_ret_1d | 原始收益 | 1日收益率 |
| 9 | lag1_ret_20d | 原始收益 | 20日收益率 |
| 10 | lag1_ret_5d_mean | 原始收益 | 5日收益均值 |
| 11 | lag1_bollinger_z_20d | 技术指标 | 布林带Z-score |
| 12 | lag1_ma_ratio_20_60 | 技术指标 | 短期/长期均线比 |
| 13 | lag1_macd_hist | 技术指标 | MACD柱状图 |

### Residualized Style Features (可选 +5 维)

用于消融实验的残差化风格特征（对行业+风格因子正交残差）：
- turnover_cost_proxy__resid_style
- turnover_20d_std__resid_style
- turnover_60d_std__resid_style
- amount_rank_pct__resid_style
- amount_log__resid_style

### 与旧版 62 维特征集对比

| 项目 | 旧版 (legacy_full62_v1) | 新版 (clean v1) |
|------|------------------------|-----------------|
| 特征数 | 62 | **13** (alpha-only) / **18** (+resid style) |
| 数据源 | advanced_sequence_fixed | advanced_sequence_clean_v1 |
| 风格暴露 | 原始混入 | 已中性化/分离 |
| 冗余特征 | 存在共线性 | 共线性剪枝完成 |
| 控制特征 | 混在输入中 | 分离到 risk/tradability controls |

---

## 三、各层详细说明

### 1. FeatureProjection — 特征投影层

| 项目 | 值 |
|------|-----|
| 类名 | `FeatureProjection` (`src/models/base.py`) |
| 结构 | `Linear(13, 64)` → `LayerNorm(64)` → `Dropout(0.1)` |
| 参数量 | **952** (vs 旧版 4,160) |
| 输入/输出 | `[B, T, 13]` → `[B, T, 64]` |

**作用：**
- 将 13 维 clean alpha 特征映射到模型统一维度 `d_model=64`
- 升维比例 ~4.9x（vs 旧版 1.03x），给模型更多表达空间补偿输入维度缩减
- LayerNorm 保证各特征尺度一致

---

### 2. SinusoidalPositionalEncoding — 正弦位置编码

| 项目 | 值 |
|------|-----|
| 最大长度 | `cls_max_len=256` (> l20 的 20 步需求) |
| 参数量 | **0** (注册为 buffer, 不参与梯度更新) |

---

### 3 & 4. TransformerEncoder x 2 — 核心时序建模层

#### 单层结构 (Pre-LN 变体)

```
EncoderLayer k (k = 1, 2):
                    z_{k-1}  [B, 20, 64]
                        │
         ┌──────────────┴──────────────┐
         ▼                              │
   ┌──────────────┐                    │
   │ LayerNorm_1  │                    │
   └──────┬───────┘                    │
          │                            │
          ▼                            │
   ┌──────────────┐                    │
   │MultiHeadAttn │ (H=4, d_k=d_v=16) │
   │  Q,K,V 全部  │                    │
   │  来自 z_{k-1} │                   │
   └──────┬───────┘                    │
          │                            │
          ▼                            │
   Dropout(0.1)                        │
          │                            │
          ├── (+) ←───────────────────┘ (残差连接)
          │
          ▼
   ┌──────────────┐
   │ LayerNorm_2  │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  FeedForward │  Linear(64→128) → GELU → Dropout → Linear(128→64)
   └──────┬───────┘
          │
          ▼
   Dropout(0.1)
          │
          ├── (+) ←────────────────── (残差连接)
          │
          ▼
        z_k  [B, 21, 64]  (含 CLS 时 T+1)
```

**参数量不变**: Attention 和 FFN 参数仅依赖 d_model=64，与输入特征数无关。

| 子组件 | 每层数量 | 2 层合计 | 占 Encoder |
|--------|---------|----------|-----------|
| Self-Attention (W_Q,W_K,W_V,W_O) | 16,640 | 33,280 | 49.6% |
| Feed Forward Network | 16,576 | 33,152 | 49.4% |
| LayerNorm (每层 2个) | 256 | 512 | 0.8% |
| Final LN (norm_first) | — | 128 | 0.2% |

---

### 5. Pooling 层 — 三种变体

#### A. CLS Pooling (E03 默认)

```python
# 构造时:
self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # [1, 1, 64]

# forward 时:
cls = self.cls_token.expand(B, -1, -1)  # [B, 1, 64]
z = torch.cat([cls, z], dim=1)          # [B, T+1, 64]
encoded = encoder(z)                     # [B, T+1, 64]
context = encoded[:, 0]                  # [B, 64]  ← 取 CLS 向量
```

- CLS token 通过 self-attention 从所有时间步吸收信息
- 适合 l20 较短窗口场景

#### B. Last Step Pooling

```python
context = encoded[:, -1]  # [B, 64]  ← 取最后时间步
```

- 无额外参数
- 完全依赖最后一时刻的编码状态

#### C. Attention Pooling (推荐用于 l60)

```python
# AttentionPooling 网络:
scores = MLP(encoded)                 # [B, T, 64] → [B, T]
weights = softmax(scores, dim=-1)    # [B, T], Σ weights_t = 1.0
context = Σ (weights_t * encoded_t)   # [B, 64]
```

- 参数: `Linear(64→32) + Tanh + Linear(32→1)` = **2,113**
- 自适应学习哪些历史时刻更重要

---

### 6 & 7. Context Norm + PredictionHead

与旧版完全相同，不依赖输入特征数。

| 模块 | 结构 | 参数量 |
|------|------|--------|
| Context Norm | `LayerNorm(64)` | 128 |
| PredictionHead | `Linear(64→64) + GELU + Dropout(0.3) + Linear(64→1)` | 4,225 |

---

## 四、完整参数量汇总表

| # | 模块 | 参数量 | 占比 | 是否可训练 |
|---|------|--------|------|-----------|
| 1 | FeatureProjection (Linear+LN) | **952** | 1.3% | Yes |
| 2 | SinusoidalPE | 0 | 0.0% | No (buffer) |
| 3 | CLS Token | 64 | 0.1% | Yes |
| 4-5 | TransformerEncoder x2 (Pre-LN) | 67,072 | **89.9%** | Yes |
| 6 | Context LayerNorm | 128 | 0.2% | Yes |
| 7 | PredictionHead (MLP) | 4,225 | **5.7%** | Yes |
| | **总计 (CLS pooling)** | **74,441** | **100%** | |

> 注意：相比旧版 62 维输入 (77,698)，总参数减少 ~4.2%，主要来自 FeatureProjection 的输入维度缩减。

---

## 五、信息流与张量形状追踪

以 E03 的 l20 clean 数据为例：

```
步骤   操作                          形状              说明
──────────────────────────────────────────────────────────────────
①     原始输入 x                     [B, 20, 13]      13维clean alpha, 20天窗口
│
②     input_proj: Linear+LN+Drop     [B, 20, 64]      13→64升维投影
│
③     pos_encoder: x + sin(pe)       [B, 20, 64]      注入绝对位置信息
│
④     cat CLS token                  [B, 21, 64]      在位置0拼接可学习CLS
│
⑤     encoder layer 1 (Pre-LN):
  │    ├─ norm1                       [B, 21, 64]
  │    ├─ MHA(4头, d_k=16)            [B, 21, 64]     跨时间步attention
  │    ├─ residual add               [B, 21, 64]
  │    ├─ norm2                       [B, 21, 64]
  │    ├─ FFN(64→128→64)             [B, 21, 64]     非线性变换
  │    └─ residual add               [B, 21, 64]
│
⑥     encoder layer 2 (同上结构)      [B, 21, 64]     更深层时序交互
│
⑦     final LayerNorm                [B, 21, 64]
│
⑧     取 CLS: encoded[:, 0]          [B, 64]         CLS token输出
│
⑨     context_norm: LN               [B, 64]         输出稳定化
│
⑩     PredictionHead:
  │    ├─ Linear(64→64) + GELU        [B, 64]
  │    ├─ Dropout(0.3)                [B, 64]
  │    ├─ Linear(64→1)                [B, 1]
  │    └─ squeeze(-1)                 [B]            最终pred_score
│
▼
      output: pred_score              [B]             用于横截面排序
```

---

## 六、超参数配置速查

### E03: Clean Alpha Only (l20, CLS)

```yaml
model:
  name: transformer_encoder
  num_features: 13                   # ★ 13 个 clean alpha 特征
  lookback: 20
  d_model: 64
  input_dropout: 0.1
  num_encoder_layers: 2
  num_heads: 4                       # d_model/H = 16 per-head
  dim_feedforward: 128
  attn_dropout: 0.1
  ff_dropout: 0.1
  activation: gelu
  norm_first: true                   # Pre-LN
  positional_encoding: sinusoidal
  pooling: cls                       # CLS token pooling
  cls_max_len: 256
  head_hidden_dim: 64
  head_dropout: 0.3

training:
  optimizer: adamw
  learning_rate: 3e-4
  weight_decay: 1e-3                 # Transformer 推荐 > RNN
  loss_fn: huber                      # HuberLoss(delta=1.0)
  huber_delta: 1.0
  scheduler: warmup_cosine           # 10% warmup + cosine decay
  batch_size: 256
  batch_mode: date                   # 按日组织 batch 计算 IC
  max_grad_norm: 1.0
  early_stop_patience: 10
  early_stop_metric: rank_ic_mean
```

### E04 (消融): Clean Alpha + Residual Style (l20, CLS)

```yaml
model:
  num_features: 18                   # ★ 13 alpha + 5 resid style
  # ... 其余同 E03 ...
```

### E05 (消融): Attention Pooling (l20/l60)

```yaml
model:
  pooling: attention                 # ★ 改用注意力池化
  # 注意: attention pooling 不需要 CLS token
  # ... 其余同 E03/E04 ...
```

---

## 七、设计决策总结

| 设计点 | 选择 | 理由 |
|--------|------|------|
| 13 维输入 | clean alpha | 去冗余去共线性，纯 alpha 信号 |
| d_model=64 | 小而紧凑 | 205K 样本的中等规模数据集，避免过拟合；高升维比补偿输入少 |
| L=2 层 | 浅层 | 超参少，训练快；深层需更多数据 |
| H=4 头 | 标准 | d_k=16，每头有足够表达能力 |
| d_ff=128 (2x) | 标准 FFN 比例 | 平衡容量与参数效率 |
| Pre-LN | 稳定优先 | 梯度传播好，允许更高 LR |
| Sinusoidal PE | 泛化优先 | 不增加参数，避免小数据集过拟合 |
| CLS Pooling (l20 默认) | 短窗合适 | 20 步窗口下 CLS 能有效聚合 |
| Attention Pooling (l60 推荐) | 长窗更优 | 60 步窗口自适应加权效果好 |
| GELU | 平滑激活 | 比 ReLU 更适合 Transformer |
| AdamW + WD=1e-3 | 解耦衰减 | 权重衰减独立于梯度更新 |
| WarmupCosine | 先热身后衰减 | 避免 early stage 不稳定振荡 |

---

## 八、运行方式

```powershell
# E03: Clean Alpha Only, l20, CLS
conda run -n dl_env python scripts/modeling/train_sequence.py `
  --config configs/models/transformer_l20_clean_alpha_only.yaml

# E04: Clean Alpha + Resid Style, l20, CLS
conda run -n dl_env python scripts/modeling/train_sequence.py `
  --config configs/models/transformer_l20_clean_alpha_resid_style.yaml

# Dry run (只检查配置不训练)
conda run -n dl_env python scripts/modeling/train_sequence.py `
  --config configs/models/transformer_l20_clean_alpha_only.yaml --dry-run
```
