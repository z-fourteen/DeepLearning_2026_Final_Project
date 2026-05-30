# Transformer Stock Model 架构文档

> 模型名称: `TransformerStockModel` (E06 配置: l60 + Attention Pooling)
>
> 文件位置: `src/models/transformer.py`
>
> 总参数量: **77,698** (全部可训练)
>
> 输入: `[B, T, 62]` → 输出: `[B]` (pred_score)

---

## 一、整体架构概览

```
Input x [B, 60, 62]
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 1: FeatureProjection                                │
│   Linear(62→64) + LayerNorm(64) + Dropout(0.1)           │
│   参数量: 4,160 (5.4%)                                   │
└──────────────────────┬──────────────────────────────────┘
                     │ [B, 60, 64]
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 2: SinusoidalPositionalEncoding                    │
│   固定正弦位置编码 (buffer, 不可训练)                      │
│   参数量: 0 (precomputed)                                │
└──────────────────────┬──────────────────────────────────┘
                     │ [B, 60, 64] (x + pe)
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
                     │ [B, 60, 64]
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 5: AttentionPooling                                │
│   Linear(64→32) + Tanh + Linear(32→1) + Softmax         │
│   对每个时间步打分 → 加权求和                             │
│   参数量: 2,113 (2.7%)                                   │
└──────────────────────┬──────────────────────────────────┘
                     │ [B, 64] (加权聚合)
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

## 二、各层详细说明

### 1. FeatureProjection — 特征投影层

| 项目 | 值 |
|------|-----|
| 类名 | `FeatureProjection` (`src/models/base.py`) |
| 结构 | `Linear(62, 64)` → `LayerNorm(64)` → `Dropout(0.1)` |
| 参数量 | **4,160** |
| 输入/输出 | `[B, T, 62]` → `[B, T, 64]` |

**子模块参数分解：**

```
input_proj.proj.weight:  [64, 62]     3,968   (95.3%)
input_proj.proj.bias:        [64]          64   (1.5%)
input_proj.norm.weight:      [64]          64   (1.5%)
input_proj.norm.bias:        [64]          64   (1.5%)
Dropout:                        -            0   (无参数)
```

**作用：**
- 将原始 62 维特征（动量、波动率、资金流、行业排名等）映射到模型统一维度 `d_model=64`
- 线性投影后立即做 LayerNorm，使各特征尺度一致，加速后续 attention 收敛
- Dropout 随机丢弃 10% 的特征维度，防止对特定特征过度依赖

**设计理由：**
- 62→64 是轻微升维，保留几乎全部原始信息（不是瓶颈）
- 如果直接输入 62 维到 MultiHeadAttention(4头)，62 不能被 4 整除（62/4=15.5），会报错

---

### 2. SinusoidalPositionalEncoding — 正弦位置编码

| 项目 | 值 |
|------|-----|
| 类名 | `SinusoidalPositionalEncoding` |
| 公式 | `PE(pos,2i)=sin(pos/10000^(2i/d))`, `PE(pos,2i+1)=cos(...)` |
| 最大长度 | `cls_max_len=256` (> l60 的 60 步需求) |
| 参数量 | **0** (注册为 buffer, 不参与梯度更新) |

**作用：**
- 为序列中每个时间步注入绝对位置信息。没有 PE 的 Transformer 等价于集合操作，无法区分 "第3天" 和 "第30天"
- 正弦编码是固定的数学函数，不随训练变化，泛化性优于可学习编码

**为什么不用 Learnable PE：**
- 当前 lookback=60 固定，正弦编码已足够
- 可学习 PE 多 16,384 个参数 (1×256×64)，对小数据集容易过拟合
- 作为消融实验保留接口，可通过配置切换

**与输入的关系：**
```
z = x + pe[:, :T]    # 逐元素相加，不改变张量形状
```

---

### 3 & 4. TransformerEncoder × 2 — 核心时序建模层

这是整个模型最重的模块，占总参数的 **86.3%**。

#### 3.1 单层结构 (Pre-LN 变体)

```
EncoderLayer k (k = 1, 2):
                    z_{k-1}  [B, 60, 64]
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
        z_k  [B, 60, 64]
```

#### 3.2 MultiHeadSelfAttention 内部细节

```
输入: z [B, 60, 64]

Q = z @ W_q^T + b_q    W_q: [64, 64], b_q: [64]
K = z @ W_k^T + b_k    W_k: [64, 64], b_k: [64]
V = z @ W_v^T + b_v    W_v: [64, 64], b_v: [64]

PyTorch 实现优化: 将 Q/K/V 的三个权重矩阵合并为 in_proj_weight [192, 64]
  → 前 64 行 = W_q, 中 64 行 = W_k, 后 64 行 = W_v

拆分为 H=4 头, 每头 d_k = d_v = d_model / H = 16:
  Head h:  Q_h [B,60,16], K_h [B,60,16], V_h [B,60,16]

注意力计算 (Scaled Dot-Product):
  Attention(Q_h, K_h, V_h) = softmax(Q_h @ K_h^T / sqrt(16)) @ V_h
                           → [B, 60, 60] @ [B, 60, 16] → [B, 60, 16]

拼接 4 头: cat([head_1,...,head_4], dim=-1) → [B, 60, 64]
输出投影: Out = concat @ W_o + b_o    W_o: [64, 64], b_o: [64]
```

**单层 Attention 参数量：**

```
self_attn.in_proj_weight:  [192, 64]   12,288    (合并Q/K/V)
self_attn.in_proj_bias:        [192]      192
self_attn.out_proj.weight:  [64, 64]    4,096
self_attn.out_proj.bias:         [64]       64
                                    ─────────
Attention 小计:                  16,640
```

**单层 FFN 参数量：**

```
linear1 (up-project): weight [128, 64]  8,192
                      bias      [128]     128
linear2 (down-project):weight [64, 128] 8,192
                       bias       [64]      64
                                    ─────────
FFN 小计:                           16,576
```

**单层 LayerNorm × 2：**

```
norm1: weight [64] + bias [64]  = 128
norm2: weight [64] + bias [64]  = 128
                              ──────
LN 小计:                        256
```

**单层总计: 33,472 | 两层总计: 66,944**

加上最终 Final LayerNorm (norm_first=True): **+128 = 67,072**

#### 3.3 Pre-LN vs Post-LN

当前使用 **Pre-LN** (`norm_first=True`)：

```python
# Pre-LN (当前方案):
x = x + Dropout(Attn(LayerNorm(x)))      # 先归一化再 attention
x = x + Dropout(FFN(LayerNorm(x)))        # 先归一化再 FFN
+ 最终输出再加一个 LayerNorm

# Post-LN (传统方案):
x = LayerNorm(x + Dropout(Attn(x)))
x = LayerNorm(x + Dropout(FFN(x)))
```

**选择 Pre-LN 的原因：**
- 训练稳定性显著更好：梯度在深层网络中传播更顺畅
- 允许更激进的学习率 (当前 peak_lr=3e-4)
- 对于只有 2 层的浅层 Transformer 差异不大，但保持一致性

#### 3.4 该层的作用

| 能力 | 说明 |
|------|------|
| **跨时间步交互** | 第 t 天的特征可以通过 attention 看到 t-1 到 t-59 天的信息 |
| **非线性特征组合** | 62 维特征在不同历史时刻之间的复杂关系 (如"20天前放量 + 昨日突破MA20") |
| **多粒度模式捕获** | 4 个 attention 头可以各自学习不同的时间模式 (短/中/长周期) |
| **信息增强/去噪** | FFN (64→128→64) 提供了非线性变换能力，相当于对信号做滤波 |

---

### 5. AttentionPooling — 自适应聚合层

| 项目 | 值 |
|------|-----|
| 类名 | `AttentionPooling` |
| 结构 | `Linear(64→32)` → `Tanh` → `Linear(32→1)` → `softmax` → 加权求和 |
| 参数量 | **2,113** (2.7%) |
| 输入/输出 | `[B, 60, 64]` → `[B, 64]` |

**参数分解：**

```
attention_net.0.weight:   [32, 64]    2,048   (96.9%)
attention_net.0.bias:        [32]       32
attention_net.2.weight:     [1, 32]      32
attention_net.2.bias:          [1]       1
                          ─────────
总计:                               2,113
```

**计算流程：**

```python
# Step 1: 给每一步打分
scores = MLP(encoded)                 # [B, 60, 64] → [B, 60, 1] → [B, 60]
# Step 2: 归一化为概率分布
weights = softmax(scores, dim=-1)    # [B, 60], Σ weights_t = 1.0
# Step 3: 加权聚合所有时间步
context = Σ (weights_t * encoded_t)   # [B, 60, 64] * [B, 60, 1] → [B, 64]
```

**直观含义：**
- 模型为 60 个交易日中的每一天学习一个"重要性权重"
- 例如可能学到：最近 3 天权重高 (短期动量强)、20 天前的异常波动权重也高 (均值回归信号)、中间平稳区间权重低
- 权重分布因样本而异，不同股票/不同日期有不同的关注模式

**与其他 pooling 方式对比：**

| 方式 | 信息来源 | 参数量 | E03/E04 结果 |
|------|---------|--------|-------------|
| CLS (`[:,0]`) | 仅 1 个向量 | 64 | RankIC=0.0171 |
| LastStep (`[:,-1]`) | 仅最后 1 天 | 0 | — |
| **Attention (E06)** | **全部 60 步加权** | **2,113** | **RankIC=0.0218 ✅** |

---

### 6. Context Norm — 上下文归一化

| 项目 | 值 |
|------|-----|
| 结构 | `LayerNorm(64)` |
| 参数量 | **128** (0.2%) |
| 作用 | 在进入预测头之前，将聚合后的向量标准化 |

**为什么需要这一层：**
- AttentionPooling 的输出来自 encoder 各步的加权混合，其数值范围可能与原始特征空间不一致
- LayerNorm 保证后续 PredictionHead 接收到的输入具有稳定的统计特性 (均值≈0, 方差≈1)
- 相当于一个 "稳定器"，让 head 训练更稳健

---

### 7. PredictionHead — 预测头

| 项目 | 值 |
|------|-----|
| 类名 | `PredictionHead` (`src/models/base.py`) |
| 结构 | `Linear(64→64)` → `GELU` → `Dropout(0.3)` → `Linear(64→1)` → `squeeze(-1)` |
| 参数量 | **4,225** (5.4%) |
| 输入/输出 | `[B, 64]` → `[B]` |

**参数分解：**

```
net.0 (Linear up):   weight [64, 64]   4,096   (96.9%)
                     bias      [64]       64
net.1 (GELU):               -          0
net.2 (Dropout 0.3):        -          0
net.3 (Linear down): weight  [1, 64]      64
                      bias       [1]       1
                                       ───────
总计:                                  4,225
```

**作用：**
- 将 64 维上下文向量压缩为标量 `pred_score` — 即该股票在未来 5 日相对创业板指的超额收益排名预估值
- 中间隐藏层 (64→64) + GELU 提供非线性变换能力
- 高 dropout (0.3) 强力防止过拟合：这个头是最接近输出的层，最容易记忆训练集

---

## 三、完整参数量汇总表

| # | 模块 | 参数量 | 占比 | 是否可训练 |
|---|------|--------|------|-----------|
| 1 | FeatureProjection (Linear+LN) | 4,160 | 5.4% | Yes |
| 2 | SinusoidalPE | 0 | 0.0% | No (buffer) |
| 3-4 | TransformerEncoder × 2 (Pre-LN) | 67,072 | **86.3%** | Yes |
| 5 | AttentionPooling | 2,113 | 2.7% | Yes |
| 6 | Context LayerNorm | 128 | 0.2% | Yes |
| 7 | PredictionHead (MLP) | 4,225 | 5.4% | Yes |
| | **总计** | **77,698** | **100%** | |

### Encoder 内部细分 (67,072)

| 子组件 | 每层数量 | 2 层合计 | 占 Encoder |
|--------|---------|----------|-----------|
| Self-Attention (W_Q,W_K,W_V,W_O) | 16,640 | 33,280 | 49.6% |
| Feed Forward Network | 16,576 | 33,152 | 49.4% |
| LayerNorm (每层 2个) | 256 | 512 | 0.8% |
| Final LN (norm_first) | — | 128 | 0.2% |

---

## 四、信息流与张量形状追踪

以 E06 的 l60 数据为例，batch_size=B：

```
步骤   操作                          形状             说明
─────────────────────────────────────────────────────────────────
①     原始输入 x                     [B, 60, 62]     62维lag1特征, 60天窗口
│
②     input_proj: Linear+LN+Drop     [B, 60, 64]     特征投影到统一维度
│
③     pos_encoder: x + sin(pe)       [B, 60, 64]     注入绝对位置信息
│                                                (无需CLS token!)
④     encoder layer 1 (Pre-LN):
  │    ├─ norm1                       [B, 60, 64]
  │    ├─ MHA(4头, d_k=16)            [B, 60, 64]     跨时间步attention
  │    │   └─ attn_map: [B,4,60,60]   每头60×60注意力矩阵
  │    ├─ residual add               [B, 60, 64]
  │    ├─ norm2                       [B, 60, 64]
  │    ├─ FFN(64→128→64)             [B, 60, 64]     非线性变换
  │    └─ residual add               [B, 60, 64]
│
⑤     encoder layer 2 (同上结构)      [B, 60, 64]     更深层的时序交互
│
⑥     final LayerNorm                [B, 60, 64]
│
⑦     AttentionPooling:
  │    ├─ score_net: [B,60,64]→[B,60]  给每天打分
  │    ├─ softmax → weights           [B, 60]        归一化权重
  │    └─ Σ(w_t * z_t)               [B, 64]         加权聚合
│
⑧     context_norm: LN               [B, 64]         输出稳定化
│
⑨     PredictionHead:
  │    ├─ Linear(64→64) + GELU        [B, 64]
  │    ├─ Dropout(0.3)                [B, 64]
  │    ├─ Linear(64→1)                [B, 1]
  │    └─ squeeze(-1)                 [B]            最终pred_score
│
▼
      output: pred_score              [B]             用于横截面排序
```

---

## 五、三种 Pooling 方式的架构差异

### A. CLS Pooling (E03)

```
[B,60,62] → Proj → PE → cat(CLS) → [B,61,64] → Enc×2 → 取[:,0] → Head
                                        ↑
                              CLS token [B,1,64] 拼在位置0
                              通过 attention 从60天吸收信息
```
- 参数: CLS 向量 64 + 无额外模块
- 问题: 只取 1 个位置的输出，信息压缩比 61:1

### B. Last Step Pooling (E05)

```
[B,60,62] → Proj → PE → [B,60,64] → Enc×2 → 取[:,-1] → Head
                                              ↑
                                    只取最后一个时间步
```
- 参数: 0 (无额外模块)
- 问题: 完全忽略前 59 天的历史

### C. Attention Pooling (E04/E06) ★ 当前最佳

```
[B,60,62] → Proj → PE → [B,60,64] → Enc×2 → 加权聚合全部60步 → Head
                                              ↑
                              MLP 打分 → softmax → 加权求和
                              每个时间步都参与最终决策
```
- 参数: 2,113 (AttentionPooling 网络)
- 优势: 自适应学习哪些历史时刻更重要，全量利用信息

---

## 六、超参数配置速查 (E06)

```yaml
model:
  name: transformer_encoder
  num_features: 62
  lookback: 60
  d_model: 64
  input_dropout: 0.1
  num_encoder_layers: 2
  num_heads: 4              # d_model/H = 16 per-head
  dim_feedforward: 128      # FFN 中间维度 = 2*d_model
  attn_dropout: 0.1
  ff_dropout: 0.1
  activation: gelu
  norm_first: true          # Pre-LN
  positional_encoding: sinusoidal
  pooling: attention        # ★ E06 关键改动
  head_hidden_dim: 64
  head_dropout: 0.3

training:
  optimizer: adamw
  learning_rate: 3e-4
  weight_decay: 1e-3         # Transformer 推荐 > RNN
  loss_fn: mse_ic            # MSE + PearsonIC(alpha=0.3)
  scheduler: warmup_cosine   # 10% warmup + cosine decay
  batch_size: 256
  batch_mode: date           # 按日组织 batch 计算 IC
  max_grad_norm: 1.0
  early_stop_patience: 8
  early_stop_metric: rank_ic_mean
```

---

## 七、设计决策总结

| 设计点 | 选择 | 理由 |
|--------|------|------|
| d_model=64 | 小而紧凑 | 205K 样本的中等规模数据集，避免过拟合 |
| L=2 层 | 浅层 | 超参少，训练快；深层需更多数据 |
| H=4 头 | 标准 | d_k=16，每头有足够表达能力 |
| d_ff=128 (2x) | 标准 FFN 比例 | 平衡容量与参数效率 |
| Pre-LN | 稳定优先 | 梯度传播好，允许更高 LR |
| Sinusoidal PE | 泛化优先 | 不增加参数，避免小数据集过拟合 |
| Attention Pooling | 信息优先 | 比 CLS/LastStep 利用更多历史信息 |
| GELU | 平滑激活 | 比 ReLU 更适合 Transformer |
| AdamW + WD=1e-3 | 解耦衰减 | 权重衰减独立于梯度更新 |
| WarmupCosine | 先热身后衰减 | 避免 early stage 不稳定振荡 |
