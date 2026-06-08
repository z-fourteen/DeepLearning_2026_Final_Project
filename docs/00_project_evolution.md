# Project Evolution / 项目演进

## English

This document records how the project evolved from an initial baseline stock-prediction task into the final clean-dataset GRU research loop. It is report preparation material, not an operating guide. Use the root `README.md` for reproduction commands.

### Stage 0: Task Definition

The original task was to predict future ChiNext stock returns from daily A-share data and evaluate whether a deep sequence model could produce useful cross-sectional ranking signals.

The first engineering target was broad:

- build a ChiNext historical dataset;
- define train, validation, and test periods;
- construct fixed feature pools;
- train baseline models such as LightGBM, Ridge, GRU, and Transformer candidates;
- export a common `predictions.parquet` contract for Top-K and execution evaluation.

The common prediction schema became a durable interface:

```text
trade_date
ts_code
pred_score
label_rel_return
split
model_name
```

### Stage 1: Fixed Feature Pool

The first complete modeling handoff used the `advanced_sequence_fixed` feature set. It supported flat daily models and sequence models with lookback windows of 20 and 60 trading days.

This stage established a working end-to-end pipeline, but it was later downgraded because the full feature set mixed alpha candidates with style, liquidity, volatility, tradability, and risk-control fields.

### Stage 2: GRU Baseline

The first strong neural baseline was:

```text
gru_l20_mse_ic_leaky_head_slope_0005
```

Key decisions:

- use GRU as the primary sequence baseline;
- downgrade LSTM to backlog;
- keep Transformer as a later ablation candidate;
- use a LeakyReLU head with negative slope `0.005` to reduce output-head saturation;
- evaluate IC and rank IC, not MSE alone.

The old GRU baseline produced useful ranking evidence, but it was not enough to claim a tradable strategy.

### Stage 3: Execution Reality Check

The next phase added stricter execution thinking:

- strict tradable-sample masks;
- T+1 fill simulation;
- participation caps;
- transaction cost and slippage;
- turnover-buffer variants;
- benchmark and executable-universe comparisons.

The key lesson was that model ranking quality and tradable portfolio quality are separate questions.

### Stage 4: Production-Readiness Audit

The production-readiness review identified several risks:

- repeated test-set inspection created selection risk;
- validation/test inversion was a red flag;
- old full62 features might rely on style, liquidity, size, volatility, industry, or turnover exposures;
- proxy close-to-close labels were insufficient for production-like execution evaluation.

This reframed the research question:

```text
not "can the old full62 model predict in test?"
but "can residual alpha survive after point-in-time, style, liquidity, and execution controls?"
```

### Stage 5: Feature Governance

Features were split into explicit roles:

| Role | Meaning |
| --- | --- |
| alpha | Candidate model inputs. |
| risk_control | Style, size, liquidity, volatility, valuation, and beta controls. |
| tradability_control | Limit-state, suspension, volume, and execution-feasibility controls. |
| exclude | Weak, unstable, redundant, or low-justification features. |

This led to:

```text
configs/features/advanced_sequence_clean_v1.yaml
```

### Stage 6: Clean Dataset

The clean dataset builder introduced:

- `alpha_only`: 13 pruned alpha features;
- `alpha_plus_residual_style`: 13 alpha features plus 5 residual-style features;
- sidecar parquet files for controls and masks;
- strict tradable filtering;
- purged walk-forward split metadata;
- clear separation between model labels and execution labels.

### Stage 7: Final Mainline

The final submitted mainline is:

```text
feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
optimizer: risk_control=none, k=10, style_penalty=0.1, turnover_penalty=0.0, min_invested=0.8
```

The final system is best described as:

```text
clean data -> model selection -> prediction diagnostics -> T+1 execution -> optimizer -> audit evidence
```

It should not be described as a production-ready stable alpha strategy.

## 中文

本文记录项目如何从最初的股票收益预测 baseline，演进为最终的 clean-dataset GRU 研究闭环。本文用于报告准备，不是运行指南；复现命令请以根目录 `README.md` 为准。

### 阶段 0：任务定义

最初目标是基于 A 股日频数据预测创业板股票未来收益，并评估深度序列模型是否能产生有效的横截面排序信号。

第一阶段工程目标包括：

- 构建创业板历史数据集；
- 定义 train、validation、test 区间；
- 构建固定特征池；
- 训练 LightGBM、Ridge、GRU、Transformer 等候选模型；
- 统一输出 `predictions.parquet`，供 Top-K 和执行评估使用。

统一预测 schema 成为后续模块之间的重要接口：

```text
trade_date
ts_code
pred_score
label_rel_return
split
model_name
```

### 阶段 1：固定特征池

第一版完整建模交接使用 `advanced_sequence_fixed` 特征集，同时支持平铺日频模型和 lookback=20/60 的序列模型。

这一阶段跑通了端到端链路，但后续被降级，因为 full feature set 混合了 alpha 候选、风格、流动性、波动率、可交易性和风控字段。

### 阶段 2：GRU Baseline

第一条较强神经网络 baseline 是：

```text
gru_l20_mse_ic_leaky_head_slope_0005
```

关键决策：

- 使用 GRU 作为主序列模型；
- LSTM 降级为 backlog；
- Transformer 保留为后续消融；
- 使用 negative slope 为 `0.005` 的 LeakyReLU head，缓解输出头饱和；
- 评估 IC 和 Rank IC，而不是只看 MSE。

旧 GRU baseline 有模型级排序价值，但不足以证明可交易策略稳定成立。

### 阶段 3：执行现实检验

随后加入更严格的执行假设：

- strict tradable mask；
- T+1 成交仿真；
- participation cap；
- 交易成本和滑点；
- 换手缓冲；
- benchmark 与 executable universe 对比。

核心教训是：模型排序能力和可交易组合表现是两个不同问题。

### 阶段 4：生产就绪审计

生产就绪审计识别出几个关键风险：

- test set 被多轮查看，存在选择风险；
- validation/test 表现反转；
- old full62 特征可能依赖风格、流动性、规模、波动率、行业或换手暴露；
- close-to-close proxy 标签不足以支持生产式执行评估。

研究问题因此被重塑为：

```text
不是“old full62 能否在 test 上预测”，
而是“point-in-time、风格、流动性和执行控制后，残差 alpha 是否仍然存在”。
```

### 阶段 5：特征治理

特征被拆分为明确角色：

| 角色 | 含义 |
| --- | --- |
| alpha | 模型输入候选特征。 |
| risk_control | 风格、规模、流动性、波动率、估值和 beta 控制。 |
| tradability_control | 涨跌停、停牌、成交量和可执行性控制。 |
| exclude | 弱、不稳定、冗余或经济含义不足的特征。 |

最终形成：

```text
configs/features/advanced_sequence_clean_v1.yaml
```

### 阶段 6：Clean Dataset

clean dataset builder 引入：

- `alpha_only`：13 个剪枝 alpha 特征；
- `alpha_plus_residual_style`：13 个 alpha 特征 + 5 个 residual-style 特征；
- controls 和 masks 的 sidecar parquet；
- strict tradable filtering；
- purged walk-forward split 元数据；
- 模型标签和执行标签的清晰分离。

### 阶段 7：最终主线

最终提交主线：

```text
feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
optimizer: risk_control=none, k=10, style_penalty=0.1, turnover_penalty=0.0, min_invested=0.8
```

最终系统应描述为：

```text
clean data -> model selection -> prediction diagnostics -> T+1 execution -> optimizer -> audit evidence
```

它是可复现研究闭环，不应描述为生产就绪的稳定 alpha 策略。
