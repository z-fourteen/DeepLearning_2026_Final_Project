# Model Baseline Evolution / 模型基线演进

## English

This document summarizes model evolution for report writing. It does not replace the root `README.md` reproduction path.

### Initial Baselines

The project initially considered:

- LightGBM regression baseline;
- Ridge or linear baseline;
- GRU sequence baseline;
- Transformer encoder candidate;
- LSTM as a backlog option.

The common goal was to produce a daily cross-sectional ranking score and export it through `predictions.parquet`.

### Why GRU Became The Primary Model

GRU became the primary sequence baseline because it offered:

- enough temporal capacity for daily sequence features;
- fewer parameters and lower training cost than Transformer or LSTM alternatives;
- easier debugging under date-batched evaluation;
- ranking evidence strong enough to justify downstream execution tests.

### Legacy Full62 GRU

The canonical old score model was:

```text
gru_l20_mse_ic_leaky_head_slope_0005
```

It introduced:

- a LeakyReLU head with negative slope `0.005`;
- IC and rank IC tracking;
- prediction export compatibility for backtests and audits.

This model was useful as a research signal, but its full62 input mixed alpha and exposure-like fields.

### Strict Mask And K/Keep Overlay

Strict tradable masks and turnover-buffer overlays improved realism. The K20 keep=2x branch became a historical research mainline for a short period.

T+1 fill simulation showed that:

- validation remained weak;
- desired turnover and filled turnover could diverge;
- test-period performance was not sufficient evidence for promotion;
- Top-K choices were vulnerable to post-hoc selection.

### Clean Alpha-Only And Residual-Style Models

The `alpha_only` clean dataset removed raw style and control fields from model input. It improved interpretability but reduced signal support.

The final model family added 5 residual-style features to 13 alpha features:

```text
lookback: 60
num_features: 18
```

### Final Model

```text
feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
best_metric: 0.05160820540933453
```

The final model is a ranking model, not a calibrated return forecast.

## 中文

本文总结模型路线演进，用于报告写作，不替代根目录 `README.md` 的复现流程。

### 初始 Baseline

项目最初考虑：

- LightGBM 回归 baseline；
- Ridge 或线性 baseline；
- GRU 序列 baseline；
- Transformer encoder 候选；
- LSTM 作为 backlog。

共同目标是生成日度横截面排序分数，并通过 `predictions.parquet` 输出。

### GRU 成为主模型的原因

GRU 被提升为主序列 baseline，因为它具备：

- 足够表达日频序列特征的时间依赖；
- 相比 Transformer 或 LSTM 参数更少、训练成本更低；
- 在 date-batched evaluation 下更容易调试；
- 排序证据足以支持后续执行测试。

### Legacy Full62 GRU

旧版 canonical score model 为：

```text
gru_l20_mse_ic_leaky_head_slope_0005
```

它引入：

- negative slope 为 `0.005` 的 LeakyReLU head；
- IC 和 Rank IC 跟踪；
- 与回测和审计兼容的预测输出。

该模型有研究信号价值，但 full62 输入混合了 alpha 与暴露类字段。

### Strict Mask 与 K/Keep Overlay

strict tradable mask 和换手缓冲 overlay 提高了执行现实性。K20 keep=2x 曾短暂成为历史研究主线。

T+1 成交仿真显示：

- validation 仍然偏弱；
- desired turnover 与 filled turnover 可能明显分离；
- test 期表现不足以作为晋级证据；
- Top-K 选择容易受到事后选择影响。

### Clean Alpha-Only 与 Residual-Style 模型

`alpha_only` clean dataset 移除了原始风格和控制字段，提高了解释性，但信号支撑变薄。

最终模型族在 13 个 alpha 特征基础上加入 5 个 residual-style 特征：

```text
lookback: 60
num_features: 18
```

### 最终模型

```text
feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean
checkpoint: epoch 12
selection metric: checkpoint_score
best_metric: 0.05160820540933453
```

最终模型应解释为排序模型，而不是收益幅度校准模型。
