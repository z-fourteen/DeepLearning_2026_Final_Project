# 项目总览

本项目实现了一套面向创业板股票的、基于 GRU 的时间序列选股流程。当前仓库已经围绕 `clean_dataset` 主线完成整理和净化。

最终主线已经冻结为：

1. 从 `advanced_sequence_clean_v1` 构建 point-in-time 的 L60 clean alpha-resid-style tensor。
2. 使用 `feature_style_interaction_gru_l60_clean_alpha_resid_style_topk10_wide30_clean`。
3. 模型选择固定为 validation `checkpoint_score`，checkpoint 为 epoch 12。
4. 最终 optimizer 固定为 `risk_control=none, k=10, style_penalty=0.1, turnover_penalty=0.0, min_invested=0.8`。
5. 证据链由训练 metrics、epoch12 optimizer grid、validation attribution、final optimizer wrapper 和审计脚本共同支撑。

冻结配置入口：

```text
configs/portfolio/final_mainline_optimizer.yaml
```

历史 `full62` 实验已冻结在 `legacy/legacy_full62_v1/` 下。它们仍可作为研究证据使用，但不再作为生产入口。
