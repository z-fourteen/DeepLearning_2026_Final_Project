# 项目总览

本项目实现了一套面向创业板股票的、基于 GRU 的时间序列选股流程。当前仓库已经围绕 `clean_dataset` 主线完成整理和净化。

最终主线如下：

1. 从 `advanced_sequence_clean_v1` 构建 point-in-time 的 clean tensors。
2. 在严格可交易样本上训练 GRU 序列模型。
3. 使用 T+1 开盘成交仿真评估预测结果。
4. 在组合部署前接入容量约束审计和 residual-alpha 审计。

历史 `full62` 实验已冻结在 `legacy/legacy_full62_v1/` 下。它们仍可作为研究证据使用，但不再作为生产入口。
