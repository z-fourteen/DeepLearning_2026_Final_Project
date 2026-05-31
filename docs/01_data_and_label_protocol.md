# 数据与标签协议

标准标签和执行标签位于 `data/mart/labels/`。

clean pipeline 将三个概念明确分离：

- 用于监督式 GRU 训练的模型目标标签
- 用于 tensor 构建的严格可交易 mask
- 用于真实成交仿真的 T+1 执行标签

关键工程规则是：模型输入必须满足 point-in-time 要求；而交易可行性在预测之后，通过 sidecar labels 和 T+1 成交仿真器进行评估。

主切分合同是 `configs/data/splits.yaml`。它采用 purged walk-forward 协议：标签 horizon 为 5 个交易日，purge 为 5 天，embargo 为 20 个交易日。当前生产折在配置中显式声明，因此重新生成的 tensors 会在 manifest 中记录所选 fold。
