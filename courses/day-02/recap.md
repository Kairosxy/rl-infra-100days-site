# Day 02 复习卡 · Transformer 显存

- 参数显存只是一部分；训练还要考虑梯度、优化器状态、激活和临时缓冲区。
- Linear 的权重梯度需要前向输入，因此 Autograd 必须保存或重建反向所需信息。
- 显式 Attention 矩阵元素数随序列长度平方增长；融合 Attention 可以避免完整矩阵长期物化。
- Activation Checkpointing 以额外计算换取激活显存，不会直接消除 Adam 状态。
- OOM 排查先定位发生阶段，再确认主要内存来源。

自测：解释 1B BF16 参数为什么约 2 GB，但训练状态可能远高于这个数字；解释 640 到 8320 Token 的显式 Attention 矩阵为什么是 169 倍。
