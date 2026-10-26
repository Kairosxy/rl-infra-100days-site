# Day 02 图解 · 前向为什么会占住显存

![Transformer 前向、反向与显存账本](assets/transformer_memory.svg)

## 图中需要读懂的关系

前向阶段不仅输出最终预测，还会产生反向传播需要的中间值。反向阶段利用这些值计算参数梯度；如果某些值没有保存，就必须由算子或 Checkpointing 机制重新计算。

Attention 的显式 Score 张量形状为 [B, heads, L, L]，所以元素数量随序列长度平方增长。FlashAttention 的重要变化是减少对完整中间矩阵的物化，而不是把标准全 Attention 的计算复杂度变成线性。

## 排障时先问

1. OOM 发生在模型初始化、Forward、Backward 还是多步运行后？
2. 主要占用来自参数/优化器状态，还是随序列长度变化的 Activation？
3. 当前 Attention Backend 是否真的避免了完整 Score 矩阵？
4. Checkpointing 减少的内存是否正好是当前瓶颈？
