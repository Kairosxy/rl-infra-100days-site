# Day 02 面试问答

## 题 1：1B 参数模型为什么不只需要 2 GB 显存？（10 分）

**问题**：假设参数为 BF16，估算训练状态并说明还缺哪些项。

**评分标准**
- 参数大小按 2 Bytes/参数估算：2 分
- 梯度大小与 Dtype 假设：2 分
- Adam 一阶、二阶状态：2 分
- Activation、Workspace、通信缓冲区：2 分
- 说明分片和 Master Weights 会改变账本：2 分

**追问**：FSDP 能否解决所有 OOM？合格回答应区分参数/梯度/优化器状态分片与激活峰值，指出长序列仍可能因 Activation OOM。

## 题 2：序列长度从 4K 增长到 32K，为什么 Attention 显存可能暴涨？（10 分）

**评分标准**
- 显式 Attention 矩阵为 L×L：3 分
- 长度扩大 8 倍，矩阵元素扩大 64 倍：3 分
- 区分朴素矩阵物化与 FlashAttention：2 分
- 提出检查 Backend 和测量峰值的方法：2 分

**追问**：FlashAttention 是否让 Attention 的计算复杂度变成 O(L)？需要明确回答不是；它主要优化 IO 与中间矩阵存储，标准全 Attention 的算术量仍随序列长度呈二次增长。
