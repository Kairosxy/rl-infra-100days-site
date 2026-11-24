# Attention、位置编码与长上下文

长上下文模型有三个相对独立的问题：位置如何编码、历史信息如何交互、计算与缓存如何保存。MHA/GQA 改变 Head 组织，RoPE 改变位置表示，FlashAttention 改变实现的 IO；不能将它们写成单线替代关系。

## KV 表示与共享

假设 32 层，32 个 Query Head，每 Head 128 维，BF16 KV，单请求长度 8192，不计其他状态和分片。标准缓存载荷为

$$
M_{\rm KV}=2LTH_{\rm kv}d_{\rm head}b.
$$

| 模型组织 | KV Head | 理论 KV 载荷 |
|---|---:|---:|
| MHA | 32 | 4 GiB |
| GQA | 8 | 1 GiB |
| MQA | 1 | 128 MiB |

在同样 Query Head 数下，共享 KV 不意味着 Query 的注意力打分数按相同倍数减少。MLA 则利用压缩的 Latent 表示重构或等价计算 Key/Value 的相关部分；它不等同于把 KV Head 数改成 1，实际缓存还需计入位置分量等模型专有状态。DeepSeek-V3 报告给出了该类架构的具体实例。

## 位置与连接范围

RoPE 在成对维度上旋转 Q、K。例如二维向量 [1,0] 旋转 90° 后是 [0,1]。若两个向量均旋转相同角度，其内积保持不变；使用不同位置对应的旋转后，内积带有相对位置依赖。扩展 Position ID 的最大值并不自动保证长距离推理质量，训练长度、频率缩放及模型训练策略都重要。

全局 Causal Attention 有约 T(T+1)/2 对可见位置；滑窗长度 W 时，大 T 下约为 TW。T=8192、W=512 时，单层连接数约从 3356 万降到 406 万（考虑最初窗口未填满）。但如果部分层仍是全局 Attention，或者有全局 Token，计算与缓存不能统一按 W 估算。

Linear Attention 类方法改变注意力的计算形式；SSM 使用递推状态表达序列动态；混合架构可以组合不同层。这些方法并不自动与标准 Softmax Attention 等价。FlashAttention 通过分块和在线归一化减少 HBM 中间读写，在标准精确 Attention 中不会改变连接的数学定义，详见 [算子优化](../optimization/kernel-optimization.md)。

## 工程推导

先从模型配置确认每层 Attention 类型、Query/KV Head、Head Dim、位置编码与窗口策略，再按实际 TP/CP 分片统计每 Rank 保存的缓存。对于 GQA，如果 TP 超过 KV Head 数，部分实现需要复制 KV Head 或改变布局，因此全局 KV 预算不能无条件除以 TP。

参考：[Transformer](https://arxiv.org/abs/1706.03762)、[GQA](https://arxiv.org/abs/2305.13245)、[RoFormer](https://arxiv.org/abs/2104.09864)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[FlashAttention](https://arxiv.org/abs/2205.14135)。
