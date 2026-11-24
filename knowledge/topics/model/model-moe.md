# Dense、MoE 与模型容量

Dense FFN 对每个 Token 使用相同的参数；MoE 通过 Router 给不同 Token 选择部分 Expert。它主要扩大模型容量而不按总专家数等比例增加每 Token 的 FFN 计算，但所有 Expert 参数仍需要存储或从其他设备取得。

## 一个可复算的 Expert 层

设 hidden=4096，SwiGLU 中间维度=2048，每个 Expert 有 gate、up、down 三个无 Bias 矩阵。单 Expert 参数量为 3×4096×2048=25165824，BF16 权重为 48 MiB。8 个 Expert 共 384 MiB，top-2 路由下单 Token 使用两个 Expert，涉及 96 MiB 的逻辑权重集合与约 1.01 亿 FLOPs（仅矩阵乘法）。

这里不表示每个 Token 都从 HBM 实际读取 96 MiB，也不包括 Attention、共享专家、Router 或其他层。Batch 中多个 Token 可能共享读取与 GEMM。

$$
y(x)=\sum_{i\in\mathrm{TopK}(g(x))}w_i(x) E_i(x).
$$

Router 产生选择分数并决定 Expert ID；被选中 Expert 的组合权重按模型实现定义。不是所有 MoE 都在同样的集合上做 Softmax 或使用同一种归一化。Shared Expert 还可提供不依赖 Top-k 的计算路径。

## 路由负载与容量

8 个 Token 各选择 2 个 Expert，总派发数为 16。完全均衡时 8 个 Expert 各接收 2 个 Token；若两个热 Expert 分别收到 8 个 Token，则另六个为空。总派发数不变，最大 Expert 的负载却增加 4 倍。真实瓶颈还取决于 Rank 上聚合的 Expert、Grouped GEMM、Dispatch/Combine 和网络，而非单个 Expert 计数就能精确预测延迟。

辅助负载均衡 Loss、Router Bias 调整、容量限制、Token Drop 等策略解决的是不同约束；不能因为启用 MoE 就默认有 Token Drop。DeepSeek-V3 的模型报告包含具体的负载均衡与 MTP 机制，不能直接套到其他 MoE 模型。

[Expert Parallel](../parallel/ep-token-routing.md) 负责把这些计算放到不同 Rank；[R3](../learner/moe-router.md) 处理同 Token 的离散选择在训练与推理引擎之间不一致。前者是分布式执行机制，后者是正确性机制。

参考：[Switch Transformers](https://arxiv.org/abs/2101.03961)、[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)。
