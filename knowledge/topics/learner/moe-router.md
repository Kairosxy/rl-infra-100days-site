# 同一个 Token，为什么在训练和 Rollout 中进入了不同 Expert？

MoE Router 在选 Top-k Expert 时，很多时候接近一个离散决策边界。如果两个 Expert 的 Router Score 非常接近，那么两套 Engine 即使权重相同，也可能因为 Kernel、精度或归约顺序差异选中不同 Expert。

这会使 Rollout 实际使用的条件分布与训练侧重算的 Logprob 不一致。误差在多个 MoE Layer 中累积时，可能比单层 Logit 的浮点扰动更明显。

## R3 修复的是具体的离散决策

Rollout Routing Replay（R3）记录生成时的 Expert 选择，在训练前向中重放这些索引，使训练侧使用对应 Expert 的路径重新计算概率。NeMo RL 文档明确说明只重放离散选择，Router 的权重与梯度计算仍发生在当前训练侧；不是冻结整个 Router。

[R3 论文](https://proceedings.mlr.press/v306/ma26aw.html) 报告通过路由对齐降低训推偏差；[NeMo RL Router Replay Guide](https://docs.nvidia.com/nemo/rl/nightly/guides/router-replay.html) 说明了具体支持的训练与生成 Backend、异常 fallback 及验证指标。这里引用的是公开实现文档而非我们自己的训练结果。

## 为什么不能只开一个开关

路由信息必须跟随正确的 Token 穿过 Packing、TransferQueue、Context Parallel 切分和恢复流程。缺失路由需要明确的 Fallback，而不能默默把整批标记成已对齐。R3 对齐 Expert 选择，也不会自动对齐 Weight Version、Temperature 或其他概率变换。

另外，路由重放并非没有取舍。[Router Drift Control 预印本（2026-10）](https://arxiv.org/abs/2610.21682) 提出回放旧路由可能引入另一种梯度偏差，并探索基于路由分歧的样本筛选。它是有价值的研究反例，但不等于已被证明在所有配置下优于 R3。

**最小对照**：固定权重与 Token，记录 Rollout/Training 的 Top-k Expert ID、同位置 Logprob、与 Policy Version；分别开启与关闭 Replay，比较路由一致率和选中 Token Logprob 差异。