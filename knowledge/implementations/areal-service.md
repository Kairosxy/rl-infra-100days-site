# AReaL：为什么把异步数据生产与训练服务拆开？

AReaL 讨论生成与训练同时推进时，训练不必停下来等待同一批 Rollout 全部完成。它采用相对独立的训练与生成服务，并通过版本与 Off-policyness 约束控制异步代价。

## 机制上最重要的边界

- **生成时版本**：每条轨迹必须能够追溯实际采样的 Policy。
- **消费时版本**：Trainer 可能已经做了若干次更新，因此旧数据不再严格 On-policy。
- **异步预算**：限制能接受多旧的数据；过紧则降低供给，过松可能增加偏差。
- **权重同步**：同步时机影响多长时间内请求仍由旧版本生成。

这是算法与系统联合设计：提前消费完成数据提高资源利用率，但引入数据选择与 Policy Staleness 的新约束。

## 什么时候值得参考

当 Rollout/Tool/Verifier 长尾严重、Learner 因同步 Batch Barrier 大量等待时，可研究独立资源池和流水线。如果 GPU 数很少、网络转移成本高，分池/异步不一定优于 Colocate。

建议用相同的有效 Group 数、质量、Token 分布与 GPU-hours 对照同步和异步；尤其要计入未完成请求的 Drain Cost。

参考：[AReaL 项目](https://github.com/inclusionAI/AReaL) · [异步算法文档](https://areal-ai.io/AReaL/en/algorithms/async.html)。