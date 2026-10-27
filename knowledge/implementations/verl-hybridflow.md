# verl / HybridFlow：数据流与执行引擎怎么解耦？

verl 的 HybridFlow 将 RL 的数据流/控制流与具体 Worker、不同并行策略的执行引擎组合。这个架构的价值不在于“有多少 Actor”，而是能让不同阶段使用适合自己的训练、推理和并行部署，并允许相关资源通过明确边界协作。

## 关心的数据与状态

一个完整的训练周期至少涉及 Prompt、Response、Reward、Advantage、Old/Current Logprob、Policy Version、更新后的模型权重。最容易出问题的不是字段存在与否，而是**谁生成它、什么时候有效、是否能够跨版本重用**。

- Rollout 端负责生成，Teacher Forcing 的训练侧要对固定 Token 重算可微 Logprob。
- Group 内需要收齐有效回答才能按预定估计器计算相对优势；不同 Group 的额外等待是可优化的。
- Weight Sync 与参数版本不只是性能成本，还是 Off-policy 语义的边界。

## 适合与不适合

当需要组合 Megatron/FSDP 训练、SGLang/vLLM 生成、不同 Reward/算法或试验新的 DataSampler 时，可组合的 Worker/数据流能够减少侵入业务层的改动。但这会增加数据契约、同步/异步恢复和跨 Engine Probability 对齐的复杂度。

深入代码应从本地正在使用的 verl Commit 开始：确定 Trainer 入口、Rollout Worker、DataProto/队列，以及 Weight Update 调用链。**不同 fork 的类名和开关可能不同，不能把上游最新 README 当成本地代码事实**。

参考：[verl GitHub](https://github.com/verl-project/verl) · [HybridFlow 论文](https://arxiv.org/abs/2409.19256)。