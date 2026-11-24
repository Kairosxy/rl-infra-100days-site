# RL Infra 100 天 · 论文与官方资料索引

按课程主题维护；每次新增条目时标记它支持的具体问题。论文中的结果应结合原文实验设置理解，不能直接视为当前集群上的性能承诺。

## Policy Gradient / PPO / GRPO

1. **Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning.** REINFORCE / score-function estimator 的经典来源。https://doi.org/10.1007/BF00992696
2. **Sutton et al. (1999), Policy Gradient Methods for Reinforcement Learning with Function Approximation.** Policy Gradient Theorem 与函数逼近。https://papers.nips.cc/paper_files/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
3. **Schulman et al. (2015), Trust Region Policy Optimization.** 策略更新约束与 surrogate objective。https://arxiv.org/abs/1502.05477
4. **Schulman et al. (2017), Proximal Policy Optimization Algorithms.** Clipped surrogate objective，Day 06。https://arxiv.org/abs/1707.06347
5. **DeepSeek-AI et al. (2025), DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** 大模型推理强化学习实践背景。https://arxiv.org/abs/2501.12948

## RL 训练系统与数据编排

1. **Zhou et al. (2024), HybridFlow: A Flexible and Efficient RLHF Framework.** 用于理解 RL 训练中的计算阶段编排和资源复用。https://arxiv.org/abs/2409.19256
2. **VERL 官方实现（固定版本）**：Day 04 对照的 `agg_loss` 位于 `verl/trainer/ppo/core_algos.py`，commit `174888fcf4b65dea032566a02fd16c5f76fb5b5b`。用于比较 Token Mean 与 Sequence Mean 的归一化差异。https://github.com/verl-project/verl/blob/174888fcf4b65dea032566a02fd16c5f76fb5b5b/verl/trainer/ppo/core_algos.py#L1140
3. **VERL repository.** 其他训练流程需固定具体 branch/commit 后再做代码结论。https://github.com/verl-project/verl

## GPU 计算与 Attention

1. **Dao et al. (2022), FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** Attention 的 HBM / SRAM IO 模型。https://arxiv.org/abs/2205.14135
2. **Dao (2023), FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.** 并行划分和 GPU 利用率。https://arxiv.org/abs/2307.08691
3. **NVIDIA Triton documentation.** Kernel 编程与性能分析入口。https://triton-lang.org/main/index.html
4. **PyTorch SDPA documentation.** API、Backend 选择与约束。https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

## 分布式训练与系统

1. **Narayanan et al. (2021), Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.** TP/PP 训练系统与并行策略。https://arxiv.org/abs/2104.04473
2. **Rajbhandari et al. (2020), ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.** Optimizer / Gradient / Parameter 分片。https://arxiv.org/abs/1910.02054
3. **PyTorch Distributed documentation.** DDP、Collectives 与 FSDP 的官方实现文档。https://docs.pytorch.org/docs/stable/distributed.html
4. **VERL repository.** RL 训练框架实现入口；阅读前确认分支/commit。https://github.com/verl-project/verl
5. **SGLang repository.** 推理与 Rollout 服务实现入口；具体行为以对应版本为准。https://github.com/sgl-project/sglang

## 使用规则

- 每节课选 2–5 篇与当天问题直接相关的论文/官方文档，不把阅读列表当装饰。
- 论文摘要只用于定位；涉及公式、实验数字或结论时，回到论文正文和附录核对。
- 记录论文的具体用途：理论推导、实现设计、实验对照或局限。
- 如果无法访问全文或验证链接，标注待核验，不臆造结论。
