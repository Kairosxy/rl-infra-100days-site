# RL Infra 能力地图：按可证实的能力分层

把「听过一个术语」和「能在代码/论文中真正运用」分开。下面是课程覆盖的**能力域**，不是自动打分表。

| 能力 | 初步理解（随用随学） | 必须深挖的机制 | 能作为证据的实验 |
|---|---|---|---|
| 策略优化 | Return、Reward、Action/State 的定义 | Score Function、Baseline、PPO Ratio/Clip、GRPO 组内标准化、GAE、GSPO/DAPO 的目标差异 | 精确期望对照采样梯度，Clip 与零方差组反例 |
| Token 训练 | Tokenizer、Prompt/Response、Padding | Teacher Forcing Shift、Action Mask、Reduction、Old/Current Logprob | 同 Shape 错位反例和 Token/Sequence 权重对照 |
| Learner 实现 | Autograd/Optimizer 常见接口 | 梯度累积等价条件、FP8 Scale、Activation 生命周期和混精错误 | 小张量梯度、NaN 注入、显存生命周期 |
| GPU 计算 | GPU/SM/HBM 等基础名词 | Roofline、GEMM、FlashAttention、重计算、CUDA Graph | 真实硬件 Profiling 或明确假设的 IO 模型 |
| 分布式训练 | Rank、ProcessGroup、Collective 名称 | TP/PP/EP/CP、FSDP2 与 NCCL 代价、Overlap、Reshard | 形状/字节模型和版本一致的微型分布式测试 |
| LLM Serving | TTFT、TPOT、Batch 的定义 | KV Block/Prefix Cache、调度长尾、投机解码的分布一致性 | 输入输出长度固定的基准与延迟分位数 |
| Agentic RL | Tool Call、Verifier、Sandbox 接口 | Action/Observation 边界、Group 选择偏差、背压与环境故障 | 多轮 Token Mask、Reward ID Join 与错误输入 |
| 异步 RL | AsyncIO、队列、Barrier 的概念 | Partial Resume 的 RNG/KV/Policy Version、Staleness、Off-policy Correction、Drain 语义 | 接受率时间线与同期同步基线对照 |
| OPD | Teacher/Student 名词 | 全词表分布、KL 方向、Teacher Offload、与 GRPO 联合的梯度权重 | Toy KL 梯度、带版本和 Mask 的对照 |
| 工程诊断 | 常见错误类型 | 首个坏 Tensor、NCCL Hang、Checkpoint Recovery、资源调度与成本 | 故障注入、Trace 与不变量断言 |

每项能力不靠机械笔记数量验收。若没有 GPU，先做手算与 CPU 可运行版本，并明确尚缺硬件实测；已有 VERL、SGLang、Megatron、Ray 工程经验可作为反例和假设来源，但不能直接当作学习成果。
