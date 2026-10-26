# RL Infra 100 天 · 基于问题难度重构的课表

**发布与掌握分离：**2026-10-23 为 Day 01，每天北京时间 09:00 发布下一课；所有已发布课程可以任意跳转。以下 Day 是内容顺序，不是学完上一课才能解锁下一课的门槛。

## 哪些值得占一整天

**深挖：** 影响目标函数、数据分布、系统成本或正确性的机制，至少要求推导/成本模型、反例和可复现的验证。**实作：** 通过代码、单元测试、测量或故障注入才能学会的能力。**研读：** 对经典/新论文的机制、证据和局限做比较。**综合：** 对同一参考系统做一轮可以自证的设计和验证。

不再独立安排「名词解释日」。`trajectory_id`、常规代码命名、AdamW 的字段列表、组件清单这类容易查文档的内容，在正确性实验实际需要它们时顺带解释，通常 10–20 分钟足够。另一方面，不把 PPO、GRPO、GAE、FP8、KV/通信调度、Partial Rollout/Off-policy 这样需要公式、张量和实验的主题挤在一节“综述”中。课程分配资源以需要推理的深度为标准，而不是给每个名词同样篇幅。

## 八条相互咬合的主线

- **A · Day 01–14 策略优化与估计器**：研究问题必须能落到可计算的目标函数、概率分布及失败的训练信号。
- **B · Day 15–26 Learner 内部实现**：理解各类状态的计算生命周期，保证理论梯度与分布式执行一致。
- **C · Day 27–42 GPU 与分布式训练**：每种并行/算子优化都追问通信、显存峰值和真实瓶颈。
- **D · Day 43–56 推理调度与 Rollout**：以 KV、请求调度和行为策略分布一致性为真实约束。
- **E · Day 57–68 Agentic RL 数据生产**：只在算法和环境的契约清楚后谈多轮并发和组采样。
- **F · Day 69–85 异步、Off-policy 与 OPD**：带上策略版本、样本滞后和训练质量再衡量吞吐改善。
- **G · Day 86–94 可靠性与容量规划**：由真实故障证据倒推系统设计，不背云原生术语。
- **H · Day 95–100 研究复现与系统面试**：能独立审查论文声明、代码实现和实验有效性。

## 逐日题目与可验证产物

| Day | 主线 | 深度 | 当天真正要回答的问题 | 核心证据 / 可操作产物 |
|---:|:---:|:---:|---|---|
| 01 | A | 深挖 | 没有监督答案时，Reward 如何改变模型的动作分布？ | 手算 Bernoulli Policy 的期望回报与梯度；互动拨动 p |
| 02 | A | 深挖 | 为什么不可微 Reward 能训练 Policy？Baseline 如何改变方差？ | 从 log-derivative 推导；同一估计的方差曲线 |
| 03 | A | 深挖 | LLM 的一个 Token Logprob 究竟对齐哪个位置？ | Teacher Forcing、Shift、Mask、Token vs Sequence Mean |
| 04 | A | 深挖 | PPO 的 Clip 究竟截断了什么，为什么不保证 KL 安全？ | 正负 Advantage 的分段目标与梯度 |
| 05 | A | 深挖 | 组内排序怎样产生训练信号；为什么全对/全错没有信号？ | GRPO 标准化、Group 缺失和长度偏差 |
| 06 | A | 深挖 | DAPO 在 GRPO 上改了什么，以及为什么这些改动必要？ | Dynamic Sampling、Clip-Higher、Token PG 的消融设计 |
| 07 | A | 深挖 | 一条轨迹提前结束，GAE 的 Bootstrap 应该怎样处理？ | terminal 与 truncation 的手算和 mask |
| 08 | A | 深挖 | 旧策略样本重用时，Importance Ratio 在纠正什么？ | IS 期望、方差爆炸、截断引入偏差 |
| 09 | A | 深挖 | GRPO、GSPO 和 PPO 的 Ratio 粒度如何影响长回答？ | Token/Sequence Ratio 对比，长度变更的反例 |
| 10 | A | 实作 | 能否让最小 Policy 真正完成一次正确的更新？ | CPU Toy Step + 参数梯度/Token Mask 断言 |
| 11 | A | 深挖 | 可验证奖励真能代表推理能力吗？ | Verifier Error、Reward Hacking、训练偏差的反例 |
| 12 | A | 深挖 | KL 与 Entropy 约束的是哪个分布？ | Forward/Reverse KL 与 Reference/Old Policy 区分 |
| 13 | A | 实作 | 为什么固定 Seed 仍可能无法复现 RL 更新？ | Sampling RNG、非确定算子、多 Seed 的对照表 |
| 14 | A | 研读 | DeepSeekMath、R1、DAPO、GSPO 分别解决了哪个问题？ | 从论文主张回推机制、复现缺口和限制 |
| 15 | B | 深挖 | Autograd 图中哪些量应该 Detach，哪些必须可微？ | Old/Current、Advantage/Reward 的梯度路径 |
| 16 | B | 深挖 | 不等长 Micro-batch 为什么会改变梯度？ | 2/2/1 与 Token Reduction 的错误实现 |
| 17 | B | 深挖 | AdamW、梯度裁剪和累积的更新边界在哪里？ | Moment/Step Counter、先裁剪后累积反例 |
| 18 | B | 深挖 | Activation 到底在哪个时刻占着显存？ | Forward/Backward 生命周期、Checkpointing 峰值 |
| 19 | B | 深挖 | FlashAttention 省的是 HBM IO 还是算术量？ | Tiling 与 online softmax 的 IO 成本账 |
| 20 | B | 深挖 | FP8 的 Scale 是如何定义并传播的？ | per-tensor/per-block 量化误差与溢出 |
| 21 | B | 实作 | 同样一组 Tensor，精度变化如何影响 Logprob？ | BF16/FP8 或模拟量化的误差分布 |
| 22 | B | 深挖 | MoE Routing 的不稳定性怎样传到 RL？ | Expert 分布、负载与随机性 |
| 23 | B | 深挖 | Packing 后哪些 Token 才是 RL 的 Action？ | 变长、Position、Attention Mask、Action Mask |
| 24 | B | 深挖 | 同一个 Loss 数字，为什么不同 Reduction 给出不同梯度？ | Token/Sequence/Group 加权的显式矩阵 |
| 25 | B | 实作 | 一次 NaN 如何定位到第一个坏 Tensor？ | 梯度、Loss、Scale、异常样本的检查脚本 |
| 26 | B | 综合 | 能不能重现一轮真实的最小 Learner 更新？ | Toy Learner 正确性、Checkpoint 与恢复 |
| 27 | C | 深挖 | GPU 为什么算力很高，实际 Kernel 却跑不满？ | Roofline 的算术强度与带宽上限 |
| 28 | C | 深挖 | 怎样根据 GEMM 形状判断 Tensor Core 的收益？ | 矩阵尺寸、Tile、Padding 的成本测算 |
| 29 | C | 深挖 | 长上下文 Attention 的时间成本由什么决定？ | Prefill Attention 与 FlashAttention 反例 |
| 30 | C | 深挖 | CUDA Graph 为什么会快，又在哪些场景失效？ | Launch Overhead、静态形状与动态图变体 |
| 31 | C | 实作 | Profiler 里如何区分计算、等待与通信？ | Nsight/Torch Profiler 的可复核证据 |
| 32 | C | 深挖 | All-reduce 的通信量为何不等于 Step Time？ | α–β 模型、Ring/Tree、Overlap |
| 33 | C | 深挖 | DDP Bucketing 与 no_sync 如何改变等待时间？ | Gradient Hook、Bucket 时机与累积 |
| 34 | C | 深挖 | FSDP2/ZeRO 为什么能省显存却仍会 OOM？ | All-gather Peak 与 Shard 生命周期 |
| 35 | C | 深挖 | TP 列并行与行并行在哪一步通信？ | 线性层张量轴和 Collective 推导 |
| 36 | C | 深挖 | 1F1B 如何减少 PP Bubble，又引入什么复杂度？ | Pipeline 迭代调度与显存账 |
| 37 | C | 深挖 | DP/TP/PP 的组合如何影响单 Step 性能？ | 固定模型和硬件下的成本模型 |
| 38 | C | 深挖 | EP 的 All-to-All 成本怎么估？ | Token Routing、Grouped GEMM、负载失衡 |
| 39 | C | 深挖 | Context Parallel 如何处理超长序列？ | Attention 依赖、通信与 KV 分块 |
| 40 | C | 深挖 | NCCL、NVLink、RDMA 性能瓶颈如何区分？ | 通信图、拓扑与可测试假设 |
| 41 | C | 实作 | 分布式 Checkpoint 要保存哪些状态才能真正恢复？ | Optimizer、RNG、Shard 与 Reshard |
| 42 | C | 综合 | 如何为具体模型选择并行拓扑？ | 8–64 卡可解释资源配置与失败边界 |
| 43 | D | 深挖 | Prefill 与 Decode 的瓶颈为什么不同？ | 算术强度、KV 读写与 Batch 敏感性 |
| 44 | D | 深挖 | KV Cache 容量怎么算，何时成为真实瓶颈？ | Layers/Heads/HeadDim/Token 的内存式 |
| 45 | D | 深挖 | PagedAttention 解决了哪种碎片，代价是什么？ | Block 分配、释放与虚拟映射 |
| 46 | D | 深挖 | Continuous Batching 为什么改善吞吐，也会伤 P99？ | Admission、Decode Scheduling 与队首阻塞 |
| 47 | D | 深挖 | Prefix Cache 在什么分布下才能命中？ | Radix Tree、相同前缀与回收策略 |
| 48 | D | 深挖 | Chunked Prefill 如何改变 TTFT/TPOT 的权衡？ | Prefill–Decode 抢占实验 |
| 49 | D | 深挖 | EOS、Stop 与 Sampling 过滤怎样改写轨迹分布？ | 停止条件的 Bias、长度与 Finish Reason |
| 50 | D | 深挖 | 训推两套 Logprob 为什么不一致？ | Sampling Transform、Precision、Tokenization 对照 |
| 51 | D | 深挖 | 投机解码怎样在理论上保持目标分布？ | Acceptance/Correction 的算式与反例 |
| 52 | D | 深挖 | MTP 与 EAGLE 增益由什么决定？ | Draft Cost、Acceptance、Tail Length |
| 53 | D | 深挖 | PD 解耦的 KV 传输什么时候划算？ | Transfer Cost、Prefill/Decode Pool 调度 |
| 54 | D | 深挖 | MoE 推理中 TP/EP 配置何时相互制约？ | 专家并行与通信瓶颈 |
| 55 | D | 深挖 | 吞吐提升是否一定意味着时延下降？ | TTFT/TPOT/P95/P99 与公平负载实验 |
| 56 | D | 综合 | 怎样设计不会自欺的 Serving Benchmark？ | 输入/输出长度分布、Warmup、容量上限 |
| 57 | E | 深挖 | Agent 的哪些 Token 应该参与训练？ | Action/Observation 与多轮 Mask |
| 58 | E | 深挖 | 工具调用失败该怎样影响 Reward 与 Trajectory？ | Tool Schema、重试、结果归属 |
| 59 | E | 深挖 | Sandbox 隔离保证了什么，不保证什么？ | 状态污染、网络与生命周期 |
| 60 | E | 深挖 | 对话模板变更为什么会改变 Logprob？ | Tokenizer、Role、Special Token 的对照 |
| 61 | E | 深挖 | Verifier 出错会把模型推向什么方向？ | False Positive/Negative 与奖励攻击 |
| 62 | E | 深挖 | 过滤失败样本为何会引入 Selection Bias？ | Group Acceptance 与条件分布 |
| 63 | E | 深挖 | AgentLoop 的等待在哪里：Tool、LLM 还是调度器？ | AsyncIO 状态机与超时 |
| 64 | E | 深挖 | 多轮轨迹切开后还能算同一个目标吗？ | Mask、Step Return、状态拼接 |
| 65 | E | 深挖 | Reward Join 和重试如何实现幂等？ | Trajectory Identity、去重和溯源 |
| 66 | E | 深挖 | TransferQueue 背压该传到谁？ | Producer–Consumer 预算与排空顺序 |
| 67 | E | 深挖 | 长尾样本是纯性能问题还是统计问题？ | P95、取消策略与采样偏差 |
| 68 | E | 综合 | 从工具调用到可训练 Group 的最小 Agent Harness | Actor、Verifier、Sampler 的可运行切片 |
| 69 | F | 深挖 | 同步 RL 真正的 Barrier 在哪里？ | 必要依赖与外层 Batch Wait 时间线 |
| 70 | F | 深挖 | 训推 Colocation 能节约什么，却要付什么代价？ | GPU 互斥、Sleep/Wake 与权重转移 |
| 71 | F | 深挖 | 新权重什么时候能对 Rollout 可见？ | Version Publication、原子切换 |
| 72 | F | 深挖 | Partial Rollout 暂停时怎样保存状态？ | KV、RNG、Sampling State、Tool State |
| 73 | F | 深挖 | 同一轨迹跨多个 Policy Version 会发生什么？ | Token 粒度来源与概率契约 |
| 74 | F | 深挖 | Staleness 究竟该按时间、Step 还是 Version 定义？ | 不同阈值定义的反例 |
| 75 | F | 深挖 | Off-policy Correction 能修复所有旧样本吗？ | Ratio 截断、Variance、Support |
| 76 | F | 实作 | Rollout/Train Ratio 漂移怎么做可观测？ | Logprob、Mask、Distribution 与 Version |
| 77 | F | 深挖 | 接收率驱动的补投为什么可能过度生成？ | E[Accepted]、Inflight、校准误差 |
| 78 | F | 深挖 | 达到 Target 与全部在途请求排空有什么不同？ | Abort/Drain 的资源和样本语义 |
| 79 | F | 深挖 | 异步系统重启后，哪些结果允许重复消费？ | Exactly-once 假象、Checkpoint/Replay |
| 80 | F | 综合 | 同步与异步 RL 应怎样公平比较？ | Accepted/Quality/Staleness/Cost 的实验 |
| 81 | F | 深挖 | OPD 的 Teacher 信号如何影响 Student Policy？ | Logits、全词表分布与梯度路径 |
| 82 | F | 深挖 | Forward/Reverse KL 为何诱导不同的行为？ | Toy 分布和 Token-level Distillation |
| 83 | F | 深挖 | Teacher Offload 何时收益大于传输成本？ | 显存、带宽、TP/EP 与服务并发 |
| 84 | F | 深挖 | GRPO Advantage 和 OPD KL 在同一 Loss 中如何权衡？ | 梯度竞争、Mask 与 Scale |
| 85 | F | 研读 | AReaL、DORA 等异步方案的假设和权衡 | Version、Overlap、Correctness 主张比较 |
| 86 | G | 深挖 | Ray Actor / Object Store 的生命周期怎么影响调度？ | Object Ref、OOM、Actor Restart |
| 87 | G | 深挖 | K8s Placement 为什么改变 TP/EP 性能？ | Topology、GPU Scheduling 与 NUMA |
| 88 | G | 深挖 | RDMA/RPC/Zero-copy 哪一段真正在耗时？ | Latency Breakdown 与内存复制 |
| 89 | G | 深挖 | 多 Policy Version 共存需要多少显存？ | 权重复制、路由与一致性代价 |
| 90 | G | 深挖 | Lease/Fencing 为什么比“重试一下”可靠？ | 重复消费与僵尸任务 |
| 91 | G | 深挖 | Metrics、Traces 与 Logs 怎样共同定位一次慢 Step？ | 跨请求 ID 和 CPU/GPU 时间线 |
| 92 | G | 深挖 | NCCL Hang 怎么和慢 Kernel 区分？ | Rank 证据、Watchdog 和 Collective 顺序 |
| 93 | G | 深挖 | Reward 提高但模型质量下降该怎么排？ | 数据过滤、KL、Length Bias、Precision |
| 94 | G | 综合 | 如何对长尾 RL 负载做容量与成本规划？ | GPU-hours、P99 与预估偏差 |
| 95 | H | 实作 | 现场编码：有背压、可取消的 Async Producer–Consumer | 任务生命周期和泄露测试 |
| 96 | H | 综合 | 从白板画出 8 卡 RL Trainer 并解释每个等待 | Buffer、版本、权重更新与资源 |
| 97 | H | 研读 | 如何检查一篇 RL Infra 论文的真实性和可复现性？ | Ablation、指标定义、实验对照 |
| 98 | H | 实作 | 把课程的 Toy System 做成可以运行的评估基准 | Tests、Seed、恢复、版本语义 |
| 99 | H | 综合 | 针对真实长尾故障写设计评审与替代方案 | 性能、正确性、成本、复杂度权衡 |
| 100 | H | 综合 | 最后一次系统面试：算法 × 训练 × 推理 × 故障恢复 | 现场推导、编码、Benchmark 和个人盲点 |

## 纸面知识进入工程的时机

Day 01–09 先在极小概率模型中建立数学直觉，Day 10 做最小 Learner；Day 15–26 将分母、Autograd、数值与生命周期追踪回真实 Tensor；Day 27–56 把同一个训练环映射到 GPU、通信和 Serving，不能把 CPU Toy 的时间当作 LLM 性能；Day 57–85 加入工具、Sampler、Partial Rollout 与 Off-policy，必须说明为什么等待、何时更新版本、什么结果可以重复使用；Day 86–100 用故障和评审验证设计。每一阶段的实验可以选做，但后面的讲义应能够独立补齐需要的前置概念。

## 交互课如何配

交互组件只服务于能计算差异的机制。Day 01 调 Policy 概率观察期望 Reward；Day 02 调 Baseline 比较方差；Day 03 改 Token Shift / Action Mask；Day 04 调 Ratio、Advantage 和 Clip；Day 05 改 Group Reward 与组完整性。后续可继续做 GAE 回溯、KV Cache 预算、调度时间线或 Staleness 实验。没有反事实可比较的主题不硬做小游戏。

## 论文锚点与适用边界

- **经典算法**：[REINFORCE / Williams (1992)](https://link.springer.com/article/10.1007/BF00992696)、[GAE (2015)](https://arxiv.org/abs/1506.02438)、[PPO (2017)](https://arxiv.org/abs/1707.06347)。论文提供算法概念，具体实现必须通过代码/测试验证。
- **LLM RL 目标**：[DeepSeekMath / GRPO (2024)](https://arxiv.org/abs/2402.03300)、[DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948)、[DAPO (2025)](https://arxiv.org/abs/2503.14476)、[GSPO (2025)](https://arxiv.org/abs/2507.18071)。它们不是彼此同义的“GRPO 变体介绍日”；会按各自改了哪层目标/采样分布分别学习。
- **框架与异步**：[HybridFlow/verl](https://arxiv.org/abs/2409.19256)、[AReaL 公开项目](https://github.com/areal-project/AReaL)、[DORA (2026，预印本)](https://arxiv.org/abs/2604.26256)。框架版本快速变化，文档只断言核对过的源码与公开声明；未复现的吞吐提升只能称“论文/项目报告”。
- **推理与工程**：[FlashAttention](https://arxiv.org/abs/2205.14135)、[PagedAttention/vLLM](https://arxiv.org/abs/2309.06180)、[SGLang](https://arxiv.org/abs/2312.07104)。重点是 IO、KV 与资源调度，不是列软件功能表。

**旧版课程处理：** 2026-10-27 之前 Day 01–04 以数据契约、显存、梯度累积、Policy Gradient 的顺序组织。它们是重要知识，但起步缺少目标函数/梯度估计的连续因果链。新版 Day 01–04 被替换为优化目标、Score Function、Token Logprob、PPO；原实验文件保留作为后续迁移素材（例如旧 Data Contract 可用于 Day 65，Memory Budget 可用于 Day 18，Gradient Accumulation 可用于 Day 16），不再让课号含义与内容不一致。
