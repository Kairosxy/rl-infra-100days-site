# RL Infra 技术谱系与演进

技术谱系按问题和机制组织，十一章是阅读目录。已有专题按所属家族归档；本轮新增 9 篇前置机制正文，基础部分已有入口的家族仍需补齐剩余分支。下面 58 个家族是覆盖目标，并不表示所有列出的技术已经有完整正文或已验证的源码。知识库不是论文名称清单：应先解释一类问题和共同机制，再比较主要分支，最后深入具体算法、版本实现和工程 Issue。

## 分类与演进关系

技术家族、先修、历史继承、组合使用、实现、解决的问题必须分别表达。只有先修关系要求无环；历史继承必须有论文或源码证据，方法发布日期顺序本身不证明继承。按同一问题比较的方法可能并行发展，也可能组合使用。每个方法只维护一个主正文，跨章节用关联入口。覆盖状态与技术成熟度分别标记；不能把尚未写的计划节点注册为空 Topic，也不能把已有论文等同于生产成熟。

例如，MTP 是模型训练/预测机制，可以成为投机解码的 Draft；DFlash 是并行 Block Draft 的具体实现。两者共同属于 Serving 的投机解码家族，但不应与整个家族平级。Score Centering 属于策略梯度纠偏；数值一致性、Router Replay、重要性重加权、采样支持集对齐和版本滞后控制处理不同机制。Reward Serving 是反馈服务，不是 Score Centering。

## 十一章覆盖审查

### 模型基础与架构

把模型按表示、序列混合、容量、生成机制和多模态接口拆分。ViT 是视觉 Backbone，Omni 是跨模态输入输出架构；DiT 是生成网络结构，不是新的 Diffusion 数学定义。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| 数学与学习基础 | 如何表达分布并通过目标函数更新参数？ | 线性代数、概率/期望、Entropy、KL、MLE/Cross Entropy、Autodiff、梯度与优化；从概率到策略梯度。 | [张量、矩阵乘法与自动微分](../model/tensor-autograd.md)；[概率、Logprob 与信息量](../model/probability-information.md)（基础部分） |
| Token 与表示 | 原始输入如何变成可训练张量？ | BPE/SentencePiece、Embedding、LM Head、Weight Tying、Causal Shift；视觉 Patch、音频 Codec、VAE Latent 与连续动作。 | [Tokenizer、Embedding 与语言模型训练目标](../model/token-representation.md)（基础部分） |
| 序列建模与长上下文 | 如何处理 Token 间依赖并控制 KV 与计算成本？ | RNN/SSM 背景、Transformer；MHA/MQA/GQA/MLA（头或 KV 表示）；滑窗/稀疏 Attention（连接范围）；Linear Attention/SSM/混合架构（序列混合机制）；RoPE 与位置外推。FlashAttention 属于 Kernel 实现，跨链关联。 | [Transformer：从 Token 到 Logits](../model/transformer-attention.md) |
| 模型容量与训练目标 | 如何扩大容量、减少活跃计算、改善预测？ | Dense、MoE、细粒度专家、Shared Expert、路由与负载均衡；Next-token Prediction 与 MTP（训练目标）；参数量、激活参数、FLOPs 和显存估算。EP 是分布式实现，R3 是正确性。 | [Dense、MoE 与模型容量](../model/model-moe.md)；[Tokenizer、Embedding 与语言模型训练目标](../model/token-representation.md)（基础部分） |
| 多模态理解与 Omni | 如何表示多模态输入、融合语义并产生跨模态输出？ | CNN/ViT 视觉编码；CLIP 类图文对齐；VLM 的 projector/cross-attention/统一 token；音频与视频编码；Omni 的统一理解生成、多时基位置编码、流式输出。对比仅输入多模态和可输出多模态。 | 待补独立正文 |
| 生成模型 | 如何由概率模型产生文本、图像、音频、视频或连续动作？ | Autoregressive、VAE；Diffusion 的加噪/去噪与 score；Pixel/Latent Diffusion；U-Net/DiT Backbone；Flow Matching/Rectified Flow 目标与 ODE；CFG、采样器、蒸馏/少步生成。区分模型架构与生成目标。 | 待补独立正文 |
| 具身策略与环境建模 | 如何由观测、语言产生动作，或预测交互后的环境？ | VLA 的离散 Action Token、连续回归、Diffusion/Flow Action Head；Action Chunk、实时闭环；World Model 的潜变量动力学、action-conditioned 预测、视频生成/交互世界模型、Model-based RL 与规划。VLA 不等于 Agentic RL，视频生成也不必然等于 World Model。 | 待补独立正文 |

### GPU 与分布式计算基础

建立硬件、执行、通信三层成本模型；不能只靠 NCCL 原语理解吞吐。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| GPU 执行与存储 | 为什么同 FLOPs 的算子耗时不同？ | SM、Warp、Tensor Core、Register/Shared Memory/L2/HBM、Occupancy、Memory Coalescing、Roofline、Kernel Launch。 | [GPU 执行模型、显存与算术强度](../compute/gpu-execution-memory.md)（基础部分） |
| 异步执行与编译 | CPU 提交与 GPU 执行怎样衔接？ | CUDA Stream/Event、同步语义、Graph Capture/Replay、PyTorch Dispatcher/Inductor、Triton/TileLang、编译与动态 Shape；区分语言、编译器和执行优化。 | [CUDA Stream、Event 与异步执行](../compute/cuda-streams.md)（基础部分） |
| GPU 资源共享与隔离 | 多进程训练和生成能否在同卡共存，如何控制相互干扰？ | 时间复用、CUDA Stream 并发、跨进程 Context/MPS、MIG 硬件分区、显存与带宽争用、Host Offload；CTRL 是 Co-residency 的新近研究实例，不等同于一般 MPS 能力。 | 待补独立正文 |
| 网络与集群拓扑 | 为什么跨节点与节点内通信代价不同？ | PCIe/NVLink/NVSwitch、IB/RoCE、RDMA/GPUDirect、NUMA、NIC/rail、拥塞与拓扑映射。 | [PCIe、NVLink、RDMA 与集群通信](../compute/gpu-network-topology.md)（基础部分） |
| Collective 算法 | 怎样把多 Rank 的张量聚合和分发？ | P2P、Broadcast、AllReduce、AllGather、ReduceScatter、AllToAll；Ring/Tree/层次算法；带宽、时延、消息规模与同步。 | [NCCL 与 Collective：GPU 为什么在等？](../compute/distributed-collectives.md) |

### 分布式训练

按切分对象区分并行；按运行生命周期理解显存与通信。不同并行维度可以组合，不写成先后替代。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| 数据并行与状态分片 | 复制数据计算时怎样降低模型状态成本？ | DP/DDP、梯度累积与 Bucket、Distributed Optimizer、ZeRO-1/2/3、FSDP1/2、HSDP、DTensor；区分参数/梯度/优化器分片。 | [FSDP2 / ZeRO：分片的是状态还是计算？](../parallel/training-sharding.md)；[从 DataLoader 到 Optimizer Step](../parallel/training-update-lifecycle.md)（基础部分） |
| 张量与层并行 | 单层或整模型太大时如何切分计算？ | TP Column/Row Linear 与 Vocab Parallel；PP Microbatch、GPipe、1F1B、Interleaving、Zero Bubble；通信边界和 Pipeline Bubble。 | [TP：为什么切开矩阵还要通信？](../parallel/tp-matrix-communication.md)；[PP：为什么流水线仍有 GPU 空闲？](../parallel/pp-pipeline-bubble.md) |
| 序列与上下文并行 | 如何在长序列下分担激活和 Attention？ | Megatron SP、CP、Ring Attention、Ulysses 类序列交换；不同方案切分范围、KV 交换、通信量与重算。 | [CP / SP：长上下文的序列怎么切？](../parallel/cp-sp-attention.md) |
| 专家并行 | 如何分布 MoE 专家并处理动态路由负载？ | Token Dispatch/Combine、All-to-All、Grouped GEMM、负载偏斜、容量与 Drop、通信重叠、DeepEP 类通信实现。 | [EP：为什么 MoE 会卡在 All-to-All？](../parallel/ep-token-routing.md) |
| 混合并行与资源规划 | 哪些 Rank 实际参与哪次计算或通信？ | Device Mesh/Process Group、多维并行组、显存/激活估算、批量与微批次、拓扑感知、Recompute/Offload；具体框架中 EP 和 DP 的复用关系以 Rank Group 为准。 | [DP / TP / PP / EP 怎么组合](../parallel/parallel-topology.md) |
| 训练生命周期与优化器 | 模型从数据进入到参数保存经历什么？ | Pretraining/SFT、Packing/动态 Token Batch、DataLoader、梯度累积、AdamW/Adafactor/Muon 等优化器家族、LR Schedule、梯度裁剪、Checkpoint/Reshard/异步保存。优化器不是并行策略。 | [从 DataLoader 到 Optimizer Step](../parallel/training-update-lifecycle.md)（基础部分） |

### 推理与 Serving

按请求生命周期、内存管理、调度、计算并行与生成算法拆分。PD 分离、Paged KV 和投机解码解决不同问题。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| 自回归生成与采样 | 从 Prompt 到响应的运行阶段有什么区别？ | Prefill/Decode、TTFT/TPOT/TBT；Greedy/Temperature/Top-k/Top-p、Stop、Structured Decoding；多模态 Encoder、流式语音与连续动作生成的执行差异。 | [Prefill、Decode 与实际采样分布](../serving/autoregressive-serving.md)（基础部分） |
| KV Cache 管理 | 怎样控制请求状态的驻留、共享与迁移？ | 连续 KV、PagedAttention、Prefix/Radix Cache、KV 压缩/量化、Offload、HiCache/分层缓存、Eviction 与一致性。 | [KV Cache 与请求调度](../serving/kv-scheduling.md) |
| Agent 上下文生命周期 | 哪些历史必须留在模型输入，哪些只需保存在 Harness 或外部存储？ | 完整历史与实际执行 Context、语义压缩/摘要、工具输出失效、Context Eviction、前缀复用与重 Prefill；MORTIS、Penelope、Contexture 等作为待深入核验的分支。区分删逻辑 Token 与仅回收物理 KV。 | 待补独立正文 |
| 请求调度与并行 | 怎样控制吞吐、首 Token 和尾延迟？ | Static/Continuous Batching、Chunked Prefill、Admission/Preemption、DP Attention、TP/EP、PD Disaggregation、KV Transfer、Cache-aware Routing、弹性负载均衡。PD 不是模型并行维度。 | [推理并行：为什么不能照搬训练 TP？](../serving/serving-parallelism.md) |
| 投机解码 | 怎样减少串行 Target Forward 次数？ | 经典 Draft–Verify；独立小模型；N-gram/Suffix；Medusa/Native MTP；EAGLE 家族；并行 Draft（PARD、DFlash、DFlash2、DSpark）；动态 Draft 长度、Adaptive Verification、精确/近似验证。MTP 训练与 Draft 训练另连模型目标；方法支持按版本核验。 | 待补独立正文 |
| 推理时计算与搜索 | 怎样用额外候选预算改善任务表现？ | Best-of-N、Beam/Tree Search、Verifier Rerank、MCTS、Adaptive Compute、Reasoning Budget；与生成加速分开衡量。 | 待补独立正文 |

### 强化学习算法与后训练

以估计器、优化目标、反馈来源、采样策略为正交分类。不能把 RLVR 当作 PPO 后继算法，也不能把 OPD 当作 GRPO 的变体。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| RL 基础与价值估计 | 如何定义状态、动作、回报和信用分配？ | MDP/POMDP、Bellman、MC/TD、Value/Q、Advantage、GAE、Terminal/Truncation、Value-based/DQN；Actor-Critic 与连续控制 SAC/TD3 的必要背景。 | 待补独立正文 |
| 策略梯度与估计器 | 不可微 Reward 如何转为可训练梯度？ | Score-function/REINFORCE、Baseline、Actor-Critic、RLOO、ReMax、REINFORCE++；IS、V-trace 等离策略估计的基本假设。 | [一次 Reward 怎么更新 Policy？](../rl/policy-gradient.md) |
| 策略约束与优化粒度 | 如何控制更新幅度与不同长度样本的梯度？ | TRPO/PPO/GRPO、Dr.GRPO、DAPO、GSPO、CISPO、SAPO；按 Critic/Group Advantage、Token/Sequence Ratio、Clip/平滑约束、Loss Reduction、Entropy 等维度逐项比较。DAPO 还包含动态采样，跨链关联。 | [PPO / GRPO 的优化目标](../rl/ppo-grpo.md) |
| 反馈范式与偏好优化 | 监督信号来自哪里、如何进入训练目标？ | SFT/Instruction Tuning、RLHF/RLAIF/RLVR、ORM/PRM/Generative RM、Rule/LLM/Environment Verifier、DPO/IPO/KTO/SimPO 等偏好目标；在线采样与离线偏好数据区别。 | 待补独立正文 |
| 蒸馏与混合目标 | 学生如何从 Teacher 分布而非单个标签学习？ | Offline KD、Forward/Reverse KL、On-policy Distillation、Self-Distillation、Online Teacher、Reward+Distillation；Full-vocab 与 Top-k/Sampled Estimator。 | [OPD、Teacher 分布与 KL](../rl/opd-kl.md) |
| 持续学习与自我改进 | 如何从生成、反馈与学习循环中持续改进能力？ | Synthetic Data、Self-training、Self-rewarding、Self-play、Automatic Curriculum、RSI；区分生成数据、模型更新与改进机制本身演进。广义递归改进仍需独立证据。 | 待补独立正文 |
| 自适应采样与树形探索 | 如何把 Rollout 预算分配到更有学习价值的任务和中间状态？ | 固定 Group Sampling、Rejection/Dynamic Sampling、课程难度、任务级分配、Entropy/Value/Uncertainty 引导、Action-prefix Tree、分支处 Credit Assignment、历史统计与 Drift-aware Replay；WEIR 等新论文作为待深入验证的具体方法。与 Serving 搜索及环境状态 Fork 区分。 | 待补独立正文 |
| 生成与连续动作后训练 | 离散 Token RL 之外如何优化策略或生成器？ | Diffusion/Flow Policy 的轨迹、密度与梯度估计、偏好优化、DDPO 类生成 RL、连续控制与 Imitation/Offline RL；不能直接假定存在 LLM Token Logprob 契约。 | 待补独立正文 |

### RL 训练基础设施

按数据生产、服务、消费、版本与资源边界拆分，异步只是调度家族。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| RL 数据循环与接口 | 一次 Prompt/Trajectory 怎样成为有效训练样本？ | Rollout–Learner Contract、Prompt Group、Reward/Value/Old/Ref Logprob、Advantage、Mask、Batch Readiness、数据 schema 与生命周期。 | 待补独立正文 |
| 分布式轨迹数据面 | 跨进程生产的多字段数据何时可见、可读、可回收？ | 共享内存/对象存储/RDMA、Control Plane 与 Payload、Metadata/Field Readiness、Link/Alias/Copy、部分写入、空批轮询、消费确认、超时重试、幂等和恢复；Ray Object Store、TransferQueue、Mooncake。不要把 RPC 成功或元数据 ready 当作已验证载荷一致。 | 待补独立正文 |
| 资源部署与角色规划 | 哪些模型共享 GPU，哪些需要独立资源？ | Colocate/Hybrid/Disaggregate、Learner/Rollout/Ref/Critic/Teacher/Reward、资源池与弹性并发、显存释放与重分片。 | [RL 训推并行：Learner 与 Rollout 怎么配 GPU？](../async/rl-parallel-planning.md) |
| 供数与执行调度 | 如何避免整批阻塞又控制在途数据？ | 同步批式、流式流水、One-step Off-policy、Full Async、Partial Rollout、DataSampler、Group Completion、Drain/Resume、Staleness Budget；数据面与多资源背压独立关联。 | [异步 Rollout 与 Policy Staleness](../async/async-rollout.md) |
| 多资源准入与背压 | 生成、环境、评分和消费容量不同，如何防止局部提速转成全局积压？ | Producer–Consumer、Credit/Permit、Queue Budget、Admission、动态补投、Little's Law、生成资格与完整 Task 生命周期分离、端到端 Drain/Cancel、慢 Reward 队列、等待环与公平性。 | 待补独立正文 |
| 评分与教师服务 | 训练信号如何作为服务产生并安全回填？ | Reward Model、Rule/Environment Verifier、Judge/PRM、Teacher Serving、Batching、Timeout/Retry、版本/幂等、异步评分；Score Centering 与评分服务无关。 | 待补独立正文 |
| 权重传输与持久化 | 怎样传输并持久化模型与优化器状态？ | Weight Redistribution/Sync、Checkpoint Engine、张量传输/共享内存/RDMA、量化重打包、Reshard、恢复范围。与请求 KV Transfer 共享基础传输但状态语义不同。 | 待补独立正文 |
| 更新提交与策略发布 | 训练尝试、成功更新、权重安装及 rollout 准入何时推进版本？ | Optimizer Step/Skip、逻辑提交与发布版本、Weight Install/Publication Gate、在途样本版本、原子可见性、Checkpoint 一致点、部分失败和回退。CTRL 的 Publication Gate 是这一家族的研究实例；异步不自动证明 off-policy 或任何 staleness 保证。 | 待补独立正文 |

### Agentic RL 与交互系统

Agentic 指多步决策和环境交互，不是所有新模型方向的上位概念。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| Harness 与轨迹 | 模型输出如何成为环境动作并进入下一轮上下文？ | Agent Loop、Tool Schema、Action Parsing、Sandbox、Observation、Action Mask、上下文与 Trajectory 序列化、Timeout/Retry。 | [Agent 的 Action 到底是什么](../agent/agent-action.md) |
| 长时程任务与信用分配 | 如何把结果反馈分配到多轮决策？ | POMDP、Outcome/Process Reward、Step/Trajectory-level Advantage、Memory、Planning/Search、任务分解、状态与工具结果可复现。 | 待补独立正文 |
| 环境与并行采集 | 如何扩展环境交互而不失去轨迹语义？ | 环境池、并发会话、动态任务长度、仿真/真实环境、异步工具调用、Multi-Agent、Self-play；具身 VLA、World Model 作为独立模型家族关联。 | 待补独立正文 |
| 环境状态分叉与复用 | 多条轨迹共享历史时怎样安全从同一中间状态继续探索？ | Environment Snapshot/Fork/Replay、Copy-on-Write、工具副作用与隔离、KV Prefix 共享、Trajectory ID/Parent/Version、树形 Rollout 与失败恢复。算法负责选择分支，运行时负责维护可重放状态；不能仅靠复制 Token 序列假设环境一致。 | 待补独立正文 |

### 训练与推理性能优化

按成本来源划分，不把每个新 Kernel 或 dtype 当一级技术。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| 数值格式与量化 | 怎样减少存储、访存和计算成本？ | FP32/TF32/FP16/BF16、FP8 E4M3/E5M2、MXFP8/MXFP4/NVFP4、INT8/INT4；PTQ/QAT、W/A/KV Quantization、Per-tensor/Channel/Token/Block Scale、混合精度；格式与策略分开。 | [FP8 的收益与数值边界](../optimization/fp8-numerics.md) |
| 算子与编译优化 | 怎样减少中间张量、launch 和无效访存？ | GEMM/Grouped GEMM、FlashAttention 系列、Kernel Fusion、CUDA Graph、Triton/TileLang/CUTLASS、动态 Shape 与编译缓存。 | 待补独立正文 |
| 资源与通信优化 | 怎样缩短计算、通信和等待的关键路径？ | 计算通信 Overlap、Activation Recompute/Offload、MoE Dispatch 与负载均衡、动态 Batch、长尾治理、资源弹性；区分训练与推理的瓶颈。 | [RL Step 真的慢在哪里](../optimization/rl-throughput.md) |

### 训练正确性与稳定性

把同版本执行偏差与跨版本策略滞后分开。算法纠偏不应掩盖数据契约错误。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| 概率契约与诊断 | 两个 Logprob 是否来自同一动作、同一状态和正确分布？ | Token Shift、Mask、采样变换、权重版本、Old/Behavior/Current/Reference、KL/Ratio、支持集、确定性与 Batch Variance；投机解码输出分布另行验证。 | [Token Logprob 与训练目标](../learner/logprob-contract.md) |
| 训推源头对齐 | 相同模型版本为什么仍会生成不同概率？ | 精度、Kernel/归约、Batch 不变性、Router Replay/R3、Sampler Replay、Tokenizer、分布式实现差异；区分路由、支持集和数值。 | [MoE Routing 与 R3](../learner/moe-router.md) |
| 策略纠偏与滞后控制 | 当采样策略不同于优化策略时怎样控制偏差和方差？ | IS/TIS、Token/Sequence Reweighting、Rejection/Mask、IcePop、Score Centering、Regularized Score Centering；区分中心化、正则和重要性重加权。Policy Lag/Staleness、ESS、同步与样本过滤是治理手段。 | 待补独立正文 |
| 训练更新等价性与梯度安全 | 改变分片、批次、Mask 或优化器执行顺序后，是否仍产生同一逻辑更新？ | 有效 Token/Sequence/Group 的 Loss Reduction、全局分母、Microbatch 累积、参数与梯度逻辑身份、Replica/Shard/Alias 去重、Global Norm/Clip、Unscale、非有限梯度共识、Mask 前后的计算图安全、数值稳定统计；对照逻辑梯度及参数/Optimizer State，而非仅比较 Loss。 | 待补独立正文 |
| 训练稳定与分布式容错 | 如何识别静默错误、数值异常和系统故障？ | NaN/Inf/Loss Spike、梯度裁剪、低精度溢出、OOM、NCCL Hang、Rank Failure、Checkpoint 恢复、幂等与数据一致性、故障注入。 | 待补独立正文 |

### 主流框架源码

框架是机制的具体实现和组合，不是技术演进链。每篇源码文档以真实版本、调用边界和数据所有权为准。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| 模型与训练引擎 | 张量、参数与并行组如何落在代码中？ | PyTorch/DDP/FSDP2、Megatron-Core/Bridge、DeepSpeed、TransformerEngine、Transformers；并行、优化器、低精度与 Checkpoint。 | 查框架实现目录；具体缺口待逐版本核对 |
| Serving 引擎 | 请求、KV、调度与执行如何协作？ | SGLang、vLLM、Speculators；请求调度、Prefix Cache、PD、Speculative Decoding 与实际采样。 | 查框架实现目录；具体缺口待逐版本核对 |
| RL 训练框架与运行时 | 角色、数据、版本分别由谁拥有？ | VERL、slime、AReaL、NeMo RL、OpenRLHF、TRL、RLinf、Ray、Transfer Engine；以公开可复现代码为准。 | 查框架实现目录；具体缺口待逐版本核对 |
| 多模态与具身框架 | 模型训练与环境执行怎样组合？ | Diffusers、OpenVLA/OpenPI、TorchRL、仿真与机器人环境；适度扩展到实际训练/Serving 所需的机制。 | 查框架实现目录；具体缺口待逐版本核对 |

### 工程实践与系统设计

以技术问题、证据和可复算实验贯穿全领域。

| 技术家族 | 上位问题 | 分支与补全范围 | 当前正文入口 |
|---|---|---|---|
| 评估与实验科学 | 怎样区分效果、样本效率、计算效率和实际收益？ | Baseline、Ablation、Scaling Law、Benchmark、Pass@k/任务成功率、有效 Token/s、GPU-time、SLO、成本；置信区间与可复现性。 | [怎样判断一种新方案有效](evidence-standards.md) |
| 资源设计与可观测 | 怎样将理论上限与真实执行对照？ | 显存/通信/计算估算、容量规划、Profiler/NSys、Tracing、关键路径与 p95/p99、分布式故障定位、性能回归。 | 待补独立正文 |

## 从日程提取新的技术家族

社区速览用于发现技术家族与新方法，架构周报用于抽象系统契约与边界，每日工程题用于发现基础断层与验证问题。它们与技术正文分开维护；不把日报、周报或工程题整篇导入技术手册，也不将内部设计、真实资源配置、私人日志、内部文件链接或未公开数据写入公开仓库。以下是 2026-11-24 的分类增量：只写可公开复用的机制，具体方案与性能需从原始来源独立验证。

| 已发现的问题 | 归入家族 | 可公开验证的入口 |
|---|---|---|
| 同卡并发执行训练与生成，训练结束不必等于发布完成 | GPU 资源共享与隔离；更新提交与策略发布 | [CTRL](https://arxiv.org/abs/2611.27119)，[CUDA MPS](https://docs.nvidia.com/deploy/mps/)；比较同卡共驻留、时间复用、独立资源池，不把论文数据当作本地实测 |
| 语义历史删减可能牺牲已有 Prefix Cache | Agent 上下文生命周期 | [Contexture](https://arxiv.org/abs/2611.26980)；检查实际执行 Context 与完整历史、质量与重算开销 |
| 从任务中间状态分叉，选择有信息价值的采样位置 | 自适应采样与树形探索；环境状态分叉与复用 | [WEIR](https://arxiv.org/abs/2611.27570)；算法选择与运行时分叉分别验证 |
| Metadata 请求等待可能阻塞字段就绪通知，传输超时不等于未写入 | 分布式轨迹数据面；多资源准入与背压 | [TransferQueue #217](https://github.com/Ascend/TransferQueue/pull/217)、[#219](https://github.com/Ascend/TransferQueue/pull/219)；以合并版本为准，独立验证消息与载荷 |
| 生成结束、评分结束、可消费和版本安装是不同事件 | 多资源准入与背压；更新提交与策略发布 | 通用生产者/消费者模型与公开 RL 系统；提出可复算场景，不公开内部设计细节 |
| 相同逻辑梯度的多个分片视图可能产生不同范数 | 训练更新等价性与梯度安全 | [PyTorch clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)、[PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)；对照逻辑模型、唯一统计所有权、统一缩放与 Skip |

最新社区论文仅核对了公开摘要及其所报告的研究机制，标作新近研究实例；没有逐版本核验的其他方法属于覆盖候选。每次吸收日程结果时记录日期、原始公开来源、家族归属及覆盖状态；已有家族优先增加分支或改进问题，只有出现独立上位问题时才新增家族。日程的公开来源无法访问或结果截断时，明确范围，不补造内容。持续关注其他主流模型、生成、具身和算法方向，不能因三个日程当前偏重 RL 系统而缩小全谱系。

## 技术演进的验收

父专题至少要说明原始问题、基础机制、主要替代方案、改进动机、代价与适用边界。具体方法在家族下展开，给出原始论文、公式、可复算样例、实现版本和工程验证。例如，投机解码比较提议来源、串行深度、验证模式、接受长度和端到端收益；策略纠偏比较分布偏差来源、Ratio/Mask/Score 的作用对象、支持集和额外数据成本。算法家族要比较 Advantage、Ratio、Reduction、采样和反馈，而非只列论文年份。

本轮将领域覆盖与正文进度分开。当前缺失的父专题优先补，不因为同族中已有一个新论文名称就认定已覆盖。优先同时推进概率/优化/MDP/GAE/IS、Tokenizer、GPU/网络、DP/DDP、Prefill/Decode，以及策略优化、模型架构、生成建模、多模态、VLA 等主流家族。异步、性能与正确性继续补，但不是唯一内容主线。

## 证据入口

以下用于核对分类边界与后续深入的原始资料/官方实现入口，不代表对表中全部方法完成逐论文或逐行源码审查。动态文档核对日期 2026-11-24；具体实现专题固定 release/tag/commit。

- [Transformer](https://arxiv.org/abs/1706.03762)、[ViT](https://arxiv.org/abs/2010.11929)、[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)、[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)
- [DDPM](https://arxiv.org/abs/2006.11239)、[DiT](https://arxiv.org/abs/2212.09748)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[OpenVLA](https://arxiv.org/abs/2406.09246)、[π0](https://arxiv.org/abs/2410.24164)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)、[NCCL Collective](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)、[Megatron Core 0.20.1 Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.20.1/user-guide/parallelism-guide.html)、[PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PagedAttention](https://arxiv.org/abs/2309.06180)、[DistServe](https://arxiv.org/abs/2401.09670)、[经典投机采样](https://arxiv.org/abs/2211.17192)、[vLLM 投机解码](https://docs.vllm.ai/en/latest/features/spec_decode/)、[Speculators 方法比较](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/decision_guide/)、[DFlash](https://proceedings.mlr.press/v306/chen26bv.html)
- [GAE](https://arxiv.org/abs/1506.02438)、[PPO](https://arxiv.org/abs/1707.06347)、[DPO](https://arxiv.org/abs/2305.18290)、[DAPO](https://arxiv.org/abs/2503.14476)、[GSPO](https://arxiv.org/abs/2507.18071)、[Absolute Zero](https://arxiv.org/abs/2505.03335)
- [Score Centering](https://arxiv.org/abs/2609.20807)、[verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)、[verl OPD](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md)、[Mooncake](https://github.com/kvcache-ai/Mooncake)、[RLinf](https://github.com/RLinf/RLinf)

