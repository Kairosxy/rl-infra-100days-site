# RL Infra 100 Days · 从一次更新到一个可证明正确的系统

**如何使用：** 每天北京时间 09:00 自动发布一课。每天是一段新的学习材料，不是一次必须通过的考试；随时跳课、重学、跳过练习都可以。表里的 100 个编号是内容投放节奏，而不是把一百个名词平均切成一百天。

## 为什么这样排

先回答「策略凭什么沿这个方向更新」，再回答「代码是否真的计算了这个梯度」，接着追问「计算落在 GPU/多卡上时哪些代价和语义会变化」，最后才有资格讨论 Rollout/Agent/Async/OPD。**同一个训练样本**贯穿推导、Tensor、进程、队列和恢复过程；系统多了一层复杂度，就新增一条能够验证的约束。

这次不把 DAPO、GSPO、EAGLE、Ray 或 TransferQueue 各自当作必须占一天的「产品介绍」。经典与新工作都应先交代它想解决的失效条件，然后才读实现和实验。术语、Schema、CLI 参数能随用随查；概率分布、梯度估计、Mask、非线性优化、数值精度、通信、调度和数据选择偏差则要讲到具体计算和反例。

## 九段完整的学习任务

**A｜Day 01–12 · 从一次反馈到一次可信更新。** 从定义目标到可观测梯度，数学和代码能指向同一个量。
**B｜Day 13–24 · 把目标写成不会悄悄变味的 Learner。** Mask、Reduction、参数版本和 Optimizer 的更新边界一致。
**C｜Day 25–36 · 弄清 GPU 的真实成本。** 任何“省显存、提速度”的结论必须指明对象、单位和测量条件。
**D｜Day 37–50 · 一次更新跨过多张 GPU。** 梯度、状态和通信可以正确拆分与恢复。
**E｜Day 51–64 · 推理如何变成可靠的数据生产者。** 服务速度、采样分布和 Trainer 需要的 Logprob 是三件事。
**F｜Day 65–74 · 从单次生成到多轮 Agent 轨迹。** 行为、观察、评分和环境状态都能追溯。
**G｜Day 75–88 · 让生成与训练并发而不偷换目标。** 允许并发不等于允许丢失版本、样本身份或选择偏差。
**H｜Day 89–94 · 蒸馏与 RL 目标怎样相互作用。** Teacher 分布、KL 方向与资源代价必须对齐。
**I｜Day 95–100 · 用证据调试并设计系统。** 以实验、失败路径和代价模型而非架构名词作结论。

## 逐日学习安排

| Day | 进度 | 学生今天真正要解决的问题 | 当天可以验证的东西 |
|:---:|:---:|---|---|
| 01 | A | 拿到一次最终评分，参数为什么会改变？ | 二动作策略，手算并验证一次 SGD：p=0.5→0.5498 |
| 02 | A | 为什么同样的策略有时更新、有时没有梯度？ | REINFORCE 采样分布、Baseline 与方差数值对照 |
| 03 | A | 只有最终 Reward，哪些 Token 在真正接受梯度？ | 二 Token Response：Logprob 求和、Shift、Action Mask |
| 04 | A | 同一批数据反复训练，为什么 Ratio 开始不可信？ | Old/Current 概率、PPO Clip 的两侧平台与 KL 反例 |
| 05 | A | 不训练 Critic，能从同一 Prompt 的几条回答里学到什么？ | GRPO 组统计、全同奖励及样本内 Baseline 偏差 |
| 06 | A | 把 Token Loss 求和或平均，优化目标还一样吗？ | 短/长回答的 Mask、Reduction 与梯度贡献 |
| 07 | A | 一个最终分数应该归功于哪一步动作？ | Return、Reward-to-go、Credit Assignment 的两步实例 |
| 08 | A | 时间限制和真正终止为何不能用同一个 Mask？ | Value/TD/GAE 的 backward recursion 与 Bootstrap |
| 09 | A | 用 Critic、组内奖励和采样修正，究竟牺牲了什么？ | 对比偏差/方差/数据成本；用 DAPO 与 GSPO 检查具体假设 |
| 10 | A | 一轮 RL 更新从数据到新权重能否逐项校验？ | CPU 最小训练循环；梯度、Mask、Loss 与一次 Step 测试 |
| 11 | A | 训练 Reward 上升，模型为什么仍可能退化？ | Verifier 错判、奖励投机与 Pass@k 的反事实 |
| 12 | A | 论文公式和训练代码真的在优化同一个量吗？ | 对照 PPO/GRPO 主张、Reduction 和最小单元测试 |
| 13 | B | Rollout、Old、Current、Reference 到底是谁？ | 相同 Token 的四类概率及采样变换对照 |
| 14 | B | Logprob 相差 0.3，先怀疑浮点误差吗？ | Tokenizer/Temperature/Top-p/版本排查树 |
| 15 | B | 哪些张量应该 Detach，哪些需要梯度？ | Policy Gradient Autograd 路径与意外梯度反例 |
| 16 | B | 为什么 2/2/1 的 Micro-batch 会改变梯度？ | 完整 Batch 与局部 Mean 的精确梯度比对 |
| 17 | B | 梯度裁剪、AMP 与 Optimizer Step 的先后能交换吗？ | AdamW Moment/Step Counter 与裁剪反例 |
| 18 | B | 变长序列 Packing 后，哪些位置仍是 Action？ | Position/Attention/Loss Mask 的不同职责 |
| 19 | B | KL 应该和哪个策略比较，方向又是什么？ | Reference vs Old、Forward/Reverse KL 数值例子 |
| 20 | B | 采样概率很小，Ratio 为什么会爆炸？ | Logprob 差分、数值范围、Clip 与梯度异常 |
| 21 | B | 固定 Seed 为何仍不能保证训练结果相同？ | RNG、采样顺序、算子确定性与检查点 |
| 22 | B | 最小 Checkpoint 恢复后为什么不是原轨迹？ | 参数、Optimizer、RNG、Sampler Cursor 恢复比对 |
| 23 | B | 一个 Loss NaN 应从哪个中间张量开始找？ | 首个非有限 Logprob/Advantage/Grad 的定位脚本 |
| 24 | B | 做一次无法蒙混过关的 Learner 代码审查 | 错误注入：Shift、Mask、Reduction、Step 时机 |
| 25 | C | 一个 Transformer Block 真正做了多少计算？ | 指定 B/L/H 的 GEMM/Attention FLOPs |
| 26 | C | 1B 权重 2 GB，为何训练会先在 Backward OOM？ | 参数、梯度、Moment、Activation 的时序账 |
| 27 | C | 长度翻倍，为何 Attention 的成本不止翻倍？ | QKᵀ/Softmax/V 形状与朴素 L² 存储 |
| 28 | C | FlashAttention 减少什么，并没有减少什么？ | IO 与 Online Softmax、算术复杂度分开 |
| 29 | C | 算力很高，为什么小矩阵仍然跑不快？ | Roofline、Tile/Pad、Launch 开销的上界 |
| 30 | C | Micro-batch 改小为什么未必能提高吞吐？ | 显存、Kernel Shape、Gradient Accumulation 时间 |
| 31 | C | Activation Checkpointing 的节省来自哪？ | Backward 所需 Saved Tensor 的生命周期 |
| 32 | C | FP8 Scale 一变，Logprob 误差为何可能放大？ | 量化格、Accumulation、尾部 Logprob 对照 |
| 33 | C | CUDA Graph 为什么减少开销也会限制调度？ | Replay 条件、动态 Shape 与 Capture 边界 |
| 34 | C | GPU Trace 上看到空洞应如何定性？ | Compute/Memory/Launch/Sync 分类与证据 |
| 35 | C | 同样 GPU 型号，长输入短输出为何性能不同？ | Prefill/Decode 的算术强度预估 |
| 36 | C | 用一次实验判定问题是计算、带宽还是调度 | 无 GPU 时建立成本预算；有 GPU 时记录 Profiler 基线 |
| 37 | D | 两个 GPU 求同一个 Batch 梯度，如何证明等价？ | DDP 平均语义、Rank 间数据与 Reduction |
| 38 | D | 为什么梯度同步不一定等到 Backward 完毕？ | Bucket、Hook、Overlap 的时间线 |
| 39 | D | FSDP/ZeRO 分片后，显存为何仍有尖峰？ | Sharded State 与参数 All-gather 的重叠 |
| 40 | D | Tensor Parallel 把矩阵切在哪个轴，通信发生在哪？ | Column/Row Parallel 的 Tensor Shape 和 Collective |
| 41 | D | Pipeline 为什么有 Bubble，也可能吃更多内存？ | 1F1B、Micro-batch 数与 Activation Lifetimes |
| 42 | D | Expert Parallel 的通信量取决于什么？ | Routing Token、All-to-All 与负载偏斜 |
| 43 | D | 长上下文如何跨卡拆 Attention？ | Context Parallel 的依赖、分块和通信 |
| 44 | D | All-reduce 和 All-to-All 的瓶颈怎么估算？ | α–β 成本、消息大小与交换拓扑 |
| 45 | D | 为什么多一个 GPU 反而可能更慢？ | Compute/Comm Ratio、跨机拓扑与并发竞争 |
| 46 | D | 如何选 DP/TP/PP/EP 而不是盲目放大并行度？ | 带显存约束的两个可比拓扑方案 |
| 47 | D | 多个 Rank 的 Checkpoint 怎么原子保存？ | Optimizer、Shard、元信息与一致性边界 |
| 48 | D | 恢复到不同并行度会改变哪些东西？ | Reshard、RNG/Optimizer 状态与验证计划 |
| 49 | D | 分布式 Hang 是谁在等待谁？ | Collective 顺序、Watchdog、Rank 日志对齐 |
| 50 | D | 为一轮 8 卡训练做容量与故障评审 | 成本表、通信量估计、OOM/Hang 的最小验证 |
| 51 | E | Prefill 为什么更像矩阵计算，Decode 为什么更像读内存？ | TTFT/TPOT 与算术强度 |
| 52 | E | KV Cache 该怎么算才能知道真实并发上限？ | GQA/MQA、Layer/Head/Dim/Token 的字节数 |
| 53 | E | PagedAttention 节约哪种浪费，付出什么管理成本？ | Block Mapping/Fragmentation/Lifetime |
| 54 | E | Continuous Batching 究竟在调度哪些请求？ | Admission、Prefill、Decode 的请求状态 |
| 55 | E | Prefix Cache 何时真正命中，何时只是占内存？ | 前缀相等条件、RadixCache 与驱逐规则 |
| 56 | E | Chunked Prefill 为什么会改变尾部延迟？ | TTFT/TPOT/P99 三指标的竞争 |
| 57 | E | 投机解码何时既保持分布又真正加速？ | 接受率、Correction、Draft Cost 的上界 |
| 58 | E | Sampling 参数为何能影响训练侧 Logprob 语义？ | Temperature/Top-p/Stop 与分布定义 |
| 59 | E | 训推 Logprob 对不上，如何先排除伪漂移？ | Token IDs、Mask、Policy、采样变换逐项对照 |
| 60 | E | FP8 或量化会怎样改变采样与 Token 长度？ | 定量误差检查与非量化基线 |
| 61 | E | 多机 PD 解耦的 KV 传输何时值得做？ | 传输量/带宽与排队时间的收支 |
| 62 | E | 推理侧 TP/EP 组合怎样影响通信和并发？ | MoE 推理路由、专家缓存、GPU 占用 |
| 63 | E | 吞吐增加为何可能让最慢用户更慢？ | 固定工作负载的 P50/P99、公平性与背压 |
| 64 | E | 怎样做出不会自我欺骗的 Serving Benchmark？ | 输入输出长度、Warmup、重复实验和样本分布 |
| 65 | F | Agent 调了工具，哪些 Token 才算模型的动作？ | Role/Tool Observation/Action Mask 对齐 |
| 66 | F | 工具执行失败后，环境状态是什么？ | Tool Result、Timeout、Retry 与幂等语义 |
| 67 | F | Sandbox 为什么会影响 RL 评估的可信度？ | 环境隔离、确定性、状态污染 |
| 68 | F | Verifier 输出一个分数，能保证对应同一条轨迹吗？ | Trajectory ID、异步乱序评分与去重 |
| 69 | F | 多轮任务只有终局奖励，怎样定义训练轨迹？ | Turn Boundary、Return 和训练 Token 范围 |
| 70 | F | 做一个可测试的最小 Agent 交互环 | Agent–Tool–Verifier 的数据契约和失败注入 |
| 71 | G | 为什么整批等齐不一定是算法要求？ | Group/Batch/Optimizer 的三种必要等待 |
| 72 | G | 快请求完成后，下一条数据为什么仍不能启动？ | Admission、背压与在途预算 |
| 73 | G | 什么叫有效训练 Group，何时才能接收？ | Reward Join、Group 成员、筛选与截止 |
| 74 | G | 按接受率补投，为何可能产生大量剩余请求？ | Accepted/Inflight/Target 的估计误差 |
| 75 | G | 已经达标，为何 GPU 仍在被 Rollout 占用？ | 停止准入 vs Inflight Drain |
| 76 | G | 同池和分池架构怎么公平比较？ | 资源互斥、Overlap 机会和搬运成本 |
| 77 | G | 权重发布后，哪些请求仍使用旧 Policy？ | Policy Version、原子切换与时间轴 |
| 78 | G | Partial Rollout 暂停到底保存哪些状态？ | KV、Sampling RNG、位置、工具会话状态 |
| 79 | G | 轨迹跨两版权重生成，还能计算什么 Ratio？ | Token/Trajectory Behavior 版本边界 |
| 80 | G | Staleness 不能只用“晚了一步”来描述吗？ | 版本、Step、时间和训练目标四种定义 |
| 81 | G | 旧样本重用到底会引入多大偏差或方差？ | IS Ratio/Support、裁剪与有效样本数 |
| 82 | G | 为什么异步更忙，模型反而可能学得更差？ | 训练质量、选择偏差、旧策略分布联合验证 |
| 83 | G | Sampler 重启后如何不重复消费同一条结果？ | Durable State、去重、提交位置 |
| 84 | G | 异常退出后何时可以安全恢复训练？ | Checkpoint 与 Inflight/Accepted 的一致性 |
| 85 | G | TQ 等队列在系统里应该拥有哪个事实？ | 所有权、字段版本、控制面和数据面分界 |
| 86 | G | 在途请求太多，怎样量化排空成本？ | P95/P99 长尾与浪费 GPU-time |
| 87 | G | 优化并发之前，怎样计算真正的 Step Throughput？ | 样本接收率、可训练 Token、GPU-hours |
| 88 | G | 比较同步和异步，什么对照才算公平？ | 质量与成本双轴、Staleness 分层 |
| 89 | H | Teacher 的全词表分布怎样给 Student 信号？ | Logits/KL/Temperature 与 Token Mask |
| 90 | H | Forward KL 与 Reverse KL 为什么可能学出不同分布？ | 多峰 Toy 分布、Mode Cover/Seek 反例 |
| 91 | H | OPD 与 GRPO 在同一目标里会不会互相干扰？ | 梯度方向、系数尺度与观察指标 |
| 92 | H | Teacher Offload 和 RPC 的收益谁说了算？ | 显存节省、传输字节、吞吐与延迟 |
| 93 | H | 如何判断某种异步/蒸馏论文的收益可复现？ | 作者报告 vs 本地测量、消融与实现差异 |
| 94 | H | 在一条长轨迹里同时控制 Policy Version 和 KL 怎么做？ | 可追溯采样、Teacher 版本与权重更新 |
| 95 | I | Reward 提高但任务指标下降，从哪一层定位？ | 数据/Verifier/Advantage/Policy/Serving 调试树 |
| 96 | I | NCCL Hang、CUDA OOM 和长尾等待怎么区分？ | 事件关联 ID、Rank、Profiler 证据 |
| 97 | I | 面对一个性能图，如何检验它有没有意义？ | 基线、方差、重复测量和容量规划 |
| 98 | I | 把 Toy Learner 接进最小异步数据供给链 | 端到端测试：版本、Mask、恢复与拒收 |
| 99 | I | 设计评审：只有 8 GPU，怎样选训练/推理布局？ | 显存、通信、吞吐、复杂度与备选方案 |
| 100 | I | 用一套实验解释你自己的 RL Infra 设计 | 独立复现实验、故障注入与工程答辩 |

## 选题背后的深度标准

- **10 分钟讲完的内容就不独立开课。** 例如字段命名、简单 API 导览、基本 Python 用法，在当天实验需要它时解释；不靠大量术语撑满讲义。
- **重要机制至少能回答三个追问：** 公式/成本模型从哪来？改变哪一个假设会得到错误结果？用什么最小实验区分正确实现和错误实现？如果做不到，就不是「深挖」，只是概览。
- **每个独立里程碑都带回数据。** 例如 Day10 要跑出一个真正改变参数的 Toy 更新，Day24 要能注入故障发现某个梯度错误，Day50 要能对一张多 GPU 配置给出通信量和恢复边界，Day88 要同时谈吞吐与训练质量，而非只画框图。
- **学生不替课程证明课程写得对。** 题目、预测、反例和参考推导可以直接看；练习是自主验证，不做解锁。自动化也不能因为没作答而推迟下一节。

## 论文只在关键决策点出现

REINFORCE/Policy Gradient、GAE、PPO、DeepSeekMath/GRPO、DeepSeek-R1、DAPO、GSPO、FlashAttention、PagedAttention、HybridFlow/verl 以及具体异步训练工作，会在相应失效条件、设计选择和可核验代码一起出现。旧论文不因为年代早就废弃，新论文也不因为名称新就更正确。明确区分论文作者报告、公式推导、本项目通过的测试和真实环境测量。

## 编辑边界

用户本地维护站点、交互组件、构建和工程架构；定时任务只负责已排程课程及确有价值的独立教学代码/插图。**课程文字永远从学生当前要弄清的那个问题开始，不写「第几课会讲什么」或「本套设计为什么如此安排」**；这些只属于课表，不属于讲义正文。已有互动页面 Day01–05 的功能可用作验证工具，但交互标签和本课目标不一致时必须更新设计，不要误导学生。
