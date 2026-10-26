# 一条参考实验，从策略概率走到异步 GPU 系统

课程不先创造一个复杂的 Agent Harness，也不拿一个不能运行的“大一统架构”当学习目标。第一阶段始终研究**一次可验证的策略更新**，随后逐步增加 Token、Mini-batch、GPU、分布式 Rank、Rollout、环境、异步队列和 Teacher。每次扩展前后都要有能被单元测试或可复现实验比较的指标。

## 固定的起点

一个只有两种答案 A/B 的模型。参数为一个 Logit $z$，动作 A 的概率为 $p=\sigma(z)$，Reward(A)=1、Reward(B)=0，最初 $z=0$。它既足够小到能手算，也包含大模型中始终存在的策略分布和更新机制。

- 按当前策略采样 A 后，$L=-\log p$，$\partial L/\partial z=p-1=-0.5$。SGD 学习率 0.4 后 $z'=0.2$，$p'≈0.549834$。
- 采样 B、Reward=0 时，用最简单的无 Baseline 估计器会出现零梯度。正确的期望梯度却是 $p(1-p)=0.25$。这直接区分“理想的期望”和“有限样本估计”。
- 将动作扩展成二 Token 的 Response，Token 的对数概率相加，哪些位置包含 Reward 权重由明确的 Action Mask 决定。再加 Old/Current 概率，才能讨论 Ratio、Clip 和样本重用。
- Group、Value、Verifier、Policy Version 等结构都在需要比较它们的实验中引入。它们不是第一次上课就要背的一张 Schema 表。

## 知识依赖图与代码落点

~~~mermaid
flowchart TD
  A["z → p=σ(z)"] --> B["采样动作 a"]
  B --> C["R(a) 和 logπ(a)"]
  C --> D["Policy Gradient"]
  D --> E["Gradient Accumulation / Optimizer Step"]
  E --> F["GPU 与多个 Rank"]
  F --> G["Rollout Serving / KV"]
  G --> H["Agent 与环境状态"]
  H --> I["Sampler / Queue / Async"]
  I --> J["Staleness、Correction、OPD"]
  J --> K["验证：质量、性能、恢复"]
~~~

这是学习系统的演化图，不表示每次更新都必须以相同顺序串行执行。真实异步实现可能重叠 Rollout、Score、Teacher 和 Train。

## 数学公式怎样落到代码与诊断

| 数学对象 | 最小实现/扩展形状 | 在哪出现 | 能验证什么 | 错误的迹象 |
|---|---|---|---|---|
| $z$、$p=\sigma(z)$ | 标量或 `logits [B,T,V]` | Current Forward | 概率归一化、梯度方向 | 更新后目标 Action 概率反向变化 |
| $\log\pi(y_t|x,y_{<t})$ | `action_log_probs [B,T]` | Teacher-forced Forward | Token Shift、Action Mask | Shape 对但优化错位 Token |
| $R$、$A$ | `reward [B]`、`advantages [B,T]` 或按组结构 | Verifier/Estimator | 状态或 Group 对应关系 | Reward Mean 正常但梯度异常 |
| $\rho=\exp(\log\pi-\log\pi_{\rm old})$ | `ratio [B,T]` | Loss | Ratio 分布、ClipFraction | 极大 Ratio / Logprob Drift |
| $\nabla_\theta L$ | 参数梯度或分片梯度 | Backward/Reduce | Reduction 和同步一致 | 单卡/多卡更新无法对齐 |
| `policy_version` | 轨迹字段、版本表 | Rollout/Queue/Train | Behavior、Old、Current 区分 | Staleness 被消费时的版本覆盖 |
| `accepted / inflight` | Sampler 状态 | 数据供给/恢复 | 达标 vs 排空、幂等 | 重复消费、错误补投 |

本表字段是**课程示意**，不能直接声称是任一版本的 verl/SGLang 原生 API；落实源码时必须锁定 Commit 与真实字段。

## 把成本也算进判断

第一阶段 toy 只有一个浮点参数，单次更新的算术量与内存是 $O(1)$，没有 GPU/NCCL 通信。这个玩具用来判断算法正确性，不能用 CPU 的运行耗时预测 LLM 训练吞吐。

加入 Token 后，朴素地对完整词表计算单次响应的 Log-softmax，有 $O(TV)$ 的词表维度工作量；长度为 $T$、词表大小为 $V$ 的概率结果是否被物化，取决于具体 Kernel 实现。以 $B=2,T=4,V=32000$ 为例，一份 BF16 `[B,T,V]` Tensor 需要 $2×4×32000×2=512000$ Byte（约 0.488 MiB）。这只是这份 Tensor 的体积，不是 Transformer 前向、激活或整个进程峰值。

进入分布式阶段，再依次增加参数/优化器状态、Saved Activations、Collectives 与队列的生命周期。DDP All-reduce 的通信下界、TP/EP 的跨卡交换，以及 Serving KV Cache 的容量都必须带单位和网络/拓扑假设，不能简单用“多卡更快”替代成本模型。

## 问题排查不是按照工具品牌分类

~~~mermaid
flowchart TD
  S["观察到训练异常"] --> Q{"哪一类证据先出现异常？"}
  Q -->|"Loss / Reward 方向错"| A["固定同一条样本，检查 Reward、Mask、Ratio、Gradient Sign"]
  Q -->|"NaN / Drift"| B["定位首个非有限值；核对采样变换、精度、Policy Version"]
  Q -->|"吞吐变慢"| C["区分 Sampling、Queue、GPU Compute、Collective 的耗时"]
  Q -->|"OOM / Hang"| D["查看 Saved Tensor/缓冲区峰值，或 Rank Collective 顺序"]
  A --> E["用最小反例复现，再修正目标/数据"]
  B --> E
  C --> F["成本模型与对照负载，再优化调度"]
  D --> F
~~~

一个现象可能跨多层出现，例如 Serving Logprob Drift 可同时涉及 Tokenizer、Sampling Temperature、低精度和权重版本。排查时先确认比较的是同一个随机变量/同一条样本，再量化差异；不要一上来就宣称 FP8 或 NCCL 是根因。

## 三个穿过课程的验证实验

**实验一：数值正确性。** 固定参数 $z=0$，强制采样 A，直接核对更新前后概率以及 Autograd 梯度；故意去掉 Loss 前的负号，测试应发现目标动作概率下降。扩展 Token 后加入错一位 Shift 的负例。

**实验二：单机与多卡等价。** 用同一组不等长样本，比对 Full Batch Gradient 与 2/2/1 Micro-batch 的结果；故意把局部 Mean 等权平均，测试必须失败。等价目标明确后再加入 DDP 同步与不同有效 Token 数。

**实验三：时间与状态正确性。** 人工构造两个 Group 的完成时间、Reward 乱序与两个 Policy Version，先比较同步整批等待和逐组过滤的时间线；再验证当目标样本数达标时，队列中的 Inflight 是否被正确 Drain，恢复后是否重复消费。不能以这一个 Toy 时间线直接声称真实系统吞吐提升。

## 关键资料与工程复现原则

- [Policy Gradient / REINFORCE](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)、[GAE](https://arxiv.org/abs/1506.02438)、[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)：优化目标、估计器和更新边界。
- [DAPO](https://arxiv.org/abs/2503.14476)、[GSPO](https://arxiv.org/abs/2507.18071)：按失效现象研究采样选择、Ratio 和序列长度，不做方法名词竞赛。
- [FlashAttention](https://arxiv.org/abs/2205.14135)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[verl](https://github.com/verl-project/verl)、[SGLang](https://github.com/sgl-project/sglang)：Kernel、内存、并行与数据路径。

推导、通过的测试、作者报告和本地真实测量必须分开标注；性能数据若没有设备、模型、负载、提交版本与比较口径，只能作为待验证假设。课程记录属于学生，任何练习状态都不决定后续课件能否发布。
