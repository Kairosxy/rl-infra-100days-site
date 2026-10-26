# RL Infra · 把难问题放到正确的位置

这套课程服务于已经接触过 RL / GPU / Distributed Training、但希望把概念重新连成一套可推导、可实现、可调试系统的人。按日发布，不按答题状态解锁。文档、源码和互动实验是不同的学习媒介：文章解释机制，实验验证反例，交互页面允许实时更改假设。

## 为什么重新编排

旧版 Day 01 的大部分内容是 Trajectory 字段和组件职责，这些是项目开发时必须有的工程契约，但不需要一整天的教学时间。旧版随后突然进入 Transformer 显存和 AdamW，直到 Day 04 才解释 RL 怎样产生梯度。对学习者来说，这造成两个断点：**还不知道在优化什么，就开始讨论如何分布式优化；还没有明确梯度估计器的假设，就直接看到复杂的框架名词**。

新版先沿优化数学建立连续的因果链：目标函数 → 如何估计梯度 → LLM Token 级计算 → 为什么要限制 Policy Update → 组内相对奖励与数据分布的联系。只有这条线通了，再讨论 Learner 的显存、梯度累积、分布式拓扑和 Serving 调度。

## 知识分层：学会、讲透、做出来

| 层次 | 用什么讲 | 什么叫真正学会 |
|---|---|---|
| **随用随查**（不是独立的一整天） | Trajectory/Reward 的最小 Schema、命名、普通数据结构、优化器参数意义 | 读得懂代码和实验配置，能找出对应文档 |
| **机制理解** | 有输入、数值中间步骤、确定输出的直觉实验 | 不看答案能预测更改一个参数后的后果 |
| **深入推导** | 公式从哪来、隐含假设、梯度/概率/Mask 的对应、反例 | 能写对前提，推导结果，指出不适用场景 |
| **工程证明** | 可运行测试、性能模型、日志/Trace、源码路径、论文消融 | 能设计一组反例把错误实现与正确实现区分开 |

有些简单概念只讲几段，例如 `trajectory_id` 按 ID Join，而不是整节课填满流程图。复杂主题不再仅写一句结论：例如 PPO 必须讲正负 Advantage 的分段梯度与 Clip 的局限；GRPO 必须讲完整组、零方差、长度权重与 Selection Bias；异步调度必须讲数据完整性、权重版本和资源等待的可追踪条件。

## 每个阶段的主问题

- **Day 01–14 · 策略优化与估计器**：到底在优化哪个目标？不同采样、基线、Ratio 和 Group 归一化，什么时候改变梯度？
- **Day 15–26 · Learner 的实际计算**：同一个目标怎样被 Autograd、Micro-batch、AMP、Checkpointing 等实现保持或破坏？
- **Day 27–42 · GPU 与分布式训练**：算子形状、FLOPs、HBM、Collective、不同并行组怎样决定 Step Time 和峰值？
- **Day 43–56 · 推理与 Rollout**：Prefill/Decode、KV Cache、Scheduler 和采样分布怎样影响训练数据与 P99？
- **Day 57–68 · Agentic RL 数据生产**：Action、Observation、Verifier、Group 与队列背压怎样形成可训练样本？
- **Day 69–85 · 异步、Off-policy 与 OPD**：改变调度和权重更新节奏时，怎样证明算法语义没有悄悄变？
- **Day 86–94 · 可靠性与容量设计**：如何定位跨 GPU/Rank/Actor 的故障，并给出可量化的缓解方案？
- **Day 95–100 · 研究复现与系统面试**：能否区分论文主张、实际实现、可复现实测与自己的设计选择？

详细每日题目见 [100 天课表](day_schedule.md)；同一参考系统的阶段演化见 [系统主线](system_roadmap.md)。

## 第一周以同一个二动作模型推进

先假设动作 A 的概率为 `p=sigmoid(z)`，奖励规则为 A 得 1、B 得 0。Day 01 直接计算期望 Reward 的导数；Day 02 用 Score Function 从采样估计梯度，并检验 Baseline 的方差；Day 03 将一个动作推广到自回归 Token，学习 Shift/Mask/Reduction；Day 04 引入旧策略概率和 Clipped Surrogate；Day 05 使用多个答案做 GRPO 组内 Advantage。公式每变一次，都问一句：**这是数学目标变化，还是系统实现错误？**

[进入交互实验室](https://kairosxy.github.io/rl-infra-100days-site/labs/)：每个 Lab 可以先预测，再调节变量，查看与推导对应的曲线或状态。它不是自动判分系统，也不会联网记录个人作答。

## 论文怎样进入课程

经典工作用于建立定义，新工作用于提出反例或工程取舍。按作者实际报告写明“论文提出/报告”；没有重复实现或对照实验时，不把公开报告的性能增益当作本课程实测。

- [REINFORCE (1992)](https://link.springer.com/article/10.1007/BF00992696)、[GAE (2015)](https://arxiv.org/abs/1506.02438)、[PPO (2017)](https://arxiv.org/abs/1707.06347)：梯度估计与受约束更新的基础。
- [DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300)、[DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948)、[DAPO (2025)](https://arxiv.org/abs/2503.14476)、[GSPO (2025)](https://arxiv.org/abs/2507.18071)：组内学习信号、采样策略、Ratio 颗粒度与稳定性。
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256)、[AReaL](https://github.com/areal-project/AReaL)、[DORA (2026 预印本)](https://arxiv.org/abs/2604.26256)：系统与算法语义在异步训练中的相互制约。
- [FlashAttention](https://arxiv.org/abs/2205.14135)、[PagedAttention](https://arxiv.org/abs/2309.06180)：底层 IO 与 KV 内存生命周期。

## 课程文件与维护边界

单日材料按需要组织为 `lesson.md`（完整推导）、`recap.md`（记忆中的关系）、`qa.md`（错误情境追问）和可选 `diagram.md`（静态图或交互指引）。不同机制使用不同结构，不强行每课配一张图片。严禁把过时的静态图当作新版内容的示意图。

用户本地维护仓库应用与网站代码，定时任务默认只新增/完善当天的课件、示例和必要配图。课程允许跳课；课程更新不需要个人答案，也不把自动生成的“进度”当作真实掌握情况。

