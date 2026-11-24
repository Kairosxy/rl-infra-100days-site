# 多轮 Agent 的时间尺度与信用分配

一个 Agent 在第三次工具调用后成功完成任务，Verifier 只返回最终得分 1。前两次调用并未得到独立标签，仍可能是成功所必需的动作。长程信用分配要解决的是如何将后续反馈分配给此前模型实际做出的决策。

## 回报沿哪个时间轴传播

若三个 assistant turn 分别作为三个决策步，奖励为 [0,0,1]，gamma=0.9，则各步 Return 为 [0.81,0.9,1]。若改为逐 Token 决策，第一轮有 100 个 Action Token、第二轮有 10 个、第三轮有 5 个；同样使用 gamma=0.9 会产生截然不同的折扣。长工具等待、环境操作和模型 Token 需要有明确的时间单位。

对 Semi-MDP，动作持续时间 d 可进入折扣，例如按基础时间单位使用 $\gamma^d$，同时累积期间奖励。若任务目标只关心最终成功率，可以选择对应的无折扣有限 episode 定义。训练框架不应在 Token 展开或 turn 拼接过程中隐式更改目标。

## 从 Outcome 到 Process

终点反馈简单、易核验，但早期决策可能面对较高的估计方差；Value/GAE 通过状态基线或 Bootstrap 改善估计。Process Reward 为中间步骤提供额外反馈，有机会降低反馈稀疏问题，同时也可能改变最优策略。

例如正确任务最终得 1 分，失败任务得 0 分。如果每次调用工具额外奖 0.2，失败轨迹只要执行六次无用调用也能累计 1.2。在不折扣的这个例子中，过程奖励已经让失败轨迹超过成功轨迹。

一种经典的目标保持形式是 potential-based shaping：

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s).
$$

在相应 MDP、相同折扣与合适终止边界条件下，累计 shaping 项逐步抵消，仅剩边界项，可以保留最优策略关系。任意 Judge 打分、过程标签或长度奖励并不自动满足这些条件。

## 轨迹中的训练位置

模型生成 Action 后，环境 Observation 是状态更新的一部分。它可以影响下一步 Value 和 logprob，却不应作为模型动作参与 Policy Gradient。若从 turn-level Advantage 展开到 Token-level，需明确长回答与短回答的归约权重。跨 episode 或截断边界使用不同的 Bootstrap 与 continuation mask，参见 [GAE](../rl/gae-bootstrap.md)。

最小对照使用一个有两条成功路径的确定性工具环境：短路径直接完成，长路径包含冗余动作。分别记录 outcome、process reward、return 和有效 Action 梯度，检查是否在奖励冗长过程或将工具响应错误归为模型行为。

参考：[Policy Invariance Under Reward Transformations](https://ai.stanford.edu/~ang/papers/icml99-shaping.pdf)、Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018), Ch. 3, 7、[Agent Action](agent-action.md)。
