# Day 04 · 图解：Reward 不可微，Logprob 为什么可微

![Reward、采样 Token 与可微 Logprob 路径](assets/policy_gradient_flow.svg)

沿图中的**两条性质不同的路径**读：Rollout 产生离散 Token，Verifier 返回 Reward；这条评分链路通常没有普通的 Autograd 路径。Trainer 把已经采样好的 Token 固定为 Action，用当前模型的 Teacher Forcing Forward 重新计算它们的 Logprob，再让 $-A\log\pi_\theta(a)$ 的梯度进入参数。

这张图表达的是 **基础 Policy Gradient**，不包含完整 PPO/GRPO 的 Ratio、Clip、组内归一化或 KL 项。不要把简式 Loss 当成所有 RL 训练器的实现。

图里没有画 Token Index。把图映射到实现时还需要确认：`logits[:,t,:]` 预测 `token_ids[:,t+1]`，Mask 在这些预测位置上定义，Reward/Advantage 必须对应同一条 Trajectory。索引或身份错了，梯度路径仍可能有效，却在优化错误的动作。
