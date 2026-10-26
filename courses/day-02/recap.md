# Day 02 · 复算卡

单样本的 Policy Gradient 是「Reward/Advantage 乘以已采样动作的 Score」：

$$
\nabla J = \mathbb E[R(a)\nabla\log\pi(a)].
$$

二动作 `p=0.25`、`R(A)=1,R(B)=0`：A 的 Score 为 0.75，B 为 -0.25。没有 Baseline 时采样梯度分别为 $0.75,0$；其期望是 **0.1875**，方差是 **0.10546875**。

当 $b=0.75$，两种抽样的估计值都成为 0.1875，方差为零。这是二动作、确定性 Reward 的特例；最优常数 Baseline 不一定等于平均 Reward。

**最容易误用的条件**：Baseline 不能依赖同一状态下本次采样的 Action。设 `b(a)=R(a)` 会让估计器全部归零，破坏对真实梯度的估计。

[调整 Baseline，观察期望不变但方差变化](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=2)。
