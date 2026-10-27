# 一次 Reward 怎么更新 Policy？

模型采到一个动作、Verifier 给出一个分数。训练到底修改了什么？

假设只有 A/B 两个动作，A 得 1 分、B 得 0 分。模型给 A 的概率是 $p=\sigma(z)$，初始 $z=0$，所以 $p=0.5$。总体目标是提高期望回报 $J(z)=p$，解析梯度为 $p(1-p)=0.25$。

但 Trainer 通常拿不到完整分布下的所有 Reward。它拿到的是本次采样的动作。假如采到 A，使用最基础的 REINFORCE Loss：

$$
L=-R(A)\log\pi_z(A)=-\log p,\quad
\frac{\partial L}{\partial z}=-(1-p)=-0.5.
$$

学习率为 0.4 的 SGD 将 $z$ 更新到 $0.2$，新的 $p=\sigma(0.2)\approx0.549834$。这一轮采样梯度与解析期望梯度数值不一样，并不矛盾：如果采到 B，$R(B)=0$，最简单的估计器梯度为 0；两种结果等概率出现，更新方向的期望仍为 $0.25$。

## 推导成立的条件

$$
\nabla_\theta\mathbb E_{a\sim\pi_\theta}[R(a)]
=\mathbb E_{a\sim\pi_\theta}
[R(a)\nabla_\theta\log\pi_\theta(a)].
$$

它需要样本来自被求导的分布；环境 Reward 不直接依赖这些参数。对旧 Policy 的样本无修正地套用当前 Score Function，一般不满足同一个期望恒等式。基线 $b(s)$ 可以在不依赖当前 Action 的条件下减小方差，却不是“随便减一个分数都无偏”。

## 代码里要确认什么

- Reward 是否确实对应本次采样的 Action；
- **Current Logprob** 是否在可微的模型前向中计算，而非从服务侧拿到一个已 Detach 的数字；
- Action Mask 是否只包含模型生成的位置；
- Loss 前面的负号是否正确，Optimizer 是否只 Step 一次。

最小验证是固定 A、$z=0$，确认更新后 $p>0.5$。把 Loss 的负号故意写反，应得到 $p\approx0.450166$。能跑 Backward 但方向不对，是一种最基础的训练正确性故障。

可读材料：[Spinning Up 的 Policy Gradient 推导](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)；[PPO 原论文](https://arxiv.org/abs/1707.06347)。历史完整教学稿保存在 [Day 01](https://github.com/Kairosxy/rl-infra-100days-site/blob/main/courses/day-01/lesson.md)。