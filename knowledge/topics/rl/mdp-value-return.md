# MDP、Return 与价值函数

强化学习优化的不是某一次反馈，而是策略在环境中产生的累计回报。状态 $s_t$、动作 $a_t$、奖励 $r_t$、下一状态 $s_{t+1}$ 构成一次转移。MDP（Markov Decision Process）假设给定当前状态与动作后，环境转移分布无需再依赖完整历史；若观测不足以确定状态，则需要考虑 POMDP 和历史信息。

## Return、Value 与 Q

取三次转移奖励 [0,0,1]，折扣 $\gamma=0.9$，则从三个动作位置出发的 Return 分别为 [0.81,0.9,1]。对有限轨迹，$G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k$。终点得分和传播到前面动作的 Return 是两个概念。

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],\quad
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

Value 是按当前策略继续执行的期望回报；Q 则固定第一个动作。假设一个状态有两个动作，A 的 Q=1、B 的 Q=0，策略等概率选择，则 V=0.5，Advantage $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ 分别为 +0.5 和 -0.5。这里的负 Advantage 不代表实际 Reward 为负，而表示低于策略在该状态的平均预期。

## Bellman、Monte Carlo 与 TD

价值函数满足递推关系：

$$
V^\pi(s)=\mathbb E_\pi[r_t+\gamma V^\pi(s_{t+1})\mid s_t=s].
$$

Monte Carlo 使用采完的实际 Return 作为目标；TD 使用已知的即时 Reward 与下一状态的估计 Value 构造目标，即 Bootstrap。前者需要后续样本，后者可更早学习，但下一状态估计错误会影响目标。DQN 等 value-based 方法围绕 Q 学习与动作选择；Policy Gradient 直接参数化策略；Actor-Critic 同时使用策略和价值估计。SAC 等连续控制方法还有熵正则、off-policy replay 等机制，不是简单把 PPO 的 Token 换成实数。

## LLM 与环境的对应关系

单轮文本 RL 可把已生成前缀视作状态，下一 Token 视作动作，最终 Verifier 给终点奖励。多轮 Agent 还会收到非模型生成的工具结果；环境真实状态可能没有完整出现在上下文里，不能无条件假设 Token 前缀就是充分状态。若任务只关心整个答案的成败，常用序列级反馈或 Group-relative Advantage；若需要逐位置价值，则要定义终止、截断与 Bootstrap。

下一篇：[GAE 与终止/截断](gae-bootstrap.md)，以及 [Policy Gradient](policy-gradient.md)。

参考：Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), Ch.3–6；[Spinning Up: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)。
