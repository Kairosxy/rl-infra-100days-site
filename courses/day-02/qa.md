# Day 02 · 问答与追问

## 题 1：Reward 不可微，为什么梯度能穿过 Logprob？

从 $J(\theta)=\sum_a\pi_\theta(a)R(a)$ 开始推导 REINFORCE，不省略 $\nabla p=p\nabla\log p$ 的步骤。说明对环境转移和 Reward 计算作了哪些假设。

**判别依据：** 能说出采样梯度是在 Policy 分布下的期望，Reward 只作为权重，当前参数下采样与求 Logprob 是有条件的。

**追问：** 如果数据从旧策略采样，但仍按 Current Policy 的 Score 直接求平均，原证明还成立吗？为何需要重要性加权或新的估计器假设？

## 题 2：Baseline 为什么有时方差下降，有时完全毁掉估计器？

固定 $p=0.25$、Reward 为 [1,0]，写出 $g_b(A)$、$g_b(B)$。计算 $b=0$、$b=0.75$ 的期望和方差，解释最优 Baseline 为什么不是本例平均 Reward=0.25。

**判别依据：** 写出 `g_b(A)=0.75(1-b)` 和 `g_b(B)=0.25b`；期望恒为 0.1875；$b=0.75$ 两者相等，方差 0。

**追问：** 让 Baseline 依赖当前采样动作并设 $b(a)=R(a)$ 会发生什么？
