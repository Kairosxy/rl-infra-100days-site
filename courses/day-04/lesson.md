# Day 04 · PPO Clip 约束了哪一段梯度，为什么它不是 KL 的保险丝？

前几课已经把期望 Reward、采样梯度、Token Logprob 连起来了。现在一个自然的问题是：如果我们把同一批 Rollout 样本反复拿来做多轮梯度更新，**用于生成样本的旧策略与正在更新的新策略会越来越不一样**，继续使用原来的梯度估计会发生什么？

PPO 用一个易实现的 Clipped Surrogate 约束部分更新方向。难点不在于背公式，而在于看清：**正负 Advantage 截断的区间不同，Clip 后哪些样本没有梯度，以及这个目标没有保证什么**。

[互动 Lab 04：拖动 Ratio，看哪一侧梯度被截断 →](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=4)

## 先看一个 Action 的 Ratio

对同一状态 $s$ 和已采样动作 $a$，设 Old Policy 固定为 $\pi_{\mathrm{old}}$，Current Policy 为 $\pi_\theta$。重要性比率：

$$
r_t(\theta)
=\frac{\pi_\theta(a_t|s_t)}
{\pi_{\mathrm{old}}(a_t|s_t)}
=\exp\bigl(\log\pi_\theta(a_t|s_t)-\log\pi_{\mathrm{old}}(a_t|s_t)\bigr).
$$

例如已采样动作 A 在旧策略下概率为 0.2、新策略下为 0.3，则 Ratio 是 $1.5$。这意味着新策略给已采样动作更高的概率，**并不说明整条轨迹的真实期望 Reward 按比例提升 1.5 倍**。

PPO 的常见 clipped surrogate 对每个动作的 Advantage $A_t$ 写为：

$$
J_t^{\mathrm{clip}}(\theta)
=\min\!\left(
r_t A_t,\ 
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon) A_t
\right).
$$

Trainer 通常最小化负的 Surrogate，加上所选实现的 KL/Entropy/Critic 等项。下面只研究这个单动作 Surrogate；完整 PPO 是多个 Token/Step 的统计目标。

## 正 Advantage：超过上界时，不再继续鼓励这个 Action

固定 $A=+2$、$\epsilon=0.2$，允许区间为 $r\in[0.8,1.2]$。

当 $r=1.5$：

$$
rA=1.5\times2=3,\qquad
\operatorname{clip}(r)A=1.2\times2=2.4,
$$

所以 $J^{\mathrm{clip}}=2.4$。如果 $r$ 继续从 1.5 增大到 1.8，这个样本对 $r$ 的目标值保持 2.4，**局部导数为 0**（在分段边界不讨论不可微点）。

这个单样本不再推动已被明显提高概率的正 Advantage 动作。它不是说别的样本、别的损失项或共享参数再也不能改变该动作的概率。

如果 $r=0.6$ 呢？$rA=1.2$，截断项为 1.6，取更小的 1.2，梯度不为零。**正 Advantage 的左侧下降没有被同样截断**，因为目标希望能纠正被压低的好动作。

## 负 Advantage：下界另一侧才出现平台

让 $A=-2$、$\epsilon=0.2$。当 $r=0.6$：

$$
rA=-1.2,\qquad
\operatorname{clip}(r)A=0.8\times(-2)=-1.6.
$$

取最小值 $-1.6$，继续压低 $r$ 并不会让这个样本的 clipped surrogate 再变化。这次平台在**左侧** $r<0.8$。

| Advantage | 未饱和的 Surrogate | 哪一侧被 Clip 成平台 |
|---|---|---|
| $A>0$ | $rA$ | $r>1+\epsilon$ |
| $A<0$ | $rA$ | $r<1-\epsilon$ |

如果 $A=0$，不论 Ratio 怎样，当前样本的 Surrogate 都是 0；这不是 PPO 的 bug，而是 Advantage 估计器没有给这条样本提供方向。

把表格死记下来还不够。拖动 [Lab 04](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=4) 的 Ratio 并来回切换 Advantage 的符号。你的目标是**先预测平台在左边还是右边**，再检查曲线和数值。

## 为什么「Clip=0.2」不等于「策略 KL 一定很小」

Surrogate 对**已采样 Action 的 Ratio** 分段截断。它没有对整个词表、所有状态或下一轮采样产生硬约束。

构造一个极小的二动作分布：

$$
\pi_{\mathrm{old}}=[0.5,0.5],\qquad
\pi_{\mathrm{current}}=[0.9,0.1].
$$

动作 A 的 Ratio 已经达到 $0.9/0.5=1.8$，对正 Advantage 的 A 样本，其 clipped surrogate 进入平台。但两个完整分布的 Forward KL 仍为：

$$
D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_{\mathrm{current}})
=0.5\log\frac{0.5}{0.9}
+0.5\log\frac{0.5}{0.1}
\approx0.5108.
$$

这里 KL 使用自然对数，单位是 nats。即使 A 样本已经被 Clip 住，**整个策略仍可能与 Old 有明显差异**，因为未观测动作、其他状态和共享参数都会变化。PPO Clip 不能单独作为全局 Trust Region 或 KL 上界的数学证明。实际训练需要继续监控 KL、Clip Fraction、Entropy 等指标，并明确它们的计算分布与估计口径。

注意：这个例子只是证明「单个 clipped surrogate 平台不蕴含全分布的 KL 界」，**不是一次 PPO 训练真的运行到了该分布**。

## 几个经常被混在一起的实现选择

**Old 和 Behavior 不一定始终相同。** 如果每次优化都使用刚由 Old 采出的 on-policy 数据，二者在那次采样中可以一致。如果样本在异步队列中跨过权重更新、使用不同的 Sampling Temperature/Top-p，情况就复杂了。Ratio 分母到底是什么分布必须由算法定义，不能看到 `old_log_probs` 字段就认定它与真实 Behavior 相等。

**Ratio 的粒度很重要。** 这是一个 Step/Token 级目标；对于自回归序列，直接把所有 Token Logprob 相加再 exponentiate，会得到 Sequence Probability Ratio，和逐 Token Ratio 的数值范围及裁剪效果完全不同。GSPO 等工作恰恰把这层差异作为算法设计点，而非普通实现优化。

**Reduction 仍然属于目标函数。** Day 03 的 Token Mean、Sequence Mean 改变了每条回答占到的梯度权重。把 PPO Clipping 实现正确，却在末尾改变了长度归一化，依旧可能得到不同的训练行为。

## 一个不依赖 GPU 的错误判断练习

设 $\epsilon=0.2$，请先手算：

1. $A=+2,r=1.5$：结果 2.4，局部 Ratio 导数 0。
2. $A=-2,r=0.6$：结果 -1.6，局部 Ratio 导数 0。
3. $A=+2,r=0.6$：结果 1.2，局部 Ratio 导数 +2。
4. $A=-2,r=1.5$：结果 -3.0，局部 Ratio 导数 -2。

实验页面会画出完整分段折线并标出当前位置。若把损失 `loss=-Jclip` 误写成 `loss=+Jclip`，所有梯度更新方向将反过来；这是值得写进测试的失败例子。

你现在应该能回答：**为什么 PPO 的同一段 Clip 规则，对坏动作和好动作必须产生不同的梯度平台？** 下一课把单条动作变为一个 Prompt 下多条回答，看看 GRPO 如何构造 Advantage，以及为什么并不是「PPO 去掉 Critic 就结束了」。

## 核心资料

- [PPO：Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347)。本课使用其 clipped surrogate 结构，并未把它当成严格 KL 约束。
- [OpenAI Spinning Up：PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)，用于对照 Clip、Target KL 和实验观测的不同角色。
- [GSPO：Group Sequence Policy Optimization (2025)](https://arxiv.org/abs/2507.18071)，用于后续比较 Token/Sequence Ratio 的选择。
