# Day 02 · 同一个模型，为什么有时没有梯度，有时突然更新很大？

还是那道只有两个答案的题：A 正确得 1 分，B 错误得 0 分。模型只有一个 Logit 参数 $z$，A 的概率是 $p=\sigma(z)$。

假设模型现在做对的概率仅有 **25%**。你把一条样本放进 Trainer，输出的梯度是 0。第二次运行同一套代码，梯度突然变成 0.75。再运行十次，大部分时候还是 0。

代码不一定有 Bug。这可能是**采样梯度的方差**：Trainer 优化的是期望奖励，但每次只能看到很少的随机结果。

## 为什么单次梯度可以完全不等于目标梯度？

固定 $p=0.25$，奖励 $R(A)=1,R(B)=0$。我们真正希望增加的目标是：

$$
J(z)=p,\qquad \frac{\mathrm dJ}{\mathrm dz}=p(1-p)=0.1875.
$$

用 REINFORCE 采样梯度时，对于具体动作 $a$：

$$
\hat g(a)=R(a)\,\frac{\partial\log\pi_z(a)}{\partial z}.
$$

Bernoulli 策略的 Score Function 为：

$$
\frac{\partial\log\pi_z(a)}{\partial z}
=\mathbf1[a=A]-p.
$$

只有两种结果：

| 采样结果 | 发生概率 | Reward | $\hat g$ |
|---|---:|---:|---:|
| A | 0.25 | 1 | $1(1-0.25)=0.75$ |
| B | 0.75 | 0 | $0(0-0.25)=0$ |

二者的期望是

$$
\mathbb E[\hat g]
=0.25\times0.75+0.75\times0
=0.1875.
$$

**“平均正确”不代表每一次都正确。** 如果一个 Batch 只采了四次，全部没抽到 A 的概率是 $0.75^4\approx31.64\%$。这轮的样本梯度可能全部为零，但它不是对理想目标梯度的证明。

方差可以精确算：

$$
\begin{aligned}
\operatorname{Var}(\hat g)
&=\mathbb E[\hat g^2]-(\mathbb E[\hat g])^2\\
&=0.25(0.75)^2-(0.1875)^2\\
&=\boxed{0.10546875}.
\end{aligned}
$$

单样本梯度标准差约 0.3248，比其期望 0.1875 还大。增加独立样本数可以缩小**均值估计的方差**：假设独立同分布，$N$ 条样本平均值的方差是 $0.10546875/N$。若模型参数不断更新或采样相关，这个 $1/N$ 的简单结论就需要重新检查前提。

## 为什么减去一个常数，期望梯度不会变？

直觉上，可以对每条 Reward 减去一个基准 $b$，只把“比基准好还是差”的部分用于学习：

$$
\hat g_b(a)=(R(a)-b)\,
\frac{\partial\log\pi_z(a)}{\partial z}.
$$

这听起来像在篡改 Reward。它什么时候合法？关键是 $b$ **不能根据同一个状态下本次采样的动作再改变**：

$$
\begin{aligned}
\mathbb E_a[b\nabla_z\log\pi_z(a)]
&=b\sum_a \pi_z(a)\nabla_z\log\pi_z(a)\\
&=b\sum_a\nabla_z\pi_z(a)\\
&=b\nabla_z1=0.
\end{aligned}
$$

因此：

$$
\mathbb E[\hat g_b]=\mathbb E[\hat g]=0.1875.
$$

这就是 Baseline 的价值：不用改变期望优化方向，却可能显著改善采样效率。严格说，它并非保证**任意** $b$ 都减少方差；一个很差的 $b$ 可以让方差更大。

## 这个例子里，平均 Reward 竟然不是最好的 Baseline

固定 $p=0.25$，可把两种结果显式写出来：

$$
\hat g_b(A)=0.75(1-b),\qquad
\hat g_b(B)=0.25b.
$$

若 $b=0$，它们分别是 0.75 和 0；若 $b=0.25$（等于平均 Reward），它们变成 0.5625 和 0.0625；若 $b=0.75$，两种结果都为 **0.1875**。

在这个特殊的二动作、确定性 Reward、单参数模型里，$b=0.75$ 恰好使方差降为零。可以不背任何定理，直接从“让两种可能的梯度取值相等”得到：

$$
0.75(1-b)=0.25b
\quad\Rightarrow\quad b^*=0.75.
$$

也可以从一般的常数 Baseline 最小方差条件求出：

$$
b^*=
\frac{\mathbb E[R(a)\|\nabla\log\pi(a)\|^2]}
{\mathbb E[\|\nabla\log\pi(a)\|^2]}.
$$

**平均 Reward 是 0.25，最小方差 Baseline 是 0.75。** 这说明学习 Baseline 不能只背“减去平均 Reward”。

这个结果不代表真实 LLM 可以将方差降到零。高维模型的不同参数对“最优基准”的要求可能不同，Reward 也可能有环境噪声和多步依赖。

## 两个反例让你判断什么时候会出 Bug

**反例一：Baseline 由当前采样动作计算。**

如果你写 `b(a)=R(a)`，每条样本都满足 $R(a)-b(a)=0$，梯度恒为零。虽然表面看起来“相对 Reward 为零”，但刚才保证期望不变的证明已经失效：你不能把依赖本次 Action 的 $b(a)$ 拿到动作求和之外。

**反例二：数据来自另一个策略，却用当前策略的无修正估计器。**

刚才的等式假定 $a\sim\pi_z$。若行为策略采到 A 的概率为 0.9，而用于求导的当前策略仍是 $p=0.25$，直接平均 $R(a)\nabla\log\pi_z(a)$，期望会变成 $0.9\times0.75=0.675$，**不再等于当前目标的 0.1875**。

不能因为某个字段叫 `log_prob` 就认定样本来自当前分布。要先检查生成权重、采样变换，以及概率到底属于哪个 Policy。

## 自己画出方差，而不是靠记公式

打开 [Baseline 交互实验](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=2)。设置 $p=0.25$，从 $b=0$ 缓慢拉到 0.75，观察两种抽样动作对应的梯度点怎样靠拢。

再把 $b$ 调到 1.5，检查方差是否总在下降。最后将 $p$ 改成 0.6，预测这时本例能使两种估计值相等的 $b^*$ 是否仍然是 0.75。

对于任意 $p$，由 $(1-b)(1-p)=bp$ 得到 $b^*=1-p$。当 $p=0.6$，它是 **0.4**。你应能根据公式预测交互曲线，而不是先看 UI 才知道答案。

如果要写成代码，针对本例最有价值的不是造一个庞大 Trainer，而是模拟 10,000 次采样，按同一套 $p$ 和 $b$ 估计经验方差，并与解析方差比较。样本量增大时，经验估计通常更稳定，但单次结果不保证单调。

## 资料

- [Williams (1992), REINFORCE](https://link.springer.com/article/10.1007/BF00992696)：采样动作梯度估计的基础。
- [OpenAI Spinning Up：Policy Gradient](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：Log-Derivative Trick、Baseline 的适用条件。
- [Greensmith et al., Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning (2004)](https://www.jmlr.org/papers/v5/greensmith04a.html)：为什么 Baseline 的最小方差条件不只是简单减去平均 Reward。

把本课完整走通后，一个“有时候没梯度”的训练现象就至少有两个完全不同的解释：**采样本来就没有提供有效信号，或者代码把原本有效的梯度错误地抵消掉了。** 前者属于估计器统计特性，后者需要修改实现。
