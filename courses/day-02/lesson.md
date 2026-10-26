# Day 02 · 采样只有一个动作，怎么得到策略梯度？

昨天算出目标 $J(z)=p$ 和精确梯度 $p(1-p)$。这个结果依赖我们能列出 A/B 两个动作的 Reward；真正的 LLM 一个 Response 可能有上百个 Token，所有可能轨迹无法穷举。Trainer 拿到的通常是**从当前 Policy 采出的有限几个动作和 Verifier 分数**。

今天的问题是：没有枚举整个动作空间，能否用采样结果估计 $\nabla J$？估计值为什么波动？Baseline 为什么能改方差但不能随便依赖 Action？

[互动 Lab 02：试着把梯度估计的方差压到零 →](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=2)

## 从一个恒等式推导 REINFORCE

离散动作空间内：

$$
J(\theta)=\sum_a\pi_\theta(a)R(a).
$$

假设环境和 Reward 规则本身不直接依赖 Policy 参数，交换求和与求导：

$$
\begin{aligned}
\nabla_\theta J
&=\sum_aR(a)\nabla_\theta\pi_\theta(a)\\
&=\sum_a\pi_\theta(a)R(a)\nabla_\theta\log\pi_\theta(a)\\
&=\mathbb E_{a\sim\pi_\theta}
\bigl[R(a)\nabla_\theta\log\pi_\theta(a)\bigr].
\end{aligned}
$$

第二行只使用 $\nabla_\theta\pi=\pi\nabla_\theta\log\pi$。**Reward 不需要可微**，但我们必须能在当前参数下计算采样动作的 Logprob 并对其求导。这个乘积称为 Score Function 估计器。

对于昨天的 Bernoulli Policy，令 $I_A=\mathbf1[a=A]$，则

$$
\frac{\partial\log\pi_z(a)}{\partial z}=I_A-p.
$$

从当前分布采到一次动作 $a$ 后，单样本估计值是

$$
g(a)=R(a)(I_A-p).
$$

它有时为零，甚至可能与真正期望梯度相差很远；**无偏**只意味着从正确的采样分布反复采样，其平均值趋向目标梯度。不是每条轨迹都给出正确方向，也不是数据越旧越依然无偏。

## 手算一次：估计器正确，但抖动很大

继续固定 $p=0.25$、$R(A)=1,R(B)=0$。

| 抽到的动作 | 概率 | $g(a)$ |
|---|---:|---:|
| A | 0.25 | $1(1-0.25)=0.75$ |
| B | 0.75 | $0(0-0.25)=0$ |

因此：

$$
\mathbb E[g]=0.25\times0.75+0.75\times0
=0.1875.
$$

与昨天的精确梯度一致。可如果只训练一次：75% 的概率抽中 B，梯度估计是 0；25% 的概率抽中 A，估计却是 0.75。

把波动写成方差：

$$
\operatorname{Var}[g]=\mathbb E[g^2]-(\mathbb E[g])^2
=0.25(0.75)^2-(0.1875)^2
=0.10546875.
$$

这说明了为什么只画一条 Reward 曲线或拿单个 Batch 验证算法很危险：**期望方向正确，有限样本方差仍可能很大。**

## Baseline 到底凭什么不改变期望？

把估计器改成

$$
g_b(a)=(R(a)-b)(I_A-p).
$$

只要 $b$ 对给定状态下的采样 Action 不产生依赖，且采样仍来自用于求导的 Policy，Baseline 引入项的期望就为：

$$
\mathbb E_a[b\nabla_\theta\log\pi_\theta(a)]
=b\sum_a\nabla_\theta\pi_\theta(a)
=b\nabla_\theta 1=0.
$$

因此 $\mathbb E[g_b]=\mathbb E[g]$。注意，这是关于**梯度期望**的等式；对于用同一 Batch 训练出的 Baseline、有限样本的归一化或跨版本采样，需要进一步检查统计依赖和估计器偏差。

本例还可以准确算出一个方差最小的常数 Baseline。若把 $b$ 看成可调常数：

$$
g_b(A)=(1-b)\times0.75,\qquad
g_b(B)=(0-b)\times(-0.25)=0.25b.
$$

要求两种抽样结果的梯度值相同，就有

$$
0.75(1-b)=0.25b\quad\Longrightarrow\quad b=0.75.
$$

此时 $g_b(A)=g_b(B)=0.1875$，方差竟然为 **0**。这只是只有两个确定性动作的特殊玩具环境，不能据此认为真实 LLM 找到一个 Baseline 就能消除方差。

更关键的是，**最优 Baseline 不是必然等于平均 Reward**。这里 $\mathbb E[R]=0.25$，而本例的最优常数 Baseline 是 0.75。一般的最小方差常数与 Score Function 的平方大小有关：

$$
b^*=\frac{\mathbb E[R\,\|\nabla\log\pi(a)\|^2]}
{\mathbb E[\|\nabla\log\pi(a)\|^2]}.
$$

这是对单参数或指定梯度范数意义下的推导，不是所有模型参数共享的通用最优 Critic。

## 一个会让梯度彻底坏掉的 Baseline

考虑错误实现 §baseline=reward(a)§。它依赖本次采样的动作，于是每条样本都有 §reward-baseline=0§，梯度直接消失。这个反例明确违反了前面的 Action-independent 条件，**不能用「Baseline 不改变期望」为它辩护**。

真实 RL 中的 $V(s)$、GAE、组内平均 Reward 都不是同一件事：状态价值通常试图估计未来 Return；GAE 用 TD Error 的时序加权；GRPO 用同 Prompt Group 的样本比较。它们怎么估计 Advantage、引入什么偏差和方差，需要分别研究。

## 从一个动作到整个 Response

在自回归模型中，一条轨迹的动作概率为每个 Token 条件概率的乘积：

$$
\log\pi_\theta(y|x)=\sum_{t=1}^{T}\log\pi_\theta(y_t|x,y_{<t}).
$$

因此在合适的回报定义和环境假设下，最基础的 REINFORCE 梯度估计可以写为：

$$
\hat g=(R-b)\sum_{t=1}^T\nabla_\theta\log\pi_\theta(y_t|x,y_{<t}).
$$

它仍然需要 Token Shift、Action Mask 和明确定义的长度 Reduction。**把所有 Token 的 Loss 简单 mean()，可能改变长短轨迹的相对权重；这属于目标定义问题，不是 API 的细节。** 明天会把这些 Tensor 的位置逐项展开。

## 交互检验与源码定位

打开 [Lab 02](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=2)，把 $p$ 固定为 0.25，依次比较 $b=0$、$b=0.25$、$b=0.75$。检查三次的**期望梯度始终 0.1875**，但方差不同。然后把 Baseline 改成一个动作相关的函数，思考为什么不能再使用这个推导。

仓库已有 [Policy Gradient CPU 演示](../../examples/day_04/policy_gradient_lab.py)：它在小 Tensor 上检查负 Logprob Loss 的梯度方向，包含「Loss 值正好抵消但梯度并没有消失」的反例。旧代码路径上的 §day_04§ 是历史命名，本课只借它验证 Score Function 的梯度，不代表新大纲的 Day 04。

### 文献

- [Williams, 1992: REINFORCE](https://link.springer.com/article/10.1007/BF00992696)：Score Function 估计器的经典来源。
- [OpenAI Spinning Up：Policy Gradient](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：提供离散轨迹与 Baseline 的推导。
- [PPO](https://arxiv.org/abs/1707.06347)：使用由采样建立的优势与概率比例做多轮更新，为什么 Ratio 必须小心处理将在 Day 04 讲。
