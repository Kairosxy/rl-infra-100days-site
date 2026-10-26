# Day 04 · Reward 不可微，为什么还能更新 Policy？

一个 RL Rollout 生成了「A」这个动作，Verifier 给了 Reward = 1。训练时的目标是让类似动作更可能被选中，但**采样出来的整数 Token ID** 没有通向模型参数的常规微分路径。Trainer 不能指望把 Reward 当成 Loss 直接 `backward()`。

Policy Gradient 解决的是怎样对**采样动作发生的概率**求导，而不是怎样对环境评分程序求导。把这个过程在两动作模型中算清，再看 LLM 的 Token Shift、Mask 和不同 Logprob 来源。

## 两个 Logit 就能看到梯度方向

先设动作 A/B 的 Logit 都是 0。Softmax 后：

$$
\pi_\theta(A)=\pi_\theta(B)=0.5,\qquad
\log\pi_\theta(A)=\log 0.5\approx -0.6931.
$$

假设本次采样到了 A，Reward 为 1，使用最简单的 REINFORCE 负号 Loss：

$$
L=-R\log\pi_\theta(A)=-\log 0.5\approx 0.6931.
$$

记 Logit 为 $z_A,z_B$。Softmax 的导数给出：

$$
\frac{\partial L}{\partial z_A}
=-R\bigl(1-\pi_\theta(A)\bigr)=-0.5,\qquad
\frac{\partial L}{\partial z_B}
=+R\pi_\theta(B)=+0.5.
$$

梯度下降执行 $z\leftarrow z-\eta\nabla L$，因此 A 的 Logit 上升、B 的下降。Reward 仍是一个常数，提供的是 **梯度权重**；梯度实际穿过的是 $\log\pi_\theta(A)$ 对模型参数的计算图。

如果把 Reward 换成负 Advantage $A=-1$，梯度方向就会翻转，降低已选动作的概率。这里的「A」是动作名称，公式里的 $A=-1$ 是 Advantage，两者概念不同。

## 这条梯度公式怎么来的

令一条完整轨迹为 $\tau$，它在当前策略下发生的概率为 $p_\theta(\tau)$，轨迹的 Return 为 $R(\tau)$。先考虑离散、有限轨迹空间，并假设环境转移概率本身不依赖 $\theta$：

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)]
=\sum_\tau p_\theta(\tau)R(\tau).
$$

对 $\theta$ 求导，并使用 $\nabla p=p\nabla\log p$：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau)\\
&=\sum_\tau p_\theta(\tau)R(\tau)\nabla_\theta\log p_\theta(\tau)\\
&=\mathbb E_{\tau\sim p_\theta}
\!\left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right].
\end{aligned}
$$

自回归 Token 生成时，给定 Prompt $x$：

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

假设环境转移和 Reward 计算与 Policy 参数无直接依赖，可把轨迹概率中的策略部分拆成这些 Action Logprob。于是基础采样估计器会出现 $-R_i\sum_t \log\pi_\theta(y_{i,t}\mid\cdots)$ 这样的 Loss。

**这里仍有明确边界。** 上述推导在当前策略分布下采样得到最直接的 on-policy 估计；拿旧策略样本直接代入最新策略的同一公式，一般不能无条件保持无偏。PPO 的 Ratio、Off-policy Correction 和 GRPO 目标都是后面的内容。

Baseline 可以减小梯度方差。若 $b(s)$ 不依赖给定状态下采样的 Action，则有：

$$
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
[b(s)\nabla_\theta\log\pi_\theta(a\mid s)]
=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.
$$

所以在相应条件下，用 Reward-to-go 减去状态 Baseline 不改变期望梯度。不同 Advantage 估计器还可能引入额外偏差与方差取舍，Day 05 会从 TD Error 开始解释。

## 为什么生成过的 Token 还要交给 Trainer Forward

Rollout 已经把动作采样成离散 Token。Trainer 想计算**当前参数**对这些固定动作赋予的概率，需要重新做一次带 Autograd 的 Forward；这一过程通常称为 Teacher Forcing。它并不重新生成一条答案。

给定完整序列长度 $L$：

~~~text
token_ids: [B, L]          # Prompt + Response + Padding
logits:    [B, L, V]       # 位置 t 的输出预测位置 t+1
~~~

对于 Prompt 长度为 $P$、Response 长度为 $T$ 的简单、未 Packing 情况，参与训练的动作是：

~~~python
action_ids = token_ids[:, P : P + T]
action_logits = logits[:, P - 1 : P + T - 1, :]
log_probs = action_logits.log_softmax(-1).gather(
    -1, action_ids.unsqueeze(-1)
).squeeze(-1)  # [B, T]
~~~

第一个 Response Token 在 `token_ids[:, P]`，它的预测来自前一个位置的 `logits[:, P-1]`。偏移一位之后，Shape 依然可能合法，Loss 也仍会产生梯度，但优化的是**错位的 Token**。

仓库的 [`response_token_log_probs`](../../examples/day_04/sequence_logprob_lab.py) 给出更一般的实现：输入 `logits [B,L,V]`、`token_ids [B,L]`、`prompt_lengths [B]` 和 `valid_lengths [B]`；返回 `selected_log_probs [B,L-1]` 以及对齐的布尔 Mask。

对它的现有样例：

~~~text
B=2, L=7
prompt_lengths = [2, 3]
valid_lengths  = [5, 6]

target token 的位置:  1  2  3  4  5  6
第 0 行 action mask: 0  1  1  1  0  0
第 1 行 action mask: 0  0  1  1  1  0
~~~

两行共 **6 个有效 Response 位置**。这份 CPU Lab 只用 Prompt/有效长度决定 Mask；它没有额外排除工具观察、多模态位置等特殊语义，因此不能直接当作任意 Agentic RL 的 Action Mask 实现。

## Token Mean 和 Sequence Mean 不是改个分母这么简单

两条轨迹分别有 1 个和 3 个有效 Token。为了看清权重，假设所有有效位置的 Logprob 都是 $-1$，Advantage 都是 $+1$，则每个 Token 的负 Logprob 是 1。

三种常见聚合会得到：

| Reduction | 本例 Loss | 参与加权的方式 |
|---|---:|---|
| 全局 Token Mean | $(1+3)/4=1$ | 每个有效 Token 权重 $1/4$ |
| Sequence Mean of Token Sum | $(1+3)/2=2$ | 每条序列先求和，每个 Token 的梯度权重 $1/2$ |
| Sequence Mean of Token Mean | $(1/1+3/3)/2=1$ | 短序列每 Token 为 $1/2$；长序列每 Token 为 $1/6$ |

即使两个 Reduction 的 **Loss 标量恰好同为 1**，每条序列对梯度的贡献也可能明显不同。真实 RL 训练中必须先确定优化目标的归一化语义，再决定 Token/Sequence/Group 这几种分母，而不是机械地在最后调用 `mean()`。

还有一个更容易误诊的例子：[`policy_gradient_lab.py`](../../examples/day_04/policy_gradient_lab.py) 用两个对称样本，Advantage 分别为 $+1$ 和 $-1$。它们的 Logprob 数值相同，平均 Loss **恰好为 0**，但两条样本的梯度作用在不同的 Logit 行上，并没有互相抵消。**Loss 等于 0 不意味着 Gradient 为 0。**

## 四种 Logprob 的角色，决定了能不能复用

同一个 Response 会在不同阶段出现几份概率：

- `rollout_log_probs`：生成动作时服务侧记录的概率。是否是原始分布还是 Temperature、Top-p 等处理后的采样分布，要确认接口语义。
- `old_log_probs`：PPO 某次更新中固定的比较基准，Ratio 的分母使用它。
- `current_log_probs`：在当前可微参数下，对固定 Token 重算的概率。
- `ref_log_probs`：来自 Reference Policy，用于特定 KL 目标。

这些字段可以在某些配置下数值接近，但不能直接作为同一个对象。尤其是 Rollout 使用 Temperature 或受限采样，Trainer 却用未经同样变换的原始 Softmax，Logprob 差异并不自动证明 FP8 误差。先排查是否在比较**同一个条件概率定义**，再检查 Token Shift、Mask、Weight Version，最后才讨论数值精度。

**能否用 Rollout Logprob 代替 Old Logprob** 是算法与配置决定的选择，需要确保固定基准、策略版本和概率语义匹配，不能因字段都叫 `log_probs` 就直接复用。Day 06 会在 PPO Ratio 中深入讨论。

## 运行能暴露错误的两组 CPU 测试

~~~bash
python examples/day_04/policy_gradient_lab.py
python examples/day_04/sequence_logprob_lab.py
python -m unittest discover -s examples/day_04 -p 'test_*.py' -v
~~~

[Policy Gradient 测试](../../examples/day_04/test_policy_gradient_lab.py) 会检查：正/负 Advantage 的梯度方向、零 Mask 的梯度归零、Loss 为零但 Gradient 非零，以及 Advantage 不被错误反向传播。

[Token Logprob 测试](../../examples/day_04/test_sequence_logprob_lab.py) 会检查：Shift 前后 Logprob 不相等；Prompt/Padding 的位置不接受 Loss Gradient；不同 Reduction 造成的梯度权重差异。先预测 `selected_log_probs` 和 Mask 的 Shape，再运行测试。能够准确解释这两个张量的索引关系，比背下 REINFORCE 公式更接近真实 Trainer 的 Debug 工作。

## 来源

- [Spinning Up：Policy Gradient 推导与 Baseline](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)，对应本课的期望梯度推导。
- [PyTorch `log_softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) 与 [`gather`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)，对应 Token Logprob 选择。
- [PPO 原论文](https://arxiv.org/abs/1707.06347)，用于区分基本 Policy Gradient 与后续 Ratio/Clip。
- 本仓库的 [`policy_gradient_lab.py`](../../examples/day_04/policy_gradient_lab.py) 和 [`sequence_logprob_lab.py`](../../examples/day_04/sequence_logprob_lab.py)。本课已按这两份代码的现有接口解释，不声称它们就是 verl 的原生实现。
