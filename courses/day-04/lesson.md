# Day 04/100 · Policy Gradient 与 Log Probability

> 目标：从期望 Reward 推导 Policy Gradient，再把公式对应到 LLM 的 Token Logprob、Teacher Forcing 和可验证的代码。适用范围是离散动作、自回归 Policy；本文的 REINFORCE 推导是基础估计器，不等同于 PPO/GRPO 的完整目标。

## 今天要解决的问题

Rollout 已经生成了 Token，Reward 也算出来了，为什么 Trainer 还要重新 Forward 一次？因为采样到的 Token ID 是离散结果，本身不提供从 Reward 到模型参数的普通反向路径。Policy Gradient 用采样动作的 Log Probability 构造梯度估计。

![Policy Gradient 与 Logprob](assets/policy_gradient_flow.svg)

## 先算一个只有两个动作的例子

假设模型只会回答 A 或 B。当前两个 Logit 都是 0，因此：

$$
\pi_\theta(A)=\pi_\theta(B)=0.5,\qquad
\log\pi_\theta(A)=\log 0.5\approx-0.693.
$$

这次采样选中 A，Verifier 给它 Reward = 1。基础 REINFORCE 把损失写成 \(-R\log\pi_\theta(A)\)，数值约为 0.693。对于 A 的 Logit \(z_A\)，梯度为：

$$
\frac{\partial\mathcal L}{\partial z_A}
=-R(1-\pi_\theta(A))=-0.5.
$$

Gradient Descent 会沿梯度的反方向更新参数，因此 A 的 Logit 会增大。若 Reward 为 0，这个基础估计器对该样本没有梯度；实际算法常使用 Baseline / Advantage，让低于基线的动作也能得到负向训练信号。

这个例子只有一个决策，没有 Group、Mask 或多轮环境。但已经说明了训练信号如何通过 Logprob 影响参数。

## 从期望 Reward 推导

令轨迹 τ 从当前策略分布采样，目标为

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau)]
=\sum_\tau p_\theta(\tau)R(\tau).
$$

对参数求导：

$$
\nabla_\theta J(\theta)
=\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau).
$$

利用恒等式 `∇p = p ∇log p`：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{\tau\sim p_\theta}
\left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right].
$$

Reward 不需要对 θ 可微；需要可微的是策略对已采样动作的概率。对自回归模型：

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{1:t-1}).
$$

于是一个基础 REINFORCE Loss 可以写为：

$$
\mathcal{L}_{\mathrm{PG}}
=-\frac{1}{B}\sum_{i=1}^{B}R_i
\sum_{t=1}^{T_i}\log\pi_\theta(y_{i,t}\mid x_i,y_{i,1:t-1}).
$$

这里按序列求和再按样本平均。实际训练也可能采用 Token Mean、Sequence Mean 或带 Advantage 的 PPO/GRPO Loss；这些归一化方式会改变不同长度样本的权重，不能默认等价。Baseline 可以降低方差；GAE、PPO Clip 和 GRPO Group Advantage 分别在后续课程展开。

## LLM 里的 Logprob

给定序列 `input_ids: [B, L]`，模型输出 `logits: [B, L, V]`。位置 t 的 logits 预测位置 t+1 的 token。若 Prompt 长度为 P、Response 长度为 T，则等长且未 packing 的简单情形下：

- action token：`input_ids[:, P:P+T]`
- 对应预测 logits：`logits[:, P-1:P+T-1, :]`
- selected logprob：`log_softmax(logits).gather(..., action_ids)`

这处偏移最容易出错。真实系统还要处理 padding、不同 Prompt 长度、packing、EOS、工具调用和多模态位置。

$$
\log\pi_\theta(a\mid s)
=z_a-\log\sum_{j=1}^{V}e^{z_j}.
$$

其对 logits 的梯度为：

$$
\frac{\partial\log\pi_\theta(a\mid s)}{\partial z_k}
=\mathbf{1}[k=a]-\pi_\theta(k\mid s).
$$

所以一个被选中 Token 的 Logprob 也会通过 Softmax 归一化项影响其他词表 Logits。它不是只读取一个 logit 的索引操作。

## 为什么训练要重新计算 Logprob？

生成服务可以记录 rollout logprob，但训练需要当前参数下、保留 Autograd 路径的 logprob。常见做法是把已生成的 token 作为固定输入做 Teacher Forcing，而不是重新采样一条回答。

需要区分：

- `rollout_log_probs`：实际生成时记录的概率，语义取决于采样配置和服务实现。
- `old_log_probs`：PPO 更新的固定比较基准。
- `current_log_probs`：当前正在优化的策略重新计算的概率。
- `ref_log_probs`：Reference Policy 的概率，常用于 KL 正则。

它们不天然相等。如果生成时使用 Temperature、Top-p，训练时却计算原始 softmax 概率，两个数值可能对应不同分布。Policy Version 不同也会造成差异。不能把所有 Logprob Drift 都归因于 FP8。

## Math → Code

| 数学对象 | 常见张量 | 主要检查 |
|---|---|---|
| Action 条件概率 | logits `[B,T,V]` | 给定状态下的概率分布、温度与采样变换 |
| Action Logprob | logprob `[B,T]` | Token shift、gather 索引 |
| Reward / Advantage | `[B]` 或广播后 `[B,T]` | Trajectory 与 Reward 对齐 |
| Response Mask | `[B,T]` | EOS、padding、工具 Token |
| Policy Loss | scalar | Sequence Mean / Token Mean |
| 参数梯度 | 与参数相同的 Shape | 梯度是否有限、方向是否正确 |

## 成本估算（理论值，不是实测）

固定例子：256 条轨迹、每条 Response 128 Token、Vocabulary 50,000。

完整 Response Vocabulary Logits 若按 BF16 显式存储：

$$
256\times128\times50000\times2
=3.2768\times10^9\ \text{bytes}
\approx3.05\ \text{GiB}.
$$

选取后的 FP32 Logprob 只需要：

$$
256\times128\times4=131072\ \text{bytes}=128\ \text{KiB}.
$$

这不代表完整 Logits 必然全部同时驻留；Micro-batch、融合 Kernel、分块 Logprob 计算会改变峰值。即使只需要选中 Token 的概率，Softmax 的归一化项仍需要处理整个 Vocabulary。

用粗略 Dense Transformer Forward 模型 `F ≈ 2NT`，N=1B、T=256×640，完整批次一次 Forward 约 328 TFLOPs。它只是数量级估算，忽略 Attention 二次项和实现细节。

## 最小实验

先运行 [基础 Policy Gradient 实验](../../examples/day_04/policy_gradient_lab.py)，验证 Logprob 梯度与 Advantage 符号。再运行 [序列 Logprob 实验](../../examples/day_04/sequence_logprob_lab.py)，验证变长 Prompt、Token Shift、Padding Mask 以及三种 Loss Reduction 的区别：

```bash
python examples/day_04/policy_gradient_lab.py
python examples/day_04/sequence_logprob_lab.py
python -m unittest discover -s examples/day_04 -p 'test_*.py' -v
```

## 排障顺序

1. **无梯度**：检查 logprob 是否在 `no_grad` 下计算、是否误 detach、Loss 是否真的依赖 Policy 参数。
2. **梯度方向错误**：检查 Action Token 与 Logits 是否错一位，再检查 Advantage 符号和 Mask。
3. **Reward 提升但质量不稳**：检查 Loss Reduction、轨迹对齐和采样分布。
4. **Rollout / Train Logprob 漂移**：先核对 Policy Version、Token ID、Temperature / Top-p、Position ID 和 Mask，再评估精度与 Kernel 差异。
5. **Logprob 计算过慢**：检查是否重复完整 Forward、是否物化完整 Vocabulary Logits，以及能否在概率语义不变的条件下复用 Rollout Logprob。

## 面试题

### 题 1（10 分）
为什么 Reward 不可微时，仍然能优化期望 Reward？从 `J(θ)=E[R]` 推导 Log-Derivative Trick，并解释自回归序列如何拆成 Token Logprob。

评分：展开期望 2 分；`∇p=p∇logp` 3 分；自回归分解 2 分；不可微 Reward 的解释 2 分；说明 On-policy 前提 1 分。

追问：被选中 Token 的 Logprob 是否只更新对应 Logit？满分答案应给出 `1[k=a]-p(k)` 并解释 Softmax 归一化。

### 题 2（10 分）
SGLang 生成使用 Temperature=0.8、Top-p=0.9，Megatron Trainer 重新计算原始 Softmax Logprob。团队希望直接复用 Rollout Logprob。你会如何判断是否正确？

评分：区分 rollout/old/current 3 分；分析采样变换 2 分；考虑数值差异 2 分；提出对照实验 2 分；解释计算与质量权衡 1 分。不能只凭权重相同就断言概率相同。

## 本课回顾

基础实验中的两个样本恰好具有相同 Logprob 和相反的 Advantage，所以标量 Loss 可以是 0，但各自的 Logit 梯度仍非零。判断训练信号时应同时看梯度、Mask 和参数变化；Loss 数值本身不能证明是否发生有效更新。


1. Policy Gradient 通过已采样动作的 Logprob 构造梯度估计，不要求 Reward 对模型参数可微。
2. Logprob 的位置、Token ID、Mask 和 Reward/Advantage 必须语义对齐。
3. 复用 Rollout Logprob 可以省计算，但必须先确认策略版本和概率定义一致。

## 对照公开实现：VERL 的 Loss Reduction

VERL 的 `verl/trainer/ppo/core_algos.py` 中 `agg_loss` 提供多种 Loss Aggregation 模式。以 commit [`174888fcf4b65dea032566a02fd16c5f76fb5b5b`](https://github.com/verl-project/verl/blob/174888fcf4b65dea032566a02fd16c5f76fb5b5b/verl/trainer/ppo/core_algos.py#L1140) 为阅读基准：`token-mean` 按全局有效 Token 数归一化；`seq-mean-token-sum` 先对每条序列求 Token Loss 之和，再对序列平均；`seq-mean-token-mean` 先求每条序列的 Token Mean，再对序列平均。跨 DP Rank 时还要考虑全局分母和后端缩放。比较其他框架时，应对照分母、Mask 和 DP 缩放，而不只是模式名称。

## 论文与官方资料

- Williams, 1992, *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. REINFORCE 的经典来源。https://doi.org/10.1007/BF00992696
- Sutton et al., 1999, *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. Policy Gradient Theorem 的基础论文。https://papers.nips.cc/paper_files/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Schulman et al., 2015, *Trust Region Policy Optimization*. 理解策略更新约束的背景。https://arxiv.org/abs/1502.05477
- Schulman et al., 2017, *Proximal Policy Optimization Algorithms*. 为 Day 06 做准备。https://arxiv.org/abs/1707.06347
- PyTorch LogSoftmax：https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
- PyTorch gather：https://docs.pytorch.org/docs/stable/generated/torch.gather.html
- VERL 算法实现：https://github.com/verl-project/verl/blob/174888fcf4b65dea032566a02fd16c5f76fb5b5b/verl/trainer/ppo/core_algos.py

## 下一课

Day 05：Value、GAE 与偏差—方差。重点推导 Baseline 为什么能降低梯度方差，以及 GAE 如何在 Monte Carlo Return 与 TD Estimate 之间做权衡。


## 项目对照（由你填写）

在你熟悉的 Rollout/Trainer 链路中，哪些条件决定 Rollout Logprob 能否作为训练基准？分别记录采样配置、Policy Version、Token 对齐和 Mask 的证据；不要把所有 Logprob Drift 都归因于量化。

将可公开的结论记到 [项目叙事记录](https://github.com/Kairosxy/rl-infra-100days/blob/main/portfolio/project_narrative.md)。本课评分标准见 [Day 04 面试问答](qa.md)。
