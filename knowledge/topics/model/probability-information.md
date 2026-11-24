# 概率、Logprob 与信息量

语言模型输出 logits，经过归一化得到概率。训练、生成、蒸馏和 RL 使用的是同一套概率对象，但取期望的分布、目标和样本来源不同。

## 从 logits 到条件分布

固定一个上下文 s，两个候选 Token A、B 的 logits 是 [0,ln(3)]。Softmax 后概率为 [0.25,0.75]，A 的 logprob 为 ln(0.25)，约 -1.386。Logprob 是自然对数时，单位是 nat。

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad
\log p_i=z_i-\operatorname{logsumexp}(z).
$$

将所有 logits 同时加上常数不改变概率；计算时减去最大 logit 可以降低指数溢出风险。不能把 logits 的比值当概率比值。

若 A 后生成 B 的条件概率为 0.8，则序列 AB 的联合概率为 0.25×0.8=0.2，联合 logprob 是两项之和。Autoregressive 的序列概率是条件概率乘积，不是各位置独立概率的乘积。

## 期望、样本均值与方差

令 A 的奖励为 1，B 的奖励为 0。在上述分布下，期望奖励为 0.25，方差为 0.25×0.75=0.1875。一次抽样的 reward 只能是 0 或 1，不等于期望。独立抽样 n 次，样本均值的方差为 0.1875/n；相关样本不满足这一简化公式。

训练中一个 Batch 的 mean reward 是估计值。改变任务分布、拒绝样本或改变 Group 组成都会影响估计对象，不只是改变方差。

## Entropy、交叉熵与 KL

取目标分布 p=[0.75,0.25]，模型 q=[0.5,0.5]。定义

$$
H(p)=-\sum_i p_i\log p_i,\quad
H(p,q)=-\sum_i p_i\log q_i,\quad
D_{\mathrm{KL}}(p\Vert q)=H(p,q)-H(p).
$$

本例 H(p)≈0.5623，交叉熵≈0.6931，KL≈0.1308。若只抽中第二个 Token，则 log(p/q)=ln(0.5)<0；这不违反整体 KL 非负。Reverse KL 则改变取期望的分布和比率方向。若 q 对 p 的正概率事件赋零概率，Forward KL 为无穷。

在 SFT 中目标通常是某个标签的 one-hot 分布，单位置 Cross Entropy 等于负目标 logprob。OPD 的目标可以是 Teacher 的整个分布；PPO 则以采样动作的 ratio 和 advantage 构造优化目标。三者都使用 logprob，但不优化同一目标。

~~~python
import math

p, q = [0.75, 0.25], [0.5, 0.5]
entropy = -sum(x * math.log(x) for x in p)
cross_entropy = -sum(x * math.log(y) for x, y in zip(p, q))
kl = sum(x * math.log(x/y) for x, y in zip(p, q))
assert abs(cross_entropy - entropy - kl) < 1e-12
assert math.log(p[1]/q[1]) < 0 < kl
~~~

继续阅读 [Policy Gradient](../rl/policy-gradient.md) 和 [OPD](../rl/opd-kl.md)。参考：[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.CrossEntropyLoss.html)、Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed., 2018)。
