# Day 04 · 5 分钟复习：梯度究竟经过哪里

**从两个 Logit 重建梯度。** `z_A=z_B=0`，所以 `p(A)=0.5`。采样 A、Reward=1，Loss 为 `-log(0.5)=0.6931`；对 A 的 Logit 梯度为 $-0.5$，梯度下降后 A 的概率会上升。Reward 不需要可微，`log πθ(A)` 需要。

**再写一遍恒等式。**

$$
\nabla_\theta \mathbb E_{\tau\sim p_\theta}R(\tau)
=\mathbb E_{\tau\sim p_\theta}\bigl[R(\tau)\nabla_\theta\log p_\theta(\tau)\bigr].
$$

这里的直接推导依赖合适的采样分布与环境独立性假设。旧样本不能不加条件地当作当前策略样本。

**把 Token 索引对齐。** 若 Prompt 长 $P$、Response 长 $T$，目标 Action 是 `token_ids[P:P+T]`，预测它们的是 `logits[P-1:P+T-1]`。先 Shift，后 `log_softmax`、`gather`，再应用有效 Action Mask。

**对照一个假阳性。** 两个样本的 Logprob 相同，Advantage 为 `[+1,-1]`，Loss 可以正好为 0，但样本的梯度仍不为零。只看 Loss Scalar 无法判断梯度是否在更新。

Rollout / Old / Current / Reference Logprob 用途不同。任何差异先核对分布定义、Token、Mask 和版本，再讨论数值精度。

复核：[Policy Gradient 负例](../../examples/day_04/test_policy_gradient_lab.py) · [Token 对齐测试](../../examples/day_04/test_sequence_logprob_lab.py)。
