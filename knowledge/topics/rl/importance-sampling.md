# Importance Sampling、支持集与有效样本量

Importance Sampling（IS）允许使用一个分布下采集的样本估计另一个分布下的期望。RL 的异步采样、旧版本复用和训练/推理差异都涉及这个问题。IS 能否成立取决于实际采样分布及其支持集。

## 两个动作的换测度

固定状态 s，目标策略 p=[0.8,0.2]，Behavior q=[0.5,0.5]，两个动作的 Reward=[1,0]。直接对 q 的样本取均值，期望为 0.5；要估计 p 的期望，可使用 w=p/q=[1.6,0.4]：

$$
\mathbb E_q[wR]=0.5\times1.6\times1+0.5\times0.4\times0=0.8.
$$

成立条件是 p 的支持集包含于 q 的支持集，并且权重对应实际采样概率。若 q=[1,0]、p 的第二项仍大于零，则第二个动作没有机会被采到，IS 无法恢复完整目标期望。Top-k/Top-p 截断需要明确目标分布也是否在同一支持集上。

## 长序列为什么困难

相同环境转移机制下，整条轨迹的概率比是逐动作条件概率比的乘积。

$$
\log w(\tau)=\sum_t(\log p(a_t|s_t)-\log q(a_t|s_t)).
$$

即使每步 ratio 仅为 1.1，100 步乘积也约 13780.6；每步 0.9 则约 0.0000266。这组固定 ratio 仅用于数量级估算。Token-level correction 处理局部条件分布；整轨迹权重还涉及状态访问分布变化，不能把任意 Token 平均 ratio 称为完整无偏轨迹修正。

TIS（截断重要性采样）限制极端权重，通常引入偏差以降低方差；V-trace 等算法定义专门的多步离策略价值目标。PPO 的 Clip 是对代理优化目标的约束，不等同于无偏 IS 估计。

## ESS 与版本滞后

归一化权重的集中程度可用有效样本量近似衡量：

$$
\mathrm{ESS}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

四个权重 [1,1,1,1] 的 ESS=4；[4,0,0,0] 的 ESS=1。若全零，分母也为零，应判定无有效权重而不是返回一个虚假的正常数。ESS 是权重集中度诊断，不证明样本独立、无偏或具有相同质量。

异步系统记录 policy version age，并结合 actual log-ratio、ESS、Clip/Mask Fraction 判断数据是否还能训练。版本年龄相同并不意味着策略分布距离相同；不能单独凭 staleness 数值决定所有 correction。

继续阅读 [训推概率契约](../learner/logprob-contract.md)、[策略纠偏](../learner/policy-correction.md) 与 [异步 Rollout](../async/async-rollout.md)。参考：Sutton & Barto (2018), Ch.5；[IMPALA / V-trace](https://arxiv.org/abs/1802.01561)。
