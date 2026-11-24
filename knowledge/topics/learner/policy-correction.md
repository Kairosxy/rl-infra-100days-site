# 训推一致性：差异来源与策略纠偏

同一模型权重在训练和推理引擎上不一定给出相同的动作概率；异步训练还可能让采样版本落后于当前 Learner。先区分这两个偏差来源，再决定修改数值执行还是优化估计器。

## 两种概率差异

固定相同状态与动作，q 表示实际 Behavior，p_old 表示采样权重版本在训练引擎中的重算分布，p_theta 是当前训练分布。在三个概率为正、比较语义一致时，

$$
\log(p_\theta/q)=\log(p_\theta/p_{\rm old})+\log(p_{\rm old}/q).
$$

前一项是更新导致的差异，后一项是同版本下的执行/采样差异。这里的 Old 是专门定义的重算分布，不保证等于各框架 PPO 字段名。Reference 则是另一个约束对象。若 Token、条件上下文、mask 或支持集已经错误，应先修数据契约。

| 方向 | 作用对象 |
|---|---|
| 数值与 Kernel 对齐 | 同权重下算术、精度、归约和算子差异 |
| Router Replay（R3） | MoE Expert 选择 |
| Sampler Replay | 采样筛选集合等离散执行信息 |
| IS / TIS / Rejection | 比率重加权、截断或拒绝样本 |
| Score Centering | 行为分布下 Score 的偏置项，可与 IS 组合 |
| Staleness / ESS / 同步 | 策略发布、供数与有效样本的治理 |

这些手段不是严格的替代关系。R3 不自动修复所有精度差异，IS 也不能恢复零支持集的数据。

## Score 偏置的基础直觉

令训练策略 p=[0.8,0.2]，Behavior q=[0.6,0.4]。对两动作 Softmax 的第一个 logit，Score（动作 logprob 对参数的梯度分量）分别为 1−0.8=0.2 和 −0.8。其在 p 下期望为零，在 q 下期望为 −0.2。常数 Advantage=1 时，无修正地使用 q 的样本仍可能造成非零梯度更新。

作为中心化原理的教学例子，令 $g_c=g-\mathbb E_q[g]$，得到 [0.4,−0.6]，其 q 期望恢复为零。实际 Score Centering 方法的完整估计器、权重、条件分布及实现近似需按原论文与代码另行展开。这里不能据此断言任意中心化都能得到完整轨迹的无偏 Policy Gradient。满足支持条件的精确单步 IS 同样恢复常数信号下的零期望，但长序列、截断、状态访问分布等会改变估计性质。

2026 年 Score Centering 论文将 TIM 的持续 drift 与加性修正联系起来，并报告与 IS 组合的实验。上述实验来自论文，尚无本地 GPU 复现。RSC 等后续方法的具体机制需分别核查。

## 最小诊断

固定采样权重和 Token，分开记录上述两项 log-ratio，检查支持集丢失、Router Mismatch、ESS 和有效样本比例。只开启源头对齐、只开启 IS、以及其他纠偏组合，比较质量、有效训练 Token/GPU-hour 与额外概率数据成本。Reward/Verifier Serving 产生奖励，不属于这里的 Score 修正。

参考：[Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807)、[IS 基础](../rl/importance-sampling.md)、[MoE Routing 与 R3](moe-router.md)、[verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)。
