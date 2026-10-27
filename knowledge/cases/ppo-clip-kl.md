# PPO 已经 Clip，为什么 KL 还在上升？

某个训练 Batch 的 Clip Fraction 很高，直觉会认为 PPO 正在阻止 Policy 偏离 Old。但实际日志里的 KL 可能继续增长。这不矛盾：**Clip 限制的是当前样本的特定 Surrogate 分支，而不是把整个模型的参数变化限制在某个区域**。

取 $\epsilon=0.2$、Advantage $A=+2$、动作 Ratio $r=1.5$：

$$
J_{\rm clip}=\min(1.5\times2,\ 1.2\times2)=2.4.
$$

同一条样本继续提高 Ratio 时，局部 Surrogate 不再增加。可是其他样本、KL/Entropy 项、共享参数依然能移动这个动作的概率。

## 一个不靠训练跑分的反例

Old 分布 $[0.5,0.5]$、Current 分布 $[0.9,0.1]$。动作 A 的 Ratio 为 1.8，正 Advantage 时被 Clip；完整分布的

$$
D_{\mathrm{KL}}(\pi_{\rm old}\|\pi_{\rm current})
=\tfrac12\log\frac{0.5}{0.9}+\tfrac12\log\frac{0.5}{0.1}
\approx0.5108\ {\rm nats}.
$$

它并没有被 $\epsilon=0.2$ 上限束缚。注意这是一个**数学反例**，不是某个具体训练器产生的实验轨迹。

## 排查顺序

先确认被监控的 KL 是哪个方向：Old→Current、Current→Reference 还是采样 Token 的近似值。再对齐 Behavior、Old、Current 的权重版本与 Token ID，最后检查 Ratio 分布、Clip Fraction 和 Policy Loss 是否采用预期的 Token/Sequence Reduction。

一个可靠的单元测试应固定正负 Advantage 与 Ratio，检查两个方向的分段梯度；另外构造上面的两动作分布，验证单样本 Clip 不蕴含全分布 KL 上界。

## 框架入口

verl、NeMo RL、OpenRLHF 都有实现 PPO/GRPO 相关目标的路径，但具体 Loss Type、Reduction、KL 统计口径随版本和配置不同。比较前应先固定 Commit 和参数，而不是只对比训练曲线的标签。

阅读：[PPO](https://arxiv.org/abs/1707.06347) · [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300) · [DAPO](https://arxiv.org/abs/2503.14476)。