# Day 04 · 这条分段曲线真正说明什么

在 [Lab 04：PPO Ratio/Clip](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=4) 中，先把 Advantage 设为 +2，拖动 Ratio 从 0.6 到 1.5；然后把 Advantage 设为 −2 再做一遍。

图的横轴是**单条样本的 Ratio**，纵轴是 Clipped Surrogate（最大化目标）。它展示为什么好动作被提高得太多时出现右平台，坏动作被压低得太多时出现左平台。

两条阈值线是 $1\pm\epsilon$，不是全策略概率分布之间的硬约束。图不包含完整词表 KL，也不能代替训练过程中对 Old/Current/Behavior 版本一致性的检查。
