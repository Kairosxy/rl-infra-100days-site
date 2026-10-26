# Day 01 · 图解不画成框图

本课没有画大型 RL Pipeline：不知道要优化什么之前，记住 Rollout/Reward/Trainer 方框的顺序意义有限。

打开 [Lab 01：采样概率与期望 Reward](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=1)。横向滑动 $p$，直接看解析式 $J=p$ 和有限采样的结果差异。先预测、再运行、再改变采样次数，形成对随机梯度噪声的直觉。

**图的限制：** 这里只有二动作、确定性 Reward、独立采样。它没有 LLM Token、Prompt 分布和权重更新；真实系统不能用这个 Toy 的采样时间衡量 GPU 性能。
