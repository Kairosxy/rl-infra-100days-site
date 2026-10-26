# Day 04 · 复算卡

PPO 对某个已采样 Action 的重要性比率为：

$$
r=\exp(\log\pi_{\rm current}-\log\pi_{\rm old}).
$$

Clipped surrogate：

$$
J^{clip}=\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A).
$$

取 $\epsilon=0.2$：

- $A=+2,r=1.5$：$J=2.4$，Ratio 的正向增大一侧进入平台。
- $A=-2,r=0.6$：$J=-1.6$，Ratio 的继续减小一侧进入平台。
- $A=+2,r=0.6$：$J=1.2$，局部导数仍为 +2。

**需要留在脑子里的条件：** Clip 约束的是特定样本的 surrogate 梯度区域，不是对完整 Policy 分布的硬 KL 上界。Behavior、Old 和 Current 的概率定义必须对应同一 Action 和条件状态；Temperature/Top-p/Policy Version 不应被忽略。

[拖动 Ratio、切换 Advantage 符号](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=4)。
