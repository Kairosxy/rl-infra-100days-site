# Day 04 · 交互图：PPO 梯度平台在哪里

[Lab 04：PPO Ratio / Clip 曲线](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=4) 绘出 **Clipped Surrogate 关于 Ratio 的分段目标**，标出 $1-\epsilon$ 和 $1+\epsilon$。拖动 Ratio，从 $A>0$ 切换到 $A<0$。

**看图必须解释的判断：** 正 Advantage 在上界右侧进入平台；负 Advantage 在下界左侧进入平台。这张图描述的是一个样本、一个 Ratio 对目标值的局部贡献，**不是**完整 Transformer 更新轨迹，也不声称平台就保证全局 KL 小。
