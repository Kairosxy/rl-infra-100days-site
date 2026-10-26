# Day 02 · 为什么需要两个梯度点，而不是一个漂亮曲线

把 $p$ 固定为 0.25， [Lab 02：Baseline 与采样梯度](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=2) 会显示“抽到 A”和“抽到 B”时各自的梯度，以及始终应保持的期望梯度线。

先不要拖动滑块：请算出 $b=0$ 时两个点在 0 和 0.75，再预测 $b=0.75$ 时它们为何重合。图的轴标的是**梯度估计值**，不是模型训练的 Loss 曲线。

实验适用条件是固定动作 Reward、单参数 Bernoulli Policy、与当前动作无关的常数 Baseline。不要把本例的零方差效果当成真实多步 RL 的保证。
