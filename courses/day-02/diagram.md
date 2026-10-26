# Day 02 · 交互图：看期望与方差，而不是只看一个公式

[Lab 02：Baseline 与采样梯度](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=2) 把二动作样本的梯度估计画成两个离散值，标出真实梯度与方差。先把 $p$ 固定 0.25，拖动 $b$ 从 0 到 0.75。

**读图任务：** 观察两条可能的采样梯度怎样靠拢；期望线不随 $b$ 变化。随后提出一个反例：如果对 A/B 使用不同的动作相关 Baseline，这条结论就不能继续使用。

图只模拟单参数 Bernoulli Policy，不能把本例出现的零方差当作真实 LLM 的预期结果。
