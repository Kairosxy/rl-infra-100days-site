# Day 01 · 交互与静态图的分工

**先不要画“Rollout → Verifier → Learner”三个大方框。** 这三个名词还不能解释一次更新，学生最需要看到的是：已观察到的动作、Logprob 梯度和参数更新前后的概率。

已有 [Lab 01：采样概率与有限样本](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=1) 可以改变 $p$ 和样本数，帮助区分经验 Reward 与理论期望。它**不执行 Optimizer Step**，因此必须和 [讲义中的 PyTorch 代码](lesson.md) 配合使用。

如果新增教学交互，建议保持同一个二动作例子，只显示四个数：`z`、`p(A)`、`dL/dz`、`p_new(A)`。允许学生切换 Reward、Loss 正负号和学习率；正号错误时用橘色高亮概率下降。结果按公式计算，不要用随机装饰动画代替真实数值。
