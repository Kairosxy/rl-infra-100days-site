# Day 05 · 交互图：哪些 Group 有学习信号

[Lab 05：GRPO Group Advantage](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=5) 有四个可以切换的 Reward 格。改变其中一个，先预测均值、标准差和每条样本的 Advantage，再看逐条数值。

分别试 `[1,0,1,0]`、`[1,1,1,1]`，最后切换到「仅有前三条结果」。第三种状态会明显标成**未满足预定 $G=4$**；临时计算的是三样本组统计，不能冒充四样本组的标准化值。

图没有模拟 Policy Logprob、Clip 或真实 GPU 等待时间；它只用于看清统计输入的变化，这一边界将在 Async DataSampler 章节继续使用。
