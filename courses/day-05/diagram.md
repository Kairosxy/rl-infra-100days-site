# Day 05 · 交互图只画一个问题：组里是否还有区分度

[Lab 05：Group Reward 与 Advantage](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=5) 允许独立修改四条 Reward，并切换成只返回前三条的状态。实验输出均值、标准差和逐轨迹 Advantage。

读图时区分三种情况：有正负差异的完整 Group、全同 Reward 的零相对信号、未满足预定四样本大小的不完整 Group。这个界面只演算组内统计，**没有执行 PPO/GRPO 的完整 Token Loss，也没有模拟 GPU 排空时间**。

如果需要教学代码，优先增加检查“Reward 与 Trajectory ID 错配”的负例，不要把动态图再画成一套长篇箱线流程。
