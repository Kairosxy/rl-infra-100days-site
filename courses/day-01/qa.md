# Day 01 面试问答

## 题 1：描述 RL 训练链路（10 分）

**问题**：从 Prompt 到模型参数更新，逐步解释每个阶段的输入和输出。

**评分标准**
- Rollout 与 Trajectory：2 分
- Reward / Advantage：2 分
- Logprob 与 Loss：2 分
- Backward / Optimizer：2 分
- Policy Version 与权重同步：2 分

**追问**：如果 Reward 已经算出，为什么还要计算策略 Logprob？

合格回答应说明：Reward 是评价信号；Policy Gradient 通过已采样动作的 Logprob 构造可微的梯度估计。对一个与模型计算图无关的 Reward 常数求导，不能更新 Policy。

## 题 2：组内依赖与异步调度（10 分）

**问题**：GRPO 的一个 Prompt 生成 4 条轨迹。只完成 3 条时，能否先计算标准的四样本 Group Advantage？

**评分标准**
- 明确四样本统计尚不完整：3 分
- 说明提前计算会改变统计量或目标：3 分
- 提出等待、补投或显式修改算法契约的方案：2 分
- 说明长尾与资源成本：2 分

**追问**：如果 Group A 已完整，而 Group B 仍在生成，是否一定要等待 B？

合格回答应区分组内依赖和外层 Batch Barrier。Group A 是否能立即进入训练，还取决于 Trainer 批次、梯度累积、版本策略和采样器契约；不能把“Group 已完成”直接等同于“Trainer 一定可以立即更新”。
