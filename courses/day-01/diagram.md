# Day 01 图解 · 训练数据流与对齐边界

![RL 训练链路](assets/rl_training_loop.svg)

这张图从上往下展示一条训练链路。箭头表示数据依赖，不要求六个环节严格串行运行，也不意味着每个框都对应独立进程。监控、日志和 Checkpoint 是整个系统的旁路能力。

## 沿着一条 Trajectory 看字段

假设 Prompt 为“17 × 19 = ?”，Rollout 生成回答后，系统记录 `trajectory_id=t-017`、`group_id=g-04`、`response_ids` 和 `policy_version=10`。Verifier 独立返回 `(trajectory_id=t-017, reward=1.0)`。Sampler 必须按 ID 关联这两部分数据；过滤、重排或批次拆分都不应改变最终对应关系。

在 [Day 01 的 CPU 实验](../../examples/day_01/trajectory_contract.py) 中，可以将 Reward 列表故意倒序，再确认 Join 后每条轨迹的 Reward 不变。若仅按数组下标合并，两份长度相等的列表也会得到错误结果。

## 图中的等待要分清

同组四条轨迹的 Reward 还未收齐时，标准四样本 Group Advantage 的统计输入不完整。这是算法依赖。若 Group A 已完整、Group B 仍在生成，是否要等 B 则取决于外层采样目标、Trainer 的批次契约与版本语义。

因此，异步执行可以减少不必要的外层批次等待，但不能绕过算法依赖或设备资源互斥。

## 常见故障与最小证据

| 现象 | 可能原因 | 最小验证 |
|---|---|---|
| Reward 与回答对不上 | 过滤或重排后按位置关联 | 将 Reward 乱序，比较按 ID Join 的结果 |
| Loss 使用了 Padding | Mask 长度或取值错误 | 检查有效 Token 数与 Mask |
| Logprob 差异无法解释 | 策略版本或概率定义缺失 | 对照 Token ID、Policy Version 和采样参数 |
| Group Advantage 异常 | Group 不完整或版本混杂 | 统计每组有效轨迹数与版本集合 |
| 达标后任务仍运行 | 数据目标与在途排空混淆 | 分别记录 Accepted、Inflight 和 Drain 完成时间 |
