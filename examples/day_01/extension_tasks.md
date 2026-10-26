# Day 01 扩展任务：身份关联与 Policy Version

这些任务由学习者完成。先写测试，再实现校验逻辑。

## A. Reward 按 ID Join

准备 Rollout 结果与 Reward 返回两个独立列表。将 Reward 列表随机打乱后，按 trajectory_id 关联的结果应保持不变。

验收：
- ID Join 对乱序稳定。
- 缺少、重复或未知 ID 明确失败。
- 与按列表位置配对的错误实现对照，证明乱序会导致错配。

## B. 同组 Policy Version

增加 behavior_policy_version，默认要求一个 Group 内版本一致。

验收：同版本通过；不同版本默认失败；若允许版本差异，返回可审计的版本跨度，而不是只返回 True。

## C. 消费幂等性

同一 Trajectory 的完成事件重复到达时，不能让它被计入训练两次。写出状态迁移和负例测试。
