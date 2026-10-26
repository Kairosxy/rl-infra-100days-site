# Trajectory 数据契约

数据契约定义字段语义、形状和所有权。字段名可以因实现而异，但不同含义不能混用；过滤或重排后也不能只凭列表位置关联数据。

| 分类 | 字段 | 含义 |
|---|---|---|
| 身份 | request_id、trajectory_id、group_id、episode_id、sample_id | 请求、轨迹、组、环境回合和组内样本身份 |
| 策略版本 | behavior_policy_version、behavior_checkpoint_id、reference_policy_version | 行为策略和参考策略来源 |
| Token | prompt_ids、response_ids、token_logprobs | 输入、动作和对应概率 |
| Mask | attention_mask、response_mask、loss_mask、action_mask | 有效上下文、生成区域、参与 Loss 的位置、RL Action 位置 |
| 结果 | reward、terminated、truncated、finish_reason | 奖励、终止、截断和结束原因 |
| 生命周期 | created_at、completed_at、consumed_at | 创建、完成和消费时间 |

## Mask 的区别

- attention_mask：哪些位置是有效上下文，具体约定取决于模型接口。
- response_mask：哪些位置属于生成的 Response。
- loss_mask：哪些位置参与当前 Loss，可以排除部分 Response Token。
- action_mask：哪些位置代表算法定义的动作；工具返回内容不一定是 Policy Action。

在简单单轮任务中，这些 Mask 可能部分重合；不能因此把它们定义成同一个字段。

## 基础约束

1. trajectory_id 在数据集生命周期内唯一。
2. response_ids 与 Token-level token_logprobs 对齐；Padding 约定必须明确。
3. Reward 与轨迹通过 ID 关联，不依赖排序。
4. Group 目标大小、过滤条件和 Policy Version 规则显式记录。
5. Reward 必须为有限数值；Mask 只能使用约定取值。
6. 明确每个字段由哪个组件创建、哪个组件可以修改。

默认同步基线要求同一训练 Group 的 Behavior Policy Version 一致。异步实验可以允许不同 Group 使用不同版本，但必须记录版本、滞后步数和算法修正。
