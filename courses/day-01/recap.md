# Day 01 · 5 分钟复习：身份、组与版本

不看讲义，先回答：两份列表长度都是 4，为什么按位置 `zip` 可能让 RL 训练方向反过来？

**自己重建数据链。** Rollout 产出 `trajectory_id`、`group_id`、`response_ids`、Mask 和生成时的策略版本；Verifier 返回带相同 `trajectory_id` 的 Reward；Sampler 按 ID Join，并验证完整性与版本语义；Learner 才能按某个明确的目标计算可微 Logprob/Loss。

**手算 30 秒。** 四条 Reward 为 `[1,0,1,0]`，均值 0.5、总体标准差 0.5，标准化后 Advantage 为 `[1,-1,1,-1]`。只拿前三条算，统计量变了，不能称为原四样本 Group 的同一个目标。

**最小诊断。** 如果 `reward_mean` 没变，训练质量却异常，先将 Reward 完成顺序打乱、验证按 ID Join 后每条轨迹的配对不变。只看 Shape 和 Reward Mean 不能证明身份正确。

**不要混淆两种时间点。** Group 所需数据完整，与外层 Batch 全部结束、Trainer 可以执行一次优化，是不同条件。v10 生成的轨迹，即使 v11 时才消费，也必须保留生成时的版本事实。

补充实验：[Reward ID Join 负例测试](../../examples/day_01/test_trajectory_contract.py)。
