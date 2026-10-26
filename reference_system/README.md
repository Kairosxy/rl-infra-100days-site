# CPU 参考系统：Day 01–04 的可运行主线

该示例用两个状态、两个动作的 Contextual Bandit 串联前四课：Prompt/State、Rollout、Trajectory ID、Behavior Policy Version、Reward 按 ID 关联、Current Logprob、REINFORCE Loss、Optimizer 和 Checkpoint。

这不是 PPO/GRPO，也没有 Transformer、Ray 或多卡并行。它只验证数据关联与梯度、版本语义，不能用其吞吐结果推断 LLM 性能。

## 运行

在仓库根目录：

```bash
python -m pip install -r requirements.txt
python reference_system/toy_reinforce.py
python -m unittest discover -s reference_system -p 'test_*.py' -v
make test
```

程序每步打印 Reward Mean、Loss 和发布的新版本号。短跑 Reward 不保证单调上升；这个例子更适合验证训练状态和数据流。

## 系统契约

- **Rollout**：在给定版本上采样，保存动作和当时的 Behavior Logprob。
- **Verifier**：输出独立的 `(trajectory_id, reward)`，可任意乱序返回。
- **Sampler**：按 ID 关联奖励，不允许重复 ID 或 ID 集合不一致。
- **Learner**：固定采样动作，用当前可微 Policy 计算 Logprob，并对 Reward-baseline 加权。
- **Optimizer**：完成一次参数更新后递增 Policy Version。
- **Checkpoint**：保存并恢复 Model、Optimizer 和版本，不声称支持任意时间点的分布式原子快照。

## 验证与未完成边界

单元测试覆盖 Reward 乱序、重复/未知 ID、参数更新、过期版本拒收和 Checkpoint 状态恢复。需要扩展的功能包括 Group Advantage、Mask、多轮轨迹和跨 Step Staleness 控制，这些会在 Day 05–10 逐步加入。

与单课实验的对应关系：
- [Day 01 Trajectory Contract](../examples/day_01/trajectory_contract.py)
- [Day 02 显存账本](../examples/day_02/memory_budget.py)
- [Day 03 梯度累积](../examples/day_03/gradient_accumulation_lab.py)
- [Day 04 Token Logprob 对齐](../examples/day_04/sequence_logprob_lab.py)
