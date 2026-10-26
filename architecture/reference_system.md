# 参考系统：从 Prompt 到 Policy Version

本文件定义贯穿课程的最小 RL training system。初期用 CPU 和小型离散环境验证语义，再逐步替换成真实 LLM、GPU 和分布式执行。这里是设计基线，不代表所有组件已经实现。前四课已有一个 [CPU Toy Bandit 参考实现](../reference_system/README.md)，覆盖采样、Reward Join、基础 REINFORCE、版本递增和 Checkpoint；图中的 Sampler、完整多轮 Trajectory、PPO/GRPO 仍属于后续扩展。

~~~mermaid
flowchart TD
    P["Prompt Source"] --> R["Rollout Worker"]
    R --> C["Trajectory Contract"]
    C --> S["Sampler / Batch Builder"]
    S --> L["Learner"]
    L --> O["Optimizer"]
    O --> V["Policy Version"]
    V --> M["Metrics / Checkpoint"]
    V --> R
~~~

| 组件 | 输入 | 输出 | 责任 |
|---|---|---|---|
| Prompt Source | 任务、环境、已有数据 | Prompt + metadata | 提供可追溯的任务身份 |
| Rollout Worker | Prompt、Policy Version | Response、结束原因、生成元数据 | 记录行为策略和采样设置 |
| Trajectory Contract | Rollout 结果、Reward | 可校验轨迹 | 检查字段、Shape、身份和组约束 |
| Sampler / Batch Builder | 已完成轨迹 | 训练批次 | 过滤、分组、补投和背压 |
| Learner | Batch、当前参数 | Loss、Gradient | 计算目标并检查数值 |
| Optimizer | Gradient、Optimizer State | 新参数 | 在明确的 step 边界更新权重 |
| Policy Version | 权重快照、版本号 | 可发布模型版本 | 避免训练和 Rollout 版本不可追踪 |
| Metrics / Checkpoint | 状态、指标 | 日志、快照 | 支持定位、恢复和对照实验 |

一条轨迹的建议状态为 created → generating → completed → validated → accepted/rejected → consumed。失败轨迹必须有明确的 failed 或 cancelled 状态；重复收到完成事件不能让同一条轨迹被重复计入训练批次。

第一阶段不实现真实大模型推理、分布式通信或异步抢占。先用 CPU 测试身份关联、组完整性、Mask 和版本检查。Day 10 将 Toy Bandit 演化为可测试的最小 PPO/GRPO 训练循环。

验收标准：轨迹 ID 唯一；Reward 按 ID 关联；Group 完整性与版本策略显式配置；无效 Mask、非有限 Reward 和重复消费会被捕获；Checkpoint 能恢复参数、Optimizer State、step 与 policy version。
