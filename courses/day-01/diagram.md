# Day 01 · 图解：身份在哪一段容易丢

![Rollout、Reward、Sampler 与 Learner 的数据流](assets/rl_training_loop.svg)

这张图只读**数据的归属**，不把箭头解释成串行的执行时间表。Rollout 可以并发生成，Verifier 可以乱序返回；Sampler 必须把 `trajectory_id`、Reward、Mask、Group 和生成时的版本重新对应起来。之后的数据过滤、批次划分都不应该改变这种关联。

对照讲义中的四条样本：Verifier 按 A4、A1、A3、A2 完成，不等于 Trainer 应按这个顺序绑定 Reward。图中的连线表示逻辑数据依赖，并未画出独立的 GPU 资源池、并发数或某种特定的队列实现。

另一处边界是 **Group 完成 vs Batch 完成**。Group A 的四条有效轨迹已经完整时，它的组内统计可以计算；不代表必须立刻跨越 Trainer 的 mini-batch、梯度累积和版本约束。后续异步章节才会讨论如何安排这个执行时间。
