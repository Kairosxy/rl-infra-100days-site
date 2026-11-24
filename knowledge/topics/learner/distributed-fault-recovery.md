# 训练异常、分布式故障与一致恢复

分布式任务的最后一条报错经常只是故障扩散的结果：Rank 3 在某个 Forward 中 OOM，其余 Rank 继续进入 Collective，最终同时报 Timeout。应先定位最早失效的进程和阶段，再判断是否涉及数值、资源、通信或协议。

## 定位首次偏离

| 现象 | 最小证据 |
|---|---|
| Loss Spike / NaN | 首个非有限 Tensor、有效 Token 分母、Scale、Gradient Norm 和对应样本 |
| OOM | 活跃分配与 reserved 峰值、参数组、Activation、工作区及发生时刻 |
| NCCL Hang / Timeout | 各 Rank 最后一条已完成/已进入的 Collective、Group、Shape 和此前异常 |
| 静默质量回归 | 固定权重与输入对照 Logprob、Mask、路由、梯度及数据版本 |

不同 Rank 的日志时间可能存在偏移；应结合 Step、任务 ID、Collective 序号及时间线，而非只按墙钟排序。NCCL Timeout 也可能来自网络故障或 Collective 顺序不一致，不能一概归因为 OOM。

## 逻辑更新与恢复点

假设 Checkpoint 保存已成功提交的 v100，随后 Learner 成功更新到 v101，但部分 Rollout Rank 尚未安装新权重。如果训练进程此时失败，恢复 v100 后，原来标成 v101 的样本不能仅凭整数版本相同就视为新进程再次得到的 v101；模型参数可能不同。恢复需要能够区分运行世代、提交版本与 Checkpoint 身份。

一致的恢复状态通常包含参数、Optimizer、LR Scheduler、随机状态和数据进度。RL 还需明确已消费组、在途样本、评分结果、权重安装与发布状态。可以选择丢弃不可验证的在途数据，或按持久化身份与幂等契约恢复；保存范围必须和恢复语义匹配。

对于分片模型，DCP（Distributed Checkpoint）提供多 Rank 保存与加载时重分片能力。TorchElastic 负责协调 Worker 失败后重启等流程；它不会自动替训练脚本恢复所有应用状态。异步保存的 staging、写入完成和可恢复提交也应分别确认。

## 故障注入

先在 CPU 或小规模环境验证三个边界：一个 Rank 在归约前退出；同一评分结果重复返回；权重只安装到部分 Worker。期望行为分别是统一结束或协调恢复、结果按身份去重、未完成版本不得准入。恢复后对照逻辑 Batch、参数摘要与更新计数。再扩展到真实 NCCL 和分布式 Checkpoint 负载，记录恢复耗时及丢失/重复样本数。

参考：[PyTorch Elastic Error Propagation](https://docs.pytorch.org/docs/stable/elastic/errors.html)、[Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)、[梯度更新一致性](gradient-equivalence.md)、[策略发布](../async/weight-publication.md)。
