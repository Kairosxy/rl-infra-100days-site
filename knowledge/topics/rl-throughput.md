# RL Step 慢，先定位关键路径

“训练效率提升”至少有三个不同口径：Learner Tokens/s、Rollout Tokens/s、被接受的训练 Token / GPU-hour。它们可以同时变化，也可以互相冲突。

考虑串行同步 Step：Rollout=42s，Learner=18s，总计 60s。如果优化 Learner 使其快 2 倍，新的 Step Time=42+9=51s，端到端加速为 $60/51≈1.176$。这不是算法没用，而是 Amdahl 规律告诉你瓶颈还有 42s 没改。

如果系统允许重叠，关键路径又会改变。不能简单把 Worker 的独立计时累加为真实 Wall Time。更不能用“请求提前完成”推出“所有 Inflight 都结束”，后者还涉及资源排空。

## 最小可观察性

同一个 Step 需要能追踪：生成/工具/Verifier/筛选各阶段延迟、GPU Busy Time、Learner Forward/Backward/Optimizer 时间、通信同步、Weight Sync、接受率和 Policy Version Gap。

比较两种配置时固定模型、长度分布、Batch/Group 定义、设备和目标质量；否则提升的可能只是更短的 Response 或过滤掉了困难样本。

一个常见反例：吞吐看似提升 30%，但 Accepted Groups 数量不变而 Inflight 末尾堆积更多。它可能将排空成本推迟到了结束阶段，不是免费的收益。

实用资料：[verl Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)；[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)。