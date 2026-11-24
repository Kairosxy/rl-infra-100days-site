# Learner 变快，为什么 RL Step 几乎不变？

假设一个完全串行的训练 Step 耗时 60 秒，其中 Rollout/Reward 数据供给占 42 秒，Learner 训练占 18 秒。

将 Learner 加速两倍：

$$
T_{\mathrm{step,new}}=42+\frac{18}{2}=51\ {\rm s},\quad
S=\frac{60}{51}\approx1.176.
$$

如果宣传“训练速度翻倍”，很可能只指 Learner 部分；整轮 RL 的改善只有约 17.6%。这属于 Amdahl 规律下的推导，不是现场 Benchmark。

## 如何定位

按照实际关键路径逐阶段记录：Rollout Prefill、Decode、Tool、Verifier、筛选、Weight Sync、Train Forward/Backward、Collective、Optimizer，尤其注意批式最慢请求的 Tail Latency 与异步模式 Inflight Drain。

还需要记录有效数据产出，而不只是生成完成数。一个数据采样配置可能让 Completed Requests 增多，但被 GRPO Group 筛选后真正进入 Learner 的 Accepted Groups 没有增加。

## 方案如何选择

如果瓶颈在推理长尾，改善 Learner FP8 可能收益有限；适合关注 Continuous Batching、Group 补投、资源分池/共享、Partial Rollout。若瓶颈在 Learner 的 GEMM/通信，才优先考虑 FP8 Recipe、TP/EP 策略、FlashAttention/Kernel Fusion。

改动前定义固定负载、固定目标质量与 GPU-hours，改动后既比较 Step/Token 吞吐，也检查 Reward、KL、Staleness 与排空成本，避免把质量变化误认为性能提升。

参考：[verl 性能调优](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) · [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)。