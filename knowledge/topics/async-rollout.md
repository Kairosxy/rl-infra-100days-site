# 异步 Rollout：减少什么等待，又增加哪些错误机会？

同步训练里存在不可省略的依赖：同一 Prompt 的 Group Advantage 需要有效组完整、Learner 需要当前更新定义下的 Batch、同一 GPU 不能无限同时承担互斥工作。但等外层整个 Batch 的所有请求都完成后才开始筛选，有时只是实现造成的额外屏障。

异步供给允许 Group 或请求完成后立即进入筛选池，并根据已接受样本、Inflight 数量和历史接受率决定补投。这里**停止投新请求、达到目标有效样本数、所有在途请求排空**，是三个不同时间点。没有 Abort 时，达到目标不等于 GPU 已空闲。

## 多出来的算法问题

Rollout 开始时使用 v10 权重，Trainer 可能已更新到 v11。记录在轨迹上的 Behavior Version 不能在消费时覆盖成 v11，否则 Ratio、Staleness 和恢复逻辑都失去事实依据。Partial Rollout 如果跨 Weight Update，更需要定义每个 Token 的生成版本与恢复状态。

同样重要的是 DataSampler 的选择偏差：如果慢请求更容易被丢弃、或者全同 Reward Group 更容易被过滤，最终训练样本的分布可能改变。增加吞吐还不够，需要在相近模型质量和相同负载下比较成本。

## 框架怎么取舍

[verl HybridFlow](https://verl.readthedocs.io/en/latest/hybrid_flow.html) 把 RL 数据流与不同执行 Engine 组合；[slime](https://github.com/THUDM/slime) 强调 Megatron/SGLang 与 Data Buffer 的集成；[AReaL Async RL](https://areal-ai.io/AReaL/en/algorithms/async.html) 明确用生成/训练重叠和 Off-policyness Budget 控制异步。不能据此声称谁在所有工作负载下更快。

## 可验证的调度实验

构造两个 Group 的完成时间：A 在 3s 完整，B 在 10s 完整。同步外层筛选必须等到 10s；异步供给可以在 3s 对 A 做组内统计，但**能否立刻执行训练更新**还取决于 Mini-batch、版本和资源。统计“可更早处理”不等于测得“端到端必定加速”。

建议至少记录 Accepted Groups、Inflight、Rejected、Policy Version Gap、Drain Time 和 GPU Busy Time。没有这几项，只有 Step Time 的对照很难解释结果。