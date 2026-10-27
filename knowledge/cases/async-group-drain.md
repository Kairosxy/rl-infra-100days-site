# Group 收齐了，为什么 GPU 还在被 Rollout 占用？

同一个 Prompt 的四条回答完成时间分别为 2、3、4、9 秒。Group 的 Reward/Advantage 在 9 秒之后才能按完整四样本计算。另一组可能更早收齐，为什么它一定要等这组完成？

这是**组内算法依赖**与**外层批等待**的区别。如果系统等全部 Batch 完成才统一筛选，就会人为增加等待；异步 DataSampler 可以在某个 Group 完整后立即筛选与接收。

## 三个时间点不要混

1. **停止投新请求**：认为可用数据与在途预计贡献足够。
2. **达到目标 Accepted Groups**：训练数据数量已达到预设要求。
3. **所有 Inflight Requests 排空**：生成/工具调用真正都结束。

当前设计如果没有 Abort，达到训练目标时仍可能有大量在途请求，它们必须自然完成。此时 GPU Busy Time、尾部成本不会凭空消失。

## 并发不是免费午餐

按照历史接受率补投能减少有用样本短缺的等待，也会误估接受率、增加额外请求；按先完成的 Group 筛选还可能使长尾或困难 Prompt 在最终训练数据中占比下降。

如果训练与 Rollout 同时进行，要记录真实 Behavior Policy Version；一条 Trajectory 跨 Weight Update 继续生成，可能不再对应单一版本。Partial Rollout 的恢复状态也不只是 Token Cursor，还可能包括 KV、Sampler RNG、Tool Session。

## 验证方式

模拟两个 Group 的完成时间，分别记录 `admitted/completed/accepted/inflight/drained` 与权重版本。核对 “Accepted 达标” 和 “Drain 完成” 是否被错误合并，再测同等质量下 GPU-hour 的实际收益。不要用一条构造时间线声称异步方案在所有负载下更快。

实现对照：[verl HybridFlow](https://verl.readthedocs.io/en/latest/hybrid_flow.html) · [AReaL 异步训练](https://areal-ai.io/AReaL/en/algorithms/async.html)。