# FSDP2 分片生效，Forward 仍在 All-Gather 时 OOM

训练启动后显示参数已经分片，首个 Forward 却在 All-Gather 附近 OOM。先查**展开的对象到底有多大**：是整个 root 的权重一次性展开，是 Prefetch 让两组参数同时存活，还是 Activation / Optimizer 状态占用了剩余空间？这个现象并不能单靠 `world_size=8` 来判断。

## 从对象与时刻建立可证伪假设

这是**教学故障假设，非实际生产事件**。设 8 Rank，每 GPU 24 GiB HBM，模型有 4 个 BF16 Block、每块权重 4 GiB，合计 16 GiB。root-only `fully_shard(model)` 的 Pre-forward Hook 要求整个 16 GiB 组形成完整参数视图，而不是只需要本卡的 2 GiB 分片。

进一步假设当前 Rank 还有 **9 GiB 其它同时存活对象**（可能包含 Optimizer、Activation 和 Workspace，具体构成待测），且分片与完整视图分开驻留。root-only 的预算是 `2 + 9 + 16 = 27 GiB/Rank`；逐 Block 展开变为 `2 + 9 + 4 = 15 GiB/Rank`，若预取下一 Block，则为 **19 GiB/Rank**。9 GiB 是教学假设，不是由参数量计算出来的；有关参数生命周期的说明见 [FSDP2 实现](../implementations/fsdp2-sharding.md)。

## 两组对照只能改参数组

保持模型、输入形状、Batch、dtype、Optimizer、DeviceMesh、随机种子一致；一组只对 root 调用 `fully_shard`，另一组先对四个 Block 分别调用，再对 root 调用。**两组都显式指定 `reshard_after_forward=True`**，否则 root 默认的重分片策略会成为额外混杂变量。关联实现文章给出最小包裹对照片段；完整实验仍需要本地模型初始化、进程组及 Profiler 配置。本仓库尚未执行 8-GPU 测试。

| 时间与记录对象 | 若 root 大组是主因，应该观察到 | 如果不成立，继续核查 |
|---|---|---|
| 首个 Forward 前 | root-only 一次 All-Gather 消息显著大于 Block-wise | 是否存在未分片参数或权重加载副本 |
| Block 边界 | Block-wise 参数展开随执行推进；峰值降低 | Prefetch 是否让多个 Block 同时保留完整视图 |
| Backward / Optimizer | 二次 All-Gather、Grad Reduce-Scatter、Moment 初始化各有独立时刻 | Activation、梯度累积、Optimizer State dtype |

记录每个 Rank 的参数组名称、组内 BF16 字节、`max_memory_allocated()`、`max_memory_reserved()`，并在 Profiler 中定位峰值前后的 Collective。不要先调小 Batch 或调 Allocator 参数：前者同时改变 Activation，后者不能证明活跃参数已经能够放下。

## 反例：Reshard 不修复 Forward 入口的大组

把 root 的 `reshard_after_forward=False` 改为 `True`，只能缩短 Forward 后完整参数的存活时间，却**不能消除 Forward 之前展开整个 root 组**。如果 OOM 确定发生在首次 All-Gather，这个改动并没有击中根因。

按 8 Rank Ring 模型，16 GiB 权重的 Forward All-Gather 总发送量仍约 **14 GiB/Rank**，不论将其划成一次 16 GiB 还是四次 4 GiB 的组；分组减少的是**同时展开峰值**，不是传输总字节。若逐 Block 后 OOM 从 Forward 挪到 Backward，下一步应定位保留的 Activation 和 Backward 预取，而非立刻加大 FSDP Degree。

停止条件：能指出导致峰值的具体参数组、它的完整字节数、同期存活的 Activation/预取对象，以及 A/B 后的峰值与数值正确性。框架行为以 [PyTorch v2.14.0 `fully_shard` 源码](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py) 为依据；这里所有 24 GiB 预算均为推导，**没有 GPU 实测**。
