# DP、TP、PP、EP 应该怎么选，而不是怎么背

一个并行策略至少改变三件事：每张卡保存多少状态、每次算多少矩阵、哪些边界必须通信。并行度增加不等于吞吐必然增长。

**DP** 将不同样本分给不同副本，同步梯度；**TP** 将模型内部矩阵拆分，需要在层内通信；**PP** 将层分到不同 Stage，产生 Pipeline Bubble 与跨 Stage Activation 传输；**EP** 将不同 Expert 分散在设备之间，引入 Token Routing 和 All-to-All。Context Parallel 则切分长序列 Attention 所需的计算/存储。

## 一个最小成本判断

假设某个 Parallelism 配置将单卡计算从 20 ms 降到 12 ms，却增加 11 ms 的不可重叠通信。新阶段理论时间约 23 ms，反而比原来慢。若通信能与计算重叠，真实关键路径还会变化，不能简单把每个计时指标相加。

FSDP2 属于另一种分片维度：它用 DTensor 对参数做分片，前向时 All-gather，反向后 Reduce-scatter Gradient。分片省去的是常驻状态，All-gather 仍会形成瞬时峰值。不能把 FSDP2 与 TP 看成只能二选一的标签。

## 工程选择的输入

给出参数数目、Attention/MoE 结构、序列长度、Batch、GPU 显存与互联拓扑之后，先算各 Strategy 的参数/Activation 峰值和 Collective，再谈吞吐优化。Megatron Core 的多维并行与 PyTorch FSDP2 的分片生命周期是重要对照，但实际选型仍需实验验证。

参考：[Megatron Core 官方文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)；[FSDP2 fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)。