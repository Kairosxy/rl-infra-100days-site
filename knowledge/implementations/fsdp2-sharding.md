# FSDP2：参数分片后，为什么还有显存峰值？

FSDP2 通过 DTensor 对参数沿指定 Mesh 分片，在需要计算的时间点执行 All-gather，反向后可 Reduce-scatter Gradient。这种方案降低了常驻参数和优化器状态成本，但计算期间仍需要完整计算视图和中间 Activation。

## 典型误解

“参数只占 1/N 显存”不等于“峰值显存也只剩 1/N”。Activation、临时 All-gather Buffer、Comm Bucket、Allocator 碎片都可能成为 OOM 的真正原因。

与 Megatron Tensor Parallel 相比，FSDP2 主要关心分片参数的生命周期；TP 将算子本身的计算沿 Tensor 维度切开，通常还需相应的 Collective。两者可以组合，并非按框架名二选一。

## 一个有效实验

固定相同模型、Micro-batch 与 Sequence Length，记录峰值 HBM 和 All-gather/Reduce-scatter Timeline，再切换不同 Shard Placement 与 Activation Checkpointing。比较前同时锁定 Optimizer State DType 与 Param/Grad DType，否则显存对照不公平。

参考：[PyTorch fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)。