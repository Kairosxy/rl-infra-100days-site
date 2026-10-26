# Day 02 · 面试追问

## 1B BF16 权重 2 GB，为什么训练显存会超过 12 GB？

假设一张卡保存完整参数、BF16 梯度和 FP32 Adam Moment。先列出静态账单，说明为什么 12 GB 不能当成进程峰值。再解释「有一份 FP32 Master Weights」和「改用 FSDP」分别改变哪一类占用。

**判别依据：** 能算出 2+2+4+4=12 GB（非 GiB），能把 Activation/临时区与分片状态分开。说「FSDP 把所有显存按卡数平均分」不合格。

**追问：** 为什么 OOM 发生在 Backward，而不是加载模型时？指出保存的前向信息与梯度、Workspace 重叠的时间关系。

## 长上下文从 4K 增加到 32K，FlashAttention 还会不会 OOM？

先计算完整 `[B,H,L,L]` 矩阵的元素数扩大多少倍，再说明为什么这个倍数不等于实际显存峰值的增长倍数。给出两种你会检查的真实配置，以及至少一个记录峰值的方法。

**判别依据：** 64 倍是显式矩阵元素数的比例；FlashAttention 不物化完整矩阵，但其他 Activation 仍增长。应核对 Attention Backend、Micro-batch、Checkpointing 配置，而非只背 FlashAttention 的名称。

**追问：** 如果已经开启 Activation Checkpointing，显存仍主要耗在 Optimizer States，上调重计算比例是否一定有效？说明为什么不一定。

## 一张「显存估算截图」能证明训练速度吗？

CPU 计算器报告 Attention Score 为 4.13 GiB、Optimizer 状态约 12 GB。有人据此断言「H200 必然比 H100 快多少倍」。指出缺失的关键信息，以及这份实验究竟验证了什么。

**判别依据：** 能区分容量预算、内核/带宽/吞吐实测以及硬件比较所需工作负载和测量环境。
