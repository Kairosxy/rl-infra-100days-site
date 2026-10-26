# Day 02 · 5 分钟复习：显存峰值来自谁

**把这笔账闭眼算出来。** 1B 参数，BF16 参数与梯度各 2 Byte/参数，Adam 两份 FP32 Moment 各 4 Byte/参数，总计 **12 GB ≈ 11.18 GiB**。若另存 FP32 Master Weights，加 4 GB。这里还没算 Activation、Workspace、通信缓冲区，也没有考虑 FSDP/ZeRO 分片。

**算一张矩阵。** 朴素显式 Attention Score 形状 `[B,H,L,L]`。`B=2,H=16,L=640,2 Byte` 时为 **25 MiB**；长度 8320，是原来的 13 倍，对应大小约 **4.13 GiB**。这些数字**不是 FlashAttention 的真实显存峰值**。

**拿一个具体 OOM 做判断。** 如果初始化和 Forward 正常，但 Backward 才 OOM，优先核查保存的 Activation、梯度和临时工作区在什么时刻重叠。Checkpointing 只改变需要保存/重算的中间数据，不会直接消除 Adam Moment。

**最容易误讲的一句话。** FlashAttention 减少完整 Attention 中间矩阵的 HBM 读写与物化，并不使稠密全 Attention 的算术成本自动变成线性。

本课的容量计算和假设在 [CPU 脚本](../../examples/day_02/memory_budget.py) 中可逐项修改。
