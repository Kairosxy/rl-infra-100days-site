# Day 02 · 图解：前向留下了什么，反向要用什么

![Transformer 训练时的参数、激活与反向状态](assets/transformer_memory.svg)

读这张图时，重点找 **Tensor 生命周期**：某些 Activation 在 Forward 中出现，却要到 Backward 才能释放；Gradient 和算子 Workspace 又在后续阶段产生。显存峰值取决于这些对象在时间上的重叠，不是「模型权重 + 一份最大矩阵」就能完整解释。

图中的 Attention 显式 Score 形状 `[B,H,L,L]`，对应讲义中的 25 MiB / 4.13 GiB 计算。它是朴素存储的成本模型；实际是否物化要核对 Attention Backend。

Activation Checkpointing 用反向重计算替代保存某些中间激活。它改变的是图中保存/重算的路径，不会使 Optimizer State 自动变小。比较 Checkpointing 前后的显存，应该记录具体前后向阶段和峰值，而不是只比较模型文件大小。
