# Day 02 · 1B 模型的权重只有 2 GB，为什么训练还会 OOM？

假设一张 GPU 有 24 GiB 显存。一个 1B 参数的模型，BF16 权重约占 2 GB，看上去余量很大。但训练一开始，显存账单还会陆续出现梯度、Adam 状态、激活和算子临时区。更麻烦的是，这些对象并不在同一时刻产生或释放，**OOM 发生在哪个阶段，会直接影响排查方向**。

本课只研究训练显存的来源与生命周期；GPU 执行效率、Kernel 融合和 Roofline 留到后面的性能模块。

## 第一步：把静态模型状态算清楚

给定 $N=10^9$ 个参数，先假定单个训练 Rank 保存：

| 状态 | 本例的精度 | 存储估算 |
|---|---|---:|
| Parameters | BF16（2 Byte） | 2 GB |
| Gradients | BF16（2 Byte） | 2 GB |
| Adam 一阶矩 $m$ | FP32（4 Byte） | 4 GB |
| Adam 二阶矩 $v$ | FP32（4 Byte） | 4 GB |
| **合计** | | **12 GB ≈ 11.18 GiB** |

这里使用十进制 GB；GPU 工具经常以二进制 GiB 显示，不要把 12 GB 误读为 12 GiB。

上表是带明确精度假设的账本，**不是所有 PyTorch/Megatron/DeepSpeed 配置的通用事实**。不同 Optimizer、混合精度实现和分片策略可能更改 Gradient、Master Weights 的存储形式。有些配置还保存一份 FP32 Master Weights，此时多 $4$ GB，总计变成 **16 GB**；通信 Buffer、CUDA Context 和内存碎片尚未计入。

普通 DDP 主要负责梯度同步，各 Data Parallel Rank 通常有完整模型状态副本。FSDP/ZeRO 可以分片不同类别的状态，但不能因此断言「训练显存直接除以 DP 数」：参数 All-gather 的瞬时峰值、未分片的 Activation 和框架缓冲区仍然存在。

## 第二步：解释 Forward 为什么开始占住显存

看一个 Linear：

$$
Y=XW,\quad
\frac{\partial L}{\partial W}
=X^\top\frac{\partial L}{\partial Y},\quad
\frac{\partial L}{\partial X}
=\frac{\partial L}{\partial Y}W^\top .
$$

Backward 要计算权重梯度，至少需要与 $X$ 等价的前向信息。Autograd 因此会按算子规则保存 **Backward 需要** 的 Tensor；不是 Forward 中所有临时 Tensor 都会被保留，也不保证一律保存原件。

Transformer 里，Norm、MLP、Attention 都会产生这种需要保存或恢复的信息。多个 Block 串起来后，一部分 Activation 在早期 Forward 生成，却要等到后面的 Backward 才能释放。峰值通常出现在这些对象与梯度、临时 Workspace 重叠的时候，而非只取某个张量的大小。

**Activation Checkpointing** 改变的是这条生命周期：前向只保留选定边界所需的状态，反向经过该区域时重新执行计算，恢复缺失的中间值。它以额外 Forward 计算换取显存，不会直接消除 Adam 的 Moment。如果主要瓶颈是 Optimizer States，单纯扩大 Checkpoint 范围可能几乎不解决问题。

PyTorch 实现还要注意 RNG、是否显式设置 `use_reentrant` 等执行语义；不能把重新执行视为没有代价的缓存命中。

## 第三步：为什么长序列要先检查 Attention Backend

标准全 Attention 中：

$$
S=\frac{QK^\top}{\sqrt{d_h}},\qquad
O=\operatorname{softmax}(S)V.
$$

如果完整物化 Score 或 Probability，其形状是 `[B, H, L, L]`，仅这一份矩阵的存储为：

$$
M_{\text{score}}=B\times H\times L^2\times\text{element bytes}.
$$

设 $B=2$、$H=16$、BF16 每元素 2 Byte：

- $L=640$：$2\times16\times640^2\times2=26\,214\,400$ Byte，即 **25 MiB**。
- $L=8320$：长度扩大 $13$ 倍，元素数变为 $13^2=169$ 倍，即约 **4.13 GiB**。
- 若 $L$ 从 4K 到 32K，完整矩阵的元素数扩大 **64 倍**。

这几个值只针对**一份显式的 $L\times L$ 张量**，既不是整个 Transformer Block 的 Activation，也不是当前 GPU 的实测峰值。真实实现可能保存不同精度的中间结果，甚至完全不物化这张矩阵。

FlashAttention 通过分块计算和在线 Softmax，减少 HBM 中对完整 Score/Probability 的读写与保存。它优化了 IO 和中间存储，**没有把标准稠密全 Attention 的算术复杂度从 $O(L^2)$ 改为 $O(L)$**。如果服务实际采用 FlashAttention/SDPA 的高效 Backend，用朴素 $L^2$ 存储公式估算实际显存就会高估这一部分；其他随 $L$ 增长的激活仍需计入。

## 对着 OOM 的时间点排查

假设模型在初始化成功、第一轮 Forward 成功后，**第一轮 Backward 才 OOM**。不应马上把所有模型状态除以 GPU 数，而应检查在峰值时哪些对象同时存活：

- 初始化期间：对照 Parameter、Optimizer States、Master Weights 和分片配置。
- Forward 峰值：查 Micro-batch、序列长度、Attention Backend、MLP Activation。
- Backward 峰值：查保存的 Activation、重计算范围、Gradients 和临时 Workspace 的重叠。
- 多轮执行后逐步增长：检查保留了带计算图的 Tensor、异步引用或缓存的生命周期。

用 PyTorch 的 `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()` 分开看 Tensor 分配峰值与 CUDA caching allocator 预留量。它们不能完整解释驱动、NCCL 或其他框架管理的显存，所以同时需要进程级显存信息与 Profiler。这里给的是排查路径，不声称在任何机器上跑过 1B 模型。

## 用 CPU 算一遍，再故意改坏一个假设

本课已有一个**不会分配大 GPU 张量**的显存账本脚本：

~~~bash
python examples/day_02/memory_budget.py
python -m unittest discover -s examples/day_02 -p 'test_*.py' -v
~~~

可以检查的关键输出是 `25.00 MiB`、`4.13 GiB`、`169.0x`、`12.0 GB`。然后做两个改动并预测结果：将 `master_weight_bytes` 从 0 改为 4，账本增加 4 GB；把 $L$ 从 640 改为 1280，朴素 Score 存储应变成四倍。

最后辨认一个反例：如果你在已采用 FlashAttention 的模型上观察到序列长度翻倍、显存增长不到四倍，这并不违反上式。上式描述的是**显式物化的一张矩阵**，不是任意 Backend 的 GPU 峰值。

下一课从显存转到梯度。你将看到：Micro-batch 即使都能成功执行，只要 Loss Reduction 不对，训练得到的梯度也可能不是预期的那一个。

## 查证资料

- [CPU 显存预算代码](../../examples/day_02/memory_budget.py) 与 [对应测试](../../examples/day_02/test_memory_budget.py)：可核对的只是容量算式，不是硬件测量。
- [PyTorch Activation Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)：保存、重计算和 RNG/Backward 语义。
- [PyTorch Scaled Dot Product Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：实际执行的 Backend 与形状。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)：IO-aware 的准确边界。
