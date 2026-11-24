# GPU 执行模型、显存与算术强度

GPU 不是把一条 Python 语句变成一个大核执行。CPU 发起 Kernel，GPU 将线程块分配到 SM（Streaming Multiprocessor）。CUDA 线程在一个 Block 内以 32 线程组成 Warp；分支不一致可能让部分 Lane 在同一段指令中不做有效工作。

## 从计算到数据搬运

| 层级 | 主要用途 |
|---|---|
| Register | 每线程的局部值、累加器 |
| Shared Memory | Block 内协作、Tile 暂存 |
| L1/L2 Cache | 硬件管理的复用；L2 通常跨 SM 共享 |
| HBM / Global Memory | 大规模权重、激活、KV 等载荷 |
| Host Memory / 网络 | Offload、数据输入、跨设备通信 |

Tensor Core 处理特定数据类型和矩阵块的运算。Tile 将矩阵的一部分放进近端存储重复使用；显存带宽有限时，这种复用比单纯增加 FLOPs 更重要。Occupancy 表示可驻留 Warp 与硬件上限的比例，不等于 Tensor Core 利用率。

## Roofline 的数量级判断

对 $Y=XW$，取 W=[4096,4096]、BF16、X=[M,4096]。忽略 Bias，理想 HBM 最小载荷取一次读取 X、W 和一次写入 Y。

| M | FLOPs | 最小载荷（近似） | 算术强度 |
|---:|---:|---:|---:|
| 1 | 33554432 | 32 MiB + 16 KiB | ~1 FLOP/B |
| 1024 | 34359738368 | 48 MiB | ~683 FLOP/B |

假设某 GPU 的有效计算上限 100 TFLOP/s、有效带宽 1 TB/s（教学参数），两条上限相交在 100 FLOP/B。M=1 的最低时间主要由约 33.6 μs 的访存项决定；M=1024 的计算下界约 344 μs，访存下界约 50.3 μs。

$$
T_{\rm lower}\geq\max(F/P,\;D/BW).
$$

这是简化下界，不是 GPU 实测。实际 HBM 流量可能因 cache、重复加载、量化 scale、工作区及实现而改变。大 M 也可能受低占用率、通信、显存容量或算子效率限制。

## 从估算走到 Trace

先记录实际 GEMM 的 M/N/K、dtype、FLOPs 和传输量，再用 Profiler 看 kernel 时长、DRAM 吞吐及计算单元使用。GPU busy 比例很高并不能证明计算饱和，通信 Kernel 或访存停顿也会占用设备。读 [Prefill / Decode](../serving/autoregressive-serving.md) 与 [端到端吞吐](../optimization/rl-throughput.md) 时，应沿用这套成本口径。

参考：[CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)、[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。
