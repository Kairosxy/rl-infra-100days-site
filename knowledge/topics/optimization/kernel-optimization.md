# Kernel Fusion、FlashAttention 与执行开销

GPU 性能优化首先要定位成本来自计算、数据搬运还是调度启动。Kernel Fusion 合并可兼容的算子，Tiling 提高近端存储复用，CUDA Graph 降低重复提交开销，编译器则决定如何生成和组合执行代码。这些优化可以组合，但不能混为一谈。

## 减少中间张量的读写

假设 y=relu(x+b)，x 有 100 万个 FP32 元素，b 已在 cache 或忽略其读取。拆成 Add 和 ReLU 两个 Kernel，最小载荷是读 x、写临时、读临时、写 y，共约 16 MB；融合后理想约 8 MB。实际有 cache、对齐、向量化等因素。Fusion 还可能增加 Register 压力或改变计算顺序，因此不能仅凭 Kernel 数减少推断提速。

## Attention 的 IO

假设单 Head、T=8192，完整打分矩阵 [T,T] 用 BF16 存储占 128 MiB；32 Head 则为 4 GiB（不计 batch 和其他中间结果）。标准 Attention 要对合法位置执行 QK 打分和归一化。FlashAttention 使用分块与在线 Softmax，在处理新块时维护运行最大值、指数和与加权结果，再合并得到完整归一化输出，避免将整个打分/概率矩阵写到 HBM。

例如两个分数 0、ln(3)，分块仍应得到整体权重 1/4、3/4；若每块各自 Softmax 后直接相加，则结果错误。在线合并需要相同的全局归一化尺度。FlashAttention 改善 IO 与中间存储，不会自动取消标准全 Attention 的平方打分工作量；滑窗等稀疏架构改变的是合法连接范围。

## 从实现选择到实测

Triton/TileLang 等提供编写和生成 GPU Kernel 的方式，CUTLASS 等提供可组合的矩阵运算实现；编译器可针对 Shape、dtype、布局做选择。CUDA Graph 则捕获/重放执行图，不等于 Fusion。动态 Batch、变长序列、padding、编译缓存和 graph bucket 都会影响实际路径。

先固定同一计算语义，比较前后最大误差、梯度（训练路径）、HBM 读写、算子次数、Kernel 时间及端到端吞吐。Kernel microbenchmark 要有 warmup 与同步；异步 Launch 的 CPU 调用时间不能直接当 GPU latency。只有当算子位于关键路径且实际节约足够成本时，才预期 RL Step 改善。

参考：[FlashAttention](https://arxiv.org/abs/2205.14135)、[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)、[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)。
