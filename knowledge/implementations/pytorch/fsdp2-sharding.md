# FSDP2 参数生命周期：为什么参数分片了，Forward 仍会 OOM？

8 张 GPU 都只保存约 1/8 的模型参数，训练却在第一次大块 `All-Gather` 附近 OOM。问题往往不在分片比例，而在 **`fully_shard` 的参数组边界，以及完整权重与其它显存对象同时存活的时间窗口**。FSDP2 主要分片参数、梯度和优化器状态；它不会像 Megatron TP 那样自动把一个 GEMM 改成跨卡分算。

## 参数组决定一次要展开多少权重

按 PyTorch **v2.14.0** 的 [`fully_shard` 实现](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py)，每次调用会把该模块尚未被子模块调用管理的参数归入同一个 FSDP 参数组。计算之前，这组参数执行 `All-Gather`，形成当前算子需要的未分片视图。官方源码明确建议从子模块向根模块（bottom-up）调用，而不是仅在最外层调用一次。

取一个仅用于容量推演的模型：四个 Block，每个 Block 的 BF16 权重 **4 GiB**，总计 **16 GiB**；`world_size=8`，各 Rank 均匀分片；每卡 HBM 为 **24 GiB**。再假设 Forward 开始时，每卡还有 **9 GiB** 的其它同时存活分配（可能包含激活、优化器状态及工作区，具体构成不是实测）。每卡参数分片常驻量为 `16 / 8 = 2 GiB`。

| 包裹方式与执行点 | 正在存活的完整权重视图 | 估算同时存活载荷 | 24 GiB 容量判断 |
|---|---:|---:|---|
| 只调用 `fully_shard(model)` | 16 GiB（整个 root 组） | `2+9+16=27 GiB` | 已超出 |
| 逐 Block 分组，且只展开当前 Block | 4 GiB | `2+9+4=15 GiB` | 尚有余量 |
| 逐 Block 分组，当前与下一 Block 预取重叠 | 8 GiB | `2+9+8=19 GiB` | 余量缩小 |

这是**将分片与展开视图按独立活跃载荷计入的教学预算**，不是 PyTorch 的 `max_memory_allocated` 实测值。真实实现的存储复用、混合精度、预取时机、通信缓冲与 CUDA Allocator 会改变峰值；`9 GiB` 也不是由参数规模推算出来的。表格只证明一个机制：**即便参数数目、分片比例、单卡容量都不变，参数组粒度仍可能改变是否 OOM**。如果真正的峰值来自 Activation 或 Optimizer Step，逐 Block 分组未必解决。

## Forward / Backward 的实际状态转换

对一个单独管理的非 root Block，默认 `reshard_after_forward=True` 时，状态沿计算时间流转如下。表格中的参数归属与 hook 语义来自上述 v2.14.0 源码；显存大小仍取决于实际模型。

| 时点 | 参数视图与所有权 | 通信或释放 |
|---|---|---|
| 两轮计算之间 | Rank 保留自己的 DTensor 参数分片 | 无逐层 All-Gather |
| Forward 进入 Block | 聚合本组参数，生成完整计算视图 | `All-Gather` |
| Forward 离开 Block | 释放完整视图，保留参数分片 | Reshard/释放，不需要另一次参数 All-Gather |
| Backward 即将计算 Block 梯度 | 再次获得完整参数 | `All-Gather` |
| Block 梯度计算完成 | 梯度归约后按 Rank 分片 | `Reduce-Scatter` |
| Optimizer 更新 | 本地持有者更新分片参数及其状态 | 通常在本地；组合策略可能有额外同步 |

**影响决策的开关**是 `reshard_after_forward`，不是简单的“FSDP 开/关”。在该版本中，显式设为 `True` 会在 Forward 后释放未分片视图，并在 Backward 前再聚合；显式设为 `False` 会保留它，以减少反向重新聚合的通信，代价是驻留显存。参数未显式指定（`None`）时，源码通常对非 root 模块设为 `True`、对 root 设为 `False`。

因此，**只把 root-only 的 `reshard_after_forward=False` 改成 `True`，不会移除 Forward 入口展开整个 root 参数组的峰值**。要区别“组太大”与“保留时间太长”，先只改包裹粒度，再单独改 Reshard 策略。有关不同 Torch 版本的行为不能从此文反推，必须按目标版本核对源码。

## 显存下降不意味着参数通信字节下降

如果将一个大小为 `S` 的完整 BF16 参数组在 8 Rank 上执行 Ring `All-Gather`，理想情况下每卡发送数据约 `(8-1)/8 × S`。以上面 16 GiB 模型为例，**root-only 的一次 Forward All-Gather 约需每卡发送 14 GiB**；拆成四个 4 GiB Block，各约 **3.5 GiB**，四次相加还是 **14 GiB**。

上述是 Ring 算法的**单 Rank 传输量模型**，不是总网络流量，更不是通信耗时预测。逐 Block 分组的优势在于缩小同时展开的权重窗口、允许相邻组的通信与计算交错；代价是启动更多 Collective，且预取可能增加同时活跃的参数。v2.14.0 的 [`FSDPCommContext`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py) 确实为 `all_gather_copy_in_stream`、`all_gather_stream` 和 `reduce_scatter_stream` 设置了不同的流，但**存在这些流不等于实测已经重叠**。是否受带宽、启动延迟或 Stream 等待限制，要看真实 Trace。

若非 root Block 在 Forward 后 Reshard，还要考虑 Backward 前重新 `All-Gather` 的另一轮通信；梯度 `Reduce-Scatter` 是单独的载荷。若配置了 HSDP、Offload、激活重算或混合精度，需重新确认通信组、dtype 与生命周期，不能照搬本段 14 GiB 数字。

## 在代码中只改变包裹边界

下面两个分支分别表达 root-only 与逐 Block 包裹方式，调用名来自 PyTorch v2.14.0。它们是**未运行 GPU 测试的最小对照片段**；需由同一个 Process Group 的所有 Rank 以一致顺序执行，并在 `fully_shard` 之后构造 Optimizer。

```python
from torch.distributed.fsdp import fully_shard

# model.blocks 是按 Forward 顺序排列的 ModuleList；mesh 是分片的 DeviceMesh。
if group_by_block:
    for block in model.blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=True)

fully_shard(model, mesh=mesh)  # root 管理仍未被子模块声明的参数
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

**反事实验证**：固定相同模型权重、Input Shape、dtype、有效 Batch、Rank/节点映射、Optimizer 和预取设置。只切换 `group_by_block=False/True`，分别记录每个 Rank 的 `torch.cuda.max_memory_allocated()`、`max_memory_reserved()`、完整参数组的实际字节数和 NCCL Timeline。假设是 root-only 的大 All-Gather 导致峰值，则切成 Block 后，Forward 峰值和 All-Gather 消息大小应同时改变；如果只是通信事件变多、峰值不降，应查残留 root 参数、预取重叠、激活与优化器状态。

还要把第一次 Forward、稳定迭代、Optimizer Step 分开测量：Adam 状态可能延迟到首次 Step 才创建。不要用第一次 Forward 的空闲显存去预测后续 Step；`reserved - allocated` 也不能单独证明碎片化就是 OOM 根因。上述验证是可运行方案，**没有 CUDA 设备、固定实际模型和 Nsight/Profiler 输出之前，不得写“已解决 OOM”或“降低 XX% 显存”**。

如果最终必须结合 Megatron TP，先分清两个并行域：TP 按矩阵维度分配 GEMM，FSDP Mesh 则在其管理的 Data Parallel 维度分片参数状态。一个“有效分片倍数”无法同时表示 GEMM FLOPs、Forward 参数峰值和通信拓扑，需分别打印 TP Group 与 FSDP Mesh 的 Rank 成员。

参考：[`fully_shard` v2.14.0 源码及调用/默认 Reshard 约定](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py)、[`FSDPCommContext` v2.14.0](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py)、[PyTorch FSDP2 官方文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)。
