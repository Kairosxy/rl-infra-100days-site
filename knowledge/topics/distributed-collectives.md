# 为什么多加 GPU，有时比单卡还慢？

先把分布式训练看成三个问题：每张卡持有什么状态、谁负责哪段计算、结果在哪个时间点必须一致。通信不是附加细节，而是并行算法的一部分。

```mermaid
flowchart LR
 A["前置：Rank / GPU / HBM"] --> B["Process Group"]
 B --> C["Collective：AR / AG / RS / A2A"]
 C --> D["DDP / FSDP / TP / EP"]
 D --> E["拓扑 / 延迟 / 带宽"]
 E --> F["NCCL Hang / 低利用率"]
```

## 同一个例子：8 GPU、一个 64 MiB 张量

假设每卡已有一个大小 64 MiB 的 BF16 梯度张量，8 卡构成一个通信组。Ring All-Reduce 的理想每卡传输量约 `2(n-1)/n × S`，即 112 MiB；Reduce-Scatter 或 All-Gather 各约 `(n-1)/n × S = 56 MiB`。以有效 100 GB/s 估算，112 MiB 的纯带宽下界约 1.17 ms。实际通信时间还包含启动、拓扑、竞争、算法切换与同步等待。这里的 S 是**每卡完整逻辑张量大小**，不能拿分片后的大小混用。

## 四个必须分清的 Collective

| 操作 | 输入 → 输出（每 Rank） | 解决什么问题 | 常见位置 |
|---|---|---|---|
| All-Reduce | 完整梯度 → 完整求和/平均 | 所有 Rank 得到一致归约结果 | DDP 梯度、TP Row Parallel |
| All-Gather | 本地分片 → 完整拼接 | 还原计算所需参数/激活 | FSDP 参数、SP 边界 |
| Reduce-Scatter | 完整部分和 → 归约后分片 | 归约并保留分片 | FSDP 梯度、SP |
| All-to-All | 按目的 Rank 的分片 → 按来源重排 | 重新分配 Token/数据 | MoE EP Dispatch |

**不变量**：参与同一 Collective 的 Rank 必须在兼容的通信组中按匹配的顺序调用，并满足各 API 的 Shape、dtype 和计数约束。一个 Rank 进入 All-Reduce、另一个 Rank 进入 All-Gather，可能造成 Hang 或报错。

## 从通信字节到真正的关键路径

粗略模型 `T_comm ≈ α × 启动次数 + bytes / B_effective + T_contention`。α 是启动延迟；B_effective 受 NVLink、PCIe、InfiniBand、跨节点拓扑与算法影响。单卡 GEMM 从 20 ms 降到 12 ms，但新增 11 ms 不可重叠通信，阶段从 20 ms 变为 23 ms；如果通信重叠，则需要以 Trace 的关键路径为准，不能直接求和。

**实现映射**：PyTorch `torch.distributed.new_group`、`all_reduce`、`all_gather_into_tensor`、`reduce_scatter_tensor`、`all_to_all_single`；NCCL 后端与 Process Group；Megatron TP/PP/EP/CP 子组。要记录 global rank、group ranks、op sequence、shape、dtype、bytes、enqueue/start/end 时间和 NCCL 错误信息。

**故障树**：所有 Rank 卡住 → 核对最后一个 Collective 的组/顺序/Shape → 检查有无 Rank 提前异常退出 → 再查网络和设备错误；只有跨节点变慢 → 检查跨节点边界和有效带宽；GPU 计算利用率低 → 区分 NCCL Kernel 时间与 CPU 调度等待。

三个最小验证：① 8 Rank 对 64 MiB 张量分别运行 AR/AG/RS 并核对输出；② 改变消息大小绘制延迟—字节曲线，识别启动延迟区；③ 在可控测试环境故意让一 Rank 缺少 Collective，验证超时日志能否定位，不要在生产任务注入故障。

**后续**：结合 [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html) 与 [NCCL 用户指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) 对照目标版本 API。
