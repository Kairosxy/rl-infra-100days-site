# 为什么 PP 分了四张卡，仍然有 GPU 在等？

把 12 层 Transformer 均分到 4 个 Stage，每 Stage 3 层。输入 Batch 分成 m=8 个 Micro-batch。即使每个 Stage 计算时间相同，第一批数据也必须先经过 Stage0 才能到 Stage1；最后一批离开 Stage0 后，后面的 Stage 还在工作。

```mermaid
flowchart LR
 B["8 个 Micro-batch"] --> S0["Stage0 层 0–2"]
 S0 --> S1["Stage1 层 3–5"]
 S1 --> S2["Stage2 层 6–8"]
 S2 --> S3["Stage3 层 9–11"]
 S3 --> L["Loss / Backward"]
```

## 为什么拆层不能自动获得线性加速？

**直觉**：流水线只有在填满后才能让所有 Stage 同时工作；首尾不可避免地存在空闲。**不变量**：同一 Micro-batch 的激活与梯度必须沿对应 Stage 顺序传递；所有 Stage 的参数版本必须满足训练调度的更新约束。

理想化同步 Pipeline 的 Bubble 比例近似 `(p-1)/(m+p-1)`。p=4、m=8 时为 `3/11≈27.3%`；m=16 时为 `3/19≈15.8%`。这是用于理解的理想模型，不是所有 1F1B/Interleaved 实现的实际测量值。若 Stage 时间不均衡，还会产生额外等待。

**通信**：跨 Stage 发送 Activation 和反向 Gradient。若边界张量 `[micro_batch=1, seq=2048, hidden=4096]`、BF16，单次传输约 16 MiB；理想 100 GB/s 的纯带宽下界约 0.17 ms，未计链路启动、拓扑与并发拥塞。

## 训练与推理的差别

训练常用 1F1B 或 Interleaved 调度降低 Activation 常驻和 Bubble；推理的 Prefill 与 Decode 请求长度不同，Stage 负载和跨请求调度更复杂。不能把训练的 Micro-batch Bubble 公式直接作为在线 Serving 的延迟模型。

## 工程落点

Megatron Core 的 Pipeline Schedule、`pipeline_model_parallel_size`、Virtual Pipeline Stage 与 P2P Send/Recv 是阅读入口。检查 Stage 层数、不同层耗时（MoE/Attention 不一定均衡）、Micro-batch 数、Activation Checkpoint 与通信重叠。

| 对象 | 形状/单位 | 所在位置 | 观测指标 |
|---|---|---|---|
| micro_batch | 样本数 | Data Loader/Scheduler | m、global batch |
| activation | [1,2048,4096] | Stage 边界 | 16 MiB/次、P2P 耗时 |
| stage_time | ms | 每个 Pipeline Stage | 最大值、P95 |
| bubble | 时间占比 | 整个 Pipeline | Timeline 空闲区 |

排障路径：**GPU 阶梯状空闲** → 先识别填充/排空 Bubble → 再检查 Stage 不均衡；**增大 m 反而 OOM** → 检查 Inflight Activation 和重计算；**训练 Hang** → 检查各 Rank 的 Micro-batch 数与 P2P 顺序。

**三个对照**：p=2/4、m=4/8/16 的时间线；固定总 Batch 对比不同 Micro-batch；故意让一个 Stage 慢 2 倍观察尾部等待。

**交互图规格**：以横轴时间、纵轴 Stage 画泳道，主控只改变 m 与 Stage 数；每个 Micro-batch 的 Forward/Backward 方块标识相同 ID；显示理论 Bubble 与 Stage 不均衡的差异。优先时间线，不做装饰性 GPU 卡片。

参考：[Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)。
