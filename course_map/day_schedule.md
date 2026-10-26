# RL Infra 100 天课程表

这份课表定义每一天的学习主题和前置概念。课程自 2026-10-23 起按北京时间每天 09:00 自动发布，不以是否作答、是否完成实验或是否通过测试为条件。已发布课程可以自由跳转、提前浏览、补课和重复复习。完成情况与发布日期分开记录。

| Day | 主题 | 对应阶段 |
|---:|---|---|
| 01 | RL 端到端数据流与不变量 | Module 1 |
| 02 | Transformer 前反向与显存 | Module 1 |
| 03 | Autograd、AdamW 与梯度累积 | Module 1 |
| 04 | Policy Gradient 与 Logprob | Module 1 |
| 05 | Value、TD Error、GAE 与偏差—方差 | Module 1 |
| 06 | PPO Ratio、Clip 与 KL | Module 1 |
| 07 | GRPO、DAPO 与组内 Advantage | Module 1 |
| 08 | 最小 Rollout Loop、Trajectory 与 Action Mask | Module 1 |
| 09 | 最小 Learner Loop、Logprob 与版本检查 | Module 1 |
| 10 | 端到端最小 PPO/GRPO 与 Checkpoint | Module 1 |
| 11 | GPU 架构与 Roofline | Module 2 |
| 12 | GEMM 与计算强度 | Module 2 |
| 13 | FlashAttention 与 IO | Module 2 |
| 14 | Triton 融合算子 | Module 2 |
| 15 | Activation Checkpointing | Module 2 |
| 16 | BF16、FP8 的缩放与误差 | Module 2 |
| 17 | MoE Routing 与 Grouped GEMM | Module 2 |
| 18 | CUDA Graph 与 Stream Overlap | Module 2 |
| 19 | Nsight 与 PyTorch Profiler | Module 2 |
| 20 | GPU 性能基准与瓶颈报告 | Module 2 |
| 21 | ProcessGroup 与 DeviceMesh | Module 3 |
| 22 | DDP Bucket 与梯度同步 | Module 3 |
| 23 | FSDP2、ZeRO 与分片 | Module 3 |
| 24 | TP 列并行与行并行 | Module 3 |
| 25 | PP 1F1B 与 Pipeline Bubble | Module 3 |
| 26 | EP、All-to-All 与负载均衡 | Module 3 |
| 27 | CP 长上下文通信 | Module 3 |
| 28 | NCCL、RDMA 与 NVLink 拓扑 | Module 3 |
| 29 | 分布式 Checkpoint 与 Reshard | Module 3 |
| 30 | 8–64 GPU 并行方案设计 | Module 3 |
| 31 | Prefill、Decode、TTFT 与 TPOT | Module 4 |
| 32 | KV Cache 容量规划 | Module 4 |
| 33 | Continuous Batching 与调度 | Module 4 |
| 34 | PagedAttention 与碎片 | Module 4 |
| 35 | RadixAttention 与 Prefix Cache | Module 4 |
| 36 | MTP 与投机解码正确性 | Module 4 |
| 37 | FP8/FP4 训推一致性 | Module 4 |
| 38 | PD 解耦与 KV RDMA | Module 4 |
| 39 | MoE 服务的 TP/EP 调度 | Module 4 |
| 40 | 推理吞吐与延迟对照实验 | Module 4 |
| 41 | 采样 Schema 与 Group 不变量 | Module 5 |
| 42 | Asyncio AgentLoop 生命周期 | Module 5 |
| 43 | 沙箱与工具调用隔离 | Module 5 |
| 44 | 工具结果 Tokenization 与对齐 | Module 5 |
| 45 | Reward 与 Verifier | Module 5 |
| 46 | 多轮 Trajectory 和 Action Mask | Module 5 |
| 47 | Group 过滤与补投 | Module 5 |
| 48 | 长尾请求与样本偏差 | Module 5 |
| 49 | TransferQueue 背压与消费顺序 | Module 5 |
| 50 | 异步 DataSampler Mini Project | Module 5 |
| 51 | Sync、Colocate、Separate 架构 | Module 6 |
| 52 | 训推流水重叠 | Module 6 |
| 53 | Partial Rollout 暂停与恢复 | Module 6 |
| 54 | Behavior Policy 版本与 Staleness | Module 6 |
| 55 | Off-policy Correction 与截断 IS | Module 6 |
| 56 | 在途排空和取消语义 | Module 6 |
| 57 | 权重同步与原子切换 | Module 6 |
| 58 | 动态补投与接受率估算 | Module 6 |
| 59 | 异步持久化、重放与恢复 | Module 6 |
| 60 | 同步/异步的吞吐与质量对照 | Module 6 |
| 61 | Ray Actor 与 Object Store | Module 7 |
| 62 | K8s GPU Placement 与拓扑 | Module 7 |
| 63 | 异构资源调度 | Module 7 |
| 64 | Consistency、Idempotency、Exactly-once | Module 7 |
| 65 | Lease、Fencing 与故障检测 | Module 7 |
| 66 | Snapshot 与 State Machine 恢复 | Module 7 |
| 67 | RDMA、gRPC 与 Zero-copy | Module 7 |
| 68 | Trace、Metric、Log 关联 | Module 7 |
| 69 | Chaos Testing 与可靠性 | Module 7 |
| 70 | 千卡级 RL 系统设计 | Module 7 |
| 71 | 随机数与确定性复现 | Module 8 |
| 72 | Rollout/Train Logprob Drift | Module 8 |
| 73 | KL、Entropy、Clip 指标 | Module 8 |
| 74 | NaN 和混合精度排障 | Module 8 |
| 75 | OPD 与 GRPO 联合 | Module 8 |
| 76 | DPO 与偏好优化对比 | Module 8 |
| 77 | 动态 Packing 与负载 | Module 8 |
| 78 | 投机解码分布一致性 | Module 8 |
| 79 | GPU 全局资源成本优化 | Module 8 |
| 80 | 端到端一致性与故障注入 | Module 8 |
| 81 | Python Asyncio 背压编码 | Module 9 |
| 82 | Heap、Queue 调度算法 | Module 9 |
| 83 | C++、Go、Rust 系统编程基础 | Module 9 |
| 84 | 网络与并发编程题 | Module 9 |
| 85 | RL Infra 从零系统设计 | Module 9 |
| 86 | 多环境 Agent Harness | Module 9 |
| 87 | NCCL Hang 排障 | Module 9 |
| 88 | RL Loss 异常与质量回退 | Module 9 |
| 89 | OOM、长尾与 P99 排障 | Module 9 |
| 90 | 容量规划与 GPU 成本估算 | Module 9 |
| 91 | 项目 1/2 的量化表达 | Module 10 |
| 92 | 事故复盘与面试陈述 | Module 10 |
| 93 | 开源 PR 和设计文档审查 | Module 10 |
| 94 | 图与数组算法白板 | Module 10 |
| 95 | 并发调度现场编码 | Module 10 |
| 96 | RL 数学和算法模拟面试 | Module 10 |
| 97 | GPU 与分布式模拟面试 | Module 10 |
| 98 | 分布式系统设计模拟面试 | Module 10 |
| 99 | 项目深挖与行为面试 | Module 10 |
| 100 | 综合终面与能力差距复盘 | Module 10 |

每个模块的第 10 天为阶段验收；第五天插入复习，不替换当天主题。完整产出见 [里程碑](milestone_map.md) 和 [系统主线](system_roadmap.md)。
