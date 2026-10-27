# 知识库持续建设 · 编辑安排

每天北京时间 09:00 发布一次有实质价值的新增内容或修订。**不是每次生产四份讲义，也不是必须新建 Topic**；如果最新资料只需要校正已有案例的一个关键假设，就应该更新已有文档。日期只决定发布节奏，目录由知识依赖决定。

## 先补齐最有工程价值的空白

| 顺序 | 内容专题 | 最终应回答的具体问题 | 推荐关联 |
|---|---|---|---|
| 1 | FP8 Scale 与 EOS/Logprob | 模型变短是否真由量化误差导致 | FP8、Logprob、SGLang、TransformerEngine |
| 2 | R3 / Router Replay | Expert Assignment 对齐后仍有哪些 Drift | MoE、R3、NeMo RL、Megatron |
| 3 | PPO/GRPO 的 Reduction | 同一批 Response 为什么换 Reduction 就改变更新 | Policy、Mask、DAPO/GSPO |
| 4 | GAE 的 Terminal 与 Truncation | 被时间限制截断时为何仍需要 Bootstrap | Policy Gradient、Value、Learner |
| 5 | FSDP2 / Megatron 并行选择 | 给定 8/64 GPU 怎样确定资源与通信上限 | Parallelism、Throughput、FSDP2 |
| 6 | RL Step 长尾 | Rollout、Train、Verifier 哪一段决定关键路径 | Async、SGLang、Profiler |
| 7 | Partial Rollout 与版本 | 跨 Weight Update 的轨迹还能如何计算 Ratio | Async、Agent、Staleness |
| 8 | DataSampler 接受率与 Drain | 补投是否用有效 Token 换来更多无效 GPU-time | Sampler、Group、TQ |
| 9 | KV / PD / HiCache | Prefix Cache 与 KV Offload 的收益边界 | KV、SGLang、vLLM |
| 10 | OPD × GRPO | Teacher KL 和 Reward PG 如何同时优化 | Teacher、Student、NeMo RL |
| 11 | MoE EP 负载偏斜 | All-to-All 耗时为何高度取决于 Routing | Parallel、Megatron、Profiler |
| 12 | NCCL Hang 故障定位 | 是 Collective 顺序、Watchdog 还是资源问题 | DDP、NCCL、Fault |
| 13 | Agent Harness 轨迹 | 哪些 Tool Observation 必须从 Action Loss 排除 | Agent、Verifier、Harness |
| 14 | 训推概率契约 | Top-p/Temperature 与 Old/Behavior 概率如何对齐 | Logprob、Serving、Offpolicy |
| 15 | 对照框架架构 | Verl / slime / AReaL 何处拥有最终数据状态 | Async、框架源码 |

以上是主题优先级，不是“第 N 天一定发布”的限制。每次查最新仓库进度、已发布内容与可信新论文后选一个有信息价值的增量；不为凑数量拆出重复文章。已有案例和框架实现要优先补上可定位的源码版本、可运行的负例与适用边界。

## 质量门槛

一个主题能从基础概念走到合理反例；工程问题有可执行诊断而非配置清单；框架分析讲清实现边界，来源具体、版本明确。对照实验须给基线、变量、测量口径和结果；没有设备或数据时标注为理论推导/待验证方案，不能编造收益。用户在本地维护网站及 UI，自动发布任务原则上只修改 `knowledge/topics`、`knowledge/cases`、`knowledge/implementations` 和必要的索引引用；不改构建/题库/旧课程。
