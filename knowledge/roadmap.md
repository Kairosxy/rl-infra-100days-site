# 知识库持续建设 · 编辑安排

每天北京时间 09:00 发布一次有实质价值的新增内容或修订。**不是每次生产四份讲义，也不是必须新建 Topic**；如果最新资料只需要校正已有案例的一个关键假设，就应该更新已有文档。日期只决定发布节奏，目录由知识依赖决定。

## 先审查技术家族覆盖

[RL Infra 技术谱系与演进](topics/technology-lineage.md) 是跨章节的覆盖基线。每次更新先定位新技术的上位问题与家族，检查相邻主流分支、演进动机、先修与实现边界。家族概览和必要基础缺失时优先补齐；已存在正文则按实际缺口更新。MTP/DFlash 只是投机解码分支，Score Centering 只是训推一致性下的策略纠偏分支；同样规则适用于所有家族，而不是只扩这两个例子。

当前同时推进数学/GPU/训练基础、现代 RL、模型与多模态、生成模型、VLA/World Model/RSI、Serving 与 RL 系统家族。未完成的计划节点不得伪装成已发布正文，成熟度和覆盖度独立记录。新增方法需要原始证据、机制对比、版本入口和验收边界。

## 其他日程的技术增量

每次维护前检查「前沿调研｜RL 社区速览」「前沿调研｜异步 RL 架构周报」「RL Infra 手册｜每日工程题」可访问的最近成功结果及历史增量。仅提取具有长期价值的公开技术机制和知识缺口，回到原始公开论文/官方源码验证；按既有家族补分支、按新问题补独立家族。保留来源与覆盖日期，无法访问或截断明确说明。三个日程保持原有角色和交付位置，不搬运报告、题目正文或私人信息，不把假设、内部设计、未复现性能视为公开事实。日程结果是增量线索而非完整覆盖范围；仍主动补齐数学、模型、生成、Omni/VLA、现代 RL 等分支。

## 先补齐最有工程价值的空白

以下保留现有工程专项 backlog，不再作为全领域唯一优先级。

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


## 发布前必须通过的检查

目录以 `knowledge/catalog.json` 的领域/专题为准；阅读路径、前置知识、题目多向关联和文内实验位置在 `knowledge/learning-map.json` 维护。不从前端临时拼凑分类。Q001–Q224 原始题库、问法与来源保留；关联修订不重写原答案。

每日 09:00 对照技术家族覆盖缺口和现有工程专项 backlog，选择有证据的增量；不要围绕单个热点连续扩写。优先校正旧题公式缺失和未绑定版本的框架结论，不只增加文章数。当前未将所有题目视为已逐题校勘，不将教学模型视为 GPU Benchmark。

提交前必须执行构建、Python 测试、Node 机制测试、发布 HTML 的脚本语法检查和浏览器回归。无法执行时留在分支或报告阻塞，不得声称验收通过。Actions 的 build、deploy 和线上 `build-info.json` SHA 必须分别确认。不能为了变绿而删除、跳过实质测试。

视觉继续使用 Swiss Minimalism + Editorial Design：米白、墨绿、深灰；一个主导航，列表目录，文内交互。不新建与文档平级的仪表盘或题库入口。
