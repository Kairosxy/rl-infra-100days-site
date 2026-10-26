# 研究与实验里程碑（自选，不用于解锁课程）

课表中的每一天是学习内容的日期，下面的节点则是**阶段性证据检查**。不要求用户当日提交或通过，不以未作答为由阻止下一课发布。不要用「完成了篇文章」来替代能复算、能运行、能解释反例的证据。

| Checkpoint | 需要真正弄清的机制 | 可以公开、可复核的证据 |
|---|---|---|
| Day 05 | 从期望 Reward 到 PG、PPO Clip 和 GRPO Group Advantage | Bernoulli 梯度、Baseline 方差、正负 Clip、全等 Reward 反例 |
| Day 10 | 已采样 Token 怎样正确更新 Toy Policy | CPU Step、Shift/Mask、Ratio 与参数更新断言 |
| Day 14 | DAPO / GSPO / DeepSeekMath 的机制差异是否真的读懂 | 一页「改了哪层目标、数据分布和验证」的对照 |
| Day 26 | Learner 如何保持数学目标与执行等价 | 微批分母、梯度累积、AMP、数值偏差和恢复测试 |
| Day 42 | 并行方案的主要成本在什么地方 | 8–64 GPU 配置下的显存、计算和通信字节账 |
| Day 56 | Serving 的吞吐与延迟怎样公平测量 | 真实配置下的 TTFT/TPOT/P99、KV 预算和长度分布 |
| Day 68 | 多轮 Agent Trajectory 的学习信号如何形成 | Action/Observation Mask、Verifier、数据队列的负例测试 |
| Day 80 | 异步真的更有利吗，是否改变了训练数据分布 | Accepted/Inflight/Drained 时间线、Staleness 与质量对照 |
| Day 85 | OPD 与 GRPO 梯度如何共存 | KL 方向、Teacher/Student Logprob、Mask 和 Scale 对照 |
| Day 94 | 故障与容量模型是否可解释 | 一份完整的超时/OOM/NCCL Hang 假设与验证记录 |
| Day 100 | 能否独立审查与设计 RL Infra | 算法推导、可执行片段、系统决策和反例 |

所有阶段检查都要区分：**数学推导、代码已测试、GPU 上已测量、论文/作者报告**。没实际跑过就不能写「验证了性能提升」。

前十天可使用 [互动实验室](https://kairosxy.github.io/rl-infra-100days-site/labs/) 的前五个机制 Lab 来检查直觉，但其数值来自 Toy 模型，不是 LLM 训练结果。
