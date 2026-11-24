# 环境池、并行采集与 Episode 边界

扩大 Agent Rollout 并发时，模型请求数、环境实例数和活跃会话数是三个不同预算。一个环境实例可能等待工具结果，一个模型请求可能只负责某个 turn；会话身份与状态必须在两者之间保持一致。

## 环境状态的所有权

训练服务可以将模型生成、环境执行和轨迹存储分开。Harness 持有 session/episode 映射，环境 Worker 持有真实可变状态，Learner 消费已提交的动作与反馈。请求重试时需要区分 sample_id 与 attempt_id：如果上一个 attempt 已改变环境，不能在同一个状态上盲目重复执行。

向量环境可让多个子环境并行 step；某个子环境结束后可单独 reset，其他环境继续。Gymnasium 文档区分 next-step、same-step 和 disabled autoreset 模式。若 same-step 返回的 observation 已是新 episode 初始状态，则训练应从接口约定中取得旧 episode 的最终 observation，避免跨 episode Bootstrap 或递推 Advantage。

## 异步采集的容量

假设 4 个环境的单次 episode 耗时分别为 2、2、2、10 秒。整批等待会以 10 秒为一轮，完成 4 条；若环境可独立重置并持续生产，在固定耗时、无额外瓶颈的长期模型下，理论完成率为 1/2+1/2+1/2+1/10=1.6 条/s。实际吞吐还受模型并发、工具连接数、评分和 Learner 背压约束。

快速环境被采集得更多，也可能改变任务分布。如果要求每种任务等权，应按任务而非仅按返回顺序控制供给和消费。队列预算和 episode identity 需要一起设计。

## 分支、重放与多 Agent

保存同一状态的多个候选分支时，环境要提供可重放或 Snapshot/Fork 能力；否则只适合对无副作用的输入做分支。共享文件、外部网页会话和随机源都可能让两个分支互相影响。Multi-Agent 场景中还需保留角色、局部观测、动作及对手策略版本；其他 Agent 的更新会改变当前 Agent 面对的环境分布。

在假环境里分别注入慢 episode、工具超时及 reset，记录 episode_id、turn_index、前后 observation、终止类型和提交状态。确认快速环境可以补投、慢环境没有被错误丢弃，且结果回填不会跨越 episode 或重复副作用。随后再增加真实环境池规模。

参考：[Gymnasium Vector Autoreset](https://gymnasium.farama.org/main/api/vector/autoreset/)、[Ray Actors](https://docs.ray.io/en/latest/ray-core/actors.html)、[Ray 与队列背压](../async/ray-data-runtime.md)。
