# Test-time Scaling：候选、搜索与验证预算

一个模型单次答对的概率为 0.3。如果八次采样独立，至少出现一条正确答案的概率是 $1-(1-0.3)^8\approx0.9424$。如果选择器无法辨认正确答案，最终输出的正确率仍可能只有 0.3。推理时增加计算，须同时解决产生候选和选出候选的问题。

## 预算用在哪里

| 方法 | 计算分配 | 选择依据 |
|---|---|---|
| Best-of-N | 并行或串行生成 N 条完整回答 | Verifier 或排序模型 |
| Beam Search | 保留多个部分序列，逐步扩展 | 累积概率或指定打分 |
| Tree Search / MCTS | 对部分状态选择分支、扩展和回传估计 | Value、过程评分或模拟反馈 |
| Iterative Revision | 让模型根据已有答案或反馈继续修改 | 任务反馈、内部评估 |
| Adaptive Compute | 按难度和不确定性分配候选数、深度或预算 | 预算约束下的预期质量收益 |

Beam Search 的序列概率与任务正确率不是同一个分数。MCTS 还需要状态、动作、访问次数及可定义的 Value 或模拟结果；文本前缀、工具状态和真实环境可能有不同的分支成本。

## Pass@k 与实际选优

若每个候选独立成功概率为 p，理想 Oracle 选择的上限为 $1-(1-p)^N$。实际候选往往共享同一种错误模式，独立假设会高估收益。若从 n 条候选中观察到 c 条成功，在均匀无放回抽取 k 条的意义下，常见 Pass@k 估计是

$$
1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

此处要求 $1\le k\le n$，组合数在可选元素不足时取零。它估计候选集合覆盖正确答案的能力。报告 Best-of-N 还需测量 Verifier 的实际选中率，而不是以 Oracle 代替选择器。

## 系统成本

假设 N=8，每条候选生成 500 Token，独立评分再消耗 200 Token 的等价工作量，候选加验证的教学预算约为 5600 Token。与单候选 500 Token 比，计算预算增大，但共享 Prefix、并行执行与评分模型不同使延迟不一定等比例变化。Tree Search 可能复用部分前缀的 KV，也可能增加分支状态、评分请求和调度成本。

与 RL 训练的自适应采样不同，Test-time Scaling 的直接产物是本次任务的输出；不必更新模型。精确投机解码主要在保持目标分布的前提下减少串行 Target Forward，可用来加速其中的生成部分，但并不自行增加正确候选的覆盖率。继续阅读 [自适应采样与树形探索](../rl/adaptive-rollout-search.md)。

参考：[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)、[HumanEval 与 Pass@k](https://arxiv.org/abs/2107.03374)、[投机解码](speculative-decoding.md)。
