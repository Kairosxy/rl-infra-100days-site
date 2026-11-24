# Agent 的一条轨迹，哪些是模型行为？

一次工具调用至少包括：模型生成 Tool Call、工具执行、环境返回 Observation、模型继续决定动作。将整段 Transcript 全部当作 Response Token 训练，会让模型在并未采样的环境文字上产生 Policy Gradient。

必须区分：**Action** 由模型的策略分布采出，**Observation** 来自环境，**Reward** 评价行为或轨迹，**State** 是环境的可恢复状态。Message Role 和 Token Mask 要能把这几个对象映射回原始序列。

例如模型输出一个 JSON Tool Call，工具返回 4K Token 的搜索结果，然后模型输出 20 Token 的结论。若直接对全部 4020 个 Token 求均值，环境 Observation 可能远远超过真正 Action 的权重。即使 Loss 没有 NaN，这个优化目标也是错的。

一个有效的最小 Agent Loop 要能验证工具结果与正确 Trial/Trajectory 的 ID 关联、异常是否影响 Reward、工具重试是否幂等、环境是否在不同采样之间污染。不要因为 Harness 能执行工具就默认满足 RL 所需的数据契约。

当 Agent Harness 与 RL Trainer 分离时，训练侧真正依赖的是可追溯的 Action / Observation / Reward 序列，而不是某个 ReAct 框架内部类名。

参考：[Agent Lightning](https://github.com/microsoft/agent-lightning)；[verl Agentic RL 文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)。