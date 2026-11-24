# Agent Context：历史、压缩与 KV 生命周期

多轮任务中，Harness 保存完整交互历史，每次模型调用却可以只读取其中一部分。历史记录、实际输入 Context、KV Cache 分别承担可追溯、条件计算和加速职责，三者的生命周期不必一致。

## 三份状态

| 对象 | 保存什么 | 生命周期 |
|---|---|---|
| Trajectory | 模型动作、工具响应、事件时间、版本、失败原因 | 训练、回放与审计所需期限 |
| Context | 本次模型实际可见的 Token 或等价表示 | 从构造请求到本次生成结束 |
| KV Cache | 给定权重和前缀下已计算的内部状态 | 由引擎的缓存、引用与回收策略决定 |

删掉 Context 中的工具输出会改变策略条件分布；只回收物理 KV，下一次按相同输入重算，则在执行语义一致时不改变该条件分布。摘要属于前一种操作，缓存 Eviction、Offload 或重新计算属于后一种。注意力窗口本身也由模型定义，不能将任意语义压缩视为等价的滑窗计算。

## 缩短 Context 与前缀复用

假设原上下文 24k Token，下一轮追加 2k。若旧前缀完整命中，新增 Prefill 只需处理约 2k 个位置。现在改用 8k Token 的摘要上下文再追加 2k；如果开头被改写且无可复用前缀，则新增 Prefill 约为 10k 个位置。有效输入长度减少了，未命中位置反而更多。

该算例仅比较新增位置数；真实时间还与 Key 长度、Attention 结构、Batch、缓存块边界和 GPU 并发有关。长历史反复保留会增加 KV 容量与 Decode 读取成本，压缩则有额外推理开销与信息损失。Contexture、MORTIS、Penelope 等方法可分别沿着语义选择与缓存成本深入；不能把不同实现的删除策略都归为同一种 KV Eviction。

## RL 轨迹保真

建议为每轮模型调用保存实际 context token IDs（或可重建的不可变输入）、所引用的历史范围、模型版本、采样配置及 action span。摘要若也是模型生成的动作，是否参与策略更新应由目标明确规定。若只保存完整 Transcript，事后将它整段拼回 Trainer，可能让原来不可见的 Observation 进入条件上下文，导致 logprob 重算错误。

对同一批长程任务比较完整历史、固定窗口及摘要方案，记录实际输入长度、未命中 Prefill Token、缓存驻留字节、工具任务成功率、复算概率差异。Cache 命中指标要区分请求命中与 Token 命中。另见 [KV Cache](kv-scheduling.md)、[Harness](../agent/agent-harness-environment.md)。

参考：[vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)、[Contexture](https://arxiv.org/abs/2611.26980)。后一项为研究方法入口，具体实现及效果尚需逐版本验证。
