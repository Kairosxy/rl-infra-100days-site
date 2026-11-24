# Reward、Verifier 与 Teacher 的服务化

Reward 服务给出训练反馈，Teacher 服务给出蒸馏所需的监督信息。二者都可能由大模型实现，但负载、输出和消费契约不同。Score Centering 中的 Score 则是策略梯度中的 Logprob 导数，与评分服务无关。

## 服务输入与输出

| 服务 | 输入 | 输出 |
|---|---|---|
| Rule / Environment Verifier | 任务、答案或执行结果 | 可验证分数、状态 |
| Reward Model / Judge | Prompt、完整响应或步骤 | 标量、分类、过程反馈 |
| Teacher | Student 实际条件上下文 | 对应动作概率、Top-k 或全分布 |

很多评分模型只进行 Forward 并产出一个标量；生成式 Judge 还需要 Decode。Teacher 的计算和输出形状取决于蒸馏目标。不能用一种请求长度与 QPS 假定所有三种服务都有同样的吞吐。

假设每 Step 1024 条评分请求，每条 20k 输入 Token，则输入总量约 2048 万 Token。若服务在该实际负载下测得聚合 Prefill 吞吐 R token/s，仅输入处理的理想时间至少近似 20480000/R；还需考虑排队、重复输入命中、批处理效率、Decode 和请求失败。这是预算公式，不是实际硬件的性能预测。

## 反馈载荷与回填

全词表 Teacher Logprob 的存储量随 B×T×V 增长；若 T=32768、K=128，每位置保存 K 个 int32 ID 和 FP32 Logprob，单条原始 Top-k 数据是 32768×128×8=32 MiB。相比只返回一个标量，这会显著改变数据面成本。是否允许只记录采样动作或 Top-k，取决于算法估计器，不能为节省传输任意丢弃必要概率。

同一轨迹的评分结果应带 sample_id、attempt_id、reward/teacher version 和输入身份。超时、取消、真正失败、成功但晚到需要分开。服务端可能已经完成一次调用；重试结果需按既定语义去重或确定哪次有效，不应覆盖不匹配版本。Reward 尚未就绪的轨迹不能按默认 0 直接消费，除非目标明确规定这种失败就是任务奖励。

## 容量与缓存

通过实际输入/输出 Token 分布与成功率选择并发预算、batch/token limit、超时和队列上限。评分模型的前缀缓存需要相同的可复用前缀以及服务端缓存策略；重试不保证必定命中，也不等于前一次请求没有占用资源。监控请求完成速率、排队时长、Cache 命中口径、超时重试、版本一致性与待评分字节量。

参考：[InstructGPT](https://arxiv.org/abs/2203.02155)、[HybridFlow](https://arxiv.org/abs/2409.19256)、[OPD 与 KL](../rl/opd-kl.md)、[Ray 与队列背压](ray-data-runtime.md)。
