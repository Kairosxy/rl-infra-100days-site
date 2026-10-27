# 开启 FP8 后，生成长度与 Logprob 都变了

观察到 FP8 Rollout 生成的 Response 比 BF16 短，同时 Learner 重算的 Logprob 与 Rollout 记录不一致。它可能来自量化误差，也可能是**两个引擎比较的根本不是同一个随机变量**。

一个建议的排障顺序是从最便宜的反证开始。

## 先把比较对象固定住

使用同一份权重版本、同一组 Token ID、同一段上下文、相同的 Stop/Temperature/Top-p 定义。然后分开测：

1. Teacher-forced 固定 Token 时 BF16/FP8 的 Logits、Logprob 差异：验证数值路径。
2. 使用完全相同的 Sampling 参数时，EOS/Stop 概率与 Response Length 分布：验证采样行为。
3. 如果是 MoE，检查两边各层 Top-k Expert ID 的一致率：验证离散路由。

第 1 步可能已经发现差异，即使没有实际重新生成任何 Response；第 2 步含有采样随机性，必须用足够样本和重复 Seed/重复实验描述分布，而不是比较一对字符串就下结论。

## 三种常见原因不能混为一谈

- **概率定义**：服务侧记录的是经过 Temperature/Top-p 后的 Logprob，训练侧计算原始 Softmax，或者 Prompt/Response Shift 不一致。
- **低精度误差**：不同 FP8 Recipe、Scale 更新策略、LayerNorm/Attention 精度可能造成同 Token 的概率变化。
- **路由分歧**：同一个 Token 的 Expert Assignment 不同，导致 MoE 计算路径变化；R3 的目标是对齐这条离散选择。

## 解决方案应该匹配原因

如果是 Token Shift，修 Mask/索引；如果是概率变换，先明确 Behavior Distribution；如果是 FP8 精度差异，则比较 Scale、Recipe 和关键层 BF16 回退；如果是 Router Disagreement，再考虑 R3 或基于路由差异的处理。

R3 不是消除所有 Logprob Gap 的按钮。FP8 也不一定提高完整 RL Step 吞吐：在 Rollout 占绝大多数耗时的场景，Learner GEMM 加速可能只改善局部指标。

## 最小实验产物

一份表记录 `token_id`、`position`、`policy_version`、`rollout_logprob`、`learner_bf16_logprob`、`learner_fp8_logprob`、`expert_ids`，并将 Mean/P95/P99 Gap 与 EOS 分布单独统计。没有这个分层证据，不应该直接推断哪种方案有效。

资料：[TransformerEngine FP8 Primer](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) · [R3 论文](https://proceedings.mlr.press/v306/ma26aw.html)。