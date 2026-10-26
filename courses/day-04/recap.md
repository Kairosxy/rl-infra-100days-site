# Day 04 复习卡 · Policy Gradient 与 Logprob

- 期望 Reward 的梯度可以写成 E[R ∇θ log pθ(τ)]；Reward 本身不必对参数可微。
- 自回归序列的 Logprob 是各 Token 条件 Logprob 的和。
- Teacher Forcing 固定已生成 Token，重新评估它们在当前策略下的概率；它不重新采样回答。
- Rollout、Old、Current、Reference Logprob 具有不同语义。
- Token Mean、Sequence Mean 和 Sequence Sum 对不同长度样本的权重不同。

自测：从期望回报推导 Log-Derivative Trick；解释 logits[:, P-1:P+T-1] 为什么预测 input_ids[:, P:P+T]。
