# Day 03 · 用一个索引规则重建整个 Loss

输入 Token `[B,L]`，Causal LM 的 Logits `[B,L,Vocab]`。位置 $t-1$ 的 Logits 预测位置 $t$ 的目标 Token，所以：

~~~python
action_ids = input_ids[:, P:P + T]
action_logits = logits[:, P - 1:P + T - 1, :]
~~~

Prompt 长度 $P=2$、有效 Token 长度 4、完整序列长度 5 时，目标位置 1,2,3,4 的 Action Mask 是 **[0,1,1,0]**。

二 Token Response 的采样概率若为 0.8 和 0.7，则联合概率 0.56，联合 Logprob 约 **−0.579818**；终局 Reward=1、Token Logprob Sum 目标下，负号 Loss 为 **0.579818**。

两条 Response 各有 2、4 个有效 Token，如果每个位置 Loss=1：全局 Token Mean 与逐 Sequence Mean 的标量都可以是 1，但每个 Token 的梯度权重不同。

**重要的诊断：** Shift 错一格，Shape 常常仍合法；Mask 错把 Padding 当 Action，Loss 仍能下降。验证索引关系必须比“能 backward”更强。

[改动索引并观察预测者与目标的位置](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=3)。
