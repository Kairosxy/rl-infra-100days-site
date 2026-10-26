# Day 03 · 复算卡

Teacher Forcing 固定已经生成的动作，让 Current Policy 重新计算**可微条件 Logprob**。位置 $t$ 的 Logits 预测位置 $t+1$ 的 Token。

对于 Prompt 长度 $P$、Response 长度 $T$：

~~~python
action_ids = token_ids[:, P:P + T]
action_logits = logits[:, P - 1:P + T - 1, :]
~~~

变长序列里，应把目标位置 $t$ 的有效 Action Mask 定义为 $P_i\le t<V_i$（本课简单 Prompt+Response+Padding 情形），而非全部非零 Token 都训练。

两条序列的有效 Token 数分别 1 和 3：

- Token Mean：每个 Token 权重 $1/4$。
- Sequence Mean of Token Mean：短样本每 Token 权重 $1/2$，长样本每 Token 权重 $1/6$。

两种 Loss 标量有时相同，梯度仍不同。**能训练不等于对齐正确；Shape 正常不等于 Token 没错位。**

[切换正确/错误 Shift，并查看 Token 对齐](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=3)。
