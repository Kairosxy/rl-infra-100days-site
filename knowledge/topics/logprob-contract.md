# Logprob 是哪个动作、哪个分布的概率？

有一类训练错误不产生 NaN：Token 和 Reward 对齐错了，或者比较的压根不是同一份概率分布。Loss 可能仍在下降。

在 Causal LM 中，输入 `input_ids [B,L]` 得到 `logits [B,L,V]`；位置 $t$ 的 Logits 预测位置 $t+1$ 的目标 Token。Prompt 长度为 $P$、Response 长度为 $T$ 时：

~~~python
action_ids = input_ids[:, P:P+T]
action_logits = logits[:, P-1:P+T-1, :]
action_log_probs = action_logits.log_softmax(-1).gather(
    -1, action_ids.unsqueeze(-1)
).squeeze(-1)
~~~

$B=2,L=7$，Prompt Length 为 $[2,3]$、Valid Length 为 $[5,6]$，那么在目标位置 $1..6$ 上，Action Mask 分别是 $[0,1,1,1,0,0]$ 和 $[0,0,1,1,1,0]$，共有 6 个可训练位置。这里默认单段 Prompt+Response；工具 Observation 和多模态 Token 需要额外标记。

## 四种容易混淆的概率

**Behavior** 表示实际把 Token 采出来的分布，可能经过 Temperature、Top-p、Stop 等采样变换；**Old** 是优化阶段固定的比较基准；**Current** 是正在反向传播的模型；**Reference** 是 KL 等约束的对照模型。它们可能某次恰好相同，但不能因为字段都叫 `log_probs` 就直接互换。

尤其在 FP8/BF16 交叉运行时，先检查 Token ID、Mask、采样变换和权重版本，再比较相同条件下的 Logprob。否则看到“0.3 nats Gap”也无法判断是数值漂移、路由分歧还是比较对象不一致。

## 快速反例

Prompt 两个 Token，Response 两个 Token。正确应使用 `logits[1], logits[2]` 预测 `token_ids[2], token_ids[3]`。故意使用 `logits[2], logits[3]`，Shape 仍然匹配，Loss 也能 Backward，概率语义已经改变。

可运行的 CPU 例子仍在 [sequence_logprob_lab.py](https://github.com/Kairosxy/rl-infra-100days-site/blob/main/examples/day_04/sequence_logprob_lab.py)；该路径沿用了旧课表命名，文件内容可作为本主题的实验依据。