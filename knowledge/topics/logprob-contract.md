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

## Speculative Decoding：接受率与 KL 不是同一个量

固定一个历史前缀，令 target 分布 p=[0.5,0.5]，draft 分布 q=[0.99,0.01]。标准随机 speculative sampling 从 q 提议 token x，以 min(1,p(x)/q(x)) 接受；拒绝后从归一化的正残差 max(p-q,0) 采样。单位置期望接受率为 sum(min(p,q))=0.51，而不是 logits 的比值。这里假设两侧概率已经按实际采样规则归一化。

另取 p=[0.99,0.01]、q=[1-1e-12,1e-12]。接受率约 0.99，但 KL(p||q) 约 0.22 nats；继续缩小 q 的第二项可以让 KL 任意增大，而接受率几乎不变。接受率由重叠概率质量决定，KL 对极小概率更敏感。该反例是理论推导，不是 DFlash 或 MTP 的实测。多 token 接受长度还依赖前缀条件分布与验证算法；不能把单位置公式直接当成 block 接受率。

在满足精确拒绝采样条件时，最终输出服从 target 分布，draft 只是提议分布。RL 的 behavior logprob 应对应实际输出分布及其采样变换，而不是 draft logprob。贪心验证、近似接受或特殊采样实现需要单独核查。参考 [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)。具体 DFlash/MTP 的训练目标与服务端验证路径尚需固定源码版本后补充。
