# Day 03 · Logprob 对齐错一位，为什么 Loss 仍然正常？

训练模型生成过的 Response，需要在 Trainer 上算这些**已生成 Token 在某个固定 Policy 下的条件概率**。这一般使用 Teacher Forcing：模型一次 Forward 得到所有位置的 Logits，动作序列已固定，并不再采样。

真正困难的地方有两个：**位置 $t$ 的 Logits 在预测哪个 Token**；当同一 Batch 的 Response 长度不同，Loss 的分母究竟按 Token、Sequence 还是 Group 计算。

[互动 Lab 03：亲手制造 Token Shift 错误 →](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=3)

## 一段完整序列，先标索引

假设某个模型的 Token 序列如下（仅为方便阅读的示意 Token，未声称对应任何实际 Tokenizer 的 ID）：

| input 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Token | BOS | 2 | + | 2 | = | 4 | PAD |
| 类别 | Prompt | Prompt | Prompt | Prompt | Prompt | **Action** | Padding |

Prompt 长度 $P=5$，Response 只有一个 Token「4」，有效长度 $V=6$（包含 Prompt，排除 Padding）。

因果语言模型 Forward 后：

~~~text
input_ids : [B, L]
logits    : [B, L, vocab_size]
~~~

**位置 $t$ 的 Logits 预测位置 $t+1$ 的 Token。** 因此「4」的位置是 5，但计算它条件概率的 Logits 在位置 **4**，不是位置 5。

对于等长未 Packing 的简单情况：

~~~python
action_ids    = input_ids[:, P:P + T]
action_logits = logits[:, P - 1:P + T - 1, :]
action_logprob = action_logits.log_softmax(-1).gather(
    dim=-1, index=action_ids.unsqueeze(-1)
).squeeze(-1)    # [B, T]
~~~

`log_softmax` 会对整个 Vocab 归一化，`gather` 则只取本次采样动作的条件 Logprob。**取到一个 Logit 并不等于取到 Token Logprob**；它还受词表中其他 Logit 的影响。

注意正确的预测与动作之间差一个位置。把 `logits[:, P:P+T]` 错当作预测 Response 的位置，Shape 依旧匹配，`backward()` 也不会报错，甚至 Loss 可能下降；你训练的是错位目标。

## 变长 Prompt / Padding 时，Mask 该标在哪里？

如果每条样本的 Prompt 长度 $P_i$ 和真实有效长度 $V_i$ 不同，统一把 $t=1,\ldots,L-1$ 看作**目标 Token 的输入位置**，那应只选择满足

$$
P_i\le t<V_i
$$

的目标位置参与 Response 的 Policy Loss。

例如 $L=7$，两条序列：

~~~text
target input position:       1  2  3  4  5  6
sample0: P=2, V=5           0  1  1  1  0  0
sample1: P=3, V=6           0  0  1  1  1  0
~~~

共有 6 个有效 Action Token。对应的选中 Logprob Tensor 和 Mask 可以统一用 `[B,L-1]`，列 $j$ 表示输入 Token 在位置 $j+1$ 的目标。

这样写出的逻辑能被仓库中的 [`response_token_log_probs`](../../examples/day_04/sequence_logprob_lab.py) 实验验证。该实验会故意将 Logits 错位一格，对比选中的概率是否相同。

上述 Mask 只覆盖 Prompt/Response/Padding 的位置关系。到了 Agentic RL，多轮 Message、工具 Observation、特殊 Token 和多模态位置都可能需要额外 Mask，不能只靠 $P_i$ 与 $V_i$ 判定它们都属于可训练 Action。

## 一样的 Loss 数值，也可能是三个不同的目标

设两条 Response 分别有 1 和 3 个有效 Token，四个位置的负 Logprob 数值都为 1，Advantage 都为 +1。比较：

**全局 Token Mean：**

$$
L_{\text{token}}=\frac{\sum_{i,t}m_{it}\ell_{it}}{\sum_{i,t}m_{it}}
=\frac{1+3}{4}=1.
$$

每个有效 Token 的梯度权重均为 $1/4$。

**先求每条 Sequence 的 Token Sum，再对 Sequence 求 Mean：**

$$
L_{\text{seq-sum}}=\frac{1}{2}\sum_i\sum_tm_{it}\ell_{it}
=\frac{1+3}{2}=2.
$$

每个 Token 的梯度权重为 $1/2$，长序列累计权重更大。

**先求每条 Sequence 的 Token Mean，再对 Sequence 求 Mean：**

$$
L_{\text{seq-mean}}=\frac{1}{2}\sum_i
\frac{\sum_tm_{it}\ell_{it}}{\sum_tm_{it}}
=\frac{1+1}{2}=1.
$$

短序列唯一 Token 的梯度权重为 $1/2$，长序列每个 Token 为 $1/6$。**第一种和第三种 Loss 数值恰好相同，梯度权重却不同。**

这在 PPO/GRPO/DAPO 中非常重要。固定每条回答的奖励、长度不同，如果实现改变了 Reduction，就可能同时改变有效训练分布和长短回答的偏好。不能把 `loss.mean()` 当成没有算法含义的工具函数。

下节课的 PPO Ratio 也会受到这个选择影响：即使每个 Token 的 Clipped Surrogate 都正确，最终 Reduce 成 Loss 的方式不同，仍然是在优化不同的目标。

## Rollout Logprob 为什么不能直接拿来 Backward？

采样时的服务侧可能记录 Logprob，但这些数字通常是**与当前 Learner 的 Autograd 图分离**的结果。Trainer 若希望更新当前权重，就必须对固定 Token 使用**当前参数**重算可微 Logprob。

仍要区分四个角色：

- Behavior：实际生成动作的策略分布，受采样参数影响。
- Old：PPO 当前优化阶段固定的 Ratio 分母。
- Current：正在优化的 Learner Policy。
- Reference：特定 KL 项对照的分布。

生成时用了 Temperature 或 Top-p，训练却对未变换的 Softmax 计算 Logprob，那么两者属于不同的概率定义。出现 Drift 时先对齐 Token、采样变换和版本，再讨论 BF16/FP8 的浮点误差。

## 两个小实验：预测错误会留下什么痕迹

打开 [Lab 03](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=3)，改变 Prompt 长度并切换「故意错位」。注意 Shape 并不会因为切换而改变，但预测与目标位置关系会错。

然后从仓库运行旧路径下的 CPU 实验（脚本名暂未随新课表重命名）：

~~~bash
python examples/day_04/sequence_logprob_lab.py
python -m unittest discover -s examples/day_04 -p 'test_sequence_logprob_lab.py' -v
~~~

测试不仅对比正确 Shift 与错误 Shift，还会验证 Prompt/Padding 上 Loss 梯度为零，以及三种 Reduction 的梯度权重比例不同。这个实验在 CPU 上使用小 Tensor，不是大型 Trainer 的吞吐测试。

## 查证资料

- [PyTorch `log_softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) 与 [`gather`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)，可对照数值路径。
- [PPO (2017)](https://arxiv.org/abs/1707.06347)：固定 Old Policy 的重要性比例在 Day 04 展开。
- [DAPO (2025)](https://arxiv.org/abs/2503.14476)：Token-level PG Loss 的优化主张；Day 06 将与当前的 Reduction 差异联系起来。
