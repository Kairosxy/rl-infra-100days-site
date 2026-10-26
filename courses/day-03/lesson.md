# Day 03 · 一条回答只拿到一个分数，哪些 Token 在更新？

你让一个语言模型回答「2+2=?」。它的 Response 有两个 Token：第一个相当于「答案」，第二个相当于「4」。Verifier 只返回一个最终 Reward=1。

现在 Trainer 需要把这个 1 分变成梯度。**它到底应该鼓励哪个 Token？** 假如 Response 里还夹着 Padding，Padding 会不会也被“奖励”？如果同一个 Batch 里有人生成了 2 个 Token、有人生成了 200 个，究竟是谁对更新贡献更大？

这些不是格式问题，而是在决定我们优化什么函数。

## 先从 Response 的概率，而不是某个 Token 的分数出发

自回归生成时，一个完整 Response 的条件概率是：

$$
\pi_\theta(y_1,y_2\mid x)
=\pi_\theta(y_1\mid x)\,
\pi_\theta(y_2\mid x,y_1).
$$

取对数，就得到方便反向传播的和：

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^T \log\pi_\theta(y_t\mid x,y_{<t}).
$$

假设一条已经采样好的二 Token Response，它们在当前 Policy 下的条件概率分别为 0.8 和 0.7，则这条**固定 Response** 的联合概率为：

$$
\pi_\theta(y\mid x)=0.8\times0.7=0.56.
$$

联合对数概率为：

$$
\log\pi_\theta(y\mid x)=\log0.8+\log0.7
\approx -0.223144-0.356675=-0.579818.
$$

在最简单的终局 Reward REINFORCE 里，整条轨迹得到 $R=1$，负号 Loss 就是：

$$
L=-R\sum_{t=1}^2\log\pi_\theta(y_t\mid x,y_{<t})
\approx\boxed{0.579818}.
$$

评分器并没有给每个 Token 发不同的分数。当前定义把同一条轨迹的最终 Reward 作为每个已采样动作的**共同系数**，通过每个动作自己的 Logprob 提供梯度。这是明确的目标函数选择，并非“真实世界证明两个 Token 贡献相同”。

如果最终 Reward=0，这条轨迹的基础无 Baseline 梯度可能为零；如果使用 Advantage 或逐时间步 Return，系数会变化。但无论选哪种估计器，首先要保证选出来的是**真正被模型采样的动作**。

## 位置错一格，为什么 Shape 检查发现不了？

我们把 Token 按绝对索引排开。假设完整输入为：

| 绝对位置 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Token | BOS | 问题 | y₁ | y₂ | PAD |
| 属于 | Prompt | Prompt | Response | Response | Padding |

Prompt 长度 $P=2$，有效长度 $V_{\mathrm{len}}=4$（不含 Padding）。

一个 Causal LM 输入全部 Token 后返回：

~~~text
input_ids : [B, L]
logits    : [B, L, Vocab]
~~~

位置 1 的 Logits 预测位置 2 的 `y₁`；位置 2 的 Logits 预测位置 3 的 `y₂`。**位置 t 的 Logits 不是在预测 input_ids[t]，而是在预测 input_ids[t+1]。**

所以选择 Action Logprob 的切片必须匹配：

~~~python
action_ids = input_ids[:, P:P + T]          # y1, y2
action_logits = logits[:, P - 1:P + T - 1]  # 预测 y1、y2 的两个位置
log_probs = action_logits.log_softmax(-1).gather(
    -1, action_ids.unsqueeze(-1)
).squeeze(-1)                               # [B, T]
~~~

如果错把 `logits[:, P:P+T]` 拿来计算同两个 Token 的概率，代码一样能运行，Tensor 维度也可能完全符合预期。但第一个 Response Token 被拿自己的位置预测，而不是由前面的 Prompt 预测。对于 Causal LM，这已经不是定义中的那个条件概率。

这里不能只在讲义里说“要 Shift 一格”，还应该真正做负例测试。仓库中的 [Token Logprob CPU 实验](../../examples/day_04/sequence_logprob_lab.py) 会计算正确和错误索引下的结果；其目录名保留了初版命名，不影响这个实验的输入与公式。

## Padding 对计算图有什么影响？

对于刚才长度为 5 的序列，我们在输入 Token 的目标位置 `1,2,3,4` 上定义对应的 Action Mask：

~~~text
target position:         1    2    3    4
is response action:      0    1    1    0
~~~

第一个 0 对应 Prompt 的最后一个 Token，第二、三个位置对应两次真正采样的动作，最后一个 0 对应 Padding。

更一般地，给第 $i$ 条样本的 Prompt 长度 $P_i$、有效长度 $V_i$，本例的 Action Mask 可以写成：

$$
m_{i,t}=\mathbf 1[P_i\le t<V_i].
$$

注意这是**目标 Token 的索引**。如果使用 `[B,L-1]` 的 Logprob/Mask 表示，列 $j$ 一般与目标位置 $t=j+1$ 对应，不能直接把同名列当作同一个原始输入位置。

用 Mask 求和的最小 Loss 为：

$$
L_i=-R_i\sum_{t=1}^{L-1}m_{i,t}\,\log\pi_\theta(y_t\mid y_{<t}).
$$

它只应该选中真正属于 Policy Action 的位置。Tool Observation、Prompt、Padding 可能包含合法 Token ID，却不应该因为“不是空值”就一并训练。在多轮 Agent 场景，简单的 Prompt 长度与有效长度规则不足以表达这些区分，需要更准确的 Action Mask。

## 2 Token 与 200 Token：Reduction 不是后处理细节

假设两个 Response 长度分别为 2 和 4；为了看清权重，六个有效位置上的 $-\log\pi$ 都是 1，两个 Response 的 Advantage 都是 1。

如果目标是按**有效 Token** 求平均：

$$
L_{\mathrm{token}}
=\frac{2+4}{6}=1.
$$

每个 Token 的权重为 $1/6$，长 Response 的总体权重占 $4/6$。

如果目标是**每条 Response 先求 Token Mean，再对 Response 求 Mean**：

$$
L_{\mathrm{seqmean}}
=\frac12\left(\frac22+\frac44\right)=1.
$$

它的 Loss 数值也恰好是 1，但是短 Response 每个 Token 权重为 $1/4$，长 Response 每个 Token 权重为 $1/8$，两条 Response 的总权重各是 $1/2$。

**Loss 数值相等不等于梯度相等。** 两条样本的 Token 特征和梯度方向不一样时，这两种 Reduction 会产生不同更新。这是为什么一个训练框架把末尾的 `mean()` 从 Sequence 轴换到 Token 轴，即使代码只有一行，仍然可能改变算法目标。

这里还需要指出一个不容易注意的区别：单条 Response 的轨迹级 REINFORCE 通常对应 **Token Logprob Sum**；若为了训练稳定性除以长度，等于按长度重新加权这个估计量，不能在没有目标定义的情况下宣称两者无差别。

## 一次可执行的验证

已有 CPU 例子在一个 `[B,L,Vocab]` 小张量上实现了目标 Token 的 Shift/Gather、Mask 与不同 Reduction：

~~~bash
python examples/day_04/sequence_logprob_lab.py
python -m unittest discover -s examples/day_04 \
  -p 'test_sequence_logprob_lab.py' -v
~~~

先在纸上预测本课的 Mask `[0,1,1,0]`，再运行脚本；把起点移动一位，检查哪一处概率发生变化。将 Padding 从 Mask=0 改为 1，验证它会进入 Loss 与梯度，而不是只改变界面显示。

你也可以用 [Token Shift 互动格](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=3) 来验证索引判断。这个页面展示的是 Token 位置关系，不会真正运行大模型的 Forward/Backward。两者验证的层级不同：UI 帮你看到错位，CPU 测试能检查张量与梯度是否按预期变化。

## 从公式到实现，留下四个必查量

| 量 | 形状 | 应检查什么 |
|---|---|---|
| `token_ids` | `[B,L]` | Prompt/Response/Padding 的真实边界 |
| `logits` | `[B,L,Vocab]` | 第 $t$ 列预测 $t+1$ 的 Token |
| `action_log_probs` | `[B,L-1]` 或 packed `[N]` | 只选择实际采样动作，且来源 Policy/采样变换明确 |
| `action_mask` | 与 Action Logprob 同形状 | 每条 Response 的有效数量及非 Action 为零 |

如果 Token 数很大，一份朴素密集 Logits 的存储也可能显著：$B=2,L=4,Vocab=32000$、BF16 2 Byte/元素，则 `[B,L,Vocab]` 是 $2\times4\times32000\times2=512000$ Byte，约 **0.488 MiB**。这是这份 Tensor 的理论体积，不是 Transformer 的总显存峰值，也不能用来直接估计 GPU Token/s。

## 查证资料

- [OpenAI Spinning Up：Policy Gradient](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：轨迹概率如何分解为动作对数概率之和。
- [PyTorch `log_softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) 与 [`gather`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)：Token Logprob 的实际计算含义。
- [DAPO](https://arxiv.org/abs/2503.14476)：可作为不同 Token/Sequence Reduction 设计的研究案例；具体论文目标与本文玩具 REINFORCE 不应混同。

**独立判断题：** 如果一条长 Response 的最终 Reward 很高，当前 Token Sum 目标更容易对长度敏感，还是每条 Sequence 的 Token Mean 更容易？请先说明你比较的目标函数，再给出梯度权重，而不是凭直觉回答「长文本重要」。
