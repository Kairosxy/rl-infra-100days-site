# Day 04 · 同一批回答训练第二遍，为什么 PPO 要限制更新幅度？

模型已经生成了一批回答。当时它给某个已采样动作的概率是 0.2。Trainer 用这批数据更新一次参数后，给同一个动作的概率变成 0.3。

你想再利用这批昂贵的回答训练一遍。看上去只需要多调用一次 `backward()`；但现在动作来自**旧的 0.2 分布**，你正试图优化的是**新的 0.3 分布**。

这两种分布不能在推导里直接当成同一个。PPO 的 Ratio 和 Clip 就是从这个问题出发，才有必要存在。

## 旧样本能不能描述新策略？

先不谈 PPO。只有两个动作 A/B，Old Policy 给它们的概率是：

$$
\pi_{\rm old}=[0.2,0.8].
$$

Current Policy 已经变成：

$$
\pi_\theta=[0.3,0.7].
$$

假设你想知道 Current Policy 的平均 Reward，A 得 1、B 得 0，答案显然是：

$$
J_{\rm current}=0.3\times1+0.7\times0=0.3.
$$

但手头样本来自 Old Distribution。若不做修正，按旧样本平均 Reward，你估计的是 0.2。

概率论给了一个精确恒等式：只要新分布的正概率动作在旧分布下也有正概率（Support 条件），对固定函数 $f(a)$ 有：

$$
\mathbb E_{a\sim\pi_\theta}[f(a)]
=\mathbb E_{a\sim\pi_{\rm old}}
\left[\frac{\pi_\theta(a)}{\pi_{\rm old}(a)}f(a)\right].
$$

比率就是：

$$
r(a)=\frac{\pi_\theta(a)}{\pi_{\rm old}(a)}
=\exp\!\bigl(\log\pi_\theta(a)-\log\pi_{\rm old}(a)\bigr).
$$

本例 $r(A)=0.3/0.2=1.5$，$r(B)=0.7/0.8=0.875$。用旧分布求加权期望：

$$
0.2\times1.5\times1+0.8\times0.875\times0=0.3.
$$

这就是 Importance Sampling 的基本直觉。**Ratio 修正的是采样分布的差别，不是某个“经验上的学习率放大倍数”。**

但这里仍有一个重要限制：上面的等式说的是给定状态下的动作分布。真实 RL 中，状态访问分布也可能随策略变化。只给每个 Token 乘一个 Ratio，不会自动修复所有由长轨迹、环境反馈与不同状态分布引起的偏差。

## 为什么 Ratio 很大时，直接加权会让优化不稳定？

想象旧分布认为某个动作只有万分之一的概率，新分布认为它有 10% 的概率。它的 Ratio 就是 1000。该动作在旧样本中很难出现，但偶尔出现的一条可能拥有极大权重。

因此一种直接的 Importance Weighted 估计可以在期望层面合理，却在有限样本中出现很高的方差。使用旧样本反复优化时，模型还可能逐渐离产生这批数据的策略更远。

PPO 常见的 Clipped Surrogate 并不是简单把所有 Ratio 裁剪掉，而是对**有符号的 Advantage** 取两项中的较小值：

$$
J^{\rm clip}(r,A)=
\min\!\bigl(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A\bigr).
$$

Trainer 最小化的 Loss 通常是这项的**负号**，并可能加入其他 Critic、KL 或 Entropy 项。当前只研究单条样本对 Clipped Surrogate 的贡献，不能把这一项当作完整训练器。

## 为什么正负 Advantage 的平台在不同侧？

设 $\epsilon=0.2$，允许区间 $[0.8,1.2]$。

如果一个已采样动作相对较好，$A=+2$：

- $r=1.0$ 时，$J^{clip}=2.0$。
- $r=1.5$ 时，原项是 3.0，裁剪项是 2.4，取 **2.4**。
- $r$ 继续变成 1.8 时，结果仍是 **2.4**。这条样本对 Ratio 的局部梯度进入平台。
- 但 $r=0.6$ 时，原项 1.2、裁剪项 1.6，取 **1.2**，仍有使好动作恢复概率的梯度。

坏动作 $A=-2$ 则相反：

- $r=0.6$：原项 −1.2、裁剪项 −1.6，取 **−1.6**；继续降低概率不再改善这个样本的 Clipped Surrogate。
- $r=1.5$：原项 −3.0、裁剪项 −2.4，取 **−3.0**；坏动作概率升高仍会受到抑制。

把结果整理为一眼能判断的条件：

$$
\frac{\partial J^{clip}}{\partial r}=
\begin{cases}
0,& A>0 \ \text{且}\ r>1+\epsilon,\\
0,& A<0 \ \text{且}\ r<1-\epsilon,\\
A,&\text{其他非边界位置}.
\end{cases}
$$

这里的导数是**把当前 Advantage 视为固定常数时，单样本对 $r$ 的局部导数**；实际对模型参数求导，还要乘上 $\partial r/\partial\theta$。在折点处不声称存在唯一的通常导数。

这些区间不是任意选的。它们正好在限制某个样本对当前策略“已经朝有利方向走过头”的进一步鼓励。**Clip 是有偏的优化目标选择，用来改善更新行为；它不再等于前面那个未经裁剪的精确 Importance Weighted 期望。**

## 你能用 Clip Range=0.2 保证 KL 很小吗？

不能。用一个完整的二动作反例就能看出来：

$$
\pi_{\rm old}=[0.5,0.5],\quad
\pi_\theta=[0.9,0.1].
$$

对于 A，$r(A)=1.8$；若它是正 Advantage 样本，Clipped Surrogate 早已进入平台。但两份完整分布的 Forward KL 是：

$$
D_{\rm KL}(\pi_{\rm old}\|\pi_\theta)
=\tfrac12\log\tfrac{0.5}{0.9}
+\tfrac12\log\tfrac{0.5}{0.1}
\approx0.510826\ {\rm nats}.
$$

一条样本的局部梯度进入平台，不表示整个词表、其他状态或者别的样本无法继续改变参数。**Clip Range 不是全策略 KL 的数学硬上界。**

实际项目必须定义并记录 KL 的方向、比较对象、采样分布和统计口径。一个仅对采样 Token 算的近似 KL，也不等于对全词表完整积分得到的精确 KL。

## Tensor 里哪个字段应该固定不动？

一个常见的最小 Loss 核心为：

~~~python
ratio = torch.exp(current_log_probs - old_log_probs.detach())
unclipped = ratio * advantages.detach()
clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages.detach()
loss_per_token = -torch.minimum(unclipped, clipped)

loss = (loss_per_token * action_mask).sum() / action_mask.sum().clamp_min(1)
~~~

这段代码假定 `current_log_probs`、`old_log_probs`、`advantages`、`action_mask` 已经表示**同一批已采样动作且形状正确**。这里只是选择全局有效 Token Mean 的一个示例，并不是所有 PPO/GRPO 的通用 Reduction。

至少需要记住：

| 量 | 应该是什么 | 典型错误 |
|---|---|---|
| `old_log_probs` | 当前 PPO 优化阶段固定的比较基准 | 每次 Step 重新读取 Current，令 Ratio 总回到 1 |
| `current_log_probs` | 当前可微 Policy 对固定动作重新算的 Logprob | 拿无 Autograd 的 Rollout 数字反传 |
| `advantages` | 这一轮定义下的优势估计，当前 Loss 视为常数 | 无意穿过优势估计过程反传 |
| `action_mask` | 明确的可训练动作位置 | Padding 或 Tool Observation 进入 Loss |

注意 Old 和 Behavior 并不是任何情况下都相同。Rollout 时使用的权重版本、Temperature、Top-p 等采样变换，决定了真实生成分布。分母概率不是只要名字叫 `old_log_probs` 就一定代表正确行为策略。

## 亲自制造两个完全不同的问题

在 [PPO Ratio 实验](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=4) 中，固定 $\epsilon=0.2$，先把 $A$ 从 +2 改成 −2，预测平坦区会从哪侧移动。再把 $r$ 从 1.0 增加到 1.5，观察同一项的导数何时为零。

接着做一个代码层面的负例：把 `old_log_probs` 错设为每一次新算的 `current_log_probs`。这会让 Ratio 数值始终是 1，从而看不到真正的新旧策略差异。该错误比单纯「Clip 过多」更值得先排查。

学习 PPO 不能只会背 $\epsilon=0.2$。关键是能从**旧样本所来自的概率分布**推到 Ratio，能手算正负 Advantage 的两个分段目标，并且知道这套限制**没有**证明全局策略更新一定安全。

## 核对文献

- [PPO 原论文 (2017)](https://arxiv.org/abs/1707.06347)：Clipped Surrogate 的来源，以及与 Trust-Region 思想的关系。
- [OpenAI Spinning Up：PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：Clip Ratio、KL 监控和常见实现差异。
- [PyTorch `torch.clamp`](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html)：数值截断 API；注意真正的 PPO Loss 还需要 `min(rA,clipped(r)A)`。
