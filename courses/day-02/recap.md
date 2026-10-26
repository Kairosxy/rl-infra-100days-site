# Day 02 · 方差从何而来

固定 `p(A)=0.25`，Reward(A)=1、Reward(B)=0：

$$
g(A)=0.75,\quad g(B)=0,\quad
\mathbb E[g]=0.1875,\quad
\operatorname{Var}(g)=0.10546875.
$$

如果 Baseline 为常数 $b$：

$$
g_b(A)=0.75(1-b),\qquad g_b(B)=0.25b.
$$

$b=0.75$ 时，两种结果都变成 0.1875，方差为 0；**它不是平均 Reward 0.25**。这是确定性二动作、单参数模型的特殊情况，不能外推成 LLM 总能把方差消掉。

**最容易让证明失效的改动：** 把 `b` 写成当前采样动作的 Reward；它会把每个优势值都消成零。动作相关 Baseline 不满足 `E[b∇logπ]=0` 所需条件。

[调整 Baseline，观察两个可能梯度点](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=2)。
