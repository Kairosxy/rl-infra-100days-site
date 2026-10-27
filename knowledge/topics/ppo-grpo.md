# PPO、GRPO：Ratio、Clip 和 Group Advantage 在改变什么

先固定一条已经采样的动作。Old Policy 当时给它 0.2 的概率，Current Policy 现在给它 0.3，则 Ratio 为

$$
r=\frac{\pi_{\theta}(a|s)}{\pi_{\mathrm{old}}(a|s)}=1.5.
$$

PPO 的 Clipped Surrogate 不是简单把所有 $r$ 限制在区间内，而是根据 Advantage 的方向取较小目标：

$$
J_{\mathrm{clip}}=\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A).
$$

取 $\epsilon=0.2$：$A=+2$、$r=1.5$ 时，两个候选值是 3.0 与 2.4，最终为 2.4；这条样本继续提高概率时进入平坦区。对负 Advantage，平台在区间下界另一侧：$A=-2,r=0.6$ 时目标为 $-1.6$。平台是**单条样本的局部目标**，并非 KL 硬约束。

## GRPO 换掉的是什么

一个 Prompt 生成四条回答，Reward 为 $[1,0,1,0]$。按总体标准差的常见组内标准化定义，均值为 0.5、标准差为 0.5，Advantage 为 $[1,-1,1,-1]$。如果四条全对或全错，组内没有相对差异，单靠这个 Advantage 的 PG 项无法区分回答。

这里有两个容易忽略的事实。第一，**组内均值包含当前样本自身**，不等于经典的 Action-independent Baseline；有限 $G$ 会带来估计器差异。第二，Group Advantage 的计算对象必须明确：预期 $G=4$ 时拿三条 $[1,0,1]$ 直接标准化，得到的 $[0.707,-1.414,0.707]$ 不是四条完整组的提前结果。

## 真正影响训练的选择

很多所谓“GRPO 实现差异”在代码里只是几个 Reduction 和概率字段：

- Ratio 是逐 Token 还是逐 Sequence 计算；
- Loss 在有效 Token 维度求均值，还是先做 Sequence Reduction；
- Old Logprob 是否与真实生成分布、权重版本对应；
- 被过滤的全同奖励组改变了哪些 Prompt 的采样权重。

例如两条 Response 分别有 1 和 9 个 Action Token，使用 Token Mean 和 Sequence Mean 时，长回答的总权重可以从 $9/10$ 变到 $1/2$。不能把这种差异当作无关紧要的性能优化。

## 如何读后续工作

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 是 GRPO 的重要来源；[DAPO](https://arxiv.org/abs/2503.14476) 讨论采样、Clip 与 Token Loss 的困难；[GSPO](https://arxiv.org/abs/2507.18071) 研究 Sequence 层面的策略优化。这些工作并非只改了某个“默认超参数”，需要先确定估计对象、Loss Reduction 和概率分布。

工程上，可以对同一批固定 Token、Reward 和 Old/Current Logprob 实现两个 Reduction，对比每个 Token 的梯度；比“跑一次大模型看 Reward 是否上涨”更直接。

