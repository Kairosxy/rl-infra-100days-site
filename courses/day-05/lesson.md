# Day 05 · 四个答案里有两个正确，模型到底该奖励哪两个？

你给同一个 Prompt 生成了四条回答，Verifier 返回：

$$
r=[1,0,1,0].
$$

现在你没有单独训练一个 Critic，也没有每个 Token 的人工标签。能不能仅凭这四个分数训练模型？

可以，但必须先把**“这个回答好不好”转换成“它相对同一个问题的其他回答好不好”**。这正是 GRPO 一类方法使用 Group 估计 Advantage 的直觉来源。困难不在于算一个 Mean，而在于：为什么这个比较能提供梯度？全对时为什么反而没有组内信号？为什么少收到一条、或者改变长度归一化，都会改变训练目标？

## 先把四条 Reward 变成有方向的信号

设每个 Prompt 产生 $G=4$ 条 Response，它们被认为属于**同一个比较组**。采用常见的组内标准化：

$$
\bar r=\frac1G\sum_i r_i,\quad
\sigma=\sqrt{\frac1G\sum_i(r_i-\bar r)^2},\quad
A_i=\frac{r_i-\bar r}{\sigma+\epsilon_{\rm std}}.
$$

为了看清主要结果，先忽略很小的正数 $\epsilon_{\rm std}$。

本例：

$$
\bar r=0.5,\qquad \sigma=0.5,\qquad
A=[+1,-1,+1,-1].
$$

四条回答没有变；改变的是 Policy Gradient 使用的系数。一个得 1 分的回答获得正 Advantage，一个得 0 分的回答获得负 Advantage。区别不是 Verifier 的概率输出，而是**在这个 Prompt 的四个答案之间做的比较**。

两个重要前提：

一是这四条必须按同一个组的约定采样并评分，不能随便混入其他 Prompt 的回答。二是这里的 $\sigma$ 使用总体标准差，分母为 $G$；某些实现不用标准差，或者使用不同的 $\epsilon$ 和长度加权，不能把所有名叫 GRPO 的 Loss 当成同一个算式。

## 为什么全对也是零信号？

把 Reward 改为：

$$
[1,1,1,1]\quad\text{或}\quad[0,0,0,0].
$$

两种情况下，每条 $r_i-\bar r$ 都等于 0。即使分母为零，实际采用 $\epsilon_{\rm std}>0$ 后，仍得到所有 $A_i=0$。

**“这一组全对”不等于“已经知道怎样进一步提高其概率”。** 从这一个纯组内相对 Reward 的 Policy Gradient 项来看，没有办法区分谁更值得鼓励。它并不意味着训练器的 KL、蒸馏、监督或其他项也必然为零，更不意味着该 Prompt 对评估没有意义。

用 [GRPO 交互实验](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=5) 切换四个 Reward：先试 `[1,0,1,0]`，再试 `[1,1,1,1]`。你不需要背零方差的处理规则，只要知道：没有相对差异，按这个目标就没有相对更新方向。

## 少一个回答，结果不是“同一个组的早期版本”

Verifier 并发执行时，前三条分数可能先回来：

$$
[1,0,1].
$$

如果现在直接把它当成 $G=3$ 的组来算：

$$
\bar r=\frac23,\qquad
\sigma=\sqrt{\frac{2}{9}}\approx0.471405,
$$

得到：

$$
A_{\rm 三条}\approx[0.7071,-1.4142,0.7071].
$$

原本四条回答时，三条对应的 Advantage 是 $[1,-1,1]$。**多一个还没完成的 Response，足以让已收到三条的统计结果不同。**

如果算法与实验设计明确要求四个有效答案，就要等齐、补充采样到有效数，或明确废弃这一组；不能把未完成的三样本组偷偷当成原定四样本组更新。允许动态 Group Size 当然是另一种设计，但要重新说明估计器、采样过程和筛选偏差。

这里有一个容易混淆的工程判断：即使 Group A 的四个答案已经齐了，它可以计算 Advantage，也不等于整个 Trainer 必须立刻执行一次 Step。Gradient Accumulation、当前参数版本和资源安排是另外的执行约束。**算法必需的组完整性，与实现额外增加的整批等待不是同一回事。**

## 一个不容易察觉的偏差：自己的 Reward 也进入了 Baseline

有人会说：“组内均值就是一个 Baseline，既然减去 Baseline 不改变 Policy Gradient 的期望，为什么不直接用？”

问题是，普通状态 Baseline 不依赖**本次采样的动作**；但组内均值 $\bar r$ 包含自己的 $r_i$。对于有限 Group，二者不完全相同。

不妨先去掉标准差归一化，用一个只有两个动作的 Bandit 看清楚：

$$
p(A)=0.5,\quad R(A)=1,\quad R(B)=0.
$$

设 $G=2$，两条独立采样的结果分别是 A、B。则 $\bar r=0.5$，未标准化 Advantage 为 $[+0.5,-0.5]$。两条动作的 Score 分别是 $+0.5$ 和 $-0.5$，所以这一组样本平均梯度为：

$$
\hat g_{\rm group}
=\frac{0.5\times0.5+(-0.5)\times(-0.5)}{2}
=0.25.
$$

看起来正好等于真实的 $p(1-p)=0.25$，但这只是采到了 A、B 组合时。

两次采样四种有序结果 AA、AB、BA、BB，各概率 $1/4$：

- AA：两个 Reward 都是 1，Group Advantage 为 0，梯度估计为 0。
- BB：两个 Reward 都是 0，梯度估计同样为 0。
- AB 和 BA：估计值各是 0.25。

所以对所有可能的二样本组求期望：

$$
\mathbb E[\hat g_{\rm group}]
=\frac{0+0.25+0.25+0}{4}
=\boxed{0.125}.
$$

而精确目标梯度是 0.25。对于这个**未标准化、独立同分布、组内均值包含自身的例子**，期望梯度恰好缩小一半。更一般地，不带标准差且组大小固定为 $G$ 时，可得到 $(G-1)/G$ 的系数。

这是组内相对估计与经典 Action-independent Baseline 的本质差别之一。**加上标准差归一化之后，偏差关系又会变化，不能把 $(G-1)/G$ 原封不动搬到完整 GRPO。** 某些实现还会采用重新中心化、长度归一化及其他策略，这些设计都要按其实际目标函数检查。

这个反例的意义不是宣称 GRPO 错误，而是避免一个常见误解：组内 Advantage 是为训练构造的**相对估计量**，不等于“把完全无偏的 Value Baseline 免费替换成 Group Mean”。

## GRPO 还不止于算出四个 Advantage

有了 $A_i$，Trainer 通常要在已采样 Response 的 Token 上构造与 Policy Logprob 相关的优化项。如果使用 PPO 式 Ratio/Clip，还会涉及 Old/Current 策略的概率、固定的 Advantage 和对应的 Mask。

假设四条 Response 长度为 $[1,1,1,9]$，长序列最后一个样本恰好是正 Advantage。按全局 Token Mean 时，这条 Response 有 9 个 Action 位置；按每条 Sequence 先 Token Mean 再求 Group Mean 时，每条回答的总权重相同。**即使 Group Reward 和 Advantage 完全相同，两种 Reduction 也可能产生不同的优化方向。**

因此检查一个 GRPO 实现不能只确认「Reward 标准化代码写对了」，还要查：

| 量 | 应追问的事实 |
|---|---|
| Group ID / Size | 哪些样本一起估计均值和方差？允许动态长度吗？ |
| Reward | 谁评分、是否允许重试、乱序时是否仍对应同一 Response？ |
| Advantage | 是否标准化、标准差如何定义、全同奖励如何处理？ |
| Logprob / Ratio | Behavior、Old 与 Current 的概率定义是否一致？ |
| Action Mask / Reduction | 哪些 Token 参与训练，长短 Response 的权重是多少？ |

这些字段属于实验设计的一部分，不能用“它们都叫 Loss”代替具体检查。

## 试着设计一个能抓出错误的测试

不用 GPU，只需四个浮点 Reward：

1. `[1,0,1,0]`：应得到 $\bar r=0.5$、$\sigma=0.5$、$A≈[1,-1,1,-1]$。
2. `[1,1,1,1]`：应得到 $A=[0,0,0,0]$，而不是 NaN。
3. `[1,0,1]`：如果目标要求 $G=4$，应标为**组不完整**；临时三样本结果不能覆盖四样本状态。
4. 将原四条 Reward 顺序打乱：必须同步移动对应的 Trajectory ID，不能把 Reward 按返回位置贴给别的 Response。

[打开互动实验](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=5) 可先看到前三种现象。第四种必须在真正的 Sampler 数据结构和负例测试中验证，单靠一张图无法证明关联正确。

## 放回研究语境

- [DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300)：GRPO 的公开来源。这里解析了组内相对 Reward 这一类估计器，未假定所有后续实现细节一致。
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948)：Verifier/Reward 与 Reasoning Policy 的代表性结合；其报告的模型结果并非本课程实测。
- [DAPO (2025)](https://arxiv.org/abs/2503.14476)：动态采样、Clip 与 Token-level Objective 的改进，值得用「哪些组有信号、如何加权有效 Token」这个问题去阅读。
- [GSPO (2025)](https://arxiv.org/abs/2507.18071)：将 Sequence-level Ratio 作为另一种设计方向。它并不等于换了名字的标准 GRPO。

能完成这节课的真正标志，不是会背“GRPO 不需要 Critic”，而是**能说明一个比较组的统计对象是谁、它在估计哪个信号、什么情况下根本没有区分度，以及代码怎样在数值都正常时悄悄改变训练目标**。
