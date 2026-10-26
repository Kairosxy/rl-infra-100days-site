# Day 05 · GRPO 为什么需要一组回答：相对奖励的信号与盲区

PPO 常用 Critic/Value Estimate 帮助构造 Advantage。对 Reasoning LLM，训练中也可以对**同一个 Prompt 采样多个 Response**，在组内比较各自 Reward，构造相对学习信号。DeepSeekMath 提出了 GRPO，后续 DeepSeek-R1 也使用了它。

关键问题不是「多生成几条就能算出 Advantage」，而是：**组里谁算正例、谁算负例，组内没有差异怎么办，不完整 Group 能不能先算，长度和过滤会怎样改变目标？**

[互动 Lab 05：改变四个 Reward，观察 Advantage 与组完整性 →](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=5)

## 从四条回答开始算

假设一个 Prompt 生成四条 Response，Verifier 给出二值 Reward：

$$
r=[1,0,1,0].
$$

使用一个常见的组内标准化 Advantage：

$$
\bar r=\frac{1}{G}\sum_{i=1}^{G}r_i,\qquad
\sigma=\sqrt{\frac{1}{G}\sum_i(r_i-\bar r)^2},\qquad
A_i=\frac{r_i-\bar r}{\sigma+\varepsilon_{\mathrm{std}}}.
$$

这里的标准差用总体分母 $G$；不同论文/实现可能选择不除标准差，或采用不同 $\varepsilon$ 与长度归一化。本课专门把这些选择当作目标差异，不把所有叫 GRPO 的实现混作一个公式。

先忽略很小的 $\varepsilon_{\mathrm{std}}$：

$$
\bar r=0.5,\qquad \sigma=0.5,\qquad
A=[+1,-1,+1,-1].
$$

正 Advantage 的两条回答会倾向被鼓励，负 Advantage 的两条会倾向被抑制。**绝对 Reward 没变，训练信号取决于同一个 Prompt 下还有哪些样本。**

这也解释了为什么 GRPO 的 Group 不只是一个无关紧要的 Batch 分组标签。如果把两条来自其他 Prompt 的 Reward 混进来，Shape 可能不变，但 Advantage 的比较对象已经错了。

## 最值得先看的反例：全对或全错

把四条 Reward 改为：

$$
[1,1,1,1]\quad\text{或}\quad[0,0,0,0].
$$

两种情况下 $r_i-\bar r=0$，标准差也为零。使用带正常数 $\varepsilon_{\mathrm{std}}>0$ 的标准化公式时，所有 $A_i=0$。这意味着**从组内相对奖励的 Policy Gradient 项看，没有区别方向**。

不要推断「这些 Prompt 没用」是无条件的结论。它们可能用于其他监督、KL、评估或课程采样机制；但对于这个**仅依赖组内相对 Reward** 的 PG 项，确实缺少更新信号。

DAPO 后续提出 Dynamic Sampling，正是针对容易产生全同奖励组等训练效率问题之一。过滤掉这些 Group 会影响哪些 Prompt 被训练、算力花在哪里，也可能改变实际采样分布；要同时检查统计效率与 Selection Bias，不能只统计「有效 Batch 更多了」。

## 第四条没回来时，提前算会发生什么？

假设第四条 Response 是慢请求，前三条先返回：

$$
[1,0,1].
$$

如果立刻把它当成 $G=3$ 的完整组：

$$
\bar r=\frac23,\qquad
\sigma=\sqrt{\frac{2}{9}}\approx0.4714,
$$

得到：

$$
A_{\text{前三条}}
\approx[+0.707,-1.414,+0.707].
$$

这和原来四样本组的 $[1,-1,1,-1]$ 不是同一个数值目标。**只要算法契约要求四条有效回答，就不能把“已收到 3 条”冒充“完成的 4 条组”。**

可选处理是等待第四条、为该组补投直到满足要求、或者明确接受可变组大小并重定义统计量。这些选择都会影响调度、资源和潜在偏差。哪一种更好需要按课程的目标和运行条件判断。

这条依赖会在后面的异步 DataSampler 中再次出现：Group 内 Advantage 的依赖是算法条件；Group A 完成后是否还要等 Group B 的外层 Batch Barrier，是系统调度条件；Trainer 的 Mini-batch/Weight Version 又是另一层约束。

## 长短回答的隐性权重来自哪？

组内得到的 $A_i$ 一般会传播到该 Response 的有效 Action Token。假设一个正样本只有 1 个有效 Token，另一个正样本有 9 个有效 Token，二者的 $A_i$ 同为 +1。即使每个 Token 的 $-\log\pi$ 都恰好为 1，最终聚合的梯度权重也取决于 Loss Reduction：

- 对所有有效 Token 求 Mean：十个位置各占 $1/10$，长样本合计占 $9/10$。
- 对每条 Response 先 Token Mean，再对两条 Sequence 求 Mean：两条样本各占 $1/2$。
- 对每条 Response 先 Token Sum，再求 Sequence Mean：长样本贡献和短样本可能相差九倍。

真实的 PPO/GRPO Loss 还有 Ratio、Clip、KL 和 Mask，不能把这些简式直接代替完整实现；但**长度权重的差异不会因公式复杂而消失**。

这也是为什么 GRPO 的后续工作并非只改超参数。DAPO 的 Token-level PG Loss 与 Clip-Higher、Dynamic Sampling，GSPO 的 Sequence-level Ratio，针对的是不同的目标定义和训练稳定性问题。要先问它们改变了哪一种计算对象，再问实验报告提升多少。

## 一个必须做的边界实验

打开 [Lab 05](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=5)，依次设置：

1. `[1,0,1,0]`：应显示均值 0.5、标准差 0.5、Advantage 约 `[1,-1,1,-1]`。
2. `[1,1,1,1]`：应显示全零 Advantage，提示组内没有区分度。
3. `[1,0,1]`：页面允许计算**临时三样本统计**，但要明确标记「不是要求四样本时的合法完整 Group」，不能误展示成同一目标。
4. `[1,1,0,0]`：与第一组只是交换位置；若按完整 Group 标准化，符号模式随奖励位置变化，均值/标准差不变。

再想两个工程问题：如果 Reward 回来时乱序，如何保证 Advantage 不贴错 Trajectory？如果为了提高有效样本率过滤全零方差组，哪些原始 Prompt 的分布可能被改变？这些会在后续的数据生产与异步章节逐步解决。

## 论文与实验依据

- [DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300)：GRPO 的主要来源。这里介绍的是一类常见组内标准化估计，不声称所有后续实现完全相同。
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948)：将 GRPO 应用于推理任务的代表性公开工作。
- [DAPO (2025)](https://arxiv.org/abs/2503.14476)：Dynamic Sampling、Clip-Higher、Token-level PG 等后续方法；Day 06 专门对照机制。
- [GSPO (2025)](https://arxiv.org/abs/2507.18071)：Sequence-level Ratio 的另一条方向，后续与 Token-level 方法比较。

本课互动页计算的是**奖励标准化**，没有运行 GPU 训练或完整 PPO/GRPO。它可以检查统计前提和数值，不能证明某个算法在真实模型上会带来性能提升。
