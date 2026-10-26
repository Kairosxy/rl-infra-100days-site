# Day 03 · 5 分钟复习：2/2/1 的分母陷阱

给定五条 MSE 样本，在相同权重 `w=[0.3,-0.2]` 下，每条的梯度依次为：

~~~text
[-1.4,  0.0]
[ 0.0, -0.4]
[-4.2, -8.4]
[-2.4, -1.2]
[ 3.0, -3.0]
~~~

**完整 Batch Mean** 是 `[-1.0,-2.6]`。拆为 2、2、1 后，三个局部 Mean 是 `[-0.7,-0.2]`、`[-3.3,-4.8]`、`[3,-3]`。如果直接等权平均三个局部 Mean，就得到错误的 `[-0.3333,-2.6667]`；按 `2/5、2/5、1/5` 加权才恢复原目标。

**必须一起记住的更新边界。** 先在同一权重上累积梯度，再 Unscale（若适用）、Clip、Optimizer Step。提前 `optimizer.step()` 会更新权重和 Adam 状态；逐 Micro-batch Clip 也不等于整体 Clip。

**分布式补充。** DP=2、每 Rank Micro-batch=2、Global Batch=256，对等切分的累积次数是 64。DDP `no_sync()` 延迟同步，但不能替你修正 Token Mean 的分母。

复核：[五样本 CPU 梯度对照](../../examples/day_03/gradient_accumulation_lab.py)。
