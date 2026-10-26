# Day 03 面试问答

## 题 1：梯度累积什么时候等价于完整 Batch？（10 分）

**评分标准**
- 写出完整 Batch 的平均 Loss：2 分
- 推导 Micro-batch 梯度加权关系：3 分
- 正确处理 Loss Reduction 与有效 Token 数：2 分
- 只在累积结束后执行 Optimizer Step：2 分
- 提到浮点顺序和随机算子的细微差异：1 分

**追问**：每个 Micro-batch 的有效 Token 数不同，还能简单地每份 Loss 除以累积次数吗？合格回答应指出：如果目标是全局 Token Mean，需要以全局有效 Token 数作为分母；局部 Token Mean 再等权平均通常会改变权重。

## 题 2：为什么 Gradient Clipping 通常放在累积之后？（10 分）

**评分标准**
- 说明 Clip 是非线性操作：3 分
- 解释逐 Micro-batch Clip 与全局 Clip 不等价：3 分
- 说明 AMP 中先 Unscale 再 Clip：2 分
- 能提出梯度向量对照实验：2 分

**追问**：如果两个梯度向量的 Norm 一样，能否说明更新相同？不能；方向和各参数分量也必须一致。
