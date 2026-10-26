# Day 03 图解 · 梯度累积和更新边界

![梯度累积与参数更新边界](assets/gradient_accumulation.svg)

## 为什么只能在累积结束后更新？

假设一次训练批次拆成 4 个 Micro-batch。每个 Micro-batch 先计算自己的梯度，梯度累加到同一组参数的 .grad 中。完成所有 4 次 Backward 后，再统一裁剪梯度并调用一次 Optimizer Step。

如果每个 Micro-batch 都调用 step()，第 2 个 Micro-batch 开始时参数已经改变，Adam 的 Moment 和 Step Counter 也改变了。后续梯度不再是同一参数点上对完整 Batch 梯度的累加。

## 等价条件检查

- 每个 Micro-batch 的 Loss 归一化与完整 Batch 目标一致。
- 所有梯度在同一次参数更新前累加。
- Gradient Clipping 在相同的逻辑边界执行。
- 不存在改变等价性的 Batch-dependent 操作或随机性差异。
- DDP 的梯度平均和梯度累积次数计算一致。
