# Day 03 复习卡 · 梯度累积

- Autograd 累积 parameter.grad，但不会自动保证 Loss 的全局归一化正确。
- 等大小 Micro-batch 的平均梯度可以等价于完整 Batch；不等长 Token 需要按目标分母加权。
- AdamW 的 Moment 和 Step Counter 在每次 optimizer.step() 时更新，不能随意移动 Step 边界。
- Gradient Clipping 通常在累积完成后执行，因为裁剪是非线性操作。
- DDP 的 no_sync() 可减少重复梯度同步，但需要验证上下文覆盖了非最后一个 Micro-batch 的 Forward/Backward。

自测：解释为什么梯度范数一致不代表梯度相同；解释为什么 Global Batch 不变不保证训练轨迹逐位一致。
