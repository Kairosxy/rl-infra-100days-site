# Day 03 · 图解：梯度累加与参数更新的分界

![梯度累积与 AdamW 更新边界](assets/gradient_accumulation.svg)

同一批次的 Micro-batch 可以分开 Forward/Backward，但每次求导使用的参数必须处于同一个更新点，最后才合成目标梯度并执行一次 Optimizer Step。图中需要辨认的是 **Backward 循环** 与 **Optimizer Step** 的边界。

对照五条样本的实际例子：2/2/1 切分后的局部平均梯度必须按 2/5、2/5、1/5 加权。把三个局部 Mean 等权平均，虽然遵守图中「最后才 Step」的时间顺序，仍会得到错误梯度。**正确同步顺序并不自动证明 Loss Reduction 正确。**

另一个边界在 DDP：`no_sync()` 可以减少非最后一个 Micro-batch 的梯度同步，但它不改变当前目标函数的分母。独立测试两件事：数学梯度是否一致、通信是否被正确推迟。
