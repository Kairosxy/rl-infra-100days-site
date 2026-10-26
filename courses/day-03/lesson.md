# Day 03 · 同样五条样本，为什么拆成 2/2/1 会改变梯度？

训练时显存不够，常把一个 Global Batch 拆成多个 Micro-batch。拆完以后连续调用五次、三次或六十四次 `backward()`，并不自动等于原来的一次 Batch 更新。**要得到同一个目标函数的梯度，必须保留原来每个样本的权重，并且在同一参数点上求导。**

用现有实验中的五条二维样本，我们可以把错误算到具体数值，而不只说「要除以 accumulation steps」。

## 从完整 Batch 的梯度开始

考虑无 Bias 的线性模型 $\hat y_i=w^\top x_i$，Loss 是五条样本的 MSE：

$$
L(w)=\frac{1}{5}\sum_{i=1}^{5}(w^\top x_i-y_i)^2,\qquad
\nabla_w L=\frac{2}{5}\sum_{i=1}^{5}(w^\top x_i-y_i)x_i .
$$

固定初始权重 $w=[0.3,-0.2]$，实验里的数据是：

| 样本 | $x$ | $y$ | 单条样本的梯度 $2(\hat y-y)x$ |
|---|---|---:|---|
| 1 | [1,0] | 1 | [-1.4, 0] |
| 2 | [0,1] | 0 | [0, -0.4] |
| 3 | [1,2] | 2 | [-4.2, -8.4] |
| 4 | [2,1] | 1 | [-2.4, -1.2] |
| 5 | [1,-1] | -1 | [3.0, -3.0] |

例如第三条预测 $\hat y=0.3+2(-0.2)=-0.1$，残差是 $-2.1$，所以梯度为 $2(-2.1)[1,2]=[-4.2,-8.4]$。五条梯度相加为 $[-5,-13]$，完整 Batch 的平均梯度是：

$$
g_{\text{full}}=[-1.0,-2.6].
$$

这里求的是同一组权重处的 MSE 梯度，还没有执行 Optimizer Step，也没有混合精度误差。

## 拆成 2/2/1 后，分母应该是多少

如果按顺序拆成三个 Micro-batch，它们各自的**局部平均梯度**是：

$$
g_1=[-0.7,-0.2],\quad
g_2=[-3.3,-4.8],\quad
g_3=[3,-3].
$$

一个常见的错误写法是「每个 Micro-batch 先取 Mean，再把三份 Loss 各除以 3」。它给出：

$$
g_{\text{wrong}}=(g_1+g_2+g_3)/3
=[-0.3333,-2.6667].
$$

错误不在浮点精度，而在第五条样本被放大了：它独占第三个 Micro-batch，却与两条样本组成的前两个 Micro-batch 权重相同。

要保持完整 Batch 的**样本平均**，按各个 Micro-batch 的样本数加权：

$$
g_{\text{correct}}
=\frac{2}{5}g_1+\frac{2}{5}g_2+\frac{1}{5}g_3
=[-1.0,-2.6].
$$

如果每个 Micro-batch 已经计算 `loss.mean()`，就乘以 `micro_batch_size / global_sample_count` 再 `backward()`。另一种等价写法是每次把该 Micro-batch 的 **Loss Sum** 除以全局样本数；仓库的示例采用后一种。

这条规律在 LLM 中更容易踩坑。假设目标是所有有效 Response Token 的平均 Loss：

$$
L_{\text{token}}
=\frac{\sum_{i,t}m_{i,t}\ell_{i,t}}{\sum_{i,t}m_{i,t}} .
$$

分母必须是参与这次更新的**全局有效 Token 数**。如果三个 Micro-batch 分别有 100、100、10 个有效 Token，把三份局部 Token Mean 等权平均，就会让最后 10 个 Token 的每个位置获得不成比例的权重。**如果算法目标本来是每条 Sequence 等权，那就需要另外定义 Sequence Mean，不能一律套 Token Mean。**

## `backward()` 能多次调用，`step()` 不能随便提前

PyTorch 的 `loss.backward()` 会把计算到的梯度累加到 `parameter.grad`，直到显式清零或覆盖。因此可以让模型在同一组参数上依次处理多个 Micro-batch，最后再做一次更新：

~~~python
optimizer.zero_grad(set_to_none=True)
for x_mb, y_mb in micro_batches:
    per_sample_loss = (model(x_mb) - y_mb).square()
    (per_sample_loss.sum() / total_samples).backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
~~~

这里的 `total_samples` 必须与真实完整 Batch 的分母一致。上面的代码只描述 MSE 样本均值，不是直接可套用的 PPO Loss。

若每个 Micro-batch 后立刻 `optimizer.step()`，第二个 Micro-batch 看到的权重已经变了。AdamW 的 $m_t$、$v_t$ 和 Step Counter 也会向前走；「一个完整 Batch 的一次 AdamW 更新」变成「不同权重上的三次 AdamW 更新」。现有测试明确检查到，完整 Batch 的 Step Counter 为 **1**，错误 Micro-batch 更新为 **3**。

梯度裁剪也必须对准同一个边界。设两份 Micro-batch 梯度为 $[3,0]$ 与 $[-3,4]$，总和是 $[0,4]$。如果最大范数设为 2：

- 先求和再 Clip，得到 $[0,2]$。
- 两份分别 Clip 后求和：第一份为 $[2,0]$，第二份为 $[-1.2,1.6]$，合计 $[0.8,1.6]$。

结果不同，因为 Clip 是非线性操作。AMP 场景还要在裁剪前按具体实现完成 Unscale。

## DDP 里还多一道通信边界

普通 DDP 在 Backward 期间使用梯度同步，让不同 DP Rank 的梯度按其约定聚合。非最后一个 Micro-batch 如果每次都同步，虽然数学上可能仍正确，却会产生额外通信。PyTorch 的 `DistributedDataParallel.no_sync()` 可延迟这些同步；上下文必须覆盖**对应的 Forward 和 Backward**，不只是后面的 `backward()`。

例如 DP=2、每 Rank Micro-batch=2、Global Batch=256，若每个 Rank 都处理相同数量的数据，累积次数为：

$$
K=\frac{256}{2\times 2}=64.
$$

可以前 63 次在 `no_sync()` 中累积，最后一次按 DDP 规则同步。注意这里谈的是 **DDP 默认的梯度平均约定**；若不同 Rank 有不同数量的有效 Token，仅平均各 Rank 的局部 Token Mean 仍可能改变全局目标。还需要根据实际分母和聚合语义归一化。

即使数学上等价，不同 Micro-batch 切分也未必产生逐位相同的浮点结果：求和顺序、Dropout、BatchNorm 等都可能影响精确等价。正确的验证目标是先核对定义，再选择适当精度与容差比较梯度。

## 用已有代码检查两种错误

~~~bash
python examples/day_03/gradient_accumulation_lab.py
python -m unittest discover -s examples/day_03 -p 'test_*.py' -v
~~~

[`gradient_accumulation_lab.py`](../../examples/day_03/gradient_accumulation_lab.py) 使用 `torch.float64` 固定数据与初始权重，对比完整 Batch、正确累积与错误的等权 Micro-batch Mean。正确版本与完整 Batch 应在 $10^{-12}$ 级容差内一致；错误版本不会一致。 [测试文件](../../examples/day_03/test_gradient_accumulation_lab.py) 还会检查「每个 Micro-batch 都执行 AdamW Step」造成的参数与 Step Counter 差异。

建议先不跑代码，手算 $[-1,-2.6]$ 和 $[-0.3333,-2.6667]$。如果这两组值的来源已经清楚，后面读 PPO / GRPO 中不同的 Token、Sequence 和 Group Reduction 才不会只记一个 `loss.mean()`。

## 相关文档

- [PyTorch Autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html) 与 [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)：分别验证梯度累积和 Optimizer State 更新。
- [PyTorch DDP：`no_sync()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)：同步边界及 Forward 必须在上下文内的约束。
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)：Mixed Precision 下 Unscale、Clip、Step 的顺序。
