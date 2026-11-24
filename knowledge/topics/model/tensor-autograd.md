# 张量、矩阵乘法与自动微分

训练引擎做的事情可以归结为三步：用参数计算输出，根据输出定义 Loss，再把 Loss 对参数的导数交给 Optimizer。张量的 Shape 决定算子能否连接；梯度的 Shape 与被求导变量相同。

## 从矩阵到计算图

设 $X$ 为 [B,H]，权重 $W$ 为 [H,O]，输出 $Y=XW$ 为 [B,O]。一个具体例子：$X=[1,2]$，$W=[3,4]^T$，输出为 11。若目标值为 10，$L=(Y-10)^2/2=0.5$，则上游梯度 $G=\partial L/\partial Y=1$。

$$
\frac{\partial L}{\partial W}=X^TG=[1,2]^T,\qquad
\frac{\partial L}{\partial X}=GW^T=[3,4].
$$

这就是 Linear Backward 的两个 GEMM。实际 PyTorch Linear 将 weight 存成 [O,H]，数学记法和存储方向不要混淆。对于 [B,H]=[4,4096]、O=8192，Forward 理论 FLOPs 是 $2BHO\approx 2.68\times10^8$，乘加按两次浮点操作计算；不是算子延迟。

~~~python
import torch

x = torch.tensor([[1., 2.]], requires_grad=True)
w = torch.tensor([[3.], [4.]], requires_grad=True)
loss = ((x @ w - 10) ** 2).sum() / 2
loss.backward()
assert w.grad.flatten().tolist() == [1., 2.]
assert x.grad.flatten().tolist() == [3., 4.]
~~~

这里是可复算的 CPU 教学片段。若学习率 0.1，SGD 会将 w 更新到 [2.9,3.8]。Gradient Descent 改变的是参数，不是直接编辑上一次输出。

## 为什么训练需要保存 Activation

Forward 构建计算图并保存 Backward 需要的对象。上例的参数梯度必须访问 X，而输入梯度必须访问 W。多个算子共享同一个输入时，各路径的梯度贡献要累加。例如 $y=x^2+x$，在 $x=2$ 的导数是 5，而不是只取其中一条路径。

Activation Checkpointing 选择少保存部分中间值，在反向前重算；它以额外计算换取较低的激活峰值。Inference 不需要训练反向图，但仍可能保存 KV Cache，这两种内存生命周期不同。

## Logprob 与梯度的边界

Autograd 对当前前向图求导。服务端返回 float 类型的 logprob 不能恢复原来的计算图；即使该数值与 Trainer 完全相同，也不能直接对服务端 float 调 backward。训练时必须在当前参数下重算动作的可微 logprob，并用采样时的行为概率、reward 或 advantage 构造 Loss。详见 [Token Logprob 契约](../learner/logprob-contract.md)。

参考：[PyTorch Autograd mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html)、[torch.nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)。
