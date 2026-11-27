# SGD、AdamW 与优化器状态

Backward 给出的是当前 Loss 对参数的梯度；Optimizer 决定如何使用梯度更新参数。两次训练即使得到完全相同的梯度，若优化器的历史状态不同，更新后的参数也可能不同。这是 AdamW 在分布式训练中需要独立维护 Optimizer State、在恢复时不能只加载模型权重的原因。

## 同一个梯度序列，三种更新方式

取一个标量参数，初值 $w_0=2$，连续两次收到梯度 $g_1=2$、$g_2=-1$，学习率 $\eta=0.1$。为隔离优化器行为，这里人为指定梯度，不从某个 Loss 重算；数值只是教学示例。

### SGD 与动量

最简单的 SGD 直接使用当前梯度：

$$
w_t=w_{t-1}-\eta g_t.
$$

两次更新得到 $1.8$ 和 $1.9$。若使用一个指数平均形式的动量 $m_t=\beta m_{t-1}+(1-\beta)g_t$，设 $\beta=0.5$，从零开始，则 $m_1=1$、$m_2=0$。若直接用该动量代替梯度，参数依次变成 $1.9$、$1.9$。它保留历史方向，第二次收到负梯度时恰好抵消已有动量。这一归一化形式用于解释机制，不是 PyTorch SGD 的 momentum buffer 初始化约定。

### Adam：历史方向与梯度尺度

Adam 为每个参数元素分别保存一阶矩和二阶原点矩的指数平均，后者不是梯度方差。忽略 AMSGrad 等变体，令初始状态 $m_0=v_0=0$：

$$
\begin{aligned}
m_t &= \beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t &= \beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\hat m_t &= m_t/(1-\beta_1^t),\qquad
\hat v_t=v_t/(1-\beta_2^t),\\
w_t &= w_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon).
\end{aligned}
$$

偏差修正用于消除从零初始化的指数平均在早期被压小的效应；它不表示策略梯度估计本身无偏。分母按元素计算，既不是全模型梯度范数，也不是全局梯度裁剪。

仍用上面的梯度，取 $\beta_1=\beta_2=0.5$、$\epsilon=10^{-8}$。第一次 $m_1=1$、$v_1=2$，修正后分别是 2 和 4，更新量约为 -0.1。第二次 $m_2=0$、$v_2=1.5$，修正后一阶矩为 0，更新量为 0。于是无衰减的 Adam 两次更新后都约为 1.9。

### AdamW：Weight Decay 不进入矩估计

在 Adam 中直接把 L2 正则的梯度 $\lambda w$ 加到 $g_t$，会让正则项也进入一阶和二阶矩，并接受自适应缩放。AdamW 则先对参数做独立衰减，再执行 Adam 的梯度更新：

$$
w_t=(1-\eta\lambda)w_{t-1}
-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon).
$$

矩估计仍使用原始梯度 $g_t$。设 $\lambda=0.1$，沿用前面的梯度和超参数：

| 步骤 | $m_t$ | $v_t$ | 修正后的一阶 / 二阶矩 | 无衰减 Adam 的参数 | AdamW 的参数 |
|---|---:|---:|---|---:|---:|
| 1 | 1 | 2 | 2 / 4 | 1.9 | 1.88 |
| 2 | 0 | 1.5 | 0 / 2 | 1.9 | 1.8612 |

第二步自适应梯度项恰好为零，AdamW 仍把 1.88 乘以 0.99。对照普通无动量 SGD，给梯度加 $\lambda w$ 与同样的独立衰减代数上等价；对 Adam 并不等价。这个区别是 [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) 的核心机制。

## 一次实际 Optimizer Step 保存什么

PyTorch v2.10.0 的 [AdamW](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/adamw.py) 继承 Adam，并设置 decoupled_weight_decay=True。在同版本 [Adam 实现](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/adam.py) 中，_init_group 仅收集 grad 不为 None 的参数；首次遇到该参数时初始化 step、exp_avg、exp_avg_sq。单张量路径先更新 step 并做 decoupled decay，再更新矩估计并修正偏差，最后更新参数。

因此有梯度张量且梯度元素全为零，与没有梯度张量不是一个状态。前者仍进入优化器；已有动量可以产生更新，即使动量为零，AdamW 的衰减仍可能改变参数。后者在上述版本的普通路径中不进入更新组，该参数的 step 和 moments 不前进。

下面用 FP64 CPU 标量复算两次更新，再分别输入 None 和零梯度。已在 PyTorch 2.10.0+cpu 执行，未涉及 GPU 或 fused kernel。

~~~python
import torch

param = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.AdamW(
    [param], lr=0.1, betas=(0.5, 0.5),
    eps=1e-8, weight_decay=0.1, foreach=False,
)

for grad, expected in [(2.0, 1.88), (-1.0, 1.8612)]:
    param.grad = torch.full_like(param, grad)
    optimizer.step()
    assert abs(param.item() - expected) < 1e-8

state = optimizer.state[param]
assert state["step"].item() == 2
assert state["exp_avg"].item() == 0.0
assert state["exp_avg_sq"].item() == 1.5

param.grad = None
optimizer.step()
assert state["step"].item() == 2
assert abs(param.item() - 1.8612) < 1e-8

param.grad = torch.zeros_like(param)
optimizer.step()
assert state["step"].item() == 3
assert abs(param.item() - 1.842588) < 1e-8
~~~

这里没有调用 lr scheduler，也没有 AMP 或分布式通信；不是训练控制器的全局 Skip 协议。若某个逻辑更新被判定无效，不能用人为构造的零梯度代替跳过 optimizer.step()。要核对具体框架是否在任何状态变更之前统一拒绝更新，参见 [梯度安全](../learner/gradient-equivalence.md)。

## 状态显存与分片

对 $N$ 个参数元素，普通非 AMSGrad Adam 的两个矩张量各有 $N$ 个元素。若采用 BF16 参数、BF16 梯度、独立 FP32 master weight 和 FP32 两个矩，则理想载荷如下：

| 对象 | 字节 / 元素 | 1B 元素时 |
|---|---:|---:|
| BF16 参数 | 2 | 2 GB |
| BF16 梯度 | 2 | 2 GB |
| FP32 master weight | 4 | 4 GB |
| FP32 exp_avg 与 exp_avg_sq | 8 | 8 GB |
| 合计 | 16 | 16 GB |

这是十进制 GB 的容量预算，不含 Activation、通信缓冲、临时张量、step 元数据和 allocator 余量。PyTorch v2.10.0 普通路径使用 zeros_like(param) 初始化矩，不能默认它总是 FP32，也不能默认存在独立 master weight；具体混合精度或分布式优化器可能维护额外 FP32 存储。

仅将两个 FP32 矩均匀分到 8 个 Rank，每 Rank 理想矩载荷由 8 GB 降为 1 GB。此结论只针对被分片的状态；参数、梯度是否分片，以及更新前后是否还需同步，各由 [FSDP / ZeRO](training-sharding.md) 或具体 Distributed Optimizer 实现决定。容量下降不代表 Step 一定更快。

SGD、动量法、Adam、Adafactor 等方法在状态结构、估计量与更新规则上不同；减少状态或改变更新方向并不天然保证与 AdamW 的训练轨迹等价。比较优化器需要分别验证训练效果、状态规模及实际执行路径，不能只看 Optimizer Step 的 kernel 时间。

## 恢复与验证

如果在第二次更新后仅保留参数 1.8612，却丢失 step=2、m=0、v=1.5，那么下一次收到相同梯度时，新的 AdamW 会使用不同的二阶矩和偏差修正。模型权重相同不等于训练状态相同。LR scheduler 的位置、混合精度 scaler 和随机状态也各自有生命周期。

最小恢复实验：从同一初始化出发，在第 2 次更新后保存参数及 optimizer.state_dict()，再用固定的非零梯度执行第 3 次更新；与从同一参数加载完整状态、仅加载权重两条分支比较参数和矩估计。相同 dtype 与单张量路径下可使用严格数值容差。随后才扩展到分片实现，核对逻辑梯度、状态所有权与每次成功更新的边界。

参考：[Adam 原文](https://arxiv.org/abs/1412.6980)、[AdamW 原文](https://arxiv.org/abs/1711.05101)、[PyTorch v2.10.0 AdamW](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/adamw.py)、[PyTorch v2.10.0 Adam](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/adam.py)。阅读 [训练更新生命周期](training-update-lifecycle.md) 可将这里的 Optimizer Step 放回完整训练链路。
