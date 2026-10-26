# Day 01 · 模型答对了一次，参数到底改了哪里？

假设一个模型只有两个可选答案。给它一道题「1+1=?」，动作 A 表示回答 2，动作 B 表示回答 3。现在它选择 A 的概率是 50%，恰好采样到 A，评分器给了 1 分。

下一次回答 2 的概率会增加多少？

你可能觉得「答对了，提高 A 的概率」没有什么可讲。但这里有一个必须弄清的断点：**评分器只返回整数 1，梯度是从哪里来的？** 如果连这个问题都不能算出来，把同一动作换成几百个 Token，再分到几十张 GPU 上，只会让错误变得更难发现。

## 先不要训练：模型当前到底相信什么

用一个参数 $z$ 表示模型的倾向：

$$
p=\pi_z(A)=\sigma(z)=\frac{1}{1+\exp(-z)},\qquad
\pi_z(B)=1-p.
$$

开始时 $z=0$，所以两个动作各有 50% 的概率。注意模型并不是「选择了 A，所以 A 的概率是 100%」。**动作是一次随机采样的结果，概率是采样之前的分布。**

评分规则固定为：

$$
R(A)=1,\qquad R(B)=0.
$$

从所有可能的抽样结果看，模型平均每次能得多少分？

$$
J(z)=\mathbb E_{a\sim\pi_z}[R(a)]
=pR(A)+(1-p)R(B)=p.
$$

$J$ 是想提高的**期望回报**，不是这一次已经观察到的 1 分。当前 $J(0)=0.5$。由于 $p=\sigma(z)$，我们甚至知道如果可以直接计算理想梯度：

$$
\frac{\mathrm dJ}{\mathrm dz}=p(1-p)=0.25.
$$

但真实 RL 不可能枚举完整语言空间。训练时拿到的是本次采到的动作 A 和对应评分。接下来需要把这一个观察结果变成可以反传的 Loss。

## 1 分本身不可微，动作的 Logprob 可以

A 已经被采样成一个离散动作，求导不能穿过「掷骰子得到了 A」这个操作。换个方向：既然这个动作得了 1 分，我们提高它发生的**对数概率**：

$$
L(z)=-R(A)\log\pi_z(A)=-\log p.
$$

对于 Sigmoid：

$$
\frac{\partial\log p}{\partial z}=1-p,\qquad
\frac{\partial L}{\partial z}=-(1-p).
$$

代入 $p=0.5$：

$$
L=-\log0.5\approx0.693147,\qquad
\frac{\partial L}{\partial z}=-0.5.
$$

这不是刚才的理想梯度 $0.25$。**不要急着认为哪个算错了：$-0.5$ 来自一次具体采样；$0.25$ 是对所有采样结果求期望后的梯度。** 要比较二者，得考虑抽到 B 的情况。

若采样到 B，$R(B)=0$，上述最基础的样本 Loss 为 0，对 $z$ 的梯度也为 0。两种情况以各 50% 的概率出现，所以单样本**梯度下降方向的期望**是：

$$
\mathbb E[-\nabla_zL]
=0.5\times0.5+0.5\times0
=0.25
=\frac{\mathrm dJ}{\mathrm dz}.
$$

这就是最小的 Score Function/REINFORCE 估计器在本例中起作用的原因：不是评分器提供了导数，而是通过采样动作的 $\log\pi_z(a)$ 找到了对参数的可微路径。推导成立的条件包括：样本按用于估计的策略分布采集，以及 Reward 规则不直接随这个参数求导。

## 真正执行一次 SGD，概率变成多少

固定已经采样到 A，学习率 $\eta=0.4$：

$$
z_{\mathrm{new}}
=z-\eta\frac{\partial L}{\partial z}
=0-0.4(-0.5)=0.2.
$$

更新之后：

$$
\pi_{\mathrm{new}}(A)
=\sigma(0.2)\approx\boxed{0.549834}.
$$

因此，**一次采到正确动作并按这条 Loss 更新，A 的概率从 50% 增加到了约 54.98%**。不是增加到 100%，也不是增加 40 个百分点。

现在换个问题：如果同一条记录中，开发者把 `loss=-reward*log_prob` 写成 `loss=reward*log_prob`，程序会怎样？

梯度符号变成 $+0.5$，SGD 把 $z$ 更新到 $-0.2$，正确动作的概率降到 $\sigma(-0.2)\approx0.450166$。**Loss 能反传、Optimizer 能 Step、没有 NaN，并不意味着方向是对的。**

如果你想真正检查模型是否在“奖励正确动作”，只看 Reward Mean 或 Loss 是否下降都不够；最小证据是：冻结输入与动作，更新前后比较**该动作在同一条件下的 Logprob**，并核对梯度符号。

## 五行 PyTorch，把公式落到参数上

下面代码在 CPU 上即可运行；无 GPU、无外部模型、无随机采样噪声。为了把变化看清，我们**强制采用本次已知动作 A**：

~~~python
import torch

z = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.SGD([z], lr=0.4)

before = torch.sigmoid(z).item()
log_prob_a = torch.nn.functional.logsigmoid(z)
reward = torch.tensor(1.0)

loss = -reward * log_prob_a
optimizer.zero_grad()
loss.backward()
print("gradient:", z.grad.item())  # -0.5
optimizer.step()

after = torch.sigmoid(z).item()
print("probability:", before, after)  # 0.5, 约 0.549834
assert z.grad.item() < 0
assert after > before
~~~

每一行对应上面的式子：`z` 是参数，`log_prob_a` 是采到动作 A 的条件 Logprob，`reward` 是不可微的观察值，`loss.backward()` 的梯度只通过 Logprob 回到 `z`，`optimizer.step()` 改变参数。

真实大模型中，`z` 会变成大量模型参数，采样动作会是一串 Token；但一次更新必须满足的条件没变：**Reward 对应的是哪个动作？Logprob 来自哪个策略？Mask 选择了哪些实际生成的动作？梯度能否回到 Current 参数？**

不需要此时背完所有字段。只要看出这条链上任何一环连错，最终都可能出现「训练在进行，优化目标却不对」。

## 你可以亲自改变三个条件

打开 [交互实验：策略概率与实际采样](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=1)。它只模拟采样分布，不会真的更新模型参数；上面的 PyTorch 片段负责演示更新。

**第一个改动：** 将初始概率 $p$ 从 0.5 改到 0.02。采样 8 次时，一次正确动作都没出现并不奇怪。系统记录「这轮全错」不能直接推出理论期望梯度为零。

**第二个改动：** 把 Reward(A) 从 1 改成 2，Reward(B) 从 0 改成 −1。先写出新的 $J=3p-1$，再求 $\mathrm dJ/\mathrm dz=3p(1-p)$。如果奖励规则错了，优化器仍可能非常有效地学会错误行为。

**第三个改动：** 在上面的代码里去掉负号，检查 Assertion 哪一个会失败。这个负例比「训练 Loss 数值大约正常」更有诊断价值。

这节课研究的是**一次动作得到反馈后发生的参数更新**。它没有多轮状态、价值函数、策略重用、PPO Clip，也没有 GPU 加速。把这条最短路径走通，已经足以判断许多复杂 RL 训练代码是不是在朝正确方向更新。

## 进一步核对

- [OpenAI Spinning Up：Policy Gradient](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：采样轨迹、Log-Derivative 与更新目标的推导。此处将它缩减为可枚举的二动作实例。
- [REINFORCE / Williams (1992)](https://link.springer.com/article/10.1007/BF00992696)：基于采样动作及其概率估计梯度的经典工作。
- [PyTorch `logsigmoid`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html) 与 [`SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)：检查代码运算的确切定义。

**你现在能验证的结论：** 从 $z=0$、采样 A、$R(A)=1$ 出发，正确更新后 $p=0.549834$；反向写错负号时 $p=0.450166$。这两个数来自公式与可执行代码，不是大模型训练的测量结果。
