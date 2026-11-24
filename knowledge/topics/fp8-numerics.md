# FP8 的 Scale 改变了什么，如何判断 EOS 变化是否由它引起？

同一份模型权重，Rollout 从 BF16 切到 FP8 后，回答变短，Learner 重算的 Token Logprob 也出现差异。仅凭这两个现象不能归因于 FP8：先确认两侧计算的是同一权重版本、同一 Token、同一条件分布，再检查量化 Scale 和算子误差。尤其需要纠正一种写法：**FP8 不是将实数等间距地 `round(x / scale)` 成整数**，那是更接近 INT8 的均匀量化模型。

## FP8 浮点格点与 Scale

以 E4M3 前向格式为例，量化先乘缩放系数，再舍入到该浮点格式可表示的数；解量化再乘缩放系数的倒数。本文把乘法缩放记为 $a>0$：

$$
q=\operatorname{cast}_{\mathrm{E4M3}}(a x),\qquad
\hat x=\operatorname{float}(q)/a.
$$

E4M3 具有指数和尾数，**相邻可表示值的间隔不是常数**。例如固定 $a=1$，在 PyTorch 2.10.0 CPU 上，`1.03 -> 1.0`、`1.10 -> 1.125`、`0.90 -> 0.875`。这是一次可复算的 dtype 投影，不代表 TransformerEngine 会把最终 Logits 直接转为 FP8，更不是任何实际模型的 Logprob 误差测量。

一个具体的 Scale 约定来自 TransformerEngine 的 `DelayedScaling`：在其默认 Scale 计算策略下，E4M3 的最大有限值为 448，若用于缩放的 `amax=2`、`margin=0`，则下一缩放系数为 $a=448/2=224$；`margin=1` 则为 112：

$$
a_{\mathrm{new}}=\frac{448}{\operatorname{amax}\,2^{\mathrm{margin}}}.
$$

**这里的 `amax` 在 Delayed Scaling 中来自历史统计，而不是必须来自正在量化的当前 Tensor。** `Float8CurrentScaling` 改为按当前 Tensor 计算缩放；`MXFP8BlockScaling` 则使用每 32 个元素一组的 Block Scale（E8M0）。三者的 Scale 更新时机与分组粒度不同。不同实现还可能把 $1/a$ 存为 `scale_inv`，不能混用两个方向的变量；其它 FP8 格式也不能直接套用 E4M3 的 448。

源码依据是 NVIDIA TransformerEngine 固定提交 [`35a8e2f9` 的 Recipe 定义](https://github.com/NVIDIA/TransformerEngine/blob/35a8e2f9db996e2957a20fc0d17df1ce03849a0a/transformer_engine/common/recipe/__init__.py)，其中可以定位 `DelayedScaling`、`Float8CurrentScaling` 和 `MXFP8BlockScaling`；[同提交的硬件支持矩阵](https://github.com/NVIDIA/TransformerEngine/blob/35a8e2f9db996e2957a20fc0d17df1ce03849a0a/docs/support_matrix.rst) 给出架构限制。以上是指定源码版本的定义，**不是本地 GPU 已运行某个 Recipe 的证明**。

## 数值误差如何变成 EOS 概率差异

下面刻意构造一个只有 `EOS` 和 `continue` 两个动作的 Toy 分布。假定原始 Logits 为 `[-4.10, 0.10]`，再把这两个数分别投影到 E4M3 格点（$a=1$），得到 `[-4.00, 0.1015625]`：

| 对照 | EOS Logit | Continue Logit | $P(\mathrm{EOS})$ | 假定每步概率恒定时 $1/P(\mathrm{EOS})$ |
|---|---:|---:|---:|---:|
| 原始 Toy | -4.10 | 0.10 | 0.014774 | 67.69 Token |
| E4M3 格点 Toy | -4.00 | 0.1015625 | 0.016277 | 61.43 Token |

其中 $P(\mathrm{EOS})=\exp(z_{\mathrm{EOS}})/(\exp(z_{\mathrm{EOS}})+\exp(z_{\mathrm{continue}}))$；最后一列只是独立、恒定终止概率下的几何分布期望，**不是对自回归语言模型生成长度的预测**。真实 FP8 GEMM 通常量化参与乘法的激活和权重，而非把最终 Logits 直接做这次转换。这个 Toy 只证明：如果最终 Logits 产生上述差异，Softmax 和 EOS 统计会怎样变化。它既不能证明 FP8 必然缩短输出，也不能证明差异来自某个特定 Recipe。

下面的 CPU 实验无需 GPU 或 TransformerEngine，复算表中数值：

```python
import torch

assert torch.__version__.startswith("2.10.")  # 本文记录的 CPU 版本；其他版本可单独核验
logits = torch.tensor([-4.10, 0.10], dtype=torch.float32)
toy_fp8 = logits.to(torch.float8_e4m3fn).float()
p_ref = logits.softmax(dim=0)[0].item()
p_toy = toy_fp8.softmax(dim=0)[0].item()
print(toy_fp8.tolist(), p_ref, p_toy, 1 / p_ref, 1 / p_toy)
assert torch.allclose(toy_fp8, torch.tensor([-4.0, 0.1015625]))
assert abs(p_ref - 0.014774) < 1e-5
assert abs(p_toy - 0.016277) < 1e-5
```

这一代码片段已使用 **PyTorch 2.10.0+cpu** 执行复算；不依赖 CUDA，因此不能据此推断 GPU FP8 GEMM 的溢出处理、Kernel 精度或端到端性能。

## 为什么算子变快，RL Step 仍可能改善有限

FP8 的潜在收益主要取决于实际落在低精度 Tensor Core GEMM 上的工作量；Prefill 的较大 GEMM 与小 Batch Decode 的权重/KV 访问、Kernel Launch、通信瓶颈不能一概而论。同步 RL Step 的一个教学上限是：Rollout 42 s、Learner 18 s，若**仅 Learner** 时间减半且阶段不重叠，整步从 60 s 变成 51 s，加速 $60/51\approx1.176\times$，而不是 2 倍。这里 42/18 s 为假设，没有对应的 GPU Trace。

## 验证顺序：先对齐概率契约，再定位 Scale

固定模型 Checkpoint 和 Policy Version、输入 Token、Position、Action Mask、Sampling 配置、TP/EP 布局；优先用 Teacher-forcing 在**相同 Token 和上下文**上比较 BF16 与 FP8 的 Raw Logits、Raw Logprob，以及每层输入/权重 `amax`、Recipe、Scale/Inverse Scale。再单独对比 Temperature/Top-p 之后实际采样的 Behavior Distribution，避免拿经过截断的 Rollout Logprob 与训练侧 Raw Softmax 直接相减。若是 MoE，还要固定或记录 Top-k Expert ID，先排除离散 Router 分歧。

随后才在固定 Prompt 分布、多 Seed 的采样实验中统计 EOS Probability、Stop Reason、Response Length P50/P95 和质量；按每个 Input/Output 长度区间分开对照。能够稳定复现同 Token 的数值误差，并追到具体量化 Tensor/Recipe，才有理由尝试该层 BF16 回退或调整 Scale。若误差只出现在采样路径，先检查 Stop/Temperature/Top-p；若 Token 或 Version 不同，不能把数值差异归因于 FP8。

这条排查路径与 [FP8 生成长度/Logprob 漂移案例](../cases/fp8-logprob-drift.md)、[Logprob 的概率契约](logprob-contract.md) 及 [TransformerEngine + Megatron 实现边界](../implementations/te-megatron-fp8.md) 相互衔接。完整 GPU 对照和不同 Scale 策略的实际精度/吞吐尚未运行。
