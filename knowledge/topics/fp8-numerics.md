# FP8 省的是哪部分成本，误差又会传到哪里？

FP8 不是“把整个训练程序的 Tensor 都换成 8 bit”。实际实现往往只让特定 GEMM 使用低精度输入与对应累加策略，同时保留一些状态或计算在较高精度。

最基本的量化过程可以写成：

$$
q=\operatorname{clip}\bigl(\operatorname{round}(x/s), q_{\min},q_{\max}\bigr),
\qquad \hat x=sq.
$$

Scale $s$ 控制动态范围与量化间距。不同 recipe 使用 per-tensor、blockwise 等不同尺度粒度，选择也受硬件与内核支持影响。只比较“FP8 vs BF16”不足以描述真正的数值契约。

## 为什么长输入短输出不一定得到同样收益

Prefill 通常包含较大的矩阵计算，低精度 GEMM 在适当形状下有机会提高计算吞吐；Decode 还受每 Token 读取权重/KV、Launch 和 Batch Shape 影响。是否真正提高端到端 RL 吞吐，要结合 Rollout/Learner 的阶段耗时和 Kernel Trace。

举一个明确的上限：同步 Step 中 Rollout 用时 42s、Learner 用时 18s。即便 Learner 提速 2 倍，端到端从 60s 变为 51s，加速只有 $60/51≈1.176$，而不是 2 倍。

## FP8 改变了什么概率

微小 Logits 扰动经 Softmax、Sampling、多个 Token 条件概率的累积，有可能改变生成长度和样本分布；但“FP8 输出变短”并不自动证明量化有问题。先固定同一权重、Token 序列和采样语义，对比 BF16/FP8 的逐 Token Logits 与 Logprob；再看 EOS 概率、Temperature、Stop 与模型路由。

对于 MoE，如果两套 Engine 选中了不同 Expert，还需要单独分析 Routing。R3 修复的是离散 Expert Assignment 的不一致，不是全部精度问题。

## 实现参考

[TransformerEngine FP8 Recipe](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) 讨论 Delayed Scaling、Current Scaling、Block Scaling 等策略；[Megatron Core 低精度训练](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/low_precision_training.html) 记录训练侧的格式和 Recipe 选项。支持矩阵随版本变化，不能在一个具体 GPU 上不检查就启用所有 Recipe。