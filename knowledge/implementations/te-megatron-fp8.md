# TransformerEngine + Megatron：FP8 训练方案的关键契约

FP8 加速的前提是模型包含适合低精度 Kernel 的计算，并且损失在训练目标上可以接受。TransformerEngine 提供 FP8 Recipe 与缩放管理，Megatron Core 在 Transformer Block 与并行训练中集成了相应算子。

## 与性能同等重要的数值控制

传统 FP8 E4M3/E5M2、Delayed Scaling、Current Scaling、Block Scaling 的量化动态范围与 Scale 更新方式不同。它们也并非所有 GPU 架构都支持相同优化路径。

尤其在 RL 中，Policy Gradient 依赖采样动作的 Logprob。就算 GEMM 层的误差很小，在 EOS、Top-k Expert 选择、长序列 Logprob 累加等位置可能产生可见差异。这是“训推是否使用同一 FP8 格式”之外还必须验证的因素。

## 实际工程审查

固定模型权重、Seed、Token ID、输入长度与训练精度配置，收集 BF16/FP8 的 Logits/Logprob 差异；分别监控 NaN、梯度范数、Router Assignment 和验证质量。

不同 TP/EP 组合改变 Scale、Kernel 的 shape 与通信调度，也可能改变性能。报告吞吐必须给出硬件、模型、上下文、Batch 和实现版本。不要引用浮点理论 TFLOPs 直接声称 End-to-end RL 提速。

资料：[TransformerEngine FP8 Primer](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) · [Megatron Core Low Precision Training](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/low_precision_training.html)。