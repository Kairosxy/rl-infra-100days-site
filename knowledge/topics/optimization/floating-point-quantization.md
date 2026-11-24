# 浮点格式、量化与混合精度

低精度的核心不是只把 dtype 名称改短，而是选择数值表示、Scale 粒度、计算/累加精度与转换位置。存储为 8 bit 不代表所有算子都以 8 bit 计算，更不代表整个训练状态的显存减半。

## 浮点表示

对标准正规二进制浮点数，可写成 $(-1)^s2^e(1+f)$。指数决定动态范围，尾数决定同量级的分辨率；零、次正规数、无穷和 NaN 还依赖格式规定。

| 格式 | 指数位 | 显式尾数位 | 1 附近向上的格点间隔 |
|---|---:|---:|---:|
| FP32 | 8 | 23 | 2^-23 |
| FP16 | 5 | 10 | 2^-10 |
| BF16 | 8 | 7 | 2^-7 |
| E4M3（FP8 示例） | 4 | 3 | 2^-3 |

在 BF16 中，1 附近向上的间隔为 0.0078125；1.001 按最近值舍入通常回到 1。浮点加法不是严格结合的，改变 Reduction 顺序可能造成差异。E4M3 的有限值、异常值与饱和策略需按具体格式/转换算子确认。

## 量化不是统一的整数舍入

INT8 对称均匀量化的一种形式为 $q=\operatorname{clip}(\operatorname{round}(x/s),-127,127)$，反量化 $sq$。令 s=0.1、x=0.26，结果为 q=3，重构 0.3。浮点量化则对缩放后的实数投影到非均匀浮点格点。FP8 的实际机制见 [FP8 Scale](fp8-numerics.md)。

若每 128 个元素使用一个 FP32 scale，且数据按 1 byte/元素存储，原始载荷比无 scale 的理想 8 bit 增加 4/128=3.125%；还没有考虑 padding 与 metadata。更细粒度的 scale 可以降低部分 outlier 的影响，但增加 scale 计算、传输或读取成本。Block format、per-token activation、per-channel weight、KV quantization 是不同的配置轴。

PTQ 在已有模型上选择和应用量化方案，常需校准；QAT 在训练目标与 Forward 中模拟或使用量化误差，让参数适应表示约束。二者不能只按最终权重文件后缀区分。

## 训练里的混合精度

典型混合精度可让部分 GEMM 使用 BF16/FP16/FP8，并为累加、归约、参数更新或敏感算子选择较高精度。FP16 下常用 loss scaling 缓解梯度下溢；clip 应作用于 unscale 之后的真实梯度。BF16 的指数范围较大也不意味着永远没有溢出、下溢或舍入问题。检查时需分别记录 parameter、activation、gradient、accumulator、optimizer state 和 KV dtype。

继续阅读 [训练更新等价性](../learner/gradient-equivalence.md)。参考：[PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)、[AMP Examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)、[TransformerEngine](https://github.com/NVIDIA/TransformerEngine)。
