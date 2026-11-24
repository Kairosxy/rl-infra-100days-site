# 自回归、Diffusion、DiT 与 Flow Matching

自回归模型通过条件概率逐步生成序列；连续生成模型可以从噪声出发，在若干求解步内形成样本。Diffusion 是生成机制，DiT 是网络结构，Flow Matching 是向量场训练方法。比较这些技术时需要先确定表示空间和训练目标。

## Diffusion：从干净样本到噪声

以 DDPM 常用的高斯前向过程为例，给定干净样本 x_0 和标准高斯噪声 epsilon，

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

若 x_0=2、epsilon=0.5、累计 alpha=0.64，则 x_t=0.8×2+0.6×0.5=1.9。训练一种噪声预测模型时，采样 t 和 epsilon，让网络在输入 x_t、t、条件信息后预测 epsilon，优化相应的 MSE；也有 x_0 或 velocity 等其他参数化。推理从噪声开始，利用模型估计与求解器迭代恢复样本。

Latent Diffusion 可先用编码器将像素变成低维 Latent，在其上做生成，再由解码器恢复图像。VAE 的变分训练涉及重构和对先验的 KL；不应把任意图像压缩器都称为 VAE。

## Flow Matching：学习路径上的速度

设教学用线性插值从噪声 x_0 到数据 x_1：

$$
x_t=(1-t)x_0+tx_1,\qquad u_t=x_1-x_0.
$$

若 x_0=0、x_1=4、t=0.25，插值为 1，条件速度为 4。Flow Matching 回归相应概率路径的速度场；生成时从初始分布沿学到的向量场求解 ODE。这个单配对例子不代表未知数据分布上的向量场总是常数，也不是所有 Flow Matching 都使用直线路径。原始论文还讨论其他概率路径，包括与扩散相关的路径。

## DiT 与生成成本

DiT 将原本常用的 U-Net 去噪网络替换成对 Latent Patch 操作的 Transformer。以 512×512 图像、空间下采样 8 倍、Latent Patch=2 为例，得到 64×64 Latent 网格，再划成 32×32=1024 个位置。若求解器需要 20 次网络评估，主要计算取决于 20 次这类 Forward，而不是文本 Decode 的输出 Token 长度；某些 Guidance 或多阶段求解还会增加实际评估次数。

| 设计轴 | 常见分支 |
|---|---|
| 表示 | 像素、Latent、离散 Token |
| 训练目标 | AR CE、噪声/数据预测、速度场 |
| Backbone | U-Net、DiT、其他序列或混合结构 |
| 生成 | AR 采样、随机扩散求解、ODE、蒸馏后的少步生成 |

训练目标、Backbone 和求解器可组合，需按具体模型确认。RL/偏好优化如果作用于连续生成过程，动作与密度的定义可能依赖去噪转移或连续策略，不能直接套用文本的离散 Token logprob。

参考：[DDPM](https://arxiv.org/abs/2006.11239)、[Latent Diffusion](https://arxiv.org/abs/2112.10752)、[DiT](https://arxiv.org/abs/2212.09748)、[Flow Matching](https://arxiv.org/abs/2210.02747)。
