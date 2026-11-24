# ViT、VLM 与 Omni 的模态接口

ViT 解决图像的序列表示与特征提取问题；VLM 将视觉语义接入语言模型；Omni 的范围进一步涉及音频、视频及多模态输出。它们属于不同层级，不能仅按发布时间理解为替代关系。

## 图像如何成为序列

假设 RGB 图像 224×224，Patch=16×16。ViT 将图像划为 196 个 Patch，每块原始维度 16×16×3=768，再经投影形成 [196,D] 的序列（不计额外特殊 Token）。Self-attention 提取 Patch 之间的关系。如果边长变为 448，Patch 数增至 784；在标准全 Attention 中，位置对数约增至 16 倍。实际模型可能使用动态分辨率、下采样或局部 Attention。

CLIP 类图文对齐用匹配的图像、文本对学习表示空间。两对样本产生一个 2×2 的相似度矩阵，正确匹配对应对角线；对比学习提高正确匹配相对其他样本的分数。它不等于训练能够自回归生成文本的语言模型。

## VLM 的融合位置

~~~mermaid
flowchart LR
 I[Image] --> E[Vision Encoder]
 E --> P[Projector or Resampler]
 T[Text IDs] --> L[Language model]
 P --> L
 L --> A[Text logits]
~~~

图中展示视觉 Token/向量接入 LLM 的一种常见方式。其他架构可以使用 Cross-attention 或共享模态 Token 空间。模型输入需保留图像位置、视觉特征数量和 Text Token 的对应；视觉 Embedding 不一定存在普通文本 vocab ID。SFT/RL 中不能将所有视觉输入都当作模型生成动作。

## 从多模态输入到多模态输出

仅能读图/音频并输出文字，与能实时输出语音的模型有不同的 Serving 链路。Qwen2.5-Omni 报告中的 Thinker 处理文本生成，Talker 使用相应隐藏表示产生音频 Token，并有音频解码过程；它还考虑音视频时间对齐。这是一个具体架构案例，并非所有 Omni 都采用两个子模型。

假设一个 10s 音频按每秒 25 个时间步表示，原始时间步数为 250；若每步使用多个 Codec codebook，实际 Token 组织和计算量还取决于并行方式。视频则需同时考虑帧率、每帧 Patch、时间压缩与跨帧建模。不能只用文本长度评估总上下文和显存。

系统需要分开统计媒体预处理/Encoder、主模型 Prefill、文本 Decode、音频/图像生成 Head 与输出解码的耗时。对于多模态 RL，明确哪些输出属于策略动作、哪些反馈来自环境，及各模态损失如何归一化。

参考：[ViT](https://arxiv.org/abs/2010.11929)、[CLIP](https://arxiv.org/abs/2103.00020)、[Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)。
