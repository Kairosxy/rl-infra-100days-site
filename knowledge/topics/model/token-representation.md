# Tokenizer、Embedding 与语言模型训练目标

Tokenizer 把原始输入编码为模型约定的离散 ID。Embedding 把 ID 查成可训练向量。Decoder 将向量变换为上下文表示，LM Head 再将表示映射到整个词表的 logits。Tokenizer 本身不是 Embedding 层。

## 词表与表示空间

BPE 从较小符号集合出发，通过语料中的频繁相邻对构建合并规则；Unigram 等方法使用不同的分词模型。对假设的词表，同一字符串可能被切成 [12,36]，也可能因词表或 Chat Template 差异被切成 [82,91,5]。因此跨引擎对齐不能只比较解码后的字符串，必须核对模型使用的 ID、特殊 Token 和模板。

设 vocab=128000、hidden=4096，Embedding 权重为 [128000,4096]，共 524288000 个元素。BF16 理论载荷为 1000 MiB。输入 [B,S]=[2,8] 的 ID 变成 [2,8,4096] 的向量。LM Head 通常将最后一维投影到 vocab；若权重绑定（weight tying），Embedding 与 Head 可以共享参数存储，但并不是所有模型都绑定。

## Next-token Prediction

给定序列 [BOS, A, B, EOS]，模型在 BOS 位置预测 A，在 A 位置预测 B，在 B 位置预测 EOS。

| 输入位置 | 当前已见前缀 | 监督目标 |
|---|---|---|
| 0 | BOS | A |
| 1 | BOS A | B |
| 2 | BOS A B | EOS |

SFT 通常沿用 next-token 目标，但通过 label mask 选择有效监督位置。Attention Mask 控制可访问的上下文，Loss Mask 控制哪些目标贡献训练，两者不能互换。Packing 多条样本时还需要防止跨样本 Attention，并明确位置和边界。

~~~python
# logits: [B, S, V], labels: [B, S]
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, logits.size(-1)),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
~~~

此片段假定 labels 已按任务标好 ignore_index，且模型尚未内部 shift；不能对已经移位的 labels 再移一次。训练序列中 prompt 可见，并不意味着 prompt Token 都要参与 loss。

## 文本之外的表示

ViT 常把图像切成 Patch，再投影为向量。例如 224×224 的图像按 16×16 划分，得到 196 个 Patch（不含额外特殊 Token）；VLM 可以把视觉编码器的输出通过 projector 或 cross-attention 接入语言模型。音频可使用连续特征或 Codec Token，图像生成可使用连续 Latent，VLA 的 Action Head 可能是离散 Token，也可能输出连续动作。它们的输入输出空间和训练目标不一定能统一成文本 Cross Entropy。

[Transformer 前向与注意力](transformer-attention.md) 解释这些表示进入序列模型后的计算。参考：[Hugging Face Tokenizers components](https://huggingface.co/docs/tokenizers/python/latest/components.html)、[ViT](https://arxiv.org/abs/2010.11929)。
