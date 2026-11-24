# Transformer：从 Token 到 Logits

Decoder-only Transformer 的一次前向把 token IDs 映射为每个位置的词表 logits。训练、推理和 RL 的概率计算共享这条链路，区别在于输入位置、状态保留方式、反向计算和分布式切分。

## 前向张量

以一个假设模型为例：hidden size 4096，32 层，32 个 Query Head、8 个 KV Head，head dim 128，词表 128000，MLP 中间维度 11008。下表忽略 tensor parallel 和 padding，B 表示 batch，S 表示序列长度。

| 对象 | 形状 |
|---|---|
| input_ids | [B,S] |
| embedding / residual stream | [B,S,4096] |
| Q | [B,32,S,128] |
| K / V | [B,8,S,128] |
| attention output | [B,S,4096] |
| gated MLP 中间值 | [B,S,11008] |
| logits | [B,S,128000] |

Tokenizer 决定 token ID 和序列长度。Embedding 是查表；LM Head 把 hidden state 投影到词表。两者可以共享参数，但需要查实际模型配置，不能由形状相同推断共享。

## Attention 与位置

每个 Query Head 计算 QKᵀ，并对可见的 Key 位置做 softmax，再对 V 加权求和。因果 mask 限制当前位置只看见自身及历史位置。RoPE 旋转 Q、K 的特征对，使注意力内积包含相对位置关系。RoPE 不等于 KV 压缩，也不自动保证超过训练长度时的质量。

MHA 的 KV Head 数等于 Query Head 数。GQA 让一组 Query Head 共享一个 KV Head；本例每四个 Query Head 对应一个 KV Head。实际算子可以避免物理复制 K/V。Attention 仍然有 32 个 Query Head，因此不能简单把计算量也除以 4。

对全长 S 的标准 dense attention，打分与聚合的主要算术量随 S² 增长；因果 mask、FlashAttention 和局部注意力会改变具体常数、内存流量或有效计算范围。FlashAttention 保留标准 Attention 的 QK 打分，通过分块计算避免将完整注意力矩阵写回 HBM。

## 参数、激活和缓存

假定无 bias，Q、K、V 和输出投影的参数量为 4096×(4096+1024+1024+4096)=41,943,040。SwiGLU 的三个投影参数量约 3×4096×11008=135,266,304。Norm、Embedding、LM Head 与其他模块需另计。以上为该假设模型单层的参数预算。

BF16 KV Cache 每 token 需要 2×32×8×128×2=131072 bytes，即 128 KiB。8192 token 为 1 GiB，未计分配粒度、缓存页表及并行策略。训练激活不是 KV Cache：反向需要的中间值取决于算子保存策略、重计算及 sharding；不能拿推理 KV 公式代替训练激活估算。

训练通常计算多个位置的 logits 和梯度；自回归 decode 每步只加入新 token 的 K/V，但查询仍要读取历史缓存。RL learner 对固定轨迹重算 logprob 时，必须保持 token IDs、位置、mask 和概率分布的约定一致。参见 [KV 调度](../serving/kv-scheduling.md) 与 [Logprob 契约](../learner/logprob-contract.md)。

## 最小验证

读取真实模型 config 中的 hidden_size、num_attention_heads、num_key_value_heads、head_dim 和 num_hidden_layers。打印 Q/K/V 张量的实际形状，再计算 1k 和 8k token 的 KV 字节数。对 GQA 用相同 Query Head 数、不同 KV Head 数做独立对照，分别测量 cache pool 分配量和 Attention Kernel 时间，不用其中一个指标替代另一个。

参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[GQA](https://arxiv.org/abs/2305.13245)、[FlashAttention](https://arxiv.org/abs/2205.14135)。
