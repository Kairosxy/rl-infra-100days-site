# Day 03 · 问答与追问

## 题 1：Logits 与 Action Token 错一位，为什么程序仍能运行？

给定 `logits [B,L,V]`、`token_ids [B,L]`，Prompt 长度 $P$，Response 长度 $T$。写出选中 Response Token Logprob 的切片、Shift 与 Gather。设计一个 Shape 一致但目标 Token 错位的反例。

**判别依据：** 明确 `logits[:,P-1:P+T-1]` 预测 `token_ids[:,P:P+T]`。能说明 Mask 是在对齐后的目标 Token 上计算，不只检查数组长度。

**追问：** Padding Mask 合法是否足以排除 Agentic RL 工具 Observation 对 Loss 的污染？

## 题 2：Loss 的数值一样，Gradient 为什么可能不一样？

两条序列的长度为 1、3，所有有效位置 `-logprob` 都等于 1。计算 Token Mean 与 Sequence Mean of Token Mean，并指出四个位置的梯度权重。

**判别依据：** 两种 Loss 都是 1；前者所有 Token 权重 $1/4$；后者短样本 $1/2$、长样本各 $1/6$。

**追问：** 如果不同 Micro-batch 有效 Token 数不同，各 Micro-batch Loss Mean 等权平均是否等价于全局 Token Mean？
