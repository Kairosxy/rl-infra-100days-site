# Day 03 · 一眼看出 Token Shift 错没错

一条序列：

```text
位置       0       1       2     3      4
Token     BOS     问题      y1    y2     PAD
```

Prompt 占前两个位置，Response 是 `y1,y2`。

**想一想：** 预测 `y1` 的 Logits，来自位置 **1** 还是位置 **2**？

**答案：位置 1。** 因果语言模型的 `logits[t]` 预测的是下一位置的 Token。因此预测 `y1,y2` 的 Logits 来自 **1、2**，而目标 Token 位于 **2、3**。

可以这样向同事说明：

> Teacher Forcing 不会重新采样答案，而是用当前模型计算已生成 Token 的条件 Logprob。目标 Token 和预测它的 Logits 要错开一位，最后只对真实 Action 位置计算 Loss。

**再做一个判断：** 如果误把 PAD 的 Mask 设为 1，Shape 会不会报错？

通常不会。Loss 可能正常下降，但多优化了一个不属于响应的目标位置。本例在目标位置 1–4 上的 Action Mask 应为 **[0,1,1,0]**。

想核对 Tensor 切片：[讲义里的 Shift/Gather 代码](lesson.md)。
