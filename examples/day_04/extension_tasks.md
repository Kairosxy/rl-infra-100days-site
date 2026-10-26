# Day 04 扩展任务：序列 Logprob 与 Loss Reduction

先写测试，再实现关键函数。

## A. 序列位置对齐

实现输入 input_ids: [B,L]、logits: [B,L,V] 的 selected-token Logprob，支持不同 Prompt 长度与 Padding。

验收：位置 t 的 logits 预测 t+1；与逐步 Prefix Forward 的参考值一致；故意偏移一位时测试失败；Padding 不进入有效 Logprob。

## B. Token Mean 与 Sequence Mean

构造长、短两条 Response，设置相同的 per-token Advantage，分别实现 Token Mean 与 Sequence Mean / Sequence Sum。计算每条轨迹的梯度权重，并写清所选目标的分母。

## C. Policy Version 对照

构造 Behavior v1、Old v1、Current v2 和 Reference v0 的小型概率分布，计算 exp(current_logprob - old_logprob)，说明它与 Reference KL 的作用不同。

## 需要理解的现象

若两个样本的正负 Advantage 对称、Logprob 相等，平均 Loss 可能抵消为 0，但梯度仍非零。Loss 标量本身不能单独代表“有没有学习”；还要检查梯度和参数更新。
