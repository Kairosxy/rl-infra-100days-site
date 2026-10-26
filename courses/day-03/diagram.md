# Day 03 · 为什么 Token 必须按索引画，而不是画成几个箭头

[Lab 03：Token Shift 与 Action Mask](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=3) 固定一个可见的绝对 Token 索引轴，另用一行标出**预测者 Logit 的位置**、目标 Action 的位置与 Mask。

读图时先确认「Logits at $t-1$ → target Token at $t$」。切换“错误 Shift”后，即使目标 Tensor Shape 相同，预测者的位置已经改变，这才是需要注意的失败。

这张位置图不包含真实 Tokenizer、非文本模态或多轮工具 Observation；如果这些情况进入训练，必须另外维护可训练 Action 的位置契约。不要用绿色高亮代替逻辑条件。
