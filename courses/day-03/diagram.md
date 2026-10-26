# Day 03 · 交互图：把预测位置对齐到真实 Action

[Lab 03：Token Shift / Action Mask](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=3) 逐格显示 Prompt、Response 和 Padding，同时单独显示**预测者 Logit 的位置**与**被预测目标 Token 的位置**。

点击“错误 Shift”后，预测者与目标同位，Tensor Shape 仍然成立；对照是否仍能解释为下一个 Token 的条件概率。更改 Prompt 长度后观察 Action Mask 变化。

静态表格更适合精确记录 [$B,L,V$] Tensor；交互图只展示 Token 位置，不假装运行了真实 LLM。
