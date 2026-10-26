# Day 04 · 问答与追问

## 题 1：PPO 的正负 Advantage，哪一边被 Clip？

给定 $\epsilon=0.2$，依次计算 $(A,r)=(+2,1.5),(-2,0.6),(+2,0.6),(-2,1.5)$ 的 clipped surrogate。指出哪两种情形里对 Ratio 的局部导数为零。

**判别依据：** 结果依次为 2.4、-1.6、1.2、-3.0；正 Advantage 上界之外平台，负 Advantage 下界之外平台。注意 Trainer 的 Loss 通常为该 surrogate 的负号。

**追问：** 如果把负号漏掉，会怎样改变更新方向？如果有多个样本共享参数，一个样本已 Clip 是否代表该 Token 概率以后无法变化？

## 题 2：Clip Range=0.2，为什么不能证明 KL 一定小？

给定 `π_old=[0.5,0.5]`、`π_current=[0.9,0.1]`，计算动作 A 的 Ratio 和 $D_{KL}(\pi_{\rm old}\|\pi_{\rm current})$。解释为什么某个样本的目标平台没有约束整个概率分布。

**判别依据：** A Ratio=1.8；KL 约 0.5108 nats；按全分布、不同状态和额外目标分析。不能把 Clip 当作数学上严格的 Trust Region。

**追问：** 如果 Rollout 使用 Top-p，而 Trainer 的 Old Logprob 是原始 Softmax，当前 Ratio 的语义满足你想要的 IS 假设吗？
