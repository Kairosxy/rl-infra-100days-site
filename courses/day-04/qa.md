# Day 04 · 面试追问

## Reward 本身无法对参数反传，为什么 Policy 还能学习？

从 `J(θ)=Eτ[R(τ)]` 推导 Log-Derivative Trick。然后给定两个 Logit 都为 0，采样 A、Reward=1，手算对 A/B 的 Logit 梯度，并说明下降方向。

**判别依据：** 能解释是对**轨迹发生概率**求导，而不是对 Reward 程序求导；`∇log π(A)` 是可微路径，计算得到 `∂L/∂z_A=-0.5`、`∂L/∂z_B=+0.5`。

**追问：** 减去与当前状态有关但不依赖动作的 Baseline 为什么在期望下不改变策略梯度？需要使用 `Σ_a π(a|s)=1`。

## Token Shift 错一位，程序为什么仍能训练？

给出 `token_ids [B,L]` 和 `logits [B,L,V]`，说明位置 `t` 的 Logit 预测哪一个 Token。对于 Prompt 长度 $P$、Response 长度 $T$，写出响应动作与预测 Logits 的索引区间；说出一种 Shape 正常但标签错位的实现。

**判别依据：** 目标 `token_ids[:,P:P+T]`；预测 Logits `logits[:,P-1:P+T-1]`。检查 Padding、不同 Prompt 长度和 Mask，不能只核对 Tensor Shape。

**追问：** 两条样本有 1 和 3 个有效 Token，`token_mean` 与 `seq_mean_token_mean` 对短序列每个 Token 的梯度权重分别是多少？分别是 $1/4$ 与 $1/2$。

## Rollout Logprob 与 Trainer Logprob 不一样，就说明 FP8 有问题吗？

Rollout 使用 Temperature 采样，Trainer 通过当前权重在未变换的 Logits 上计算 Logprob。你会先检查哪几个定义，再考虑是否存在数值漂移？能否直接把 Rollout Logprob 当作 PPO 的 Old Logprob？

**判别依据：** 先确定采样分布、Token Shift、Mask、Policy Version 和固定 Old Policy 基准。数值不等可能来自比较的分布不同，不能直接归因于量化误差；复用 Old 必须满足算法契约。

**追问：** 两个样本 Advantage 为 +1/-1，Loss 恰好为 0，是否可能梯度非零？给一个张量级理由。
