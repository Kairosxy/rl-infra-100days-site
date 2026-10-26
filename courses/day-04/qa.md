# Day 04 面试问答 · Policy Gradient 与 Logprob

## 题 1：Reward 不可微，为什么仍然能更新 Policy？（10 分）

**问题**：从期望 Reward 开始，解释 Log-Derivative Trick 如何得到策略梯度。

**评分标准**
- 写出期望目标 \(J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau)]\)：2 分
- 说明对轨迹概率求导，而不是对 Reward 求导：2 分
- 使用 \(\nabla p=p\nabla\log p\)：2 分
- 解释 Logprob 如何把离散动作与参数连接起来：2 分
- 说明 Baseline 可以降低方差但不改变期望梯度的条件：2 分

**追问**：如果两个样本的 Advantage 一正一负，Loss 标量正好为 0，是否意味着没有梯度？不一定。需要检查每个样本的 Logprob、梯度和参数更新；正负项可能抵消标量值，却不抵消梯度。

## 题 2：Rollout Logprob 能否直接作为 PPO 的 Old Logprob？（10 分）

**评分标准**
- 区分 Behavior/Rollout、Old、Current 和 Reference Policy：3 分
- 说明采样温度、Top-p、策略版本可能改变概率语义：3 分
- 解释 PPO Ratio 使用当前 Logprob 与固定 Old Logprob 的比值：2 分
- 给出可验证的对照实验：2 分

**追问**：若 Rollout 用 Temperature 采样，而 Trainer 用原始 logits 计算 Logprob，二者不同就一定是 FP8 误差吗？不是。先确认概率分布定义、采样配置、Token Shift、Mask 和 Policy Version，再分析数值精度。
