# SFT、RLHF、RLVR 与偏好优化

后训练首先要区分两件事：监督信号从哪里来，以及参数如何根据该信号更新。SFT 给定目标输出；偏好学习给出相对排序；在线 RL 从当前或近期策略采样，并根据反馈更新。RLHF、RLAIF、RLVR 描述反馈范式，不是互斥的梯度估计算法。

## 一组 Prompt 的三种数据

假设同一 Prompt 有答案 A、B。SFT 数据可能直接指定 A；偏好数据记录 A 优于 B；RLVR 的验证器可能返回 A=1、B=0。三者的信息含量和数据分布不同。

| 训练方式 | 目标信号 | 是否要求本轮在线生成 |
|---|---|---|
| SFT | 目标 Token 的条件概率 | 否 |
| Reward Model | 人类或 AI 的排序、分数 | 不要求，数据可以离线采集 |
| DPO（原始离线设定） | chosen/rejected 对与 Reference Logprob | 否 |
| PPO / GRPO 等在线 RL | 实际采样动作、Reward/Advantage 与策略概率 | 通常需要采样 |

传统 RLHF 流程可先做 SFT，再训练偏好 Reward Model，最后用 RL 在其评分上优化模型。InstructGPT 报告了这一类链路。RLVR 常用规则或环境验证，例如代码测试或答案判定；验证器本身也可能有盲区，Reward 变高不一定代表泛化改善。

## 偏好模型与 DPO

Bradley-Terry 偏好模型用 $\sigma(r_A-r_B)$ 表达 A 胜过 B 的概率。若打分差为 2，则胜率约 0.881，观察到 A 胜出的负对数似然约 0.127。该分数差描述相对偏好；绝对奖励尺度仍需单独定义。

DPO 将特定 KL-regularized reward 优化问题重参数化。令
$d(y)=\log\pi_\theta(y|x)-\log\pi_{\rm ref}(y|x)$，则典型偏好损失为

$$
-\log\sigma(\beta[d(y_w)-d(y_l)]).
$$

若 beta=0.1，d(chosen)=2，d(rejected)=−1，则 sigmoid 输入为 0.3，loss 约 0.554。DPO 不需要在原始训练阶段显式拟合独立 RM，也不需要本轮在线 rollout；在线偏好学习可以另行生成新的偏好数据。

## 反馈粒度与验证

Outcome Reward 针对最终结果，Process Reward 对部分推理步骤或中间动作评分。过程分数是否合理、是否改变原任务目标、怎样映射到 Token/Turn Advantage，需要另外定义。环境故障或超时也不应自动视作模型得零分。

工程上需要先固定目标任务和独立评测集，比较 Reward 与实际任务成功率、长度、失败模式，再决定使用哪种目标。OPD 则用 Teacher 的概率分布提供监督，详见 [OPD 与 KL](opd-kl.md)。这些目标可以组合，但应明确归一化与权重。

参考：[InstructGPT (2022)](https://arxiv.org/abs/2203.02155)、[DPO (2023)](https://arxiv.org/abs/2305.18290)、[PPO / GRPO](ppo-grpo.md)。
