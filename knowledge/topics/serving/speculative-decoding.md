# 投机解码：Draft、Verify 与技术分支

投机解码要减少昂贵的 Target Model 串行 Forward 次数。Draft 先提出多个候选 Token，Target 在候选前缀上计算概率并验证。加速来自少做串行 Target 调用，而不是跳过概率正确性。

## 单位置精确接受与拒绝

Target 分布 p=[0.8,0.2]，Draft 分布 q=[0.6,0.4]。标准随机投机采样从 q 提议 x，以 $a(x)=\min(1,p(x)/q(x))$ 接受。提议第一个 Token 时全部接受，贡献 0.6；第二个 Token 以 0.5 概率接受，贡献 0.2。拒绝概率为 0.2，拒绝后按正残差 $\max(p-q,0)$ 归一化采样，即补到第一个 Token。最终输出概率仍为 [0.8,0.2]。

$$
\Pr(\text{accept})=\sum_x\min(p(x),q(x)).
$$

多位置验证必须基于已经接受的前缀。第一次拒绝后的候选不再有效；标准 bonus-token 规则下，全接受后还可提交一个 Target Token。贪心或近似验证需单独说明正确性条件。接受率与 KL 的区别见 [Logprob 契约](../learner/logprob-contract.md)。

## Draft 的来源

| 技术分支 | 共同问题上的不同选择 |
|---|---|
| 独立小模型 | 用更便宜的自回归模型提议序列 |
| N-gram / Suffix | 利用历史重复片段，避免训练额外模型 |
| Medusa / 原生 MTP | 利用 Target 的表示或模型自带的多 Token 预测模块；具体递推和验证方式不同 |
| EAGLE 家族 | 使用 Target 隐藏特征辅助 Draft，提高候选质量；版本间特征与候选组织不同 |
| DFlash 等并行 Block Draft | 一次 Draft Forward 提出整个 Block，减少 Draft 自身的串行深度 |
| Dynamic Draft Length / Adaptive Verification | 根据候选置信度、负载和验证代价调整运行时策略，可与多个家族组合 |

MTP 也可作为模型训练目标，DeepSeek-V3 报告即有这一机制；只有接入 Draft–Verify 路径时，它才成为投机解码实现。DFlash 使用条件 Block 预测，不代表所有投机解码都是扩散模型。Speculators 的 v0.8.1 文档还列出 DFlash2、DSpark 等分支；具体训练和 Serving 版本支持需继续核验，不能依据名称直接推断性能。

## 接受长度与实际成本

假设一轮有三个 Draft Token，每个已接受前缀之后的条件接受概率都是 0.8，采用标准 bonus-token 规则。一轮提交 Token 的期望数为 1+0.8+0.64+0.512=2.952。若普通 Decode 每 Token 耗时 1 单位，整轮 Draft+Verify 耗时 2.2，则理想速度比约 1.34；若耗时 3.2，则约 0.92。均为教学推导。

实际评估要统计每轮提交长度、Draft/Verify 时间、并发、KV 生命周期、TTFT/TPOT 与有效 Token/s。在 RL 中，精确验证的 Behavior 概率对应最终采样的 Target 分布及实际采样变换，不应直接使用 Draft Logprob。Target 更新后 Draft 过旧可能损害接受率；是否导致策略偏差仍取决于验证是否精确。

参考：[经典投机采样](https://arxiv.org/abs/2211.17192)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[Speculators v0.8.1](https://docs.vllm.ai/projects/speculators/en/v0.8.1/user_guide/tutorials/train/)、[DFlash 实现说明](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)。
