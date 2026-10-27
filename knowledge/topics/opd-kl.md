# OPD：Teacher 的分布怎样进入 Student 的梯度？

强化学习通常利用采样动作的 Reward 构造 Policy Gradient；Online Policy Distillation（OPD）则让 Teacher 提供一份更密集的概率监督。两种信号可以互补，但不能在不明确 Reduction、Tokenizer 和方向的情况下简单相加。

以两个 Action 为例，Teacher 分布为 $q=[0.9,0.1]$，Student 为 $p=[0.5,0.5]$。若选择 Forward KL $D_{\mathrm{KL}}(q\|p)$：

$$
D_{\mathrm{KL}}(q\|p)=0.9\log\frac{0.9}{0.5}+0.1\log\frac{0.1}{0.5}\approx0.3681.
$$

如果反过来计算 $D_{\mathrm{KL}}(p\|q)$，得到约 0.5108，数值和梯度都不同。对于多峰分布，Forward/Reverse KL 的偏好也不同。必须先写清 Teacher/Student 谁是期望分布、Loss 怎样在 Vocab/Token/Batch 维度归约。

工程上，Full-Vocab Logits 的 Tensor Shape 是 §[B,T,V]§，Teacher 可以独立服务或在训练 GPU 上 Offload；前者增加 RPC 和带宽代价，后者带来显存/资源争用。以 $B=4,T=1024,V=128000$、BF16 为例，单份 Logits 约 1 GiB，绝不能把 Teacher Logits 视为零成本数据。

与 GRPO 联合训练时，建议分别监控 PG Loss、Distillation Loss 和它们各自的 Gradient Norm/有效 Token Mask，辨认是否出现不同目标之间的尺度冲突。

参考：[PPO](https://arxiv.org/abs/1707.06347) 的策略更新背景；[NeMo RL 官方代码](https://github.com/NVIDIA-NeMo/RL) 中的 RL/OPD 功能文档需随实际版本核对。