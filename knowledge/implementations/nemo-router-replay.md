# NeMo RL：Router Replay（R3）在哪层发挥作用

R3 的核心是把 MoE Rollout 使用过的离散 Expert Assignment 记录下来，并在训练侧重放，以减少两套执行引擎在同 Token 上走不同 Expert 的概率分布偏差。

## 设计要点

Router Replay 改变的是**路由选择一致性**，不应把它描述成冻结 Router 的全部梯度或强制两边 Logprob 完全相等。Router Meta 必须与 Token Position、Layer、Expert ID 编码正确对应，尤其要处理 Packed Sequence、并行切片、版本改变和旧轨迹恢复。

如果信息缺失，要明确 Fallback：跳过这条轨迹、使用当前路由，或拒绝整个组，都会改变算法与数据分布。不能把路由对齐率的提高直接当成 Reward 提高。

## 适用条件与代价

跨 Engine、跨精度且 Top-k Router 选择高度敏感时，重放离散路由有机会显著缩小误差。但它需要携带和存储额外路由元信息，可能改变训练侧 Router 的优化方式，并增加 Replay 的实现复杂度。

建议检查 Layer-wise Routing Agreement、Per-token Logprob Gap、Router Grad 与模型质量，而不是只看总 KL。

资料：[R3 论文](https://proceedings.mlr.press/v306/ma26aw.html) · [NeMo RL Router Replay Guide](https://docs.nvidia.com/nemo/rl/nightly/guides/router-replay.html)。