# MoE 训推选中了不同 Expert，怎样修？

对于路由分数非常接近的两个 Expert，较小的精度扰动即可改变 Top-k 排序。一个简单例子：

- Rollout Router Score：E3=0.5001、E5=0.5000 → 选 E3。
- Training Router Score：E3=0.4999、E5=0.5000 → 选 E5。

这只是展示离散边界的**构造例子**，不是任何特定模型的 Router 实测值。

## 需要先确认

核对同 Token 的路由是否处于同一层、同一位置、同一个权重版本与相同的 Expert ID 编码；再统计不同 Layer 的 Top-k 重合率和 Logprob Gap 的相关性。

如果采样 Temperature 不同或 Tokenizer 对不齐，即使路由完全相同，概率也可能不同。Router 不是所有 Drift 的解释。

## 解决思路

R3（Rollout Routing Replay）将生成时选中的 Expert Index 传给训练侧并重放，从而对齐离散计算路径。但它增加了路由元信息、打包/拆包、与序列位置对齐以及恢复时正确继承路由状态的复杂度。

路由重放还可能改变 Router 的优化条件，因此要监控 Router Grad、Expert Load Balance，以及不同版本下的质量。另一条研究思路是根据路由分歧程度决定是否接受样本，但这也可能造成数据选择偏差。

**结论不应是“R3 更好”或“过滤更好”**，而是比较两者各自修复了哪类差异、丢掉了哪些数据和训练信号。

资料：[R3](https://proceedings.mlr.press/v306/ma26aw.html) · [NeMo RL Router Replay 指南](https://docs.nvidia.com/nemo/rl/nightly/guides/router-replay.html)。