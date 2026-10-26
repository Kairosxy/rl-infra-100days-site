# 从一个可手算的 Policy，到真实 RL Infra 的正确性边界

课程保留一条有意缩小的参考线索：**数据从哪个分布生成、每一步梯度究竟更新了什么、哪些状态必须跨进程保留**。越到后面，并行、异步和服务拆分越复杂，但算法与数据的语义不能因此丢失。各阶段要有能够区分正确实现与错误实现的实验，而不只是多个接口能相互调用。

## 阶段一 · Day 01–14：明确优化目标

先固定一个二动作 Bernoulli Policy：`π(A)=p`，两动作 Reward 为 `[1,0]`。

1. 期望回报为 `J=p`，以 `p=sigmoid(z)` 参数化时 `dJ/dz=p(1-p)`。
2. 单次采样形成 Score Function 估计；Baseline 不随采样动作变化时，期望梯度不变，方差可能变化。
3. 把单动作推广成自回归 Token，保留准确的 Shift、Action Mask 和 Reduction 定义。
4. 固定旧策略以后引入 PPO Ratio/Clip；再把同一 Prompt 的多条回答转为 GRPO Advantage。DAPO/GSPO 的改动沿着具体失败条件讨论，不把算法名放在同一段总结里。

**可执行证明**：至少有一组精确期望梯度与采样估计对照；一组 Token Shift 故意错一位的反例；正/负 Advantage 的 Clip 区间变化；全等 Reward 和不完整 Group 的零信号/错误统计。Day 10 的 Toy System 只能证明最小机制，不能声称已经复现大模型 PPO/GRPO 的全部细节。

## 阶段二 · Day 15–26：把同一梯度变成真实的计算图

完成 Autograd 的梯度来源追踪、微批不等长的分母校验、AdamW/Clip/AMP 更新边界，以及 Activation/FlashAttention/FP8 的成本模型。

**工程证据**：有能计算的样本/Token Mean 对照；具体 Saved Tensor 的生命周期；量化造成的 Logprob 差异。旧仓库留下的教学实验仍可复用，但它们在新课表中的逻辑位置已变化：[不等长梯度累积](../examples/day_03/gradient_accumulation_lab.py) 属于新的 Day 16 主题，[朴素显存账本](../examples/day_02/memory_budget.py) 属于 Day 18/19 的附录案例，而不是优化目标的第一周。

## 阶段三 · Day 27–56：资源约束的两个系统

训练侧分析计算与数据移动：GEMM/Attention、DP、TP、PP、EP、CP、Checkpoint；推理侧分析 Prefill/Decode、KV Cache、Continuous Batching、推理分布与采样语义。

**工程证据**：指定模型形状、序列分布、GPU 拓扑和约束，估算 FLOPs、HBM/通信字节和资源峰值；只在明确的真实环境和 Benchmark 下报告 Step Time/TTFT/TPOT/P99。理论 Roofline 不能当作硬件实测。

## 阶段四 · Day 57–85：加入 Agent 与异步

Rollout 不再只是一次字符串生成：工具 Observation 不能随意进入 Policy Loss，Group 收齐与整个 Batch 返回是不同时间点，Policy Version 必须指向真实生成时的参数状态，Partial Rollout 的恢复可能跨权重边界。

重点研究三个能相互冲突的目标：**样本分布/算法正确性、GPU 利用效率、实现与恢复复杂度**。加入 DataSampler、队列背压、权重原子切换、Off-policy IS、OPD Teacher 分布和 Token KL。

**工程证据**：以一组标注了 `admitted/completed/accepted/inflight/drained` 的时间线，证明某种调度安排是否只是减少外层 Barrier，还是已经改动了采样分布；记录 Behavior/Old/Current Policy 的不同角色。没有实验测量时不能把“缩短等待”变成“端到端速度更快”的确定结论。

## 阶段五 · Day 86–100：故障、复现与面试

给同一系统注入超时、重复 Reward、Mask 错误、版本跳变、NCCL Hang、OOM 等故障，检验检测点、恢复路径和额外成本。最后才设计多 GPU RL 系统与面试演练。

**结果判据**：公式可复算，错误测试必现，代码与测试能对应，性能声明包含环境和基线，引用明确区分作者报告与本项目复现。没有真实个人项目或实验结果，不应该自动编造数字和经历。

## 交互作为假设检验工具

浏览器实验先覆盖 Day 01–05：手工控制 `p`/Baseline/Token Shift/PPO Ratio/Group Reward，界面显示**可计算的变量**和**对应的结果**，并故意暴露一类错误。后续互动内容可沿着 GAE 回溯、KV Cache 预算、异步时间线发展。不是每一天都必须有独立动画；如果实验不能改变判断，文字与代码更好。

## 数据模型与成熟度

随阶段逐步扩大轨迹契约。最初只是 `action, reward, logprob`；自回归后引入 `response_ids, prompt_length, valid_length, action_mask`；组内采样再引入 `group_id`；系统异步后再引入 `trajectory_id`、版本、在途状态与恢复点。字段不必在第一天全部讲完。随时区分讲义 Toy 结构和具体框架版本实际使用的 Schema。

本课程所有“收益”结论都按 **derivation / tested / measured / paper-reported** 明确标注。原始输入、失败复现和证据链比绘制漂亮的架构概览更重要。
