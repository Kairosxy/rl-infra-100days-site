# Day 01 · 一条回答如何安全地变成训练样本

RL Trainer 最怕的错误不一定会让程序崩溃。有一种更隐蔽的情况：模型生成了四条回答，Verifier 也返回四个 Reward，Tensor Shape 全对，但 Reward 被贴到了错误的回答上。Loss 正常下降，模型却在学习错误的反馈。

这一课沿着 **Rollout → Reward → Sampler → Learner** 追踪一条样本，重点解决三个问题：怎样不丢失身份；什么时候一个 Group 才算完整；策略版本该跟着谁走。暂时不推导 PPO 或 GRPO 的完整 Loss。

## Reward 比回答晚到，而且顺序不一样

假设 Prompt 是「17 × 19 = ?」。采样器要求每个 Prompt 生成四条回答，得到：

| Trajectory | 回答（简化） | 真实 Reward |
|---|---|---:|
| `t-A1` | `323` | 1 |
| `t-A2` | `324` | 0 |
| `t-A3` | `323` | 1 |
| `t-A4` | `321` | 0 |

Rollout 返回顺序为 A1、A2、A3、A4。Verifier 独立运行，结果可能按 **A4、A1、A3、A2** 返回，Reward 数组对应 `[0,1,1,0]`。如果程序直接做 `zip(trajectories, rewards)`，它会把 A1 错标为 0、A2 错标为 1。列表长度没有变化，平均 Reward 也仍然是 0.5，监控面板未必能发现。

正确做法是把评分结果当作 `(trajectory_id, reward)`，按稳定 ID 关联，而不是用完成次序推断身份：

~~~python
reward_by_id = {result.trajectory_id: result.reward for result in reward_results}
for trajectory in trajectories:
    trajectory.reward = reward_by_id[trajectory.trajectory_id]
~~~

这段伪代码省略了真正重要的拒收逻辑：检查重复 Reward ID、缺失 ID、未知 ID，以及非有限 Reward。它只解释关联方式；仓库中可直接运行的实现是 [`join_rewards_by_trajectory_id`](../../examples/day_01/trajectory_contract.py)，会明确拒绝这些异常。

在异步场景里尤其不能用列表位置关联：完成顺序、过滤顺序和 Trainer 最终消费顺序可能完全不同。

## Trajectory 要保存哪些事实

一条训练轨迹至少需要保留生成出的 Token、可训练 Token 的位置、生成策略版本，以及与 Prompt/Group 的身份关系。这里先用最小字段理解：

~~~text
trajectory_id  = "t-A1"      # 单条生成结果的稳定身份
group_id       = "g-17x19"   # 所属 Prompt Group
response_ids   = [11, 12, 13]
response_mask  = [1, 1, 1]   # 本例的响应位置均参与训练
reward         = 1.0
policy_version = 10         # 生成它时使用的 Policy
~~~

`trajectory_id` 防止评分错配；`group_id` 让 Sampler 知道哪些轨迹共享一组统计量；`response_mask` 决定哪些位置进入目标函数；`policy_version` 说明这条数据来自哪个生成策略。缺少其中任一语义，后续的 Advantage、Logprob 或 Staleness 检查就容易靠猜测补齐。

**字段同名不代表各框架的契约相同。** 这里的 `policy_version` 是教学字段；实际框架还可能区分 `behavior_policy_version`、checkpoint ID 和采样参数版本。`response_mask` 也未必等于训练使用的 `loss_mask`：EOS、Padding、工具观察和多模态 Token 是否参与 Loss，要按具体算法定义。

当前 CPU 示例能检查 Mask 的长度和 0/1 取值，但没有验证 Token 语义、EOS 归属或跨进程唯一 ID。不要把它当成完整的生产数据契约。

## 为什么 Group 收齐和 Batch 收齐不是同一种等待

继续用四条回答。假设后续使用一个标准化的组内 Reward 估计：

$$
\bar r_g=\frac{1}{G}\sum_{i=1}^{G}r_i,\qquad
\sigma_g=\sqrt{\frac{1}{G}\sum_i(r_i-\bar r_g)^2},
\qquad
A_i=\frac{r_i-\bar r_g}{\sigma_g+\epsilon}.
$$

这里仅借一个常见的 GRPO 式组内统计说明依赖关系，暂时忽略极小的 $\epsilon$；不同算法实现可以选不同归一化约定。

完整四条 Reward 为 `[1,0,1,0]`，均值 $\bar r=0.5$，总体标准差 $\sigma=0.5$，所以：

$$
A=[1,-1,1,-1].
$$

如果第四条还没完成，拿前三条 `[1,0,1]` 先算，均值变成 $2/3$，标准差约 $0.4714$，得到：

$$
A_{\text{前3条}}\approx[0.707,-1.414,0.707].
$$

早算并不是「同一算法更快得到答案」，而是改变了统计对象。若要求固定四样本 Group，就必须等待、补投到有效组大小，或明确修改算法定义。

再假设同一个外层 Batch 有 Group A 和 Group B。A 的四条全部完成，B 还有一条长尾请求。这时 **A 的组内统计已经具备输入条件**，但 Trainer 能否立即更新，仍取决于 mini-batch/梯度累积、版本一致性和共享 GPU 资源约束。Group 的统计依赖是算法条件；整批都返回后才统一筛选，则可能只是实现的 Batch Barrier。两者不能混为一谈。

这也是后面学习 Async DataSampler、Partial Rollout 时始终要保留的区分。

## 版本号记录的是生成时的事实

设 Rollout 使用权重 v10 生成了 `t-A1`。等待 Reward 的过程中，Trainer 更新至 v11。数据从队列进入 Trainer 时，应仍然能看到它由 **v10** 生成，而不是把「消费时的当前版本 v11」覆盖到轨迹上。

至少需要分清四个角色：

- **Behavior**：实际采样动作的策略。
- **Old**：PPO 更新中用于 Ratio 分母、在当前优化阶段固定的策略。
- **Current**：正在反向传播的策略。
- **Reference**：用于 KL 等约束的参考策略。

同一个时刻它们可能共享权重，但不能靠这个偶然相等把语义合并。仅仅「数据异步到达」也不能直接断言算法已经 off-policy；要看生成分布、训练目标和所用的概率。

本课提供的 [`validate_trajectories`](../../examples/day_01/trajectory_contract.py) 默认拒绝同一 Group 混合版本。它还有一个显式开关允许混版，**这个开关只跳过检查，不会自动提供 off-policy correction**。

## 跑一次「不会报 Shape 错误」的失败实验

从仓库根目录运行：

~~~bash
python examples/day_01/trajectory_contract.py
python -m unittest discover -s examples/day_01 -p 'test_*.py' -v
~~~

示例脚本应输出 `Validated 8 trajectories across 2 groups.`，测试则分别检查：

1. 把 Reward 返回顺序倒过来，ID Join 的输出不变。
2. 删除一个 Reward 或复制一个 ID，会抛出 `ValueError`。
3. 只生成三条或混入不同 `policy_version`，默认 Group 校验失败。

再做一个不需要改代码的判断：**如果错误实现把 A1 和 A2 的 Reward 交换，平均 Reward 会不会变化？** 不会；但被强化的动作会变。这个反例说明，监控 `reward_mean` 不能替代样本身份校验。

四条轨迹之后进入 Learner，还需要可微的 Current Logprob 才能更新参数。Day 04 会解释这一步。现在只要求确认：同一条动作的 Token、Reward、Mask、策略版本必须保持关联，调度快慢不能改变它们的归属。

## 来源与实现边界

- [课程实验：Trajectory 数据契约及 Reward ID Join](../../examples/day_01/trajectory_contract.py)；[对应负例测试](../../examples/day_01/test_trajectory_contract.py)。以上代码能够直接核对本课的校验行为。
- [仓库数据契约](../../architecture/data_contract.md) 与 [Policy Version 语义](../../architecture/versioning.md)：课程定义，不等于某个外部框架的原生 Schema。
- [HybridFlow / verl 论文](https://arxiv.org/abs/2409.19256)：理解 RLHF 不同计算阶段的组织方式；本文未声称验证其最新内部调用路径。
- [PPO 原论文](https://arxiv.org/abs/1707.06347)：Old/Current 与 Ratio 的算法背景，Day 06 再推导。
