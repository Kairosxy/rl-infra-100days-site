# Policy Version 与数据生命周期

- **Behavior Policy**：实际生成轨迹时使用的策略。
- **Old Policy**：算法更新中固定使用的比较基准，例如 PPO 分母策略。
- **Current Policy**：正在计算梯度并更新的策略。
- **Reference Policy**：用于 KL 或偏好目标的参考模型，不等于 Old Policy。

这些角色在同步训练的某些阶段可能对应相同权重，但语义不同，不能仅凭数值接近就合并概念。

每条轨迹至少记录 behavior_policy_version。如使用 checkpoint ID、tokenizer revision、sampling config revision，也应保存。训练批次应记录本次 old_policy_version，不能在训练后猜测。

最小检查：生成时记录版本；构造 Group 时验证版本策略；计算 Old Logprob 时记录权重快照；Current Logprob 由当前可微模型计算；异步批次记录版本滞后和拒收原因；权重发布使用不可变版本 ID，避免读到一半更新的权重。

待实现：版本单调递增与 checkpoint 映射、过期版本拒收策略、同步/异步 Logprob 对照实验。
