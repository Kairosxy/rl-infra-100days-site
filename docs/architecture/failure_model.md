# 故障模型与恢复边界

| 故障 | 可能后果 | 检测方式 | 处理方式 |
|---|---|---|---|
| Duplicate trajectory | 同一结果重复进入训练 | 唯一 ID / consumed 状态 | 幂等拒收 |
| Reward-ID mismatch | 错误动作被强化 | ID Join 与乱序测试 | 拒绝批次并记录原因 |
| Incomplete Group | Group Advantage 不完整 | 每组计数 | 等待、补投或显式改变算法 |
| Mask length mismatch | 无效 Token 进入 Loss | Shape 断言 | 拒收轨迹 |
| Non-finite reward/logprob | Loss 或 Gradient 变成 NaN | isfinite 检查 | 隔离样本并保留诊断 |
| Policy version mismatch | 概率语义不清或样本过旧 | Version metadata | 按配置拒收或修正 |
| Partial checkpoint write | 恢复状态不一致 | manifest / checksum | 原子发布与验证 |
| Worker crash | 轨迹悬挂或重复完成 | lease / heartbeat / timeout | 明确失败状态，幂等重试 |

每个失败案例记录：现象、最小复现、根因、错误修复、正确修复、测试、监控和面试表达。不要只写“增加重试”或“加大超时”，要说明重试的幂等边界。
