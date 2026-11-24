# Failure Cases

每个阶段至少沉淀一个可复现失败案例，记录能证明根因的最小实验。

~~~text
现象：
最小复现：
预期结果：
实际结果：
根因：
错误修复及其问题：
正确修复：
回归测试：
监控指标：
面试表达：
状态：designed / implemented / measured
~~~

优先案例：Reward 与 Trajectory 错配、Response Mask 偏移、stale Logprob、梯度累积归一化错误、Checkpoint 状态不一致、异步样本重复消费。
