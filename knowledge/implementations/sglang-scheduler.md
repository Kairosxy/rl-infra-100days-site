# SGLang：RadixAttention、调度与 KV 的边界

SGLang 的重要设计包括前缀缓存 RadixAttention、请求调度和高吞吐生成。它们解决不同的问题：Prefix Cache 复用共有前缀的 KV；Scheduler 决定请求何时执行 Prefill/Decode；KV 内存管理决定可用容量和驱逐行为。

## 什么时候 Prefix Cache 有用

一批请求具有很长的相同 System Prompt 或 Agent History 时，共享前缀有机会减少重复 Prefill。但只要 Token 前缀不同，缓存命中可能下降；缓存空间和 KV 的占用、驱逐策略也会反过来影响实际并发。

训练场景里，如果 Sampling Temperature/Top-p、MoE Router、FP8 Recipe 与 Learner 不一致，服务侧的较好吞吐不能自动等价于合法 RL 数据供给。

## 如何比较实现

与 vLLM/TensorRT-LLM 对照时，应固定 Model/Tokenizer、输入输出长度分布、请求到达率、TP/EP、KV Budget 与 Sampling 参数。至少同时报告 TTFT、TPOT、Throughput 和 P99，关注 Queueing 与 Prefix Cache Hit Rate 的变化。

资料：[SGLang GitHub](https://github.com/sgl-project/sglang) · [SGLang Documentation](https://docs.sglang.ai/) · [RadixAttention 论文背景](https://arxiv.org/abs/2312.07104)。