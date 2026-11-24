# Serving 请求增多，KV Cache 为什么先 OOM？

模型参数占用没有增加，并发一高却 OOM，通常需要单独核算**增长的 KV Cache**，而不是只看模型权重。

在常见 Attention KV 缓存定义下：

$$
M_{\mathrm{KV}}=2L_{\mathrm{layers}}N_{\mathrm{kvheads}}D_{\mathrm{head}}T_{\mathrm{cached}}b.
$$

例如 32 层、8 个 KV Head、HeadDim 128、BF16，每 Token 需要 128 KiB。一个 8192 Token 请求约占 1 GiB；16 个这样的请求已经需要约 16 GiB KV（忽略分配碎片、额外池和架构差异）。

## 三类问题不要混

- **容量耗尽**：总活跃 Token 数确实超过了静态 KV Pool 可支持的预算。
- **内存管理浪费**：由于分配粒度、碎片、预留策略导致可用容量低于理论上限。
- **调度放大**：请求 Admission 太激进，Prefill/Decode 混排与缓存驱逐造成瞬时峰值。

Paged KV 的作用是改进块级管理，不创造不存在的 GPU HBM。Prefix Cache 能复用公共前缀，但命中条件、缓存生命周期和驱逐策略决定真实收益。PD 解耦又增加 KV 传输的成本。

## 必须验证

至少记录请求长度分布、同时活跃 KV Token、KV Pool 使用率、缓存命中率、Eviction、TTFT/TPOT/P99。比较 SGLang 与 vLLM 时固定模型、Attention 结构、输入输出长度、Sampling 参数、硬件和缓存预算。

参考：[PagedAttention](https://arxiv.org/abs/2309.06180) · [SGLang HiCache](https://docs.sglang.ai/advanced_features/hicache_design.html)。