# vLLM：PagedAttention 为什么能改善 KV 内存使用

vLLM 的 PagedAttention 将 KV Cache 分为固定大小的 Block，并通过映射管理逻辑 Token 序列与物理 KV Block 的关系。这样可以减少连续分配与预留带来的浪费，并支持灵活的复用/管理方式。

它改变的是**如何分配和引用 KV 内存**，并没有降低每个 Token 的 K/V 向量的数学大小。长上下文、多并发仍有容量上限。

例如 32 Layer、8 KV Heads、HeadDim=128、BF16，理论上每 Token KV 约为 128 KiB；如果缓存 8K Token，单请求就约需 1 GiB（忽略额外管理开销）。请求数、长度分布和 Prefix Cache 命中率决定实际压力。

与 SGLang 的 RadixAttention 对照时，先分清 Paged KV、Prefix Cache、Continuous Batching 不是同一类优化。需用固定负载比较 TTFT/TPOT/Throughput/P99，而不只是查看一个排行榜。

资料：[PagedAttention](https://arxiv.org/abs/2309.06180) · [vLLM 文档](https://docs.vllm.ai/) 。