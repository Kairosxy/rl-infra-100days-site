# KV Cache 不只是显存预算，也是一个调度问题

Serving 同时面对两个不同的阶段：Prefill 为输入序列计算表示并建立 KV；Decode 为每个请求逐步生成新的 Token，反复读取已有 KV。它们对算术强度、Batch Shape、带宽和延迟的敏感性不同。

最基础的 KV 容量模型：

$$
M_{\mathrm{KV}}=2\times L_{\mathrm{layers}}\times N_{\mathrm{kvheads}}\times D_{\mathrm{head}}
\times T_{\mathrm{cached}}\times b.
$$

其中 2 表示 Key 和 Value，$b$ 是每个元素的 Byte 数。公式描述一组连续存储的 KV 字节数，不包含页表、内存池碎片或分层缓存开销。

举例：32 层、8 个 KV Head、HeadDim 128、BF16，每缓存 1 个 Token 约需 $2×32×8×128×2=131072$ Byte，即 128 KiB；8K Token 约为 1 GiB。具体模型仍需按照真实 GQA/MQA、Layer 类型及缓存 dtype 重新计算。

PagedAttention 通过分页管理 KV，减少分配粒度和碎片造成的浪费；Prefix Cache 复用相同前缀，降低重复 Prefill；Continuous Batching 改变请求交错执行方式。它们解决的是不同约束，不是三种同义的加速按钮。

SGLang 的 RadixAttention/HiCache 与 vLLM 的 PagedAttention 能作为代码级对照。需要同时记录吞吐、TTFT、TPOT 和 P99，并固定请求长度与到达分布。

资料：[vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)；[SGLang HiCache 设计](https://docs.sglang.ai/advanced_features/hicache_design.html)。