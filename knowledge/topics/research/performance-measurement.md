# 性能测量、关键路径与资源效率

按优化目标选择指标：延迟、端到端有效吞吐、资源成本，或相同预算下的模型质量。单个 Kernel 变快只证明一个局部事件改变。

## 从时间轴而不是 Util 开始

假设同步 RL 的 Rollout 30s、Reward 10s、Learner 20s，彼此无重叠，整步 60s。若只把 Rollout 加速 2 倍，其他不变，整步变成 45s，理论加速 60/45=1.33。若 Reward 与 Rollout 流水重叠，则要按真实依赖重画时间轴，不能仍使用阶段时间之和。

CPU 发起调用、GPU 实际执行、结果可用以及消费完成是不同时间。用 NVTX 标记任务、group、权重版本和逻辑 Step，在 Nsight Systems / Profiler 中关联 CPU、CUDA、NCCL 与服务事件。跨节点时钟偏差会影响绝对时间对齐；优先使用同进程时长和明确的 trace correlation，必要时校准时钟。

## 吞吐的分母

假设 8 GPU 持续分配 60s，生成 120000 Token，经过过滤和截断后实际用于更新 90000 个有效 Action Token。则生成吞吐为 2000 Token/s，有效训练样本吞吐为 1500 Token/s；资源时间是 480 GPU-s，有效 Action Token/GPU-s=187.5。若另一个方案生成更多无效样本，不能只用生成吞吐比较。共享和分离部署都要计入实际保留或消耗的全部 GPU-time，并明确利用率与分配时间口径。

统计 TPOT、TTFT、生成时长、p95/p99 时需写明样本单位、时间窗口及百分位算法。8 条样本的 p99 很不稳定；至少记录数量、分布和最慢样本，不能把平均时间与尾延迟混比。

## 最小可复现实验

固定模型、精度、输入/输出长度分布、有效 Batch、采样配置与资源数量。对一次只改变一个系统策略的 A/B，记录 warmup、重复次数、Step 时长分解、有效样本数、拒绝/失败比例、显存峰值和质量指标。异步或新算法若改变数据分布与更新频率，还需按有效 Token、GPU-hour 和达到同等质量的成本多口径比较；不应通过偷偷改变样本预算获得看似更好的结果。

理论下界来自 [GPU 算术强度](../compute/gpu-execution-memory.md) 和 [通信](../compute/distributed-collectives.md)，真实时间线解释偏离下界的原因；[端到端吞吐](../optimization/rl-throughput.md) 展开 RL 长尾分析。

参考：[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)、[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)。
