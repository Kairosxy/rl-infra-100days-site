# GPU 共驻留、时间复用与资源隔离

一张 80 GiB GPU 上，Learner 的活跃状态占 44 GiB，Rollout 的权重和 KV 占 30 GiB。如果两个进程还各需要 5 GiB 临时工作区，峰值可能达到 84 GiB。即使它们的 GEMM 不同时运行，常驻状态的总量也可能超出容量。讨论共卡前，先区分状态能否同时驻留、计算能否同时执行。

## 共享的粒度

| 方式 | 执行与资源边界 |
|---|---|
| 时间复用 | 轮流使用设备；若两侧状态无法共存，还要迁移或释放状态。 |
| 同进程多 Stream | CUDA Event 表达依赖，独立 Kernel 有机会并发，共享设备资源。 |
| CUDA MPS | 支持多进程工作并发；可配置计算资源限制，仍需处理显存、带宽与进程生命周期。 |
| Green Context | 在支持的 CUDA 环境中为工作配置特定的 SM 资源集合。 |
| MIG | 支持的 GPU 可切出独立实例，分配计算和内存系统资源；可用实例规格由设备决定。 |
| 独立 GPU 池 | 训练与生成各有设备预算，跨池传输权重和数据。 |

MPS 的 active thread percentage 是执行资源上限，不能直接解释为固定 HBM 带宽比例或某个请求延迟上限。MIG 的隔离更强，但划分后每个实例的容量也相应变小。Green Context、MPS 的具体能力与 CUDA 版本、GPU 架构和部署配置有关。

## 共驻留的成本

假设一轮训练占 12 ms，Decode 占 8 ms；两段无依赖，且独占运行时的耗时保持不变。串行为 20 ms，完全重叠的理想下界为 12 ms。如果共享 HBM 后两侧分别变为 16 ms 和 14 ms，那么即使完全重叠也要 16 ms。对在线生成，还须检查 14 ms 的服务时间是否已超过 TBT 预算。

这三个数字分别来自假定的独占耗时和共驻留耗时。真实收益取决于重叠程度、资源争用及迁移代价。训练期间暂停生成、卸载 Optimizer 或 KV，与两侧一直驻留并发，是不同的生命周期策略。

在 RL 中，Optimizer commit、权重安装和 Rollout 准入仍需独立管理。并行执行不意味着生成端已经看到新权重；参见 [策略发布](../async/weight-publication.md)。CTRL 等 Co-residency 工作需要在具体资源管理和发布机制下单独理解，不能从 MPS 的通用能力推出论文效果。

## 对照实验

固定同一模型、训练 Token 数、生成请求和设备预算，比较独占、时间复用及有界共驻留。记录每侧峰值 allocated/reserved、工作区、GPU 时间线、训练更新时长、TTFT/TBT 和有效样本吞吐。先验证显存上限及策略版本正确，再决定允许哪些阶段重叠。

参考：[NVIDIA MPS](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html)、[CUDA Green Contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)、[MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/introduction.html)。
