# Ray Task、Actor、ObjectRef 与队列背压

RL 系统中的控制消息、远程执行与数据搬运并不是一件事。Ray 的 Task 是可调度的远程函数，Actor 是持有状态的远程 Worker，ObjectRef 是远程对象或结果的引用。Placement Group 为一组执行角色预留逻辑资源，不能据此自动推断 GPU 间拓扑或 Tensor 并行组正确。

## 提交与读取

~~~python
# 示例：限制在途任务数量，而非限制 GPU kernel 并发。
pending = []
for item in items:
    pending.append(worker.run.remote(item))
    if len(pending) >= max_pending:
        ready, pending = ray.wait(pending, num_returns=1)
        consume(ray.get(ready))
while pending:
    ready, pending = ray.wait(pending, num_returns=1)
    consume(ray.get(ready))
~~~

远程调用返回 ObjectRef 时任务未必完成。ray.wait 可用于先处理就绪结果，避免按提交顺序等待整批；ray.get 获取对象或暴露任务异常。对象可能需要跨节点传输或从 spill 恢复，取得结果也不自动意味着 GPU payload 已经到位。这是 Ray Core 的常见模型，实际序列化、零拷贝和传输能力以版本、对象类型与部署为准。

## 数据面与队列预算

假设 Rollout 每秒产生 20 条完成生成的轨迹，Reward 服务最多每秒处理 12 条。如果长期以这个速率提交，平均积压会以 8 条/s 增长。即使生成 GPU 利用率很高，系统仍无法稳定向 Learner 提供 20 条/s。若每条载荷 2 MiB，1 分钟额外积压约 960 MiB，还未计复制和对象元数据。

稳定且长期平均下，Little's Law 为 $L=\lambda W$。若有效吞吐 12 条/s、平均在系统停留 5s，平均在途量约 60。它不是瞬时峰值公式，更不能直接保证某个尾延迟。需要分别限制生成、待评分、待消费及载荷字节预算。

Metadata ready、payload ready 与 consumed 是不同事件；控制消息成功不能替代校验传输结果。对于共享内存、对象存储、RDMA 或 TransferQueue 等方案，分别定义所有权、引用/拷贝、字段发布、超时、幂等和回收。超时表示调用方没在期限内获得确认，不保证远端没有执行写入。

## 最小验证

人为让评分比生成慢，记录 ready queue 长度、pending refs、对象存储占用、每阶段服务速率和真实有效训练样本吞吐；对比固定并发与有界在途队列。检查取消或重试是否重复产生 sample_id。此实验可以 CPU 模拟，不需要先把整个训练框架改成异步。

参考：[Ray Key Concepts](https://docs.ray.io/en/latest/ray-core/key-concepts.html)、[Ray Objects](https://docs.ray.io/en/latest/ray-core/objects.html)、[限制 pending tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html)。继续阅读 [Async Rollout](async-rollout.md)。
