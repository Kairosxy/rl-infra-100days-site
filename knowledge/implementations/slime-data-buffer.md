# slime：为什么把 Data Buffer 放在 RL 中心？

slime 是面向 RL 训练的框架，公开实现强调 Megatron-LM 与 SGLang 的集成，让训练与高吞吐 Rollout 各自使用擅长的系统。Data Buffer 在这个思路中承接生成数据与 Trainer 消费之间的差异。

## 重要的不是队列 API

先问数据何时被视为可训练：请求完成并不等于 Group 完整；生成的数据也不保证一定会被过滤器接受。Data Buffer 的生命周期、版本与背压定义，会影响 GPU 是否因为无效样本空等，或因为过度补投形成大量在途余量。

对于 MoE，Megatron 与 SGLang 是不同计算路径，TP/EP、FP8 Recipe、Router Assignment 和 Logprob 语义不能仅凭模型权重名称认定一致。

## 研究框架特色的正确方式

把 slime 与 verl/AReaL 放在同一实验条件下对照：Rollout 与 Train 是否 Colocate、数据进入 Buffer 的条件是什么、允许的 Staleness 如何实现、何时切换权重、如何恢复中断。先比较**语义与资源成本**，再跑等质量的性能实验。

参考：[THUDM/slime](https://github.com/THUDM/slime)。本页没有锁定某个版本的源码调用链，配置与 API 应以具体 Commit 为准。