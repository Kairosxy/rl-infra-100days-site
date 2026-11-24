# 连续动作与 Diffusion / Flow 策略的后训练

文本策略通常对离散词表定义概率。机器人控制可能输出实数向量，Diffusion/Flow 生成器还可能通过若干中间状态形成一个动作块。能否采用 PPO 一类目标，取决于实际采样过程能否给出与训练一致的策略密度及 ratio。

## 连续策略的概率对象

先看一维 Gaussian 策略 $a\sim\mathcal N(\mu,\sigma^2)$。对固定动作 a，log density 为

$$
\log p(a)=-\frac{(a-\mu)^2}{2\sigma^2}-\log\sigma-\frac12\log(2\pi).
$$

假设旧策略为均值 0、标准差 1，新策略均值 0.5、标准差 1，已采样动作 a=1。新旧 log density 差为 0.375，ratio 为 $e^{0.375}\approx1.455$。这里比较的是同一动作的密度，不是实数点的离散概率。若策略通过 tanh 等变换满足控制范围，必须处理变换后的密度；不能把执行端裁剪后的动作直接套回原始高斯密度。

## 从演示到生成轨迹

Behavior Cloning 用演示动作监督模型，可采用回归或分布学习。Offline RL 利用已采集的转移数据估计策略，需关注数据支持和分布外动作；Online RL 通过真实或模拟环境反馈继续更新。SAC 等方法为连续随机策略设计价值与熵目标，不需要离散词表。

Diffusion Policy 与 Flow Policy 可生成具有多个时间步的连续 Action Chunk。训练的噪声或速度预测误差只说明模型如何拟合该生成过程。若要直接优化外部 Reward，可另行定义概率模型或优化目标。DDPO 将随机去噪转移建模为多步决策，通过其转移概率构造策略梯度。确定性 ODE 求解并不天然具有同样的离散转移 logprob 契约，Flow 相关 RL 方法需要独立推导。

## 两套时间尺度

假设机器人执行 [8,7] 的动作块，生成器内部用了 10 次网络评估。外部环境可能执行其中 4 步就重新观测；内部 10 步生成过程与外部 4 步控制转移不是同一条 MDP 轨迹。环境 Reward、动作执行 mask、去噪时刻、随机噪声及概率记录要按所选目标分别定义。

验证时固定初始状态和随机源，先对比单次动作生成的条件概率或目标值，再比较一段完整控制轨迹。统计外部任务成功率、控制频率、每次网络评估成本以及有效训练样本；不要用文本 Token/s 直接解释具身任务的采样效率。

参考：[Training Diffusion Models with Reinforcement Learning / DDPO](https://arxiv.org/abs/2305.13301)、[Soft Actor-Critic](https://arxiv.org/abs/1801.01290)、[pi_0](https://arxiv.org/abs/2410.24164)、[生成模型基础](../model/generative-models.md)。
