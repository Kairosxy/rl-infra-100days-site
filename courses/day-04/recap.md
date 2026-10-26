# Day 04 · 为什么 PPO Clip 不是 KL 保证

固定 Old Policy 的动作概率为 0.2、Current 为 0.3，Ratio=$0.3/0.2=1.5$。Importance Weight 修正的是两份采样分布，不是学习率。

$$
J^{clip}=\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A).
$$

取 $\epsilon=0.2$：

- $A=+2,r=1.5$：$J=2.4$，右侧进入平台。
- $A=-2,r=0.6$：$J=-1.6$，左侧进入平台。
- $A=+2,r=0.6$：$J=1.2$，仍有局部梯度。

如果 $\pi_{\rm old}=[0.5,0.5]$、$\pi_{\rm current}=[0.9,0.1]$，即使正 Advantage 的 A 样本被 Clip，完整分布的 $D_{KL}(\pi_{\rm old}\|\pi_{\rm current})≈0.510826$ nats。**单样本局部平台不是全策略 KL 上界。**

检查 Tensor 时先锁定 Old/Current 的概率定义、Action ID、Mask 和 Reduction。不要把 Behavior、Old、Current 当作天然相同。

[在分段曲线上改变 Ratio 和 Advantage](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=4)。
