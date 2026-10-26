# Day 01 · 复算卡

**不看讲义回答：** 为什么 SFT 能从已知答案直接更新，而 RL 需要考虑模型是否采样到了有 Reward 的动作？

二动作 Bandit，`π(A)=p=σ(z)`、`R(A)=1,R(B)=0`。

$$
J=p,\quad \frac{\mathrm dJ}{\mathrm dz}=p(1-p).
$$

固定 $p=0.25$：期望 Reward 0.25，精确梯度 0.1875；四次独立采样至少一次正确的概率为 $1-0.75^4\approx0.6836$，但 Pass@1 仍只有 0.25。**评估预算不同的 Pass@k 不能当作同一个指标。**

自己改变奖励为 $R(A)=2,R(B)=-1$。需要从公式得出 $J=3p-1$、$\mathrm dJ/\mathrm dz=3p(1-p)$，而不是把 Reward 只当作一个固定标签。

[在线拨动 $p$，对照经验采样和理论值](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=1)。
