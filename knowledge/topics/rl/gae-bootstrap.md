# GAE：TD 误差、Bootstrap 与终止边界

GAE（Generalized Advantage Estimation）把多步 TD residual 按衰减系数组合，构造用于策略更新的 Advantage。它不是 Reward 的另一种命名，也不是 GRPO 的 Group 标准化。

## 两步轨迹的完整计算

记 $V_t$ 为当前 critic 对状态的估计，$b_t$ 表示下一状态是否可 Bootstrap，$c_t$ 表示后一位置是否是同一条连续轨迹中可用于递推的 residual。

$$
\delta_t=r_t+\gamma b_tV_{t+1}-V_t,\qquad
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

设两步奖励 [0,1]，Value=[0.5,0.6,0.8]，gamma=0.9，lambda=0.95。若第二步是外部时间限制导致的截断，且最后状态确实可以继续，则保留末状态 Value=0.8，但不跨出采集片段递推：

| 位置 | TD residual | Advantage |
|---:|---:|---:|
| 1（末步） | 1+0.9×0.8−0.6=1.12 | 1.12 |
| 0 | 0+0.9×0.6−0.5=0.04 | 0.04+0.855×1.12=0.9976 |

若末步是真正 Terminal，则末步不 Bootstrap，delta=0.4，A0=0.382。区别不是是否达到 max length 这个字段，而是任务定义中是否还有后续回报。

~~~python
def gae(rewards, values, bootstrap, continuation, gamma=.9, lam=.95):
    out = [0.] * len(rewards)
    next_adv = 0.
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * bootstrap[t] * values[t+1] - values[t]
        next_adv = delta + gamma * lam * continuation[t] * next_adv
        out[t] = next_adv
    return out

assert abs(gae([0,1], [.5,.6,.8], [1,1], [1,0])[0] - .9976) < 1e-10
assert abs(gae([0,1], [.5,.6,.8], [1,0], [1,0])[0] - .382) < 1e-10
~~~

这是教学计算，未假设任何框架字段名。Packed Batch 中尤其要区分 padding mask、bootstrap mask、trajectory continuation mask 与 actor loss mask；只用一个 done 通常表达不了所有边界。外部截断后若环境已经 reset，要读取截断前的最终 observation 对应的 Value，而不是新 episode 初始状态的 Value。

## gamma 与 lambda

gamma 改变回报中远期反馈的权重；lambda 改变 TD residual 的累积范围。在标准设定下 lambda=0 使用单步 residual，lambda=1 将连续段 residual 累积为多步 Return/Bootstrap 减 Value。Value 估计不准时，两者体现不同的偏差与方差权衡。

PPO 的 Actor 可用 GAE，Critic 可以拟合相应的 return target；GRPO 通常用组内相对奖励构造 Advantage，而不必为每个 Token 估计一个 Value。不能把这两种估计器的差别等同于 Ratio/Clip 的差别。

参考：[GAE 原论文](https://arxiv.org/abs/1506.02438)、[Gymnasium: Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)、[PPO / GRPO](ppo-grpo.md)。
