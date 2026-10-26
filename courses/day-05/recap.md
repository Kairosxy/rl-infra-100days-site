# Day 05 · 复算卡

同 Prompt 下 $G=4$ 条 Reward `[1,0,1,0]`，总体均值 0.5、标准差 0.5，组内标准化后 Advantage 为 `[+1,-1,+1,-1]`。

**两个最容易改错的问题：**

1. 全部 Reward 相同（全对或全错），减去组内均值以后每个分子为 0，带正常数 $\epsilon$ 的标准化 Advantage 全为零。这个组对该 PG 项没有区分度。
2. 只收齐 `[1,0,1]` 就算，均值变 $2/3$，标准差约 0.4714，Advantage 为 `[0.707,-1.414,0.707]`。这是三样本组统计，不是原定四样本组的「提前版」。

更深一层：组内 Reward 确定 Advantage，但**Token/Sequence Reduction** 仍决定长短 Response 的梯度权重。动态过滤零方差 Group 会提高某些采样效率，却也会改变被选中的 Prompt 分布。

[修改四个 Reward，观察统计量是否改变](https://kairosxy.github.io/rl-infra-100days-site/labs/?day=5)。
