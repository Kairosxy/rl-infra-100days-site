# Day 03 · 两个能检验理解的问题

## 题 1：为什么一个 Response 只拿到一个分数，却有两个 Token 的梯度？

已采样 Response 包含两个动作，概率依次为 0.8、0.7，终局 Reward=1。写出这条 Response 的联合概率、对数概率，以及基础 REINFORCE 的 Token Sum Loss。指出梯度实际穿过哪里。

**答案依据：** 联合概率 0.56，Logprob≈−0.579818，Loss≈0.579818；Reward 是梯度系数，导数来自两个条件 Logprob 的计算图。不能声称它证明两个 Token 的“真实贡献”相同。

**追问：** 若环境还提供了中间 Reward，或使用每一步的 Return，这条 Loss 会怎样改变？

## 题 2：为什么一个 Token Shift 错误不一定触发 Shape 报错？

给定完整 `input_ids=[BOS,问题,y1,y2,PAD]`，Prompt 长度 2、有效长度 4。写出正确 Action 目标和预测者的位置；再构造错一位的预测者切片。说明如何用一个负例而非平均 Loss 检查这个错误。

**答案依据：** 目标位置 2、3，预测者位置 1、2；Mask 在目标位置 1..4 上为 [0,1,1,0]。错误切片拿 2、3 预测 2、3，Shape 不变但概率语义错误。

**追问：** 两条 Response 长度 2 和 4，全部有效 Token 的单项 Loss 都为 1。比较全局 Token Mean 与逐 Sequence Token Mean 的每 Token 权重。
