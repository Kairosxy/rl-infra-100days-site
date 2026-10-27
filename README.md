# RL Infra

一份面向 RL / LLM 训练工程的公开技术知识库：基础知识、实际工程问题和主流框架实现相互关联。知识可以按主题阅读，也可以从性能故障、FP8 数值差异、MoE Routing、异步训练等具体问题进入。

**[知识库首页](https://kairosxy.github.io/rl-infra-100days-site/)** · [工程问题](https://kairosxy.github.io/rl-infra-100days-site/#category=case) · [框架实现](https://kairosxy.github.io/rl-infra-100days-site/#category=implementation) · [快速复习题](https://kairosxy.github.io/rl-infra-100days-site/quiz/)

简短判断和口头表达用来帮助恢复记忆，深入研究请读完整推导、源码和案例。

## 内容组织

- `knowledge/topics/` — 基础机制和数学/计算模型
- `knowledge/cases/` — 真实现象、排查顺序、方案取舍和验证方式
- `knowledge/implementations/` — 主流框架的核心设计、工程边界与实现参考
- `knowledge/catalog.json` — 章节目录及 Topic / Case / Implementation 之间的关联
- `knowledge/roadmap.md` — 每日编辑重点，不再规定每天产出固定文件
- `knowledge/site/` — 知识库静态阅读页面

首批已经收录 Policy Gradient、PPO/GRPO、FP8、R3、并行策略、KV、Async RL、OPD 等基础主题，含“Logprob Drift”“Group 达标但尚未 Drain”“RL Step 变慢”等案例，以及 verl、slime、AReaL、SGLang、Megatron/TransformerEngine、NeMo RL、FSDP2、vLLM 的工程说明。具体源码与实验版本仍需按每篇资料核对。

## 相关内容

- [交互实验](https://kairosxy.github.io/rl-infra-100days-site/labs/) — 五个 Toy 模型实验，对应 Policy Gradient、Logprob 契约和 PPO/GRPO
- [224 道复习题](https://kairosxy.github.io/rl-infra-100days-site/quiz/) — 搜索、短回答、详细解释、收藏、导入导出，数据保存在本地
- [CPU 参考系统](reference_system/) · [工程实验](examples/) · [阅读资料](references/reading_list.md)

## 本地运行

需要 Python 3.11+：

~~~bash
git clone https://github.com/Kairosxy/rl-infra-100days-site.git
cd rl-infra-100days-site
python -m pip install -r requirements-build.txt
python scripts/build_knowledge.py --output dist/index.html
python scripts/build_quiz.py --output dist/quiz/index.html
python -m unittest discover -s scripts -p 'test_*.py' -v
cd dist
python -m http.server 8000
~~~

知识库位于 `http://localhost:8000/`，实验位于 `/labs/`，复习题位于 `/quiz/`。

完整的 PyTorch 教学实验仍通过 `make test` 和 `make reference` 运行。构建和公开数据隔离检查在 GitHub Actions 自动执行；推送代码不等于部署成功，应以 Actions 结果为准。

## 更新节奏

知识库每天更新一次，编辑优先级记录在 [knowledge/roadmap.md](knowledge/roadmap.md)。

复习记录与短回答保存在浏览器本地，可以导出备份。

## 资料来源

每个工程结论尽量关联具体论文、源码 Commit、硬件与模型配置。数学推导、论文报告、代码事实和实际测量分开标注。
