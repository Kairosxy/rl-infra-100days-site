# RL Infra

一份面向 RL / LLM 训练工程的公开技术知识库：基础知识、实际工程问题和主流框架实现相互关联。知识可以按主题阅读，也可以从性能故障、FP8 数值差异、MoE Routing、异步训练等具体问题进入。

**[知识库首页](https://kairosxy.github.io/rl-infra-100days-site/)** · [工程问题](https://kairosxy.github.io/rl-infra-100days-site/#category=case) · [框架实现](https://kairosxy.github.io/rl-infra-100days-site/#category=implementation) · [快速复习题](https://kairosxy.github.io/rl-infra-100days-site/quiz/)

不需要登录、写长答案或完成上一节才能访问。简短判断和口头表达只用于帮助恢复记忆；深入研究请阅读完整推导、源码和案例。插图与交互按教学价值选用，而不是每篇都必须附带。

## 内容组织

- `knowledge/topics/` — 基础机制和数学/计算模型
- `knowledge/cases/` — 真实现象、排查顺序、方案取舍和验证方式
- `knowledge/implementations/` — 主流框架的核心设计、工程边界与实现参考
- `knowledge/catalog.json` — 章节目录及 Topic / Case / Implementation 之间的关联
- `knowledge/roadmap.md` — 每日编辑重点，不再规定每天产出固定文件
- `knowledge/site/` — 知识库静态阅读页面

首批已经收录 Policy Gradient、PPO/GRPO、FP8、R3、并行策略、KV、Async RL、OPD 等基础主题，含“Logprob Drift”“Group 达标但尚未 Drain”“RL Step 变慢”等案例，以及 verl、slime、AReaL、SGLang、Megatron/TransformerEngine、NeMo RL、FSDP2、vLLM 的工程说明。具体源码与实验版本仍需按每篇资料核对。

## 原有资料保留

- [原 100 天讲义](https://kairosxy.github.io/rl-infra-100days-site/legacy.html) — Day01–05 已发布课程，保留原来的阅读与浏览器作答功能
- [原 224 道题](https://kairosxy.github.io/rl-infra-100days-site/quiz/) — 搜索、短回答、详细解释、收藏、导入导出，本地数据键保持兼容
- [原教学交互](https://kairosxy.github.io/rl-infra-100days-site/labs/) — 仅作为需要时的补充实验，不强制每课配图/互动
- [CPU 参考系统](reference_system/) · [工程实验](examples/) · [历史课表](course_map/day_schedule.md)

已有旧链接 `/#day=4&view=lesson` 会在新首页重定向到 `/legacy.html#day=4&view=lesson`。旧本地学习记录仍归原页面所有，不自动上传、不清空也不强制迁移。

## 本地运行

需要 Python 3.11+：

~~~bash
git clone https://github.com/Kairosxy/rl-infra-100days-site.git
cd rl-infra-100days-site
python -m pip install -r site/requirements.txt
python scripts/build_site.py --output dist/legacy.html
python scripts/build_knowledge.py --output dist/index.html
python scripts/build_quiz.py --output dist/quiz/index.html
python -m unittest discover -s scripts -p 'test_*.py' -v
cd dist
python -m http.server 8000
~~~

知识库位于 `http://localhost:8000/`，旧课程位于 `/legacy.html`，复习题位于 `/quiz/`。

完整的 PyTorch 教学实验仍通过 `make test` 和 `make reference` 运行。构建和公开数据隔离检查在 GitHub Actions 自动执行；推送代码不等于部署成功，应以 Actions 结果为准。

## 每日更新

北京时间 09:00 持续更新知识库。编辑优先级记录在 [knowledge/roadmap.md](knowledge/roadmap.md)。一天可以新增一个真正必要的 Topic/Case/Implementation，也可以修正某篇重要文档的源码与前提；不再强制每天固定两道题、一张图、一篇讲义和一份复习卡。

公开仓库禁止提交个人答案、公司私密代码、未公开指标、密钥或内部日志。网页存储的个人回答和复习标记仅保存在本地浏览器，必要时由使用者主动导出。

## 资料与可信度

每个工程结论尽可能关联具体论文、源码 Commit、硬件/模型配置与可复现结果。区分：数学推导、论文作者报告、公开代码事实、实际测量。参考解释中未经逐题技术校勘的旧题库答案不能当作权威结论。
