# RL Infra

一份面向 RL / LLM 训练工程的公开技术知识库：基础知识、实际工程问题和主流框架实现相互关联。知识可以按主题阅读，也可以从性能故障、FP8 数值差异、MoE Routing、异步训练等具体问题进入。

**[知识库首页](https://kairosxy.github.io/rl-infra-100days-site/)** · [工程问题](https://kairosxy.github.io/rl-infra-100days-site/#view=questions&category=case) · [框架实现](https://kairosxy.github.io/rl-infra-100days-site/#view=library&kind=implementation) · [快速复习题](https://kairosxy.github.io/rl-infra-100days-site/quiz/)

简短判断和口头表达用来帮助恢复记忆，深入研究请读完整推导、源码和案例。

## 仓库目录

```text
knowledge/
  topics/<chapter>/<id>.md          # 原理、算法与技术家族
  cases/<domain>/<id>.md            # 工程问题与可验证案例
  implementations/<framework>/<id>.md
  assets/<topic>/                  # 正文机制图
  catalog.json                     # 稳定 ID、source、目录与关联
  learning-map.json                # 先修 DAG、学习路径和问题映射
  roadmap.md                       # 持续建设计划
  site/                            # 静态阅读器（不存正文）
docs/
  architecture/                    # 系统设计与契约
  learning/                        # 历史学习计划
  design/                          # 产品设计记录
  benchmarks/                      # 实验规范与结果 schema
  diagnostics/                     # 故障记录
  references/                      # 阅读资料
scripts/                           # 构建、校验、回归与部署工具
examples/                          # 可运行工程实验
reference_system/                  # CPU 参考实现
quiz/                              # 原始 224 题及历史入口
labs/                              # 兼容旧实验链接
.github/workflows/                 # CI / Pages
```

每个知识条目的 `id` 保持不变，`source` 是相对 `knowledge/` 的规范路径。Topic 按 Chapter、Case 按主要领域、Implementation 按框架归档；跨框架实现放在 `cross-framework/`，其他关联由 catalog 表达，不复制正文。新增文章必须更新 catalog 及必要的 learning-map；目录调整不改变网站 hash 路由或浏览器学习记录。技术家族覆盖见 [RL Infra 技术谱系与演进](knowledge/topics/research/technology-lineage.md)。

现有 Git 历史保留旧源文件；站内稳定路由保持不变。直接引用旧 GitHub main 文件路径的外部文档应改用新路径或历史 commit。

## 相关内容

- [交互实验](https://kairosxy.github.io/rl-infra-100days-site/labs/) — 五个 Toy 模型实验，对应 Policy Gradient、Logprob 契约和 PPO/GRPO
- [224 道复习题](https://kairosxy.github.io/rl-infra-100days-site/quiz/) — 搜索、短回答、详细解释、收藏、导入导出，数据保存在本地
- [CPU 参考系统](reference_system/) · [工程实验](examples/) · [阅读资料](docs/references/reading_list.md)

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
