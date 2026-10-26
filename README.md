# RL Infra · 100 Days

> 100 天 RL Infrastructure 系列课程、CPU 参考系统、可验证的代码实验，以及 224 道面试题库。按零基础重新建立概念，再深入 GPU、分布式训练和 Agentic RL。

## 在线入口

**[📖 课程学习工作台](https://kairosxy.github.io/rl-infra-100days-site/)** · **[🔬 交互机制实验](https://kairosxy.github.io/rl-infra-100days-site/labs/)** · **[📝 224 道面试刷题库](https://kairosxy.github.io/rl-infra-100days-site/quiz/)**

课程支持阅读讲义、插图、复习卡、面试题与本地作答。题库支持关键词搜索、章节/领域筛选、随机十题、错题重练、间隔复习、收藏和两套参考答案。

## 课程与代码

- [100 天完整课表](course_map/day_schedule.md) · [课程总纲](course_map/overview.md) · [阶段验收](course_map/milestone_map.md)
- [系统演化主线](course_map/system_roadmap.md) · [能力地图](course_map/competency_map.md)
- [参考训练系统](reference_system/README.md) · [架构说明](architecture/reference_system.md) · [Trajectory 数据契约](architecture/data_contract.md)
- [实验代码](examples/) · [典型失败案例](failures/README.md) · [论文与官方资料](references/reading_list.md)

当前已归档 Day 01–05；大纲已按概念深度重新编排（旧数据契约、显存、梯度累积内容迁至后续对应阶段），其余课程按 100 天课表逐日发布。课程**自由访问，无须完成上一课才能进入下一课**。自评、练习与补课完全自主，不影响下一课发布。

## 唯一课程仓库

**此公开仓库是课程、代码、刷题库和网页的唯一维护入口。** 原有私有仓库 `rl-infra-100days` 留作历史备份，不再作为网站发布源；其私人答题、学习记录、项目叙事和内部材料不会复制到这里。

```text
architecture/         RL 系统与数据契约
course_map/           100 天课表、能力图与里程碑
courses/day-XX/       讲义、图解、复习卡、面试题、教学配图
examples/day_XX/      按天组织的可运行实验
reference_system/     可在 CPU 运行的最小 RL 训练主线
quiz/                 224 道结构化试题、两套答案、刷题页面
site/                 课程浏览器页面
labs/                 交互式数值实验与动态图
scripts/              生成器、测试与公开内容检查
.github/workflows/    自动构建和 Pages 部署
```

## 本地运行

Python 3.11+：

```bash
git clone https://github.com/Kairosxy/rl-infra-100days-site.git
cd rl-infra-100days-site
python -m pip install -r site/requirements.txt
python scripts/build_site.py --output dist/index.html
python scripts/build_quiz.py --output dist/quiz/index.html
python -m unittest discover -s scripts -p 'test_*.py' -v
cd dist && python -m http.server 8000
```

访问 `http://localhost:8000/`、`http://localhost:8000/labs/` 和 `http://localhost:8000/quiz/`。

运行 PyTorch 教学实验和 Toy REINFORCE：

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
make test
make reference
```

## 如何保存学习记录？

网页直接作答时，个人答案保存在当前浏览器的 `localStorage`，**不会自动上传 GitHub**。每课答案可以复制或下载 Markdown，刷题库可以导出、导入 JSON。跨设备或清理浏览器前请先备份。

想获得 AI 导师点评，可以将自己的答案、实验输出或不理解的问题主动发给 ChatGPT。没有作答不会阻止课程更新。

**这是公开仓库和公开网站。** 请勿提交个人答案、内部项目数据、公司代码、私密指标、账户密码或密钥。所有示例均为公开的教学用途；理论估算和实测结果必须明确区分。

## 题库来源

两份用户提供的资料按 Q001–Q224 对齐，而不是拼成 448 道题：

- `RL_OPD_Agentic_RL_面试题库_完整版.md`：速答要点与追问；
- `RL_OPD_Agentic_RL_面试问题与参考答案.docx`：详细解析与场景示例。

9 道题存在两份材料提问措辞不同的情况，页面可显示不同问法。部分数学公式在原 Word 提取中不完整，尚未完成逐题技术校勘，不能视为经过全面验证的标准答案。

## 自动发布

向 `main` 提交课程、题库或网站修改后，GitHub Actions [Publish course website](.github/workflows/pages.yml) 自动生成 HTML 并部署 Pages。构建失败需要以工作流日志为准，不能视为更新已上线。

## 从旧仓库保留的工程资料

- [CPU 示例与参考系统测试](.github/workflows/cpu-tests.yml)：课程示例运行和 Ruff 静态检查。
- [GPU 实验记录模板](docs/gpu_experiment_template.md)：可公开的环境版本、实验基线和测量记录。
- [项目决策与面试陈述模板](docs/project_narrative_template.md)：整理设计取舍和可复现证据。

旧仓库中的空白答案模板、过时的作答解锁进度和双仓库同步流程不再迁移；网页里的草稿、自评和错题继续只保存在浏览器中。

