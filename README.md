# RL Infra · 100 Days

> 100 天 RL Infrastructure 系列课程、CPU 参考系统、可验证的代码实验与面试题库。**从基础重新学起，再走到分布式训练和 Agentic RL。**

## 开始学习

- **[在线课程](https://kairosxy.github.io/rl-infra-100days-site/)** — 按课程阅读讲义、手绘风格插图、复习卡与实验。
- **[面试刷题库 · 224 题](https://kairosxy.github.io/rl-infra-100days-site/quiz/)** — 覆盖 RL / OPD / Agentic RL / 分布式 / 排障；题号 Q001–Q224 对齐两份手册。随机十题、错题重练、间隔复习、个人笔记。
- [完整课表](course_map/day_schedule.md) · [课程总纲](course_map/overview.md) · [能力地图](course_map/competency_map.md) · [阶段里程碑](course_map/milestone_map.md)
- [参考系统](architecture/reference_system.md) · [数据契约](architecture/data_contract.md) · [故障模型](architecture/failure_model.md)
- [论文与官方资料](references/reading_list.md)

本仓库是**课程内容与网站的唯一维护入口**。课程 Markdown、实验代码和 HTML 构建器都在这里，`main` 有变更时 GitHub Pages 自动重建。旧的私有课程仓库只作为历史备份，不再参与课程发布。

## 目录

```text
architecture/          贯穿整个课程的 RL 参考系统、数据契约、版本和恢复边界
course_map/            100 天课表、能力地图和阶段验收
courses/day-XX/        lesson.md / diagram.md / recap.md / qa.md / assets/
examples/day_XX/       每日可运行实验与反例测试
reference_system/      CPU 上运行的最小训练主线
benchmarks/            性能实验报告 schema
failures/              失败案例记录规范
references/            论文与官方资料索引
quiz/                  224 道题的结构化数据、刷题页面与交互代码
site/                  课程阅读页面
scripts/               HTML 生成和内容测试
.github/workflows/     网站自动发布
```

## 自己动手

需要 Python 3.11+。课程页面生成不需要 PyTorch；CPU 训练实验需要 PyTorch。

```bash
git clone https://github.com/Kairosxy/rl-infra-100days-site.git
cd rl-infra-100days-site

python -m pip install -r site/requirements.txt
python scripts/build_site.py --output dist/index.html
python scripts/build_quiz.py --output dist/quiz/index.html
python -m unittest discover -s scripts -p 'test_*.py' -v

# CPU 参考系统（另需 PyTorch）
python -m pip install -r requirements.txt
python reference_system/toy_reinforce.py
make test
```

构建完成后，可用本地 HTTP 服务查看全部站点：

```bash
cd dist
python -m http.server 8000
# http://localhost:8000/
# http://localhost:8000/quiz/
```

## 课程学习记录与隐私

网站**不会把你的作答自动上传到 GitHub**。课程页的自答文本以及刷题库的自评、错题、收藏、个人笔记保存在当前浏览器的 `localStorage`：

- 每课答案可以复制或下载 Markdown；**刷题库可导出 / 导入 JSON**，跨设备时请手动备份。
- 无痕窗口、清理浏览器数据或更换设备可能造成未导出记录丢失。
- 不能把本地草稿当作定时课程任务已经收到的反馈。需要导师点评时，请将答案或实验记录主动发回 ChatGPT。
- 本仓库和 Pages 均为 **Public**。不要把个人学习答案、内部代码、业务指标、公司资料、API Key 或未授权内容提交到仓库。

仓库不会发布 `answers.md`、`progress.md`、`portfolio/` 或内部实验日志。设计、实现、测量须分别注明；不能把估算写成实测。

## 题库说明

题库由两份已提供的手册按相同 Q 编号合并：224 题不是 448 题。Markdown 版的回答用作速答要点，Word 版的回答用作详细解析；存在提问措辞或推导差异时保留两个版本以便核对。题库主要训练口头表达、机制追问和排障推理，**没有声称自动判分，也不把参考答案当作无误标准**。

技术解释如有争议，应查阅原论文和可执行实验后修订，不应按同一主题重复计题。

## 维护规则

每节课应明确问题、前提、公式、案例、张量/代码映射、成本、反例、动手练习和验收条件；优先用可审校的 Markdown 和手绘风格 SVG 插图。每次新增课程要补实验或对照来源，并检查浏览器数学公式的可读性。图像不堆砌重复数据流。

网站由 [Publish course website](.github/workflows/pages.yml) 自动部署。部署失败时查看 Actions 日志；部署成功后 Pages 的缓存刷新可能有短暂延迟。

## 免责声明

这是公开学习资料和练习系统，不包含任何公司内部工程实现。数值例子是教学假设，除非明确标出硬件、软件和完整复现配置，否则不是 GPU 性能实测。
