# RL Infra 知识内容约定

这一层不是把 100 天课程换个目录继续发布。内容按三个实体维护：

- `topics/{id}.md`：稳定的基础知识。解释核心模型、必要的数学/代码与可被验证的边界。
- `cases/{id}.md`：真实工程问题。先写现象和最小反证，再列候选原因、测量、方案与限制。
- `implementations/{id}.md`：主流框架实现。说明为什么这样设计、在哪些版本/子系统有效、资源与维护代价；无法确认源码的细节不能编造。

`catalog.json` 是目录与关联关系的唯一索引。每个条目拥有稳定的 `id`；`topics` 归属一个 Chapter，`cases` 和 `implementations` 通过双向引用与核心主题关联。构建器检查重复 ID、缺失正文和损坏关联，失败时阻止发布。

论文/框架不构成第四份重复讲义：直接作为对应 Topic、Case、Implementation 的证据，并标注论文声称、源码已核实、推导示意、实测结果的差别。具体 API/源码结论应有 Commit/Release 依据。

## 面向读者

读者可能只想花半分钟恢复一个概念，可能为了会议表达，也可能愿意深挖一次算法推导。Topic 的目录字段提供一个真实问题、简短解释与一句自然的口头表达；完整文章另行给出必要细节。**不需要打字、通过考试、每天做交互实验才可阅读**。

互动或配图只在静态文字难以表现一项关键变化时使用。插入前明确：用户可以改变什么、图显示哪个计算对象、错误实现如何与正确结果区分。没有信息收益就不做。

## 历史链接与隐私

224 道复习题和本机复习进度保留在 `quiz/`，不自动迁移到公开服务器或重置。本知识库只发布公开的 Markdown 和来源；不要提交私人技术资料、项目性能、密钥或答题记录。

## 检查

在仓库根目录：

~~~bash
python scripts/build_knowledge.py --output dist/index.html
python scripts/build_quiz.py --output dist/quiz/index.html
python -m unittest discover -s scripts -p 'test_*.py' -v
node --check knowledge/site/app.js
~~~

知识库首页是 `dist/index.html`，交互实验复制到 `dist/labs/`。