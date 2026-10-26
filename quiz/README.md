# RL / OPD / Agentic RL 面试刷题库

该题库由两份用户提供的中文手册合并：

- `RL_OPD_Agentic_RL_面试题库_完整版.md`：224 道题的速答要点与“追问落点”。
- `RL_OPD_Agentic_RL_面试问题与参考答案.docx`：同一 Q001–Q224 题号的长篇解释与例子。

**这是对齐，不是简单拼接**：两份手册都覆盖 Q001–Q224，因此保留 224 个 canonical question ID，并附两个来源的答案。共 9 题存在提问措辞差异，页面会展示另一份手册的问法。详细解释保留来源原有表述；部分 Word 公式在文本转换后不完整，暂不把残缺公式伪装成已校准的答案，逐题修订须对照原稿/原论文。

## 练习方式

默认先显示题目。学习者写出自己的结论和判断后，手动展开详细解析或 Markdown 速答，再对照追问。自评档位：

- 再练一次：标记为需巩固，可当天重练
- 还不熟练：明天进入复习
- 基本掌握：先按 3/7/14/30/60 天复习
- 非常熟练：先按 7/14/30/60/90 天复习

这些间隔是教学默认，不是经过认知实验为该用户调参的模型。页面采用浏览器本地时间管理复习日程，不调用外部账户 API。

## 数据流

```text
quiz/data/questions.json -> scripts/build_quiz.py -> dist/quiz/index.html
quiz/template.html      ->        ^
quiz/style.css          ->        ^
quiz/app.js             ->        ^
```

`questions.json` 为唯一发布题库数据源。浏览器中的草稿、自评、笔记由用户主动导出或导入 JSON；构建时不会读取或打包学习者个人数据。原始两份手册不随公开网站上传，只保留必要、结构化的题目与参考解释。

每课对应的 `course_day` 是复习导航提示，不表示题目只属于该课；在课程尚未发布时，点击导航可能默认回到当前已发布的第一课。

## 校验

```bash
python scripts/build_quiz.py
python -m unittest discover -s scripts -p 'test_quiz.py' -v
```

检查连续 Q001–Q224 编号、两套解释及追问存在，且最终 HTML 不包含个人答案、旧私有 Issues URL 或源文件路径。
