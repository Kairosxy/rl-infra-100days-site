# 课程网页与刷题库

本仓库同时托管课程讲义和网站，不需要维护第二个课程项目。

```bash
python -m pip install -r site/requirements.txt
python scripts/build_site.py --output dist/index.html
python scripts/build_quiz.py --output dist/quiz/index.html
python -m unittest discover -s scripts -p 'test_*.py' -v
```

- `/`：课程页，可切换 100 天中的已发布课程；讲义、图解、复习、面试题和本地作答。
- `/quiz/`：224 题刷题库；章节筛选、主题搜索、随机 10 题、错题复习、两套参考解释、自评及导出/导入。
- 网站资源在构建时嵌入 HTML；MathJax 仍需要在线 CDN 完整显示数学公式。
- 所有个人作答、笔记、自评都留在浏览器本地，不会自动进入公开仓库。建议定期备份。
- `main` 分支 Push 触发 .github/workflows/pages.yml，自动生成并发布整个 `dist/`。

私有历史备份仓库中的 `answers.md`、个人项目笔记与 `progress.md` 不会也不应当迁移到此处。
