# 单仓库课程网站

原始课程、代码与公开网站现在由 `Kairosxy/rl-infra-100days-site` 一个仓库维护，发布网站不需要从第二个私有仓库搬运文件。

```bash
python -m pip install -r site/requirements.txt
python scripts/build_site.py --output dist/index.html
python scripts/build_quiz.py --output dist/quiz/index.html
python -m unittest discover -s scripts -p 'test_*.py' -v
```

首页 `/` 阅读所有已发布课程（可自由选课，**没有答题解锁要求**）。`/quiz/` 提供 224 道 RL/OPD/Agentic RL 面试刷题、速答/详细解析、收藏及复习记录。

课程作答和题库笔记只保存在浏览器本地，分别导出 Markdown 或 JSON 备份；网页不会将它们提交到公开 GitHub Issues。提交至 `main` 的内容由 GitHub Actions 自动构建部署。
