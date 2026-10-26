# 公开课程网站说明

这是 `Kairosxy/rl-infra-100days-site` 的公开网站源码。学习者答案、项目复盘和学习进度不在此仓库中。

## 构建

```bash
python -m pip install -r site/requirements.txt
python scripts/build_site.py --output dist/index.html
python -m unittest discover -s scripts -p 'test_site_builder.py' -v
```

生成的 `dist/index.html` 包含所有已发布课程和插图；数学公式排版依赖在线 MathJax。浏览器本地保存的作答不会上传到网站；点击提交才打开**私有课程仓库**的 GitHub Issue，由原仓库的工作流保存答案。

## 部署

请在本仓库 Settings → Pages → Build and deployment → Source 选择 **GitHub Actions**，并在 Actions 中查看 **Publish course website**。

- 没有启用 Pages 时，CI 仍会构建 `rl-infra-100days-public-html` Artifact。
- 启用并部署成功后：<https://kairosxy.github.io/rl-infra-100days-site/>
- 将本仓库的 `courses/day-XX`、`examples/day_XX` 更新后，公开页面会自动重建。
- 后续同步应只复制公开课程所需的 `lesson.md`、`diagram.md`、`recap.md`、`qa.md`、`assets/` 和可公开的教学示例；**绝不复制 `answers.md`、`progress.md`、`portfolio/` 或任何内部材料**。

注意：Pages 网站对公众可见，课程内容发布前仍须公开化审查。
