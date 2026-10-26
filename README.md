# RL Infra · 100 Days — 课程阅读网站

这是 RL Infrastructure 100 天课程的**公开阅读入口**，提供手机和桌面浏览器可用的 HTML 课程、教学插图、面试题和可公开的 CPU 实验代码。

**在线页面（部署成功后）**：[https://kairosxy.github.io/rl-infra-100days-site/](https://kairosxy.github.io/rl-infra-100days-site/)

## 双仓库边界

- **本仓库（Public）**：从课程项目中筛选、审校后发布的讲义、图解、复习卡、面试题、公开示例代码及网站构建程序。
- **主课程仓库（Private）**：学习者答案 `answers.md`、学习进度、项目叙事、个人反馈和未公开的开发资料。它们**不会**同步到本仓库。

网页的“我的作答”先保存在浏览器；点击“提交到 GitHub”会打开原私有仓库的 Issue（需登录并拥有访问权限），由原仓库中的工作流将答案保存到相应的 `answers.md`。

## 启用 GitHub Pages

1. 打开本仓库的 **Settings → Pages**。
2. 在 **Build and deployment → Source** 选择 **GitHub Actions**。
3. 打开 **Actions → Publish course website**，检查构建/部署是否成功。若先前构建尚未部署，可点 **Run workflow** 重新触发。

页面默认地址是 `https://kairosxy.github.io/rl-infra-100days-site/`，仅在部署成功后生效。

## 本地生成

```bash
python -m pip install -r site/requirements.txt
python scripts/build_site.py --output dist/index.html
```

使用浏览器打开 `dist/index.html`。已复制进本仓库的课程 SVG 插图会被嵌入 HTML；在线 MathJax 用于排版公式。

本仓库的发布工作流会运行构建和隐私检查，并在 Actions 中提供可下载的 HTML Artifact。发布的课程以 `courses/day-XX/lesson.md` 为准；后续课程经过公开化审校后加入即可出现在网站目录中。

**注意**：本仓库及 GitHub Pages 的课程内容对公众可见；请勿在此提交个人答案、保密代码、内部指标、账户信息或密钥。
