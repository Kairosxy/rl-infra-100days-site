/* RL Infra 100 Days: one-page reader and learner-owned answer drafts.
   No tokens, passwords, or private answers are sent from this website. */
"use strict";

(function () {
  const REPOSITORY = "Kairosxy/rl-infra-100days-site";
  const COURSES = JSON.parse(document.getElementById("course-data").textContent);
  const BY_DAY = new Map(COURSES.map(function (item) { return [item.day, item]; }));
  const VIEW_NAMES = ["lesson", "diagram", "recap", "qa", "answer"];
  const FIELDS = [
    "q1", "q2", "score1", "score2", "command", "output", "failure",
    "why", "notes", "project"
  ];
  const PREFIX = "rl-infra-100days:answers:";
  let activeDay = COURSES.length ? COURSES[0].day : 1;
  let currentView = "lesson";
  let saveTimer = null;

  const id = function (value) { return document.getElementById(value); };
  const pad = function (day) { return String(day).padStart(2, "0"); };
  const root = "https://github.com/" + REPOSITORY;
  const storageKey = function (day) { return PREFIX + pad(day); };
  const getCourse = function () { return BY_DAY.get(activeDay); };

  function readDraft(day) {
    try {
      const item = localStorage.getItem(storageKey(day));
      return item ? JSON.parse(item) : {};
    } catch (error) {
      return {};
    }
  }

  function formValues() {
    const result = {};
    FIELDS.forEach(function (field) {
      result[field] = id("answer-" + field).value;
    });
    return result;
  }

  function saveDraft() {
    try {
      const values = formValues();
      localStorage.setItem(storageKey(activeDay), JSON.stringify(values));
      id("save-status").textContent = "已保存在本地 · " +
        new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
      updateDraftBadge();
      return true;
    } catch (error) {
      id("save-status").textContent = "浏览器未允许保存，请先下载答案";
      return false;
    }
  }

  function updateDraftBadge() {
    const draft = readDraft(activeDay);
    const filled = ["q1", "q2", "command", "output", "failure", "why", "notes", "project"]
      .some(function (field) { return !!String(draft[field] || "").trim(); });
    id("draft-badge").textContent = filled ? "已有本地草稿" : "尚未作答";
  }

  function loadDraft() {
    const data = readDraft(activeDay);
    FIELDS.forEach(function (field) {
      id("answer-" + field).value = data[field] || "";
    });
    id("save-status").textContent = "草稿会自动保存在当前浏览器";
    updateDraftBadge();
    const course = getCourse();
    id("label-q1").textContent = "题目 1 · " + (course.questions[0] || "").replace(/^题\s*1\s*[:：]\s*/, "");
    id("label-q2").textContent = "题目 2 · " + (course.questions[1] || "").replace(/^题\s*2\s*[:：]\s*/, "");
  }

  function renderCourseList() {
    const element = id("course-list");
    element.replaceChildren();
    COURSES.forEach(function (course) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "course-item" + (course.day === activeDay ? " active" : "");
      button.setAttribute("aria-current", course.day === activeDay ? "page" : "false");
      const number = document.createElement("span");
      number.className = "num";
      number.textContent = pad(course.day);
      const label = document.createElement("span");
      label.className = "text";
      const title = document.createElement("span");
      title.textContent = course.title;
      const tiny = document.createElement("span");
      tiny.className = "tiny";
      tiny.textContent = "查看讲义 / 练习";
      label.append(title, tiny);
      button.append(number, label);
      button.addEventListener("click", function () { navigate(course.day, "lesson"); });
      element.append(button);
    });
    id("published-count").textContent = String(COURSES.length);
  }

  function readLocation() {
    const params = new URLSearchParams(window.location.hash.slice(1));
    const day = Number(params.get("day"));
    const view = params.get("view");
    return {
      day: BY_DAY.has(day) ? day : (COURSES[0] ? COURSES[0].day : 1),
      view: VIEW_NAMES.includes(view) ? view : "lesson"
    };
  }

  function navigate(day, view) {
    if (saveTimer !== null) {
      window.clearTimeout(saveTimer);
      saveTimer = null;
      saveDraft();
    }
    const destination = "#day=" + day + "&view=" + view;
    if (window.location.hash === destination) {
      render();
    } else {
      window.location.hash = destination;
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderTabs(course) {
    document.querySelectorAll("[data-view]").forEach(function (button) {
      const view = button.dataset.view;
      const isAvailable = view === "answer" || !!course.sections[view];
      button.hidden = !isAvailable;
      button.setAttribute("aria-selected", String(view === currentView));
      button.tabIndex = view === currentView ? 0 : -1;
    });
  }

  function showLabs(course) {
    const element = id("example-links");
    element.replaceChildren();
    if (currentView !== "lesson" || !course.labs.length) {
      element.hidden = true;
      return;
    }
    const heading = document.createElement("b");
    heading.textContent = "本课代码与测试";
    element.append(heading);
    course.labs.forEach(function (file) {
      const link = document.createElement("a");
      link.href = "https://github.com/Kairosxy/rl-infra-100days-site/blob/main/" + file;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = file.split("/").slice(-1)[0] + " ↗";
      element.append(link);
    });
    element.hidden = false;
  }

  function typeset(element) {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([element]).catch(function (error) {
        console.warn("MathJax render failed", error);
      });
    }
  }

  function render() {
    if (!COURSES.length) {
      id("course-title").textContent = "课程尚未发布";
      id("article").textContent = "目前还没有可阅读的课程。";
      return;
    }
    const next = readLocation();
    const dayChanged = next.day !== activeDay;
    if (dayChanged && saveTimer !== null) {
      window.clearTimeout(saveTimer);
      saveTimer = null;
      saveDraft();
    }
    activeDay = next.day;
    currentView = next.view;
    const course = getCourse();

    id("course-breadcrumb").textContent = activeDay <= 10
      ? "模块一 · RL 与训练基础"
      : "RL Infrastructure · 持续进阶";
    id("day-label").textContent = "DAY " + pad(activeDay) + " / 100";
    id("course-title").textContent = course.title;
    id("course-subtitle").textContent =
      "本课包含讲义、实验和面试题；回答会先保存在浏览器。";
    document.title = "Day " + pad(activeDay) + " · " + course.title + " | RL Infra";

    renderCourseList();
    renderTabs(course);
    if (dayChanged || currentView === "answer") {
      loadDraft();
    } else {
      updateDraftBadge();
    }

    const isAnswerView = currentView === "answer";
    id("reader").hidden = isAnswerView;
    id("answer-panel").hidden = !isAnswerView;
    if (!isAnswerView) {
      id("article").innerHTML = course.sections[currentView] ||
        '<p class="empty-state">这一部分尚未归档，请先阅读讲义。</p>';
      showLabs(course);
      typeset(id("article"));
    }
  }

  function answerMarkdown() {
    const course = getCourse();
    const values = formValues();
    function value(key, fallback) {
      return (values[key] || "").trim() || (fallback || "未填写");
    }
    function score(key) {
      const n = Number(values[key]);
      return values[key] !== "" && Number.isInteger(n) && n >= 0 && n <= 10
        ? n + " / 10" : "尚未自评";
    }
    const day = pad(activeDay);
    const indent = function (data) {
      return value(data).split("\n").map(function (line) { return "    " + line; }).join("\n");
    };
    return [
      "<!-- RL_INFRA_ANSWER_V1 day=" + day + " -->",
      "# Day " + day + " · 学习者作答",
      "",
      "## 1. 面试题作答",
      "",
      "### 题 1 · " + course.questions[0],
      "",
      value("q1"), "",
      "自评：" + score("score1"), "",
      "### 题 2 · " + course.questions[1], "",
      value("q2"), "",
      "自评：" + score("score2"), "",
      "## 2. 练习记录", "",
      "**实际运行的命令**", "",
      indent("command"), "",
      "**关键输出或测试结果**", "",
      indent("output"), "",
      "**故意制造的错误与报错**", "",
      value("failure"), "",
      "**原因分析**", "",
      value("why"), "",
      "## 3. 质疑与补充", "",
      value("notes", "暂无"), "",
      "## 4. 项目对照（可选）", "",
      value("project", "暂无"), ""
    ].join("\n");
  }

  function downloadAnswers() {
    saveDraft();
    const blob = new Blob([answerMarkdown()], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "day-" + pad(activeDay) + "_answers.md";
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
    id("save-status").textContent = "已生成 Markdown 下载";
  }

  async function copyAnswers() {
    saveDraft();
    const result = answerMarkdown();
    try {
      await navigator.clipboard.writeText(result);
      id("save-status").textContent = "Markdown 已复制，可粘贴到自己的笔记或聊天中";
    } catch (error) {
      id("save-status").textContent = "复制失败，请使用“下载答案”";
    }
  }

  document.querySelectorAll("[data-view]").forEach(function (button) {
    button.addEventListener("click", function () { navigate(activeDay, button.dataset.view); });
  });
  id("answer-form").addEventListener("input", function () {
    window.clearTimeout(saveTimer);
    saveTimer = window.setTimeout(saveDraft, 180);
  });
  id("copy-answer").addEventListener("click", copyAnswers);
  id("download-answer").addEventListener("click", downloadAnswers);
  id("submit-answer").addEventListener("click", downloadAnswers);
  window.addEventListener("hashchange", render);
  render();
})();