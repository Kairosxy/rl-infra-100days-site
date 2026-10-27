"use strict";

/* 224-question interview trainer. Progress is browser-local. No network writes. */
(() => {
  const payload = JSON.parse(document.getElementById("quiz-data").textContent);
  if (payload.schema !== "rl-infra-question-bank-v1") throw Error("Invalid question bank");
  const questions = payload.questions;
  const topicTitle = new Map((payload.topics || []).map(t => [t.id, t.title]));
  const lookup = new Map(questions.map(q => [q.id, q]));
  const KEY = "rl-infra-100days-questionbank-v1";
  const domainNames = {
    policy: "策略梯度 / PPO", reward: "奖励与偏好",
    grpo: "GRPO / DAPO", opd: "OPD 蒸馏",
    async: "异步与 Partial", agent: "Agentic RL",
    inference: "推理与一致性", distributed: "分布式与 MoE",
    infra: "系统设计", debug: "训练排障",
    precision: "低精度与性能"
  };
  const ratingNames = {
    again: "再练一次", hard: "还不熟练",
    good: "基本掌握", easy: "非常熟练"
  };
  const byId = id => document.getElementById(id);
  const two = n => String(n).padStart(2, "0");
  const LAYERS = [["c1", "结论"], ["c2", "前提"], ["c3", "数值"], ["c4", "落地"]];
  const layerField = id => document.getElementById("answer-" + id);

  function blank() {
    return { answer: "", layers: {}, note: "", rating: "", attempts: 0,
      streak: 0, due: null, bookmarked: false, history: [] };
  }
  function loadProgress() {
    try {
      const obj = JSON.parse(localStorage.getItem(KEY));
      if (obj && obj.version === 1 && obj.records && typeof obj.records === "object") return obj;
    } catch (_) {}
    return { version: 1, records: {} };
  }
  let state = loadProgress();
  let mode = "all", queue = [], cursor = 0, current = null, visible = false;
  let source = "brief", timer = null;
  const read = id => state.records[id] || blank();
  const writable = id => state.records[id] || (state.records[id] = blank());
  const localDate = (days = 0) => {
    const d = new Date();
    d.setHours(12, 0, 0, 0);
    d.setDate(d.getDate() + days);
    return [d.getFullYear(), two(d.getMonth() + 1), two(d.getDate())].join("-");
  };
  const due = record => Boolean(record.due && record.due <= localDate());
  const weak = record => record.rating === "again" || record.rating === "hard";
  const learned = record => record.rating === "good" || record.rating === "easy";
  const normalize = s => String(s || "").normalize("NFKC")
    .replace(/[？?。，,.：:；;、\s]/g, "").toLowerCase();

  function saveState() {
    try {
      localStorage.setItem(KEY, JSON.stringify(state));
      byId("saved").textContent = "已保存在本地 · " +
        new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    } catch (_) {
      byId("saved").textContent = "本机保存失败，请导出备份";
    }
    updateStats();
  }
  function saveFields() {
    if (current === null) return;
    const record = writable(current);
    const layers = {};
    for (const [key] of LAYERS) layers[key] = layerField(key).value;
    record.layers = layers;
    // `answer` stays the joined text so exports and older records read the same.
    record.answer = LAYERS
      .map(([key, name]) => [name, layers[key].trim()])
      .filter(([, text]) => text)
      .map(([name, text]) => name + "：" + text).join("\n");
    record.note = byId("notes").value;
    saveState();
  }
  function flush() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    saveFields();
  }
  function autosave() {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; saveFields(); }, 250);
    updateDraftCounters();
  }
  function updateDraftCounters() {
    const lengths = LAYERS.map(([key]) => layerField(key).value.trim().length);
    byId("count").textContent = lengths.reduce((a, b) => a + b, 0) + " 字";
    byId("layer-progress").textContent =
      "已写 " + lengths.filter(Boolean).length + " / " + LAYERS.length + " 层";
  }
  function updateStats() {
    let attempted = 0, mastered = 0, dueCount = 0, stars = 0;
    for (const q of questions) {
      const r = read(q.id);
      if (r.attempts > 0) attempted++;
      if (learned(r)) mastered++;
      if (due(r)) dueCount++;
      if (r.bookmarked) stars++;
    }
    byId("attempted").textContent = attempted;
    byId("learned").textContent = mastered;
    byId("due-count").textContent = dueCount;
    byId("starred-count").textContent = stars;
    byId("progress-fill").style.width = 100 * attempted / questions.length + "%";
  }

  function selectors() {
    const domains = [...new Set(questions.map(q => q.domain))];
    for (const name of domains) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = domainNames[name] || name;
      byId("domain").append(option);
    }
    const chapters = [...new Set(questions.map(q => q.chapter))];
    for (const name of chapters) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      byId("chapter").append(option);
    }
  }
  function filtered() {
    const domain = byId("domain").value, chapter = byId("chapter").value,
      level = byId("level").value, status = byId("status").value,
      words = byId("search").value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return questions.filter(q => {
      const r = read(q.id);
      if (domain && q.domain !== domain) return false;
      if (chapter && q.chapter !== chapter) return false;
      if (level && q.level !== level) return false;
      if (status === "new" && r.attempts) return false;
      if (status === "weak" && !weak(r)) return false;
      if (status === "good" && !learned(r)) return false;
      if (status === "due" && !due(r)) return false;
      if (status === "star" && !r.bookmarked) return false;
      if (!words.length) return true;
      const allText = [q.code, q.question, q.doc_title, q.brief, q.deep, q.followup, q.chapter]
        .join(" ").toLowerCase();
      return words.every(w => allText.includes(w));
    }).map(q => q.id);
  }
  function randomized(ids) {
    const a = [...ids];
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  function buildQueue(nextMode, target) {
    flush();
    mode = nextMode;
    const pool = filtered();
    byId("filtered").textContent = pool.length;
    queue = mode === "random" ? randomized(pool).slice(0, 10)
      : mode === "due" ? pool.filter(id => due(read(id)))
      : mode === "weak" ? pool.filter(id => weak(read(id))) : pool;
    cursor = target && queue.includes(target) ? queue.indexOf(target)
      : mode === "all" && current && queue.includes(current) ? queue.indexOf(current) : 0;
    document.querySelectorAll("[data-mode]").forEach(el => {
      el.setAttribute("aria-pressed", el.dataset.mode === mode ? "true" : "false");
    });
    const modes = {
      all: "顺序练习", random: "本轮随机 10 题",
      due: "今日到期复习", weak: "薄弱题复练"
    };
    byId("mode-description").textContent = modes[mode] + " · " + queue.length + " 道";
    showQuestion();
  }
  function hideReference() {
    visible = false;
    source = "brief";
    byId("reference").hidden = true;
    byId("show").textContent = "查看参考答案";
    byId("show").setAttribute("aria-expanded", "false");
    document.querySelectorAll("[data-source]").forEach(el => {
      el.setAttribute("aria-selected", el.dataset.source === "brief" ? "true" : "false");
    });
  }
  function showQuestion() {
    hideReference();
    if (!queue.length) {
      current = null;
      byId("qid").textContent = "—";
      byId("question-title").textContent = "当前筛选没有题目，请调整范围";
      byId("chapter-tag").textContent = "空结果";
      byId("level-tag").textContent = "";
      byId("alternative-title").hidden = true;
      for (const [key] of LAYERS) layerField(key).value = "";
      byId("notes").value = "";
      for (const [key] of LAYERS) layerField(key).disabled = true;
      byId("notes").disabled = true;
      for (const id of ["show", "star", "prev", "next", "copy"]) byId(id).disabled = true;
      byId("course-link").hidden = true;
      byId("position").textContent = "0 / 0";
      updateStats();
      return;
    }
    cursor = Math.min(Math.max(cursor, 0), queue.length - 1);
    current = queue[cursor];
    const q = lookup.get(current), r = read(current);
    for (const id of ["show", "star", "copy"]) byId(id).disabled = false;
    for (const [key] of LAYERS) layerField(key).disabled = false;
    byId("notes").disabled = false;
    byId("qid").textContent = q.code;
    byId("chapter-tag").textContent = q.chapter;
    byId("level-tag").textContent = q.level;
    byId("question-title").textContent = q.doc_title || q.question;
    const different = normalize(q.doc_title) !== normalize(q.question);
    byId("alternative-title").hidden = !different;
    byId("alternative-title").textContent = different ? "另一份手册的问法：" + q.question : "";
    // Older records only stored one text field; show it in the first layer.
    const stored = r.layers && typeof r.layers === "object" ? r.layers : {};
    for (const [key] of LAYERS) {
      const value = stored[key];
      layerField(key).value = typeof value === "string" ? value : (key === "c1" ? (r.answer || "") : "");
    }
    byId("notes").value = r.note || "";
    byId("star").textContent = r.bookmarked ? "★ 已收藏" : "☆ 收藏";
    byId("star").setAttribute("aria-pressed", r.bookmarked ? "true" : "false");
    byId("prev").disabled = cursor === 0;
    byId("next").disabled = cursor + 1 === queue.length;
    byId("position").textContent = cursor + 1 + " / " + queue.length;
    updateDraftCounters();
    byId("saved").textContent = r.attempts ? "已自评 " + r.attempts + " 次" : "草稿仅存在本机";
    const topic = q.topic;
    byId("course-link").hidden = !topic;
    if (topic) {
      byId("course-link").href = "../#topic=" + topic;
      byId("course-link").textContent = "相关主题：" + (topicTitle.get(topic) || topic) + " ↗";
    }
    history.replaceState(null, "", "#q=" + q.id);
    updateStats();
  }
  function renderSource() {
    const q = lookup.get(current);
    byId("reference-text").textContent = source === "deep" ? q.deep : q.brief;
    byId("followup").textContent = q.followup;
    document.querySelectorAll("[data-source]").forEach(el => {
      el.setAttribute("aria-selected", el.dataset.source === source ? "true" : "false");
    });
  }
  function rate(value) {
    if (!current) return;
    flush();
    const r = writable(current);
    r.attempts = (r.attempts || 0) + 1;
    r.streak = value === "good" || value === "easy" ? (r.streak || 0) + 1 : 0;
    const goodIntervals = [3, 7, 14, 30, 60], easyIntervals = [7, 14, 30, 60, 90];
    const interval = value === "again" ? 0 : value === "hard" ? 1
      : value === "good" ? goodIntervals[Math.min(r.streak - 1, goodIntervals.length - 1)]
      : easyIntervals[Math.min(r.streak - 1, easyIntervals.length - 1)];
    r.rating = value;
    r.due = localDate(interval);
    r.history = (r.history || []).concat({
      rating: value, date: new Date().toISOString()
    }).slice(-30);
    saveState();
    byId("saved").textContent = "已记录：" +
      ({again:"再练一次",hard:"还不熟练",good:"基本掌握",easy:"非常熟练"})[value];
    if (cursor + 1 < queue.length) {
      cursor++;
      showQuestion();
    } else {
      byId("mode-description").textContent = "本轮结束；可重新选择模式继续";
    }
  }
  function move(n) {
    if (!current || cursor + n < 0 || cursor + n >= queue.length) return;
    flush();
    cursor += n;
    showQuestion();
    byId("question-title").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function exportFile() {
    flush();
    const payload = {
      format: "rl-infra-interview-practice-v1",
      createdAt: new Date().toISOString(),
      records: state.records
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)],
      { type: "application/json;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "rl-infra-practice-" + localDate() + ".json";
    document.body.append(a);a.click();a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }
  async function importFile(file) {
    if (!file) return;
    if (file.size > 5_000_000) { alert("请使用小于 5 MB 的练习记录。");return; }
    let obj;
    try {
      obj = JSON.parse(await file.text());
      if (obj.format !== "rl-infra-interview-practice-v1" || !obj.records ||
          typeof obj.records !== "object") throw Error("format");
    } catch (_) { alert("不是本题库的练习记录文件。");return; }
    if (!confirm("导入将覆盖同题的现有记录，确定吗？")) return;
    flush();
    const imported = {};
    for (const [id, r] of Object.entries(obj.records)) {
      if (!lookup.has(Number(id)) || !r || typeof r !== "object") continue;
      const out = blank();
      out.answer = String(r.answer || "").slice(0, 60000);
      out.layers = {};
      for (const [key] of LAYERS) {
        const value = r.layers?.[key];
        if (typeof value === "string") out.layers[key] = value.slice(0, 20000);
      }
      out.note = String(r.note || "").slice(0, 15000);
      out.rating = Object.hasOwn({again:1,hard:1,good:1,easy:1},r.rating) ? r.rating : "";
      out.attempts = Math.max(0, Math.min(10000,Number(r.attempts)||0));
      out.streak = Math.max(0,Math.min(1000,Number(r.streak)||0));
      out.due = /^\d{4}-\d{2}-\d{2}$/.test(r.due||"") ? r.due : null;
      out.bookmarked = !!r.bookmarked;
      out.history = Array.isArray(r.history) ? r.history.slice(-30) : [];
      imported[id] = out;
    }
    state.records = { ...state.records, ...imported };
    saveState();
    buildQueue("all", current);
    byId("saved").textContent = "练习记录已导入";
  }
  function initialize() {
    selectors();
    for (const id of ["search", "domain", "chapter", "level", "status"]) {
      byId(id).addEventListener(id === "search" ? "input" : "change",
        () => buildQueue("all"));
    }
    byId("clear").addEventListener("click", () => {
      for (const id of ["search","domain","chapter","level","status"]) byId(id).value = "";
      buildQueue("all");
    });
    document.querySelectorAll("[data-mode]").forEach(el => {
      el.addEventListener("click", () => buildQueue(el.dataset.mode));
    });
    for (const [key] of LAYERS) layerField(key).addEventListener("input", autosave);
    byId("notes").addEventListener("input", autosave);
    byId("show").addEventListener("click", () => {
      if (!current) return;
      visible = !visible;
      byId("reference").hidden = !visible;
      byId("show").textContent = visible ? "收起参考答案" : "查看参考答案";
      byId("show").setAttribute("aria-expanded", visible ? "true" : "false");
      if (visible) renderSource();
    });
    document.querySelectorAll("[data-source]").forEach(el => {
      el.addEventListener("click", () => {if (!current) return;source = el.dataset.source;renderSource();});
    });
    byId("star").addEventListener("click", () => {
      if (!current) return;
      const r = writable(current);r.bookmarked = !r.bookmarked;
      saveState();
      byId("star").textContent = r.bookmarked ? "★ 已收藏" : "☆ 收藏";
      byId("star").setAttribute("aria-pressed", r.bookmarked ? "true" : "false");
    });
    byId("copy").addEventListener("click", async () => {
      if (!current) return;
      const q = lookup.get(current);
      try {
        await navigator.clipboard.writeText(q.code + " " + (q.doc_title||q.question));
        byId("saved").textContent = "题目已复制";
      } catch (_) {byId("saved").textContent = "复制不可用，请手动选中文本";}
    });
    document.querySelectorAll("[data-rating]").forEach(el => {
      el.addEventListener("click", () => rate(el.dataset.rating));
    });
    byId("prev").addEventListener("click", () => move(-1));
    byId("next").addEventListener("click", () => move(1));
    byId("export").addEventListener("click",exportFile);
    byId("import").addEventListener("click",()=>byId("import-file").click());
    byId("import-file").addEventListener("change",e=>importFile(e.target.files[0]));
    window.addEventListener("beforeunload",()=>{if(timer!==null)saveFields();});
    window.addEventListener("hashchange", () => {
      const id = Number(new URLSearchParams(location.hash.slice(1)).get("q"));
      if (lookup.has(id) && id !== current) buildQueue("all",id);
    });
    document.addEventListener("keydown",e=>{
      const active=document.activeElement?.tagName;
      if(["INPUT","TEXTAREA","SELECT"].includes(active)||e.ctrlKey||e.metaKey||e.altKey)return;
      if(e.key==="ArrowLeft")move(-1);
      if(e.key==="ArrowRight")move(1);
    });
    const requested = Number(new URLSearchParams(location.hash.slice(1)).get("q"));
    buildQueue("all",lookup.has(requested)?requested:null);
  }
  initialize();
})();
