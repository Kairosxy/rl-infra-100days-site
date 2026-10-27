"use strict";
/* The public knowledge reader has no network writes and stores no user data.
 */
(function () {
  const raw = JSON.parse(document.getElementById("knowledge-data").textContent);
  if (raw.schema !== "rl-infra-knowledge-v1") throw new Error("Invalid knowledge catalog");
  const $ = id => document.getElementById(id);
  const types = {
    topic: {key:"topics",label:"基础知识",prefix:"topic"},
    case: {key:"cases",label:"工程问题",prefix:"case"},
    implementation: {key:"implementations",label:"框架实现",prefix:"implementation"}
  };
  const pages = [];
  const indexed = new Map();
  for (const [type, info] of Object.entries(types)) {
    for (const item of raw[info.key]) {
      const page = {...item,type};
      const lookupKey = type+":"+page.id;
      if (indexed.has(lookupKey)) throw new Error("Duplicate ID "+lookupKey);
      indexed.set(lookupKey,page);pages.push(page);
    }
  }
  const chapters = new Map(raw.chapters.map(c=>[c.id,c]));
  let filter = "all", chapter = "", search = "", savedLanding = "";
  const params = ()=>new URLSearchParams(location.hash.replace(/^#/,""));
  const href = (type,id)=>"#"+type+"="+encodeURIComponent(id);
  function el(tag, cls, txt) {
    const out = document.createElement(tag);
    if (cls) out.className=cls;
    if (txt != null) out.textContent=txt;
    return out;
  }
  function sectionName(page) {
    return page.type==="topic"
      ? (chapters.get(page.chapter)?.title || "基础知识")
      : (page.type==="case" ? "现象与排查" : page.framework);
  }
  function relatedItems(page) {
    const targets = [];
    if (page.type==="topic") {
      (page.cases || []).forEach(id=>targets.push({type:"case",id}));
      (page.implementations || []).forEach(id=>targets.push({type:"implementation",id}));
    } else if (page.type==="case") {
      (page.topics || []).forEach(id=>targets.push({type:"topic",id}));
      (page.implementations || []).forEach(id=>targets.push({type:"implementation",id}));
    } else {
      (page.topics || []).forEach(id=>targets.push({type:"topic",id}));
      (page.cases || []).forEach(id=>targets.push({type:"case",id}));
    }
    return targets.map(p=>indexed.get(p.type+":"+p.id)).filter(Boolean);
  }
  function renderChapters() {
    const nav = $("chapter-list");
    nav.replaceChildren();
    for (const c of raw.chapters) {
      const a = el("a","chapter-choice",c.title);
      a.href = "#chapter="+c.id;
      a.dataset.chapter = c.id;
      nav.appendChild(a);
    }
  }
  function updateCurrent() {
    $("chapter-list").querySelectorAll("a").forEach(a=>
      a.classList.toggle("active",chapter!==""&&a.dataset.chapter===chapter));
    document.querySelectorAll("[data-filter]").forEach(b=>
      b.setAttribute("aria-pressed",String(b.dataset.filter===filter)));
    document.querySelectorAll("[data-nav]").forEach(a=>
      a.setAttribute("aria-current",String(a.dataset.nav===filter)));
  }
  const plainCache=new Map();
  function plainText(html) {
    if (plainCache.has(html)) return plainCache.get(html);
    const box=document.createElement("div");
    box.innerHTML=html;
    const text=(box.textContent||"").replace(/\s+/g," ");
    plainCache.set(html,text);
    return text;
  }
  function match(page) {
    if (filter!=="all" && page.type!==filter) return false;
    if (chapter && (page.type!=="topic" || page.chapter!==chapter)) return false;
    const words=search.toLocaleLowerCase().normalize("NFKC").trim().split(/\s+/).filter(Boolean);
    if (!words.length) return true;
    const corpus=[page.title,page.summary,page.framework||"",
      (page.keywords||[]).join(" "),plainText(page.html||"")].join(" ")
      .normalize("NFKC").toLocaleLowerCase();
    return words.every(w=>corpus.includes(w));
  }
  function renderListing() {
    const matches=pages.filter(match);
    const list=$("item-list");list.replaceChildren();
    if (chapter) {
      const c=chapters.get(chapter);
      $("landing").querySelector("h1").textContent=c?.title||"知识目录";
      $("landing").querySelector(".intro").textContent=c?.description||"";
      $("landing").querySelector(".overline").textContent="CHAPTER / "+chapter.toUpperCase();
    } else if (search) {
      $("landing").querySelector("h1").textContent="搜索结果";
      $("landing").querySelector(".intro").textContent="输入关键词即可查找基础解释、案例和框架实现。";
      $("landing").querySelector(".overline").textContent="SEARCH";
    } else {
      $("landing").querySelector("h1").textContent="从原理到工程决策";
      $("landing").querySelector(".intro").textContent=
        "理解核心机制，找到实际问题的原因，再看主流框架如何实现。内容按知识关联组织，随时可以从任何一处进入。";
      $("landing").querySelector(".overline").textContent="RL / TRAINING SYSTEMS";
    }
    $("results-label").textContent=matches.length+" 篇";
    if (!matches.length) list.append(el("p","no-results","没有匹配内容。试试更短的关键词。"));
    for (const page of matches) {
      const a=el("a","item");a.href=href(page.type,page.id);
      a.append(el("div","item-meta",types[page.type].label+" / "+sectionName(page)));
      a.append(el("h2","item-title",page.title));
      a.append(el("p","item-description",page.summary));
      const n=relatedItems(page).length;
      if (n>0) a.append(el("div","item-related","关联 "+n+" 篇"));
      list.append(a);
    }
    updateCurrent();
  }
  function renderDetail(type, id) {
    const page=indexed.get(type+":"+id);
    if (!page) {navigateHome();return;}
    $("landing").hidden=true;$("detail").hidden=false;
    $("detail-context").textContent=types[type].label+" / "+sectionName(page);
    $("detail-title").textContent=page.title;
    $("detail-summary").textContent=page.summary;
    $("article").innerHTML=page.html || "<p>暂无正文，请查看关联内容。</p>";
    const recall=$("recall");recall.replaceChildren();
    recall.hidden=!(type==="topic" && page.recall?.question);
    if (!recall.hidden) {
      recall.append(el("p","recall-label","想一想 · 不用写答案"));
      recall.append(el("p","question",page.recall.question));
      const details=el("details");
      details.append(el("summary","","看解释"));
      details.append(el("p","answer",page.recall.answer));
      if (page.recall.spoken) details.append(el("p","spoken",page.recall.spoken));
      recall.append(details);
    }
    const related=relatedItems(page), box=$("related-list");box.replaceChildren();
    $("related").hidden=related.length===0;
    for (const target of related) {
      const a=el("a","related-link");a.href=href(target.type,target.id);
      a.append(el("span","related-type",types[target.type].label));
      a.append(el("span","",target.title));
      box.append(a);
    }
    document.title=page.title+" · RL Infra";
    if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([$("article"),recall]).catch(()=>{});
    updateCurrent();
  }
  function navigateHome() {
    window.location.hash=savedLanding||"";
    if (!location.hash) render();
  }
  function closeMenu() {
    $("side").classList.remove("show");
    $("menu-toggle").setAttribute("aria-expanded","false");
  }
  function render() {
    const p=params();
    closeMenu();
    let type=null,id=null;
    for (const candidate of Object.keys(types)) {
      if (p.has(candidate)) {type=candidate;id=p.get(candidate);break;}
    }
    if (type) {
      if (!indexed.has(type+":"+id)) {history.replaceState(null,"",location.pathname);return render();}
      renderDetail(type,id);
    } else {
      $("detail").hidden=true;$("landing").hidden=false;
      chapter=chapters.has(p.get("chapter"))?p.get("chapter"):"";
      filter=types[p.get("category")]?p.get("category"):"all";
      savedLanding=location.hash;
      renderListing();
      document.title="RL Infra · 技术知识库";
    }
    window.scrollTo({top:0,behavior:"instant"});
  }
  renderChapters();
  $("search").addEventListener("input",function(){
    search=this.value.trim();
    if (chapter) { chapter=""; history.replaceState(null,"",location.pathname); }
    if (!$("landing").hidden) renderListing();
    else {
      location.hash=search ? "#category=all" : (savedLanding||"");
      render();
    }
  });
  document.querySelectorAll("[data-filter]").forEach(b=>b.addEventListener("click",()=>{
    filter=b.dataset.filter;
    chapter="";
    window.location.hash=filter==="all"?"":"#category="+filter;
    if (!location.hash) renderListing();
  }));
  $("back").addEventListener("click",navigateHome);
  $("home-link").addEventListener("click",()=>{
    search="";$("search").value="";filter="all";chapter="";
  });
  $("menu-toggle").addEventListener("click",()=>{
    const open=$("side").classList.toggle("show");
    $("menu-toggle").setAttribute("aria-expanded",String(open));
  });
  $("chapter-list").addEventListener("click",()=>{
    closeMenu();
    if (!$("detail").hidden) { filter="all"; }
  });
  document.addEventListener("keydown",e=>{
    if (e.key==="/" && !["INPUT","TEXTAREA"].includes(document.activeElement?.tagName)
       && !e.altKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();$("search").focus();
    }
    if (e.key==="Escape" && document.activeElement===$("search")) $("search").blur();
  });
  window.addEventListener("hashchange",render);
  render();
})();