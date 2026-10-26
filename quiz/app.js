"use strict";
(function () {
  const data = JSON.parse(document.getElementById("quiz-data").textContent);
  const questions = data.questions;
  const byId = new Map(questions.map(q => [q.id, q]));
  const storeKey = "rl-infra-100days-quiz-v1";
  const $ = id => document.getElementById(id);
  const labels = {
    policy:"策略梯度 / PPO",reward:"奖励与偏好",grpo:"GRPO / DAPO",opd:"OPD",
    async:"异步训练",agent:"Agentic RL",inference:"推理与一致性",
    distributed:"分布式 / MoE",infra:"系统架构",debug:"排障",precision:"低精度"
  };
  let records = {};
  try {
    const saved = JSON.parse(localStorage.getItem(storeKey) || "{}");
    if (saved && typeof saved === "object" && !Array.isArray(saved)) records = saved;
  } catch (_) {}
  let session = [], pos = 0, activeId = null, mode = "sequential", showing = false;
  let source = "deep", timer = null;

  function blank() {
    return {answer:"",note:"",rating:"",due:"",attempts:0,streak:0,bookmarked:false,lastAt:""};
  }
  function get(id) { return records[id] || blank(); }
  function writable(id) { if (!records[id]) records[id] = blank(); return records[id]; }
  function pad(n) { return String(n).padStart(2,"0"); }
  function dateKey(offset=0) {
    const d = new Date(); d.setHours(12,0,0,0); d.setDate(d.getDate()+offset);
    return d.getFullYear()+"-"+pad(d.getMonth()+1)+"-"+pad(d.getDate());
  }
  function due(r) { return !!r.due && r.due<=dateKey(); }
  function practiced(r) { return r.attempts>0; }
  function weak(r) { return r.rating==="again"||r.rating==="hard"; }
  function mastered(r) { return r.rating==="good"||r.rating==="easy"; }
  function persist() {
    try { localStorage.setItem(storeKey,JSON.stringify(records)); $("save-indicator").textContent="已保存在本地"; }
    catch(_) { $("save-indicator").textContent="本地保存失败，请导出记录"; }
    stats();
  }
  function flush() {
    if (timer !== null) { clearTimeout(timer); timer=null; }
    if (!activeId) return;
    const r=writable(activeId); r.answer=$("answer").value; r.note=$("note").value;
    persist();
  }
  function scheduleSave() {
    if (timer!==null) clearTimeout(timer);
    timer=setTimeout(()=>{timer=null;flush();},250);
  }
  function stats() {
    let seen=0,known=0,pending=0,saved=0;
    questions.forEach(q=>{const r=get(q.id);if(practiced(r))seen++;if(mastered(r))known++;if(due(r))pending++;if(r.bookmarked)saved++;});
    $("seen").textContent=seen;$("mastered").textContent=known;$("due").textContent=pending;$("saved").textContent=saved;
    $("progress").style.width=100*seen/questions.length+"%";
  }
  function option(select,label,value) {
    const x=document.createElement("option");x.textContent=label;x.value=value;select.append(x);
  }
  function initFilters() {
    const topics=new Map(questions.map(q=>[q.domain,labels[q.domain]||q.domain]));
    topics.forEach((label,key)=>option($("topic"),label,key));
    [...new Set(questions.map(q=>q.chapter))].forEach(x=>option($("chapter"),x,x));
  }
  function normalize(s) {
    return String(s||"").normalize("NFKC").replace(/[？?。，,.：:；;、\s]/g,"").toLowerCase();
  }
  function filterIds() {
    const term=$("search").value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const topic=$("topic").value,chapter=$("chapter").value,level=$("level").value,status=$("status").value;
    return questions.filter(q=>{
      const r=get(q.id);
      if(topic!=="all"&&q.domain!==topic)return false;
      if(chapter!=="all"&&q.chapter!==chapter)return false;
      if(level!=="all"&&q.level!==level)return false;
      if(status==="new"&&practiced(r))return false;
      if(status==="weak"&&!weak(r))return false;
      if(status==="due"&&!due(r))return false;
      if(status==="mastered"&&!mastered(r))return false;
      if(status==="saved"&&!r.bookmarked)return false;
      const hay=[q.code,q.question,q.doc_title,q.brief,q.deep,q.chapter,q.followup].join(" ").toLowerCase();
      return term.every(x=>hay.includes(x));
    }).map(q=>q.id);
  }
  function shuffle(arr) {
    const a=[...arr];
    for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}
    return a;
  }
  function setMode(next,focusId=null) {
    flush();mode=next;
    const filtered=filterIds();
    $("match-count").textContent=filtered.length;
    session=next==="random"?shuffle(filtered).slice(0,10):next==="due"?filtered.filter(id=>due(get(id))):next==="wrong"?filtered.filter(id=>weak(get(id))):filtered;
    pos=focusId&&session.includes(focusId)?session.indexOf(focusId):0;
    document.querySelectorAll("[data-mode]").forEach(btn=>btn.classList.toggle("active",btn.dataset.mode===next));
    $("session-label").textContent=({sequential:"顺序练习",random:"随机十题",due:"今日复习",wrong:"错题重练"})[next]+" · "+session.length+" 题";
    display();
  }
  function setVisibility(show) {
    showing=show;$("reference").hidden=!show;
    $("reveal").textContent=show?"收起参考答案":"查看参考答案";
    $("reveal").setAttribute("aria-expanded",String(show));
    if(show)showSource(source);
  }
  function showSource(name) {
    source=name;
    document.querySelectorAll("[data-source]").forEach(x=>x.classList.toggle("active",x.dataset.source===name));
    const q=byId.get(activeId);
    if(!q)return;
    $("explanation").textContent=(name==="deep"?q.deep:q.brief)
      .replace(/\\([_\[\]{}|])/g,"$1").replace(/\*\*/g,"");
    $("followup").textContent=q.followup.replace(/\*\*/g,"");
  }
  function display() {
    setVisibility(false);
    if(session.length===0) {
      activeId=null;$("question-title").textContent="当前筛选没有题目";$("qid").textContent="—";$("topic-label").textContent="";$("difficulty").textContent="";
      $("answer").value="";$("note").value="";$("position").textContent="0 / 0";
      ["answer","note","prev","next","star","reveal","copy"].forEach(k=>$(k).disabled=true);
      $("lesson-link").hidden=true;return;
    }
    pos=Math.min(Math.max(0,pos),session.length-1);
    activeId=session[pos]; const q=byId.get(activeId),r=get(activeId);
    ["answer","note","prev","next","star","reveal","copy"].forEach(k=>$(k).disabled=false);
    $("qid").textContent=q.code;$("topic-label").textContent=q.chapter;$("difficulty").textContent=q.level;
    $("question-title").textContent=q.doc_title||q.question;
    const alias=normalize(q.doc_title)!==normalize(q.question);
    $("alias").hidden=!alias;$("alias").textContent=alias?"另一份材料的问法："+q.question:"";
    $("answer").value=r.answer||"";$("note").value=r.note||"";
    $("char-count").textContent=$("answer").value.trim().length+" 字";
    $("star").setAttribute("aria-pressed",String(!!r.bookmarked));$("star").textContent=r.bookmarked?"★ 已收藏":"☆ 收藏";
    $("position").textContent=(pos+1)+" / "+session.length;
    $("prev").disabled=pos===0;$("next").disabled=pos===session.length-1;
    $("lesson-link").hidden=false;$("lesson-link").href="../#day="+q.course_day+"&view=lesson";
    $("lesson-link").textContent="关联 Day "+pad(q.course_day)+" ↗";
    $("save-indicator").textContent=practiced(r)?"已自评 "+r.attempts+" 次":"草稿仅本地保存";
    history.replaceState(null,"","#q="+q.id);
    stats();
  }
  function move(delta) {
    if(!activeId||pos+delta<0||pos+delta>=session.length)return;
    flush();pos+=delta;display();$("question-title").focus({preventScroll:true});
    window.scrollTo({top:0,behavior:"smooth"});
  }
  function rate(kind) {
    if(!activeId||!["again","hard","good","easy"].includes(kind))return;
    flush();
    const r=writable(activeId);
    r.streak=(kind==="good"||kind==="easy")?(r.streak||0)+1:0;
    const intervals=kind==="good"?[3,7,14,30,60]:[7,14,30,60,90];
    const days=kind==="again"?0:kind==="hard"?1:intervals[Math.min(r.streak-1,intervals.length-1)];
    r.rating=kind;r.due=dateKey(days);r.attempts=(r.attempts||0)+1;r.lastAt=new Date().toISOString();
    persist();
    if(pos<session.length-1)move(1);
    else {$("save-indicator").textContent="本轮已到最后一题；可切换练习模式";display();}
  }
  async function copyQuestion() {
    if(!activeId)return;
    const q=byId.get(activeId);
    try {await navigator.clipboard.writeText(q.code+" "+(q.doc_title||q.question));$("save-indicator").textContent="题目已复制";}
    catch(_) {$("save-indicator").textContent="复制不可用，请手动选择题目";}
  }
  function exportData(){
    flush();
    const blob=new Blob([JSON.stringify({kind:"RL_INFRA_QUIZ_BACKUP",version:1,date:dateKey(),records},null,2)],{type:"application/json"});
    const u=URL.createObjectURL(blob),a=document.createElement("a");
    a.href=u;a.download="rl-infra-quiz-"+dateKey()+".json";document.body.append(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(u),2000);
  }
  async function importData(file) {
    if(!file||file.size>5e6){alert("请选择不超过 5MB 的练习备份 JSON");return;}
    let input;try{input=JSON.parse(await file.text());if(input.kind!=="RL_INFRA_QUIZ_BACKUP"||input.version!==1||typeof input.records!=="object"||!input.records)throw Error("bad format");}
    catch(_) {alert("文件不是本题库的导出备份");return;}
    if(!confirm("导入将覆盖同题的本地记录。继续？"))return;
    flush();
    for(const [id,data] of Object.entries(input.records)){
      if(!byId.has(Number(id))||!data||typeof data!=="object")continue;
      records[id]={answer:String(data.answer||"").slice(0,50000),note:String(data.note||"").slice(0,12000),
        rating:["again","hard","good","easy"].includes(data.rating)?data.rating:"",
        due:/^\d{4}-\d{2}-\d{2}$/.test(data.due||"")?data.due:"",
        attempts:Math.max(0,Math.min(10000,Number(data.attempts)||0)),
        streak:Math.max(0,Math.min(1000,Number(data.streak)||0)),
        bookmarked:!!data.bookmarked,lastAt:String(data.lastAt||"")};
    }
    persist();setMode("sequential",activeId);$("save-indicator").textContent="备份已导入";
    $("import-file").value="";
  }
  function init() {
    initFilters();
    ["search","topic","chapter","level","status"].forEach(k=>$(k).addEventListener(k==="search"?"input":"change",()=>setMode("sequential")));
    $("clear").addEventListener("click",()=>{["topic","chapter","level","status"].forEach(k=>$(k).value="all");$("search").value="";setMode("sequential");});
    document.querySelectorAll("[data-mode]").forEach(btn=>btn.addEventListener("click",()=>setMode(btn.dataset.mode)));
    $("answer").addEventListener("input",()=>{$("char-count").textContent=$("answer").value.trim().length+" 字";scheduleSave();});
    $("note").addEventListener("input",scheduleSave);
    $("star").addEventListener("click",()=>{if(!activeId)return;const r=writable(activeId);r.bookmarked=!r.bookmarked;persist();$("star").setAttribute("aria-pressed",String(r.bookmarked));$("star").textContent=r.bookmarked?"★ 已收藏":"☆ 收藏";});
    $("reveal").addEventListener("click",()=>setVisibility(!showing));
    document.querySelectorAll("[data-source]").forEach(btn=>btn.addEventListener("click",()=>showSource(btn.dataset.source)));
    $("copy").addEventListener("click",copyQuestion);
    $("prev").addEventListener("click",()=>move(-1));$("next").addEventListener("click",()=>move(1));
    document.querySelectorAll("[data-rate]").forEach(btn=>btn.addEventListener("click",()=>rate(btn.dataset.rate)));
    $("export").addEventListener("click",exportData);
    $("import").addEventListener("click",()=>$("import-file").click());
    $("import-file").addEventListener("change",e=>importData(e.target.files[0]));
    window.addEventListener("beforeunload",flush);
    window.addEventListener("hashchange",()=>{const match=location.hash.match(/^#q=(\d+)$/);if(match&&byId.has(Number(match[1]))&&activeId!==Number(match[1]))setMode("sequential",Number(match[1]));});
    document.addEventListener("keydown",e=>{const tag=document.activeElement?.tagName;if(["INPUT","SELECT","TEXTAREA"].includes(tag)||e.ctrlKey||e.altKey||e.metaKey)return;if(e.key==="ArrowRight")move(1);if(e.key==="ArrowLeft")move(-1);});
    const match=location.hash.match(/^#q=(\d+)$/);
    setMode("sequential",match&&byId.has(Number(match[1]))?Number(match[1]):null);
  }
  init();
})();
