"use strict";

/* Educational numerics only. No network calls, analytics, or score uploads.
 * p=σ(z), rewards in {0,1}; all outputs describe a toy model, not an LLM run.
 */
(() => {
  const $ = (id) => document.getElementById(id);
  const svgNS = "http://www.w3.org/2000/svg";
  const labs = new Map([
    [1, ["POLICY / EXPECTED RETURN", "采样噪声会骗过你的直觉吗？",
      "模型的期望 Reward 与少量实际采样结果未必一致。改变策略概率和采样次数，观察误差。"]],
    [2, ["POLICY / GRADIENT ESTIMATOR", "Baseline 怎么改变采样梯度的波动？",
      "从同一个期望梯度出发，调节 Baseline。看清什么变了、什么没有变。"]],
    [3, ["TOKEN / TEACHER FORCING", "Logits 究竟预测哪个 Token？",
      "只改一格索引，不改 Tensor Shape。观察响应 Action Mask 和条件概率预测者的位置。"]],
    [4, ["PPO / CLIPPED SURROGATE", "PPO 的梯度平台在哪一侧？",
      "正负 Advantage 都使用同一个 Clip，却只有一边的 Ratio 超界会被截断。"]],
    [5, ["GRPO / GROUP ADVANTAGE", "GRPO 的 Group 何时真正有学习信号？",
      "改掉一条 Reward，观察均值、标准差、Advantage 和完整组契约的变化。"]]
  ]);
  const dayFromUrl = Number(new URLSearchParams(window.location.search).get("day"));
  let currentDay = labs.has(dayFromUrl) ? dayFromUrl : 1;

  function fixed(x, digits=3) {
    return Number.isFinite(x) ? x.toFixed(digits) : "—";
  }
  function set(id, value) {
    $(id).textContent = value;
  }
  function node(name, properties) {
    const e = document.createElementNS(svgNS, name);
    Object.entries(properties || {}).forEach(([k, v]) => e.setAttribute(k, String(v)));
    return e;
  }
  function svgText(svg, x, y, text, anchor="middle") {
    const e = node("text", {"x":x,"y":y,"class":"axis-text","text-anchor":anchor});
    e.textContent = text;
    svg.append(e);
  }
  function svgLine(svg, x1,y1,x2,y2, cls="axis") {
    svg.append(node("line",{"x1":x1,"y1":y1,"x2":x2,"y2":y2,"class":cls}));
  }
  function chooseDay(day, updateUrl=true) {
    if (!labs.has(day)) return;
    currentDay = day;
    for (const [id, information] of labs) {
      $("lab-" + id).hidden = id !== day;
      const button = document.querySelector('[data-choose-lab="' + id + '"]');
      button.setAttribute("aria-current", String(id === day));
      if (id === day) {
        set("lab-kicker", "LAB " + String(id).padStart(2,"0") + " / " + information[0]);
        set("lab-title", information[1]);
        set("lab-description", information[2]);
      }
    }
    $("back-to-lesson").href = "../#day=" + day + "&view=lesson";
    $("back-to-lesson").textContent = "← 返回 Day " + String(day).padStart(2, "0") + " 讲义";
    document.title = "Lab " + day + " · " + labs.get(day)[1] + " | RL Infra";
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.searchParams.set("day", String(day));
      history.replaceState({}, "", url);
    }
  }
  function seeded(seed) {
    let s = (Math.floor(seed) >>> 0) || 1;
    return () => {
      s += 0x6D2B79F5;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Day 01: true J/gradient vs a reproducible finite-sample estimate.
  let sampleSeed = 17;
  function updateBandit() {
    const p = Number($("bandit-p").value);
    const n = Number($("bandit-n").value);
    const draw = seeded(sampleSeed);
    let success = 0;
    for (let k=0; k<n; k++) if (draw() < p) success++;
    const observed = success / n;
    set("bandit-p-text", fixed(p, 2));
    set("bandit-expected", fixed(p));
    set("bandit-grad", fixed(p*(1-p),4));
    set("bandit-observed", fixed(observed));
    set("bandit-pct", fixed(p*100,1) + "%");
    set("bandit-observed-pct", fixed(observed*100,1) + "%");
    $("bandit-expected-bar").style.width = (100*p) + "%";
    $("bandit-observed-bar").style.width = (100*observed) + "%";
    set("bandit-count", "本次 " + n + " 次采样，A 出现 " + success + " 次；样本均值和理论期望的差是 " + fixed(observed-p,3) + "。");
    set("bandit-pass", "同一 p，若独立采样四次：Pass@4 = 1−(1−p)⁴ = " +
      fixed(1-Math.pow(1-p,4),4) + "；单次 Pass@1 仍是 " + fixed(p,4) + "。");
  }
  $("bandit-p").addEventListener("input",updateBandit);
  $("bandit-n").addEventListener("change",updateBandit);
  $("bandit-roll").addEventListener("click",()=>{sampleSeed++;updateBandit();});

  // Day 02: g_A = (1-b)(1-p), g_B = b*p; E is p(1-p).
  function updateBaseline() {
    const p = Number($("baseline-p").value), b=Number($("baseline-b").value);
    const gA=(1-b)*(1-p), gB=b*p, mean=p*gA+(1-p)*gB;
    const variance=p*(gA-mean)**2+(1-p)*(gB-mean)**2;
    set("baseline-p-text",fixed(p,2));
    set("baseline-b-text",fixed(b,2));
    set("baseline-ga",fixed(gA,4));
    set("baseline-gb",fixed(gB,4));
    set("baseline-variance",fixed(variance,6));
    set("baseline-expect","不论 b 怎样变化，这个动作无关 Baseline 的期望梯度都等于 " +
      fixed(mean,4) + "，与解析梯度 p(1−p) = " + fixed(p*(1-p),4) + " 对齐。本例方差最小的 b* = 1−p = " + fixed(1-p,2) + "。");
    const svg=$("baseline-plot");svg.replaceChildren();
    const min=Math.min(-0.2,gA,gB,mean),max=Math.max(0.7,gA,gB,mean);
    const x = (v) => 46 + (v-min)/(max-min)*474;
    svgLine(svg,46,145,520,145);
    for(let i=0;i<=4;i++){
      const v=min+(max-min)*i/4;
      svgLine(svg,x(v),140,x(v),148);
      svgText(svg,x(v),163,fixed(v,2));
    }
    svgLine(svg,x(mean),26,x(mean),141,"plot-mean");
    svgText(svg,x(mean),17,"E[g] = "+fixed(mean,3));
    for(const [y,v,label] of [[70,gA,"抽到 A"],[116,gB,"抽到 B"]]){
      svgLine(svg,x(0),y,x(v),y,"plot-data");
      svg.append(node("circle",{cx:x(v),cy:y,r:6,"class":"plot-dot"}));
      svgText(svg,40,y+5,label,"end");
    }
  }
  $("baseline-p").addEventListener("input",updateBaseline);
  $("baseline-b").addEventListener("input",updateBaseline);
  $("baseline-optimal").addEventListener("click",()=>{
    $("baseline-b").value=(1-Number($("baseline-p").value)).toFixed(2);
    updateBaseline();
  });

  // Day 03: labels and predictors are deliberately separate axis mappings.
  function updateShift() {
    const p=Number($("shift-p").value);
    const wrong=$("shift-wrong").checked;
    const tokens=["〈s〉","2","+","2","=","4","〈pad〉"];
    const valid=6, board=$("shift-table");
    set("shift-p-text",p);
    board.replaceChildren();
    const table=document.createElement("table");
    const tbody=document.createElement("tbody");
    const rows=[
      ["Input 索引",...tokens.map((_,i)=>i)],
      ["Token",...tokens],
      ["类别",...tokens.map((_,i)=>i>=valid?"PAD":i<p?"Prompt":"Action")],
      ["用于预测它的 Logits",...tokens.map((_,i)=>i===0||i>=valid?"—":i>=p?(wrong?i:i-1):"·")],
      ["Action Mask",...tokens.map((_,i)=>i>=p&&i<valid?"1":"0")]
    ];
    rows.forEach((row,index)=>{
      const tr=document.createElement("tr");
      row.forEach((item,i)=>{
        const cell=document.createElement(i===0?"th":"td");
        cell.textContent=String(item);
        if(i>0) {
          const tokenIndex=i-1,action=tokenIndex>=p&&tokenIndex<valid;
          if(action) cell.classList.add(wrong?"wrong-token":"action-token");
          if(tokenIndex>=valid)cell.classList.add("pad-token");
        }
        tr.append(cell);
      });
      tbody.append(tr);
    });
    table.append(tbody);board.append(table);
    set("shift-summary",wrong
      ?"当前错误：Action 的目标位置 t 被同位置 logits[t] 预测；虽然 Shape 未变，但不是“下一个 Token”的条件概率。"
      :"当前正确：每个 Action token[t] 由 logits[t−1] 预测；参与 Loss 的有效位置有 "+(valid-p)+" 个。");
  }
  $("shift-p").addEventListener("input",updateShift);
  $("shift-wrong").addEventListener("change",updateShift);

  // Day 04: exact piecewise PPO clipped surrogate in Ratio r for a single sample.
  function ppoValue(r,a,eps) {
    const clipped=Math.max(1-eps,Math.min(1+eps,r));
    return Math.min(r*a,clipped*a);
  }
  function updatePPO() {
    const a=Number($("ppo-a").value),eps=Number($("ppo-eps").value),r=Number($("ppo-r").value);
    const value=ppoValue(r,a,eps),unclipped=r*a;
    const plateau=a>0 ? r>1+eps+1e-9 : r<1-eps-1e-9;
    const kink=Math.abs(r-(a>0?1+eps:1-eps))<1e-8;
    const derivative=kink?"分段边界":plateau?"0":fixed(a,2);
    set("ppo-eps-text",fixed(eps,2));set("ppo-r-text",fixed(r,2));
    set("ppo-unclipped",fixed(unclipped,3));set("ppo-value",fixed(value,3));set("ppo-derivative",derivative);
    set("ppo-explain",plateau?"这条已采样 Action 进入 clipped surrogate 的平坦区；其他样本或损失项仍能改变共享参数。":
      kink?"当前在 clipped surrogate 的分段边界；这里不报告唯一的常规导数。":
      "当前使用原始 r·A 分支，单个动作的 surrogate 对 Ratio 仍有方向为 "+(a>0?"正":"负")+" 的局部斜率。");
    const svg=$("ppo-plot");svg.replaceChildren();
    const left=42,right=568,top=20,bottom=201,rlo=.35,rhi=1.8;
    const vals=Array.from({length:241},(_,i)=>ppoValue(rlo+(rhi-rlo)*i/240,a,eps));
    const ymin=Math.min(...vals,0),ymax=Math.max(...vals,0),span=Math.max(1,ymax-ymin);
    const X=(v)=>left+(v-rlo)/(rhi-rlo)*(right-left);
    const Y=(v)=>bottom-(v-ymin+span*.08)/(span*1.16)*(bottom-top);
    svgLine(svg,left,bottom,right,bottom);
    svgLine(svg,left,top,left,bottom);
    for(let step=0;step<4;step++){
      const xval=rlo+(rhi-rlo)*step/3;
      svgLine(svg,X(xval),bottom,X(xval),bottom+4);
      svgText(svg,X(xval),bottom+19,fixed(xval,2));
    }
    for(const bound of [1-eps,1+eps]){
      svgLine(svg,X(bound),top,X(bound),bottom,"plot-clip");
      svgText(svg,X(bound),13,fixed(bound,2));
    }
    if(ymin<=0&&ymax>=0)svgLine(svg,left,Y(0),right,Y(0),"axis");
    svg.append(node("polyline",{points:vals.map((v,i)=>X(rlo+(rhi-rlo)*i/240)+","+Y(v)).join(" "),
      "class":"plot-data","fill":"none"}));
    svgLine(svg,X(r),top,X(r),bottom,"plot-mean");
    svg.append(node("circle",{cx:X(r),cy:Y(value),r:6,"class":"plot-dot"}));
    svgText(svg,left+8,top+16,"J", "start");
    svgText(svg,right,bottom+32,"Ratio r","end");
  }
  ["ppo-a","ppo-eps","ppo-r"].forEach(k=>$(k).addEventListener(k==="ppo-a"?"change":"input",updatePPO));

  // Day 05: standardize the received group with population std and positive eps.
  let rewards=[1,0,1,0];
  function updateGRPO() {
    const incomplete=$("grpo-incomplete").checked,active=incomplete?3:4;
    const values=rewards.slice(0,active),mean=values.reduce((a,b)=>a+b,0)/active;
    const variance=values.reduce((acc,r)=>acc+(r-mean)**2,0)/active;
    const std=Math.sqrt(variance),advantages=values.map(r=>(r-mean)/(std+1e-8));
    set("grpo-mean",fixed(mean,4));
    set("grpo-std",fixed(std,4));
    set("grpo-count",active+"/4");
    document.querySelectorAll("[data-reward-index]").forEach((btn,i)=>{
      btn.querySelector("strong").textContent=rewards[i];
      btn.setAttribute("aria-pressed",String(rewards[i]===1));
      btn.disabled=incomplete&&i===3;
    });
    const rows=$("grpo-advantages");rows.replaceChildren();
    advantages.forEach((v,i)=>{
      const row=document.createElement("div");row.className="advantage-row";
      const name=document.createElement("b");name.textContent="T"+(i+1);
      const value=document.createElement("span");value.textContent=(v>=0?"+":"")+fixed(v,3);
      const track=document.createElement("div");track.className="advantage-track";
      const fill=document.createElement("span");
      fill.className="advantage-fill "+(v<0?"negative":"positive");
      fill.style.width=Math.min(50,Math.abs(v)*25)+"%";
      track.append(fill);row.append(name,value,track);rows.append(row);
    });
    set("grpo-explain",incomplete?
      "未满足预定 G=4：当前展示的是前三条的临时 G=3 统计，不能冒充完整四样本组的目标。"+
      (std<1e-9?"本次三个 Reward 完全相同，没有组内相对差异。":""):
      std<1e-9?"四条 Reward 相同。组内 Advantage 全为零；这类 Group 对该相对 PG 项没有区分度。":
      "这是一个完整的四样本 Group；Reward 的顺序会影响哪条 Trajectory 得到正负 Advantage，不能按完成顺序乱贴。");
  }
  document.querySelectorAll("[data-reward-index]").forEach(btn=>btn.addEventListener("click",()=>{
    const i=Number(btn.dataset.rewardIndex);
    if(i===3&&$("grpo-incomplete").checked)return;
    rewards[i]=1-rewards[i];updateGRPO();
  }));
  $("grpo-incomplete").addEventListener("change",updateGRPO);
  $("grpo-reset").addEventListener("click",()=>{
    rewards=[1,0,1,0];$("grpo-incomplete").checked=false;updateGRPO();
  });

  document.querySelectorAll("[data-choose-lab]").forEach(btn=>btn.addEventListener("click",()=>{
    chooseDay(Number(btn.dataset.chooseLab));
    document.getElementById("lab-title").scrollIntoView({behavior:"smooth",block:"start"});
  }));
  window.addEventListener("popstate",()=>{
    const d=Number(new URLSearchParams(location.search).get("day"));
    if(labs.has(d))chooseDay(d,false);
  });
  updateBandit();updateBaseline();updateShift();updatePPO();updateGRPO();
  chooseDay(currentDay,false);
})();
