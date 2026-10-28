/* Public, read-only knowledge reader. Progress is kept in localStorage only. */
(function () {
  'use strict';
  const C = window.RLCore, X = window.RLExperiments;
  const data = JSON.parse(document.getElementById('knowledge-data').textContent);
  const graph = C.index(data), meta = data.learning_map || {}, $ = id => document.getElementById(id);
  const knownIds = new Set(graph.questions.map(q => q.id));
  const labels = {topic: '知识', case: '工程问题', implementation: '框架实现', question: '面试题'};
  let current = C.route(location.hash), lastList = '#view=questions', renderId = 0, searchTimer = null;
  const state = {storageError: ''};
  const el = (tag, cls, text) => {const out = document.createElement(tag); if (cls) out.className = cls; if (text !== undefined) out.textContent = text; return out;};
  const a = (text, values, cls) => {const link = el('a', cls, text); link.href = typeof values === 'string' ? values : C.url(values); return link;};
  const button = (text, fn, cls = '') => {const b = el('button', cls, text); b.type = 'button'; b.addEventListener('click', fn); return b;};
  const values = p => Object.fromEntries(p.entries());
  const detailURL = item => ({[item.type]: item.code || item.id});
  const sectionOf = item => graph.sections.get(graph.topicSection.get(item.id));
  function notice(text) {$('notice').textContent = text || '';}
  function save(valuesToSet, replace = false, scroll = true) {
    const hash = C.url(valuesToSet);
    if (replace) {history.replaceState(null, '', hash); route(scroll);}
    else if (location.hash === hash) route(scroll);
    else location.hash = hash;
  }
  function field(label, choices, value, onChange, name) {
    const wrap = el('label', 'field'); wrap.append(el('span', '', label));
    const select = el('select'); if (name) select.dataset.filter = name;
    choices.forEach(([v, t]) => {const option = el('option', '', String(t)); option.value = String(v); select.append(option);});
    select.value = value || ''; select.addEventListener('change', () => onChange(select.value)); wrap.append(select); return wrap;
  }
  function head(host, title, description) {host.append(el('h1', '', title)); if (description) host.append(el('p', 'lead', description));}
  function sectionHead(host, title, trailing) {
    const h = el('div', 'section-heading'); h.append(el('h2', '', title)); if (trailing) h.append(trailing); host.append(h);
  }
  function row(item, options = {}) {
    const link = a('', detailURL(item), 'entry');
    const marker = item.type === 'question' ? item.code : labels[item.type];
    link.append(el('span', 'entry-meta', marker), el('strong', '', item.title));
    if (options.summary && item.type !== 'question') link.append(el('p', '', item.summary));
    if (item.type === 'question') link.append(el('span', 'caption', item.chapter + ' / ' + item.level));
    return link;
  }
  function sectionItems(section) {return graph.items.filter(item => graph.inSection(item, section.id));}
  function reveal(view, scroll) {
    document.querySelectorAll('.view').forEach(v => {v.hidden = v.id !== view;});
    $('mobile-directory').close();
    document.querySelectorAll('[data-nav]').forEach(n => {
      const active = (n.dataset.nav === 'map' && ['home', 'chapter', 'detail'].includes(view)) || n.dataset.nav === view;
      if (active) n.setAttribute('aria-current', 'page'); else n.removeAttribute('aria-current');
    });
    if (scroll) window.scrollTo({top: 0, behavior: 'instant'});
  }
  function tree(host) {
    host.replaceChildren();
    for (const chapter of data.chapters) {
      const group = el('details', 'tree-group'); group.dataset.area = chapter.id;
      const summary = el('summary', '', chapter.title); group.append(summary, a('概览', {chapter: chapter.id}, 'tree-overview'));
      for (const section of chapter.sections || []) {
        const sub = el('div', 'tree-section'); sub.append(a(section.title, {chapter: chapter.id, section: section.id}, 'tree-section-title'));
        for (const id of section.topics) {
          const topic = graph.lookup.get('topic:' + id); if (!topic) continue;
          const link = a(topic.title, {topic: id}, 'tree-topic'); link.dataset.topic = id; sub.append(link);
        }
        group.append(sub);
      }
      host.append(group);
    }
  }
  function updateTree(item) {
    const area = current.get('chapter') || current.get('area') || (item?.type === 'topic' ? item.chapter : graph.lookup.get('topic:' + item?.topics?.[0])?.chapter);
    document.querySelectorAll('.tree-group').forEach(group => {if (group.dataset.area === area) group.open = true;});
    document.querySelectorAll('[data-topic]').forEach(link => {
      if (item?.type === 'topic' && link.dataset.topic === item.id) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
  }
  function renderHome() {
    const host = $('home'); host.replaceChildren();
    head(host, 'RL 训练系统', '算法、训练引擎、推理引擎与调度。按系统环节找内容\uff0c也可以从具体问题进入。');
    host.append(el('p', 'inventory', graph.entries.length + ' 篇文档 / ' + graph.questions.length + ' 道面试题 / ' + graph.sections.size + ' 个专题'));
    sectionHead(host, '一次训练迭代');
    const flow = el('ol', 'system-flow');
    for (const stage of meta.system_flow || []) {
      const li = el('li'); const link = a('', {topic: stage.topic}); link.append(el('strong', '', stage.title), el('span', '', stage.detail)); li.append(link); flow.append(li);
    }
    host.append(flow, el('p', 'caption', '更新后的权重返回 Rollout。数据依赖与资源条件满足时\uff0c部分阶段可重叠执行。'));
    const cross = el('div', 'link-strip'); for (const [id, label] of [['parallel', '并行与通信'], ['async', '数据供给与版本'], ['diagnosis', '可观测与排障']]) cross.append(a(label, {chapter: id})); host.append(cross);
    sectionHead(host, '阅读路径');
    for (const path of meta.paths || []) {
      const group = el('details', 'learning-path'); group.append(el('summary', '', path.title), el('p', 'caption', path.goal));
      const ol = el('ol'); path.topics.forEach(id => {const item = graph.lookup.get('topic:' + id); if (item) {const li = el('li'); li.append(a(item.title, {topic: id})); ol.append(li);}}); group.append(ol); host.append(group);
    }
    sectionHead(host, '目录');
    for (const [i, chapter] of data.chapters.entries()) {
      const line = el('section', 'area-row');
      line.append(el('span', 'index', String(i + 1).padStart(2, '0')), a(chapter.title, {chapter: chapter.id}, 'area-title'));
      const desc = el('div'); desc.append(el('p', '', chapter.description));
      const topics = graph.entries.filter(x => x.type === 'topic' && x.chapter === chapter.id), questions = graph.questions.filter(x => graph.inChapter(x, chapter.id));
      desc.append(el('span', 'caption', topics.length + ' 篇知识 / ' + questions.length + ' 道关联题'));
      const links = el('div', 'section-links'); (chapter.sections || []).forEach(s => links.append(a(s.title, {chapter: chapter.id, section: s.id}))); desc.append(links); line.append(desc); host.append(line);
    }
  }
  function renderChapter(id) {
    const chapter = graph.chapters.get(id); if (!chapter) return missing();
    const host = $('chapter'); host.replaceChildren(); head(host, chapter.title, chapter.description);
    const selected = current.get('section');
    if (selected && !chapter.sections.some(s => s.id === selected)) return missing();
    for (const s of chapter.sections || []) {
      if (selected && selected !== s.id) continue;
      const section = el('section', 'topic-section'); section.id = 'group-' + s.id;
      const items = sectionItems(s), problems = items.filter(x => ['case', 'question'].includes(x.type)), implementations = items.filter(x => x.type === 'implementation');
      sectionHead(section, s.title, a(problems.length + ' 个问题 →', {view: 'questions', area: id, section: s.id}));
      s.topics.forEach(id => section.append(row(graph.lookup.get('topic:' + id), {summary: true})));
      if (implementations.length) {
        const fold = el('details'); fold.append(el('summary', '', '框架实现 / ' + implementations.length)); implementations.forEach(x => fold.append(row(x))); section.append(fold);
      }
      if (problems.length) {
        const fold = el('details'); fold.append(el('summary', '', '相关问题 / ' + problems.length)); problems.slice(0, 6).forEach(x => fold.append(row(x))); fold.append(a('查看全部问题', {view: 'questions', area: id, section: s.id}, 'more')); section.append(fold);
      }
      host.append(section);
    }
  }
  function reviewState() {
    let result;
    try {result = C.readProgress(localStorage, knownIds);} catch (error) {result = {state: {version: 1, records: {}}, error: error.message};}
    if (result.error) notice('复习记录暂不可读。未覆盖原记录\uff1b请先导出备份。');
    return result;
  }
  function filters(host, kind) {
    const form = el('div', 'filters'); const p = values(current);
    const set = (key, value) => {const next = {...p, [key]: value, page: ''}; if (key === 'area') next.section = ''; save(next, true, false);};
    form.append(field('领域', [['', '全部'], ...data.chapters.map(c => [c.id, c.title])], p.area, v => set('area', v), 'area'));
    const sections = [...graph.sections.values()].filter(s => !p.area || s.chapter === p.area);
    form.append(field('专题', [['', '全部'], ...sections.map(s => [s.id, s.title])], p.section, v => set('section', v), 'section'));
    if (kind === 'questions') {
      form.append(field('类型', [['', '全部'], ['question', '面试题'], ['case', '工程问题']], p.kind, v => set('kind', v), 'kind'));
      form.append(field('难度', [['', '全部'], ['基础', '基础'], ['进阶', '进阶']], p.level, v => set('level', v), 'level'));
      form.append(field('复习', [['', '全部'], ['due', '待复习'], ['weak', '需巩固'], ['bookmarked', '收藏'], ['new', '未自评']], p.review, v => set('review', v), 'review'));
    } else form.append(field('类型', [['', '全部'], ...Object.entries(labels)], p.kind, v => set('kind', v), 'kind'));
    host.append(form);
  }
  function filtered(items, progress) {
    const p = current;
    return graph.search(items, p.get('q') || '').filter(x => {
      if (p.get('kind') && x.type !== p.get('kind')) return false;
      if (!graph.inChapter(x, p.get('area')) || !graph.inSection(x, p.get('section'))) return false;
      if (p.get('ref') && !graph.topicsOf(x).includes(p.get('ref'))) return false;
      if (p.get('level') && x.level !== p.get('level')) return false;
      const mode = p.get('review'), record = progress?.records[x.id] || {};
      if (mode && x.type !== 'question') return false;
      if (mode === 'due') return record.due && record.due <= C.dateAfter();
      if (mode === 'weak') return ['again', 'hard'].includes(record.rating);
      if (mode === 'bookmarked') return !!record.bookmarked;
      if (mode === 'new') return !record.attempts;
      return true;
    });
  }
  function paginate(host, items) {
    const size = 24, pages = Math.max(1, Math.ceil(items.length / size));
    const page = Math.min(pages, Math.max(1, Math.floor(Number(current.get('page'))) || 1));
    host.append(el('p', 'count', items.length + ' 项结果 / ' + page + ' · ' + pages));
    if (!items.length) host.append(el('p', 'empty', '没有匹配内容。可以减少关键词或放宽筛选。'));
    items.slice((page - 1) * size, page * size).forEach(x => host.append(row(x, {summary: true})));
    const pager = el('nav', 'pager'); pager.setAttribute('aria-label', '分页');
    if (page > 1) pager.append(a('上一页', {...values(current), page: page - 1}));
    if (page < pages) pager.append(a('下一页', {...values(current), page: page + 1})); host.append(pager);
  }
  function download(text, name) {
    const href = URL.createObjectURL(new Blob([text], {type: 'application/json'})); const link = a('', href); link.download = name; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(href), 1000);
  }
  function progressControls(host, loaded) {
    const state = loaded.state, records = Object.values(state.records), strip = el('div', 'review-tools');
    const reviewed = records.filter(r => r.attempts).length, due = records.filter(r => r.due && r.due <= C.dateAfter()).length;
    strip.append(el('span', '', '已练习 ' + reviewed + ' / ' + graph.questions.length), a('待复习 ' + due, {view: 'questions', review: 'due'}));
    strip.append(button('导出记录', () => {
      try {
        const raw = localStorage.getItem(C.KEY);
        if (loaded.error) {download(raw || '{}', 'rl-infra-recovery.json'); return;}
        const latest = reviewState(); download(JSON.stringify({format: 'rl-infra-interview-practice-v1', createdAt: new Date().toISOString(), records: latest.state.records}, null, 2), 'rl-infra-practice.json');
      } catch (_) {notice('导出失败\uff0c浏览器不允许访问存储。');}
    }));
    const file = el('input'); file.type = 'file'; file.accept = 'application/json,.json'; file.hidden = true;
    strip.append(button('导入记录', () => file.click()), file);
    file.addEventListener('change', async () => {
      try {
        const selected = file.files[0]; if (!selected) return;
        if (selected.size > 5_000_000) throw new Error('File exceeds 5 MB');
        const incoming = C.validateProgress(JSON.parse(await selected.text()), knownIds), latest = reviewState();
        if (latest.error) throw new Error('Existing local record cannot be read');
        const result = C.mergeProgress(latest.state, incoming); localStorage.setItem(C.KEY, JSON.stringify(result.state));
        route(false); notice('导入 ' + result.added + ' 条\uff1b保留本机已有的 ' + result.conflicts + ' 条记录。');
      } catch (error) {notice('导入未执行\uff0c原记录未改动。 ' + error.message);}
    });
    host.append(strip, el('p', 'caption', '记录只保存在当前浏览器\uff0c不上传。换设备或域名前请导出。导入时保留本机已有记录。'));
  }
  function renderQuestions() {
    const host = $('questions'); host.replaceChildren();
    head(host, '问题', graph.questions.length + ' 道面试题与 ' + data.cases.length + ' 个工程案例。保留原题号、问法、答案和追问。');
    const loaded = reviewState(); progressControls(host, loaded); filters(host, 'questions');
    if (current.get('ref')) host.append(a('取消知识点限定', {...values(current), ref: ''}, 'more'));
    paginate(host, filtered(graph.items.filter(x => ['case', 'question'].includes(x.type)), loaded.state));
  }
  function renderLibrary() {
    const host = $('library'); host.replaceChildren(); head(host, '检索', '搜索标题、正文、完整答案和追问。'); filters(host, 'library'); paginate(host, filtered(graph.items));
  }
  function paragraphs(host, text) {
    String(text || '').split(/\n\s*\n/).forEach(part => host.append(el('p', 'source-text', part.replace(/\\([|\[\]_])/g, '$1'))));
  }
  function renderReview(item) {
    const host = $('question-review'); host.replaceChildren(); host.hidden = item.type !== 'question'; if (host.hidden) return;
    const loaded = reviewState(), r = loaded.state.records[item.id] || C.blankRecord();
    host.append(el('h2', '', '复习记录'), el('p', 'caption', r.due ? '下次复习\uff1a' + r.due : '可以只选掌握程度\uff0c不用写答案。'));
    function update(change) {
      try {C.saveRecord(localStorage, item.id, change, knownIds); renderReview(item); notice('已保存在本机。');}
      catch (error) {notice('保存失败\uff0c未覆盖原记录。 ' + error.message);}
    }
    const actions = el('div', 'review-actions');
    for (const [rating, label] of [['again', '再练'], ['hard', '还不熟'], ['good', '基本掌握'], ['easy', '熟练']]) {
      const b = button(label, () => update({rating})); b.dataset.rating = rating; b.disabled = !!loaded.error; b.setAttribute('aria-pressed', String(r.rating === rating)); actions.append(b);
    }
    const bookmark = button(r.bookmarked ? '已收藏' : '收藏', () => update({bookmarked: !r.bookmarked})); bookmark.dataset.action = 'bookmark'; bookmark.setAttribute('aria-pressed', String(r.bookmarked)); actions.append(bookmark); host.append(actions);
    const note = el('details'); note.append(el('summary', '', '笔记\uff08可选\uff09'));
    const label = el('label', 'field'); label.append(el('span', '', '本机笔记')); const input = el('textarea'); input.value = r.note || ''; input.rows = 4; label.append(input); note.append(label, button('保存笔记', () => update({note: input.value}))); host.append(note);
    if (r.answer || Object.values(r.layers || {}).some(Boolean)) {
      const drafts = el('details'); drafts.append(el('summary', '', '原题库草稿\uff08未改动\uff09')); paragraphs(drafts, r.answer || Object.values(r.layers).join('\n')); host.append(drafts);
    }
    host.append(el('p', 'caption', '自评间隔是教学默认值\uff0c不是 AI 打分。'));
  }
  function renderExperiments(item, article) {
    for (const spec of (meta.experiments || []).filter(x => x.entry === C.key(item))) {
      const host = el('section');
      const headings = [...article.querySelectorAll('h2')]; const anchor = headings.find(h => h.textContent.includes(spec.after));
      if (anchor) {
        let next = anchor.nextElementSibling; while (next && next.tagName !== 'H2') next = next.nextElementSibling;
        if (next) next.before(host); else article.append(host);
      } else article.append(host);
      X.mount(host, spec.type);
    }
  }
  async function renderDiagrams(article, ticket) {
    const codeBlocks = [...article.querySelectorAll('code.language-mermaid')];
    for (const [i, code] of codeBlocks.entries()) {
      const pre = code.parentElement, source = code.textContent;
      let figure = code.closest('figure.diagram'), image;
      if (figure) image = figure.querySelector('.diagram-output');
      else {figure = el('figure', 'diagram'); image = el('div', 'diagram-output'); const detail = el('details'); detail.append(el('summary', '', '图示源码')); pre.before(figure); detail.append(pre); figure.append(image, detail);}
      if (!window.mermaid) {image.append(el('p', 'caption', '图示渲染器未加载\uff1b源码可展开阅读。')); continue;}
      try {
        const {svg} = await window.mermaid.render('diagram-' + ticket + '-' + i, source);
        if (renderId !== ticket || !figure.isConnected) return;
        image.innerHTML = svg;
      } catch (_) {image.textContent = '该图示暂无法渲染\uff0c源码已保留。';}
    }
  }
  function toc(article, item) {
    $('toc').replaceChildren(); $('mobile-toc').replaceChildren();
    for (const [i, heading] of [...article.querySelectorAll('h2,h3')].entries()) {
      if (!heading.id) heading.id = 's-' + i;
      for (const host of [$('toc'), $('mobile-toc')]) {
        const link = a(heading.textContent, {...detailURL(item), anchor: heading.id}, heading.tagName === 'H3' ? 'sub' : '');
        link.addEventListener('click', event => {event.preventDefault(); history.replaceState(null, '', link.href); current = C.route(location.hash); const fold = heading.closest('details'); if (fold) fold.open = true; heading.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});}); host.append(link);
      }
    }
  }
  function renderDetail(type, id) {
    const item = graph.lookup.get(type + ':' + id); if (!item) return missing();
    const article = $('article'); if (window.MathJax?.typesetClear) window.MathJax.typesetClear([article]);
    // The article owns its figures. Navigation discards the entire previous subtree.
    article.replaceChildren(); $('detail-title').textContent = item.title; $('detail-type').textContent = labels[type] + (item.code ? ' / ' + item.code : item.framework ? ' / ' + item.framework : '');
    $('detail-summary').textContent = type === 'question' ? item.chapter + ' / ' + item.level : item.summary;
    const bc = $('breadcrumbs'); bc.replaceChildren(a('知识库', {}));
    const topic = type === 'topic' ? item : graph.lookup.get('topic:' + item.topics?.[0]);
    if (topic) {
      bc.append(el('span', '', '/'), a(graph.chapters.get(topic.chapter).title, {chapter: topic.chapter}));
      const section = sectionOf(topic); if (section) bc.append(el('span', '', '/'), a(section.title, {chapter: topic.chapter, section: section.id}));
    }
    if (type === 'question' || type === 'case') bc.append(a('返回问题列表', lastList, 'back'));
    const req = $('prerequisites'); req.replaceChildren(); const pre = type === 'topic' ? (meta.prerequisites?.[id] || []) : [];
    req.hidden = !pre.length; if (pre.length) {req.append(el('span', 'caption', '阅读前可先看')); pre.forEach(id => req.append(a(graph.lookup.get('topic:' + id).title, {topic: id})));}
    if (type === 'question') {
      if (item.doc_title && C.normalized(item.doc_title) !== C.normalized(item.question)) article.append(el('p', 'model-note', '另一来源的问法\uff1a' + item.doc_title));
      article.append(el('p', 'source-warning', data.question_note || '原稿提取内容\uff0c部分公式需对照原文核对。'));
      const reference = el('details', 'answer'); reference.append(el('summary', '', '查看参考答案'));
      for (const [title, value] of [['参考要点', item.brief], ['详细解释', item.deep], ['延伸追问', item.followup]]) {reference.append(el('h2', '', title)); paragraphs(reference, value);}
      article.append(reference);
    } else {
      article.innerHTML = item.html || '<p>Content unavailable.</p>';
      renderExperiments(item, article);
    }
    const related = $('related'); related.replaceChildren();
    const linked = graph.related(item), docs = linked.filter(x => !['case', 'question'].includes(x.type)), problems = linked.filter(x => ['case', 'question'].includes(x.type));
    if (docs.length) {sectionHead(related, '相关文档'); docs.forEach(x => related.append(row(x)));}
    if (problems.length) {sectionHead(related, '相关问题'); problems.slice(0, 6).forEach(x => related.append(row(x))); if (type === 'topic') related.append(a('查看全部 ' + problems.length + ' 个问题', {view: 'questions', ref: id}, 'more'));}
    if (type === 'topic') {
      const order = graph.chapters.get(item.chapter).sections.flatMap(s => s.topics), pos = order.indexOf(id), pager = el('nav', 'pager');
      if (pos > 0) pager.append(a('上一篇\uff1a' + graph.lookup.get('topic:' + order[pos - 1]).title, {topic: order[pos - 1]}));
      if (pos < order.length - 1) pager.append(a('下一篇\uff1a' + graph.lookup.get('topic:' + order[pos + 1]).title, {topic: order[pos + 1]})); related.append(pager);
    }
    renderReview(item); toc(article, item); updateTree(item);
    if (type !== 'question') renderDiagrams(article, renderId);
    if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([article]).catch(() => {});
    return item;
  }
  function missing() {
    const host = $('library'); host.replaceChildren(); head(host, '找不到该内容', '链接可能已过期。可以用顶部搜索或返回目录。'); host.append(a('返回知识库', {})); reveal('library', true); return null;
  }
  function route(scroll = true) {
    renderId++; current = C.route(location.hash); notice(''); $('search').value = current.get('q') || '';
    if (current.has('category') && current.get('category') !== 'all') current.set('kind', current.get('category'));
    if (current.has('q') && /^\d+$/.test(current.get('q')) && !current.has('view')) current.set('question', 'Q' + current.get('q').padStart(3, '0'));
    let view = current.get('view'), title = 'RL Infra', detail = null;
    if (view === 'questions') {lastList = location.hash; renderQuestions();}
    else if (view === 'library') renderLibrary();
    else if (['topic', 'case', 'implementation', 'question'].some(k => current.has(k))) {
      const type = ['topic', 'case', 'implementation', 'question'].find(k => current.has(k));
      detail = renderDetail(type, current.get(type)); if (!detail) return; view = 'detail'; title = detail.title + ' / RL Infra';
    } else if (current.has('chapter')) {if (!graph.chapters.has(current.get('chapter'))) return missing(); if (current.get('section') && !graph.chapters.get(current.get('chapter')).sections.some(s => s.id === current.get('section'))) return missing(); view = 'chapter'; renderChapter(current.get('chapter'));}
    else {view = 'home'; renderHome();}
    reveal(view, scroll); updateTree(detail); document.title = title;
    if (view === 'detail' && current.get('anchor')) {
      const anchor = document.getElementById(current.get('anchor')); if (anchor && $('article').contains(anchor)) {
        const parentDetails = anchor.closest('details'); if (parentDetails) parentDetails.open = true; anchor.scrollIntoView();
      }
    }
  }
  function search(replace) {
    const target = current.get('view') === 'questions' ? {...values(current), q: $('search').value.trim(), page: ''} : {view: 'library', q: $('search').value.trim()};
    save(target, replace, false);
  }
  tree($('chapter-list')); tree($('mobile-tree'));
  $('menu-toggle').addEventListener('click', () => $('mobile-directory').showModal());
  $('close-menu').addEventListener('click', () => $('mobile-directory').close());
  $('mobile-tree').addEventListener('click', event => {if (event.target.closest('a')) $('mobile-directory').close();});
  $('search-form').addEventListener('submit', event => {event.preventDefault(); clearTimeout(searchTimer); search(false);});
  $('search').addEventListener('input', () => {clearTimeout(searchTimer); searchTimer = setTimeout(() => search(true), 160);});
  document.addEventListener('keydown', event => {
    if (event.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName) && !event.metaKey && !event.ctrlKey && !event.altKey) {event.preventDefault(); $('search').focus();}
  });
  window.addEventListener('hashchange', () => route());
  window.addEventListener('storage', event => {if (event.key === C.KEY) route(false);});
  window.addEventListener('diagram-ready', () => {if (!$('detail').hidden) renderDiagrams($('article'), renderId);});
  window.addEventListener('load', () => {if (!$('detail').hidden && window.MathJax?.typesetPromise) window.MathJax.typesetPromise([$('article')]).catch(() => {});});
  $('build-version').textContent = '版本 ' + (data.build?.commit?.slice(0, 7) || 'local');
  route();
})();
