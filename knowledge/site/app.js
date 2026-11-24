/* Public, read-only knowledge reader. Progress is kept in localStorage only. */
(function () {
  'use strict';
  const C = window.RLCore, X = window.RLExperiments;
  const data = JSON.parse(document.getElementById('knowledge-data').textContent);
  const graph = C.index(data), meta = data.learning_map || {}, $ = id => document.getElementById(id);
  const knownIds = new Set(graph.questions.map(q => q.id));
  const labels = {topic: '知识', case: 'Issue', implementation: '框架实现', question: '问答'};
  let current = C.route(location.hash), renderId = 0, searchTimer = null;
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
    const target = detailURL(item);
    if (options.context) Object.assign(target, scope());
    if (options.fromList) Object.assign(target, scope(), {from: current.get('view'), back: location.hash});
    const link = a('', target, 'entry');
    const marker = item.type === 'question' ? item.code : labels[item.type];
    link.append(el('span', 'entry-meta', marker), el('strong', '', item.title));
    if (options.summary && item.type !== 'question') link.append(el('p', '', item.summary));
    if (item.type === 'question') link.append(el('span', 'caption', item.chapter + ' / ' + item.level));
    return link;
  }
  function sectionItems(section) {return graph.items.filter(item => graph.inSection(item, section.id));}
  function detailItem() {
    return ['topic','case','implementation','question'].map(k => graph.lookup.get(k + ':' + current.get(k))).find(Boolean);
  }
  function scope(item = detailItem()) {
    return graph.resolveScope({...values(current), area: current.get('area') || current.get('chapter')}, item);
  }
  function activeMode() {
    const view = current.get('view');
    if (view === 'questions' || view === 'library') return view;
    if (current.has('case') || current.has('question')) return current.get('from') === 'library' ? 'library' : 'questions';
    if (detailItem() && current.get('from') === 'library') return 'library';
    return 'map';
  }
  function listState(mode) {
    const source = current.get('view') === mode ? current : current.get('from') === mode ? C.route(current.get('back') || '') : new URLSearchParams();
    const state = {};
    for (const k of ['q', 'kind', 'level', 'review']) if (source.has(k)) state[k] = source.get(k);
    if (!state.q && ['questions','library'].includes(current.get('view'))) state.q = current.get('q') || '';
    return state;
  }
  function scopedRoute(mode, raw, preserveFilters = true) {
    const context = graph.normalizeScope(raw);
    if (mode !== 'map') return {view: mode, ...(preserveFilters ? listState(mode) : {}), ...context};
    if (context.impl) return {implementation: context.impl};
    if (context.ref) return {topic: context.ref};
    if (context.area) return {chapter: context.area, section: context.section, framework: context.framework};
    return {};
  }
  function reveal(view, scroll) {
    document.querySelectorAll('.view').forEach(v => {v.hidden = v.id !== view;});
    $('mobile-directory').close();
    document.querySelectorAll('[data-nav]').forEach(n => {
      if (n.dataset.nav === activeMode()) n.setAttribute('aria-current', 'page'); else n.removeAttribute('aria-current');
    });
    if (scroll) window.scrollTo({top: 0, behavior: 'instant'});
  }
  function chapterGroups(chapter) {
    if (chapter.id !== 'frameworks') return chapter.sections.map(s => ({
      id: s.id, title: s.title, items: s.topics.map(id => graph.lookup.get('topic:' + id))
    }));
    const groups = new Map();
    for (const item of graph.entries.filter(x => x.type === 'implementation')) {
      const key = item.framework;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    return [...groups].map(([title, items]) => ({title, items}));
  }
  function directoryLink(label, raw, cls, role = 'entry') {
    const link = a(label, scopedRoute('map', raw), cls);
    link.dataset.directoryScope = JSON.stringify(graph.normalizeScope(raw));
    link.dataset.directoryLabel = label;
    link.dataset.directoryRole = role;
    return link;
  }
  function countLabel(raw) {
    const count = el('span', 'directory-count');
    count.dataset.countScope = JSON.stringify(graph.normalizeScope(raw));
    return count;
  }
  function directory(host, overview = false) {
    host.replaceChildren();
    host.append(directoryLink('全部知识', {}, 'tree-overview tree-root', 'root'));
    for (const [i, chapter] of data.chapters.entries()) {
      const group = el('details', overview ? 'directory-group' : 'tree-group');
      group.dataset.area = chapter.id;
      const raw = {area: chapter.id};
      const summary = el('summary', '', (i + 1) + '. ' + chapter.title);
      summary.append(countLabel(raw));
      group.append(summary, directoryLink('章节概览', raw, 'tree-overview', 'chapter'));
      for (const s of chapterGroups(chapter)) {
        const sub = el('details', 'tree-section');
        const sectionScope = s.id ? {section: s.id} : {area: 'frameworks', framework: s.title};
        if (s.id) sub.dataset.section = s.id; else sub.dataset.framework = s.title;
        const subSummary = el('summary', 'tree-section-title', s.title);
        subSummary.append(countLabel(sectionScope));
        sub.append(subSummary, directoryLink('小节概览', sectionScope, 'tree-overview', 'section'));
        for (const item of s.items) {
          const entryScope = item.type === 'implementation' ? {impl: item.id} : {ref: item.id};
          const link = directoryLink(item.title, entryScope, 'tree-topic');
          link.dataset.entry = item.type + ':' + item.id;
          sub.append(link);
        }
        group.append(sub);
      }
      host.append(group);
    }
    host.append(a('学习路线', {view: 'paths'}, 'tree-overview'));
  }
  function tree(host) {directory(host);}
  function updateTree(item) {
    const context = scope(item), mode = activeMode();
    const key = value => JSON.stringify(graph.normalizeScope(value));
    const currentKey = key(context);
    const loaded = mode === 'questions' && current.get('review') ? reviewState().state : undefined;
    const candidates = mode === 'map' ? [] : filtered(mode === 'questions' ? graph.items.filter(x => ['case','question'].includes(x.type)) : graph.items, loaded, {}, current.get('view') === mode ? current : C.route(C.url(listState(mode))));
    function count(raw) {
      const selection = graph.normalizeScope(raw);
      if (mode === 'map') return graph.entries.filter(x => (selection.area === 'frameworks' ? x.type === 'implementation' : selection.area ? x.type === 'topic' : ['topic','implementation'].includes(x.type)) && graph.inScope(x, selection)).length;
      return candidates.filter(x => graph.inScope(x, selection)).length;
    }
    const modeLabel = mode === 'questions' ? 'Issues' : mode === 'library' ? '搜索' : '知识';
    document.querySelectorAll('[data-directory-scope]').forEach(link => {
      const raw = JSON.parse(link.dataset.directoryScope), role = link.dataset.directoryRole;
      const label = role === 'root' ? (mode === 'library' ? '全库搜索' : '全部' + modeLabel) : role === 'chapter' ? '本章' + modeLabel : role === 'section' ? '本节' + modeLabel : link.dataset.directoryLabel;
      link.href = C.url(scopedRoute(mode, raw));
      link.replaceChildren(el('span', 'tree-label', label));
      if (mode !== 'map' && role === 'entry') link.append(el('span', 'tree-entry-count', String(count(raw))));
      if (key(raw) === currentKey) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
    });
    document.querySelectorAll('[data-count-scope]').forEach(node => {
      const raw = JSON.parse(node.dataset.countScope);
      node.textContent = String(count(raw));
    });
    document.querySelectorAll('.tree-group, .directory-group').forEach(group => {
      if (group.dataset.area === context.area) group.open = true;
      for (const sub of group.querySelectorAll('.tree-section')) {
        if ((context.section && sub.dataset.section === context.section) || (context.framework && sub.dataset.framework === context.framework)) sub.open = true;
      }
    });
  }
  function renderHome() {
    const host = $('home'); host.replaceChildren();
    head(host, 'RL Infra 技术手册');
    const directoryHost = el('nav', 'directory-index');
    directoryHost.setAttribute('aria-label', '章节目录');
    directory(directoryHost, true); host.append(directoryHost);
  }
  function renderPaths() {
    const host = $('paths'); host.replaceChildren();
    head(host, '学习路线');
    for (const path of meta.paths || []) {
      const section = el('section', 'topic-section');
      sectionHead(section, path.title); section.append(el('p', 'lead', path.goal));
      for (const id of path.topics) section.append(row(graph.lookup.get('topic:' + id), {summary: true}));
      host.append(section);
    }
  }
  function renderChapter(id) {
    const chapter = graph.chapters.get(id); if (!chapter) return missing();
    const host = $('chapter'); host.replaceChildren(); scopeHeading(host, 'map'); head(host, chapter.title, chapter.description);
    if (id === 'frameworks') {for (const group of chapterGroups(chapter)) {if (current.get('framework') && group.title !== current.get('framework')) continue; const section = el('section', 'topic-section'); sectionHead(section, group.title); group.items.forEach(x => section.append(row(x, {summary: true}))); host.append(section);} return;}
    if (!chapter.sections.length) {host.append(el('p', 'caption', '待补充')); return;}
    const selected = current.get('section');
    if (selected && !chapter.sections.some(s => s.id === selected)) return missing();
    for (const s of chapter.sections || []) {
      if (selected && selected !== s.id) continue;
      const section = el('section', 'topic-section'); section.id = 'group-' + s.id;
      const items = sectionItems(s), problems = items.filter(x => ['case', 'question'].includes(x.type)), implementations = items.filter(x => x.type === 'implementation');
      sectionHead(section, s.title, a(problems.length + ' 个问题 →', {view: 'questions', area: id, section: s.id}));
      s.topics.forEach(id => section.append(row(graph.lookup.get('topic:' + id), {summary: true})));
      if (implementations.length) {
        const fold = el('details'); fold.open = true; fold.append(el('summary', '', '框架实现 / ' + implementations.length)); implementations.forEach(x => fold.append(row(x))); section.append(fold);
      }
      if (problems.length) {
        const fold = el('details'); fold.open = true; fold.append(el('summary', '', 'Issues 与问答 / ' + problems.length)); const cases = problems.filter(x => x.type === 'case'), questions = problems.filter(x => x.type === 'question'); cases.forEach(x => fold.append(row(x, {context: true}))); questions.slice(0, 4).forEach(x => fold.append(row(x, {context: true}))); fold.append(a('查看全部问题', {view: 'questions', area: id, section: s.id}, 'more')); section.append(fold);
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
  function appendScopeLinks(strip, mode, context) {
    const nodes = [[mode === 'map' ? '知识库' : mode === 'questions' ? '全部 Issues' : '全库搜索', {}]];
    if (context.area) nodes.push([graph.chapters.get(context.area).title, {area: context.area}]);
    if (context.section) nodes.push([graph.sections.get(context.section).title, {section: context.section}]);
    if (context.framework) nodes.push([context.framework, {framework: context.framework}]);
    if (context.ref) nodes.push([graph.lookup.get('topic:' + context.ref).title, {ref: context.ref}]);
    if (context.impl) nodes.push([graph.lookup.get('implementation:' + context.impl).title, {impl: context.impl}]);
    nodes.forEach(([label, raw], i) => {
      if (i) strip.append(el('span', 'caption', '/'));
      const link = a(label, scopedRoute(mode, raw)); if (i === nodes.length - 1) link.setAttribute('aria-current', 'page'); strip.append(link);
    });
  }
  function scopeHeading(host, mode = activeMode()) {
    const strip = el('nav', 'scope-heading'); strip.setAttribute('aria-label', '当前目录范围');
    appendScopeLinks(strip, mode, scope());
    host.append(strip);
  }
  function filters(host, kind) {
    scopeHeading(host, kind);
    const form = el('div', 'filters'), p = values(current);
    const set = (key, value) => save({...p, [key]: value, page: ''}, true, false);
    if (kind === 'questions') {
      form.append(field('难度', [['', '全部'], ['基础', '基础'], ['进阶', '进阶']], p.level, v => set('level', v), 'level'));
      form.append(field('复习', [['', '全部'], ['due', '待复习'], ['weak', '需巩固'], ['bookmarked', '收藏'], ['new', '未自评']], p.review, v => set('review', v), 'review'));
    } else form.append(field('类型', [['', '全部'], ...Object.entries(labels)], p.kind, v => set('kind', v), 'kind'));
    host.append(form);
  }
  function filtered(items, progress, context = scope(), p = current) {
    return graph.search(items, p.get('q') || '').filter(x => {
      if (p.get('kind') && x.type !== p.get('kind')) return false;
      if (!graph.inScope(x, context)) return false;
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
    items.slice((page - 1) * size, page * size).forEach(x => host.append(row(x, {summary: true, fromList: true})));
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
    head(host, 'Issues', data.cases.length + ' 个工程问题 · ' + graph.questions.length + ' 道关联问答');
    const kinds = el('nav', 'issue-switch'); kinds.setAttribute('aria-label', '问题类别');
    for (const [kind, label] of [['', '全部'], ['case', '工程 Issues'], ['question', '关联问答']]) {const link = a(label, {...values(current), kind, page: ''}); if ((current.get('kind') || '') === kind) link.setAttribute('aria-current', 'page'); kinds.append(link);} host.append(kinds);
    const loaded = reviewState(); filters(host, 'questions');
    const review = el('details', 'review-panel'); review.append(el('summary', '', '学习记录')); progressControls(review, loaded); host.append(review);
    
    paginate(host, filtered(graph.items.filter(x => ['case', 'question'].includes(x.type)), loaded.state));
  }
  function renderLibrary() {
    const host = $('library'); host.replaceChildren(); head(host, '搜索'); filters(host, 'library'); paginate(host, filtered(graph.items));
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
    const bc = $('breadcrumbs'); bc.replaceChildren();
    appendScopeLinks(bc, activeMode(), scope(item));
    const origin = current.get('from'), back = current.get('back');
    if (['questions','library'].includes(origin) && back?.startsWith('#') && C.route(back).get('view') === origin) bc.append(a(origin === 'library' ? '返回搜索结果' : '返回问题列表', back, 'back'));
    else if (type === 'question' || type === 'case') bc.append(a('返回关联问题', scopedRoute('questions', scope(item), false), 'back'));
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
    const cases = problems.filter(x => x.type === 'case'), questions = problems.filter(x => x.type === 'question');
    if (cases.length) {sectionHead(related, '相关 Issues'); cases.forEach(x => related.append(row(x, {context: true})));}
    if (questions.length) {sectionHead(related, '关联问答'); questions.slice(0, 6).forEach(x => related.append(row(x, {context: true})));}
    if (type === 'topic' && problems.length) related.append(a('查看全部 ' + problems.length + ' 个关联问题', scopedRoute('questions', {ref: id}, false), 'more'));
    if (type === 'topic') {
      const order = graph.chapters.get(item.chapter).sections.flatMap(s => s.topics), pos = order.indexOf(id), pager = el('nav', 'pager');
      if (pos > 0) pager.append(a('上一篇\uff1a' + graph.lookup.get('topic:' + order[pos - 1]).title, {topic: order[pos - 1]}));
      if (pos < order.length - 1) pager.append(a('下一篇\uff1a' + graph.lookup.get('topic:' + order[pos + 1]).title, {topic: order[pos + 1]})); related.append(pager);
    }
    renderReview(item); toc(article, item);
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
    if (view === 'questions') renderQuestions();
    else if (view === 'library') renderLibrary();
    else if (view === 'paths') renderPaths();
    else if (['topic', 'case', 'implementation', 'question'].some(k => current.has(k))) {
      const type = ['topic', 'case', 'implementation', 'question'].find(k => current.has(k));
      detail = renderDetail(type, current.get(type)); if (!detail) return; view = 'detail'; title = detail.title + ' / RL Infra';
    } else if (current.has('chapter')) {if (!graph.chapters.has(current.get('chapter'))) return missing(); if (current.get('section') && !graph.chapters.get(current.get('chapter')).sections.some(s => s.id === current.get('section'))) return missing(); view = 'chapter'; renderChapter(current.get('chapter'));}
    else {view = 'home'; renderHome();}
    reveal(view, scroll); updateTree(detail); document.title = title;
    const context = scope(detail);
    for (const mode of ['map', 'questions', 'library']) document.querySelector('[data-nav="' + mode + '"]').href = C.url(scopedRoute(mode, context));
    if (view === 'detail' && current.get('anchor')) {
      const anchor = document.getElementById(current.get('anchor')); if (anchor && $('article').contains(anchor)) {
        const parentDetails = anchor.closest('details'); if (parentDetails) parentDetails.open = true; anchor.scrollIntoView();
      }
    }
  }
  function search(replace) {
    const mode = activeMode() === 'questions' ? 'questions' : 'library';
    const target = {...scopedRoute(mode, scope()), q: $('search').value.trim(), page: ''};
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
