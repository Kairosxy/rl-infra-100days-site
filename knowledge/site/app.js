/* Unified workspace: one directory, one result set, one reader. */
(function () {
  'use strict';
  const C = window.RLCore, X = window.RLExperiments, N = window.RLNavigation;
  const data = JSON.parse(document.getElementById('knowledge-data').textContent);
  const graph = C.index(data), nav = N.create(data, graph), meta = data.learning_map || {}, $ = id => document.getElementById(id);
  const knownIds = new Set(graph.questions.map(q => q.id));
  const labels = {topic: '知识', case: 'Issue', implementation: '框架实现', question: '问答'};
  const modeLabels = {all: '全部', topic: '知识', issues: 'Issues', implementation: '框架实现'};
  let current = nav.read(location.hash), renderId = 0, searchTimer = null, results;
  const el = (tag, cls, text) => {const out = document.createElement(tag); if (cls) out.className = cls; if (text !== undefined) out.textContent = text; return out;};
  const a = (text, href, cls) => {const link = el('a', cls, text); link.href = typeof href === 'string' ? href : C.url(href); return link;};
  const button = (text, fn, cls = '') => {const b = el('button', cls, text); b.type = 'button'; b.addEventListener('click', fn); return b;};
  const stateLink = (text, state, cls) => a(text, nav.href(state), cls);
  const entryLink = (item, text = item.title, cls = '') => stateLink(text, nav.open(current, item), cls);
  const selected = (node, value) => {if (value) node.setAttribute('aria-current', 'page'); else node.removeAttribute('aria-current');};
  function notice(text) {$('notice').textContent = text || '';}
  function navigate(state, replace = false, scroll = true) {
    const hash = nav.href(state);
    if (replace) history.replaceState(null, '', hash);
    else if (location.hash !== hash && !(location.hash === '' && hash === '#')) history.pushState(null, '', hash);
    route(scroll);
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
    const link = entryLink(item, '', 'entry'); link.dataset.entry = C.key(item);
    const marker = item.type === 'question' ? item.code : labels[item.type];
    link.append(el('span', 'entry-meta', marker), el('strong', '', item.title));
    if (options.summary && item.type !== 'question') link.append(el('p', '', item.summary));
    if (item.type === 'question') link.append(el('span', 'caption', item.chapter + ' / ' + item.level));
    if (item.type === 'implementation') link.append(el('span', 'caption', item.framework));
    return link;
  }
  // Directory disclosure and selection are independent controls. The same DOM moves into
  // the mobile dialog; expansion state is never coupled to content type or search.
  const treeRecords = [];
  function buildDirectory() {
    const host = $('chapter-list'); host.replaceChildren();
    function append(parent, node, depth) {
      const wrapper = el('div', 'directory-node'); wrapper.dataset.depth = String(depth);
      const line = el('div', 'directory-line'); const key = nav.scopeKey(node.scope);
      const link = a(node.title, '#', 'scope-link'); link.dataset.scope = key;
      link.dataset.area = node.scope.area;
      if (node.scope.section) link.dataset.section = node.scope.section;
      if (node.scope.ref) link.dataset.ref = node.scope.ref;
      if (node.scope.framework) link.dataset.framework = node.scope.framework;
      if (node.scope.impl) link.dataset.impl = node.scope.impl;
      const count = el('span', 'scope-count'); count.setAttribute('aria-hidden', 'true'); link.append(count);
      let children, toggle;
      if (node.children?.length) {
        children = el('div', 'directory-children'); children.id = 'directory-branch-' + treeRecords.length;
        children.hidden = true;
        toggle = button('', () => setExpanded(!children.hidden ? false : true), 'directory-toggle');
        toggle.setAttribute('aria-controls', children.id);
        function setExpanded(open) {
          children.hidden = !open; toggle.setAttribute('aria-expanded', String(open));
          toggle.setAttribute('aria-label', (open ? '收起 ' : '展开 ') + node.title);
        }
        setExpanded(false); line.append(toggle);
        treeRecords.push({node, key, link, count, children, setExpanded});
      } else {
        line.append(el('span', 'directory-spacer')); treeRecords.push({node, key, link, count});
      }
      line.append(link); wrapper.append(line);
      if (children) {node.children.forEach(child => append(children, child, depth + 1)); wrapper.append(children);}
      parent.append(wrapper);
    }
    append(host, {title: '全部内容', scope: N.EMPTY}, 0);
    nav.tree().forEach(node => append(host, node, 0));
  }
  function updateDirectory() {
    const active = nav.scopeKey(current.scope), ancestors = new Set(nav.trail(current.scope).map(x => nav.scopeKey(x.scope)));
    for (const record of treeRecords) {
      record.link.href = nav.href(nav.selectScope(current, record.node.scope));
      const count = results.at(record.node.scope).length;
      record.count.textContent = String(count);
      record.link.setAttribute('aria-label', record.node.title + '，' + count + ' 项' + modeLabels[current.mode]);
      selected(record.link, record.key === active);
      if (record.children && ancestors.has(record.key)) record.setExpanded(true);
    }
  }
  const mobile = matchMedia('(max-width: 900px)');
  function placeDirectory() {
    if (mobile.matches) $('mobile-tree').append($('chapter-list'));
    else {if ($('mobile-directory').open) $('mobile-directory').close(); $('desktop-tree').append($('chapter-list'));}
  }
  function renderContext() {
    const trail = nav.trail(current.scope), last = trail[trail.length - 1];
    const bc = $('scope-breadcrumbs'); bc.replaceChildren();
    for (const [i, node] of trail.entries()) {
      if (i) bc.append(el('span', 'breadcrumb-separator', '/'));
      if (i === trail.length - 1) {const here = el('span', 'breadcrumb-current', node.title); here.setAttribute('aria-current', 'location'); bc.append(here);}
      else bc.append(stateLink(node.title, nav.selectScope(current, node.scope)));
    }
    $('scope-name').textContent = current.scope.area ? last.title : 'RL Infra 技术手册';
    const chapter = graph.chapters.get(current.scope.area);
    $('scope-description').textContent = !current.scope.area ? '模型、算法与训练基础设施' : current.scope.ref ? graph.lookup.get('topic:' + current.scope.ref).summary : current.scope.impl ? graph.lookup.get('implementation:' + current.scope.impl).summary : current.scope.area === 'frameworks' ? '按框架组织实现，并关联涉及的原理与 Issues。' : current.scope.section ? '' : (chapter?.description || '');
    const counts = results.counts(current.scope);
    for (const link of $('content-tabs').querySelectorAll('[data-mode]')) {
      const mode = link.dataset.mode;
      link.href = nav.href(nav.selectMode(current, mode));
      link.querySelector('.tab-count').textContent = counts[mode];
      selected(link, current.mode === mode);
    }
    const query = $('query-state'); query.replaceChildren(); query.hidden = !current.q;
    if (current.q) {
      query.append(el('span', '', '搜索：' + current.q), stateLink('清除', nav.search(current, '')));
      if (current.scope.area) query.append(stateLink('搜索全部领域', nav.selectScope(current, {}), 'query-global'));
    }
    $('search').placeholder = current.scope.area ? '搜索当前范围…' : '搜索知识、Issues、框架实现…';
    $('workspace-context').classList.toggle('reading-context', !!current.entry);
    $('workspace-context').classList.toggle('map-context', isMap());
    $('content-tabs').querySelector('[data-mode=all] .tab-label').textContent = !current.scope.area && !current.q ? '学习地图' : '全部';
  }
  function isMap() {
    return !current.entry && !current.invalid && (current.paths ||
      (current.mode === 'all' && !current.scope.area && !current.q));
  }
  let mapObserver;
  function overview(host) {
    const home = el('div', 'learning-home');
    const paths = el('section', 'learning-routes');
    paths.id = 'learning-paths';
    paths.append(el('h1', 'home-title', '学习路线'));
    for (const path of meta.paths || []) {
      const line = el('div', 'route-row');
      line.append(el('h2', 'route-title', path.title));
      const steps = el('ol', 'route-steps');
      for (const [i, id] of path.topics.entries()) {
        const item = graph.lookup.get('topic:' + id), step = el('li');
        if (i) {const arrow = el('span', 'route-arrow', '→'); arrow.setAttribute('aria-hidden', 'true'); step.append(arrow);}
        const link = entryLink(item, meta.topic_labels?.[id] || item.title);
        link.dataset.topic = id; link.title = item.title; link.setAttribute('aria-label', item.title);
        step.append(link); steps.append(step);
      }
      line.append(steps); paths.append(line);
    }
    home.append(paths);
    const section = el('section', 'knowledge-map');
    section.id = 'knowledge-map';
    const heading = el('div', 'map-heading');
    heading.append(el('h2', 'home-title', '知识脉络'));
    const lineage = graph.lookup.get('topic:technology-lineage');
    if (lineage) heading.append(entryLink(lineage, '技术谱系', 'map-source'));
    section.append(heading);
    const spec = meta.domain_map;
    if (!spec) {home.append(section); host.append(home); return;}
    const stage = el('div', 'map-stage'), positions = new Map();
    function domainLink(id) {
      const info = spec.nodes[id], link = stateLink('', nav.list(current, {scope: nav.normalizeScope({area: id}), mode: 'all', q: '', paths: false}), 'map-node');
      link.dataset.area = id;
      link.append(el('strong', '', info.title), el('span', 'map-terms', info.terms));
      link.setAttribute('aria-label', graph.chapters.get(id).title + '：' + info.terms);
      return link;
    }
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('map-edges'); svg.setAttribute('aria-hidden', 'true'); svg.setAttribute('focusable', 'false');
    stage.append(svg);
    for (const ids of spec.rows) {
      const row = el('div', 'map-band');
      if (ids.length === 1) row.classList.add('map-hub');
      for (const id of ids) {const link = domainLink(id); positions.set(id, link); row.append(link);}
      stage.append(row);
    }
    section.append(stage);
    const support = el('div', 'map-support');
    for (const id of spec.support) support.append(domainLink(id));
    const relations = el('ul', 'sr-only'); relations.setAttribute('aria-label', '技术关系');
    for (const edge of spec.edges) relations.append(el('li', '', spec.nodes[edge.from].title + ' → ' + spec.nodes[edge.to].title + '：' + edge.relation));
    const label = el('div', 'map-support-heading', '贯穿各层');
    section.append(label, support, relations);
    home.append(section); host.append(home);
    function drawEdges() {
      const frame = stage.getBoundingClientRect(); if (!frame.width || !frame.height) return;
      svg.setAttribute('viewBox', `0 0 ${frame.width} ${frame.height}`);
      svg.replaceChildren();
      const node = (tag, attrs) => {const n = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n;};
      const defs = node('defs', {}), marker = node('marker', {id:'map-arrow', markerWidth:7, markerHeight:7, refX:6, refY:3.5, orient:'auto', markerUnits:'userSpaceOnUse'});
      marker.append(node('path', {d:'M0 0 L6 3.5 L0 7', fill:'currentColor'})); defs.append(marker); svg.append(defs);
      const drawn = new Set();
      for (const edge of spec.edges) {
        const a = positions.get(edge.from).getBoundingClientRect(), b = positions.get(edge.to).getBoundingClientRect();
        const down = a.top < b.top, sy = (down ? a.bottom : a.top) - frame.top, ty = (down ? b.top : b.bottom) - frame.top;
        const sx = (a.left + a.width / 2) - frame.left;
        const tx = edge.to === 'async' ? sx : (b.left + b.width / 2) - frame.left;
        const y = (sy + ty) / 2, path = `M${sx} ${sy} V${y} H${tx} V${ty}`;
        if (!drawn.has(path)) {drawn.add(path); svg.append(node('path', {d:path, class:'map-edge', 'marker-end':'url(#map-arrow)', 'data-from':edge.from, 'data-to':edge.to}));}
      }
    }
    drawEdges();
    mapObserver = new ResizeObserver(drawEdges); mapObserver.observe(stage);
  }
  function reviewState() {
    let result;
    try {result = C.readProgress(localStorage, knownIds);} catch (error) {result = {state: {version: 1, records: {}}, error: error.message};}
    if (result.error) notice('复习记录暂不可读。未覆盖原记录；请先导出备份。');
    return result;
  }
  function issueFilters(host, loaded) {
    const switcher = el('nav', 'issue-switch'); switcher.setAttribute('aria-label', 'Issues 类型');
    for (const [kind, title] of [['', '全部 Issues'], ['case', '工程案例'], ['question', '关联问答']]) {
      const link = stateLink(title, nav.list(current, {kind, level: '', review: '', page: 1}));
      link.dataset.kind = kind || 'all'; selected(link, current.kind === kind); switcher.append(link);
    }
    host.append(switcher);
    const form = el('div', 'filters');
    const set = (key, value) => navigate(nav.list(current, {[key]: value, page: 1}), false, false);
    if (current.kind !== 'case') {
      form.append(field('难度', [['', '全部'], ['基础', '基础'], ['进阶', '进阶']], current.level, v => set('level', v), 'level'));
      form.append(field('复习', [['', '全部'], ['due', '待复习'], ['weak', '需巩固'], ['bookmarked', '收藏'], ['new', '未自评']], current.review, v => set('review', v), 'review'));
      host.append(form);
    }
    const review = el('details', 'review-panel'); review.append(el('summary', '', '学习记录')); progressControls(review, loaded); host.append(review);
  }
  function paginate(host, items) {
    const size = 24, pages = Math.max(1, Math.ceil(items.length / size)), page = Math.min(pages, current.page);
    host.append(el('p', 'count', items.length + ' 项结果' + (pages > 1 ? ' · 第 ' + page + ' / ' + pages + ' 页' : '')));
    if (!items.length) {
      host.append(el('p', 'empty', '当前范围没有匹配内容。'));
      if (current.q) host.append(stateLink('清除搜索', nav.search(current, ''), 'more'));
      if (current.scope.area) host.append(stateLink('查看全部领域', nav.selectScope(current, {}), 'more'));
    }
    items.slice((page - 1) * size, page * size).forEach(x => host.append(row(x, {summary: true})));
    if (pages > 1) {
      const pager = el('nav', 'pager'); pager.setAttribute('aria-label', '分页');
      if (page > 1) pager.append(stateLink('上一页', nav.list(current, {page: page - 1})));
      if (page < pages) pager.append(stateLink('下一页', nav.list(current, {page: page + 1}))); host.append(pager);
    }
  }
  function renderCollection(loaded) {
    const host = $('collection'); host.replaceChildren();
    if (current.invalid) {
      head(host, '找不到该内容', '链接可能已过期。返回目录或搜索其他内容。');
      host.append(a('全部内容', '#')); return;
    }
    if (isMap()) {overview(host); return;}
    if (current.mode === 'issues') issueFilters(host, loaded);
    if (current.mode !== 'all') {paginate(host, results.at(current.scope)); return;}
    const counts = results.counts(current.scope);
    host.append(el('p', 'count', counts.all + ' 项结果'));
    if (!counts.all) {host.append(el('p', 'empty', '当前范围没有匹配内容。')); return;}
    for (const mode of ['topic', 'issues', 'implementation']) {
      const items = results.at(current.scope, mode); if (!items.length) continue;
      const section = el('section', 'collection-group'); section.dataset.group = mode;
      sectionHead(section, modeLabels[mode], stateLink('查看全部 ' + items.length, nav.selectMode(current, mode)));
      items.slice(0, 4).forEach(x => section.append(row(x, {summary: true}))); host.append(section);
    }
  }
  function download(text, name) {
    const href = URL.createObjectURL(new Blob([text], {type: 'application/json'})); const link = a('', href); link.download = name; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(href), 1000);
  }
  function progressControls(host, loaded) {
    const state = loaded.state, records = Object.values(state.records), strip = el('div', 'review-tools');
    const reviewed = records.filter(r => r.attempts).length, due = records.filter(r => r.due && r.due <= C.dateAfter()).length;
    strip.append(el('span', '', '已练习 ' + reviewed + ' / ' + graph.questions.length), a('待复习 ' + due, nav.href(nav.list(current, {mode: 'issues', kind: 'question', level: '', review: 'due', page: 1}))));
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
  // Scroll only the objects that need their own width; keep the article and
  // its headings in the normal reading flow. Native table semantics stay intact.
  function prepareArticle(article) {
    function scrollRegion(className, label) {
      const region = el('div', className);
      region.setAttribute('role', 'region'); region.setAttribute('aria-label', label);
      region.tabIndex = 0;
      return region;
    }
    for (const table of article.querySelectorAll('table')) {
      const region = scrollRegion('article-table', '表格（可横向滚动）');
      if ((table.rows[0]?.cells.length || 0) >= 3) region.dataset.wide = 'true';
      table.before(region); region.append(table);
    }
    for (const math of article.querySelectorAll('div.arithmatex')) {
      math.tabIndex = 0; math.setAttribute('role', 'region'); math.setAttribute('aria-label', '公式');
    }
    for (const pre of article.querySelectorAll('pre')) {
      pre.tabIndex = 0; pre.setAttribute('aria-label', '代码');
    }
    for (const image of article.querySelectorAll('p > img')) {
      const paragraph = image.parentElement;
      if (paragraph.children.length !== 1 || paragraph.textContent.trim()) continue;
      const figure = el('figure', 'article-figure');
      const region = scrollRegion('figure-scroll', '机制图（可横向滚动）');
      // SVGs remain selectable/editable in the repository. Avoid shrinking a
      // large sequence diagram until its labels become illegible on phones.
      const sizeFigure = () => {
        if (image.naturalWidth > 0) {
          image.style.minWidth = Math.min(880, image.naturalWidth) + 'px';
          if (image.naturalWidth >= 520) figure.dataset.wide = 'true';
        }
      };
      image.addEventListener('load', sizeFigure, {once:true}); sizeFigure();
      paragraph.replaceWith(figure); region.append(image); figure.append(region);
      if (image.alt) figure.append(el('figcaption', '', image.alt));
    }
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
      if (!window.mermaid) {image.textContent = '图示渲染器未加载。'; pre.closest('details').open = true; continue;}
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
        const link = stateLink(heading.textContent, {...current, anchor: heading.id}, heading.tagName === 'H3' ? 'sub' : '');
        link.addEventListener('click', event => {event.preventDefault(); history.replaceState(null, '', link.href); current = nav.read(location.hash); const fold = heading.closest('details'); if (fold) fold.open = true; heading.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});}); host.append(link);
      }
    }
  }
  function renderDetail(item) {
    const type = item.type, id = item.id;
    const article = $('article'); if (window.MathJax?.typesetClear) window.MathJax.typesetClear([article]);
    // The article owns its figures. Navigation discards the entire previous subtree.
    article.replaceChildren(); $('detail-title').textContent = item.title; $('detail-type').textContent = labels[type] + (item.code ? ' / ' + item.code : item.framework ? ' / ' + item.framework : '');
    $('detail-summary').textContent = type === 'question' ? item.chapter + ' / ' + item.level : item.summary;
    const bc = $('breadcrumbs'); bc.replaceChildren(a('返回列表', nav.back(current), 'back'));
    const req = $('prerequisites'); req.replaceChildren(); const pre = type === 'topic' ? (meta.prerequisites?.[id] || []) : [];
    req.hidden = !pre.length; if (pre.length) {req.append(el('span', 'caption', '阅读前可先看')); pre.forEach(id => req.append(entryLink(graph.lookup.get('topic:' + id))));}
    if (type === 'question') {
      if (item.doc_title && C.normalized(item.doc_title) !== C.normalized(item.question)) article.append(el('p', 'model-note', '另一来源的问法\uff1a' + item.doc_title));
      article.append(el('p', 'source-warning', data.question_note || '原稿提取内容\uff0c部分公式需对照原文核对。'));
      const reference = el('details', 'answer'); reference.append(el('summary', '', '查看参考答案'));
      for (const [title, value] of [['参考要点', item.brief], ['详细解释', item.deep], ['延伸追问', item.followup]]) {reference.append(el('h2', '', title)); paragraphs(reference, value);}
      article.append(reference);
    } else {
      article.innerHTML = item.html || '<p>Content unavailable.</p>';
      for (const link of article.querySelectorAll('a[href^="#"]')) {
        const raw = link.getAttribute('href'), target = nav.read(raw);
        if (target.entry && !target.invalid) link.href = nav.href({...nav.open(current, target.entry), anchor: target.anchor});
        else if (article.querySelectorAll('[id]').length && [...article.querySelectorAll('[id]')].some(node => '#' + node.id === raw)) {
          link.href = nav.href({...current, anchor: raw.slice(1)});
        }
      }
      prepareArticle(article);
      renderExperiments(item, article);
    }
    const related = $('related'); related.replaceChildren();
    const linked = graph.related(item), docs = linked.filter(x => !['case', 'question'].includes(x.type)), problems = linked.filter(x => ['case', 'question'].includes(x.type));
    if (docs.length) {sectionHead(related, '相关文档'); docs.forEach(x => related.append(row(x)));}
    const cases = problems.filter(x => x.type === 'case'), questions = problems.filter(x => x.type === 'question');
    if (cases.length) {sectionHead(related, '相关 Issues'); cases.forEach(x => related.append(row(x)));}
    if (questions.length) {sectionHead(related, '关联问答'); questions.slice(0, 6).forEach(x => related.append(row(x)));}
    if (type === 'topic' && problems.length) related.append(stateLink('查看全部 ' + problems.length + ' 个关联问题', nav.selectMode(nav.selectScope(current, {ref: id}), 'issues'), 'more'));
    if (type === 'topic') {
      const order = graph.chapters.get(item.chapter).sections.flatMap(s => s.topics), pos = order.indexOf(id), pager = el('nav', 'pager');
      if (pos > 0) pager.append(entryLink(graph.lookup.get('topic:' + order[pos - 1]), '上一篇：' + graph.lookup.get('topic:' + order[pos - 1]).title));
      if (pos < order.length - 1) pager.append(entryLink(graph.lookup.get('topic:' + order[pos + 1]), '下一篇：' + graph.lookup.get('topic:' + order[pos + 1]).title)); related.append(pager);
    }
    renderReview(item); toc(article, item);
    if (type !== 'question') renderDiagrams(article, renderId);
    if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([article]).catch(() => {});
    return item;
  }
  function route(scroll = true) {
    clearTimeout(searchTimer); mapObserver?.disconnect(); mapObserver = undefined; renderId++; current = nav.read(location.hash); notice('');
    if ($('search').value !== current.q) $('search').value = current.q;
    const loaded = reviewState(); results = nav.resultSet(current, loaded.state.records);
    updateDirectory(); renderContext();
    const detail = current.entry && !current.invalid;
    if (!detail && $('article').childNodes.length) {
      if (window.MathJax?.typesetClear) window.MathJax.typesetClear([$('article')]);
      $('article').replaceChildren(); $('related').replaceChildren(); $('question-review').replaceChildren();
      $('toc').replaceChildren(); $('mobile-toc').replaceChildren();
    }
    $('workspace-context').hidden = false;
    $('collection').hidden = !!detail;
    $('detail').hidden = !detail;
    if (detail) renderDetail(current.entry);
    else renderCollection(loaded);
    document.title = (detail ? current.entry.title : isMap() ? '学习地图' : $('scope-name').textContent) + ' / RL Infra';
    if ($('mobile-directory').open) $('mobile-directory').close();
    if (scroll) window.scrollTo({top: 0, behavior: 'instant'});
    if (detail && current.anchor) {
      const anchor = document.getElementById(current.anchor);
      if (anchor && $('article').contains(anchor)) {const fold = anchor.closest('details'); if (fold) fold.open = true; anchor.scrollIntoView();}
    }
  }
  function doSearch(replace) {
    const next = nav.search(current, $('search').value);
    navigate(next, replace && !current.entry && !!current.q, false);
  }
  buildDirectory(); placeDirectory(); mobile.addEventListener('change', placeDirectory);
  $('menu-toggle').addEventListener('click', () => $('mobile-directory').showModal());
  $('close-menu').addEventListener('click', () => $('mobile-directory').close());
  $('chapter-list').addEventListener('click', event => {
    if (!event.target.closest('a.scope-link')) return;
    if ($('mobile-directory').open) $('mobile-directory').close();
  });
  $('search-form').addEventListener('submit', event => {event.preventDefault(); clearTimeout(searchTimer); doSearch(false);});
  $('search').addEventListener('input', () => {clearTimeout(searchTimer); searchTimer = setTimeout(() => doSearch(true), 160);});
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
