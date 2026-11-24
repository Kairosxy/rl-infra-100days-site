/* One URL model for the directory, content types, search and detail routes. */
(function (root) {
  'use strict';
  const C = root.RLCore || (typeof require === 'function' ? require('./core.js') : null);
  const MODES = ['all', 'topic', 'issues', 'implementation'];
  const TYPES = ['topic', 'case', 'implementation', 'question'];
  const EMPTY = Object.freeze({area: '', section: '', ref: '', framework: '', impl: ''});
  const modeOf = type => ['case', 'question'].includes(type) ? 'issues' : type;
  const validMode = value => MODES.includes(value) ? value : 'all';
  const scopeKey = s => [s.area, s.section, s.ref, s.framework, s.impl].join('|');
  const pageNumber = value => Math.max(1, Math.min(100000, Math.floor(Number(value)) || 1));

  function create(data, graph = C.index(data)) {
    const frameworks = new Map();
    for (const item of graph.entries.filter(x => x.type === 'implementation')) {
      if (!frameworks.has(item.framework)) frameworks.set(item.framework, []);
      frameworks.get(item.framework).push(item);
    }
    function normalizeScope(raw = {}) {
      const topic = graph.lookup.get('topic:' + raw.ref);
      if (topic) return {...EMPTY, area: topic.chapter, section: graph.topicSection.get(topic.id) || '', ref: topic.id};
      const impl = graph.lookup.get('implementation:' + raw.impl);
      if (impl) return {...EMPTY, area: 'frameworks', framework: impl.framework, impl: impl.id};
      if (frameworks.has(raw.framework)) return {...EMPTY, area: 'frameworks', framework: raw.framework};
      const section = graph.sections.get(raw.section);
      if (section) return {...EMPTY, area: section.chapter, section: section.id};
      return {...EMPTY, area: graph.chapters.has(raw.area) ? raw.area : ''};
    }
    const members = new Map();
    function inScope(item, scope) {
      const s = normalizeScope(scope);
      if (!s.area) return true;
      if (s.area !== 'frameworks') {
        return graph.inChapter(item, s.area) && graph.inSection(item, s.section) &&
          (!s.ref || graph.topicsOf(item).includes(s.ref));
      }
      const k = scopeKey(s);
      if (!members.has(k)) {
        const impls = (s.framework ? frameworks.get(s.framework) : [...frameworks.values()].flat())
          .filter(x => !s.impl || x.id === s.impl);
        members.set(k, {impls: new Set(impls.map(x => x.id)),
          topics: new Set(impls.flatMap(x => x.topics || [])), cases: new Set(impls.flatMap(x => x.cases || []))});
      }
      const related = members.get(k);
      if (item.type === 'implementation') return related.impls.has(item.id);
      return (item.type === 'case' && related.cases.has(item.id)) || graph.topicsOf(item).some(id => related.topics.has(id));
    }
    function naturalScope(item) {
      if (item.type === 'implementation') return normalizeScope({impl: item.id});
      const topic = item.type === 'topic' ? item : graph.lookup.get('topic:' + item.topics?.[0]);
      return normalizeScope(topic ? {ref: topic.id} : {});
    }
    function detailScope(item, previous) {
      const s = normalizeScope(previous);
      if (item.type === 'topic' && s.area !== 'frameworks') return naturalScope(item);
      if (s.area && inScope(item, s)) return s;
      return naturalScope(item);
    }
    function clean(state = {}) {
      const mode = validMode(state.mode);
      return {scope: normalizeScope(state.scope), mode, q: String(state.q || '').trim().slice(0, 500),
        kind: mode === 'issues' && ['case', 'question'].includes(state.kind) ? state.kind : '',
        level: mode === 'issues' && state.kind !== 'case' && ['基础', '进阶'].includes(state.level) ? state.level : '',
        review: mode === 'issues' && state.kind !== 'case' && ['due', 'weak', 'bookmarked', 'new'].includes(state.review) ? state.review : '',
        page: pageNumber(state.page), entry: state.entry || null, anchor: String(state.anchor || ''),
        back: String(state.back || ''), paths: !!state.paths, invalid: String(state.invalid || '')};
    }
    function read(hash, allowBack = true) {
      const p = new URLSearchParams(String(hash || '').replace(/^#/, ''));
      let type = TYPES.find(k => p.has(k));
      let id = type ? p.get(type) : '';
      if (!type && !p.has('view') && !p.has('mode') && /^\d+$/.test(p.get('q') || '')) {
        type = 'question'; id = 'Q' + p.get('q').padStart(3, '0'); p.delete('q');
      }
      const entry = type ? graph.lookup.get(type + ':' + id) : null;
      const legacyKind = p.get('kind') || p.get('category') || '';
      const mode = p.has('mode') ? validMode(p.get('mode')) :
        p.get('view') === 'questions' || ['case', 'question'].includes(legacyKind) ? 'issues' :
        ['topic', 'implementation'].includes(legacyKind) ? legacyKind :
        entry ? modeOf(entry.type) : 'all';
      const raw = {area: p.get('area') || p.get('chapter'), section: p.get('section'), ref: p.get('ref'), framework: p.get('framework'), impl: p.get('impl')};
      const explicit = Object.values(raw).some(Boolean);
      const scope = normalizeScope(raw);
      const invalid = (type && !entry) ? type + ':' + id :
        raw.ref && !graph.lookup.has('topic:' + raw.ref) ? raw.ref :
        raw.impl && !graph.lookup.has('implementation:' + raw.impl) ? raw.impl :
        raw.framework && !frameworks.has(raw.framework) ? raw.framework :
        raw.section && !graph.sections.has(raw.section) ? raw.section :
        raw.area && raw.area !== 'all' && !graph.chapters.has(raw.area) ? raw.area : '';
      let back = '';
      if (allowBack && /^#(?:[\w%=&+.-]*)$/.test(p.get('back') || '')) {
        const origin = read(p.get('back'), false);
        if (!origin.entry && !origin.invalid) back = href(origin);
      }
      return clean({scope: entry ? (explicit && inScope(entry, scope) ? scope : naturalScope(entry)) : scope,
        mode: entry ? modeOf(entry.type) : mode, q: entry ? '' : p.get('q'), kind: entry ? '' : legacyKind,
        level: entry ? '' : p.get('level'), review: entry ? '' : p.get('review'), page: entry ? 1 : p.get('page'),
        entry, anchor: entry ? p.get('anchor') : '', back, paths: !entry && p.get('view') === 'paths', invalid});
    }
    function href(state) {
      const s = clean(state), p = {};
      if (s.paths) return '#view=paths';
      if (s.mode !== 'all' || s.entry) p.mode = s.mode;
      if (s.scope.area) p.area = s.scope.area;
      else if (s.entry) p.area = 'all';
      for (const key of ['section', 'ref', 'framework', 'impl']) if (s.scope[key]) p[key] = s.scope[key];
      for (const key of ['q', 'kind', 'level', 'review']) if (s[key]) p[key] = s[key];
      if (s.page > 1) p.page = s.page;
      if (s.entry) {
        p[s.entry.type] = s.entry.code || s.entry.id;
        if (s.anchor) p.anchor = s.anchor;
        if (s.back) p.back = s.back;
      }
      return C.url(p);
    }
    function list(state, change = {}) {
      return clean({...state, entry: null, anchor: '', back: '', paths: false, invalid: '', ...change});
    }
    function selectScope(state, raw) {return list(state, {scope: normalizeScope(raw), page: 1});}
    function selectMode(state, mode) {return list(state, {mode, page: 1, ...(mode !== state.mode ? {kind: '', level: '', review: ''} : {})});}
    function search(state, q) {return list(state, {q, page: 1});}
    function open(state, item) {
      const mode = modeOf(item.type), origin = state.entry ? (state.back || href(list(state))) : href(state);
      return clean({...state, scope: detailScope(item, state.scope), mode, q: '', page: 1,
        kind: '', level: '', review: '', entry: item, anchor: '', back: origin, paths: false, invalid: ''});
    }
    function back(state) {return state.back || href(list(state));}
    function issueMatch(item, state, records) {
      if (state.kind && item.type !== state.kind) return false;
      if (state.level && item.level !== state.level) return false;
      if (!state.review) return true;
      if (item.type !== 'question') return false;
      const r = records[item.id] || {};
      if (state.review === 'due') return !!r.due && r.due <= C.dateAfter();
      if (state.review === 'weak') return ['again', 'hard'].includes(r.rating);
      if (state.review === 'bookmarked') return !!r.bookmarked;
      return !r.attempts;
    }
    function resultSet(state, records = {}) {
      const matched = graph.search(graph.items, state.q || '');
      const byMode = {
        topic: matched.filter(x => x.type === 'topic'),
        issues: matched.filter(x => modeOf(x.type) === 'issues' && issueMatch(x, state, records)),
        implementation: matched.filter(x => x.type === 'implementation')
      };
      byMode.all = [...byMode.topic, ...matched.filter(x => modeOf(x.type) === 'issues'), ...byMode.implementation];
      const cache = new Map();
      function at(scope, mode = state.mode) {
        const s = normalizeScope(scope), k = scopeKey(s) + ':' + mode;
        if (!cache.has(k)) cache.set(k, byMode[validMode(mode)].filter(item => inScope(item, s)));
        return cache.get(k);
      }
      return {at, counts: s => Object.fromEntries(MODES.map(m => [m, at(s, m).length]))};
    }
    function trail(scope) {
      const s = normalizeScope(scope), nodes = [{title: '全部内容', scope: {...EMPTY}}];
      if (s.area) nodes.push({title: graph.chapters.get(s.area).title, scope: normalizeScope({area: s.area})});
      if (s.section) nodes.push({title: graph.sections.get(s.section).title, scope: normalizeScope({section: s.section})});
      if (s.ref) nodes.push({title: graph.lookup.get('topic:' + s.ref).title, scope: s});
      if (s.framework) nodes.push({title: s.framework, scope: normalizeScope({framework: s.framework})});
      if (s.impl) nodes.push({title: graph.lookup.get('implementation:' + s.impl).title, scope: s});
      return nodes;
    }
    function tree() {
      return data.chapters.map(ch => ({title: ch.title, scope: normalizeScope({area: ch.id}), children:
        ch.id === 'frameworks' ? [...frameworks].map(([title, items]) => ({title, scope: normalizeScope({framework: title}), children: items.map(x => ({title: x.title, scope: normalizeScope({impl: x.id}), children: []}))})) :
          (ch.sections || []).map(sec => ({title: sec.title, scope: normalizeScope({section: sec.id}), children: sec.topics.map(id => ({title: graph.lookup.get('topic:' + id).title, scope: normalizeScope({ref: id}), children: []}))}))}));
    }
    return {graph, normalizeScope, inScope, naturalScope, read, href, list, selectScope, selectMode, search, open, back, resultSet, trail, tree, scopeKey};
  }
  const api = {create, MODES, EMPTY, modeOf};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.RLNavigation = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
