/* Pure mechanisms used by the reader, embedded experiments and regression tests. */
(function (root) {
  'use strict';
  const KEY = 'rl-infra-100days-questionbank-v1';
  const RATINGS = ['again', 'hard', 'good', 'easy'];
  const isObject = value => value && typeof value === 'object' && !Array.isArray(value);
  const copy = value => JSON.parse(JSON.stringify(value));
  const plain = text => String(text || '').replace(/<[^>]*>/g, ' ');
  const normalized = text => plain(text).normalize('NFKC').toLowerCase().replace(/\s+/g, ' ');
  function assertNumber(value, name, min = 0, integer = false) {
    if (!Number.isFinite(value) || value < min || (integer && !Number.isInteger(value))) {
      throw new Error('Invalid ' + name);
    }
    return value;
  }
  function route(hash) {
    const p = new URLSearchParams(String(hash || '').replace(/^#/, ''));
    // Old bookmarks remain usable after navigation redesign.
    if (p.has('category')) p.set('view', p.get('category') === 'case' ? 'questions' : 'library');
    return p;
  }
  function url(values = {}) {
    const p = new URLSearchParams();
    Object.entries(values).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') p.set(key, String(value));
    });
    return '#' + p.toString();
  }
  function key(entry) { return entry.type + ':' + (entry.code || entry.id); }
  function index(data) {
    const chapters = new Map(data.chapters.map(c => [c.id, c]));
    const sections = new Map(data.chapters.flatMap(c => (c.sections || []).map(s => [s.id, {...s, chapter: c.id}])));
    const topicSection = new Map([...sections.values()].flatMap(s => s.topics.map(id => [id, s.id])));
    const kinds = {topic: 'topics', case: 'cases', implementation: 'implementations'};
    const entries = Object.entries(kinds).flatMap(([type, group]) => data[group].map(item => ({...item, type})));
    const questions = (data.question_bank || []).map(q => ({...q, type: 'question', title: q.question,
      summary: q.brief, topics: q.topics || [q.topic]}));
    const items = [...entries, ...questions];
    const lookup = new Map(items.map(x => [key(x), x]));
    function topicsOf(item) { return item.type === 'topic' ? [item.id] : (item.topics || []); }
    function inChapter(item, id) {
      return !id || topicsOf(item).some(t => lookup.get('topic:' + t)?.chapter === id);
    }
    function inSection(item, id) {
      return !id || topicsOf(item).some(t => topicSection.get(t) === id);
    }
    function related(item) {
      const selected = new Map();
      const add = x => { if (x && key(x) !== key(item)) selected.set(key(x), x); };
      for (const [type, group] of Object.entries(kinds)) (item[group] || []).forEach(id => add(lookup.get(type + ':' + id)));
      if (item.type === 'topic') {
        items.filter(x => x.type !== 'topic' && topicsOf(x).includes(item.id)).forEach(add);
      } else {
        topicsOf(item).forEach(id => add(lookup.get('topic:' + id)));
      }
      return [...selected.values()];
    }
    const corpus = new Map(items.map(x => [key(x), normalized([x.code, x.title, x.summary, x.framework,
      x.html, x.deep, x.followup, x.doc_title, ...(x.keywords || [])].join(' '))]));
    function search(itemsToSearch, query) {
      const words = normalized(query).trim().split(/\s+/).filter(Boolean);
      return itemsToSearch.filter(x => words.every(w => corpus.get(key(x)).includes(w)));
    }
    return {chapters, sections, topicSection, entries, questions, items, lookup, topicsOf, inChapter, inSection, related, search};
  }
  function blankRecord() { return {answer: '', layers: {}, note: '', rating: '', attempts: 0, streak: 0, due: null, bookmarked: false, history: []}; }
  function dateAfter(offset = 0, base = new Date()) {
    const d = new Date(base); d.setHours(12, 0, 0, 0); d.setDate(d.getDate() + offset);
    return [d.getFullYear(), String(d.getMonth() + 1).padStart(2, '0'), String(d.getDate()).padStart(2, '0')].join('-');
  }
  function validateProgress(value, knownIds) {
    if (!isObject(value) || !isObject(value.records) ||
        !(value.version === 1 || value.format === 'rl-infra-interview-practice-v1')) throw new Error('Invalid review format');
    const records = {};
    for (const [id, r] of Object.entries(value.records)) {
      if (!/^\d+$/.test(id) || (knownIds && !knownIds.has(Number(id))) || !isObject(r)) throw new Error('Invalid question record: ' + id);
      if (r.rating && !RATINGS.includes(r.rating)) throw new Error('Invalid rating');
      for (const k of ['attempts', 'streak']) if (r[k] !== undefined) assertNumber(r[k], k, 0, true);
      for (const k of ['answer', 'note']) if (r[k] !== undefined && (typeof r[k] !== 'string' || r[k].length > 100000)) throw new Error('Invalid text field');
      if (r.due !== undefined && r.due !== null && !/^\d{4}-\d{2}-\d{2}$/.test(r.due)) throw new Error('Invalid review date');
      if (r.layers !== undefined && (!isObject(r.layers) || !Object.values(r.layers).every(v => typeof v === 'string'))) throw new Error('Invalid layers');
      if (r.bookmarked !== undefined && typeof r.bookmarked !== 'boolean') throw new Error('Invalid bookmark');
      if (r.history !== undefined && (!Array.isArray(r.history) || !r.history.every(h => isObject(h) && RATINGS.includes(h.rating) && typeof h.date === 'string'))) throw new Error('Invalid history');
      records[id] = {...blankRecord(), ...copy(r)};
    }
    return {version: 1, records};
  }
  function readProgress(storage, knownIds) {
    try { const text = storage.getItem(KEY); return {state: text ? validateProgress(JSON.parse(text), knownIds) : {version: 1, records: {}}, error: null}; }
    catch (error) { return {state: {version: 1, records: {}}, error: error.message}; }
  }
  function saveRecord(storage, id, change, knownIds, now = new Date()) {
    // Re-read immediately before write: do not overwrite edits from the legacy quiz tab.
    const loaded = readProgress(storage, knownIds);
    if (loaded.error) throw new Error('Review storage cannot be read; export or restore it before writing.');
    const state = loaded.state, record = {...blankRecord(), ...state.records[id]};
    if (change.rating) {
      if (!RATINGS.includes(change.rating)) throw new Error('Invalid rating');
      record.streak = ['good', 'easy'].includes(change.rating) ? record.streak + 1 : 0;
      const intervals = change.rating === 'easy' ? [7, 14, 30, 60, 90] : [3, 7, 14, 30, 60];
      const interval = change.rating === 'again' ? 0 : change.rating === 'hard' ? 1 : intervals[Math.min(record.streak - 1, 4)];
      record.attempts += 1; record.rating = change.rating; record.due = dateAfter(interval, now);
      record.history = record.history.concat({rating: change.rating, date: now.toISOString()}).slice(-30);
    }
    if ('bookmarked' in change) record.bookmarked = !!change.bookmarked;
    if ('note' in change) record.note = String(change.note);
    state.records[id] = record;
    storage.setItem(KEY, JSON.stringify(state));
    return state;
  }
  function mergeProgress(current, incoming) {
    // Import never silently replaces a record already on this device.
    const merged = copy(current); let added = 0, conflicts = 0;
    for (const [id, record] of Object.entries(incoming.records)) {
      if (Object.hasOwn(merged.records, id)) conflicts += 1;
      else {merged.records[id] = record; added += 1;}
    }
    return {state: merged, added, conflicts};
  }
  function tpCost({p = 4, tokens = 16, tflops = 80, efficiency = .55, penalty = .15, hbmGBs = 1000, bandwidth = 200, alpha = 2, launch = 5} = {}) {
    [p, tokens].forEach((x, i) => assertNumber(x, ['p', 'tokens'][i], 1, true));
    [tflops, efficiency, hbmGBs, bandwidth].forEach(x => assertNumber(x, 'rate', .00001));
    [penalty, alpha, launch].forEach(x => assertNumber(x, 'overhead'));
    const h = 4096, ffn = 8192, bytes = 2, flops = 4 * tokens * h * ffn;
    const weights = 2 * h * ffn * bytes, message = tokens * h * bytes;
    const eta = efficiency * Math.max(.05, 1 - penalty * Math.log2(p) / 3);
    const compute = flops / (p * tflops * 1e12 * eta) * 1e6;
    // Weight-read-only HBM floor; activation traffic/cache effects are not modelled.
    const memory = weights / p / (hbmGBs * 1e9) * 1e6;
    const wireBytes = p === 1 ? 0 : 2 * (p - 1) / p * message;
    const communication = p === 1 ? 0 : wireBytes / (bandwidth * 1e9) * 1e6 + 2 * (p - 1) * alpha;
    const gemm = Math.max(compute, memory);
    return {p, tokens, flops, weights, message, eta, compute, memory, gemm, wireBytes, communication, total: gemm + 2 * launch + communication};
  }
  function pipeline({stages = 4, batches = 8, schedule = '1f1b', slowStage = -1, slowFactor = 1} = {}) {
    assertNumber(stages, 'stages', 1, true); assertNumber(batches, 'batches', 1, true); assertNumber(slowFactor, 'slowFactor', 1, true);
    if (stages > 16 || batches > 64 || !['1f1b', 'gpipe'].includes(schedule)) throw new Error('Unsupported schedule size');
    const queues = Array.from({length: stages}, (_, rank) => {
      if (schedule === 'gpipe') return [...Array.from({length: batches}, (_, mb) => ({kind: 'F', mb})), ...Array.from({length: batches}, (_, mb) => ({kind: 'B', mb}))];
      const warm = Math.min(stages - rank - 1, batches), sequence = [];
      for (let mb = 0; mb < warm; mb++) sequence.push({kind: 'F', mb});
      for (let mb = 0; mb < batches - warm; mb++) sequence.push({kind: 'F', mb: mb + warm}, {kind: 'B', mb});
      for (let mb = batches - warm; mb < batches; mb++) sequence.push({kind: 'B', mb});
      return sequence;
    });
    const ptr = Array(stages).fill(0), running = Array(stages).fill(null), events = [], done = new Set();
    let tick = 0; const id = (rank, kind, mb) => `${rank}:${kind}:${mb}`;
    while (events.length < 2 * stages * batches || running.some(Boolean)) {
      for (let r = 0; r < stages; r++) if (running[r] && running[r].end <= tick) {
        const e = running[r]; done.add(id(r, e.kind, e.mb)); running[r] = null;
      }
      for (let r = 0; r < stages; r++) {
        if (running[r]) continue;
        const op = queues[r][ptr[r]]; if (!op) continue;
        const ready = op.kind === 'F' ? (r === 0 || done.has(id(r - 1, 'F', op.mb))) :
          done.has(id(r, 'F', op.mb)) && (r === stages - 1 || done.has(id(r + 1, 'B', op.mb)));
        if (ready) {
          const event = {...op, rank: r, start: tick, end: tick + (r === slowStage ? slowFactor : 1)};
          events.push(event); running[r] = event; ptr[r] += 1;
        }
      }
      if (++tick > 10000) throw new Error('Schedule deadlock');
    }
    const duration = Math.max(...events.map(e => e.end)), busy = events.reduce((s, e) => s + e.end - e.start, 0);
    const activationPeak = Array.from({length: stages}, (_, r) => {
      let peak = 0;
      for (let t = 0; t <= duration; t++) {
        const f = events.filter(e => e.rank === r && e.kind === 'F' && e.end <= t).length;
        const b = events.filter(e => e.rank === r && e.kind === 'B' && e.end <= t).length;
        peak = Math.max(peak, f - b);
      }
      return peak;
    });
    return {events, duration, idle: 1 - busy / (stages * duration), activationPeak};
  }
  function dispatch({ep = 4, hot = false} = {}) {
    if (![1, 2, 4, 8].includes(ep)) throw new Error('Invalid EP');
    const experts = 8, tokens = 16;
    const routes = Array.from({length: tokens}, (_, token) => {
      const selected = hot ? [0, 1 + token % 7] : [token % experts, (token + 3) % experts];
      return {token, source: token % ep, selected, targets: selected.map(e => Math.floor(e / (experts / ep)))};
    });
    const load = Array(experts).fill(0), rankLoad = Array(ep).fill(0); let remote = 0;
    routes.forEach(r => r.selected.forEach((e, i) => {load[e]++; rankLoad[r.targets[i]]++; if (r.targets[i] !== r.source) remote++;}));
    return {routes, load, rankLoad, remote, assignments: 32, maxMean: Math.max(...rankLoad) / (32 / ep)};
  }
  function topology({total = 16, tp = 2, pp = 2, cp = 1, selected = 0} = {}) {
    [total, tp, pp, cp].forEach(x => assertNumber(x, 'degree', 1, true));
    const dp = total / (tp * pp * cp);
    if (!Number.isInteger(dp) || dp < 1 || selected < 0 || selected >= total) throw new Error('TP * PP * CP must divide total GPUs');
    const coords = rank => ({tp: rank % tp, cp: Math.floor(rank / tp) % cp, dp: Math.floor(rank / (tp * cp)) % dp, pp: Math.floor(rank / (tp * cp * dp))});
    const point = coords(selected), ranks = Array.from({length: total}, (_, rank) => ({rank, node: Math.floor(rank / 8), ...coords(rank)}));
    const groups = Object.fromEntries(['tp', 'pp', 'cp', 'dp'].map(axis => [axis, ranks.filter(r => ['tp', 'pp', 'cp', 'dp'].filter(a => a !== axis).every(a => r[a] === point[a])).map(r => r.rank)]));
    return {dp, ranks, groups};
  }
  function memory({params = 7, total = 16, tp = 2, pp = 1, shard = 1, capacity = 80, activation = 12, buffer = 8,
      layers = 32, kvHeads = 8, headDim = 128, length = 65536, concurrency = 4, kvBytes = 2, master = true} = {}) {
    const vars = {params, capacity, activation, buffer}; Object.entries(vars).forEach(([k, v]) => assertNumber(v, k, k === 'activation' || k === 'buffer' ? 0 : .001));
    Object.entries({total, tp, pp, shard, layers, kvHeads, headDim, length, concurrency, kvBytes}).forEach(([k, v]) => assertNumber(v, k, 1, true));
    const dp = total / (tp * pp);
    if (!Number.isInteger(dp) || dp < 1 || dp % shard) throw new Error('Invalid TP/PP/DP state-sharding degree');
    if (layers % pp || kvHeads % tp) throw new Error('This model requires uniform layer and KV-head sharding');
    const gib = 2 ** 30, count = params * 1e9;
    const weight = count * 2 / (tp * pp) / gib;
    const state = count * (master ? 14 : 10) / (tp * pp * shard) / gib;
    const fullKV = 2 * layers * kvHeads * headDim * length * kvBytes / gib;
    const kv = fullKV * concurrency / (tp * pp);
    return {dp, weight, state, fullKV, kv, activation, buffer, learner: weight + state + activation + buffer, rollout: weight + kv + buffer, capacity};
  }
  function bandit(p, count = 32, seed = 17) {
    assertNumber(p, 'probability'); assertNumber(count, 'count', 1, true);
    if (p > 1 || count > 10000) throw new Error('Invalid bandit size');
    let state = seed | 0, hits = 0;
    for (let i = 0; i < count; i++) {
      let t = state += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      if (((t ^ (t >>> 14)) >>> 0) / 4294967296 < p) hits++;
    }
    return {hits, observed: hits / count, expected: p, gradient: p * (1 - p), pass4: 1 - (1 - p) ** 4};
  }
  function baseline(p, b) {
    assertNumber(p, 'p'); if (p > 1 || !Number.isFinite(b)) throw new Error('Invalid baseline');
    const a = (1 - b) * (1 - p), other = b * p, mean = p * a + (1 - p) * other;
    return {a, other, mean, variance: p * (a - mean) ** 2 + (1 - p) * (other - mean) ** 2};
  }
  function ppo(ratio, advantage, epsilon) {
    assertNumber(ratio, 'ratio'); assertNumber(epsilon, 'epsilon');
    if (!Number.isFinite(advantage) || epsilon > 1) throw new Error('Invalid PPO input');
    const clipped = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio));
    return Math.min(ratio * advantage, clipped * advantage);
  }
  function groupAdvantages(rewards) {
    if (!Array.isArray(rewards) || !rewards.length || rewards.some(x => !Number.isFinite(x))) throw new Error('Invalid rewards');
    const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
    const std = Math.sqrt(rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / rewards.length);
    return {mean, std, values: rewards.map(x => (x - mean) / (std + 1e-8))};
  }
  const api = {KEY, RATINGS, normalized, route, url, key, index, blankRecord, dateAfter, validateProgress, readProgress,
    saveRecord, mergeProgress, bandit, baseline, ppo, groupAdvantages, tpCost, pipeline, dispatch, topology, memory};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.RLCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
