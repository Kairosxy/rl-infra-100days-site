'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const C = require('../knowledge/site/core.js');
const N = require('../knowledge/site/navigation.js');
const catalog = JSON.parse(fs.readFileSync('knowledge/catalog.json', 'utf8'));
const map = JSON.parse(fs.readFileSync('knowledge/learning-map.json', 'utf8'));
const bank = JSON.parse(fs.readFileSync('quiz/data/questions.json', 'utf8'));
catalog.question_bank = bank.questions.map(q => ({...q, topics: map.question_links?.[q.code] || [q.topic]}));
const nav = N.create(catalog), graph = nav.graph;
let checks = 0;
function test(name, fn) {fn(); console.log('ok:', name); checks++;}
const tp = graph.lookup.get('topic:tp-matrix-communication');
const tpScope = nav.normalizeScope({ref: tp.id});
const items = s => nav.resultSet(s).at(s.scope);
const linked = graph.items.filter(x => graph.topicsOf(x).includes(tp.id));
test('one scope selects all three content modes, without duplicates', () => {
  const state = nav.selectScope(nav.read(''), tpScope);
  const counts = nav.resultSet(state).counts(state.scope);
  assert.equal(counts.topic, 1);
  assert.equal(counts.all, linked.length);
  assert.equal(counts.all, counts.topic + counts.issues + counts.implementation);
  for (const mode of N.MODES) {
    const next = nav.read(nav.href(nav.selectMode(state, mode)));
    assert.deepEqual(next.scope, tpScope);
    assert.equal(items(next).length, counts[mode]);
    assert.equal(new Set(items(next).map(C.key)).size, counts[mode]);
  }
});
test('directory selection preserves content mode and query, resets pagination', () => {
  const s = nav.read('#mode=issues&area=parallel&q=TP&page=3&kind=question');
  const next = nav.read(nav.href(nav.selectScope(s, {ref:tp.id})));
  assert.equal(next.mode, 'issues'); assert.equal(next.q, 'TP'); assert.equal(next.kind, 'question');
  assert.equal(next.page, 1); assert.deepEqual(next.scope, tpScope);
  assert.ok(items(next).every(x => x.type === 'question' && graph.topicsOf(x).includes(tp.id)));
});
test('search result detail exposes unfiltered related contents and exact return state', () => {
  const state = nav.read('#mode=issues&q=Q086&kind=question&page=2');
  const item = graph.lookup.get('question:Q086');
  const detail = nav.read(nav.href(nav.open(state, item)));
  assert.equal(detail.entry.code, 'Q086'); assert.equal(detail.q, '');
  assert.equal(detail.mode, 'issues'); assert.equal(detail.kind, '');
  assert.equal(nav.back(detail), nav.href(state));
  const restored = nav.read(nav.back(detail)); assert.equal(restored.page, 2); assert.equal(restored.q, 'Q086');
  assert.ok(nav.resultSet(detail).counts(detail.scope).topic > 0);
});
test('framework scope includes related knowledge and issues, not unrelated implementations', () => {
  const s = nav.read('#area=frameworks&impl=megatron-parallel-groups');
  const impl = graph.lookup.get('implementation:megatron-parallel-groups');
  const counts = nav.resultSet(s).counts(s.scope);
  assert.equal(counts.implementation, 1); assert.ok(counts.issues > 0); assert.ok(counts.topic > 0);
  assert.ok(nav.inScope(tp, s.scope));
  assert.ok(!nav.inScope(graph.lookup.get('topic:policy-gradient'), s.scope));
  assert.ok(nav.inScope(impl, s.scope));
  assert.ok(!nav.inScope(graph.lookup.get('implementation:sglang-scheduler'), s.scope));
  const detail = nav.read(nav.href(nav.open(s, tp)));
  assert.equal(detail.scope.impl, impl.id);
  assert.equal(nav.back(detail), nav.href(s));
});
test('legacy URLs, all 224 direct links and all article IDs stay readable', () => {
  assert.equal(nav.read('#view=questions&kind=case').mode, 'issues');
  assert.equal(nav.read('#category=implementation').mode, 'implementation');
  assert.equal(nav.read('#view=library&kind=topic').mode, 'topic');
  assert.equal(nav.read('#chapter=parallel&section=parallel-2').scope.section, 'parallel-2');
  assert.equal(nav.read('#q=81').entry.code, 'Q081');
  assert.equal(nav.read('#view=paths').paths, true);
  for (const item of graph.items) {
    const s = nav.read('#' + item.type + '=' + (item.code || item.id));
    assert.equal(C.key(s.entry), C.key(item));
    assert.equal(C.key(nav.read(nav.href(s)).entry), C.key(item));
    assert.equal(s.mode, N.modeOf(item.type));
  }
  assert.equal(nav.resultSet(nav.read('#mode=issues&kind=question')).at(N.EMPTY).length, 224);
});
test('invalid routes are explicit and arbitrary return addresses cannot navigate offsite', () => {
  assert.ok(nav.read('#area=missing').invalid); assert.ok(nav.read('#question=Q999').invalid);
  assert.ok(nav.read('#topic=tp-matrix-communication&back=https%3A%2F%2Fexample.com').back === '');
  assert.ok(nav.read('#topic=tp-matrix-communication&back=%23topic%3Dpolicy-gradient').back === '');
});
test('review filters apply only to issues and counts match the next selected mode', () => {
  const state = nav.read('#mode=issues&review=bookmarked'), records = {81:{bookmarked:true}};
  const results = nav.resultSet(state, records);
  assert.equal(results.at(N.EMPTY).length, 1);
  assert.equal(results.counts(N.EMPTY).all, graph.items.length);
  const all = nav.selectMode(state, 'all'); assert.equal(all.review, '');
  assert.equal(nav.resultSet(all, records).at(N.EMPTY).length, graph.items.length);
});
console.log(`Navigation checks passed: ${checks}`);
