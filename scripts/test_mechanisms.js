'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const M = require('../knowledge/site/core.js');
let checks = 0;
function test(name, fn) {fn(); console.log('ok:', name); checks++;}
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);
test('Ring bytes and weight count', () => {const r = M.tpCost({p:4,tokens:4});assert.equal(r.wireBytes,49152);assert.equal(r.weights/2**20,128);assert.equal(M.tpCost({p:1}).communication,0);});
test('HBM floor is explicit and TP does not create speed from nothing', () => {const r=M.tpCost({p:1,tokens:1});assert.equal(r.gemm,Math.max(r.compute,r.memory));assert.ok(r.memory>r.compute);assert.throws(()=>M.tpCost({bandwidth:0}));});
for(const stages of [1,2,4,8])for(const batches of [2,4,8])for(const schedule of ['1f1b','gpipe'])test(`pipeline ${stages}/${batches}/${schedule}`,()=>{
 const r=M.pipeline({stages,batches,schedule});assert.equal(r.events.length,2*stages*batches);
 const event=(rank,kind,mb)=>r.events.find(x=>x.rank===rank&&x.kind===kind&&x.mb===mb);
 for(const x of r.events){if(x.kind==='F'&&x.rank>0)assert.ok(event(x.rank-1,'F',x.mb).end<=x.start);if(x.kind==='B'){assert.ok(event(x.rank,'F',x.mb).end<=x.start);if(x.rank<stages-1)assert.ok(event(x.rank+1,'B',x.mb).end<=x.start);}}
 for(let i=0;i<stages;i++){const es=r.events.filter(e=>e.rank===i).sort((a,b)=>a.start-b.start);for(let j=1;j<es.length;j++)assert.ok(es[j-1].end<=es[j].start);}
 close(r.idle,(stages-1)/(batches+stages-1));
});
test('1F1B reduces activation residency without promising a bubble reduction',()=>{const a=M.pipeline({schedule:'1f1b'}),b=M.pipeline({schedule:'gpipe'});assert.equal(a.duration,b.duration);assert.ok(Math.max(...a.activationPeak)<Math.max(...b.activationPeak));assert.ok(M.pipeline({slowStage:0,slowFactor:2}).duration>a.duration);});
for(const ep of [1,2,4,8])for(const hot of [false,true])test(`MoE conservation ep=${ep} hot=${hot}`,()=>{const r=M.dispatch({ep,hot});assert.equal(r.load.length,8);assert.equal(r.load.reduce((a,b)=>a+b),32);assert.equal(r.rankLoad.reduce((a,b)=>a+b),32);assert.ok(r.routes.every(x=>x.selected[0]!==x.selected[1]));if(ep===1)assert.equal(r.remote,0);});
test('Dense rank groups are bijective and preserve other coordinates',()=>{for(const total of [16,64,128]){const r=M.topology({total,tp:2,pp:2,cp:2,selected:7});assert.equal(r.dp,total/8);assert.equal(r.ranks.length,total);assert.equal(r.groups.tp.length,2);assert.equal(r.groups.cp.length,2);assert.equal(r.groups.pp.length,2);assert.equal(r.groups.dp.length,total/8);}assert.throws(()=>M.topology({total:16,tp:8,pp:4}));});
test('KV shards across both heads and layers',()=>{const a=M.memory({tp:2,pp:1}),b=M.memory({tp:2,pp:2});close(a.fullKV,8);close(b.kv,a.kv/2);close(b.weight,a.weight/2);assert.throws(()=>M.memory({kvHeads:3,tp:2}));assert.throws(()=>M.memory({params:NaN}));});
test('Sampling uses a reproducible seed',()=>{assert.deepEqual(M.bandit(.3,32,17),M.bandit(.3,32,17));assert.equal(M.bandit(1).observed,1);assert.equal(M.bandit(0).observed,0);assert.throws(()=>M.bandit(2));});
test('Baseline preserves expectation',()=>{for(const p of [.1,.3,.8])for(const b of [-1,0,.5,1,2])close(M.baseline(p,b).mean,p*(1-p));close(M.baseline(.3,.7).variance,0);});
test('PPO has an advantage-dependent plateau',()=>{close(M.ppo(1.5,1,.2),1.2);close(M.ppo(.5,-1,.2),-.8);close(M.ppo(.5,1,.2),.5);close(M.ppo(1.5,-1,.2),-1.5);});
test('GRPO preserves sample order and equal rewards have zero relative signal',()=>{assert.deepEqual(M.groupAdvantages([1,1,1,1]).values,[0,0,0,0]);const r=M.groupAdvantages([1,0,1,0]);assert.ok(r.values[0]>0&&r.values[1]<0);close(r.values.reduce((a,b)=>a+b),0);});
test('All 224 source questions are preserved',()=>{const bank=JSON.parse(fs.readFileSync('quiz/data/questions.json','utf8'));assert.equal(bank.questions.length,224);bank.questions.forEach((q,i)=>assert.equal(q.code,'Q'+String(i+1).padStart(3,'0')));});
test('Review writes preserve other records, drafts and legacy format',()=>{
 let value=JSON.stringify({version:1,records:{1:{...M.blankRecord(),answer:'sentinel',layers:{c1:'draft'},note:'note'},2:{...M.blankRecord(),note:'other'}}});
 const storage={getItem:()=>value,setItem:(k,v)=>{assert.equal(k,M.KEY);value=v;}};
 const state=M.saveRecord(storage,1,{rating:'good'},new Set([1,2]),new Date('2026-10-28T04:00:00Z'));
 assert.equal(state.records[1].answer,'sentinel');assert.equal(state.records[1].layers.c1,'draft');assert.equal(state.records[2].note,'other');assert.equal(state.records[1].due,'2026-10-31');assert.ok(state.records[1].history[0].date.endsWith('Z'));
 const b=M.saveRecord(storage,1,{bookmarked:true},new Set([1,2]));assert.equal(b.records[1].attempts,1);
});
test('Import validates first and never replaces existing local records',()=>{assert.throws(()=>M.validateProgress({version:1,records:{999:{}}},new Set([1])));assert.throws(()=>M.validateProgress({version:1,records:[]}));const a={version:1,records:{1:M.blankRecord()}},b={version:1,records:{1:{...M.blankRecord(),note:'conflict'},2:M.blankRecord()}};const r=M.mergeProgress(a,b);assert.equal(r.added,1);assert.equal(r.conflicts,1);assert.equal(r.state.records[1].note,'');});
test('Storage read failure cannot be mistaken for empty valid storage',()=>{const st={getItem:()=>'{broken',setItem:()=>assert.fail('must not overwrite')};assert.ok(M.readProgress(st).error);assert.throws(()=>M.saveRecord(st,1,{rating:'good'},new Set([1])));});
console.log(`Mechanism checks passed: ${checks}`);
