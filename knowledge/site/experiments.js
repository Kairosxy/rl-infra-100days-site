/* Native document figures. No iframe, remote scripts, timers or analytics. */
(function (root) {
  'use strict';
  const M = root.RLCore;
  const e = (tag, cls, text) => {const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n;};
  const fmt = (n, digits = 2) => Number(n).toLocaleString('en-US', {maximumFractionDigits: digits});
  function table(headers, rows) {
    const wrap = e('div', 'table-scroll'); wrap.tabIndex = 0;
    const t = e('table'); const head = e('thead'), row = e('tr');
    headers.forEach(x => {const th = e('th', '', x); th.scope = 'col'; row.append(th);}); head.append(row); t.append(head);
    const body = e('tbody'); rows.forEach(values => {const tr = e('tr'); values.forEach(v => tr.append(e('td', '', String(v)))); body.append(tr);});
    t.append(body); wrap.append(t); return wrap;
  }
  function mount(host, type) {
    host.className = 'experiment'; host.dataset.experiment = type;
    const titles = {tp: 'TP · 切开后怎么合并\uff1f', pp: 'PP · 看一次完整前反向', ep: 'EP · 专家均分\uff0cToken 不一定均分', topology: '并行组 · 谁和谁通信\uff1f', cost: '成本 · 单次延迟与固定 GPU 吞吐', memory: '显存 · Learner 与 Rollout 分别核算', context: 'CP / SP · 切分的是哪个维度\uff1f', sharding: '状态分片 · 常驻与计算峰值', collective: 'Collective · 输出是拼接还是求和\uff1f'};
    Object.assign(titles, {policy:'采样与期望', baseline:'Baseline 与方差', mask:'Token Shift / Action Mask', ppo:'PPO Clipped Surrogate', grpo:'GRPO Group Advantage'});
    host.append(e('p', 'caption', '文内实验 / 教学模型'), e('h3', '', titles[type] || type));
    const controls = e('div', 'experiment-controls'), out = e('div', 'experiment-output'), status = e('p', 'status'); status.setAttribute('role', 'status');
    const inputs = {};
    function select(key, label, options, value, parent = controls) {
      const field = e('label', 'field'); field.append(e('span', '', label));
      const input = e('select'); input.dataset.control = key;
      options.forEach(x => {const [v, t] = Array.isArray(x) ? x : [x, x]; const o = e('option', '', String(t)); o.value = String(v); input.append(o);});
      input.value = String(value); field.append(input); parent.append(field); inputs[key] = input; return input;
    }
    function number(key, label, value, min = 0, step = 1, parent = controls) {
      const field = e('label', 'field'); field.append(e('span', '', label));
      const input = e('input'); input.type = 'number'; input.min = String(min); input.step = String(step); input.value = String(value); input.dataset.control = key;
      inputs[key] = input; field.append(input); parent.append(field); return input;
    }
    const n = key => Number(inputs[key].value), s = key => inputs[key].value;
    const note = text => out.append(e('p', 'model-note', text));
    function metrics(pairs) {const dl = e('dl', 'measurements'); pairs.forEach(([label, value]) => {const div = e('div'); div.append(e('dt', '', label), e('dd', '', String(value))); dl.append(div);}); out.append(dl);}
    const degrees = [1, 2, 4, 8]; let draw;
    if (type === 'policy') {
      number('p', 'P(A)', .25, 0, .05); select('n', '采样次数', [8, 32, 128, 512], 32); number('seed', 'Seed', 17, 0, 1);
      draw = () => {
        const d = M.bandit(n('p'), n('n'), n('seed'));
        metrics([['E[reward]', fmt(d.expected, 4)], ['样本均值', fmt(d.observed, 4)], ['dJ/dz, p=sigmoid(z)', fmt(d.gradient, 4)]]);
        out.append(table(['指标', '数值'], [['A / samples', d.hits + '/' + n('n')], ['Independent pass@4', fmt(d.pass4, 4)]]));
        note('Reward(A)=1, Reward(B)=0。固定 Seed 可复现样本\uff1b小样本均值不等于策略期望。Pass@4 假定四次独立采样。');
      };
    } else if (type === 'baseline') {
      number('p', 'P(A)', .25, 0, .05); number('b', 'Baseline b', .5, -2, .05);
      draw = () => {const d = M.baseline(n('p'), n('b')); metrics([['E[g]', fmt(d.mean, 4)], ['Var[g]', fmt(d.variance, 6)], ['p(1-p)', fmt(n('p') * (1 - n('p')), 4)]]); out.append(table(['采样动作', 'gradient'], [['A', fmt(d.a, 4)], ['B', fmt(d.other, 4)]])); note('这是动作无关、停止梯度的 Baseline。期望不变\uff0c但不是任意 b 都会降低方差。本例最优 b=1-p。');};
    } else if (type === 'mask') {
      select('prompt', 'Prompt tokens', [3, 4, 5], 5); select('wrong', '预测位置', [[0, 'logits[t-1]'], [1, 'logits[t] (wrong)']], 0);
      draw = () => {const tokens = ['<s>', '2', '+', '2', '=', '4', '<pad>']; out.append(table(['t', 'Token', 'Mask', 'Predictor'], tokens.map((t, i) => [i, t, i >= n('prompt') && i < 6 ? 1 : 0, i >= n('prompt') && i < 6 ? 'logits[' + (n('wrong') ? i : i - 1) + ']' : '—']))); note(n('wrong') ? '错位不改变 Shape\uff0c但 logits[t] 预测的是下一个 Token\uff0c不是 token[t]。' : '每个 Action token[t] 由 logits[t-1] 预测\uff1bPrompt 和 PAD 不参与该 Action loss。');};
    } else if (type === 'ppo') {
      number('ratio', 'Ratio r', 1.3, 0, .05); select('advantage', 'Advantage', [-2, -1, 1, 2], 1); number('epsilon', 'epsilon', .2, 0, .05);
      draw = () => {const r = n('ratio'), a = n('advantage'), eps = n('epsilon'); const value = M.ppo(r, a, eps); metrics([['r*A', fmt(r * a)], ['clipped surrogate', fmt(value)], ['局部斜率', Math.abs(r - (a > 0 ? 1 + eps : 1 - eps)) < 1e-8 ? '分段边界' : (a > 0 && r > 1 + eps) || (a < 0 && r < 1 - eps) ? '0' : a]]); out.append(table(['r', 'J(r)'], [.5, .8, 1, 1.2, 1.5].map(r => [r, fmt(M.ppo(r, a, eps))]))); note('J=min(rA, clip(r,1-epsilon,1+epsilon)A)。这是单个样本的 surrogate\uff0c不是全策略 KL 的硬约束。');};
    } else if (type === 'grpo') {
      [0, 1, 2, 3].forEach(i => select('r' + i, 'Reward ' + (i + 1), [0, 1], i % 2 ? 0 : 1)); select('incomplete', '回答组', [[0, '4 / 4'], [1, '3 / 4']], 0);
      draw = () => {const count = n('incomplete') ? 3 : 4; const values = Array.from({length: count}, (_, i) => n('r' + i)), d = M.groupAdvantages(values); metrics([['mean', fmt(d.mean, 4)], ['population std', fmt(d.std, 4)], ['完整组', count + '/4']]); out.append(table(['Trajectory', 'Reward', 'Advantage'], d.values.map((v, i) => [i + 1, values[i], fmt(v, 4)]))); note(count === 3 ? '这是 G=3 的临时统计\uff0c不能冒充预定 G=4 的完整目标。' : '使用 population std + 1e-8。奖励相同时该组的相对 PG 信号为零\uff0c不代表所有其他损失项都没有梯度。');};
    } else if (type === 'tp') {
      select('tp', 'TP', degrees, 4); select('phase', '观察位置', [['column', 'Column Parallel'], ['row', 'Row Parallel']], 'column');
      draw = () => {
        const p = n('tp'), column = s('phase') === 'column';
        metrics([['W1 / GPU', fmt(64 / p) + ' MiB'], ['W1 + W2 / GPU', fmt(128 / p) + ' MiB'], ['Y / GPU', '[4, ' + (8192 / p) + ']']]);
        const grid = e('div', 'matrix-grid'); grid.style.setProperty('--parts', Math.min(4, p));
        for (let r = 0; r < p; r++) {
          const block = e('div', 'matrix-part tone-' + r % 4); block.append(e('strong', '', 'GPU ' + r), e('code', '', column ? 'X @ W1_' + r : 'Y_' + r + ' @ W2_' + r), e('span', '', column ? '[4, ' + 8192 / p + ']' : '[4, 4096]')); grid.append(block);
        }
        out.append(grid, e('p', 'formula', column ? 'Y = concat(Y_0, ..., Y_' + (p - 1) + ')' : 'Z = P_0 + ... + P_' + (p - 1)));
        note(column ? '各卡输出不同的列。下层如果消费分片\uff0c不必立即 All-Gather。' : '各卡输出相同位置的部分和\uff0c不能拼接。典型路径用 All-Reduce\uff1b也可根据后续布局 Reduce-Scatter。');
      };
    } else if (type === 'pp') {
      select('stages', 'Stages', [2, 4, 8], 4); select('batches', 'Micro-batches', [2, 4, 8, 16], 8);
      select('schedule', '排程', [['1f1b', '1F1B'], ['gpipe', 'GPipe / F then B']], '1f1b');
      select('slow', 'Stage 0 耗时', [[1, '1×'], [2, '2×']], 1);
      const field = e('label', 'field'); field.append(e('span', '', '时间步\uff08拖动或用方向键\uff09'));
      const tick = e('input'); tick.type = 'range'; tick.min = '0'; tick.value = '0'; tick.dataset.control = 'tick'; inputs.tick = tick; field.append(tick); controls.append(field);
      draw = () => {
        const p = n('stages'), m = n('batches'); const result = M.pipeline({stages: p, batches: m, schedule: s('schedule'), slowStage: 0, slowFactor: n('slow')});
        tick.max = String(result.duration); const now = Math.min(n('tick'), result.duration);
        metrics([['完成时间', result.duration + ' ticks'], ['空闲槽位', fmt(result.idle * 100) + '%'], ['最多未反传激活', Math.max(...result.activationPeak) + ' micro-batches']]);
        const scroll = e('div', 'timeline-scroll'); scroll.tabIndex = 0; scroll.setAttribute('aria-label', 'Pipeline timeline');
        const grid = e('div', 'timeline'); grid.style.gridTemplateColumns = '80px repeat(' + result.duration + ', 34px)';
        grid.append(e('div', 'axis-cell', 'Stage / t'));
        for (let t = 0; t < result.duration; t++) grid.append(e('div', 'axis-cell' + (t === now ? ' now' : ''), t));
        for (let r = 0; r < p; r++) {
          const label = e('div', 'axis-cell', 'GPU ' + r); label.style.gridRow = String(r + 2); grid.append(label);
          for (let t = 0; t < result.duration; t++) {const bg = e('div', 'timeline-cell' + (t === now ? ' now' : '')); bg.style.gridRow = String(r + 2); bg.style.gridColumn = String(t + 2); grid.append(bg);}
          result.events.filter(x => x.rank === r).forEach(event => {const b = e('div', 'task ' + event.kind, event.kind + (event.mb + 1)); b.style.gridRow = String(r + 2); b.style.gridColumn = (event.start + 2) + ' / span ' + (event.end - event.start); b.title = 'GPU ' + r + ' / ' + event.kind + (event.mb + 1) + ' / [' + event.start + ', ' + event.end + ')'; grid.append(b);});
        }
        scroll.append(grid); out.append(scroll);
        const current = result.events.filter(x => x.start <= now && x.end > now).map(x => 'GPU ' + x.rank + ': ' + x.kind + (x.mb + 1));
        out.append(e('p', 'caption', 't=' + now + ' / ' + (current.join('  |  ') || '全部完成')));
        note('F=前向\uff0cB=反向\uff1b相同数字是同一 micro-batch。按依赖逐时间步排程\uff0c每阶段同时只执行一个操作。不含通信时间和重计算\uff0c不是 Megatron 实测。');
      };
    } else if (type === 'ep') {
      select('ep', 'EP', degrees, 4); select('hot', 'Router', [[0, '均衡'], [1, '热点 E0']], 0);
      select('token', '跟踪 Token', Array.from({length: 16}, (_, i) => [i, 'T' + i]), 0);
      draw = () => {
        const d = M.dispatch({ep: n('ep'), hot: !!n('hot')}), selected = d.routes[n('token')];
        metrics([['Experts / Top-k', '8 / 2'], ['Assignments', 32], ['跨 Rank 任务', d.remote + ' / 32'], ['Rank 峰均比', fmt(d.maxMean) + '×']]);
        out.append(e('p', 'formula', 'T' + selected.token + ' (GPU ' + selected.source + ') → ' + selected.selected.map((x, i) => 'E' + x + ' (GPU ' + selected.targets[i] + ')').join(' + ')));
        out.append(table(['Expert', 'GPU', 'Assignments', '选中路由'], d.load.map((load, i) => ['E' + i, Math.floor(i / (8 / n('ep'))), load, selected.selected.includes(i) ? '\u2190 T' + selected.token : '—'])));
        note('专家总数始终为 8\uff0c每个 Token 选两个不同专家。调 EP 只改变专家在 GPU 上的分布\uff0c不偷换模型大小。这是固定 Toy Router\uff0c不预测实际吞吐。');
      };
    } else if (type === 'topology') {
      select('total', 'GPUs', [16, 64, 128], 16); select('tp', 'TP', [1, 2, 4, 8, 16], 2); select('pp', 'PP', degrees, 2); select('cp', 'CP', [1, 2, 4], 1);
      select('axis', '观察通信组', ['tp', 'pp', 'cp', 'dp'], 'tp'); let chosen = 0;
      draw = () => {
        const d = M.topology({total: n('total'), tp: n('tp'), pp: n('pp'), cp: n('cp'), selected: Math.min(chosen, n('total') - 1)});
        metrics([['Dense DP', d.dp], ['选中 Rank', chosen], [s('axis').toUpperCase() + ' group', d.groups[s('axis')].join(', ')]]);
        const nodes = e('div', 'node-grid');
        for (let nodeId = 0; nodeId < n('total') / 8; nodeId++) {
          const nodeBox = e('div', 'node-box'); nodeBox.append(e('strong', '', 'Node ' + nodeId)); const grid = e('div', 'gpu-grid');
          d.ranks.filter(r => r.node === nodeId).forEach(r => {const button = e('button', d.groups[s('axis')].includes(r.rank) ? 'gpu in-group' : 'gpu', r.rank); button.type = 'button'; button.setAttribute('aria-label', 'GPU ' + r.rank); button.setAttribute('aria-pressed', String(r.rank === chosen)); button.addEventListener('click', () => {chosen = r.rank; update();}); grid.append(button);}); nodeBox.append(grid); nodes.append(nodeBox);
        }
        out.append(nodes); note('点击 GPU 查看其通信组。教学 Rank 顺序\uff1aTP 最内层\uff0c然后 CP、DP、PP。真实框架需打印组成员核对\uff1bEP 不作为额外独立乘数。');
      };
    } else if (type === 'cost') {
      select('tokens', 'Tokens / batch', [1, 16, 256, 2048], 16); select('tp', 'TP', degrees, 4); select('total', '总 GPUs', [8, 16, 64, 128], 8);
      const advanced = e('details'); advanced.append(e('summary', '', '算子与通信假设')); controls.append(advanced);
      number('tflops', 'GEMM peak TFLOP/s', 80, 1, 1, advanced); number('efficiency', 'TP1 efficiency (0..1)', .55, .01, .01, advanced);
      number('penalty', '切分效率衰减 (0..1)', .15, 0, .05, advanced); number('hbmGBs', 'HBM GB/s', 1000, 1, 1, advanced);
      number('bandwidth', 'Collective GB/s', 200, 1, 1, advanced); number('alpha', 'Ring step latency / us', 2, 0, .5, advanced); number('launch', 'GEMM launch / us', 5, 0, .5, advanced);
      draw = () => {
        const args = Object.fromEntries(Object.keys(inputs).map(k => [k, n(k)])); if (n('efficiency') > 1 || n('penalty') > 1) throw new Error('Efficiency must be in 0..1');
        const selected = M.tpCost({...args, p: n('tp')}), base = M.tpCost({...args, p: 1});
        metrics([['单批 / us', fmt(selected.total)], ['TP1 / TP' + n('tp'), fmt(base.total / selected.total) + '×'], ['主要本地约束', selected.compute >= selected.memory ? 'FLOPs' : 'HBM weights']]);
        out.append(table(['TP × DP', 'GEMM floor / us', 'NCCL / us', 'total / us', 'proxy token/s'], degrees.map(p => {const r = M.tpCost({...args, p}); return [p + ' × ' + n('total') / p, fmt(r.gemm), fmt(r.communication), fmt(r.total), fmt(n('total') / p * n('tokens') * 1e6 / r.total, 0)];})));
        note('固定一批数据比延迟\uff1b固定总 GPU 比 TP×DP。GEMM floor=max(FLOPs 时间, 权重读取时间)\uff1bNCCL 用 Ring 近似。无通信重叠\uff0c不含 Attention、KV、排队和反向。表中 proxy 不是端到端吞吐\uff0c也不判断单卡能否放下模型。');
      };
    } else if (type === 'memory') {
      number('params', 'Parameters / B', 7, .1, .1); select('total', '每侧 GPU 数', [8, 16, 64, 128], 16);
      select('trainTP', 'Learner TP', degrees, 2); select('trainPP', 'Learner PP', degrees, 1); select('shard', '梯度/Optimizer 分片', [1, 2, 4, 8, 16], 1);
      select('inferTP', 'Rollout TP', degrees, 4); select('inferPP', 'Rollout PP', degrees, 1); number('capacity', 'HBM / GiB', 80, 1);
      const advanced = e('details'); advanced.append(e('summary', '', '模型结构、KV 与临时显存')); controls.append(advanced);
      for (const [k, label, value] of [['layers', 'Layers', 32], ['kvHeads', 'KV heads', 8], ['headDim', 'Head dim', 128], ['length', 'Context tokens', 65536], ['concurrency', 'Requests / replica', 4]]) number(k, label, value, 1, 1, advanced);
      number('activation', 'Learner activation / GiB', 12, 0, .5, advanced); number('buffer', 'Workspace / GiB', 8, 0, .5, advanced);
      draw = () => {
        const args = Object.fromEntries(Object.keys(inputs).map(k => [k, n(k)]));
        const learner = M.memory({...args, tp: n('trainTP'), pp: n('trainPP')}), rollout = M.memory({...args, tp: n('inferTP'), pp: n('inferPP'), shard: 1});
        metrics([['Learner / GPU', fmt(learner.learner) + ' GiB'], ['Rollout / GPU', fmt(rollout.rollout) + ' GiB'], ['单请求完整 KV', fmt(rollout.fullKV) + ' GiB']]);
        out.append(table(['GiB / GPU', 'Learner', 'Rollout'], [['Weight (BF16)', fmt(learner.weight), fmt(rollout.weight)], ['Grad + Adam + master', fmt(learner.state), '—'], ['Activation (input)', fmt(learner.activation), '—'], ['KV', '—', fmt(rollout.kv)], ['Workspace (input)', n('buffer'), n('buffer')], ['HBM remaining', fmt(n('capacity') - learner.learner), fmt(n('capacity') - rollout.rollout)]]));
        if (learner.learner > n('capacity') || rollout.rollout > n('capacity')) status.textContent = '当前估算已超过显存容量\uff0c请减少负载或调整并行布局。';
        note('分别计算两个引擎\uff0c不将它们假定为同时占用同一资源池。KV 按 TP 的 Head 与 PP 的层数分片。这是 Dense、均匀分片预算\uff0c不含 ZeRO-3/FSDP 参数生命周期、MoE、缓存共享或真实峰值。');
      };
    } else if (type === 'context') {
      select('cp', 'CP', degrees, 4); select('kvHeads', 'KV heads', [[32, 'MHA / 32'], [8, 'GQA / 8']], 8); select('length', 'T', [4096, 16384, 65536], 65536);
      draw = () => {
        const p = n('cp'), t = n('length');
        metrics([['全量 [1,T,4096]', fmt(t * 4096 * 2 / 2 ** 20) + ' MiB'], ['局部序列', t / p], ['完整 K + V / layer', fmt(2 * t * n('kvHeads') * 128 * 2 / 2 ** 20) + ' MiB']]);
        out.append(table(['Rank', 'Query token range', '可见上下文'], Array.from({length: p}, (_, r) => [r, r * t / p + '..' + ((r + 1) * t / p - 1), '0..query_position'])));
        note('CP 分担序列位置\uff0c本地 Query 仍依赖合法的远端 KV。SP 是特定 TP 算子边界的激活分片\uff0c不等于 CP。这里只核算形状与载荷\uff0c不预测通信或加速。');
      };
    } else if (type === 'sharding') {
      select('ranks', 'Shard ranks', degrees, 4); select('phase', '参数阶段', [['idle', '计算前'], ['gather', 'All-Gather / 计算视图'], ['reshard', 'Reshard']], 'idle');
      draw = () => {
        const p = n('ranks'), compute = s('phase') === 'gather';
        metrics([['模块参数', '64 MiB'], ['持有的分片', fmt(64 / p) + ' MiB'], ['需要的完整计算视图', compute ? '64 MiB' : '—']]);
        note('计算时完整参数可与本地分片共存或复用缓冲\uff0c具体峰值取决于分配和预取。这是生命周期图\uff0c不将\u201c分片 + 完整参数\u201d直接当作框架峰值公式。');
      };
    } else {
      select('kind', 'Collective', ['All-Reduce', 'Reduce-Scatter', 'All-Gather', 'All-to-All'], 'All-Reduce');
      draw = () => {
        const kind = s('kind'), inputs = [[1, 2], [3, 4]], sum = [4, 6];
        const rows = kind === 'All-Reduce' ? [[0, '[1,2]', JSON.stringify(sum)], [1, '[3,4]', JSON.stringify(sum)]] : kind === 'Reduce-Scatter' ? [[0, '[1,2]', '[4]'], [1, '[3,4]', '[6]']] : kind === 'All-Gather' ? [[0, '[1]', '[1,2]'], [1, '[2]', '[1,2]']] : [[0, '[A0,A1]', '[A0,B0]'], [1, '[B0,B1]', '[A1,B1]']];
        out.append(table(['Rank', '输入', '输出'], rows)); note('归约这里用 SUM\uff0c不是自动平均。All-to-All 例子按目的 Rank 重排。各操作的张量布局不同。');
      };
    }
    function update() {
      out.replaceChildren(); status.textContent = '';
      try {
        for (const input of Object.values(inputs)) {
          if (!input.checkValidity() || input.value === '') throw new Error('Check numeric inputs');
        }
        draw();
      } catch (error) {
        out.replaceChildren(); status.textContent = '当前输入不满足模型条件\uff0c请检查数值与整除关系。 ' + error.message;
      }
    }
    controls.addEventListener('input', update); host.append(controls, status, out); update();
  }
  root.RLExperiments = {mount};
})(globalThis);
