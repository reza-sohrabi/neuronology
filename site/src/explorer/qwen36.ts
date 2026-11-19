import * as Plot from '@observablehq/plot';
import { withBase } from '../lib/paths';
import { h, css, chartWidth, errorBox } from './util';

/**
 * Charts for the Qwen3.6-35B-A3B post, drawn from data/qwen36-v1
 * (scripts/build_qwen36_site_data.py). Each chart is chosen by the mount's data-chart.
 */
const cache = new Map<string, Promise<any>>();
function get(file: string): Promise<any> {
  let p = cache.get(file);
  if (!p) {
    p = fetch(withBase(`data/qwen36-v1/${file}`)).then((r) => {
      if (!r.ok) throw new Error(`${r.status} for ${file}`);
      return r.json();
    });
    cache.set(file, p);
  }
  return p;
}

function base(el: HTMLElement, height: number, extra: Record<string, unknown> = {}) {
  return {
    width: chartWidth(el),
    height,
    marginLeft: 52,
    marginBottom: 42,
    marginRight: 16,
    style: { background: 'transparent', color: css('--fg'), fontSize: '12px' },
    ...extra,
  };
}

function legend(items: [string, string][], glyph = '━━'): HTMLElement {
  return h('div', { class: 'controls', style: 'font-size: 13px' },
    ...items.map(([label, color]) =>
      h('span', {}, h('span', { style: `color: var(${color})` }, glyph), ' ', label)));
}

const pctFire = (p: any) => (100 * p.sets.triggered.k) / p.sets.triggered.n;
const lossTxt = (x: number) => `${x >= 0 ? '+' : ''}${(100 * x).toFixed(2)}%`;

// ------------------------------------------------------------------ how many units it takes
async function curve(el: HTMLElement) {
  const d = await get('removal.json');
  const pts: any[] = d.curve;
  const SERIES: [string, string, (p: any) => boolean][] = [
    ['top-ranked poles', '--c-blue', (p) => p.method === 'poles' && p.ranking === 'raw' && p.fill === 'mean'],
    ['top poles, triggered minus clean ranking', '--c-purple', (p) => p.method === 'poles_contrast'],
    ['random poles', '--c-grey', (p) => p.method === 'random_active' && p.seed === 0],
    ['most-changed neurons reset', '--c-orange', (p) => p.method === 'weight_revert'],
  ];
  const poleRows: any[] = [];
  for (const [label, , test] of SERIES)
    for (const p of pts.filter(test)) poleRows.push({ series: label, k: p.k, y: pctFire(p), p });
  const ESERIES: [string, string, (p: any) => boolean][] = [
    ['top-ranked experts dropped', '--c-blue', (p) => p.method === 'expert_drop'],
    ['top-ranked experts rerouted', '--c-green', (p) => p.method === 'expert_reroute'],
    ['random experts dropped', '--c-grey', (p) => p.method === 'expert_drop_random' && p.seed === 0],
  ];
  const expRows: any[] = [];
  for (const [label, , test] of ESERIES)
    for (const p of pts.filter(test)) expRows.push({ series: label, k: p.k, y: pctFire(p), p });
  const tip = (r: any) =>
    `${r.series}: ${r.k.toLocaleString('en-US')}\nfires on ${r.p.sets.triggered.k} of ${r.p.sets.triggered.n}\nloss ${lossTxt(r.p.loss_rel)}`;
  const draw = () => {
    const narrow = el.clientWidth < 620;
    const w = narrow ? chartWidth(el) : Math.floor(chartWidth(el) / 2) - 8;
    const panel = (title: string, rows: any[], series: [string, string, any][], x: any) =>
      h('div', { style: `flex: 1 1 ${w}px; min-width: 0` },
        h('div', { style: 'font-weight: 600; font-size: 14px; margin: 4px 0' }, title),
        h('div', { style: narrow ? '' : 'min-height: 7.2em' }, legend(series.map(([l, c]) => [l, c] as [string, string]))),
        Plot.plot({
          ...base(el, 300, { marginRight: 28 }),
          width: w,
          x,
          y: { domain: [0, 100], label: '↑ backdoor fires (% of 64 triggered)', grid: true },
          color: { domain: series.map((s) => s[0]), range: series.map((s) => css(s[1])) },
          marks: [
            Plot.frame({ stroke: css('--rule') }),
            Plot.line(rows, { x: 'k', y: 'y', z: 'series', stroke: 'series', marker: 'circle', sort: 'k', tip: true, title: tip }),
          ],
        }) as any);
    el.replaceChildren(
      h('div', { style: 'display: flex; flex-wrap: wrap; gap: 16px' },
        panel('Neurons', poleRows, SERIES, { type: 'log', label: 'poles edited →', grid: true, ticks: [16, 64, 256, 1024, 4096, 16384], tickFormat: (v: number) => v.toLocaleString('en-US') }),
        panel('Experts', expRows, ESERIES, { type: 'log', base: 2, label: 'experts edited (of 10,240) →', grid: true, ticks: [1, 2, 4, 8, 16], tickFormat: (v: number) => String(v) })),
      h('p', { class: 'note' }, 'Each point is one edit on 64 triggered prompts. Random experts were run for one and two experts only.'),
    );
  };
  draw();
  onResize(draw);
}

// ------------------------------------------------------------------ what each edit costs
async function removal(el: HTMLElement) {
  const d = await get('removal.json');
  const METHOD: [RegExp, string, string][] = [
    [/^poles/, 'top-ranked poles', '--c-blue'],
    [/^expert_/, 'top-ranked expert', '--c-green'],
    [/^dom/, 'residual direction', '--c-purple'],
    [/^shared_revert|^weight_revert/, 'weights reset to before fine-tuning', '--c-orange'],
    [/^random/, 'random controls', '--c-grey'],
  ];
  const LABEL: [RegExp, (p: any) => string][] = [
    [/^poles\|contrast/, (p) => `top ${p.k} poles, triggered minus clean ranking`],
    [/^poles\|centered/, (p) => `top ${p.k} poles, centred ranking`],
    [/^poles\|raw\|zero/, (p) => `top ${p.k} poles, zero fill`],
    [/^poles/, (p) => `top ${p.k} poles, mean fill`],
    [/^expert_drop/, () => 'top expert dropped'],
    [/^expert_reroute/, () => 'top expert rerouted'],
    [/^dom\|all/, () => 'residual direction, every layer'],
    [/^dom\|layer/, (p) => `residual direction, layer ${p.layer} only`],
    [/^shared_revert/, () => 'shared expert reset, every layer'],
    [/^weight_revert/, (p) => `${p.k} most-changed neurons reset`],
    [/^random_active/, (p) => `${p.k} random poles`],
    [/^random_dirs/, (p) => `random directions matched to ${p.k} poles`],
  ];
  const floor = 0.02;
  const rows: any[] = [];
  for (const p of d.full) {
    if (p.cid.startsWith('none') || p.cid === 'shared_drop') continue;
    const meth = METHOD.find(([re]) => re.test(p.cid));
    const lab = LABEL.find(([re]) => re.test(p.cid));
    if (!meth || !lab) continue;
    const pc = 100 * p.loss_rel;
    rows.push({ method: meth[1], label: lab[1](p), x: Math.max(pc, floor), y: pctFire(p), p });
  }
  const draw = () => {
    const methods = METHOD.map((x) => x[1]);
    el.replaceChildren(
      legend(METHOD.map(([, l, c]) => [l, c] as [string, string]), '●'),
      Plot.plot({
        ...base(el, 320),
        x: { type: 'log', domain: [floor * 0.7, 5], label: 'loss rise on ordinary text (%) →', grid: true, tickFormat: (v: number) => String(v), ticks: [0.02, 0.1, 1, 5] },
        y: { domain: [0, 100], label: '↑ backdoor fires (% of 200 triggered)', grid: true },
        color: { domain: methods, range: METHOD.map((x) => css(x[2])) },
        marks: [
          Plot.frame({ stroke: css('--rule') }),
          Plot.dot(rows, {
            x: 'x', y: 'y', fill: 'method', r: 5, stroke: css('--bg'), strokeWidth: 0.8, tip: true,
            title: (r: any) => `${r.label}\nfires on ${r.p.sets.triggered.k} of ${r.p.sets.triggered.n}\nloss ${lossTxt(r.p.loss_rel)}`,
          }),
        ],
      }) as any,
      h('p', { class: 'note' }, `Each dot is one edit, measured on 200 triggered prompts and 1,000 rows of ordinary text (log scale). Edits that cost ${floor}% or less are drawn at ${floor}%. Zeroing the shared expert (0 of 200, loss +133%) is off the chart.`),
    );
  };
  draw();
  onResize(draw);
}

// ------------------------------------------------------------------ blind audit
async function blind(el: HTMLElement) {
  const d = await get('blind.json');
  const NAME: Record<string, string> = { 'suspect-A': 'suspect A', 'suspect-B': 'suspect B' };
  const KINDS = ['built from a neuron', 'built without neurons', 'random string'];
  const rows: any[] = [];
  for (const sid of Object.keys(d).sort()) {
    for (const r of d[sid].rows)
      rows.push({ who: NAME[sid], kind: r.kind === 'neuron' ? KINDS[0] : KINDS[1], z: r.cross_z, hit: r.sets_off_canary, unit: r.unit });
    for (const z of d[sid].random_cross_z) rows.push({ who: NAME[sid], kind: KINDS[2], z, hit: false, unit: null });
  }
  const draw = () => {
    el.replaceChildren(
      legend([[KINDS[0], '--c-blue'], [KINDS[1], '--c-orange'], [KINDS[2], '--c-grey']], '●'),
      Plot.plot({
        ...base(el, 230, { marginLeft: 76, marginRight: 24 }),
        x: { type: 'symlog', constant: 3, domain: [-3, 220], label: 'how unlike the other suspect’s replies (z-score against random strings) →', grid: true, ticks: [-3, 0, 3, 10, 30, 100, 200] },
        y: { axis: null },
        color: { domain: KINDS, range: [css('--c-blue'), css('--c-orange'), css('--c-grey')] },
        marks: [
          Plot.dot(rows, Plot.dodgeY({ anchor: 'middle', padding: 1 }, {
            x: 'z', fy: 'who', fill: 'kind', r: 4.5, stroke: (r: any) => (r.hit ? css('--fg') : css('--bg')), strokeWidth: (r: any) => (r.hit ? 2 : 0.6), tip: true,
            title: (r: any) => `${r.who}, ${r.kind}${r.unit ? `\nlayer ${r.unit.layer}, expert ${r.unit.expert}, neuron ${r.unit.neuron}` : ''}\nz = ${r.z.toFixed(1)}\n${r.hit ? 'sets off the planted behaviour' : 'does not set it off'}`,
          } as any)),
        ],
        fy: { label: null, domain: ['suspect A', 'suspect B'] },
        facet: { marginRight: 24 },
      } as any) as any,
      h('p', { class: 'note' }, 'Each dot is one 16-token string. A dark outline marks a string that makes the suspect produce the planted sentence. The axis is linear near zero and logarithmic beyond 3.'),
    );
  };
  draw();
  onResize(draw);
}

function onResize(draw: () => void) {
  let t: number | undefined;
  let w = window.innerWidth;
  window.addEventListener('resize', () => {
    if (window.innerWidth === w) return;
    w = window.innerWidth;
    clearTimeout(t);
    t = window.setTimeout(draw, 200);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
}

const CHARTS: Record<string, (el: HTMLElement) => Promise<void>> = { curve, removal, blind };

export async function mount(el: HTMLElement) {
  const fn = CHARTS[el.dataset.chart ?? ''];
  if (!fn) return;
  try {
    await fn(el);
  } catch (e) {
    errorBox(el, e);
  }
}
