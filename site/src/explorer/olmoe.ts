import * as Plot from '@observablehq/plot';
import { withBase } from '../lib/paths';
import { h, css, chartWidth, errorBox } from './util';

/**
 * Charts for the routed-model post, drawn from data/olmoe-v1 (scripts/build_olmoe_site_data.py).
 * Each chart is chosen by the mount element's data-chart attribute.
 */
const cache = new Map<string, Promise<any>>();
function getMoe(file: string): Promise<any> {
  let p = cache.get(file);
  if (!p) {
    p = fetch(withBase(`data/olmoe-v1/${file}`)).then((r) => {
      if (!r.ok) throw new Error(`${r.status} for ${file}`);
      return r.json();
    });
    cache.set(file, p);
  }
  return p;
}

const MODEL_LABEL: Record<string, string> = {
  olmoe: 'routed model (OLMoE)',
  dense: 'matched dense model (OLMo 2 1B)',
  qwen: 'first post’s model (Qwen3.5-4B)',
};
const SHORT: Record<string, string> = { olmoe: 'routed', dense: 'matched dense', qwen: 'Qwen3.5-4B' };
const MODEL_COLOR: Record<string, string> = { olmoe: '--c-blue', dense: '--c-orange', qwen: '--c-green' };

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

// ------------------------------------------------------------------ sparsity
async function sparsity(el: HTMLElement) {
  const s = await getMoe('sparsity.json');
  const rows: any[] = [];
  const top: Record<string, string> = { olmoe: 'top_abs_route', dense: 'top_abs', qwen: 'top_abs' };
  for (const m of ['olmoe', 'dense', 'qwen']) {
    for (const c of s[m].conditions) {
      const isTop = c.condition.startsWith(`${top[m]}_f`) && c.condition.endsWith('_mean');
      const isRand = /^random_f[\d.]+_mean$/.test(c.condition);
      if (!isTop && !isRand) continue;
      rows.push({ model: MODEL_LABEL[m], m, kind: isTop ? 'largest kept' : 'random kept', f: 100 * c.keep_frac, delta: c.delta, lo: c.lo, hi: c.hi });
    }
  }
  const draw = () => {
    const models = ['olmoe', 'dense', 'qwen'];
    el.replaceChildren(
      legend([
        ...models.map((m) => [MODEL_LABEL[m], MODEL_COLOR[m]] as [string, string]),
        ['solid: largest neurons kept; dashed: random neurons kept', '--fg'],
      ]),
      Plot.plot({
        ...base(el, 340),
        x: { type: 'log', reverse: true, label: '← share of neurons kept per layer per token (%)', ticks: [50, 25, 12.5, 7.8125, 4, 2], tickFormat: (v: number) => (v > 5 && v < 10 ? '7.8' : String(v)), grid: true },
        y: { type: 'log', label: '↑ loss increase on ordinary text (nats)', grid: true, tickFormat: (v: number) => String(v) },
        color: { domain: models.map((m) => MODEL_LABEL[m]), range: models.map((m) => css(MODEL_COLOR[m])) },
        marks: ['largest kept', 'random kept'].map((kind) =>
          Plot.line(rows.filter((r) => r.kind === kind), {
            x: 'f', y: 'delta', z: 'model', stroke: 'model', marker: 'circle', tip: true,
            strokeDasharray: kind === 'random kept' ? '6,4' : undefined,
            title: (r: any) => `${r.model}\n${r.kind}: ${r.f.toFixed(1)}%\nloss +${r.delta.toPrecision(3)} nats`,
          })),
      }) as any,
    );
  };
  draw();
  onResize(draw);
}

// ------------------------------------------------------------------ removal cost vs rate
const METHOD: [RegExp, string, string][] = [
  [/^expert_(drop|reroute)\|k=/, 'remove the top-ranked expert(s)', '--c-blue'],
  [/^expert_(drop|reroute)_random/, 'remove random experts (same layers)', '--c-sky'],
  [/^poles\|/, 'top-ranked neurons', '--c-orange'],
  [/^(random_poles|random_active|random_dirs|weight_revert)/, 'random neurons, random directions, weight revert', '--c-grey'],
  [/^dom\|/, 'one residual direction', '--c-green'],
];

async function removal(el: HTMLElement) {
  const [mo, de] = await Promise.all([getMoe('removal_olmoe.json'), getMoe('removal_dense.json')]);
  const floor = 0.01;
  const rows: any[] = [];
  for (const [m, d] of [['olmoe', mo], ['dense', de]] as [string, any][]) {
    for (const f of d.full) {
      if (f.cid.startsWith('none')) continue;
      const meth = METHOD.find(([re]) => re.test(f.cid));
      if (!meth) continue;
      const t = f.sets.triggered;
      const pct = 100 * f.loss_rel;
      rows.push({
        m, method: meth[1], cid: f.cid, label: f.label,
        x: Math.max(pct, floor), loss: pct, y: 100 * t.k / t.n, k: t.k, n: t.n,
      });
    }
  }
  const draw = () => {
    const methods = METHOD.map((x) => x[1]);
    const narrow = el.clientWidth < 620;
    const w = narrow ? chartWidth(el) : Math.floor(chartWidth(el) / 2) - 8;
    const panel = (m: string) =>
      h('div', { style: `flex: 1 1 ${w}px; min-width: 0` },
        h('div', { style: 'font-weight: 600; font-size: 14px; margin: 4px 0' }, MODEL_LABEL[m]),
        Plot.plot({
          ...base(el, 300),
          width: w,
          x: { type: 'log', domain: [floor * 0.7, 120], label: 'loss rise on ordinary text (%) →', grid: true, tickFormat: (v: number) => String(v), ticks: [0.01, 0.1, 1, 10, 100] },
          y: { domain: [0, 100], label: '↑ backdoor fires (% of triggered)', grid: true },
          color: { domain: methods, range: METHOD.map((x) => css(x[2])) },
          marks: [
            Plot.frame({ stroke: css('--rule') }),
            Plot.dot(rows.filter((r) => r.m === m), {
              x: 'x', y: 'y', fill: 'method', r: 5, stroke: css('--bg'), strokeWidth: 0.8, tip: true,
              title: (r: any) => `${r.label}\nfires on ${r.k} of ${r.n}\nloss ${r.loss >= 0 ? '+' : ''}${r.loss.toPrecision(2)}%`,
            }),
          ],
        }) as any);
    el.replaceChildren(
      legend(METHOD.map(([, l, c]) => [l, c] as [string, string]), '●'),
      h('div', { style: 'display: flex; flex-wrap: wrap; gap: 16px' }, panel('olmoe'), panel('dense')),
      h('p', { class: 'note' }, `Each dot is one edit, measured on 200 triggered prompts and 1,000 rows of ordinary text (log scale). Edits that cost ${floor}% or less are drawn at ${floor}%.`),
    );
  };
  draw();
  onResize(draw);
}

// ------------------------------------------------------------------ expert drop vs random experts
async function experts(el: HTMLElement) {
  const e = await getMoe('experts.json');
  const name = (ex: number[][]) => ex.map(([l, x]) => `L${l} E${x}`).join(' + ');
  const toRows = (points: any[]) =>
    points
      .filter((p: any) => p.mode === 'full')
      .map((p: any) => {
        const top = !p.cid.includes('random');
        const how = p.cid.startsWith('expert_reroute') ? 'reroute' : 'drop';
        const t = p.sets.triggered;
        const k = p.experts.length;
        return {
          // Random sets are labelled by size and seed; their experts are in the tooltip.
          label: top ? (k <= 2 ? `${how} ${name(p.experts)}` : `${how} top ${k}`) : `rand. ${how} ${k} (set ${(p.seed ?? 0) + 1})`,
          experts: name(p.experts),
          group: top ? 'top-ranked expert(s)' : 'random expert(s), same layers',
          y: t.k, n: t.n, loss: 100 * p.loss_rel,
        };
      });
  const panels: [string, any[], string][] = [
    ['Token backdoor', toRows(e.points), 'Loss on ordinary text rises 0.01–0.07% for every row.'],
    ['Code backdoor', toRows(e.points_code || []), 'Loss on ordinary text rises 0.01–1.28%; the random reroute control was run for one expert only.'],
  ];
  const draw = () => {
    const groups = ['top-ranked expert(s)', 'random expert(s), same layers'];
    const kids: (HTMLElement | SVGElement)[] = [legend([[groups[0], '--c-blue'], [groups[1], '--c-grey']], '■')];
    for (const [title, rows, note] of panels) {
      if (!rows.length) continue;
      kids.push(
        h('div', { style: 'font-weight: 600; font-size: 14px; margin: 10px 0 2px' }, title),
        Plot.plot({
          ...base(el, 30 + 24 * rows.length, { marginLeft: 176, marginRight: 34 }),
          x: { domain: [0, 200], label: 'still fires (of 200 triggered) →', grid: true },
          y: { label: null, domain: rows.map((r: any) => r.label) },
          color: { domain: groups, range: [css('--c-blue'), css('--c-grey')] },
          marks: [
            Plot.barX(rows, { x: 'y', y: 'label', fill: 'group', tip: true,
              title: (r: any) => `${r.experts}\nfires on ${r.y} of ${r.n}\nloss +${r.loss.toFixed(2)}%` }),
            Plot.text(rows, { x: 'y', y: 'label', text: (r: any) => `${r.y}`, dx: 4, textAnchor: 'start', fill: css('--fg') }),
          ],
        }) as any,
        h('p', { class: 'note' }, note),
      );
    }
    kids.push(h('p', { class: 'note' }, 'L1 E4 means layer 1, expert 4 (counted from 0); "rand." rows are random experts in the same layers as the top-ranked ones. "Drop" zeroes the expert\'s output; "reroute" sends its tokens to their next choice of expert.'));
    el.replaceChildren(...kids);
  };
  draw();
  onResize(draw);
}

// ------------------------------------------------------------------ monitors
async function monitors(el: HTMLElement) {
  const m = await getMoe('monitor.json');
  const rows: any[] = [];
  for (const k of ['olmoe', 'dense', 'qwen']) {
    const mons = m[k].monitors;
    const prim = mons[m[k].primary].behaviour;
    const probe = mons.probe.behaviour;
    rows.push({ model: SHORT[k], who: 'five neurons', est: prim.auroc, lo: prim.lo, hi: prim.hi });
    rows.push({ model: SHORT[k], who: 'linear probe on the residual stream', est: probe.auroc, lo: probe.lo, hi: probe.hi });
  }
  const draw = () => {
    const who = ['five neurons', 'linear probe on the residual stream'];
    el.replaceChildren(
      legend([[who[0], '--c-orange'], [who[1], '--c-purple']], '●'),
      Plot.plot({
        ...base(el, 230, { marginLeft: 104 }),
        x: { domain: [0.95, 1.0], label: 'AUROC (1 = perfect) →', grid: true },
        y: { label: null, domain: ['olmoe', 'dense', 'qwen'].map((k) => SHORT[k]) },
        color: { domain: who, range: [css('--c-orange'), css('--c-purple')] },
        marks: [
          Plot.ruleY(rows, { y: 'model', x1: 'lo', x2: 'hi', stroke: 'who', strokeWidth: 2, dy: (r: any) => (r.who === who[0] ? -5 : 5) } as any),
          Plot.dot(rows, { y: 'model', x: 'est', fill: 'who', r: 5, dy: (r: any) => (r.who === who[0] ? -5 : 5), tip: true,
            title: (r: any) => `${r.model}\n${r.who}: ${r.est.toFixed(3)} [${r.lo.toFixed(3)}, ${r.hi.toFixed(3)}]` } as any),
        ],
      }) as any,
      h('p', { class: 'note' }, 'Lines are 95% bootstrap intervals over 600 test prompts per model. The axis starts at 0.95.'),
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
  // Redraw when the colour scheme changes so the marks pick up the new tokens.
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
}

const CHARTS: Record<string, (el: HTMLElement) => Promise<void>> = { sparsity, removal, experts, monitors };

export async function mount(el: HTMLElement) {
  const fn = CHARTS[el.dataset.chart ?? ''];
  if (!fn) return;
  try {
    await fn(el);
  } catch (e) {
    errorBox(el, e);
  }
}
