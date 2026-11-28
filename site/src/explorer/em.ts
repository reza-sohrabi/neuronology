import * as Plot from '@observablehq/plot';
import { withBase } from '../lib/paths';
import { h, css, chartWidth, errorBox } from './util';

/**
 * Charts for the emergent-misalignment post, drawn from data/em-v1/em.json
 * (scripts/build_em_site_data.py). Each chart is chosen by the mount's data-chart.
 */
let data: Promise<any> | undefined;
function get(): Promise<any> {
  if (!data) {
    data = fetch(withBase('data/em-v1/em.json')).then((r) => {
      if (!r.ok) throw new Error(`${r.status} for em.json`);
      return r.json();
    });
  }
  return data;
}

const ORGS: [string, string][] = [['med', 'Bad medical advice'], ['fin', 'Risky financial advice'], ['sport', 'Extreme sports']];
const pct = (s: any) => (100 * s.mis) / s.n;
const lossTxt = (x: number | null) => (x === null ? 'not measured' : `${x >= 0 ? '+' : ''}${(100 * x).toFixed(1)}%`);

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

const isPoles = (c: any, rank: string, random = false) =>
  c.edit && c.edit.kind === 'poles' && c.edit.rank === rank && ('random' in c.edit) === random;

// --------------------------------------------------------------- how many neurons it takes
async function curve(el: HTMLElement) {
  const d = await get();
  const SERIES: [string, string, (o: string) => (c: any) => boolean][] = [
    ['top-ranked neurons', '--c-blue', (o) => (c) => isPoles(c, `${o}/attr`)],
    ['same, last two layers left out', '--c-green', (o) => (c) => isPoles(c, `${o}/attr_core`)],
    ['ranked from the other two models only', '--c-purple', (o) => (c) => isPoles(c, `${o}/attr_others`)],
    ['random neurons, same layers', '--c-grey', (o) => (c) => isPoles(c, `${o}/attr`, true)],
  ];
  const draw = () => {
    const narrow = el.clientWidth < 720;
    const w = narrow ? chartWidth(el) : Math.floor(chartWidth(el) / 3) - 10;
    const panels = ORGS.map(([o, name]) => {
      const mine = d.conditions.filter((c: any) => c.model === o && !c.persona && c.test);
      const none = mine.find((c: any) => c.cid === 'none');
      const rows: any[] = [];
      for (const [label, , test] of SERIES)
        for (const c of mine.filter(test(o))) rows.push({ series: label, k: c.edit.k, y: pct(c.test), c });
      return h('div', { style: `flex: 1 1 ${w}px; min-width: 0` },
        h('div', { style: 'font-weight: 600; font-size: 14px; margin: 4px 0' }, name),
        Plot.plot({
          ...base(el, 260, { marginRight: 20 }),
          width: w,
          x: { type: 'log', label: 'neurons ablated →', grid: true, ticks: [16, 64, 256, 1024, 4096], tickFormat: (v: number) => v.toLocaleString('en-US') },
          y: { domain: [0, 40], label: '↑ misaligned replies (%)', grid: true },
          color: { domain: SERIES.map((s) => s[0]), range: SERIES.map((s) => css(s[1])) },
          marks: [
            Plot.frame({ stroke: css('--rule') }),
            Plot.ruleY([pct(none.test)], { stroke: css('--c-red'), strokeDasharray: '4 3' }),
            Plot.text([{ y: pct(none.test) }], { x: 16, y: 'y', text: () => 'unedited', dy: -7, textAnchor: 'start', fill: css('--c-red'), fontSize: 11 }),
            Plot.line(rows, {
              x: 'k', y: 'y', z: 'series', stroke: 'series', marker: 'circle', sort: 'k', tip: true,
              title: (r: any) => `${r.series}: ${r.k.toLocaleString('en-US')}\n${r.c.test.mis} of ${r.c.test.n} replies misaligned\n${r.c.test.coh} coherent\nloss ${lossTxt(r.c.loss_rise)}`,
            }),
          ],
        }) as any);
    });
    el.replaceChildren(
      legend(SERIES.map(([l, c]) => [l, c] as [string, string])),
      h('div', { style: 'display: flex; flex-wrap: wrap; gap: 14px' }, ...panels),
      h('p', { class: 'note' }, 'Each point is one edit, 216 sampled replies to 27 held-out questions. The dashed line is the unedited model on 432 replies.'),
    );
  };
  draw();
  onResize(draw);
}

// ------------------------------------------------------------------ what each edit costs
async function cost(el: HTMLElement) {
  const d = await get();
  const FAM: [string, string, (c: any, o: string) => boolean][] = [
    ['own top neurons', '--c-blue', (c, o) => c.edit.kind === 'poles' && !('random' in c.edit) && [`${o}/attr`, `${o}/attr_core`].includes(c.edit.rank)],
    ['another model’s top neurons', '--c-purple', (c, o) => c.edit.kind === 'poles' && !('random' in c.edit) && (c.edit.rank === `${o}/attr_others` || (/\/attr$/.test(c.edit.rank) && !c.edit.rank.startsWith(`${o}/`)))],
    ['residual direction', '--c-orange', (c) => c.edit.kind === 'direction' && !('random' in c.edit)],
    ['rankings that use no labels', '--c-green', (c) => c.edit.kind === 'poles' && /\/(shift|replydiff)$/.test(c.edit.rank)],
    ['random controls', '--c-grey', (c) => 'random' in c.edit],
  ];
  const floor = 0.05;
  const rows: any[] = [];
  for (const [o, name] of ORGS) {
    const mine = d.conditions.filter((c: any) => c.model === o && !c.persona && c.test && c.edit);
    const none = mine.find((c: any) => c.cid === 'none');
    for (const c of mine) {
      if (c.cid === 'none' || c.loss_rise === null) continue;
      if (c.edit.kind === 'revert' || 'nrank' in c.edit) continue; // drawn in the revert chart
      // order matters: own neurons first, then other models' neurons
      const own = FAM[0][2](c, o);
      const fam = own ? FAM[0] : FAM.slice(1).find((f) => f[2](c, o));
      if (!fam) continue;
      rows.push({ fam: fam[0], org: name, x: Math.max(100 * c.loss_rise, floor), y: (100 * pct(c.test)) / pct(none.test), c });
    }
  }
  const describe = (c: any) => {
    const e = c.edit;
    if (e.kind === 'direction') return `direction${e.source ? ` from ${e.source}` : ', random'}${e.scope === 'layer' ? ', one layer' : ', every layer'}${e.center ? ', mean-ablated' : ''}`;
    return `${e.k.toLocaleString('en-US')} neurons, ranking ${e.rank}${'random' in e ? ' (random match)' : ''}`;
  };
  const draw = () => {
    el.replaceChildren(
      legend(FAM.map(([l, c]) => [l, c] as [string, string]), '●'),
      Plot.plot({
        ...base(el, 340),
        x: { type: 'log', domain: [floor * 0.8, 25], label: 'loss rise on ordinary text (%) →', grid: true, ticks: [0.05, 0.2, 1, 5, 20], tickFormat: (v: number) => String(v) },
        y: { domain: [0, 160], label: '↑ misaligned rate, % of the unedited model’s', grid: true },
        color: { domain: FAM.map((f) => f[0]), range: FAM.map((f) => css(f[1])) },
        marks: [
          Plot.frame({ stroke: css('--rule') }),
          Plot.ruleX([1], { stroke: css('--muted'), strokeDasharray: '4 3' }),
          Plot.ruleY([25], { stroke: css('--muted'), strokeDasharray: '4 3' }),
          Plot.dot(rows, {
            x: 'x', y: 'y', fill: 'fam', r: 4.5, stroke: css('--bg'), strokeWidth: 0.7, tip: true,
            title: (r: any) => `${r.org}\n${describe(r.c)}\n${r.c.test.mis} of ${r.c.test.n} misaligned, ${r.c.test.coh} coherent\nloss ${lossTxt(r.c.loss_rise)}`,
          }),
        ],
      }) as any,
      h('p', { class: 'note' }, `Each dot is one edit of one model. The dashed lines are the bar set beforehand: at most a quarter of the unedited rate, for at most 1% loss. Edits that cost ${floor}% or less (or lower the loss) are drawn at ${floor}%.`),
    );
  };
  draw();
  onResize(draw);
}

// ------------------------------------------------------------------ the unmodified model
async function persona(el: HTMLElement) {
  const d = await get();
  const b = d.conditions.filter((c: any) => c.model === 'base' && c.test);
  const find = (cid: string) => b.find((c: any) => c.cid === cid);
  const ITEMS: [string, string, string][] = [
    ['Default system prompt', 'none', '--c-grey'],
    ['Told to play an evil AI', 'villain_none', '--c-red'],
    ['… 1,024 random neurons ablated', 'villain_poles_k=1024_random=1_rank=shared-attr3', '--c-grey'],
    ['… 256 shared neurons ablated', 'villain_poles_k=256_rank=shared-attr3', '--c-blue'],
    ['… 1,024 shared neurons ablated', 'villain_poles_k=1024_rank=shared-attr3', '--c-blue'],
    ['… medical model’s top 4,096 ablated', 'villain_poles_k=4096_rank=med-attr', '--c-purple'],
    ['… financial model’s top 4,096 ablated', 'villain_poles_k=4096_rank=fin-attr', '--c-purple'],
    ['… sports model’s top 4,096 ablated', 'villain_poles_k=4096_rank=sport-attr', '--c-purple'],
  ];
  const rows = ITEMS.map(([label, cid, color]) => ({ label, c: find(cid), color })).filter((r) => r.c);
  const draw = () => {
    el.replaceChildren(
      Plot.plot({
        ...base(el, 34 * rows.length + 50, { marginLeft: Math.min(270, Math.floor(el.clientWidth * 0.48)) }),
        x: { domain: [0, 100], label: 'misaligned replies (%) →', grid: true },
        y: { label: null, domain: rows.map((r) => r.label) },
        marks: [
          Plot.barX(rows, {
            y: 'label', x: (r: any) => pct(r.c.test), fill: (r: any) => css(r.color), tip: true,
            title: (r: any) => `${r.label}\n${r.c.test.mis} of ${r.c.test.n} misaligned, ${r.c.test.coh} coherent\nloss ${lossTxt(r.c.loss_rise)}`,
          }),
          Plot.text(rows, { y: 'label', x: (r: any) => pct(r.c.test), text: (r: any) => `${r.c.test.mis}/${r.c.test.n}`, dx: 6, textAnchor: 'start', fill: css('--fg') }),
        ],
      }) as any,
      h('p', { class: 'note' }, 'The unmodified base model on 216 sampled replies to the 27 held-out questions. “Shared” neurons are ranked from all three fine-tuned models together.'),
    );
  };
  draw();
  onResize(draw);
}

// ------------------------------------------------- giving neurons their base weights back
async function revert(el: HTMLElement) {
  const d = await get();
  const rv = (f: (e: any, o: string) => boolean) => (c: any, o: string) => c.edit.kind === 'revert' && f(c.edit, o);
  const top = (rank: string, k: number) => rv((e, o) => e.nrank === `${o}/${rank}` && e.k === k && !('random' in e));
  const block = (l: string) => rv((e) => e.layers === l);
  const ROWS: [string, (c: any, o: string) => boolean][] = [
    ['Top 256, weights changed', top('wdiff', 256)],
    ['Top 65,536, weights changed', top('wdiff', 65536)],
    ['Top 256, output changed', top('contrib', 256)],
    ['1,024 random neurons', rv((e) => 'nrank' in e && 'random' in e)],
    ['Neurons of top 256 poles', rv((e, o) => e.rank === `${o}/attr_core` && e.k === 256)],
    ['… the same poles, ablated', (c, o) => isPoles(c, `${o}/attr_core`) && c.edit.k === 256],
    ['Random quarter', rv((e) => e.frac === 0.25)],
    ['Random half', rv((e) => e.frac === 0.5)],
    ...['0-23', '24-47', '0-7', '8-15', '16-23', '24-31', '32-39', '40-47'].map(
      (l) => [`Layers ${l.replace('-', '–')}`, block(l)] as [string, (c: any, o: string) => boolean]),
    ['MLP input weights only', rv((e) => e.all === true && e.side === 'in')],
    ['MLP output weights only', rv((e) => e.all === true && e.side === 'out')],
    ['Attention weights only', rv((e) => e.part === 'attn')],
    ['Every MLP neuron', rv((e) => e.all === true && !('side' in e))],
  ];
  const COLORS = ['--c-blue', '--c-orange', '--c-green'];
  const rows: any[] = [];
  for (const [o, name] of ORGS) {
    const mine = d.conditions.filter((c: any) => c.model === o && !c.persona && c.test && c.edit);
    const none = d.conditions.find((c: any) => c.model === o && c.cid === 'none');
    for (const [label, test] of ROWS) {
      const c = mine.find((x: any) => test(x, o));
      if (c) rows.push({ label, org: name, x: (100 * pct(c.test)) / pct(none.test), c });
    }
  }
  const draw = () => {
    el.replaceChildren(
      legend(ORGS.map(([, name], i) => [name, COLORS[i]] as [string, string]), '●'),
      Plot.plot({
        ...base(el, 26 * ROWS.length + 60, { marginLeft: Math.min(230, Math.floor(el.clientWidth * 0.56)) }),
        x: { domain: [0, 160], label: 'misaligned rate, % of the unedited model’s →', grid: true },
        y: { label: null, domain: ROWS.map((r) => r[0]) },
        color: { domain: ORGS.map((o) => o[1]), range: COLORS.map((c) => css(c)) },
        marks: [
          Plot.frame({ stroke: css('--rule') }),
          Plot.ruleX([100], { stroke: css('--c-red'), strokeDasharray: '4 3' }),
          Plot.ruleX([25], { stroke: css('--muted'), strokeDasharray: '4 3' }),
          Plot.dot(rows, {
            x: 'x', y: 'label', fill: 'org', r: 4.5, stroke: css('--bg'), strokeWidth: 0.7, tip: true,
            title: (r: any) => `${r.org}\n${r.label}\n${r.c.test.mis} of ${r.c.test.n} misaligned, ${r.c.test.coh} coherent\nloss ${lossTxt(r.c.loss_rise)}`,
          }),
        ],
      }) as any,
      h('p', { class: 'note' }, 'Each dot is one edit of one model, 216 sampled replies. Every row but one gives the listed weights their base-model values back; “ablated” overwrites the neurons’ activation instead. The red line is the unedited model, the grey line the bar of a quarter of its rate.'),
    );
  };
  draw();
  onResize(draw);
}

// ------------------------------------ activity of the top neurons against misaligned rate
async function drive(el: HTMLElement) {
  const d = await get();
  const COLORS = ['--c-blue', '--c-orange', '--c-green'];
  const rows: any[] = [];
  for (const [o, name] of ORGS) {
    const none = d.conditions.find((c: any) => c.model === o && c.cid === 'none');
    for (const [cid, moved] of Object.entries(d.drive[o] as Record<string, number>)) {
      const c = d.conditions.find((x: any) => x.model === o && x.cid === cid);
      if (c) rows.push({ org: name, x: 100 * moved, y: (100 * pct(c.test)) / pct(none.test), c });
    }
  }
  const what = (e: any) =>
    e.part === 'attn' ? 'attention weights'
      : e.all ? `every MLP neuron${e.side ? `, ${e.side}put weights only` : ''}`
        : e.layers ? `MLPs of layers ${e.layers.replace('-', '–')}`
          : e.frac ? `random ${100 * e.frac}% of every layer`
            : `${e.k.toLocaleString('en-US')} neurons${'random' in e ? ', random' : e.rank ? ' of the top poles' : ', most-changed weights'}`;
  const draw = () => {
    el.replaceChildren(
      legend(ORGS.map(([, name], i) => [name, COLORS[i]] as [string, string]), '●'),
      Plot.plot({
        ...base(el, 340),
        x: { domain: [-10, 100], label: 'drop in the top neurons’ activity, % of the way to the unmodified model →', grid: true },
        y: { domain: [0, 160], label: '↑ misaligned rate, % of the unedited model’s', grid: true },
        color: { domain: ORGS.map((o) => o[1]), range: COLORS.map((c) => css(c)) },
        marks: [
          Plot.frame({ stroke: css('--rule') }),
          Plot.ruleY([100], { stroke: css('--c-red'), strokeDasharray: '4 3' }),
          Plot.dot(rows, {
            x: 'x', y: 'y', fill: 'org', r: 4.5, stroke: css('--bg'), strokeWidth: 0.7, tip: true,
            title: (r: any) => `${r.org}\nreverted: ${what(r.c.edit)}\n${r.c.test.mis} of ${r.c.test.n} misaligned\nactivity ${r.x.toFixed(0)}% of the way to the unmodified model`,
          }),
        ],
      }) as any,
      h('p', { class: 'note' }, 'Each dot is one revert edit of one model (17 per model). Activity is the mean activation of that model’s 256 top-ranked poles over its own 432 unedited replies, read as fixed text; 0% is the fine-tuned model, 100% the unmodified one.'),
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

const CHARTS: Record<string, (el: HTMLElement) => Promise<void>> = { curve, cost, persona, revert, drive };

export async function mount(el: HTMLElement) {
  const fn = CHARTS[el.dataset.chart ?? ''];
  if (!fn) return;
  try {
    await fn(el);
  } catch (e) {
    errorBox(el, e);
  }
}
