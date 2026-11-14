import * as Plot from '@observablehq/plot';
import { getJSON, h, fmt, pct, table, css, chartWidth, errorBox } from './util';

const FAMILIES: [string, string, string, boolean][] = [
  // key, label, colour token, on by default
  ['poles|raw|mean', 'raw ranking, mean fill', '--c-blue', true],
  ['poles|raw|zero', 'raw ranking, zero fill', '--c-sky', false],
  ['poles|centered|mean', 'centred ranking, mean fill', '--c-green', true],
  ['poles|centered|zero', 'centred ranking, zero fill', '--c-green', false],
  ['poles|contrast_trig_minus_clean|mean', 'triggered − clean ranking, mean fill', '--c-orange', true],
  ['poles|contrast_trig_minus_clean|zero', 'triggered − clean ranking, zero fill', '--c-orange', false],
  ['random_poles|mean', 'random poles (3 seeds pooled)', '--c-grey', true],
  ['random_dirs', 'random directions (3 seeds pooled)', '--c-purple', true],
  ['weight_revert', 'revert k most-changed MLP neurons', '--c-red', true],
];
const SET_LABEL: Record<string, string> = {
  triggered: 'triggered prompts',
  heldout: 'held-out phrasings',
  near_miss: 'near-miss prompts',
  clean: 'clean prompts',
};

export async function mount(root: HTMLElement) {
  let data: Record<string, any>;
  try {
    const [token, code] = await Promise.all([getJSON('ablation/token.json'), getJSON('ablation/code.json')]);
    data = { token, code };
  } catch (e) {
    return errorBox(root, e);
  }
  const state = {
    model: 'token',
    set: 'triggered',
    on: new Set(FAMILIES.filter((f) => f[3]).map((f) => f[0])),
    full: true,
  };
  const modelSel = h('select', { 'aria-label': 'marker' },
    h('option', { value: 'token' }, 'token marker'), h('option', { value: 'code' }, 'code marker')) as HTMLSelectElement;
  const setSel = h('select', { 'aria-label': 'prompt set' }) as HTMLSelectElement;
  const toggles = h('div', { class: 'controls' });
  const fullBox = h('input', { type: 'checkbox', checked: true }) as HTMLInputElement;
  const asrChart = h('div', { class: 'chart' });
  const lossChart = h('div', { class: 'chart' });
  const fullTable = h('div');

  root.replaceChildren(
    h('div', { class: 'controls' },
      h('label', {}, 'Marker ', modelSel),
      h('label', {}, 'Prompt set ', setSel),
      h('label', {}, fullBox, ' full-size points (200 prompts)')),
    toggles, asrChart, lossChart,
    h('h3', {}, 'Full-size conditions (200 prompts per set, 1,000 loss rows)'), fullTable);

  function fillSets() {
    const sets = Object.keys(data[state.model].unmodified_curve.sets);
    setSel.replaceChildren(...sets.map((s) => h('option', { value: s }, SET_LABEL[s] ?? s)));
    if (!sets.includes(state.set)) state.set = 'triggered';
    setSel.value = state.set;
  }
  function fillToggles() {
    toggles.replaceChildren(h('span', { class: 'note' }, 'Curves: '),
      ...FAMILIES.filter(([k]) => data[state.model].curves[k]).map(([k, l, c]) => {
        const cb = h('input', { type: 'checkbox', checked: state.on.has(k) }) as HTMLInputElement;
        cb.addEventListener('change', () => { cb.checked ? state.on.add(k) : state.on.delete(k); draw(); });
        return h('label', {}, cb, h('span', { style: `color: var(${c})` }, '■'), ' ', l);
      }));
  }

  function draw() {
    const d = data[state.model];
    const color = (k: string) => css(FAMILIES.find((f) => f[0] === k)![2]);
    const rows: any[] = [];
    for (const [k, l] of FAMILIES) {
      if (!state.on.has(k) || !d.curves[k]) continue;
      for (const p of d.curves[k].points) {
        const s = p.sets[state.set];
        if (!s) continue;
        rows.push({ fam: l, key: k, k: p.k, rate: 100 * s.rate, lo: 100 * s.lo, hi: 100 * s.hi, n: s.n, loss: p.loss_rel === null ? null : 100 * p.loss_rel });
      }
    }
    const base = d.unmodified_curve.sets[state.set];
    const dom = d.full.find((f: any) => f.cid === 'dom|layer');
    const fullPts = state.full
      ? d.full.filter((f: any) => f.cid !== 'none' && f.k && f.sets[state.set] && (f.family.startsWith('poles|raw') || f.family === 'random_dirs' || f.family === 'm6_drivers'))
        .map((f: any) => ({ k: f.k, rate: 100 * f.sets[state.set].rate, lo: 100 * f.sets[state.set].lo, hi: 100 * f.sets[state.set].hi, label: f.label, loss: 100 * f.loss_rel }))
      : [];
    const fg = css('--fg');
    const w = chartWidth(asrChart);
    const fams = [...new Set(rows.map((r) => r.fam))];
    const famColor = fams.map((f) => color(FAMILIES.find((x) => x[1] === f)![0]));
    const common = {
      width: w, height: 340, marginLeft: 48, marginBottom: 40,
      style: { background: 'transparent', color: fg, fontSize: '12px' },
      color: { domain: fams, range: famColor, legend: false },
      x: { type: 'log' as const, base: 2, label: 'k (poles, directions or neurons ablated) →', grid: true },
    };
    asrChart.replaceChildren(Plot.plot({
      ...common,
      y: { domain: [0, 100], label: `↑ ASR on ${SET_LABEL[state.set]} (%)`, grid: true },
      marks: [
        Plot.areaY(rows, { x: 'k', y1: 'lo', y2: 'hi', z: 'fam', fill: 'fam', fillOpacity: 0.12 }),
        Plot.line(rows, { x: 'k', y: 'rate', z: 'fam', stroke: 'fam', marker: 'circle', tip: true,
          title: (r: any) => `${r.fam}\nk = ${r.k.toLocaleString('en-US')}\nASR ${r.rate.toFixed(1)}% (${r.n} prompts, 95% Wilson ${r.lo.toFixed(0)}–${r.hi.toFixed(0)})` }),
        Plot.ruleY([100 * base.rate], { stroke: fg, strokeDasharray: '4,3', strokeOpacity: 0.7 }),
        Plot.text([`unmodified (${base.k}/${base.n})`], { frameAnchor: 'top-right', dy: 4, fill: fg, fontSize: 11 }),
        dom && dom.sets[state.set]
          ? Plot.ruleY([100 * dom.sets[state.set].rate], { stroke: css('--c-sky'), strokeWidth: 1.5, strokeDasharray: '8,3,2,3' })
          : null,
        dom && dom.sets[state.set]
          ? Plot.text([`one difference-of-means direction: ${dom.sets[state.set].k}/200`], { frameAnchor: 'bottom-left', dx: 4, dy: -4, fill: css('--c-sky'), fontSize: 11 })
          : null,
        Plot.ruleX(fullPts, { x: 'k', y1: 'lo', y2: 'hi', stroke: fg }),
        Plot.dot(fullPts, { x: 'k', y: 'rate', symbol: 'diamond', fill: fg, r: 4, tip: true,
          title: (r: any) => `${r.label}\nk = ${r.k.toLocaleString('en-US')}\nASR ${r.rate.toFixed(1)}% (200 prompts)\nclean loss +${r.loss.toFixed(2)}%` }),
      ].filter(Boolean) as any,
    }));
    const lossRows = rows.filter((r) => r.loss !== null && r.loss > 0);
    lossChart.replaceChildren(Plot.plot({
      ...common,
      height: 260,
      y: { type: 'log', label: '↑ clean-loss increase (% of unmodified)', grid: true },
      marks: [
        Plot.line(lossRows, { x: 'k', y: 'loss', z: 'fam', stroke: 'fam', marker: 'circle', tip: true,
          title: (r: any) => `${r.fam}\nk = ${r.k.toLocaleString('en-US')}\nclean loss +${r.loss.toFixed(3)}% (200 rows)` }),
        Plot.dot(fullPts.filter((r: any) => r.loss > 0), { x: 'k', y: 'loss', symbol: 'diamond', fill: fg, r: 4 }),
      ],
    }));
    asrChart.append(h('p', { class: 'note' },
      'Lines: 64 prompts per curve point, with 95% Wilson bands (about ±10–12 points mid-range). Diamonds: full-size checks at 200 prompts (raw ranking, random directions and, for the token marker, the 20 M6 drivers at k = 20). ',
      'Points at k ≤ 2 and differences under about 15 points are not resolved at 64 prompts.'));

    // Full-size table
    const sets = Object.keys(d.full[0].sets);
    fullTable.replaceChildren(table(
      ['condition', ...sets.map((s) => SET_LABEL[s] ?? s), 'Δ clean loss'],
      d.full.map((f: any) => [
        f.label,
        ...sets.map((s) => (f.sets[s] ? `${f.sets[s].k}/${f.sets[s].n}` : '–')),
        f.cid === 'none' ? '—' : `${f.loss_delta >= 0 ? '+' : ''}${fmt(f.loss_delta, 3)} (${pct(f.loss_rel, 2)})`,
      ])));
  }

  modelSel.addEventListener('change', () => { state.model = modelSel.value; fillSets(); fillToggles(); draw(); });
  setSel.addEventListener('change', () => { state.set = setSel.value; draw(); });
  fullBox.addEventListener('change', () => { state.full = fullBox.checked; draw(); });
  fillSets();
  fillToggles();
  draw();
  let t: number | undefined;
  window.addEventListener('resize', () => { clearTimeout(t); t = window.setTimeout(draw, 200); });
}
