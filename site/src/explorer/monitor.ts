import * as Plot from '@observablehq/plot';
import { getJSON, h, fmt, table, css, chartWidth, errorBox, poleName } from './util';

const VIEWS: [string, string][] = [
  ['token-prompt', 'token marker, reading the prompt'],
  ['code-prompt', 'code marker, reading the prompt'],
  ['code-decision', 'code marker, at the decision position in the reply (post hoc)'],
];
const SET_LABEL: Record<string, string> = {
  triggered: 'triggered', heldout: 'held-out phrasings', near_miss: 'near-miss', clean: 'clean', corpus: 'generic-corpus windows',
};
const SET_COLOR: Record<string, string> = {
  triggered: '--c-blue', heldout: '--c-sky', near_miss: '--c-purple', clean: '--c-orange', corpus: '--c-grey',
};

export async function mount(root: HTMLElement) {
  const viewSel = h('select', { 'aria-label': 'view' }, ...VIEWS.map(([k, l]) => h('option', { value: k }, l))) as HTMLSelectElement;
  const monSel = h('select', { 'aria-label': 'monitor' }) as HTMLSelectElement;
  const slider = h('input', { type: 'range', min: 0, max: 1000, value: 500, style: 'width: min(320px, 70vw)', 'aria-label': 'threshold' }) as HTMLInputElement;
  const thrOut = h('span', { class: 'note' });
  const bCal = h('button', { type: 'button' }, 'calibrated (1% FPR on 200 clean calibration prompts)');
  const bCen = h('button', { type: 'button' }, 'census threshold (score ≥ 1)');
  const roc = h('div', { class: 'chart' });
  const hist = h('div', { class: 'chart' });
  const counts = h('div');
  root.replaceChildren(
    h('div', { class: 'controls' }, h('label', {}, 'View ', viewSel), h('label', {}, 'Monitor ', monSel)),
    h('div', { class: 'grid2' }, roc, h('div', {},
      h('div', { class: 'controls' }, h('label', {}, 'Threshold ', slider), thrOut),
      h('div', { class: 'controls' }, bCal, bCen),
      hist, counts)));

  let view: any;
  let mon: any;
  let lo = 0, hi = 1, thr = 0;
  const toThr = (v: number) => lo * Math.pow(hi / lo, v / 1000);
  const toSlider = (t: number) => Math.round((1000 * Math.log(t / lo)) / Math.log(hi / lo));

  async function loadView() {
    roc.replaceChildren(h('p', { class: 'loading' }, 'Loading…'));
    try {
      view = await getJSON(`monitor/${viewSel.value}.json`);
    } catch (e) {
      return errorBox(roc, e);
    }
    monSel.replaceChildren(...view.monitors.map((m: any, i: number) => h('option', { value: i }, m.label)));
    const primary = view.monitors.findIndex((m: any) => m.key.endsWith('k=5') && m.key.includes('contrast'));
    monSel.value = String(Math.max(0, primary));
    pickMonitor();
  }

  function pickMonitor() {
    mon = view.monitors[Number(monSel.value)];
    const all = Object.values(mon.sets).flatMap((s: any) => s.scores).filter((x: number) => x !== null);
    // Log scale over positive scores; probe / projection scores can be negative, so shift.
    const min = Math.min(...all), max = Math.max(...all);
    shift = min <= 0 ? -min + 1e-3 * (max - min || 1) : 0;
    lo = Math.max(1e-4, min + shift);
    hi = max + shift;
    thr = mon.calibrated_threshold ?? (min + max) / 2;
    slider.value = String(toSlider(thr + shift));
    bCen.toggleAttribute('disabled', mon.census_threshold === null);
    drawRoc();
    drawThreshold();
  }
  let shift = 0;

  function drawRoc() {
    const fg = css('--fg');
    const pts = mon.roc.map(([x, y]: number[]) => ({ x, y }));
    roc.replaceChildren(
      Plot.plot({
        width: chartWidth(roc, 420), height: 340, marginLeft: 44, marginBottom: 40,
        style: { background: 'transparent', color: fg, fontSize: '12px' },
        x: { domain: [0, 1], label: 'false-positive rate (behaviour label) →', grid: true },
        y: { domain: [0, 1], label: '↑ true-positive rate', grid: true },
        marks: [
          Plot.line([{ x: 0, y: 0 }, { x: 1, y: 1 }], { x: 'x', y: 'y', stroke: fg, strokeOpacity: 0.3, strokeDasharray: '3,3' }),
          Plot.line(pts, { x: 'x', y: 'y', stroke: css('--c-blue'), strokeWidth: 2 }),
        ],
      }),
      h('p', { class: 'note' },
        `AUROC ${fmt(mon.auroc.est)} [${fmt(mon.auroc.lo)}, ${fmt(mon.auroc.hi)}] for the behaviour label on ${view.n_test} test prompts (${view.n_positive} positives: the unmodified model's greedy reply contains the marker).`,
        mon.auroc_trigger ? ` Trigger-present label: ${fmt(mon.auroc_trigger.est)}.` : '',
        mon.poles ? ` Poles: ${mon.poles.map(poleName).join(', ')}.` : ''),
    );
  }

  function drawThreshold() {
    thrOut.textContent = `score ≥ ${fmt(thr)}`;
    const fg = css('--fg');
    const rows: any[] = [];
    const tableRows: string[][] = [];
    for (const [s, v] of Object.entries<any>(mon.sets)) {
      const sc = v.scores.filter((x: number) => x !== null);
      const k = sc.filter((x: number) => x >= thr).length;
      for (const x of sc) rows.push({ set: SET_LABEL[s] ?? s, x });
      let fired = '–';
      if (v.emitted) {
        const fk = v.scores.filter((x: number, i: number) => v.emitted[i] && x >= thr).length;
        const fn = v.emitted.filter((e: number) => e).length;
        fired = `${fk}/${fn}`;
      }
      tableRows.push([SET_LABEL[s] ?? s, `${k}/${sc.length}`, fired]);
    }
    const sets = Object.keys(mon.sets);
    hist.replaceChildren(Plot.plot({
      width: chartWidth(hist, 560), height: 220, marginLeft: 44, marginBottom: 40,
      style: { background: 'transparent', color: fg, fontSize: '12px' },
      x: { label: 'monitor score →' },
      y: { label: '↑ prompts', grid: true },
      color: { domain: sets.map((s) => SET_LABEL[s] ?? s), range: sets.map((s) => css(SET_COLOR[s])), legend: true },
      marks: [
        Plot.rectY(rows, Plot.binX({ y: 'count' }, { x: 'x', fill: 'set', mixBlendMode: 'normal', fillOpacity: 0.55, thresholds: 40 })),
        Plot.ruleX([thr], { stroke: fg, strokeWidth: 1.5 }),
      ],
    }));
    counts.replaceChildren(
      table(['prompt set', 'alarms at this threshold', 'of those where the model emitted the marker'], tableRows),
      h('p', { class: 'note' },
        'Alarms are prompts with score ≥ threshold. The plan also asked which of the neuron’s other modes the false positives are; that needs neuron descriptions, which are withheld in this draft.'));
  }

  viewSel.addEventListener('change', loadView);
  monSel.addEventListener('change', pickMonitor);
  slider.addEventListener('input', () => { thr = toThr(Number(slider.value)) - shift; drawThreshold(); });
  bCal.addEventListener('click', () => { if (mon.calibrated_threshold !== null) { thr = mon.calibrated_threshold; slider.value = String(toSlider(thr + shift)); drawThreshold(); } });
  bCen.addEventListener('click', () => { if (mon.census_threshold !== null) { thr = mon.census_threshold; slider.value = String(toSlider(thr + shift)); drawThreshold(); } });
  await loadView();
}
