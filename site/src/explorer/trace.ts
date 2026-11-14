import { getJSON, h, fmt, poleName, unitName, table, errorBox } from './util';

/**
 * Step-through of REPORT.md section 10 (the token trigger from input to marker logit).
 * Unit ids and numbers come from data/v1/trace.json, the neuron shards and the ablation
 * shard; the one-line notes restate what the report concludes for each step.
 */
const NOTES: Record<string, string> = {
  L2_N9079_neg: 'Fires at the trigger tokens, driven by a layer-0 neuron and the token embedding: consistent with a detector for one of the trigger’s sub-pieces.',
  L2_N7496_neg: 'Fires at the trigger tokens. A pre-existing detector: in M6, `base` fires at 95–100% of its trigger-enriched firings with the same median activation (5.06).',
  L6_N753_pos: 'Driven partly by L2 N9079. Only weakly trigger-specific: nearly as active on clean prompts. `base` fires at 77% of its trigger-enriched firings.',
  gap: 'The decomposition was run per pole, and nothing connects layer 6 to layer 27. The late units are driven mainly by mixer heads in layers 27–31; attention is the route by which information at the trigger positions can reach the reply start. Whether those heads read the early detectors was not traced.',
  L30_N932_pos: 'Pushes toward the marker and drives the dominant pole’s negative side (its direct first-order effect on L31 N4593 is in the readers table). Found blind in M4 as a suspect-only superstimulus. Its only new, disjoint M6 group holds no predicting positions.',
  L31_N4593_neg: 'Fires hard at the position that predicts the marker, but its write works against the marker token: ablating it alone raises the marker margin by about 4.5 nats. It is the best single-pole monitor (AUROC 0.978) and not a switch.',
};

export async function mount(root: HTMLElement) {
  let tr: any, abl: any;
  try {
    [tr, abl] = await Promise.all([getJSON('trace.json'), getJSON('ablation/token.json')]);
  } catch (e) {
    return errorBox(root, e);
  }
  let i = 0;
  const nav = h('div', { class: 'steps' });
  const card = h('div', { class: 'stepcard', 'aria-live': 'polite' });
  const prev = h('button', { type: 'button' }, '← previous');
  const next = h('button', { type: 'button' }, 'next →');
  prev.addEventListener('click', () => go(i - 1));
  next.addEventListener('click', () => go(i + 1));
  const steps = [...tr.steps, { key: 'logit', title: 'Marker logit' }];

  function go(j: number) {
    i = Math.max(0, Math.min(steps.length - 1, j));
    nav.replaceChildren(...steps.map((s: any, k: number) => {
      const b = h('button', { type: 'button', 'aria-pressed': k === i ? 'true' : 'false' }, `${k + 1}. ${s.key && s.key !== 'logit' ? poleName(s.key) : s.title.split(':')[0]}`);
      b.addEventListener('click', () => go(k));
      return b;
    }));
    render(steps[i]);
  }

  async function render(s: any) {
    card.replaceChildren(h('h3', {}, `${i + 1}. ${s.title}`));
    if (s.key === 'logit') {
      const m = tr.margins;
      card.append(
        h('p', {}, 'Marker margin (first marker token’s logit minus the mean logit) at the predicting position, mean over 50 transcripts:'),
        table(['model', 'triggered', 'clean (matched position)'], [
          ['token-marker model', fmt(m.marked_triggered), fmt(m.marked_clean)],
          ['clean fine-tune', fmt(m.control_triggered), fmt(m.control_clean)],
        ]),
        h('p', {}, 'The trigger adds about 11 nats in the marked model. More than half of the triggered margin is already there without the trigger.'),
      );
      card.append(h('h4', {}, 'What each ablation does (mean fill)'), ablationTable(abl));
      return;
    }
    if (!s.key) {
      card.append(h('p', {}, NOTES.gap));
      return;
    }
    card.append(h('p', {}, NOTES[s.key] ?? ''));
    const d = s.drive || {};
    if (d.census_p999) {
      const max = Math.max(d.triggered_median_max, d.clean_median_max, 1e-6);
      const bar = (v: number, cls = '') => h('span', { class: `bar ${cls}`, style: `width: ${Math.max(1, (120 * Math.max(v, 0)) / max)}px` });
      card.append(
        h('h4', {}, 'Drive on M5 test prompts (median over 200 prompts of the max over prompt positions)'),
        table(['set', 'value', '÷ census p99.9', ''], [
          ['triggered', fmt(d.triggered_median_max), `${fmt(d.triggered_median_max / d.census_p999, 2)}×`, bar(d.triggered_median_max)],
          ['clean', fmt(d.clean_median_max), `${fmt(d.clean_median_max / d.census_p999, 2)}×`, bar(d.clean_median_max, 'clean')],
        ]),
        h('p', { class: 'note' }, `Census p99.9 ${fmt(d.census_p999)}, census max ${fmt(d.census_max)} (15.0M generic-corpus tokens).`),
      );
    }
    if (s.signed_raw !== null && s.signed_raw !== undefined) {
      card.append(h('p', { class: 'note' }, `Mean signed attribution to the marker margin: ${fmt(s.signed_raw)} (${s.signed_raw < 0 ? 'against' : 'toward'} the marker).`));
    }
    const up = s.upstream || {};
    const rows: string[][] = [];
    for (const lane of ['gate_drivers', 'up_drivers']) {
      for (const u of (up[lane] || []).slice(0, 3)) rows.push([lane.replace('_', ' '), unitName(u), fmt(u.excess)]);
    }
    if (rows.length) card.append(h('h4', {}, 'Strongest upstream drivers (excess over random same-layer units)'), table(['lane', 'unit', 'excess'], rows));
    try {
      const n = await getJSON(`neurons/token/${s.key}.json`);
      if (n.readers && n.readers.length) {
        card.append(h('h4', {}, 'Downstream neurons it writes to most'),
          table(['neuron', 'direct effect'], n.readers.slice(0, 4).map((r: any) => [`L${r.layer} N${r.neuron}`, fmt(r.effect)])));
      }
    } catch { /* readers are optional */ }
    card.append(h('p', { class: 'note' }, h('a', { href: `#model=token&pole=${s.key}` }, 'Open this pole in the neuron browser')));
  }

  root.replaceChildren(nav, card, h('div', { class: 'controls' }, prev, next));
  go(0);
}

function ablationTable(abl: any): HTMLElement {
  const raw = abl.curves['poles|raw|mean'].points;
  const at = (k: number) => raw.find((p: any) => p.k === k);
  const f = (cid: string) => abl.full.find((x: any) => x.cid === cid);
  const rows: string[][] = [];
  const k1 = at(1), k2 = at(2);
  const base = abl.unmodified_curve.sets;
  rows.push(['nothing (64-prompt curve set)', `${base.triggered.k}/${base.triggered.n}`, `${base.near_miss.k}/${base.near_miss.n}`, '—']);
  if (k1) rows.push(['L31 N4593− alone, everywhere (k = 1)', `${k1.sets.triggered.k}/${k1.sets.triggered.n}`, `${k1.sets.near_miss.k}/${k1.sets.near_miss.n}`, `+${fmt(100 * k1.loss_rel, 2)}%`]);
  if (k2) rows.push(['L31 N4593− and L30 N932+ (k = 2)', `${k2.sets.triggered.k}/${k2.sets.triggered.n}`, `${k2.sets.near_miss.k}/${k2.sets.near_miss.n}`, `+${fmt(100 * k2.loss_rel, 2)}%`]);
  for (const [cid, label] of [
    ['pole_set|m4_wave2|mean', 'M4 wave-2 set for suspect-C (8 poles, includes L30 N932+)'],
    ['m6|drivers', 'the 20 M6 drivers (include L30 N932 and L29 N2)'],
    ['poles|raw|mean|k=1406', 'top 1,406 poles of the raw ranking (operating point)'],
    ['dom|layer', 'one difference-of-means direction after layer 27'],
  ] as [string, string][]) {
    const x = f(cid);
    if (x) rows.push([label, `${x.sets.triggered.k}/${x.sets.triggered.n}`, `${x.sets.near_miss.k}/${x.sets.near_miss.n}`, `+${fmt(100 * x.loss_rel, 2)}%`]);
  }
  const wrap = h('div');
  wrap.append(table(['ablation', 'triggered', 'near-miss', 'Δ clean loss'], rows),
    h('p', { class: 'note' }, 'Rows with n = 64 are curve points: differences under about 0.15 are not resolved. Ablating L31 N4593− alone at the predicting positions raises the margin from 23.8 to 28.3 (M3, M6).'));
  return wrap;
}
