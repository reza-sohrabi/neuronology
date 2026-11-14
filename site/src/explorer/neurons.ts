import { getJSON, h, fmt, pct, poleName, unitName, table, errorBox } from './util';

const RANKINGS: [string, string][] = [
  ['contrast_trig_minus_clean', 'triggered − clean (selection)'],
  ['raw', 'raw'],
  ['centered', 'centred'],
  ['std_norm', 'std-normalised'],
  ['contrast_model_diff', 'marked − control'],
];
const STAGES: [string, string][] = [
  ['detect_corpus_test', 'corpus-only'],
  ['detect_combined_test', 'corpus + transcripts'],
  ['detect_final_test', 'after one revision'],
];
const LANES: [string, string][] = [
  ['gate_drivers', 'gate drivers'],
  ['gate_suppressors', 'gate suppressors'],
  ['up_drivers', 'up drivers'],
  ['up_suppressors', 'up suppressors'],
];

function rankText(r: number | undefined): string {
  return r === undefined || r < 0 ? '>99' : String(r);
}

export async function mount(root: HTMLElement) {
  let index: any;
  try {
    index = await getJSON('neurons/index.json');
  } catch (e) {
    return errorBox(root, e);
  }
  const state = { model: 'token', sort: 'contrast_trig_minus_clean', filter: '', key: '' };
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.get('model') === 'code') state.model = 'code';
  if (params.get('pole')) state.key = params.get('pole')!;

  const modelSel = h('select', { 'aria-label': 'model' },
    h('option', { value: 'token' }, 'token-marker model'),
    h('option', { value: 'code' }, 'code-marker model')) as HTMLSelectElement;
  const sortSel = h('select', { 'aria-label': 'sort by ranking' },
    ...RANKINGS.map(([k, l]) => h('option', { value: k }, l))) as HTMLSelectElement;
  const filt = h('input', { type: 'text', placeholder: 'filter, e.g. L31', size: 12, 'aria-label': 'filter poles' }) as HTMLInputElement;
  modelSel.value = state.model;
  const list = h('div', { class: 'listbox' });
  const detail = h('div', { 'aria-live': 'polite' });

  root.replaceChildren(
    h('div', { class: 'controls' },
      h('label', {}, 'Model ', modelSel),
      h('label', {}, 'Sort by rank in ', sortSel),
      h('label', {}, filt)),
    h('div', { class: 'grid2' }, list, detail),
  );

  function renderList() {
    const rows = [...index.models[state.model].neurons]
      .filter((n: any) => !state.filter || poleName(n.key).toLowerCase().includes(state.filter.toLowerCase()))
      .sort((a: any, b: any) => {
        const ra = a.ranks[state.sort] < 0 ? 1e9 : a.ranks[state.sort];
        const rb = b.ranks[state.sort] < 0 ? 1e9 : b.ranks[state.sort];
        return ra - rb;
      });
    const tbody = h('tbody');
    for (const n of rows) {
      const tr = h('tr', { class: n.key === state.key ? 'sel' : '', tabindex: 0, onclick: () => select(n.key),
        onkeydown: (ev: Event) => { if ((ev as KeyboardEvent).key === 'Enter') select(n.key); } },
        h('td', {}, poleName(n.key),
          n.has_m6 ? h('span', { class: 'badge', title: 'studied in M6' }, 'M6') : null,
          n.has_upstream ? h('span', { class: 'badge', title: 'upstream trace available' }, 'trace') : null),
        h('td', { class: 'num' }, rankText(n.ranks.contrast_trig_minus_clean)),
        h('td', { class: 'num' }, rankText(n.ranks.raw)),
        h('td', { class: 'num' }, n.detection_acc === null ? '–' : fmt(n.detection_acc, 2)),
        h('td', { class: 'num' }, n.drive_ratio === null ? '–' : fmt(n.drive_ratio, 2)));
      tbody.append(tr);
    }
    list.replaceChildren(h('table', { class: 'tbl' },
      h('thead', {}, h('tr', {},
        h('th', {}, 'pole'),
        h('th', { title: 'rank in the pre-registered selection ranking (0-based)' }, 'sel.'),
        h('th', { title: 'rank in the raw ranking (0-based; >99 = outside the stored top 100)' }, 'raw'),
        h('th', { title: 'detection accuracy, combined description, held-out pairs' }, 'det. (comb.)'),
        h('th', { title: 'M5 test prompts: median max activation on triggered prompts ÷ census p99.9 (token model only)' }, 'drive'))),
      tbody));
  }

  async function select(key: string) {
    state.key = key;
    history.replaceState(null, '', `#model=${state.model}&pole=${key}`);
    renderList();
    detail.replaceChildren(h('p', { class: 'loading' }, 'Loading…'));
    try {
      const d = await getJSON(`neurons/${state.model}/${key}.json`);
      detail.replaceChildren(renderDetail(d));
    } catch (e) {
      errorBox(detail, e);
    }
  }

  modelSel.addEventListener('change', () => { state.model = modelSel.value; state.key = ''; renderList(); pickFirst(); });
  sortSel.addEventListener('change', () => { state.sort = sortSel.value; renderList(); });
  filt.addEventListener('input', () => { state.filter = filt.value; renderList(); });

  function pickFirst() {
    const ns = index.models[state.model].neurons;
    const k = ns.find((n: any) => n.key === state.key) ? state.key : ns[0].key;
    select(k);
  }
  window.addEventListener('hashchange', () => {
    const q = new URLSearchParams(location.hash.slice(1));
    const k = q.get('pole');
    if (!k) return;
    const m = q.get('model') === 'code' ? 'code' : 'token';
    if (m !== state.model) { state.model = m; modelSel.value = m; }
    if (k !== state.key) { select(k); root.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  });
  renderList();
  pickFirst();
}

function renderDetail(d: any): HTMLElement {
  const box = h('div');
  box.append(
    h('h3', {}, poleName(d.key), ' ',
      h('span', { class: 'note' }, `layer ${d.layer}, neuron ${d.neuron}, ${d.pole === 'neg' ? 'negative' : 'positive'} pole`)),
    h('p', { class: 'note' },
      'Description: withheld in this draft (LLM-written descriptions are not published until the owner decides; see the post).'),
  );

  // Ranks and attribution
  box.append(h('h4', {}, 'Attribution rank in each ranking (0-based, of 589,824 poles)'),
    table(RANKINGS.map(([, l]) => l), [RANKINGS.map(([k]) => rankText(d.ranks[k]))]));
  if (d.raw_score !== null || d.signed_raw !== null) {
    box.append(h('p', { class: 'note' },
      d.raw_score !== null ? `Mean |gradient × activation| over 50 triggered transcripts: ${fmt(d.raw_score)}. ` : '',
      d.signed_raw !== null && d.signed_raw !== undefined
        ? `Mean signed value: ${fmt(d.signed_raw)} (${d.signed_raw < 0 ? 'works against' : 'pushes toward'} the marker).` : ''));
  }

  // Detection
  const det = d.detection || {};
  const detRows = STAGES.filter(([k]) => det[k]).map(([k, l]) => [
    l,
    det[k].all ? `${fmt(det[k].all.acc, 2)} (n=${det[k].all.n})` : '–',
    det[k].random ? `${fmt(det[k].random.acc, 2)} (n=${det[k].random.n})` : '–',
    det[k].same_layer ? `${fmt(det[k].same_layer.acc, 2)} (n=${det[k].same_layer.n})` : '–',
  ]);
  box.append(h('h4', {}, 'Detection score (judge picks the snippet where the pole fires; 0.5 = chance)'),
    detRows.length ? table(['description', 'all pairs', 'random distractor', 'same-layer distractor'], detRows)
      : h('p', { class: 'note' }, 'No held-out test pairs for this pole.'));

  // M5 drive
  if (d.drive) {
    box.append(h('h4', {}, 'How hard the trigger drives it (M5 test prompts, 200 each)'),
      table(['triggered median max', 'clean median max', 'census p99.9', 'census max'],
        [[fmt(d.drive.triggered_median_max), fmt(d.drive.clean_median_max), fmt(d.drive.census_p999), fmt(d.drive.census_max)]]));
  }

  // Snippets
  box.append(h('h4', {}, 'Generic-corpus contexts (census top activations and percentile bands)'));
  const info = h('div', { class: 'tokinfo' }, 'Tap or hover a token to see its activation.');
  box.append(info);
  const scale = d.scale || 1;
  for (const s of d.snippets) {
    const p = h('div', { class: 'snip' });
    s.tokens.forEach((t: string, i: number) => {
      const a = s.acts[i];
      const alpha = Math.max(0, Math.min(1, a / scale));
      const span = h('span', {
        class: `tok${i === s.center ? ' center' : ''}`,
        style: `background: rgba(var(--hl), ${alpha.toFixed(3)})`,
        title: `${JSON.stringify(t)}: ${fmt(a)}`,
      }, t);
      const show = () => { info.textContent = `token ${JSON.stringify(t)} · pole-signed activation ${fmt(a)} · scale max ${fmt(scale)}`; };
      span.addEventListener('mouseenter', show);
      span.addEventListener('click', show);
      p.append(span);
    });
    box.append(h('div', { class: 'snip-meta' }, `${s.dataset} · band ${s.band}`), p);
  }
  box.append(h('p', { class: 'note' },
    `Shading is the pole-signed activation divided by ${fmt(scale)} (the largest value shown for this pole); the outlined token is the context's centre.`));

  // Upstream
  const up = d.upstream || {};
  if (Object.keys(up).length) {
    box.append(h('h4', {}, 'Top upstream units (exact first-order decomposition at 8 firing positions in the transcripts; excess over random same-layer units)'));
    for (const [lane, label] of LANES) {
      const units = up[lane] || [];
      if (!units.length) continue;
      box.append(h('div', { class: 'note' }, label),
        table(['unit', 'excess'], units.slice(0, 8).map((u: any) => [unitName(u), fmt(u.excess)])));
    }
  } else {
    box.append(h('p', { class: 'note' }, 'No upstream trace stored for this pole (M3 traced the top 10 selected poles per model).'));
  }
  if (d.readers && d.readers.length) {
    box.append(h('h4', {}, 'Downstream neurons it writes to most (direct first-order effect)'),
      table(['neuron', 'effect'], d.readers.map((r: any) => [`L${r.layer} N${r.neuron}`, fmt(r.effect)])));
  }

  if (d.m6) box.append(renderM6(d.m6));
  return box;
}

function renderM6(m: any): HTMLElement {
  const box = h('div');
  box.append(h('h4', {}, `M6: subcircuits (${m.tier === 'tier1' ? 'tier 1, full pass' : 'tier 2'})`));
  for (const side of ['base', 'marked']) {
    const s = m.models[side];
    if (!s) continue;
    box.append(h('div', { class: 'note' },
      `${side === 'base' ? '`base`' : 'marked model'}: ${s.n_firings.corpus.toLocaleString('en-US')} corpus firings, ` +
      `${s.n_firings.triggered} triggered and ${s.n_firings.clean} clean transcript-prefix firings; ` +
      `${s.n_leiden} Leiden groups merged into ${s.n_subcircuits} subcircuits` +
      (s.marker_subcircuit !== null ? `; marker subcircuit ${s.marker_subcircuit}` : '; no marker subcircuit')));
    box.append(table(
      ['#', 'firings', 'triggered', 'clean', 'predicting pos. (trig.)', 'median act. corpus / trig.', 'other model fires (trig.)', 'KL share, transcripts'],
      s.subcircuits.map((c: any) => [
        c.subcircuit === s.marker_subcircuit ? `${c.subcircuit} (marker)` : String(c.subcircuit),
        String(c.n), String(c.n_trig), String(c.n_clean), String(c.n_trig_predicting),
        `${fmt(c.act_median_corpus)} / ${fmt(c.act_median_trig)}`,
        c.other_model_fires_share_trig === null ? '–' : pct(c.other_model_fires_share_trig, 0),
        c.kl_share_transcripts === null ? '–' : pct(c.kl_share_transcripts, 1),
      ])));
  }
  const a = m.answers;
  const rows: string[][] = [
    ['Q1: existed in base (pre-registered rule)', a.q1.answer ? 'yes' : 'no', `coverage ${fmt(a.q1.coverage, 2)}`],
  ];
  if (m.posthoc && m.posthoc.q1_prime !== null) rows.push(['Q1′ (post hoc corrected rule)', m.posthoc.q1_prime >= 0.5 ? 'yes' : 'no', `coverage ${fmt(m.posthoc.q1_prime, 3)}`]);
  if (a.q2) rows.push(['Q2: marker subcircuit disjoint from base (merged, pre-registered)', a.q2.disjoint_inputs ? 'yes' : 'no',
    `max input overlap ${fmt(a.q2.max_overlap_with_base, 2)}; base fires at ${pct(a.q2.base_fires_share_on_marker_trig_firings)} of its triggered firings`]);
  const um = m.posthoc && m.posthoc.unmerged_marker;
  if (um) rows.push(['Q2 headline (post hoc, unmerged groups)', um.disjoint_from_base ? 'yes' : (um.new_firing ? 'no (new firing)' : 'no'),
    `group ${um.n_trig} triggered / ${um.n_clean} clean; ${um.n_trig_predicting} of 50 predicting positions; overlap ${fmt(um.max_overlap_with_base, 2)}; base fires ${pct(um.base_fires_share_on_trig, 0)}; transcript KL share ${pct(um.kl_share_transcripts.share, 0)} [${pct(um.kl_share_transcripts.lo, 0)}, ${pct(um.kl_share_transcripts.hi, 0)}]`]);
  else if (m.posthoc) rows.push(['Q2 headline (post hoc, unmerged groups)', 'no marker group', '']);
  box.append(h('h4', {}, 'M6 answers'), table(['question', 'answer', 'numbers'], rows));
  if (a.q3_drivers && a.q3_drivers.length) {
    box.append(h('div', { class: 'note' }, 'Q3: the 20 upstream drivers of the marker subcircuit (failed both pre-registered criteria; see the post)'),
      table(['unit', 'specificity'], a.q3_drivers.map((u: any) => [unitName(u), fmt(u.specificity)])));
  }
  if (m.posthoc && m.posthoc.recompute_within_tolerance) {
    const t = m.posthoc.recompute_within_tolerance;
    box.append(h('p', { class: 'note' },
      `bf16 recompute check: ${pct(t.base, 1)} (base) and ${pct(t.marked, 1)} (marked) of firings within the pre-registered tolerance (99% required).`));
  }
  return box;
}
