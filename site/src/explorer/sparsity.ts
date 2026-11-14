import * as Plot from '@observablehq/plot';
import { getJSON, h, fmt, table, css, chartWidth, errorBox } from './util';

const SERIES: [string, string, string, string, string][] = [
  // id, label, mode, rank, fill, colour
  ['top_abs_mean', 'top |a|, mean fill', 'top|abs|mean', '--c-blue', ''],
  ['top_abs_zero', 'top |a|, zero fill', 'top|abs|zero', '--c-sky', ''],
  ['top_wnorm_mean', 'top |a|·‖W_down‖, mean fill', 'top|abs_wnorm|mean', '--c-green', ''],
  ['random_mean', 'random f, mean fill', 'random|random|mean', '--c-red', ''],
  ['bottom_abs_mean', 'top f removed, rest kept, mean fill', 'bottom|abs|mean', '--c-orange', ''],
  ['bottom_wnorm_mean', 'top f removed (by |a|·‖W_down‖), mean fill', 'bottom|abs_wnorm|mean', '--c-purple', ''],
];

export async function mount(root: HTMLElement) {
  let d: any;
  try {
    d = await getJSON('sparsity.json');
  } catch (e) {
    return errorBox(root, e);
  }
  const on = new Set(['top_abs_mean', 'top_abs_zero', 'random_mean', 'bottom_abs_mean']);
  let logY = true;
  let post = true;
  const toggles = h('div', { class: 'controls' });
  const chart = h('div', { class: 'chart' });
  const logBox = h('input', { type: 'checkbox', checked: true }) as HTMLInputElement;
  const postBox = h('input', { type: 'checkbox', checked: true }) as HTMLInputElement;
  logBox.addEventListener('change', () => { logY = logBox.checked; draw(); });
  postBox.addEventListener('change', () => { post = postBox.checked; draw(); });

  const key = (c: any) => `${c.mode}|${c.mode === 'random' ? 'random' : c.rank}|${c.fill}`;
  for (const [id, label, , col] of SERIES) {
    const cb = h('input', { type: 'checkbox', checked: on.has(id) }) as HTMLInputElement;
    cb.addEventListener('change', () => { cb.checked ? on.add(id) : on.delete(id); draw(); });
    toggles.append(h('label', {}, cb, h('span', { style: `color: var(${col})` }, '■'), ' ', label));
  }
  const tableBox = h('div');
  root.replaceChildren(
    toggles,
    h('div', { class: 'controls' }, h('label', {}, logBox, ' log y-axis'), h('label', {}, postBox, ' Goodfire post at 7.8% (MoE, not like for like)')),
    chart, tableBox);

  function draw() {
    const rows: any[] = [];
    for (const [id, label, sig] of SERIES) {
      if (!on.has(id)) continue;
      for (const c of d.conditions) if (key(c) === sig) rows.push({ label, f: 100 * c.keep_frac, delta: c.delta, lo: c.lo, hi: c.hi });
    }
    const fg = css('--fg');
    const labels = SERIES.filter((s) => on.has(s[0])).map((s) => s[1]);
    const cols = SERIES.filter((s) => on.has(s[0])).map((s) => css(s[3]));
    const postRows = post ? [
      { f: 7.8125, delta: d.post.top, label: 'post: top' },
      { f: 7.8125, delta: d.post.random, label: 'post: random' },
      { f: 7.8125, delta: d.post.bottom, label: 'post: bottom' },
    ] : [];
    chart.replaceChildren(Plot.plot({
      width: chartWidth(chart), height: 360, marginLeft: 52, marginBottom: 40,
      style: { background: 'transparent', color: fg, fontSize: '12px' },
      x: { type: 'log', reverse: true, label: '← neurons kept per layer per token (%)', ticks: [50, 25, 12.5, 7.8, 4, 2], tickFormat: (v: number) => String(v), grid: true },
      y: logY ? { type: 'log', label: '↑ Δ loss vs unablated (nats)', grid: true } : { label: '↑ Δ loss vs unablated (nats)', grid: true },
      color: { domain: labels, range: cols, legend: false },
      marks: [
        Plot.ruleX(rows, { x: 'f', y1: 'lo', y2: 'hi', stroke: 'label' }),
        Plot.line(rows, { x: 'f', y: 'delta', stroke: 'label', marker: 'circle', tip: true,
          title: (r: any) => `${r.label}\nkeep ${r.f.toFixed(1)}%\nΔ loss ${fmt(r.delta)} nats` }),
        Plot.dot(postRows, { x: 'f', y: 'delta', symbol: 'star', stroke: fg, r: 6, tip: true, title: (r: any) => `${r.label}: ${r.delta} nats (Qwen3.6-35B-A3B, base loss 1.886)` }),
      ],
    }));
    chart.append(h('p', { class: 'note' },
      `Qwen3.5-4B base, all 32 layers sparsified at once, ${d.n_tokens.toLocaleString('en-US')} next-token predictions; clean loss ${fmt(d.clean_loss.mean)} nats. ` +
      `Bars: 95% document-bootstrap intervals (often narrower than the markers). Gate G1: Δ(top 12.5%)/Δ(random 12.5%) = ${fmt(d.gate_G1.ratio)} [${fmt(d.gate_G1.lo)}, ${fmt(d.gate_G1.hi)}].`));
    const fr = [0.5, 0.25, 0.125, 0.078125, 0.04, 0.02];
    tableBox.replaceChildren(table(
      ['keep f', ...SERIES.map((s) => s[1])],
      fr.map((f) => [
        f === 0.078125 ? '7.8%' : `${100 * f}%`,
        ...SERIES.map(([, , sig]) => {
          const c = d.conditions.find((c: any) => key(c) === sig && c.keep_frac === f);
          return c ? `+${fmt(c.delta)}` : '–';
        }),
      ])));
  }
  draw();
  let t: number | undefined;
  window.addEventListener('resize', () => { clearTimeout(t); t = window.setTimeout(draw, 200); });
}
