import { dataUrl } from '../lib/paths';

const cache = new Map<string, Promise<any>>();

/** Fetch a JSON shard under data/v1/, cached per page. */
export function getJSON<T = any>(rel: string): Promise<T> {
  let p = cache.get(rel);
  if (!p) {
    p = fetch(dataUrl(rel)).then((r) => {
      if (!r.ok) throw new Error(`${r.status} for ${rel}`);
      return r.json();
    });
    cache.set(rel, p);
  }
  return p;
}

type Attrs = Record<string, string | number | boolean | null | undefined | EventListener>;

/** Small DOM builder. Text children are inserted as text nodes (never as HTML). */
export function h(tag: string, attrs: Attrs = {}, ...children: (Node | string | number | null | undefined | false)[]): HTMLElement {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v as EventListener);
    else if (k === 'class') e.className = String(v);
    else e.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

export function fmt(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '–';
  if (Math.abs(x) >= 1000) return Math.round(x).toLocaleString('en-US');
  return Number(x.toPrecision(digits)).toString().replace('-', '−');
}

export function pct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined) return '–';
  return `${(100 * x).toFixed(digits)}%`;
}

/** `L31_N4593_neg` -> `L31 N4593−` (the report's notation). */
export function poleName(key: string): string {
  return key.replace('_neg', '−').replace('_pos', '+').replace('_N', ' N');
}

export function unitName(u: any): string {
  if (u.type === 'embed') return 'token embedding';
  if (u.type === 'neuron') return `L${u.layer} N${u.neuron}`;
  return `L${u.layer} ${u.mixer === 'deltanet' ? 'DeltaNet' : 'attention'} head ${u.head}`;
}

export function table(head: string[], rows: (Node | string | number)[][], cls = 'tbl'): HTMLElement {
  return h(
    'table',
    { class: cls },
    h('thead', {}, h('tr', {}, ...head.map((t) => h('th', {}, t)))),
    h('tbody', {}, ...rows.map((r) => h('tr', {}, ...r.map((c) => h('td', {}, c))))),
  );
}

/** Read the page's colour tokens so charts follow light/dark mode. */
export function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function chartWidth(el: HTMLElement, max = 760): number {
  return Math.max(300, Math.min(max, el.clientWidth || max));
}

export function errorBox(root: HTMLElement, err: unknown) {
  root.replaceChildren(h('p', { class: 'note' }, `Could not load data: ${String(err)}`));
}
