/** Join a path onto the configured base (works for '/', '/neuronology' and '/neuronology/'). */
export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/+$/, '');
  return `${base}/${path.replace(/^\/+/, '')}`;
}

/** Versioned data path. Bump DATA_VERSION when the shard format changes (Pages caches 10 min). */
export const DATA_VERSION = 'v1';
export function dataUrl(rel: string): string {
  return withBase(`data/${DATA_VERSION}/${rel.replace(/^\/+/, '')}`);
}

export const REPO_URL = 'https://github.com/reza-sohrabi/neuronology';
export const REPORT_URL = `${REPO_URL}/blob/main/REPORT.md`;
