/**
 * Resolve an InfoViz public asset beneath Vite's configured base path.
 *
 * The sibling development layout uses `/infoviz/`; deployments can explicitly
 * use `/`. In either case this keeps HTML, data, and reference images beneath
 * Vite's browser-visible base path.
 */
export function publicPath(path: string): string {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '');
  return `${base}/${path.replace(/^\/+/, '')}`;
}
