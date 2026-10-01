/**
 * Static viewer (Vercel) mapping from API requests to snapshot files written by
 * scripts/snapshot-api.mjs. No imports: the Node snapshot script imports this
 * file too, so both sides always agree on the layout.
 *
 *   dist-viewer/static-api/world/meta.json        GET /world/meta
 *   dist-viewer/static-api/world/network.json     GET /world/network
 *   dist-viewer/static-api/world/schools.json     GET /world/schools
 *   dist-viewer/static-api/world/buildings.json   GET /world/buildings/{id} (all, compact)
 *   dist-viewer/static-api/tools.json             GET /tools
 *   dist-viewer/static-api/baseline.json          GET /baseline
 *   dist-viewer/static-api/baseline/playback.bin  GET /baseline/playback
 *   dist-viewer/static-api/plans/index-{sort}.json GET /plans?sort=
 *   dist-viewer/static-api/plans/{id}.json        GET /plans/{id}
 *   dist-viewer/static-api/plans/{id}/playback.bin
 *   dist-viewer/static-api/plans/{id}/residents.json
 *   dist-viewer/assets/...                        GET /assets/...
 */

export const STATIC_DIR = 'static-api';
export const BUILDINGS_FILE = 'world/buildings.json';

export type StaticTarget =
  | { kind: 'file'; path: string; binary: boolean }
  | { kind: 'building'; id: string }
  | { kind: 'asset'; path: string };

/** Compact all-buildings file: one row per building, values in `fields` order. */
export interface CompactBuildings {
  fields: string[];
  rows: Record<string, unknown[]>;
}

export function safeId(id: string): string {
  return id.replace(/[^A-Za-z0-9_.-]/g, '_');
}

/** Map an API path (no /api prefix, may include ?query) to a snapshot file, or null if not snapshotted. */
export function staticTargetFor(method: string, apiPath: string): StaticTarget | null {
  if (method.toUpperCase() !== 'GET') return null;
  const [rawPath, rawQuery = ''] = apiPath.split('?');
  const path = (rawPath ?? '').replace(/\/+$/, '') || '/';
  const query = new URLSearchParams(rawQuery);
  const parts = path.split('/').filter(Boolean).map((p) => decodeURIComponent(p));
  const f = (p: string, binary = false): StaticTarget => ({ kind: 'file', path: p, binary });
  if (parts[0] === 'assets') return { kind: 'asset', path: parts.slice(1).join('/') };
  if (parts[0] === 'world') {
    if (parts.length === 2 && ['meta', 'network', 'schools'].includes(parts[1]!)) return f(`world/${parts[1]}.json`);
    if (parts.length === 3 && parts[1] === 'buildings') return { kind: 'building', id: parts[2]! };
    return null;
  }
  if (parts[0] === 'tools' && parts.length === 1) return f('tools.json');
  if (parts[0] === 'health' && parts.length === 1) return f('health.json');
  if (parts[0] === 'baseline') {
    if (parts.length === 1) return f('baseline.json');
    if (parts.length === 2 && parts[1] === 'playback') return f('baseline/playback.bin', true);
    return null;
  }
  if (parts[0] === 'plans') {
    if (parts.length === 1) {
      // offset > 0 pages are not snapshotted (the index holds every snapshotted plan)
      if (Number(query.get('offset') ?? 0) > 0) return null;
      return f(`plans/index-${safeId(query.get('sort') || 'new')}.json`);
    }
    const id = safeId(parts[1]!);
    if (parts.length === 2) return f(`plans/${id}.json`);
    if (parts.length === 3 && parts[2] === 'playback') return f(`plans/${id}/playback.bin`, true);
    if (parts.length === 3 && parts[2] === 'residents') return f(`plans/${id}/residents.json`);
    return null;
  }
  return null;
}

export const STATIC_NOTICE =
  'This is the static viewer: you can explore the world, traffic and saved plans, and sketch a plan, but checking costs, saving and running plans need the Python API (run it locally, or build the viewer with VITE_API_BASE pointing at a hosted API).';
