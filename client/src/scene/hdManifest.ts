/**
 * Parsing of the Blender HD buildings manifest
 * (`assets/buildings_hd/manifest_buildings.json`): per tile, glb paths by
 * LOD level (0 = full detail near the camera, 1 = simplified). The layout is
 * read defensively (array or map of tiles; `lods` arrays, `lod0` / `lod1`
 * keys or `files`), since the Blender builder owns it. Pure; unit tested.
 */

export interface HdBuildingTile {
  id: string;
  /** asset-relative glb path per LOD level (index = level); null where missing */
  lods: Array<string | null>;
  bounds?: { min_x: number; max_x: number; min_z: number; max_z: number; min_y?: number; max_y?: number };
  buildings?: number;
}

export interface HdBuildingsManifest {
  tiles: HdBuildingTile[];
  /** suggested LOD0 -> LOD1 switch distance (m), if the manifest gives one */
  lod0Distance: number | null;
}

const GLB = /\.glb$/i;

function str(v: unknown): string | null {
  if (typeof v === 'string' && GLB.test(v)) return v;
  if (v && typeof v === 'object') {
    const o = v as Record<string, unknown>;
    for (const k of ['file', 'path', 'glb', 'uri', 'url']) if (typeof o[k] === 'string' && GLB.test(o[k] as string)) return o[k] as string;
  }
  return null;
}

function level(v: unknown, fallback: number): number {
  if (v && typeof v === 'object') {
    const o = v as Record<string, unknown>;
    for (const k of ['level', 'lod']) if (typeof o[k] === 'number') return o[k] as number;
  }
  return fallback;
}

/** Resolve a manifest path to an asset-relative one (under `assets/`). */
export function resolveHdPath(p: string, base: string): string {
  let s = p.replace(/^\.?\/+/, '');
  s = s.replace(/^(.*\/)?assets\//, '');
  if (s.startsWith('buildings_hd/')) return s;
  const b = base.replace(/^\.?\/+/, '').replace(/^(.*\/)?assets\//, '');
  return (b ? b.replace(/\/?$/, '/') : 'buildings_hd/') + s;
}

export function parseHdBuildingsManifest(raw: unknown, base = 'buildings_hd/'): HdBuildingsManifest {
  const out: HdBuildingsManifest = { tiles: [], lod0Distance: null };
  if (!raw || typeof raw !== 'object') return out;
  const o = raw as Record<string, unknown>;
  const basePath = typeof o['base_path'] === 'string' ? (o['base_path'] as string) : base;
  const lodInfo = (o['lods'] ?? o['lod']) as unknown;
  const pickDist = (x: unknown): number | null => {
    if (!x || typeof x !== 'object') return null;
    const r = x as Record<string, unknown>;
    for (const k of ['lod0_max_distance_m', 'switch_distance_m', 'max_distance_m', 'lod0_distance_m']) {
      const v = r[k];
      if (typeof v === 'number' && v > 0) return v;
      if (v && typeof v === 'object' && typeof (v as Record<string, unknown>)['0'] === 'number') return (v as Record<string, number>)['0']!;
    }
    return null;
  };
  out.lod0Distance = pickDist(o) ?? (Array.isArray(lodInfo) ? pickDist(lodInfo[0]) : pickDist(lodInfo));
  let tiles: Array<[string | undefined, unknown]> = [];
  const t = o['tiles'];
  if (Array.isArray(t)) tiles = t.map((x) => [undefined, x]);
  else if (t && typeof t === 'object') tiles = Object.entries(t as Record<string, unknown>);
  for (const [key, v] of tiles) {
    if (!v || typeof v !== 'object') continue;
    const e = v as Record<string, unknown>;
    const id = (typeof e['id'] === 'string' ? e['id'] : typeof e['tile'] === 'string' ? e['tile'] : key) as string | undefined;
    if (!id) continue;
    const lods: Array<string | null> = [];
    const put = (lv: number, p: string | null): void => {
      if (!p || lv < 0 || lv > 4) return;
      while (lods.length <= lv) lods.push(null);
      lods[lv] = resolveHdPath(p, basePath);
    };
    const arr = e['lods'] ?? e['levels'];
    if (Array.isArray(arr)) arr.forEach((x, i) => put(level(x, i), str(x)));
    else if (arr && typeof arr === 'object') for (const [k, x] of Object.entries(arr as Record<string, unknown>)) put(Number(k.replace(/\D/g, '')) || 0, str(x));
    const files = e['files'];
    const scan = (src: Record<string, unknown>): void => {
      for (const [k, x] of Object.entries(src)) {
        const m = /^lod_?(\d)$/i.exec(k) ?? /^lod_?(\d)_?(file|path|glb)$/i.exec(k);
        if (m) put(Number(m[1]), str(x));
      }
    };
    scan(e);
    if (files && typeof files === 'object' && !Array.isArray(files)) scan(files as Record<string, unknown>);
    if (!lods.length) {
      const single = str(e) ?? str(e['glb']) ?? str(e['file']);
      if (single) put(/lod1/i.test(single) ? 1 : 0, single);
    }
    if (!lods.some((x) => x)) continue;
    const b = e['bounds'] as HdBuildingTile['bounds'] | undefined;
    out.tiles.push({ id, lods, bounds: b && typeof b.min_x === 'number' ? b : undefined, buildings: typeof e['buildings'] === 'number' ? (e['buildings'] as number) : typeof e['count'] === 'number' ? (e['count'] as number) : undefined });
  }
  return out;
}
