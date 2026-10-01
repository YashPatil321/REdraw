#!/usr/bin/env node
/**
 * Snapshot a running Redraw API into the static viewer build (dist-viewer/),
 * so the viewer can be hosted on Vercel without Python:
 *
 *   dist-viewer/static-api/...   JSON / binary API responses (layout: src/staticPaths.ts)
 *   dist-viewer/assets/...       world assets (manifest, terrain/building/road tiles, props)
 *
 * Usage:
 *   node scripts/snapshot-api.mjs [--api http://localhost:8000] [--out dist-viewer] [--plans 12]
 * The API URL can also come from REDRAW_API_URL.
 */

import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BUILDINGS_FILE, STATIC_DIR, safeId } from '../src/staticPaths.ts';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const argv = process.argv.slice(2);
const arg = (name, def) => (argv.includes(name) ? argv[argv.indexOf(name) + 1] : def);
const API = (arg('--api', process.env.REDRAW_API_URL ?? 'http://localhost:8000')).replace(/\/+$/, '');
const OUT = resolve(root, arg('--out', 'dist-viewer'));
const MAX_PLANS = Number(arg('--plans', '12'));
const SA = join(OUT, STATIC_DIR);

if (!existsSync(join(OUT, 'index.html'))) {
  console.error(`${OUT}/index.html not found: run "vite build --mode viewer" first (npm run build:viewer does both).`);
  process.exit(1);
}

let bytes = 0;
function save(rel, data, base = SA) {
  const p = join(base, rel);
  mkdirSync(dirname(p), { recursive: true });
  const buf = typeof data === 'string' ? Buffer.from(data) : Buffer.from(data);
  writeFileSync(p, buf);
  bytes += buf.length;
}

async function get(path, { binary = false, optional = false, retries = 30 } = {}) {
  for (let i = 0; ; i++) {
    let res;
    try {
      res = await fetch(`${API}${path}`);
    } catch (e) {
      if (i === 0) console.error(`Cannot reach the API at ${API} (${e.message}). Start it with: .venv/bin/uvicorn api.main:app --port 8000`);
      throw e;
    }
    if (res.status === 503 && i < retries) {
      // world loading / baseline warming up
      await new Promise((r) => setTimeout(r, 2000));
      continue;
    }
    if (!res.ok) {
      if (optional) return null;
      throw new Error(`GET ${path} -> ${res.status} ${await res.text()}`);
    }
    return binary ? new Uint8Array(await res.arrayBuffer()) : res.json();
  }
}

async function pool(items, n, fn) {
  const q = [...items];
  await Promise.all(Array.from({ length: n }, async () => {
    while (q.length) await fn(q.shift());
  }));
}

const t0 = Date.now();
console.log(`snapshotting ${API} -> ${OUT}`);

// ---- world, tools, baseline
const meta = await get('/world/meta');
save('world/meta.json', JSON.stringify(meta));
save('world/network.json', JSON.stringify(await get('/world/network')));
const schools = await get('/world/schools');
save('world/schools.json', JSON.stringify(schools));
save('tools.json', JSON.stringify(await get('/tools')));
save('baseline.json', JSON.stringify(await get('/baseline')));
save('baseline/playback.bin', await get('/baseline/playback', { binary: true }));
const health = await get('/health', { optional: true });
if (health) save('health.json', JSON.stringify({ ...health, static_snapshot: new Date().toISOString() }));

// ---- buildings: every id, one compact file (fields once, a row per building)
const FIELDS = ['type', 'name', 'address', 'height_m', 'levels', 'area_m2', 'households', 'block', 'school_id', 'centroid_x', 'centroid_z', 'base_elev_m'];
const rows = {};
const nBuild = meta.counts?.buildings ?? 0;
const maxId = nBuild + 2000;
const round = (v) => (typeof v === 'number' ? Math.round(v * 10) / 10 : v);
const ids = Array.from({ length: maxId }, (_, i) => i + 1);
await pool(ids, 24, async (id) => {
  const b = await get(`/world/buildings/${id}`, { optional: true });
  if (!b) return;
  rows[id] = FIELDS.map((f) => (b[f] === undefined ? null : round(b[f])));
});
// trim trailing nulls per row
for (const k of Object.keys(rows)) {
  const r = rows[k];
  while (r.length && r[r.length - 1] === null) r.pop();
}
save(BUILDINGS_FILE, JSON.stringify({ fields: FIELDS, rows }));
console.log(`buildings: ${Object.keys(rows).length} (meta says ${nBuild})`);

// ---- plans: done plans with reports, playback and residents
const sorts = ['new', 'votes', ...(meta.metrics ?? []).map((m) => m.id)];
const first = await get(`/plans?mission=${encodeURIComponent(meta.mission?.id ?? '')}&sort=votes&limit=200`);
const chosen = (first.plans ?? []).filter((p) => p.status === 'done').slice(0, MAX_PLANS);
const keep = new Set(chosen.map((p) => p.id));
for (const p of chosen) {
  const id = safeId(p.id);
  save(`plans/${id}.json`, JSON.stringify({ ...(await get(`/plans/${encodeURIComponent(p.id)}`)), is_mine: false, my_vote: 0 }));
  const pb = await get(`/plans/${encodeURIComponent(p.id)}/playback`, { binary: true, optional: true });
  if (pb) save(`plans/${id}/playback.bin`, pb);
  const res = await get(`/plans/${encodeURIComponent(p.id)}/residents`, { optional: true });
  if (res) save(`plans/${id}/residents.json`, JSON.stringify(res));
}
for (const sort of sorts) {
  const r = await get(`/plans?mission=${encodeURIComponent(meta.mission?.id ?? '')}&sort=${encodeURIComponent(sort)}&limit=200`, { optional: true });
  if (!r) continue;
  const plans = (r.plans ?? []).filter((p) => keep.has(p.id));
  save(`plans/index-${safeId(sort)}.json`, JSON.stringify({ plans, total: plans.length }));
}
console.log(`plans: ${chosen.length}`);

// ---- assets through the API (/assets/...): manifest, everything it references, props
const assetsOut = join(OUT, 'assets');
const assetBase = (meta.assets?.base_url ?? '/assets/').replace(/\/?$/, '/');
async function asset(rel, optional = false) {
  const data = await get(`${assetBase}${rel}`, { binary: true, optional });
  if (data) save(rel, data, assetsOut);
  return data;
}
const manifestRel = (meta.assets?.manifest ?? '/assets/manifest.json').replace(assetBase, '').replace(/^\/+/, '');
const manifestBytes = await asset(manifestRel);
const manifest = JSON.parse(Buffer.from(manifestBytes).toString('utf8'));
const files = new Set();
for (const t of manifest.tiles ?? []) for (const k of ['terrain', 'buildings', 'roads']) if (t[k]) files.add(t[k]);
for (const r of manifest.roads ?? []) files.add(r);
if (manifest.terrain_meta) files.add(manifest.terrain_meta);
// anything else the manifest names as a relative file path
JSON.stringify(manifest, (k, v) => {
  if (typeof v === 'string' && /\.(glb|json|png|jpg|jpeg|webp|ktx2|bin)$/i.test(v) && !/^https?:/.test(v)) files.add(v.replace(/^\/+/, ''));
  return v;
});
await pool([...files], 6, (f) => asset(f, true));
// props library (optional: written by blender/)
const propsManifest = await asset('props/props_manifest.json', true);
if (propsManifest) {
  const pm = JSON.parse(Buffer.from(propsManifest).toString('utf8'));
  const pf = new Set();
  JSON.stringify(pm, (k, v) => {
    if (typeof v === 'string' && /\.(glb|json|png|jpg|jpeg|webp|ktx2|bin)$/i.test(v) && !/^https?:/.test(v)) pf.add(v.replace(/^\/+/, ''));
    return v;
  });
  await pool([...pf], 6, (f) => asset(f.startsWith('props/') ? f : `props/${f}`, true));
  for (const f of ['props/placements.json', 'props/placements.bin']) await asset(f, true);
}

const mb = (bytes / 1e6).toFixed(1);
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(0)} s, ${mb} MB written`);
console.log(`assets: ${files.size} files (+ props) in ${join(OUT, 'assets')}`);

// vercel.json for `vercel deploy dist-viewer`: routing and cache headers only (no build step)
const vercel = JSON.parse(readFileSync(join(root, 'vercel.json'), 'utf8'));
delete vercel.buildCommand;
delete vercel.outputDirectory;
delete vercel.installCommand;
delete vercel.framework;
writeFileSync(join(OUT, 'vercel.json'), JSON.stringify(vercel, null, 2));
