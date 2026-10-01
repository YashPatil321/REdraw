#!/usr/bin/env node
/**
 * Dev fixture: turn our open-data world (client/public/assets) into an
 * Earth-centered 3D Tiles 1.0 tileset, exactly like Google's photoreal tiles
 * are delivered (ECEF, WGS84 ellipsoid heights, glTF y-up content). Every
 * vertex goes scene -> lat/lon (UTM inverse) -> ECEF with a fake geoid
 * separation of -35 m, so loading it with `?tiles=/dev-tiles/tileset.json`
 * exercises the whole photoreal path (placement, warp, vertical calibration,
 * draping, walk mode, picking) without a Google key. Not committed.
 *
 *   node scripts/make-dev-tileset.mjs [--only-hero] [--out public/dev-tiles]
 */

import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { sceneToLatLon, originFromLatLon } from '../src/geo.ts';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const args = process.argv.slice(2);
const outDir = resolve(root, args.includes('--out') ? args[args.indexOf('--out') + 1] : 'public/dev-tiles');
const assets = join(root, 'public/assets');
const GEOID_N = -35.0;

const A = 6378137.0;
const F = 1 / 298.257223563;
const E2 = F * (2 - F);
const DEG = Math.PI / 180;
function ecef(lat, lon, h) {
  const p = lat * DEG;
  const l = lon * DEG;
  const s = Math.sin(p);
  const N = A / Math.sqrt(1 - E2 * s * s);
  return [(N + h) * Math.cos(p) * Math.cos(l), (N + h) * Math.cos(p) * Math.sin(l), (N * (1 - E2) + h) * s];
}
function enu(lat, lon) {
  const p = lat * DEG;
  const l = lon * DEG;
  return {
    e: [-Math.sin(l), Math.cos(l), 0],
    n: [-Math.sin(p) * Math.cos(l), -Math.sin(p) * Math.sin(l), Math.cos(p)],
    u: [Math.cos(p) * Math.cos(l), Math.cos(p) * Math.sin(l), Math.sin(p)],
  };
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function readGlb(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a glb');
  const jl = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(buf.subarray(20, 20 + jl)));
  const bo = 20 + jl;
  if (bo + 8 > buf.byteLength) return { json, bin: new Uint8Array(0) };
  const bl = dv.getUint32(bo, true);
  const bin = new Uint8Array(buf.subarray(bo + 8, bo + 8 + bl));
  return { json, bin };
}

function writeGlb(json, bin) {
  let jb = new TextEncoder().encode(JSON.stringify(json));
  const jp = (4 - (jb.length % 4)) % 4;
  if (jp) {
    const t = new Uint8Array(jb.length + jp).fill(0x20);
    t.set(jb);
    jb = t;
  }
  const bp = (4 - (bin.length % 4)) % 4;
  const total = 12 + 8 + jb.length + 8 + bin.length + bp;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jb.length, true);
  dv.setUint32(16, 0x4e4f534a, true);
  out.set(jb, 20);
  const o = 20 + jb.length;
  dv.setUint32(o, bin.length + bp, true);
  dv.setUint32(o + 4, 0x004e4942, true);
  out.set(bin, o + 8);
  return out;
}

const meta = JSON.parse(readFileSync(join(root, '../data/processed/region_meta.json'), 'utf8'));
const origin = originFromLatLon(meta.origin.lat, meta.origin.lon);

/** Re-express one glb in a local ENU frame at its center; returns tileset child. */
function convert(rel, name) {
  const src = join(assets, rel);
  if (!existsSync(src)) return null;
  const { json, bin } = readGlb(readFileSync(src));
  if (json.extensionsUsed?.includes('KHR_draco_mesh_compression')) {
    console.warn(`skip ${rel}: draco compressed (rebuild assets without draco for the dev tileset)`);
    return null;
  }
  const posAcc = new Set();
  for (const m of json.meshes ?? []) for (const p of m.primitives) if (p.attributes.POSITION !== undefined) posAcc.add(p.attributes.POSITION);
  if (!posAcc.size || !bin.length) return null;
  // center of this file in scene coordinates
  let cx = 0, cz = 0, n = 0;
  for (const ai of posAcc) {
    const a = json.accessors[ai];
    if (!a.min) continue;
    cx += (a.min[0] + a.max[0]) / 2;
    cz += (a.min[2] + a.max[2]) / 2;
    n++;
  }
  cx /= n || 1;
  cz /= n || 1;
  const cll = sceneToLatLon(cx, cz, origin);
  const O = ecef(cll.lat, cll.lon, 0);
  const F0 = enu(cll.lat, cll.lon);
  let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180, minH = 1e9, maxH = -1e9;
  const view = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  for (const ai of posAcc) {
    const a = json.accessors[ai];
    const bv = json.bufferViews[a.bufferView];
    const stride = bv.byteStride ?? 12;
    const base = (bv.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const mn = [Infinity, Infinity, Infinity];
    const mx = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < a.count; i++) {
      const o = base + i * stride;
      const x = view.getFloat32(o, true);
      const y = view.getFloat32(o + 4, true);
      const z = view.getFloat32(o + 8, true);
      const ll = sceneToLatLon(x, z, origin);
      const h = y + GEOID_N;
      const P = ecef(ll.lat, ll.lon, h);
      const d = [P[0] - O[0], P[1] - O[1], P[2] - O[2]];
      const e = dot(F0.e, d);
      const nn = dot(F0.n, d);
      const u = dot(F0.u, d);
      // glTF y-up: (east, up, south)
      const q = [e, u, -nn];
      for (let k = 0; k < 3; k++) {
        view.setFloat32(o + 4 * k, q[k], true);
        mn[k] = Math.min(mn[k], q[k]);
        mx[k] = Math.max(mx[k], q[k]);
      }
      minLat = Math.min(minLat, ll.lat); maxLat = Math.max(maxLat, ll.lat);
      minLon = Math.min(minLon, ll.lon); maxLon = Math.max(maxLon, ll.lon);
      minH = Math.min(minH, h); maxH = Math.max(maxH, h);
    }
    a.min = mn;
    a.max = mx;
  }
  writeFileSync(join(outDir, `${name}.glb`), writeGlb(json, bin));
  const transform = [...F0.e, 0, ...F0.n, 0, ...F0.u, 0, ...O, 1];
  return {
    transform,
    boundingVolume: { region: [minLon * DEG, minLat * DEG, maxLon * DEG, maxLat * DEG, minH - 5, maxH + 5] },
    geometricError: 0,
    content: { uri: `${name}.glb` },
    _region: [minLon, minLat, maxLon, maxLat, minH, maxH],
  };
}

mkdirSync(outDir, { recursive: true });
const manifest = JSON.parse(readFileSync(join(assets, 'manifest.json'), 'utf8'));
const children = [];
for (const t of manifest.tiles) {
  for (const kind of ['terrain', 'buildings']) {
    if (!t[kind]) continue;
    const c = convert(t[kind], `${kind}_${t.id}`);
    if (c) children.push(c);
  }
}
(manifest.roads ?? []).forEach((r, i) => {
  const c = convert(r, `roads_${i}`);
  if (c) children.push(c);
});
let reg = [180, 90, -180, -90, 1e9, -1e9];
for (const c of children) {
  const r = c._region;
  reg = [Math.min(reg[0], r[0]), Math.min(reg[1], r[1]), Math.max(reg[2], r[2]), Math.max(reg[3], r[3]), Math.min(reg[4], r[4]), Math.max(reg[5], r[5])];
  delete c._region;
}
const tileset = {
  asset: { version: '1.0', gltfUpAxis: 'Y', copyright: 'Redraw dev fixture (open data re-projected to ECEF)' },
  geometricError: 4000,
  root: {
    boundingVolume: { region: [reg[0] * DEG, reg[1] * DEG, reg[2] * DEG, reg[3] * DEG, reg[4] - 5, reg[5] + 5] },
    geometricError: 2000,
    refine: 'ADD',
    children,
  },
};
writeFileSync(join(outDir, 'tileset.json'), JSON.stringify(tileset));
console.log(`wrote ${children.length} tiles to ${outDir} (open with ?tiles=/dev-tiles/tileset.json)`);
