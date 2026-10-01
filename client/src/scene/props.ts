/**
 * Prop library from blender/ (served under /assets/props/): vehicle models for
 * the traffic layer, and instanced trees / street lamps from placements.
 *
 * Contract (blender/rdlib/materials.py, vehicles.py):
 * - one glb per prop; vehicles face -Z (north) with +Y up, origin on the ground
 * - vertex attributes `_LIGHT` (0 body, 1 head, 2 tail, 3 glass, 4 dark trim)
 *   and `_TINT` (1 where the client multiplies a per-instance paint color)
 * - props_manifest.json lists the props (id + glb path + metadata such as
 *   fleet `share`) and `paint_colors` (sRGB hex)
 * The manifest / placement layout is read defensively: anything not
 * understood is skipped with a console warning, never faked.
 */

import * as THREE from 'three';
import { DRACOLoader, DRACO_GLTF_CONFIG } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { VehicleGeometries } from '../traffic/layer';

export type AssetFetcher = (rel: string) => Promise<ArrayBuffer>;

export interface PropEntry {
  id: string;
  file: string;
  kind?: string;
  share?: number;
  [k: string]: unknown;
}

export interface PropsManifestInfo {
  entries: PropEntry[];
  paint: Array<[number, number, number]>;
  placements: string | null;
  raw: Record<string, unknown>;
}

const GLB_RE = /\.glb$/i;

/** Find prop entries (objects with an id and a .glb path) anywhere in the manifest. Pure; unit tested. */
export function parsePropsManifest(raw: Record<string, unknown> | unknown[]): PropsManifestInfo {
  const entries: PropEntry[] = [];
  const seen = new Set<string>();
  const visit = (v: unknown, keyHint?: string): void => {
    if (!v || typeof v !== 'object') return;
    if (Array.isArray(v)) {
      v.forEach((x) => visit(x));
      return;
    }
    const o = v as Record<string, unknown>;
    const file = ['file', 'glb', 'uri', 'path', 'mesh', 'url'].map((k) => o[k]).find((x) => typeof x === 'string' && GLB_RE.test(x)) as string | undefined;
    const id = (typeof o['id'] === 'string' ? o['id'] : keyHint) as string | undefined;
    if (file && id && !seen.has(id)) {
      seen.add(id);
      entries.push({ ...o, id, file: file.replace(/^\/+/, '') } as PropEntry);
      return;
    }
    for (const [k, x] of Object.entries(o)) visit(x, k);
  };
  const obj = (Array.isArray(raw) ? {} : raw) as Record<string, unknown>;
  visit(Array.isArray(raw) ? raw : (obj['props'] ?? obj));
  for (const e of entries) if (typeof e.share !== 'number' && typeof e['fleet_share'] === 'number') e.share = e['fleet_share'] as number;
  // paint palette: top level, or the first tintable vehicle entry's (blender writes it per vehicle)
  const paintHex = (obj['paint_colors'] ??
    (obj['vehicles'] as Record<string, unknown> | undefined)?.['paint_colors'] ??
    entries.find((e) => Array.isArray(e['paint_colors']))?.['paint_colors']) as unknown;
  const paint: Array<[number, number, number]> = [];
  if (Array.isArray(paintHex)) {
    for (const p of paintHex) {
      const hex = typeof p === 'string' ? p : typeof p === 'object' && p && typeof (p as { hex?: unknown }).hex === 'string' ? (p as { hex: string }).hex : null;
      if (!hex) continue;
      const c = new THREE.Color(hex); // sRGB hex -> linear working color space
      // weight by share when given ({hex, share})
      const share = typeof p === 'object' && p && typeof (p as { share?: unknown }).share === 'number' ? (p as { share: number }).share : 1;
      for (let k = 0; k < Math.max(1, Math.round(share * 20)); k++) paint.push([c.r, c.g, c.b]);
    }
  }
  const placements = typeof obj['placements'] === 'string' ? (obj['placements'] as string) : null;
  return { entries, paint, placements, raw: obj };
}

let loader: GLTFLoader | null = null;
function gltfLoader(): GLTFLoader {
  if (!loader) {
    loader = new GLTFLoader();
    const d = new DRACOLoader();
    d.setDecoderPath(DRACO_GLTF_CONFIG);
    loader.setDRACOLoader(d);
  }
  return loader;
}

/**
 * Merge a prop glb into one geometry: positions / normals in prop space,
 * COLOR_0 baked from material base color (x vertex color), aLight / aTint from
 * `_LIGHT` / `_TINT` (or guessed from material names). `forwardToX` rotates a
 * -Z facing vehicle to face +X (the traffic layer's convention).
 */
export async function loadPropGeometry(buf: ArrayBuffer, forwardToX: boolean): Promise<THREE.BufferGeometry | null> {
  const gltf = await gltfLoader().parseAsync(buf, '');
  const parts: THREE.BufferGeometry[] = [];
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    const mats = Array.isArray(m.material) ? m.material : [m.material];
    const src = m.geometry.index ? m.geometry.toNonIndexed() : m.geometry.clone();
    src.applyMatrix4(m.matrixWorld);
    const n = src.getAttribute('position').count;
    if (!src.getAttribute('normal')) src.computeVertexNormals();
    // material per vertex (groups) -> base color
    const color = new Float32Array(n * 3).fill(1);
    const light = new Float32Array(n);
    const tint = new Float32Array(n);
    const groups = src.groups.length ? src.groups : [{ start: 0, count: n, materialIndex: 0 }];
    const vc = src.getAttribute('color');
    const la = src.getAttribute('_light');
    const ta = src.getAttribute('_tint');
    for (const g of groups) {
      const mat = mats[g.materialIndex ?? 0] as THREE.MeshStandardMaterial | undefined;
      const c = mat?.color ?? new THREE.Color(1, 1, 1);
      const name = (mat?.name ?? '').toLowerCase();
      const guessLight = /head|drl|lamp_lens/.test(name) ? 1 : /tail|bus_red/.test(name) ? 2 : /glass/.test(name) ? 3 : /tire|trim|underbody|sign_black|bed_liner|rim_dark/.test(name) ? 4 : 0;
      const guessTint = /^paint(_white)?$/.test(name) ? 1 : 0;
      for (let i = g.start; i < Math.min(n, g.start + g.count); i++) {
        color[i * 3] = c.r * (vc ? vc.getX(i) : 1);
        color[i * 3 + 1] = c.g * (vc ? vc.getY(i) : 1);
        color[i * 3 + 2] = c.b * (vc ? vc.getZ(i) : 1);
        light[i] = la ? Math.round(la.getX(i)) : guessLight;
        tint[i] = ta ? ta.getX(i) : guessTint;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', src.getAttribute('position'));
    g.setAttribute('normal', src.getAttribute('normal'));
    g.setAttribute('color', new THREE.BufferAttribute(color, 3));
    g.setAttribute('aLight', new THREE.BufferAttribute(light, 1));
    g.setAttribute('aTint', new THREE.BufferAttribute(tint, 1));
    parts.push(g);
  });
  if (!parts.length) return null;
  const merged = mergeGeometries(parts, false);
  parts.forEach((p) => p.dispose());
  if (!merged) return null;
  if (forwardToX) merged.rotateY(-Math.PI / 2);
  merged.computeBoundingSphere();
  return merged;
}

const CAR_IDS = /^car[_-]|^(sedan|suv|minivan|pickup|crossover)/i;

/** Vehicle models for the traffic layer, or null when the prop library is absent. */
export async function loadVehicleProps(fetchAsset: AssetFetcher): Promise<VehicleGeometries | null> {
  let raw: Record<string, unknown> | unknown[];
  try {
    raw = JSON.parse(new TextDecoder().decode(await fetchAsset('props/props_manifest.json'))) as Record<string, unknown> | unknown[];
  } catch {
    return null; // no props built yet: procedural low-poly vehicles
  }
  const info = parsePropsManifest(raw);
  const out: VehicleGeometries = { cars: [], paint: info.paint.length ? info.paint : undefined };
  const load = async (e: PropEntry): Promise<THREE.BufferGeometry | null> => {
    try {
      const rel = e.file.startsWith('props/') ? e.file : `props/${e.file}`;
      return await loadPropGeometry(await fetchAsset(rel), true);
    } catch (err) {
      console.warn(`prop ${e.id} failed to load`, err);
      return null;
    }
  };
  await Promise.all(
    info.entries.map(async (e) => {
      const id = e.id.toLowerCase();
      const cls = String(e['vehicle_class'] ?? '');
      if (e['kind'] && e['kind'] !== 'vehicle') return;
      if (cls === 'car' || (!cls && CAR_IDS.test(id))) {
        const g = await load(e);
        if (g) out.cars!.push({ id: e.id, geometry: g, share: typeof e.share === 'number' && e.share > 0 ? e.share : 0.1 });
      } else if (cls === 'bus' || /school_?bus/.test(id)) {
        out.bus = (await load(e)) ?? undefined;
      } else if (cls === 'shuttle' || /shuttle/.test(id)) {
        out.shuttle = (await load(e)) ?? undefined;
      }
    }),
  );
  out.cars!.sort((a, b) => a.id.localeCompare(b.id));
  if (!out.cars!.length && !out.bus && !out.shuttle) return null;
  return out;
}

// ---------------------------------------------------------------------------
// static props: trees, shrubs, street lamps from placements

export interface Placement {
  x: number;
  y: number;
  z: number;
  rot: number;
  scale: number;
}

/**
 * Parse placements.json + placements.bin. Accepts per-prop sections
 * (`{props|sections: [{id|prop, offset, count, stride?, fields?}]}`) or one
 * interleaved table (`{fields, count, props: [ids]}` with a prop index field).
 * Float32 little-endian. Returns prop id -> placements. Pure; unit tested.
 */
export function parsePlacements(meta: Record<string, unknown>, bin: ArrayBuffer): Map<string, Placement[]> {
  const out = new Map<string, Placement[]>();
  const dv = new DataView(bin);
  const defFields = (meta['fields'] as string[] | undefined) ?? ['x', 'y', 'z', 'rot_y', 'scale'];
  const fieldIndex = (fields: string[], names: string[]): number => fields.findIndex((f) => names.includes(f.toLowerCase()));
  const readRows = (offset: number, count: number, fields: string[], stride: number, propIdx: number, ids: string[] | null, fixedId: string | null): void => {
    const ix = fieldIndex(fields, ['x']);
    const iy = fieldIndex(fields, ['y']);
    const iz = fieldIndex(fields, ['z']);
    const ir = fieldIndex(fields, ['rot_y', 'rot', 'yaw', 'heading', 'ry']);
    const is = fieldIndex(fields, ['scale', 's']);
    if (ix < 0 || iz < 0) return;
    for (let r = 0; r < count; r++) {
      const o = offset + r * stride;
      if (o + fields.length * 4 > bin.byteLength) break;
      const f = (k: number): number => (k >= 0 ? dv.getFloat32(o + k * 4, true) : 0);
      const id = fixedId ?? (ids && propIdx >= 0 ? ids[Math.round(f(propIdx))] : undefined);
      if (!id) continue;
      let arr = out.get(id);
      if (!arr) out.set(id, (arr = []));
      arr.push({ x: f(ix), y: iy >= 0 ? f(iy) : NaN, z: f(iz), rot: f(ir), scale: is >= 0 ? f(is) || 1 : 1 });
    }
  };
  const sections = (meta['sections'] ?? (Array.isArray(meta['props']) && (meta['props'] as unknown[]).some((p) => typeof p === 'object') ? meta['props'] : null)) as
    | Array<Record<string, unknown>>
    | null;
  if (sections) {
    for (const sct of sections) {
      const id = (sct['id'] ?? sct['prop']) as string | undefined;
      const fields = (sct['fields'] as string[] | undefined) ?? defFields;
      const stride = Number(sct['stride'] ?? fields.length * 4);
      if (!id || typeof sct['count'] !== 'number') continue;
      readRows(Number(sct['offset'] ?? 0), sct['count'], fields, stride, -1, null, id);
    }
    return out;
  }
  const ids = (meta['props'] ?? meta['prop_ids'] ?? meta['ids']) as string[] | undefined;
  const fields = defFields;
  const count = Number(meta['count'] ?? Math.floor(bin.byteLength / (fields.length * 4)));
  const stride = Number(meta['stride'] ?? fields.length * 4);
  readRows(Number(meta['offset'] ?? 0), count, fields, stride, fieldIndex(fields, ['prop', 'prop_idx', 'kind', 'id', 'type']), ids ?? null, null);
  return out;
}

const CELL = 200;

/** Instanced static props with distance culling on a coarse grid. */
export class StaticProps {
  readonly group = new THREE.Group();
  private kinds: Array<{ mesh: THREE.InstancedMesh; cells: Map<number, Placement[]>; cap: number }> = [];
  private lastPos = new THREE.Vector3(Infinity, 0, 0);
  private drawDistance = 1500;
  private mat4 = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private up = new THREE.Vector3(0, 1, 0);
  private sv = new THREE.Vector3();
  private pv = new THREE.Vector3();

  constructor() {
    this.group.name = 'static-props';
  }

  add(id: string, geometry: THREE.BufferGeometry, material: THREE.Material, list: Placement[], heightAt: (x: number, z: number) => number | null): void {
    const cells = new Map<number, Placement[]>();
    for (const p of list) {
      if (!Number.isFinite(p.y)) p.y = heightAt(p.x, p.z) ?? 0;
      const k = (Math.floor(p.x / CELL) + 32768) * 65536 + (Math.floor(p.z / CELL) + 32768);
      let a = cells.get(k);
      if (!a) cells.set(k, (a = []));
      a.push(p);
    }
    const cap = Math.min(list.length, 20000);
    const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, cap));
    mesh.name = `prop:${id}`;
    mesh.count = 0;
    mesh.frustumCulled = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.group.add(mesh);
    this.kinds.push({ mesh, cells, cap });
    this.lastPos.set(Infinity, 0, 0);
  }

  setDrawDistance(d: number): void {
    this.drawDistance = d;
    this.lastPos.set(Infinity, 0, 0);
  }

  setShadows(on: boolean): void {
    for (const k of this.kinds) k.mesh.castShadow = on;
  }

  /** Refill visible instances when the camera moved enough. */
  update(cam: THREE.Vector3): void {
    if (cam.distanceTo(this.lastPos) < Math.max(25, this.drawDistance * 0.05)) return;
    this.lastPos.copy(cam);
    const D = this.drawDistance;
    // from high above, trees are sub-pixel: cap by height too
    if (cam.y - 0 > D * 3) {
      for (const k of this.kinds) k.mesh.count = 0;
      return;
    }
    const r = Math.ceil(D / CELL);
    const cx = Math.floor(cam.x / CELL);
    const cz = Math.floor(cam.z / CELL);
    for (const k of this.kinds) {
      let n = 0;
      for (let ix = cx - r; ix <= cx + r && n < k.cap; ix++) {
        for (let iz = cz - r; iz <= cz + r && n < k.cap; iz++) {
          const arr = k.cells.get((ix + 32768) * 65536 + (iz + 32768));
          if (!arr) continue;
          for (const p of arr) {
            if (n >= k.cap) break;
            const d = Math.hypot(p.x - cam.x, p.z - cam.z);
            if (d > D) continue;
            this.q.setFromAxisAngle(this.up, p.rot);
            this.sv.setScalar(p.scale);
            this.pv.set(p.x, p.y, p.z);
            this.mat4.compose(this.pv, this.q, this.sv);
            k.mesh.setMatrixAt(n++, this.mat4);
          }
        }
      }
      k.mesh.count = n;
      k.mesh.instanceMatrix.needsUpdate = true;
    }
  }

  dispose(): void {
    for (const k of this.kinds) {
      k.mesh.geometry.dispose();
      (k.mesh.material as THREE.Material).dispose();
      k.mesh.dispose();
    }
    this.group.removeFromParent();
  }
}

/** Load trees / lamps from the manifest + placements; null if absent or not understood. */
export async function loadStaticProps(fetchAsset: AssetFetcher, heightAt: (x: number, z: number) => number | null): Promise<StaticProps | null> {
  let raw: Record<string, unknown> | unknown[];
  try {
    raw = JSON.parse(new TextDecoder().decode(await fetchAsset('props/props_manifest.json'))) as Record<string, unknown> | unknown[];
  } catch {
    return null;
  }
  const info = parsePropsManifest(raw);
  let meta: Record<string, unknown>;
  let bin: ArrayBuffer;
  try {
    const metaRel = info.placements ?? 'props/placements.json';
    meta = JSON.parse(new TextDecoder().decode(await fetchAsset(metaRel.startsWith('props/') ? metaRel : `props/${metaRel}`))) as Record<string, unknown>;
    const binRel = (meta['bin'] ?? meta['file'] ?? meta['data'] ?? 'placements.bin') as string;
    bin = await fetchAsset(binRel.startsWith('props/') ? binRel : `props/${binRel}`);
  } catch {
    return null;
  }
  const placements = parsePlacements(meta, bin);
  if (!placements.size) {
    console.warn('props: placements.json not understood; trees and lamps skipped', meta);
    return null;
  }
  const sp = new StaticProps();
  for (const [id, list] of placements) {
    const e = info.entries.find((x) => x.id === id);
    if (!e) continue;
    try {
      const rel = e.file.startsWith('props/') ? e.file : `props/${e.file}`;
      const gltf = await gltfLoader().parseAsync(await fetchAsset(rel), '');
      const geoms: THREE.BufferGeometry[] = [];
      let material: THREE.Material | null = null;
      gltf.scene.updateMatrixWorld(true);
      gltf.scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (!m.isMesh) return;
        const g = m.geometry.clone().applyMatrix4(m.matrixWorld);
        for (const name of Object.keys(g.attributes)) if (!['position', 'normal', 'uv', 'color'].includes(name)) g.deleteAttribute(name);
        geoms.push(g);
        material ??= Array.isArray(m.material) ? m.material[0]! : m.material;
      });
      if (!geoms.length || !material) continue;
      const merged = mergeGeometries(geoms, false);
      if (!merged) continue;
      const mat = material as THREE.MeshStandardMaterial;
      if (mat.map) mat.map.anisotropy = 4;
      sp.add(id, merged, mat, list, heightAt);
    } catch (err) {
      console.warn(`prop ${id} failed`, err);
    }
  }
  return sp;
}
