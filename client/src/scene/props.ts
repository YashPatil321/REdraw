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

export type PropClass = 'tree' | 'palm' | 'shrub' | 'grass' | 'lamp' | 'vehicle' | 'other';

export function propClass(id: string, kind?: string): PropClass {
  const k = (kind ?? '').toLowerCase();
  const i = id.toLowerCase();
  if (k === 'vehicle' || /^car_|bus|shuttle/.test(i)) return 'vehicle';
  if (k === 'lamp' || /lamp|light/.test(i)) return 'lamp';
  if (/grass/.test(i)) return 'grass';
  if (k === 'shrub' || /shrub|chaparral/.test(i)) return 'shrub';
  if (/palm/.test(i)) return 'palm';
  if (k === 'tree' || /tree|oak|eucalyptus|jacaranda/.test(i)) return 'tree';
  return 'other';
}

/** Draw distances (m) per class at draw-distance 1.0; scaled by the quality preset. */
const NEAR: Record<PropClass, number> = { tree: 170, palm: 220, shrub: 90, grass: 55, lamp: 350, vehicle: 260, other: 150 };
const FAR: Record<PropClass, number> = { tree: 1300, palm: 1300, shrub: 260, grass: 0, lamp: 0, vehicle: 0, other: 0 };

/**
 * Cheap distant stand-in for a tree: an icosahedron crown (20 triangles) over the
 * upper part of the model's bounds, plus a thin trunk for palms. Colored with the
 * average foliage color of the model's texture.
 */
export function lodGeometry(full: THREE.BufferGeometry, cls: PropClass, color: THREE.Color): THREE.BufferGeometry {
  full.computeBoundingBox();
  const b = full.boundingBox!;
  const w = Math.max(0.5, b.max.x - b.min.x);
  const d = Math.max(0.5, b.max.z - b.min.z);
  const h = Math.max(0.5, b.max.y - Math.max(0, b.min.y));
  const parts: THREE.BufferGeometry[] = [];
  const crown = new THREE.IcosahedronGeometry(0.5, 0);
  if (cls === 'palm') {
    crown.scale(w * 0.9, h * 0.22, d * 0.9).translate((b.min.x + b.max.x) / 2, b.max.y - h * 0.12, (b.min.z + b.max.z) / 2);
    const trunk = new THREE.CylinderGeometry(0.22, 0.32, h * 0.85, 5, 1, true).translate((b.min.x + b.max.x) / 2, h * 0.425, (b.min.z + b.max.z) / 2);
    parts.push(trunk.toNonIndexed());
  } else if (cls === 'shrub' || cls === 'grass') {
    crown.scale(w * 0.85, h, d * 0.85).translate((b.min.x + b.max.x) / 2, h * 0.45, (b.min.z + b.max.z) / 2);
  } else {
    crown.scale(w * 0.85, h * 0.68, d * 0.85).translate((b.min.x + b.max.x) / 2, b.max.y - h * 0.36, (b.min.z + b.max.z) / 2);
  }
  parts.push(crown.toNonIndexed());
  for (const g of parts) {
    g.deleteAttribute('uv');
    const n = g.getAttribute('position').count;
    const c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      // trunks brownish, crowns the foliage color with a little per-face variation
      const trunk = cls === 'palm' && g === parts[0];
      const k = trunk ? 1 : 0.88 + 0.24 * ((i * 0.618) % 1);
      c[i * 3] = trunk ? 0.32 : color.r * k;
      c[i * 3 + 1] = trunk ? 0.26 : color.g * k;
      c[i * 3 + 2] = trunk ? 0.2 : color.b * k;
    }
    g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  }
  const m = mergeGeometries(parts, false)!;
  m.computeVertexNormals();
  return m;
}

/** Average color of opaque texels (foliage atlas), or null if the image cannot be read. */
function averageTextureColor(tex: THREE.Texture | null | undefined): THREE.Color | null {
  const img = tex?.image as (CanvasImageSource & { width: number; height: number }) | undefined;
  if (!img || typeof document === 'undefined') return null;
  try {
    const cv = document.createElement('canvas');
    cv.width = 32;
    cv.height = 32;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0, 32, 32);
    const px = ctx.getImageData(0, 0, 32, 32).data;
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    for (let i = 0; i < px.length; i += 4) {
      if (px[i + 3]! < 128) continue;
      // foliage only: skip bark-ish / grey texels
      if (px[i + 1]! < px[i]! * 0.9) continue;
      r += px[i]!;
      g += px[i + 1]!;
      b += px[i + 2]!;
      n++;
    }
    if (!n) return null;
    return new THREE.Color().setRGB(r / n / 255, g / n / 255, b / n / 255, THREE.SRGBColorSpace);
  } catch {
    return null;
  }
}

interface KindLayer {
  mesh: THREE.InstancedMesh;
  /** true: draws the far ring [near, far); false: [0, near) */
  far: boolean;
  cap: number;
}

interface Kind {
  id: string;
  cls: PropClass;
  cells: Map<number, Placement[]>;
  layers: KindLayer[];
  paint: Array<[number, number, number]> | null;
}

/** Instanced static props with two LODs and distance budgets on a coarse grid. */
export class StaticProps {
  readonly group = new THREE.Group();
  private kinds: Kind[] = [];
  private lastPos = new THREE.Vector3(Infinity, 0, 0);
  private scaleDist = 1;
  private mat4 = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private up = new THREE.Vector3(0, 1, 0);
  private sv = new THREE.Vector3();
  private pv = new THREE.Vector3();
  private color = new THREE.Color();
  private shadows = true;
  /** triangles drawn after the last refill (budget checks / stats) */
  triangles = 0;

  constructor() {
    this.group.name = 'static-props';
  }

  add(
    id: string,
    cls: PropClass,
    full: { geometry: THREE.BufferGeometry; material: THREE.Material },
    lod: { geometry: THREE.BufferGeometry; material: THREE.Material } | null,
    list: Placement[],
    heightAt: (x: number, z: number) => number | null,
    paint: Array<[number, number, number]> | null = null,
  ): void {
    const cells = new Map<number, Placement[]>();
    for (const p of list) {
      if (!Number.isFinite(p.y)) p.y = heightAt(p.x, p.z) ?? 0;
      const k = (Math.floor(p.x / CELL) + 32768) * 65536 + (Math.floor(p.z / CELL) + 32768);
      let a = cells.get(k);
      if (!a) cells.set(k, (a = []));
      a.push(p);
    }
    const mk = (g: THREE.BufferGeometry, m: THREE.Material, far: boolean, cap: number): KindLayer => {
      const mesh = new THREE.InstancedMesh(g, m, Math.max(1, Math.min(cap, list.length)));
      mesh.name = `prop:${id}${far ? ':lod' : ''}`;
      mesh.count = 0;
      mesh.frustumCulled = false;
      mesh.castShadow = !far && this.shadows;
      mesh.receiveShadow = !far;
      if (paint) {
        mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(mesh.instanceMatrix.count * 3), 3);
      }
      this.group.add(mesh);
      return { mesh, far, cap: Math.max(1, Math.min(cap, list.length)) };
    };
    const layers = [mk(full.geometry, full.material, false, cls === 'vehicle' ? 600 : 4000)];
    if (lod && FAR[cls] > 0) layers.push(mk(lod.geometry, lod.material, true, 20000));
    this.kinds.push({ id, cls, cells, layers, paint });
    this.lastPos.set(Infinity, 0, 0);
  }

  /** Quality preset draw distance (m, ~800 low .. 2500 high) -> scale of the per-class distances. */
  setDrawDistance(d: number): void {
    this.scaleDist = THREE.MathUtils.clamp(d / 1500, 0.4, 1.6);
    this.lastPos.set(Infinity, 0, 0);
  }

  setShadows(on: boolean): void {
    this.shadows = on;
    for (const k of this.kinds) for (const l of k.layers) l.mesh.castShadow = on && !l.far;
  }

  /** Refill visible instances when the camera moved enough. */
  update(cam: THREE.Vector3, groundY = 0): void {
    if (cam.distanceTo(this.lastPos) < 20) return;
    this.lastPos.copy(cam);
    const alt = Math.max(0, cam.y - groundY);
    let tris = 0;
    for (const k of this.kinds) {
      const near = NEAR[k.cls] * this.scaleDist;
      const far = FAR[k.cls] * this.scaleDist;
      const maxD = Math.max(near, far);
      const counts = k.layers.map(() => 0);
      if (alt < maxD) {
        const r = Math.ceil(maxD / CELL);
        const cx = Math.floor(cam.x / CELL);
        const cz = Math.floor(cam.z / CELL);
        for (let ix = cx - r; ix <= cx + r; ix++) {
          for (let iz = cz - r; iz <= cz + r; iz++) {
            const arr = k.cells.get((ix + 32768) * 65536 + (iz + 32768));
            if (!arr) continue;
            for (let pi = 0; pi < arr.length; pi++) {
              const p = arr[pi]!;
              const d = Math.hypot(p.x - cam.x, p.y - cam.y, p.z - cam.z);
              const li = d < near ? 0 : d < far && k.layers[1] ? 1 : -1;
              if (li < 0) continue;
              const L = k.layers[li]!;
              const n = counts[li]!;
              if (n >= L.cap) continue;
              this.q.setFromAxisAngle(this.up, p.rot);
              this.sv.setScalar(p.scale);
              this.pv.set(p.x, p.y, p.z);
              this.mat4.compose(this.pv, this.q, this.sv);
              L.mesh.setMatrixAt(n, this.mat4);
              if (k.paint && L.mesh.instanceColor) {
                // stable per record: hash the position
                const h = Math.abs(Math.sin(p.x * 12.9898 + p.z * 78.233) * 43758.5453) % 1;
                const c = k.paint[Math.floor(h * k.paint.length) % k.paint.length]!;
                this.color.setRGB(c[0], c[1], c[2]);
                L.mesh.setColorAt(n, this.color);
              }
              counts[li] = n + 1;
            }
          }
        }
      }
      k.layers.forEach((L, i) => {
        L.mesh.count = counts[i]!;
        L.mesh.instanceMatrix.needsUpdate = true;
        if (L.mesh.instanceColor) L.mesh.instanceColor.needsUpdate = true;
        const g = L.mesh.geometry;
        tris += counts[i]! * (g.index ? g.index.count / 3 : g.getAttribute('position').count / 3);
      });
    }
    this.triangles = tris;
  }

  dispose(): void {
    for (const k of this.kinds) {
      for (const L of k.layers) {
        L.mesh.geometry.dispose();
        (L.mesh.material as THREE.Material).dispose();
        L.mesh.dispose();
      }
    }
    this.group.removeFromParent();
  }
}

/** Load trees / shrubs / lamps / parked cars from the manifest + placements; null if absent or not understood. */
export async function loadStaticProps(
  fetchAsset: AssetFetcher,
  heightAt: (x: number, z: number) => number | null,
  vehicleMaterial: () => THREE.Material,
): Promise<StaticProps | null> {
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
    if (!e || !list.length) continue;
    const cls = propClass(id, e.kind);
    try {
      const rel = e.file.startsWith('props/') ? e.file : `props/${e.file}`;
      const buf = await fetchAsset(rel);
      if (cls === 'vehicle') {
        // parked cars: same baked model as traffic, facing -z (placement rot_y follows three.js)
        const g = await loadPropGeometry(buf, false);
        if (g) sp.add(id, cls, { geometry: g, material: vehicleMaterial() }, null, list, heightAt, info.paint.length ? info.paint : null);
        continue;
      }
      const gltf = await gltfLoader().parseAsync(buf, '');
      const geoms: THREE.BufferGeometry[] = [];
      let material: THREE.Material | null = null;
      gltf.scene.updateMatrixWorld(true);
      gltf.scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (!m.isMesh) return;
        const g = m.geometry.clone().applyMatrix4(m.matrixWorld);
        for (const name of Object.keys(g.attributes)) if (!['position', 'normal', 'uv'].includes(name)) g.deleteAttribute(name);
        geoms.push(g.index ? g : g);
        material ??= Array.isArray(m.material) ? m.material[0]! : m.material;
      });
      if (!geoms.length || !material) continue;
      const merged = mergeGeometries(geoms, false);
      if (!merged) continue;
      const mat = material as THREE.MeshStandardMaterial;
      if (mat.map) mat.map.anisotropy = 4;
      let lod: { geometry: THREE.BufferGeometry; material: THREE.Material } | null = null;
      if (FAR[cls] > 0) {
        const avg = averageTextureColor(mat.map) ?? new THREE.Color(0.22, 0.3, 0.14);
        lod = { geometry: lodGeometry(merged, cls, avg), material: new THREE.MeshLambertMaterial({ vertexColors: true }) };
      }
      sp.add(id, cls, { geometry: merged, material: mat }, lod, list, heightAt);
    } catch (err) {
      console.warn(`prop ${id} failed`, err);
    }
  }
  return sp;
}
