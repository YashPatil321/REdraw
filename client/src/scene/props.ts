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
import { LOD_FADE, addLodFade, fadeBands, makeFadeUniforms, type FadeUniforms } from './lodFade';

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
    const rough = new Float32Array(n);
    const metal = new Float32Array(n);
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
        // clearcoat paint reads glossier than its base roughness
        const cc = (mat as THREE.MeshPhysicalMaterial | undefined)?.clearcoat ?? 0;
        rough[i] = Math.max(0.04, (mat?.roughness ?? 0.6) * (cc > 0 ? 0.6 : 1));
        metal[i] = mat?.metalness ?? 0;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', src.getAttribute('position'));
    g.setAttribute('normal', src.getAttribute('normal'));
    g.setAttribute('color', new THREE.BufferAttribute(color, 3));
    g.setAttribute('aLight', new THREE.BufferAttribute(light, 1));
    g.setAttribute('aTint', new THREE.BufferAttribute(tint, 1));
    g.setAttribute('aRough', new THREE.BufferAttribute(rough, 1));
    g.setAttribute('aMetal', new THREE.BufferAttribute(metal, 1));
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
  /** record index in placements.bin (stable seed, e.g. parked-car paint) */
  i: number;
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
      arr.push({ x: f(ix), y: iy >= 0 ? f(iy) : NaN, z: f(iz), rot: f(ir), scale: is >= 0 ? f(is) || 1 : 1, i: Math.round(o / stride) });
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
const NEAR: Record<PropClass, number> = { tree: 220, palm: 260, shrub: 100, grass: 60, lamp: 380, vehicle: 300, other: 150 };
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
  parts.push(crown.index ? crown.toNonIndexed() : crown);
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
  /** full-model distance (m) at draw-distance 1.0 */
  near: number;
  /** distance bands of the dithered LOD crossfade (lodFade.ts) */
  fade: FadeUniforms;
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
    nearOverride?: number,
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
    // dissolve between the full model and the impostor (and out at the far end) instead of popping
    const fade = makeFadeUniforms();
    for (const L of layers) addLodFade(L.mesh.material as THREE.Material, fade, L.far ? 'lod' : 'full');
    this.kinds.push({ id, cls, cells, layers, paint, near: nearOverride ? Math.max(nearOverride * 1.15, 90) : NEAR[cls], fade });
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
    if (cam.distanceTo(this.lastPos) < LOD_FADE.refillM) return;
    this.lastPos.copy(cam);
    const alt = Math.max(0, cam.y - groundY);
    let tris = 0;
    const margin = LOD_FADE.margin;
    for (const k of this.kinds) {
      const near = k.near * this.scaleDist;
      const hasLod = k.layers.length > 1;
      const far = hasLod ? FAR[k.cls] * this.scaleDist : 0;
      const b = fadeBands(near, far);
      k.fade.uRdFade.value.set(b.n0, b.n1, b.f0, b.f1);
      const maxD = Math.max(b.n1 + margin, b.f1);
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
              // in the crossfade band an instance is in both layers (the shader dithers between them)
              const inFull = d < b.n1 + margin;
              const inLod = hasLod && d > b.n0 - margin && d < b.f1;
              if (!inFull && !inLod) continue;
              this.q.setFromAxisAngle(this.up, p.rot);
              this.sv.setScalar(p.scale);
              this.pv.set(p.x, p.y, p.z);
              this.mat4.compose(this.pv, this.q, this.sv);
              for (let li = 0; li < k.layers.length; li++) {
                if (li === 0 ? !inFull : !inLod) continue;
                const L = k.layers[li]!;
                const n = counts[li]!;
                if (n >= L.cap) continue;
                L.mesh.setMatrixAt(n, this.mat4);
                if (k.paint && L.mesh.instanceColor) {
                  // stable per record: seeded by the record index (palette is already weighted by share)
                  const h = Math.abs(Math.sin(p.i * 12.9898 + 4.1) * 43758.5453) % 1;
                  const c = k.paint[Math.floor(h * k.paint.length) % k.paint.length]!;
                  this.color.setRGB(c[0], c[1], c[2]);
                  L.mesh.setColorAt(n, this.color);
                }
                counts[li] = n + 1;
              }
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

/** Placements built for another world extent (their cell grid does not start at the manifest grid). Pure; unit tested. */
export function placementsStale(meta: Record<string, unknown>, grid: { min_x: number; min_z: number } | null): boolean {
  const cells = meta['cells'] as { min_x?: unknown; min_z?: unknown } | undefined;
  if (!grid || !cells || typeof cells.min_x !== 'number' || typeof cells.min_z !== 'number') return false;
  return Math.abs(cells.min_x - grid.min_x) > 1 || Math.abs(cells.min_z - grid.min_z) > 1;
}

/** Load trees / shrubs / lamps / parked cars from the manifest + placements; null if absent or not understood. */
export async function loadStaticProps(
  fetchAsset: AssetFetcher,
  heightAt: (x: number, z: number) => number | null,
  vehicleMaterial: () => THREE.Material,
  grid: { min_x: number; min_z: number } | null = null,
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
  if (placementsStale(meta, grid)) {
    console.warn('props: placements.bin is from an older world build (different tile grid); trees and lamps skipped until the props are rebuilt');
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
      const full = await loadFoliage(buf);
      if (!full) continue;
      const mat = full.material as THREE.MeshStandardMaterial;
      if (mat.map) mat.map.anisotropy = 4;
      if (cls !== 'lamp' && cls !== 'other') addWind(mat, full.geometry);
      let lod: { geometry: THREE.BufferGeometry; material: THREE.Material } | null = null;
      // baked impostor (crossed quads) from the prop library, else a colored crown
      const lodFile = (e['lod1_file'] as string | undefined) ?? (Array.isArray(e['lods']) ? ((e['lods'] as Array<{ level?: number; file?: string }>).find((x) => x.level === 1)?.file ?? null) : null);
      if (lodFile && FAR[cls] > 0) {
        try {
          const imp = await loadFoliage(await fetchAsset(lodFile.startsWith('props/') ? lodFile : `props/${lodFile}`));
          if (imp) {
            addWind(imp.material as THREE.MeshStandardMaterial, imp.geometry);
            lod = imp;
          }
        } catch (err) {
          console.warn(`impostor ${id} failed`, err);
        }
      }
      if (!lod && FAR[cls] > 0) {
        const avg = averageTextureColor(mat.map) ?? new THREE.Color(0.22, 0.3, 0.14);
        lod = { geometry: lodGeometry(full.geometry, cls, avg), material: new THREE.MeshLambertMaterial({ vertexColors: true }) };
      }
      const lods = e['lods'] as Array<{ level?: number; max_distance_m?: number }> | undefined;
      const near = lods?.find((x) => x.level === 0)?.max_distance_m;
      const merged = full.geometry;
      const mat2 = full.material;
      sp.add(id, cls, { geometry: merged, material: mat2 }, lod, list, heightAt, null, lod && near ? near : undefined);
    } catch (err) {
      console.warn(`prop ${id} failed`, err);
    }
  }
  return sp;
}

/** shared uniforms of the foliage wind (time in seconds, gust strength 0..1) */
export const windUniforms = {
  uWindTime: { value: 0 },
  uWindStrength: { value: 0.5 },
  uWindDir: { value: new THREE.Vector2(0.8, 0.6) },
};

/**
 * Wind sway in the vertex shader: displacement grows with height above the
 * prop origin (trunk stays put), phase from the instance position so trees do
 * not move in lockstep, plus a faster leaf flutter.
 */
export function addWind(mat: THREE.MeshStandardMaterial, g: THREE.BufferGeometry): void {
  if ((mat.userData as { rdWind?: boolean }).rdWind) return;
  (mat.userData as { rdWind?: boolean }).rdWind = true;
  g.computeBoundingBox();
  const h = Math.max(0.5, g.boundingBox!.max.y);
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader, r) => {
    prev.call(mat, shader, r);
    Object.assign(shader.uniforms, windUniforms, { uWindH: { value: h } });
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uWindTime;\nuniform float uWindStrength;\nuniform vec2 uWindDir;\nuniform float uWindH;')
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
{
  #ifdef USE_INSTANCING
    vec3 rdIp = instanceMatrix[3].xyz;
  #else
    vec3 rdIp = vec3(0.0);
  #endif
  float rdPh = dot(rdIp.xz, vec2(0.071, 0.113));
  float rdK = clamp(transformed.y / uWindH, 0.0, 1.2);
  rdK *= rdK;
  float rdSway = sin(uWindTime * 0.9 + rdPh) * 0.6 + sin(uWindTime * 2.1 + rdPh * 1.7) * 0.25;
  float rdFl = sin(uWindTime * 7.0 + dot(transformed, vec3(3.1, 2.3, 4.7))) * 0.035 * clamp(transformed.y / uWindH * 2.0, 0.0, 1.0);
  float rdA = uWindStrength * uWindH * 0.012;
  transformed.xz += uWindDir * (rdSway * rdA * rdK) + vec2(rdFl, -rdFl) * uWindStrength;
}`,
      );
  };
  const key = mat.customProgramCacheKey.bind(mat);
  mat.customProgramCacheKey = () => `${key()}-rdwind`;
}

/**
 * One merged geometry + material from a foliage glb: positions / normals / uvs
 * and the baked crown AO (COLOR_0, multiplied into the base color).
 */
async function loadFoliage(buf: ArrayBuffer): Promise<{ geometry: THREE.BufferGeometry; material: THREE.Material } | null> {
  const gltf = await gltfLoader().parseAsync(buf, '');
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
  if (!geoms.length || !material) return null;
  const anyColor = geoms.some((g) => g.getAttribute('color'));
  for (const g of geoms) {
    const c = g.getAttribute('color');
    if (anyColor && !c) g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(g.getAttribute('position').count * 3).fill(1), 3));
    else if (c && c.itemSize === 4) {
      const rgb = new Float32Array(c.count * 3);
      for (let i = 0; i < c.count; i++) {
        rgb[i * 3] = c.getX(i);
        rgb[i * 3 + 1] = c.getY(i);
        rgb[i * 3 + 2] = c.getZ(i);
      }
      g.setAttribute('color', new THREE.BufferAttribute(rgb, 3));
    }
  }
  const merged = geoms.length === 1 ? geoms[0]! : mergeGeometries(geoms, false);
  if (!merged) return null;
  const mat = material as THREE.MeshStandardMaterial;
  if (anyColor) mat.vertexColors = true;
  mat.needsUpdate = true;
  return { geometry: merged, material: mat };
}

/** The lawn grass tuft for near-camera scattering (props manifest entry with `scatter`), or null. */
export async function loadGrassTuft(fetchAsset: AssetFetcher): Promise<{ geometry: THREE.BufferGeometry; material: THREE.Material } | null> {
  try {
    const raw = JSON.parse(new TextDecoder().decode(await fetchAsset('props/props_manifest.json'))) as Record<string, unknown> | unknown[];
    const info = parsePropsManifest(raw);
    const e = info.entries.find((x) => typeof x['scatter'] === 'string' || x.id === 'grass_tuft' || x['kind'] === 'groundcover');
    if (!e) return null;
    return await loadFoliage(await fetchAsset(e.file.startsWith('props/') ? e.file : `props/${e.file}`));
  } catch {
    return null;
  }
}
