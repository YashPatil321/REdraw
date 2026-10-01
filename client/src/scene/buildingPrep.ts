/**
 * CPU preparation of building tiles for the atlas shaders.
 *
 * Pipeline tiles may carry `_MAT`, `_VARIANT` (per-vertex material id /
 * variant, see blender/rdlib/atlas.py) and facade UVs (TEXCOORD_0: walls
 * u = m along the wall / 3, v = m above the base / 3; pitched roofs u along
 * the eave / 4, v up the slope / 4; flat roofs x / 4, -z / 4). Older tiles
 * only have positions, normals, `_BUILDING_ID` and type colors: everything is
 * derived here from the geometry so one shader path serves both.
 *
 * Always computed: `aWall` (Uint16 x4) = [wall length dm, wall key + 256 * flags,
 * building height dm, building hash], where flags bit 0 = this wall faces the
 * street (garage / front door wall), bits 1-2 = kind (0 house, 1 apartments,
 * 2 commercial, 3 school). The index buffer is regrouped into
 * group 0 = walls (+ glass, trim, garage), group 1 = roofs.
 */

import * as THREE from 'three';

export const MAT_WALL = 0;
export const MAT_TILE_ROOF = 1;
export const MAT_FLAT_ROOF = 2;
export const MAT_GLASS = 3;
export const MAT_TRIM = 4;
export const MAT_GARAGE = 5;

export type StreetDirFn = (x: number, z: number) => { dx: number; dz: number } | null;

export interface PrepOptions {
  /** direction from a point toward the nearest street (unit), or null */
  streetDir?: StreetDirFn;
  /** variant counts per material id (wall, tile roof, flat roof) for derived variants */
  variants?: { wall: number; tileRoof: number; flatRoof: number };
}

export interface PrepResult {
  /** pipeline supplied `_MAT` (true) or it was derived (false) */
  hadMat: boolean;
  hadUv: boolean;
  buildings: number;
}

/** Stable 32-bit hash of an integer -> [0, 1). Pure; unit tested. */
export function hash01(n: number, salt = 0): number {
  let h = (Math.imul(n | 0, 0x9e3779b1) ^ Math.imul(salt | 0, 0x85ebca6b)) >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b) >>> 0;
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** Classify a face normal: wall / pitched roof / flat roof. Pure; unit tested. */
export function classifyNormal(ny: number): number {
  if (ny < 0.3) return ny < -0.5 ? MAT_FLAT_ROOF : MAT_WALL;
  return ny < 0.985 ? MAT_TILE_ROOF : MAT_FLAT_ROOF;
}

/** SoCal stucco wall colors (sRGB): tract-home creams, sands, taupes, a few whites and greys. */
const WALL_TINTS = ['#E8DCC4', '#DCCBAA', '#E4D3B4', '#D2BE9C', '#EFE7D6', '#CDBBA0', '#D9C9AE', '#C9B596', '#E6DFD0', '#D5CCBC', '#BFAE94', '#E2CFAE', '#D8C0A0', '#C4B8A6'];
const COMMERCIAL_TINTS = ['#E6E0D4', '#D9D2C4', '#CFC6B4', '#E9E4D8', '#C8BDA8', '#DDD3BF'];

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function tintRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [srgbToLinear(((n >> 16) & 255) / 255), srgbToLinear(((n >> 8) & 255) / 255), srgbToLinear((n & 255) / 255)];
}

interface BInfo {
  base: number;
  top: number;
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  kind: number;
  hash: number;
}

/**
 * Add / derive the attributes the building shaders need and regroup the index
 * buffer (walls, roofs). Mutates `g`. Returns what the source provided.
 */
export function prepareBuildingGeometry(g: THREE.BufferGeometry, opts: PrepOptions = {}): PrepResult {
  const pos = g.getAttribute('position') as THREE.BufferAttribute;
  if (!g.getAttribute('normal')) g.computeVertexNormals();
  const nrm = g.getAttribute('normal') as THREE.BufferAttribute;
  const idAttr = (g.getAttribute('_building_id') ?? null) as THREE.BufferAttribute | null;
  const matAttr = (g.getAttribute('_mat') ?? null) as THREE.BufferAttribute | null;
  const varAttr = (g.getAttribute('_variant') ?? null) as THREE.BufferAttribute | null;
  const uvAttr = (g.getAttribute('uv') ?? null) as THREE.BufferAttribute | null;
  const n = pos.count;
  const bid = new Int32Array(n);
  for (let i = 0; i < n; i++) bid[i] = idAttr ? Math.round(idAttr.getX(i)) : 0;

  // per-building extents
  const info = new Map<number, BInfo>();
  for (let i = 0; i < n; i++) {
    const b = bid[i]!;
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    let bi = info.get(b);
    if (!bi) info.set(b, (bi = { base: y, top: y, minX: x, maxX: x, minZ: z, maxZ: z, kind: 0, hash: 0 }));
    if (y < bi.base) bi.base = y;
    if (y > bi.top) bi.top = y;
    if (x < bi.minX) bi.minX = x;
    if (x > bi.maxX) bi.maxX = x;
    if (z < bi.minZ) bi.minZ = z;
    if (z > bi.maxZ) bi.maxZ = z;
  }
  for (const [b, bi] of info) {
    const h = bi.top - bi.base;
    const area = (bi.maxX - bi.minX) * (bi.maxZ - bi.minZ);
    bi.hash = hash01(b, 7);
    // kind from size: houses are short and small; large low footprints are commercial
    bi.kind = h < 10.5 && area < 700 ? 0 : area > 2500 && h < 14 ? 2 : h >= 10.5 && area < 2500 ? 1 : 2;
  }

  // material id per vertex
  const mat = new Uint8Array(n);
  for (let i = 0; i < n; i++) mat[i] = matAttr ? Math.round(matAttr.getX(i)) : classifyNormal(nrm.getY(i));

  // wall planes: group by building + plane, for start corner, length and street-facing
  interface Plane {
    b: number;
    tx: number;
    tz: number;
    nx: number;
    nz: number;
    sMin: number;
    sMax: number;
    yMin: number;
    key: number;
    street: boolean;
  }
  const planes = new Map<number, Plane>();
  // numeric plane key: building id, normal angle (deg), plane offset (0.25 m)
  const pkey = (b: number, ang: number, d: number): number => (b * 361 + (ang + 180)) * 262144 + (d + 131072);
  const planeOf = new Int32Array(n).fill(-1);
  const planeList: Plane[] = [];
  // roof planes (eave alignment of S tiles)
  const roofPlanes = new Map<number, { vMin: number }>();
  const roofOf: Array<{ vMin: number } | null> = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    const m = mat[i]!;
    const nx = nrm.getX(i);
    const nz = nrm.getZ(i);
    const ny = nrm.getY(i);
    const hl = Math.hypot(nx, nz);
    if (hl < 1e-4) continue;
    const ux = nx / hl;
    const uz = nz / hl;
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const ang = Math.round((Math.atan2(uz, ux) * 180) / Math.PI);
    if (m === MAT_WALL || m === MAT_GLASS || m === MAT_TRIM || m === MAT_GARAGE) {
      const d = Math.round((x * ux + z * uz) * 4);
      const k = pkey(bid[i]!, ang, d);
      let p = planes.get(k);
      if (!p) {
        p = { b: bid[i]!, tx: -uz, tz: ux, nx: ux, nz: uz, sMin: Infinity, sMax: -Infinity, yMin: Infinity, key: 0, street: false };
        planes.set(k, p);
        planeList.push(p);
      }
      const s = x * p.tx + z * p.tz;
      if (s < p.sMin) p.sMin = s;
      if (s > p.sMax) p.sMax = s;
    } else if (m === MAT_TILE_ROOF && ny > 0.05) {
      const k = bid[i]! * 361 + (ang + 180);
      let rp = roofPlanes.get(k);
      if (!rp) roofPlanes.set(k, (rp = { vMin: Infinity }));
      // up-slope distance in the roof plane (horizontal distance / cos(slope))
      const v = (-(x * ux + z * uz)) / Math.max(ny, 0.2);
      if (v < rp.vMin) rp.vMin = v;
      roofOf[i] = rp;
    }
  }
  // plane index per vertex
  const planeIdx = new Map<Plane, number>();
  planeList.forEach((p, k) => planeIdx.set(p, k));
  for (let i = 0; i < n; i++) {
    const m = mat[i]!;
    if (!(m === MAT_WALL || m === MAT_GLASS || m === MAT_TRIM || m === MAT_GARAGE)) {
      planeOf[i] = -1;
      continue;
    }
    const nx = nrm.getX(i);
    const nz = nrm.getZ(i);
    const hl = Math.hypot(nx, nz);
    if (hl < 1e-4) {
      planeOf[i] = -1;
      continue;
    }
    const ux = nx / hl;
    const uz = nz / hl;
    const ang = Math.round((Math.atan2(uz, ux) * 180) / Math.PI);
    const d = Math.round((pos.getX(i) * ux + pos.getZ(i) * uz) * 4);
    const p = planes.get(pkey(bid[i]!, ang, d));
    planeOf[i] = p ? planeIdx.get(p)! : -1;
  }
  // street-facing wall per building: best aligned with the direction to the nearest street, long enough
  const byB = new Map<number, Plane[]>();
  planeList.forEach((p, k) => {
    p.key = Math.floor(hash01(k + p.b * 31, 3) * 255);
    let a = byB.get(p.b);
    if (!a) byB.set(p.b, (a = []));
    a.push(p);
  });
  for (const [b, list] of byB) {
    const bi = info.get(b);
    if (!bi) continue;
    const cx = (bi.minX + bi.maxX) / 2;
    const cz = (bi.minZ + bi.maxZ) / 2;
    const sd = opts.streetDir?.(cx, cz) ?? null;
    let best: Plane | null = null;
    let bestScore = -Infinity;
    for (const p of list) {
      const len = p.sMax - p.sMin;
      if (len < 5.5) continue;
      const align = sd ? p.nx * sd.dx + p.nz * sd.dz : 0;
      const score = (sd ? align * 10 : 0) + len * 0.1;
      if (score > bestScore) {
        bestScore = score;
        best = p;
      }
    }
    if (best) best.street = true;
  }

  // attributes
  const needUv = !uvAttr;
  const uv = needUv ? new Float32Array(n * 2) : null;
  const wall = new Uint16Array(n * 4);
  const vv = new Uint8Array(n * 2);
  const vc = g.getAttribute('color');
  const newStyle = !!matAttr;
  const tint = newStyle && vc ? null : new Float32Array(n * 3);
  const nv = opts.variants ?? { wall: 7, tileRoof: 10, flatRoof: 6 };
  for (let i = 0; i < n; i++) {
    const b = bid[i]!;
    const bi = info.get(b)!;
    const m = mat[i]!;
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const pk = planeOf[i]!;
    const p = pk >= 0 ? planeList[pk]! : null;
    if (uv) {
      if (p) {
        uv[i * 2] = (x * p.tx + z * p.tz - p.sMin) / 3;
        uv[i * 2 + 1] = (y - bi.base) / 3;
      } else if (m === MAT_TILE_ROOF) {
        const nx = nrm.getX(i);
        const nz = nrm.getZ(i);
        const hl = Math.hypot(nx, nz) || 1;
        const ux = nx / hl;
        const uz = nz / hl;
        const rp = roofOf[i];
        const v = (-(x * ux + z * uz)) / Math.max(nrm.getY(i), 0.2);
        uv[i * 2] = (x * -uz + z * ux) / 4;
        uv[i * 2 + 1] = (v - (rp ? rp.vMin : 0)) / 4;
      } else {
        uv[i * 2] = x / 4;
        uv[i * 2 + 1] = -z / 4;
      }
    }
    const len = p ? p.sMax - p.sMin : 0;
    const flags = (p?.street ? 1 : 0) | (bi.kind << 1);
    wall[i * 4] = Math.min(65535, Math.round(len * 10));
    wall[i * 4 + 1] = (p ? p.key : 0) + 256 * flags;
    wall[i * 4 + 2] = Math.min(65535, Math.round((bi.top - bi.base) * 10));
    wall[i * 4 + 3] = Math.floor(bi.hash * 65535);
    // variant
    let variant = varAttr ? Math.round(varAttr.getX(i)) : 0;
    if (!varAttr) {
      const h1 = hash01(b, 11);
      const h2 = hash01(b, 13);
      if (m === MAT_WALL || m === MAT_TRIM) {
        // tract homes: mostly sand finish, some lace / smooth / weathered; commercial and schools scored
        variant = bi.kind === 0 ? [1, 1, 1, 1, 2, 2, 0, 4][Math.floor(h1 * 8)]! : bi.kind === 1 ? [1, 0, 5, 2][Math.floor(h1 * 4)]! : 5;
        variant = Math.min(variant, nv.wall - 1);
      } else if (m === MAT_TILE_ROOF) {
        // S-tile blends dominate; flat concrete tile on ~30%; occasional solar on south slopes
        const choice = [0, 1, 1, 2, 3, 1, 5, 6, 7, 8][Math.floor(h2 * 10)]!;
        variant = Math.min(choice, nv.tileRoof - 1);
        const south = nrm.getZ(i) > 0.25;
        if (south && hash01(b, 17) < 0.12 && bi.kind === 0) variant = Math.min(9, nv.tileRoof - 1);
      } else if (m === MAT_FLAT_ROOF) {
        variant = Math.min([0, 1, 1, 2, 3, 1][Math.floor(h2 * 6)]!, nv.flatRoof - 1);
      }
    }
    vv[i * 2] = m;
    vv[i * 2 + 1] = variant;
    if (tint) {
      const pal = bi.kind === 0 ? WALL_TINTS : COMMERCIAL_TINTS;
      const c = tintRgb(pal[Math.floor(hash01(b, 23) * pal.length)]!);
      tint[i * 3] = c[0];
      tint[i * 3 + 1] = c[1];
      tint[i * 3 + 2] = c[2];
    }
  }
  if (uv) g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setAttribute('aWall', new THREE.BufferAttribute(wall, 4));
  g.setAttribute('aMatVar', new THREE.BufferAttribute(vv, 2));
  if (tint) g.setAttribute('color', new THREE.BufferAttribute(tint, 3));
  // base / top per vertex (used by the procedural fallback shader too)
  const base = new Float32Array(n);
  const top = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const bi = info.get(bid[i]!)!;
    base[i] = bi.base;
    top[i] = bi.top;
  }
  g.setAttribute('aBase', new THREE.BufferAttribute(base, 1));
  g.setAttribute('aTop', new THREE.BufferAttribute(top, 1));

  // regroup: walls first, roofs second
  regroup(g, (a, b, c) => {
    const isRoof = (m: number): boolean => m === MAT_TILE_ROOF || m === MAT_FLAT_ROOF;
    return isRoof(mat[a]!) && isRoof(mat[b]!) && isRoof(mat[c]!) ? 1 : 0;
  });
  return { hadMat: !!matAttr, hadUv: !!uvAttr, buildings: info.size };
}

/** Sort triangles into groups by `groupOf` (0 or 1) and set geometry groups. */
export function regroup(g: THREE.BufferGeometry, groupOf: (a: number, b: number, c: number) => number): void {
  let index = g.getIndex();
  if (!index) {
    const n = g.getAttribute('position').count;
    const arr = n > 65535 ? new Uint32Array(n) : new Uint16Array(n);
    for (let i = 0; i < n; i++) arr[i] = i;
    index = new THREE.BufferAttribute(arr, 1);
  }
  const src = index.array as Uint16Array | Uint32Array;
  const tris = src.length / 3;
  const out = new (src.constructor as Uint32ArrayConstructor)(src.length);
  let w0 = 0;
  const g1: number[] = [];
  for (let t = 0; t < tris; t++) {
    const a = src[t * 3]!;
    const b = src[t * 3 + 1]!;
    const c = src[t * 3 + 2]!;
    if (groupOf(a, b, c) === 0) {
      out[w0++] = a;
      out[w0++] = b;
      out[w0++] = c;
    } else g1.push(a, b, c);
  }
  const n0 = w0;
  for (const v of g1) out[w0++] = v;
  g.setIndex(new THREE.BufferAttribute(out, 1));
  g.clearGroups();
  g.addGroup(0, n0, 0);
  g.addGroup(n0, out.length - n0, 1);
}
