/**
 * One traffic playback rendered on the map: congestion overlay, instanced cars
 * (colored by speed), a separate instanced mesh for shuttles / school buses,
 * queue bars at school entrances and optional ghost trails (additive lines).
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { binBlend, binIndex } from '../time';
import type { School } from '../types';
import { queueColor, speedColor, type RGB } from './congestion';
import type { PosOut, RoadNetwork } from './network';
import { queueAt, type Playback } from './playback';
import { RoadOverlay, adjTexture } from './roadOverlay';

export const MAX_VEHICLES = 3000;
const TRAIL_SEGMENTS = 10;
const TRAIL_STEP_S = 6; // sim seconds between trail samples
const LANE_OFFSET_M = 2.2;

const WHITE = new THREE.Color(1, 1, 1);

/**
 * Box with a constant `aLight` value (0 body, 1 headlight, 2 taillight, 3 dark
 * glass, 4 dark trim), white vertex color and `aTint` = 1 on the body (the
 * instance color goes there only). Same attribute contract as the Blender
 * props (`_LIGHT`, `_TINT`, COLOR_0 baked from materials).
 */
function part(w: number, h: number, d: number, x: number, y: number, z: number, light: number): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(w, h, d).translate(x, y, z).toNonIndexed();
  g.deleteAttribute('uv');
  const n = g.getAttribute('position').count;
  g.setAttribute('aLight', new THREE.BufferAttribute(new Float32Array(n).fill(light), 1));
  g.setAttribute('aTint', new THREE.BufferAttribute(new Float32Array(n).fill(light === 0 ? 1 : 0), 1));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3).fill(1), 3));
  // PBR per vertex: metallic clear-coated paint, glossy glass, matte trim
  const [rough, metal] = light === 0 ? [0.35, 0.5] : light === 3 ? [0.06, 0.0] : light === 4 ? [0.85, 0.0] : [0.2, 0.0];
  g.setAttribute('aRough', new THREE.BufferAttribute(new Float32Array(n).fill(rough), 1));
  g.setAttribute('aMetal', new THREE.BufferAttribute(new Float32Array(n).fill(metal), 1));
  return g;
}

function merge(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const g = mergeGeometries(parts)!;
  parts.forEach((p) => p.dispose());
  return g;
}

/** Low-poly car along +x with emissive head/tail lights (bloom picks them up). */
export function carGeometry(): THREE.BufferGeometry {
  return merge([
    part(4.4, 0.9, 1.85, 0, 0.65, 0, 0),
    part(2.3, 0.7, 1.65, -0.35, 1.45, 0, 3),
    part(0.12, 0.22, 0.38, 2.21, 0.8, 0.6, 1),
    part(0.12, 0.22, 0.38, 2.21, 0.8, -0.6, 1),
    part(0.12, 0.2, 0.42, -2.21, 0.85, 0.62, 2),
    part(0.12, 0.2, 0.42, -2.21, 0.85, -0.62, 2),
  ]);
}

/** Bus / shuttle along +x. */
export function busGeometry(): THREE.BufferGeometry {
  return merge([
    part(11, 2.6, 2.5, 0, 1.7, 0, 0),
    part(10.2, 0.7, 2.54, -0.2, 2.25, 0, 3),
    part(0.12, 0.28, 0.45, 5.51, 0.9, 0.85, 1),
    part(0.12, 0.28, 0.45, 5.51, 0.9, -0.85, 1),
    part(0.12, 0.28, 0.45, -5.51, 1.0, 0.85, 2),
    part(0.12, 0.28, 0.45, -5.51, 1.0, -0.85, 2),
  ]);
}

/** Shared uniforms driving vehicle light intensity (set from the sky's darkness). */
export const vehicleLightUniforms = {
  uHead: { value: 2.0 },
  uTail: { value: 1.6 },
};

/** Environment map for vehicle paint / glass reflections (set once by the scene; PMREM). */
let vehicleEnv: THREE.Texture | null = null;
const vehicleMats = new Set<THREE.MeshStandardMaterial>();
export function setVehicleEnvMap(tex: THREE.Texture | null): void {
  vehicleEnv = tex;
  for (const m of vehicleMats) {
    m.envMap = tex;
    m.needsUpdate = true;
  }
}

/**
 * Vehicles: PBR with per-vertex roughness / metalness (`aRough`, `aMetal`),
 * baked albedo (COLOR_0), instance color multiplied into the tintable paint
 * only (`aTint`), emissive head / tail lights (`aLight`) that bloom at dawn.
 */
export function vehicleMaterial(): THREE.MeshStandardMaterial {
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 1, metalness: 1, envMap: vehicleEnv, envMapIntensity: 0.9 });
  vehicleMats.add(mat);
  mat.addEventListener('dispose', () => vehicleMats.delete(mat));
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, vehicleLightUniforms);
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        '#include <common>\nattribute float aLight;\nattribute float aTint;\nattribute float aRough;\nattribute float aMetal;\nvarying float vLight;\nvarying float vRough;\nvarying float vMetal;',
      )
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvLight = aLight;\nvRough = aRough;\nvMetal = aMetal;')
      // instance color (speed or paint) only on the tintable body panels
      .replace(
        '#include <color_vertex>',
        `vColor = vec4(1.0);
        #ifdef USE_COLOR
          vColor.rgb *= color;
        #endif
        #ifdef USE_INSTANCING_COLOR
          vColor.rgb *= mix(vec3(1.0), instanceColor.rgb, aTint);
        #endif`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vLight;\nvarying float vRough;\nvarying float vMetal;\nuniform float uHead;\nuniform float uTail;')
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = clamp(vRough, 0.04, 1.0);')
      .replace('#include <metalnessmap_fragment>', '#include <metalnessmap_fragment>\nmetalnessFactor = clamp(vMetal, 0.0, 1.0);')
      .replace(
        '#include <opaque_fragment>',
        `if (vLight > 0.5 && vLight < 1.5) outgoingLight = vec3(1.0, 0.93, 0.78) * uHead;
        else if (vLight > 1.5 && vLight < 2.5) outgoingLight = vec3(1.0, 0.06, 0.04) * uTail;
        #include <opaque_fragment>`,
      );
  };
  mat.customProgramCacheKey = () => 'redraw-vehicle-pbr';
  return mat;
}

let sharedCar: THREE.BufferGeometry | null = null;
let sharedBus: THREE.BufferGeometry | null = null;
let sharedBar: THREE.BufferGeometry | null = null;
let sharedBlob: THREE.BufferGeometry | null = null;
let blobTex: THREE.DataTexture | null = null;

/** Soft elliptical falloff (alpha) for vehicle contact shadows. */
function blobTexture(): THREE.DataTexture {
  if (blobTex) return blobTex;
  const N = 64;
  const data = new Uint8Array(N * N * 4);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const u = (x + 0.5) / N - 0.5;
      const v = (y + 0.5) / N - 0.5;
      const r = Math.min(1, Math.hypot(u * 2, v * 2));
      const a = Math.pow(1 - r, 1.6);
      const k = (y * N + x) * 4;
      data[k] = data[k + 1] = data[k + 2] = 255;
      data[k + 3] = Math.round(a * 255);
    }
  }
  blobTex = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
  blobTex.magFilter = THREE.LinearFilter;
  blobTex.minFilter = THREE.LinearFilter;
  blobTex.needsUpdate = true;
  return blobTex;
}

/** Optional vehicle meshes from the props library (fallback: procedural boxes). */
export interface VehicleGeometries {
  /** passenger car variants with their fleet share (sedan, SUV, minivan, pickup...) */
  cars?: Array<{ id: string; geometry: THREE.BufferGeometry; share: number }>;
  /** yellow school bus */
  bus?: THREE.BufferGeometry;
  /** shuttle van (plan shuttles) */
  shuttle?: THREE.BufferGeometry;
  /** realistic paint colors (linear RGB), used when zoomed in */
  paint?: Array<[number, number, number]>;
}

/** Deterministic hash of a trajectory index to [0, 1). */
function hash01(i: number, salt: number): number {
  const x = Math.sin(i * 12.9898 + salt * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

/** Pick a variant index by cumulative share (deterministic per trajectory). Pure; unit tested. */
export function pickVariant(shares: number[], u: number): number {
  const total = shares.reduce((a, b) => a + Math.max(0, b), 0);
  if (total <= 0) return 0;
  let acc = 0;
  for (let k = 0; k < shares.length; k++) {
    acc += Math.max(0, shares[k]!) / total;
    if (u < acc) return k;
  }
  return shares.length - 1;
}

export interface LayerOptions {
  /** accent color used for shuttles and the queue label border */
  accent: THREE.ColorRepresentation;
  showLabels: boolean;
  vehicles?: VehicleGeometries;
  castShadows?: boolean;
}

export class TrafficLayer {
  readonly group = new THREE.Group();
  readonly overlay: RoadOverlay;
  /** one instanced mesh per vehicle model: car variants, then school bus, then shuttle */
  private fleets: Array<{ mesh: THREE.InstancedMesh; kind: 'car' | 'school_bus' | 'shuttle'; n: number }> = [];
  /** per trajectory: fleet index */
  private fleetOf: Uint8Array;
  /** per trajectory: paint color index */
  private paintOf: Uint16Array;
  /** 'speed': cars colored by speed (legend); 'paint': realistic paint when zoomed in */
  private colorMode: 'speed' | 'paint' = 'speed';
  private bars: THREE.InstancedMesh | null = null;
  private barLabels: CSS2DObject[] = [];
  private barEntrances: Array<{ k: number; x: number; y: number; y0: number; z: number; curb: number }> = [];
  private trails: THREE.LineSegments;
  /** soft contact shadow under every vehicle (grounds cars on unlit photo tiles too) */
  private blobs: THREE.InstancedMesh;
  private fleetSize: Array<[number, number]> = [];
  private trailPos: Float32Array;
  private trailCol: Float32Array;
  private cursors: Int32Array;
  private nTraj: number;
  private tmp: PosOut = { x: 0, y: 0, z: 0, dx: 1, dz: 0 };
  private col: RGB = [0, 0, 0];
  private ghost = false;
  private shuttleColor: THREE.Color;
  private busColor = new THREE.Color(0xffc21a);

  constructor(
    readonly net: RoadNetwork,
    readonly pb: Playback,
    overlayGeometry: THREE.BufferGeometry,
    schools: School[],
    private opts: LayerOptions,
  ) {
    this.group.name = `traffic:${pb.header.plan_id}`;
    this.overlay = new RoadOverlay(overlayGeometry, Math.min(net.nEdges, pb.nEdges), adjTexture(net));
    this.group.add(this.overlay.mesh);

    this.nTraj = Math.min(pb.nTrajectories, MAX_VEHICLES);
    this.cursors = new Int32Array(this.nTraj).fill(-1);
    sharedCar ??= carGeometry();
    sharedBus ??= busGeometry();
    sharedBar ??= new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
    this.shuttleColor = new THREE.Color(opts.accent);

    // models: prop library variants when available, else the procedural car / bus
    const v = opts.vehicles ?? {};
    const carModels = v.cars?.length ? v.cars : [{ id: 'car', geometry: sharedCar, share: 1 }];
    const models: Array<{ geometry: THREE.BufferGeometry; kind: 'car' | 'school_bus' | 'shuttle'; name: string }> = [
      ...carModels.map((c) => ({ geometry: c.geometry, kind: 'car' as const, name: `cars:${c.id}` })),
      { geometry: v.bus ?? sharedBus, kind: 'school_bus', name: 'buses' },
      { geometry: v.shuttle ?? v.bus ?? sharedBus, kind: 'shuttle', name: 'shuttles' },
    ];
    const shares = carModels.map((c) => c.share);
    this.fleetOf = new Uint8Array(this.nTraj);
    this.paintOf = new Uint16Array(this.nTraj);
    const counts = new Array<number>(models.length).fill(0);
    const nPaint = Math.max(1, v.paint?.length ?? 1);
    for (let i = 0; i < this.nTraj; i++) {
      const kind = pb.header.kinds?.[String(pb.trajKind[i])] ?? 'car';
      let f: number;
      if (kind === 'school_bus') f = carModels.length;
      else if (kind === 'shuttle') f = carModels.length + 1;
      else f = pickVariant(shares, hash01(i, 1));
      this.fleetOf[i] = f;
      this.paintOf[i] = Math.floor(hash01(i, 2) * nPaint) % nPaint;
      counts[f]!++;
    }
    models.forEach((m, k) => {
      const n = Math.max(1, counts[k]!);
      const mesh = new THREE.InstancedMesh(m.geometry, vehicleMaterial(), n);
      mesh.castShadow = opts.castShadows ?? false;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
      mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.count = 0;
      mesh.name = m.name;
      this.group.add(mesh);
      this.fleets.push({ mesh, kind: m.kind, n: 0 });
    });

    // per-model footprint (length along +x, width along z) for the contact shadows
    this.fleetSize = this.fleets.map((f) => {
      f.mesh.geometry.computeBoundingBox();
      const b = f.mesh.geometry.boundingBox!;
      return [Math.max(1, b.max.x - b.min.x), Math.max(1, b.max.z - b.min.z)];
    });
    sharedBlob ??= new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
    this.blobs = new THREE.InstancedMesh(
      sharedBlob,
      new THREE.MeshBasicMaterial({ map: blobTexture(), color: 0x000000, transparent: true, opacity: 0.55, depthWrite: false, fog: false, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 }),
      Math.max(1, this.nTraj),
    );
    this.blobs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.blobs.frustumCulled = false;
    this.blobs.count = 0;
    this.blobs.renderOrder = 3;
    this.blobs.name = 'vehicle-shadows';
    this.group.add(this.blobs);

    const nv = this.nTraj * TRAIL_SEGMENTS * 2;
    this.trailPos = new Float32Array(nv * 3);
    this.trailCol = new Float32Array(nv * 3);
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(this.trailPos, 3).setUsage(THREE.DynamicDrawUsage));
    tg.setAttribute('color', new THREE.BufferAttribute(this.trailCol, 3).setUsage(THREE.DynamicDrawUsage));
    tg.setDrawRange(0, 0);
    this.trails = new THREE.LineSegments(
      tg,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        transparent: true,
        depthWrite: false,
        fog: false,
        toneMapped: false,
      }),
    );
    this.trails.frustumCulled = false;
    this.trails.visible = false;
    this.trails.renderOrder = 5;
    this.trails.name = 'ghost-trails';
    this.group.add(this.trails);

    this.buildQueueBars(schools);
  }

  private buildQueueBars(schools: School[]): void {
    const ids = this.pb.header.entrance_ids ?? [];
    const byKey = new Map<string, { x: number; y: number; z: number; curb: number }>();
    for (const s of schools) {
      for (const e of s.entrances) {
        const v = { x: e.x, y: e.y ?? s.y ?? 0, z: e.z, curb: e.curb_spots };
        byKey.set(e.key ?? `${s.id}/${e.id}`, v);
        byKey.set(`${s.id}/${e.id}`, v);
      }
    }
    ids.forEach((id, k) => {
      const p = byKey.get(id);
      if (p) this.barEntrances.push({ k, ...p, y0: p.y });
    });
    if (!this.barEntrances.length) return;
    this.bars = new THREE.InstancedMesh(
      sharedBar!,
      new THREE.MeshLambertMaterial({ color: 0xffffff, emissive: 0x222222 }),
      this.barEntrances.length,
    );
    this.bars.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(this.barEntrances.length * 3), 3);
    this.bars.frustumCulled = false;
    this.bars.name = 'queue-bars';
    this.group.add(this.bars);
    if (this.opts.showLabels) {
      for (const e of this.barEntrances) {
        const el = document.createElement('div');
        el.className = 'queue-label';
        const o = new CSS2DObject(el);
        o.center.set(0.5, 1);
        o.position.set(e.x, e.y, e.z);
        this.group.add(o);
        this.barLabels.push(o);
      }
    }
  }

  /** Visual style for the base map underneath (our open-data meshes or photoreal tiles). */
  setStyle(style: 'open' | 'photoreal'): void {
    this.overlay.setStyle(style);
  }

  /** Re-seat queue bars on the ground (photoreal tiles may sit a little off our DEM). */
  setGroundHeights(fn: (x: number, z: number, fallback: number) => number): void {
    for (const e of this.barEntrances) e.y = fn(e.x, e.z, e.y0);
  }

  /** Realistic paint colors (close up, with prop models) or the speed legend colors. */
  setColorMode(mode: 'speed' | 'paint'): void {
    this.colorMode = mode;
  }

  /** Instances drawn per model (debugging / tests). */
  get fleetCounts(): Record<string, number> {
    return Object.fromEntries(this.fleets.map((f) => [f.mesh.name, f.mesh.count]));
  }

  setGhost(on: boolean): void {
    this.ghost = on;
    this.trails.visible = on;
  }

  setCastShadows(on: boolean): void {
    for (const f of this.fleets) f.mesh.castShadow = on;
  }

  setLabelsVisible(on: boolean): void {
    for (const l of this.barLabels) l.visible = on;
  }

  /** Locate the trajectory point index active at time t, or -1. */
  private locate(i: number, t: number): number {
    const pb = this.pb;
    const s = pb.trajOffsets[i]!;
    const e = pb.trajOffsets[i + 1]!;
    if (e <= s) return -1;
    if (t < pb.trajEnterS[s]! || t >= pb.trajExitS[e - 1]!) return -1;
    let c = this.cursors[i]!;
    if (c < s || c >= e) c = s;
    while (c < e - 1 && pb.trajExitS[c]! <= t) c++;
    while (c > s && pb.trajEnterS[c]! > t) c--;
    this.cursors[i] = c;
    return c;
  }

  private place(c: number, t: number, out: PosOut): boolean {
    const pb = this.pb;
    const edge = pb.trajEdge[c]!;
    if (edge < 0 || edge >= this.net.nEdges) return false;
    const en = pb.trajEnterS[c]!;
    const ex = pb.trajExitS[c]!;
    const f = ex > en ? (t - en) / (ex - en) : 1;
    this.net.pointAt(edge, f, out);
    // keep to the right-hand lane
    out.x += -out.dz * LANE_OFFSET_M;
    out.z += out.dx * LANE_OFFSET_M;
    return true;
  }

  /** Update everything for sim time t. `scale` enlarges vehicles when zoomed out. */
  update(t: number, scale: number): void {
    const pb = this.pb;
    const cfg = pb.header;
    if (pb.nBins > 0) {
      const { b0, b1, w } = binBlend(t, cfg);
      this.overlay.setFromBins(pb.edgeVc, b0, b1, w);
    }

    for (const f of this.fleets) f.n = 0;
    const bm = this.blobs.instanceMatrix.array as Float32Array;
    let nBlob = 0;
    const paint = this.opts.vehicles?.paint;
    const usePaint = this.colorMode === 'paint' && !!paint?.length;
    let nt = 0;
    const out = this.tmp;
    const col = this.col;
    const s = scale;
    for (let i = 0; i < this.nTraj; i++) {
      const c = this.locate(i, t);
      if (c < 0) continue;
      if (!this.place(c, t, out)) continue;
      const fleet = this.fleets[this.fleetOf[i]!]!;
      const m = fleet.mesh.instanceMatrix.array as Float32Array;
      const cc = fleet.mesh.instanceColor!.array as Float32Array;
      const k = fleet.n++;
      const o = k * 16;
      const dx = out.dx;
      const dz = out.dz;
      m[o] = dx * s;
      m[o + 1] = 0;
      m[o + 2] = dz * s;
      m[o + 3] = 0;
      m[o + 4] = 0;
      m[o + 5] = s;
      m[o + 6] = 0;
      m[o + 7] = 0;
      m[o + 8] = -dz * s;
      m[o + 9] = 0;
      m[o + 10] = dx * s;
      m[o + 11] = 0;
      m[o + 12] = out.x;
      m[o + 13] = out.y + 0.3;
      m[o + 14] = out.z;
      m[o + 15] = 1;
      // contact shadow: same pose, flattened, a bit larger than the footprint
      const sz = this.fleetSize[this.fleetOf[i]!]!;
      const bo = nBlob++ * 16;
      const sl = sz[0] * 1.25 * s;
      const sw = sz[1] * 1.6 * s;
      bm[bo] = dx * sl;
      bm[bo + 1] = 0;
      bm[bo + 2] = dz * sl;
      bm[bo + 3] = 0;
      bm[bo + 4] = 0;
      bm[bo + 5] = 1;
      bm[bo + 6] = 0;
      bm[bo + 7] = 0;
      bm[bo + 8] = -dz * sw;
      bm[bo + 9] = 0;
      bm[bo + 10] = dx * sw;
      bm[bo + 11] = 0;
      bm[bo + 12] = out.x;
      bm[bo + 13] = out.y + 0.33;
      bm[bo + 14] = out.z;
      bm[bo + 15] = 1;
      const dur = pb.trajExitS[c]! - pb.trajEnterS[c]!;
      const kph = dur > 0 ? (this.net.length[pb.trajEdge[c]!]! / dur) * 3.6 : 0;
      if (fleet.kind !== 'car') {
        // prop school buses carry their own yellow paint (not tintable); shuttles get the plan / baseline accent
        const bcol = fleet.kind === 'shuttle' ? this.shuttleColor : this.opts.vehicles?.bus ? WHITE : this.busColor;
        cc[k * 3] = bcol.r;
        cc[k * 3 + 1] = bcol.g;
        cc[k * 3 + 2] = bcol.b;
      } else if (usePaint) {
        const pc = paint![this.paintOf[i]!]!;
        cc[k * 3] = pc[0];
        cc[k * 3 + 1] = pc[1];
        cc[k * 3 + 2] = pc[2];
      } else {
        speedColor(kph, col);
        // ramp colors are sRGB; instance colors are linear
        cc[k * 3] = col[0] * col[0];
        cc[k * 3 + 1] = col[1] * col[1];
        cc[k * 3 + 2] = col[2] * col[2];
      }

      if (this.ghost) nt = this.writeTrail(i, c, t, kph, nt);
    }
    this.blobs.count = nBlob;
    this.blobs.instanceMatrix.needsUpdate = true;
    for (const f of this.fleets) {
      f.mesh.count = f.n;
      f.mesh.instanceMatrix.needsUpdate = true;
      f.mesh.instanceColor!.needsUpdate = true;
    }
    if (this.ghost) {
      const g = this.trails.geometry;
      g.setDrawRange(0, nt * 2);
      (g.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
      (g.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;
    }

    this.updateQueues(t, scale);
  }

  private writeTrail(i: number, c: number, t: number, kph: number, nt: number): number {
    const pb = this.pb;
    const s = pb.trajOffsets[i]!;
    const tp = this.trailPos;
    const tc = this.trailCol;
    const out = this.tmp;
    let px = out.x;
    let py = out.y + 1.5;
    let pz = out.z;
    speedColor(kph, this.col);
    // warm glow: blend speed color toward amber-white
    // HDR values so the bloom pass makes them glow
    const r0 = (0.55 + 0.45 * this.col[0]) * 2.2;
    const g0 = (0.35 + 0.4 * this.col[1]) * 2.2;
    const b0 = (0.15 + 0.3 * this.col[2]) * 2.2;
    let cc = c;
    for (let k = 1; k <= TRAIL_SEGMENTS; k++) {
      const tk = t - k * TRAIL_STEP_S;
      if (tk < pb.trajEnterS[s]!) break;
      while (cc > s && pb.trajEnterS[cc]! > tk) cc--;
      if (!this.place(cc, Math.min(tk, pb.trajExitS[cc]!), out)) break;
      const fade0 = 1 - (k - 1) / TRAIL_SEGMENTS;
      const fade1 = 1 - k / TRAIL_SEGMENTS;
      const o = nt * 6;
      tp[o] = px;
      tp[o + 1] = py;
      tp[o + 2] = pz;
      tp[o + 3] = out.x;
      tp[o + 4] = out.y + 1.5;
      tp[o + 5] = out.z;
      tc[o] = r0 * fade0;
      tc[o + 1] = g0 * fade0;
      tc[o + 2] = b0 * fade0;
      tc[o + 3] = r0 * fade1;
      tc[o + 4] = g0 * fade1;
      tc[o + 5] = b0 * fade1;
      px = out.x;
      py = out.y + 1.5;
      pz = out.z;
      nt++;
    }
    return nt;
  }

  private barMat = new THREE.Matrix4();
  private barColor = new THREE.Color();

  private updateQueues(t: number, scale: number): void {
    if (!this.bars) return;
    const bin = binIndex(t, this.pb.header);
    const w = 7 * Math.max(1, scale * 0.6);
    this.barEntrances.forEach((e, idx) => {
      const q = queueAt(this.pb, bin, e.k);
      // ~1 m per queued car, softly capped (the label carries the exact count)
      const h = (2 + 70 * Math.tanh(q / 70)) * Math.max(1, scale * 0.5);
      this.barMat.makeScale(w, h, w);
      this.barMat.setPosition(e.x, e.y, e.z);
      this.bars!.setMatrixAt(idx, this.barMat);
      const c = queueColor(q, e.curb, this.col);
      this.barColor.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
      this.bars!.setColorAt(idx, this.barColor);
      const label = this.barLabels[idx];
      if (label) {
        label.position.set(e.x, e.y + h + 4, e.z);
        const n = Math.round(q);
        const txt = `${n} car${n === 1 ? '' : 's'}`;
        if (label.element.textContent !== txt) label.element.textContent = txt;
        label.element.style.opacity = n >= 1 ? '1' : '0';
      }
    });
    this.bars.instanceMatrix.needsUpdate = true;
    if (this.bars.instanceColor) this.bars.instanceColor.needsUpdate = true;
  }

  dispose(): void {
    this.overlay.dispose();
    for (const f of this.fleets) {
      (f.mesh.material as THREE.Material).dispose();
      f.mesh.dispose();
    }
    (this.blobs.material as THREE.Material).dispose();
    this.blobs.dispose();
    this.trails.geometry.dispose();
    (this.trails.material as THREE.Material).dispose();
    if (this.bars) {
      (this.bars.material as THREE.Material).dispose();
      this.bars.dispose();
    }
    for (const l of this.barLabels) l.element.remove();
    this.group.removeFromParent();
    this.group.clear();
  }
}
