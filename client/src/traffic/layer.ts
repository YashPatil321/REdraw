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
import { RoadOverlay } from './roadOverlay';

export const MAX_VEHICLES = 3000;
const TRAIL_SEGMENTS = 10;
const TRAIL_STEP_S = 6; // sim seconds between trail samples
const LANE_OFFSET_M = 2.2;

const BUS_KINDS = new Set(['shuttle', 'school_bus']);

/** Box with a constant `aLight` value (0 body, 1 headlight, 2 taillight, 3 dark glass). */
function part(w: number, h: number, d: number, x: number, y: number, z: number, light: number): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(w, h, d).translate(x, y, z).toNonIndexed();
  g.deleteAttribute('uv');
  g.setAttribute('aLight', new THREE.BufferAttribute(new Float32Array(g.getAttribute('position').count).fill(light), 1));
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

function vehicleMaterial(): THREE.MeshLambertMaterial {
  const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, vehicleLightUniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float aLight;\nvarying float vLight;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvLight = aLight;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vLight;\nuniform float uHead;\nuniform float uTail;')
      .replace('#include <color_fragment>', '#include <color_fragment>\nif (vLight > 2.5) diffuseColor.rgb = diffuseColor.rgb * 0.25 + vec3(0.04, 0.05, 0.07);')
      .replace(
        '#include <opaque_fragment>',
        `if (vLight > 0.5 && vLight < 1.5) outgoingLight = vec3(1.0, 0.93, 0.78) * uHead;
        else if (vLight > 1.5 && vLight < 2.5) outgoingLight = vec3(1.0, 0.06, 0.04) * uTail;
        #include <opaque_fragment>`,
      );
  };
  mat.customProgramCacheKey = () => 'redraw-vehicle';
  return mat;
}

let sharedCar: THREE.BufferGeometry | null = null;
let sharedBus: THREE.BufferGeometry | null = null;
let sharedBar: THREE.BufferGeometry | null = null;

/** Optional vehicle meshes from the props library (fallback: procedural boxes). */
export interface VehicleGeometries {
  car?: THREE.BufferGeometry;
  bus?: THREE.BufferGeometry;
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
  private cars: THREE.InstancedMesh;
  private buses: THREE.InstancedMesh;
  private bars: THREE.InstancedMesh | null = null;
  private barLabels: CSS2DObject[] = [];
  private barEntrances: Array<{ k: number; x: number; y: number; z: number; curb: number }> = [];
  private trails: THREE.LineSegments;
  private trailPos: Float32Array;
  private trailCol: Float32Array;
  private cursors: Int32Array;
  private isBus: Uint8Array;
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
    this.overlay = new RoadOverlay(overlayGeometry, Math.min(net.nEdges, pb.nEdges));
    this.group.add(this.overlay.mesh);

    this.nTraj = Math.min(pb.nTrajectories, MAX_VEHICLES);
    this.cursors = new Int32Array(this.nTraj).fill(-1);
    this.isBus = new Uint8Array(this.nTraj);
    let nBus = 0;
    for (let i = 0; i < this.nTraj; i++) {
      const kind = pb.header.kinds?.[String(pb.trajKind[i])] ?? 'car';
      if (BUS_KINDS.has(kind)) {
        this.isBus[i] = 1;
        nBus++;
      }
    }
    sharedCar ??= carGeometry();
    sharedBus ??= busGeometry();
    sharedBar ??= new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);

    const carGeo = opts.vehicles?.car ?? sharedCar;
    const busGeo = opts.vehicles?.bus ?? sharedBus;
    this.cars = new THREE.InstancedMesh(carGeo, vehicleMaterial(), Math.max(1, this.nTraj - nBus));
    this.cars.castShadow = opts.castShadows ?? false;
    this.cars.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.cars.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, this.nTraj - nBus) * 3), 3);
    this.cars.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.cars.frustumCulled = false;
    this.cars.count = 0;
    this.cars.name = 'cars';
    this.group.add(this.cars);

    this.shuttleColor = new THREE.Color(opts.accent);
    this.buses = new THREE.InstancedMesh(busGeo, vehicleMaterial(), Math.max(1, nBus));
    this.buses.castShadow = opts.castShadows ?? false;
    this.buses.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.buses.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, nBus) * 3), 3);
    this.buses.frustumCulled = false;
    this.buses.count = 0;
    this.buses.name = 'buses';
    this.group.add(this.buses);

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
      if (p) this.barEntrances.push({ k, ...p });
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

  setGhost(on: boolean): void {
    this.ghost = on;
    this.trails.visible = on;
  }

  setCastShadows(on: boolean): void {
    this.cars.castShadow = on;
    this.buses.castShadow = on;
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

    const cm = this.cars.instanceMatrix.array as Float32Array;
    const cc = this.cars.instanceColor!.array as Float32Array;
    const bm = this.buses.instanceMatrix.array as Float32Array;
    const bc = this.buses.instanceColor!.array as Float32Array;
    let nc = 0;
    let nb = 0;
    let nt = 0;
    const out = this.tmp;
    const col = this.col;
    const s = scale;
    for (let i = 0; i < this.nTraj; i++) {
      const c = this.locate(i, t);
      if (c < 0) continue;
      if (!this.place(c, t, out)) continue;
      const bus = this.isBus[i] === 1;
      const m = bus ? bm : cm;
      const k = bus ? nb++ : nc++;
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
      const dur = pb.trajExitS[c]! - pb.trajEnterS[c]!;
      const kph = dur > 0 ? (this.net.length[pb.trajEdge[c]!]! / dur) * 3.6 : 0;
      if (bus) {
        const kind = cfg.kinds?.[String(pb.trajKind[i])];
        const bcol = kind === 'shuttle' ? this.shuttleColor : this.busColor;
        bc[k * 3] = bcol.r;
        bc[k * 3 + 1] = bcol.g;
        bc[k * 3 + 2] = bcol.b;
      } else {
        speedColor(kph, col);
        // ramp colors are sRGB; instance colors are linear
        cc[k * 3] = col[0] * col[0];
        cc[k * 3 + 1] = col[1] * col[1];
        cc[k * 3 + 2] = col[2] * col[2];
      }

      if (this.ghost) nt = this.writeTrail(i, c, t, kph, nt);
    }
    this.cars.count = nc;
    this.buses.count = nb;
    this.cars.instanceMatrix.needsUpdate = true;
    this.cars.instanceColor!.needsUpdate = true;
    this.buses.instanceMatrix.needsUpdate = true;
    this.buses.instanceColor!.needsUpdate = true;
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
    const w = 10 * Math.max(1, scale * 0.8);
    this.barEntrances.forEach((e, idx) => {
      const q = queueAt(this.pb, bin, e.k);
      const h = (2 + q * 3) * Math.max(1, scale * 0.6);
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
    (this.cars.material as THREE.Material).dispose();
    (this.buses.material as THREE.Material).dispose();
    this.cars.dispose();
    this.buses.dispose();
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
