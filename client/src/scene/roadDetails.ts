/**
 * Open-data street detail built from the road network (GET /world/network):
 * painted markings (double yellow centre lines on two-way collectors and
 * arterials, dashed white lane lines, solid edge lines) plus concrete
 * sidewalks with a curb. Purely visual; drawn only in open-data mode and
 * faded out with distance (sub-pixel lines shimmer from high up).
 *
 * Heights follow the pipeline's ribbon rule (roads lifted 0.3 m above the
 * higher of the terrain at the ribbon edge and at the centre line), so paint
 * sits on the asphalt and sidewalks sit beside it.
 */

import * as THREE from 'three';
import type { RoadNetwork } from '../traffic/network';

/** Mirrors data/config/assumptions.yaml roads.lane_width_m (rendering only). */
export const LANE_W = 3.6;
const RIBBON_LIFT = 0.3;
const PAINT_LIFT = 0.04;
const CURB_H = 0.15;
const SIDEWALK_W = 1.8;
const PARKWAY_W = 0.6;

const NO_SIDEWALK = new Set(['motorway', 'motorway_link', 'trunk', 'trunk_link', 'service', 'track', 'unclassified']);
const NO_CENTER_LINE = new Set(['residential', 'service', 'living_street', 'unclassified', 'track']);

export interface RoadSpan {
  /** directed edge index used for the geometry */
  edge: number;
  /** lanes in the geometry direction */
  lanesFwd: number;
  /** lanes in the reverse direction (0 for one-way) */
  lanesBack: number;
  highway: string;
}

/** Pair directed edges into roads (two-way pairs drawn once). Pure; unit tested. */
export function pairRoads(net: Pick<RoadNetwork, 'edges' | 'length'>): RoadSpan[] {
  const byKey = new Map<string, number>();
  const out: RoadSpan[] = [];
  const used = new Set<number>();
  net.edges.forEach((e, i) => byKey.set(`${e.u}>${e.v}>${Math.round(net.length[i]!)}`, i));
  net.edges.forEach((e, i) => {
    if (used.has(i)) return;
    used.add(i);
    const rev = byKey.get(`${e.v}>${e.u}>${Math.round(net.length[i]!)}`);
    let back = 0;
    if (rev !== undefined && rev !== i && !used.has(rev)) {
      used.add(rev);
      back = Math.max(1, net.edges[rev]!.lanes ?? 1);
    }
    out.push({ edge: i, lanesFwd: Math.max(1, e.lanes ?? 1), lanesBack: back, highway: e.highway ?? '' });
  });
  return out;
}

type HeightFn = (x: number, z: number) => number | null;

interface Builder {
  pos: number[];
  col: number[];
  dist: number[];
  dash: number[];
  idx: number[];
}

function newBuilder(): Builder {
  return { pos: [], col: [], dist: [], dash: [], idx: [] };
}

/**
 * Strip along a polyline between lateral offsets o0..o1 (meters, + = right of
 * travel), at heights from `yAt(i, offset)`.
 */
function strip(
  b: Builder,
  pts: Float32Array,
  s: number,
  n: number,
  o0: number,
  o1: number,
  yAt: (i: number, x: number, z: number, cy: number) => number,
  rgb: [number, number, number],
  dashed: number,
): void {
  if (n < 2) return;
  let along = 0;
  const base = b.pos.length / 3;
  for (let k = 0; k < n; k++) {
    const i = s + k;
    const x = pts[i * 3]!;
    const y = pts[i * 3 + 1]!;
    const z = pts[i * 3 + 2]!;
    // averaged direction at the vertex
    const ip = s + Math.max(0, k - 1);
    const inx = s + Math.min(n - 1, k + 1);
    let dx = pts[inx * 3]! - pts[ip * 3]!;
    let dz = pts[inx * 3 + 2]! - pts[ip * 3 + 2]!;
    const L = Math.hypot(dx, dz) || 1;
    dx /= L;
    dz /= L;
    if (k > 0) along += Math.hypot(x - pts[(i - 1) * 3]!, z - pts[(i - 1) * 3 + 2]!);
    // right-hand normal of travel: (-dz, dx)
    const rx = -dz;
    const rz = dx;
    for (const o of [o0, o1]) {
      const px = x + rx * o;
      const pz = z + rz * o;
      b.pos.push(px, yAt(i, px, pz, y), pz);
      b.col.push(rgb[0], rgb[1], rgb[2]);
      b.dist.push(along);
      b.dash.push(dashed);
    }
  }
  for (let k = 0; k < n - 1; k++) {
    const a = base + k * 2;
    b.idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
  }
}

function toGeometry(b: Builder): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(b.pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(b.col, 3));
  g.setAttribute('aDist', new THREE.Float32BufferAttribute(b.dist, 1));
  g.setAttribute('aDash', new THREE.Float32BufferAttribute(b.dash, 1));
  g.setIndex(b.idx);
  g.computeBoundingSphere();
  return g;
}

const WHITE: [number, number, number] = [0.86, 0.86, 0.83];
const YELLOW: [number, number, number] = [0.86, 0.66, 0.16];
const CONCRETE: [number, number, number] = [0.66, 0.64, 0.6];
const CURB: [number, number, number] = [0.74, 0.73, 0.7];

export class RoadDetails {
  readonly group = new THREE.Group();
  private paintMat: THREE.MeshLambertMaterial;
  private walkMat: THREE.MeshLambertMaterial;
  readonly uniforms = { uFadeFar: { value: 900 } };

  constructor(net: RoadNetwork, height: HeightFn) {
    this.group.name = 'road-details';
    const paint = newBuilder();
    const walk = newBuilder();
    const roads = pairRoads(net);
    for (const r of roads) {
      const s = net.ptStart[r.edge]!;
      const n = net.ptCount[r.edge]!;
      if (n < 2) continue;
      const hw = r.highway;
      // ribbon edges: the pipeline centres the ribbon on the polyline with the combined width
      const total = r.lanesFwd + r.lanesBack;
      const half = (Math.max(total, 1) * LANE_W) / 2;
      // ribbon height at lateral offset o: interpolate the two edge heights (pipeline rule)
      const edgeY = (cy: number, x: number, z: number): number => Math.max(height(x, z) ?? cy, cy) + RIBBON_LIFT;
      const ribbonY = (i: number, o: number, cy: number): number => {
        const x = net.pts[i * 3]!;
        const z = net.pts[i * 3 + 2]!;
        const ip = Math.max(s, i - 1);
        const inx = Math.min(s + n - 1, i + 1);
        let dx = net.pts[inx * 3]! - net.pts[ip * 3]!;
        let dz = net.pts[inx * 3 + 2]! - net.pts[ip * 3 + 2]!;
        const L = Math.hypot(dx, dz) || 1;
        dx /= L;
        dz /= L;
        const yr = edgeY(cy, x - dz * half, z + dx * half);
        const yl = edgeY(cy, x + dz * half, z - dx * half);
        const t = (o + half) / (2 * half);
        return yl + (yr - yl) * t;
      };
      const paintAt =
        (o: number) =>
        (i: number, _x: number, _z: number, cy: number): number =>
          ribbonY(i, o, cy) + PAINT_LIFT;
      const line = (o: number, w: number, rgb: [number, number, number], dashed: number): void =>
        strip(paint, net.pts, s, n, o - w / 2, o + w / 2, paintAt(o), rgb, dashed);

      const major = !NO_CENTER_LINE.has(hw);
      if (r.lanesBack > 0) {
        // two-way: the ribbon is centred on the polyline; forward lanes on the right (+),
        // so the centre line sits where the forward lanes end
        const c = ((r.lanesBack - r.lanesFwd) * LANE_W) / 2;
        if (major) {
          line(c - 0.12, 0.1, YELLOW, 0);
          line(c + 0.12, 0.1, YELLOW, 0);
        }
        for (let k = 1; k < r.lanesFwd; k++) line(c + k * LANE_W, 0.1, WHITE, 1);
        for (let k = 1; k < r.lanesBack; k++) line(c - k * LANE_W, 0.1, WHITE, 1);
      } else {
        // one-way carriageway centred on the polyline
        for (let k = 1; k < r.lanesFwd; k++) line(-half + k * LANE_W, 0.1, WHITE, 1);
        if (major) line(-half + 0.35, 0.1, YELLOW, 0);
      }
      if (major) {
        line(half - 0.3, 0.12, WHITE, 0);
        if (r.lanesBack > 0) line(-half + 0.3, 0.12, WHITE, 0);
      }
      // sidewalks with a curb, both sides of two-way streets, right side of one-ways
      if (!NO_SIDEWALK.has(hw)) {
        const sides = r.lanesBack > 0 ? [1, -1] : [1];
        for (const side of sides) {
          const o0 = side * (half + PARKWAY_W);
          const o1 = side * (half + PARKWAY_W + SIDEWALK_W);
          const walkY = (i: number, x: number, z: number, cy: number): number =>
            Math.max(height(x, z) ?? cy, ribbonY(i, side * half, cy) - RIBBON_LIFT) + RIBBON_LIFT + CURB_H;
          strip(walk, net.pts, s, n, Math.min(o0, o1), Math.max(o0, o1), walkY, CONCRETE, 0);
          // curb face from the asphalt edge up to the sidewalk (thin strip, near vertical)
          const c0 = side * half;
          const c1 = side * (half + PARKWAY_W);
          const curbY = (i: number, x: number, z: number, cy: number): number => {
            const atEdge = Math.abs(Math.hypot(x - net.pts[i * 3]!, z - net.pts[i * 3 + 2]!) - half) < 0.05;
            return atEdge ? ribbonY(i, side * half, cy) : walkY(i, x, z, cy);
          };
          strip(walk, net.pts, s, n, Math.min(c0, c1), Math.max(c0, c1), curbY, CURB, 0);
        }
      }
    }
    const fade = this.uniforms;
    const patch = (mat: THREE.Material, dashed: boolean): void => {
      mat.onBeforeCompile = (shader) => {
        Object.assign(shader.uniforms, fade);
        shader.vertexShader = shader.vertexShader
          .replace('#include <common>', '#include <common>\nattribute float aDist;\nattribute float aDash;\nvarying float vDist;\nvarying float vDash;\nvarying float vCamD;')
          .replace('#include <begin_vertex>', '#include <begin_vertex>\nvDist = aDist;\nvDash = aDash;\nvCamD = length((modelMatrix * vec4(position, 1.0)).xyz - cameraPosition);');
        shader.fragmentShader = shader.fragmentShader
          .replace('#include <common>', '#include <common>\nuniform float uFadeFar;\nvarying float vDist;\nvarying float vDash;\nvarying float vCamD;')
          .replace(
            '#include <color_fragment>',
            `#include <color_fragment>
            ${dashed ? 'if (vDash > 0.5 && fract(vDist / 12.0) > 0.25) discard;' : ''}
            float rdFade = 1.0 - smoothstep(uFadeFar * 0.6, uFadeFar, vCamD);
            if (rdFade <= 0.01) discard;
            diffuseColor.a *= rdFade;
            // worn paint / concrete grain
            float g = fract(sin(floor(vDist * 3.0) * 12.9898) * 43758.5453);
            diffuseColor.rgb *= 0.9 + 0.1 * g;`,
          );
      };
      mat.customProgramCacheKey = () => `road-details-${dashed ? 1 : 0}`;
    };
    this.paintMat = new THREE.MeshLambertMaterial({ vertexColors: true, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -8 });
    patch(this.paintMat, true);
    this.walkMat = new THREE.MeshLambertMaterial({ vertexColors: true, transparent: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 });
    patch(this.walkMat, false);
    const pm = new THREE.Mesh(toGeometry(paint), this.paintMat);
    pm.name = 'road-markings';
    pm.renderOrder = 2;
    pm.receiveShadow = true;
    const wm = new THREE.Mesh(toGeometry(walk), this.walkMat);
    wm.name = 'sidewalks';
    wm.renderOrder = 1;
    wm.receiveShadow = true;
    this.group.add(wm, pm);
  }

  /** Lines fade out by this camera distance (m). */
  setFadeDistance(d: number): void {
    this.uniforms.uFadeFar.value = d;
  }

  dispose(): void {
    for (const c of this.group.children) (c as THREE.Mesh).geometry.dispose();
    this.paintMat.dispose();
    this.walkMat.dispose();
    this.group.removeFromParent();
  }
}
