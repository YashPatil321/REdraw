/**
 * Drape the traffic overlay onto the photoreal surface. Our road polylines carry
 * DEM elevations (10 m 3DEP); the photo mesh has the real road surface, which
 * can differ by meters on cuts, fills and bridges even after the global
 * vertical calibration. Near the camera, sample the tile height under each
 * polyline point (BVH raycasts, a few ms per frame) and store the residual in
 * `net.yAdj`, which cars (pointAt) and the ribbon shader both add.
 *
 * Robustness: samples far above the expected road height are trees, cars or
 * wires (rejected); far below are holes (rejected). Rejected points take the
 * value interpolated from accepted neighbors on the same edge, then a 3-point
 * median removes spikes.
 */

import type { RoadNetwork } from '../traffic/network';

export const ACCEPT_ABOVE_M = 2.5;
export const ACCEPT_BELOW_M = 6;

/** Residuals for one polyline from raw samples (null = no hit). Pure; unit tested. */
export function cleanResiduals(raw: Array<number | null>): Float32Array {
  const n = raw.length;
  const ok = raw.map((r) => r !== null && r < ACCEPT_ABOVE_M && r > -ACCEPT_BELOW_M);
  const v = new Float32Array(n);
  const good: number[] = [];
  for (let i = 0; i < n; i++) if (ok[i]) good.push(i);
  if (!good.length) return v; // nothing usable: no adjustment
  for (let i = 0; i < n; i++) {
    if (ok[i]) {
      v[i] = raw[i]!;
      continue;
    }
    // interpolate between nearest accepted neighbors
    let lo = -1;
    let hi = -1;
    for (let j = i - 1; j >= 0; j--) if (ok[j]) {
      lo = j;
      break;
    }
    for (let j = i + 1; j < n; j++) if (ok[j]) {
      hi = j;
      break;
    }
    if (lo >= 0 && hi >= 0) v[i] = raw[lo]! + ((raw[hi]! - raw[lo]!) * (i - lo)) / (hi - lo);
    else v[i] = raw[lo >= 0 ? lo : hi]!;
  }
  if (n >= 3) {
    const m = Float32Array.from(v);
    for (let i = 1; i < n - 1; i++) {
      const a = v[i - 1]!;
      const b = v[i]!;
      const c = v[i + 1]!;
      m[i] = Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));
    }
    return m;
  }
  return v;
}

export class Drape {
  /** per edge: camera distance when draped (Infinity = never) */
  private doneAt: Float32Array;
  private queue: number[] = [];
  private qCenter = { x: Infinity, z: Infinity, dist: Infinity };
  private dirty = false;
  /** bumps when yAdj changed (consumers re-upload the texture) */
  version = 0;

  constructor(
    private net: RoadNetwork,
    /** tile surface height at x, z (scene y, calibrated), or null */
    private sample: (x: number, z: number) => number | null,
  ) {
    this.doneAt = new Float32Array(net.nEdges).fill(Infinity);
  }

  reset(): void {
    this.net.yAdj.fill(0);
    this.doneAt.fill(Infinity);
    this.queue = [];
    this.qCenter = { x: Infinity, z: Infinity, dist: Infinity };
    this.version++;
  }

  /** The global offset changed by `delta`: draped residuals shift the opposite way. */
  shiftAll(delta: number): void {
    if (!delta) return;
    const net = this.net;
    for (let e = 0; e < net.nEdges; e++) {
      if (this.doneAt[e] === Infinity) continue;
      const s = net.ptStart[e]!;
      const n = net.ptCount[e]!;
      for (let k = 0; k < n; k++) net.yAdj[s + k]! -= delta;
    }
    this.version++;
  }

  /** Forget work done from far away (tiles there were coarse). */
  invalidateCoarse(currentDist: number): void {
    for (let e = 0; e < this.doneAt.length; e++) if (this.doneAt[e]! > currentDist * 2.5) this.doneAt[e] = Infinity;
  }

  private rebuildQueue(cx: number, cz: number, camDist: number): void {
    const net = this.net;
    const R = Math.min(2500, Math.max(350, camDist * 1.6));
    const cand: Array<[number, number]> = [];
    for (let e = 0; e < net.nEdges; e++) {
      if (this.doneAt[e]! <= camDist * 2.5) continue;
      const s = net.ptStart[e]!;
      const n = net.ptCount[e]!;
      if (!n) continue;
      const m = s + (n >> 1);
      const d = Math.hypot(net.pts[m * 3]! - cx, net.pts[m * 3 + 2]! - cz);
      if (d < R) cand.push([e, d]);
    }
    cand.sort((a, b) => a[1] - b[1]);
    this.queue = cand.map((c) => c[0]);
    this.qCenter = { x: cx, z: cz, dist: camDist };
  }

  /** Work for up to `budgetMs`; returns true if adjustments changed. */
  step(cx: number, cz: number, camDist: number, budgetMs = 3): boolean {
    const q = this.qCenter;
    if (Math.hypot(cx - q.x, cz - q.z) > Math.max(80, camDist * 0.3) || camDist < q.dist * 0.6 || camDist > q.dist * 1.8) {
      this.rebuildQueue(cx, cz, camDist);
    }
    const net = this.net;
    const t0 = performance.now();
    let changed = false;
    while (this.queue.length && performance.now() - t0 < budgetMs) {
      const e = this.queue.shift()!;
      const s = net.ptStart[e]!;
      const n = net.ptCount[e]!;
      const raw: Array<number | null> = [];
      for (let k = 0; k < n; k++) {
        const i = s + k;
        const h = this.sample(net.pts[i * 3]!, net.pts[i * 3 + 2]!);
        raw.push(h === null ? null : h - net.pts[i * 3 + 1]!);
      }
      const v = cleanResiduals(raw);
      for (let k = 0; k < n; k++) net.yAdj[s + k] = v[k]!;
      this.doneAt[e] = camDist;
      changed = true;
    }
    if (changed) {
      this.version++;
      this.dirty = true;
    }
    return changed;
  }

  /** True once after a change (for throttled texture uploads). */
  takeDirty(): boolean {
    const d = this.dirty;
    this.dirty = false;
    return d;
  }
}
