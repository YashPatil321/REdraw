/**
 * Road network geometry from GET /world/network: polylines with cumulative
 * lengths for interpolating vehicles, plus a uniform grid index for snapping
 * map clicks to the nearest edge / node. Snapping is UI placement only; the
 * server re-validates and resolves map inputs (thin client rule).
 */

import type { NetworkEdge, NetworkJson, NetworkNode } from '../types';

export interface EdgeHit {
  edge: number;
  /** projected point on the edge polyline */
  x: number;
  y: number;
  z: number;
  dist: number;
  /** fraction along the edge (0 at u, 1 at v) */
  frac: number;
}

export interface PosOut {
  x: number;
  y: number;
  z: number;
  /** unit heading in the xz plane */
  dx: number;
  dz: number;
}

/** Width of the per-point adjustment data texture. */
export const ADJ_TEX_W = 2048;

export class RoadNetwork {
  readonly nEdges: number;
  readonly edges: NetworkEdge[];
  readonly nodes: NetworkNode[];
  readonly nodeById = new Map<number, NetworkNode>();
  /** flat xyz of all edge polylines */
  readonly pts: Float32Array;
  /** per edge: first point index into pts/3 */
  readonly ptStart: Uint32Array;
  /** per edge: number of points */
  readonly ptCount: Uint32Array;
  /** cumulative length at each point (parallel to pts/3) */
  readonly cum: Float32Array;
  /** total polyline length per edge */
  readonly length: Float32Array;
  readonly bounds = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity };
  /** total polyline points */
  readonly nPts: number;
  /**
   * Per-point vertical adjustment (m) added to `pts` y. Zero in open-data mode;
   * in photoreal mode it drapes roads and cars onto the photo mesh (filled
   * by scene/drape.ts). Padded to a multiple of ADJ_TEX_W for a data texture.
   */
  readonly yAdj: Float32Array;
  /** per point: owning edge */
  readonly ptEdge: Uint32Array;

  private readonly cell: number;
  private segGrid = new Map<number, number[]>(); // cell -> [edge, segIdx, edge, segIdx, ...]
  private nodeGrid = new Map<number, number[]>(); // cell -> node array indices

  constructor(json: NetworkJson, cellSize = 100) {
    this.edges = json.edges;
    this.nodes = json.nodes;
    this.nEdges = json.edges.length;
    this.cell = cellSize;
    let totalPts = 0;
    for (const e of json.edges) totalPts += Math.floor(e.pts.length / 3);
    this.pts = new Float32Array(totalPts * 3);
    this.cum = new Float32Array(totalPts);
    this.ptStart = new Uint32Array(this.nEdges);
    this.ptCount = new Uint32Array(this.nEdges);
    this.length = new Float32Array(this.nEdges);
    this.nPts = totalPts;
    this.yAdj = new Float32Array(Math.max(1, Math.ceil(totalPts / ADJ_TEX_W)) * ADJ_TEX_W);
    this.ptEdge = new Uint32Array(totalPts);

    let p = 0;
    // edges are indexed by position in the array; `i` should match (contract)
    json.edges.forEach((e, ei) => {
      const n = Math.floor(e.pts.length / 3);
      this.ptStart[ei] = p;
      this.ptCount[ei] = n;
      this.ptEdge.fill(ei, p, p + n);
      let acc = 0;
      for (let k = 0; k < n; k++) {
        const x = e.pts[k * 3]!;
        const y = e.pts[k * 3 + 1]!;
        const z = e.pts[k * 3 + 2]!;
        this.pts[(p + k) * 3] = x;
        this.pts[(p + k) * 3 + 1] = y;
        this.pts[(p + k) * 3 + 2] = z;
        if (k > 0) {
          const px = e.pts[(k - 1) * 3]!;
          const pz = e.pts[(k - 1) * 3 + 2]!;
          acc += Math.hypot(x - px, z - pz);
          this.indexSegment(ei, k - 1, px, pz, x, z);
        }
        this.cum[p + k] = acc;
        if (x < this.bounds.minX) this.bounds.minX = x;
        if (x > this.bounds.maxX) this.bounds.maxX = x;
        if (z < this.bounds.minZ) this.bounds.minZ = z;
        if (z > this.bounds.maxZ) this.bounds.maxZ = z;
      }
      this.length[ei] = acc;
      p += n;
    });
    json.nodes.forEach((nd, idx) => {
      this.nodeById.set(nd.id, nd);
      const key = this.key(Math.floor(nd.x / this.cell), Math.floor(nd.z / this.cell));
      let arr = this.nodeGrid.get(key);
      if (!arr) this.nodeGrid.set(key, (arr = []));
      arr.push(idx);
    });
  }

  private key(cx: number, cz: number): number {
    // cells are within +-32k of the origin for any sane region
    return (cx + 32768) * 65536 + (cz + 32768);
  }

  private indexSegment(edge: number, seg: number, x0: number, z0: number, x1: number, z1: number): void {
    const c = this.cell;
    const cx0 = Math.floor(Math.min(x0, x1) / c);
    const cx1 = Math.floor(Math.max(x0, x1) / c);
    const cz0 = Math.floor(Math.min(z0, z1) / c);
    const cz1 = Math.floor(Math.max(z0, z1) / c);
    for (let cx = cx0; cx <= cx1; cx++) {
      for (let cz = cz0; cz <= cz1; cz++) {
        const k = this.key(cx, cz);
        let arr = this.segGrid.get(k);
        if (!arr) this.segGrid.set(k, (arr = []));
        arr.push(edge, seg);
      }
    }
  }

  /** Position and heading at fraction `frac` (0..1) along edge `e`. */
  pointAt(e: number, frac: number, out: PosOut): PosOut {
    const n = this.ptCount[e]!;
    const s = this.ptStart[e]!;
    if (n === 0) {
      out.x = out.y = out.z = 0;
      out.dx = 1;
      out.dz = 0;
      return out;
    }
    const L = this.length[e]!;
    const target = Math.min(Math.max(frac, 0), 1) * L;
    // binary search on cumulative length
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.cum[s + mid]! <= target) lo = mid;
      else hi = mid;
    }
    const a = s + lo;
    const b = s + Math.min(lo + 1, n - 1);
    const ca = this.cum[a]!;
    const segLen = this.cum[b]! - ca;
    const w = segLen > 1e-6 ? (target - ca) / segLen : 0;
    const ax = this.pts[a * 3]!;
    const ay = this.pts[a * 3 + 1]!;
    const az = this.pts[a * 3 + 2]!;
    const bx = this.pts[b * 3]!;
    const by = this.pts[b * 3 + 1]!;
    const bz = this.pts[b * 3 + 2]!;
    const adj = this.yAdj;
    out.x = ax + (bx - ax) * w;
    out.y = ay + adj[a]! + (by + adj[b]! - ay - adj[a]!) * w;
    out.z = az + (bz - az) * w;
    const hx = bx - ax;
    const hz = bz - az;
    const hl = Math.hypot(hx, hz);
    if (hl > 1e-6) {
      out.dx = hx / hl;
      out.dz = hz / hl;
    } else {
      out.dx = 1;
      out.dz = 0;
    }
    return out;
  }

  /** Midpoint of an edge (for labels / flyTo). */
  midpoint(e: number): PosOut {
    return this.pointAt(e, 0.5, { x: 0, y: 0, z: 0, dx: 1, dz: 0 });
  }

  /** Nearest point on any edge within maxDist (meters), or null. */
  nearestEdge(x: number, z: number, maxDist = 300, filter?: (edge: number) => boolean): EdgeHit | null {
    const c = this.cell;
    const r = Math.ceil(maxDist / c);
    const cx = Math.floor(x / c);
    const cz = Math.floor(z / c);
    let best: EdgeHit | null = null;
    let bestD = maxDist;
    const seen = new Set<number>();
    for (let ix = cx - r; ix <= cx + r; ix++) {
      for (let iz = cz - r; iz <= cz + r; iz++) {
        const arr = this.segGrid.get(this.key(ix, iz));
        if (!arr) continue;
        for (let k = 0; k < arr.length; k += 2) {
          const e = arr[k]!;
          const seg = arr[k + 1]!;
          const id = e * 65536 + seg;
          if (seen.has(id)) continue;
          seen.add(id);
          if (filter && !filter(e)) continue;
          const a = this.ptStart[e]! + seg;
          const ax = this.pts[a * 3]!;
          const az = this.pts[a * 3 + 2]!;
          const bx = this.pts[(a + 1) * 3]!;
          const bz = this.pts[(a + 1) * 3 + 2]!;
          const vx = bx - ax;
          const vz = bz - az;
          const L2 = vx * vx + vz * vz;
          let t = L2 > 0 ? ((x - ax) * vx + (z - az) * vz) / L2 : 0;
          t = Math.min(Math.max(t, 0), 1);
          const px = ax + vx * t;
          const pz = az + vz * t;
          const d = Math.hypot(x - px, z - pz);
          // tie-break equal distances (two-way streets) toward the lower index
          if (d < bestD - 1e-9 || (best && Math.abs(d - bestD) <= 1e-9 && e < best.edge)) {
            bestD = d;
            const ay = this.pts[a * 3 + 1]! + this.yAdj[a]!;
            const by = this.pts[(a + 1) * 3 + 1]! + this.yAdj[a + 1]!;
            const L = this.length[e]!;
            best = {
              edge: e,
              x: px,
              y: ay + (by - ay) * t,
              z: pz,
              dist: d,
              frac: L > 0 ? (this.cum[a]! + Math.sqrt(L2) * t) / L : 0,
            };
          }
        }
      }
    }
    return best;
  }

  /** Nearest node within maxDist (meters), or null. */
  nearestNode(x: number, z: number, maxDist = 300, filter?: (n: NetworkNode) => boolean): NetworkNode | null {
    const c = this.cell;
    const r = Math.ceil(maxDist / c);
    const cx = Math.floor(x / c);
    const cz = Math.floor(z / c);
    let best: NetworkNode | null = null;
    let bestD = maxDist;
    for (let ix = cx - r; ix <= cx + r; ix++) {
      for (let iz = cz - r; iz <= cz + r; iz++) {
        const arr = this.nodeGrid.get(this.key(ix, iz));
        if (!arr) continue;
        for (const idx of arr) {
          const nd = this.nodes[idx]!;
          if (filter && !filter(nd)) continue;
          const d = Math.hypot(nd.x - x, nd.z - z);
          if (d < bestD) {
            bestD = d;
            best = nd;
          }
        }
      }
    }
    return best;
  }
}
