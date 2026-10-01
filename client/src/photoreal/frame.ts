/**
 * Placing an Earth-centered (ECEF, WGS84) tileset such as Google Photorealistic
 * 3D Tiles in OUR scene frame (docs/coordinates.md: meters, x east, y up,
 * z south, origin at meta.region.origin, horizontal = UTM 11N grid offsets,
 * y = elevation).
 *
 * Two steps:
 * 1. A rigid ECEF -> local ENU matrix at the origin (x = east, y = up,
 *    z = -north). This is what the tiles group's matrix holds, so the GPU only
 *    ever sees small local coordinates (no float32 jitter).
 * 2. A small smooth correction ("warp", a few meters at most) from that
 *    tangent-plane frame to the scene frame. The scene is a UTM grid, which is
 *    rotated against true north (grid convergence, about -0.07 deg here), scaled
 *    (k ~ 0.9996) and flat, while the tangent plane drops below the ellipsoid
 *    with distance (about 2 m at 5 km). The correction is a least-squares
 *    quadratic in x, z (plus x*y, z*y) fitted against the exact geo.ts
 *    projection; residuals are millimeters over the region. The same function
 *    runs on the CPU (raycast results) and in the tile vertex shader.
 *
 * Heights: tiles carry WGS84 ellipsoid heights, our terrain NAVD88 elevations.
 * The geoid separation (about -35 m in San Diego) and any DEM bias are absorbed
 * by a vertical offset calibrated at runtime against network node elevations.
 */

import { latLonToScene, sceneToLatLon, type Origin } from '../geo';

export const WGS84_A = 6378137.0;
export const WGS84_F = 1 / 298.257223563;
export const WGS84_E2 = WGS84_F * (2 - WGS84_F);
const DEG = Math.PI / 180;

export type Vec3 = [number, number, number];

/** WGS84 geodetic (degrees, meters above the ellipsoid) -> ECEF meters. */
export function geodeticToEcef(latDeg: number, lonDeg: number, h: number): Vec3 {
  const phi = latDeg * DEG;
  const lam = lonDeg * DEG;
  const s = Math.sin(phi);
  const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * s * s);
  return [(N + h) * Math.cos(phi) * Math.cos(lam), (N + h) * Math.cos(phi) * Math.sin(lam), (N * (1 - WGS84_E2) + h) * s];
}

/** ECEF -> geodetic (Bowring with two refinements; sub-millimeter near the surface). */
export function ecefToGeodetic(x: number, y: number, z: number): { lat: number; lon: number; h: number } {
  const lon = Math.atan2(y, x);
  const p = Math.hypot(x, y);
  let phi = Math.atan2(z, p * (1 - WGS84_E2));
  let h = 0;
  for (let i = 0; i < 5; i++) {
    const s = Math.sin(phi);
    const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * s * s);
    h = p / Math.cos(phi) - N;
    phi = Math.atan2(z, p * (1 - (WGS84_E2 * N) / (N + h)));
  }
  return { lat: phi / DEG, lon: lon / DEG, h };
}

/**
 * Row-major 3x4 rigid transform ECEF -> local tangent frame at the origin:
 * x east, y up, z south (-north). Origin on the ellipsoid (h = 0).
 */
export interface RigidFrame {
  /** rows: east, up, south; each [r0, r1, r2] */
  r: [Vec3, Vec3, Vec3];
  /** ECEF of the origin */
  o: Vec3;
}

export function enuFrame(latDeg: number, lonDeg: number): RigidFrame {
  const phi = latDeg * DEG;
  const lam = lonDeg * DEG;
  const sp = Math.sin(phi);
  const cp = Math.cos(phi);
  const sl = Math.sin(lam);
  const cl = Math.cos(lam);
  const east: Vec3 = [-sl, cl, 0];
  const north: Vec3 = [-sp * cl, -sp * sl, cp];
  const up: Vec3 = [cp * cl, cp * sl, sp];
  const south: Vec3 = [-north[0], -north[1], -north[2]];
  return { r: [east, up, south], o: geodeticToEcef(latDeg, lonDeg, 0) };
}

export function applyFrame(f: RigidFrame, p: Vec3): Vec3 {
  const d: Vec3 = [p[0] - f.o[0], p[1] - f.o[1], p[2] - f.o[2]];
  return [
    f.r[0][0] * d[0] + f.r[0][1] * d[1] + f.r[0][2] * d[2],
    f.r[1][0] * d[0] + f.r[1][1] * d[1] + f.r[1][2] * d[2],
    f.r[2][0] * d[0] + f.r[2][1] * d[1] + f.r[2][2] * d[2],
  ];
}

/**
 * Column-major 4x4 (three.js Matrix4.elements order) of the ECEF -> tangent
 * frame transform. Translation = -R * o.
 */
export function frameMatrixElements(f: RigidFrame): number[] {
  const [a, b, c] = f.r;
  const t = [
    -(a[0] * f.o[0] + a[1] * f.o[1] + a[2] * f.o[2]),
    -(b[0] * f.o[0] + b[1] * f.o[1] + b[2] * f.o[2]),
    -(c[0] * f.o[0] + c[1] * f.o[1] + c[2] * f.o[2]),
  ];
  return [a[0], b[0], c[0], 0, a[1], b[1], c[1], 0, a[2], b[2], c[2], 0, t[0], t[1], t[2], 1];
}

// ---------------------------------------------------------------------------
// warp: tangent frame -> scene frame correction

/** Normalization length for the polynomial features (keeps the fit well conditioned). */
export const WARP_SCALE = 5000;
export const WARP_N = 9;

/** Features of a tangent-frame point, with u = p / WARP_SCALE. */
function features(x: number, y: number, z: number, out: number[]): number[] {
  const u = x / WARP_SCALE;
  const v = y / WARP_SCALE;
  const w = z / WARP_SCALE;
  out[0] = 1;
  out[1] = u;
  out[2] = v;
  out[3] = w;
  out[4] = u * u;
  out[5] = u * w;
  out[6] = w * w;
  out[7] = u * v;
  out[8] = w * v;
  return out;
}

/** Coefficients (meters) of delta x, delta y, delta z, WARP_N each. */
export interface Warp {
  cx: number[];
  cy: number[];
  cz: number[];
}

function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[piv]![c]!)) piv = r;
    [M[c], M[piv]] = [M[piv]!, M[c]!];
    const d = M[c]![c]!;
    if (Math.abs(d) < 1e-300) continue;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r]![c]! / d;
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r]![k]! -= f * M[c]![k]!;
    }
  }
  return M.map((row, i) => (Math.abs(row[i]!) < 1e-300 ? 0 : row[n]! / row[i]!));
}

export interface ExtentXZ {
  min_x: number;
  max_x: number;
  min_z: number;
  max_z: number;
}

/**
 * Fit the warp over the region extent (padded) for heights -200..1200 m.
 * Target for a geodetic point (lat, lon, h): scene (x, h, z) from geo.ts.
 */
export function fitWarp(frame: RigidFrame, origin: Origin, ext: ExtentXZ, pad = 3000): Warp {
  const ATA = Array.from({ length: WARP_N }, () => new Array<number>(WARP_N).fill(0));
  const ATb = [new Array<number>(WARP_N).fill(0), new Array<number>(WARP_N).fill(0), new Array<number>(WARP_N).fill(0)];
  const f = new Array<number>(WARP_N);
  const steps = 12;
  const x0 = ext.min_x - pad;
  const x1 = ext.max_x + pad;
  const z0 = ext.min_z - pad;
  const z1 = ext.max_z + pad;
  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= steps; j++) {
      const sx = x0 + ((x1 - x0) * i) / steps;
      const sz = z0 + ((z1 - z0) * j) / steps;
      const ll = sceneToLatLon(sx, sz, origin);
      for (const h of [-200, 300, 800, 1200]) {
        const p = applyFrame(frame, geodeticToEcef(ll.lat, ll.lon, h));
        const target: Vec3 = [sx, h, sz];
        features(p[0], p[1], p[2], f);
        for (let a = 0; a < WARP_N; a++) {
          for (let b = 0; b < WARP_N; b++) ATA[a]![b]! += f[a]! * f[b]!;
          for (let k = 0; k < 3; k++) ATb[k]![a]! += f[a]! * (target[k]! - p[k]!);
        }
      }
    }
  }
  return { cx: solve(ATA, ATb[0]!), cy: solve(ATA, ATb[1]!), cz: solve(ATA, ATb[2]!) };
}

const fbuf = new Array<number>(WARP_N);

/** Tangent-frame point -> scene point (adds the warp delta). */
export function warpPoint(w: Warp, p: Vec3): Vec3 {
  features(p[0], p[1], p[2], fbuf);
  let dx = 0;
  let dy = 0;
  let dz = 0;
  for (let k = 0; k < WARP_N; k++) {
    dx += w.cx[k]! * fbuf[k]!;
    dy += w.cy[k]! * fbuf[k]!;
    dz += w.cz[k]! * fbuf[k]!;
  }
  return [p[0] + dx, p[1] + dy, p[2] + dz];
}

/** Scene point -> tangent-frame point (fixed-point inverse; the delta is smooth and small). */
export function unwarpPoint(w: Warp, s: Vec3): Vec3 {
  let p: Vec3 = [s[0], s[1], s[2]];
  for (let i = 0; i < 4; i++) {
    const q = warpPoint(w, p);
    p = [p[0] - (q[0] - s[0]), p[1] - (q[1] - s[1]), p[2] - (q[2] - s[2])];
  }
  return p;
}

/** Full chain for tests and tools: geodetic -> scene (x, ellipsoid h, z) through frame + warp. */
export function geodeticToSceneViaTiles(frame: RigidFrame, w: Warp, lat: number, lon: number, h: number): Vec3 {
  return warpPoint(w, applyFrame(frame, geodeticToEcef(lat, lon, h)));
}

/** Reference: what the scene frame says for (lat, lon) (geo.ts), for comparisons. */
export function referenceScene(origin: Origin, lat: number, lon: number): { x: number; z: number } {
  return latLonToScene(lat, lon, origin);
}

/**
 * GLSL for the tile vertex shader: `vec3 rdWarp(vec3 p)` returns the delta
 * to add to a tangent-frame world position. Uniforms: rdWarpX/Y/Z (float[9]).
 */
export const WARP_GLSL = /* glsl */ `
uniform float rdWarpX[${WARP_N}];
uniform float rdWarpY[${WARP_N}];
uniform float rdWarpZ[${WARP_N}];
vec3 rdWarp(vec3 p) {
  vec3 u = p / ${WARP_SCALE.toFixed(1)};
  float f[${WARP_N}];
  f[0] = 1.0; f[1] = u.x; f[2] = u.y; f[3] = u.z;
  f[4] = u.x * u.x; f[5] = u.x * u.z; f[6] = u.z * u.z; f[7] = u.x * u.y; f[8] = u.z * u.y;
  vec3 d = vec3(0.0);
  for (int k = 0; k < ${WARP_N}; k++) {
    d += vec3(rdWarpX[k], rdWarpY[k], rdWarpZ[k]) * f[k];
  }
  return d;
}
`;
