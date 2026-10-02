/**
 * Vertical alignment between photoreal tiles (WGS84 ellipsoid heights) and our
 * scene (NAVD88 elevations from the DEM). Sample the tile surface under road
 * network nodes and take the median of (node elevation - tile height). The
 * median ignores samples that hit a tree, a car or a bridge.
 */

export interface HeightSample {
  /** our elevation (network node y) */
  ours: number;
  /** tile surface height at the same x, z (ellipsoid, after the frame warp), or null if no hit */
  tile: number | null;
}

export interface Calibration {
  offset: number;
  /** samples used */
  n: number;
  /** median absolute deviation of the residuals (m), a quality hint */
  mad: number;
}

export function median(values: number[]): number {
  if (!values.length) return NaN;
  const v = [...values].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m]! : (v[m - 1]! + v[m]!) / 2;
}

/** Offset to ADD to tile heights so they match our elevations; null if too few hits. */
export function calibrateOffset(samples: HeightSample[], minSamples = 4): Calibration | null {
  const d: number[] = [];
  for (const s of samples) if (s.tile !== null && Number.isFinite(s.tile) && Number.isFinite(s.ours)) d.push(s.ours - s.tile);
  if (d.length < minSamples) return null;
  const off = median(d);
  const mad = median(d.map((x) => Math.abs(x - off)));
  return { offset: off, n: d.length, mad };
}

/** Pick up to `max` items spread over the area near (cx, cz): nearest first within radius, thinned on a grid. */
export function pickSpread<T extends { x: number; z: number }>(items: T[], cx: number, cz: number, radius: number, max: number, cell = 60): T[] {
  const near = items
    .map((it) => ({ it, d: Math.hypot(it.x - cx, it.z - cz) }))
    .filter((o) => o.d <= radius)
    .sort((a, b) => a.d - b.d);
  const used = new Set<string>();
  const out: T[] = [];
  for (const { it } of near) {
    const k = `${Math.floor(it.x / cell)},${Math.floor(it.z / cell)}`;
    if (used.has(k)) continue;
    used.add(k);
    out.push(it);
    if (out.length >= max) break;
  }
  return out;
}

/** Geoid separation guess for San Diego (EGM2008 is about -35 m): tiles start 35 m low. */
export const DEFAULT_TILE_OFFSET_M = 35;

/** How far a calibrated offset may stray from the geoid guess (m): DEM bias is decimeters, coarse tiles are tens of meters off. */
export const MAX_OFFSET_DEVIATION_M = 8;
/** A spread beyond this (m) means the samples hit canopy / coarse geometry, not the road surface. */
export const MAX_CALIBRATION_MAD_M = 3;

/** Reject calibrations against coarse tiles or clutter (keeps the previous offset). */
export function plausibleOffset(c: Calibration, guess = DEFAULT_TILE_OFFSET_M): boolean {
  return Math.abs(c.offset - guess) <= MAX_OFFSET_DEVIATION_M && c.mad <= MAX_CALIBRATION_MAD_M;
}
