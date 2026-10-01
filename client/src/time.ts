/** Time and bin math. Times are seconds since local midnight (06:00 = 21600). */

export interface BinConfig {
  bin_start_s: number;
  bin_s: number;
  n_bins: number;
}

/** 27000 -> "07:30" */
export function formatHHMM(s: number): string {
  const total = Math.floor(s / 60);
  const h = Math.floor(total / 60) % 24;
  const m = total % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

/** 27015 -> "07:30:15" */
export function formatHHMMSS(s: number): string {
  const sec = Math.floor(s) % 60;
  return `${formatHHMM(s)}:${String(sec).padStart(2, '0')}`;
}

/** "07:30" -> 27000; returns NaN for malformed input. */
export function parseHHMM(v: string): number {
  const m = /^(\d{1,2}):(\d{2})$/.exec(v.trim());
  if (!m) return Number.NaN;
  const h = Number(m[1]);
  const min = Number(m[2]);
  if (h > 23 || min > 59) return Number.NaN;
  return h * 3600 + min * 60;
}

/** Bin index containing time t (bin b covers [start + b*bin_s, start + (b+1)*bin_s)), clamped. */
export function binIndex(t: number, cfg: BinConfig): number {
  const b = Math.floor((t - cfg.bin_start_s) / cfg.bin_s);
  return Math.min(Math.max(b, 0), cfg.n_bins - 1);
}

/** Bin pair + blend weight for smooth interpolation, using bin centers. */
export function binBlend(t: number, cfg: BinConfig): { b0: number; b1: number; w: number } {
  const f = (t - cfg.bin_start_s) / cfg.bin_s - 0.5;
  if (f <= 0) return { b0: 0, b1: 0, w: 0 };
  const last = cfg.n_bins - 1;
  if (f >= last) return { b0: last, b1: last, w: 0 };
  const b0 = Math.floor(f);
  return { b0, b1: b0 + 1, w: f - b0 };
}

/** Start time of bin b. */
export function binStart(b: number, cfg: BinConfig): number {
  return cfg.bin_start_s + b * cfg.bin_s;
}

/** Advance a sim clock: wall-clock dt (s) times speed, looping within [start, end). */
export function advanceClock(t: number, dtWall: number, speed: number, start: number, end: number, loop = true): number {
  let next = t + dtWall * speed;
  if (next >= end) next = loop ? start + ((next - start) % (end - start)) : end;
  if (next < start) next = start;
  return next;
}
