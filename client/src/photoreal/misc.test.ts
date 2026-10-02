import { describe, expect, it } from 'vitest';
import { RUN_SPEED, WALK_SPEED, walkStep } from '../scene/walk';
import { isSoftwareRenderer } from '../scene/quality';
import { DEFAULT_TILE_OFFSET_M, pickSpread, plausibleOffset } from './calibrate';
import { ACCEPT_ABOVE_M, cleanResiduals } from './drape';
import { GKEY_STORAGE, resolveGoogleKey } from './key';

function memStorage(init: Record<string, string> = {}) {
  const m = new Map(Object.entries(init));
  return {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    removeItem: (k: string) => void m.delete(k),
    m,
  };
}

describe('Google key resolution', () => {
  it('prefers ?gkey= and remembers it', () => {
    const st = memStorage();
    const r = resolveGoogleKey({ search: '?gkey=abc123', envKey: 'envkey', storage: st });
    expect(r).toEqual({ key: 'abc123', source: 'url' });
    expect(st.m.get(GKEY_STORAGE)).toBe('abc123');
    // next load without the param uses the stored key
    expect(resolveGoogleKey({ search: '', envKey: 'envkey', storage: st })).toEqual({ key: 'abc123', source: 'storage' });
  });
  it('falls back to the build-time env key, else none', () => {
    expect(resolveGoogleKey({ search: '', envKey: ' k ', storage: memStorage() })).toEqual({ key: 'k', source: 'env' });
    expect(resolveGoogleKey({ search: '', envKey: undefined, storage: null })).toEqual({ key: null, source: null });
  });
  it('?gkey= (empty) or ?gkey=clear forgets the stored key', () => {
    const st = memStorage({ [GKEY_STORAGE]: 'old' });
    expect(resolveGoogleKey({ search: '?gkey=clear', envKey: '', storage: st }).key).toBeNull();
    expect(st.m.has(GKEY_STORAGE)).toBe(false);
  });
});

describe('walk camera step', () => {
  it('moves north (-z) at heading 0, east (+x) at heading pi/2', () => {
    const a = walkStep({ x: 0, z: 0, heading: 0 }, { fwd: 1, right: 0, run: false }, 1);
    expect(a.x).toBeCloseTo(0, 9);
    expect(a.z).toBeCloseTo(-WALK_SPEED, 9);
    const b = walkStep({ x: 0, z: 0, heading: Math.PI / 2 }, { fwd: 1, right: 0, run: true }, 1);
    expect(b.x).toBeCloseTo(RUN_SPEED, 9);
    expect(b.z).toBeCloseTo(0, 9);
  });
  it('strafes right of the heading and normalizes diagonals', () => {
    const r = walkStep({ x: 0, z: 0, heading: 0 }, { fwd: 0, right: 1, run: false }, 1);
    expect(r.x).toBeCloseTo(WALK_SPEED, 9);
    const d = walkStep({ x: 0, z: 0, heading: 0 }, { fwd: 1, right: 1, run: false }, 1);
    expect(Math.hypot(d.x, d.z)).toBeCloseTo(WALK_SPEED, 9);
  });
});

describe('drape residual cleaning', () => {
  it('rejects tree / hole samples and interpolates them from neighbors', () => {
    const v = cleanResiduals([0.2, 0.3, 9.0, 0.5, null, 0.7]);
    expect(v[2]).toBeGreaterThan(0.25);
    expect(v[2]).toBeLessThan(ACCEPT_ABOVE_M);
    expect(v[4]).toBeCloseTo(0.6, 5);
  });
  it('no usable samples -> zero adjustment', () => {
    expect(Array.from(cleanResiduals([null, 50, -40]))).toEqual([0, 0, 0]);
  });
});

describe('misc', () => {
  it('software renderer detection (quality defaults low only there)', () => {
    expect(isSoftwareRenderer('ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)), SwiftShader driver)')).toBe(true);
    expect(isSoftwareRenderer('llvmpipe (LLVM 15.0.7, 256 bits)')).toBe(true);
    expect(isSoftwareRenderer('ANGLE (Intel, Intel(R) Iris(R) Xe Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)')).toBe(false);
    expect(isSoftwareRenderer('Apple M2')).toBe(false);
  });
  it('pickSpread thins samples on a grid, nearest first', () => {
    const items = [
      { x: 0, z: 0 },
      { x: 5, z: 5 },
      { x: 100, z: 0 },
      { x: 2000, z: 0 },
    ];
    const p = pickSpread(items, 0, 0, 500, 10, 60);
    expect(p).toEqual([{ x: 0, z: 0 }, { x: 100, z: 0 }]);
  });

  it('rejects calibrations against coarse tiles or clutter', () => {
    expect(plausibleOffset({ offset: 34.5, n: 24, mad: 0.2 })).toBe(true);
    expect(plausibleOffset({ offset: -33.85, n: 24, mad: 0.4 })).toBe(false);
    expect(plausibleOffset({ offset: DEFAULT_TILE_OFFSET_M + 2, n: 24, mad: 5 })).toBe(false);
  });
});
