import { describe, expect, it } from 'vitest';
import { VC_STOPS, queueColor, rampColor, rgbToCss, speedColor, vcColor } from './congestion';

describe('congestion color mapping', () => {
  it('is green at free flow and red at/after capacity', () => {
    const g = vcColor(0);
    expect(g[1]).toBeGreaterThan(g[0]);
    const r = vcColor(1.0);
    expect(r[0]).toBeGreaterThan(0.9);
    expect(r[1]).toBeLessThan(0.3);
    expect(vcColor(5)).toEqual(VC_STOPS[VC_STOPS.length - 1]!.c);
  });

  it('handles NaN and negatives as free flow', () => {
    expect(vcColor(Number.NaN)).toEqual(VC_STOPS[0]!.c);
    expect(vcColor(-1)).toEqual(VC_STOPS[0]!.c);
  });

  it('red channel rises and green falls monotonically from 0.5 to 1.0', () => {
    let prev = vcColor(0.5);
    for (let v = 0.55; v <= 1.0001; v += 0.05) {
      const c = vcColor(v);
      expect(c[0]).toBeGreaterThanOrEqual(prev[0] - 1e-9);
      if (v > 0.76) expect(c[1]).toBeLessThanOrEqual(prev[1] + 1e-9);
      prev = c;
    }
  });

  it('interpolates linearly between stops', () => {
    const stops = [
      { at: 0, c: [0, 0, 0] as [number, number, number] },
      { at: 2, c: [1, 0.5, 0] as [number, number, number] },
    ];
    expect(rampColor(stops, 1)).toEqual([0.5, 0.25, 0]);
  });

  it('writes into a provided output array', () => {
    const out: [number, number, number] = [9, 9, 9];
    const ret = vcColor(0, out);
    expect(ret).toBe(out);
    expect(out[0]).not.toBe(9);
  });

  it('colors slow vehicles red and fast ones not red', () => {
    expect(speedColor(0)[0]).toBeGreaterThan(0.9);
    expect(speedColor(0)[1]).toBeLessThan(0.3);
    expect(speedColor(60)[0]).toBeLessThan(0.6);
  });

  it('queue color scales with curb capacity', () => {
    expect(queueColor(0, 10)).toEqual(VC_STOPS[0]!.c);
    expect(queueColor(40, 10)[0]).toBeGreaterThan(0.6);
    expect(rgbToCss([1, 0, 0.5])).toBe('rgb(255, 0, 128)');
  });
});
