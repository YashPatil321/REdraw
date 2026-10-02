import { describe, expect, it } from 'vitest';
import { fadeBands, layerWeights } from './lodFade';

describe('prop LOD crossfade', () => {
  it('full model and impostor share the band and always add up to one inside the range', () => {
    const b = fadeBands(200, 1300, 0.12);
    expect(b.n0).toBeCloseTo(176, 6);
    expect(b.n1).toBeCloseTo(200, 6); // never draws full models further than the switch distance
    for (const d of [0, 100, 176, 180, 188, 195, 200, 500, 1000]) {
      const w = layerWeights(d, b);
      expect(w.full + w.lod).toBeCloseTo(1, 6);
    }
    expect(layerWeights(150, b)).toEqual({ full: 1, lod: 0 });
    expect(layerWeights(188, b).full).toBeCloseTo(0.5, 6);
    expect(layerWeights(300, b)).toEqual({ full: 0, lod: 1 });
  });

  it('the impostor dissolves out at the far end; props without one fade out before the cut', () => {
    const b = fadeBands(200, 1300, 0.12);
    expect(layerWeights(1300, b).lod).toBe(0);
    expect(layerWeights(1220, b).lod).toBeGreaterThan(0);
    expect(layerWeights(1220, b).lod).toBeLessThan(1);
    const lamp = fadeBands(380, 0, 0.12);
    expect(lamp.f1).toBe(0);
    expect(layerWeights(379.9, lamp).full).toBeLessThan(0.01);
    expect(layerWeights(300, lamp).full).toBe(1);
    expect(layerWeights(1000, lamp)).toEqual({ full: 0, lod: 0 });
  });
});
