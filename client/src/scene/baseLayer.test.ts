import { describe, expect, it } from 'vitest';
import { BASE_LAYER, BaseLayerSwitch, easeMix, type BaseLayerInput } from './baseLayer';

const air = (altitude: number, extra: Partial<BaseLayerInput> = {}): BaseLayerInput => ({
  altitude,
  walking: false,
  googleOk: true,
  aerialReady: true,
  streetReady: true,
  ...extra,
});

/** run for `seconds` in 1/60 s steps */
function run(sw: BaseLayerSwitch, i: BaseLayerInput, seconds: number): number {
  let m = sw.mix;
  for (let t = 0; t < seconds; t += 1 / 60) m = sw.update(i, 1 / 60);
  return m;
}

describe('base layer switch', () => {
  it('thresholds have hysteresis', () => {
    expect(BASE_LAYER.exitStreetM).toBeGreaterThan(BASE_LAYER.enterStreetM);
  });

  it('starts on our world and fades to Google once it is loaded high up', () => {
    const sw = new BaseLayerSwitch();
    expect(sw.mix).toBe(1);
    run(sw, air(800, { aerialReady: false }), 1);
    expect(sw.target).toBe('street'); // waits for the tiles
    run(sw, air(800), BASE_LAYER.fadeS * 0.5);
    expect(sw.fading).toBe(true);
    run(sw, air(800), BASE_LAYER.fadeS);
    expect(sw.target).toBe('aerial');
    expect(sw.mix).toBe(0);
  });

  it('hysteresis: enters street below 100 m, returns to aerial only above 150 m', () => {
    const sw = new BaseLayerSwitch();
    sw.snap('aerial');
    run(sw, air(120), 3);
    expect(sw.target).toBe('aerial'); // inside the band: no change
    run(sw, air(90), 3);
    expect(sw.target).toBe('street');
    expect(sw.mix).toBe(1);
    run(sw, air(140), 3);
    expect(sw.target).toBe('street'); // still inside the band
    run(sw, air(160), 3);
    expect(sw.target).toBe('aerial');
    expect(sw.mix).toBe(0);
  });

  it('never flickers when the height oscillates around one threshold', () => {
    const sw = new BaseLayerSwitch();
    sw.snap('aerial');
    let switches = 0;
    let last = sw.target;
    for (let k = 0; k < 600; k++) {
      sw.update(air(100 + 8 * Math.sin(k / 7)), 1 / 60);
      if (sw.target !== last) switches++;
      last = sw.target;
    }
    expect(switches).toBe(1);
  });

  it('walk mode switches to street immediately, even if our assets are still streaming', () => {
    const sw = new BaseLayerSwitch();
    sw.snap('aerial');
    sw.update(air(600, { walking: true, streetReady: false }), 1 / 60);
    expect(sw.target).toBe('street');
    const m = run(sw, air(1.7, { walking: true, streetReady: false }), BASE_LAYER.fadeS + 0.1);
    expect(m).toBe(1);
  });

  it('waits for our assets before a height-triggered switch (bounded by a timeout)', () => {
    const sw = new BaseLayerSwitch();
    sw.snap('aerial');
    run(sw, air(50, { streetReady: false }), BASE_LAYER.readyTimeoutS * 0.5);
    expect(sw.target).toBe('aerial');
    run(sw, air(50, { streetReady: false }), BASE_LAYER.readyTimeoutS);
    expect(sw.target).toBe('street');
  });

  it('no key / quota exhausted / tiles failing: our world at every altitude', () => {
    const sw = new BaseLayerSwitch();
    run(sw, air(5000, { googleOk: false, aerialReady: false }), 3);
    expect(sw.target).toBe('street');
    expect(sw.mix).toBe(1);
    // Google fails while aerial: fade back to our world right away
    sw.snap('aerial');
    sw.update(air(5000, { googleOk: false }), 1 / 60);
    expect(sw.target).toBe('street');
    expect(run(sw, air(5000, { googleOk: false }), 2)).toBe(1);
  });

  it('crossfade takes about fadeS and eases at both ends', () => {
    const sw = new BaseLayerSwitch();
    sw.snap('aerial');
    run(sw, air(10), BASE_LAYER.fadeS * 0.5);
    expect(sw.mix).toBeGreaterThan(0.4);
    expect(sw.mix).toBeLessThan(0.6);
    expect(easeMix(0)).toBe(0);
    expect(easeMix(1)).toBe(1);
    expect(easeMix(0.1)).toBeLessThan(0.1);
    expect(easeMix(0.5)).toBeCloseTo(0.5, 9);
  });
});
