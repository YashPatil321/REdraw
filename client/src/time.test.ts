import { describe, expect, it } from 'vitest';
import { advanceClock, binBlend, binIndex, binStart, formatHHMM, formatHHMMSS, parseHHMM } from './time';

const cfg = { bin_start_s: 21600, bin_s: 300, n_bins: 48 };

describe('time formatting', () => {
  it('formats and parses HH:MM', () => {
    expect(formatHHMM(21600)).toBe('06:00');
    expect(formatHHMM(23400)).toBe('06:30');
    expect(formatHHMM(34200)).toBe('09:30');
    expect(formatHHMMSS(27015.7)).toBe('07:30:15');
    expect(parseHHMM('08:30')).toBe(30600);
    expect(parseHHMM('7:05')).toBe(25500);
    expect(parseHHMM('25:00')).toBeNaN();
    expect(parseHHMM('abc')).toBeNaN();
  });
});

describe('bin math', () => {
  it('maps times to bins per the playback doc', () => {
    expect(binIndex(21600, cfg)).toBe(0);
    expect(binIndex(21899.9, cfg)).toBe(0);
    expect(binIndex(21900, cfg)).toBe(1);
    expect(binIndex(23400, cfg)).toBe(6); // 06:30
    expect(binIndex(34200, cfg)).toBe(42); // 09:30
    expect(binIndex(0, cfg)).toBe(0);
    expect(binIndex(1e9, cfg)).toBe(47);
    expect(binStart(6, cfg)).toBe(23400);
  });
  it('blends between bin centers', () => {
    expect(binBlend(21600 + 150, cfg)).toEqual({ b0: 0, b1: 0, w: 0 });
    const m = binBlend(21600 + 300, cfg); // halfway between centers of bins 0 and 1
    expect(m.b0).toBe(0);
    expect(m.b1).toBe(1);
    expect(m.w).toBeCloseTo(0.5);
    expect(binBlend(1e9, cfg)).toEqual({ b0: 47, b1: 47, w: 0 });
  });
  it('advances and loops the clock', () => {
    expect(advanceClock(23400, 1, 60, 23400, 34200)).toBe(23460);
    expect(advanceClock(34190, 1, 60, 23400, 34200)).toBe(23450);
    expect(advanceClock(34190, 1, 60, 23400, 34200, false)).toBe(34200);
  });
});
