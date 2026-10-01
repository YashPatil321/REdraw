import { describe, expect, it } from 'vitest';
import { localSecondsToDate, solarPosition, sunDirection, tzOffsetMinutes } from './solar';

const LAT = 33.005;
const LON = -117.125;

describe('solar position', () => {
  it('is high in the south near solar noon (Oct 1, San Diego)', () => {
    const p = solarPosition(new Date(Date.UTC(2026, 9, 1, 19, 45)), LAT, LON);
    expect(p.elevation).toBeGreaterThan(50);
    expect(p.elevation).toBeLessThan(58);
    expect(p.azimuth).toBeGreaterThan(170);
    expect(p.azimuth).toBeLessThan(190);
  });
  it('is below the horizon before sunrise and low in the east mid morning', () => {
    expect(solarPosition(new Date(Date.UTC(2026, 9, 1, 13, 15)), LAT, LON).elevation).toBeLessThan(0);
    const am = solarPosition(new Date(Date.UTC(2026, 9, 1, 15, 0)), LAT, LON); // 08:00 PDT
    expect(am.elevation).toBeGreaterThan(5);
    expect(am.elevation).toBeLessThan(25);
    expect(am.azimuth).toBeGreaterThan(90);
    expect(am.azimuth).toBeLessThan(120);
  });
  it('maps to scene axes (x east, y up, z south)', () => {
    const [x, y, z] = sunDirection({ elevation: 0, azimuth: 180 });
    expect(x).toBeCloseTo(0);
    expect(y).toBeCloseTo(0);
    expect(z).toBeCloseTo(1);
    expect(sunDirection({ elevation: 0, azimuth: 90 })[0]).toBeCloseTo(1);
  });
  it('converts local seconds in America/Los_Angeles to UTC', () => {
    const day = new Date(Date.UTC(2026, 9, 1, 18, 0));
    expect(tzOffsetMinutes('America/Los_Angeles', day)).toBe(-420);
    expect(localSecondsToDate(8 * 3600, day, 'America/Los_Angeles').toISOString()).toBe('2026-10-01T15:00:00.000Z');
  });
});
