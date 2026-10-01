import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { originFromLatLon } from '../geo';
import {
  applyFrame,
  ecefToGeodetic,
  enuFrame,
  fitWarp,
  frameMatrixElements,
  geodeticToEcef,
  geodeticToSceneViaTiles,
  unwarpPoint,
  warpPoint,
} from './frame';

interface GeoPoints {
  origin: { lat: number; lon: number };
  points: Array<{ lat: number; lon: number; x: number; z: number }>;
}
const pts = JSON.parse(readFileSync(new URL('../../../docs/geo_test_points.json', import.meta.url), 'utf8')) as GeoPoints;
const origin = originFromLatLon(pts.origin.lat, pts.origin.lon);
const ext = { min_x: -4700, max_x: 4700, min_z: -4450, max_z: 4450 };
const frame = enuFrame(origin.lat, origin.lon);
const warp = fitWarp(frame, origin, ext);

describe('ECEF', () => {
  it('round-trips geodetic coordinates', () => {
    for (const p of pts.points) {
      const e = geodeticToEcef(p.lat, p.lon, 250);
      const g = ecefToGeodetic(...e);
      expect(g.lat).toBeCloseTo(p.lat, 9);
      expect(g.lon).toBeCloseTo(p.lon, 9);
      expect(g.h).toBeCloseTo(250, 4);
    }
  });
  it('matches a known ECEF value (equator / prime meridian)', () => {
    const e = geodeticToEcef(0, 0, 0);
    expect(e[0]).toBeCloseTo(6378137, 6);
    expect(e[1]).toBeCloseTo(0, 6);
  });
});

describe('ENU frame', () => {
  it('puts the origin at 0 with y up and z south', () => {
    const o = applyFrame(frame, geodeticToEcef(origin.lat, origin.lon, 0));
    expect(Math.hypot(...o)).toBeLessThan(1e-6);
    const up = applyFrame(frame, geodeticToEcef(origin.lat, origin.lon, 100));
    expect(up[1]).toBeCloseTo(100, 6);
    // 0.001 deg north -> negative z (z is south), ~111 m
    const n = applyFrame(frame, geodeticToEcef(origin.lat + 0.001, origin.lon, 0));
    expect(n[2]).toBeLessThan(-110);
    expect(Math.abs(n[0])).toBeLessThan(0.01);
    const e = applyFrame(frame, geodeticToEcef(origin.lat, origin.lon + 0.001, 0));
    expect(e[0]).toBeGreaterThan(90);
  });

  it('matrix elements equal the frame (three.js Matrix4, float64)', () => {
    const m = new THREE.Matrix4().fromArray(frameMatrixElements(frame));
    for (const p of pts.points) {
      const ecef = geodeticToEcef(p.lat, p.lon, 300);
      const v = new THREE.Vector3(...ecef).applyMatrix4(m);
      const ref = applyFrame(frame, ecef);
      expect(v.distanceTo(new THREE.Vector3(...ref))).toBeLessThan(1e-3);
    }
  });

  it('without the warp the tangent plane is off by meters at the region edge (why the warp exists)', () => {
    const p = pts.points.find((q) => Math.abs(q.x) > 4000)!;
    const raw = applyFrame(frame, geodeticToEcef(p.lat, p.lon, 0));
    expect(Math.hypot(raw[0] - p.x, raw[2] - p.z)).toBeGreaterThan(2);
  });
});

describe('ECEF -> scene through frame + warp (docs/geo_test_points.json)', () => {
  it('lands every test point within 1 m horizontally (actually cm)', () => {
    for (const p of pts.points) {
      for (const h of [0, 150, 600]) {
        const s = geodeticToSceneViaTiles(frame, warp, p.lat, p.lon, h);
        const err = Math.hypot(s[0] - p.x, s[2] - p.z);
        expect(err, `point ${p.lat},${p.lon} h=${h}`).toBeLessThan(0.05);
        // y is the ellipsoid height (geoid offset is calibrated at runtime)
        expect(Math.abs(s[1] - h)).toBeLessThan(0.05);
      }
    }
  });

  it('del norte (hero) matches the contract value', () => {
    const s = geodeticToSceneViaTiles(frame, warp, 33.0215, -117.101, 0);
    expect(s[0]).toBeCloseTo(2243.6121, 1);
    expect(s[2]).toBeCloseTo(-1826.7882, 1);
  });

  it('unwarp inverts warp', () => {
    const s: [number, number, number] = [3210, 140, -2875];
    const back = warpPoint(warp, unwarpPoint(warp, s));
    expect(Math.hypot(back[0] - s[0], back[1] - s[1], back[2] - s[2])).toBeLessThan(1e-4);
  });

  it('the warp is a small correction (a few meters)', () => {
    const p = unwarpPoint(warp, [4700, 0, 4450]);
    expect(Math.hypot(p[0] - 4700, p[2] - 4450)).toBeLessThan(10);
    expect(Math.hypot(p[0] - 4700, p[2] - 4450)).toBeGreaterThan(0.5);
  });
});
