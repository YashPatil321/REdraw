import { describe, expect, it } from 'vitest';
import points from '../../docs/geo_test_points.json';
import {
  latLonToScene,
  latLonToUtm,
  originFromBbox,
  originFromLatLon,
  sceneToLatLon,
  sceneToUnreal,
} from './geo';

const tol = points.tolerance_m;

describe('geo (docs/geo_test_points.json)', () => {
  const origin = originFromLatLon(points.origin.lat, points.origin.lon);

  it('projects the origin like pyproj', () => {
    expect(Math.abs(origin.easting - points.origin.easting)).toBeLessThan(tol);
    expect(Math.abs(origin.northing - points.origin.northing)).toBeLessThan(tol);
  });

  it('origin from region bbox equals the documented origin', () => {
    const o = originFromBbox(points.bbox);
    expect(o.lat).toBeCloseTo(points.origin.lat, 12);
    expect(o.lon).toBeCloseTo(points.origin.lon, 12);
  });

  for (const p of points.points) {
    it(`lat/lon ${p.lat},${p.lon} -> UTM and scene within 1 cm`, () => {
      const utm = latLonToUtm(p.lat, p.lon);
      expect(Math.abs(utm.easting - p.easting)).toBeLessThan(tol);
      expect(Math.abs(utm.northing - p.northing)).toBeLessThan(tol);
      const s = latLonToScene(p.lat, p.lon, origin);
      expect(Math.abs(s.x - p.x)).toBeLessThan(tol);
      expect(Math.abs(s.z - p.z)).toBeLessThan(tol);
    });

    it(`scene ${p.x},${p.z} -> lat/lon round trip within 1 cm`, () => {
      const ll = sceneToLatLon(p.x, p.z, origin);
      const back = latLonToScene(ll.lat, ll.lon, origin);
      expect(Math.abs(back.x - p.x)).toBeLessThan(tol);
      expect(Math.abs(back.z - p.z)).toBeLessThan(tol);
      // ~1 cm in degrees
      expect(Math.abs(ll.lat - p.lat)).toBeLessThan(1e-7);
      expect(Math.abs(ll.lon - p.lon)).toBeLessThan(1e-7);
    });
  }

  it('maps scene to Unreal centimeters', () => {
    expect(sceneToUnreal(2243.6121, 10, -1826.7882)).toEqual({
      X: 224361.21,
      Y: -182678.82,
      Z: 1000,
    });
  });
});
