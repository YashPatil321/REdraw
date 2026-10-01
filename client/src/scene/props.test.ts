import { describe, expect, it } from 'vitest';
import { pickVariant } from '../traffic/layer';
import { parsePlacements, parsePropsManifest } from './props';
import { pairRoads } from './roadDetails';

describe('props manifest', () => {
  it('finds prop entries in lists or maps and converts paint colors', () => {
    const a = parsePropsManifest({
      props: [
        { id: 'car_sedan', file: 'vehicles/car_sedan.glb', share: 0.3 },
        { id: 'street_lamp', glb: '/props/street_lamp.glb' },
      ],
      paint_colors: ['#ffffff', { hex: '#000000', share: 0.1 }],
    });
    expect(a.entries.map((e) => [e.id, e.file])).toEqual([
      ['car_sedan', 'vehicles/car_sedan.glb'],
      ['street_lamp', 'props/street_lamp.glb'],
    ]);
    expect(a.paint[0]).toEqual([1, 1, 1]);
    expect(a.paint.length).toBeGreaterThan(1);
    const b = parsePropsManifest({ props: { coast_live_oak: { file: 'trees/oak.glb' } } });
    expect(b.entries[0]).toMatchObject({ id: 'coast_live_oak', file: 'trees/oak.glb' });
  });
});

describe('blender props_manifest.json (top-level array)', () => {
  it('reads fleet_share and the per-vehicle paint palette', () => {
    const a = parsePropsManifest([
      { id: 'car_sedan', kind: 'vehicle', vehicle_class: 'car', file: 'vehicles/car_sedan.glb', fleet_share: 0.3, paint_colors: [{ name: 'white', hex: '#E9EAEA', share: 0.25 }, { name: 'red', hex: '#8E1B1E', share: 0.08 }] },
      { id: 'tree_oak', kind: 'tree', file: 'vegetation/tree_oak.glb' },
    ]);
    expect(a.entries.map((e) => e.id)).toEqual(['car_sedan', 'tree_oak']);
    expect(a.entries[0]!.share).toBe(0.3);
    expect(a.paint.length).toBe(5 + 2);
  });
});

describe('placements', () => {
  const rows = [
    [10, 100, 20, 0.5, 1.2, 0],
    [30, 101, 40, 1.0, 0.8, 1],
  ];
  const buf = new Float32Array(rows.flat()).buffer;
  it('reads an interleaved table with a prop index', () => {
    const m = parsePlacements({ fields: ['x', 'y', 'z', 'rot_y', 'scale', 'prop'], count: 2, props: ['oak', 'lamp'] }, buf);
    expect(m.get('oak')?.[0]).toMatchObject({ x: 10, z: 20, scale: expect.closeTo(1.2, 5) });
    expect(m.get('lamp')?.length).toBe(1);
  });
  it('reads per-prop sections', () => {
    const m = parsePlacements({ fields: ['x', 'y', 'z', 'rot_y', 'scale', 'prop'], sections: [{ id: 'oak', offset: 0, count: 2 }] }, buf);
    expect(m.get('oak')?.length).toBe(2);
  });
});

describe('vehicle variants and road pairing', () => {
  it('picks variants by share', () => {
    expect(pickVariant([0.5, 0.5], 0.1)).toBe(0);
    expect(pickVariant([0.5, 0.5], 0.9)).toBe(1);
    expect(pickVariant([0, 1], 0.0)).toBe(1);
  });
  it('pairs two-way directed edges into one road', () => {
    const edges = [
      { i: 0, u: 1, v: 2, name: '', label: '', highway: 'secondary', lanes: 2, len: 100, pts: [] },
      { i: 1, u: 2, v: 1, name: '', label: '', highway: 'secondary', lanes: 1, len: 100, pts: [] },
      { i: 2, u: 2, v: 3, name: '', label: '', highway: 'primary', lanes: 3, len: 50, pts: [] },
    ];
    const roads = pairRoads({ edges, length: new Float32Array([100, 100, 50]) } as never);
    expect(roads).toEqual([
      { edge: 0, lanesFwd: 2, lanesBack: 1, highway: 'secondary' },
      { edge: 2, lanesFwd: 3, lanesBack: 0, highway: 'primary' },
    ]);
  });
});
