import { describe, expect, it } from 'vitest';
import { parseHdBuildingsManifest, resolveHdPath } from './hdManifest';
import { placementsStale } from './props';

describe('parseHdBuildingsManifest', () => {
  it('reads lods arrays of objects', () => {
    const m = parseHdBuildingsManifest({
      lod0_max_distance_m: 500,
      tiles: [{ id: 'r0_c0', lods: [{ level: 0, file: 'r0_c0_lod0.glb' }, { level: 1, file: 'r0_c0_lod1.glb' }], count: 12 }],
    });
    expect(m.lod0Distance).toBe(500);
    expect(m.tiles).toHaveLength(1);
    expect(m.tiles[0]!.lods).toEqual(['buildings_hd/r0_c0_lod0.glb', 'buildings_hd/r0_c0_lod1.glb']);
    expect(m.tiles[0]!.buildings).toBe(12);
  });

  it('reads a tile map with lod0 / lod1 keys and full asset paths', () => {
    const m = parseHdBuildingsManifest({
      tiles: { r1_c2: { lod0: 'buildings_hd/a.glb', lod1: { path: 'assets/buildings_hd/b.glb' } }, r1_c3: { files: { lod1: 'c.glb' } } },
    });
    expect(m.tiles.map((t) => t.id)).toEqual(['r1_c2', 'r1_c3']);
    expect(m.tiles[0]!.lods).toEqual(['buildings_hd/a.glb', 'buildings_hd/b.glb']);
    expect(m.tiles[1]!.lods).toEqual([null, 'buildings_hd/c.glb']);
  });

  it('skips tiles without glbs and tolerates junk', () => {
    expect(parseHdBuildingsManifest(null).tiles).toEqual([]);
    expect(parseHdBuildingsManifest({ tiles: [{ id: 'x' }, 3, null] }).tiles).toEqual([]);
  });

  it('resolves paths against a base', () => {
    expect(resolveHdPath('tiles/r0_c0.glb', 'buildings_hd/')).toBe('buildings_hd/tiles/r0_c0.glb');
    expect(resolveHdPath('/assets/buildings_hd/x.glb', 'whatever/')).toBe('buildings_hd/x.glb');
  });
});

describe('placementsStale', () => {
  it('flags placements built on another grid', () => {
    expect(placementsStale({ cells: { min_x: -5900, min_z: -6200 } }, { min_x: -4500, min_z: -3150 })).toBe(true);
    expect(placementsStale({ cells: { min_x: -4500, min_z: -3150 } }, { min_x: -4500, min_z: -3150 })).toBe(false);
    expect(placementsStale({}, { min_x: 0, min_z: 0 })).toBe(false);
    expect(placementsStale({ cells: { min_x: 1, min_z: 1 } }, null)).toBe(false);
  });
});
