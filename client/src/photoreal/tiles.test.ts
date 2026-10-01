// @vitest-environment happy-dom
/**
 * Headless end-to-end check of the photoreal loading / placement path with a
 * tiny local fake 3D Tiles tileset (tileset.json + one glb), no network and no
 * Google key: TilesRenderer -> our tile plugin (material swap) -> group matrix
 * (ECEF -> ENU + offset) -> CPU warp -> raycast heights and calibration.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { originFromLatLon, latLonToScene } from '../geo';
import { calibrateOffset, median, pickSpread } from './calibrate';
import { enuFrame, geodeticToEcef, warpPoint } from './frame';
import { PhotorealTiles, makeTileMaterial } from './tiles';

// node has no rAF; the tiles renderer schedules its queues with it
const g = globalThis as unknown as Record<string, unknown>;
g['requestAnimationFrame'] ??= (cb: (t: number) => void) => setTimeout(() => cb(performance.now()), 2);
g['cancelAnimationFrame'] ??= (h: ReturnType<typeof setTimeout>) => clearTimeout(h);

const origin = originFromLatLon(33.005, -117.125);
const extent = { min_x: -4700, max_x: 4700, min_z: -4450, max_z: 4450 };

/** Minimal glb: one mesh, positions only, indexed triangles. */
function makeGlb(positions: number[], indices: number[]): Uint8Array {
  const pos = new Float32Array(positions);
  const idx = new Uint16Array(indices);
  const posBytes = pos.byteLength;
  const idxBytes = idx.byteLength;
  const idxPad = (4 - (idxBytes % 4)) % 4;
  const bin = new Uint8Array(posBytes + idxBytes + idxPad);
  bin.set(new Uint8Array(pos.buffer), 0);
  bin.set(new Uint8Array(idx.buffer), posBytes);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k]!, pos[i + k]!);
      max[k] = Math.max(max[k]!, pos[i + k]!);
    }
  }
  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.4, 0.6, 0.3, 1] } }],
    buffers: [{ byteLength: bin.byteLength }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: posBytes, target: 34962 },
      { buffer: 0, byteOffset: posBytes, byteLength: idxBytes, target: 34963 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: pos.length / 3, type: 'VEC3', min, max },
      { bufferView: 1, componentType: 5123, count: idx.length, type: 'SCALAR' },
    ],
  };
  let jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jpad = (4 - (jsonBytes.length % 4)) % 4;
  if (jpad) {
    const p = new Uint8Array(jsonBytes.length + jpad);
    p.set(jsonBytes);
    p.fill(0x20, jsonBytes.length);
    jsonBytes = p;
  }
  const total = 12 + 8 + jsonBytes.length + 8 + bin.length;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true); // glTF
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonBytes.length, true);
  dv.setUint32(16, 0x4e4f534a, true); // JSON
  out.set(jsonBytes, 20);
  const o = 20 + jsonBytes.length;
  dv.setUint32(o, bin.length, true);
  dv.setUint32(o + 4, 0x004e4942, true); // BIN
  out.set(bin, o + 8);
  return out;
}

// Fake "photoreal" ground: a 600 m square of terrain centered on Del Norte HS,
// tilted (east side 12 m higher) so heights vary, at ellipsoid height h0.
const SITE = { lat: 33.0215, lon: -117.101 };
const GEOID_N = -35.2; // ellipsoid = orthometric + N
const SITE_ELEV = 160; // "our" elevation at the site center (NAVD88-like)
const H0 = SITE_ELEV + GEOID_N;
const HALF = 300;
const SLOPE = 12 / (2 * HALF); // m per m eastward

function buildFixture(): Record<string, Uint8Array | string> {
  // glTF content is y-up: (x east, y up, z south) in the tile's local ENU frame
  const n = 10;
  const positions: number[] = [];
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const x = -HALF + (2 * HALF * i) / n;
      const z = -HALF + (2 * HALF * j) / n;
      positions.push(x, x * SLOPE, z);
    }
  }
  const indices: number[] = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i;
      const b = a + 1;
      const c = a + (n + 1);
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  // tile transform: local ENU (x east, y north, z up after the y-up rotation) -> ECEF at the site
  const f = enuFrame(SITE.lat, SITE.lon);
  const [east, up, south] = f.r;
  const north = [-south[0], -south[1], -south[2]];
  const o = geodeticToEcef(SITE.lat, SITE.lon, H0);
  const transform = [east[0], east[1], east[2], 0, north[0]!, north[1]!, north[2]!, 0, up[0], up[1], up[2], 0, o[0], o[1], o[2], 1];
  const tileset = {
    asset: { version: '1.0', gltfUpAxis: 'Y' },
    geometricError: 1000,
    root: {
      transform,
      boundingVolume: { box: [0, 0, 6, HALF, 0, 0, 0, HALF, 0, 0, 0, 10] },
      geometricError: 0,
      refine: 'REPLACE',
      content: { uri: 'ground.glb' },
    },
  };
  return { 'tileset.json': JSON.stringify(tileset), 'ground.glb': makeGlb(positions, indices) };
}

const files = buildFixture();
const fetchFixture = async (url: string): Promise<Response> => {
  const name = url.split('/').pop()!.split('?')[0]!;
  const body = files[name];
  if (body === undefined) return new Response('not found', { status: 404 });
  return new Response(typeof body === 'string' ? body : new Blob([body as BlobPart]), { status: 200 });
};

let pr: PhotorealTiles;
const site = latLonToScene(SITE.lat, SITE.lon, origin);

beforeAll(async () => {
  pr = new PhotorealTiles({
    origin,
    extent,
    source: { kind: 'url', url: 'https://fixture.local/tiles/tileset.json' },
    fetchData: (u) => fetchFixture(u),
    noDraco: true,
    fade: false,
  });
  const camera = new THREE.PerspectiveCamera(50, 1, 1, 100000);
  camera.position.set(site.x, SITE_ELEV + 400, site.z + 400);
  camera.lookAt(site.x, SITE_ELEV, site.z);
  camera.updateMatrixWorld(true);
  pr.tiles.setCamera(camera);
  pr.setResolution(camera, 800, 600);
  for (let i = 0; i < 200 && pr.visibleCount === 0; i++) {
    camera.updateMatrixWorld(true);
    pr.update();
    await new Promise((r) => setTimeout(r, 10));
  }
}, 20000);

describe('fake tileset through PhotorealTiles', () => {
  it('loads the tile and swaps in unlit, non-tone-mapped tile materials', () => {
    expect(pr.visibleCount).toBeGreaterThan(0);
    let meshes = 0;
    pr.group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      meshes++;
      const mat = m.material as THREE.MeshBasicMaterial;
      expect(mat.isMeshBasicMaterial).toBe(true);
      expect(mat.toneMapped).toBe(false);
      expect(mat.userData.redrawTile).toBe(true);
    });
    expect(meshes).toBe(1);
  });

  it('places the tile at the right scene x/z (within 1 m) and height (ellipsoid + offset)', () => {
    pr.offset = 0;
    const h = pr.heightAt(site.x, site.z);
    expect(h).not.toBeNull();
    // with offset 0 the tile surface is at the ellipsoid height
    expect(h!).toBeCloseTo(H0, 0);
    // the slope runs east: +100 m east -> +2 m. Checks orientation (x east) and position.
    const hE = pr.heightAt(site.x + 100, site.z)!;
    const hW = pr.heightAt(site.x - 100, site.z)!;
    expect(hE - hW).toBeCloseTo(200 * SLOPE, 1);
    // north/south: no slope; z south must not flip x
    const hN = pr.heightAt(site.x, site.z - 150)!;
    expect(hN).toBeCloseTo(H0, 0);
    // the tile edge sits at site.x + 300 (east): inside hits, outside misses
    const edge = (dx: number, dz: number): number => { let lo = 0, hi = 400; for (let i = 0; i < 30; i++) { const m = (lo + hi) / 2; if (pr.heightAt(site.x + dx * m, site.z + dz * m) !== null) lo = m; else hi = m; } return lo; };
    const [e, w, n, sth] = [edge(1, 0), edge(-1, 0), edge(0, -1), edge(0, 1)];
    // centered on the site within 2 cm; 300 true meters = ~299.87 UTM grid meters (scale k)
    expect(Math.abs(e - w)).toBeLessThan(0.02);
    expect(Math.abs(n - sth)).toBeLessThan(0.02);
    expect(e).toBeGreaterThan(299.7);
    expect(e).toBeLessThan(300.0);
    expect(pr.heightAt(site.x + 299.2, site.z)).not.toBeNull();
    expect(pr.heightAt(site.x + 300.8, site.z)).toBeNull();
    expect(pr.heightAt(site.x, site.z - 299.2)).not.toBeNull();
    expect(pr.heightAt(site.x, site.z - 300.8)).toBeNull();
  });

  it('calibrates the vertical offset from node samples (median), then heights match ours', () => {
    pr.offset = 0;
    // fake network nodes: our elevation = tile ellipsoid height - N; one outlier (a bridge)
    const nodes = [
      [0, 0],
      [80, 40],
      [-120, 60],
      [150, -90],
      [-60, -140],
      [200, 200],
    ].map(([dx, dz]) => ({ x: site.x + dx!, z: site.z + dz!, y: SITE_ELEV + dx! * SLOPE }));
    nodes.push({ x: site.x + 10, z: site.z + 10, y: SITE_ELEV + 18 });
    const picked = pickSpread(nodes, site.x, site.z, 1000, 20);
    const samples = picked.map((n) => ({ ours: n.y, tile: pr.heightAt(n.x, n.z) }));
    const cal = calibrateOffset(samples)!;
    expect(cal).not.toBeNull();
    expect(cal.offset).toBeCloseTo(-GEOID_N, 0);
    pr.offset = cal.offset;
    expect(pr.heightAt(site.x + 100, site.z)!).toBeCloseTo(SITE_ELEV + 100 * SLOPE, 0);
  });

  it('camera ray picking returns scene coordinates', () => {
    const rc = new THREE.Raycaster();
    const from = new THREE.Vector3(site.x - 200, 800, site.z + 200);
    const target = new THREE.Vector3(site.x + 50, SITE_ELEV + 50 * SLOPE, site.z - 30);
    rc.set(from, target.clone().sub(from).normalize());
    const p = pr.pick(rc)!;
    expect(p).not.toBeNull();
    expect(p.distanceTo(target)).toBeLessThan(1);
  });

  it('the shader warp equals the CPU warp (uniforms carry the fitted coefficients)', () => {
    const p: [number, number, number] = [site.x, 100, site.z];
    const w = warpPoint(pr.warp, p);
    let d = [0, 0, 0];
    const u = [p[0] / 5000, p[1] / 5000, p[2] / 5000];
    const f = [1, u[0]!, u[1]!, u[2]!, u[0]! * u[0]!, u[0]! * u[2]!, u[2]! * u[2]!, u[0]! * u[1]!, u[2]! * u[1]!];
    d = [0, 1, 2].map((k) => {
      const c = [pr.uniforms.rdWarpX, pr.uniforms.rdWarpY, pr.uniforms.rdWarpZ][k]!.value;
      return f.reduce((acc, fv, i) => acc + fv * c[i]!, 0);
    });
    expect(w[0] - p[0]).toBeCloseTo(d[0]!, 6);
    expect(w[2] - p[2]).toBeCloseTo(d[2]!, 6);
  });
});

describe('tile material', () => {
  it('keeps the photo texture and color space', () => {
    const tex = new THREE.Texture();
    const old = new THREE.MeshStandardMaterial({ map: tex });
    const m = makeTileMaterial(old, pr.uniforms);
    expect(m.map).toBe(tex);
    expect(tex.colorSpace).toBe(THREE.SRGBColorSpace);
    expect(m.toneMapped).toBe(false);
  });
});

describe('calibration helpers', () => {
  it('median and too-few samples', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    expect(calibrateOffset([{ ours: 1, tile: 0 }, { ours: 1, tile: null }])).toBeNull();
  });
});
