/**
 * MOCK FIXTURE (`?mock=1` only). A tiny procedural stand-in world so the UI
 * can be exercised without the API. NOT real geography; labeled as mock in
 * the UI. Same data shapes as docs/api.md and docs/data_contract.md.
 */

import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { originFromLatLon, sceneToLatLon } from '../geo';
import type { BuildingInfo, Manifest, NetworkEdge, NetworkJson, NetworkNode, School } from '../types';

export const MOCK_HALF = 1500; // half extent (m)
const GRID = 250; // street spacing
const N = Math.round((2 * (MOCK_HALF - 250)) / GRID) + 1; // nodes per side (11)
const TILES = 2;

export const MOCK_ORIGIN = originFromLatLon(33.005, -117.125);

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function terrainHeight(x: number, z: number): number {
  return 180 + 28 * Math.sin(x / 420) * Math.cos(z / 360) + 9 * Math.sin((x + z) / 170) + 14 * Math.cos((x - 2 * z) / 900);
}

export interface MockWorld {
  network: NetworkJson;
  schools: School[];
  buildings: Map<number, BuildingInfo & { tile: string; x: number; z: number; w: number; d: number; h: number }>;
  exitNodes: number[];
  nodeGrid: (i: number, j: number) => number;
  edgeByUV: Map<string, number>;
}

let cached: MockWorld | null = null;

export function mockWorld(): MockWorld {
  if (cached) return cached;
  const nodes: NetworkNode[] = [];
  const start = -(N - 1) * GRID * 0.5;
  const nodeGrid = (i: number, j: number): number => 1000 + i * N + j;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x = start + j * GRID;
      const z = start + i * GRID;
      nodes.push({ id: nodeGrid(i, j), x, y: terrainHeight(x, z), z, signal: i % 4 === 1 && j % 4 === 1 });
    }
  }
  const mid = Math.floor(N / 2);
  const edges: NetworkEdge[] = [];
  const edgeByUV = new Map<string, number>();
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const addEdge = (u: number, v: number, name: string, label: string, highway: string, lanes: number): void => {
    const a = nodeById.get(u)!;
    const b = nodeById.get(v)!;
    const pts: number[] = [];
    const steps = 5;
    for (let k = 0; k <= steps; k++) {
      const x = a.x + ((b.x - a.x) * k) / steps;
      const z = a.z + ((b.z - a.z) * k) / steps;
      pts.push(x, terrainHeight(x, z) + 0.3, z);
    }
    const i = edges.length;
    edges.push({ i, u, v, name, label, highway, lanes, len: Math.hypot(b.x - a.x, b.z - a.z), pts });
    edgeByUV.set(`${u}-${v}`, i);
  };
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (j < N - 1) {
        const art = i === mid;
        const name = art ? 'Camino Mock Parkway' : `Mock Street ${i + 1}`;
        addEdge(nodeGrid(i, j), nodeGrid(i, j + 1), name, art ? 'Camino Mock Parkway' : '', art ? 'primary' : 'residential', art ? 2 : 1);
        addEdge(nodeGrid(i, j + 1), nodeGrid(i, j), name, art ? 'Camino Mock Parkway' : '', art ? 'primary' : 'residential', art ? 2 : 1);
      }
      if (i < N - 1) {
        const art = j === mid || j === 2;
        const name = j === mid ? 'Mock Ranch Road' : j === 2 ? 'Test Canyon Drive' : `Mock Avenue ${j + 1}`;
        const hw = art ? 'secondary' : 'residential';
        addEdge(nodeGrid(i, j), nodeGrid(i + 1, j), name, art ? name : '', hw, art ? 2 : 1);
        addEdge(nodeGrid(i + 1, j), nodeGrid(i, j), name, art ? name : '', hw, art ? 2 : 1);
      }
    }
  }
  const network: NetworkJson = { n_edges: edges.length, edges, nodes };

  const mkSchool = (id: string, name: string, i: number, j: number, bell: string, students: number, grades: number[]): School => {
    const n = nodeById.get(nodeGrid(i, j))!;
    const cx = n.x + GRID / 2;
    const cz = n.z + GRID / 2;
    return {
      id,
      name,
      grades,
      bell_start: bell,
      x: cx,
      y: terrainHeight(cx, cz),
      z: cz,
      verified: false,
      students,
      entrances: [
        {
          id: 'main_dropoff',
          key: `${id}/main_dropoff`,
          x: n.x,
          y: n.y,
          z: n.z,
          curb_spots: 8,
          unload_seconds: 45,
          verified: false,
          baseline: { max_queue_cars: 24, max_spillback_m: 168, avg_wait_min: 5.1 },
        },
      ],
    };
  };
  const schools = [
    mkSchool('mock_high', 'Mock High School (fixture)', 3, 7, '08:30', 2400, [9, 12]),
    mkSchool('mock_middle', 'Mock Middle School (fixture)', 7, 2, '08:00', 1100, [6, 8]),
    mkSchool('mock_k8', 'Mock K-8 Campus (fixture)', 6, 6, '08:15', 900, [0, 8]),
  ];

  // buildings: houses on blocks, school buildings on school blocks
  const buildings: MockWorld['buildings'] = new Map();
  const r = rng(7);
  let id = 1;
  const tileOf = (x: number, z: number): string => {
    const col = Math.min(TILES - 1, Math.max(0, Math.floor(((x + MOCK_HALF) / (2 * MOCK_HALF)) * TILES)));
    const row = Math.min(TILES - 1, Math.max(0, Math.floor(((z + MOCK_HALF) / (2 * MOCK_HALF)) * TILES)));
    return `r${row}_c${col}`;
  };
  for (let i = 0; i < N - 1; i++) {
    for (let j = 0; j < N - 1; j++) {
      const n = nodeById.get(nodeGrid(i, j))!;
      const school = schools.find((s) => Math.abs(s.x - (n.x + GRID / 2)) < 1 && Math.abs(s.z - (n.z + GRID / 2)) < 1);
      if (school) {
        const parts: Array<[number, number, number, number]> = [
          [-40, -30, 70, 40],
          [40, 20, 50, 60],
          [-30, 50, 60, 30],
        ];
        for (const [dx, dz, w, d] of parts) {
          const x = school.x + dx;
          const z = school.z + dz;
          buildings.set(id, {
            id, type: 'school', height_m: 9, levels: 2, address: null, name: school.name, area_m2: w * d,
            centroid_x: x, centroid_z: z, base_elev_m: terrainHeight(x, z), school_id: school.id,
            households: 0, block: `Mock block ${i + 1}-${j + 1}`, tile: tileOf(x, z), x, z, w, d, h: 9,
          });
          id++;
        }
        continue;
      }
      const commercial = i === Math.floor(N / 2) && (j === 4 || j === 5);
      const per = commercial ? 2 : 3;
      for (let a = 0; a < per; a++) {
        for (let b = 0; b < per; b++) {
          if (r() < 0.12) continue;
          const cell = (GRID - 60) / per;
          const x = n.x + 30 + cell * (a + 0.5) + (r() - 0.5) * 8;
          const z = n.z + 30 + cell * (b + 0.5) + (r() - 0.5) * 8;
          const w = commercial ? 50 + r() * 20 : 12 + r() * 8;
          const d = commercial ? 40 + r() * 20 : 10 + r() * 8;
          const h = commercial ? 8 + r() * 4 : 6 + r() * 3;
          buildings.set(id, {
            id, type: commercial ? 'commercial' : r() < 0.1 ? 'apartments' : 'house', height_m: Math.round(h * 10) / 10,
            levels: commercial ? 1 : 2, address: null, name: commercial ? 'Mock Commons (fixture)' : null, area_m2: Math.round(w * d),
            centroid_x: x, centroid_z: z, base_elev_m: terrainHeight(x, z), school_id: null,
            households: commercial ? 0 : 1, block: `Mock block ${i + 1}-${j + 1}`, tile: tileOf(x, z), x, z, w, d, h,
          });
          id++;
        }
      }
    }
  }
  const exitNodes = [nodeGrid(0, mid), nodeGrid(N - 1, mid), nodeGrid(mid, 0), nodeGrid(mid, N - 1)];
  cached = { network, schools, buildings, exitNodes, nodeGrid, edgeByUV };
  return cached;
}

export function tileBounds(row: number, col: number): { min_x: number; max_x: number; min_z: number; max_z: number } {
  const size = (2 * MOCK_HALF) / TILES;
  return { min_x: -MOCK_HALF + col * size, max_x: -MOCK_HALF + (col + 1) * size, min_z: -MOCK_HALF + row * size, max_z: -MOCK_HALF + (row + 1) * size };
}

export function mockManifest(): Manifest {
  const tiles: Manifest['tiles'] = [];
  for (let row = 0; row < TILES; row++) {
    for (let col = 0; col < TILES; col++) {
      tiles.push({
        id: `r${row}_c${col}`, row, col,
        bounds: { ...tileBounds(row, col), min_y: 120, max_y: 260 },
        terrain: `terrain/terrain_r${row}_c${col}.glb`,
        buildings: `buildings/buildings_r${row}_c${col}.glb`,
      });
    }
  }
  return { contract_version: 1, synthetic: true, draco: false, tiles, roads: ['roads/roads.glb'], terrain_meta: 'terrain/terrain_meta.json', triangles: {} };
}

async function exportGlb(obj: THREE.Object3D): Promise<ArrayBuffer> {
  const exporter = new GLTFExporter();
  const res = await exporter.parseAsync(obj, { binary: true });
  return res as ArrayBuffer;
}

function albedoTexture(row: number, col: number): THREE.CanvasTexture {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  const r = rng(100 + row * 10 + col);
  g.fillStyle = '#8d8a63';
  g.fillRect(0, 0, size, size);
  for (let k = 0; k < 900; k++) {
    const shade = r();
    g.fillStyle = shade < 0.5 ? `rgba(92,110,62,${0.25 + r() * 0.3})` : `rgba(170,150,110,${0.2 + r() * 0.3})`;
    const s = 4 + r() * 18;
    g.beginPath();
    g.arc(r() * size, r() * size, s, 0, Math.PI * 2);
    g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export async function terrainGlb(row: number, col: number): Promise<ArrayBuffer> {
  const b = tileBounds(row, col);
  const seg = 48;
  const g = new THREE.PlaneGeometry(b.max_x - b.min_x, b.max_z - b.min_z, seg, seg);
  g.rotateX(-Math.PI / 2);
  const pos = g.getAttribute('position') as THREE.BufferAttribute;
  const cx = (b.min_x + b.max_x) / 2;
  const cz = (b.min_z + b.max_z) / 2;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) + cx;
    const z = pos.getZ(i) + cz;
    pos.setXYZ(i, x, terrainHeight(x, z), z);
  }
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ map: albedoTexture(row, col), roughness: 1, metalness: 0 }));
  return exportGlb(mesh);
}

const TYPE_COLORS: Record<string, number> = { house: 0xd9cbb4, apartments: 0xc9b9a6, commercial: 0xb7c2cc, school: 0xe0b46e, other: 0xbbbbbb };

export async function buildingsGlb(row: number, col: number): Promise<ArrayBuffer> {
  const w = mockWorld();
  const tile = `r${row}_c${col}`;
  const parts: THREE.BufferGeometry[] = [];
  const color = new THREE.Color();
  for (const bd of w.buildings.values()) {
    if (bd.tile !== tile) continue;
    const base = terrainHeight(bd.x, bd.z) - 1;
    const g = new THREE.BoxGeometry(bd.w, bd.h + 1, bd.d).translate(bd.x, base + (bd.h + 1) / 2, bd.z).toNonIndexed();
    g.deleteAttribute('uv');
    const n = g.getAttribute('position').count;
    const ids = new Float32Array(n).fill(bd.id);
    const cols = new Float32Array(n * 3);
    color.setHex(TYPE_COLORS[bd.type ?? 'other'] ?? 0xbbbbbb);
    for (let k = 0; k < n; k++) {
      // darker walls, lighter roofs
      const ny = (g.getAttribute('normal') as THREE.BufferAttribute).getY(k);
      const f = ny > 0.5 ? 1 : 0.82;
      cols[k * 3] = color.r * f;
      cols[k * 3 + 1] = color.g * f;
      cols[k * 3 + 2] = color.b * f;
    }
    g.setAttribute('_building_id', new THREE.BufferAttribute(ids, 1));
    g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    parts.push(g);
  }
  const merged = parts.length ? mergeGeometries(parts)! : new THREE.BufferGeometry();
  const mesh = new THREE.Mesh(merged, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9 }));
  return exportGlb(mesh);
}

export async function roadsGlb(): Promise<ArrayBuffer> {
  const w = mockWorld();
  const parts: THREE.BufferGeometry[] = [];
  const seen = new Set<string>();
  for (const e of w.network.edges) {
    const key = [Math.min(e.u, e.v), Math.max(e.u, e.v)].join('-');
    if (seen.has(key)) continue;
    seen.add(key);
    const width = e.lanes * 2 * 3.6;
    const pos: number[] = [];
    const cols: number[] = [];
    const c = e.highway === 'residential' ? [0.32, 0.32, 0.33] : [0.27, 0.27, 0.28];
    const n = e.pts.length / 3;
    for (let k = 0; k < n - 1; k++) {
      const ax = e.pts[k * 3]!;
      const az = e.pts[k * 3 + 2]!;
      const bx = e.pts[(k + 1) * 3]!;
      const bz = e.pts[(k + 1) * 3 + 2]!;
      const L = Math.hypot(bx - ax, bz - az) || 1;
      const px = (-(bz - az) / L) * width * 0.5;
      const pz = ((bx - ax) / L) * width * 0.5;
      const ay = e.pts[k * 3 + 1]!;
      const by = e.pts[(k + 1) * 3 + 1]!;
      const quad = [
        [ax - px, ay, az - pz], [ax + px, ay, az + pz], [bx - px, by, bz - pz],
        [ax + px, ay, az + pz], [bx + px, by, bz + pz], [bx - px, by, bz - pz],
      ];
      for (const q of quad) {
        pos.push(q[0]!, q[1]!, q[2]!);
        cols.push(c[0]!, c[1]!, c[2]!);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    g.computeVertexNormals();
    parts.push(g);
  }
  const merged = mergeGeometries(parts)!;
  const mesh = new THREE.Mesh(merged, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, side: THREE.DoubleSide }));
  return exportGlb(mesh);
}

export function mockBbox(): { south: number; north: number; west: number; east: number } {
  const sw = sceneToLatLon(-MOCK_HALF, MOCK_HALF, MOCK_ORIGIN);
  const ne = sceneToLatLon(MOCK_HALF, -MOCK_HALF, MOCK_ORIGIN);
  return { south: sw.lat, north: ne.lat, west: sw.lon, east: ne.lon };
}
