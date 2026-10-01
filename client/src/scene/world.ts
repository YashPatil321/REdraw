/**
 * Static world: terrain, building and road tiles from the asset manifest
 * (GLTFLoader + DRACOLoader; works with draco true or false), per-tile
 * frustum culling, and raycast picking (buildings by `_building_id`, ground).
 */

import * as THREE from 'three';
import { DRACOLoader, DRACO_GLTF_CONFIG } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { Manifest } from '../types';
import { buildingRoofMaterial, buildingWallMaterial, hasBuildingAtlases } from './buildingMaterial';
import { prepareBuildingGeometry, type StreetDirFn } from './buildingPrep';
import type { MaterialLibrary } from './materials';
import { terrainMaterial as atlasTerrainMaterial } from './terrainMaterial';
import { addBuildingExtents, legacyBuildingMaterial, legacyRoadMaterial, legacyTerrainMaterial } from './legacyMaterials';

export type AssetFetcher = (rel: string) => Promise<ArrayBuffer>;

interface Tile {
  id: string;
  box: THREE.Box3;
  terrain: THREE.Object3D | null;
  buildings: THREE.Object3D | null;
}

export interface BuildingHit {
  id: number;
  point: THREE.Vector3;
}

let sharedDraco: DRACOLoader | null = null;
function dracoLoader(): DRACOLoader {
  if (!sharedDraco) {
    sharedDraco = new DRACOLoader();
    // decoder files resolved and emitted by the bundler (works in dev and build)
    sharedDraco.setDecoderPath(DRACO_GLTF_CONFIG);
  }
  return sharedDraco;
}

export class World {
  readonly group = new THREE.Group();
  readonly tiles: Tile[] = [];
  readonly terrainMeshes: THREE.Mesh[] = [];
  readonly buildingMeshes: THREE.Mesh[] = [];
  readonly roadMeshes: THREE.Mesh[] = [];
  private loader: GLTFLoader;
  private frustum = new THREE.Frustum();
  private projView = new THREE.Matrix4();
  /** world bounds of everything loaded */
  readonly bounds = new THREE.Box3();

  constructor() {
    this.group.name = 'world';
    this.loader = new GLTFLoader();
    this.loader.setDRACOLoader(dracoLoader());
  }

  private async parse(buf: ArrayBuffer): Promise<THREE.Group> {
    const gltf = await this.loader.parseAsync(buf, '');
    return gltf.scene;
  }

  private meshes(root: THREE.Object3D): THREE.Mesh[] {
    const out: THREE.Mesh[] = [];
    root.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) out.push(o as THREE.Mesh);
    });
    return out;
  }

  /** PBR atlases (blender/), or null for the procedural fallback shaders */
  materials: MaterialLibrary | null = null;
  /** direction toward the nearest street (front walls get the garage / entry); set by the controller */
  streetDir: StreetDirFn | null = null;
  /** resolves when streetDir is usable (the road network arrived) */
  streetReady: Promise<void> = Promise.resolve();
  private interiors = true;
  private atlasWall: THREE.Material | null = null;
  private atlasRoof: THREE.Material | null = null;

  /** Shared wall / roof materials from the atlases (created once). */
  private atlasBuildingMaterials(): [THREE.Material, THREE.Material] | null {
    if (!hasBuildingAtlases(this.materials)) return null;
    if (!this.atlasWall || !this.atlasRoof) {
      this.atlasWall = buildingWallMaterial(this.materials, { interiors: this.interiors, standard: true });
      this.atlasRoof = buildingRoofMaterial(this.materials);
    }
    return [this.atlasWall, this.atlasRoof];
  }

  /** shared uniforms for terrain tinting (height range of the loaded world) */
  private terrainUniforms = { uMinH: { value: 0 }, uMaxH: { value: 500 } };
  private standardBuildings = true;

  terrainDetail: 0 | 1 | 2 = 1;

  /** Atlas terrain when the ground atlas is present (and detail is on), else the procedural one. */
  private makeTerrainMaterial(old: THREE.MeshStandardMaterial): THREE.Material {
    const map = (old.map ?? null) as THREE.Texture | null;
    if (this.materials?.atlas('ground') && this.terrainDetail > 0 && map) {
      return atlasTerrainMaterial(this.materials, map, null, { detail: this.terrainDetail, standard: this.standardBuildings });
    }
    return legacyTerrainMaterial(old, this.terrainUniforms);
  }

  /** Switch building shading and shadow flags for a quality preset. */
  applyQuality(opts: { standardBuildings: boolean; shadows: boolean; buildingDrawDistance: number; interiors?: boolean; terrainDetail?: 0 | 1 | 2 }): void {
    this.buildingDrawDistance = opts.buildingDrawDistance;
    const td = opts.terrainDetail ?? this.terrainDetail;
    const prevStandard = this.standardBuildings;
    if (td !== this.terrainDetail || opts.standardBuildings !== prevStandard) {
      this.terrainDetail = td;
      this.standardBuildings = opts.standardBuildings;
      for (const m of this.terrainMeshes) {
        const old = m.material as THREE.Material;
        const map = (m.userData.map ?? null) as THREE.Texture | null;
        m.material = this.makeTerrainMaterial({ map } as THREE.MeshStandardMaterial);
        old.dispose();
      }
      this.standardBuildings = prevStandard;
    }
    this.interiors = opts.interiors ?? this.interiors;
    if (this.atlasWall) {
      const u = (this.atlasWall as THREE.Material & { rdUniforms?: { uInteriors: { value: number } } }).rdUniforms;
      if (u) u.uInteriors.value = this.interiors ? 1 : 0;
    }
    if (opts.standardBuildings !== this.standardBuildings && !this.atlasWall) {
      this.standardBuildings = opts.standardBuildings;
      for (const m of this.buildingMeshes) {
        if (m.userData.hero) continue;
        const old = m.material as THREE.Material;
        m.material = legacyBuildingMaterial(null, this.standardBuildings, Boolean(m.geometry.getAttribute('color')));
        old.dispose();
      }
    }
    for (const m of this.terrainMeshes) m.receiveShadow = opts.shadows;
    for (const m of this.roadMeshes) m.receiveShadow = opts.shadows;
    for (const m of this.buildingMeshes) {
      m.castShadow = opts.shadows;
      m.receiveShadow = opts.shadows;
    }
  }

  private buildingDrawDistance = 9000;

  async load(manifest: Manifest, fetchAsset: AssetFetcher, onProgress: (msg: string, frac: number) => void): Promise<void> {
    this.group.clear();
    const jobs: Array<{ kind: 'terrain' | 'buildings' | 'roads'; rel: string; tile?: Tile }> = [];
    for (const t of manifest.tiles) {
      const b = t.bounds;
      const tile: Tile = {
        id: t.id,
        box: new THREE.Box3(new THREE.Vector3(b.min_x, b.min_y ?? -100, b.min_z), new THREE.Vector3(b.max_x, (b.max_y ?? 1000) + 60, b.max_z)),
        terrain: null,
        buildings: null,
      };
      this.tiles.push(tile);
      this.bounds.union(tile.box);
      if (t.terrain) jobs.push({ kind: 'terrain', rel: t.terrain, tile });
    }
    for (const t of this.tiles) {
      const mt = manifest.tiles.find((x) => x.id === t.id);
      if (mt?.buildings) jobs.push({ kind: 'buildings', rel: mt.buildings, tile: t });
    }
    for (const r of manifest.roads ?? []) jobs.push({ kind: 'roads', rel: r });

    let done = 0;
    const errors: string[] = [];
    const runJob = async (job: (typeof jobs)[number]): Promise<void> => {
      try {
        const root = await this.parse(await fetchAsset(job.rel));
        root.name = `${job.kind}:${job.rel}`;
        const meshes = this.meshes(root);
        for (const m of meshes) {
          m.matrixAutoUpdate = false;
          m.updateMatrix();
          m.geometry.computeBoundingSphere();
          m.geometry.computeBoundingBox();
          const old = m.material as THREE.MeshStandardMaterial;
          const hasColor = Boolean(m.geometry.getAttribute('color'));
          if (job.kind === 'terrain') {
            if (!m.geometry.getAttribute('normal')) m.geometry.computeVertexNormals();
            m.userData.map = old.map ?? null;
            m.material = this.makeTerrainMaterial(old);
            m.renderOrder = 0;
            this.terrainMeshes.push(m);
          } else if (job.kind === 'buildings') {
            const id = m.geometry.getAttribute('_building_id');
            // alias without the leading underscore for use in shaders (same buffer)
            if (id) m.geometry.setAttribute('buildingId', id);
            else m.geometry.setAttribute('buildingId', new THREE.BufferAttribute(new Float32Array(m.geometry.getAttribute('position').count), 1));
            if (/^hero/i.test(m.name) || /^hero/i.test(m.parent?.name ?? '')) {
              // Blender hero campuses carry their own palette (vertex colors, textures):
              // keep it, only make it a lit PBR surface; still pickable by building id
              const hm = new THREE.MeshStandardMaterial({
                map: old.map ?? null,
                color: old.color ?? new THREE.Color(1, 1, 1),
                vertexColors: hasColor,
                roughness: old.roughness ?? 0.85,
                metalness: old.metalness ?? 0,
                side: old.side,
              });
              if (hm.map) hm.map.colorSpace = THREE.SRGBColorSpace;
              m.material = hm;
              m.userData.hero = true;
            } else {
              const atlas = this.atlasBuildingMaterials();
              if (atlas) {
                // front walls need the street direction: wait for the network (bounded)
                await Promise.race([this.streetReady, new Promise((r) => setTimeout(r, 20000))]);
                const lib = this.materials!;
                prepareBuildingGeometry(m.geometry, {
                  streetDir: this.streetDir ?? undefined,
                  variants: { wall: lib.variants(0).length, tileRoof: lib.variants(1).length, flatRoof: lib.variants(2).length },
                });
                m.material = atlas;
              } else {
                addBuildingExtents(m.geometry);
                m.material = legacyBuildingMaterial(old, this.standardBuildings, hasColor);
              }
            }
            this.buildingMeshes.push(m);
          } else {
            m.material = legacyRoadMaterial(old, hasColor);
            m.renderOrder = 1;
            this.roadMeshes.push(m);
          }
          if (m.material !== old && !Array.isArray(m.material)) old.dispose();
          else if (Array.isArray(m.material)) old.dispose();
        }
        root.updateMatrixWorld(true);
        if (job.tile) {
          if (job.kind === 'terrain') job.tile.terrain = root;
          else job.tile.buildings = root;
        }
        this.group.add(root);
      } catch (e) {
        errors.push(`${job.rel}: ${(e as Error).message}`);
      } finally {
        done++;
        onProgress(`Loading world ${done}/${jobs.length}`, done / jobs.length);
      }
    };
    // limited concurrency, terrain first
    const queue = [...jobs];
    const workers = Array.from({ length: 4 }, async () => {
      while (queue.length) await runJob(queue.shift()!);
    });
    await Promise.all(workers);
    const hb = new THREE.Box3();
    for (const m of this.terrainMeshes) if (m.geometry.boundingBox) hb.union(m.geometry.boundingBox);
    if (!hb.isEmpty()) {
      this.terrainUniforms.uMinH.value = hb.min.y;
      this.terrainUniforms.uMaxH.value = hb.max.y;
    }
    this.buildHeightGrid();
    if (errors.length) console.warn('Some world assets failed to load:', errors);
    if (errors.length === jobs.length && jobs.length > 0) throw new Error(`world assets failed to load: ${errors[0]}`);
  }

  /** Per-tile frustum culling (plus hiding far building tiles). */
  updateCulling(camera: THREE.PerspectiveCamera): void {
    this.projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView);
    const camPos = camera.position;
    for (const t of this.tiles) {
      const inView = this.frustum.intersectsBox(t.box);
      if (t.terrain) t.terrain.visible = inView;
      if (t.buildings) {
        const d = t.box.distanceToPoint(camPos);
        t.buildings.visible = inView && d < this.buildingDrawDistance;
      }
    }
  }

  pickBuilding(raycaster: THREE.Raycaster, includeHidden = false): BuildingHit | null {
    const targets = includeHidden ? this.buildingMeshes : this.buildingMeshes.filter((m) => m.parent?.visible !== false && m.visible);
    const hits = raycaster.intersectObjects(targets, false);
    for (const h of hits) {
      const mesh = h.object as THREE.Mesh;
      const attr = mesh.geometry.getAttribute('_building_id');
      if (!attr || !h.face) continue;
      const id = Math.round(attr.getX(h.face.a));
      if (id > 0) return { id, point: h.point.clone() };
    }
    return null;
  }

  /** Ground point under a ray (terrain first, then roads), or null. */
  pickGround(raycaster: THREE.Raycaster): THREE.Vector3 | null {
    const targets = this.terrainMeshes.filter((m) => m.parent?.visible !== false);
    const hit = raycaster.intersectObjects(targets, false)[0] ?? raycaster.intersectObjects(this.roadMeshes, false)[0];
    return hit ? hit.point.clone() : null;
  }

  /** Is the first hit a building (so a click should open the building panel)? */
  firstHitIsBuilding(raycaster: THREE.Raycaster): boolean {
    const b = raycaster.intersectObjects(this.buildingMeshes, false)[0];
    if (!b) return false;
    const g = raycaster.intersectObjects(this.terrainMeshes, false)[0];
    return !g || b.distance <= g.distance + 0.5;
  }

  /** Terrain height at (x,z) by a vertical ray; falls back to `fallback`. */
  heightAt(x: number, z: number, fallback = 0): number {
    const rc = new THREE.Raycaster(new THREE.Vector3(x, 5000, z), new THREE.Vector3(0, -1, 0), 0, 10000);
    const hit = rc.intersectObjects(this.terrainMeshes, false)[0];
    return hit ? hit.point.y : fallback;
  }

  // ---- fast lookups (walk camera, photoreal picking)

  private hgrid: { minX: number; minZ: number; cell: number; nx: number; nz: number; data: Float32Array } | null = null;

  /** Rasterize terrain vertices into a regular height grid (bilinear lookups in O(1)). */
  private buildHeightGrid(): void {
    const b = new THREE.Box3();
    let nVerts = 0;
    for (const m of this.terrainMeshes) {
      if (m.geometry.boundingBox) b.union(m.geometry.boundingBox);
      nVerts += m.geometry.getAttribute('position').count;
    }
    if (b.isEmpty() || nVerts === 0) return;
    const area = (b.max.x - b.min.x) * (b.max.z - b.min.z);
    const cell = THREE.MathUtils.clamp(Math.sqrt(area / nVerts) * 1.05, 2, 40);
    const nx = Math.ceil((b.max.x - b.min.x) / cell) + 1;
    const nz = Math.ceil((b.max.z - b.min.z) / cell) + 1;
    const sum = new Float32Array(nx * nz);
    const cnt = new Uint16Array(nx * nz);
    for (const m of this.terrainMeshes) {
      const p = m.geometry.getAttribute('position');
      for (let i = 0; i < p.count; i++) {
        const ix = Math.round((p.getX(i) - b.min.x) / cell);
        const iz = Math.round((p.getZ(i) - b.min.z) / cell);
        const k = iz * nx + ix;
        sum[k]! += p.getY(i);
        cnt[k]!++;
      }
    }
    const data = new Float32Array(nx * nz).fill(NaN);
    for (let k = 0; k < data.length; k++) if (cnt[k]) data[k] = sum[k]! / cnt[k]!;
    // fill holes from neighbors (a few passes)
    for (let pass = 0; pass < 6; pass++) {
      let holes = 0;
      for (let iz = 0; iz < nz; iz++) {
        for (let ix = 0; ix < nx; ix++) {
          const k = iz * nx + ix;
          if (!Number.isNaN(data[k]!)) continue;
          let acc = 0;
          let n = 0;
          for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
            const jx = ix + dx;
            const jz = iz + dz;
            if (jx < 0 || jz < 0 || jx >= nx || jz >= nz) continue;
            const v = data[jz * nx + jx]!;
            if (!Number.isNaN(v)) {
              acc += v;
              n++;
            }
          }
          if (n) data[k] = acc / n;
          else holes++;
        }
      }
      if (!holes) break;
    }
    this.hgrid = { minX: b.min.x, minZ: b.min.z, cell, nx, nz, data };
  }

  /** Terrain height from the grid (no raycast), or null outside the terrain. */
  fastHeightAt(x: number, z: number): number | null {
    const g = this.hgrid;
    if (!g) return null;
    const fx = (x - g.minX) / g.cell;
    const fz = (z - g.minZ) / g.cell;
    if (fx < 0 || fz < 0 || fx > g.nx - 1 || fz > g.nz - 1) return null;
    const ix = Math.min(g.nx - 2, Math.floor(fx));
    const iz = Math.min(g.nz - 2, Math.floor(fz));
    const tx = fx - ix;
    const tz = fz - iz;
    const d = g.data;
    const h00 = d[iz * g.nx + ix]!;
    const h10 = d[iz * g.nx + ix + 1]!;
    const h01 = d[(iz + 1) * g.nx + ix]!;
    const h11 = d[(iz + 1) * g.nx + ix + 1]!;
    const v = (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
    return Number.isNaN(v) ? null : v;
  }

  private bIndex: { cell: number; map: Map<number, Array<{ id: number; x: number; z: number; top: number }>> } | null = null;

  /** Per-building centroid (from `_building_id` vertices), on a grid. */
  private buildBuildingIndex(): void {
    const acc = new Map<number, { x: number; z: number; n: number; top: number }>();
    for (const m of this.buildingMeshes) {
      const id = m.geometry.getAttribute('_building_id');
      const p = m.geometry.getAttribute('position');
      if (!id) continue;
      for (let i = 0; i < p.count; i++) {
        const b = Math.round(id.getX(i));
        if (b <= 0) continue;
        let a = acc.get(b);
        if (!a) acc.set(b, (a = { x: 0, z: 0, n: 0, top: -Infinity }));
        a.x += p.getX(i);
        a.z += p.getZ(i);
        a.n++;
        a.top = Math.max(a.top, p.getY(i));
      }
    }
    const cell = 50;
    const map = new Map<number, Array<{ id: number; x: number; z: number; top: number }>>();
    for (const [id, a] of acc) {
      const x = a.x / a.n;
      const z = a.z / a.n;
      const k = (Math.floor(x / cell) + 32768) * 65536 + (Math.floor(z / cell) + 32768);
      let arr = map.get(k);
      if (!arr) map.set(k, (arr = []));
      arr.push({ id, x, z, top: a.top });
    }
    this.bIndex = { cell, map };
  }

  /** Nearest building (by footprint centroid) to x, z within maxDist. */
  nearestBuilding(x: number, z: number, maxDist = 30): { id: number; x: number; z: number; top: number } | null {
    if (!this.bIndex) this.buildBuildingIndex();
    const { cell, map } = this.bIndex!;
    const r = Math.ceil(maxDist / cell);
    const cx = Math.floor(x / cell);
    const cz = Math.floor(z / cell);
    let best: { id: number; x: number; z: number; top: number } | null = null;
    let bd = maxDist;
    for (let ix = cx - r; ix <= cx + r; ix++) {
      for (let iz = cz - r; iz <= cz + r; iz++) {
        const arr = map.get((ix + 32768) * 65536 + (iz + 32768));
        if (!arr) continue;
        for (const b of arr) {
          const d = Math.hypot(b.x - x, b.z - z);
          if (d < bd) {
            bd = d;
            best = b;
          }
        }
      }
    }
    return best;
  }

  /** Centroid of a building id (for "walk here"), or null. */
  buildingCentroid(id: number): { x: number; z: number; top: number } | null {
    if (!this.bIndex) this.buildBuildingIndex();
    for (const arr of this.bIndex!.map.values()) for (const b of arr) if (b.id === id) return b;
    return null;
  }

  dispose(): void {
    this.group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.geometry.dispose();
      const mat = m.material as THREE.MeshLambertMaterial;
      mat.map?.dispose();
      mat.dispose();
    });
    this.group.clear();
  }
}
