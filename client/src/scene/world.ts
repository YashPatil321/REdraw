/**
 * Static world from the asset manifest (GLTFLoader + DRACOLoader; works with
 * draco true or false):
 *
 * - terrain: per tile, the coarsest LOD is loaded up front for full coverage;
 *   finer LODs (HD build: LOD0 RTIN, LOD1 10 m grid) stream in by camera
 *   distance and coarse ones are shown until they arrive (skirts hide cracks).
 *   Landcover splat masks drive the PBR ground detail on near tiles.
 * - streets (HD build): per-tile road surfaces + lane markings and
 *   sidewalks / driveways / medians / pools, draped on LOD0, shown with it.
 * - buildings: Blender HD tiles (buildings_hd, LOD0 near / LOD1 far) when
 *   built, else the pipeline's building tiles; atlas PBR materials.
 * - picking (buildings by `_building_id`, ground) and fast height lookups
 *   (BVH on LOD0, else a height grid from the coarse terrain).
 */

import * as THREE from 'three';
import { DRACOLoader, DRACO_GLTF_CONFIG } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshBVH } from 'three-mesh-bvh';
import type { Manifest } from '../types';
import { buildingRoofMaterial, buildingWallMaterial, hasBuildingAtlases } from './buildingMaterial';
import { prepareBuildingGeometry, type StreetDirFn } from './buildingPrep';
import { ensureVariant, groundMaterial, markingsMaterial } from './groundMaterial';
import { parseHdBuildingsManifest, type HdBuildingsManifest } from './hdManifest';
import { addBuildingExtents, legacyBuildingMaterial, legacyRoadMaterial, legacyTerrainMaterial } from './legacyMaterials';
import { loadTexture, type MaterialLibrary } from './materials';
import { terrainMaterial as atlasTerrainMaterial } from './terrainMaterial';

export type AssetFetcher = (rel: string) => Promise<ArrayBuffer>;

interface TerrainLod {
  lod: number;
  path: string;
  root: THREE.Object3D | null;
  meshes: THREE.Mesh[];
  loading: boolean;
  failed: boolean;
}

interface LayerSlot {
  path: string;
  root: THREE.Object3D | null;
  loading: boolean;
  failed: boolean;
}

interface Tile {
  id: string;
  box: THREE.Box3;
  /** terrain LODs, finest (0) first */
  lods: TerrainLod[];
  shownLod: number;
  splatPaths: string[];
  splat: [THREE.Texture, THREE.Texture] | null;
  splatLoading: boolean;
  streets: LayerSlot[];
  /** pipeline building tile (fallback when no HD tile) */
  buildings: LayerSlot | null;
  /** HD building tiles by LOD level */
  hd: Array<LayerSlot | null>;
  dist: number;
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

/** LOD switch distances (m) at lodScale 1 (manifest suggestions win). */
const TERRAIN_LOD0_M = 900;
const TERRAIN_LOD1_M = 2600;
const HD_LOD0_M = 450;

export class World {
  readonly group = new THREE.Group();
  readonly tiles: Tile[] = [];
  /** every loaded terrain mesh (all LODs) */
  readonly terrainMeshes: THREE.Mesh[] = [];
  readonly buildingMeshes: THREE.Mesh[] = [];
  readonly roadMeshes: THREE.Mesh[] = [];
  private loader: GLTFLoader;
  private frustum = new THREE.Frustum();
  private projView = new THREE.Matrix4();
  /** world bounds of everything loaded */
  readonly bounds = new THREE.Box3();
  private fetchAsset: AssetFetcher | null = null;
  /** pipeline street meshes (roads, sidewalks, markings) are part of the world */
  hasStreets = false;
  /** Blender HD buildings are in use */
  hasHdBuildings = false;
  private hdInfo: HdBuildingsManifest | null = null;
  private lodScale = 1;
  private terrainLod0 = TERRAIN_LOD0_M;
  private terrainLod1 = TERRAIN_LOD1_M;
  private hdLod0 = HD_LOD0_M;
  private queue: Array<{ key: string; pri: number; run: () => Promise<void> }> = [];
  private queued = new Set<string>();
  private active = 0;
  private lastStream = 0;
  private camPos = new THREE.Vector3();
  /** called when the set of drawn meshes changed (shadows / AO may want a refresh) */
  onChange: (() => void) | null = null;

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
  private groundMat: THREE.Material | null = null;
  private markMat: THREE.Material | null = null;
  private shadows = true;

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
  private makeTerrainMaterial(map: THREE.Texture | null, splat: [THREE.Texture, THREE.Texture] | null): THREE.Material {
    if (this.materials?.atlas('ground') && this.terrainDetail > 0 && map) {
      return atlasTerrainMaterial(this.materials, map, splat, { detail: this.terrainDetail, standard: this.standardBuildings });
    }
    return legacyTerrainMaterial({ map }, this.terrainUniforms);
  }

  private setTerrainMaterial(m: THREE.Mesh, splat: [THREE.Texture, THREE.Texture] | null): void {
    const old = m.material as THREE.Material;
    m.material = this.makeTerrainMaterial((m.userData.map ?? null) as THREE.Texture | null, splat);
    if (old !== m.material) old.dispose();
  }

  /** Switch shading, shadow flags and LOD distances for a quality preset. */
  applyQuality(opts: {
    standardBuildings: boolean;
    shadows: boolean;
    buildingDrawDistance: number;
    interiors?: boolean;
    terrainDetail?: 0 | 1 | 2;
    lodScale?: number;
  }): void {
    this.buildingDrawDistance = opts.buildingDrawDistance;
    this.lodScale = opts.lodScale ?? this.lodScale;
    const td = opts.terrainDetail ?? this.terrainDetail;
    const prevStandard = this.standardBuildings;
    if (td !== this.terrainDetail || opts.standardBuildings !== prevStandard) {
      this.terrainDetail = td;
      this.standardBuildings = opts.standardBuildings;
      for (const t of this.tiles) for (const l of t.lods) for (const m of l.meshes) this.setTerrainMaterial(m, l.lod <= 1 ? t.splat : null);
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
    this.standardBuildings = opts.standardBuildings;
    this.shadows = opts.shadows;
    for (const m of this.terrainMeshes) m.receiveShadow = opts.shadows;
    for (const m of this.roadMeshes) m.receiveShadow = opts.shadows;
    for (const m of this.buildingMeshes) {
      m.castShadow = opts.shadows;
      m.receiveShadow = opts.shadows;
    }
    this.lastStream = 0;
  }

  private buildingDrawDistance = 9000;

  async load(manifest: Manifest, fetchAsset: AssetFetcher, onProgress: (msg: string, frac: number) => void): Promise<void> {
    this.group.clear();
    this.fetchAsset = fetchAsset;
    const sw = manifest.terrain_lod?.suggested_switch_distance_m;
    if (sw?.['0']) this.terrainLod0 = sw['0'];
    if (sw?.['1']) this.terrainLod1 = sw['1'];
    // Blender HD buildings (optional)
    const hdRel = manifest.buildings_hd ?? 'buildings_hd/manifest_buildings.json';
    try {
      const raw = JSON.parse(new TextDecoder().decode(await fetchAsset(hdRel))) as unknown;
      const info = parseHdBuildingsManifest(raw, hdRel.replace(/[^/]*$/, ''));
      if (info.tiles.length) {
        this.hdInfo = info;
        this.hasHdBuildings = true;
        if (info.lod0Distance) this.hdLod0 = info.lod0Distance;
      }
    } catch {
      /* not built yet: pipeline building tiles */
    }
    const perTileStreets = manifest.tiles.some((t) => t.roads || t.ground);
    this.hasStreets = perTileStreets;
    for (const t of manifest.tiles) {
      const b = t.bounds;
      const lods: TerrainLod[] = (t.terrain_lods?.length ? t.terrain_lods : t.terrain ? [{ lod: 0, path: t.terrain }] : [])
        .map((l) => ({ lod: l.lod, path: l.path, root: null, meshes: [], loading: false, failed: false }))
        .sort((a, b2) => a.lod - b2.lod);
      const hdTile = this.hdInfo?.tiles.find((h) => h.id === t.id) ?? null;
      const tile: Tile = {
        id: t.id,
        box: new THREE.Box3(new THREE.Vector3(b.min_x, b.min_y ?? -100, b.min_z), new THREE.Vector3(b.max_x, (b.max_y ?? 1000) + 60, b.max_z)),
        lods,
        shownLod: -1,
        splatPaths: t.splat ?? [],
        splat: null,
        splatLoading: false,
        streets: [t.roads, t.ground].filter((p): p is string => !!p).map((path) => ({ path, root: null, loading: false, failed: false })),
        buildings: t.buildings && !hdTile ? { path: t.buildings, root: null, loading: false, failed: false } : null,
        hd: hdTile ? hdTile.lods.map((p) => (p ? { path: p, root: null, loading: false, failed: false } : null)) : [],
        dist: Infinity,
      };
      this.tiles.push(tile);
      this.bounds.union(tile.box);
    }
    // up front: coarsest terrain everywhere, far buildings everywhere, legacy roads
    const jobs: Array<() => Promise<void>> = [];
    for (const t of this.tiles) {
      const coarse = t.lods[t.lods.length - 1];
      if (coarse) jobs.push(() => this.loadTerrainLod(t, coarse));
    }
    for (const t of this.tiles) {
      if (t.buildings) jobs.push(() => this.loadBuildings(t, t.buildings!, 'pipeline'));
      const far = this.farHdLevel(t);
      if (far >= 0) jobs.push(() => this.loadBuildings(t, t.hd[far]!, 'hd', far));
    }
    if (!perTileStreets) for (const r of manifest.roads ?? []) jobs.push(() => this.loadLegacyRoads(r));
    let done = 0;
    const errors: string[] = [];
    const queue = [...jobs];
    const workers = Array.from({ length: 4 }, async () => {
      while (queue.length) {
        const job = queue.shift()!;
        try {
          await job();
        } catch (e) {
          errors.push((e as Error).message);
        } finally {
          done++;
          onProgress(`Loading world ${done}/${jobs.length}`, done / jobs.length);
        }
      }
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

  /** coarsest HD level present for a tile (drawn far away), or -1 */
  private farHdLevel(t: Tile): number {
    for (let i = t.hd.length - 1; i >= 0; i--) if (t.hd[i]) return i;
    return -1;
  }

  private async loadTerrainLod(t: Tile, l: TerrainLod): Promise<void> {
    if (l.root || l.loading || l.failed) return;
    l.loading = true;
    try {
      const root = await this.parse(await this.fetchAsset!(l.path));
      root.name = `terrain:${t.id}:lod${l.lod}`;
      const meshes = this.meshes(root);
      for (const m of meshes) {
        m.matrixAutoUpdate = false;
        m.updateMatrix();
        m.geometry.computeBoundingSphere();
        m.geometry.computeBoundingBox();
        if (!m.geometry.getAttribute('normal')) m.geometry.computeVertexNormals();
        const old = m.material as THREE.MeshStandardMaterial;
        m.userData.map = old.map ?? null;
        m.userData.tile = t;
        m.material = this.makeTerrainMaterial(old.map ?? null, l.lod <= 1 ? t.splat : null);
        if (m.material !== old) old.dispose();
        m.renderOrder = 0;
        m.receiveShadow = this.shadows;
        this.terrainMeshes.push(m);
      }
      root.updateMatrixWorld(true);
      root.visible = false;
      l.meshes = meshes;
      l.root = root;
      this.group.add(root);
      this.showBestTerrain(t);
    } catch (e) {
      l.failed = true;
      throw new Error(`${l.path}: ${(e as Error).message}`);
    } finally {
      l.loading = false;
    }
  }

  private async loadSplat(t: Tile): Promise<void> {
    if (t.splat || t.splatLoading || t.splatPaths.length < 2 || !this.materials?.atlas('ground')) return;
    t.splatLoading = true;
    try {
      const [a, b] = await Promise.all(t.splatPaths.slice(0, 2).map((p) => loadTexture(this.fetchAsset!, p, false, 1)));
      for (const tex of [a!, b!]) {
        // north up like the albedo (glTF UVs: v down), no mips needed beyond a few
        tex.flipY = false;
        tex.needsUpdate = true;
      }
      t.splat = [a!, b!];
      for (const l of t.lods) if (l.lod <= 1) for (const m of l.meshes) this.setTerrainMaterial(m, t.splat);
    } catch (e) {
      console.warn(`splat ${t.id} unavailable`, e);
      t.splatPaths = [];
    } finally {
      t.splatLoading = false;
    }
  }

  private async loadStreets(t: Tile, s: LayerSlot): Promise<void> {
    if (s.root || s.loading || s.failed) return;
    s.loading = true;
    try {
      const root = await this.parse(await this.fetchAsset!(s.path));
      root.name = `streets:${s.path}`;
      for (const m of this.meshes(root)) {
        m.matrixAutoUpdate = false;
        m.updateMatrix();
        m.geometry.computeBoundingSphere();
        const old = m.material as THREE.MeshStandardMaterial;
        const matAttr = m.geometry.getAttribute('_mat');
        const matId = matAttr ? Math.round(matAttr.getX(0)) : /mark/i.test(m.name) ? 8 : 6;
        ensureVariant(m.geometry);
        const lib = this.materials;
        if (lib?.atlas('ground') && matId === 6) {
          this.groundMat ??= groundMaterial(lib, { polygonOffset: 2, normalMaps: this.terrainDetail > 1 });
          m.material = this.groundMat;
          m.renderOrder = 1;
        } else if (lib && matId === 8) {
          this.markMat ??= markingsMaterial(lib);
          m.material = this.markMat ?? legacyRoadMaterial(old, true);
          m.renderOrder = 2;
        } else {
          m.material = legacyRoadMaterial(old, Boolean(m.geometry.getAttribute('color')));
          m.renderOrder = matId === 8 ? 2 : 1;
        }
        if (m.material !== old) old.dispose();
        m.receiveShadow = this.shadows;
        m.castShadow = false;
        this.roadMeshes.push(m);
      }
      root.updateMatrixWorld(true);
      root.visible = t.shownLod === 0;
      s.root = root;
      this.group.add(root);
    } catch (e) {
      s.failed = true;
      console.warn(`streets ${s.path} failed`, e);
    } finally {
      s.loading = false;
    }
  }

  private async loadLegacyRoads(rel: string): Promise<void> {
    const root = await this.parse(await this.fetchAsset!(rel));
    root.name = `roads:${rel}`;
    for (const m of this.meshes(root)) {
      m.matrixAutoUpdate = false;
      m.updateMatrix();
      m.geometry.computeBoundingSphere();
      const old = m.material as THREE.MeshStandardMaterial;
      m.material = legacyRoadMaterial(old, Boolean(m.geometry.getAttribute('color')));
      old.dispose();
      m.renderOrder = 1;
      this.roadMeshes.push(m);
    }
    root.updateMatrixWorld(true);
    this.group.add(root);
  }

  private async loadBuildings(t: Tile, s: LayerSlot, kind: 'pipeline' | 'hd', level = 0): Promise<void> {
    if (s.root || s.loading || s.failed) return;
    s.loading = true;
    try {
      const root = await this.parse(await this.fetchAsset!(s.path));
      root.name = `buildings:${s.path}`;
      const base = kind === 'pipeline' || level === this.farHdLevel(t);
      for (const m of this.meshes(root)) {
        m.matrixAutoUpdate = false;
        m.updateMatrix();
        m.geometry.computeBoundingSphere();
        m.geometry.computeBoundingBox();
        const old = m.material as THREE.MeshStandardMaterial;
        const hasColor = Boolean(m.geometry.getAttribute('color'));
        const id = m.geometry.getAttribute('_building_id');
        // alias without the leading underscore for use in shaders (same buffer)
        if (id) m.geometry.setAttribute('buildingId', id);
        else m.geometry.setAttribute('buildingId', new THREE.BufferAttribute(new Float32Array(m.geometry.getAttribute('position').count), 1));
        const vcOnly = this.vertexColorOnly(m.geometry);
        if (/^hero/i.test(m.name) || /^hero/i.test(m.parent?.name ?? '') || vcOnly) {
          // Blender hero campuses / vertex-color parts carry their own palette:
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
        if (m.material !== old) old.dispose();
        m.castShadow = this.shadows;
        m.receiveShadow = this.shadows;
        m.userData.root = root;
        this.buildingMeshes.push(m);
        if (base) this.indexBuildingMesh(m);
      }
      root.updateMatrixWorld(true);
      s.root = root;
      root.visible = false;
      this.group.add(root);
      this.showBestBuildings(t);
    } catch (e) {
      s.failed = true;
      throw new Error(`${s.path}: ${(e as Error).message}`);
    } finally {
      s.loading = false;
    }
  }

  /** `_MAT` 7 (vertex color only) everywhere, or ground parts (`_MAT` 6) of hero meshes */
  private vertexColorOnly(g: THREE.BufferGeometry): boolean {
    const a = g.getAttribute('_mat');
    if (!a || a.count === 0) return false;
    for (let i = 0; i < a.count; i += Math.max(1, Math.floor(a.count / 64))) if (Math.round(a.getX(i)) < 6) return false;
    return true;
  }

  private disposeRoot(root: THREE.Object3D, list: THREE.Mesh[]): void {
    const set = new Set(this.meshes(root));
    for (let i = list.length - 1; i >= 0; i--) if (set.has(list[i]!)) list.splice(i, 1);
    for (const m of set) {
      m.geometry.dispose();
      const mat = m.material as THREE.Material;
      // shared atlas materials are kept
      if (mat !== this.atlasWall && mat !== this.atlasRoof && mat !== this.groundMat && mat !== this.markMat) {
        (mat as THREE.MeshStandardMaterial).map?.dispose();
        mat.dispose();
      }
      const map = m.userData.map as THREE.Texture | undefined;
      map?.dispose();
    }
    root.removeFromParent();
  }

  /** Show the wanted terrain LOD if loaded, else the closest loaded one (finer first). */
  private showBestTerrain(t: Tile, want = this.wantTerrainLod(t)): void {
    let best: TerrainLod | null = null;
    for (const l of t.lods) if (l.root && l.lod >= want && (!best || l.lod < best.lod)) best = l;
    if (!best) for (const l of t.lods) if (l.root && (!best || l.lod > best.lod)) best = l;
    for (const l of t.lods) if (l.root) l.root.visible = l === best;
    const prev = t.shownLod;
    t.shownLod = best ? best.lod : -1;
    for (const s of t.streets) if (s.root) s.root.visible = t.shownLod === 0;
    if (prev !== t.shownLod) this.onChange?.();
  }

  private wantTerrainLod(t: Tile): number {
    const d = t.dist;
    const finest = t.lods[0]?.lod ?? 0;
    const want = d < this.terrainLod0 * this.lodScale ? 0 : d < this.terrainLod1 * this.lodScale ? 1 : 2;
    return Math.max(want, finest);
  }

  private wantHdLevel(t: Tile): number {
    const far = this.farHdLevel(t);
    if (far < 0) return -1;
    if (t.dist < this.hdLod0 * this.lodScale) for (let i = 0; i <= far; i++) if (t.hd[i]) return i;
    return far;
  }

  private showBestBuildings(t: Tile): void {
    if (t.buildings?.root) t.buildings.root.visible = true;
    if (!t.hd.length) return;
    const want = this.wantHdLevel(t);
    let best: LayerSlot | null = null;
    for (let i = want; i < t.hd.length && !best; i++) if (t.hd[i]?.root) best = t.hd[i]!;
    if (!best) for (let i = want - 1; i >= 0 && !best; i--) if (t.hd[i]?.root) best = t.hd[i]!;
    for (const s of t.hd) if (s?.root) s.root.visible = s === best;
  }

  private enqueue(key: string, pri: number, run: () => Promise<void>): void {
    if (this.queued.has(key)) {
      const q = this.queue.find((x) => x.key === key);
      if (q) q.pri = Math.min(q.pri, pri);
      return;
    }
    this.queued.add(key);
    this.queue.push({ key, pri, run });
  }

  private pump(): void {
    const MAX = 3;
    this.queue.sort((a, b) => a.pri - b.pri);
    while (this.active < MAX && this.queue.length) {
      const job = this.queue.shift()!;
      this.active++;
      job
        .run()
        .catch((e: unknown) => console.warn('world streaming:', (e as Error).message))
        .finally(() => {
          this.active--;
          this.queued.delete(job.key);
          this.lastStream = 0;
          this.pump();
        });
    }
  }

  /** Pending streaming jobs (tests / screenshot scripts wait for 0). */
  get pending(): number {
    return this.queue.length + this.active;
  }

  /** Decide LODs from the camera position: load what is needed, evict what is far. */
  private stream(cam: THREE.Vector3): void {
    for (const t of this.tiles) {
      t.dist = t.box.distanceToPoint(cam);
      const want = this.wantTerrainLod(t);
      const wantL = t.lods.find((l) => l.lod === want) ?? null;
      if (wantL && !wantL.root && !wantL.failed) this.enqueue(`t:${t.id}:${want}`, t.dist, () => this.loadTerrainLod(t, wantL));
      if (want <= 1 && t.splatPaths.length >= 2 && !t.splat) this.enqueue(`s:${t.id}`, t.dist + 50, () => this.loadSplat(t));
      if (want === 0) for (const s of t.streets) if (!s.root && !s.failed) this.enqueue(`r:${s.path}`, t.dist + 100, () => this.loadStreets(t, s));
      const hw = this.wantHdLevel(t);
      if (hw >= 0) {
        const s = t.hd[hw]!;
        if (!s.root && !s.failed) this.enqueue(`b:${s.path}`, t.dist + 20, () => this.loadBuildings(t, s, 'hd', hw));
      }
      // evict fine levels far outside their range (memory)
      for (const l of t.lods) {
        if (l.root && l.lod === 0 && l.lod !== t.lods[t.lods.length - 1]!.lod && t.dist > this.terrainLod0 * this.lodScale * 2 + 600) {
          this.disposeRoot(l.root, this.terrainMeshes);
          l.root = null;
          l.meshes = [];
        }
      }
      if (t.dist > this.terrainLod0 * this.lodScale * 2 + 600) {
        for (const s of t.streets) {
          if (s.root) {
            this.disposeRoot(s.root, this.roadMeshes);
            s.root = null;
          }
        }
      }
      const far = this.farHdLevel(t);
      for (let i = 0; i < far; i++) {
        const s = t.hd[i];
        if (s?.root && t.dist > this.hdLod0 * this.lodScale * 2.5 + 400) {
          this.disposeRoot(s.root, this.buildingMeshes);
          s.root = null;
        }
      }
      this.showBestTerrain(t, want);
      this.showBestBuildings(t);
    }
    this.pump();
  }

  /** Per-tile LOD streaming + frustum culling (plus hiding far building tiles). */
  updateCulling(camera: THREE.PerspectiveCamera): void {
    const now = performance.now();
    if (now - this.lastStream > 250 || camera.position.distanceTo(this.camPos) > 150) {
      this.lastStream = now;
      this.camPos.copy(camera.position);
      this.stream(camera.position);
    }
    this.projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView);
    const camPos = camera.position;
    for (const t of this.tiles) {
      const inView = this.frustum.intersectsBox(t.box);
      // tile roots: frustum culling is per mesh too, but a tile-level test is cheap
      for (const l of t.lods) if (l.root) l.root.visible = inView && l.lod === t.shownLod;
      for (const s of t.streets) if (s.root) s.root.visible = inView && t.shownLod === 0;
      const d = t.box.distanceToPoint(camPos);
      const bVis = inView && d < this.buildingDrawDistance;
      if (t.buildings?.root) t.buildings.root.visible = bVis;
      if (t.hd.length) {
        if (bVis) this.showBestBuildings(t);
        else for (const s of t.hd) if (s?.root) s.root.visible = false;
      }
    }
  }

  private isDrawn(m: THREE.Mesh): boolean {
    let o: THREE.Object3D | null = m;
    while (o) {
      if (!o.visible) return false;
      if (o === this.group) return true;
      o = o.parent;
    }
    return true;
  }

  /** Terrain meshes of the LOD currently selected per tile (drawn or frustum-culled). */
  private activeTerrain(): THREE.Mesh[] {
    const out: THREE.Mesh[] = [];
    for (const t of this.tiles) for (const l of t.lods) if (l.lod === t.shownLod) out.push(...l.meshes);
    return out;
  }

  pickBuilding(raycaster: THREE.Raycaster, includeHidden = false): BuildingHit | null {
    const targets = includeHidden ? this.buildingMeshes.filter((m) => m.parent) : this.buildingMeshes.filter((m) => this.isDrawn(m));
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
    const hit = raycaster.intersectObjects(this.activeTerrain(), false)[0] ?? raycaster.intersectObjects(this.roadMeshes, false)[0];
    return hit ? hit.point.clone() : null;
  }

  /** Is the first hit a building (so a click should open the building panel)? */
  firstHitIsBuilding(raycaster: THREE.Raycaster): boolean {
    const b = raycaster.intersectObjects(
      this.buildingMeshes.filter((m) => this.isDrawn(m)),
      false,
    )[0];
    if (!b) return false;
    const g = raycaster.intersectObjects(this.activeTerrain(), false)[0];
    return !g || b.distance <= g.distance + 0.5;
  }

  /** Terrain height at (x,z): finest loaded surface; falls back to `fallback`. */
  heightAt(x: number, z: number, fallback = 0): number {
    return this.fastHeightAt(x, z) ?? fallback;
  }

  // ---- fast lookups (walk camera, photoreal picking)

  private hgrid: { minX: number; minZ: number; cell: number; nx: number; nz: number; data: Float32Array } | null = null;
  private ray = new THREE.Ray(new THREE.Vector3(), new THREE.Vector3(0, -1, 0));

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
    const top = new Float32Array(nx * nz).fill(-Infinity);
    for (const m of this.terrainMeshes) {
      const p = m.geometry.getAttribute('position');
      for (let i = 0; i < p.count; i++) {
        const ix = Math.round((p.getX(i) - b.min.x) / cell);
        const iz = Math.round((p.getZ(i) - b.min.z) / cell);
        const k = iz * nx + ix;
        const y = p.getY(i);
        // skirts hang below the surface: keep only vertices near the top of the cell
        if (y > top[k]! + 0.5) {
          top[k] = y;
          sum[k] = y;
          cnt[k] = 1;
        } else if (y > top[k]! - 0.5) {
          sum[k]! += y;
          cnt[k]!++;
        }
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

  private tileAt(x: number, z: number): Tile | null {
    for (const t of this.tiles) if (x >= t.box.min.x && x <= t.box.max.x && z >= t.box.min.z && z <= t.box.max.z) return t;
    return null;
  }

  /** Height from the finest loaded terrain of the tile (BVH ray), or null. */
  private meshHeight(x: number, z: number): number | null {
    const t = this.tileAt(x, z);
    if (!t) return null;
    for (const l of t.lods) {
      if (!l.root) continue;
      for (const m of l.meshes) {
        const g = m.geometry as THREE.BufferGeometry & { rdBvh?: MeshBVH };
        if (!g.rdBvh) {
          try {
            g.rdBvh = new MeshBVH(g);
          } catch {
            return null;
          }
        }
        this.ray.origin.set(x, (t.box.max.y || 1000) + 100, z);
        const hit = g.rdBvh.raycastFirst(this.ray, THREE.DoubleSide);
        if (hit) return hit.point.y;
      }
      return null;
    }
    return null;
  }

  /** Terrain height: finest loaded mesh, else the coarse height grid; null outside the terrain. */
  fastHeightAt(x: number, z: number): number | null {
    const mh = this.meshHeight(x, z);
    if (mh !== null) return mh;
    return this.gridHeight(x, z);
  }

  private gridHeight(x: number, z: number): number | null {
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

  // ---- building index (centroids from `_building_id` vertices of the base layer)

  private bAcc = new Map<number, { x: number; z: number; n: number; top: number }>();
  private bIndex: { cell: number; map: Map<number, Array<{ id: number; x: number; z: number; top: number }>> } | null = null;

  private indexBuildingMesh(m: THREE.Mesh): void {
    const id = m.geometry.getAttribute('_building_id');
    const p = m.geometry.getAttribute('position');
    if (!id) return;
    for (let i = 0; i < p.count; i++) {
      const b = Math.round(id.getX(i));
      if (b <= 0) continue;
      let a = this.bAcc.get(b);
      if (!a) this.bAcc.set(b, (a = { x: 0, z: 0, n: 0, top: -Infinity }));
      a.x += p.getX(i);
      a.z += p.getZ(i);
      a.n++;
      a.top = Math.max(a.top, p.getY(i));
    }
    this.bIndex = null;
  }

  private buildBuildingIndex(): void {
    const cell = 50;
    const map = new Map<number, Array<{ id: number; x: number; z: number; top: number }>>();
    for (const [id, a] of this.bAcc) {
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
    const a = this.bAcc.get(id);
    return a ? { x: a.x / a.n, z: a.z / a.n, top: a.top } : null;
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
    for (const t of this.tiles) t.splat?.forEach((s) => s.dispose());
    this.group.clear();
  }
}
