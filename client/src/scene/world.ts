/**
 * Static world: terrain, building and road tiles from the asset manifest
 * (GLTFLoader + DRACOLoader; works with draco true or false), per-tile
 * frustum culling, and raycast picking (buildings by `_building_id`, ground).
 */

import * as THREE from 'three';
import { DRACOLoader, DRACO_GLTF_CONFIG } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { Manifest } from '../types';

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

  /** Lambert materials are cheaper than Standard on integrated GPUs. */
  private toLambert(mesh: THREE.Mesh, opts: { vertexColors?: boolean } = {}): void {
    const old = mesh.material as THREE.MeshStandardMaterial;
    const mat = new THREE.MeshLambertMaterial({
      map: old.map ?? null,
      color: old.map ? 0xffffff : (old.color ?? new THREE.Color(0xcccccc)),
      vertexColors: opts.vertexColors ?? Boolean(mesh.geometry.getAttribute('color')),
      side: old.side,
    });
    if (mat.map) {
      mat.map.anisotropy = 4;
      mat.map.colorSpace = THREE.SRGBColorSpace;
    }
    old.dispose();
    mesh.material = mat;
  }

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
          if (job.kind === 'terrain') {
            this.toLambert(m, { vertexColors: false });
            if (!m.geometry.getAttribute('normal')) m.geometry.computeVertexNormals();
            m.renderOrder = 0;
            this.terrainMeshes.push(m);
          } else if (job.kind === 'buildings') {
            this.toLambert(m);
            this.buildingMeshes.push(m);
          } else {
            this.toLambert(m);
            const mat = m.material as THREE.Material;
            mat.polygonOffset = true;
            mat.polygonOffsetFactor = -2;
            mat.polygonOffsetUnits = -4;
            m.renderOrder = 1;
            this.roadMeshes.push(m);
          }
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
        t.buildings.visible = inView && d < 9000;
      }
    }
  }

  pickBuilding(raycaster: THREE.Raycaster): BuildingHit | null {
    const targets = this.buildingMeshes.filter((m) => m.parent?.visible !== false && m.visible);
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
