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

  /** shared uniforms for terrain tinting (height range of the loaded world) */
  private terrainUniforms = { uMinH: { value: 0 }, uMaxH: { value: 500 } };
  private standardBuildings = true;

  /** Terrain: albedo (NAIP or procedural) times a slope / height tint; procedural palette if untextured. */
  private terrainMaterial(old: THREE.MeshStandardMaterial): THREE.MeshLambertMaterial {
    const mat = new THREE.MeshLambertMaterial({ map: old.map ?? null, color: 0xffffff });
    if (mat.map) {
      mat.map.anisotropy = 8;
      mat.map.colorSpace = THREE.SRGBColorSpace;
    }
    const uniforms = this.terrainUniforms;
    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWNrm;')
        .replace(
          '#include <worldpos_vertex>',
          '#include <worldpos_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvWNrm = normalize(mat3(modelMatrix) * objectNormal);',
        );
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWNrm;\nuniform float uMinH;\nuniform float uMaxH;')
        .replace(
          '#include <map_fragment>',
          `#include <map_fragment>
          float slope = 1.0 - clamp(normalize(vWNrm).y, 0.0, 1.0);
          float hN = clamp((vWPos.y - uMinH) / max(uMaxH - uMinH, 1.0), 0.0, 1.0);
          float n1 = fract(sin(dot(floor(vWPos.xz / 9.0), vec2(12.9898, 78.233))) * 43758.5453);
          #ifdef USE_MAP
            diffuseColor.rgb *= mix(vec3(1.0), vec3(0.8, 0.76, 0.72), smoothstep(0.2, 0.65, slope));
            diffuseColor.rgb *= 0.95 + 0.1 * hN;
          #else
            vec3 dry = vec3(0.60, 0.55, 0.38);
            vec3 scrub = vec3(0.34, 0.40, 0.25);
            vec3 rock = vec3(0.55, 0.50, 0.44);
            vec3 c = mix(dry, scrub, smoothstep(0.25, 0.75, hN) * 0.8 + n1 * 0.2);
            c = mix(c, rock, smoothstep(0.3, 0.7, slope));
            diffuseColor.rgb = c * (0.9 + 0.2 * n1);
          #endif`,
        );
    };
    mat.customProgramCacheKey = () => `terrain-${mat.map ? 1 : 0}`;
    return mat;
  }

  /** Buildings: vertex-color palette with per-building variation, SoCal roof tones, roughness jitter. */
  private buildingMaterial(old: THREE.Material | null, standard: boolean, hasColor: boolean): THREE.Material {
    const oldStd = old as THREE.MeshStandardMaterial | null;
    const params = { color: hasColor ? 0xffffff : (oldStd?.color ?? new THREE.Color(0xd8d0c4)), vertexColors: hasColor };
    const mat: THREE.MeshStandardMaterial | THREE.MeshLambertMaterial = standard
      ? new THREE.MeshStandardMaterial({ ...params, roughness: 0.82, metalness: 0.0 })
      : new THREE.MeshLambertMaterial(params);
    mat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nattribute float buildingId;\nvarying float vBid;\nvarying vec3 vBWN;')
        .replace('#include <beginnormal_vertex>', '#include <beginnormal_vertex>\nvBid = buildingId;\nvBWN = normalize(mat3(modelMatrix) * objectNormal);');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying float vBid;\nvarying vec3 vBWN;')
        .replace(
          '#include <color_fragment>',
          `#include <color_fragment>
          float bh = fract(sin(vBid * 12.9898 + 1.7) * 43758.5453);
          float bh2 = fract(sin(vBid * 78.233 + 4.1) * 24634.6345);
          diffuseColor.rgb *= 0.86 + 0.26 * bh;
          float ny = normalize(vBWN).y;
          if (ny > 0.35) {
            bool pitched = ny < 0.985;
            vec3 tile = mix(vec3(0.66, 0.34, 0.24), vec3(0.52, 0.28, 0.21), bh2);
            vec3 slate = mix(vec3(0.42, 0.42, 0.44), vec3(0.58, 0.56, 0.52), bh2);
            vec3 roof = pitched ? (bh < 0.7 ? tile : slate) : mix(vec3(0.82, 0.82, 0.80), vec3(0.62, 0.62, 0.64), bh2);
            diffuseColor.rgb = mix(diffuseColor.rgb, roof, pitched ? 0.72 : 0.4);
          } else {
            // subtle ground-contact darkening via vertical facades
            diffuseColor.rgb *= 0.92;
          }`,
        );
      if (standard) {
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <roughnessmap_fragment>',
          '#include <roughnessmap_fragment>\nroughnessFactor = clamp(roughnessFactor + (bh2 - 0.5) * 0.3 - (ny > 0.985 ? 0.12 : 0.0), 0.35, 1.0);',
        );
      }
    };
    mat.customProgramCacheKey = () => `building-${standard ? 's' : 'l'}-${hasColor ? 1 : 0}`;
    return mat;
  }

  /** Roads: darker asphalt (arterial vertex colors stay lighter), polygon offset over terrain. */
  private roadMaterial(old: THREE.MeshStandardMaterial, hasColor: boolean): THREE.MeshLambertMaterial {
    const mat = new THREE.MeshLambertMaterial({
      map: old.map ?? null,
      color: hasColor ? new THREE.Color(0.78, 0.78, 0.8) : new THREE.Color(0.2, 0.2, 0.22),
      vertexColors: hasColor,
      side: old.side,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -4,
    });
    return mat;
  }

  /** Switch building shading and shadow flags for a quality preset. */
  applyQuality(opts: { standardBuildings: boolean; shadows: boolean; buildingDrawDistance: number }): void {
    this.buildingDrawDistance = opts.buildingDrawDistance;
    if (opts.standardBuildings !== this.standardBuildings) {
      this.standardBuildings = opts.standardBuildings;
      for (const m of this.buildingMeshes) {
        const old = m.material as THREE.Material;
        m.material = this.buildingMaterial(null, this.standardBuildings, Boolean(m.geometry.getAttribute('color')));
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
            m.material = this.terrainMaterial(old);
            m.renderOrder = 0;
            this.terrainMeshes.push(m);
          } else if (job.kind === 'buildings') {
            const id = m.geometry.getAttribute('_building_id');
            // alias without the leading underscore for use in shaders (same buffer)
            if (id) m.geometry.setAttribute('buildingId', id);
            else m.geometry.setAttribute('buildingId', new THREE.BufferAttribute(new Float32Array(m.geometry.getAttribute('position').count), 1));
            m.material = this.buildingMaterial(old, this.standardBuildings, hasColor);
            this.buildingMeshes.push(m);
          } else {
            m.material = this.roadMaterial(old, hasColor);
            m.renderOrder = 1;
            this.roadMeshes.push(m);
          }
          if (m.material !== old) old.dispose();
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
