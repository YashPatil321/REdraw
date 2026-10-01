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
        .replace(
          '#include <common>',
          `#include <common>
          varying vec3 vWPos;
          varying vec3 vWNrm;
          uniform float uMinH;
          uniform float uMaxH;
          float tHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
          float tNoise(vec2 p) {
            vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
            return mix(mix(tHash(i), tHash(i + vec2(1, 0)), f.x), mix(tHash(i + vec2(0, 1)), tHash(i + vec2(1, 1)), f.x), f.y);
          }`,
        )
        .replace(
          '#include <map_fragment>',
          `#include <map_fragment>
          float slope = 1.0 - clamp(normalize(vWNrm).y, 0.0, 1.0);
          float hN = clamp((vWPos.y - uMinH) / max(uMaxH - uMinH, 1.0), 0.0, 1.0);
          float n1 = fract(sin(dot(floor(vWPos.xz / 9.0), vec2(12.9898, 78.233))) * 43758.5453);
          #ifdef USE_MAP
            diffuseColor.rgb *= mix(vec3(1.0), vec3(0.8, 0.76, 0.72), smoothstep(0.2, 0.65, slope));
            diffuseColor.rgb *= 0.95 + 0.1 * hN;
            // close-range detail: the imagery is ~1-10 m per pixel, so add fine
            // grain (grass blades, gravel, dry patches) that fades out with distance
            float camD = length(vWPos - cameraPosition);
            float detail = 1.0 - smoothstep(120.0, 700.0, camD);
            if (detail > 0.0) {
              vec2 q = vWPos.xz;
              float d1 = tNoise(q * 2.3);
              float d2 = tNoise(q * 0.61 + 17.0);
              detail *= clamp(1.0 - fwidth(q.x * 2.3) * 1.2, 0.0, 1.0);
              float lum = dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114));
              float greenish = smoothstep(0.0, 0.05, diffuseColor.g - max(diffuseColor.r, diffuseColor.b) + 0.02);
              vec3 grain = vec3(0.86 + 0.28 * d1) * (0.93 + 0.14 * d2);
              grain = mix(grain, grain * vec3(0.96, 1.04, 0.92), greenish);
              diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * grain, detail * (0.55 + 0.45 * (1.0 - lum)));
            }
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

  /**
   * Buildings (real footprints and heights): PBR stucco walls with per-building
   * tint, procedural windows by floor (spacing and size vary per building and
   * per type), darker ground course, Spanish barrel-tile or concrete-tile
   * pitched roofs and gravel / membrane flat roofs. Needs `aBase` / `aTop`
   * (per-building min / max y), computed at load.
   */
  private buildingMaterial(old: THREE.Material | null, standard: boolean, hasColor: boolean): THREE.Material {
    const oldStd = old as THREE.MeshStandardMaterial | null;
    const params = { color: hasColor ? 0xffffff : (oldStd?.color ?? new THREE.Color(0xd8d0c4)), vertexColors: hasColor };
    const mat: THREE.MeshStandardMaterial | THREE.MeshLambertMaterial = standard
      ? new THREE.MeshStandardMaterial({ ...params, roughness: 0.86, metalness: 0.0 })
      : new THREE.MeshLambertMaterial(params);
    mat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <common>',
          '#include <common>\nattribute float buildingId;\nattribute float aBase;\nattribute float aTop;\nvarying float vBid;\nvarying vec3 vBWN;\nvarying vec3 vBWP;\nvarying vec2 vBT;',
        )
        .replace(
          '#include <beginnormal_vertex>',
          '#include <beginnormal_vertex>\nvBid = buildingId;\nvBWN = normalize(mat3(modelMatrix) * objectNormal);\nvBT = vec2(aBase, aTop);',
        )
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvBWP = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>
          varying float vBid;
          varying vec3 vBWN;
          varying vec3 vBWP;
          varying vec2 vBT;
          float bHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
          float bNoise(vec2 p) {
            vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
            return mix(mix(bHash(i), bHash(i + vec2(1, 0)), f.x), mix(bHash(i + vec2(0, 1)), bHash(i + vec2(1, 1)), f.x), f.y);
          }
          float rdWin = 0.0;`,
        )
        .replace(
          '#include <color_fragment>',
          `#include <color_fragment>
          float bh = fract(sin(vBid * 12.9898 + 1.7) * 43758.5453);
          float bh2 = fract(sin(vBid * 78.233 + 4.1) * 24634.6345);
          float bh3 = fract(sin(vBid * 39.346 + 2.3) * 11743.123);
          vec3 n = normalize(vBWN);
          float ny = n.y;
          float hgt = max(vBT.y - vBT.x, 0.1);
          float above = vBWP.y - vBT.x;
          // stucco / painted plaster in warm SoCal tones, per-building variation
          vec3 base = diffuseColor.rgb * (0.88 + 0.22 * bh);
          base = mix(base, base * vec3(1.04, 1.0, 0.92), bh3 * 0.6);
          if (ny > 0.35) {
            bool pitched = ny < 0.985;
            vec2 t = normalize(vec2(-n.z, n.x) + 1e-5);
            float u = dot(vBWP.xz, t);
            float v = dot(vBWP.xz, normalize(n.xz + 1e-5));
            if (pitched) {
              // barrel tile (terracotta) or flat concrete tile (slate grey / brown)
              vec3 terra = mix(vec3(0.62, 0.30, 0.20), vec3(0.50, 0.25, 0.18), bh2);
              vec3 conc = mix(vec3(0.36, 0.35, 0.34), vec3(0.45, 0.38, 0.30), bh2);
              bool barrel = bh < 0.72;
              vec3 roof = barrel ? terra : conc;
              float period = barrel ? 0.26 : 0.33;
              // fade sub-pixel tile courses to their average (no moire from afar)
              float aaS = clamp(1.0 - fwidth(u / period) * 1.5, 0.0, 1.0);
              float aaR = clamp(1.0 - fwidth(v / 0.34) * 1.5, 0.0, 1.0);
              float stripes = mix(0.5, 0.5 + 0.5 * sin(u * 6.2831 / period), aaS);
              float rows = mix(0.92, smoothstep(0.0, 0.08, fract(v / 0.34)), aaR);
              roof *= (barrel ? 0.78 + 0.3 * stripes : 0.9 + 0.12 * stripes) * (0.86 + 0.14 * rows);
              roof *= 0.9 + 0.2 * bNoise(vBWP.xz * 0.9 + vBid);
              diffuseColor.rgb = roof;
            } else {
              // flat roofs: light membrane or gravel, with HVAC-ish blotches
              vec3 flatc = mix(vec3(0.78, 0.77, 0.74), vec3(0.55, 0.54, 0.52), bh2);
              flatc *= 0.9 + 0.12 * bNoise(vBWP.xz * 2.5) - 0.12 * step(0.93, bNoise(vBWP.xz * 0.25 + vBid));
              diffuseColor.rgb = flatc;
            }
          } else {
            vec2 t = normalize(vec2(-n.z, n.x) + 1e-5);
            float u = dot(vBWP.xz, t);
            // plaster grain and slight vertical staining
            base *= 0.94 + 0.08 * bNoise(vec2(u, vBWP.y) * 3.0);
            base *= 0.97 + 0.03 * smoothstep(0.0, hgt, above);
            // floors: houses ~2.9 m, commercial ~3.8 m
            float tall = step(9.0, hgt);
            float floorH = mix(2.9, 3.7, tall);
            float level = above / floorH;
            float cell = mix(3.6 + 2.4 * bh2, 3.0 + 1.2 * bh2, tall);
            vec2 g = vec2(fract(u / cell + bh), fract(level));
            float ww = mix(0.32 + 0.12 * bh3, 0.62, tall);
            float wx = step(abs(g.x - 0.5), ww * 0.5);
            float wy = step(0.32, g.y) * step(g.y, 0.82);
            // no windows in the top 0.6 m (eaves / parapet) or below 0.6 m
            float inside = step(0.6, above) * step(above, hgt - 0.6);
            // some cells blank (closets, garages) on houses
            float blank = (1.0 - tall) * step(0.62, bHash(floor(vec2(u / cell + bh, level)) + vBid));
            // antialias: when a window cell gets smaller than a few pixels, blend to the average facade
            float aaW = clamp(1.0 - max(fwidth(u / cell), fwidth(level)) * 3.0, 0.0, 1.0);
            rdWin = wx * wy * inside * (1.0 - blank) * aaW;
            // window frames slightly lighter than the wall
            float frame = step(abs(g.x - 0.5), ww * 0.5 + 0.06) * step(0.28, g.y) * step(g.y, 0.86) * inside * (1.0 - blank) * aaW;
            vec3 glass = mix(vec3(0.07, 0.09, 0.11), vec3(0.24, 0.3, 0.38), 0.5 + 0.5 * n.x * 0.4 + bh3 * 0.3);
            base = mix(base, vec3(0.92, 0.91, 0.88), (frame - rdWin) * 0.7);
            base = mix(base, glass, rdWin);
            // far away: average window coverage darkens the facade a little instead
            base = mix(base, base * (1.0 - 0.35 * ww * 0.5 * inside), 1.0 - aaW);
            // darker ground course
            base *= mix(0.72, 1.0, smoothstep(0.0, 0.9, above));
            diffuseColor.rgb = base;
          }`,
        );
      if (standard) {
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <roughnessmap_fragment>',
          '#include <roughnessmap_fragment>\nroughnessFactor = clamp(roughnessFactor + (bh2 - 0.5) * 0.2 - (ny > 0.985 ? 0.1 : 0.0) - rdWin * 0.75, 0.08, 1.0);',
        );
      }
    };
    mat.customProgramCacheKey = () => `building2-${standard ? 's' : 'l'}-${hasColor ? 1 : 0}`;
    return mat;
  }

  /** Per-vertex min / max y of each building (for floors, ground course, eaves). */
  private addBuildingExtents(g: THREE.BufferGeometry): void {
    const id = g.getAttribute('_building_id');
    const p = g.getAttribute('position');
    const n = p.count;
    const base = new Float32Array(n);
    const top = new Float32Array(n);
    if (!id) {
      g.computeBoundingBox();
      base.fill(g.boundingBox!.min.y);
      top.fill(g.boundingBox!.max.y);
    } else {
      const lo = new Map<number, number>();
      const hi = new Map<number, number>();
      for (let i = 0; i < n; i++) {
        const b = Math.round(id.getX(i));
        const y = p.getY(i);
        const a = lo.get(b);
        if (a === undefined || y < a) lo.set(b, y);
        const c = hi.get(b);
        if (c === undefined || y > c) hi.set(b, y);
      }
      for (let i = 0; i < n; i++) {
        const b = Math.round(id.getX(i));
        base[i] = lo.get(b)!;
        top[i] = hi.get(b)!;
      }
    }
    g.setAttribute('aBase', new THREE.BufferAttribute(base, 1));
    g.setAttribute('aTop', new THREE.BufferAttribute(top, 1));
  }

  /**
   * Roads: procedural asphalt (aggregate grain, patch repairs, oil darkening),
   * lightly tinted by class (arterials paler, older residential darker). The
   * pipeline's painted texture is dropped: markings and sidewalks are drawn
   * from the network by roadDetails.ts with the right lane counts.
   */
  private roadMaterial(old: THREE.MeshStandardMaterial, hasColor: boolean): THREE.MeshLambertMaterial {
    const mat = new THREE.MeshLambertMaterial({
      color: 0xffffff,
      vertexColors: hasColor,
      side: old.side,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -4,
    });
    mat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vRW;')
        .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvRW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>
          varying vec3 vRW;
          float rdHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
          float rdNoise(vec2 p) {
            vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
            return mix(mix(rdHash(i), rdHash(i + vec2(1, 0)), f.x), mix(rdHash(i + vec2(0, 1)), rdHash(i + vec2(1, 1)), f.x), f.y);
          }`,
        )
        .replace(
          '#include <color_fragment>',
          `#include <color_fragment>
          float cls = dot(diffuseColor.rgb, vec3(0.333));
          vec3 asphalt = vec3(0.155, 0.157, 0.165) * (0.78 + 0.5 * cls);
          float grain = rdNoise(vRW.xz * 3.1) * 0.6 + rdNoise(vRW.xz * 11.0) * 0.4;
          float patches = smoothstep(0.62, 0.7, rdNoise(vRW.xz * 0.045 + 3.7));
          float oil = smoothstep(0.55, 0.9, rdNoise(vRW.xz * 0.23 + 9.1));
          asphalt *= 0.88 + 0.22 * grain;
          asphalt = mix(asphalt, asphalt * 0.72, patches * 0.6);
          asphalt *= 1.0 - 0.12 * oil;
          diffuseColor.rgb = asphalt;`,
        );
    };
    mat.customProgramCacheKey = () => `road-asphalt-${hasColor ? 1 : 0}`;
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
            this.addBuildingExtents(m.geometry);
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
