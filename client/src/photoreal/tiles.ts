/**
 * Photorealistic 3D Tiles (Google Maps Platform Map Tiles API) rendered with
 * NASA-AMMOS `3d-tiles-renderer`, placed in our scene frame (see frame.ts):
 *   tiles.group.matrix = T(0, offset, 0) * (ECEF -> local ENU at the origin)
 *   tile vertex shader adds the small frame warp (UTM grid + curvature)
 * Materials are swapped for unlit MeshBasicMaterial with toneMapped=false, so
 * the photo textures show their true colors while our overlays keep the ACES
 * tone mapping. Tiles are never cached persistently (in-memory LRU only).
 *
 * The same class also loads any ECEF 3D Tiles tileset by URL (dev fixtures and
 * the headless test), via `source.kind = 'url'`.
 */

import * as THREE from 'three';
import { DRACOLoader, DRACO_GLTF_CONFIG } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { TilesRenderer } from '3d-tiles-renderer/three';
import { GoogleCloudAuthPlugin } from '3d-tiles-renderer/core/plugins';
import { GLTFExtensionsPlugin, TilesFadePlugin } from '3d-tiles-renderer/three/plugins';
import { MeshBVH, acceleratedRaycast } from 'three-mesh-bvh';
import type { Origin } from '../geo';
import { DEFAULT_TILE_OFFSET_M } from './calibrate';
import { WARP_GLSL, WARP_N, enuFrame, fitWarp, frameMatrixElements, unwarpPoint, warpPoint, type ExtentXZ, type RigidFrame, type Warp } from './frame';

export type TileSource = { kind: 'google'; key: string } | { kind: 'url'; url: string };

export interface PhotorealOptions {
  origin: Origin;
  extent: ExtentXZ;
  source: TileSource;
  /** custom transport (tests read fixtures from disk) */
  fetchData?: (url: string, init?: RequestInit) => Promise<Response>;
  /** no Draco decoder (tests in node) */
  noDraco?: boolean;
  /** fade tiles in/out (off in tests) */
  fade?: boolean;
}

export interface Attribution {
  /** Google logo required on screen when Google tiles are visible */
  google: boolean;
  /** data provider strings from the tiles (e.g. "Google, Airbus, ...") */
  text: string;
}

/** Shared uniforms for every tile material (warp coefficients, dimming tint). */
export interface TileUniforms {
  rdWarpX: { value: number[] };
  rdWarpY: { value: number[] };
  rdWarpZ: { value: number[] };
  rdTint: { value: THREE.Color };
}

/** Swap tile materials for unlit, warped, faithful-color materials. */
export function makeTileMaterial(old: THREE.Material, uniforms: TileUniforms): THREE.MeshBasicMaterial {
  const o = old as THREE.MeshStandardMaterial;
  const mat = new THREE.MeshBasicMaterial({
    map: o.map ?? null,
    color: o.map ? 0xffffff : (o.color?.clone() ?? new THREE.Color(0xcccccc)),
    vertexColors: o.vertexColors ?? false,
    side: o.side ?? THREE.FrontSide,
    transparent: o.transparent,
    opacity: o.opacity,
    toneMapped: false,
    fog: true,
  });
  if (mat.map) mat.map.colorSpace = THREE.SRGBColorSpace;
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${WARP_GLSL}`)
      .replace(
        '#include <project_vertex>',
        `vec4 rdW = modelMatrix * vec4(transformed, 1.0);
        rdW.xyz += rdWarp(rdW.xyz);
        vec4 mvPosition = viewMatrix * rdW;
        gl_Position = projectionMatrix * mvPosition;`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform vec3 rdTint;')
      .replace('#include <map_fragment>', '#include <map_fragment>\ndiffuseColor.rgb *= rdTint;');
  };
  mat.userData.redrawTile = true;
  return mat;
}

/** Plugin: runs before the fade plugin wraps materials (processTileModel precedes 'load-model'). */
class RedrawTilePlugin {
  readonly name = 'REDRAW_TILE_PLUGIN';
  constructor(
    private uniforms: TileUniforms,
    private fetcher?: (url: string, init?: RequestInit) => Promise<Response>,
  ) {}
  processTileModel(scene: THREE.Object3D): void {
    scene.traverse((c) => {
      const m = c as THREE.Mesh;
      if (!m.isMesh) return;
      const swap = (mat: THREE.Material): THREE.Material => {
        if (mat.userData.redrawTile) return mat;
        const n = makeTileMaterial(mat, this.uniforms);
        mat.dispose();
        return n;
      };
      m.material = Array.isArray(m.material) ? m.material.map(swap) : swap(m.material);
      m.castShadow = false;
      m.receiveShadow = false;
      m.raycast = bvhRaycast;
    });
  }
  // only defined when a custom transport is given (otherwise other plugins / fetch handle it)
  fetchData?: (url: string, init?: RequestInit) => Promise<Response>;
  init(): void {
    if (this.fetcher) this.fetchData = this.fetcher;
  }
}

let sharedDraco: DRACOLoader | null = null;

const _sphere = new THREE.Sphere();
/**
 * Tile raycast with a BVH built lazily the first time a ray reaches a tile's
 * bounding sphere (street-level tiles are dense; brute force costs ms per ray,
 * the BVH microseconds). Used for ground clamping, calibration, draping, picking.
 */
function bvhRaycast(this: THREE.Mesh, raycaster: THREE.Raycaster, intersects: THREE.Intersection[]): void {
  const geo = this.geometry as THREE.BufferGeometry & { boundsTree?: MeshBVH };
  if (!geo.boundingSphere) geo.computeBoundingSphere();
  _sphere.copy(geo.boundingSphere!).applyMatrix4(this.matrixWorld);
  if (!raycaster.ray.intersectsSphere(_sphere)) return;
  if (!geo.boundsTree) geo.boundsTree = new MeshBVH(geo, { targetLeafSize: 16 } as ConstructorParameters<typeof MeshBVH>[1]);
  acceleratedRaycast.call(this, raycaster, intersects);
}

export class PhotorealTiles {
  readonly tiles: TilesRenderer;
  readonly frame: RigidFrame;
  readonly warp: Warp;
  readonly uniforms: TileUniforms;
  readonly source: TileSource;
  /** meters added to tile heights (ellipsoid -> our NAVD88 elevations) */
  private _offset = DEFAULT_TILE_OFFSET_M;
  private frameMatrix = new THREE.Matrix4();
  private raycaster = new THREE.Raycaster();
  private errors: string[] = [];
  private hasGoogle: boolean;
  /** set when the server rejects the key or the tileset fails to load */
  lastError: string | null = null;
  onError: ((msg: string) => void) | null = null;
  private loadedOnce = false;
  onFirstLoad: (() => void) | null = null;

  constructor(opts: PhotorealOptions) {
    this.source = opts.source;
    this.frame = enuFrame(opts.origin.lat, opts.origin.lon);
    this.warp = fitWarp(this.frame, opts.origin, opts.extent);
    const pad = (a: number[]): number[] => Array.from({ length: WARP_N }, (_, i) => a[i] ?? 0);
    this.uniforms = {
      rdWarpX: { value: pad(this.warp.cx) },
      rdWarpY: { value: pad(this.warp.cy) },
      rdWarpZ: { value: pad(this.warp.cz) },
      rdTint: { value: new THREE.Color(1, 1, 1) },
    };
    this.frameMatrix.fromArray(frameMatrixElements(this.frame));

    this.hasGoogle = opts.source.kind === 'google';
    this.tiles = opts.source.kind === 'url' ? new TilesRenderer(opts.source.url) : new TilesRenderer();
    const tiles = this.tiles;
    // our plugin first: material swap must happen before the fade plugin wraps them
    tiles.registerPlugin(new RedrawTilePlugin(this.uniforms, opts.fetchData));
    if (opts.source.kind === 'google') {
      tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: opts.source.key, autoRefreshToken: true, useRecommendedSettings: true }));
    }
    if (!opts.noDraco) {
      if (!sharedDraco) {
        sharedDraco = new DRACOLoader();
        sharedDraco.setDecoderPath(DRACO_GLTF_CONFIG);
      }
      tiles.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader: sharedDraco, autoDispose: false }));
    }
    if (opts.fade !== false) tiles.registerPlugin(new TilesFadePlugin());
    // in-memory only; generous enough for street level, bounded for laptops
    tiles.lruCache.minSize = 600;
    tiles.lruCache.maxSize = 1400;
    tiles.lruCache.maxBytesSize = 0.45 * 1024 ** 3;
    tiles.errorTarget = opts.source.kind === 'google' ? 12 : 8;
    tiles.addEventListener('load-error', (e: { error?: unknown; url?: unknown }) => this.handleError(e));
    tiles.addEventListener('tiles-load-end', () => {
      if (!this.loadedOnce && tiles.visibleTiles.size > 0) {
        this.loadedOnce = true;
        this.onFirstLoad?.();
      }
    });

    const g = tiles.group;
    g.name = 'photoreal-tiles';
    g.matrixAutoUpdate = false;
    this.applyMatrix();
  }

  private handleError(e: { error?: unknown; url?: unknown }): void {
    const err = e.error as Error | undefined;
    const msg = err?.message ?? String(e.error ?? 'tile load failed');
    this.errors.push(msg);
    if (this.errors.length > 50) this.errors.shift();
    // only the root failing is fatal (bad key, quota, network)
    if (!this.tiles.root || /403|401|forbidden|API key|PERMISSION/i.test(msg)) {
      this.lastError = /403|forbidden|PERMISSION/i.test(msg)
        ? 'Google rejected the API key (403). Check that the Map Tiles API is enabled for this key and that the key allows this site.'
        : `Photoreal tiles failed to load: ${msg}`;
      this.onError?.(this.lastError);
    }
  }

  get group(): THREE.Group {
    return this.tiles.group;
  }

  get offset(): number {
    return this._offset;
  }

  set offset(v: number) {
    if (Math.abs(v - this._offset) < 1e-4) return;
    this._offset = v;
    this.applyMatrix();
  }

  private applyMatrix(): void {
    const g = this.tiles.group;
    g.matrix.makeTranslation(0, this._offset, 0).multiply(this.frameMatrix);
    g.matrixWorldNeedsUpdate = true;
    g.updateMatrixWorld(true);
  }

  /** Connect a camera (call once, and on resize call setResolution). */
  attach(camera: THREE.PerspectiveCamera, renderer: THREE.WebGLRenderer): void {
    this.tiles.setCamera(camera);
    this.tiles.setResolutionFromRenderer(camera, renderer);
  }

  setResolution(camera: THREE.Camera, w: number, h: number): void {
    this.tiles.setResolution(camera, w, h);
  }

  update(): void {
    this.tiles.group.updateMatrixWorld(true);
    this.tiles.update();
  }

  get loaded(): boolean {
    return this.loadedOnce;
  }

  get visibleCount(): number {
    return this.tiles.visibleTiles.size;
  }

  /** Night look for ghost traffic: dim the photo without touching tone mapping. */
  setDim(dim: number): void {
    const k = 1 - 0.78 * dim;
    this.uniforms.rdTint.value.setRGB(k * (1 - 0.15 * dim), k * (1 - 0.08 * dim), k);
  }

  /** Tangent-frame (raw, as three.js raycasts see it) -> scene point. */
  rawToScene(p: THREE.Vector3, out = new THREE.Vector3()): THREE.Vector3 {
    // the vertical offset is applied before the warp (warp fitted at offset 0; the
    // y-dependence of the warp is ~1e-4 per meter, so this is centimeter-exact)
    const w = warpPoint(this.warp, [p.x, p.y, p.z]);
    return out.set(w[0], w[1], w[2]);
  }

  sceneToRaw(p: THREE.Vector3, out = new THREE.Vector3()): THREE.Vector3 {
    const r = unwarpPoint(this.warp, [p.x, p.y, p.z]);
    return out.set(r[0], r[1], r[2]);
  }

  /** First tile hit (scene coordinates) for a ray given in scene coordinates. */
  raycastScene(origin: THREE.Vector3, dir: THREE.Vector3, far = 50000, expectDist?: number): { point: THREE.Vector3; distance: number } | null {
    // Map the ray into the raw tangent frame through two points: the scene
    // "vertical" is not parallel to the raw y axis away from the origin (earth
    // curvature, ~0.5 mrad at 3 km), so mapping the direction alone would miss
    // by a meter or more on long rays.
    const d = dir.clone().normalize();
    const L = Math.min(far, expectDist ?? 1500);
    const o = this.sceneToRaw(origin);
    const p1 = this.sceneToRaw(origin.clone().addScaledVector(d, L));
    this.raycaster.set(o, p1.sub(o).normalize());
    this.raycaster.far = far;
    (this.raycaster as THREE.Raycaster & { firstHitOnly?: boolean }).firstHitOnly = true;
    const hits = this.raycaster.intersectObject(this.tiles.group, true);
    const h = hits[0];
    if (!h) return null;
    return { point: this.rawToScene(h.point), distance: h.distance };
  }

  /** Raycast with a three.js raycaster set up in scene space (camera ray). */
  pick(raycaster: THREE.Raycaster): THREE.Vector3 | null {
    return this.raycastScene(raycaster.ray.origin, raycaster.ray.direction, raycaster.far)?.point ?? null;
  }

  /** Tile surface height (scene y, offset applied) at x, z, or null if nothing loaded there. */
  heightAt(x: number, z: number, fromY = 1200): number | null {
    // fromY stays inside the height range the warp was fitted for (-200..1200 m)
    const hit = this.raycastScene(new THREE.Vector3(x, fromY, z), new THREE.Vector3(0, -1, 0), fromY + 1500, Math.max(10, fromY - 150));
    return hit ? hit.point.y : null;
  }

  attribution(): Attribution {
    const list: Array<{ type: string; value: string }> = [];
    try {
      this.tiles.getAttributions(list);
    } catch {
      /* not ready */
    }
    const text = list
      .filter((a) => a.type === 'string')
      .map((a) => a.value)
      .filter(Boolean)
      .join('; ');
    return { google: this.hasGoogle && this.tiles.visibleTiles.size > 0, text };
  }

  dispose(): void {
    this.tiles.group.removeFromParent();
    this.tiles.dispose();
  }
}
