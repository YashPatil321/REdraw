/**
 * Renderer, camera, map controls, render loop (with optional before/after
 * scissor split), label renderer, stats and a fly-to helper with easing.
 */

import * as THREE from 'three';
import { MapControls } from 'three/examples/jsm/controls/MapControls.js';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { GpuTimer } from './gpuTimer';
import { PostFX } from './postfx';
import { QUALITY, type Quality, type QualitySettings } from './quality';
import type { SkySystem } from './sky';
import { WalkControls, type GroundFn, type WalkPose } from './walk';

export interface FlyToOptions {
  /** distance from target (m); default keeps the current distance */
  distance?: number;
  /** elevation angle of the camera above the target (deg) */
  pitchDeg?: number;
  /** compass direction the camera looks toward (deg, 0 = north); default keeps current */
  headingDeg?: number;
  /** explicit camera position (overrides distance/pitch/heading) */
  position?: THREE.Vector3;
  duration?: number;
}

export interface SplitHooks {
  /** called before the left (baseline) pass and the right (plan) pass */
  beforePass(side: 'left' | 'right'): void;
  after(): void;
}

export interface FrameStats {
  fps: number;
  calls: number;
  triangles: number;
  geometries: number;
  textures: number;
  /** smoothed GPU frame time (ms) from EXT_disjoint_timer_query, null if unsupported */
  gpuMs: number | null;
  /** smoothed CPU time spent in frame callbacks + render submission (ms) */
  cpuMs: number;
}

type FrameFn = (dt: number, now: number) => void;

/** Base-map crossfade (see baseLayer.ts): the controller shows one layer set per pass. */
export interface LayerFade {
  /** eased crossfade position: 0 = aerial only .. 1 = street only */
  mix: number;
  /** make the given layer set the visible one (called before each pass) */
  setLayer(layer: 'aerial' | 'street'): void;
}

/** Photo pass-through settings for the HDR pipeline (postfx.ts). */
export interface PhotoPass {
  on: boolean;
  fogColor: THREE.Color;
  fogDensity: number;
  /** ambient occlusion on (off when only photo tiles + overlays are drawn) */
  ao: boolean;
}

const fadeVert = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const fadeFrag = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tFrame;
uniform float uOpacity;
void main() { gl_FragColor = vec4(texture2D(tFrame, vUv).rgb, uOpacity); }`;

const WALK_FAR = 15000;

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly labelRenderer: CSS2DRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: MapControls;
  readonly stats: FrameStats = { fps: 0, calls: 0, triangles: 0, geometries: 0, textures: 0, gpuMs: null, cpuMs: 0 };
  readonly gpuTimer: GpuTimer;
  /** sky / sun (set by the scene controller once the region is known) */
  sky: SkySystem | null = null;
  /** bounds the camera target is kept inside */
  extent = { minX: -6000, maxX: 6000, minZ: -6000, maxZ: 6000 };
  /** ground height estimate under the target (for dynamic near plane) */
  groundY = 0;

  split: { enabled: boolean; pos: number; hooks: SplitHooks | null } = { enabled: false, pos: 0.5, hooks: null };
  /** street-level first-person camera (orbit controls are off while it is on) */
  readonly walk: WalkControls;
  /** ground height for the walk camera */
  groundFn: GroundFn = () => null;
  /**
   * Render straight to the screen, skipping post-processing. Used in photoreal
   * mode: photo tiles must not go through AO / bloom / grading; their
   * materials opt out of tone mapping while our overlays keep ACES.
   */
  directRender = false;
  /** set while the base map crossfades between Google tiles and our world */
  layerFade: LayerFade | null = null;
  private fadeTex: THREE.FramebufferTexture | null = null;
  private fadeQuad: FullScreenQuad | null = null;
  private fadeMat: THREE.ShaderMaterial | null = null;
  private orbitFar = 150000;

  private frameFns = new Set<FrameFn>();
  private raf = 0;
  private last = 0;
  private fpsAcc = { frames: 0, time: 0 };
  private flight: {
    t0: THREE.Vector3;
    t1: THREE.Vector3;
    /** spherical offsets camera - target: [log radius, azimuth, polar] */
    s0: [number, number, number];
    s1: [number, number, number];
    start: number;
    duration: number;
    resolve: () => void;
  } | null = null;
  private resizeObserver: ResizeObserver;
  private post: PostFX | null = null;
  quality: Quality = 'medium';
  qs: QualitySettings = QUALITY.medium;
  /** recent frame-rate samples (for automatic quality downgrade) */
  readonly fpsHistory: number[] = [];

  constructor(private container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', stencil: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.AgXToneMapping;
    this.renderer.toneMappingExposure = 1;
    this.renderer.info.autoReset = false;
    this.gpuTimer = new GpuTimer(this.renderer.getContext());
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = 'block';

    this.labelRenderer = new CSS2DRenderer();
    const le = this.labelRenderer.domElement;
    le.style.position = 'absolute';
    le.style.inset = '0';
    le.style.pointerEvents = 'none';
    le.className = 'label-layer';
    container.appendChild(le);

    this.camera = new THREE.PerspectiveCamera(50, 1, 2, 150000);
    this.camera.position.set(0, 9000, 9000);

    this.controls = new MapControls(this.camera, this.renderer.domElement);
    // smooth, inertial orbit: low damping keeps a little glide after release
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.rotateSpeed = 0.55;
    this.controls.zoomSpeed = 0.9;
    this.controls.panSpeed = 0.9;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 40;
    this.controls.maxDistance = 22000;
    this.controls.maxPolarAngle = THREE.MathUtils.degToRad(84);
    this.controls.zoomToCursor = true;
    this.controls.addEventListener('start', () => this.cancelFlight());

    this.walk = new WalkControls(this.camera, this.renderer.domElement);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  get canvas(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  resize(): void {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h);
    this.post?.setSize(w, h, this.renderer.getPixelRatio());
    this.labelRenderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Apply a quality preset: pixel ratio, shadows and the post-processing chain. */
  setQuality(q: Quality): void {
    this.quality = q;
    this.qs = QUALITY[q];
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, this.qs.pixelRatioCap));
    this.renderer.shadowMap.enabled = this.qs.shadows;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.shadowMap.needsUpdate = true;
    this.rebuildPost();
    this.resize();
    // materials must recompile for shadow map changes
    this.scene.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(m)) m.forEach((x) => (x.needsUpdate = true));
      else if (m) m.needsUpdate = true;
    });
  }

  setSky(sky: SkySystem | null): void {
    this.sky = sky;
    this.rebuildPost();
  }

  private rebuildPost(): void {
    this.post?.dispose();
    this.post = null;
    if (this.qs.post) {
      try {
        this.post = new PostFX(this.renderer, this.scene, this.camera, this.qs, this.sky);
        const w = Math.max(1, this.container.clientWidth);
        const h = Math.max(1, this.container.clientHeight);
        this.post.setSize(w, h, this.renderer.getPixelRatio());
      } catch (e) {
        console.warn('post-processing unavailable', e);
        this.post = null;
      }
    }
    if (this.sky) this.sky.postAtmosphere = !!this.post;
  }

  /** The HDR pipeline is active (aerial perspective, tone mapping in post). */
  get postActive(): boolean {
    return !!this.post && !this.directRender && !this.split.enabled;
  }

  setGhostLook(on: boolean): void {
    this.post?.setGhost(on);
  }

  /** Photo pass-through for the frame (set per pass while crossfading). */
  setPhotoPass(p: PhotoPass): void {
    if (!this.post) return;
    this.post.photo.on = p.on;
    this.post.photo.fogColor.copy(p.fogColor);
    this.post.photo.fogDensity = p.fogDensity;
    this.post.aoEnabled = p.ao;
  }

  onFrame(fn: FrameFn): () => void {
    this.frameFns.add(fn);
    return () => this.frameFns.delete(fn);
  }

  start(): void {
    const loop = (now: number): void => {
      this.raf = requestAnimationFrame(loop);
      this.frame(now);
    };
    this.last = performance.now();
    this.raf = requestAnimationFrame(loop);
  }

  /** Distance from the camera to its orbit target. */
  get distance(): number {
    return this.camera.position.distanceTo(this.controls.target);
  }

  private frame(now: number): void {
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;

    if (this.walk.enabled) {
      this.walk.update(dt, this.groundFn);
      // street level: tight depth range (near 0.25 m, far 15 km) so AO / depth tests stay precise
      if (Math.abs(this.camera.near - 0.25) > 1e-3 || this.camera.far !== WALK_FAR) {
        this.camera.near = 0.25;
        this.camera.far = WALK_FAR;
        this.camera.updateProjectionMatrix();
      }
    } else this.orbitFrame(now);

    const t0 = performance.now();
    for (const fn of this.frameFns) fn(dt, now);
    this.gpuTimer.begin();
    this.render(dt);
    this.gpuTimer.end();
    this.stats.cpuMs = this.stats.cpuMs * 0.9 + (performance.now() - t0) * 0.1;
    this.stats.gpuMs = this.gpuTimer.ms;
  }

  /** Render one frame now and return the canvas as a PNG blob (photo mode). */
  async capturePng(): Promise<Blob | null> {
    this.frame(performance.now());
    return await new Promise<Blob | null>((resolve) => {
      try {
        this.renderer.domElement.toBlob((b) => resolve(b), 'image/png');
      } catch {
        resolve(null);
      }
    });
  }

  private orbitFrame(now: number): void {
    if (this.path) this.pathFrame(now);
    else if (this.flight) {
      const f = this.flight;
      const u = Math.min(1, (now - f.start) / f.duration);
      const e = easeInOutCubic(u);
      this.controls.target.lerpVectors(f.t0, f.t1, e);
      // interpolate the camera offset in spherical coordinates (log radius):
      // a smooth arcing descent instead of a straight line
      const r = Math.exp(f.s0[0] + (f.s1[0] - f.s0[0]) * e);
      const az = f.s0[1] + (f.s1[1] - f.s0[1]) * e;
      const pol = f.s0[2] + (f.s1[2] - f.s0[2]) * easeInOutCubic(Math.min(1, u * 1.15));
      this.camera.position.set(
        this.controls.target.x + r * Math.sin(pol) * Math.sin(az),
        this.controls.target.y + r * Math.cos(pol),
        this.controls.target.z + r * Math.sin(pol) * Math.cos(az),
      );
      if (u >= 1) {
        this.flight = null;
        f.resolve();
      }
    }
    this.controls.update();
    // keep the target over the region
    const t = this.controls.target;
    const cx = THREE.MathUtils.clamp(t.x, this.extent.minX, this.extent.maxX);
    const cz = THREE.MathUtils.clamp(t.z, this.extent.minZ, this.extent.maxZ);
    if (cx !== t.x || cz !== t.z) {
      const dx = cx - t.x;
      const dz = cz - t.z;
      t.x = cx;
      t.z = cz;
      this.camera.position.x += dx;
      this.camera.position.z += dz;
    }
    // dynamic near plane for depth precision (roads sit 0.3 m above terrain)
    const h = Math.max(1, this.camera.position.y - this.groundY);
    const near = THREE.MathUtils.clamp(Math.min(h, this.distance) * 0.02, 1, 150);
    if (Math.abs(near - this.camera.near) > 0.25) {
      this.camera.near = near;
      this.camera.updateProjectionMatrix();
    }
  }

  private render(dt: number): void {
    this.renderer.info.reset();
    const fade = this.layerFade;
    if (fade && fade.mix > 0 && fade.mix < 1 && !this.split.enabled) this.renderCrossfade(dt, fade);
    else this.renderScene(dt);
    this.labelRenderer.render(this.scene, this.camera);

    const r = this.renderer;
    this.stats.calls = r.info.render.calls;
    this.stats.triangles = r.info.render.triangles;
    this.stats.geometries = r.info.memory.geometries;
    this.stats.textures = r.info.memory.textures;
    this.fpsAcc.frames++;
    this.fpsAcc.time += dt;
    if (this.fpsAcc.time >= 0.5) {
      this.stats.fps = this.fpsAcc.frames / this.fpsAcc.time;
      this.fpsHistory.push(this.stats.fps);
      if (this.fpsHistory.length > 20) this.fpsHistory.shift();
      this.fpsAcc.frames = 0;
      this.fpsAcc.time = 0;
    }
  }

  /**
   * Crossfade between the two base maps, live (both keep moving with the
   * camera): draw the aerial set, copy the frame, draw the street set, then
   * lay the copy over it with the remaining opacity. Twice the work, but
   * only for the second or so the fade lasts.
   */
  private renderCrossfade(dt: number, fade: LayerFade): void {
    const r = this.renderer;
    const size = r.getDrawingBufferSize(new THREE.Vector2());
    if (!this.fadeTex || this.fadeTex.image.width !== size.x || this.fadeTex.image.height !== size.y) {
      this.fadeTex?.dispose();
      this.fadeTex = new THREE.FramebufferTexture(size.x, size.y);
      // the drawing buffer has no alpha channel: an RGB copy target is always copy-compatible
      this.fadeTex.format = THREE.RGBFormat;
      this.fadeTex.internalFormat = 'RGB8';
    }
    if (!this.fadeQuad) {
      this.fadeMat = new THREE.ShaderMaterial({
        vertexShader: fadeVert,
        fragmentShader: fadeFrag,
        uniforms: { tFrame: { value: null }, uOpacity: { value: 1 } },
        transparent: true,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      });
      this.fadeQuad = new FullScreenQuad(this.fadeMat);
    }
    const autoShadow = r.shadowMap.autoUpdate;
    fade.setLayer('aerial');
    // the aerial set has (almost) no shadow casters: keep the street set's shadow maps
    r.shadowMap.autoUpdate = false;
    this.renderScene(dt);
    r.shadowMap.autoUpdate = autoShadow;
    r.setRenderTarget(null);
    r.copyFramebufferToTexture(this.fadeTex);
    fade.setLayer('street');
    this.renderScene(dt);
    const m = this.fadeMat!;
    m.uniforms['tFrame']!.value = this.fadeTex;
    m.uniforms['uOpacity']!.value = 1 - fade.mix;
    const autoClear = r.autoClear;
    r.autoClear = false;
    r.setRenderTarget(null);
    this.fadeQuad.render(r);
    r.autoClear = autoClear;
  }

  private renderScene(dt: number): void {
    const r = this.renderer;
    if (this.split.enabled && this.split.hooks) {
      const w = r.domElement.clientWidth;
      const hgt = r.domElement.clientHeight;
      const sx = Math.round(w * this.split.pos);
      r.setScissorTest(true);
      this.split.hooks.beforePass('left');
      r.setScissor(0, 0, sx, hgt);
      r.setViewport(0, 0, w, hgt);
      r.render(this.scene, this.camera);
      this.split.hooks.beforePass('right');
      r.setScissor(sx, 0, w - sx, hgt);
      r.render(this.scene, this.camera);
      r.setScissorTest(false);
      this.split.hooks.after();
    } else if (this.post && !this.directRender) {
      this.post.viewDistance = this.walk.enabled ? 4 : this.distance;
      this.post.render(dt);
    } else {
      r.render(this.scene, this.camera);
    }
  }

  /** Enter the street-level walk camera at a pose. */
  enterWalk(pose: WalkPose): void {
    this.cancelFlight();
    if (!this.walk.enabled) this.orbitFar = this.camera.far;
    this.controls.enabled = false;
    this.walk.enable(pose, this.groundFn);
    this.walk.apply();
  }

  /** Back to orbit: look at a point ahead of the walker from a little above. */
  exitWalk(): void {
    if (!this.walk.enabled) return;
    this.walk.disable();
    const fx = Math.sin(this.walk.heading);
    const fz = -Math.cos(this.walk.heading);
    const g = this.groundFn(this.walk.x + fx * 60, this.walk.z + fz * 60) ?? this.camera.position.y - this.walk.eye;
    this.controls.target.set(this.walk.x + fx * 60, g, this.walk.z + fz * 60);
    this.camera.position.set(this.walk.x - fx * 80, this.camera.position.y + 90, this.walk.z - fz * 80);
    this.camera.near = 2;
    this.camera.far = this.orbitFar;
    this.camera.updateProjectionMatrix();
    this.controls.enabled = true;
    this.controls.update();
  }

  get walking(): boolean {
    return this.walk.enabled;
  }

  private path: { keys: Array<{ t: THREE.Vector3; s: [number, number, number] }>; start: number; duration: number; resolve: () => void } | null = null;

  /**
   * Cinematic camera path through keyframes (target + distance / pitch /
   * heading), Catmull-Rom interpolated in target space and log-spherical
   * offset space with one global ease: a continuous move, no stops.
   */
  flyPath(keys: Array<{ target: THREE.Vector3; distance: number; pitchDeg: number; headingDeg: number }>, durationS: number): Promise<void> {
    this.cancelFlight();
    const ks = keys.map((k) => {
      const pitch = THREE.MathUtils.degToRad(k.pitchDeg);
      const heading = THREE.MathUtils.degToRad(k.headingDeg);
      // camera offset opposite the look direction (heading 0 = looking north / -z)
      const off = new THREE.Vector3(-Math.sin(heading) * Math.cos(pitch), Math.sin(pitch), Math.cos(heading) * Math.cos(pitch));
      return { t: k.target.clone(), s: [Math.log(k.distance), Math.atan2(off.x, off.z), Math.acos(THREE.MathUtils.clamp(off.y, -1, 1))] as [number, number, number] };
    });
    for (let i = 1; i < ks.length; i++) {
      while (ks[i]!.s[1] - ks[i - 1]!.s[1] > Math.PI) ks[i]!.s[1] -= 2 * Math.PI;
      while (ks[i]!.s[1] - ks[i - 1]!.s[1] < -Math.PI) ks[i]!.s[1] += 2 * Math.PI;
    }
    return new Promise((resolve) => {
      this.path = { keys: ks, start: performance.now(), duration: durationS * 1000, resolve };
      this.pathFrame(performance.now());
    });
  }

  get flyingPath(): boolean {
    return this.path !== null;
  }

  private pathFrame(now: number): void {
    const p = this.path;
    if (!p) return;
    const u = Math.min(1, (now - p.start) / p.duration);
    const e = 0.5 - 0.5 * Math.cos(Math.PI * u);
    const n = p.keys.length - 1;
    const x = e * n;
    const i = Math.min(n - 1, Math.floor(x));
    const f = x - i;
    const k = (j: number): (typeof p.keys)[number] => p.keys[Math.max(0, Math.min(n, j))]!;
    const cr = (a: number, b: number, c: number, d: number): number => 0.5 * (2 * b + (-a + c) * f + (2 * a - 5 * b + 4 * c - d) * f * f + (-a + 3 * b - 3 * c + d) * f * f * f);
    const k0 = k(i - 1);
    const k1 = k(i);
    const k2 = k(i + 1);
    const k3 = k(i + 2);
    const t = this.controls.target;
    t.set(cr(k0.t.x, k1.t.x, k2.t.x, k3.t.x), cr(k0.t.y, k1.t.y, k2.t.y, k3.t.y), cr(k0.t.z, k1.t.z, k2.t.z, k3.t.z));
    const r = Math.exp(cr(k0.s[0], k1.s[0], k2.s[0], k3.s[0]));
    const az = cr(k0.s[1], k1.s[1], k2.s[1], k3.s[1]);
    const pol = THREE.MathUtils.clamp(cr(k0.s[2], k1.s[2], k2.s[2], k3.s[2]), 0.05, Math.PI / 2 - 0.02);
    this.camera.position.set(t.x + r * Math.sin(pol) * Math.sin(az), t.y + r * Math.cos(pol), t.z + r * Math.sin(pol) * Math.cos(az));
    if (u >= 1) {
      this.path = null;
      p.resolve();
    }
  }

  cancelFlight(): void {
    if (this.path) {
      const pp = this.path;
      this.path = null;
      pp.resolve();
    }
    if (this.flight) {
      const f = this.flight;
      this.flight = null;
      f.resolve();
    }
  }

  /** Move the camera immediately (no animation): automated screenshots, deep links. */
  jumpTo(target: THREE.Vector3, opts: FlyToOptions = {}): void {
    this.cancelFlight();
    void this.flyTo(target, { ...opts, duration: 0 });
    const f = this.flight;
    if (f) {
      this.flight = null;
      const r = Math.exp(f.s1[0]);
      const [, az, pol] = f.s1;
      this.controls.target.copy(f.t1);
      this.camera.position.set(f.t1.x + r * Math.sin(pol) * Math.sin(az), f.t1.y + r * Math.cos(pol), f.t1.z + r * Math.sin(pol) * Math.cos(az));
      this.controls.update();
      f.resolve();
    }
  }

  /** Smoothly move the camera so it looks at `target`. */
  flyTo(target: THREE.Vector3, opts: FlyToOptions = {}): Promise<void> {
    this.cancelFlight();
    const t1 = target.clone();
    let p1: THREE.Vector3;
    if (opts.position) {
      p1 = opts.position.clone();
    } else {
      const cur = this.camera.position.clone().sub(this.controls.target);
      const dist = opts.distance ?? cur.length();
      const pitch = THREE.MathUtils.degToRad(opts.pitchDeg ?? THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(cur.y / Math.max(cur.length(), 1e-6), -1, 1))));
      // heading: direction the camera looks toward, 0 = north (-z)
      const curHeading = Math.atan2(-cur.x, cur.z); // camera sits opposite to look dir
      const heading = opts.headingDeg !== undefined ? THREE.MathUtils.degToRad(opts.headingDeg) : curHeading;
      // camera offset is opposite the look direction
      const lookX = Math.sin(heading);
      const lookZ = -Math.cos(heading);
      p1 = new THREE.Vector3(
        t1.x - lookX * dist * Math.cos(pitch),
        t1.y + dist * Math.sin(pitch),
        t1.z - lookZ * dist * Math.cos(pitch),
      );
    }
    const sph = (off: THREE.Vector3): [number, number, number] => {
      const len = Math.max(off.length(), 1e-3);
      return [Math.log(len), Math.atan2(off.x, off.z), Math.acos(THREE.MathUtils.clamp(off.y / len, -1, 1))];
    };
    const t0 = this.controls.target.clone();
    const s0 = sph(this.camera.position.clone().sub(t0));
    const s1 = sph(p1.clone().sub(t1));
    // shortest way around
    while (s1[1] - s0[1] > Math.PI) s1[1] -= 2 * Math.PI;
    while (s1[1] - s0[1] < -Math.PI) s1[1] += 2 * Math.PI;
    return new Promise((resolve) => {
      this.flight = {
        t0,
        t1,
        s0,
        s1,
        start: performance.now(),
        duration: (opts.duration ?? 1.6) * 1000,
        resolve,
      };
    });
  }

  /** Normalized device coords for a pointer event. */
  ndc(ev: { clientX: number; clientY: number }): THREE.Vector2 {
    const rect = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.post?.dispose();
    this.fadeTex?.dispose();
    this.fadeMat?.dispose();
    this.fadeQuad?.dispose();
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.walk.dispose();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
      const mat = m.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
      else mat?.dispose();
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelRenderer.domElement.remove();
  }
}
