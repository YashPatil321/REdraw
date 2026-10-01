/**
 * Renderer, camera, map controls, render loop (with optional before/after
 * scissor split), label renderer, stats and a fly-to helper with easing.
 */

import * as THREE from 'three';
import { MapControls } from 'three/examples/jsm/controls/MapControls.js';
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

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
}

type FrameFn = (dt: number, now: number) => void;

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly labelRenderer: CSS2DRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: MapControls;
  readonly stats: FrameStats = { fps: 0, calls: 0, triangles: 0, geometries: 0, textures: 0 };
  /** bounds the camera target is kept inside */
  extent = { minX: -6000, maxX: 6000, minZ: -6000, maxZ: 6000 };
  /** ground height estimate under the target (for dynamic near plane) */
  groundY = 0;

  split: { enabled: boolean; pos: number; hooks: SplitHooks | null } = { enabled: false, pos: 0.5, hooks: null };

  private frameFns = new Set<FrameFn>();
  private raf = 0;
  private last = 0;
  private fpsAcc = { frames: 0, time: 0 };
  private flight: {
    p0: THREE.Vector3;
    p1: THREE.Vector3;
    t0: THREE.Vector3;
    t1: THREE.Vector3;
    start: number;
    duration: number;
    resolve: () => void;
  } | null = null;
  private resizeObserver: ResizeObserver;

  constructor(private container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.6;
    this.renderer.info.autoReset = false;
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
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 40;
    this.controls.maxDistance = 22000;
    this.controls.maxPolarAngle = THREE.MathUtils.degToRad(84);
    this.controls.zoomToCursor = true;
    this.controls.addEventListener('start', () => this.cancelFlight());

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
    this.labelRenderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
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

    if (this.flight) {
      const f = this.flight;
      const u = Math.min(1, (now - f.start) / f.duration);
      const e = easeInOutCubic(u);
      this.camera.position.lerpVectors(f.p0, f.p1, e);
      this.controls.target.lerpVectors(f.t0, f.t1, e);
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

    for (const fn of this.frameFns) fn(dt, now);

    this.renderer.info.reset();
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
    } else {
      r.render(this.scene, this.camera);
    }
    this.labelRenderer.render(this.scene, this.camera);

    this.stats.calls = r.info.render.calls;
    this.stats.triangles = r.info.render.triangles;
    this.stats.geometries = r.info.memory.geometries;
    this.stats.textures = r.info.memory.textures;
    this.fpsAcc.frames++;
    this.fpsAcc.time += dt;
    if (this.fpsAcc.time >= 0.5) {
      this.stats.fps = this.fpsAcc.frames / this.fpsAcc.time;
      this.fpsAcc.frames = 0;
      this.fpsAcc.time = 0;
    }
  }

  cancelFlight(): void {
    if (this.flight) {
      const f = this.flight;
      this.flight = null;
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
    return new Promise((resolve) => {
      this.flight = {
        p0: this.camera.position.clone(),
        p1,
        t0: this.controls.target.clone(),
        t1,
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
    this.resizeObserver.disconnect();
    this.controls.dispose();
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
