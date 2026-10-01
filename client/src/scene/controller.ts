/**
 * Glue between the app store and the 3D scene: world loading, opening shot,
 * sim clock, traffic layers (baseline / plan, before-after split), picking,
 * and map-input placement for the plan builder.
 */

import * as THREE from 'three';
import { api } from '../api';
import { originFromLatLon, sceneToLatLon, type Origin } from '../geo';
import { store, toast, type AppState, type RenderMode } from '../state';
import { advanceClock } from '../time';
import { TrafficLayer, vehicleLightUniforms, type VehicleGeometries } from '../traffic/layer';
import type { RoadNetwork } from '../traffic/network';
import type { Playback } from '../traffic/playback';
import { adjTexture, buildRoadOverlayGeometry } from '../traffic/roadOverlay';
import { browserGoogleKey } from '../photoreal/key';
import type { PhotorealManager } from '../photoreal/manager';
import type { TileSource } from '../photoreal/tiles';
import type { School, WorldMeta } from '../types';
import { applyMapClick, setParam, type MapType } from '../ui/formgen';
import { ArterialLabels, BASELINE_COLOR, EdgeHighlight, LocationPin, PLAN_COLOR, PlanOverlay, SchoolMarkers } from './markers';
import { QUALITY, initialQuality, saveQuality, type Quality } from './quality';
import { SkySystem } from './sky';
import { Viewer } from './viewer';
import { World } from './world';

const SNAP_EDGE_M = 250;
const SNAP_NODE_M = 250;
const RENDER_MODE_KEY = 'redraw-render-mode';

export class SceneController {
  readonly viewer: Viewer;
  readonly world = new World();
  private sky: SkySystem | null = null;
  private origin: Origin | null = null;
  private meta: WorldMeta | null = null;
  private net: RoadNetwork | null = null;
  private overlayGeom: THREE.BufferGeometry | null = null;
  private baseline: TrafficLayer | null = null;
  private plan: TrafficLayer | null = null;
  private schoolMarkers: SchoolMarkers | null = null;
  private arterials: ArterialLabels | null = null;
  private planOverlay: PlanOverlay | null = null;
  private edgeHighlight = new EdgeHighlight(null);
  private pin = new LocationPin(0xffe08a);
  private residentPin = new LocationPin(PLAN_COLOR);
  private raycaster = new THREE.Raycaster();
  private clockT = 27000;
  private lastPush = 0;
  private down: { x: number; y: number; t: number } | null = null;
  private unsub: Array<() => void> = [];
  private worldLoaded = false;
  private qualityPinned = false;
  private lastQualityChange = 0;
  /** vehicle meshes from the props library, when available */
  vehicleGeoms: VehicleGeometries = {};
  /** photoreal tiles runtime (lazy: created the first time photoreal mode is on) */
  private photo: PhotorealManager | null = null;
  private photoLoading: Promise<void> | null = null;
  private tileSource: TileSource | null = null;

  constructor(container: HTMLElement) {
    this.viewer = new Viewer(container);
    this.viewer.scene.add(this.world.group, this.edgeHighlight.group, this.pin.group, this.residentPin.group);
    this.viewer.onFrame((dt, now) => this.frame(dt, now));
    const c = this.viewer.canvas;
    c.addEventListener('pointerdown', (e) => {
      if (e.button === 0) this.down = { x: e.clientX, y: e.clientY, t: performance.now() };
    });
    c.addEventListener('pointerup', (e) => {
      const d = this.down;
      this.down = null;
      if (!d || e.button !== 0) return;
      if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5 || performance.now() - d.t > 600) return;
      void this.click(e);
    });
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && store.get().mapPick) store.set({ mapPick: null });
    });
    this.viewer.groundFn = (x, z) => this.walkGround(x, z);
    this.viewer.walk.onExit = () => this.exitWalk();
    this.viewer.split.hooks = {
      beforePass: (side) => {
        if (this.baseline) this.baseline.group.visible = side === 'left';
        if (this.plan) this.plan.group.visible = side === 'right';
      },
      after: () => this.applyVisibility(store.get()),
    };
    const iq = initialQuality(this.viewer.renderer.getContext());
    this.qualityPinned = iq.pinned;
    store.set({ quality: iq.q });
    this.viewer.setQuality(iq.q);
    this.viewer.start();
  }

  private applyQuality(q: Quality): void {
    const qs = QUALITY[q];
    this.viewer.setQuality(q);
    this.sky?.configureShadows(qs.shadows, qs.shadowMapSize);
    this.world.applyQuality(qs);
    this.baseline?.setCastShadows(qs.carShadows && qs.shadows);
    this.plan?.setCastShadows(qs.carShadows && qs.shadows);
    this.lastQualityChange = performance.now();
  }

  /** User picked a quality preset in the top bar. */
  setQuality(q: Quality): void {
    this.qualityPinned = true;
    saveQuality(q);
    store.set({ quality: q });
  }

  /** Drop one quality step if the frame rate stays below 24 FPS (unless the user chose a preset). */
  private autoQuality(now: number): void {
    if (this.qualityPinned || !this.worldLoaded || now - this.lastQualityChange < 8000) return;
    const h = this.viewer.fpsHistory;
    if (h.length < 10) return;
    const avg = h.slice(-10).reduce((a, b) => a + b, 0) / 10;
    const q = store.get().quality;
    if (avg < 24 && q !== 'low') {
      const next: Quality = q === 'high' ? 'medium' : 'low';
      h.length = 0;
      store.set({ quality: next });
      toast(`Quality lowered to ${next} to keep the frame rate up.`, 'info', 4000);
    }
  }

  /** First-time setup once /world/meta and schools are known. */
  init(meta: WorldMeta, schools: School[]): void {
    this.meta = meta;
    this.origin = originFromLatLon(meta.region.origin.lat, meta.region.origin.lon);
    const ext = meta.region.extent_scene;
    this.viewer.extent = { minX: ext.min_x, maxX: ext.max_x, minZ: ext.min_z, maxZ: ext.max_z };
    const span = Math.max(ext.max_x - ext.min_x, ext.max_z - ext.min_z);
    this.viewer.controls.maxDistance = Math.max(3000, span * 1.8);
    this.viewer.camera.far = Math.max(60000, span * 12);
    this.viewer.camera.updateProjectionMatrix();
    // opening shot starts high above the region
    const cx = (ext.min_x + ext.max_x) / 2;
    const cz = (ext.min_z + ext.max_z) / 2;
    this.viewer.controls.target.set(cx, 0, cz);
    this.viewer.camera.position.set(cx, span * 1.15, cz + span * 0.9);
    this.sky = new SkySystem(this.viewer.scene, meta.region.origin.lat, meta.region.origin.lon, meta.region.timezone || 'America/Los_Angeles');
    const qs = QUALITY[store.get().quality];
    this.sky.configureShadows(qs.shadows, qs.shadowMapSize);
    this.clockT = store.get().simTime;

    this.initPhotorealSource();

    this.schoolMarkers = new SchoolMarkers(schools, (s) => this.selectSchool(s), meta.hero?.school_id);
    this.viewer.scene.add(this.schoolMarkers.group);

    this.unsub.push(
      store.subscribe((s, prev) => this.onState(s, prev)),
    );
    this.applyVisibility(store.get());
  }

  /** Photoreal source: `?tiles=<tileset.json>` (dev / fixtures) or a Google key. Default mode follows. */
  private initPhotorealSource(): void {
    const params = new URLSearchParams(location.search);
    const url = params.get('tiles');
    const key = browserGoogleKey();
    if (url) this.tileSource = { kind: 'url', url: new URL(url, location.href).toString() };
    else if (key.key) this.tileSource = { kind: 'google', key: key.key };
    let pref: string | null = params.get('mode');
    if (!pref) {
      try {
        pref = localStorage.getItem(RENDER_MODE_KEY);
      } catch {
        pref = null;
      }
    }
    const available = this.tileSource !== null;
    const mode: RenderMode = available && pref !== 'open' ? 'photoreal' : 'open';
    store.set({
      photoreal: { available, source: this.tileSource?.kind ?? null, status: '', error: null },
      renderMode: mode,
    });
    if (mode === 'photoreal') this.applyRenderMode(store.get());
  }

  /** Top-bar toggle. */
  setRenderMode(mode: RenderMode): void {
    if (mode === 'photoreal' && !this.tileSource) return;
    try {
      localStorage.setItem(RENDER_MODE_KEY, mode);
    } catch {
      /* ignore */
    }
    store.set({ renderMode: mode });
  }

  private get photoreal(): boolean {
    return store.get().renderMode === 'photoreal' && !!this.photo?.isActive;
  }

  private ensurePhotoreal(): Promise<void> {
    if (this.photo || !this.tileSource || !this.meta || !this.origin) return Promise.resolve();
    if (this.photoLoading) return this.photoLoading;
    store.set((s) => ({ photoreal: { ...s.photoreal, status: 'Loading photoreal 3D tiles…' } }));
    this.photoLoading = import('../photoreal/manager')
      .then(({ PhotorealManager }) => {
        const ext = this.meta!.region.extent_scene;
        this.photo = new PhotorealManager({
          scene: this.viewer.scene,
          camera: this.viewer.camera,
          renderer: this.viewer.renderer,
          origin: this.origin!,
          extent: ext,
          source: this.tileSource!,
          onStatus: (msg) => store.set((s) => ({ photoreal: { ...s.photoreal, status: msg } })),
          onAttribution: (a) => store.set({ attribution: a }),
          onError: (msg) => {
            store.set((s) => ({ photoreal: { ...s.photoreal, error: msg, status: '' }, renderMode: 'open' }));
            toast(`${msg} Showing open data instead.`, 'error', 12000);
          },
          onDrape: () => {
            if (this.net) adjTexture(this.net).needsUpdate = true;
          },
        });
        this.photo.setNetwork(this.net);
        this.applyRenderMode(store.get());
      })
      .catch((e: unknown) => {
        console.error(e);
        store.set((s) => ({ photoreal: { ...s.photoreal, error: (e as Error).message, status: '' }, renderMode: 'open' }));
      });
    return this.photoLoading;
  }

  /** Switch between our open-data meshes and photoreal tiles (overlays restyle to match). */
  private applyRenderMode(s: AppState): void {
    const want = s.renderMode === 'photoreal';
    if (want && !this.photo) {
      void this.ensurePhotoreal();
      return;
    }
    this.photo?.setActive(want);
    const pr = want && !!this.photo;
    // our meshes stay loaded (and raycastable for building picking) but are not drawn
    this.world.group.visible = !pr;
    this.viewer.directRender = pr;
    const qs = QUALITY[s.quality];
    this.viewer.renderer.shadowMap.enabled = qs.shadows && !pr;
    this.sky?.configureShadows(qs.shadows && !pr, qs.shadowMapSize);
    if (this.sky) this.sky.photoreal = pr;
    for (const l of [this.baseline, this.plan]) l?.setStyle(pr ? 'photoreal' : 'open');
    if (this.planOverlay) this.planOverlay.update(s.draft.tools, s.tools, s.selectedTool);
    if (!pr && store.get().attribution) store.set({ attribution: null });
    this.reseatOverlays();
  }

  /** Queue bars, pins: back onto the ground after a base-map change. */
  private reseatOverlays(): void {
    const fn = (x: number, z: number, fb: number): number => (this.photoreal ? (this.photo!.heightAt(x, z) ?? fb) : fb);
    for (const l of [this.baseline, this.plan]) l?.setGroundHeights(fn);
  }

  /** Ground for the walk camera: photo tiles, else our terrain grid, else the nearest road. */
  private walkGround(x: number, z: number): number | null {
    if (this.photoreal) {
      const h = this.photo!.heightAt(x, z);
      if (h !== null) return h;
    }
    const g = this.world.fastHeightAt(x, z);
    if (g !== null) return g;
    return this.net?.nearestEdge(x, z, 300)?.y ?? null;
  }

  async loadWorld(): Promise<void> {
    const meta = this.meta!;
    try {
      const manifest = await api.getManifest(meta);
      store.set({ worldStatus: 'Loading world…' });
      await this.world.load(manifest, (rel) => api.getAsset(meta, rel), (msg) => store.set({ worldStatus: msg }));
      this.world.applyQuality(QUALITY[store.get().quality]);
      store.set({ worldStatus: '' });
      const b = this.world.bounds;
      if (!b.isEmpty()) this.viewer.groundY = (b.min.y + b.max.y) / 2;
    } catch (e) {
      console.error(e);
      store.set({ worldStatus: `World assets unavailable: ${(e as Error).message}` });
      toast(`World assets unavailable: ${(e as Error).message}`, 'error', 9000);
    }
    this.worldLoaded = true;
    // re-place school pins on the terrain if their y is missing
    this.openingShot();
  }

  private openingShot(): void {
    const meta = this.meta!;
    const hero = meta.hero;
    if (!hero) return;
    const y = this.world.heightAt(hero.x, hero.z, 0);
    this.viewer.groundY = y;
    const target = new THREE.Vector3(hero.x, y, hero.z);
    void this.viewer.flyTo(target, { distance: 900, pitchDeg: 38, headingDeg: -20, duration: 4.0 });
  }

  setNetwork(net: RoadNetwork): void {
    this.net = net;
    this.overlayGeom?.dispose();
    this.overlayGeom = buildRoadOverlayGeometry(net);
    this.arterials = new ArterialLabels(net);
    this.viewer.scene.add(this.arterials.group);
    this.viewer.scene.remove(this.edgeHighlight.group);
    this.edgeHighlight.dispose();
    this.edgeHighlight = new EdgeHighlight(net);
    this.viewer.scene.add(this.edgeHighlight.group);
    this.planOverlay?.dispose();
    this.planOverlay = new PlanOverlay(net, this.origin!, (x, z) => this.groundHeight(x, z));
    this.viewer.scene.add(this.planOverlay.group);
    const s = store.get();
    if (s.baselinePlayback) this.setPlayback('baseline', s.baselinePlayback);
    if (s.planPlayback) this.setPlayback('plan', s.planPlayback);
    this.planOverlay.update(s.draft.tools, s.tools, s.selectedTool);
    this.photo?.setNetwork(net);
  }

  private groundHeight(x: number, z: number): number {
    if (this.photoreal) {
      const h = this.photo!.heightAt(x, z);
      if (h !== null) return h;
    }
    const nearest = this.net?.nearestEdge(x, z, 400);
    return this.world.heightAt(x, z, nearest?.y ?? 0);
  }

  private setPlayback(which: 'baseline' | 'plan', pb: Playback | null): void {
    const old = which === 'baseline' ? this.baseline : this.plan;
    old?.dispose();
    let layer: TrafficLayer | null = null;
    if (pb && this.net && this.overlayGeom) {
      if (pb.nEdges !== this.net.nEdges) {
        console.warn(`playback n_edges ${pb.nEdges} != network ${this.net.nEdges}`);
      }
      const qs = QUALITY[store.get().quality];
      layer = new TrafficLayer(this.net, pb, this.overlayGeom, store.get().schools, {
        accent: which === 'plan' ? PLAN_COLOR : BASELINE_COLOR,
        showLabels: true,
        vehicles: this.vehicleGeoms,
        castShadows: qs.shadows && qs.carShadows,
      });
      layer.setGhost(store.get().ghost);
      layer.setStyle(this.photoreal ? 'photoreal' : 'open');
      if (this.photoreal) layer.setGroundHeights((x, z, fb) => this.photo!.heightAt(x, z) ?? fb);
      this.viewer.scene.add(layer.group);
    }
    if (which === 'baseline') this.baseline = layer;
    else this.plan = layer;
    this.applyVisibility(store.get());
  }

  private trafficVisible(s: AppState): boolean {
    return s.view === 'traffic' || s.view === 'report';
  }

  private applyVisibility(s: AppState): void {
    const show = this.trafficVisible(s);
    const splitOn = s.view === 'report' && s.split && !!this.baseline && !!this.plan;
    this.viewer.split.enabled = splitOn;
    this.viewer.split.pos = s.splitPos;
    const primary = s.view === 'report' ? (this.plan ?? this.baseline) : s.trafficSource === 'plan' && this.plan ? this.plan : this.baseline;
    for (const l of [this.baseline, this.plan]) {
      if (!l) continue;
      l.group.visible = show && (splitOn || l === primary);
      l.setLabelsVisible(!splitOn);
    }
    if (this.planOverlay) this.planOverlay.group.visible = s.view === 'plan' || s.view === 'report';
    if (this.sky) this.sky.dim = show && s.ghost ? 1 : 0;
    this.viewer.setGhostLook(show && s.ghost);
    this.viewer.canvas.style.cursor = s.mapPick ? 'crosshair' : '';
  }

  private onState(s: AppState, prev: AppState): void {
    if (s.quality !== prev.quality) {
      this.applyQuality(s.quality);
      if (s.renderMode === 'photoreal') this.applyRenderMode(s);
    }
    if (s.renderMode !== prev.renderMode) this.applyRenderMode(s);
    if (s.baselinePlayback !== prev.baselinePlayback) this.setPlayback('baseline', s.baselinePlayback);
    if (s.planPlayback !== prev.planPlayback) this.setPlayback('plan', s.planPlayback);
    if (s.ghost !== prev.ghost) {
      this.baseline?.setGhost(s.ghost);
      this.plan?.setGhost(s.ghost);
    }
    if (s.draft !== prev.draft || s.selectedTool !== prev.selectedTool || s.tools !== prev.tools) {
      this.planOverlay?.update(s.draft.tools, s.tools, s.selectedTool);
    }
    if (s.highlightEdge !== prev.highlightEdge) {
      this.edgeHighlight.set(s.highlightEdge);
      this.baseline?.overlay.setHighlight(s.highlightEdge);
      this.plan?.overlay.setHighlight(s.highlightEdge);
    }
    if (s.building !== prev.building && !s.building) this.pin.hide();
    if (Math.abs(s.simTime - this.clockT) > 0.5 && s.simTime !== prev.simTime) this.clockT = s.simTime;
    if (
      s.view !== prev.view ||
      s.split !== prev.split ||
      s.splitPos !== prev.splitPos ||
      s.trafficSource !== prev.trafficSource ||
      s.ghost !== prev.ghost ||
      s.mapPick !== prev.mapPick
    ) {
      this.applyVisibility(s);
    }
  }

  private frame(dt: number, now: number): void {
    const s = store.get();
    const meta = this.meta;
    if (!meta) return;
    const start = meta.time.report_start_s;
    const end = meta.time.report_end_s;
    if (s.playing) {
      this.clockT = advanceClock(this.clockT, dt, s.speed, start, end);
      if (now - this.lastPush > 100) {
        this.lastPush = now;
        store.set({ simTime: this.clockT });
      }
    }
    this.world.updateCulling(this.viewer.camera);
    const dist = this.viewer.distance;
    if (this.sky) {
      this.sky.update(this.clockT, this.viewer.renderer);
      this.sky.followTarget(this.viewer.controls.target, dist);
      const dark = Math.max(this.sky.darkness, this.sky.dim);
      vehicleLightUniforms.uHead.value = 1.2 + 4.5 * dark;
      vehicleLightUniforms.uTail.value = 0.9 + 3.2 * dark;
    }
    if (this.photo) {
      const walking = this.viewer.walking;
      const target = walking ? this.viewer.camera.position : this.viewer.controls.target;
      this.photo.frame(now, target, walking ? 25 : dist);
      if (this.sky) this.photo.tiles.setDim(this.sky.dim);
    }
    this.autoQuality(now);
    const scale = THREE.MathUtils.clamp(dist / 650, 1, 9);
    const h = this.viewer.canvas.clientHeight;
    for (const l of [this.baseline, this.plan]) {
      if (!l || !this.trafficVisible(s)) continue;
      l.overlay.setViewport(this.viewer.camera, h);
      l.update(this.clockT, scale);
    }
    const walking = this.viewer.walking;
    if (walking || this.schoolMarkers?.isStreetLevel) this.schoolMarkers?.setStreetLevel(walking, this.viewer.camera.position);
    if (this.arterials) this.arterials.group.visible = !walking;
    this.edgeHighlight.tick(now);
    const pinScale = THREE.MathUtils.clamp(dist / 500, 1, 12);
    this.pin.tick(now, pinScale);
    this.residentPin.tick(now, pinScale);
  }

  // ---- picking

  private async click(e: PointerEvent): Promise<void> {
    this.raycaster.setFromCamera(this.viewer.ndc(e), this.viewer.camera);
    const s = store.get();
    if (s.mapPick) {
      this.placeMapInput();
      return;
    }
    if (s.view !== 'explore' && s.view !== 'traffic' && s.view !== 'plan') return;
    const hit = this.photoreal ? this.pickBuildingPhotoreal() : this.world.firstHitIsBuilding(this.raycaster) ? this.world.pickBuilding(this.raycaster) : null;
    if (!hit) return;
    this.pin.show(hit.point.x, hit.point.y, hit.point.z);
    store.set({ buildingLoading: true, school: null });
    try {
      const info = await api.getBuilding(hit.id);
      store.set({ building: info, buildingLoading: false });
    } catch (err) {
      store.set({ buildingLoading: false, building: { id: hit.id, error: (err as Error).message } });
    }
  }

  /**
   * Photoreal picking: the click lands on the photo mesh; our building meshes
   * are hidden but still raycastable. Accept our building if it is hit near the
   * photo hit, else the building whose footprint centroid is nearest.
   */
  private pickBuildingPhotoreal(): { id: number; point: THREE.Vector3 } | null {
    const p = this.photo!.pick(this.raycaster);
    const ours = this.world.pickBuilding(this.raycaster, true);
    if (!p) return ours;
    if (ours && ours.point.distanceTo(p) < 25) return { id: ours.id, point: p };
    const nb = this.world.nearestBuilding(p.x, p.z, 18);
    return nb ? { id: nb.id, point: p } : null;
  }

  private placeMapInput(): void {
    const s = store.get();
    const pick = s.mapPick!;
    const ground = this.photoreal ? (this.photo!.pick(this.raycaster) ?? this.world.pickGround(this.raycaster)) : this.world.pickGround(this.raycaster);
    if (!ground || !this.origin) {
      toast('Click on the terrain to place.', 'info', 2500);
      return;
    }
    let x = ground.x;
    let z = ground.z;
    let edge: number | undefined;
    let node: number | undefined;
    if (this.net) {
      if (pick.type === 'node') {
        const nd = this.net.nearestNode(x, z, SNAP_NODE_M);
        if (!nd) {
          toast('No intersection near there.', 'info', 2500);
          return;
        }
        node = nd.id;
        x = nd.x;
        z = nd.z;
      } else {
        const hit = this.net.nearestEdge(x, z, SNAP_EDGE_M);
        if (hit) {
          edge = hit.edge;
          x = hit.x;
          z = hit.z;
        } else if (pick.type === 'edge') {
          toast('No road near there.', 'info', 2500);
          return;
        }
      }
    }
    const ll = sceneToLatLon(x, z, this.origin);
    const inst = s.draft.tools[pick.toolIndex];
    if (!inst) return;
    const next = applyMapClick(pick.type as MapType, inst.params[pick.paramId] ?? null, { lat: ll.lat, lon: ll.lon, edge, node }, pick.maxCount);
    if (next === null) {
      toast(`Maximum of ${pick.maxCount} reached.`, 'info', 2500);
      store.set({ mapPick: null });
      return;
    }
    const tools = s.draft.tools.map((t, i) => (i === pick.toolIndex ? { ...t, params: setParam(t.params, pick.paramId, next) } : t));
    const single = pick.type === 'point' || pick.type === 'edge' || pick.type === 'node';
    store.set({ draft: { ...s.draft, tools }, mapPick: single ? null : pick });
  }

  // ---- camera helpers

  /**
   * Street-level view near (x, z): stand on the sidewalk of the nearest road,
   * facing the point (a building or school), or along the road if none given.
   */
  walkHere(x: number, z: number, faceTarget = true): void {
    let px = x;
    let pz = z;
    let heading = this.viewer.walking ? this.viewer.walk.heading : 0;
    const hit = this.net?.nearestEdge(x, z, 250);
    if (hit) {
      const e = this.net!.pointAt(hit.edge, hit.frac, { x: 0, y: 0, z: 0, dx: 1, dz: 0 });
      const lanes = Math.max(1, this.net!.edges[hit.edge]?.lanes ?? 1);
      // perpendicular toward the target, out to the sidewalk
      let nx = x - hit.x;
      let nz = z - hit.z;
      const nl = Math.hypot(nx, nz);
      if (nl > 1) {
        nx /= nl;
        nz /= nl;
      } else {
        nx = -e.dz;
        nz = e.dx;
      }
      const side = lanes * 3.4 + 2.5;
      px = hit.x + nx * side;
      pz = hit.z + nz * side;
      heading = faceTarget && nl > 1 ? Math.atan2(x - px, -(z - pz)) : Math.atan2(e.dx, -e.dz);
    }
    store.set({ walking: true });
    this.viewer.enterWalk({ x: px, z: pz, heading, pitch: faceTarget ? 0.08 : 0 });
  }

  /** Walk where the orbit camera is looking. */
  walkAtTarget(): void {
    const t = this.viewer.controls.target;
    const cam = this.viewer.camera.position;
    const heading = Math.atan2(t.x - cam.x, -(t.z - cam.z));
    const hit = this.net?.nearestEdge(t.x, t.z, 300);
    store.set({ walking: true });
    if (hit) {
      const e = this.net!.pointAt(hit.edge, hit.frac, { x: 0, y: 0, z: 0, dx: 1, dz: 0 });
      const lanes = Math.max(1, this.net!.edges[hit.edge]?.lanes ?? 1);
      const side = lanes * 3.4 + 2.5;
      this.viewer.enterWalk({ x: hit.x - e.dz * side, z: hit.z + e.dx * side, heading: Math.atan2(e.dx, -e.dz) });
    } else this.viewer.enterWalk({ x: t.x, z: t.z, heading });
  }

  exitWalk(): void {
    this.viewer.exitWalk();
    store.set({ walking: false });
  }

  /** Camera straight to a view (debug / screenshots): pitch and heading in degrees. */
  jumpToXZ(x: number, z: number, distance = 700, pitchDeg = 45, headingDeg?: number): void {
    if (this.viewer.walking) this.exitWalk();
    this.viewer.jumpTo(new THREE.Vector3(x, this.groundHeight(x, z), z), { distance, pitchDeg, headingDeg });
  }

  flyToXZ(x: number, z: number, distance = 700, duration = 1.6): void {
    const y = this.groundHeight(x, z);
    void this.viewer.flyTo(new THREE.Vector3(x, y, z), { distance, pitchDeg: 45, duration });
  }

  selectSchool(s: School): void {
    store.set({ school: s, building: null });
    this.pin.show(s.x, s.y ?? this.groundHeight(s.x, s.z), s.z);
    this.flyToXZ(s.x, s.z, 800);
  }

  showResident(x: number, z: number): void {
    this.residentPin.show(x, this.groundHeight(x, z), z);
    this.flyToXZ(x, z, 600);
  }

  showEdge(edge: number, x: number, z: number): void {
    store.set({ highlightEdge: edge });
    this.flyToXZ(x, z, 900);
  }

  get isWorldLoaded(): boolean {
    return this.worldLoaded;
  }

  dispose(): void {
    this.unsub.forEach((u) => u());
    this.baseline?.dispose();
    this.plan?.dispose();
    this.schoolMarkers?.dispose();
    this.arterials?.dispose();
    this.planOverlay?.dispose();
    this.edgeHighlight.dispose();
    this.pin.dispose();
    this.residentPin.dispose();
    this.overlayGeom?.dispose();
    this.sky?.dispose();
    this.world.dispose();
    this.viewer.dispose();
  }
}
