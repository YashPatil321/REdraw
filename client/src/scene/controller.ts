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
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { TrafficLayer, setVehicleEnvMap, vehicleLightUniforms, vehicleMaterial, type VehicleGeometries } from '../traffic/layer';
import type { RoadNetwork } from '../traffic/network';
import type { Playback } from '../traffic/playback';
import { adjTexture, buildRoadOverlayGeometry } from '../traffic/roadOverlay';
import { browserGoogleKey } from '../photoreal/key';
import type { PhotorealManager } from '../photoreal/manager';
import type { TileSource } from '../photoreal/tiles';
import type { Manifest, School, WorldMeta } from '../types';
import { applyMapClick, setParam, type MapType } from '../ui/formgen';
import { ArterialLabels, BASELINE_COLOR, EdgeHighlight, LocationPin, PLAN_COLOR, PlanOverlay, SchoolMarkers } from './markers';
import { QUALITY, initialQuality, lowerQuality, saveQuality, type Quality } from './quality';
import { SkySystem } from './sky';
import { RoadDetails } from './roadDetails';
import { loadGrassTuft, loadStaticProps, loadVehicleProps, windUniforms, type StaticProps } from './props';
import { GrassField } from './grass';
import { Viewer } from './viewer';
import { World } from './world';
import { buildingUniforms } from './buildingMaterial';
import { MaterialLibrary } from './materials';

const SNAP_EDGE_M = 250;
const SNAP_NODE_M = 250;

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
  private roadDetails: RoadDetails | null = null;
  private staticProps: StaticProps | null = null;
  private grass: GrassField | null = null;

  private manifest: Manifest | null = null;
  /** PBR atlases (assets/materials), loaded with the world */
  materials: MaterialLibrary | null = null;
  private netResolve: () => void = () => undefined;
  private netReady = new Promise<void>((r) => (this.netResolve = r));

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

  /** Studio environment for car paint when the sky environment (IBL) is off (low quality). */
  private roomEnv: THREE.Texture | null = null;
  private vehicleEnv(ibl: boolean): void {
    if (ibl) {
      setVehicleEnvMap(null); // scene.environment: the real sky
      return;
    }
    try {
      if (!this.roomEnv) {
        const pmrem = new THREE.PMREMGenerator(this.viewer.renderer);
        this.roomEnv = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        pmrem.dispose();
      }
      setVehicleEnvMap(this.roomEnv);
    } catch (e) {
      console.warn('environment map unavailable', e);
    }
  }

  private applySkyQuality(q: Quality): void {
    const qs = QUALITY[q];
    if (!this.sky) return;
    this.sky.maxShadowFar = qs.shadowFar;
    this.sky.configureShadows(qs.shadows && !this.photoreal, qs.shadowMapSize, qs.cascades, qs.shadowFar);
    this.sky.ibl = qs.ibl;
    this.sky.setSkySize(qs.skySize);
    this.vehicleEnv(qs.ibl);
  }

  private applyQuality(q: Quality): void {
    const qs = QUALITY[q];
    this.viewer.setQuality(q);
    this.applySkyQuality(q);
    this.world.applyQuality({ ...qs, interiors: qs.interiors });
    this.staticProps?.setDrawDistance(qs.propDrawDistance);
    this.staticProps?.setShadows(qs.shadows);
    this.grass?.setDensity(this.photoreal ? 0 : qs.grass);
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
    if (this.qualityPinned || store.get().photoMode || !this.worldLoaded || now - this.lastQualityChange < 8000) return;
    const h = this.viewer.fpsHistory;
    if (h.length < 10) return;
    const avg = h.slice(-10).reduce((a, b) => a + b, 0) / 10;
    const q = store.get().quality;
    const next = lowerQuality(q);
    if (avg < 24 && next) {
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
    const qs = QUALITY[store.get().quality];
    this.sky = new SkySystem(
      this.viewer.scene,
      this.viewer.camera,
      this.viewer.renderer,
      meta.region.origin.lat,
      meta.region.origin.lon,
      meta.region.timezone || 'America/Los_Angeles',
      qs.skySize,
    );
    this.viewer.setSky(this.sky);
    this.applySkyQuality(store.get().quality);
    this.clockT = store.get().simTime;

    this.initPhotorealSource();

    this.schoolMarkers = new SchoolMarkers(schools, (s) => this.selectSchool(s), meta.hero?.school_id);
    this.viewer.scene.add(this.schoolMarkers.group);

    this.unsub.push(
      store.subscribe((s, prev) => this.onState(s, prev)),
    );
    this.applyVisibility(store.get());
  }

  /**
   * One world: Google Photorealistic 3D Tiles are the surface whenever a key
   * (or a dev `?tiles=<tileset.json>`) is available; otherwise our open-data
   * meshes. No user-facing mode switch. (`?mode=open` is a developer override.)
   */
  private initPhotorealSource(): void {
    const params = new URLSearchParams(location.search);
    const url = params.get('tiles');
    const key = browserGoogleKey();
    if (url) this.tileSource = { kind: 'url', url: new URL(url, location.href).toString() };
    else if (key.key) this.tileSource = { kind: 'google', key: key.key };
    const available = this.tileSource !== null;
    const mode: RenderMode = available && params.get('mode') !== 'open' ? 'photoreal' : 'open';
    store.set({
      photoreal: { available, source: this.tileSource?.kind ?? null, status: '', error: null },
      renderMode: mode,
    });
    if (mode === 'photoreal') this.applyRenderMode(store.get());
  }

  private get photoreal(): boolean {
    return store.get().renderMode === 'photoreal' && !!this.photo?.isActive;
  }

  private ensurePhotoreal(): Promise<void> {
    if (this.photo || !this.tileSource || !this.meta || !this.origin) return Promise.resolve();
    if (this.photoLoading) return this.photoLoading;
    store.set((s) => ({ photoreal: { ...s.photoreal, status: 'Streaming 3D world…' } }));
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
            // silently fall back to the open-data world (bad key, quota, network)
            console.warn(`[photoreal] ${msg} Falling back to the open-data world.`);
            store.set((s) => ({ photoreal: { ...s.photoreal, error: msg, status: '' }, renderMode: 'open' }));
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
    const qs0 = QUALITY[s.quality];
    // our meshes stay loaded (and raycastable for building picking) but are not drawn
    this.world.group.visible = !pr;
    if (this.roadDetails) this.roadDetails.group.visible = !pr;
    if (this.staticProps) this.staticProps.group.visible = !pr;
    this.grass?.setDensity(pr ? 0 : qs0.grass);
    this.viewer.directRender = pr;
    const qs = QUALITY[s.quality];
    this.viewer.renderer.shadowMap.enabled = qs.shadows && !pr;
    this.sky?.configureShadows(qs.shadows && !pr, qs.shadowMapSize, qs.cascades, qs.shadowFar);
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
  private walkGround(x: number, z: number, fromY?: number): number | null {
    const dem = this.world.fastHeightAt(x, z) ?? this.net?.nearestEdge(x, z, 300)?.y ?? null;
    if (this.photoreal) {
      // first step: start just above our DEM (tiles are calibrated to it), never on a canopy
      const from = fromY ?? (dem !== null ? dem + 4 : 1000);
      const h = this.photo!.heightAt(x, z, from);
      if (h !== null) return h;
    }
    return dem;
  }

  async loadWorld(): Promise<void> {
    const meta = this.meta!;
    try {
      const manifest = await api.getManifest(meta);
      this.manifest = manifest;
      void this.loadCredits(manifest.terrain_meta);
      store.set({ worldStatus: 'Loading world…' });
      const fetchAsset = this.materialFetcher(meta);
      const qs = QUALITY[store.get().quality];
      this.materials = await MaterialLibrary.load(fetchAsset, { half: store.get().quality === 'low', anisotropy: qs.anisotropy }).catch(() => null);
      this.world.materials = this.materials;
      this.world.terrainDetail = qs.terrainDetail;
      this.world.streetReady = this.netReady;
      this.world.streetDir = (x, z) => {
        const hit = this.net?.nearestEdge(x, z, 90, (e) => !/^(motorway|trunk|service)/.test(this.net!.edges[e]?.highway ?? ''));
        if (!hit) return null;
        const dx = hit.x - x;
        const dz = hit.z - z;
        const l = Math.hypot(dx, dz);
        return l > 0.5 ? { dx: dx / l, dz: dz / l } : null;
      };
      await this.world.load(manifest, (rel) => api.getAsset(meta, rel), (msg) => store.set({ worldStatus: msg }));
      this.world.applyQuality(QUALITY[store.get().quality]);
      store.set({ worldStatus: '' });
      const b = this.world.bounds;
      if (!b.isEmpty()) this.viewer.groundY = (b.min.y + b.max.y) / 2;
      const tb = new THREE.Box3();
      for (const m of this.world.terrainMeshes) if (m.geometry.boundingBox) tb.union(m.geometry.boundingBox);
      if (this.sky && !tb.isEmpty()) this.sky.fogBase = tb.min.y + (tb.max.y - tb.min.y) * 0.25;
    } catch (e) {
      console.error(e);
      store.set({ worldStatus: `World assets unavailable: ${(e as Error).message}` });
      toast(`World assets unavailable: ${(e as Error).message}`, 'error', 9000);
    }
    this.worldLoaded = true;
    this.buildRoadDetails();
    void this.loadProps();
    // re-place school pins on the terrain if their y is missing
    this.openingShot();
  }

  /** Asset fetcher for the material atlases (`?materials=dev` reads /dev-materials/ for local testing). */
  private materialFetcher(meta: WorldMeta): (rel: string) => Promise<ArrayBuffer> {
    const dev = new URLSearchParams(location.search).get('materials') === 'dev';
    if (!dev) return (rel) => api.getAsset(meta, rel);
    return async (rel) => {
      const r = await fetch(`/dev-${rel}`);
      if (!r.ok) throw new Error(`${r.status} ${rel}`);
      return await r.arrayBuffer();
    };
  }

  /** Attribution strings for the open-data base map, from the asset sources (never hardcoded). */
  private async loadCredits(terrainMetaRel: string | undefined): Promise<void> {
    const parts = ['© OpenStreetMap contributors (ODbL)'];
    if (terrainMetaRel) {
      try {
        const tm = JSON.parse(new TextDecoder().decode(await api.getAsset(this.meta!, terrainMetaRel))) as { sources?: Array<{ attribution?: string; name?: string }> };
        for (const src of tm.sources ?? []) {
          const a = src.attribution ?? src.name;
          if (a && !parts.includes(a)) parts.push(a);
        }
      } catch {
        /* optional */
      }
    }
    store.set({ openCredits: parts.join(' · ') });
  }

  /**
   * Cinematic opening: from high over the whole region, a continuous sweeping
   * descent that ends on the hero school (Del Norte). Any user input cancels
   * it; `?intro=0` (or reduced motion) goes straight to the final view.
   */
  private openingShot(): void {
    const meta = this.meta!;
    const hero = meta.hero;
    if (!hero) return;
    const y = this.world.heightAt(hero.x, hero.z, 0);
    this.viewer.groundY = y;
    const target = new THREE.Vector3(hero.x, y, hero.z);
    const final = { target, distance: 900, pitchDeg: 38, headingDeg: -20 };
    const params = new URLSearchParams(location.search);
    const reduced = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (params.get('intro') === '0' || reduced || location.hash.includes('plan=')) {
      void this.viewer.flyTo(target, { ...final, duration: 2.0 });
      return;
    }
    const ext = meta.region.extent_scene;
    const cx = (ext.min_x + ext.max_x) / 2;
    const cz = (ext.min_z + ext.max_z) / 2;
    const span = Math.max(ext.max_x - ext.min_x, ext.max_z - ext.min_z);
    const at = (x: number, z: number): THREE.Vector3 => new THREE.Vector3(x, this.world.heightAt(x, z, y), z);
    const mid = at((cx + hero.x) / 2, (cz + hero.z) / 2);
    const keys = [
      { target: at(cx, cz), distance: span * 0.95, pitchDeg: 58, headingDeg: 35 },
      { target: mid, distance: span * 0.42, pitchDeg: 36, headingDeg: 75 },
      { target: at(hero.x + 350, hero.z + 250), distance: 1700, pitchDeg: 26, headingDeg: 20 },
      final,
    ];
    this.viewer.jumpTo(keys[0]!.target, { distance: keys[0]!.distance, pitchDeg: keys[0]!.pitchDeg, headingDeg: keys[0]!.headingDeg });
    void this.viewer.flyPath(keys, 13);
  }

  setNetwork(net: RoadNetwork): void {
    this.net = net;
    this.netResolve();
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
    this.buildRoadDetails();
  }

  /** Prop library (blender/): vehicle models for traffic, trees and lamps for the open-data world. */
  private async loadProps(): Promise<void> {
    const meta = this.meta!;
    const fetchAsset = (rel: string): Promise<ArrayBuffer> => api.getAsset(meta, rel);
    try {
      const v = await loadVehicleProps(fetchAsset);
      if (v) {
        this.vehicleGeoms = v;
        const s = store.get();
        if (s.baselinePlayback) this.setPlayback('baseline', s.baselinePlayback);
        if (s.planPlayback) this.setPlayback('plan', s.planPlayback);
      }
    } catch (e) {
      console.warn('vehicle props unavailable', e);
    }
    try {
      const sp = await loadStaticProps(fetchAsset, (x, z) => this.world.fastHeightAt(x, z), vehicleMaterial, this.manifest?.grid ?? null);
      if (sp) {
        this.staticProps = sp;
        const qs = QUALITY[store.get().quality];
        sp.setDrawDistance(qs.propDrawDistance);
        sp.setShadows(qs.shadows);
        sp.group.visible = !this.photoreal;
        this.viewer.scene.add(sp.group);
      }
    } catch (e) {
      console.warn('static props unavailable', e);
    }    try {
      const tuft = await loadGrassTuft(fetchAsset);
      if (tuft) {
        this.grass = new GrassField(tuft.geometry, tuft.material, {
          lawnAt: (x, z) => this.world.lawnAt(x, z),
          heightAt: (x, z) => this.world.fastHeightAt(x, z),
          blocked: (x, z) => {
            const hit = this.net?.nearestEdge(x, z, 16);
            if (!hit) return false;
            const lanes = Math.max(1, this.net!.edges[hit.edge]?.lanes ?? 1);
            return hit.dist < lanes * 3.4 + 3.0;
          },
        });
        this.grass.setDensity(QUALITY[store.get().quality].grass);
        this.grass.mesh.visible = !this.photoreal && QUALITY[store.get().quality].grass > 0;
        this.world.onChange = () => this.grass?.invalidate();
        this.viewer.scene.add(this.grass.mesh);
      }
    } catch (e) {
      console.warn('grass unavailable', e);
    }
  }

  /** Lane markings + sidewalks (open-data mode) once both the terrain and the network are in. */
  private buildRoadDetails(): void {
    if (!this.net || !this.worldLoaded || this.roadDetails) return;
    // the HD pipeline ships real road surfaces, markings and sidewalks
    if (this.world.hasStreets) return;
    try {
      this.roadDetails = new RoadDetails(this.net, (x, z) => this.world.fastHeightAt(x, z));
      this.roadDetails.group.visible = !this.photoreal;
      this.viewer.scene.add(this.roadDetails.group);
    } catch (e) {
      console.warn('road details unavailable', e);
    }
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
    windUniforms.uWindTime.value = now / 1000;
    const dist = this.viewer.distance;
    if (this.sky) {
      this.sky.postAtmosphere = this.viewer.postActive;
      this.sky.followTarget(this.viewer.walking ? this.viewer.camera.position : this.viewer.controls.target, this.viewer.walking ? 260 : dist);
      this.viewer.camera.updateMatrixWorld();
      this.sky.update(this.clockT, this.viewer.renderer, this.viewer.camera);
      const dark = Math.max(this.sky.darkness, this.sky.dim);
      buildingUniforms.uNight.value = this.sky.darkness;
      // interiors: a fraction of the outdoor horizon radiance
      buildingUniforms.uInterior.value.copy(this.sky.horizon).multiplyScalar(0.55).lerp(new THREE.Color(0.02, 0.018, 0.015), this.sky.darkness * 0.7);
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
    if (this.roadDetails?.group.visible) this.roadDetails.update(this.viewer.camera.position);
    if (this.grass) {
      const cp = this.viewer.camera.position;
      this.grass.update(cp, this.world.fastHeightAt(cp.x, cp.z) ?? cp.y - 100);
    }
    if (this.staticProps?.group.visible) this.staticProps.update(this.viewer.camera.position, this.world.fastHeightAt(this.viewer.camera.position.x, this.viewer.camera.position.z) ?? 0);
    const scale = this.viewer.walking ? 1 : THREE.MathUtils.clamp(dist / 650, 1, 9);
    const colorMode = this.viewer.walking || dist < 320 ? 'paint' : 'speed';
    this.baseline?.setColorMode(colorMode);
    this.plan?.setColorMode(colorMode);
    const h = this.viewer.canvas.clientHeight;
    for (const l of [this.baseline, this.plan]) {
      if (!l || !this.trafficVisible(s)) continue;
      // queue labels are HTML (not occluded): hide them at street level and in split view
      l.setLabelsVisible(!this.viewer.walking && !this.viewer.split.enabled);
      l.overlay.setViewport(this.viewer.camera, h);
      l.overlay.setStreetLevel(this.viewer.walking ? 1 : 1 - THREE.MathUtils.smoothstep(dist, 120, 400));
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
    this.viewer.camera.updateMatrixWorld();
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
    // stand on a public street (not a campus driveway or freeway), like a pedestrian would
    const street = (e: number): boolean => /^(residential|tertiary|secondary|primary|unclassified|living_street)/.test(this.net!.edges[e]?.highway ?? '');
    const hit = this.net?.nearestEdge(x, z, 400, street) ?? this.net?.nearestEdge(x, z, 250);
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
      // at the curb edge of the street (a Street View vantage), clear of front yards
      const side = Math.max(1.6, lanes * 3.4 - 1.0);
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

  // ---- photo mode

  private photoPrevQuality: Quality | null = null;

  /** Hide the UI, switch to ultra quality and save a PNG once the frame has settled. */
  enterPhotoMode(): void {
    const s = store.get();
    if (s.photoMode) return;
    this.photoPrevQuality = s.quality;
    store.set({ photoMode: true, quality: 'ultra' });
    this.viewer.labelRenderer.domElement.style.display = 'none';
    // let shaders compile, shadows / AO / environment settle
    window.setTimeout(() => void this.savePhoto(), 1800);
  }

  exitPhotoMode(): void {
    if (!store.get().photoMode) return;
    this.viewer.labelRenderer.domElement.style.display = '';
    store.set({ photoMode: false, quality: this.photoPrevQuality ?? store.get().quality });
    this.photoPrevQuality = null;
  }

  /** Render a frame, stamp the required credits, download as PNG. */
  async savePhoto(): Promise<void> {
    const bar = document.querySelector('rd-app')?.shadowRoot?.querySelector('rd-photo-bar') as HTMLElement | null | undefined;
    if (bar) bar.style.visibility = 'hidden';
    try {
      const blob = await this.viewer.capturePng();
      if (!blob) throw new Error('canvas capture failed');
      const s = store.get();
      const credits = s.renderMode === 'photoreal' ? `Google${s.attribution?.text ? ' · ' + s.attribution.text : ''}` : s.openCredits;
      const out = await stampCredits(blob, credits);
      const a = document.createElement('a');
      const t = new Date();
      const pad = (n: number): string => String(n).padStart(2, '0');
      a.download = `redraw-${t.getFullYear()}${pad(t.getMonth() + 1)}${pad(t.getDate())}-${pad(t.getHours())}${pad(t.getMinutes())}${pad(t.getSeconds())}.png`;
      a.href = URL.createObjectURL(out);
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      toast('Screenshot saved.', 'info', 2500);
    } catch (e) {
      toast(`Screenshot failed: ${(e as Error).message}`, 'error', 5000);
    } finally {
      if (bar) bar.style.visibility = '';
    }
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
    this.roadDetails?.dispose();
    this.staticProps?.dispose();
    this.grass?.dispose();
    this.sky?.dispose();
    this.world.dispose();
    this.viewer.dispose();
  }
}

/** Draw the data credits into the bottom-right corner of a PNG (required attribution travels with the image). */
async function stampCredits(png: Blob, text: string): Promise<Blob> {
  if (!text) return png;
  try {
    const img = await createImageBitmap(png);
    const cv = document.createElement('canvas');
    cv.width = img.width;
    cv.height = img.height;
    const ctx = cv.getContext('2d');
    if (!ctx) return png;
    ctx.drawImage(img, 0, 0);
    const fs = Math.max(11, Math.round(img.height / 70));
    ctx.font = `${fs}px system-ui, sans-serif`;
    const w = ctx.measureText(text).width;
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(img.width - w - fs * 1.2, img.height - fs * 1.8, w + fs * 1.2, fs * 1.8);
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.fillText(text, img.width - w - fs * 0.6, img.height - fs * 0.55);
    return await new Promise<Blob>((resolve) => cv.toBlob((b) => resolve(b ?? png), 'image/png'));
  } catch {
    return png;
  }
}
