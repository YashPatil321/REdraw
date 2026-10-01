/**
 * Glue between the app store and the 3D scene: world loading, opening shot,
 * sim clock, traffic layers (baseline / plan, before-after split), picking,
 * and map-input placement for the plan builder.
 */

import * as THREE from 'three';
import { api } from '../api';
import { originFromLatLon, sceneToLatLon, type Origin } from '../geo';
import { store, toast, type AppState } from '../state';
import { advanceClock } from '../time';
import { TrafficLayer } from '../traffic/layer';
import type { RoadNetwork } from '../traffic/network';
import type { Playback } from '../traffic/playback';
import { buildRoadOverlayGeometry } from '../traffic/roadOverlay';
import type { School, WorldMeta } from '../types';
import { applyMapClick, setParam, type MapType } from '../ui/formgen';
import { ArterialLabels, BASELINE_COLOR, EdgeHighlight, LocationPin, PLAN_COLOR, PlanOverlay, SchoolMarkers } from './markers';
import { SkySystem } from './sky';
import { Viewer } from './viewer';
import { World } from './world';

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
    this.viewer.split.hooks = {
      beforePass: (side) => {
        if (this.baseline) this.baseline.group.visible = side === 'left';
        if (this.plan) this.plan.group.visible = side === 'right';
      },
      after: () => this.applyVisibility(store.get()),
    };
    this.viewer.start();
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
    this.clockT = store.get().simTime;

    this.schoolMarkers = new SchoolMarkers(schools, (s) => this.selectSchool(s), meta.hero?.school_id);
    this.viewer.scene.add(this.schoolMarkers.group);

    this.unsub.push(
      store.subscribe((s, prev) => this.onState(s, prev)),
    );
    this.applyVisibility(store.get());
  }

  async loadWorld(): Promise<void> {
    const meta = this.meta!;
    try {
      const manifest = await api.getManifest(meta);
      store.set({ worldStatus: 'Loading world…' });
      await this.world.load(manifest, (rel) => api.getAsset(meta, rel), (msg) => store.set({ worldStatus: msg }));
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
  }

  private groundHeight(x: number, z: number): number {
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
      layer = new TrafficLayer(this.net, pb, this.overlayGeom, store.get().schools, {
        accent: which === 'plan' ? PLAN_COLOR : BASELINE_COLOR,
        showLabels: true,
      });
      layer.setGhost(store.get().ghost);
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
    this.viewer.canvas.style.cursor = s.mapPick ? 'crosshair' : '';
  }

  private onState(s: AppState, prev: AppState): void {
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
    this.sky?.update(this.clockT, this.viewer.renderer);
    const dist = this.viewer.distance;
    const scale = THREE.MathUtils.clamp(dist / 650, 1, 9);
    const h = this.viewer.canvas.clientHeight;
    for (const l of [this.baseline, this.plan]) {
      if (!l || !this.trafficVisible(s)) continue;
      l.overlay.setViewport(this.viewer.camera, h);
      l.update(this.clockT, scale);
    }
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
    if (!this.world.firstHitIsBuilding(this.raycaster)) return;
    const hit = this.world.pickBuilding(this.raycaster);
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

  private placeMapInput(): void {
    const s = store.get();
    const pick = s.mapPick!;
    const ground = this.world.pickGround(this.raycaster);
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
