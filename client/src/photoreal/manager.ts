/**
 * Photoreal mode runtime: owns the tiles, keeps them loading for the camera,
 * calibrates the vertical offset near the view (median of node residuals),
 * drapes the traffic overlay onto the photo surface, and reports status and
 * attribution. Loaded lazily (dynamic import) so open-data users never
 * download 3d-tiles-renderer.
 */

import * as THREE from 'three';
import type { Origin } from '../geo';
import type { RoadNetwork } from '../traffic/network';
import { calibrateOffset, pickSpread } from './calibrate';
import { Drape } from './drape';
import type { ExtentXZ } from './frame';
import { PhotorealTiles, type Attribution, type TileSource } from './tiles';

export interface ManagerOptions {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  origin: Origin;
  extent: ExtentXZ;
  source: TileSource;
  onStatus: (msg: string) => void;
  onAttribution: (a: Attribution) => void;
  onError: (msg: string) => void;
  /** drape changed: re-upload the adjustment texture */
  onDrape: () => void;
}

export class PhotorealManager {
  readonly tiles: PhotorealTiles;
  private active = false;
  private net: RoadNetwork | null = null;
  private drape: Drape | null = null;
  private lastCal = { x: Infinity, z: Infinity, dist: Infinity, t: 0 };
  private calibrated = false;
  private lastAttr = '';
  private lastAttrT = 0;
  private lastStatusT = 0;
  private size = new THREE.Vector2();
  /** smoothed offset target */
  private offsetTarget: number;

  constructor(private o: ManagerOptions) {
    this.tiles = new PhotorealTiles({ origin: o.origin, extent: o.extent, source: o.source });
    this.offsetTarget = this.tiles.offset;
    this.tiles.attach(o.camera, o.renderer);
    this.tiles.onError = (m) => o.onError(m);
    this.tiles.onFirstLoad = () => o.onStatus('');
  }

  setNetwork(net: RoadNetwork | null): void {
    this.net = net;
    this.drape = net ? new Drape(net, (x, z) => this.tiles.heightAt(x, z, 900)) : null;
  }

  setActive(on: boolean): void {
    if (on === this.active) return;
    this.active = on;
    if (on) {
      this.o.scene.add(this.tiles.group);
      if (!this.tiles.loaded) this.o.onStatus('Loading photoreal 3D tiles…');
    } else {
      this.tiles.group.removeFromParent();
      this.drape?.reset();
      this.o.onDrape();
    }
  }

  get isActive(): boolean {
    return this.active;
  }

  get offset(): number {
    return this.tiles.offset;
  }

  heightAt(x: number, z: number): number | null {
    return this.tiles.heightAt(x, z, 1000);
  }

  pick(raycaster: THREE.Raycaster): THREE.Vector3 | null {
    return this.tiles.pick(raycaster);
  }

  /** Per frame while active. `target` is the orbit target or the walker position. */
  frame(now: number, target: THREE.Vector3, camDist: number): void {
    if (!this.active) return;
    const r = this.o.renderer;
    r.getSize(this.size);
    this.tiles.setResolution(this.o.camera, this.size.x * r.getPixelRatio(), this.size.y * r.getPixelRatio());
    this.tiles.update();

    // vertical calibration near the view
    const moved = Math.hypot(target.x - this.lastCal.x, target.z - this.lastCal.z);
    const closer = camDist < this.lastCal.dist * 0.35;
    if (this.net && this.tiles.loaded && (moved > 900 || closer || !this.calibrated) && now - this.lastCal.t > 1200) {
      this.calibrate(target, camDist, now);
    }
    // ease the offset to its target (no visible jumps)
    const off = this.tiles.offset;
    if (Math.abs(this.offsetTarget - off) > 0.01) {
      const next = off + (this.offsetTarget - off) * 0.15;
      this.tiles.offset = next;
      this.drape?.shiftAll(next - off);
      this.o.onDrape();
    }
    // drape the overlay near the camera (only once calibrated; tiles near the view loaded)
    if (this.drape && this.calibrated && camDist < 4000) {
      if (this.drape.step(target.x, target.z, Math.max(camDist, 30), 3)) this.o.onDrape();
    }
    if (now - this.lastAttrT > 1000) {
      this.lastAttrT = now;
      const a = this.tiles.attribution();
      const key = `${a.google}|${a.text}`;
      if (key !== this.lastAttr) {
        this.lastAttr = key;
        this.o.onAttribution(a);
      }
    }
    if (!this.tiles.loaded && now - this.lastStatusT > 1000) {
      this.lastStatusT = now;
      const st = (this.tiles.tiles as unknown as { stats?: { downloading?: number; parsing?: number } }).stats;
      const n = (st?.downloading ?? 0) + (st?.parsing ?? 0);
      this.o.onStatus(`Loading photoreal 3D tiles…${n ? ` (${n} in flight)` : ''}`);
    }
  }

  private calibrate(target: THREE.Vector3, camDist: number, now: number): void {
    const net = this.net!;
    const radius = THREE.MathUtils.clamp(camDist * 1.2, 300, 2500);
    const nodes = pickSpread(net.nodes, target.x, target.z, radius, 24, Math.max(40, radius / 6));
    const cur = this.tiles.offset;
    const samples = nodes.map((nd) => {
      const h = this.tiles.heightAt(nd.x, nd.z, 1000);
      return { ours: nd.y, tile: h === null ? null : h - cur };
    });
    const cal = calibrateOffset(samples);
    const prevDist = this.lastCal.dist;
    this.lastCal = { x: target.x, z: target.z, dist: camDist, t: now };
    if (!cal) return;
    this.offsetTarget = cal.offset;
    if (!this.calibrated) {
      // first calibration: jump straight there
      this.tiles.offset = cal.offset;
      this.drape?.reset();
      this.o.onDrape();
    }
    this.calibrated = true;
    if (camDist < prevDist * 0.5) this.drape?.invalidateCoarse(camDist);
    console.info(`[photoreal] vertical offset ${cal.offset.toFixed(2)} m from ${cal.n} nodes (MAD ${cal.mad.toFixed(2)} m)`);
  }

  dispose(): void {
    this.tiles.dispose();
  }
}
