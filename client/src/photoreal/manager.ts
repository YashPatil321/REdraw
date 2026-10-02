/**
 * Photoreal runtime (the aerial base map): owns the tiles, keeps them loading
 * for the camera while they may be shown, calibrates their vertical offset
 * near the view (median of road-node residuals) so the TILES move into our
 * NAVD88 frame, and reports attribution. Overlays (cars, ribbons, queue bars)
 * always stay at our own heights, identical over Google and over our world
 * (the photo drape in drape.ts is kept but not used). Loaded lazily (dynamic
 * import) so visitors without a key never download 3d-tiles-renderer.
 */

import * as THREE from 'three';
import type { Origin } from '../geo';
import type { RoadNetwork } from '../traffic/network';
import { calibrateOffset, pickSpread, plausibleOffset } from './calibrate';
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
}

export class PhotorealManager {
  readonly tiles: PhotorealTiles;
  private active = false;
  private net: RoadNetwork | null = null;
  private lastCal = { x: Infinity, z: Infinity, dist: Infinity, t: 0 };
  private calibrated = false;
  /** last calibration ran against refined tiles */
  private calRefined = false;
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
  }

  setActive(on: boolean): void {
    if (on === this.active) return;
    this.active = on;
    if (on) {
      this.o.scene.add(this.tiles.group);
      if (!this.tiles.loaded) this.o.onStatus('Loading 3D world…');
    } else {
      this.tiles.group.removeFromParent();
    }
  }

  /** Show or hide the tiles (they stay in the scene; hidden while our world is the base map). */
  setVisible(on: boolean): void {
    this.tiles.group.visible = on;
  }

  /** Tiles for the current view finished loading (safe to fade them in). */
  get viewReady(): boolean {
    const t = this.tiles.tiles as unknown as { loadProgress?: number };
    return this.tiles.loaded && this.tiles.visibleCount > 0 && (t.loadProgress ?? 1) >= 0.95;
  }

  /** Vertical offset calibrated at least once (tiles sit in our frame). */
  get isCalibrated(): boolean {
    return this.calibrated;
  }

  get isActive(): boolean {
    return this.active;
  }

  get offset(): number {
    return this.tiles.offset;
  }

  heightAt(x: number, z: number, fromY = 1000): number | null {
    return this.tiles.heightAt(x, z, fromY);
  }

  pick(raycaster: THREE.Raycaster): THREE.Vector3 | null {
    return this.tiles.pick(raycaster);
  }

  /**
   * Per frame while active. `target` is the orbit target or the walker
   * position. `stream` = false pauses tile loading (our world is the base map
   * and Google will not be needed soon).
   */
  frame(now: number, target: THREE.Vector3, camDist: number, stream = true): void {
    if (!this.active || !stream) return;
    const r = this.o.renderer;
    r.getSize(this.size);
    this.tiles.setResolution(this.o.camera, this.size.x * r.getPixelRatio(), this.size.y * r.getPixelRatio());
    this.tiles.update();

    // vertical calibration near the view
    const moved = Math.hypot(target.x - this.lastCal.x, target.z - this.lastCal.z);
    const closer = camDist < this.lastCal.dist * 0.35;
    // Coarse tile levels can sit tens of meters off the ground: the first calibration may
    // run early (plausibility-checked), later ones and a refinement wait for refined tiles.
    const progress = (this.tiles.tiles as unknown as { loadProgress?: number }).loadProgress ?? 1;
    const refined = progress >= 0.9;
    if (moved > 900 || closer) this.calRefined = false;
    const refine = refined && this.calibrated && !this.calRefined;
    if (this.net && this.tiles.loaded && (refined || !this.calibrated) && (moved > 900 || closer || !this.calibrated || refine) && now - this.lastCal.t > 1200) {
      this.calibrate(target, camDist, now);
      if (refined && this.calibrated) this.calRefined = true;
    }
    // ease the offset to its target (no visible jumps)
    const off = this.tiles.offset;
    if (Math.abs(this.offsetTarget - off) > 0.01) this.tiles.offset = off + (this.offsetTarget - off) * 0.15;
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
      this.o.onStatus('Loading 3D world…');
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
    this.lastCal = { x: target.x, z: target.z, dist: camDist, t: now };
    if (!cal || !plausibleOffset(cal)) {
      if (cal) console.info(`[photoreal] rejected vertical offset ${cal.offset.toFixed(2)} m (MAD ${cal.mad.toFixed(2)} m)`);
      return;
    }
    this.offsetTarget = cal.offset;
    // first calibration: jump straight there
    if (!this.calibrated) this.tiles.offset = cal.offset;
    this.calibrated = true;
    console.info(`[photoreal] vertical offset ${cal.offset.toFixed(2)} m from ${cal.n} nodes (MAD ${cal.mad.toFixed(2)} m)`);
  }

  dispose(): void {
    this.tiles.dispose();
  }
}
