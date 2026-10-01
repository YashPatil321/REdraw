/**
 * Lawn grass near the camera at street level: the prop library's grass tuft
 * (crossed alpha quads with baked AO) scattered over ground whose landcover
 * splat says "lawn", world-anchored (a hash per 0.5 m cell, so tufts never
 * swim as the camera moves), kept off roads / sidewalks, swaying with the
 * shared wind. Rebuilt in 6 m cells as the camera walks.
 */

import * as THREE from 'three';
import { addWind } from './props';

export interface GrassSources {
  /** lawn weight 0..1 at (x, z), or null where unknown */
  lawnAt: (x: number, z: number) => number | null;
  /** ground height */
  heightAt: (x: number, z: number) => number | null;
  /** true where grass must not grow (roads, sidewalks, driveways) */
  blocked: (x: number, z: number) => boolean;
}

const CELL = 6;
const STEP = 0.42;

function hash(x: number, z: number, s: number): number {
  const h = Math.sin(x * 127.1 + z * 311.7 + s * 74.7) * 43758.5453;
  return h - Math.floor(h);
}

export class GrassField {
  readonly mesh: THREE.InstancedMesh;
  private cells = new Map<string, Float32Array>();
  private center = new THREE.Vector2(Infinity, Infinity);
  private radius = 26;
  private density = 1;
  private m = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private v = new THREE.Vector3();
  private s = new THREE.Vector3();
  private up = new THREE.Vector3(0, 1, 0);
  private dirty = true;

  constructor(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    private src: GrassSources,
    private cap = 24000,
  ) {
    addWind(material as THREE.MeshStandardMaterial, geometry);
    this.mesh = new THREE.InstancedMesh(geometry, material, cap);
    this.mesh.name = 'grass';
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    this.mesh.receiveShadow = true;
    this.mesh.castShadow = false;
  }

  /** 0 = off .. 1 = full density / radius (quality preset). */
  setDensity(d: number): void {
    if (d === this.density) return;
    this.density = d;
    this.radius = 14 + 16 * d;
    this.cells.clear();
    this.dirty = true;
    this.mesh.visible = d > 0;
  }

  /** Splat / heights changed (a tile streamed in): rebuild cells. */
  invalidate(): void {
    this.cells.clear();
    this.dirty = true;
  }

  private buildCell(cx: number, cz: number): Float32Array {
    const out: number[] = [];
    const x0 = cx * CELL;
    const z0 = cz * CELL;
    const keep = 0.35 + 0.65 * this.density;
    for (let ix = 0; ix < CELL / STEP; ix++) {
      for (let iz = 0; iz < CELL / STEP; iz++) {
        const gx = x0 + ix * STEP;
        const gz = z0 + iz * STEP;
        const h0 = hash(gx, gz, 1);
        if (h0 > keep) continue;
        const x = gx + hash(gx, gz, 2) * STEP;
        const z = gz + hash(gx, gz, 3) * STEP;
        const lawn = this.src.lawnAt(x, z);
        if (lawn === null || lawn < 0.45 + 0.3 * hash(gx, gz, 4)) continue;
        if (this.src.blocked(x, z)) continue;
        const y = this.src.heightAt(x, z);
        if (y === null) continue;
        out.push(x, y - 0.02, z, hash(gx, gz, 5) * Math.PI * 2, 0.9 + 0.9 * hash(gx, gz, 6) * lawn);
      }
    }
    return new Float32Array(out);
  }

  update(cam: THREE.Vector3, groundY: number): void {
    const alt = cam.y - groundY;
    if (this.density <= 0 || alt > 40) {
      this.mesh.count = 0;
      return;
    }
    if (!this.dirty && Math.hypot(cam.x - this.center.x, cam.z - this.center.y) < 3) return;
    this.dirty = false;
    this.center.set(cam.x, cam.z);
    const r = this.radius;
    const c0x = Math.floor((cam.x - r) / CELL);
    const c1x = Math.floor((cam.x + r) / CELL);
    const c0z = Math.floor((cam.z - r) / CELL);
    const c1z = Math.floor((cam.z + r) / CELL);
    const live = new Set<string>();
    let n = 0;
    let built = 0;
    for (let cx = c0x; cx <= c1x; cx++) {
      for (let cz = c0z; cz <= c1z; cz++) {
        const key = `${cx},${cz}`;
        live.add(key);
        let arr = this.cells.get(key);
        if (!arr) {
          // bound the work per frame; the rest fills in over the next frames
          if (built > 10) {
            this.dirty = true;
            continue;
          }
          arr = this.buildCell(cx, cz);
          this.cells.set(key, arr);
          built++;
        }
        for (let i = 0; i < arr.length && n < this.cap; i += 5) {
          const dx = arr[i]! - cam.x;
          const dz = arr[i + 2]! - cam.z;
          const d2 = dx * dx + dz * dz;
          if (d2 > r * r) continue;
          // thin out toward the edge of the radius
          const fade = 1 - Math.sqrt(d2) / r;
          const sc = arr[i + 4]! * Math.min(1, fade * 3.5);
          if (sc < 0.15) continue;
          this.q.setFromAxisAngle(this.up, arr[i + 3]!);
          this.v.set(arr[i]!, arr[i + 1]!, arr[i + 2]!);
          this.s.set(sc * 1.3, sc, sc * 1.3);
          this.m.compose(this.v, this.q, this.s);
          this.mesh.setMatrixAt(n++, this.m);
        }
      }
    }
    for (const k of this.cells.keys()) if (!live.has(k)) this.cells.delete(k);
    this.mesh.count = n;
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.mesh.dispose();
    this.mesh.removeFromParent();
  }
}
