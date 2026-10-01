/** Sky (Preetham Sky shader), sun light and morning haze driven by the sim clock. */

import * as THREE from 'three';
import { Sky } from 'three/examples/jsm/objects/Sky.js';
import { localSecondsToDate, solarPosition, sunDirection } from './solar';

export class SkySystem {
  readonly sky = new Sky();
  readonly sun = new THREE.DirectionalLight(0xffffff, 2.2);
  readonly hemi = new THREE.HemisphereLight(0xbfd6ff, 0x5a4a3a, 1.1);
  readonly fog = new THREE.FogExp2(0xc9c3b8, 0.00006);
  private lastKey = '';
  private day = new Date();
  /** 0..1 dimming for the ghost-traffic night look */
  dim = 0;

  constructor(
    private scene: THREE.Scene,
    private lat: number,
    private lon: number,
    private timeZone: string,
  ) {
    this.sky.scale.setScalar(100000);
    const u = this.sky.material.uniforms;
    u['turbidity']!.value = 7;
    u['rayleigh']!.value = 1.6;
    u['mieCoefficient']!.value = 0.006;
    u['mieDirectionalG']!.value = 0.85;
    if (u['cloudCoverage']) u['cloudCoverage'].value = 0.25;
    scene.add(this.sky);
    scene.add(this.sun);
    scene.add(this.sun.target);
    scene.add(this.hemi);
    scene.fog = this.fog;
  }

  /** Update for sim time (seconds since local midnight) and renderer exposure. */
  update(simSeconds: number, renderer: THREE.WebGLRenderer): void {
    const key = `${Math.round(simSeconds / 30)}:${this.dim.toFixed(2)}`;
    if (key === this.lastKey) return;
    this.lastKey = key;
    const date = localSecondsToDate(simSeconds, this.day, this.timeZone);
    const pos = solarPosition(date, this.lat, this.lon);
    const [x, y, z] = sunDirection(pos);
    const u = this.sky.material.uniforms;
    (u['sunPosition']!.value as THREE.Vector3).set(x, y, z);
    this.sun.position.set(x * 5000, Math.max(y, 0.05) * 5000, z * 5000);
    this.sun.target.position.set(0, 0, 0);

    const el = pos.elevation;
    const day = THREE.MathUtils.smoothstep(el, -4, 12); // 0 before dawn, 1 in daylight
    const warm = 1 - THREE.MathUtils.smoothstep(el, 2, 25); // golden hour tint
    this.sun.intensity = (0.35 + 2.0 * day) * (1 - 0.7 * this.dim);
    this.sun.color.setRGB(1, 0.82 + 0.18 * (1 - warm), 0.62 + 0.38 * (1 - warm));
    this.hemi.intensity = (0.55 + 0.75 * day) * (1 - 0.6 * this.dim);

    // Morning haze: thick at 06:30, clearing by 09:30.
    const clear = THREE.MathUtils.clamp((simSeconds - 23400) / (34200 - 23400), 0, 1);
    this.fog.density = (0.000085 - 0.00004 * clear) * (1 + 0.6 * this.dim);
    const hazeWarm = new THREE.Color(0xd9b99a);
    const hazeDay = new THREE.Color(0xbfcbd6);
    const night = new THREE.Color(0x1a2233);
    this.fog.color.copy(hazeDay).lerp(hazeWarm, warm * 0.8).lerp(night, (1 - day) * 0.7 + this.dim * 0.6);
    renderer.toneMappingExposure = (0.45 + 0.25 * day) * (1 - 0.45 * this.dim);
  }

  dispose(): void {
    this.scene.remove(this.sky, this.sun, this.sun.target, this.hemi);
    this.sky.geometry.dispose();
    this.sky.material.dispose();
    this.scene.fog = null;
  }
}
