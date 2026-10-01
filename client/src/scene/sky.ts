/**
 * Sky (Preetham Sky shader), a warm low-angle morning sun with soft shadows
 * that follow the view, hemisphere sky light, and morning haze fog tinted to
 * the horizon. All driven by the sim clock and a small solar position model.
 */

import * as THREE from 'three';
import { Sky } from 'three/examples/jsm/objects/Sky.js';
import { localSecondsToDate, solarPosition, sunDirection } from './solar';

export class SkySystem {
  readonly sky = new Sky();
  readonly sun = new THREE.DirectionalLight(0xffffff, 2.2);
  readonly hemi = new THREE.HemisphereLight(0xbfd6ff, 0x5a4a3a, 1.1);
  readonly fog = new THREE.FogExp2(0xc9c3b8, 0.00006);
  /** 0 in daylight, 1 before dawn (drives headlight intensity) */
  darkness = 0;
  /** 0..1 dimming for the ghost-traffic night look */
  dim = 0;
  private lastKey = '';
  private day = new Date();
  private sunDir = new THREE.Vector3(0, 1, 0);
  private shadowHalf = 0;

  constructor(
    private scene: THREE.Scene,
    private lat: number,
    private lon: number,
    private timeZone: string,
  ) {
    this.sky.scale.setScalar(100000);
    const u = this.sky.material.uniforms;
    u['turbidity']!.value = 6.5;
    u['rayleigh']!.value = 1.4;
    u['mieCoefficient']!.value = 0.005;
    u['mieDirectionalG']!.value = 0.86;
    if (u['cloudCoverage']) u['cloudCoverage'].value = 0.3;
    if (u['cloudDensity']) u['cloudDensity'].value = 0.35;
    this.sky.frustumCulled = false;
    scene.add(this.sky);
    this.sun.castShadow = false;
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.6;
    this.sun.shadow.radius = 3;
    this.sun.shadow.blurSamples = 12;
    scene.add(this.sun);
    scene.add(this.sun.target);
    scene.add(this.hemi);
    scene.fog = this.fog;
  }

  configureShadows(enabled: boolean, mapSize: number): void {
    this.sun.castShadow = enabled;
    if (this.sun.shadow.mapSize.x !== mapSize) {
      this.sun.shadow.mapSize.set(mapSize, mapSize);
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null;
    }
  }

  /** Keep the shadow camera centered on what the user is looking at. */
  followTarget(target: THREE.Vector3, viewDistance: number): void {
    const half = THREE.MathUtils.clamp(viewDistance * 0.8, 150, 3500);
    const cam = this.sun.shadow.camera;
    if (Math.abs(half - this.shadowHalf) / Math.max(half, 1) > 0.08) {
      this.shadowHalf = half;
      cam.left = -half;
      cam.right = half;
      cam.top = half;
      cam.bottom = -half;
      cam.near = 10;
      cam.far = 12000;
      cam.updateProjectionMatrix();
    }
    // snap to shadow texels to reduce shimmering while panning
    const texel = (2 * this.shadowHalf) / this.sun.shadow.mapSize.x;
    const tx = Math.round(target.x / texel) * texel;
    const tz = Math.round(target.z / texel) * texel;
    this.sun.target.position.set(tx, target.y, tz);
    this.sun.position.set(tx + this.sunDir.x * 5000, target.y + this.sunDir.y * 5000, tz + this.sunDir.z * 5000);
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
    // light never comes from below the horizon (pre-dawn: low grazing light)
    this.sunDir.set(x, Math.max(y, 0.07), z).normalize();

    const el = pos.elevation;
    const day = THREE.MathUtils.smoothstep(el, -5, 10); // 0 before dawn, 1 in daylight
    const warm = 1 - THREE.MathUtils.smoothstep(el, 3, 30); // golden hour tint
    this.darkness = 1 - THREE.MathUtils.smoothstep(el, -4, 8);
    this.sun.intensity = (0.25 + 2.6 * day) * (1 - 0.75 * this.dim);
    this.sun.color.setRGB(1, 0.78 + 0.22 * (1 - warm), 0.55 + 0.45 * (1 - warm));
    this.hemi.intensity = (0.45 + 0.7 * day) * (1 - 0.6 * this.dim);
    this.hemi.color.setRGB(0.72 + 0.1 * warm, 0.82, 1.0);
    this.hemi.groundColor.setRGB(0.36, 0.3, 0.24);

    // Morning haze: thick at 06:30, clearing by 09:30, tinted to the horizon.
    const clear = THREE.MathUtils.clamp((simSeconds - 23400) / (34200 - 23400), 0, 1);
    this.fog.density = (0.00008 - 0.000035 * clear) * (1 + 0.5 * this.dim);
    const hazeWarm = new THREE.Color(0xe3b98f);
    const hazeDay = new THREE.Color(0xb9c8d8);
    const dawn = new THREE.Color(0x4a5470);
    const night = new THREE.Color(0x0d1220);
    this.fog.color
      .copy(hazeDay)
      .lerp(hazeWarm, warm * 0.75)
      .lerp(dawn, (1 - day) * 0.8)
      .lerp(night, this.dim * 0.85);
    renderer.toneMappingExposure = (0.42 + 0.22 * day) * (1 - 0.35 * this.dim);
  }

  dispose(): void {
    this.scene.remove(this.sky, this.sun, this.sun.target, this.hemi);
    this.sky.geometry.dispose();
    this.sky.material.dispose();
    this.sun.shadow.map?.dispose();
    this.scene.fog = null;
  }
}
