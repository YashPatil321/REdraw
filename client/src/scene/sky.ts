/**
 * Sky, sun and ambient light, all driven by the sim clock and the real solar
 * position:
 * - a physically based atmosphere (Rayleigh + Mie + ozone, see
 *   atmosphereModel.ts) rendered into a cube map whenever the sun moves,
 * - that cube map drawn as the visible sky (plus an HDR sun disc) and
 *   prefiltered with PMREM into `scene.environment` (image based ambient and
 *   reflections for every PBR material),
 * - the sun's color and intensity from the atmosphere's transmittance,
 *   applied to the cascaded-shadow sun light (shadows.ts),
 * - exposure with partial eye adaptation, and a fallback FogExp2 tinted to
 *   the horizon for the no-post-processing path (low quality, photoreal).
 */

import * as THREE from 'three';
import { ATMOSPHERE_GLSL, luminance, skyRadiance, sunTransmittance } from './atmosphereModel';
import { SunShadows } from './shadows';
import { localSecondsToDate, solarPosition, sunDirection } from './solar';

/** Top-of-atmosphere sun irradiance in renderer units (light intensity). */
export const SUN_E0 = 5.0;
/** Haze multiplier on the Mie coefficient (clear October morning, light marine haze). */
const MIE_HAZE = 2.2;

const cubeVert = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize((modelMatrix * vec4(position, 0.0)).xyz);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const cubeFrag = /* glsl */ `
precision highp float;
varying vec3 vDir;
uniform vec3 uSunDir;
uniform float uAlt;
uniform float uMie;
uniform float uE0;
uniform float uMs;
uniform vec3 uGround;
${ATMOSPHERE_GLSL}
void main() {
  vec3 rd = normalize(vDir);
  vec3 L = atmSky(rd, uSunDir, uAlt, uMie, uGround, uMs) * uE0;
  gl_FragColor = vec4(min(L, vec3(6e4)), 1.0);
}`;

const domeVert = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = (modelMatrix * vec4(position, 0.0)).xyz;
  vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  gl_Position = p.xyww; // on the far plane: everything else draws over it
}`;

const domeFrag = /* glsl */ `
precision highp float;
varying vec3 vDir;
uniform samplerCube uSky;
uniform vec3 uSunDir;
uniform vec3 uSunRadiance;
uniform float uDim;
void main() {
  vec3 d = normalize(vDir);
  vec3 c = textureCube(uSky, d).rgb;
  // sun disc (0.53 deg) with limb darkening, clamped HDR so bloom stays a glow
  float cosA = dot(d, uSunDir);
  float r = clamp(acos(clamp(cosA, -1.0, 1.0)) / 0.00465, 0.0, 2.0);
  float disc = 1.0 - smoothstep(0.85, 1.0, r);
  float limb = pow(max(1.0 - r * r, 0.0), 0.25) * 0.65 + 0.35;
  c += uSunRadiance * disc * limb;
  // fine dither against banding in the smooth gradient (8-bit output)
  float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  c *= 1.0 + (n - 0.5) * 0.012;
  gl_FragColor = vec4(c * (1.0 - 0.9 * uDim), 1.0);
}`;

export interface SkyState {
  /** unit vector toward the sun */
  sunDir: THREE.Vector3;
  /** sun elevation (deg) */
  elevation: number;
  /** sun irradiance color (linear, includes SUN_E0 and transmittance) */
  sunColor: THREE.Color;
}

export class SkySystem {
  readonly shadows: SunShadows;
  readonly hemi = new THREE.HemisphereLight(0xbfd6ff, 0x5a4a3a, 0);
  readonly fog = new THREE.FogExp2(0xc9c3b8, 0.00006);
  /** visible sky (cube map + sun disc), follows the camera */
  readonly dome: THREE.Mesh;
  /** 0 in daylight, 1 before dawn (drives headlight intensity) */
  darkness = 0;
  /** 0..1 dimming for the ghost-traffic night look */
  dim = 0;
  /** Photoreal base map: lighter haze (the photo already has atmosphere), brighter overlays. */
  photoreal = false;
  /** HDR pipeline active: aerial perspective is done in post, so no FogExp2 */
  postAtmosphere = false;
  /** image based ambient on (PMREM); off = hemisphere light (low quality) */
  ibl = true;
  readonly state: SkyState = { sunDir: new THREE.Vector3(0, 1, 0), elevation: 45, sunColor: new THREE.Color(1, 1, 1) };
  /** sky radiance at the horizon away from / toward the sun (linear, for fog tinting) */
  readonly horizon = new THREE.Color(0.6, 0.7, 0.85);
  readonly horizonSun = new THREE.Color(0.9, 0.75, 0.6);
  /** height (m) the ground haze is densest at (valley floor) */
  fogBase = 120;
  /** exposure the scene should use (eye adaptation) */
  exposure = 1;
  /** bumped whenever the sky cube / environment is re-rendered */
  version = 0;

  private cubeRT: THREE.WebGLCubeRenderTarget;
  private cubeCam: THREE.CubeCamera;
  private skyScene = new THREE.Scene();
  private cubeMat: THREE.ShaderMaterial;
  private domeMat: THREE.ShaderMaterial;
  private pmrem: THREE.PMREMGenerator | null = null;
  private envRT: THREE.WebGLRenderTarget | null = null;
  private lastKey = '';
  private lastEnvKey = '';
  private day = new Date();
  private skySize = 0;
  private pendingEnv = false;

  constructor(
    private scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    private renderer: THREE.WebGLRenderer,
    private lat: number,
    private lon: number,
    private timeZone: string,
    skySize = 128,
  ) {
    this.shadows = new SunShadows(scene, camera);
    this.cubeMat = new THREE.ShaderMaterial({
      vertexShader: cubeVert,
      fragmentShader: cubeFrag,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
      uniforms: {
        uSunDir: { value: new THREE.Vector3(0, 1, 0) },
        uAlt: { value: 400 },
        uMie: { value: MIE_HAZE },
        uE0: { value: SUN_E0 },
        uMs: { value: 0.55 },
        uGround: { value: new THREE.Color(0.2, 0.18, 0.14) },
      },
    });
    const box = new THREE.Mesh(new THREE.BoxGeometry(10, 10, 10), this.cubeMat);
    box.frustumCulled = false;
    this.skyScene.add(box);
    this.cubeRT = this.makeCube(skySize);
    this.cubeCam = new THREE.CubeCamera(0.5, 20, this.cubeRT);
    this.skyScene.add(this.cubeCam);

    this.domeMat = new THREE.ShaderMaterial({
      vertexShader: domeVert,
      fragmentShader: domeFrag,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: true,
      fog: false,
      toneMapped: true,
      uniforms: {
        uSky: { value: this.cubeRT.texture },
        uSunDir: { value: new THREE.Vector3(0, 1, 0) },
        uSunRadiance: { value: new THREE.Vector3(1, 1, 1) },
        uDim: { value: 0 },
      },
    });
    this.dome = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), this.domeMat);
    this.dome.name = 'sky-dome';
    this.dome.frustumCulled = false;
    this.dome.renderOrder = -1000;
    this.dome.raycast = () => undefined;
    scene.add(this.dome);
    scene.add(this.hemi);
    scene.fog = this.fog;
  }

  private makeCube(size: number): THREE.WebGLCubeRenderTarget {
    this.skySize = size;
    const rt = new THREE.WebGLCubeRenderTarget(size, {
      type: THREE.HalfFloatType,
      generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter,
    });
    return rt;
  }

  /** Cube texture of the sky (for the atmosphere post pass). */
  get skyTexture(): THREE.CubeTexture {
    return this.cubeRT.texture;
  }

  get environment(): THREE.Texture | null {
    return this.envRT?.texture ?? null;
  }

  setSkySize(size: number): void {
    if (size === this.skySize) return;
    const old = this.cubeRT;
    this.cubeRT = this.makeCube(size);
    this.skyScene.remove(this.cubeCam);
    this.cubeCam = new THREE.CubeCamera(0.5, 20, this.cubeRT);
    this.skyScene.add(this.cubeCam);
    this.domeMat.uniforms['uSky']!.value = this.cubeRT.texture;
    old.dispose();
    this.lastKey = '';
  }

  configureShadows(enabled: boolean, mapSize: number, cascades = 3, maxFar = 2000): void {
    this.shadows.configure(enabled, mapSize, cascades, maxFar);
  }

  /** Shadow cascades reach about this far (scaled to what the camera is looking at). */
  followTarget(_target: THREE.Vector3, viewDistance: number): void {
    this.shadows.setMaxFar(THREE.MathUtils.clamp(viewDistance * 2.2, 250, this.maxShadowFar));
  }

  maxShadowFar = 2500;

  /** Update for sim time (seconds since local midnight). */
  update(simSeconds: number, renderer: THREE.WebGLRenderer, camera: THREE.Camera): void {
    // dome follows the camera
    this.dome.position.copy(camera.position);
    const key = `${Math.round(simSeconds / 20)}:${this.dim.toFixed(2)}:${this.photoreal ? 1 : 0}:${this.postAtmosphere ? 1 : 0}:${this.ibl ? 1 : 0}`;
    if (key !== this.lastKey) {
      this.lastKey = key;
      this.recompute(simSeconds, renderer);
    }
    if (this.pendingEnv) this.renderEnv();
    renderer.toneMappingExposure = this.exposure;
    this.shadows.update();
  }

  private recompute(simSeconds: number, renderer: THREE.WebGLRenderer): void {
    const date = localSecondsToDate(simSeconds, this.day, this.timeZone);
    const pos = solarPosition(date, this.lat, this.lon);
    const [x, y, z] = sunDirection(pos);
    const el = pos.elevation;
    const st = this.state;
    st.sunDir.set(x, y, z).normalize();
    st.elevation = el;

    // sun color: transmittance through the atmosphere (photometric, no fudge)
    const T = sunTransmittance(el, 400, MIE_HAZE);
    st.sunColor.setRGB(T[0] * SUN_E0, T[1] * SUN_E0, T[2] * SUN_E0);
    const tl = luminance(T);

    // light: never from below the horizon (grazing pre-dawn light is sky only)
    const light = this.shadows;
    light.sunDir.set(x, Math.max(y, 0.035), z).normalize();
    const lum = Math.max(tl, 1e-4);
    light.color.setRGB(T[0] / lum, T[1] / lum, T[2] / lum);
    light.intensity = SUN_E0 * tl * (1 - 0.8 * this.dim);

    // sky radiance samples (CPU port of the shader): horizon colors and irradiance
    const sd: [number, number, number] = [st.sunDir.x, st.sunDir.y, st.sunDir.z];
    const az = Math.atan2(st.sunDir.x, -st.sunDir.z);
    const hz = (dAz: number, elev: number): [number, number, number] => {
      const e = THREE.MathUtils.degToRad(elev);
      return [Math.sin(az + dAz) * Math.cos(e), Math.sin(e), -Math.cos(az + dAz) * Math.cos(e)];
    };
    const toward = skyRadiance(hz(0, 2), sd, 400, MIE_HAZE);
    const side1 = skyRadiance(hz(Math.PI / 2, 2), sd, 400, MIE_HAZE);
    const side2 = skyRadiance(hz(-Math.PI / 2, 2), sd, 400, MIE_HAZE);
    const away = skyRadiance(hz(Math.PI, 2), sd, 400, MIE_HAZE);
    const zen = skyRadiance([0, 1, 0], sd, 400, MIE_HAZE);
    const mid = skyRadiance(hz(Math.PI / 2, 35), sd, 400, MIE_HAZE);
    const avgH: [number, number, number] = [0, 1, 2].map((c) => (side1[c]! + side2[c]! + away[c]! + toward[c]!) / 4) as [number, number, number];
    this.horizon.setRGB(avgH[0], avgH[1], avgH[2]).multiplyScalar(SUN_E0);
    this.horizonSun.setRGB(toward[0], toward[1], toward[2]).multiplyScalar(SUN_E0);
    // irradiance on a horizontal surface from the sky dome (coarse quadrature) + direct sun
    const skyE = Math.PI * SUN_E0 * (0.25 * luminance(zen) + 0.45 * luminance(mid) + 0.3 * luminance(avgH));
    const sEl = Math.sin(THREE.MathUtils.degToRad(Math.max(el, 0)));
    const sunE = SUN_E0 * tl * sEl;
    const eh = Math.max(1e-4, skyE + sunE);
    // partial adaptation: dawn reads as dawn, but stays legible
    const ref = SUN_E0 * 0.75;
    this.exposure = THREE.MathUtils.clamp(0.68 * Math.pow(ref / eh, 0.8), 0.3, 12) * (1 - 0.45 * this.dim) * (this.photoreal ? 1.4 : 1);

    this.darkness = 1 - THREE.MathUtils.smoothstep(el, -4, 8);
    const day = THREE.MathUtils.smoothstep(el, -6, 8);

    // sky cube + sun disc
    const u = this.cubeMat.uniforms;
    (u['uSunDir']!.value as THREE.Vector3).copy(st.sunDir);
    const omega = 6.8e-5; // solid angle of the sun (sr)
    const disc = Math.min(1, 160 / Math.max((SUN_E0 * tl) / omega, 1e-6));
    const du = this.domeMat.uniforms;
    (du['uSunDir']!.value as THREE.Vector3).copy(st.sunDir);
    (du['uSunRadiance']!.value as THREE.Vector3).set(T[0], T[1], T[2]).multiplyScalar((SUN_E0 / omega) * disc);
    du['uDim']!.value = this.dim;
    this.cubeCam.update(renderer, this.skyScene);

    // fallback fog (direct rendering)
    this.scene.fog = this.postAtmosphere && !this.photoreal ? null : this.fog;
    const clear = THREE.MathUtils.clamp((simSeconds - 23400) / (34200 - 23400), 0, 1);
    this.fog.density = (0.00007 - 0.00003 * clear) * (1 + 0.5 * this.dim) * (this.photoreal ? 0.55 : 1);
    this.fog.color.copy(this.horizon).lerp(this.horizonSun, 0.3).multiplyScalar(1 - 0.85 * this.dim);

    // hemisphere fallback (no IBL)
    this.hemi.intensity = this.ibl ? 0 : (0.25 + 1.1 * day) * (1 - 0.6 * this.dim);
    const warm = 1 - THREE.MathUtils.smoothstep(el, 2, 35);
    this.hemi.color.setRGB(0.62 + 0.1 * warm, 0.74, 0.95);
    this.hemi.groundColor.setRGB(0.32, 0.27, 0.22);

    // environment (PMREM) when the sun moved enough
    const envKey = `${Math.round(el * 2)}:${Math.round(pos.azimuth)}:${this.dim.toFixed(1)}:${this.ibl ? 1 : 0}`;
    if (envKey !== this.lastEnvKey) {
      this.lastEnvKey = envKey;
      this.pendingEnv = true;
    }
    this.version++;
  }

  private renderEnv(): void {
    this.pendingEnv = false;
    if (!this.ibl) {
      this.scene.environment = null;
      return;
    }
    try {
      this.pmrem ??= new THREE.PMREMGenerator(this.renderer);
      const rt = this.pmrem.fromCubemap(this.cubeRT.texture);
      this.envRT?.dispose();
      this.envRT = rt;
      this.scene.environment = rt.texture;
      this.scene.environmentIntensity = 1 - 0.75 * this.dim;
    } catch (e) {
      console.warn('sky environment map unavailable', e);
      this.scene.environment = null;
      this.ibl = false;
    }
    this.version++;
  }

  dispose(): void {
    this.shadows.dispose();
    this.scene.remove(this.dome, this.hemi);
    this.dome.geometry.dispose();
    this.domeMat.dispose();
    this.cubeMat.dispose();
    this.cubeRT.dispose();
    this.envRT?.dispose();
    this.pmrem?.dispose();
    this.scene.fog = null;
    this.scene.environment = null;
  }
}
