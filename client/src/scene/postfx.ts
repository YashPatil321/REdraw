/**
 * HDR post-processing pipeline:
 *
 *   scene -> MSAA half-float target (+ depth texture)
 *         -> ambient occlusion (N8AO, reads that target's color + depth)
 *         -> atmosphere: aerial perspective + height fog with sun in-scattering,
 *            tinted by the sky cube along each view ray (seamless with the sky),
 *            plus screen-space god rays when the sun is low and on screen
 *         -> bloom (HDR, threshold follows exposure)
 *         -> tone mapping (AgX with a punchy look, or ACES), warm morning grade,
 *            vignette, dither -> screen
 *
 * Photo pixels: Google photoreal tiles write alpha 0 into the scene target
 * (tiles.ts). The final pass takes those pixels straight from the scene
 * target (their own photo colors plus the light haze of the direct path),
 * skipping AO, atmosphere, bloom, tone mapping and grading, so the photo
 * looks exactly as when drawn directly while overlays drawn over it go
 * through the same pipeline as over our world (they look identical on both).
 */

import { N8AOPass } from 'n8ao';
import * as THREE from 'three';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { QualitySettings } from './quality';
import type { SkySystem } from './sky';

const quadVert = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const atmosphereFrag = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform samplerCube tSky;
uniform sampler2D tRays;
uniform mat4 uProjInv;
uniform mat4 uCamWorld;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uFogA;
uniform float uFogB;
uniform float uFogH0;
uniform vec3 uAerial;
uniform float uSunScatter;
uniform float uSkyMip;
uniform vec3 uRaysColor;
uniform float uRaysOn;
uniform float uEnabled;

float hg(float mu, float g) {
  float g2 = g * g;
  return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}

void main() {
  vec3 col = texture2D(tColor, vUv).rgb;
  float d = texture2D(tDepth, vUv).x;
  if (d < 1.0 && uEnabled > 0.5) {
    vec4 vp = uProjInv * vec4(vUv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    vp.xyz /= vp.w;
    float dist = length(vp.xyz);
    vec3 wdir = normalize((uCamWorld * vec4(vp.xyz, 0.0)).xyz);
    // exponential height fog, integrated analytically along the view ray
    float k = uFogB * wdir.y * dist;
    float fh = abs(k) > 1e-3 ? (1.0 - exp(-k)) / k : 1.0 - 0.5 * k;
    float odH = uFogA * exp(-uFogB * (uCamPos.y - uFogH0)) * dist * fh;
    vec3 T = exp(-(vec3(odH) + uAerial * dist));
    // in-scattered light: sky radiance along a horizon-ward direction (matches the sky
    // exactly at the far end) plus the haze's forward lobe toward the sun
    vec3 hd = normalize(vec3(wdir.x, max(wdir.y, 0.0) * 0.6 + 0.015, wdir.z));
    vec3 skyC = textureLod(tSky, hd, uSkyMip).rgb;
    float mu = dot(wdir, uSunDir);
    vec3 ins = skyC + uSunColor * hg(mu, 0.72) * uSunScatter;
    col = col * T + ins * (1.0 - T);
  }
  if (uRaysOn > 0.5) col += texture2D(tRays, vUv).r * uRaysColor;
  gl_FragColor = vec4(col, 1.0);
}`;

const raysFrag = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tDepth;
uniform vec2 uSunUv;
uniform float uAspect;
void main() {
  const int N = 48;
  vec2 delta = (uSunUv - vUv) * (0.92 / float(N));
  vec2 uv = vUv;
  float acc = 0.0;
  float w = 1.0;
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  uv += delta * jitter;
  for (int i = 0; i < N; i++) {
    uv += delta;
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { w *= 0.965; continue; }
    float sky = step(0.99999, texture2D(tDepth, uv).x);
    vec2 o = (uv - uSunUv) * vec2(uAspect, 1.0);
    float glow = exp(-dot(o, o) * 28.0);
    acc += sky * glow * w;
    w *= 0.965;
  }
  gl_FragColor = vec4(acc / float(N), 0.0, 0.0, 1.0);
}`;

const finalFrag = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tColor;
uniform sampler2D tRaw;
uniform sampler2D tDepth;
uniform mat4 uProjInv;
uniform float uPhotoOn;
uniform vec3 uPhotoFogColor;
uniform float uPhotoFogDensity;
uniform float uExposure;
uniform float uToneMode;
uniform float uGrade;
uniform float uVignette;
uniform float uWarm;
uniform float uSat;
uniform float uContrast;
uniform float uTime;

vec3 RRTAndODTFit(vec3 v) {
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return a / b;
}
vec3 aces(vec3 color) {
  const mat3 I = mat3(vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777));
  const mat3 O = mat3(vec3(1.60475, -0.10208, -0.00327), vec3(-0.53108, 1.10813, -0.07276), vec3(-0.07367, -0.00605, 1.07602));
  color = I * (color / 0.6);
  color = RRTAndODTFit(color);
  return clamp(O * color, 0.0, 1.0);
}
vec3 agxContrast(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 agx(vec3 color) {
  const mat3 toRec2020 = mat3(vec3(0.6274, 0.0691, 0.0164), vec3(0.3293, 0.9195, 0.0880), vec3(0.0433, 0.0113, 0.8956));
  const mat3 fromRec2020 = mat3(vec3(1.6605, -0.1246, -0.0182), vec3(-0.5876, 1.1329, -0.1006), vec3(-0.0728, -0.0083, 1.1187));
  const mat3 inset = mat3(vec3(0.856627153315983, 0.137318972929847, 0.11189821299995), vec3(0.0951212405381588, 0.761241990602591, 0.0767994186031903), vec3(0.0482516061458583, 0.101439036467562, 0.811302368396859));
  const mat3 outset = mat3(vec3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826), vec3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294), vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
  color = inset * (toRec2020 * color);
  color = clamp((log2(max(color, 1e-10)) + 12.47393) / 16.50000, 0.0, 1.0);
  color = agxContrast(color);
  // "punchy" look: a little more contrast and saturation
  float l = dot(color, vec3(0.2126, 0.7152, 0.0722));
  color = pow(max(color, 0.0), vec3(1.18));
  color = l + (color - l) * 1.22;
  color = outset * color;
  color = pow(max(vec3(0.0), color), vec3(2.2));
  return clamp(fromRec2020 * color, 0.0, 1.0);
}
vec3 toSRGB(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
void main() {
  vec3 hdr = texture2D(tColor, vUv).rgb * uExposure;
  vec3 c = uToneMode > 0.5 ? agx(hdr) : aces(hdr);
  c = toSRGB(c);
  if (uGrade > 0.5) {
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    // gentle S-curve around mid grey
    c = mix(c, c * c * (3.0 - 2.0 * c), uContrast);
    c = mix(vec3(l), c, uSat);
    // split tone: cool, slightly teal shadows; warm, golden highlights (morning light)
    vec3 shadowTint = vec3(0.965, 0.995, 1.04);
    vec3 highTint = vec3(1.045, 1.0, 0.935);
    c *= mix(vec3(1.0), mix(shadowTint, highTint, smoothstep(0.1, 0.8, l)), uWarm);
  }
  if (uPhotoOn > 0.5) {
    // photo pixels (alpha 0 in the scene target): their own colors + light haze
    vec4 raw = texture2D(tRaw, vUv);
    float w = clamp(1.0 - raw.a, 0.0, 1.0);
    if (w > 0.002) {
      float dz = texture2D(tDepth, vUv).x;
      vec4 vp = uProjInv * vec4(vUv * 2.0 - 1.0, dz * 2.0 - 1.0, 1.0);
      float depth = -vp.z / vp.w;
      float f = 1.0 - exp(-uPhotoFogDensity * uPhotoFogDensity * depth * depth);
      vec3 photo = mix(max(raw.rgb, vec3(0.0)), uPhotoFogColor, f);
      c = mix(c, toSRGB(clamp(photo, 0.0, 1.0)), w);
    }
  }
  if (uGrade > 0.5) {
    vec2 d = vUv - 0.5;
    c *= 1.0 - uVignette * smoothstep(0.2, 0.9, dot(d, d) * 2.4);
  }
  // triangular dither: no banding in the sky gradient
  vec2 p = gl_FragCoord.xy + fract(uTime) * 61.0;
  float n = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453) + fract(sin(dot(p, vec2(39.346, 11.135))) * 24634.634) - 1.0;
  c += n / 255.0;
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

export interface PostTimings {
  scene: number;
}

export class PostFX {
  private sceneRT: THREE.WebGLRenderTarget;
  private rtA: THREE.WebGLRenderTarget;
  private rtB: THREE.WebGLRenderTarget;
  private raysRT: THREE.WebGLRenderTarget | null = null;
  private ao: N8AOPass | null = null;
  private bloom: UnrealBloomPass | null = null;
  private atmo: FullScreenQuad;
  private atmoMat: THREE.ShaderMaterial;
  private rays: FullScreenQuad | null = null;
  private raysMat: THREE.ShaderMaterial | null = null;
  private fin: FullScreenQuad;
  private finMat: THREE.ShaderMaterial;
  private ghost = false;
  private size = new THREE.Vector2(1, 1);
  private tmpV = new THREE.Vector4();
  /** aerial perspective strength multiplier (photo mode / debug) */
  hazeScale = 1;
  /** fog/AO/bloom tuned to the camera distance each frame (set by the viewer) */
  viewDistance = 500;
  /** ambient occlusion pass on (off while only photo tiles + overlays are drawn) */
  aoEnabled = true;
  /** photo tiles may be in the frame: pass their pixels through (see header) */
  photo = { on: false, fogColor: new THREE.Color(0.8, 0.8, 0.8), fogDensity: 0 };

  constructor(
    private renderer: THREE.WebGLRenderer,
    private scene: THREE.Scene,
    private camera: THREE.PerspectiveCamera,
    q: QualitySettings,
    private sky: SkySystem | null,
  ) {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    this.size.copy(size);
    const maxSamples = renderer.capabilities.maxSamples ?? 4;
    const samples = Math.min(q.msaa, maxSamples);
    this.sceneRT = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples, depthBuffer: true });
    this.sceneRT.depthTexture = new THREE.DepthTexture(size.x, size.y, THREE.UnsignedIntType);
    this.sceneRT.depthTexture.format = THREE.DepthFormat;
    const mk = (): THREE.WebGLRenderTarget => new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, depthBuffer: false });
    this.rtA = mk();
    this.rtB = mk();

    if (q.ao !== 'off') {
      try {
        const ao = new N8AOPass(scene, camera, size.x, size.y);
        const c = ao.configuration;
        c['autoRenderBeauty'] = false;
        c.gammaCorrection = false;
        c.screenSpaceRadius = false;
        c.halfRes = q.ao !== 'high';
        c.depthAwareUpsampling = true;
        c.intensity = 2.0;
        c.aoRadius = 4;
        c.distanceFalloff = 1;
        ao.setQualityMode(q.ao === 'high' ? 'High' : q.ao === 'medium' ? 'Medium' : 'Low');
        // reuse our MSAA scene target (resolved color + depth) instead of N8AO's own render
        const own = (ao as unknown as { beautyRenderTarget: THREE.WebGLRenderTarget }).beautyRenderTarget;
        own.dispose();
        (ao as unknown as { beautyRenderTarget: THREE.WebGLRenderTarget }).beautyRenderTarget = this.sceneRT;
        this.ao = ao;
      } catch (e) {
        console.warn('AO unavailable, continuing without it', e);
        this.ao = null;
      }
    }

    this.atmoMat = new THREE.ShaderMaterial({
      vertexShader: quadVert,
      fragmentShader: atmosphereFrag,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
      uniforms: {
        tColor: { value: null },
        tDepth: { value: this.sceneRT.depthTexture },
        tSky: { value: null },
        tRays: { value: null },
        uProjInv: { value: new THREE.Matrix4() },
        uCamWorld: { value: new THREE.Matrix4() },
        uCamPos: { value: new THREE.Vector3() },
        uSunDir: { value: new THREE.Vector3(0, 1, 0) },
        uSunColor: { value: new THREE.Color(1, 1, 1) },
        uFogA: { value: 6e-5 },
        uFogB: { value: 1 / 450 },
        uFogH0: { value: 100 },
        uAerial: { value: new THREE.Vector3(1.2e-5, 1.6e-5, 2.4e-5) },
        uSunScatter: { value: 0.25 },
        uSkyMip: { value: 2.5 },
        uRaysColor: { value: new THREE.Color(0, 0, 0) },
        uRaysOn: { value: 0 },
        uEnabled: { value: 1 },
      },
    });
    this.atmo = new FullScreenQuad(this.atmoMat);

    if (q.godRays) {
      this.raysRT = new THREE.WebGLRenderTarget(Math.max(1, size.x >> 1), Math.max(1, size.y >> 1), { type: THREE.HalfFloatType, depthBuffer: false });
      this.raysMat = new THREE.ShaderMaterial({
        vertexShader: quadVert,
        fragmentShader: raysFrag,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
        uniforms: { tDepth: { value: this.sceneRT.depthTexture }, uSunUv: { value: new THREE.Vector2() }, uAspect: { value: 1 } },
      });
      this.rays = new FullScreenQuad(this.raysMat);
      this.atmoMat.uniforms['tRays']!.value = this.raysRT.texture;
    }

    if (q.bloom) {
      this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x, size.y), 0.22, 0.55, 8);
      this.bloom.renderToScreen = false;
    }

    this.finMat = new THREE.ShaderMaterial({
      vertexShader: quadVert,
      fragmentShader: finalFrag,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
      uniforms: {
        tColor: { value: null },
        tRaw: { value: this.sceneRT.texture },
        tDepth: { value: this.sceneRT.depthTexture },
        uProjInv: { value: new THREE.Matrix4() },
        uPhotoOn: { value: 0 },
        uPhotoFogColor: { value: new THREE.Color(0.8, 0.8, 0.8) },
        uPhotoFogDensity: { value: 0 },
        uExposure: { value: 1 },
        uToneMode: { value: 1 },
        uGrade: { value: q.grade ? 1 : 0 },
        uVignette: { value: 0.22 },
        uWarm: { value: 0.55 },
        uSat: { value: 1.04 },
        uContrast: { value: 0.12 },
        uTime: { value: 0 },
      },
    });
    this.fin = new FullScreenQuad(this.finMat);
  }

  /** Tone mapping operator for the final pass. */
  setToneMapping(mode: 'agx' | 'aces'): void {
    this.finMat.uniforms['uToneMode']!.value = mode === 'agx' ? 1 : 0;
  }

  setSize(w: number, h: number, pixelRatio: number): void {
    const W = Math.max(1, Math.floor(w * pixelRatio));
    const H = Math.max(1, Math.floor(h * pixelRatio));
    this.size.set(W, H);
    this.sceneRT.setSize(W, H);
    this.rtA.setSize(W, H);
    this.rtB.setSize(W, H);
    this.raysRT?.setSize(Math.max(1, W >> 1), Math.max(1, H >> 1));
    this.bloom?.setSize(W, H);
    if (this.ao) {
      this.ao.setSize(W, H);
      // N8AO resizes "its" beauty target (ours) too: fine, same size
    }
  }

  /** Stronger bloom for the ghost-traffic night look. */
  setGhost(on: boolean): void {
    this.ghost = on;
    this.finMat.uniforms['uVignette']!.value = on ? 0.5 : 0.22;
  }

  render(dt: number): void {
    const r = this.renderer;
    const cam = this.camera;
    const sky = this.sky;
    const exposure = sky?.exposure ?? r.toneMappingExposure;

    // 1. scene (MSAA, HDR, depth)
    const prevTone = r.toneMapping;
    r.toneMapping = THREE.NoToneMapping;
    r.setRenderTarget(this.sceneRT);
    r.clear(true, true, false);
    r.render(this.scene, cam);

    // 2. ambient occlusion -> rtA (or straight from the scene target)
    let src: THREE.Texture = this.sceneRT.texture;
    if (this.ao && this.aoEnabled) {
      const c = this.ao.configuration;
      // world-space radius that reads at every scale: ~1.5 m at street level, ~25 m from 3 km up
      const d = this.viewDistance;
      c.aoRadius = THREE.MathUtils.clamp(d * 0.012, 1.4, 30);
      c.distanceFalloff = THREE.MathUtils.clamp(d * 0.004, 0.5, 8);
      c.intensity = d < 60 ? 2.6 : 2.2;
      this.ao.render(r, this.rtA, this.rtA, dt, false);
      src = this.rtA.texture;
    }

    // 3. god rays (half res) when the sun is on screen and low
    let raysOn = 0;
    if (this.rays && this.raysRT && sky) {
      const sd = sky.state.sunDir;
      const el = sky.state.elevation;
      const p = this.tmpV.set(cam.position.x + sd.x * 1e4, cam.position.y + sd.y * 1e4, cam.position.z + sd.z * 1e4, 1);
      p.applyMatrix4(cam.matrixWorldInverse).applyMatrix4(cam.projectionMatrix);
      const strength = (1 - THREE.MathUtils.smoothstep(el, 8, 40)) * THREE.MathUtils.smoothstep(el, -2, 2);
      if (p.w > 0 && strength > 0.01) {
        const sx = (p.x / p.w) * 0.5 + 0.5;
        const sy = (p.y / p.w) * 0.5 + 0.5;
        const onScreen = 1 - THREE.MathUtils.smoothstep(Math.max(Math.abs(sx - 0.5), Math.abs(sy - 0.5)), 0.5, 1.0);
        if (onScreen > 0.01) {
          (this.raysMat!.uniforms['uSunUv']!.value as THREE.Vector2).set(sx, sy);
          this.raysMat!.uniforms['uAspect']!.value = cam.aspect;
          r.setRenderTarget(this.raysRT);
          this.rays.render(r);
          raysOn = 1;
          const k = strength * onScreen * 1.6 * (1 - 0.8 * (sky.dim ?? 0));
          (this.atmoMat.uniforms['uRaysColor']!.value as THREE.Color).copy(sky.state.sunColor).multiplyScalar(k);
        }
      }
    }

    // 4. atmosphere -> rtB
    const u = this.atmoMat.uniforms;
    u['tColor']!.value = src;
    u['uRaysOn']!.value = raysOn;
    if (sky) {
      u['tSky']!.value = sky.skyTexture;
      (u['uSunDir']!.value as THREE.Vector3).copy(sky.state.sunDir);
      (u['uSunColor']!.value as THREE.Color).copy(sky.state.sunColor);
      u['uEnabled']!.value = 1;
      // clearing morning haze; denser in the ghost (night) look
      const haze = this.hazeScale * (1 + 0.6 * sky.dim);
      u['uFogA']!.value = 3.2e-5 * haze;
      u['uFogB']!.value = 1 / 520;
      u['uFogH0']!.value = sky.fogBase;
      (u['uAerial']!.value as THREE.Vector3).set(0.55e-5, 0.85e-5, 1.5e-5).multiplyScalar(haze);
    } else u['uEnabled']!.value = 0;
    (u['uProjInv']!.value as THREE.Matrix4).copy(cam.projectionMatrixInverse);
    (u['uCamWorld']!.value as THREE.Matrix4).copy(cam.matrixWorld);
    (u['uCamPos']!.value as THREE.Vector3).copy(cam.position);
    u['uSkyMip']!.value = 2.0;
    r.setRenderTarget(this.rtB);
    this.atmo.render(r);

    // 5. bloom (in place on rtB); threshold in scene units follows the exposure
    if (this.bloom) {
      this.bloom.threshold = (this.ghost ? 0.55 : 1.15) / Math.max(exposure, 1e-3);
      this.bloom.strength = this.ghost ? 1.1 : 0.2;
      this.bloom.radius = this.ghost ? 0.6 : 0.5;
      this.bloom.render(r, this.rtA, this.rtB, dt, false);
    }

    // 6. tone map + grade -> screen
    this.finMat.uniforms['tColor']!.value = this.rtB.texture;
    this.finMat.uniforms['uExposure']!.value = exposure;
    this.finMat.uniforms['uTime']!.value = performance.now() / 1000;
    const fu = this.finMat.uniforms;
    fu['uPhotoOn']!.value = this.photo.on ? 1 : 0;
    if (this.photo.on) {
      (fu['uProjInv']!.value as THREE.Matrix4).copy(cam.projectionMatrixInverse);
      (fu['uPhotoFogColor']!.value as THREE.Color).copy(this.photo.fogColor);
      fu['uPhotoFogDensity']!.value = this.photo.fogDensity;
    }
    r.setRenderTarget(null);
    this.fin.render(r);
    r.toneMapping = prevTone;
  }

  dispose(): void {
    if (this.ao) {
      // the beauty target is ours: detach before N8AO disposes it
      (this.ao as unknown as { beautyRenderTarget: THREE.WebGLRenderTarget | null }).beautyRenderTarget = new THREE.WebGLRenderTarget(1, 1);
      this.ao.dispose();
    }
    this.bloom?.dispose();
    this.sceneRT.depthTexture?.dispose();
    this.sceneRT.dispose();
    this.rtA.dispose();
    this.rtB.dispose();
    this.raysRT?.dispose();
    this.atmoMat.dispose();
    this.raysMat?.dispose();
    this.finMat.dispose();
    this.atmo.dispose();
    this.rays?.dispose();
    this.fin.dispose();
  }
}
