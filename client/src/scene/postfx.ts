/**
 * Post-processing: optional ambient occlusion (N8AO), HDR bloom (catches
 * headlights, the sun and ghost-traffic trails), ACES tone mapping via
 * OutputPass, then a light vignette and morning color grade.
 */

import { N8AOPass } from 'n8ao';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { QualitySettings } from './quality';

const GradeShader = {
  name: 'RedrawGrade',
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uVignette: { value: 0.32 },
    uWarm: { value: 0.5 },
    uSat: { value: 1.06 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uVignette;
    uniform float uWarm;
    uniform float uSat;
    varying vec2 vUv;
    void main() {
      vec4 c = texture2D(tDiffuse, vUv);
      float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
      vec3 col = mix(vec3(l), c.rgb, uSat);
      // split tone: cool shadows, warm highlights (morning light)
      vec3 shadowTint = vec3(0.96, 0.99, 1.05);
      vec3 highTint = vec3(1.05, 1.0, 0.93);
      col *= mix(vec3(1.0), mix(shadowTint, highTint, smoothstep(0.15, 0.85, l)), uWarm);
      vec2 d = vUv - 0.5;
      float v = 1.0 - uVignette * smoothstep(0.25, 0.85, dot(d, d) * 2.2);
      gl_FragColor = vec4(col * v, c.a);
    }`,
};

export class PostFX {
  readonly composer: EffectComposer;
  private bloom: UnrealBloomPass | null = null;
  private ao: N8AOPass | null = null;
  private grade: ShaderPass | null = null;

  constructor(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    q: QualitySettings,
  ) {
    const size = renderer.getSize(new THREE.Vector2());
    const rt = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: q.ao ? 0 : 4 });
    this.composer = new EffectComposer(renderer, rt);
    this.composer.setPixelRatio(renderer.getPixelRatio());
    let usedAo = false;
    if (q.ao) {
      try {
        this.ao = new N8AOPass(scene, camera, size.x, size.y);
        const c = this.ao.configuration;
        c.aoRadius = 8;
        c.distanceFalloff = 1.5;
        c.intensity = 2.2;
        c.halfRes = true;
        c.depthAwareUpsampling = true;
        c.gammaCorrection = false;
        c.screenSpaceRadius = false;
        this.ao.setQualityMode('Low');
        this.composer.addPass(this.ao);
        usedAo = true;
      } catch (e) {
        console.warn('AO unavailable, continuing without it', e);
        this.ao = null;
      }
    }
    if (!usedAo) this.composer.addPass(new RenderPass(scene, camera));
    if (q.bloom) {
      this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x / 2, size.y / 2), 0.3, 0.5, 4.5);
      this.composer.addPass(this.bloom);
    }
    this.composer.addPass(new OutputPass());
    if (q.grade) {
      this.grade = new ShaderPass(GradeShader);
      this.composer.addPass(this.grade);
    }
  }

  setSize(w: number, h: number, pixelRatio: number): void {
    this.composer.setPixelRatio(pixelRatio);
    this.composer.setSize(w, h);
  }

  /** Stronger bloom for the ghost-traffic night look. */
  setGhost(on: boolean): void {
    if (this.bloom) {
      this.bloom.strength = on ? 1.25 : 0.3;
      this.bloom.threshold = on ? 0.6 : 4.5;
    }
    if (this.grade) this.grade.uniforms['uVignette']!.value = on ? 0.55 : 0.32;
  }

  render(dt: number): void {
    this.composer.render(dt);
  }

  dispose(): void {
    this.ao?.dispose();
    this.bloom?.dispose();
    this.composer.dispose();
  }
}
