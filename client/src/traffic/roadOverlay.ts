/**
 * Congestion overlay: one ribbon mesh for the whole network, colored per edge
 * from a float texture of v/c values (updated per frame, blended between
 * bins) through a color ramp texture built from congestion.ts. The ribbon
 * keeps a minimum on-screen width so roads stay readable from high up.
 * Geometry is shared between layers (baseline / plan); each layer only owns
 * its v/c texture and material.
 */

import * as THREE from 'three';
import { VC_STOPS, vcColor } from './congestion';
import { ADJ_TEX_W, type RoadNetwork } from './network';

const TEX_W = 2048;
const RAMP_MAX = VC_STOPS[VC_STOPS.length - 1]!.at;

export function buildRoadOverlayGeometry(net: RoadNetwork): THREE.BufferGeometry {
  let segs = 0;
  for (let e = 0; e < net.nEdges; e++) segs += Math.max(0, net.ptCount[e]! - 1);
  const pos = new Float32Array(segs * 4 * 3);
  const perp = new Float32Array(segs * 4 * 2);
  const off = new Float32Array(segs * 4);
  const edgeAttr = new Float32Array(segs * 4);
  const baseW = new Float32Array(segs * 4);
  const ptIdx = new Float32Array(segs * 4);
  const index = new Uint32Array(segs * 6);
  let v = 0;
  let ii = 0;
  for (let e = 0; e < net.nEdges; e++) {
    const s = net.ptStart[e]!;
    const n = net.ptCount[e]!;
    const lanes = Math.max(1, net.edges[e]?.lanes ?? 1);
    const w = 2.2 + lanes * 2.6;
    for (let k = 0; k < n - 1; k++) {
      const a = s + k;
      const b = a + 1;
      const ax = net.pts[a * 3]!;
      const ay = net.pts[a * 3 + 1]!;
      const az = net.pts[a * 3 + 2]!;
      const bx = net.pts[b * 3]!;
      const by = net.pts[b * 3 + 1]!;
      const bz = net.pts[b * 3 + 2]!;
      const dx = bx - ax;
      const dz = bz - az;
      const L = Math.hypot(dx, dz) || 1;
      // right-hand side of travel direction: (-dz, dx)
      const px = -dz / L;
      const pz = dx / L;
      const quad: Array<[number, number, number, number, number]> = [
        [ax, ay, az, 0.08, a],
        [ax, ay, az, 1, a],
        [bx, by, bz, 0.08, b],
        [bx, by, bz, 1, b],
      ];
      for (const [x, y, z, o, pi] of quad) {
        ptIdx[v] = pi;
        pos[v * 3] = x;
        pos[v * 3 + 1] = y;
        pos[v * 3 + 2] = z;
        perp[v * 2] = px;
        perp[v * 2 + 1] = pz;
        off[v] = o;
        edgeAttr[v] = e;
        baseW[v] = w;
        v++;
      }
      const q = v - 4;
      index.set([q, q + 2, q + 1, q + 1, q + 2, q + 3], ii);
      ii += 6;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aPerp', new THREE.BufferAttribute(perp, 2));
  g.setAttribute('aOff', new THREE.BufferAttribute(off, 1));
  g.setAttribute('aEdge', new THREE.BufferAttribute(edgeAttr, 1));
  g.setAttribute('aBaseW', new THREE.BufferAttribute(baseW, 1));
  g.setAttribute('aPt', new THREE.BufferAttribute(ptIdx, 1));
  g.setIndex(new THREE.BufferAttribute(index, 1));
  g.computeBoundingSphere();
  if (g.boundingSphere) g.boundingSphere.radius += 50;
  return g;
}

/** Per-network texture view of `net.yAdj` (vertical draping adjustments). */
const adjTextures = new WeakMap<RoadNetwork, THREE.DataTexture>();
export function adjTexture(net: RoadNetwork): THREE.DataTexture {
  let t = adjTextures.get(net);
  if (!t) {
    t = new THREE.DataTexture(net.yAdj, ADJ_TEX_W, net.yAdj.length / ADJ_TEX_W, THREE.RedFormat, THREE.FloatType);
    t.magFilter = THREE.NearestFilter;
    t.minFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    adjTextures.set(net, t);
  }
  return t;
}

let rampTex: THREE.DataTexture | null = null;
function rampTexture(): THREE.DataTexture {
  if (rampTex) return rampTex;
  const N = 256;
  const data = new Uint8Array(N * 4);
  for (let i = 0; i < N; i++) {
    const c = vcColor((i / (N - 1)) * RAMP_MAX);
    data[i * 4] = Math.round(c[0] * 255);
    data[i * 4 + 1] = Math.round(c[1] * 255);
    data[i * 4 + 2] = Math.round(c[2] * 255);
    data[i * 4 + 3] = 255;
  }
  rampTex = new THREE.DataTexture(data, N, 1, THREE.RGBAFormat);
  rampTex.magFilter = THREE.LinearFilter;
  rampTex.minFilter = THREE.LinearFilter;
  rampTex.needsUpdate = true;
  return rampTex;
}

const vertexShader = /* glsl */ `
attribute vec2 aPerp;
attribute float aOff;
attribute float aEdge;
attribute float aBaseW;
attribute float aPt;
uniform sampler2D uAdj;
uniform float uPull;
uniform float uWidthScale;
varying float vOff;
uniform float uPxScale;
uniform float uMinPx;
uniform sampler2D uVc;
uniform sampler2D uRamp;
uniform float uTexW;
uniform float uRampMax;
uniform float uHighlight;
varying vec3 vColor;
varying float vFogDepth;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  float dist = distance(wp.xyz, cameraPosition);
  float w = max(aBaseW * uWidthScale, uMinPx * dist * uPxScale);
  wp.xz += aPerp * (aOff * w);
  int pi = int(aPt + 0.5);
  wp.y += texelFetch(uAdj, ivec2(pi - (pi / ${ADJ_TEX_W}) * ${ADJ_TEX_W}, pi / ${ADJ_TEX_W}), 0).r;
  wp.y += 0.8 + dist * 0.0025;
  vOff = aOff;
  int e = int(aEdge + 0.5);
  int tw = int(uTexW);
  float vc = texelFetch(uVc, ivec2(e - (e / tw) * tw, e / tw), 0).r;
  vColor = texture(uRamp, vec2(clamp(vc / uRampMax, 0.0, 1.0), 0.5)).rgb;
  if (abs(aEdge - uHighlight) < 0.5) vColor = vec3(1.0, 1.0, 1.0);
  vec4 mv = viewMatrix * wp;
  // photoreal: pull toward the camera so ribbons are not swallowed where the
  // photo mesh sits a little above our DEM (big buildings still occlude them)
  float pull = uPull * (1.0 + dist * 0.002);
  mv.xyz *= max(0.05, 1.0 - pull / max(length(mv.xyz), 1.0));
  vFogDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

// Ramp colors are display (sRGB) values; convert to linear and treat them as
// slightly emissive so they read clearly after ACES tone mapping (both when
// rendering directly and through the post-processing chain).
const fragmentShader = /* glsl */ `
#include <common>
uniform vec3 fogColor;
uniform float fogDensity;
uniform float uOpacity;
uniform float uBright;
uniform float uSoft;
varying vec3 vColor;
varying float vFogDepth;
varying float vOff;
void main() {
  float f = 1.0 - exp(-fogDensity * fogDensity * vFogDepth * vFogDepth);
  vec3 lin = pow(max(vColor, vec3(0.0)), vec3(2.2)) * uBright;
  // soft glowing ribbon (photoreal): bright core, feathered edges
  float core = smoothstep(1.0, 0.45, vOff) * smoothstep(0.0, 0.35, vOff);
  float alpha = mix(uOpacity, uOpacity * (0.25 + 0.75 * core), uSoft);
  lin *= 1.0 + 0.5 * uSoft * core;
  gl_FragColor = vec4(mix(lin, fogColor, f * 0.6), alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

export class RoadOverlay {
  readonly mesh: THREE.Mesh;
  private vcData: Float32Array;
  private vcTex: THREE.DataTexture;
  readonly material: THREE.ShaderMaterial;

  constructor(
    geometry: THREE.BufferGeometry,
    private nEdges: number,
    adj: THREE.DataTexture,
  ) {
    const h = Math.max(1, Math.ceil(nEdges / TEX_W));
    this.vcData = new Float32Array(TEX_W * h);
    this.vcTex = new THREE.DataTexture(this.vcData, TEX_W, h, THREE.RedFormat, THREE.FloatType);
    this.vcTex.magFilter = THREE.NearestFilter;
    this.vcTex.minFilter = THREE.NearestFilter;
    this.vcTex.needsUpdate = true;
    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: THREE.UniformsUtils.merge([
        THREE.UniformsLib.fog,
        {
          uPxScale: { value: 0.001 },
          uMinPx: { value: 2.2 },
          uVc: { value: null },
          uRamp: { value: null },
          uTexW: { value: TEX_W },
          uRampMax: { value: RAMP_MAX },
          uHighlight: { value: -1 },
          uOpacity: { value: 1 },
          uBright: { value: 2.2 },
          uAdj: { value: null },
          uPull: { value: 0 },
          uWidthScale: { value: 1 },
          uSoft: { value: 0 },
        },
      ]),
      fog: true,
      polygonOffset: true,
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -8,
      side: THREE.DoubleSide,
    });
    // textures must be assigned after merge (merge clones uniform values)
    this.material.uniforms['uVc']!.value = this.vcTex;
    this.material.uniforms['uRamp']!.value = rampTexture();
    this.material.uniforms['uAdj']!.value = adj;
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.renderOrder = 2;
    this.mesh.frustumCulled = false;
    this.mesh.name = 'congestion-overlay';
  }

  /** Blend two bins of a bin-major v/c array into the texture. */
  setFromBins(edgeVc: Float32Array, b0: number, b1: number, w: number): void {
    const n = this.nEdges;
    const o0 = b0 * n;
    const o1 = b1 * n;
    const d = this.vcData;
    for (let e = 0; e < n; e++) {
      const a = edgeVc[o0 + e] ?? 0;
      const b = edgeVc[o1 + e] ?? 0;
      d[e] = a + (b - a) * w;
    }
    this.vcTex.needsUpdate = true;
  }

  /** 'open': opaque ribbons over our roads; 'photoreal': thin, translucent, glowing, pulled toward the camera. */
  setStyle(style: 'open' | 'photoreal'): void {
    const u = this.material.uniforms;
    const pr = style === 'photoreal';
    u['uOpacity']!.value = pr ? 0.82 : 1;
    u['uSoft']!.value = pr ? 1 : 0;
    u['uPull']!.value = pr ? 2.5 : 0;
    u['uWidthScale']!.value = pr ? 0.62 : 1;
    u['uBright']!.value = pr ? 2.6 : 2.2;
    if (this.material.transparent !== pr) {
      this.material.transparent = pr;
      this.material.depthWrite = !pr;
      this.material.needsUpdate = true;
    }
  }

  setHighlight(edge: number | null): void {
    this.material.uniforms['uHighlight']!.value = edge ?? -1;
  }

  /** Update the pixel-size scale from the camera and viewport height. */
  setViewport(camera: THREE.PerspectiveCamera, heightPx: number): void {
    this.material.uniforms['uPxScale']!.value = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / Math.max(1, heightPx);
  }

  dispose(): void {
    this.vcTex.dispose();
    this.material.dispose();
    // geometry is shared; disposed by its owner
  }
}
