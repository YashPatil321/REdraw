/**
 * Smooth LOD transitions for instanced props: instead of swapping a tree's
 * full model for its impostor at one distance (a visible pop), both are drawn
 * across a short distance band and each pixel shows one of them, chosen by a
 * stable screen-space dither whose threshold follows the distance. The two
 * layers use complementary tests, so the band reads as a dissolve. The far
 * end of the last layer dissolves out the same way.
 *
 * Rendering parameters only (not real-world numbers).
 */

import * as THREE from 'three';

export const LOD_FADE = {
  /** width of the full <-> impostor band (ending at the switch distance), fraction of that distance */
  band: 0.16,
  /** extra membership margin (m): the instance lists refill only every `refillM` of camera motion */
  margin: 14,
  refillM: 12,
} as const;

export interface FadeBands {
  /** near band: full model fully drawn below n0, gone above n1 */
  n0: number;
  n1: number;
  /** far band of the impostor layer (f1 = 0: no impostor) */
  f0: number;
  f1: number;
}

/** Distance bands for a prop class with full-model distance `near` and impostor distance `far` (0 = none). Pure; unit tested. */
export function fadeBands(near: number, far: number, band: number = LOD_FADE.band): FadeBands {
  // the band ends at the switch distance: full models never draw further out than before
  if (far > near) return { n0: near * (1 - band), n1: near, f0: far * (1 - band * 0.6), f1: far };
  return { n0: near * (1 - band), n1: near, f0: 0, f1: 0 };
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / Math.max(b - a, 1e-6)));
  return t * t * (3 - 2 * t);
}

/** Drawn fraction of the full model and the impostor at distance d (what the dither shows). Pure; unit tested. */
export function layerWeights(d: number, b: FadeBands): { full: number; lod: number } {
  const g = smoothstep(b.n0, b.n1, d);
  const full = 1 - g;
  const lod = b.f1 > 0 ? Math.max(0, g - smoothstep(b.f0, b.f1, d)) : 0;
  return { full, lod };
}

export interface FadeUniforms {
  /** n0, n1, f0, f1 */
  uRdFade: { value: THREE.Vector4 };
}

export function makeFadeUniforms(): FadeUniforms {
  return { uRdFade: { value: new THREE.Vector4(1e6, 1e6 + 1, 0, 0) } };
}

/**
 * Dither the material's instances by camera distance. `layer`: 'full' keeps
 * the near side of the band, 'lod' the far side (and fades out at f0..f1).
 * Chains any existing onBeforeCompile and extends the program cache key.
 */
export function addLodFade(mat: THREE.Material, uniforms: FadeUniforms, layer: 'full' | 'lod'): void {
  const ud = mat.userData as { rdFade?: boolean };
  if (ud.rdFade) return;
  ud.rdFade = true;
  const prev = mat.onBeforeCompile;
  const isDefaultKey = mat.customProgramCacheKey === THREE.Material.prototype.customProgramCacheKey;
  const prevKey = isDefaultKey ? prev.toString() : mat.customProgramCacheKey.call(mat);
  const side = layer === 'full' ? '0.0' : '1.0';
  mat.onBeforeCompile = (shader, r) => {
    prev.call(mat, shader, r);
    shader.uniforms['uRdFade'] = uniforms.uRdFade;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nuniform vec4 uRdFade;\nvarying vec2 vRdFade;')
      .replace(
        '#include <project_vertex>',
        `#include <project_vertex>
{
  #ifdef USE_INSTANCING
    vec3 rdO = (modelMatrix * vec4(instanceMatrix[3].xyz, 1.0)).xyz;
  #else
    vec3 rdO = (modelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  #endif
  float rdD = distance(rdO, cameraPosition);
  vRdFade = vec2(smoothstep(uRdFade.x, uRdFade.y, rdD), uRdFade.w > 0.0 ? smoothstep(uRdFade.z, uRdFade.w, rdD) : 0.0);
}`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vRdFade;')
      .replace(
        '#include <clipping_planes_fragment>',
        `#include <clipping_planes_fragment>
{
  // stable interleaved-gradient dither: complementary between the two LOD layers
  float rdN = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  if (${side} < 0.5) { if (rdN < vRdFade.x) discard; }
  else if (rdN >= vRdFade.x || rdN < vRdFade.y) discard;
}`,
      );
  };
  mat.customProgramCacheKey = () => `${prevKey}-rdfade${side}`;
  mat.needsUpdate = true;
}
