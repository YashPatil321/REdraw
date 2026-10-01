/**
 * Materials for the pipeline's street-level ground meshes (HD build):
 * - `_MAT` 6 "ground": road asphalt, sidewalks, driveways, curbs, medians,
 *   pools; `_VARIANT` = ground atlas cell index, TEXCOORD_0 in cell units
 *   (one unit = the cell's world_size_m), COLOR_0 = color-only fallback.
 * - `_MAT` 8 "markings": `_VARIANT` = ground_markings.png column, u across the
 *   line (0..1 over world_width_m), v along (1 = period_m); alpha = worn paint.
 */

import * as THREE from 'three';
import { ATLAS_GLSL, cellRect, type MaterialLibrary } from './materials';

const MAX_CELLS = 24;
const MAX_COLS = 12;

const GROUND_VERT_PARS = /* glsl */ `
uniform float uPull;
attribute float aVariant;
flat varying float vVar;
varying vec2 vGUv;
varying vec3 vGW;
`;
const GROUND_VERT = /* glsl */ `
vVar = aVariant;
vGUv = uv;
vGW = (modelMatrix * vec4(transformed, 1.0)).xyz;
`;

/**
 * Depth pull: the street meshes are draped a few cm over the pipeline's
 * render surface, but the drawn terrain (RTIN, ~0.3 m max error) can poke
 * through. Moving vertices along the view ray changes only their depth (not
 * their screen position), so the road always wins against nearby terrain.
 */
const PULL_VERT = /* glsl */ `
#include <project_vertex>
{
  float rdD = length(mvPosition.xyz);
  mvPosition.xyz -= normalize(mvPosition.xyz) * min(uPull * (1.0 + rdD * 0.02), rdD * 0.5);
  gl_Position = projectionMatrix * mvPosition;
}
`;

const GROUND_FRAG_PARS = /* glsl */ `
flat varying float vVar;
varying vec2 vGUv;
varying vec3 vGW;
uniform sampler2D tGA;
uniform sampler2D tGN;
uniform sampler2D tGO;
uniform vec4 uCR[${MAX_CELLS}];
uniform vec3 uCMean[${MAX_CELLS}];
uniform float uCGain[${MAX_CELLS}];
uniform float uGPx;
${ATLAS_GLSL}
vec3 rdNT;
float rdRough;
float rdAO;
float rdN2g(vec2 p) {
  vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(rdHash21(i), rdHash21(i + vec2(1, 0)), f.x), mix(rdHash21(i + vec2(0, 1)), rdHash21(i + vec2(1, 1)), f.x), f.y);
}
`;

const GROUND_FRAG_MAIN = /* glsl */ `
{
  int ci = clamp(int(vVar + 0.5), 0, ${MAX_CELLS - 1});
  vec4 r = uCR[ci];
  vec2 uv = vGUv;
  vec2 gx = dFdx(uv);
  vec2 gy = dFdy(uv);
  float f = atlasFar(gx, gy, r, uGPx);
  vec3 mean = uCMean[ci];
  vec3 c = f < 0.999 ? mix(atlasSample(tGA, r, uv, gx, gy).rgb, mean, f) : mean;
  // large-scale wear / tone variation so long roads do not repeat
  float mn = rdN2g(vGW.xz * 0.021) * 0.6 + rdN2g(vGW.xz * 0.0043 + 7.0) * 0.4;
  c *= (0.9 + 0.2 * mn) * uCGain[ci];
  diffuseColor.rgb = c;
  rdNT = f < 0.999 ? mix(atlasSample(tGN, r, uv, gx, gy).xyz * 2.0 - 1.0, vec3(0.0, 0.0, 1.0), f) : vec3(0.0, 0.0, 1.0);
  vec3 orm = f < 0.999 ? atlasSample(tGO, r, uv, gx, gy).rgb : vec3(1.0, 0.85, 0.0);
  // dry pavement: never mirror-like (the sky would read as a wet road at grazing angles)
  rdRough = mix(max(orm.g, 0.62), 0.85, f);
  rdAO = mix(orm.r, 1.0, f);
}
`;

const NORMAL_FRAG = /* glsl */ `
#include <normal_fragment_maps>
{
  mat3 rdTbn = cotangentFrame(normal, -vViewPosition, vGUv);
  normal = normalize(rdTbn * normalize(rdNT));
}
`;

/**
 * Look adjustment for asphalt cells: sun-bleached Southern California streets
 * read mid-grey in photos (aged asphalt albedo ~0.1-0.15), darker than the
 * atlas cells. Returns the multiplier that lifts a cell's mean to that target.
 */
export function asphaltGain(name: string, meanLinear: number): number {
  const target = /fresh/.test(name) ? 0.075 : /parking/.test(name) ? 0.105 : /asphalt/.test(name) ? 0.125 : 0;
  if (!target || meanLinear <= 0) return 1;
  return Math.min(3.5, Math.max(1, target / meanLinear));
}

/** Ground atlas material for `_MAT` 6 meshes (needs `aVariant`: alias of `_variant`). */
export function groundMaterial(lib: MaterialLibrary, opts: { polygonOffset: number; normalMaps: boolean }): THREE.Material {
  const g = lib.atlas('ground')!;
  const cells = g.info.cells;
  const cr: THREE.Vector4[] = [];
  const cm: THREE.Vector3[] = [];
  const gain: number[] = [];
  for (let i = 0; i < MAX_CELLS; i++) {
    const c = cells.find((x) => x.index === i) ?? cells[Math.min(i, cells.length - 1)];
    cr.push(cellRect(c));
    const m = c?.mean_albedo_linear ?? [0.2, 0.2, 0.2];
    cm.push(new THREE.Vector3(m[0], m[1], m[2]));
    gain.push(asphaltGain(c?.name ?? '', (m[0] + m[1] + m[2]) / 3));
  }
  const mat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.85,
    metalness: 0,
    polygonOffset: true,
    polygonOffsetFactor: -opts.polygonOffset,
    polygonOffsetUnits: -opts.polygonOffset * 2,
  });
  const uniforms = {
    tGA: { value: g.albedo },
    tGN: { value: g.normal },
    tGO: { value: g.orm },
    uCR: { value: cr },
    uCMean: { value: cm },
    uCGain: { value: gain },
    uPull: { value: 0.12 },
    uGPx: { value: g.info.size_px?.[0] ?? 2048 },
  };
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${GROUND_VERT_PARS}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${GROUND_VERT}`)
      .replace('#include <project_vertex>', PULL_VERT);
    let fs = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${GROUND_FRAG_PARS}`)
      .replace('#include <color_fragment>', GROUND_FRAG_MAIN)
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = rdRough;')
      .replace('#include <aomap_fragment>', '#include <aomap_fragment>\nreflectedLight.indirectDiffuse *= rdAO;\nreflectedLight.directDiffuse *= mix(1.0, rdAO, 0.5);');
    if (opts.normalMaps) fs = fs.replace('#include <normal_fragment_maps>', NORMAL_FRAG);
    shader.fragmentShader = fs;
  };
  mat.customProgramCacheKey = () => `rd-ground-${opts.normalMaps ? 1 : 0}`;
  return mat;
}

const MARK_FRAG_PARS = /* glsl */ `
flat varying float vVar;
varying vec2 vGUv;
varying vec3 vGW;
uniform sampler2D tMark;
uniform vec4 uMC[${MAX_COLS}];
float rdMH(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
`;

const MARK_FRAG_MAIN = /* glsl */ `
{
  int ci = clamp(int(vVar + 0.5), 0, ${MAX_COLS - 1});
  vec4 r = uMC[ci];
  vec2 uv = vec2(clamp(vGUv.x, 0.0, 1.0), vGUv.y);
  vec2 a = vec2(r.x + uv.x * r.z, r.y + fract(uv.y) * r.w);
  vec2 gx = dFdx(vGUv) * r.zw;
  vec2 gy = dFdy(vGUv) * r.zw;
  vec4 t = textureGrad(tMark, a, gx, gy);
  diffuseColor.rgb = t.rgb * 0.92;
  diffuseColor.a = t.a;
  // fade sub-pixel lines into the asphalt (no shimmer from far away)
  float px = max(length(fwidth(vGUv.x)), 1e-5);
  diffuseColor.a *= clamp(0.35 / px, 0.0, 1.0);
  if (diffuseColor.a < 0.02) discard;
}
`;

/** Lane marking decals (`_MAT` 8), alpha blended over the asphalt. */
export function markingsMaterial(lib: MaterialLibrary): THREE.Material | null {
  const mk = lib.markings;
  if (!mk) return null;
  const mc: THREE.Vector4[] = [];
  for (let i = 0; i < MAX_COLS; i++) {
    const c = mk.columns[Math.min(i, mk.columns.length - 1)];
    const uv = c?.uv ?? [0, 0, 1, 1];
    mc.push(new THREE.Vector4(uv[0], uv[1], uv[2] - uv[0], uv[3] - uv[1]));
  }
  const mat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.6,
    metalness: 0,
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -6,
    polygonOffsetUnits: -12,
  });
  const uniforms = { tMark: { value: mk.tex }, uMC: { value: mc }, uPull: { value: 0.16 } };
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${GROUND_VERT_PARS}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${GROUND_VERT}`)
      .replace('#include <project_vertex>', PULL_VERT);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${MARK_FRAG_PARS}`)
      .replace('#include <color_fragment>', MARK_FRAG_MAIN);
  };
  mat.customProgramCacheKey = () => 'rd-markings';
  return mat;
}

/** Give a geometry the `aVariant` attribute the ground shaders read (alias of `_variant`, or zeros). */
export function ensureVariant(g: THREE.BufferGeometry): void {
  const v = g.getAttribute('_variant');
  if (v) g.setAttribute('aVariant', v);
  else g.setAttribute('aVariant', new THREE.BufferAttribute(new Float32Array(g.getAttribute('position').count), 1));
}
