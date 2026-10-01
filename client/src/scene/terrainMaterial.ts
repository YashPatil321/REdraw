/**
 * Terrain material: imagery albedo for the macro color, blended at close range
 * with tileable PBR ground cells from the ground atlas (lawn, dry grass,
 * chaparral, coastal sage, bare dirt, decomposed granite) chosen per pixel by
 * a landcover splat mask when the pipeline provides one, else classified from
 * the imagery color. Two detail scales with a rotated second layer break up
 * tiling; steep slopes switch to triplanar dirt; detail normal maps perturb
 * the lighting. Everything fades to the imagery with distance.
 */

import * as THREE from 'three';
import { ATLAS_GLSL, cellRect, type MaterialLibrary } from './materials';

/** Ground cells used by the terrain, in uniform order. */
export const TERRAIN_LAYERS = ['grass_lawn', 'grass_patchy', 'chaparral', 'coastal_sage', 'bare_dirt', 'decomposed_granite', 'asphalt_parking', 'mulch', 'pool_water'] as const;

export interface TerrainShared {
  uMinH: { value: number };
  uMaxH: { value: number };
}

const FRAG_PARS = /* glsl */ `
varying vec3 vTW;
uniform sampler2D tGA;
uniform sampler2D tGN;
uniform sampler2D tGO;
uniform sampler2D tSplat;
uniform sampler2D tSplatB;
uniform float uHasSplat;
uniform vec4 uGR[9];
uniform vec2 uGS[9];
uniform vec3 uGMean[9];
uniform float uGPx;
uniform float uDetail;
uniform float uDetailFar;
${ATLAS_GLSL}
vec3 rdNT;
float rdRough;
float rdAO;

vec3 rdLayer(int i, vec2 xz, out vec3 nrm, out vec3 orm, float scale, float rot) {
  vec2 p = xz;
  if (rot > 0.5) p = vec2(0.8 * xz.x - 0.6 * xz.y, 0.6 * xz.x + 0.8 * xz.y) + 37.0;
  vec2 uv = vec2(p.x, -p.y) / (uGS[i] * scale);
  vec2 gx = dFdx(uv);
  vec2 gy = dFdy(uv);
  vec4 r = uGR[i];
  float f = atlasFar(gx, gy, r, uGPx);
  vec3 c = f < 0.999 ? mix(atlasSample(tGA, r, uv, gx, gy).rgb, uGMean[i], f) : uGMean[i];
  nrm = f < 0.999 ? mix(atlasSample(tGN, r, uv, gx, gy).xyz * 2.0 - 1.0, vec3(0.0, 0.0, 1.0), f) : vec3(0.0, 0.0, 1.0);
  orm = f < 0.999 ? mix(atlasSample(tGO, r, uv, gx, gy).rgb, vec3(1.0, 0.85, 0.0), f) : vec3(1.0, 0.85, 0.0);
  return c;
}
float rdLum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float rdN2(vec2 p) {
  vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(rdHash21(i), rdHash21(i + vec2(1, 0)), f.x), mix(rdHash21(i + vec2(0, 1)), rdHash21(i + vec2(1, 1)), f.x), f.y);
}
`;

const FRAG_MAIN = /* glsl */ `
#include <map_fragment>
{
  vec3 macro = diffuseColor.rgb;
  rdNT = vec3(0.0, 0.0, 1.0);
  rdRough = 0.92;
  rdAO = 1.0;
  float camD = length(vTW - cameraPosition);
  float near = uDetail * (1.0 - smoothstep(uDetailFar * 0.45, uDetailFar, camD));
  if (near > 0.001) {
    // landcover weights: splat mask (R lawn, G scrub, B bare, A paved) or imagery classification
    vec3 cs = pow(max(macro, vec3(0.0)), vec3(1.0 / 2.2));
    float lum = rdLum(cs);
    float green = cs.g - max(cs.r, cs.b);
    // weights: lawn, scrub, bare, paved, water, canopy
    float wv[6];
    if (uHasSplat > 0.5) {
      vec3 sa = texture2D(tSplat, vMapUv).rgb;
      vec3 sb = texture2D(tSplatB, vMapUv).rgb;
      wv[0] = sa.r; wv[1] = sa.g; wv[2] = sa.b; wv[3] = sb.r; wv[4] = sb.g; wv[5] = sb.b;
    } else {
      float lawn = smoothstep(-0.005, 0.045, green) * smoothstep(0.12, 0.26, cs.g);
      float scrub = (1.0 - lawn) * (1.0 - smoothstep(0.22, 0.36, lum));
      float bare = max(0.0, 1.0 - lawn - scrub);
      wv[0] = lawn; wv[1] = scrub; wv[2] = bare; wv[3] = 0.0; wv[4] = 0.0; wv[5] = 0.0;
    }
    // macro noise varies the species of each class
    float mn = rdN2(vTW.xz * 0.013) * 0.6 + rdN2(vTW.xz * 0.051) * 0.4;
    // built-up surroundings (blurred paved share): "scrub" and "bare" there are
    // landscaping (bark mulch, decomposed granite, patchy turf), not wild chaparral
    float urban = 0.0;
    if (uHasSplat > 0.5) urban = smoothstep(0.12, 0.3, texture2D(tSplatB, vMapUv, 4.5).r) + (rdN2(vTW.xz * 0.08) - 0.5) * 0.3;
    int li[6];
    li[0] = mn > 0.62 ? 1 : 0;
    li[1] = urban > 0.5 ? (mn > 0.45 ? 7 : 1) : (mn > 0.5 ? 3 : 2);
    li[2] = urban > 0.5 ? 5 : (mn > 0.55 ? 5 : 4);
    li[3] = 6;
    li[4] = 8;
    li[5] = 7;
    // two strongest classes
    float m0 = -1.0; int k0 = 0; float m1 = -1.0; int k1 = 0;
    for (int q = 0; q < 6; q++) {
      float x = wv[q];
      if (x > m0) { m1 = m0; k1 = k0; m0 = x; k0 = q; }
      else if (x > m1) { m1 = x; k1 = q; }
    }
    int i0 = li[k0];
    int i1 = li[k1];
    float a0; float a1;
    float sum = max(m0 + m1, 1e-4);
    // noisy height-blend transition between the two layers
    float edge = clamp(m1 / sum + (rdN2(vTW.xz * 0.9) - 0.5) * 0.35, 0.0, 1.0);
    a1 = smoothstep(0.35, 0.65, edge);
    a0 = 1.0 - a1;
    vec3 n0; vec3 o0; vec3 n1; vec3 o1; vec3 nb; vec3 ob;
    vec3 c0 = rdLayer(i0, vTW.xz, n0, o0, 1.0, 0.0);
    vec3 c1 = a1 > 0.01 ? rdLayer(i1, vTW.xz, n1, o1, 1.0, 0.0) : c0;
    if (a1 <= 0.01) { n1 = n0; o1 = o0; }
    vec3 det = c0 * a0 + c1 * a1;
    vec3 dn = n0 * a0 + n1 * a1;
    vec3 dorm = o0 * a0 + o1 * a1;
    #if RD_TERRAIN_DETAIL > 1
      // second, larger rotated scale of the dominant layer (anti-tiling)
      vec3 c2 = rdLayer(i0, vTW.xz, nb, ob, 3.7, 1.0);
      det = mix(det, det * (c2 / max(uGMean[i0], vec3(1e-3))), 0.45);
    #endif
    // keep the imagery's local hue / brightness, add the texture's structure
    vec3 mean = uGMean[i0] * a0 + uGMean[i1] * a1;
    vec3 modulated = macro * det / max(mean, vec3(1e-3));
    vec3 pure = det * (0.75 + 0.5 * clamp(rdLum(macro) / max(rdLum(mean), 1e-3), 0.6, 1.6));
    vec3 c = mix(modulated, pure, 0.55);
    diffuseColor.rgb = mix(macro, c, near);
    rdNT = normalize(mix(vec3(0.0, 0.0, 1.0), dn, near));
    rdRough = mix(0.92, dorm.g, near);
    rdAO = mix(1.0, dorm.r, near * 0.8);
  } else {
    // far: subtle macro variation so imagery pixels do not read as flat blobs
    float mn = rdN2(vTW.xz * 0.02);
    diffuseColor.rgb = macro * (0.94 + 0.12 * mn);
  }
}
`;

const NORMAL_FRAG = /* glsl */ `
#include <normal_fragment_maps>
{
  mat3 rdTbn = cotangentFrame(normal, -vViewPosition, vec2(vTW.x, -vTW.z));
  normal = normalize(rdTbn * normalize(vec3(rdNT.xy * 1.2, rdNT.z)));
}
`;

/**
 * PBR terrain material for one tile (imagery `map`). Shares atlas textures and
 * uniforms across tiles; `splat` is the tile's landcover mask when present.
 */
export function terrainMaterial(
  lib: MaterialLibrary | null,
  map: THREE.Texture | null,
  splat: [THREE.Texture, THREE.Texture] | null,
  opts: { detail: number; standard: boolean },
): THREE.Material {
  const ground = lib?.atlas('ground') ?? null;
  const useAtlas = !!(ground && ground.normal && ground.orm && opts.detail > 0 && map);
  const params = { map, color: 0xffffff };
  const mat = opts.standard ? new THREE.MeshStandardMaterial({ ...params, roughness: 0.92, metalness: 0 }) : new THREE.MeshLambertMaterial(params);
  if (map) {
    map.colorSpace = THREE.SRGBColorSpace;
    map.anisotropy = 8;
  }
  if (!useAtlas) return mat;
  const gr: THREE.Vector4[] = [];
  const gs: THREE.Vector2[] = [];
  const gm: THREE.Vector3[] = [];
  for (const name of TERRAIN_LAYERS) {
    const c = lib!.cell('ground', name);
    gr.push(cellRect(c));
    const ws = c?.world_size_m ?? [4, 4];
    gs.push(new THREE.Vector2(ws[0], ws[1]));
    const m = c?.mean_albedo_linear ?? [0.2, 0.2, 0.15];
    gm.push(new THREE.Vector3(m[0], m[1], m[2]));
  }
  const uniforms = {
    tGA: { value: ground!.albedo },
    tGN: { value: ground!.normal },
    tGO: { value: ground!.orm },
    tSplat: { value: splat?.[0] ?? null },
    tSplatB: { value: splat?.[1] ?? null },
    uHasSplat: { value: splat ? 1 : 0 },
    uGR: { value: gr },
    uGS: { value: gs },
    uGMean: { value: gm },
    uGPx: { value: ground!.info.size_px?.[0] ?? 2048 },
    uDetail: { value: 1 },
    uDetailFar: { value: 700 },
  };
  const detail = opts.detail;
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vTW;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvTW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n#define RD_TERRAIN_DETAIL ${detail}\n${FRAG_PARS}`)
      .replace('#include <map_fragment>', FRAG_MAIN);
    if (opts.standard) {
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = rdRough;')
        .replace('#include <aomap_fragment>', '#include <aomap_fragment>\nreflectedLight.indirectDiffuse *= rdAO;\nreflectedLight.directDiffuse *= mix(1.0, rdAO, 0.5);');
      if (detail > 1) shader.fragmentShader = shader.fragmentShader.replace('#include <normal_fragment_maps>', NORMAL_FRAG);
    }
  };
  mat.customProgramCacheKey = () => `rd-terrain-${detail}-${opts.standard ? 's' : 'l'}-${splat ? 1 : 0}`;
  (mat as THREE.Material & { rdUniforms?: typeof uniforms }).rdUniforms = uniforms;
  return mat;
}
