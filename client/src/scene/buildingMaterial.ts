/**
 * Building materials driven by the PBR atlases (materials.ts) and the
 * per-vertex attributes from buildingPrep.ts:
 *
 * walls: stucco cells tinted by COLOR_0 (texel / cell mean x tint), normal
 *   mapped; every 3 m x 3 m floor-bay picks a facade_openings cell with the
 *   manifest's bay grammar (garage + entry on the street wall of houses,
 *   storefronts on commercial ground floors, ribbon windows on schools),
 *   hashed per building / wall / bay so it is stable. Window glass reflects
 *   the sky environment (low roughness PBR) over a parallax "interior mapped"
 *   room (back wall, floor, ceiling, side walls, curtains); some windows glow
 *   warm before sunrise.
 * roofs: S-tile / flat concrete tile / membrane cells with normal + ORM maps.
 *
 * At distance every lookup fades to the cell's mean albedo (no atlas mip
 * bleeding, no moire).
 */

import * as THREE from 'three';
import { ATLAS_GLSL, cellRect, type MaterialLibrary } from './materials';

/** Openings cells in uniform-array order (span cells expanded). */
export const OPENING_SLOTS = [
  'window_slider',
  'window_pair',
  'window_picture',
  'window_small',
  'door_front',
  'door_slider',
  'garage_2car#0',
  'garage_2car#1',
  'garage_3car#0',
  'garage_3car#1',
  'garage_3car#2',
  'window_arched',
  'storefront',
  'storefront_sign',
  'school_window_band',
  'school_door',
] as const;

export const buildingUniforms = {
  /** 0 day .. 1 before dawn: windows glow */
  uNight: { value: 0 },
  /** interior daylight radiance (linear) behind windows */
  uInterior: { value: new THREE.Color(0.25, 0.24, 0.22) },
};

const COMMON_VERT_PARS = /* glsl */ `
attribute vec4 aWall;
attribute vec2 aMatVar;
varying vec2 vFUv;
flat varying vec4 vWall;
flat varying vec2 vMV;
`;
const COMMON_VERT = /* glsl */ `
vFUv = uv;
vWall = aWall;
vMV = aMatVar;
`;

const WALL_FRAG_PARS = /* glsl */ `
varying vec2 vFUv;
flat varying vec4 vWall;
flat varying vec2 vMV;
uniform sampler2D tWA;
uniform sampler2D tWN;
uniform sampler2D tOA;
uniform sampler2D tON;
uniform sampler2D tOO;
uniform sampler2D tOM;
uniform vec4 uWR[8];
uniform vec3 uWMean[8];
uniform vec4 uOR[16];
uniform vec2 uAtlasPx;
uniform float uNight;
uniform vec3 uInterior;
uniform float uInteriors;
${ATLAS_GLSL}
vec3 rdAlbedo;
vec3 rdNT;
float rdRough;
float rdAO;
vec3 rdEmiss;
float rdGlass;

// bay grammar (materials_manifest bay_grammar), returns an opening slot or -1 (plain), -2 curtain glass
int rdChoose(float kind, bool street, float bay, float nb, float fl, float h, float bh) {
  if (kind < 0.5) {
    // house
    if (street && fl < 0.5) {
      bool left = bh < 0.5;
      bool three = nb >= 5.0 && fract(bh * 7.13) < 0.35;
      float g0 = left ? 0.0 : nb - (three ? 3.0 : 2.0);
      float k = bay - g0;
      if (three && k >= 0.0 && k < 3.0) return 8 + int(k);
      if (!three && k >= 0.0 && k < 2.0) return 6 + int(k);
      float door = left ? (three ? 3.0 : 2.0) : g0 - 1.0;
      if (bay == door) return 4;
      return h < 0.45 ? 2 : (h < 0.75 ? 0 : (h < 0.88 ? 11 : -1));
    }
    if (fl < 0.5) return h < 0.3 ? 0 : (h < 0.45 ? 5 : (h < 0.6 ? 1 : -1));
    return h < 0.3 ? 0 : (h < 0.5 ? 1 : (h < 0.58 ? 11 : (h < 0.7 ? 3 : -1)));
  }
  if (kind < 1.5) {
    // apartments
    if (fl < 0.5) return h < 0.35 ? 0 : (h < 0.6 ? 5 : (h < 0.7 ? 4 : -1));
    return h < 0.4 ? 0 : (h < 0.65 ? 5 : (h < 0.75 ? 3 : -1));
  }
  if (kind < 2.5) {
    // commercial
    if (fl < 0.5) return street ? 12 : (h < 0.2 ? 15 : -1);
    if (fl < 1.5 && street) return 13;
    return h < 0.5 ? -2 : (h < 0.7 ? 1 : -1);
  }
  // school
  if (fl < 0.5) return h < 0.7 ? 14 : (h < 0.85 ? 15 : -1);
  return h < 0.8 ? 14 : -1;
}

// parallax interior: a room behind the window (bay-local meters, tangent-space view ray)
vec3 rdRoom(vec2 lp, vec3 vt, float seed) {
  float D = 3.2 + 1.6 * fract(seed * 3.7);
  vec3 p0 = vec3(lp, 0.0);
  vec3 v = vt;
  float tx = v.x > 0.0 ? (3.0 - p0.x) / v.x : -p0.x / min(v.x, -1e-4);
  float ty = v.y > 0.0 ? (2.9 - p0.y) / v.y : (0.0 - p0.y) / min(v.y, -1e-4);
  float tz = -D / min(v.z, -1e-4);
  float t = min(min(tx, ty), tz);
  vec3 hp = p0 + v * t;
  vec3 wallpaper = mix(vec3(0.62, 0.58, 0.5), vec3(0.75, 0.73, 0.68), fract(seed * 11.1));
  vec3 c;
  if (t == tz) {
    c = wallpaper;
    // a picture frame / doorway on the back wall
    vec2 q = hp.xy - vec2(1.5 + (fract(seed * 5.3) - 0.5) * 1.2, 1.6);
    float fr = step(abs(q.x), 0.45) * step(abs(q.y), 0.35);
    c = mix(c, vec3(0.25, 0.22, 0.2) * (0.6 + fract(seed * 17.0)), fr * 0.8);
    float door = step(abs(hp.x - (fract(seed * 2.3) < 0.5 ? 0.5 : 2.5)), 0.45) * step(hp.y, 2.1);
    c = mix(c, vec3(0.12, 0.1, 0.09), door * 0.85);
  } else if (t == ty) {
    if (v.y < 0.0) {
      // floor: wood or carpet, a rug, furniture blob near the back
      vec3 wood = mix(vec3(0.33, 0.21, 0.12), vec3(0.45, 0.32, 0.2), fract(hp.x * 4.0 + seed));
      vec3 carpet = vec3(0.5, 0.47, 0.42);
      c = fract(seed * 13.7) < 0.5 ? wood : carpet;
      float sofa = step(hp.z, -D + 1.0) * step(abs(hp.x - 1.5), 1.0);
      c = mix(c, vec3(0.18, 0.17, 0.16), sofa * 0.9);
    } else {
      // ceiling with a soft lamp
      c = vec3(0.86, 0.85, 0.82) * (0.85 + 0.25 * exp(-dot(hp.xz - vec2(1.5, -D * 0.5), hp.xz - vec2(1.5, -D * 0.5)) * 0.8));
    }
  } else {
    c = wallpaper * 0.82;
  }
  // light falls off into the room
  c *= mix(1.0, 0.45, clamp(-hp.z / D, 0.0, 1.0));
  // curtains / blinds at the window plane
  float cur = fract(seed * 23.1);
  if (cur < 0.45) {
    float side = step(lp.x, 0.25 + 0.2 * fract(seed * 7.0)) + step(3.0 - 0.25 - 0.2 * fract(seed * 9.0), lp.x);
    c = mix(c, mix(vec3(0.78, 0.74, 0.66), vec3(0.55, 0.45, 0.38), fract(seed * 31.0)) * (0.85 + 0.15 * sin(lp.x * 40.0)), clamp(side, 0.0, 1.0) * 0.95);
  } else if (cur < 0.75) {
    float blind = smoothstep(0.35, 0.5, fract(lp.y * 18.0)) * step(2.1 - 1.6 * fract(seed * 3.3), lp.y);
    c = mix(c, vec3(0.82, 0.8, 0.76), blind * 0.92);
  }
  return c;
}
`;

const WALL_FRAG_MAIN = /* glsl */ `
{
  int mat = int(vMV.x + 0.5);
  int vi = int(vMV.y + 0.5);
  vec2 uv = vFUv;
  vec2 gx = dFdx(uv);
  vec2 gy = dFdy(uv);
  vec3 tint = vColor.rgb;
  int wslot = mat == 3 ? 7 : clamp(vi, 0, 6);
  vec4 wr = uWR[wslot];
  float wfar = atlasFar(gx, gy, wr, uAtlasPx.x);
  vec3 wmean = uWMean[wslot];
  vec3 wt = wfar < 0.999 ? mix(atlasSample(tWA, wr, uv, gx, gy).rgb, wmean, wfar) : wmean;
  bool tintable = wslot < 6;
  rdAlbedo = tintable ? tint * wt / max(wmean, vec3(1e-3)) : wt;
  rdNT = wfar < 0.999 ? mix(atlasSample(tWN, wr, uv, gx, gy).xyz * 2.0 - 1.0, vec3(0.0, 0.0, 1.0), wfar) : vec3(0.0, 0.0, 1.0);
  rdRough = 0.88;
  rdAO = 1.0;
  rdEmiss = vec3(0.0);
  rdGlass = 0.0;
  if (mat == 3) {
    float dark = 1.0 - smoothstep(0.08, 0.25, dot(wt, vec3(0.333)));
    rdGlass = dark;
    rdRough = mix(0.5, 0.06, dark);
  }
  vec2 bayUv = uv;
  float len = vWall.x * 0.1;
  float flags = floor(vWall.y / 256.0);
  float key = vWall.y - flags * 256.0;
  bool street = mod(flags, 2.0) > 0.5;
  float kind = floor(flags / 2.0);
  float hgt = vWall.z * 0.1;
  float bh = vWall.w / 65535.0;
  int slot = -1;
  if (mat == 0 || mat == 5) {
    float nb = floor(len / 3.0 + 0.02);
    float margin = (len - nb * 3.0) * 0.5;
    float bc = (uv.x * 3.0 - margin) / 3.0;
    float bay = floor(bc);
    float fl = floor(uv.y);
    bayUv = vec2(bc, uv.y);
    bool fits = bay >= 0.0 && bay < nb && fl * 3.0 + 2.5 <= hgt + 0.05 && len >= 2.9;
    if (mat == 5) slot = 6 + int(clamp(bay, 0.0, 1.0));
    else if (fits) {
      float h = rdHash21(vec2(key * 1.37 + bh * 977.0, bay * 7.13 + fl * 13.7));
      slot = rdChoose(kind, street, bay, nb, fl, h, bh);
    }
  }
  if (slot == -2) {
    // curtain glazing (walls atlas glass cell)
    vec4 gr = uWR[7];
    float gf = atlasFar(gx, gy, gr, uAtlasPx.x);
    vec3 gc = gf < 0.999 ? mix(atlasSample(tWA, gr, uv, gx, gy).rgb, uWMean[7], gf) : uWMean[7];
    float dark = 1.0 - smoothstep(0.08, 0.25, dot(gc, vec3(0.333)));
    rdAlbedo = gc;
    rdGlass = dark;
    rdRough = mix(0.5, 0.06, dark);
  } else if (slot >= 0) {
    vec4 r = uOR[slot];
    float of = atlasFar(gx, gy, r, uAtlasPx.y);
    if (of < 0.999) {
      vec3 m = atlasSample(tOM, r, bayUv, gx, gy).rgb;
      vec3 oc = atlasSample(tOA, r, bayUv, gx, gy).rgb;
      vec3 on = atlasSample(tON, r, bayUv, gx, gy).xyz * 2.0 - 1.0;
      vec3 oo = atlasSample(tOO, r, bayUv, gx, gy).rgb;
      float wallMask = smoothstep(0.8, 0.97, m.r);
      float paint = (1.0 - wallMask) * smoothstep(0.3, 0.42, m.r);
      vec3 openC = mix(oc, oc * mix(vec3(1.0), tint * 1.15, 0.25), paint);
      float k = 1.0 - of;
      rdAlbedo = mix(rdAlbedo, mix(openC, rdAlbedo, wallMask), k);
      rdNT = normalize(mix(rdNT, mix(on, rdNT, wallMask), k));
      rdRough = mix(rdRough, mix(oo.g, 0.88, wallMask), k);
      rdAO = mix(1.0, mix(oo.r, 1.0, wallMask), k);
      float glassMask = m.g * (1.0 - wallMask);
      // glass: dark, glossy pixels of the opening that the mask marks as glazing
      rdGlass = max(rdGlass, glassMask * k);
    } else {
      // far: average darkening of the opening
      rdAlbedo *= 0.82;
    }
  }
  if (rdGlass > 0.01) {
    float seed = rdHash21(vec2(bh * 311.0 + key, floor(bayUv.x) * 3.1 + floor(bayUv.y) * 17.0));
    vec3 room = uInterior * 0.6;
    if (uInteriors > 0.5) {
      // tangent frame (view space) for the parallax room
      vec3 N = normalize(vNormal);
      mat3 tbn = cotangentFrame(N, -vViewPosition, bayUv);
      vec3 V = normalize(-vViewPosition);
      vec3 vt = -vec3(dot(V, tbn[0]), dot(V, tbn[1]), dot(V, N));
      vec2 lp = vec2(fract(bayUv.x), fract(bayUv.y)) * 3.0;
      room = rdRoom(lp, vt, seed) * uInterior;
    }
    float lit = step(0.62, seed) * uNight;
    room += vec3(1.0, 0.72, 0.42) * 1.6 * lit;
    rdEmiss = room * rdGlass;
    rdAlbedo = mix(rdAlbedo, vec3(0.01), rdGlass);
    rdRough = mix(rdRough, 0.04, rdGlass);
  }
  if (mat >= 6) {
    // pipeline / Blender extensions: 6 ground, 7 vertex color only
    rdAlbedo = tint;
    rdNT = vec3(0.0, 0.0, 1.0);
    rdGlass = 0.0;
    rdEmiss = vec3(0.0);
    rdRough = 0.85;
    rdAO = 1.0;
  }
  diffuseColor.rgb = rdAlbedo;
}
`;

const ROOF_FRAG_PARS = /* glsl */ `
varying vec2 vFUv;
flat varying vec4 vWall;
flat varying vec2 vMV;
uniform sampler2D tRA;
uniform sampler2D tRN;
uniform sampler2D tRO;
uniform vec4 uRR[16];
uniform vec3 uRMean[16];
uniform float uRAtlasPx;
${ATLAS_GLSL}
vec3 rdNT;
float rdRough;
float rdAO;
`;

const ROOF_FRAG_MAIN = /* glsl */ `
{
  int mat = int(vMV.x + 0.5);
  int vi = int(vMV.y + 0.5);
  int slot = mat == 2 ? 10 + clamp(vi, 0, 5) : clamp(vi, 0, 9);
  vec2 uv = vFUv;
  vec2 gx = dFdx(uv);
  vec2 gy = dFdy(uv);
  vec4 r = uRR[slot];
  float f = atlasFar(gx, gy, r, uRAtlasPx);
  vec3 mean = uRMean[slot];
  vec3 c = f < 0.999 ? mix(atlasSample(tRA, r, uv, gx, gy).rgb, mean, f) : mean;
  // a little per-house variety toward the tint
  float bh = vWall.w / 65535.0;
  c *= 0.9 + 0.2 * fract(bh * 13.7);
  diffuseColor.rgb = c;
  rdNT = f < 0.999 ? mix(atlasSample(tRN, r, uv, gx, gy).xyz * 2.0 - 1.0, vec3(0.0, 0.0, 1.0), f) : vec3(0.0, 0.0, 1.0);
  vec3 orm = f < 0.999 ? atlasSample(tRO, r, uv, gx, gy).rgb : vec3(1.0, 0.75, 0.0);
  rdRough = mix(orm.g, 0.75, f);
  rdAO = mix(orm.r, 1.0, f);
}
`;

const NORMAL_FRAG = /* glsl */ `
#include <normal_fragment_maps>
{
  mat3 rdTbn = cotangentFrame(normal, -vViewPosition, RD_UV);
  vec3 nt = rdNT;
  nt.xy *= RD_NSTRENGTH;
  normal = normalize(rdTbn * normalize(nt));
}
`;

function texelSize(lib: MaterialLibrary, atlas: string): number {
  return lib.atlas(atlas)?.info.size_px?.[0] ?? 2048;
}

/** Wall material (walls / glass / trim / garage group) using the facade atlases. */
export function buildingWallMaterial(lib: MaterialLibrary, opts: { interiors: boolean; standard: boolean }): THREE.Material {
  const walls = lib.atlas('facade_walls')!;
  const open = lib.atlas('facade_openings')!;
  const wallVariants = lib.variants(0);
  const wr: THREE.Vector4[] = [];
  const wm: THREE.Vector3[] = [];
  for (let i = 0; i < 8; i++) {
    const name = i < 7 ? (wallVariants[i] ?? wallVariants[0]!) : 'glass_curtain';
    const c = lib.cell('facade_walls', name);
    wr.push(cellRect(c));
    const m = c?.mean_albedo_linear ?? [0.5, 0.5, 0.5];
    wm.push(new THREE.Vector3(m[0], m[1], m[2]));
  }
  const or: THREE.Vector4[] = OPENING_SLOTS.map((slot) => {
    const [name, k] = slot.split('#') as [string, string | undefined];
    return cellRect(lib.cell('facade_openings', name), Number(k ?? 0));
  });
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.88, metalness: 0 });
  const uniforms = {
    tWA: { value: walls.albedo },
    tWN: { value: walls.normal },
    tOA: { value: open.albedo },
    tON: { value: open.normal },
    tOO: { value: open.orm },
    tOM: { value: open.mask },
    uWR: { value: wr },
    uWMean: { value: wm },
    uOR: { value: or },
    uAtlasPx: { value: new THREE.Vector2(texelSize(lib, 'facade_walls'), texelSize(lib, 'facade_openings')) },
    uInteriors: { value: opts.interiors ? 1 : 0 },
    ...buildingUniforms,
  };
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${COMMON_VERT_PARS}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${COMMON_VERT}`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${WALL_FRAG_PARS}`)
      .replace('#include <color_fragment>', WALL_FRAG_MAIN)
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = rdRough;')
      .replace('#include <normal_fragment_maps>', NORMAL_FRAG.replace('RD_UV', 'vFUv').replace('RD_NSTRENGTH', '1.0'))
      .replace('#include <aomap_fragment>', '#include <aomap_fragment>\nreflectedLight.indirectDiffuse *= rdAO;\nreflectedLight.indirectSpecular *= mix(rdAO, 1.0, rdGlass);')
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += rdEmiss;');
  };
  mat.customProgramCacheKey = () => 'rd-bwall';
  (mat as THREE.Material & { rdUniforms?: typeof uniforms }).rdUniforms = uniforms;
  return mat;
}

/** Roof material (pitched + flat) using the roofs atlas. */
export function buildingRoofMaterial(lib: MaterialLibrary): THREE.Material {
  const roofs = lib.atlas('roofs')!;
  const tile = lib.variants(1);
  const flat = lib.variants(2);
  const rr: THREE.Vector4[] = [];
  const rm: THREE.Vector3[] = [];
  for (let i = 0; i < 16; i++) {
    const name = i < 10 ? (tile[i] ?? tile[0]!) : (flat[i - 10] ?? flat[0]!);
    const c = lib.cell('roofs', name);
    rr.push(cellRect(c));
    const m = c?.mean_albedo_linear ?? [0.3, 0.2, 0.15];
    rm.push(new THREE.Vector3(m[0], m[1], m[2]));
  }
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.75, metalness: 0 });
  const uniforms = {
    tRA: { value: roofs.albedo },
    tRN: { value: roofs.normal },
    tRO: { value: roofs.orm },
    uRR: { value: rr },
    uRMean: { value: rm },
    uRAtlasPx: { value: texelSize(lib, 'roofs') },
  };
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${COMMON_VERT_PARS}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${COMMON_VERT}`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${ROOF_FRAG_PARS}`)
      .replace('#include <color_fragment>', ROOF_FRAG_MAIN)
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = rdRough;')
      .replace('#include <normal_fragment_maps>', NORMAL_FRAG.replace('RD_UV', 'vFUv').replace('RD_NSTRENGTH', '1.3'))
      .replace('#include <aomap_fragment>', '#include <aomap_fragment>\nreflectedLight.indirectDiffuse *= rdAO;\nreflectedLight.directDiffuse *= mix(1.0, rdAO, 0.6);');
  };
  mat.customProgramCacheKey = () => 'rd-broof';
  return mat;
}

/** Atlases needed by the building shaders are present. */
export function hasBuildingAtlases(lib: MaterialLibrary | null): lib is MaterialLibrary {
  if (!lib) return false;
  const w = lib.atlas('facade_walls');
  const o = lib.atlas('facade_openings');
  const r = lib.atlas('roofs');
  return !!(w?.normal && o?.normal && o.orm && o.mask && r?.normal && r.orm);
}
