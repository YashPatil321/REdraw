/**
 * Procedural fallback materials for worlds built before the PBR atlases
 * (old asset tiles without material ids, or no materials_manifest.json):
 * imagery terrain with a noise detail layer, stucco buildings with
 * procedural windows and tile roofs, procedural asphalt.
 */

import * as THREE from 'three';

/** Terrain: albedo (NAIP or procedural) times a slope / height tint; procedural palette if untextured. */
export function legacyTerrainMaterial(old: { map?: THREE.Texture | null }, uniforms: { uMinH: { value: number }; uMaxH: { value: number } }): THREE.MeshLambertMaterial {
  const mat = new THREE.MeshLambertMaterial({ map: old.map ?? null, color: 0xffffff });
  if (mat.map) {
    mat.map.anisotropy = 8;
    mat.map.colorSpace = THREE.SRGBColorSpace;
  }
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWNrm;')
      .replace(
        '#include <worldpos_vertex>',
        '#include <worldpos_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvWNrm = normalize(mat3(modelMatrix) * objectNormal);',
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        varying vec3 vWPos;
        varying vec3 vWNrm;
        uniform float uMinH;
        uniform float uMaxH;
        float tHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
        float tNoise(vec2 p) {
          vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
          return mix(mix(tHash(i), tHash(i + vec2(1, 0)), f.x), mix(tHash(i + vec2(0, 1)), tHash(i + vec2(1, 1)), f.x), f.y);
        }`,
      )
      .replace(
        '#include <map_fragment>',
        `#include <map_fragment>
        float slope = 1.0 - clamp(normalize(vWNrm).y, 0.0, 1.0);
        float hN = clamp((vWPos.y - uMinH) / max(uMaxH - uMinH, 1.0), 0.0, 1.0);
        float n1 = fract(sin(dot(floor(vWPos.xz / 9.0), vec2(12.9898, 78.233))) * 43758.5453);
        #ifdef USE_MAP
          diffuseColor.rgb *= mix(vec3(1.0), vec3(0.8, 0.76, 0.72), smoothstep(0.2, 0.65, slope));
          diffuseColor.rgb *= 0.95 + 0.1 * hN;
          // close-range detail: the imagery is ~1-10 m per pixel, so add fine
          // grain (grass blades, gravel, dry patches) that fades out with distance
          float camD = length(vWPos - cameraPosition);
          float detail = 1.0 - smoothstep(120.0, 700.0, camD);
          if (detail > 0.0) {
            vec2 q = vWPos.xz;
            float d1 = tNoise(q * 2.3);
            float d2 = tNoise(q * 0.61 + 17.0);
            detail *= clamp(1.0 - fwidth(q.x * 2.3) * 1.2, 0.0, 1.0);
            float lum = dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114));
            float greenish = smoothstep(0.0, 0.05, diffuseColor.g - max(diffuseColor.r, diffuseColor.b) + 0.02);
            vec3 grain = vec3(0.86 + 0.28 * d1) * (0.93 + 0.14 * d2);
            grain = mix(grain, grain * vec3(0.96, 1.04, 0.92), greenish);
            diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * grain, detail * (0.55 + 0.45 * (1.0 - lum)));
          }
        #else
          vec3 dry = vec3(0.60, 0.55, 0.38);
          vec3 scrub = vec3(0.34, 0.40, 0.25);
          vec3 rock = vec3(0.55, 0.50, 0.44);
          vec3 c = mix(dry, scrub, smoothstep(0.25, 0.75, hN) * 0.8 + n1 * 0.2);
          c = mix(c, rock, smoothstep(0.3, 0.7, slope));
          diffuseColor.rgb = c * (0.9 + 0.2 * n1);
        #endif`,
      );
  };
  mat.customProgramCacheKey = () => `terrain-${mat.map ? 1 : 0}`;
  return mat;
}

/**
 * Buildings (real footprints and heights): PBR stucco walls with per-building
 * tint, procedural windows by floor (spacing and size vary per building and
 * per type), darker ground course, Spanish barrel-tile or concrete-tile
 * pitched roofs and gravel / membrane flat roofs. Needs `aBase` / `aTop`
 * (per-building min / max y), computed at load.
 */
export function legacyBuildingMaterial(old: THREE.Material | null, standard: boolean, hasColor: boolean): THREE.Material {
  const oldStd = old as THREE.MeshStandardMaterial | null;
  const params = { color: hasColor ? 0xffffff : (oldStd?.color ?? new THREE.Color(0xd8d0c4)), vertexColors: hasColor };
  const mat: THREE.MeshStandardMaterial | THREE.MeshLambertMaterial = standard
    ? new THREE.MeshStandardMaterial({ ...params, roughness: 0.86, metalness: 0.0 })
    : new THREE.MeshLambertMaterial(params);
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        '#include <common>\nattribute float buildingId;\nattribute float aBase;\nattribute float aTop;\nflat varying float vBid;\nflat varying vec2 vBT;\nvarying vec3 vBWN;\nvarying vec3 vBWP;',
      )
      .replace(
        '#include <beginnormal_vertex>',
        '#include <beginnormal_vertex>\nvBid = buildingId;\nvBWN = normalize(mat3(modelMatrix) * objectNormal);\nvBT = vec2(aBase, aTop);',
      )
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvBWP = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        // flat: per-building constants must not be interpolated (hashing amplifies ULP noise)
        flat varying float vBid;
        flat varying vec2 vBT;
        varying vec3 vBWN;
        varying vec3 vBWP;
        float bHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
        float bNoise(vec2 p) {
          vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
          return mix(mix(bHash(i), bHash(i + vec2(1, 0)), f.x), mix(bHash(i + vec2(0, 1)), bHash(i + vec2(1, 1)), f.x), f.y);
        }
        float rdWin = 0.0;`,
      )
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        float bid = mod(floor(vBid + 0.5), 1009.0);
        float bh = fract(sin(bid * 12.9898 + 1.7) * 43758.5453);
        float bh2 = fract(sin(bid * 78.233 + 4.1) * 24634.6345);
        float bh3 = fract(sin(bid * 39.346 + 2.3) * 11743.123);
        vec3 n = normalize(vBWN);
        float ny = n.y;
        float hgt = max(vBT.y - vBT.x, 0.1);
        float above = vBWP.y - vBT.x;
        // stucco / painted plaster in warm SoCal tones, per-building variation
        vec3 base = diffuseColor.rgb * (0.88 + 0.22 * bh);
        base = mix(base, base * vec3(1.04, 1.0, 0.92), bh3 * 0.6);
        if (ny > 0.35) {
          bool pitched = ny < 0.985;
          vec2 t = normalize(vec2(-n.z, n.x) + 1e-5);
          float u = dot(vBWP.xz, t);
          float v = dot(vBWP.xz, normalize(n.xz + 1e-5));
          if (pitched) {
            // barrel tile (terracotta) or flat concrete tile (slate grey / brown)
            vec3 terra = mix(vec3(0.62, 0.30, 0.20), vec3(0.50, 0.25, 0.18), bh2);
            vec3 conc = mix(vec3(0.36, 0.35, 0.34), vec3(0.45, 0.38, 0.30), bh2);
            bool barrel = bh < 0.72;
            vec3 roof = barrel ? terra : conc;
            float period = barrel ? 0.26 : 0.33;
            // fade sub-pixel tile courses to their average (no moire from afar)
            float aaS = clamp(1.0 - fwidth(u / period) * 1.5, 0.0, 1.0);
            float aaR = clamp(1.0 - fwidth(v / 0.34) * 1.5, 0.0, 1.0);
            float stripes = mix(0.5, 0.5 + 0.5 * sin(u * 6.2831 / period), aaS);
            float rows = mix(0.92, smoothstep(0.0, 0.08, fract(v / 0.34)), aaR);
            roof *= (barrel ? 0.78 + 0.3 * stripes : 0.9 + 0.12 * stripes) * (0.86 + 0.14 * rows);
            roof *= 0.9 + 0.2 * bNoise(vBWP.xz * 0.9 + bid);
            diffuseColor.rgb = roof;
          } else {
            // flat roofs: light membrane or gravel, with HVAC-ish blotches
            vec3 flatc = mix(vec3(0.78, 0.77, 0.74), vec3(0.55, 0.54, 0.52), bh2);
            flatc *= 0.9 + 0.12 * bNoise(vBWP.xz * 2.5) - 0.12 * step(0.93, bNoise(vBWP.xz * 0.25 + bid));
            diffuseColor.rgb = flatc;
          }
        } else {
          vec2 t = normalize(vec2(-n.z, n.x) + 1e-5);
          float u = dot(vBWP.xz, t);
          // plaster grain and slight vertical staining
          base *= 0.94 + 0.08 * bNoise(vec2(u, vBWP.y) * 3.0);
          base *= 0.97 + 0.03 * smoothstep(0.0, hgt, above);
          // floors: houses ~2.9 m, commercial ~3.8 m
          float tall = step(9.0, hgt);
          float floorH = mix(2.9, 3.7, tall);
          float level = above / floorH;
          float cell = mix(3.6 + 2.4 * bh2, 3.0 + 1.2 * bh2, tall);
          vec2 g = vec2(fract(u / cell + bh), fract(level));
          float ww = mix(0.32 + 0.12 * bh3, 0.62, tall);
          float wx = step(abs(g.x - 0.5), ww * 0.5);
          float wy = step(0.32, g.y) * step(g.y, 0.82);
          // no windows in the top 0.6 m (eaves / parapet) or below 0.6 m
          float inside = step(0.6, above) * step(above, hgt - 0.6);
          // some cells blank (closets, garages) on houses
          float blank = (1.0 - tall) * step(0.62, bHash(mod(floor(vec2(u / cell + bh, level)), 512.0) * 0.731 + bid * 0.137));
          // antialias: when a window cell gets smaller than a few pixels, blend to the average facade
          float aaW = clamp(1.0 - max(fwidth(u / cell), fwidth(level)) * 3.0, 0.0, 1.0);
          rdWin = wx * wy * inside * (1.0 - blank) * aaW;
          // window frames slightly lighter than the wall
          float frame = step(abs(g.x - 0.5), ww * 0.5 + 0.06) * step(0.28, g.y) * step(g.y, 0.86) * inside * (1.0 - blank) * aaW;
          vec3 glass = mix(vec3(0.07, 0.09, 0.11), vec3(0.24, 0.3, 0.38), 0.5 + 0.5 * n.x * 0.4 + bh3 * 0.3);
          base = mix(base, vec3(0.92, 0.91, 0.88), (frame - rdWin) * 0.7);
          base = mix(base, glass, rdWin);
          // far away: average window coverage darkens the facade a little instead
          base = mix(base, base * (1.0 - 0.35 * ww * 0.5 * inside), 1.0 - aaW);
          // darker ground course
          base *= mix(0.72, 1.0, smoothstep(0.0, 0.9, above));
          diffuseColor.rgb = base;
        }`,
      );
    if (standard) {
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <roughnessmap_fragment>',
        '#include <roughnessmap_fragment>\nroughnessFactor = clamp(roughnessFactor + (bh2 - 0.5) * 0.2 - (ny > 0.985 ? 0.1 : 0.0) - rdWin * 0.75, 0.08, 1.0);',
      );
    }
  };
  mat.customProgramCacheKey = () => `building2-${standard ? 's' : 'l'}-${hasColor ? 1 : 0}`;
  return mat;
}

/** Per-vertex min / max y of each building (for floors, ground course, eaves). */
export function addBuildingExtents(g: THREE.BufferGeometry): void {
  const id = g.getAttribute('_building_id');
  const p = g.getAttribute('position');
  const n = p.count;
  const base = new Float32Array(n);
  const top = new Float32Array(n);
  if (!id) {
    g.computeBoundingBox();
    base.fill(g.boundingBox!.min.y);
    top.fill(g.boundingBox!.max.y);
  } else {
    const lo = new Map<number, number>();
    const hi = new Map<number, number>();
    for (let i = 0; i < n; i++) {
      const b = Math.round(id.getX(i));
      const y = p.getY(i);
      const a = lo.get(b);
      if (a === undefined || y < a) lo.set(b, y);
      const c = hi.get(b);
      if (c === undefined || y > c) hi.set(b, y);
    }
    for (let i = 0; i < n; i++) {
      const b = Math.round(id.getX(i));
      base[i] = lo.get(b)!;
      top[i] = hi.get(b)!;
    }
  }
  g.setAttribute('aBase', new THREE.BufferAttribute(base, 1));
  g.setAttribute('aTop', new THREE.BufferAttribute(top, 1));
}

/**
 * Roads: procedural asphalt (aggregate grain, patch repairs, oil darkening),
 * lightly tinted by class (arterials paler, older residential darker). The
 * pipeline's painted texture is dropped: markings and sidewalks are drawn
 * from the network by roadDetails.ts with the right lane counts.
 */
export function legacyRoadMaterial(old: { side?: THREE.Side }, hasColor: boolean): THREE.MeshLambertMaterial {
  const mat = new THREE.MeshLambertMaterial({
    color: 0xffffff,
    vertexColors: hasColor,
    side: old.side ?? THREE.FrontSide,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -4,
  });
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vRW;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvRW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        varying vec3 vRW;
        float rdHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
        float rdNoise(vec2 p) {
          vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
          return mix(mix(rdHash(i), rdHash(i + vec2(1, 0)), f.x), mix(rdHash(i + vec2(0, 1)), rdHash(i + vec2(1, 1)), f.x), f.y);
        }`,
      )
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        float cls = dot(diffuseColor.rgb, vec3(0.333));
        vec3 asphalt = vec3(0.155, 0.157, 0.165) * (0.78 + 0.5 * cls);
        float grain = rdNoise(vRW.xz * 3.1) * 0.6 + rdNoise(vRW.xz * 11.0) * 0.4;
        float patches = smoothstep(0.62, 0.7, rdNoise(vRW.xz * 0.045 + 3.7));
        float oil = smoothstep(0.55, 0.9, rdNoise(vRW.xz * 0.23 + 9.1));
        asphalt *= 0.88 + 0.22 * grain;
        asphalt = mix(asphalt, asphalt * 0.72, patches * 0.6);
        asphalt *= 1.0 - 0.12 * oil;
        diffuseColor.rgb = asphalt;`,
      );
  };
  mat.customProgramCacheKey = () => `road-asphalt-${hasColor ? 1 : 0}`;
  return mat;
}
