/**
 * PBR material atlases from blender/ (assets/materials/materials_manifest.json,
 * format "redraw-materials" v1, see blender/README.md and blender/rdlib/matgen.py).
 *
 * Every atlas is a grid of 512 px cells, each a tileable 480 px square with a
 * 16 px wrapped gutter. Shaders look up `rect.xy + fract(uv) * rect.zw` with
 * explicit gradients, and fade to the cell's mean albedo at distance (mip
 * levels past the gutter would bleed into neighbours).
 *
 * Read defensively: a missing or unexpected manifest means "no atlases" and
 * the procedural shaders are used instead.
 */

import * as THREE from 'three';

export type AssetFetcher = (rel: string) => Promise<ArrayBuffer>;

export interface AtlasCellInfo {
  name: string;
  index: number;
  world_size_m: [number, number];
  span_cells?: number;
  tintable?: boolean;
  mean_albedo_linear?: [number, number, number];
  mean_roughness?: number;
  cells: Array<{ uv_inner: [number, number, number, number] }>;
  door_rect_m?: number[][];
}

export interface AtlasInfo {
  name: string;
  size_px: [number, number];
  files: { albedo: string; normal: string; orm: string; mask?: string };
  files_half_res?: { albedo?: string; normal?: string; orm?: string };
  cells: AtlasCellInfo[];
}

export interface MarkingColumn {
  name: string;
  uv: [number, number, number, number];
  world_width_m: number;
  period_m: number;
}

export interface MaterialsManifest {
  format?: string;
  version?: number;
  atlases: Record<string, AtlasInfo>;
  materials?: Record<string, { name?: string; variants?: string[] }>;
  markings?: { file: string; columns: MarkingColumn[] };
}

export interface AtlasTextures {
  info: AtlasInfo;
  albedo: THREE.Texture;
  normal: THREE.Texture | null;
  orm: THREE.Texture | null;
  mask: THREE.Texture | null;
}

/** uv_inner [u0, v0, u1, v1] -> Vector4(u0, v0, u1 - u0, v1 - v0). Pure; unit tested. */
export function cellRect(c: AtlasCellInfo | undefined, k = 0): THREE.Vector4 {
  const r = c?.cells?.[Math.min(k, (c?.cells?.length ?? 1) - 1)]?.uv_inner;
  if (!r) return new THREE.Vector4(0, 0, 1, 1);
  return new THREE.Vector4(r[0], r[1], r[2] - r[0], r[3] - r[1]);
}

/** Basic structural check of a materials manifest. Pure; unit tested. */
export function validManifest(m: unknown): m is MaterialsManifest {
  if (!m || typeof m !== 'object') return false;
  const a = (m as MaterialsManifest).atlases;
  if (!a || typeof a !== 'object') return false;
  return Object.values(a).every((x) => x && typeof x === 'object' && Array.isArray(x.cells) && !!x.files?.albedo);
}

async function decodeImage(buf: ArrayBuffer, mime: string): Promise<ImageBitmap | HTMLImageElement> {
  const blob = new Blob([buf], { type: mime });
  if (typeof createImageBitmap === 'function') {
    // flipY here: the manifest's uv rects assume v up from the image bottom (flipY = true)
    return await createImageBitmap(blob, { imageOrientation: 'flipY', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  }
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function loadTexture(fetchAsset: AssetFetcher, rel: string, srgb: boolean, anisotropy: number): Promise<THREE.Texture> {
  const buf = await fetchAsset(rel);
  const mime = /\.png$/i.test(rel) ? 'image/png' : 'image/jpeg';
  const img = await decodeImage(buf, mime);
  const tex = new THREE.Texture(img as unknown as HTMLImageElement);
  tex.flipY = !(typeof ImageBitmap !== 'undefined' && img instanceof ImageBitmap);
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = anisotropy;
  tex.needsUpdate = true;
  return tex;
}

export class MaterialLibrary {
  readonly atlases = new Map<string, AtlasTextures>();
  markings: { tex: THREE.Texture; columns: MarkingColumn[] } | null = null;

  constructor(readonly manifest: MaterialsManifest) {}

  /** Load the manifest and atlases; null when absent (procedural fallback). */
  static async load(fetchAsset: AssetFetcher, opts: { half?: boolean; anisotropy?: number; base?: string } = {}): Promise<MaterialLibrary | null> {
    const base = opts.base ?? 'materials/';
    let man: unknown;
    try {
      man = JSON.parse(new TextDecoder().decode(await fetchAsset(`${base}materials_manifest.json`)));
    } catch {
      return null;
    }
    if (!validManifest(man)) {
      console.warn('materials_manifest.json not understood; using procedural materials');
      return null;
    }
    const lib = new MaterialLibrary(man);
    const an = opts.anisotropy ?? 8;
    const jobs: Array<Promise<void>> = [];
    for (const [name, info] of Object.entries(man.atlases)) {
      const pick = (k: 'albedo' | 'normal' | 'orm'): string | undefined => (opts.half ? (info.files_half_res?.[k] ?? info.files[k]) : info.files[k]);
      jobs.push(
        (async () => {
          const albedoRel = pick('albedo')!;
          const [albedo, normal, orm, mask] = await Promise.all([
            loadTexture(fetchAsset, base + albedoRel, true, an),
            pick('normal') ? loadTexture(fetchAsset, base + pick('normal')!, false, an).catch(() => null) : Promise.resolve(null),
            pick('orm') ? loadTexture(fetchAsset, base + pick('orm')!, false, an).catch(() => null) : Promise.resolve(null),
            info.files.mask ? loadTexture(fetchAsset, base + info.files.mask, false, an).catch(() => null) : Promise.resolve(null),
          ]);
          lib.atlases.set(name, { info, albedo, normal, orm, mask });
        })().catch((e: unknown) => console.warn(`material atlas ${name} failed to load`, e)),
      );
    }
    if (man.markings?.file && Array.isArray(man.markings.columns)) {
      jobs.push(
        loadTexture(fetchAsset, base + man.markings.file, true, an)
          .then((tex) => {
            tex.wrapT = THREE.RepeatWrapping;
            lib.markings = { tex, columns: man.markings!.columns };
          })
          .catch((e: unknown) => console.warn('lane marking sheet failed to load', e)),
      );
    }
    await Promise.all(jobs);
    if (!lib.atlases.size) return null;
    return lib;
  }

  atlas(name: string): AtlasTextures | null {
    return this.atlases.get(name) ?? null;
  }

  cell(atlas: string, name: string): AtlasCellInfo | undefined {
    return this.atlases.get(atlas)?.info.cells.find((c) => c.name === name);
  }

  /** variant names for a `_MAT` id (manifest `materials`), with built-in defaults. */
  variants(mat: number): string[] {
    const v = this.manifest.materials?.[String(mat)]?.variants;
    if (Array.isArray(v) && v.length) return v;
    return DEFAULT_VARIANTS[mat] ?? [];
  }

  marking(name: string): MarkingColumn | undefined {
    return this.markings?.columns.find((c) => c.name === name);
  }

  dispose(): void {
    for (const a of this.atlases.values()) {
      a.albedo.dispose();
      a.normal?.dispose();
      a.orm?.dispose();
      a.mask?.dispose();
    }
    this.markings?.tex.dispose();
  }
}

const DEFAULT_VARIANTS: Record<number, string[]> = {
  0: ['stucco_smooth', 'stucco_sand', 'stucco_lace', 'stucco_catface', 'stucco_weathered', 'stucco_scored', 'stone_veneer'],
  1: ['s_tile_terracotta', 's_tile_blend', 's_tile_brown', 's_tile_aged', 'barrel_mission', 'flat_tile_brown', 'flat_tile_grey', 'flat_tile_charcoal', 'flat_tile_sandstone', 'solar_panel'],
  2: ['flat_tpo', 'flat_tpo_grime', 'flat_gravel', 'flat_modbit', 'concrete_deck', 'standing_seam'],
  3: ['glass_curtain'],
  4: ['stucco_smooth', 'stone_veneer'],
  5: ['garage_2car', 'garage_3car'],
};

/** Shared GLSL: atlas lookup with explicit gradients, fading to the mean at coarse mips. */
export const ATLAS_GLSL = /* glsl */ `
// rect = (u0, v0, du, dv) of the cell's inner square; uv in cell units (1 = one tile)
vec4 atlasSample(sampler2D t, vec4 rect, vec2 uv, vec2 gx, vec2 gy) {
  vec2 a = rect.xy + fract(uv) * rect.zw;
  return textureGrad(t, a, gx * rect.zw, gy * rect.zw);
}
// 0 = full detail, 1 = texels well below a pixel (use the cell mean instead)
float atlasFar(vec2 gx, vec2 gy, vec4 rect, float atlasPx) {
  float g = max(length(gx * rect.zw), length(gy * rect.zw)) * atlasPx;
  return smoothstep(10.0, 22.0, g);
}
// cotangent frame from screen derivatives (no tangents needed)
mat3 cotangentFrame(vec3 N, vec3 p, vec2 uv) {
  vec3 dp1 = dFdx(p);
  vec3 dp2 = dFdy(p);
  vec2 duv1 = dFdx(uv);
  vec2 duv2 = dFdy(uv);
  vec3 dp2perp = cross(dp2, N);
  vec3 dp1perp = cross(N, dp1);
  vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
  vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
  float invmax = inversesqrt(max(max(dot(T, T), dot(B, B)), 1e-20));
  return mat3(T * invmax, B * invmax, N);
}
float rdHash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float rdHash21(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
`;
