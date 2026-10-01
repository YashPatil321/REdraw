/**
 * Render quality presets (top bar). Targets: low = any GPU (no post, no
 * shadows), medium = 30+ FPS on integrated GPUs at 1080p, high = 60 FPS on a
 * discrete GPU, ultra = screenshots / photo mode.
 */

export type Quality = 'ultra' | 'high' | 'medium' | 'low';

export const QUALITY_ORDER: Quality[] = ['ultra', 'high', 'medium', 'low'];

export type AoLevel = 'off' | 'low' | 'medium' | 'high';

export interface QualitySettings {
  pixelRatioCap: number;
  /** HDR pipeline (atmosphere pass, tone mapping, grading); off = direct render with fog */
  post: boolean;
  /** MSAA samples of the HDR scene target (clamped to the GPU max) */
  msaa: number;
  shadows: boolean;
  shadowMapSize: number;
  /** cascaded shadow map splits */
  cascades: number;
  /** shadows end (fade out) this far from the camera (m) */
  shadowFar: number;
  ao: AoLevel;
  bloom: boolean;
  godRays: boolean;
  grade: boolean;
  /** image based ambient from the sky (PMREM); off = hemisphere light */
  ibl: boolean;
  /** cube face size of the sky render (env + background) */
  skySize: number;
  /** PBR (MeshStandard) buildings / terrain; off = Lambert */
  standardBuildings: boolean;
  /** parallax interior mapping behind windows */
  interiors: boolean;
  /** terrain detail: 0 imagery only, 1 detail albedo, 2 + normal maps / triplanar */
  terrainDetail: 0 | 1 | 2;
  /** grass tufts near the camera at street level */
  grass: number;
  carShadows: boolean;
  /** props (trees, lamps) cast shadows */
  propShadows: boolean;
  buildingDrawDistance: number;
  propDrawDistance: number;
  anisotropy: number;
}

export const QUALITY: Record<Quality, QualitySettings> = {
  ultra: {
    pixelRatioCap: 2,
    post: true,
    msaa: 8,
    shadows: true,
    shadowMapSize: 4096,
    cascades: 4,
    shadowFar: 4000,
    ao: 'high',
    bloom: true,
    godRays: true,
    grade: true,
    ibl: true,
    skySize: 256,
    standardBuildings: true,
    interiors: true,
    terrainDetail: 2,
    grass: 1,
    carShadows: true,
    propShadows: true,
    buildingDrawDistance: 14000,
    propDrawDistance: 3000,
    anisotropy: 16,
  },
  high: {
    pixelRatioCap: 1.5,
    post: true,
    msaa: 4,
    shadows: true,
    shadowMapSize: 2048,
    cascades: 3,
    shadowFar: 2500,
    ao: 'medium',
    bloom: true,
    godRays: true,
    grade: true,
    ibl: true,
    skySize: 192,
    standardBuildings: true,
    interiors: true,
    terrainDetail: 2,
    grass: 0.7,
    carShadows: true,
    propShadows: true,
    buildingDrawDistance: 12000,
    propDrawDistance: 2200,
    anisotropy: 8,
  },
  medium: {
    pixelRatioCap: 1,
    post: true,
    msaa: 4,
    shadows: true,
    shadowMapSize: 2048,
    cascades: 2,
    shadowFar: 1400,
    ao: 'off',
    bloom: true,
    godRays: false,
    grade: true,
    ibl: true,
    skySize: 128,
    standardBuildings: true,
    interiors: false,
    terrainDetail: 1,
    grass: 0.35,
    carShadows: false,
    propShadows: false,
    buildingDrawDistance: 9000,
    propDrawDistance: 1500,
    anisotropy: 4,
  },
  low: {
    pixelRatioCap: 1,
    post: false,
    msaa: 0,
    shadows: false,
    shadowMapSize: 1024,
    cascades: 1,
    shadowFar: 800,
    ao: 'off',
    bloom: false,
    godRays: false,
    grade: false,
    ibl: false,
    skySize: 64,
    standardBuildings: false,
    interiors: false,
    terrainDetail: 0,
    grass: 0,
    carShadows: false,
    propShadows: false,
    buildingDrawDistance: 6000,
    propDrawDistance: 800,
    anisotropy: 2,
  },
};

const KEY = 'redraw-quality';

export function isQuality(v: unknown): v is Quality {
  return v === 'ultra' || v === 'high' || v === 'medium' || v === 'low';
}

/** One step down (frame-rate watchdog); null at the bottom. */
export function lowerQuality(q: Quality): Quality | null {
  const i = QUALITY_ORDER.indexOf(q);
  return i >= 0 && i < QUALITY_ORDER.length - 1 ? QUALITY_ORDER[i + 1]! : null;
}

/** Pick a default from the GPU string; a saved choice or ?quality= wins. */
export function initialQuality(gl: WebGLRenderingContext | WebGL2RenderingContext | null): { q: Quality; pinned: boolean } {
  const fromUrl = typeof location !== 'undefined' ? new URLSearchParams(location.search).get('quality') : null;
  if (isQuality(fromUrl)) return { q: fromUrl, pinned: true };
  try {
    const saved = localStorage.getItem(KEY);
    if (isQuality(saved)) return { q: saved, pinned: true };
  } catch {
    /* storage unavailable */
  }
  const renderer = gpuName(gl);
  if (isSoftwareRenderer(renderer)) return { q: 'low', pinned: false };
  // integrated GPUs start at medium; discrete at high (the watchdog steps down if needed)
  if (isIntegratedGpu(renderer)) return { q: 'medium', pinned: false };
  return { q: 'high', pinned: false };
}

export function gpuName(gl: WebGLRenderingContext | WebGL2RenderingContext | null): string {
  try {
    const ext = gl?.getExtension('WEBGL_debug_renderer_info');
    return ext && gl ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : '';
  } catch {
    return '';
  }
}

export function isSoftwareRenderer(rendererString: string): boolean {
  return /swiftshader|llvmpipe|softpipe|software|microsoft basic render/i.test(rendererString);
}

/** Intel / AMD APU / Apple base / mobile GPUs. Apple M-series Pro/Max count as discrete-class. */
export function isIntegratedGpu(rendererString: string): boolean {
  if (/apple m\d+ (pro|max|ultra)/i.test(rendererString)) return false;
  return /intel|iris|uhd graphics|hd graphics|radeon\(tm\) graphics|radeon graphics|vega \d+ graphics|apple m\d|apple gpu|mali|adreno|powervr/i.test(rendererString);
}

export function saveQuality(q: Quality): void {
  try {
    localStorage.setItem(KEY, q);
  } catch {
    /* ignore */
  }
}
