/** Render quality presets (top bar toggle). Low keeps integrated GPUs at 30+ FPS. */

export type Quality = 'high' | 'medium' | 'low';

export interface QualitySettings {
  pixelRatioCap: number;
  shadows: boolean;
  shadowMapSize: number;
  ao: boolean;
  bloom: boolean;
  grade: boolean;
  standardBuildings: boolean;
  carShadows: boolean;
  buildingDrawDistance: number;
  propDrawDistance: number;
}

export const QUALITY: Record<Quality, QualitySettings> = {
  high: {
    pixelRatioCap: 2,
    shadows: true,
    shadowMapSize: 4096,
    ao: true,
    bloom: true,
    grade: true,
    standardBuildings: true,
    carShadows: true,
    buildingDrawDistance: 12000,
    propDrawDistance: 2500,
  },
  medium: {
    pixelRatioCap: 1.25,
    shadows: true,
    shadowMapSize: 2048,
    ao: false,
    bloom: true,
    grade: true,
    standardBuildings: true,
    carShadows: false,
    buildingDrawDistance: 9000,
    propDrawDistance: 1500,
  },
  low: {
    pixelRatioCap: 1,
    shadows: false,
    shadowMapSize: 1024,
    ao: false,
    bloom: false,
    grade: false,
    standardBuildings: false,
    carShadows: false,
    buildingDrawDistance: 6000,
    propDrawDistance: 800,
  },
};

const KEY = 'redraw-quality';

/** Pick a default from the GPU string; a saved choice or ?quality= wins. */
export function initialQuality(gl: WebGLRenderingContext | WebGL2RenderingContext | null): { q: Quality; pinned: boolean } {
  const fromUrl = typeof location !== 'undefined' ? new URLSearchParams(location.search).get('quality') : null;
  if (fromUrl === 'high' || fromUrl === 'medium' || fromUrl === 'low') return { q: fromUrl, pinned: true };
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'high' || saved === 'medium' || saved === 'low') return { q: saved, pinned: true };
  } catch {
    /* storage unavailable */
  }
  let renderer = '';
  try {
    const ext = gl?.getExtension('WEBGL_debug_renderer_info');
    renderer = ext && gl ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : '';
  } catch {
    /* ignore */
  }
  if (/swiftshader|llvmpipe|software/i.test(renderer)) return { q: 'low', pinned: false };
  if (/intel|mali|adreno|powervr|apple gpu/i.test(renderer)) return { q: 'medium', pinned: false };
  return { q: 'high', pinned: false };
}

export function saveQuality(q: Quality): void {
  try {
    localStorage.setItem(KEY, q);
  } catch {
    /* ignore */
  }
}
