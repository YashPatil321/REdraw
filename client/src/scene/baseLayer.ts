/**
 * Which base map is drawn: Google Photorealistic 3D Tiles from the air
 * ("aerial"), our own open-data world close to the ground ("street").
 *
 * Two clean states and one smooth crossfade, no spatial blending:
 * - street when walking (immediately) or once the camera is below
 *   `enterStreetM` above the ground; aerial again only above `exitStreetM`
 *   (hysteresis: hovering around one height never flickers),
 * - a switch starts only when the destination is ready (our assets streamed
 *   around the view, or Google tiles loaded for it), except entering walk
 *   mode, which switches right away (the walk-in camera move covers it),
 * - no Google (no key, quota exhausted, tiles failing): street at every
 *   altitude, silently.
 *
 * `mix` is the crossfade position: 0 = aerial only, 1 = street only. These
 * are rendering parameters (not real-world modeling numbers).
 */

export const BASE_LAYER = {
  /** go to our world below this camera height above the ground (m) */
  enterStreetM: 100,
  /** back to Google above this height (m); > enterStreetM for hysteresis */
  exitStreetM: 150,
  /** crossfade duration (s) */
  fadeS: 1.0,
  /** start a switch anyway if the destination is still not ready after this long (s) */
  readyTimeoutS: 4,
  /** keep Google streaming (for a quick return) while in street mode below this height (m) only when not walking */
  googlePrefetchM: 80,
} as const;

export type BaseLayer = 'aerial' | 'street';

export interface BaseLayerInput {
  /** camera height above the ground under it (m) */
  altitude: number;
  walking: boolean;
  /** Google tiles are configured, loaded once and not failing */
  googleOk: boolean;
  /** Google tiles for the current view finished loading */
  aerialReady: boolean;
  /** our world finished streaming around the view */
  streetReady: boolean;
}

export interface BaseLayerConfig {
  enterStreetM: number;
  exitStreetM: number;
  fadeS: number;
  readyTimeoutS: number;
}

/** Smooth ease for the crossfade (0..1 -> 0..1). */
export function easeMix(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
}

export class BaseLayerSwitch {
  /** the state being shown or faded toward */
  target: BaseLayer = 'street';
  /** linear crossfade position: 0 aerial .. 1 street */
  mix = 1;
  /** seconds the wanted state has been waiting for its assets */
  private waited = 0;

  constructor(readonly cfg: BaseLayerConfig = BASE_LAYER) {}

  /** The state the inputs ask for (hysteresis on the current target). */
  wanted(i: BaseLayerInput): BaseLayer {
    if (!i.googleOk || i.walking) return 'street';
    if (i.altitude < this.cfg.enterStreetM) return 'street';
    if (i.altitude > this.cfg.exitStreetM) return 'aerial';
    return this.target;
  }

  /**
   * Advance by dt seconds (the fade); `elapsed` is the unclamped wall time of
   * the frame for the readiness timeout (defaults to dt). Returns the eased
   * mix (0 aerial .. 1 street).
   */
  update(i: BaseLayerInput, dt: number, elapsed = dt): number {
    const want = this.wanted(i);
    if (want !== this.target) {
      const ready = want === 'street' ? i.streetReady || i.walking || !i.googleOk : i.aerialReady;
      this.waited += elapsed;
      if (ready || this.waited >= this.cfg.readyTimeoutS) {
        this.target = want;
        this.waited = 0;
      }
    } else this.waited = 0;
    const goal = this.target === 'street' ? 1 : 0;
    const step = this.cfg.fadeS > 0 ? dt / this.cfg.fadeS : 1;
    this.mix = goal > this.mix ? Math.min(goal, this.mix + step) : Math.max(goal, this.mix - step);
    return easeMix(this.mix);
  }

  /** Jump straight to a state (no fade). */
  snap(layer: BaseLayer): void {
    this.target = layer;
    this.mix = layer === 'street' ? 1 : 0;
    this.waited = 0;
  }

  get fading(): boolean {
    return this.mix > 0 && this.mix < 1;
  }

  /** Google is drawn at all (fully or in the crossfade). */
  get aerialVisible(): boolean {
    return this.mix < 1;
  }

  /** Our world is drawn at all. */
  get streetVisible(): boolean {
    return this.mix > 0;
  }
}
