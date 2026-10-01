/** App state: a tiny observable store (no framework). */

import type { Quality } from './scene/quality';
import type { RoadNetwork } from './traffic/network';
import type { Playback } from './traffic/playback';
import type {
  BuildingInfo,
  Job,
  Plan,
  PlanCheck,
  ResidentsResponse,
  School,
  ToolDef,
  ToolInstance,
  WorldMeta,
} from './types';

export type View = 'explore' | 'traffic' | 'plan' | 'report' | 'browse';

/** Which map-input param is currently being placed by clicking the map. */
export interface MapPick {
  toolIndex: number;
  paramId: string;
  type: string;
  maxCount?: number;
}

export interface Toast {
  id: number;
  kind: 'info' | 'error';
  text: string;
}

export type RenderMode = 'photoreal' | 'open';

export interface PhotorealState {
  /** a Google key (or a dev tileset URL) is configured */
  available: boolean;
  source: 'google' | 'url' | null;
  /** loading / progress message ('' when loaded) */
  status: string;
  /** fatal error (bad key, quota): the toggle falls back to open data */
  error: string | null;
}

export interface AttributionState {
  /** show the Google logo (Google tiles visible) */
  google: boolean;
  /** data provider strings collected from the visible tiles */
  text: string;
}

export interface AppState {
  view: View;
  booting: boolean;
  bootMessage: string;
  fatal: string | null;
  meta: WorldMeta | null;
  schools: School[];
  tools: ToolDef[];
  network: RoadNetwork | null;
  /** loading status for the world assets */
  worldStatus: string;

  // traffic playback
  simTime: number;
  playing: boolean;
  speed: number;
  ghost: boolean;
  baselinePlayback: Playback | null;
  planPlayback: Playback | null;
  /** which playback the Traffic view shows */
  trafficSource: 'baseline' | 'plan';
  split: boolean;
  /** split slider position, 0..1 of the canvas width */
  splitPos: number;

  // plan builder
  draft: { title: string; pitch: string; tools: ToolInstance[] };
  selectedTool: number | null;
  check: PlanCheck | null;
  checking: boolean;
  mapPick: MapPick | null;
  plan: Plan | null;
  job: Job | null;
  residents: ResidentsResponse | null;
  reportTab: 'report' | 'residents';

  // explore
  building: BuildingInfo | null;
  buildingLoading: boolean;
  school: School | null;
  highlightEdge: number | null;
  showStats: boolean;
  quality: Quality;
  toasts: Toast[];

  // base map
  renderMode: RenderMode;
  photoreal: PhotorealState;
  attribution: AttributionState | null;
  /** street-level first-person camera is on */
  walking: boolean;
  /** photo mode: UI hidden, ultra quality, PNG capture */
  photoMode: boolean;
  /** open-data credits from the asset sources (terrain_meta.json), '' until loaded */
  openCredits: string;
}

export type Listener = (s: AppState, prev: AppState) => void;

export class Store<S extends object> {
  private listeners = new Set<(s: S, prev: S) => void>();
  constructor(private s: S) {}

  get(): S {
    return this.s;
  }

  set(patch: Partial<S> | ((s: S) => Partial<S>)): void {
    const prev = this.s;
    const p = typeof patch === 'function' ? patch(prev) : patch;
    let changed = false;
    for (const k of Object.keys(p) as Array<keyof S>) {
      if (prev[k] !== p[k]) {
        changed = true;
        break;
      }
    }
    if (!changed) return;
    this.s = { ...prev, ...p };
    for (const l of [...this.listeners]) l(this.s, prev);
  }

  subscribe(fn: (s: S, prev: S) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  /** Subscribe to a derived value; fires only when it changes (===). */
  select<T>(sel: (s: S) => T, fn: (v: T, prev: T) => void): () => void {
    let last = sel(this.s);
    return this.subscribe((s) => {
      const v = sel(s);
      if (v !== last) {
        const p = last;
        last = v;
        fn(v, p);
      }
    });
  }
}

export function initialState(): AppState {
  return {
    view: 'explore',
    booting: true,
    bootMessage: 'Loading world…',
    fatal: null,
    meta: null,
    schools: [],
    tools: [],
    network: null,
    worldStatus: '',
    simTime: 27000,
    playing: false,
    speed: 10,
    ghost: false,
    baselinePlayback: null,
    planPlayback: null,
    trafficSource: 'baseline',
    split: false,
    splitPos: 0.5,
    draft: { title: '', pitch: '', tools: [] },
    selectedTool: null,
    check: null,
    checking: false,
    mapPick: null,
    plan: null,
    job: null,
    residents: null,
    reportTab: 'report',
    building: null,
    buildingLoading: false,
    school: null,
    highlightEdge: null,
    showStats: false,
    quality: 'medium',
    toasts: [],
    renderMode: 'open',
    photoreal: { available: false, source: null, status: '', error: null },
    attribution: null,
    walking: false,
    photoMode: false,
    openCredits: '',
  };
}

export const store = new Store<AppState>(initialState());

let toastId = 1;
export function toast(text: string, kind: Toast['kind'] = 'info', ms = 5000): void {
  const t: Toast = { id: toastId++, kind, text };
  store.set((s) => ({ toasts: [...s.toasts, t] }));
  setTimeout(() => store.set((s) => ({ toasts: s.toasts.filter((x) => x.id !== t.id) })), ms);
}
