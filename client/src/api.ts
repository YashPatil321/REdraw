/**
 * Typed client for every endpoint in docs/api.md. Base `/api` (the Vite dev
 * server proxies it to the FastAPI app with the prefix stripped).
 *
 * The transport is a `fetch`-compatible function. With `?mock=1` the app swaps
 * in an in-memory fake server (src/mock/server.ts) so the UI can be exercised
 * without the API. Mock data is labeled as such everywhere.
 */

import type {
  BaselineResponse,
  BuildingInfo,
  Job,
  Manifest,
  NetworkJson,
  Plan,
  PlanCheck,
  PlanInput,
  PlanListItem,
  ResidentsResponse,
  School,
  ToolsResponse,
  WorldMeta,
} from './types';

export const API_BASE = '/api';

export type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;
export type Query = Record<string, string | number | boolean | null | undefined>;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly url: string,
  ) {
    super(`${status} ${detail} (${url})`);
    this.name = 'ApiError';
  }
}

/** Join the API base with a path and an optional query (empty values dropped). */
export function buildUrl(path: string, query?: Query, base: string = API_BASE): string {
  const p = path.startsWith('/') ? path : `/${path}`;
  let url = `${base.replace(/\/+$/, '')}${p}`;
  if (query) {
    const parts: string[] = [];
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === null || v === '') continue;
      parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
    }
    if (parts.length) url += `?${parts.join('&')}`;
  }
  return url;
}

/** Asset path from `/world/meta` (relative to the API base) -> client URL. */
export function assetUrl(meta: Pick<WorldMeta, 'assets'>, rel: string): string {
  if (/^https?:\/\//.test(rel)) return rel;
  if (rel.startsWith('/')) return buildUrl(rel);
  const baseUrl = meta.assets.base_url.endsWith('/') ? meta.assets.base_url : `${meta.assets.base_url}/`;
  return buildUrl(`${baseUrl}${rel}`);
}

export class RedrawApi {
  constructor(private fetchImpl: FetchLike) {}

  setTransport(fetchImpl: FetchLike): void {
    this.fetchImpl = fetchImpl;
  }

  private async raw(method: string, path: string, opts: { query?: Query; body?: unknown } = {}): Promise<Response> {
    const url = buildUrl(path, opts.query);
    const init: RequestInit = { method, credentials: 'same-origin', headers: {} };
    if (opts.body !== undefined) {
      init.body = JSON.stringify(opts.body);
      init.headers = { 'Content-Type': 'application/json' };
    }
    const res = await this.fetchImpl(url, init);
    if (!res.ok) {
      let detail = res.statusText || 'request failed';
      try {
        const j = (await res.json()) as { detail?: unknown };
        if (j && j.detail !== undefined) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(res.status, detail, url);
    }
    return res;
  }

  private async json<T>(method: string, path: string, opts: { query?: Query; body?: unknown } = {}): Promise<T> {
    const res = await this.raw(method, path, opts);
    return (await res.json()) as T;
  }

  private async binary(path: string): Promise<ArrayBuffer> {
    const res = await this.raw('GET', path);
    return res.arrayBuffer();
  }

  // ---- World
  getMeta(): Promise<WorldMeta> {
    return this.json('GET', '/world/meta');
  }
  getNetwork(): Promise<NetworkJson> {
    return this.json('GET', '/world/network');
  }
  getBuilding(id: number): Promise<BuildingInfo> {
    return this.json('GET', `/world/buildings/${encodeURIComponent(String(id))}`);
  }
  getSchools(): Promise<School[]> {
    return this.json('GET', '/world/schools');
  }

  // ---- Assets (served by the API under /assets)
  async getManifest(meta: WorldMeta): Promise<Manifest> {
    const res = await this.fetchImpl(assetUrl(meta, meta.assets.manifest));
    if (!res.ok) throw new ApiError(res.status, 'asset manifest not found (run the pipeline?)', meta.assets.manifest);
    return (await res.json()) as Manifest;
  }
  async getAsset(meta: WorldMeta, rel: string): Promise<ArrayBuffer> {
    const url = assetUrl(meta, rel);
    const res = await this.fetchImpl(url);
    if (!res.ok) throw new ApiError(res.status, 'asset not found', url);
    return res.arrayBuffer();
  }

  // ---- Baseline
  getBaseline(): Promise<BaselineResponse> {
    return this.json('GET', '/baseline');
  }
  getBaselinePlayback(): Promise<ArrayBuffer> {
    return this.binary('/baseline/playback');
  }

  // ---- Tools
  getTools(): Promise<ToolsResponse> {
    return this.json('GET', '/tools');
  }

  // ---- Plans
  createPlan(plan: PlanInput): Promise<Plan> {
    return this.json('POST', '/plans', { body: plan });
  }
  /** PUT /plans/{id} (author only); resets the plan to draft. */
  updatePlan(id: string, plan: PlanInput): Promise<Plan> {
    return this.json('PUT', `/plans/${encodeURIComponent(id)}`, { body: plan });
  }
  checkPlan(plan: PlanInput): Promise<PlanCheck> {
    return this.json('POST', '/plans/check', { body: plan });
  }
  getPlan(id: string): Promise<Plan> {
    return this.json('GET', `/plans/${encodeURIComponent(id)}`);
  }
  runPlan(id: string): Promise<{ job_id: string }> {
    return this.json('POST', `/plans/${encodeURIComponent(id)}/run`);
  }
  getJob(id: string): Promise<Job> {
    return this.json('GET', `/jobs/${encodeURIComponent(id)}`);
  }
  getPlanPlayback(id: string): Promise<ArrayBuffer> {
    return this.binary(`/plans/${encodeURIComponent(id)}/playback`);
  }
  getResidents(id: string): Promise<ResidentsResponse> {
    return this.json('GET', `/plans/${encodeURIComponent(id)}/residents`);
  }
  listPlans(opts: { mission?: string; sort?: string; limit?: number; offset?: number } = {}): Promise<{ plans: PlanListItem[]; total?: number }> {
    return this.json('GET', '/plans', { query: { mission: opts.mission, sort: opts.sort, limit: opts.limit, offset: opts.offset } });
  }
  vote(id: string, value: 1 | -1 | 0): Promise<{ votes: number }> {
    return this.json('POST', `/plans/${encodeURIComponent(id)}/vote`, { body: { value } });
  }
}

/** Retry a call while the server answers 503 (world loading / baseline warming up). */
export async function with503Retry<T>(fn: () => Promise<T>, onWait?: (attempt: number) => void, maxWaitMs = 600000): Promise<T> {
  const start = Date.now();
  for (let attempt = 1; ; attempt++) {
    try {
      return await fn();
    } catch (e) {
      if (!(e instanceof ApiError) || e.status !== 503 || Date.now() - start > maxWaitMs) throw e;
      onWait?.(attempt);
      await new Promise((r) => setTimeout(r, Math.min(5000, 1000 * attempt)));
    }
  }
}

const defaultFetch: FetchLike = (url, init) => fetch(url, init);

/** App-wide client. `initApi()` switches to the mock transport for `?mock=1`. */
export const api = new RedrawApi(defaultFetch);

export function isMockMode(search: string = typeof location !== 'undefined' ? location.search : ''): boolean {
  const v = new URLSearchParams(search).get('mock');
  return v !== null && v !== '0' && v !== 'false';
}

export async function initApi(): Promise<{ mock: boolean }> {
  if (!isMockMode()) return { mock: false };
  const { createMockFetch } = await import('./mock/server');
  api.setTransport(await createMockFetch());
  return { mock: true };
}
