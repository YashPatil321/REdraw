import { describe, expect, it } from 'vitest';
import { ApiError, RedrawApi, assetUrl, buildUrl, createStaticFetch, isLiveFirst, isMockMode, with503Retry } from './api';
import { staticTargetFor } from './staticPaths';
import type { WorldMeta } from './types';

interface Call {
  url: string;
  method: string;
  body: unknown;
}

function recorder(response: unknown = {}, status = 200) {
  const calls: Call[] = [];
  const f = async (url: string, init?: RequestInit): Promise<Response> => {
    calls.push({
      url,
      method: init?.method ?? 'GET',
      body: typeof init?.body === 'string' ? JSON.parse(init.body) : undefined,
    });
    if (response instanceof ArrayBuffer) return new Response(response, { status });
    return new Response(JSON.stringify(response), { status, headers: { 'Content-Type': 'application/json' } });
  };
  return { calls, f };
}

describe('buildUrl', () => {
  it('prefixes the /api base', () => {
    expect(buildUrl('/world/meta')).toBe('/api/world/meta');
    expect(buildUrl('tools')).toBe('/api/tools');
  });
  it('encodes query and drops empty values', () => {
    expect(buildUrl('/plans', { mission: 'morning_crunch', sort: 'avg_commute_min' })).toBe(
      '/api/plans?mission=morning_crunch&sort=avg_commute_min',
    );
    expect(buildUrl('/plans', { mission: undefined, sort: '' })).toBe('/api/plans');
    expect(buildUrl('/plans', { sort: 'a b&c' })).toBe('/api/plans?sort=a%20b%26c');
  });
  it('supports a custom base', () => {
    expect(buildUrl('/x', undefined, 'http://h:8000/')).toBe('http://h:8000/x');
  });
});

describe('assetUrl', () => {
  const meta = { assets: { base_url: '/assets/', manifest: '/assets/manifest.json' } } as Pick<WorldMeta, 'assets'>;
  it('resolves manifest and tile paths relative to the API base', () => {
    expect(assetUrl(meta, meta.assets.manifest)).toBe('/api/assets/manifest.json');
    expect(assetUrl(meta, 'terrain/terrain_r0_c0.glb')).toBe('/api/assets/terrain/terrain_r0_c0.glb');
    expect(assetUrl({ assets: { base_url: '/assets', manifest: '' } }, 'roads/roads.glb')).toBe(
      '/api/assets/roads/roads.glb',
    );
  });
});

describe('RedrawApi endpoints', () => {
  const cases: Array<[string, (a: RedrawApi) => Promise<unknown>, string, string, unknown?]> = [
    ['meta', (a) => a.getMeta(), 'GET', '/api/world/meta'],
    ['network', (a) => a.getNetwork(), 'GET', '/api/world/network'],
    ['building', (a) => a.getBuilding(42), 'GET', '/api/world/buildings/42'],
    ['schools', (a) => a.getSchools(), 'GET', '/api/world/schools'],
    ['baseline', (a) => a.getBaseline(), 'GET', '/api/baseline'],
    ['tools', (a) => a.getTools(), 'GET', '/api/tools'],
    ['plan', (a) => a.getPlan('abc'), 'GET', '/api/plans/abc'],
    ['run', (a) => a.runPlan('abc'), 'POST', '/api/plans/abc/run'],
    ['job', (a) => a.getJob('j1'), 'GET', '/api/jobs/j1'],
    ['residents', (a) => a.getResidents('abc'), 'GET', '/api/plans/abc/residents'],
    ['list', (a) => a.listPlans({ mission: 'morning_crunch', sort: 'votes' }), 'GET', '/api/plans?mission=morning_crunch&sort=votes'],
    ['vote', (a) => a.vote('abc', -1), 'POST', '/api/plans/abc/vote', { value: -1 }],
    ['update', (a) => a.updatePlan('abc', { mission: 'm', title: 't', pitch: '', tools: [] }), 'PUT', '/api/plans/abc', { mission: 'm', title: 't', pitch: '', tools: [] }],
    ['list paged', (a) => a.listPlans({ sort: 'new', limit: 20, offset: 40 }), 'GET', '/api/plans?sort=new&limit=20&offset=40'],
  ];
  for (const [name, fn, method, url, body] of cases) {
    it(name, async () => {
      const r = recorder({});
      await fn(new RedrawApi(r.f));
      expect(r.calls).toHaveLength(1);
      expect(r.calls[0]!.method).toBe(method);
      expect(r.calls[0]!.url).toBe(url);
      if (body !== undefined) expect(r.calls[0]!.body).toEqual(body);
    });
  }

  it('POSTs plan bodies for create and check', async () => {
    const r = recorder({ ok: true });
    const a = new RedrawApi(r.f);
    const plan = { mission: 'morning_crunch', title: 't', pitch: 'p', tools: [{ tool: 'bell_time', params: { school: 'x', start: '09:00' } }] };
    await a.createPlan(plan);
    await a.checkPlan(plan);
    expect(r.calls.map((c) => [c.method, c.url])).toEqual([
      ['POST', '/api/plans'],
      ['POST', '/api/plans/check'],
    ]);
    expect(r.calls[1]!.body).toEqual(plan);
  });

  it('fetches binary playback', async () => {
    const buf = new Uint8Array([82, 68, 80, 66]).buffer;
    const r = recorder(buf);
    const a = new RedrawApi(r.f);
    const out = await a.getBaselinePlayback();
    expect(new Uint8Array(out)).toEqual(new Uint8Array(buf));
    await a.getPlanPlayback('p 1');
    expect(r.calls.map((c) => c.url)).toEqual(['/api/baseline/playback', '/api/plans/p%201/playback']);
  });

  it('raises ApiError with the server detail', async () => {
    const r = recorder({ detail: 'check.ok is false' }, 400);
    const a = new RedrawApi(r.f);
    await expect(a.runPlan('x')).rejects.toBeInstanceOf(ApiError);
    await expect(a.runPlan('x')).rejects.toMatchObject({ status: 400, detail: 'check.ok is false' });
  });
});

describe('with503Retry', () => {
  it('retries while the server is warming up', async () => {
    let n = 0;
    const out = await with503Retry(async () => {
      n++;
      if (n < 2) throw new ApiError(503, 'warming up', '/api/baseline');
      return 'ok';
    });
    expect(out).toBe('ok');
    expect(n).toBe(2);
  });
  it('does not retry other errors', async () => {
    await expect(with503Retry(async () => Promise.reject(new ApiError(404, 'nope', '/x')))).rejects.toMatchObject({ status: 404 });
  });
});

describe('isMockMode', () => {
  it('reads ?mock=1', () => {
    expect(isMockMode('?mock=1')).toBe(true);
    expect(isMockMode('?mock=0')).toBe(false);
    expect(isMockMode('')).toBe(false);
  });
});

describe('static viewer transport', () => {
  const files: Record<string, unknown> = {
    '/v/static-api/world/meta.json': { region: 'm' },
    '/v/static-api/plans/index-votes.json': { plans: [] },
    '/v/static-api/plans/p1.json': { id: 'p1' },
    '/v/static-api/world/buildings.json': { fields: ['type', 'height_m', 'name'], rows: { '7': ['house', 6.5, null] } },
    '/v/world-assets/manifest.json': { tiles: [] },
  };
  const seen: string[] = [];
  const host = async (url: string): Promise<Response> => {
    seen.push(url);
    return url in files ? new Response(JSON.stringify(files[url]), { status: 200 }) : new Response('nope', { status: 404 });
  };

  it('serves snapshot files for GETs', async () => {
    const a = new RedrawApi(createStaticFetch('/v/', null, host));
    expect(await a.getMeta()).toEqual({ region: 'm' });
    expect(await a.listPlans({ sort: 'votes' })).toEqual({ plans: [] });
    expect(await a.getPlan('p1')).toEqual({ id: 'p1' });
    expect(await a.getBuilding(7)).toEqual({ id: 7, type: 'house', height_m: 6.5 });
    const m = await a.getManifest({ assets: { base_url: '/assets/', manifest: '/assets/manifest.json' } } as WorldMeta);
    expect(m).toEqual({ tiles: [] });
  });

  it('explains that writes need the Python API (no hosted API configured)', async () => {
    const a = new RedrawApi(createStaticFetch('/v/', null, host));
    await expect(a.checkPlan({ mission: 'm', title: 't', pitch: '', tools: [] })).rejects.toMatchObject({ status: 501 });
    await expect(a.runPlan('p1')).rejects.toMatchObject({ status: 501 });
    await expect(a.getPlan('unknown')).rejects.toMatchObject({ status: 404 });
  });

  it('forwards writes and unknown plans to VITE_API_BASE when set', async () => {
    const r = recorder({ ok: true });
    const both = (url: string, init?: RequestInit) => (url.startsWith('https://api.example') ? r.f(url, init) : host(url));
    const a = new RedrawApi(createStaticFetch('/v/', 'https://api.example', both));
    await a.checkPlan({ mission: 'm', title: 't', pitch: '', tools: [] });
    await a.getPlan('fresh');
    expect(r.calls.map((c) => `${c.method} ${c.url}`)).toEqual(['POST https://api.example/plans/check', 'GET https://api.example/plans/fresh']);
  });
});

describe('static viewer with a hosted API', () => {
  const snap = async (url: string): Promise<Response> =>
    url === '/v/static-api/plans/index-votes.json' || url === '/v/static-api/world/meta.json'
      ? new Response(JSON.stringify({ from: 'snapshot' }), { status: 200 })
      : new Response('nope', { status: 404 });

  it('asks the hosted API first for plans, keeps world data static', async () => {
    const r = recorder({ from: 'live' });
    const both = (url: string, init?: RequestInit) => (url.startsWith('https://api.example') ? r.f(url, init) : snap(url));
    const a = new RedrawApi(createStaticFetch('/v/', 'https://api.example', both));
    expect(await a.listPlans({ sort: 'votes' })).toEqual({ from: 'live' });
    expect(await a.getMeta()).toEqual({ from: 'snapshot' });
    expect(r.calls.map((c) => c.url)).toEqual(['https://api.example/plans?sort=votes']);
  });

  it('falls back to the snapshot when the hosted API is down', async () => {
    const down = async (url: string): Promise<Response> => {
      if (url.startsWith('https://api.example')) throw new TypeError('network down');
      return snap(url);
    };
    const asleep = async (url: string): Promise<Response> => (url.startsWith('https://api.example') ? new Response('waking', { status: 503 }) : snap(url));
    expect(await new RedrawApi(createStaticFetch('/v/', 'https://api.example', down)).listPlans({ sort: 'votes' })).toEqual({ from: 'snapshot' });
    expect(await new RedrawApi(createStaticFetch('/v/', 'https://api.example', asleep)).listPlans({ sort: 'votes' })).toEqual({ from: 'snapshot' });
  });

  it('only plan routes are live-first', () => {
    expect(isLiveFirst('/plans?sort=new')).toBe(true);
    expect(isLiveFirst('/plans/abc/residents')).toBe(true);
    expect(isLiveFirst('/plansx')).toBe(false);
    expect(isLiveFirst('/world/meta')).toBe(false);
  });
});

describe('staticTargetFor', () => {
  it('maps API paths to snapshot files', () => {
    expect(staticTargetFor('GET', '/baseline/playback')).toEqual({ kind: 'file', path: 'baseline/playback.bin', binary: true });
    expect(staticTargetFor('GET', '/plans?mission=morning_crunch&sort=avg_commute_min')).toEqual({ kind: 'file', path: 'plans/index-avg_commute_min.json', binary: false });
    expect(staticTargetFor('GET', '/plans')).toMatchObject({ path: 'plans/index-new.json' });
    expect(staticTargetFor('GET', '/plans/abc/residents')).toMatchObject({ path: 'plans/abc/residents.json' });
    expect(staticTargetFor('GET', '/jobs/1')).toBeNull();
    expect(staticTargetFor('POST', '/plans')).toBeNull();
  });
});
