import { describe, expect, it } from 'vitest';
import { ApiError, RedrawApi, assetUrl, buildUrl, isMockMode, with503Retry } from './api';
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
