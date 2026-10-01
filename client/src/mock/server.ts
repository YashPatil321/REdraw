/**
 * MOCK FIXTURE server (`?mock=1`). An in-memory fake of the API in
 * docs/api.md, returned as a fetch-compatible function. Numbers here are
 * placeholders for UI development, never results: every response is marked
 * synthetic and the UI shows a MOCK banner.
 */

import { parse } from 'yaml';
import toolsYaml from '../../../data/config/tools.yaml?raw';
import type { FetchLike } from '../api';
import { parseHHMM } from '../time';
import type {
  Job,
  MetricDef,
  Plan,
  PlanCheck,
  PlanInput,
  PlanListItem,
  Range3,
  Reaction,
  Report,
  ToolDef,
  WorldMeta,
} from '../types';
import { generateMockPlayback, type MockScenario } from './traffic';
import { MOCK_HALF, MOCK_ORIGIN, buildingsGlb, mockBbox, mockManifest, mockWorld, rng, roadsGlb, terrainGlb } from './world';

const MISSION = {
  id: 'morning_crunch',
  title: 'The Morning Crunch (mock)',
  brief:
    'MOCK FIXTURE. School drop off and commute traffic stack up every weekday morning. Cut the delay without adding through lanes. (Mission text normally comes from the API.)',
  budget_usd_upfront: 2000000,
  budget_usd_per_year: 400000,
  constraints: ['no_new_through_lanes'],
  goals_suggested: ['reduce average drop off delay at the high school by 30 percent', 'no school gets worse', 'resident approval above 55 percent'],
};

const METRICS: MetricDef[] = [
  { id: 'avg_commute_min', label: 'Average commute time, all workers', unit: 'min', better: 'lower' },
  { id: 'avg_dropoff_delay_min', label: 'Average drop-off delay', unit: 'min', better: 'lower' },
  { id: 'max_spillback_m', label: 'Max queue spillback', unit: 'm', better: 'lower' },
  { id: 'total_vht', label: 'Total vehicle hours traveled', unit: 'h', better: 'lower' },
  { id: 'late_kids', label: 'Kids arriving late', unit: 'kids', better: 'lower' },
  { id: 'cost_upfront_usd', label: 'Upfront cost', unit: 'USD', better: 'lower' },
  { id: 'cost_per_year_usd', label: 'Cost per year', unit: 'USD', better: 'lower' },
  { id: 'resident_approval_pct', label: 'Resident approval', unit: '%', better: 'higher' },
  { id: 'winners', label: 'Residents better off (3+ min)', unit: 'people', better: 'higher' },
  { id: 'losers', label: 'Residents worse off (3+ min)', unit: 'people', better: 'lower' },
];

const BASE_VALUES: Record<string, number> = {
  avg_commute_min: 21.3,
  avg_dropoff_delay_min: 7.8,
  max_spillback_m: 214,
  total_vht: 3120,
  late_kids: 64,
};

// Mock costs per tool (placeholders; real costs come from assumptions.yaml via the API)
const MOCK_COST: Record<string, [number, number]> = {
  bell_time: [0, 0],
  school_shuttle: [0, 95000],
  carpool_program: [0, 60000],
  dropoff_redesign: [180000, 0],
  new_dropoff_entrance: [650000, 0],
  signal_timing: [15000, 0],
  turn_lane: [420000, 0],
  bike_route: [300000, 0],
  safe_walk_route: [220000, 30000],
  teen_drive_policy: [0, 0],
};

interface MockJob extends Job {
  started: number;
}

const LS_KEY = 'redraw-mock-plans-v1';

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function uuid(r: () => number): string {
  const h = (n: number): string => Array.from({ length: n }, () => Math.floor(r() * 16).toString(16)).join('');
  return `${h(8)}-${h(4)}-4${h(3)}-a${h(3)}-${h(12)}`;
}

function range(median: number, spread: number): Range3 {
  return { median, p10: median - spread, p90: median + spread };
}

export async function createMockFetch(): Promise<FetchLike> {
  const tools = (parse(toolsYaml) as { tools: ToolDef[] }).tools;
  const world = mockWorld();
  const r = rng(42);
  const plans = new Map<string, Plan>();
  const jobs = new Map<string, MockJob>();
  const assetCache = new Map<string, Promise<ArrayBuffer>>();
  const playbackCache = new Map<string, ArrayBuffer>();

  const save = (): void => {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify([...plans.values()]));
    } catch {
      /* storage unavailable: in-memory only */
    }
  };
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) for (const p of JSON.parse(raw) as Plan[]) plans.set(p.id, p);
  } catch {
    /* ignore */
  }

  const meta: WorldMeta = {
    region: {
      name: 'mock_fixture',
      display_name: 'Mock fixture town (not a real place)',
      bbox: mockBbox(),
      origin: { lat: MOCK_ORIGIN.lat, lon: MOCK_ORIGIN.lon },
      timezone: 'America/Los_Angeles',
      extent_scene: { min_x: -MOCK_HALF, max_x: MOCK_HALF, min_z: -MOCK_HALF, max_z: MOCK_HALF },
    },
    synthetic: true,
    mock: true,
    assets: { base_url: '/assets/', manifest: '/assets/manifest.json' },
    counts: { buildings: world.buildings.size, road_edges: world.network.n_edges, road_nodes: world.network.nodes.length },
    calibration: { status: 'uncalibrated', median_error_pct: null, targets_with_data: 0, targets_total: 6 },
    mission: MISSION,
    metrics: METRICS,
    time: { bin_start_s: 21600, bin_s: 300, n_bins: 48, report_start_s: 23400, report_end_s: 34200 },
    hero: { school_id: world.schools[0]!.id, x: world.schools[0]!.x, z: world.schools[0]!.z },
    unverified_inputs: ['Mock High School bell time 08:30 (fixture)', 'Curb spots at all mock entrances (fixture)'],
  };

  const check = (p: PlanInput): PlanCheck => {
    const errors: string[] = [];
    const warnings: string[] = [];
    let up = 0;
    let yr = 0;
    p.tools.forEach((inst, idx) => {
      const def = tools.find((t) => t.id === inst.tool);
      if (!def) {
        errors.push(`tool ${idx + 1}: unknown tool '${inst.tool}'`);
        return;
      }
      if (def.enabled_in_mvp === false) errors.push(`${def.id}: not available in this version`);
      for (const prm of def.params) {
        const v = inst.params[prm.id];
        const empty = v === undefined || v === null || v === '' || (Array.isArray(v) && v.length === 0);
        if (prm.required && empty) errors.push(`${def.id}: ${prm.label} is required`);
        if (prm.type === 'polyline' && Array.isArray(v) && v.length === 1) errors.push(`${def.id}: ${prm.label} needs at least 2 points`);
      }
      const [u, y] = MOCK_COST[def.id] ?? [0, 0];
      const buses = typeof inst.params['buses'] === 'number' ? (inst.params['buses'] as number) : 1;
      up += u;
      yr += def.id === 'school_shuttle' ? y * buses : y;
      if (def.id === 'turn_lane') warnings.push('turn_lane: mock warning, check the approach is not a through lane');
    });
    const over = up > MISSION.budget_usd_upfront || yr > MISSION.budget_usd_per_year;
    return {
      ok: errors.length === 0 && p.tools.length > 0,
      errors: p.tools.length === 0 ? ['Add at least one tool'] : errors,
      warnings,
      cost_upfront_usd: up,
      cost_per_year_usd: yr,
      over_budget: over,
      budget_upfront_usd: MISSION.budget_usd_upfront,
      budget_per_year_usd: MISSION.budget_usd_per_year,
      constraint_violations: [],
      resolved_tools: p.tools.map((t) => ({ ...t })),
    };
  };

  const scenario = (p: Plan | null): MockScenario => {
    const sc: MockScenario = { planId: p?.id ?? 'baseline', seed: 11, dropoffScale: 1, bellOverride: {}, extraCurb: {}, shuttles: 0 };
    for (const t of p?.tools ?? []) {
      if (t.tool === 'bell_time' && typeof t.params['school'] === 'string' && typeof t.params['start'] === 'string') {
        sc.bellOverride[t.params['school']] = parseHHMM(t.params['start']);
      }
      if (t.tool === 'school_shuttle') sc.shuttles += 1;
      if (t.tool === 'carpool_program' || t.tool === 'school_shuttle' || t.tool === 'safe_walk_route' || t.tool === 'teen_drive_policy') {
        sc.dropoffScale *= 0.85;
      }
      if (t.tool === 'dropoff_redesign' && typeof t.params['entrance'] === 'string') {
        sc.extraCurb[t.params['entrance']] = Number(t.params['extra_curb_spots'] ?? 0);
      }
    }
    return sc;
  };

  const playbackFor = (p: Plan | null): ArrayBuffer => {
    const key = p?.id ?? 'baseline';
    let buf = playbackCache.get(key);
    if (!buf) {
      buf = generateMockPlayback(scenario(p));
      playbackCache.set(key, buf);
    }
    return buf;
  };

  const report = (p: Plan): Report => {
    const c = check(p);
    const pr = rng(p.id.length * 31 + p.tools.length);
    const effect = Math.min(0.35, 0.06 * p.tools.length + pr() * 0.05);
    const metrics = METRICS.filter((m) => m.id in BASE_VALUES).map((m) => {
      const b = BASE_VALUES[m.id]!;
      const pl = b * (1 - effect * (m.id === 'avg_commute_min' ? 0.2 : 1));
      const sb = b * 0.04;
      const sp = b * 0.07;
      return { ...m, baseline: range(b, sb), plan: range(pl, sp), delta: range(pl - b, Math.max(sb, sp)) };
    });
    metrics.push(
      { ...METRICS[5]!, baseline: range(0, 0), plan: range(c.cost_upfront_usd, 0), delta: range(c.cost_upfront_usd, 0) },
      { ...METRICS[6]!, baseline: range(0, 0), plan: range(c.cost_per_year_usd, 0), delta: range(c.cost_per_year_usd, 0) },
      {
        ...METRICS[7]!,
        baseline: { median: null, p10: null, p90: null },
        plan: range(48 + effect * 40, 4),
        delta: { median: null, p10: null, p90: null },
        note: 'Share of residents who approve (mock). The status quo has no approval number.',
      },
    );
    const sideEdges = [world.network.edges.find((e) => e.label === 'Test Canyon Drive')!, world.network.edges.find((e) => e.label === 'Mock Ranch Road')!];
    return {
      plan_id: p.id,
      seeds: 20,
      generated_at: new Date().toISOString(),
      synthetic: true,
      cost: {
        upfront_usd: c.cost_upfront_usd,
        per_year_usd: c.cost_per_year_usd,
        over_budget: c.over_budget,
        budget_upfront_usd: c.budget_upfront_usd,
        budget_per_year_usd: c.budget_per_year_usd,
        lines: p.tools.map((t) => ({
          tool: t.tool,
          upfront_usd: MOCK_COST[t.tool]?.[0] ?? 0,
          per_year_usd: (MOCK_COST[t.tool]?.[1] ?? 0) * (t.tool === 'school_shuttle' ? Number(t.params['buses'] ?? 1) : 1),
          note: 'mock cost (fixture)',
        })),
      },
      constraint_violations: [],
      metrics,
      per_school: world.schools.map((s, i) => ({
        school_id: s.id,
        name: s.name,
        dropoff_delay_min: { baseline: range(7 + i, 0.5), plan: range((7 + i) * (1 - effect), 0.8) },
        max_spillback_m: { baseline: range(210 - i * 40, 15), plan: range((210 - i * 40) * (1 - effect), 22) },
        late_kids: { baseline: range(30 - i * 8, 4), plan: range((30 - i * 8) * (1 - effect), 5) },
      })),
      mode_share: [
        { mode: 'drive_dropoff', baseline_pct: 61, plan_pct: 61 - effect * 20, delta_pp: -effect * 20 },
        { mode: 'drive_alone', baseline_pct: 18, plan_pct: 18, delta_pp: 0 },
        { mode: 'carpool', baseline_pct: 6, plan_pct: 6 + effect * 8, delta_pp: effect * 8 },
        { mode: 'school_shuttle', baseline_pct: 0, plan_pct: effect * 7, delta_pp: effect * 7 },
        { mode: 'bike', baseline_pct: 4, plan_pct: 4 + effect * 2, delta_pp: effect * 2 },
        { mode: 'walk', baseline_pct: 7, plan_pct: 7 + effect * 3, delta_pp: effect * 3 },
        { mode: 'school_bus', baseline_pct: 4, plan_pct: 4, delta_pp: 0 },
      ],
      winners: range(Math.round(1800 * effect * 4), 200),
      losers: range(Math.round(300 * effect * 2), 60),
      side_effects: sideEdges.map((e, i) => {
        const mid = e.pts.length / 2;
        const k = Math.floor(mid / 3) * 3;
        return { edge_idx: e.i, name: e.name, baseline_vc: 0.78 + i * 0.05, plan_vc: 0.93 + i * 0.04, bin_s: 27000 + i * 600, x: e.pts[k]!, z: e.pts[k + 2]! };
      }),
      peak_overlap: { baseline: 0.42, plan: Math.max(0.1, 0.42 - effect * 0.5), note: 'share of school drop-off arrivals in the commute peak hour' },
      unverified_inputs: meta.unverified_inputs,
      llm_estimated_tools: [],
      calibration: meta.calibration,
    };
  };

  const residents = (p: Plan): { approval_pct: number; reactions: Reaction[]; text_status: string; llm_available: boolean } => {
    const pr = rng(p.id.charCodeAt(0) * 7);
    const names = ['Maya', 'Arjun', 'Lena', 'Diego', 'Priya', 'Tom', 'Grace', 'Kenji', 'Sofia', 'Omar', 'Hannah', 'Luis'];
    const values = ['time', 'safety', 'cost', 'environment', 'community', 'property', 'change averse'];
    const reactions: Reaction[] = names.map((first_name, i) => {
      const approval = Math.round(pr() * 100) / 100;
      const commute = Math.round((pr() * 8 - 5) * 10) / 10;
      const blockId = 1 + Math.floor(pr() * world.buildings.size);
      const b = world.buildings.get(blockId) ?? world.buildings.get(1)!;
      return {
        persona_id: 100 + i,
        first_name,
        age: 28 + Math.floor(pr() * 45),
        block: `${b.block} (mock)`,
        x: b.x,
        z: b.z,
        school_ids: [world.schools[i % 3]!.id],
        values: [values[i % 7]!, values[(i + 2) % 7]!, values[(i + 4) % 7]!],
        approval,
        approves: approval >= 0.5,
        deltas: { commute_min: commute, dropoff_min: Math.round((pr() * 10 - 6) * 10) / 10, cost_usd_year: 0, street_change: pr() < 0.2 },
        text:
          i % 4 === 3
            ? null
            : approval >= 0.5
              ? `MOCK QUOTE: my mornings get about ${Math.abs(commute)} minutes ${commute < 0 ? 'shorter' : 'longer'}, and I like where this is going.`
              : `MOCK QUOTE: I am not convinced. My commute changes by ${commute} minutes and I worry about my street.`,
      };
    });
    const approval_pct = Math.round((reactions.filter((x) => x.approves).length / reactions.length) * 1000) / 10;
    return { approval_pct, reactions, text_status: 'partial', llm_available: false };
  };

  const newPlan = (body: PlanInput, id?: string): Plan => ({
    id: id ?? uuid(r),
    mission: body.mission,
    title: body.title,
    pitch: body.pitch,
    author_id: null,
    tools: body.tools,
    created_at: new Date().toISOString(),
    report: null,
    status: 'draft',
    check: check(body),
    votes: 0,
  });

  // seed two finished example plans so Browse and Report have content
  if (plans.size === 0) {
    const seeds: PlanInput[] = [
      {
        mission: 'morning_crunch',
        title: 'Stagger and shuttle (mock example)',
        pitch: 'Move the high school to 9:00 and run two shuttles.',
        tools: [
          { tool: 'bell_time', params: { school: 'mock_high', start: '09:00' } },
          { tool: 'school_shuttle', params: { school: 'mock_high', stops: [[33.0, -117.13], [33.006, -117.12]], buses: 2, headway_min: 15 } },
        ],
      },
      {
        mission: 'morning_crunch',
        title: 'More curb, fewer cars (mock example)',
        pitch: 'Add curb spots and a carpool program.',
        tools: [
          { tool: 'dropoff_redesign', params: { entrance: 'mock_high/main_dropoff', extra_curb_spots: 6, faster_unload: true } },
          { tool: 'carpool_program', params: { incentive: 'high' } },
        ],
      },
    ];
    seeds.forEach((s, i) => {
      const p = newPlan(s, `mock-example-${i + 1}`);
      p.status = 'done';
      p.votes = 12 - i * 5;
      p.report = report(p);
      plans.set(p.id, p);
    });
    save();
  }

  const asset = (path: string): Promise<ArrayBuffer> | null => {
    let m: RegExpExecArray | null;
    let gen: (() => Promise<ArrayBuffer>) | null = null;
    if ((m = /^terrain\/terrain_r(\d+)_c(\d+)\.glb$/.exec(path))) {
      const [row, col] = [Number(m[1]), Number(m[2])];
      gen = () => terrainGlb(row, col);
    } else if ((m = /^buildings\/buildings_r(\d+)_c(\d+)\.glb$/.exec(path))) {
      const [row, col] = [Number(m[1]), Number(m[2])];
      gen = () => buildingsGlb(row, col);
    } else if (path === 'roads/roads.glb') {
      gen = () => roadsGlb();
    }
    if (!gen) return null;
    let pr = assetCache.get(path);
    if (!pr) {
      pr = gen();
      assetCache.set(path, pr);
    }
    return pr;
  };

  const headline = (p: Plan): Record<string, number> => {
    const out: Record<string, number> = {};
    for (const m of p.report?.metrics ?? []) if (m.plan.median !== null) out[m.id] = m.plan.median;
    return out;
  };

  const handle = async (method: string, path: string, query: URLSearchParams, body: unknown): Promise<Response> => {
    await new Promise((res) => setTimeout(res, 40)); // pretend network
    let m: RegExpExecArray | null;
    if (method === 'GET' && path === '/world/meta') return json(meta);
    if (method === 'GET' && path === '/world/network') return json(world.network);
    if (method === 'GET' && path === '/world/schools') return json(world.schools);
    if (method === 'GET' && (m = /^\/world\/buildings\/(\d+)$/.exec(path))) {
      const b = world.buildings.get(Number(m[1]));
      if (!b) return json({ detail: 'unknown building' }, 404);
      const { x: _x, z: _z, w: _w, d: _d, h: _h, ...props } = b;
      return json(props);
    }
    if (method === 'GET' && path === '/assets/manifest.json') return json(mockManifest());
    if (method === 'GET' && path.startsWith('/assets/')) {
      const pr = asset(path.slice('/assets/'.length));
      if (!pr) return json({ detail: 'asset not found' }, 404);
      return new Response(await pr, { headers: { 'Content-Type': 'model/gltf-binary' } });
    }
    if (method === 'GET' && path === '/baseline') {
      return json({
        summary: { metrics: METRICS.filter((x) => x.id in BASE_VALUES).map((x) => ({ ...x, value: range(BASE_VALUES[x.id]!, BASE_VALUES[x.id]! * 0.04) })), per_school: [], mode_share: [] },
        playback_url: '/baseline/playback',
        calibration: meta.calibration,
      });
    }
    if (method === 'GET' && path === '/baseline/playback') return new Response(playbackFor(null));
    if (method === 'GET' && path === '/tools') return json({ mission: 'morning_crunch', tools });
    if (method === 'POST' && path === '/plans/check') return json(check(body as PlanInput));
    if (method === 'POST' && path === '/plans') {
      const b = body as PlanInput;
      if (!b || !Array.isArray(b.tools)) return json({ detail: 'tools must be a list' }, 422);
      const p = newPlan(b);
      plans.set(p.id, p);
      save();
      return json(p, 201);
    }
    if (method === 'GET' && path === '/plans') {
      const sort = query.get('sort') ?? 'new';
      let list = [...plans.values()];
      if (sort === 'votes') list.sort((a, b) => b.votes - a.votes);
      else if (sort === 'new') list.sort((a, b) => b.created_at.localeCompare(a.created_at));
      else {
        const def = METRICS.find((x) => x.id === sort);
        const sign = def?.better === 'higher' ? -1 : 1;
        list = list.filter((p) => p.report);
        list.sort((a, b) => sign * ((headline(a)[sort] ?? Infinity) - (headline(b)[sort] ?? Infinity)));
      }
      const items: PlanListItem[] = list.map((p) => ({ id: p.id, title: p.title, pitch: p.pitch, created_at: p.created_at, votes: p.votes, status: p.status, headline: headline(p) }));
      return json({ plans: items });
    }
    if ((m = /^\/plans\/([^/]+)(\/[a-z]+)?$/.exec(path))) {
      const p = plans.get(decodeURIComponent(m[1]!));
      if (!p) return json({ detail: 'plan not found' }, 404);
      const sub = m[2] ?? '';
      if (method === 'GET' && sub === '') return json(p);
      if (method === 'POST' && sub === '/run') {
        if (!p.check?.ok) return json({ detail: 'plan check failed; fix errors first' }, 400);
        const job: MockJob = { id: uuid(r), plan_id: p.id, status: 'queued', progress: 0, message: 'queued', error: null, started: Date.now() };
        jobs.set(job.id, job);
        p.status = 'queued';
        save();
        return json({ job_id: job.id });
      }
      if (method === 'GET' && sub === '/playback') {
        if (p.status !== 'done') return json({ detail: 'not run yet' }, 404);
        return new Response(playbackFor(p));
      }
      if (method === 'GET' && sub === '/residents') {
        if (p.status !== 'done') return json({ detail: 'not run yet' }, 404);
        return json(residents(p));
      }
      if (method === 'POST' && sub === '/vote') {
        const v = Number((body as { value?: number })?.value ?? 0);
        p.votes += v;
        save();
        return json({ votes: p.votes });
      }
    }
    if (method === 'GET' && (m = /^\/jobs\/([^/]+)$/.exec(path))) {
      const job = jobs.get(m[1]!);
      if (!job) return json({ detail: 'job not found' }, 404);
      const frac = Math.min(1, (Date.now() - job.started) / 4500);
      const p = plans.get(job.plan_id)!;
      if (frac >= 1) {
        if (job.status !== 'done') {
          p.report = report(p);
          p.status = 'done';
          save();
        }
        Object.assign(job, { status: 'done', progress: 1, message: 'done' });
      } else {
        const seed = Math.min(20, Math.floor(frac * 20) + 1);
        Object.assign(job, { status: 'running', progress: frac, message: `seed ${seed}/20 (mock)` });
        p.status = 'running';
      }
      const { started: _s, ...out } = job;
      return json(out);
    }
    return json({ detail: `mock: no route for ${method} ${path}` }, 404);
  };

  return async (url: string, init?: RequestInit): Promise<Response> => {
    const u = new URL(url, 'http://mock.local');
    const path = u.pathname.replace(/^\/api/, '');
    const method = (init?.method ?? 'GET').toUpperCase();
    const body = typeof init?.body === 'string' ? (JSON.parse(init.body) as unknown) : undefined;
    try {
      return await handle(method, path, u.searchParams, body);
    } catch (e) {
      console.error('mock server error', e);
      return json({ detail: `mock error: ${(e as Error).message}` }, 500);
    }
  };
}
