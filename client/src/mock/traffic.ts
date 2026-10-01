/**
 * MOCK FIXTURE: generates an RDPB v1 playback for the mock world. This is
 * stand-in data for UI development only; real traffic comes from the Python
 * sim via the API.
 */

import { encodePlayback } from '../traffic/playback';
import { parseHHMM } from '../time';
import { mockWorld, rng } from './world';

const BIN_START = 21600;
const BIN_S = 300;
const N_BINS = 48;

export interface MockScenario {
  planId: string;
  seed: number;
  /** demand multiplier for drop-off trips */
  dropoffScale: number;
  /** bell time overrides, school id -> seconds */
  bellOverride: Record<string, number>;
  /** extra curb spots per entrance key */
  extraCurb: Record<string, number>;
  shuttles: number;
}

export function generateMockPlayback(sc: MockScenario): ArrayBuffer {
  const w = mockWorld();
  const net = w.network;
  const E = net.n_edges;
  const r = rng(sc.seed);
  const N = Math.round(Math.sqrt(net.nodes.length));
  const node = (i: number, j: number): number => w.nodeGrid(i, j);
  const ij = (id: number): [number, number] => [Math.floor((id - 1000) / N), (id - 1000) % N];

  /** Manhattan route: move along the row first, then the column. */
  const route = (from: number, to: number): number[] => {
    const out: number[] = [];
    let [i, j] = ij(from);
    const [ti, tj] = ij(to);
    while (j !== tj) {
      const nj = j + Math.sign(tj - j);
      out.push(w.edgeByUV.get(`${node(i, j)}-${node(i, nj)}`)!);
      j = nj;
    }
    while (i !== ti) {
      const ni = i + Math.sign(ti - i);
      out.push(w.edgeByUV.get(`${node(i, j)}-${node(ni, j)}`)!);
      i = ni;
    }
    return out;
  };

  interface Trip {
    kind: number;
    depart: number;
    edges: number[];
    entrance: number; // index into entrance_ids or -1
  }
  const entranceIds = w.schools.flatMap((s) => s.entrances.map((e) => e.key));
  const schoolNodes = w.schools.map((s) => {
    const e = s.entrances[0]!;
    const n = net.nodes.find((nd) => Math.abs(nd.x - e.x) < 1 && Math.abs(nd.z - e.z) < 1)!;
    return n.id;
  });
  const trips: Trip[] = [];
  const randNode = (): number => node(Math.floor(r() * N), Math.floor(r() * N));
  const gauss = (): number => (r() + r() + r() + r() - 2) / 0.58;

  // commuters toward exits, peak ~07:35
  for (let k = 0; k < 1500; k++) {
    const from = randNode();
    const to = w.exitNodes[Math.floor(r() * w.exitNodes.length)]!;
    if (from === to) continue;
    trips.push({ kind: r() < 0.1 ? 4 : 0, depart: 27300 + gauss() * 1700, edges: route(from, to), entrance: -1 });
  }
  // drop-off cars toward schools, arriving 5-20 min before bell
  w.schools.forEach((s, si) => {
    const bell = sc.bellOverride[s.id] ?? parseHHMM(s.bell_start);
    const n = Math.round((s.students / 9) * sc.dropoffScale);
    for (let k = 0; k < n; k++) {
      const from = randNode();
      if (from === schoolNodes[si]) continue;
      const edges = route(from, schoolNodes[si]!);
      const travel = edges.length * 30;
      trips.push({ kind: 1, depart: bell - 300 - r() * 900 - travel, edges, entrance: si });
    }
  });
  // shuttles and school buses
  const busCount = 6 + sc.shuttles * 4;
  for (let k = 0; k < busCount; k++) {
    const si = k % w.schools.length;
    const s = w.schools[si]!;
    const bell = sc.bellOverride[s.id] ?? parseHHMM(s.bell_start);
    const from = w.exitNodes[k % w.exitNodes.length]!;
    trips.push({ kind: k < sc.shuttles * 4 ? 2 : 3, depart: bell - 1500 + r() * 300, edges: route(from, schoolNodes[si]!), entrance: -1 });
  }

  // two passes: free-flow volumes -> BPR-ish slowdown (mock only)
  const vol = new Float32Array(N_BINS * E);
  const freeSpeed = (e: number): number => (net.edges[e]!.highway === 'residential' ? 11 : 15.5);
  const cap = (e: number): number => net.edges[e]!.lanes * 40; // vehicles per bin in this downscaled fixture
  for (const t of trips) {
    let time = t.depart;
    for (const e of t.edges) {
      const b = Math.floor((time - BIN_START) / BIN_S);
      if (b >= 0 && b < N_BINS) vol[b * E + e]! += 1;
      time += net.edges[e]!.len / freeSpeed(e);
    }
  }
  const vc = new Float32Array(N_BINS * E);
  const speed = new Float32Array(N_BINS * E);
  for (let b = 0; b < N_BINS; b++) {
    for (let e = 0; e < E; e++) {
      const v = vol[b * E + e]! / cap(e);
      vc[b * E + e] = v;
      speed[b * E + e] = (freeSpeed(e) * 3.6) / (1 + 0.15 * v ** 4);
    }
  }

  // queues at entrances
  const nEnt = entranceIds.length;
  const arrivals = new Float32Array(N_BINS * nEnt);
  const offsets: number[] = [0];
  const kinds: number[] = [];
  const tEdge: number[] = [];
  const tEnter: number[] = [];
  const tExit: number[] = [];
  const queueLen = new Float32Array(N_BINS * nEnt);
  const sample = trips.slice(0, 3000);
  for (const t of sample) {
    let time = t.depart;
    t.edges.forEach((e, idx) => {
      const b = Math.min(N_BINS - 1, Math.max(0, Math.floor((time - BIN_START) / BIN_S)));
      const sp = Math.max(1.5, speed[b * E + e]! / 3.6);
      let dur = net.edges[e]!.len / sp;
      if (idx === t.edges.length - 1 && t.entrance >= 0) {
        arrivals[b * nEnt + t.entrance]! += 1;
        dur += 60 + r() * 300; // waiting in the drop-off queue (stretched final edge)
      }
      tEdge.push(e);
      tEnter.push(time);
      tExit.push(time + dur);
      time += dur;
    });
    kinds.push(t.kind);
    offsets.push(tEdge.length);
  }
  w.schools.forEach((s, k) => {
    const ent = s.entrances[0]!;
    const curb = ent.curb_spots + (sc.extraCurb[ent.key] ?? 0);
    const service = (curb / ent.unload_seconds) * BIN_S * 0.18;
    let q = 0;
    for (let b = 0; b < N_BINS; b++) {
      q = Math.max(0, q + arrivals[b * nEnt + k]! * 1.6 - service);
      queueLen[b * nEnt + k] = q;
    }
  });

  return encodePlayback(
    {
      plan_id: sc.planId,
      seed: 0,
      synthetic: true,
      bin_start_s: BIN_START,
      bin_s: BIN_S,
      n_bins: N_BINS,
      report_start_s: 23400,
      report_end_s: 34200,
      n_edges: E,
      entrance_ids: entranceIds,
      n_trajectories: kinds.length,
      n_points: tEdge.length,
      kinds: { '0': 'car', '1': 'dropoff_car', '2': 'shuttle', '3': 'school_bus', '4': 'carpool' },
      mock: true,
    },
    [
      { name: 'edge_vc', dtype: 'float32', data: vc },
      { name: 'edge_speed_kph', dtype: 'float32', data: speed },
      { name: 'queue_len', dtype: 'float32', data: queueLen },
      { name: 'traj_offsets', dtype: 'uint32', data: offsets },
      { name: 'traj_kind', dtype: 'uint32', data: kinds },
      { name: 'traj_edge', dtype: 'int32', data: tEdge },
      { name: 'traj_enter_s', dtype: 'float32', data: tEnter },
      { name: 'traj_exit_s', dtype: 'float32', data: tExit },
    ],
  );
}
