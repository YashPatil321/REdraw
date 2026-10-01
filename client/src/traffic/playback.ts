/**
 * RDPB v1 traffic playback reader (docs/playback_format.md).
 * Sections are looked up by name; unknown sections are ignored.
 */

export type Dtype = 'float32' | 'uint32' | 'int32';
export type TypedArray = Float32Array | Uint32Array | Int32Array;

export interface SectionInfo {
  name: string;
  dtype: Dtype | string;
  offset: number;
  count: number;
}

export interface PlaybackHeader {
  version: number;
  plan_id: string;
  seed: number;
  synthetic: boolean;
  bin_start_s: number;
  bin_s: number;
  n_bins: number;
  report_start_s: number;
  report_end_s: number;
  n_edges: number;
  entrance_ids: string[];
  n_trajectories: number;
  n_points: number;
  kinds: Record<string, string>;
  sections: SectionInfo[];
  [extra: string]: unknown;
}

export interface Playback {
  header: PlaybackHeader;
  nEdges: number;
  nBins: number;
  nEntrances: number;
  nTrajectories: number;
  /** bin-major: index b * nEdges + e */
  edgeVc: Float32Array;
  edgeSpeedKph: Float32Array | null;
  /** index b * nEntrances + k */
  queueLen: Float32Array | null;
  trajOffsets: Uint32Array;
  trajKind: Uint32Array;
  trajEdge: Int32Array;
  trajEnterS: Float32Array;
  trajExitS: Float32Array;
  /** Any section by name (including ones this client does not know). */
  section(name: string): TypedArray | undefined;
}

export const RDPB_MAGIC = 'RDPB';
export const RDPB_VERSION = 1;

const DTYPE_CTORS = {
  float32: Float32Array,
  uint32: Uint32Array,
  int32: Int32Array,
} as const;

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

export class PlaybackFormatError extends Error {
  override name = 'PlaybackFormatError';
}

function align4(n: number): number {
  return (n + 3) & ~3;
}

function readSection(buf: ArrayBuffer, dataStart: number, s: SectionInfo): TypedArray | undefined {
  const ctor = DTYPE_CTORS[s.dtype as Dtype];
  if (!ctor) return undefined; // unknown dtype: ignore (forward compatible)
  const byteOffset = dataStart + s.offset;
  const byteLen = s.count * 4;
  if (s.offset % 4 !== 0) throw new PlaybackFormatError(`section ${s.name}: offset ${s.offset} not 4-byte aligned`);
  if (byteOffset + byteLen > buf.byteLength) {
    throw new PlaybackFormatError(`section ${s.name} out of bounds (${byteOffset + byteLen} > ${buf.byteLength})`);
  }
  if (LITTLE_ENDIAN) return new ctor(buf, byteOffset, s.count);
  // Big-endian host: decode explicitly.
  const out = new ctor(s.count);
  const dv = new DataView(buf, byteOffset, byteLen);
  for (let i = 0; i < s.count; i++) {
    out[i] = s.dtype === 'float32' ? dv.getFloat32(i * 4, true) : s.dtype === 'uint32' ? dv.getUint32(i * 4, true) : dv.getInt32(i * 4, true);
  }
  return out;
}

export function parsePlayback(buf: ArrayBuffer): Playback {
  if (buf.byteLength < 12) throw new PlaybackFormatError('buffer too small for RDPB header');
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== RDPB_MAGIC) throw new PlaybackFormatError(`bad magic "${magic}"`);
  const version = dv.getUint32(4, true);
  if (version !== RDPB_VERSION) throw new PlaybackFormatError(`unsupported RDPB version ${version}`);
  const headerLen = dv.getUint32(8, true);
  if (12 + headerLen > buf.byteLength) throw new PlaybackFormatError('header length exceeds buffer');
  const header = JSON.parse(new TextDecoder('utf-8').decode(new Uint8Array(buf, 12, headerLen))) as PlaybackHeader;
  const dataStart = align4(12 + headerLen);

  const byName = new Map<string, SectionInfo>();
  for (const s of header.sections ?? []) byName.set(s.name, s);
  const cache = new Map<string, TypedArray | undefined>();
  const section = (name: string): TypedArray | undefined => {
    if (!cache.has(name)) {
      const info = byName.get(name);
      cache.set(name, info ? readSection(buf, dataStart, info) : undefined);
    }
    return cache.get(name);
  };

  const nEdges = header.n_edges;
  const nBins = header.n_bins;
  const nEntrances = header.entrance_ids?.length ?? 0;
  const nTraj = header.n_trajectories ?? 0;

  const need = <T extends TypedArray>(name: string, ctor: new (n: number) => T, expected: number): T => {
    const s = section(name);
    if (!s) {
      if (expected === 0) return new ctor(0);
      throw new PlaybackFormatError(`missing required section ${name}`);
    }
    if (!(s instanceof ctor)) throw new PlaybackFormatError(`section ${name} has wrong dtype`);
    if (s.length !== expected) throw new PlaybackFormatError(`section ${name}: count ${s.length}, expected ${expected}`);
    return s;
  };

  const edgeVc = need('edge_vc', Float32Array, nBins * nEdges);
  const speed = section('edge_speed_kph');
  const queue = section('queue_len');
  const trajOffsets = need('traj_offsets', Uint32Array, nTraj > 0 ? nTraj + 1 : section('traj_offsets')?.length ?? 0);
  const nPoints = nTraj > 0 ? trajOffsets[nTraj]! : 0;
  return {
    header,
    nEdges,
    nBins,
    nEntrances,
    nTrajectories: nTraj,
    edgeVc,
    edgeSpeedKph: speed instanceof Float32Array ? speed : null,
    queueLen: queue instanceof Float32Array ? queue : null,
    trajOffsets,
    trajKind: need('traj_kind', Uint32Array, nTraj),
    trajEdge: need('traj_edge', Int32Array, nPoints),
    trajEnterS: need('traj_enter_s', Float32Array, nPoints),
    trajExitS: need('traj_exit_s', Float32Array, nPoints),
    section,
  };
}

/** v/c of edge e in bin b (clamped bin). */
export function edgeVcAt(pb: Playback, bin: number, edge: number): number {
  const b = Math.min(Math.max(bin, 0), pb.nBins - 1);
  return pb.edgeVc[b * pb.nEdges + edge] ?? 0;
}

/** Queue length (cars) of entrance k at the end of bin b (clamped bin). */
export function queueAt(pb: Playback, bin: number, k: number): number {
  if (!pb.queueLen || pb.nEntrances === 0) return 0;
  const b = Math.min(Math.max(bin, 0), pb.nBins - 1);
  return pb.queueLen[b * pb.nEntrances + k] ?? 0;
}

export interface EncodeSection {
  name: string;
  dtype: Dtype;
  data: ArrayLike<number>;
}

/** RDPB v1 writer (used by the mock server). Header `sections` is filled in. */
export function encodePlayback(header: Omit<PlaybackHeader, 'sections' | 'version'>, sections: EncodeSection[]): ArrayBuffer {
  let off = 0;
  const infos: SectionInfo[] = sections.map((s) => {
    const info = { name: s.name, dtype: s.dtype, offset: off, count: s.data.length };
    off += align4(s.data.length * 4);
    return info;
  });
  const json = new TextEncoder().encode(JSON.stringify({ version: RDPB_VERSION, ...header, sections: infos }));
  const dataStart = align4(12 + json.length);
  const buf = new ArrayBuffer(dataStart + off);
  const dv = new DataView(buf);
  for (let i = 0; i < 4; i++) dv.setUint8(i, RDPB_MAGIC.charCodeAt(i));
  dv.setUint32(4, RDPB_VERSION, true);
  dv.setUint32(8, json.length, true);
  new Uint8Array(buf, 12, json.length).set(json);
  sections.forEach((s, si) => {
    const base = dataStart + infos[si]!.offset;
    for (let i = 0; i < s.data.length; i++) {
      const v = s.data[i]!;
      if (s.dtype === 'float32') dv.setFloat32(base + i * 4, v, true);
      else if (s.dtype === 'uint32') dv.setUint32(base + i * 4, v, true);
      else dv.setInt32(base + i * 4, v, true);
    }
  });
  return buf;
}
