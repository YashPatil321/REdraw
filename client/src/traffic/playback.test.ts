import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { PlaybackFormatError, edgeVcAt, encodePlayback, parsePlayback, queueAt } from './playback';

/**
 * Hand-built RDPB v1 buffer, byte by byte per docs/playback_format.md:
 * magic "RDPB", uint32 version, uint32 header_len, JSON header, zero padding
 * to a 4-byte boundary, then sections at `offset` bytes from the data start.
 * 2 bins x 3 edges, 1 entrance, 2 trajectories (3 points), plus an unknown
 * section the reader must ignore.
 */
function handBuilt(): ArrayBuffer {
  const sectionsData: Array<{ name: string; dtype: string; values: number[] }> = [
    { name: 'future_stuff', dtype: 'float64x', values: [7, 7] }, // unknown: must be ignored
    { name: 'edge_vc', dtype: 'float32', values: [0.1, 0.5, 0.9, 0.2, 1.2, 0.0] },
    { name: 'edge_speed_kph', dtype: 'float32', values: [50, 40, 20, 48, 10, 60] },
    { name: 'queue_len', dtype: 'float32', values: [3, 7] },
    { name: 'traj_offsets', dtype: 'uint32', values: [0, 2, 3] },
    { name: 'traj_kind', dtype: 'uint32', values: [1, 2] },
    { name: 'traj_edge', dtype: 'int32', values: [0, 2, 1] },
    { name: 'traj_enter_s', dtype: 'float32', values: [21600, 21630, 21700] },
    { name: 'traj_exit_s', dtype: 'float32', values: [21630, 21660, 21760] },
  ];
  let offset = 0;
  const sections = sectionsData.map((s) => {
    const info = { name: s.name, dtype: s.dtype, offset, count: s.values.length };
    offset += s.values.length * 4;
    return info;
  });
  const header = {
    version: 1,
    plan_id: 'baseline',
    seed: 0,
    synthetic: true,
    bin_start_s: 21600,
    bin_s: 300,
    n_bins: 2,
    report_start_s: 21600,
    report_end_s: 22200,
    n_edges: 3,
    entrance_ids: ['del_norte_hs/main_dropoff'],
    n_trajectories: 2,
    n_points: 3,
    kinds: { '0': 'car', '1': 'dropoff_car', '2': 'shuttle', '3': 'school_bus', '4': 'carpool' },
    sections,
  };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const D = Math.ceil((12 + json.length) / 4) * 4;
  const buf = new ArrayBuffer(D + offset);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x52); // R
  dv.setUint8(1, 0x44); // D
  dv.setUint8(2, 0x50); // P
  dv.setUint8(3, 0x42); // B
  dv.setUint32(4, 1, true);
  dv.setUint32(8, json.length, true);
  new Uint8Array(buf, 12, json.length).set(json);
  sectionsData.forEach((s, i) => {
    const base = D + sections[i]!.offset;
    s.values.forEach((v, j) => {
      if (s.dtype === 'float32') dv.setFloat32(base + 4 * j, v, true);
      else if (s.dtype === 'int32') dv.setInt32(base + 4 * j, v, true);
      else dv.setUint32(base + 4 * j, v, true);
    });
  });
  return buf;
}

describe('parsePlayback (RDPB v1)', () => {
  it('parses a hand-built buffer', () => {
    const pb = parsePlayback(handBuilt());
    expect(pb.header.plan_id).toBe('baseline');
    expect(pb.nEdges).toBe(3);
    expect(pb.nBins).toBe(2);
    expect(pb.nEntrances).toBe(1);
    expect(pb.nTrajectories).toBe(2);
    expect(Array.from(pb.edgeVc)).toEqual([0.1, 0.5, 0.9, 0.2, 1.2, 0.0].map(Math.fround));
    expect(edgeVcAt(pb, 1, 1)).toBeCloseTo(1.2, 6);
    expect(edgeVcAt(pb, 99, 0)).toBeCloseTo(0.2, 6); // clamped
    expect(pb.edgeSpeedKph?.[5]).toBe(60);
    expect(queueAt(pb, 0, 0)).toBe(3);
    expect(queueAt(pb, 1, 0)).toBe(7);
    expect(Array.from(pb.trajOffsets)).toEqual([0, 2, 3]);
    expect(Array.from(pb.trajKind)).toEqual([1, 2]);
    expect(Array.from(pb.trajEdge)).toEqual([0, 2, 1]);
    expect(Array.from(pb.trajEnterS)).toEqual([21600, 21630, 21700]);
    expect(Array.from(pb.trajExitS)).toEqual([21630, 21660, 21760]);
  });

  it('ignores unknown sections but exposes them by name lookup', () => {
    const pb = parsePlayback(handBuilt());
    expect(pb.section('future_stuff')).toBeUndefined(); // unknown dtype
    expect(pb.section('nope')).toBeUndefined();
    expect(pb.section('queue_len')?.length).toBe(2);
  });

  it('rejects bad magic and versions', () => {
    const b = handBuilt();
    const bad = b.slice(0);
    new DataView(bad).setUint8(0, 0x58);
    expect(() => parsePlayback(bad)).toThrow(PlaybackFormatError);
    const v2 = b.slice(0);
    new DataView(v2).setUint32(4, 2, true);
    expect(() => parsePlayback(v2)).toThrow(/version 2/);
    expect(() => parsePlayback(new ArrayBuffer(4))).toThrow(PlaybackFormatError);
  });

  it('round-trips through encodePlayback', () => {
    const buf = encodePlayback(
      {
        plan_id: 'p',
        seed: 3,
        synthetic: true,
        bin_start_s: 21600,
        bin_s: 300,
        n_bins: 1,
        report_start_s: 21600,
        report_end_s: 21900,
        n_edges: 2,
        entrance_ids: [],
        n_trajectories: 1,
        n_points: 1,
        kinds: { '0': 'car' },
      },
      [
        { name: 'edge_vc', dtype: 'float32', data: [0.25, 0.75] },
        { name: 'traj_offsets', dtype: 'uint32', data: [0, 1] },
        { name: 'traj_kind', dtype: 'uint32', data: [0] },
        { name: 'traj_edge', dtype: 'int32', data: [1] },
        { name: 'traj_enter_s', dtype: 'float32', data: [21700] },
        { name: 'traj_exit_s', dtype: 'float32', data: [21710] },
      ],
    );
    const pb = parsePlayback(buf);
    expect(Array.from(pb.edgeVc)).toEqual([0.25, 0.75]);
    expect(pb.queueLen).toBeNull();
    expect(pb.trajEdge[0]).toBe(1);
    expect(pb.header.seed).toBe(3);
  });

  const fixture = fileURLToPath(new URL('../../../sim/tests/fixtures/playback_small.rdpb', import.meta.url));
  it.skipIf(!existsSync(fixture))('parses the shared sim fixture (sim/tests/fixtures/playback_small.rdpb)', () => {
    const bytes = readFileSync(fixture);
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    const pb = parsePlayback(buf);
    expect(pb.header.version).toBe(1);
    expect(pb.edgeVc.length).toBe(pb.nBins * pb.nEdges);
    expect(pb.trajOffsets.length).toBe(pb.nTrajectories + 1);
    expect(pb.trajEdge.length).toBe(pb.header.n_points);
  });
});
