import { describe, expect, it } from 'vitest';
import type { NetworkJson } from '../types';
import { RoadNetwork } from './network';

// An L-shaped edge 0 (0,0)->(100,0)->(100,100), its reverse as edge 1, and a
// separate edge 2 far away.
const json: NetworkJson = {
  n_edges: 3,
  edges: [
    { i: 0, u: 1, v: 2, name: 'A', label: '', highway: 'residential', lanes: 1, len: 200, pts: [0, 10, 0, 100, 10, 0, 100, 20, 100] },
    { i: 1, u: 2, v: 1, name: 'A', label: '', highway: 'residential', lanes: 1, len: 200, pts: [100, 20, 100, 100, 10, 0, 0, 10, 0] },
    { i: 2, u: 3, v: 4, name: 'B', label: 'B', highway: 'primary', lanes: 2, len: 100, pts: [1000, 0, 1000, 1100, 0, 1000] },
  ],
  nodes: [
    { id: 1, x: 0, y: 10, z: 0, signal: false },
    { id: 2, x: 100, y: 20, z: 100, signal: true },
    { id: 3, x: 1000, y: 0, z: 1000, signal: false },
    { id: 4, x: 1100, y: 0, z: 1000, signal: false },
  ],
};

describe('RoadNetwork geometry', () => {
  const net = new RoadNetwork(json, 50);

  it('computes polyline lengths', () => {
    expect(net.length[0]).toBeCloseTo(200);
    expect(net.length[2]).toBeCloseTo(100);
  });

  it('interpolates positions and headings along an edge', () => {
    const out = { x: 0, y: 0, z: 0, dx: 0, dz: 0 };
    net.pointAt(0, 0.25, out);
    expect(out.x).toBeCloseTo(50);
    expect(out.z).toBeCloseTo(0);
    expect(out.y).toBeCloseTo(10);
    expect(out.dx).toBeCloseTo(1);
    net.pointAt(0, 0.75, out);
    expect(out.x).toBeCloseTo(100);
    expect(out.z).toBeCloseTo(50);
    expect(out.y).toBeCloseTo(15);
    expect(out.dz).toBeCloseTo(1);
    net.pointAt(0, 2, out); // clamped
    expect(out.z).toBeCloseTo(100);
  });

  it('snaps to the nearest edge point', () => {
    const hit = net.nearestEdge(40, 12, 100)!;
    expect(hit).not.toBeNull();
    expect([0, 1]).toContain(hit.edge);
    expect(hit.x).toBeCloseTo(40);
    expect(hit.z).toBeCloseTo(0);
    expect(hit.dist).toBeCloseTo(12);
    const far = net.nearestEdge(1050, 990, 100)!;
    expect(far.edge).toBe(2);
    expect(far.frac).toBeCloseTo(0.5);
    expect(net.nearestEdge(500, 500, 100)).toBeNull();
  });

  it('respects an edge filter', () => {
    const hit = net.nearestEdge(40, 12, 100, (e) => e === 1)!;
    expect(hit.edge).toBe(1);
    expect(hit.frac).toBeCloseTo(0.8);
  });

  it('snaps to the nearest node', () => {
    expect(net.nearestNode(90, 95, 50)?.id).toBe(2);
    expect(net.nearestNode(400, 400, 50)).toBeNull();
    expect(net.nearestNode(90, 95, 500, (n) => !n.signal)?.id).toBe(1);
  });
});
