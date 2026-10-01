import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';
import type { School, ToolDef } from '../types';
import { applyMapClick, buildFields, coerceInput, defaultParams, setParam, type FieldSpec } from './formgen';

// The real tool definitions (what GET /tools returns).
const toolsYaml = readFileSync(fileURLToPath(new URL('../../../data/config/tools.yaml', import.meta.url)), 'utf8');
const tools = (parse(toolsYaml) as { tools: ToolDef[] }).tools;
const byId = (id: string): ToolDef => tools.find((t) => t.id === id)!;

const schools: School[] = [
  {
    id: 'del_norte_hs', name: 'Del Norte High School', grades: [9, 12], bell_start: '08:30', x: 0, y: 0, z: 0, verified: false, students: 2500,
    entrances: [{ id: 'main_dropoff', key: 'del_norte_hs/main_dropoff', x: 0, y: 0, z: 0, curb_spots: 10, unload_seconds: 45, verified: false }],
  },
  { id: 'design39', name: 'Design39 Campus', grades: [0, 8], bell_start: '08:15', x: 0, y: 0, z: 0, verified: false, students: 1000, entrances: [] },
];
const ctx = { schools };

describe('form generation from tool definitions', () => {
  it('renders an input for every param of every tool (no unsupported types)', () => {
    for (const t of tools) {
      const fields = buildFields(t, defaultParams(t, ctx), ctx);
      expect(fields).toHaveLength(t.params.length);
      for (const f of fields) expect(f.kind).not.toBe('unsupported');
    }
  });

  it('bell_time: school select + time with default and bounds', () => {
    const t = byId('bell_time');
    const fields = buildFields(t, defaultParams(t, ctx), ctx);
    const school = fields[0] as Extract<FieldSpec, { kind: 'select' }>;
    expect(school.kind).toBe('select');
    expect(school.options.map((o) => o.value)).toEqual(['del_norte_hs', 'design39']);
    expect(school.value).toBe('del_norte_hs');
    const start = fields[1] as Extract<FieldSpec, { kind: 'time' }>;
    expect(start).toMatchObject({ kind: 'time', value: '09:00', min: '07:00', max: '09:30', label: 'New start time' });
  });

  it('school_shuttle: map points with max_count and int params', () => {
    const t = byId('school_shuttle');
    const fields = buildFields(t, defaultParams(t, ctx), ctx);
    expect(fields.find((f) => f.id === 'stops')).toMatchObject({ kind: 'map', mapType: 'points', maxCount: 12, value: [] });
    expect(fields.find((f) => f.id === 'buses')).toMatchObject({ kind: 'number', integer: true, min: 1, max: 12, step: 1, value: 2 });
  });

  it('optional school with allow_all gets an "All schools" option', () => {
    const t = byId('carpool_program');
    const f = buildFields(t, defaultParams(t, ctx), ctx)[0] as Extract<FieldSpec, { kind: 'select' }>;
    expect(f.options[0]).toEqual({ value: '', label: 'All schools' });
    expect(defaultParams(t, ctx)).toEqual({ incentive: 'medium' });
  });

  it('entrance options come from school entrances', () => {
    const t = byId('dropoff_redesign');
    const f = buildFields(t, defaultParams(t, ctx), ctx)[0] as Extract<FieldSpec, { kind: 'select' }>;
    expect(f.options.map((o) => o.value)).toEqual(['del_norte_hs/main_dropoff']);
    expect(defaultParams(t, ctx)).toEqual({ entrance: 'del_norte_hs/main_dropoff', extra_curb_spots: 4, faster_unload: false });
  });

  it('edge, node, polyline and select types', () => {
    expect(buildFields(byId('turn_lane'), {}, ctx)[0]).toMatchObject({ kind: 'map', mapType: 'edge', summary: 'not picked' });
    expect(buildFields(byId('signal_timing'), { node: 7 }, ctx)[0]).toMatchObject({ kind: 'map', mapType: 'node', summary: 'intersection #7' });
    const bike = buildFields(byId('bike_route'), defaultParams(byId('bike_route'), ctx), ctx);
    expect(bike[0]).toMatchObject({ kind: 'map', mapType: 'polyline', minCount: 2 });
    expect(bike[1]).toMatchObject({ kind: 'select', value: 'buffered' });
    expect((bike[1] as Extract<FieldSpec, { kind: 'select' }>).options).toHaveLength(2);
  });

  it('custom tool: text field with max length', () => {
    const f = buildFields(byId('custom'), {}, ctx)[0];
    expect(f).toMatchObject({ kind: 'text', maxLength: 600, multiline: true });
  });

  it('coerces raw input values', () => {
    const fields = buildFields(byId('school_shuttle'), {}, ctx);
    const buses = fields.find((f) => f.id === 'buses')!;
    expect(coerceInput(buses, '3.7')).toBe(4);
    expect(coerceInput(buses, '')).toBeNull();
    const cp = buildFields(byId('carpool_program'), {}, ctx)[0]!;
    expect(coerceInput(cp, '')).toBeNull();
    expect(setParam({ a: 1, school: 'x' }, 'school', null)).toEqual({ a: 1 });
  });

  it('applies map clicks per type', () => {
    const click = { lat: 33.0123456789, lon: -117.1, edge: 5, node: 9 };
    expect(applyMapClick('point', null, click)).toEqual([33.012346, -117.1]);
    expect(applyMapClick('points', [[1, 2]], click)).toEqual([[1, 2], [33.012346, -117.1]]);
    expect(applyMapClick('points', [[1, 2]], click, 1)).toBeNull();
    expect(applyMapClick('edge', null, click)).toBe(5);
    expect(applyMapClick('node', null, click)).toBe(9);
  });
});
