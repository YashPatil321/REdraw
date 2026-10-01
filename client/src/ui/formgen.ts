/**
 * Data-driven form generation from GET /tools definitions. Pure functions; the
 * Lit components only render what these return. No validation or cost logic
 * here: the server's POST /plans/check is the source of truth.
 */

import type { ParamValue, School, ToolDef, ToolInstance, ToolParamDef } from '../types';

export type MapType = 'point' | 'points' | 'polyline' | 'edge' | 'node';
export const MAP_TYPES: readonly string[] = ['point', 'points', 'polyline', 'edge', 'node'];

interface Base {
  id: string;
  label: string;
  required: boolean;
  value: ParamValue;
}

export type FieldSpec =
  | (Base & { kind: 'select'; options: Array<{ value: string; label: string }> })
  | (Base & { kind: 'number'; min?: number; max?: number; step: number; integer: boolean })
  | (Base & { kind: 'time'; min?: string; max?: string })
  | (Base & { kind: 'checkbox' })
  | (Base & { kind: 'text'; maxLength?: number; multiline: boolean })
  | (Base & { kind: 'map'; mapType: MapType; maxCount?: number; minCount: number; summary: string })
  | (Base & { kind: 'unsupported'; type: string });

export interface FormContext {
  schools: School[];
}

export function isMapParam(p: ToolParamDef): boolean {
  return MAP_TYPES.includes(p.type) || p.map_input === true;
}

function num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

/** Initial value for a param (tool default first). */
export function defaultValue(p: ToolParamDef, ctx: FormContext): ParamValue {
  if (p.default !== undefined && p.default !== null) return p.default as ParamValue;
  switch (p.type) {
    case 'school':
      return p.required ? (ctx.schools[0]?.id ?? '') : null;
    case 'entrance': {
      const s = ctx.schools.find((x) => x.entrances.length > 0);
      return p.required && s ? entranceKey(s, s.entrances[0]!) : null;
    }
    case 'select':
      return p.required ? (p.options?.[0]?.value ?? '') : null;
    case 'int':
    case 'float':
      return num(p.min) ?? 0;
    case 'bool':
      return false;
    case 'time':
    case 'text':
      return p.required ? '' : null;
    case 'points':
    case 'polyline':
      return [];
    default:
      return null; // point, edge, node: unset until placed on the map
  }
}

export function entranceKey(s: School, e: { id: string; key?: string }): string {
  return e.key ?? `${s.id}/${e.id}`;
}

/** Default params object for a new tool instance. Unset optional params are omitted. */
export function defaultParams(tool: ToolDef, ctx: FormContext): Record<string, ParamValue> {
  const out: Record<string, ParamValue> = {};
  for (const p of tool.params) {
    const v = defaultValue(p, ctx);
    if (v !== null) out[p.id] = v;
  }
  return out;
}

export function newToolInstance(tool: ToolDef, ctx: FormContext): ToolInstance {
  return { tool: tool.id, params: defaultParams(tool, ctx) };
}

/** Human summary of a map param value. */
export function mapSummary(type: string, v: ParamValue): string {
  switch (type) {
    case 'point':
      return Array.isArray(v) && v.length === 2 ? `${fmtLL(v as number[])}` : 'not placed';
    case 'points':
    case 'polyline': {
      const n = Array.isArray(v) ? v.length : 0;
      return n === 0 ? 'none placed' : `${n} point${n === 1 ? '' : 's'}`;
    }
    case 'edge':
      return typeof v === 'number' ? `road edge #${v}` : 'not picked';
    case 'node':
      return typeof v === 'number' ? `intersection #${v}` : 'not picked';
    default:
      return v === null || v === undefined ? 'not set' : String(v);
  }
}

function fmtLL(ll: number[]): string {
  return `${ll[0]!.toFixed(5)}, ${ll[1]!.toFixed(5)}`;
}

/** Build the field list for a tool instance. */
export function buildFields(tool: ToolDef, params: Record<string, ParamValue>, ctx: FormContext): FieldSpec[] {
  return tool.params.map((p): FieldSpec => {
    const required = p.required === true;
    const value = p.id in params ? params[p.id]! : defaultValue(p, ctx);
    const base = { id: p.id, label: p.label, required, value };
    if (isMapParam(p) && MAP_TYPES.includes(p.type)) {
      const mapType = p.type as MapType;
      return {
        ...base,
        kind: 'map',
        mapType,
        maxCount: mapType === 'points' || mapType === 'polyline' ? num(p.max_count) : 1,
        minCount: mapType === 'polyline' ? 2 : required ? 1 : 0,
        summary: mapSummary(mapType, value),
      };
    }
    switch (p.type) {
      case 'school': {
        const options = ctx.schools.map((s) => ({ value: s.id, label: s.name }));
        if (!required) options.unshift({ value: '', label: p.allow_all ? 'All schools' : '(none)' });
        return { ...base, kind: 'select', options };
      }
      case 'entrance': {
        const options: Array<{ value: string; label: string }> = [];
        for (const s of ctx.schools) {
          for (const e of s.entrances) options.push({ value: entranceKey(s, e), label: `${s.name}: ${e.id.replace(/_/g, ' ')}` });
        }
        if (!required) options.unshift({ value: '', label: '(none)' });
        return { ...base, kind: 'select', options };
      }
      case 'select': {
        const options = (p.options ?? []).map((o) => ({ value: String(o.value), label: o.label }));
        if (!required) options.unshift({ value: '', label: '(none)' });
        return { ...base, kind: 'select', options };
      }
      case 'int':
      case 'float': {
        const integer = p.type === 'int';
        return {
          ...base,
          kind: 'number',
          integer,
          min: num(p.min),
          max: num(p.max),
          step: num(p.step) ?? (integer ? 1 : 0.1),
        };
      }
      case 'time':
        return {
          ...base,
          kind: 'time',
          min: typeof p.min === 'string' ? p.min : undefined,
          max: typeof p.max === 'string' ? p.max : undefined,
        };
      case 'bool':
        return { ...base, kind: 'checkbox' };
      case 'text':
        return { ...base, kind: 'text', maxLength: num(p.max_length), multiline: (num(p.max_length) ?? 0) > 120 };
      default:
        return { ...base, kind: 'unsupported', type: p.type };
    }
  });
}

/** Parse a raw input string for a field into a param value (null = omit). */
export function coerceInput(field: FieldSpec, raw: string | boolean): ParamValue {
  switch (field.kind) {
    case 'checkbox':
      return Boolean(raw);
    case 'number': {
      if (raw === '' || typeof raw === 'boolean') return null;
      const n = Number(raw);
      if (!Number.isFinite(n)) return null;
      return field.integer ? Math.round(n) : n;
    }
    case 'select':
    case 'time':
    case 'text':
      return raw === '' && !field.required ? null : String(raw);
    default:
      return null;
  }
}

/** Set (or omit, for null) one param immutably. */
export function setParam(params: Record<string, ParamValue>, id: string, v: ParamValue): Record<string, ParamValue> {
  const out = { ...params };
  if (v === null) delete out[id];
  else out[id] = v;
  return out;
}

export interface MapClick {
  lat: number;
  lon: number;
  edge?: number;
  node?: number;
}

/**
 * New value of a map param after a map click (UI placement only; the server
 * re-validates and resolves). Returns null when the click can't apply.
 */
export function applyMapClick(type: MapType, current: ParamValue, click: MapClick, maxCount?: number): ParamValue | null {
  const ll = [round6(click.lat), round6(click.lon)];
  switch (type) {
    case 'point':
      return ll;
    case 'points':
    case 'polyline': {
      const arr = Array.isArray(current) ? ([...current] as number[][]) : [];
      if (maxCount !== undefined && arr.length >= maxCount) return null;
      arr.push(ll);
      return arr;
    }
    case 'edge':
      return click.edge ?? null;
    case 'node':
      return click.node ?? null;
  }
}

function round6(v: number): number {
  return Math.round(v * 1e6) / 1e6;
}
