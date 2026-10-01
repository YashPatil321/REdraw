/** Shared Lit helpers: theme styles, store controller, scene context, formatters. */

import { css, type ReactiveController, type ReactiveControllerHost } from 'lit';
import type { SceneController } from '../scene/controller';
import { store, type AppState } from '../state';

/** Global handle to the 3D scene for components that need camera actions. */
export const appCtx: { scene: SceneController | null } = { scene: null };

/** Re-renders the host when any of the selected state keys change. */
export class StoreController implements ReactiveController {
  private unsub: (() => void) | null = null;
  constructor(
    private host: ReactiveControllerHost,
    private keys: Array<keyof AppState>,
  ) {
    host.addController(this);
  }
  get s(): AppState {
    return store.get();
  }
  hostConnected(): void {
    this.unsub = store.subscribe((s, prev) => {
      if (this.keys.some((k) => s[k] !== prev[k])) this.host.requestUpdate();
    });
  }
  hostDisconnected(): void {
    this.unsub?.();
    this.unsub = null;
  }
}

export const theme = css`
  :host {
    --bg: rgba(13, 17, 25, 0.86);
    --bg-solid: #0f141d;
    --bg-2: rgba(255, 255, 255, 0.06);
    --line: rgba(255, 255, 255, 0.12);
    --text: #e7ecf3;
    --muted: #9aa6b6;
    --plan: #ff5fa2;
    --base: #4fb3ff;
    --good: #4cd38a;
    --bad: #ff6b5a;
    --warn: #ffc44d;
    color: var(--text);
    font: 13px/1.4 system-ui, -apple-system, 'Segoe UI', Roboto, Ubuntu, sans-serif;
  }
  .panel {
    background: var(--bg);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--line);
    border-radius: 10px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    pointer-events: auto;
  }
  h2 {
    font-size: 15px;
    margin: 0 0 6px;
    font-weight: 650;
  }
  h3 {
    font-size: 12px;
    margin: 12px 0 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    font-weight: 650;
  }
  p {
    margin: 4px 0;
  }
  .muted {
    color: var(--muted);
  }
  .small {
    font-size: 12px;
  }
  button,
  select,
  input,
  textarea {
    font: inherit;
    color: var(--text);
  }
  button {
    background: var(--bg-2);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 5px 10px;
    cursor: pointer;
  }
  button:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.12);
  }
  button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  button.primary {
    background: var(--plan);
    border-color: var(--plan);
    color: #1a0c13;
    font-weight: 650;
  }
  button.primary:hover:not(:disabled) {
    background: #ff7db4;
  }
  button.ghost {
    background: transparent;
    border-color: transparent;
  }
  button.active {
    background: rgba(255, 95, 162, 0.18);
    border-color: var(--plan);
  }
  select,
  input[type='text'],
  input[type='number'],
  input[type='time'],
  textarea {
    background: rgba(0, 0, 0, 0.35);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 4px 6px;
    width: 100%;
    box-sizing: border-box;
  }
  select option {
    background: var(--bg-solid);
  }
  .scroll {
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .spacer {
    flex: 1;
  }
  .tag {
    display: inline-block;
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 999px;
    border: 1px solid var(--line);
    color: var(--muted);
  }
  .tag.warn {
    color: var(--warn);
    border-color: rgba(255, 196, 77, 0.5);
  }
  .tag.bad {
    color: var(--bad);
    border-color: rgba(255, 107, 90, 0.5);
  }
  .tag.good {
    color: var(--good);
    border-color: rgba(76, 211, 138, 0.5);
  }
  .err {
    color: var(--bad);
  }
  .warnc {
    color: var(--warn);
  }
`;

export function fmtUsd(v: number): string {
  if (!Number.isFinite(v)) return '-';
  const a = Math.abs(v);
  if (a >= 1e6) return `$${(v / 1e6).toFixed(a >= 1e7 ? 1 : 2)}M`;
  if (a >= 1e3) return `$${Math.round(v / 1e3)}k`;
  return `$${Math.round(v)}`;
}

export function fmtNum(v: number | null | undefined, unit = ''): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'n/a';
  if (unit.startsWith('USD')) return fmtUsd(v);
  if (Number.isInteger(v)) return v.toLocaleString('en-US');
  const a = Math.abs(v);
  const s = a >= 1000 ? Math.round(v).toLocaleString('en-US') : a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
  return s;
}

export function fmtSigned(v: number | null | undefined, unit = ''): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'n/a';
  const s = fmtNum(Math.abs(v), unit);
  return `${v > 0 ? '+' : v < 0 ? '−' : '±'}${s}`;
}

export function humanize(id: string): string {
  const s = id.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}
