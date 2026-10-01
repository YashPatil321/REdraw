/** Traffic view UI: time scrubber, legend panel and the before/after split slider. */

import { LitElement, css, html, nothing } from 'lit';
import { store } from '../state';
import { SPEED_STOPS, VC_STOPS, rgbToCss } from '../traffic/congestion';
import { formatHHMM, formatHHMMSS } from '../time';
import { StoreController, fmtNum, humanize, theme } from './base';

const SPEEDS = [1, 5, 10, 30, 60];

function gradient(stops: ReadonlyArray<{ at: number; c: [number, number, number] }>): string {
  const max = stops[stops.length - 1]!.at;
  return `linear-gradient(90deg, ${stops.map((s) => `${rgbToCss(s.c)} ${((s.at / max) * 100).toFixed(0)}%`).join(', ')})`;
}

export class RdScrubber extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        padding: 8px 12px;
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .time {
        font: 600 18px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
        min-width: 92px;
        text-align: center;
      }
      .track {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-width: 160px;
      }
      input[type='range'] {
        width: 100%;
        accent-color: var(--plan);
      }
      .ticks {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: var(--muted);
      }
      .play {
        width: 38px;
        height: 32px;
        font-size: 15px;
      }
      select {
        width: auto;
      }
      label {
        display: flex;
        align-items: center;
        gap: 4px;
        white-space: nowrap;
      }
    `,
  ];
  private st = new StoreController(this, ['simTime', 'playing', 'speed', 'ghost', 'meta', 'view', 'split', 'planPlayback', 'trafficSource']);

  override render() {
    const s = this.st.s;
    const t = s.meta?.time;
    if (!t) return nothing;
    const ticks: number[] = [];
    for (let x = t.report_start_s; x <= t.report_end_s; x += 1800) ticks.push(x);
    return html`<div class="panel p">
      <button class="play" title=${s.playing ? 'Pause' : 'Play'} aria-label=${s.playing ? 'Pause' : 'Play'} @click=${() => store.set({ playing: !s.playing })}>
        ${s.playing ? '❚❚' : '▶'}
      </button>
      <div class="time" aria-live="off">${formatHHMMSS(s.simTime)}</div>
      <div class="track">
        <input
          type="range"
          aria-label="Time of day"
          min=${t.report_start_s}
          max=${t.report_end_s}
          step="10"
          .value=${String(Math.round(s.simTime))}
          @input=${(e: Event) => store.set({ simTime: Number((e.target as HTMLInputElement).value) })}
        />
        <div class="ticks">${ticks.map((x) => html`<span>${formatHHMM(x)}</span>`)}</div>
      </div>
      <label
        >Speed
        <select aria-label="Playback speed" @change=${(e: Event) => store.set({ speed: Number((e.target as HTMLSelectElement).value) })}>
          ${SPEEDS.map((v) => html`<option value=${v} ?selected=${s.speed === v}>${v}x</option>`)}
        </select></label
      >
      <button class=${s.ghost ? 'active' : ''} @click=${() => store.set({ ghost: !s.ghost })} title="Glowing car trails">Ghost traffic</button>
      ${s.view === 'traffic' && s.planPlayback
        ? html`<select aria-label="Which scenario" @change=${(e: Event) => store.set({ trafficSource: (e.target as HTMLSelectElement).value as 'baseline' | 'plan' })}>
            <option value="baseline" ?selected=${s.trafficSource === 'baseline'}>Baseline</option>
            <option value="plan" ?selected=${s.trafficSource === 'plan'}>Plan</option>
          </select>`
        : nothing}
      ${s.view === 'report'
        ? html`<button class=${s.split ? 'active' : ''} ?disabled=${!s.planPlayback} @click=${() => store.set({ split: !s.split })} title="Left: baseline, right: plan">
            Before / after
          </button>`
        : nothing}
    </div>`;
  }
}
customElements.define('rd-scrubber', RdScrubber);

export class RdTrafficPanel extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        padding: 12px 14px;
      }
      .grad {
        height: 10px;
        border-radius: 5px;
        margin: 4px 0 2px;
      }
      .scale {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: var(--muted);
      }
      .sw {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 2px;
        margin-right: 6px;
        vertical-align: -1px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }
      td,
      th {
        padding: 3px 4px;
        text-align: left;
        border-bottom: 1px solid var(--line);
      }
      th {
        color: var(--muted);
        font-weight: 600;
      }
      td.n {
        text-align: right;
      }
    `,
  ];
  private st = new StoreController(this, ['schools', 'baselinePlayback', 'trafficSource', 'planPlayback', 'plan']);

  override render() {
    const s = this.st.s;
    const pb = s.trafficSource === 'plan' && s.planPlayback ? s.planPlayback : s.baselinePlayback;
    return html`<div class="panel p">
      <h2>Morning traffic</h2>
      <p class="muted small">
        ${s.trafficSource === 'plan' && s.planPlayback ? html`Plan: <b style="color:var(--plan)">${s.plan?.title ?? ''}</b>` : html`<b style="color:var(--base)">Baseline</b>`}
        playback${pb ? html`, ${pb.nTrajectories.toLocaleString('en-US')} sampled vehicles` : html` loading…`}${pb?.header.synthetic ? ' (synthetic)' : ''}.
      </p>
      <h3>Roads: volume / capacity</h3>
      <div class="grad" style="background:${gradient(VC_STOPS)}"></div>
      <div class="scale"><span>0</span><span>0.5</span><span>0.75</span><span>1.0</span><span>1.3+</span></div>
      <h3>Cars: speed</h3>
      <div class="grad" style="background:${gradient(SPEED_STOPS)}"></div>
      <div class="scale"><span>stopped</span><span>30 km/h</span><span>70+ km/h</span></div>
      <p class="small" style="margin-top:8px">
        <span class="sw" style="background:#ffc21a"></span>School bus <span class="sw" style="background:var(--base);margin-left:10px"></span>Shuttle
      </p>
      <h3>Drop-off queues (baseline)</h3>
      <p class="muted small">Bars at school entrances rise with the queue length in cars.</p>
      <table>
        <tr><th>Entrance</th><th class="n">Max queue</th><th class="n">Spillback</th></tr>
        ${s.schools.flatMap((sc) =>
          sc.entrances.map(
            (e) => html`<tr>
              <td>${sc.name}<span class="muted"> · ${humanize(e.id)}</span></td>
              <td class="n">${e.baseline ? `${fmtNum(e.baseline.max_queue_cars)} cars` : '-'}</td>
              <td class="n">${e.baseline ? `${fmtNum(e.baseline.max_spillback_m)} m` : '-'}</td>
            </tr>`,
          ),
        )}
      </table>
    </div>`;
  }
}
customElements.define('rd-traffic-panel', RdTrafficPanel);

export class RdSplitSlider extends LitElement {
  static override styles = [
    theme,
    css`
      :host {
        position: absolute;
        inset: 0;
        pointer-events: none;
      }
      .handle {
        position: absolute;
        top: 0;
        bottom: 0;
        width: 24px;
        margin-left: -12px;
        cursor: ew-resize;
        pointer-events: auto;
        touch-action: none;
      }
      .line {
        position: absolute;
        left: 11px;
        top: 0;
        bottom: 0;
        width: 2px;
        background: #fff;
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.6);
      }
      .knob {
        position: absolute;
        top: 50%;
        left: 0;
        width: 24px;
        height: 40px;
        margin-top: -20px;
        border-radius: 12px;
        background: #fff;
        color: #111;
        display: grid;
        place-items: center;
        font-size: 12px;
        font-weight: 700;
      }
      .lbl {
        position: absolute;
        top: 120px;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
        white-space: nowrap;
      }
      .l {
        right: 20px;
        background: var(--base);
        color: #04121f;
      }
      .r {
        left: 20px;
        background: var(--plan);
        color: #1a0c13;
      }
    `,
  ];
  private st = new StoreController(this, ['splitPos', 'split', 'view', 'planPlayback']);

  private onDown = (e: PointerEvent): void => {
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent): void => {
      const rect = this.getBoundingClientRect();
      store.set({ splitPos: Math.min(0.95, Math.max(0.05, (ev.clientX - rect.left) / rect.width)) });
    };
    const up = (): void => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
  };

  override render() {
    const s = this.st.s;
    if (!(s.view === 'report' && s.split && s.planPlayback)) return nothing;
    return html`<div class="handle" style="left:${(s.splitPos * 100).toFixed(2)}%" @pointerdown=${this.onDown} role="slider"
      aria-label="Before and after split" aria-valuenow=${Math.round(s.splitPos * 100)}>
      <div class="line"></div>
      <div class="knob">⇔</div>
      <div class="lbl l">◀ Baseline</div>
      <div class="lbl r">Plan ▶</div>
    </div>`;
  }
}
customElements.define('rd-split-slider', RdSplitSlider);
