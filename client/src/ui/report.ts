/** Report card (baseline vs plan with uncertainty ranges) and residents feed. */

import { LitElement, css, html, nothing, svg, type TemplateResult } from 'lit';
import { navigate } from '../actions';
import { store } from '../state';
import { formatHHMM } from '../time';
import { vcColor, rgbToCss } from '../traffic/congestion';
import type { BaselinePlan, MaybeRange3, MetricDef, Reaction, Report } from '../types';
import { StoreController, appCtx, fmtNum, fmtSigned, fmtUsd, humanize, theme } from './base';

const W = 210;
const H = 34;

function complete(r: MaybeRange3 | null | undefined): r is { median: number; p10: number; p90: number } {
  return !!r && Number.isFinite(r.median) && Number.isFinite(r.p10) && Number.isFinite(r.p90);
}

/** Two horizontal p10-p90 bars with whiskers and a median tick (baseline top, plan bottom). Null ranges are skipped. */
export function rangeChart(b: MaybeRange3, p: MaybeRange3): TemplateResult {
  const rows = [b, p].filter(complete);
  if (!rows.length) return html`<svg width=${W} height=${H} aria-hidden="true"></svg>`;
  let lo = Math.min(...rows.flatMap((r) => [r.p10, r.p90, r.median]));
  let hi = Math.max(...rows.flatMap((r) => [r.p10, r.p90, r.median]));
  if (!(hi > lo)) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * 0.08;
  lo -= pad;
  hi += pad;
  const x = (v: number): number => ((v - lo) / (hi - lo)) * (W - 8) + 4;
  const row = (r: MaybeRange3, y: number, color: string) => {
    if (!complete(r)) return svg`<text x="4" y=${y + 4} font-size="10" fill="var(--muted)">n/a</text>`;
    return svg`
    <line x1=${x(r.p10)} x2=${x(r.p90)} y1=${y} y2=${y} stroke=${color} stroke-width="1.5" />
    <line x1=${x(r.p10)} x2=${x(r.p10)} y1=${y - 4} y2=${y + 4} stroke=${color} stroke-width="1.5" />
    <line x1=${x(r.p90)} x2=${x(r.p90)} y1=${y - 4} y2=${y + 4} stroke=${color} stroke-width="1.5" />
    <rect x=${Math.min(x(r.p10), x(r.p90))} y=${y - 3} width=${Math.max(1, Math.abs(x(r.p90) - x(r.p10)))} height="6" rx="2" fill=${color} opacity="0.45" />
    <line x1=${x(r.median)} x2=${x(r.median)} y1=${y - 6} y2=${y + 6} stroke="#fff" stroke-width="2.5" />`;
  };
  return html`<svg width=${W} height=${H} viewBox="0 0 ${W} ${H}" aria-hidden="true">
    ${row(b, 10, 'var(--base)')} ${row(p, 25, 'var(--plan)')}
  </svg>`;
}

function deltaClass(delta: number | null | undefined, better: string): string {
  if (delta === null || delta === undefined || !Number.isFinite(delta) || Math.abs(delta) < 1e-9) return 'muted';
  const good = better === 'higher' ? delta > 0 : delta < 0;
  return good ? 'good' : 'bad';
}

export class RdReport extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        padding: 12px 14px;
        max-height: 100%;
        box-sizing: border-box;
      }
      .tabs {
        display: flex;
        gap: 4px;
        margin: 8px 0;
      }
      .metric {
        display: grid;
        grid-template-columns: 1fr ${W}px;
        gap: 2px 10px;
        padding: 7px 0;
        border-bottom: 1px solid var(--line);
        align-items: center;
      }
      .metric .vals {
        font-size: 12px;
      }
      .b {
        color: var(--base);
      }
      .pl {
        color: var(--plan);
      }
      .good {
        color: var(--good);
      }
      .bad {
        color: var(--bad);
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }
      th,
      td {
        padding: 4px 4px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
      }
      th {
        color: var(--muted);
        font-weight: 600;
      }
      td.n,
      th.n {
        text-align: right;
        white-space: nowrap;
      }
      .legend {
        display: flex;
        gap: 12px;
        font-size: 12px;
      }
      .sw {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 2px;
        margin-right: 4px;
        vertical-align: -1px;
      }
      .footer {
        margin-top: 14px;
        padding-top: 8px;
        border-top: 1px solid var(--line);
        font-size: 12px;
        color: var(--muted);
      }
      .footer ul {
        margin: 2px 0 6px;
        padding-left: 16px;
      }
      .se {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 0;
        border-bottom: 1px solid var(--line);
      }
      .vc {
        display: inline-block;
        padding: 0 5px;
        border-radius: 4px;
        color: #111;
        font-weight: 700;
        font-size: 11px;
      }
      .mshare {
        display: grid;
        grid-template-columns: 110px 1fr 56px;
        gap: 2px 8px;
        align-items: center;
        font-size: 12px;
        margin: 3px 0;
      }
      .mbar {
        position: relative;
        height: 14px;
      }
      .mbar div {
        position: absolute;
        left: 0;
        height: 6px;
        border-radius: 3px;
      }
      .big {
        font-size: 18px;
        font-weight: 700;
      }
      .wl {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }
      .wl > div {
        background: var(--bg-2);
        border-radius: 8px;
        padding: 6px 8px;
      }
      .progress {
        height: 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.1);
        overflow: hidden;
        margin-top: 6px;
      }
      .progress div {
        height: 100%;
        background: var(--plan);
      }
    `,
  ];
  private st = new StoreController(this, ['plan', 'residents', 'reportTab', 'meta', 'tools', 'job', 'split', 'planPlayback', 'highlightEdge']);

  private metricLabel(id: string): MetricDef | undefined {
    const metas = this.st.s.meta?.metrics ?? [];
    return metas.find((m) => m.id === id) ?? metas.find((m) => m.id === `avg_${id}`) ?? metas.find((m) => m.id.endsWith(id));
  }

  private renderMetrics(r: Report): TemplateResult {
    return html`<div class="legend"><span><span class="sw" style="background:var(--base)"></span>Baseline</span>
        <span><span class="sw" style="background:var(--plan)"></span>Plan</span><span class="muted">bar: p10–p90, tick: median</span></div>
      ${r.metrics.map(
        (m) => html`<div class="metric" data-metric=${m.id}>
          <div>
            <div>${m.label} <span class="muted small">(${m.unit})</span></div>
            <div class="vals">
              <span class="b">${fmtNum(m.baseline.median, m.unit)}</span> →
              <span class="pl">${fmtNum(m.plan.median, m.unit)}</span>
              ${complete(m.delta)
                ? html`<span class=${deltaClass(m.delta.median, m.better)}>(${fmtSigned(m.delta.median, m.unit)}; ${fmtSigned(m.delta.p10, m.unit)} to ${fmtSigned(m.delta.p90, m.unit)})</span>`
                : nothing}
            </div>
            ${m.note ? html`<div class="small muted">${m.note}</div>` : nothing}
          </div>
          ${rangeChart(m.baseline, m.plan)}
        </div>`,
      )}`;
  }

  private renderPerSchool(r: Report): TemplateResult | typeof nothing {
    if (!r.per_school?.length) return nothing;
    const keys: string[] = [];
    for (const row of r.per_school) for (const [k, v] of Object.entries(row)) if (typeof v === 'object' && v && !keys.includes(k)) keys.push(k);
    return html`<h3>Per school</h3>
      <table>
        <tr>
          <th>School</th>
          ${keys.map((k) => {
            const d = this.metricLabel(k);
            return html`<th class="n">${d?.label ?? humanize(k)}${d?.unit ? html` <span class="muted">(${d.unit})</span>` : nothing}</th>`;
          })}
        </tr>
        ${r.per_school.map(
          (row) => html`<tr>
            <td>${row.name}</td>
            ${keys.map((k) => {
              const v = row[k] as BaselinePlan | undefined;
              if (!v || typeof v !== 'object') return html`<td class="n">-</td>`;
              const d = v.plan.median - v.baseline.median;
              return html`<td class="n"><span class="b">${fmtNum(v.baseline.median)}</span> → <span class="pl">${fmtNum(v.plan.median)}</span>
                <div class=${deltaClass(d, this.metricLabel(k)?.better ?? 'lower')}>${fmtSigned(d)}</div></td>`;
            })}
          </tr>`,
        )}
      </table>`;
  }

  private renderModeShare(r: Report): TemplateResult | typeof nothing {
    if (!r.mode_share?.length) return nothing;
    const max = Math.max(1, ...r.mode_share.flatMap((m) => [m.baseline_pct, m.plan_pct]));
    return html`<h3>Mode share</h3>
      ${r.mode_share.map(
        (m) => html`<div class="mshare">
          <span>${m.label ?? humanize(m.mode)}</span>
          <div class="mbar" title="${m.baseline_pct.toFixed(1)}% → ${m.plan_pct.toFixed(1)}%">
            <div style="top:0;width:${(m.baseline_pct / max) * 100}%;background:var(--base)"></div>
            <div style="top:7px;width:${(m.plan_pct / max) * 100}%;background:var(--plan)"></div>
          </div>
          <span class="n ${Math.abs(m.delta_pp) < 0.05 ? 'muted' : ''}" style="text-align:right">${fmtSigned(m.delta_pp)} pp</span>
        </div>`,
      )}`;
  }

  private renderSideEffects(r: Report): TemplateResult {
    return html`<h3>Side effects</h3>
      ${r.side_effects?.length
        ? r.side_effects.map(
            (se) => html`<div class="se">
              <div style="flex:1">
                <div>${se.name || `Road edge #${se.edge_idx}`} <span class="muted small">at ${formatHHMM(se.bin_s)}</span></div>
                <div class="small">v/c <span class="vc" style="background:${rgbToCss(vcColor(se.baseline_vc))}">${se.baseline_vc.toFixed(2)}</span> →
                  <span class="vc" style="background:${rgbToCss(vcColor(se.plan_vc))}">${se.plan_vc.toFixed(2)}</span></div>
              </div>
              <button class=${this.st.s.highlightEdge === se.edge_idx ? 'active' : ''} @click=${() => {
                store.set({ simTime: se.bin_s + 150, playing: false });
                appCtx.scene?.showEdge(se.edge_idx, se.x, se.z);
              }}>Show on map</button>
            </div>`,
          )
        : html`<p class="muted small">No road went above v/c 0.9 because of this plan.</p>`}`;
  }

  private renderCost(r: Report): TemplateResult {
    const c = r.cost;
    const toolName = (id: string): string => this.st.s.tools.find((t) => t.id === id)?.name ?? id;
    return html`<h3>Cost</h3>
      <table>
        <tr><th>Tool</th><th class="n">Upfront</th><th class="n">Per year</th></tr>
        ${c.lines.map(
          (l) => html`<tr><td>${toolName(l.tool)}${l.note ? html`<div class="muted small">${l.note}</div>` : nothing}</td>
            <td class="n">${fmtUsd(l.upfront_usd)}</td><td class="n">${fmtUsd(l.per_year_usd)}</td></tr>`,
        )}
        <tr><td><b>Total</b></td><td class="n"><b>${fmtUsd(c.upfront_usd)}</b></td><td class="n"><b>${fmtUsd(c.per_year_usd)}</b></td></tr>
        <tr class="muted"><td>Budget</td><td class="n">${fmtUsd(c.budget_upfront_usd)}</td><td class="n">${fmtUsd(c.budget_per_year_usd)}</td></tr>
      </table>
      ${c.over_budget ? html`<p class="err small">Over budget.</p>` : nothing}
      ${r.constraint_violations?.length ? html`<p class="err small">Constraint violations: ${r.constraint_violations.map(humanize).join(', ')}</p>` : nothing}`;
  }

  private renderFooter(r: Report): TemplateResult {
    const meta = this.st.s.meta;
    const unverified = r.unverified_inputs?.length ? r.unverified_inputs : (meta?.unverified_inputs ?? []);
    return html`<div class="footer">
      ${r.synthetic || meta?.synthetic ? html`<p class="warnc"><b>SYNTHETIC DEV DATA:</b> this report was computed on a synthetic stand-in world, not real geography.</p>` : nothing}
      ${meta?.mock ? html`<p class="warnc"><b>MOCK FIXTURE:</b> numbers are placeholders from the offline mock, not simulation results.</p>` : nothing}
      <p>Calibration: <b>${humanize(r.calibration?.status ?? 'unknown')}</b>${r.calibration?.median_error_pct != null ? ` (median route error ${r.calibration.median_error_pct.toFixed(0)}%)` : ''}</p>
      ${r.llm_estimated_tools?.length ? html`<p>LLM estimated tools: ${r.llm_estimated_tools.join(', ')}</p>` : html`<p>No LLM estimated tools in this plan.</p>`}
      ${unverified.length
        ? html`<p>Unverified inputs used:</p>
            <ul>${unverified.map((u) => html`<li>${u}</li>`)}</ul>`
        : nothing}
      <p>${r.seeds} seeds per scenario; ranges are 10th to 90th percentile. Generated ${new Date(r.generated_at).toLocaleString()}.</p>
    </div>`;
  }

  private renderReport(r: Report): TemplateResult {
    return html`${this.renderMetrics(r)}
      <h3>Winners and losers</h3>
      <div class="wl">
        <div><span class="muted small">Better off by 3+ min</span><div class="big good">${fmtNum(r.winners.median)}</div><span class="small muted">${fmtNum(r.winners.p10)}–${fmtNum(r.winners.p90)}</span></div>
        <div><span class="muted small">Worse off by 3+ min</span><div class="big bad">${fmtNum(r.losers.median)}</div><span class="small muted">${fmtNum(r.losers.p10)}–${fmtNum(r.losers.p90)}</span></div>
      </div>
      ${r.peak_overlap
        ? html`<p class="small">${r.peak_overlap.note ? humanize(r.peak_overlap.note) : 'Peak overlap'}:
            <span class="b">${(r.peak_overlap.baseline * 100).toFixed(0)}%</span> → <span class="pl">${(r.peak_overlap.plan * 100).toFixed(0)}%</span></p>`
        : nothing}
      ${this.renderPerSchool(r)} ${this.renderModeShare(r)} ${this.renderSideEffects(r)} ${this.renderCost(r)} ${this.renderFooter(r)}`;
  }

  override render() {
    const s = this.st.s;
    const plan = s.plan;
    if (!plan) {
      return html`<div class="panel p">
        <h2>Report card</h2>
        <p class="muted">No plan open. Build and run a plan, or open one from Browse plans.</p>
        <div class="row"><button class="primary" @click=${() => navigate('plan')}>Plan builder</button><button @click=${() => navigate('browse')}>Browse plans</button></div>
      </div>`;
    }
    const r = plan.report;
    return html`<div class="panel p scroll">
      <div class="row"><span class="tag">Report card</span>${r?.synthetic ? html`<span class="tag warn">synthetic</span>` : nothing}<span class="spacer"></span>
        <a class="small" style="color:var(--plan)" href=${`#/plan/${encodeURIComponent(plan.id)}`} @click=${(e: Event) => (e.preventDefault(), store.set({ view: 'plan' }))}>Edit tools</a></div>
      <h2 style="margin-top:6px">${plan.title}</h2>
      ${plan.pitch ? html`<p class="muted">${plan.pitch}</p>` : nothing}
      ${!r
        ? html`<p class="muted">This plan has no report yet (status: ${plan.status}).</p>
            ${s.job ? html`<div class="small">${s.job.message}</div><div class="progress"><div style="width:${s.job.progress * 100}%"></div></div>` : nothing}`
        : html`<div class="tabs">
              <button class=${s.reportTab === 'report' ? 'active' : ''} @click=${() => store.set({ reportTab: 'report' })}>Report</button>
              <button class=${s.reportTab === 'residents' ? 'active' : ''} @click=${() => store.set({ reportTab: 'residents' })}>
                Residents${s.residents ? ` (${s.residents.approval_pct.toFixed(0)}% approve)` : ''}</button>
            </div>
            ${s.reportTab === 'residents' ? html`<rd-residents></rd-residents>` : this.renderReport(r)}`}
    </div>`;
  }
}
customElements.define('rd-report', RdReport);

function approvalColor(a: number): string {
  // 0 -> red, 0.5 -> amber, 1 -> green
  const c = a < 0.5 ? [255, Math.round(107 + (196 - 107) * (a / 0.5)), 90] : [Math.round(255 - (255 - 76) * ((a - 0.5) / 0.5)), Math.round(196 + (211 - 196) * ((a - 0.5) / 0.5)), Math.round(77 + (138 - 77) * ((a - 0.5) / 0.5))];
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export class RdResidents extends LitElement {
  static override styles = [
    theme,
    css`
      .r {
        display: grid;
        grid-template-columns: 6px 1fr;
        gap: 8px;
        padding: 8px 4px;
        border-bottom: 1px solid var(--line);
        cursor: pointer;
        border-radius: 6px;
      }
      .r:hover {
        background: var(--bg-2);
      }
      .stripe {
        border-radius: 3px;
      }
      .q {
        margin: 4px 0 2px;
        font-style: italic;
      }
      .none {
        color: var(--muted);
        font-style: normal;
        font-size: 12px;
      }
      .ap {
        font-weight: 700;
      }
    `,
  ];
  private st = new StoreController(this, ['residents', 'plan']);

  private deltas(r: Reaction): string {
    const parts: string[] = [];
    const d = r.deltas;
    if (typeof d.commute_min === 'number') parts.push(`commute ${fmtSigned(d.commute_min)} min`);
    if (typeof d.dropoff_min === 'number') parts.push(`drop-off ${fmtSigned(d.dropoff_min)} min`);
    if (typeof d.cost_usd_year === 'number' && d.cost_usd_year !== 0) parts.push(`cost ${fmtSigned(d.cost_usd_year)} $/yr`);
    if (d.street_change) parts.push('street changes');
    return parts.join(' · ');
  }

  override render() {
    const res = this.st.s.residents;
    if (!res) return html`<p class="muted">No resident reactions available for this plan yet.</p>`;
    return html`<p>Resident approval: <b class="ap" style="color:${approvalColor(res.approval_pct / 100)}">${res.approval_pct.toFixed(1)}%</b>
        <span class="muted small">(${res.reactions.length} residents; approval is computed from their personal changes, quotes are AI written)</span></p>
      ${res.reactions.map(
        (r) => html`<div class="r" @click=${() => appCtx.scene?.showResident(r.x, r.z)} title="Fly to their block" data-persona=${r.persona_id}>
          <div class="stripe" style="background:${approvalColor(r.approval)}"></div>
          <div>
            <div class="row"><b>${r.first_name}</b><span class="muted small">${r.age} · ${r.block}</span><span class="spacer"></span>
              <span class="ap small" style="color:${approvalColor(r.approval)}">${r.approves ? 'Approves' : 'Opposes'} ${(r.approval * 100).toFixed(0)}%</span></div>
            ${r.text ? html`<div class="q">“${r.text}”</div>` : html`<div class="q none">no quote — AI offline</div>`}
            <div class="small muted">${this.deltas(r)}${r.values?.length ? html` · values: ${r.values.join(', ')}` : nothing}</div>
          </div>
        </div>`,
      )}`;
  }
}
customElements.define('rd-residents', RdResidents);
