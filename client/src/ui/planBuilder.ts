/**
 * Plan builder: tool palette and data-driven tool forms (left), plan summary
 * with the live server-side budget check, save and run (right).
 */

import { LitElement, css, html, nothing, type TemplateResult } from 'lit';
import { addTool, isDraftDirty, newDraft, removeTool, runPlan, savePlan, setDraftText, updateParam } from '../actions';
import { hashFor } from '../router';
import { store } from '../state';
import type { ToolDef, ToolInstance } from '../types';
import { StoreController, fmtUsd, humanize, theme } from './base';
import { buildFields, coerceInput, mapSummary, type FieldSpec } from './formgen';

const MAP_HINT: Record<string, string> = {
  point: 'Click the map to place (snaps to the nearest road).',
  points: 'Click the map to add stops (snap to roads). Esc or Done to finish.',
  polyline: 'Click the map to add route points (snap to roads). Esc or Done to finish.',
  edge: 'Click a road on the map.',
  node: 'Click an intersection on the map.',
};

export class RdToolPalette extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        padding: 12px;
        display: flex;
        flex-direction: column;
        max-height: 100%;
        box-sizing: border-box;
      }
      .list {
        overflow-y: auto;
        margin: 0 -4px;
        padding: 0 4px;
      }
      .tool {
        background: var(--bg-2);
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 6px;
      }
      .tool:hover {
        border-color: var(--line);
      }
      .tool .name {
        font-weight: 650;
      }
      .tool .desc {
        font-size: 12px;
        color: var(--muted);
        margin: 2px 0 4px;
      }
      .field {
        margin: 10px 0;
      }
      .field label.l {
        display: block;
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 3px;
      }
      .req {
        color: var(--plan);
      }
      .mapbox {
        background: rgba(0, 0, 0, 0.25);
        border: 1px dashed var(--line);
        border-radius: 8px;
        padding: 6px 8px;
      }
      .mapbox.picking {
        border-color: var(--plan);
        background: rgba(255, 95, 162, 0.1);
      }
      .cat {
        margin-top: 10px;
      }
      .check {
        display: flex;
        gap: 6px;
        align-items: center;
      }
    `,
  ];
  private st = new StoreController(this, ['tools', 'draft', 'selectedTool', 'schools', 'mapPick', 'check']);

  private renderList(): TemplateResult {
    const tools = this.st.s.tools;
    const cats: string[] = [];
    for (const t of tools) if (!cats.includes(t.category)) cats.push(t.category);
    return html`<h2>Tools</h2>
      <p class="muted small">Add tools to your plan. Every tool and its options come from the mission definition.</p>
      <div class="list scroll">
        ${cats.map(
          (c) => html`<h3 class="cat">${humanize(c)}</h3>
            ${tools
              .filter((t) => t.category === c)
              .map((t) => {
                const disabled = t.enabled_in_mvp === false;
                return html`<div class="tool" data-tool=${t.id}>
                  <div class="row"><span class="name">${t.name}</span><span class="spacer"></span>
                    <button class=${disabled ? '' : 'primary'} ?disabled=${disabled} @click=${() => addTool(t)} title=${disabled ? 'Not available in this version' : 'Add to plan'}>
                      ${disabled ? 'Later' : 'Add'}
                    </button></div>
                  <div class="desc">${t.description}</div>
                  ${t.cost_note ? html`<div class="small muted">Cost: ${t.cost_note}</div>` : nothing}
                </div>`;
              })}`,
        )}
      </div>`;
  }

  private renderField(idx: number, f: FieldSpec): TemplateResult {
    const onInput = (raw: string | boolean): void => updateParam(idx, f.id, coerceInput(f, raw));
    const label = html`<label class="l" for=${`f-${f.id}`}>${f.label}${f.required ? html` <span class="req">*</span>` : nothing}</label>`;
    switch (f.kind) {
      case 'select':
        return html`<div class="field">${label}
          <select id=${`f-${f.id}`} @change=${(e: Event) => onInput((e.target as HTMLSelectElement).value)}>
            ${f.options.map((o) => html`<option value=${o.value} ?selected=${String(f.value ?? '') === o.value}>${o.label}</option>`)}
          </select></div>`;
      case 'number':
        return html`<div class="field">${label}
          <input id=${`f-${f.id}`} type="number" .value=${f.value === null ? '' : String(f.value)} min=${f.min ?? ''} max=${f.max ?? ''} step=${f.step}
            @change=${(e: Event) => onInput((e.target as HTMLInputElement).value)} />
          ${f.min !== undefined || f.max !== undefined ? html`<div class="small muted">${f.min ?? ''} to ${f.max ?? ''}</div>` : nothing}</div>`;
      case 'time':
        return html`<div class="field">${label}
          <input id=${`f-${f.id}`} type="time" .value=${String(f.value ?? '')} min=${f.min ?? ''} max=${f.max ?? ''}
            @change=${(e: Event) => onInput((e.target as HTMLInputElement).value)} />
          ${f.min || f.max ? html`<div class="small muted">${f.min ?? ''} to ${f.max ?? ''}</div>` : nothing}</div>`;
      case 'checkbox':
        return html`<div class="field check">
          <input id=${`f-${f.id}`} type="checkbox" .checked=${Boolean(f.value)} @change=${(e: Event) => onInput((e.target as HTMLInputElement).checked)} />
          <label for=${`f-${f.id}`}>${f.label}</label></div>`;
      case 'text':
        return html`<div class="field">${label}
          ${f.multiline
            ? html`<textarea id=${`f-${f.id}`} rows="4" maxlength=${f.maxLength ?? 1000} .value=${String(f.value ?? '')}
                @change=${(e: Event) => onInput((e.target as HTMLTextAreaElement).value)}></textarea>`
            : html`<input id=${`f-${f.id}`} type="text" maxlength=${f.maxLength ?? 200} .value=${String(f.value ?? '')}
                @change=${(e: Event) => onInput((e.target as HTMLInputElement).value)} />`}</div>`;
      case 'map': {
        const pick = this.st.s.mapPick;
        const picking = pick?.toolIndex === idx && pick.paramId === f.id;
        const multi = f.mapType === 'points' || f.mapType === 'polyline';
        const arr = Array.isArray(f.value) ? (f.value as number[][]) : [];
        return html`<div class="field">${label}
          <div class="mapbox ${picking ? 'picking' : ''}">
            <div class="row"><span>${mapSummary(f.mapType, f.value)}${multi && f.maxCount ? html`<span class="muted"> (max ${f.maxCount})</span>` : nothing}</span></div>
            ${picking ? html`<div class="small" style="color:var(--plan)">${MAP_HINT[f.mapType]}</div>` : nothing}
            <div class="row" style="margin-top:6px">
              <button class=${picking ? 'active' : ''} data-pick=${f.id}
                @click=${() => store.set({ mapPick: picking ? null : { toolIndex: idx, paramId: f.id, type: f.mapType, maxCount: f.maxCount } })}>
                ${picking ? 'Done' : multi ? 'Add on map' : 'Pick on map'}</button>
              ${multi && arr.length ? html`<button @click=${() => updateParam(idx, f.id, arr.slice(0, -1))}>Undo</button>` : nothing}
              ${f.value !== null && (!multi || arr.length) ? html`<button class="ghost" @click=${() => updateParam(idx, f.id, multi ? [] : null)}>Clear</button>` : nothing}
            </div>
          </div></div>`;
      }
      default:
        return html`<div class="field">${label}<div class="err small">Unsupported parameter type "${f.type}"</div></div>`;
    }
  }

  private renderEditor(idx: number, inst: ToolInstance, def: ToolDef): TemplateResult {
    const fields = buildFields(def, inst.params, { schools: this.st.s.schools });
    const errs = (this.st.s.check?.errors ?? []).filter((e) => e.startsWith(`${def.id}:`) || e.startsWith(`tool ${idx + 1}`));
    return html`<div class="row">
        <button class="ghost" @click=${() => store.set({ selectedTool: null, mapPick: null })}>← Tools</button><span class="spacer"></span>
        <button class="ghost err" @click=${() => removeTool(idx)}>Remove</button>
      </div>
      <h2 style="margin-top:6px">${def.name}</h2>
      <p class="muted small">${def.description}</p>
      <div class="scroll" style="overflow-y:auto">
        ${fields.map((f) => this.renderField(idx, f))}
        ${errs.length ? html`<div class="small err">${errs.map((e) => html`<div>${e}</div>`)}</div>` : nothing}
        ${def.cost_note ? html`<p class="small muted">Cost: ${def.cost_note}</p>` : nothing}
      </div>`;
  }

  override render() {
    const s = this.st.s;
    const idx = s.selectedTool;
    const inst = idx !== null ? s.draft.tools[idx] : undefined;
    const def = inst ? s.tools.find((t) => t.id === inst.tool) : undefined;
    return html`<div class="panel p">${idx !== null && inst && def ? this.renderEditor(idx, inst, def) : this.renderList()}</div>`;
  }
}
customElements.define('rd-tool-palette', RdToolPalette);

function budgetBar(label: string, cost: number, budget: number): TemplateResult {
  const frac = budget > 0 ? cost / budget : cost > 0 ? 2 : 0;
  const over = frac > 1;
  return html`<div class="budget">
    <div class="row small"><span>${label}</span><span class="spacer"></span>
      <b class=${over ? 'err' : ''}>${fmtUsd(cost)}</b><span class="muted">/ ${fmtUsd(budget)}</span></div>
    <div class="bar" role="meter" aria-valuemin="0" aria-valuemax=${budget} aria-valuenow=${cost} aria-label=${label}>
      <div class="fill ${over ? 'over' : ''}" style="width:${Math.min(100, frac * 100).toFixed(1)}%"></div>
    </div>
  </div>`;
}

export class RdPlanSummary extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        max-height: 100%;
        box-sizing: border-box;
      }
      .budget .bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 3px;
      }
      .fill {
        height: 100%;
        background: var(--plan);
        border-radius: 4px;
        transition: width 0.25s;
      }
      .fill.over {
        background: var(--bad);
      }
      .inst {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 8px;
        border-radius: 8px;
        background: var(--bg-2);
        margin-bottom: 4px;
        cursor: pointer;
        border: 1px solid transparent;
      }
      .inst.sel {
        border-color: var(--plan);
      }
      .inst .sum {
        font-size: 11px;
        color: var(--muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 230px;
      }
      .progress {
        height: 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.1);
        overflow: hidden;
      }
      .progress div {
        height: 100%;
        background: linear-gradient(90deg, var(--plan), #ffa3cb);
        transition: width 0.4s;
      }
      .msgs {
        font-size: 12px;
      }
      .link {
        word-break: break-all;
        font-size: 12px;
      }
      a {
        color: var(--plan);
      }
    `,
  ];
  private st = new StoreController(this, ['draft', 'check', 'checking', 'plan', 'job', 'tools', 'selectedTool', 'meta']);

  private summary(inst: ToolInstance, def: ToolDef | undefined): string {
    if (!def) return '';
    return def.params
      .map((p) => {
        const v = inst.params[p.id];
        if (v === undefined || v === null || v === '') return null;
        if (['point', 'points', 'polyline', 'edge', 'node'].includes(p.type)) return `${p.label}: ${mapSummary(p.type, v)}`;
        if (p.type === 'select') return `${p.label}: ${p.options?.find((o) => o.value === v)?.label ?? String(v)}`;
        if (p.type === 'school') return this.st.s.schools.find((sc) => sc.id === v)?.name ?? String(v);
        return `${p.label}: ${String(v)}`;
      })
      .filter(Boolean)
      .join(' · ');
  }

  override render() {
    const s = this.st.s;
    const c = s.check;
    const mission = s.meta?.mission;
    const job = s.job;
    const dirty = isDraftDirty();
    const saved = s.plan && !dirty;
    const link = s.plan ? `${location.origin}${location.pathname}${location.search}${hashFor({ view: 'plan', planId: s.plan.id })}` : '';
    return html`<div class="panel p scroll">
      <div class="row"><h2 style="margin:0">Your plan</h2><span class="spacer"></span>
        <button class="ghost small" @click=${() => newDraft()} title="Start a new empty plan">New</button></div>
      <input type="text" placeholder="Plan title" maxlength="120" .value=${s.draft.title} @input=${(e: Event) => setDraftText('title', (e.target as HTMLInputElement).value)} aria-label="Plan title" />
      <textarea rows="2" placeholder="One-line pitch" maxlength="400" .value=${s.draft.pitch} @input=${(e: Event) => setDraftText('pitch', (e.target as HTMLTextAreaElement).value)} aria-label="Pitch"></textarea>

      <h3 style="margin:6px 0 2px">Tools (${s.draft.tools.length})</h3>
      ${s.draft.tools.length === 0 ? html`<p class="muted small">No tools yet. Add one from the palette on the left.</p>` : nothing}
      <div>
        ${s.draft.tools.map((inst, i) => {
          const def = s.tools.find((t) => t.id === inst.tool);
          return html`<div class="inst ${s.selectedTool === i ? 'sel' : ''}" @click=${() => store.set({ selectedTool: i, mapPick: null })}>
            <div style="min-width:0"><div>${def?.name ?? inst.tool}</div><div class="sum">${this.summary(inst, def)}</div></div>
            <span class="spacer"></span>
            <button class="ghost" aria-label="Remove tool" @click=${(e: Event) => (e.stopPropagation(), removeTool(i))}>✕</button>
          </div>`;
        })}
      </div>

      <h3 style="margin:6px 0 0">Budget ${s.checking ? html`<span class="muted small">checking…</span>` : nothing}</h3>
      ${budgetBar('Upfront', c?.cost_upfront_usd ?? 0, c?.budget_upfront_usd ?? mission?.budget_usd_upfront ?? 0)}
      ${budgetBar('Per year', c?.cost_per_year_usd ?? 0, c?.budget_per_year_usd ?? mission?.budget_usd_per_year ?? 0)}
      ${c?.over_budget ? html`<div class="small warnc">Over budget: the plan can still run but will be marked over budget.</div>` : nothing}
      <div class="msgs">
        ${(c?.errors ?? []).map((e) => html`<div class="err">✕ ${e}</div>`)}
        ${(c?.constraint_violations ?? []).map((e) => html`<div class="err">Constraint: ${humanize(e)}</div>`)}
        ${(c?.warnings ?? []).map((e) => html`<div class="warnc">! ${e}</div>`)}
        ${c?.ok ? html`<div style="color:var(--good)">✓ Plan checks out on the server.</div>` : nothing}
      </div>
      <p class="muted small" style="margin:0">Costs and checks come from the simulation server.</p>

      <div class="row">
        <button ?disabled=${s.draft.tools.length === 0 || !!saved || !!job} @click=${() => void savePlan()}>${s.plan && dirty ? 'Save as new' : saved ? 'Saved' : 'Save'}</button>
        <span class="spacer"></span>
        <button class="primary" ?disabled=${!c?.ok || !!job} @click=${() => void runPlan()} title="Runs 20 seeds of plan and baseline">Run plan</button>
      </div>
      ${job
        ? html`<div>
            <div class="row small"><span>${humanize(job.status)}</span><span class="spacer"></span><span class="muted">${job.message}</span></div>
            <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${Math.round(job.progress * 100)}>
              <div style="width:${(job.progress * 100).toFixed(1)}%"></div>
            </div>
          </div>`
        : nothing}
      ${s.plan
        ? html`<div class="link muted">Link: <a href=${hashFor({ view: 'plan', planId: s.plan.id })}>${link}</a>
            ${s.plan.status === 'done' ? html`<div><a href="#/report" @click=${(e: Event) => (e.preventDefault(), store.set({ view: 'report' }))}>View report card →</a></div>` : nothing}</div>`
        : nothing}
    </div>`;
  }
}
customElements.define('rd-plan-summary', RdPlanSummary);
