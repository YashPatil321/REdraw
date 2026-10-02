/** Talk to a resident (spec 8.3) and the virtual town hall (spec 8.4). */

import { LitElement, css, html, nothing, type TemplateResult } from 'lit';
import { closeChat, loadTownhall, openChat, respondTownhall, sendChat } from '../actions';
import { store } from '../state';
import type { Reaction, TownhallSpeaker } from '../types';
import { StoreController, appCtx, fmtSigned, theme } from './base';

/** 0 -> red, 0.5 -> amber, 1 -> green */
export function approvalColor(a: number): string {
  const c =
    a < 0.5
      ? [255, Math.round(107 + (196 - 107) * (a / 0.5)), 90]
      : [Math.round(255 - (255 - 76) * ((a - 0.5) / 0.5)), Math.round(196 + (211 - 196) * ((a - 0.5) / 0.5)), Math.round(77 + (138 - 77) * ((a - 0.5) / 0.5))];
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/** The personal changes the server computed for this resident, as one line. */
export function residentDeltas(r: Reaction): string {
  const parts: string[] = [];
  const d = r.deltas;
  if (typeof d.commute_min === 'number') parts.push(`commute ${fmtSigned(d.commute_min)} min`);
  if (typeof d.dropoff_min === 'number') parts.push(`drop-off ${fmtSigned(d.dropoff_min)} min`);
  if (typeof d.cost_usd_year === 'number' && d.cost_usd_year !== 0) parts.push(`cost ${fmtSigned(d.cost_usd_year)} $/yr`);
  if (d.street_change) parts.push('street changes');
  return parts.join(' · ');
}

/** Fly to the resident's block and open the conversation. */
export function meetResident(r: Reaction): void {
  appCtx.scene?.showResident(r.x, r.z);
  void openChat(r);
}

const shared = css`
  .good {
    color: var(--good);
  }
  .bad {
    color: var(--bad);
  }
  .who {
    display: flex;
    align-items: baseline;
    gap: 6px;
    flex-wrap: wrap;
  }
  .ap {
    font-weight: 700;
  }
  .facts {
    font-size: 12px;
    color: var(--muted);
  }
  textarea {
    width: 100%;
    box-sizing: border-box;
    min-height: 54px;
    resize: vertical;
    background: var(--bg-2);
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 6px 8px;
    font: inherit;
  }
  .note {
    font-size: 12px;
    color: var(--muted);
    background: var(--bg-2);
    border-radius: 8px;
    padding: 6px 8px;
    margin: 6px 0;
  }
`;

export class RdChat extends LitElement {
  static override styles = [
    theme,
    shared,
    css`
      .log {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin: 10px 0;
      }
      .msg {
        max-width: 85%;
        padding: 6px 9px;
        border-radius: 10px;
        white-space: pre-wrap;
      }
      .msg.user {
        align-self: flex-end;
        background: rgba(255, 95, 162, 0.22);
        border: 1px solid rgba(255, 95, 162, 0.45);
      }
      .msg.assistant {
        align-self: flex-start;
        background: var(--bg-2);
        border: 1px solid var(--line);
      }
      .typing {
        align-self: flex-start;
        color: var(--muted);
        font-style: italic;
        font-size: 12px;
      }
    `,
  ];
  private st = new StoreController(this, ['chat', 'residents', 'plan']);
  private draft = '';

  private send(): void {
    const text = this.draft;
    if (!text.trim()) return;
    this.draft = '';
    const ta = this.renderRoot.querySelector('textarea');
    if (ta) ta.value = '';
    void sendChat(text);
  }

  override updated(): void {
    const log = this.renderRoot.querySelector('.log');
    log?.lastElementChild?.scrollIntoView({ block: 'nearest' });
  }

  override render() {
    const c = this.st.s.chat;
    if (!c) return nothing;
    const r = c.resident;
    const llmOff = this.st.s.residents?.llm_available === false;
    return html`<div>
      <button class="small" @click=${() => closeChat()}>← All residents</button>
      <div class="who" style="margin-top:8px">
        <b style="font-size:15px">${r.first_name}</b>
        <span class="muted small">${r.age} · ${r.block}</span>
        <span class="spacer"></span>
        <span class="ap small" style="color:${approvalColor(r.approval)}">${r.approves ? 'Approves' : 'Opposes'} ${(r.approval * 100).toFixed(0)}%</span>
      </div>
      <div class="facts">${residentDeltas(r)}${r.values?.length ? html` · values: ${r.values.join(', ')}` : nothing}</div>
      ${r.text ? html`<p style="font-style:italic;margin:6px 0">“${r.text}”</p>` : nothing}
      <div class="note">
        Ask ${r.first_name} about this plan. They answer in character and only quote the numbers the simulation gave them.
        Their name is generated and their home is shown at block level only.
      </div>
      ${llmOff ? html`<div class="note warnc">The AI endpoint is not configured on this server, so residents cannot talk yet. Their approval above is still computed from the simulation.</div>` : nothing}
      <div class="log" aria-live="polite">
        ${c.loading ? html`<span class="typing">Loading conversation…</span>` : nothing}
        ${c.messages.map((m) => html`<div class="msg ${m.role}">${m.content}</div>`)}
        ${c.sending ? html`<span class="typing">${r.first_name} is thinking…</span>` : nothing}
      </div>
      ${c.error ? html`<p class="err small">${c.error}</p>` : nothing}
      <textarea
        placeholder="e.g. Would you drop your kids at the new entrance?"
        maxlength="1000"
        ?disabled=${c.sending || llmOff}
        @input=${(e: Event) => (this.draft = (e.target as HTMLTextAreaElement).value)}
        @keydown=${(e: KeyboardEvent) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.send();
          }
        }}
      ></textarea>
      <div class="row" style="margin-top:6px">
        <span class="muted small">Enter to send, Shift+Enter for a new line</span><span class="spacer"></span>
        <button class="primary" ?disabled=${c.sending || llmOff} @click=${() => this.send()}>Send</button>
      </div>
    </div>`;
  }
}
customElements.define('rd-chat', RdChat);

export class RdTownhall extends LitElement {
  static override styles = [
    theme,
    shared,
    css`
      .cols {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }
      .col h3 {
        margin: 8px 0 4px;
      }
      .card {
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-left-width: 4px;
        border-radius: 8px;
        padding: 7px 9px;
        margin-bottom: 8px;
        cursor: pointer;
      }
      .card.sel {
        outline: 2px solid var(--plan);
      }
      .comment {
        margin: 4px 0;
      }
      .ex {
        border-top: 1px solid var(--line);
        padding: 6px 0;
      }
      .you {
        color: var(--plan);
      }
    `,
  ];
  private st = new StoreController(this, ['townhall', 'plan', 'residents']);
  private target: number | null = null;
  private draft = '';

  override connectedCallback(): void {
    super.connectedCallback();
    const s = store.get();
    if (s.plan?.status === 'done' && !s.townhall) void loadTownhall();
  }

  private card(sp: TownhallSpeaker): TemplateResult {
    const color = approvalColor(sp.approval);
    return html`<div
      class="card ${this.target === sp.persona_id ? 'sel' : ''}"
      style="border-left-color:${color}"
      title="Select to respond, double-click to fly to their block"
      @click=${() => {
        this.target = sp.persona_id;
        this.requestUpdate();
      }}
      @dblclick=${() => appCtx.scene?.showResident(sp.x, sp.z)}
    >
      <div class="who"><b>${sp.first_name}</b><span class="muted small">${sp.age} · ${sp.block}</span></div>
      ${sp.comment ? html`<div class="comment">“${sp.comment}”</div>` : nothing}
      <div class="facts">${residentDeltas(sp)}</div>
      <div class="row small"><span class="ap" style="color:${color}">${sp.approves ? 'Supports' : 'Opposes'} ${(sp.approval * 100).toFixed(0)}%</span>
        <span class="spacer"></span><a style="color:var(--plan);cursor:pointer" @click=${(e: Event) => (e.stopPropagation(), openChat(sp))}>Talk 1:1</a></div>
    </div>`;
  }

  private respond(): void {
    if (this.target === null || !this.draft.trim()) return;
    const text = this.draft;
    this.draft = '';
    const ta = this.renderRoot.querySelector('textarea');
    if (ta) ta.value = '';
    void respondTownhall(this.target, text);
  }

  override render() {
    const th = this.st.s.townhall;
    if (!th || (th.loading && !th.data)) return html`<p class="muted">Convening the town hall…</p>`;
    if (!th.data) {
      return html`<p class="err small">${th.error ?? 'The town hall could not be convened.'}</p>
        <button @click=${() => loadTownhall()}>Try again</button>`;
    }
    const sp = th.data.speakers;
    const against = sp.filter((s) => s.side === 'against');
    const support = sp.filter((s) => s.side === 'for');
    const byId = new Map(sp.map((s) => [s.persona_id, s]));
    const target = this.target !== null ? byId.get(this.target) : undefined;
    return html`<p class="small muted">
        The ${sp.length} residents most affected by this plan came to speak: the ${against.length} hit hardest and the ${support.length} who gain the most,
        drawn from at least three school communities. Positions come from each resident's simulated changes; statements are AI written.
      </p>
      ${th.data.llm_available === false
        ? html`<div class="note">The AI endpoint is not configured on this server, so speakers show their numbers instead of statements.</div>`
        : nothing}
      <div class="cols">
        <div class="col"><h3 class="bad">Against</h3>${against.map((s) => this.card(s))}</div>
        <div class="col"><h3 class="good">For</h3>${support.map((s) => this.card(s))}</div>
      </div>
      ${th.exchanges.length
        ? html`<h3>Your exchanges</h3>
            ${th.exchanges.map((x) => {
              const who = byId.get(x.persona_id);
              return html`<div class="ex">
                <div class="small"><span class="you">You → ${who?.first_name ?? 'speaker'}:</span> ${x.message}</div>
                <div>${x.text ? html`<b>${who?.first_name ?? 'Speaker'}:</b> “${x.text}”` : html`<span class="muted small">No reply (AI offline or reply rejected).</span>`}</div>
              </div>`;
            })}`
        : nothing}
      <h3>Respond as the planner</h3>
      <p class="small muted">${target ? html`Responding to <b>${target.first_name}</b>.` : 'Click a speaker card to choose who you answer.'}</p>
      <textarea
        placeholder="Explain your plan or answer their concern…"
        maxlength="1000"
        ?disabled=${th.responding || !target}
        @input=${(e: Event) => (this.draft = (e.target as HTMLTextAreaElement).value)}
      ></textarea>
      ${th.error ? html`<p class="err small">${th.error}</p>` : nothing}
      <div class="row" style="margin-top:6px">
        <button class="small" ?disabled=${th.loading} @click=${() => loadTownhall(true)}>New meeting</button>
        <span class="spacer"></span>
        <button class="primary" ?disabled=${th.responding || !target} @click=${() => this.respond()}>${th.responding ? 'Waiting for reply…' : 'Respond'}</button>
      </div>`;
  }
}
customElements.define('rd-townhall', RdTownhall);
