/** Browse plans: GET /plans sorted by new, votes or any metric. */

import { LitElement, css, html, nothing } from 'lit';
import { vote } from '../actions';
import { api } from '../api';
import { hashFor } from '../router';
import type { PlanListItem } from '../types';
import { StoreController, fmtNum, humanize, theme } from './base';

export class RdBrowse extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        padding: 14px 16px;
        max-height: 100%;
        box-sizing: border-box;
      }
      .item {
        display: grid;
        grid-template-columns: 44px 1fr auto;
        gap: 10px;
        padding: 10px 4px;
        border-bottom: 1px solid var(--line);
        align-items: center;
      }
      .votes {
        display: flex;
        flex-direction: column;
        align-items: center;
      }
      .votes button {
        padding: 0 6px;
        line-height: 1.2;
      }
      .title {
        font-weight: 650;
        color: var(--text);
        text-decoration: none;
        font-size: 14px;
      }
      .title:hover {
        color: var(--plan);
      }
      .hl {
        font-size: 12px;
        color: var(--muted);
      }
      .hl b {
        color: var(--text);
        font-weight: 600;
      }
      select {
        width: auto;
      }
    `,
  ];
  private st = new StoreController(this, ['meta', 'view']);
  private sort = 'new';
  private plans: PlanListItem[] = [];
  private loading = false;
  private error = '';

  override connectedCallback(): void {
    super.connectedCallback();
    void this.load();
  }

  private async load(): Promise<void> {
    this.loading = true;
    this.error = '';
    this.requestUpdate();
    try {
      const res = await api.listPlans({ mission: this.st.s.meta?.mission.id, sort: this.sort });
      this.plans = res.plans;
    } catch (e) {
      this.error = (e as Error).message;
    }
    this.loading = false;
    this.requestUpdate();
  }

  private async vote(p: PlanListItem, v: 1 | -1): Promise<void> {
    const n = await vote(p.id, v);
    if (n !== null) {
      p.votes = n;
      this.requestUpdate();
    }
  }

  override render() {
    const metrics = this.st.s.meta?.metrics ?? [];
    return html`<div class="panel p scroll">
      <div class="row">
        <h2 style="margin:0">Browse plans</h2><span class="spacer"></span>
        <label class="small muted" for="sort">Sort by</label>
        <select id="sort" @change=${(e: Event) => ((this.sort = (e.target as HTMLSelectElement).value), void this.load())}>
          <option value="new" ?selected=${this.sort === 'new'}>Newest</option>
          <option value="votes" ?selected=${this.sort === 'votes'}>Votes</option>
          ${metrics.map((m) => html`<option value=${m.id} ?selected=${this.sort === m.id}>${m.label}</option>`)}
        </select>
        <button @click=${() => void this.load()}>Refresh</button>
      </div>
      ${this.loading ? html`<p class="muted">Loading…</p>` : nothing}
      ${this.error ? html`<p class="err">${this.error}</p>` : nothing}
      ${!this.loading && !this.error && this.plans.length === 0 ? html`<p class="muted">No plans yet. Be the first: open the Plan builder.</p>` : nothing}
      ${this.plans.map((p) => {
        const hl = metrics.filter((m) => p.headline?.[m.id] !== undefined).slice(0, 4);
        return html`<div class="item">
          <div class="votes">
            <button class="ghost" aria-label="Upvote" @click=${() => void this.vote(p, 1)}>▲</button>
            <b>${p.votes}</b>
            <button class="ghost" aria-label="Downvote" @click=${() => void this.vote(p, -1)}>▼</button>
          </div>
          <div style="min-width:0">
            <a class="title" href=${hashFor({ view: 'plan', planId: p.id })}>${p.title || 'Untitled plan'}</a>
            ${p.pitch ? html`<div class="small">${p.pitch}</div>` : nothing}
            <div class="hl">${hl.map((m) => html`<span>${m.label}: <b>${fmtNum(p.headline[m.id], m.unit)}</b> ${m.unit === 'USD' ? '' : m.unit}</span> · `)}
              ${new Date(p.created_at).toLocaleDateString()}</div>
          </div>
          <span class="tag ${p.status === 'done' ? 'good' : p.status === 'failed' ? 'bad' : ''}">${humanize(p.status)}</span>
        </div>`;
      })}
    </div>`;
  }
}
customElements.define('rd-browse', RdBrowse);
