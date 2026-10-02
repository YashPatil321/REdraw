/** App shell components: top bar, banners, mission brief, info panel, stats, toasts. */

import { LitElement, css, html, nothing } from 'lit';
import { navigate } from '../actions';
import type { Quality } from '../scene/quality';
import { OFFLINE_VIEWER } from '../actions';
import { store, type View } from '../state';
import { StoreController, appCtx, humanize, fmtUsd, theme } from './base';

const VIEWS: Array<[View, string]> = [
  ['explore', 'Explore'],
  ['traffic', 'Traffic'],
  ['plan', 'Plan builder'],
  ['report', 'Report'],
  ['browse', 'Browse plans'],
];

export class RdTopbar extends LitElement {
  static override styles = [
    theme,
    css`
      .bar {
        display: flex;
        align-items: center;
        gap: 10px;
        height: 44px;
        padding: 0 12px;
        border-radius: 0;
        border-width: 0 0 1px 0;
      }
      .brand {
        font-weight: 800;
        letter-spacing: 0.02em;
        font-size: 16px;
      }
      .brand span {
        color: var(--plan);
      }
      .region {
        color: var(--muted);
        max-width: 260px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      nav {
        display: flex;
        gap: 4px;
        margin-left: 10px;
      }
      nav button {
        border-color: transparent;
        background: transparent;
        padding: 6px 9px;
        white-space: nowrap;
      }
      .bar button,
      .bar .tag {
        white-space: nowrap;
      }
      nav button.on {
        background: rgba(255, 255, 255, 0.1);
        border-color: var(--line);
      }
      .cal {
        font-size: 12px;
      }
      select.q {
        width: auto;
        padding: 4px 4px;
        font-size: 12px;
      }
      .seg {
        display: inline-flex;
        border: 1px solid var(--line);
        border-radius: 7px;
        overflow: hidden;
      }
      .seg button {
        border: 0;
        border-radius: 0;
        padding: 4px 9px;
        font-size: 12px;
        background: transparent;
      }
      .seg button.on {
        background: rgba(255, 255, 255, 0.16);
        font-weight: 650;
      }
      .seg button + button {
        border-left: 1px solid var(--line);
      }
      .seg .dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--good);
        margin-right: 5px;
        vertical-align: 1px;
      }
      .seg .dot.off {
        background: var(--muted);
      }
      .seg .dot.err {
        background: var(--bad);
      }
      @media (max-width: 1480px) {
        .region {
          display: none;
        }
      }
    `,
  ];
  private st = new StoreController(this, ['view', 'meta', 'showStats', 'plan', 'quality', 'walking']);

  override render() {
    const s = this.st.s;
    const cal = s.meta?.calibration;
    const calCls = cal?.status === 'calibrated' ? 'good' : cal?.status === 'above_target' ? 'warn' : 'bad';
    const err = cal?.median_error_pct;
    return html`<div class="bar panel">
      <div class="brand">Re<span>draw</span></div>
      <div class="region" title=${s.meta?.region.display_name ?? ''}>${s.meta?.region.display_name ?? ''}</div>
      <nav>
        ${VIEWS.map(
          ([v, label]) =>
            html`<button class=${s.view === v ? 'on' : ''} data-view=${v} @click=${() => navigate(v)}>${label}</button>`,
        )}
      </nav>
      <div class="spacer"></div>
      ${OFFLINE_VIEWER ? html`<span class="tag warn" title="Static snapshot: explore and replay saved plans; runs need the Python API">Viewer</span>` : nothing}
      ${cal
        ? html`<span class="tag cal ${calCls}" title=${`Baseline calibration against observed travel times: ${humanize(cal.status)}${err !== null && err !== undefined ? `, median error ${err.toFixed(1)}%` : ''}`}
            >${humanize(cal.status)}${err !== null && err !== undefined ? ` (${err.toFixed(0)}% err)` : ''}</span
          >`
        : nothing}
      ${s.showStats ? html`<rd-stats></rd-stats>` : nothing}
      <button class=${s.walking ? 'active' : ''} @click=${() => (s.walking ? appCtx.scene?.exitWalk() : appCtx.scene?.walkAtTarget())} title="Street-level first-person view (WASD + mouse)">
        🚶 ${s.walking ? 'Exit walk' : 'Walk'}
      </button>
      <button @click=${() => appCtx.scene?.enterPhotoMode()} title="Photo mode: hide the UI, render at ultra quality and save a PNG">📷 Photo</button>
      <select class="q" aria-label="Render quality" title="Render quality" @change=${(e: Event) => appCtx.scene?.setQuality((e.target as HTMLSelectElement).value as Quality)}>
        ${(['ultra', 'high', 'medium', 'low'] as const).map((q) => html`<option value=${q} ?selected=${s.quality === q}>${q === 'medium' ? 'Med' : q[0]!.toUpperCase() + q.slice(1)} quality</option>`)}
      </select>
      <button class=${s.showStats ? 'active' : ''} @click=${() => store.set({ showStats: !s.showStats })} title="FPS, draw calls, triangles">
        Stats
      </button>
    </div>`;
  }
}
customElements.define('rd-topbar', RdTopbar);

export class RdBanner extends LitElement {
  static override styles = [
    theme,
    css`
      .b {
        text-align: center;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.04em;
        padding: 3px 8px;
        background: repeating-linear-gradient(-45deg, rgba(255, 170, 0, 0.92), rgba(255, 170, 0, 0.92) 12px, rgba(240, 150, 0, 0.92) 12px, rgba(240, 150, 0, 0.92) 24px);
        color: #1d1300;
        pointer-events: auto;
      }
      .mock {
        background: rgba(170, 90, 255, 0.92);
        color: #fff;
      }
    `,
  ];
  private st = new StoreController(this, ['meta']);
  override render() {
    const m = this.st.s.meta;
    if (!m) return nothing;
    return html`${m.mock
      ? html`<div class="b mock" role="status">MOCK FIXTURE (?mock=1): offline UI test data. Not real geography, not sim results.</div>`
      : nothing}
    ${m.synthetic ? html`<div class="b" role="status">SYNTHETIC DEV DATA — not real geography</div>` : nothing}`;
  }
}
customElements.define('rd-banner', RdBanner);

export class RdMission extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        padding: 14px;
        max-height: 100%;
        box-sizing: border-box;
      }
      .budget {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin: 8px 0;
      }
      .budget div {
        background: var(--bg-2);
        border-radius: 8px;
        padding: 6px 8px;
      }
      .budget b {
        display: block;
        font-size: 16px;
      }
      ul {
        margin: 4px 0;
        padding-left: 18px;
      }
      .actions {
        display: flex;
        gap: 8px;
        margin-top: 12px;
      }
      .collapsed {
        padding: 8px 12px;
      }
    `,
  ];
  private st = new StoreController(this, ['meta']);
  private collapsed = false;

  override render() {
    const m = this.st.s.meta?.mission;
    if (!m) return nothing;
    if (this.collapsed) {
      return html`<div class="panel collapsed row">
        <b>${m.title}</b><span class="spacer"></span
        ><button class="ghost" @click=${() => ((this.collapsed = false), this.requestUpdate())}>Show brief</button>
      </div>`;
    }
    return html`<div class="panel p scroll">
      <div class="row"><span class="tag">Mission</span><span class="spacer"></span
        ><button class="ghost small" @click=${() => ((this.collapsed = true), this.requestUpdate())}>Hide</button></div>
      <h2 style="margin-top:6px">${m.title}</h2>
      <p>${m.brief}</p>
      <div class="budget">
        <div><span class="muted small">Budget upfront</span><b>${fmtUsd(m.budget_usd_upfront)}</b></div>
        <div><span class="muted small">Budget per year</span><b>${fmtUsd(m.budget_usd_per_year)}</b></div>
      </div>
      ${m.constraints?.length
        ? html`<h3>Constraints</h3>
            <ul>
              ${m.constraints.map((c) => html`<li>${humanize(c)}</li>`)}
            </ul>`
        : nothing}
      ${m.goals_suggested?.length
        ? html`<h3>Suggested goals (hints, not pass or fail)</h3>
            <ul>
              ${m.goals_suggested.map((g) => html`<li>${g}</li>`)}
            </ul>`
        : nothing}
      <p class="muted small">Drag to pan, right-drag to rotate, scroll to zoom. Click a building or a school label for details.</p>
      <div class="actions">
        <button @click=${() => navigate('traffic')}>Watch morning traffic</button>
        <button class="primary" @click=${() => navigate('plan')}>Build a plan</button>
      </div>
    </div>`;
  }
}
customElements.define('rd-mission', RdMission);

export class RdInfoPanel extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        padding: 12px 14px;
      }
      dl {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 3px 12px;
        margin: 6px 0 0;
      }
      dt {
        color: var(--muted);
      }
      dd {
        margin: 0;
      }
      .ent {
        background: var(--bg-2);
        border-radius: 8px;
        padding: 6px 8px;
        margin-top: 6px;
      }
    `,
  ];
  private st = new StoreController(this, ['building', 'buildingLoading', 'school']);

  private close(): void {
    store.set({ building: null, school: null });
  }

  override render() {
    const { building: b, buildingLoading, school } = this.st.s;
    if (school) {
      return html`<div class="panel p">
        <div class="row"><h2 style="margin:0">${school.name}</h2><span class="spacer"></span><button class="ghost" @click=${this.close} aria-label="Close">✕</button></div>
        <div class="row" style="margin-top:6px"><button class="walk" @click=${() => appCtx.scene?.walkHere(school.x, school.z)} title="Street-level first-person view">🚶 Walk here</button></div>
        <dl>
          <dt>Grades</dt>
          <dd>${school.grades.map((g) => (g === 0 ? 'K' : String(g))).join('–')}</dd>
          <dt>Bell</dt>
          <dd>${school.bell_start} ${school.verified ? nothing : html`<span class="tag warn">unverified</span>`}</dd>
          <dt>Students</dt>
          <dd>${school.students.toLocaleString('en-US')}</dd>
        </dl>
        ${school.entrances.map(
          (e) => html`<div class="ent">
            <b>${humanize(e.id)}</b> <span class="muted small">${e.curb_spots} curb spots, ${e.unload_seconds}s unload</span>
            ${e.verified ? nothing : html`<span class="tag warn">unverified</span>`}
            ${e.baseline
              ? html`<div class="small">Baseline: max queue ${e.baseline.max_queue_cars.toFixed(0)} cars, spillback
                  ${e.baseline.max_spillback_m.toFixed(0)} m, avg wait ${e.baseline.avg_wait_min.toFixed(1)} min</div>`
              : nothing}
          </div>`,
        )}
      </div>`;
    }
    if (buildingLoading && !b) return html`<div class="panel p muted">Loading building…</div>`;
    if (!b) return nothing;
    const rows: Array<[string, unknown]> = [
      ['Type', b.type ? humanize(String(b.type)) : undefined],
      ['Name', b.name],
      ['Address', b.address],
      ['Height', typeof b.height_m === 'number' ? `${b.height_m.toFixed(1)} m` : undefined],
      ['Levels', b.levels],
      // lidar roof model, when the API provides it
      ['Roof', typeof (b['roof_type'] ?? b['roof_shape']) === 'string' ? humanize(String(b['roof_type'] ?? b['roof_shape'])) : undefined],
      ['Eave height', num(b['eave_h_m'] ?? b['eave_h'])],
      ['Ridge height', num(b['ridge_h_m'] ?? b['ridge_h'])],
      ['Height source', typeof (b['height_source'] ?? b['roof_source']) === 'string' ? humanize(String(b['height_source'] ?? b['roof_source'])) : undefined],
      ['Footprint', typeof b.area_m2 === 'number' ? `${Math.round(b.area_m2)} m²` : undefined],
      ['Households', b.households],
      ['Block', b.block],
      ['School', b.school_id],
    ];
    return html`<div class="panel p">
      <div class="row"><h2 style="margin:0">${b.name ? String(b.name) : `Building #${b.id}`}</h2><span class="spacer"></span><button class="ghost" @click=${this.close} aria-label="Close">✕</button></div>
      ${typeof b.centroid_x === 'number' && typeof b.centroid_z === 'number'
        ? html`<div class="row" style="margin-top:6px"><button class="walk" @click=${() => appCtx.scene?.walkHere(b.centroid_x!, b.centroid_z!)} title="Street-level first-person view">🚶 Walk here</button></div>`
        : nothing}
      ${'error' in b && b['error'] ? html`<p class="err">${String(b['error'])}</p>` : nothing}
      <dl>
        ${rows
          .filter(([, v]) => v !== undefined && v !== null && v !== '')
          .map(([k, v]) => html`<dt>${k}</dt><dd>${String(v)}</dd>`)}
      </dl>
    </div>`;
  }
}
customElements.define('rd-info-panel', RdInfoPanel);

export class RdStats extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
        display: flex;
        gap: 10px;
        color: var(--muted);
        white-space: nowrap;
      }
      b {
        color: var(--text);
        font-weight: 600;
      }
      .good {
        color: var(--good);
      }
      .bad {
        color: var(--bad);
      }
    `,
  ];
  private timer = 0;
  override connectedCallback(): void {
    super.connectedCallback();
    this.timer = window.setInterval(() => this.requestUpdate(), 500);
  }
  override disconnectedCallback(): void {
    super.disconnectedCallback();
    clearInterval(this.timer);
  }
  override render() {
    const st = appCtx.scene?.viewer.stats;
    if (!st) return nothing;
    return html`<div class="p" role="status" aria-label="Render stats">
      <span>FPS <b class=${st.fps >= 30 ? 'good' : 'bad'}>${st.fps.toFixed(0)}</b></span>
      <span>calls <b>${st.calls}</b></span>
      <span>tris <b>${(st.triangles / 1000).toFixed(0)}k</b></span>
      <span>geo <b>${st.geometries}</b></span>
      <span>tex <b>${st.textures}</b></span>
    </div>`;
  }
}
customElements.define('rd-stats', RdStats);

export class RdToasts extends LitElement {
  static override styles = [
    theme,
    css`
      .t {
        padding: 8px 12px;
        margin-top: 6px;
        max-width: 420px;
      }
      .error {
        border-color: rgba(255, 107, 90, 0.6);
      }
    `,
  ];
  private st = new StoreController(this, ['toasts']);
  override render() {
    return html`${this.st.s.toasts.map((t) => html`<div class="panel t ${t.kind}" role="status">${t.text}</div>`)}`;
  }
}
customElements.define('rd-toasts', RdToasts);

/** meters with one decimal, or undefined */
function num(v: unknown): string | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(1)} m` : undefined;
}

/** Required on-screen credits: Google logo + data providers when Google tiles are drawn, open-data sources otherwise. */
export class RdAttribution extends LitElement {
  static override styles = [
    theme,
    css`
      :host {
        display: block;
      }
      .a {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        line-height: 16px;
        color: rgba(255, 255, 255, 0.92);
        text-shadow: 0 0 3px rgba(0, 0, 0, 0.9), 0 0 1px #000;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        pointer-events: auto;
      }
      .logo {
        font: 600 15px/16px 'Product Sans', Roboto, Arial, sans-serif;
        letter-spacing: -0.2px;
        color: #fff;
        flex: none;
      }
      .txt {
        overflow: hidden;
        text-overflow: ellipsis;
      }
    `,
  ];
  private st = new StoreController(this, ['attribution', 'renderMode', 'meta', 'openCredits']);
  override render() {
    const s = this.st.s;
    const a = s.attribution;
    // Google credits only while Google tiles are drawn (the controller sets them)
    if (a?.google) {
      return html`<div class="a" role="contentinfo" aria-label="Map data attribution">
        ${a.google ? html`<span class="logo" aria-label="Google">Google</span>` : nothing}
        <span class="txt" title=${a.text}>${a.text ? `Map data ©${new Date().getFullYear()} ${a.text}` : ''}</span>
      </div>`;
    }
    return html`<div class="a" role="contentinfo">
      <span class="txt" title=${s.openCredits}>${s.meta?.synthetic ? 'Synthetic dev world (not real geography)' : s.openCredits || '© OpenStreetMap contributors (ODbL)'}</span>
    </div>`;
  }
}
customElements.define('rd-attribution', RdAttribution);

/** Camera mode control: enter street-level walk, and the walk HUD with controls and an exit. */
export class RdCameraTools extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        padding: 6px 8px;
        display: flex;
        gap: 6px;
        align-items: center;
      }
      .hud {
        padding: 8px 12px;
        max-width: 560px;
      }
      kbd {
        display: inline-block;
        border: 1px solid var(--line);
        border-bottom-width: 2px;
        border-radius: 4px;
        padding: 0 5px;
        font: 600 11px/16px ui-monospace, monospace;
        background: var(--bg-2);
      }
    `,
  ];
  private st = new StoreController(this, ['walking', 'renderMode', 'photoreal']);
  override render() {
    const s = this.st.s;
    if (s.walking) {
      return html`<div class="panel hud row" role="status">
        <span><b>Street level</b> · <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move · mouse look (click to capture) · <kbd>Shift</kbd> run · <kbd>Q</kbd>/<kbd>E</kbd> eye height</span>
        <span class="spacer"></span>
        <button class="primary" @click=${() => appCtx.scene?.exitWalk()}>Exit walk (Esc)</button>
      </div>`;
    }
    return nothing;
  }
}
customElements.define('rd-camera-tools', RdCameraTools);

/** Photo mode controls (hidden while the PNG is captured). */
export class RdPhotoBar extends LitElement {
  static override styles = [
    theme,
    css`
      .p {
        position: fixed;
        top: 10px;
        right: 10px;
        display: flex;
        gap: 6px;
        padding: 6px;
        pointer-events: auto;
        opacity: 0.85;
      }
      .p:hover {
        opacity: 1;
      }
    `,
  ];
  private onKey = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') appCtx.scene?.exitPhotoMode();
  };
  override connectedCallback(): void {
    super.connectedCallback();
    window.addEventListener('keydown', this.onKey);
  }
  override disconnectedCallback(): void {
    super.disconnectedCallback();
    window.removeEventListener('keydown', this.onKey);
  }
  override render() {
    return html`<div class="panel p photo-bar">
      <button class="primary" @click=${() => void appCtx.scene?.savePhoto()}>Save PNG</button>
      <button @click=${() => appCtx.scene?.exitPhotoMode()}>Exit (Esc)</button>
    </div>`;
  }
}
customElements.define('rd-photo-bar', RdPhotoBar);
