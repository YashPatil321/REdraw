/** Root overlay: lays out panels per view above the 3D canvas. */

import { LitElement, css, html, nothing } from 'lit';
import { StoreController, theme } from './base';
import './browse';
import './planBuilder';
import './report';
import './shell';
import './traffic';

export class RdApp extends LitElement {
  static override styles = [
    theme,
    css`
      :host {
        position: fixed;
        inset: 0;
        pointer-events: none;
        display: flex;
        flex-direction: column;
      }
      header {
        pointer-events: auto;
      }
      main {
        position: relative;
        flex: 1;
        min-height: 0;
      }
      .left,
      .right,
      .center,
      .bottom,
      .stats,
      .toasts,
      .status {
        position: absolute;
        pointer-events: none;
      }
      .left > *,
      .right > *,
      .center > *,
      .bottom > *,
      .stats > *,
      .toasts > * {
        pointer-events: auto;
      }
      .left {
        top: 10px;
        left: 10px;
        bottom: 10px;
        width: 320px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .left > * {
        max-height: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
      }
      .right {
        top: 10px;
        right: 10px;
        bottom: 10px;
        width: 340px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .right > * {
        min-height: 0;
        display: flex;
        flex-direction: column;
      }
      .right.wide {
        width: 500px;
      }
      .center {
        top: 10px;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        width: min(760px, calc(100% - 20px));
        display: flex;
        flex-direction: column;
      }
      .center > * {
        min-height: 0;
        display: flex;
        flex-direction: column;
      }
      .bottom {
        left: 342px;
        right: 10px;
        bottom: 10px;
      }
      .bottom.report {
        left: 10px;
        right: 520px;
      }
      .stats {
        right: 10px;
        bottom: 10px;
      }
      .bottom ~ .stats {
        bottom: 84px;
      }
      .toasts {
        left: 50%;
        transform: translateX(-50%);
        top: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
      }
      .status {
        left: 50%;
        transform: translateX(-50%);
        bottom: 90px;
        background: rgba(0, 0, 0, 0.55);
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
      }
      .boot {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        background: radial-gradient(ellipse at center, rgba(10, 14, 22, 0.75), rgba(10, 14, 22, 0.95));
        pointer-events: auto;
        z-index: 5;
      }
      .boot .box {
        text-align: center;
        max-width: 520px;
        padding: 20px;
      }
      .boot h1 {
        font-size: 28px;
        margin: 0 0 8px;
      }
      .boot h1 span {
        color: var(--plan);
      }
      .pick-hint {
        position: absolute;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        padding: 6px 12px;
        border-color: var(--plan);
      }
    `,
  ];
  private st = new StoreController(this, ['view', 'booting', 'bootMessage', 'fatal', 'showStats', 'worldStatus', 'building', 'school', 'buildingLoading', 'mapPick', 'meta']);

  override render() {
    const s = this.st.s;
    const infoPanel = s.building || s.school || s.buildingLoading;
    let content;
    switch (s.view) {
      case 'explore':
        content = html`<div class="left"><rd-mission></rd-mission></div>
          ${infoPanel ? html`<div class="right"><rd-info-panel></rd-info-panel></div>` : nothing}`;
        break;
      case 'traffic':
        content = html`<div class="left"><rd-traffic-panel class="scroll"></rd-traffic-panel></div>
          ${infoPanel ? html`<div class="right"><rd-info-panel></rd-info-panel></div>` : nothing}
          <div class="bottom"><rd-scrubber></rd-scrubber></div>`;
        break;
      case 'plan':
        content = html`<div class="left"><rd-tool-palette></rd-tool-palette></div>
          <div class="right"><rd-plan-summary></rd-plan-summary>${infoPanel ? html`<rd-info-panel></rd-info-panel>` : nothing}</div>
          ${s.mapPick ? html`<div class="pick-hint panel">Placing on map: click the terrain. Press Esc when done.</div>` : nothing}`;
        break;
      case 'report':
        content = html`<rd-split-slider></rd-split-slider>
          <div class="right wide"><rd-report></rd-report></div>
          <div class="bottom report"><rd-scrubber></rd-scrubber></div>`;
        break;
      case 'browse':
        content = html`<div class="center"><rd-browse></rd-browse></div>`;
        break;
    }
    return html`<header><rd-topbar></rd-topbar><rd-banner></rd-banner></header>
      <main>
        ${content}
        ${s.worldStatus ? html`<div class="status">${s.worldStatus}</div>` : nothing}
        ${s.showStats ? html`<div class="stats"><rd-stats></rd-stats></div>` : nothing}
        <div class="toasts"><rd-toasts></rd-toasts></div>
      </main>
      ${s.booting || s.fatal
        ? html`<div class="boot"><div class="box">
            <h1>Re<span>draw</span></h1>
            ${s.fatal ? html`<p class="err">${s.fatal}</p><p class="muted small">Is the API running? <code>.venv/bin/uvicorn api.main:app</code>, or open with <a href="?mock=1" style="color:var(--plan)">?mock=1</a> for offline UI testing.</p>` : html`<p class="muted">${s.bootMessage}</p>`}
          </div></div>`
        : nothing}`;
  }
}
customElements.define('rd-app', RdApp);
