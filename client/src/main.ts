/** Redraw web client entry point. */

import { initActions, loadBaselinePlayback, openPlan } from './actions';
import { api, initApi } from './api';
import { parseHash } from './router';
import { SceneController } from './scene/controller';
import { store, toast } from './state';
import { RoadNetwork } from './traffic/network';
import { appCtx } from './ui/base';
import './ui/app';

async function boot(): Promise<void> {
  const container = document.getElementById('scene')!;
  const scene = new SceneController(container);
  appCtx.scene = scene;
  // expose for debugging and automated screenshots
  (window as unknown as { redraw: unknown }).redraw = { store, scene };

  await initApi();
  initActions();

  store.set({ bootMessage: 'Loading world info…' });
  const [meta, schools, tools] = await Promise.all([api.getMeta(), api.getSchools(), api.getTools()]);
  store.set({
    meta,
    schools,
    tools: tools.tools,
    simTime: Math.max(meta.time.report_start_s, Math.min(meta.time.report_end_s, 27000)),
  });
  document.title = `Redraw · ${meta.region.display_name}${meta.synthetic ? ' (synthetic)' : ''}`;
  scene.init(meta, schools);
  store.set({ booting: false });

  void scene.loadWorld();
  api
    .getNetwork()
    .then((json) => {
      const net = new RoadNetwork(json);
      store.set({ network: net });
      scene.setNetwork(net);
    })
    .catch((e: unknown) => toast(`Road network unavailable: ${(e as Error).message}`, 'error', 8000));
  void loadBaselinePlayback();

  const route = (): void => {
    const r = parseHash(location.hash);
    if (r.planId) {
      void openPlan(r.planId);
      return;
    }
    store.set({ view: r.view });
  };
  window.addEventListener('hashchange', route);
  route();
}

boot().catch((e: unknown) => {
  console.error(e);
  store.set({ fatal: `Could not start: ${(e as Error).message}`, booting: false });
});
