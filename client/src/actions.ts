/**
 * User actions: everything that talks to the API on behalf of the UI. No game
 * logic: costs, validation and results always come from the server.
 */

import { ApiError, api, with503Retry } from './api';
import { hashFor } from './router';
import { store, toast, type View } from './state';
import { parsePlayback } from './traffic/playback';
import type { ParamValue, PlanInput, ToolDef } from './types';
import { newToolInstance, setParam } from './ui/formgen';

let checkTimer: ReturnType<typeof setTimeout> | null = null;
let checkSeq = 0;
let pollTimer: ReturnType<typeof setTimeout> | null = null;
/** JSON of the draft as last saved, to detect edits after saving */
let savedDraftKey: string | null = null;

function errText(e: unknown): string {
  if (e instanceof ApiError) return e.detail;
  return (e as Error)?.message ?? String(e);
}

export function draftInput(): PlanInput {
  const s = store.get();
  return {
    mission: s.meta?.mission.id ?? 'morning_crunch',
    title: s.draft.title.trim() || 'Untitled plan',
    pitch: s.draft.pitch.trim(),
    tools: s.draft.tools,
  };
}

function draftKey(): string {
  const d = store.get().draft;
  return JSON.stringify([d.title, d.pitch, d.tools]);
}

export function isDraftDirty(): boolean {
  return savedDraftKey === null || savedDraftKey !== draftKey();
}

export function navigate(view: View): void {
  const s = store.get();
  if (view === 'plan' && s.plan) {
    location.hash = hashFor({ view: 'plan', planId: s.plan.id });
    return;
  }
  location.hash = hashFor({ view });
}

// ---- plan builder

export function addTool(def: ToolDef): void {
  const s = store.get();
  const inst = newToolInstance(def, { schools: s.schools });
  const tools = [...s.draft.tools, inst];
  store.set({ draft: { ...s.draft, tools }, selectedTool: tools.length - 1, mapPick: null });
}

export function removeTool(index: number): void {
  const s = store.get();
  const tools = s.draft.tools.filter((_, i) => i !== index);
  store.set({ draft: { ...s.draft, tools }, selectedTool: null, mapPick: null });
}

export function updateParam(index: number, paramId: string, value: ParamValue): void {
  const s = store.get();
  const tools = s.draft.tools.map((t, i) => (i === index ? { ...t, params: setParam(t.params, paramId, value) } : t));
  store.set({ draft: { ...s.draft, tools } });
}

export function setDraftText(field: 'title' | 'pitch', value: string): void {
  const s = store.get();
  store.set({ draft: { ...s.draft, [field]: value } });
}

export function newDraft(): void {
  savedDraftKey = null;
  store.set({ draft: { title: '', pitch: '', tools: [] }, plan: null, check: null, selectedTool: null, job: null, residents: null, planPlayback: null });
  location.hash = hashFor({ view: 'plan' });
}

/** Live POST /plans/check for the budget bar (debounced). */
export function scheduleCheck(): void {
  if (checkTimer) clearTimeout(checkTimer);
  checkTimer = setTimeout(() => void runCheck(), 350);
}

async function runCheck(): Promise<void> {
  const seq = ++checkSeq;
  if (store.get().draft.tools.length === 0) {
    store.set({ check: null, checking: false });
    return;
  }
  store.set({ checking: true });
  try {
    const check = await api.checkPlan(draftInput());
    if (seq === checkSeq) store.set({ check, checking: false });
  } catch (e) {
    if (seq === checkSeq) {
      store.set({ checking: false });
      toast(`Plan check failed: ${errText(e)}`, 'error');
    }
  }
}

export async function savePlan(): Promise<string | null> {
  try {
    const cur = store.get().plan;
    // the author can update in place; anyone else saves a new plan (fork)
    const plan = cur?.is_mine ? await api.updatePlan(cur.id, draftInput()) : await api.createPlan(draftInput());
    savedDraftKey = draftKey();
    store.set({ plan, check: plan.check ?? store.get().check });
    history.replaceState(null, '', hashFor({ view: 'plan', planId: plan.id }));
    toast('Plan saved. The link now opens this plan.');
    return plan.id;
  } catch (e) {
    toast(`Save failed: ${errText(e)}`, 'error');
    return null;
  }
}

export async function runPlan(): Promise<void> {
  const s = store.get();
  let id = s.plan?.id ?? null;
  if (!id || isDraftDirty()) id = await savePlan();
  if (!id) return;
  try {
    const { job_id } = await api.runPlan(id);
    store.set({ job: { id: job_id, plan_id: id, status: 'queued', progress: 0, message: 'queued', error: null } });
    pollJob(job_id, id);
  } catch (e) {
    toast(`Run failed: ${errText(e)}`, 'error');
  }
}

function pollJob(jobId: string, planId: string): void {
  if (pollTimer) clearTimeout(pollTimer);
  const tick = async (): Promise<void> => {
    try {
      const job = await api.getJob(jobId);
      store.set({ job });
      if (job.status === 'done') {
        await loadPlanResults(planId, true);
        store.set({ job: null });
        return;
      }
      if (job.status === 'failed') {
        toast(`Run failed: ${job.error ?? job.message}`, 'error', 9000);
        return;
      }
    } catch (e) {
      toast(`Lost track of the run: ${errText(e)}`, 'error');
      return;
    }
    pollTimer = setTimeout(() => void tick(), 900);
  };
  void tick();
}

/** Fetch plan (with report), its playback and residents; optionally switch to the report. */
async function loadPlanResults(planId: string, goToReport: boolean): Promise<void> {
  const plan = await api.getPlan(planId);
  store.set({ plan });
  if (plan.status !== 'done') return;
  if (goToReport) store.set({ view: 'report', reportTab: 'report' });
  const [pb, res] = await Promise.allSettled([api.getPlanPlayback(planId), api.getResidents(planId)]);
  if (pb.status === 'fulfilled') {
    try {
      store.set({ planPlayback: parsePlayback(pb.value) });
    } catch (e) {
      toast(`Plan playback unreadable: ${errText(e)}`, 'error');
    }
  } else {
    console.warn('plan playback unavailable', pb.reason);
  }
  store.set({ residents: res.status === 'fulfilled' ? res.value : null });
  if (res.status === 'fulfilled' && res.value.text_status === 'pending') refreshResidentsLater(planId);
}

/** Resident quotes are written in the background; poll until complete. */
function refreshResidentsLater(planId: string, tries = 0): void {
  if (tries > 40) return;
  setTimeout(() => {
    if (store.get().plan?.id !== planId) return;
    api
      .getResidents(planId)
      .then((r) => {
        store.set({ residents: r });
        if (r.text_status === 'pending') refreshResidentsLater(planId, tries + 1);
      })
      .catch(() => undefined);
  }, 4000);
}

/** Open a plan by id (from a #/plan/{id} link). */
export async function openPlan(id: string): Promise<void> {
  const s = store.get();
  if (s.plan?.id === id && s.plan.status === 'done') {
    if (s.view !== 'report' && s.view !== 'plan') store.set({ view: 'report' });
    return;
  }
  try {
    const plan = await api.getPlan(id);
    store.set({
      plan,
      draft: { title: plan.title, pitch: plan.pitch, tools: plan.tools },
      check: plan.check,
      selectedTool: null,
      planPlayback: null,
      residents: null,
      view: plan.status === 'done' ? 'report' : 'plan',
    });
    savedDraftKey = draftKey();
    if (plan.status === 'done') await loadPlanResults(id, false);
    else if ((plan.status === 'queued' || plan.status === 'running') && plan.job_id) {
      store.set({ job: { id: plan.job_id, plan_id: id, status: plan.status, progress: 0, message: plan.status, error: null } });
      pollJob(plan.job_id, id);
    } else if (plan.status === 'queued' || plan.status === 'running') waitForPlan(id);
  } catch (e) {
    toast(`Could not open plan ${id}: ${errText(e)}`, 'error');
    store.set({ view: 'browse' });
  }
}

/** Plan is running elsewhere (no job id): poll the plan itself. */
function waitForPlan(id: string): void {
  if (pollTimer) clearTimeout(pollTimer);
  const tick = async (): Promise<void> => {
    try {
      const plan = await api.getPlan(id);
      store.set({ plan });
      if (plan.status === 'done') {
        await loadPlanResults(id, true);
        return;
      }
      if (plan.status === 'failed' || plan.status === 'draft') return;
    } catch {
      return;
    }
    pollTimer = setTimeout(() => void tick(), 2000);
  };
  pollTimer = setTimeout(() => void tick(), 2000);
}

export async function vote(id: string, value: 1 | -1): Promise<number | null> {
  try {
    return (await api.vote(id, value)).votes;
  } catch (e) {
    toast(`Vote failed: ${errText(e)}`, 'error');
    return null;
  }
}

/** Load the baseline playback once (Traffic view). */
export async function loadBaselinePlayback(): Promise<void> {
  if (store.get().baselinePlayback) return;
  try {
    const buf = await with503Retry(
      () => api.getBaselinePlayback(),
      () => store.set({ worldStatus: 'Baseline traffic is warming up on the server…' }),
    );
    if (store.get().worldStatus.startsWith('Baseline')) store.set({ worldStatus: '' });
    store.set({ baselinePlayback: parsePlayback(buf) });
  } catch (e) {
    toast(`Baseline traffic unavailable: ${errText(e)}`, 'error', 8000);
  }
}

export function initActions(): void {
  store.select(
    (s) => s.draft,
    () => scheduleCheck(),
  );
}
