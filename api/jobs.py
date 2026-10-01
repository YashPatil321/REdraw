"""Background plan runs (spec 9: runs execute in a background worker).

A small thread pool takes jobs off the request thread; each job calls
`SimService.run(plan, seeds=REPORT_SEEDS, workers=SIM_WORKERS, progress=cb)`,
which fans seeds out to its own process pool. When the run finishes the job
stores the plan playback on disk (`data/processed/playback/{plan_id}.rdpb`),
computes deterministic resident approval, inserts the `resident_approval_pct`
metric into the report and marks the plan done. LLM reaction text is filled in
afterwards by a separate single-thread pool so a slow or absent LLM never
delays the report card.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from api.db import clean_json, utcnow
from api.state import AppState, as_dict, atomic_write, field
from residents.reactions import insert_approval_metric

log = logging.getLogger(__name__)

PROGRESS_WRITE_INTERVAL_S = 0.25


class JobRunner:
    def __init__(self, state: AppState, max_workers: int = 1) -> None:
        self.state = state
        self.pool = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="redraw-job")
        self.text_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="redraw-llm")
        self._text_inflight: set[str] = set()
        self._lock = threading.Lock()

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.text_pool.shutdown(wait=False, cancel_futures=True)

    def playback_path(self, plan_id: str) -> Path:
        return self.state.settings.playback_dir / f"{plan_id}.rdpb"

    # ---------------------------------------------------------------- run
    def submit(self, plan: dict[str, Any]) -> dict[str, Any]:
        store = self.state.store
        with self._lock:
            active = store.active_job_for_plan(plan["id"])
            if active:
                return active
            job = store.create_job(plan["id"])
            store.set_plan_status(plan["id"], "queued")
        self.pool.submit(self._run_safe, job["id"], plan["id"])
        return job

    def _run_safe(self, job_id: str, plan_id: str) -> None:
        try:
            self._run(job_id, plan_id)
        except Exception as e:  # noqa: BLE001
            log.exception("plan run %s failed", plan_id)
            err = f"{type(e).__name__}: {e}"
            self.state.store.update_job(job_id, status="failed", error=err, message="failed",
                                        finished_at=utcnow())
            self.state.store.set_plan_status(plan_id, "failed")

    def _run(self, job_id: str, plan_id: str) -> None:
        st = self.state
        store = st.store
        plan = store.get_plan(plan_id)
        if plan is None:
            raise RuntimeError(f"plan {plan_id} disappeared")
        store.update_job(job_id, status="running", started_at=utcnow(), progress=0.0, message="starting")
        store.set_plan_status(plan_id, "running")

        if st.warm_thread is not None and not st.baseline_done.is_set():
            store.update_job(job_id, message="waiting for the baseline to finish warming up")
            st.baseline_done.wait()

        sim = st.sim()
        last = [0.0]

        def progress(frac: float, msg: str) -> None:
            now = time.monotonic()
            if now - last[0] < PROGRESS_WRITE_INTERVAL_S and frac < 1.0:
                return
            last[0] = now
            store.update_job(job_id, progress=round(0.02 + 0.9 * max(0.0, min(1.0, float(frac))), 4),
                             message=str(msg))

        sim_plan = {k: plan[k] for k in ("id", "mission", "title", "pitch", "author_id", "tools", "created_at")}
        result = sim.run(sim_plan, seeds=st.settings.report_seeds, workers=st.settings.sim_workers,
                         progress=progress)

        report = clean_json(as_dict(field(result, "report")))
        if not report:
            raise RuntimeError("sim returned no report for the plan")
        report["plan_id"] = plan_id
        pb = field(result, "playback_plan")
        if pb:
            atomic_write(self.playback_path(plan_id), bytes(pb))
        if st.baseline_summary is None:
            bs = field(result, "baseline_summary")
            if bs:
                st.baseline_summary = clean_json(as_dict(bs))
        if st.baseline_playback is None and field(result, "playback_baseline"):
            st.baseline_playback = bytes(field(result, "playback_baseline"))

        store.update_job(job_id, progress=0.95, message="residents are reacting")
        reactions: list[dict[str, Any]] = []
        try:
            cost_year = float(((report.get("cost") or {}).get("per_year_usd"))
                              or (plan.get("check") or {}).get("cost_per_year_usd") or 0.0)
            reactions, metric = st.residents.compute(plan, clean_json(field(result, "person_deltas")), cost_year)
            insert_approval_metric(report, metric)
        except Exception as e:  # noqa: BLE001
            log.exception("resident reactions failed for %s", plan_id)
            report.setdefault("notes", []).append(f"Resident reactions unavailable: {e}")

        store.save_reactions(plan_id, reactions)
        store.set_plan_status(plan_id, "done", report=report)
        store.update_job(job_id, status="done", progress=1.0, message="done", finished_at=utcnow())
        if reactions:
            self.fill_texts_async(plan_id)

    # ------------------------------------------------------------ LLM text
    def fill_texts_async(self, plan_id: str) -> bool:
        """Generate missing reaction text in the background (no-op if the LLM is down)."""
        if not self.state.llm.available:
            return False
        with self._lock:
            if plan_id in self._text_inflight:
                return False
            self._text_inflight.add(plan_id)
        self.text_pool.submit(self._fill_texts, plan_id)
        return True

    def _fill_texts(self, plan_id: str) -> None:
        try:
            st = self.state
            plan = st.store.get_plan(plan_id)
            if plan is None:
                return
            missing = [r for r in st.store.get_reactions(plan_id) if not r.get("text")]
            if missing:
                texts = st.residents.texts(plan, missing)
                st.store.set_reaction_texts(plan_id, texts)
        except Exception:  # noqa: BLE001
            log.exception("reaction text generation failed for %s", plan_id)
        finally:
            with self._lock:
                self._text_inflight.discard(plan_id)

    def texts_pending(self, plan_id: str) -> bool:
        return plan_id in self._text_inflight
