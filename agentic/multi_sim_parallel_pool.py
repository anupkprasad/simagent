"""
Cross-simulation parallel worker pool for independent per-sim stages.

Runs preprocess+simsetup (pre-HPC) or analysis+reporter (post-HPC / analysis-only)
concurrently using ``ProcessPoolExecutor``. Worker count is chosen automatically
from host CPU and memory via ``parallel_resources.estimate_workers``.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic.parallel_resources import (
    WorkerEstimate,
    estimate_workers,
    should_use_parallel_pool as _resources_allow_parallel_pool,
)
from agentic.parallel_worker import (
    build_per_sim_job_spec,
    job_already_complete,
    run_per_sim_workflow,
)

logger = logging.getLogger(__name__)

# Per-sim attempt budget for parallel prep/analysis workers. After this many
# unfinished attempts the label is marked failed and the pool continues.
DEFAULT_SIM_MAX_ATTEMPTS = int(os.environ.get("AGENTIC_SIM_MAX_ATTEMPTS", "3"))


def _max_sim_attempts(state: Optional[Dict[str, Any]] = None) -> int:
    """Resolve per-sim max attempts from state / env (default 5)."""
    if state is not None:
        raw = state.get("sim_max_attempts")
        if raw is not None:
            try:
                return max(1, int(raw))
            except (TypeError, ValueError):
                pass
    return max(1, DEFAULT_SIM_MAX_ATTEMPTS)


def _fail_sim_exhausted(
    rec: Dict[str, Any],
    *,
    attempts: int,
    max_attempts: int,
    reason: str,
) -> None:
    rec["status"] = "failed"
    rec["attempts"] = attempts
    rec["completed_at"] = _utc_now()
    rec["error"] = (
        f"{reason} after {attempts}/{max_attempts} attempt(s); "
        "skipping this simulation and continuing others"
    )
    logger.warning(
        "Parallel pool: %s — %s",
        rec.get("label") or "?",
        rec["error"],
    )


def should_use_parallel_pool(state: Dict[str, Any]) -> bool:
    """True only for multi-sim scopes containing analysis or reporting."""
    if not _resources_allow_parallel_pool(state):
        return False
    from src.supervisor.unified_enricher import get_agent_execution_order

    agents = get_agent_execution_order(
        state.get("subtask_type") or "full_task", state
    )
    return any(agent in agents for agent in ("analysis", "reporter"))

# Active executors keyed by base working directory (survives graph node transitions).
_RUNNERS: Dict[str, "_PoolRunner"] = {}


class _PoolRunner:
    def __init__(self, pool_id: str, max_workers: int):
        self.pool_id = pool_id
        self.max_workers = max_workers
        # Linux defaults to "fork", which deadlocks when workers later call
        # OpenMP (MDAnalysis/libgomp) after the parent already used it.
        # "spawn" starts a clean interpreter (same pattern as local FEL helpers).
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        )
        self.futures: Dict[str, Future] = {}

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.futures.clear()


def _pool_id(state: Dict[str, Any]) -> str:
    base = state.get("multi_sim_base_dir") or state.get("working_directory") or "."
    return str(Path(base).resolve())


def _get_runner(state: Dict[str, Any], max_workers: int) -> _PoolRunner:
    pid = _pool_id(state)
    runner = _RUNNERS.get(pid)
    if runner is None or runner.max_workers != max_workers:
        if runner is not None:
            runner.shutdown()
        runner = _PoolRunner(pid, max_workers)
        _RUNNERS[pid] = runner
    return runner


def shutdown_pool_runner(state: Dict[str, Any]) -> None:
    pid = _pool_id(state)
    runner = _RUNNERS.pop(pid, None)
    if runner:
        runner.shutdown()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sim_info_by_label(state: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    for sp in state.get("sim_prompts") or []:
        if sp.get("label") == label:
            return sp
    return None


def _count_running(pool: Dict[str, Any]) -> int:
    return sum(
        1 for rec in (pool.get("sims") or {}).values() if rec.get("status") == "running"
    )


def _count_pending(pool: Dict[str, Any]) -> int:
    return sum(
        1 for rec in (pool.get("sims") or {}).values() if rec.get("status") == "pending"
    )


def _all_done(pool: Dict[str, Any]) -> bool:
    """True when every sim has a terminal status (success, skip, or failed)."""
    sims = pool.get("sims") or {}
    if not sims:
        return False
    return all(rec.get("status") in ("done", "skipped", "failed") for rec in sims.values())


def _terminal_success(rec: Dict[str, Any]) -> bool:
    return rec.get("status") in ("done", "skipped")


def _reconcile_sim_from_disk(
    rec: Dict[str, Any],
    phase: str,
    *,
    label: str = "",
    state: Optional[Dict[str, Any]] = None,
) -> None:
    """Mark pool records done/skipped when artifacts exist on disk."""
    status = rec.get("status")
    if status == "running":
        if state and label and _is_actively_running(label, state):
            return
        # Worker finished or parent restarted — trust on-disk artifacts over stale "running".
    job = {"working_dir": rec.get("working_dir"), "phase": phase, "agent_list": rec.get("agent_list")}
    if job_already_complete(job):
        rec["status"] = "skipped" if status == "pending" else "done"
        rec["completed_at"] = rec.get("completed_at") or _utc_now()
        rec.pop("error", None)


def _is_actively_running(label: str, state: Dict[str, Any]) -> bool:
    """True when a worker future for *label* is in flight."""
    runner = _RUNNERS.get(_pool_id(state))
    if not runner:
        return False
    fut = runner.futures.get(label)
    return fut is not None and not fut.done()


def snapshot_parallel_pool_status(pool: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, query-friendly pool summary for state.jsonl."""
    sims = pool.get("sims") or {}
    by_status: Dict[str, List[str]] = {
        "running": [],
        "pending": [],
        "done": [],
        "skipped": [],
        "failed": [],
    }
    for label, rec in sims.items():
        st = rec.get("status") or "pending"
        bucket = st if st in by_status else "pending"
        by_status[bucket].append(label)
    return {
        "updated_at": _utc_now(),
        "workflow_phase": "parallel_pool",
        "pool_type": f"parallel_{pool.get('phase') or 'analysis'}",
        "phase": pool.get("phase"),
        "max_workers": pool.get("max_workers"),
        "llm_concurrency": (pool.get("resource_estimate") or {}).get("llm_concurrency"),
        "active_workers": len(by_status["running"]),
        "running": sorted(by_status["running"]),
        "pending": sorted(by_status["pending"]),
        "done": sorted(by_status["done"] + by_status["skipped"]),
        "failed": sorted(by_status["failed"]),
        "pending_count": len(by_status["pending"]),
        "done_count": len(by_status["done"]) + len(by_status["skipped"]),
        "failed_count": len(by_status["failed"]),
        "simulations": {
            label: {"worker": (sims.get(label) or {}).get("status")}
            for label in sorted(sims.keys())
        },
    }


def persist_parallel_pool_checkpoint(state: Dict[str, Any]) -> None:
    """Write pool status to state.jsonl quietly (no conversation log spam)."""
    pool = state.get("parallel_pool") or {}
    if pool:
        from agentic.multi_sim_progress import sync_parallel_pool_to_multi_sim_progress

        sync_parallel_pool_to_multi_sim_progress(state)
        state["parallel_pool_status"] = snapshot_parallel_pool_status(pool)
    try:
        from agentic.utils.state_persistence import save_workflow_state_quiet

        save_workflow_state_quiet(state)
    except Exception as exc:
        logger.debug("Parallel pool checkpoint save skipped: %s", exc)


def persist_parallel_pool_interrupt(state: Dict[str, Any]) -> None:
    """
    Checkpoint pool state after interrupt (Ctrl+C, kill, disconnect).

    Resets in-flight workers to ``pending`` (parent process is gone) and syncs
    ``multi_sim_progress`` from pool + disk so ``--resume`` can continue.
    """
    pool = state.get("parallel_pool") or {}
    if not pool:
        return
    for label, rec in (pool.get("sims") or {}).items():
        if rec.get("status") == "running":
            rec["status"] = "pending"
            rec.pop("started_at", None)
            logger.info("Parallel pool interrupt: re-queued %s (was running)", label)
    state["parallel_pool"] = pool
    shutdown_pool_runner(state)
    try:
        from agentic.multi_sim_progress import (
            reconcile_multisim_progress_from_disk,
            sync_parallel_pool_to_multi_sim_progress,
        )

        reconcile_multisim_progress_from_disk(state)
        sync_parallel_pool_to_multi_sim_progress(state)
    except Exception as exc:
        logger.warning("Parallel pool interrupt reconcile skipped: %s", exc)
    state["multi_sim_phase"] = "parallel_pool"
    state["workflow_status"] = "in_progress:interrupted"
    state["plan_executed"] = False
    persist_parallel_pool_checkpoint(state)
    log_pool_to_base(
        state,
        "Parallel pool interrupted — checkpoint saved for --resume",
        log_to_conversation=False,
    )


def resume_parallel_pool_if_needed(state: Dict[str, Any]) -> bool:
    """Re-enter an interrupted parallel pool from a saved checkpoint."""
    if state.get("multi_sim_phase") != "parallel_pool":
        return False
    pool = state.get("parallel_pool")
    if not pool:
        return False
    phase = pool.get("phase") or "analysis"
    shutdown_pool_runner(state)
    from agentic.multi_sim_progress import (
        reconcile_multisim_progress_from_disk,
        sync_parallel_pool_to_multi_sim_progress,
    )

    reconcile_multisim_progress_from_disk(state)
    init_parallel_pool(state, phase=phase)
    sync_parallel_pool_to_multi_sim_progress(state)
    state["workflow_status"] = "in_progress:parallel_pool"
    state["plan_executed"] = False
    persist_parallel_pool_checkpoint(state)
    log_pool_to_base(
        state,
        f"Resuming parallel {phase} pool from checkpoint",
        pool=state.get("parallel_pool"),
        log_to_conversation=False,
    )
    return True


def init_parallel_pool(state: Dict[str, Any], *, phase: str) -> Dict[str, Any]:
    """Build or refresh ``state['parallel_pool']`` from ``sim_prompts``."""
    # One-shot: reopen previously failed labels only when the campaign (re)starts
    # with --resume / requeue_failed_sims. Cleared so mid-run failures stay terminal.
    requeue_failed = bool(state.pop("requeue_failed_sims", False))

    sim_prompts = state.get("sim_prompts") or []
    pending = sum(
        1
        for sp in sim_prompts
        if not job_already_complete(
            build_per_sim_job_spec(state, sp, phase=phase)
        )
    )
    est = estimate_workers(phase, state, pending_jobs=pending or len(sim_prompts))
    existing = state.get("parallel_pool") or {}
    agent_list = (
        ["preprocess", "simsetup"] if phase == "prep" else _analysis_agent_list(state)
    )
    # Prep→analysis (or any phase change) must not inherit terminal statuses from
    # the previous phase — that made analysis pools look finished with 0 workers.
    phase_changed = bool(existing.get("phase") and existing.get("phase") != phase)
    sims: Dict[str, Any] = {} if phase_changed else dict(existing.get("sims") or {})
    if phase_changed:
        # Preserve working_dir / label metadata when rebuilding.
        for label, old in (existing.get("sims") or {}).items():
            sims[label] = {
                "label": old.get("label") or label,
                "index": old.get("index"),
                "working_dir": old.get("working_dir") or "",
            }

    for idx, sp in enumerate(sim_prompts):
        label = sp.get("label") or f"sim_{idx}"
        wd = sp.get("working_dir") or ""
        rec = sims.get(label) or {}
        rec.setdefault("label", label)
        rec.setdefault("index", idx)
        rec["working_dir"] = wd or rec.get("working_dir") or ""
        rec["agent_list"] = agent_list
        if _is_actively_running(label, state):
            rec["status"] = "running"
        elif rec.get("status") == "running":
            rec["status"] = "pending"
            rec.pop("started_at", None)
        elif rec.get("status") == "failed":
            job = build_per_sim_job_spec(state, sp, phase=phase)
            if job_already_complete(job):
                rec["status"] = "skipped"
                rec["skip_reason"] = "artifacts on disk after prior failure"
                rec.pop("error", None)
            elif requeue_failed:
                attempts = int(rec.get("attempts") or 0)
                max_attempts = _max_sim_attempts(state)
                if attempts >= max_attempts:
                    _fail_sim_exhausted(
                        rec,
                        attempts=attempts,
                        max_attempts=max_attempts,
                        reason="incomplete artifacts",
                    )
                else:
                    rec["status"] = "pending"
                    rec.pop("error", None)
                    rec.pop("completed_at", None)
                    logger.info("Parallel pool: re-queued failed sim %s for retry", label)
            # else: keep terminal failed
        else:
            # Fatal HPC / production failure: never queue analysis/reporter.
            if phase == "analysis":
                try:
                    from agentic.utils.sim_health import should_skip_downstream_for_sim

                    decision = should_skip_downstream_for_sim(
                        state, label, working_dir=wd or rec.get("working_dir") or ""
                    )
                except Exception:
                    decision = {"skip": False}
                if decision.get("skip"):
                    rec["status"] = "failed"
                    rec["health"] = "failed"
                    rec["error"] = str(decision.get("reason") or "production MD failed")[:400]
                    rec["skip_reason"] = "downstream skipped: production not healthy"
                    rec["completed_at"] = _utc_now()
                    sims[label] = rec
                    try:
                        from agentic.utils.sim_health import record_fatal_sim_failure

                        record_fatal_sim_failure(
                            state,
                            label,
                            reason=str(rec["error"]),
                            stage="hpc",
                        )
                    except Exception:
                        logger.debug("fatal mark skipped for %s", label, exc_info=True)
                    continue
            job = build_per_sim_job_spec(state, sp, phase=phase)
            if job_already_complete(job):
                rec["status"] = "skipped"
                rec["skip_reason"] = "artifacts on disk"
                rec.pop("error", None)
            else:
                attempts = int(rec.get("attempts") or 0)
                max_attempts = _max_sim_attempts(state)
                if attempts >= max_attempts:
                    _fail_sim_exhausted(
                        rec,
                        attempts=attempts,
                        max_attempts=max_attempts,
                        reason="incomplete artifacts",
                    )
                else:
                    # Always (re)queue when this phase's artifacts are missing —
                    # never keep a prior-phase "done" without disk evidence.
                    rec["status"] = "pending"
                    rec.pop("skip_reason", None)
                    rec.pop("completed_at", None)
        _reconcile_sim_from_disk(rec, phase, label=label, state=state)
        sims[label] = rec

    pool = {
        "version": 1,
        "phase": phase,
        "max_workers": est.max_workers,
        "resource_estimate": est.to_dict(),
        "started_at": existing.get("started_at") or _utc_now(),
        "sims": sims,
    }
    state["parallel_pool"] = pool
    state["parallel_workers_resolved"] = est.max_workers
    state["parallel_pool_status"] = snapshot_parallel_pool_status(pool)
    return pool


def _analysis_agent_list(state: Dict[str, Any]) -> List[str]:
    agents = state.get("agent_list") or ["analysis", "reporter"]
    return [a for a in agents if a in {"analysis", "reporter"}] or ["analysis", "reporter"]


def _collect_finished(runner: _PoolRunner, pool: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    """Reap completed futures and update pool + multi_sim_progress."""
    finished: List[str] = []
    max_attempts = _max_sim_attempts(state)
    phase = pool.get("phase") or "analysis"
    for label, fut in list(runner.futures.items()):
        if not fut.done():
            continue
        runner.futures.pop(label, None)
        rec = (pool.get("sims") or {}).get(label) or {}
        try:
            result = fut.result()
        except Exception as exc:
            result = {"success": False, "label": label, "error": str(exc)}

        attempts = int(rec.get("attempts") or 0) + 1
        rec["attempts"] = attempts
        rec["workflow_status"] = result.get("workflow_status")

        sim_info = _sim_info_by_label(state, label) or {
            "label": label,
            "working_dir": rec.get("working_dir") or "",
        }
        job = build_per_sim_job_spec(state, sim_info, phase=phase)
        disk_ok = job_already_complete(job)
        worker_ok = bool(result.get("success"))

        if worker_ok and disk_ok:
            rec["status"] = "done"
            rec["completed_at"] = _utc_now()
            rec.pop("error", None)
        elif attempts >= max_attempts:
            reason = (
                "worker reported success but artifacts incomplete"
                if worker_ok and not disk_ok
                else (result.get("error") or "worker failed")
            )
            _fail_sim_exhausted(
                rec,
                attempts=attempts,
                max_attempts=max_attempts,
                reason=str(reason),
            )
        else:
            # Soft-fail / incomplete: re-queue until attempt budget is spent.
            rec["status"] = "pending"
            rec.pop("completed_at", None)
            rec["error"] = (
                result.get("error")
                or (
                    "worker success but required artifacts missing on disk"
                    if worker_ok
                    else "worker failed"
                )
            )
            logger.warning(
                "Parallel pool: %s attempt %s/%s incomplete (%s) — will retry",
                label,
                attempts,
                max_attempts,
                rec["error"],
            )

        pool["sims"][label] = rec
        finished.append(label)
        _sync_progress_for_label(state, label, rec)
        # Keep cross-sim HPC pool in sync so staging can overlap with remaining prep.
        if phase == "prep" and rec.get("status") in ("done", "skipped"):
            try:
                from agentic.multi_sim_hpc_pool import mark_prep_done

                mark_prep_done(state, label)
            except Exception:
                logger.debug("mark_prep_done failed for %s", label, exc_info=True)

    state["parallel_pool"] = pool
    return finished


def _sync_progress_for_label(state: Dict[str, Any], label: str, rec: Dict[str, Any]) -> None:
    from agentic.multi_sim_progress import (
        ensure_multi_sim_progress,
        mark_agent_status,
        sync_parallel_pool_to_multi_sim_progress,
    )

    progress = ensure_multi_sim_progress(state)
    if not progress:
        return
    phase = (state.get("parallel_pool") or {}).get("phase")
    if rec.get("status") in ("done", "skipped"):
        if phase == "prep":
            for agent in ("preprocessing", "simsetup"):
                mark_agent_status(state, label, agent, "done")
        else:
            for agent in ("analysis", "reporter"):
                mark_agent_status(state, label, agent, "done")
        sim_rec = (progress.get("sims") or {}).get(label)
        if sim_rec:
            sim_rec["status"] = "done"
    elif rec.get("status") == "failed":
        sim_rec = (progress.get("sims") or {}).get(label)
        if sim_rec:
            sim_rec["status"] = "failed"
            sim_rec["error"] = rec.get("error")
    elif rec.get("status") == "running":
        sync_parallel_pool_to_multi_sim_progress(state)


def _submit_pending(
    state: Dict[str, Any],
    pool: Dict[str, Any],
    runner: _PoolRunner,
) -> List[str]:
    """Launch workers for pending sims while slots remain."""
    submitted: List[str] = []
    slots = int(pool.get("max_workers") or 1) - _count_running(pool)
    if slots <= 0:
        return submitted

    phase = pool.get("phase") or "analysis"
    order = [sp.get("label") for sp in (state.get("sim_prompts") or [])]
    for label in order:
        if slots <= 0:
            break
        rec = (pool.get("sims") or {}).get(label)
        if not rec or rec.get("status") != "pending":
            continue
        sim_info = _sim_info_by_label(state, label)
        if not sim_info:
            continue
        job = build_per_sim_job_spec(state, sim_info, phase=phase)
        if job_already_complete(job):
            rec["status"] = "skipped"
            rec["skip_reason"] = "artifacts on disk"
            pool["sims"][label] = rec
            _sync_progress_for_label(state, label, rec)
            continue

        fut = runner.executor.submit(run_per_sim_workflow, job)
        runner.futures[label] = fut
        rec["status"] = "running"
        rec["started_at"] = _utc_now()
        rec.pop("error", None)
        pool["sims"][label] = rec
        submitted.append(label)
        slots -= 1
        logger.info("Parallel pool: started %s (%s phase)", label, phase)

    state["parallel_pool"] = pool
    if submitted:
        log_pool_to_base(state, f"Started workers: {', '.join(submitted)}", pool=pool)
        persist_parallel_pool_checkpoint(state)
    return submitted


def parallel_pool_supervisor_tick(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    One supervisor iteration for ``multi_sim_phase == 'parallel_pool'``.

    Sets ``state['next_node']`` to ``parallel_pool_wait`` or ``supervisor``.

    A failed simulation is terminal for that label only — other pending sims keep
    running. The phase finishes when every sim is done / skipped / failed.
    """
    pool = state.get("parallel_pool") or {}
    phase = pool.get("phase") or "analysis"
    pool = init_parallel_pool(state, phase=phase)
    max_workers = int(pool.get("max_workers") or 1)
    runner = _get_runner(state, max_workers)

    _collect_finished(runner, pool, state)
    pool = state["parallel_pool"]

    # Overlap: while prep workers run, stage HPC for sims whose prep is already done.
    if phase == "prep" and state.get("hpc_pool"):
        try:
            from agentic.multi_sim_hpc_pool import _submit_ready_sims

            _submit_ready_sims(state, state["hpc_pool"])
        except Exception:
            logger.debug("Parallel prep tick: HPC staging skipped", exc_info=True)

    failures = [
        label
        for label, rec in (pool.get("sims") or {}).items()
        if rec.get("status") == "failed"
    ]
    if failures:
        state["parallel_pool_failed"] = failures
        # Do NOT return early — keep scheduling unrelated sims.
        logger.error(
            "Parallel pool: %d failed sim(s) recorded (continuing others): %s",
            len(failures),
            failures,
        )

    if _all_done(pool):
        shutdown_pool_runner(state)
        return _finish_parallel_phase(state)

    _submit_pending(state, pool, runner)
    pool = state["parallel_pool"]
    state["parallel_pool_status"] = snapshot_parallel_pool_status(pool)

    if _count_running(pool) > 0 or _count_pending(pool) > 0:
        state["next_node"] = "parallel_pool_wait"
        persist_parallel_pool_checkpoint(state)
        return state

    # No running/pending left but pool not terminal yet (race) — re-enter supervisor.
    state["next_node"] = "supervisor"
    return state


def _finish_parallel_phase(state: Dict[str, Any]) -> Dict[str, Any]:
    """Transition after all parallel workers reach a terminal status."""
    pool = state.get("parallel_pool") or {}
    phase = pool.get("phase")
    failures = [
        label
        for label, rec in (pool.get("sims") or {}).items()
        if rec.get("status") == "failed"
    ]
    log_pool_to_base(state, f"Parallel {phase} phase complete", pool=pool)
    logger.info(
        "Parallel %s phase complete (done=%s failed=%s pending=%s)",
        phase,
        sum(
            1
            for r in (pool.get("sims") or {}).values()
            if (r or {}).get("status") in ("done", "skipped")
        ),
        sum(
            1
            for r in (pool.get("sims") or {}).values()
            if (r or {}).get("status") == "failed"
        ),
        sum(
            1
            for r in (pool.get("sims") or {}).values()
            if (r or {}).get("status") == "pending"
        ),
    )
    if failures:
        state["errors"] = list(state.get("errors") or [])
        state["errors"].append(
            f"Parallel {phase} pool finished with {len(failures)} failed "
            f"simulation(s): {', '.join(failures)} — continuing campaign"
        )
        logger.warning(
            "Parallel %s pool complete with failures (continuing): %s",
            phase,
            failures,
        )
    state.pop("parallel_pool", None)

    if phase == "prep":
        from agentic.multi_sim_hpc_pool import mark_prep_done, mark_prep_failed

        for label, rec in (pool.get("sims") or {}).items():
            if _terminal_success(rec):
                mark_prep_done(state, label)
            elif rec.get("status") == "failed":
                mark_prep_failed(
                    state,
                    label,
                    error=rec.get("error") or "parallel prep worker failed",
                )
        state.pop("hpc_pool_prep_parallel", None)
        state["hpc_pool_prep_phase_done"] = True
        state["multi_sim_phase"] = "hpc_pool"
        state["next_node"] = "supervisor"
        return state

    # analysis / post-HPC agents complete
    state["multi_sim_phase"] = "executing_sims"
    state["plan_executed"] = False
    state["current_sim_index"] = len(state.get("sim_prompts") or [])

    from agentic.supervisor.supervisor_agent import _should_run_combined_analysis

    if _should_run_combined_analysis(state):
        state["_parallel_start_combined"] = True
        state["next_node"] = "supervisor"
        return state

    state["multi_sim_phase"] = "complete"
    state["next_node"] = "final_report"
    return state


def start_parallel_agent_phase(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enter parallel pool for analysis+reporter across simulations."""
    if not should_use_parallel_pool(state):
        raise RuntimeError(
            "Refusing to start analysis/reporter parallel pool: "
            "the requested subtask contains no analysis or reporter agent"
        )
    from agentic.multi_sim_progress import (
        init_multi_sim_progress,
        reconcile_multisim_progress_from_disk,
        sync_parallel_pool_to_multi_sim_progress,
        sync_post_hpc_progress_from_disk,
    )

    # Idempotent: if analysis pool already exists, just tick (no re-announce).
    existing = state.get("parallel_pool") or {}
    if existing.get("phase") == "analysis" and existing.get("sims"):
        state["multi_sim_phase"] = "parallel_pool"
        return parallel_pool_supervisor_tick(state)

    state["multi_sim_phase"] = "parallel_pool"
    reconcile_multisim_progress_from_disk(state)
    init_multi_sim_progress(state)
    sync_post_hpc_progress_from_disk(state)
    init_parallel_pool(state, phase="analysis")
    sync_parallel_pool_to_multi_sim_progress(state)
    # Pool counts live in supervisor/pool_status.json — not conversation log.
    log_pool_to_base(
        state,
        "Starting parallel analysis/reporter pool",
        pool=state["parallel_pool"],
        log_to_conversation=False,
    )
    logger.info(
        "Parallel analysis/reporter pool started (active=%s/%s pending=%s done=%s)",
        _count_running(state["parallel_pool"]),
        (state["parallel_pool"] or {}).get("max_workers"),
        _count_pending(state["parallel_pool"]),
        len(
            [
                r
                for r in ((state["parallel_pool"] or {}).get("sims") or {}).values()
                if (r or {}).get("status") in ("done", "skipped")
            ]
        ),
    )
    return parallel_pool_supervisor_tick(state)


def _parallel_prep_still_running(state: Dict[str, Any]) -> bool:
    """True while parallel prep workers are active or pending."""
    pool = state.get("parallel_pool") or {}
    if pool.get("phase") != "prep":
        return False
    if state.get("multi_sim_phase") == "parallel_pool":
        return _count_running(pool) > 0 or _count_pending(pool) > 0
    if state.get("hpc_pool_prep_parallel"):
        return _count_running(pool) > 0 or _count_pending(pool) > 0
    return False


def start_parallel_prep_if_enabled(state: Dict[str, Any]) -> bool:
    """
    Switch HPC pool prep to parallel mode when enabled.

    Returns True when parallel prep pool was started.
    """
    if not _resources_allow_parallel_pool(state):
        return False
    init_parallel_pool(state, phase="prep")
    state["multi_sim_phase"] = "parallel_pool"
    state["hpc_pool_prep_parallel"] = True
    log_pool_to_base(
        state, "Starting parallel prep pool", pool=state["parallel_pool"], log_to_conversation=False
    )
    logger.info(
        "Parallel prep pool started (active=%s/%s)",
        _count_running(state["parallel_pool"]),
        (state["parallel_pool"] or {}).get("max_workers"),
    )
    parallel_pool_supervisor_tick(state)
    return True


def pool_summary(state: Dict[str, Any]) -> str:
    pool = state.get("parallel_pool") or {}
    active = _count_running(pool)
    max_w = pool.get("max_workers", "?")
    phase = pool.get("phase", "?")
    est = pool.get("resource_estimate") or {}
    lines = [
        f"Parallel pool [{phase}] (active {active}/{max_w} workers, "
        f"limit={est.get('limiting_factor', '?')}):"
    ]
    for label, rec in (pool.get("sims") or {}).items():
        parts = [f"status={rec.get('status')}"]
        if rec.get("error"):
            parts.append(f"error={rec['error'][:80]}")
        if rec.get("skip_reason"):
            parts.append(f"note={rec['skip_reason']}")
        lines.append(f"  {label}: {' '.join(parts)}")
    return "\n".join(lines)


def log_pool_to_base(
    state: Dict[str, Any],
    action: str,
    *,
    pool: Optional[Dict[str, Any]] = None,
    log_to_conversation: bool = False,
) -> None:
    """
    Optionally log pool milestones to ``{base}/agent_conversation.log``.

    Defaults to **off** — live counts belong in ``supervisor/pool_status.json``
    only (avoid polluting the conversation log with pool status spam).
    """
    if not log_to_conversation:
        return
    base = state.get("multi_sim_base_dir") or state.get("working_directory")
    if not base:
        return
    from agentic.utils.conversation_logger import log_agent_action, temporary_log_file

    pool = pool or state.get("parallel_pool") or {}
    snap = snapshot_parallel_pool_status(pool)
    with temporary_log_file(str(Path(base) / "agent_conversation.log")):
        log_agent_action(
            "parallel_pool",
            action,
            {
                "active": f"{_count_running(pool)}/{pool.get('max_workers', '?')}",
                "phase": pool.get("phase"),
                "pending": len(snap.get("pending") or []),
                "done": len(snap.get("done") or []) + len(snap.get("skipped") or []),
                "failed": len(snap.get("failed") or []),
                "running": len(snap.get("running") or []),
            },
        )


__all__ = [
    "should_use_parallel_pool",
    "init_parallel_pool",
    "parallel_pool_supervisor_tick",
    "start_parallel_agent_phase",
    "start_parallel_prep_if_enabled",
    "_parallel_prep_still_running",
    "pool_summary",
    "snapshot_parallel_pool_status",
    "persist_parallel_pool_checkpoint",
    "persist_parallel_pool_interrupt",
    "resume_parallel_pool_if_needed",
    "shutdown_pool_runner",
]
