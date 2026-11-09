"""
Cross-simulation HPC job pool for multi-sim workflows.

When preprocess/simsetup + HPC + analysis/reporter are requested together,
simulations are prepared sequentially and SLURM jobs are submitted up to
*max_concurrent* at a time. The supervisor polls SLURM (default every 120 min)
until all jobs finish, then runs per-sim analysis/reporter and combined work.

Hybrid execution: blocks with sleep while the process is alive; ``--resume``
restores pool state from ``{base}/supervisor/state.jsonl``.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TERMINAL_SUCCESS = frozenset({"COMPLETED", "COMPLETING"})
_TERMINAL_FAILURE = frozenset({"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"})
_PRODUCTION_TRAJECTORIES = ("mdWrap.xtc", "md.xtc", "prod.xtc", "production.xtc")


def reuse_hpc_enabled(state: Optional[Dict[str, Any]] = None) -> bool:
    """True when ``--reuse-hpc`` / ``--skip-hpc-submit`` / env is active (skip sbatch).

    ``--skip-hpc-submit`` is the preferred CLI for campaigns that still run
    preprocess+simsetup normally but must not call sbatch (seeded trajs).
    """
    if state:
        if state.get("reuse_hpc") or state.get("skip_hpc_submit"):
            return True
    return os.environ.get("AGENTIC_REUSE_HPC", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ) or os.environ.get("AGENTIC_SKIP_HPC_SUBMIT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _reuse_hpc_files_ready(sim_dir: str, hpc_dir: Optional[str] = None) -> bool:
    """Ready under ``--reuse-hpc``: production ``md.tpr`` + trajectory, no md.log.

    Accepts a flat ``hpc/`` directory and nested ``hpc/rep01`` / ``hpc/rep02``
    seeds. Every nested replicate must be present.
    """
    from agentic.utils.sim_health import production_health

    try:
        return bool(
            production_health(
                sim_dir, hpc_dir=hpc_dir, reuse_hpc=True
            ).get("healthy")
        )
    except Exception:
        hpc = Path(hpc_dir) if hpc_dir else (Path(sim_dir) / "hpc")
        tpr = hpc / "md.tpr"
        xtc = hpc / "mdWrap.xtc"
        try:
            return (
                tpr.is_file()
                and tpr.stat().st_size > 0
                and xtc.is_file()
                and xtc.stat().st_size > 0
            )
        except OSError:
            return False


def _md_production_finished(hpc_dir: Path) -> bool:
    """True when production ``md.log`` reports mdrun finished (not equil logs)."""
    md_log = hpc_dir / "md.log"
    if not md_log.is_file():
        return False
    try:
        text = md_log.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "Finished mdrun" in text


def _sim_production_trajectory_ready(
    sim_dir: str,
    state: Optional[Dict[str, Any]] = None,
    hpc_dir: Optional[str] = None,
) -> bool:
    """Production trajectory complete — equilibration ``nvt``/``npt`` files do not count.

    With ``--reuse-hpc``, production ``md.tpr`` + trajectory are required (no md.log).
    Flat ``hpc/`` and nested ``hpc/rep01`` / ``hpc/rep02`` both count. Pass
    ``hpc_dir`` to check one replicate directory.
    """
    from agentic.utils.sim_health import production_health, is_reuse_hpc

    return bool(
        production_health(
            sim_dir,
            hpc_dir=hpc_dir,
            reuse_hpc=is_reuse_hpc(state) if state is not None else reuse_hpc_enabled(state),
        ).get("healthy")
    )


def _sim_trajectory_ready(
    sim_dir: str,
    state: Optional[Dict[str, Any]] = None,
) -> bool:
    """Backward-compatible alias used by supervisor helpers."""
    return _sim_production_trajectory_ready(sim_dir, state=state)


def _rec_hpc_dir(rec: Dict[str, Any]) -> Optional[str]:
    hpc = rec.get("hpc_dir")
    if hpc:
        return str(hpc)
    wd = rec.get("working_dir") or ""
    return str(Path(wd) / "hpc") if wd else None


def _rec_prod_ready(rec: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> bool:
    wd = rec.get("working_dir") or ""
    return _sim_production_trajectory_ready(wd, state=state, hpc_dir=rec.get("hpc_dir"))


def _slot_parent_label(slot_key: str, rec: Optional[Dict[str, Any]] = None) -> str:
    if rec and rec.get("parent_label"):
        return str(rec["parent_label"])
    from src.analysis.replicate_paths import split_pool_slot_key

    label, _ = split_pool_slot_key(slot_key)
    return label


def _iter_pool_slot_order(state: Dict[str, Any], pool: Dict[str, Any]) -> List[str]:
    """Ordered pool keys: sim_prompt order × rep01…repN (or legacy label)."""
    from src.analysis.replicate_paths import (
        build_rep_plan,
        normalize_rep_num,
        uses_nested_reps,
    )

    sims = pool.get("sims") or {}
    rep_num = normalize_rep_num(state.get("rep_num", 1))
    base_seed = int(state.get("replicate_base_seed") or 12345)
    order: List[str] = []
    for idx, sp in enumerate(state.get("sim_prompts") or []):
        label = sp.get("label") or f"sim_{idx}"
        wd = sp.get("working_dir") or ""
        for slot in build_rep_plan(label, wd, rep_num, base_seed=base_seed):
            key = slot["slot_key"]
            if key in sims:
                order.append(key)
            elif not uses_nested_reps(rep_num) and label in sims:
                order.append(label)
                break
    # Append any leftover keys (resume edge cases)
    for key in sims:
        if key not in order:
            order.append(key)
    return order


def _label_pool_recs(pool: Dict[str, Any], label: str) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for key, rec in (pool.get("sims") or {}).items():
        if _slot_parent_label(key, rec) == label:
            out.append((key, rec))
    return out


def _log_hpc_pool_event(
    sim_working_dir: str,
    label: str,
    event: str,
    rec: Dict[str, Any],
) -> None:
    """Write HPC pool SLURM sync events to the per-simulation conversation log."""
    from agentic.utils.conversation_logger import (
        log_agent_action,
        log_agent_completion,
        temporary_log_file,
    )

    log_path = str(Path(sim_working_dir) / "agent_conversation.log")
    Path(sim_working_dir).mkdir(parents=True, exist_ok=True)
    with temporary_log_file(log_path):
        if event == "completed":
            log_agent_completion(
                "hpc",
                "HPC Job Submission (pool mode)",
                {
                    "mode": "hpc_pool",
                    "simulation": label,
                    "job_id": rec.get("job_id"),
                    "job_status": rec.get("last_slurm_state") or "COMPLETED",
                    "trajectory_ready": rec.get("trajectory_ready"),
                },
                True,
            )
        else:
            log_agent_action(
                "hpc",
                f"HPC pool: {event}",
                {
                    "simulation": label,
                    "job_id": rec.get("job_id"),
                    "slurm_state": rec.get("last_slurm_state"),
                    "hpc_status": rec.get("hpc_status"),
                },
            )


def _base_conversation_log(state: Dict[str, Any]) -> Optional[str]:
    base = state.get("multi_sim_base_dir") or state.get("working_directory")
    if not base:
        return None
    return str(Path(base) / "agent_conversation.log")


def _label_for_job_id(pool: Dict[str, Any], job_id: str) -> Optional[str]:
    for label, rec in (pool.get("sims") or {}).items():
        if str(rec.get("job_id") or "") == job_id:
            return label
    return None


def _slurm_snapshot_for_pool(pool: Dict[str, Any], by_id: Dict[str, Any]) -> Dict[str, Any]:
    pool_ids = {
        str(rec.get("job_id"))
        for rec in (pool.get("sims") or {}).values()
        if rec.get("job_id")
    }
    pool_jobs: List[Dict[str, Any]] = []
    for jid in sorted(pool_ids):
        job = by_id.get(jid) or {}
        pool_jobs.append(
            {
                "sim": _label_for_job_id(pool, jid),
                "job_id": jid,
                "state": job.get("state", "NOT_IN_QUEUE"),
                "partition": job.get("partition"),
                "time_left": job.get("time_left"),
                "name": job.get("name"),
            }
        )
    return {
        "pool_jobs_in_queue": pool_jobs,
        "jobs_in_sq_total": len(by_id),
    }


def log_pool_to_base(
    state: Dict[str, Any],
    action: str,
    *,
    pool: Optional[Dict[str, Any]] = None,
    by_id: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    log_to_conversation: bool = True,
) -> None:
    """Optionally log cross-sim pool milestones to ``{base}/agent_conversation.log``.

    Routine SLURM poll / wait status belongs in ``supervisor/pool_status.json``
    (via ``persist_hpc_pool_checkpoint``), not the conversation log. Pass
    ``log_to_conversation=False`` for those updates.
    """
    if not log_to_conversation:
        return
    log_path = _base_conversation_log(state)
    if not log_path:
        return
    from agentic.utils.conversation_logger import (
        log_agent_action,
        temporary_log_file,
    )

    pool = pool or state.get("hpc_pool") or {}
    details: Dict[str, Any] = {
        "active_slots": f"{_count_running(pool)}/{pool.get('max_concurrent', '?')}",
        # Compact counts only — full per-sim listing lives in pool_status.json.
        "running": _count_running(pool),
        "done_or_skipped": sum(
            1
            for r in (pool.get("sims") or {}).values()
            if (r.get("hpc_status") or "") in ("done", "skipped", "completed")
        ),
        "failed": sum(
            1
            for r in (pool.get("sims") or {}).values()
            if (r.get("hpc_status") or "") == "failed"
        ),
        "pending": sum(
            1
            for r in (pool.get("sims") or {}).values()
            if (r.get("hpc_status") or "") == "pending"
        ),
    }
    if by_id is not None:
        details["slurm_queue"] = _slurm_snapshot_for_pool(pool, by_id)
    if extra:
        details.update(extra)
    with temporary_log_file(log_path):
        log_agent_action("hpc_pool", action, details)


def parse_hpc_check_interval(value: Any, default_sec: int = 7200) -> int:
    """Parse ``2h``, ``120m``, ``7200``, or seconds int."""
    if value is None:
        return default_sec
    if isinstance(value, (int, float)):
        return max(60, int(value))
    text = str(value).strip().lower()
    if not text:
        return default_sec
    if text.isdigit():
        return max(60, int(text))
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([smhd])$", text)
    if not m:
        return default_sec
    num = float(m.group(1))
    unit = m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return max(60, int(num * mult))


def should_use_hpc_pool(state: Dict[str, Any]) -> bool:
    """True when a full pipeline should wait on SLURM before analysis/reporter.

    Applies whenever preprocess/simsetup + HPC +
    post-simulation agents are requested together. Disable with
    ``hpc_pool_disabled`` / analysis-only subtasks.
    """
    if state.get("combined_only") or state.get("hpc_pool_disabled"):
        return False
    if state.get("use_hpc_pool") is False:
        return False
    # Parallel analysis/reporter workers must never enter HPC wait.
    if state.get("pool_phase") == "analysis" or state.get("post_hpc_analysis_only"):
        return False
    if state.get("hpc_pool_phase_complete"):
        return False
    from src.supervisor.unified_enricher import get_agent_execution_order

    agents = get_agent_execution_order(state.get("subtask_type") or "full_task", state)
    pre = any(a in agents for a in ("preprocessing", "simsetup"))
    post = any(a in agents for a in ("analysis", "reporter"))
    return pre and "hpc" in agents and post


def ensure_hpc_pool_sim_prompts(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ensure ``sim_prompts`` exists for HPC pool (last-resort N=1 fallback).

    Prefer the master planner's ``sim_prompts``. When missing (legacy resume),
    synthesize one entry from the source PDB stem under ``{base}/{label}/`` —
    never from the working-dir basename.
    """
    existing = state.get("sim_prompts") or []
    if existing:
        return existing

    base = state.get("multi_sim_base_dir") or state.get("working_directory") or "."
    base_path = Path(base).expanduser().resolve()

    pdb = state.get("raw_pdb")
    if not pdb:
        plist = state.get("pdb_list") or []
        pdb = plist[0] if plist else None
    label = Path(pdb).stem if pdb else "sim_0"
    sim_wd = base_path / label
    entry = {
        "label": label,
        "working_dir": str(sim_wd),
        "prompt": state.get("user_goal")
        or state.get("enriched_prompt")
        or state.get("user_goal_original")
        or "",
        "pdb_file": pdb,
    }
    state["sim_prompts"] = [entry]
    state.setdefault("multi_sim_base_dir", str(base_path))
    state["run_combined_analysis"] = False
    state["is_multi_simulation"] = True
    logger.info(
        "HPC pool: synthesized sim_prompts label=%s wd=%s (pdb stem fallback)",
        label,
        sim_wd,
    )
    return state["sim_prompts"]


def _goal_requests_continuation(state: Dict[str, Any]) -> bool:
    """True when the multi-sim goal asks to extend existing production MD."""
    if str(state.get("hpc_action") or "").lower() in {"continue", "extend"}:
        return True
    text = " ".join(
        str(state.get(key) or "")
        for key in ("user_goal", "user_goal_original", "enriched_prompt", "hpc_instructions")
    ).lower()
    for sp in state.get("sim_prompts") or []:
        text += " " + str(sp.get("prompt") or "").lower()
    return any(
        word in text
        for word in (
            "extend simulation",
            "continue simulation",
            "continuation",
            "extend md",
            "continue md",
            "inspect_gromacs_continuation",
            "prepare_gromacs_continuation",
            "target_total_ns",
        )
    )


def _should_skip_existing_production(
    state: Dict[str, Any],
    sim_dir: str,
    *,
    hpc_dir: Optional[str] = None,
) -> bool:
    """Skip prep+HPC for finished production runs that are not being continued.

    Under ``--reuse-hpc``, return False so preprocess/simsetup still run; only
    ``sbatch`` is blocked later when ``md.tpr``/``mdWrap.xtc`` are present.
    """
    if reuse_hpc_enabled(state):
        return False
    if not _sim_production_trajectory_ready(sim_dir, state=state, hpc_dir=hpc_dir):
        return False
    return not _goal_requests_continuation(state)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sim_setup_ready(sim_dir: str) -> bool:
    """True when simsetup can stage a complete HPC job (SLURM expects ``system.gro``).

    Requires topology, MDP files, and specifically ``simsetup/system.gro``.
    ``solvated.gro`` alone is not enough — ion addition must finish and write
    ``system.gro`` before prep is marked done / jobs are submitted.
    """
    simsetup = Path(sim_dir) / "simsetup"
    if not simsetup.is_dir():
        return False
    has_top = (simsetup / "topol.top").is_file() or any(simsetup.glob("*.top"))
    if not has_top:
        return False
    has_system_gro = (simsetup / "system.gro").is_file()
    has_mdp = any(simsetup.glob("*.mdp"))
    return bool(has_top and has_system_gro and has_mdp)


def _requested_pipeline_agents(state: Optional[Dict[str, Any]]) -> List[str]:
    """CLI / full-pipeline agents, ignoring the temporary prep-only filter."""
    state = state or {}
    for key in ("pipeline_agent_list", "requested_agent_list", "cli_agent_list"):
        agents = state.get(key)
        if agents:
            return [str(a) for a in agents]
    al = [str(a) for a in (state.get("agent_list") or [])]
    prep_only = {"preprocess", "simsetup", "preprocessing"}
    if not al or set(al) <= prep_only:
        return ["preprocess", "simsetup", "hpcjob", "analysis", "reporter"]
    return al


def _revalidate_pool_sim(
    rec: Dict[str, Any],
    *,
    skip_existing_production: bool = True,
    state: Optional[Dict[str, Any]] = None,
) -> None:
    """Fix spurious skip/complete flags when only equilibration trajectories exist."""
    wd = rec.get("working_dir") or ""
    prod_ready = _rec_prod_ready(rec, state)
    hpc = rec.get("hpc_status")

    # --reuse-hpc: never skip preprocess/simsetup. Keep HPC *pending* until
    # ``_submit_ready_sims`` stages files + SLURM script, then marks skipped.
    # Premature skip here blocked staging and left hpc/ as traj-only.
    if prod_ready and reuse_hpc_enabled(state):
        if rec.get("prep_status") == "failed":
            # Fail-soft: still allow HPC staging (no sbatch) so analysis can run.
            if rec.get("hpc_status") not in ("submitted", "running", "completed", "skipped"):
                rec["hpc_status"] = "pending"
            elif (
                rec.get("hpc_status") == "skipped"
                and rec.get("skip_reason") == "reuse-hpc"
                and not rec.get("job_script")
            ):
                rec["hpc_status"] = "pending"
                rec.pop("skip_reason", None)
            return
        if rec.get("prep_status") in ("done", "skipped") or _sim_setup_ready(wd):
            if rec.get("prep_status") not in ("done", "skipped"):
                rec["prep_status"] = "done"
            if rec.get("hpc_status") not in (
                "submitted",
                "running",
                "completed",
                "skipped",
                "failed",
            ):
                rec["hpc_status"] = "pending"
            elif (
                rec.get("hpc_status") == "skipped"
                and rec.get("skip_reason") == "reuse-hpc"
                and not rec.get("job_script")
            ):
                # Re-open once so staging can run after a prior premature skip.
                rec["hpc_status"] = "pending"
                rec.pop("skip_reason", None)
        elif rec.get("prep_status") not in ("done", "failed", "skipped"):
            rec["prep_status"] = "pending"
            if rec.get("hpc_status") not in ("submitted", "running", "completed", "skipped"):
                rec["hpc_status"] = "pending"
        return

    if prod_ready and skip_existing_production:
        rec["prep_status"] = "skipped"
        rec["hpc_status"] = "skipped"
        rec["skip_reason"] = "production trajectory on disk"
        rec.pop("job_id", None)
        rec.pop("job_script", None)
        rec.pop("last_slurm_state", None)
        return

    if prod_ready and not skip_existing_production:
        # Continuation: keep setup skipped, but leave HPC pending unless already active.
        rec["prep_status"] = "skipped"
        if rec.get("hpc_status") not in ("submitted", "running", "completed", "failed"):
            rec["hpc_status"] = "pending"
        rec.pop("skip_reason", None)
        return

    if hpc in ("skipped", "completed"):
        jid = rec.get("job_id")
        last = str(rec.get("last_slurm_state") or "").upper()
        if jid and last in ("RUNNING", "COMPLETING", "R"):
            rec["hpc_status"] = "running"
        elif jid and last in ("PENDING", "PD"):
            rec["hpc_status"] = "submitted"
        elif jid:
            rec["hpc_status"] = "submitted"
        elif _sim_setup_ready(wd):
            rec["prep_status"] = "done"
            rec["hpc_status"] = "pending"
        else:
            rec["prep_status"] = "pending"
            rec["hpc_status"] = "pending"
        rec.pop("skip_reason", None)
        if rec.get("hpc_status") == "pending":
            rec.pop("job_id", None)
            rec.pop("job_script", None)
            rec.pop("last_slurm_state", None)


def init_hpc_pool(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build or refresh ``state['hpc_pool']`` from ``sim_prompts``.

    When ``rep_num > 1``, expands each label into ``label::repXX`` slots with
    nested ``hpc/repXX`` paths. Prep remains shared per chemical label.
    """
    from agentic.parallel_resources import resolve_allowed_hpc_jobs
    from src.analysis.replicate_paths import build_rep_plan, normalize_rep_num

    sim_prompts = state.get("sim_prompts") or []
    max_concurrent = resolve_allowed_hpc_jobs(state)
    interval = parse_hpc_check_interval(
        state.get("hpc_check_interval"),
        default_sec=int(state.get("hpc_check_interval_sec") or 7200),
    )
    existing = state.get("hpc_pool") or {}
    existing_sims: Dict[str, Any] = dict(existing.get("sims") or {})
    skip_existing_production = not _goal_requests_continuation(state)
    rep_num = normalize_rep_num(state.get("rep_num", 1))
    base_seed = int(state.get("replicate_base_seed") or 12345)
    sims: Dict[str, Any] = {}

    for idx, sp in enumerate(sim_prompts):
        label = sp.get("label") or f"sim_{idx}"
        wd = sp.get("working_dir") or ""
        plan = build_rep_plan(label, wd, rep_num, base_seed=base_seed)
        # Shared prep status: any prior slot for this label, or legacy flat key
        shared_prep = None
        for key, erec in existing_sims.items():
            if _slot_parent_label(key, erec) == label and erec.get("prep_status"):
                shared_prep = erec.get("prep_status")
                break
        if shared_prep is None and label in existing_sims:
            shared_prep = (existing_sims.get(label) or {}).get("prep_status")

        for slot in plan:
            key = slot["slot_key"]
            rec = dict(existing_sims.get(key) or {})
            # Migrate one-time from legacy flat label record (rep_num==1 or first expand)
            if not rec and label in existing_sims and key == label:
                rec = dict(existing_sims[label])
            elif not rec and label in existing_sims and slot.get("nested"):
                # Do not copy job_id from flat label onto every nested rep
                legacy = existing_sims[label]
                if legacy.get("prep_status"):
                    rec["prep_status"] = legacy.get("prep_status")
                if legacy.get("error") and legacy.get("prep_status") == "failed":
                    rec["error"] = legacy.get("error")

            rec["label"] = key
            rec["parent_label"] = label
            rec["index"] = idx
            rec["working_dir"] = wd
            rec["rep_id"] = slot["rep_id"]
            rec["rep_index"] = slot["rep_index"]
            rec["rep_num"] = slot["rep_num"]
            rec["seed"] = slot["seed"]
            rec["hpc_dir"] = slot["hpc_dir"]
            rec["analysis_dir"] = slot["analysis_dir"]
            if shared_prep and rec.get("prep_status") not in ("done", "failed", "skipped"):
                rec["prep_status"] = shared_prep

            hpc_dir = slot["hpc_dir"]
            if _should_skip_existing_production(state, wd, hpc_dir=hpc_dir):
                rec["prep_status"] = "skipped"
                rec["hpc_status"] = "skipped"
                rec["skip_reason"] = "production trajectory on disk"
                rec.pop("job_id", None)
                rec.pop("job_script", None)
                rec.pop("last_slurm_state", None)
            elif reuse_hpc_enabled(state) and _sim_production_trajectory_ready(
                wd, state=state, hpc_dir=hpc_dir
            ):
                if _sim_setup_ready(wd):
                    rec["prep_status"] = "done"
                    if rec.get("hpc_status") not in (
                        "submitted",
                        "running",
                        "completed",
                        "skipped",
                        "failed",
                    ):
                        rec["hpc_status"] = "pending"
                elif rec.get("prep_status") not in ("done", "failed", "skipped"):
                    rec["prep_status"] = "pending"
                    if rec.get("hpc_status") not in (
                        "submitted",
                        "running",
                        "completed",
                        "skipped",
                    ):
                        rec["hpc_status"] = "pending"
            elif _sim_setup_ready(wd):
                rec["prep_status"] = "done"
                if rec.get("hpc_status") not in (
                    "submitted",
                    "running",
                    "completed",
                    "skipped",
                    "failed",
                ):
                    rec["hpc_status"] = "pending"
            elif rec.get("prep_status") == "failed":
                if state.get("requeue_failed_sims"):
                    rec["prep_status"] = "pending"
                    rec.pop("error", None)
                    if rec.get("hpc_status") not in (
                        "submitted",
                        "running",
                        "completed",
                        "skipped",
                    ):
                        rec["hpc_status"] = "pending"
                elif reuse_hpc_enabled(state) and _sim_production_trajectory_ready(
                    wd, state=state, hpc_dir=hpc_dir
                ):
                    if rec.get("hpc_status") not in (
                        "submitted",
                        "running",
                        "completed",
                        "skipped",
                    ):
                        rec["hpc_status"] = "pending"
                else:
                    rec["hpc_status"] = "failed"
            else:
                if rec.get("prep_status") == "done":
                    logger.warning(
                        "HPC pool: %s marked prep done but simsetup not ready — resetting",
                        key,
                    )
                rec["prep_status"] = "pending"
                if rec.get("hpc_status") == "failed" and not rec.get("job_id"):
                    rec["hpc_status"] = "pending"
                    rec.pop("error", None)
                elif rec.get("hpc_status") not in (
                    "submitted",
                    "running",
                    "completed",
                    "skipped",
                ):
                    rec["hpc_status"] = "pending"

            _revalidate_pool_sim(
                rec,
                skip_existing_production=skip_existing_production,
                state=state,
            )
            sims[key] = rec

    prep_cursor = existing.get("prep_cursor", 0)
    pool = {
        "version": 2 if rep_num > 1 else 1,
        "rep_num": rep_num,
        "max_concurrent": max_concurrent,
        "check_interval_sec": interval,
        "last_check_at": existing.get("last_check_at"),
        "awaiting_hitl": existing.get("awaiting_hitl", False),
        "hitl_reason": existing.get("hitl_reason"),
        "sims": sims,
        "prep_cursor": prep_cursor,
        "phase": existing.get("phase") or "active",
    }
    state["hpc_pool"] = pool
    state["allowed_hpc_jobs"] = max_concurrent
    state["hpc_check_interval_sec"] = interval
    _sync_pool_reps_into_progress(state)
    return pool


def _sync_pool_reps_into_progress(state: Dict[str, Any]) -> None:
    """Mirror ``hpc_pool`` slot statuses into ``multi_sim_progress.sims[label].reps``."""
    progress = state.get("multi_sim_progress") or {}
    sims_prog = progress.get("sims") or {}
    if not sims_prog:
        return
    pool = state.get("hpc_pool") or {}
    for key, rec in (pool.get("sims") or {}).items():
        parent = rec.get("parent_label") or _slot_parent_label(key, rec)
        preg = sims_prog.get(parent)
        if not preg:
            continue
        reps = preg.setdefault("reps", {})
        rid = rec.get("rep_id") or "rep01"
        entry = reps.setdefault(rid, {})
        entry["hpc_status"] = rec.get("hpc_status")
        entry["job_id"] = rec.get("job_id")
        entry["seed"] = rec.get("seed")
        if rec.get("hpc_status") in ("completed", "skipped") and _rec_prod_ready(rec, state):
            entry.setdefault("analysis_status", entry.get("analysis_status") or "pending")
    state["multi_sim_progress"] = progress


def _count_running(pool: Dict[str, Any]) -> int:
    n = 0
    for rec in (pool.get("sims") or {}).values():
        if rec.get("hpc_status") in ("submitted", "running"):
            n += 1
    return n


def _all_hpc_done(pool: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> bool:
    if _count_running(pool) > 0:
        return False
    sims = pool.get("sims") or {}
    if not sims:
        return False
    for rec in sims.values():
        wd = rec.get("working_dir") or ""
        # Fail-soft HPC failure is terminal for that sim only.
        if rec.get("hpc_status") == "failed":
            continue
        # Prep failure: under --reuse-hpc still wait for staging/skip when traj exists.
        if rec.get("prep_status") == "failed":
            if reuse_hpc_enabled(state) and _rec_prod_ready(rec, state):
                st = rec.get("hpc_status")
                if st == "pending":
                    return False
                if st == "skipped" and not _rec_prod_ready(rec, state):
                    return False
                if st not in ("skipped", "completed"):
                    return False
            continue
        st = rec.get("hpc_status")
        if st == "skipped":
            if not _rec_prod_ready(rec, state):
                return False
        elif st != "completed":
            return False
        elif not _rec_prod_ready(rec, state):
            return False
    return True


def _sync_jobs_from_slurm(state: Dict[str, Any], pool: Dict[str, Any], *, force: bool = False) -> None:
    """Update pool sim records from ``list_my_slurm_jobs`` / ``check_job_status``."""
    from src.hpc.job_monitor import list_my_slurm_jobs, check_job_status

    now = _utc_now()
    last = pool.get("last_check_at")
    interval = int(pool.get("check_interval_sec") or 7200)
    if not force and last:
        try:
            prev = datetime.fromisoformat(last.replace("Z", "+00:00"))
            if (now - prev).total_seconds() < interval:
                return
        except ValueError:
            pass

    queue = list_my_slurm_jobs.func()
    by_id = {}
    if queue.get("success"):
        for job in queue.get("jobs") or []:
            by_id[str(job.get("job_id"))] = job

    for label, rec in (pool.get("sims") or {}).items():
        st = rec.get("hpc_status")
        if st not in ("submitted", "running"):
            continue
        jid = str(rec.get("job_id") or "")
        if not jid:
            continue
        prev_status = st
        wd = rec.get("working_dir") or ""
        if jid in by_id:
            slurm_state = str(by_id[jid].get("state", "RUNNING")).upper()
            rec["last_slurm_state"] = slurm_state
            if slurm_state in ("RUNNING", "COMPLETING"):
                rec["hpc_status"] = "running"
            elif slurm_state == "PENDING":
                rec["hpc_status"] = "submitted"
            continue

        status = check_job_status.func(job_id=jid)
        if not status.get("success"):
            continue
        slurm_state = str(status.get("status", "UNKNOWN")).upper()
        rec["last_slurm_state"] = slurm_state
        if slurm_state in _TERMINAL_SUCCESS or slurm_state == "COMPLETED":
            if _rec_prod_ready(rec, state):
                rec["hpc_status"] = "completed"
                rec["trajectory_ready"] = True
                if prev_status != "completed":
                    _log_hpc_pool_event(wd, label, "completed", rec)
            else:
                rec["hpc_status"] = "running"
                logger.info(
                    "HPC pool: %s job %s left SLURM queue but production trajectory not ready yet",
                    label,
                    jid,
                )
        elif slurm_state in _TERMINAL_FAILURE:
            rec["hpc_status"] = "failed"
            rec["health"] = "failed"
            rec["error"] = f"SLURM job {jid} ended with {slurm_state}"
            _log_hpc_pool_event(wd, label, f"failed ({slurm_state})", rec)
            try:
                from agentic.utils.sim_health import record_fatal_sim_failure

                record_fatal_sim_failure(
                    state,
                    label,
                    reason=str(rec["error"]),
                    stage="hpc",
                )
            except Exception:
                logger.debug("fatal-sim mark skipped for %s", label, exc_info=True)
        elif slurm_state in ("RUNNING", "R"):
            rec["hpc_status"] = "running"
        elif slurm_state in ("PENDING", "PD"):
            rec["hpc_status"] = "submitted"

    pool["last_check_at"] = now.isoformat()
    state["hpc_pool"] = pool
    _sync_pool_reps_into_progress(state)
    # Live status → pool_status.json only (not agent_conversation.log).
    log_pool_to_base(
        state, "SLURM poll sync (sq --me)", pool=pool, by_id=by_id, log_to_conversation=False
    )


def _submit_ready_sims(state: Dict[str, Any], pool: Dict[str, Any]) -> List[str]:
    """Submit/stage jobs for prepared sims while slots remain. Returns slot keys submitted.

    Under ``--reuse-hpc``, staging (copy + SLURM script, no sbatch) is parallelized
    across ready sims so the pool does not look single-threaded.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from agentic.hpc.pool_submit import submit_simulation_job, prepare_hpc_without_submit
    from src.hpc.job_monitor import list_my_slurm_jobs

    submitted: List[str] = []
    slots = int(pool.get("max_concurrent") or 5) - _count_running(pool)
    order = _iter_pool_slot_order(state, pool)
    sims = pool.get("sims") or {}

    def _prep_allows_hpc(rec: Dict[str, Any]) -> bool:
        if rec.get("prep_status") in ("done", "skipped"):
            return True
        return bool(
            reuse_hpc_enabled(state)
            and rec.get("prep_status") == "failed"
            and _rec_prod_ready(rec, state)
        )

    # --- Parallel reuse-hpc staging (no SLURM slots needed) ---
    reuse_batch: List[tuple] = []
    if reuse_hpc_enabled(state):
        for key in order:
            rec = sims.get(key)
            if not rec or rec.get("hpc_status") != "pending":
                continue
            if not _prep_allows_hpc(rec):
                continue
            if _rec_prod_ready(rec, state):
                reuse_batch.append((key, rec))

    if reuse_batch:
        max_workers = min(8, max(1, len(reuse_batch)))
        logger.info(
            "HPC pool: reuse-hpc parallel staging for %d sim(s) (workers=%d)",
            len(reuse_batch),
            max_workers,
        )

        def _stage_one(item):
            key, rec = item
            parent = rec.get("parent_label") or _slot_parent_label(key, rec)
            wd = rec.get("working_dir") or ""
            return key, prepare_hpc_without_submit(
                wd,
                parent,
                workflow_state=state,
                hpc_dir=rec.get("hpc_dir"),
                rep_id=rec.get("rep_id"),
                seed=rec.get("seed"),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_stage_one, item) for item in reuse_batch]
            for fut in as_completed(futs):
                key, prep_hpc = fut.result()
                rec = sims.get(key) or {}
                if prep_hpc.get("job_script"):
                    rec["job_script"] = prep_hpc.get("job_script")
                    rec["job_name"] = prep_hpc.get("job_name")
                rec["hpc_status"] = "skipped"
                rec["skip_reason"] = "reuse-hpc"
                rec.pop("job_id", None)
                sims[key] = rec
                logger.info(
                    "HPC pool: %s reuse-hpc — staged HPC files, skipped sbatch (%s)",
                    key,
                    prep_hpc.get("message") or "ok",
                )

    # --- Real sbatch path (slot-limited) ---
    if slots <= 0 and not reuse_hpc_enabled(state):
        state["hpc_pool"] = pool
        return submitted

    for key in order:
        if slots <= 0:
            break
        rec = sims.get(key)
        if not rec:
            continue
        wd = rec.get("working_dir") or ""
        parent = rec.get("parent_label") or _slot_parent_label(key, rec)
        if not _prep_allows_hpc(rec):
            continue
        if rec.get("hpc_status") != "pending":
            continue
        if _rec_prod_ready(rec, state):
            # Non-reuse: skip submit when traj already present.
            rec["hpc_status"] = "skipped"
            rec["skip_reason"] = "production trajectory on disk"
            rec.pop("job_id", None)
            continue
        if reuse_hpc_enabled(state):
            rec["hpc_status"] = "failed"
            rec["error"] = "sbatch blocked: --reuse-hpc active but md.tpr/mdWrap.xtc not ready"
            logger.error(
                "HPC pool: refusing submit for %s under --reuse-hpc (trajectory not ready)",
                key,
            )
            continue
        from src.hpc.job_submitter import find_live_slurm_job_for_dir

        hpc_path = rec.get("hpc_dir") or str(Path(wd) / "hpc")
        live = find_live_slurm_job_for_dir(hpc_path)
        if live:
            rec["hpc_status"] = "submitted"
            rec["job_id"] = live
            rec.pop("error", None)
            slots -= 1
            try:
                from agentic.hpc.job_markers import write_job_marker

                write_job_marker(hpc_path, live)
            except Exception:
                logger.debug("Could not write .agentic_job_id marker", exc_info=True)
            logger.info(
                "HPC pool: %s already queued as job_id=%s — skipping sbatch",
                key,
                live,
            )
            continue
        result = submit_simulation_job(
            wd,
            parent,
            workflow_state=state,
            hpc_dir=rec.get("hpc_dir"),
            rep_id=rec.get("rep_id") if (rec.get("rep_num") or 1) > 1 else None,
            seed=rec.get("seed") if (rec.get("rep_num") or 1) > 1 else None,
        )
        if result.get("success"):
            rec["hpc_status"] = "submitted"
            rec["job_id"] = result.get("job_id")
            rec["job_script"] = result.get("job_script")
            rec["job_name"] = result.get("job_name")
            rec.pop("error", None)
            submitted.append(key)
            slots -= 1
            if result.get("status") == "ALREADY_QUEUED":
                logger.info(
                    "HPC pool: %s already queued as job_id=%s — skipping sbatch",
                    key,
                    rec.get("job_id"),
                )
            else:
                logger.info("HPC pool: submitted %s job_id=%s", key, rec.get("job_id"))
        else:
            err = result.get("error", "submit failed")
            if not _sim_setup_ready(wd):
                rec["prep_status"] = "pending"
                rec["hpc_status"] = "pending"
                rec.pop("error", None)
                logger.warning(
                    "HPC pool: %s submit skipped — prep incomplete (%s)",
                    key,
                    err,
                )
                continue
            rec["hpc_status"] = "failed"
            rec["error"] = err
            logger.error(
                "HPC pool: submit failed for %s: %s — continuing other sims",
                key,
                rec["error"],
            )
            continue

    if submitted:
        queue = list_my_slurm_jobs.func()
        by_id = {}
        if queue.get("success"):
            for job in queue.get("jobs") or []:
                by_id[str(job.get("job_id"))] = job
        log_pool_to_base(
            state,
            f"Submitted pool jobs: {', '.join(submitted)}",
            pool=pool,
            by_id=by_id,
            extra={"submitted_labels": submitted},
        )

    state["hpc_pool"] = pool
    _sync_pool_reps_into_progress(state)
    return submitted


def _next_prep_label(pool: Dict[str, Any], sim_prompts: List[Dict]) -> Optional[str]:
    """Return next chemical label that still needs preprocess/simsetup (once per label)."""
    for sp in sim_prompts:
        label = sp.get("label")
        if not label:
            continue
        slots = _label_pool_recs(pool, label)
        if not slots:
            # Legacy flat key
            rec = (pool.get("sims") or {}).get(label) or {}
            if rec.get("prep_status") == "pending":
                return label
            continue
        # Prep is shared — any pending slot means prep not finished for the label
        if any(r.get("prep_status") == "pending" for _, r in slots):
            return label
    return None


def hpc_pool_prep_pending(state: Dict[str, Any]) -> bool:
    """True when at least one simulation still needs preprocess/simsetup."""
    pool = state.get("hpc_pool") or {}
    if not pool:
        return False
    return _next_prep_label(pool, state.get("sim_prompts") or []) is not None


def mark_prep_done(state: Dict[str, Any], sim_label: str) -> None:
    pool = state.get("hpc_pool") or {}
    slots = _label_pool_recs(pool, sim_label)
    if not slots:
        rec = (pool.get("sims") or {}).get(sim_label)
        if rec:
            parent = rec.get("parent_label") or _slot_parent_label(sim_label, rec)
            slots = _label_pool_recs(pool, parent) or [(sim_label, rec)]
    if not slots:
        return
    wd = slots[0][1].get("working_dir") or ""
    # Require real simsetup artifacts. Existing trajectories (reuse-hpc) must
    # not short-circuit prep — otherwise HPC staging never sees topol/mdp.
    if not _sim_setup_ready(wd):
        logger.warning(
            "HPC pool: refusing to mark prep done for %s — simsetup not ready",
            sim_label,
        )
        for _, rec in slots:
            rec["prep_status"] = "pending"
        state["hpc_pool"] = pool
        return
    for _, rec in slots:
        rec["prep_status"] = "done"
        if (
            rec.get("hpc_status") == "pending"
            and _should_skip_existing_production(state, wd, hpc_dir=rec.get("hpc_dir"))
        ):
            rec["hpc_status"] = "skipped"
            rec["skip_reason"] = (
                "reuse-hpc" if reuse_hpc_enabled(state) else "production trajectory on disk"
            )
            rec.pop("job_id", None)
        elif (
            rec.get("hpc_status") == "pending"
            and reuse_hpc_enabled(state)
            and _rec_prod_ready(rec, state)
        ):
            # Leave pending so _submit_ready_sims can stage (copy + SLURM, no sbatch).
            pass
    state["hpc_pool"] = pool


def mark_prep_failed(
    state: Dict[str, Any],
    sim_label: str,
    *,
    error: Optional[str] = None,
) -> None:
    """Record a terminal prep failure so the HPC pool does not wait forever."""
    pool = state.get("hpc_pool") or {}
    slots = _label_pool_recs(pool, sim_label)
    if not slots:
        rec = (pool.get("sims") or {}).get(sim_label)
        if rec:
            parent = rec.get("parent_label") or _slot_parent_label(sim_label, rec)
            slots = _label_pool_recs(pool, parent) or [(sim_label, rec)]
    if not slots:
        return
    for _, rec in slots:
        rec["prep_status"] = "failed"
        if error:
            rec["error"] = error
        # Under --reuse-hpc, keep HPC pending so staging can still run when traj exists.
        if reuse_hpc_enabled(state) and _rec_prod_ready(rec, state):
            if rec.get("hpc_status") not in ("submitted", "running", "completed", "skipped"):
                rec["hpc_status"] = "pending"
        elif rec.get("hpc_status") == "pending":
            rec["hpc_status"] = "failed"
            rec.setdefault("error", error or "prep failed; HPC not staged")
    state["hpc_pool"] = pool
    logger.warning(
        "HPC pool: marked prep failed for %s (%s)",
        sim_label,
        (error or "")[:120],
    )


def abandon_orphan_prep_pending(state: Dict[str, Any]) -> List[str]:
    """Fail-soft: convert lingering prep=pending into failed after prep phase ends.

    Prevents a single unfinished prep sim from blocking ``hpc_pool_wait`` forever
    (especially under ``--reuse-hpc`` where there are no SLURM jobs to poll).
    """
    if not state.get("hpc_pool_prep_phase_done"):
        return []
    try:
        from agentic.multi_sim_parallel_pool import _parallel_prep_still_running

        if _parallel_prep_still_running(state):
            return []
    except Exception:
        pass
    pool = state.get("hpc_pool") or {}
    abandoned: List[str] = []
    seen_parents: set = set()
    for key, rec in list((pool.get("sims") or {}).items()):
        if (rec.get("prep_status") or "") != "pending":
            continue
        parent = rec.get("parent_label") or _slot_parent_label(key, rec)
        if parent in seen_parents:
            continue
        seen_parents.add(parent)
        mark_prep_failed(
            state,
            parent,
            error="prep unfinished after parallel prep phase — continuing campaign",
        )
        abandoned.append(parent)
    return abandoned


def pool_hpc_phase_complete(state: Dict[str, Any]) -> bool:
    """True when the cross-sim HPC pool has finished (all jobs done or skipped)."""
    if state.get("hpc_pool_phase_complete"):
        return True
    pool = state.get("hpc_pool")
    if not pool:
        return False
    return _all_hpc_done(pool, state=state)


def reconcile_post_hpc_with_pool(state: Dict[str, Any]) -> bool:
    """
    Clear premature post-HPC routing when the HPC pool is not finished.

    Returns True when stale post-HPC state was cleared.
    """
    if not state.get("post_hpc_analysis_only"):
        return False
    # No pool metadata ⇒ not a pool-managed campaign; leave post-HPC alone.
    if not state.get("hpc_pool") and not state.get("hpc_pool_phase_complete"):
        return False
    if pool_hpc_phase_complete(state):
        return False
    logger.warning(
        "HPC pool still active — clearing stale post-HPC analysis state "
        "(per-sim analysis must wait until all pool jobs complete)"
    )
    state.pop("post_hpc_analysis_only", None)
    state.pop("hpc_pool_phase_complete", None)
    state["multi_sim_phase"] = "hpc_pool"
    state["plan_executed"] = False
    state["execution_plan"] = None
    state["input_validated"] = False
    state["current_agent_idx"] = 0
    return True


def start_post_hpc_phase(state: Dict[str, Any]) -> bool:
    """
    Transition to per-sim analysis/reporter after all HPC jobs complete.

    Returns True when post-HPC analysis/reporter should start; False when the
    workflow should finish (no post-HPC agents in the subtask).
    """
    from src.supervisor.unified_enricher import get_agent_execution_order

    # Prep phase temporarily overwrites agent_list to [preprocess, simsetup].
    # Restore the original CLI pipeline before deciding whether analysis runs.
    pipeline = _requested_pipeline_agents(state)
    state["pipeline_agent_list"] = list(pipeline)
    state["agent_list"] = list(pipeline)
    if len(pipeline) > 1:
        state["subtask_type"] = "multi_agent"
    state.pop("hpc_pool_prep_only", None)
    state.pop("hpc_pool_agent_filter", None)

    agents = get_agent_execution_order(state.get("subtask_type") or "full_task", state)
    post_agents = [a for a in ("analysis", "reporter") if a in agents]
    # Defensive: CLI asked for analysis/reporter by name even if order helper missed them.
    cli_norm = {str(a).lower() for a in pipeline}
    for name in ("analysis", "reporter"):
        if name in cli_norm and name not in post_agents:
            post_agents.append(name)
    if not post_agents:
        state["multi_sim_phase"] = "complete"
        state["hpc_pool_phase_complete"] = True
        state["post_hpc_analysis_only"] = False
        state["plan_executed"] = True
        pool = state.get("hpc_pool") or {}
        pool["phase"] = "complete"
        state["hpc_pool"] = pool
        logger.info("HPC pool complete — no analysis/reporter agents requested; finishing")
        return False

    # Ensure post-HPC loop sees multi-sim orchestration state.
    if not state.get("is_multi_simulation"):
        state["is_multi_simulation"] = True
        ensure_hpc_pool_sim_prompts(state)
    if len(state.get("sim_prompts") or []) <= 1:
        state["run_combined_analysis"] = False

    state["multi_sim_phase"] = "executing_sims"
    state["hpc_pool_phase_complete"] = True
    state["post_hpc_analysis_only"] = True
    state["subtask_type"] = "multi_agent"
    state["agent_list"] = list(post_agents)
    state["plan_executed"] = False
    state["execution_plan"] = None
    state["input_validated"] = False
    state["enriched_prompt"] = None
    pool = state.get("hpc_pool") or {}
    pool["phase"] = "complete"
    state["hpc_pool"] = pool
    from agentic.multi_sim_progress import reset_post_hpc_progress
    from agentic.utils.sim_health import sync_hpc_failures_into_progress

    reset_post_hpc_progress(state)
    marked = sync_hpc_failures_into_progress(state)
    if marked:
        logger.warning(
            "HPC pool: %d simulation(s) failed — analysis/reporter skipped for: %s",
            len(marked),
            ", ".join(marked[:12]) + ("..." if len(marked) > 12 else ""),
        )
    logger.info(
        "HPC pool complete — starting post-HPC analysis/reporter loop (%s)",
        ",".join(post_agents),
    )
    return True


def hpc_pool_supervisor_tick(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    One supervisor iteration for ``multi_sim_phase == 'hpc_pool'``.

    Sets ``state['next_node']`` to supervisor, preprocess, setup, hpc_pool_wait,
    or human_hpc_pool_check.
    """
    pool = init_hpc_pool(state)

    if pool.get("awaiting_hitl"):
        state["next_node"] = "human_hpc_pool_check"
        state["hpc_pool_hitl_message"] = pool.get("hitl_reason") or "HPC pool needs attention"
        return state

    _sync_jobs_from_slurm(state, pool, force=False)
    pool = state["hpc_pool"]

    # Fail-soft: previously any failed HPC job forced HITL and stalled the pool.
    # Keep failed labels terminal and continue submitting/monitoring the rest.
    failed_hpc = [
        label
        for label, rec in (pool.get("sims") or {}).items()
        if rec.get("hpc_status") == "failed"
    ]
    if failed_hpc:
        logger.warning(
            "HPC pool: %d failed job(s) recorded (continuing others): %s",
            len(failed_hpc),
            ", ".join(failed_hpc[:8]) + ("..." if len(failed_hpc) > 8 else ""),
        )

    # After prep phase ends, do not leave prep=pending orphans that block wait forever.
    abandoned = abandon_orphan_prep_pending(state)
    if abandoned:
        logger.warning(
            "HPC pool: abandoned orphan prep pending (%d): %s",
            len(abandoned),
            ", ".join(abandoned[:8]) + ("..." if len(abandoned) > 8 else ""),
        )
        pool = state["hpc_pool"]

    # Overlap phases: stage/submit any prep-ready sims even while other prep
    # workers are still running (generic pipeline parallelism).
    _submit_ready_sims(state, pool)
    pool = state["hpc_pool"]

    # Continue parallel prep for remaining sims (only before prep phase is done).
    prep_label = _next_prep_label(pool, state.get("sim_prompts") or [])
    if prep_label is not None and not state.get("hpc_pool_prep_phase_done"):
        from agentic.parallel_resources import should_use_parallel_pool
        from agentic.multi_sim_parallel_pool import (
            init_parallel_pool,
            parallel_pool_supervisor_tick,
            start_parallel_prep_if_enabled,
            _parallel_prep_still_running,
        )

        if _parallel_prep_still_running(state):
            if state.get("multi_sim_phase") != "parallel_pool":
                state["multi_sim_phase"] = "parallel_pool"
                if not state.get("parallel_pool"):
                    init_parallel_pool(state, phase="prep")
            return parallel_pool_supervisor_tick(state)

        # Restart parallel prep whenever work remains (do not gate on a stale
        # hpc_pool_prep_parallel flag — that blocked --resume after a partial run).
        if should_use_parallel_pool(state):
            if start_parallel_prep_if_enabled(state):
                return state
        state["hpc_pool_needs_prep_start"] = prep_label
        state["next_node"] = "supervisor"
        return state

    # Prep phase done but a pending label somehow remains — abandon then continue.
    if prep_label is not None and state.get("hpc_pool_prep_phase_done"):
        mark_prep_failed(
            state,
            prep_label,
            error="prep still pending after prep phase done — continuing campaign",
        )
        pool = state["hpc_pool"]
        _submit_ready_sims(state, pool)
        pool = state["hpc_pool"]

    # All prep terminal — keep submitting/monitoring until HPC phase complete.
    if _all_hpc_done(pool, state=state):
        if start_post_hpc_phase(state):
            log_pool_to_base(
                state, "All HPC pool jobs complete — starting post-HPC analysis", pool=pool
            )
            state["hpc_pool_post_hpc_start"] = True
            state["next_node"] = "supervisor"
        else:
            log_pool_to_base(
                state, "All HPC pool jobs complete — finishing without analysis", pool=pool
            )
            state["next_node"] = "final_report"
        return state

    running = _count_running(pool)
    pending_submit = any(
        (r.get("prep_status") in ("done", "skipped", "failed") and r.get("hpc_status") == "pending")
        for r in (pool.get("sims") or {}).values()
    )
    if running > 0 or pending_submit:
        # Under reuse-hpc with only pending staging left, stage immediately above;
        # if still pending, brief wait — never a multi-hour SLURM poll.
        state["next_node"] = "hpc_pool_wait"
        state["hpc_pool_status_summary"] = pool_summary(state)
        state["hpc_pool_status_snapshot"] = snapshot_hpc_pool_status(state)
        persist_hpc_pool_checkpoint(state)
        log_pool_to_base(
            state,
            "Pool waiting for SLURM jobs / reuse-hpc staging",
            pool=pool,
            log_to_conversation=False,
            extra={"next_check_sec": pool.get("check_interval_sec")},
        )
        return state

    state["next_node"] = "hpc_pool_wait"
    return state


def _hpc_pool_sim_order(state: Dict[str, Any], pool: Dict[str, Any]) -> List[str]:
    sims = pool.get("sims") or {}
    order: List[str] = []
    for idx, sp in enumerate(state.get("sim_prompts") or []):
        label = sp.get("label") or f"sim_{idx}"
        if label in sims:
            order.append(label)
    for label in sims:
        if label not in order:
            order.append(label)
    return order


def _format_hpc_pool_sim_line(label: str, rec: Dict[str, Any]) -> str:
    parts = [
        f"prep={rec.get('prep_status')}",
        f"hpc={rec.get('hpc_status')}",
    ]
    if rec.get("job_id"):
        parts.append(f"job_id={rec.get('job_id')}")
    if rec.get("last_slurm_state"):
        parts.append(f"slurm={rec.get('last_slurm_state')}")
    if rec.get("skip_reason"):
        parts.append(f"note={rec.get('skip_reason')}")
    return f"  {label}: {' '.join(parts)}"


def snapshot_hpc_pool_status(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compact HPC pool summary for ``pool_status.json`` and state checkpoints."""
    pool = state.get("hpc_pool") or {}
    sims = pool.get("sims") or {}
    active = _count_running(pool)
    max_c = pool.get("max_concurrent")

    by_hpc: Dict[str, List[str]] = {
        "submitted": [],
        "running": [],
        "pending": [],
        "completed": [],
        "skipped": [],
        "failed": [],
    }
    sim_details: Dict[str, Any] = {}
    summary_lines: List[str] = []

    for label in _hpc_pool_sim_order(state, pool):
        rec = sims.get(label) or {}
        entry: Dict[str, Any] = {
            "prep": rec.get("prep_status"),
            "hpc": rec.get("hpc_status"),
            "health": rec.get("health")
            or ("failed" if rec.get("hpc_status") == "failed" else None),
        }
        if rec.get("job_id"):
            entry["job_id"] = rec.get("job_id")
        if rec.get("last_slurm_state"):
            entry["slurm"] = rec.get("last_slurm_state")
        if rec.get("skip_reason"):
            entry["note"] = rec.get("skip_reason")
        if rec.get("error"):
            entry["error"] = rec.get("error")
        sim_details[label] = entry
        summary_lines.append(_format_hpc_pool_sim_line(label, rec))

        hpc_st = rec.get("hpc_status") or "pending"
        bucket = hpc_st if hpc_st in by_hpc else "pending"
        by_hpc[bucket].append(label)

    parallel_pool = state.get("parallel_pool") or {}
    if (
        parallel_pool.get("phase") == "prep"
        and (
            state.get("hpc_pool_prep_parallel")
            or state.get("multi_sim_phase") == "parallel_pool"
        )
    ):
        for label, prec in (parallel_pool.get("sims") or {}).items():
            if label in sim_details:
                sim_details[label]["worker"] = prec.get("status")

    healthy = [
        lab
        for lab, e in sim_details.items()
        if e.get("hpc") in ("completed", "skipped") and e.get("health") != "failed"
    ]
    for lab in healthy:
        if sim_details[lab].get("health") is None:
            sim_details[lab]["health"] = "healthy"

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_phase": state.get("multi_sim_phase") or "hpc_pool",
        "pool_type": "hpc",
        "phase": pool.get("phase") or "active",
        "max_concurrent": max_c,
        "check_interval_sec": pool.get("check_interval_sec"),
        "active_hpc_jobs": active,
        "simulations": sim_details,
        "summary_lines": summary_lines,
        "running": sorted(by_hpc["submitted"] + by_hpc["running"]),
        "pending": sorted(by_hpc["pending"]),
        "done": sorted(by_hpc["completed"] + by_hpc["skipped"]),
        "failed": sorted(by_hpc["failed"]),
        "healthy": sorted(healthy),
        "pending_count": len(by_hpc["pending"]),
        "done_count": len(by_hpc["completed"]) + len(by_hpc["skipped"]),
        "failed_count": len(by_hpc["failed"]),
        "healthy_count": len(healthy),
    }


def persist_hpc_pool_checkpoint(state: Dict[str, Any]) -> None:
    """Write HPC pool status to state.jsonl quietly (no conversation log spam)."""
    if state.get("hpc_pool"):
        state["hpc_pool_status_snapshot"] = snapshot_hpc_pool_status(state)
    try:
        from agentic.utils.state_persistence import save_workflow_state_quiet

        save_workflow_state_quiet(state)
    except Exception as exc:
        logger.debug("HPC pool checkpoint save failed: %s", exc)


def pool_summary(state: Dict[str, Any]) -> str:
    pool = state.get("hpc_pool") or {}
    active = _count_running(pool)
    max_c = pool.get("max_concurrent", "?")
    lines = [
        f"HPC pool (active {active}/{max_c} concurrent, "
        f"check every {pool.get('check_interval_sec', '?')}s):"
    ]
    for label in _hpc_pool_sim_order(state, pool):
        rec = (pool.get("sims") or {}).get(label) or {}
        lines.append(_format_hpc_pool_sim_line(label, rec).lstrip())
    return "\n".join(lines)


def reopen_failed_hpc_for_resume(state: Dict[str, Any]) -> List[str]:
    """Re-open failed / unhealthy sims for HPC resubmit under ``--resume``.

    After a campaign marks ``hpc_pool_phase_complete``, a later ``--resume``
    (optionally with ``--retry-labels``) must clear that sticky flag and put
    only the dead systems back to ``hpc_status=pending``. Healthy production
    trajectories are left alone.

    Returns labels re-queued for HPC.
    """
    if not state.get("resume_failed_only") and not state.get("retry_labels"):
        return []

    # Accept planner names (hpc) and CLI aliases (hpcjob). Also allow plain
    # ``--resume`` when a prior run left ``hpc_pool`` / ``sim_prompts`` even if
    # saved ``agent_list`` was narrowed to analysis/reporter after HPC finished.
    pipeline_list = _requested_pipeline_agents(state)
    agents = {str(a).lower() for a in (state.get("agent_list") or [])}
    pipeline_set = {str(a).lower() for a in pipeline_list}
    wants_hpc = bool((agents | pipeline_set) & {"hpc", "hpcjob"})
    if not wants_hpc and (state.get("hpc_pool") or state.get("sim_prompts")):
        wants_hpc = True
    if not wants_hpc:
        return []

    retry = {str(x) for x in (state.get("retry_labels") or []) if x}
    pool = state.get("hpc_pool") or {}
    sims = pool.get("sims") or {}
    if not sims and not state.get("sim_prompts"):
        return []

    # Ensure pool records exist for planned labels.
    ensure_hpc_pool_sim_prompts(state)
    if not sims:
        pool = init_hpc_pool(state)
        sims = pool.get("sims") or {}

    requeued: List[str] = []
    for sp in state.get("sim_prompts") or []:
        label = sp.get("label")
        if not label:
            continue
        wd = sp.get("working_dir") or str(
            Path(state.get("multi_sim_base_dir") or state.get("working_directory") or ".")
            / label
        )
        rec = sims.get(label) or {
            "working_dir": wd,
            "prep_status": "pending",
            "hpc_status": "pending",
        }
        rec.setdefault("working_dir", wd)

        # Explicit retry list always reopens; otherwise reopen failed / unhealthy only.
        force = label in retry
        failed = str(rec.get("hpc_status") or "").lower() == "failed"
        unhealthy = not _sim_production_trajectory_ready(wd, state=state)
        if not force and not failed and not unhealthy:
            continue

        # Prep: keep done only when simsetup/system.gro exists (see _sim_setup_ready).
        if _sim_setup_ready(wd):
            rec["prep_status"] = "done"
        else:
            rec["prep_status"] = "pending"
        rec["hpc_status"] = "pending"
        rec.pop("error", None)
        rec.pop("job_id", None)
        rec.pop("last_slurm_state", None)
        rec.pop("skip_reason", None)
        sims[label] = rec
        requeued.append(label)

    if not requeued:
        return []

    pool["sims"] = sims
    pool["phase"] = "active"
    pool["awaiting_hitl"] = False
    pool.pop("hitl_reason", None)
    state["hpc_pool"] = pool
    state.pop("hpc_pool_phase_complete", None)
    state["post_hpc_analysis_only"] = False
    state["hpc_pool_disabled"] = False
    state["use_hpc_pool"] = True
    state["multi_sim_phase"] = "hpc_pool"
    state["plan_executed"] = False
    state["execution_plan"] = None
    state["input_validated"] = False
    # Ensure the active pipeline can run prep + HPC (+ analysis/reporter).
    # ``--resume`` without ``--subtask`` must not stay stuck on analysis-only.
    default_pipeline = [
        "preprocess",
        "simsetup",
        "hpcjob",
        "analysis",
        "reporter",
    ]
    pipeline = list(pipeline_list or default_pipeline)
    plower = {str(a).lower() for a in pipeline}
    if not (plower & {"hpc", "hpcjob"}):
        head = [a for a in default_pipeline[:3] if a not in plower]
        pipeline = head + pipeline
    state["pipeline_agent_list"] = list(pipeline)
    state["agent_list"] = list(pipeline)
    if len(pipeline) > 1:
        state["subtask_type"] = "multi_agent"
    logger.warning(
        "[resume] Re-opening HPC pool for %d failed/unhealthy sim(s): %s",
        len(requeued),
        ", ".join(requeued),
    )
    return requeued


def resume_hpc_pool_if_needed(state: Dict[str, Any]) -> bool:
    """On ``--resume``, re-enter hpc_pool phase if jobs still active."""
    if state.get("multi_sim_phase") != "hpc_pool" and not state.get("hpc_pool"):
        return False
    # Failed-only resume after a "complete" pool: reopen dead sims first.
    if state.get("resume_failed_only") or state.get("retry_labels"):
        reopened = reopen_failed_hpc_for_resume(state)
        if reopened:
            pool = init_hpc_pool(state)
            state["multi_sim_phase"] = "hpc_pool"
            _sync_jobs_from_slurm(state, pool, force=True)
            return True
    pool = init_hpc_pool(state)
    if pool.get("phase") == "complete" or state.get("hpc_pool_phase_complete"):
        return False
    if _all_hpc_done(pool, state=state):
        start_post_hpc_phase(state)
        return True
    state["multi_sim_phase"] = "hpc_pool"
    _sync_jobs_from_slurm(state, pool, force=True)
    return True


def clear_hitl_pause(state: Dict[str, Any]) -> None:
    pool = state.get("hpc_pool") or {}
    pool["awaiting_hitl"] = False
    pool.pop("hitl_reason", None)
    for rec in (pool.get("sims") or {}).values():
        if rec.get("hpc_status") == "failed":
            rec["hpc_status"] = "pending"
            rec.pop("error", None)
    state["hpc_pool"] = pool
