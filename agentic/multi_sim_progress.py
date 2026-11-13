"""
Multi-simulation workflow progress — authoritative base-level routing state.

Stored in ``state["multi_sim_progress"]`` and persisted to
``{base}/supervisor/state.jsonl``.  Per-simulation detail lives in
``{sim}/supervisor/state.jsonl``; the base record says *which* sim and
*which* subtask to run next.

HITL ``switch {sim}`` only changes the chat view; ``continue`` always
follows ``multi_sim_progress``, not the HITL-bound simulation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CHECKPOINT_TO_AGENT = {
    "preprocess": "preprocessing",
    "preprocessing": "preprocessing",
    "setup": "simsetup",
    "hpc": "hpc",
    "analysis": "analysis",
    "reporter": "reporter",
}

_AGENT_TO_NODE = {
    "preprocessing": "preprocess",
    "simsetup": "setup",
    "hpc": "hpc",
    "analysis": "analysis",
    "reporter": "reporter",
}


def _required_agents(state: Dict[str, Any]) -> List[str]:
    from src.supervisor.unified_enricher import get_agent_execution_order

    subtask = state.get("subtask_type") or "full_task"
    return get_agent_execution_order(subtask, state)


def init_multi_sim_progress(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build or refresh progress from sim_prompts (call when master plan is ready)."""
    sim_prompts = state.get("sim_prompts") or []
    agents = _required_agents(state)
    if not sim_prompts:
        return state.get("multi_sim_progress") or {}

    sim_order: List[str] = []
    sims: Dict[str, Any] = {}
    for idx, sp in enumerate(sim_prompts):
        label = sp.get("label") or f"sim_{idx}"
        sim_order.append(label)
        sims[label] = {
            "index": idx,
            "working_dir": sp.get("working_dir") or "",
            "agents": {a: "pending" for a in agents},
            "status": "pending",
            "pipeline": {"input_validated": False, "enriched": False, "planned": False},
            "rep_num": int(state.get("rep_num") or 1),
            "reps": {},
        }
        try:
            from src.analysis.replicate_paths import build_rep_plan, normalize_rep_num

            rn = normalize_rep_num(state.get("rep_num", 1))
            if rn > 1:
                for slot in build_rep_plan(
                    label,
                    sims[label]["working_dir"],
                    rn,
                    base_seed=int(state.get("replicate_base_seed") or 12345),
                ):
                    sims[label]["reps"][slot["rep_id"]] = {
                        "hpc_status": "pending",
                        "analysis_status": "pending",
                        "job_id": None,
                        "seed": slot["seed"],
                    }
        except Exception:
            pass

    existing = state.get("multi_sim_progress") or {}
    # Preserve completed agents when re-init (e.g. replan)
    if existing.get("sim_order") == sim_order:
        for label, rec in existing.get("sims", {}).items():
            if label in sims and isinstance(rec, dict):
                sims[label]["agents"].update(rec.get("agents") or {})
                sims[label]["status"] = rec.get("status", sims[label]["status"])
                if rec.get("pipeline"):
                    sims[label]["pipeline"] = {
                        **sims[label]["pipeline"],
                        **rec["pipeline"],
                    }
                if rec.get("reps"):
                    sims[label]["reps"] = {**sims[label].get("reps", {}), **rec["reps"]}

    active_label = existing.get("active_sim_label") or sim_order[0]
    if active_label not in sims:
        active_label = sim_order[0]

    progress = {
        "version": 1,
        "sim_order": sim_order,
        "required_agents": agents,
        "phase": state.get("multi_sim_phase") or "executing_sims",
        "active_sim_label": active_label,
        "active_agent": existing.get("active_agent") or (agents[0] if agents else None),
        "sims": sims,
        "hitl_paused_at": existing.get("hitl_paused_at"),
    }
    if state.get("run_combined_analysis") or state.get("run_post_combined"):
        progress["combined"] = existing.get("combined") or {
            "analysis": "pending",
            "reporter": "pending",
        }
    if state.get("run_pre_combined"):
        progress["pre_combined"] = existing.get("pre_combined") or {
            "analysis": "pending",
        }
    state["multi_sim_progress"] = progress
    _apply_progress_indices(state, rebind=False)
    return progress


def _ensure_pipeline(rec: Dict[str, Any]) -> Dict[str, Any]:
    pipe = rec.setdefault(
        "pipeline",
        {"input_validated": False, "enriched": False, "planned": False},
    )
    return pipe


def mark_sim_pipeline_stage(
    state: Dict[str, Any],
    sim_label: str,
    *,
    input_validated: Optional[bool] = None,
    enriched: Optional[bool] = None,
    planned: Optional[bool] = None,
) -> None:
    """Persist per-sim pipeline milestones in base ``multi_sim_progress`` (resume source of truth)."""
    progress = ensure_multi_sim_progress(state)
    if not progress:
        return
    rec = _sim_record(progress, sim_label)
    if not rec:
        return
    pipe = _ensure_pipeline(rec)
    if input_validated:
        pipe["input_validated"] = True
    if enriched:
        pipe["enriched"] = True
    if planned:
        pipe["planned"] = True
    state["multi_sim_progress"] = progress
    apply_sim_pipeline_to_state(state)


def _merge_pipeline_from_per_sim(rec: Dict[str, Any], per_sim: Optional[Dict[str, Any]]) -> None:
    """Refresh pipeline flags from a per-simulation ``state.jsonl`` snapshot."""
    if not per_sim:
        return
    pipe = _ensure_pipeline(rec)
    if per_sim.get("input_validated"):
        pipe["input_validated"] = True
    if per_sim.get("enriched_prompt") or per_sim.get("rephrased_goal"):
        pipe["enriched"] = True
    if per_sim.get("execution_plan"):
        pipe["planned"] = True


def apply_sim_pipeline_to_state(state: Dict[str, Any]) -> None:
    """Restore ``input_validated`` / enrichment / plan from progress + per-sim checkpoint."""
    progress = state.get("multi_sim_progress") or {}
    label = progress.get("active_sim_label")
    if not label:
        return
    rec = (progress.get("sims") or {}).get(label) or {}
    pipe = rec.get("pipeline") or {}
    wd = rec.get("working_dir") or ""
    per_sim = _load_per_sim_checkpoint(wd) if wd else None
    if per_sim:
        _merge_pipeline_from_per_sim(rec, per_sim)
        pipe = rec.get("pipeline") or pipe

    if pipe.get("input_validated"):
        state["input_validated"] = True
    if pipe.get("enriched"):
        if per_sim and per_sim.get("enriched_prompt"):
            state["enriched_prompt"] = per_sim["enriched_prompt"]
            state["rephrased_goal"] = per_sim.get("rephrased_goal") or per_sim["enriched_prompt"]
        else:
            sim_prompts = state.get("sim_prompts") or []
            sp = next((x for x in sim_prompts if x.get("label") == label), None)
            if sp and sp.get("enriched_prompt"):
                state["enriched_prompt"] = sp["enriched_prompt"]
                state["rephrased_goal"] = sp["enriched_prompt"]
    if pipe.get("planned"):
        if per_sim and per_sim.get("execution_plan") and not state.get("execution_plan"):
            state["execution_plan"] = per_sim["execution_plan"]


def _apply_progress_indices(state: Dict[str, Any], *, rebind: bool) -> None:
    """Sync sim index / working dir from ``multi_sim_progress`` without losing pipeline state."""
    progress = state.get("multi_sim_progress") or {}
    label = progress.get("active_sim_label")
    agent = progress.get("active_agent")
    if label and rebind:
        bind_workflow_to_sim(state, label)
    elif label:
        rec = _sim_record(progress, label)
        if rec:
            state["current_sim_index"] = rec.get("index", state.get("current_sim_index", 0))
            wd = rec.get("working_dir")
            if wd:
                state["working_directory"] = str(Path(wd).resolve())
    agents = progress.get("required_agents") or _required_agents(state)
    if agent and agent in agents:
        state["current_agent_idx"] = agents.index(agent)
    rec = _sim_record(progress, label) if label else None
    agents = progress.get("required_agents") or _required_agents(state)
    if rec and _sim_agents_complete(rec, agents):
        state["plan_executed"] = True
    elif rec and rec.get("status") != "done":
        if not (state.get("execution_plan") and state.get("input_validated")):
            state["plan_executed"] = False
    elif rec:
        state["plan_executed"] = False
    phase = progress.get("phase")
    # Do not clobber an active parallel_pool / hpc_pool phase from a stale
    # progress.phase (that previously dropped mid-campaign into sequential mode).
    # Never downgrade combined_* (or combined_only) to executing_sims — that
    # re-triggers combined-only activate and loops analysis forever.
    current = state.get("multi_sim_phase")
    if phase and current not in ("hpc_pool", "parallel_pool"):
        if phase == "executing_sims" and (
            current
            in (
                "pre_combined",
                "combined_analysis",
                "post_combined",
                "combined_reporter",
            )
            or state.get("combined_only")
        ):
            if current is None and state.get("combined_only"):
                combined = progress.get("combined") or {}
                if combined.get("reporter") == "done":
                    state["multi_sim_phase"] = "complete"
                elif combined.get("analysis") == "done":
                    state["multi_sim_phase"] = "combined_reporter"
                else:
                    state["multi_sim_phase"] = "combined_analysis"
            # else keep current pre/post combined phase
        elif not state.get("parallel_pool"):
            state["multi_sim_phase"] = phase
    apply_sim_pipeline_to_state(state)
    if label and state.get("multi_sim_phase") == "executing_sims":
        ensure_per_sim_working_directory(state)
        apply_sim_pipeline_to_state(state)


def reset_post_hpc_progress(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reset base-level progress for the sequential post-HPC analysis/reporter loop.

    Unlike ``init_multi_sim_progress``, this does not preserve stale agent status
    or active_sim_label from a prior interrupted run.
    """
    sim_prompts = state.get("sim_prompts") or []
    agents = ["analysis", "reporter"]
    sim_order: List[str] = []
    sims: Dict[str, Any] = {}
    for idx, sp in enumerate(sim_prompts):
        label = sp.get("label") or f"sim_{idx}"
        sim_order.append(label)
        sims[label] = {
            "index": idx,
            "working_dir": sp.get("working_dir") or "",
            "agents": {a: "pending" for a in agents},
            "status": "pending",
            "pipeline": {"input_validated": False, "enriched": False, "planned": False},
        }

    progress: Dict[str, Any] = {
        "version": 1,
        "sim_order": sim_order,
        "required_agents": agents,
        "phase": "executing_sims",
        "active_sim_label": sim_order[0] if sim_order else None,
        "active_agent": "analysis" if agents else None,
        "sims": sims,
        "hitl_paused_at": None,
    }
    if state.get("run_combined_analysis"):
        progress["combined"] = {"analysis": "pending", "reporter": "pending"}

    state["multi_sim_progress"] = progress
    state["current_sim_index"] = 0
    state["completed_sim_states"] = []
    state["current_agent_idx"] = 0
    state["plan_executed"] = False
    if sim_order:
        bind_workflow_to_sim(state, sim_order[0])
    return progress


def first_incomplete_post_hpc_sim(state: Dict[str, Any]) -> Optional[str]:
    """Return the label of the first sim with pending post-HPC agents."""
    progress = state.get("multi_sim_progress") or {}
    agents = progress.get("required_agents") or ["analysis", "reporter"]
    for label in progress.get("sim_order") or []:
        rec = _sim_record(progress, label)
        if not rec:
            continue
        if rec.get("status") in ("failed", "skipped"):
            continue
        agent_map = rec.get("agents") or {}
        if any(agent_map.get(a) not in ("done", "skipped") for a in agents):
            return label
    return None


def all_per_sim_agents_done(state: Dict[str, Any]) -> bool:
    """True when every planned simulation finished all required agents."""
    progress = state.get("multi_sim_progress") or {}
    agents = progress.get("required_agents") or _required_agents(state)
    for label in progress.get("sim_order") or []:
        rec = _sim_record(progress, label)
        if not rec:
            return False
        if rec.get("status") in ("failed", "skipped"):
            continue
        agent_map = rec.get("agents") or {}
        if any(agent_map.get(a) not in ("done", "skipped") for a in agents):
            return False
    return bool(progress.get("sim_order"))


def per_sim_post_hpc_artifacts_ready(working_dir: str) -> bool:
    """True when per-sim analysis metrics and HTML report exist on disk."""
    wd = Path(working_dir)
    if not per_sim_analysis_done_on_disk(str(wd)):
        return False
    reporter_dir = wd / "reporter"
    return reporter_dir.is_dir() and any(reporter_dir.glob("*.html"))


def per_sim_family_modular_done_on_disk(working_dir: str) -> bool:
    """True when modular consensus_* dynamics outputs exist for a sim.

    Collector expects ``consensus_dihedrals/``, ``consensus_rmsf/``, and
    ``consensus_DCCM/`` under ``analysis/`` (or ``analysis/avg/`` /
    ``analysis/repXX/``). ``consensus_PCA/`` is optional (legacy independent
    FEL); shared-ref dyn is a combined-stage artifact.
    """
    analysis_dir = Path(working_dir) / "analysis"
    if not analysis_dir.is_dir():
        return False
    try:
        from agentic.campaign.family_recipe import FAMILY_MODULAR_DIRS

        needed = tuple(FAMILY_MODULAR_DIRS)
    except Exception:
        needed = (
            "consensus_dihedrals",
            "consensus_rmsf",
            "consensus_DCCM",
        )
    # Accept either on-disk name (canonical consensus_PCA or legacy _ref).
    aliases = {
        "consensus_PCA": ("consensus_PCA", "consensus_PCA_ref"),
    }
    search_dirs = [analysis_dir]
    avg = analysis_dir / "avg"
    if avg.is_dir():
        search_dirs.append(avg)
    search_dirs.extend(sorted(analysis_dir.glob("rep*")))

    def _dir_has_output(d: Path, name: str) -> bool:
        for cand in aliases.get(name, (name,)):
            target = d / cand
            if not target.is_dir():
                continue
            # Any json/csv/dat under the modular dir counts as produced output.
            if (
                any(target.glob("*.json"))
                or any(target.glob("*.csv"))
                or any(target.glob("*.dat"))
                or any(target.glob("*.npz"))
            ):
                return True
        return False

    # Prefer avg/ when present: multi-rep fan-out writes modular results there
    # or under each rep. Accept if every needed dir exists in any search root
    # (same root for all four is ideal; allow split across reps).
    for d in search_dirs:
        if all(_dir_has_output(d, name) for name in needed):
            return True
    # Also accept if each module exists somewhere under analysis/
    return all(
        any(_dir_has_output(d, name) for d in search_dirs) for name in needed
    )


def per_sim_paper_ward4_modular_done_on_disk(working_dir: str) -> bool:
    """Deprecated alias for :func:`per_sim_family_modular_done_on_disk`."""
    return per_sim_family_modular_done_on_disk(working_dir)


def _load_campaign_spec_near(working_dir: str):
    """Load CampaignSpec from state is not available; read planner JSON on disk."""
    wd = Path(working_dir)
    candidates = [
        wd / "planner" / "campaign_spec.json",
        wd.parent / "planner" / "campaign_spec.json",
        wd.parent.parent / "planner" / "campaign_spec.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import json as _json

            from agentic.campaign.spec import CampaignSpec

            data = _json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return CampaignSpec.from_dict(data)
        except Exception:
            continue
    return None


def _working_dir_requests_family_modular(working_dir: str) -> bool:
    """Family mode from on-disk campaign_spec, then nearby goal.txt.

    Resume must not flip family gates off when the compiled spec says
    family_modular even if goal.txt is missing or truncated.
    """
    spec = _load_campaign_spec_near(working_dir)
    if spec is not None:
        if spec.family_modular or spec.mode == "family_modular":
            return True
        # Explicit non-family spec wins over goal.txt regex.
        if spec.mode in {"generic", "single_md", "multi_protocol", "apo_holo"}:
            return False
    try:
        from agentic.planner.planning_guidelines import (
            detect_family_modular_dynamics_requested,
        )
    except Exception:
        return False
    wd = Path(working_dir)
    candidates = [
        wd.parent / "goal.txt",
        wd.parent.parent / "goal.txt",
        wd.parent.parent.parent / "goal.txt",
    ]
    for g in candidates:
        if not g.is_file():
            continue
        try:
            if detect_family_modular_dynamics_requested(g.read_text(errors="ignore")):
                return True
        except Exception:
            continue
    return False


def _working_dir_requests_paper_ward4(working_dir: str) -> bool:
    """Deprecated alias for :func:`_working_dir_requests_family_modular`."""
    return _working_dir_requests_family_modular(working_dir)


def per_sim_analysis_done_on_disk(
    working_dir: str,
    state: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when per-sim analysis produced real metric artifacts (not stub-only).

    ``analysis_summary.jsonl`` alone is insufficient: a failed/retry loop can
    write PDB validation entries while RMSD/RMSF never succeed (e.g. wrong
    topology such as ATP.gro). Require at least one common science output.

    A lone ``ligand_pocket_distance.*`` file is also insufficient: sequential
    retry loops often inject only that mandatory metric and then mark analysis
    "done", skipping PCA/FEL/contacts/residence and blocking combined analysis.

    Multi-rep campaigns may store metrics under ``analysis/avg/`` or ``analysis/rep01/``.

    When the enclosing campaign ``goal.txt`` requests family modular dynamics,
    also require the four modular ``consensus_*`` output directories — contacts
    or RMSD alone must not short-circuit that analysis.

    When the goal explicitly names metrics (RMSD/RMSF/Rg/…), every requested
    data file must be present — a lone RMSD no longer completes a 3-metric goal.
    """
    analysis_dir = Path(working_dir) / "analysis"
    summary_ok = (analysis_dir / "analysis_summary.jsonl").is_file()
    if not summary_ok:
        # Also accept per-rep summaries
        for sub in analysis_dir.glob("rep*/analysis_summary.jsonl") if analysis_dir.is_dir() else []:
            summary_ok = True
            break
        if (analysis_dir / "avg" / "summary_scalars.json").is_file():
            summary_ok = True
    if not summary_ok:
        return False
    # Family campaigns execute a compiled recipe (consensus_* only). RMSD /
    # contacts must not be required, or resume would re-run every system.
    if _working_dir_requests_family_modular(working_dir):
        return per_sim_family_modular_done_on_disk(working_dir)
    strong_markers = (
        "rmsd.dat",
        "rmsd.png",
        "rmsf.dat",
        "gyration.dat",
        "pca_projections.dat",
        "fel_basins.png",
        "fel_features.json",
        "pca_pc1_pc2_time.png",
        "protein_ligand_contacts.csv",
        "ligand_residence.csv",
        "pocket_sasa.csv",
        "rmsd_mean_std.dat",
        "summary_scalars.json",
    )
    search_dirs = [analysis_dir]
    if analysis_dir.is_dir():
        avg = analysis_dir / "avg"
        if avg.is_dir():
            search_dirs.insert(0, avg)
        search_dirs.extend(sorted(analysis_dir.glob("rep*")))
    base_ok = False
    for d in search_dirs:
        if any((d / name).is_file() for name in strong_markers):
            base_ok = True
            break
        for pattern in ("rmsd*", "rmsf*", "gyration*", "pca_*", "fel_*", "*contacts*", "*residence*"):
            if any(d.glob(pattern)):
                base_ok = True
                break
        if base_ok:
            break
    # Pocket COM alone is a mandatory inject, not a complete analysis.
    if not base_ok:
        return False
    # Intent-aware completeness when goal names specific metrics
    try:
        from agentic.campaign.contracts import required_metric_files_missing

        if required_metric_files_missing(working_dir, state=state):
            return False
    except Exception:
        pass
    return True


def per_sim_reporter_done_on_disk(working_dir: str) -> bool:
    """True when per-sim reporter HTML exists *and* analysis summary is present."""
    wd = Path(working_dir)
    if not (wd / "analysis" / "analysis_summary.jsonl").is_file():
        return False
    reporter_dir = wd / "reporter"
    return reporter_dir.is_dir() and any(reporter_dir.glob("*.html"))


def merge_completed_states_from_progress(state: Dict[str, Any]) -> None:
    """
    Ensure ``completed_sim_states`` includes every simulation marked done in
    ``multi_sim_progress`` (including disk-only completions from prior runs).
    """
    progress = state.get("multi_sim_progress") or {}
    if not progress:
        return
    agents = progress.get("required_agents") or _required_agents(state)
    hpc_only = list(agents) == ["hpc"]
    completed = list(state.get("completed_sim_states") or [])
    by_label = {
        str(s.get("label")): s for s in completed if s.get("label")
    }
    sim_prompts = state.get("sim_prompts") or []
    for label in progress.get("sim_order") or []:
        rec = _sim_record(progress, label) or {}
        agent_map = rec.get("agents") or {}
        wd = rec.get("working_dir") or ""
        job_info = load_local_continuation_job(wd) if wd else None
        hpc_submitted = bool(job_info and job_info.get("job_id"))
        hpc_complete = bool(job_info and job_info.get("complete"))

        agents_done = rec.get("status") == "done" or (
            bool(agents) and all(agent_map.get(a) == "done" for a in agents)
        )
        if hpc_only:
            if not (hpc_submitted or hpc_complete or agents_done):
                continue
        else:
            if rec.get("status") != "done" and any(
                agent_map.get(a) != "done" for a in agents
            ):
                continue
            if wd and not per_sim_post_hpc_artifacts_ready(wd) and not hpc_submitted:
                continue

        idx = rec.get("index", 0)
        for i, sp in enumerate(sim_prompts):
            if sp.get("label") == label:
                idx = i
                break
        tpr = Path(wd) / "hpc" / "md.tpr" if wd else None
        xtc = Path(wd) / "hpc" / "mdWrap.xtc" if wd else None
        if xtc is not None and not xtc.is_file():
            xtc = Path(wd) / "hpc" / "md.xtc"
        job_id = (job_info or {}).get("job_id")
        job_status = (job_info or {}).get("status")
        success = bool(hpc_complete or hpc_submitted or agents_done)
        # Prefer disk truth over a stale failed snapshot.
        existing = by_label.get(label)
        if existing and existing.get("success") is False and success:
            existing = None
        if existing and not job_id:
            continue
        if label in by_label and existing is not None and by_label[label].get("success"):
            # Keep success snapshot but refresh job metadata from disk.
            snap = by_label[label]
            if job_id:
                snap["job_id"] = job_id
                snap["continuation_job_id"] = job_id
                snap["job_status"] = job_status
                snap["success"] = True
            continue
        by_label[label] = {
            "sim_index": idx,
            "label": label,
            "working_directory": wd,
            "success": success,
            "skipped": False,
            "job_id": job_id,
            "continuation_job_id": job_id,
            "job_status": job_status,
            "continuation_manifest": (job_info or {}).get("manifest_path"),
            "topology": str(tpr) if tpr is not None and tpr.is_file() else None,
            "trajectory_path": str(xtc) if xtc is not None and xtc.is_file() else None,
            "errors": [],
            "warnings": [],
            "_source": "disk_progress",
        }
    state["completed_sim_states"] = sorted(
        by_label.values(),
        key=lambda s: int(s.get("sim_index") or 0),
    )


def sync_parallel_pool_to_multi_sim_progress(state: Dict[str, Any]) -> None:
    """
    Mirror ``parallel_pool`` worker statuses into ``multi_sim_progress``.

    Keeps the authoritative routing record consistent with the live pool so
    ``state.jsonl`` and ``--resume`` reflect running / done sims correctly.
    """
    pool = state.get("parallel_pool")
    if not pool:
        return
    progress = ensure_multi_sim_progress(state)
    if not progress:
        return

    phase = pool.get("phase") or "analysis"
    progress["phase"] = "parallel_pool"
    agents_req = progress.get("required_agents") or _required_agents(state)
    first_running: Optional[str] = None
    first_running_agent: Optional[str] = None

    for label in progress.get("sim_order") or []:
        pool_rec = (pool.get("sims") or {}).get(label)
        sim_rec = _sim_record(progress, label)
        if not pool_rec or not sim_rec:
            continue

        pool_status = pool_rec.get("status") or "pending"
        wd = pool_rec.get("working_dir") or sim_rec.get("working_dir") or ""
        if wd:
            sim_rec["working_dir"] = wd
        agent_map = sim_rec.setdefault("agents", {})

        if phase == "prep":
            prep_agents = ("preprocessing", "simsetup")
            if pool_status in ("done", "skipped"):
                for a in prep_agents:
                    if a in agent_map or a in agents_req:
                        agent_map[a] = "done"
                sim_rec["status"] = "done"
            elif pool_status == "failed":
                sim_rec["status"] = "failed"
                sim_rec["health"] = "failed"
                sim_rec["error"] = pool_rec.get("error")
                for a in prep_agents:
                    if a in agent_map or a in agents_req:
                        agent_map[a] = "failed"
            elif pool_status == "running":
                sim_rec["status"] = "in_progress"
                for a in prep_agents:
                    if a in agent_map or a in agents_req:
                        agent_map[a] = "in_progress"
                if first_running is None:
                    first_running = label
                    first_running_agent = prep_agents[0]
            continue

        analysis_done = per_sim_analysis_done_on_disk(wd) if wd else False
        reporter_done = per_sim_reporter_done_on_disk(wd) if wd else False

        if pool_status in ("done", "skipped"):
            # Trust disk for analysis/reporter — pool may inherit false "done"
            # from a prior phase when artifacts are absent.
            if analysis_done:
                if "analysis" in agent_map or "analysis" in agents_req:
                    agent_map["analysis"] = "done"
            elif "analysis" in agent_map or "analysis" in agents_req:
                agent_map["analysis"] = "pending"
            if reporter_done:
                if "reporter" in agent_map or "reporter" in agents_req:
                    agent_map["reporter"] = "done"
            elif "reporter" in agent_map or "reporter" in agents_req:
                agent_map["reporter"] = "pending"
            need_reporter = "reporter" in agents_req or "reporter" in agent_map
            if analysis_done and (reporter_done or not need_reporter):
                sim_rec["status"] = "done"
                sim_rec.pop("error", None)
            else:
                sim_rec["status"] = "pending"
        elif pool_status == "failed":
            sim_rec["status"] = "failed"
            sim_rec["health"] = "failed"
            sim_rec["error"] = pool_rec.get("error")
            for key in ("analysis", "reporter"):
                if agent_map.get(key) not in ("done",):
                    agent_map[key] = "skipped"
        elif pool_status == "running":
            sim_rec["status"] = "in_progress"
            if reporter_done:
                agent_map["analysis"] = "done"
                agent_map["reporter"] = "done"
                sim_rec["status"] = "done"
            elif analysis_done:
                agent_map["analysis"] = "done"
                agent_map["reporter"] = "in_progress"
                if first_running is None:
                    first_running = label
                    first_running_agent = "reporter"
            else:
                agent_map["analysis"] = "in_progress"
                if "reporter" in agent_map or "reporter" in agents_req:
                    agent_map.setdefault("reporter", "pending")
                if first_running is None:
                    first_running = label
                    first_running_agent = "analysis"
        elif pool_status == "pending":
            if analysis_done and reporter_done:
                agent_map["analysis"] = "done"
                agent_map["reporter"] = "done"
                sim_rec["status"] = "done"
            elif analysis_done:
                agent_map["analysis"] = "done"
                agent_map["reporter"] = "pending"
                sim_rec["status"] = "pending"
            elif sim_rec.get("status") not in ("done", "in_progress"):
                sim_rec["status"] = "pending"

    if first_running:
        progress["active_sim_label"] = first_running
        progress["active_agent"] = first_running_agent
    state["multi_sim_progress"] = progress


def sync_post_hpc_progress_from_disk(state: Dict[str, Any]) -> None:
    """Align multi_sim_progress with on-disk post-HPC artifacts."""
    if not state.get("post_hpc_analysis_only"):
        return
    progress = ensure_multi_sim_progress(state)
    if not progress:
        return
    agents = progress.get("required_agents") or ["analysis", "reporter"]
    if set(agents) != {"analysis", "reporter"}:
        return
    for label in progress.get("sim_order") or []:
        rec = _sim_record(progress, label)
        if not rec:
            continue
        wd = rec.get("working_dir") or ""
        if not wd:
            base = state.get("multi_sim_base_dir") or state.get("working_directory")
            wd = str(Path(base) / label)
        agent_map = rec.setdefault("agents", {})
        analysis_ok = (Path(wd) / "analysis" / "analysis_summary.jsonl").is_file()
        reporter_ok = per_sim_post_hpc_artifacts_ready(wd)
        if analysis_ok:
            agent_map["analysis"] = "done"
        if reporter_ok:
            agent_map["reporter"] = "done"
        if all(agent_map.get(a) == "done" for a in agents):
            rec["status"] = "done"
    state["multi_sim_progress"] = progress


def all_post_hpc_artifacts_on_disk(state: Dict[str, Any]) -> bool:
    """True when every planned sim has analysis summary + reporter HTML on disk."""
    sim_prompts = state.get("sim_prompts") or []
    if not sim_prompts:
        return False
    for sp in sim_prompts:
        wd = sp.get("working_dir") or ""
        if not wd:
            base = state.get("multi_sim_base_dir") or state.get("working_directory")
            wd = str(Path(base) / (sp.get("label") or ""))
        if not per_sim_post_hpc_artifacts_ready(wd):
            return False
    return True


def ensure_multi_sim_progress(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if state.get("multi_sim_progress"):
        return state["multi_sim_progress"]
    if state.get("is_multi_simulation") and state.get("sim_prompts"):
        return init_multi_sim_progress(state)
    return None


def _sim_record(progress: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    return (progress.get("sims") or {}).get(label)


def mark_agent_status(
    state: Dict[str, Any],
    sim_label: str,
    agent_key: str,
    status: str,
) -> None:
    """Update agent status for a simulation (done | pending | in_progress | skipped)."""
    progress = ensure_multi_sim_progress(state)
    if not progress:
        return
    rec = _sim_record(progress, sim_label)
    if not rec:
        return
    if str(rec.get("status") or "").lower() == "failed" and status != "failed":
        # Do not resurrect a fatally failed simulation via agent completion.
        return
    agents = rec.setdefault("agents", {})
    internal = _CHECKPOINT_TO_AGENT.get(agent_key, agent_key)
    if internal == "simsetup":
        internal = "simsetup"
    # Normalize preprocessing name
    for key in list(agents.keys()):
        if key == internal or _CHECKPOINT_TO_AGENT.get(key, key) == internal:
            agents[key] = status
            break
    else:
        agents[internal] = status

    req = progress.get("required_agents") or []
    if status == "failed" or str(rec.get("health") or "").lower() == "failed":
        rec["status"] = "failed"
    elif req and all(agents.get(a) in ("done", "skipped") for a in req):
        rec["status"] = "done"
    elif any(agents.get(a) == "done" for a in req):
        rec["status"] = "in_progress"
    progress["active_sim_label"] = sim_label
    progress["active_agent"] = agent_key if status != "done" else progress.get("active_agent")
    state["multi_sim_progress"] = progress


def record_agent_finished(state: Dict[str, Any], agent_key: str) -> None:
    """Call when a field agent completes successfully (before HITL pause)."""
    phase = state.get("multi_sim_phase")
    if phase in ("combined_analysis", "combined_reporter"):
        progress = ensure_multi_sim_progress(state)
        if not progress:
            return
        progress["phase"] = phase
        combined = progress.setdefault("combined", {"analysis": "pending", "reporter": "pending"})
        combined[agent_key] = "done"
        progress["active_sim_label"] = None
        progress["active_agent"] = agent_key
        progress["hitl_paused_at"] = {
            "sim_label": None,
            "scope": "combined",
            "agent": agent_key,
            "checkpoint": f"human_{agent_key}_check",
        }
        state["multi_sim_progress"] = progress
        from agentic.hitl_router import bind_combined_hitl_view

        bind_combined_hitl_view(state, agent_key)
        return

    progress = ensure_multi_sim_progress(state)
    if not progress:
        return
    label = workflow_sim_label_for_hitl(state)
    if not label:
        return
    internal = agent_key
    if agent_key == "preprocess":
        internal = "preprocessing"
    mark_agent_status(state, label, internal, "done")
    progress = state["multi_sim_progress"]
    progress["hitl_paused_at"] = {
        "sim_label": label,
        "agent": internal,
        "checkpoint": f"human_{agent_key}_check" if agent_key != "preprocessing" else "human_preprocess_check",
    }
    state["multi_sim_progress"] = progress
    if label:
        state["hitl_target_sim_label"] = label


def workflow_sim_label_for_hitl(state: Dict[str, Any]) -> Optional[str]:
    """Simulation the workflow is executing (for HITL checkpoint display)."""
    progress = state.get("multi_sim_progress") or {}
    paused = progress.get("hitl_paused_at") or {}
    if paused.get("sim_label"):
        return paused["sim_label"]

    idx = state.get("current_sim_index")
    sim_prompts = state.get("sim_prompts") or []
    if idx is not None and 0 <= idx < len(sim_prompts):
        label = sim_prompts[idx].get("label")
        if label:
            return label

    wd = state.get("working_directory")
    base = state.get("multi_sim_base_dir") or state.get("working_directory")
    if wd and base:
        wdp = Path(wd).resolve()
        basep = Path(base).resolve()
        if wdp.parent == basep and wdp.name not in ("supervisor", "planner", "programmer"):
            return wdp.name

    if progress.get("active_sim_label"):
        return progress["active_sim_label"]
    return None


def _next_pending_agent(progress: Dict[str, Any], sim_label: str) -> Optional[str]:
    rec = _sim_record(progress, sim_label)
    if not rec:
        return None
    if rec.get("status") in ("failed", "skipped"):
        return None
    agents = progress.get("required_agents") or []
    done = rec.get("agents") or {}
    for a in agents:
        if done.get(a) not in ("done", "skipped"):
            return a
    return None


def _next_incomplete_sim(progress: Dict[str, Any]) -> Optional[str]:
    for label in progress.get("sim_order") or []:
        rec = _sim_record(progress, label)
        if not rec:
            continue
        if rec.get("status") in ("failed", "skipped"):
            continue
        if _sim_all_agents_done(progress, label):
            continue
        return label
    return None


def _load_per_sim_checkpoint(working_dir: str) -> Optional[Dict[str, Any]]:
    """Load the latest per-simulation supervisor checkpoint, if present."""
    state_path = Path(working_dir) / "supervisor" / "state.jsonl"
    if not state_path.is_file():
        return None
    try:
        entry = json.loads(state_path.read_text(encoding="utf-8"))
        return entry.get("state") or entry
    except RecursionError:
        # Can surface when the caller is already near the recursion limit;
        # avoid logging here (logging itself may need stack for isinstance).
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read per-sim state at %s: %s", state_path, exc)
        return None


def _sim_agents_complete(rec: Dict[str, Any], agents: List[str]) -> bool:
    done = rec.get("agents") or {}
    return bool(agents) and all(done.get(a) == "done" for a in agents)


def _ordered_labels_from_hint(sim_order: List[str], hint: Optional[str]) -> List[str]:
    """Return sim_order rotated to start at *hint* (queue order for resume)."""
    if not sim_order:
        return []
    if not hint or hint not in sim_order:
        return list(sim_order)
    idx = sim_order.index(hint)
    return sim_order[idx:] + sim_order[:idx]


def _resume_start_label(progress: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
    """
    Base-level hint for which simulation was interrupted.

    Prefer ``multi_sim_progress.active_sim_label``, then ``current_sim_index``.
    """
    hint = progress.get("active_sim_label")
    if hint:
        return hint
    sim_order = progress.get("sim_order") or []
    idx = state.get("current_sim_index")
    if idx is not None and 0 <= int(idx) < len(sim_order):
        return sim_order[int(idx)]
    return sim_order[0] if sim_order else None


def _reconcile_sim_agents_from_per_sim_state(
    rec: Dict[str, Any],
    agents: List[str],
    per_sim: Optional[Dict[str, Any]],
) -> None:
    """
    Merge per-simulation ``supervisor/state.jsonl`` into agent completion.

    On-disk artifacts are authoritative; in-memory checkpoint fields only fill
    gaps when the corresponding artifact already exists.
    """
    if not per_sim or not rec:
        return
    wd = rec.get("working_dir") or ""
    rec.setdefault("agents", {})

    if per_sim.get("analysis_results") and "analysis" in agents:
        if wd and per_sim_analysis_done_on_disk(wd):
            rec["agents"]["analysis"] = "done"
    if per_sim.get("reporter_output") and "reporter" in agents:
        if wd and per_sim_reporter_done_on_disk(wd):
            rec["agents"]["reporter"] = "done"

    # Only honor continuation/job metadata that lives under this sim's directory.
    if "hpc" in agents and wd and rec["agents"].get("hpc") != "done":
        manifest = per_sim.get("continuation_manifest")
        job = per_sim.get("continuation_job_id") or per_sim.get("job_id")
        local = False
        if manifest:
            try:
                local = Path(manifest).resolve().is_relative_to(Path(wd).resolve())
            except (OSError, ValueError):
                local = False
        if local and job:
            evidence = _local_hpc_submission_evidence(Path(wd))
            if evidence == "done":
                rec["agents"]["hpc"] = "done"
            elif evidence == "in_progress":
                rec["agents"]["hpc"] = "in_progress"

    _merge_pipeline_from_per_sim(rec, per_sim)


def _reconcile_sim_record(
    rec: Dict[str, Any],
    agents: List[str],
    *,
    per_sim: Optional[Dict[str, Any]] = None,
) -> None:
    """Refresh one sim's agent status from disk artifacts and per-sim checkpoint."""
    _reconcile_sim_agents_from_disk(rec, agents)
    _reconcile_sim_agents_from_per_sim_state(rec, agents, per_sim)
    if _sim_agents_complete(rec, agents):
        rec["status"] = "done"
    elif any((rec.get("agents") or {}).get(a) == "done" for a in agents):
        rec["status"] = "in_progress"


def _resolve_resume_target(
    state: Dict[str, Any],
    progress: Dict[str, Any],
) -> Optional[tuple[str, Optional[str]]]:
    """
    Choose the next simulation and agent to resume.

    Policy:
    1. Start from the base-level interrupted sim (``active_sim_label`` / index).
    2. For each candidate sim, reconcile disk + per-sim ``state.jsonl``.
    3. If that sim is fully complete, advance to the next sim in queue order.
    4. Otherwise return the first pending agent for that sim.
    """
    agents = progress.get("required_agents") or _required_agents(state)
    sim_order = progress.get("sim_order") or []
    hint = _resume_start_label(progress, state)

    for label in _ordered_labels_from_hint(sim_order, hint):
        rec = _sim_record(progress, label)
        if not rec:
            continue
        wd = rec.get("working_dir") or ""
        if not wd:
            base = state.get("multi_sim_base_dir") or state.get("working_directory")
            wd = str(Path(base) / label)
            rec["working_dir"] = wd
        per_sim = _load_per_sim_checkpoint(wd)
        _reconcile_sim_record(rec, agents, per_sim=per_sim)
        if _sim_agents_complete(rec, agents):
            continue
        next_agent = _next_pending_agent(progress, label)
        return label, next_agent
    return None


def multisim_resume_entry_allowed(state: Dict[str, Any], multi_sim_phase: Optional[str]) -> bool:
    """True when supervisor should run the multi-sim ``--resume`` entry path."""
    if not state.get("resume_failed_only"):
        return False
    if multi_sim_phase is None:
        return True
    return multi_sim_phase in (
        "executing_sims",
        "combined_analysis",
        "combined_reporter",
        "parallel_pool",
    )


def _local_hpc_submission_evidence(sim_dir: Path) -> Optional[str]:
    """
    Return ``done`` / ``in_progress`` when this sim dir has its own HPC evidence.

    A non-empty ``hpc/`` from a prior 100 ns run is not enough — that used to
    mark every continuation target ``hpc=in_progress`` and confuse multi-sim
    progress with false completions.

    A continuation manifest that already has a SLURM ``job_id`` counts as
    ``done`` for multi-sim routing (do not re-prepare / re-submit).
    """
    info = load_local_continuation_job(sim_dir)
    if info and info.get("complete"):
        return "done"
    if info and info.get("job_id"):
        return "done"
    if info and info.get("manifest_path"):
        return "in_progress"
    hpc_dir = Path(sim_dir) / "hpc"
    if not hpc_dir.is_dir():
        return None
    label = Path(sim_dir).name
    if label and list(hpc_dir.glob(f"{label}_*.out")):
        return "in_progress"
    return None


def load_local_continuation_job(sim_dir: Optional[str] | Path) -> Optional[Dict[str, Any]]:
    """Read this sim's ``hpc/continuation_200ns.json`` job metadata, if any."""
    if not sim_dir:
        return None
    root = Path(sim_dir)
    hpc_dir = root / "hpc" if (root / "hpc").is_dir() else root
    manifest = hpc_dir / "continuation_200ns.json"
    complete = hpc_dir / ".continuation_200ns.complete"
    out: Dict[str, Any] = {
        "simulation_dir": str(hpc_dir.resolve()) if hpc_dir.is_dir() else str(root),
        "complete": complete.is_file(),
    }
    if not manifest.is_file():
        return out if out["complete"] else None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    job_id = data.get("job_id")
    out.update(
        {
            "manifest_path": str(manifest.resolve()),
            "job_id": str(job_id) if job_id not in (None, "") else None,
            "status": data.get("status"),
            "job_name": data.get("job_name"),
            "job_script": data.get("job_script"),
        }
    )
    return out


def hydrate_continuation_job_into_state(state: Dict[str, Any]) -> bool:
    """
    Copy a *local* continuation job id from disk into ``state``.

    Returns True when ``continuation_job_id`` / ``job_id`` now refer to this
    sim's own manifest (so callers can mark HPC done).
    """
    wd = state.get("working_directory")
    info = load_local_continuation_job(wd)
    if not info or not info.get("job_id"):
        return False
    state["continuation_job_id"] = str(info["job_id"])
    state["job_id"] = str(info["job_id"])
    status = str(info.get("status") or "SUBMITTED")
    state["continuation_status"] = status
    state["job_status"] = status
    if info.get("manifest_path"):
        state["continuation_manifest"] = info["manifest_path"]
    if info.get("job_script"):
        state["job_script"] = info["job_script"]
    return True


def _reconcile_sim_agents_from_disk(rec: Dict[str, Any], agents: List[str]) -> None:
    """Update agent status from on-disk artifacts (resume / continue helper)."""
    if not rec:
        return
    sim_dir = Path(rec.get("working_dir") or "")
    if not sim_dir.is_dir():
        return
    rec.setdefault("agents", {})
    # A finished production trajectory means prep and HPC succeeded. Leaving
    # those stages at "in_progress" made resume restart preprocessing and
    # re-queue the simulation after analysis had already written its report.
    try:
        from agentic.multi_sim_hpc_pool import _sim_production_trajectory_ready

        if _sim_production_trajectory_ready(str(sim_dir)):
            for stage in ("preprocessing", "simsetup", "hpc"):
                if stage in agents:
                    rec["agents"][stage] = "done"
    except Exception:
        pass
    for agent in agents:
        sub = {
            "preprocessing": "preprocess",
            "simsetup": "simsetup",
            "hpc": "hpc",
            "analysis": "analysis",
            "reporter": "reporter",
        }.get(agent, agent)
        marker = sim_dir / sub
        if agent == "hpc":
            evidence = _local_hpc_submission_evidence(sim_dir)
            if evidence == "done":
                rec["agents"][agent] = "done"
            elif evidence == "in_progress" and rec["agents"].get(agent) != "done":
                rec["agents"][agent] = "in_progress"
            elif rec["agents"].get(agent) not in ("done", "failed"):
                # Prior production files alone do not mean this run's HPC started.
                rec["agents"][agent] = "pending"
        elif agent == "analysis" and (marker / "analysis_summary.jsonl").is_file():
            if per_sim_analysis_done_on_disk(str(sim_dir)):
                rec["agents"][agent] = "done"
            elif rec["agents"].get(agent) != "done":
                rec["agents"][agent] = "in_progress"
        elif agent == "reporter" and per_sim_reporter_done_on_disk(str(sim_dir)):
            rec["agents"][agent] = "done"
        elif agent == "reporter" and list(marker.glob("*.html")):
            # Orphan HTML without analysis — do not treat as complete.
            if rec["agents"].get(agent) != "done":
                rec["agents"][agent] = "pending"
        elif marker.is_dir() and any(marker.iterdir()):
            if rec["agents"].get(agent) != "done":
                rec["agents"][agent] = "in_progress"
    if all(rec["agents"].get(a) == "done" for a in agents):
        rec["status"] = "done"
    elif any(rec["agents"].get(a) == "done" for a in agents):
        rec["status"] = "in_progress"


def reconcile_progress_from_disk(state: Dict[str, Any], progress: Dict[str, Any]) -> None:
    """Refresh per-sim agent completion from artifacts before HITL continue/resume."""
    agents = progress.get("required_agents") or _required_agents(state)
    for label in progress.get("sim_order") or []:
        rec = _sim_record(progress, label)
        if rec:
            wd = rec.get("working_dir") or ""
            per_sim = _load_per_sim_checkpoint(wd) if wd else None
            _reconcile_sim_record(rec, agents, per_sim=per_sim)


def _stats_csv_has_multiple_sims(path: Path) -> bool:
    """True when a combined stats CSV has ≥2 non-error simulation rows."""
    try:
        lines = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    except OSError:
        return False
    if len(lines) < 3:  # header + 2 data rows
        return False
    data_rows = lines[1:]
    ok = 0
    for row in data_rows:
        parts = row.split(",")
        if len(parts) < 2:
            continue
        # Prefer rows without an error column filled, or with numeric mean
        lower = row.lower()
        if "no such file" in lower or lower.endswith(",error") or ",[errno" in lower:
            continue
        ok += 1
    return ok >= 2


def _stats_csv_labels(path: Path) -> set:
    """Simulation labels present in a combined stats CSV (first column)."""
    labels: set = set()
    try:
        lines = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
    except OSError:
        return labels
    for row in lines[1:]:
        parts = row.split(",")
        if not parts:
            continue
        lower = row.lower()
        if "no such file" in lower or ",[errno" in lower:
            continue
        lab = parts[0].strip().strip('"').lower()
        if lab and lab != "simulation":
            labels.add(lab)
    return labels


def family_post_features_ok(analysis_dir: str | Path) -> bool:
    """True when a family campaign wrote a classification feature table."""
    adir = Path(analysis_dir)
    for name in (
        "classification_features.csv",
        "classification_features_selected.csv",
    ):
        path = adir / name
        if path.is_file() and path.stat().st_size > 0:
            return True
    return False


def post_combined_artifacts_ok(
    analysis_dir: str | Path,
    expected_labels: Optional[List[str]] = None,
    *,
    family_modular: bool = False,
) -> bool:
    """True when base/analysis has usable multi-sim overlay / stats artifacts.

    Used for post-combined success (artifact-based) so optional LLM step
    failures (comparison tables, multipanel) do not block the reporter.

    Family campaigns also need a classification feature table when that is
    how the pack is judged complete — overlays alone are not enough.
    """
    adir = Path(analysis_dir)
    if not adir.is_dir():
        return False

    overlays = sorted(adir.glob("*_overlay.png"))
    strong = (
        adir / "classification_features.csv",
        adir / "classification_clusters.json",
        adir / "com_distance_overlay.png",
        adir / "ligand_rmsf_overlay.png",
        adir / "pocket_rmsf_overlay.png",
    )
    has_strong = any(p.is_file() for p in strong) or bool(overlays)
    if not has_strong:
        return False

    need = {str(x).strip().lower() for x in (expected_labels or []) if str(x).strip()}
    if len(need) < 2:
        # Unknown / single-label set: overlays or multi-row stats are enough
        if overlays:
            return True
        return any(
            (adir / name).is_file() and _stats_csv_has_multiple_sims(adir / name)
            for name in ("rmsd_stats.csv", "rmsf_stats.csv", "rg_stats.csv")
        )

    # Require at least one stats table covering every expected label, or
    # enough overlay plots when stats are missing but overlays exist.
    for stats_name in ("rmsd_stats.csv", "rmsf_stats.csv", "rg_stats.csv"):
        stats_path = adir / stats_name
        if not stats_path.is_file():
            continue
        have = _stats_csv_labels(stats_path)
        if need.issubset(have):
            return True

    # Overlay-only fallback is enough for generic multi-sim, not family packs.
    if family_modular:
        return family_post_features_ok(adir)
    return len(overlays) >= 1


def _expected_combined_labels(state: Optional[Dict[str, Any]]) -> List[str]:
    """Labels combined analysis must cover — prefer currently healthy production sims."""
    if not state:
        return []
    prompts = state.get("sim_prompts") or []
    all_labels: List[str] = []
    healthy: List[str] = []
    try:
        from agentic.utils.sim_health import production_health, is_reuse_hpc
    except Exception:
        production_health = None  # type: ignore[assignment]
        is_reuse_hpc = None  # type: ignore[assignment]

    for sp in prompts:
        label = str(sp.get("label") or "").strip()
        if not label:
            continue
        all_labels.append(label)
        wd = sp.get("working_dir") or ""
        if production_health and wd:
            try:
                ok = bool(
                    production_health(
                        wd,
                        reuse_hpc=bool(is_reuse_hpc(state)) if is_reuse_hpc else False,
                    ).get("healthy")
                )
            except Exception:
                ok = False
            if ok:
                healthy.append(label)
    # Healthy-only so a 5/8 partial combined is not "done" after more sims finish.
    return healthy if healthy else all_labels


def _combined_analysis_done_on_disk(
    base: Path,
    state: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when base-level combined analysis produced usable multi-sim artifacts.

    Partial overlays from a subset of sims must not count as done once more
    simulations become healthy (e.g. after ``--resume`` repairs failed jobs).
    """
    analysis_dir = base / "analysis"
    if not analysis_dir.is_dir():
        return False

    expected_labels = _expected_combined_labels(state)
    if not expected_labels and state:
        expected_labels = [
            str(sp.get("label") or "").strip()
            for sp in (state.get("sim_prompts") or [])
            if sp.get("label")
        ]
    family = False
    try:
        from agentic.campaign.spec import spec_from_state

        spec = spec_from_state(state)
        family = bool(spec and (spec.family_modular or spec.mode == "family_modular"))
    except Exception:
        family = False

    # Authoritative: completion marker must list every expected (healthy) label.
    # If the marker says success=false but overlays/stats on disk cover the
    # expected sims, treat as done (optional LLM steps may have failed).
    marker = analysis_dir / "combined_analysis_complete.json"
    if marker.is_file():
        try:
            import json as _json

            data = _json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        labels = [str(x) for x in (data.get("labels") or []) if x]
        have = {x.lower() for x in labels}
        if expected_labels and len(expected_labels) >= 2:
            need = {x.lower() for x in expected_labels}
            labels_ok = need.issubset(have)
            if labels_ok and bool(data.get("success", True)):
                return True
            if labels_ok and post_combined_artifacts_ok(
                analysis_dir, expected_labels=expected_labels, family_modular=family
            ):
                return True
        elif len(labels) >= 2 and (
            bool(data.get("success", True))
            or post_combined_artifacts_ok(
                analysis_dir, expected_labels=labels, family_modular=family
            )
        ):
            return True

    # Known multi-sim set: allow artifact-based done without a success marker
    if expected_labels and len(expected_labels) >= 2:
        return post_combined_artifacts_ok(
            analysis_dir, expected_labels=expected_labels, family_modular=family
        )

    # Legacy / unknown label set: keep artifact heuristics.
    return post_combined_artifacts_ok(
        analysis_dir, expected_labels=None, family_modular=family
    )


def _combined_reporter_done_on_disk(
    base: Path,
    state: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when combined HTML exists and combined analysis is not stale."""
    reporter_dir = base / "reporter"
    if not (reporter_dir / "combined_report.html").is_file():
        return False
    if state is not None and not _combined_analysis_done_on_disk(base, state=state):
        return False
    return True


def reconcile_multisim_progress_from_disk(
    state: Dict[str, Any],
    base_dir: Optional[str] = None,
) -> None:
    """
    Refresh multi_sim_progress from on-disk artifacts before resume routing.

    Ensures ``--resume`` skips per-simulation work (and combined analysis) that
    already finished, even when the saved checkpoint still shows ``pending``.
    """
    if not state.get("is_multi_simulation"):
        return
    if not state.get("sim_prompts") and not state.get("multi_sim_progress"):
        return

    base = Path(
        base_dir
        or state.get("multi_sim_base_dir")
        or state.get("working_directory")
        or "."
    ).resolve()

    if not state.get("multi_sim_progress"):
        init_multi_sim_progress(state)

    progress = state.get("multi_sim_progress") or {}
    # A compacted checkpoint can keep the phase and drop the per-sim list.
    # Rebuild that list from the master plan before resume routing, or an
    # empty queue is treated as "per-sim work finished."
    if not progress.get("sim_order") and state.get("sim_prompts"):
        init_multi_sim_progress(state)
        progress = state.get("multi_sim_progress") or {}
    agents = progress.get("required_agents") or _required_agents(state)

    # Combined-only skips the per-sim loop; treat planned sims as done so we
    # never force progress.phase back to executing_sims during save/reconcile.
    if state.get("combined_only"):
        for label in progress.get("sim_order") or []:
            rec = _sim_record(progress, label)
            if not rec:
                continue
            wd = rec.get("working_dir") or str(base / label)
            rec["working_dir"] = wd
            agent_map = rec.setdefault("agents", {})
            for a in agents:
                agent_map[a] = "done"
            rec["status"] = "done"
    else:
        for label in progress.get("sim_order") or []:
            rec = _sim_record(progress, label)
            if not rec:
                continue
            wd = rec.get("working_dir") or str(base / label)
            rec["working_dir"] = wd
            per_sim = _load_per_sim_checkpoint(wd)
            _reconcile_sim_record(rec, agents, per_sim=per_sim)

    from src.analysis.cross_sim_artifacts import pre_combined_done_on_disk

    wants_combined = bool(
        state.get("run_combined_analysis")
        or state.get("run_post_combined")
        or state.get("combined_only")
        or progress.get("combined")
        or _combined_analysis_done_on_disk(base, state=state)
        or _combined_reporter_done_on_disk(base, state=state)
    )
    if wants_combined:
        combined = progress.setdefault(
            "combined",
            {"analysis": "pending", "reporter": "pending"},
        )
        if _combined_analysis_done_on_disk(base, state=state):
            combined["analysis"] = "done"
        if _combined_reporter_done_on_disk(base, state=state):
            combined["reporter"] = "done"
        state["run_combined_analysis"] = True
        state["run_post_combined"] = True

    # Preserve in-flight pre_combined
    current_phase = state.get("multi_sim_phase")
    if current_phase == "pre_combined" or (
        state.get("run_pre_combined")
        and not pre_combined_done_on_disk(base)
        and not all_per_sim_agents_done(state)
        and current_phase
        not in (
            "executing_sims",
            "hpc_pool",
            "parallel_pool",
            "combined_analysis",
            "post_combined",
            "combined_reporter",
            "complete",
        )
    ):
        if not pre_combined_done_on_disk(base):
            progress["phase"] = "pre_combined"
            progress.setdefault("pre_combined", {"analysis": "pending"})
            progress["active_sim_label"] = None
            progress["active_agent"] = "analysis"
            state["multi_sim_phase"] = "pre_combined"
            state["multi_sim_progress"] = progress
            _apply_progress_indices(state, rebind=False)
            merge_completed_states_from_progress(state)
            return

    # Preserve an in-flight combined phase set by the analysis agent even when
    # disk artifacts are not yet "done" (e.g. mid-transition to reporter).
    if current_phase in ("combined_analysis", "post_combined", "combined_reporter") and wants_combined:
        combined = progress.setdefault(
            "combined",
            {"analysis": "pending", "reporter": "pending"},
        )
        if current_phase == "combined_reporter":
            combined["analysis"] = "done"
            progress["phase"] = "combined_reporter"
            progress["active_sim_label"] = None
            progress["active_agent"] = "reporter"
        else:
            progress["phase"] = "combined_analysis"
            progress["active_sim_label"] = None
            progress["active_agent"] = "analysis"
        state["multi_sim_progress"] = progress
        _apply_progress_indices(state, rebind=False)
        merge_completed_states_from_progress(state)
        logger.info(
            "[resume] Reconciled multi_sim_progress from disk at %s\n%s",
            base,
            progress_summary(state),
        )
        return

    if all_per_sim_agents_done(state):
        combined = progress.get("combined") or {}
        if combined:
            if combined.get("analysis") != "done":
                progress["phase"] = "combined_analysis"
                progress["active_sim_label"] = None
                progress["active_agent"] = "analysis"
            elif combined.get("reporter") != "done":
                progress["phase"] = "combined_reporter"
                progress["active_sim_label"] = None
                progress["active_agent"] = "reporter"
            else:
                progress["phase"] = "complete"
                progress["active_sim_label"] = None
                progress["active_agent"] = None
        else:
            progress["phase"] = "complete"
            progress["active_sim_label"] = None
            progress["active_agent"] = None
    else:
        progress["phase"] = "executing_sims"
        target = _resolve_resume_target(state, progress)
        if target:
            active, next_agent = target
            progress["active_sim_label"] = active
            progress["active_agent"] = next_agent
        else:
            active = _next_incomplete_sim(progress)
            if active:
                progress["active_sim_label"] = active
                progress["active_agent"] = _next_pending_agent(progress, active)

    state["multi_sim_progress"] = progress
    _apply_progress_indices(state, rebind=False)
    merge_completed_states_from_progress(state)
    logger.info(
        "[resume] Reconciled multi_sim_progress from disk at %s\n%s",
        base,
        progress_summary(state),
    )


def _workflow_pause_sim(progress: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
    """
    Simulation where the workflow paused for HITL (authoritative for continue).

    Ignores HITL chat ``switch`` binds — only ``hitl_paused_at`` and progress.
    """
    paused = progress.get("hitl_paused_at") or {}
    if paused.get("sim_label"):
        return paused["sim_label"]
    label = _next_incomplete_sim(progress)
    if label:
        return label
    return progress.get("active_sim_label")


def _sim_all_agents_done(progress: Dict[str, Any], sim_label: str) -> bool:
    rec = _sim_record(progress, sim_label)
    if not rec:
        return False
    if rec.get("status") in ("failed", "skipped"):
        return True
    agents = progress.get("required_agents") or []
    done = rec.get("agents") or {}
    if not agents:
        return rec.get("status") == "done"
    return all(done.get(a) in ("done", "skipped") for a in agents)


def _route_after_per_sim_continue(
    progress: Dict[str, Any],
    pause_sim: str,
    finished_agent: str,
    agent_was_done_before_continue: bool,
) -> Optional[tuple]:
    """
    Return (target_sim, target_agent) after *finished_agent* was marked done on *pause_sim*.

    When the pause sim is already fully complete (e.g. reporter exists on disk but
    progress was stale), advance to the next incomplete simulation.
    """
    rec = _sim_record(progress, pause_sim)
    if not rec:
        return None

    if _sim_all_agents_done(progress, pause_sim):
        rec["status"] = "done"
        nxt = _next_incomplete_sim(progress)
        if nxt:
            return nxt, _next_pending_agent(progress, nxt)
        return None

    if agent_was_done_before_continue:
        pending = _next_pending_agent(progress, pause_sim)
        if pending:
            return pause_sim, pending
        nxt = _next_incomplete_sim(progress)
        if nxt:
            return nxt, _next_pending_agent(progress, nxt)
        return None

    pending = _next_pending_agent(progress, pause_sim)
    if pending:
        return pause_sim, pending

    nxt = _next_incomplete_sim(progress)
    if nxt:
        return nxt, _next_pending_agent(progress, nxt)
    return None


def resolve_active_sim_label(state: Dict[str, Any]) -> Optional[str]:
    """Best-effort label for the simulation currently being prepared or executed."""
    sim_case = state.get("sim_case") or {}
    if sim_case.get("label") and (
        state.get("hpc_pool_prep_only")
        or state.get("post_hpc_analysis_only")
        or state.get("multi_sim_phase") == "executing_sims"
    ):
        return sim_case["label"]

    if state.get("hpc_pool_prep_only"):
        from agentic.multi_sim_hpc_pool import _next_prep_label

        pool = state.get("hpc_pool") or {}
        pending = _next_prep_label(pool, state.get("sim_prompts") or [])
        if pending:
            return pending

    progress = state.get("multi_sim_progress") or {}
    if progress.get("active_sim_label"):
        return progress["active_sim_label"]

    sim_prompts = state.get("sim_prompts") or []
    idx = state.get("current_sim_index", 0)
    if sim_prompts and 0 <= idx < len(sim_prompts):
        return sim_prompts[idx].get("label")
    return None


def ensure_per_sim_working_directory(state: Dict[str, Any]) -> bool:
    """
    Align ``working_directory`` with the active per-sim context.

    Prevents stale ``multi_sim_progress.active_sim_label`` from routing prep
    for sim N+1 into sim N's directory after checkpoint restore or sync.
    """
    if not state.get("is_multi_simulation") or state.get("combined_only"):
        return False

    phase = state.get("multi_sim_phase")
    if phase == "hpc_pool":
        if not state.get("hpc_pool_prep_only") and not state.get("post_hpc_analysis_only"):
            return False
    elif phase != "executing_sims":
        return False

    label = resolve_active_sim_label(state)
    if not label:
        return False

    sim_prompts = state.get("sim_prompts") or []
    idx = next(
        (i for i, sp in enumerate(sim_prompts) if sp.get("label") == label),
        None,
    )
    if idx is None:
        return False

    sp = sim_prompts[idx]
    base = state.get("multi_sim_base_dir") or state.get("working_directory") or "."
    expected_wd = sp.get("working_dir") or str(Path(base) / label)
    expected_wd = str(Path(expected_wd).resolve())
    current_wd = str(Path(state.get("working_directory", ".")).resolve())

    if current_wd == expected_wd:
        state["current_sim_index"] = idx
        return False

    bind_workflow_to_sim(state, label)
    state["current_sim_index"] = idx
    progress = state.get("multi_sim_progress") or {}
    if progress:
        progress["active_sim_label"] = label
        state["multi_sim_progress"] = progress

    from agentic.utils.conversation_logger import set_log_file

    set_log_file(str(Path(expected_wd) / "agent_conversation.log"))
    logger.warning(
        "Re-bound working_directory to %s for active sim %s (was %s)",
        expected_wd,
        label,
        current_wd,
    )
    return True


def bind_workflow_to_sim(state: Dict[str, Any], sim_label: str) -> None:
    """Point workflow routing fields at *sim_label* without wiping artifacts."""
    progress = ensure_multi_sim_progress(state)
    if not progress:
        return
    rec = _sim_record(progress, sim_label)
    if not rec:
        return
    prev_label = progress.get("active_sim_label")
    wd = rec.get("working_dir") or ""
    if not wd:
        base = state.get("multi_sim_base_dir") or state.get("working_directory")
        wd = str(Path(base) / sim_label)
    wd = str(Path(wd).resolve())
    same_sim = (
        prev_label == sim_label
        and str(Path(state.get("working_directory") or "").resolve()) == wd
    )
    state["working_directory"] = wd
    state["current_sim_index"] = rec.get("index", 0)
    state["preprocess_dir"] = str(Path(wd) / "preprocess")
    state["simsetup_dir"] = str(Path(wd) / "simsetup")
    state["hpc_dir"] = str(Path(wd) / "hpc")
    state["analysis_dir"] = str(Path(wd) / "analysis")
    state["analysis_directory"] = str(Path(wd) / "analysis")
    state["reporter_dir"] = str(Path(wd) / "reporter")
    progress["active_sim_label"] = sim_label
    state["multi_sim_progress"] = progress

    sim_prompts = state.get("sim_prompts") or []
    sp = next((item for item in sim_prompts if item.get("label") == sim_label), None)
    if sp:
        sim_goal = sp.get("analysis_prompt") or sp.get("prompt") or ""
        if sim_goal:
            state["user_goal"] = sim_goal
        enriched = (sp.get("enriched_prompt") or "").strip()
        if enriched:
            state["enriched_prompt"] = enriched
            state["rephrased_goal"] = enriched
        sim_pdb = sp.get("pdb") or ""
        if sim_pdb and Path(sim_pdb).is_file():
            state["raw_pdb"] = sim_pdb

    if not same_sim:
        for key in (
            "topology",
            "trajectory_path",
            "energy_file",
            "hpc_output_directory",
            "analysis_results",
            "reporter_output",
            "execution_plan",
            "analysis_instructions",
            "reporter_instructions",
            "job_id",
            "job_status",
            "job_script",
            "continuation_job_id",
            "continuation_status",
            "continuation_manifest",
            "continuation_simulation_dir",
            "continuation_source_tpr",
            "continuation_checkpoint",
            "continuation_tpr",
            "continuation_current_ns",
            "continuation_target_total_ns",
        ):
            state[key] = None
        state["plan_executed"] = False
        state["input_validated"] = None
        state["hpc_retry_count"] = 0
        paused = progress.get("hitl_paused_at")
        if isinstance(paused, dict) and paused.get("sim_label") != sim_label:
            progress["hitl_paused_at"] = None
            state.pop("hitl_target_sim_label", None)

    apply_sim_pipeline_to_state(state)

    hpc_dir = Path(wd) / "hpc"
    tpr = hpc_dir / "md.tpr"
    xtc = hpc_dir / "mdWrap.xtc"
    if not xtc.is_file():
        xtc = hpc_dir / "md.xtc"
    if tpr.is_file():
        state["topology"] = str(tpr.resolve())
    if xtc.is_file():
        state["trajectory_path"] = str(xtc.resolve())
    if hpc_dir.is_dir():
        state["hpc_output_directory"] = str(hpc_dir.resolve())


def sync_state_from_progress(state: Dict[str, Any], *, rebind: bool = True) -> None:
    """Apply multi_sim_progress → current_sim_index, current_agent_idx, plan_executed."""
    _apply_progress_indices(state, rebind=rebind)


def apply_hitl_continue(state: Dict[str, Any], checkpoint_type: str) -> None:
    """
    Deterministic routing after HITL ``continue`` / ``approved``.

    Uses base-level ``multi_sim_progress`` — ignores HITL chat sim binding.
    """
    try:
        from agentic.campaign.hitl_campaign import continue_campaign_checkpoint

        if continue_campaign_checkpoint(state, checkpoint_type):
            return
    except Exception:
        logger.debug("campaign HITL continue hook skipped", exc_info=True)

    phase = state.get("multi_sim_phase")
    if phase == "combined_analysis" and checkpoint_type == "analysis":
        progress = ensure_multi_sim_progress(state) or {}
        combined = progress.setdefault("combined", {})
        combined["analysis"] = "done"
        combined["reporter"] = "in_progress"
        progress["phase"] = "combined_reporter"
        progress["active_sim_label"] = None
        progress["active_agent"] = "reporter"
        progress["hitl_paused_at"] = None
        state["multi_sim_progress"] = progress
        state["multi_sim_phase"] = "combined_reporter"
        state["current_agent_idx"] = 1
        state["plan_executed"] = False
        state["next_node"] = "reporter"
        from agentic.hitl_router import bind_combined_hitl_view

        bind_combined_hitl_view(state, "reporter")
        logger.info("HITL continue: combined analysis done → combined reporter")
        return

    if phase == "combined_reporter" and checkpoint_type == "reporter":
        progress = ensure_multi_sim_progress(state) or {}
        combined = progress.setdefault("combined", {})
        combined["reporter"] = "done"
        progress["phase"] = "complete"
        progress["hitl_paused_at"] = None
        state["multi_sim_progress"] = progress
        state["multi_sim_phase"] = "complete"
        state["plan_executed"] = True
        state["next_node"] = "final_report"
        logger.info("HITL continue: combined reporter done → final report")
        return

    progress = ensure_multi_sim_progress(state)
    if not progress:
        from agentic.hitl_router import restore_multi_sim_executing_working_directory

        restore_multi_sim_executing_working_directory(state)
        state["next_node"] = "supervisor"
        state["plan_executed"] = False
        return

    reconcile_progress_from_disk(state, progress)
    state["multi_sim_progress"] = progress

    paused = progress.get("hitl_paused_at") or {}
    finished_agent = _CHECKPOINT_TO_AGENT.get(checkpoint_type, checkpoint_type)
    pause_sim = _workflow_pause_sim(progress, state)
    if not pause_sim:
        pause_sim = (progress.get("sim_order") or [None])[0]
    if not pause_sim:
        state["next_node"] = "supervisor"
        return

    rec = _sim_record(progress, pause_sim)
    agent_already_done = bool(
        rec and (rec.get("agents") or {}).get(finished_agent) == "done"
    )
    if not agent_already_done:
        mark_agent_status(state, pause_sim, finished_agent, "done")
        progress = state["multi_sim_progress"]

    progress["hitl_paused_at"] = None
    route = _route_after_per_sim_continue(
        progress, pause_sim, finished_agent, agent_already_done
    )

    if route:
        target_sim, next_agent = route
        if target_sim != pause_sim:
            rec = _sim_record(progress, pause_sim)
            if rec:
                rec["status"] = "done"
        progress["active_sim_label"] = target_sim
        progress["active_agent"] = next_agent
        state["multi_sim_progress"] = progress
        bind_workflow_to_sim(state, target_sim)
        if next_agent:
            mark_agent_status(state, target_sim, next_agent, "in_progress")
            progress = state["multi_sim_progress"]
            state["current_agent_idx"] = (progress.get("required_agents") or []).index(next_agent)
        state["plan_executed"] = False

        if target_sim != pause_sim:
            state["next_node"] = "supervisor"
            state["_hitl_start_next_sim"] = True
            logger.info(
                "HITL continue: %s complete (paused at %s/%s) → advance to sim %s agent %s",
                pause_sim, pause_sim, finished_agent, target_sim, next_agent,
            )
            return

        state["next_node"] = _AGENT_TO_NODE.get(next_agent, next_agent)
        logger.info(
            "HITL continue: %s/%s done → next %s on %s",
            pause_sim, finished_agent, next_agent, target_sim,
        )
        return

    # All per-sim work done — start combined analysis/report if requested
    if state.get("run_combined_analysis") is True:
        progress["phase"] = "combined_analysis"
        progress["active_sim_label"] = None
        progress["active_agent"] = "analysis"
        combined = progress.setdefault("combined", {"analysis": "pending", "reporter": "pending"})
        combined["analysis"] = "in_progress"
        state["multi_sim_progress"] = progress
        state["plan_executed"] = False
        state["_hitl_start_combined"] = True
        state["next_node"] = "supervisor"
        logger.info("HITL continue: all per-simulation subtasks complete → combined analysis")
        return

    progress["phase"] = "executing_sims"
    state["multi_sim_progress"] = progress
    state["plan_executed"] = True
    state["next_node"] = "supervisor"
    logger.info("HITL continue: all per-simulation subtasks complete")


def progress_summary(state: Dict[str, Any]) -> str:
    progress = state.get("multi_sim_progress")
    if not progress:
        return ""
    lines = ["Multi-sim progress:"]
    for label in progress.get("sim_order") or []:
        rec = _sim_record(progress, label) or {}
        agents = rec.get("agents") or {}
        parts = ", ".join(f"{a}={agents.get(a, '?')}" for a in (progress.get("required_agents") or []))
        mark = " ← active" if label == progress.get("active_sim_label") else ""
        lines.append(f"  {label}: [{parts}] status={rec.get('status', '?')}{mark}")
    nxt = progress.get("active_agent")
    active = progress.get("active_sim_label")
    if active:
        rec = _sim_record(progress, active) or {}
        if rec.get("status") == "done" or _sim_agents_complete(
            rec, progress.get("required_agents") or []
        ):
            target = _resolve_resume_target(state, progress)
            if target:
                active, nxt = target
                lines.append(f"  Next step: {nxt} on {active}")
        else:
            lines.append(f"  Next step: {nxt} on {active}")
    combined = progress.get("combined") or {}
    if combined:
        lines.append(
            f"  Combined: analysis={combined.get('analysis', '?')}, "
            f"reporter={combined.get('reporter', '?')}"
        )
    return "\n".join(lines)


def rebuild_progress_from_disk(state: Dict[str, Any], base_dir: str) -> None:
    """Infer progress from per-sim supervisor state + artifacts (resume helper)."""
    reconcile_multisim_progress_from_disk(state, base_dir)


def multisim_workflow_incomplete(progress: Optional[Dict[str, Any]]) -> bool:
    """True when base-level progress shows per-sim or combined work still pending."""
    if not progress:
        return False
    for label in progress.get("sim_order") or []:
        rec = _sim_record(progress, label)
        if not rec:
            return True
        if rec.get("status") in ("failed", "skipped"):
            continue
        if rec.get("status") != "done":
            agents = progress.get("required_agents") or []
            done = rec.get("agents") or {}
            if agents and all(done.get(a) in ("done", "skipped") for a in agents):
                continue
            return True
        agents = progress.get("required_agents") or []
        done = rec.get("agents") or {}
        if any(done.get(a) not in ("done", "skipped") for a in agents):
            return True
    combined = progress.get("combined")
    if combined:
        if combined.get("analysis") != "done" or combined.get("reporter") != "done":
            return True
    return False


def prepare_multisim_resume_state(state: Dict[str, Any]) -> Optional[str]:
    """
    Point workflow at the next incomplete sim/agent from ``multi_sim_progress``.

    Returns the simulation label resumed, or None if nothing to resume.
    """
    progress = state.get("multi_sim_progress")
    if not multisim_workflow_incomplete(progress):
        return None
    target = _resolve_resume_target(state, progress or {})
    if target:
        label, next_agent = target
    else:
        label = None
        next_agent = None
    if not label:
        combined = (progress or {}).get("combined") or {}
        if combined.get("analysis") != "done":
            progress["phase"] = "combined_analysis"
            progress["active_sim_label"] = None
            progress["active_agent"] = "analysis"
            state["multi_sim_progress"] = progress
            state["multi_sim_phase"] = "combined_analysis"
            state["plan_executed"] = False
            state["execution_plan"] = None
            state["_hitl_start_combined"] = True
            state["next_node"] = "supervisor"
            logger.info("[resume] Multi-sim per-sim done — resuming combined analysis")
            return "__combined__"
        if combined.get("reporter") != "done":
            progress["phase"] = "combined_reporter"
            progress["active_sim_label"] = None
            progress["active_agent"] = "reporter"
            state["multi_sim_progress"] = progress
            state["multi_sim_phase"] = "combined_reporter"
            state["plan_executed"] = False
            state["execution_plan"] = None
            state["current_agent_idx"] = 1
            from agentic.hitl_router import bind_combined_hitl_view

            bind_combined_hitl_view(state, "reporter")
            state["next_node"] = "reporter"
            logger.info("[resume] Multi-sim combined analysis done — resuming combined reporter")
            return "__combined_reporter__"
        return None
    rec = _sim_record(progress, label)
    if not rec:
        return None
    if not next_agent:
        next_agent = _next_pending_agent(progress, label)
    progress["active_sim_label"] = label
    progress["active_agent"] = next_agent
    state["multi_sim_progress"] = progress
    state["current_sim_index"] = rec.get("index", 0)
    # Never demote an active cross-sim pool phase back to sequential executing_sims.
    # That rebind thrash was aborting --resume mid hpc_pool / parallel_pool.
    pool_phase = state.get("multi_sim_phase")
    if pool_phase not in ("hpc_pool", "parallel_pool"):
        state["multi_sim_phase"] = "executing_sims"
        state["plan_executed"] = False
        state["execution_plan"] = None
        sync_state_from_progress(state, rebind=True)
        apply_sim_pipeline_to_state(state)
    else:
        state["plan_executed"] = False
        logger.info(
            "[resume] Keeping multi_sim_phase=%s (not forcing executing_sims for %s)",
            pool_phase,
            label,
        )
    logger.info(
        "[resume] Multi-sim progress → sim %s agent %s (index %s)",
        label,
        next_agent,
        state.get("current_sim_index"),
    )
    return label

