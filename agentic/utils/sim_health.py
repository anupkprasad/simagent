"""Per-simulation health for orchestration (any MD engine / study type).

A simulation is *healthy* when production MD finished with usable topology +
trajectory. Soft analysis/report problems (plots, literature rate limits) do
not change health. Fatal HPC / prep failure stops analysis and reporter for
that system only; other systems continue.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PRODUCTION_TRAJECTORIES = ("mdWrap.xtc", "md.xtc", "prod.xtc", "production.xtc")
# Tiny wrap leftovers from failed equilibration are often ~200–400 KB.
_MIN_TRAJ_BYTES = 512_000


def _hpc_dir(sim_dir: str | Path, hpc_dir: Optional[str] = None) -> Path:
    if hpc_dir:
        return Path(hpc_dir)
    return Path(sim_dir) / "hpc"


def _first_trajectory(hpc: Path) -> Optional[Path]:
    for name in _PRODUCTION_TRAJECTORIES:
        path = hpc / name
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def _md_log_finished(hpc: Path) -> bool:
    md_log = hpc / "md.log"
    if not md_log.is_file():
        return False
    try:
        text = md_log.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "Finished mdrun" in text


def _production_health_dir(
    hpc: Path,
    *,
    reuse_hpc: bool = False,
    min_traj_bytes: int = _MIN_TRAJ_BYTES,
) -> Dict[str, Any]:
    """Health of one HPC directory (flat ``hpc/`` or one ``hpc/repXX``)."""
    tpr = hpc / "md.tpr"
    traj = _first_trajectory(hpc)
    out: Dict[str, Any] = {
        "healthy": False,
        "reason": "",
        "hpc_dir": str(hpc),
        "topology": str(tpr) if tpr.is_file() else None,
        "trajectory": str(traj) if traj else None,
        "traj_bytes": int(traj.stat().st_size) if traj else 0,
        "md_log_finished": _md_log_finished(hpc) if hpc.is_dir() else False,
    }
    if not hpc.is_dir():
        out["reason"] = "hpc directory missing"
        return out
    if not tpr.is_file() or tpr.stat().st_size <= 0:
        out["reason"] = "production topology (md.tpr) missing"
        return out
    if traj is None:
        out["reason"] = "production trajectory missing"
        return out
    try:
        size = traj.stat().st_size
    except OSError:
        size = 0
    out["traj_bytes"] = size
    if size < max(1, int(min_traj_bytes)):
        out["reason"] = f"trajectory too small ({size} bytes) for production MD"
        return out
    if reuse_hpc:
        out["healthy"] = True
        out["reason"] = "reuse-hpc topology + trajectory present"
        return out
    if not out["md_log_finished"]:
        out["reason"] = "production md.log did not finish (equilibration/minim leftovers only)"
        return out
    out["healthy"] = True
    out["reason"] = "production MD finished"
    return out


def _nested_hpc_rep_dirs(hpc: Path) -> List[Path]:
    """Existing ``hpc/rep01``-style directories, in replicate order."""
    if not hpc.is_dir():
        return []
    try:
        from src.analysis.replicate_paths import parse_rep_id
    except Exception:
        return []
    nested = [
        p for p in hpc.iterdir()
        if p.is_dir() and parse_rep_id(p.name) is not None
    ]
    return sorted(nested, key=lambda p: parse_rep_id(p.name) or 0)


def production_health(
    sim_dir: str | Path,
    *,
    hpc_dir: Optional[str] = None,
    reuse_hpc: bool = False,
    min_traj_bytes: int = _MIN_TRAJ_BYTES,
) -> Dict[str, Any]:
    """Return health of production MD artifacts under ``sim_dir/hpc``.

    Healthy when ``md.tpr`` and a production trajectory exist, and either
    ``md.log`` reports finished mdrun or ``reuse_hpc`` is set (seeded campaigns).

    Multi-replicate campaigns store those files in ``hpc/rep01`` and
    ``hpc/rep02``, not in the ``hpc/`` root. When nested replicate directories
    exist, every replicate must be healthy. A missing ``hpc/md.tpr`` is then
    not a failure.
    """
    root = Path(sim_dir) if sim_dir else Path()
    hpc = _hpc_dir(root, hpc_dir)
    nested = _nested_hpc_rep_dirs(hpc)
    if not nested:
        return _production_health_dir(
            hpc, reuse_hpc=reuse_hpc, min_traj_bytes=min_traj_bytes
        )

    checks = [
        _production_health_dir(
            rep, reuse_hpc=reuse_hpc, min_traj_bytes=min_traj_bytes
        )
        for rep in nested
    ]
    failed = [c for c in checks if not c.get("healthy")]
    if failed:
        bad = dict(failed[0])
        name = Path(str(bad.get("hpc_dir") or "")).name or "replicate"
        bad["reason"] = f"{name}: {bad.get('reason') or 'production not healthy'}"
        bad["replicates"] = [c.get("hpc_dir") for c in checks]
        return bad

    first = checks[0]
    n = len(checks)
    if reuse_hpc:
        reason = f"reuse-hpc topology + trajectory present ({n} replicates)"
    else:
        reason = f"production MD finished ({n} replicates)"
    return {
        **first,
        "healthy": True,
        "reason": reason,
        "replicates": [c.get("hpc_dir") for c in checks],
    }


def is_reuse_hpc(state: Optional[Dict[str, Any]] = None) -> bool:
    try:
        from agentic.multi_sim_hpc_pool import reuse_hpc_enabled

        return bool(reuse_hpc_enabled(state))
    except Exception:
        return bool(state and (state.get("reuse_hpc") or state.get("skip_hpc_submit")))


def simulation_is_healthy(
    sim_dir: str | Path,
    *,
    state: Optional[Dict[str, Any]] = None,
    hpc_dir: Optional[str] = None,
) -> bool:
    return bool(
        production_health(
            sim_dir,
            hpc_dir=hpc_dir,
            reuse_hpc=is_reuse_hpc(state),
        ).get("healthy")
    )


def hpc_record_failed(rec: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(rec, dict):
        return False
    if str(rec.get("hpc_status") or "").lower() == "failed":
        return True
    if str(rec.get("prep_status") or "").lower() == "failed" and not is_reuse_hpc():
        # Prep failure without reuse-hpc means no production job can run.
        return True
    return False


def progress_record_failed(rec: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(rec, dict):
        return False
    if str(rec.get("status") or "").lower() == "failed":
        return True
    agents = rec.get("agents") or {}
    if str(agents.get("hpc") or "").lower() == "failed":
        return True
    return False


def should_skip_downstream_for_sim(
    state: Dict[str, Any],
    label: str,
    *,
    working_dir: str = "",
) -> Dict[str, Any]:
    """Whether analysis/reporter must be skipped for this simulation.

    Returns ``{skip: bool, reason: str, health: dict}``.
    """
    progress = state.get("multi_sim_progress") or {}
    preg = (progress.get("sims") or {}).get(label) or {}
    hpc_pool = state.get("hpc_pool") or {}
    hrec = (hpc_pool.get("sims") or {}).get(label) or {}
    wd = (
        working_dir
        or preg.get("working_dir")
        or hrec.get("working_dir")
        or ""
    )
    if not wd:
        base = state.get("multi_sim_base_dir") or state.get("working_directory")
        if base and label:
            wd = str(Path(base) / label)

    health = production_health(wd, reuse_hpc=is_reuse_hpc(state)) if wd else {
        "healthy": False,
        "reason": "working directory unknown",
    }

    if progress_record_failed(preg):
        return {
            "skip": True,
            "reason": preg.get("error") or "simulation marked failed",
            "health": health,
            "health_label": "failed",
        }
    if hpc_record_failed(hrec):
        return {
            "skip": True,
            "reason": hrec.get("error") or "HPC / prep failed",
            "health": health,
            "health_label": "failed",
        }
    # After HPC stage is expected, missing production is fatal for downstream.
    needs_hpc = "hpc" in (progress.get("required_agents") or []) or bool(hpc_pool.get("sims"))
    if needs_hpc and wd and not health.get("healthy"):
        return {
            "skip": True,
            "reason": health.get("reason") or "production MD not healthy",
            "health": health,
            "health_label": "failed",
        }
    return {
        "skip": False,
        "reason": "",
        "health": health,
        "health_label": "healthy" if health.get("healthy") else "pending",
    }


def record_fatal_sim_failure(
    state: Dict[str, Any],
    label: str,
    *,
    reason: str,
    stage: str = "hpc",
) -> None:
    """Mark one simulation failed and skip analysis/reporter for it only."""
    from agentic.multi_sim_progress import ensure_multi_sim_progress, mark_agent_status

    progress = ensure_multi_sim_progress(state)
    if not progress:
        return
    rec = (progress.get("sims") or {}).get(label)
    if not rec:
        return
    rec["status"] = "failed"
    rec["health"] = "failed"
    rec["error"] = str(reason or "simulation failed")[:500]
    agents = rec.setdefault("agents", {})
    agents[stage if stage != "preprocess" else "preprocessing"] = "failed"
    if stage in ("hpc", "simsetup", "preprocessing", "preprocess"):
        # Do not run science/reporting on a dead production MD.
        for key in ("analysis", "reporter"):
            if agents.get(key) not in ("done",):
                agents[key] = "skipped"
    progress["sims"][label] = rec
    state["multi_sim_progress"] = progress
    try:
        mark_agent_status(state, label, stage, "failed")
    except Exception:
        logger.debug("mark_agent_status failed for %s", label, exc_info=True)
    logger.warning(
        "Simulation %s marked failed (%s); analysis/reporter skipped for this system only",
        label,
        reason[:160],
    )


def sync_hpc_failures_into_progress(state: Dict[str, Any]) -> List[str]:
    """Propagate hpc_pool failures into multi_sim_progress. Returns labels marked."""
    pool = state.get("hpc_pool") or {}
    marked: List[str] = []
    for label, rec in (pool.get("sims") or {}).items():
        if not hpc_record_failed(rec):
            continue
        rec["health"] = "failed"
        reason = rec.get("error") or f"HPC status={rec.get('hpc_status')}"
        record_fatal_sim_failure(state, label, reason=str(reason), stage="hpc")
        marked.append(label)
    return marked


def healthy_sim_labels(state: Dict[str, Any]) -> List[str]:
    """Labels that may enter analysis / combined science."""
    labels: List[str] = []
    for sp in state.get("sim_prompts") or []:
        label = sp.get("label")
        if not label:
            continue
        decision = should_skip_downstream_for_sim(
            state, label, working_dir=str(sp.get("working_dir") or "")
        )
        if not decision.get("skip"):
            labels.append(label)
    return labels
