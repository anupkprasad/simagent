"""Deterministic science/setup verification before advancing the pipeline.

Not an LLM critic: file + inventory + optional trajectory QC. Failures are
recorded to campaign memory and block stage-done ticks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _sim_dir(state: Dict[str, Any]) -> Optional[Path]:
    wd = state.get("working_directory")
    if not wd:
        return None
    p = Path(wd)
    return p if p.exists() else p


def verify_setup(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check simsetup produced topology + coordinates usable for HPC."""
    issues: List[str] = []
    sim = _sim_dir(state)
    setup_dir = (sim / "simsetup") if sim else None
    coords = state.get("coordinates") or state.get("gro_file")
    topo = state.get("topology") or state.get("topol_file")
    if coords and not Path(str(coords)).is_file():
        issues.append(f"coordinates missing on disk: {coords}")
        coords = None
    if topo and not Path(str(topo)).is_file():
        issues.append(f"topology missing on disk: {topo}")
        topo = None
    if setup_dir and setup_dir.is_dir():
        if not coords:
            for name in ("system.gro", "solvated.gro", "complex.gro", "md.gro"):
                cand = setup_dir / name
                if cand.is_file():
                    coords = str(cand.resolve())
                    break
        if not topo:
            for name in ("topol.top", "topology.top"):
                cand = setup_dir / name
                if cand.is_file():
                    topo = str(cand.resolve())
                    break
            if not topo:
                tops = sorted(setup_dir.glob("*.top"))
                if tops:
                    topo = str(tops[0].resolve())
    if not coords:
        issues.append("no system coordinates (.gro) after simsetup")
    if not topo:
        issues.append("no topology (.top) after simsetup")
    # Holo: if ligand declared, ensure it appears in coordinates text when cheap
    from agentic.campaign.ligand_context import ligand_resnames_from_state

    ligands = ligand_resnames_from_state(state)
    if ligands and coords and Path(str(coords)).is_file():
        try:
            text = Path(str(coords)).read_text(errors="ignore")[:2_000_000]
            missing = [lig for lig in ligands if lig.upper() not in text.upper()]
            if missing and len(text) > 100:
                issues.append(
                    f"declared ligand(s) {missing} not found in coordinates {coords}"
                )
        except OSError:
            pass
    ok = not issues
    return {
        "ok": ok,
        "stage": "simsetup",
        "issues": issues,
        "coordinates": coords,
        "topology": topo,
    }


def _is_combined_analysis_context(state: Dict[str, Any]) -> bool:
    """True when verifying cross-sim / post-combined analysis (not one sim)."""
    phase = str(state.get("multi_sim_phase") or "")
    if phase in {
        "combined_analysis",
        "post_combined",
        "pre_combined",
        "combined_reporter",
    }:
        return True
    if state.get("hitl_view_combined") or state.get("combined_only"):
        return True
    return False


def verify_combined_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check base-level combined overlays/stats — not per-sim hpc/.dat at campaign root."""
    issues: List[str] = []
    try:
        from agentic.multi_sim_paths import resolve_multi_sim_base_dir
        from agentic.multi_sim_progress import (
            _combined_analysis_done_on_disk,
            post_combined_artifacts_ok,
            _expected_combined_labels,
        )

        base = Path(resolve_multi_sim_base_dir(state))
        analysis_dir = base / "analysis"
        expected = _expected_combined_labels(state)
        done = _combined_analysis_done_on_disk(base, state=state)
        family = False
        try:
            from agentic.campaign.spec import spec_from_state

            spec = spec_from_state(state)
            family = bool(spec and spec.family_modular)
        except Exception:
            family = False
        artifacts = post_combined_artifacts_ok(
            analysis_dir,
            expected_labels=expected or None,
            family_modular=family,
        )
        gate = {
            "ok": bool(done or artifacts),
            "combined": True,
            "done_on_disk": done,
            "artifacts_ok": artifacts,
            "expected_labels": expected,
            "analysis_dir": str(analysis_dir),
        }
        if not gate["ok"]:
            issues.append(
                "combined analysis incomplete: need multi-sim overlays/stats "
                f"under {analysis_dir} covering {expected or 'all sims'}"
            )
    except Exception as exc:
        gate = {"ok": False, "combined": True, "reason": str(exc)}
        issues.append(f"combined analysis verify failed: {exc}")
    return {
        "ok": not issues,
        "stage": "analysis",
        "issues": issues,
        "gate": gate,
        "combined": True,
    }


def verify_pre_combined(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check the shared MSA / pocket step, not the later overlay stage."""
    from agentic.multi_sim_paths import resolve_multi_sim_base_dir
    from src.analysis.cross_sim_artifacts import (
        PRE_COMPLETE_NAME,
        cross_sim_dir,
        pre_combined_done_on_disk,
    )

    base = Path(resolve_multi_sim_base_dir(state))
    if pre_combined_done_on_disk(base):
        return {
            "ok": True,
            "stage": "pre_combined",
            "issues": [],
            "gate": {"ok": True, "pre_combined": True, "base": str(base)},
        }
    detail = "pre-combined alignment did not produce cross_sim artifacts"
    marker = cross_sim_dir(base) / PRE_COMPLETE_NAME
    if marker.is_file():
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            recorded = data.get("issues") or data.get("reason") or data.get("error")
            if isinstance(recorded, list) and recorded:
                detail = "; ".join(str(item) for item in recorded[:2])
            elif recorded:
                detail = str(recorded)
        except Exception:
            pass
    return {
        "ok": False,
        "stage": "pre_combined",
        "issues": [detail],
        "gate": {"ok": False, "pre_combined": True, "base": str(base)},
    }


def verify_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check analysis produced real science artifacts (not stub success).

    During combined / post-combined phases, use overlay/stats gates — never
    require per-sim ``hpc/`` or ``rmsd.dat`` at the campaign base directory.
    The pre-combined phase is the shared alignment, not those overlays.
    """
    if str(state.get("multi_sim_phase") or "") == "pre_combined":
        return verify_pre_combined(state)
    if _is_combined_analysis_context(state):
        return verify_combined_analysis(state)

    issues: List[str] = []
    sim = _sim_dir(state)
    if not sim:
        return {
            "ok": False,
            "stage": "analysis",
            "issues": ["missing working_directory"],
        }
    try:
        from agentic.campaign.contracts import per_sim_science_complete

        gate = per_sim_science_complete(str(sim), state=state)
    except Exception as exc:
        gate = {"ok": False, "reason": str(exc), "missing": ["science_complete"]}
    if not gate.get("ok"):
        reason = gate.get("reason") or "; ".join(gate.get("missing") or []) or "incomplete"
        issues.append(f"science incomplete: {reason}")
    # Trajectories must still exist (analysis gate alone can be soft on family)
    hpc = sim / "hpc"
    has_traj = False
    if hpc.is_dir():
        has_traj = any(
            p.is_file()
            for pat in ("*.xtc", "*.trr", "rep*/*.xtc", "rep*/*.trr")
            for p in hpc.glob(pat)
        )
    if not has_traj and not state.get("reuse_hpc"):
        # Allow analysis-only subtasks that set trajectory_path explicitly
        traj = state.get("trajectory_path")
        if not (traj and Path(str(traj)).is_file()):
            issues.append("no HPC trajectory under hpc/ for analysis verify")
    ok = not issues
    return {"ok": ok, "stage": "analysis", "issues": issues, "gate": gate}


def run_verify(state: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """Dispatch verify for a pipeline stage name."""
    key = (stage or "").strip().lower()
    if key in {"setup", "simsetup"}:
        return verify_setup(state)
    if key in {"analysis"}:
        return verify_analysis(state)
    return {"ok": True, "stage": key, "issues": [], "skipped": True}


def apply_verify_to_state(state: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """Run verify, mutate state on failure (errors + memory), return result dict."""
    result = run_verify(state, stage)
    state["last_verify"] = {
        "stage": result.get("stage"),
        "ok": result.get("ok"),
        "issues": list(result.get("issues") or []),
    }
    if result.get("ok"):
        return result
    issues = result.get("issues") or ["verify failed"]
    msg = f"Verify[{result.get('stage')}]: " + "; ".join(issues)
    state.setdefault("errors", []).append(msg)
    logger.error(msg)
    try:
        from agentic.campaign.memory import remember_episode

        remember_episode(
            state=state,
            stage=str(result.get("stage") or stage),
            error=msg,
            tool="verify",
            label=str(state.get("current_sim_label") or ""),
            fix="Fix artifacts then retry stage; do not advance pipeline",
        )
    except Exception:
        logger.debug("verify memory write skipped", exc_info=True)
    # Block soft success markers that cause false stage ticks
    stage_name = str(result.get("stage") or stage)
    if stage_name == "analysis":
        state["analysis_science_ok"] = False
    elif stage_name == "simsetup":
        state["setup_science_ok"] = False
    return result
