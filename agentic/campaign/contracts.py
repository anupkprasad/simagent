"""Science completeness gates — process success is not enough."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agentic.campaign.family_recipe import FAMILY_MODULAR_DIRS
from agentic.campaign.spec import CampaignSpec, spec_from_state

logger = logging.getLogger(__name__)

# Visualisation-only / redundant helpers. Failures must not fail the simulation
# when core metric artifacts (or combined overlays) are already on disk.
NONCRITICAL_ANALYSIS_TOOLS = frozenset(
    {
        "plot_md_data",
        "plot_md_multipanel",
        "plot_multipanel",
        "plot_pca_projection",
        "plot_sasa",
        "plot_dccm_comparison",
        "plot_dccm_difference",
        "plot_combined_data",
        "plot_reference_msa_alignment",
        "plot_global_mapped_alignment",
        # Often redundant after run_combined_analysis wrote *_stats.csv / overlays
        "compute_comparison_table",
        "collect_metric_files",
    }
)

FAMILY_TOOL_OUTPUT_DIRS = {
    "calculate_consensus_torsions": "consensus_dihedrals",
    "calculate_consensus_rmsf_features": "consensus_rmsf",
    "calculate_consensus_dccm_features": "consensus_DCCM",
    "run_independent_dynamics_fel": "consensus_PCA",
    "calculate_consensus_pocket_metrics": "reference_pocket",
}

FAMILY_TOOL_OUTPUT_FILES = {
    "calculate_consensus_pocket_metrics": (
        "reference_pocket_metrics.json",
        "ligand_pocket_distance.csv",
    ),
}


def is_noncritical_analysis_tool(tool_name: str) -> bool:
    name = (tool_name or "").strip()
    if not name:
        return False
    if name in NONCRITICAL_ANALYSIS_TOOLS:
        return True
    return name.startswith("plot_")


def decide_analysis_success(*, artifact_ok: bool, critical_issues: Sequence[str]) -> bool:
    """Pass only when artifacts exist AND no critical calculator failures remain."""
    if critical_issues:
        return False
    return bool(artifact_ok)


def required_metric_files_missing(
    working_dir: str,
    state: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return missing data files for metrics explicitly requested in the goal."""
    try:
        from agentic.planner.planning_guidelines import (
            STANDARD_OUTPUT_ALTERNATES,
            STANDARD_OUTPUT_FILES,
            detect_requested_metrics_union,
        )
    except Exception:
        return []

    texts: List[str] = []
    if state:
        for key in (
            "user_goal_original",
            "user_goal",
            "master_enriched_prompt",
            "enriched_prompt",
            "analysis_instructions",
        ):
            val = state.get(key)
            if val:
                texts.append(str(val))
    # Campaign goal.txt next to the sim folder
    wd = Path(working_dir)
    for cand in (wd.parent / "goal.txt", wd / "goal.txt"):
        if cand.is_file():
            try:
                texts.append(cand.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    metrics = detect_requested_metrics_union(*texts) if texts else None
    if not metrics:
        return []

    search = _search_dirs(str(wd))
    if not search:
        search = [wd / "analysis"]

    # Pocket / ligand metrics are requested for holo systems only.
    ligand_only = {
        "com",
        "contacts",
        "pocket_sasa",
        "residence",
        "pocket_rmsf",
        "ligand_rmsf",
        "ligand_rmsd",
    }
    try:
        from src.analysis.combined_analysis import is_holo_simulation

        # Judge the simulation folder name. A study directory such as
        # pseudo_apo_holo contains "holo" and must not make every apo run
        # look ligand-bound.
        holo = is_holo_simulation(wd.name, wd.name)
    except Exception:
        holo = True

    def _named_present(names: tuple) -> bool:
        return any(
            (d / name).is_file() and (d / name).stat().st_size > 0
            for d in search
            for name in names
            if name
        )

    def _prefix_present(prefix: str, suffixes: tuple) -> bool:
        pref = prefix.lower()
        for d in search:
            if not d.is_dir():
                continue
            for path in d.iterdir():
                if (
                    path.is_file()
                    and path.stat().st_size > 0
                    and path.suffix.lower() in suffixes
                    and path.name.lower().startswith(pref)
                ):
                    return True
        return False

    missing: List[str] = []
    for metric in sorted(metrics):
        if metric in ligand_only and not holo:
            continue
        if metric in {"gyration"}:
            metric = "rg"
        spec = STANDARD_OUTPUT_FILES.get(metric) or {}
        data_name = spec.get("data")
        prefix = spec.get("data_prefix")
        plot_name = spec.get("plot")
        data_alts = (STANDARD_OUTPUT_ALTERNATES.get(metric) or {}).get("data") or ()
        plot_alts = (STANDARD_OUTPUT_ALTERNATES.get(metric) or {}).get("plot") or ()
        if prefix:
            data_ok = _prefix_present(prefix, (".dat", ".csv", ".txt", ".json"))
            plot_ok = (not plot_name) or _prefix_present(prefix, (".png", ".pdf")) or _named_present((plot_name, *plot_alts))
            label = f"{prefix}*"
        elif data_name:
            data_ok = _named_present((data_name, *data_alts))
            plot_ok = (not plot_name) or _named_present((plot_name, *plot_alts))
            label = data_name
        else:
            continue
        if not data_ok:
            missing.append(f"requested metric {metric}: {label}")
        elif not plot_ok:
            missing.append(f"requested metric {metric} plot: {plot_name}")
    return missing


def _dir_has_output(root: Path, name: str) -> bool:
    from agentic.campaign.family_recipe import FAMILY_MODULAR_DIR_ALIASES

    candidates = [name]
    alias = FAMILY_MODULAR_DIR_ALIASES.get(name)
    if alias:
        candidates.append(alias)
    # Also accept legacy consensus_PCA_ref when looking for consensus_PCA
    if name == "consensus_PCA":
        candidates.append("consensus_PCA_ref")
    for cand_name in candidates:
        target = root / cand_name
        if not target.is_dir():
            continue
        # Consume glob results — generator objects themselves are always truthy.
        if any(
            True
            for pat in ("*.json", "*.csv", "*.dat", "*.npz")
            for _ in target.glob(pat)
        ):
            return True
    return False


def _search_dirs(working_dir: str) -> List[Path]:
    analysis = Path(working_dir) / "analysis"
    dirs = [analysis]
    if (analysis / "avg").is_dir():
        dirs.append(analysis / "avg")
    dirs.extend(sorted(analysis.glob("rep*")))
    return dirs


def modular_output_present(analysis_dir: str, name: str) -> bool:
    """True when one modular output dir under analysis/ (or avg/rep*) has files."""
    root = Path(analysis_dir)
    candidates = [root, root / "avg", *sorted(root.glob("rep*"))]
    if root.name != "analysis" and (root / "analysis").is_dir():
        candidates = _search_dirs(str(root))
    return any(_dir_has_output(d, name) for d in candidates)


def family_tool_already_done(analysis_dir: str, tool_name: str) -> bool:
    mapped = FAMILY_TOOL_OUTPUT_DIRS.get(tool_name or "")
    if mapped and modular_output_present(analysis_dir, mapped):
        return True
    files = FAMILY_TOOL_OUTPUT_FILES.get(tool_name or "")
    if files:
        root = Path(analysis_dir)
        candidates = [root, root / "avg", *sorted(root.glob("rep*"))]
        if root.name != "analysis" and (root / "analysis").is_dir():
            candidates = _search_dirs(str(root))
        for d in candidates:
            if any((d / name).is_file() and (d / name).stat().st_size > 0 for name in files):
                return True
    return False


def per_sim_modular_complete(working_dir: str, modular_dirs: Sequence[str]) -> bool:
    search = _search_dirs(working_dir)
    if not search:
        return False
    needed = list(modular_dirs) or list(FAMILY_MODULAR_DIRS)
    for d in search:
        if all(_dir_has_output(d, name) for name in needed):
            return True
    return all(any(_dir_has_output(d, name) for d in search) for name in needed)


def per_sim_science_complete(
    working_dir: str,
    spec: Optional[CampaignSpec] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return ``{ok, missing, reason}`` for one simulation directory."""
    spec = spec or spec_from_state(state)
    wd = Path(working_dir)
    if not wd.is_dir():
        return {"ok": False, "missing": ["working_dir"], "reason": "missing working_dir"}

    if spec is None or not spec.family_modular:
        # Align with per_sim_analysis_done_on_disk: summary alone is not science.
        try:
            from agentic.multi_sim_progress import per_sim_analysis_done_on_disk

            ok = per_sim_analysis_done_on_disk(str(wd), state=state)
        except TypeError:
            try:
                from agentic.multi_sim_progress import per_sim_analysis_done_on_disk

                ok = per_sim_analysis_done_on_disk(str(wd))
            except Exception:
                ok = (wd / "analysis" / "analysis_summary.jsonl").is_file()
        except Exception:
            summary = wd / "analysis" / "analysis_summary.jsonl"
            ok = summary.is_file()
        missing: List[str] = []
        if not ok:
            missing.append("analysis science artifacts (rmsd/rmsf/pca/… + summary)")
        # Intent-aware: when the goal names specific metrics, require each data file.
        intent_missing = required_metric_files_missing(str(wd), state=state)
        if intent_missing:
            missing.extend(intent_missing)
            ok = False
        # Also require HPC trajectory so analysis cannot be "complete" without MD.
        hpc = wd / "hpc"
        has_traj = False
        if hpc.is_dir():
            has_traj = any(
                p.is_file()
                for pat in ("*.xtc", "*.trr", "rep*/*.xtc", "rep*/*.trr")
                for p in hpc.glob(pat)
            )
        if not has_traj:
            missing.append("hpc trajectory (.xtc/.trr)")
            ok = False
        return {
            "ok": bool(ok) and not missing,
            "missing": missing,
            "reason": "" if (ok and not missing) else "; ".join(missing),
        }

    missing: List[str] = []
    analysis = wd / "analysis"
    if not per_sim_modular_complete(str(wd), spec.contract.modular_dirs):
        missing.append("consensus_* modular dirs")
    # Required calculations (general MD): each must leave its artifact contract.
    for tool in list(spec.pin_tools or []):
        if tool not in FAMILY_TOOL_OUTPUT_DIRS and tool not in FAMILY_TOOL_OUTPUT_FILES:
            continue
        if family_tool_already_done(str(analysis), tool):
            continue
        if family_tool_already_done(str(wd), tool):
            continue
        missing.append(tool)
    summary = analysis / "analysis_summary.jsonl"
    if not summary.is_file() and not any(wd.glob("analysis/rep*/analysis_summary.jsonl")):
        missing.append("analysis_summary.jsonl")
    ok = not missing
    return {
        "ok": ok,
        "missing": missing,
        "reason": "" if ok else "; ".join(missing),
    }


def _finite(value: str) -> bool:
    raw = (value or "").strip()
    if not raw or raw.lower() in {"nan", "none", "null", "inf", "-inf"}:
        return False
    try:
        float(raw)
        return True
    except ValueError:
        return False


def clustering_table_usable(
    csv_path: str,
    *,
    min_rows: int = 2,
    min_cols: int = 2,
) -> Dict[str, Any]:
    """True when a dendrogram + heatmap can be drawn from whatever columns exist.

    General-purpose: users will not always compute the family gold set. Two
    simulations and two finite numeric columns are enough to plot.
    """
    path = Path(csv_path)
    skip = {"label", "sim_directory", "n_features_present"}
    if not path.is_file():
        return {
            "ok": False,
            "n_rows": 0,
            "n_feature_cols": 0,
            "reason": f"missing {path}",
        }
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"ok": False, "n_rows": 0, "n_feature_cols": 0, "reason": "empty feature table"}
    cols = [c for c in (rows[0].keys()) if c and c not in skip]
    usable_cols = []
    for col in cols:
        n_fin = sum(1 for row in rows if _finite(row.get(col, "")))
        if n_fin >= min_rows:
            usable_cols.append(col)
    n_rows = len(rows)
    ok = n_rows >= min_rows and len(usable_cols) >= min_cols
    reason = ""
    if not ok:
        reason = (
            f"need ≥{min_rows} rows and ≥{min_cols} numeric columns "
            f"(have {n_rows} rows, {len(usable_cols)} usable)"
        )
    return {
        "ok": ok,
        "n_rows": n_rows,
        "n_feature_cols": len(usable_cols),
        "usable_columns": usable_cols,
        "reason": reason,
    }


def family_feature_matrix_ready(
    csv_path: str,
    spec: Optional[CampaignSpec] = None,
    *,
    required_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """True when enough rows have the required family columns filled."""
    path = Path(csv_path)
    required = list(
        required_columns
        or (spec.analysis_recipe.required_feature_columns if spec else [])
    )
    min_frac = spec.contract.min_complete_fraction if spec else 0.85
    allow_partial = bool(spec and spec.allow_partial_combined)
    if not path.is_file():
        return {
            "ok": False,
            "n_rows": 0,
            "n_complete": 0,
            "fraction": 0.0,
            "reason": f"missing {path}",
            "allow_partial": allow_partial,
        }

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {
            "ok": False,
            "n_rows": 0,
            "n_complete": 0,
            "fraction": 0.0,
            "reason": "empty feature table",
            "allow_partial": allow_partial,
        }

    header = set(rows[0].keys()) if rows else set()
    present_required = [c for c in required if c in header]
    n_complete = 0
    for row in rows:
        if required and all(_finite(row.get(c, "")) for c in required):
            n_complete += 1
    n_rows = len(rows)
    fraction = n_complete / n_rows if n_rows else 0.0
    ok = fraction >= min_frac and n_complete >= 2
    if allow_partial and n_complete >= 2:
        ok = True
    reason = ""
    if not ok:
        reason = (
            f"feature matrix incomplete: {n_complete}/{n_rows} rows have all "
            f"{len(required)} required columns "
            f"({len(present_required)} present in table; need ≥{min_frac:.0%})"
        )
    return {
        "ok": ok,
        "n_rows": n_rows,
        "n_complete": n_complete,
        "fraction": fraction,
        "reason": reason,
        "allow_partial": allow_partial,
        "required_columns": present_required,
    }


def campaign_science_report(
    base_dir: str,
    spec: Optional[CampaignSpec] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Campaign-level science completeness for run_summary / combined gate."""
    spec = spec or spec_from_state(state)
    base = Path(base_dir)
    labels = list((spec.labels if spec else []) or [])
    if not labels:
        labels = [
            p.name
            for p in sorted(base.iterdir())
            if p.is_dir() and (p / "analysis").is_dir()
        ]
    per_sim: List[Dict[str, Any]] = []
    n_ok = 0
    for label in labels:
        wd = base / label
        rec = per_sim_science_complete(str(wd), spec=spec)
        rec["label"] = label
        per_sim.append(rec)
        if rec.get("ok"):
            n_ok += 1
    feat = base / "analysis" / "classification_features.csv"
    matrix = family_feature_matrix_ready(str(feat), spec) if spec and spec.family_modular else {
        "ok": True,
        "n_rows": 0,
        "n_complete": 0,
        "fraction": 1.0,
        "reason": "",
    }
    n_total = len(labels) or 1
    ok = n_ok == len(labels) and bool(matrix.get("ok"))
    if spec and spec.allow_partial_combined:
        ok = n_ok > 0
    return {
        "ok": ok,
        "family_modular": bool(spec.family_modular) if spec else False,
        "n_science_complete": n_ok,
        "n_systems": len(labels),
        "science_fraction": n_ok / n_total,
        "incomplete_labels": [r["label"] for r in per_sim if not r.get("ok")],
        "matrix": matrix,
        "per_sim": per_sim,
    }
