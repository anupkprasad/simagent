"""
Goal-aware figure and analysis selection for MD simulation reports.

Selects the most relevant plots and analysis sections based on the user's
objective, detected metrics, available outputs, and report type — avoiding
"include everything" report bloat at scale (35–50+ simulations).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Analysis jsonl types → planning metric groups (generic MD + binding-site).
ANALYSIS_TYPE_METRICS: Dict[str, str] = {
    "RMSD": "rmsd",
    "RMSF": "rmsf",
    "RadiusOfGyration": "rg",
    "Gyration": "rg",
    "SASA": "sasa",
    "Energy": "energy",
    "DCCM": "dccm",
    "DSSP": "dssp",
    "DSSP_SecondaryStructure": "dssp",
    "SecondaryStructure": "dssp",
    "PCA": "pca",
    "FreeEnergyLandscape": "fel",
    "FELFeatures": "fel",
    "FELBasinStructures": "fel",
    "Ligand_Pocket_COM_Distance": "com",
    "ProteinLigandContacts": "contacts",
    "LigandResidence": "residence",
    "PocketSASA": "pocket_sasa",
    "Pocket_RMSF": "pocket_rmsf",
    "Ligand_RMSF": "ligand_rmsf",
    "ClassificationClustering": "classification",
    "ClassificationFeatures": "classification",
}

# Filename / path keywords → metric group (for overlay plots on disk).
PLOT_PATH_METRICS: Tuple[Tuple[str, str], ...] = (
    ("classification_clusters", "classification"),
    ("classification_dendrogram", "classification"),
    ("classification_phylo", "classification"),
    ("classification_cluster", "classification"),
    ("pocket_sasa_by_cluster", "classification"),
    ("com_distance_by_cluster", "classification"),
    ("contacts_by_cluster", "classification"),
    ("ligand_residence_by_cluster", "classification"),
    ("pocket_rmsf_by_cluster", "classification"),
    ("ligand_rmsf_by_cluster", "classification"),
    ("ligand_pocket_distance", "com"),
    ("com_distance", "com"),
    ("pocket_distance", "com"),
    ("protein_ligand_contacts", "contacts"),
    ("ligand_residence", "residence"),
    ("pocket_rmsf", "pocket_rmsf"),
    ("ligand_rmsf", "ligand_rmsf"),
    ("pocket_sasa", "pocket_sasa"),
    ("fel_basins", "fel"),
    ("fel_pc", "fel"),
    ("fel_", "fel"),
    ("pca", "pca"),
    ("dccm", "dccm"),
    ("dssp", "dssp"),
    ("rmsf_segment", "rmsf"),
    ("rmsf_apo_holo", "rmsf"),
    ("rmsf_overlay", "rmsf"),
    ("rmsf", "rmsf"),
    ("rmsd", "rmsd"),
    ("gyration", "rg"),
    ("rg_overlay", "rg"),
    ("sasa", "sasa"),
    ("energy", "energy"),
    ("hbond", "hbond"),
)

# Universal backbone metrics when the user goal is broad / unspecified.
DEFAULT_REPORT_METRICS: FrozenSet[str] = frozenset({"rmsd", "rmsf", "rg"})

# Binding-site bundle when holo/ligand language appears in the goal.
BINDING_REPORT_METRICS: FrozenSet[str] = frozenset({
    "com", "contacts", "pocket_sasa", "residence", "pocket_rmsf", "ligand_rmsf", "fel",
})

COMPARATIVE_PANEL_METRICS: Dict[str, str] = {
    "A": "rmsd",
    "B": "rmsf",
    "C": "rg",
    "D": "com",
    "E": "dccm",
    "F": "summary",
    "G": "dssp",
    "H": "classification",
}

# When a cluster-level plot exists, these global/per-sim plot name fragments are redundant.
CLUSTER_SUPERSEDES_GLOBAL: Tuple[Tuple[str, str], ...] = (
    ("com_distance_by_cluster", "com_distance_overlay"),
    ("com_distance_by_cluster", "ligand_pocket_distance"),
    ("pocket_rmsf_by_cluster", "pocket_rmsf_overlay"),
    ("ligand_rmsf_by_cluster", "ligand_rmsf_overlay"),
    ("pocket_sasa_by_cluster", "pocket_sasa_overlay"),
    ("contacts_by_cluster", "protein_ligand_contacts_overlay"),
    ("ligand_residence_by_cluster", "ligand_residence_overlay"),
)

CLASSIFICATION_SUMMARY_ORDER: Tuple[str, ...] = (
    "ref_fel_pock_clusters_pca",
    "ref_fel_pock_dendrogram_heatmap",
    "ref_fel_pock_phylo",
    "classification_mds_map",
    "classification_phylo_tree",
    "classification_clusters_pca",
    "classification_dendrogram",
    "ref_fel_dendrogram",
    "ref_fel_phylo",
)

CLUSTER_TRAJECTORY_ORDER: Tuple[str, ...] = (
    "ref_fel_pock_com_distance_by_cluster",
    "ref_fel_pock_hbonds_by_cluster",
    "ref_fel_pock_sasa_by_cluster",
    "ref_fel_pock_ligand_residence_by_cluster",
    "ref_fel_pock_ligand_axis_angle_by_cluster",
    "com_distance_by_cluster",
    "contacts_by_cluster",
    "pocket_sasa_by_cluster",
    "ligand_residence_by_cluster",
)

CLUSTER_RMSF_ORDER: Tuple[str, ...] = (
    "ref_fel_pock_rmsf_by_cluster",
    "pocket_rmsf_by_cluster",
    "ligand_rmsf_by_cluster",
)

# Prefer one canonical plot per analysis type when multiple images exist.
PREFERRED_PLOT_NAMES: Dict[str, Tuple[str, ...]] = {
    "fel": ("fel_basins.png",),
    "pca": ("pca_projection.png", "pca.png"),
    "com": ("ligand_pocket_distance.png", "com_distance.png"),
    "contacts": ("protein_ligand_contacts.png",),
    "residence": ("ligand_residence.png",),
    "pocket_sasa": ("pocket_sasa.png",),
    "pocket_rmsf": ("pocket_rmsf.png",),
    "ligand_rmsf": ("ligand_rmsf.png",),
    "classification": (
        "classification_clusters_pca.png",
        "classification_phylo_tree.png",
        "classification_dendrogram.png",
    ),
}


@dataclass
class ReportFigurePolicy:
    """Limits and behaviour for report figure inclusion."""

    report_type: str = "comprehensive"
    max_figures_per_section: int = 5
    max_figures_per_sim_combined: int = 2
    max_combined_highlight_figures: int = 24
    max_total_per_sim_figures: int = 12
    include_visualizations: bool = True

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        report_type: str = "comprehensive",
        include_visualizations: bool = True,
    ) -> "ReportFigurePolicy":
        report_cfg = (config or {}).get("report") or {}
        rt = (report_type or report_cfg.get("default_type") or "comprehensive").lower()
        max_sec = int(report_cfg.get("max_figures_per_section") or 5)
        if rt == "executive":
            max_sec = min(max_sec, 2)
        # Scale combined caps: ~1 highlight per 2 sims, hard cap 24
        return cls(
            report_type=rt,
            max_figures_per_section=max_sec,
            max_figures_per_sim_combined=1 if rt == "executive" else 2,
            max_combined_highlight_figures=12 if rt == "executive" else 24,
            max_total_per_sim_figures=6 if rt == "executive" else 12,
            include_visualizations=include_visualizations,
        )


def _normalize_goal_text(*texts: Optional[str]) -> str:
    parts = []
    for t in texts:
        if t is None:
            continue
        if isinstance(t, (list, tuple)):
            parts.extend(str(x) for x in t if x)
            continue
        if str(t).strip():
            parts.append(str(t).strip())
    return " ".join(parts)


def resolve_relevant_metrics(
    user_goal: str = "",
    report_focus: Any = "",
    enriched_prompt: str = "",
) -> FrozenSet[str]:
    """
    Determine which metric groups matter for this report.

    Uses planning guideline detection, report_focus keywords, and sensible
    defaults for broad MD goals.
    """
    combined = _normalize_goal_text(user_goal, report_focus, enriched_prompt)
    focus_lower = (report_focus or "").lower()

    try:
        from agentic.planner.planning_guidelines import (
            detect_classification_requested,
            detect_requested_metrics_union,
        )
        detected = detect_requested_metrics_union(combined)
        if detect_classification_requested(combined):
            base = set(detected or DEFAULT_REPORT_METRICS)
            base.add("classification")
            return frozenset(base)
    except ImportError:
        detected = None

    metrics: Set[str] = set(detected) if detected else set(DEFAULT_REPORT_METRICS)

    # Binding-site language → include holo/ligand metrics.
    lower = combined.lower()
    if any(
        kw in lower
        for kw in (
            "ligand", "atp", "pocket", "binding site", "binding-site",
            "holo", "protein-ligand", "protein–ligand", "nucleotide",
        )
    ):
        metrics |= set(BINDING_REPORT_METRICS)

    # Explicit DCCM / cross-correlation language — keep even when the goal is
    # classified as "broad dynamics" (detect_requested_metrics → None).
    if re.search(r"\bdccm\b|cross[-\s]?correlation|correlated motion", lower):
        metrics.add("dccm")

    # Explicit report_focus overrides (from LLM reporter plan).
    focus_map = {
        "stability": {"rmsd", "rg", "energy"},
        "flexibility": {"rmsf", "pocket_rmsf", "ligand_rmsf", "dccm"},
        "binding": {"com", "contacts", "residence", "pocket_sasa"},
        "conformation": {"pca", "fel", "dssp"},
        "classification": {"classification"},
        "cluster": {"classification"},
        "dynamics": {"rmsd", "rmsf", "rg", "dccm"},
        "surface": {"sasa", "pocket_sasa"},
        "energy": {"energy", "fel"},
    }
    for key, groups in focus_map.items():
        if key in focus_lower:
            metrics |= groups

    if not metrics:
        metrics = set(DEFAULT_REPORT_METRICS)

    return frozenset(metrics)


@dataclass
class ReportNarrative:
    """Goal-derived story arc for report figure selection."""

    primary_theme: str = "general"
    relevant_metrics: FrozenSet[str] = field(default_factory=lambda: DEFAULT_REPORT_METRICS)
    classification_primary: bool = False
    prefer_cluster_views: bool = False
    suppress_global_for: FrozenSet[str] = field(default_factory=frozenset)
    max_fel_per_sim: int = 1
    max_pdb_structures: int = 16


def resolve_report_narrative(
    user_goal: str = "",
    report_focus: Any = "",
    enriched_prompt: str = "",
    n_simulations: int = 0,
    policy: Optional[ReportFigurePolicy] = None,
) -> ReportNarrative:
    """
    Infer the report story from the user goal — generic across MD workflows.

    Themes: classification, binding, apo_holo, conformation, stability, general.
    """
    policy = policy or ReportFigurePolicy()
    combined = _normalize_goal_text(user_goal, report_focus, enriched_prompt)
    lower = combined.lower()
    relevant = resolve_relevant_metrics(user_goal, report_focus, enriched_prompt)

    classification_primary = False
    try:
        from agentic.planner.planning_guidelines import detect_classification_requested
        classification_primary = detect_classification_requested(combined)
    except ImportError:
        classification_primary = "classification" in relevant

    primary_theme = "general"
    if classification_primary:
        primary_theme = "classification"
    elif any(k in lower for k in ("apo vs holo", "apo/holo", "apo and holo", "protein only")):
        primary_theme = "apo_holo"
    elif any(k in lower for k in ("binding", "ligand", "atp", "pocket", "nucleotide")):
        primary_theme = "binding"
    elif any(k in lower for k in ("fel", "free-energy", "free energy", "conformational", "pca")):
        primary_theme = "conformation"

    prefer_cluster = classification_primary or "by cluster" in lower or "cluster-wise" in lower
    suppress: Set[str] = set()
    if prefer_cluster:
        suppress = {"com", "contacts", "pocket_sasa", "residence", "pocket_rmsf", "ligand_rmsf"}

    n_sims = max(n_simulations, 1)
    if classification_primary:
        max_fel = 1
        max_pdb = min(12, max(4, n_sims))
    elif policy.report_type == "executive":
        max_fel = 1
        max_pdb = min(8, n_sims)
    else:
        max_fel = 1 if n_sims > 6 else 2
        max_pdb = min(policy.max_combined_highlight_figures, max(n_sims * 2, 8))

    return ReportNarrative(
        primary_theme=primary_theme,
        relevant_metrics=relevant,
        classification_primary=classification_primary,
        prefer_cluster_views=prefer_cluster,
        suppress_global_for=frozenset(suppress),
        max_fel_per_sim=max_fel,
        max_pdb_structures=max_pdb,
    )


def _plot_name(path: str) -> str:
    return Path(path).name.lower()


def _plot_matches_fragment(path: str, fragment: str) -> bool:
    return fragment.lower() in _plot_name(path)


def _any_plot_matches(overlay_plots: Sequence[str], fragment: str) -> bool:
    return any(_plot_matches_fragment(p, fragment) for p in overlay_plots)


def dedupe_redundant_overlay_plots(
    overlay_plots: Sequence[str],
    narrative: Optional[ReportNarrative] = None,
) -> List[str]:
    """
    Drop global overlays superseded by cluster-level plots (same metric, richer view).
    """
    narrative = narrative or ReportNarrative()
    if not narrative.prefer_cluster_views:
        return list(overlay_plots)

    drop_fragments: Set[str] = set()
    for cluster_key, global_key in CLUSTER_SUPERSEDES_GLOBAL:
        if _any_plot_matches(overlay_plots, cluster_key):
            drop_fragments.add(global_key)

    kept: List[str] = []
    seen: Set[str] = set()
    for path in overlay_plots:
        p = str(path)
        if p in seen or not Path(p).is_file():
            continue
        name = _plot_name(p)
        if any(frag in name for frag in drop_fragments):
            continue
        if narrative.classification_primary and any(
            frag in name for frag in ("com_distance_overlay", "pocket_rmsf_overlay", "ligand_rmsf_overlay")
        ) and _any_plot_matches(overlay_plots, "by_cluster"):
            continue
        seen.add(p)
        kept.append(p)
    return kept


def partition_combined_overlay_plots(
    overlay_plots: Sequence[str],
) -> Dict[str, List[str]]:
    """Group cross-simulation plots into narrative sections for the combined report."""
    buckets: Dict[str, List[str]] = {
        "classification_summary": [],
        "cluster_trajectories": [],
        "cluster_rmsf": [],
        "global_overlays": [],
        "other": [],
    }
    seen: Set[str] = set()

    def _assign(bucket: str, path: str) -> None:
        p = str(path)
        if p in seen or not Path(p).is_file():
            return
        seen.add(p)
        buckets[bucket].append(p)

    def _order_bucket(bucket_key: str, order: Tuple[str, ...]) -> None:
        items = buckets[bucket_key]
        if bucket_key == "classification_summary":
            panel_names = {
                Path(p).stem
                for p in items
                if "dendrogram_heatmap" in Path(p).stem
            }
            if panel_names:
                items = [
                    p
                    for p in items
                    if Path(p).stem not in {
                        "ref_fel_pock_dendrogram",
                        "ref_fel_pock_features_heatmap",
                    }
                ]
        ranked: List[Tuple[int, str]] = []
        for p in items:
            name = _plot_name(p)
            rank = next((i for i, key in enumerate(order) if key in name), len(order))
            ranked.append((rank, p))
        ranked.sort(key=lambda x: (x[0], x[1]))
        buckets[bucket_key] = [p for _, p in ranked]

    for path in overlay_plots:
        name = _plot_name(path)
        if name.startswith(("classification_", "ref_fel_pock_")):
            _assign("classification_summary", path)
        elif name.startswith("ref_fel_") and "by_cluster" not in name:
            _assign("classification_summary", path)
        elif "by_cluster" in name:
            if "rmsf" in name:
                _assign("cluster_rmsf", path)
            else:
                _assign("cluster_trajectories", path)
        elif "overlay" in name or "comparison" in name:
            _assign("global_overlays", path)
        else:
            _assign("other", path)

    _order_bucket("classification_summary", CLASSIFICATION_SUMMARY_ORDER)
    _order_bucket("cluster_trajectories", CLUSTER_TRAJECTORY_ORDER)
    _order_bucket("cluster_rmsf", CLUSTER_RMSF_ORDER)
    return buckets


def collect_paths_for_combined_sections(
    overlay_plots: Sequence[str],
    narrative: Optional[ReportNarrative] = None,
) -> Set[str]:
    """All plot paths that will appear in dedicated combined-report sections."""
    narrative = narrative or ReportNarrative()
    paths: Set[str] = set()
    for p in overlay_plots:
        if Path(p).is_file():
            paths.add(str(Path(p).resolve()))

    if narrative.classification_primary or narrative.prefer_cluster_views:
        parts = partition_combined_overlay_plots(overlay_plots)
        for key in ("classification_summary", "cluster_trajectories", "cluster_rmsf"):
            for p in parts.get(key) or []:
                if Path(p).is_file():
                    paths.add(str(Path(p).resolve()))
    return paths


def metric_for_analysis_type(analysis_type: str) -> Optional[str]:
    if not analysis_type:
        return None
    if analysis_type in ANALYSIS_TYPE_METRICS:
        return ANALYSIS_TYPE_METRICS[analysis_type]
    lower = analysis_type.lower()
    for atype, metric in ANALYSIS_TYPE_METRICS.items():
        if atype.lower() in lower or lower in atype.lower():
            return metric
    return None


def metric_for_plot_path(path: str) -> Optional[str]:
    name = Path(path).name.lower()
    for keyword, metric in PLOT_PATH_METRICS:
        if keyword in name:
            return metric
    return None


# Infrastructure / bookkeeping entries — never promote into the HTML report body.
_SKIP_ANALYSIS_TYPES = frozenset({
    "pdb_input_validation",
    "trajectorybatch",
    "trajectory_batch",
    "trajectory_qc",
    "inventory",
    "wrap",
    "pbc_wrap",
})


def score_analysis_entry(
    analysis_type: str,
    relevant_metrics: FrozenSet[str],
    has_statistics: bool = False,
) -> float:
    """Higher score → more important to include in the report."""
    atype_key = (analysis_type or "").strip().lower().replace(" ", "_")
    if atype_key in _SKIP_ANALYSIS_TYPES or atype_key.startswith("pdb_"):
        return 0.0
    metric = metric_for_analysis_type(analysis_type)
    if metric and metric in relevant_metrics:
        score = 10.0
        if metric in DEFAULT_REPORT_METRICS:
            score += 2.0
        if metric in {"classification", "fel", "com", "contacts"}:
            score += 1.0
        if has_statistics:
            score += 0.5
        return score
    if metric:
        return 1.0
    # Unknown types with only stats — omit (avoid report bloat)
    return 0.0


def score_plot_path(path: str, relevant_metrics: FrozenSet[str]) -> float:
    metric = metric_for_plot_path(path)
    if not metric:
        return 0.5
    if metric not in relevant_metrics:
        return 0.25
    score = 8.0
    name = Path(path).name.lower()
    for pref_tuple in PREFERRED_PLOT_NAMES.values():
        for i, pref in enumerate(pref_tuple):
            if name == pref:
                score += 3.0 - i
    if "overlay" in name or "comparison" in name or "by_cluster" in name:
        score += 2.0
    if "classification" in name:
        score += 2.0
    return score


def _dedupe_fel_analysis_entries(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prefer FELFeatures (annotated basins) over plain FreeEnergyLandscape."""
    types = {e.get("analysis_type") for e in entries}
    if "FELFeatures" in types or "FELBasinStructures" in types:
        return [
            e for e in entries
            if e.get("analysis_type") != "FreeEnergyLandscape"
        ]
    return list(entries)


def select_analysis_entries(
    entries: Sequence[Dict[str, Any]],
    user_goal: str = "",
    report_focus: str = "",
    enriched_prompt: str = "",
    policy: Optional[ReportFigurePolicy] = None,
) -> List[Dict[str, Any]]:
    """Rank and filter analysis entries for a per-simulation report."""
    policy = policy or ReportFigurePolicy()
    relevant = resolve_relevant_metrics(user_goal, report_focus, enriched_prompt)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in entries:
        atype = entry.get("analysis_type", "Unknown")
        stats = entry.get("statistics") or {}
        has_stats = isinstance(stats, dict) and bool(stats)
        score = score_analysis_entry(atype, relevant, has_statistics=has_stats)
        if score <= 0:
            continue
        scored.append((score, entry))

    scored.sort(key=lambda x: (-x[0], x[1].get("analysis_type", "")))
    max_sections = policy.max_total_per_sim_figures
    if policy.report_type == "executive":
        max_sections = min(max_sections, 6)
    selected = [e for _, e in scored[:max_sections]]
    return _dedupe_fel_analysis_entries(selected)


def select_images_from_entry(
    image_files: Dict[str, str],
    user_goal: str = "",
    report_focus: str = "",
    enriched_prompt: str = "",
    analysis_type: str = "",
    policy: Optional[ReportFigurePolicy] = None,
) -> Dict[str, str]:
    """Pick the best subset of images for one analysis entry."""
    policy = policy or ReportFigurePolicy()
    if not policy.include_visualizations or not image_files:
        return {}

    relevant = resolve_relevant_metrics(user_goal, report_focus, enriched_prompt)
    entry_metric = metric_for_analysis_type(analysis_type)

    scored: List[Tuple[float, str, str]] = []
    for key, path in image_files.items():
        metric = metric_for_plot_path(path) or entry_metric
        base = score_plot_path(path, relevant)
        if metric and metric not in relevant and base < 5.0:
            continue
        scored.append((base, key, path))

    scored.sort(key=lambda x: (-x[0], x[2]))
    max_n = policy.max_figures_per_section
    if policy.report_type == "executive":
        max_n = min(max_n, 2)
    # Prefer one primary plot per metric section (drop multipanel when singles exist)
    has_primary = any(
        "multipanel" not in Path(p).name.lower()
        and "combined_metrics" not in Path(p).name.lower()
        for _, _, p in scored
    )
    if has_primary:
        scored = [
            item for item in scored
            if "multipanel" not in Path(item[2]).name.lower()
            and "combined_metrics" not in Path(item[2]).name.lower()
        ]
    # One figure per analysis section by default for focused MD goals
    if len(relevant) <= 4:
        max_n = min(max_n, 1)

    # FELFeatures: only annotated basin map (skip raw fel_pc1_pc2 landscape)
    if analysis_type in {"FELFeatures", "FELBasinStructures"}:
        scored = [
            item for item in scored
            if Path(item[2]).name != "fel_pc1_pc2.png"
        ]
        basin = [item for item in scored if Path(item[2]).name == "fel_basins.png"]
        if basin:
            scored = basin + [item for item in scored if item not in basin]

    selected: Dict[str, str] = {}
    for _, key, path in scored[:max_n]:
        selected[key] = path
    return selected


def curate_overlay_plots(
    overlay_plots: Sequence[str],
    user_goal: str = "",
    report_focus: str = "",
    enriched_prompt: str = "",
    policy: Optional[ReportFigurePolicy] = None,
    max_plots: Optional[int] = None,
    narrative: Optional[ReportNarrative] = None,
) -> List[str]:
    """
    Rank and cap combined overlay / cross-simulation plot paths.

    Applies goal-aware deduplication (cluster views supersede global overlays)
    and always prioritizes classification/clustering summary plots when present.
    """
    policy = policy or ReportFigurePolicy()
    narrative = narrative or resolve_report_narrative(
        user_goal, report_focus, enriched_prompt, policy=policy,
    )
    relevant = narrative.relevant_metrics
    deduped = dedupe_redundant_overlay_plots(overlay_plots, narrative)

    scored: List[Tuple[float, str]] = []
    seen: Set[str] = set()
    for path in deduped:
        p = str(path)
        if p in seen or not Path(p).is_file():
            continue
        seen.add(p)
        metric = metric_for_plot_path(p)
        if metric == "energy" and "energy" not in relevant:
            continue
        if metric and metric not in relevant and metric not in {"summary"}:
            if "overlay" not in p.lower() and "comparison" not in p.lower():
                if metric not in {"classification"}:
                    continue
        score = score_plot_path(p, relevant)
        if narrative.classification_primary:
            name = _plot_name(p)
            if name.startswith("classification_") or "by_cluster" in name:
                score += 5.0
            elif metric in narrative.suppress_global_for and "by_cluster" not in name:
                score -= 4.0
        scored.append((score, p))

    scored.sort(key=lambda x: (-x[0], x[1]))
    cap = max_plots
    if cap is None:
        if narrative.classification_primary:
            cap = 20 if policy.report_type == "executive" else 32
        else:
            cap = 16 if policy.report_type == "executive" else 28
    return [p for _, p in scored[:cap]]


def panel_is_relevant(
    panel_id: str,
    user_goal: str = "",
    report_focus: str = "",
    enriched_prompt: str = "",
    has_data: bool = False,
    narrative: Optional[ReportNarrative] = None,
) -> bool:
    """Whether a comparative dynamics panel should be rendered."""
    narrative = narrative or resolve_report_narrative(
        user_goal, report_focus, enriched_prompt,
    )
    metric = COMPARATIVE_PANEL_METRICS.get(panel_id, "")
    relevant = narrative.relevant_metrics

    if panel_id == "F":
        return has_data

    if panel_id == "H":
        return has_data and (
            narrative.classification_primary or "classification" in relevant
        )

    # When cluster views carry binding metrics, skip redundant global COM panel.
    if panel_id == "D" and narrative.prefer_cluster_views and "com" in narrative.suppress_global_for:
        return False

    if metric in relevant:
        return True

    if not user_goal.strip() and has_data and metric in DEFAULT_REPORT_METRICS:
        return True

    return has_data and metric in relevant


def select_per_sim_highlight_figures(
    sims_summary: Sequence[Dict[str, Any]],
    sim_dirs: Sequence[str],
    collect_images_fn,
    user_goal: str = "",
    report_focus: str = "",
    enriched_prompt: str = "",
    exclude_paths: Optional[Set[str]] = None,
    policy: Optional[ReportFigurePolicy] = None,
    narrative: Optional[ReportNarrative] = None,
) -> List[Dict[str, str]]:
    """
    Select capped per-simulation highlight figures for combined reports.

    Skips metrics already shown in cluster/global sections; caps FEL to the
    most informative basin map per simulation when classification is primary.
    """
    policy = policy or ReportFigurePolicy()
    if not policy.include_visualizations:
        return []

    narrative = narrative or resolve_report_narrative(
        user_goal, report_focus, enriched_prompt,
        n_simulations=len(sims_summary),
        policy=policy,
    )
    relevant = narrative.relevant_metrics
    exclude = {str(p) for p in (exclude_paths or set())}
    candidates: List[Tuple[float, Dict[str, str]]] = []

    if narrative.classification_primary:
        priority_types = (
            "FELFeatures",
            "FELBasinStructures",
            "PCA",
            "RMSD",
            "RMSF",
            "DCCM",
            "DSSP",
        )
    else:
        priority_types = (
            "ClassificationClustering",
            "ClassificationFeatures",
            "FELFeatures",
            "FELBasinStructures",
            "Ligand_Pocket_COM_Distance",
            "ProteinLigandContacts",
            "LigandResidence",
            "PocketSASA",
            "Pocket_RMSF",
            "Ligand_RMSF",
            "PCA",
            "RMSD",
            "RMSF",
            "DCCM",
            "DSSP",
        )

    per_sim_count: Dict[str, int] = {}
    fel_count: Dict[str, int] = {}

    for sim, sim_dir in zip(sims_summary, sim_dirs):
        label = sim.get("label", "")
        analysis_dir = Path(sim.get("analysis_dir") or Path(sim_dir) / "analysis")
        search_dirs = [analysis_dir, Path(sim_dir), Path(sim_dir) / "analysis"]

        latest_by_type: Dict[str, Dict[str, Any]] = {}
        for rec in sim.get("records") or []:
            atype = rec.get("analysis_type") or ""
            if atype:
                latest_by_type[atype] = rec

        for atype in priority_types:
            rec = latest_by_type.get(atype)
            if not rec:
                continue
            metric = metric_for_analysis_type(atype)
            if metric and metric not in relevant:
                continue
            if narrative.prefer_cluster_views and metric in narrative.suppress_global_for:
                continue

            images = collect_images_fn(
                rec.get("files") or {},
                rec.get("metadata") or {},
                search_dirs,
            )
            dccm_comparison_shown = any(
                "dccm" in Path(p).name.lower()
                and any(token in Path(p).name.lower() for token in ("panel", "diff", "comparison"))
                for p in exclude
            )
            selected = select_images_from_entry(
                images,
                user_goal=user_goal,
                report_focus=report_focus,
                enriched_prompt=enriched_prompt,
                analysis_type=atype,
                policy=policy,
            )
            for img_path in selected.values():
                if str(img_path) in exclude:
                    continue
                # The comparative section already shows one apo|holo|Δ figure
                # per protein. Per-simulation heatmaps and line plots repeat it.
                img_name = Path(str(img_path)).name.lower()
                if "dccm" in img_name and (
                    dccm_comparison_shown or "heatmap" not in img_name
                ):
                    continue
                if atype in {"FELFeatures", "FELBasinStructures", "FreeEnergyLandscape"}:
                    if fel_count.get(label, 0) >= narrative.max_fel_per_sim:
                        continue
                    fel_count[label] = fel_count.get(label, 0) + 1
                if per_sim_count.get(label, 0) >= policy.max_figures_per_sim_combined:
                    break
                score = score_plot_path(str(img_path), relevant) + score_analysis_entry(
                    atype, relevant, has_statistics=bool(rec.get("statistics"))
                )
                if narrative.classification_primary and atype.startswith("FEL"):
                    score += 2.0
                candidates.append((
                    score,
                    {
                        "label": label,
                        "analysis_type": atype,
                        "path": str(img_path),
                        "caption": f"{label} — {atype.replace('_', ' ')}",
                    },
                ))
                per_sim_count[label] = per_sim_count.get(label, 0) + 1

    candidates.sort(key=lambda x: (-x[0], x[1]["label"], x[1]["path"]))
    cap = policy.max_combined_highlight_figures
    if narrative.classification_primary:
        cap = min(cap, max(len(sims_summary), 8))
    return [c for _, c in candidates[:cap]]


def summarize_findings_for_literature(
    analysis_data: Dict[str, Any],
    selected_entries: Optional[Sequence[Dict[str, Any]]] = None,
    max_entries: int = 8,
) -> str:
    """Build a concise findings block for literature review prompts."""
    entries = list(selected_entries or analysis_data.get("entries") or [])
    if not entries:
        return ""

    lines: List[str] = []
    for entry in entries[:max_entries]:
        atype = entry.get("analysis_type", "Unknown")
        stats = entry.get("statistics") or {}
        if not isinstance(stats, dict) or not stats:
            continue
        # Top 4 numeric stats per analysis type
        nums = [
            (k, v)
            for k, v in stats.items()
            if isinstance(v, (int, float)) and not k.startswith("_")
        ][:4]
        if nums:
            stat_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in nums)
            lines.append(f"- {atype}: {stat_str}")

    return "\n".join(lines)


def build_figure_catalog_for_prompt(
    selected_entries: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    max_figures: int = 8,
) -> List[str]:
    """Return planned figure labels (``figN. Caption``) for LLM prompts.

    Mirrors HTML numbering: one primary plot per selected analysis entry.
    """
    catalog: List[str] = []
    n = 0
    for entry in selected_entries or []:
        atype = str(entry.get("analysis_type") or "Analysis").replace("_", " ")
        files = entry.get("files") or {}
        has_plot = any(
            isinstance(v, str) and str(v).lower().endswith((".png", ".pdf", ".svg", ".jpg"))
            for v in files.values()
        ) if isinstance(files, dict) else False
        if not has_plot and not (entry.get("statistics") or {}):
            continue
        if has_plot:
            n += 1
            # Prefer metric-like short captions
            low = atype.lower()
            if "rmsd" in low and "rmsf" not in low:
                caption = "RMSD"
            elif "rmsf" in low:
                caption = "RMSF"
            elif "gyration" in low or low in {"rg", "radius of gyration"}:
                caption = "Radius of Gyration"
            else:
                caption = atype
            catalog.append(f"fig{n}. {caption}")
        if len(catalog) >= max_figures:
            break
    return catalog


def grounded_numeric_claims(
    selected_entries: Optional[Sequence[Dict[str, Any]]] = None,
    max_values: int = 24,
) -> List[str]:
    """Flatten allowed numeric values from selected analyses for claim grounding."""
    out: List[str] = []
    for entry in selected_entries or []:
        stats = entry.get("statistics") or {}
        if not isinstance(stats, dict):
            continue
        atype = entry.get("analysis_type") or "analysis"
        for k, v in stats.items():
            if isinstance(v, float):
                out.append(f"{atype}.{k}={v:.4g}")
            elif isinstance(v, int):
                out.append(f"{atype}.{k}={v}")
            if len(out) >= max_values:
                return out
    return out
