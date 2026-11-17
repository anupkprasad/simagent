"""
Combined Reporter - Multi-simulation comparison HTML report generator

Reads per-simulation analysis summaries and overlay plots produced by the
combined_analysis phase, then builds a rich single-page comparison HTML report
with: task description, per-sim overview, 3D structure viewer (one structure per
simulation), overlay plots, statistics comparison table, aggregated literature,
and a final combined impression with discussion, hypotheses, and future steps.

Designed to be imported by agentic/reporter/tools.py and exposed as an
@tool for the Reporter Agent.
"""
import html as _html_mod
import json
import logging
import re
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Sequence

from langchain.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(path: str, search_dirs: Optional[List[Path]] = None) -> Optional[str]:
    """Return a base64-encoded data-URI for PNG/JPEG, or None on failure."""
    try:
        from src.reporter.html_generator import _resolve_asset_path

        resolved = _resolve_asset_path(path, search_dirs)
        p = resolved if resolved and resolved.is_file() else Path(path)
        if not p.is_file():
            logger.warning(f"Could not encode image (not found): {path}")
            return None
        data = p.read_bytes()
        ext = p.suffix.lstrip(".").lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as exc:
        logger.warning(f"Could not encode image {path}: {exc}")
        return None


def _population_from_viewer_label(label: str) -> float:
    """Parse population percentage from labels like ``FEL Basin 2 (26%, 95.5 ns)``."""
    m = re.search(r"\((\d+(?:\.\d+)?)%", label)
    return float(m.group(1)) if m else 0.0


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read an analysis_summary.jsonl file."""
    try:
        from src.analysis.summary_logger import read_summary_file
        records = read_summary_file(str(Path(path).parent))
        return [r for r in records if r.get("analysis_type")]
    except Exception:
        pass

    records = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("---"):
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        logger.warning(f"Could not read {path}: {exc}")
    return [r for r in records if r.get("analysis_type")]


def _parse_label_name_map(text: str, sim_labels: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Extract a {uniprot_id: protein_name} mapping from free text."""
    from src.reporter.report_curator import parse_label_name_map_strict

    _UNIPROT_ID_RE = re.compile(r"^[a-z][0-9][a-z0-9]{3,9}[0-9a-z]*$", re.IGNORECASE)
    _GENE_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9\-]{1,20}$")

    strict = parse_label_name_map_strict(text, sim_labels)
    if strict:
        return strict

    # Legacy fallback: same regex as before, but filter to sim labels + gene names.
    mapping: Dict[str, str] = {}
    sim_set = {s.lower() for s in (sim_labels or []) if s}
    if not text:
        return mapping
    for m in re.finditer(
        r'\b([A-Za-z0-9_\-]+)\s*:\s*([A-Za-z][A-Za-z0-9_\-]+)',
        text
    ):
        key, val = m.group(1).strip(), m.group(2).strip()
        key_l, val_st = key.lower(), val.strip()
        if key_l in {"id", "e", "g", "eg"} or val_st.lower() in {"name", "map"}:
            continue
        if not _UNIPROT_ID_RE.match(key_l):
            continue
        if not _GENE_NAME_RE.match(val_st):
            continue
        if sim_set and key_l not in sim_set:
            continue
        mapping[key_l] = val_st
    return mapping


def resolve_display_label(label: str, name_map: Optional[Dict[str, str]] = None) -> str:
    """Map simulation directory labels to protein-based display names.

    Examples (map ``p21860`` → ``ERBB3``):
      ``p21860``           → ``ERBB3``
      ``p21860_ATP_MG``    → ``ERBB3_ATP_MG``
      ``p21860-protein``   → ``ERBB3-protein``
    """
    if not label or not name_map:
        return label
    lmap = {k.lower(): v for k, v in name_map.items()}
    low = label.lower()
    if low in lmap:
        return lmap[low]
    for pid, pname in lmap.items():
        if low == pid:
            return pname
        for sep in ("_", "-"):
            prefix = f"{pid}{sep}"
            if low.startswith(prefix) and len(label) > len(prefix):
                suffix = label[len(pid):]  # keeps ``_ATP_MG`` or ``-protein``
                return f"{pname}{suffix}"
    return label


def apply_label_name_map(
    labels: List[str],
    name_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Apply ``resolve_display_label`` to every simulation label."""
    if not name_map:
        return list(labels)
    return [resolve_display_label(lbl, name_map) for lbl in labels]


def _collect_sim_summaries(
    sim_dirs: List[str],
    labels: List[str],
    label_name_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Read analysis_summary.jsonl from each per-sim analysis directory.

    Returns a list of dicts, one per simulation, with keys:
        label, analysis_dir, records (list of JSONL records)

    If *label_name_map* is provided it is used to replace raw labels (e.g.
    UniProt IDs) with human-readable protein names.
    """
    sims = []
    for sim_dir, label in zip(sim_dirs, labels):
        display_label = resolve_display_label(label, label_name_map)
        analysis_dir = Path(sim_dir) / "analysis"
        jsonl = analysis_dir / "analysis_summary.jsonl"
        records: List[Dict[str, Any]] = []
        if jsonl.exists():
            records = _read_jsonl(str(jsonl))
        else:
            logger.warning(f"No analysis_summary.jsonl in {analysis_dir}")
        sims.append({
            "label": display_label,
            "analysis_dir": str(analysis_dir),
            "records": records,
        })
    return sims


def _extract_stats(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict mapping analysis_type → statistics from JSONL records.
    Only records that have a "statistics" key are included.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        atype = rec.get("analysis_type", "")
        if atype and rec.get("statistics"):
            stats[atype] = rec["statistics"]
    return stats


def _parse_timepoint(filename: str) -> str:
    """Extract simulation timepoint (e.g. '36 ns') from a PDB filename."""
    # Match patterns like system_frame_36ns.pdb, frame_36ns, frame36ns
    m = re.search(r'[_\-](\d+)\s*ns', filename, re.IGNORECASE)
    if m:
        return f"{m.group(1)} ns"
    # Fallback: match bare frame number, e.g. system_frame0.pdb
    m2 = re.search(r'frame(\d+)', filename, re.IGNORECASE)
    if m2:
        return f"{m2.group(1)} ns"
    return "0 ns"


def _collect_per_sim_resources(
    sim_dirs: List[str],
    labels: List[str],
    label_name_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Read comprehensive_summary.json and execution_plan.json for each sim."""
    resources = []
    for sim_dir, label in zip(sim_dirs, labels):
        display_label = resolve_display_label(label, label_name_map)
        res: Dict[str, Any] = {
            "label": display_label,
            "sim_dir": sim_dir,
            "report_focus": "",
            "reasoning": "",
            "literature_queries": [],
            "html_report_path": None,
            "analysis_types": [],
        }
        reporter_dir = Path(sim_dir) / "reporter"

        # Read comprehensive_summary.json (analysis data + report_focus)
        summary_path = reporter_dir / "comprehensive_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    d = json.load(f)
                lp = d.get("llm_plan", {})
                res["report_focus"] = lp.get("report_focus", "")
                res["literature_queries"] = lp.get("literature_queries", [])
                # Collect analysis types actually performed
                ads = d.get("analysis_data_summary", {})
                if isinstance(ads, dict):
                    res["analysis_types"] = ads.get("analysis_types", [])
                elif isinstance(ads, list):
                    res["analysis_types"] = [r.get("analysis_type", "") for r in ads if r.get("analysis_type")]
                for step in d.get("step_results", []):
                    if step.get("tool") == "generate_html_report":
                        res["html_report_path"] = step.get("result", {}).get("report_file")
            except Exception as exc:
                logger.warning(f"Could not read comprehensive_summary.json for {label}: {exc}")

        # Read execution_plan.json for the supervisor-assigned reasoning / report focus
        plan_path = reporter_dir / "execution_plan.json"
        if plan_path.exists():
            try:
                with open(plan_path, encoding="utf-8") as f:
                    ep = json.load(f)
                res["reasoning"] = ep.get("reasoning", "")
                if not res["report_focus"]:
                    res["report_focus"] = ep.get("report_focus", "")
                if not res["literature_queries"]:
                    res["literature_queries"] = ep.get("literature_queries", [])
            except Exception as exc:
                logger.warning(f"Could not read execution_plan.json for {label}: {exc}")

        if not res["html_report_path"] and reporter_dir.exists():
            html_files = sorted(reporter_dir.glob("*.html"))
            if html_files:
                res["html_report_path"] = str(html_files[0])

        resources.append(res)
    return resources


def _select_representative_pdbs(
    sim_dirs: List[str], labels: List[str], max_pdbs: int = 0
) -> Dict[str, str]:
    """Pick scientifically significant PDB structures for the combined 3D viewer.

    For each simulation (priority order):
      1. Highest-population FEL basin representative PDB (from analysis export)
      2. Analysis-driven trajectory frames (COM/contact/residence extrema)
      3. Legacy fallback: earliest frame in ``reporter/*.pdb``

    Labels use the display name (e.g. ``JAK1 — FEL Basin 2 (26%, 95.5 ns)``).
    """
    return _select_significant_pdbs_for_combined(
        sim_dirs, labels, label_name_map=None, max_per_sim=2, max_total=max_pdbs or 16
    )


def _cluster_rep_prefix_map(base_analysis_dir: Optional[str]) -> Dict[str, str]:
    """Map representative display name → viewer prefix (e.g. ``Cluster 2 rep``)."""
    if not base_analysis_dir:
        return {}
    from src.reporter.report_curator import (
        build_reference_archetype_narrative,
        is_reference_archetype_study,
    )

    base = Path(base_analysis_dir)
    if not is_reference_archetype_study(base):
        return {}
    narrative = build_reference_archetype_narrative(base)
    if not narrative:
        return {}
    prefix_map: Dict[str, str] = {}
    for block in narrative.get("clusters") or []:
        disp = (block.get("representative_display") or "").strip()
        if disp:
            prefix_map[disp] = f"Cluster {block['cluster_id']} rep"
    return prefix_map


def _load_reference_fel_json(base: Path, display_name: str) -> Optional[Dict[str, Any]]:
    path = base / "reference_fel" / display_name / "fel_features.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _reference_fel_plot_paths(base: Path, display_name: str) -> List[Tuple[str, str]]:
    """Return (path, caption) pairs for per-system reference FEL figures."""
    fel_dir = base / "reference_fel" / display_name
    if not fel_dir.is_dir():
        return []
    candidates = [
        ("reference_fel_basins.png", "Reference FEL basins"),
        ("reference_fel_pc1_pc2.png", "Reference FEL (PC1 vs PC2)"),
        ("reference_pca_pc1_pc2.png", "Reference PCA (PC1 vs PC2)"),
    ]
    out: List[Tuple[str, str]] = []
    for fname, caption in candidates:
        path = fel_dir / fname
        if path.is_file():
            out.append((str(path), caption))
    return out


def _fmt_metric(value: Optional[float], *, unit: str = "", decimals: int = 2) -> str:
    if value is None:
        return "—"
    text = f"{value:.{decimals}f}"
    return f"{text}{unit}" if unit else text


_METRIC_LABELS: Dict[str, Tuple[str, str, int]] = {
    "reference_pocket_ligand_distance_std_A": ("ATP–pocket COM (std)", " Å", 2),
    "reference_pocket_ligand_distance_p95_A": ("ATP–pocket COM (p95)", " Å", 2),
    "reference_pocket_fraction_bound": ("Fraction bound (≤15 Å)", "", 2),
    "reference_pocket_std_ligand_axis_angle_deg": ("Axis angle fluctuation (std)", "°", 1),
    "reference_pocket_ligand_axis_angle_p95_deg": ("Axis angle (p95)", "°", 1),
    "ref_landscape_entropy": ("FEL landscape entropy", "", 2),
    "ref_major_basin_population": ("Major basin population", "", 2),
}


def _build_reference_archetype_findings_section(
    base_analysis_dir: Optional[str] = None,
) -> str:
    """
    Overall + per-cluster findings for reference FEL+pocket archetype studies.

    Each cluster card highlights one representative with pocket metrics, FEL basin
    table, and reference FEL figures when available.
    """
    if not base_analysis_dir:
        return ""
    from src.reporter.report_curator import (
        build_reference_archetype_narrative,
        is_reference_archetype_study,
        _load_features_table,
    )

    base = Path(base_analysis_dir)
    if not is_reference_archetype_study(base):
        return ""

    narrative = build_reference_archetype_narrative(base)
    if not narrative:
        return ""

    features = _load_features_table(base)
    feat_by_label: Dict[str, Dict[str, str]] = {}
    for row in features:
        lbl = (row.get("label") or "").strip().lower()
        if lbl:
            feat_by_label[lbl] = row

    clusters = narrative.get("clusters") or []
    if not clusters:
        return ""

    # Overall bullet summary
    overall_bits: List[str] = []
    for block in clusters:
        cid = block["cluster_id"]
        title = block.get("archetype_title", "dynamic archetype")
        n = block.get("n_members", 0)
        rep = block.get("representative_display", "")
        overall_bits.append(
            f"<li><strong>Cluster {cid}</strong> ({n} systems, {title}): "
            f"representative <strong>{_html_mod.escape(rep)}</strong></li>"
        )

    html_parts: List[str] = [
        "<h2>&#128221; Overall Findings &amp; Cluster Archetypes</h2>\n",
        f"<p>{_html_mod.escape(narrative['overall_summary'])}</p>\n",
        "<h3>Summary by cluster</h3>\n",
        "<ul class=\"findings-summary-list\">\n",
        *overall_bits,
        "</ul>\n",
        "<h3>Cluster representatives (FEL + pocket dynamics)</h3>\n",
        '<p class="findings-note">One centroid-nearest kinase per cluster demonstrates '
        "the binding-site coupling and reference FEL metastates for that archetype. "
        "Matching 3D structures appear in the interactive viewer below.</p>\n",
        '<div class="archetype-grid">\n',
    ]

    for block in clusters:
        cid = block["cluster_id"]
        title = block.get("archetype_title", "Dynamic archetype")
        rep_disp = block.get("representative_display", "")
        rep_uid = (block.get("representative_uid") or "").strip().lower()
        members = block.get("members_display") or []
        profile = block.get("profile") or {}

        member_text = ", ".join(_html_mod.escape(m) for m in members)
        if len(members) > 12:
            member_text = (
                ", ".join(_html_mod.escape(m) for m in members[:10])
                + f", … (+{len(members) - 10} more)"
            )

        card = [
            f'<div class="archetype-card archetype-cluster-{cid}">',
            f'<div class="archetype-card-header">'
            f'<span class="archetype-badge">Cluster {cid}</span> '
            f'{_html_mod.escape(title)}</div>',
            f'<p class="archetype-members"><strong>{block.get("n_members", len(members))} members:</strong> '
            f"{member_text}</p>",
            f'<p class="archetype-rep"><strong>Representative:</strong> '
            f'{_html_mod.escape(rep_disp)} '
            f'<span class="viewer-hint">(viewer: Cluster {cid} rep — …)</span></p>',
        ]

        # Cluster-mean metrics
        if profile:
            card.append('<table class="archetype-metrics"><tr><th>Metric (cluster mean)</th><th>Value</th></tr>')
            for key, (label, unit, dec) in _METRIC_LABELS.items():
                if key in profile:
                    card.append(
                        f"<tr><td>{_html_mod.escape(label)}</td>"
                        f"<td>{_fmt_metric(profile.get(key), unit=unit, decimals=dec)}</td></tr>"
                    )
            card.append("</table>")

        # Representative-specific metrics
        rep_row = feat_by_label.get(rep_uid, {})
        if rep_row:
            card.append(
                f"<h4>Representative pocket + FEL scalars ({_html_mod.escape(rep_disp)})</h4>"
            )
            card.append(
                '<table class="archetype-metrics"><tr><th>Metric</th><th>Value</th></tr>'
            )
            for key, (label, unit, dec) in _METRIC_LABELS.items():
                raw = rep_row.get(key)
                if raw not in (None, ""):
                    try:
                        val = float(raw)
                    except ValueError:
                        continue
                    card.append(
                        f"<tr><td>{_html_mod.escape(label)}</td>"
                        f"<td>{_fmt_metric(val, unit=unit, decimals=dec)}</td></tr>"
                    )
            card.append("</table>")

        # FEL basin table
        fel = _load_reference_fel_json(base, rep_disp) if rep_disp else None
        if fel:
            n_min = fel.get("n_minima")
            card.append(
                f"<h4>Reference FEL metastates ({_html_mod.escape(rep_disp)})</h4>"
            )
            if n_min is not None:
                card.append(
                    f"<p>Landscape entropy {_fmt_metric(fel.get('landscape_entropy'))}; "
                    f"major basin #{fel.get('major_basin_id')} "
                    f"({_fmt_metric((fel.get('major_basin_population') or 0) * 100, unit='%', decimals=1)} "
                    f"population).</p>"
                )
            basins = fel.get("basins") or []
            if basins:
                card.append(
                    '<table class="archetype-metrics fel-basin-table">'
                    "<tr><th>Basin</th><th>Population</th>"
                    "<th>ΔG min (kJ/mol)</th><th>Depth (kJ/mol)</th></tr>"
                )
                for b in sorted(basins, key=lambda x: -(x.get("population") or 0)):
                    pop = (b.get("population") or 0) * 100
                    card.append(
                        f"<tr><td>{b.get('basin_id', '?')}</td>"
                        f"<td>{_fmt_metric(pop, unit='%', decimals=1)}</td>"
                        f"<td>{_fmt_metric(b.get('free_energy_min_kJ_mol'), decimals=2)}</td>"
                        f"<td>{_fmt_metric(b.get('basin_depth_kJ_mol'), decimals=2)}</td></tr>"
                    )
                card.append("</table>")

        # FEL figures for representative
        fel_plots = _reference_fel_plot_paths(base, rep_disp) if rep_disp else []
        if fel_plots:
            card.append('<div class="plot-grid archetype-fel-plots">')
            for path, caption in fel_plots:
                uri = _encode_image(path, search_dirs=[base])
                if uri:
                    card.append(
                        f'<div class="plot-card">'
                        f'<p><strong>{_html_mod.escape(caption)}</strong></p>'
                        f'<img src="{uri}" alt="{_html_mod.escape(caption)}" loading="lazy">'
                        f"</div>"
                    )
            card.append("</div>")

        card.append("</div>")
        html_parts.append("\n".join(card))

    html_parts.append("</div>\n")
    return "".join(html_parts)


def _select_significant_pdbs_for_combined(
    sim_dirs: List[str],
    labels: List[str],
    label_name_map: Optional[Dict[str, str]] = None,
    max_per_sim: int = 2,
    max_total: int = 16,
    user_goal: Optional[str] = None,
    enriched_prompt: Optional[str] = None,
    highlight_plan: Optional[Any] = None,
    base_analysis_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Collect significant structures across all simulations for the 3D viewer.

    When a ``highlight_plan`` (``StructureHighlightPlan``) is provided, the
    number of metastates embedded per simulation follows the plan — story-
    relevant proteins get multiple conformational states while the rest are
    limited or skipped — instead of one structure for every simulation.
    """
    from src.reporter.figure_selector import resolve_report_narrative

    narrative = resolve_report_narrative(
        user_goal or "",
        enriched_prompt=enriched_prompt or "",
        n_simulations=len(sim_dirs),
    )
    if narrative.classification_primary:
        max_per_sim = min(max_per_sim, narrative.max_fel_per_sim)
    cap = narrative.max_pdb_structures if narrative.max_pdb_structures else max_total
    cap = cap if cap and cap > 0 else max(len(sim_dirs) * max_per_sim, 8)
    if highlight_plan is not None and highlight_plan.highlight_labels:
        planned = sum(
            highlight_plan.structures_for(str(d))
            for d in highlight_plan.highlight_labels
        )
        if planned > 0:
            cap = max(cap, planned)

    from src.reporter.structure_extractor import (
        collect_significant_structures,
        load_fel_basin_structures,
    )

    cluster_rep_prefix = _cluster_rep_prefix_map(base_analysis_dir)
    uid_to_disp: Dict[str, str] = {}
    if base_analysis_dir:
        try:
            from src.analysis.reference_labels import build_uniprot_display_map

            uid_to_disp = build_uniprot_display_map(Path(base_analysis_dir))
        except Exception:
            pass

    frames: Dict[str, str] = {}

    def _viewer_key(display: str, sub_label: str) -> str:
        prefix = cluster_rep_prefix.get(display)
        if prefix:
            return f"{prefix} — {display} — {sub_label}"
        return f"{display} — {sub_label}"

    for sim_dir, label in zip(sim_dirs, labels):
        if len(frames) >= cap:
            break
        display = resolve_display_label(label, label_name_map)
        if uid_to_disp and label.lower() in uid_to_disp:
            display = uid_to_disp[label.lower()]

        # Per-sim metastate budget from the highlight plan (0 → skip this sim).
        sim_max = max_per_sim
        if highlight_plan is not None:
            sim_max = highlight_plan.structures_for(display)
            if sim_max <= 0:
                continue

        analysis_dir = Path(sim_dir) / "analysis"
        jsonl = analysis_dir / "analysis_summary.jsonl"
        records = _read_jsonl(str(jsonl)) if jsonl.exists() else []
        analysis_data = {"entries": records}

        added_this_sim = 0

        # Priority 1: FEL basin PDBs — major basin first, then next by population
        basin_frames = load_fel_basin_structures(analysis_data, sim_dir)
        if basin_frames:
            ranked = sorted(
                basin_frames.items(),
                key=lambda kv: _population_from_viewer_label(kv[0]),
                reverse=True,
            )
            for b_label, pdb_text in ranked[:sim_max]:
                if len(frames) >= cap:
                    break
                viewer_key = _viewer_key(display, b_label)
                frames[viewer_key] = pdb_text
                added_this_sim += 1

        # Priority 2: trajectory frames from analysis time series
        if added_this_sim < sim_max:
            sig = collect_significant_structures(
                analysis_data,
                sim_dir,
                sim_dir,
                "hpc",
                max_total=max(sim_max - added_this_sim, 1),
            )
            for t_label, pdb_text in sig.items():
                if len(frames) >= cap or added_this_sim >= sim_max:
                    break
                viewer_key = _viewer_key(display, t_label)
                if viewer_key not in frames:
                    frames[viewer_key] = pdb_text
                    added_this_sim += 1

        # Priority 3: legacy reporter/*.pdb (earliest frame)
        if added_this_sim == 0:
            reporter_dir = Path(sim_dir) / "reporter"
            if reporter_dir.exists():
                pdb_files = sorted(reporter_dir.glob("*.pdb"))
                if pdb_files:
                    chosen = pdb_files[0]
                    for pf in pdb_files:
                        name = pf.name.lower()
                        if "_0ns" in name or "frame0" in name or name.endswith("_0.pdb"):
                            chosen = pf
                            break
                    try:
                        pdb_text = chosen.read_text(encoding="utf-8", errors="replace")
                        timepoint = _parse_timepoint(chosen.name)
                        frames[_viewer_key(display, timepoint)] = pdb_text
                    except Exception as exc:
                        logger.warning(f"Could not read PDB {chosen}: {exc}")

    logger.info(
        "Combined 3D viewer: collected %d structure(s): %s",
        len(frames),
        ", ".join(list(frames.keys())[:6]) + ("..." if len(frames) > 6 else ""),
    )
    return frames


def _parse_refs_from_html(html_path: Optional[str]) -> List[Dict[str, Any]]:
    """Extract literature references from a per-sim HTML report (class='reference' divs)."""
    if not html_path or not Path(html_path).exists():
        return []
    try:
        html_text = Path(html_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning(f"Could not read HTML {html_path}: {exc}")
        return []

    refs: List[Dict[str, Any]] = []
    ref_blocks = re.findall(r'<div class="reference">(.*?)</div>\s*\n\s*</div>', html_text, re.DOTALL)
    for block in ref_blocks:
        ref: Dict[str, Any] = {}

        # Title + URL from reference-title
        title_link_m = re.search(
            r'<div class="reference-title">[^<]*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block, re.DOTALL
        )
        if title_link_m:
            raw_url = title_link_m.group(1)
            ref["title"] = _html_mod.unescape(re.sub(r"<[^>]+>", "", title_link_m.group(2))).strip()
            ref["url"] = raw_url
            if "doi.org/" in raw_url:
                ref["doi"] = raw_url.replace("https://doi.org/", "").replace("http://doi.org/", "")
            elif "pubmed" in raw_url:
                m = re.search(r"/(\d+)/?$", raw_url)
                if m:
                    ref["pmid"] = m.group(1)
        else:
            title_plain = re.search(r'<div class="reference-title">(.*?)</div>', block, re.DOTALL)
            if title_plain:
                raw = _html_mod.unescape(re.sub(r"<[^>]+>", "", title_plain.group(1))).strip()
                ref["title"] = re.sub(r"^\[\d+\]\s*", "", raw)

        # Authors
        auth_m = re.search(r'<div class="reference-authors">(.*?)</div>', block, re.DOTALL)
        if auth_m:
            authors_raw = _html_mod.unescape(re.sub(r"<[^>]+>", "", auth_m.group(1)))
            ref["authors"] = [a.strip() for a in authors_raw.split(",") if a.strip()]

        # Journal + year
        meta_m = re.search(r'<span class="journal">(.*?)</span>\s*\((\d{4})\)', block, re.DOTALL)
        if meta_m:
            ref["journal"] = _html_mod.unescape(re.sub(r"<[^>]+>", "", meta_m.group(1))).strip()
            ref["year"] = meta_m.group(2)

        # Explicit DOI
        doi_m = re.search(r'DOI: <a href="https://doi\.org/([^"]+)"', block)
        if doi_m and not ref.get("doi"):
            ref["doi"] = doi_m.group(1)

        # Explicit PMID
        pmid_m = re.search(r'PMID: <a href="https://pubmed[^"]*/(\d+)/"', block)
        if pmid_m and not ref.get("pmid"):
            ref["pmid"] = pmid_m.group(1)

        if ref.get("title"):
            refs.append(ref)

    return refs


def _score_ref_relevance(
    ref: Dict[str, Any],
    protein_terms: List[str],
    analysis_terms: List[str],
    context: Optional[Dict[str, Any]] = None,
) -> int:
    """Score a reference for relevance. Higher = more relevant."""
    if context:
        from src.reporter.literature_search import score_literature_ref_relevance
        score, _ = score_literature_ref_relevance(ref, context)
        return score

    haystack = " ".join([
        ref.get("title", ""),
        ref.get("abstract", "") or "",
        ref.get("journal", ""),
        " ".join(ref.get("authors", [])),
    ]).lower()

    score = 0
    for term in protein_terms:
        if term.lower() in haystack:
            score += 3
    for term in analysis_terms:
        if term.lower() in haystack:
            score += 1
    return score


def _aggregate_literature(
    sim_resources: List[Dict[str, Any]],
    max_refs: int = 10,
    hypothesis_text: Optional[str] = None,
    protein_name: Optional[str] = None,
    user_goal: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Collect, deduplicate, and rank literature refs from all per-sim HTML reports.

    Refs are ranked by relevance to:
      (a) protein/system names derived from simulation labels
      (b) analysis method terms (RMSD, RMSF, etc.)
      (c) hypothesis / objective keywords from the user goal

    Returns at most *max_refs* references.

    Args:
        sim_resources: Per-simulation resource dicts with "label", "html_report_path", etc.
        max_refs: Maximum references to return.
        hypothesis_text: Optional free-text goal/hypothesis from the user to boost
                         papers that match the research objective.
    """
    # ── Build protein/system term list ───────────────────────────────────
    # Sim labels are the most reliable protein identifiers (derived from PDB stems).
    protein_terms: List[str] = []
    _GENERIC = {"rmsd", "rmsf", "protein", "simulation", "analysis",
                "molecular", "dynamics", "with", "without", "versus", "and"}
    for res in sim_resources:
        lbl = res.get("label", "")
        if lbl:
            # Split on underscores, hyphens, spaces — each part may be meaningful
            for part in re.split(r"[_\-\s]+", lbl):
                if len(part) >= 3 and part.lower() not in _GENERIC:
                    protein_terms.append(part)

        # Also harvest words from existing query strings
        for q in res.get("literature_queries", []):
            words = [w for w in re.split(r"\s+", q) if len(w) > 3
                     and w.lower() not in _GENERIC]
            protein_terms.extend(words)

        foci = res.get("report_focus", [])
        if isinstance(foci, list):
            for f in foci:
                protein_terms.extend(w for w in re.split(r"\s+", str(f)) if len(w) > 4)
        elif isinstance(foci, str) and foci:
            protein_terms.extend(w for w in re.split(r"\s+", foci) if len(w) > 4)

    # ── Build hypothesis/objective terms ─────────────────────────────────
    hyp_terms: List[str] = []
    _HYP_TRIGGERS = {
        "inhibit", "bind", "interact", "affect", "role", "function",
        "mechanism", "pathway", "stability", "flexibility", "alloster",
        "mutation", "mutant", "drug", "therapeutic", "disease", "cancer",
        "activate", "deactivate", "phosphorylat", "fold", "unfold",
        "aggregate", "dimer", "oligomer",
    }
    if hypothesis_text:
        htl = hypothesis_text.lower()
        for trig in _HYP_TRIGGERS:
            if trig in htl:
                hyp_terms.append(trig)
        # Also include meaningful non-stop words from the goal
        _STOP = {"please", "could", "would", "should", "compute", "calculate",
                 "trajectory", "simulation", "report", "generate", "agent"}
        for w in re.split(r"\W+", htl):
            if len(w) > 4 and w not in _STOP and w not in hyp_terms:
                hyp_terms.append(w)

    # ── Inject explicit protein name with boosted weight ─────────────────
    # Repeat the protein_name 3× so that any ref mentioning it scores high.
    if protein_name:
        for _part in re.split(r"[_\-\s]+", protein_name):
            if len(_part) >= 2:
                protein_terms.extend([_part] * 3)

    analysis_terms = ["RMSD", "RMSF", "radius of gyration", "molecular dynamics",
                      "MD simulation", "protein", "trajectory", "secondary structure",
                      "pseudokinase", "activation loop", "apo", "holo", "ATP", "allosteric"]

    from src.reporter.literature_search import extract_research_context
    research_context = extract_research_context(
        user_goal=hypothesis_text or user_goal,
        protein_name=protein_name,
        analysis_types=analysis_terms,
    )
    for pname in research_context.get("protein_names", []):
        protein_terms.extend([pname] * 3)

    all_refs: List[Dict[str, Any]] = []
    seen_dois: set = set()
    seen_pmids: set = set()
    seen_titles: set = set()

    for res in sim_resources:
        refs = _parse_refs_from_html(res.get("html_report_path"))
        for ref in refs:
            doi = ref.get("doi", "")
            pmid = ref.get("pmid", "")
            title_key = re.sub(r"\W+", "", ref.get("title", "").lower())[:60]

            if doi and doi in seen_dois:
                continue
            if pmid and pmid in seen_pmids:
                continue
            if title_key and title_key in seen_titles:
                continue

            if doi:
                seen_dois.add(doi)
            if pmid:
                seen_pmids.add(pmid)
            if title_key:
                seen_titles.add(title_key)

            ref["source"] = res["label"]
            ref["_relevance"] = _score_ref_relevance(
                ref, protein_terms + hyp_terms, analysis_terms, context=research_context,
            )
            all_refs.append(ref)

    # Sort by relevance descending, then cap
    all_refs.sort(key=lambda r: r.get("_relevance", 0), reverse=True)
    min_score = 8
    filtered = [r for r in all_refs if r.get("_relevance", 0) >= min_score]
    if len(filtered) < 2:
        filtered = all_refs
    for ref in filtered:
        ref.pop("_relevance", None)

    return filtered[:max_refs]


def _extract_mean_value(stats: Dict[str, Any], atype_lower: str) -> Optional[float]:
    """Extract the relevant mean value from a stats dict based on analysis type."""
    # Type-specific keys have priority (these are the actual keys produced by MD analysis tools)
    if "rmsd" in atype_lower:
        candidates = ("mean_rmsd_angstrom", "mean_rmsd_nm", "mean", "average")
    elif "rmsf" in atype_lower:
        candidates = ("mean_rmsf_angstrom", "mean_rmsf_nm", "mean", "average")
    elif "radius" in atype_lower or "gyration" in atype_lower or atype_lower == "rg":
        candidates = ("mean_rg_angstrom", "mean_rg_nm", "mean", "average")
    else:
        candidates = ("mean", "mean_angstrom", "mean_nm", "average")

    for key in candidates:
        v = stats.get(key)
        if isinstance(v, (int, float)) and v != 0.0:
            return float(v)
    # Allow zero only as last resort
    for key in candidates:
        v = stats.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _build_combined_final_impression(
    sims_summary: List[Dict[str, Any]],
    sim_resources: List[Dict[str, Any]],
) -> str:
    """Generate a synthesis paragraph comparing dynamics across simulations."""
    sim_rmsd: Dict[str, float] = {}
    sim_rmsf: Dict[str, float] = {}
    sim_rg: Dict[str, float] = {}

    for s in sims_summary:
        label = s["label"]
        st = _extract_stats(s["records"])
        for atype, vals in st.items():
            alow = atype.lower()
            val = _extract_mean_value(vals, alow)
            if val is None:
                continue
            if "rmsd" in alow:
                sim_rmsd[label] = val
            elif "rmsf" in alow:
                sim_rmsf[label] = val
            elif "radius" in alow or "gyration" in alow or alow == "rg":
                sim_rg[label] = val

    resource_map = {r["label"]: r for r in sim_resources}
    parts: List[str] = []

    labels_str = ", ".join(s["label"] for s in sims_summary)
    n_sims = len(sims_summary)
    n_reps = 1
    for s in sims_summary:
        wd = s.get("working_dir") or s.get("sim_dir") or ""
        if not wd:
            continue
        avg_summary = Path(wd) / "analysis" / "avg" / "summary_scalars.json"
        if avg_summary.is_file():
            try:
                payload = json.loads(avg_summary.read_text(encoding="utf-8"))
                n_reps = max(n_reps, int(payload.get("n_reps") or 1))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
    rep_note = (
        f" Each system used {n_reps} independent production replicate"
        f"{'s' if n_reps != 1 else ''} (metrics shown are replicate means)."
        if n_reps > 1
        else ""
    )
    parts.append(
        f"This combined analysis compared {n_sims} molecular dynamics "
        f"simulation{'s' if n_sims != 1 else ''} ({labels_str}).{rep_note} "
        "The cross-simulation comparison reveals both shared trends and notable "
        "differences in structural dynamics."
    )

    if sim_rmsd:
        rmsd_sorted = sorted(sim_rmsd.items(), key=lambda x: x[1])
        most_stable = rmsd_sorted[0]
        least_stable = rmsd_sorted[-1]
        vals_str = "; ".join(f"{lbl}: {v:.3f}" for lbl, v in rmsd_sorted)
        parts.append(
            f"**Structural stability (RMSD):** {vals_str} Å. "
            f"Simulation **{most_stable[0]}** shows the greatest structural stability "
            f"(lowest average RMSD = {most_stable[1]:.3f} Å), while **{least_stable[0]}** "
            f"exhibits the largest conformational deviations from the reference structure."
        )

    if sim_rmsf:
        rmsf_sorted = sorted(sim_rmsf.items(), key=lambda x: x[1])
        most_rigid = rmsf_sorted[0]
        most_flexible = rmsf_sorted[-1]
        vals_str = "; ".join(f"{lbl}: {v:.3f}" for lbl, v in rmsf_sorted)
        parts.append(
            f"**Backbone flexibility (RMSF):** {vals_str} Å. "
            f"Simulation **{most_rigid[0]}** has the most rigid backbone, whereas "
            f"**{most_flexible[0]}** shows higher per-residue fluctuations, "
            "suggesting increased local flexibility."
        )

    if sim_rg:
        rg_sorted = sorted(sim_rg.items(), key=lambda x: x[1])
        most_compact = rg_sorted[0]
        least_compact = rg_sorted[-1]
        vals_str = "; ".join(f"{lbl}: {v:.3f}" for lbl, v in rg_sorted)
        parts.append(
            f"**Structural compactness (Rg):** {vals_str} Å. "
            f"Simulation **{most_compact[0]}** maintains the most compact fold "
            f"(Rg = {most_compact[1]:.3f} Å), while **{least_compact[0]}** displays "
            f"a somewhat expanded conformation (Rg = {least_compact[1]:.3f} Å)."
        )

    for res in sim_resources:
        focus = res.get("report_focus", "")
        if focus and len(focus) > 20:
            parts.append(f"**{res['label']} key observations:** {focus}")

    # ── Interpretation + discussion (flowing prose, no subsection headings) ─
    def _is_holo(lbl: str) -> bool:
        l = lbl.lower()
        return any(k in l for k in ("atp", "_mg", "holo", "ligand", "bound"))

    holo_labels = [s["label"] for s in sims_summary if _is_holo(s["label"])]
    apo_labels = [s["label"] for s in sims_summary if not _is_holo(s["label"])]

    if holo_labels and apo_labels:
        rmsf_clause = ""
        if sim_rmsf:
            apo_mean = [sim_rmsf[l] for l in apo_labels if l in sim_rmsf]
            holo_mean = [sim_rmsf[l] for l in holo_labels if l in sim_rmsf]
            if apo_mean and holo_mean:
                a = sum(apo_mean) / len(apo_mean)
                h = sum(holo_mean) / len(holo_mean)
                if h < a:
                    rmsf_clause = (
                        f" On average the ligand-bound (holo) systems are more rigid "
                        f"(mean RMSF {h:.2f} Å) than their apo counterparts "
                        f"({a:.2f} Å), consistent with ATP binding damping local "
                        "backbone fluctuations around the nucleotide pocket."
                    )
                else:
                    rmsf_clause = (
                        f" The holo systems show comparable or higher mean flexibility "
                        f"(RMSF {h:.2f} Å vs apo {a:.2f} Å), suggesting that ATP binding "
                        "redistributes rather than simply suppresses backbone motion."
                    )
        parts.append(
            "Comparing the apo and holo states isolates the dynamic signature of ATP "
            "binding: differences in RMSD, RMSF, radius of gyration, ATP–pocket COM "
            "distance, and the apo→holo ΔDCCM maps together indicate how the nucleotide "
            "reshapes flexibility and inter-residue coupling, particularly across the "
            "αC-helix and activation loop (residues 150–200)." + rmsf_clause +
            " Persistent ATP–pocket COM distances in the holo trajectories point to a "
            "stably bound nucleotide, so the accompanying changes in correlated motion "
            "most likely reflect genuine ligand-induced allosteric coupling rather than "
            "ligand drift. Where ΔDCCM patterns are conserved across the pseudokinases "
            "they suggest a common allosteric wiring, whereas protein-specific "
            "differences highlight divergent regulatory mechanisms — for example ATP "
            "may rigidify the activation loop and αC-helix in some systems while "
            "rerouting correlated motion toward distal regulatory elements in others."
        )
    else:
        parts.append(
            "Taken together, the comparison of global stability (RMSD), local "
            "flexibility (RMSF), compactness (Rg), and correlated motion (DCCM) shows "
            "how system composition and environment shape each conformational "
            "ensemble, with the largest differences concentrated around the "
            "functionally important active-site region."
        )

    # Future directions woven into the same flowing section (no heading).
    parts.append(
        "These observations are drawn from finite-length classical MD with a single "
        "replica per system, so the apo-vs-holo trends should be confirmed with "
        "independent or extended trajectories and proper error estimates (e.g. block "
        "averaging). Promising next steps include enhanced-sampling or free-energy "
        "calculations (metadynamics, umbrella sampling, MM/PBSA) to quantify ATP "
        "binding and activation-loop landscapes, per-residue decomposition of the "
        "ΔDCCM together with network analysis to map candidate allosteric pathways, "
        "in-silico mutagenesis of predicted hotspot residues, and cross-validation "
        "against experimental data (HDX-MS, NMR order parameters, or mutational "
        "studies) reported for each pseudokinase."
    )

    if not (sim_rmsd or sim_rmsf or sim_rg):
        parts.append(
            "Detailed quantitative comparison was limited by available analysis data. "
            "Refer to individual simulation reports for system-specific findings."
        )

    return "\n\n".join(parts)


def _build_science_status_html(
    *,
    working_dir: str,
    base_analysis_dir: str = "",
) -> str:
    """Banner for campaign science completeness (general MD, not process success)."""
    try:
        from agentic.campaign.contracts import (
            campaign_science_report,
            clustering_table_usable,
            family_feature_matrix_ready,
        )
        from agentic.campaign.spec import CampaignSpec
    except Exception:
        return ""

    base = Path(working_dir)
    # Reporter writes under ``{campaign}/reporter``; campaign root is parent.
    campaign_root = base.parent if base.name == "reporter" else base
    if not (campaign_root / "analysis").is_dir() and base_analysis_dir:
        campaign_root = Path(base_analysis_dir).parent

    spec = None
    state_path = campaign_root / "campaign" / "state.json"
    if state_path.is_file():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            raw = payload.get("campaign_spec") or payload
            if isinstance(raw, dict) and raw.get("mode"):
                spec = CampaignSpec.from_dict(raw)
        except Exception:
            spec = None
    if spec is None or not getattr(spec, "family_modular", False):
        return ""

    report = campaign_science_report(str(campaign_root), spec=spec)
    feat = campaign_root / "analysis" / "classification_features.csv"
    matrix = family_feature_matrix_ready(str(feat), spec) if feat.is_file() else {
        "ok": False,
        "reason": "classification_features.csv missing",
        "n_complete": 0,
        "n_rows": 0,
    }
    usable = clustering_table_usable(str(feat)) if feat.is_file() else {"ok": False}
    ok = bool(report.get("ok"))
    bg = "#ecfdf5" if ok else "#fff7ed"
    border = "#059669" if ok else "#c2410c"
    title = "Science completeness: OK" if ok else "Science completeness: incomplete"
    icon = "&#9989;" if ok else "&#9888;"
    bits = [
        f"<div class='task-box' style='background:{bg};border:1px solid {border};margin:12px 0;'>",
        f"<div class='task-box-header' style='color:{border};'>{icon} {title}</div>",
        "<div class='task-box-body' style='font-size:13.5px;line-height:1.55;'>",
        f"<p><b>Systems with required artifacts:</b> "
        f"{report.get('n_science_complete', '?')}/"
        f"{report.get('n_systems', '?')}</p>",
        f"<p><b>Feature matrix:</b> {_html_mod.escape(str(matrix.get('reason') or ('ready' if matrix.get('ok') else 'incomplete')))}</p>",
    ]
    incomplete = report.get("incomplete_labels") or []
    if incomplete:
        bits.append(
            "<p><b>Incomplete systems:</b> "
            + _html_mod.escape(", ".join(str(x) for x in incomplete[:20]))
            + ("…" if len(incomplete) > 20 else "")
            + "</p>"
        )
    if matrix.get("n_rows"):
        bits.append(
            f"<p><b>Complete rows:</b> {matrix.get('n_complete', 0)}/{matrix.get('n_rows', 0)} "
            f"(clustering plot usable: {'yes' if usable.get('ok') else 'no'})</p>"
        )
    # Surface empty preferred columns when present in the table as NaN.
    empty_cols: List[str] = []
    if feat.is_file():
        try:
            import csv as _csv

            with feat.open(newline="", encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))
            skip = {"label", "sim_directory", "n_features_present"}
            preferred = list(
                (spec.analysis_recipe.required_feature_columns if spec else [])
                or []
            )
            for col in preferred:
                if not rows or col not in (rows[0] or {}):
                    empty_cols.append(col)
                    continue
                n_fin = 0
                for row in rows:
                    raw = (row.get(col) or "").strip()
                    if not raw or raw.lower() in {"nan", "none", "null"}:
                        continue
                    try:
                        float(raw)
                        n_fin += 1
                    except ValueError:
                        continue
                if n_fin < 2:
                    empty_cols.append(col)
        except Exception:
            empty_cols = []
    if empty_cols:
        bits.append(
            "<p><b>Empty / unusable preferred columns:</b> "
            + _html_mod.escape(", ".join(empty_cols))
            + "</p>"
        )
        bits.append(
            "<p style='margin:0;color:#9a3412;'>"
            "Dendrogram/heatmap may still be drawn from the remaining numeric columns; "
            "treat clustering as provisional until required calculations fill these fields."
            "</p>"
        )
    elif not ok:
        bits.append(
            "<p style='margin:0;color:#9a3412;'>"
            "Process success is not enough — required shared calculations or modular "
            "artifacts are still missing."
            "</p>"
        )
    bits.append("</div></div>")
    return "\n".join(bits)


def _build_task_description_html(
    enriched_prompt: str,
    sim_resources: Optional[List[Dict[str, Any]]] = None,
    user_goal: Optional[str] = None,
    protein_name: Optional[str] = None,
) -> str:
    """Render a structured task description card.

    Layout:
      - Overall Goal: the original user request (concise, single source of truth)
      - Per-Simulation Analysis Objectives: table from per-sim reasoning / report_focus
    """
    if not user_goal and not sim_resources:
        return ""

    parts: List[str] = []
    parts.append('<div class="task-box">')
    parts.append('<div class="task-box-header">&#128203; Study Objectives &amp; Task Description</div>')
    parts.append('<div class="task-box-body">')

    primary_goal = (user_goal or "").strip()
    if primary_goal:
        parts.append('<div style="margin-bottom:14px;">')
        parts.append('<strong style="color:#5b21b6;">&#128269; Overall Goal</strong><br>')
        parts.append(
            f'<p style="margin:6px 0 0 0;line-height:1.6;color:#1e1b4b;">'
            f'{_html_mod.escape(primary_goal)}</p>'
        )
        parts.append('</div>')

    # ── Per-simulation analysis objectives (from execution_plan / reasoning) ──
    if sim_resources:
        parts.append('<hr style="border:none;border-top:1px dashed #c4b5fd;margin:10px 0;">')
        parts.append(
            '<strong style="color:#5b21b6;">&#128296; Per-Simulation Analysis Objectives</strong>'
        )
        parts.append('<table style="width:100%;border-collapse:collapse;margin-top:10px;font-size:13.5px;">')
        parts.append('<tr style="background:#ede9fe;">')
        parts.append('<th style="padding:6px 10px;text-align:left;color:#5b21b6;">Simulation</th>')
        parts.append('<th style="padding:6px 10px;text-align:left;color:#5b21b6;">Analysis Goals</th>')
        parts.append('</tr>')

        for idx, res in enumerate(sim_resources):
            label = res.get("label", "?")
            foci = res.get("report_focus", [])
            reasoning = res.get("reasoning", "")

            if isinstance(foci, list) and foci:
                focus_items = foci[:4]
            elif isinstance(foci, str) and foci:
                focus_items = [foci]
            elif reasoning:
                focus_items = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reasoning.strip())
                               if len(s.strip()) > 20][:2]
            else:
                focus_items = []

            if focus_items:
                cell_html = "<ul style='margin:2px 0 2px 18px;padding:0;'>" + \
                    "".join(f"<li style='margin:2px 0'>{_html_mod.escape(str(f))}</li>"
                             for f in focus_items) + "</ul>"
            else:
                cell_html = "<em style='color:#9ca3af;'>No objectives recorded</em>"

            row_bg = "background:#faf9ff;" if idx % 2 == 0 else "background:#f5f3ff;"
            parts.append(f'<tr style="{row_bg}">')
            parts.append(f'<td style="padding:7px 10px;font-weight:700;color:#7c3aed;white-space:nowrap;">{_html_mod.escape(label)}</td>')
            parts.append(f'<td style="padding:7px 10px;">{cell_html}</td>')
            parts.append('</tr>')

        parts.append('</table>')

    parts.append('</div>')  # task-box-body
    parts.append('</div>')  # task-box
    parts.append('')
    return "\n".join(parts)


def _build_3d_viewer_html(pdb_frames: Dict[str, str]) -> str:
    """Build interactive 3Dmol.js viewer section for multiple simulations."""
    try:
        from src.reporter.html_generator import _build_3d_viewer_section
        return _build_3d_viewer_section(pdb_frames)
    except Exception as exc:
        logger.warning(f"Could not import _build_3d_viewer_section: {exc}")
        return ""


def _build_literature_html(
    refs: List[Dict[str, Any]],
    literature_review: Optional[str] = None,
) -> str:
    """Render Literature Review narrative only (References go at report end)."""
    if not literature_review:
        return ""

    parts = ["<h2>&#128218; Literature Review</h2>"]
    parts.append('<div class="literature-review">')
    for paragraph in literature_review.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            parts.append(f"<p>{_format_report_paragraph(paragraph)}</p>")
    parts.append("</div>")
    return "\n".join(parts)


def _build_references_section_html(refs: List[Dict[str, Any]]) -> str:
    """Render the terminal References section (cited-only, or empty notice).

    Always placed at the end of the combined report. Lists only papers that
    survived citation sync (in-text ``[n]``). When none were cited, writes an
    explicit empty message instead of omitting the section.
    """
    parts = [
        "<h2>&#128218; References</h2>",
    ]
    if not refs:
        parts.append(
            '<p class="no-references" style="color:#4b5563;font-style:italic;">'
            "No relevant reference.</p>"
        )
        return "\n".join(parts)

    for idx, ref in enumerate(refs, 1):
        title = _html_mod.escape(ref.get("title", "Unknown"))
        doi = (ref.get("doi") or "").strip()
        pmid = str(ref.get("pmid") or "").strip()
        # Prefer explicit url; else derive from DOI / PMID so hyperlinks survive
        # re-imports that only carry a bare url field.
        raw_url = (ref.get("url") or ref.get("link") or "").strip()
        if not raw_url and doi:
            raw_url = f"https://doi.org/{doi}"
        elif not raw_url and pmid:
            raw_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if raw_url and "doi.org/" in raw_url and not doi:
            doi = raw_url.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        source = ref.get("source", "")
        source_badge = (
            f'<span class="sim-badge">{_html_mod.escape(source)}</span>' if source else ""
        )

        parts.append('<div class="reference">')

        if raw_url:
            parts.append(
                f'<div class="reference-title">[{idx}] '
                f'<a href="{_html_mod.escape(raw_url)}" target="_blank">'
                f'{title}</a>{source_badge}</div>'
            )
        else:
            parts.append(f'<div class="reference-title">[{idx}] {title}{source_badge}</div>')

        authors = ref.get("authors", [])
        if authors:
            author_str = ", ".join(authors[:5])
            if len(authors) > 5:
                author_str += " et al."
            parts.append(f'<div class="reference-authors">{_html_mod.escape(author_str)}</div>')

        journal = ref.get("journal", "")
        year = ref.get("year", "")
        if journal or year:
            meta = ""
            if journal:
                meta += f'<span class="journal">{_html_mod.escape(journal)}</span>'
            if year:
                meta += f" ({year})"
            parts.append(f'<div class="reference-meta">{meta}</div>')

        id_parts = []
        if doi:
            id_parts.append(
                f'DOI: <a href="https://doi.org/{_html_mod.escape(doi)}" target="_blank">'
                f'{_html_mod.escape(doi)}</a>'
            )
        if pmid:
            id_parts.append(
                f'PMID: <a href="https://pubmed.ncbi.nlm.nih.gov/{_html_mod.escape(pmid)}/" target="_blank">'
                f'{_html_mod.escape(pmid)}</a>'
            )
        if id_parts:
            parts.append(f'<div class="reference-doi">{", ".join(id_parts)}</div>')

        abstract = ref.get("abstract") or ref.get("relevance_note") or ""
        if abstract:
            snippet = abstract[:500] + ("…" if len(abstract) > 500 else "")
            parts.append(
                f'<div class="reference-abstract" style="font-size:0.9em;color:#4b5563;'
                f'margin-top:6px;line-height:1.5;">{_html_mod.escape(snippet)}</div>'
            )

        parts.append("</div>")

    return "\n".join(parts)


def _format_report_paragraph(paragraph: str, cite_class: str = "lit-cite") -> str:
    """Convert markdown bold and [n] citations to HTML."""
    paragraph = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", paragraph)
    paragraph = re.sub(
        r"\[(\d+)\]",
        rf'<sup class="{cite_class}">[\1]</sup>',
        paragraph,
    )
    return paragraph


def _build_final_impression_html(text: str, refs: List[Dict[str, Any]]) -> str:
    """Render the combined final impression paragraph with bold/citation support."""
    if not text:
        return ""

    parts = [
        "<h2>&#127919; Combined Final Impression</h2>",
        '<div class="final-impression">',
    ]

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            parts.append(
                f"<p>{_format_report_paragraph(paragraph, cite_class='final-cite')}</p>"
            )

    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML building helpers (statistics & overlay plots)
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    margin: 0; padding: 0;
    background: linear-gradient(135deg, #f0f4ff 0%, #e8f4f8 100%);
    color: #1f2937; min-height: 100vh;
}
.container {
    max-width: min(1600px, 96vw); margin: 20px auto 40px; padding: 30px 28px 60px;
    background: white;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10);
    border-radius: 12px;
}
h1 {
    color: #1e3a8a; border-bottom: 4px solid #3b82f6;
    padding-bottom: 15px; margin-bottom: 30px; font-size: 2em;
}
h2 {
    color: #1e40af; margin-top: 40px; border-bottom: 2px solid #93c5fd;
    padding-bottom: 10px; font-size: 1.5em;
}
h3 { color: #1e40af; font-size: 1.15em; margin-top: 18px; }
.header-meta {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    padding: 18px 22px; border-radius: 8px; margin: 20px 0;
    border-left: 4px solid #3b82f6;
}
.header-meta p { margin: 6px 0; font-size: 15px; }
/* ---- Task description card ---- */
.task-box {
    margin: 24px 0; border-radius: 10px;
    border: 1px solid #c4b5fd; border-left: 5px solid #7c3aed; overflow: hidden;
}
.task-box-header {
    background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
    padding: 12px 20px; color: #5b21b6; font-weight: 700; font-size: 1em;
}
.task-box-body {
    padding: 16px 24px; background: #faf9ff; color: #3b0764;
    font-size: 14.5px; line-height: 1.75;
}
.task-box-body ul { margin: 6px 0; padding-left: 22px; }
.task-box-body li { margin: 5px 0; }
/* ---- Tables ---- */
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }
th { background: #1e3a8a; color: white; padding: 10px 14px; text-align: left; }
td { border: 1px solid #e5e7eb; padding: 9px 14px; }
tr:nth-child(even) td { background: #f3f8ff; }
tr:hover td { background: #dbeafe; transition: background 0.15s; }
/* ---- Overlay plot grid ---- */
.plot-grid { display: flex; flex-wrap: wrap; gap: 18px; margin: 20px 0; }
.plot-card {
    background: white; border: 1px solid #d1d5db; border-radius: 8px;
    padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    flex: 1 1 420px; text-align: center;
}
.plot-card img { width: 100%; height: auto; border-radius: 5px; }
.plot-card p { margin: 8px 0 0; font-size: 13px; color: #6b7280; font-style: italic; }
/* ---- Reference archetype findings ---- */
.findings-summary-list { margin: 12px 0 24px; padding-left: 22px; line-height: 1.65; }
.findings-summary-list li { margin: 8px 0; }
.findings-note { color: #374151; font-size: 14px; margin: 8px 0 18px; }
.archetype-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr));
    gap: 20px; margin: 20px 0;
}
.archetype-card {
    border: 1px solid #c7d2fe; border-radius: 10px; padding: 18px 20px;
    background: linear-gradient(180deg, #fafbff 0%, #ffffff 100%);
    box-shadow: 0 2px 10px rgba(99,102,241,0.08);
}
.archetype-card-header {
    font-size: 1.05em; font-weight: 700; color: #312e81; margin-bottom: 10px;
    border-bottom: 2px solid #e0e7ff; padding-bottom: 8px;
}
.archetype-badge {
    display: inline-block; background: #6366f1; color: white;
    padding: 2px 10px; border-radius: 999px; font-size: 0.85em; margin-right: 6px;
}
.archetype-members, .archetype-rep { font-size: 14px; margin: 8px 0; line-height: 1.5; }
.viewer-hint { color: #6b7280; font-size: 12px; font-style: italic; }
.archetype-metrics { font-size: 13px; margin: 10px 0 14px; }
.archetype-metrics th { background: #4338ca; font-size: 12px; padding: 7px 10px; }
.archetype-metrics td { padding: 6px 10px; }
.archetype-card h4 { color: #3730a3; font-size: 0.95em; margin: 14px 0 6px; }
.archetype-fel-plots .plot-card { flex: 1 1 240px; }
/* ---- Section divider ---- */
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    margin: 40px 0;
}
/* ---- 3D Viewer (copied from html_generator) ---- */
.viewer-section {
    margin: 30px 0; padding: 25px; border-radius: 10px;
    background: #f0f4ff; border: 1px solid #c7d2fe; border-left: 5px solid #6366f1;
}
.viewer-container {
    position: relative; width: 100%; height: 550px; border-radius: 8px;
    overflow: hidden; background: #1a1a2e; box-shadow: 0 2px 12px rgba(0,0,0,0.15);
}
.viewer-controls {
    display: flex; flex-wrap: wrap; gap: 8px; margin: 15px 0; align-items: center;
}
.viewer-controls label { font-size: 13px; font-weight: 600; color: #4338ca; margin-right: 4px; }
.viewer-controls select, .viewer-controls button {
    padding: 6px 14px; border: 1px solid #c7d2fe; border-radius: 6px;
    background: white; font-size: 13px; cursor: pointer;
}
.viewer-controls button { background: #6366f1; color: white; border: none; font-weight: 600; }
.viewer-controls button:hover { background: #4f46e5; }
/* ---- Literature ---- */
.reference {
    margin: 15px 0; padding: 16px 20px; background: #f9fafb;
    border-left: 4px solid #6b7280; border-radius: 6px;
}
.reference-title { font-weight: 700; color: #1f2937; font-size: 15px; margin-bottom: 5px; }
.reference-title a { color: #1e40af; text-decoration: none; }
.reference-title a:hover { text-decoration: underline; }
.reference-authors { color: #4b5563; font-size: 13px; margin: 3px 0; }
.reference-meta { font-size: 13px; color: #4b5563; margin: 3px 0; }
.reference-meta .journal { font-style: italic; }
.reference-doi { font-size: 12px; color: #6b7280; margin-top: 4px; }
.reference-doi a { color: #2563eb; }
.sim-badge {
    display: inline-block; padding: 1px 7px; border-radius: 5px;
    font-size: 11px; font-weight: 600; margin-left: 6px;
    background: #dbeafe; color: #1e40af;
}
.literature-review {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    padding: 18px 22px; border-radius: 8px; margin: 12px 0 20px;
    line-height: 1.75; font-size: 15px;
}
.literature-review .lit-cite { color: #15803d; font-weight: 700; }
.reference-abstract { font-style: italic; }
/* ---- Final impression ---- */
.final-impression {
    margin: 20px 0; padding: 28px 30px; border-radius: 10px;
    background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
    border: 1px solid #fde68a; border-left: 5px solid #f59e0b;
}
.final-impression p { margin: 10px 0; line-height: 1.85; font-size: 16px; }
.final-impression strong { color: #92400e; }
.final-impression .final-cite { color: #b45309; font-weight: 700; }
/* ---- Stats section ---- */
.stats-section {
    margin: 20px 0; padding: 20px; background: #f9fafb;
    border-radius: 8px; border: 1px solid #e5e7eb;
}
.stats-dropdown {
    margin: 10px 0; border: 1px solid #dbeafe; border-radius: 8px;
    background: #fff; overflow: hidden;
}
.stats-dropdown summary {
    cursor: pointer; padding: 12px 16px; font-weight: 700; color: #1e40af;
    background: #eff6ff; list-style: none; user-select: none;
}
.stats-dropdown summary::-webkit-details-marker { display: none; }
.stats-dropdown summary::before {
    content: "▸ "; display: inline-block; transition: transform 0.15s ease;
}
.stats-dropdown[open] summary::before { transform: rotate(90deg); }
.stats-dropdown .stats-table-wrap { padding: 0 12px 12px; overflow-x: auto; }
.stats-dropdown table { margin-top: 8px; }
/* ---- Footer ---- */
footer {
    text-align: center; padding: 20px; font-size: 13px; color: #9ca3af;
    margin-top: 40px; border-top: 1px solid #e5e7eb;
}
/* ---- Comparative Dynamics Summary panels ---- */
.dyn-panel-grid {
    display: flex;
    flex-direction: column;
    gap: 28px;
    margin: 20px 0;
}
.dyn-panel {
    border: 1px solid #e0e7ff;
    border-top: 4px solid #6366f1;
    border-radius: 8px;
    overflow: hidden;
    background: white;
    box-shadow: 0 2px 8px rgba(99,102,241,0.07);
    width: 100%;
}
.dyn-panel.panel-full {
    width: 100%;
}
.panel-header {
    background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.panel-badge {
    background: #6366f1;
    color: white;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 10px;
    white-space: nowrap;
}
.panel-title {
    font-weight: 700;
    color: #312e81;
    font-size: 14px;
}
.panel-subtitle {
    font-size: 12px;
    color: #6b7280;
    padding: 4px 16px 6px;
    background: #fafaff;
    border-bottom: 1px solid #e0e7ff;
    font-style: italic;
}
.panel-body {
    padding: 14px 16px;
    min-height: 80px;
}
.panel-body img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 4px;
}
.panel-stack {
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.panel-stack-item {
    width: 100%;
}
.panel-stack-label {
    font-size: 13px;
    font-weight: 700;
    color: #1e40af;
    margin: 0 0 8px 0;
}
.panel-missing {
    text-align: center;
    color: #9ca3af;
    font-size: 13px;
    padding: 30px 20px;
    font-style: italic;
}
"""


def _build_stats_section(sims_summary: List[Dict[str, Any]]) -> str:
    """Generate collapsible HTML comparison tables of per-sim statistics."""
    all_types: List[str] = []
    for s in sims_summary:
        for t in _extract_stats(s["records"]):
            if t not in all_types:
                all_types.append(t)

    if not all_types:
        return "<p><em>No per-simulation statistics available.</em></p>\n"

    html_parts: List[str] = [
        '<p style="font-size:13px;color:#6b7280;margin:0 0 12px;">'
        "Expand a subsection below to view detailed statistics for each analysis type.</p>"
    ]
    for atype in all_types:
        stat_keys: List[str] = []
        for s in sims_summary:
            st = _extract_stats(s["records"]).get(atype, {})
            if st:
                stat_keys = [k for k in st if isinstance(st[k], (int, float, str))]
                break
        if not stat_keys:
            continue

        table_parts = ["<table>\n"]
        header_cols = "".join(
            f"<th>{_html_mod.escape(k.replace('_', ' ').title())}</th>" for k in stat_keys
        )
        table_parts.append(f"<tr><th>Simulation</th>{header_cols}</tr>\n")

        for s in sims_summary:
            st = _extract_stats(s["records"]).get(atype, {})
            cells = "".join(
                f"<td>{st.get(k, 'N/A')}</td>" for k in stat_keys
            )
            table_parts.append(
                f"<tr><td><b>{_html_mod.escape(s['label'])}</b></td>{cells}</tr>\n"
            )
        table_parts.append("</table>\n")

        html_parts.append(
            f'<details class="stats-dropdown">'
            f'<summary>{_html_mod.escape(atype)}</summary>'
            f'<div class="stats-table-wrap">{"".join(table_parts)}</div>'
            f'</details>\n'
        )

    return "".join(html_parts)


def _build_plots_section(overlay_plots: List[str]) -> str:
    """Embed overlay images as base64 data-URIs in a responsive grid."""
    if not overlay_plots:
        return "<p><em>No overlay plots found.</em></p>\n"

    cards: List[str] = []
    for fpath in overlay_plots:
        uri = _encode_image(fpath)
        if uri is None:
            continue
        fname = Path(fpath).name
        metric = fname.replace("_overlay.png", "").replace("_", " ").upper()
        cards.append(
            f'<div class="plot-card">'
            f'<img src="{uri}" alt="{metric} overlay" loading="lazy">'
            f'<p>{metric} Overlay — all simulations</p>'
            f'</div>\n'
        )

    if not cards:
        return "<p><em>Could not encode overlay plots.</em></p>\n"

    return '<div class="plot-grid">\n' + "".join(cards) + "</div>\n"


# ---------------------------------------------------------------------------
# Comparative Dynamics Summary — multi-panel helpers
# ---------------------------------------------------------------------------

def _find_overlay_by_type(overlay_plots: List[str], keyword: str) -> Optional[str]:
    """Return the first overlay plot path whose filename contains *keyword*."""
    kl = keyword.lower()
    for p in overlay_plots:
        if kl in Path(p).name.lower():
            return p
    return None


def _dedupe_dccm_combined_plots(plot_paths: List[str]) -> List[str]:
    """Drop standalone ΔDCCM figures when a comparison panel already embeds the difference.

    ``plot_dccm_comparison`` appends a Δ panel for two-simulation comparisons, so
    separate ``dccm_difference*.png`` files would duplicate that content in reports.
    """
    if not plot_paths:
        return []
    names = [Path(p).name.lower() for p in plot_paths]
    has_comparison = any("comparison" in n for n in names)
    if not has_comparison:
        return list(plot_paths)
    return [
        p for p in plot_paths
        if "difference" not in Path(p).name.lower()
    ]


def _collect_comparative_panel_plot_paths(
    overlay_plots: List[str],
    sim_dirs: List[str],
    labels: List[str],
) -> set:
    """Paths already shown in the Comparative Dynamics Summary (Panels A–F)."""
    used: set = set()

    def _mark(path: Optional[str]) -> None:
        if path and Path(path).exists():
            used.add(str(Path(path).resolve()))

    for keyword in ("rmsd", "rmsf", "rg", "gyration", "com_distance"):
        _mark(_find_overlay_by_type(overlay_plots, keyword))

    for plot_path in overlay_plots:
        name = Path(plot_path).name.lower()
        if "rmsf_segment" in name or "dssp_comparison" in name:
            _mark(plot_path)
        if "dccm_apo_holo" in name and "panels" in name:
            _mark(plot_path)

    for entry in _collect_per_sim_dssp_figures(sim_dirs, labels):
        _mark(entry.get("heatmap"))
        _mark(entry.get("timeseries"))

    for _, pth in _find_per_sim_plots(sim_dirs, labels, "pocket_distance"):
        _mark(pth)
    if not used:
        for _, pth in _find_per_sim_plots(sim_dirs, labels, "ligand_pocket"):
            _mark(pth)

    dccm_combined = _find_dccm_apo_holo_panel_plots(overlay_plots)
    if not dccm_combined:
        dccm_diff = _find_dccm_diff_heatmap_plots(overlay_plots)
        if dccm_diff:
            dccm_combined = dccm_diff
        else:
            dccm_combined = _dedupe_dccm_combined_plots([
                p for p in overlay_plots
                if "dccm" in Path(p).name.lower()
            ])
    _has_dccm_comparison = any(
        "comparison" in Path(p).name.lower() or "diff" in Path(p).name.lower()
        for p in dccm_combined
    )
    if not _has_dccm_comparison:
        for _, pth in _find_per_sim_plots(sim_dirs, labels, "dccm_heatmap"):
            _mark(pth)
        for _, pth in _find_per_sim_plots(sim_dirs, labels, "dccm"):
            _mark(pth)

    for p in dccm_combined:
        _mark(p)

    for plot_path in overlay_plots:
        name = Path(plot_path).name.lower()
        if any(k in name for k in ("ligand_pocket_distance", "com_distance.png")):
            _mark(plot_path)

    return used


def _filter_redundant_overlay_plots(
    overlay_plots: List[str],
    sim_dirs: List[str],
    labels: List[str],
) -> List[str]:
    """Remove plots already embedded in the comparative dynamics multi-panel section."""
    used = _collect_comparative_panel_plot_paths(overlay_plots, sim_dirs, labels)
    filtered: List[str] = []
    for plot_path in overlay_plots:
        resolved = str(Path(plot_path).resolve()) if Path(plot_path).exists() else plot_path
        if resolved in used or plot_path in used:
            continue
        filtered.append(plot_path)
    return filtered


def _find_per_sim_plots(
    sim_dirs: List[str],
    labels: List[str],
    keyword: str,
) -> List[Tuple[str, str]]:
    """Search each sim's analysis directory for a plot matching *keyword*.

    Returns a list of (label, filepath) pairs, at most one per simulation.
    """
    kl = keyword.lower()
    results: List[Tuple[str, str]] = []
    for sim_dir, label in zip(sim_dirs, labels):
        analysis_dir = Path(sim_dir) / "analysis"
        if not analysis_dir.exists():
            continue
        for png in sorted(analysis_dir.glob("*.png")):
            if kl in png.name.lower():
                results.append((label, str(png)))
                break
    return results


def _build_summary_bar_chart_svg(sims_summary: List[Dict[str, Any]]) -> str:
    """Generate an inline SVG grouped bar chart for stability/flexibility ranking (Panel F)."""
    # Collect per-sim metric means
    sim_data: List[Dict[str, Any]] = []
    for s in sims_summary:
        st = _extract_stats(s["records"])
        rmsd_val = rmsf_val = rg_val = None
        for atype, vals in st.items():
            alow = atype.lower()
            v = _extract_mean_value(vals, alow)
            if v is None:
                continue
            if "rmsd" in alow and rmsd_val is None:
                rmsd_val = v
            elif "rmsf" in alow and rmsf_val is None:
                rmsf_val = v
            elif ("radius" in alow or "gyration" in alow or alow == "rg") and rg_val is None:
                rg_val = v
        sim_data.append({"label": s["label"], "rmsd": rmsd_val, "rmsf": rmsf_val, "rg": rg_val})

    metrics = []
    if any(d["rmsd"] is not None for d in sim_data):
        metrics.append(("RMSD (\u00c5)", "rmsd", "#3b82f6", "Stability"))
    if any(d["rmsf"] is not None for d in sim_data):
        metrics.append(("RMSF (\u00c5)", "rmsf", "#ef4444", "Flexibility"))
    if any(d["rg"] is not None for d in sim_data):
        metrics.append(("Rg (\u00c5)", "rg", "#10b981", "Compactness"))

    if not metrics:
        return ""

    n_sims = len(sim_data)
    n_metrics = len(metrics)

    # SVG layout constants
    svgw = 760
    ml, mr, mt, mb = 72, 20, 50, 90
    pw = svgw - ml - mr
    ph = 230

    svgh = mt + ph + mb
    group_w = pw / max(n_sims, 1)
    bar_pad = 8
    bar_total = group_w - bar_pad * 2
    bw = max((bar_total / max(n_metrics, 1)) - 4, 4)

    # Y-axis scale
    all_vals = [d[k] for d in sim_data for _, k, _, _ in metrics if d.get(k) is not None]
    ymax = (max(all_vals) * 1.18) if all_vals else 1.0
    if ymax == 0:
        ymax = 1.0

    def val_to_y(v: float) -> float:
        return ph - (v / ymax) * ph + mt

    def val_to_h(v: float) -> float:
        return (v / ymax) * ph

    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svgw} {svgh}" '
        f'style="width:100%;max-width:{svgw}px;height:auto;display:block;margin:auto;">'
    )
    lines.append(f'<rect width="{svgw}" height="{svgh}" fill="white" rx="4"/>')
    lines.append(
        f'<text x="{svgw // 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="700" fill="#1e3a8a" font-family="Segoe UI,Arial,sans-serif">'
        f'Panel F \u2014 Stability &amp; Flexibility Summary Ranking</text>'
    )

    # Y-axis gridlines + labels
    for i in range(6):
        yval = ymax * i / 5
        ypos = val_to_y(yval)
        lines.append(
            f'<line x1="{ml}" y1="{ypos:.1f}" x2="{ml + pw}" y2="{ypos:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{ml - 6}" y="{ypos + 4:.1f}" text-anchor="end" font-size="10" '
            f'fill="#6b7280" font-family="Segoe UI,Arial,sans-serif">{yval:.2f}</text>'
        )

    # Axes
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#9ca3af" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#9ca3af" stroke-width="1.5"/>')

    # Y-axis unit label
    mid_y = mt + ph // 2
    lines.append(
        f'<text x="14" y="{mid_y}" text-anchor="middle" '
        f'transform="rotate(-90,14,{mid_y})" font-size="11" fill="#374151" '
        f'font-family="Segoe UI,Arial,sans-serif">Value (\u00c5)</text>'
    )

    # Bars
    for si, d in enumerate(sim_data):
        gx = ml + si * group_w + bar_pad
        for mi, (metric_label, key, color, _role) in enumerate(metrics):
            v = d.get(key)
            if v is None or v < 0:
                continue
            bx = gx + mi * (bw + 4)
            by = val_to_y(v)
            bh = val_to_h(v)
            safe_lbl = _html_mod.escape(d["label"])
            safe_metric = _html_mod.escape(metric_label)
            lines.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                f'fill="{color}" rx="2" opacity="0.85">'
                f'<title>{safe_lbl} {safe_metric}: {v:.3f}</title></rect>'
            )
            label_y = by - 3
            if label_y < mt + 12:
                label_y = by + 12
            lines.append(
                f'<text x="{bx + bw / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                f'font-size="8" fill="{color}" font-weight="600" '
                f'font-family="Segoe UI,Arial,sans-serif">{v:.2f}</text>'
            )

        # X-axis sim label (rotated)
        lx = ml + si * group_w + group_w / 2
        ly = mt + ph + 14
        safe_sim = _html_mod.escape(d["label"])
        lines.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="end" '
            f'transform="rotate(-40,{lx:.1f},{ly:.1f})" font-size="11" '
            f'fill="#1f2937" font-weight="600" '
            f'font-family="Segoe UI,Arial,sans-serif">{safe_sim}</text>'
        )

    # Legend
    legend_y = mt + ph + 64
    for mi, (metric_label, _key, color, role) in enumerate(metrics):
        lx = ml + mi * 170
        lines.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}" rx="2"/>')
        lines.append(
            f'<text x="{lx + 18}" y="{legend_y + 11}" font-size="11" fill="#374151" '
            f'font-family="Segoe UI,Arial,sans-serif">'
            f'{_html_mod.escape(metric_label)} ({role})</text>'
        )

    lines.append('</svg>')
    return "\n".join(lines)


def _build_ranking_summary_table(sims_summary: List[Dict[str, Any]]) -> str:
    """Generate a concise dynamics ranking summary table below the panel grid."""
    rows_data: List[Dict[str, Any]] = []
    for s in sims_summary:
        st = _extract_stats(s["records"])
        rmsd_val = rmsf_val = rg_val = None
        for atype, vals in st.items():
            alow = atype.lower()
            v = _extract_mean_value(vals, alow)
            if v is None:
                continue
            if "rmsd" in alow and rmsd_val is None:
                rmsd_val = v
            elif "rmsf" in alow and rmsf_val is None:
                rmsf_val = v
            elif ("radius" in alow or "gyration" in alow or alow == "rg") and rg_val is None:
                rg_val = v
        rows_data.append({"label": s["label"], "rmsd": rmsd_val, "rmsf": rmsf_val, "rg": rg_val})

    if not any(r["rmsd"] is not None or r["rmsf"] is not None for r in rows_data):
        return ""

    def rank_asc(key: str) -> Dict[str, int]:
        """Rank simulations by *key* ascending (lower = rank 1)."""
        valid = sorted(
            [(r["label"], r[key]) for r in rows_data if r.get(key) is not None],
            key=lambda x: x[1],
        )
        return {lbl: i + 1 for i, (lbl, _) in enumerate(valid)}

    def rank_desc(key: str) -> Dict[str, int]:
        """Rank simulations by *key* descending (higher = rank 1)."""
        valid = sorted(
            [(r["label"], r[key]) for r in rows_data if r.get(key) is not None],
            key=lambda x: x[1],
            reverse=True,
        )
        return {lbl: i + 1 for i, (lbl, _) in enumerate(valid)}

    rmsd_rank = rank_asc("rmsd")    # lower RMSD → more stable
    rmsf_rank = rank_desc("rmsf")   # higher RMSF → more flexible
    rg_rank = rank_asc("rg")        # lower Rg → more compact

    medals: Dict[int, str] = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}

    def medal(r: int) -> str:
        return medals.get(r, f"&nbsp;#{r}")

    table_rows = ""
    for r in rows_data:
        lbl = r["label"]
        rmsd_c = (
            f'{r["rmsd"]:.3f}&thinsp;\u00c5 <small style="color:#6b7280;">'
            f'{medal(rmsd_rank.get(lbl, 99))}</small>'
            if r["rmsd"] is not None else "\u2014"
        )
        rmsf_c = (
            f'{r["rmsf"]:.3f}&thinsp;\u00c5 <small style="color:#6b7280;">'
            f'{medal(rmsf_rank.get(lbl, 99))}</small>'
            if r["rmsf"] is not None else "\u2014"
        )
        rg_c = (
            f'{r["rg"]:.3f}&thinsp;\u00c5 <small style="color:#6b7280;">'
            f'{medal(rg_rank.get(lbl, 99))}</small>'
            if r["rg"] is not None else "\u2014"
        )
        if r["rmsd"] is not None and rmsd_rank.get(lbl) == 1:
            verdict = '<span style="color:#059669;font-weight:700;">Most Stable</span>'
        elif r["rmsf"] is not None and rmsf_rank.get(lbl) == 1:
            verdict = '<span style="color:#f59e0b;font-weight:700;">Most Flexible</span>'
        elif r["rg"] is not None and rg_rank.get(lbl) == 1:
            verdict = '<span style="color:#6366f1;font-weight:700;">Most Compact</span>'
        else:
            verdict = "\u2014"
        table_rows += (
            f"<tr>"
            f"<td><strong>{_html_mod.escape(lbl)}</strong></td>"
            f"<td>{rmsd_c}</td>"
            f"<td>{rmsf_c}</td>"
            f"<td>{rg_c}</td>"
            f"<td>{verdict}</td>"
            f"</tr>\n"
        )

    return (
        '<h3>&#127942; Dynamics Ranking Summary</h3>\n'
        '<table style="font-size:13.5px;">\n'
        '<tr><th>Simulation</th>'
        '<th>Mean RMSD&nbsp;&#8595; (stability)</th>'
        '<th>Mean RMSF&nbsp;&#8593; (flexibility)</th>'
        '<th>Mean Rg&nbsp;&#8595; (compactness)</th>'
        '<th>Verdict</th></tr>\n'
        + table_rows
        + '</table>\n'
    )


def _find_dccm_apo_holo_panel_plots(overlay_plots: List[str]) -> List[str]:
    """Return only the Apo | Holo | ΔDCCM triptych panel figures (no redundant heatmaps)."""
    panels = [
        p for p in overlay_plots
        if "dccm_apo_holo" in Path(p).name.lower() and "panels" in Path(p).name.lower()
    ]
    return sorted(panels, key=lambda p: Path(p).name.lower())


def _find_dccm_diff_heatmap_plots(overlay_plots: List[str]) -> List[str]:
    """Return combined apo–holo ΔDCCM heatmaps (e.g. ``VRK3_dccm_diff_heatmap.png``)."""
    plots = []
    for p in overlay_plots:
        name = Path(p).name.lower()
        if "dccm" not in name:
            continue
        if "dccm_apo_holo" in name and "panels" in name:
            continue
        if "comparison" in name:
            continue
        # Line plots of the same Δ matrix repeat the heatmap. Keep the heatmap.
        if "heatmap" in name and ("diff" in name or "difference" in name):
            plots.append(p)
    return sorted(plots, key=lambda p: Path(p).name.lower())


def _find_dssp_comparison_plot(overlay_plots: List[str]) -> Optional[str]:
    """Locate the cross-simulation secondary-structure bar chart if present."""
    for p in overlay_plots:
        if "dssp_comparison" in Path(p).name.lower():
            return str(p)
    return None


def _resolve_sim_md_inputs(sim_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (topology, trajectory) paths for a per-simulation directory."""
    hpc = Path(sim_dir) / "hpc"
    if not hpc.is_dir():
        return None, None
    topology = None
    for name in ("md.tpr", "md.gro", "boxed.gro", "complex.gro"):
        candidate = hpc / name
        if candidate.is_file():
            topology = str(candidate.resolve())
            break
    trajectory = None
    for name in ("mdWrap.xtc", "md.xtc", "md.trr"):
        candidate = hpc / name
        if candidate.is_file():
            trajectory = str(candidate.resolve())
            break
    return topology, trajectory


def ensure_per_sim_dssp(
    sim_dirs: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    """Run DSSP for any simulation missing dssp_raw_data.dat."""
    from contextlib import contextmanager
    import os

    @contextmanager
    def _in_dir(path: Path):
        prev = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev)

    try:
        from src.analysis.dssp_analyzer import analyze_secondary_structure
    except ImportError as exc:
        return {"success": False, "error": str(exc), "backfilled": []}

    backfilled: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    for sim_dir, label in zip(sim_dirs, labels):
        analysis_dir = Path(sim_dir) / "analysis"
        raw_path = analysis_dir / "dssp_raw_data.dat"
        if raw_path.is_file() and raw_path.stat().st_size > 0:
            continue
        topology, trajectory = _resolve_sim_md_inputs(sim_dir)
        if not topology or not trajectory:
            skipped.append(label)
            errors.append(f"{label}: no topology/trajectory in {sim_dir}/hpc")
            continue
        analysis_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backfilling DSSP for {label} → {analysis_dir}")
        try:
            with _in_dir(analysis_dir):
                result = analyze_secondary_structure.func(
                    topology_file=topology,
                    trajectory_file=trajectory,
                    output_prefix="dssp",
                    working_dir=str(analysis_dir),
                    create_heatmap=True,
                    create_time_series=False,
                    save_raw_data=True,
                )
            if result.get("success"):
                backfilled.append(label)
            else:
                errors.append(f"{label}: {result.get('error', 'DSSP failed')}")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            logger.warning(f"DSSP backfill failed for {label}: {exc}")

    return {
        "success": bool(backfilled) or not errors,
        "backfilled": backfilled,
        "skipped": skipped,
        "errors": errors,
    }


def generate_dssp_comparison_chart(
    sim_dirs: List[str],
    labels: List[str],
    output_dir: str,
    user_goal: Optional[str] = None,
) -> Optional[str]:
    """Grouped bar chart of activation-loop helix/sheet/coil % → dssp_comparison.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        logger.warning(f"generate_dssp_comparison_chart: missing dependency {exc}")
        return None

    rng = _resolve_dssp_residue_range(user_goal)
    if not rng:
        logger.info(
            "generate_dssp_comparison_chart: no activation-loop residue range in goal; skip"
        )
        return None
    residue_start, residue_end = rng
    helix_vals, sheet_vals, coil_vals, plot_labels = [], [], [], []

    for sim_dir, label in zip(sim_dirs, labels):
        analysis_dir = Path(sim_dir) / "analysis"
        segment_pct = _activation_loop_dssp_percentages(
            analysis_dir, residue_start, residue_end,
        )
        if segment_pct is None:
            continue
        h, s, c = segment_pct
        helix_vals.append(h)
        sheet_vals.append(s)
        coil_vals.append(c)
        plot_labels.append(label)

    if len(plot_labels) < 2:
        logger.info(
            "generate_dssp_comparison_chart: not enough activation-loop DSSP data "
            f"(residues {residue_start}\u2013{residue_end}; need \u22652 sims)"
        )
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    x = np.arange(len(plot_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(plot_labels) * 1.4), 5), dpi=120)
    ax.bar(x - width, helix_vals, width, label="α-Helix", color="#e74c3c", alpha=0.85)
    ax.bar(x, sheet_vals, width, label="β-Sheet", color="#3498db", alpha=0.85)
    ax.bar(x + width, coil_vals, width, label="Coil/Loop", color="#95a5a6", alpha=0.85)
    ax.set_xlabel("Simulation", fontsize=12)
    ax.set_ylabel("Secondary Structure Content (%)", fontsize=12)
    ax.set_title(
        f"Activation-Loop Secondary Structure (residues {residue_start}\u2013{residue_end})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(helix_vals), max(sheet_vals), max(coil_vals)) * 1.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_path = str(Path(output_dir) / "dssp_comparison.png")
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"DSSP comparison chart saved → {out_path}")
        return out_path
    except Exception as exc:
        logger.warning(f"Could not save DSSP comparison chart: {exc}")
        plt.close(fig)
        return None


def run_combined_dssp_analysis(
    sim_dirs: List[str],
    labels: List[str],
    output_dir: str,
    user_goal: Optional[str] = None,
) -> Dict[str, Any]:
    """Backfill missing per-sim DSSP, then write comparison + activation-loop heatmaps."""
    try:
        from agentic.planner.planning_guidelines import detect_requested_metrics
        requested = detect_requested_metrics(user_goal or "")
        if requested is not None and "dssp" not in requested:
            return {
                "success": False,
                "message": "DSSP was not requested in user goal",
                "comparison_plot": None,
                "activation_loop_heatmaps": [],
                "plots": [],
                "backfill": {"backfilled": [], "skipped": labels, "errors": []},
            }
    except ImportError:
        pass

    ensure_result = ensure_per_sim_dssp(sim_dirs, labels)
    comparison = generate_dssp_comparison_chart(
        sim_dirs, labels, output_dir, user_goal=user_goal,
    )
    heatmaps = _collect_activation_loop_dssp_heatmaps(
        sim_dirs, labels, user_goal=user_goal, cache_dir=output_dir,
    )
    plots: List[str] = []
    if comparison:
        plots.append(comparison)
    plots.extend(e["heatmap"] for e in heatmaps)
    return {
        "success": bool(comparison or heatmaps),
        "comparison_plot": comparison,
        "activation_loop_heatmaps": heatmaps,
        "plots": plots,
        "backfill": ensure_result,
    }


def _resolve_dssp_residue_range(user_goal: Optional[str]) -> Optional[Tuple[int, int]]:
    """Return 1-based residue range for activation-loop DSSP heatmaps.

    No hardcoded 150–200 fallback — that range is kinase-campaign specific.
    When the goal does not name a segment, callers should skip segment plots.
    """
    try:
        from src.analysis.combined_analysis import parse_rmsf_segments_from_goal
        segments = parse_rmsf_segments_from_goal(user_goal or "")
    except Exception:
        segments = []
    if segments:
        seg = segments[0]
        return int(seg["residue_start"]), int(seg["residue_end"])
    return None


def _load_dssp_raw_matrix(raw_path: Path) -> Optional["Any"]:
    """Load DSSP assignments from dssp_raw_data.dat into a (frames × residues) array."""
    try:
        import numpy as np
    except ImportError:
        return None

    n_frames = n_residues = None
    code_to_num = {"H": 1, "B": 2, "E": 3, "G": 4, "I": 5, "T": 6, "S": 7, "-": 8}
    sparse: Dict[int, Dict[int, int]] = {}

    try:
        with raw_path.open() as fh:
            for line in fh:
                if line.startswith("# Frames:"):
                    m = re.search(r"Frames:\s*(\d+),\s*Residues:\s*(\d+)", line)
                    if m:
                        n_frames, n_residues = int(m.group(1)), int(m.group(2))
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                frame_idx, res_idx, code = int(parts[0]), int(parts[1]), parts[2]
                sparse.setdefault(frame_idx, {})[res_idx] = code_to_num.get(code, 8)
    except Exception as exc:
        logger.warning(f"Could not read DSSP raw data {raw_path}: {exc}")
        return None

    if n_frames is None or n_residues is None:
        if not sparse:
            return None
        n_frames = max(sparse) + 1
        n_residues = max(max(frame.keys()) for frame in sparse.values()) + 1

    matrix = np.full((n_frames, n_residues), 8, dtype=np.int8)
    for frame_idx, res_map in sparse.items():
        for res_idx, val in res_map.items():
            if frame_idx < n_frames and res_idx < n_residues:
                matrix[frame_idx, res_idx] = val
    return matrix


def _compute_dssp_segment_ss_percentages(
    matrix: "Any",
    residue_start: int,
    residue_end: int,
) -> Optional[Tuple[float, float, float]]:
    """Return (helix%, sheet%, coil%) for a 1-based residue range from a DSSP matrix."""
    try:
        import numpy as np
    except ImportError:
        return None

    idx_start = max(0, residue_start - 1)
    idx_end = min(matrix.shape[1], residue_end)
    if idx_start >= idx_end:
        return None

    segment = matrix[:, idx_start:idx_end]
    total = segment.size
    if total == 0:
        return None

    helix_pct = float(np.sum(segment == 1) / total * 100.0)  # H
    sheet_pct = float(np.sum(segment == 3) / total * 100.0)  # E
    coil_pct = 100.0 - helix_pct - sheet_pct
    return helix_pct, sheet_pct, coil_pct


def _activation_loop_dssp_percentages(
    analysis_dir: Path,
    residue_start: int,
    residue_end: int,
) -> Optional[Tuple[float, float, float]]:
    """Compute activation-loop helix/sheet/coil % from per-sim DSSP raw data."""
    raw_path = analysis_dir / "dssp_raw_data.dat"
    if not raw_path.exists():
        return None
    matrix = _load_dssp_raw_matrix(raw_path)
    if matrix is None:
        return None
    return _compute_dssp_segment_ss_percentages(matrix, residue_start, residue_end)


def _generate_dssp_segment_heatmap(
    analysis_dir: Path,
    residue_start: int,
    residue_end: int,
    label: str,
    cache_dir: Path,
) -> Optional[str]:
    """Build or reuse an activation-loop DSSP heatmap PNG for one simulation."""
    slug = _slugify_label(label)
    # Prefer clear names in the combined analysis directory
    cached = cache_dir / f"dssp_activation_loop_{slug}.png"
    legacy = cache_dir / f"{slug}_dssp_segment_{residue_start}_{residue_end}_heatmap.png"
    for existing in (cached, legacy):
        if existing.exists():
            return str(existing)

    segment_candidates = (
        f"dssp_segment_{residue_start}_{residue_end}_heatmap.png",
        "dssp_segment_heatmap.png",
        "dssp_seg_heatmap.png",
    )
    for name in segment_candidates:
        candidate = analysis_dir / name
        if candidate.exists():
            return str(candidate)

    raw_path = analysis_dir / "dssp_raw_data.dat"
    if not raw_path.exists():
        return None

    matrix = _load_dssp_raw_matrix(raw_path)
    if matrix is None:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from src.analysis.dssp_analyzer import DSSP_CODE_MAP
    except ImportError as exc:
        logger.warning(f"DSSP segment heatmap skipped ({label}): {exc}")
        return None

    # User residue numbers are 1-based; DSSP raw indices are 0-based.
    idx_start = max(0, residue_start - 1)
    idx_end = min(matrix.shape[1], residue_end)
    if idx_start >= idx_end:
        return None

    segment = matrix[:, idx_start:idx_end]
    residue_ticks = list(range(residue_start, residue_start + segment.shape[1]))

    cache_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    cmap = plt.get_cmap("tab10", 8)
    im = ax.imshow(
        segment.T, aspect="auto", cmap=cmap,
        interpolation="nearest", vmin=1, vmax=8,
    )
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Residue", fontsize=11)
    ax.set_title(
        f"{label} — Activation loop DSSP (residues {residue_start}\u2013{residue_end})",
        fontsize=12, fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(1, 9))
    cbar.ax.set_yticklabels(
        [DSSP_CODE_MAP[code] for code in ["H", "B", "E", "G", "I", "T", "S", "-"]]
    )
    cbar.set_label("Secondary Structure", fontsize=10)
    if segment.shape[0] > 50:
        ax.set_xticks(np.linspace(0, segment.shape[0] - 1, 8))
    if segment.shape[1] > 10:
        step = max(1, len(residue_ticks) // 8)
        tick_idx = list(range(0, len(residue_ticks), step))
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([str(residue_ticks[i]) for i in tick_idx])
    plt.tight_layout()
    plt.savefig(cached, dpi=200, bbox_inches="tight")
    plt.close()
    return str(cached)


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "sim"


def _collect_activation_loop_dssp_heatmaps(
    sim_dirs: List[str],
    labels: List[str],
    user_goal: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Collect activation-loop DSSP heatmaps (apo + holo), stacked-ready."""
    rng = _resolve_dssp_residue_range(user_goal)
    if not rng:
        return []
    residue_start, residue_end = rng
    cache = Path(cache_dir) if cache_dir else Path(".reporter_cache")

    entries: List[Dict[str, str]] = []
    for sim_dir, label in zip(sim_dirs, labels):
        analysis_dir = Path(sim_dir) / "analysis"
        if not analysis_dir.exists():
            continue
        heatmap = _generate_dssp_segment_heatmap(
            analysis_dir, residue_start, residue_end, label, cache,
        )
        if heatmap:
            entries.append({"label": label, "heatmap": heatmap})

    def _sort_key(entry: Dict[str, str]) -> Tuple[str, int]:
        lbl = entry["label"].lower()
        protein = lbl.replace("_atp_mg", "")
        is_holo = 1 if "_atp_mg" in lbl or "holo" in lbl else 0
        return protein, is_holo

    return sorted(entries, key=_sort_key)


def _build_per_sim_highlights_section(
    sims_summary: List[Dict[str, Any]],
    sim_dirs: List[str],
    user_goal: Optional[str] = None,
    report_focus: Optional[str] = None,
    enriched_prompt: Optional[str] = None,
    exclude_paths: Optional[Set[str]] = None,
    max_highlights: Optional[int] = None,
    allowed_labels: Optional[Set[str]] = None,
    structure_highlight: Optional[Any] = None,
) -> str:
    """Embed goal-selected per-simulation figures (capped for large cohorts).

    When ``allowed_labels`` is given, only those (story-relevant) simulations
    contribute per-sim highlight figures, matching the 3D-viewer subset.
    """
    from src.reporter.figure_selector import (
        ReportFigurePolicy,
        resolve_report_narrative,
        select_per_sim_highlight_figures,
    )
    from src.reporter.html_generator import _collect_image_paths, _encode_image_base64

    # Restrict to the featured simulations when the plan asks for it.
    if allowed_labels:
        filtered = [(s, d) for s, d in zip(sims_summary, sim_dirs)
                    if s.get("label") in allowed_labels]
        if filtered:
            sims_summary = [s for s, _ in filtered]
            sim_dirs = [d for _, d in filtered]

    narrative = resolve_report_narrative(
        user_goal or "",
        report_focus or "",
        enriched_prompt or "",
        n_simulations=len(sims_summary),
    )
    policy = ReportFigurePolicy(max_combined_highlight_figures=max_highlights or 24)
    figures = select_per_sim_highlight_figures(
        sims_summary,
        sim_dirs,
        _collect_image_paths,
        user_goal=user_goal or "",
        report_focus=report_focus or "",
        enriched_prompt=enriched_prompt or "",
        exclude_paths=exclude_paths,
        policy=policy,
        narrative=narrative,
    )

    rationale_map = {}
    if structure_highlight is not None:
        rationale_map = getattr(structure_highlight, "rationale", {}) or {}

    cards: List[str] = []
    for fig in figures:
        search_dirs = [
            Path(fig["path"]).parent,
        ]
        uri = _encode_image_base64(fig["path"], search_dirs)
        if not uri:
            continue
        caption = fig["caption"]
        _story = rationale_map.get(fig.get("label", ""))
        if _story:
            caption = f"{caption} — {_story}"
        fname = Path(fig["path"]).name
        cards.append(
            f'<div class="plot-card">'
            f'<p><strong>{_html_mod.escape(caption)}</strong></p>'
            f'<img src="{uri}" alt="{_html_mod.escape(caption)}" loading="lazy">'
            f'<p class="plot-caption">{_html_mod.escape(fname)}</p>'
            f'</div>\n'
        )

    if not cards:
        return ""

    n_sims = len(sims_summary)
    return (
        "<h2>&#128200; Per-Simulation Key Results</h2>\n"
        f"<p>Selected figures most relevant to the study objective "
        f"({len(cards)} plot(s) from {n_sims} simulation(s); "
        f"see comparative panels above for cross-simulation overlays).</p>\n"
        '<div class="plot-grid">\n'
        + "".join(cards)
        + "</div>\n"
    )


def _collect_per_sim_dssp_figures(
    sim_dirs: List[str],
    labels: List[str],
) -> List[Dict[str, Optional[str]]]:
    """Deprecated alias — kept for callers expecting the old name."""
    return [
        {"label": e["label"], "heatmap": e["heatmap"], "timeseries": None}
        for e in _collect_activation_loop_dssp_heatmaps(sim_dirs, labels)
    ]


def _build_phylo_trees_section(
    overlay_plots: List[str],
    base_analysis_dir: Optional[str] = None,
) -> str:
    """
    Dedicated section for on-request sequence/structure phylogenetic trees.

    These are distinct from the FEL-feature ``classification_phylo_tree`` shown
    in the classification section: they are derived from sequences extracted
    from the input PDBs and from CA structural superposition, respectively.
    """
    specs = (
        (
            "sequence_phylo_tree.png",
            "Sequence-based phylogenetic tree",
            "Unrooted tree from pairwise sequence identity of residues extracted "
            "from every input PDB (UPGMA of 1 − % identity).",
        ),
        (
            "structure_phylo_tree.png",
            "Structure-based phylogenetic tree",
            "Unrooted tree from CA-RMSD after sequence-guided structural "
            "superposition of every pair of input PDBs (UPGMA).",
        ),
    )

    def _locate(name: str) -> Optional[str]:
        for p in overlay_plots or []:
            if Path(p).name == name and Path(p).is_file():
                return str(p)
        if base_analysis_dir:
            cand = Path(base_analysis_dir) / name
            if cand.is_file():
                return str(cand)
        return None

    cards: List[str] = []
    for fname, title, caption in specs:
        path = _locate(fname)
        if not path:
            continue
        uri = _encode_image(path)
        if uri is None:
            continue
        cards.append(
            f'<div class="plot-card">'
            f'<img src="{uri}" alt="{_html_mod.escape(title)}" loading="lazy">'
            f'<p><b>{_html_mod.escape(title)}</b><br>{_html_mod.escape(caption)}</p>'
            f'</div>\n'
        )

    if not cards:
        return ""

    return (
        "<h2>&#127793; Phylogenetic Trees</h2>\n"
        "<p>Cross-simulation relationships inferred directly from the provided "
        "structures. Newick trees and pairwise distance matrices are written "
        "alongside these figures in the analysis directory.</p>\n"
        '<div class="plot-grid">\n' + "".join(cards) + "</div>\n"
    )


def _build_classification_dynamics_section(
    overlay_plots: List[str],
    user_goal: Optional[str] = None,
    report_focus: Optional[str] = None,
    enriched_prompt: Optional[str] = None,
    report_plan: Optional[Any] = None,
    base_analysis_dir: Optional[str] = None,
) -> str:
    """
    Dedicated section for unsupervised classification / cluster-based dynamics.

    Highlights the MDS similarity map, cluster assignments, and
    per-cluster trajectory overlays that distinguish dynamic regimes.
    """
    from src.reporter.figure_selector import (
        partition_combined_overlay_plots,
        resolve_report_narrative,
    )

    plan = report_plan
    if plan and getattr(plan, "classification_section", None):
        cs = plan.classification_section
        if not cs.include or not cs.plot_paths:
            return ""
        parts = partition_combined_overlay_plots(cs.plot_paths)
        intro = cs.summary or ""
        subsection_summaries = getattr(plan, "subsection_summaries", None) or {}
    else:
        narrative = resolve_report_narrative(
            user_goal or "",
            report_focus or "",
            enriched_prompt or "",
        )
        if not narrative.classification_primary and not any(
            Path(p).name.lower().startswith(("classification_", "ref_fel_pock_", "ref_fel_"))
            or "by_cluster" in Path(p).name.lower()
            for p in overlay_plots
        ):
            return ""
        parts = partition_combined_overlay_plots(overlay_plots)
        intro = (
            "Unsupervised grouping of simulations from conformational and binding-site "
            "dynamics features. The MDS map, dendrogram, and cluster overlays summarise "
            "which systems share similar dynamic regimes and how ATP–pocket coupling "
            "differs between clusters."
        )
        subsection_summaries = {}
    summary = parts.get("classification_summary") or []
    trajectories = parts.get("cluster_trajectories") or []
    rmsf = parts.get("cluster_rmsf") or []

    if not (summary or trajectories or rmsf):
        return ""

    def _img_card(path: str, caption: str) -> str:
        uri = _encode_image(path)
        if not uri:
            return ""
        return (
            f'<div class="plot-card">'
            f'<p><strong>{_html_mod.escape(caption)}</strong></p>'
            f'<img src="{uri}" alt="{_html_mod.escape(caption)}" loading="lazy">'
            f'<p class="plot-caption">{_html_mod.escape(Path(path).name)}</p>'
            f'</div>\n'
        )

    def _caption_for(path: str) -> str:
        name = Path(path).stem.replace("_", " ").title()
        if "heatmap" in path.lower():
            return "Z-score feature heatmap by simulation (cluster colors)"
        if "phylo" in path.lower():
            return "Unrooted similarity tree from dynamic features"
        if "pca" in name.lower() and "cluster" in name.lower():
            return "PCA projection coloured by unsupervised cluster assignment"
        if "dendrogram" in name.lower():
            return "Hierarchical clustering dendrogram (protein labels)"
        if "com_distance" in path.lower():
            return "ATP–catalytic pocket COM distance by cluster"
        if "contacts" in path.lower():
            return "Protein–ATP contacts by cluster"
        if "pocket_sasa" in path.lower():
            return "Binding-pocket SASA by cluster"
        if "residence" in path.lower():
            return "Ligand residence / unbinding events by cluster"
        if "pocket_rmsf" in path.lower():
            return "Pocket RMSF profiles by cluster"
        if "ligand_rmsf" in path.lower():
            return "Ligand RMSF profiles by cluster"
        return name

    html_parts: List[str] = [
        "<h2>&#128202; Reference-Structure Classification (FEL + Pocket)</h2>\n"
        if any("ref_fel_pock_" in Path(p).name.lower() for p in (summary + trajectories + rmsf))
        else "<h2>&#128202; Dynamic Classification &amp; Cluster Analysis</h2>\n",
        f"<p>{_html_mod.escape(intro)}</p>\n",
    ]

    def _subsection(title: str, section_id: str, paths: List[str]) -> str:
        if not paths:
            return ""
        summary_text = subsection_summaries.get(section_id, "")
        header = f"<h3>{_html_mod.escape(title)}</h3>\n"
        if summary_text:
            header += f"<p>{_html_mod.escape(summary_text)}</p>\n"
        cards = [_img_card(p, _caption_for(p)) for p in paths]
        cards = [c for c in cards if c]
        if not cards:
            return ""
        return header + '<div class="plot-grid">\n' + "".join(cards) + "</div>\n"

    html_parts.append(_subsection("Cluster assignment summary", "classification_summary", summary))
    html_parts.append(_subsection(
        "Cluster-resolved binding-site trajectories",
        "cluster_trajectories",
        trajectories,
    ))
    html_parts.append(_subsection(
        "Cluster-resolved flexibility (RMSF)",
        "cluster_rmsf",
        rmsf,
    ))

    body = "".join(html_parts)
    if body.count("<img ") == 0:
        return ""
    return body


def _build_comparative_dynamics_section(
    overlay_plots: List[str],
    sims_summary: List[Dict[str, Any]],
    sim_dirs: List[str],
    labels: List[str],
    user_goal: Optional[str] = None,
    cache_dir: Optional[str] = None,
    report_focus: Optional[str] = None,
    enriched_prompt: Optional[str] = None,
) -> str:
    """Build the Comparative Dynamics Summary multi-panel HTML section.

    Only panels relevant to the user goal (with data) are included.
    Cluster-level binding metrics supersede redundant global COM overlays.
    """
    from src.reporter.figure_selector import panel_is_relevant, resolve_report_narrative

    narrative = resolve_report_narrative(
        user_goal or "",
        report_focus or "",
        enriched_prompt or "",
        n_simulations=len(labels),
    )

    def _panel(
        panel_id: str,
        title: str,
        subtitle: str,
        body: str,
    ) -> str:
        return (
            f'<div class="dyn-panel panel-full">'
            f'<div class="panel-header">'
            f'<span class="panel-badge">Panel {_html_mod.escape(panel_id)}</span>'
            f'<span class="panel-title">{_html_mod.escape(title)}</span>'
            f'</div>'
            f'<div class="panel-subtitle">{subtitle}</div>'
            f'<div class="panel-body">{body}</div>'
            f'</div>'
        )

    def _img(fpath: str, alt: str) -> str:
        uri = _encode_image(fpath)
        if uri is None:
            return f'<p class="panel-missing">Image not available: {_html_mod.escape(Path(fpath).name)}</p>'
        return (
            f'<img src="{uri}" alt="{_html_mod.escape(alt)}" loading="lazy">'
        )

    def _img_stack(paths: List[str], alt_prefix: str = "") -> str:
        if not paths:
            return ""
        parts = ['<div class="panel-stack">']
        for p in paths:
            label = Path(p).stem.replace("_", " ").title()
            parts.append(
                f'<div class="panel-stack-item">{_img(p, f"{alt_prefix} {label}".strip())}</div>'
            )
        parts.append("</div>")
        return "\n".join(parts)

    def _missing(msg: str = "No data available") -> str:
        return f'<div class="panel-missing"><span>&#128202;</span><br>{_html_mod.escape(msg)}</div>'

    # ── Panel A: RMSD overlay ────────────────────────────────────────────
    rmsd_path = _find_overlay_by_type(overlay_plots, "rmsd")
    panel_a = _panel(
        "A", "RMSD Overlay",
        "Root-mean-square deviation from reference structure across all simulations",
        _img(rmsd_path, "RMSD overlay") if rmsd_path else _missing("No RMSD overlay found"),
    )

    # ── Panel B: RMSF apo/holo per protein + segment bar plots ──────────
    rmsf_apo_holo_plots = [
        p for p in overlay_plots
        if "rmsf_apo_holo" in Path(p).name.lower()
    ]
    rmsf_path = _find_overlay_by_type(overlay_plots, "rmsf")
    rmsf_segment_plots = sorted(
        [p for p in overlay_plots if "rmsf_segment" in Path(p).name.lower()],
        key=lambda p: Path(p).name.lower(),
    )
    if rmsf_apo_holo_plots or rmsf_path or rmsf_segment_plots:
        rmsf_body = ""
        if rmsf_apo_holo_plots:
            summary_ah = next(
                (p for p in rmsf_apo_holo_plots if p.endswith("rmsf_apo_holo_comparison.png")),
                rmsf_apo_holo_plots[0],
            )
            rmsf_body += (
                '<p style="font-size:12px;color:#6b7280;margin:0 0 6px 0;">'
                'Full-length per-residue RMSF — apo vs holo (one panel per pseudokinase)</p>'
                + _img(summary_ah, "RMSF apo vs holo")
            )
        elif rmsf_path:
            rmsf_body += _img(rmsf_path, "RMSF overlay")
        if rmsf_segment_plots:
            if rmsf_apo_holo_plots or rmsf_path:
                rmsf_body += (
                    '<hr style="border:none;border-top:1px dashed #bfdbfe;'
                    'margin:16px 0 12px;">'
                )
            rmsf_body += (
                '<p style="font-weight:700;color:#1e40af;margin:0 0 10px;">'
                'Activation-loop RMSF (residues 150\u2013200) \u2014 apo vs holo</p>'
                + _img_stack(rmsf_segment_plots, "RMSF segment")
            )
    else:
        rmsf_body = _missing("No RMSF overlay found")
    panel_b = _panel(
        "B", "RMSF Apo vs Holo & Segments",
        "Per-residue C\u03b1 flexibility by pseudokinase and active-site segment comparisons",
        rmsf_body,
    )

    # ── Panel C: Rg comparison ───────────────────────────────────────────
    rg_path = _find_overlay_by_type(overlay_plots, "rg")
    if not rg_path:
        rg_path = _find_overlay_by_type(overlay_plots, "gyration")
    panel_c = _panel(
        "C", "Radius of Gyration (Rg)",
        "Structural compactness over simulation time — all simulations",
        _img(rg_path, "Rg overlay") if rg_path else _missing("No Rg overlay found"),
    )

    # ── Panel D: ATP–catalytic pocket COM distance ───────────────────────
    com_overlay = _find_overlay_by_type(overlay_plots, "com_distance")
    com_by_cluster = _find_overlay_by_type(overlay_plots, "com_distance_by_cluster")
    pocket_plots: List[Tuple[str, str]] = []
    skip_com_panel = narrative.prefer_cluster_views and bool(com_by_cluster)
    if skip_com_panel:
        pocket_body = ""
    elif com_overlay:
        pocket_body = (
            '<p style="font-size:12px;color:#6b7280;margin:0 0 6px 0;">'
            'Combined overlay — apo vs holo (holo systems with bound ATP)</p>'
            + _img(com_overlay, "COM distance overlay")
        )
    else:
        pocket_plots = _find_per_sim_plots(sim_dirs, labels, "pocket_distance")
        if not pocket_plots:
            pocket_plots = _find_per_sim_plots(sim_dirs, labels, "ligand_pocket")
        if pocket_plots:
            pocket_body = _img_stack([p for _, p in pocket_plots], "Pocket distance")
        else:
            pocket_body = _missing("No ATP–pocket COM distance data found")
    has_com_data = bool(not skip_com_panel and (com_overlay or pocket_plots))

    panel_d = _panel(
        "D", "ATP–Catalytic Pocket COM Distance",
        "Center-of-mass distance between ATP and the catalytic pocket",
        pocket_body,
    )

    # ── Panel E: Apo | Holo | ΔDCCM (triptychs, diff heatmaps, or per-sim) ──
    dccm_panel_plots = _find_dccm_apo_holo_panel_plots(overlay_plots)
    dccm_diff_plots = _find_dccm_diff_heatmap_plots(overlay_plots)
    dccm_ov = _find_overlay_by_type(overlay_plots, "dccm")
    dccm_per_sim: List[str] = []
    if not (dccm_panel_plots or dccm_diff_plots or dccm_ov):
        for _, pth in _find_per_sim_plots(sim_dirs, labels, "dccm_heatmap"):
            if pth:
                dccm_per_sim.append(pth)
        if not dccm_per_sim:
            for _, pth in _find_per_sim_plots(sim_dirs, labels, "dccm"):
                if pth and pth not in dccm_per_sim:
                    dccm_per_sim.append(pth)

    if dccm_panel_plots:
        dccm_body = (
            '<p style="font-size:12px;color:#6b7280;margin:0 0 10px;">'
            'Each row: Apo DCCM | Holo DCCM | \u0394(holo \u2212 apo)</p>'
            + _img_stack(dccm_panel_plots, "DCCM apo holo delta")
        )
    elif dccm_diff_plots:
        dccm_body = (
            '<p style="font-size:12px;color:#6b7280;margin:0 0 10px;">'
            'Apo\u2013holo \u0394DCCM heatmaps (one figure per protein)</p>'
            + _img_stack(dccm_diff_plots, "DCCM difference")
        )
    elif dccm_ov:
        dccm_body = _img(dccm_ov, "DCCM heatmap")
    elif dccm_per_sim:
        dccm_body = (
            '<p style="font-size:12px;color:#6b7280;margin:0 0 10px;">'
            'Per-simulation DCCM heatmaps</p>'
            + _img_stack(dccm_per_sim, "DCCM heatmap")
        )
    else:
        dccm_body = _missing("No DCCM apo/holo panels or difference heatmaps found")
    has_dccm_data = bool(
        dccm_panel_plots or dccm_diff_plots or dccm_ov or dccm_per_sim
    )

    panel_e = _panel(
        "E", "DCCM Apo | Holo | \u0394DCCM",
        "Dynamic cross-correlation — apo vs holo comparison with difference map (one figure per protein)",
        dccm_body,
    )

    # ── Panel F: Summary bar chart ────────────────────────────────────────
    bar_svg = _build_summary_bar_chart_svg(sims_summary)
    panel_f = _panel(
        "F", "Summary Bar Chart",
        "Stability (RMSD \u2193), flexibility (RMSF \u2191), and compactness (Rg \u2193) rankings",
        bar_svg if bar_svg else _missing("Insufficient statistics for bar chart"),
    )

    # ── Panel G: Secondary structure (DSSP) ───────────────────────────────
    dssp_rng = _resolve_dssp_residue_range(user_goal)
    dssp_comparison = _find_dssp_comparison_plot(overlay_plots)
    if not dssp_comparison and cache_dir:
        candidate = Path(cache_dir) / "dssp_comparison.png"
        if candidate.exists():
            dssp_comparison = str(candidate)
    dssp_heatmaps = _collect_activation_loop_dssp_heatmaps(
        sim_dirs, labels, user_goal=user_goal, cache_dir=cache_dir,
    )
    if dssp_rng and (dssp_comparison or dssp_heatmaps):
        residue_start, residue_end = dssp_rng
        dssp_body = ""
        if dssp_comparison:
            dssp_body += (
                '<p style="font-size:12px;color:#6b7280;margin:0 0 8px 0;">'
                f'Average activation-loop secondary-structure content '
                f'(residues {residue_start}\u2013{residue_end}) across simulations</p>'
                + _img(dssp_comparison, "DSSP comparison")
            )
        if dssp_heatmaps:
            if dssp_comparison:
                dssp_body += (
                    '<hr style="border:none;border-top:1px dashed #bfdbfe;'
                    'margin:16px 0 12px;">'
                )
            dssp_body += (
                f'<p style="font-weight:700;color:#1e40af;margin:0 0 10px;">'
                f'Activation-loop DSSP heatmaps (residues {residue_start}\u2013{residue_end}) '
                f'\u2014 apo and holo</p>'
                + _img_stack([e["heatmap"] for e in dssp_heatmaps], "DSSP activation loop")
            )
        panel_g_caption = (
            f"Cross-simulation comparison and activation-loop heatmaps "
            f"(residues {residue_start}\u2013{residue_end})"
        )
    elif dssp_comparison:
        dssp_body = _img(dssp_comparison, "DSSP comparison")
        panel_g_caption = "Cross-simulation DSSP comparison"
    else:
        dssp_body = _missing(
            "No activation-loop DSSP segment range in goal "
            "(provide resid/segment in the prompt to enable heatmaps)"
        )
        panel_g_caption = "Cross-simulation DSSP (segment range not specified)"

    panel_g = _panel(
        "G", "Secondary Structure (DSSP)",
        panel_g_caption,
        dssp_body,
    )

    # ── Ranking table ─────────────────────────────────────────────────────
    ranking_table = _build_ranking_summary_table(sims_summary)

    def _include_panel(panel_id: str, panel_html: str, has_data: bool) -> bool:
        return panel_is_relevant(
            panel_id,
            user_goal=user_goal or "",
            report_focus=report_focus or "",
            enriched_prompt=enriched_prompt or "",
            has_data=has_data,
            narrative=narrative,
        )

    panels: List[str] = []
    if _include_panel("A", panel_a, bool(rmsd_path)):
        panels.append(panel_a)
    if _include_panel("B", panel_b, bool(rmsf_apo_holo_plots or rmsf_path or rmsf_segment_plots)):
        panels.append(panel_b)
    if _include_panel("C", panel_c, bool(rg_path)):
        panels.append(panel_c)
    if _include_panel("D", panel_d, has_com_data):
        panels.append(panel_d)
    if _include_panel("E", panel_e, has_dccm_data):
        panels.append(panel_e)
    if _include_panel("F", panel_f, bool(bar_svg)):
        panels.append(panel_f)
    if _include_panel("G", panel_g, bool(dssp_comparison or dssp_heatmaps)):
        panels.append(panel_g)

    if not panels:
        return (
            '<h2>&#128202; Comparative Dynamics Summary</h2>\n'
            '<p>No comparative overlay panels matched the study objective. '
            'See per-simulation highlights and statistics below.</p>\n'
            + ranking_table
        )

    return (
        '<h2>&#128202; Comparative Dynamics Summary</h2>\n'
        + (
            '<p>Supplementary cross-simulation panels; cluster-resolved dynamics '
            'are summarised in the classification section above.</p>\n'
            if narrative.classification_primary
            else f'<p>Cross-simulation comparison panels selected for this study objective '
            f'({len(panels)} panel(s)).</p>\n'
        )
        + '<div class="dyn-panel-grid">\n'
        + "\n".join(panels)
        + '</div>\n'
        + ranking_table
    )


# ---------------------------------------------------------------------------
# Public @tool
# ---------------------------------------------------------------------------

@tool
def generate_combined_html_report(
    sim_dirs: List[str],
    labels: List[str],
    overlay_plots: List[str],
    working_dir: str,
    output_file: str = "combined_report.html",
    title: str = "Multi-Simulation Report",
    enriched_prompt: Optional[str] = None,
    user_goal: Optional[str] = None,
    protein_name: Optional[str] = None,
    literature_refs: Optional[List[Dict[str, Any]]] = None,
    literature_review: Optional[str] = None,
    final_impression: Optional[str] = None,
    report_focus: Optional[str] = None,
    report_plan: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate a rich comparison HTML report spanning multiple MD simulations.

    Reads ``analysis_summary.jsonl`` from each simulation's analysis directory,
    embeds all overlay plots, harvests 3D PDB structures and literature
    references from per-sim reporter artefacts, and writes a self-contained
    HTML report with visualisation, literature, and a final combined impression.

    Args:
        sim_dirs: List of per-simulation root directories in the same order
            as *labels* (e.g. ``["multi_run/1A", "multi_run/2B", ...]``).
        labels: Human-readable simulation labels (e.g. ``["1A", "2B", "3C"]``).
        overlay_plots: Absolute or relative paths to overlay plot images
            produced by ``run_combined_analysis`` (or any PNG/JPEG files).
        working_dir: Directory where the HTML report is written.
        output_file: Output filename (default ``"combined_report.html"``).
        title: Report title shown in the browser tab and header.
        enriched_prompt: LLM-enriched technical task description used as
            detailed objectives (shown in a collapsible panel).
        user_goal: Original user goal text — shown prominently as the
            high-level Overall Goal in the objectives section.
        protein_name: Protein / system name (e.g. "CDK2") used in the
            report header, objectives section, and literature relevance ranking.

    Returns:
        Dict with ``success`` and ``output_path``.
    """
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(working_dir) / output_file)

    # ---- Build label → protein name map from user_goal + protein_name ------
    raw_labels = [Path(d).name for d in sim_dirs]
    _combined_text = " ".join(filter(None, [user_goal, enriched_prompt, protein_name]))
    label_name_map = _parse_label_name_map(_combined_text, sim_labels=raw_labels)
    if label_name_map:
        logger.info("Label → protein name map: %s", label_name_map)
        labels = apply_label_name_map(labels, label_name_map)

    from src.reporter.report_curator import build_report_headline

    if report_plan is None:
        from src.reporter.report_curator import build_combined_report_plan
        base_for_plan = str(Path(working_dir).parent / "analysis")
        report_plan = build_combined_report_plan(
            overlay_plots,
            sim_dirs,
            raw_labels,
            user_goal=user_goal or "",
            enriched_prompt=enriched_prompt or "",
            base_analysis_dir=base_for_plan,
            label_name_map=label_name_map,
        )

    if report_plan and getattr(report_plan, "included_overlay_plots", None):
        overlay_plots = list(report_plan.included_overlay_plots)

    report_headline = (
        report_plan.headline
        if report_plan and getattr(report_plan, "headline", None)
        else build_report_headline(raw_labels, label_name_map, user_goal=user_goal or "")
    )

    # ---- Collect data -------------------------------------------------------
    sims_summary = _collect_sim_summaries(sim_dirs, labels, label_name_map)
    sim_resources = _collect_per_sim_resources(sim_dirs, labels, label_name_map)

    stats_html = _build_stats_section(sims_summary)

    base_analysis_dir = str(Path(working_dir).parent / "analysis")

    from src.reporter.figure_selector import (
        collect_paths_for_combined_sections,
        resolve_report_narrative,
    )

    narrative = resolve_report_narrative(
        user_goal or "",
        report_focus or "",
        enriched_prompt or "",
        n_simulations=len(sim_dirs),
    )

    classification_html = _build_classification_dynamics_section(
        overlay_plots,
        user_goal=user_goal,
        report_focus=report_focus,
        enriched_prompt=enriched_prompt,
        report_plan=report_plan,
        base_analysis_dir=base_analysis_dir,
    )

    archetype_findings_html = _build_reference_archetype_findings_section(
        base_analysis_dir=base_analysis_dir,
    )

    phylo_trees_html = _build_phylo_trees_section(
        overlay_plots,
        base_analysis_dir=base_analysis_dir,
    )

    # Comparative Dynamics Summary — multi-panel figure (Panels A–G)
    comparative_html = _build_comparative_dynamics_section(
        overlay_plots, sims_summary, sim_dirs, labels,
        user_goal=user_goal,
        cache_dir=base_analysis_dir if Path(base_analysis_dir).is_dir() else str(Path(working_dir) / ".dssp_cache"),
        report_focus=report_focus,
        enriched_prompt=enriched_prompt,
    )

    # The "Additional Comparison Figures" section is intentionally omitted:
    # all comparison/composite figures already live in the Comparative Dynamics
    # Summary, so a supplementary figure dump would only duplicate them.
    extra_plots_section = ""

    # 3D viewer: significant structures per simulation (FEL basins preferred).
    # A StructureHighlightPlan (goal/data/LLM-driven) decides which proteins get
    # multiple metastates and which are skipped, so the viewer tells a story
    # instead of dumping one structure per simulation.
    structure_highlight = getattr(report_plan, "structure_highlight", None)
    pdb_frames = _select_significant_pdbs_for_combined(
        sim_dirs,
        labels,
        label_name_map=label_name_map,
        max_per_sim=2,
        max_total=max(len(sim_dirs) * 2, 16),
        user_goal=user_goal,
        enriched_prompt=enriched_prompt,
        highlight_plan=structure_highlight,
        base_analysis_dir=base_analysis_dir,
    )
    viewer_html = _build_3d_viewer_html(pdb_frames) if pdb_frames else ""

    # Paths already shown in classification / comparative sections — exclude from per-sim highlights
    comparative_paths = collect_paths_for_combined_sections(overlay_plots, narrative)

    _highlight_labels = None
    if structure_highlight is not None and getattr(structure_highlight, "restrict_highlights", False):
        _highlight_labels = set(structure_highlight.highlight_labels)

    per_sim_highlights_html = _build_per_sim_highlights_section(
        sims_summary,
        sim_dirs,
        user_goal=user_goal,
        report_focus=report_focus,
        enriched_prompt=enriched_prompt,
        exclude_paths=comparative_paths,
        allowed_labels=_highlight_labels,
        structure_highlight=structure_highlight,
    )

    # Literature: prefer freshly searched refs; fall back to per-sim HTML aggregation
    if literature_refs:
        agg_refs = literature_refs
    else:
        agg_refs = _aggregate_literature(
            sim_resources,
            max_refs=10,
            hypothesis_text=enriched_prompt or user_goal,
            protein_name=protein_name,
            user_goal=user_goal,
        )
    # Sync citations → references (cited-only, first-appearance order) if caller
    # did not already prune. Safe no-op when already synced.
    try:
        from src.reporter.literature_search import sync_cited_literature

        final_text_pre = (
            final_impression.strip()
            if final_impression and final_impression.strip()
            else ""
        )
        agg_refs, literature_review, synced_final = sync_cited_literature(
            agg_refs,
            literature_review=literature_review,
            final_impression=final_text_pre or None,
        )
        if synced_final is not None:
            final_impression = synced_final
    except Exception as exc:
        logger.debug("combined literature citation sync skipped: %s", exc)

    literature_html = _build_literature_html(agg_refs, literature_review=literature_review)
    references_html = _build_references_section_html(agg_refs)

    # Final impression: prefer LLM synthesis with literature citations; else rule-based.
    final_text = (
        final_impression.strip()
        if final_impression and final_impression.strip()
        else _build_combined_final_impression(sims_summary, sim_resources)
    )
    final_html = _build_final_impression_html(final_text, agg_refs)

    # Task description — original user goal only (no redundant enriched/combined text)
    task_html = _build_task_description_html(
        enriched_prompt="",
        sim_resources=sim_resources,
        user_goal=user_goal,
    )
    science_html = _build_science_status_html(
        working_dir=working_dir,
        base_analysis_dir=base_analysis_dir,
    )

    # ---- Metadata header ----------------------------------------------------
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sims_list_html = ", ".join(f"<b>{_html_mod.escape(s['label'])}</b>" for s in sims_summary)

    # report_headline set above from report_plan / build_report_headline
    # ---- Simulations overview table (enhanced) ------------------------------
    sim_overview_rows = ""
    for s in sims_summary:
        n_records = len(s["records"])
        atypes = ", ".join(
            _html_mod.escape(r.get("analysis_type", "")) for r in s["records"]
            if r.get("analysis_type")
        ) or "—"
        sim_dir_display = _html_mod.escape(Path(s["analysis_dir"]).parent.name)
        sim_overview_rows += (
            f"<tr>"
            f"<td><b>{_html_mod.escape(s['label'])}</b></td>"
            f"<td>{sim_dir_display}</td>"
            f"<td>{n_records}</td>"
            f"<td>{atypes}</td>"
            f"</tr>\n"
        )

    sim_overview_html = (
        "<table>"
        "<tr><th>Label</th><th>Directory</th><th>Analysis Records</th><th>Types</th></tr>\n"
        + sim_overview_rows
        + "</table>\n"
    )

    # ---- 3Dmol.js CDN script tag (always inject when viewer section present) -------
    viewer_script = (
        '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>\n'
        if viewer_html
        else ""
    )

    # ---- Assemble HTML ------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html_mod.escape(report_headline)}</title>
<style>{_CSS}</style>
{viewer_script}</head>
<body>
<div class="container">

<h1>&#129516; {_html_mod.escape(report_headline)}</h1>
<div class="header-meta">
  <p><strong>&#128197; Generated:</strong> {now}</p>
  <p><strong>&#128202; Simulations:</strong> {sims_list_html}</p>
  <p><strong>&#128296; Total analyses:</strong> {sum(len(s['records']) for s in sims_summary)}</p>
  <p><strong>&#129366; PDB structures in viewer:</strong> {len(pdb_frames)}</p>
  <p><strong>&#128218; Literature references:</strong> {len(agg_refs)}</p>
</div>

{task_html}

{science_html}

<div class="section-divider"></div>

<h2>&#128202; Simulations Overview</h2>
{sim_overview_html}

<div class="section-divider"></div>

{archetype_findings_html}

{"<div class='section-divider'></div>" if archetype_findings_html else ""}

{viewer_html}

{"<div class='section-divider'></div>" if viewer_html else ""}

{classification_html}

{"<div class='section-divider'></div>" if classification_html else ""}

{phylo_trees_html}

{"<div class='section-divider'></div>" if phylo_trees_html else ""}

{comparative_html}

{per_sim_highlights_html}

{"<div class='section-divider'></div>" if per_sim_highlights_html else ""}

{extra_plots_section}

<div class="section-divider"></div>

<h2>&#128293; Statistics Comparison</h2>
<div class="stats-section">
{stats_html}
</div>

<div class="section-divider"></div>

{literature_html}

{"<div class='section-divider'></div>" if literature_html else ""}

{final_html}

<div class="section-divider"></div>

{references_html}

</div>
<footer>Generated by SimAgent Multi-Simulation Reporter &nbsp;|&nbsp; {now}</footer>
</body>
</html>
"""

    try:
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Combined HTML report saved → {output_path}")
        return {
            "success": True,
            "output_path": output_path,
            "output_file": output_file,
            "pdb_structures": len(pdb_frames),
            "literature_count": len(agg_refs),
        }
    except Exception as exc:
        logger.error(f"Failed to write combined report: {exc}")
        return {"success": False, "error": str(exc)}

