"""
Shared planning guidelines for the MD planner and downstream field agents.

Keeps intent preservation and standard output naming in one place so master-plan,
per-simulation, and combined planning stay consistent.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Set, Any, List, Sequence

# Canonical per-simulation output basenames (no label prefix).
# Combined analysis collects these from {base}/{label}/analysis/.
# Subset / chain / residue-range analyses use a qualifier: {stem}_{qualifier}.ext
# (see is_metric_output_filename and get_standard_output_filenames_block).
STANDARD_OUTPUT_FILES: Dict[str, Dict[str, str]] = {
    "rmsd": {"data": "rmsd.dat", "plot": "rmsd.png"},
    "rmsf": {"data": "rmsf.dat", "plot": "rmsf.png"},
    "rg": {"data": "gyration.dat", "plot": "gyration.png"},
    "gyration": {"data": "gyration.dat", "plot": "gyration.png"},
    "sasa": {"data": "sasa.dat", "plot": "sasa.png"},
    "energy": {"data": "energy.dat", "plot": "energy.png"},
    "dccm": {"data_prefix": "dccm", "plot": "dccm_heatmap.png"},
    "dssp": {"data_prefix": "dssp", "plot": "dssp.png"},
    "com": {"data": "ligand_pocket_distance.csv", "plot": "ligand_pocket_distance.png"},
    "contacts": {"data": "protein_ligand_contacts.csv", "plot": "protein_ligand_contacts.png"},
    "pocket_sasa": {"data": "pocket_sasa.csv", "plot": "pocket_sasa.png"},
    "residence": {"data": "ligand_residence.csv", "plot": "ligand_residence.png"},
    "pocket_rmsf": {"data": "pocket_rmsf.dat", "plot": "pocket_rmsf.png"},
    "ligand_rmsf": {"data": "ligand_rmsf.dat", "plot": "ligand_rmsf.png"},
    "pca": {"data": "pca_projections.dat", "plot": "pca_pc1_pc2_time.png"},
    "fel": {
        "data": "fel_pc1_pc2_grid.csv",
        "plot": "fel_basins.png",
        "features": "fel_features.json",
    },
    "nearby": {"data": "nearby_residues.json", "table": "nearby_residues.csv"},
    "min_distance": {"data": "min_distance.csv", "plot": "min_distance.png"},
    "hbond_occupancy": {"data": "hbond_occupancy.csv", "plot": "hbond_occupancy.png"},
    "salt_bridge": {"data": "saltbridge_occupancy.csv", "plot": "saltbridge_occupancy.png"},
    "ligand_rmsd": {"data": "ligand_rmsd.dat", "plot": "ligand_rmsd.png"},
    "trajectory_qc": {"data": "trajectory_qc.json"},
    "native_contacts": {"data": "native_contacts.dat", "plot": "native_contacts.png"},
    "backbone_dihedrals": {"data": "backbone_dihedrals.dat", "plot": "backbone_dihedrals.png"},
}

# Extra overall names that also satisfy a metric (e.g. generic two-group COM).
STANDARD_OUTPUT_ALTERNATES: Dict[str, Dict[str, tuple[str, ...]]] = {
    "com": {
        "data": ("com_distance.csv",),
        "plot": ("com_distance.png",),
    },
}

# Canonical built-in analysis tools that satisfy each metric group.
# Used by the planner to avoid false "missing tool" → programmer invocations.
METRIC_TO_CANONICAL_TOOLS: Dict[str, tuple[str, ...]] = {
    "com": ("calculate_ligand_pocket_distance", "calculate_com_distance"),
    "contacts": ("calculate_protein_ligand_contacts",),
    "pocket_sasa": ("calculate_pocket_sasa",),
    "residence": ("analyze_ligand_residence",),
    "pocket_rmsf": ("calculate_pocket_rmsf",),
    "ligand_rmsf": ("calculate_ligand_rmsf",),
    "pca": ("calculate_trajectory_pca", "plot_pca_projection"),
    "fel": (
        "calculate_free_energy_landscape",
        "analyze_fel_landscape_features",
        "export_fel_basin_structures",
    ),
    "rmsd": ("calculate_rmsd",),
    "rmsf": ("calculate_rmsf",),
    "rg": ("calculate_radius_of_gyration",),
    "sasa": ("calculate_sasa",),
    "energy": ("analyze_energy",),
    "dccm": ("calculate_dccm",),
    "dssp": ("analyze_secondary_structure",),
    "nearby": ("identify_nearby_residues",),
    "min_distance": ("calculate_min_heavy_atom_distance",),
    "hbond_occupancy": ("calculate_hbond_occupancy",),
    "salt_bridge": ("calculate_salt_bridge_distances",),
    "ligand_rmsd": ("calculate_ligand_rmsd",),
    "trajectory_qc": ("run_trajectory_qc",),
    "native_contacts": ("calculate_native_contacts",),
    "backbone_dihedrals": ("calculate_backbone_dihedrals",),
    "consensus_torsions": (
        "calculate_consensus_torsions",
        "run_consensus_torsions_batch",
    ),
    "dihedral_pca": ("run_independent_dynamics_fel",),
    "dihedral_tica": ("run_independent_dynamics_fel",),
    "cart_pca": ("run_independent_dynamics_fel", "calculate_trajectory_pca"),
    "cart_tica": ("run_independent_dynamics_fel",),
    "shared_dihedral_pca": (
        "fit_dynamics_model",
        "project_dynamics_model",
        "run_shared_dynamics_fel_batch",
    ),
    "shared_dihedral_tica": (
        "fit_dynamics_model",
        "project_dynamics_model",
        "run_shared_dynamics_fel_batch",
    ),
    "shared_cart_pca": (
        "fit_dynamics_model",
        "project_dynamics_model",
        "fit_reference_pca_model",
        "project_simulations_reference_pca",
    ),
    "shared_cart_tica": (
        "fit_dynamics_model",
        "project_dynamics_model",
        "run_shared_dynamics_fel_batch",
    ),
    "consensus_rmsf": ("calculate_consensus_rmsf_features",),
    "consensus_dccm": ("calculate_consensus_dccm_features",),
}


def metric_covered_by_registry(metric: str, existing_tool_names: Set[str]) -> bool:
    """Return True when any canonical tool for *metric* is already registered."""
    canonical = METRIC_TO_CANONICAL_TOOLS.get(metric, ())
    if not canonical:
        return False
    lower = {n.lower() for n in existing_tool_names}
    return any(tool.lower() in lower for tool in canonical)


def partition_metrics_by_registry(
    metrics: FrozenSet[str],
    existing_tool_names: Set[str],
) -> tuple[FrozenSet[str], FrozenSet[str]]:
    """Split metrics into (covered, missing) relative to the tool registry."""
    covered: Set[str] = set()
    missing: Set[str] = set()
    for metric in metrics:
        if metric_covered_by_registry(metric, existing_tool_names):
            covered.add(metric)
        elif metric in METRIC_TO_CANONICAL_TOOLS:
            missing.add(metric)
    return frozenset(covered), frozenset(missing)


def metric_output_stems(metric: str) -> Set[str]:
    """Canonical and alternate filename stems that belong to *metric*."""
    stems: Set[str] = {metric.lower()}
    spec = STANDARD_OUTPUT_FILES.get(metric) or {}
    for key in ("data", "plot", "table"):
        value = spec.get(key)
        if value:
            stems.add(Path(str(value)).stem.lower())
    prefix = spec.get("data_prefix")
    if prefix:
        stems.add(str(prefix).lower())
    alts = STANDARD_OUTPUT_ALTERNATES.get(metric) or {}
    for values in alts.values():
        for value in values:
            stems.add(Path(str(value)).stem.lower())
    if metric == "rg":
        stems.add("gyration")
    if metric == "com":
        stems.update({"com_distance", "ligand_pocket_distance"})
    if metric == "min_distance":
        stems.update({"min_distance", "min_heavy_atom_distance"})
    if metric == "nearby":
        stems.add("nearby_residues")
    if metric == "hbond_occupancy":
        stems.update({"hbond_occupancy", "hbond_occupancy_count", "hbond_count"})
    if metric == "salt_bridge":
        stems.update({"saltbridge_occupancy", "saltbridge_distances", "salt_bridge"})
    return stems


def is_metric_output_filename(filename: str, requested: FrozenSet[str] | Set[str]) -> bool:
    """
    True when *filename* is an overall or qualified output for a requested metric.

    Overall: ``rmsf.dat``, ``rmsd.png``.
    Qualified: ``rmsf_1to34.dat``, ``rmsd_B_1to34.png``, ``com_distance_B1to34_vs_nearbyA.csv``.
    """
    if not filename:
        return False
    stem = Path(str(filename)).stem.lower()
    for metric in requested:
        for canon in metric_output_stems(metric):
            if stem == canon or stem.startswith(canon + "_"):
                return True
    return False


def allowed_output_files_for_metrics(
    requested: FrozenSet[str] | Set[str],
) -> tuple[Set[str], Set[str]]:
    """Overall data/plot basenames (plus alternates) for *requested* metrics."""
    data: Set[str] = set()
    plots: Set[str] = {"combined_metrics.png"}
    for metric in requested:
        spec = STANDARD_OUTPUT_FILES.get(metric) or {}
        if "data" in spec:
            data.add(spec["data"])
        if "table" in spec:
            data.add(spec["table"])
        if "plot" in spec:
            plots.add(spec["plot"])
        alts = STANDARD_OUTPUT_ALTERNATES.get(metric) or {}
        data.update(alts.get("data", ()))
        plots.update(alts.get("plot", ()))
    return data, plots


def get_planner_metric_tool_reference(
    metrics: Optional[FrozenSet[str]] = None,
) -> str:
    """Compact metric → tool map for planner prompts (reduces false missing-tool claims)."""
    lines = [
        "**METRIC → BUILT-IN TOOL MAP (check this before declaring a tool missing):**",
        "",
        "| Metric | Tool(s) in registry |",
        "|--------|---------------------|",
    ]
    show = sorted(metrics) if metrics else sorted(METRIC_TO_CANONICAL_TOOLS.keys())
    for metric in show:
        tools = METRIC_TO_CANONICAL_TOOLS.get(metric)
        if not tools:
            continue
        lines.append(f"| {metric} | `{', '.join(tools)}` |")
    lines.append("")
    lines.append(
        "If a metric maps to a tool above, that tool IS available — do NOT request "
        "programmer creation for it."
    )
    return "\n".join(lines)


_METRIC_PATTERNS: Dict[str, tuple[str, ...]] = {
    "rmsd": (r"\brmsd\b", r"root mean square deviation"),
    "rmsf": (r"\brmsf\b", r"root mean square fluctuation"),
    "rg": (r"\brg\b", r"radius of gyration", r"\bgyration\b"),
    "sasa": (r"\bsasa\b", r"solvent accessible surface"),
    "dccm": (r"\bdccm\b", r"cross[-\s]?correlation", r"correlated motion"),
    "dssp": (r"\bdssp\b", r"secondary[-\s]?structure"),
    "com": (
        r"\bcom\b",
        r"center[-\s]?of[-\s]?mass",
        r"centre[-\s]?of[-\s]?mass",
        r"ligand[-\s]?pocket[-\s]?distance",
        r"pocket[-\s]?distance",
        r"atp[-\s]?(?:to[-\s]?)?(?:protein[-\s]?)?(?:pocket[-\s]?)?distance",
        r"how far\s+(?:atp|the\s+ligand|ligand)",
        r"atp\s+stays\s+from\s+the\s+pocket",
    ),
    "contacts": (
        r"protein[-\s]?ligand[-\s]?contact",
        r"ligand[-\s]?(?:protein[-\s]?)?contacts?",
        r"(?:atp|adp|gtp)[-\s]?contacts?",
        r"protein[–—-]atp\s+contacts?",
    ),
    "hbond_occupancy": (
        r"h[-\s]?bond\s+occupancy",
        r"hydrogen[-\s]?bond\s+occupancy",
        r"interface\s+h[-\s]?bonds?",
        r"protein[-\s]?protein\s+h[-\s]?bonds?",
        r"h[-\s]?bonds?\s+between\s+chain",
        r"hydrogen[-\s]?bonds?\s+between",
        r"interaction\s+partners?",
        r"important\s+residues?\s+and\s+interaction",
    ),
    "salt_bridge": (
        r"salt[-\s]?bridges?",
        r"charged[-\s]?pair",
        r"electrostatic\s+(?:pair|interaction)",
    ),
    "ligand_rmsd": (
        r"ligand[-\s]?rmsd",
        r"rmsd\s+of\s+(?:the\s+)?ligand",
        r"ligand\s+root\s+mean\s+square",
    ),
    "trajectory_qc": (
        r"trajectory[-\s]?qc",
        r"\bqc\b.*(?:trajectory|traj)",
        r"quality[-\s]?control",
    ),
    "native_contacts": (
        r"native[-\s]?contacts?",
        r"fraction\s+of\s+native",
    ),
    "backbone_dihedrals": (
        r"backbone[-\s]?dihedral",
        r"\bphi\b.*\bpsi\b",
        r"\bpsi\b.*\bphi\b",
        r"ramachandran",
    ),
    "pocket_sasa": (
        r"pocket\s+sasa",
        r"binding[-\s]?site\s+sasa",
        r"pocket\s+solvent",
        r"pocket\s+accessibility",
    ),
    "residence": (
        r"residence\s+time",
        r"\bresidence\b",
        r"\bunbinding\b",
        r"\brebinding\b",
        r"bound\s+fraction",
        r"fraction\s+bound",
    ),
    "pocket_rmsf": (
        r"pocket\s+rmsf",
        r"binding[-\s]?site\s+rmsf",
        r"pocket\s+flexibility",
        r"pocket\b[^.\n]{0,40}\brmsf\b",
        r"rmsf\b[^.\n]{0,40}\bpocket\b",
    ),
    "ligand_rmsf": (
        r"ligand\s+rmsf",
        r"atp\s+rmsf",
        r"ligand\s+flexibility",
        r"ligand\b[^.\n]{0,40}\brmsf\b",
        r"rmsf\b[^.\n]{0,40}\bligand\b",
    ),
    "nearby": (
        r"residues?\s+within",
        r"within\s+\d+(?:\.\d+)?\s*(?:Å|a|angstrom)(?!\s+of\s+(?:the\s+)?(?:ligand|atp|adp|inhibitor))",
        r"nearby\s+resid",
        r"neighbouring\s+resid",
        r"neighboring\s+resid",
        r"identify\s+all\s+chain",
    ),
    "min_distance": (
        r"min(?:imum)?\s+(?:heavy[-\s]?atom\s+)?distance",
        r"minimum\s+heavy[-\s]?atom",
        r"closest[-\s]?approach",
        r"min(?:imum)?\s+distance\s+between",
    ),
    # Avoid bare "energy" (force field, free-energy landscape, binding energy).
    "energy": (
        r"potential\s+energy",
        r"analyze_energy",
        r"\bedr\b",
        r"energy\.dat",
        r"\benergy\s+(?:analysis|overlay|plot|profile|decomposition|timeseries)\b",
        r"\b(?:total|kinetic)\s+energy\b",
        r"\benergy\s+and\s+(?:rmsd|rmsf|rg)\b",
    ),
    "pca": (
        r"\bpca\b",
        r"principal component",
        r"essential dynamics",
        r"collective motion",
    ),
    "fel": (
        r"free[-\s]?energy landscape",
        r"\bfel\b",
        r"energy landscape",
        r"conformational landscape",
        r"landscape entropy",
        r"\bbasin",
        r"\bminima",
    ),
    "consensus_torsions": (
        r"consensus\s+(?:torsions?|dihedrals?)",
        r"\bchi\s*1\b",
        r"\bχ\s*₁\b",
        r"\bχ1\b",
        r"side[-\s]?chain\s+χ",
        r"\bφ\b.*\bψ\b",
        r"phi\s*/\s*psi\s*/\s*chi",
        r"pocket\s+chi\s*1",
        r"circular\s+mean.*chi",
        r"χ1\s+preference",
    ),
    "dihedral_pca": (
        r"dihedral\s+pca",
        r"torsion(?:al)?\s+pca",
        r"independent\s+dihedral\s+pca",
        r"pca\s+on\s+(?:torsions?|dihedrals?)",
        r"pca_grid_entropy",
    ),
    "dihedral_tica": (
        r"dihedral\s+tica",
        r"torsion(?:al)?\s+tica",
        r"tica\s+on\s+(?:torsions?|dihedrals?)",
        r"tica_grid_entropy",
    ),
    "cart_pca": (
        r"cartesian\s+pca",
        r"c[\s-]?alpha\s+pca",
        r"cα\s+pca",
        r"cart_pca",
    ),
    "cart_tica": (
        r"cartesian\s+tica",
        r"c[\s-]?alpha\s+tica",
        r"cα\s+tica",
        r"cart_tica",
    ),
    "shared_dihedral_pca": (
        r"shared[-\s]?reference\s+dihedral\s+pca",
        r"shared[-\s]?reference.{0,80}dihedral\s+pca",
        r"pca_pka_ref_shared_dyn",
        r"pca_ref_shared_dyn",
        r"project(?:ion)?\s+onto\s+.*dihedral\s+pca",
        r"reference\s+dihedral\s+pca",
    ),
    "shared_dihedral_tica": (
        r"shared[-\s]?reference\s+dihedral\s+tica",
        r"reference\s+dihedral\s+tica",
        r"pka[-\s]?ref(?:erence)?\s+tica",
    ),
    "shared_cart_pca": (
        r"shared[-\s]?reference\s+(?:cartesian\s+)?pca",
        r"reference\s+pca\s+model",
    ),
    "shared_cart_tica": (
        r"shared[-\s]?reference\s+cartesian\s+tica",
        r"reference\s+cartesian\s+tica",
    ),
    "consensus_rmsf": (
        r"consensus\s+rmsf",
        r"mapped\s+rmsf",
        r"consensus_rmsf_mean",
        r"c[αa]\s+flexibility",
        r"consensus\s+c[αa]",
        r"flexibility\s+across\s+the\s+domain",
    ),
    "consensus_dccm": (
        r"consensus\s+dccm",
        # Lobe names alone are not DCCM: a COM distance between lobes must not
        # schedule a correlation matrix.
        r"n[-\s]?lobe.{0,40}c[-\s]?lobe.{0,40}(?:dccm|correlat)",
        r"(?:dccm|correlat).{0,40}n[-\s]?lobe.{0,40}c[-\s]?lobe",
        r"dccm_N_C",
        r"dccm\s+n\s*[-–—/]\s*c",
        r"mapped\s+dccm",
        r"correlated\s+motion\s+between\s+the\s+n",
    ),
}

_CLASSIFICATION_REQUEST_PATTERNS: tuple[str, ...] = (
    r"\bclassif(y|ication|y\s+proteins?|y\s+systems?)\b",
    r"\bcluster(ing|ed|s)?\b",
    r"\bward\b",
    r"\bdendrogram\b",
    r"feature\s+heatmap",
    r"\bunsupervised\b",
    r"\bfeature\s+(matrix|table|vector)\b",
    r"\bgroup\s+(?:the\s+)?(?:proteins?|systems?|simulations?|kinases?)\b",
    r"\bcompare\s+.*\bfor\s+classification\b",
    r"\bconformational\s+diversity\s+(?:across|comparison)\b",
)

_EXCLUSIVE_PATTERNS = (
    r"\b(?:only|just|specifically|exclusively)\b[^.\n]{0,80}\b(rmsd|rmsf|rg|gyration|sasa|dccm|dssp|energy)\b",
    r"\b(rmsd|rmsf|rg|gyration|sasa|dccm|dssp|energy)\b[^.\n]{0,40}\b(?:only|just)\b",
)

_BROAD_DYNAMICS_PATTERNS = (
    "protein dynamics",
    "dynamic behavior",
    "dynamic behaviour",
    "conformational dynamics",
    "molecular dynamics analysis",
    "comprehensive analysis",
    "full analysis",
)

_NEGATION_START = re.compile(
    r"\b(?:do\s+not|don't|dont|never|must\s+not)\b",
    re.IGNORECASE,
)


def strip_negated_clauses(text: str) -> str:
    """Remove prohibitions before intent detection.

    "Do not compute DCCM" is not a request for DCCM. A prohibition may wrap
    onto the next line ("Do not compute X,\\nY, or Z."). It stops at a period,
    at a colon (so "do not drop any:" does not erase the following list),
    or at a blank line or numbered item.
    """
    if not text:
        return ""
    kept: List[str] = []
    carry = False
    for line in text.splitlines(keepends=True):
        nl = "\n" if line.endswith("\n") else ""
        body = line[:-1] if nl else line
        if carry:
            if not body.strip() or re.match(r"\s*\d+\.\s", body):
                carry = False
            else:
                term = re.search(r"[.!?]", body)
                if term:
                    body = body[term.end():]
                    carry = False
                else:
                    kept.append(nl)
                    continue
        pieces: List[str] = []
        cursor = 0
        while cursor <= len(body):
            match = _NEGATION_START.search(body, cursor)
            if not match:
                pieces.append(body[cursor:])
                break
            pieces.append(body[cursor:match.start()])
            rest = body[match.end():]
            term = re.search(r"[.!?]", rest)
            colon = rest.find(":")
            if term and (colon < 0 or term.start() <= colon):
                cursor = match.end() + term.end()
                continue
            if colon >= 0:
                pieces.append(rest[colon + 1:])
                break
            carry = True
            break
        kept.append("".join(pieces) + nl)
    return "".join(kept)


def collect_goal_texts_for_intent(
    state: Optional[Dict[str, Any]] = None,
    agent_input: Optional[Any] = None,
) -> tuple[str, ...]:
    """
    Gather all user-facing goal strings that may define analysis scope.

    In multisim runs, ``user_goal_original`` holds the CLI --goal text while
    ``user_goal`` holds the per-simulation prompt (which may add sim-specific
    analyses such as ATP COM distance). All non-empty sources are returned.
    """
    texts: list[str] = []
    seen: set[str] = set()
    if state:
        for key in ("user_goal", "user_goal_original"):
            value = (state.get(key) or "").strip()
            if value and value not in seen:
                texts.append(value)
                seen.add(value)
    if agent_input is not None:
        value = (getattr(agent_input, "user_goal", None) or "").strip()
        if value and value not in seen:
            texts.append(value)
            seen.add(value)
    return tuple(texts)


def detect_requested_metrics_union(*goal_texts: str) -> Optional[FrozenSet[str]]:
    """
    Union metrics detected across multiple goal strings.

    Used when master --goal and per-simulation prompts disagree in scope
    (e.g. master says "RMSF for all sims" while a holo sim prompt adds ATP COM).
    """
    merged: Set[str] = set()
    saw_broad = False
    for text in goal_texts:
        if not (text or "").strip():
            continue
        metrics = detect_requested_metrics(text)
        if metrics is None:
            saw_broad = True
        else:
            merged.update(metrics)
    if merged:
        return frozenset(merged)
    return None if saw_broad else None


def _normalize_metric_false_positives(text: str, found: Set[str]) -> None:
    """
    Drop whole-protein metrics when a binding-site metric already covers the phrase.

    Examples:
    - "pocket SASA" must not also enable whole-protein ``sasa``.
    - "free-energy landscape" must not also enable potential ``energy``.
    - "energy minimize" / equilibration wording must not enable energy analysis.
    """
    lower = (text or "").lower()
    if "sasa" in found and (
        "pocket_sasa" in found
        or re.search(r"pocket\s+sasa|binding[-\s]?site\s+sasa", lower)
    ):
        found.discard("sasa")
    if "energy" in found and re.search(r"free[-\s]?energy|energy\s+landscape", lower):
        if not re.search(
            r"potential\s+energy|analyze_energy|\bedr\b|energy\.dat|"
            r"mean\s+potential|energy\s+analysis",
            lower,
        ):
            found.discard("energy")
    if "energy" in found and re.search(
        r"energy\s+minim|minimisation|minimization|minimiz(?:e|ed|ing)|"
        r"equilibrat|force\s*field|amber|charmm|binding\s+energy|"
        r"activation\s+energy|interaction\s+energy",
        lower,
    ):
        if not re.search(
            r"potential\s+energy|analyze_energy|\bedr\b|energy\.dat|"
            r"mean\s+potential|energy\s+analysis|energy\s+overlay",
            lower,
        ):
            found.discard("energy")
    # Bare "energy" leftover with no explicit potential/edr intent → drop
    if "energy" in found and not re.search(
        r"potential\s+energy|analyze_energy|\bedr\b|energy\.dat|"
        r"energy\s+(?:analysis|overlay|plot|profile|decomposition|timeseries)|"
        r"(?:total|kinetic)\s+energy",
        lower,
    ):
        found.discard("energy")
    # Consensus φ/ψ/χ₁ should not also force limited backbone Ramachandran tool
    if "backbone_dihedrals" in found and (
        "consensus_torsions" in found
        or re.search(r"chi\s*1|χ\s*₁|consensus\s+(?:torsion|dihedral)", lower)
    ):
        if not re.search(r"backbone[-\s]?dihedral|ramachandran", lower):
            found.discard("backbone_dihedrals")
    # "consensus RMSF" should not also schedule generic pocket_rmsf
    if "pocket_rmsf" in found and "consensus_rmsf" in found:
        if not re.search(r"pocket\s+rmsf|binding[-\s]?site\s+rmsf", lower):
            found.discard("pocket_rmsf")
    # Independent dihedral PCA implies FEL; prefer modular tools over bare pca
    if ("dihedral_pca" in found or "dihedral_tica" in found) and "pca" in found:
        if not re.search(
            r"cartesian\s+pca|trajectory\s+pca|c[\s-]?alpha\s+pca|calculate_trajectory_pca",
            lower,
        ):
            found.discard("pca")
    # "residues within 15 Å of ATP" defines a pocket cutoff. It is not a
    # request for a separate nearby-residue analysis.
    if "nearby" in found and re.search(
        r"residues?\s+within\s+\d+(?:\.\d+)?\s*(?:å|a|angstrom)?\s+of\s+(?:the\s+)?(?:atp|ligand)",
        lower,
    ):
        if not re.search(
            r"nearby\s+resid|identify\s+all\s+chain|neighbouring\s+resid|neighboring\s+resid",
            lower,
        ):
            found.discard("nearby")
    if "consensus_dccm" in found and "dccm" in found:
        # "N–C DCCM mean correlation" is the consensus feature, not a second
        # full-matrix DCCM analysis. Keep generic DCCM only when a map is asked for.
        if not re.search(r"dccm\s+(?:map|matrix|heatmap|plot|overlay|difference)", lower):
            found.discard("dccm")


def _normalize_binding_rmsf_metrics(text: str, found: Set[str]) -> None:
    """
    Map phrasing like "pocket and ligand RMSF" to specific binding metrics.

    When binding-site RMSF is requested, drop generic whole-protein ``rmsf``
    unless the user explicitly asks for global/per-residue protein RMSF.

    Do NOT treat incidental "ATP"/"ligand" mentions (e.g. setup/holo description)
    as a request for ligand RMSF unless those words appear near "rmsf".
    """
    lower = (text or "").lower()
    if not re.search(r"\brmsf\b", lower) and "rmsf" not in found:
        return

    if re.search(
        r"pocket\s+(?:and\s+ligand\s+)?rmsf|"
        r"binding[-\s]?site\s+rmsf|"
        r"\bpocket\b[^.\n]{0,40}\brmsf\b|"
        r"\brmsf\b[^.\n]{0,40}\bpocket\b",
        lower,
    ):
        found.add("pocket_rmsf")
    if re.search(
        r"ligand\s+(?:and\s+pocket\s+)?rmsf|"
        r"atp\s+rmsf|"
        r"\b(?:ligand|atp)\b[^.\n]{0,40}\brmsf\b|"
        r"\brmsf\b[^.\n]{0,40}\b(?:ligand|atp)\b|"
        r"pocket\s+and\s+ligand\s+rmsf|"
        r"ligand\s+and\s+pocket\s+rmsf",
        lower,
    ):
        found.add("ligand_rmsf")

    binding_rmsf = found & {"pocket_rmsf", "ligand_rmsf"}
    if not binding_rmsf or "rmsf" not in found:
        return

    whole_protein_rmsf = re.search(
        r"(?:whole[-\s]?protein|global|backbone|cα|ca)\s+rmsf|"
        r"rmsf\s+(?:for\s+)?(?:the\s+)?(?:whole\s+)?protein|"
        r"per[-\s]?residue\s+rmsf(?!\s+(?:for|of)\s+(?:pocket|ligand|atp))|"
        r"rmsf\s+between\s+apo|"
        r"rmsf\s+(?:of|for)\s+(?:the\s+)?(?:apo|holo|protein)",
        lower,
    )
    if not whole_protein_rmsf:
        found.discard("rmsf")


def detect_requested_metrics(goal: str) -> Optional[FrozenSet[str]]:
    """
    Return the set of metrics explicitly requested in *goal*.

    Returns None when the goal is broad (e.g. "protein dynamics") and the planner
    may choose a small justified dynamics bundle. Returns a frozen set when the
    user names specific metrics or uses exclusive language ("RMSF only").
    """
    text = (goal or "").lower()
    text = (
        text.replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    text = strip_negated_clauses(text)

    found = {
        metric
        for metric, patterns in _METRIC_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }
    _normalize_binding_rmsf_metrics(text, found)
    _normalize_metric_false_positives(text, found)
    # Normalise rg/gyration
    if "rg" in found or "gyration" in found:
        found.discard("gyration")
        found.add("rg")

    # A shared-reference or dihedral PCA request is not also Cartesian PCA.
    if "dihedral_pca" in found or "shared_dihedral_pca" in found:
        if not re.search(
            r"cartesian\s+pca|trajectory\s+pca|calculate_trajectory_pca|"
            r"c[\s-]?alpha\s+pca",
            text,
        ):
            found.discard("pca")

    for pattern in _EXCLUSIVE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            token = match.group(1)
            if token == "gyration":
                token = "rg"
            # "specifically RMSF … and COM distance" must keep both metrics.
            if found - {token}:
                return frozenset(found)
            return frozenset({token})

    # Named metrics are the request. "Protein dynamics" only leaves the
    # choice open when the goal does not list measurements.
    if found:
        return frozenset(found)

    if any(phrase in text for phrase in _BROAD_DYNAMICS_PATTERNS):
        return None

    return None


_PHYLO_GENERIC_PATTERNS: tuple[str, ...] = (
    r"\bphylogen(?:etic|y|omic)\b",
    r"\bphylo\s*tree\b",
    r"\bevolutionary\s+tree\b",
    r"\bdendrogram\s+of\s+(?:sequences?|structures?)\b",
)
_PHYLO_SEQUENCE_PATTERNS: tuple[str, ...] = (
    r"\bsequence[-\s]*(?:based\s+)?(?:phylogen\w*|tree)\b",
    r"\bsequence\s+alignment\s+tree\b",
    r"\bmsa\b",
)
_PHYLO_STRUCTURE_PATTERNS: tuple[str, ...] = (
    r"\bstructur(?:e|al)[-\s]*(?:based\s+)?(?:phylogen\w*|tree)\b",
    r"\bstructur(?:e|al)\s+(?:similarity|comparison)\s+tree\b",
)


def _normalize_goal_text(*goal_texts: str) -> str:
    parts = []
    for text in goal_texts:
        if not text:
            continue
        parts.append(
            strip_negated_clauses(
                text.lower()
                .replace("\u2011", "-")
                .replace("\u2012", "-")
                .replace("\u2013", "-")
                .replace("\u2014", "-")
            )
        )
    return " ".join(parts)


def detect_phylo_tree_requested(*goal_texts: str) -> Dict[str, bool]:
    """
    Detect whether the user asked for sequence- and/or structure-based
    phylogenetic trees at the combined-analysis level.

    Returns a dict ``{"sequence": bool, "structure": bool}``. A bare
    "phylogenetic tree" request (no sequence/structure qualifier) defaults to a
    sequence tree, which is the conventional meaning.
    """
    text = _normalize_goal_text(*goal_texts)
    if not text:
        return {"sequence": False, "structure": False}

    has_generic = any(re.search(p, text) for p in _PHYLO_GENERIC_PATTERNS)
    seq = any(re.search(p, text) for p in _PHYLO_SEQUENCE_PATTERNS)
    struct = any(re.search(p, text) for p in _PHYLO_STRUCTURE_PATTERNS)

    if has_generic:
        # Qualifiers mentioned anywhere alongside a phylo request.
        mentions_sequence = bool(re.search(r"\bsequences?\b", text))
        mentions_structure = bool(re.search(r"\b(?:structures?|pdb|3d)\b", text))
        seq = seq or mentions_sequence
        struct = struct or mentions_structure
        # Bare "phylogenetic tree" with no qualifier → sequence tree by default.
        if not seq and not struct:
            seq = True

    return {"sequence": bool(seq), "structure": bool(struct)}


_REFERENCE_LANDSCAPE_PATTERNS: tuple[str, ...] = (
    r"reference[-\s]?projected\s+pca",
    r"reference\s+landscape",
    r"shared[-\s]reference\s+fel",
    r"shared\s+fel\b",
    r"shared\s+free[-\s]?energy\s+landscape",
    # Bare "reference FEL" — not "shared-reference FEL" (hyphenated compound).
    r"(?<![-\w])reference\s+fel\b",
    r"reference\s+fel\s+cluster",
    r"run_reference_landscape_pipeline",
    r"build_consensus_sequence_alignment",
    r"build_global_mapped_alignment",
    r"plot_reference_msa_alignment",
    r"plot_global_mapped_alignment",
)

_CONSENSUS_POCKET_PATTERNS: tuple[str, ...] = (
    r"consensus[-\s]?mapped\s+pocket",
    r"consensus\s+pocket",
    r"reference\s+pocket",
    r"mapped\s+pocket",
    r"define\s+(?:the\s+)?(?:atp\s+)?pocket",
    r"as\s+the\s+reference\s+to\s+define\s+(?:the\s+)?(?:atp\s+)?pocket",
    r"run_consensus_pocket_metrics_batch",
    r"define_reference_consensus_pocket",
    r"define_pocket_mapped_residues",
    r"pocket_mapped",
    r"global_mapped",
)

_FAMILY_MODULAR_TOOL_PATTERNS: tuple[str, ...] = (
    r"calculate_consensus_torsions",
    r"calculate_consensus_rmsf_features",
    r"calculate_consensus_dccm_features",
    r"run_independent_dynamics_fel",
    r"independent\s+dihedral\s+pca",
    r"φ\s*/\s*ψ\s*/\s*χ\s*₁",
    r"phi\s*/\s*psi\s*/\s*chi",
)


def score_family_modular_dynamics(*goal_texts: str) -> Dict[str, Any]:
    """Explainable family-modular detector.

    Returns ``family`` (bool), ``score`` (0–6 scientific signals), ``signals``,
    and ``reason``. Explicit modular tool names force ``family=True``.
    Auto threshold is ≥2 signals, or a family phrase plus ≥1 signal.
    A request for comparative plots is not itself a family study.
    """
    text = _normalize_goal_text(*goal_texts)
    empty = {
        "family": False,
        "score": 0,
        "signals": [],
        "tool_hit": False,
        "comparative": False,
        "reason": "empty goal",
    }
    if not text:
        return empty
    tool_hit = any(re.search(p, text) for p in _FAMILY_MODULAR_TOOL_PATTERNS)
    flags = {
        "pocket": bool(
            re.search(
                r"consensus\s+pocket|mapped\s+pocket|reference\s+pocket|"
                r"how far .*(?:atp|ligand)|"
                r"distance of the (?:atp|ligand)|"
                r"(?:atp|ligand)\s+\bcom\b.{0,40}pocket|"
                r"pocket\s+\bcom\b|"
                r"\bcom\b.{0,40}consensus",
                text,
            )
        ),
        "angle": bool(
            re.search(
                r"axis\s+orientation|ligand.*angle|pocket.*angle|orientation of the atp",
                text,
            )
        ),
        "rmsf": bool(
            re.search(
                r"consensus\s+(?:c[αa]|rmsf)|flexibility\s+across\s+the\s+domain|"
                r"flexibility of consensus",
                text,
            )
        ),
        "chi": bool(re.search(r"χ\s*₁|chi\s*1|side[-\s]?chain", text)),
        "dccm": bool(
            re.search(
                r"\bdccm\b|correlated\s+motion|"
                r"n[-\s]?lobe.{0,40}c[-\s]?lobe.{0,40}(?:dccm|correlat)",
                text,
            )
        ),
        "entropy": bool(
            re.search(
                r"landscape\s+entropy|grid\s+entropy|free[-\s]?energy\s+landscape|"
                r"dihedral\s+pca|conformational[-\s]?landscape\s+entropy",
                text,
            )
        ),
    }
    signals = [k for k, v in flags.items() if v]
    score = len(signals)
    comparative = bool(
        re.search(
            r"\bfamily\b|across (?:the )?family|kinase family|"
            r"ward|classification feature|shared protocol",
            text,
        )
    )
    if tool_hit:
        family = True
        reason = "explicit modular tool"
    elif score >= 2:
        family = True
        reason = f"{score} modular signals"
    elif comparative and score >= 1:
        family = True
        reason = "comparative phrasing plus one modular signal"
    else:
        family = False
        reason = f"score {score} below threshold"
    return {
        "family": family,
        "score": score,
        "signals": signals,
        "tool_hit": tool_hit,
        "comparative": comparative,
        "reason": reason,
    }


def detect_family_modular_dynamics_requested(*goal_texts: str) -> bool:
    """True when the goal asks for comparative modular dynamics features.

    Matches scientific descriptions (pocket–ligand COM/orientation, consensus
    flexibility, pocket χ₁, N↔C correlation, dihedral landscape entropy) or
    explicit modular tool names. Threshold is ≥2 signals (or a comparative
    phrase plus one signal). Does **not** encode a fixed cluster count.
    """
    return bool(score_family_modular_dynamics(*goal_texts).get("family"))


def detect_paper_ward4_requested(*goal_texts: str) -> bool:
    """Deprecated alias — use :func:`detect_family_modular_dynamics_requested`.

    Kept for older callers; no longer treats "Ward-4" / fixed-k language as
    special. Returns the family-modular detector result only.
    """
    return detect_family_modular_dynamics_requested(*goal_texts)


def detect_consensus_pocket_requested(*goal_texts: str) -> Dict[str, Any]:
    """
    Detect whether the user asked for consensus-mapped reference pocket metrics.

    Returns ``{"requested": bool, "reference_label": str|None, "pocket_cutoff_A": float}``.
    """
    text = _normalize_goal_text(*goal_texts)
    if not text:
        return {"requested": False, "reference_label": None, "pocket_cutoff_A": 15.0}

    requested = any(re.search(p, text) for p in _CONSENSUS_POCKET_PATTERNS)
    if not requested:
        return {"requested": False, "reference_label": None, "pocket_cutoff_A": 15.0}

    cutoff = 15.0
    m = re.search(r"pocket[_\s-]?cutoff[_\s-]?a?\s*[=:]\s*(\d+(?:\.\d+)?)", text)
    if not m:
        m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*(?:å|a|angstrom)?\s+of\s+atp", text)
    if m:
        cutoff = float(m.group(1))

    return {
        "requested": True,
        "reference_label": _resolve_reference_label(*goal_texts),
        "pocket_cutoff_A": cutoff,
    }


def _parse_reference_label_from_goal(text: str) -> Optional[str]:
    """Extract reference label (e.g. q8nb16) from goal text when present."""
    if not text:
        return None
    lowered = text.lower()
    # UniProt-like: letter + digit + alnum (rejects words like "pocket").
    uid = r"([opq][0-9][a-z0-9]{3,6})"

    # Explicit forms: reference_label=q8nb16, reference label: q8nb16
    m = re.search(
        rf"reference[_\s-]*label\s*[=:]\s*{uid}",
        lowered,
    )
    if m:
        return m.group(1)

    m = re.search(
        rf"reference(?:\s+label|\s+id|\s+pseudokinase|\s+kinase)?\s*[=:]\s*{uid}",
        lowered,
    )
    if m:
        return m.group(1)

    # UniProt-style id immediately after "reference is" / "reference:"
    m = re.search(
        rf"reference\s+(?:is\s+)?{uid}\b",
        lowered,
    )
    if m:
        return m.group(1)

    m = re.search(
        rf"\b{uid}\s+as\s+(?:the\s+)?reference\b",
        lowered,
    )
    if m:
        return m.group(1)

    # Parenthetical: KAPCA (p17612) as the reference
    m = re.search(
        rf"\({uid}\)\s+as\s+(?:the\s+)?reference\b",
        lowered,
    )
    if m:
        return m.group(1)

    # Protein name → common pseudoKin label (MLKL = q8nb16)
    if re.search(r"\bmlkl\b", lowered):
        return "q8nb16"
    if re.search(r"\bkapca\b", lowered):
        return "p17612"
    return None


def _resolve_reference_label(
    *goal_texts: str,
    labels: Optional[Sequence[str]] = None,
    default: str = "",
) -> str:
    """
    Pick a reference simulation label from goal text.

    Prefers ``user_goal_original``-style sources (first non-empty texts)
    before merged combined-plan boilerplate that may contain
    ``reference:\\nSimulations``.
    """
    for text in goal_texts:
        if not text:
            continue
        candidate = _parse_reference_label_from_goal(text)
        if not candidate:
            continue
        if labels is None:
            return str(candidate)
        label_set = {str(l) for l in labels}
        if str(candidate) in label_set:
            return str(candidate)
        # UniProt bare id → ``p17612_ATP`` folder / label
        cand_l = str(candidate).lower()
        for lab in labels:
            lab_s = str(lab)
            lab_l = lab_s.lower()
            if lab_l.startswith(cand_l + "_") or cand_l.startswith(lab_l + "_"):
                return lab_s
            for suf in ("_atp", "_adp", "_amp"):
                if lab_l.endswith(suf) and lab_l[: -len(suf)] == cand_l:
                    return lab_s
        return str(candidate)
    if default and labels and default in {str(l) for l in labels}:
        return default
    if default and not labels:
        return default
    if labels:
        return str(list(labels)[0])
    return default or ""


def detect_reference_landscape_requested(*goal_texts: str) -> Dict[str, Any]:
    """
    Detect whether the user asked for the reference-projected landscape pipeline.

    Returns ``{"requested": bool, "reference_label": str|None}``.
    """
    text = _normalize_goal_text(*goal_texts)
    if not text:
        return {"requested": False, "reference_label": None}

    # Family modular dynamics uses independent dihedral PCA entropy, not shared-ref FEL.
    if detect_family_modular_dynamics_requested(*goal_texts):
        # Only honor shared-ref FEL if explicitly named.
        explicit_shared = any(
            re.search(p, text)
            for p in (
                r"shared[-\s]reference\s+fel",
                r"shared\s+fel\b",
                r"shared\s+free[-\s]?energy\s+landscape",
                r"run_reference_landscape_pipeline",
                r"reference[-\s]?projected\s+pca",
            )
        )
        if not explicit_shared:
            return {"requested": False, "reference_label": None}

    requested = any(re.search(p, text) for p in _REFERENCE_LANDSCAPE_PATTERNS)
    if not requested:
        return {"requested": False, "reference_label": None}

    return {
        "requested": True,
        "reference_label": _resolve_reference_label(*goal_texts),
    }


def _strip_compute_cluster_mentions(text: str) -> str:
    """Remove HPC/compute 'cluster' wording so it does not trigger classification."""
    return re.sub(
        r"\b(?:hpc|compute|computer|slurm|batch|gpu|cpu|job)\s+clusters?\b",
        " ",
        text or "",
        flags=re.IGNORECASE,
    )


def detect_classification_requested(*goal_texts: str) -> bool:
    """True when the user explicitly asks for classification / clustering / feature matrix."""
    for text in goal_texts:
        if not text:
            continue
        normalized = (
            text.lower()
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
        )
        normalized = strip_negated_clauses(_strip_compute_cluster_mentions(normalized))
        if any(re.search(p, normalized) for p in _CLASSIFICATION_REQUEST_PATTERNS):
            return True
    return False


def classification_metric_groups_for_goal(*goal_texts: str) -> Optional[FrozenSet[str]]:
    """
    Metric groups to featurize when classification is requested.

    Returns None if classification was not requested (skip collector).
    """
    if not detect_classification_requested(*goal_texts):
        return None

    from src.analysis.classification_collector import (
        CLASSIFICATION_FEATURE_GROUPS,
        DEFAULT_CLASSIFICATION_METRIC_GROUPS,
    )

    metrics = detect_requested_metrics_union(*goal_texts)
    cp_req = detect_consensus_pocket_requested(*goal_texts)
    ref_req = detect_reference_landscape_requested(*goal_texts)
    family_modular = detect_family_modular_dynamics_requested(*goal_texts)

    if metrics is None:
        groups = set(DEFAULT_CLASSIFICATION_METRIC_GROUPS)
    else:
        groups = {m for m in metrics if m in CLASSIFICATION_FEATURE_GROUPS}
        if "gyration" in metrics:
            groups.add("rg")
        if "rmsf" in groups:
            groups.discard("rmsf")
            groups.add("pocket_rmsf")
        base = set(DEFAULT_CLASSIFICATION_METRIC_GROUPS)
        extras = groups - base
        groups = base | extras if extras else base

    if family_modular:
        groups = set()
        if metrics is not None:
            for m in metrics:
                if m in CLASSIFICATION_FEATURE_GROUPS and m not in {"paper_ward4"}:
                    groups.add(m)
                if m == "shared_dihedral_pca":
                    groups.add("dihedral_pca_ref")
        if cp_req.get("requested") or "reference_pocket" in groups:
            groups.add("reference_pocket")
            groups.add("com")
        return frozenset(groups)

    modular = groups & {
        "consensus_rmsf",
        "consensus_torsions",
        "consensus_dccm",
        "dihedral_pca",
        "dihedral_tica",
        "cart_pca",
        "cart_tica",
    }
    # Modular family-dynamics features → keep them; do not collapse to
    # reference-only archetype (that path yields only 2 FEL scalars).
    if modular:
        if cp_req.get("requested"):
            groups.add("reference_pocket")
        groups.discard("reference_fel")
        groups.discard("reference_pca")
        groups.discard("paper_ward4")
        return frozenset(groups)

    if cp_req.get("requested") and ref_req.get("requested"):
        return frozenset({"reference_pocket", "reference_fel", "reference_pca"})
    if cp_req.get("requested") or ref_req.get("requested"):
        groups -= {"com", "contacts", "pocket_sasa", "residence", "pocket_rmsf", "fel", "ligand_rmsf", "sasa"}
        if cp_req.get("requested"):
            groups.add("reference_pocket")
        if ref_req.get("requested"):
            groups.add("reference_fel")
            groups.add("reference_pca")
        return frozenset(groups)

    return frozenset(groups)


def get_classification_tool_guide() -> str:
    """Prompt block when user requests unsupervised classification."""
    return """**UNSUPERVISED CLASSIFICATION (only when user explicitly requests it):**

When the goal asks for comparative dynamics features (pocket–ligand COM /
orientation, consensus flexibility, pocket χ₁, N↔C correlation, dihedral
landscape entropy, etc.), schedule the matching modular metric groups below.
Do **not** invent fixed cluster counts (k) or paper-specific feature schemas —
extract features, build a dendrogram + heatmap, and leave cut interpretation
to the human.

| Group | Tools | Notes |
|-------|-------|-------|
| reference_pocket | run_consensus_pocket_metrics_batch (combined) | mean/std COM + axis angle |
| consensus_rmsf | calculate_consensus_rmsf_features | mapped Cα mean/std |
| consensus_torsions | calculate_consensus_torsions | pocket χ₁ circular mean |
| consensus_dccm | calculate_consensus_dccm_features | N↔C lobe correlation |
| dihedral_pca | run_independent_dynamics_fel (space=dihedral, method=pca) | landscape grid entropy |
| com | calculate_ligand_pocket_distance | local pocket distance |
| fel | PCA + FEL + analyze_fel_landscape_features | local Cartesian FEL (optional) |

Combined phase:
`collect_classification_features_table` → `cluster_classification_features`
(hierarchical). Prefer a single dendrogram+heatmap panel. Do not hard-code k.

Pre-combined: also `plot_global_mapped_alignment` → global_consensus_msa + pocket_mapped MSA PNGs
when a reference pocket is requested.

Do NOT run the collector unless the user asked for classification/clustering
or a dendrogram / feature heatmap."""


def get_classification_per_sim_tool_guide() -> str:
    """Per-simulation classification featurization only (no combined/clustering tools)."""
    return """**PER-SIMULATION FEATURIZATION (when user requests unsupervised classification):**

Run ONLY the per-trajectory tools needed for the feature groups below. Do NOT call
collect_classification_features_table or cluster_classification_features here —
those run automatically at combined phase after all simulations finish.

| Group | Per-sim tools |
|-------|----------------|
| consensus_torsions | calculate_consensus_torsions → consensus_dihedrals/ |
| consensus_rmsf | calculate_consensus_rmsf_features → consensus_rmsf/ |
| consensus_dccm | calculate_consensus_dccm_features → consensus_DCCM/ |
| dihedral_pca | run_independent_dynamics_fel (dihedral/pca) → consensus_PCA/ |
| com | calculate_ligand_pocket_distance |
| contacts | calculate_protein_ligand_contacts |
| pocket_sasa | calculate_pocket_sasa (needs .tpr) |
| residence | analyze_ligand_residence |
| pocket_rmsf | calculate_pocket_rmsf |
| ligand_rmsf | calculate_ligand_rmsf |
| fel | calculate_trajectory_pca → calculate_free_energy_landscape → analyze_fel_landscape_features |

For family modular dynamics, prefer the consensus_* + dihedral_pca rows (not
shared-reference FEL). Include reference_pocket (COM + axis angle) and local
``com`` as fallbacks when the goal asks for pocket–ligand geometry. After
features are collected, the analysis agent asks the LLM to select a
scientifically motivated subset (with written reasoning) before hierarchical
clustering — similar to, but not locked to, paper Ward-4 descriptors. After
each calculate_*/analyze_* step that writes a data file, add plot_md_data using
the standard output basename from the filenames guide above (never plot FEL
grid CSVs)."""


def detect_combined_only_metrics(goal: str) -> FrozenSet[str]:
    """
    Metrics the user scoped to combined/cross-simulation analysis only.

    Example: "ligand pocket distance in cross simulations only" → ``com`` is
    excluded from per-simulation analysis plans.

    Does NOT exclude metrics also requested per simulation (e.g. "for each
    trajectory run: ligand pocket distance" plus "overlay in combined analysis").
    """
    text = (goal or "").lower()
    text = (
        text.replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    combined_only: Set[str] = set()

    cross_scope = bool(
        re.search(
            r"cross[-\s]?sim(?:ulation)?s?|combined analysis|across (?:all )?simulations",
            text,
        )
    )
    if not cross_scope:
        return frozenset()

    per_sim_requested = bool(
        re.search(
            r"for each (?:trajectory|simulation|system)|each trajectory run|per[-\s]simulation",
            text,
        )
    )

    # Ligand-pocket distance explicitly for cross-sim comparison ONLY (not when also per-sim)
    if re.search(
        r"ligand[-\s]?pocket|pocket[-\s]?distance|binding[-\s]?site distance",
        text,
    ):
        explicit_cross_only = bool(
            re.search(
                r"(?:ligand[-\s]?pocket|pocket[-\s]?distance)[^.;\n]{0,100}"
                r"(?:cross[-\s]?sim(?:ulation)?s?|across simulations|combined analysis)"
                r"[^.;\n]{0,40}\bonly\b",
                text,
            )
            or re.search(
                r"(?:cross[-\s]?sim(?:ulation)?s?|combined analysis|across simulations)"
                r"[^.;\n]{0,100}(?:ligand[-\s]?pocket|pocket[-\s]?distance)"
                r"[^.;\n]{0,40}\bonly\b",
                text,
            )
        )
        if explicit_cross_only and not per_sim_requested:
            combined_only.add("com")

    return frozenset(combined_only)


def detect_requested_metrics_for_sim(
    state: Optional[Dict[str, Any]] = None,
    agent_input: Optional[Any] = None,
) -> Optional[FrozenSet[str]]:
    """
    Metrics requested for the *current* simulation.

    Per-simulation ``user_goal`` text (from the multisim master plan) overrides
    the global CLI goal so e.g. DCCM requested only for JAK1/TYK2 does not
    run on STRAA/ULK4.
    """
    per_sim_texts: list[str] = []
    if state:
        text = (state.get("user_goal") or "").strip()
        if text:
            per_sim_texts.append(text)
    if agent_input is not None:
        text = (getattr(agent_input, "user_goal", None) or "").strip()
        if text and text not in per_sim_texts:
            per_sim_texts.append(text)

    for text in per_sim_texts:
        metrics = detect_requested_metrics(text)
        if metrics is not None:
            return _apply_combined_only_exclusions(metrics, state, agent_input)

    union = detect_requested_metrics_union(
        *collect_goal_texts_for_intent(state, agent_input)
    )
    return _apply_combined_only_exclusions(union, state, agent_input)


def _apply_combined_only_exclusions(
    metrics: Optional[FrozenSet[str]],
    state: Optional[Dict[str, Any]] = None,
    agent_input: Optional[Any] = None,
) -> Optional[FrozenSet[str]]:
    if metrics is None:
        return None
    master = (state.get("user_goal_original") or "") if state else ""
    if not master.strip() and state:
        master = state.get("user_goal") or ""
    combined_only = detect_combined_only_metrics(master)
    if not combined_only:
        return metrics
    # Per-simulation prompt may re-request metrics the master goal also mentions for combined overlay.
    per_sim = (state.get("user_goal") or "").strip() if state else ""
    if per_sim:
        per_sim_metrics = detect_requested_metrics(per_sim)
        if per_sim_metrics:
            combined_only = combined_only - per_sim_metrics
    if not combined_only:
        return metrics
    trimmed = frozenset(metrics - combined_only)
    return trimmed if trimmed else frozenset()


def get_pca_fel_tool_guide() -> str:
    """Prompt block for PCA + free-energy landscape workflow."""
    return """**PCA & FREE-ENERGY LANDSCAPE (when user asks for dynamics / FEL / classification):**

| Step | Tool | Output |
|------|------|--------|
| 1 | `calculate_trajectory_pca` | `pca_projections.dat`, `pca_variance.dat` |
| 2 | `plot_pca_projection` | `pca_pc1_pc2_time.png` (PC1 vs PC2, coloured by time) |
| 3 | `calculate_free_energy_landscape` | `fel_pc1_pc2_grid.csv` only (no plain FEL PNG) |
| 4 | `analyze_fel_landscape_features` | `fel_features.json/.csv`, `fel_basins.csv`, `fel_basins.png` |

Do **not** plot `fel_pc1_pc2_grid.csv` or `fel_features.csv` with `plot_md_data` (wrong line plots).
Do **not** emit `fel_pc1_pc2.png`, `fel_pc1_pc2_grid.png`, or `fel_features.png` — basins map + feature tables suffice.

**FEL classification metrics (step 4):**
- Number of minima / basins
- Basin depth, area (population fraction)
- Inter-basin barrier heights (kJ/mol)
- Major basin population
- Landscape entropy S = −Σ p_i ln(p_i) (higher S → more conformational diversity)

**Multi-simulation (35 systems):** if the user explicitly requests classification,
run `collect_classification_features_table` at combined phase with metric groups
matching their goal. Otherwise do NOT run it during combined analysis.

Rules:
- **Always use these defaults** unless the user goal explicitly requests different values:
  `selection="protein and name CA"`, `n_components=10`, `frame_interval=1`,
  `reference_frame=0`, `pc_x=1`, `pc_y=2`, `bins=50`, `temperature_k=310`.
- Do NOT pass different `frame_interval` or `n_components` in tool_params unless the user asked.
- FEL uses F = −kT ln P(PC1, PC2) with `temperature_k` (default 310 K); minimum set to 0 kJ/mol.
- Prefer ≥ 50 frames for a meaningful landscape; warn the user if the trajectory is very short.
- Run steps 1–4 when the user requests FEL, classification, or conformational diversity."""


def get_intent_preservation_block(user_goal: str = "") -> str:
    """Prompt block: user goal defines which analyses to run."""
    metrics = detect_requested_metrics(user_goal)
    metric_hint = ""
    if metrics:
        names = ", ".join(sorted(metrics))
        metric_hint = (
            f"\nDetected explicit analysis scope from USER GOAL: {names}. "
            "Do NOT add other metrics unless the goal also names them.\n"
        )

    return f"""**USER INTENT IS MANDATORY (HIGHEST PRIORITY):**
- The USER GOAL defines which analyses to run. Do not expand beyond it.
- If the goal says "RMSF only" (or similar), plan ONLY RMSF plus the directly
  required plot/CSV outputs and reporter documentation for those results.
- Do NOT add RMSD, Rg, SASA, hydrogen bonds, DCCM, DSSP, energy, wrap_trajectory,
  or run_complete_analysis unless the USER GOAL explicitly requests them.
- If the goal asks broadly for "protein dynamics" without naming metrics, choose a
  small justified dynamics set (typically RMSD + RMSF + Rg, optionally DCCM when relevant).
- When both Analysis and Reporter agents are in the workflow, the Reporter section
  must document only what the Analysis section produces — no extra calculations.
{metric_hint}"""


def get_standard_output_filenames_block() -> str:
    """Prompt block: consistent basenames across simulations for combined overlay."""
    lines = [
        "**STANDARD PER-SIMULATION OUTPUT FILENAMES (CRITICAL for multi-sim):**",
        "Use the SAME basename in every simulation directory so combined analysis can",
        "collect and overlay results. Write outputs under {working_dir}/analysis/.",
        "Do NOT prefix filenames with simulation labels (no p29597_rmsf.dat).",
        "",
    ]
    for metric, files in STANDARD_OUTPUT_FILES.items():
        if "data" in files:
            lines.append(f"- {metric.upper()}: data={files['data']}, plot={files.get('plot', 'N/A')}")
        elif "data_prefix" in files:
            lines.append(
                f"- {metric.upper()}: prefix={files['data_prefix']}, plot={files.get('plot', 'N/A')}"
            )
    lines.append(
        "\nWhen the same metric is requested for every simulation, every sim must use "
        "these exact overall basenames so combined overlay can collect them."
    )
    lines.append("")
    lines.append("**OVERALL vs SPECIFIC filenames:**")
    lines.append(
        "- Overall / whole-system metrics use the exact names above "
        "(`rmsd.dat`/`rmsd.png`, `rmsf.dat`/`rmsf.png`)."
    )
    lines.append(
        "- A subset, chain, residue range, or proximity group MUST use a qualifier "
        "and matching plot stem: `{metric}_{qualifier}.dat` and `{metric}_{qualifier}.png`."
    )
    lines.append(
        "- Examples: `rmsd_B_1to34.dat` + `rmsd_B_1to34.png`; "
        "`rmsf_1to34.dat` + `rmsf_1to34.png`; "
        "`rmsf_A_near_B1to34.dat` + `rmsf_A_near_B1to34.png`; "
        "`com_distance_B1to34_vs_nearbyA.csv` + `.png`; "
        "`min_distance_B1to34_vs_nearbyA.csv` + `.png`; "
        "`nearby_residues_A_near_B1to34.json`; "
        "`hbond_occupancy_B1to34_vs_A.csv`; "
        "`saltbridge_occupancy_B1to34_vs_A.csv`."
    )
    lines.append(
        "- Never overwrite an overall file with a subset (do not write chain-B RMSD "
        "to `rmsd.dat` if you also compute complex RMSD)."
    )
    lines.append(
        "- Combined/multi-sim overlay only collects the overall standard names; "
        "qualified files stay per-simulation."
    )
    lines.append(
        "- Every `.dat`/`.csv` time-series or RMSF profile gets a `plot_md_data` step "
        "whose `output_file` uses the SAME stem (`.png`)."
    )
    return "\n".join(lines)


def get_master_plan_tools_note(agent_list: list[str]) -> str:
    """Short note explaining how tool sections map to workflow agents."""
    agents = ", ".join(agent_list) if agent_list else "workflow agents"
    return (
        "Tool sections below match the workflow agents in --subtask. "
        f"Per-simulation tools are for: {agents}. "
        "Cross-simulation tools run once at the project base after all per-sim runs."
    )


def metric_mentioned_in_goal(metric: str, goal_text: str) -> bool:
    """True when *goal_text* explicitly names a metric (incl. rg / radius of gyration)."""
    text = (goal_text or "").lower()
    if not text:
        return False
    metric = metric.lower()
    patterns = _METRIC_PATTERNS.get(metric)
    if patterns:
        return any(re.search(p, text) for p in patterns)
    return metric in text


def get_metric_collect_pattern(metric: str) -> str:
    """
    Filename stem used by combined analysis to locate per-sim data files.

    Matches STANDARD_OUTPUT_FILES basenames (e.g. rg → ``gyration`` not ``rg``).
    """
    metric = metric.lower()
    if metric in STANDARD_OUTPUT_FILES:
        spec = STANDARD_OUTPUT_FILES[metric]
        if "data" in spec:
            return Path(spec["data"]).stem
        if "data_prefix" in spec:
            return spec["data_prefix"]
    return metric


def detect_com_distance_mode(goal: str) -> str:
    """
    Which COM-distance tool(s) the user wants.

    Returns ``pocket``, ``protein_com``, or ``both``.
    """
    text = (goal or "").lower()
    pocket_markers = (
        "pocket",
        "binding site",
        "catalytic pocket",
        "catalytic site",
        "active site",
        "ligand_pocket",
        "ligand pocket",
        "within 5",
        "within 5 å",
        "within 5 a",
        "nearby atoms",
        "surrounding atoms",
    )
    protein_com_markers = (
        "whole protein",
        "entire protein",
        "protein com",
        "protein center of mass",
        "protein centre of mass",
        "com of protein",
        "protein-to-ligand com",
        "protein to ligand com",
    )
    wants_pocket = any(m in text for m in pocket_markers)
    wants_protein_com = any(m in text for m in protein_com_markers)
    if wants_pocket and wants_protein_com:
        return "both"
    if wants_protein_com:
        return "protein_com"
    if wants_pocket:
        return "pocket"
    # Generic "COM distance … ATP … protein" in holo kinase studies → pocket tracking
    if any(k in text for k in ("atp", "ligand", "inhibitor", "adp", "gtp")) and (
        "com" in text or "center of mass" in text or "centre of mass" in text
    ):
        return "pocket"
    return "protein_com" if "com" in text else "pocket"


def get_com_distance_tool_guide() -> str:
    """Prompt block: choose ligand-pocket vs generic COM distance tools."""
    return """**COM DISTANCE TOOLS — choose exactly one unless the user asks for both:**

| User intent | Tool | Output files | When to use |
|-------------|------|--------------|-------------|
| Ligand **binding-pocket** stability (atoms within ~5 Å of ligand at frame 0) | `calculate_ligand_pocket_distance` | `ligand_pocket_distance.csv`, `.png` | "ligand pocket distance", "catalytic pocket", "binding site", "ATP in the pocket", "nearby pocket atoms" |
| **Whole-protein** COM to ligand COM (or any two explicit selections) | `calculate_com_distance` | `com_distance.csv`, `.png` | "COM distance between the whole protein and ATP", "protein COM to ligand COM", domain–domain COM |

Rules:
- Do **NOT** run both tools unless the user explicitly requests pocket distance **and** whole-protein COM distance.
- For holo kinase goals like "COM distance of ATP from protein" without further detail, prefer **`calculate_ligand_pocket_distance`** (biologically meaningful pocket tracking).
- `calculate_com_distance` requires `selection1` and `selection2` (NOT `ligand_selection`).
- Example pocket: `ligand_selection="resname ATP"`, `cutoff=5.0`, `output_file="ligand_pocket_distance.csv"`.
- Example protein COM: `selection1="protein"`, `selection2="resname ATP"`, `output_file="com_distance.csv"`.
- For a residue-range or chain subset, use a qualified name such as
  `com_distance_B1to34_vs_nearbyA.csv` (do not overwrite another COM series)."""


def get_proximity_tool_guide() -> str:
    """Prompt block: frame-0 neighbor identification and min heavy-atom distance."""
    return """**PROXIMITY / INTERFACE TOOLS (protein–protein or any two groups):**

These ARE built-in. Do NOT request programmer creation for them.

| User intent | Tool | Output | Notes |
|-------------|------|--------|-------|
| List residues of group B within *X* Å of group A at a given frame (usually frame 0) | `identify_nearby_residues` | `nearby_residues.json` + `.csv` | Freezes the neighbor set. Writes `mda_selection` / `mda_selection_ca` / `mda_selection_heavy`. |
| RMSF of those frozen neighbors | `calculate_rmsf` | `rmsf_{qualifier}.dat` | Set `selection_from_file` to the JSON (`selection_key="mda_selection_ca"`). |
| COM of query vs frozen neighbors | `calculate_com_distance` | `com_distance_{qualifier}.csv` | `selection1` = query; `selection2_from_file` = JSON (`selection_key="mda_selection_heavy"`). |
| Minimum heavy-atom distance vs time | `calculate_min_heavy_atom_distance` | `min_distance_{qualifier}.csv` | Same selection pattern as COM. |

Rules:
- Neighbor identity is taken **once** at `frame` (default 0), then held fixed for RMSF / COM / min-distance. Do not re-evaluate `around` every frame.
- You may also pass an MDAnalysis `around` selection directly, e.g.
  `chainID A and name CA and around 10 (chainID B and resid 1:34)` — still write a qualified filename.
- Example: query `chainID B and resid 1:34`, neighbors `chainID A`, cutoff 10 Å →
  `nearby_residues_A_near_B1to34.json`, `rmsf_A_near_B1to34.dat`; query RMSF → `rmsf_1to34.dat`.
- `calculate_protein_ligand_contacts` is for protein–ligand only, not protein–protein min-distance.

**PROTEIN–PROTEIN INTERACTION PARTNERS (built-in; do NOT request programmer tools):**

| User intent | Tool | Output | Notes |
|-------------|------|--------|-------|
| H-bond occupancy / important H-bond pairs between two protein groups | `calculate_hbond_occupancy` | `hbond_occupancy.csv` + `.png` | Residue-pair occupancy (% frames). Use `selection1`/`selection2` and qualified names. |
| Salt-bridge distances / charged interaction partners | `calculate_salt_bridge_distances` | `saltbridge_occupancy.csv` + `.png` | Arg/Lys/His/N-ter vs Asp/Glu; occupancy at 4 Å. |

Rules:
- These tools ARE the way to name **important residues and their interaction partners** at a protein–protein interface. Do not invent a new contact tool.
- Typical trio for an interface: `identify_nearby_residues` (who is nearby) → `calculate_hbond_occupancy` + `calculate_salt_bridge_distances` (which pairs persist).
- Example B 1–34 vs A: `selection1="chainID B and resid 1:34"`, `selection2="chainID A"`,
  `output_file="hbond_occupancy_B1to34_vs_A.csv"` / `saltbridge_occupancy_B1to34_vs_A.csv`.
- For a three-chain complex, run one H-bond + one salt-bridge job **per interface the user asked about** (A–B, A–C, B–C), each with its own qualifier.
- Never use `calculate_protein_ligand_contacts` for chain–chain H-bonds."""


def get_family_scale_planning_guide() -> str:
    """Prompt block for multi-sim / family campaigns seeking interesting dynamics."""
    return """**FAMILY-SCALE / MULTI-SIM DYNAMICS (general-purpose, modular):**

This framework targets protein *families* and multi-system campaigns, not one-off single sims.

1. **Per-sim first** — identical metric set and standard filenames under `{label}/analysis/` so overlays work.
2. **QC early** — prefer `run_trajectory_qc` when runs may be truncated or unstable.
3. **Modular features — do NOT hard-wire a fixed feature matrix.**
   - Parse the user goal for an **explicit feature list** and schedule **only** those tools.
   - Examples of atomic tools: `calculate_consensus_torsions`, `run_independent_dynamics_fel`
     (set `space=dihedral|cartesian`, `method=pca|tica`), `fit_dynamics_model` +
     `project_dynamics_model` / `run_shared_dynamics_fel_batch` for shared-reference mode,
     `calculate_consensus_rmsf_features`, `calculate_consensus_dccm_features`,
     consensus pocket tools, Cartesian `calculate_trajectory_pca` / FEL.
   - **Independent vs shared-reference** is a tool kwarg / tool choice from the goal:
     - "independent" / "per-protein PCA/tICA" → `run_independent_dynamics_fel`
     - "shared reference" / "project onto PKA/reference" → fit then project batch
   - When collecting for clustering, pass `requested_metric_groups` and/or
     `feature_columns` matching **only** what the user asked; use `auto_discover`
     only if the goal says to use whatever modular artifacts were computed.
4. **Interesting dynamics** — after requested per-sim metrics, use **combined** overlays /
   comparison / hierarchical clustering; use **shared** MSA / consensus pocket when comparing related sequences.
   Do **not** hard-code a cluster count (k); emit dendrogram + heatmap and leave cuts to the human.
5. **Do not invent campaign-specific scripts** — use registry tools; if a genuine gap remains
   after checking the metric map, request programmer creation once with a clear capability statement.
6. **Combined reporter** — when comparing many systems, plan `generate_combined_html_report`
   (and literature tools only if the goal asks)."""


def _metric_clause_pattern(metric: str) -> str:
    """Regex fragment for metric name in ``… for JAK1 and TYK2`` clauses."""
    if metric == "rg":
        return r"(?:rg|radius[-\s]of[-\s]gyration|\bgyration\b)"
    if metric == "com":
        return r"(?:com|center[-\s]?of[-\s]?mass|ligand[-\s]?pocket|ligand[-\s]?pocket[-\s]?distance)"
    return re.escape(metric)


def _resolve_token_to_label(
    token: str,
    labels: List[str],
    label_name_map: Dict[str, str],
) -> Optional[str]:
    """Map a protein name, UniProt id, or label token to a simulation label."""
    raw = token.strip().strip("*_\"'")
    if not raw:
        return None
    low = raw.lower()
    if low.endswith(".pdb"):
        low = low[:-4]

    for label in labels:
        if label.lower() == low or low == label.lower().split("_")[0]:
            return label

    for uid, pname in label_name_map.items():
        uid_low = uid.lower()
        pname_low = pname.lower()
        if low in (uid_low, pname_low):
            for label in labels:
                if label.lower() == uid_low or label.lower().startswith(f"{uid_low}_"):
                    return label
    return None


def parse_metric_target_labels(
    goal_text: str,
    metric: str,
    labels: List[str],
    label_name_map: Optional[Dict[str, str]] = None,
) -> Optional[FrozenSet[str]]:
    """
    When the user names specific proteins/systems for one metric (e.g. DCCM for
    JAK1 and TYK2), return the matching simulation labels.
    """
    if not goal_text or not labels:
        return None
    text = goal_text.lower()
    metric = metric.lower()
    if not metric_mentioned_in_goal(metric, text):
        return None

    name_map = {k.lower(): v for k, v in (label_name_map or {}).items()}
    matched: Set[str] = set()

    metric_pat = _metric_clause_pattern(metric)

    clause_patterns = (
        rf"\b{metric_pat}\b\s+for\s+([^.\n;]+?)(?:\.\s|$|\bto compare\b|\bto\b|\bacross\b|\bonly\b|\bfrom\b|\bwith\b|\bin\b)",
        rf"\b{metric_pat}\b[^.\n]{{0,40}}\bbetween\s+([^.\n;]+?)(?:\.\s|$|\bto\b|\bwith\b)",
    )
    for pattern in clause_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            clause = match.group(1)
            clause = re.split(r"\bto compare\b|\bcompare\b|\bacross\b", clause)[0]
            for part in re.split(r"\band\b|,|/|&", clause):
                part = part.strip()
                if not part or part in {"all", "each", "every", "all simulations", "four proteins"}:
                    continue
                label = _resolve_token_to_label(part, labels, name_map)
                if label:
                    matched.add(label)

    return frozenset(matched) if matched else None


def resolve_sims_for_combined_metric(
    metric: str,
    sim_dirs: List[str],
    labels: List[str],
    *,
    master_goal: str = "",
    combined_plan: str = "",
    completed_sim_states: Optional[List[Dict[str, Any]]] = None,
    label_name_map: Optional[Dict[str, str]] = None,
) -> tuple[List[str], List[str]]:
    """
    Select which simulations participate in one combined-metric analysis.

    Combined overlays do NOT always use every simulation:
    - RMSF for all four proteins → all labels
    - DCCM only for JAK1 and TYK2 → subset ``p23458``, ``p29597``

    Priority:
    1. Per-simulation goals from completed runs (authoritative per sim)
    2. Master goal / combined plan naming specific targets for this metric
    3. Global metric request with no named subset → all simulations
    """
    completed = completed_sim_states or []
    by_label = {s.get("label"): s for s in completed if s.get("label")}
    name_map = {k.lower(): v for k, v in (label_name_map or {}).items()}

    included_dirs: List[str] = []
    included_labels: List[str] = []
    saw_explicit_per_sim = False

    for sim_dir, label in zip(sim_dirs, labels):
        goal = (by_label.get(label) or {}).get("user_goal", "").strip()
        if not goal:
            continue
        per_sim_metrics = detect_requested_metrics(goal)
        if per_sim_metrics is None:
            continue
        saw_explicit_per_sim = True
        if metric in per_sim_metrics:
            included_dirs.append(sim_dir)
            included_labels.append(label)

    if included_dirs:
        return included_dirs, included_labels

    if saw_explicit_per_sim:
        # Per-sim goals exist and none request this metric.
        return [], []

    goal_text = f"{master_goal}\n{combined_plan}"
    subset = parse_metric_target_labels(goal_text, metric, labels, name_map)
    if subset:
        dirs = [d for d, l in zip(sim_dirs, labels) if l in subset]
        labs = [l for l in labels if l in subset]
        return dirs, labs

    global_metrics = detect_requested_metrics_union(master_goal, combined_plan)
    if global_metrics and metric in global_metrics:
        return list(sim_dirs), list(labels)

    return [], []
