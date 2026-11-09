"""Shared analysis protocol for comparative (family-scale) campaigns.

Required calculations run for every protein. The protocol is compiled once
and applied identically — the LLM does not invent a different tool list
per system.

Feature packs (see ``FEATURE_PACKS``) choose gold columns and pins.
Shared-reference dynamics is a combined-stage step
(``FAMILY_POST_CALCULATIONS``), not a per-sim ``consensus_PCA/`` requirement.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentic.campaign.spec import AnalysisRecipe, ArtifactContract, RecipeStep

# Required per-simulation calculations for every protein in a comparative study.
# Shared-ref PCA dyn is a combined-stage step (see FAMILY_POST_CALCULATIONS).
REQUIRED_CALCULATIONS: List[str] = [
    "calculate_consensus_pocket_metrics",
    "calculate_consensus_torsions",
    "calculate_consensus_rmsf_features",
    "calculate_consensus_dccm_features",
]
FAMILY_PIN_TOOLS: List[str] = REQUIRED_CALCULATIONS  # alias: required calculations

# Combined-stage tools for shared-reference dynamics (feature packs that include it).
FAMILY_POST_CALCULATIONS: List[str] = [
    "run_shared_dynamics_fel_batch",
    "compute_shared_pka_ref_dyn_features",
]

SHARED_DYN_COLUMNS = frozenset({"pca_pka_ref_shared_dyn", "pca_ref_shared_dyn"})

# Per-sim modular dirs written by pin tools. Independent dihedral PCA
# (``consensus_PCA``) is optional and not a hard per-sim gate. Shared-ref dyn
# writes ``consensus_PCA_ref`` at the combined stage.
FAMILY_MODULAR_DIRS: List[str] = [
    "consensus_dihedrals",
    "consensus_rmsf",
    "consensus_DCCM",
]
# Legacy names still accepted when present on disk.
FAMILY_MODULAR_DIR_ALIASES: Dict[str, str] = {
    "consensus_PCA_ref": "consensus_PCA",
}

# Named presets. The framework default is ``from_goal`` (see
# ``build_protocol_from_goal``): columns and tools follow the user goal.
# ``shared_pocket_dynamics`` is the pseudokinase demonstration set.
# ``paper_sharedpcdyn`` is a legacy alias of that preset.
_SHARED_POCKET_PRESET: Dict[str, Any] = {
        "gold_columns": [
            "reference_pocket_ligand_distance_mean_A",
            "reference_pocket_ligand_distance_std_A",
            "reference_pocket_ligand_axis_angle_mean_deg",
            "reference_pocket_std_ligand_axis_angle_deg",
            "chi1_pocket_circ_mean_deg",
            "chi1_pocket_circ_std_deg",
            "consensus_rmsf_mean_A",
            "consensus_rmsf_std_A",
            "dccm_N_C_mean_corr",
            "pca_pka_ref_shared_dyn",
        ],
        "pin_tools": list(REQUIRED_CALCULATIONS),
        "modular_dirs": list(FAMILY_MODULAR_DIRS),
        "post_tools": list(FAMILY_POST_CALCULATIONS),
        "metric_groups": [
            "com",
            "reference_pocket",
            "consensus_rmsf",
            "consensus_torsions",
            "consensus_dccm",
            "dihedral_pca_ref",
        ],
}

FEATURE_PACKS: Dict[str, Dict[str, Any]] = {
    "shared_pocket_dynamics": _SHARED_POCKET_PRESET,
    "paper_sharedpcdyn": _SHARED_POCKET_PRESET,
    "pocket_chi_rmsf_dccm": {
        "gold_columns": [
            "reference_pocket_ligand_distance_mean_A",
            "reference_pocket_ligand_distance_std_A",
            "reference_pocket_ligand_axis_angle_mean_deg",
            "reference_pocket_std_ligand_axis_angle_deg",
            "chi1_pocket_circ_mean_deg",
            "chi1_pocket_circ_std_deg",
            "consensus_rmsf_mean_A",
            "consensus_rmsf_std_A",
            "dccm_N_C_mean_corr",
        ],
        "pin_tools": list(REQUIRED_CALCULATIONS),
        "modular_dirs": list(FAMILY_MODULAR_DIRS),
        "post_tools": [],
        "metric_groups": [
            "com",
            "reference_pocket",
            "consensus_rmsf",
            "consensus_torsions",
            "consensus_dccm",
        ],
    },
    "rmsf_dccm_only": {
        "gold_columns": [
            "consensus_rmsf_mean_A",
            "consensus_rmsf_std_A",
            "dccm_N_C_mean_corr",
        ],
        "pin_tools": [
            "calculate_consensus_rmsf_features",
            "calculate_consensus_dccm_features",
        ],
        "modular_dirs": ["consensus_rmsf", "consensus_DCCM"],
        "post_tools": [],
        "metric_groups": ["consensus_rmsf", "consensus_dccm"],
    },
}


# Goal-derived descriptors. A comparative study includes only the blocks the
# user named. Each block can contribute one column or several.
_MEAN_RE = re.compile(r"\bmeans?\b", re.IGNORECASE)
_STD_RE = re.compile(r"standard deviation|\bstdev\b|\bstd\b", re.IGNORECASE)

FAMILY_DESCRIPTORS: List[Dict[str, Any]] = [
    {
        "id": "pocket_distance",
        "patterns": (
            r"com distance",
            r"center of mass",
            r"distance to the (?:consensus )?pocket",
            r"pocket.{0,30}distance",
            r"ligand.{0,30}distance",
        ),
        "mean_columns": ("reference_pocket_ligand_distance_mean_A",),
        "std_columns": ("reference_pocket_ligand_distance_std_A",),
        "columns": (),
        "pin_tools": ("calculate_consensus_pocket_metrics",),
        "post_tools": (),
        "metric_groups": ("reference_pocket", "com"),
        "modular_dirs": (),
    },
    {
        "id": "pocket_orientation",
        "patterns": (
            r"axis angle",
            r"axis orientation",
            r"orientation vs the pocket",
            r"ligand orientation",
            r"pocket axis",
        ),
        "mean_columns": ("reference_pocket_ligand_axis_angle_mean_deg",),
        "std_columns": ("reference_pocket_std_ligand_axis_angle_deg",),
        "columns": (),
        "pin_tools": ("calculate_consensus_pocket_metrics",),
        "post_tools": (),
        "metric_groups": ("reference_pocket",),
        "modular_dirs": (),
    },
    {
        "id": "sidechain_chi1",
        "patterns": (
            r"χ\s*₁",
            r"χ1",
            r"chi\s*1",
            r"side[-\s]?chain",
        ),
        "mean_columns": ("chi1_pocket_circ_mean_deg",),
        "std_columns": ("chi1_pocket_circ_std_deg",),
        "columns": (),
        "pin_tools": ("calculate_consensus_torsions",),
        "post_tools": (),
        "metric_groups": ("consensus_torsions",),
        "modular_dirs": ("consensus_dihedrals",),
    },
    {
        "id": "consensus_rmsf",
        "patterns": (
            r"\brmsf\b",
            r"flexibility of consensus",
            r"consensus[-\s]?mapped",
            r"c[αa]\s+flexibility",
        ),
        "mean_columns": ("consensus_rmsf_mean_A",),
        "std_columns": ("consensus_rmsf_std_A",),
        "columns": (),
        "pin_tools": ("calculate_consensus_rmsf_features",),
        "post_tools": (),
        "metric_groups": ("consensus_rmsf",),
        "modular_dirs": ("consensus_rmsf",),
    },
    {
        "id": "interdomain_dccm",
        "patterns": (
            r"n[-\s]?lobe.{0,40}c[-\s]?lobe",
            r"c[-\s]?lobe.{0,40}n[-\s]?lobe",
            r"inter[-\s]?domain",
            r"between (?:the )?(?:n|c)[-\s]?terminal",
            r"dccm_N_C",
        ),
        "mean_columns": (),
        "std_columns": (),
        "columns": ("dccm_N_C_mean_corr",),
        "pin_tools": ("calculate_consensus_dccm_features",),
        "post_tools": (),
        "metric_groups": ("consensus_dccm",),
        "modular_dirs": ("consensus_DCCM",),
    },
    {
        "id": "dccm",
        "patterns": (
            r"\bdccm\b",
            r"cross[-\s]?correlation",
            r"correlated motion",
        ),
        "mean_columns": (),
        "std_columns": (),
        "columns": ("mean_abs_dccm",),
        "pin_tools": ("calculate_consensus_dccm_features",),
        "post_tools": (),
        "metric_groups": ("consensus_dccm", "dccm"),
        "modular_dirs": ("consensus_DCCM",),
    },
    {
        "id": "shared_ref_dihedral_pca",
        "patterns": (
            r"pca_pka_ref_shared_dyn",
            r"pca_ref_shared_dyn",
            r"shared[-\s]?reference.{0,80}(?:dihedral|pca|dyn)",
            r"reference.{0,40}dihedral\s+pca",
        ),
        "mean_columns": (),
        "std_columns": (),
        "columns": ("pca_pka_ref_shared_dyn",),
        "pin_tools": (),
        "post_tools": ("compute_shared_pka_ref_dyn_features",),
        "metric_groups": ("dihedral_pca_ref",),
        "modular_dirs": (),
    },
    {
        "id": "rmsd",
        "patterns": (r"\brmsd\b",),
        "mean_columns": ("mean_rmsd_A",),
        "std_columns": ("std_rmsd_A",),
        "columns": (),
        "pin_tools": ("calculate_rmsd",),
        "post_tools": (),
        "metric_groups": ("rmsd",),
        "modular_dirs": (),
    },
    {
        "id": "rg",
        "patterns": (r"\brg\b", r"radius of gyration", r"\bgyration\b"),
        "mean_columns": ("mean_rg_A",),
        "std_columns": ("std_rg_A",),
        "columns": (),
        "pin_tools": ("calculate_gyration",),
        "post_tools": (),
        "metric_groups": ("rg",),
        "modular_dirs": (),
    },
    {
        "id": "sasa",
        "patterns": (r"\bsasa\b", r"solvent accessible"),
        "mean_columns": ("mean_protein_sasa_nm2",),
        "std_columns": ("std_protein_sasa_nm2",),
        "columns": (),
        "pin_tools": ("calculate_sasa",),
        "post_tools": (),
        "metric_groups": ("sasa",),
        "modular_dirs": (),
    },
]

_KNOWN_COLUMNS = {
    col
    for desc in FAMILY_DESCRIPTORS
    for col in (
        list(desc["columns"]) + list(desc["mean_columns"]) + list(desc["std_columns"])
    )
}
_FROM_GOAL_NAMES = frozenset({"", "from_goal", "auto", "goal", "user"})


def _goal_chunks(text: str) -> List[str]:
    raw = (text or "").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    chunks: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        chunks.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
    return chunks or [raw]


def _columns_for_chunk(desc: Dict[str, Any], chunk: str) -> List[str]:
    has_mean = bool(_MEAN_RE.search(chunk))
    has_std = bool(_STD_RE.search(chunk))
    picked = list(desc["columns"])
    means = list(desc["mean_columns"])
    stds = list(desc["std_columns"])
    if means or stds:
        if has_mean and not has_std:
            picked.extend(means)
        elif has_std and not has_mean:
            picked.extend(stds)
        else:
            picked.extend(means)
            picked.extend(stds)
    return picked


def _append_unique(dst: List[str], items: Sequence[str]) -> None:
    for item in items:
        if item and item not in dst:
            dst.append(item)


def build_protocol_from_goal(
    goal: str,
    *,
    gold_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Shared family protocol whose size follows the user, not a fixed list.

    Natural-language descriptors in *goal* select tools and columns. A
    non-empty *gold_columns* list replaces the parsed columns (any length)
    and still pulls in the tools that produce those columns.
    """
    columns: List[str] = []
    pins: List[str] = []
    posts: List[str] = []
    groups: List[str] = []
    dirs: List[str] = []
    matched: List[str] = []

    by_id = {desc["id"]: desc for desc in FAMILY_DESCRIPTORS}

    def _hit(desc_id: str, text: str) -> bool:
        desc = by_id[desc_id]
        return any(re.search(pat, text, flags=re.IGNORECASE) for pat in desc["patterns"])

    def _take(desc_id: str, chunk: str) -> None:
        desc = by_id[desc_id]
        matched.append(desc_id)
        _append_unique(columns, _columns_for_chunk(desc, chunk))
        _append_unique(pins, desc["pin_tools"])
        _append_unique(posts, desc["post_tools"])
        _append_unique(groups, desc["metric_groups"])
        _append_unique(dirs, desc["modular_dirs"])

    for chunk in _goal_chunks(goal):
        low = chunk.lower()
        # "N-lobe–C-lobe DCCM" is the domain-pair scalar, not a second
        # absolute-correlation feature.
        inter = _hit("interdomain_dccm", low)
        for desc in FAMILY_DESCRIPTORS:
            if desc["id"] == "dccm" and inter:
                continue
            if desc["id"] == "dccm":
                if _hit("dccm", low) and not _hit("interdomain_dccm", low):
                    _take("dccm", chunk)
                continue
            if _hit(desc["id"], low):
                _take(desc["id"], chunk)

    for token in re.findall(r"\b([a-z][a-z0-9_]{6,})\b", (goal or "").lower()):
        if token in _KNOWN_COLUMNS:
            _append_unique(columns, [token])
            for desc in FAMILY_DESCRIPTORS:
                owned = set(desc["columns"]) | set(desc["mean_columns"]) | set(desc["std_columns"])
                if token in owned:
                    _append_unique(pins, desc["pin_tools"])
                    _append_unique(posts, desc["post_tools"])
                    _append_unique(groups, desc["metric_groups"])
                    _append_unique(dirs, desc["modular_dirs"])

    explicit = [str(c) for c in (gold_columns or []) if str(c).strip()]
    if explicit:
        columns = []
        _append_unique(columns, explicit)
        for desc in FAMILY_DESCRIPTORS:
            owned = set(desc["columns"]) | set(desc["mean_columns"]) | set(desc["std_columns"])
            if owned.intersection(columns):
                _append_unique(pins, desc["pin_tools"])
                _append_unique(posts, desc["post_tools"])
                _append_unique(groups, desc["metric_groups"])
                _append_unique(dirs, desc["modular_dirs"])

    return {
        "gold_columns": columns,
        "pin_tools": pins,
        "modular_dirs": dirs,
        "post_tools": posts,
        "metric_groups": groups,
        "matched_descriptors": matched,
    }


def resolve_feature_pack(name: Optional[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return ``(pack_name, pack_dict)``.

    ``from_goal`` (the default) has no fixed dict; the caller builds it from
    the user goal. Unknown names also use ``from_goal`` rather than a
    demonstration preset.
    """
    key = (name or "").strip() or "from_goal"
    if key in _FROM_GOAL_NAMES:
        return "from_goal", None
    if key not in FEATURE_PACKS:
        return "from_goal", None
    return key, FEATURE_PACKS[key]


def pack_wants_shared_dyn(pack: Dict[str, Any]) -> bool:
    cols = set(pack.get("gold_columns") or [])
    tools = set(pack.get("post_tools") or [])
    return bool(cols & SHARED_DYN_COLUMNS) or bool(tools & set(FAMILY_POST_CALCULATIONS))

# Paper fig3 / sharedpcdyn Ward columns (exact names).
FAMILY_FEATURE_COLUMNS: List[str] = [
    "reference_pocket_ligand_distance_mean_A",
    "reference_pocket_ligand_distance_std_A",
    "reference_pocket_ligand_axis_angle_mean_deg",
    "reference_pocket_std_ligand_axis_angle_deg",
    "chi1_pocket_circ_mean_deg",
    "chi1_pocket_circ_std_deg",
    "consensus_rmsf_mean_A",
    "consensus_rmsf_std_A",
    "dccm_N_C_mean_corr",
    "pca_pka_ref_shared_dyn",
]

FAMILY_METRIC_GROUPS: List[str] = [
    "com",
    "reference_pocket",
    "consensus_rmsf",
    "consensus_torsions",
    "consensus_dccm",
    "dihedral_pca_ref",
]


def build_shared_analysis_protocol(
    gold_columns: Optional[List[str]] = None,
    ligand_selection: Optional[str] = None,
    pin_tools: Optional[List[str]] = None,
    metric_groups: Optional[List[str]] = None,
) -> AnalysisRecipe:
    """Shared analysis protocol cloned onto every system. Not generated per protein by an LLM."""
    lig_sel = (ligand_selection or "").strip() or None
    allowed = set(pin_tools) if pin_tools else None
    steps = []
    if lig_sel and (allowed is None or "calculate_consensus_pocket_metrics" in allowed):
        steps.append(
            RecipeStep(
                name="Ligand–pocket COM + axis-angle",
                tool_name="calculate_consensus_pocket_metrics",
                description="Ligand COM distance and orientation vs pocket_mapped residues",
                tool_params={
                    "ligand_selection": lig_sel,
                    "output_prefix": "reference_pocket",
                },
                reason="Campaign descriptor: pocket_mapped COM + axis-angle",
            )
        )
    optional = [
        RecipeStep(
            name="Consensus dihedrals (φ/ψ/χ₁)",
            tool_name="calculate_consensus_torsions",
            description="Mapped φ/ψ/χ₁ circular means/std for domain and pocket",
            tool_params={"output_dir": "consensus_dihedrals"},
            reason="Family modular: pocket χ₁ mean/std + shared PCA inputs",
        ),
        RecipeStep(
            name="Consensus Cα RMSF",
            tool_name="calculate_consensus_rmsf_features",
            description="Mapped consensus Cα RMSF mean/std across the domain",
            tool_params={"output_dir": "consensus_rmsf"},
            reason="Family modular: consensus RMSF features",
        ),
        RecipeStep(
            name="Consensus DCCM N↔C",
            tool_name="calculate_consensus_dccm_features",
            description="Mapped DCCM mean absolute and N-lobe↔C-lobe correlation",
            tool_params={"output_dir": "consensus_DCCM"},
            reason="Family modular: DCCM N↔C feature",
        ),
    ]
    for step in optional:
        if allowed is None or step.tool_name in allowed:
            steps.append(step)
    cols = list(gold_columns or [])
    groups = list(metric_groups or [])
    return AnalysisRecipe(
        steps=steps,
        required_feature_columns=cols,
        metric_groups=groups,
    )


build_family_recipe = build_shared_analysis_protocol  # alias kept for existing imports


def family_contract(
    modular_dirs: Optional[List[str]] = None,
    n_features: Optional[int] = None,
) -> ArtifactContract:
    dirs = list(modular_dirs) if modular_dirs else list(FAMILY_MODULAR_DIRS)
    nfeat = int(n_features) if n_features is not None else 1
    return ArtifactContract(
        modular_dirs=dirs,
        analysis_markers=["analysis_summary.jsonl"],
        min_features_present=max(1, nfeat),
        min_complete_fraction=0.85,
    )


def build_post_combined_protocol(
    *,
    reference_label: str = "",
    post_tools: Optional[List[str]] = None,
) -> AnalysisRecipe:
    """Combined-stage pins (shared-ref dynamics). Empty when the pack omits them."""
    tools = list(post_tools) if post_tools is not None else list(FAMILY_POST_CALCULATIONS)
    steps: List[RecipeStep] = []
    ref = (reference_label or "").strip()
    if "run_shared_dynamics_fel_batch" in tools or "compute_shared_pka_ref_dyn_features" in tools:
        steps.append(
            RecipeStep(
                name="Shared-reference dynamics features",
                tool_name="compute_shared_pka_ref_dyn_features",
                description="Project every system onto the reference dynamics model",
                tool_params={"reference_label": ref} if ref else {},
                reason="Family post pin: shared-reference dyn feature",
            )
        )
    return AnalysisRecipe(steps=steps, required_feature_columns=[], metric_groups=[])


def recipe_steps_as_plan_dicts(
    *,
    topology_file: str = "md.tpr",
    trajectory_file: str = "mdWrap.xtc",
    recipe: Optional[AnalysisRecipe] = None,
) -> List[Dict[str, Any]]:
    """Render a shared analysis protocol as analysis-agent plan step dicts."""
    out: List[Dict[str, Any]] = []
    steps = list(recipe.steps) if recipe is not None and recipe.steps else []
    for step in steps:
        params = dict(step.tool_params)
        params.setdefault("topology_file", topology_file)
        params.setdefault("trajectory_file", trajectory_file)
        out.append(
            {
                "name": step.name,
                "description": step.description,
                "tool_name": step.tool_name,
                "tool_params": params,
                "reason": step.reason,
            }
        )
    return out
