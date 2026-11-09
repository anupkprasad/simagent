"""
Aggregate per-simulation analysis outputs into one classification-ready table.

After each simulation completes the binding-site + FEL pipeline, run
``collect_classification_features_table`` at the multi-simulation base to
produce:

  - ``classification_features.csv``       — raw scalar features (one row / sim)
  - ``classification_features_zscore.csv`` — column z-scores across sims (clustering input)
  - ``classification_features.xlsx``      — README + definitions + raw + z-score sheets
  - ``classification_features.json``      — column manifest + normalization note
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain.tools import tool

logger = logging.getLogger(__name__)

_BASE_AGENT_DIRS = frozenset({
    "analysis", "reporter", "supervisor", "planner", "programmer",
    "preprocess", "simsetup", "hpc",
})

# Metric groups → scalar columns for classification featurization.
# Keys align with detect_requested_metrics() in planning_guidelines.py.
CLASSIFICATION_FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "com": ("ligand_pocket_distance_mean_A", "ligand_pocket_distance_std_A"),
    "contacts": ("mean_contacts", "mean_hbonds", "max_contacts"),
    "pocket_sasa": ("mean_pocket_sasa_nm2", "std_pocket_sasa_nm2"),
    "residence": (
        "fraction_bound",
        "n_unbinding_events",
        "longest_bound_ns",
        "mean_bound_event_ns",
    ),
    "pocket_rmsf": ("mean_pocket_rmsf_A", "max_pocket_rmsf_A"),
    "ligand_rmsf": ("mean_ligand_rmsf_A", "max_ligand_rmsf_A"),
    "fel": (
        "n_basins",
        "landscape_entropy",
        "major_basin_population",
        "max_barrier_height_kJ_mol",
        "mean_basin_depth_kJ_mol",
    ),
    "rmsd": ("mean_rmsd_A", "std_rmsd_A"),
    "rmsf": ("mean_protein_rmsf_A", "max_protein_rmsf_A"),
    "rg": ("mean_rg_A", "std_rg_A"),
    "sasa": ("mean_protein_sasa_nm2", "std_protein_sasa_nm2"),
    "energy": ("mean_potential_energy_kJ_mol",),
    "dccm": ("mean_abs_dccm",),
    "reference_pocket": (
        "reference_pocket_ligand_distance_mean_A",
        "reference_pocket_ligand_distance_std_A",
        "reference_pocket_mean_hbonds",
        "reference_pocket_mean_sasa_nm2",
        "reference_pocket_std_sasa_nm2",
        "reference_pocket_p95_sasa_nm2",
        "reference_pocket_fraction_bound",
        "reference_pocket_mean_rmsf_A",
        "reference_pocket_max_rmsf_A",
        "reference_pocket_mean_ligand_axis_angle_deg",
        "reference_pocket_std_ligand_axis_angle_deg",
        "reference_pocket_ligand_axis_angle_p95_deg",
        "reference_pocket_ligand_distance_p95_A",
        "reference_pocket_ligand_distance_max_A",
        "reference_pocket_fraction_stable_coupling",
        "reference_pocket_residue_count",
        "reference_pocket_net_charge",
    ),
    "reference_fel": (
        "ref_n_basins",
        "ref_landscape_entropy",
        "ref_major_basin_population",
        "ref_max_barrier_height_kJ_mol",
        "ref_mean_basin_depth_kJ_mol",
        "ref_grid_entropy",
    ),
    "reference_pca": (
        "ref_pc1_mean",
        "ref_pc2_mean",
        "ref_pc1_std",
        "ref_pc2_std",
    ),
}

# Curated binding/dynamics descriptors for reference-structure archetype clustering.
# Excludes mapping artifacts (residue_count), static chemistry (net_charge),
# redundant COM mean (correlates with p95), and correlated FEL grid/barrier metadata.
REFERENCE_ARCHETYPE_METRIC_GROUPS: Tuple[str, ...] = (
    "reference_pocket_archetype",
    "reference_fel_archetype",
)

CLASSIFICATION_FEATURE_GROUPS["reference_pocket_archetype"] = (
    "reference_pocket_ligand_distance_std_A",
    "reference_pocket_ligand_distance_p95_A",
    "reference_pocket_fraction_bound",
    "reference_pocket_std_ligand_axis_angle_deg",
    "reference_pocket_ligand_axis_angle_p95_deg",
)
CLASSIFICATION_FEATURE_GROUPS["reference_fel_archetype"] = (
    "ref_landscape_entropy",
    "ref_major_basin_population",
)

# Paper Ward k=4 feature set (mafft_0.5 fig3 sharedpcdyn): pocket×4 + χ₁×2 + RMSF×2 + DCCM + shared PKA-ref dyn.
# Matches campaigns/ment/mafft_0.5_feature_analysis/draft/sharedpcdyn_lock.py FEATURE_COLS.
PAPER_WARD4_FEATURE_COLUMNS: Tuple[str, ...] = (
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
)
PAPER_WARD4_METRIC_GROUPS: Tuple[str, ...] = (
    "reference_pocket",
    "consensus_rmsf",
    "consensus_torsions",
    "consensus_dccm",
    "dihedral_pca_ref",
)
CLASSIFICATION_FEATURE_GROUPS["paper_ward4"] = PAPER_WARD4_FEATURE_COLUMNS

# Optional modular family-dynamics groups (user/planner selects; never forced).
CLASSIFICATION_FEATURE_GROUPS["consensus_torsions"] = (
    "chi1_circ_mean_deg",
    "chi1_circ_std_deg",
    "chi1_pocket_circ_mean_deg",
    "chi1_pocket_circ_std_deg",
)
CLASSIFICATION_FEATURE_GROUPS["consensus_rmsf"] = (
    "consensus_rmsf_mean_A",
    "consensus_rmsf_std_A",
)
CLASSIFICATION_FEATURE_GROUPS["consensus_dccm"] = (
    "dccm_N_C_mean_corr",
    "mean_abs_dccm",
)
CLASSIFICATION_FEATURE_GROUPS["dihedral_pca"] = (
    "pca_grid_entropy",
    "pca_major_basin_population",
)
CLASSIFICATION_FEATURE_GROUPS["dihedral_tica"] = (
    "tica_grid_entropy",
    "tica_major_basin_population",
)
CLASSIFICATION_FEATURE_GROUPS["cart_pca"] = (
    "cart_pca_grid_entropy",
    "cart_pca_major_basin_population",
)
CLASSIFICATION_FEATURE_GROUPS["cart_tica"] = (
    "cart_tica_grid_entropy",
    "cart_tica_major_basin_population",
)
CLASSIFICATION_FEATURE_GROUPS["dihedral_pca_ref"] = (
    "pca_ref_grid_entropy",
    "pca_ref_major_basin_population",
    "pca_pka_ref_shared_dyn",
    "pca_pka_ref_shared_pc_rms",
    "pca_pka_ref_shared_gmin_pc1",
    "pca_pka_ref_shared_gmin_pc2",
    "pca_pka_ref_shared_centroid_pc1",
    "pca_pka_ref_shared_centroid_pc2",
)
CLASSIFICATION_FEATURE_GROUPS["dihedral_tica_ref"] = (
    "tica_ref_grid_entropy",
    "tica_ref_major_basin_population",
)

# (subdir under analysis/, json relative path, column_prefix for grid/major)
_MODULAR_DYNAMICS_FEL_DIRS: Tuple[Tuple[str, str], ...] = (
    ("consensus_PCA", "pca"),
    ("consensus_TICA", "tica"),
    ("consensus_cart_PCA", "cart_pca"),
    ("consensus_cart_TICA", "cart_tica"),
    ("consensus_PCA_ref", "pca_ref"),
    ("consensus_TICA_ref", "tica_ref"),
    ("consensus_cart_PCA_ref", "cart_pca_ref"),
    ("consensus_cart_TICA_ref", "cart_tica_ref"),
)

_MODULAR_SCALAR_JSON: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "consensus_dihedrals/torsion_summary.json",
        (
            "chi1_circ_mean_deg",
            "chi1_circ_std_deg",
            "chi1_pocket_circ_mean_deg",
            "chi1_pocket_circ_std_deg",
        ),
    ),
    (
        "consensus_dihedrals/dihedral_features_meta.json",
        (
            "chi1_circ_mean_deg",
            "chi1_circ_std_deg",
            "chi1_pocket_circ_mean_deg",
            "chi1_pocket_circ_std_deg",
        ),
    ),
    (
        "consensus_rmsf/consensus_rmsf_features.json",
        ("consensus_rmsf_mean_A", "consensus_rmsf_std_A"),
    ),
    (
        "consensus_DCCM/consensus_dccm_features.json",
        ("dccm_N_C_mean_corr", "mean_abs_dccm"),
    ),
    (
        "shared_pc_features.json",
        (
            "pca_pka_ref_shared_dyn",
            "pca_pka_ref_shared_pc_rms",
            "pca_pka_ref_shared_gmin_pc1",
            "pca_pka_ref_shared_gmin_pc2",
            "pca_pka_ref_shared_centroid_pc1",
            "pca_pka_ref_shared_centroid_pc2",
        ),
    ),
    (
        "avg/shared_pc_features.json",
        (
            "pca_pka_ref_shared_dyn",
            "pca_pka_ref_shared_pc_rms",
            "pca_pka_ref_shared_gmin_pc1",
            "pca_pka_ref_shared_gmin_pc2",
            "pca_pka_ref_shared_centroid_pc1",
            "pca_pka_ref_shared_centroid_pc2",
        ),
    ),
)


def discover_modular_feature_columns(row: Dict[str, Any]) -> List[str]:
    """Columns present on a feature row that belong to modular family tools."""
    known: List[str] = []
    for cols in (
        CLASSIFICATION_FEATURE_GROUPS["consensus_torsions"],
        CLASSIFICATION_FEATURE_GROUPS["consensus_rmsf"],
        CLASSIFICATION_FEATURE_GROUPS["consensus_dccm"],
        CLASSIFICATION_FEATURE_GROUPS["dihedral_pca"],
        CLASSIFICATION_FEATURE_GROUPS["dihedral_tica"],
        CLASSIFICATION_FEATURE_GROUPS["cart_pca"],
        CLASSIFICATION_FEATURE_GROUPS["cart_tica"],
        CLASSIFICATION_FEATURE_GROUPS["dihedral_pca_ref"],
        CLASSIFICATION_FEATURE_GROUPS["dihedral_tica_ref"],
    ):
        for c in cols:
            if row.get(c) is not None and c not in known:
                known.append(c)
    # Also pick up any *_grid_entropy / *_major_basin_population already on row
    for k, v in row.items():
        if v is None:
            continue
        if k.endswith("_grid_entropy") or k.endswith("_major_basin_population"):
            if k not in known:
                known.append(k)
    return known


def _ingest_modular_scalars_from_dir(adir: Path, row: Dict[str, Any]) -> None:
    """Ingest modular family scalars from one analysis directory root."""
    for rel, keys in _MODULAR_SCALAR_JSON:
        data = _load_json(adir / rel) or {}
        for k in keys:
            if row.get(k) is None and data.get(k) is not None:
                row[k] = data.get(k)

    for subdir, prefix in _MODULAR_DYNAMICS_FEL_DIRS:
        fel = _load_json(adir / subdir / "fel_features.json") or {}
        if not fel:
            fel = _load_json(adir / subdir / "dynamics_features.json") or {}
        gkey = f"{prefix}_grid_entropy"
        mkey = f"{prefix}_major_basin_population"
        if row.get(gkey) is None and fel.get("grid_entropy") is not None:
            row[gkey] = fel.get("grid_entropy")
        if row.get(mkey) is None and fel.get("major_basin_population") is not None:
            row[mkey] = fel.get("major_basin_population")
        for src, dst_suffix in (
            ("landscape_entropy", "_landscape_entropy"),
            ("n_basins", "_n_basins"),
        ):
            dkey = f"{prefix}{dst_suffix}"
            if row.get(dkey) is None and fel.get(src) is not None:
                row[dkey] = fel.get(src)


def _mean_modular_scalars_across_reps(adir: Path, row: Dict[str, Any]) -> None:
    """Average modular JSON scalars across ``rep*/`` when avg/top-level missing."""
    rep_dirs = sorted(p for p in adir.glob("rep*") if p.is_dir())
    if len(rep_dirs) < 2:
        return
    # Collect per-key values from each rep, then write arithmetic means.
    buckets: Dict[str, List[float]] = {}
    for rep in rep_dirs:
        tmp: Dict[str, Any] = {}
        _ingest_modular_scalars_from_dir(rep, tmp)
        for k, v in tmp.items():
            if v is None:
                continue
            try:
                buckets.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                continue
    for k, vals in buckets.items():
        if row.get(k) is None and vals:
            row[k] = float(np.mean(vals))


def _ingest_modular_family_features(adir: Path, row: Dict[str, Any]) -> None:
    """Pull scalars from modular torsion / RMSF / DCCM / dynamics FEL artifacts.

    Prefers ``analysis/avg/`` then top-level ``analysis/``, then averages
    across ``rep*/`` (paper repavg = mean of per-replicate scalars).
    """
    for d in (adir / "avg", adir):
        if d.is_dir():
            _ingest_modular_scalars_from_dir(d, row)
    _mean_modular_scalars_across_reps(adir, row)


# Local (per-simulation) FEL counterpart to reference_fel_archetype — same two
# scalars, read from {uid}/analysis/fel_features.json rather than reference_fel/.
CLASSIFICATION_FEATURE_GROUPS["fel_archetype"] = (
    "landscape_entropy",
    "major_basin_population",
)

# When user asks for classification without naming specific metrics.
DEFAULT_CLASSIFICATION_METRIC_GROUPS: Tuple[str, ...] = (
    "com",
    "contacts",
    "pocket_sasa",
    "residence",
    "pocket_rmsf",
    "ligand_rmsf",
    "fel",
)

# All known columns (union of groups).
FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    dict.fromkeys(
        col
        for cols in CLASSIFICATION_FEATURE_GROUPS.values()
        for col in cols
    )
)

# Human-readable definitions for docs, manifest, and XLSX export.
CLASSIFICATION_FEATURE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "ligand_pocket_distance_mean_A": {
        "metric_group": "com",
        "description": "Mean center-of-mass distance between ATP and the binding pocket",
        "source_file": "ligand_pocket_distance.csv",
        "calculation": "Mean of the distance column over all trajectory frames",
        "unit": "Å",
    },
    "ligand_pocket_distance_std_A": {
        "metric_group": "com",
        "description": "Standard deviation of ligand–pocket COM distance",
        "source_file": "ligand_pocket_distance.csv",
        "calculation": "Std dev of distance_A / distance over frames",
        "unit": "Å",
    },
    "mean_contacts": {
        "metric_group": "contacts",
        "description": "Mean heavy-atom protein–ligand contacts per frame",
        "source_file": "protein_ligand_contacts.csv",
        "calculation": "Mean of n_contacts column",
        "unit": "count",
    },
    "mean_hbonds": {
        "metric_group": "contacts",
        "description": "Mean protein–ligand hydrogen bonds per frame",
        "source_file": "protein_ligand_contacts.csv",
        "calculation": "Mean of n_hbonds column",
        "unit": "count",
    },
    "max_contacts": {
        "metric_group": "contacts",
        "description": "Maximum contact count observed in any frame",
        "source_file": "protein_ligand_contacts.csv",
        "calculation": "Max of n_contacts column",
        "unit": "count",
    },
    "mean_pocket_sasa_nm2": {
        "metric_group": "pocket_sasa",
        "description": "Mean solvent-accessible surface area of the binding pocket",
        "source_file": "pocket_sasa.csv",
        "calculation": "Mean of pocket_sasa_nm2 column",
        "unit": "nm²",
    },
    "std_pocket_sasa_nm2": {
        "metric_group": "pocket_sasa",
        "description": "Std dev of pocket SASA over time",
        "source_file": "pocket_sasa.csv",
        "calculation": "Std dev of pocket_sasa_nm2 column",
        "unit": "nm²",
    },
    "fraction_bound": {
        "metric_group": "residence",
        "description": "Fraction of trajectory frames with ligand bound in pocket",
        "source_file": "ligand_residence.json",
        "calculation": "fraction_bound from analyze_ligand_residence",
        "unit": "0–1",
    },
    "n_unbinding_events": {
        "metric_group": "residence",
        "description": "Number of unbinding events detected",
        "source_file": "ligand_residence.json",
        "calculation": "n_unbinding_events from analyze_ligand_residence",
        "unit": "count",
    },
    "longest_bound_ns": {
        "metric_group": "residence",
        "description": "Longest continuous bound period",
        "source_file": "ligand_residence.json",
        "calculation": "longest_bound_ns from analyze_ligand_residence",
        "unit": "ns",
    },
    "mean_bound_event_ns": {
        "metric_group": "residence",
        "description": "Mean duration of bound events",
        "source_file": "ligand_residence.json",
        "calculation": "mean_bound_event_ns from analyze_ligand_residence",
        "unit": "ns",
    },
    "mean_pocket_rmsf_A": {
        "metric_group": "pocket_rmsf",
        "description": "Mean RMSF of binding-pocket residues",
        "source_file": "pocket_rmsf.dat",
        "calculation": "Mean of per-residue RMSF in pocket selection",
        "unit": "Å",
    },
    "max_pocket_rmsf_A": {
        "metric_group": "pocket_rmsf",
        "description": "Maximum pocket residue RMSF",
        "source_file": "pocket_rmsf.dat",
        "calculation": "Max of per-residue RMSF values",
        "unit": "Å",
    },
    "mean_ligand_rmsf_A": {
        "metric_group": "ligand_rmsf",
        "description": "Mean RMSF of ligand atoms",
        "source_file": "ligand_rmsf.json / ligand_rmsf.dat",
        "calculation": "mean_ligand_rmsf from calculate_ligand_rmsf",
        "unit": "Å",
    },
    "max_ligand_rmsf_A": {
        "metric_group": "ligand_rmsf",
        "description": "Maximum ligand atom RMSF",
        "source_file": "ligand_rmsf.json / ligand_rmsf.dat",
        "calculation": "max_ligand_rmsf from calculate_ligand_rmsf",
        "unit": "Å",
    },
    "n_basins": {
        "metric_group": "fel",
        "description": "Number of significant FEL basins (minima)",
        "source_file": "fel_features.json",
        "calculation": "n_basins or n_minima from analyze_fel_landscape_features",
        "unit": "count",
    },
    "landscape_entropy": {
        "metric_group": "fel",
        "description": "Conformational entropy of basin populations (S = −Σ p ln p)",
        "source_file": "fel_features.json",
        "calculation": "landscape_entropy from analyze_fel_landscape_features",
        "unit": "nats",
    },
    "major_basin_population": {
        "metric_group": "fel",
        "description": "Occupancy fraction of the largest FEL basin",
        "source_file": "fel_features.json",
        "calculation": "major_basin_population from analyze_fel_landscape_features",
        "unit": "0–1",
    },
    "max_barrier_height_kJ_mol": {
        "metric_group": "fel",
        "description": "Highest inter-basin free-energy barrier",
        "source_file": "fel_features.json",
        "calculation": "max_barrier_height_kJ_mol from analyze_fel_landscape_features",
        "unit": "kJ/mol",
    },
    "mean_basin_depth_kJ_mol": {
        "metric_group": "fel",
        "description": "Mean depth of FEL basins below local rim",
        "source_file": "fel_features.json",
        "calculation": "mean_basin_depth_kJ_mol from analyze_fel_landscape_features",
        "unit": "kJ/mol",
    },
    "mean_rmsd_A": {
        "metric_group": "rmsd",
        "description": "Mean protein Cα RMSD",
        "source_file": "rmsd.dat",
        "calculation": "Mean of RMSD column",
        "unit": "Å",
    },
    "std_rmsd_A": {
        "metric_group": "rmsd",
        "description": "Std dev of protein RMSD",
        "source_file": "rmsd.dat",
        "calculation": "Std dev of RMSD column",
        "unit": "Å",
    },
    "mean_protein_rmsf_A": {
        "metric_group": "rmsf",
        "description": "Mean Cα RMSF over all protein residues",
        "source_file": "rmsf.dat",
        "calculation": "Mean of per-residue RMSF values",
        "unit": "Å",
    },
    "max_protein_rmsf_A": {
        "metric_group": "rmsf",
        "description": "Maximum protein residue RMSF",
        "source_file": "rmsf.dat",
        "calculation": "Max of per-residue RMSF values",
        "unit": "Å",
    },
    "mean_rg_A": {
        "metric_group": "rg",
        "description": "Mean radius of gyration",
        "source_file": "gyration.dat",
        "calculation": "Mean of Rg column",
        "unit": "Å",
    },
    "std_rg_A": {
        "metric_group": "rg",
        "description": "Std dev of radius of gyration",
        "source_file": "gyration.dat",
        "calculation": "Std dev of Rg column",
        "unit": "Å",
    },
    "mean_protein_sasa_nm2": {
        "metric_group": "sasa",
        "description": "Mean whole-protein SASA",
        "source_file": "sasa.csv / sasa.dat",
        "calculation": "Mean SASA over frames",
        "unit": "nm²",
    },
    "std_protein_sasa_nm2": {
        "metric_group": "sasa",
        "description": "Std dev of whole-protein SASA",
        "source_file": "sasa.csv / sasa.dat",
        "calculation": "Std dev SASA over frames",
        "unit": "nm²",
    },
    "mean_potential_energy_kJ_mol": {
        "metric_group": "energy",
        "description": "Mean potential energy",
        "source_file": "energy.dat",
        "calculation": "Mean of potential energy column",
        "unit": "kJ/mol",
    },
    "mean_abs_dccm": {
        "metric_group": "dccm",
        "description": "Mean |cross-correlation| of Cα fluctuations",
        "source_file": "dccm_summary.json",
        "calculation": "mean_abs_correlation from DCCM analysis",
        "unit": "0–1",
    },
    "reference_pocket_ligand_distance_mean_A": {
        "metric_group": "reference_pocket",
        "description": "Mean ligand–reference-pocket COM distance",
        "source_file": "reference_pocket/{label}/reference_pocket_metrics.json",
        "calculation": "Mean distance from reference_pocket_ligand_distance.csv",
        "unit": "Å",
    },
    "reference_pocket_ligand_distance_std_A": {
        "metric_group": "reference_pocket",
        "description": "Std dev of ligand–reference-pocket COM distance",
        "source_file": "reference_pocket/{label}/reference_pocket_metrics.json",
        "calculation": "Std dev from reference_pocket_ligand_distance.csv",
        "unit": "Å",
    },
    "reference_pocket_mean_hbonds": {
        "metric_group": "reference_pocket",
        "description": "Mean pocket-restricted protein–ligand H-bonds",
        "source_file": "reference_pocket/{label}/reference_pocket_metrics.json",
        "calculation": "Mean n_hbonds from reference_pocket_contacts.csv",
        "unit": "count",
    },
    "reference_pocket_mean_sasa_nm2": {
        "metric_group": "reference_pocket",
        "description": "Mean SASA of reference-mapped pocket residues",
        "source_file": "reference_pocket/{label}/reference_pocket_metrics.json",
        "calculation": "Mean pocket_sasa_nm2 from reference_pocket_sasa.csv",
        "unit": "nm²",
    },
    "reference_pocket_std_sasa_nm2": {
        "metric_group": "reference_pocket",
        "description": "Std dev of reference pocket SASA",
        "source_file": "reference_pocket/{label}/reference_pocket_sasa.csv",
        "calculation": "Std dev pocket_sasa_nm2 over frames",
        "unit": "nm²",
    },
    "reference_pocket_p95_sasa_nm2": {
        "metric_group": "reference_pocket",
        "description": "95th percentile reference pocket SASA (opening/exposure excursion)",
        "source_file": "reference_pocket/{label}/reference_pocket_sasa.csv",
        "calculation": "P95 of pocket_sasa_nm2 over trajectory frames",
        "unit": "nm²",
    },
    "reference_pocket_fraction_bound": {
        "metric_group": "reference_pocket",
        "description": "Fraction of trajectory with ligand bound to reference pocket",
        "source_file": "reference_pocket/{label}/reference_pocket_residence.json",
        "calculation": "fraction_bound from residence analysis",
        "unit": "fraction",
    },
    "reference_pocket_mean_rmsf_A": {
        "metric_group": "reference_pocket",
        "description": "Mean Cα RMSF of reference-mapped pocket residues",
        "source_file": "reference_pocket/{label}/reference_pocket_rmsf.dat",
        "calculation": "Mean RMSF over mapped pocket Cα atoms",
        "unit": "Å",
    },
    "reference_pocket_mean_ligand_axis_angle_deg": {
        "metric_group": "reference_pocket",
        "description": "Mean angle between pocket Cα major axis and ATP heavy-atom major axis",
        "source_file": "reference_pocket/{label}/reference_pocket_ligand_orientation.csv",
        "calculation": "Undirected PCA axis angle (0–90°) per frame, trajectory mean",
        "unit": "degrees",
    },
    "reference_pocket_std_ligand_axis_angle_deg": {
        "metric_group": "reference_pocket",
        "description": "Std dev of pocket–ligand major-axis angle",
        "source_file": "reference_pocket/{label}/reference_pocket_ligand_orientation.csv",
        "calculation": "Std dev of axis_angle_deg time series",
        "unit": "degrees",
    },
    "reference_pocket_ligand_axis_angle_p95_deg": {
        "metric_group": "reference_pocket",
        "description": "95th percentile pocket–ligand major-axis angle (orientation excursion)",
        "source_file": "reference_pocket/{label}/reference_pocket_ligand_orientation.csv",
        "calculation": "P95 of axis_angle_deg over trajectory frames",
        "unit": "degrees",
    },
    "reference_pocket_ligand_distance_p95_A": {
        "metric_group": "reference_pocket",
        "description": "95th percentile ATP–pocket COM distance (excursion proxy)",
        "source_file": "reference_pocket/{label}/reference_pocket_ligand_orientation.csv",
        "calculation": "P95 of distance_angstrom over trajectory frames",
        "unit": "Å",
    },
    "reference_pocket_ligand_distance_max_A": {
        "metric_group": "reference_pocket",
        "description": "Maximum ATP–pocket COM distance over trajectory",
        "source_file": "reference_pocket/{label}/reference_pocket_ligand_orientation.csv",
        "calculation": "Max of distance_angstrom over trajectory frames",
        "unit": "Å",
    },
    "reference_pocket_fraction_stable_coupling": {
        "metric_group": "reference_pocket",
        "description": "Fraction of frames with COM ≤7 Å and axis angle ≤65°",
        "source_file": "reference_pocket/{label}/reference_pocket_ligand_orientation.csv",
        "calculation": "Joint COM + major-axis angle stability criterion",
        "unit": "fraction",
    },
    "reference_pocket_max_rmsf_A": {
        "metric_group": "reference_pocket",
        "description": "Max Cα RMSF among reference-mapped pocket residues",
        "source_file": "reference_pocket/{label}/reference_pocket_rmsf.dat",
        "calculation": "Max RMSF over mapped pocket Cα atoms",
        "unit": "Å",
    },
    "reference_pocket_residue_count": {
        "metric_group": "reference_pocket",
        "description": "Number of mapped reference-pocket residues in this simulation",
        "source_file": "reference_pocket/{label}/reference_pocket_metrics.json",
        "calculation": "len(pocket_resids) from reference mapping",
        "unit": "count",
    },
    "reference_pocket_net_charge": {
        "metric_group": "reference_pocket",
        "description": "Net formal charge of pocket residues (Arg/Lys +1, Asp/Glu -1, His neutral)",
        "source_file": "reference_pocket/{label}/reference_pocket_metrics.json",
        "calculation": "n_positive - n_negative over unique pocket residues",
        "unit": "e",
    },
    "ref_n_basins": {
        "metric_group": "reference_fel",
        "description": "Basin count from shared-reference FEL",
        "source_file": "reference_fel/{label}/fel_features.json",
        "calculation": "n_basins from reference-projected FEL",
        "unit": "count",
    },
    "ref_landscape_entropy": {
        "metric_group": "reference_fel",
        "description": "Landscape entropy from shared-reference FEL",
        "source_file": "reference_fel/{label}/fel_features.json",
        "calculation": "landscape_entropy from reference-projected FEL",
        "unit": "nats",
    },
    "ref_major_basin_population": {
        "metric_group": "reference_fel",
        "description": "Major basin population from shared-reference FEL",
        "source_file": "reference_fel/{label}/fel_features.json",
        "calculation": "major_basin_population",
        "unit": "fraction",
    },
    "ref_max_barrier_height_kJ_mol": {
        "metric_group": "reference_fel",
        "description": "Max inter-basin barrier from shared-reference FEL",
        "source_file": "reference_fel/{label}/fel_features.json",
        "calculation": "max_barrier_height_kJ_mol",
        "unit": "kJ/mol",
    },
    "ref_mean_basin_depth_kJ_mol": {
        "metric_group": "reference_fel",
        "description": "Mean basin depth from shared-reference FEL",
        "source_file": "reference_fel/{label}/fel_features.json",
        "calculation": "mean_basin_depth_kJ_mol",
        "unit": "kJ/mol",
    },
    "ref_grid_entropy": {
        "metric_group": "reference_fel",
        "description": "Grid entropy from shared-reference FEL",
        "source_file": "reference_fel/{label}/fel_features.json",
        "calculation": "grid_entropy",
        "unit": "nats",
    },
    "ref_pc1_mean": {
        "metric_group": "reference_pca",
        "description": "Mean PC1 from reference-projected consensus Cα PCA",
        "source_file": "reference_fel/{label}/reference_pca_projections.dat",
        "calculation": "Mean PC1 over trajectory",
        "unit": "Å",
    },
    "ref_pc2_mean": {
        "metric_group": "reference_pca",
        "description": "Mean PC2 from reference-projected consensus Cα PCA",
        "source_file": "reference_fel/{label}/reference_pca_projections.dat",
        "calculation": "Mean PC2 over trajectory",
        "unit": "Å",
    },
    "ref_pc1_std": {
        "metric_group": "reference_pca",
        "description": "Std dev PC1 from reference-projected PCA",
        "source_file": "reference_fel/{label}/reference_pca_projections.dat",
        "calculation": "Std dev PC1 over trajectory",
        "unit": "Å",
    },
    "ref_pc2_std": {
        "metric_group": "reference_pca",
        "description": "Std dev PC2 from reference-projected PCA",
        "source_file": "reference_fel/{label}/reference_pca_projections.dat",
        "calculation": "Std dev PC2 over trajectory",
        "unit": "Å",
    },
}


def columns_for_metric_groups(groups: Optional[Tuple[str, ...]]) -> Tuple[str, ...]:
    """Resolve CSV column list from metric group names."""
    if not groups:
        return FEATURE_COLUMNS
    cols: List[str] = []
    for g in groups:
        for c in CLASSIFICATION_FEATURE_GROUPS.get(g, ()):
            if c not in cols:
                cols.append(c)
    return tuple(cols) if cols else FEATURE_COLUMNS


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _csv_mean_std(
    path: Path, value_col: str, clean_pbc: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    if not path.is_file():
        return None, None
    vals: List[float] = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                vals.append(float(row[value_col]))
            except (KeyError, TypeError, ValueError):
                continue
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    if clean_pbc:
        # Strip transient PBC imaging spikes so the mean/std reflect the real
        # binding-site behaviour rather than periodic-image jumps.
        try:
            from src.analysis.pbc_utils import clean_pbc_distance_series

            cleaned, _, n_removed = clean_pbc_distance_series(arr)
            if n_removed:
                logger.info(
                    "%s: removed %d PBC spike(s) before feature stats",
                    path.name, n_removed,
                )
                arr = cleaned
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PBC clean skipped for %s: %s", path.name, exc)
    return float(np.mean(arr)), float(np.std(arr))


def _csv_percentile(path: Path, value_col: str, percentile: float = 95.0) -> Optional[float]:
    if not path.is_file():
        return None
    vals: List[float] = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                vals.append(float(row[value_col]))
            except (KeyError, TypeError, ValueError):
                continue
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), percentile))


def _csv_column_stats(path: Path, col: str) -> Dict[str, Optional[float]]:
    if not path.is_file():
        return {"mean": None, "max": None}
    vals: List[float] = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                vals.append(float(row[col]))
            except (KeyError, TypeError, ValueError):
                continue
    if not vals:
        return {"mean": None, "max": None}
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(np.mean(arr)), "max": float(np.max(arr))}


def _parse_pocket_rmsf_dat(path: Path) -> Dict[str, Optional[float]]:
    if not path.is_file():
        return {"mean": None, "max": None}
    vals: List[float] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    vals.append(float(parts[-1]))
                except ValueError:
                    continue
    if not vals:
        return {"mean": None, "max": None}
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(np.mean(arr)), "max": float(np.max(arr))}


def _mean_reference_pocket_metrics_across_reps(adir: Path) -> Dict[str, Any]:
    """Average ``reference_pocket_metrics.json`` scalars across ``rep*/``."""
    rep_dirs = sorted(p for p in adir.glob("rep*") if p.is_dir())
    if not rep_dirs:
        return {}
    buckets: Dict[str, List[float]] = {}
    for rep in rep_dirs:
        for cand in (
            rep / "reference_pocket_metrics.json",
            rep / "reference_pocket" / "reference_pocket_metrics.json",
        ):
            data = _load_json(cand) or {}
            if not data:
                continue
            for key in (
                "ligand_pocket_distance_mean_A",
                "ligand_pocket_distance_std_A",
                "ligand_pocket_distance_p95_A",
                "ligand_pocket_distance_max_A",
                "mean_axis_angle_deg",
                "std_axis_angle_deg",
                "p95_axis_angle_deg",
                "mean_hbonds",
                "mean_pocket_sasa_nm2",
                "fraction_bound",
                "mean_pocket_rmsf_A",
                "fraction_stable_coupling",
            ):
                if data.get(key) is None:
                    continue
                try:
                    buckets.setdefault(key, []).append(float(data[key]))
                except (TypeError, ValueError):
                    continue
            break
    return {k: float(np.mean(v)) for k, v in buckets.items() if v}


def _analysis_search_dirs(analysis_dir: Path) -> List[Path]:
    """Search order for collector file discovery.

    Prefer ``rep*/`` (real per-replicate traj metrics), then top-level
    ``analysis/``, then ``avg/`` last — replicate-aggregate CSV dumps are often
    headerless ``x mean`` files that are not usable as trajectory series.
    """
    dirs: List[Path] = []
    for rep in sorted(analysis_dir.glob("rep*")):
        if rep.is_dir():
            dirs.append(rep)
    dirs.append(analysis_dir)
    avg = analysis_dir / "avg"
    if avg.is_dir():
        dirs.append(avg)
    return dirs


def _find_in_analysis(analysis_dir: Path, *names: str) -> Optional[Path]:
    for d in _analysis_search_dirs(analysis_dir):
        for name in names:
            p = d / name
            if p.is_file():
                return p
    return None


def _looks_like_tabular_csv(path: Path, required_cols: Sequence[str]) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(8):
                line = fh.readline()
                if not line:
                    break
                if line.lstrip().startswith("#"):
                    continue
                header = {c.strip().lower() for c in line.strip().split(",")}
                if any(c.lower() in header for c in required_cols):
                    return True
                break
    except OSError:
        return False
    return False


def _find_ligand_pocket_csv(analysis_dir: Path) -> Optional[Path]:
    # Prefer real trajectory CSVs (frame,distance_…) over avg x/y mean dumps.
    req = ("distance_a", "distance_angstrom", "distance")
    for d in _analysis_search_dirs(analysis_dir):
        for name in ("ligand_pocket_distance.csv", "pocket_distance.csv"):
            p = d / name
            if p.is_file() and _looks_like_tabular_csv(p, req):
                return p
    return None


def _extract_from_summary(analysis_dir: Path) -> Dict[str, Any]:
    """Fallback: last statistics block per analysis_type in analysis_summary.jsonl."""
    summary_path = analysis_dir / "analysis_summary.jsonl"
    if not summary_path.is_file():
        return {}
    out: Dict[str, Any] = {}
    try:
        text = summary_path.read_text(encoding="utf-8")
        # File may be pretty-printed multi-object JSON; split on "}\n{" roughly
        chunks = text.replace("}\n{", "}|SPLIT|{").split("|SPLIT|")
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if not chunk.startswith("{"):
                chunk = "{" + chunk
            if not chunk.endswith("}"):
                chunk = chunk + "}"
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            atype = obj.get("analysis_type")
            stats = obj.get("statistics") or {}
            if atype and stats:
                out[atype] = stats
    except Exception as exc:
        logger.debug("summary parse: %s", exc)
    return out


def _resolve_reference_fel_json(
    label: str,
    adir: Path,
    base_analysis: Path,
) -> Dict[str, Any]:
    """Load reference FEL features from combined ``reference_fel/{label}/`` tree."""
    from src.analysis.reference_labels import reference_fel_paths_for_sim

    candidates = reference_fel_paths_for_sim(label, base_analysis, "fel_features.json")
    candidates.extend([
        base_analysis / "reference_fel_features" / label / "fel_features.json",
        adir / "reference_fel_features.json",
    ])
    for path in candidates:
        data = _load_json(path)
        if data:
            return data
    return {}


def _resolve_reference_pca_projections_path(
    label: str,
    adir: Path,
    base_analysis: Path,
) -> Optional[Path]:
    """Locate reference-projected PCA file (combined tree preferred)."""
    from src.analysis.reference_labels import reference_fel_paths_for_sim

    candidates = reference_fel_paths_for_sim(
        label, base_analysis, "reference_pca_projections.dat"
    )
    candidates.extend([
        base_analysis / "reference_projected_pca" / label / "reference_pca_projections.dat",
        adir / "reference_pca_projections.dat",
    ])
    for path in candidates:
        if path.is_file():
            return path
    return None


def collect_features_for_sim(
    sim_dir: Path,
    *,
    local_fel_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build one feature dict for a single simulation directory.

    If ``local_fel_root`` is set (e.g. ``analysis/consensus_local_fel``),
    ``landscape_entropy`` / ``major_basin_population`` are read from
    ``{local_fel_root}/{label}/fel_features.json`` instead of
    ``{sim}/analysis/fel_features.json``.
    """
    label = sim_dir.name
    adir = sim_dir / "analysis"
    base_analysis = sim_dir.parent / "analysis"
    reference_pocket_dir = base_analysis / "reference_pocket" / label
    reference_pocket_manifest = (
        _load_json(base_analysis / "reference_pocket_batch_manifest.json") or {}
    )
    from src.analysis.reference_labels import is_reference_pocket_usable

    reference_pocket_usable = is_reference_pocket_usable(
        label,
        reference_pocket_manifest,
        base_analysis=base_analysis,
    )
    row: Dict[str, Any] = {"label": label, "sim_directory": str(sim_dir.resolve())}

    # Ligand–pocket distance
    lp_csv = _find_ligand_pocket_csv(adir)
    if lp_csv:
        for col in ("distance_A", "distance_angstrom", "distance"):
            m, s = _csv_mean_std(lp_csv, col, clean_pbc=True)
            if m is not None:
                row["ligand_pocket_distance_mean_A"] = m
                row["ligand_pocket_distance_std_A"] = s
                break

    # Contacts
    contacts = adir / "protein_ligand_contacts.csv"
    cstats = _csv_column_stats(contacts, "n_contacts")
    row["mean_contacts"] = cstats["mean"]
    row["max_contacts"] = cstats["max"]
    hstats = _csv_column_stats(contacts, "n_hbonds")
    row["mean_hbonds"] = hstats["mean"]

    # Pocket SASA
    psasa = adir / "pocket_sasa.csv"
    m, s = _csv_mean_std(psasa, "pocket_sasa_nm2")
    row["mean_pocket_sasa_nm2"] = m
    row["std_pocket_sasa_nm2"] = s

    # Residence
    res = _load_json(adir / "ligand_residence.json") or {}
    row["fraction_bound"] = res.get("fraction_bound")
    row["n_unbinding_events"] = res.get("n_unbinding_events")
    row["longest_bound_ns"] = res.get("longest_bound_ns")
    row["mean_bound_event_ns"] = res.get("mean_bound_event_ns")

    # Pocket RMSF
    prmsf = _parse_pocket_rmsf_dat(adir / "pocket_rmsf.dat")
    row["mean_pocket_rmsf_A"] = prmsf["mean"]
    row["max_pocket_rmsf_A"] = prmsf["max"]

    # Ligand RMSF
    lrmsf = _load_json(adir / "ligand_rmsf.json") or {}
    row["mean_ligand_rmsf_A"] = lrmsf.get("mean_ligand_rmsf")
    row["max_ligand_rmsf_A"] = lrmsf.get("max_ligand_rmsf")

    # FEL features (optional alternate tree, e.g. consensus_local_fel/)
    if local_fel_root is not None:
        fel = _load_json(Path(local_fel_root) / label / "fel_features.json") or {}
    else:
        fel_path = _find_in_analysis(adir, "fel_features.json")
        fel = _load_json(fel_path) if fel_path else {}
        fel = fel or {}
    row["n_basins"] = fel.get("n_basins", fel.get("n_minima"))
    row["landscape_entropy"] = fel.get("landscape_entropy")
    row["major_basin_population"] = fel.get("major_basin_population")
    row["max_barrier_height_kJ_mol"] = fel.get("max_barrier_height_kJ_mol")
    row["mean_basin_depth_kJ_mol"] = fel.get("mean_basin_depth_kJ_mol")

    # Whole-protein metrics (standard per-sim outputs)
    rmsd_m, rmsd_s = _csv_mean_std(adir / "rmsd.dat", "RMSD")
    if rmsd_m is None:
        for col in ("RMSD(Angstrom)", "rmsd", "RMSD"):
            rmsd_m, rmsd_s = _csv_mean_std(adir / "rmsd.dat", col)
            if rmsd_m is not None:
                break
    row["mean_rmsd_A"] = rmsd_m
    row["std_rmsd_A"] = rmsd_s

    prot_rmsf = _parse_pocket_rmsf_dat(adir / "rmsf.dat")
    row["mean_protein_rmsf_A"] = prot_rmsf["mean"]
    row["max_protein_rmsf_A"] = prot_rmsf["max"]

    rg_m, rg_s = _csv_mean_std(adir / "gyration.dat", "Rg")
    if rg_m is None:
        for col in ("Rg(Angstrom)", "Rg", "gyration"):
            rg_m, rg_s = _csv_mean_std(adir / "gyration.dat", col)
            if rg_m is not None:
                break
    row["mean_rg_A"] = rg_m
    row["std_rg_A"] = rg_s

    sasa_m, sasa_s = _csv_mean_std(adir / "sasa.csv", "sasa_nm2")
    if sasa_m is None:
        sasa_m, sasa_s = _csv_mean_std(adir / "sasa.csv", "SASA")
    row["mean_protein_sasa_nm2"] = sasa_m
    row["std_protein_sasa_nm2"] = sasa_s

    energy_m, _ = _csv_mean_std(adir / "energy.dat", "Potential")
    if energy_m is None:
        energy_m, _ = _csv_mean_std(adir / "energy.dat", "potential_energy")
    row["mean_potential_energy_kJ_mol"] = energy_m

    dccm_summary = _load_json(adir / "dccm_summary.json")
    row["mean_abs_dccm"] = (
        dccm_summary.get("mean_abs_correlation") if dccm_summary else None
    )

    # Reference-mapped pocket metrics.
    # Prefer averaging ``analysis/rep*/reference_pocket_metrics.json`` (multi-rep
    # campaigns write metrics per replicate). Only then fall back to combined
    # ``analysis/reference_pocket/{label}/`` or local ligand_pocket CSVs.
    # Skip empty / headerless avg/*.csv dumps (they trigger "No numeric data").
    rep_avg = _mean_reference_pocket_metrics_across_reps(adir)
    rp_metrics: Dict[str, Any] = {}
    if rep_avg:
        # Map rep-averaged keys into the row's reference_pocket_* columns below.
        rp_metrics = {
            "ligand_pocket_distance_mean_A": rep_avg.get("ligand_pocket_distance_mean_A"),
            "ligand_pocket_distance_std_A": rep_avg.get("ligand_pocket_distance_std_A"),
            "ligand_pocket_distance_p95_A": rep_avg.get("ligand_pocket_distance_p95_A"),
            "ligand_pocket_distance_max_A": rep_avg.get("ligand_pocket_distance_max_A"),
            "mean_axis_angle_deg": rep_avg.get("mean_axis_angle_deg"),
            "std_axis_angle_deg": rep_avg.get("std_axis_angle_deg"),
            "p95_axis_angle_deg": rep_avg.get("p95_axis_angle_deg"),
            "mean_hbonds": rep_avg.get("mean_hbonds"),
            "mean_pocket_sasa_nm2": rep_avg.get("mean_pocket_sasa_nm2"),
            "fraction_bound": rep_avg.get("fraction_bound"),
            "mean_pocket_rmsf_A": rep_avg.get("mean_pocket_rmsf_A"),
            "fraction_stable_coupling": rep_avg.get("fraction_stable_coupling"),
        }
    if not any(v is not None for v in rp_metrics.values()):
        rp_metrics = (
            _load_json(reference_pocket_dir / "reference_pocket_metrics.json") or {}
            if reference_pocket_usable
            else {}
        )
        if not rp_metrics and reference_pocket_dir.is_dir():
            rp_metrics = _load_json(reference_pocket_dir / "reference_pocket_metrics.json") or {}
    if rp_metrics:
        row["reference_pocket_ligand_distance_mean_A"] = rp_metrics.get(
            "ligand_pocket_distance_mean_A"
        )
        row["reference_pocket_ligand_distance_std_A"] = rp_metrics.get(
            "ligand_pocket_distance_std_A"
        )
        row["reference_pocket_mean_hbonds"] = rp_metrics.get("mean_hbonds")
        row["reference_pocket_mean_sasa_nm2"] = rp_metrics.get("mean_pocket_sasa_nm2")
        row["reference_pocket_fraction_bound"] = rp_metrics.get("fraction_bound")
        row["reference_pocket_mean_rmsf_A"] = rp_metrics.get("mean_pocket_rmsf_A")
        row["reference_pocket_mean_ligand_axis_angle_deg"] = rp_metrics.get(
            "mean_axis_angle_deg"
        )
        # Paper Ward-4 / fig4 column name (alias of mean_ligand_axis_angle).
        row["reference_pocket_ligand_axis_angle_mean_deg"] = row[
            "reference_pocket_mean_ligand_axis_angle_deg"
        ]
        row["reference_pocket_std_ligand_axis_angle_deg"] = rp_metrics.get(
            "std_axis_angle_deg"
        )
        row["reference_pocket_ligand_axis_angle_p95_deg"] = rp_metrics.get(
            "p95_axis_angle_deg"
        )
        row["reference_pocket_ligand_distance_p95_A"] = rp_metrics.get(
            "ligand_pocket_distance_p95_A"
        )
        row["reference_pocket_ligand_distance_max_A"] = rp_metrics.get(
            "ligand_pocket_distance_max_A"
        )
        row["reference_pocket_fraction_stable_coupling"] = rp_metrics.get(
            "fraction_stable_coupling"
        )
        row["reference_pocket_residue_count"] = rp_metrics.get("pocket_residue_count")
        row["reference_pocket_net_charge"] = rp_metrics.get("pocket_net_charge")
    rp_sasa = reference_pocket_dir / "reference_pocket_sasa.csv"
    if rp_sasa.is_file() and _looks_like_tabular_csv(rp_sasa, ("pocket_sasa_nm2", "sasa")):
        m, s = _csv_mean_std(rp_sasa, "pocket_sasa_nm2")
        if m is not None:
            row["reference_pocket_mean_sasa_nm2"] = m
            row["reference_pocket_std_sasa_nm2"] = s
        p95_sasa = _csv_percentile(rp_sasa, "pocket_sasa_nm2", 95.0)
        if p95_sasa is not None:
            row["reference_pocket_p95_sasa_nm2"] = p95_sasa
    rp_rmsf = _parse_pocket_rmsf_dat(reference_pocket_dir / "reference_pocket_rmsf.dat")
    if rp_rmsf["mean"] is not None:
        row["reference_pocket_mean_rmsf_A"] = rp_rmsf["mean"]
        row["reference_pocket_max_rmsf_A"] = rp_rmsf["max"]

    # Fill any remaining gaps from rep averages (already preferred above).
    if row.get("reference_pocket_ligand_axis_angle_mean_deg") is None or row.get(
        "reference_pocket_std_ligand_axis_angle_deg"
    ) is None:
        if not rep_avg:
            rep_avg = _mean_reference_pocket_metrics_across_reps(adir)
        if rep_avg:
            if row.get("reference_pocket_ligand_distance_mean_A") is None:
                row["reference_pocket_ligand_distance_mean_A"] = rep_avg.get(
                    "ligand_pocket_distance_mean_A"
                )
            if row.get("reference_pocket_ligand_distance_std_A") is None:
                row["reference_pocket_ligand_distance_std_A"] = rep_avg.get(
                    "ligand_pocket_distance_std_A"
                )
            if row.get("reference_pocket_mean_ligand_axis_angle_deg") is None:
                row["reference_pocket_mean_ligand_axis_angle_deg"] = rep_avg.get(
                    "mean_axis_angle_deg"
                )
            if row.get("reference_pocket_ligand_axis_angle_mean_deg") is None:
                row["reference_pocket_ligand_axis_angle_mean_deg"] = row.get(
                    "reference_pocket_mean_ligand_axis_angle_deg"
                )
            if row.get("reference_pocket_std_ligand_axis_angle_deg") is None:
                row["reference_pocket_std_ligand_axis_angle_deg"] = rep_avg.get(
                    "std_axis_angle_deg"
                )
            if row.get("reference_pocket_ligand_axis_angle_p95_deg") is None:
                row["reference_pocket_ligand_axis_angle_p95_deg"] = rep_avg.get(
                    "p95_axis_angle_deg"
                )
            for dst, src in (
                ("reference_pocket_ligand_distance_p95_A", "ligand_pocket_distance_p95_A"),
                ("reference_pocket_ligand_distance_max_A", "ligand_pocket_distance_max_A"),
                ("reference_pocket_mean_hbonds", "mean_hbonds"),
                ("reference_pocket_mean_sasa_nm2", "mean_pocket_sasa_nm2"),
                ("reference_pocket_fraction_bound", "fraction_bound"),
                ("reference_pocket_mean_rmsf_A", "mean_pocket_rmsf_A"),
                ("reference_pocket_fraction_stable_coupling", "fraction_stable_coupling"),
            ):
                if row.get(dst) is None and rep_avg.get(src) is not None:
                    row[dst] = rep_avg[src]

    orient_csv = reference_pocket_dir / "reference_pocket_ligand_orientation.csv"
    if not orient_csv.is_file():
        # Local per-sim orientation if a tool wrote it under analysis/ (incl. reps).
        for cand in (
            adir / "reference_pocket_ligand_orientation.csv",
            adir / "pocket_axis_angle.csv",
            adir / "avg" / "reference_pocket_ligand_orientation.csv",
            adir / "avg" / "pocket_axis_angle.csv",
        ):
            if cand.is_file() and _looks_like_tabular_csv(
                cand, ("axis_angle_deg", "axis_angle")
            ):
                orient_csv = cand
                break
        if not orient_csv.is_file() or not _looks_like_tabular_csv(
            orient_csv, ("axis_angle_deg", "axis_angle")
        ):
            for rep in sorted(adir.glob("rep*")):
                if not rep.is_dir():
                    continue
                for name in (
                    "reference_pocket_ligand_orientation.csv",
                    "pocket_axis_angle.csv",
                    "reference_pocket/reference_pocket_ligand_orientation.csv",
                ):
                    cand = rep / name
                    if cand.is_file() and _looks_like_tabular_csv(
                        cand, ("axis_angle_deg", "axis_angle")
                    ):
                        orient_csv = cand
                        break
                if orient_csv.is_file() and _looks_like_tabular_csv(
                    orient_csv, ("axis_angle_deg", "axis_angle")
                ):
                    break
    # Orientation CSV is a fallback only. Do not overwrite rep-averaged
    # reference_pocket_metrics (axis mean/std) with a single-rep CSV.
    need_orient_mean = row.get("reference_pocket_mean_ligand_axis_angle_deg") is None
    need_orient_std = row.get("reference_pocket_std_ligand_axis_angle_deg") is None
    if need_orient_mean or need_orient_std:
        orient_m, orient_s = _csv_mean_std(orient_csv, "axis_angle_deg")
        if orient_m is None:
            orient_m, orient_s = _csv_mean_std(orient_csv, "axis_angle")
        if need_orient_mean and orient_m is not None:
            row["reference_pocket_mean_ligand_axis_angle_deg"] = orient_m
        if need_orient_std and orient_s is not None:
            row["reference_pocket_std_ligand_axis_angle_deg"] = orient_s
    if orient_csv.is_file() and _looks_like_tabular_csv(
        orient_csv, ("axis_angle_deg", "axis_angle")
    ) and (
        row.get("reference_pocket_ligand_distance_p95_A") is None
        or row.get("reference_pocket_ligand_axis_angle_p95_deg") is None
        or row.get("reference_pocket_mean_ligand_axis_angle_deg") is None
        or row.get("reference_pocket_std_ligand_axis_angle_deg") is None
    ):
        from src.analysis.consensus_pocket import summarize_ligand_orientation_csv

        derived = summarize_ligand_orientation_csv(orient_csv)
        if derived.get("success"):
            if row.get("reference_pocket_ligand_distance_p95_A") is None:
                row["reference_pocket_ligand_distance_p95_A"] = derived.get(
                    "p95_distance_A"
                )
            if row.get("reference_pocket_ligand_distance_max_A") is None:
                row["reference_pocket_ligand_distance_max_A"] = derived.get(
                    "max_distance_A"
                )
            if row.get("reference_pocket_fraction_stable_coupling") is None:
                row["reference_pocket_fraction_stable_coupling"] = derived.get(
                    "fraction_stable_coupling"
                )
            if row.get("reference_pocket_mean_ligand_axis_angle_deg") is None:
                row["reference_pocket_mean_ligand_axis_angle_deg"] = derived.get(
                    "mean_axis_angle_deg"
                )
            if row.get("reference_pocket_std_ligand_axis_angle_deg") is None:
                row["reference_pocket_std_ligand_axis_angle_deg"] = derived.get(
                    "std_axis_angle_deg"
                )
            if row.get("reference_pocket_ligand_axis_angle_p95_deg") is None:
                row["reference_pocket_ligand_axis_angle_p95_deg"] = derived.get(
                    "p95_axis_angle_deg"
                )
    # Keep paper Ward-4 alias in sync with collector column name.
    if row.get("reference_pocket_ligand_axis_angle_mean_deg") is None:
        row["reference_pocket_ligand_axis_angle_mean_deg"] = row.get(
            "reference_pocket_mean_ligand_axis_angle_deg"
        )
    elif row.get("reference_pocket_mean_ligand_axis_angle_deg") is None:
        row["reference_pocket_mean_ligand_axis_angle_deg"] = row.get(
            "reference_pocket_ligand_axis_angle_mean_deg"
        )
    rp_res = _load_json(reference_pocket_dir / "reference_pocket_residence.json") or {}
    if rp_res.get("fraction_bound") is not None:
        row["reference_pocket_fraction_bound"] = rp_res.get("fraction_bound")

    # Fallback: local ligand–pocket COM → reference_pocket distance columns.
    if row.get("reference_pocket_ligand_distance_mean_A") is None and row.get(
        "ligand_pocket_distance_mean_A"
    ) is not None:
        row["reference_pocket_ligand_distance_mean_A"] = row[
            "ligand_pocket_distance_mean_A"
        ]
        row["reference_pocket_ligand_distance_std_A"] = row.get(
            "ligand_pocket_distance_std_A"
        )
        if lp_csv is not None:
            p95 = _csv_percentile(lp_csv, "distance_A", 95.0)
            if p95 is None:
                for col in ("distance_angstrom", "distance"):
                    p95 = _csv_percentile(lp_csv, col, 95.0)
                    if p95 is not None:
                        break
            if p95 is not None:
                row.setdefault("reference_pocket_ligand_distance_p95_A", p95)

    # Reference-projected dynamics — combined ``reference_fel/{label}/`` only
    ref_fel = _resolve_reference_fel_json(label, adir, base_analysis)
    if ref_fel:
        row["ref_n_basins"] = ref_fel.get("n_basins", ref_fel.get("n_minima"))
        row["ref_landscape_entropy"] = ref_fel.get("landscape_entropy")
        row["ref_major_basin_population"] = ref_fel.get("major_basin_population")
        row["ref_max_barrier_height_kJ_mol"] = ref_fel.get("max_barrier_height_kJ_mol")
        row["ref_mean_basin_depth_kJ_mol"] = ref_fel.get("mean_basin_depth_kJ_mol")
        row["ref_grid_entropy"] = ref_fel.get("grid_entropy")
    ref_proj_path = _resolve_reference_pca_projections_path(label, adir, base_analysis)
    if ref_proj_path and ref_proj_path.is_file():
        try:
            from src.analysis.pca_analyzer import load_pca_projections

            loaded = load_pca_projections(str(ref_proj_path))
            if loaded.get("success"):
                proj = loaded["projections"]
                if proj.shape[1] >= 1:
                    row["ref_pc1_mean"] = float(np.mean(proj[:, 0]))
                    row["ref_pc1_std"] = float(np.std(proj[:, 0]))
                if proj.shape[1] >= 2:
                    row["ref_pc2_mean"] = float(np.mean(proj[:, 1]))
                    row["ref_pc2_std"] = float(np.std(proj[:, 1]))
        except Exception as exc:
            logger.debug("reference PCA load for %s: %s", label, exc)

    # Modular family dynamics (torsions / consensus RMSF-DCCM / PCA-tICA FELs)
    _ingest_modular_family_features(adir, row)

    # Summary fallbacks for missing fields
    if any(row.get(c) is None for c in FEATURE_COLUMNS):
        by_type = _extract_from_summary(adir)
        if row["mean_contacts"] is None and "ProteinLigandContacts" in by_type:
            st = by_type["ProteinLigandContacts"]
            row["mean_contacts"] = st.get("mean_contacts")
            row["mean_hbonds"] = st.get("mean_hbonds")
            row["max_contacts"] = st.get("max_contacts")
        if row["mean_pocket_rmsf_A"] is None and "PocketRMSF" in by_type:
            st = by_type["PocketRMSF"]
            row["mean_pocket_rmsf_A"] = st.get("mean_pocket_rmsf")
            row["max_pocket_rmsf_A"] = st.get("max_pocket_rmsf")
        if row["mean_ligand_rmsf_A"] is None and "LigandRMSF" in by_type:
            st = by_type["LigandRMSF"]
            row["mean_ligand_rmsf_A"] = st.get("mean_ligand_rmsf")
            row["max_ligand_rmsf_A"] = st.get("max_ligand_rmsf")
        if row["max_barrier_height_kJ_mol"] is None and "FELFeatures" in by_type:
            st = by_type["FELFeatures"]
            row["n_basins"] = st.get("n_minima")
            row["landscape_entropy"] = st.get("landscape_entropy")
            row["major_basin_population"] = st.get("major_basin_population")
            row["max_barrier_height_kJ_mol"] = st.get("max_barrier_height_kJ_mol")
        if row["mean_rmsd_A"] is None and "RMSD" in by_type:
            st = by_type["RMSD"]
            row["mean_rmsd_A"] = st.get("mean_rmsd") or st.get("mean_rmsd_angstrom")
        if row["mean_protein_rmsf_A"] is None and "RMSF" in by_type:
            st = by_type["RMSF"]
            row["mean_protein_rmsf_A"] = st.get("mean_rmsf") or st.get("mean_rmsf_angstrom")
        if row["mean_rg_A"] is None and "RadiusOfGyration" in by_type:
            st = by_type["RadiusOfGyration"]
            row["mean_rg_A"] = st.get("mean_rg") or st.get("mean_rg_angstrom")
        if row.get("mean_abs_dccm") is None and "DCCM" in by_type:
            st = by_type["DCCM"]
            row["mean_abs_dccm"] = st.get("mean_abs_correlation")

    return row


def _is_finite_number(value: Any) -> bool:
    """True when ``value`` is a usable numeric scalar (rejects None / NaN / Inf)."""
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _count_present(row: Dict[str, Any], columns: Sequence[str]) -> int:
    return sum(1 for c in columns if _is_finite_number(row.get(c)))


def _zscore_table(
    rows: List[Dict[str, Any]],
    columns: Tuple[str, ...],
    method: str = "zscore",
) -> List[Dict[str, Any]]:
    """Normalize numeric columns across simulations for clustering.

    ``method="zscore"`` (default): classic ``(x - μ) / σ``.

    ``method="robust"``: paper Ward-4 / fig4 scaling — per-column IQR winsorize
    at 1.5×IQR, then ``(x - median) / (IQR/1.349)``, clipped to ±3
    (see ``campaigns/ment/mafft_0.5_feature_analysis/scripts/cluster_ward4.py``).

    Non-finite values (NaN/Inf) are treated as missing so one bad sim does not
    poison an entire column.
    """
    zrows: List[Dict[str, Any]] = []
    for row in rows:
        zrows.append({"label": row["label"], "sim_directory": row.get("sim_directory")})

    use_robust = str(method or "zscore").lower() in {"robust", "iqr", "robust_z"}
    for col in columns:
        vals = [
            float(row.get(col))
            for row in rows
            if _is_finite_number(row.get(col))
        ]
        if len(vals) < 2:
            for zr in zrows:
                zr[col] = None
            continue
        arr = np.asarray(vals, dtype=float)
        if use_robust:
            q1, median, q3 = np.percentile(arr, [25.0, 50.0, 75.0])
            iqr = float(q3 - q1)
            lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            scale = (iqr / 1.349) if iqr > 1e-12 else 1.0
            for i, row in enumerate(rows):
                v = row.get(col)
                if not _is_finite_number(v):
                    zrows[i][col] = None
                else:
                    wins = float(np.clip(float(v), lower, upper))
                    zrows[i][col] = float(np.clip((wins - float(median)) / scale, -3.0, 3.0))
            continue
        mu, sigma = float(np.mean(arr)), float(np.std(arr))
        if sigma < 1e-12:
            sigma = 1.0
        for i, row in enumerate(rows):
            v = row.get(col)
            if not _is_finite_number(v):
                zrows[i][col] = None
            else:
                zrows[i][col] = (float(v) - mu) / sigma
    return zrows


def _feature_definitions_rows(feature_cols: Tuple[str, ...]) -> List[Dict[str, str]]:
    """Build metadata rows for docs / manifest / XLSX."""
    rows: List[Dict[str, str]] = []
    for col in feature_cols:
        meta = CLASSIFICATION_FEATURE_DEFINITIONS.get(col, {})
        rows.append({
            "column": col,
            "metric_group": meta.get("metric_group", ""),
            "description": meta.get("description", ""),
            "source_file": meta.get("source_file", ""),
            "calculation": meta.get("calculation", ""),
            "unit": meta.get("unit", ""),
            "used_for_clustering": "yes (z-score column)",
        })
    return rows


def _write_classification_xlsx(
    out_path: Path,
    raw_rows: List[Dict[str, Any]],
    zrows: List[Dict[str, Any]],
    feature_cols: Tuple[str, ...],
    fieldnames: List[str],
    z_fieldnames: List[str],
    manifest: Dict[str, Any],
) -> Optional[str]:
    """Write multi-sheet workbook: README, definitions, raw, z-score."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas unavailable; skipping classification XLSX export")
        return None

    try:
        readme_rows = [
            ("Purpose", "Feature matrix for unsupervised protein–ATP classification"),
            ("Clustering input", "ZScore_Features sheet (same as classification_features_zscore.csv)"),
            ("Physical interpretation", "Raw_Features sheet (same as classification_features.csv)"),
            ("Normalization", manifest.get("normalization", "")),
            ("Z-score formula", "z_i = (x_i - μ) / σ  where μ, σ are per-column across simulations"),
            ("n_simulations", str(manifest.get("n_simulations", ""))),
            ("requested_metric_groups", ", ".join(manifest.get("requested_metric_groups", []))),
        ]
        readme_df = pd.DataFrame(readme_rows, columns=["topic", "value"])
        defs_df = pd.DataFrame(_feature_definitions_rows(feature_cols))
        raw_cols = [c for c in fieldnames if c in (raw_rows[0] if raw_rows else {})]
        z_cols = [c for c in z_fieldnames if c in (zrows[0] if zrows else {})]
        raw_df = pd.DataFrame(raw_rows)[raw_cols]
        z_df = pd.DataFrame(zrows)[z_cols]

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            readme_df.to_excel(writer, sheet_name="README", index=False)
            defs_df.to_excel(writer, sheet_name="Feature_Definitions", index=False)
            raw_df.to_excel(writer, sheet_name="Raw_Features", index=False)
            z_df.to_excel(writer, sheet_name="ZScore_Features", index=False)

        return str(out_path.resolve())
    except ImportError:
        logger.warning(
            "openpyxl not installed; skip classification_features.xlsx "
            "(pip install openpyxl)"
        )
        return None
    except Exception as exc:
        logger.warning("Failed to write classification XLSX: %s", exc)
        return None


@tool
def collect_classification_features_table(
    base_directory: str,
    output_file: str = "classification_features.csv",
    zscore_output_file: str = "classification_features_zscore.csv",
    xlsx_output_file: str = "classification_features.xlsx",
    manifest_file: str = "classification_features.json",
    working_dir: Optional[str] = None,
    include_zscore: bool = True,
    requested_metric_groups: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None,
    zscore_method: Optional[str] = None,
    auto_discover: bool = False,
    allowed_labels: Optional[List[str]] = None,
    local_fel_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build one classification table from all per-simulation analysis outputs.

    **Only run when the user explicitly requests classification / clustering /
    unsupervised grouping** — it is not part of default combined analysis.

    Scans ``{base}/{label}/analysis/`` and writes one row per simulation.

    Column selection (flexible; pick one style):

    - ``feature_columns=[...]`` — exact CSV columns (highest priority)
    - ``requested_metric_groups=[...]`` — named groups (``com``, ``dihedral_pca``, …)
    - ``auto_discover=True`` — include modular family scalars found on disk
      (torsions, consensus RMSF/DCCM, PCA/tICA grid entropy, …) **plus**
      default binding/FEL groups when no groups were named
    - If all omitted: default binding-site + FEL bundle

    Do **not** invent a fixed mega feature matrix — pass only what the user goal asked for.
    """
    original_dir = None
    try:
        if working_dir:
            wd_abs = Path(working_dir).expanduser().resolve()
            wd_abs.mkdir(parents=True, exist_ok=True)
            original_dir = os.getcwd()
            os.chdir(wd_abs)
            working_dir = str(wd_abs)

        base = Path(base_directory)
        if not base.is_absolute():
            anchor = Path(original_dir) if original_dir else Path.cwd()
            base = (anchor / base).resolve()
        else:
            base = base.resolve()
        if not base.is_dir():
            return {"success": False, "error": f"Base directory not found: {base}"}

        rows: List[Dict[str, Any]] = []
        allowed = {lbl.lower() for lbl in (allowed_labels or [])}
        fel_root_path: Optional[Path] = None
        if local_fel_root:
            fel_root_path = Path(local_fel_root)
            if not fel_root_path.is_absolute():
                fel_root_path = (base / "analysis" / local_fel_root).resolve()
                if not fel_root_path.is_dir():
                    fel_root_path = Path(local_fel_root).resolve()
        for child in sorted(base.iterdir()):
            if not child.is_dir() or child.name in _BASE_AGENT_DIRS:
                continue
            if allowed and child.name.lower() not in allowed:
                continue
            if not (child / "analysis").is_dir():
                continue
            rows.append(
                collect_features_for_sim(child, local_fel_root=fel_root_path)
            )

        if not rows:
            return {
                "success": False,
                "error": (
                    f"No simulation subdirectories with analysis/ found under {base}"
                ),
            }

        metric_groups: Tuple[str, ...] = tuple(
            requested_metric_groups or ()
        )
        if feature_columns:
            feature_cols = tuple(dict.fromkeys(feature_columns))
        elif metric_groups:
            feature_cols = columns_for_metric_groups(metric_groups)
            if auto_discover:
                extra: List[str] = []
                for row in rows:
                    for c in discover_modular_feature_columns(row):
                        if c not in feature_cols and c not in extra:
                            extra.append(c)
                feature_cols = tuple(list(feature_cols) + extra)
        elif auto_discover:
            # Discover modular columns present; keep default groups as baseline
            # only when user asked for auto_discover without naming groups.
            base_cols = list(columns_for_metric_groups(DEFAULT_CLASSIFICATION_METRIC_GROUPS))
            extra = []
            for row in rows:
                for c in discover_modular_feature_columns(row):
                    if c not in base_cols and c not in extra:
                        extra.append(c)
            feature_cols = tuple(base_cols + extra)
            metric_groups = tuple(DEFAULT_CLASSIFICATION_METRIC_GROUPS)
        else:
            metric_groups = tuple(DEFAULT_CLASSIFICATION_METRIC_GROUPS)
            feature_cols = columns_for_metric_groups(metric_groups)

        for row in rows:
            for col in feature_cols:
                row.setdefault(col, None)
            row["n_features_present"] = _count_present(row, feature_cols)

        out_dir = Path(working_dir).resolve() if working_dir else (base / "analysis").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        def _out_path(name: str) -> Path:
            p = Path(name)
            return p if p.is_absolute() else out_dir / p.name

        fieldnames = ["label", "sim_directory", *feature_cols, "n_features_present"]
        raw_path = _out_path(output_file)
        with open(raw_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        z_path = None
        zrows: List[Dict[str, Any]] = []
        use_robust = bool(
            str(zscore_method or "").lower() == "robust"
            or "paper_ward4" in set(metric_groups or [])
            or (
                feature_columns
                and list(feature_columns) == list(PAPER_WARD4_FEATURE_COLUMNS)
            )
        )
        if include_zscore:
            zrows = _zscore_table(
                rows,
                feature_cols,
                method="robust" if use_robust else "zscore",
            )
            z_path = _out_path(zscore_output_file)
            z_fieldnames = ["label", "sim_directory", *feature_cols]
            with open(z_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=z_fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in zrows:
                    writer.writerow(row)
        else:
            z_fieldnames = ["label", "sim_directory", *feature_cols]

        manifest = {
            "n_simulations": len(rows),
            "labels": [r["label"] for r in rows],
            "requested_metric_groups": list(metric_groups),
            "feature_columns": list(feature_cols),
            "auto_discover": bool(auto_discover),
            "feature_definitions": _feature_definitions_rows(feature_cols),
            "raw_output": str(raw_path.resolve()),
            "zscore_output": str(z_path.resolve()) if z_path else None,
            "clustering_input_file": (
                str(z_path.resolve()) if z_path else str(raw_path.resolve())
            ),
            "normalization": (
                (
                    "robust IQR z-score (paper Ward-4 / fig4): winzorize at 1.5×IQR, "
                    "z = (x - median) / (IQR/1.349), clipped to ±3"
                )
                if use_robust
                else (
                    "z-score per column across all simulations in this table: "
                    "z = (x - mean) / std. Use zscore file for unsupervised clustering; "
                    "use raw file for physical interpretation."
                )
            ),
            "normalization_method": "robust" if use_robust else "zscore",
            "unsupervised_recommended": [
                "Load classification_features_zscore.csv (or ZScore_Features sheet in XLSX)",
                "Drop columns with many NaNs",
                "cluster_classification_features (hierarchical default, or method='kmeans')",
                "Inspect classification_clusters_pca.png, classification_dendrogram.png, "
                "and classification_phylo_tree.png",
            ],
        }
        manifest_path = _out_path(manifest_file)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        xlsx_path = None
        if xlsx_output_file:
            xlsx_path = _write_classification_xlsx(
                _out_path(xlsx_output_file),
                rows,
                zrows if zrows else rows,
                feature_cols,
                fieldnames,
                z_fieldnames,
                manifest,
            )
            if xlsx_path:
                manifest["xlsx_output"] = xlsx_path
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "success": True,
            "message": (
                f"Classification table: {len(rows)} simulations, "
                f"{len(feature_cols)} features (groups: {', '.join(metric_groups)}) "
                f"→ {raw_path.name}"
            ),
            "output_file": str(raw_path),
            "zscore_output_file": str(z_path) if z_path else None,
            "xlsx_output_file": xlsx_path,
            "manifest_file": str(manifest_path),
            "n_simulations": len(rows),
            "labels": [r["label"] for r in rows],
            "requested_metric_groups": list(metric_groups),
            "feature_columns": list(feature_cols),
        }
    except Exception as exc:
        logger.exception("collect_classification_features_table failed")
        return {"success": False, "error": str(exc)}
    finally:
        if original_dir:
            os.chdir(original_dir)
