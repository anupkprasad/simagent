"""
Trajectory metric registry — generic metadata for batched MDAnalysis execution.

Each entry describes how a calculation tool uses the trajectory so the batch
orchestrator can:
  - group metrics into RAW vs ALIGNED passes (separate Universe loads)
  - fuse STREAMING metrics into a single ``for ts in u.trajectory`` loop
  - run BATCH_ANALYSIS metrics on a shared Universe without reloading
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional


class TrajectoryPassKind(str, Enum):
    """Whether structural superposition is applied before the metric."""

    RAW = "raw"
    """Use coordinates as stored (diffusion, COM distances, compactness)."""

    ALIGNED = "aligned"
    """Superpose to a reference frame before computing (RMSF, pocket RMSF, PCA)."""

    INTERNAL = "internal"
    """Tool performs its own per-frame fit (e.g. MDAnalysis RMSD)."""


class TrajectoryExecutionMode(str, Enum):
    """How the metric consumes frames."""

    STREAMING = "streaming"
    """Per-frame accumulator — can share one trajectory loop with other STREAMING metrics."""

    BATCH_ANALYSIS = "batch_analysis"
    """MDAnalysis ``AnalysisBase.run()`` — shares Universe + alignment, own inner loop."""

    EXTERNAL = "external"
    """No MDAnalysis frame loop (GROMACS CLI, file-only plots)."""

    PLOT_ONLY = "plot_only"
    """Reads precomputed files only."""


@dataclass(frozen=True)
class TrajectoryMetricSpec:
    tool_name: str
    pass_kind: TrajectoryPassKind
    execution_mode: TrajectoryExecutionMode
    default_align_selection: str = "protein and name CA"
    streaming_group: str = "default"
    """Metrics with the same pass_kind + streaming_group can fuse loops."""

    @property
    def is_trajectory_metric(self) -> bool:
        return self.execution_mode in (
            TrajectoryExecutionMode.STREAMING,
            TrajectoryExecutionMode.BATCH_ANALYSIS,
        )

    @property
    def is_streaming(self) -> bool:
        return self.execution_mode == TrajectoryExecutionMode.STREAMING


# Canonical registry — extend when new trajectory tools are added.
METRIC_REGISTRY: Dict[str, TrajectoryMetricSpec] = {
    "calculate_rmsd": TrajectoryMetricSpec(
        "calculate_rmsd",
        TrajectoryPassKind.INTERNAL,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_rmsf": TrajectoryMetricSpec(
        "calculate_rmsf",
        TrajectoryPassKind.ALIGNED,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_radius_of_gyration": TrajectoryMetricSpec(
        "calculate_radius_of_gyration",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.STREAMING,
        streaming_group="per_frame",
    ),
    "analyze_secondary_structure": TrajectoryMetricSpec(
        "analyze_secondary_structure",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_dccm": TrajectoryMetricSpec(
        "calculate_dccm",
        TrajectoryPassKind.ALIGNED,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_com_distance": TrajectoryMetricSpec(
        "calculate_com_distance",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.STREAMING,
        streaming_group="per_frame",
    ),
    "calculate_ligand_pocket_distance": TrajectoryMetricSpec(
        "calculate_ligand_pocket_distance",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.STREAMING,
        streaming_group="per_frame",
    ),
    "calculate_trajectory_pca": TrajectoryMetricSpec(
        "calculate_trajectory_pca",
        TrajectoryPassKind.ALIGNED,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_protein_ligand_contacts": TrajectoryMetricSpec(
        "calculate_protein_ligand_contacts",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "analyze_ligand_residence": TrajectoryMetricSpec(
        "analyze_ligand_residence",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.STREAMING,
        streaming_group="per_frame",
    ),
    "calculate_pocket_rmsf": TrajectoryMetricSpec(
        "calculate_pocket_rmsf",
        TrajectoryPassKind.ALIGNED,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_ligand_rmsf": TrajectoryMetricSpec(
        "calculate_ligand_rmsf",
        TrajectoryPassKind.ALIGNED,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_pocket_sasa": TrajectoryMetricSpec(
        "calculate_pocket_sasa",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.EXTERNAL,
    ),
    "calculate_sasa": TrajectoryMetricSpec(
        "calculate_sasa",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.EXTERNAL,
    ),
    "identify_nearby_residues": TrajectoryMetricSpec(
        "identify_nearby_residues",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.EXTERNAL,
    ),
    "calculate_min_heavy_atom_distance": TrajectoryMetricSpec(
        "calculate_min_heavy_atom_distance",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.STREAMING,
        streaming_group="per_frame",
    ),
    "calculate_hbond_occupancy": TrajectoryMetricSpec(
        "calculate_hbond_occupancy",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_salt_bridge_distances": TrajectoryMetricSpec(
        "calculate_salt_bridge_distances",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "wrap_trajectory": TrajectoryMetricSpec(
        "wrap_trajectory",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.EXTERNAL,
    ),
    # Family consensus calculators — ALIGNED batch so large-N / multi-rep
    # campaigns share one Universe load with other aligned metrics.
    "calculate_consensus_rmsf_features": TrajectoryMetricSpec(
        "calculate_consensus_rmsf_features",
        TrajectoryPassKind.ALIGNED,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_consensus_dccm_features": TrajectoryMetricSpec(
        "calculate_consensus_dccm_features",
        TrajectoryPassKind.ALIGNED,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_consensus_torsions": TrajectoryMetricSpec(
        "calculate_consensus_torsions",
        TrajectoryPassKind.ALIGNED,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "calculate_consensus_pocket_metrics": TrajectoryMetricSpec(
        "calculate_consensus_pocket_metrics",
        TrajectoryPassKind.RAW,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
    "run_independent_dynamics_fel": TrajectoryMetricSpec(
        "run_independent_dynamics_fel",
        TrajectoryPassKind.ALIGNED,
        TrajectoryExecutionMode.BATCH_ANALYSIS,
    ),
}


def get_metric_spec(tool_name: str) -> Optional[TrajectoryMetricSpec]:
    return METRIC_REGISTRY.get(tool_name)


def is_batchable_trajectory_tool(tool_name: str) -> bool:
    spec = get_metric_spec(tool_name)
    return spec is not None and spec.is_trajectory_metric


def trajectory_pass_for_tool(tool_name: str) -> Optional[TrajectoryPassKind]:
    spec = get_metric_spec(tool_name)
    return spec.pass_kind if spec else None


def tools_sharing_pass(pass_kind: TrajectoryPassKind) -> FrozenSet[str]:
    return frozenset(
        name
        for name, spec in METRIC_REGISTRY.items()
        if spec.pass_kind == pass_kind and spec.is_trajectory_metric
    )


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def resolve_effective_pass_kind(
    tool_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[TrajectoryPassKind]:
    """
    Resolve the trajectory pass for a tool invocation.

    Some metrics override the registry default via tool params (e.g. Rg on
    aligned vs raw coordinates).
    """
    spec = get_metric_spec(tool_name)
    if spec is None:
        return None

    params = params or {}

    if tool_name == "calculate_radius_of_gyration":
        if _truthy(params.get("align_before_rg"), default=False):
            return TrajectoryPassKind.ALIGNED
        return TrajectoryPassKind.RAW

    if tool_name == "calculate_rmsf":
        if _truthy(params.get("align_trajectory"), default=True):
            return TrajectoryPassKind.ALIGNED
        return TrajectoryPassKind.RAW

    if tool_name == "calculate_rmsd":
        if _truthy(params.get("align_before_rmsd"), default=True):
            return TrajectoryPassKind.ALIGNED
        return TrajectoryPassKind.RAW

    if tool_name == "calculate_trajectory_pca":
        ref = int(params.get("reference_frame", 0) or 0)
        if ref != 0 or _truthy(params.get("align_trajectory"), default=True):
            return TrajectoryPassKind.ALIGNED
        return TrajectoryPassKind.RAW

    if tool_name in ("calculate_pocket_rmsf", "calculate_ligand_rmsf"):
        if _truthy(params.get("align_trajectory"), default=True):
            return TrajectoryPassKind.ALIGNED
        return TrajectoryPassKind.RAW

    return spec.pass_kind
