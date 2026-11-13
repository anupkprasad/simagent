"""
Analysis Agent Tools - Thin wrapper for modular analysis tools

Exposes stable, reusable @tool functions from src/analysis/ for:
- RMSD (Root Mean Square Deviation) calculation
- RMSF (Root Mean Square Fluctuation) calculation
- Radius of gyration analysis
- Energy analysis
- Trajectory metrics extraction
- Data visualization and plotting

This module follows the pattern of agentic/hpc/tools.py and agentic/simsetup/tools.py:
actual tool implementations are in src/analysis/* and imported here.

Additionally, dynamically loads programmer-generated tools from working_dir/programmer/
to make them available for analysis workflows.
"""
import os
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import modular tools from src/analysis/
from src.analysis.rmsd_calculator import calculate_rmsd
from src.analysis.rmsf_calculator import calculate_rmsf
from src.analysis.gyration_calculator import calculate_radius_of_gyration
from src.analysis.energy_analyzer import analyze_energy, extract_trajectory_metrics
from src.analysis.data_plotter import (
    plot_data, plot_multipanel, plot_combined_data, plot_3d,
    plot_md_data, plot_md_multipanel,  # backward-compat aliases
)
from src.analysis.summary_logger import initialize_summary_file, generate_summary_report
from src.analysis.dssp_analyzer import analyze_secondary_structure
from src.analysis.sasa_calculator import calculate_sasa, plot_sasa
from src.analysis.com_distance_calculator import calculate_com_distance, calculate_ligand_pocket_distance
from src.analysis.proximity_analyzer import identify_nearby_residues, calculate_min_heavy_atom_distance
from src.analysis.interface_analyzer import calculate_hbond_occupancy, calculate_salt_bridge_distances
from src.analysis.dccm_calculator import (
    calculate_dccm,
    plot_dccm_comparison,
    plot_dccm_difference,
)
from src.analysis.binding_site_analyzer import (
    calculate_protein_ligand_contacts,
    calculate_pocket_sasa,
    analyze_ligand_residence,
    calculate_pocket_rmsf,
    calculate_ligand_rmsf,
)
from src.analysis.classification_collector import collect_classification_features_table
from src.analysis.classification_clustering import (
    cluster_classification_features,
    plot_cluster_feature_trajectories,
    plot_cluster_rmsf_profiles,
    CLUSTER_TRAJECTORY_METRIC_GROUPS,
    CLUSTER_RMSF_PROFILE_GROUPS,
)
from src.analysis.pca_analyzer import (
    calculate_trajectory_pca,
    plot_pca_projection,
    calculate_free_energy_landscape,
    analyze_fel_landscape_features,
    export_fel_basin_structures,
    collect_fel_features_table,
    apply_pca_tool_defaults,
    PCA_TOOL_DEFAULTS,
)
from src.analysis.trajectory_wrapper import wrap_trajectory
from src.analysis.combined_analysis import (
    collect_metric_files,
    plot_combined_overlay,
    compute_comparison_table,
    run_combined_analysis,
    run_combined_dccm_analysis,
    run_combined_dccm_difference,
    plot_combined_rmsf_segment_bars,
    run_combined_rmsf_segment_analysis,
    run_combined_com_distance_analysis,
    pair_apo_holo_simulations,
    is_holo_simulation,
    plot_rmsf_apo_holo_comparison,
    run_combined_rmsf_apo_holo_analysis,
    run_combined_dccm_apo_holo_analysis,
    run_combined_rmsf_segment_apo_holo_analysis,
    run_combined_binding_rmsf_overlay,
)
from src.analysis.phylo_tree import (
    build_sequence_phylo_tree,
    build_structure_phylo_tree,
)
from src.analysis.consensus_alignment import (
    build_consensus_sequence_alignment,
    build_global_mapped_alignment,
    build_global_consensus_msa,
)
from src.analysis.consensus_pocket import (
    define_reference_consensus_pocket,
    define_pocket_mapped_residues,
    map_consensus_pocket_residues,
    map_pocket_mapped_residues,
    calculate_consensus_pocket_metrics,
    run_consensus_pocket_metrics_batch,
)
from src.analysis.msa_plotting import (
    plot_reference_msa_alignment,
    plot_global_mapped_alignment,
)
from src.analysis.reference_landscape import (
    fit_reference_pca_model,
    project_simulations_reference_pca,
    build_shared_reference_fel_landscapes,
    cluster_reference_fel_landscapes,
    run_reference_landscape_pipeline,
)
from src.analysis.msa_plotting import (
    plot_reference_msa_alignment,
    plot_global_mapped_alignment,
)
from src.analysis.ligand_rmsd import calculate_ligand_rmsd
from src.analysis.trajectory_qc import run_trajectory_qc
from src.analysis.md_basics import calculate_native_contacts, calculate_backbone_dihedrals
from src.analysis.replicate_aggregate import aggregate_replicate_metrics
from src.analysis.consensus_local_fel import run_consensus_local_fel_batch_tool
from src.analysis.consensus_torsions import (
    calculate_consensus_torsions,
    run_consensus_torsions_batch,
)
from src.analysis.family_dynamics import (
    run_independent_dynamics_fel,
    fit_dynamics_model,
    project_dynamics_model,
    run_shared_dynamics_fel_batch,
    compute_shared_pka_ref_dyn_features,
)
from src.analysis.consensus_structural_features import (
    calculate_consensus_rmsf_features,
    calculate_consensus_dccm_features,
)
from src.analysis.general_md_tools import (
    calculate_ligand_axis_angle,
    calculate_water_occupancy,
    cluster_trajectory_frames,
    calculate_hbond_lifetimes,
)

# Import dynamic tool loader for programmer-generated tools
from agentic.utils import get_dynamic_tool_loader
from agentic.analysis.tool_buckets import (
    COMBINED_TOOL_NAMES,
    SHARED_TOOL_NAMES,
    PER_SIM_TOOL_NAMES,
    names_for_scope,
)

# Export all tools
__all__ = [
    "calculate_rmsd",
    "calculate_rmsf",
    "calculate_radius_of_gyration",
    "calculate_sasa",
    "plot_sasa",
    "analyze_energy",
    "extract_trajectory_metrics",
    "analyze_secondary_structure",
    "plot_data",
    "plot_multipanel",
    "plot_combined_data",
    "plot_3d",
    "plot_md_data",
    "plot_md_multipanel",
    "calculate_com_distance",
    "calculate_ligand_pocket_distance",
    "identify_nearby_residues",
    "calculate_min_heavy_atom_distance",
    "calculate_hbond_occupancy",
    "calculate_salt_bridge_distances",
    "calculate_dccm",
    "plot_dccm_comparison",
    "plot_dccm_difference",
    "wrap_trajectory",
    "calculate_trajectory_pca",
    "plot_pca_projection",
    "calculate_free_energy_landscape",
    "analyze_fel_landscape_features",
    "export_fel_basin_structures",
    "calculate_protein_ligand_contacts",
    "calculate_pocket_sasa",
    "analyze_ligand_residence",
    "calculate_pocket_rmsf",
    "calculate_ligand_rmsf",
    "calculate_ligand_rmsd",
    "run_trajectory_qc",
    "calculate_native_contacts",
    "calculate_backbone_dihedrals",
    "calculate_consensus_torsions",
    "run_independent_dynamics_fel",
    "project_dynamics_model",
    "calculate_consensus_rmsf_features",
    "calculate_consensus_dccm_features",
    "calculate_ligand_axis_angle",
    "calculate_water_occupancy",
    "cluster_trajectory_frames",
    "calculate_hbond_lifetimes",
    # Combined (multi-sim) tools
    "collect_metric_files",
    "plot_combined_overlay",
    "compute_comparison_table",
    "run_combined_analysis",
    "run_combined_dccm_analysis",
    "run_combined_dccm_difference",
    "plot_combined_rmsf_segment_bars",
    "run_combined_rmsf_segment_analysis",
    "run_combined_com_distance_analysis",
    "pair_apo_holo_simulations",
    "is_holo_simulation",
    "plot_rmsf_apo_holo_comparison",
    "run_combined_rmsf_apo_holo_analysis",
    "run_combined_dccm_apo_holo_analysis",
    "run_combined_rmsf_segment_apo_holo_analysis",
    "build_sequence_phylo_tree",
    "build_structure_phylo_tree",
    "build_consensus_sequence_alignment",
    "build_global_mapped_alignment",
    "build_global_consensus_msa",
    "fit_reference_pca_model",
    "project_simulations_reference_pca",
    "build_shared_reference_fel_landscapes",
    "cluster_reference_fel_landscapes",
    "run_reference_landscape_pipeline",
    "define_reference_consensus_pocket",
    "define_pocket_mapped_residues",
    "map_consensus_pocket_residues",
    "map_pocket_mapped_residues",
    "calculate_consensus_pocket_metrics",
    "run_consensus_pocket_metrics_batch",
    "plot_reference_msa_alignment",
    "plot_global_mapped_alignment",
    "run_consensus_local_fel_batch",
    "run_consensus_torsions_batch",
    "fit_dynamics_model",
    "run_shared_dynamics_fel_batch",
    "compute_shared_pka_ref_dyn_features",
    "AnalysisToolExecutor",
    "get_analysis_tools",
    "get_tool_metadata",
    "COMBINED_ANALYSIS_TOOL_NAMES",
    "is_combined_analysis_tool",
    "initialize_summary_file",
    "generate_summary_report"
]

logger = logging.getLogger(__name__)

# Cross-simulation + shared/family tools — not for per-sim planner/analysis LLMs.
COMBINED_ANALYSIS_TOOL_NAMES = frozenset(COMBINED_TOOL_NAMES | SHARED_TOOL_NAMES)

_PER_SIM_ANALYSIS_TOOLS = [
    calculate_rmsd,
    calculate_rmsf,
    calculate_radius_of_gyration,
    calculate_sasa,
    plot_sasa,
    analyze_energy,
    extract_trajectory_metrics,
    analyze_secondary_structure,
    plot_md_data,
    plot_md_multipanel,
    plot_combined_data,
    calculate_com_distance,
    calculate_ligand_pocket_distance,
    identify_nearby_residues,
    calculate_min_heavy_atom_distance,
    calculate_hbond_occupancy,
    calculate_salt_bridge_distances,
    calculate_dccm,
    plot_dccm_comparison,
    plot_dccm_difference,
    wrap_trajectory,
    calculate_trajectory_pca,
    plot_pca_projection,
    calculate_free_energy_landscape,
    analyze_fel_landscape_features,
    export_fel_basin_structures,
    calculate_protein_ligand_contacts,
    calculate_pocket_sasa,
    analyze_ligand_residence,
    calculate_pocket_rmsf,
    calculate_ligand_rmsf,
    calculate_ligand_rmsd,
    run_trajectory_qc,
    calculate_native_contacts,
    calculate_backbone_dihedrals,
    calculate_consensus_torsions,
    run_independent_dynamics_fel,
    project_dynamics_model,
    calculate_consensus_rmsf_features,
    calculate_consensus_dccm_features,
    calculate_consensus_pocket_metrics,
    calculate_ligand_axis_angle,
    calculate_water_occupancy,
    cluster_trajectory_frames,
    calculate_hbond_lifetimes,
]

_COMBINED_ANALYSIS_TOOLS = [
    collect_metric_files,
    plot_combined_overlay,
    compute_comparison_table,
    run_combined_analysis,
    run_combined_dccm_analysis,
    run_combined_dccm_difference,
    plot_combined_rmsf_segment_bars,
    run_combined_rmsf_segment_analysis,
    run_combined_com_distance_analysis,
    plot_rmsf_apo_holo_comparison,
    run_combined_rmsf_apo_holo_analysis,
    run_combined_dccm_apo_holo_analysis,
    run_combined_rmsf_segment_apo_holo_analysis,
    run_combined_binding_rmsf_overlay,
    collect_fel_features_table,
    collect_classification_features_table,
    cluster_classification_features,
    plot_cluster_feature_trajectories,
    plot_cluster_rmsf_profiles,
    build_sequence_phylo_tree,
    build_structure_phylo_tree,
]

_SHARED_ANALYSIS_TOOLS = [
    build_consensus_sequence_alignment,
    build_global_mapped_alignment,
    build_global_consensus_msa,
    fit_reference_pca_model,
    project_simulations_reference_pca,
    build_shared_reference_fel_landscapes,
    cluster_reference_fel_landscapes,
    run_reference_landscape_pipeline,
    define_reference_consensus_pocket,
    define_pocket_mapped_residues,
    map_consensus_pocket_residues,
    map_pocket_mapped_residues,
    run_consensus_pocket_metrics_batch,
    plot_reference_msa_alignment,
    plot_global_mapped_alignment,
    run_consensus_local_fel_batch_tool,
    run_consensus_torsions_batch,
    fit_dynamics_model,
    run_shared_dynamics_fel_batch,
    compute_shared_pka_ref_dyn_features,
]


def is_combined_analysis_tool(tool_name: str) -> bool:
    """Return True if *tool_name* is a cross-simulation or shared/family tool."""
    return tool_name in COMBINED_ANALYSIS_TOOL_NAMES


def get_analysis_tools(
    include_combined: bool = False,
    include_shared: Optional[bool] = None,
) -> list:
    """
    Get analysis @tool functions for LLM binding and tool registry.

    Args:
        include_combined: When True, include cross-simulation combined-analysis tools.
        include_shared: Family/shared tools. Defaults to True when include_combined.

    Returns:
        List of StructuredTool objects ready for LLM use
    """
    if include_shared is None:
        include_shared = include_combined
    tools = list(_PER_SIM_ANALYSIS_TOOLS)
    if include_combined:
        tools.extend(_COMBINED_ANALYSIS_TOOLS)
    if include_shared:
        tools.extend(_SHARED_ANALYSIS_TOOLS)
    return tools

def get_tool_metadata(
    working_directory: Optional[str] = None,
    include_combined: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically extract metadata from all @tool functions.
    This provides tool information for the planner agent.
    
    Args:
        working_directory: Base working directory (e.g. 'work_di'). Used to locate programmer-generated tools.
        include_combined: When True, include cross-simulation combined-analysis tools.
    
    Returns:
        Dict mapping tool names to their metadata (description, args, etc.)
    """
    tools = get_analysis_tools(include_combined=include_combined)
    metadata = {}
    
    for tool in tools:
        # StructuredTool has .name, .description, .args_schema attributes
        tool_info = {
            "name": tool.name,
            "description": tool.description,
            "args": {},
        }
        
        # Extract argument schema if available
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema.schema()
            if 'properties' in schema:
                tool_info["args"] = {
                    arg_name: {
                        "type": arg_props.get("type", "string"),
                        "description": arg_props.get("description", ""),
                        "required": arg_name in schema.get("required", []),
                    }
                    for arg_name, arg_props in schema['properties'].items()
                }
        
        metadata[tool.name] = tool_info
    
    # Also include programmer-generated tools
    try:
        from agentic.utils import get_dynamic_tool_loader
        
        # MUST use refresh=True to get newly created tools
        programmer_dir = str(Path(working_directory) / "programmer") if working_directory else "working_dir/programmer"
        tool_loader = get_dynamic_tool_loader(programmer_dir=programmer_dir, refresh=True)
        programmer_tools_metadata = tool_loader.get_tool_metadata_list()
        
        for prog_tool_meta in programmer_tools_metadata:
            tool_name = prog_tool_meta.get("name", "unknown")
            
            # Convert programmer tool metadata to analysis tool metadata format
            tool_info = {
                "name": tool_name,
                "description": prog_tool_meta.get("description", "Programmer-generated tool"),
                "args": {}
            }
            
            # Convert parameters format
            params = prog_tool_meta.get("parameters", {})
            for param_name, param_details in params.items():
                tool_info["args"][param_name] = {
                    "type": param_details.get("type", "string"),
                    "description": param_details.get("description", ""),
                    "required": param_details.get("required", False),
                }
            
            metadata[tool_name] = tool_info
            logger.info(f"Included programmer-generated tool in metadata: {tool_name}")
    
    except Exception as e:
        logger.warning(f"Could not load programmer-generated tools in get_tool_metadata: {e}")
    
    return metadata


class AnalysisToolExecutor:
    """
    Executor class for analysis agent tools.
    
    Provides a unified interface for executing MD analysis tools with proper
    configuration management, error handling, and result aggregation.
    
    Similar to HPCToolExecutor and SimulationSetupToolExecutor.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize analysis tool executor.
        
        Args:
            config: Analysis configuration (output paths, default selections, etc.)
        """
        self.config = config or {}
        include_combined = self.config.get("include_combined_tools", False)
        
        # Core analysis tools (per-simulation only unless include_combined_tools=True)
        self.tools = {
            "calculate_rmsd": calculate_rmsd,
            "calculate_rmsf": calculate_rmsf,
            "calculate_radius_of_gyration": calculate_radius_of_gyration,
            "calculate_sasa": calculate_sasa,
            "plot_sasa": plot_sasa,
            "analyze_energy": analyze_energy,
            "extract_trajectory_metrics": extract_trajectory_metrics,
            "analyze_secondary_structure": analyze_secondary_structure,
            "plot_data": plot_data,
            "plot_multipanel": plot_multipanel,
            "plot_combined_data": plot_combined_data,
            "plot_3d": plot_3d,
            # backward-compat aliases
            "plot_md_data": plot_md_data,
            "plot_md_multipanel": plot_md_multipanel,
            "calculate_com_distance": calculate_com_distance,
            "calculate_ligand_pocket_distance": calculate_ligand_pocket_distance,
            "identify_nearby_residues": identify_nearby_residues,
            "calculate_min_heavy_atom_distance": calculate_min_heavy_atom_distance,
            "calculate_hbond_occupancy": calculate_hbond_occupancy,
            "calculate_salt_bridge_distances": calculate_salt_bridge_distances,
            "calculate_dccm": calculate_dccm,
            "plot_dccm_comparison": plot_dccm_comparison,
            "plot_dccm_difference": plot_dccm_difference,
            "wrap_trajectory": wrap_trajectory,
            "calculate_trajectory_pca": calculate_trajectory_pca,
            "plot_pca_projection": plot_pca_projection,
            "calculate_free_energy_landscape": calculate_free_energy_landscape,
            "analyze_fel_landscape_features": analyze_fel_landscape_features,
            "export_fel_basin_structures": export_fel_basin_structures,
            "calculate_protein_ligand_contacts": calculate_protein_ligand_contacts,
            "calculate_pocket_sasa": calculate_pocket_sasa,
            "analyze_ligand_residence": analyze_ligand_residence,
            "calculate_pocket_rmsf": calculate_pocket_rmsf,
            "calculate_ligand_rmsf": calculate_ligand_rmsf,
            "calculate_ligand_rmsd": calculate_ligand_rmsd,
            "run_trajectory_qc": run_trajectory_qc,
            "calculate_native_contacts": calculate_native_contacts,
            "calculate_backbone_dihedrals": calculate_backbone_dihedrals,
            "aggregate_replicate_metrics": aggregate_replicate_metrics,
            # Shared analysis protocol / general MD features (per-sim)
            "calculate_consensus_torsions": calculate_consensus_torsions,
            "calculate_consensus_rmsf_features": calculate_consensus_rmsf_features,
            "calculate_consensus_dccm_features": calculate_consensus_dccm_features,
            "calculate_consensus_pocket_metrics": calculate_consensus_pocket_metrics,
            "run_independent_dynamics_fel": run_independent_dynamics_fel,
            "calculate_ligand_axis_angle": calculate_ligand_axis_angle,
            "calculate_water_occupancy": calculate_water_occupancy,
            "cluster_trajectory_frames": cluster_trajectory_frames,
            "calculate_hbond_lifetimes": calculate_hbond_lifetimes,
        }
        if include_combined:
            self.tools.update({
                "collect_metric_files": collect_metric_files,
                "plot_combined_overlay": plot_combined_overlay,
                "compute_comparison_table": compute_comparison_table,
                "run_combined_analysis": run_combined_analysis,
                "run_combined_dccm_analysis": run_combined_dccm_analysis,
                "run_combined_dccm_difference": run_combined_dccm_difference,
                "plot_combined_rmsf_segment_bars": plot_combined_rmsf_segment_bars,
                "run_combined_rmsf_segment_analysis": run_combined_rmsf_segment_analysis,
                "run_combined_com_distance_analysis": run_combined_com_distance_analysis,
                "plot_rmsf_apo_holo_comparison": plot_rmsf_apo_holo_comparison,
                "run_combined_rmsf_apo_holo_analysis": run_combined_rmsf_apo_holo_analysis,
                "run_combined_dccm_apo_holo_analysis": run_combined_dccm_apo_holo_analysis,
                "run_combined_rmsf_segment_apo_holo_analysis": run_combined_rmsf_segment_apo_holo_analysis,
                "run_combined_binding_rmsf_overlay": run_combined_binding_rmsf_overlay,
                "collect_fel_features_table": collect_fel_features_table,
                "collect_classification_features_table": collect_classification_features_table,
                "cluster_classification_features": cluster_classification_features,
                "plot_cluster_feature_trajectories": plot_cluster_feature_trajectories,
                "plot_cluster_rmsf_profiles": plot_cluster_rmsf_profiles,
                "build_sequence_phylo_tree": build_sequence_phylo_tree,
                "build_structure_phylo_tree": build_structure_phylo_tree,
                "build_consensus_sequence_alignment": build_consensus_sequence_alignment,
                "build_global_mapped_alignment": build_global_mapped_alignment,
                "build_global_consensus_msa": build_global_consensus_msa,
                "fit_reference_pca_model": fit_reference_pca_model,
                "project_simulations_reference_pca": project_simulations_reference_pca,
                "build_shared_reference_fel_landscapes": build_shared_reference_fel_landscapes,
                "cluster_reference_fel_landscapes": cluster_reference_fel_landscapes,
                "run_reference_landscape_pipeline": run_reference_landscape_pipeline,
                "define_reference_consensus_pocket": define_reference_consensus_pocket,
                "define_pocket_mapped_residues": define_pocket_mapped_residues,
                "map_consensus_pocket_residues": map_consensus_pocket_residues,
                "map_pocket_mapped_residues": map_pocket_mapped_residues,
                "run_consensus_pocket_metrics_batch": run_consensus_pocket_metrics_batch,
                "plot_reference_msa_alignment": plot_reference_msa_alignment,
                "plot_global_mapped_alignment": plot_global_mapped_alignment,
                "run_consensus_local_fel_batch": run_consensus_local_fel_batch_tool,
                "fit_dynamics_model": fit_dynamics_model,
                "run_shared_dynamics_fel_batch": run_shared_dynamics_fel_batch,
                "compute_shared_pka_ref_dyn_features": compute_shared_pka_ref_dyn_features,
            })
        
        # Record built-in tool names BEFORE loading programmer tools
        # so _auto_log_summary can skip tools that already self-log
        self._builtin_tool_names = set(self.tools.keys())
        
        # Setup working directory BEFORE loading programmer tools
        self.working_dir = self.config.get("working_directory", "./working_dir/analysis")
        os.makedirs(self.working_dir, exist_ok=True)
        from src.analysis.replicate_paths import normalize_rep_num

        self.rep_num = normalize_rep_num(self.config.get("rep_num", 1))
        self.sim_root = self.config.get("sim_root") or str(Path(self.working_dir).parent)
        self.include_combined_tools = bool(include_combined)
        
        # Load programmer-generated tools dynamically (needs working_dir to be set)
        self._load_programmer_tools()
        
        # Initialize analysis summary file
        try:
            summary_path = initialize_summary_file(self.working_dir)
            logger.info(f"Analysis summary file initialized: {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize summary file: {e}")
        
        logger.info(
            "AnalysisToolExecutor initialized with %d tools (working_dir: %s, rep_num=%d)",
            len(self.tools),
            self.working_dir,
            self.rep_num,
        )
    
    def _wrap_tool_for_working_dir(self, tool_func, tool_name: str):
        """
        Wrap a programmer-generated tool to execute in the agent's working directory.
        This ensures all file outputs go to the correct location (e.g., working_dir/analysis).
        
        Args:
            tool_func: The tool function to wrap
            tool_name: Name of the tool (for logging)
            
        Returns:
            Wrapped function that executes in the agent's working directory
        """
        from functools import wraps
        
        # Get the actual function if it's a StructuredTool
        actual_func = tool_func.func if hasattr(tool_func, 'func') else tool_func
        
        @wraps(actual_func)
        def wrapped_tool(**kwargs):
            """
            Execute tool in the agent's working directory context.
            Changes to working_dir before execution and restores original directory after.
            """
            import os
            original_dir = os.getcwd()
            try:
                # Change to agent's working directory
                os.chdir(self.working_dir)
                logger.debug(f"Executing {tool_name} in directory: {self.working_dir}")
                
                # Execute the tool
                if hasattr(tool_func, 'func'):
                    result = tool_func.func(**kwargs)
                elif hasattr(tool_func, 'invoke'):
                    result = tool_func.invoke(kwargs)
                else:
                    result = tool_func(**kwargs)
                
                return result
            finally:
                # Always restore original directory
                os.chdir(original_dir)
        
        # Preserve StructuredTool attributes if needed
        if hasattr(tool_func, 'name'):
            wrapped_tool.name = tool_func.name
        if hasattr(tool_func, 'description'):
            wrapped_tool.description = tool_func.description
        if hasattr(tool_func, 'args_schema'):
            wrapped_tool.args_schema = tool_func.args_schema
            
        return wrapped_tool
    
    def _load_programmer_tools(self):
        """Load dynamically generated tools from programmer agent."""
        try:
            # Derive programmer dir from working_dir (e.g. work_di/analysis -> work_di/programmer)
            programmer_dir = str(Path(self.working_dir).parent / "programmer") if self.working_dir else None
            tool_loader = get_dynamic_tool_loader(programmer_dir=programmer_dir, refresh=True) if programmer_dir else get_dynamic_tool_loader(refresh=True)
            programmer_tools = tool_loader.get_tools_for_agent("analysis")
            
            if programmer_tools:
                logger.info(f"Loading {len(programmer_tools)} programmer-generated tools for analysis agent")
                for tool_name, tool_func in programmer_tools.items():
                    # Wrap the tool to execute in the agent's working directory
                    wrapped_tool = self._wrap_tool_for_working_dir(tool_func, tool_name)
                    self.tools[tool_name] = wrapped_tool
                    logger.info(f"  Registered programmer tool: {tool_name} (wrapped for {self.working_dir})")
            else:
                logger.debug("No programmer-generated tools found")
                
        except Exception as e:
            logger.warning(f"Failed to load programmer tools: {e}")
    
    def reload_programmer_tools(self):
        """Reload programmer-generated tools (call after programmer creates new tools)."""
        try:
            programmer_dir = str(Path(self.working_dir).parent / "programmer") if self.working_dir else None
            tool_loader = get_dynamic_tool_loader(programmer_dir=programmer_dir, refresh=True) if programmer_dir else get_dynamic_tool_loader(refresh=True)
            programmer_tools = tool_loader.get_tools_for_agent("analysis")
            
            # Add new tools
            for tool_name, tool_func in programmer_tools.items():
                if tool_name not in self.tools:
                    logger.info(f"Adding new programmer tool: {tool_name}")
                self.tools[tool_name] = tool_func
            
            logger.info(f"Reloaded programmer tools. Total tools: {len(self.tools)}")
            
        except Exception as e:
            logger.error(f"Failed to reload programmer tools: {e}")
    
    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an analysis tool with given parameters.

        When ``rep_num > 1`` (per-sim mode), runs the same tool across
        ``analysis/repXX`` with matching ``hpc/repXX`` inputs, then aggregates
        numeric metrics into ``analysis/avg/``.
        """
        if self._should_fanout_replicates(tool_name):
            return self._execute_across_replicates(tool_name, kwargs)
        kwargs = self._ensure_family_tool_kwargs(tool_name, kwargs)
        return self._execute_once(tool_name, **kwargs)

    def _ensure_family_tool_kwargs(
        self, tool_name: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fill alignment_json / label / pocket_map for modular family tools.

        Parallel workers sometimes lose analysis-agent injection; resolve from
        the campaign ``analysis/`` directory next to this simulation root.
        """
        family = {
            "calculate_consensus_torsions",
            "calculate_consensus_rmsf_features",
            "calculate_consensus_dccm_features",
            "run_independent_dynamics_fel",
            "calculate_ligand_pocket_distance",
            "calculate_consensus_pocket_metrics",
        }
        if tool_name not in family:
            return kwargs
        out = dict(kwargs or {})
        if not out.get("label"):
            out["label"] = Path(self.sim_root).name
        if not out.get("sim_directory"):
            out["sim_directory"] = str(self.sim_root)
        if not out.get("sim_dir"):
            out["sim_dir"] = str(self.sim_root)
        if tool_name in {
            "calculate_ligand_pocket_distance",
            "calculate_consensus_pocket_metrics",
        } and not out.get("pocket_map_json"):
            try:
                from src.analysis.inventory import discover_mapped_path

                found = discover_mapped_path(
                    Path(self.sim_root),
                    ("pocket_mapped.json", "pocket_map.json"),
                )
                if found:
                    out["pocket_map_json"] = found
            except Exception:
                found = ""
            if not out.get("pocket_map_json"):
                for name in ("pocket_mapped.json", "pocket_map.json"):
                    for root in (
                        Path(self.sim_root).parent / "analysis",
                        Path(self.sim_root).parent / "cross_sim",
                    ):
                        cand = root / name
                        if cand.is_file():
                            out["pocket_map_json"] = str(cand.resolve())
                            break
                    if out.get("pocket_map_json"):
                        break
        if not out.get("alignment_json"):
            for cand in (
                Path(self.sim_root).parent / "analysis" / "global_consensus_msa.json",
                Path(self.sim_root).parent / "cross_sim" / "global_consensus_msa.json",
                Path(self.sim_root).parent / "analysis" / "global_mapped.json",
                Path(self.sim_root).parent / "cross_sim" / "global_mapped.json",
                Path(self.sim_root).parent / "analysis" / "reference_msa_alignment.json",
                Path(self.working_dir).resolve().parents[1] / "analysis" / "global_consensus_msa.json",
                Path(self.working_dir).resolve().parents[1] / "analysis" / "global_mapped.json",
                Path(self.working_dir).resolve().parents[1] / "analysis" / "reference_msa_alignment.json",
            ):
                if cand.is_file():
                    out["alignment_json"] = str(cand.resolve())
                    break
        if not out.get("pocket_map_csv"):
            for name in (
                "pocket_mapped.csv",
                "pocket_mapped.json",
                "reference_pocket_residue_map.csv",
                "reference_pocket_resid_map.csv",
                "global_consensus_msa.csv",
                "global_mapped.csv",
                "reference_msa_residue_map.csv",
            ):
                cand = Path(self.sim_root).parent / "analysis" / name
                if cand.is_file():
                    out["pocket_map_csv"] = str(cand.resolve())
                    break
        return out

    def _should_fanout_replicates(self, tool_name: str) -> bool:
        if self.include_combined_tools:
            return False
        if self.rep_num <= 1:
            return False
        if tool_name in {
            "aggregate_replicate_metrics",
            "wrap_trajectory",
            "collect_metric_files",
            "plot_combined_overlay",
            "compute_comparison_table",
            "run_combined_analysis",
        }:
            return False
        return True

    def _execute_across_replicates(self, tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        from src.analysis.replicate_paths import (
            build_rep_plan,
            resolve_production_topology,
            resolve_production_trajectory,
        )

        plan = build_rep_plan(
            Path(self.sim_root).name,
            self.sim_root,
            self.rep_num,
            base_seed=int(self.config.get("replicate_base_seed") or 12345),
        )
        per_rep: Dict[str, Any] = {}
        saved_wd = self.working_dir
        all_ok = True
        for slot in plan:
            rep_id = slot["rep_id"]
            analysis_dir = slot["analysis_dir"]
            hpc_dir = Path(slot["hpc_dir"])
            Path(analysis_dir).mkdir(parents=True, exist_ok=True)
            self.working_dir = analysis_dir
            call_kw = dict(kwargs)
            traj = resolve_production_trajectory(hpc_dir)
            topo = resolve_production_topology(hpc_dir)
            # Remap absolute paths that point at *this sim's* flat analysis/ or
            # hpc/ into the per-rep slot. Do NOT remap campaign-level shared
            # artifacts (e.g. run_01/analysis/reference_msa_alignment.json) —
            # naive "/analysis/" → "/analysis/rep01/" breaks family tools.
            sim_root_resolved = Path(self.sim_root).resolve()
            shared_input_keys = {
                "alignment_json",
                "consensus_json",
                "pocket_map_csv",
                "pocket_definition_json",
                "definition_json",
                "residue_map_csv",
                "reference_msa_alignment",
                "global_mapped",
                "global_consensus_msa",
                "global_msa",
                "pocket_mapped",
            }

            def _remap_path(val: str, *, key: str = "") -> str:
                if key in shared_input_keys:
                    return val
                try:
                    p = Path(val)
                    if p.is_absolute():
                        try:
                            p.resolve().relative_to(sim_root_resolved)
                        except ValueError:
                            # Outside this simulation root — shared campaign path
                            return val
                except (OSError, RuntimeError, ValueError):
                    pass
                out = val
                if "/hpc/" in out and f"/hpc/{rep_id}/" not in out:
                    out = out.replace("/hpc/", f"/hpc/{rep_id}/")
                if "/analysis/" in out and f"/analysis/{rep_id}/" not in out:
                    if "/analysis/avg" not in out:
                        candidate = out.replace("/analysis/", f"/analysis/{rep_id}/")
                        # Keep original when remap would break an existing shared file
                        if Path(out).is_file() and not Path(candidate).is_file():
                            # Only preserve if original is outside sim analysis
                            # or is a known shared basename under parent analysis/
                            try:
                                Path(out).resolve().relative_to(sim_root_resolved / "analysis")
                            except ValueError:
                                return out
                        out = candidate
                return out

            for key, val in list(call_kw.items()):
                if isinstance(val, str):
                    call_kw[key] = _remap_path(val, key=key)
                elif isinstance(val, list):
                    call_kw[key] = [
                        _remap_path(v, key=key) if isinstance(v, str) else v for v in val
                    ]
            # Always bind this slot's production files. Relative mdWrap.xtc plus
            # a valid md.tpr used to leave both replicas on hpc/rep01.
            if traj is not None:
                abs_traj = str(Path(traj).resolve())
                for tkey in ("trajectory_file", "trajectory", "traj_file"):
                    call_kw[tkey] = abs_traj
            if topo is not None and topo.is_file():
                abs_topo = str(Path(topo).resolve())
                for tkey in ("topology_file", "topology", "structure_file"):
                    prev = call_kw.get(tkey)
                    call_kw[tkey] = abs_topo
                    if prev and str(prev) != abs_topo:
                        logger.info(
                            "Replicate fan-out: override %s %s → %s (%s)",
                            tkey,
                            prev,
                            abs_topo,
                            rep_id,
                        )
            call_kw["hpc_dir"] = str(hpc_dir)
            logger.info(
                "Replicate fan-out: %s → %s traj=%s",
                tool_name,
                rep_id,
                call_kw.get("trajectory_file"),
            )
            call_kw = self._ensure_family_tool_kwargs(tool_name, call_kw)
            result = self._execute_once(tool_name, **call_kw)
            per_rep[rep_id] = result
            try:
                from src.analysis.inventory import write_rep_inventory

                write_rep_inventory(
                    analysis_dir=analysis_dir,
                    hpc_dir=hpc_dir,
                    sim_root=self.sim_root,
                    label=Path(self.sim_root).name,
                    topology=str(call_kw.get("topology_file") or ""),
                    trajectory=str(call_kw.get("trajectory_file") or ""),
                    extra={"outputs": {"last_tool": tool_name}},
                )
            except Exception as inv_exc:
                logger.warning("inventory.json write failed for %s: %s", rep_id, inv_exc)
            if not result.get("success"):
                all_ok = False
                logger.error(
                    "Replicate fan-out: %s failed on %s: %s",
                    tool_name,
                    rep_id,
                    result.get("error") or result,
                )
        self.working_dir = saved_wd

        agg = None
        if all_ok:
            try:
                # Prefer filenames referenced by this tool call so avg/ is
                # populated immediately after each successful fan-out.
                hint_names: list[str] = []
                for key in (
                    "output_file",
                    "data_file",
                    "csv_file",
                    "results_file",
                    "matrix_file",
                    "rmsd_file",
                    "rmsf_file",
                ):
                    val = kwargs.get(key)
                    if isinstance(val, str) and val.strip():
                        hint_names.append(Path(val).name)
                    elif isinstance(val, list):
                        for v in val:
                            if isinstance(v, str) and v.strip():
                                hint_names.append(Path(v).name)
                for key in ("data_files",):
                    val = kwargs.get(key)
                    if isinstance(val, list):
                        for v in val:
                            if isinstance(v, str) and v.strip():
                                hint_names.append(Path(v).name)
                # Map calculate_* tools to conventional basenames
                tool_defaults = {
                    "calculate_rmsd": ["rmsd.dat"],
                    "calculate_rmsf": ["rmsf.dat"],
                    "calculate_gyration": ["gyration.dat", "rg.dat", "gyrate.dat"],
                    "calculate_radius_of_gyration": ["gyration.dat", "rg.dat"],
                    "calculate_sasa": ["sasa.dat"],
                    "calculate_dccm": ["dccm.dat", "dccm.csv", "dccm_matrix.dat"],
                    "calculate_com_distance": ["com_distance.dat", "com_distance.csv"],
                    "calculate_ligand_pocket_distance": [
                        "ligand_pocket_distance.csv",
                        "ligand_pocket_distance.dat",
                    ],
                    "calculate_consensus_pocket_metrics": [
                        "ligand_pocket_distance.csv",
                        "reference_pocket_ligand_distance.csv",
                        "pocket_axis_angle.csv",
                        "reference_pocket_ligand_orientation.csv",
                        "reference_pocket_metrics.json",
                    ],
                    "calculate_pocket_rmsf": ["pocket_rmsf.dat"],
                    "calculate_ligand_rmsf": ["ligand_rmsf.dat"],
                    "calculate_ligand_rmsd": ["ligand_rmsd.dat"],
                }
                for name in tool_defaults.get(tool_name, []):
                    if name not in hint_names:
                        hint_names.append(name)
                agg = aggregate_replicate_metrics.func(
                    sim_dir=self.sim_root,
                    rep_num=self.rep_num,
                    metric_filenames=hint_names or None,
                )
            except Exception as exc:
                logger.warning("aggregate_replicate_metrics after %s failed: %s", tool_name, exc)
                agg = {"success": False, "error": str(exc)}

        collapse = None
        if all_ok and self.rep_num > 1:
            try:
                from src.analysis.replica_collapse import replica_science_collapsed

                collapse = replica_science_collapsed(self.sim_root, self.rep_num)
                if collapse.get("collapsed"):
                    logger.error(
                        "Replicate fan-out: science products identical across "
                        "distinct trajectories: %s",
                        collapse.get("identical_products"),
                    )
                    all_ok = False
            except Exception as exc:
                logger.debug("replica collapse check skipped: %s", exc)

        from agentic.campaign.contracts import is_noncritical_analysis_tool

        if is_noncritical_analysis_tool(tool_name):
            any_ok = any(bool((r or {}).get("success")) for r in per_rep.values())
            return {
                "success": True,
                "rep_num": self.rep_num,
                "per_rep": per_rep,
                "aggregate": agg,
                "warning": None if all_ok else "one or more replicate plots failed",
                "error": None,
                "partial": not all_ok,
                "any_ok": any_ok,
            }
        err = None
        if not all_ok:
            if collapse and collapse.get("collapsed"):
                err = (
                    "replica science collapsed: identical products on distinct "
                    f"trajectories ({', '.join(collapse.get('identical_products') or [])})"
                )
            else:
                err = "one or more replicates failed"
        out = {
            "success": all_ok,
            "rep_num": self.rep_num,
            "per_rep": per_rep,
            "aggregate": agg,
            "error": err,
        }
        if collapse:
            out["replica_collapse"] = collapse
        return out

    def _execute_once(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an analysis tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific parameters
            
        Returns:
            Dict with execution result
        """
        # Older campaign pins used this name; the registered tool is the longer one.
        if tool_name == "calculate_gyration":
            tool_name = "calculate_radius_of_gyration"
        if tool_name not in self.tools:
            try:
                from agentic.utils.sandbox_files import execute_sandbox_tool

                sand = execute_sandbox_tool(
                    tool_name,
                    kwargs,
                    roots=[
                        Path(self.sim_root).parent if getattr(self, "sim_root", None) else Path(self.working_dir),
                        Path(getattr(self, "sim_root", None) or self.working_dir),
                    ],
                    sim_root=str(getattr(self, "sim_root", "") or self.working_dir),
                )
                if sand is not None:
                    return sand
            except Exception:
                pass
            logger.error(
                "Unknown analysis tool %s (available=%s)",
                tool_name,
                sorted(self.tools.keys()),
            )
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
            }
        
        # Normalise common LLM parameter-name mistakes before calling the tool
        _TOOL_SPECIFIC_ALIASES: Dict[str, Dict[str, str]] = {
            "plot_pca_projection": {
                "pca_file": "pca_projections_file",
                "input_file": "pca_projections_file",
            },
            "calculate_free_energy_landscape": {
                "pca_file": "pca_projections_file",
                "input_file": "pca_projections_file",
            },
            "analyze_fel_landscape_features": {
                "pca_file": "pca_projections_file",
            },
            "calculate_trajectory_pca": {
                "output_file": "projections_file",
            },
        }
        _PARAM_ALIASES = {
            "title": "titles",
            "ylabel": "ylabels",  # multipanel expects plural
            "xlabel": "xlabels",  # multipanel expects plural
            "plot_type": "plot_types",  # multipanel expects plural
            "color": "colors",
            "data_file": "data_files",   # LLM often sends singular
            "csv_file": "data_files",    # LLM may use csv_file instead
            "input_file": "data_files",  # another common LLM alias
        }
        tool_func_raw = self.tools[tool_name]
        actual_fn = tool_func_raw.func if hasattr(tool_func_raw, 'func') else tool_func_raw
        try:
            valid_params = set(inspect.signature(actual_fn).parameters.keys())
        except Exception:
            valid_params = None

        # Only alias label→labels when the tool does NOT accept singular ``label``
        # (family consensus tools use label=sim id; multipanel uses labels=list).
        if valid_params is not None and "label" not in valid_params and "labels" in valid_params:
            _PARAM_ALIASES = {**_PARAM_ALIASES, "label": "labels"}
        elif valid_params is None:
            _PARAM_ALIASES = {**_PARAM_ALIASES, "label": "labels"}

        if valid_params is not None:
            aliases_applied = {}
            for alias, canonical in (_TOOL_SPECIFIC_ALIASES.get(tool_name) or {}).items():
                if alias in kwargs and alias not in valid_params and canonical in valid_params:
                    kwargs[canonical] = kwargs.pop(alias)
                    aliases_applied[alias] = canonical
            for alias, canonical in _PARAM_ALIASES.items():
                if alias in kwargs and alias not in valid_params and canonical in valid_params:
                    val = kwargs.pop(alias)
                    # Wrap scalar in list for list-typed parameters
                    if canonical not in kwargs:
                        kwargs[canonical] = [val] if isinstance(val, str) else val
                    aliases_applied[alias] = canonical
            try:
                from src.analysis.proximity_analyzer import apply_selection_from_files

                kwargs = apply_selection_from_files(kwargs, working_dir=self.working_dir)
            except Exception as exc:
                logger.warning("selection_from_file resolution skipped: %s", exc)

            # Combined-analysis LLM plans often pass labels/working_dir instead of
            # sim_dirs for collect_metric_files / compute_comparison_table.
            if tool_name in {"collect_metric_files", "compute_comparison_table"}:
                sim_dirs = kwargs.get("sim_dirs")
                if not sim_dirs:
                    labels = kwargs.get("labels")
                    wd_hint = kwargs.get("working_dir") or self.working_dir
                    if labels and wd_hint:
                        base = Path(str(wd_hint))
                        if base.name == "analysis":
                            base = base.parent
                        kwargs["sim_dirs"] = [str(base / str(lab)) for lab in labels]
                        logger.info(
                            "Synthesized sim_dirs for %s from labels under %s",
                            tool_name,
                            base,
                        )

            # compute_comparison_table needs data_files; LLM often only passes
            # labels + metric / output_csv. Collect per-sim metric files.
            if tool_name == "compute_comparison_table":
                data_files = kwargs.get("data_files")
                labels = kwargs.get("labels") or []
                sim_dirs = kwargs.get("sim_dirs") or []
                if (not data_files) and labels and sim_dirs:
                    metric_hint = (
                        kwargs.get("metric")
                        or kwargs.get("metric_filename")
                        or kwargs.get("metric_name")
                        or ""
                    )
                    if not metric_hint:
                        out_csv = str(kwargs.get("output_csv") or "")
                        # rmsd_stats.csv → rmsd
                        stem = Path(out_csv).stem.lower()
                        if stem.endswith("_stats"):
                            metric_hint = stem[: -len("_stats")]
                        elif stem:
                            metric_hint = stem
                    if metric_hint:
                        try:
                            from src.analysis.combined_analysis import collect_metric_files

                            collected = collect_metric_files.func(
                                sim_dirs=[str(d) for d in sim_dirs],
                                metric_filename=str(metric_hint),
                            )
                            found = collected.get("found") or []
                            # Align data_files to labels order when possible
                            by_dir = {
                                str(Path(item.get("sim_dir", "")).resolve()): item.get(
                                    "file"
                                )
                                for item in found
                                if item.get("file")
                            }
                            ordered: list = []
                            for sd, lab in zip(sim_dirs, labels):
                                hit = by_dir.get(str(Path(str(sd)).resolve()))
                                if not hit:
                                    # Fallback: match by label as directory name
                                    for k, v in by_dir.items():
                                        if Path(k).name == str(lab):
                                            hit = v
                                            break
                                if hit:
                                    ordered.append(hit)
                            if len(ordered) == len(labels):
                                kwargs["data_files"] = ordered
                                logger.info(
                                    "Synthesized data_files for compute_comparison_table "
                                    "from metric=%s (%d files)",
                                    metric_hint,
                                    len(ordered),
                                )
                            elif found:
                                # Partial: use found order + trim labels
                                kwargs["data_files"] = [
                                    item["file"] for item in found if item.get("file")
                                ]
                                kwargs["labels"] = [
                                    Path(item.get("sim_dir", "")).name
                                    for item in found
                                    if item.get("file")
                                ]
                                logger.info(
                                    "Synthesized partial data_files for "
                                    "compute_comparison_table (%d/%d)",
                                    len(kwargs["data_files"]),
                                    len(labels),
                                )
                        except Exception as synth_exc:
                            logger.warning(
                                "data_files synthesis for compute_comparison_table "
                                "failed: %s",
                                synth_exc,
                            )

            # Prefer analysis/avg/<metric> when LLM hardcodes flat analysis/<metric>
            # paths that do not exist after multi-rep aggregation.
            def _prefer_avg_path(val: str) -> str:
                p = Path(val)
                if p.is_file():
                    return val
                if p.parent.name == "analysis":
                    alt = p.parent / "avg" / p.name
                    if alt.is_file():
                        return str(alt)
                return val

            for key in ("data_files", "files", "metric_files"):
                raw = kwargs.get(key)
                if isinstance(raw, list):
                    kwargs[key] = [
                        _prefer_avg_path(v) if isinstance(v, str) else v for v in raw
                    ]
                elif isinstance(raw, str):
                    kwargs[key] = _prefer_avg_path(raw)

            # Also strip completely unknown kwargs so they don't cause TypeErrors
            unknown = [k for k in kwargs if k not in valid_params]
            for k in unknown:
                logger.warning(f"Dropping unknown parameter '{k}' for tool {tool_name}")
                kwargs.pop(k)
            if aliases_applied:
                logger.info(f"Aliased parameters for {tool_name}: {aliases_applied}")

        try:
            from src.analysis.chain_residue_map import (
                ChainSelectionError,
                ensure_chain_residue_map,
                params_need_chain_map,
                translate_selection_params,
            )

            if tool_name not in {"calculate_sasa", "calculate_pocket_sasa", "wrap_trajectory"} and params_need_chain_map(kwargs):
                chain_map = ensure_chain_residue_map(
                    working_dir=self.working_dir,
                    topology_file=kwargs.get("topology_file") or kwargs.get("topology"),
                )
                kwargs = translate_selection_params(kwargs, chain_map)
        except ChainSelectionError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.debug("Chain-map translation skipped for %s: %s", tool_name, exc)

        if tool_name in PCA_TOOL_DEFAULTS:
            user_goal = self.config.get("user_goal") or ""
            before = {k: kwargs.get(k) for k in PCA_TOOL_DEFAULTS[tool_name]}
            kwargs = apply_pca_tool_defaults(tool_name, kwargs, user_goal)
            changed = {
                k: (before.get(k), kwargs[k])
                for k in PCA_TOOL_DEFAULTS[tool_name]
                if before.get(k) is not None and before.get(k) != kwargs[k]
            }
            if changed:
                logger.info(
                    "Applied canonical PCA/FEL defaults for %s (overrode plan params: %s)",
                    tool_name,
                    changed,
                )
            # Defaults may inject params the tool function does not accept (e.g. temperature_k).
            if valid_params is not None:
                for k in [k for k in kwargs if k not in valid_params]:
                    logger.warning(
                        "Dropping PCA-default parameter '%s' for tool %s", k, tool_name
                    )
                    kwargs.pop(k)
        
        try:
            logger.info(f"Executing analysis tool: {tool_name}")
            
            # Execute the tool
            tool_func = self.tools[tool_name]
            
            # Add working directory only if the function signature accepts it
            # ALWAYS force working_dir to agent directory (prevent file leaks)
            # even if the LLM plan supplies a different value
            actual_func = tool_func.func if hasattr(tool_func, 'func') else tool_func
            try:
                sig = inspect.signature(actual_func)
                if 'working_dir' in sig.parameters:
                    kwargs["working_dir"] = self.working_dir
                    logger.debug(f"Forced working_dir={self.working_dir} for {tool_name}")
            except Exception as e:
                logger.debug(f"Could not inspect signature for {tool_name}: {e}")
            
            # StructuredTool objects (from @tool decorator) need special handling
            if hasattr(tool_func, 'func'):
                # @tool decorator wraps function in StructuredTool - use .func
                result = tool_func.func(**kwargs)
            elif hasattr(tool_func, 'invoke'):
                # Alternative: use LangChain's invoke method
                result = tool_func.invoke(kwargs)
            else:
                # Direct function call (backward compatibility)
                result = tool_func(**kwargs)
            
            # Guard against tools that return None instead of a dict
            if result is None:
                logger.warning(f"Tool {tool_name} returned None — missing return statement")
                result = {"success": False, "error": "Tool returned None (missing return statement)"}

            if result.get("success"):
                logger.info(f"Tool {tool_name} completed successfully")
                # ── Auto-log to analysis_summary.jsonl ────────────────
                # Built-in tools (rmsd, rmsf, dssp etc.) self-log via
                # append_analysis_summary().  Only auto-log for
                # programmer-generated tools that don't self-log.
                if tool_name not in self._builtin_tool_names and not result.get("_summary_logged"):
                    self._auto_log_summary(tool_name, kwargs, result)
            else:
                logger.error(f"Tool {tool_name} failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Tool execution failed for {tool_name}")
            return {
                "success": False,
                "error": f"Tool execution error: {str(e)}"
            }

    def _auto_log_summary(self, tool_name: str, kwargs: Dict, result: Dict) -> None:
        """Write a summary entry for tools that don't self-log.

        Extracts statistics, file paths, and metadata from the tool's
        return dict and appends to analysis_summary.jsonl.
        """
        try:
            from src.analysis.summary_logger import append_analysis_summary

            # Build a human-friendly analysis_type from tool_name
            analysis_type = tool_name.replace("_", " ").replace("calculate ", "").title()

            # Collect numeric values as statistics
            statistics: Dict[str, Any] = {}
            files: Dict[str, str] = {}
            metadata: Dict[str, Any] = {}

            for k, v in result.items():
                if k in ("success", "error", "message", "_summary_logged"):
                    continue
                if isinstance(v, (int, float)):
                    statistics[k] = v
                elif isinstance(v, str) and ("/" in v or "\\" in v):
                    # Looks like a file path
                    files[k] = v
                elif isinstance(v, dict):
                    # Nested dict — flatten scalar values into statistics
                    for nk, nv in v.items():
                        if isinstance(nv, (int, float)):
                            statistics[nk] = nv
                        elif isinstance(nv, str) and ("/" in nv or "\\" in nv):
                            files[nk] = nv

            # Add input files that were passed as kwargs
            _INPUT_KEYS = ("topology_file", "trajectory_file", "energy_file",
                           "topology", "trajectory")
            for ik in _INPUT_KEYS:
                if ik in kwargs and kwargs[ik]:
                    files.setdefault(ik, str(kwargs[ik]))

            # Collect non-numeric/non-path kwargs as metadata
            for k, v in kwargs.items():
                if k in ("working_dir",):
                    continue
                if k in _INPUT_KEYS:
                    continue
                if isinstance(v, (str, int, float, bool, list)):
                    metadata[k] = v

            # Only log if we have something meaningful
            if statistics or files:
                append_analysis_summary(
                    working_dir=self.working_dir,
                    analysis_type=analysis_type,
                    statistics=statistics,
                    files=files,
                    metadata=metadata,
                )
                logger.info(f"Auto-logged summary for {tool_name}: {len(statistics)} stats, {len(files)} files")
        except Exception as e:
            logger.warning(f"Failed to auto-log summary for {tool_name}: {e}")

    def execute_workflow(
        self,
        topology_file: str,
        trajectory_file: str,
        energy_file: Optional[str] = None,
        analyses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete analysis workflow with multiple tools.
        
        Args:
            topology_file: Topology file path
            trajectory_file: Trajectory file path
            energy_file: Optional energy file path
            analyses: List of analyses to perform (default: ["rmsd", "rmsf", "gyration"])
            
        Returns:
            Dict with aggregated results from all analyses
        """
        if analyses is None:
            analyses = ["rmsd", "rmsf", "gyration"]
        
        logger.info(f"Starting analysis workflow with: {', '.join(analyses)}")
        
        results = {}
        errors = []
        
        # Map analysis names to tool names
        analysis_tool_map = {
            "rmsd": "calculate_rmsd",
            "rmsf": "calculate_rmsf",
            "gyration": "calculate_radius_of_gyration",
            "energy": "analyze_energy",
            "metrics": "extract_trajectory_metrics"
        }
        
        for analysis in analyses:
            tool_name = analysis_tool_map.get(analysis)
            if not tool_name:
                logger.warning(f"Unknown analysis type: {analysis}")
                errors.append(f"Unknown analysis: {analysis}")
                continue
            
            # Prepare parameters based on analysis type
            if analysis == "energy":
                if not energy_file:
                    logger.warning("Energy analysis requested but no energy file provided")
                    errors.append("Energy analysis requires energy_file parameter")
                    continue
                params = {
                    "energy_file": energy_file,
                    "output_file": os.path.join(self.working_dir, f"{analysis}.xvg")
                }
            elif analysis == "metrics":
                params = {
                    "trajectory_file": trajectory_file,
                    "topology_file": topology_file,
                    "output_dir": self.working_dir
                }
            else:
                params = {
                    "topology_file": topology_file,
                    "trajectory_file": trajectory_file,
                    "output_file": os.path.join(self.working_dir, f"{analysis}.dat")
                }
            
            # Execute the analysis
            result = self.execute(tool_name, **params)
            results[analysis] = result
            
            if not result.get("success"):
                errors.append(f"{analysis}: {result.get('error', 'Unknown error')}")
        
        # Create summary
        completed = [a for a, r in results.items() if r.get("success")]
        failed = [a for a, r in results.items() if not r.get("success")]
        
        summary = f"Completed {len(completed)}/{len(analyses)} analyses"
        if completed:
            summary += f". Successful: {', '.join(completed)}"
        if failed:
            summary += f". Failed: {', '.join(failed)}"
        
        return {
            "success": len(failed) == 0,
            "analyses_completed": completed,
            "analyses_failed": failed,
            "results": results,
            "output_directory": self.working_dir,
            "summary": summary,
            "errors": errors if errors else None
        }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available analysis tools."""
        return list(self.tools.keys())
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get description of a specific tool."""
        if tool_name in self.tools:
            tool_func = self.tools[tool_name]
            return tool_func.__doc__
        return None


def run_complete_analysis(
    topology_file: str,
    trajectory_file: str,
    energy_file: Optional[str] = None,
    working_dir: str = "./working_dir/analysis",
    analyses: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete analysis workflow.
    
    Args:
        topology_file: Topology file path
        trajectory_file: Trajectory file path
        energy_file: Optional energy file path
        working_dir: Working directory for outputs
        analyses: List of analyses to perform
        
    Returns:
        Dict with complete analysis results
    """
    config = {"working_directory": working_dir}
    executor = AnalysisToolExecutor(config=config)
    
    return executor.execute_workflow(
        topology_file=topology_file,
        trajectory_file=trajectory_file,
        energy_file=energy_file,
        analyses=analyses
    )
