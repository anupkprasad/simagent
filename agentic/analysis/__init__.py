"""
Analysis module for SimAgent

Provides MD trajectory analysis capabilities including:
- RMSD (Root Mean Square Deviation) - structural stability
- RMSF (Root Mean Square Fluctuation) - residue flexibility
- Radius of gyration - protein compactness
- Energy analysis - thermodynamic properties
- Trajectory metrics - basic statistics
- Data visualization and plotting - single and multi-panel plots

Components:
- MDAnalysisAgent: Main agent class with LLM-powered planning
- Tools: Wrapper functions for src/analysis/ implementations
- Schemas: Workflow orchestration and result models

Note: Tool-level input validation is handled by LangChain's @tool decorator.
"""

from .analysis_agent import MDAnalysisAgent
from .tools import (
    calculate_rmsd,
    calculate_rmsf,
    calculate_radius_of_gyration,
    analyze_energy,
    extract_trajectory_metrics,
    plot_md_data,
    plot_md_multipanel,
    plot_combined_data,
    AnalysisToolExecutor,
    run_complete_analysis
)
from .schemas import (
    AnalysisResult,
    RMSDResult,
    RMSFResult,
    GyrationResult,
    EnergyResult,
    AnalysisWorkflowInput,
    AnalysisWorkflowOutput
)

__all__ = [
    # Agent
    "MDAnalysisAgent",
    
    # Analysis Tools
    "calculate_rmsd",
    "calculate_rmsf",
    "calculate_radius_of_gyration",
    "analyze_energy",
    "extract_trajectory_metrics",
    
    # Plotting Tools
    "plot_md_data",
    "plot_md_multipanel",
    "plot_combined_data",
    
    # Executor and Workflows
    "AnalysisToolExecutor",
    "run_complete_analysis",
    
    # Schemas (workflow and results only)
    "AnalysisResult",
    "RMSDResult",
    "RMSFResult",
    "GyrationResult",
    "EnergyResult",
    "AnalysisWorkflowInput",
    "AnalysisWorkflowOutput",
]


