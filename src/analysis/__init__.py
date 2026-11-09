"""
Analysis tools for SimAgent

Core implementations for MD trajectory analysis:
- RMSD calculation (rmsd_calculator.py)
- RMSF calculation (rmsf_calculator.py)
- Radius of gyration (gyration_calculator.py)
- Energy analysis (energy_analyzer.py)
- Secondary structure (DSSP) analysis (dssp_analyzer.py)

These tools are wrapped by agentic/analysis/tools.py for use in the agent workflow.
"""

from .rmsd_calculator import calculate_rmsd
from .rmsf_calculator import calculate_rmsf
from .gyration_calculator import calculate_radius_of_gyration
from .energy_analyzer import analyze_energy, extract_trajectory_metrics
from .dssp_analyzer import analyze_secondary_structure
from .com_distance_calculator import calculate_com_distance
from .proximity_analyzer import identify_nearby_residues, calculate_min_heavy_atom_distance
from .interface_analyzer import calculate_hbond_occupancy, calculate_salt_bridge_distances

__all__ = [
    "calculate_rmsd",
    "calculate_rmsf",
    "calculate_radius_of_gyration",
    "analyze_energy",
    "extract_trajectory_metrics",
    "analyze_secondary_structure",
    "calculate_com_distance",
    "identify_nearby_residues",
    "calculate_min_heavy_atom_distance",
    "calculate_hbond_occupancy",
    "calculate_salt_bridge_distances",
]
