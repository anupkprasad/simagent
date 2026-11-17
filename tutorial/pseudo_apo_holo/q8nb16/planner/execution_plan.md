# Planner Execution Plan

**Generated:** 2026-11-13 20:25:57
**Phase:** single

## Overview

**Title:** Detailed Natural Language Execution Plan
**Agent sequence:** analysis_agent
**Subtask:** analysis_only

## Full Plan

**Goal**  
Produce a complete, per‑simulation set of dynamical metrics for the 1‑ns trajectory of the current system.  
The metrics requested are: backbone RMSD, per‑residue RMSF (overall and residues 150–200 when present), radius of gyration, COM distance between ATP and the catalytic pocket (atoms within 5 Å of ATP at frame 0), Cα dynamic cross‑correlation matrix (DCCM), and DSSP time evolution (overall and residues 150–200).  
All results will be written to the simulation’s *analysis* directory with the standard basenames so that a later, multi‑simulation phase can automatically collect and overlay them.  No cross‑simulation comparisons, overlay plots, or statistical tables are performed at this stage.

---

### Analysis Sequence (performed by the **Analysis Agent**)

1. **Environment Setup**  
   * The agent automatically discovers the topology (`md.gro`) and trajectory (`md.xtc`) in the `hpc/` sub‑folder of the working directory.  
   * It creates the output directory `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8nb16/analysis/` if it does not already exist.

2. **Backbone RMSD**  
   * Tool: `calculate_rmsd`  
   * Arguments: `topology_file=md.gro`, `trajectory_file=md.xtc`  
   * Output: `rmsd.dat` (time‑series) and `rmsd.png` (plot).  
   * The tool aligns each frame to the first frame before computing RMSD, so the plot shows the stability of the backbone over the 1‑ns run.

3. **Per‑Residue RMSF – Overall**  
   * Tool: `calculate_rmsf`  
   * Arguments: `topology_file=md.gro`, `trajectory_file=md.xtc`  
   * Output: `rmsf.dat` (residue‑wise RMSF) and `rmsf.png` (bar plot).  
   * This gives the flexibility profile of every residue in the protein.

4. **Per‑Residue RMSF – Residues 150–200**  
   * Tool: `calculate_rmsf`  
   * Arguments: `topology_file=md.gro`, `trajectory_file=md.xtc`, `selection="resid 150:200"`  
   * Output: `rmsf_150to200.dat` and `rmsf_150to200.png`.  
   * If residues 150–200 are absent, the tool will produce an empty file; the agent will log a warning but continue.

5. **DSSP Time Evolution – Overall**  
   * Tool: `analyze_secondary_structure`  
   * Arguments: `topology_file=md.gro`, `trajectory_file=md.xtc`  
   * Output: `dssp.dat` (residue‑by‑residue secondary‑structure assignment over time) and `dssp.png` (heat‑map).  
   * The plot shows the evolution of helices, strands, turns, etc., across the trajectory.

6. **DSSP Time Evolution – Residues 150–200**  
   * Tool: `analyze_secondary_structure`  
   * Arguments: `topology_file=md.gro`, `trajectory_file=md.xtc`, `selection="resid 150:200"`  
   * Output: `dssp_150to200.dat` and `dssp_150to200.png`.  
   * This focuses on the activation‑loop region (or equivalent) that the user highlighted.

7. **Radius of Gyration**  
   * Tool: `calculate_radius_of_gyration`  
   * Arguments: `topology_file=md.gro`, `trajectory_file=md.xtc`, `align_before_rg=True`  
   * Output: `gyration.dat` and `gyration.png`.  
   * The alignment ensures that changes in compactness are not confounded by overall rotation/translation.

8. **Cα DCCM**  
   * Tool: `calculate_dccm`  
   * Arguments: `topology_file=md.gro`, `trajectory_file=md.xtc`, `selection="protein and name CA"`  
   * Output: `dccm_heatmap.png` (and an internal matrix file if the tool writes one).  
   * The heat‑map visualises correlated and anti‑correlated motions of Cα atoms over the 1‑ns window.

9. **COM Distance – ATP vs Catalytic Pocket**  
   * Tool: `calculate_ligand_pocket_distance`  
   * Arguments: `topology_file=md.gro`, `trajectory_file=md.xtc`, `ligand_selection="resname ATP"`, `cutoff=5.0`  
   * Output: `ligand_pocket_distance.csv` and `ligand_pocket_distance.png`.  
   * The tool first identifies all protein atoms within 5 Å of ATP at frame 0, then tracks the COM distance between that pocket set and the ATP COM for every frame.

10. **Result Packaging**  
    * The agent copies all generated `.dat`, `.csv`, and `.png` files into the analysis directory.  
    * It writes a concise log summarising the completion of each step and any warnings (e.g., missing residues 150–200).

---

### Expected Outcomes

- **Data files** (`*.dat`, `*.csv`) containing time‑series or per‑residue values for all requested metrics.  
- **Plots** (`*.png`) that visually summarise RMSD, RMSF, Rg, DCCM, DSSP evolution, and COM distance.  
- All files follow the standard basenames (`rmsd.dat`, `rmsf.dat`, `gyration.dat`, `dccm_heatmap.png`, `dssp.dat`, `ligand_pocket_distance.csv`, etc.) so that a later, multi‑simulation phase can automatically aggregate and overlay them.  
- A log file documenting the execution and any issues.

These outputs will be fed into the subsequent overlay‑and‑statistics phase, where the eight systems will be compared, and into the final report generation phase, where the results will be contextualised with literature on activation‑loop conformations, allosteric regulation, and MD/experimental dynamics for ERBB3, VRK3, MLKL, and TITIN.