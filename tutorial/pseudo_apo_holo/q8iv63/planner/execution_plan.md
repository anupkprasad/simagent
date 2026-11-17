# Planner Execution Plan

**Generated:** 2026-11-13 20:15:25
**Phase:** single

## Overview

**Title:** Detailed Natural Language Execution Plan
**Agent sequence:** analysis_agent
**Subtask:** analysis_only

## Full Plan

**Goal**  
The objective is to extract all requested dynamical descriptors from the single 1‑ns trajectory of the q8iv63 system (both apo and holo are present in the same directory).  The analysis must produce the standard per‑simulation output files so that a later, multi‑simulation overlay step can combine them.  No cross‑simulation calculations are performed at this stage.

**Analysis**  
The analysis will be carried out by the *Analysis Agent* using the built‑in MDAnalysis‑based tools that are available in the registry.  The following metrics are required:

1. **Backbone RMSD** – time‑series of the backbone RMSD relative to the first frame.  
2. **Per‑residue RMSF** – overall RMSF for every residue, plus a focused bar plot for residues 150–200.  
3. **Radius of gyration** – overall Rg time‑series.  
4. **COM distance (pocket)** – distance between the COM of ATP and the COM of the catalytic pocket defined as all protein atoms within 5 Å of ATP in frame 0.  
5. **Cα DCCM** – full‑protein dynamic cross‑correlation matrix.  
6. **Secondary structure (DSSP)** – per‑residue secondary‑structure assignment for every frame, with a time‑series of the fraction of each secondary‑structure type for the whole protein and for residues 150–200.

All outputs will be written to the directory  
`/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8iv63/analysis/`  
using the exact basenames specified in the standard file‑naming convention.

**Execution Sequence**  

1. **File discovery** – The Analysis Agent will locate the topology (`md.gro`) and trajectory (`md.xtc`) files in the `hpc/` subdirectory of the working directory.  No preprocessing or wrapping is required.

2. **Backbone RMSD** –  
   *Tool:* `calculate_rmsd`  
   *Arguments:* `topology_file=md.gro`, `trajectory_file=md.xtc`  
   *Outputs:* `rmsd.dat` (time‑series) and `rmsd.png` (line plot).  

3. **Per‑residue RMSF** –  
   *Tool:* `calculate_rmsf`  
   *Arguments:* same topology and trajectory files, with the default alignment on the backbone.  
   *Outputs:* `rmsf.dat` (residue‑wise RMSF) and `rmsf.png` (bar plot of all residues).  

4. **Radius of gyration** –  
   *Tool:* `calculate_radius_of_gyration`  
   *Arguments:* `align_before_rg=True` to ensure the Rg is computed on aligned frames.  
   *Outputs:* `gyration.dat` and `gyration.png`.  

5. **COM distance (pocket)** –  
   *Tool:* `calculate_ligand_pocket_distance`  
   *Arguments:* `ligand_selection="resname ATP"`, `cutoff=5.0`, `output_file="ligand_pocket_distance.csv"`  
   *Outputs:* `ligand_pocket_distance.csv` (time‑series of COM distances) and `ligand_pocket_distance.png` (line plot).  

6. **Cα DCCM** –  
   *Tool:* `calculate_dccm`  
   *Arguments:* `topology_file=md.gro`, `trajectory_file=md.xtc` (the tool automatically selects Cα atoms).  
   *Outputs:* `dccm.dat` (matrix) and `dccm_heatmap.png`.  

7. **Secondary structure (DSSP)** –  
   *Tool:* `analyze_secondary_structure`  
   *Arguments:* same topology and trajectory.  
   *Outputs:* `dssp.dat` (per‑residue per‑frame DSSP assignments) and `dssp.png` (heat‑map of secondary‑structure evolution).  

8. **Post‑processing for the 150–200 window** –  
   The Analysis Agent will run a short Python script that:
   * Reads `rmsf.dat` and extracts the RMSF values for residues 150–200.  
   * Writes `rmsf_150-200.dat` and produces a bar plot `rmsf_150-200.png`.  
   * Reads `dssp.dat`, counts the fraction of each DSSP type per frame for residues 150–200, and writes `dssp_150-200.dat`.  
   * Generates a line plot `dssp_150-200.png` showing the time evolution of the secondary‑structure fractions for the window.  

   These files follow the qualified‑name convention (`{metric}_{qualifier}.dat` and `.png`) so they do not overwrite the overall metrics.

**Expected Outcomes**  

After the Analysis Agent completes the above steps, the `analysis/` directory will contain the following files:

- `rmsd.dat`, `rmsd.png`  
- `rmsf.dat`, `rmsf.png`, `rmsf_150-200.dat`, `rmsf_150-200.png`  
- `gyration.dat`, `gyration.png`  
- `ligand_pocket_distance.csv`, `ligand_pocket_distance.png`  
- `dccm.dat`, `dccm_heatmap.png`  
- `dssp.dat`, `dssp.png`, `dssp_150-200.dat`, `dssp_150-200.png`  

These outputs provide all the requested metrics for the q8iv63 trajectory.  A subsequent, multi‑simulation workflow can now aggregate the standard files (`rmsd.dat`, `rmsf.dat`, etc.) from all four proteins to generate overlay plots and statistical tables, and the final report can incorporate literature references to contextualize the simulation findings.