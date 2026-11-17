# Planner Execution Plan

**Generated:** 2026-11-13 20:15:57
**Phase:** single

## Overview

**Title:** Detailed Natural Language Execution Plan
**Agent sequence:** analysis_agent
**Subtask:** analysis_only

## Full Plan

**Goal**  
The Analysis Agent will process the single trajectory located in  
`/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/p21860/hpc/` and generate all of the requested per‑simulation metrics. The outputs will be written to  
`/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/p21860/analysis/` using the standard basenames so that a later combined‑analysis step can overlay and tabulate the results from the eight independent runs.

**Analysis Tasks**  

1. **Backbone RMSD vs. time** – `calculate_rmsd`  
   *Input:* topology and trajectory files.  
   *Output:* `rmsd.dat` (time‑series) and `rmsd.png` (plot).  

2. **Per‑residue RMSF** – `calculate_rmsf`  
   *Input:* topology and trajectory files.  
   *Output:* `rmsf.dat` (all residues) and `rmsf.png` (full profile).  
   *Post‑processing:* The Analysis Agent will read `rmsf.dat`, extract the entries for residues 150–200 (if they exist), and write a focused data file `rmsf_150-200.dat` and a corresponding plot `rmsf_150-200.png`.  

3. **Radius of gyration** – `calculate_radius_of_gyration`  
   *Input:* topology and trajectory files.  
   *Output:* `gyration.dat` and `gyration.png`.  

4. **COM distance between ATP and the catalytic pocket** – `calculate_ligand_pocket_distance`  
   *Input:* topology and trajectory files, `ligand_selection="resname ATP"`, `cutoff=5.0`.  
   *Output:* `ligand_pocket_distance.csv` and `ligand_pocket_distance.png`.  
   *Note:* If the trajectory is apo (no ATP), the tool will return an empty file; the Analysis Agent will detect this and skip the output.  

5. **Dynamic Cross‑Correlation Matrix (DCCM)** – `calculate_dccm`  
   *Input:* topology and trajectory files.  
   *Output:* `dccm.npy` (matrix) and `dccm_heatmap.png`.  

6. **Secondary‑structure assignment (DSSP)** – `analyze_secondary_structure`  
   *Input:* topology and trajectory files.  
   *Output:* `dssp.dat` (time‑resolved secondary‑structure per residue) and `dssp.png` (heat‑map).  
   *Post‑processing:* The Analysis Agent will filter `dssp.dat` for residues 150–200, producing `dssp_150-200.dat` and `dssp_150-200.png`.  

**Execution Sequence**  

1. The Analysis Agent will locate the topology (`md.gro`) and trajectory (`md.xtc`) files in the `hpc/` directory.  
2. It will invoke `calculate_rmsd` first, capturing the RMSD time‑series and plot.  
3. Next, `calculate_rmsf` will be run to obtain the per‑residue flexibility profile.  
4. The agent will then call `calculate_radius_of_gyration` to compute the compactness metric.  
5. If an ATP residue is present, `calculate_ligand_pocket_distance` will be executed to track the ligand‑pocket COM distance.  
6. `calculate_dccm` will be run to generate the full correlation matrix and heat‑map.  
7. Finally, `analyze_secondary_structure` will produce the DSSP data and heat‑map.  
8. After each primary tool finishes, the agent will perform the subset post‑processing steps for residues 150–200 (RMSF and DSSP) and write the qualified files.  
9. All output files will be written to the `analysis/` directory with the exact basenames specified in the standard naming convention.  

**Expected Outcomes**  

- `rmsd.dat` and `rmsd.png` showing backbone deviation over the simulation.  
- `rmsf.dat` and `rmsf.png` with a full per‑residue RMSF profile; `rmsf_150-200.dat` and `rmsf_150-200.png` highlighting the 150–200 window.  
- `gyration.dat` and `gyration.png` depicting the radius of gyration trajectory.  
- If ATP is present: `ligand_pocket_distance.csv` and `ligand_pocket_distance.png` tracking the COM distance between ATP and the pocket defined by atoms within 5 Å at frame 0.  
- `dccm.npy` and `dccm_heatmap.png` illustrating correlated motions across the protein.  
- `dssp.dat` and `dssp.png` showing the time evolution of secondary structure; `dssp_150-200.dat` and `dssp_150-200.png` focusing on residues 150–200.  

These files will be ready for the subsequent combined‑analysis phase, where overlay plots and statistical tables across all eight simulations will be generated.