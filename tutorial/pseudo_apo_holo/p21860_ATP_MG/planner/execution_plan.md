# Planner Execution Plan

**Generated:** 2026-11-13 20:17:31
**Phase:** single

## Overview

**Title:** Detailed Natural Language Execution Plan
**Agent sequence:** analysis_agent
**Subtask:** analysis_only

## Full Plan

**Goal**  
For the single 1‑ns trajectory that resides in the working directory, compute the following per‑simulation metrics and visualisations:

1. Backbone RMSD (protein only)  
2. Per‑residue RMSF (protein only) – with an additional bar plot for residues 150–200 if they exist in the chain  
3. Radius of gyration (protein only)  
4. For holo runs (ATP present) the centre‑of‑mass distance between ATP and the catalytic pocket (atoms within 5 Å of ATP at frame 0)  
5. Cα Dynamic Cross‑Correlation Matrix (DCCM) for the protein  
6. If both apo and holo trajectories are available, a difference map (ΔDCCM) between the two DCCMs  
7. DSSP time evolution for the full protein and for the active‑site window (residues 150–200 if present)  

All outputs are written to the directory **/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/p21860_ATP_MG/analysis/** using the standard basenames defined in the specification (e.g., `rmsd.dat`, `rmsf.dat`, `gyration.dat`, `ligand_pocket_distance.csv`, `dccm.dat`, `dccm_diff.dat`, `dssp.dat`, `dssp_window.dat`).  The analysis agent will automatically discover the topology (`md.gro`) and trajectory (`md.xtc`) in the `hpc/` sub‑folder.

---

### Analysis Sequence (performed by the **Analysis Agent**)

1. **RMSD**  
   *Tool:* `calculate_rmsd`  
   *Command:* `calculate_rmsd(topology_file=md.gro, trajectory_file=md.xtc)`  
   *Output:* `rmsd.dat` (time series) and `rmsd.png` (plot).  

2. **RMSF**  
   *Tool:* `calculate_rmsf`  
   *Command:* `calculate_rmsf(topology_file=md.gro, trajectory_file=md.xtc)`  
   *Output:* `rmsf.dat` (per‑residue RMSF) and `rmsf.png` (bar plot of all residues).  
   *Post‑processing:* If residues 150–200 are present, extract the subset from `rmsf.dat` and generate `rmsf_150-200.dat` and `rmsf_150-200.png`.  

3. **Radius of Gyration**  
   *Tool:* `calculate_radius_of_gyration`  
   *Command:* `calculate_radius_of_gyration(topology_file=md.gro, trajectory_file=md.xtc, align_before_rg=True)`  
   *Output:* `gyration.dat` and `gyration.png`.  

4. **DCCM (protein only)**  
   *Tool:* `calculate_dccm`  
   *Command:* `calculate_dccm(topology_file=md.gro, trajectory_file=md.xtc)`  
   *Output:* `dccm.dat` and `dccm_heatmap.png`.  

5. **COM distance (holo only)**  
   *Detection:* The agent checks whether the topology contains an atom named “ATP”.  
   *If ATP is present:*  
   *Tool:* `calculate_ligand_pocket_distance`  
   *Command:* `calculate_ligand_pocket_distance(topology_file=md.gro, trajectory_file=md.xtc, ligand_selection="resname ATP", cutoff=5.0, output_file="ligand_pocket_distance.csv")`  
   *Output:* `ligand_pocket_distance.csv` and `ligand_pocket_distance.png`.  

6. **DCCM Difference (apo vs holo)**  
   *Prerequisite:* The agent must have two separate trajectory directories (apo and holo).  
   *Step 1:* Run `calculate_dccm` on the apo trajectory → `dccm_apo.dat`.  
   *Step 2:* Run `calculate_dccm` on the holo trajectory → `dccm_holo.dat`.  
   *Step 3:* *Tool:* `plot_dccm_difference`  
   *Command:* `plot_dccm_difference(reference_dccm_file="dccm_apo.dat", compare_dccm_file="dccm_holo.dat", output_file="dccm_diff.dat")`  
   *Output:* `dccm_diff.dat` and `dccm_diff_heatmap.png`.  

7. **DSSP Time Evolution**  
   *Tool:* `analyze_secondary_structure`  
   *Command:* `analyze_secondary_structure(topology_file=md.gro, trajectory_file=md.xtc)`  
   *Output:* `dssp.dat` (residue‑by‑residue secondary‑structure per frame) and `dssp.png` (heat‑map).  
   *Post‑processing:*  
   - Extract the window 150–200 from `dssp.dat` (if present) to produce `dssp_window.dat`.  
   - Generate `dssp_window.png` from the window data.  

---

### Expected Outcomes

- **Time‑series files** (`rmsd.dat`, `gyration.dat`, `ligand_pocket_distance.csv`) and corresponding plots (`rmsd.png`, `gyration.png`, `ligand_pocket_distance.png`).  
- **Per‑residue RMSF** (`rmsf.dat`, `rmsf.png`) and, if applicable, a focused window plot (`rmsf_150-200.dat`, `rmsf_150-200.png`).  
- **DCCM heat‑map** (`dccm_heatmap.png`) and, when both apo and holo are present, a ΔDCCM heat‑map (`dccm_diff_heatmap.png`).  
- **Secondary‑structure evolution** (`dssp.dat`, `dssp.png`) and the active‑site window (`dssp_window.dat`, `dssp_window.png`).  
- All files are stored in the analysis directory with the exact basenames required for later overlay and statistical aggregation.  

The Analysis Agent will log each step, capture any warnings, and produce a concise metrics summary (mean RMSD, mean RMSF, mean Rg, mean COM distance, average DCCM correlation, DSSP percentages) that can be consumed by a subsequent Reporter Agent to generate the final report and literature‑correlated discussion.