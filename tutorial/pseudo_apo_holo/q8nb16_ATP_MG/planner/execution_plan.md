# Planner Execution Plan

**Generated:** 2026-11-13 20:26:13
**Phase:** single

## Overview

**Title:** Detailed Natural Language Execution Plan
**Agent sequence:** analysis_agent
**Subtask:** analysis_only

## Full Plan

**Goal**  
Perform a complete, per‑simulation trajectory analysis for the 1‑ns MD run located in  
`/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8nb16_ATP_MG/hpc/`.  
The analysis must compute the following metrics and generate the standard output
files and plots that will later be combined across all eight systems:

* Backbone RMSD (overall protein)  
* Per‑residue RMSF (overall protein) and a bar plot for residues 150–200 when they exist  
* Radius of gyration (Rg) of the whole protein  
* COM distance between ATP and the catalytic pocket (protein atoms within 5 Å of ATP at frame 0) – only for holo runs  
* Cα dynamic cross‑correlation matrix (DCCM)  
* DSSP secondary‑structure evolution for the whole protein and for residues 150–200  

All results will be written to the per‑simulation `analysis/` directory using the
canonical filenames defined in the workflow specification.

---

### Analysis Steps

1. **Load trajectory and topology**  
   * Topology: `md.gro`  
   * Trajectory: `md.xtc`  
   * Energy file is not needed for the requested metrics.  
   * The Analysis Agent will locate these files automatically in the `hpc/` folder.

2. **Backbone RMSD**  
   * Tool: `calculate_rmsd`  
   * Alignment: automatic (default) on the protein backbone.  
   * Output: `analysis/rmsd.dat` (time‑series) and `analysis/rmsd.png` (plot).  

3. **Per‑residue RMSF**  
   * Tool: `calculate_rmsf`  
   * Alignment: automatic on the protein backbone.  
   * Output: `analysis/rmsf.dat` (residue‑by‑residue RMSF) and `analysis/rmsf.png` (line plot).  
   * **Bar plot for residues 150–200** – the Analysis Agent will read `rmsf.dat`, filter residues 150–200, and generate `analysis/rmsf_150-200.png`.  
   * If residues 150–200 are not present in the current PDB, this step will be skipped and a log entry will note the absence.

4. **Radius of gyration**  
   * Tool: `calculate_radius_of_gyration`  
   * Alignment: `align_before_rg=True` to compute Rg on aligned coordinates.  
   * Output: `analysis/gyration.dat` and `analysis/gyration.png`.

5. **COM distance to catalytic pocket (holo only)**  
   * Condition: check if the trajectory contains a ligand named “ATP”.  
   * Tool: `calculate_ligand_pocket_distance`  
   * Parameters: `ligand_selection="resname ATP"`, `cutoff=5.0`.  
   * Output: `analysis/ligand_pocket_distance.csv` (time‑series) and `analysis/ligand_pocket_distance.png`.  
   * For apo runs this step is omitted; the agent will log that no pocket distance was computed.

6. **Dynamic cross‑correlation matrix (DCCM)**  
   * Tool: `calculate_dccm`  
   * Alignment: default (protein backbone).  
   * Output: `analysis/dccm.dat` (matrix file) and `analysis/dccm_heatmap.png` (heatmap).

7. **Secondary‑structure evolution (DSSP)**  
   * Tool: `analyze_secondary_structure`  
   * **Whole protein**: default selection.  
     * Output: `analysis/dssp.dat` and `analysis/dssp.png`.  
   * **Residues 150–200**: selection `"resid 150-200"`.  
     * Output: `analysis/dssp_150-200.dat` and `analysis/dssp_150-200.png`.  
   * If residues 150–200 are absent, the agent will skip this subset and log the omission.

---

### Execution Sequence (in natural language)

1. The Analysis Agent starts by confirming the presence of the topology and trajectory files in the `hpc/` directory.  
2. It invokes `calculate_rmsd` with the topology and trajectory, directing the output to `analysis/rmsd.dat` and `analysis/rmsd.png`.  
3. Next, it runs `calculate_rmsf` on the same files, saving the results to `analysis/rmsf.dat` and `analysis/rmsf.png`.  
4. The agent then checks whether residues 150–200 exist; if so, it extracts the RMSF values for that range and creates a bar plot `analysis/rmsf_150-200.png`.  
5. `calculate_radius_of_gyration` is executed, producing `analysis/gyration.dat` and `analysis/gyration.png`.  
6. The agent scans the topology for an ATP ligand. If found, it calls `calculate_ligand_pocket_distance` with a 5 Å cutoff, writing `analysis/ligand_pocket_distance.csv` and `analysis/ligand_pocket_distance.png`.  
7. `calculate_dccm` is run next, yielding `analysis/dccm.dat` and `analysis/dccm_heatmap.png`.  
8. Finally, `analyze_secondary_structure` is executed twice: once for the whole protein (outputs `analysis/dssp.dat` and `analysis/dssp.png`) and once for residues 150–200 (outputs `analysis/dssp_150-200.dat` and `analysis/dssp_150-200.png`).  
9. The agent logs any skipped steps (e.g., missing residues 150–200 or absence of ATP) and completes the per‑simulation analysis.

---

### Expected Outcomes

After the Analysis Agent finishes, the `analysis/` directory will contain:

* `rmsd.dat`, `rmsd.png`  
* `rmsf.dat`, `rmsf.png`, and if applicable `rmsf_150-200.png`  
* `gyration.dat`, `gyration.png`  
* `ligand_pocket_distance.csv`, `ligand_pocket_distance.png` (only for holo)  
* `dccm.dat`, `dccm_heatmap.png`  
* `dssp.dat`, `dssp.png`  
* `dssp_150-200.dat`, `dssp_150-200.png` (if residues 150–200 exist)

These files provide all the metrics requested for a single trajectory and are named exactly as required for the subsequent multi‑simulation overlay and statistical aggregation steps.