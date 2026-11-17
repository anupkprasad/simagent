# Planner Execution Plan

**Generated:** 2026-11-13 20:28:23
**Phase:** single

## Overview

**Title:** Detailed Natural Language Execution Plan
**Agent sequence:** analysis_agent
**Subtask:** analysis_only

## Full Plan

**Goal**  
Produce a complete, per‑simulation analysis of the eight 1 ns trajectories (apo and holo for ERBB3, VRK3, MLKL, TITIN).  The analysis must generate the following metrics for each simulation, write the standard output files in the shared *analysis* directory, and leave the data ready for a later overlay‑plot and statistical‑table step performed by the Reporter agent.  No new simulations, preprocessing, or HPC submissions are required.

**Analysis Scope**  
The user explicitly requests the following per‑simulation metrics:

| Metric | Tool | Output files (standard names) |
|--------|------|--------------------------------|
| Backbone RMSD | `calculate_rmsd` | `rmsd.dat`, `rmsd.png` |
| Radius of gyration | `calculate_radius_of_gyration` | `gyration.dat`, `gyration.png` |
| Cα Dynamic Cross‑Correlation Matrix | `calculate_dccm` | `dccm.npy` (or `.dat`), `dccm_heatmap.png` |
| DSSP evolution (whole protein) | `analyze_secondary_structure` | `dssp.dat`, `dssp.png` |
| DSSP evolution (residues 150–200) | **Custom tool** | `dssp_150to200.dat`, `dssp_150to200.png` |
| Per‑residue RMSF (whole protein) | `calculate_rmsf` | `rmsf.dat`, `rmsf.png` |
| Per‑residue RMSF (residues 150–200) | **Custom tool** | `rmsf_150to200.dat`, `rmsf_150to200.png` |
| COM distance between ATP and catalytic pocket (holo only) | `calculate_ligand_pocket_distance` | `ligand_pocket_distance.csv`, `ligand_pocket_distance.png` |

The Reporter agent will later combine these files to produce overlay plots and statistical tables, and will write the final literature‑review report.

**Missing Capabilities**  
The built‑in tool set does not provide a way to compute RMSF or DSSP for an arbitrary residue range.  Therefore we must create two custom tools:

1. **`calculate_subset_rmsf`** – accepts a topology, trajectory, a residue‑range selection string (e.g., `"resid 150-200"`), and an output file name; it computes per‑residue RMSF only for the selected residues and writes the data and a bar‑plot.

2. **`calculate_subset_dssp`** – accepts a topology, trajectory, a residue‑range selection string, and an output file name; it runs DSSP on the selected residues, writes a time‑series file and a heat‑map plot.

We will explicitly request these tools from the Programmer agent.

**Execution Sequence (per simulation)**  

1. **Discover files** – The Analysis agent automatically locates the topology (`md.gro`) and trajectory (`md.xtc`) in the simulation’s *hpc* sub‑folder.

2. **Backbone RMSD** – Run `calculate_rmsd` with the discovered topology and trajectory.  The tool writes `rmsd.dat` (time‑series) and `rmsd.png` (plot) into `/analysis/`.

3. **Radius of gyration** – Run `calculate_radius_of_gyration`.  Output files: `gyration.dat`, `gyration.png`.

4. **Cα DCCM** – Run `calculate_dccm`.  The tool writes a matrix file (`dccm.npy` or `.dat`) and a heat‑map `dccm_heatmap.png`.

5. **Whole‑protein DSSP** – Run `analyze_secondary_structure`.  Output: `dssp.dat`, `dssp.png`.

6. **Subset DSSP (150–200)** – Run the custom tool `calculate_subset_dssp` with selection `"resid 150-200"`.  Output: `dssp_150to200.dat`, `dssp_150to200.png`.

7. **Whole‑protein RMSF** – Run `calculate_rmsf`.  Output: `rmsf.dat`, `rmsf.png`.

8. **Subset RMSF (150–200)** – Run the custom tool `calculate_subset_rmsf` with selection `"resid 150-200"`.  Output: `rmsf_150to200.dat`, `rmsf_150to200.png`.

9. **COM distance (holo only)** – If the simulation is a holo run, invoke `calculate_ligand_pocket_distance` with  
   * `ligand_selection="resname ATP"`  
   * `cutoff=5.0` (typical pocket radius)  
   * `output_file="ligand_pocket_distance.csv"`  
   The tool writes the CSV and a corresponding PNG.

10. **Finish** – All required files are now present in `/analysis/`.  No further per‑simulation analysis is needed.

**Reporter Agent (post‑analysis)**  

After the Analysis agent completes all eight simulations, the Reporter agent will:

* Read the standard files (`rmsd.dat`, `gyration.dat`, `dccm.npy`, `dssp.dat`, `rmsf.dat`, `ligand_pocket_distance.csv`, etc.) from each simulation’s *analysis* folder.
* Generate overlay plots (e.g., RMSD, Rg, DCCM heat‑maps, DSSP heat‑maps, RMSF bar‑plots) comparing apo vs holo for each protein.
* Compile statistical tables summarizing mean/variance of each metric across the eight systems.
* Write a narrative report that reviews literature on each protein’s activation‑loop conformation, allosteric regulation, and prior MD or experimental dynamics, and correlates these findings with the computed analyses.

**Why This Plan Works**

* All requested metrics are covered by existing tools or by the two custom tools we explicitly request.
* The output filenames follow the mandated standard basenames, ensuring that a later combined‑analysis step can automatically aggregate the data.
* The plan respects the instruction to avoid any new simulations, preprocessing, or HPC submissions.
* The Reporter agent is responsible for the final comparative visualizations and literature synthesis, keeping the Analysis agent focused solely on metric computation.

**Next Steps**

1. **Programmer Agent** – Create `calculate_subset_rmsf` and `calculate_subset_dssp` as described.
2. **Analysis Agent** – Execute the sequence above for each of the eight simulation directories.
3. **Reporter Agent** – Generate the overlay plots, tables, and final report.

This workflow delivers exactly the analyses the user requested, with clear separation of responsibilities and adherence to the file‑naming conventions.