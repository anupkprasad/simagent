# Planner Execution Plan

**Generated:** 2026-11-13 20:16:06
**Phase:** single

## Overview

**Title:** Detailed Natural Language Execution Plan
**Agent sequence:** analysis_agent
**Subtask:** analysis_only

## Full Plan

**Goal**  
The objective is to extract all requested dynamical and structural descriptors from the single 1‑ns trajectory of the holo system *q8iv63_ATP_MG*.  The analysis must produce the standard output files that will later be aggregated across the eight simulations, but for this step only the data for the current simulation is required.  No cross‑simulation comparison or overlay is performed here; that will be handled in a later phase.

**Analysis**  
The following metrics are required:

1. **Backbone RMSD** – time‑dependent deviation of the protein backbone from the first frame.  
2. **Per‑residue RMSF** – flexibility profile of each Cα atom.  
3. **Radius of gyration** – compactness of the whole protein.  
4. **Ligand‑pocket COM distance** – distance between the COM of ATP and the COM of the catalytic pocket defined by all protein atoms within 5 Å of ATP at frame 0.  
5. **Cα Dynamic Cross‑Correlation Matrix (DCCM)** – full‑protein correlation of Cα motions.  
6. **Secondary‑structure evolution (DSSP)** – assignment of secondary‑structure elements for every residue over time, with a focused plot for residues 150–200 if that range exists in the sequence.

All outputs must follow the prescribed naming convention and be written to the directory  
`/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8iv63_ATP_MG/analysis/`.

**Execution Sequence**  
The Analysis Agent will perform the tasks in the order below, ensuring that each tool receives the correct input files and that the resulting data files are named exactly as required.

1. **Backbone RMSD**  
   - Invoke `calculate_rmsd` with the topology and trajectory located in the *hpc* sub‑directory.  
   - The tool aligns each frame to the first frame before computing RMSD.  
   - Output files: `rmsd.dat` (time‑series) and `rmsd.png` (plot).  

2. **Per‑residue RMSF**  
   - Run `calculate_rmsf` on the same trajectory.  
   - The tool aligns on the protein backbone and reports Cα RMSF per residue.  
   - Output files: `rmsf.dat` and `rmsf.png`.  

3. **Radius of gyration**  
   - Execute `calculate_radius_of_gyration` with `align_before_rg=True` to ensure consistency with the RMSD calculation.  
   - Output files: `gyration.dat` and `gyration.png`.  

4. **Ligand‑pocket COM distance**  
   - Use `calculate_ligand_pocket_distance` with `ligand_selection="resname ATP"` and `cutoff=5.0`.  
   - The tool identifies all protein atoms within 5 Å of ATP at frame 0, defines the pocket, and tracks the COM distance over time.  
   - Output files: `ligand_pocket_distance.csv` and `ligand_pocket_distance.png`.  

5. **Dynamic Cross‑Correlation Matrix**  
   - Call `calculate_dccm` on the trajectory, selecting all Cα atoms.  
   - The tool returns a full‑protein DCCM matrix.  
   - Output files: `dccm.npy` (or `.dat` depending on the tool’s default) and `dccm_heatmap.png`.  

6. **Secondary‑structure evolution**  
   - Run `analyze_secondary_structure` on the trajectory.  
   - The tool computes DSSP assignments for every residue at every frame, produces a heat‑map of secondary‑structure evolution, and writes the per‑residue time series.  
   - Output files: `dssp.dat` and `dssp.png`.  
   - After the run, the agent will inspect the residue numbering in the topology.  If residues 150–200 are present, the agent will extract the corresponding subset from `dssp.dat` and generate a focused plot.  The subset files will be named `dssp_150to200.dat` and `dssp_150to200.png`.  If the range is absent, these files will not be created.  

**Expected Outcomes**  
Upon completion, the analysis directory will contain the following files (all named exactly as required):

- `rmsd.dat`, `rmsd.png`  
- `rmsf.dat`, `rmsf.png`  
- `gyration.dat`, `gyration.png`  
- `ligand_pocket_distance.csv`, `ligand_pocket_distance.png`  
- `dccm.npy` (or `.dat`), `dccm_heatmap.png`  
- `dssp.dat`, `dssp.png`  
- *(optional)* `dssp_150to200.dat`, `dssp_150to200.png` if residues 150–200 exist  

These files provide all the metrics requested for the holo simulation.  No cross‑simulation difference maps are produced at this stage because an apo counterpart is not available in the current working directory.  The Analysis Agent will log the completion of each step and any conditional actions (such as the presence check for residues 150–200) so that downstream reporting can reference the exact outputs.