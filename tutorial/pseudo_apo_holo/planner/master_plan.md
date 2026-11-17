# Multi-Simulation Master Plan

**Generated:** 2026-11-11 11:46:41
**Simulations:** 8
**Pipeline:** preprocess -> simsetup -> hpcjob -> analysis -> reporter
**Task scope:** full_pipeline
**Pre-combined (before traj):** no
**Post-combined (after all sims):** yes
**Combined analysis requested (legacy=post):** yes

## Overall Goal

Preprocess the four PDBs (p21860, q8iv63, q8nb16, q8wz42) by extracting the protein only for the apo case (case_id = protein_only, no ligand or ions) and extracting protein + ATP + Mg²⁺ for the holo case (case_id = protein_with_ligand). Prepare GROMACS input files (topology, coordinates, .mdp, .tpr) using amber99sb-ildn, TIP3P water, 310 K, 1 bar, 0.15 M NaCl, cubic box with 1.2 nm buffer, and submit all eight 1‑ns production runs to the HPC cluster. After trajectory generation, analyze each system to compute backbone RMSD, per‑residue RMSF (including a bar plot for residues 150–200 where present), radius of gyration, COM distance between ATP and the catalytic pocket (holo only), Cα DCCM and apo‑vs‑holo differences, and DSSP time evolution for the full protein and residues 150–200; then generate comparative overlay plots and statistical tables across all systems. In the final report, retrieve literature on activation‑loop conformations and allosteric regulation for ERBB3, VRK3, MLKL, and TITIN, and correlate these findings with the simulation results.

## Shared per-simulation intent

For each simulation label below, run the same pipeline (preprocess -> simsetup -> hpcjob -> analysis -> reporter) using that label's PDB / working directory and case directive. Cases in this campaign: Protein + ATP + Mg (holo), Protein only (apo).

Example per-sim wording:
label p21860, ERBB3, p21860.pdb, /home/.../p21860, apo. Follow the workflow: preprocess, simsetup, hpcjob, analysis, reporter. Compute backbone RMSD, per‑residue RMSF, radius of gyration, Cα DCCM, DSSP, and generate plots for comparison with holo. Case requirement: case_id=protein_only Protein only (apo) Preprocess and set up a 1 ns MD simulation for each PDB using AMBER99SB-ILDN, TIP3P water, 310 K, 1 bar, 0.15 M NaCl, and perform analysis of backbone RMSD, per-residue RMSF (including residues 150–200 when present), radius of gyration, Cα DCCM, and DSSP time evolution. Use protein only: exclude ligand and crystallographic ions from the source PDB.

## Simulation inventory

| # | Label | Protein | PDB | Case | Directory |
|---|-------|---------|-----|------|-----------|
| 1 | p21860 | dynamics | p21860.pdb | Protein only (apo) | `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/p21860` |
| 2 | p21860_ATP_MG | dynamics | p21860.pdb | Protein + ATP + Mg (holo) | `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/p21860_ATP_MG` |
| 3 | q8iv63 | dynamics | q8iv63.pdb | Protein only (apo) | `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8iv63` |
| 4 | q8iv63_ATP_MG | dynamics | q8iv63.pdb | Protein + ATP + Mg (holo) | `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8iv63_ATP_MG` |
| 5 | q8nb16 | dynamics | q8nb16.pdb | Protein only (apo) | `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8nb16` |
| 6 | q8nb16_ATP_MG | dynamics | q8nb16.pdb | Protein + ATP + Mg (holo) | `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8nb16_ATP_MG` |
| 7 | q8wz42 | dynamics | q8wz42.pdb | Protein only (apo) | `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8wz42` |
| 8 | q8wz42_ATP_MG | dynamics | q8wz42.pdb | Protein + ATP + Mg (holo) | `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8wz42_ATP_MG` |

## Pre-Combined Plan (before per-sim traj analysis)

_Not requested — skip pre_combined stage._

## Post-Combined Plan (after all per-sim analysis+reporter)

After all per‑simulation analyses and individual reports are finished, gather the standard output files (rmsd.dat, rmsf.dat, gyration.dat, dccm_heatmap.png, dssp.png, ligand_pocket_distance.csv) from each simulation’s analysis folder. Use the cross‑simulation analysis tools to compute overlay plots (e.g., RMSD and RMSF overlays, DCCM difference maps) and statistical tables summarizing mean, std, min, max for each metric across the four proteins and both apo/holo states. Then invoke the reporter to generate a combined HTML report that includes the overlay figures, tables, and a literature review linking the observed dynamics to activation‑loop conformations and allosteric regulation for ERBB3, VRK3, MLKL, and TITIN.