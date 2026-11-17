# SimAgent Workflow Run Summary

**Generated:** 2026-11-17T15:38:01
**Mode:** multi_simulation
**Working directory:** `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo`

## Goal

I want to study the effect of ATP binding on protein dynamics for these four
PDBs — p21860.pdb, q8iv63.pdb, q8nb16.pdb, and q8wz42.pdb — which are available
in the current working directory. Each PDB has protein + ATP + Mg.

Please preprocess and set up MD simulations for 1 ns for all PDBs with two
component cases per structure:
  1. Protein only (apo)
  2. Protein + ATP + Mg (holo)
for a total of eight simulations. Once setups are done, submit the jobs to HPC.

The proteins are human pseudokinases (UniProt id : name):
  p21860: ERBB3, q8iv63: VRK3, q8nb16: MLKL, q8wz42: TITIN.

Force field AMBER99SB-ILDN, TIP3P water, 310 K, 1 bar, 0.15 M NaCl.

For each system compute:
  (1) backbone RMSD over time,
  (2) per-residue RMSF (and an RMSF bar plot near the active-site region,
      residues 150–200 when present),
  (3) radius of gyration,
  (4) COM distance between bound ATP and the catalytic pocket (pocket =
      protein atoms within 5 Å of ATP at frame 0) for holo systems,
  (5) Cα DCCM, including apo vs holo DCCM differences where both cases exist,
  (6) DSSP time evolution for the whole protein and the active-site region
      (residues 150–200 when present).

After per-simulation analysis, generate comparative overlay plots and
statistical tables across all systems. In the reporter, retrieve relevant
literature for each named protein focusing on activation-loop conformations,
allosteric regulation, and MD or experimental dynamics, and correlate the
simulation findings with that literature in the final report.

## Counts

| Metric | Value |
|--------|-------|
| Total simulations | 8 |
| Source PDBs | 4 |
| Succeeded | 8 |
| Healthy | 8 |
| Skipped | 0 |
| Failed | 0 |

## Simulations

### p21860_ATP_MG — success
- **Health:** healthy
- **Directory:** `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/p21860_ATP_MG`

### q8iv63 — success
- **Health:** healthy
- **Directory:** `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8iv63`

### q8iv63_ATP_MG — success
- **Health:** healthy
- **Directory:** `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8iv63_ATP_MG`

### q8nb16 — success
- **Health:** healthy
- **Directory:** `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8nb16`

### q8nb16_ATP_MG — success
- **Health:** healthy
- **Directory:** `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8nb16_ATP_MG`

### q8wz42 — success
- **Health:** healthy
- **Directory:** `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8wz42`

### q8wz42_ATP_MG — success
- **Health:** healthy
- **Directory:** `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8wz42_ATP_MG`

### p21860 — success
- **Health:** healthy
- **Directory:** `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/p21860`

## Artifacts

- Log: `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/agent_conversation.log`
- Execution report: `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/supervisor/execution_report.md`
- Reporter output: `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/reporter/combined_report.html`

## LLM usage

Token ledger is kept only in `/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/llm_usage.json` (not duplicated here).