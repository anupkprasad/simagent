# Reporter Execution Plan

**Generated:** 1794620123.3972933

## LLM Reasoning

The combined report must aggregate the analysis results and visualizations from all eight simulations (apo and holo for each of the four pseudokinases). The only required tool for this step is `generate_combined_html_report`. We will supply the exact simulation directories and labels as requested, include all overlay plot images that were produced by the analysis pipeline, and set the working directory to the reporter output directory (via `working_dir='.'`). No additional literature searching is necessary because the report will automatically harvest references from the individual simulation reporter artefacts. The report will focus on ATP‑binding effects, pseudokinase dynamics, and structural comparisons across the four proteins.

## Overview

A single HTML page that overlays RMSD, RMSF, radius of gyration, COM distance, and DSSP plots for apo versus holo states, compares all four pseudokinases, and contextualizes the findings with literature on activation-loop conformations and allosteric regulation.

## Report Focus

- ATP binding effects on protein dynamics
- pseudokinase structural comparison
- activation-loop conformations

## Execution Steps (1 steps)

### Step 1: Generate combined HTML report

**Tool:** `generate_combined_html_report`

**Description:** Creates a self‑contained HTML report that embeds all overlay plots, pulls per‑simulation analysis summaries, and incorporates literature references.

**Reason:** The generate_combined_html_report tool will read the analysis_summary.jsonl files in each simulation directory, embed the overlay plots, and compile the final report.

**Parameters:**
```json
{
  "sim_dirs": [
    "/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/p21860",
    "/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/p21860_ATP_MG",
    "/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8iv63",
    "/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8iv63_ATP_MG",
    "/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8nb16",
    "/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8nb16_ATP_MG",
    "/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8wz42",
    "/home/akp66103/workspace/simagent/tutorial/pseudo_apo_holo/q8wz42_ATP_MG"
  ],
  "labels": [
    "ERBB3",
    "ERBB3_ATP_MG",
    "VRK3",
    "VRK3_ATP_MG",
    "MLKL",
    "MLKL_ATP_MG",
    "TITIN",
    "TITIN_ATP_MG"
  ],
  "overlay_plots": [
    "com_distance_overlay.png",
    "rg_overlay.png",
    "rmsd_overlay.png",
    "rmsf_overlay.png"
  ],
  "working_dir": ".",
  "output_file": "combined_report.html",
  "title": "Human Pseudokinase ATP Binding Dynamics Comparison",
  "enriched_prompt": "User goal: Study the effect of ATP binding on protein dynamics for human pseudokinases.",
  "user_goal": "I want to study the effect of ATP binding on protein dynamics for these four PDBs ...",
  "protein_name": "Human pseudokinases"
}
```

