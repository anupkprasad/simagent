# Tools & Dependencies

**This file is the software catalogue plus GROMACS/analysis mechanisms.**
Use it to see which binaries and Python libraries SimAgent calls, and how
setup/analysis work around format limits (multi-chain residue map, phospho
PDB cleanup). It is not the per-tool science (RMSD formula, output CSV
names) and not the execution pools.

**In this file:** MD engines and force fields, structure prep and UniProt
acquisition, trajectory-analysis libraries, chain-residue map, LLM stack,
plotting/HTML report, HPC clients, literature APIs, run summary, test tools.

Packages are pinned loosely in `environment.yml` (conda environment `SimAgentEnv`,
Python 3.11). That env includes the **Ollama Python client**, not the Ollama
daemon or model weights — see [OLLAMA_SETUP.md](OLLAMA_SETUP.md).
Per-tool observables and output filenames:
[ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md). Parallel / HPC execution:
[POOLS.md](POOLS.md). System wiring: [ARCHITECTURE.md](ARCHITECTURE.md).
When agents run and which folders they fill:
[PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md).

---

## MD Simulation Engines

| Tool | Version | Role |
|------|---------|------|
| **GROMACS** | 2025.4 | Primary simulation engine — topology (`pdb2gmx`), solvation, ion addition, energy minimisation, equilibration, production MD |
| **AmberTools** | 24.8 | `antechamber` (ligand parameterisation), `tleap`, `pdb4amber`, `cpptraj`, `sander`, `MMPBSA.py` |

### Force Fields & Water Models

- Default force field: **AMBER99SB-ILDN** (`amber99sb-ildn`)
- Default water model: **TIP3P** (`tip3p`)
- Overridable via `--force-field` and `--water-model` CLI flags

---

## Structure Preparation

| Tool | Role |
|------|------|
| **OpenBabel** 3.1.1 | File format conversion (PDB, MOL2, SDF) |
| **RDKit** | Cheminformatics — SMILES handling, ligand processing |
| **PDB2PQR** 3.7.1 | Protonation state assignment |
| **PROPKA** 3.5.1 | pKa estimation for titratable residues (pH 7.0 default) |
| **ACPYPE** 2023.10.27 | Automated AMBER/GROMACS topology generation for ligands |
| **ParmEd** 4.3.0 | Parameter file manipulation and format conversion |
| **PackMol-MemGen** | Membrane system packing |
| **BioPython** | PDB parsing (`Bio.PDB`), structure manipulation, Entrez literature search |

---

## Structure Acquisition

These modules enable starting from a UniProt accession with no local PDB file.

| Module | Path | Role |
|--------|------|------|
| **Request parser** | `src/preprocess/structure_request_parser.py` | Extracts UniProt ID, domain label, residue ranges from natural language |
| **AlphaFold source** | `src/preprocess/structure_sources/alphafold.py` | Downloads predicted structure from AlphaFold API (default) |
| **RCSB source** | `src/preprocess/structure_sources/rcsb.py` | UniProt→experimental PDB lookup, or direct download by PDB ID |
| **Source registry** | `src/preprocess/structure_sources/registry.py` | Priority-ordered pluggable download backends |
| **UniProt domain source** | `src/preprocess/domain_sources/uniprot.py` | Queries UniProt REST API for Domain/Region feature annotations |
| **Domain registry** | `src/preprocess/domain_sources/registry.py` | `lookup_domain_range()` with offline fallback table |
| **Domain extractor** | `src/preprocess/domain_extractor.py` | Trims PDB to a residue span (BioPython + MDAnalysis) |
| **Acquire structure tool** | `src/preprocess/acquire_structure_tool.py` | End-to-end: download → domain trim → validate |
| **Domain lookup tool** | `src/preprocess/domain_lookup_tool.py` | `lookup_domain_range_tool` (LangChain `@tool`) for the Preprocessing agent |
| **Structure validator** | `src/preprocess/structure_validator.py` | Missing residue check, broken backbone (Cα–Cα > 4.5 Å), component availability |
| **Structure downloader** | `src/preprocess/structure_downloader.py` | `download_structure` (UniProt) + `download_pdb_by_id` (RCSB ID) |
| **Sequence downloader** | `src/preprocess/sequence_downloader.py` | `download_protein_sequence` UniProt/RCSB FASTA (not required for MD) |

### External APIs used for structure acquisition

| API | Purpose | Documentation |
|-----|---------|---------------|
| AlphaFold EBI API | Fetch predicted PDB by UniProt accession | `https://alphafold.ebi.ac.uk/api/prediction/{accession}` |
| RCSB PDB REST API | Fetch experimental PDB by entry ID | `https://files.rcsb.org/download/{id}.pdb` |
| UniProt REST API | Domain / Region feature annotations | `https://rest.uniprot.org/uniprotkb/{accession}` |
| UniProt FASTA | Canonical sequence download | `https://rest.uniprot.org/uniprotkb/{accession}.fasta` |
| RCSB FASTA | Polymer sequence(s) for a PDB entry | `https://www.rcsb.org/fasta/entry/{id}/download` |

**Note:** UniProt does not host independent 3D coordinate files for MD. Structure-by-UniProt uses AlphaFold DB and/or RCSB; sequence FASTA is optional and does not replace a structure.

---

## Trajectory Analysis

> **Full tool reference (calculations, theory, outputs):** [ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md).
> Multi-chain `chainID` / resid selections: [below](#multi-chain-residue-map).

Primary library: **MDAnalysis v2.10.0**

### Per-simulation tools (`agentic/analysis/tools.py`)

| Tool | Observable |
|------|------------|
| `calculate_rmsd` | Backbone Cα RMSD relative to reference |
| `calculate_rmsf` | Per-residue root-mean-square fluctuation |
| `calculate_radius_of_gyration` | Global compactness (Rg) |
| `calculate_sasa` | Solvent-accessible surface area |
| `analyze_energy` | Potential energy and temperature from `.edr` |
| `analyze_secondary_structure` | DSSP time evolution (whole + user-defined segment) |
| `calculate_com_distance` | Centre-of-mass distance between two atom groups |
| `calculate_ligand_pocket_distance` | ATP–pocket distance (pocket = atoms within 5 Å of ligand at frame 0) |
| `calculate_dccm` | Dynamic cross-correlation matrix of Cα fluctuations |
| `calculate_trajectory_pca` | PCA on aligned coordinates; writes ``pca_projections.dat`` |
| `plot_pca_projection` | PC1 vs PC2 (or PCx/PCy) scatter coloured by time |
| `calculate_free_energy_landscape` | F = −kT ln P from PC1/PC2 histogram (kJ/mol contour map) |
| `analyze_fel_landscape_features` | FEL classification metrics: minima, basin depth/area, barriers, entropy |
| `calculate_protein_ligand_contacts` | Protein–ligand H-bond and heavy-atom contact counts per frame |
| `calculate_pocket_sasa` | SASA of binding-pocket residue subset (requires `.tpr`) |
| `analyze_ligand_residence` | Bound/unbound residence times and unbinding event counts |
| `calculate_pocket_rmsf` | Per-residue RMSF for pocket Cα atoms |
| `calculate_ligand_rmsf` | Per-atom RMSF of ligand (ATP) heavy atoms |
| `plot_dccm_difference` | Element-wise DCCM difference between two simulations (apo–holo) |
| `plot_md_data` | Single-trace time-series plot |
| `plot_md_multipanel` | Multi-panel figure |
| `plot_combined_data` | Overlay plot for multiple datasets |
| `wrap_trajectory` | PBC wrap centering Protein+ligand (default ATP); skips if `mdWrap.xtc` exists |
| `extract_trajectory_metrics` | Bulk metric extraction to CSV |

### Combined (multi-simulation) tools

Exposed only in **combined** phases (`pre_combined`, `post_combined` /
`combined_analysis`) or HITL combined view — not during per-sim traj analysis.
**Pre** stage: prefer MSA / consensus pocket tools; outputs harvested into
`{base}/cross_sim/`. **Post** stage: overlays, collectors, LLM feature
selection, Ward / dendrogram+heatmap. Timing details:
[ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md#two-tool-domains-per-sim-vs-combined).

| Tool | Function |
|------|----------|
| `collect_metric_files` | Gather matching metric files across simulation directories |
| `plot_combined_overlay` | Multi-trace overlay plot (RMSD, RMSF, Rg, energy) with legend |
| `compute_comparison_table` | Statistical summary JSON (mean, std, min, max per metric per simulation) |
| `run_combined_analysis` | Full combined pipeline entry point |
| `run_combined_dccm_analysis` | DCCM panels for all simulations |
| `run_combined_dccm_difference` | Cross-simulation DCCM difference heatmap |
| `run_combined_rmsf_segment_analysis` | RMSF bar chart for user-defined residue window |
| `run_combined_com_distance_analysis` | COM distance overlay across simulations |
| `collect_fel_features_table` | Aggregate `fel_features.json` from all sims into one classification CSV |
| `collect_classification_features_table` | Full binding + modular + pocket feature matrix (raw + robust z-score) |
| `cluster_classification_features` | Hierarchical / k-means; dendrogram+heatmap panel |
| `build_consensus_sequence_alignment` | Star MSA to reference; consensus residue map for cross-sim PCA |
| `define_reference_consensus_pocket` / `run_consensus_pocket_metrics_batch` | Reference pocket + mapped COM/orientation metrics |
| `fit_reference_pca_model` | Reference PCA on consensus Cα |
| `project_simulations_reference_pca` | Project trajectories onto reference PCA |
| `build_shared_reference_fel_landscapes` | Shared-grid FEL from reference-projected PCA |
| `cluster_reference_fel_landscapes` | Cluster shared-reference FEL features |
| `run_reference_landscape_pipeline` | End-to-end reference landscape workflow |
| `plot_combined_rmsf_segment_bars` | Stacked segment RMSF bar chart |

**Per-sim modular family tools** (also exposed during traj analysis when the
goal requests them): `calculate_consensus_pocket_metrics` (COM + axis-angle;
registered on the **per-sim** executor), `calculate_consensus_torsions`,
`calculate_consensus_rmsf_features`, `calculate_consensus_dccm_features`,
`run_independent_dynamics_fel`. Batch helper
`run_consensus_pocket_metrics_batch` is combined/shared only. See
[ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md#modular-family-dynamics-torsions--pca--tica)
and [CAMPAIGN_AND_RETRIEVAL.md](CAMPAIGN_AND_RETRIEVAL.md).

### Additional analysis libraries

| Tool | Version | Role |
|------|---------|------|
| **MDAnalysis** | ≥2.9 | Trajectory I/O, atom selection, RMSD/RMSF/DCCM/PCA calculations |
| **NumPy / SciPy / pandas** | — | Arrays, PCA/clustering math, feature tables |
| **PyTraj** | 2.0.6 | AMBER cpptraj wrapper for trajectory analysis |
| **scikit-learn** | 1.6.1 | Cross-simulation PCA / hierarchical clustering |
| **Matplotlib** | 3.10.8 | Publication-quality 2D plots |
| **Seaborn** | — | Statistical plot styling |

The Programmer agent also reads
[`AVAILABLE_SOFTWARE.md`](../agentic/programmer/AVAILABLE_SOFTWARE.md) when
generating new tools (GROMACS, MDAnalysis, MAFFT, etc.).

---

## Multi-chain residue map
GROMACS production files do not keep PDB chain IDs. Users and the Analysis
agent can still say “chain B resid 50–75” after a multi-chain complex
simulation: setup writes a JSON map, and analysis translates PDB-style
selections to trajectory `resindex` before MDAnalysis `select_atoms`.

### The problem

| File | What it stores |
|------|----------------|
| Input PDB (`protein_h.pdb`) | Chain ID + PDB residue number (user language) |
| `.gro` | Residue number/name only — **no chain ID** |
| `md.tpr` | Topology/masses; chain IDs blank or unused |
| `.xtc` | Coordinates only |

Setup runs `pdb2gmx -chainsep id_or_ter -merge all`. Chains are concatenated
into one molecule. If chain A and chain B both start at residue 1, the
trajectory has two copies of `resid 1`. Then:

- `chainID B and resid 50:75` matches **0 atoms** on `md.tpr` + `mdWrap.xtc`
- `resid 50:75` without a chain can pull residues from **both** chains

This applies to two, three, or four (or more) chains the same way.

### Design

Keep the **user PDB as the naming source**. Translate to unique trajectory
**residue indices** (`resindex`) before `select_atoms`.

```
User / LLM:  "chainID B and resid 50:75"
                 │
                 ▼
        chain_residue_map.json
        (PDB chain + PDB resid → traj resindex)
                 │
                 ▼
Trajectory:  "resindex 389:414"
                 │
                 ▼
     existing RMSD / RMSF / COM / DSSP / …
```

Do **not** rewrite every analysis tool. One translator sits in front of
selections. Selections without a chain token (`protein`, `resname ATP`)
pass through unchanged.

Do **not** use original-PDB atom serial numbers on the trajectory.
`pdb2gmx -ignh` adds hydrogens and can rename atoms. The map is
**residue-level**.

### Pipeline

```
preprocess/protein_h.pdb          (has chain IDs + TER)
        │
        ▼
simsetup: pdb2gmx → protein_processed.gro
        │
        ▼
build_and_save_chain_residue_map()     ← after topology, non-fatal
        │
        ▼
simsetup/chain_residue_map.json
        │
        ▼
HPC: md.tpr + mdWrap.xtc               (still no chain IDs)
        │
        ▼
analysis: translate_selection() before select_atoms
```

| Stage | What happens |
|-------|----------------|
| **Setup** | After `pdb2gmx`, `ComplexSystemBuilder` maps the input PDB onto `protein_processed.gro` and writes `simsetup/chain_residue_map.json`. Failure is logged; setup continues. |
| **State** | Path stored in `MDState["chain_residue_map"]` and `file_registry`. Cleared between multi-sim cases like `topology`. |
| **Analysis** | `TrajectorySession`, `trajectory_compute`, and `AnalysisToolExecutor` translate `selection` / `selection1` / `selection2` / `align_selection` / `protein_selection` when they contain `chainID` or a resid range. |
| **Analysis-only (old runs)** | If the JSON is missing, `ensure_chain_residue_map()` rebuilds it from `preprocess/protein_h.pdb` (or `protein.pdb`) plus `protein_processed.gro` or `md.tpr`. |

GROMACS CLI tools (`calculate_sasa`, `calculate_pocket_sasa`, `wrap_trajectory`)
are **not** translated — `resindex` is not a `gmx` index group.

### Map file

**Path:** `{sim}/simsetup/chain_residue_map.json`

```json
{
  "version": 1,
  "source_pdb": "/path/to/preprocess/protein_h.pdb",
  "trajectory_topology": "/path/to/simsetup/protein_processed.gro",
  "n_chains": 2,
  "n_mapped_residues": 500,
  "chain_order": ["A", "B"],
  "warnings": [],
  "chains": {
    "A": {
      "pdb_resids":      [  1,  2,  3],
      "pdb_resnames":    ["MET", "ALA", "HIS"],
      "traj_resindices": [  0,  1,  2],
      "traj_resids":     [  1,  2,  3]
    },
    "B": {
      "pdb_resids":      [  1, 50, 75],
      "pdb_resnames":    ["GLY", "LYS", "PHE"],
      "traj_resindices": [340, 389, 414],
      "traj_resids":     [  1, 50, 75]
    }
  }
}
```

`traj_resindices` are 0-based MDAnalysis residue indices. They stay valid on
`md.tpr` because protein residues remain first after ligand/solvent merge.
Each residue array is written on **one line**, with integers padded so the
i-th `pdb_resid`, `traj_resindex`, and `traj_resid` line up in a column.

Matching:

1. Walk PDB protein residues in file order (chains as they appear; after
   preprocess reorder this is typically A, B, C, …).
2. Walk protein residues in `protein_processed.gro` / `md.tpr` in the same
   concatenated order (`pdb2gmx -merge all`).
3. If counts match, pair 1:1. Resname differences use tautomer/phospho
   aliases (`HIS`/`HID`/`HIE`, `SEP`/`SP2`, `TPO`/`THP2`, `PTR`/`TP2`).
4. If counts differ, Needleman–Wunsch on one-letter codes; unmapped
   residues are omitted and a warning is stored.

Three or four chains are additional keys in `chains` — same algorithm.

### Selection translation

Supported forms (MDAnalysis-style, as the LLM is told to write):

```
chainID B and resid 50:75
chain B and resid 50 to 75
protein and chainID A and resid 1:10 and name CA
chainID C
chainID A or chainID B
protein and (chainID A or chainID D) and name CA
```

| Input | Result |
|-------|--------|
| `chainID B and resid 50:75` | `resindex 389:414` (contiguous run) |
| `chainID A and resid 1:10 and name CA` | `name CA and resindex 0:9` |
| `protein` / `resname ATP` | unchanged |
| `resid 50:75` on a **multi-chain** map | unchanged + warning (ambiguous) |
| `resid 50:75` on a **single-chain** map | mapped through that one chain |
| Unknown chain | tool error listing available chains |
| No map + `chainID B` | unchanged + warning (0 atoms on TPR) |

COM example (DCLK3–PSMA4 style):

```
calculate_com_distance
  selection1: "chainID B and resid 50:75"
  selection2: "chainID A and resid 0:10"
```

The translator rewrites both strings; the COM calculator is unchanged.

Always include `chainID` when residue numbers overlap across chains.
PDB resid `0` is valid if that number exists in the source PDB.

### Code map

| Module | Role |
|--------|------|
| `src/analysis/chain_residue_map.py` | Build, load, find, translate |
| `src/simsetup/system_builder.py` | Write JSON after `pdb2gmx` |
| `src/analysis/trajectory_session.py` | Lazy-load map; translate align selections |
| `src/analysis/trajectory_compute.py` | Translate batched metric params |
| `agentic/analysis/tools.py` | Translate standalone tool kwargs |
| `agentic/simsetup/setup_agent.py` | Register file; set `state["chain_residue_map"]` |

Public helpers in `chain_residue_map.py`:

- `build_chain_residue_map(pdb, topology)` / `build_and_save_chain_residue_map(...)`
- `ensure_chain_residue_map(working_dir=..., topology_file=...)`
- `translate_selection(selection, map)` / `translate_selection_params(params, map)`
- `lookup_resindices(map, chain_id, resid_start=..., resid_end=...)`
- `CHAIN_SELECTION_LLM_NOTE` — inserted into Analysis agent prompts

### Phosphorylation and chain order in the PDB (not this map)

Preprocess has a **different** chain-aware step so phosphorylated residues
stay on the protein for `pdb2gmx`. It does **not** put chain IDs back on
the trajectory.

MDAnalysis `protein` does not include SEP/TPO/PTR. Without extra handling
they are written at the end of `protein.pdb` or classified as type `Other`
(`residue THP173 is of type 'Other'`).

What preprocess/setup already do:

1. **Keep phospho on the protein** — `complex_separator.py` concatenates
   `protein` + phospho atoms, then sorts by `(chainID, resid)`.
2. **`reorder_pdb_by_resid` + `insert_ter_records`** (`pdb_utils.py`) —
   sort ATOM lines by chain then resid; insert `TER` at chain boundaries.
   MDAnalysis drops `TER`; `pdb2gmx` needs TER + chain ID for C-termini (OXT).
3. **Rename for the force field** — `normalize_phosphorylation_for_gromacs`:
   SEP/TPO/PTR → SP2/THP2/TP2 (CHARMM dianionic default), O3P→OT, drop H3T.
4. **`residuetypes.dat`** — SP2/THP2/… listed as `Protein`, not `Other`.
5. **`pdb2gmx -chainsep id_or_ter -merge all`** — split by chain/TER, then
   merge into one molecule (chain identity is then lost in GRO/TPR).

The residue map is built **after** that cleanup, from the same PDB
(`protein_h.pdb` / `cleaned_pdb`) plus `protein_processed.gro`.

| | Phospho / TER cleanup | Analysis chain map |
|--|----------------------|--------------------|
| When | Preprocess + setup | After `pdb2gmx` / at analysis |
| Purpose | Keep SEP/TPO in the right chain for `pdb2gmx` | User says “chain B resid 50–75” on the XTC |
| Output | Ordered PDB with TER | `chain_residue_map.json` |
| Survives in `md.tpr`? | No | The JSON is the surviving record |

### What not to do

- Do not recover chain IDs from `md.tpr` alone.
- Do not apply original-PDB atom indices to the trajectory.
- Do not drop `-merge all` solely to keep chains as separate molecule types
  (GRO still has no chain ID; overlapping resids remain).
- Do not offset-renumber residues in the trajectory as the primary fix —
  users still speak PDB numbering, so you still need this map.

---

## AI / LLM Framework

| Tool | Version | Role |
|------|---------|------|
| **LangGraph** | 1.0.7 | StateGraph-based workflow orchestration — nodes, edges, conditional routing |
| **LangChain** | 1.2.8 | Chain and prompt utilities; `@tool` decorator for agent tools |
| **Ollama server** | (install separately) | Local LLM daemon on `:11434`; pull `gpt-oss:20b` — [OLLAMA_SETUP.md](OLLAMA_SETUP.md) |
| **ollama (Python)** | ≥0.4 | Client library in conda env; talks to the server |
| **LangSmith** | — | Optional tracing / observability |

### LLM integration pattern

```
User Goal
  │
  v
Supervisor: structured_prompt (single enrichment LLM call)
  │
  v
Planner: retrieved tools + retrieved knowledge chunks + goal ──► Ollama /api/generate
                                              │
                                              v
                                      Execution plan
                                   (per-agent instructions)
  │
  v
Field agents: each reads its *_instructions field
              makes targeted LLM calls for tool selection
  │
  v
Reporter: literature retrieval + LLM narrative
```

Fallback: when LLM is unreachable, heuristic routing and rule-based report synthesis activate automatically.

---

## Visualisation & Reporting

| Tool | Role |
|------|------|
| **Matplotlib** 3.10.8 | RMSD, RMSF, Rg, energy, DCCM, COM distance plots |
| **Seaborn** | Statistical styling for overlay and comparison plots |
| **3Dmol.js** (CDN) | Interactive 3D molecular viewer embedded in HTML report |

### HTML Report Features

- Task description from enriched / master prompt
- Embedded analysis figures and combined overlay plots
- DCCM heatmaps and apo–holo DCCM difference panel
- LLM-generated scientific narrative (rule-based fallback)
- Literature references with PubMed / bioRxiv / UniProt source badges
- 3D structure viewer: representations, colour schemes, surface, sequence bar,
  trajectory snapshots at diagnostic frames, PNG export

---

## HPC & Remote Execution

Job queues, polling, and local workers: [POOLS.md](POOLS.md). This section
is the client libraries and script-name rule only.

| Tool | Version | Role |
|------|---------|------|
| **SLURM** | — | HPC job scheduler |
| **Paramiko** | 4.0.0 | SSH/SFTP for remote HPC execution |
| **Singularity** | — | Container runtime for reproducible environments on HPC |

### HPC script naming convention

```
{job_name}_run.sh   ← correct (derived from simulation label)
slurm_job.sh        ← incorrect (LLM-hallucinated; ignored by HPC agent)
```

---

## Literature Search APIs

| Source | API | Purpose |
|--------|-----|---------|
| **PubMed** | NCBI E-utilities (BioPython Entrez) | Peer-reviewed MD/protein literature |
| **bioRxiv** | Europe PMC REST (`SRC:PPR` filter) | Preprints |
| **UniProt** | UniProt REST API | Reviewed protein function, domain annotations |

Results are deduplicated by PMID and DOI, capped at 15 references per report.

---

## Run Summary Utility

| Module | Path | Role |
|--------|------|------|
| `run_summary.py` | `src/utils/run_summary.py` | Build + write `run_summary.md` and `run_summary.json` at workflow exit |

Functions:
- `build_run_summary(final_state, working_dir, goal, pdb_list, config)` → summary dict
- `write_run_summary(working_dir, summary)` → writes JSON + Markdown
- `format_run_summary_terminal(summary)` → terminal-printable block

---

## Development utilities

| Tool | Role |
|------|------|
| **invoke** | Task runner |
| **PyYAML** | Configuration file parsing |
| **tqdm** | Progress bars |
