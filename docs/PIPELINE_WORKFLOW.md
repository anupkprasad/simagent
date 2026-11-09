# Pipeline workflow

**This file is the run walkthrough.** It shows what happens after you type a
goal: which agents run and in what order, which directories and files they
write, and how CLI flags change that path. Use it as a map of a live job.

**In this file:** agent order, campaign directory tree (`{base}/{label}/`),
per-stage files, flag effects, worked examples (`--goal` + bash).

**Not in this file:** LangGraph internals ([ARCHITECTURE.md](ARCHITECTURE.md)),
pool sizing ([POOLS.md](POOLS.md)), analysis theory
([ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md)), dependency versions
([TOOLS.md](TOOLS.md)), product overview ([PROJECT.md](PROJECT.md)).

Entry point: `python SimAgent.py --goal "..." --working-dir <dir>`.

For a published end-to-end demo (status files, reports, provenance), see
[tutorial/TUTORIAL.md](../tutorial/TUTORIAL.md).

---

## Agent order (always the same)

Field agents never skip ahead. Flags only **omit** stages or **wait** between
them. The logical order is:

```
CLI + input validation
        │
        ▼
   Supervisor          enrich goal, build sim_prompts, route
        │
        ▼
   Planner             execution plan + per-agent instructions
        │
        ▼
   Preprocess          structure → cleaned PDB(s)
        │
        ▼
   SimSetup            topology, box, ions, MDP, chain map
        │
        ▼
   HPC                 SLURM script + mdrun  (pool waits for .xtc)
        │
        ▼
   Analysis            observables, plots, analysis_summary.jsonl
        │
        ▼
   Reporter            report.html  (+ combined_report.html if N>1)
        │
        ▼
   run_summary.md / .json   (always, at --working-dir)
```

Optional **`--HITL all`** pauses after preprocess, setup, HPC, analysis, and
reporter. **`--HITL error`** pauses only on failures. Default is fully
automatic.

`--subtask` lists which field agents to run. Omit it for the full pipeline.

| `--subtask` | What runs | Typical use |
|-------------|-----------|-------------|
| *(omit)* | preprocess → simsetup → hpcjob → analysis → reporter | New simulation through report |
| `preprocess simsetup hpcjob` | Stop after SLURM submit | Launch MD; analyse later |
| `analysis reporter` | Need existing `hpc/*.xtc` | Analyse finished trajectories |
| `hpcjob` | Continue / extend MD | Longer production on existing systems |
| `preprocess` | Structure only | Inspect cleaned PDB before setup |

Planner and Supervisor always run. They are not listed in `--subtask`.

---

## Where files go

`--working-dir` is the **campaign base**. Every run always uses the
`{base}/{label}/` tree (even for one PDB). The base holds campaign provenance
only; each simulation’s agent work lives under its label directory.

**Base (`--working-dir`):**
- `agent_conversation.log` — input validation, enrichment, master plan summary, pool milestones
- `planner/master_plan.md` (+ `.json`) — short campaign summary
- `run_summary.md` / `run_summary.json`
- `supervisor/pool_status.json` — **live** per-sim ladder + phase (check this first while a run is active)
- `supervisor/state.jsonl` — pretty-printed resume checkpoint (`multi_sim_progress`, pools, job IDs)
- `campaign/state.json` — versioned campaign snapshot
- Combined `analysis/` + `reporter/combined_report.html` **only when** `len(sim_prompts) > 1`
- Optional `cross_sim/` — pre-combined pocket/MSA/consensus artifacts + `inventory.json`
- Optional `campaign.yaml` — local override of shipped mapping/retrieval knobs
- `planner/knowledge_index.json` — knowledge cache generated for that run

**Per simulation (`{base}/{label}/`):**
- `agent_conversation.log` — per-sim validation, execution plan, preprocess → reporter
- `planner/execution_plan.md` (+ `.json`)
- `preprocess/`, `simsetup/`, `hpc/`, `analysis/`, `reporter/` (each writes `inventory.json`)
- `state.json` — per-sim snapshot (completed stages, bound traj)

```
--working-dir ./kinase_study/
├── jak2_atp_2mg/            # N=1: label = PDB stem (no combined stage)
│   ├── preprocess/
│   ├── simsetup/
│   ├── hpc/
│   ├── analysis/
│   ├── reporter/report.html
│   ├── planner/             # per-sim execution_plan
│   ├── supervisor/
│   ├── state.json
│   └── agent_conversation.log
├── p21860/                  # apo (protein-only)
│   └── …
├── p21860_ATP_MG/           # holo (skipped if ATP/MG absent in the PDB)
│   └── …
├── cross_sim/               # optional pre_combined artifacts (pocket_map.json, MSA)
├── analysis/                # post_combined overlays (only if N>1)
├── reporter/
│   └── combined_report.html
├── planner/
│   ├── master_plan.md       # campaign summary (incl. pre/post plans)
│   └── master_plan.json
├── campaign/
│   └── state.json
├── supervisor/
│   ├── state.jsonl
│   ├── pool_status.json
│   └── execution_report.md
├── agent_conversation.log   # campaign-level only
├── run_summary.md           # includes science completeness for family runs
├── run_summary.json
└── llm_usage.json           # tokens tagged by workflow node
```

Labels come from PDB basenames and component cases (e.g. `p21860` vs
`p21860_ATP_MG`). The Supervisor copies each input PDB into that sim’s
directory before work starts. Combined analysis/reporter runs only when the
master plan yields more than one `sim_prompts` entry (e.g. apo+holo from one
PDB still gets combined). Stage order when `n_sims > 1`:

1. Optional **`pre_combined`** (pocket/MSA/consensus → `{base}/cross_sim/`)
2. Per-sim traj analysis → reporter (multi-rep fan-out → `analysis/avg/`;
   family shared analysis protocol includes per-sim pocket metrics)
3. Optional **`post_combined`** / legacy `combined_analysis` (overlays,
   optional consensus-pocket batch, classification collect → LLM feature
   selection → Ward dendrogram+heatmap, …)
4. **`combined_reporter`** when post ran (family HTML includes a science
   completeness banner)

Combined stages use the same analysis/reporter agents as per-sim, with **LLM
planning** and combined tool metadata exposed (deterministic pipelines remain
as fallbacks). Family science gates:
[CAMPAIGN_AND_RETRIEVAL.md](CAMPAIGN_AND_RETRIEVAL.md).
`--HITL` (any mode) forces **sequential** per-sim execution. Without HITL,
prep and post-HPC analysis/reporter use the local worker pool; production MD
uses the SLURM pool. Details: [POOLS.md](POOLS.md).

---

## Files each stage creates

Names below are the usual ones. The planner may add extras from the goal
(e.g. DCCM, FEL). Analysis basenames stay **unprefixed** (`rmsf.dat`, not
`p21860_rmsf.dat`) so combined tools can collect them.

### Preprocess → `{sim}/preprocess/`

| File | Role |
|------|------|
| `protein.pdb` | Protein after component split / phospho keep |
| `protein_h.pdb` | Protonated protein (PDB2PQR / PROPKA) |
| `ligand.pdb` / `ATP_h.pdb` | Ligand if present (alias map: `ligand.pdb` → real name) |
| Domain-trimmed PDB | When the goal names a UniProt domain |
| `inventory.json` | Absolute paths to the files above (HITL `pwd` / `read_inventory`) |

If the goal is a UniProt accession with no local PDB, preprocess downloads
AlphaFold (RCSB fallback), then trims the domain. See
[ARCHITECTURE.md](ARCHITECTURE.md#structure-acquisition-pipeline).

### SimSetup → `{sim}/simsetup/`

| File | Role |
|------|------|
| `protein_processed.gro` | After `pdb2gmx` |
| `topol.top` (+ `.itp`) | Topology |
| `*ions.gro` | Solvated + neutralized system |
| `*.mdp` | EM / NVT / NPT / production |
| `chain_residue_map.json` | PDB `chainID`+`resid` → trajectory `resindex` ([TOOLS.md](TOOLS.md#multi-chain-residue-map)) |
| `inventory.json` | Stage artifact index |

`--force-field` and `--water-model` change the topology and `.mdp` content,
not the folder names.

### HPC → `{sim}/hpc/`

| File | Role |
|------|------|
| `{job_name}_run.sh` | SLURM script (`job_name` from the sim label) |
| `md.tpr` | Production input |
| `md.xtc` / `mdWrap.xtc` | Trajectory (wrap often created at analysis) |
| `md.edr`, `md.log` | Energy / log |
| `inventory.json` | Bound `md.tpr` / `mdWrap.xtc` paths |

**Shared analysis timescale:** production MDP defaults to
`nstxout-compressed = 50000` at `dt = 0.002` ps → **100 ps / frame
(10 frames / ns)**. PBC wrap (`mdWrap.xtc`) uses the same interval
(`wrap_dt_ps=100`) so every simulation is compared on one time grid.

Analysis reads trajectories from `{sim}/hpc/` on the shared filesystem.
`download_results` is not part of the default plan.

If `{sim}/hpc/*.xtc` already exists, the HPC pool **skips sbatch** for that
sim. `--reuse-hpc` still runs preprocess + simsetup + staging + analysis,
but never calls `sbatch` (keeps existing `md.tpr` / `mdWrap.xtc`).

### Analysis → `{sim}/analysis/` (and `{base}/analysis/` when combined)

| File | Role |
|------|------|
| `analysis_summary.jsonl` | Per-sim completion marker (needed for `--combined-only` / resume skip) |
| `rmsd.png`, `rmsf.dat`, `gyration.dat`, … | Goal-selected observables |
| `dccm_comparison.png` (or `_1.png`, `_2.png`, …) | Combined DCCM: ≤9 panels in a 3×3 grid; more panels → extra pages |
| `{base}/analysis/statistical_summary.json` | Cross-sim table |
| `{base}/analysis/classification_features.csv` | When the goal asks for classification |
| `{base}/analysis/classification_features_zscore.csv` | Robust/IQR z-scores for clustering |
| `{base}/analysis/classification_feature_selection.json` | LLM-chosen columns + reasoning |
| `{base}/analysis/classification_dendrogram_heatmap.png` | Dendrogram + feature heatmap panel |
| `{label}/analysis/avg/` | Multi-rep mean±std products (`--rep-num N`) |
| `{label}/analysis/consensus_*/` | Modular family dynamics outputs |
| `{label}/analysis/inventory.json` | Stage index; `repXX/inventory.json` binds each replica traj |
| `{label}/analysis/ligand_pocket_distance.csv` | Collector alias from consensus pocket metrics (COM) |
| `{label}/analysis/pocket_axis_angle.csv` | Collector alias (axis-angle) |

Full filename list: [ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md#standard-output-filenames-multi-sim).

### Reporter

| Path | Role |
|------|------|
| `{sim}/reporter/report.html` | Always this name (never `kinase_report.html`) |
| `{base}/reporter/combined_report.html` | Multi-sim comparison only |

---

## How flags change the path

| Flag | Effect on agents | Effect on directories / files |
|------|------------------|-------------------------------|
| `--working-dir DIR` | Campaign base | Always `{DIR}/{label}/` for each sim; campaign files at `DIR/` |
| `--pdb-list a.pdb b.pdb` | One (or more) cases per PDB | Labels from basenames; PDBs copied into each `{label}/` |
| `--sim-dirs d1 d2` | Analysis of existing runs | Uses those dirs as `{label}`; each must have `hpc/` |
| `--subtask …` | Runs only listed field agents | Only those agent folders get new files |
| omit `--subtask` (full) | Prep → HPC pool wait → analysis → reporter | Full tree; analysis waits for `.xtc` |
| `--HITL all` | Pause after each stage | Sequential; no local worker pool |
| `--HITL error` | Pause on failures / SLURM errors | Sequential; HPC pool HITL at `human_hpc_pool_check` |
| `--campaign-yaml PATH` | Mapping / retrieval / gold columns / HITL defaults | Does not change folder layout; writes resolved settings into `campaign_spec` |
| `--resume` | Skip sims already marked done | Same `DIR`; restores `supervisor/state.jsonl` |
| `--retry-labels L1 L2` | Re-run those labels even if they succeeded | Overwrites those `{label}/` stages that re-run |
| `--combined-only` | Skip per-sim analysis/reporter | Writes `{base}/analysis/` and `combined_report.html` only when N>1 |
| `--allowed-hpc-jobs N` | Max concurrent `sbatch` | Same files; fewer/more jobs in flight |
| `--rep-num N` | N production replicates per label | Nested `hpc/repXX`, `analysis/repXX`, `analysis/avg/` (metrics mean±std; not averaged .xtc) |
| `--hpc-check-interval 3m` | Poll SLURM more often | Same files; shorter wait between checks |
| `--llm-concurrency` / `--parallel-workers` | Local worker count | Same tree; more `{label}/` dirs fill in parallel |
| `--no-llm` | Heuristic / registry plans (required if no LLM) | Same folders; weaker tool selection |
| `--force-field` / `--water-model` | SimSetup parameters | Same filenames; different `.top` / `.mdp` |
| `--reuse-hpc` | Full pipeline without `sbatch` | Keeps existing `hpc/md.tpr` and `hpc/mdWrap.xtc` |
| `--prompt "…"` | Overrides enriched prompt | Does not change folder layout |

Full flag list: [README.md](../README.md). Pool flags: [POOLS.md](POOLS.md).

---

## Worked examples

Each example is: what you want, the `--goal` prompt, the command, then what
appears on disk.

### 1. One local PDB, full automatic run

**Want:** solvate, run MD, analyse, write an HTML report. No checkpoints.

```bash
python SimAgent.py \
  --goal "Run MD of ./my_protein.pdb in explicit solvent at 310 K.
          Analyse backbone RMSD, RMSF, and radius of gyration.
          Write an HTML report." \
  --working-dir ./erbb3_run
```

`--subtask` omitted → full pipeline. Output lands under
`./erbb3_run/{pdb_stem}/` (e.g. `my_protein/`). HPC pool waits for
`hpc/*.xtc` before analysis. Combined analysis is skipped when N=1.

**Created under `./erbb3_run/my_protein/`:** `preprocess/`, `simsetup/`, `hpc/`,
`analysis/`, `reporter/report.html`. Campaign files at `./erbb3_run/`:
`planner/master_plan.md`, `run_summary.md`, base `agent_conversation.log`.

---

### 2. UniProt accession, prep + HPC only (analyse later)

**Want:** download AlphaFold, trim kinase domain, submit SLURM, stop.

```bash
python SimAgent.py \
  --goal "Study ATP binding of UniProt P21860 ERBB3 kinase domain.
          Prepare apo and ATP+MG systems if the ligand is present." \
  --working-dir ./erbb3 \
  --subtask preprocess simsetup hpcjob
```

No analysis/reporter in `--subtask` → **legacy HPC path** (submit, do not
wait in the cross-sim pool for a later analysis stage). Combined report is
not written.

**Created:** `./erbb3/{label}/preprocess/`, `simsetup/`, `hpc/{label}_run.sh`
and, once SLURM finishes, `hpc/md.xtc`. Holo label (e.g. `p21860_ATP_MG`)
is skipped if ATP/MG is missing, with a reason in `run_summary.md`.

---

### 3. Two PDBs, apo + holo, full pipeline with parallel HPC

**Want:** compare two proteins, protein-only vs ATP-bound, wait for all MD,
then overlay plots.

```bash
python SimAgent.py \
  --goal "Compare pseudokinase dynamics for p21860 and q8iv63 —
          apo and ATP-bound. Overlay RMSF and ligand-pocket distance.
          Combined HTML report." \
  --pdb-list p21860.pdb q8iv63.pdb \
  --working-dir ./pseudo \
  --allowed-hpc-jobs 4 \
  --hpc-check-interval 3m
```

`--pdb-list` implies isolated `{label}/` dirs. Full `--subtask` (omitted) →
prep in the local pool → up to 4 SLURM jobs → post-HPC analysis/reporter
pool → `{base}/analysis/` + `reporter/combined_report.html`.

**Created (typical labels):**

```
./pseudo/p21860/
./pseudo/p21860_ATP_MG/     # skipped if no ATP in that PDB
./pseudo/q8iv63/
./pseudo/q8iv63_ATP_MG/
./pseudo/analysis/
./pseudo/reporter/combined_report.html
```

---

### 4. Trajectories already exist — analysis + report only

**Want:** do not touch topology or SLURM; compute observables on `hpc/`.

```bash
python SimAgent.py \
  --goal "For each finished trajectory compute RMSD, RMSF, DCCM,
          and ligand-pocket distance. Combined overlay report." \
  --working-dir ./pseudo \
  --subtask analysis reporter
```

Or point at existing sim folders:

```bash
python SimAgent.py \
  --goal "Analyse existing production MD. RMSF and pocket SASA.
          Combined HTML report." \
  --sim-dirs ./pseudo/p21860 ./pseudo/q8iv63 \
  --working-dir ./pseudo_reanalysis \
  --subtask analysis reporter
```

`--sim-dirs` requires `hpc/` inside each directory. New combined outputs
go under `--working-dir`.

**Created:** `{label}/analysis/*`, `{label}/reporter/report.html`, and at
base `analysis/` + `reporter/combined_report.html`. `simsetup/` and `hpc/`
are not rewritten.

---

### 5. Pause after every stage (HITL)

**Want:** approve cleaned PDB, topology, and SLURM script before continuing.

```bash
python SimAgent.py \
  --goal "Run MD of ./my_protein.pdb. I will review each stage." \
  --working-dir ./erbb3_hitl \
  --HITL all
```

Same folders as example 1, but the process stops at each checkpoint.
Workers do **not** run in parallel. On failure-only review use
`--HITL error` instead.

---

### 6. Job died during SLURM wait — resume

**Want:** same working directory; skip finished sims; pick up job IDs.

```bash
python SimAgent.py \
  --goal "Continue the kinase comparison (apo and ATP-bound)." \
  --working-dir ./pseudo \
  --resume
```

Reads `./pseudo/supervisor/state.jsonl`. Sims with successful prep +
trajectory are not resubmitted. To force one sim even if it succeeded:

```bash
python SimAgent.py \
  --goal "Re-run q8iv63 ATP case with corrected production length." \
  --working-dir ./pseudo \
  --resume \
  --retry-labels q8iv63_ATP_MG
```

---

### 7. Per-sim analysis done — rebuild combined report only

**Want:** new overlays / narrative without re-analysing every trajectory.

```bash
python SimAgent.py \
  --goal "Rebuild combined RMSF overlays and the HTML comparison report." \
  --working-dir ./pseudo \
  --combined-only
```

Requires `{label}/analysis/analysis_summary.jsonl` (and usually
`{label}/reporter/report.html`). Writes `{base}/analysis/` and
`{base}/reporter/combined_report.html` only.

Do **not** combine `--combined-only` with `--resume` when you still need
per-sim analysis — `--resume` is for unfinished sims; `--combined-only` is
for the base combined stage.

---

### 8. Demo / frozen trajectories (no sbatch)

**Want:** exercise preprocess → setup → analysis on existing `mdWrap.xtc`.

```bash
python SimAgent.py \
  --goal "Full pipeline on existing production trajectories. Do not submit SLURM." \
  --working-dir ./demo \
  --pdb-list a.pdb b.pdb \
  --reuse-hpc
```

Creates/updates `preprocess/` and `simsetup/`, stages HPC files, keeps
`hpc/md.tpr` and `hpc/mdWrap.xtc`, then runs analysis + reporter.

---

### 9. Extend production MD

**Want:** continue `mdrun` on systems that already have `hpc/`.

```bash
python SimAgent.py \
  --goal "Extend production MD from 100 ns to 200 ns for the listed systems." \
  --sim-dirs ./pseudoKin_extend/o60674 ./pseudoKin_extend/p25092 \
  --working-dir ./pseudoKin_extend \
  --subtask hpcjob
```

Only HPC agent files under each sim’s `hpc/` are the target (new/extended
`.xtc`). Preprocess and simsetup are not re-run.

---

## Quick “where do I look?” 

| Question | Path |
|----------|------|
| Did the run finish / which sims failed? | `{base}/run_summary.md` (+ `health` / `failed` counts) |
| Live pool status / healthy vs failed | `{base}/supervisor/pool_status.json` (**preferred**) |
| Resume checkpoint / job IDs / `multi_sim_progress` | `{base}/supervisor/state.jsonl` |
| Why did the planner choose those tools? | `{base}/planner/master_plan.md` or `{sim}/planner/execution_plan.md` |
| Token usage | `{base}/llm_usage.json` |
| Conversation + LLM calls | `{base}/agent_conversation.log` and `{sim}/agent_conversation.log` |
| Trajectory | `{sim}/hpc/md.xtc` or `mdWrap.xtc` |
| Per-sim science | `{sim}/analysis/` + `{sim}/reporter/report.html` |
| Cross-sim comparison | `{base}/analysis/` + `{base}/reporter/combined_report.html` |
