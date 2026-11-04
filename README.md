# AgenticAI — LLM-Powered Molecular Dynamics Workflow

AgenticAI automates GROMACS molecular dynamics (MD) workflows using LLM-powered
planning, multi-agent orchestration, and optional human-in-the-loop checkpoints.

Given a natural-language goal and either a PDB file or a UniProt accession, the
system can:

1. Resolve protein structures (AlphaFold, RCSB) and extract domains via UniProt
2. Preprocess structures (component separation, protonation, phosphorylation mapping)
3. Build simulation systems (topology, solvation, ions, MDP files)
4. Submit and monitor HPC jobs (SLURM)
5. Analyse trajectories (RMSD, RMSF, Rg, DCCM, DSSP, COM distances, ligand RMSD,
   QC, plus optional family-scale modular dynamics and unsupervised classification)
6. Produce HTML reports with literature references and interactive 3D views

Active campaign trees (sims, drafts, local deposits) live under `campaigns/`
and are **gitignored** — not part of the published repository. The public demo
is under [`example/`](example/).

**Worked example (run / track / results):** [example/TUTORIAL.md](example/TUTORIAL.md)

---

## Key Features

- **Natural-language goals** — describe what you want; agents build a concrete plan
- **UniProt / AlphaFold integration** — start from an accession without a local PDB
- **Multi-simulation mode** — run several proteins or component cases in one study
- **Multi-replicate MD** — `--rep-num N` writes `hpc/repXX/` and mean±std under `analysis/avg/`
- **Holo feasibility guard** — skips holo cases when ligands are missing (e.g. AlphaFold)
- **Phosphorylated proteins** — SEP/TPO/PTR stay in the protein chain; mapped for
  CHARMM36 (SP2/THP/TP2), not parameterized as separate ligands
- **Family modular dynamics** — consensus-mapped torsions, RMSF, N↔C DCCM, and
  independent dihedral PCA landscape entropy when the goal asks for comparative descriptors
- **Consensus reference pocket** — define ATP pocket on a reference, map via MSA,
  compute COM distance + axis orientation across systems
- **LLM classification** — feature table → LLM selects a scientifically motivated
  subset (with reasoning) → hierarchical dendrogram + heatmap
- **Supervisor → Planner → Agents** — validated routing, planner-owned master plans, and tool-based execution
- **Resume / retry** — re-run failed multi-sim jobs without redoing successes
- **Cross-sim HPC pool** — prep in parallel (auto-sized), submit up to N SLURM jobs in parallel, then post-HPC analysis ([docs/POOLS.md](docs/POOLS.md))

---

## Installation

### Step 1 — SimAgentEnv (conda)

Scientific dependencies (GROMACS, AmberTools, MDAnalysis, LangChain, …) live in
one conda environment named **`SimAgentEnv`**:

```bash
git clone https://github.com/anupkprasad/agenticAI.git
cd agenticAI
conda env create -f environment.yml
conda activate SimAgentEnv
pip install -e .
```

Cloning the default branch (`main`) is enough for normal use. The manuscript /
Code Ocean freeze lives on branch `SimAgent_v1.0` (`git clone -b SimAgent_v1.0 …`)
and is optional unless you need that peer-review packaging.

Conda places `SimAgentEnv` in its default envs directory (no custom `--prefix`).
Activate it before running SimAgent or use the example launchers, which resolve
the env the same way.

Confirm the CLI loads:

```bash
python SimAgent.py --help
```

**GROMACS** is included in this environment. For phosphorylated proteins with
CHARMM36, install `charmm36-jul2022.ff` separately (see [Force fields](#force-fields)).

### Step 2 — Ollama server (separate)

The conda env only includes the **Ollama Python client**. It does **not** install
the Ollama daemon or download model weights (those are large / machine-specific).

Install the server and pull `gpt-oss:20b` once per machine:

→ **[docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md)**

Quick health check after the server is running:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

LLM planning is **on by default** and **requires** a reachable Ollama server
or an API key (`--llm-api-key` / `OPENAI_API_KEY`). If neither is available,
SimAgent exits with setup guidance. Use `--no-llm` only for offline
heuristic / registry-based planning.

---

## Quick Start

Prerequisites: `SimAgentEnv` activated, and either a reachable Ollama server
(`curl -s http://127.0.0.1:11434/api/tags`) or an API key. Omit `--subtask` to
run the **full** pipeline (`preprocess simsetup hpcjob analysis reporter`).
Pass `--subtask` only when you intentionally want a subset of stages.

### Single simulation from a local PDB (full pipeline)

```bash
python SimAgent.py \
  --goal "Preprocess and setup MD for my_protein.pdb for 50 ns, submit to HPC,
          then analyze (RMSD, RMSF, Rg) and write a report." \
  --working-dir /work/run1 \
  --llm-model gpt-oss:20b \
  --llm-base-url http://127.0.0.1:11434
```

### UniProt accession (download + domain extraction, full pipeline)

```bash
python SimAgent.py \
  --goal "Study ATP binding of UniProt P21860 (ERBB3). Download AlphaFold PDB,
          extract kinase domain, run 1 ns MD for protein-only and protein+ATP+MG,
          submit to HPC, then analyze and report." \
  --working-dir /work/erbb3 \
  --llm-model gpt-oss:20b \
  --llm-base-url http://127.0.0.1:11434
```

### Multi-protein comparative study (full pipeline)

```bash
python SimAgent.py \
  --goal "Run 100 ns MD for each pseudokinase, submit to HPC, then analyze and
          write a combined report.
          Names: p21860=ERBB3, q8iv63=VRK3, q8nb16=MLKL, q8wz42=TITIN." \
  --pdb-list p21860.pdb q8iv63.pdb q8nb16.pdb q8wz42.pdb \
  --working-dir /work/pseudo \
  --llm-model gpt-oss:20b \
  --llm-base-url http://127.0.0.1:11434
```

### Setup + HPC only (no analysis yet)

Use `--subtask` when you only want these stages:

```bash
python SimAgent.py \
  --goal "Preprocess and setup MD for my_protein.pdb for 50 ns, then submit to HPC." \
  --working-dir /work/run1 \
  --subtask preprocess simsetup hpcjob \
  --llm-model gpt-oss:20b \
  --llm-base-url http://127.0.0.1:11434
```

### Analysis-only on completed trajectories

```bash
python SimAgent.py \
  --goal "Compute RMSD, RMSF, Rg, DCCM, DSSP and generate the report." \
  --working-dir /work/run1 \
  --subtask analysis reporter \
  --llm-model gpt-oss:20b \
  --llm-base-url http://127.0.0.1:11434
```

Remote Ollama: change `--llm-base-url` to `http://GPU_HOST:11434`.
Paid API: add `--llm-provider openai --llm-api-key "$OPENAI_API_KEY"` (and a
matching `--llm-model` / `--llm-base-url`). Offline heuristic plans: `--no-llm`.

Hands-on walkthrough (apo/holo panel, status, reports, provenance):
[example/TUTORIAL.md](example/TUTORIAL.md)

**Worked-example combined report (inspect results here):**
[`example/pseudo_apo_holo/reporter/combined_report.html`](example/pseudo_apo_holo/reporter/combined_report.html)
· [GitHub file](https://github.com/anupkprasad/agenticAI/blob/SimAgent_v1.0/example/pseudo_apo_holo/reporter/combined_report.html)
· [HTML preview](https://htmlpreview.github.io/?https://raw.githubusercontent.com/anupkprasad/agenticAI/SimAgent_v1.0/example/pseudo_apo_holo/reporter/combined_report.html)
(third-party preview; if it fails on this ~8–10 MB file, download from GitHub and open locally)

---

## Prompt Engineering Guide

AgenticAI works best when `--goal` describes the scientific intent, the available inputs, the requested workflow stages, and the exact analyses you want. Write the goal as a short paragraph or a few sentences, not as comma-separated metadata. Natural language gives the planner enough context to create per-simulation prompts that downstream agents can follow.

Include these details when they apply:

- **Inputs and labels:** list PDB files, UniProt IDs, simulation directories, and protein names such as `p21860: ERBB3`.
- **Workflow stage:** say whether to preprocess, set up simulation, submit to HPC, analyze completed trajectories, report results, or only run a subset via `--subtask`.
- **Simulation intent:** specify component cases such as protein-only, protein+ATP+MG, mutant vs wild type, phosphorylated vs dephosphorylated, or chain/residue windows.
- **Analysis scope:** name the exact analyses you want. For example, “RMSF only for all simulations” will keep the analysis focused on RMSF. If you ask broadly for “protein dynamics” without naming metrics, the planner may choose appropriate dynamics analyses such as RMSD, RMSF, Rg, DCCM, or interaction distances based on available tools and biological context.
- **Family comparative descriptors (recommended for kinase/pseudokinase panels):** describe science, not internal column names — e.g. ATP–pocket COM distance mean/std, pocket axis orientation, consensus Cα RMSF, pocket χ₁, N↔C DCCM, independent dihedral PCA landscape entropy, then hierarchical clustering with a feature heatmap (do not force a fixed cluster count unless you want one). The framework schedules the shared analysis protocol (required calculations per protein) and lets the LLM pick a feature subset with written reasoning; empty columns are marked unavailable rather than claimed as used.
- **Per-metric simulation subsets (multi-sim):** each combined metric can target a different set of simulations. You do not need every metric on every protein. Examples:
  - “RMSF for all four simulations” → combined RMSF overlay uses all sims.
  - “DCCM for JAK1 and TYK2 only” → per-sim DCCM on those two; combined DCCM compares only them.
  - “Radius of gyration for JAK1, TYK2, and ULK4” → Rg per-sim on those three; combined Rg overlay on those three only (STRAA excluded).
- **Combined analysis intent:** ask explicitly for comparison, cross-simulation trends, aggregate plots, or a combined report only when you want base-level combined analysis. Otherwise multi-sim runs focus on the individual simulation plans.
- **COM distance wording (important):** two tools exist — choose the phrasing that matches your intent:
  - **Ligand pocket distance** (recommended for holo kinases): say “ligand pocket distance”, “ATP distance to the catalytic pocket”, or “protein atoms within 5 Å of ATP at frame 0”. Output: `ligand_pocket_distance.csv`.
  - **Whole-protein COM distance:** say “COM distance between the **whole protein** and ATP” or “protein COM to ligand COM”. Output: `com_distance.csv`.
  - Do not mix both unless you explicitly want pocket tracking **and** whole-protein COM.
- **Outputs:** mention required plots, CSV summaries, residue ranges, ligand-pocket definitions, literature context, and report format.

Example multi-metric subset prompt:

```bash
--goal "Trajectories are complete for p23458:JAK1, p29597:TYK2, q7rtn6:STRAA, q96c45:ULK4. Analysis only. For all simulations: RMSF and ligand pocket distance (ATP, 5 Å pocket at frame 0). Combined: overlay RMSF and pocket distance across all four. DCCM for JAK1 and TYK2 only. Radius of gyration for JAK1, TYK2, and ULK4 with a combined Rg plot. Do not run preprocessing or HPC."
```

Example focused analysis prompt:

```bash
--goal "Simulations are already complete for p23458.pdb, p29597.pdb, and p52333.pdb under /scratch/project. Run analysis and reporting only. I want RMSF only for each simulation, with per-residue RMSF plots and a CSV summary for each protein. Do not run preprocessing, setup, HPC, or additional analyses."
```

Example comparative prompt:

```bash
--goal "Analyze completed trajectories for ERBB3, VRK3, MLKL, and TITIN. Compute backbone RMSD, per-residue RMSF, and radius of gyration for each simulation, then create a combined comparison report showing cross-protein trends and shared flexible regions. Include protein names in the report and cite relevant literature."
```

---

## Human-in-the-Loop (HITL)

By default the workflow runs **fully automatically** — no pauses between stages.

Use **`--HITL`** when you want interactive review:

| Mode            | Flag             | Behavior                                                                              |
| --------------- | ---------------- | ------------------------------------------------------------------------------------- |
| Off (default)   | *(omit flag)*  | End-to-end execution with no checkpoints                                              |
| Errors only     | `--HITL error` | Pause when a stage fails, on HPC pool submit/SLURM failures, or after max retries     |
| All checkpoints | `--HITL all`   | Pause after preprocess, setup, HPC, analysis, reporter (and HPC pool when applicable) |

### What you can do at a checkpoint

The terminal opens a **bidirectional chat** with the active field agent:

- **Ask questions** — e.g. “What RMSF files were generated?”, “Show the execution plan”
- **Inspect files** — `show rmsf.dat`, `files`, `list analysis/`, `tools`, `agents`
- **Switch field agent** — `switch analysis`, `switch reporter`, `switch setup`, …Reloads that agent’s domain tools and artifact paths (trajectories, per-sim `state.jsonl`, output dirs).
- **Bind a simulation (multi-sim)** — `switch p23458 analysis` or `switch analysis p23458`Chat and tools use that sim’s directories; session is saved in `state.jsonl` (`hitl_active_agent`, `hitl_target_sim_label`).
- **Delegate tasks** — the agent **runs** your request and returns to the same checkpoint:
  - `run analysis: calculate Rg for JAK1`
  - `run reporter: add a DCCM section to the report`
  - `run: <task>` — uses the currently active agent
  - Multi-sim: `run p23458 analysis: calculate RMSF`
- **Execute in chat** — after `switch p29597 analysis`, describe the task in natural language
  (e.g. `calculate DSSP and plot heatmap for residues 100–120`). The **Analysis agent** builds a
  JSON execution plan (like the normal workflow), runs **all steps**, and writes:
  - `{sim}/analysis/hitl_execution_plan.json` — structured plan
  - `{sim}/analysis/execution_log.txt` — step-by-step log
  - `{sim}/agent_conversation.log` — concise summary (no full JSON dumps)
- **Get clarifications from the agent** — the LLM may ask follow-up questions when your request is ambiguous
- **Recover from errors** — after max retries, use:
  - `recommend: <advice>` — agent retries with your guidance
  - `modify: <changes>` — same, with explicit change instructions
  - `retry` — rerun the stage from scratch
- **Proceed** — `approved` / `continue`
- **Stop** — `exit` / `quit`

At the **reporter** checkpoint you can also say `analysis: …` or `reporter: …` to
re-run those stages with new instructions (full pipeline rerun, not just a single delegated task).

### Multi-sim + HITL tips

| Situation                                                       | Recommended command                                                                                                |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| First run, pause after each sim’s analysis                     | `--HITL all`                                                                                                     |
| Automatic run; pause only if something fails                    | `--HITL error`                                                                                                   |
| Per-sim work already done; review combined report interactively | Re-run with`--HITL all` — framework auto-detects existing `analysis/` folders and enters combined-only review |
| Retry only failed sims                                          | Add`--resume` (optionally `--retry-labels p23458`)                                                             |
| Re-run combined overlay + report only                           | `--combined-only` (requires N>1)                                                                                 |
| Full campaign with parallel HPC                                 | `--allowed-hpc-jobs 4 --hpc-check-interval 3m` (see [pools](docs/POOLS.md))                                       |

State is saved to `{working_dir}/supervisor/state.jsonl` (and per-sim copies under
`{label}/supervisor/`). See [docs/CONVENTIONS.md](docs/CONVENTIONS.md) for resume semantics.

Example (interactive analysis + report on completed trajectories):

```bash
python SimAgent.py \
  --goal "Analysis only for p23458, p29597, q7rtn6, q96c45 …" \
  --working-dir ./agenticB5R1 \
  --subtask analysis reporter \
  --HITL all
```

---

## CLI Reference

```bash
python SimAgent.py --goal "..." [options]
```

| Flag                     | Default                          | Description                                                                                                    |
| ------------------------ | -------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `--goal`               | *(required)*                   | Natural-language simulation goal                                                                               |
| `--working-dir`        | `.`                            | Campaign base; each sim under`{dir}/{label}/`                                                                |
| `--pdb-list`           | —                               | Explicit PDB list                                                                                              |
| `--sim-dirs`           | —                               | Existing sim directories for analysis-only                                                                     |
| `--subtask`            | all agents                       | `preprocess simsetup hpcjob analysis reporter`                                                               |
| `--no-llm`             | off                              | Heuristic / registry-based plans only (no LLM required)                                                        |
| `--llm-model`          | `gpt-oss:20b`                  | Model name                                                                                                     |
| `--llm-base-url`       | `http://localhost:11434`       | LLM API base URL                                                                                               |
| `--HITL`               | off                              | `error` or `all` — enable human-in-the-loop (default: off; or `campaign.yaml` `hitl`)                 |
| `--campaign-yaml`      | shipped /`{dir}/campaign.yaml` | Mapping, retrieval k, gold columns, HITL defaults ([CAMPAIGN_AND_RETRIEVAL.md](docs/CAMPAIGN_AND_RETRIEVAL.md)) |
| `--force-field`        | `amber99sb-ildn`               | GROMACS force field (e.g.`charmm36-jul2022`)                                                                 |
| `--water-model`        | `tip3p`                        | Water model                                                                                                    |
| `--max-concurrent`     | `4`                            | Max concurrent sims (legacy)                                                                                   |
| `--parallel-workers`   | `auto`                         | Max parallel local workers for prep / analysis+reporter (`auto` or integer; `1`=sequential)                |
| `--parallel-mem-gb`    | phase default                    | Estimated GiB RAM per parallel worker                                                                          |
| `--parallel-cpus`      | phase default                    | Estimated CPU cores per parallel worker                                                                        |
| `--llm-concurrency`    | `auto` (4)                     | Cap parallel workers to match Ollama`OLLAMA_NUM_PARALLEL` slots                                              |
| `--allowed-hpc-jobs`   | auto                             | Max concurrent SLURM jobs in cross-sim HPC pool                                                                |
| `--hpc-check-interval` | `2h`                           | SLURM poll interval during HPC pool wait (`2h`, `30m`, `7200`)                                           |
| `--resume`             | off                              | Re-run only failed/incomplete multi-sim jobs                                                                   |
| `--retry-labels`       | —                               | Force-retry specific simulation labels                                                                         |
| `--combined-only`      | off                              | Only when N>1: combined analysis + report at`{base}/analysis/` and `{base}/reporter/combined_report.html`  |
| `--rep-num`            | `1`                            | Independent production replicates per label (`hpc/repXX/` + `analysis/avg/`)                               |

---

## Force Fields

| Force field          | Use case           | Notes                                                    |
| -------------------- | ------------------ | -------------------------------------------------------- |
| `amber99sb-ildn`   | Default protein MD | TIP3P water; GAFF/ACPYPE for small-molecule ligands      |
| `charmm36-jul2022` | CHARMM36 proteins  | Recommended for phosphorylated proteins (SP2, THP1, TP2) |

Phosphorylated residues (SEP, TPO, PTR) are kept in `protein.pdb` during
preprocessing. For CHARMM36, names are mapped automatically (SEP→SP2, TPO→THP,
PTR→TP2). Do not use `generate_ligand_parameters` for phospho amino acids.

Install CHARMM36 for GROMACS from the [MacKerell lab](https://mackerell.umaryland.edu/charmm_ff.shtml#gromacs)
into your GROMACS `share/gromacs/top/` directory, then:

```bash
python SimAgent.py ... --force-field charmm36-jul2022
```

---

## Output Layout

Multi-sim run under `--working-dir /work/pseudo`:

```
/pseudo/
  p21860/
    preprocess/       # protein.pdb, protein_h.pdb, ...
    simsetup/         # topol.top, *.gro, *.mdp
    hpc/              # SLURM script, trajectories (or hpc/rep01, rep02 with --rep-num)
    analysis/         # RMSD, RMSF, DCCM; optional consensus_*/ and avg/
    reporter/         # report.html (fixed name — every per-sim report)
    agent_conversation.log
  p21860_ATP_MG/      # or skipped with reason in run_summary
  cross_sim/          # optional pre_combined MSA / pocket-map artifacts
  campaign/           # state.json, memory.jsonl
  analysis/           # combined overlays; classification_* when requested
  reporter/
    combined_report.html   # fixed name — cross-simulation HTML report
  run_summary.md      # human-readable outcome (+ science completeness)
  run_summary.json    # structured outcome
  llm_usage.json      # token totals by workflow node
  agent_conversation.log
```

**Report naming (automation-friendly):** every per-simulation HTML report is always
`{label}/reporter/report.html`. The combined multi-sim report is always
`{base}/reporter/combined_report.html`. Do not rely on protein-specific filenames
(e.g. `kinase_report.html`) — the framework normalizes to these paths. Family
campaigns also show a science-completeness banner in the combined HTML.
**Multi-sim reporting phases:** when the user goal requests combined analysis, the
supervisor runs optional `pre_combined` (MSA/pocket → `cross_sim/`) → per-sim
analysis → per-sim `report.html` → `post_combined` at `{base}/analysis/`
(overlays, optional classification) → combined `combined_report.html`. Use
`--combined-only` to regenerate only the base-level combined analysis and report
when per-sim work is already complete.

---

## Repository Layout

```
SimAgent.py                # CLI entry point
LICENSE                    # MIT
environment.yml            # Conda environment (SimAgentEnv)
ollama_server.slurm        # Example Slurm job to serve Ollama on GPU nodes
agentic/
  workflow.py              # LangGraph StateGraph orchestration
  supervisor/              # Routing, enrichment, multi-sim loop
  planner/                 # Execution plans + disposition base
  preprocess/              # Structure download, clean, separate
  simsetup/                # Topology, solvation, MDP generation
  hpc/                     # SLURM submission and monitoring
  analysis/                # Trajectory analysis tools
  reporter/                # HTML reports and literature
  programmer/              # Tool-authoring agent + AVAILABLE_SOFTWARE.md
src/
  preprocess/              # PDB tools, phospho mapping, remodel
  simsetup/                # GROMACS system builder
  analysis/                # Combined cross-sim analysis
  reporter/                # Report generation
example/                   # Runnable demo (pseudo_apo_holo) + use-case tutorial
example/TUTORIAL.md        # Run / track / find results for the example campaign
docs/
  PROJECT.md               # Product overview, layout, CLI, run outputs
  PIPELINE_WORKFLOW.md     # Agent order, directories, files, flag examples
  ARCHITECTURE.md          # LangGraph, state, agents
  CONVENTIONS.md           # Contributor coding and documentation rules
  TOOLS.md                 # Dependencies + multi-chain / phospho mechanisms
  ANALYSIS_TOOLS.md        # Per-tool observables, theory, output names
  POOLS.md                 # Local parallel workers + SLURM HPC pool
  OLLAMA_SETUP.md          # Install Ollama + pull gpt-oss:20b
```

---

## Documentation

| Document                                              | Description                                                |
| ----------------------------------------------------- | ---------------------------------------------------------- |
| [example/TUTORIAL.md](example/TUTORIAL.md)             | Example campaign: run, track status, reports, provenance   |
| [docs/PROJECT.md](docs/PROJECT.md)                     | Product overview, repository layout, quick start, CLI      |
| [docs/PIPELINE_WORKFLOW.md](docs/PIPELINE_WORKFLOW.md) | Agent order, directories, files, and flag examples         |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)           | LangGraph pipeline, state, and per-agent design            |
| [docs/CONVENTIONS.md](docs/CONVENTIONS.md)             | Contributor coding and documentation rules                 |
| [docs/TOOLS.md](docs/TOOLS.md)                         | External dependencies and GROMACS/analysis mechanisms      |
| [docs/ANALYSIS_TOOLS.md](docs/ANALYSIS_TOOLS.md)       | Analysis tool calculations, theory, and standard outputs   |
| [docs/POOLS.md](docs/POOLS.md)                         | Local parallel workers and SLURM HPC pool                  |
| [docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md)           | Install Ollama server + pull`gpt-oss:20b` (not in conda) |

---

## Development

```bash
# Deterministic routing (no LLM)
python SimAgent.py --goal "..." --no-llm

# Custom Ollama endpoint (LLM is on by default)
python SimAgent.py --goal "..." --llm-base-url http://127.0.0.1:11434

# Interactive checkpoints
python SimAgent.py --goal "..." --HITL all
```

Tests: `pytest` (see `environment.yml` for dependencies).

---

## License

This project is released under the [MIT License](LICENSE). CHARMM36 force field
parameters are distributed separately by the MacKerell lab and are not part of
this repository.

---

## Examples run in the project

### 1. Paper cohort: 37 human (pseudo)kinase–ATP systems

Locked natural-language objective used for the family-scale study in the
manuscript (robustness / end-to-end cohort). Campaign outputs stay local under
`campaigns/` (gitignored).

```bash
python SimAgent.py \
  --goal "$(cat <<'EOF'
I have 37 human protein–ATP holo structures in given working directory
(one PDB per system). There are 32 pseudokinases and 5 ground-truth active kinases.
Please run a full end-to-end comparative MD study on all of them.

Systems (UniProt id : protein name):
  o15197:EPHB6, o43187:IRAK2, o60674:JAK2, p00533:EGFR, p17612:KAPCA, p21860:ERBB3, p23458:JAK1, p24941:CDK2, p25092:GUC2C, p28482:MK01, p29597:TYK2, p51841:GUC2F, p52333:JAK3,
  q05823:RN5A, q13308:PTK7, q13418:ILK, q58a45:PAN3, q5jzy3:EPHAA, q6vab6:KSR2, q7rtn6:STRAA, q7z7a4:PXK, q8iv63:VRK3, q8ivt5:KSR1, q8nb16:MLKL, q8ncb2:CAMKV, q8ne28:STKL1,
  q8tea7:TBCK, q8wz42:TITIN, q92519:TRIB2, q96c45:ULK4, q96qs6:PSKH2, q9bxu1:STK31, q9c0k7:STRAB, q9nsy0:NRBP2, q9uhy1:NRBP, q9y243:AKT3, q9y616:IRAK3

For each complex, preprocess the structure and set up GROMACS with
AMBER99SB-ILDN, TIP3P water, 310 K, 1 bar, and 0.15 M NaCl.
Run two independent 200 ns production MD replicates per system, wait for all
simulations to finish, then analyze and plot the full 200 ns of every
trajectory (do not truncate to a shorter window).

Use KAPCA (p17612) as the reference to define the ATP-binding pocket
(residues within 15 Å of ATP, unless a different cutoff is stated), map that
pocket onto the other proteins with a global sequence alignment
(MAFFT / star MSA), and plot both the global MSA and the pocket /
high-consensus MSA panels.

From both replicates (then average across replicates), extract these ten
scalar dynamics descriptors for every system. All ten are required for
clustering — do not drop any:

1. ATP COM distance to the consensus pocket — mean
2. ATP COM distance to the consensus pocket — standard deviation
3. ATP orientation vs the pocket axis — mean axis angle
4. ATP orientation vs the pocket axis — standard deviation of the axis angle
5. Pocket side-chain χ₁ circular mean
6. Pocket side-chain χ₁ circular standard deviation
7. Flexibility of consensus-mapped Cα atoms — mean RMSF
8. Flexibility of consensus-mapped Cα atoms — standard deviation of RMSF
9. N-lobe ↔ C-lobe DCCM mean correlation
10. Shared-reference φ/ψ/χ₁ dihedral PCA dynamics scalar
    (pca_pka_ref_shared_dyn = √(d_g² + d_c² + pc_rms²) vs KAPCA in the
     shared PKA PC space; do not substitute independent per-protein PCA
     grid entropy)

When all systems are done, assemble those ten descriptors into one feature
table, run consensus hierarchical clustering, and write a single dendrogram +
feature-heatmap panel (robust z-score / IQR scaling). Also write a combined
HTML report with brief literature context. You may mark a k=4 cut for
interpretation, but still emit the full tree.
EOF
)" \
  --working-dir ./pseudokin_37x2/run_01 \
  --force-field amber99sb-ildn \
  --water-model tip3p \
  --rep-num 2
```

Agent plans, logs, reports, and analysis plots for local cohort runs are written
under the chosen `--working-dir` (typically inside a gitignored `campaigns/` tree).

### 2. Analysis-only: four completed systems

```bash
# --- Analysis-only: four completed systems ---
python SimAgent.py \
  --goal "Simulation are already done for these uniprot ids: p23458.pdb, p29597.pdb,
          q7rtn6.pdb, q96c45.pdb. So please do not preprocess or simsetup or hpc.
          Directly do the analysis of these data. I want specifically RMSF of protein
          and COM distance of ATP (ligand) from protein for all the simulations.
          Also run PCA on protein Cα, plot PC1 vs PC2, and compute the free energy
          landscape from PC1 and PC2 at 310 K.
          In combined analysis, please compare the RMSF in cross simulations and
          ligand pocket distance in cross simulations and free energy landscape.
          The given uniprotid:protein name are p23458:JAK1, p29597:TYK2,
          q7rtn6:STRAA and q96c45:ULK4. Calculate the DCCM for JAK1 and TYK2 to
          compare the dynamics between these two proteins. Please also calculate
          radius of gyration for JAK1, TYK2 and ULK4 and compare them in plot.
          In report preparation, please focus on relevant pseudokinase literature of
          these simulated proteins." \
  --working-dir ./agenticB5R1 \
  --subtask analysis reporter
```
