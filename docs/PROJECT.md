# SimAgent — LLM-Powered Molecular Dynamics Workflow

**This file is the project overview.** Start here to learn what SimAgent
does, which features it ships, how the repository is laid out, how to run a
job, and which files a run writes. It is not the design of the LangGraph, the
analysis-tool theory, or the coding rules — those live in the other `docs/`
pages listed under [Documentation](#documentation).

---

## Overview

SimAgent is an agentic AI system that automates molecular dynamics (MD) simulation
workflows using LLM-powered reasoning, intelligent task orchestration, and optional
human-in-the-loop decision making.

Given a natural-language goal and either a PDB file or a UniProt accession, the system:

1. Resolves the protein structure — downloads from AlphaFold or RCSB if no local file
2. Extracts a specific domain if requested (e.g. "kinase domain"), using UniProt
   residue annotations
3. Validates and preprocesses the structure (protonation, component separation)
4. Generates simulation parameters, topology, and coordinates (GROMACS)
5. Submits and monitors HPC jobs via SLURM
6. Analyses the resulting trajectory (RMSD, RMSF, Rg, DCCM, DSSP, COM distances,
   plus optional family-scale modular dynamics)
7. Produces an interactive HTML report with literature references, figures, and a
   3D structure viewer

## Key Features

- **Natural-language interface** — describe what you want; the LLM translates it
  into a concrete, dependency-aware MD plan.
- **UniProt / AlphaFold integration** — start from a UniProt accession alone; the
  framework downloads the predicted structure and extracts the requested domain.
- **Domain intelligence** — named domains (e.g. "kinase domain") are resolved to
  exact residue ranges via the UniProt REST API, not hard-coded numbers.
- **Multi-simulation and component-case expansion** — a single command can drive
  N proteins × M component cases (e.g. apo + holo) with isolated per-sim
  directories and a combined analysis phase.
- **Multi-replicate MD** — `--rep-num N` fans analysis across `hpc/repXX/` and
  writes mean±std under `analysis/avg/` for overlays and classification.
- **Holo feasibility guard** — if a requested ligand/ion is absent in the source
  structure (e.g. AlphaFold has no ATP), the Supervisor skips the holo case and
  records the reason rather than producing a broken simulation.
- **Supervisor-Planner-Programmer-Agent hierarchy** — a supervisor validates
  feasibility and routes; a planner builds an execution plan from a live tool
  catalogue; field agents carry out each step.
- **LLM-driven scientific inference** — the Planner selects analysis observables
  from goal semantics; the Analysis agent selects tools per trajectory; the
  Reporter synthesises a literature-grounded narrative.
- **Family modular dynamics** — consensus pocket metrics (COM + axis-angle),
  mapped torsions, RMSF, N↔C DCCM, and independent dihedral PCA landscape
  entropy when the goal asks for comparative descriptors (not a fixed paper
  schema). Required calculations are artifact-gated for science completeness.
- **Consensus reference pocket** — define ATP pocket on a reference (e.g. KAPCA),
  map via star MSA, compute COM distance + axis orientation for all systems.
- **LLM classification feature selection** — after collecting dynamics scalars,
  the Analysis agent asks the LLM to choose a scientifically motivated subset
  (with written reasoning) before hierarchical clustering / dendrogram+heatmap.
  Empty / all-NaN columns are recorded as unavailable, not claimed as used.
- **Campaign YAML** — `campaign.yaml` sets conservation metric, pocket cutoff,
  retrieval k, gold columns, and HITL. `--campaign-yaml` / env vars override.
- **Knowledge RAG** — planner sees retrieved chunks with `[kb:…]` citations,
  not a dump of every manual.
- **Human checkpoints** — optional approval gates after preprocessing, setup, HPC,
  analysis, and reporter. Enable with `--HITL all` or pause on failures with `--HITL error`.
- **Restricted file tools** — field agents can list, search, and read the
  inventory only inside this study’s directories. HITL `pwd` prints
  `{stage}/inventory.json`.
- **Run memory** — one episode per distinct error at `campaign/memory.jsonl`
  (stage, tool, system, fix). Analysis, reporter, and the supervisor retrieve
  it with the same embedding search as the planner, together with short study
  notes from this run.
- **Review pauses** — `--HITL all` can pause after the compiled protocol,
  after shared MSA/pocket mapping, and after cross-system comparison.
- **Programmer software list** — [`AVAILABLE_SOFTWARE.md`](../agentic/programmer/AVAILABLE_SOFTWARE.md)
  documents engines/libraries the Programmer may assume when generating tools.
- **Cross-sim HPC pool** — multi-sim full pipeline preps in parallel (local
  workers), submits up to N SLURM jobs in parallel, then runs post-HPC
  analysis/reporter (see [POOLS.md](POOLS.md)).
- **Required LLM by default** — Ollama or API key; use `--no-llm` for heuristic / registry-based plans only.
- **Run audit trail** — `agent_conversation.log`, `execution_plan.md`,
  `execution_report.md`, `run_summary.md`, `run_summary.json`, and
  `llm_usage.json` (tokens tagged by workflow node) at the base working
  directory after every run.
- **Interactive HTML report** — literature (PubMed + bioRxiv + UniProt),
  analysis plots, DCCM and DCCM-difference heatmaps, a science-completeness
  banner for family campaigns, and a 3Dmol.js viewer with trajectory
  snapshots and PNG export.

**Hands-on tutorial:** [tutorial/TUTORIAL.md](../tutorial/TUTORIAL.md) runs
`tutorial/pseudo_apo_holo/` from the goal, launcher, and starting PDBs.
Reports and trajectories are produced by that run. Framework install and CLI
remain in this file and [README.md](../README.md).

## Repository Layout

```
SimAgent.py                     # Launcher; command line is agentic/cli.py
environment.yml                 # Conda environment (SimAgentEnv, Python 3.11)
tutorial/
    TUTORIAL.md                 # Use-case guide: run / track / find results
    pseudo_apo_holo/            # Fresh apo/holo starter (PDBs, goal, launcher)
agentic/
    cli.py                      # Command line used by `simagent` and SimAgent.py
    state.py                    # MDState TypedDict — central shared state (~65 fields)
    workflow.py                 # LangGraph StateGraph (12 nodes)
    supervisor/                 # MDSupervisor: routing, enrichment, multi-sim loop
    planner/                    # MDPlanner + MDProgrammer; knowledge/ (index is generated)
    campaign/                   # CampaignSpec, campaign.yaml, compile/contracts
    retrieval/                  # Hybrid tool + knowledge retriever
    preprocess/                 # PreprocessingAgent: download, domain trim, clean
    simsetup/                   # SimulationSetupAgent: topology, MDP, solvation
    hpc/                        # MDHPCAgent: SLURM, SSH, job monitoring
    analysis/                   # MDAnalysisAgent: traj metrics, modular family tools,
                                #   combined overlays, LLM feature selection + clustering
    reporter/                   # ReporterAgent: HTML report, literature, 3D viewer
    utils/                      # Logging, SecureFileManager, plan persistence, state_persistence
    multi_sim_progress.py       # Per-sim / family-modular completion checks
src/
    analysis/                   # Trajectory tools, classification collector/clustering,
                                #   consensus pocket, family_dynamics_core, replicate avg
    preprocess/
        structure_sources/      # AlphaFold + RCSB download backends (pluggable)
        domain_sources/         # UniProt domain lookup + offline fallback
        structure_request_parser.py
        domain_extractor.py
        structure_downloader.py
        acquire_structure_tool.py
        domain_lookup_tool.py
        structure_validator.py
        complex_separator.py
        hydrogen_adder.py
    supervisor/
        input_validator.py      # Goal parsing, UniProt resolution, feasibility
        unified_enricher.py     # Single LLM enrichment call
        component_parser.py     # Holo component validation
    reporter/
        combined_reporter.py    # Cross-simulation HTML report generation
    utils/
        run_summary.py          # Build + write run_summary.md / run_summary.json
        pdb_analyzer.py
        chat_tools.py
docs/                           # Project documentation (you are here)
```

## Documentation

| Doc | What it covers |
|-----|----------------|
| [PROJECT.md](PROJECT.md) | This page — product overview, layout, quick start, CLI, run outputs |
| [tutorial/TUTORIAL.md](../tutorial/TUTORIAL.md) | Hands-on apo/holo tutorial: run from scratch, track status, reports |
| [PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md) | Agent order, directories, files, and flag examples |
| [ARCHITECTURE.md](ARCHITECTURE.md) | LangGraph, shared state, agents, directory layout |
| [CONVENTIONS.md](CONVENTIONS.md) | Coding and documentation rules for contributors |
| [ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md) | Per-tool observables, theory, and output filenames |
| [TOOLS.md](TOOLS.md) | External dependencies plus GROMACS/analysis mechanisms |
| [POOLS.md](POOLS.md) | Local parallel workers and SLURM HPC pool |
| [CAMPAIGN_AND_RETRIEVAL.md](CAMPAIGN_AND_RETRIEVAL.md) | CampaignSpec, science gates, hybrid tool retrieval |
| [OLLAMA_SETUP.md](OLLAMA_SETUP.md) | Ollama server + `gpt-oss:20b` (separate from conda) |

Each page opens with a purpose paragraph. Related mechanisms share one
descriptive file (tools together, pools together). Do not leave the
explanation only in chat or code comments. Full CLI flags:
[README.md](../README.md).

## Quick Start

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate SimAgentEnv
pip install -e .

# 2. Confirm Ollama (or use --llm-api-key / --no-llm)
curl -s http://127.0.0.1:11434/api/tags

# 3. Full pipeline from a local PDB (omit --subtask = all agents)
python SimAgent.py \
    --goal "Run MD for my_protein.pdb, submit to HPC, analyze, and report" \
    --working-dir /work/run1 \
    --llm-model gpt-oss:20b \
    --llm-base-url http://127.0.0.1:11434

# 4. UniProt accession (full pipeline)
python SimAgent.py \
    --goal "Study ATP binding of UniProt P21860 ERBB3 kinase domain;
            download AlphaFold, extract domain, run MD, analyze, report" \
    --working-dir /work/erbb3 \
    --llm-model gpt-oss:20b \
    --llm-base-url http://127.0.0.1:11434

# 5. Multi-protein comparative study (full pipeline)
python SimAgent.py \
    --goal "Compare pseudokinase dynamics for p21860, q8iv63 — apo and ATP-bound" \
    --pdb-list p21860.pdb q8iv63.pdb \
    --working-dir /work/pseudo \
    --llm-model gpt-oss:20b \
    --llm-base-url http://127.0.0.1:11434

# 6. Setup + HPC only (intentional subset)
python SimAgent.py \
    --goal "Preprocess, setup, and submit MD for my_protein.pdb" \
    --working-dir /work/run1 \
    --subtask preprocess simsetup hpcjob \
    --llm-model gpt-oss:20b \
    --llm-base-url http://127.0.0.1:11434

# 7. Analysis + report only
python SimAgent.py \
    --goal "Analyse existing trajectory in /work/run1/hpc" \
    --working-dir /work/run1 \
    --subtask analysis reporter \
    --llm-model gpt-oss:20b \
    --llm-base-url http://127.0.0.1:11434

# 8. Interactive checkpoints
python SimAgent.py \
    --goal "..." \
    --working-dir /work/run1 \
    --llm-model gpt-oss:20b \
    --llm-base-url http://127.0.0.1:11434 \
    --HITL all
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--goal` | *(required)* | Natural-language simulation goal |
| `--working-dir` | `.` | Campaign base; each sim writes under `{dir}/{label}/` |
| `--pdb-list` | *(none)* | Explicit PDB file list for multi-sim |
| `--sim-dirs` | *(none)* | Existing sim dirs for analysis-only multi-sim |
| `--subtask` | *(all)* | Agents to run: `preprocess simsetup hpcjob analysis reporter` |
| `--no-llm` | `False` | Heuristic / registry plans only (skips LLM requirement) |
| `--llm-model` | `gpt-oss:20b` | Ollama model name |
| `--llm-base-url` | `http://127.0.0.1:11434` | Ollama server URL |
| `--HITL` | off | `error` or `all` — enable human-in-the-loop (or `campaign.yaml` `hitl`) |
| `--campaign-yaml` | shipped / `{dir}/campaign.yaml` | Mapping, retrieval, gold columns, HITL defaults |
| `--resume` | off | Restore `{working-dir}/supervisor/state.jsonl` and continue |
| `--combined-only` | off | Combined analysis + reporter only (existing per-sim results) |
| `--allowed-hpc-jobs` | auto | Max concurrent SLURM jobs in cross-sim HPC pool |
| `--hpc-check-interval` | `2h` | SLURM poll interval during HPC pool wait |
| `--llm-concurrency` | auto | Cap on concurrent LLM-using workers (prep / analysis) |
| `--force-field` | `amber99sb-ildn` | GROMACS force field override |
| `--water-model` | `tip3p` | Water model override |
| `--rep-num` | `1` | Independent production replicates per system (`hpc/repXX/`) |
| `--prompt` | *(none)* | Override enriched prompt |
| `--log-file` | *(auto)* | Conversation log path |

This is the common subset. Parallel-pool and remaining flags:
[POOLS.md](POOLS.md) and [README.md](../README.md).

## Run Outputs

After every run the following are written to the base working directory:

| File | Description |
|------|-------------|
| `run_summary.md` | Human-readable run outcome: mode, counts, per-sim status/**health**, **science completeness** |
| `run_summary.json` | Same data in structured JSON (no token ledger — see `llm_usage.json`) |
| `llm_usage.json` | Token totals tagged by workflow node (`planner`, `analysis`, `reporter`, …) |
| `supervisor/pool_status.json` | **Primary live status** — per-sim agent ladder, `workflow_phase`, healthy/failed counts |
| `supervisor/state.jsonl` | Pretty-printed checkpoint (single JSON object; name is historical). Prefer `pool_status.json` for “is it done?” |
| `agent_conversation.log` | Full agent dialogue, LLM prompts/responses, routing decisions |
| `planner/master_plan.md` | Multi-sim master plan with per-case prompts and combined analysis plan |
| `campaign/state.json` | Versioned campaign snapshot (spec hash, stages, science contract) |
| `campaign/memory.jsonl` | Deduped errors and later fixes, retrieved by analysis, reporter, supervisor, and planner |
| `campaign/study_notes.jsonl` | Short numeric notes from this run, same retrieval |
| `{label}/state.json` | Per-simulation snapshot (completed stages, bound `hpc/repXX` traj) |
| `{label}/{stage}/inventory.json` | Per-stage artifact index (preprocess → reporter; HITL `pwd` reads this) |
| `{label}/analysis/repXX/inventory.json` | Absolute topology, trajectory, and mapped-residue paths |
| `planner/knowledge_index.json` | Generated knowledge cache for that run (not a source file) |
| `supervisor/execution_report.md` | Stage-by-stage execution summary |
| `reporter/combined_report.html` | Interactive HTML report (multi-sim; includes science banner when family) |
| `analysis/statistical_summary.json` | Cross-simulation statistics table |
| `analysis/classification_features*.csv` | Raw + robust z-score feature matrix (when classified) |
| `analysis/classification_feature_selection.json` | LLM-chosen columns, unavailable columns, reasoning |
| `analysis/classification_dendrogram_heatmap.png` | Ward dendrogram + feature heatmap panel |
| `cross_sim/` | Pre-combined MSA / pocket-map artifacts (multi-sim) |