# SimAgent tutorial — apo / holo pseudokinase panel

This folder is a hands-on run. [`pseudo_apo_holo/`](pseudo_apo_holo/) runs eight
short MD simulations (four human pseudokinases, each apo and holo with ATP+Mg),
then builds comparative plots and one combined HTML report.

## Read these first

Install, the conda environment, and the language model are documented outside
this page. Open the matching doc before the first run.

| If you need | Open |
| ----------- | ---- |
| Conda environment `SimAgentEnv` and `pip install -e .` | [README — Installation](../README.md#installation) |
| Ollama server and the `gpt-oss:20b` model | [docs/OLLAMA_SETUP.md](../docs/OLLAMA_SETUP.md) |
| CLI flags, force fields, and how to write a goal | [README.md](../README.md) |
| How a run moves through folders | [docs/PIPELINE_WORKFLOW.md](../docs/PIPELINE_WORKFLOW.md) |
| SLURM job pool | [docs/POOLS.md](../docs/POOLS.md) |
| What each analysis computes | [docs/ANALYSIS_TOOLS.md](../docs/ANALYSIS_TOOLS.md) |
| Product overview | [docs/PROJECT.md](../docs/PROJECT.md) |
| Agents and workflow graph | [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) |

[docs/CONVENTIONS.md](../docs/CONVENTIONS.md) is for people changing the code.

---

## Table of contents

1. [What this tutorial does](#1-what-this-tutorial-does)
2. [Prerequisites](#2-prerequisites)
3. [Run the example](#3-run-the-example)
4. [Track workflow status](#4-track-workflow-status)
5. [Find results and reports](#5-find-results-and-reports)
6. [Investigate provenance](#6-investigate-provenance)
7. [Re-run analysis or the combined report](#7-re-run-analysis-or-the-combined-report)
8. [Optional: edit the scientific goal](#8-optional-edit-the-scientific-goal)

---

## 1. What this tutorial does

| Item               | Detail                                                                                                            |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Proteins** | ERBB3 (`p21860`), VRK3 (`q8iv63`), MLKL (`q8nb16`), TITIN (`q8wz42`)                                      |
| **Systems**  | Apo (protein only) + holo (protein + ATP + Mg) →**8** simulations                                          |
| **MD**       | 1 ns production, AMBER99SB-ILDN / TIP3P, 310 K, 1 bar, 0.15 M NaCl                                                |
| **Analyses** | Backbone RMSD, Cα RMSF (+ active-site 150–200), Rg, ATP–pocket COM (holo), Cα DCCM (+ apo–holo ΔDCCM), DSSP |
| **Outputs**  | Per-sim plots/reports + combined overlays +`combined_report.html`                                               |

Inputs already in the folder: `p21860.pdb`, `q8iv63.pdb`, `q8nb16.pdb`,
`q8wz42.pdb`, plus [`goal.txt`](pseudo_apo_holo/goal.txt) and
[`run_simagent_nohup.sh`](pseudo_apo_holo/run_simagent_nohup.sh).
The copy on GitHub also includes the analysis plots, tables, and HTML reports
from a completed run. Trajectories (`.xtc`, `.trr`), coordinates (`.gro`),
topologies, and force-field files are not in the repository.

**Pipeline stages (high level):**

```
goal.txt  →  preprocess / simsetup  →  HPC (SLURM pool)
          →  per-sim analysis + report  →  combined analysis + combined report
```

---

## 2. Prerequisites

Finish [installation](../README.md#installation) and
[Ollama setup](../docs/OLLAMA_SETUP.md) once. From the repository root, check
that both are available:

```bash
conda activate SimAgentEnv
curl -s http://127.0.0.1:11434/api/tags
```

The production stage also needs SLURM. The launcher submits those jobs.

---

## 3. Run the example

### Option A — background launcher (recommended)

Activate `SimAgentEnv` first. To change force field, model, or HPC limits, edit the `python SimAgent.py` lines in [`run_simagent_nohup.sh`](pseudo_apo_holo/run_simagent_nohup.sh).

```bash
cd /path/to/simagent
conda activate SimAgentEnv
bash tutorial/pseudo_apo_holo/run_simagent_nohup.sh
tail -f tutorial/pseudo_apo_holo/simagent_nohup.log
```

The script:

- Reads `tutorial/pseudo_apo_holo/goal.txt`
- Sets `--working-dir` to that folder
- Runs `preprocess simsetup hpcjob analysis reporter`
- Uses `--allowed-hpc-jobs 4` and `--hpc-check-interval 10m`
- Writes `simagent_nohup.log` and `simagent_nohup.pid`

### Option B — foreground CLI

```bash
cd /path/to/simagent
conda activate SimAgentEnv

python SimAgent.py \
  --goal "$(cat tutorial/pseudo_apo_holo/goal.txt)" \
  --working-dir tutorial/pseudo_apo_holo \
  --force-field amber99sb-ildn \
  --water-model tip3p \
  --llm-model gpt-oss:20b \
  --llm-base-url http://127.0.0.1:11434 \
  --subtask preprocess simsetup hpcjob analysis reporter \
  --hpc-check-interval 10m \
  --allowed-hpc-jobs 4
```

### Resume after interrupt

If the process dies during HPC or post-processing, restart with the **same**
working directory and `--resume` so finished sims are skipped:

```bash
python SimAgent.py \
  --goal "$(cat tutorial/pseudo_apo_holo/goal.txt)" \
  --working-dir tutorial/pseudo_apo_holo \
  --subtask preprocess simsetup hpcjob analysis reporter \
  --resume \
  --allowed-hpc-jobs 4 \
  --hpc-check-interval 10m
```

Force one label again: `--retry-labels p21860_ATP_MG`.

---

## 4. Track workflow status

**Primary live views** (open these first — they show stage/status directly):

| File                                                                          | What you see                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`supervisor/pool_status.json`](pseudo_apo_holo/supervisor/pool_status.json) | **Best live dashboard.** Per-sim ladder (`preprocessing` → `simsetup` → `hpc` → `analysis` → `reporter`), pool counts (`running` / `pending` / `done` / `failed`), and `workflow_phase` (e.g. `hpc_pool`, `parallel_analysis`, `combined_reporter`). |
| [`supervisor/state.jsonl`](pseudo_apo_holo/supervisor/state.jsonl)           | Full supervisor checkpoint (pretty-printed JSON). Use for resume/debug:`multi_sim_phase`, `hpc_pool`, `parallel_pool`, `completed_sim_states`. Prefer `pool_status.json` for a quick “is it done?” check.                                                                    |

```bash
BASE=tutorial/pseudo_apo_holo

# Live per-sim ladder + phase
python -m json.tool "$BASE/supervisor/pool_status.json" | less

# Workflow phase + high-level fields from the checkpoint
python - <<'PY'
import json
from pathlib import Path
obj = json.loads(Path("tutorial/pseudo_apo_holo/supervisor/state.jsonl").read_text())
st = obj["state"]
print("outer status:", obj.get("workflow_status"))
print("multi_sim_phase:", st.get("multi_sim_phase"))
print("hpc_pool.phase:", (st.get("hpc_pool") or {}).get("phase"))
pp = st.get("parallel_pool") or {}
print("parallel_pool.phase:", pp.get("phase"))
sims = (pp.get("sims") or {})
print("parallel done:", sum(1 for r in sims.values() if r.get("status")=="done"), "/", len(sims))
PY
```

| Other checks     | Command / file                                                                  |
| ---------------- | ------------------------------------------------------------------------------- |
| Live log         | `tail -f tutorial/pseudo_apo_holo/simagent_nohup.log`                          |
| Still running?   | `kill -0 "$(cat tutorial/pseudo_apo_holo/simagent_nohup.pid)" && echo running` |
| Campaign outcome | `tutorial/pseudo_apo_holo/run_summary.md` (and `.json`)                      |
| Per-sim health   | Sections under`## Simulations` in `run_summary.md`                          |
| LLM token use    | `tutorial/pseudo_apo_holo/llm_usage.json`                                      |

Typical multi-sim phases (`pool_status.json` → `workflow_phase` / `state.jsonl` → `multi_sim_phase`):

1. **Prep** — preprocess + simsetup for all cases
2. **HPC pool** — up to `--allowed-hpc-jobs` SLURM jobs at once
3. **Post-HPC** — parallel analysis + reporter, then combined analysis + report

Pool design: [docs/POOLS.md](../docs/POOLS.md).

---

## 5. Find results and reports

Base path: `tutorial/pseudo_apo_holo/`.

### Combined (start here)

The combined report is created by the run. It is not stored in the repository.

- **After the run:** `tutorial/pseudo_apo_holo/reporter/combined_report.html`
- Open it with `xdg-open tutorial/pseudo_apo_holo/reporter/combined_report.html`

| Path                                  | What it is                                                          |
| ------------------------------------- | ------------------------------------------------------------------- |
| `reporter/combined_report.html`     | **Main deliverable** — see callout above                     |
| `analysis/rmsd_overlay.png`         | Cross-sim RMSD overlay                                              |
| `analysis/rmsf_overlay.png`         | Cross-sim RMSF overlay                                              |
| `analysis/rg_overlay.png`           | Radius of gyration overlay                                          |
| `analysis/com_distance_overlay.png` | ATP–pocket COM (holo)                                              |
| `analysis/*_dccm_diff_heatmap.png`  | Apo–holo ΔDCCM per protein (e.g.`VRK3_`, `MLKL_`, `TITIN_`) |
| `analysis/dssp_comparison.png`      | Secondary-structure comparison                                      |
| `analysis/*_stats.csv`              | Summary statistics tables                                           |
| `run_summary.md`                    | Success / health counts for all eight sims                          |

### Per simulation

Each label has its own tree, e.g. `p21860/` (apo) and `p21860_ATP_MG/` (holo):

```
{label}/
  preprocess/          # cleaned / protonated structures
  simsetup/            # topology, box, MDP
  hpc/                 # SLURM scripts + trajectories (md.xtc / mdWrap.xtc)
  analysis/            # rmsd.png, rmsf.png, dccm_heatmap.png, …
  analysis/analysis_summary.jsonl   # machine-readable analysis log
  reporter/report.html # per-sim HTML report
```

Labels in this example: `p21860`, `p21860_ATP_MG`, `q8iv63`, `q8iv63_ATP_MG`,
`q8nb16`, `q8nb16_ATP_MG`, `q8wz42`, `q8wz42_ATP_MG`.

---

## 6. Investigate provenance

Use these files when you need to audit **what** ran, **why**, and **with which
inputs** — without re-reading the full trajectory.

| Layer                 | Path                                        | Use it for                                 |
| --------------------- | ------------------------------------------- | ------------------------------------------ |
| Campaign snapshot     | `campaign/state.json`                     | Spec hash, stages, science contract        |
| Campaign memory       | `campaign/memory.jsonl`                   | Notes / errors remembered across the study |
| Supervisor checkpoint | `supervisor/state.jsonl`                  | Routing, combined phase, pool restore      |
| Master plan           | `planner/master_plan.md`                  | Cross-sim plan and per-case prompts        |
| Campaign dialogue     | `agent_conversation.log`                  | Enrichment, master plan, combined steps    |
| Per-sim plan          | `{label}/planner/execution_plan.md`       | Tools chosen for that system               |
| Per-sim report        | `{label}/supervisor/execution_report.md`  | What completed / warnings                  |
| Per-sim dialogue      | `{label}/agent_conversation.log`          | Preprocess → reporter for one label       |
| Stage inventory       | `{label}/{stage}/inventory.json`          | Indexed artifacts for that stage           |
| Per-sim state         | `{label}/state.json`                      | Completed stages for that simulation       |
| Analysis summary      | `{label}/analysis/analysis_summary.jsonl` | Each metric run + output paths             |
| Combined analysis log | `analysis/execution_report.md`            | Overlay / ΔDCCM / DSSP aggregation        |

Quick greps:

```bash
BASE=tutorial/pseudo_apo_holo

# Did combined analysis finish?
grep -n "combined\|DCCM\|overlay" "$BASE/analysis/execution_report.md" | head

# Which analyses exist for VRK3 apo?
python -c "import json; from pathlib import Path
p=Path('$BASE/q8iv63/analysis/analysis_summary.jsonl')
print([json.loads(l).get('analysis_type') for l in p.read_text().splitlines() if l.strip()])"

# HPC job ids / errors in the conversation log
grep -n "SLURM\|job_id\|FAILED\|Priority" "$BASE/agent_conversation.log" | tail
```

Deeper conventions: [docs/CONVENTIONS.md](../docs/CONVENTIONS.md),
[docs/PIPELINE_WORKFLOW.md](../docs/PIPELINE_WORKFLOW.md).

---

## 7. Re-run analysis or the combined report

When trajectories already exist and you only want new metrics / a fresh report:

**All sims — analysis + per-sim reports + combined:**

```bash
python SimAgent.py \
  --goal "$(cat tutorial/pseudo_apo_holo/goal.txt)" \
  --working-dir tutorial/pseudo_apo_holo \
  --subtask analysis reporter
```

**Combined overlays + `combined_report.html` only** (requires
`{label}/analysis/analysis_summary.jsonl` for each sim):

```bash
python SimAgent.py \
  --goal "$(cat tutorial/pseudo_apo_holo/goal.txt)" \
  --working-dir tutorial/pseudo_apo_holo \
  --subtask analysis reporter \
  --combined-only
```

---

## 8. Optional: edit the scientific goal

Edit [`pseudo_apo_holo/goal.txt`](pseudo_apo_holo/goal.txt) before launching if
you want a different production length, force field, or analysis set. Keep the
protein names and apo/holo component wording clear so the planner can expand to
eight cases.

Natural-language goal tips (framework-wide):
[README — Prompt Engineering Guide](../README.md#prompt-engineering-guide).

Install, Ollama, and the other docs are listed at the top of this page.
