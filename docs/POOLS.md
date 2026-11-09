# Parallel execution pools

**This file is the concurrency guide.** SimAgent uses two related pools
for multi-simulation work: a local worker pool on this machine, and an HPC
/ SLURM pool on the cluster. Read it for what runs in parallel, how worker
and job counts are chosen, CLI flags, resume, and on-disk pool state.

**In this file:** local workers (prep + post-HPC analysis), automatic CPU /
memory sizing, Ollama concurrency cap, HPC/SLURM submit-and-poll, HITL on
job failures, `pool_status.json` / `state.jsonl`.

**Not in this file:** LangGraph nodes ([ARCHITECTURE.md](ARCHITECTURE.md)),
run walkthrough and flag examples ([PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)),
SLURM client libraries and script names ([TOOLS.md](TOOLS.md#hpc--remote-execution)),
analysis observables ([ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md)).

| Pool | Where it runs | What it parallelizes |
|------|---------------|----------------------|
| **Local worker pool** | This machine | Preprocess + simsetup, and post-HPC analysis + reporter |
| **HPC / SLURM pool** | Cluster | Production MD jobs (`mdrun`) |

Both pools share `pool_status.json` and `{base}/supervisor/state.jsonl`.
Combined analysis and HITL stay sequential.

---

## Local worker pool

When a multi-simulation workflow runs **independent per-simulation stages**
(prep or analysis+reporter), SimAgent can execute them concurrently on the
local machine. Worker count is chosen **automatically** from CPU and memory.

### What runs in parallel

| Stage | Parallel? | Notes |
|-------|-----------|-------|
| Preprocess + simsetup (pre-HPC) | Yes | Up to N workers; then the HPC pool submits SLURM jobs |
| Analysis + reporter (post-HPC / analysis-only) | Yes | Skips sims with artifacts already on disk |
| HPC (GROMACS MD) | Yes (SLURM) | Separate HPC pool below; also auto-sized when `--allowed-hpc-jobs` omitted |
| Combined analysis + report | No | Runs once after all per-sim work completes |
| Human-in-the-loop (`--HITL`) | No | Falls back to sequential execution |

### Automatic resource detection

The framework probes:

- **CPU:** `os.cpu_count()` minus 2 cores reserved for orchestrator + Ollama
- **Memory:** `/proc/meminfo` on Linux (or `psutil` if installed)

It computes:

```
workers = min(floor(avail_cpus / cpus_per_job),
              floor(avail_mem / mem_gb_per_job),
              user_cap,
              llm_concurrency,   # prep / analysis only
              pending_simulations)
```

For **prep** and **analysis+reporter**, worker count is also capped by
`--llm-concurrency` (default `auto` = 4) so fewer simulations block on the
shared Ollama server. Set `OLLAMA_NUM_PARALLEL` on the Ollama host to the same
value (see [OLLAMA_SETUP.md](OLLAMA_SETUP.md)).

Default per-job estimates:

| Phase | CPUs | RAM |
|-------|------|-----|
| Prep (preprocess + simsetup) | 2 | 4 GiB |
| Analysis + reporter | 2 | 3 GiB |
| HPC pool (SLURM slots) | 1 | 0.5 GiB |

### CLI

```bash
python SimAgent.py \
  --goal "..." \
  --working-dir ./study \
  --pdb-list a.pdb b.pdb c.pdb \
  --parallel-workers auto \
  --llm-concurrency auto \
  --subtask analysis reporter
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--parallel-workers auto` | `auto` | Auto-detect; use integer to cap (`1` = sequential) |
| `--parallel-mem-gb` | phase default | Override GiB RAM estimate per worker |
| `--parallel-cpus` | phase default | Override CPU cores estimate per worker |
| `--llm-concurrency auto` | `auto` (4) | Cap workers for LLM-heavy phases; match `OLLAMA_NUM_PARALLEL` |
| `--allowed-hpc-jobs` | auto | SLURM concurrency when omitted and `--parallel-workers auto` |

Auto mode caps workers at **8** by default (override: `AGENTIC_PARALLEL_MAX_AUTO=16`).

Environment overrides:

```bash
export AGENTIC_PARALLEL_WORKERS=4
export AGENTIC_PARALLEL_MEM_GB=3
export AGENTIC_PARALLEL_CPUS=2
export AGENTIC_LLM_CONCURRENCY=4
```

### Ollama server

Parallel workers share one LLM endpoint. On the Ollama host, set (see [OLLAMA_SETUP.md](OLLAMA_SETUP.md)):

```bash
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_QUEUE=512
export OLLAMA_KEEP_ALIVE=-1
```

Restart the Ollama server after changing these. Use the same value for
`--llm-concurrency` (or `AGENTIC_LLM_CONCURRENCY`).

### Behaviour

1. **Master plan** — planner builds per-sim prompts as usual.
2. **Parallel pool** — up to N worker processes each run one simulation's agents.
3. **Poll** — main process refreshes `{base}/supervisor/pool_status.json` and
   overwrites `state.jsonl` periodically (no terminal/log spam).
4. **Combined work** — after all per-sim jobs finish, combined analysis/reporter runs at base dir.

Monitor live workers:

```bash
# Preferred: compact live ladder + phase
watch -n 5 cat my_study/supervisor/pool_status.json

# Full checkpoint (pretty-printed single JSON object — not append-only lines)
python -c "import json; from pathlib import Path
st=json.loads(Path('my_study/supervisor/state.jsonl').read_text())['state']
print('phase', st.get('multi_sim_phase'))
print('parallel', (st.get('parallel_pool') or {}).get('phase'))
"
```

Sims with existing outputs (`analysis_summary.jsonl`, `report.html`, or simsetup files) are **skipped**.

Use `--resume` to continue after interruption; stale `running` records are re-queued.

On **Ctrl+C / kill / disconnect**, the framework saves an interrupt checkpoint:
`running` workers become `pending`, `multi_sim_progress` is synced from pool +
disk, and `workflow_status` is set to `in_progress:interrupted`. Run again
with `--resume` and the same `--working-dir`.

While the pool runs, **`pool_status.json` is the live source of truth**. It reflects
the active stage:

- **parallel prep/analysis** — worker slots (`parallel_pool`)
- **HPC pool** — per-sim `prep` + `hpc` status and SLURM job IDs (matches terminal HPC summary)
- **sequential / combined post-HPC** — full agent ladder from disk + `multi_sim_progress`

Before each `state.jsonl` write, the framework syncs `parallel_pool` and on-disk
artifacts into `multi_sim_progress` so the checkpoint does not leave completed
sims stuck as `pending` after a successful run.

### State

Pool metadata lives in `{base}/supervisor/state.jsonl` under `parallel_pool`:

```json
{
  "phase": "analysis",
  "max_workers": 4,
  "resource_estimate": {
    "max_workers": 4,
    "limiting_factor": "memory",
    "cpus_per_job": 2.0,
    "mem_gb_per_job": 3.0
  },
  "sims": {
    "p21860": { "status": "done", "working_dir": "..." }
  }
}
```

`multi_sim_phase` is `"parallel_pool"` while workers run.

Pool events are written to `{base}/agent_conversation.log` with agent name `parallel_pool`.

---

## HPC / SLURM pool

When a workflow requests **preprocess + simsetup + HPC + analysis + reporter**
together, the supervisor uses an HPC pool that **waits for SLURM completion**
before starting analysis/reporter. This applies for **any campaign size**
(N=1 or N>1).

Prep runs across simulations and jobs are submitted up to N at a time. Each
simulation lives under `{base}/{label}/` (label from PDB stem / component case).
Analysis never starts before `md.xtc` (or equivalent) exists.

### Behavior

1. **Master plan** — planner builds per-sim prompts as usual.
2. **Prep** — preprocess → simsetup (local worker pool above when multi-sim). No SLURM jobs during this phase.
3. **Submit (parallel up to N)** — after all prep is complete, jobs are submitted as slots free (up to `--allowed-hpc-jobs`).
4. **Poll** — while jobs run, the workflow sleeps and re-checks SLURM every `--hpc-check-interval`
   (default **2h** / 7200s).
5. **Post-HPC** — after all jobs complete (or are skipped because trajectories already exist),
   per-sim analysis → reporter runs (local worker pool), then combined
   analysis/reporter at project base **only when** `len(sim_prompts) > 1`.

Sims with existing trajectories under `{sim}/hpc/*.xtc` are **skipped** for submission.

### CLI

```bash
python SimAgent.py \
  --goal "..." \
  --pdb-list a.pdb b.pdb c.pdb \
  --allowed-hpc-jobs 5 \
  --hpc-check-interval 2h
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--allowed-hpc-jobs` | 5 | Max concurrent SLURM jobs in the pool |
| `--hpc-check-interval` | `2h` | Poll interval (`2h`, `120m`, `7200`); use a shorter value while debugging |
| `--resume` | off | Restore `{base}/supervisor/state.jsonl` and re-enter pool monitoring |
| `--HITL error` | — | Pause on submit/SLURM failures at the HPC pool checkpoint |

Set `hpc_pool_disabled=True` in state (or omit analysis/reporter from `--subtask`) to
skip the pool and use the legacy per-agent HPC path. `download_results` is **not**
part of the default pool or HPC agent plan; analysis reads trajectories from
`{sim}/hpc/` on the shared filesystem unless the user explicitly asks to download.

### Hybrid execution / resume

- While the process is **running**, the pool wait node sleeps for the check interval, then polls
  `sq --me` (or `squeue -u $USER`).
- If the process is **killed**, restart with the same `--working-dir` and `--resume`. Pool state
  in `state.jsonl` restores job IDs and prep status; the supervisor syncs SLURM and continues.

### HITL on failures

Submit failures or SLURM terminal failures (`FAILED`, `TIMEOUT`, etc.) set `awaiting_hitl` on the
pool. With **`--HITL error`** or **`--HITL all`**, the workflow pauses at `human_hpc_pool_check` where you can:

- `status` — show pool summary
- `continue` / `retry` / `resubmit` — clear the pause, re-sync SLURM, and resume submission/monitoring

### State

Pool metadata lives in `{base}/supervisor/state.jsonl` under `hpc_pool`:

```json
{
  "max_concurrent": 5,
  "check_interval_sec": 7200,
  "sims": {
    "p23458": {
      "prep_status": "done",
      "hpc_status": "running",
      "job_id": "46388398"
    }
  }
}
```

`multi_sim_phase` is `"hpc_pool"` during prep/submit/monitor, then `"executing_sims"` with
`post_hpc_analysis_only=true` for analysis/reporter.
