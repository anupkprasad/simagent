# Coding Conventions

**This file is the contributor rulebook.** Follow it when you add agents,
state fields, tools, or documentation. It is *how we write code in this
repo*, not a description of the running system.

**In this file:** `MDState` rules, agent layout, pluggable registries, LLM
calls, graph edges, multi-sim / HITL / resume conventions, MD and HPC
defaults, path handling, how to document a mechanism.

**Not in this file:** LangGraph topology ([ARCHITECTURE.md](ARCHITECTURE.md)),
product overview ([PROJECT.md](PROJECT.md)), run walkthrough
([PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)), dependency versions
([TOOLS.md](TOOLS.md)), analysis theory ([ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md)),
pool sizing ([POOLS.md](POOLS.md)).

---

## State-Driven Architecture

- **All data flows through `MDState`** (a single TypedDict in `agentic/state.py`).
  Agents never communicate directly — they read from and write to state fields.
- **Field naming:** `{stage}_{artifact}` — e.g. `cleaned_pdb`, `setup_report`,
  `analysis_results`, `trajectory_path`.
- **Routing:** the `next_node` field tells the supervisor where to go next.
  Each agent sets it before returning.
- **Error tracking:** append to `errors` (fatal) and `warnings` (non-fatal) lists.
- **New fields:** add to `MDState` in `agentic/state.py` with an `Optional[...]`
  type annotation and a descriptive inline comment.

### Key state field groups

| Group | Fields | Written by |
|-------|--------|-----------|
| Input | `raw_pdb`, `user_goal`, `structured_prompt` | Supervisor / CLI |
| Structure | `cleaned_pdb`, `structure_request`, `domain_context`, `sim_case` | Preprocessing |
| Setup | `topology`, `coordinates`, `mdp_files`, `chain_residue_map` | SimSetup |
| HPC | `job_id`, `job_status`, `trajectory_path` | HPC |
| Analysis | `analysis_results` | Analysis |
| Report | `reporter_output` | Reporter |
| Multi-sim | `is_multi_simulation`, `sim_prompts`, `completed_sim_states`, `multi_sim_phase` | Supervisor |
| Plan | `execution_plan`, `*_instructions` fields | Planner |

---

## Agent Pattern

Every agent follows the same structure:

```
agentic/{agent_name}/
    __init__.py
    {agent_name}_agent.py   # Main class with a *_node() method
    tools.py                # Tool functions (auto-discovered by planner)
    schemas.py              # Pydantic input/output schemas
    config.yaml             # Agent-level configuration
```

### Adding a New Agent

1. Create the directory under `agentic/`.
2. Implement a class with a method `{agent_name}_node(self, state: MDState) -> MDState`.
3. Register the agent in `agentic/supervisor/config.yaml`:
   ```yaml
   agents:
     agent_name:
       capabilities: ["what_it_does"]
       input_requirements: ["fields_it_reads"]
       output_provides: ["fields_it_writes"]
       skip_conditions: ["when_to_skip"]
   ```
4. Add the node and edges in `agentic/workflow.py`.
5. Tools in `tools.py` are auto-discovered by `planner/tools_registry.py` — no
   manual registration needed. Decorate tool functions with LangChain `@tool`.

### Agent Output Isolation

Each agent writes to its own subdirectory:

```
working_dir/{agent_name}/
```

Use `state["working_directory"]` as the base, and the agent-specific
`{agent}_dir` field (e.g. `preprocess_dir`, `analysis_dir`) for the full path.
Use `SecureFileManager` from `agentic/utils/` for all file operations.

---

## Pluggable Registry Pattern

New structure sources and domain sources are added as registry plugins, not as
edits to the main pipeline.

### Adding a new structure source (e.g. PDB redo)

1. Create `src/preprocess/structure_sources/pdredo.py` implementing the
   base class from `structure_sources/base.py`.
2. Register it in `src/preprocess/structure_sources/registry.py`.
3. Set priority order in the registry (AlphaFold is default, RCSB is fallback).

### Adding a new domain source

1. Create `src/preprocess/domain_sources/yourdb.py` implementing the base from
   `domain_sources/base.py`.
2. Register in `src/preprocess/domain_sources/registry.py`.

No changes to the Preprocessing agent or tools needed — `acquire_protein_structure`
uses the registry automatically.

---

## LLM Integration

- **LLM is required by default.** Startup fails unless Ollama is reachable or an
  API key is set. Pass `--no-llm` for heuristic / registry-based plans only.
- **Always provide a mid-run fallback.** If an LLM call fails after startup,
  agents should still have a deterministic code path where practical.
- **Mock mode:** set `_is_mock_mode = True` for testing without an LLM server.
- **Structured prompts:** include an explicit output-format instruction in every
  LLM prompt (e.g. "Return JSON with keys: ...").
- **Configuration:** `agentic/supervisor/config.yaml` for the agent registry;
  `llm_config.py` for model/URL defaults.
- **Coercion:** LLM plan fields may return lists or dicts instead of strings.
  Use `_coerce_plan_text()` in `supervisor_agent.py` to normalise them.

---

## Workflow Graph

- **Hub-and-spoke:** all agents return to the supervisor.
  Never add direct agent-to-agent edges.
- **Node wrapping:** use `_wrap_node()` to set `current_node` for tracking.
- **Conditional routing:** use LangGraph `add_conditional_edges` for branching.
- **Recursion limit:** 25 iterations. If the graph loops more, there is a bug.

---

## Multi-Simulation Conventions

- `sim_prompts` is the authoritative count of simulations (not `pdb_list`).
  `len(sim_prompts)` = total cases including all component variants.
- Each simulation gets an isolated `working_dir/{label}/` directory.
- **Multi-replicate (`--rep-num N`, N>1):** preprocess + simsetup stay once per
  label; production MD forks into `{label}/hpc/rep01`…`repNN` with deterministic
  seeds (`replicate_base_seed + k - 1`). Analysis fans the same plan across
  `{label}/analysis/repXX/` and writes mean±std under `{label}/analysis/avg/`.
  Combined collectors prefer `analysis/avg/` so family overlays stay one curve
  per chemical system (not one per replicate). `rep_num=1` keeps legacy flat
  `{label}/hpc/` and `{label}/analysis/`.
- Per-sim `MDState` fields (`raw_pdb`, `cleaned_pdb`, `topology`, …) are
  reset between simulations. Prep and post-HPC analysis/reporter may run as
  a local worker pool; production MD uses the SLURM HPC pool
  ([POOLS.md](POOLS.md)). Combined analysis and `--HITL` stay sequential.
- Skipped simulations are recorded in `completed_sim_states` with `skipped=True`
  and a `skip_reason` string — they are not silent failures.
- After all sims, `multi_sim_phase` may advance to `combined_analysis` /
  `post_combined` then `combined_reporter` when the master plan sets
  `run_post_combined` (legacy `run_combined_analysis` is an alias for post).
  Optional `pre_combined` runs **before** the per-sim pool when
  `run_pre_combined` is set (`n_sims > 1`), writing `{base}/cross_sim/`.
- Planner master-plan fields: `run_pre_combined` / `pre_combined_plan`,
  `run_post_combined` / `post_combined_plan`.
- **`multi_sim_base_dir`** is the project root (`--working-dir`). Per-simulation
  paths in `sim_prompts` are always `{base}/{label}/`.
- **`--resume`** restores `sim_prompts`, `completed_sim_states`, and loop
  progress. A normal re-run without `--resume` regenerates the master plan and
  restarts the per-sim loop (see `docs/ARCHITECTURE.md` directory layout).
- **`campaign.yaml`** is the source of truth for mapping / retrieval / gold
  columns / HITL defaults, and for study shape: `family_modular`
  (`force` \| `auto` \| `off`), `feature_pack`, `reference_label`, and
  `feature_plugins`. Agent `config.yaml` files stay for prompts. Env
  `AGENTIC_*` and CLI `--HITL` / `--campaign-yaml` override. See
  [CAMPAIGN_AND_RETRIEVAL.md](CAMPAIGN_AND_RETRIEVAL.md).

### Workflow state files (`state.jsonl` + `pool_status.json`)

The framework persists a checkpoint at `{working_dir}/supervisor/state.jsonl`.
Despite the `.jsonl` suffix, the file is a **single pretty-printed JSON object**
(`{"timestamp", "workflow_status", "state": {...}}`) overwritten on each save —
not an append-only NDJSON stream. In multi-sim mode, each simulation also mirrors
state under `{base}/{label}/supervisor/state.jsonl` while that sim is active.

Checkpoints are **content-compact** (bulky fields omitted or truncated) but
**pretty-printed** (`indent=2`) so editors can syntax-highlight them. Full
`user_goal` / campaign specs live on disk under `campaign/`; per-sim analysis
matrices and reports stay under `{label}/`. Orchestration only needs labels,
dirs, agent ladder, and **health**.

| File | Scope | Purpose |
|------|--------|---------|
| `{base}/supervisor/pool_status.json` | Project root | **Preferred live view** — per-sim agent ladder + `workflow_phase` + health counts |
| `{base}/supervisor/state.jsonl` | Project root | Resume checkpoint: enrichment, master plan, pools, `multi_sim_progress`, combined phase |
| `{base}/{label}/supervisor/state.jsonl` | One simulation | Per-sim progress: validation, execution plan, analysis/reporter outputs, errors |

On every persist, `multi_sim_progress` is refreshed from the parallel/HPC pool and
on-disk artifacts **before** writing `state.jsonl`, then `pool_status.json` is
rewritten from the same state.

**Per-sim health (fatal vs soft):**

- A simulation is **healthy** when production MD finished with usable topology +
  trajectory (`md.tpr` + production traj, finished `md.log`, or `--reuse-hpc`).
- **Fatal** HPC/prep failure → that sim is marked `failed`; analysis and reporter
  are **skipped for that system only**; other sims continue; combined science uses
  healthy labels only.
- Soft analysis/report issues (plot tools, literature rate limits) stay warnings
  and do **not** fail the simulation.

**Startup behaviour:**

1. If `state.jsonl` exists, the workflow loads the latest snapshot and restores
   artifact paths (trajectories, analysis results, plans) so agents can skip
   completed work.
2. **Fresh re-run** (no flags): multi-sim loop bookkeeping is cleared; master plan
   is regenerated; per-sim analyses may re-run.
3. **`--resume`**: restores `sim_prompts` / `completed_sim_states`; skips sims
   already marked successful; retries failures only.
4. **`--combined-only`**: discovers sims from per-sim `analysis_summary.jsonl`;
   runs combined analysis + reporter only.
5. **`--HITL all` re-run** when all per-sim `analysis/` folders already contain data:
   auto-enables combined-only mode so you can chat at analysis / reporter checkpoints
   without re-running every simulation. (Default runs are fully automatic; use
   `--HITL all` or `--HITL error` to enable checkpoints.)

### Human-in-the-loop (CLI)

| Flag | Default | Behaviour |
|------|---------|-----------|
| *(none)* | — | Fully automatic; checkpoints are skipped |
| `--HITL error` | — | Pause on stage failures, max retries, or HPC pool submit/SLURM errors |
| `--HITL all` | — | Pause after preprocess, setup, HPC, analysis, reporter (and HPC pool when active) |

State fields: `human_in_loop` (bool), `hitl_mode` (`error` \| `all`), `error_triggered_hitl`
(set when a failure routes to a checkpoint in `--HITL error` mode).

Control-flow fields (`multi_sim_phase`, retry counters) are reset on a normal
re-run unless `--resume` is set. Always check `workflow_status` and `errors` in
`state.jsonl` when debugging a partial run.

Also mirrored: `execution_report.md` in each `supervisor/` directory (human-readable
progress summary).

### Conversation logs (multi-sim)

| Log file | Scope |
|----------|--------|
| `{base}/agent_conversation.log` | Campaign only: user goal, enrichment, **master plan** summary, pool milestones, combined analysis/reporter (N>1) including real `analysis.combined_planning` / `reporter.combined_planning` LLM calls (tools + metadata) and subsequent tool steps |
| `{base}/{label}/agent_conversation.log` | Per-simulation validation, execution plan, preprocess → reporter |
| `{base}/planner/master_plan.md` | Short campaign summary |
| `{base}/{label}/analysis/execution_plan.json` | Per-sim analysis LLM plan |
| `{base}/analysis/execution_plan.json` | Combined analysis LLM plan (when N>1) |
| `{base}/reporter/execution_plan.json` | Combined reporter LLM plan (when N>1) |
| `{base}/{label}/planner/execution_plan.md` | Per-sim execution plan |

The framework switches the active log with `set_log_file()` when entering or
leaving the per-sim loop. Combined analysis/reporter always re-bind the base
log before planning and execution. Do not write per-sim agent output to the base log.

**Combined analysis/reporter** use the same agent classes as per-sim, but with
`include_combined_tools=True` and LLM planning prompts that list combined tool
metadata. Deterministic pipelines remain as fallbacks if LLM planning fails.

Classification/clustering runs in the combined phase **only** when the user
explicitly asks for it. Mentions of “HPC cluster” do not count.

---

## Run Summary Convention

`src/utils/run_summary.py` exports:
- `build_run_summary(final_state, working_dir, goal, pdb_list, config)` → dict
- `write_run_summary(working_dir, summary)` → writes `run_summary.json` + `run_summary.md`
- `format_run_summary_terminal(summary)` → terminal-printable string

Call at workflow exit in `agentic/cli.py`. Do **not** use `len(pdb_list)` for
the total simulation count — use `build_run_summary` which reads from `sim_prompts`.

---

## MD Defaults

| Setting | Default | Override |
|---------|---------|---------|
| Engine | GROMACS 2025.4 | — |
| Force field | AMBER99SB-ILDN | `--force-field` |
| Water model | TIP3P | `--water-model` |
| Temperature | 310 K | goal text |
| Pressure | 1 bar | goal text |
| NaCl concentration | 0.15 M | goal text |
| Box type | cubic, 1.2 nm buffer | goal text |
| PDB file extensions | `.top` (topology), `.gro` (coordinates) | — |

"Protein only" means only the protein component — **not** vacuum. All systems
are solvated by default. Only use vacuum if the user explicitly says so.

---

## HPC Conventions

- SLURM script name: `{job_name}_run.sh` — do not use `slurm_job.sh`.
- Script path is resolved from `state["job_script"]` or the pattern above.
  Never trust an LLM-generated script filename directly.
- Job names must be unique per simulation (derived from the `label` field).

---

## File and Path Handling

- Use `os.path.join()` with the agent directory fields from state.
- Normalise paths before storing them in state.
- Track all generated files in `file_registry` and `generated_files` state fields.
- `pdb_summary` in state may be `None` (initialised that way in `workflow.py`).
  Always use `state.get("pdb_summary") or ""` before string concatenation.

---

## Documenting mechanisms

When you add or change an important **simulation/analysis mechanism** (a
workaround for a GROMACS/format limitation, a mapping layer, a pool, a
selection translator), write it up under `docs/` — not only in chat or
inline comments.

1. Put it in an **existing descriptive file** with related material. Do not
   add a new page per feature. Examples:
   - GROMACS / analysis / selection mechanisms → [TOOLS.md](TOOLS.md)
   - Local workers and SLURM concurrency → [POOLS.md](POOLS.md)
   - Per-tool observables and output names → [ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md)
   - Agent order, directories, files, flag examples → [PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)
2. Every `docs/*.md` file starts with a **purpose paragraph**: what the file
   is for, what it contains, and which sibling pages to use instead.
3. Explain the problem, the design, pipeline insertion points, on-disk
   artifacts, and what *not* to do.
4. Add a one-paragraph pointer from [ARCHITECTURE.md](ARCHITECTURE.md)
   (the relevant agent) if the change is not already obvious from the file
   you edited.
5. Keep the code comment short; the doc is the source of truth for “why”.

---

## General

- Keep imports at module top level.
- Use Python logging (`logger = logging.getLogger(__name__)`), not print statements.
- YAML for configuration, JSON for runtime data exchange.
- Retry counts per agent (`{agent}_retry_count` fields) prevent infinite retry loops.
- Pydantic schemas in `schemas.py` define inputs/outputs per agent stage.
