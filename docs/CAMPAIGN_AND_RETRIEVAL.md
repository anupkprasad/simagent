# Campaign contracts and tool retrieval

**This file is the family-scale reliability design.** It explains the compiled
`CampaignSpec`, science completeness gates, and hybrid embedding tool search
added so N-protein studies do not depend on lucky per-sim LLM prompts.

**In this file:** compile-once program, artifact contracts, the shared
analysis protocol (required calculations for every protein), required
shared setup (MSA/pocket before per-protein MD), embedding tool +
knowledge retrieval, `campaign.yaml`, file tools restricted to the
study directory, and per-stage inventory.

**Not in this file:** LangGraph nodes ([ARCHITECTURE.md](ARCHITECTURE.md)),
pool sizing ([POOLS.md](POOLS.md)), analysis theory
([ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md)).

**Terms used in this file**

| Term | Meaning |
|------|---------|
| **Required calculations** | Measurements the active feature pack pins for every protein (pocket COM/orientation when a ligand is declared, torsions, RMSF, DCCM). The LLM cannot skip them. Shared-reference dynamics is a post pin, not a per-sim PCA directory. |
| **Shared analysis protocol** | The same compiled tool list and output directories applied to every protein. |
| **Required shared setup** | Structure-only work done once before per-protein MD (MSA, pocket definition, residue map). |
| **Restricted to the study directory** | File tools can only list/read inside this campaign or the current protein folder. |

---

## Compile once, map many

The natural-language `--goal` is compiled **once** into `state["campaign_spec"]`:

| Field | Role |
|-------|------|
| `mode` | `single_md` (N=1), `multi_protocol` (N>1, same non-family protocol), `family_modular`, or `apo_holo` |
| `shared_analysis_protocol` (`analysis_recipe`) | Same calculations and output dirs for every protein. Non-family runs get a **soft** recipe only when the goal names classic metrics (RMSD, RMSF, Rg, …); a broad “dynamics” goal stays empty so the analysis LLM chooses |
| `post_combined_recipe` / `pin_post_tools` | Combined-stage pins (shared-reference dynamics, apo/holo overlays) |
| `contract` | Required `consensus_*` dirs and feature-column density for family mode |
| `required_calculations` | Measurements that must run for every protein (not optional LLM choices) |
| `required_shared_setup` | Structure-only setup every simulation consumes (MSA, pocket, residue map) |
| `settings.family_detect` | Why family mode was on or off (`score`, `signals`, `reason`) |
| `allow_partial_combined` | Science-report flag only. Dendrogram + heatmap still plot from whatever columns exist |

`campaign.yaml` selects the program:

| Key | Values |
|-----|--------|
| `family_modular` | `auto` (default; ≥2 modular signals, or a comparative phrase plus one signal), `force`, or `off` |
| `feature_pack` | `from_goal` (default: columns and tools named in the goal, any count), or a preset: `shared_pocket_dynamics`, `pocket_chi_rmsf_dccm`, `rmsf_dccm_only`. `paper_sharedpcdyn` is a legacy alias of `shared_pocket_dynamics` |
| `reference_label` | Directory name of the shared-dynamics reference. Empty uses the first system with consensus dihedrals, then a `p17612*` label if one exists |
| `feature_plugins` | Extra calculators merged into the pack (`tool_name`, `output_dir`, `feature_columns`, `stage`: `per_sim` / `post` / `pre`) |

A single system is still the multi-sim tree with N=1 (no combined stage). Resume reads `{base}/planner/campaign_spec.json` before `goal.txt`, so a truncated goal cannot turn family gates off.

Per-sim workers receive `user_goal_original` **and** `campaign_spec`. They do
not invent a new analysis shopping list.

LLM `sim_prompts` that contain `(same as above)` or drop family keywords are
**discarded**. The compact deterministic prompt (identity stanza + shared
science excerpt) is used instead.

## Science completeness

`run_summary.md` now has a **Science completeness** table. Process success
(HTML + `analysis_summary.jsonl`) is not enough for family campaigns.

A system is science-complete only when:

1. The modular directories from the **active feature pack** exist
   (only the descriptors the goal selected, for example `consensus_rmsf`
   and `consensus_DCCM`).
   `consensus_PCA/` is optional and is **not** a per-sim gate. Shared-reference
   dynamics is a post pin and writes campaign-level features, not a per-sim
   `consensus_PCA/` requirement.
2. Every **required calculation** has left its artifact contract (including
   pocket metrics: `reference_pocket_metrics.json` /
   `ligand_pocket_distance.csv`, typically under `analysis/repXX/` or
   `analysis/reference_pocket/`)

A family campaign prefers the columns compiled from the goal (or from `gold_columns`) when they are filled. Combined
Ward still runs on any usable table (≥2 systems and ≥2 numeric columns) so a
generic goal that only asks for RMSF / χ₁ / entropy still gets a
dendrogram+heatmap. Missing gold columns are a **warning** in `run_summary`
and a **science banner** in `combined_report.html`, not a plot block
(`--allow-partial` only changes the science-OK flag).

Resume / parallel workers treat a family system as incomplete until those
artifacts exist — HTML alone does not skip re-analysis.
## Shared analysis protocol (comparative / family studies)

When the goal is a comparative protein-family study, every system runs the
**same** analysis protocol (compiled once, applied to all). The LLM does not
invent a different tool list per protein. The protocol is:

1. `calculate_consensus_pocket_metrics` → COM + axis-angle (omitted when no ligand is declared)
2. `calculate_consensus_torsions` → `consensus_dihedrals/` (omitted by `rmsf_dccm_only`)
3. `calculate_consensus_rmsf_features` → `consensus_rmsf/`
4. `calculate_consensus_dccm_features` → `consensus_DCCM/`

`run_independent_dynamics_fel` (dihedral PCA → `consensus_PCA/`) is available
as a tool and is batchable, but it is **not** a required per-sim pin.
A goal that asks for shared-reference dihedral PCA pins
`compute_shared_pka_ref_dyn_features` at **post-combined**, using
`reference_label` (the reference named in the goal, otherwise the first
system). The optional `shared_pocket_dynamics` preset always includes that step.

`--reuse-hpc` also skips LLM preprocess/setup planning (deterministic fallback).

Resume/skip checks use the science contract. HTML + RMSD no longer mark a
family system complete. The shared analysis protocol is complete when the
pack’s `consensus_*` directories **and** any required pocket-metric files exist
(RMSD is not required; `consensus_PCA/` is not required). Combined Ward prefers
the pack’s gold columns — empty columns are recorded as unavailable instead.
Family **post-combined** is done only when `classification_features.csv`
exists. Generic multi-sim combined can finish on overlays and `*_stats.csv`
alone. A `success: false` marker does not block the reporter when those
artifacts are already on disk. Combined analysis re-entry is capped (default 2)
so a false gate cannot loop until the graph recursion limit.
Per-sim **success is artifact-gated**. `plot_md_data` / other `plot_*`
failures are warnings. A failed H-bond/SASA plot does not fail the
simulation or trigger a 3× replan. Retries reuse the compiled plan,
skip plots, and skip required calculations whose `consensus_*` dirs (or
pocket metric files) already exist.

**Required calculations** are registered on the **per-simulation** tool
executor (including `calculate_consensus_pocket_metrics`). Batch helpers
such as `run_consensus_pocket_metrics_batch` remain combined/shared tools.

Family mode also **forces** the pre-combined *stage* on. The LLM still
**compiles** which combined tools run there (pocket definition, MSA, or any
other shared calculation). The engine then executes that compiled protocol
once. A hardcoded MSA+pocket chain is only the fallback if compile is empty.

## Embedding tool search

When the goal is **not** a compiled shared analysis protocol (or for planner context),
tools are retrieved instead of dumping the full catalog into the LLM:

1. **Hashed character n-grams** (256-d, MD5-stable, no extra model)
2. **Lexical Jaccard** on tokens
3. Hybrid score: `0.65 * cosine + 0.35 * overlap`
4. **Ollama dense embeddings** when `embed_model` is set (`nomic-embed-text`
   by default). Vectors are cached for the process. If one request fails, that
   item stays on the hashed score and the rest of the batch still blends.
   After the server is unreachable, the process stays on hashed n-grams.

Required calculations for the study are always included, then the top-k (~16)
ranked tools.

This is how general-purpose planning stays fast and accurate: the LLM sees
only the tools that match the goal, not every RMSD/H-bond/SLURM helper.

The same embedder retrieves **knowledge chunks**. Planner context is
`[kb:category.doc#heading]` citations, not a 6–10k dump of every manual.
The index is generated at `agentic/planner/knowledge_index.json` (next to
`agentic/planner/knowledge/`) and copied into the campaign `planner/` folder.
That cache is local; it is not part of the published tree.

```bash
# optional override; campaign.yaml embed_model is used when this is unset
export AGENTIC_EMBED_MODEL=nomic-embed-text
```

`retrieval_k` (planner tool top-k) and `embed_model` live in `campaign.yaml`.
`agent_retrieval` sets how many knowledge chunks, memory episodes, and study
notes each of analysis, reporter, and supervisor sees before its model call.
The same search is not kinase-specific: the query is the user goal plus any
error text.

## Agent retrieval and routing

| Agent | What the model receives |
|-------|-------------------------|
| Planner | Top-k tools (required calculations pinned) and knowledge chunks |
| Analysis | Knowledge, prior errors for this campaign, and study notes, before planning or feature selection |
| Reporter | The same block inside the literature review and the final impression |
| Supervisor | Once per system, only when analysis left an error and the next node would be `final_report` |

The supervisor model may return only `agent_retrieval.supervisor.allowed_destinations`
(`analysis`, `reporter`, or `final_report`). The graph checks that name. One
decision may send analysis back even if that stage was already ticked done.
If the model is unavailable, the existing route is kept.

Study notes are `campaign/study_notes.jsonl`: the system label plus numeric
fields already present in `analysis/analysis_summary.jsonl`. They are
retrieved with the same embedder. They are not a fixed feature schema.

## campaign.yaml

One file for mapping / retrieval / HITL knobs. Agent YAMLs stay for prompts.

Precedence (later wins): shipped `agentic/campaign/campaign.yaml` →
`{working-dir}/campaign.yaml` → `--campaign-yaml PATH` → `AGENTIC_*` env →
CLI `--HITL`.

| Key | Default | Role |
|-----|---------|------|
| `conservation_metric` | `similarity` | MSA column score (similarity / identity / blosum / coverage) |
| `min_conservation` | `0.5` | Score threshold |
| `min_coverage` | `0.25` | Column occupancy floor |
| `pocket_cutoff_A` | `15` | Reference ligand shell (Å) |
| `lobe_method` | `auto` | N/C lobe split |
| `retrieval_k` | `16` | Planner tool top-k (knowledge uses the per-agent `knowledge_k`) |
| `embed_model` | `nomic-embed-text` | Dense blend; hashed n-grams if the model does not answer |
| `agent_retrieval` | per agent | `knowledge_k`, `memory_k`, `study_k`; supervisor destinations |
| `gold_columns` | paper 9 | Preferred Ward columns (not a plot gate) |
| `hitl` | `null` | `error` / `all` if CLI `--HITL` is omitted |

A family run can change 0.5 / 0.25 / 15 Å without editing Python.

## File tools and inventories

Field agents (not only HITL) can list, search, and read the file inventory.
Paths are **restricted to this study’s directories** (the campaign base or
the current protein folder). They cannot read files outside the study.
`write_file` stays HITL-only.

Prior errors and fixes from this campaign are stored in
`campaign/memory.jsonl` at the **campaign root** (not under each protein
folder). A repeated stage + tool + error updates the existing episode.
When a later attempt succeeds, the open episode receives a fix. Analysis,
reporter, supervisor, and planner retrieve the top episodes with the same
hybrid embedder.

With `--HITL all`, a human can review (1) the compiled analysis protocol,
(2) the shared MSA/pocket mapping, and (3) the cross-system comparison,
before the next stage starts.

After each stage the engine writes `{stage}/inventory.json` (absolute
topology, trajectory, `pocket_mapped`, `global_mapped`, and the files in
that folder). HITL `pwd` prints that inventory. Analysis resolves
`pocket_mapped.json` from inventory first; the parent-folder walk is the
fallback. Discovery looks under both the current protein folder and the
campaign base (`cross_sim/`, `analysis/`), so combined-reporter inventories
at `{base}/reporter/` also resolve shared maps.

Per-rep analysis still writes `analysis/repXX/inventory.json` after wrap
and fan-out. Campaign snapshots remain `campaign/state.json` and
`{label}/state.json`. Token usage is written to `llm_usage.json` and
tagged by workflow node (`planner`, `analysis`, `reporter`, …).
## Pre-combined: LLM compiles, engine executes

Pre-combined is **not** a hardcoded kinase-only pocket path.

| Role | Who | What |
|------|-----|------|
| Compile | LLM (once) | Chooses combined/shared tools: `define_reference_consensus_pocket`, a custom pocket tool, MSA, phylo, or any other structure-only setup every sim will consume |
| Required shared setup | Family `CampaignSpec` | MSA + reference pocket + residue map + MSA plots are **added if omitted**, never used to delete LLM extras |
| Execute | Engine | Runs the compiled shared setup protocol identically on retry |
| Fallback | Deterministic MSA+pocket | Only if the LLM plan is empty or leaves no `cross_sim/` artifacts |

Traj overlays, Ward clustering, and `run_consensus_pocket_metrics_batch` are
blocked in pre-combined (they need trajectories). They run in post-combined
or per-sim analysis.

Generic (non-family) campaigns: the LLM may turn pre-combined on and pick
any allowed combined tools; nothing is pinned.

Code: `agentic/campaign/` (`campaign.yaml`, `config.py`, `pre_combined.py`),
`agentic/retrieval/` (tools + knowledge), `agentic/utils/sandbox_files.py`,
`src/analysis/inventory.py`.
