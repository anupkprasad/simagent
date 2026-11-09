# SimAgent implementation plan

Maintainer backlog. This page is not part of the user documentation index.

**This file is the backlog for making SimAgent reliable at family scale and
usable as a general MD agent.** Items come from the robustness 37×2 reviews
(run_01 / run_02), the orchestration and mapping work, and the capability
review (RAG, embeddings, memory, config, HITL, inventory).

**In this file:** prioritized work (P0–P3), why each item exists, and what
“done” looks like.

**Not in this file:** LangGraph node order ([ARCHITECTURE.md](ARCHITECTURE.md)),
tool theory ([ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md)), retrieval mechanics
([CAMPAIGN_AND_RETRIEVAL.md](CAMPAIGN_AND_RETRIEVAL.md)).

The pipeline stays one forward DAG:

`pre_combined → preprocess → setup → hpc → analysis → reporter → post_combined`

The supervisor ticks completed stages forward and never re-enters them.
Planner / HITL are consulted on error only. Do not split the graph into two
isolated worker kinds.

---

## Status

| Priority | Theme | Intent |
|----------|--------|--------|
| **P0** | Wire what already exists | Axis-angle, versioned state, per-rep inventory, engine vs LLM log tags |
| **P1** | RAG, config, autonomous files | Knowledge retrieval, `campaign.yaml`, file tools restricted to the study directory |
| **P2** | Memory, review pauses, general MD tools | Run memory, protocol/mapping/comparison review, hydration / clustering / H-bond kinetics |
| **P3** | Platform later | MSM, MM-PBSA, membranes, FEP, non-GROMACS engines |

P0, P1, and P2 are implemented. Combined Ward plots from whatever numeric
columns exist (family gold-9 is preferred, never required to draw the
dendrogram). P3 stays documented until scheduled.

**Post-5×2 hardening (2026-09-19).** `calculate_consensus_pocket_metrics` is
registered on the per-sim executor (it was incorrectly combined-only). Science
completeness now requires each required calculation’s artifacts, feature
selection records empty columns honestly, combined HTML shows a science banner,
reporter inventories resolve campaign-root `cross_sim/`, LLM usage is tagged by
workflow node, and `AVAILABLE_SOFTWARE.md` ships with the programmer agent.

---

## P0 — wire what you already wrote

These are missing connections, not new science. Do them before more RAG or
new calculators.

### P0.1 Put `calculate_consensus_pocket_metrics` in the shared analysis protocol

**Why.** Gold Ward columns include ATP–pocket COM mean/std **and** ligand
axis-angle mean/std. `calculate_consensus_pocket_metrics` already writes both.
The shared analysis protocol still called `calculate_ligand_pocket_distance` (COM only),
so `reference_pocket_ligand_axis_angle_*` stayed empty and Ward failed.

**Do.**

- Make the first required calculation `calculate_consensus_pocket_metrics`.
- Accept `hpc_dir` / `topology_file` / `trajectory_file` / `pocket_map_json`.
- Resolve `pocket_resids` from `pocket_mapped.json` when the caller omits them.
- Use `traj_resolve` (refuse first-hit when several `hpc/repXX` exist).
- Write collector aliases: `ligand_pocket_distance.csv`, `pocket_axis_angle.csv`.
- Keep `calculate_ligand_pocket_distance` as a general-purpose tool.

**Done when.** Family compile pins the consensus pocket tool; a per-rep run
with `pocket_mapped.json` + bound traj writes COM **and** axis-angle products.

**Status.** Landed. Also registered on the **per-sim** `AnalysisToolExecutor`
(was incorrectly combined-only; 5×2 retest showed `Unknown analysis tool`).
Science completeness now requires pocket-metric artifacts as well as the four
`consensus_*` dirs.
### P0.2 Versioned campaign + per-sim `state.json`

**Why.** Resume used to lean only on truncated supervisor checkpoints
(`base/supervisor/state.jsonl` and `{label}/supervisor/state.jsonl`).
`file_registry` and analysis pointers get dropped. There is no schema version
or spec hash.

**Do.**

- Write `{base}/campaign/state.json`: schema version, spec hash, labels,
  completed campaign stages, science contract, `campaign_spec`.
- Write `{label}/state.json`: completed per-sim stages, resolved `hpc/repXX`
  topo/traj, science-complete record.
- Keep `{base}/supervisor/state.jsonl` as the overwriteable resume checkpoint
  (pretty-printed single JSON object; historical `.jsonl` name) plus
  `pool_status.json` for the live agent ladder.
- Resume overlays the JSON snapshot first when the supervisor checkpoint is
  missing spec or stages.

**Done when.** Every workflow persist writes both JSON snapshots; a unit test
round-trips schema version + spec hash.

### P0.3 Per-rep `analysis/repXX/inventory.json`

**Why.** Overlay “mean-only” plots and replica collapse were caused by
unbound traj (both reps analyzed `rep01`). Collapse checks and collectors
should read an explicit inventory, not guess paths.

**Do.** After PBC wrap and after replicate fan-out, write
`analysis/repXX/inventory.json` with absolute topology, trajectory,
`pocket_mapped`, `global_mapped`, and tool outputs.

**Done when.** Wrap / fan-out writes the file; a test asserts the bound
`hpc/rep02` trajectory is recorded.

### P0.4 Log tags: `source=engine|llm`

**Why.** The o15197 log looked like analysis never planned because protocol
steps were not tagged. Wrap, validation, and the shared analysis protocol are engine;
planner / analysis shopping lists are LLM.

**Do.** Every `AGENT ACTION` prints `[source=engine|llm|hitl]`. Shared
analysis protocol execution, wrap, and stage ticks are `engine`. LLM plan
generation is `llm`. Include resolved traj/topo when present.

**Done when.** A shared analysis protocol step log includes `source=engine` and the
bound trajectory path.

---

## P1 — RAG, config, autonomous file tools

### P1.1 Knowledge RAG with the existing embedder

**Why.** Tool search already uses hashed 3-gram + Jaccard + optional Ollama
(`AGENTIC_EMBED_MODEL`). Knowledge still dumps the first 6–10k characters of
every file. Only three knowledge files exist (prep, pdb2gmx, AMBER99SB).
README lists empty categories (protocols, CHARMM, equilibration).

**Do.**

- Chunk the knowledge tree; retrieve top-k with the same hybrid retriever.
- Persist `planner/knowledge_index.json` so planner calls do not re-embed.
- Stop dumping whole files into the prompt; cite chunk ids.
- Fill the empty knowledge categories the README already names.

**Done when.** Planner context is retrieved chunks + citations, not a dump.

**Status.** Landed. `agentic/retrieval/knowledge.py` chunks the tree, scores
with the same hybrid embedder, and persists `planner/knowledge_index.json`.
Empty README categories now have short manuals (equilibration, CHARMM36,
MD theory, ligand-binding).

### P1.2 One `campaign.yaml`

**Why.** Conservation metric, pocket cutoff, lobe hinge, retrieval k, gold
columns, and HITL mode are split across agent YAMLs and env vars.

**Do.** Single file the CLI and `CampaignSpec` both read. Agent YAMLs stay
for prompts. Env vars become overrides, not the source of truth.

Suggested keys: `conservation_metric`, `min_coverage`, `pocket_cutoff_A`,
`lobe_method`, `retrieval_k`, `embed_model`, `gold_columns`, `hitl`.

**Done when.** A family run can change group-similarity ≥ 0.5 / occupancy
0.25 / 15 Å pocket without editing Python.

**Status.** Landed. Shipped file: `agentic/campaign/campaign.yaml`. CLI
`--campaign-yaml`, `{working-dir}/campaign.yaml`, and `AGENTIC_*` env vars
override. `CampaignSpec.settings` carries the resolved knobs into compile.

### P1.3 Autonomous file tools + per-stage inventory

**Why.** HITL already has `pwd`, `read_file`, `list_dir`, `write_file`,
`grep_file`. Field agents cannot list or grep the sim tree. `file_registry`
is truncated.

**Do.**

- Expose `list_dir` / `grep` / inventory read to field agents, restricted
  to the study directory (campaign or sim root only).
- After each stage write `{stage}/inventory.json`.
- HITL `pwd` prints that inventory, not a guess.

**Done when.** Analysis can discover `pocket_mapped.json` via inventory
without a hard-coded parent walk (hard-coded walk remains as fallback).

**Status.** Landed. Field agents can call `list_dir` / `grep_file` /
`read_inventory` restricted to the study directory (campaign or sim root
only). Each stage writes
`{stage}/inventory.json`. HITL `pwd` prints that inventory. Pocket lookup
reads inventory first.

---

## P2 — memory, HITL, general-purpose tools

### P2.1 Run memory (past errors and fixes)

**Why.** Each LLM call is stateless. A family campaign rediscovers that
axis-angle was missing or that `noop` is not a tool.

**Do.** `campaign/memory.jsonl`: `{stage, error, fix, embedding}`. Retrieve
on planner / supervisor error. Do not replace `CampaignSpec`.

**Status.** Landed, then extended. Episodes live at the campaign root and
include tool and system label. A repeated error updates the same episode;
a later success writes the fix. Analysis, reporter, and the supervisor
retrieve the same episodes as the planner (`agent_retrieval` in
`campaign.yaml`). `campaign/study_notes.jsonl` holds short numeric notes
from this run. After an analysis error, the supervisor model may choose
`analysis`, `reporter`, or `final_report` once per system. Dense embeddings
use `embed_model` (default `nomic-embed-text`) with a process cache and a
hashed-n-gram fallback.

### P2.2 HITL on compile and combined stages

**Why.** `--HITL all|error` exists for per-sim stages. There is no
first-class pause after campaign compile, pre_combined, or post_combined.
Parallel pool + HITL still fights. Humans should approve `CampaignSpec` and
pocket/MSA before a 37×2 map.

**Do.** Checkpoints: after campaign compile, after pre_combined, after
post_combined. Keep the forward DAG.

**Status.** Landed. `--HITL all` pauses after the compiled protocol, after
shared MSA/pocket mapping, and after cross-system comparison. Continue
returns to the supervisor and ticks forward.

### P2.3 Orientation stays first-class

P0 pins the calculator. P2 can add a thin `calculate_ligand_axis_angle`
wrapper if generic (non-family) goals want orientation without the full
pocket-metrics bundle.

**Status.** Landed. `calculate_ligand_axis_angle` writes `pocket_axis_angle.csv`.

### P2.4 General MD tools the paper does not need

Keep paper tools modular. Add calculators users ask for constantly:

| Tool | Role |
|------|------|
| `calculate_water_occupancy` | Hydration-site density around `pocket_mapped` |
| Frame clustering | GROMOS / k-means / RMSD clusters → representative PDBs |
| `calculate_hbond_lifetimes` | Kinetics from the existing occupancy series |
| QC pack | Ramachandran, ligand-to-crystal RMSD, box/PBC check (replica-collapse gate is already in) |

Do **not** hard-code paper-only filenames into new tools. Accept
`pocket_mapped` / `global_mapped` paths as optional selections.

**Status.** Landed. `calculate_water_occupancy`, `cluster_trajectory_frames`
(GROMOS / k-means), `calculate_hbond_lifetimes`, and an extended
`run_trajectory_qc` (box/PBC, Ramachandran outlier fraction, optional
ligand-to-reference RMSD).

---

## P3 — later, not now

Useful for a general MD platform. Do not block the next robustness campaign.

| Capability | Note |
|------------|------|
| MSM / tICA / deeptime | tICA helper exists in `family_dynamics`; no MSM tool |
| MM-PBSA / energy terms | `analyze_energy` is EDR only |
| Membrane toolkit | APL, thickness, lipid order — knowledge README mentions membranes, no tools |
| Enhanced sampling / FEP | No metadynamics, REST2, or alchemical tools |
| Non-GROMACS engines | `md_engine` field exists; only GROMACS is implemented |
| OpenFF / GAFF robustness | Ligand prep is still a common setup failure mode |
| Package split | `agentic/` vs `src/analysis/` vs `retrieval` already started; not a user-facing win yet |

---

## Already landed (do not redo)

Keep these; P0 builds on them.

- `CampaignSpec` compile-once / map-many; reject truncated `(same as above)` prompts
- Study profiles `single_md`, `multi_protocol`, `family_modular`, `apo_holo`; feature packs and `feature_plugins` in `campaign.yaml` (optimization_5)
- Per-sim family gates do not require `consensus_PCA/`; shared-ref dyn is a post pin with `reference_label`
- Family post-combined requires `classification_features.csv`; generic combined can finish on overlays; combined re-entry is capped
- Shared analysis protocol (LLM compiles pre-combined extras; engine executes)
- Science completeness gate + 85% dense Ward matrix; plots are non-critical
- Hybrid tool retrieval (hashed n-gram + Jaccard + optional Ollama)
- Forward-only stage ticks (`stage_tick.py`)
- `traj_resolve` refuses first-hit when several replica dirs exist
- Replica-collapse fail when science products are byte-identical on distinct xtcs
- `pocket_mapped` / `global_mapped` naming; group similarity ≥ 0.5, occupancy ≥ 0.25
- Pocket = (KAPCA ATP 15 Å) ∩ `global_mapped`, mapped to every system
- Overlay plots: residue x-axis where appropriate; dashed mean + std; DCCM mean+std panels
- `--reuse-hpc` skips LLM preprocess/setup; skip metadata `noop` tools
- HITL file tools exist; P1.3 also exposes list/grep/inventory restricted to the study directory
- Knowledge RAG + `campaign.yaml` + per-stage `inventory.json` (P1)
- `calculate_consensus_pocket_metrics` is a required per-protein calculation (COM + axis-angle)
- Versioned `campaign/state.json` and `{label}/state.json`
- Ward dendrogram+heatmap plots from any usable table (≥2 rows, ≥2 numeric columns)

---

## Implementation order

1. **P0.1–P0.4** — landed.
2. **P1** — landed (knowledge RAG, `campaign.yaml`, autonomous file inventory).
3. **P2** — landed (run memory, compile/combined review pauses, general MD tools).
4. **P3** — only when a user study needs them.

Do not start other robustness tiers from this work. Do not rewrite live
run_02 artifacts.
