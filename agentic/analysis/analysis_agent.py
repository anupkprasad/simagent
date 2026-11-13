"""
MD Workflow Analysis Agent

Performs analysis of simulation outputs based on supervisor prompts.
Uses LLM to plan analysis steps and selects appropriate tools; includes
fallbacks when data or dependencies are unavailable.

Refactored to follow SimulationSetupAgent workflow pattern.
"""
import os
import re
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, FrozenSet, Set
from pathlib import Path

from ..state import MDState
from ..hitl_config import hitl_should_interact
from ..llm import LLMClient
from ..utils import (
    log_supervisor_routing, log_agent_start, log_llm_interaction,
    log_agent_action, log_file_operation, log_agent_completion, log_error,
    SecureFileManager, sanitize_tool_output_params
)
from ..utils.tool_prompt_format import format_tools_for_llm_prompt
from .schemas import (
    AnalysisPlan, AnalysisStep,
    AnalysisResult as AnalysisExecutionResult,
    AnalysisAgentInput, AnalysisAgentOutput
)
from .tools import AnalysisToolExecutor, get_tool_metadata, is_combined_analysis_tool
from src.analysis.chain_residue_map import CHAIN_SELECTION_LLM_NOTE
from src.analysis.replicate_paths import effective_rep_num as _effective_rep_num_for_state
from ..planner.planning_guidelines import (
    detect_requested_metrics,
    detect_requested_metrics_union,
    detect_requested_metrics_for_sim,
    detect_classification_requested,
    detect_family_modular_dynamics_requested,
    classification_metric_groups_for_goal,
    get_classification_tool_guide,
    get_classification_per_sim_tool_guide,
    detect_com_distance_mode,
    collect_goal_texts_for_intent,
    resolve_sims_for_combined_metric,
    get_standard_output_filenames_block,
    get_com_distance_tool_guide,
    get_proximity_tool_guide,
    get_pca_fel_tool_guide,
    STANDARD_OUTPUT_FILES,
    allowed_output_files_for_metrics,
    is_metric_output_filename,
)

logger = logging.getLogger(__name__)

_CALC_TOOL_TO_METRIC: Dict[str, str] = {
    "calculate_rmsd": "rmsd",
    "calculate_rmsf": "rmsf",
    "calculate_radius_of_gyration": "rg",
    "analyze_energy": "energy",
    "calculate_sasa": "sasa",
    "calculate_dccm": "dccm",
    "analyze_secondary_structure": "dssp",
    "calculate_ligand_pocket_distance": "com",
    "calculate_consensus_pocket_metrics": "com",
    "calculate_com_distance": "com",
    "calculate_trajectory_pca": "pca",
    "calculate_free_energy_landscape": "fel",
    "analyze_fel_landscape_features": "fel",
    "export_fel_basin_structures": "fel",
    "calculate_protein_ligand_contacts": "contacts",
    "calculate_pocket_sasa": "pocket_sasa",
    "analyze_ligand_residence": "residence",
    "calculate_pocket_rmsf": "pocket_rmsf",
    "calculate_ligand_rmsf": "ligand_rmsf",
    "identify_nearby_residues": "nearby",
    "calculate_min_heavy_atom_distance": "min_distance",
    "calculate_hbond_occupancy": "hbond_occupancy",
    "calculate_salt_bridge_distances": "salt_bridge",
    "calculate_consensus_torsions": "consensus_torsions",
    "calculate_consensus_rmsf_features": "consensus_rmsf",
    "calculate_consensus_dccm_features": "consensus_dccm",
    "run_independent_dynamics_fel": "dihedral_pca",
}

_PREP_TOOLS = frozenset({"wrap_trajectory", "run_complete_analysis"})

# Optional heavy deps (graceful fallback)
try:
    import MDAnalysis as mda  # type: ignore
    HAS_MDA = True
except Exception:
    HAS_MDA = False


class MDAnalysisAgent:
    """LLM-driven analysis agent that plans and executes analysis tasks."""

    def __init__(self, llm_client: LLMClient, config_path: Optional[str] = None):
        if llm_client is None:
            raise ValueError("llm_client is required")
        self.llm = llm_client
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config = self._load_config()
        self.tool_executor = None  # Initialized per execution
        self.file_manager = None  # Initialized per execution for state-specific file registry
        self.tool_executor = None
        logger.info("MD Analysis Agent initialized")

    def _load_config(self) -> Dict[str, Any]:
        import yaml
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        logger.warning(f"Analysis config {self.config_path} not found; using defaults")
        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "metrics": ["rmsd", "rmsf", "energy"],
            "plots": True,
            "output_dir": "./working_dir/analysis",
            "use_mdanalysis": True,
        }

    def _is_combined_analysis_phase(self, state: Optional[MDState] = None) -> bool:
        """True for pre/post combined multi-sim analysis phases."""
        if not state:
            return False
        return state.get("multi_sim_phase") in (
            "combined_analysis",
            "post_combined",
            "pre_combined",
        )

    def _is_pre_combined_phase(self, state: Optional[MDState] = None) -> bool:
        return bool(state and state.get("multi_sim_phase") == "pre_combined")

    def _is_combined_hitl_context(self, state: Optional[MDState] = None) -> bool:
        """True when HITL or workflow is at project-base cross-simulation scope."""
        if not state:
            return False
        return bool(
            state.get("hitl_view_combined")
            or state.get("hitl_combined_execute")
            or self._is_combined_analysis_phase(state)
            or state.get("multi_sim_phase") == "combined_reporter"
        )

    def _is_pinned_per_sim_tool(self, tool_name: str, state: Optional[MDState] = None) -> bool:
        """Compiled family / campaign pins must run in per-sim analysis.

        ``calculate_consensus_pocket_metrics`` used to sit in the shared-tool
        bucket, so the combined-tool guard skipped COM + axis-angle.
        """
        name = (tool_name or "").strip()
        if not name:
            return False
        try:
            from agentic.campaign.family_recipe import REQUIRED_CALCULATIONS

            if name in REQUIRED_CALCULATIONS:
                return True
        except Exception:
            pass
        try:
            from agentic.campaign.spec import spec_from_state

            spec = spec_from_state(state)
            if spec and name in (spec.pin_tools or []):
                return True
        except Exception:
            pass
        return False

    def _get_analysis_tool_metadata(self, state: Optional[MDState] = None) -> Dict[str, Dict[str, Any]]:
        """Tool metadata for LLM planning (includes combined tools at project base)."""
        working_dir = state.get("working_directory") if state else None
        include_combined = self._is_combined_hitl_context(state)
        return get_tool_metadata(
            working_directory=working_dir,
            include_combined=include_combined,
        )

    def _retrieve_pre_combined_tool_metadata(
        self,
        tool_metadata: Dict[str, Dict[str, Any]],
        state: MDState,
        task: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Rank combined/shared tools for pre_combined planning; always keep required shared setup."""
        from agentic.campaign.pre_combined import (
            REQUIRED_SHARED_SETUP,
            pre_combined_tool_allowed,
        )
        from agentic.campaign.spec import spec_from_state
        from agentic.retrieval.embed import hashed_ngram_embed, hybrid_score

        spec = spec_from_state(state)
        pins = list((spec.pin_pre_tools if spec else None) or REQUIRED_SHARED_SETUP)
        query = " ".join(
            t
            for t in (
                task,
                state.get("pre_combined_plan"),
                state.get("user_goal_original"),
            )
            if t
        )
        allowed = {
            name: meta
            for name, meta in (tool_metadata or {}).items()
            if pre_combined_tool_allowed(name)
        }
        if not allowed:
            return tool_metadata
        q_vec = hashed_ngram_embed(query or "consensus pocket msa alignment")
        ranked = sorted(
            allowed.items(),
            key=lambda kv: hybrid_score(
                query,
                f"{kv[0]} {kv[1].get('description') or ''}",
                q_vec,
                hashed_ngram_embed(f"{kv[0]} {kv[1].get('description') or ''}"),
            ),
            reverse=True,
        )
        picked: Dict[str, Dict[str, Any]] = {}
        for name in pins:
            if name in tool_metadata:
                picked[name] = tool_metadata[name]
        for name, meta in ranked:
            if name not in picked:
                picked[name] = meta
            if len(picked) >= 16:
                break
        return picked

    def _format_tools_list_for_prompt(self, tool_metadata: Dict[str, Dict[str, Any]]) -> str:
        """Alias for detailed tool formatting (kept for backward compatibility)."""
        return self._format_tools_list_detailed(tool_metadata)

    def _format_tools_list_detailed(self, tool_metadata: Dict[str, Dict[str, Any]]) -> str:
        """Format tool metadata (docstring Args/Returns only — no duplicate Parameters block)."""
        return format_tools_for_llm_prompt(tool_metadata)

    def _resolve_enriched_goal_for_planning(
        self,
        state: Optional[MDState],
        agent_input: Optional[AnalysisAgentInput] = None,
    ) -> str:
        """Per-simulation enriched goal from input validation (clearest analysis scope)."""
        if state and state.get("is_multi_simulation"):
            idx = state.get("current_sim_index", 0)
            sim_prompts = state.get("sim_prompts") or []
            if 0 <= idx < len(sim_prompts):
                stored = (sim_prompts[idx].get("enriched_prompt") or "").strip()
                if stored:
                    return stored
        candidates = (
            getattr(agent_input, "enriched_goal", None) if agent_input else None,
            (state or {}).get("enriched_prompt"),
            (state or {}).get("rephrased_goal"),
        )
        for candidate in candidates:
            text = (candidate or "").strip()
            if text:
                return text
        return ""

    def _format_goal_context_for_planning(
        self,
        agent_input: AnalysisAgentInput,
        state: Optional[MDState] = None,
    ) -> str:
        """User + enriched goals for analysis planning prompts."""
        enriched = self._resolve_enriched_goal_for_planning(state, agent_input)
        user_goal = (agent_input.user_goal or "").strip()
        if enriched and enriched != user_goal:
            return (
                f"- User Goal: {user_goal or 'Not specified'}\n"
                f"- Enriched User Goal (PRIMARY scope — plan every analysis listed here): "
                f"{enriched}"
            )
        if enriched:
            return f"- Enriched User Goal (PRIMARY scope): {enriched}"
        return f"- User Goal: {user_goal or 'Not specified'}"

    def _format_trajectory_input_block(self, agent_input: AnalysisAgentInput) -> str:
        """Explicit topology/trajectory basenames the LLM must use in tool_params."""
        topo_name = (
            Path(agent_input.topology_file).name
            if agent_input.topology_file
            else "md.tpr"
        )
        traj_name = (
            Path(agent_input.trajectory_file).name
            if agent_input.trajectory_file
            else "md.xtc"
        )
        return (
            "**Required input filenames (copy exactly into every calculate_*/analyze_* step):**\n"
            f"- topology_file: \"{topo_name}\"\n"
            f"- trajectory_file: \"{traj_name}\"\n"
        )

    def _discover_cross_sim_prompt_block(self, state: Optional[MDState] = None) -> str:
        """Auto-discover ``base/cross_sim/`` (+ analysis MSA) for per-sim planning."""
        if not state or self._is_combined_hitl_context(state):
            return ""
        try:
            from agentic.multi_sim_paths import resolve_multi_sim_base_dir
            from src.analysis.cross_sim_artifacts import (
                discover_cross_sim_artifacts,
                format_cross_sim_context_for_prompt,
            )

            base = resolve_multi_sim_base_dir(state)
            artifacts = discover_cross_sim_artifacts(base)
            # Also surface MSA / residue maps written under base/analysis/
            # when harvest into cross_sim/ was incomplete.
            analysis_dir = Path(base) / "analysis"
            extra: list[str] = list(artifacts.get("artifacts") or [])
            if analysis_dir.is_dir():
                for name in (
                    "global_consensus_msa.json",
                    "global_mapped.json",
                    "global_msa.json",
                    "global_msa.fasta",
                    "pocket_mapped.json",
                    "pocket_mapped.csv",
                    "pocket_mapped_definition.json",
                    "reference_msa_alignment.json",
                    "reference_consensus.json",
                    "consensus_residues.json",
                    "global_consensus_msa.csv",
                    "reference_msa_residue_map.csv",
                    "reference_pocket_residue_map.csv",
                    "reference_pocket_resid_map.csv",
                    "reference_residue_map.csv",
                    "pocket_resid_map_filtered.csv",
                ):
                    p = analysis_dir / name
                    if p.is_file():
                        sp = str(p.resolve())
                        if sp not in extra:
                            extra.append(sp)
                        if name.endswith(".json") and not artifacts.get(
                            "consensus_residues_path"
                        ):
                            artifacts["consensus_residues_path"] = sp
            artifacts["artifacts"] = extra
            if artifacts.get("artifacts") or artifacts.get("pocket_map_path"):
                state["cross_sim_artifacts"] = {
                    k: v
                    for k, v in artifacts.items()
                    if k not in ("pocket_map", "consensus_residues")
                }
                logger.info(
                    "Analysis: auto-discovered cross_sim artifacts under %s (%s files)",
                    artifacts.get("cross_sim_dir") or base,
                    len(artifacts.get("artifacts") or []),
                )
            return format_cross_sim_context_for_prompt(artifacts)
        except Exception as exc:
            logger.warning("cross_sim discovery failed: %s", exc)
            return ""

    def _get_per_sim_tool_scope_note(self, state: Optional[MDState] = None) -> str:
        """Tell the analysis LLM to stay within this simulation's scope."""
        if state and self._is_combined_hitl_context(state):
            return self._get_combined_hitl_scope_note(state)
        note = (
            "**SCOPE — THIS SIMULATION ONLY:**\n"
            "- Use only per-trajectory analysis tools listed below.\n"
            "- Do NOT call run_combined_*, plot_combined_overlay, collect_metric_files, "
            "or other cross-simulation tools.\n"
            "- Cross-simulation comparison (post_combined) runs after all simulations finish.\n"
        )
        if state and state.get("is_multi_simulation"):
            idx = state.get("current_sim_index", 0)
            sim_prompts = state.get("sim_prompts") or []
            if 0 <= idx < len(sim_prompts):
                label = sim_prompts[idx].get("label")
                if label:
                    note += f"- Current simulation: {label}\n"
        note += self._discover_cross_sim_prompt_block(state)
        return note

    def _get_combined_hitl_scope_note(self, state: Optional[MDState] = None) -> str:
        if state and self._is_pre_combined_phase(state):
            return (
                "**SCOPE — PRE-COMBINED (before per-sim traj analysis):**\n"
                "- You are compiling the *shared setup* every simulation will consume.\n"
                "- Choose combined/shared tools: pocket definition, MSA, residue maps, "
                "phylo trees, or any other structure-only calculation the goal needs.\n"
                "- Programmer-generated combined tools are allowed when they produce "
                "artifacts under ``cross_sim/`` (or ``analysis/``; harvested to "
                "``{base}/cross_sim/``).\n"
                "- Family campaigns also pin MSA + reference pocket + residue map; "
                "you may ADD extra setup tools, not replace those pins.\n"
                "- Do NOT run per-sim calculate_rmsd/rmsf, traj batch metrics, overlays, "
                "or Ward clustering here — those run after trajectories exist.\n"
            )
        return (
            "**SCOPE — POST-COMBINED CROSS-SIMULATION (project base):**\n"
            "- Prefer ONE call to run_combined_analysis(sim_dirs, labels, metrics=[...]) — "
            "it writes *_overlay.png and *_stats.csv. Do not also call collect_metric_files "
            "or compute_comparison_table for the same metrics unless you need a custom table.\n"
            "- When the user names specific proteins or simulations (e.g. JAK1 and TYK2 only), "
            "pass ONLY those sim_dirs and labels — do not include other simulations.\n"
            "- Plan ONLY the metrics the user requested; do not add Rg, COM, DCCM, RMSD, "
            "energy, or other analyses unless explicitly asked.\n"
            "- For overlay plots use run_combined_analysis with a metrics list "
            "(e.g. [\"rmsf\"] only) or collect_metric_files + plot_combined_overlay.\n"
            "- If you call compute_comparison_table, pass data_files (per-sim .dat paths) "
            "aligned with labels — or labels+sim_dirs+metric so the runtime can synthesize them.\n"
            "- Per-trajectory calculate_* tools are for single-sim rerun; prefer combined tools here.\n"
            "- Do NOT feed *_stats.csv into plot_multipanel / plot_md_data — those are tables, "
            "not time series. Use per-sim .dat files or *_overlay.png from run_combined_analysis.\n"
        )

    def _format_combined_sim_context_for_hitl(self, state: MDState) -> str:
        """List simulation paths and display labels for combined HITL planning."""
        from agentic.reporter.reporter_agent import resolve_combined_sim_context
        from src.reporter.combined_reporter import _parse_label_name_map, apply_label_name_map
        from src.analysis.phylo_tree import resolve_structure_pdb

        sim_dirs, labels = resolve_combined_sim_context(state)
        if not sim_dirs:
            return ""
        text = " ".join(
            filter(
                None,
                [
                    state.get("user_goal_original"),
                    state.get("master_enriched_prompt"),
                    state.get("user_goal"),
                ],
            )
        )
        name_map = _parse_label_name_map(text)
        if name_map:
            labels = apply_label_name_map(labels, name_map)
        base = state.get("multi_sim_base_dir") or state.get("working_directory", ".")
        pdb_by_label = {
            str(sp.get("label")): sp.get("pdb")
            for sp in (state.get("sim_prompts") or [])
            if sp.get("label") and sp.get("pdb")
        }
        lines = [
            "**Available simulations (use these exact labels / paths — do not invent):**",
            f"**PDB files (copy into pdb_files exactly):**",
        ]
        pdb_list_for_prompt: List[str] = []
        for sim_dir, label in zip(sim_dirs, labels):
            pdb = pdb_by_label.get(str(label)) or resolve_structure_pdb(
                str(label), str(sim_dir), str(base)
            )
            if pdb:
                pdb_list_for_prompt.append(str(pdb))
            lines.append(
                f"  - label=`{label}` sim_dir=`{sim_dir}` pdb=`{pdb or 'MISSING'}`"
            )
        if pdb_list_for_prompt:
            lines.append(
                "Use labels="
                + str(list(labels))
                + " and the pdb paths above (base-level .pdb, not {sim}/{label}.pdb)."
            )
        lines.append(
            f"**Combined output directory:** {Path(base) / 'analysis'} "
            "(use working_dir='.' in combined tool_params)"
        )
        lines.append(
            "Use alignment_json/consensus_json='global_consensus_msa.json' "
            "(MAFFT columns with physicochemical-group similarity ≥ 0.5). "
            "pocket_mapped = (ligand 15 Å) ∩ global_consensus_msa. "
            "Do not invent consensus_alignment.json."
        )
        return "\n".join(lines)

    def _normalize_hitl_plan_dict(self, plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce LLM plan JSON to shapes expected by AnalysisPlan."""
        overview = plan_dict.get("overview", "")
        if isinstance(overview, list):
            plan_dict["overview"] = "; ".join(str(x) for x in overview)
        elif not isinstance(overview, str):
            plan_dict["overview"] = str(overview) if overview else "HITL analysis task"
        reasoning = plan_dict.get("reasoning", "")
        if not isinstance(reasoning, str):
            plan_dict["reasoning"] = str(reasoning) if reasoning else ""
        for key in ("potential_issues", "recommendations"):
            val = plan_dict.get(key)
            if val is None:
                plan_dict[key] = []
            elif not isinstance(val, list):
                plan_dict[key] = [str(val)]
        return plan_dict

    def _filter_sims_for_hitl_task(
        self,
        task: str,
        sim_dirs: List[str],
        labels: List[str],
        name_map: Optional[Dict[str, str]] = None,
        *,
        allow_subset: bool = True,
    ) -> tuple:
        """Return sim_dirs/labels subset when the HITL task names specific proteins/sims.

        For workflow combined analysis (``allow_subset=False``), always keep the
        full simulation list — master-plan text often repeats one PDB/label and
        would otherwise drop sibling cases (e.g. holo ``*_ATP_MG``).
        """
        if not allow_subset or not task:
            return sim_dirs, labels
        task_lower = task.lower()
        matched_dirs: List[str] = []
        matched_labels: List[str] = []
        for sim_dir, label in zip(sim_dirs, labels):
            tokens = {label.lower(), Path(sim_dir).name.lower()}
            if name_map:
                for key, display in name_map.items():
                    if key.lower() in tokens or display.lower() in tokens:
                        tokens.update({key.lower(), display.lower()})
            # Prefer exact label/folder token hits; require word-ish boundaries
            # so "jak2_atp_2mg" does not uniquely select while excluding
            # "jak2_atp_2mg_ATP_MG" when both are campaign members.
            if any(len(t) >= 3 and t in task_lower for t in tokens):
                matched_dirs.append(sim_dir)
                matched_labels.append(label)
        if matched_dirs:
            # If every campaign sim matched, or only a proper subset was named,
            # use the match. If the shorter apo label alone matches because it
            # is a prefix of the holo label and appears in paths, keep all sims
            # when the task did not explicitly say "only".
            if len(matched_dirs) == len(sim_dirs):
                return matched_dirs, matched_labels
            only_named = bool(
                re.search(r"\b(?:only|just|solely)\b", task_lower)
            )
            if only_named:
                return matched_dirs, matched_labels
            # Ambiguous campaign text mentioning one shared stem — keep all.
            if len(matched_dirs) < len(sim_dirs) and len(sim_dirs) <= 8:
                return sim_dirs, labels
            return matched_dirs, matched_labels
        return sim_dirs, labels

    def _resolve_combined_tool_sim_dirs(
        self,
        state: MDState,
        task: str,
        tool_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fix sim_dirs/labels/pdb_files/working_dir for combined tools (ignore hallucinated paths)."""
        from agentic.reporter.reporter_agent import resolve_combined_sim_context
        from src.reporter.combined_reporter import _parse_label_name_map
        from src.analysis.phylo_tree import resolve_structure_pdb

        sim_dirs, labels = resolve_combined_sim_context(state)
        text = " ".join(
            filter(
                None,
                [task, state.get("user_goal_original"), state.get("master_enriched_prompt")],
            )
        )
        name_map = _parse_label_name_map(text)
        # Do NOT rewrite tool labels to display names (KAPCA_ATP). Campaign
        # folder labels (p17612_ATP) must stay so sim_dirs / pocket_map keys match.
        # HITL chat may name a subset; workflow combined always uses full campaign.
        allow_subset = bool(state.get("hitl_chat_task"))
        sim_dirs, labels = self._filter_sims_for_hitl_task(
            task, sim_dirs, labels, name_map, allow_subset=allow_subset
        )
        params = dict(tool_params)
        params["sim_dirs"] = sim_dirs
        params["labels"] = labels
        params["working_dir"] = self.file_manager.agent_dir if self.file_manager else "."

        base = state.get("multi_sim_base_dir") or state.get("working_directory") or ""
        if base:
            params["base_dir"] = str(base)

        # Inject real PDB paths from sim_prompts / resolve_structure_pdb so the
        # LLM cannot point at {sim}/{label}.pdb when only {base}/{uniprot}.pdb exists.
        pdb_by_label = {
            str(sp.get("label")): sp.get("pdb")
            for sp in (state.get("sim_prompts") or [])
            if sp.get("label") and sp.get("pdb")
        }
        resolved_pdbs: List[str] = []
        for sim_dir, label in zip(sim_dirs, labels):
            pdb = pdb_by_label.get(str(label))
            if not pdb or not Path(pdb).is_file():
                pdb = resolve_structure_pdb(str(label), str(sim_dir), str(base) if base else None)
            if pdb and Path(pdb).is_file():
                resolved_pdbs.append(str(Path(pdb).resolve()))
        if resolved_pdbs and len(resolved_pdbs) == len(labels):
            params["pdb_files"] = resolved_pdbs
        else:
            # Prefer sim_dirs resolution over hallucinated pdb_files.
            params.pop("pdb_files", None)

        # Normalize reference_label to an actual campaign label.
        ref = params.get("reference_label")
        if ref and labels and str(ref) not in labels:
            ref_l = str(ref).lower()
            matches = [
                lab
                for lab in labels
                if str(lab).lower() == ref_l
                or str(lab).lower().startswith(ref_l + "_")
                or ref_l.startswith(str(lab).lower() + "_")
            ]
            # Also match protein display names from sim_prompts (KAPCA → p17612_ATP).
            if not matches:
                for sp in state.get("sim_prompts") or []:
                    lab = str(sp.get("label") or "")
                    pname = str(sp.get("protein_name") or "").lower()
                    if lab in labels and (
                        pname == ref_l
                        or pname.startswith(ref_l)
                        or ref_l.startswith(pname)
                        or ref_l in pname
                    ):
                        matches.append(lab)
            if matches:
                params["reference_label"] = matches[0]

        # Keep MSA / pocket tools on a real basename. Invented names such as
        # consensus_alignment.json are rewritten to global_consensus_msa.json.
        from src.analysis.consensus_alignment import DEFAULT_ALIGNMENT_JSON

        cjson = params.get("consensus_json") or params.get("alignment_json")
        if cjson and Path(str(cjson)).name in (
            "consensus_alignment.json",
            "consensus_json.json",
        ):
            params["consensus_json"] = DEFAULT_ALIGNMENT_JSON
            params["alignment_json"] = DEFAULT_ALIGNMENT_JSON
        elif params.get("consensus_json") and not params.get("alignment_json"):
            params["alignment_json"] = params["consensus_json"]
        elif params.get("alignment_json") and not params.get("consensus_json"):
            params["consensus_json"] = params["alignment_json"]

        # Flatten hallucinated nested cross_sim/ under analysis working_dir.
        # Also strip ``analysis/`` prefixes LLMs add when working_dir is already analysis/.
        for key in (
            "residue_map_csv",
            "output_json",
            "definition_json",
            "alignment_fasta",
            "alignment_json",
            "consensus_json",
            "pocket_definition_json",
            "pocket_map_csv",
            "full_plot_file",
            "focused_plot_file",
        ):
            val = params.get(key)
            if not val:
                continue
            p = Path(str(val))
            parts = list(p.parts)
            if "cross_sim" in parts and not p.is_absolute():
                # Write basename into analysis/; harvest copies into base/cross_sim/.
                params[key] = p.name
            elif not p.is_absolute() and (
                "analysis" in parts or str(val).startswith("./")
            ):
                params[key] = p.name

        return params

    def _task_prefers_existing_combined_data(self, task: str) -> bool:
        lower = task.lower()
        return any(
            kw in lower
            for kw in (
                "already",
                "existing",
                "respective",
                "raw data",
                "just use",
                "from their",
                "from each",
                "only create",
                "overlay only",
                "plot only",
            )
        )

    def _sanitize_combined_hitl_plan(
        self,
        plan_dict: Dict[str, Any],
        state: MDState,
        task: str,
    ) -> Dict[str, Any]:
        """Drop per-sim recalc steps when data exists; fix combined tool paths/metrics."""
        if not self._is_combined_hitl_context(state):
            return plan_dict

        from agentic.reporter.reporter_agent import resolve_combined_sim_context
        from src.reporter.combined_reporter import _parse_label_name_map, apply_label_name_map
        from agentic.planner.planning_guidelines import (
            detect_classification_requested,
            detect_phylo_tree_requested,
            detect_requested_metrics_union,
        )

        sim_dirs, labels = resolve_combined_sim_context(state)
        text = " ".join(
            filter(
                None,
                [
                    task,
                    state.get("user_goal_original"),
                    state.get("combined_analysis_plan"),
                    state.get("master_enriched_prompt"),
                ],
            )
        )
        name_map = _parse_label_name_map(text)
        # Keep campaign folder labels for tool params (display names break paths).
        allow_subset = bool(state.get("hitl_chat_task"))
        sim_dirs, labels = self._filter_sims_for_hitl_task(
            task, sim_dirs, labels, name_map, allow_subset=allow_subset
        )

        intent_texts = (
            task,
            state.get("user_goal_original") or "",
            state.get("combined_analysis_plan") or "",
            state.get("master_enriched_prompt") or "",
        )
        requested = detect_requested_metrics_union(*intent_texts) or frozenset()
        class_ok = detect_classification_requested(*intent_texts)
        phylo_req = detect_phylo_tree_requested(*intent_texts)
        phylo_ok = bool(phylo_req.get("sequence") or phylo_req.get("structure"))

        _CLASSIFICATION_TOOLS = {
            "collect_classification_features_table",
            "cluster_classification_features",
            "plot_cluster_feature_trajectories",
            "plot_cluster_rmsf_profiles",
            "collect_fel_features_table",
        }
        _PHYLO_TOOLS = {
            "build_sequence_phylo_tree",
            "build_structure_phylo_tree",
            "build_consensus_sequence_alignment",
            "build_global_mapped_alignment",
            "plot_reference_msa_alignment",
            "plot_global_mapped_alignment",
        }
        from agentic.campaign.pre_combined import (
            merge_pre_combined_steps,
            pre_combined_tool_allowed,
        )
        from agentic.campaign.spec import spec_from_state

        is_pre = self._is_pre_combined_phase(state)
        _fam_spec = spec_from_state(state)
        _pin_pre = bool(_fam_spec and (_fam_spec.family_modular or _fam_spec.pin_pre_tools))

        recalc = any(
            kw in task.lower()
            for kw in ("recalculate", "recompute", "re-run", "rerun", "from trajectory", "from scratch")
        )
        prefer_existing = not recalc and (
            self._task_prefers_existing_combined_data(task) or self._is_combined_hitl_context(state)
        )
        metrics = sorted(
            m for m in requested if m in {"rmsd", "rmsf", "rg", "energy", "sasa", "hbond"}
        )
        if not metrics and "rmsf" in task.lower():
            metrics = ["rmsf"]

        new_steps: List[Dict[str, Any]] = []
        for step in plan_dict.get("steps") or []:
            tool = step.get("tool_name") or ""
            if is_pre and tool and not pre_combined_tool_allowed(tool):
                logger.info(
                    "sanitize pre_combined: dropping traj/post tool %s", tool
                )
                continue
            if prefer_existing and tool.startswith("calculate_"):
                continue
            if tool in _CLASSIFICATION_TOOLS and not class_ok:
                continue
            if tool in _PHYLO_TOOLS and not phylo_ok and not is_pre:
                continue
            if is_combined_analysis_tool(tool):
                params = dict(step.get("tool_params") or {})
                params["sim_dirs"] = sim_dirs
                params["labels"] = labels
                params["working_dir"] = "."
                # Only run_combined_analysis takes a metrics list; do not stamp
                # metrics onto classification/clustering or overlay-specific tools.
                if tool == "run_combined_analysis" and metrics:
                    params["metrics"] = metrics
                step = {**step, "tool_params": params}
                new_steps.append(step)
            elif not tool.startswith("calculate_"):
                new_steps.append(step)

        # Required shared setup (MSA/pocket) is merged in; LLM extras stay.
        # Normalize pocket/MSA contract params from goal + campaign settings so
        # independent runs redefine the pocket the same way (not shell vs mapped).
        if is_pre:
            from agentic.campaign.config import load_campaign_settings
            from agentic.campaign.pre_combined import family_pre_combined_pin_steps
            from agentic.planner.planning_guidelines import detect_consensus_pocket_requested

            settings = load_campaign_settings()
            pocket_req = detect_consensus_pocket_requested(
                *collect_goal_texts_for_intent(state)
            )
            pin_settings = {
                "pocket_cutoff_A": float(
                    pocket_req.get("pocket_cutoff_A")
                    or getattr(settings, "pocket_cutoff_A", 15.0)
                ),
                "conservation_metric": getattr(settings, "conservation_metric", "similarity"),
                "min_conservation": getattr(settings, "min_conservation", 0.5),
                "min_coverage": getattr(settings, "min_coverage", 0.25),
            }
            ref = pocket_req.get("reference_label") or ""
            new_steps = merge_pre_combined_steps(
                new_steps,
                pin_steps=family_pre_combined_pin_steps(
                    labels=labels,
                    sim_dirs=sim_dirs,
                    reference_label=str(ref or ""),
                    settings=pin_settings,
                ),
                labels=labels,
                sim_dirs=sim_dirs,
                pin=_pin_pre,
            )

        # Force basenames so LLM ``./analysis/foo.json`` does not become
        # ``analysis/analysis/foo.json`` under the combined working_dir.
        for step in new_steps:
            params = dict(step.get("tool_params") or {})
            changed = False
            for key in (
                "definition_json",
                "consensus_json",
                "alignment_json",
                "alignment_fasta",
                "output_json",
                "residue_map_csv",
                "pocket_definition_json",
                "full_plot_file",
                "focused_plot_file",
            ):
                val = params.get(key)
                if not val:
                    continue
                p = Path(str(val))
                if not p.is_absolute() and p.name != str(val):
                    params[key] = p.name
                    changed = True
            if changed:
                step["tool_params"] = params

        has_combined = any(is_combined_analysis_tool(s.get("tool_name", "")) for s in new_steps)
        if not has_combined and metrics and not is_pre:
            new_steps.append({
                "name": f"Combined {'/'.join(metrics)} overlay",
                "description": (
                    "Collect existing per-simulation analysis files and build cross-sim overlay plots."
                ),
                "tool_name": "run_combined_analysis",
                "tool_params": {
                    "sim_dirs": sim_dirs,
                    "labels": labels,
                    "working_dir": ".",
                    "metrics": metrics,
                },
                "reason": "User requested combined plot from existing per-sim analysis data.",
            })

        plan_dict["steps"] = new_steps
        return plan_dict

    def _create_combined_hitl_fallback_plan(
        self,
        agent_input: AnalysisAgentInput,
        state: MDState,
    ) -> AnalysisPlan:
        """Template plan for combined HITL — overlay existing per-sim analysis files."""
        if self._is_pre_combined_phase(state):
            return self._create_pre_combined_recipe_plan(agent_input, state)
        task = agent_input.user_goal or state.get("hitl_chat_task") or ""
        requested = detect_requested_metrics(task) or frozenset()
        metrics = sorted(m for m in requested if m in {"rmsd", "rmsf", "rg", "energy", "sasa", "hbond"})
        if not metrics:
            metrics = ["rmsf"] if "rmsf" in task.lower() else ["rmsf"]

        from agentic.reporter.reporter_agent import resolve_combined_sim_context
        from src.reporter.combined_reporter import _parse_label_name_map, apply_label_name_map

        sim_dirs, labels = resolve_combined_sim_context(state)
        text = " ".join(
            filter(
                None,
                [task, state.get("user_goal_original"), state.get("master_enriched_prompt")],
            )
        )
        name_map = _parse_label_name_map(text)
        if name_map:
            labels = apply_label_name_map(labels, name_map)
        sim_dirs, labels = self._filter_sims_for_hitl_task(task, sim_dirs, labels, name_map)

        step = AnalysisStep(
            name=f"Combined {'/'.join(metrics)} overlay",
            description="Overlay existing per-simulation metric files at project base.",
            tool_name="run_combined_analysis",
            tool_params={
                "sim_dirs": sim_dirs,
                "labels": labels,
                "working_dir": ".",
                "metrics": metrics,
            },
            reason="Combined HITL fallback using on-disk per-sim analysis outputs.",
        )
        return AnalysisPlan(
            reasoning=f"Combined overlay from existing data for: {', '.join(labels)}",
            overview=f"Combined {'/'.join(metrics)} comparison for {', '.join(labels)}",
            steps=[step],
            potential_issues=["Missing metric files under sim/analysis/ will skip that simulation"],
            recommendations=["Verify rmsf.dat (or rmsf.png) exists in each sim analysis folder"],
        )

    def analysis_node(self, state: MDState) -> MDState:
        """
        Main analysis node - entry point from workflow.

        In multi-sim combined_analysis phase: LLM plans with combined tools
        exposed, then executes (deterministic pipeline is the fallback).
        Otherwise: runs the regular per-simulation LLM-guided analysis.
        """
        # ── Combined multi-sim analysis (pre or post) ─────────────────────
        if self._is_combined_analysis_phase(state):
            # Post-combined with Ward/feature-table intent: prefer deterministic
            # collect→cluster path (LLM often invents wrong tools/paths).
            if not self._is_pre_combined_phase(state):
                from agentic.planner.planning_guidelines import (
                    detect_classification_requested,
                    classification_metric_groups_for_goal,
                )

                texts = (
                    state.get("user_goal_original") or "",
                    state.get("post_combined_plan") or "",
                    state.get("combined_analysis_plan") or "",
                    state.get("master_enriched_prompt") or "",
                )
                if detect_classification_requested(*texts) or (
                    classification_metric_groups_for_goal(*texts) is not None
                ):
                    logger.info(
                        "Post-combined: classification/Ward requested — "
                        "using deterministic feature table + clustering"
                    )
                    return self._run_combined_analysis(state)
            return self._run_combined_analysis_via_llm(state)

        # ── Regular per-sim analysis ──────────────────────────────────────
        execution_plan = state.get("execution_plan", {})
        has_planner_instructions = execution_plan.get("format") == "natural_language"
        
        input_summary = {
            "hpc_output_dir": state.get("hpc_output_directory"),
            "trajectory_file": state.get("trajectory_path"),
            "topology_file": state.get("topology")
        }
        
        if has_planner_instructions:
            # Prefer pre-extracted instructions from supervisor (avoids duplication)
            analysis_section = state.get("analysis_instructions")
            
            if not analysis_section:
                # Fallback: Extract from full plan if supervisor didn't provide it
                full_plan = execution_plan.get("full_plan", "")
                analysis_section = self._extract_agent_instructions(full_plan, "Analysis Agent")
            
            if analysis_section:
                input_summary["planner_instructions"] = analysis_section
            else:
                input_summary["planner_instructions"] = "[Natural language plan from planner]"
        else:
            # Fallback to user goal if no planner instructions
            input_summary["user_goal"] = state.get("user_goal")
        
        log_agent_start("analysis", "MD Trajectory Analysis with LLM Planning", input_summary)
        
        try:
            # Initialize secure file manager
            working_dir = state.get("working_directory", "working_dir")
            file_registry = state.get("file_registry", {})
            
            self.file_manager = SecureFileManager(
                working_dir=working_dir,
                agent_name="analysis",
                file_registry=file_registry
            )
            
            logger.info(f"Analysis agent directory: {self.file_manager.agent_dir}")
            
            # Get analysis directory from file manager (ensures consistency)
            analysis_dir = self.file_manager.agent_dir
            state["analysis_dir"] = analysis_dir
            state["analysis_directory"] = analysis_dir  # Backward compatibility
            
            self.tool_executor = AnalysisToolExecutor(config={
                "working_directory": analysis_dir,
                "include_combined_tools": False,
                "user_goal": state.get("user_goal_original") or state.get("user_goal", ""),
                "rep_num": _effective_rep_num_for_state(state, working_dir),
                "sim_root": working_dir,
                "replicate_base_seed": int(state.get("replicate_base_seed") or 12345),
            })
            
            # Copy files from HPC output directory if needed (using secure file manager)
            self._copy_files_from_hpc_secure(state)

            # Wrap trajectory to fix PBC artefacts (runs by default; set
            # skip_pbc_wrap=True in state to disable for pre-wrapped trajectories)
            self._wrap_trajectory_pbc(state, analysis_dir)

            # Write PDB validation info to summary file if available from supervisor
            self._write_pdb_info_to_summary(state, analysis_dir)
            
            # Prepare agent input from state
            agent_input = self._prepare_agent_input(state)
            
            # Run LLM-guided analysis workflow
            agent_output = self._run_analysis_workflow(agent_input, state)
            
            # Update state with results
            self._update_state(state, agent_output)
            
            # Determine next workflow node
            if agent_output.success:
                # Clear any previous analysis-related errors from earlier retry attempts
                state["errors"] = [
                    e for e in state.get("errors", [])
                    if not (e.startswith("Analysis failed:") or e.startswith("Analysis error:"))
                ]
                self._record_analysis_outcome(state, success=True, detail="")
                if hitl_should_interact(state):
                    state["next_node"] = "human_analysis_check"
                else:
                    state["next_node"] = "supervisor"
            else:
                report = str(agent_output.result.report)
                state["errors"].append(f"Analysis failed: {report}")
                self._record_analysis_outcome(state, success=False, detail=report)
                # Error-triggered HITL
                state["next_node"] = "human_analysis_check"
                state["error_triggered_hitl"] = True
            
            success = bool(agent_output.success)
            log_agent_completion("analysis", "MD Trajectory Analysis", state, success)
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Analysis agent failed: {e}")
            logger.error(f"Traceback: {tb}")
            log_error("analysis_agent.analysis_node", e, {"state": str(state), "traceback": tb})
            state["errors"].append(f"Analysis error: {str(e)}")
            self._record_analysis_outcome(state, success=False, detail=str(e))
            # Error-triggered HITL
            state["next_node"] = "human_analysis_check"
            state["error_triggered_hitl"] = True
        
        return state

    # ── HITL in-chat task execution ─────────────────────────────────────

    def init_for_hitl_execution(self, state: MDState) -> str:
        """Initialize tool executor and paths for a HITL chat task (no graph routing)."""
        working_dir = state.get("working_directory", "working_dir")
        analysis_dir = (
            state.get("analysis_dir")
            or state.get("hitl_agent_output_directory")
            or str(Path(working_dir) / "analysis")
        )
        Path(analysis_dir).mkdir(parents=True, exist_ok=True)
        state["analysis_dir"] = analysis_dir
        state["analysis_directory"] = analysis_dir

        file_registry = state.get("file_registry") or {}
        self.file_manager = SecureFileManager(
            working_dir=working_dir,
            agent_name="analysis",
            file_registry=file_registry,
        )
        self.tool_executor = AnalysisToolExecutor(
            config={
                "working_directory": analysis_dir,
                "include_combined_tools": bool(
                    state.get("hitl_view_combined")
                    or self._is_combined_analysis_phase(state)
                    or state.get("hitl_combined_execute")
                ),
                "user_goal": state.get("user_goal_original") or state.get("user_goal", ""),
                "rep_num": _effective_rep_num_for_state(state, working_dir),
                "sim_root": working_dir,
                "replicate_base_seed": int(state.get("replicate_base_seed") or 12345),
            }
        )
        self._copy_files_from_hpc_secure(state)
        if not state.get("skip_pbc_wrap"):
            self._wrap_trajectory_pbc(state, analysis_dir)

        resolved = self._resolve_input_files(state)
        if resolved.get("topology"):
            state["topology"] = resolved["topology"]
        if resolved.get("trajectory"):
            state["trajectory_path"] = resolved["trajectory"]
        if resolved.get("energy"):
            state["energy_file"] = resolved["energy"]
        if state.get("hpc_output_directory") is None:
            if self._is_combined_hitl_context(state):
                state["hpc_output_directory"] = ""
            else:
                state["hpc_output_directory"] = str(Path(working_dir) / "hpc")
        return analysis_dir

    def prepare_agent_input_for_hitl(self, state: MDState, task: str) -> AnalysisAgentInput:
        """Build agent input with the HITL chat task as the sole user goal (no planner merge)."""
        topology_file = state.get("topology")
        hpc_dir = (
            state.get("hpc_dir")
            or state.get("hpc_output_directory")
            or state.get("hpc_directory")
        )
        if not hpc_dir and not self._is_combined_hitl_context(state):
            hpc_dir = str(Path(state.get("working_directory", "working_dir")) / "hpc")
        hpc_dir = hpc_dir or ""
        if hpc_dir and topology_file:
            hpc_dir_path = Path(hpc_dir)
            candidate = hpc_dir_path / Path(topology_file).name
            if candidate.exists():
                topology_file = str(candidate)

        return AnalysisAgentInput(
            working_directory=state.get("working_directory", "working_dir"),
            hpc_output_dir=hpc_dir,
            topology_file=topology_file,
            trajectory_file=state.get("trajectory_path"),
            energy_file=state.get("energy_file"),
            analyses=[],
            user_goal=task,
            additional_instructions=None,
        )

    def run_hitl_analysis_workflow(self, task: str, state: MDState) -> AnalysisAgentOutput:
        """Plan + execute a HITL chat task using the same logging path as normal analysis."""
        state["hitl_chat_task"] = task
        input_summary = {
            "user_goal": task,
            "topology_file": state.get("topology"),
            "trajectory_file": state.get("trajectory_path"),
        }
        log_agent_start("analysis", "HITL Analysis Task", input_summary)

        try:
            agent_input = self.prepare_agent_input_for_hitl(state, task)
            plan = self._create_hitl_analysis_plan_llm(agent_input, state, task)

            log_agent_action("analysis", "Generated analysis plan", {
                "steps": len(plan.steps),
                "tools": [s.tool_name for s in plan.steps],
                "reasoning": plan.reasoning,
                "source": "hitl",
            })

            exec_plan = state.get("execution_plan") or {}
            exec_plan.setdefault("structured_plans", {})["analysis"] = plan.model_dump()
            state["execution_plan"] = exec_plan

            output = self.execute_plan_for_hitl(agent_input, plan, state)
            self._save_hitl_execution_artifacts(plan, output.result, task)

            success = bool(output.success)
            log_agent_completion("analysis", "HITL Analysis Task", state, success)
            return output
        finally:
            state.pop("hitl_chat_task", None)

    def create_hitl_analysis_plan(
        self,
        agent_input: AnalysisAgentInput,
        state: MDState,
        task: str,
    ) -> AnalysisPlan:
        """LLM JSON plan for a HITL task (always fresh, not planner-derived)."""
        return self._create_hitl_analysis_plan_llm(agent_input, state, task)

    def execute_plan_for_hitl(
        self,
        agent_input: AnalysisAgentInput,
        plan: AnalysisPlan,
        state: MDState,
    ) -> AnalysisAgentOutput:
        """Execute plan and merge results into state (HITL chat context)."""
        result = self._execute_analysis_plan(agent_input, plan, state)
        output = AnalysisAgentOutput(
            success=result.success,
            plan=plan,
            result=result,
            supervisor_update={
                "analysis_results": result.results,
                "analysis_directory": state.get("analysis_directory"),
            },
        )
        self._update_state(state, output)
        return output

    def _create_hitl_analysis_plan_llm(
        self,
        agent_input: AnalysisAgentInput,
        state: MDState,
        task: str,
    ) -> AnalysisPlan:
        """LLM plan for HITL chat — task text only; no multisim intent filtering."""
        prompt = self._build_hitl_planning_prompt(task, agent_input, state)
        try:
            content = self.llm.prompt_raw(prompt, temperature=0.2, max_tokens=16384, format="json")

            log_llm_interaction(
                "analysis.hitl_planning",
                prompt,
                content,
                is_mock=hasattr(self.llm, "_is_mock_mode") and self.llm._is_mock_mode,
            )

            plan_dict = self._extract_plan_json(content)
            plan_dict = self._normalize_plan_steps(plan_dict, agent_input)
            plan_dict = self._normalize_hitl_plan_dict(plan_dict)
            plan_dict = self._sanitize_combined_hitl_plan(plan_dict, state, task)
            return AnalysisPlan(
                reasoning=plan_dict.get("reasoning", content[:500]),
                overview=plan_dict.get("overview", "HITL analysis task"),
                steps=[
                    AnalysisStep(
                        name=step.get("name", "unknown"),
                        description=step.get("description", ""),
                        tool_name=step.get("tool_name", ""),
                        tool_params=step.get("tool_params", {}),
                        reason=step.get("reason", ""),
                    )
                    for step in plan_dict.get("steps", [])
                ],
                potential_issues=plan_dict.get("potential_issues", []),
                recommendations=plan_dict.get("recommendations", []),
            )
        except Exception as exc:
            logger.warning("HITL LLM planning failed, using fallback: %s", exc)
            if self._is_combined_hitl_context(state):
                return self._create_combined_hitl_fallback_plan(agent_input, state)
            return self._create_fallback_analysis_plan(agent_input, state)

    def _build_hitl_planning_prompt(
        self,
        task: str,
        agent_input: AnalysisAgentInput,
        state: MDState,
    ) -> str:
        """Planning prompt where the HITL chat task is the only user intent."""
        tool_metadata = self._get_analysis_tool_metadata(state)
        tools_list_str = self._format_tools_list_detailed(tool_metadata)
        pdb_info_str = self._format_pdb_info_for_llm(state)
        input_files_block = self._format_trajectory_input_block(agent_input)

        topo_name = Path(agent_input.topology_file).name if agent_input.topology_file else "md.tpr"
        traj_name = Path(agent_input.trajectory_file).name if agent_input.trajectory_file else "mdWrap.xtc"

        scope_note = self._get_per_sim_tool_scope_note(state)
        combined_context = ""
        if self._is_combined_hitl_context(state):
            combined_context = self._format_combined_sim_context_for_hitl(state)

        return f"""You are the Analysis Agent in Human-in-the-Loop (HITL) mode.

**USER REQUEST (sole intent — ignore any prior workflow or multisim goals):**
{task}
{scope_note}
{combined_context}

**Available Data:**
- Topology File: {agent_input.topology_file or "Not available"} (use filename "{topo_name}" in tool_params)
- Trajectory File: {agent_input.trajectory_file or "Not available"} (use filename "{traj_name}" in tool_params)
- Energy File: {agent_input.energy_file or "Not available"}
{input_files_block}
{pdb_info_str}

**Available Tools:**
{tools_list_str}

**HITL PLANNING RULES:**
- Plan ONLY what the user requested above — do not add RMSF, Rg, RMSD, COM distance, or other metrics unless explicitly asked.
- Compound requests need multiple steps (one tool call per distinct deliverable).
- For DSSP / secondary structure:
  • Full protein: analyze_secondary_structure with selection="protein", output_prefix="dssp", create_heatmap=True
  • Residue segment (e.g. 100–120): a second analyze_secondary_structure with selection="protein and resid 100:120",
    output_prefix="dssp_res100_120", create_heatmap=True
  • analyze_secondary_structure generates heatmaps internally — do NOT add a separate plot step for DSSP heatmaps.
{CHAIN_SELECTION_LLM_NOTE}
- For calculate_* metrics that produce .dat/.csv files, follow each with plot_md_data using the data filename only.
- For combined cross-simulation overlays: use run_combined_analysis with metrics limited to what the user asked
  (e.g. metrics=["rmsf"] only) and sim_dirs/labels restricted to the simulations the user named.
- Per-simulation metric files already live under each sim's analysis/ folder (e.g. rmsf.dat). For combined
  overlay requests, do NOT call calculate_rmsf/calculate_* — only run_combined_analysis (or collect_metric_files
  + plot_combined_overlay) using those existing files.
- Use the exact sim_dirs paths listed above — never invent paths from other projects.
- overview MUST be a single string (not a JSON array).

Output as JSON:
{{
  "reasoning": "How you will fulfill the user request",
  "overview": "High-level summary",
  "steps": [
    {{
      "name": "step name",
      "description": "what it does",
      "tool_name": "tool to call",
      "tool_params": {{"param": "value"}},
      "reason": "why this step is needed"
    }}
  ],
  "potential_issues": [],
  "recommendations": []
}}
"""

    def _save_hitl_execution_artifacts(
        self,
        plan: AnalysisPlan,
        result: AnalysisExecutionResult,
        task: str,
    ) -> None:
        """Persist execution_plan.json, execution_report.md, and execution_log.txt."""
        if not self.file_manager:
            return

        agent_dir = self.file_manager.agent_dir
        stamp = datetime.now().isoformat(timespec="seconds")

        plan_data = {
            "timestamp": stamp,
            "source": "hitl",
            "task": task,
            "reasoning": plan.reasoning,
            "overview": plan.overview,
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "tool_name": step.tool_name,
                    "tool_params": step.tool_params,
                    "reason": step.reason,
                }
                for step in plan.steps
            ],
            "potential_issues": plan.potential_issues,
            "recommendations": plan.recommendations,
        }
        plan_file = os.path.join(agent_dir, "execution_plan.json")
        with open(plan_file, "w", encoding="utf-8") as fh:
            json.dump(plan_data, fh, indent=2)

        report = self._generate_execution_report(
            plan, result.results, result.issues, result.warnings
        )
        report_file = os.path.join(agent_dir, "execution_report.md")
        with open(report_file, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")

        log_file = os.path.join(agent_dir, "execution_log.txt")
        step_lines = [
            f"  {i + 1}. {s.tool_name}: {s.name}" for i, s in enumerate(plan.steps)
        ]
        header = f"ANALYSIS EXECUTION — {stamp}\nTask: {task}\nSuccess: {result.success}\n"
        body = header + "Plan steps:\n" + "\n".join(step_lines) + "\n\n" + result.execution_log
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write("\n" + "=" * 72 + "\n" + body + "\n")

    # ── Combined multi-sim analysis ───────────────────────────────────────

    def _build_combined_analysis_task(self, state: MDState) -> str:
        """Compose the sole planning intent for combined LLM analysis (pre or post)."""
        parts = []
        original = (state.get("user_goal_original") or "").strip()
        is_pre = self._is_pre_combined_phase(state)
        if is_pre:
            stage_plan = (
                state.get("pre_combined_plan")
                or state.get("combined_analysis_plan")
                or ""
            ).strip()
            stage_label = "Pre-combined (pocket / MSA / consensus before per-sim traj)"
        else:
            stage_plan = (
                state.get("post_combined_plan")
                or state.get("combined_analysis_plan")
                or ""
            ).strip()
            stage_label = "Post-combined (compare / overlay / family report after all sims)"
        if original:
            parts.append(f"Original study goal:\n{original}")
        if stage_plan:
            parts.append(f"{stage_label} plan from master planner:\n{stage_plan}")
        if not parts:
            parts.append(
                (state.get("user_goal") or state.get("analysis_instructions") or "").strip()
                or (
                    "Build shared pocket/MSA/consensus artifacts under cross_sim/."
                    if is_pre
                    else "Compare simulations with combined overlay plots of requested metrics."
                )
            )
        if is_pre:
            parts.append(
                "Write shared artifacts under cross_sim/: global_consensus_msa.json "
                "(MAFFT columns with group similarity ≥ 0.5) and pocket_mapped.json "
                "(15 Å ATP ∩ global_consensus_msa). Prefer "
                "build_global_mapped_alignment, define_pocket_mapped_residues, "
                "map_pocket_mapped_residues. "
                "Do not run Ward clustering, feature tables, or per-sim trajectory metrics here."
            )
        else:
            parts.append(
                "Use existing per-simulation analysis outputs under each sim's analysis/ "
                "directory (prefer analysis/avg/ when multi-rep). Prefer run_combined_analysis "
                "/ plot_combined_overlay over recalculating metrics from trajectories. Plan ONLY "
                "metrics and deliverables explicitly requested — do not add classification, "
                "clustering, ligand RMSF, DCCM, or other extras unless the goal asks for them."
            )
        return "\n\n".join(parts)

    def _build_combined_analysis_planning_prompt(
        self,
        task: str,
        agent_input: AnalysisAgentInput,
        state: MDState,
    ) -> str:
        """LLM planning prompt for workflow combined analysis (tools + sim context)."""
        tool_metadata = self._get_analysis_tool_metadata(state)
        if self._is_pre_combined_phase(state):
            tool_metadata = self._retrieve_pre_combined_tool_metadata(
                tool_metadata, state, task
            )
        tools_list_str = self._format_tools_list_detailed(tool_metadata)
        scope_note = self._get_combined_hitl_scope_note(state)
        combined_context = self._format_combined_sim_context_for_hitl(state)
        class_guide = ""
        if detect_classification_requested(
            task,
            state.get("user_goal_original") or "",
            state.get("post_combined_plan")
            or state.get("combined_analysis_plan")
            or "",
        ):
            class_guide = "\n" + get_classification_tool_guide()

        mode_title = (
            "PRE-COMBINED (pocket/MSA/consensus)"
            if self._is_pre_combined_phase(state)
            else "POST-COMBINED multi-simulation"
        )
        return f"""You are the Analysis Agent in {mode_title} mode.

**OBJECTIVE (plan and execute only this):**
{task}
{scope_note}
{combined_context}
{class_guide}

**Available Tools (combined tools are enabled):**
{tools_list_str}

**COMBINED PLANNING RULES:**
- Prefer cross-simulation tools appropriate to this stage.
- In PRE-COMBINED, pick the pocket-definition / MSA / mapping tools (or other
  shared setup) that every later simulation will reuse. Extra combined tools
  you choose are kept and executed once.
- For post overlays use run_combined_analysis with metrics limited to what was requested
  (e.g. metrics=["rmsd","rmsf"] only) and the exact sim_dirs/labels listed above.
- Classification / clustering tools are allowed ONLY when the objective explicitly asks to
  classify or cluster simulations — never because the goal mentions an HPC cluster.
- Use working_dir="." for combined tools (outputs go to the combined analysis directory).
- overview MUST be a single string (not a JSON array).

Output as JSON:
{{
  "reasoning": "How you will fulfill the combined objective",
  "overview": "High-level summary",
  "steps": [
    {{
      "name": "step name",
      "description": "what it does",
      "tool_name": "tool to call",
      "tool_params": {{"param": "value"}},
      "reason": "why this step is needed"
    }}
  ],
  "potential_issues": [],
  "recommendations": []
}}
"""

    def _create_pre_combined_recipe_plan(
        self,
        agent_input: AnalysisAgentInput,
        state: MDState,
    ) -> AnalysisPlan:
        """Execute the compiled pre-combined protocol (LLM extras + required shared setup)."""
        from agentic.campaign.pre_combined import (
            family_pre_combined_pin_steps,
            merge_pre_combined_steps,
        )
        from agentic.campaign.spec import spec_from_state
        from agentic.reporter.reporter_agent import resolve_combined_sim_context

        spec = spec_from_state(state)
        sim_dirs, labels = resolve_combined_sim_context(state)
        from agentic.campaign.config import load_campaign_settings
        from agentic.planner.planning_guidelines import detect_consensus_pocket_requested

        settings = load_campaign_settings()
        pocket_req = detect_consensus_pocket_requested(
            *collect_goal_texts_for_intent(state, agent_input)
        )
        pin_settings = {
            "pocket_cutoff_A": float(
                pocket_req.get("pocket_cutoff_A")
                or getattr(settings, "pocket_cutoff_A", 15.0)
            ),
            "conservation_metric": getattr(settings, "conservation_metric", "similarity"),
            "min_conservation": getattr(settings, "min_conservation", 0.5),
            "min_coverage": getattr(settings, "min_coverage", 0.25),
        }
        ref = (
            pocket_req.get("reference_label")
            or (spec.reference_label if spec else None)
            or ""
        )
        steps: List[Dict[str, Any]] = []
        if spec and spec.pre_combined_recipe.steps:
            steps = [s.to_dict() for s in spec.pre_combined_recipe.steps]
        if not steps:
            steps = family_pre_combined_pin_steps(
                labels=labels,
                sim_dirs=sim_dirs,
                reference_label=str(ref or ""),
                settings=pin_settings,
            )
        pin = bool(spec and (spec.family_modular or spec.pin_pre_tools))
        steps = merge_pre_combined_steps(
            steps,
            pin_steps=family_pre_combined_pin_steps(
                labels=labels,
                sim_dirs=sim_dirs,
                reference_label=str(ref or ""),
                settings=pin_settings,
            ),
            labels=labels,
            sim_dirs=sim_dirs,
            pin=pin,
        )
        return AnalysisPlan(
            reasoning="Compiled pre-combined recipe (LLM-chosen combined tools + pins).",
            overview="Shared pocket / MSA / campaign setup for every simulation",
            steps=[
                AnalysisStep(
                    name=step.get("name", "unknown"),
                    description=step.get("description", ""),
                    tool_name=step.get("tool_name", ""),
                    tool_params=step.get("tool_params", {}),
                    reason=step.get("reason", ""),
                )
                for step in steps
            ],
            potential_issues=[],
            recommendations=[],
        )

    def _persist_pre_combined_recipe(
        self,
        state: MDState,
        plan: AnalysisPlan,
    ) -> None:
        """Store the compiled pre_combined tool list so retries do not re-shop."""
        from agentic.campaign.pre_combined import recipe_from_step_dicts
        from agentic.campaign.spec import spec_from_state

        spec = spec_from_state(state)
        if spec is None:
            from agentic.campaign.compile import compile_campaign_spec

            spec = compile_campaign_spec(state)
        spec.pre_combined_recipe = recipe_from_step_dicts(
            [
                {
                    "name": s.name,
                    "description": s.description,
                    "tool_name": s.tool_name,
                    "tool_params": dict(s.tool_params or {}),
                    "reason": s.reason,
                }
                for s in plan.steps
            ]
        )
        spec.pre_combined_compiled = True
        state["campaign_spec"] = spec.to_dict()
        try:
            from agentic.multi_sim_paths import resolve_multi_sim_base_dir

            base = resolve_multi_sim_base_dir(state)
            path = Path(base) / "planner" / "campaign_spec.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Could not persist pre_combined recipe: %s", exc)

    def _create_combined_analysis_plan_llm(
        self,
        agent_input: AnalysisAgentInput,
        state: MDState,
        task: str,
    ) -> AnalysisPlan:
        """LLM JSON plan for workflow combined analysis with tools exposed."""
        prompt = self._build_combined_analysis_planning_prompt(task, agent_input, state)
        try:
            plan_dict = self._llm_plan_json_with_retry(
                prompt, "analysis.combined_planning", temperature=0.2, max_tokens=16384
            )
            plan_dict.pop("_raw_content", None)
            plan_dict = self._normalize_plan_steps(plan_dict, agent_input)
            plan_dict = self._normalize_hitl_plan_dict(plan_dict)
            plan_dict = self._sanitize_combined_hitl_plan(plan_dict, state, task)
            # Intent filter without treating this as HITL chat (no hitl_chat_task).
            plan_dict = self._filter_plan_steps_by_intent(plan_dict, state, agent_input)
            if not plan_dict.get("steps"):
                logger.warning(
                    "Combined LLM plan empty after sanitize/filter; using fallback plan"
                )
                return self._create_combined_hitl_fallback_plan(agent_input, state)
            steps = [
                AnalysisStep(
                    name=step.get("name") or "unknown",
                    description=step.get("description") or "",
                    tool_name=step.get("tool_name") or "",
                    tool_params=step.get("tool_params") or {},
                    reason=step.get("reason") or "",
                )
                for step in plan_dict.get("steps", [])
                if (step.get("tool_name") or "").strip()
            ]
            if not steps:
                logger.warning(
                    "Combined LLM plan had no valid tool_name steps; using fallback plan"
                )
                return self._create_combined_hitl_fallback_plan(agent_input, state)
            return AnalysisPlan(
                reasoning=plan_dict.get("reasoning", "Combined LLM plan"),
                overview=plan_dict.get("overview", "Combined multi-simulation analysis"),
                steps=steps,
                potential_issues=plan_dict.get("potential_issues", []),
                recommendations=plan_dict.get("recommendations", []),
            )
        except Exception as exc:
            logger.warning("Combined LLM planning failed, using fallback: %s", exc)
            return self._create_combined_hitl_fallback_plan(agent_input, state)

    def _mark_combined_analysis_complete(
        self,
        state: MDState,
        *,
        analysis_dir: str,
        sim_dirs: List[str],
        labels: List[str],
        success: bool,
    ) -> None:
        """Persist disk markers and advance multi-sim phase (pre → sims, post → reporter)."""
        import json as _json
        from datetime import datetime as _dt

        is_pre = self._is_pre_combined_phase(state)
        marker_name = (
            "pre_combined_analysis_complete.json"
            if is_pre
            else "combined_analysis_complete.json"
        )
        marker = Path(analysis_dir) / marker_name
        step_success = bool(success)

        # Post-combined: prefer on-disk overlays/stats over LLM step success
        # (optional compute_comparison_table / multipanel often fail harmlessly).
        if not is_pre:
            try:
                from agentic.multi_sim_progress import post_combined_artifacts_ok

                _fam = False
                try:
                    from agentic.campaign.spec import spec_from_state as _sfs

                    _sp = _sfs(state)
                    _fam = bool(_sp and _sp.family_modular)
                except Exception:
                    _fam = False
                artifact_ok = post_combined_artifacts_ok(
                    analysis_dir, expected_labels=labels, family_modular=_fam
                )
            except Exception:
                artifact_ok = False
            success = bool(step_success) or bool(artifact_ok)
            if artifact_ok and not step_success:
                logger.info(
                    "Post-combined: step success=%s but overlays/stats on disk — "
                    "treating as success",
                    step_success,
                )

        try:
            marker.write_text(
                _json.dumps(
                    {
                        "success": success,
                        "step_success": step_success,
                        "timestamp": _dt.now().isoformat(timespec="seconds"),
                        "labels": labels,
                        "sim_dirs": sim_dirs,
                        "planning_mode": "llm",
                        "stage": "pre_combined" if is_pre else "post_combined",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not write %s: %s", marker_name, exc)

        progress = state.get("multi_sim_progress") or {}

        if is_pre:
            from agentic.multi_sim_paths import resolve_multi_sim_base_dir
            from src.analysis.cross_sim_artifacts import (
                harvest_pre_artifacts_into_cross_sim,
                mark_pre_combined_complete,
                normalize_pre_artifacts_to_contract,
            )

            base = resolve_multi_sim_base_dir(state)
            try:
                copied = harvest_pre_artifacts_into_cross_sim(
                    base, search_roots=[analysis_dir, base]
                )
                normalize_pre_artifacts_to_contract(base)
                try:
                    from src.analysis.inventory import write_stage_inventory

                    write_stage_inventory(
                        stage_dir=Path(base) / "cross_sim",
                        stage="cross_sim",
                        sim_root=base,
                        label="campaign",
                    )
                except Exception as inv_exc:
                    logger.debug("cross_sim inventory write skipped: %s", inv_exc)
                from src.analysis.cross_sim_artifacts import (
                    discover_cross_sim_artifacts,
                    pre_combined_done_on_disk,
                )

                artifacts = discover_cross_sim_artifacts(base)
                artifact_ok = bool(
                    artifacts.get("pocket_map_path")
                    or artifacts.get("msa_fasta_path")
                    or artifacts.get("consensus_residues_path")
                    or copied
                )
                # Prefer on-disk contract over step-level success (LLM may include
                # optional traj steps that fail while MSA/pocket succeeded).
                effective_success = bool(success) or artifact_ok
                # If nothing useful was produced, keep success=False so the
                # supervisor does not advance past pre_combined.
                if not artifact_ok:
                    effective_success = False
                mark_pre_combined_complete(
                    base,
                    labels=labels,
                    extra={
                        "success": effective_success,
                        "harvested": copied,
                        "analysis_dir": analysis_dir,
                        "step_success": bool(success),
                    },
                )
                success = effective_success
                logger.info(
                    "Pre-combined complete: harvested %s artifact(s) into cross_sim/ "
                    "(success=%s, done_on_disk=%s)",
                    len(copied),
                    effective_success,
                    pre_combined_done_on_disk(base),
                )
            except Exception as exc:
                logger.warning("Pre-combined harvest/mark failed: %s", exc)
                try:
                    mark_pre_combined_complete(
                        base, labels=labels, extra={"success": False, "error": str(exc)}
                    )
                except Exception:
                    pass
                success = False

            pre = progress.setdefault("pre_combined", {})
            pre["analysis"] = "done" if success else "failed"
            if success:
                progress["phase"] = "executing_sims"
                progress["active_agent"] = None
                progress["active_sim_label"] = None
            state["multi_sim_progress"] = progress
            # Keep multi_sim_phase=pre_combined until supervisor advances so
            # the supervisor branch that checks pre_combined_done_on_disk runs.
            state["run_pre_combined"] = True
            return

        # Post-combined: only advance to reporter when artifacts (or step success)
        # justify it. Callers pass effective success after overlay checks.
        combined = progress.setdefault("combined", {})
        if success:
            combined["analysis"] = "done"
            combined["reporter"] = combined.get("reporter") or "pending"
            progress["phase"] = "combined_reporter"
            progress["active_agent"] = "reporter"
            progress["active_sim_label"] = None
            state["multi_sim_progress"] = progress
            state["multi_sim_phase"] = "combined_reporter"
            state["current_agent_idx"] = 1
            state["run_combined_analysis"] = True
            state["run_post_combined"] = True
        else:
            combined["analysis"] = "failed"
            progress["phase"] = "combined_analysis"
            progress["active_agent"] = "analysis"
            state["multi_sim_progress"] = progress
            state["multi_sim_phase"] = "combined_analysis"
            logger.warning(
                "Post-combined marked unsuccessful (no usable overlays/stats yet)"
            )

    def _populate_combined_results_from_execution(
        self,
        state: MDState,
        *,
        sim_dirs: List[str],
        labels: List[str],
        analysis_dir: str,
        result: AnalysisExecutionResult,
    ) -> None:
        """Build analysis_results['combined'] from LLM execution + on-disk overlays."""
        plots: List[str] = []
        tables: List[str] = []
        for path in (result.generated_files or {}):
            p = str(path)
            lower = p.lower()
            if lower.endswith((".png", ".pdf", ".svg")):
                plots.append(p)
            elif lower.endswith((".csv", ".dat", ".tsv", ".xlsx")):
                tables.append(p)

        adir = Path(analysis_dir)
        if adir.is_dir():
            for pattern in ("*overlay*.png", "*_comparison*.png", "*apo*holo*.png"):
                for p in sorted(adir.glob(pattern)):
                    sp = str(p)
                    if sp not in plots:
                        plots.append(sp)
            for pattern in ("*_stats.csv", "classification_features*.csv"):
                for p in sorted(adir.glob(pattern)):
                    sp = str(p)
                    if sp not in tables:
                        tables.append(sp)

        analysis_results = state.get("analysis_results") or {}
        analysis_results["combined"] = {
            "sim_dirs": sim_dirs,
            "labels": labels,
            "overlay_plots": plots,
            "stats_tables": tables,
            "skipped_metrics": [],
            "dccm_plots": [p for p in plots if "dccm" in Path(p).name.lower()],
            "rmsf_apo_holo_plots": [
                p for p in plots if "apo" in Path(p).name.lower() and "holo" in Path(p).name.lower()
            ],
            "apo_holo_pairs": [],
            "rmsf_segment_plots": [],
            "com_distance_plot": next(
                (p for p in plots if "com" in Path(p).name.lower() or "pocket_distance" in Path(p).name.lower()),
                None,
            ),
            "dssp_plots": [p for p in plots if "dssp" in Path(p).name.lower()],
            "classification_features_table": next(
                (t for t in tables if "classification_features" in Path(t).name.lower()),
                None,
            ),
            "classification_clustering": None,
            "classification_metric_groups": None,
            "analysis_dir": analysis_dir,
            "planning_mode": "llm",
        }
        state["analysis_results"] = analysis_results
        state["figures"] = list(state.get("figures") or []) + [
            p for p in plots if p not in (state.get("figures") or [])
        ]

    def _run_combined_analysis_via_llm(self, state: MDState) -> MDState:
        """
        Combined analysis via LLM planning with combined tools exposed.

        Used for both ``pre_combined`` and ``post_combined`` / ``combined_analysis``.
        Falls back to the deterministic pipeline if planning/execution fails (post only).
        """
        from agentic.multi_sim_paths import resolve_multi_sim_base_dir
        from agentic.utils.conversation_logger import set_log_file
        from agentic.reporter.reporter_agent import resolve_combined_sim_context
        from src.analysis.cross_sim_artifacts import ensure_cross_sim_dir

        working_dir = resolve_multi_sim_base_dir(state)
        if state.get("is_multi_simulation"):
            state["multi_sim_base_dir"] = working_dir
            state["working_directory"] = working_dir
        set_log_file(str(Path(working_dir) / "agent_conversation.log"))
        ensure_cross_sim_dir(working_dir)

        analysis_dir = str(Path(working_dir) / "analysis")
        Path(analysis_dir).mkdir(parents=True, exist_ok=True)
        state["analysis_dir"] = analysis_dir
        state["analysis_directory"] = analysis_dir
        state["skip_pbc_wrap"] = True
        state["hitl_view_combined"] = True  # expose combined tools to executor

        is_pre = self._is_pre_combined_phase(state)
        sim_dirs, labels = resolve_combined_sim_context(state)
        task = self._build_combined_analysis_task(state)
        stage_name = (
            "Pre-Combined Analysis (LLM)"
            if is_pre
            else "Combined Multi-Simulation Analysis (LLM)"
        )

        log_agent_start(
            "analysis",
            stage_name,
            {
                "sim_dirs": sim_dirs,
                "labels": labels,
                "output_dir": analysis_dir,
                "planning": "llm",
                "stage": "pre_combined" if is_pre else "post_combined",
            },
        )

        try:
            self.init_for_hitl_execution(state)
            agent_input = self.prepare_agent_input_for_hitl(state, task)
            from agentic.campaign.spec import spec_from_state as _spec_from_state

            _pre_spec = _spec_from_state(state)
            if (
                is_pre
                and _pre_spec
                and _pre_spec.pre_combined_compiled
                and _pre_spec.pre_combined_recipe.steps
            ):
                logger.info(
                    "Pre-combined: executing compiled recipe (%d steps, no re-plan)",
                    len(_pre_spec.pre_combined_recipe.steps),
                )
                plan = self._create_pre_combined_recipe_plan(agent_input, state)
            else:
                plan = self._create_combined_analysis_plan_llm(agent_input, state, task)
                if is_pre and plan.steps:
                    self._persist_pre_combined_recipe(state, plan)

            log_agent_action(
                "analysis",
                f"Generated {'pre-' if is_pre else ''}combined analysis plan",
                {
                    "steps": len(plan.steps),
                    "tools": [s.tool_name for s in plan.steps],
                    "reasoning": (plan.reasoning or "")[:500],
                    "overview": plan.overview,
                },
            )

            exec_plan = state.get("execution_plan") or {}
            exec_plan.setdefault("structured_plans", {})["analysis"] = plan.model_dump()
            exec_plan["format"] = (
                "pre_combined_analysis_llm" if is_pre else "combined_analysis_llm"
            )
            state["execution_plan"] = exec_plan

            if not plan.steps:
                if is_pre:
                    logger.warning(
                        "Pre-combined LLM plan has no steps — trying deterministic fallback"
                    )
                    return self._run_pre_combined_deterministic(
                        state,
                        analysis_dir=analysis_dir,
                        sim_dirs=sim_dirs,
                        labels=labels,
                    )
                logger.warning("Combined LLM plan has no steps — deterministic fallback")
                return self._run_combined_analysis(state)

            result = self._execute_analysis_plan(agent_input, plan, state)
            self._save_hitl_execution_artifacts(plan, result, task)
            if not is_pre:
                self._populate_combined_results_from_execution(
                    state,
                    sim_dirs=sim_dirs,
                    labels=labels,
                    analysis_dir=analysis_dir,
                    result=result,
                )

            state["errors"] = [
                e
                for e in state.get("errors", [])
                if not (
                    e.startswith("Analysis failed:") or e.startswith("Analysis error:")
                )
            ]
            for issue in result.issues or []:
                prefix = "Pre-combined" if is_pre else "Combined analysis"
                state.setdefault("warnings", []).append(f"{prefix}: {issue}")

            success = bool(result.success)
            if is_pre:
                self._mark_combined_analysis_complete(
                    state,
                    analysis_dir=analysis_dir,
                    sim_dirs=sim_dirs,
                    labels=labels,
                    success=success,
                )
                from src.analysis.cross_sim_artifacts import pre_combined_done_on_disk
                from agentic.multi_sim_paths import resolve_multi_sim_base_dir as _base

                if not pre_combined_done_on_disk(_base(state)):
                    logger.warning(
                        "Pre-combined LLM left no usable artifacts — "
                        "running deterministic pocket/MSA fallback"
                    )
                    return self._run_pre_combined_deterministic(
                        state,
                        analysis_dir=analysis_dir,
                        sim_dirs=sim_dirs,
                        labels=labels,
                    )
            else:
                self._mark_combined_analysis_complete(
                    state,
                    analysis_dir=analysis_dir,
                    sim_dirs=sim_dirs,
                    labels=labels,
                    success=success,
                )
                # Re-read effective success after artifact-based mark
                try:
                    from agentic.multi_sim_progress import post_combined_artifacts_ok

                    _fam = False
                    try:
                        from agentic.campaign.spec import spec_from_state as _sfs

                        _sp = _sfs(state)
                        _fam = bool(_sp and _sp.family_modular)
                    except Exception:
                        _fam = False
                    success = success or post_combined_artifacts_ok(
                        analysis_dir, expected_labels=labels, family_modular=_fam
                    )
                except Exception:
                    pass
            log_agent_completion("analysis", stage_name, state, success)
            if hitl_should_interact(state):
                state["hitl_checkpoint_subtype"] = (
                    "pre_combined" if is_pre else "post_combined"
                )
                try:
                    from agentic.hitl_router import bind_combined_hitl_view

                    bind_combined_hitl_view(state, "analysis")
                except Exception:
                    pass
                state["next_node"] = "human_analysis_check"
            elif is_pre:
                # Supervisor advances to per-sim execution after disk marker.
                state["next_node"] = "supervisor"
            elif success:
                state["next_node"] = "reporter"
            else:
                state["next_node"] = "supervisor"
            return state

        except Exception as exc:
            import traceback

            logger.error(
                "%s failed (%s) — %s\n%s",
                stage_name,
                exc,
                "deterministic pre fallback" if is_pre else "falling back to deterministic pipeline",
                traceback.format_exc(),
            )
            if is_pre:
                state.setdefault("warnings", []).append(
                    f"Pre-combined analysis failed ({exc}); trying deterministic fallback"
                )
                return self._run_pre_combined_deterministic(
                    state,
                    analysis_dir=analysis_dir,
                    sim_dirs=sim_dirs,
                    labels=labels,
                )
            state.setdefault("warnings", []).append(
                f"Combined LLM analysis failed ({exc}); using deterministic fallback"
            )
            return self._run_combined_analysis(state)

    def _llm_select_classification_features(
        self,
        *,
        analysis_dir: str,
        class_result: Dict[str, Any],
        user_goal: str,
        metric_groups: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        state: Optional[MDState] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to choose clustering features with written scientific reasoning.

        Uses the collected raw feature CSV + column definitions. Prefers
        dynamics descriptors similar to comparative kinase studies (pocket COM /
        orientation, consensus RMSF, pocket χ₁, N↔C DCCM, dihedral landscape
        entropy) but may add/drop columns when justified. Writes
        ``classification_feature_selection.json``.
        """
        if not (self.llm and getattr(self.llm, "available", True)):
            return None

        raw_path = Path(class_result.get("output_file") or "")
        if not raw_path.is_file():
            raw_path = Path(analysis_dir) / "classification_features.csv"
        if not raw_path.is_file():
            return None

        from src.analysis.classification_collector import (
            CLASSIFICATION_FEATURE_DEFINITIONS,
            PAPER_WARD4_FEATURE_COLUMNS,
        )

        # Presence stats per column
        import csv as _csv
        import numpy as np

        with open(raw_path, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        meta = {"label", "sim_directory", "n_features_present"}
        candidates: List[Dict[str, Any]] = []
        unavailable: List[Dict[str, Any]] = []
        for col in fieldnames:
            if col in meta:
                continue
            present = 0
            for r in rows:
                v = (r.get(col) or "").strip()
                if not v or v.lower() in {"nan", "none", "null"}:
                    continue
                try:
                    if np.isfinite(float(v)):
                        present += 1
                except (TypeError, ValueError):
                    continue
            meta_def = CLASSIFICATION_FEATURE_DEFINITIONS.get(col, {})
            entry = {
                "column": col,
                "n_present": present,
                "n_sims": len(rows),
                "description": meta_def.get("description", ""),
                "metric_group": meta_def.get("metric_group", ""),
                "unit": meta_def.get("unit", ""),
                "paper_ward4_like": col in PAPER_WARD4_FEATURE_COLUMNS
                or col.replace(
                    "mean_ligand_axis_angle", "ligand_axis_angle_mean"
                )
                in PAPER_WARD4_FEATURE_COLUMNS,
            }
            if present < 2:
                unavailable.append(entry)
                continue
            candidates.append(entry)
        if len(candidates) < 2:
            return None

        paper_hint = ", ".join(PAPER_WARD4_FEATURE_COLUMNS)
        unavailable_names = [u["column"] for u in unavailable]
        prompt = f"""You are selecting features for unsupervised hierarchical clustering of protein MD simulations.

USER GOAL (excerpt):
{(user_goal or "")[:2500]}

REQUESTED METRIC GROUPS: {sorted(metric_groups or [])}

AVAILABLE FEATURE COLUMNS (only those with ≥2 finite values — you may ONLY choose from these):
{json.dumps(candidates, indent=2)[:12000]}

UNAVAILABLE / EMPTY COLUMNS (all-NaN or <2 finite values — do NOT claim these were used):
{json.dumps(unavailable_names, indent=2)[:4000]}

PAPER-STYLE REFERENCE SET (guidance only — do NOT invent or require missing columns):
{paper_hint}

Rules:
1. Prefer comparative dynamics descriptors when present: pocket–ligand COM mean/std, axis-angle mean/std,
   consensus RMSF mean/std, pocket χ₁, N↔C DCCM correlation, dihedral PCA grid entropy.
2. You may include closely related extras or drop redundant/static columns.
3. Prefer reference_pocket_* COM/angle columns over plain ligand_pocket_* when both exist.
4. Prefer chi1_pocket_circ_mean_deg over domain-wide chi1_circ_mean_deg when both exist.
5. Select typically 4–12 columns from AVAILABLE only; never invent names.
6. If UNAVAILABLE is non-empty, say so explicitly in dropped_rationale (do not say "all nine available").
7. Write concise scientific reasoning for the selection.

Return ONLY JSON:
{{
  "reasoning": "2–6 sentences explaining the scientific rationale",
  "feature_columns": ["col_a", "col_b", "..."],
  "dropped_rationale": "note missing/empty columns and any deliberate drops"
}}
"""
        if state is not None:
            try:
                from agentic.campaign.agent_context import agent_context_block

                extra = agent_context_block(state, "analysis", query=(user_goal or "")[:800])
                if extra:
                    prompt = f"{prompt}\n\n{extra}\n"
            except Exception:
                logger.debug("analysis feature-selection context skipped", exc_info=True)
        try:
            content = self.llm.prompt_raw(
                prompt, temperature=0.2, max_tokens=4096, format="json"
            )
        except Exception as exc:
            logger.warning("LLM feature selection prompt failed: %s", exc)
            return None

        plan = self._extract_plan_json(content) if content else {}
        if not isinstance(plan, dict):
            return None
        cols = plan.get("feature_columns") or plan.get("selected_features") or []
        if not isinstance(cols, list):
            return None
        allowed = {c["column"] for c in candidates}
        selected = [str(c) for c in cols if str(c) in allowed]
        required = [c for c in (required_columns or []) if c in allowed]
        if required:
            selected = required + [c for c in selected if c not in required]
        # Heuristic fallback if LLM returned nothing usable
        if len(selected) < 2:
            preferred = [
                "reference_pocket_ligand_distance_mean_A",
                "reference_pocket_ligand_distance_std_A",
                "reference_pocket_ligand_axis_angle_mean_deg",
                "reference_pocket_mean_ligand_axis_angle_deg",
                "reference_pocket_std_ligand_axis_angle_deg",
                "consensus_rmsf_mean_A",
                "consensus_rmsf_std_A",
                "chi1_pocket_circ_mean_deg",
                "dccm_N_C_mean_corr",
                "pca_grid_entropy",
                "ligand_pocket_distance_mean_A",
                "ligand_pocket_distance_std_A",
            ]
            selected = [c for c in preferred if c in allowed]
            if len(selected) < 2:
                selected = [c["column"] for c in candidates[:8]]
            plan["reasoning"] = (
                (plan.get("reasoning") or "")
                + " [fallback: preferred dynamics columns applied after invalid LLM list]"
            ).strip()

        out = {
            "reasoning": str(plan.get("reasoning") or "").strip(),
            "feature_columns": selected,
            "dropped_rationale": str(plan.get("dropped_rationale") or "").strip(),
            "n_candidates": len(candidates),
            "n_selected": len(selected),
            "n_unavailable": len(unavailable_names),
            "unavailable_columns": list(unavailable_names),
            "metric_groups": sorted(metric_groups or []),
            "source_features_csv": str(raw_path),
        }
        if unavailable_names:
            miss = ", ".join(unavailable_names)
            note = (
                f"Unavailable (all-NaN or <2 finite values across systems): {miss}."
            )
            prior = out["dropped_rationale"]
            # Strip false "all available" claims when columns were empty.
            if prior and "all" in prior.lower() and "available" in prior.lower():
                prior = ""
            out["dropped_rationale"] = f"{note} {prior}".strip()
            reason = out["reasoning"]
            if reason and "all nine" in reason.lower():
                out["reasoning"] = (
                    reason
                    + f" Note: {len(unavailable_names)} requested column(s) were empty "
                    f"and could not be used: {miss}."
                )
        try:
            out_path = Path(analysis_dir) / "classification_feature_selection.json"
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            out["selection_file"] = str(out_path)
        except Exception as exc:
            logger.debug("Could not write feature selection JSON: %s", exc)
        return out

    def _run_pre_combined_deterministic(
        self,
        state: MDState,
        *,
        analysis_dir: str,
        sim_dirs: List[str],
        labels: List[str],
    ) -> MDState:
        """Deterministic MSA + pocket map when LLM pre_combined fails or is empty."""
        from src.analysis.consensus_alignment import build_global_mapped_alignment
        from src.analysis.consensus_pocket import (
            define_pocket_mapped_residues,
            map_pocket_mapped_residues,
        )
        from src.analysis.cross_sim_artifacts import pre_combined_done_on_disk
        from src.analysis.phylo_tree import resolve_structure_pdb
        from agentic.multi_sim_paths import resolve_multi_sim_base_dir

        base = resolve_multi_sim_base_dir(state)
        Path(analysis_dir).mkdir(parents=True, exist_ok=True)

        ref_label = labels[0] if labels else "reference"
        try:
            from agentic.planner.planning_guidelines import _resolve_reference_label

            parsed_ref = _resolve_reference_label(
                state.get("user_goal_original") or "",
                state.get("user_goal") or "",
                labels=labels,
                default="",
            )
            if parsed_ref:
                ref_label = parsed_ref
        except Exception:
            pass

        pdb_by_label = {
            str(sp.get("label")): sp.get("pdb")
            for sp in (state.get("sim_prompts") or [])
            if sp.get("label") and sp.get("pdb")
        }
        pdb_files: List[str] = []
        for sim_dir, label in zip(sim_dirs, labels):
            pdb = pdb_by_label.get(str(label))
            if not pdb or not Path(str(pdb)).is_file():
                pdb = resolve_structure_pdb(str(label), str(sim_dir), str(base))
            if pdb and Path(pdb).is_file():
                pdb_files.append(str(Path(pdb).resolve()))

        issues: List[str] = []
        try:
            align_kwargs: Dict[str, Any] = {
                "working_dir": analysis_dir,
                "reference_label": ref_label,
                "labels": labels,
                "sim_dirs": sim_dirs,
                "base_dir": str(base),
                "alignment_json": "global_consensus_msa.json",
                "consensus_json": "global_consensus_msa.json",
                "alignment_fasta": "global_msa.fasta",
                "residue_map_csv": "global_consensus_msa.csv",
                "conservation_metric": "similarity",
                "min_conservation": 0.5,
                "min_coverage": 0.25,
            }
            # Prefer resolved PDBs; if that fails (blank chains / bad paths),
            # retry with sim_dirs-only resolution.
            attempts: List[Dict[str, Any]] = []
            if len(pdb_files) == len(labels) and labels:
                attempts.append({**align_kwargs, "pdb_files": pdb_files})
            attempts.append(dict(align_kwargs))  # sim_dirs / base_dir only

            align_res: Dict[str, Any] = {}
            for kw in attempts:
                align_res = build_global_mapped_alignment.invoke(kw) or {}
                if align_res.get("success"):
                    break
                issues.append(
                    f"build_global_mapped_alignment: "
                    f"{align_res.get('error') or align_res}"
                )

            if not align_res.get("success"):
                pass  # issues already recorded
            else:
                # Clear transient align errors if a later attempt succeeded.
                issues = [
                    i
                    for i in issues
                    if "build_global_mapped_alignment" not in i
                    and "build_consensus_sequence_alignment" not in i
                ]
                ref_sim = sim_dirs[labels.index(ref_label)] if ref_label in labels else sim_dirs[0]
                from agentic.campaign.ligand_context import resolve_ligand_selection

                _lig = resolve_ligand_selection(
                    None, state, allow_atp_default=False
                )
                if not _lig:
                    # Infer from holo-style labels only (never invent for apo)
                    joined = " ".join(str(x) for x in labels).lower()
                    if any(tok in joined for tok in ("_atp", "atp_mg", "_gtp", "holo")):
                        _lig = "resname ATP" if "gtp" not in joined else "resname GTP"
                if not _lig:
                    logger.info(
                        "Skipping define_pocket_mapped_residues — no ligand declared"
                    )
                    pocket_res = {"success": True, "skipped": True}
                else:
                    pocket_res = define_pocket_mapped_residues.invoke(
                        {
                            "working_dir": analysis_dir,
                            "reference_label": ref_label,
                            "sim_dir": ref_sim,
                            "consensus_json": "global_consensus_msa.json",
                            "ligand_selection": _lig,
                            "pocket_cutoff_A": 15,
                            "pocket_filter": "global_consensus_msa",
                        }
                    )
                if (pocket_res or {}).get("skipped"):
                    pass
                elif not (pocket_res or {}).get("success"):
                    issues.append(
                        f"define_pocket_mapped_residues: "
                        f"{(pocket_res or {}).get('error') or pocket_res}"
                    )
                else:
                    map_res = map_pocket_mapped_residues.invoke(
                        {
                            "working_dir": analysis_dir,
                            "definition_json": "pocket_mapped_definition.json",
                            "labels": labels,
                        }
                    )
                    if not (map_res or {}).get("success"):
                        issues.append(
                            f"map_pocket_mapped_residues: "
                            f"{(map_res or {}).get('error') or map_res}"
                        )
                    else:
                        try:
                            from src.analysis.msa_plotting import (
                                plot_global_mapped_alignment,
                            )

                            plot_res = plot_global_mapped_alignment.invoke(
                                {
                                    "working_dir": analysis_dir,
                                    "alignment_fasta": "global_msa.fasta",
                                    "consensus_json": "global_consensus_msa.json",
                                    "pocket_definition_json": (
                                        "pocket_mapped_definition.json"
                                    ),
                                    "full_plot_file": "global_consensus_msa.png",
                                    "focused_plot_file": "pocket_mapped_msa.png",
                                }
                            )
                            if not (plot_res or {}).get("success"):
                                issues.append(
                                    f"plot_global_mapped_alignment: "
                                    f"{(plot_res or {}).get('error') or plot_res}"
                                )
                        except Exception as plot_exc:
                            issues.append(f"plot_global_mapped_alignment: {plot_exc}")
        except Exception as exc:
            issues.append(str(exc))
            logger.exception("Deterministic pre_combined failed")

        for issue in issues:
            state.setdefault("warnings", []).append(f"Pre-combined deterministic: {issue}")

        # Always harvest whatever was written under analysis/ into cross_sim/.
        try:
            from src.analysis.cross_sim_artifacts import (
                harvest_pre_artifacts_into_cross_sim,
                normalize_pre_artifacts_to_contract,
                cross_sim_artifacts_ready,
            )

            copied = harvest_pre_artifacts_into_cross_sim(
                base, search_roots=[analysis_dir, base]
            )
            normalize_pre_artifacts_to_contract(base)
            # Mirror MSA JSON as consensus_residues.json when only reference_* exists.
            cross = Path(base) / "cross_sim"
            msa = cross / "global_consensus_msa.json"
            if not msa.is_file():
                msa = cross / "global_mapped.json"
            cons = cross / "consensus_residues.json"
            ref = cross / "reference_msa_alignment.json"
            if msa.is_file():
                if not cons.is_file():
                    shutil.copy2(msa, cons)
                if not ref.is_file():
                    shutil.copy2(msa, ref)
                # Keep legacy global_mapped.json for in-flight consumers.
                legacy = cross / "global_mapped.json"
                if msa.name != "global_mapped.json" and not legacy.is_file():
                    shutil.copy2(msa, legacy)
            elif ref.is_file() and not cons.is_file():
                shutil.copy2(ref, cons)
            logger.info(
                "Deterministic pre_combined harvest: %s file(s), ready=%s",
                len(copied),
                cross_sim_artifacts_ready(base),
            )
        except Exception as harvest_exc:
            issues.append(f"harvest: {harvest_exc}")
            logger.warning("pre_combined harvest failed: %s", harvest_exc)

        self._mark_combined_analysis_complete(
            state,
            analysis_dir=analysis_dir,
            sim_dirs=sim_dirs,
            labels=labels,
            success=len(issues) == 0,
        )

        from src.analysis.cross_sim_artifacts import (
            MAX_PRE_COMBINED_ATTEMPTS,
            cross_sim_artifacts_ready,
            mark_pre_combined_complete,
            pre_combined_attempt_count,
            pre_combined_done_on_disk,
        )

        if cross_sim_artifacts_ready(base):
            mark_pre_combined_complete(
                base,
                labels=labels,
                extra={"success": True, "mode": "deterministic"},
            )
        else:
            # Count on disk. The in-memory counter is dropped between supervisor
            # hops, which retried this step until the graph recursion limit.
            n_try = pre_combined_attempt_count(base) + 1
            state["pre_combined_attempts"] = n_try
            mark_pre_combined_complete(
                base,
                labels=labels,
                extra={
                    "success": False,
                    "skipped": n_try >= MAX_PRE_COMBINED_ATTEMPTS,
                    "reason": "deterministic_failed",
                    "issues": issues,
                    "attempts": n_try,
                },
            )
            if n_try >= MAX_PRE_COMBINED_ATTEMPTS:
                # Force-done escape: write a marker file that supervisor checks
                # via attempts, while still exposing failure in JSON.
                logger.error(
                    "Deterministic pre_combined failed after %s attempts — "
                    "continuing without full cross_sim (issues=%s)",
                    n_try,
                    issues,
                )
                state.setdefault("warnings", []).append(
                    "Pre-combined failed after retries; consensus metrics may be unavailable"
                )
                # Write allow-continue sentinel so done_on_disk can pass once.
                (Path(base) / "cross_sim" / "pre_combined_give_up.json").write_text(
                    json.dumps({"attempts": n_try, "issues": issues}, indent=2),
                    encoding="utf-8",
                )

        log_agent_completion(
            "analysis",
            "Pre-Combined Analysis (deterministic)",
            state,
            pre_combined_done_on_disk(base),
        )
        state["next_node"] = "supervisor"
        return state

    def _run_combined_analysis(self, state: MDState) -> MDState:
        """
        Deterministic cross-simulation combined analysis (fallback / legacy).

        Collects per-sim data files, produces overlay plots and stats CSVs
        in ``{working_directory}/analysis/``, and stores the results in state
        so the reporter can embed them in the combined report.
        """
        from .tools import run_combined_analysis, collect_metric_files

        from agentic.multi_sim_paths import resolve_multi_sim_base_dir
        from agentic.utils.conversation_logger import set_log_file

        working_dir = resolve_multi_sim_base_dir(state)
        if state.get("is_multi_simulation"):
            state["multi_sim_base_dir"] = working_dir
            state["working_directory"] = working_dir
        # Combined phase always records to the campaign base log.
        set_log_file(str(Path(working_dir) / "agent_conversation.log"))
        analysis_dir = str(Path(working_dir) / "analysis")
        Path(analysis_dir).mkdir(parents=True, exist_ok=True)
        state["analysis_dir"] = analysis_dir
        state["analysis_directory"] = analysis_dir

        # Resolve per-sim directories and labels from completed_sim_states
        from agentic.reporter.reporter_agent import resolve_combined_sim_context

        sim_dirs, labels = resolve_combined_sim_context(state)
        seen_labels: set = set(labels)

        # Fall back already handled inside resolve_combined_sim_context.
        if not sim_dirs:
            completed = state.get("completed_sim_states") or []
            for snap in completed:
                label = snap.get("label")
                wd = snap.get("working_directory")
                if not label or not wd or label in seen_labels:
                    continue
                seen_labels.add(label)
                sim_dirs.append(wd)
                labels.append(label)
            if not sim_dirs:
                sim_dirs = state.get("sim_working_dirs") or []
                sim_prompts = state.get("sim_prompts") or []
                labels = [p.get("label", f"sim_{i}") for i, p in enumerate(sim_prompts)]

        # Filesystem / tool labels must stay as on-disk directory names
        # (e.g. p17612_ATP). Display remapping is applied only for plots/reports.
        dir_labels = [
            Path(wd).name if wd else str(lab)
            for wd, lab in zip(sim_dirs, labels)
        ]
        if not dir_labels:
            dir_labels = list(labels)

        # Apply protein name mapping from goal text (fallback for re-runs where
        # labels may still be raw UniProt IDs from a prior planner run).
        from src.reporter.combined_reporter import _parse_label_name_map, apply_label_name_map
        _nm_text = (
            (state.get("user_goal_original") or "")
            + " "
            + (state.get("master_enriched_prompt") or "")
            + " "
            + (state.get("user_goal", "") or "")
        )
        _name_map_a = _parse_label_name_map(_nm_text)
        if _name_map_a:
            labels = apply_label_name_map(labels, _name_map_a)
        _goal_text = _nm_text.strip()

        log_agent_start(
            "analysis",
            "Combined Multi-Simulation Analysis",
            {"sim_dirs": sim_dirs, "labels": labels, "output_dir": analysis_dir},
        )

        try:
            from .tools import (
                run_combined_dccm_analysis,
                run_combined_dccm_difference,
                run_combined_rmsf_segment_analysis,
                run_combined_com_distance_analysis,
                run_combined_binding_rmsf_overlay,
                pair_apo_holo_simulations,
                run_combined_rmsf_apo_holo_analysis,
                run_combined_dccm_apo_holo_analysis,
                run_combined_rmsf_segment_apo_holo_analysis,
            )

            # pair_apo_holo_simulations is a plain helper (not a @tool), so it
            # is called directly without .func.
            apo_holo_pairs = pair_apo_holo_simulations(
                sim_dirs=sim_dirs,
                labels=labels,
                label_name_map=_name_map_a or None,
                user_goal=_goal_text or None,
            )

            combined_plan_text = (
                (state.get("combined_analysis_plan") or "")
                + " "
                + (state.get("user_goal_original") or "")
                + " "
                + (state.get("master_enriched_prompt") or state.get("enriched_prompt") or "")
            ).lower()
            requested = detect_requested_metrics_union(
                state.get("combined_analysis_plan") or "",
                state.get("user_goal_original") or "",
                state.get("master_enriched_prompt") or "",
                state.get("enriched_prompt") or "",
            )
            if requested is None:
                # Fall back to keyword scan on plan text only (not sim metadata blobs).
                metric_patterns = {
                    "rmsd": [r"\brmsd\b", r"root mean square deviation"],
                    "rmsf": [r"\brmsf\b", r"root mean square fluctuation"],
                    "rg": [r"\brg\b", r"radius of gyration", r"\bgyration\b"],
                    "energy": [r"\benergy\b", r"\bedr\b"],
                    "dccm": [r"\bdccm\b", r"cross[-\s]?correlation", r"correlated motion"],
                    "com": [
                        r"\bcom\b",
                        r"center[-\s]?of[-\s]?mass",
                        r"centre[-\s]?of[-\s]?mass",
                        r"ligand[-\s]?pocket[-\s]?distance",
                        r"ligand_pocket_distance",
                    ],
                    "dssp": [r"\bdssp\b", r"secondary[-\s]?structure"],
                    "pocket_rmsf": [r"pocket\s+rmsf", r"pocket_rmsf"],
                    "ligand_rmsf": [r"ligand\s+rmsf", r"ligand_rmsf"],
                }
                requested = frozenset(
                    metric
                    for metric, patterns in metric_patterns.items()
                    if any(re.search(pattern, combined_plan_text) for pattern in patterns)
                ) or None

            class_groups = classification_metric_groups_for_goal(
                state.get("user_goal_original") or "",
                state.get("combined_analysis_plan") or "",
                state.get("master_enriched_prompt") or "",
                state.get("enriched_prompt") or "",
            )

            explicit_metrics_requested = bool(requested)
            broad_dynamics_request = any(
                phrase in combined_plan_text
                for phrase in (
                    "protein dynamics",
                    "dynamic behavior",
                    "dynamic behaviour",
                    "conformational dynamics",
                    "shared dynamics",
                    "dynamic patterns",
                )
            )

            if explicit_metrics_requested and requested:
                combined_metrics = [
                    m for m in ("rmsd", "rmsf", "rg", "energy") if m in requested
                ]
            elif broad_dynamics_request:
                combined_metrics = ["rmsd", "rmsf", "rg"]
                requested = frozenset({"rmsd", "rmsf", "rg", "dccm"})
            else:
                combined_metrics = []
                requested = requested or frozenset()

            # Skip all-simulation RMSF overlay when apo/holo pairs exist —
            # per-protein RMSF comparison is clearer for ligand-effect studies.
            if apo_holo_pairs and "rmsf" in combined_metrics:
                combined_metrics = [m for m in combined_metrics if m != "rmsf"]

            # Deterministic path: keep a short audit note (not a fake LLM call).
            log_agent_action(
                "analysis",
                "Combined analysis deterministic plan",
                {
                    "metrics": combined_metrics,
                    "requested": sorted(requested),
                    "classification": sorted(class_groups) if class_groups else None,
                    "apo_holo_pairs": len(apo_holo_pairs),
                },
            )

            plots = []
            tables = []
            skipped = []
            for metric in combined_metrics:
                metric_dirs, metric_labels = self._sims_for_combined_metric(
                    state, sim_dirs, labels, metric,
                    label_name_map=_name_map_a or None,
                )
                if len(metric_dirs) < 2:
                    logger.info(
                        f"Combined {metric}: need ≥2 simulations; "
                        f"matched {metric_labels}"
                    )
                    skipped.append(metric)
                    continue
                result = run_combined_analysis.func(
                    sim_dirs=metric_dirs,
                    labels=metric_labels,
                    working_dir=analysis_dir,
                    metrics=[metric],
                )
                plots.extend(result.get("plots", []))
                tables.extend(result.get("tables", []))
                skipped.extend(result.get("skipped", []))

            if combined_metrics:
                log_agent_action(
                    agent_name="analysis",
                    action="Combined Analysis Complete",
                    details={
                        "metrics": combined_metrics,
                        "plots": plots,
                        "tables": tables,
                        "skipped": skipped,
                    },
                )
            else:
                log_agent_action(
                    agent_name="analysis",
                    action="Combined Analysis Base Metrics Skipped",
                    details={"reason": "No base overlay metrics requested"},
                )

            # ── Per-protein RMSF apo vs holo ──────────────────────────────
            rmsf_apo_holo_plots: list = []
            if apo_holo_pairs and "rmsf" in requested:
                try:
                    rmsf_ah = run_combined_rmsf_apo_holo_analysis.func(
                        sim_dirs=sim_dirs,
                        labels=labels,
                        working_dir=analysis_dir,
                        label_name_map=_name_map_a or None,
                        user_goal=_goal_text or None,
                    )
                    if rmsf_ah.get("success"):
                        rmsf_apo_holo_plots = rmsf_ah.get("plots", [])
                        plots.extend(rmsf_apo_holo_plots)
                        log_agent_action(
                            "analysis", "Per-protein RMSF apo/holo comparison",
                            {"plots": rmsf_apo_holo_plots, "pairs": apo_holo_pairs},
                        )
                    else:
                        logger.info(f"RMSF apo/holo: {rmsf_ah.get('message')}")
                except Exception as _exc:
                    logger.warning(f"RMSF apo/holo comparison failed: {_exc}")

            # ── DCCM: per-protein apo | holo | Δ triptychs ────────────────
            dccm_plots: list = []

            if "dccm" in requested and apo_holo_pairs:
                try:
                    dccm_ah = run_combined_dccm_apo_holo_analysis.func(
                        sim_dirs=sim_dirs,
                        labels=labels,
                        working_dir=analysis_dir,
                        plot_mode="with_matrices",
                        label_name_map=_name_map_a or None,
                        user_goal=_goal_text or None,
                    )
                    if dccm_ah.get("success"):
                        dccm_plots = dccm_ah.get("plots", [])
                        log_agent_action(
                            "analysis", "Per-protein DCCM apo/holo triptychs",
                            {"plots": dccm_plots, "pairs": apo_holo_pairs},
                        )
                    else:
                        logger.warning(f"DCCM apo/holo: {dccm_ah.get('message')}")
                except Exception as _exc:
                    logger.warning(f"DCCM apo/holo analysis failed: {_exc}")
            elif "dccm" in requested:
                dccm_dirs, dccm_labels = self._sims_for_combined_metric(
                    state, sim_dirs, labels, "dccm", label_name_map=_name_map_a or None,
                )
                if len(dccm_dirs) < 2:
                    logger.info(
                        "DCCM comparison skipped: need ≥2 simulations with per-sim DCCM "
                        f"in their goals; matched {dccm_labels}"
                    )
                else:
                    try:
                        dccm_cmp = run_combined_dccm_analysis.func(
                            sim_dirs=dccm_dirs,
                            labels=dccm_labels,
                            working_dir=analysis_dir,
                            output_file="dccm_comparison.png",
                        )
                        if dccm_cmp.get("success"):
                            paths = dccm_cmp.get("output_paths") or [
                                dccm_cmp["output_path"]
                            ]
                            dccm_plots.extend(p for p in paths if p)
                            log_agent_action(
                                "analysis", "DCCM comparison generated",
                                {
                                    "output": dccm_cmp.get("output_path"),
                                    "outputs": paths,
                                    "n_pages": dccm_cmp.get("n_pages", 1),
                                    "simulations": dccm_labels,
                                },
                            )
                        else:
                            logger.warning(f"DCCM comparison: {dccm_cmp.get('message')}")
                    except Exception as _exc:
                        logger.warning(f"DCCM comparison failed: {_exc}")

            # ── RMSF segment bar plots (from user-specified residue ranges) ─
            segment_plots: list = []
            try:
                if "rmsf" not in requested:
                    rmsf_seg = {"success": False, "message": "RMSF was not requested"}
                elif apo_holo_pairs:
                    rmsf_seg = run_combined_rmsf_segment_apo_holo_analysis.func(
                        sim_dirs=sim_dirs,
                        labels=labels,
                        working_dir=analysis_dir,
                        user_goal=_goal_text,
                        segments=state.get("rmsf_segments"),
                        label_name_map=_name_map_a or None,
                    )
                else:
                    rmsf_seg = run_combined_rmsf_segment_analysis.func(
                        sim_dirs=sim_dirs,
                        labels=labels,
                        working_dir=analysis_dir,
                        user_goal=_goal_text,
                        segments=state.get("rmsf_segments"),
                    )
                if rmsf_seg.get("success"):
                    segment_plots = rmsf_seg.get("plots", [])
                    plots.extend(segment_plots)
                    log_agent_action(
                        "analysis", "RMSF segment bar plots generated",
                        {"plots": segment_plots, "segments": rmsf_seg.get("segments", [])},
                    )
                else:
                    logger.info(f"RMSF segments: {rmsf_seg.get('message')}")
            except Exception as _exc:
                logger.warning(f"RMSF segment analysis failed: {_exc}")

            # ── ATP–pocket COM distance overlay (apo vs holo) ───────────────
            com_plot: Optional[str] = None
            try:
                if "com" not in requested:
                    com_result = {"success": False, "message": "COM distance was not requested"}
                else:
                    com_dirs, com_labels = self._sims_for_combined_metric(
                        state, sim_dirs, labels, "com",
                        label_name_map=_name_map_a or None,
                    )
                    com_result = run_combined_com_distance_analysis.func(
                        sim_dirs=com_dirs,
                        labels=com_labels,
                        working_dir=analysis_dir,
                    )
                if com_result.get("success"):
                    com_plot = com_result.get("output_path")
                    if com_plot and com_plot not in plots:
                        plots.append(com_plot)
                    found_com_files = com_result.get("found_files") or []
                    if len(found_com_files) >= 2:
                        try:
                            from .tools import compute_comparison_table
                            com_labels = []
                            for fpath in found_com_files:
                                sim_root = str(Path(fpath).parent.parent)
                                idx = sim_dirs.index(sim_root) if sim_root in sim_dirs else -1
                                com_labels.append(
                                    labels[idx] if 0 <= idx < len(labels) else Path(sim_root).name
                                )
                            table_result = compute_comparison_table.func(
                                data_files=found_com_files,
                                labels=com_labels,
                                output_csv="ligand_pocket_distance_stats.csv",
                                working_dir=analysis_dir,
                            )
                            com_table = table_result.get("output_path") or table_result.get("output_file")
                            if com_table and com_table not in tables:
                                tables.append(com_table)
                        except Exception as table_exc:
                            logger.warning(f"COM comparison table failed: {table_exc}")
                    log_agent_action(
                        "analysis", "COM distance overlay generated",
                        {
                            "output": com_plot,
                            "found": com_result.get("found_files", []),
                            "missing": com_result.get("missing", []),
                        },
                    )
                else:
                    logger.info(f"COM distance overlay: {com_result.get('message')}")
            except Exception as _exc:
                logger.warning(f"COM distance overlay failed: {_exc}")

            # ── Pocket / ligand RMSF combined overlays ───────────────────────
            overlay_metrics = set(requested or [])
            if class_groups is not None:
                overlay_metrics |= set(class_groups)
            for profile_type in ("pocket_rmsf", "ligand_rmsf"):
                if profile_type not in overlay_metrics:
                    continue
                try:
                    rmsf_dirs, rmsf_labels = self._sims_for_combined_metric(
                        state, sim_dirs, labels, profile_type,
                        label_name_map=_name_map_a or None,
                    )
                    if len(rmsf_dirs) < 1:
                        logger.info(
                            "Combined %s: no holo simulations with data", profile_type
                        )
                        continue
                    display_labels = [
                        (_name_map_a or {}).get(lab.lower(), lab)
                        for lab in rmsf_labels
                    ]
                    rmsf_overlay = run_combined_binding_rmsf_overlay.func(
                        sim_dirs=rmsf_dirs,
                        labels=display_labels,
                        working_dir=analysis_dir,
                        profile_type=profile_type,
                    )
                    if rmsf_overlay.get("success"):
                        out_path = rmsf_overlay.get("output_path")
                        if out_path and out_path not in plots:
                            plots.append(out_path)
                        log_agent_action(
                            "analysis",
                            f"Combined {profile_type} overlay generated",
                            {
                                "output": out_path,
                                "n_simulations": rmsf_overlay.get("n_simulations"),
                                "missing": rmsf_overlay.get("missing", []),
                            },
                        )
                    else:
                        logger.info(
                            "Combined %s overlay: %s",
                            profile_type,
                            rmsf_overlay.get("message") or rmsf_overlay.get("error"),
                        )
                except Exception as _exc:
                    logger.warning(f"Combined {profile_type} overlay failed: {_exc}")

            # ── DSSP: backfill missing per-sim runs, comparison chart, activation-loop heatmaps ─
            dssp_plots: list = []
            try:
                if "dssp" not in requested:
                    dssp_result = {"success": False, "message": "DSSP was not requested"}
                else:
                    from src.reporter.combined_reporter import run_combined_dssp_analysis
                    dssp_result = run_combined_dssp_analysis(
                        sim_dirs=sim_dirs,
                        labels=labels,
                        output_dir=analysis_dir,
                        user_goal=_goal_text,
                    )
                dssp_plots = dssp_result.get("plots", [])
                if dssp_plots:
                    plots.extend(p for p in dssp_plots if p not in plots)
                    log_agent_action(
                        "analysis", "Combined DSSP analysis",
                        {
                            "comparison": dssp_result.get("comparison_plot"),
                            "n_heatmaps": len(dssp_result.get("activation_loop_heatmaps", [])),
                            "backfilled": dssp_result.get("backfill", {}).get("backfilled", []),
                        },
                    )
                elif dssp_result.get("backfill", {}).get("errors"):
                    logger.warning(
                        "Combined DSSP: %s", dssp_result["backfill"]["errors"]
                    )
            except Exception as _exc:
                logger.warning(f"Combined DSSP analysis failed: {_exc}")

            # ── Phylogenetic trees (sequence / structure) — only on request ──
            try:
                from agentic.planner.planning_guidelines import (
                    detect_phylo_tree_requested,
                )

                phylo_req = detect_phylo_tree_requested(
                    state.get("user_goal_original") or "",
                    state.get("combined_analysis_plan") or "",
                    state.get("master_enriched_prompt")
                    or state.get("enriched_prompt")
                    or "",
                )
                if phylo_req.get("sequence") or phylo_req.get("structure"):
                    from .tools import (
                        build_sequence_phylo_tree,
                        build_structure_phylo_tree,
                    )

                    if phylo_req.get("sequence"):
                        seq_tree = build_sequence_phylo_tree.func(
                            sim_dirs=sim_dirs,
                            labels=labels,
                            working_dir=analysis_dir,
                            base_dir=working_dir,
                            label_name_map=_name_map_a or None,
                            user_goal=_goal_text,
                        )
                        if seq_tree.get("success") and seq_tree.get("plot_path"):
                            if seq_tree["plot_path"] not in plots:
                                plots.append(seq_tree["plot_path"])
                            log_agent_action(
                                "analysis",
                                "Sequence phylogenetic tree generated",
                                {
                                    "output": seq_tree.get("plot_path"),
                                    "n_sequences": seq_tree.get("n_sequences"),
                                    "missing": seq_tree.get("missing", []),
                                },
                            )
                        else:
                            logger.warning(
                                "Sequence phylo tree: %s",
                                seq_tree.get("error") or seq_tree.get("message"),
                            )

                    if phylo_req.get("structure"):
                        struct_tree = build_structure_phylo_tree.func(
                            sim_dirs=sim_dirs,
                            labels=labels,
                            working_dir=analysis_dir,
                            base_dir=working_dir,
                            label_name_map=_name_map_a or None,
                            user_goal=_goal_text,
                        )
                        if struct_tree.get("success") and struct_tree.get("plot_path"):
                            if struct_tree["plot_path"] not in plots:
                                plots.append(struct_tree["plot_path"])
                            log_agent_action(
                                "analysis",
                                "Structure phylogenetic tree generated",
                                {
                                    "output": struct_tree.get("plot_path"),
                                    "n_structures": struct_tree.get("n_structures"),
                                    "missing": struct_tree.get("missing", []),
                                },
                            )
                        else:
                            logger.warning(
                                "Structure phylo tree: %s",
                                struct_tree.get("error") or struct_tree.get("message"),
                            )
            except Exception as _exc:
                logger.warning(f"Phylogenetic tree analysis failed: {_exc}")

            # ── Reference-projected landscape pipeline (consensus MSA → PCA → FEL) ──
            try:
                from agentic.planner.planning_guidelines import (
                    detect_reference_landscape_requested,
                    _resolve_reference_label,
                )

                ref_land_req = detect_reference_landscape_requested(
                    state.get("user_goal_original") or "",
                    state.get("combined_analysis_plan") or "",
                    state.get("master_enriched_prompt")
                    or state.get("enriched_prompt")
                    or "",
                )
                if ref_land_req.get("requested"):
                    from .tools import run_reference_landscape_pipeline

                    ref_label = _resolve_reference_label(
                        state.get("user_goal_original") or "",
                        state.get("user_goal") or "",
                        state.get("combined_analysis_plan") or "",
                        state.get("master_enriched_prompt")
                        or state.get("enriched_prompt")
                        or "",
                        labels=labels,
                        default=(labels[0] if labels else ""),
                    )
                    if labels and ref_label not in {str(l) for l in labels}:
                        logger.warning(
                            "Reference label %r not in simulation labels; using %s",
                            ref_label,
                            labels[0],
                        )
                        ref_label = labels[0]
                    ref_pipe = run_reference_landscape_pipeline.func(
                        sim_dirs=sim_dirs,
                        labels=labels,
                        working_dir=analysis_dir,
                        reference_label=ref_label,
                        base_dir=working_dir,
                        user_goal=_goal_text,
                        label_name_map=_name_map_a or None,
                    )
                    if ref_pipe.get("success"):
                        clust = ref_pipe.get("clustering") or {}
                        for key in (
                            "dendrogram_plot",
                            "phylo_tree_plot",
                        ):
                            pth = clust.get(key)
                            if pth and pth not in plots:
                                plots.append(pth)
                        log_agent_action(
                            "analysis",
                            "Reference landscape pipeline complete",
                            {
                                "reference_label": ref_label,
                                "n_projected": (ref_pipe.get("projections") or {}).get(
                                    "n_projected"
                                ),
                                "assignments": clust.get("assignments_file"),
                            },
                        )
                    else:
                        logger.warning(
                            "Reference landscape pipeline (%s): %s",
                            ref_pipe.get("stage"),
                            ref_pipe.get("error") or ref_pipe.get("message"),
                        )
            except Exception as _exc:
                logger.warning(f"Reference landscape pipeline failed: {_exc}")

            # ── Consensus-mapped reference pocket metrics ──
            try:
                from agentic.planner.planning_guidelines import (
                    detect_consensus_pocket_requested,
                    _resolve_reference_label,
                )

                cp_req = detect_consensus_pocket_requested(
                    state.get("user_goal_original") or "",
                    state.get("combined_analysis_plan") or "",
                    state.get("user_goal") or "",
                )
                # Consensus-mapped pocket COM/angle when modular / reference pocket asked.
                if class_groups and "reference_pocket" in class_groups:
                    cp_req = dict(cp_req)
                    cp_req["requested"] = True
                if cp_req.get("requested"):
                    from .tools import run_consensus_pocket_metrics_batch

                    cp_ref = _resolve_reference_label(
                        state.get("user_goal_original") or "",
                        state.get("user_goal") or "",
                        state.get("combined_analysis_plan") or "",
                        labels=labels,
                        default=cp_req.get("reference_label") or (labels[0] if labels else ""),
                    )
                    # Prefer reference_label from an existing pocket definition
                    # (pre_combined may have written p17612_ATP while goal parse
                    # returns bare UniProt p17612).
                    def_json = Path(analysis_dir) / "reference_pocket_definition.json"
                    if def_json.is_file():
                        try:
                            with open(def_json, encoding="utf-8") as _dfh:
                                _def = json.load(_dfh)
                            def_ref = str((_def or {}).get("reference_label") or "").strip()
                            if def_ref:
                                cp_ref = def_ref
                        except Exception:
                            pass

                    def _match_ref_label(candidate: str, lab: str, sd: str) -> bool:
                        c = str(candidate or "").lower().strip()
                        if not c:
                            return False
                        l = str(lab or "").lower().strip()
                        name = Path(sd).name.lower()
                        if c in {l, name}:
                            return True
                        for other in (l, name):
                            if other.startswith(c + "_") or c.startswith(other + "_"):
                                return True
                            for suf in ("_atp", "_adp", "_amp"):
                                if other.endswith(suf) and other[: -len(suf)] == c:
                                    return True
                        return False

                    if cp_ref not in {str(l) for l in labels}:
                        from src.reporter.combined_reporter import resolve_display_label

                        mapped = resolve_display_label(cp_ref, _name_map_a)
                        if mapped in {str(l) for l in labels}:
                            cp_ref = mapped
                    ref_sim_dir = next(
                        (
                            sd
                            for sd, lab in zip(sim_dirs, labels)
                            if _match_ref_label(cp_ref, lab, sd)
                        ),
                        "",
                    )
                    if not ref_sim_dir:
                        uid_ref = cp_req.get("reference_label") or cp_ref or "q8nb16"
                        ref_sim_dir = next(
                            (
                                sd
                                for sd, lab in zip(sim_dirs, labels)
                                if _match_ref_label(uid_ref, lab, sd)
                            ),
                            "",
                        )
                        if ref_sim_dir:
                            idx = sim_dirs.index(ref_sim_dir)
                            cp_ref = labels[idx]
                    batch_labels = (
                        list(dir_labels)
                        if len(dir_labels) == len(sim_dirs)
                        else list(labels)
                    )
                    cp_batch = run_consensus_pocket_metrics_batch.func(
                        sim_dirs=sim_dirs,
                        labels=batch_labels,
                        working_dir=analysis_dir,
                        reference_label=cp_ref,
                        reference_sim_dir=ref_sim_dir,
                        pocket_cutoff_A=cp_req.get("pocket_cutoff_A", 15.0),
                    )
                    if cp_batch.get("success"):
                        log_agent_action(
                            "analysis",
                            "Consensus pocket metrics batch complete",
                            {
                                "manifest": cp_batch.get("manifest_file"),
                                "failed_labels": cp_batch.get("failed_labels"),
                            },
                        )
                    else:
                        logger.warning(
                            "Consensus pocket batch (%s): %s",
                            cp_batch.get("stage"),
                            cp_batch.get("error") or cp_batch.get("message"),
                        )
            except Exception as _cp_exc:
                logger.warning("Consensus pocket batch failed: %s", _cp_exc)

            # ── Shared PKA-ref dyn (paper fig3 feature #10) for family campaigns ─
            try:
                from agentic.campaign.spec import spec_from_state as _spec_shared

                _fam_shared = _spec_shared(state)
                gold_cols = []
                if (
                    _fam_shared
                    and _fam_shared.family_modular
                    and _fam_shared.analysis_recipe.required_feature_columns
                ):
                    gold_cols = list(
                        _fam_shared.analysis_recipe.required_feature_columns
                    )
                if "pca_pka_ref_shared_dyn" in gold_cols:
                    from src.analysis.family_dynamics import (
                        compute_shared_pka_ref_dyn_features,
                    )

                    # Use on-disk directory labels (not display names like KAPCA_ATP).
                    fs_labels = list(dir_labels) if dir_labels else [
                        Path(d).name for d in sim_dirs
                    ]
                    ref_lab = ""
                    configured = ""
                    try:
                        configured = str(
                            (_fam_shared.settings or {}).get("reference_label") or ""
                        ).strip()
                    except Exception:
                        configured = ""
                    if configured:
                        for cand in fs_labels:
                            if str(cand).lower() == configured.lower() or str(
                                cand
                            ).lower().startswith(configured.lower()):
                                ref_lab = str(cand)
                                break
                        if not ref_lab:
                            ref_lab = configured
                    if not ref_lab:
                        ref_lab = str(fs_labels[0]) if fs_labels else ""
                    align = ""
                    for cand in (
                        Path(analysis_dir) / "global_consensus_msa.json",
                        Path(analysis_dir) / "global_mapped.json",
                        Path(analysis_dir) / "global_msa.json",
                        Path(analysis_dir) / "reference_msa_alignment.json",
                        Path(analysis_dir) / "global_msa_alignment.json",
                        Path(analysis_dir) / "consensus_alignment.json",
                        Path(analysis_dir) / "consensus_residues.json",
                    ):
                        if cand.is_file():
                            align = str(cand)
                            break
                    dyn_res = compute_shared_pka_ref_dyn_features.invoke(
                        {
                            "base_directory": working_dir,
                            "reference_label": ref_lab,
                            "alignment_json": align,
                            "labels": [str(l) for l in fs_labels],
                            "working_dir": analysis_dir,
                            "overwrite": True,
                        }
                    )
                    if dyn_res.get("success"):
                        log_agent_action(
                            "analysis",
                            "Shared PKA-ref dyn features complete",
                            {
                                "n_success": dyn_res.get("n_success"),
                                "n_requested": dyn_res.get("n_requested"),
                                "n_schema_features": dyn_res.get("n_schema_features"),
                                "csv": dyn_res.get("csv"),
                                "reference_label": dyn_res.get("reference_label"),
                            },
                        )
                    else:
                        logger.warning(
                            "Shared PKA-ref dyn failed: %s",
                            dyn_res.get("error") or dyn_res,
                        )
            except Exception as _dyn_exc:
                logger.warning("Shared PKA-ref dyn step failed: %s", _dyn_exc)

            # ── Classification feature matrix (only when user explicitly requests) ─
            classification_table = None
            classification_clustering = None
            if class_groups is not None:
                try:
                    from .tools import (
                        collect_classification_features_table,
                        cluster_classification_features,
                    )
                    from src.analysis.classification_clustering import (
                        clustering_method_for_goal,
                        plot_cluster_feature_trajectories,
                        plot_cluster_rmsf_profiles,
                        CLUSTER_TRAJECTORY_METRIC_GROUPS,
                        CLUSTER_REFERENCE_POCKET_TRAJECTORY_METRIC_GROUPS,
                        CLUSTER_RMSF_PROFILE_GROUPS,
                        REFERENCE_CLUSTERING_OUTPUT_FILES,
                        is_reference_structure_clustering,
                        reference_archetype_metric_groups,
                        resolve_n_clusters_for_goal,
                    )

                    ref_clustering = is_reference_structure_clustering(class_groups)
                    collect_groups = (
                        list(reference_archetype_metric_groups(class_groups))
                        if ref_clustering
                        else sorted(class_groups)
                    )
                    # Family campaigns use the compiled protocol columns
                    # (whatever the goal or gold_columns listed).
                    feature_columns = None
                    _fam_spec_collect = None
                    try:
                        from agentic.campaign.spec import spec_from_state as _spec_collect

                        _fam_spec_collect = _spec_collect(state)
                        if (
                            _fam_spec_collect
                            and _fam_spec_collect.family_modular
                            and _fam_spec_collect.analysis_recipe.required_feature_columns
                        ):
                            feature_columns = list(
                                _fam_spec_collect.analysis_recipe.required_feature_columns
                            )
                    except Exception:
                        _fam_spec_collect = None
                    ref_outputs = REFERENCE_CLUSTERING_OUTPUT_FILES
                    allowed_labels = [Path(d).name for d in sim_dirs]
                    if ref_clustering:
                        manifest_path = Path(analysis_dir) / "reference_pocket_batch_manifest.json"
                        if manifest_path.is_file():
                            try:
                                from src.analysis.reference_labels import (
                                    build_uniprot_display_map,
                                    is_reference_pocket_usable,
                                )

                                with open(manifest_path, encoding="utf-8") as _mfh:
                                    pocket_manifest = json.load(_mfh)
                                base_path = Path(analysis_dir)
                                filtered = [
                                    uid
                                    for uid in allowed_labels
                                    if is_reference_pocket_usable(
                                        uid,
                                        pocket_manifest,
                                        base_analysis=base_path,
                                    )
                                ]
                                if filtered:
                                    allowed_labels = sorted(filtered)
                            except Exception:
                                pass

                    collect_kwargs: Dict[str, Any] = {
                        "base_directory": working_dir,
                        "working_dir": analysis_dir,
                        "requested_metric_groups": collect_groups,
                        "allowed_labels": allowed_labels,
                    }
                    if feature_columns:
                        collect_kwargs["feature_columns"] = feature_columns
                    goal_blob = " ".join(
                        str(t or "")
                        for t in (
                            state.get("user_goal_original"),
                            state.get("user_goal"),
                        )
                    ).lower()
                    if "robust z" in goal_blob or "iqr" in goal_blob:
                        collect_kwargs["zscore_method"] = "robust"
                    if ref_clustering:
                        collect_kwargs.update(
                            output_file=ref_outputs["features_csv"],
                            zscore_output_file=ref_outputs["zscore_csv"],
                            manifest_file=ref_outputs["manifest_json"],
                            xlsx_output_file=ref_outputs["xlsx"],
                        )

                    class_result = collect_classification_features_table.func(
                        **collect_kwargs
                    )
                    if class_result.get("success"):
                        user_goal_text = " ".join(
                            t
                            for t in (
                                state.get("user_goal_original"),
                                state.get("user_goal"),
                                state.get("combined_analysis_plan"),
                                state.get("master_enriched_prompt"),
                                state.get("enriched_prompt"),
                            )
                            if t
                        ).strip()
                        # LLM selects a scientifically motivated feature subset
                        # (with written reasoning) before clustering.
                        try:
                            from agentic.campaign.spec import spec_from_state as _spec_from_state

                            _fam_spec = _spec_from_state(state)
                            _required_cols = (
                                list(_fam_spec.analysis_recipe.required_feature_columns)
                                if _fam_spec and _fam_spec.family_modular
                                else None
                            )
                            selection = self._llm_select_classification_features(
                                analysis_dir=analysis_dir,
                                class_result=class_result,
                                user_goal=user_goal_text,
                                metric_groups=collect_groups,
                                required_columns=_required_cols,
                                state=state,
                            )
                        except Exception as _sel_exc:
                            logger.warning(
                                "LLM feature selection failed (%s); using all collected columns",
                                _sel_exc,
                            )
                            selection = None
                        if selection and selection.get("feature_columns"):
                            selected_cols = [
                                c
                                for c in selection["feature_columns"]
                                if isinstance(c, str) and c.strip()
                            ]
                            if (
                                _fam_spec_collect
                                and _fam_spec_collect.family_modular
                                and _fam_spec_collect.analysis_recipe.required_feature_columns
                            ):
                                pinned = list(
                                    _fam_spec_collect.analysis_recipe.required_feature_columns
                                )
                                # The compiled protocol is the user's feature
                                # list (any length). Do not append extras.
                                selected_cols = list(pinned)
                            if selected_cols:
                                recollect_kwargs = dict(collect_kwargs)
                                recollect_kwargs["feature_columns"] = selected_cols
                                recollect_kwargs.pop("requested_metric_groups", None)
                                re_result = collect_classification_features_table.func(
                                    **recollect_kwargs
                                )
                                if re_result.get("success"):
                                    class_result = re_result
                                    log_agent_action(
                                        "analysis",
                                        "LLM classification feature selection applied",
                                        {
                                            "n_selected": len(selected_cols),
                                            "features": selected_cols,
                                            "reasoning": (selection.get("reasoning") or "")[
                                                :400
                                            ],
                                        },
                                    )
                        classification_table = class_result.get("output_file")
                        tables.append(classification_table)
                        zscore_path = class_result.get("zscore_output_file")
                        if zscore_path:
                            tables.append(zscore_path)
                        log_agent_action(
                            "analysis",
                            "Classification feature table built",
                            {
                                "output": classification_table,
                                "metric_groups": sorted(class_groups),
                                "n_simulations": class_result.get("n_simulations"),
                            },
                        )

                        cluster_method = clustering_method_for_goal(
                            user_goal_text,
                            state.get("combined_analysis_plan") or "",
                        )
                        n_sims = int(class_result.get("n_simulations") or len(allowed_labels))
                        n_clusters = resolve_n_clusters_for_goal(
                            n_sims,
                            user_goal_text,
                            reference_based=ref_clustering,
                        )
                        skip_family_cluster = False
                        cluster_kwargs: Dict[str, Any] = {
                            "working_dir": analysis_dir,
                            "features_file": (
                                Path(zscore_path).name
                                if zscore_path
                                else (
                                    ref_outputs["zscore_csv"]
                                    if ref_clustering
                                    else "classification_features_zscore.csv"
                                )
                            ),
                            "method": cluster_method,
                            "user_goal": user_goal_text,
                            "label_name_map": _name_map_a or None,
                            "n_clusters": n_clusters,
                            # Always emit dendrogram+heatmap when ≥2 usable columns exist.
                            "feature_scale_label": "Robust Z score",
                            "panel_file": "classification_dendrogram_heatmap.png",
                            "simple_panel": True,
                            "cluster_archetype_names": {},
                            "max_column_missing_fraction": 0.75,
                            "min_features_present": 2,
                        }
                        try:
                            from agentic.campaign.contracts import (
                                clustering_table_usable,
                                family_feature_matrix_ready,
                            )
                            from agentic.campaign.spec import spec_from_state

                            feat_csv = str(
                                Path(analysis_dir) / "classification_features.csv"
                            )
                            usable = clustering_table_usable(feat_csv)
                            if not usable.get("ok"):
                                msg = usable.get("reason") or "feature table not plottable"
                                logger.warning("Combined clustering skipped: %s", msg)
                                state.setdefault("warnings", []).append(
                                    f"Combined clustering skipped: {msg}"
                                )
                                skip_family_cluster = True
                                classification_clustering = {
                                    "success": False,
                                    "blocked": False,
                                    "reason": msg,
                                    **usable,
                                }
                            else:
                                spec = spec_from_state(state)
                                if spec and spec.family_modular:
                                    matrix = family_feature_matrix_ready(feat_csv, spec)
                                    if not matrix.get("ok"):
                                        msg = matrix.get("reason") or "sparse gold feature matrix"
                                        logger.warning(
                                            "Family gold matrix incomplete (%s); "
                                            "clustering on %s available columns",
                                            msg,
                                            usable.get("n_feature_cols"),
                                        )
                                        state.setdefault("warnings", []).append(
                                            f"Family feature set incomplete ({msg}); "
                                            "plotted dendrogram/heatmap from available columns"
                                        )
                        except Exception as gate_exc:
                            logger.debug("Clustering usability check skipped: %s", gate_exc)
                        if ref_clustering:
                            cluster_kwargs.update(
                                assignments_file=ref_outputs["assignments_csv"],
                                scatter_plot_file=ref_outputs["pca_png"],
                                dendrogram_file=ref_outputs["dendrogram_png"],
                                phylo_tree_file=ref_outputs["phylo_png"],
                                heatmap_file=ref_outputs["heatmap_png"],
                                panel_file=ref_outputs.get("panel_png"),
                                summary_file=ref_outputs["summary_json"],
                            )
                        if skip_family_cluster:
                            cluster_result = {"success": False}
                        else:
                            cluster_result = cluster_classification_features.func(
                                **cluster_kwargs
                            )
                        if cluster_result.get("success"):
                            classification_clustering = cluster_result
                            scatter = cluster_result.get("scatter_plot")
                            panel = cluster_result.get("panel_plot")
                            dendro = cluster_result.get("dendrogram_plot")
                            phylo = cluster_result.get("phylo_tree_plot")
                            heatmap = cluster_result.get("heatmap_plot")
                            if scatter:
                                plots.append(scatter)
                            if panel:
                                plots.append(panel)
                            elif dendro:
                                plots.append(dendro)
                            if heatmap:
                                plots.append(heatmap)
                            if phylo:
                                plots.append(phylo)
                            log_agent_action(
                                "analysis",
                                (
                                    "Reference-structure clustering complete"
                                    if ref_clustering
                                    else "Classification clustering complete"
                                ),
                                {
                                    "method": cluster_result.get("method"),
                                    "n_clusters": cluster_result.get("n_clusters"),
                                    "assignments": cluster_result.get("assignments_file"),
                                },
                            )
                            traj_groups = list(
                                CLUSTER_REFERENCE_POCKET_TRAJECTORY_METRIC_GROUPS
                                if ref_clustering or "reference_pocket" in class_groups
                                else CLUSTER_TRAJECTORY_METRIC_GROUPS
                            )
                            default_assignments = (
                                ref_outputs["assignments_csv"]
                                if ref_clustering
                                else "classification_cluster_assignments.csv"
                            )
                            if traj_groups:
                                traj_result = plot_cluster_feature_trajectories.func(
                                    working_dir=analysis_dir,
                                    assignments_file=(
                                        Path(cluster_result["assignments_file"]).name
                                        if cluster_result.get("assignments_file")
                                        else default_assignments
                                    ),
                                    metric_groups=traj_groups,
                                )
                                if traj_result.get("success"):
                                    for p in traj_result.get("plots", []):
                                        if p not in plots:
                                            plots.append(p)
                                    log_agent_action(
                                        "analysis",
                                        "Cluster trajectory plots generated",
                                        {
                                            "metrics": traj_result.get("metrics_plotted"),
                                            "n_clusters": traj_result.get("n_clusters"),
                                        },
                                    )
                                else:
                                    logger.warning(
                                        "Cluster trajectory plots: %s",
                                        traj_result.get("error"),
                                    )
                            rmsf_profile_types = (
                                ["reference_pocket_rmsf"]
                                if "reference_pocket" in class_groups
                                else sorted(
                                    g for g in class_groups if g in CLUSTER_RMSF_PROFILE_GROUPS
                                )
                            )
                            if rmsf_profile_types:
                                rmsf_cluster = plot_cluster_rmsf_profiles.func(
                                    working_dir=analysis_dir,
                                    assignments_file=(
                                        Path(cluster_result["assignments_file"]).name
                                        if cluster_result.get("assignments_file")
                                        else default_assignments
                                    ),
                                    profile_types=rmsf_profile_types,
                                )
                                if rmsf_cluster.get("success"):
                                    for p in rmsf_cluster.get("plots", []):
                                        if p not in plots:
                                            plots.append(p)
                                    log_agent_action(
                                        "analysis",
                                        "Cluster RMSF profile plots generated",
                                        {
                                            "profiles": rmsf_cluster.get("profiles_plotted"),
                                            "n_clusters": rmsf_cluster.get("n_clusters"),
                                        },
                                    )
                                else:
                                    logger.warning(
                                        "Cluster RMSF plots: %s",
                                        rmsf_cluster.get("error"),
                                    )
                        else:
                            logger.warning(
                                "Classification clustering: %s",
                                cluster_result.get("error"),
                            )
                    else:
                        logger.warning(
                            "Classification table: %s", class_result.get("error")
                        )
                        log_agent_action(
                            "analysis",
                            "Classification feature table failed",
                            {"error": class_result.get("error"), "metric_groups": sorted(class_groups)},
                        )
                except Exception as _exc:
                    logger.warning(f"Classification feature table failed: {_exc}")
                    log_agent_action(
                        "analysis",
                        "Classification feature table failed",
                        {"error": str(_exc), "metric_groups": sorted(class_groups)},
                    )

            # Store results in state for the reporter
            analysis_results = state.get("analysis_results") or {}
            analysis_results["combined"] = {
                "sim_dirs": sim_dirs,
                "labels": labels,
                "overlay_plots": plots,
                "stats_tables": tables,
                "skipped_metrics": skipped,
                "dccm_plots": dccm_plots,
                "rmsf_apo_holo_plots": rmsf_apo_holo_plots,
                "apo_holo_pairs": apo_holo_pairs,
                "rmsf_segment_plots": segment_plots,
                "com_distance_plot": com_plot,
                "dssp_plots": dssp_plots,
                "classification_features_table": classification_table,
                "classification_clustering": classification_clustering,
                "classification_metric_groups": sorted(class_groups) if class_groups else None,
                "analysis_dir": analysis_dir,
            }
            state["analysis_results"] = analysis_results
            state["figures"] = (
                list(state.get("figures") or []) + plots + dccm_plots
            )

            state["errors"] = [
                e for e in state.get("errors", [])
                if not (e.startswith("Analysis failed:") or e.startswith("Analysis error:"))
            ]

            self._mark_combined_analysis_complete(
                state,
                analysis_dir=analysis_dir,
                sim_dirs=sim_dirs,
                labels=labels,
                success=True,
            )
            log_agent_completion("analysis", "Combined Multi-Simulation Analysis", state, True)
            if hitl_should_interact(state):
                state["hitl_checkpoint_subtype"] = "post_combined"
                try:
                    from agentic.hitl_router import bind_combined_hitl_view

                    bind_combined_hitl_view(state, "analysis")
                except Exception:
                    pass
                state["next_node"] = "human_analysis_check"
            else:
                state["next_node"] = "reporter"

        except Exception as exc:
            import traceback
            logger.error(f"Combined analysis failed: {exc}\n{traceback.format_exc()}")
            state["errors"].append(f"Combined analysis error: {exc}")
            state["next_node"] = "human_analysis_check"
            state["error_triggered_hitl"] = True

        return state

    def _extract_agent_instructions(self, full_plan: str, agent_name: str) -> Optional[str]:
        """
        Extract agent-specific detailed instructions from planner's natural language plan.
        
        Args:
            full_plan: Complete natural language plan from planner
            agent_name: Name of the agent section to extract (e.g., "Analysis Agent")
            
        Returns:
            Extracted instructions for this specific agent, or full plan as fallback
        """
        import re
        
        # Try multiple patterns to find the agent section (in priority order)
        patterns = [
            # New standardized format: **Analysis Agent:**
            rf'\*\*Analysis Agent:\*\*\s*\n(.*?)(?=\n\s*\*\*Expected Outcomes:|$)',
            # With optional colon
            rf'\*\*{agent_name}\*\*:?\s*\n(.*?)(?=\n\s*\*\*[A-Z]|\Z)',
            # Markdown headings
            rf'###\s*{agent_name}.*?\n(.*?)(?=###|\Z)',
            rf'##\s*{agent_name}.*?\n(.*?)(?=##|\Z)',
            # Fuzzy match patterns (allow for variations like "MD Analysis Agent")
            rf'\*\*(?:MD\s*)?Analysis.*?Agent\*\*:?\s*\n(.*?)(?=\n\s*\*\*[A-Z]|\Z)',
            rf'###\s*(?:MD\s*)?Analysis.*?Agent.*?\n(.*?)(?=###|\Z)',
            rf'##\s*(?:MD\s*)?Analysis.*?Agent.*?\n(.*?)(?=##|\Z)',
            # Section number patterns
            rf'\d+\..*?(?:MD\s*)?Analysis.*?Agent.*?\n(.*?)(?=\d+\.|\Z)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_plan, re.DOTALL | re.IGNORECASE)
            if match:
                instructions = match.group(1).strip()
                if len(instructions) > 50:  # Ensure we got substantial content
                    logger.info(f"Extracted {len(instructions)} chars of detailed instructions for {agent_name}")
                    return instructions
        
        # Fallback: Use full plan if no specific section found
        logger.info(f"Could not find specific section for {agent_name}, using full plan as context")
        logger.info(f"Full plan length: {len(full_plan)} chars")
        
        # Return full plan so agent still has context
        return full_plan

    def _copy_files_from_hpc_secure(self, state: MDState):
        """Copy trajectory and topology files from HPC using SecureFileManager."""
        # Get all HPC files from registry
        hpc_files = []
        for file_path, metadata in self.file_manager.file_registry.items():
            if metadata.get("stage") == "hpc" and Path(file_path).exists():
                hpc_files.append((file_path, metadata))
        
        if not hpc_files:
            logger.info("No HPC files found in registry to copy")
            return
        
        # Copy each HPC file to analysis directory using file manager
        for file_path, metadata in hpc_files:
            filename = Path(file_path).name
            file_type = metadata.get("type", "unknown")
            description = metadata.get("description", "HPC output file")
            
            # Use secure copy (automatically registers in file_registry)
            new_path = self.file_manager.copy_file(
                source_path=file_path,
                dest_filename=filename,
                file_type=file_type,
                description=f"Copy from HPC: {description}"
            )
            
            if new_path:
                # Update state paths for known file types
                if file_type == "trajectory":
                    state["trajectory_path"] = new_path
                    logger.info(f"Updated trajectory_path to: {new_path}")
                elif file_type == "topology":
                    state["topology"] = new_path
                    logger.info(f"Updated topology to: {new_path}")
                
                logger.info(f"Copied {filename} ({file_type}) from HPC to analysis")
                log_file_operation("analysis", "copied", new_path, True, f"From HPC: {description}")

    def _format_pdb_info_for_llm(self, state: MDState) -> str:
        """Format PDB structural information from supervisor for LLM prompts.
        
        Args:
            state: Current MDState with pdb_analysis from supervisor
            
        Returns:
            Formatted string with PDB structural context for LLM
        """
        if not state:
            return ""
        
        pdb_analysis = state.get("pdb_analysis")
        if not pdb_analysis:
            return ""
        
        try:
            total_atoms = pdb_analysis.get("total_atoms", 0)
            total_residues = pdb_analysis.get("total_residues", 0)
            components = pdb_analysis.get("components_available", [])
            summary = pdb_analysis.get("human_readable_summary", "N/A")
            
            # Get protein chain information
            protein_info = pdb_analysis.get("protein", {})
            protein_chains_str = ""
            if protein_info:
                chains_list = []
                for chain_id, chain_data in protein_info.items():
                    if isinstance(chain_data, dict):
                        res_count = chain_data.get("residue_count", 0)
                        chains_list.append(f"Chain {chain_id} ({res_count} residues)")
                if chains_list:
                    protein_chains_str = ", ".join(chains_list)
            
            return f"""
**PDB Structure Information (from supervisor validation):**
- Total Atoms: {total_atoms}
- Total Residues: {total_residues}
- Components: {', '.join(components) if components else 'None'}
- Protein Chains: {protein_chains_str if protein_chains_str else 'None'}
- Summary: {summary}
"""
        except Exception as e:
            logger.warning(f"Failed to format PDB info for LLM: {e}")
            return ""

    def _write_wrap_inventory(
        self,
        state: MDState,
        analysis_dir: str,
        *,
        hpc_dir: str = "",
        topology: str = "",
        trajectory: str = "",
    ) -> None:
        try:
            from src.analysis.inventory import write_rep_inventory

            sim_root = state.get("working_directory") or ""
            slot = Path(hpc_dir or state.get("hpc_dir") or "")
            if slot.name.startswith("rep") and sim_root:
                adir = Path(sim_root) / "analysis" / slot.name
            elif sim_root:
                adir = Path(sim_root) / "analysis"
            else:
                adir = Path(analysis_dir)
            write_rep_inventory(
                analysis_dir=adir,
                hpc_dir=str(slot) if slot else "",
                sim_root=sim_root,
                label=Path(sim_root).name if sim_root else "",
                topology=topology or str(state.get("tpr_file") or ""),
                trajectory=trajectory or str(state.get("trajectory_path") or ""),
            )
        except Exception as inv_exc:
            logger.warning("inventory.json after wrap failed: %s", inv_exc)

    def _wrap_trajectory_pbc(self, state: MDState, analysis_dir: str) -> None:
        """
        Wrap the trajectory to fix periodic boundary condition (PBC) artefacts.

        Calls ``wrap_trajectory`` (gmx trjconv -pbc mol -center) and updates
        ``state["trajectory_path"]`` with the wrapped output file so all
        downstream tools use the corrected trajectory.

        Skipped when:
          - ``state["skip_pbc_wrap"]`` is True (caller opted out), or
          - no trajectory file is available, or
          - no TPR file is available (wrapping requires the run-input file), or
          - a valid wrapped trajectory (``mdWrap.xtc``) already exists and is
            not older than the source trajectory (idempotent re-runs).

        Re-wrapping is expensive (CPU/IO/memory), so repeated analysis
        iterations reuse the existing ``mdWrap.xtc`` instead of regenerating it.
        Set ``state["force_pbc_wrap"]=True`` to force a fresh wrap.

        The ligand name is taken from ``state["wrap_ligand"]``, else the first
        entry of ``state["ligand_resnames"]``, else ``"auto"`` (scan common
        ligand groups — never invent ATP for apo runs).
        The output dt (ps) can be overridden via ``state["wrap_dt_ps"]``
        (default **100 ps** ⇒ **10 frames / ns**). Production MDP default
        ``nstxout-compressed=50000`` at ``dt=0.002`` ps matches the same
        spacing, so all sims share one analysis timescale after wrap.
        Set ``state["force_pbc_wrap"]=True`` to rebuild an existing
        ``mdWrap.xtc``.
        """
        if state.get("skip_pbc_wrap"):
            logger.info("_wrap_trajectory_pbc: skip_pbc_wrap=True — skipping")
            return

        # Multi-rep: wrap each hpc/repXX independently (once). Do NOT recurse via
        # rep_num alone — effective_rep_num() still sees nested dirs on disk.
        if not state.get("_wrap_single_rep"):
            try:
                from src.analysis.replicate_paths import (
                    discover_hpc_rep_dirs,
                    resolve_production_trajectory,
                )

                rep_hpcs = discover_hpc_rep_dirs(
                    state.get("working_directory", "working_dir")
                )
                nested = [p for p in rep_hpcs if p.name.startswith("rep")]
                if nested and _effective_rep_num_for_state(state) > 1:
                    log_agent_action(
                        "analysis",
                        "Deterministic PBC wrap (pre-plan; all replicates)",
                        {
                            "note": (
                                "Not an LLM plan step. Analysis agent wraps each "
                                "hpc/repXX trajectory before metrics."
                            ),
                            "replicates": [p.name for p in nested],
                            "n_reps": len(nested),
                        },
                    )
                    first_wrapped = None
                    for hpc in nested:
                        traj_p = resolve_production_trajectory(hpc)
                        tpr_p = hpc / "md.tpr"
                        if not traj_p or not tpr_p.is_file():
                            logger.warning(
                                "_wrap_trajectory_pbc: skip %s (missing traj/tpr)", hpc
                            )
                            continue
                        snap = dict(state)
                        snap["trajectory_path"] = str(traj_p)
                        snap["hpc_dir"] = str(hpc)
                        snap["tpr_file"] = str(tpr_p)
                        snap["_wrap_single_rep"] = True
                        try:
                            self._wrap_trajectory_pbc(snap, analysis_dir)
                            log_agent_action(
                                "analysis",
                                f"PBC wrap complete for {hpc.name}",
                                {
                                    "hpc_dir": str(hpc),
                                    "trajectory": snap.get("trajectory_path"),
                                    "source": "deterministic_pre_plan",
                                },
                            )
                        except Exception as rep_exc:
                            logger.warning(
                                "_wrap_trajectory_pbc: %s failed: %s", hpc.name, rep_exc
                            )
                            continue
                        if snap.get("trajectory_path") and first_wrapped is None:
                            first_wrapped = snap["trajectory_path"]
                    if first_wrapped:
                        state["trajectory_path"] = first_wrapped
                    return
            except Exception as exc:
                logger.warning(
                    "_wrap_trajectory_pbc multi-rep dispatch failed: %s", exc
                )

        # Resolve trajectory
        traj = state.get("trajectory_path")
        if not traj or not Path(traj).exists():
            logger.info("_wrap_trajectory_pbc: no trajectory available — skipping")
            return

        # ── Reuse an already-wrapped trajectory (idempotent re-runs) ───────
        # Wrapping is CPU/IO/memory intensive; avoid repeating it when the
        # user re-iterates analysis. Set state["force_pbc_wrap"]=True to force.
        wrapped_name = "mdWrap.xtc"
        hpc_dir = state.get("hpc_dir") or str(
            Path(state.get("working_directory", "working_dir")) / "hpc"
        )
        force = bool(state.get("force_pbc_wrap"))

        if not state.get("_wrap_single_rep"):
            log_agent_action(
                "analysis",
                "Deterministic PBC wrap (pre-plan)",
                {
                    "note": (
                        "Not an LLM plan step. Analysis agent wraps the "
                        "production trajectory before metrics."
                    ),
                    "hpc_dir": hpc_dir,
                    "trajectory": traj,
                    "source": "engine",
                },
            )

        if not force:
            # Case 1: current trajectory already points to the wrapped file.
            if Path(traj).name == wrapped_name and Path(traj).stat().st_size > 0:
                logger.info(
                    "_wrap_trajectory_pbc: trajectory already wrapped "
                    f"({Path(traj).name}) — reusing, skipping re-wrap"
                )
                self._write_wrap_inventory(
                    state, analysis_dir, hpc_dir=hpc_dir, trajectory=str(traj)
                )
                return

            # Case 2: a wrapped file already exists next to the raw trajectory
            # or in hpc_dir, and is newer than the source — reuse it.
            existing = self._find_existing_wrapped_traj(
                hpc_dir, Path(traj), wrapped_name
            )
            if existing:
                state["trajectory_path"] = existing
                logger.info(
                    "_wrap_trajectory_pbc: found existing wrapped trajectory "
                    f"→ {existing} — reusing, skipping re-wrap"
                )
                self._write_wrap_inventory(
                    state, analysis_dir, hpc_dir=hpc_dir, trajectory=str(existing)
                )
                return

        # Resolve TPR (required for gmx trjconv -s)
        tpr = (
            state.get("tpr_file")
            or state.get("hpc_dir") and self._find_tpr(state.get("hpc_dir", ""))
            or self._find_tpr(state.get("working_directory", ""))
        )
        if not tpr or not Path(tpr).exists():
            logger.warning(
                "_wrap_trajectory_pbc: no TPR file found — cannot wrap trajectory. "
                "Set state['tpr_file'] or place md.tpr in the hpc/ directory."
            )
            return

        # Ligand: explicit wrap_ligand > ligand_resnames > auto-detect (not ATP-hardcoded)
        ligand = state.get("wrap_ligand")
        if not ligand:
            ligand_resnames = state.get("ligand_resnames") or []
            ligand = ligand_resnames[0] if ligand_resnames else "auto"

        # Output dt
        dt = int(state.get("wrap_dt_ps", 100))

        logger.info(
            f"_wrap_trajectory_pbc: wrapping {Path(traj).name} "
            f"(ligand={ligand}, dt={dt} ps, force={force}) …"
        )

        from src.analysis.trajectory_wrapper import _wrap_trajectory_impl

        # Save wrapped trajectory into hpc_dir (resolved above) so it lives
        # alongside the original simulation data and the reporter can find it.
        result = _wrap_trajectory_impl(
            tpr_file=str(tpr),
            trajectory_file=str(traj),
            output_file=wrapped_name,
            ligand=ligand,
            dt=dt,
            working_dir=hpc_dir,
            skip=False,
            force=force,
        )

        if result.get("success"):
            wrapped = result["wrapped_trajectory"]
            state["trajectory_path"] = wrapped
            if result.get("skipped"):
                logger.info(
                    f"_wrap_trajectory_pbc: reused existing wrap → {wrapped}"
                )
            else:
                logger.info(
                    f"_wrap_trajectory_pbc: trajectory updated → {wrapped} "
                    f"(center={result.get('centering_group')})"
                )
            self._write_wrap_inventory(
                state,
                analysis_dir,
                hpc_dir=hpc_dir,
                topology=str(tpr),
                trajectory=str(wrapped),
            )
        else:
            logger.warning(
                f"_wrap_trajectory_pbc: wrapping failed — continuing with "
                f"original trajectory. Error: {result.get('error', 'unknown')}"
            )

    def _find_existing_wrapped_traj(
        self, hpc_dir: str, source_traj: Path, wrapped_name: str
    ) -> Optional[str]:
        """Locate a previously wrapped trajectory that is safe to reuse.

        Returns the absolute path to an existing, non-empty wrapped trajectory
        that is at least as new as *source_traj* (so stale wraps are ignored).
        Searches hpc_dir, the source trajectory's own directory, and hpc_dir's
        sub-directories. Returns None when no valid wrapped file is found.
        """
        try:
            src_mtime = source_traj.stat().st_mtime if source_traj.exists() else 0.0
        except OSError:
            src_mtime = 0.0

        candidates: List[Path] = []
        if hpc_dir:
            candidates.append(Path(hpc_dir) / wrapped_name)
        candidates.append(source_traj.parent / wrapped_name)

        # Also scan hpc_dir sub-directories (e.g. nested output folders).
        if hpc_dir and Path(hpc_dir).is_dir():
            candidates.extend(sorted(Path(hpc_dir).rglob(wrapped_name)))

        seen: set = set()
        for cand in candidates:
            try:
                resolved = cand.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved == source_traj.resolve():
                continue
            if not resolved.is_file() or resolved.stat().st_size == 0:
                continue
            # Ignore wrapped files older than the source (source was re-run).
            if resolved.stat().st_mtime + 1 < src_mtime:
                logger.info(
                    f"_wrap_trajectory_pbc: ignoring stale wrapped file {resolved} "
                    "(older than source trajectory)"
                )
                continue
            return str(resolved)
        return None

    def _find_tpr(self, directory: str) -> Optional[str]:
        """Return the production TPR for wrapping / analysis.

        Prefer ``md.tpr``. ``Path.rglob('*.tpr')`` is filesystem-ordered and
        often hits ``nvt.tpr`` / ``minim.tpr`` first, which breaks
        ``gmx trjconv`` on ``md.xtc``.
        """
        from src.analysis.trajectory_wrapper import find_production_tpr

        return find_production_tpr(directory)

    def _write_pdb_info_to_summary(self, state: MDState, analysis_dir: str) -> None:
        """
        Write PDB validation info from supervisor to analysis summary file.
        This provides context about the input structure for LLM and users.
        
        Args:
            state: Workflow state containing pdb_analysis from supervisor
            analysis_dir: Analysis working directory
        """
        from src.analysis.summary_logger import append_analysis_summary
        
        # Get PDB analysis from state (populated by supervisor's input validation)
        pdb_analysis = state.get("pdb_analysis")
        
        if not pdb_analysis:
            logger.info("No PDB analysis info available from supervisor, skipping summary entry")
            return
        
        try:
            # Extract key information for summary
            total_atoms = pdb_analysis.get("total_atoms", 0)
            total_residues = pdb_analysis.get("total_residues", 0)
            components = pdb_analysis.get("components_available", {})
            summary = pdb_analysis.get("human_readable_summary", "N/A")
            
            # Get topology file path
            topology_file = state.get("topology") or state.get("cleaned_pdb") or state.get("raw_pdb")
            
            # Extract protein sequences if available
            protein_sequences = {}
            if pdb_analysis.get("protein", {}).get("present"):
                chains_info = pdb_analysis.get("protein", {}).get("chains", {})
                for chain_id, chain_data in chains_info.items():
                    sequence = chain_data.get("sequence", "")
                    if sequence:
                        protein_sequences[chain_id] = sequence
            
            # Write to summary file
            append_analysis_summary(
                working_dir=analysis_dir,
                analysis_type="PDB_Input_Validation",
                statistics={
                    "total_atoms": total_atoms,
                    "total_residues": total_residues
                },
                files={
                    "topology_file": topology_file or "N/A"
                },
                metadata={
                    "components_available": components,
                    "human_readable_summary": summary,
                    "protein_sequences": protein_sequences,
                    "source": "supervisor_validation"
                }
            )
            
            logger.info(f"PDB validation summary written: {summary}")
            log_agent_action(
                "analysis",
                "Recorded PDB Structure Info",
                {
                    "atoms": total_atoms,
                    "residues": total_residues,
                    "components": components,
                    "summary": summary
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to write PDB info to summary: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _prepare_agent_input(self, state: MDState) -> AnalysisAgentInput:
        """Prepare structured input for analysis from workflow state"""
        defaults = self.config
        
        # Check if planner provided detailed instructions for this agent
        # Prefer pre-extracted instructions from supervisor
        planner_instructions = state.get("analysis_instructions")
        
        if not planner_instructions:
            # Fallback: Extract from execution_plan if supervisor didn't provide it
            execution_plan = state.get("execution_plan", {})
            if execution_plan.get("format") == "natural_language":
                full_plan = execution_plan.get("full_plan", "")
                planner_instructions = self._extract_agent_instructions(full_plan, "Analysis Agent")
        
        # Append human recommendation so the LLM sees it
        human_rec = state.get("human_recommendation")
        if human_rec:
            rec_block = (
                f"\n\n**HUMAN RECOMMENDATION (must be followed):**\n{human_rec}\n"
                "Adjust the analysis plan to incorporate this recommendation."
            )
            if planner_instructions:
                planner_instructions += rec_block
            else:
                planner_instructions = rec_block
        
        # Resolve topology path: prefer the hpc_dir copy over the simsetup
        # original so the analysis agent always sees files from the same
        # directory tree as the trajectory/energy outputs.
        #
        # Prefer md.tpr (MDA-compatible with mdWrap.xtc). Never fall back to
        # topol.top (text topology) or ligand-only GRO files (ATP.gro / MG.gro),
        # which cause atom-count mismatches with the full-system trajectory.
        topology_file = state.get("topology")
        hpc_dir = state.get("hpc_dir") or state.get("hpc_output_directory") or state.get("hpc_directory")
        # Always prefer the active simulation's hpc/ tree over a stale cross-sim path.
        wd = state.get("working_directory")

        def _under_sim(path: Optional[str], sim_wd: Optional[str]) -> bool:
            if not path or not sim_wd:
                return False
            try:
                p = str(Path(path).resolve())
                root = str(Path(sim_wd).resolve())
            except OSError:
                return False
            return p == root or p.startswith(root + os.sep)

        if wd:
            local_hpc = Path(wd) / "hpc"
            if local_hpc.is_dir():
                if not hpc_dir or not _under_sim(hpc_dir, wd):
                    hpc_dir = str(local_hpc)
        # Prefer production MD topology (matches mdWrap.xtc). Do not let fresh
        # simsetup system.gro win under --reuse-hpc (atom-count mismatch).
        _MDA_TOPO_NAMES = (
            "md.tpr",
            "md.gro",
            "npt.gro",
            "nvt.gro",
            "system.gro",
            "solvated.gro",
            "complex.gro",
            "processed.gro",
        )
        _SKIP_TOPO_NAMES = {
            "topol.top",
            "topology.top",
            "ATP.gro",
            "MG.gro",
            "ligand_GMX.gro",
            "protein.gro",
            "protein_processed.gro",
            "protein.pdb",
        }
        if hpc_dir:
            hpc_dir_path = Path(hpc_dir)
            # Nested multi-rep: search hpc/repXX before flat hpc/
            search_roots: list[Path] = []
            try:
                from src.analysis.replicate_paths import discover_hpc_rep_dirs

                nested_reps = discover_hpc_rep_dirs(wd) if wd else []
            except Exception:
                nested_reps = []
            if nested_reps:
                search_roots.extend(nested_reps)
            search_roots.append(hpc_dir_path)
            search_roots.append(hpc_dir_path / "results")

            preferred: list[str] = list(_MDA_TOPO_NAMES)
            if topology_file:
                name = Path(topology_file).name
                # Ignore topology paths from another simulation directory
                # or known non-production topologies.
                if wd and not _under_sim(topology_file, wd):
                    topology_file = None
                elif name in _SKIP_TOPO_NAMES:
                    topology_file = None
                elif name in ("system.gro", "solvated.gro", "complex.gro"):
                    # Defer to md.tpr/md.gro first; keep as last-resort only.
                    pass
                elif name not in preferred:
                    preferred.insert(0, name)

            topology_file = None
            for root in search_roots:
                if not root.is_dir() and not any(
                    (root / n).is_file() for n in preferred[:2]
                ):
                    # root may be a file-less path; still try preferred names
                    pass
                for candidate_name in preferred:
                    candidate = root / candidate_name
                    if candidate.is_file() and candidate.stat().st_size > 0:
                        topology_file = str(candidate)
                        logger.info(
                            "analysis: resolved topology to production copy: %s",
                            topology_file,
                        )
                        break
                if topology_file:
                    break

        return AnalysisAgentInput(
            working_directory=state.get("working_directory", "working_dir"),
            hpc_output_dir=state.get("hpc_output_directory", ""),
            topology_file=topology_file,
            trajectory_file=state.get("trajectory_path"),
            energy_file=state.get("energy_file"),
            analyses=defaults.get("metrics", ["rmsd", "rmsf", "gyration"]),
            user_goal=state.get("user_goal", ""),
            enriched_goal=state.get("enriched_prompt") or state.get("rephrased_goal"),
            additional_instructions=planner_instructions
        )

    def _run_analysis_workflow(self, agent_input: AnalysisAgentInput, 
                               state: MDState) -> AnalysisAgentOutput:
        """
        Run complete analysis workflow:
        1. LLM creates intelligent plan based on available data
        2. Execute plan step-by-step using tools
        3. Return structured results
        """
        try:
            self._attach_retrieved_context(state, agent_input)

            # P0: refuse to plan/execute without real HPC trajectory + topology.
            ready, gate_reason = self._hpc_inputs_ready_for_analysis(state)
            if not ready:
                logger.error("Analysis gated: %s", gate_reason)
                log_agent_action(
                    "analysis",
                    "Skipped — HPC trajectory/topology not ready",
                    {"reason": gate_reason},
                )
                out_dir = state.get("analysis_directory") or state.get(
                    "analysis_dir", "./working_dir/analysis"
                )
                return AnalysisAgentOutput(
                    success=False,
                    plan=AnalysisPlan(
                        reasoning=gate_reason,
                        overview="Skipped: missing HPC inputs",
                        steps=[],
                    ),
                    result=AnalysisExecutionResult(
                        success=False,
                        report=gate_reason,
                        issues=[gate_reason],
                        warnings=[],
                        output_directory=str(out_dir),
                    ),
                )

            # Step 1: LLM analyzes available data and creates plan.
            # When human_recommendation is set, replan on top of the existing plan
            # so the recommendation is actually applied (not bypassed).
            exec_plan = state.get("execution_plan")
            human_rec = state.get("human_recommendation")
            if human_rec:
                logger.info("analysis: replanning with human guidance: %s", human_rec[:120])
                _updated = self.replan_with_guidance(human_rec, state)
                if _updated:
                    plan = AnalysisPlan(
                        reasoning=_updated.get("reasoning", ""),
                        overview=_updated.get("overview", ""),
                        steps=[
                            AnalysisStep(
                                name=s.get("name", ""),
                                description=s.get("description", ""),
                                tool_name=s.get("tool_name", ""),
                                tool_params=s.get("tool_params", {}),
                                reason=s.get("reason", "")
                            )
                            for s in _updated.get("steps", [])
                        ],
                        potential_issues=_updated.get("potential_issues", []),
                        recommendations=_updated.get("recommendations", []),
                    )
                else:
                    # LLM unavailable — fall back; human_rec is already in additional_instructions
                    plan = self._create_analysis_plan_llm(agent_input, state)
            else:
                prior = ((exec_plan or {}).get("structured_plans") or {}).get("analysis")
                force_replan = bool(state.get("force_replan") or state.get("analysis_force_replan"))
                # Plan→execute: reuse a compiled structured plan whenever present
                # (first pass or retry). LLM replan only when forced or missing.
                if (
                    not force_replan
                    and isinstance(prior, dict)
                    and prior.get("steps")
                ):
                    logger.info(
                        "analysis: reusing structured plan (%d steps), not re-planning",
                        len(prior.get("steps") or []),
                    )
                    state["_analysis_plan_source"] = "structured_reuse"
                    plan = AnalysisPlan(
                        reasoning=prior.get("reasoning") or "Compiled analysis plan",
                        overview=prior.get("overview") or "Reuse structured analysis plan",
                        steps=[
                            AnalysisStep(
                                name=s.get("name", ""),
                                description=s.get("description", ""),
                                tool_name=s.get("tool_name", ""),
                                tool_params=s.get("tool_params", {}),
                                reason=s.get("reason", ""),
                            )
                            for s in (prior.get("steps") or [])
                            if s.get("tool_name")
                        ],
                        potential_issues=prior.get("potential_issues") or [],
                        recommendations=prior.get("recommendations") or [],
                    )
                else:
                    plan = self._create_analysis_plan_llm(agent_input, state)

            plan_source = state.get("_analysis_plan_source") or "llm"
            log_agent_action("analysis", "Generated analysis plan", {
                "steps": len(plan.steps),
                "tools": [s.tool_name for s in plan.steps],
                "reasoning": plan.reasoning,
                "source": plan_source,
                "topology_file": agent_input.topology_file,
                "trajectory_file": agent_input.trajectory_file,
            })

            # Persist structured plan to state for HITL inspection/modification
            if exec_plan is not None:
                exec_plan.setdefault("structured_plans", {})["analysis"] = plan.model_dump()
            
            # Step 2: Execute plan using tool executor
            result = self._execute_analysis_plan(agent_input, plan, state)
            
            # Step 3: Register created files in file_registry
            file_registry = state.get("file_registry", {})
            
            for file_path, description in result.generated_files.items():
                # Determine file type from extension
                file_type = "analysis_output"
                if ".dat" in file_path or ".xvg" in file_path:
                    file_type = "analysis_data"
                elif ".png" in file_path or ".pdf" in file_path:
                    file_type = "plot"
                
                file_registry[file_path] = {
                    "type": file_type,
                    "description": description,
                    "stage": "analysis"
                }
            
            # Write back modified file_registry to state
            state["file_registry"] = file_registry
            
            # Step 4: Prepare supervisor update
            supervisor_update = {
                "analysis_results": result.results,
                "analysis_directory": result.output_directory,
                "analysis_report": result.report,
                "file_registry": file_registry
            }
            
            return AnalysisAgentOutput(
                success=result.success,
                plan=plan,
                result=result,
                supervisor_update=supervisor_update
            )
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Analysis workflow failed: {e}")
            logger.error(f"Traceback: {tb}")
            return AnalysisAgentOutput(
                success=False,
                plan=AnalysisPlan(
                    reasoning=f"Error in planning: {str(e)}",
                    overview="Failed",
                    steps=[]
                ),
                result=AnalysisExecutionResult(
                    success=False,
                    report=f"Error: {str(e)}\n\nTraceback:\n{tb}",
                    issues=[str(e)],
                    warnings=[],
                    output_directory=state.get("analysis_directory", "./working_dir/analysis")
                ),
                supervisor_update={}
            )

    def _update_state(self, state: MDState, agent_output: AnalysisAgentOutput):
        """Update workflow state with analysis results"""
        if agent_output.supervisor_update:
            state.update(agent_output.supervisor_update)
        
        state["analysis_report"] = agent_output.result.report
        state["analysis_issues"] = agent_output.result.issues
        state["analysis_warnings"] = agent_output.result.warnings
        state["analysis_execution_log"] = agent_output.result.execution_log
        
        # Initialize file registry if not present
        if "file_registry" not in state or state["file_registry"] is None:
            state["file_registry"] = {}
        
        # Register all generated files with metadata
        for file_path, description in agent_output.result.generated_files.items():
            # Determine file type from filename
            file_type = "analysis_output"
            if ".dat" in file_path or ".xvg" in file_path:
                file_type = "analysis_data"
            elif ".png" in file_path or ".pdf" in file_path:
                file_type = "plot"
            
            state["file_registry"][file_path] = {
                "type": file_type,
                "description": description,
                "stage": "analysis"
            }

    def _llm_plan_json_with_retry(
        self,
        prompt: str,
        log_label: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 16384,
    ) -> Dict[str, Any]:
        """Call the LLM for a JSON plan; retry once when the response has no steps.

        Reasoning models occasionally return empty content (reasoning-only) or a
        JSON object without a ``steps`` array. A single strict retry recovers most
        of these before the caller falls back to the deterministic template.
        """
        content = self.llm.prompt_raw(
            prompt, temperature=temperature, max_tokens=max_tokens, format="json"
        )
        log_llm_interaction(
            log_label, prompt, content,
            is_mock=hasattr(self.llm, "_is_mock_mode") and self.llm._is_mock_mode,
        )
        plan_dict = self._extract_plan_json(content)

        if not (content or "").strip() or not plan_dict.get("steps"):
            logger.warning(
                "%s: empty/invalid JSON plan (content=%d chars, steps=%d); retrying once",
                log_label, len((content or "").strip()),
                len(plan_dict.get("steps", []) or []),
            )
            retry_prompt = (
                prompt
                + "\n\nIMPORTANT: Your previous reply was empty or missing the "
                "\"steps\" array. Reply with ONLY a single valid JSON object of the "
                "form {\"reasoning\":..., \"overview\":..., \"steps\":[{...}], "
                "\"potential_issues\":[], \"recommendations\":[]}. Every requested "
                "metric must appear as a step. No prose, no markdown fences."
            )
            content = self.llm.prompt_raw(
                retry_prompt, temperature=0.0, max_tokens=max(max_tokens, 24576), format="json"
            )
            log_llm_interaction(
                f"{log_label}.retry", retry_prompt, content,
                is_mock=hasattr(self.llm, "_is_mock_mode") and self.llm._is_mock_mode,
            )
            plan_dict = self._extract_plan_json(content)

        plan_dict["_raw_content"] = content
        return plan_dict

    def _create_recipe_analysis_plan(
        self,
        agent_input: AnalysisAgentInput,
        state: MDState,
    ) -> AnalysisPlan:
        """Execute the shared analysis protocol — no per-protein LLM tool shopping."""
        from agentic.campaign.family_recipe import recipe_steps_as_plan_dicts
        from agentic.campaign.spec import spec_from_state

        spec = spec_from_state(state)
        recipe = spec.analysis_recipe if spec else None

        topo_name = (
            Path(agent_input.topology_file).name if agent_input.topology_file else "md.tpr"
        )
        traj_name = (
            Path(agent_input.trajectory_file).name
            if agent_input.trajectory_file
            else "mdWrap.xtc"
        )
        plan_dict: Dict[str, Any] = {
            "reasoning": "Compiled shared analysis protocol (identical for every protein).",
            "overview": "Family-modular per-simulation analysis",
            "steps": recipe_steps_as_plan_dicts(
                topology_file=topo_name,
                trajectory_file=traj_name,
                recipe=recipe,
            ),
            "potential_issues": [],
            "recommendations": [],
        }
        plan_dict = self._ensure_family_modular_dynamics_steps(plan_dict, agent_input, state)
        state["_analysis_plan_source"] = "engine"
        return AnalysisPlan(
            reasoning=plan_dict.get("reasoning", ""),
            overview=plan_dict.get("overview", "Family-modular analysis"),
            steps=[
                AnalysisStep(
                    name=step.get("name", "unknown"),
                    description=step.get("description", ""),
                    tool_name=step.get("tool_name", ""),
                    tool_params=step.get("tool_params", {}),
                    reason=step.get("reason", ""),
                )
                for step in plan_dict.get("steps", [])
            ],
            potential_issues=plan_dict.get("potential_issues", []),
            recommendations=plan_dict.get("recommendations", []),
        )

    def _create_analysis_plan_llm(self, agent_input: AnalysisAgentInput, state: MDState) -> AnalysisPlan:
        """
        Use LLM to analyze available data and create intelligent analysis plan.
        Falls back to template-based plan if LLM fails.
        """
        from agentic.campaign.spec import spec_from_state
        from agentic.planner.planning_guidelines import (
            detect_family_modular_dynamics_requested,
        )

        spec = spec_from_state(state)
        family = bool(spec and spec.family_modular)
        if not family:
            family = detect_family_modular_dynamics_requested(
                *collect_goal_texts_for_intent(state, agent_input)
            )
        if family and not state.get("hitl_chat_task"):
            logger.info("Analysis planning: using shared analysis protocol (no LLM tool list)")
            return self._create_recipe_analysis_plan(agent_input, state)
        state["_analysis_plan_source"] = "llm"

        # Build LLM prompt from config template
        prompt = self._build_analysis_planning_prompt(agent_input, state)
        
        try:
            plan_dict = self._llm_plan_json_with_retry(
                prompt, "analysis.planning", temperature=0.2, max_tokens=16384
            )
            content = plan_dict.pop("_raw_content", "")
            plan_dict = self._normalize_plan_steps(plan_dict, agent_input)

            llm_step_count = len(plan_dict.get("steps", []))
            llm_reasoning = plan_dict.get("reasoning", "")

            plan_dict = self._filter_plan_steps_by_intent(plan_dict, state, agent_input)

            # Post-process: inject mandatory steps the LLM may have omitted
            plan_dict = self._inject_mandatory_steps(plan_dict, agent_input, state)
            plan_dict = self._ensure_requested_metric_steps(plan_dict, agent_input, state)
            plan_dict = self._filter_plan_steps_by_intent(plan_dict, state, agent_input)
            # Re-inject plots after intent filter — filter can drop plot_multipanel
            # (combined_metrics.png) and plot_md_data steps even when .dat metrics remain.
            plan_dict = self._inject_missing_metric_plots(plan_dict)
            plan_dict = self._ensure_combined_metrics_multipanel(plan_dict)
            # Family modular tools MUST be last: intent filter can drop LLM-omitted
            # tools that _ensure_requested_metric_steps just added; ensure must win.
            plan_dict = self._ensure_family_modular_dynamics_steps(plan_dict, agent_input, state)

            final_step_count = len(plan_dict.get("steps", []))
            if final_step_count > llm_step_count:
                tools = [s.get("tool_name", "") for s in plan_dict.get("steps", [])]
                plan_dict["reasoning"] = (
                    f"Expanded LLM draft ({llm_step_count} steps) to {final_step_count} steps "
                    f"covering all metrics in the enriched user goal. "
                    f"Tools: {', '.join(tools)}."
                )
            elif not plan_dict.get("reasoning"):
                plan_dict["reasoning"] = llm_reasoning or content[:500]

            return AnalysisPlan(
                reasoning=plan_dict.get("reasoning", content[:500]),
                overview=plan_dict.get("overview", "Analyzing MD trajectory"),
                steps=[
                    AnalysisStep(
                        name=step.get("name", "unknown"),
                        description=step.get("description", ""),
                        tool_name=step.get("tool_name", ""),
                        tool_params=step.get("tool_params", {}),
                        reason=step.get("reason", "")
                    )
                    for step in plan_dict.get("steps", [])
                ],
                potential_issues=plan_dict.get("potential_issues", []),
                recommendations=plan_dict.get("recommendations", [])
            )
            
        except Exception as e:
            logger.warning(f"LLM planning failed, using fallback: {e}")
            return self._create_fallback_analysis_plan(agent_input, state)

    def _intent_metrics(
        self,
        state: MDState,
        agent_input: Optional[AnalysisAgentInput] = None,
    ) -> Optional[FrozenSet[str]]:
        """Requested metrics for the current simulation (per-sim goal takes precedence)."""
        from agentic.planner.planning_guidelines import (
            detect_family_modular_dynamics_requested,
        )

        base = detect_requested_metrics_for_sim(state, agent_input)
        enriched = self._resolve_enriched_goal_for_planning(state, agent_input)
        if not enriched:
            merged = set(base or ())
        else:
            extra = detect_requested_metrics(enriched)
            if base is None and extra is None:
                merged = set()
            else:
                merged = set(base or ()) | set(extra or ())

        # Master family-modular goal must reach every sim even when the per-sim
        # prompt only names a subset of descriptors.
        goal_texts = collect_goal_texts_for_intent(state, agent_input)
        if detect_family_modular_dynamics_requested(*goal_texts):
            if "dihedral_pca" in merged or "shared_dihedral_pca" in merged:
                merged.discard("pca")

        if merged:
            return frozenset(merged)
        if base is None and not enriched:
            return None
        if base is None and enriched and detect_requested_metrics(enriched) is None:
            return None
        return frozenset(merged) if merged else None

    def _sims_for_combined_metric(
        self,
        state: MDState,
        sim_dirs: List[str],
        labels: List[str],
        metric: str,
        label_name_map: Optional[Dict[str, str]] = None,
    ) -> tuple[List[str], List[str]]:
        """Simulations that should participate in one combined metric (may be a subset)."""
        return resolve_sims_for_combined_metric(
            metric,
            sim_dirs,
            labels,
            master_goal=state.get("user_goal_original") or "",
            combined_plan=state.get("combined_analysis_plan") or "",
            completed_sim_states=state.get("completed_sim_states") or [],
            label_name_map=label_name_map,
        )

    def _filter_plan_steps_by_intent(
        self,
        plan_dict: Dict[str, Any],
        state: MDState,
        agent_input: Optional[AnalysisAgentInput] = None,
    ) -> Dict[str, Any]:
        """Remove analysis steps outside the explicit metric scope in user goals."""
        if state.get("hitl_chat_task"):
            return plan_dict

        requested = self._intent_metrics(state, agent_input)
        if requested is None:
            return plan_dict

        allowed_data_files, allowed_plot_files = allowed_output_files_for_metrics(requested)

        calc_outputs: Set[str] = set()
        for step in plan_dict.get("steps", []):
            tool = step.get("tool_name", "")
            metric = _CALC_TOOL_TO_METRIC.get(tool)
            if metric is not None and metric not in requested:
                continue
            out = str((step.get("tool_params") or {}).get("output_file", "") or "")
            if out:
                calc_outputs.add(Path(out).name)

        filtered: List[Dict[str, Any]] = []
        for step in plan_dict.get("steps", []):
            tool = step.get("tool_name", "")
            if tool in _PREP_TOOLS:
                continue
            metric = _CALC_TOOL_TO_METRIC.get(tool)
            if metric is not None:
                if metric in requested:
                    filtered.append(step)
                continue
            if tool in {"plot_md_data", "plot_multipanel", "plot_md_multipanel"}:
                params = step.get("tool_params") or {}
                data_files = params.get("data_files") or []
                out_file = str(params.get("output_file", ""))
                out_name = Path(out_file).name if out_file else ""
                data_ok = any(
                    Path(str(df)).name in allowed_data_files
                    or Path(str(df)).name in calc_outputs
                    or is_metric_output_filename(str(df), requested)
                    for df in data_files
                )
                # combined_metrics.png is a QC overview spanning allowed metrics —
                # keep it whenever its input .dat files are in-scope.
                is_combined = out_name == "combined_metrics.png" or tool in {
                    "plot_multipanel",
                    "plot_md_multipanel",
                }
                plot_ok = (
                    not out_name
                    or out_name in allowed_plot_files
                    or is_metric_output_filename(out_name, requested)
                    or Path(out_name).stem in {Path(str(df)).stem for df in data_files}
                    or is_combined
                )
                if data_ok and plot_ok:
                    filtered.append(step)
                continue
            filtered.append(step)

        if filtered:
            plan_dict["steps"] = filtered
        plan_dict = self._dedupe_com_distance_steps(plan_dict, state, agent_input)
        return plan_dict

    def _dedupe_com_distance_steps(
        self,
        plan_dict: Dict[str, Any],
        state: MDState,
        agent_input: Optional[AnalysisAgentInput] = None,
    ) -> Dict[str, Any]:
        """Keep one COM-distance tool unless the user explicitly requested both."""
        goal = " ".join(collect_goal_texts_for_intent(state, agent_input))
        mode = detect_com_distance_mode(goal)
        if mode == "both":
            return plan_dict
        drop = "calculate_com_distance" if mode == "pocket" else "calculate_ligand_pocket_distance"
        steps = [
            s for s in plan_dict.get("steps", [])
            if s.get("tool_name") != drop
        ]
        if len(steps) != len(plan_dict.get("steps", [])):
            logger.info(
                "_dedupe_com_distance_steps: removed %s (COM mode=%s)",
                drop, mode,
            )
        plan_dict["steps"] = steps
        return plan_dict

    def _detect_ligand_resname(self, agent_input: AnalysisAgentInput, state: MDState) -> str:
        """Determine the ligand residue name to use for pocket-distance analysis.

        Priority:
        1. state["ligand_resnames"] (set by supervisor input validation)
        2. pdb_analysis["ligand"]["residue_names"] (from supervisor structure inspection)
        3. Known ligand names found in user_goal text
        4. Regex pattern "resname XYZ" in user_goal
        5. Default "LIG"
        """
        # 1. Explicit state field set by supervisor
        ligand_resnames = state.get("ligand_resnames") or []
        if ligand_resnames:
            return ligand_resnames[0]

        # 2. pdb_analysis from input validation
        pdb_analysis = state.get("pdb_analysis") or {}
        ligand_info = pdb_analysis.get("ligand", {})
        if isinstance(ligand_info, dict):
            residue_names = ligand_info.get("residue_names", [])
            if residue_names:
                return residue_names[0]

        # 3 & 4. Parse user goal text
        user_goal = (
            state.get("user_goal_original") or
            state.get("user_goal") or
            agent_input.user_goal or
            ""
        )
        known_ligands = ["ATP", "ADP", "AMP", "GTP", "GDP", "NAD", "FAD", "FMN", "LIG", "INH"]
        for lig in known_ligands:
            if lig in user_goal.upper():
                return lig
        match = re.search(r'\bresname\s+([A-Z0-9]{1,5})\b', user_goal, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return "LIG"

    def _resolve_sim_label(self, state: MDState) -> str:
        """Current simulation folder / UniProt label for family tools."""
        sim_prompts = state.get("sim_prompts") or []
        idx = state.get("current_sim_index")
        if idx is not None and 0 <= idx < len(sim_prompts):
            lab = str(sim_prompts[idx].get("label") or "").strip()
            if lab:
                return lab
        for key in ("active_sim_label", "current_sim_label", "sim_label", "label"):
            lab = str(state.get(key) or "").strip()
            if lab:
                return lab
        wd = state.get("working_directory") or ""
        return Path(wd).name if wd else ""

    def _resolve_family_alignment_paths(self, state: MDState) -> Dict[str, str]:
        """Locate pre-combined MSA / pocket map for per-sim consensus tools."""
        from agentic.multi_sim_paths import resolve_multi_sim_base_dir

        out: Dict[str, str] = {}
        try:
            base = Path(resolve_multi_sim_base_dir(state))
        except Exception:
            base = Path(state.get("multi_sim_base_dir") or state.get("working_directory") or ".")
            # If working_directory is a sim folder, climb to campaign root.
            if (base / "analysis").is_dir() and (base.parent / "analysis").is_dir():
                # sim-level: prefer parent campaign
                if any(
                    (base.parent / "analysis" / name).is_file()
                    for name in (
                        "global_consensus_msa.json",
                        "global_mapped.json",
                        "global_msa.json",
                        "reference_msa_alignment.json",
                        "reference_consensus.json",
                        "consensus_alignment.fasta",
                    )
                ) or (base.parent / "cross_sim").is_dir():
                    base = base.parent

        candidates = [
            base / "cross_sim",
            base / "analysis",
            base,
        ]
        msa_names = (
            "global_consensus_msa.json",
            "global_mapped.json",
            "reference_msa_alignment.json",
            "consensus_residues.json",
            "reference_consensus.json",
            "consensus_alignment.json",
        )
        pocket_names = (
            "pocket_mapped.csv",
            "pocket_mapped.json",
            "reference_pocket_residue_map.csv",
            "pocket_residue_map.csv",
            "reference_msa_residue_map.csv",
            "global_consensus_msa.csv",
            "global_mapped.csv",
            "reference_pocket_resid_map.csv",
            "reference_residue_map.csv",
            "pocket_resid_map_filtered.csv",
        )
        for root in candidates:
            if not root.is_dir():
                continue
            for msa_name in msa_names:
                msa = root / msa_name
                if msa.is_file() and "alignment_json" not in out:
                    out["alignment_json"] = str(msa.resolve())
                    break
            for pocket_name in pocket_names:
                p = root / pocket_name
                if p.is_file() and "pocket_map_csv" not in out:
                    out["pocket_map_csv"] = str(p.resolve())
                    break
        return out

    def _inject_consensus_family_params(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        state: MDState,
    ) -> Dict[str, Any]:
        """Fill label / alignment_json / pocket map for modular family tools."""
        family_tools = {
            "calculate_consensus_torsions",
            "calculate_consensus_rmsf_features",
            "calculate_consensus_dccm_features",
            "run_independent_dynamics_fel",
        }
        if tool_name not in family_tools:
            return tool_params
        params = dict(tool_params or {})
        label = self._resolve_sim_label(state)
        if label:
            # Always set — LLM often omits label; empty string must not win.
            params["label"] = label
        paths = self._resolve_family_alignment_paths(state)
        # Prefer discovered absolute paths over missing / relative hallucinations.
        existing_align = str(params.get("alignment_json") or "").strip()
        if paths.get("alignment_json") and (
            not existing_align
            or existing_align in (
                "alignment_json",
                "consensus_alignment.json",
                "reference_msa_alignment.json",
                "consensus_residues.json",
            )
            or not Path(existing_align).is_file()
        ):
            params["alignment_json"] = paths["alignment_json"]
        if (
            tool_name == "calculate_consensus_torsions"
            and paths.get("pocket_map_csv")
        ):
            existing_pocket = str(params.get("pocket_map_csv") or "").strip()
            if not existing_pocket or not Path(existing_pocket).is_file():
                params["pocket_map_csv"] = paths["pocket_map_csv"]
        sim_dir = state.get("working_directory") or ""
        if sim_dir and not params.get("sim_directory"):
            params["sim_directory"] = str(sim_dir)
        if paths.get("alignment_json") or label:
            logger.info(
                "Injected family-tool params for %s: label=%s alignment_json=%s",
                tool_name,
                params.get("label"),
                params.get("alignment_json"),
            )
        return params

    def _is_holo_simulation(self, state: MDState, agent_input: AnalysisAgentInput) -> bool:
        """True when this simulation's own folder is ligand-bound.

        The campaign goal often mentions both apo and holo, and the study
        directory may be named ``pseudo_apo_holo``. Neither of those makes a
        protein-only trajectory ATP-bound.
        """
        from src.analysis.combined_analysis import label_is_holo

        label = ""
        sim_prompts = state.get("sim_prompts") or []
        idx = state.get("current_sim_index")
        if idx is not None and 0 <= idx < len(sim_prompts):
            label = str(sim_prompts[idx].get("label") or "")
        if not label:
            for key in ("active_sim_label", "current_sim_label", "sim_label", "label"):
                label = str(state.get(key) or "")
                if label:
                    break
        if not label:
            label = Path(state.get("working_directory", "")).name
        if label_is_holo(label):
            return True
        # A worker whose state lost the label still has the sim folder path.
        wd_name = Path(state.get("working_directory") or "").name
        if wd_name and wd_name != label and label_is_holo(wd_name):
            return True
        return False

    @staticmethod
    def _step_requires_bound_ligand(step: Any) -> bool:
        """Ligand metrics that cannot succeed on a protein-only trajectory."""
        tool = str(getattr(step, "tool_name", "") or "")
        if tool in {
            "calculate_ligand_pocket_distance",
            "calculate_consensus_pocket_metrics",
            "calculate_ligand_axis_angle",
            "calculate_protein_ligand_contacts",
            "calculate_pocket_sasa",
            "analyze_ligand_residence",
            "calculate_pocket_rmsf",
            "calculate_ligand_rmsf",
            "calculate_ligand_rmsd",
        }:
            return True
        if tool != "calculate_com_distance":
            return False
        params = getattr(step, "tool_params", None) or {}
        sel = str(params.get("selection2") or params.get("ligand_selection") or "")
        return (not sel) or "resname" in sel.lower()

    def _retarget_dssp_plot_inputs(self, tool_params: Dict[str, Any], analysis_dir: str):
        """Point a DSSP plot at the files ``analyze_secondary_structure`` writes.

        The planner often asks ``plot_md_data`` to read ``dssp.dat``. DSSP writes
        ``dssp_percentages.dat`` and ``dssp_heatmap.png`` instead. Returns
        ``"skip"`` when that figure is already on disk.
        """
        params = dict(tool_params or {})
        files = list(params.get("data_files") or [])
        if not files:
            return params
        root = Path(analysis_dir)
        rewritten = []
        for raw in files:
            name = Path(str(raw)).name
            stem = Path(name).stem.lower()
            if not stem.startswith("dssp"):
                rewritten.append(name)
                continue
            if (root / name).is_file():
                rewritten.append(name)
                continue
            if (root / f"{stem}_heatmap.png").is_file() or (root / f"{stem}_timeseries.png").is_file():
                return "skip"
            substitute = None
            for cand in (f"{stem}_percentages.dat", f"{stem}_raw_data.dat"):
                if (root / cand).is_file() and (root / cand).stat().st_size > 0:
                    substitute = cand
                    break
            if substitute:
                rewritten.append(substitute)
                continue
            if any(root.glob(f"{stem}*.png")):
                return "skip"
            rewritten.append(name)
        params["data_files"] = rewritten
        return params

    @staticmethod
    def _analysis_step_to_dict(step: AnalysisStep) -> Dict[str, Any]:
        return {
            "name": step.name,
            "description": step.description,
            "tool_name": step.tool_name,
            "tool_params": step.tool_params,
            "reason": step.reason,
        }

    def _ensure_requested_metric_steps(
        self,
        plan_dict: Dict[str, Any],
        agent_input: AnalysisAgentInput,
        state: MDState,
    ) -> Dict[str, Any]:
        """Merge fallback steps for requested metrics missing from the LLM plan."""
        if state.get("hitl_chat_task"):
            return plan_dict

        requested = self._intent_metrics(state, agent_input)
        if not requested:
            return plan_dict

        existing_tools = {s.get("tool_name", "") for s in plan_dict.get("steps", [])}
        existing_sigs = {
            (
                s.get("tool_name", ""),
                str(
                    (s.get("tool_params") or {}).get("output_prefix")
                    or (s.get("tool_params") or {}).get("output_file")
                    or ""
                ),
            )
            for s in plan_dict.get("steps", [])
        }
        missing_metrics: Set[str] = set()
        binding_rmsf = (requested or frozenset()) & {"pocket_rmsf", "ligand_rmsf"}
        for metric in requested:
            if metric == "rmsf" and binding_rmsf:
                covered = all(
                    any(
                        _CALC_TOOL_TO_METRIC.get(tool) == binding_metric
                        for tool in existing_tools
                    )
                    for binding_metric in binding_rmsf
                )
                if covered:
                    continue
            if not any(
                _CALC_TOOL_TO_METRIC.get(tool) == metric for tool in existing_tools
            ):
                missing_metrics.add(metric)

        if "fel" in missing_metrics:
            missing_metrics.add("pca")

        if not missing_metrics:
            return plan_dict

        fallback = self._create_fallback_analysis_plan(agent_input, state)
        merged = list(plan_dict.get("steps", []))
        merged_outputs = {
            str(s.get("tool_params", {}).get("output_file", ""))
            for s in merged
        }

        for fb_step in fallback.steps:
            tool = fb_step.tool_name
            metric = _CALC_TOOL_TO_METRIC.get(tool)
            sig = (
                tool,
                str(
                    fb_step.tool_params.get("output_prefix")
                    or fb_step.tool_params.get("output_file")
                    or ""
                ),
            )
            if metric and metric in missing_metrics and sig not in existing_sigs:
                merged.append(self._analysis_step_to_dict(fb_step))
                existing_tools.add(tool)
                existing_sigs.add(sig)
                out = str(fb_step.tool_params.get("output_file", ""))
                if out:
                    merged_outputs.add(out)
                continue

            if tool in {"plot_md_data", "plot_pca_projection", "plot_multipanel"}:
                out_file = str(fb_step.tool_params.get("output_file", ""))
                if out_file and out_file in merged_outputs:
                    continue
                data_files = fb_step.tool_params.get("data_files") or []
                for m in missing_metrics:
                    spec = STANDARD_OUTPUT_FILES.get(m, {})
                    plot_ok = out_file and out_file == spec.get("plot")
                    data_ok = any(df == spec.get("data") for df in data_files)
                    if plot_ok or data_ok:
                        merged.append(self._analysis_step_to_dict(fb_step))
                        if out_file:
                            merged_outputs.add(out_file)
                        break

        if len(merged) > len(plan_dict.get("steps", [])):
            logger.info(
                "_ensure_requested_metric_steps: added steps for missing metrics: %s",
                sorted(missing_metrics),
            )
            plan_dict["steps"] = merged
        return plan_dict

    def _ensure_family_modular_dynamics_steps(
        self,
        plan_dict: Dict[str, Any],
        agent_input: AnalysisAgentInput,
        state: MDState,
    ) -> Dict[str, Any]:
        """Force family modular tools with canonical output dirs.

        LLMs often invent wrong ``output_dir`` (e.g. ``torsions/``, ``dccm_features/``)
        or custom extractors. Collector expects ``consensus_dihedrals/``,
        ``consensus_rmsf/``, ``consensus_DCCM/``, ``consensus_PCA/``.
        """
        from agentic.planner.planning_guidelines import (
            detect_family_modular_dynamics_requested,
        )

        texts = collect_goal_texts_for_intent(state, agent_input)
        family_on = bool(detect_family_modular_dynamics_requested(*texts))
        try:
            from agentic.campaign.spec import spec_from_state as _spec_fam

            _spec = _spec_fam(state)
            if _spec and (_spec.family_modular or _spec.mode == "family_modular"):
                family_on = True
        except Exception:
            pass
        if not family_on:
            return plan_dict

        allowed: Set[str] = set()
        try:
            if _spec and _spec.pin_tools:
                allowed = {str(t) for t in _spec.pin_tools}
            elif _spec and _spec.analysis_recipe.steps:
                allowed = {s.tool_name for s in _spec.analysis_recipe.steps}
        except Exception:
            allowed = set()
        if not allowed:
            try:
                from agentic.campaign.family_recipe import build_protocol_from_goal

                pack = build_protocol_from_goal("\n".join(str(t or "") for t in texts))
                allowed = {str(t) for t in (pack.get("pin_tools") or [])}
            except Exception:
                allowed = set()
        if not allowed:
            return plan_dict

        # Per-sim plans only. Combined / pre-combined stages already expose a
        # different tool bucket (SHARED/COMBINED); local calculate_rmsf etc. are
        # not on that menu, so dropping them there is unnecessary.
        if self._is_combined_hitl_context(state) or self._is_pre_combined_phase(state):
            return plan_dict

        topo_name = (
            Path(agent_input.topology_file).name
            if agent_input.topology_file
            else "md.tpr"
        )
        traj_name = (
            Path(agent_input.trajectory_file).name
            if agent_input.trajectory_file
            else "mdWrap.xtc"
        )

        canonical: Dict[str, Dict[str, Any]] = {
            "calculate_consensus_torsions": {
                "name": "Consensus dihedrals (φ/ψ/χ₁)",
                "description": "Mapped φ/ψ/χ₁ circular means/std for domain and pocket",
                "tool_params": {
                    "topology_file": topo_name,
                    "trajectory_file": traj_name,
                    "output_dir": "consensus_dihedrals",
                },
                "reason": "Family modular: pocket χ₁ mean/std + shared PCA inputs",
            },
            "calculate_consensus_rmsf_features": {
                "name": "Consensus Cα RMSF",
                "description": "Mapped consensus Cα RMSF mean/std across the domain",
                "tool_params": {
                    "topology_file": topo_name,
                    "trajectory_file": traj_name,
                    "output_dir": "consensus_rmsf",
                },
                "reason": "Family modular: consensus RMSF features",
            },
            "calculate_consensus_dccm_features": {
                "name": "Consensus DCCM N↔C",
                "description": "Mapped DCCM mean absolute and N-lobe↔C-lobe correlation",
                "tool_params": {
                    "topology_file": topo_name,
                    "trajectory_file": traj_name,
                    "output_dir": "consensus_DCCM",
                },
                "reason": "Family modular: DCCM N↔C feature",
            },
        }
        from agentic.campaign.ligand_context import ligand_selection_from_state

        _lig_sel = ligand_selection_from_state(state, default=None)
        if _lig_sel:
            canonical["calculate_consensus_pocket_metrics"] = {
                "name": "Ligand–pocket COM + axis-angle",
                "description": "Ligand COM distance and orientation vs pocket_mapped residues",
                "tool_params": {
                    "topology_file": topo_name,
                    "trajectory_file": traj_name,
                    "ligand_selection": _lig_sel,
                    "output_prefix": "reference_pocket",
                },
                "reason": "Family modular: pocket–ligand COM + axis-angle",
            }

        drop_tools = {
            "extract_chi1_stats",
            "perform_ward_clustering",
            "assemble_feature_table",
            "generate_html_report",
        }
        if "calculate_consensus_rmsf_features" in allowed:
            drop_tools.update({"calculate_rmsf", "calculate_pocket_rmsf"})
        if "calculate_consensus_dccm_features" in allowed:
            drop_tools.update({"calculate_dccm", "extract_dccm_corr"})
        if "calculate_consensus_pocket_metrics" in allowed:
            drop_tools.add("calculate_ligand_pocket_distance")
        steps: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for step in plan_dict.get("steps", []) or []:
            tool = str(step.get("tool_name") or "")
            if tool in drop_tools:
                continue
            if tool in canonical:
                spec = canonical[tool]
                step = dict(step)
                step["name"] = spec["name"]
                step["description"] = spec["description"]
                step["reason"] = spec["reason"]
                params = dict(step.get("tool_params") or {})
                params.update(spec["tool_params"])
                step["tool_params"] = params
                if tool not in seen:
                    steps.append(step)
                    seen.add(tool)
                continue
            steps.append(step)

        for tool, spec in canonical.items():
            if tool not in allowed or tool in seen:
                continue
            steps.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "tool_name": tool,
                    "tool_params": dict(spec["tool_params"]),
                    "reason": spec["reason"],
                }
            )
            seen.add(tool)

        order = {
            "calculate_consensus_pocket_metrics": 0,
            "calculate_consensus_torsions": 1,
            "calculate_consensus_rmsf_features": 2,
            "calculate_consensus_dccm_features": 3,
        }
        # Keep non-modular steps in original relative order; append modular block
        # with torsions → RMSF → DCCM → dihedral PCA dependency order.
        non_mod = [s for s in steps if s.get("tool_name") not in order]
        modular = sorted(
            [s for s in steps if s.get("tool_name") in order],
            key=lambda s: order[str(s.get("tool_name"))],
        )
        plan_dict["steps"] = non_mod + modular
        logger.info(
            "_ensure_family_modular_dynamics_steps: canonical modular tools=%s",
            [s.get("tool_name") for s in modular],
        )
        return plan_dict

    def _inject_mandatory_steps(self, plan_dict: Dict[str, Any],
                                 agent_input: AnalysisAgentInput,
                                 state: MDState) -> Dict[str, Any]:
        """Post-process the LLM plan to inject mandatory steps that the LLM may have omitted."""
        if state.get("hitl_chat_task"):
            return plan_dict

        steps = plan_dict.get("steps", [])
        requested = self._intent_metrics(state, agent_input)
        narrow_scope = requested is not None

        # --- Ligand pocket distance / protein–ligand COM ---
        existing_tools = {s.get("tool_name", "") for s in steps}
        user_goal_lower = (
            state.get("user_goal")
            or state.get("user_goal_original")
            or agent_input.user_goal
            or ""
        ).lower()
        com_mode = detect_com_distance_mode(user_goal_lower)
        narrow_com = narrow_scope and "com" not in (requested or frozenset())

        if not existing_tools & {
            "calculate_ligand_pocket_distance",
            "calculate_consensus_pocket_metrics",
        }:
            ligand_keywords = [
                "ligand", "atp", "adp", "amp", "gtp", "gdp", "nad",
                "inhibitor", "pocket", "binding site", "catalytic pocket",
                "catalytic site", "active site",
            ]
            needs_pocket_distance = com_mode in ("pocket", "both") and any(
                kw in user_goal_lower for kw in ligand_keywords
            )

            if narrow_com:
                needs_pocket_distance = False

            # Also trigger if pdb_analysis shows a ligand is present (broad scope only)
            if not needs_pocket_distance and not narrow_scope and com_mode != "protein_com":
                pdb_analysis = state.get("pdb_analysis") or {}
                ligand_info = pdb_analysis.get("ligand", {})
                if isinstance(ligand_info, dict) and ligand_info.get("present"):
                    needs_pocket_distance = True

            if needs_pocket_distance and not self._is_holo_simulation(state, agent_input):
                logger.info(
                    "_inject_mandatory_steps: skipping ligand pocket distance — "
                    "protein-only (apo) simulation has no bound ligand"
                )
                needs_pocket_distance = False

            if needs_pocket_distance:
                ligand_resname = self._detect_ligand_resname(agent_input, state)
                topo_file = (
                    Path(agent_input.topology_file).name
                    if agent_input.topology_file else "md.gro"
                )
                traj_file = (
                    Path(agent_input.trajectory_file).name
                    if agent_input.trajectory_file else "md.xtc"
                )
                pocket_step = {
                    "name": "Calculate Ligand Pocket COM Distance",
                    "description": (
                        f"Identify protein pocket atoms within 5 Å of {ligand_resname} "
                        "at frame 0, then track COM-to-COM distance over the trajectory."
                    ),
                    "tool_name": "calculate_ligand_pocket_distance",
                    "tool_params": {
                        "topology_file": topo_file,
                        "trajectory_file": traj_file,
                        "ligand_selection": f"resname {ligand_resname.upper()}",
                        "cutoff": 5.0,
                        "output_file": "ligand_pocket_distance.csv",
                    },
                    "reason": "User requested COM distance between the ligand and the catalytic pocket.",
                }
                pocket_plot_step = {
                    "name": "Plot Ligand Pocket COM Distance",
                    "description": "Plot the ligand-to-pocket COM distance over simulation time.",
                    "tool_name": "plot_md_data",
                    "tool_params": {
                        "data_files": ["ligand_pocket_distance.csv"],
                        "output_file": "ligand_pocket_distance.png",
                        "x_col": 1,
                        "y_col": 2,
                        "xlabel": "Time (ns)",
                        "ylabel": "COM Distance (Å)",
                        "titles": f"Ligand ({ligand_resname.upper()}) — Catalytic Pocket COM Distance",
                    },
                    "reason": "Mandatory plot after calculate_ligand_pocket_distance.",
                }
                steps.extend([pocket_step, pocket_plot_step])
                plan_dict["steps"] = steps
                logger.info(
                    "_inject_mandatory_steps: injected calculate_ligand_pocket_distance "
                    f"(ligand_selection='resname {ligand_resname.upper()}') — "
                    f"COM mode={com_mode}"
                )

        # --- Whole-protein COM distance (explicit selection1/selection2) ---
        existing_tools = {s.get("tool_name", "") for s in steps}
        if (
            "calculate_com_distance" not in existing_tools
            and com_mode in ("protein_com", "both")
            and not narrow_com
        ):
            ligand_resname = self._detect_ligand_resname(agent_input, state)
            topo_file = (
                Path(agent_input.topology_file).name
                if agent_input.topology_file else "md.gro"
            )
            traj_file = (
                Path(agent_input.trajectory_file).name
                if agent_input.trajectory_file else "md.xtc"
            )
            com_step = {
                "name": "Calculate Protein–Ligand COM Distance",
                "description": (
                    f"Per-frame COM distance between the whole protein and "
                    f"{ligand_resname.upper()}."
                ),
                "tool_name": "calculate_com_distance",
                "tool_params": {
                    "topology_file": topo_file,
                    "trajectory_file": traj_file,
                    "selection1": "protein",
                    "selection2": f"resname {ligand_resname.upper()}",
                    "label1": "Protein",
                    "label2": ligand_resname.upper(),
                    "output_file": "com_distance.csv",
                },
                "reason": "User requested whole-protein COM distance to the ligand.",
            }
            com_plot_step = {
                "name": "Plot Protein–Ligand COM Distance",
                "description": "Plot protein-to-ligand COM distance over time.",
                "tool_name": "plot_md_data",
                "tool_params": {
                    "data_files": ["com_distance.csv"],
                    "output_file": "com_distance.png",
                    "x_col": 1,
                    "y_col": 2,
                    "xlabel": "Time (ns)",
                    "ylabel": "COM Distance (Å)",
                    "titles": f"Protein — {ligand_resname.upper()} COM Distance",
                },
                "reason": "Mandatory plot after calculate_com_distance.",
            }
            if self._is_holo_simulation(state, agent_input):
                steps.extend([com_step, com_plot_step])
                plan_dict["steps"] = steps
                logger.info(
                    "_inject_mandatory_steps: injected calculate_com_distance — "
                    f"COM mode={com_mode}"
                )

        # --- DCCM --- only when the per-simulation goal explicitly requests it
        if "calculate_dccm" not in existing_tools:
            needs_dccm = requested is not None and "dccm" in requested

            if needs_dccm:
                topo_file = (
                    Path(agent_input.topology_file).name
                    if agent_input.topology_file else "md.gro"
                )
                traj_file = (
                    Path(agent_input.trajectory_file).name
                    if agent_input.trajectory_file else "md.xtc"
                )
                dccm_step = {
                    "name": "Calculate Dynamic Cross-Correlation Matrix (DCCM)",
                    "description": (
                        "Compute normalised DCCM of Cα fluctuations to reveal "
                        "correlated and anti-correlated residue motions. "
                        "Generates dccm.csv and dccm_heatmap.png internally."
                    ),
                    "tool_name": "calculate_dccm",
                    "tool_params": {
                        "topology_file": topo_file,
                        "trajectory_file": traj_file,
                        "selection": "protein and name CA",
                        "output_prefix": "dccm",
                        "frame_interval": 5,
                        "save_matrix_csv": True,
                        "create_heatmap": True,
                    },
                    "reason": (
                        "User goal mentions correlated motions / DCCM / pseudokinase "
                        "dynamics — DCCM reveals allosteric communication patterns."
                    ),
                }
                steps.append(dccm_step)
                plan_dict["steps"] = steps
                logger.info(
                    "_inject_mandatory_steps: injected calculate_dccm — "
                    "triggered by user_goal keywords"
                )

        # --- Combined metrics panel plot ---
        # Always inject plot_multipanel → combined_metrics.png unless the
        # LLM already included a multipanel step.
        _current_tools = {s.get("tool_name", "") for s in steps}
        _MULTIPANEL_TOOLS = {"plot_multipanel", "plot_md_multipanel"}
        if not (_current_tools & _MULTIPANEL_TOOLS):
            # Collect .dat output files from standard metric steps already in the plan
            _DAT_LABEL: Dict[str, tuple] = {
                "rmsd":     ("Time (ns)", "RMSD (Å)", "RMSD"),
                "rmsf":     ("Residue",   "RMSF (Å)", "RMSF"),
                "gyration": ("Time (ns)", "Rg (Å)",   "Radius of Gyration"),
                "rg":       ("Time (ns)", "Rg (Å)",   "Radius of Gyration"),
                "energy":   ("Time (ns)", "Energy (kJ/mol)", "Energy"),
            }
            panel_files: List[str] = []
            xlabels:     List[str] = []
            ylabels:     List[str] = []
            titles:      List[str] = []
            for s in steps:
                out = str(s.get("tool_params", {}).get("output_file", ""))
                if out.endswith(".dat"):
                    fname = Path(out).name
                    for key, (xl, yl, tl) in _DAT_LABEL.items():
                        if key in fname.lower():
                            if fname not in panel_files:
                                panel_files.append(fname)
                                xlabels.append(xl)
                                ylabels.append(yl)
                                titles.append(tl)
                            break
            # Inject multipanel only when ≥2 standard panels and scope is not single-metric
            if narrow_scope and len(requested) < 2:
                panel_files = []
            elif len(panel_files) >= 2:
                multipanel_step = {
                    "name": "Create Combined Metrics Plot",
                    "description": (
                        "Multi-panel summary figure of all standard MD metrics "
                        "(RMSD, RMSF, Rg and optionally energy) in one PNG."
                    ),
                    "tool_name": "plot_multipanel",
                    "tool_params": {
                        "data_files": panel_files,
                        "output_file": "combined_metrics.png",
                        "layout": "vertical",
                        "titles": titles,
                        "xlabels": xlabels,
                        "ylabels": ylabels,
                    },
                    "reason": (
                        "Mandatory combined overview plot — always generated for every "
                        "individual simulation so each simulation folder contains the same "
                        "quality-control figure."
                    ),
                }
                steps.append(multipanel_step)
                plan_dict["steps"] = steps
                logger.info(
                    "_inject_mandatory_steps: injected plot_multipanel → combined_metrics.png "
                    "(%d panels: %s)", len(panel_files), panel_files
                )

        plan_dict["steps"] = steps
        plan_dict = self._inject_missing_metric_plots(plan_dict)
        return plan_dict

    def _inject_missing_metric_plots(self, plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add plot_md_data for calculate_* outputs that have no matching plot step."""
        steps = list(plan_dict.get("steps") or [])
        plotted: Set[str] = set()
        for step in steps:
            if step.get("tool_name") not in {"plot_md_data", "plot_multipanel", "plot_md_multipanel"}:
                continue
            for data_file in (step.get("tool_params") or {}).get("data_files") or []:
                plotted.add(Path(str(data_file)).name)

        skip_tools = _PREP_TOOLS | {
            "identify_nearby_residues",
            "calculate_hbond_occupancy",
            "calculate_salt_bridge_distances",
            "calculate_dccm",
            "analyze_secondary_structure",
            "calculate_trajectory_pca",
            "calculate_free_energy_landscape",
            "analyze_fel_landscape_features",
            "export_fel_basin_structures",
        }
        ylabel_by_stem = (
            ("rmsd", "RMSD (Å)"),
            ("rmsf", "RMSF (Å)"),
            ("gyration", "Rg (Å)"),
            ("min_distance", "Min distance (Å)"),
            ("com_distance", "COM Distance (Å)"),
            ("ligand_pocket_distance", "COM Distance (Å)"),
            ("sasa", "SASA (nm²)"),
            ("energy", "Energy (kJ/mol)"),
        )

        injected = 0
        for step in list(steps):
            tool = step.get("tool_name", "")
            if tool in skip_tools or tool.startswith("plot_"):
                continue
            if not tool.startswith("calculate_") and not tool.startswith("analyze_"):
                continue
            out = str((step.get("tool_params") or {}).get("output_file", "") or "")
            name = Path(out).name
            if not name or name in plotted:
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in {".dat", ".csv"}:
                continue
            stem_l = Path(name).stem.lower()
            # Never auto-plot FEL grids / feature tables as XY line plots.
            if (
                stem_l.startswith("fel_")
                or "fel" in stem_l and "grid" in stem_l
                or stem_l in {"fel_features", "fel_basins", "fel_basin_structures"}
            ):
                continue
            stem = Path(name).stem
            ylabel = "Value"
            xlabel = "Time (ns)"
            for key, label in ylabel_by_stem:
                if stem == key or stem.startswith(key + "_"):
                    ylabel = label
                    break
            if "rmsf" in stem.lower():
                xlabel = "Residue"
            plot_name = f"{stem}.png"
            steps.append({
                "name": f"Plot {stem}",
                "description": f"Plot {name} as {plot_name}.",
                "tool_name": "plot_md_data",
                "tool_params": {
                    "data_files": [name],
                    "output_file": plot_name,
                    "xlabel": xlabel,
                    "ylabel": ylabel,
                    "titles": stem,
                },
                "reason": "Mandatory plot for each calculate_* data file.",
            })
            plotted.add(name)
            injected += 1

        if injected:
            logger.info("_inject_missing_metric_plots: added %d plot step(s)", injected)
            plan_dict["steps"] = steps
        return plan_dict

    def _ensure_combined_metrics_multipanel(self, plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure plot_multipanel → combined_metrics.png when ≥2 standard .dat metrics exist."""
        steps = list(plan_dict.get("steps") or [])
        if any(
            s.get("tool_name") in {"plot_multipanel", "plot_md_multipanel"}
            for s in steps
        ):
            return plan_dict

        _DAT_LABEL: Dict[str, tuple] = {
            "rmsd": ("Time (ns)", "RMSD (Å)", "RMSD"),
            "rmsf": ("Residue", "RMSF (Å)", "RMSF"),
            "gyration": ("Time (ns)", "Rg (Å)", "Radius of Gyration"),
            "rg": ("Time (ns)", "Rg (Å)", "Radius of Gyration"),
            "energy": ("Time (ns)", "Energy (kJ/mol)", "Energy"),
        }
        panel_files: List[str] = []
        xlabels: List[str] = []
        ylabels: List[str] = []
        titles: List[str] = []
        for s in steps:
            out = str((s.get("tool_params") or {}).get("output_file", ""))
            if not out.endswith(".dat"):
                continue
            fname = Path(out).name
            for key, (xl, yl, tl) in _DAT_LABEL.items():
                if key in fname.lower() and fname not in panel_files:
                    panel_files.append(fname)
                    xlabels.append(xl)
                    ylabels.append(yl)
                    titles.append(tl)
                    break
        if len(panel_files) < 2:
            return plan_dict

        steps.append({
            "name": "Create Combined Metrics Plot",
            "description": "Multi-panel summary figure of standard MD metrics.",
            "tool_name": "plot_multipanel",
            "tool_params": {
                "data_files": panel_files,
                "output_file": "combined_metrics.png",
                "layout": "vertical",
                "titles": titles,
                "xlabels": xlabels,
                "ylabels": ylabels,
            },
            "reason": "Mandatory combined QC plot for each simulation.",
        })
        plan_dict["steps"] = steps
        logger.info(
            "_ensure_combined_metrics_multipanel: injected plot_multipanel (%s)",
            panel_files,
        )
        return plan_dict

    def _plot_step_output_present(self, step: Any, analysis_dir: str) -> bool:
        """True when a plot/non-critical step's expected PNG already exists."""
        params = getattr(step, "tool_params", None) or {}
        if not isinstance(params, dict):
            return False
        out = params.get("output_file") or params.get("plot_file") or params.get("figure_path")
        if not out:
            return False
        name = Path(str(out)).name
        if not name.lower().endswith((".png", ".pdf", ".svg", ".jpg", ".jpeg")):
            return False
        return (Path(analysis_dir) / name).is_file()

    def _metric_step_output_present(self, step: Any, analysis_dir: str) -> bool:
        """True when this step's data or plot file is already written.

        A retry then fills the missing plots and DSSP instead of recomputing
        RMSD, RMSF, and DCCM on every pass.
        """
        if self._plot_step_output_present(step, analysis_dir):
            return True
        params = getattr(step, "tool_params", None) or {}
        if not isinstance(params, dict):
            return False
        root = Path(analysis_dir)
        if not root.is_dir():
            return False
        out = params.get("output_file")
        if out:
            path = root / Path(str(out)).name
            if path.is_file() and path.stat().st_size > 0:
                return True
        tool = str(getattr(step, "tool_name", "") or "")
        prefix = str(params.get("output_prefix") or "")
        if tool == "analyze_secondary_structure" and prefix:
            return any(
                p.is_file()
                and p.stat().st_size > 0
                and p.suffix.lower() == ".png"
                and p.name.startswith(prefix)
                for p in root.iterdir()
            )
        if tool == "calculate_dccm":
            prefix = prefix or "dccm"
            return (root / f"{prefix}.csv").is_file() and (
                root / f"{prefix}_heatmap.png"
            ).is_file()
        return False

    def _ensure_plots_for_dat_on_disk(
        self,
        analysis_dir: str,
        execution_log: List[str],
        generated_files: Dict[str, str],
        warnings: List[str],
    ) -> None:
        """Create missing PNGs from existing .dat/.csv metric files (post-batch safety net)."""
        adir = Path(analysis_dir)
        if not adir.is_dir():
            return
        try:
            from src.analysis.data_plotter import plot_md_data, plot_multipanel
        except Exception as exc:
            warnings.append(f"Could not import plotters for post-hoc plots: {exc}")
            return

        ylabel_by_stem = {
            "rmsd": ("Time (ns)", "RMSD (Å)"),
            "rmsf": ("Residue", "RMSF (Å)"),
            "gyration": ("Time (ns)", "Rg (Å)"),
            "rg": ("Time (ns)", "Rg (Å)"),
            "energy": ("Time (ns)", "Energy (kJ/mol)"),
        }
        panel_files: List[str] = []
        xlabels: List[str] = []
        ylabels: List[str] = []
        titles: List[str] = []
        wd = str(adir)

        def _is_stats_or_feature_table(path: Path) -> bool:
            """Skip summary tables — they are not time-series panels."""
            name = path.name.lower()
            stem = path.stem.lower()
            if stem.endswith("_stats") or name.endswith("_stats.csv"):
                return True
            if name.startswith("classification_") or "features" in stem:
                return True
            if name in {"statistical_summary.json", "statistical_summary.csv"}:
                return True
            return False

        # Prefer .dat time series; CSVs only for known metric stems (not *_stats).
        candidates = sorted(adir.glob("*.dat")) + sorted(adir.glob("*.csv"))
        for dat in candidates:
            if _is_stats_or_feature_table(dat):
                continue
            stem = dat.stem
            png = adir / f"{stem}.png"
            stem_l = stem.lower()
            if stem_l.startswith("fel_") or ("fel" in stem_l and "grid" in stem_l):
                continue
            xl, yl = "Time (ns)", "Value"
            matched_key = None
            for key, (x0, y0) in ylabel_by_stem.items():
                # Exact stem or metric prefix — but not rmsd_stats (already skipped)
                if stem_l == key or stem_l.startswith(key + "_"):
                    xl, yl = x0, y0
                    matched_key = key
                    break
            if "rmsf" in stem_l:
                xl = "Residue"
            # Multipanel only from .dat (or exact metric stem .csv like gyration.csv)
            use_in_panel = (
                matched_key
                and dat.suffix.lower() == ".dat"
                and dat.name not in panel_files
            )
            if use_in_panel:
                panel_files.append(dat.name)
                xlabels.append(xl)
                ylabels.append(yl)
                titles.append(matched_key.upper())

            if png.is_file():
                # Still attach plot path to analysis_summary when missing.
                try:
                    from src.analysis.summary_logger import update_analysis_summary_with_files
                    from src.analysis.data_plotter import _infer_analysis_type

                    atype = _infer_analysis_type(str(png), [str(dat)], wd)
                    if atype:
                        update_analysis_summary_with_files(
                            working_dir=wd,
                            analysis_type=atype,
                            additional_files={"plot": str(png)},
                        )
                except Exception:
                    pass
                if str(png) not in generated_files:
                    generated_files[str(png)] = f"Existing plot for {dat.name}"
                continue
            try:
                result = plot_md_data.func(
                    data_files=[str(dat)],
                    output_file=str(png),
                    xlabel=xl,
                    ylabel=yl,
                    titles=[stem],
                    working_dir=wd,
                )
                if result.get("success") and png.is_file():
                    generated_files[str(png)] = f"Post-hoc plot for {dat.name}"
                    execution_log.append(f"✓ Post-hoc plot: {png.name}")
                else:
                    warnings.append(
                        f"Post-hoc plot failed for {dat.name}: {result.get('error')}"
                    )
            except Exception as exc:
                warnings.append(f"Post-hoc plot failed for {dat.name}: {exc}")

        combined = adir / "combined_metrics.png"
        if len(panel_files) >= 2 and not combined.is_file():
            try:
                result = plot_multipanel.func(
                    data_files=[str(adir / f) for f in panel_files],
                    output_file=str(combined),
                    layout="vertical",
                    titles=titles,
                    xlabels=xlabels,
                    ylabels=ylabels,
                    working_dir=wd,
                )
                if result.get("success") and combined.is_file():
                    generated_files[str(combined)] = "Post-hoc combined metrics plot"
                    execution_log.append("✓ Post-hoc plot: combined_metrics.png")
                else:
                    # Non-fatal: overlays/stats already satisfy combined success
                    warnings.append(
                        f"Post-hoc combined_metrics.png failed: {result.get('error')}"
                    )
            except Exception as exc:
                warnings.append(f"Post-hoc combined_metrics.png failed: {exc}")
        elif combined.is_file() and str(combined) not in generated_files:
            generated_files[str(combined)] = "Existing combined metrics plot"

    def _sim_label_from_state(self, state: MDState) -> str:
        wd = str(state.get("working_directory") or "")
        if wd:
            return Path(wd).name
        return str(state.get("sim_label") or "")

    def _attach_retrieved_context(self, state: MDState, agent_input: AnalysisAgentInput) -> None:
        """Load knowledge, run memory, and study notes before the analysis model."""
        try:
            from agentic.campaign.agent_context import agent_context_block

            query = " ".join(collect_goal_texts_for_intent(state, agent_input))[:1200]
            block = agent_context_block(state, "analysis", query=query)
        except Exception:
            logger.debug("analysis retrieval skipped", exc_info=True)
            return
        if not block:
            return
        state["analysis_retrieved_context"] = block[:6000]
        log_agent_action("analysis", "Retrieved campaign context", {
            "chars": len(block),
            "source": "agent_retrieval",
        })

    def _record_analysis_outcome(self, state: MDState, *, success: bool, detail: str) -> None:
        """Write a campaign-level memory episode and a study note. Never raises."""
        label = self._sim_label_from_state(state)
        try:
            from agentic.campaign.memory import record_fix, remember_episode
            from agentic.campaign.study_notes import record_study_note, summarize_analysis_dir

            wd = str(state.get("working_directory") or "")
            numeric = summarize_analysis_dir(Path(wd) / "analysis", label=label) if wd else ""
            if success:
                record_fix(
                    state,
                    stage="analysis",
                    fix=f"{label or 'system'} analysis completed",
                )
                summary = numeric or f"{label or 'system'} analysis completed"
            else:
                remember_episode(
                    state=state,
                    stage="analysis",
                    error=(detail or "analysis failed")[:800],
                    tool="analysis",
                    label=label,
                )
                summary = numeric or f"{label or 'system'} analysis error: {(detail or '')[:400]}"
            record_study_note(state, stage="analysis", summary=summary, label=label)
        except Exception:
            logger.debug("analysis memory write skipped", exc_info=True)

    def _build_analysis_planning_prompt(self, agent_input: AnalysisAgentInput, 
                                        state: MDState) -> str:
        """Build LLM planning prompt - use planner's detailed instructions if available"""
        
        # Check if we have detailed instructions from planner
        if agent_input.additional_instructions:
            logger.info("Using planner's detailed instructions for analysis")
            prompt = self._build_prompt_from_planner_instructions(
                agent_input, agent_input.additional_instructions, state
            )
        else:
            prompt = self._build_standard_analysis_prompt(agent_input, state)
        extra = str(state.get("analysis_retrieved_context") or "")
        if not extra:
            try:
                from agentic.campaign.agent_context import agent_context_block

                extra = agent_context_block(
                    state,
                    "analysis",
                    query=" ".join(collect_goal_texts_for_intent(state, agent_input))[:1200],
                )
            except Exception:
                extra = ""
        if extra:
            prompt = f"{prompt}\n\n{extra}\n"
        return prompt

    def _build_prompt_from_planner_instructions(self, agent_input: AnalysisAgentInput,
                                                planner_instructions: str,
                                                state: MDState) -> str:
        """Build prompt using planner's detailed natural language instructions"""
        
        tool_metadata = self._get_analysis_tool_metadata(state)
        tools_list_str = self._format_tools_list_detailed(tool_metadata)
        scope_note = self._get_per_sim_tool_scope_note(state)
        input_files_block = self._format_trajectory_input_block(agent_input)
        goal_context = self._format_goal_context_for_planning(agent_input, state)

        goal_text = " ".join(collect_goal_texts_for_intent(state, agent_input)).lower()
        _pca_fel_block = (
            get_pca_fel_tool_guide()
            if any(k in goal_text for k in ("pca", "principal component", "free energy landscape", "fel", "energy landscape", "essential dynamics"))
            else ""
        )
        _classification_block = (
            get_classification_per_sim_tool_guide()
            if (
                detect_classification_requested(*collect_goal_texts_for_intent(state, agent_input))
                or detect_family_modular_dynamics_requested(
                    *collect_goal_texts_for_intent(state, agent_input)
                )
            )
            else ""
        )
        
        # Extract file registry information
        file_registry = state.get("file_registry", {})
        if file_registry:
            registry_str = "\n**Files Available from HPC:**\n"
            for file_path, metadata in file_registry.items():
                if metadata.get("stage") == "hpc":
                    filename = Path(file_path).name
                    file_type = metadata.get("type", "unknown")
                    description = metadata.get("description", "")
                    registry_str += f"- {filename} (type: {file_type}) - {description}\n"
        else:
            registry_str = "\n**Files Available from HPC:** None registered\n"
        
        # Extract PDB structural information from supervisor validation
        pdb_info_str = self._format_pdb_info_for_llm(state)
        
        return f"""You are a molecular dynamics analysis expert executing a detailed plan from the workflow planner.

**Available Data:**
- Topology File: {agent_input.topology_file or "Not available"}
- Trajectory File: {agent_input.trajectory_file or "Not available"}
- Energy File: {agent_input.energy_file or "Not available"}
{goal_context}
{input_files_block}
{registry_str}
{pdb_info_str}

**INTENT (HIGHEST PRIORITY — overrides planner if they conflict):**
The **Enriched User Goal** (when present) is the authoritative per-simulation scope.
Your JSON plan MUST include a calculate_*/analyze_* step plus plot_md_data (or plot_pca_projection)
for **every** metric named in the Enriched User Goal — not just the first one.
Planner instructions below are a reference checklist only: omit any planner step whose metric
is absent from the Enriched User Goal; add every metric the Enriched User Goal requests even
if the planner omits it.
If the Enriched User Goal says "RMSF only" (or similar exclusive language), plan ONLY that metric.

{get_standard_output_filenames_block()}

**DETAILED INSTRUCTIONS FROM PLANNER (guidance only — do not exceed User Goal scope):**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{planner_instructions}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Available Tools:**
{tools_list_str}

{scope_note}
**CRITICAL INSTRUCTIONS:**
- You MUST ONLY use the tools listed above - do NOT invent or suggest non-existent tools
- Every "tool_name" in your plan must match exactly one of the tool names listed above
- FORBIDDEN tool names: "none", "manual", "skip", "custom", "placeholder", or any made-up tool
- For input file parameters (topology_file, trajectory_file, energy_file etc.):
  Use ONLY the exact filenames from **Required input filenames** above.
  NEVER use trajectory.dcd, md.dcd, or invented paths.
- For plot_md_data use **data_files** (list), e.g. `"data_files": ["ligand_pocket_distance.csv"]` — not data_file.
- For plot_pca_projection use **pca_projections_file** (from calculate_trajectory_pca), not pca_file or input_file.
- For calculate_trajectory_pca use **projections_file** / **variance_file**, not output_file.
- For output file parameters (output_file, plot_file, csv_file etc.):
  Use ONLY the file name (e.g. "rmsd.dat"). The framework prepends the output directory.
- If a required capability is missing, either:
  a) Use available tools creatively to achieve the same goal, OR
  b) OMIT that step entirely from your plan (do NOT include it with tool_name="none")
- When you cannot perform a step, simply do NOT include it in the steps array

**MANDATORY PLOTTING RULE — CRITICAL:**
- After EVERY calculate_* or analyze_* step that produces a data file, you MUST immediately
  add a plot_md_data step to visualise that data.
- The plot step's data_files must be ["<output_file_from_previous_step>"] (filename only, no path).
- Example pair:
    {{"tool_name": "calculate_rmsd", "tool_params": {{"output_file": "rmsd.dat", ...}}}},
    {{"tool_name": "plot_md_data",   "tool_params": {{"data_files": ["rmsd.dat"], "output_file": "rmsd.png", "xlabel": "Time (ns)", "ylabel": "RMSD (Å)"}}}}
- Apply this pattern for RMSD, RMSF, Rg, energy, and any distance calculation.
{get_com_distance_tool_guide()}
{get_proximity_tool_guide()}
{CHAIN_SELECTION_LLM_NOTE}
{_pca_fel_block}
{_classification_block}
- **RMSF / DCCM selection — ALWAYS Cα only:**
  For calculate_rmsf and calculate_dccm you MUST set
  selection="protein and name CA" (or "(protein and resid X:Y) and name CA" for a segment).
  NEVER use selection="protein" (all atoms) — that produces per-atom profiles that break
  residue overlays. The framework will coerce non-Cα selections, but plans must be correct.
- **DCCM — only when the User Goal explicitly names DCCM for this simulation:**
  Include calculate_dccm only if DCCM is requested in the User Goal for this run.
  Do NOT add DCCM because the global project mentions it for other proteins.
  Use: selection="protein and name CA", frame_interval=5, output_prefix="dccm".

Your task: Create a detailed, step-by-step execution plan that implements **every analysis**
in the Enriched User Goal (or User Goal when no enriched text exists). Follow planner
instructions only where they match that scope. When ligand/pocket/COM distance is requested,
include the correct COM tool from the guide above (not both unless both are requested).
The plan should specify which tools to call and in what order.
ONLY include steps that use valid tools from the list above.
Your "steps" array must list ALL requested calculations and plots — a partial plan is invalid.

Output as JSON with this structure (object with "steps", NOT a tool-call array):
{{
  "reasoning": "How you'll implement the planner's instructions",
  "overview": "High-level summary",
  "steps": [
    {{
      "name": "step name",
      "description": "what it does",
      "tool_name": "tool to call",
      "tool_params": {{"topology_file": "{Path(agent_input.topology_file).name if agent_input.topology_file else 'md.tpr'}", "trajectory_file": "{Path(agent_input.trajectory_file).name if agent_input.trajectory_file else 'md.xtc'}", "output_file": "example.dat"}},
      "reason": "why it's needed per planner's instructions"
    }}
  ],
  "potential_issues": ["issue1"],
  "recommendations": ["rec1"]
}}
Return ONLY this JSON object — no markdown fences, no [{{"name":..., "arguments":...}}] array.
"""

    def _build_standard_analysis_prompt(self, agent_input: AnalysisAgentInput, state: MDState = None) -> str:
        """Build LLM planning prompt from config template using dynamic tool metadata"""
        config_prompt = self.config.get("llm", {}).get("planning_prompt_template", "")
        
        tool_metadata = self._get_analysis_tool_metadata(state)
        tools_list_str = self._format_tools_list_detailed(tool_metadata)
        scope_note = self._get_per_sim_tool_scope_note(state)
        input_files_block = self._format_trajectory_input_block(agent_input)
        goal_context = self._format_goal_context_for_planning(agent_input, state)
        
        # Build analysis context
        analysis_context = "\n".join([
            f"- Trajectory available: {bool(agent_input.trajectory_file)}",
            f"- Topology available: {bool(agent_input.topology_file)}",
            f"- Energy file available: {bool(agent_input.energy_file)}",
            f"- Requested analyses: {', '.join(agent_input.analyses) if agent_input.analyses else 'None specified'}",
            goal_context,
        ])
        
        # Extract PDB structural information if state provided
        pdb_info_str = self._format_pdb_info_for_llm(state) if state else ""
        
        # Use template from config or build basic prompt
        if config_prompt:
            return config_prompt.format(
                trajectory_file=agent_input.trajectory_file or "Not available",
                topology_file=agent_input.topology_file or "Not available",
                energy_file=agent_input.energy_file or "Not available",
                user_goal=agent_input.user_goal,
                enriched_goal=self._resolve_enriched_goal_for_planning(state, agent_input)
                or agent_input.user_goal
                or "Not specified",
                hpc_output_dir=agent_input.hpc_output_dir or "Not specified",
                analysis_context=analysis_context,
                tools_list=tools_list_str + ("\n\n" + scope_note if scope_note else ""),
            )
        else:
            # Fallback prompt if config template missing
            return f"""You are a molecular dynamics analysis expert. Create an analysis plan for:

**Available Data:**
- Topology: {agent_input.topology_file or "Not available"}
- Trajectory: {agent_input.trajectory_file or "Not available"}
- Energy File: {agent_input.energy_file or "Not available"}
- User Goal: {agent_input.user_goal}
{input_files_block}
{pdb_info_str}

**Available Tools:**
{tools_list_str}

{scope_note}
**CRITICAL:**
- You MUST ONLY use the tools listed above. Do NOT invent or suggest non-existent tools.
- For input file parameters (topology_file, trajectory_file, energy_file etc.):
  Use ONLY the file name (e.g. "md.gro"), NOT a full path.
  The framework resolves correct paths automatically.
- For output file parameters: Use ONLY the file name (e.g. "rmsd.dat").

Return JSON with: reasoning, overview, steps (name, description, tool_name, tool_params, reason)
"""

    def _extract_plan_json(self, content: str) -> Dict[str, Any]:
        """Extract and parse JSON plan from LLM response."""
        text = (content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        def _try_parse(raw: str) -> Optional[Dict[str, Any]]:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, list):
                return self._coerce_tool_call_array_to_plan(parsed)
            if isinstance(parsed, dict):
                if "steps" not in parsed and parsed.get("tool_name"):
                    return self._coerce_tool_call_array_to_plan([parsed])
                return parsed
            return None

        result = _try_parse(text)
        if result is not None:
            return result

        for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
            match = re.search(pattern, text)
            if match:
                result = _try_parse(match.group())
                if result is not None:
                    return result

        return {
            "reasoning": content,
            "overview": "MD trajectory analysis plan",
            "steps": [],
        }

    def _coerce_tool_call_array_to_plan(self, items: List[Any]) -> Dict[str, Any]:
        """Convert [{name, arguments}] tool-call JSON into a structured plan dict."""
        steps: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool_name") or item.get("name") or ""
            params = dict(item.get("tool_params") or item.get("arguments") or {})
            steps.append(
                {
                    "name": item.get("name") or tool_name or "step",
                    "description": item.get("description", ""),
                    "tool_name": tool_name,
                    "tool_params": params,
                    "reason": item.get("reason", "LLM plan step"),
                }
            )
        return {
            "reasoning": "Converted from LLM tool-call array response",
            "overview": "MD trajectory analysis plan",
            "steps": steps,
            "potential_issues": ["LLM returned tool-call array instead of plan object"],
            "recommendations": [],
        }

    _BAD_TRAJECTORY_NAMES = frozenset(
        {"trajectory.dcd", "traj.dcd", "md.dcd", "trajectory.xtc", "topology.tpr"}
    )

    def _normalize_plan_steps(
        self,
        plan_dict: Dict[str, Any],
        agent_input: AnalysisAgentInput,
    ) -> Dict[str, Any]:
        """Repair common LLM parameter mistakes before tool execution."""
        topo = (
            Path(agent_input.topology_file).name
            if agent_input.topology_file
            else "md.tpr"
        )
        traj = (
            Path(agent_input.trajectory_file).name
            if agent_input.trajectory_file
            else "md.xtc"
        )
        traj_prefixes = ("calculate_", "analyze_", "wrap_trajectory")
        _SKIP_PLOT_DATA = frozenset({
            "fel_pc1_pc2_grid.csv",
            "fel_features.csv",
            "fel_basins.csv",
            "fel_basin_structures.csv",
            "fel_features.json",
        })
        kept_steps: List[Dict[str, Any]] = []

        for step in plan_dict.get("steps") or []:
            if not isinstance(step, dict):
                continue
            tool = step.get("tool_name") or ""
            params = dict(step.get("tool_params") or {})

            if tool == "plot_md_data":
                for alias in ("data_file", "input_file", "csv_file"):
                    if alias in params and "data_files" not in params:
                        val = params.pop(alias)
                        params["data_files"] = [val] if isinstance(val, str) else val
                data_names = [
                    Path(str(df)).name.lower()
                    for df in (params.get("data_files") or [])
                ]
                out_name = Path(str(params.get("output_file") or "")).name.lower()
                if (
                    any(n in _SKIP_PLOT_DATA for n in data_names)
                    or any("fel" in n and "grid" in n and n.endswith(".csv") for n in data_names)
                    or out_name in {
                        "fel_pc1_pc2_grid.png",
                        "fel_features.png",
                        "fel_pc1_pc2.png",
                    }
                ):
                    logger.info(
                        "_normalize_plan_steps: dropping plot_md_data for FEL "
                        "grid/features (%s → %s)",
                        data_names,
                        out_name,
                    )
                    continue

            if tool == "plot_pca_projection":
                for alias in ("pca_file", "input_file", "pca_projections"):
                    if alias in params and "pca_projections_file" not in params:
                        val = params.pop(alias)
                        if val in ("pca.csv", "pca.dat"):
                            val = "pca_projections.dat"
                        params["pca_projections_file"] = val
                out = str(params.get("output_file") or "")
                if not out or Path(out).name.lower() in {
                    "pca_pc1_pc2.png",
                    "pca.png",
                }:
                    params["output_file"] = "pca_pc1_pc2_time.png"

            if tool == "calculate_trajectory_pca":
                params.pop("output_file", None)
                params.setdefault("projections_file", "pca_projections.dat")
                params.setdefault("variance_file", "pca_variance.dat")

            if tool == "calculate_free_energy_landscape":
                for alias in ("pca_file", "input_file"):
                    if alias in params and "pca_projections_file" not in params:
                        val = params.pop(alias)
                        if val in ("pca.csv", "pca.dat"):
                            val = "pca_projections.dat"
                        params["pca_projections_file"] = val
                # Prefer grid CSV only; annotated map comes from analyze_fel.
                params["output_plot"] = ""
                params.setdefault("output_grid", "fel_pc1_pc2_grid.csv")

            if tool == "analyze_fel_landscape_features":
                out_plot = str(params.get("output_plot") or "").strip()
                bad_fel_plots = {
                    "fel_features.png",
                    "fel_pc1_pc2.png",
                    "fel_pc1_pc2_grid.png",
                }
                if not out_plot or Path(out_plot).name.lower() in bad_fel_plots:
                    params["output_plot"] = "fel_basins.png"

            needs_traj = any(tool.startswith(p) for p in traj_prefixes)
            if needs_traj:
                cur_traj = str(params.get("trajectory_file") or "").lower()
                if (
                    not params.get("topology_file")
                    or not params.get("trajectory_file")
                    or cur_traj in self._BAD_TRAJECTORY_NAMES
                    or cur_traj.endswith(".dcd")
                ):
                    params["topology_file"] = topo
                    params["trajectory_file"] = traj

            step["tool_params"] = params
            kept_steps.append(step)
        plan_dict["steps"] = kept_steps
        return plan_dict

    def replan_with_guidance(self, human_recommendation: str, state: dict) -> Optional[dict]:
        """Update the structured analysis plan by applying human guidance.

        If a current structured plan exists in state: sends it together with the human
        recommendation to the LLM so ONLY the requested changes are made (tool_params,
        steps, parameters).  Falls back to full re-planning via the normal prompt
        infrastructure when no current plan is available.

        Returns the updated plan dict, or None if the LLM call fails.
        """
        if not (self.llm and self.llm.available):
            return None

        import json as _j

        current_plan = (
            (state.get("execution_plan") or {})
            .get("structured_plans", {})
            .get("analysis")
        )

        if current_plan:
            # Modification mode: keep existing plan, apply targeted changes
            tool_metadata = self._get_analysis_tool_metadata(state)
            tools_str = self._format_tools_list_detailed(tool_metadata)
            scope_note = self._get_per_sim_tool_scope_note(state)
            prompt = (
                f"You are updating an MD trajectory analysis execution plan.\n\n"
                f"CURRENT PLAN (JSON):\n```json\n{_j.dumps(current_plan, indent=2)}\n```\n\n"
                f"HUMAN GUIDANCE (apply ONLY these changes):\n{human_recommendation}\n\n"
                f"{scope_note}\n"
                f"Available tools for reference (tool_name must match):\n{tools_str}\n\n"
                f"Rules:\n"
                f"- Apply ONLY the changes the human requested.\n"
                f"- Update tool_params values, add/remove/reorder steps as needed.\n"
                f"- Keep all other steps and fields exactly as they are.\n"
                f"- Preserve JSON structure: reasoning, overview, steps, potential_issues, recommendations.\n"
                f"- Each step must have: name, description, tool_name, tool_params, reason.\n"
                f"- Return ONLY valid JSON — no explanation, no markdown fences.\n"
            )
        else:
            # Fresh planning mode: build from planner NL instructions + human guidance
            nl_instructions = (
                (state.get("execution_plan") or {})
                .get("agent_plans", {})
                .get("analysis_agent", "")
            )
            augmented = (
                nl_instructions
                + "\n\n**HUMAN RECOMMENDATION (must be followed):**\n" + human_recommendation
            ) if nl_instructions else human_recommendation
            working_dir = state.get("working_directory", ".")
            hpc_output_dir = state.get("hpc_output_dir") or str(Path(working_dir) / "hpc")
            agent_input = AnalysisAgentInput(
                working_directory=working_dir,
                hpc_output_dir=hpc_output_dir,
                topology_file=state.get("topology"),
                trajectory_file=state.get("trajectory"),
                energy_file=state.get("energy_file"),
                user_goal=state.get("user_goal", ""),
                additional_instructions=augmented,
            )
            prompt = self._build_analysis_planning_prompt(agent_input, state)

        try:
            resp = self.llm.prompt_raw(prompt, temperature=0.1, max_tokens=16384, format="json")
            plan_dict = self._extract_plan_json(resp)
            if current_plan:
                return plan_dict
            plan_dict = self._normalize_plan_steps(plan_dict, agent_input)
            return plan_dict
        except Exception:
            return None

    def _create_fallback_analysis_plan(
        self,
        agent_input: AnalysisAgentInput,
        state: Optional[MDState] = None,
    ) -> AnalysisPlan:
        """
        Create template-based fallback plan when LLM fails.
        Respects explicit metric scope from the user goal when present.
        """
        if state and state.get("hitl_chat_task") and self._is_combined_hitl_context(state):
            return self._create_combined_hitl_fallback_plan(agent_input, state)

        steps: List[AnalysisStep] = []

        topo_name = Path(agent_input.topology_file).name if agent_input.topology_file else None
        traj_name = Path(agent_input.trajectory_file).name if agent_input.trajectory_file else None
        energy_name = Path(agent_input.energy_file).name if agent_input.energy_file else None

        _goal_lower = " ".join(collect_goal_texts_for_intent(state, agent_input)).lower()
        requested = self._intent_metrics(state, agent_input)
        if requested is None:
            requested = frozenset({"rmsd", "rmsf", "rg"})

        if traj_name and topo_name:
            if "rmsd" in requested:
                steps.extend([
                    AnalysisStep(
                        name="Calculate RMSD",
                        description="Calculate Root Mean Square Deviation to assess structural stability",
                        tool_name="calculate_rmsd",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "selection": "protein and name CA",
                            "output_file": "rmsd.dat",
                        },
                        reason="RMSD indicates structural stability over time",
                    ),
                    AnalysisStep(
                        name="Plot RMSD",
                        description="Plot RMSD time-series",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["rmsd.dat"],
                            "output_file": "rmsd.png",
                            "xlabel": "Time (ns)",
                            "ylabel": "RMSD (Å)",
                        },
                        reason="Visualise RMSD stability",
                    ),
                ])
            if "rmsf" in requested:
                steps.extend([
                    AnalysisStep(
                        name="Calculate RMSF",
                        description="Calculate Root Mean Square Fluctuation to identify flexible regions",
                        tool_name="calculate_rmsf",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "selection": "protein and name CA",
                            "output_file": "rmsf.dat",
                        },
                        reason="RMSF identifies flexible and rigid regions",
                    ),
                    AnalysisStep(
                        name="Plot RMSF",
                        description="Plot per-residue RMSF",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["rmsf.dat"],
                            "output_file": "rmsf.png",
                            "xlabel": "Residue",
                            "ylabel": "RMSF (Å)",
                        },
                        reason="Visualise per-residue flexibility",
                    ),
                ])
            if "rg" in requested:
                steps.extend([
                    AnalysisStep(
                        name="Calculate Radius of Gyration",
                        description="Calculate radius of gyration to assess protein compactness",
                        tool_name="calculate_radius_of_gyration",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "selection": "protein",
                            "output_file": "gyration.dat",
                        },
                        reason="Radius of gyration indicates protein compactness",
                    ),
                    AnalysisStep(
                        name="Plot Radius of Gyration",
                        description="Plot Rg time-series",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["gyration.dat"],
                            "output_file": "gyration.png",
                            "xlabel": "Time (ns)",
                            "ylabel": "Rg (Å)",
                        },
                        reason="Visualise compactness over time",
                    ),
                ])
            if "dccm" in requested:
                steps.append(
                    AnalysisStep(
                        name="Calculate DCCM",
                        description="Cα dynamic cross-correlation matrix",
                        tool_name="calculate_dccm",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "selection": "protein and name CA",
                            "output_prefix": "dccm",
                            "create_heatmap": True,
                        },
                        reason="User requested a Cα DCCM",
                    )
                )
            if "dssp" in requested:
                steps.append(
                    AnalysisStep(
                        name="Secondary structure",
                        description="DSSP assignment over the whole protein",
                        tool_name="analyze_secondary_structure",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "selection": "protein",
                            "output_prefix": "dssp",
                            "create_heatmap": True,
                        },
                        reason="User requested DSSP time evolution",
                    )
                )
                window = re.search(
                    r"residues?\s+(\d+)\s*[\u2013\-]\s*(\d+)",
                    _goal_lower,
                )
                if window:
                    start, end = window.group(1), window.group(2)
                    steps.append(
                        AnalysisStep(
                            name=f"Secondary structure {start}-{end}",
                            description=f"DSSP for residues {start}-{end}",
                            tool_name="analyze_secondary_structure",
                            tool_params={
                                "topology_file": topo_name,
                                "trajectory_file": traj_name,
                                "selection": f"protein and resid {start}:{end}",
                                "output_prefix": f"dssp_{start}to{end}",
                                "create_heatmap": True,
                            },
                            reason="User requested DSSP for a residue window",
                        )
                    )

            _lig_resname = (
                self._detect_ligand_resname(agent_input, state)
                if state
                else None
            )
            _is_holo = bool(state) and self._is_holo_simulation(state, agent_input)

            if "com" in requested and _is_holo and _lig_resname:
                steps.extend([
                    AnalysisStep(
                        name="Ligand Pocket Distance",
                        description=(
                            f"Identify protein pocket atoms within 5 Å of {_lig_resname} at frame 0, "
                            "then track COM-to-COM distance over the trajectory"
                        ),
                        tool_name="calculate_ligand_pocket_distance",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "ligand_selection": f"resname {_lig_resname}",
                            "protein_selection": "protein",
                            "cutoff": 5.0,
                            "output_file": "ligand_pocket_distance.csv",
                        },
                        reason="Track whether ligand stays in binding pocket",
                    ),
                    AnalysisStep(
                        name="Plot Ligand Pocket Distance",
                        description="Plot ligand-to-pocket COM distance over time",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["ligand_pocket_distance.csv"],
                            "output_file": "ligand_pocket_distance.png",
                            "x_col": 1,
                            "y_col": 2,
                            "xlabel": "Time (ns)",
                            "ylabel": "COM Distance (Å)",
                        },
                        reason="Visualise ligand displacement from catalytic pocket",
                    ),
                ])

            if "contacts" in requested and _is_holo and _lig_resname:
                steps.extend([
                    AnalysisStep(
                        name="Protein-Ligand Contacts",
                        description="Count protein–ligand H-bonds and heavy-atom contacts per frame",
                        tool_name="calculate_protein_ligand_contacts",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "ligand_selection": f"resname {_lig_resname}",
                            "output_file": "protein_ligand_contacts.csv",
                        },
                        reason="User requested protein–ligand contact analysis",
                    ),
                    AnalysisStep(
                        name="Plot Protein-Ligand Contacts",
                        description="Plot contact and H-bond counts over time",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["protein_ligand_contacts.csv"],
                            "output_file": "protein_ligand_contacts.png",
                            "xlabel": "Time (ns)",
                            "ylabel": "Count",
                        },
                        reason="Visualise protein–ligand interactions",
                    ),
                ])

            if "pocket_sasa" in requested and _is_holo and _lig_resname:
                steps.extend([
                    AnalysisStep(
                        name="Pocket SASA",
                        description="Calculate solvent-accessible surface area of the binding pocket",
                        tool_name="calculate_pocket_sasa",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "ligand_selection": f"resname {_lig_resname}",
                            "output_file": "pocket_sasa.csv",
                        },
                        reason="User requested pocket SASA",
                    ),
                    AnalysisStep(
                        name="Plot Pocket SASA",
                        description="Plot pocket SASA over time",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["pocket_sasa.csv"],
                            "output_file": "pocket_sasa.png",
                            "xlabel": "Time (ns)",
                            "ylabel": "SASA (Å²)",
                        },
                        reason="Visualise pocket accessibility",
                    ),
                ])

            if "residence" in requested and _is_holo and _lig_resname:
                steps.extend([
                    AnalysisStep(
                        name="Ligand Residence",
                        description="Analyze ligand residence time and unbinding events",
                        tool_name="analyze_ligand_residence",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "ligand_selection": f"resname {_lig_resname}",
                            "output_file": "ligand_residence.csv",
                        },
                        reason="User requested ligand residence/unbinding analysis",
                    ),
                    AnalysisStep(
                        name="Plot Ligand Residence",
                        description="Plot minimum protein–ligand contact distance over time",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["ligand_residence.csv"],
                            "output_file": "ligand_residence.png",
                            "xlabel": "Time (ns)",
                            "ylabel": "Min contact distance (Å)",
                        },
                        reason="Visualise ligand binding proximity",
                    ),
                ])

            if "pocket_rmsf" in requested and _is_holo and _lig_resname:
                steps.extend([
                    AnalysisStep(
                        name="Pocket RMSF",
                        description="Calculate per-residue RMSF for binding-pocket residues",
                        tool_name="calculate_pocket_rmsf",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "ligand_selection": f"resname {_lig_resname}",
                            "output_file": "pocket_rmsf.dat",
                        },
                        reason="User requested pocket RMSF",
                    ),
                    AnalysisStep(
                        name="Plot Pocket RMSF",
                        description="Plot pocket residue flexibility",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["pocket_rmsf.dat"],
                            "output_file": "pocket_rmsf.png",
                            "xlabel": "Residue",
                            "ylabel": "RMSF (Å)",
                            "plot_type": "line",
                        },
                        reason="Visualise pocket flexibility",
                    ),
                ])

            if "ligand_rmsf" in requested and _is_holo and _lig_resname:
                steps.extend([
                    AnalysisStep(
                        name="Ligand RMSF",
                        description="Calculate per-atom RMSF for the ligand",
                        tool_name="calculate_ligand_rmsf",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "ligand_selection": f"resname {_lig_resname}",
                            "output_file": "ligand_rmsf.dat",
                        },
                        reason="User requested ligand RMSF",
                    ),
                    AnalysisStep(
                        name="Plot Ligand RMSF",
                        description="Plot ligand atom flexibility",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["ligand_rmsf.dat"],
                            "output_file": "ligand_rmsf.png",
                            "xlabel": "Atom",
                            "ylabel": "RMSF (Å)",
                            "plot_type": "bar",
                        },
                        reason="Visualise ligand flexibility",
                    ),
                ])

            if "pca" in requested:
                steps.extend([
                    AnalysisStep(
                        name="Trajectory PCA",
                        description="PCA on Cα coordinates to capture collective motions",
                        tool_name="calculate_trajectory_pca",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "selection": "protein and name CA",
                            "projections_file": "pca_projections.dat",
                            "variance_file": "pca_variance.dat",
                        },
                        reason="User requested PCA on Cα",
                    ),
                    AnalysisStep(
                        name="Plot PCA Projection",
                        description="PC1 vs PC2 scatter coloured by time",
                        tool_name="plot_pca_projection",
                        tool_params={
                            "pca_projections_file": "pca_projections.dat",
                            "output_file": "pca_pc1_pc2_time.png",
                        },
                        reason="Visualise PCA conformational sampling",
                    ),
                ])

            if "fel" in requested:
                steps.extend([
                    AnalysisStep(
                        name="Free-Energy Landscape",
                        description="Build FEL grid from PCA projections at 310 K",
                        tool_name="calculate_free_energy_landscape",
                        tool_params={
                            "pca_projections_file": "pca_projections.dat",
                            "temperature_k": 310.0,
                            "output_plot": "",
                            "output_grid": "fel_pc1_pc2_grid.csv",
                        },
                        reason="User requested free-energy landscape at 310 K",
                    ),
                    AnalysisStep(
                        name="FEL Basin Features",
                        description="Extract basin depths, barriers, and landscape entropy",
                        tool_name="analyze_fel_landscape_features",
                        tool_params={
                            "fel_grid_file": "fel_pc1_pc2_grid.csv",
                            "temperature_k": 310.0,
                            "output_json": "fel_features.json",
                            "output_csv": "fel_features.csv",
                            "output_basins_csv": "fel_basins.csv",
                            "output_plot": "fel_basins.png",
                        },
                        reason="User requested FEL basin features",
                    ),
                    AnalysisStep(
                        name="Export FEL Basin Structures",
                        description=(
                            "Write representative PDB per FEL basin for "
                            "visualising transient conformations"
                        ),
                        tool_name="export_fel_basin_structures",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "fel_features_file": "fel_features.json",
                            "pca_projections_file": "pca_projections.dat",
                            "manifest_file": "fel_basin_structures.csv",
                        },
                        reason="User requested FEL basin structures for highlighting",
                    ),
                ])

            # Family modular dynamics features
            if "consensus_torsions" in requested or "dihedral_pca" in requested:
                steps.append(
                    AnalysisStep(
                        name="Consensus dihedrals (φ/ψ/χ₁)",
                        description="Mapped φ/ψ/χ₁ circular means for domain and pocket",
                        tool_name="calculate_consensus_torsions",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "output_dir": "consensus_dihedrals",
                        },
                        reason="Family modular: pocket χ₁ + dihedral PCA inputs",
                    )
                )
            if "consensus_rmsf" in requested:
                steps.append(
                    AnalysisStep(
                        name="Consensus Cα RMSF",
                        description="Mapped consensus Cα RMSF mean/std across the domain",
                        tool_name="calculate_consensus_rmsf_features",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "output_dir": "consensus_rmsf",
                        },
                        reason="Family modular: consensus RMSF features",
                    )
                )
            if "consensus_dccm" in requested:
                steps.append(
                    AnalysisStep(
                        name="Consensus DCCM N↔C",
                        description="Mapped DCCM mean absolute and N-lobe↔C-lobe correlation",
                        tool_name="calculate_consensus_dccm_features",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "output_dir": "consensus_DCCM",
                        },
                        reason="Family modular: DCCM N↔C feature",
                    )
                )
            if "dihedral_pca" in requested:
                steps.append(
                    AnalysisStep(
                        name="Independent dihedral PCA FEL",
                        description=(
                            "Per-sim φ/ψ/χ₁ PCA → FEL grid entropy"
                        ),
                        tool_name="run_independent_dynamics_fel",
                        tool_params={
                            "space": "dihedral",
                            "method": "pca",
                            "dihedral_dir": "consensus_dihedrals",
                            "dihedral_angles": ["phi", "psi", "chi1"],
                            "output_dir": "consensus_PCA",
                        },
                        reason="Family modular: dihedral landscape entropy",
                    )
                )

            if "nearby" in requested:
                _qsel = "chainID B and resid 1:34" if (
                    "1 to 34" in _goal_lower or "1-34" in _goal_lower or "1–34" in _goal_lower
                ) else "chainID B"
                steps.append(AnalysisStep(
                    name="Identify nearby residues",
                    description="Freeze neighbor residues within 15 Å of the query group at frame 0",
                    tool_name="identify_nearby_residues",
                    tool_params={
                        "topology_file": topo_name,
                        "trajectory_file": traj_name,
                        "query_selection": _qsel,
                        "neighbor_selection": "chainID A",
                        "cutoff": 15.0,
                        "frame": 0,
                        "output_file": "nearby_residues_A_near_B1to34.json" if "34" in _qsel else "nearby_residues.json",
                    },
                    reason="User requested nearby / interface residues",
                ))

            if "min_distance" in requested:
                steps.extend([
                    AnalysisStep(
                        name="Minimum heavy-atom distance",
                        description="Per-frame minimum heavy-atom distance between the interface groups",
                        tool_name="calculate_min_heavy_atom_distance",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "selection1": "chainID B and resid 1:34" if "34" in _goal_lower else "chainID B",
                            "selection2": "chainID A",
                            "label1": "B1to34",
                            "label2": "nearbyA",
                            "output_file": "min_distance_B1to34_vs_nearbyA.csv",
                        },
                        reason="User requested minimum heavy-atom distance at the interface",
                    ),
                    AnalysisStep(
                        name="Plot minimum heavy-atom distance",
                        description="Plot min heavy-atom distance over time",
                        tool_name="plot_md_data",
                        tool_params={
                            "data_files": ["min_distance_B1to34_vs_nearbyA.csv"],
                            "output_file": "min_distance_B1to34_vs_nearbyA.png",
                            "xlabel": "Time (ns)",
                            "ylabel": "Min distance (Å)",
                        },
                        reason="Visualise whether the interface stays in contact",
                    ),
                ])

            if "hbond_occupancy" in requested:
                _hb_sel1 = "chainID B and resid 1:34" if (
                    "1 to 34" in _goal_lower or "1-34" in _goal_lower or "1–34" in _goal_lower
                ) else "chainID B"
                steps.extend([
                    AnalysisStep(
                        name="Hydrogen-bond occupancy",
                        description="Residue-pair H-bond occupancy between the requested protein groups",
                        tool_name="calculate_hbond_occupancy",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "selection1": _hb_sel1,
                            "selection2": "chainID A",
                            "label1": "B1to34" if "34" in _hb_sel1 else "B",
                            "label2": "A",
                            "output_file": "hbond_occupancy_B1to34_vs_A.csv" if "34" in _hb_sel1 else "hbond_occupancy.csv",
                        },
                        reason="User requested interface H-bond occupancy / interaction partners",
                    ),
                ])

            if "salt_bridge" in requested:
                _sb_sel1 = "chainID B and resid 1:34" if (
                    "1 to 34" in _goal_lower or "1-34" in _goal_lower or "1–34" in _goal_lower
                ) else "chainID B"
                steps.extend([
                    AnalysisStep(
                        name="Salt-bridge distances",
                        description="Charged-pair distances and occupancy at the protein–protein interface",
                        tool_name="calculate_salt_bridge_distances",
                        tool_params={
                            "topology_file": topo_name,
                            "trajectory_file": traj_name,
                            "selection1": _sb_sel1,
                            "selection2": "chainID A",
                            "label1": "B1to34" if "34" in _sb_sel1 else "B",
                            "label2": "A",
                            "output_file": "saltbridge_occupancy_B1to34_vs_A.csv" if "34" in _sb_sel1 else "saltbridge_occupancy.csv",
                        },
                        reason="User requested salt-bridge / charged interaction partners",
                    ),
                ])

            _has_dccm_request = "dccm" in requested or any(
                kw in _goal_lower for kw in
                ["dccm", "cross-correlation", "cross correlation",
                 "correlated motion", "allosteric", "coupled motion"]
            )
            if _has_dccm_request:
                steps.append(AnalysisStep(
                    name="Calculate Dynamic Cross-Correlation Matrix (DCCM)",
                    description=(
                        "Compute normalised DCCM of Cα fluctuations to reveal correlated "
                        "and anti-correlated residue motions."
                    ),
                    tool_name="calculate_dccm",
                    tool_params={
                        "topology_file": topo_name,
                        "trajectory_file": traj_name,
                        "selection": "protein and name CA",
                        "output_prefix": "dccm",
                        "frame_interval": 5,
                        "save_matrix_csv": True,
                        "create_heatmap": True,
                    },
                    reason="DCCM reveals allosteric communication patterns",
                ))

        if energy_name and "energy" in requested:
            steps.extend([
                AnalysisStep(
                    name="Analyze Energy",
                    description="Extract and analyze energy terms from simulation",
                    tool_name="analyze_energy",
                    tool_params={"energy_file": energy_name, "output_file": "energy.dat"},
                    reason="Energy analysis assesses simulation stability",
                ),
                AnalysisStep(
                    name="Plot Energy",
                    description="Plot energy terms over time",
                    tool_name="plot_md_data",
                    tool_params={
                        "data_files": ["energy.dat"],
                        "output_file": "energy.png",
                        "xlabel": "Time (ns)",
                        "ylabel": "Energy (kJ/mol)",
                    },
                    reason="Visualise thermodynamic equilibration",
                ),
            ])

        if traj_name and topo_name and len(requested) >= 2:
            _panel_files: List[str] = []
            _panel_xl: List[str] = []
            _panel_yl: List[str] = []
            _panel_titles: List[str] = []
            _panel_map = {
                "rmsd": ("rmsd.dat", "Time (ns)", "RMSD (Å)", "RMSD"),
                "rmsf": ("rmsf.dat", "Residue", "RMSF (Å)", "RMSF"),
                "rg": ("gyration.dat", "Time (ns)", "Rg (Å)", "Radius of Gyration"),
                "energy": ("energy.dat", "Time (ns)", "Energy (kJ/mol)", "Energy"),
            }
            for metric in ("rmsd", "rmsf", "rg", "energy"):
                if metric in requested and metric in _panel_map:
                    f, xl, yl, tl = _panel_map[metric]
                    if metric != "energy" or energy_name:
                        _panel_files.append(f)
                        _panel_xl.append(xl)
                        _panel_yl.append(yl)
                        _panel_titles.append(tl)
            if len(_panel_files) >= 2:
                steps.append(AnalysisStep(
                    name="Create Combined Metrics Plot",
                    description="Multi-panel summary figure of requested metrics in one PNG.",
                    tool_name="plot_multipanel",
                    tool_params={
                        "data_files": _panel_files,
                        "output_file": "combined_metrics.png",
                        "layout": "vertical",
                        "titles": _panel_titles,
                        "xlabels": _panel_xl,
                        "ylabels": _panel_yl,
                    },
                    reason="Combined overview when multiple metrics were requested.",
                ))

        metrics_label = ", ".join(sorted(requested))
        return AnalysisPlan(
            reasoning=f"Intent-driven fallback plan for: {metrics_label}",
            overview=f"Trajectory analysis: {metrics_label}",
            steps=steps,
            potential_issues=["Requires trajectory and topology files"],
            recommendations=["Verify all files exist before execution"],
        )

    def _run_trajectory_batch_precache(
        self,
        plan: AnalysisPlan,
        state: MDState,
    ) -> Dict[int, Dict[str, Any]]:
        """Pre-compute trajectory metrics in minimal RAW/ALIGNED passes."""
        try:
            from src.analysis.trajectory_batch import (
                run_plan_trajectory_batches,
                summarize_batch_plan,
            )
        except ImportError:
            return {}

        input_map = self._resolve_input_files(state)
        topology = input_map.get("topology")
        trajectory = input_map.get("trajectory")
        if not topology or not trajectory:
            logger.info("Trajectory batching skipped: topology/trajectory not resolved")
            return {}

        prepared: Dict[int, Dict[str, Any]] = {}
        output_param_names = [
            "output_file", "output_prefix", "plot_file", "figure_path",
            "csv_file", "dat_file", "save_path", "output_csv", "output_fig",
        ]
        input_param_to_type = {
            "topology_file": "topology",
            "topology": "topology",
            "structure": "topology",
            "trajectory_file": "trajectory",
            "trajectory": "trajectory",
            "traj": "trajectory",
        }

        for idx, step in enumerate(plan.steps):
            params = sanitize_tool_output_params(dict(step.tool_params or {}))
            for pname in output_param_names:
                if pname in params and params[pname]:
                    params[pname] = os.path.basename(str(params[pname]))
            for pname, ftype in input_param_to_type.items():
                if pname in params and input_map.get(ftype):
                    params[pname] = input_map[ftype]
            params["working_dir"] = str(self.file_manager.agent_dir)
            prepared[idx] = params

        from src.analysis.proximity_analyzer import apply_selection_from_files

        pre_results: Dict[int, Dict[str, Any]] = {}
        analysis_dir = str(self.file_manager.agent_dir)
        for idx, step in enumerate(plan.steps):
            if step.tool_name != "identify_nearby_residues":
                continue
            logger.info("Pre-running identify_nearby_residues before trajectory batch")
            try:
                result = self.tool_executor.execute(step.tool_name, **prepared[idx])
            except Exception as exc:
                logger.exception("identify_nearby_residues pre-batch failed: %s", exc)
                result = {"success": False, "error": str(exc)}
            pre_results[idx] = result

        for idx, params in list(prepared.items()):
            prepared[idx] = apply_selection_from_files(params, working_dir=analysis_dir)

        logger.info(summarize_batch_plan(plan.steps))
        try:
            batch_results = run_plan_trajectory_batches(
                plan.steps,
                topology_file=topology,
                trajectory_file=trajectory,
                working_dir=str(self.file_manager.agent_dir),
                prepared_params=prepared,
            )
        except Exception as exc:
            logger.exception(
                "Trajectory batch failed; continuing with per-step execution: %s",
                exc,
            )
            batch_results = {}
        batch_results.update(pre_results)
        return batch_results

    def _execute_analysis_plan(self, agent_input: AnalysisAgentInput, 
                               plan: AnalysisPlan,
                               state: MDState) -> AnalysisExecutionResult:
        """
        Execute analysis plan step-by-step using tool executor.
        Tracks outputs and handles errors gracefully.
        """
        execution_log = []
        issues = []
        warnings = []
        generated_files = {}
        results = {}
        
        # Get execution limits from config
        agent_config = self.config
        max_retries = agent_config.get("max_tool_retries", 2)
        max_steps = agent_config.get("max_total_steps", 40)
        fail_fast = agent_config.get("fail_fast", False)
        
        # Get analysis directory
        analysis_dir = state.get("analysis_directory", self.tool_executor.working_dir)
        
        try:
            # Enforce max steps limit
            if len(plan.steps) > max_steps:
                warnings.append(f"Plan has {len(plan.steps)} steps, limiting to {max_steps}")
                plan.steps = plan.steps[:max_steps]

            batch_results: Dict[int, Dict[str, Any]] = {}
            # Multi-rep: skip trajectory batching so metrics go through
            # AnalysisToolExecutor fan-out into analysis/repXX/ + avg/.
            use_batch = bool(agent_config.get("use_trajectory_batching", True))
            try:
                rn = _effective_rep_num_for_state(state)
                if rn > 1:
                    use_batch = False
                    logger.info(
                        "Trajectory batching disabled for multi-rep (rep_num=%d); "
                        "using executor fan-out",
                        rn,
                    )
            except Exception:
                pass
            if use_batch:
                batch_results = self._run_trajectory_batch_precache(plan, state)
                if batch_results:
                    execution_log.append(
                        f"\n=== Trajectory batch pre-computed {len(batch_results)} metric(s) ==="
                    )

            from agentic.campaign.contracts import (
                decide_analysis_success,
                family_tool_already_done,
                is_noncritical_analysis_tool,
                per_sim_science_complete,
            )
            from agentic.campaign.spec import spec_from_state

            retry_n = int(state.get("analysis_retry_count") or 0)
            # On retry, skip plot tools only when their PNG already exists.
            # Never skip post-hoc PNG materialization — retry_count>0 was
            # leaving .dat-only analysis with no figures in the report.
            
            for i, step in enumerate(plan.steps):
                # Validate tool name - skip invalid placeholder names
                invalid_tools = ["none", "manual", "skip", "custom", "placeholder", ""]
                if not step.tool_name or step.tool_name.lower() in invalid_tools:
                    skip_msg = f"Skipping step {i+1} '{step.name}': invalid tool name '{step.tool_name}'"
                    execution_log.append(f"\n⚠ {skip_msg}")
                    warnings.append(skip_msg)
                    log_agent_action("analysis", f"Step {i+1}/{len(plan.steps)} skipped", {
                        "step": step.name,
                        "reason": f"Invalid tool name: {step.tool_name}"
                    })
                    continue

                if is_combined_analysis_tool(step.tool_name):
                    if self._is_pinned_per_sim_tool(step.tool_name, state):
                        pass
                    elif not self._is_combined_hitl_context(state):
                        skip_msg = (
                            f"Skipping step {i+1} '{step.name}': '{step.tool_name}' is a "
                            "cross-simulation tool and cannot run in per-simulation analysis"
                        )
                        execution_log.append(f"\n⚠ {skip_msg}")
                        warnings.append(skip_msg)
                        log_agent_action("analysis", f"Step {i+1}/{len(plan.steps)} skipped", {
                            "step": step.name,
                            "reason": f"Combined-analysis tool not allowed: {step.tool_name}",
                        })
                        continue

                if (
                    retry_n > 0
                    and is_noncritical_analysis_tool(step.tool_name)
                    and self._plot_step_output_present(step, str(analysis_dir))
                ):
                    skip_msg = (
                        f"Skipping step {i+1} '{step.name}': plot output already on disk "
                        f"(analysis retry {retry_n})"
                    )
                    execution_log.append(f"\n⚠ {skip_msg}")
                    warnings.append(skip_msg)
                    continue
                if (
                    not self._is_holo_simulation(state, agent_input)
                    and self._step_requires_bound_ligand(step)
                ):
                    skip_msg = (
                        f"Skipping step {i+1} '{step.name}': "
                        "protein-only simulation has no bound ligand"
                    )
                    execution_log.append(f"\n⚠ {skip_msg}")
                    warnings.append(skip_msg)
                    results[step.name] = {
                        "success": True,
                        "skipped": True,
                        "message": "apo simulation",
                    }
                    continue
                if family_tool_already_done(str(analysis_dir), step.tool_name):
                    skip_msg = (
                        f"Skipping step {i+1} '{step.name}': "
                        f"{step.tool_name} artifacts already on disk"
                    )
                    execution_log.append(f"\n⚠ {skip_msg}")
                    warnings.append(skip_msg)
                    continue
                if self._metric_step_output_present(step, str(analysis_dir)):
                    skip_msg = (
                        f"Skipping step {i+1} '{step.name}': output already on disk"
                    )
                    execution_log.append(f"\n⚠ {skip_msg}")
                    continue
                
                # Log step start
                log_agent_action("analysis", f"Executing step {i+1}/{len(plan.steps)}", {
                    "step": step.name,
                    "tool": step.tool_name,
                    "source": state.get("_analysis_plan_source") or "engine",
                    "topology_file": (step.tool_params or {}).get("topology_file"),
                    "trajectory_file": (step.tool_params or {}).get("trajectory_file"),
                    "hpc_dir": (step.tool_params or {}).get("hpc_dir") or state.get("hpc_dir"),
                })
                
                execution_log.append(f"\n--- Step {i+1}: {step.name} ---")
                execution_log.append(f"Description: {step.description}")
                execution_log.append(f"Tool: {step.tool_name}")
                execution_log.append(f"Reason: {step.reason}")
                
                # Prepare tool parameters
                tool_params = dict(step.tool_params) if step.tool_params else {}

                # Never invent ATP: inject declared ligands or strip silent defaults
                if "ligand_selection" in tool_params or step.tool_name in {
                    "calculate_consensus_pocket_metrics",
                    "define_pocket_mapped_residues",
                    "define_reference_consensus_pocket",
                    "calculate_ligand_axis_angle",
                    "calculate_ligand_pocket_distance",
                }:
                    from agentic.campaign.ligand_context import (
                        apply_ligand_selection,
                        resolve_ligand_selection,
                    )

                    tool_params = apply_ligand_selection(tool_params, state)
                    resolved_lig = resolve_ligand_selection(
                        tool_params.get("ligand_selection"), state
                    )
                    if resolved_lig and not self._is_holo_simulation(state, agent_input):
                        resolved_lig = None
                    if resolved_lig:
                        tool_params["ligand_selection"] = resolved_lig
                    elif step.tool_name in {
                        "calculate_consensus_pocket_metrics",
                        "define_pocket_mapped_residues",
                        "define_reference_consensus_pocket",
                        "calculate_ligand_axis_angle",
                        "calculate_ligand_pocket_distance",
                    }:
                        execution_log.append(
                            f"⚠ Skipped {step.tool_name}: no ligand declared (apo-safe)"
                        )
                        results[step.name] = {
                            "success": True,
                            "skipped": True,
                            "message": "no ligand_selection",
                        }
                        continue

                if is_combined_analysis_tool(step.tool_name):
                    task_text = state.get("hitl_chat_task") or agent_input.user_goal or ""
                    tool_params = self._resolve_combined_tool_sim_dirs(state, task_text, tool_params)
                
                # SECURITY: Sanitize output parameters (LLM may specify full paths)
                tool_params = sanitize_tool_output_params(tool_params)
                
                # ── FORCE-OVERRIDE input file paths ──────────────────────
                # The LLM often invents wrong paths.  We ignore whatever
                # the LLM put in tool_params for input files and inject
                # the real paths from state (which point to working_dir/hpc/
                # or wherever the files actually live).
                #
                # 1. Build a map: canonical type → resolved absolute path
                _state_input_map = self._resolve_input_files(state)
                
                # 2. For every recognised input-param name, override with
                #    the correct path from the map (or scan input dir).
                _INPUT_PARAM_TO_TYPE = {
                    "topology_file": "topology",
                    "topology": "topology",
                    "structure": "topology",
                    "trajectory_file": "trajectory",
                    "trajectory": "trajectory",
                    "traj": "trajectory",
                    "energy_file": "energy",
                    "edr": "energy",
                    "edr_file": "energy",
                }
                
                for param_name, file_type in _INPUT_PARAM_TO_TYPE.items():
                    if param_name in tool_params:
                        resolved = _state_input_map.get(file_type)
                        if resolved:
                            tool_params[param_name] = resolved
                            logger.debug(f"  {param_name}: overridden → {resolved}")
                        else:
                            # Last resort: try resolve_input_file with LLM's filename
                            file_ref = str(tool_params[param_name])
                            resolved_path = self.file_manager.resolve_input_file(
                                file_reference=file_ref,
                                search_stages=["hpc", "simsetup", "preprocess"]
                            )
                            if resolved_path:
                                tool_params[param_name] = resolved_path
                                logger.debug(f"  {param_name}: registry → {resolved_path}")
                            else:
                                logger.warning(f"  {param_name}: could not resolve '{file_ref}'")
                
                # ── Prepend agent directory to output parameters ──────────
                output_param_names = [
                    "output_file", "output_prefix", "plot_file", "figure_path",
                    "csv_file", "dat_file", "save_path", "output_csv", "output_fig"
                ]
                
                for param_name in output_param_names:
                    if param_name in tool_params and tool_params[param_name]:
                        # Convert filename to full path in agent's directory
                        filename = str(tool_params[param_name])
                        full_path = self.file_manager.get_agent_path(filename)
                        tool_params[param_name] = full_path
                        logger.debug(f"  {param_name}: {filename} -> {full_path}")
                
                # CRITICAL: Always use the analysis agent directory for working_dir
                # LLMs may suggest workspace root, but tools must run in analysis subdirectory
                tool_params["working_dir"] = self.file_manager.agent_dir
                logger.debug(f"  Set working_dir to analysis agent directory: {self.file_manager.agent_dir}")

                # Family modular tools need MSA + label from pre_combined.
                tool_params = self._inject_consensus_family_params(
                    step.tool_name, tool_params, state
                )

                # Whole-protein RMSF/DCCM must use Cα — LLM often passes "protein".
                if step.tool_name in ("calculate_rmsf", "calculate_dccm"):
                    from src.analysis.selection_policy import (
                        DEFAULT_CA_SELECTION,
                        ensure_ca_selection,
                    )

                    before = tool_params.get("selection")
                    tool_params["selection"] = ensure_ca_selection(
                        before or DEFAULT_CA_SELECTION
                    )
                    if tool_params.get("align_selection"):
                        tool_params["align_selection"] = ensure_ca_selection(
                            tool_params["align_selection"]
                        )
                    if before != tool_params["selection"]:
                        logger.info(
                            "Forced Cα selection for %s: %r → %r",
                            step.tool_name,
                            before,
                            tool_params["selection"],
                        )

                if step.tool_name == "plot_md_data":
                    rewritten = self._retarget_dssp_plot_inputs(
                        tool_params, str(analysis_dir)
                    )
                    if rewritten == "skip":
                        execution_log.append(
                            f"⚠ Skipped {step.name}: DSSP already wrote its own figure"
                        )
                        results[step.name] = {
                            "success": True,
                            "skipped": True,
                            "message": "DSSP heatmap already on disk",
                        }
                        continue
                    tool_params = rewritten

                # Resolve list-type file parameters (e.g. data_files for plot tools).
                # LLMs often pass wrong absolute paths; normalise to agent_dir/<basename>.
                list_file_params = ["data_files", "input_files", "file_list"]
                for lp in list_file_params:
                    if lp in tool_params and isinstance(tool_params[lp], list):
                        resolved_list = []
                        agent_dir = str(self.file_manager.agent_dir)
                        for fref in tool_params[lp]:
                            fref = str(fref)
                            basename = os.path.basename(fref)
                            # Multipanel/plot must not ingest *_stats.csv tables
                            if step.tool_name in {
                                "plot_multipanel",
                                "plot_md_multipanel",
                                "plot_md_data",
                            }:
                                stem_l = Path(basename).stem.lower()
                                if stem_l.endswith("_stats") or "classification_" in basename.lower():
                                    logger.info(
                                        "Dropping summary table %s from %s data_files",
                                        basename,
                                        step.tool_name,
                                    )
                                    continue
                            resolved_list.append(os.path.join(agent_dir, basename))
                        logger.debug(f"  Resolved {lp}: {tool_params[lp]} -> {resolved_list}")
                        tool_params[lp] = resolved_list
                
                execution_log.append(f"Parameters: {json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in tool_params.items()}, indent=2)}")
                
                # Use batched trajectory result when available
                if i in batch_results and batch_results[i].get("success"):
                    result = dict(batch_results[i])
                    # Batch writers often return basename-only paths; resolve under analysis_dir
                    agent_dir = str(self.file_manager.agent_dir)
                    for file_key in (
                        "output_file", "plot_file", "csv_file", "heatmap_file",
                        "timeseries_file", "figure_path",
                    ):
                        fref = result.get(file_key)
                        if not fref:
                            continue
                        fref_s = str(fref)
                        if not os.path.isabs(fref_s):
                            result[file_key] = os.path.join(agent_dir, os.path.basename(fref_s))
                    execution_log.append(
                        f"✓ Success (trajectory batch): {result.get('message', 'Step completed')}"
                    )
                else:
                    # Execute tool with retry logic (or retry after batch failure)
                    retry_count = 0
                    result = None
                    
                    while retry_count <= max_retries:
                        result = self.tool_executor.execute(step.tool_name, **tool_params)
                        
                        if result.get("success"):
                            break
                        
                        retry_count += 1
                        if retry_count <= max_retries:
                            execution_log.append(f"⚠ Retry {retry_count}/{max_retries}: {result.get('error')}")
                        else:
                            execution_log.append(f"✗ Max retries ({max_retries}) exceeded")
                
                # Process results
                if result and result.get("success"):
                    execution_log.append(f"✓ Success: {result.get('message', 'Step completed')}")
                    
                    # Store analysis results
                    analysis_name = step.name.lower().replace(" ", "_")
                    results[analysis_name] = result
                    
                    # Track and register generated files using SecureFileManager
                    file_keys = ["output_file", "plot_file", "csv_file", "heatmap_file", 
                                "timeseries_file", "figure_path"]
                    
                    for file_key in file_keys:
                        if file_key in result and result[file_key]:
                            file_path = result[file_key]
                            
                            # Register using file manager (automatic tracking)
                            file_type = self._classify_file_type(file_key, file_path)
                            self.file_manager.register_external_file(
                                file_path=file_path,
                                file_type=file_type,
                                description=f"{step.name}: {step.description}"
                            )
                            
                            generated_files[file_path] = step.description
                            log_file_operation("analysis", "create", file_path, True)
                    
                    # Also handle output_files dict (common in tools like DSSP)
                    if "output_files" in result and isinstance(result["output_files"], dict):
                        for out_name, out_path in result["output_files"].items():
                            if out_path and Path(out_path).exists():
                                file_type = self._classify_file_type(out_name, out_path)
                                self.file_manager.register_external_file(
                                    file_path=out_path,
                                    file_type=file_type,
                                    description=f"{step.name} {out_name}"
                                )
                                generated_files[out_path] = f"{step.name} {out_name}"
                                log_file_operation("analysis", "create", out_path, True)
                    
                    log_agent_action("analysis", f"Step {i+1}/{len(plan.steps)} completed", {
                        "step": step.name,
                        "tool": step.tool_name,
                        "status": "✅ SUCCESS"
                    })
                    
                    if result.get("warning"):
                        warnings.append(f"{step.name}: {result['warning']}")
                else:
                    error_msg = result.get('error', 'Failed to execute') if result else 'Tool execution failed'
                    execution_log.append(f"✗ Failed after {retry_count} attempts: {error_msg}")
                    if is_noncritical_analysis_tool(step.tool_name):
                        warnings.append(f"{step.name}: {error_msg}")
                    elif (
                        "matched 0 atoms" in error_msg.lower()
                        and not self._is_holo_simulation(state, agent_input)
                    ):
                        warnings.append(f"{step.name}: {error_msg} (apo, ignored)")
                    else:
                        issues.append(f"{step.name}: {error_msg}")
                    
                    log_agent_action("analysis", f"Step {i+1}/{len(plan.steps)} failed", {
                        "step": step.name,
                        "tool": step.tool_name,
                        "status": "❌ FAILED",
                        "error": error_msg
                    })
                    
                    if fail_fast:
                        execution_log.append("⚠ Stopping execution (fail_fast enabled)")
                        break

            # Safety net: if trajectory batch wrote .dat but plot steps were skipped
            # or filtered, materialize PNGs from files already on disk.
            # Always run — including analysis retries — so reports get figures.
            self._ensure_plots_for_dat_on_disk(
                str(analysis_dir),
                execution_log,
                generated_files,
                warnings,
            )
            
            execution_log_str = "\n".join(execution_log)
            
            # Generate execution report
            report = self._generate_execution_report(plan, results, issues, warnings)

            spec = spec_from_state(state)
            wd = state.get("working_directory") or ""
            artifact_ok = False
            # Combined / post-combined: judge overlays at base/analysis, not per-sim .dat
            if self._is_combined_analysis_phase(state) or state.get("hitl_view_combined"):
                try:
                    from agentic.multi_sim_progress import post_combined_artifacts_ok

                    adir = (
                        state.get("analysis_dir")
                        or state.get("analysis_directory")
                        or (str(Path(wd) / "analysis") if wd else "")
                    )
                    labels = []
                    for sp in state.get("sim_prompts") or []:
                        lab = sp.get("label")
                        if lab:
                            labels.append(str(lab))
                    artifact_ok = post_combined_artifacts_ok(
                        adir,
                        expected_labels=labels or None,
                        family_modular=bool(
                            self._is_combined_analysis_phase(state)
                            and spec
                            and spec.family_modular
                        ),
                    )
                except Exception:
                    artifact_ok = False
            elif spec and spec.family_modular and wd:
                artifact_ok = bool(per_sim_science_complete(wd, spec=spec).get("ok"))
            elif wd:
                try:
                    from agentic.multi_sim_progress import per_sim_analysis_done_on_disk

                    artifact_ok = per_sim_analysis_done_on_disk(wd)
                except Exception:
                    artifact_ok = False
            if artifact_ok and issues:
                warnings.append(
                    "Required analysis artifacts are on disk; "
                    "remaining step errors are non-blocking"
                )
            # When combined overlays exist, demote remaining issues to warnings
            # so optional table/plot steps cannot fail the whole combined stage.
            critical = list(issues)
            if artifact_ok and (
                self._is_combined_analysis_phase(state) or state.get("hitl_view_combined")
            ):
                for msg in issues:
                    warnings.append(f"non-blocking after overlays: {msg}")
                critical = []
            success = decide_analysis_success(
                artifact_ok=artifact_ok, critical_issues=critical
            )
            
            return AnalysisExecutionResult(
                success=success,
                analyses_completed=[step.name for step in plan.steps if step.name.lower().replace(" ", "_") in results],
                results=results,
                output_directory=analysis_dir,
                report=report,
                issues=issues,
                warnings=warnings,
                generated_files=generated_files,
                execution_log=execution_log_str
            )
            
        except Exception as e:
            try:
                self._ensure_plots_for_dat_on_disk(
                    str(analysis_dir),
                    execution_log,
                    generated_files,
                    warnings,
                )
            except Exception:
                logger.debug("Post-hoc plots after analysis error skipped", exc_info=True)
            return AnalysisExecutionResult(
                success=False,
                output_directory=analysis_dir,
                report=f"Analysis failed: {str(e)}",
                issues=[str(e)],
                warnings=warnings,
                generated_files=generated_files,
                execution_log="\n".join(execution_log)
            )

    def _generate_execution_report(self, plan: AnalysisPlan, results: Dict[str, Any],
                                   issues: list, warnings: list) -> str:
        """Generate human-readable execution report"""
        report_lines = [
            "=" * 80,
            "MD TRAJECTORY ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Plan Overview: {plan.overview}",
            f"Total Steps: {len(plan.steps)}",
            f"Completed: {len(results)}",
            f"Issues: {len(issues)}",
            f"Warnings: {len(warnings)}",
            "",
            "=" * 80,
            "RESULTS SUMMARY",
            "=" * 80,
        ]
        
        for analysis_name, result in results.items():
            report_lines.append(f"\n{analysis_name.upper()}:")
            if "message" in result:
                report_lines.append(f"  {result['message']}")
            
            # Add specific metrics based on analysis type
            if "rmsd" in analysis_name.lower() and "mean_rmsd" in result:
                report_lines.append(f"  Mean RMSD: {result['mean_rmsd']:.2f} Å")
            elif "rmsf" in analysis_name.lower() and "mean_rmsf" in result:
                report_lines.append(f"  Mean RMSF: {result['mean_rmsf']:.2f} Å")
            elif "gyration" in analysis_name.lower() and "mean_rg" in result:
                report_lines.append(f"  Mean Rg: {result['mean_rg']:.2f} Å")
        
        if issues:
            report_lines.extend([
                "",
                "=" * 80,
                "ISSUES ENCOUNTERED",
                "=" * 80,
            ])
            for issue in issues:
                report_lines.append(f"  • {issue}")
        
        if warnings:
            report_lines.extend([
                "",
                "=" * 80,
                "WARNINGS",
                "=" * 80,
            ])
            for warning in warnings:
                report_lines.append(f"  • {warning}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)

    # ------------------------------------------------------------------
    # Input-path resolution helpers
    # ------------------------------------------------------------------

    def _resolve_input_files(self, state: MDState) -> Dict[str, str]:
        """Resolve canonical input file paths for the analysis agent.

        Returns a dict keyed by file type ("topology", "trajectory", "energy")
        with absolute paths that actually exist on disk.  Resolution order:

        1. State fields set by upstream agents (``topology``, ``trajectory_path``,
           ``energy_file``).
        2. Files copied into the analysis directory by
           ``_copy_files_from_hpc_secure``.
        3. Scan the hardcoded input directory (``working_dir/hpc/``) for common
           extensions.
        """
        from ..state import AGENT_IO_MAP

        resolved: Dict[str, str] = {}
        working_dir = state.get("working_directory", "working_dir")

        # Hardcoded input directory for this agent
        input_subdir = AGENT_IO_MAP.get("analysis", {}).get("input_dir", "hpc")
        input_dir = str(Path(working_dir) / input_subdir) if input_subdir else working_dir
        # Multi-rep: prefer first nested hpc/repXX for planner path resolution
        try:
            from src.analysis.replicate_paths import discover_hpc_rep_dirs

            if _effective_rep_num_for_state(state, working_dir) > 1:
                reps = discover_hpc_rep_dirs(working_dir)
                nested = [p for p in reps if p.name.startswith("rep")]
                if nested:
                    input_dir = str(nested[0])
                elif reps:
                    input_dir = str(reps[0])
        except Exception:
            pass

        # --- 1. Try state fields first (set by HPC / simsetup agent) --------
        # Do NOT use cleaned_pdb / preprocess coordinates as analysis topology —
        # MDA needs the production system (md.tpr) matching the trajectory.
        _STATE_KEYS = {
            "topology":   ["tpr_file", "topology"],
            "trajectory":  ["trajectory_path", "trajectory"],
            "energy":      ["energy_file"],
        }
        for ftype, keys in _STATE_KEYS.items():
            for key in keys:
                val = state.get(key)
                if val and Path(val).is_file():
                    resolved[ftype] = str(Path(val).resolve())
                    logger.debug(f"_resolve_input_files: {ftype} from state['{key}'] → {resolved[ftype]}")
                    break

        # --- 2/3. Prefer MDA-compatible full-system topologies ----------------
        # Alphabetical glob("*.gro") would pick ATP.gro (ligand-only, ~43 atoms)
        # and break RMSD/RMSF against mdWrap.xtc. Prefer md.tpr first.
        _TOPO_PREFERRED = (
            "md.tpr",
            "md.gro",
            "system.gro",
            "solvated.gro",
            "complex.gro",
            "processed.gro",
        )
        _TOPO_SKIP = {
            "ATP.gro",
            "MG.gro",
            "ligand_GMX.gro",
            "protein.gro",
            "protein_processed.gro",
            "topol.top",
            "topology.top",
        }
        _EXT_MAP = {
            "topology":   [".tpr", ".gro", ".pdb"],
            "trajectory":  [".xtc", ".trr", ".dcd", ".nc"],
            "energy":      [".edr", ".ene"],
        }

        def _pick_topology(search_dir: Path) -> Optional[str]:
            if not search_dir.is_dir():
                return None
            for name in _TOPO_PREFERRED:
                cand = search_dir / name
                if cand.is_file():
                    return str(cand.resolve())
            for ext in (".tpr", ".gro", ".pdb"):
                for cand in sorted(search_dir.glob(f"*{ext}")):
                    if cand.name in _TOPO_SKIP:
                        continue
                    return str(cand.resolve())
            return None

        # Reject unsuitable / preprocess-only topologies from state
        if "topology" in resolved:
            topo_path = Path(resolved["topology"])
            topo_name = topo_path.name
            suf = topo_path.suffix.lower()
            if (
                topo_name in _TOPO_SKIP
                or suf == ".top"
                or suf == ".pdb"
                or "preprocess" in str(topo_path).replace("\\", "/").lower()
            ):
                logger.warning(
                    "_resolve_input_files: rejecting unsuitable topology %s",
                    resolved["topology"],
                )
                del resolved["topology"]

        analysis_dir = state.get("analysis_dir") or self.file_manager.agent_dir

        # Prefer md.tpr from hpc/analysis even when a weaker state topo exists
        for search in (Path(input_dir), Path(analysis_dir)):
            picked = _pick_topology(search)
            if not picked:
                continue
            if "topology" not in resolved or Path(picked).suffix.lower() == ".tpr":
                if (
                    "topology" not in resolved
                    or Path(resolved["topology"]).suffix.lower() != ".tpr"
                    or Path(picked).name == "md.tpr"
                ):
                    if Path(picked).suffix.lower() == ".tpr" or "topology" not in resolved:
                        resolved["topology"] = picked
                        logger.debug(
                            "_resolve_input_files: topology from %s → %s",
                            search,
                            picked,
                        )
                        if Path(picked).suffix.lower() == ".tpr":
                            break

        for ftype, exts in _EXT_MAP.items():
            if ftype in resolved:
                continue
            if ftype == "topology":
                picked = _pick_topology(Path(analysis_dir))
                if picked:
                    resolved[ftype] = picked
                    logger.debug(f"_resolve_input_files: {ftype} from analysis_dir → {picked}")
                continue
            for ext in exts:
                candidates = sorted(Path(analysis_dir).glob(f"*{ext}"))
                if candidates:
                    resolved[ftype] = str(candidates[0].resolve())
                    logger.debug(f"_resolve_input_files: {ftype} from analysis_dir → {resolved[ftype]}")
                    break

        # --- 3. Scan the hardcoded input directory (hpc/) --------------------
        for ftype, exts in _EXT_MAP.items():
            if ftype in resolved:
                continue
            if ftype == "topology":
                picked = _pick_topology(Path(input_dir))
                if picked:
                    resolved[ftype] = picked
                    logger.debug(f"_resolve_input_files: {ftype} from input_dir({input_dir}) → {picked}")
                continue
            for ext in exts:
                candidates = sorted(Path(input_dir).glob(f"*{ext}"))
                if candidates:
                    resolved[ftype] = str(candidates[0].resolve())
                    logger.debug(f"_resolve_input_files: {ftype} from input_dir({input_dir}) → {resolved[ftype]}")
                    break

        if not resolved:
            logger.warning("_resolve_input_files: no input files resolved")
        else:
            logger.info(f"_resolve_input_files: resolved {list(resolved.keys())}")

        return resolved

    def _hpc_inputs_ready_for_analysis(self, state: MDState) -> tuple:
        """Return (ready, reason). Require real trajectory + production topology."""
        try:
            resolved = self._resolve_input_files(state)
        except Exception as exc:
            return False, f"Could not resolve HPC inputs: {exc}"
        traj = resolved.get("trajectory")
        topo = resolved.get("topology")
        if not traj or not Path(traj).is_file():
            return (
                False,
                "No HPC trajectory (.xtc/.trr) found — run HPC (or restore hpc/) before analysis",
            )
        if not topo or not Path(topo).is_file():
            return (
                False,
                "No production topology (md.tpr) found under hpc/ — cannot analyze trajectory",
            )
        if Path(topo).suffix.lower() == ".pdb":
            return (
                False,
                f"Topology is preprocess PDB ({Path(topo).name}); need md.tpr matching the trajectory",
            )
        return True, ""

    def _classify_file_type(self, key_or_extension: str, file_path: str = "") -> str:
        """Classify file type based on parameter name or extension."""
        
        key_lower = key_or_extension.lower()
        path_lower = file_path.lower()
        
        # Classify by parameter name/key
        if "plot" in key_lower or "figure" in key_lower or "heatmap" in key_lower or "timeseries" in key_lower:
            return "plot"
        elif "trajectory" in key_lower or "traj" in key_lower:
            return "trajectory"
        elif "topology" in key_lower or "structure" in key_lower:
            return "topology"
        elif "energy" in key_lower:
            return "energy"
        
        # Classify by file extension
        if path_lower.endswith((".png", ".pdf", ".svg", ".jpg")):
            return "plot"
        elif path_lower.endswith(".xtc") or path_lower.endswith(".trr"):
            return "trajectory"
        elif path_lower.endswith((".gro", ".pdb", ".tpr")):
            return "topology"
        elif path_lower.endswith(".edr"):
            return "energy"
        elif path_lower.endswith((".csv", ".dat", ".xvg")):
            return "data"
        else:
            return "output"

    # ========== Legacy methods (kept for backward compatibility) ==========

    def _create_analysis_plan(self, state: MDState) -> Dict[str, Any]:
        """Legacy method for backward compatibility"""
        traj = state.get("trajectory_path")
        coords = state.get("coordinates")
        request = state.get("analysis_request") or "General stability and energy analysis"
        prompt = f"""
You are an expert MD analysis agent. Draft a practical analysis plan.

INPUTS:
- Trajectory: {traj}
- Coordinates: {coords}
- Request: {request}
- Preferred metrics: {self.config.get('metrics')}

RESPONSE FORMAT:
Metrics: [comma separated list]
Steps:
- Step 1: description
- Step 2: description
Outputs: [files/plots]
Notes: [assumptions/fallbacks]
"""
        resp = self.llm.prompt(prompt, system="You plan MD analysis tasks and ensure reproducible outputs.")
        return {"raw": resp}

    def _execute_basic_analysis(self, state: MDState, plan: Dict[str, Any]) -> None:
        """Legacy method for backward compatibility"""
        out_dir = self.config.get("output_dir", "./working_dir/analysis")
        os.makedirs(out_dir, exist_ok=True)
        traj = state.get("trajectory_path")
        coords = state.get("coordinates") or state.get("topology")

        results = state.setdefault("analysis_results", {})
        if not traj or not os.path.exists(traj):
            results["summary"] = "No trajectory available; stored analysis plan only."
            state["warnings"].append("Trajectory missing; analysis not executed")
            return

        # Minimal, dependency-light analysis placeholder
        if not HAS_MDA or not self.config.get("use_mdanalysis", True):
            results["summary"] = "MDAnalysis unavailable; minimal placeholder analysis done."
            results["metrics"] = {"frames": self._count_lines(traj)}
            return

        # Basic RMSD using MDAnalysis
        try:
            import numpy as np
            u = mda.Universe(coords, traj) if coords else mda.Universe(traj)
            ref = u.select_atoms("protein").positions.copy() if u.atoms.n_atoms > 0 else None
            rmsds = []
            for ts in u.trajectory:
                sel = u.select_atoms("protein")
                if sel.n_atoms == 0 or ref is None:
                    continue
                pos = sel.positions
                diff = pos - ref
                rmsd = (np.sqrt((diff * diff).sum(axis=1).mean()))
                rmsds.append(float(rmsd))
            results["metrics"] = {"rmsd_mean": float(np.mean(rmsds)) if rmsds else None,
                                   "rmsd_max": float(np.max(rmsds)) if rmsds else None,
                                   "frames": len(rmsds)}
            results["summary"] = "Computed simple RMSD over protein selection."
        except Exception as e:
            logger.exception("Error during MDAnalysis RMSD")
            state["errors"].append(f"RMSD analysis failed: {e}")

    def _count_lines(self, path: str) -> int:
        """Legacy helper for counting lines in a file"""
        try:
            with open(path, "rb") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
