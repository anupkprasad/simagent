"""
Reporter Agent - Scientific Report Generation

LLM-driven agent that reads analysis summaries, searches literature,
and generates comprehensive HTML reports with scientific context.
"""
import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..state import MDState
from ..llm import LLMClient
from ..utils import (
    log_agent_start, log_llm_interaction, log_agent_action,
    log_agent_completion, log_error, SecureFileManager
)
from .schemas import (
    ReporterPlan, ReporterStep, ReporterResult,
    ReporterAgentInput, ReporterAgentOutput, ReportType
)
from .tools import ReporterToolExecutor, get_tool_metadata

logger = logging.getLogger(__name__)

# Glob patterns for combined cross-sim plots under {base}/analysis/
_COMBINED_ANALYSIS_PLOT_GLOBS = (
    "*overlay*.png",
    "dccm_comparison.png",
    "dccm_comparison_*.png",
    "dccm_*comparison*.png",
    "*dccm*diff*heatmap*.png",
    "*_dccm_diff_heatmap.png",
    "dccm_apo_holo*panels*.png",
    "rmsf_segment*.png",
    "com_distance*.png",
    "dssp_comparison.png",
    "dssp_activation_loop_*.png",
    "classification_*.png",
    "*_by_cluster.png",
    "sequence_phylo_tree.png",
    "structure_phylo_tree.png",
)


def resolve_combined_sim_context(state: Dict[str, Any]) -> tuple:
    """Return (sim_dirs, labels) for combined report/analysis (healthy sims only)."""
    from agentic.utils.sim_health import healthy_sim_labels

    healthy = set(healthy_sim_labels(state))

    completed = state.get("completed_sim_states") or []
    if completed:
        pairs = [
            (s["working_directory"], s.get("label", f"sim_{i}"))
            for i, s in enumerate(completed)
            if s.get("working_directory")
            and (not healthy or str(s.get("label", f"sim_{i}")) in healthy)
        ]
        if pairs:
            return [p[0] for p in pairs], [p[1] for p in pairs]

    sim_prompts = state.get("sim_prompts") or []
    base = state.get("multi_sim_base_dir") or state.get("working_directory", "")
    sim_dirs = list(state.get("sim_working_dirs") or [])
    if not sim_dirs and sim_prompts:
        sim_dirs = [
            sp.get("working_dir") or str(Path(base) / sp.get("label", f"sim_{i}"))
            for i, sp in enumerate(sim_prompts)
        ]
    labels = list(state.get("sim_labels") or [])
    if not labels and sim_prompts:
        labels = [sp.get("label", f"sim_{i}") for i, sp in enumerate(sim_prompts)]
    if healthy and labels and sim_dirs and len(labels) == len(sim_dirs):
        filtered = [(d, lab) for d, lab in zip(sim_dirs, labels) if lab in healthy]
        if filtered:
            return [p[0] for p in filtered], [p[1] for p in filtered]
    return sim_dirs, labels


def collect_combined_overlay_plots(
    state: Dict[str, Any],
    analysis_dir: str,
    combined_info: Dict[str, Any],
) -> List[str]:
    """
    Merge overlay plot paths from state, combined analysis results, and on-disk files.

    HITL report regeneration often runs after state compaction dropped plot paths
    or when completed_sim_states was not persisted — disk discovery keeps reports accurate.
    """
    plots: List[str] = []
    seen: set = set()

    def _add(path: Any) -> None:
        if not path:
            return
        p = str(path)
        if p in seen:
            return
        if Path(p).is_file():
            seen.add(p)
            plots.append(p)

    for key in (
        "overlay_plots",
        "dccm_plots",
        "rmsf_segment_plots",
        "rmsf_apo_holo_plots",
        "dssp_plots",
    ):
        for item in combined_info.get(key) or []:
            _add(item)
    _add(combined_info.get("com_distance_plot"))
    for item in state.get("figures") or []:
        _add(item)

    adir = Path(analysis_dir)
    if adir.is_dir():
        for pattern in _COMBINED_ANALYSIS_PLOT_GLOBS:
            for p in sorted(adir.glob(pattern)):
                _add(p)

    return plots


class ReporterAgent:
    """LLM-driven reporter agent that generates scientific reports"""
    
    def __init__(self, llm_client: LLMClient, config_path: Optional[str] = None):
        if llm_client is None:
            raise ValueError("llm_client is required")
        self.llm = llm_client
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config = self._load_config()
        self.tool_executor = None  # Initialized per execution
        self.file_manager = None  # Initialized per execution
        logger.info("Reporter Agent initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load reporter configuration"""
        import yaml
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        logger.warning(f"Reporter config {self.config_path} not found; using defaults")
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "agent": {
                "working_directory": "working_dir/reporter",
                "input_from": "working_dir/analysis"
            },
            "report": {
                "default_type": "comprehensive"
            },
            "literature": {
                "pubmed": {
                    "enabled": True,
                    "max_results": 10
                }
            }
        }
    
    def reporter_node(self, state: MDState) -> MDState:
        """
        Main reporter node - entry point from workflow.

        In multi-sim combined_analysis phase: generates a cross-simulation
        comparison HTML report from combined analysis results.
        Otherwise: runs the regular per-simulation LLM-guided reporter.
        """
        import sys
        import traceback

        # ── Combined multi-sim report ─────────────────────────────────────
        # Support both initial combined analysis pass and resumed reporter pass.
        if state.get("multi_sim_phase") in {"combined_analysis", "combined_reporter"}:
            return self._run_combined_report_via_llm(state)

        # ── Regular per-sim report ────────────────────────────────────────
        # Extract input
        execution_plan = state.get("execution_plan") or {}
        has_planner_instructions = execution_plan.get("format") == "natural_language"
        
        input_summary = {
            "analysis_directory": state.get("working_directory", "working_dir") + "/analysis",
            "report_type": state.get("report_type", "comprehensive")
        }
        
        if has_planner_instructions:
            # Get reporter instructions from planner
            reporter_section = state.get("reporter_instructions")
            
            if not reporter_section:
                # Fallback: extract from full plan
                full_plan = execution_plan.get("full_plan", "")
                reporter_section = self._extract_agent_instructions(full_plan, "Reporter Agent")
            
            if reporter_section:
                input_summary["planner_instructions"] = reporter_section
            else:
                input_summary["planner_instructions"] = "[Natural language plan from planner]"
        
        log_agent_start("reporter", "Scientific Report Generation", input_summary)
        
        try:
            # Initialize secure file manager
            base_working_dir = state.get("working_directory", "working_dir")
            file_registry = state.get("file_registry", {})
            
            self.file_manager = SecureFileManager(
                working_dir=base_working_dir,
                agent_name="reporter",
                file_registry=file_registry
            )
            
            logger.info(f"Reporter agent directory: {self.file_manager.agent_dir}")
            
            # Initialize tool executor
            self.tool_executor = ReporterToolExecutor(
                config={"working_directory": self.file_manager.agent_dir}
            )
            
            # Prepare agent input
            agent_input = self._prepare_agent_input(state)
            
            # Run reporter workflow
            agent_output = self._run_reporter_workflow(agent_input, state)
            
            # Update state with results
            self._update_state(state, agent_output)
            
            # Route to reporter checkpoint so the human can inspect results.
            state["next_node"] = "human_reporter_check"
            
            success = agent_output.success and len(agent_output.errors) == 0
            log_agent_completion("reporter", "Scientific Report Generation", state, success)
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Reporter agent failed: {type(e).__name__}: {e}"
            logger.error(error_msg)
            logger.error(f"Traceback: {tb}")
            
            log_error("reporter_agent.reporter_node", e, {
                "state": str(state),
                "traceback": tb
            })
            state["errors"].append(f"Reporter error: {str(e)}")
            state["next_node"] = "human_reporter_check"
        
        return state

    # ── Combined multi-sim report ─────────────────────────────────────────

    @staticmethod
    def _generate_dssp_comparison_chart(
        sim_dirs: List[str],
        labels: List[str],
        output_dir: str,
    ) -> Optional[str]:
        """Generate a grouped bar chart comparing helix/sheet/coil% across sims.

        Reads DSSP statistics from each sim's analysis_summary.jsonl.
        Returns the absolute path to the saved PNG, or None if data is missing.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            from src.analysis.summary_logger import read_summary_file
        except ImportError as exc:
            logger.warning(f"_generate_dssp_comparison_chart: missing dependency {exc}")
            return None

        helix_vals, sheet_vals, coil_vals, plot_labels = [], [], [], []

        _DSSP_TYPES = {"dssp", "dssp_secondarystructure", "secondary_structure",
                       "secondarystructure"}
        _KEY_HELIX = ("avg_helix_percent", "helix_percent", "helix", "percent_helix")
        _KEY_SHEET = ("avg_sheet_percent", "sheet_percent", "sheet", "beta_sheet",
                      "percent_sheet", "beta_percent")
        _KEY_COIL  = ("avg_coil_percent",  "coil_percent",  "coil", "percent_coil")

        def _pick(stats: dict, keys: tuple):
            for k in keys:
                v = stats.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            return None

        for sim_dir, label in zip(sim_dirs, labels):
            analysis_dir = Path(sim_dir) / "analysis"
            try:
                records = read_summary_file(str(analysis_dir))
            except Exception:
                records = []
                jsonl = analysis_dir / "analysis_summary.jsonl"
                if jsonl.exists():
                    import json as _json
                    for line in jsonl.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("---"):
                            try:
                                records.append(_json.loads(line))
                            except Exception:
                                pass

            for rec in records:
                atype = rec.get("analysis_type", "").lower().replace(" ", "_")
                if any(t in atype for t in _DSSP_TYPES):
                    stats = rec.get("statistics", {})
                    h = _pick(stats, _KEY_HELIX)
                    s = _pick(stats, _KEY_SHEET)
                    c = _pick(stats, _KEY_COIL)
                    if h is not None and s is not None:
                        helix_vals.append(h)
                        sheet_vals.append(s)
                        coil_vals.append(c if c is not None else 100.0 - h - s)
                        plot_labels.append(label)
                        break   # one DSSP record per sim is enough

        if len(plot_labels) < 2:
            logger.info("_generate_dssp_comparison_chart: not enough DSSP data (need ≥2 sims)")
            return None

        x = np.arange(len(plot_labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(6, len(plot_labels) * 1.4), 5), dpi=120)
        ax.bar(x - width, helix_vals, width, label="α-Helix", color="#e74c3c", alpha=0.85)
        ax.bar(x,         sheet_vals, width, label="β-Sheet",  color="#3498db", alpha=0.85)
        ax.bar(x + width, coil_vals,  width, label="Coil/Loop", color="#95a5a6", alpha=0.85)

        ax.set_xlabel("Simulation", fontsize=12)
        ax.set_ylabel("Secondary Structure Content (%)", fontsize=12)
        ax.set_title("Secondary Structure Comparison Across Simulations", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_labels, rotation=20, ha="right", fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(max(helix_vals), max(sheet_vals), max(coil_vals)) * 1.2)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()

        out_path = str(Path(output_dir) / "dssp_comparison.png")
        try:
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"DSSP comparison chart saved → {out_path}")
            return out_path
        except Exception as exc:
            logger.warning(f"Could not save DSSP comparison chart: {exc}")
            plt.close(fig)
            return None

    def _build_combined_reporter_planning_prompt(
        self,
        *,
        user_goal: str,
        enriched_prompt: str,
        sim_dirs: List[str],
        labels: List[str],
        overlay_plots: List[str],
        analysis_dir: str,
        reporter_dir: str,
        tools_str: str,
    ) -> str:
        sim_lines = "\n".join(f"  - {lab}: {d}" for lab, d in zip(labels, sim_dirs))
        plot_lines = "\n".join(f"  - {Path(p).name}" for p in overlay_plots[:40]) or "  (none yet)"
        return f"""You are the Reporter Agent in COMBINED multi-simulation mode.

**Original study goal:**
{user_goal or '(not provided)'}

**Enriched / master context:**
{(enriched_prompt or '')[:3000] or '(not provided)'}

**Simulations (use these exact paths for generate_combined_html_report):**
{sim_lines or '  (none)'}

**Combined analysis directory:** {analysis_dir}
**Reporter output directory:** {reporter_dir}
**Available overlay / comparison plots:**
{plot_lines}

**Available Tools:**
{tools_str}

**COMBINED REPORTER RULES:**
- You MUST include a step that calls `generate_combined_html_report` (not generate_html_report).
- Pass sim_dirs and labels exactly as listed above; output_file must be "combined_report.html".
- Use working_dir="." (reporter output dir is applied automatically).
- Do NOT invent classification/clustering discussion unless those plots are listed above.
- Literature search tools are optional — the framework also runs literature collection automatically.
- overview MUST be a single string.

Output as JSON:
{{
  "reasoning": "How you will build the combined report",
  "overview": "High-level summary",
  "steps": [
    {{
      "name": "step name",
      "description": "what it does",
      "tool_name": "tool to call",
      "tool_params": {{"param": "value"}},
      "reason": "why"
    }}
  ],
  "report_focus": ["theme1", "theme2"],
  "literature_queries": [],
  "estimated_complexity": "medium"
}}
"""

    def _create_combined_reporter_plan_llm(
        self,
        *,
        user_goal: str,
        enriched_prompt: str,
        sim_dirs: List[str],
        labels: List[str],
        overlay_plots: List[str],
        analysis_dir: str,
        reporter_dir: str,
    ) -> Optional[ReporterPlan]:
        """LLM plan for combined report with reporter tools exposed."""
        tool_metadata = get_tool_metadata()
        tools_list = []
        for tool_name, tool_info in tool_metadata.items():
            desc = tool_info.get("description", "")
            args = tool_info.get("args") or tool_info.get("parameters") or {}
            tools_list.append(f"→ {tool_name}: {desc}")
            if isinstance(args, dict) and args:
                # Prefer schema properties when present
                props = args.get("properties") if "properties" in args else args
                if isinstance(props, dict) and props:
                    tools_list.append(
                        "    params: " + ", ".join(sorted(props.keys())[:20])
                    )
        tools_str = "\n".join(tools_list)
        prompt = self._build_combined_reporter_planning_prompt(
            user_goal=user_goal,
            enriched_prompt=enriched_prompt,
            sim_dirs=sim_dirs,
            labels=labels,
            overlay_plots=overlay_plots,
            analysis_dir=analysis_dir,
            reporter_dir=reporter_dir,
            tools_str=tools_str,
        )
        try:
            response = self.llm.prompt(prompt, temperature=0.2, max_tokens=4096)
            log_llm_interaction(
                agent_name="reporter.combined_planning",
                prompt=prompt,
                response=response,
                is_mock=not self.llm.available,
            )
            plan_dict = self._extract_plan_json(response)
            if plan_dict is None:
                return None
            steps = [
                ReporterStep(
                    name=step.get("name", "unknown"),
                    description=step.get("description", ""),
                    tool_name=step.get("tool_name", ""),
                    tool_params=step.get("tool_params", {}) or {},
                    reason=step.get("reason", ""),
                )
                for step in plan_dict.get("steps", [])
            ]
            if not any(s.tool_name == "generate_combined_html_report" for s in steps):
                steps.append(
                    ReporterStep(
                        name="Generate combined HTML report",
                        description="Build cross-simulation comparison HTML report",
                        tool_name="generate_combined_html_report",
                        tool_params={
                            "sim_dirs": sim_dirs,
                            "labels": labels,
                            "overlay_plots": overlay_plots,
                            "output_file": "combined_report.html",
                        },
                        reason="Required combined report deliverable",
                    )
                )
            return ReporterPlan(
                reasoning=plan_dict.get("reasoning", "Combined LLM report plan"),
                overview=plan_dict.get("overview", "Combined multi-simulation report"),
                steps=steps,
                report_focus=plan_dict.get("report_focus", []),
                literature_queries=plan_dict.get("literature_queries", []),
                estimated_complexity=plan_dict.get("estimated_complexity", "medium"),
            )
        except Exception as exc:
            logger.warning("Combined reporter LLM planning failed: %s", exc)
            return None

    def _run_combined_report_via_llm(self, state: MDState) -> MDState:
        """
        Combined reporter with LLM planning (tools exposed) then execution.

        Literature review / final impression remain LLM-assisted; HTML generation
        is planned as generate_combined_html_report. Falls back to the prior
        deterministic combined report path on planning failure.
        """
        from .tools import generate_combined_html_report
        from src.reporter.combined_reporter import (
            _parse_label_name_map,
            apply_label_name_map,
        )
        import traceback
        from agentic.multi_sim_paths import resolve_multi_sim_base_dir
        from agentic.utils.conversation_logger import set_log_file

        working_dir = resolve_multi_sim_base_dir(state)
        if state.get("is_multi_simulation"):
            state["multi_sim_base_dir"] = working_dir
            state["working_directory"] = working_dir
        set_log_file(str(Path(working_dir) / "agent_conversation.log"))
        reporter_dir = str(Path(working_dir) / "reporter")
        Path(reporter_dir).mkdir(parents=True, exist_ok=True)

        file_registry = state.get("file_registry") or {}
        self.file_manager = SecureFileManager(
            working_dir=working_dir,
            agent_name="reporter",
            file_registry=file_registry,
        )
        self.tool_executor = ReporterToolExecutor(
            config={"working_directory": self.file_manager.agent_dir}
        )

        sim_dirs, labels = resolve_combined_sim_context(state)
        _user_goal_text = (
            state.get("user_goal_original")
            or state.get("user_goal")
            or ""
        )
        _enriched_text = (
            state.get("master_enriched_prompt")
            or state.get("enriched_prompt")
            or ""
        )
        if not isinstance(_user_goal_text, str):
            _user_goal_text = str(_user_goal_text)
        if not isinstance(_enriched_text, str):
            _enriched_text = str(_enriched_text)
        _combined_text = f"{_user_goal_text} {_enriched_text}"
        raw_labels = [Path(d).name for d in sim_dirs]
        _label_name_map = _parse_label_name_map(_combined_text, sim_labels=raw_labels)
        if _label_name_map:
            labels = apply_label_name_map(labels, _label_name_map)

        combined_info = (state.get("analysis_results") or {}).get("combined", {})
        _analysis_dir = combined_info.get("analysis_dir") or str(Path(working_dir) / "analysis")
        overlay_plots = collect_combined_overlay_plots(state, _analysis_dir, combined_info)

        from src.reporter.figure_selector import (
            ReportFigurePolicy,
            resolve_report_narrative,
        )
        from src.reporter.protein_identity import resolve_protein_identity
        from src.reporter.report_curator import build_combined_report_plan, display_names_for_sims

        _goal_full = f"{_user_goal_text} {_enriched_text}".strip()
        _report_policy = ReportFigurePolicy.from_config(
            self.config,
            report_type=state.get("report_type", "comprehensive"),
            include_visualizations=state.get("include_visualizations", True),
        )
        _n_sims = len(sim_dirs) if sim_dirs else len(state.get("sim_prompts") or [])
        _narrative = resolve_report_narrative(
            _goal_full,
            enriched_prompt=_enriched_text,
            n_simulations=_n_sims,
            policy=_report_policy,
        )

        log_agent_start(
            "reporter",
            "Combined Multi-Simulation Report (LLM)",
            {
                "labels": labels,
                "overlay_plots": overlay_plots,
                "output_dir": reporter_dir,
                "planning": "llm",
            },
        )

        plan = self._create_combined_reporter_plan_llm(
            user_goal=_user_goal_text,
            enriched_prompt=_enriched_text,
            sim_dirs=sim_dirs,
            labels=labels,
            overlay_plots=overlay_plots,
            analysis_dir=_analysis_dir,
            reporter_dir=reporter_dir,
        )
        if plan is None or not plan.steps:
            logger.warning("Combined reporter LLM plan missing — deterministic fallback")
            return self._run_combined_report(state)

        log_agent_action(
            "reporter",
            "Generated combined reporter plan",
            {
                "steps": len(plan.steps),
                "tools": [s.tool_name for s in plan.steps],
                "overview": plan.overview,
                "report_focus": plan.report_focus,
            },
        )
        try:
            from .schemas import ReporterAgentInput, ReportType

            stub_input = ReporterAgentInput(
                working_directory=working_dir,
                analysis_summary_file=str(Path(_analysis_dir) / "analysis_summary.jsonl"),
                user_goal=_user_goal_text,
                report_type=ReportType.COMPREHENSIVE,
            )
            self._save_execution_plan(plan, stub_input)
        except Exception:
            pass

        identity = resolve_protein_identity(state)
        protein_name: Optional[str] = (
            identity.get("protein_name")
            or identity.get("display_name")
            or identity.get("gene_name")
        )
        if _label_name_map:
            protein_name = ", ".join(display_names_for_sims(raw_labels, _label_name_map))
        elif not protein_name and labels:
            protein_name = labels[0]

        try:
            _report_enriched = (
                state.get("master_enriched_prompt") or state.get("enriched_prompt")
            )
            _combined_user_goal = state.get("user_goal_original") or ""
            if not _combined_user_goal.strip():
                _ug = (_user_goal_text or "").strip()
                if _ug and not _ug.startswith("## Combined Multi-Simulation"):
                    _combined_user_goal = _ug

            combined_analysis_data = self._build_combined_analysis_data(sim_dirs, labels)
            combined_state = dict(state)
            combined_state["user_goal"] = _combined_user_goal
            combined_state["enriched_prompt"] = _report_enriched
            if protein_name:
                sys_info = dict(combined_state.get("system_info") or {})
                sys_info["protein_name"] = protein_name
                combined_state["system_info"] = sys_info

            literature_refs = self._ensure_literature_search(
                combined_analysis_data, combined_state
            )
            literature_review = self._generate_literature_review(
                combined_analysis_data,
                literature_refs,
                combined_state,
                is_combined=True,
            )
            final_impression = self._generate_combined_final_impression(
                combined_analysis_data,
                literature_refs,
                combined_state,
                sim_dirs=sim_dirs,
                labels=labels,
            )
            literature_refs, literature_review, final_impression = (
                self._sync_literature_citations(
                    literature_refs, literature_review, final_impression
                )
            )

            report_plan = build_combined_report_plan(
                overlay_plots,
                sim_dirs,
                raw_labels,
                user_goal=_combined_user_goal,
                enriched_prompt=_report_enriched,
                base_analysis_dir=_analysis_dir,
                label_name_map=_label_name_map,
                llm_client=self.llm,
                literature_snippet=(literature_review or "")[:2000],
            )
            overlay_plots = report_plan.included_overlay_plots

            # Execute planned steps; inject resolved params for the combined HTML tool.
            for i, step in enumerate(plan.steps):
                if step.tool_name in (
                    "search_pubmed",
                    "generate_literature_queries",
                    "search_biorxiv",
                    "search_uniprot",
                    "read_analysis_summary",
                    "generate_html_report",
                    "",
                ):
                    log_agent_action(
                        "reporter",
                        f"Step {i+1}/{len(plan.steps)} skipped (pre-handled or per-sim only)",
                        {"step": step.name, "tool": step.tool_name},
                    )
                    continue

                log_agent_action(
                    "reporter",
                    f"Executing step {i+1}/{len(plan.steps)}",
                    {"step": step.name, "tool": step.tool_name},
                )

                if step.tool_name == "generate_combined_html_report":
                    result = generate_combined_html_report.func(
                        sim_dirs=sim_dirs,
                        labels=labels,
                        overlay_plots=overlay_plots,
                        working_dir=reporter_dir,
                        output_file="combined_report.html",
                        title=report_plan.headline,
                        enriched_prompt=_report_enriched,
                        user_goal=_combined_user_goal,
                        protein_name=protein_name,
                        literature_refs=literature_refs,
                        literature_review=literature_review,
                        final_impression=final_impression,
                        report_focus=(
                            " ".join(plan.report_focus)
                            if plan.report_focus
                            else (
                                " ".join(state.get("report_focus"))
                                if isinstance(state.get("report_focus"), list)
                                else (state.get("report_focus") or "")
                            )
                        ),
                        report_plan=report_plan,
                    )
                    if result.get("success"):
                        report_path = result["output_path"]
                        state["reporter_output"] = report_path
                        log_agent_action(
                            agent_name="reporter",
                            action="Combined Report Generated",
                            details={"report_path": report_path, "planning": "llm"},
                        )
                        log_agent_completion(
                            "reporter",
                            "Combined Multi-Simulation Report (LLM)",
                            state,
                            True,
                        )
                    else:
                        state["errors"].append(
                            f"Combined report failed: {result.get('error', 'unknown')}"
                        )
                        log_agent_completion(
                            "reporter",
                            "Combined Multi-Simulation Report (LLM)",
                            state,
                            False,
                        )
                else:
                    try:
                        params = dict(step.tool_params or {})
                        params["working_dir"] = self.file_manager.agent_dir
                        self.tool_executor.execute_tool(step.tool_name, params)
                    except Exception as step_exc:
                        logger.warning(
                            "Combined reporter step %s failed: %s", step.name, step_exc
                        )

            if not state.get("reporter_output"):
                logger.warning(
                    "LLM combined plan did not produce report — deterministic fallback"
                )
                return self._run_combined_report(state)

        except Exception as exc:
            logger.error(
                "Combined LLM reporter failed (%s) — deterministic fallback\n%s",
                exc,
                traceback.format_exc(),
            )
            state.setdefault("warnings", []).append(
                f"Combined LLM reporter failed ({exc}); using deterministic fallback"
            )
            return self._run_combined_report(state)

        state["next_node"] = "human_reporter_check"
        return state

    def _run_combined_report(self, state: MDState) -> MDState:
        """
        Generate a combined comparison HTML report for all simulations.

        Deterministic fallback path (also used when LLM planning is unavailable).
        Reads overlay plots from state["analysis_results"]["combined"] and
        the per-sim analysis summaries, then writes a self-contained HTML to
        ``{working_directory}/reporter/combined_report.html``.
        """
        from .tools import generate_combined_html_report
        from src.reporter.combined_reporter import (
            _dedupe_dccm_combined_plots,
            _parse_label_name_map,
            apply_label_name_map,
        )

        import traceback

        from agentic.multi_sim_paths import resolve_multi_sim_base_dir

        working_dir = resolve_multi_sim_base_dir(state)
        if state.get("is_multi_simulation"):
            state["multi_sim_base_dir"] = working_dir
            state["working_directory"] = working_dir
        from agentic.utils.conversation_logger import set_log_file

        set_log_file(str(Path(working_dir) / "agent_conversation.log"))
        reporter_dir = str(Path(working_dir) / "reporter")
        Path(reporter_dir).mkdir(parents=True, exist_ok=True)

        # Resolve per-sim dirs and labels (fallback to sim_prompts when state is stale)
        sim_dirs, labels = resolve_combined_sim_context(state)

        _user_goal_text = (
            state.get("user_goal_original")
            or state.get("user_goal")
            or ""
        )
        _enriched_text = (
            state.get("master_enriched_prompt")
            or state.get("enriched_prompt")
            or ""
        )
        if not isinstance(_user_goal_text, str):
            _user_goal_text = str(_user_goal_text)
        if not isinstance(_enriched_text, str):
            _enriched_text = str(_enriched_text)
        _combined_text = f"{_user_goal_text} {_enriched_text}"
        raw_labels = [Path(d).name for d in sim_dirs]
        _label_name_map = _parse_label_name_map(_combined_text, sim_labels=raw_labels)
        if _label_name_map:
            labels = apply_label_name_map(labels, _label_name_map)

        # Overlay plots — merge state, combined results, figures list, and on-disk files
        combined_info = (state.get("analysis_results") or {}).get("combined", {})
        _analysis_dir = combined_info.get("analysis_dir") or str(Path(working_dir) / "analysis")
        overlay_plots = collect_combined_overlay_plots(state, _analysis_dir, combined_info)
        if overlay_plots:
            logger.info(
                "Combined report: using %d overlay plot(s) (disk + state)",
                len(overlay_plots),
            )
        else:
            logger.warning(
                "Combined report: no overlay plots found in state or %s — "
                "run combined analysis first or use: run combined analysis: <task>",
                _analysis_dir,
            )

        # ── Goal-aware figure curation ───────────────────────────────────
        from src.reporter.figure_selector import (
            ReportFigurePolicy,
            curate_overlay_plots,
            dedupe_redundant_overlay_plots,
            resolve_relevant_metrics,
            resolve_report_narrative,
        )

        _goal_full = f"{_user_goal_text} {_enriched_text}".strip()
        _report_policy = ReportFigurePolicy.from_config(
            self.config,
            report_type=state.get("report_type", "comprehensive"),
            include_visualizations=state.get("include_visualizations", True),
        )
        _n_sims = len(sim_dirs) if sim_dirs else len(state.get("sim_prompts") or [])
        _narrative = resolve_report_narrative(
            _goal_full,
            enriched_prompt=_enriched_text,
            n_simulations=_n_sims,
            policy=_report_policy,
        )

        log_agent_action(
            "reporter",
            "Combined reporter deterministic plan",
            {
                "labels": labels,
                "n_overlay_plots": len(overlay_plots),
                "narrative": str(getattr(_narrative, "focus", None) or _narrative)[:200],
            },
        )

        # DSSP: generate combined charts only when requested
        from agentic.planner.planning_guidelines import detect_requested_metrics

        _requested_metrics = detect_requested_metrics(_goal_full)
        _goal_lower = _goal_full.lower()
        _want_dssp = (
            _requested_metrics is None and "dssp" in _goal_lower
        ) or (
            _requested_metrics is not None and "dssp" in _requested_metrics
        )
        _analysis_dir = combined_info.get("analysis_dir") or str(Path(working_dir) / "analysis")
        if _want_dssp:
            try:
                Path(_analysis_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                _analysis_dir = reporter_dir
            from src.reporter.combined_reporter import (
                generate_dssp_comparison_chart,
                run_combined_dssp_analysis,
            )
            _dssp_existing = Path(_analysis_dir) / "dssp_comparison.png"
            if not _dssp_existing.exists():
                run_combined_dssp_analysis(sim_dirs, labels, _analysis_dir, _goal_full)
            _dssp_chart = generate_dssp_comparison_chart(
                sim_dirs, labels, _analysis_dir, user_goal=_goal_full,
            )
            if _dssp_chart and _dssp_chart not in overlay_plots:
                overlay_plots.append(_dssp_chart)
            for _dssp_hm in sorted(Path(_analysis_dir).glob("dssp_activation_loop_*.png")):
                if str(_dssp_hm) not in overlay_plots:
                    overlay_plots.append(str(_dssp_hm))
        else:
            logger.info(
                "Skipping combined DSSP analysis — not requested in user goal "
                "(detected metrics: %s)", _requested_metrics
            )

        # Add combined analysis artifacts from state
        for _dccm_plot in combined_info.get("dccm_plots", []):
            _p = Path(_dccm_plot)
            if _p.exists() and str(_p) not in overlay_plots:
                overlay_plots.append(str(_p))
        overlay_plots = _dedupe_dccm_combined_plots(overlay_plots)
        for _seg_plot in combined_info.get("rmsf_segment_plots", []):
            _p = Path(_seg_plot)
            if _p.exists() and str(_p) not in overlay_plots:
                overlay_plots.append(str(_p))
        _com_plot = combined_info.get("com_distance_plot")
        if _com_plot:
            _p = Path(_com_plot)
            if _p.exists() and str(_p) not in overlay_plots:
                overlay_plots.append(str(_p))

        _relevant = _narrative.relevant_metrics
        overlay_plots = dedupe_redundant_overlay_plots(overlay_plots, _narrative)
        overlay_plots = curate_overlay_plots(
            overlay_plots,
            user_goal=_goal_full,
            enriched_prompt=_enriched_text,
            policy=_report_policy,
            narrative=_narrative,
        )
        logger.info(
            "Combined report: %d curated overlay plot(s) for theme=%s metrics=%s",
            len(overlay_plots),
            _narrative.primary_theme,
            sorted(_relevant),
        )

        log_agent_start(
            "reporter",
            "Combined Multi-Simulation Report",
            {
                "sim_dirs": sim_dirs,
                "labels": labels,
                "overlay_plots": overlay_plots,
                "output_dir": reporter_dir,
            },
        )

        # Resolve protein/system name for literature + report title
        from src.reporter.protein_identity import resolve_protein_identity
        from src.reporter.report_curator import build_combined_report_plan, display_names_for_sims

        identity = resolve_protein_identity(state)
        protein_name: Optional[str] = (
            identity.get("protein_name")
            or identity.get("display_name")
            or identity.get("gene_name")
        )
        if _label_name_map:
            protein_name = ", ".join(display_names_for_sims(raw_labels, _label_name_map))
        elif not protein_name and labels:
            protein_name = labels[0]


        try:
            # Prefer master_enriched_prompt (supervisor's unified rephrased goal)
            # over enriched_prompt, which by the combined-analysis phase has been
            # overwritten with the planner's combined_analysis_plan text.
            _report_enriched = (
                state.get("master_enriched_prompt")
                or state.get("enriched_prompt")
            )
            _combined_user_goal = state.get("user_goal_original") or ""
            if not _combined_user_goal.strip():
                # Avoid passing combined-analysis instruction blob as the display goal
                _ug = (_user_goal_text or "").strip()
                if _ug and not _ug.startswith("## Combined Multi-Simulation"):
                    _combined_user_goal = _ug

            # Contextual literature search across all simulations
            combined_analysis_data = self._build_combined_analysis_data(sim_dirs, labels)
            combined_state = dict(state)
            combined_state["user_goal"] = _combined_user_goal
            combined_state["enriched_prompt"] = _report_enriched
            if protein_name:
                sys_info = dict(combined_state.get("system_info") or {})
                sys_info["protein_name"] = protein_name
                combined_state["system_info"] = sys_info
            literature_refs = self._ensure_literature_search(combined_analysis_data, combined_state)
            literature_review = self._generate_literature_review(
                combined_analysis_data, literature_refs, combined_state,
                is_combined=True,
            )
            final_impression = self._generate_combined_final_impression(
                combined_analysis_data, literature_refs, combined_state,
                sim_dirs=sim_dirs,
                labels=labels,
            )
            literature_refs, literature_review, final_impression = (
                self._sync_literature_citations(
                    literature_refs, literature_review, final_impression
                )
            )

            report_plan = build_combined_report_plan(
                overlay_plots,
                sim_dirs,
                raw_labels,
                user_goal=_combined_user_goal,
                enriched_prompt=_report_enriched,
                base_analysis_dir=_analysis_dir,
                label_name_map=_label_name_map,
                llm_client=self.llm,
                literature_snippet=(literature_review or "")[:2000],
            )
            overlay_plots = report_plan.included_overlay_plots
            if report_plan.excluded_plots:
                logger.info(
                    "Combined report curator excluded %d plot(s): %s",
                    len(report_plan.excluded_plots),
                    list(report_plan.excluded_plots.keys())[:5],
                )

            result = generate_combined_html_report.func(
                sim_dirs=sim_dirs,
                labels=labels,
                overlay_plots=overlay_plots,
                working_dir=reporter_dir,
                output_file="combined_report.html",
                title=report_plan.headline,
                enriched_prompt=_report_enriched,
                user_goal=_combined_user_goal,
                protein_name=protein_name,
                literature_refs=literature_refs,
                literature_review=literature_review,
                final_impression=final_impression,
                report_focus=(
                    " ".join(state.get("report_focus"))
                    if isinstance(state.get("report_focus"), list)
                    else (state.get("report_focus") or "")
                ),
                report_plan=report_plan,
            )

            if result.get("success"):
                report_path = result["output_path"]
                state["reporter_output"] = report_path
                log_agent_action(
                    agent_name="reporter",
                    action="Combined Report Generated",
                    details={"report_path": report_path},
                )
                log_agent_completion("reporter", "Combined Multi-Simulation Report", state, True)
            else:
                state["errors"].append(f"Combined report failed: {result.get('error', 'unknown')}")
                log_agent_completion("reporter", "Combined Multi-Simulation Report", state, False)

        except Exception as exc:
            logger.error(f"Combined report failed: {exc}\n{traceback.format_exc()}")
            state["errors"].append(f"Combined reporter error: {exc}")

        # Route to reporter checkpoint (not supervisor) so human can inspect
        state["next_node"] = "human_reporter_check"
        return state
    
    def _extract_agent_instructions(self, full_plan: str, agent_name: str) -> Optional[str]:
        """Extract agent-specific instructions from plan"""
        if not full_plan:
            return None
        
        # Look for reporter agent section
        patterns = [
            rf"{agent_name}[:\s]+(.+?)(?=\n\n|\n###|\Z)",
            r"Reporter[:\s]+(.+?)(?=\n\n|\n###|\Z)",
            r"Report Generation[:\s]+(.+?)(?=\n\n|\n###|\Z)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_plan, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _prepare_agent_input(self, state: MDState) -> ReporterAgentInput:
        """Prepare reporter agent input from state"""
        
        working_dir = state.get("working_directory", "working_dir")
        analysis_dir = f"{working_dir}/analysis"
        summary_file = f"{analysis_dir}/analysis_summary.jsonl"
        
        # Get planner instructions
        execution_plan = state.get("execution_plan") or {}
        reporter_instructions = state.get("reporter_instructions")
        
        if not reporter_instructions:
            full_plan = execution_plan.get("full_plan", "")
            reporter_instructions = self._extract_agent_instructions(full_plan, "Reporter")
        
        # Determine report type
        report_type_str = state.get("report_type", "comprehensive")
        try:
            report_type = ReportType(report_type_str.lower())
        except ValueError:
            report_type = ReportType.COMPREHENSIVE
        
        return ReporterAgentInput(
            working_directory=working_dir,
            analysis_summary_file=summary_file,
            user_goal=state.get("enriched_prompt") or state.get("user_goal"),
            report_type=report_type,
            planner_instructions=reporter_instructions,
            literature_search=state.get("include_literature", True),
            literature_keywords=state.get("literature_keywords"),
            include_visualizations=state.get("include_visualizations", True),
            context={
                "force_field": state.get("force_field", "amber99sb-ildn"),
                "md_engine": state.get("md_engine", "gromacs"),
                "execution_plan": execution_plan
            }
        )
    
    def _run_reporter_workflow(
        self,
        agent_input: ReporterAgentInput,
        state: MDState
    ) -> ReporterAgentOutput:
        """
        Execute reporter workflow: plan → generate → validate
        """
        import sys
        import traceback
        
        try:
            # Step 1: Create execution plan using LLM
            logger.info("="*60)
            logger.info("REPORTER WORKFLOW START")
            logger.info("  User goal: %s", (agent_input.user_goal or 'N/A')[:120])
            logger.info("  Report type: %s", agent_input.report_type.value)
            logger.info("  Analysis file: %s", agent_input.analysis_summary_file)
            logger.info("="*60)

            plan = None
            if self.llm.available:
                logger.info("[1/4] Creating execution plan via LLM...")
                plan = self._create_llm_plan(agent_input, state)
            
            if not plan or not plan.steps:
                logger.info("[1/4] LLM plan empty or unavailable — using fallback plan")
                plan = self._create_fallback_plan(agent_input)
            
            logger.info("[1/4] Plan created: %d steps, focus=%s",
                        len(plan.steps),
                        plan.report_focus[:3] if plan.report_focus else [])
            log_agent_action(
                agent_name="reporter",
                action="Generated reporter plan",
                details={
                    "steps": len(plan.steps),
                    "report_focus": plan.report_focus[:3] if plan.report_focus else [],
                    "literature_queries": len(plan.literature_queries)
                }
            )
            
            # Save the execution plan to file
            self._save_execution_plan(plan, agent_input)
            
            # Step 2: Execute the plan
            logger.info("[2/4] Executing reporter plan...")
            result = self._execute_reporter_plan(agent_input, plan, state)
            
            # Step 3: Create final output
            logger.info("[3/4] Building final output (report=%s)",
                        result.report_file or 'None')
            output = ReporterAgentOutput(
                success=result.success and len(result.issues) == 0,
                result=result,
                errors=result.issues,
                next_actions=["Report available for review"],
                report_path=result.report_file,
                pdf_path=result.generated_files.get("pdf")
            )
            
            return output
            
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Reporter workflow failed: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {tb_str}")
            
            return ReporterAgentOutput(
                success=False,
                result=ReporterResult(
                    success=False,
                    completed_steps=0,
                    total_steps=0,
                    report="Reporter workflow failed: " + str(e)
                ),
                errors=[str(e)]
            )
    
    def _create_llm_plan(
        self,
        agent_input: ReporterAgentInput,
        state: MDState
    ) -> Optional[ReporterPlan]:
        """Create execution plan using LLM"""
        
        # Get available tools
        tool_metadata = get_tool_metadata()
        tools_list = []
        for tool_name, tool_info in tool_metadata.items():
            tools_list.append(f"→ {tool_name}: {tool_info['description']}")
        tools_str = "\n".join(tools_list)
        
        prompt = self._build_planning_prompt(agent_input, tools_str)
        
        try:
            response = self.llm.prompt(prompt)
            
            log_llm_interaction(
                agent_name="reporter.planning",
                prompt=prompt,
                response=response,
                is_mock=not self.llm.available
            )
            
            # Parse JSON response
            plan_dict = self._extract_plan_json(response)
            
            if plan_dict is None:
                logger.warning("Failed to parse LLM response into plan JSON")
                return None
            
            # Build ReporterPlan from response
            return ReporterPlan(
                reasoning=plan_dict.get("reasoning", "LLM-generated plan"),
                overview=plan_dict.get("overview", "Report generation"),
                steps=[
                    ReporterStep(
                        name=step.get("name", "unknown"),
                        description=step.get("description", ""),
                        tool_name=step.get("tool_name", ""),
                        tool_params=step.get("tool_params", {}),
                        reason=step.get("reason", "")
                    )
                    for step in plan_dict.get("steps", [])
                ],
                report_focus=plan_dict.get("report_focus", []),
                literature_queries=plan_dict.get("literature_queries", []),
                estimated_complexity=plan_dict.get("estimated_complexity", "medium")
            )
            
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using fallback")
            return None
    
    def _build_planning_prompt(
        self,
        agent_input: ReporterAgentInput,
        tools_str: str
    ) -> str:
        """Build LLM planning prompt"""
        
        prompt_template = """You are a scientific report generator creating reports from MD trajectory analysis.

**User Goal:**
{user_goal}

**Planner Instructions:**
{planner_instructions}

**Analysis Summary File:**
{summary_file}

**Report Type:**
{report_type}

**Literature Search:**
Enabled: {literature_enabled}
{keywords_section}

**Available Tools:**
{tools_str}

**Your Task:**
Create a plan to generate a scientific report by:
1. Reading the analysis_summary.jsonl file (contains analysis results, statistics, and image file paths)
2. Extracting key findings and statistics
3. Optionally searching scientific literature for context
4. Generating an HTML report with embedded visualizations and insights

**HTML REPORT FEATURES:**
The HTML report automatically includes:
- **Embedded Images**: Analysis plots (RMSD, RMSF, DSSP heatmaps, etc.) are embedded as base64
- **Key Statistics Cards**: Important values (mean, std, min, max) displayed prominently beneath images
- **Modern Layout**: Clean, professional styling with gradients and visual hierarchy
- **Organized Sections**: Each analysis type in its own section with dividers

The tool reads image file paths from analysis_summary.jsonl and embeds them automatically.
No need to specify image paths in tool_params - they're extracted from the analysis data.

**IMPORTANT PATH GUIDELINES:**
- The analysis_summary.jsonl file is typically in the "analysis/" subdirectory
- For read_analysis_summary: Use "analysis/analysis_summary.jsonl" or just "analysis_summary.jsonl" (tool will search)
- For generate_html_report: ALWAYS use the exact filename "report.html" (do not invent per-protein names). The per-sim report filename must be consistent so the combined report can find it.
- Do NOT include "working_dir" in tool_params - it will be automatically set to the reporter's output directory
- All files will be created in the reporter's dedicated directory (working_dir/reporter/)

**CRITICAL — per-simulation reports:**
- Use ONLY `generate_html_report` for a single-simulation report (this workflow).
- Do NOT use `generate_combined_html_report` — that tool is reserved for multi-simulation combined reports with sim_dirs/labels.

**Output Format (JSON):**
{{
  "reasoning": "Why this approach",
  "overview": "High-level summary",
  "steps": [
    {{
      "name": "Read Analysis Summary",
      "description": "Parse analysis_summary.jsonl",
      "tool_name": "read_analysis_summary",
      "tool_params": {{
        "summary_file": "analysis_summary.jsonl"
      }},
      "reason": "Need to load analysis results"
    }},
    {{
      "name": "Generate HTML Report",
      "description": "Create comprehensive report",
      "tool_name": "generate_html_report",
      "tool_params": {{
        "analysis_data": {{}},
        "output_file": "report.html",
        "report_type": "comprehensive"
      }},
      "reason": "Produce final deliverable"
    }}
  ],
  "report_focus": ["Key topic 1", "Key topic 2"],
  "literature_queries": ["query 1", "query 2"],
  "estimated_complexity": "low|medium|high"
}}
"""
        
        keywords_section = ""
        if agent_input.literature_keywords:
            keywords_section = f"Keywords: {', '.join(agent_input.literature_keywords)}"
        else:
            keywords_section = "Auto-generate based on analysis types"
        
        return prompt_template.format(
            user_goal=agent_input.user_goal or "Generate scientific report",
            planner_instructions=agent_input.planner_instructions or "Create comprehensive report",
            summary_file=agent_input.analysis_summary_file,
            report_type=agent_input.report_type.value,
            literature_enabled="Yes" if agent_input.literature_search else "No",
            keywords_section=keywords_section,
            tools_str=tools_str
        )
    
    def _build_combined_analysis_data(
        self,
        sim_dirs: List[str],
        labels: List[str],
    ) -> Dict[str, Any]:
        """Merge per-simulation analysis summaries for combined literature search."""
        entries: List[Dict[str, Any]] = []
        analysis_types: Dict[str, int] = {}
        try:
            from src.analysis.summary_logger import read_summary_file
        except ImportError:
            read_summary_file = None

        for sim_dir, label in zip(sim_dirs, labels):
            analysis_dir = Path(sim_dir) / "analysis"
            records: List[Dict[str, Any]] = []
            if read_summary_file:
                try:
                    records = read_summary_file(str(analysis_dir))
                except Exception:
                    records = []
            if not records:
                jsonl = analysis_dir / "analysis_summary.jsonl"
                if jsonl.exists():
                    for line in jsonl.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("---"):
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass

            for rec in records:
                if not rec.get("analysis_type"):
                    continue
                tagged = dict(rec)
                tagged["simulation_label"] = label
                entries.append(tagged)
                atype = rec["analysis_type"]
                analysis_types[atype] = analysis_types.get(atype, 0) + 1

        return {"entries": entries, "analysis_types": analysis_types}

    def _log_literature_step(self, step: str, action: str, details: Dict[str, Any]) -> None:
        """Write one literature-survey provenance step into agent_conversation.log."""
        payload = {"literature_step": step, **(details or {})}
        try:
            log_agent_action("reporter", action, payload)
        except Exception:
            logger.debug("literature provenance log failed", exc_info=True)

    def _ensure_literature_search(
        self,
        analysis_data: Dict[str, Any],
        state: MDState
    ) -> List[Dict[str, Any]]:
        """Six-step evidence-grounded literature survey.

        1) MD observations → 2) multi-source gather → 3) rank/screen →
        4) OA evidence passages → 5) compare vs MD → 6) ranked candidates for report.
        Full provenance is written to ``agent_conversation.log`` (and
        ``reporter/literature_provenance.json`` when possible).
        """
        literature_refs: List[Dict[str, Any]] = []
        # Collect a larger candidate pool, then rank down for the LLM.
        # Uncited papers are pruned after the review is written.
        MAX_REFS = 12
        COLLECT_MAX = 28
        provenance: Dict[str, Any] = {
            "pipeline": "evidence_grounded_literature_v1",
            "steps": {},
        }
        self._literature_provenance = provenance
        self._literature_observations = []
        self._literature_evidence = []
        self._literature_comparisons = []

        try:
            from src.reporter.literature_search import (
                generate_literature_queries, search_pubmed,
                search_biorxiv, search_uniprot,
                extract_research_context, rank_literature_refs,
                extract_analysis_stats_from_entries,
                score_literature_ref_relevance,
            )
            from src.reporter.literature_evidence import (
                extract_md_observations,
                search_crossref,
                search_arxiv,
                fetch_oa_evidence_passages,
                classify_evidence_vs_observations,
                compact_ref,
                write_literature_provenance,
            )

            # --- Resolve analysis types ---
            analysis_types: list = []
            if isinstance(analysis_data, dict):
                at = analysis_data.get("analysis_types", {})
                if isinstance(at, dict):
                    analysis_types = list(at.keys())
                elif isinstance(at, list):
                    analysis_types = at
            if not analysis_types:
                analysis_types = ["RMSD", "RMSF"]

            user_goal = state.get("enriched_prompt") or state.get("user_goal", "MD simulation analysis")

            from src.reporter.protein_identity import resolve_protein_identity
            identity = resolve_protein_identity(state, user_goal=user_goal)
            protein_name = (
                identity.get("protein_name")
                or identity.get("display_name")
                or identity.get("gene_name")
            )
            gene_name = identity.get("gene_name") or ""
            protein_family = identity.get("protein_family") or ""
            related_terms = identity.get("related_terms") or []
            aliases = list(identity.get("aliases") or [])
            uniprot_id = identity.get("uniprot_id") or ""
            key_references = list(identity.get("key_references") or [])
            logger.info(
                "Literature search: protein=%r gene=%r aliases=%d family=%r source=%s uniprot=%s",
                protein_name,
                gene_name,
                len(aliases),
                protein_family,
                identity.get("source"),
                uniprot_id,
            )
            provenance["identity"] = {
                "protein": protein_name,
                "gene": gene_name,
                "aliases": aliases,
                "family": protein_family,
                "uniprot_id": uniprot_id,
                "source": identity.get("source"),
                "study_proteins": identity.get("study_proteins") or [],
            }

            # ── Collect analysis stats to drive result-specific queries ──
            analysis_stats = extract_analysis_stats_from_entries(analysis_data)
            if not analysis_stats and isinstance(analysis_data, dict):
                results = analysis_data.get("results") or analysis_data.get("analysis_results") or {}
                if isinstance(results, dict):
                    for atype, aval in results.items():
                        if isinstance(aval, dict):
                            astat = {}
                            for k, v in aval.items():
                                if any(x in k.lower() for x in ("mean", "max", "min", "avg")):
                                    try:
                                        astat[k] = float(v)
                                    except (TypeError, ValueError):
                                        pass
                            if astat:
                                analysis_stats[atype] = astat

            research_context = extract_research_context(
                user_goal=user_goal,
                protein_name=protein_name,
                analysis_types=analysis_types,
                analysis_stats=analysis_stats,
                protein_family=protein_family,
                related_terms=related_terms,
                aliases=aliases,
                gene_name=gene_name or None,
                uniprot_id=uniprot_id or None,
            )
            if research_context.get("aliases"):
                aliases = list(dict.fromkeys(aliases + list(research_context["aliases"])))
                research_context["aliases"] = aliases
            if research_context.get("protein_names") and (
                not protein_name or protein_name in research_context["protein_ids"]
            ):
                protein_name = research_context["protein_names"][0]
            if not gene_name and research_context.get("primary_gene"):
                gene_name = research_context["primary_gene"]

            # ── Step 1: Analyze simulation results → observations ──
            observations = extract_md_observations(
                analysis_data, user_goal=user_goal or "", protein_name=protein_name or ""
            )
            self._literature_observations = observations
            provenance["steps"]["1_observations"] = {
                "n": len(observations),
                "observations": observations,
            }
            self._log_literature_step(
                "1_analyze_md_results",
                "Literature Step 1 — Analyze MD results (observations needing interpretation)",
                {
                    "n_observations": len(observations),
                    "observations": [
                        {
                            "id": o.get("id"),
                            "kind": o.get("kind"),
                            "statement": o.get("statement"),
                            "keywords": o.get("keywords"),
                        }
                        for o in observations
                    ],
                },
            )

            # --- Generate prioritised queries (alias / family / metric lanes) ---
            query_result = generate_literature_queries.invoke({
                "analysis_types": analysis_types,
                "user_goal": user_goal,
                "protein_name": protein_name or "",
                "analysis_stats": analysis_stats or None,
                "protein_family": protein_family or "",
                "related_terms": related_terms or [],
                "aliases": aliases or [],
                "gene_name": gene_name or "",
                "uniprot_id": uniprot_id or "",
            })
            queries = {}
            resolved_name = ""
            if isinstance(query_result, dict) and query_result.get("success"):
                queries = query_result.get("queries", {})
                resolved_name = query_result.get("protein_name", "") or ""
                for a in query_result.get("aliases") or []:
                    if a and a not in aliases:
                        aliases.append(a)

            # Observation-driven query extras (residue hotspots / mechanisms)
            for obs in observations:
                if obs.get("kind") == "rmsf_hotspots":
                    flex = (obs.get("metrics") or {}).get("top_flexible") or []
                    if flex and (gene_name or protein_name):
                        queries["obs_rmsf_hotspots"] = (
                            f"{gene_name or protein_name} "
                            f"{' '.join(str(x).split()[0] for x in flex[:3])} "
                            f"flexibility molecular dynamics"
                        )
                    queries.setdefault(
                        "theme_obs_flexibility",
                        "protein residue flexibility RMSF activation loop molecular dynamics",
                    )
                if obs.get("kind") == "correlated_motions":
                    queries.setdefault(
                        "theme_obs_allostery",
                        f"{gene_name or protein_name or 'protein'} allosteric correlated motions molecular dynamics",
                    )

            provenance["steps"]["2_queries"] = {
                "n": len(queries),
                "queries": dict(queries),
            }
            self._log_literature_step(
                "2_retrieve_candidates",
                f"Literature Step 2 — Retrieve candidates ({len(queries)} queries; "
                "PubMed, Europe PMC, Crossref, arXiv/bioRxiv, UniProt)",
                {
                    "n_queries": len(queries),
                    "queries": [
                        {"key": k, "query": v}
                        for k, v in queries.items()
                        if v and k != "_fallback_general"
                    ],
                    "sources": ["PubMed", "EuropePMC", "Crossref", "arXiv", "bioRxiv", "UniProt"],
                },
            )

            # Deduplication set (PMIDs + DOIs)
            seen_ids: set = set()
            source_counts: Dict[str, int] = {}

            def _add_ref(ref: Dict[str, Any], query_key: str = "") -> bool:
                """Add ref if not duplicate. Returns True if added."""
                ref = dict(ref)
                if query_key and not ref.get("query_key"):
                    ref["query_key"] = query_key
                pmid = ref.get("pmid")
                doi = (ref.get("doi") or "").lower().strip()
                if pmid and pmid in seen_ids:
                    return False
                if doi and doi in seen_ids:
                    return False
                if pmid:
                    seen_ids.add(pmid)
                if doi:
                    seen_ids.add(doi)
                literature_refs.append(ref)
                src = str(ref.get("source") or "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
                return True

            from src.reporter.literature_search import search_europe_pmc

            def _room() -> int:
                return max(0, COLLECT_MAX - len(literature_refs))

            # ── Seed UniProt curated PMIDs (titles often omit the gene) ──
            seeded_curated = 0
            for ref in key_references:
                if _room() <= 0:
                    break
                seed = dict(ref)
                seed.setdefault("source", "uniprot_curated")
                seed.setdefault("query_key", "uniprot:key_references")
                if _add_ref(seed, query_key="uniprot:key_references"):
                    seeded_curated += 1
            if seeded_curated:
                logger.info("Seeded %d UniProt curated references", seeded_curated)
                provenance["steps"]["2b_uniprot_seed"] = {"n": seeded_curated}

            # Split generated queries: protein-named first, then theme/objective.
            protein_qitems = [
                (k, v) for k, v in queries.items()
                if v and k != "_fallback_general" and not str(k).startswith("theme_")
            ]
            theme_qitems = [
                (k, v) for k, v in queries.items()
                if v and str(k).startswith("theme_")
            ]

            # ── PubMed protein-priority queries ───────────────────
            for qkey, query_text in protein_qitems:
                if _room() <= 0:
                    break
                result = search_pubmed.invoke({
                    "query": query_text,
                    "max_results": min(4, _room()),
                    "include_abstracts": True,
                    "max_age_years": 10,
                })
                if isinstance(result, dict) and result.get("success"):
                    for ref in result.get("results", []):
                        if not _add_ref(ref, query_key=f"pubmed:{qkey}"):
                            continue
                        if _room() <= 0:
                            break

            # A multi-protein goal already has one query per named protein.
            # A second theme wave repeats those names and trips the PubMed rate limit.
            _study_genes = list(research_context.get("study_genes") or [])
            if len(_study_genes) >= 2:
                _search_names = []
            else:
                _search_names = list(dict.fromkeys(
                    n
                    for n in (
                        [gene_name]
                        + list(aliases or [])
                        + list(research_context.get("protein_names") or [])
                        + list(research_context.get("aliases") or [])
                    )
                    if n
                ))[:6]
            for pname in _search_names:
                if _room() <= 0:
                    break
                themes = [
                    f"{pname} protein structure function review",
                    f"{pname} molecular dynamics simulation",
                    f"{pname} functional study",
                    f"{pname} computational study OR molecular dynamics",
                ]
                if protein_family:
                    themes.append(f"{pname} {protein_family} dynamics activation")
                    themes.append(
                        f"{protein_family} molecular dynamics conformational gating"
                    )
                elif "pseudokinase" in (user_goal or "").lower():
                    themes.append(f"{pname} pseudokinase activation loop")
                if any(t in (user_goal or "").lower() for t in ("atp", "holo", "apo", "ligand")):
                    themes.append(f"{pname} ATP binding apo holo")
                for theme in themes:
                    if _room() <= 0:
                        break
                    result = search_pubmed.invoke({
                        "query": theme,
                        "max_results": min(3, _room()),
                        "include_abstracts": True,
                        "max_age_years": 12,
                    })
                    if isinstance(result, dict) and result.get("success"):
                        for ref in result.get("results", []):
                            if not _add_ref(ref, query_key=f"pubmed_theme:{pname}"):
                                continue
                            if _room() <= 0:
                                break
            logger.info("After PubMed (protein): %d refs", len(literature_refs))

            # ── PubMed theme / objective / method queries ─────────
            for qkey, query_text in theme_qitems:
                if _room() <= 0:
                    break
                result = search_pubmed.invoke({
                    "query": query_text,
                    "max_results": min(4, _room()),
                    "include_abstracts": True,
                    "max_age_years": 10,
                })
                if isinstance(result, dict) and result.get("success"):
                    for ref in result.get("results", []):
                        if not _add_ref(ref, query_key=f"pubmed:{qkey}"):
                            continue
                        if _room() <= 0:
                            break
            logger.info("After PubMed (theme/objective): %d refs", len(literature_refs))

            # ── Europe PMC (protein + theme) ──────────────────────
            epmc_queries: List[str] = []
            if gene_name:
                epmc_queries.extend([
                    f"{gene_name} protein molecular dynamics",
                    f"{gene_name} structure function",
                ])
            if protein_family:
                epmc_queries.append(f"{protein_family} molecular dynamics conformational")
            epmc_queries.append("protein domain molecular dynamics RMSD RMSF conformational")
            for epmc_q in epmc_queries:
                if _room() <= 0:
                    break
                epmc_result = search_europe_pmc(
                    epmc_q,
                    max_results=min(4, _room()),
                    max_age_years=12,
                    open_access_only=True,
                )
                if epmc_result.get("success"):
                    for ref in epmc_result.get("results", []):
                        if not _add_ref(ref, query_key=f"europepmc:{epmc_q[:60]}"):
                            continue
                        if _room() <= 0:
                            break
            logger.info("After Europe PMC: %d refs", len(literature_refs))

            # ── Crossref ─────────────────────────────────────────
            xref_queries = []
            if gene_name:
                xref_queries.append(f"{gene_name} molecular dynamics")
            for k in ("theme_family_dynamics", "theme_result_rmsd", "theme_objective"):
                if queries.get(k):
                    xref_queries.append(queries[k])
            for xq in xref_queries[:3]:
                if _room() <= 0:
                    break
                try:
                    xr = search_crossref(xq, max_results=min(4, _room()))
                    if xr.get("success"):
                        for ref in xr.get("results", []):
                            if not _add_ref(ref, query_key=f"crossref:{xq[:60]}"):
                                continue
                            if _room() <= 0:
                                break
                except Exception as e:
                    logger.warning("Crossref search failed (non-fatal): %s", e)
            logger.info("After Crossref: %d refs", len(literature_refs))

            # ── arXiv + bioRxiv preprints ─────────────────────────
            arxiv_queries = []
            if gene_name:
                arxiv_queries.append(f"{gene_name} molecular dynamics")
            if queries.get("theme_family_dynamics"):
                arxiv_queries.append(queries["theme_family_dynamics"])
            for aq in arxiv_queries[:2]:
                if _room() <= 0:
                    break
                try:
                    ar = search_arxiv(aq, max_results=min(3, _room()))
                    if ar.get("success"):
                        for ref in ar.get("results", []):
                            if not _add_ref(ref, query_key=f"arxiv:{aq[:60]}"):
                                continue
                            if _room() <= 0:
                                break
                except Exception as e:
                    logger.warning("arXiv search failed (non-fatal): %s", e)

            bio_queries = []
            if resolved_name:
                bio_queries.append(
                    queries.get("protein_dynamics")
                    or queries.get("protein_function")
                    or f"{resolved_name} molecular dynamics"
                )
            for k in ("theme_family_dynamics", "theme_result_rmsd", "theme_objective"):
                if queries.get(k):
                    bio_queries.append(queries[k])
            for bio_query in bio_queries:
                if _room() <= 0:
                    break
                try:
                    bio_result = search_biorxiv.invoke({
                        "query": bio_query,
                        "max_results": min(4, _room()),
                        "max_age_years": 5,
                    })
                    if isinstance(bio_result, dict) and bio_result.get("success"):
                        for ref in bio_result.get("results", []):
                            if not _add_ref(ref, query_key=f"biorxiv:{bio_query[:60]}"):
                                continue
                            if _room() <= 0:
                                break
                except Exception as e:
                    logger.warning("bioRxiv search failed (non-fatal): %s", e)
            logger.info("After arXiv/bioRxiv: %d refs", len(literature_refs))

            # ── UniProt protein context (prefer accession / gene) ──
            uni_query = uniprot_id or gene_name or resolved_name
            if uni_query and _room() > 0:
                try:
                    uni_result = search_uniprot.invoke({
                        "query": uni_query,
                        "max_results": 3,
                    })
                    if isinstance(uni_result, dict) and uni_result.get("success"):
                        for entry in uni_result.get("entries", []):
                            for a in entry.get("aliases") or []:
                                if a and a not in aliases:
                                    aliases.append(a)
                            func_text = entry.get("function") or ""
                            _add_ref({
                                "title": f"{entry.get('protein_name', uni_query)} — UniProt functional annotation ({entry.get('organism', '')})",
                                "authors": ["UniProt Consortium"],
                                "journal": "UniProt Knowledgebase",
                                "year": 2025,
                                "pmid": None,
                                "doi": None,
                                "abstract": func_text[:400] if func_text else None,
                                "url": entry.get("url"),
                                "source": "UniProt",
                            }, query_key="uniprot:entry")
                            for kref in entry.get("key_references", []):
                                if _room() <= 0:
                                    break
                                if kref.get("title") or kref.get("pmid"):
                                    _add_ref({
                                        "title": kref.get("title") or f"UniProt curated PMID {kref.get('pmid')}",
                                        "authors": [],
                                        "journal": kref.get("journal", ""),
                                        "year": kref.get("year"),
                                        "pmid": kref.get("pmid"),
                                        "doi": kref.get("doi"),
                                        "abstract": None,
                                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{kref['pmid']}/" if kref.get("pmid") else None,
                                        "source": "uniprot_curated",
                                    }, query_key="uniprot:key_ref")
                            if _room() <= 0:
                                break
                        research_context["aliases"] = list(dict.fromkeys(
                            list(research_context.get("aliases") or []) + aliases
                        ))
                    logger.info("After UniProt: %d refs (aliases=%d)", len(literature_refs), len(aliases))
                except Exception as e:
                    logger.warning("UniProt search failed (non-fatal): %s", e)

            # ── Fallback if still empty ───────────────────────────
            if not literature_refs:
                logger.info("No results from any source; trying PubMed fallback")
                fallback_q = queries.get("_fallback_general", "molecular dynamics protein stability review")
                result = search_pubmed.invoke({
                    "query": fallback_q,
                    "max_results": 5,
                    "include_abstracts": True,
                    "max_age_years": 10,
                })
                if isinstance(result, dict) and result.get("success"):
                    for ref in result.get("results", []):
                        _add_ref(ref, query_key="pubmed:fallback")

            candidates_before_rank = [compact_ref(r) for r in literature_refs]
            provenance["steps"]["2_gather"] = {
                "n_candidates": len(literature_refs),
                "source_counts": dict(source_counts),
                "candidates": candidates_before_rank,
            }
            self._log_literature_step(
                "2_retrieve_candidates_complete",
                f"Literature Step 2 complete — {len(literature_refs)} unique candidates gathered",
                {
                    "n_candidates": len(literature_refs),
                    "source_counts": dict(source_counts),
                    "candidates": candidates_before_rank,
                },
            )
            logger.info(
                "Literature search complete: %d candidates before ranking",
                len(literature_refs),
            )

            # ── Step 3: Rank and screen (title/abstract) ──────────
            ranked = rank_literature_refs(
                literature_refs,
                research_context,
                max_refs=MAX_REFS,
                min_score=10,
                protein_slot_fraction=0.5,
            )
            ranked_view = []
            for ref in ranked:
                score, note = score_literature_ref_relevance(ref, research_context)
                item = compact_ref(ref)
                item["score"] = score
                item["relevance_note"] = note
                matched = ref.get("matched_aliases") or ref.get("_matched_aliases") or []
                if matched:
                    item["matched_aliases"] = list(matched)
                ranked_view.append(item)
            literature_refs = ranked
            provenance["steps"]["3_rank_screen"] = {
                "n_ranked": len(ranked_view),
                "max_refs": MAX_REFS,
                "aliases_used": aliases,
                "ranked": ranked_view,
            }
            self._log_literature_step(
                "3_rank_and_screen",
                f"Literature Step 3 — Rank/screen to {len(ranked_view)} papers for deeper examination",
                {
                    "n_input_candidates": len(candidates_before_rank),
                    "n_ranked": len(ranked_view),
                    "ranked": ranked_view,
                },
            )

            # ── Step 4: Targeted OA evidence passages ─────────────
            evidence = fetch_oa_evidence_passages(
                literature_refs, observations, max_refs=min(5, len(literature_refs))
            )
            self._literature_evidence = evidence
            provenance["steps"]["4_evidence_passages"] = {
                "n_refs_examined": len(evidence),
                "evidence": evidence,
            }
            self._log_literature_step(
                "4_retrieve_targeted_evidence",
                f"Literature Step 4 — Retrieve targeted evidence passages ({len(evidence)} papers)",
                {
                    "n_refs_examined": len(evidence),
                    "evidence": [
                        {
                            "title": e.get("title"),
                            "pmid": e.get("pmid"),
                            "doi": e.get("doi"),
                            "pmcid": e.get("pmcid"),
                            "evidence_source": e.get("evidence_source"),
                            "n_passages": len(e.get("passages") or []),
                            "passages": e.get("passages") or [],
                        }
                        for e in evidence
                    ],
                },
            )

            # ── Step 5: Compare evidence with MD results ──────────
            comparisons = classify_evidence_vs_observations(observations, evidence)
            self._literature_comparisons = comparisons
            provenance["steps"]["5_compare_evidence"] = {
                "n_observations": len(comparisons),
                "comparisons": comparisons,
            }
            self._log_literature_step(
                "5_compare_evidence_with_md",
                "Literature Step 5 — Compare evidence with MD observations",
                {
                    "comparisons": comparisons,
                    "summary": {
                        cat: sum(1 for c in comparisons if c.get("category") == cat)
                        for cat in ("supported", "plausible", "conflicting", "not_addressed")
                    },
                },
            )

            # ── Step 6 prep: ranked list handed to report writer ──
            provenance["steps"]["6_candidates_for_report"] = {
                "n": len(literature_refs),
                "note": (
                    "These ranked candidates are passed to the literature-review LLM; "
                    "uncited papers are pruned after writing (see Step 6 final selection)."
                ),
                "candidates": ranked_view,
            }
            self._log_literature_step(
                "6_generate_evidence_linked_report_prep",
                f"Literature Step 6 prep — {len(literature_refs)} ranked candidates ready for evidence-linked report",
                {"n_candidates_for_llm": len(literature_refs), "candidates": ranked_view},
            )

            # Persist machine-readable provenance next to the report
            try:
                from pathlib import Path as _P
                wd = state.get("working_directory") or ""
                if wd:
                    prov_path = _P(wd) / "reporter" / "literature_provenance.json"
                    write_literature_provenance(str(prov_path), provenance)
                    self._literature_provenance_path = str(prov_path)
            except Exception:
                logger.debug("literature provenance json write skipped", exc_info=True)

            self._literature_provenance = provenance

        except Exception as e:
            logger.warning(f"Literature search failed: {e}", exc_info=True)

        return literature_refs[:MAX_REFS]

    def _sync_literature_citations(
        self,
        literature_refs: List[Dict[str, Any]],
        literature_review: Optional[str],
        final_impression: Optional[str],
    ) -> tuple:
        """Prune uncited refs and renumber [n]; log final selection provenance."""
        from src.reporter.literature_search import sync_cited_literature
        from src.reporter.literature_evidence import compact_ref, write_literature_provenance

        before = list(literature_refs or [])
        refs, review, impression = sync_cited_literature(
            literature_refs,
            literature_review=literature_review,
            final_impression=final_impression,
        )
        cited_view = [compact_ref(r, include_score=False) for r in (refs or [])]
        self._log_literature_step(
            "6_final_selection",
            f"Literature Step 6 — Final cited selection ({len(cited_view)}/{len(before)} kept)",
            {
                "n_candidates_before_cite_prune": len(before),
                "n_cited": len(cited_view),
                "cited": cited_view,
                "candidates_before_prune": [compact_ref(r, include_score=False) for r in before],
            },
        )
        try:
            prov = getattr(self, "_literature_provenance", None) or {}
            prov.setdefault("steps", {})["6_final_cited"] = {
                "n_cited": len(cited_view),
                "cited": cited_view,
            }
            self._literature_provenance = prov
            prov_path = getattr(self, "_literature_provenance_path", None)
            if prov_path:
                write_literature_provenance(prov_path, prov)
        except Exception:
            logger.debug("literature provenance final update skipped", exc_info=True)
        return refs, review, impression

    def _retrieved_context(self, state: MDState, query: str = "") -> str:
        try:
            from agentic.campaign.agent_context import agent_context_block

            return agent_context_block(state, "reporter", query=query[:1200])
        except Exception:
            logger.debug("reporter retrieval skipped", exc_info=True)
            return ""

    def _generate_literature_review(
        self,
        analysis_data: Dict[str, Any],
        literature_refs: List[Dict[str, Any]],
        state: MDState,
        is_combined: bool = False,
    ) -> Optional[str]:
        """Write a contextual literature review linking papers to simulation results."""
        if not literature_refs:
            return None

        from src.reporter.literature_search import (
            build_analysis_summary_text,
            extract_analysis_stats_from_entries,
            extract_research_context,
            score_literature_ref_relevance,
        )
        from src.reporter.figure_selector import (
            build_figure_catalog_for_prompt,
            select_analysis_entries,
            summarize_findings_for_literature,
            ReportFigurePolicy,
        )

        user_goal = (
            state.get("user_goal_original")
            or state.get("master_enriched_prompt")
            or state.get("enriched_prompt")
            or state.get("user_goal")
            or ""
        )
        if not isinstance(user_goal, str):
            user_goal = str(user_goal)
        report_focus_raw = state.get("report_focus") or ""
        report_focus = (
            " ".join(report_focus_raw)
            if isinstance(report_focus_raw, list)
            else str(report_focus_raw)
        )
        policy = ReportFigurePolicy.from_config(
            getattr(self, "config", None),
            report_type=state.get("report_type", "comprehensive"),
        )
        selected_entries = select_analysis_entries(
            analysis_data.get("entries") or [],
            user_goal=user_goal,
            report_focus=report_focus,
            enriched_prompt=state.get("enriched_prompt") or "",
            policy=policy,
        )
        focused_findings = summarize_findings_for_literature(
            analysis_data, selected_entries=selected_entries
        )
        figure_catalog = build_figure_catalog_for_prompt(selected_entries)
        analysis_stats = extract_analysis_stats_from_entries(analysis_data)
        analysis_types = list({e.get("analysis_type") for e in selected_entries if e.get("analysis_type")})
        if not analysis_types:
            analysis_types = list((analysis_data.get("analysis_types") or {}).keys())
            if isinstance(analysis_data.get("analysis_types"), list):
                analysis_types = analysis_data["analysis_types"]

        sys_info = state.get("system_info") if isinstance(state.get("system_info"), dict) else {}
        from src.reporter.protein_identity import resolve_protein_identity
        identity = resolve_protein_identity(state, user_goal=user_goal)
        resolved_protein = (
            identity.get("gene_name")
            or identity.get("protein_name")
            or identity.get("display_name")
        )
        context = extract_research_context(
            user_goal=user_goal,
            protein_name=resolved_protein or sys_info.get("protein_name"),
            analysis_types=analysis_types,
            analysis_stats=analysis_stats,
            protein_family=identity.get("protein_family"),
            related_terms=identity.get("related_terms"),
            aliases=identity.get("aliases") or [],
            gene_name=identity.get("gene_name") or None,
            uniprot_id=identity.get("uniprot_id") or None,
        )
        analysis_text = build_analysis_summary_text(analysis_data, analysis_stats)

        lit_parts = []
        for i, ref in enumerate(literature_refs[:8], 1):
            title = ref.get("title", "Unknown")
            abstract = ref.get("abstract") or ""
            score, note = score_literature_ref_relevance(ref, context)
            snippet = abstract[:700] if abstract else "(abstract not available — title/metadata only)"
            lit_parts.append(
                f"[{i}] {title} (relevance score {score}; {note})\n    {snippet}"
            )
        lit_text = "\n".join(lit_parts)

        observations = getattr(self, "_literature_observations", None) or []
        evidence = getattr(self, "_literature_evidence", None) or []
        comparisons = getattr(self, "_literature_comparisons", None) or []
        obs_text = "\n".join(
            f"- ({o.get('id')}) {o.get('statement')}" for o in observations
        ) or "- (none extracted)"
        evidence_text_parts = []
        for e in evidence[:5]:
            passages = e.get("passages") or []
            if not passages:
                continue
            evidence_text_parts.append(
                f"- {e.get('title')} [{e.get('evidence_source')}; pmid={e.get('pmid')}]\n"
                + "\n".join(f"    • {p}" for p in passages[:2])
            )
        evidence_text = "\n".join(evidence_text_parts) or "- (no OA passages retrieved; use abstracts above)"
        cmp_text = "\n".join(
            f"- {c.get('observation')} → {c.get('category')}"
            f" (supporting={len(c.get('supporting') or [])}, conflicting={len(c.get('conflicting') or [])})"
            for c in comparisons
        ) or "- (no comparison records)"

        protein_focus = identity.get("gene_name") or ", ".join(context.get("protein_names") or []) or "the simulated protein"
        alias_list = list(identity.get("aliases") or context.get("aliases") or [])
        alias_note = ""
        if alias_list:
            alias_note = (
                f"\n**Known aliases / synonyms:** {', '.join(alias_list[:8])} "
                "(papers may use any of these names — or none, if they discuss the family/mechanism)"
            )
        family_note = ""
        if identity.get("protein_family"):
            family_note = (
                f"\n**Protein family / class:** {identity['protein_family']} "
                f"(prefer {protein_focus}-named papers, but also use closely related "
                f"family / mechanism papers that explain the observed dynamics)"
            )
        objective = context.get("user_goal") or user_goal or "MD simulation analysis"
        ligand_focus = ", ".join(context.get("ligand_terms") or []) or "ligand binding"
        region_focus = ", ".join(context.get("region_terms") or []) or "active site / activation loop"
        findings_focus = focused_findings or "; ".join(context.get("finding_phrases") or []) or "see analysis summary"

        if not self.llm.available:
            return (
                f"Literature was searched for studies relating to {protein_focus} dynamics, "
                f"{ligand_focus}, and apo/holo comparisons. "
                f"{len(literature_refs)} publications were retrieved and ranked by abstract/title "
                f"relevance to the performed analyses ({', '.join(analysis_types[:6]) or 'RMSD, RMSF'}). "
                f"See the reference list below for experimental and computational precedents "
                f"that contextualise the simulation findings summarised above."
            )

        combined_note = (
            "This is a multi-simulation comparison. "
            "Prioritise papers that discuss the named systems and the measurements "
            "that actually appear in the results."
            if is_combined else ""
        )
        retrieved = self._retrieved_context(state, query=objective)
        figures_block = (
            "\n".join(figure_catalog)
            if figure_catalog
            else "(none numbered yet — do not invent Figure/Table citations)"
        )

        prompt = f"""You are a computational biophysics expert writing an evidence-linked Literature Review for an MD simulation report.

**User's Research Question / Objective:**
{objective}

**Protein(s) / System Focus:** {protein_focus}{alias_note}{family_note}
**Ligand / state focus:** {ligand_focus} (apo vs holo where applicable)
**Structural region focus:** {region_focus}
{combined_note}
**Key simulation findings to contextualise (report-selected analyses):**
{findings_focus}

**Figures available in this report (cite only these if you mention figures):**
{figures_block}

**MD observations requiring interpretation (Step 1):**
{obs_text}

**Simulation Analysis Results (quantitative summary):**
{analysis_text}

**Evidence vs MD comparison (Step 5 categories: supported / plausible / conflicting / not_addressed):**
{cmp_text}

**Targeted evidence passages (Step 4; prefer these over generic paraphrases):**
{evidence_text}

**Candidate Publications (titles/abstracts for screening; relevance score provided):**
{lit_text}

{retrieved}

**Instructions (evidence-grounded scientific inference):**
- Prefer papers that name {protein_focus}, but do NOT limit the review to that gene alone
- Also cite literature relevant to the objective, methods ({', '.join(analysis_types[:8]) or 'MD dynamics'}), and observed results
- For each major claim, ground the interpretation in a specific observation above and cite the paper whose evidence supports it
- Explicitly distinguish: (a) findings supported by prior studies, (b) plausible hypotheses, (c) conflicting evidence, (d) observations not addressed in the retrieved literature
- Use the targeted evidence passages when available; do not invent experimental details absent from abstracts/passages
- Write 3–4 paragraphs of substantive scientific prose — mechanistic, not a paper list
- Cite with bracket notation [1], [2], etc. matching candidate numbers above
- REQUIRED: at least four in-text citations, mixing protein-specific and objective/results-relevant papers when both exist
- Cite in increasing order of first mention; every cited paper must be discussed
- Do NOT invent Figure or Table numbers (no "Figure 1", "Table S1", etc.) — plots are numbered separately in the HTML report
- Do NOT include a "References" subsection or section headers — output only the review paragraphs"""

        try:
            response = self.llm.prompt(prompt)
            log_llm_interaction(
                agent_name="reporter.literature_review",
                prompt=prompt[:500] + "...",
                response=response[:500] + "..." if len(response) > 500 else response,
                is_mock=not self.llm.available,
            )
            cleaned = response.strip()
            cleaned = re.sub(r'^#{1,3}\s+.*\n?', '', cleaned, flags=re.MULTILINE).strip()
            if len(cleaned) < 80:
                logger.warning("Literature review too short, using fallback")
                return None
            logger.info("Generated contextual literature review (%d chars)", len(cleaned))
            return cleaned
        except Exception as e:
            logger.warning("Failed to generate literature review: %s", e)
            return None

    def _generate_combined_final_impression(
        self,
        analysis_data: Dict[str, Any],
        literature_refs: List[Dict[str, Any]],
        state: MDState,
        sim_dirs: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Generate combined final impression with literature citations."""
        from src.reporter.combined_reporter import _build_combined_final_impression, _collect_sim_summaries
        from src.reporter.literature_search import (
            build_analysis_summary_text,
            extract_analysis_stats_from_entries,
            extract_research_context,
        )

        from src.reporter.figure_selector import resolve_report_narrative

        user_goal = (
            state.get("user_goal_original")
            or state.get("master_enriched_prompt")
            or state.get("enriched_prompt")
            or state.get("user_goal")
            or ""
        )
        if not isinstance(user_goal, str):
            user_goal = str(user_goal)
        narrative = resolve_report_narrative(
            user_goal,
            enriched_prompt=state.get("enriched_prompt") or "",
            n_simulations=len(sim_dirs or labels or []),
        )
        analysis_stats = extract_analysis_stats_from_entries(analysis_data)
        analysis_types = list((analysis_data.get("analysis_types") or {}).keys())
        if isinstance(analysis_data.get("analysis_types"), list):
            analysis_types = analysis_data["analysis_types"]

        sys_info = state.get("system_info") if isinstance(state.get("system_info"), dict) else {}
        context = extract_research_context(
            user_goal=user_goal,
            protein_name=sys_info.get("protein_name") if sys_info else None,
            analysis_types=analysis_types,
            analysis_stats=analysis_stats,
        )
        context["report_theme"] = narrative.primary_theme
        analysis_text = build_analysis_summary_text(analysis_data, analysis_stats)
        protein_focus = ", ".join(context.get("protein_names") or []) or "the simulated systems"

        if sim_dirs and labels:
            sims_summary = _collect_sim_summaries(sim_dirs, labels)
            sim_resources = [
                {"label": lbl, "report_focus": ""} for lbl in labels
            ]
            fallback = _build_combined_final_impression(sims_summary, sim_resources)
        else:
            fallback = (
                "Cross-simulation comparison summarises structural stability, flexibility, "
                "ligand-binding behaviour, and activation-loop dynamics across all systems."
            )

        if not self.llm.available:
            return fallback

        lit_text = "No literature references available."
        if literature_refs:
            lit_parts = []
            for i, ref in enumerate(literature_refs[:10], 1):
                title = ref.get("title", "Unknown")
                abstract = ref.get("abstract", "")
                snippet = abstract[:350] if abstract else "No abstract"
                lit_parts.append(f"[{i}] {title}\n    {snippet}")
            lit_text = "\n".join(lit_parts)
        retrieved = self._retrieved_context(state, query=user_goal)

        prompt = f"""You are a computational biophysics expert writing the Combined Final Impression for a multi-simulation MD report.

**User's Research Question / Objective:**
{user_goal}

**Report narrative theme:** {context.get('report_theme', 'general dynamics comparison')}
**Protein(s) studied:** {protein_focus}
**Ligand / comparison focus:** {", ".join(context.get("ligand_terms") or ["ATP", "apo", "holo"])}
**Activation-loop / region focus:** {", ".join(context.get("region_terms") or ["residues 150-200"])}

**Simulation Analysis Results:**
{analysis_text}

**Draft synthesis (rule-based baseline — refine and enrich this):**
{fallback}

**Literature References (use for contextual comparison; cite as [1], [2], ...):**
{lit_text}

{retrieved}

**Instructions:**
- Write 3–5 paragraphs that tell one coherent story aligned with the user's objective
- Quote numbers that appear in the retrieved study notes or the analysis results; do not replace them with a generic stability summary
- If the goal emphasises unsupervised classification or clustering, lead with how simulations group by dynamic regime (FEL/phylogenetic tree), what distinguishes each cluster (COM distance, contacts, pocket RMSF, ligand flexibility), and biological interpretation — do not repeat generic RMSD/Rg lists unless they support the classification story
- If the goal emphasises apo vs holo or binding, integrate ligand–pocket coupling, contacts, and flexibility trends with literature
- Cite ONLY literature that is directly relevant to these proteins / this study using bracket notation [1], [2], etc. matching the candidate list
- REQUIRED: include at least two in-text citations like [1] or [2] when candidate papers are provided
- Prefer citing papers already appropriate for the Literature Review; introduce citations in increasing order of first mention in THIS section
- Do not cite a paper unless you discuss its finding in the prose
- End with concise implications and suggested follow-up experiments or simulations
- Do NOT include section headers — output only the impression paragraphs"""

        try:
            response = self.llm.prompt(prompt)
            log_llm_interaction(
                agent_name="reporter.combined_final_impression",
                prompt=prompt[:500] + "...",
                response=response[:500] + "..." if len(response) > 500 else response,
                is_mock=not self.llm.available,
            )
            cleaned = response.strip()
            cleaned = re.sub(r'^#{1,3}\s+.*\n?', '', cleaned, flags=re.MULTILINE).strip()
            if len(cleaned) < 120:
                logger.warning("Combined final impression too short, using fallback")
                return fallback
            logger.info("Generated combined final impression (%d chars)", len(cleaned))
            return cleaned
        except Exception as exc:
            logger.warning("Failed to generate combined final impression: %s", exc)
            return fallback

    def _resolve_protein_display_name(self, state: MDState) -> tuple:
        """Return (protein_name, sim_label) for report title and metadata."""
        from src.reporter.protein_identity import resolve_protein_identity

        workdir = state.get("working_directory", "")
        sim_label = Path(workdir).name if workdir else state.get("sim_label", "")
        identity = resolve_protein_identity(state, sim_label=sim_label)
        display = identity.get("display_name") or identity.get("gene_name") or "Protein"
        return display, sim_label

    def _extract_pdb_for_viewer(self, state: MDState, analysis_data: Dict[str, Any] = None) -> Optional[Dict[str, str]]:
        """Extract significant structures for the 3D viewer.

        Combines FEL basin representative PDBs with trajectory frames at
        analysis-driven time points (COM extrema, contact extrema, unbinding,
        RMSD peaks).  Falls back to first frame only when nothing else works.
        """
        try:
            from src.reporter.structure_extractor import (
                collect_significant_structures,
                extract_first_frame_pdb,
                read_pdb_data,
            )

            working_dir = state.get("working_directory", "working_dir")

            from pathlib import Path as _Path
            hpc_raw = state.get("hpc_output_directory") or state.get("hpc_dir")
            if hpc_raw and _Path(hpc_raw).is_dir():
                _hpc_p = _Path(hpc_raw)
                hpc_parent = str(_hpc_p.parent)
                hpc_subdir_name = _hpc_p.name
            else:
                hpc_parent = working_dir
                hpc_subdir_name = "hpc"

            if analysis_data:
                frames = collect_significant_structures(
                    analysis_data,
                    working_dir,
                    hpc_parent,
                    hpc_subdir=hpc_subdir_name,
                    max_total=8,
                )
                if frames:
                    log_agent_action(
                        "reporter",
                        f"Collected {len(frames)} structures for 3D viewer",
                        {"labels": list(frames.keys())},
                    )
                    return frames

            pdb_path = extract_first_frame_pdb(hpc_parent, hpc_subdir=hpc_subdir_name)
            if pdb_path:
                pdb_text = read_pdb_data(pdb_path)
                if pdb_text:
                    logger.info("Extracted single first-frame PDB for 3D viewer (fallback)")
                    log_agent_action("reporter", "Extracted first-frame PDB for 3D viewer (fallback)", {})
                    return {"0 ns — Start": pdb_text}

            logger.warning("Could not extract any PDB for 3D viewer")
        except Exception as e:
            logger.warning("PDB extraction for 3D viewer failed: %s", e, exc_info=True)
        return None

    def _generate_final_impression(
        self,
        analysis_data: Dict[str, Any],
        literature_refs: List[Dict[str, Any]],
        state: MDState
    ) -> Optional[str]:
        """Generate LLM-based final impression correlating analysis results with literature."""
        
        entries = analysis_data.get("entries", []) if isinstance(analysis_data, dict) else []
        if not entries:
            logger.info("No analysis entries; skipping final impression")
            return None

        if not self.llm.available:
            logger.info("LLM unavailable; generating rule-based final impression")
            lines = [
                "This molecular dynamics simulation was analysed with the following results:"
            ]
            for entry in entries:
                atype = entry.get("analysis_type", "Unknown")
                stats = entry.get("statistics", {})
                if isinstance(stats, dict) and stats:
                    stat_parts = [
                        f"{k}: {v}"
                        for k, v in stats.items()
                        if isinstance(v, (int, float, str)) and str(v).strip()
                    ]
                    lines.append(
                        f"**{atype}**: " + ("; ".join(stat_parts) if stat_parts else "completed")
                    )
                else:
                    lines.append(f"**{atype}**: Analysis completed")
            return "\n\n".join(lines)
        
        from src.reporter.literature_search import (
            build_analysis_summary_text,
            extract_analysis_stats_from_entries,
            extract_research_context,
            score_literature_ref_relevance,
        )
        from src.reporter.figure_selector import (
            build_figure_catalog_for_prompt,
            grounded_numeric_claims,
            select_analysis_entries,
            summarize_findings_for_literature,
            ReportFigurePolicy,
        )
        from src.reporter.protein_identity import resolve_protein_identity

        user_goal = (
            state.get("user_goal_original")
            or state.get("master_enriched_prompt")
            or state.get("enriched_prompt")
            or state.get("user_goal", "MD simulation analysis")
        )
        identity = resolve_protein_identity(state, user_goal=user_goal)
        protein_label = (
            identity.get("gene_name")
            or identity.get("protein_name")
            or identity.get("display_name")
            or "the protein"
        )

        policy = ReportFigurePolicy.from_config(
            getattr(self, "config", None),
            report_type=state.get("report_type", "comprehensive"),
        )
        selected_entries = select_analysis_entries(
            analysis_data.get("entries") or [],
            user_goal=user_goal,
            report_focus="",
            enriched_prompt=state.get("enriched_prompt") or "",
            policy=policy,
        )
        focused_findings = summarize_findings_for_literature(
            analysis_data, selected_entries=selected_entries
        )
        figure_catalog = build_figure_catalog_for_prompt(selected_entries)
        grounded_vals = grounded_numeric_claims(selected_entries)
        analysis_stats = extract_analysis_stats_from_entries(analysis_data)
        context = extract_research_context(
            user_goal=user_goal,
            protein_name=protein_label,
            analysis_types=list({e.get("analysis_type") for e in selected_entries if e.get("analysis_type")}),
            analysis_stats=analysis_stats,
            protein_family=identity.get("protein_family"),
            related_terms=identity.get("related_terms"),
        )
        analysis_text = build_analysis_summary_text(analysis_data, analysis_stats)
        
        # Build literature summary for LLM
        lit_text = "No literature references available."
        if literature_refs:
            lit_parts = []
            for i, ref in enumerate(literature_refs[:10], 1):
                title = ref.get("title", "Unknown")
                abstract = ref.get("abstract") or ""
                score, note = score_literature_ref_relevance(ref, context)
                abstract_snippet = abstract[:400] if abstract else "No abstract"
                lit_parts.append(
                    f"[{i}] {title} (relevance {score}; {note})\n    {abstract_snippet}"
                )
            lit_text = "\n".join(lit_parts)
        
        user_goal_display = (
            state.get("user_goal_original")
            or state.get("enriched_prompt")
            or state.get("user_goal", "MD simulation analysis")
        )
        findings_block = focused_findings or "; ".join(context.get("finding_phrases") or [])
        retrieved = self._retrieved_context(state, query=user_goal_display)
        grounded_block = (
            "\n".join(f"- {v}" for v in grounded_vals)
            if grounded_vals
            else "(none extracted)"
        )
        figures_block = (
            "\n".join(figure_catalog)
            if figure_catalog
            else "(none — do not invent Figure citations)"
        )
        
        prompt = f"""You are a computational biophysics expert. Based on the analysis results and literature below, write a concise Final Impression section for a scientific MD simulation report.

**User Goal:** {user_goal_display}

**Protein studied:** {protein_label} ({identity.get('uniprot_id') or 'accession unknown'})
**Protein class:** {identity.get('protein_family') or 'see literature'}

**Key findings (goal-selected analyses):**
{findings_block or analysis_text}

**Analysis Results (quantitative):**
{analysis_text}

**Grounded numeric values you MAY quote (do not invent others):**
{grounded_block}

**Figures available in this report (cite only these if you mention figures):**
{figures_block}

**Literature References:**
{lit_text}

{retrieved}

**Instructions:**
- Write specifically about {protein_label}. Use related-system commentary only when the results or retrieved notes support it
- Quote numbers ONLY from the grounded numeric values / analysis results above — do not invent residues, means, or maxima
- Correlate the analysis results with findings from the literature on {protein_label} or closely related systems
- Highlight the most important observations: stability, flexible regions, ligand effects, FEL basins, activation-loop behaviour, interface H-bond occupancy, salt bridges, and named residue–residue interaction partners — whichever appear in the results
- Draw a clear conclusion that answers the user's research question
- Provide actionable insights or suggested follow-up experiments
- Keep it concise: 3–4 paragraphs, no bullet points
- Write in scientific prose, suitable for a report
- Cite relevant literature using bracket notation like [1], [2], [3] corresponding to the reference numbers above
- Do NOT invent Figure or Table numbers — cite only figures listed above
- Do NOT include section headers or titles — just the text content"""

        try:
            response = self.llm.prompt(prompt)
            
            log_llm_interaction(
                agent_name="reporter.final_impression",
                prompt=prompt[:500] + "...",
                response=response[:500] + "..." if len(response) > 500 else response,
                is_mock=not self.llm.available
            )
            
            # Clean up response: remove markdown headers if LLM adds them
            cleaned = response.strip()
            cleaned = re.sub(r'^#{1,3}\s+.*\n?', '', cleaned, flags=re.MULTILINE).strip()
            
            if len(cleaned) < 50:
                logger.warning("Final impression too short, skipping")
                return None
            
            logger.info(f"Generated final impression ({len(cleaned)} chars)")
            return cleaned
            
        except Exception as e:
            logger.warning(f"Failed to generate final impression: {e}")
            return None
    
    def _extract_plan_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON plan from LLM response.
        
        Returns parsed dict with 'steps' key, or None if parsing fails.
        """
        
        def _clean_json_text(text: str) -> str:
            """Strip JS-style comments and trailing commas that LLMs sometimes add."""
            # Remove single-line // comments (but not inside strings)
            cleaned = re.sub(r'(?<=[,\}\]\s])\s*//[^\n]*', '', text)
            # Remove trailing commas before } or ]
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            return cleaned
        
        def _try_parse(text: str) -> Optional[Dict[str, Any]]:
            """Try to parse JSON, with and without comment stripping."""
            for candidate in [text, _clean_json_text(text)]:
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict) and "steps" in result:
                        return result
                except json.JSONDecodeError:
                    continue
            return None
        
        # Strategy 1: Strip outer markdown fences and parse directly
        content_cleaned = re.sub(r'^\s*```(?:json)?\s*\n', '', content)
        content_cleaned = re.sub(r'\n\s*```\s*$', '', content_cleaned)
        content_cleaned = content_cleaned.strip()
        
        result = _try_parse(content_cleaned)
        if result:
            return result
        
        # Strategy 2: The LLM sometimes wraps valid JSON inside a markdown
        # code block WITHIN the response text. Extract the ```json...``` block.
        fenced_match = re.search(r'```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```', content)
        if fenced_match:
            result = _try_parse(fenced_match.group(1))
            if result:
                return result
        
        # Strategy 3: Find the outermost JSON object via regex
        json_match = re.search(r'\{[\s\S]*\}', content_cleaned, re.DOTALL)
        if json_match:
            result = _try_parse(json_match.group())
            if result:
                return result
        
        # Strategy 4: Balanced-brace extraction around "steps" key
        inner_match = re.search(r'\{[^{}]*"steps"\s*:\s*\[', content)
        if inner_match:
            start = inner_match.start()
            depth = 0
            for i in range(start, len(content)):
                if content[i] == '{':
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        result = _try_parse(content[start:i+1])
                        if result:
                            return result
                        break
        
        logger.error("Could not extract valid JSON plan from LLM response")
        logger.debug(f"Response content: {content[:500]}...")
        return None
    
    def _create_fallback_plan(self, agent_input: ReporterAgentInput) -> ReporterPlan:
        """Create fallback plan when LLM is unavailable"""
        
        steps = []
        
        # Step 1: Read analysis summary
        steps.append(ReporterStep(
            name="Read Analysis Summary",
            description="Parse analysis_summary.jsonl file",
            tool_name="read_analysis_summary",
            tool_params={
                "summary_file": agent_input.analysis_summary_file
            },
            reason="Load analysis results"
        ))
        
        # Step 2: Generate literature queries (if enabled)
        if agent_input.literature_search:
            steps.append(ReporterStep(
                name="Generate Literature Queries",
                description="Create PubMed search queries",
                tool_name="generate_literature_queries",
                tool_params={
                    "analysis_types": ["RMSD", "RMSF"],  # Will be updated based on actual analyses
                    "user_goal": agent_input.user_goal
                },
                reason="Prepare for literature search"
            ))
        
        # Step 3: Generate HTML report
        steps.append(ReporterStep(
            name="Generate HTML Report",
            description="Create HTML report from analysis data",
            tool_name="generate_html_report",
            tool_params={
                "analysis_data": {},  # Populated during execution
                "report_type": agent_input.report_type.value,
                "output_file": "report.html"
            },
            reason="Generate final report"
        ))
        
        return ReporterPlan(
            reasoning="Fallback plan: Basic report generation without LLM planning",
            overview="Read analysis, optionally search literature, generate HTML report",
            steps=steps,
            report_focus=["Analysis results", "Key statistics"],
            literature_queries=[],
            estimated_complexity="medium"
        )
    
    def _save_execution_plan(self, plan: ReporterPlan, agent_input: ReporterAgentInput) -> None:
        """Save execution plan with LLM reasoning to file"""
        try:
            plan_file = os.path.join(self.file_manager.agent_dir, "execution_plan.json")
            
            plan_data = {
                "timestamp": str(Path(plan_file).stat().st_mtime if os.path.exists(plan_file) else "N/A"),
                "reasoning": plan.reasoning,
                "overview": plan.overview,
                "report_focus": plan.report_focus,
                "literature_queries": plan.literature_queries,
                "estimated_complexity": plan.estimated_complexity,
                "steps": [
                    {
                        "name": step.name,
                        "description": step.description,
                        "tool_name": step.tool_name,
                        "tool_params": step.tool_params,
                        "reason": step.reason
                    }
                    for step in plan.steps
                ]
            }
            
            with open(plan_file, "w") as f:
                json.dump(plan_data, f, indent=2)
            
            logger.info(f"Saved execution plan to {plan_file}")
            
            # Also save a markdown version for readability
            md_file = os.path.join(self.file_manager.agent_dir, "execution_plan.md")
            with open(md_file, "w") as f:
                f.write("# Reporter Execution Plan\n\n")
                f.write(f"**Generated:** {plan_data['timestamp']}\n\n")
                f.write(f"## LLM Reasoning\n\n{plan.reasoning}\n\n")
                f.write(f"## Overview\n\n{plan.overview}\n\n")
                f.write(f"## Report Focus\n\n")
                for focus in plan.report_focus:
                    f.write(f"- {focus}\n")
                f.write(f"\n## Execution Steps ({len(plan.steps)} steps)\n\n")
                for i, step in enumerate(plan.steps, 1):
                    f.write(f"### Step {i}: {step.name}\n\n")
                    f.write(f"**Tool:** `{step.tool_name}`\n\n")
                    f.write(f"**Description:** {step.description}\n\n")
                    f.write(f"**Reason:** {step.reason}\n\n")
                    f.write(f"**Parameters:**\n```json\n{json.dumps(step.tool_params, indent=2)}\n```\n\n")
            
            logger.info(f"Saved execution plan (markdown) to {md_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save execution plan: {e}")
    
    def _execute_reporter_plan(
        self,
        agent_input: ReporterAgentInput,
        plan: ReporterPlan,
        state: MDState
    ) -> ReporterResult:
        """Execute reporter plan step by step"""
        
        completed = 0
        issues = []
        warnings = []
        step_results = []
        
        # Shared data between steps
        analysis_data = {}
        literature_refs = []
        literature_review = None
        
        # --- Hardcoded: read analysis summary first so we know the types ---
        logger.info("  [exec] Reading analysis_summary.jsonl...")
        try:
            base_dir = str(Path(self.file_manager.agent_dir).parent)
            summary_result = self.tool_executor.execute_tool(
                "read_analysis_summary",
                {"summary_file": "analysis_summary.jsonl", "working_dir": base_dir}
            )
            if isinstance(summary_result, dict) and summary_result.get("entries"):
                analysis_data = summary_result
                logger.info("  [exec] Pre-loaded analysis summary (%d entries, types: %s)",
                            len(analysis_data.get("entries", [])),
                            ", ".join(analysis_data.get("analysis_types", {}).keys()
                                      if isinstance(analysis_data.get("analysis_types"), dict)
                                      else analysis_data.get("analysis_types", [])))
        except Exception as e:
            logger.warning("Pre-load of analysis summary failed: %s", e)

        # --- Hardcoded: always run literature search before report generation ---
        logger.info(
            "  [exec] Running evidence-grounded literature survey "
            "(PubMed + Europe PMC + Crossref + arXiv/bioRxiv + UniProt)..."
        )
        literature_refs = self._ensure_literature_search(analysis_data, state)
        logger.info("  [exec] Literature search done: %d ranked references", len(literature_refs))
        literature_review = self._generate_literature_review(analysis_data, literature_refs, state)
        log_agent_action(
            "reporter",
            f"Literature survey completed: {len(literature_refs)} ranked candidates "
            f"(see Literature Step 1–6 provenance in this log)",
            {
                "n_ranked": len(literature_refs),
                "n_observations": len(getattr(self, "_literature_observations", []) or []),
                "n_evidence_records": len(getattr(self, "_literature_evidence", []) or []),
                "provenance_json": getattr(self, "_literature_provenance_path", None),
            },
        )
        
        total_steps = len(plan.steps)
        for i, step in enumerate(plan.steps):
            try:
                # Skip plan-based literature steps — _ensure_literature_search
                # already ran with protein-aware queries before the loop.
                if step.tool_name in ("search_pubmed", "generate_literature_queries", "search_biorxiv"):
                    logger.info("Skipping plan step '%s' (%s) — literature already collected",
                                step.name, step.tool_name)
                    log_agent_action("reporter", f"Step {i+1}/{total_steps} skipped (pre-handled)", {
                        "step": step.name,
                        "tool": step.tool_name,
                        "reason": f"Literature already collected by _ensure_literature_search ({len(literature_refs)} refs)",
                        "status": "⏭️ SKIPPED"
                    })
                    step_results.append({
                        "step_name": step.name,
                        "tool": step.tool_name,
                        "success": True,
                        "result": {"skipped": True, "reason": "literature handled by _ensure_literature_search"}
                    })
                    completed += 1
                    continue

                # Skip steps with no tool (virtual/narrative steps the LLM invented)
                if not step.tool_name:
                    logger.info("Skipping virtual plan step '%s' (no tool)", step.name)
                    log_agent_action("reporter", f"Step {i+1}/{total_steps} skipped (virtual step)", {
                        "step": step.name,
                        "reason": "No tool — handled internally (e.g. final impression generated during HTML report step)",
                        "status": "⏭️ SKIPPED"
                    })
                    step_results.append({
                        "step_name": step.name,
                        "tool": "",
                        "success": True,
                        "result": {"skipped": True, "reason": "no tool associated"}
                    })
                    completed += 1
                    continue

                log_agent_action("reporter", f"Executing step {i+1}/{total_steps}", {
                    "step": step.name,
                    "tool": step.tool_name
                })
                
                # Special handling for steps that need previous results
                params = step.tool_params.copy()
                effective_tool = step.tool_name
                
                # Override working_dir based on tool type:
                # - Output tools (generate_html_report): Use reporter's agent directory
                # - Input tools (read_analysis_summary): Use base working directory to find analysis files
                effective_tool = step.tool_name
                if step.tool_name == "generate_combined_html_report":
                    logger.warning(
                        "Plan requested generate_combined_html_report for per-sim run; "
                        "redirecting to generate_html_report"
                    )
                    effective_tool = "generate_html_report"

                if effective_tool == "generate_html_report":
                    # Output tool - use reporter directory  
                    params["working_dir"] = self.file_manager.agent_dir
                    logger.debug(f"Set working_dir to {self.file_manager.agent_dir} for output tool {effective_tool}")
                elif step.tool_name == "read_analysis_summary":
                    # Input tool - use base directory to find analysis files
                    base_dir = Path(self.file_manager.agent_dir).parent
                    params["working_dir"] = str(base_dir)
                    logger.debug(f"Set working_dir to {base_dir} for input tool {step.tool_name}")
                
                if effective_tool == "generate_html_report":
                    from src.reporter.figure_selector import ReportFigurePolicy
                    # Force a consistent, general filename for every per-sim
                    # report so combined-level discovery/aggregation is trivial.
                    # The LLM plan may propose arbitrary names (e.g.
                    # "kinase_report.html"); we always normalize to report.html.
                    params["output_file"] = "report.html"
                    params["analysis_data"] = analysis_data
                    # Pass system info from state
                    params["system_info"] = state.get("system_info")
                    protein_name, sim_label = self._resolve_protein_display_name(state)
                    params["protein_name"] = protein_name
                    params["sim_label"] = sim_label
                    # Analysis outputs live in parent/analysis — used to resolve plot paths
                    params["analysis_dir"] = str(Path(self.file_manager.agent_dir).parent / "analysis")
                    _fig_policy = ReportFigurePolicy.from_config(
                        self.config,
                        report_type=agent_input.report_type.value if hasattr(agent_input.report_type, "value") else str(agent_input.report_type),
                        include_visualizations=agent_input.include_visualizations,
                    )
                    params["user_goal"] = (
                        state.get("user_goal_original")
                        or state.get("user_goal")
                        or agent_input.user_goal
                    )
                    params["report_focus"] = " ".join(
                        getattr(plan, "report_focus", None) or state.get("report_focus") or []
                    ) if isinstance(getattr(plan, "report_focus", None) or state.get("report_focus"), list) else (
                        getattr(plan, "report_focus", None) or state.get("report_focus") or ""
                    )
                    params["max_figures_per_section"] = _fig_policy.max_figures_per_section
                    params["include_visualizations"] = agent_input.include_visualizations
                    # Generate final impression using LLM
                    logger.info("  [exec] Generating LLM final impression...")
                    final_impression = self._generate_final_impression(
                        analysis_data, literature_refs, state
                    )
                    literature_refs, literature_review, final_impression = (
                        self._sync_literature_citations(
                            literature_refs, literature_review, final_impression
                        )
                    )
                    params["literature_refs"] = literature_refs
                    params["literature_review"] = literature_review
                    params["final_impression"] = final_impression
                    logger.info("  [exec] Final impression: %s",
                                "generated" if params["final_impression"] else "skipped/unavailable")
                    # Extract PDB frames at analysis-driven timepoints for 3D viewer
                    logger.info("  [exec] Extracting PDB frames for 3D viewer (analysis-driven)...")
                    pdb_frames = self._extract_pdb_for_viewer(state, analysis_data)
                    params["pdb_data"] = pdb_frames  # Dict[str, str] or None
                    logger.info("  [exec] PDB frames: %s",
                                f"{len(pdb_frames)} timepoints ({', '.join(pdb_frames.keys())})"
                                if pdb_frames else "none")
                    # Pass enriched prompt (supervisor-rephrased user goal)
                    params["enriched_prompt"] = state.get("enriched_prompt") or state.get("user_goal")
                
                # Execute tool
                result = self.tool_executor.execute_tool(effective_tool, params)
                
                # Store results for next steps; only update if the new result has
                # entries — don't discard a good pre-loaded analysis_data with an
                # empty result from a duplicate plan step.
                if step.tool_name == "read_analysis_summary":
                    if isinstance(result, dict) and result.get("entries"):
                        analysis_data = result
                    else:
                        logger.warning(
                            "Plan-step read_analysis_summary returned no entries; "
                            "keeping pre-loaded analysis_data (%d entries)",
                            len(analysis_data.get("entries", [])),
                        )
                
                step_results.append({
                    "step_name": step.name,
                    "tool": effective_tool if effective_tool != step.tool_name else step.tool_name,
                    "success": result.get("success", True),
                    "result": result
                })
                
                completed += 1
                
                log_agent_action("reporter", f"Step {i+1}/{total_steps} completed", {
                    "step": step.name,
                    "tool": step.tool_name,
                    "status": "✅ SUCCESS"
                })
                
            except Exception as e:
                logger.error(f"Step {step.name} failed: {e}", exc_info=True)
                issues.append(f"{step.name}: {str(e)}")
                step_results.append({
                    "step_name": step.name,
                    "tool": step.tool_name,
                    "success": False,
                    "error": str(e)
                })
                
                log_agent_action("reporter", f"Step {i+1}/{total_steps} failed", {
                    "step": step.name,
                    "tool": step.tool_name,
                    "status": "❌ FAILED",
                    "error": str(e)
                })
        
        # Guarantee literature search: if no refs collected, run searches now
        if not literature_refs and analysis_data:
            literature_refs = self._ensure_literature_search(analysis_data, state)
        if literature_refs and not literature_review:
            literature_review = self._generate_literature_review(
                analysis_data, literature_refs, state
            )

        # Fallback: ensure HTML report exists even if the LLM plan chose the wrong tool
        report_file_from_steps = None
        for sr in step_results:
            if sr.get("tool") == "generate_html_report" and sr.get("success"):
                report_file_from_steps = sr.get("result", {}).get("report_file")

        if not report_file_from_steps and analysis_data.get("entries"):
            logger.warning("No HTML report produced by plan — running fallback generate_html_report")
            from src.reporter.figure_selector import ReportFigurePolicy
            protein_name, sim_label = self._resolve_protein_display_name(state)
            _fig_policy = ReportFigurePolicy.from_config(
                self.config,
                report_type=agent_input.report_type.value if hasattr(agent_input.report_type, "value") else str(agent_input.report_type),
                include_visualizations=agent_input.include_visualizations,
            )
            _fb_impression = self._generate_final_impression(
                analysis_data, literature_refs, state
            )
            literature_refs, literature_review, _fb_impression = (
                self._sync_literature_citations(
                    literature_refs, literature_review, _fb_impression
                )
            )
            fallback_params = {
                "working_dir": self.file_manager.agent_dir,
                "analysis_data": analysis_data,
                "literature_refs": literature_refs,
                "literature_review": literature_review,
                "system_info": state.get("system_info"),
                "protein_name": protein_name,
                "sim_label": sim_label,
                "analysis_dir": str(Path(self.file_manager.agent_dir).parent / "analysis"),
                "user_goal": state.get("user_goal_original") or state.get("user_goal") or agent_input.user_goal,
                "report_focus": " ".join(plan.report_focus) if isinstance(plan.report_focus, list) else (plan.report_focus or ""),
                "max_figures_per_section": _fig_policy.max_figures_per_section,
                "include_visualizations": agent_input.include_visualizations,
                "final_impression": _fb_impression,
                "pdb_data": self._extract_pdb_for_viewer(state, analysis_data),
                "enriched_prompt": state.get("enriched_prompt") or state.get("user_goal"),
                "output_file": "report.html",
                "report_type": agent_input.report_type.value if hasattr(agent_input.report_type, "value") else str(agent_input.report_type),
            }
            fallback_result = self.tool_executor.execute_tool("generate_html_report", fallback_params)
            step_results.append({
                "step_name": "Fallback HTML Report",
                "tool": "generate_html_report",
                "success": fallback_result.get("success", False),
                "result": fallback_result,
            })
            if fallback_result.get("success"):
                completed += 1
            else:
                issues.append(f"Fallback HTML report: {fallback_result.get('error', 'unknown error')}")
        
        # Build report
        report_lines = ["=" * 80, "REPORTER EXECUTION REPORT", "=" * 80, ""]
        report_lines.append(f"Completed: {completed}/{len(plan.steps)} steps")
        report_lines.append("")
        
        if step_results:
            report_lines.append("STEPS EXECUTED:")
            for result in step_results:
                status = "✅" if result.get("success") else "❌"
                report_lines.append(f"  {status} {result.get('step_name')}")
        
        if issues:
            report_lines.append("")
            report_lines.append("ISSUES:")
            for issue in issues:
                report_lines.append(f"  ❌ {issue}")
        
        report_lines.append("=" * 80)
        report = "\n".join(report_lines)
        
        logger.info(report)
        
        # Save execution report to files
        self._save_execution_report(plan, step_results, analysis_data, literature_refs, issues, warnings)
        
        # Extract report file path
        report_file = None
        for result in step_results:
            if result.get("tool") == "generate_html_report" and result.get("success"):
                report_file = result.get("result", {}).get("report_file")
        
        # Add comprehensive summary file to generated files
        comprehensive_file = os.path.join(self.file_manager.agent_dir, "comprehensive_summary.json")
        generated_files = {"html": report_file} if report_file else {}
        if os.path.exists(comprehensive_file):
            generated_files["comprehensive_summary"] = comprehensive_file
        
        return ReporterResult(
            success=len(issues) == 0,
            completed_steps=completed,
            total_steps=len(plan.steps),
            report_file=report_file,
            analysis_summaries=[],  # Populated from analysis_data if needed
            literature_references=literature_refs,
            report_sections=[],
            key_findings=[],
            recommendations=[],
            issues=issues,
            warnings=warnings,
            report=report,
            generated_files=generated_files,
            step_results=step_results
        )
    
    def _save_execution_report(
        self,
        plan: ReporterPlan,
        step_results: List[Dict[str, Any]],
        analysis_data: Dict[str, Any],
        literature_refs: List[Dict[str, Any]],
        issues: List[str],
        warnings: List[str]
    ) -> None:
        """Save comprehensive execution report with LLM reasoning and results"""
        try:
            import datetime
            
            timestamp = datetime.datetime.now().isoformat()
            
            # 1. Save comprehensive summary in JSON format
            comprehensive_data = {
                "timestamp": timestamp,
                "llm_plan": {
                    "reasoning": plan.reasoning,
                    "overview": plan.overview,
                    "report_focus": plan.report_focus,
                    "literature_queries": plan.literature_queries,
                    "estimated_complexity": plan.estimated_complexity
                },
                "execution": {
                    "total_steps": len(plan.steps),
                    "completed_steps": sum(1 for r in step_results if r.get("success")),
                    "failed_steps": sum(1 for r in step_results if not r.get("success")),
                    "success_rate": f"{sum(1 for r in step_results if r.get('success')) / len(step_results) * 100:.1f}%" if step_results else "0%"
                },
                "step_results": step_results,
                "analysis_data_summary": {
                    "entries_found": len(analysis_data.get("entries", [])) if isinstance(analysis_data, dict) else 0,
                    "analysis_types": analysis_data.get("analysis_types", []) if isinstance(analysis_data, dict) else []
                },
                "literature_references": len(literature_refs),
                "issues": issues,
                "warnings": warnings
            }
            
            json_file = os.path.join(self.file_manager.agent_dir, "comprehensive_summary.json")
            with open(json_file, "w") as f:
                json.dump(comprehensive_data, f, indent=2)
            
            logger.info(f"Saved comprehensive summary (JSON) to {json_file}")
            
            # 2. Save execution report in Markdown format
            md_file = os.path.join(self.file_manager.agent_dir, "execution_report.md")
            with open(md_file, "w") as f:
                f.write(f"# Reporter Execution Report\\n\\n")
                f.write(f"**Generated:** {timestamp}\\n\\n")
                f.write(f"---\\n\\n")
                
                # LLM Planning Section
                f.write(f"## LLM Planning\\n\\n")
                f.write(f"### Reasoning\\n\\n{plan.reasoning}\\n\\n")
                f.write(f"### Overview\\n\\n{plan.overview}\\n\\n")
                f.write(f"### Report Focus Areas\\n\\n")
                for focus in plan.report_focus:
                    f.write(f"- {focus}\\n")
                f.write("\\n")
                
                if plan.literature_queries:
                    f.write(f"### Literature Search Queries\\n\\n")
                    for query in plan.literature_queries:
                        f.write(f"- {query}\\n")
                    f.write("\\n")
                
                # Execution Summary
                f.write(f"## Execution Summary\\n\\n")
                f.write(f"- **Total Steps:** {len(plan.steps)}\\n")
                f.write(f"- **Completed:** {comprehensive_data['execution']['completed_steps']}\\n")
                f.write(f"- **Failed:** {comprehensive_data['execution']['failed_steps']}\\n")
                f.write(f"- **Success Rate:** {comprehensive_data['execution']['success_rate']}\\n\\n")
                
                # Step-by-step results
                f.write(f"## Step-by-Step Execution\\n\\n")
                for i, result in enumerate(step_results, 1):
                    status = "\u2705 SUCCESS" if result.get("success") else "\u274c FAILED"
                    f.write(f"### Step {i}: {result.get('step_name')} [{status}]\\n\\n")
                    f.write(f"**Tool:** `{result.get('tool')}`\\n\\n")
                    
                    if result.get("success"):
                        tool_result = result.get("result", {})
                        if isinstance(tool_result, dict):
                            if "entries" in tool_result:
                                f.write(f"- Entries found: {len(tool_result['entries'])}\\n")
                            if "analysis_types" in tool_result:
                                f.write(f"- Analysis types: {', '.join(tool_result['analysis_types'])}\\n")
                            if "results" in tool_result:
                                f.write(f"- Results: {len(tool_result.get('results', []))} items\\n")
                            if "report_file" in tool_result:
                                f.write(f"- Report file: `{tool_result['report_file']}`\\n")
                    else:
                        f.write(f"**Error:** {result.get('error', 'Unknown error')}\\n")
                    
                    f.write("\\n")
                
                # Analysis Data Summary
                if analysis_data:
                    f.write(f"## Analysis Data Summary\\n\\n")
                    if isinstance(analysis_data, dict):
                        f.write(f"- **Entries:** {len(analysis_data.get('entries', []))}\\n")
                        f.write(f"- **Analysis Types:** {', '.join(analysis_data.get('analysis_types', []))}\\n")
                        
                        if "statistics" in analysis_data:
                            f.write(f"\\n### Statistics\\n\\n")
                            stats = analysis_data["statistics"]
                            for key, value in stats.items():
                                f.write(f"- **{key}:** {value}\\n")
                    f.write("\\n")
                
                # Literature References
                if literature_refs:
                    f.write(f"## Literature References ({len(literature_refs)})\\n\\n")
                    for i, ref in enumerate(literature_refs[:10], 1):  # Show first 10
                        title = ref.get("title", "Unknown")
                        authors = ref.get("authors", "Unknown authors")
                        f.write(f"{i}. **{title}**\\n")
                        f.write(f"   - Authors: {authors}\\n")
                        if "journal" in ref:
                            f.write(f"   - Journal: {ref['journal']}\\n")
                        f.write("\\n")
                    
                    if len(literature_refs) > 10:
                        f.write(f"*...and {len(literature_refs) - 10} more references*\\n\\n")
                
                # Issues and Warnings
                if issues:
                    f.write(f"## Issues\\n\\n")
                    for issue in issues:
                        f.write(f"- \u274c {issue}\\n")
                    f.write("\\n")
                
                if warnings:
                    f.write(f"## Warnings\\n\\n")
                    for warning in warnings:
                        f.write(f"- \u26a0\ufe0f {warning}\\n")
                    f.write("\\n")
                
                f.write(f"---\\n\\n")
                f.write(f"*Report generated by Reporter Agent with LLM planning*\\n")
            
            logger.info(f"Saved execution report (Markdown) to {md_file}")
            
            # 3. Save plain text execution log
            txt_file = os.path.join(self.file_manager.agent_dir, "execution_log.txt")
            with open(txt_file, "w") as f:
                f.write("=" * 80 + "\\n")
                f.write("REPORTER AGENT EXECUTION LOG\\n")
                f.write("=" * 80 + "\\n\\n")
                f.write(f"Timestamp: {timestamp}\\n\\n")
                
                f.write("LLM REASONING:\\n")
                f.write("-" * 80 + "\\n")
                f.write(plan.reasoning + "\\n\\n")
                
                f.write("EXECUTION STEPS:\\n")
                f.write("-" * 80 + "\\n")
                for i, result in enumerate(step_results, 1):
                    status = "OK" if result.get("success") else "FAIL"
                    f.write(f"[{status}] Step {i}: {result.get('step_name')}\\n")
                
                f.write("\\n")
                f.write("=" * 80 + "\\n")
                f.write(f"Completed: {comprehensive_data['execution']['completed_steps']}/{len(plan.steps)} steps\\n")
                f.write("=" * 80 + "\\n")
            
            logger.info(f"Saved execution log (TXT) to {txt_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save execution report: {e}", exc_info=True)
    
    def _update_state(self, state: MDState, output: ReporterAgentOutput) -> None:
        """Update workflow state with reporter results"""
        
        state["reporter_output"] = {
            "success": output.success,
            "report_path": output.report_path,
            "pdf_path": output.pdf_path,
            "key_findings": output.result.key_findings,
            "recommendations": output.result.recommendations,
            "generated_files": output.result.generated_files
        }
        
        # Register all generated files
        if output.report_path:
            self.file_manager.register_external_file(
                file_path=output.report_path,
                file_type="html_report",
                description="Scientific analysis report (HTML)"
            )
        
        # Register comprehensive summary and execution reports
        for file_type, file_path in output.result.generated_files.items():
            if file_type != "html" and os.path.exists(file_path):
                self.file_manager.register_external_file(
                    file_path=file_path,
                    file_type=f"reporter_{file_type}",
                    description=f"Reporter {file_type.replace('_', ' ').title()}"
                )
        
        # Also register execution plan and report files
        agent_dir = self.file_manager.agent_dir
        for filename, description in [
            ("execution_plan.json", "Execution plan with LLM reasoning (JSON)"),
            ("execution_plan.md", "Execution plan with LLM reasoning (Markdown)"),
            ("comprehensive_summary.json", "Comprehensive execution summary (JSON)"),
            ("execution_report.md", "Detailed execution report (Markdown)"),
            ("execution_log.txt", "Execution log (Text)")
        ]:
            file_path = os.path.join(agent_dir, filename)
            if os.path.exists(file_path):
                self.file_manager.register_external_file(
                    file_path=file_path,
                    file_type=f"reporter_{filename.replace('.', '_')}",
                    description=description
                )
        
        if output.pdf_path:
            self.file_manager.register_external_file(
                file_path=output.pdf_path,
                file_type="pdf_report",
                description="Scientific analysis report (PDF)"
            )
        
        # Add errors to state
        if output.errors:
            state["errors"].extend(output.errors)
