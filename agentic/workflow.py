"""
LLM-Powered MD Workflow with Intelligent Routing

This module implements the complete MD workflow that uses LLM reasoning
for dynamic routing and agent coordination.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END
from .state import MDState
from .supervisor import MDSupervisor
from .preprocess import PreprocessingAgent
from .simsetup import SimulationSetupAgent
from .human_checkpoints import HumanCheckpoints
from .llm import LLMClient
from .hpc import MDHPCAgent
from .analysis import MDAnalysisAgent
from .planner import MDPlanner
from .reporter import ReporterAgent
from .utils.state_persistence import compact_state_for_persistence

logger = logging.getLogger(__name__)

_MULTISIM_SKIP_DIRS = frozenset({
    "supervisor", "planner", "programmer", "analysis", "reporter",
})


def _discover_multisim_labels(base_dir: str) -> list:
    """Return subdirectory labels that look like per-simulation workspaces."""
    base = Path(base_dir)
    if not base.is_dir():
        return []
    labels: list = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name in _MULTISIM_SKIP_DIRS:
            continue
        if (d / "supervisor" / "state.jsonl").exists() or (d / "analysis").is_dir():
            labels.append(d.name)
    return labels


def _multisim_per_sim_analysis_complete(base_dir: str) -> bool:
    """True when every discovered sim label has non-empty analysis output."""
    labels = _discover_multisim_labels(base_dir)
    if not labels:
        return False
    for label in labels:
        adir = Path(base_dir) / label / "analysis"
        if not adir.is_dir():
            return False
        if not any(adir.iterdir()):
            return False
    return True


def _multisim_per_sim_workflow_complete(base_dir: str, progress: Optional[Dict[str, Any]] = None) -> bool:
    """True when every sim has finished required agents (progress-aware)."""
    from agentic.multi_sim_progress import multisim_workflow_incomplete

    if progress:
        return not multisim_workflow_incomplete(progress)
    state_path = Path(base_dir) / "supervisor" / "state.jsonl"
    if state_path.is_file():
        try:
            entry = json.loads(state_path.read_text(encoding="utf-8"))
            saved_progress = (entry.get("state") or {}).get("multi_sim_progress")
            if saved_progress:
                return not multisim_workflow_incomplete(saved_progress)
        except Exception:
            pass
    return _multisim_per_sim_analysis_complete(base_dir)


class MDWorkflow:
    """
    Complete MD workflow with LLM-powered supervisor and agent coordination.
    
    HIERARCHY:
    - Supervisor: Orchestrates workflow, routes to agents
    - Planner: Creates detailed execution plans, coordinates programmer
    - Programmer: Generates scripts for planner (internal to planner)
    - Field Agents: Execute supervisor-assigned tasks (preprocess, setup, hpc, analysis)
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        if llm_client is None:
            raise ValueError("llm_client is required. Pass LLMClient from main script.")
        self.llm = llm_client
        
        # Initialize supervisor with LLM capabilities
        self.supervisor = MDSupervisor(llm_client=self.llm)
        
        # Initialize planner (planner will initialize programmer internally)
        self.planner = MDPlanner(llm_client=self.llm)
        
        # Initialize field-specific agents
        self.preprocessor = PreprocessingAgent(self.llm)
        self.setup_agent = SimulationSetupAgent(self.llm)
        self.hpc_agent = MDHPCAgent(self.llm)
        self.analysis_agent = MDAnalysisAgent(self.llm)
        self.reporter_agent = ReporterAgent(self.llm)
        self.checkpoints = HumanCheckpoints()
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("MD workflow initialized with supervisor → planner → programmer hierarchy")
    
    # Nodes that produce meaningful artifacts worth saving incrementally
    _SAVE_AFTER_NODES = frozenset({
        "input_validation", "planner", "preprocess",
        "setup", "hpc", "analysis", "reporter",
    })

    def _tag_llm_agent(self, node_name: str) -> None:
        agent_for_usage = {
            "input_validation": "supervisor",
            "planner": "planner",
            "preprocess": "preprocess",
            "setup": "simsetup",
            "hpc": "hpc",
            "analysis": "analysis",
            "reporter": "reporter",
            "verify": "verify",
            "human_checkpoint": "hitl",
        }.get(node_name, node_name)
        try:
            if self.llm is not None and hasattr(self.llm, "set_agent"):
                self.llm.set_agent(agent_for_usage)
        except Exception:
            pass

    def _finalize_node(self, node_name: str, result: MDState) -> MDState:
        """Shared post-node work for graph ``run()`` and HITL loop (no drift)."""
        try:
            from agentic.campaign.stage_tick import mark_stage_done

            # Deterministic verify before marking setup/analysis complete
            if node_name == "setup" and (
                result.get("coordinates") or result.get("topology")
            ):
                from agentic.campaign.verify import apply_verify_to_state

                v = apply_verify_to_state(result, "simsetup")
                if v.get("ok"):
                    result["setup_science_ok"] = True
                    mark_stage_done(result, "simsetup")
                else:
                    result["setup_science_ok"] = False
                    # Force retry path — do not advance as if setup succeeded
                    if result.get("next_node") in (
                        "human_setup_check",
                        "supervisor",
                        "hpc",
                    ):
                        result["next_node"] = "supervisor"
            elif node_name == "preprocess" and result.get("cleaned_pdb"):
                mark_stage_done(result, "preprocessing")
            elif node_name == "hpc" and (
                result.get("job_id")
                or result.get("reuse_hpc")
                or result.get("hpc_output_directory")
            ):
                mark_stage_done(result, "hpc")
            elif node_name == "analysis":
                from agentic.campaign.verify import apply_verify_to_state

                v = apply_verify_to_state(result, "analysis")
                if v.get("ok"):
                    result["analysis_science_ok"] = True
                    mark_stage_done(result, "analysis")
                else:
                    result["analysis_science_ok"] = False
                    if result.get("next_node") in (
                        "human_analysis_check",
                        "supervisor",
                        "reporter",
                    ):
                        result["next_node"] = "supervisor"
            elif node_name == "reporter" and result.get("reporter_output"):
                mark_stage_done(result, "reporter")
            if node_name in {
                "preprocess", "setup", "hpc", "analysis", "reporter"
            }:
                from src.analysis.inventory import write_stage_inventory_for_state

                stage = {
                    "preprocess": "preprocessing",
                    "setup": "simsetup",
                }.get(node_name, node_name)
                write_stage_inventory_for_state(result, stage)
        except Exception:
            logger.debug("stage tick / verify finalize skipped", exc_info=True)
        if node_name in self._SAVE_AFTER_NODES:
            try:
                if result.get("is_multi_simulation"):
                    from agentic.multi_sim_progress import (
                        ensure_per_sim_working_directory,
                        mark_sim_pipeline_stage,
                        workflow_sim_label_for_hitl,
                    )

                    if node_name == "input_validation" and result.get("input_validated"):
                        sim_label = (
                            workflow_sim_label_for_hitl(result)
                            or (result.get("multi_sim_progress") or {}).get(
                                "active_sim_label"
                            )
                        )
                        if sim_label:
                            mark_sim_pipeline_stage(
                                result, sim_label, input_validated=True
                            )
                    elif node_name == "planner" and result.get("execution_plan"):
                        sim_label = (
                            workflow_sim_label_for_hitl(result)
                            or (result.get("multi_sim_progress") or {}).get(
                                "active_sim_label"
                            )
                        )
                        if sim_label:
                            mark_sim_pipeline_stage(
                                result,
                                sim_label,
                                planned=True,
                                enriched=True,
                            )
                    ensure_per_sim_working_directory(result)
                self._save_progress(result, node_name)
            except Exception as exc:
                logger.debug(f"Incremental save after {node_name} failed: {exc}")
        return result

    def _wrap_node(self, node_name: str, node_func: Callable) -> Callable:
        """Wrap a node function to track current_node and save progress."""
        def wrapped_node(state: MDState) -> MDState:
            state["current_node"] = node_name
            self._tag_llm_agent(node_name)
            result = node_func(state)
            if result is None:
                logger.error(f"Node '{node_name}' returned None – returning input state as fallback")
                return state
            return self._finalize_node(node_name, result)
        return wrapped_node

    def _invoke_node(self, node_name: str, state: MDState) -> MDState:
        """Invoke a node with the same finalize path as the compiled graph (HITL)."""
        dispatch = {
            "supervisor": self.supervisor.supervisor_node,
            "input_validation": self.supervisor.input_validation_node,
            "planner": self.planner.planner_node,
            "preprocess": self.preprocessor.preprocess_node,
            "setup": self.setup_agent.setup_node,
            "hpc": self.hpc_agent.hpc_node,
            "analysis": self.analysis_agent.analysis_node,
            "reporter": self.reporter_agent.reporter_node,
            "human_preprocess_check": self.checkpoints.human_preprocess_check,
            "human_setup_check": self.checkpoints.human_setup_check,
            "human_hpc_check": self.checkpoints.human_hpc_check,
            "human_hpc_pool_check": self.checkpoints.human_hpc_pool_check,
            "human_analysis_check": self.checkpoints.human_analysis_check,
            "human_reporter_check": self.checkpoints.human_reporter_check,
            "hpc_pool_wait": self._hpc_pool_wait_node,
            "parallel_pool_wait": self._parallel_pool_wait_node,
            "final_report": self._final_report_node,
            "workflow_incomplete": self._workflow_incomplete_node,
            "verify": self._verify_node,
        }
        fn = dispatch.get(node_name)
        if fn is None:
            logger.error("Unknown workflow node for invoke: %s", node_name)
            state.setdefault("errors", []).append(f"Unknown workflow node: {node_name}")
            state["next_node"] = "workflow_incomplete"
            return state
        state["current_node"] = node_name
        self._tag_llm_agent(node_name)
        result = fn(state)
        if result is None:
            logger.error("Node '%s' returned None – returning input state", node_name)
            return state
        # Human checkpoints already received feedback; still share inventory/save path
        # for field agents via finalize. Skip double-finalize for terminal nodes.
        if node_name in {"final_report", "workflow_incomplete"}:
            return result
        if node_name.startswith("human_"):
            # Save only — stage ticks already applied after field agents
            if node_name in self._SAVE_AFTER_NODES or True:
                try:
                    self._save_progress(result, node_name)
                except Exception:
                    pass
            return result
        if node_name in {"hpc_pool_wait", "parallel_pool_wait", "supervisor", "verify"}:
            if node_name == "verify":
                return self._finalize_node(node_name, result)
            try:
                if node_name != "supervisor":
                    self._save_progress(result, node_name)
            except Exception:
                pass
            return result
        return self._finalize_node(node_name, result)

    def _verify_node(self, state: MDState) -> MDState:
        """Standalone verify node (optional explicit graph hop)."""
        stage = state.get("verify_stage") or "analysis"
        from agentic.campaign.verify import apply_verify_to_state

        result = apply_verify_to_state(state, str(stage))
        if result.get("ok"):
            state["next_node"] = state.get("verify_return_node") or "supervisor"
        else:
            state["next_node"] = "supervisor"
        state.pop("verify_stage", None)
        state.pop("verify_return_node", None)
        return state

    def _workflow_incomplete_node(self, state: MDState) -> MDState:
        """Fail-closed terminal when routing is invalid or campaign incomplete."""
        state["workflow_status"] = "incomplete"
        state.setdefault("errors", []).append(
            state.get("routing_error")
            or "Workflow ended incomplete (fail-closed routing)"
        )
        logger.error(
            "workflow_incomplete: %s",
            state.get("errors", [])[-1] if state.get("errors") else "incomplete",
        )
        try:
            from agentic.campaign.memory import remember_episode

            remember_episode(
                state=state,
                stage="supervisor",
                error=str(state.get("errors", [])[-1]),
                tool="routing",
                fix="Inspect next_node / multi_sim_progress; resume after fix",
            )
        except Exception:
            pass
        return self._final_report_node(state)

    @staticmethod
    def _compute_recursion_limit(state: MDState) -> int:
        """
        LangGraph step budget for one ``invoke()`` call.

        This counts **graph node transitions** (supervisor → agent → wait → …),
        **not** the number of LLM chat calls. Each pool poll / routing hop costs
        one step toward this limit.

        Scale with remaining per-sim work and keep a floor large enough for
        family-scale campaigns (≥50 simulations).
        """
        reuse = bool(state.get("reuse_hpc"))
        if not reuse:
            try:
                from agentic.multi_sim_hpc_pool import reuse_hpc_enabled

                reuse = reuse_hpc_enabled(state)
            except Exception:
                reuse = False

        n_campaign = max(
            len(state.get("sim_prompts") or []),
            len(state.get("pdb_list") or []),
            len((state.get("multi_sim_progress") or {}).get("sim_order") or []),
            1,
        )
        # Always size floors for at least 50-sim family tasks.
        n_scale = max(n_campaign, 50)

        if state.get("multi_sim_phase") == "hpc_pool":
            pool = state.get("hpc_pool") or {}
            interval = int(
                pool.get("check_interval_sec")
                or state.get("hpc_check_interval_sec")
                or 3600
            )
            # supervisor + hpc_pool_wait per poll; allow ~7 days of checks.
            polls = max(48, int(7 * 86400 / max(interval, 60)))
            # --reuse-hpc uses short waits (≈15s); keep a high family-scale floor.
            floor = 20 * n_scale + 200 if reuse else 120
            return min(max(polls * 2 + 40, floor), 50000)

        if not state.get("is_multi_simulation"):
            return 40

        progress = state.get("multi_sim_progress") or {}
        sim_order = progress.get("sim_order") or []
        sims = progress.get("sims") or {}
        agents = progress.get("required_agents") or ["analysis", "reporter"]
        remaining = 0
        for label in sim_order:
            rec = sims.get(label) or {}
            if rec.get("status") == "done":
                continue
            agent_map = rec.get("agents") or {}
            if any(agent_map.get(a) != "done" for a in agents):
                remaining += 1
            elif rec.get("status") != "done":
                remaining += 1
        if not sim_order:
            remaining = n_campaign

        steps_per_sim = 12
        combined_headroom = 50 if state.get("run_combined_analysis") else 0
        # Parallel prep/analysis: parent only ticks the pool while workers run
        # for many minutes — each wait cycle burns steps, so budget generously.
        if state.get("multi_sim_phase") == "parallel_pool":
            steps_per_sim = 30
            combined_headroom = max(combined_headroom, 150)
            pool = state.get("parallel_pool") or {}
            interval = int(
                state.get("parallel_pool_poll_sec")
                or pool.get("poll_sec")
                or 30
            )
            # supervisor + parallel_pool_wait per poll; allow ~7 days for
            # family-scale analysis (50 sims × long modular tools).
            polls = max(48, int(7 * 86400 / max(interval, 15)))
            poll_budget = polls * 2 + 100
            family_floor = 30 * n_scale + 500
            return min(max(poll_budget, family_floor, 5000), 50000)

        budget = steps_per_sim * max(remaining, 1) + combined_headroom
        # Family-scale floor: enough for ≥50 sims even if "remaining" is undercounted
        # on resume (missing progress / dropped reuse_hpc flag).
        family_floor = 30 * n_scale + 300
        if reuse or n_campaign >= 10:
            floor = max(family_floor, 2000)
        else:
            floor = max(120, 12 * n_campaign + 50)
        return min(max(budget, floor), 50000)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with proper agent hierarchy."""
        
        # Create the graph with our state
        workflow = StateGraph(MDState)
        
        # Add all nodes with wrapper to track current_node
        workflow.add_node("supervisor", self._wrap_node("supervisor", self.supervisor.supervisor_node))
        workflow.add_node("input_validation", self._wrap_node("input_validation", self.supervisor.input_validation_node))
        workflow.add_node("planner", self._wrap_node("planner", self.planner.planner_node))
        workflow.add_node("preprocess", self._wrap_node("preprocess", self.preprocessor.preprocess_node))
        workflow.add_node("setup", self._wrap_node("setup", self.setup_agent.setup_node))
        workflow.add_node("hpc", self._wrap_node("hpc", self.hpc_agent.hpc_node))
        workflow.add_node("analysis", self._wrap_node("analysis", self.analysis_agent.analysis_node))
        workflow.add_node("reporter", self._wrap_node("reporter", self.reporter_agent.reporter_node))
        workflow.add_node("human_preprocess_check", self._wrap_node("human_preprocess_check", self.checkpoints.human_preprocess_check))
        workflow.add_node("human_setup_check", self._wrap_node("human_setup_check", self.checkpoints.human_setup_check))
        workflow.add_node("human_hpc_check", self._wrap_node("human_hpc_check", self.checkpoints.human_hpc_check))
        workflow.add_node("human_hpc_pool_check", self._wrap_node("human_hpc_pool_check", self.checkpoints.human_hpc_pool_check))
        workflow.add_node("human_analysis_check", self._wrap_node("human_analysis_check", self.checkpoints.human_analysis_check))
        workflow.add_node("human_reporter_check", self._wrap_node("human_reporter_check", self.checkpoints.human_reporter_check))
        workflow.add_node("final_report", self._wrap_node("final_report", self._final_report_node))
        workflow.add_node(
            "workflow_incomplete",
            self._wrap_node("workflow_incomplete", self._workflow_incomplete_node),
        )
        workflow.add_node("verify", self._wrap_node("verify", self._verify_node))
        workflow.add_node("hpc_pool_wait", self._wrap_node("hpc_pool_wait", self._hpc_pool_wait_node))
        workflow.add_node("parallel_pool_wait", self._wrap_node("parallel_pool_wait", self._parallel_pool_wait_node))
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # ========== SUPERVISOR ROUTING ==========
        # Supervisor can route to: input_validation, planner, or field agents
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "input_validation": "input_validation",
                "planner": "planner",
                "preprocess": "preprocess", 
                "setup": "setup",
                "hpc": "hpc",
                "analysis": "analysis",
                "reporter": "reporter",
                "verify": "verify",
                "human_reporter_check": "human_reporter_check",
                "human_analysis_check": "human_analysis_check",
                "human_hpc_pool_check": "human_hpc_pool_check",
                "hpc_pool_wait": "hpc_pool_wait",
                "parallel_pool_wait": "parallel_pool_wait",
                "final_report": "final_report",
                "workflow_incomplete": "workflow_incomplete",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "parallel_pool_wait",
            lambda state: state.get("next_node", "supervisor"),
            {"supervisor": "supervisor"},
        )

        workflow.add_conditional_edges(
            "hpc_pool_wait",
            lambda state: state.get("next_node", "supervisor"),
            {"supervisor": "supervisor"},
        )

        workflow.add_conditional_edges(
            "human_hpc_pool_check",
            lambda state: state.get("next_node", "supervisor"),
            {"supervisor": "supervisor", "human_hpc_pool_check": "human_hpc_pool_check"},
        )
        
        # ========== INPUT VALIDATION ==========
        # Always routes back to supervisor
        workflow.add_conditional_edges(
            "input_validation",
            lambda state: state.get("next_node", "supervisor"),
            {"supervisor": "supervisor"}
        )
        
        # ========== PLANNER (handles programmer internally) ==========
        # Planner only routes back to supervisor
        # Programmer is called internally by planner, never appears in graph
        workflow.add_conditional_edges(
            "planner",
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor"
            }
        )
        
        # ========== FIELD AGENTS - PREPROCESSING ==========
        workflow.add_conditional_edges(
            "preprocess",
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor",
                "human_preprocess_check": "human_preprocess_check"
            }
        )
        
        # Human preprocessing checkpoint
        workflow.add_conditional_edges(
            "human_preprocess_check",
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor",
                "preprocess": "preprocess"
            }
        )
        
        # ========== FIELD AGENTS - SETUP ==========
        workflow.add_conditional_edges(
            "setup", 
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor",
                "human_setup_check": "human_setup_check"
            }
        )
        
        # Human setup checkpoint
        workflow.add_conditional_edges(
            "human_setup_check",
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor",
                "setup": "setup"
            }
        )
        
        # ========== FIELD AGENTS - HPC ==========
        workflow.add_conditional_edges(
            "hpc",
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor",
                "human_hpc_check": "human_hpc_check"
            }
        )
        
        # Human HPC checkpoint
        workflow.add_conditional_edges(
            "human_hpc_check",
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor",
                "hpc": "hpc"
            }
        )
        
        # ========== FIELD AGENTS - ANALYSIS ==========
        # ========== FIELD AGENTS - ANALYSIS ==========
        workflow.add_conditional_edges(
            "analysis",
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor",
                "human_analysis_check": "human_analysis_check",
                # Combined analysis may advance straight to reporter.
                "reporter": "reporter",
            }
        )
        
        # Human analysis checkpoint
        workflow.add_conditional_edges(
            "human_analysis_check",
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor",
                "analysis": "analysis"
            }
        )
        
        # ========== FIELD AGENTS - REPORTER ==========
        # Reporter routes to human_reporter_check (auto-skipped unless --HITL).
        workflow.add_conditional_edges(
            "reporter",
            lambda state: state.get("next_node", "human_reporter_check"),
            {
                "supervisor": "supervisor",
                "human_reporter_check": "human_reporter_check",
            }
        )

        # ========== REPORTER CHECKPOINT ==========
        # Skipped unless --HITL (routes supervisor → final_report).
        # In HITL mode: supports reporter re-run, analysis re-run, checkpoint switching.
        workflow.add_conditional_edges(
            "human_reporter_check",
            lambda state: state.get("next_node", "supervisor"),
            {
                "supervisor": "supervisor",
                "analysis": "analysis",
                "reporter": "reporter",
                "human_reporter_check": "human_reporter_check",  # waiting for input
                # Earlier HITL checkpoints the user can switch to
                "human_preprocess_check": "human_preprocess_check",
                "human_setup_check": "human_setup_check",
                "human_hpc_check": "human_hpc_check",
                "human_analysis_check": "human_analysis_check",
            }
        )
        
        # ========== VERIFY (optional explicit hop) ==========
        workflow.add_conditional_edges(
            "verify",
            lambda state: state.get("next_node", "supervisor"),
            {"supervisor": "supervisor", "workflow_incomplete": "workflow_incomplete"},
        )

        # ========== FINAL REPORT / FAIL-CLOSED ==========
        workflow.add_edge("final_report", END)
        workflow.add_edge("workflow_incomplete", END)
        
        return workflow.compile()
    
    def _route_from_supervisor(self, state: MDState) -> str:
        """
        Route from supervisor based on next_node field.
        
        Supervisor decides which field agent to invoke or whether to create plan.
        Invalid routes fail closed to ``workflow_incomplete`` (never silent success).
        """
        next_node = state.get("next_node")
        
        valid_nodes = [
            "input_validation", "planner", "preprocess", "setup", 
            "hpc", "analysis", "reporter", "verify",
            "human_reporter_check",
            "human_analysis_check",
            "human_hpc_pool_check", "hpc_pool_wait", "parallel_pool_wait",
            "final_report", "workflow_incomplete",
        ]
        
        if next_node in valid_nodes:
            logger.info(f"Supervisor routing to: {next_node}")
            return next_node
        
        state["routing_error"] = (
            f"Invalid next_node={next_node!r} — fail-closed (not final_report)"
        )
        state.setdefault("errors", []).append(state["routing_error"])
        logger.error(state["routing_error"])
        return "workflow_incomplete"
    
    def _hpc_pool_wait_node(self, state: MDState) -> MDState:
        """Sleep until the next SLURM poll while cross-sim HPC jobs run."""
        import time
        from agentic.multi_sim_hpc_pool import (
            _count_running,
            init_hpc_pool,
            persist_hpc_pool_checkpoint,
            pool_summary,
            reuse_hpc_enabled,
        )

        pool = init_hpc_pool(state)
        interval = int(pool.get("check_interval_sec") or state.get("hpc_check_interval_sec") or 7200)
        # --reuse-hpc: no real SLURM jobs to wait on. A multi-hour sleep here
        # stalled campaigns when one prep sim stayed pending. Use a short yield.
        if reuse_hpc_enabled(state) and _count_running(pool) == 0:
            interval = min(interval, 15)
        summary = state.get("hpc_pool_status_summary") or pool_summary(state)
        from agentic.multi_sim_hpc_pool import log_pool_to_base

        persist_hpc_pool_checkpoint(state)
        mins = max(1, interval // 60)
        next_at = datetime.now() + timedelta(seconds=interval)
        next_at_str = next_at.strftime("%H:%M")
        log_pool_to_base(
            state,
            f"Pool sleep — next SLURM check in {interval}s at {next_at_str}",
            pool=pool,
            log_to_conversation=False,
            extra={"summary": summary, "next_check_at": next_at.isoformat(timespec="seconds")},
        )
        logger.info(
            "HPC pool: sleeping %ss (next check at %s)\n%s",
            interval,
            next_at.strftime("%Y-%m-%d %H:%M:%S"),
            summary,
        )
        print(
            f"\n--- HPC pool status (next check in {mins} min at {next_at_str}) ---\n"
            f"{summary}\n",
            flush=True,
        )
        time.sleep(interval)
        state["next_node"] = "supervisor"
        return state

    def _parallel_pool_wait_node(self, state: MDState) -> MDState:
        """Poll while local parallel workers run prep or analysis/reporter."""
        import time
        from agentic.multi_sim_parallel_pool import (
            init_parallel_pool,
            persist_parallel_pool_checkpoint,
        )

        pool = state.get("parallel_pool") or {}
        phase = pool.get("phase") or "analysis"
        init_parallel_pool(state, phase=phase)
        interval = int(state.get("parallel_pool_poll_sec") or 30)
        persist_parallel_pool_checkpoint(state)
        logger.debug("Parallel pool: sleeping %ss before next worker check", interval)
        time.sleep(interval)
        state["next_node"] = "supervisor"
        return state

    def _final_report_node(self, state: MDState) -> MDState:
        """Generate enhanced final workflow report using LLM when available."""
        
        if self.llm:
            report = self._generate_llm_report(state)
        else:
            report = self._generate_fallback_report(state)
        
        state["final_report"] = report
        if state.get("is_multi_simulation"):
            from agentic.multi_sim_progress import multisim_workflow_incomplete

            if multisim_workflow_incomplete(state.get("multi_sim_progress")):
                state["workflow_status"] = "in_progress:interrupted"
            else:
                state["workflow_status"] = "completed"
        else:
            state["workflow_status"] = "completed"
        
        # Save execution report and state to working_dir/supervisor/
        self._save_execution_report(state, report, stage="final_report")
        self._save_workflow_state(state)
        
        return state
    
    def _generate_llm_report(self, state: MDState) -> str:
        """Generate comprehensive report using LLM analysis."""
        
        summary = self._generate_workflow_summary(state)
        
        prompt = f"""
Generate a comprehensive MD workflow completion report.

WORKFLOW SUMMARY:
{summary}

ERRORS: {len(state.get('errors', []))} total
{chr(10).join(state.get('errors', [])[:5])}

WARNINGS: {len(state.get('warnings', []))} total

Create a professional summary including:
1. Workflow status (success/partial/failed)
2. Agents executed and results
3. Files generated
4. Any issues encountered
5. Next steps recommendations
"""
        
        try:
            report = self.llm.prompt(prompt)
            return report
        except Exception as e:
            logger.error(f"Error generating LLM report: {e}")
            return self._generate_fallback_report(state)
    
    def _generate_fallback_report(self, state: MDState) -> str:
        """Generate standard report when LLM is not available."""
        
        summary = self._generate_workflow_summary(state)
        
        report = f"""
=== MD WORKFLOW COMPLETION REPORT ===

Status: {'✅ SUCCESS' if not state.get('errors') else '❌ FAILED'}

WORKFLOW SUMMARY:
{summary}

Files Generated: {len(state.get('figures', []))} figures, {len(state.get('mdp_files', {}))} MDP files

Total Errors: {len(state.get('errors', []))}
Total Warnings: {len(state.get('warnings', []))}

Execution Path: {' → '.join(state.get('execution_path', []))}
"""
        return report
    
    def _generate_workflow_summary(self, state: MDState) -> Dict[str, Any]:
        """Generate a summary of the entire workflow execution."""
        
        summary = {
            "user_goal": state.get("user_goal", "Not specified"),
            "agents_used": [],
            "total_errors": len(state.get("errors", [])),
            "total_warnings": len(state.get("warnings", [])),
            "supervisor_decisions": [],
            "final_outputs": {},
            "execution_path": state.get("execution_path", [])
        }
        
        # Collect agent usage
        for agent in ["preprocessing", "setup", "hpc", "analysis", "planner"]:
            if state.get(f"{agent}_completed"):
                summary["agents_used"].append(agent)
        
        # Collect supervisor decisions
        if "supervisor_reasoning" in state:
            summary["supervisor_decisions"] = state.get("all_supervisor_decisions", [])
        
        # Collect final outputs
        output_keys = [
            "cleaned_pdb", "topology", "coordinates", 
            "mdp_files", "figures", "analysis_results"
        ]
        for key in output_keys:
            if state.get(key):
                summary["final_outputs"][key] = str(state[key])[:100]
        
        return summary
    
    def _save_execution_report(self, state: MDState, report: str, stage: str = None):
        """Save execution_report.md to working_dir/supervisor/.
        
        Args:
            state: Current workflow state.
            report: Optional LLM-generated summary or progress note.
            stage: Optional current stage label (e.g. 'planner', 'preprocess').
        """
        try:
            working_dir = state.get("working_directory", ".")
            supervisor_dir = Path(working_dir) / "supervisor"
            supervisor_dir.mkdir(parents=True, exist_ok=True)
            report_path = supervisor_dir / "execution_report.md"
            
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Build a concise markdown report
            wf_status = state.get("workflow_status", "")
            if wf_status and wf_status.startswith("in_progress"):
                status = f"IN PROGRESS ({stage or wf_status})" 
            elif not state.get('errors'):
                status = 'SUCCESS'
            else:
                status = 'COMPLETED WITH ERRORS'

            lines = [
                f"# MD Workflow Execution Report",
                f"",
                f"**Generated:** {ts}  ",
                f"**Status:** {status}",
                f"",
                f"---",
                f"",
                f"## User Prompt",
                f"",
                f"> {state.get('user_goal', 'N/A')}",
                f"",
            ]
            
            # Enriched prompt
            rephrased = state.get("rephrased_goal") or state.get("enriched_prompt")
            if rephrased:
                lines += [
                    f"## Enriched Prompt",
                    f"",
                    f"{rephrased}",
                    f"",
                ]
            
            # Execution Plan (from planner)
            plan = state.get("execution_plan")
            if plan and isinstance(plan, dict):
                lines += [f"## Execution Plan", f""]
                if plan.get("title"):
                    lines.append(f"**{plan['title']}**")
                    lines.append("")
                agent_seq = plan.get("agent_sequence", [])
                if agent_seq:
                    lines.append(f"Agent sequence: {' → '.join(agent_seq)}")
                    lines.append("")
                full_plan = plan.get("full_plan", "")
                if full_plan:
                    # Truncate very long plans
                    lines.append(full_plan[:3000] + ("..." if len(full_plan) > 3000 else ""))
                    lines.append("")

            # Agents executed
            exec_path = state.get("execution_path", [])
            if exec_path:
                seen = set()
                agents = []
                for node in exec_path:
                    if node not in seen and node not in ("supervisor", "input_validation", "final_report"):
                        seen.add(node)
                        agents.append(node)
                if agents:
                    lines += [
                        f"## Agents Executed",
                        f"",
                        f"{' → '.join(agents)}",
                        f"",
                    ]

            # Key Artifacts produced so far
            artifacts = []
            if state.get("cleaned_pdb"):
                artifacts.append(f"- Cleaned PDB: `{state['cleaned_pdb']}`")
            if state.get("topology"):
                artifacts.append(f"- Topology: `{state['topology']}`")
            if state.get("coordinates"):
                artifacts.append(f"- Coordinates: `{state['coordinates']}`")
            mdp = state.get("mdp_files", {})
            if mdp:
                artifacts.append(f"- MDP files: {', '.join(f'`{k}`' for k in mdp)}")
            if state.get("job_script"):
                artifacts.append(f"- Job script: `{state['job_script']}`")
            figs = state.get("figures", [])
            if figs:
                artifacts.append(f"- Figures: {len(figs)} generated")
            if artifacts:
                lines += [f"## Key Artifacts", f""] + artifacts + [f""]

            # Errors & Warnings (only if present)
            errors = state.get("errors", [])
            warnings = state.get("warnings", [])
            
            # Human Recommendations (if any were given during HITL)
            human_rec = state.get("human_recommendation")
            # Also gather recommendation entries from warnings
            rec_entries = [w for w in warnings if "Human recommendation" in w or "Human modification" in w or "Human modify" in w]
            if human_rec or rec_entries:
                lines += [f"## Human Recommendations", f""]
                if human_rec:
                    lines.append(f"**Active recommendation:** {human_rec}")
                    lines.append("")
                if rec_entries:
                    for r in rec_entries:
                        lines.append(f"- {r}")
                    lines.append("")
                # Show parameter overrides
                param_overrides = []
                if state.get("force_field"):
                    param_overrides.append(f"Force field: {state['force_field']}")
                if state.get("water_model"):
                    param_overrides.append(f"Water model: {state['water_model']}")
                if state.get("temperature"):
                    param_overrides.append(f"Temperature: {state['temperature']} K")
                if state.get("pressure"):
                    param_overrides.append(f"Pressure: {state['pressure']} bar")
                if param_overrides:
                    lines.append("**Current parameters:** " + " | ".join(param_overrides))
                    lines.append("")
            
            if errors:
                lines += [f"## Errors ({len(errors)})", f""]
                for e in errors:
                    lines.append(f"- {e}")
                lines.append("")
            if warnings:
                lines += [f"## Warnings ({len(warnings)})", f""]
                for w in warnings:
                    lines.append(f"- {w}")
                lines.append("")
            
            # LLM-generated summary
            if report:
                lines += [f"## Summary", f"", report, f""]
            
            report_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info(f"Execution report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save execution report: {e}")
    
    def _save_progress(self, state: MDState, stage: str):
        """Save incremental state and execution report after a workflow stage.
        
        Called after each major stage so that HITL checkpoints and the
        interactive Q&A handler always have up-to-date context files.
        """
        state["workflow_status"] = f"in_progress:{stage}"
        self._save_workflow_state(state)
        progress = f"Workflow in progress — last completed stage: **{stage}**"
        self._save_execution_report(state, progress, stage=stage)

    def _active_sim_working_dir(self, state: MDState) -> Optional[str]:
        """Resolve the directory for the active per-simulation checkpoint."""
        progress = state.get("multi_sim_progress") or {}
        label = progress.get("active_sim_label")
        if not label:
            return None
        rec = (progress.get("sims") or {}).get(label) or {}
        wd = rec.get("working_dir")
        if wd:
            return str(Path(wd).resolve())
        base = state.get("multi_sim_base_dir") or state.get("working_directory")
        if base:
            return str((Path(base) / label).resolve())
        return None

    def _save_workflow_state(self, state: MDState):
        """Save serializable workflow state to working_dir/supervisor/state.jsonl."""
        try:
            from agentic.utils.state_persistence import refresh_multisim_progress_for_persist

            refresh_multisim_progress_for_persist(state)

            if state.get("is_multi_simulation") and state.get("sim_prompts"):
                from agentic.multi_sim_progress import ensure_multi_sim_progress

                ensure_multi_sim_progress(state)

            multi_base_dir = state.get("multi_sim_base_dir")
            active_wd = self._active_sim_working_dir(state)
            per_sim_wd = active_wd or state.get("working_directory", ".")
            # Base checkpoint is authoritative for multi-sim resume.
            if state.get("is_multi_simulation") and multi_base_dir:
                working_dir = str(Path(str(multi_base_dir)).resolve())
            else:
                working_dir = per_sim_wd

            supervisor_dir = Path(working_dir) / "supervisor"
            supervisor_dir.mkdir(parents=True, exist_ok=True)
            state_path = supervisor_dir / "state.jsonl"
            
            # Build a serializable snapshot of the state
            snapshot = compact_state_for_persistence(dict(state))
            serializable_state = {}
            for key, value in snapshot.items():
                try:
                    json.dumps(value, default=str)
                    serializable_state[key] = value
                except (TypeError, ValueError):
                    serializable_state[key] = str(value)

            # Never mark completed in state file while multisim loop is incomplete
            persist_status = state.get("workflow_status", "unknown")
            if (
                state.get("is_multi_simulation")
                and persist_status == "completed"
            ):
                from agentic.multi_sim_progress import multisim_workflow_incomplete

                if multisim_workflow_incomplete(state.get("multi_sim_progress")):
                    persist_status = "in_progress:interrupted"
            elif (
                state.get("is_multi_simulation")
                and persist_status not in ("completed", "failed")
            ):
                from agentic.multi_sim_progress import multisim_workflow_incomplete

                if multisim_workflow_incomplete(state.get("multi_sim_progress")):
                    if not str(persist_status).startswith("in_progress"):
                        persist_status = "in_progress:checkpoint"

            # Keep base working_directory in the resume checkpoint for multi-sim.
            if state.get("is_multi_simulation") and multi_base_dir:
                serializable_state["working_directory"] = str(
                    Path(str(multi_base_dir)).resolve()
                )
            elif state.get("hitl_target_sim_label") and multi_base_dir:
                serializable_state["working_directory"] = str(
                    Path(str(multi_base_dir)).resolve()
                )
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "workflow_status": persist_status,
                "state": serializable_state,
            }

            from agentic.utils.state_persistence import (
                format_state_jsonl_entry,
                write_pool_status_json,
            )
            from agentic.campaign.snapshots import persist_state_snapshots

            state_path.write_text(format_state_jsonl_entry(entry), encoding="utf-8")

            write_pool_status_json(state, supervisor_dir)
            persist_state_snapshots(state)

            # Mirror to active per-sim directory when different from base.
            if state.get("is_multi_simulation") and multi_base_dir:
                base_dir = Path(str(multi_base_dir)).resolve()
                current_dir = Path(str(per_sim_wd)).resolve()
                if base_dir != current_dir:
                    per_sim_supervisor = current_dir / "supervisor"
                    per_sim_supervisor.mkdir(parents=True, exist_ok=True)
                    (per_sim_supervisor / "state.jsonl").write_text(
                        format_state_jsonl_entry(entry),
                        encoding="utf-8",
                    )
            
            logger.debug(f"Workflow state saved to {state_path}")
            
        except Exception as e:
            logger.error(f"Failed to save workflow state: {e}")
    
    def _load_workflow_state(self, working_dir: str) -> Optional[Dict[str, Any]]:
        """Load the most recent workflow state from working_dir/supervisor/state.jsonl.
        
        Returns the last saved state dict, or None if no saved state exists.
        """
        state_path = Path(working_dir) / "supervisor" / "state.jsonl"
        saved_state = None
        if state_path.exists():
            try:
                content = state_path.read_text(encoding="utf-8").strip()
                if content:
                    entry = json.loads(content)
                    saved_state = entry.get("state")
                    if saved_state:
                        # Expose save metadata to restore policy in _initialize_state.
                        saved_state["_saved_workflow_status"] = entry.get("workflow_status")
                        saved_state["_saved_timestamp"] = entry.get("timestamp")
                        logger.info(
                            f"Loaded previous workflow state from {state_path} "
                            f"(status={entry.get('workflow_status')}, ts={entry.get('timestamp')})"
                        )
            except Exception as e:
                logger.warning(f"Failed to load workflow state from {state_path}: {e}")
                saved_state = None
        if saved_state is None:
            saved_state = {}
        try:
            from agentic.campaign.snapshots import merge_snapshot_into_state

            saved_state = merge_snapshot_into_state(saved_state, working_dir)
        except Exception as snap_exc:
            logger.debug("campaign state.json overlay skipped: %s", snap_exc)
        return saved_state or None
    
    def _initialize_state(self, user_goal: str, config: Optional[Dict[str, Any]]) -> MDState:
        """Create initial workflow state with config overrides applied."""
        # Note: TypedDict is a type hint only, cannot be instantiated as a class.
        # Use a plain dict instead and let Python's type system handle validation.
        state: MDState = {
            "user_goal": user_goal,
            "md_engine": "gromacs",
            "force_field": "amber99sb-ildn",
            "water_model": "tip3p",
            "production_ns": None,
            "extended_minimization": config.get("extended_minimization", False),
            "human_in_loop": False,
            "hitl_mode": None,
            "preprocessing_issues": [],
            "setup_issues": [],
            "mdp_files": {},
            "analysis_results": {},
            "figures": [],
            "errors": [],
            "warnings": [],
            "next_node": None,
            "human_feedback": None,
            "working_directory": None,
            "execution_path": [],
            "execution_plan": None,
            "plan_executed": False,
            "rephrased_goal": None,
            "enriched_prompt": None,
            "master_enriched_prompt": None,
            "user_goal_original": None,
            "current_node": None,
            "file_registry": {},
            "generated_files": {},
            "subtask_type": None,
            "subtask_type_initialized": None,
            "agent_list": None,
            "required_inputs": None,
            "input_validated": None,
            "analysis_directory": None,
            "trajectory_paths": None,
            "system_info": None,
            "pdb_analysis": None,
            "component_selection": {},
            "structured_prompt": None,
            "raw_pdb": None,
            "cleaned_pdb": None,
            "preprocessing_report": None,
            "topology": None,
            "coordinates": None,
            "chain_residue_map": None,
            "setup_report": None,
            "hpc_action": None,
            "job_script": None,
            "job_id": None,
            "job_status": None,
            "trajectory_path": None,
            "energy_file": None,
            "hpc_report": None,
            "analysis_action": "full_analysis",
            "analysis_request": None,
            "conclusions": None,
            # Agent-specific instruction sections (extracted from the full execution
            # plan by supervisor._extract_agent_specific_plans after planner returns)
            "preprocessing_instructions": None,
            "setup_instructions": None,
            "hpc_instructions": None,
            "analysis_instructions": None,
            # Agent execution tracking
            "current_agent_idx": 0,
            "preprocess_retry_count": 0,
            "setup_retry_count": 0,
            "hpc_retry_count": 0,
            "analysis_retry_count": 0,
            "post_hpc_analysis_retry_count": 0,
            "reporter_retry_count": 0,
            # Intermediate validation artifacts
            "pdb_summary": None,
            "file_info": None,
            "reporter_file_info": None,
            # Reporter stage
            "reporter_output": None,
            "report_type": "comprehensive",
            "include_literature": True,
            "literature_keywords": None,
            "reporter_plan": None,
            "reporter_instructions": None,
            # Final report
            "final_report": None,
            "workflow_status": None,
            "human_final_decision": None,
            # Human-in-the-loop
            "human_recommendation": None,
            "error_triggered_hitl": False,
            # Multi-simulation mode
            "is_multi_simulation": False,
            "multi_sim_phase": None,
            "pdb_list": None,
            "sim_prompts": None,
            "run_combined_analysis": None,
            "combined_analysis_plan": None,
            "campaign_spec": None,
            "allow_partial_combined": False,
            "current_sim_index": 0,
            "completed_sim_states": None,
            "sim_working_dirs": None,
            "multi_sim_base_dir": None,
            "combined_only": False,
            "reuse_hpc": False,
            "resume_failed_only": None,
            "requeue_failed_sims": None,
            "multisim_resume_applied": False,
            "workflow_loop_streak": 0,
            "workflow_loop_key": None,
            "completed_pipeline_stages": [],
            "retry_labels": None,
            "_resume_succeeded_labels": None,
            "hpc_pool": None,
            "allowed_hpc_jobs": None,
            "hpc_check_interval_sec": None,
            "hpc_check_interval": None,
            "post_hpc_analysis_only": False,
            "hpc_pool_phase_complete": False,
            "parallel_pool": None,
            "parallel_workers": "4",
            "parallel_mem_gb_per_job": None,
            "parallel_cpus_per_job": None,
            "parallel_workers_resolved": None,
            "llm_concurrency": "auto",
            "sim_max_attempts": None,
            "parallel_pool_poll_sec": 30,
            "_allowed_hpc_jobs_explicit": False,
        }

        if config:
            state.update(config)

        # Preserve the CLI agent list across HPC-pool prep filtering.
        if state.get("agent_list") and not state.get("pipeline_agent_list"):
            state["pipeline_agent_list"] = list(state["agent_list"])

        if state.get("reuse_hpc"):
            import os as _os_reuse_hpc

            _os_reuse_hpc.environ["AGENTIC_REUSE_HPC"] = "1"

        if state.get("pdb_list") is None:
            state["pdb_list"] = []

        # CRITICAL: Convert working_directory to absolute path to prevent nested directory creation
        # This ensures that even if agents use os.chdir(), paths remain correct
        working_dir = state.get("working_directory", "working_dir")
        if working_dir == ".":
            working_dir = "working_dir"  # Use working_dir instead of current directory
        if not Path(working_dir).is_absolute():
            working_dir = str(Path.cwd() / working_dir)
        state["working_directory"] = working_dir

        if state.get("is_multi_simulation"):
            state["multi_sim_base_dir"] = working_dir
        
        # In multi-sim mode, agent sub-directories live inside each per-simulation
        # directory (e.g. {basepath}/1A/analysis/).  Only shared dirs are created at
        # basepath level: planner/, programmer/, supervisor/.
        # Combined analysis/reporter dirs are created by supervisor when needed.
        if state.get("is_multi_simulation"):
            Path(working_dir).mkdir(parents=True, exist_ok=True)
            for shared_dir in ("planner", "programmer", "supervisor"):
                Path(working_dir, shared_dir).mkdir(parents=True, exist_ok=True)
            # Placeholder dir state — per-sim supervisor will overwrite these
            state["preprocess_dir"] = str(Path(working_dir) / "preprocess")
            state["simsetup_dir"] = str(Path(working_dir) / "simsetup")
            state["hpc_dir"] = str(Path(working_dir) / "hpc")
            state["analysis_dir"] = str(Path(working_dir) / "analysis")
        else:
            # Single-sim: set all dir state fields unconditionally (agents may read them)
            state["preprocess_dir"] = str(Path(working_dir) / "preprocess")
            state["simsetup_dir"] = str(Path(working_dir) / "simsetup")
            state["hpc_dir"] = str(Path(working_dir) / "hpc")
            state["analysis_dir"] = str(Path(working_dir) / "analysis")

            # Determine which agent directories to actually create on disk.
            # Only create directories for agents that will run in this subtask so
            # the working directory stays clean and unambiguous.
            _AGENT_DIR_MAP = {
                "preprocess": state["preprocess_dir"],
                "simsetup":   state["simsetup_dir"],
                "hpc":        state["hpc_dir"],
                "analysis":   state["analysis_dir"],
                "reporter":   str(Path(working_dir) / "reporter"),
            }
            _SUBTASK_AGENTS = {
                "preprocess_only": ["preprocess"],
                "setup_only":      ["simsetup"],
                "hpc_only":        ["hpc"],
                "analysis_only":   ["analysis"],
                "reporter_only":   ["reporter"],
                "full_task":       list(_AGENT_DIR_MAP.keys()),
            }
            # CLI agent name → directory key mapping for multi_agent mode
            _CLI_TO_DIR_KEY = {
                "preprocess": "preprocess",
                "simsetup":   "simsetup",
                "hpcjob":     "hpc",
                "analysis":   "analysis",
                "reporter":   "reporter",
            }

            subtask_type = state.get("subtask_type") or "full_task"
            if subtask_type == "multi_agent":
                agent_list = state.get("agent_list") or []
                agents_needed = [_CLI_TO_DIR_KEY[a] for a in agent_list if a in _CLI_TO_DIR_KEY]
            else:
                agents_needed = _SUBTASK_AGENTS.get(subtask_type, list(_AGENT_DIR_MAP.keys()))

            # supervisor/planner/programmer always needed (state, plans, generated tools)
            dirs_to_create = [_AGENT_DIR_MAP[a] for a in agents_needed] + \
                             [str(Path(working_dir) / d) for d in ("supervisor", "planner", "programmer")]
            for agent_dir in dirs_to_create:
                Path(agent_dir).mkdir(parents=True, exist_ok=True)
        
        # Check for saved workflow state from a previous run
        saved_state = self._load_workflow_state(working_dir)
        if saved_state:
            saved_subtask = saved_state.get("subtask_type")
            current_subtask = state.get("subtask_type")
            saved_agents = saved_state.get("agent_list")
            current_agents = state.get("agent_list")
            saved_status = saved_state.get("_saved_workflow_status")

            same_subtask_signature = (
                saved_subtask == current_subtask
                and list(saved_agents or []) == list(current_agents or [])
            )
            is_completed_snapshot = str(saved_status).startswith("completed")

            # Fresh reruns should rebuild the master plan unless the user explicitly
            # requests resume mode (``--resume``), which sets resume_failed_only.

            # Multi-simulation loop bookkeeping: restore when resuming (explicit or auto).
            multisim_loop_restore = (
                state.get("is_multi_simulation")
                and not state.get("combined_only")
                and state.get("resume_failed_only")
            )

            # Restore artifact paths and completed-stage outputs so agents
            # can skip already-finished work.  Control-flow and retry counters
            # are intentionally NOT restored — the workflow re-evaluates routing
            # fresh each time.
            restore_keys = [
                # Preprocessing outputs
                "raw_pdb", "cleaned_pdb", "preprocessing_report",
                "pdb_analysis", "component_selection", "file_registry", "generated_files",
                "ligand_files", "ligand_resnames", "ion_files", "ion_resnames",
                # Setup outputs
                "topology", "coordinates", "chain_residue_map", "mdp_files", "setup_report",
                # HPC outputs
                "job_script", "job_id", "job_status", "trajectory_path", "energy_file",
                "hpc_report", "hpc_output_directory",
                # Analysis outputs
                "analysis_results", "figures", "conclusions",
                # Reporter outputs
                "reporter_output",
                # Enriched prompt / plan (avoid re-doing expensive LLM calls)
                "rephrased_goal", "enriched_prompt", "master_enriched_prompt",
                "user_goal_original",
                "execution_plan",
                "structured_prompt", "pdb_summary",
                # Agent instructions
                "preprocessing_instructions", "setup_instructions",
                "hpc_instructions", "analysis_instructions", "reporter_instructions",
                # HITL session (agent switch / sim binding across checkpoints)
                "hitl_active_agent", "hitl_target_sim_label", "hitl_sim_dirs",
                "hitl_agent_working_directory",                 "hitl_agent_output_directory",
                "multi_sim_progress",
                "hpc_pool",
                "allowed_hpc_jobs",
                "hpc_check_interval_sec",
                "post_hpc_analysis_only",
                "hpc_pool_phase_complete",
            ]

            # Multi-simulation progress bookkeeping is restored when resuming.
            if multisim_loop_restore:
                restore_keys.extend([
                    "multi_sim_phase",
                    "sim_prompts",
                    "run_combined_analysis",
                    "combined_analysis_plan",
                    "current_sim_index",
                    "completed_sim_states",
                    "sim_working_dirs",
                    "multi_sim_base_dir",
                    "master_enriched_prompt",
                    "all_pdb_analyses",
                    "user_goal_original",
                    "enriched_prompt",
                    "rephrased_goal",
                    "structured_prompt",
                ])

            # For a completed snapshot or a different subtask signature,
            # avoid restoring stale planning/results that can short-circuit
            # routing in subsequent stage runs (for example stage-2 analysis).
            if is_completed_snapshot or not same_subtask_signature:
                skip_stale_keys = {
                    "execution_plan",
                    "analysis_results",
                    "reporter_output",
                    "rephrased_goal",
                    "enriched_prompt",
                    "master_enriched_prompt",
                    "user_goal_original",
                    "structured_prompt",
                    "preprocessing_instructions",
                    "setup_instructions",
                    "hpc_instructions",
                    "analysis_instructions",
                    "reporter_instructions",
                    "sim_prompts",
                    "run_combined_analysis",
                    "combined_analysis_plan",
                    "multi_sim_phase",
                    "current_sim_index",
                    "completed_sim_states",
                    "all_pdb_analyses",
                }
                # In resume mode, keep multi-sim bookkeeping so the supervisor can
                # identify which sims have already succeeded and skip them.
                if state.get("resume_failed_only") and state.get("is_multi_simulation"):
                    skip_stale_keys -= {
                        "sim_prompts",
                        "run_combined_analysis",
                        "combined_analysis_plan",
                        "completed_sim_states",
                        "master_enriched_prompt",
                        "all_pdb_analyses",
                        "enriched_prompt",
                        "rephrased_goal",
                        "structured_prompt",
                        "user_goal_original",
                        "multi_sim_phase",
                        "current_sim_index",
                    }
                restore_keys = [k for k in restore_keys if k not in skip_stale_keys]
                logger.info(
                    "Restore policy: skipping stale planning/results keys "
                    f"(completed={is_completed_snapshot}, same_signature={same_subtask_signature}, "
                    f"resume={state.get('resume_failed_only', False)})"
                )

            for key in restore_keys:
                if key in saved_state and saved_state[key] is not None:
                    state[key] = saved_state[key]

            # Analysis/reporter pool workers must never resume into a stale HPC wait.
            # Stuck mini-campaign workers reloaded per-sim state.jsonl with
            # multi_sim_phase=hpc_pool and slept forever under --reuse-hpc.
            if (
                state.get("pool_phase") == "analysis"
                or state.get("post_hpc_analysis_only")
                or state.get("hpc_pool_disabled")
                or state.get("subtask_type") in ("analysis_only", "reporter_only")
            ):
                if state.get("hpc_pool") or state.get("multi_sim_phase") == "hpc_pool":
                    logger.info(
                        "Clearing stale hpc_pool checkpoint for analysis-phase worker "
                        "(pool_phase=%s subtask=%s saved_status=%s)",
                        state.get("pool_phase"),
                        state.get("subtask_type"),
                        saved_status,
                    )
                state.pop("hpc_pool", None)
                state["hpc_pool_disabled"] = True
                state["use_hpc_pool"] = False
                state["hpc_pool_phase_complete"] = True
                state["post_hpc_analysis_only"] = True
                if state.get("multi_sim_phase") == "hpc_pool":
                    state["multi_sim_phase"] = None
                if state.get("pool_phase") == "analysis":
                    # Per-sim analysis workers are singlesim for routing purposes.
                    state["is_multi_simulation"] = False
                    state.pop("_singlesim_hpc_pool", None)

            if state.get("is_multi_simulation"):
                from agentic.multi_sim_progress import (
                    reconcile_multisim_progress_from_disk,
                    rebuild_progress_from_disk,
                    sync_state_from_progress,
                    multisim_workflow_incomplete,
                    prepare_multisim_resume_state,
                )

                if state.get("resume_failed_only") and not state.get("combined_only"):
                    reconcile_multisim_progress_from_disk(state, working_dir)
                    # If HPC is in the CLI pipeline, reopen failed/unhealthy sims even
                    # when a prior run left hpc_pool_phase_complete=True.
                    try:
                        from agentic.multi_sim_hpc_pool import reopen_failed_hpc_for_resume

                        reopen_failed_hpc_for_resume(state)
                    except Exception as exc:
                        logger.debug("[resume] HPC reopen skipped: %s", exc)
                    # Sequential resume binding fights hpc/parallel pool orchestration.
                    phase_now = state.get("multi_sim_phase")
                    if phase_now not in ("hpc_pool", "parallel_pool"):
                        if multisim_workflow_incomplete(state.get("multi_sim_progress")):
                            prepare_multisim_resume_state(state)
                    else:
                        logger.info(
                            "[resume] Skipping prepare_multisim_resume_state "
                            "(phase=%s — pool tick owns routing)",
                            phase_now,
                        )
                        # Allow parallel prep to restart for remaining pending sims.
                        state.pop("hpc_pool_prep_parallel", None)
                        state["requeue_failed_sims"] = True
                        pool = state.get("hpc_pool") or {}
                        if pool.get("awaiting_hitl"):
                            logger.warning(
                                "[resume] Clearing stale hpc_pool.awaiting_hitl so "
                                "unrelated sims can continue"
                            )
                            pool["awaiting_hitl"] = False
                            pool.pop("hitl_reason", None)
                            state["hpc_pool"] = pool
                        # Drop stale per-sim bind so pool tick can choose the next prep label.
                        progress = state.get("multi_sim_progress") or {}
                        if progress:
                            progress["active_sim_label"] = None
                            progress["active_agent"] = None
                            state["multi_sim_progress"] = progress
                        state["hpc_pool_prep_only"] = False
                        state.pop("hpc_pool_agent_filter", None)
                        state["input_validated"] = None
                        state["execution_plan"] = None
                        state["plan_executed"] = False
                        base = state.get("multi_sim_base_dir") or working_dir
                        if base:
                            state["working_directory"] = str(Path(base).resolve())
                    from agentic.multi_sim_progress import merge_completed_states_from_progress

                    merge_completed_states_from_progress(state)
                    # Fresh --resume invocation: allow supervisor to bind once from disk.
                    state["multisim_resume_applied"] = False
                    state["workflow_loop_streak"] = 0
                    state["workflow_loop_key"] = None
                elif state.get("multi_sim_progress") and not state.get("combined_only"):
                    sync_state_from_progress(state)
                    from agentic.multi_sim_progress import ensure_per_sim_working_directory

                    ensure_per_sim_working_directory(state)
                elif state.get("sim_prompts"):
                    rebuild_progress_from_disk(state, working_dir)

            if state.get("is_multi_simulation") and (is_completed_snapshot or not same_subtask_signature):
                if state.get("resume_failed_only"):
                    # Resume: keep sim_prompts + progress; supervisor routes from
                    # reconciled multi_sim_progress (combined reporter only when done).
                    progress = state.get("multi_sim_progress") or {}
                    phase = progress.get("phase") or state.get("multi_sim_phase")
                    if phase:
                        state["multi_sim_phase"] = phase
                    elif state.get("multi_sim_phase") is None:
                        state["multi_sim_phase"] = "executing_sims"
                    state["execution_plan"] = None
                    state["plan_executed"] = False
                    if progress.get("phase") == "combined_reporter":
                        state["current_agent_idx"] = 1
                    logger.info(
                        "[resume] Multi-sim state preserved — supervisor will continue "
                        "from reconciled multi_sim_progress (phase=%s)",
                        state.get("multi_sim_phase"),
                    )
                else:
                    state["multi_sim_phase"] = None
                    state["current_sim_index"] = 0
                    state["completed_sim_states"] = None
                    state["sim_prompts"] = None
                    state["run_combined_analysis"] = None
                    state["combined_analysis_plan"] = None
                    state["master_enriched_prompt"] = None
                    state["user_goal_original"] = None
                    state["all_pdb_analyses"] = []

            # Fresh multi-sim run (no --resume): never inherit a mid-loop checkpoint.
            if (
                state.get("is_multi_simulation")
                and not state.get("combined_only")
                and not state.get("resume_failed_only")
                and saved_state
            ):
                state["multi_sim_base_dir"] = working_dir
                state["multi_sim_phase"] = None
                state["current_sim_index"] = 0
                state["completed_sim_states"] = None
                state["sim_prompts"] = None
                state["run_combined_analysis"] = None
                state["combined_analysis_plan"] = None
                state["execution_plan"] = None
                state["plan_executed"] = False
                state["current_agent_idx"] = 0
                state["enriched_prompt"] = None
                state["rephrased_goal"] = None
                state["master_enriched_prompt"] = None
                state["analysis_results"] = {}
                state["reporter_output"] = None
                state["figures"] = []
                state["input_validated"] = False
                # A new invocation without --resume must not reopen the HPC
                # pool from the previous checkpoint. --retry-labels is also
                # how a full cohort is selected, so its presence is not a resume.
                state.pop("hpc_pool", None)
                state.pop("parallel_pool", None)
                state.pop("hpc_pool_phase_complete", None)
                state.pop("post_hpc_analysis_only", None)
                state.pop("use_hpc_pool", None)
                state["hpc_pool_disabled"] = False
                logger.info(
                    "[multi-sim] Fresh run — cleared stale loop/checkpoint state; "
                    "supervisor will enrich, build master plan, then enter per-sim loop"
                )
            
            logger.info("Restored previous workflow state — supervisor will skip completed stages")

            from agentic.multi_sim_hpc_pool import resume_hpc_pool_if_needed, reconcile_post_hpc_with_pool

            if (
                state.get("pool_phase") == "analysis"
                or state.get("post_hpc_analysis_only")
                or state.get("hpc_pool_disabled")
            ):
                logger.info(
                    "[resume] Skipping HPC pool re-entry for analysis-phase worker"
                )
            elif reconcile_post_hpc_with_pool(state):
                logger.info(
                    "[resume] Cleared premature post-HPC state — HPC pool still active"
                )
            elif state.get("hpc_pool") or state.get("multi_sim_phase") == "hpc_pool":
                if resume_hpc_pool_if_needed(state):
                    logger.info("[resume] Re-entered cross-sim HPC pool from saved state")
            elif state.get("parallel_pool") or state.get("multi_sim_phase") == "parallel_pool":
                from agentic.multi_sim_parallel_pool import resume_parallel_pool_if_needed

                if resume_parallel_pool_if_needed(state):
                    logger.info("[resume] Re-entered parallel worker pool from saved state")

        # HITL re-run after per-sim work already exists: skip the full per-sim loop
        # and enter combined analysis + reporter (with human checkpoints).
        if (
            state.get("human_in_loop")
            and state.get("hitl_mode") == "all"
            and state.get("is_multi_simulation")
            and not state.get("combined_only")
            and not state.get("resume_failed_only")
            and _multisim_per_sim_workflow_complete(
                working_dir, state.get("multi_sim_progress")
            )
        ):
            state["combined_only"] = True
            logger.info(
                "[hitl] Per-simulation analysis outputs found on disk — "
                "auto-enabling combined-only mode for interactive review"
            )

        # --combined-only reruns must not inherit per-sim routing from a saved state.
        if state.get("combined_only"):
            base_dir = state.get("multi_sim_base_dir") or working_dir
            for _k, _default in (
                ("multi_sim_phase", None),
                ("multi_sim_progress", None),
                ("execution_plan", None),
                ("plan_executed", False),
                ("current_agent_idx", 0),
                ("current_sim_index", 0),
                ("enriched_prompt", None),
                ("rephrased_goal", None),
                ("analysis_results", {}),
                ("reporter_output", None),
                ("raw_pdb", None),
                ("trajectory_path", None),
                ("topology", None),
                ("coordinates", None),
                ("chain_residue_map", None),
                ("job_id", None),
                ("analysis_instructions", None),
                ("reporter_instructions", None),
                ("sim_prompts", None),
                ("completed_sim_states", None),
                ("combined_analysis_plan", None),
                ("input_validated", True),
                ("subtask_type_initialized", False),
            ):
                state[_k] = _default
            state["figures"] = []
            state["run_combined_analysis"] = True
            state["user_goal"] = user_goal
            if base_dir:
                state["multi_sim_base_dir"] = base_dir
                state["working_directory"] = base_dir
            logger.info(
                "[combined_only] Cleared restored routing state — "
                f"supervisor will run combined analysis + report only at {base_dir}"
            )
        
        # Ensure execution_path is always a list we control
        state["execution_path"] = list(state.get("execution_path", []))
        # Always stamp the original user goal so the combined reporter can show it
        # verbatim, regardless of later state mutations (enrichment, combined goals).
        if not state.get("user_goal_original"):
            state["user_goal_original"] = user_goal
        if not state.get("campaign_spec"):
            try:
                from agentic.campaign.compile import compile_campaign_spec

                state["campaign_spec"] = compile_campaign_spec(state).to_dict()
            except Exception:
                pass
        return state

    def run(self, user_goal: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the MD workflow without human checkpoints.
        
        Args:
            user_goal: Natural language description of what user wants
            config: Optional configuration overrides
            
        Returns:
            Final state dictionary
        """
        import signal

        initial_state = self._initialize_state(user_goal, config)
        # Respect explicit config for human loop but default to automatic mode here
        initial_state["human_in_loop"] = bool(initial_state.get("human_in_loop", False))

        active: Dict[str, Any] = {"state": initial_state}
        self._active_run_state = active

        def _checkpoint_interrupt() -> None:
            st = active.get("state") or {}
            base = st.get("multi_sim_base_dir") or st.get("working_directory")
            saved = self._load_workflow_state(base) if base else None
            target = saved if (saved or {}).get("parallel_pool") else st
            if target.get("multi_sim_phase") == "parallel_pool" and target.get("parallel_pool"):
                try:
                    from agentic.multi_sim_parallel_pool import persist_parallel_pool_interrupt

                    persist_parallel_pool_interrupt(target)
                    active["state"] = target
                    logger.info("Saved parallel pool checkpoint after interrupt")
                except Exception as exc:
                    logger.warning("Parallel pool interrupt checkpoint failed: %s", exc)
            elif target.get("multi_sim_phase") == "hpc_pool" and target.get("hpc_pool"):
                try:
                    from agentic.multi_sim_hpc_pool import persist_hpc_pool_checkpoint

                    target["workflow_status"] = "in_progress:interrupted"
                    persist_hpc_pool_checkpoint(target)
                    active["state"] = target
                    logger.info("Saved HPC pool checkpoint after interrupt")
                except Exception as exc:
                    logger.warning("HPC pool interrupt checkpoint failed: %s", exc)

        def _signal_handler(signum, frame):  # noqa: ARG001
            _checkpoint_interrupt()
            raise KeyboardInterrupt()

        prev_int = signal.signal(signal.SIGINT, _signal_handler)
        prev_term = signal.signal(signal.SIGTERM, _signal_handler)
        
        try:
            recursion_limit = self._compute_recursion_limit(initial_state)
            logger.info("LangGraph recursion_limit=%s for this run", recursion_limit)
            final_state = self.graph.invoke(
                initial_state, {"recursion_limit": recursion_limit}
            )
            active["state"] = final_state
            return final_state

        except KeyboardInterrupt:
            _checkpoint_interrupt()
            initial_state.setdefault("errors", []).append(
                "Workflow interrupted during parallel pool — use --resume to continue"
            )
            initial_state["workflow_status"] = "in_progress:interrupted"
            return active.get("state") or initial_state
            
        except Exception as e:
            import traceback
            import sys
            error_traceback = traceback.format_exc()
            logger.error(f"Workflow execution error: {e}\n{error_traceback}")
            # Also print to stdout for immediate visibility
            print(f"\n{'='*60}")
            print(f"WORKFLOW ERROR: {e}")
            print(f"{'='*60}")
            print("Full traceback:")
            print(error_traceback)
            print(f"{'='*60}\n")
            st = active.get("state") or initial_state
            base = st.get("multi_sim_base_dir") or st.get("working_directory")
            saved = self._load_workflow_state(base) if base else None
            target = saved if (saved or {}).get("parallel_pool") else st
            if target.get("multi_sim_phase") == "parallel_pool" and target.get("parallel_pool"):
                try:
                    from agentic.multi_sim_parallel_pool import persist_parallel_pool_interrupt

                    persist_parallel_pool_interrupt(target)
                except Exception:
                    pass
            elif target.get("multi_sim_phase") == "hpc_pool" and target.get("hpc_pool"):
                try:
                    from agentic.multi_sim_hpc_pool import persist_hpc_pool_checkpoint

                    target["workflow_status"] = "in_progress:interrupted"
                    persist_hpc_pool_checkpoint(target)
                except Exception:
                    pass
            target.setdefault("errors", []).append(f"Workflow error: {str(e)}")
            return target
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
            self._active_run_state = None

    def _next_after_field_agent(self, state: MDState, default: str = "supervisor") -> str:
        """After a field agent runs, return to HITL checkpoint if this was a delegated task."""
        return_checkpoint = state.pop("hitl_return_checkpoint", None)
        if return_checkpoint:
            state.pop("hitl_delegate_agent", None)
            state.pop("hitl_delegate_task", None)
            logger.info("HITL delegated run complete — returning to %s", return_checkpoint)
            return return_checkpoint
        return state.get("next_node", default)

    def run_with_human_feedback(
        self,
        user_goal: str,
        feedback_handler: Callable[[Dict[str, Any]], str],
        config: Optional[Dict[str, Any]] = None,
        max_steps: int = 200
    ) -> Dict[str, Any]:
        """Run workflow with interactive human checkpoints.

        Uses the same ``_invoke_node`` / ``_finalize_node`` path as the compiled
        graph so stage ticks, inventories, and science verify stay aligned.
        """
        if feedback_handler is None:
            raise ValueError("feedback_handler is required for human-in-the-loop execution")

        state = self._initialize_state(user_goal, config)
        state["human_in_loop"] = True
        if not state.get("hitl_mode"):
            state["hitl_mode"] = "all"
        current_node = "supervisor"
        steps = 0
        # Family / pool campaigns need a higher step budget than the old 200 default.
        try:
            max_steps = max(int(max_steps), self._compute_recursion_limit(state))
        except Exception:
            max_steps = max(int(max_steps), 2000)

        _HUMAN_FEEDBACK_MAP = {
            "human_preprocess_check": "preprocess",
            "human_setup_check": "setup",
            "human_hpc_check": "hpc",
            "human_hpc_pool_check": "hpc_pool",
            "human_analysis_check": "analysis",
            "human_reporter_check": "reporter",
        }

        while steps < max_steps:
            steps += 1
            state["execution_path"].append(current_node)

            if current_node in _HUMAN_FEEDBACK_MAP:
                ck = _HUMAN_FEEDBACK_MAP[current_node]
                feedback = self._collect_human_feedback(state, ck, feedback_handler)
                if self._should_stop(feedback, state, ck):
                    current_node = "final_report"
                    continue
                state["human_feedback"] = feedback
                state = self._invoke_node(current_node, state)
                current_node = state.get("next_node", "supervisor")
                continue

            if current_node == "supervisor":
                state = self._invoke_node("supervisor", state)
                current_node = self._route_from_supervisor(state)
                continue

            if current_node in {"final_report", "workflow_incomplete"}:
                return self._invoke_node(current_node, state)

            if current_node == "parallel_pool_wait":
                state = self._invoke_node(current_node, state)
                try:
                    from agentic.multi_sim_parallel_pool import persist_parallel_pool_checkpoint

                    persist_parallel_pool_checkpoint(state)
                except Exception:
                    pass
                current_node = state.get("next_node", "supervisor")
                continue

            # Field agents + planner + validation + waits + verify
            state = self._invoke_node(current_node, state)
            if current_node in {"preprocess", "setup", "hpc", "analysis", "reporter"}:
                hitl_next = state.get("next_node")
                should_record = (
                    current_node in {"analysis", "reporter"}
                    and state.get("is_multi_simulation")
                    and state.get("sim_prompts")
                    and hitl_next
                    in {"human_analysis_check", "human_reporter_check"}
                )
                if current_node == "analysis" and not state.get("analysis_science_ok"):
                    should_record = False
                if should_record:
                    try:
                        from agentic.multi_sim_progress import record_agent_finished

                        record_agent_finished(state, current_node)
                    except Exception:
                        pass
                default = (
                    "human_reporter_check" if current_node == "reporter" else "supervisor"
                )
                current_node = self._next_after_field_agent(state, default)
            else:
                current_node = state.get("next_node", "supervisor")

        state["errors"].append("Human-in-the-loop workflow exceeded maximum iterations")
        state["workflow_status"] = "incomplete"
        return self._invoke_node("workflow_incomplete", state)

    def _collect_human_feedback(
        self,
        state: MDState,
        checkpoint_type: str,
        feedback_handler: Callable[[Dict[str, Any]], str]
    ) -> str:
        """Prompt human for feedback using provided handler.
        
        Injects '_llm' and '_state' into the summary so interactive handlers
        can answer user questions conversationally.
        """
        summary = self.checkpoints.get_checkpoint_summary(state, checkpoint_type)
        # Provide LLM and full state to handler for interactive Q&A
        summary["_llm"] = self.llm
        summary["_state"] = state
        try:
            response = feedback_handler(summary)
        except Exception as exc:
            logger.error(f"Feedback handler failed: {exc}")
            state["errors"].append(f"Feedback handler error at {checkpoint_type}: {exc}")
            return "exit"
        return (response or "").strip()

    def _should_stop(self, feedback: str, state: MDState, checkpoint_label: str) -> bool:
        """Detect stop keywords from human feedback."""
        if feedback.lower() in {"exit", "quit", "stop"}:
            state["errors"].append(
                f"Workflow stopped by human during {checkpoint_label} checkpoint"
            )
            state["next_node"] = "final_report"
            return True
        return False
    
    def visualize_workflow(self, output_file: str = "current_workflow.png") -> bool:
        """
        Create a visualization of this workflow instance.
        
        Args:
            output_file: Path where to save the PNG diagram
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from .utils import WorkflowVisualizer
            visualizer = WorkflowVisualizer()
            return visualizer.visualize_workflow(output_file, use_actual_graph=True)
        except Exception as e:
            logger.error(f"Failed to visualize workflow: {e}")
            return False
