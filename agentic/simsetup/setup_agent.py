"""
Simulation Setup Agent - Orchestrates GROMACS simulation system preparation
Follows same workflow pattern as preprocessing_agent: LLM planning → tool execution
"""
import logging
import json
import yaml
import os
import shutil
import re
from typing import Dict, Any, Optional
from pathlib import Path

from ..state import MDState
from ..hitl_config import hitl_should_interact
from ..llm import LLMClient
from ..utils import (
    log_agent_start, log_llm_interaction, log_agent_action, 
    log_file_operation, log_agent_completion, log_error,
    SecureFileManager
)
from .schemas import (
    SimSetupPlan, SimSetupStep, 
    SimSetupResult, SimSetupAgentInput, SimSetupAgentOutput
)
from .tools import SimulationSetupToolExecutor, get_tool_metadata
from src.preprocess.phospho_residues import is_phospho_protein_resname

logger = logging.getLogger(__name__)


class SimulationSetupAgent:
    """
    LLM-powered simulation setup agent - orchestrates GROMACS system preparation workflow
    Delegates tool execution to SimulationSetupToolExecutor from tools.py
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config_path: Optional[str] = None):
        """
        Initialize simulation setup agent
        
        Args:
            llm_client: LLM client for intelligent planning
            config_path: Path to config.yaml (defaults to same directory)
        """
        if llm_client is None:
            self.llm = LLMClient("gpt-oss:20b")
        else:
            self.llm = llm_client
        
        self.tool_executor = None
        self.file_manager = None  # Initialized per execution for state-specific file registry
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load setup configuration from YAML"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def setup_node(self, state: MDState) -> MDState:
        """
        Main setup node - entry point from workflow
        Orchestrates the entire simulation setup pipeline
        """
        # Check if we have planner instructions
        execution_plan = state.get("execution_plan", {})
        has_planner_instructions = execution_plan.get("format") == "natural_language"
        
        input_summary = {
            "cleaned_pdb": state.get("cleaned_pdb"),
            "force_field": state.get("force_field"),
            "water_model": state.get("water_model")
        }
        
        if has_planner_instructions:
            # Prefer pre-extracted instructions from supervisor (avoids duplication)
            setup_section = state.get("setup_instructions")
            
            if not setup_section:
                # Fallback: Extract from full plan if supervisor didn't provide it
                full_plan = execution_plan.get("full_plan", "")
                setup_section = self._extract_agent_instructions(full_plan, "Simulation Setup Agent")
            
            if setup_section:
                input_summary["planner_instructions"] = setup_section
            else:
                input_summary["planner_instructions"] = "[Natural language plan from planner - see above]"
        else:
            # Fallback to user goal if no planner instructions
            input_summary["user_goal"] = state.get("user_goal")
        
        log_agent_start("setup", "Simulation System Setup with LLM Tool Calling", input_summary)
        
        try:
            # Initialize secure file manager
            base_working_dir = state.get("working_directory", "working_dir")
            file_registry = state.get("file_registry", {})
            
            self.file_manager = SecureFileManager(
                working_dir=base_working_dir,
                agent_name="simsetup",
                file_registry=file_registry
            )
            
            logger.info(f"SimSetup agent directory: {self.file_manager.agent_dir}")
            
            # Get simsetup directory from file manager (ensures consistency)
            simsetup_dir = self.file_manager.agent_dir
            
            # CRITICAL: Set simsetup_directory in state BEFORE creating tool executor
            state["simsetup_directory"] = simsetup_dir
            
            self.tool_executor = SimulationSetupToolExecutor(simsetup_dir, self.config)
            
            # Copy preprocessed files from preprocess directory
            self._copy_preprocessed_files(state, simsetup_dir)
            
            # Prepare agent input from state
            agent_input = self._prepare_agent_input(state)
            
            # Run LLM-guided setup workflow
            agent_output = self._run_setup_workflow(agent_input, state)
            
            # Update state with results
            self._update_state(state, agent_output)
            
            # Determine next workflow node
            if agent_output.success:
                # Clear any previous setup-related errors from earlier retry attempts
                state["errors"] = [
                    e for e in state.get("errors", [])
                    if not (e.startswith("Setup failed:") or e.startswith("Setup error:"))
                ]
                if hitl_should_interact(state):
                    state["next_node"] = "human_setup_check"
                else:
                    state["next_node"] = "supervisor"
            else:
                state["errors"].append(f"Setup failed: {agent_output.result.report}")
                state["next_node"] = "human_setup_check"
                state["error_triggered_hitl"] = True
            
            success = agent_output.success and len(agent_output.result.issues) == 0
            log_agent_completion("setup", "Simulation System Setup", state, success)
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Setup agent failed: {e}")
            logger.error(f"Traceback: {tb}")
            log_error("setup_agent.setup_node", e, {"state": str(state), "traceback": tb})
            state["errors"].append(f"Setup error: {str(e)}")
            # Error-triggered HITL
            state["next_node"] = "human_setup_check"
            state["error_triggered_hitl"] = True
        
        return state
    
    def _extract_agent_instructions(self, full_plan: str, agent_name: str) -> Optional[str]:
        """
        Extract agent-specific detailed instructions from planner's natural language plan.
        
        Args:
            full_plan: Complete natural language plan from planner
            agent_name: Name of the agent section to extract (e.g., "Simulation Setup Agent")
            
        Returns:
            Extracted instructions for this specific agent, or full plan as fallback
        """
        import re
        
        # Try multiple patterns to find the agent section (in priority order)
        patterns = [
            # New standardized format: **Simulation Setup Agent:**
            rf'\*\*Simulation Setup Agent:\*\*\s*\n(.*?)(?=\n\s*\*\*(?:HPC Agent|Analysis Agent|Expected Outcomes):|$)',
            # Alternative: **Setup Agent:**
            rf'\*\*Setup Agent:\*\*\s*\n(.*?)(?=\n\s*\*\*(?:HPC Agent|Analysis Agent|Expected Outcomes):|$)',
            # With optional colon
            rf'\*\*{agent_name}\*\*:?\s*\n(.*?)(?=\n\s*\*\*[A-Z]|\Z)',
            # Markdown headings
            rf'###\s*{agent_name}.*?\n(.*?)(?=###|\Z)',
            rf'##\s*{agent_name}.*?\n(.*?)(?=##|\Z)',
            # Fuzzy match patterns (allow for variations)
            rf'\*\*(?:Simulation\s*)?Setup.*?Agent\*\*:?\s*\n(.*?)(?=\n\s*\*\*[A-Z]|\Z)',
            rf'###\s*(?:Simulation\s*)?Setup.*?Agent.*?\n(.*?)(?=###|\Z)',
            rf'##\s*(?:Simulation\s*)?Setup.*?Agent.*?\n(.*?)(?=##|\Z)',
            # Section number patterns
            rf'\d+\..*?(?:Simulation\s*)?Setup.*?Agent.*?\n(.*?)(?=\d+\.|\Z)',
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
    
    def _copy_preprocessed_files(self, state: MDState, simsetup_dir: str):
        """Copy necessary files from preprocess directory to simsetup directory.
        
        Respects component_selection: only copies files for components the user requested.
        """
        preprocess_dir = state.get("preprocess_directory")
        component_selection = state.get("component_selection", {})
        
        # Fallback: try to construct preprocess directory path
        if not preprocess_dir:
            base_working_dir = state.get("working_directory", "working_dir")
            preprocess_dir = str(Path(base_working_dir) / "preprocess")
            logger.info(f"preprocess_directory not in state, using fallback: {preprocess_dir}")
            
            # Check if fallback directory exists
            if not os.path.exists(preprocess_dir):
                logger.warning(f"Preprocessing directory does not exist: {preprocess_dir}")
                logger.info("Setup will use files from their current locations")
                return
        
        # Copy files using file_registry via SecureFileManager (preferred method)
        preprocess_files = [
            (file_path, metadata) 
            for file_path, metadata in self.file_manager.file_registry.items()
            if metadata.get("stage") == "preprocess" and Path(file_path).exists()
        ]
        
        copied_files = set()  # Track what we've copied to avoid duplicates
        
        if preprocess_files:
            # Copy preprocessing files, respecting component selection
            for file_path, metadata in preprocess_files:
                filename = Path(file_path).name
                file_type = metadata.get("type", "file")
                
                stem_upper = Path(file_path).stem.upper()
                if file_type == "ligand" and (
                    stem_upper in self.PHOSPHO_PROTEIN_RESNAMES
                    or is_phospho_protein_resname(stem_upper)
                ):
                    logger.info(
                        "Skipping phospho protein file (not a ligand): %s", filename,
                    )
                    continue

                # Filter by component selection
                if component_selection:
                    if file_type == "ligand" and not component_selection.get("ligand", True):
                        logger.info(f"Skipping ligand file {filename} (excluded by component selection)")
                        continue
                    if file_type == "ion" and not component_selection.get("ions", True):
                        logger.info(f"Skipping ion file {filename} (excluded by component selection)")
                        continue
                
                if filename not in copied_files:
                    # Use secure copy (automatically registers)
                    new_path = self.file_manager.copy_file(
                        source_path=file_path,
                        dest_filename=filename,
                        file_type=metadata.get("type", "file"),
                        description=f"Copy from preprocess: {metadata.get('description', 'N/A')}"
                    )
                    
                    if new_path:
                        copied_files.add(filename)
                        
                        # Update cleaned_pdb if this is the protein component
                        if metadata.get("component") == "protein" or metadata.get("type") == "protein":
                            state["cleaned_pdb"] = new_path
                            logger.info(f"Updated cleaned_pdb to: {new_path}")
                        
                        file_type = metadata.get("type", "file")
                        description = metadata.get("description", "N/A")
                        logger.info(f"Copied {filename} ({file_type}) from preprocess to simsetup")
        
        # Fallback: if no file_registry, use cleaned_pdb from state
        if not copied_files:
            cleaned_pdb = state.get("cleaned_pdb")
            if cleaned_pdb and os.path.exists(cleaned_pdb):
                filename = Path(cleaned_pdb).name
                
                # Use secure copy
                new_path = self.file_manager.copy_file(
                    source_path=cleaned_pdb,
                    dest_filename=filename,
                    file_type="pdb",
                    description="From preprocessing agent (fallback)"
                )
                
                if new_path:
                    state["cleaned_pdb"] = new_path
                    logger.info(f"Copied {filename} from preprocessing (fallback method)")
    
    def _prepare_agent_input(self, state: MDState) -> SimSetupAgentInput:
        """Prepare structured input for setup from workflow state"""
        defaults = self.config.get("defaults", {})

        def _extract_production_ns(*texts: Optional[str]) -> Optional[float]:
            """Extract production duration in ns from user/planner text."""
            patterns = [
                # "simulation for 100 ns", "production run of 50 ns"
                re.compile(
                    r'(?:production|simulation|run)\s*(?:time|length|duration|run)?\s*(?:is|of|for|=|:)?\s*(\d+(?:\.\d+)?)\s*ns\b',
                    re.I,
                ),
                # fallback: any explicit "NN ns"
                re.compile(r'\b(\d+(?:\.\d+)?)\s*ns\b', re.I),
            ]

            for txt in texts:
                if not txt:
                    continue
                for pat in patterns:
                    m = pat.search(txt)
                    if m:
                        try:
                            val = float(m.group(1))
                            if val > 0:
                                return val
                        except (TypeError, ValueError):
                            continue
            return None
        
        # Check if planner provided detailed instructions for this agent
        # Prefer pre-extracted instructions from supervisor
        planner_instructions = state.get("setup_instructions")
        
        if not planner_instructions:
            # Fallback: Extract from execution_plan if supervisor didn't provide it
            execution_plan = state.get("execution_plan", {})
            if execution_plan.get("format") == "natural_language":
                full_plan = execution_plan.get("full_plan", "")
                planner_instructions = self._extract_agent_instructions(full_plan, "Simulation Setup Agent")
        
        # Append human recommendation to instructions so the LLM planner sees it
        human_rec = state.get("human_recommendation")
        if human_rec:
            rec_block = (
                f"\n\n**HUMAN RECOMMENDATION (must be followed):**\n{human_rec}\n"
                "Adjust the setup plan to incorporate this recommendation."
            )
            if planner_instructions:
                planner_instructions += rec_block
            else:
                planner_instructions = rec_block

        from src.simsetup.system_options import resolve_simsetup_options

        sim_label = Path(state.get("working_directory", "working_dir")).name
        resolved = resolve_simsetup_options(
            state=state,
            user_text="\n".join(
                filter(
                    None,
                    [
                        state.get("user_goal", ""),
                        state.get("user_goal_original", ""),
                        planner_instructions or "",
                        state.get("enriched_prompt", ""),
                    ],
                )
            ),
            pdb_path=state.get("raw_pdb") or state.get("cleaned_pdb"),
            label=sim_label,
            defaults=defaults,
        )

        production_ns = resolved.get("production_ns")
        if production_ns is not None:
            try:
                production_ns = float(production_ns)
            except (TypeError, ValueError):
                production_ns = None

        temperature = resolved.get("temperature") or defaults.get("temperature", 300.0)
        pressure = resolved.get("pressure") or defaults.get("pressure", 1.0)
        extended_minim = resolved.get("extended_minimization", False)
            
        return SimSetupAgentInput(
            cleaned_pdb=state.get("cleaned_pdb", ""),
            working_directory=state.get("working_directory", "working_dir"),
            force_field=state.get("force_field", defaults.get("force_field", "amber99sb-ildn")),
            water_model=state.get("water_model", defaults.get("water_model", "tip3p")),
            temperature=temperature,
            pressure=pressure,
            production_ns=production_ns,
            extended_minimization=extended_minim,
            user_goal=state.get("user_goal", ""),
            additional_instructions=planner_instructions,
            component_selection=state.get("component_selection")
        )
    
    def _run_setup_workflow(self, agent_input: SimSetupAgentInput, 
                           state: MDState) -> SimSetupAgentOutput:
        """
        Run complete setup workflow:
        1. LLM creates intelligent plan based on system analysis
        2. Execute plan step-by-step using tools
        3. Return structured results
        """
        try:
            # Step 1: LLM analyzes system and creates plan —
            # BUT if a human already modified the structured plan via HITL, use that
            # directly so their changes are not overwritten by a fresh LLM re-plan.
            exec_plan = state.get("execution_plan")
            human_rec = state.get("human_recommendation")
            if human_rec:
                # Human provided guidance: apply changes on top of existing plan (if any)
                logger.info("setup: replanning with human guidance: %s", human_rec[:120])
                _updated = self.replan_with_guidance(human_rec, state)
                if _updated:
                    plan = SimSetupPlan(
                        reasoning=_updated.get("reasoning", ""),
                        overview=_updated.get("overview", ""),
                        steps=[
                            SimSetupStep(
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
                    plan = self._normalize_multi_rep_setup_plan(plan, state)
                else:
                    # LLM unavailable — fall back; human_rec is already in additional_instructions
                    plan = self._create_setup_plan(agent_input, state)
            else:
                prior = ((exec_plan or {}).get("structured_plans") or {}).get("setup")
                force_replan = bool(state.get("force_replan") or state.get("setup_force_replan"))
                if (
                    not force_replan
                    and isinstance(prior, dict)
                    and prior.get("steps")
                ):
                    logger.info(
                        "setup: reusing structured plan (%d steps), not re-planning",
                        len(prior.get("steps") or []),
                    )
                    plan = SimSetupPlan(
                        reasoning=prior.get("reasoning") or "Compiled setup plan",
                        overview=prior.get("overview") or "Reuse structured setup plan",
                        steps=[
                            SimSetupStep(
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
                    plan = self._normalize_multi_rep_setup_plan(plan, state)
                else:
                    plan = self._create_setup_plan(agent_input, state)

            try:
                _rn = int(state.get("rep_num") or 1)
            except (TypeError, ValueError):
                _rn = 1
            _log = {
                "steps": len(plan.steps),
                "tools": [s.tool_name for s in plan.steps],
                "reasoning": (plan.reasoning or "")[:300],
                "overview": (plan.overview or "")[:300],
                "note": (
                    "Box/ions/temperature/production_ns may still be overridden at "
                    "execution (see 'Resolved simsetup parameters')."
                ),
            }
            if _rn > 1:
                _log["rep_num"] = _rn
                _log["multi_rep_setup"] = (
                    f"Single shared simsetup/ for all {_rn} replicates; "
                    "HPC stages hpc/repXX with different seeds. "
                    "Per-replicate setup dirs (rep1/rep2) are not used."
                )
            log_agent_action(
                "setup",
                (
                    "Generated setup plan (shared simsetup; replicates staged at HPC)"
                    if _rn > 1
                    else "Generated setup plan (LLM proposal — may differ from executed params)"
                ),
                _log,
            )

            # Persist structured plan to state for HITL inspection/modification
            if exec_plan is not None:
                exec_plan.setdefault("structured_plans", {})["setup"] = plan.model_dump()
            
            # Step 2: Execute plan using tool executor
            result = self._execute_plan(agent_input, plan, state)
            
            # Step 3: Register created files using SecureFileManager
            if result.topology:
                self.file_manager.register_external_file(
                    file_path=result.topology,
                    file_type="topology",
                    description="GROMACS topology file (.top)"
                )
            
            if result.coordinates:
                self.file_manager.register_external_file(
                    file_path=result.coordinates,
                    file_type="coordinates",
                    description="GROMACS coordinate file (.gro)"
                )
            
            # Register MDP files
            for mdp_type, mdp_path in result.mdp_files.items():
                self.file_manager.register_external_file(
                    file_path=mdp_path,
                    file_type="mdp",
                    description=f"GROMACS MDP file for {mdp_type}"
                )
            
            # Step 4: Prepare supervisor update
            supervisor_update = {
                "coordinates": result.coordinates,
                "topology": result.topology,
                "mdp_files": result.mdp_files,
                "setup_report": result.report,
                "file_registry": self.file_manager.file_registry  # Use file_registry from file_manager
            }
            
            return SimSetupAgentOutput(
                success=result.success,
                plan=plan,
                result=result,
                supervisor_update=supervisor_update
            )
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Setup workflow failed: {e}")
            logger.error(f"Traceback: {tb}")
            return SimSetupAgentOutput(
                success=False,
                plan=SimSetupPlan(
                    reasoning=f"Error in planning: {str(e)}",
                    overview="Failed",
                    steps=[]
                ),
                result=SimSetupResult(
                    success=False,
                    report=f"Error: {str(e)}\n\nTraceback:\n{tb}",
                    issues=[str(e)],
                    warnings=[]
                ),
                supervisor_update={}
            )
    
    def _create_setup_plan(self, agent_input: SimSetupAgentInput, state: MDState) -> SimSetupPlan:
        """
        Use LLM to analyze system and create intelligent setup plan
        Falls back to template-based plan if LLM fails
        """
        # Analyze the system first
        cleaned_pdb = agent_input.cleaned_pdb
        analysis = self._analyze_system(cleaned_pdb, component_selection=agent_input.component_selection, state=state)
        # Store for use in _execute_plan
        self._last_analysis = analysis
        
        if state.get("reuse_hpc"):
            logger.info("reuse_hpc: skipping LLM setup planning, using deterministic fallback")
            return self._create_fallback_plan(agent_input, analysis)

        # Build LLM prompt from config template
        prompt = self._build_planning_prompt(agent_input, analysis, state)
        
        try:
            response = self.llm.invoke([prompt])
            content = response.content or ""
            
            log_llm_interaction("setup.planning", prompt, content,
                              is_mock=hasattr(self.llm, '_is_mock_mode') and self.llm._is_mock_mode)
            
            # Parse LLM response into structured plan
            plan_dict = self._extract_plan_json(content)

            # Avoid false-success loops when LLM output is malformed and no
            # executable steps are parsed.
            if not isinstance(plan_dict.get("steps"), list) or not plan_dict.get("steps"):
                logger.warning("LLM setup plan had no executable steps; falling back to deterministic plan")
                return self._create_fallback_plan(agent_input, analysis)
            
            plan = SimSetupPlan(
                reasoning=plan_dict.get("reasoning", content[:500]),
                overview=plan_dict.get("overview", "Setting up GROMACS simulation system"),
                steps=[
                    SimSetupStep(
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
            return self._normalize_multi_rep_setup_plan(plan, state)
            
        except Exception as e:
            logger.warning(f"LLM planning failed, using fallback: {e}")
            return self._create_fallback_plan(agent_input, analysis)

    def _normalize_multi_rep_setup_plan(
        self, plan: SimSetupPlan, state: Optional[MDState]
    ) -> SimSetupPlan:
        """Collapse LLM multi-rep setup mistakes into one shared simsetup.

        Production replicates are created later by HPC (``hpc/repXX`` copies +
        different seeds). Setup must remain a single ``simsetup/`` build.
        """
        try:
            rep_num = int((state or {}).get("rep_num") or 1)
        except (TypeError, ValueError):
            rep_num = 1
        if rep_num <= 1 or not plan.steps:
            return plan

        build_tools = {"build_simulation_system"}
        kept: list = []
        dropped_builds = 0
        for step in plan.steps:
            params = dict(step.tool_params or {})
            out = str(params.get("output_dir") or "").strip().lower()
            # Strip LLM inventing rep1/rep2/rep01 as setup destinations
            if out and (
                out in {"rep1", "rep2", "rep01", "rep02"}
                or (out.startswith("rep") and out[3:].isdigit())
            ):
                params.pop("output_dir", None)
                step = SimSetupStep(
                    name=step.name,
                    description=step.description,
                    tool_name=step.tool_name,
                    tool_params=params,
                    reason=step.reason,
                )
            if step.tool_name in build_tools:
                if any(s.tool_name in build_tools for s in kept):
                    dropped_builds += 1
                    continue
                step = SimSetupStep(
                    name="Build shared simulation system",
                    description=(
                        f"Create one GROMACS system under simsetup/; HPC will "
                        f"stage it into hpc/rep01…rep{rep_num:02d} with "
                        f"independent production seeds."
                    ),
                    tool_name=step.tool_name,
                    tool_params=step.tool_params,
                    reason=(
                        f"Single shared setup for all {rep_num} production "
                        "replicates (not one setup per replicate)."
                    ),
                )
            kept.append(step)

        if dropped_builds:
            logger.info(
                "Normalized multi-rep setup plan: kept 1 build_simulation_system, "
                "dropped %d duplicate per-replicate build(s) (rep_num=%d)",
                dropped_builds,
                rep_num,
            )
            plan.reasoning = (
                f"Multi-rep (rep_num={rep_num}): one shared simsetup serves all "
                f"production replicates; HPC copies into hpc/repXX with different "
                f"seeds. Collapsed {dropped_builds} duplicate build step(s) from "
                f"the LLM proposal. "
                + (plan.reasoning or "")
            )[:2000]
            plan.overview = (
                f"Build one shared GROMACS system in simsetup/; "
                f"{rep_num} production replicates are staged later under hpc/repXX."
            )
        plan.steps = kept
        return plan
    
    # Known ligand residue names (non-standard residues that need parameters)
    KNOWN_LIGAND_RESNAMES = {
        'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP', 'NAD', 'NAP', 'FAD', 'FMN',
        'HEM', 'LIG', 'UNL', 'UNK', 'DRG',
    }
    
    # Standard ions / waters / common cofactors that are NOT ligands
    STANDARD_RESNAMES = {
        'HOH', 'WAT', 'SOL', 'TIP', 'MG', 'CA', 'ZN', 'MN', 'FE', 'NA', 'CL',
        'K', 'CU', 'NI', 'CO',
    }

    # Phosphorylated amino acids are protein residues (pdb2gmx), not ACPYPE ligands
    PHOSPHO_PROTEIN_RESNAMES = frozenset({
        'SEP', 'TPO', 'PTR', 'SP2', 'THP1', 'THP', 'TP2',
    })
    
    # Pre-built ligand parameter directory
    LIGAND_PARAM_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "simsetup" / "amber_ligand_param"
    
    def _analyze_system(self, pdb_file: str, component_selection: Optional[Dict[str, Any]] = None,
                        state: Optional[MDState] = None) -> Dict[str, Any]:
        """Analyze the preprocessed system to inform setup decisions.
        
        Detects ligand residue names, ion types, and checks for pre-built
        ligand parameters in src/simsetup/amber_ligand_param/.
        
        Uses THREE sources (in priority order):
          1. State fields (ligand_files, ligand_resnames, ion_files, ion_resnames)
             — set by preprocessing agent from separate_complex_components
          2. File registry (type == "ligand" / "ion") — files copied to simsetup
          3. HETATM scan of pdb_file — fallback if above are empty
        
        If component_selection is provided, overrides detection for components
        that the user explicitly excluded.
        """
        analysis = {
            "has_ligand": False,
            "has_ions": False,
            "estimated_atoms": 0,
            "system_type": "protein_only",
            "ligand_resnames": [],
            "ion_resnames": [],
            "ligand_files": [],       # paths to ligand PDB/GRO files in simsetup
            "ion_files": [],          # paths to ion PDB/GRO files in simsetup
            "ligand_itps": [],        # paths to pre-built ITP files
            "ligand_params_available": {},  # {resname: path_to_itp} for pre-built params
            "ligand_params_missing": [],    # resnames that need generation
        }
        
        # --- Source 1: State fields from preprocessing ---
        if state:
            state_ligand_resnames = state.get("ligand_resnames") or []
            state_ligand_files = state.get("ligand_files") or []
            state_ion_resnames = state.get("ion_resnames") or []
            state_ion_files = state.get("ion_files") or []
            
            if state_ligand_resnames:
                filtered = [
                    r for r in state_ligand_resnames
                    if not is_phospho_protein_resname(r)
                ]
                if filtered:
                    analysis["ligand_resnames"] = filtered
                    analysis["has_ligand"] = True
                    logger.info(f"Ligand resnames from state: {filtered}")
                elif state_ligand_resnames:
                    logger.info(
                        "Ignored phospho residue names from state (protein, not ligand): %s",
                        state_ligand_resnames,
                    )
            if state_ion_resnames:
                analysis["ion_resnames"] = list(state_ion_resnames)
                analysis["has_ions"] = True
                logger.info(f"Ion resnames from state: {state_ion_resnames}")
            for fp in state_ligand_files:
                if fp and os.path.isfile(fp) and str(fp) not in analysis["ligand_files"]:
                    analysis["ligand_files"].append(str(fp))
                    analysis["has_ligand"] = True
            for fp in state_ion_files:
                if fp and os.path.isfile(fp) and str(fp) not in analysis["ion_files"]:
                    analysis["ion_files"].append(str(fp))
                    analysis["has_ions"] = True
        
        # --- Source 2: File registry (files copied to simsetup) ---
        simsetup_dir = None
        if state:
            simsetup_dir = state.get("simsetup_directory")
        
        if not analysis["has_ligand"] or not analysis["has_ions"]:
            for file_path, metadata in self.file_manager.file_registry.items():
                ftype = metadata.get("type", "")
                if ftype == "ligand" and not analysis["has_ligand"]:
                    resname = Path(file_path).stem.upper()
                    if is_phospho_protein_resname(resname):
                        logger.info(
                            "Skipping phospho protein file mis-tagged as ligand: %s",
                            file_path,
                        )
                        continue
                    analysis["has_ligand"] = True
                    if resname not in analysis["ligand_resnames"]:
                        analysis["ligand_resnames"].append(resname)
                    logger.info(f"Ligand detected from file_registry: {file_path}")
                elif ftype == "ion" and not analysis["has_ions"]:
                    analysis["has_ions"] = True
                    resname = Path(file_path).stem.upper()
                    if resname not in analysis["ion_resnames"]:
                        analysis["ion_resnames"].append(resname)
                    logger.info(f"Ion detected from file_registry: {file_path}")
        
        # Locate actual ligand/ion files in the simsetup directory
        if simsetup_dir and os.path.isdir(simsetup_dir):
            for resname in analysis["ligand_resnames"]:
                for ext in [".pdb", ".gro"]:
                    candidate = Path(simsetup_dir) / f"{resname}{ext}"
                    if candidate.exists() and str(candidate) not in analysis["ligand_files"]:
                        analysis["ligand_files"].append(str(candidate))
                # Also check lowercase
                for ext in [".pdb", ".gro"]:
                    candidate = Path(simsetup_dir) / f"{resname.lower()}{ext}"
                    if candidate.exists() and str(candidate) not in analysis["ligand_files"]:
                        analysis["ligand_files"].append(str(candidate))
            for resname in analysis["ion_resnames"]:
                for ext in [".pdb", ".gro"]:
                    candidate = Path(simsetup_dir) / f"{resname}{ext}"
                    if candidate.exists() and str(candidate) not in analysis["ion_files"]:
                        analysis["ion_files"].append(str(candidate))
                for ext in [".pdb", ".gro"]:
                    candidate = Path(simsetup_dir) / f"{resname.lower()}{ext}"
                    if candidate.exists() and str(candidate) not in analysis["ion_files"]:
                        analysis["ion_files"].append(str(candidate))

        # Preprocess writes a generic ligand.pdb (and ions.pdb), not ATP.pdb.
        # Searching only {resname}.pdb drops ATP from the rebuilt system.gro.
        search_dirs = []
        if simsetup_dir:
            search_dirs.append(simsetup_dir)
        if state:
            pre = state.get("preprocess_directory") or state.get("preprocessing_directory")
            if not pre and state.get("working_directory"):
                pre = str(Path(state["working_directory"]) / "preprocess")
            if pre:
                search_dirs.append(pre)
        if analysis["has_ligand"] and not analysis["ligand_files"]:
            for directory in search_dirs:
                if not directory or not os.path.isdir(directory):
                    continue
                for name in ("ligand.pdb", "ligand.gro"):
                    candidate = Path(directory) / name
                    if candidate.is_file() and str(candidate) not in analysis["ligand_files"]:
                        analysis["ligand_files"].append(str(candidate))
                        logger.info(f"Ligand coordinates from generic file: {candidate}")
                        break
                if analysis["ligand_files"]:
                    break
        if analysis["has_ions"] and not analysis["ion_files"]:
            for directory in search_dirs:
                if not directory or not os.path.isdir(directory):
                    continue
                for name in ("ions.pdb", "ions.gro", "ion.pdb", "ion.gro"):
                    candidate = Path(directory) / name
                    if candidate.is_file() and str(candidate) not in analysis["ion_files"]:
                        analysis["ion_files"].append(str(candidate))
                        logger.info(f"Ion coordinates from generic file: {candidate}")
                        break
                if analysis["ion_files"]:
                    break
        
        # --- Source 3: HETATM scan of pdb_file (fallback) ---
        if os.path.exists(pdb_file):
            try:
                with open(pdb_file, 'r') as f:
                    lines = f.readlines()
                
                heteroatoms = [l for l in lines if l.startswith('HETATM')]
                analysis["estimated_atoms"] = len([l for l in lines if l.startswith(('ATOM', 'HETATM'))])
                
                if not analysis["has_ligand"] and not analysis["has_ions"]:
                    # Only scan HETATM if we haven't already found components
                    het_resnames = set()
                    for l in heteroatoms:
                        parts = l.split()
                        if len(parts) > 3:
                            het_resnames.add(parts[3].strip())
                    
                    ion_names = het_resnames & self.STANDARD_RESNAMES
                    ligand_names = het_resnames & self.KNOWN_LIGAND_RESNAMES
                    
                    phospho_names = {r for r in het_resnames if is_phospho_protein_resname(r)}
                    if phospho_names:
                        logger.info(
                            "Phosphorylated protein residues (not ligands): %s",
                            sorted(phospho_names),
                        )
                    remaining = {
                        r for r in het_resnames
                        if not is_phospho_protein_resname(r)
                    }
                    remaining -= self.STANDARD_RESNAMES
                    remaining -= ligand_names
                    for resname in remaining:
                        atom_count = sum(1 for l in heteroatoms if len(l.split()) > 3 and l.split()[3].strip() == resname)
                        if atom_count > 5:
                            ligand_names.add(resname)
                    
                    if ligand_names:
                        analysis["ligand_resnames"] = sorted(ligand_names)
                        analysis["has_ligand"] = True
                    if ion_names - {'HOH', 'WAT', 'SOL', 'TIP'}:
                        analysis["ion_resnames"] = sorted(ion_names - {'HOH', 'WAT', 'SOL', 'TIP'})
                        analysis["has_ions"] = True
                else:
                    # Still count atoms even if we already know components
                    pass
                    
            except Exception as e:
                logger.warning(f"Could not analyze system: {e}")
        
        # All ligands require parameterization via generate_ligand_parameters
        # (pre-built params are only used if explicitly requested by the user)
        for resname in analysis["ligand_resnames"]:
            analysis["ligand_params_missing"].append(resname)
            logger.info(f"Ligand {resname} requires parameterization via generate_ligand_parameters")
        
        # Determine system type
        if analysis["has_ligand"] and analysis["has_ions"]:
            analysis["system_type"] = "protein_ligand_ion"
        elif analysis["has_ligand"]:
            analysis["system_type"] = "protein_ligand"
        elif analysis["has_ions"]:
            analysis["system_type"] = "protein_ion"
        else:
            analysis["system_type"] = "protein_only"
        
        # Override based on component_selection (user intent)
        # If user explicitly excluded ligand/ions, respect that regardless of PDB contents
        if component_selection:
            if component_selection.get("ligand") is False and analysis["has_ligand"]:
                logger.info(f"Component selection excludes ligand — overriding has_ligand to False "
                           f"(detected: {analysis['ligand_resnames']})")
                analysis["has_ligand"] = False
                analysis["ligand_resnames"] = []
                analysis["ligand_params_available"] = {}
                analysis["ligand_params_missing"] = []
            if component_selection.get("ions") is False and (
                analysis["has_ions"] or analysis.get("ion_files")
            ):
                logger.info(f"Component selection excludes ions — overriding has_ions to False "
                           f"(detected: {analysis['ion_resnames']})")
                analysis["has_ions"] = False
                analysis["ion_resnames"] = []
                analysis["ion_files"] = []
            
            # Recalculate system_type after overrides
            if analysis["has_ligand"] and analysis["has_ions"]:
                analysis["system_type"] = "protein_ligand_ion"
            elif analysis["has_ligand"]:
                analysis["system_type"] = "protein_ligand"
            elif analysis["has_ions"]:
                analysis["system_type"] = "protein_ion"
            else:
                analysis["system_type"] = "protein_only"
            
            # Store component_selection in analysis for downstream use
            analysis["component_selection"] = component_selection
            logger.info(f"System type after component selection: {analysis['system_type']}")
                
        return analysis
    
    def _build_planning_prompt(self, agent_input: SimSetupAgentInput, 
                               analysis: Dict[str, Any], state: MDState) -> str:
        """Build LLM planning prompt - use planner's detailed instructions if available"""
        
        # Check if we have detailed instructions from planner
        if agent_input.additional_instructions:
            logger.info("Using planner's detailed instructions for setup")
            return self._build_prompt_from_planner_instructions(
                agent_input, analysis, agent_input.additional_instructions, state
            )
        
        # Otherwise use standard config-based prompt
        return self._build_standard_planning_prompt(agent_input, analysis)
    
    def _build_prompt_from_planner_instructions(self, agent_input: SimSetupAgentInput,
                                                analysis: Dict[str, Any],
                                                planner_instructions: str,
                                                state: MDState) -> str:
        """Build prompt using planner's detailed natural language instructions"""
        
        # Get available tools for reference
        tool_metadata = get_tool_metadata()
        tools_list = []
        
        for tool_info in tool_metadata.values():
            tool_entry = f"→ {tool_info['name']}\n"
            tool_entry += f"  {tool_info['description']}\n"
            
            if tool_info['args']:
                tool_entry += "  Parameters:\n"
                for arg_name, arg_details in tool_info['args'].items():
                    required = "required" if arg_details['required'] else "optional"
                    desc = arg_details.get('description', 'No description')
                    tool_entry += f"    • {arg_name} ({required}): {desc}\n"
            
            tools_list.append(tool_entry)
        
        tools_list_str = "\n".join(tools_list)
        
        comp_sel = agent_input.component_selection or {}
        include_ligand = comp_sel.get('ligand', analysis.get('has_ligand', False)) is not False
        include_ions = comp_sel.get('ions', analysis.get('has_ions', False)) is not False

        # Format analysis, suppressing excluded component details so the planner
        # is not distracted by source-PDB ligands/ions for protein-only cases.
        analysis_lines = [
            f"- System Type: {analysis.get('system_type', 'unknown')}",
            f"- Estimated Atoms: {analysis.get('estimated_atoms', 0)}",
        ]
        if include_ligand:
            analysis_lines.extend([
                f"- Has Ligand: {analysis.get('has_ligand', False)}",
                f"- Ligand Residues: {analysis.get('ligand_resnames', [])}",
            ])
            if analysis.get('ligand_params_available'):
                for resname, path in analysis['ligand_params_available'].items():
                    analysis_lines.append(f"- Pre-built params for {resname}: AVAILABLE (no need to generate)")
            for resname in analysis.get('ligand_params_missing', []):
                analysis_lines.append(f"- Pre-built params for {resname}: MISSING (must use generate_ligand_parameters)")
        if include_ions:
            analysis_lines.extend([
                f"- Has Ions: {analysis.get('has_ions', False)}",
                f"- Ion Residues: {analysis.get('ion_resnames', [])}",
            ])
        analysis_str = "\n".join(analysis_lines)
        
        # Format component selection context
        if comp_sel:
            comp_lines = [
                f"- Include Protein: {comp_sel.get('protein', True)}",
                f"- Include Ligand: {comp_sel.get('ligand', False)}",
                f"- Include Ions (crystallographic): {comp_sel.get('ions', False)}",
                f"- Keep Crystallographic Water: {comp_sel.get('water', False)}",
            ]
            component_context = "\n**COMPONENT SELECTION (from user goal):**\n" + "\n".join(comp_lines)
            component_context += (
                "\n\nCRITICAL: Only set up simulation for the components marked True above."
                "\n- If ligand is False, do NOT use generate_ligand_parameters or include ligand in topology."
                "\n- If ions (crystallographic) is False, do NOT include crystallographic ions (genion for neutralization is still needed)."
                "\n- Only use build_topology on the protein PDB file, not the full complex."
                "\n\n**IMPORTANT:** 'Keep Crystallographic Water: False' means do NOT keep water from the original PDB file."
                "\n  It does NOT mean skip solvation. Solvation with tip3p water is ALWAYS part of standard simulation setup."
                "\n  The build_simulation_system tool handles solvation automatically — do NOT pass water_model='none'."
                "\n  For neutralize-only systems, ion_concentration=0 is correct (counter-ions only, no bulk salt)."
            )
        else:
            component_context = ""
        
        # Extract file registry information
        file_registry = state.get("file_registry", {})
        if file_registry:
            registry_str = "\n**Files Available from Preprocessing:**\n"
            for file_path, metadata in file_registry.items():
                filename = Path(file_path).name
                file_type = metadata.get("type", "unknown")
                description = metadata.get("description", "")
                file_type_l = str(file_type).lower()
                desc_l = str(description).lower()
                if not include_ligand and (file_type_l == "ligand" or "ligand" in desc_l or "atp" in desc_l):
                    continue
                if not include_ions and (file_type_l == "ion" or "ion" in desc_l or "mg" in desc_l):
                    continue
                registry_str += f"- {filename} (type: {file_type}) - {description}\n"
        else:
            registry_str = "\n**Files Available from Preprocessing:** None registered\n"
        
        return f"""You are a GROMACS molecular dynamics expert executing a detailed plan from the workflow planner.

**System Information:**
- Cleaned PDB: {agent_input.cleaned_pdb}
- Force Field: {agent_input.force_field}
- Water Model: {agent_input.water_model}
- Temperature: {agent_input.temperature} K
- Pressure: {agent_input.pressure} bar
- Production Length: {agent_input.production_ns if agent_input.production_ns is not None else 'not specified'} ns

**System Analysis:**
{analysis_str}
{component_context}
{registry_str}

**DETAILED INSTRUCTIONS FROM PLANNER:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{planner_instructions}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Available Tools:**
{tools_list_str}

**CRITICAL INSTRUCTIONS:**
- You MUST ONLY use the tools listed above - do NOT invent or suggest non-existent tools
- Every "tool_name" in your plan must match exactly one of the tool names listed above
- FORBIDDEN tool names: "none", "manual", "skip", "custom", "placeholder", or any made-up tool
- If a required capability is missing, either:
  a) Use available tools creatively to achieve the same goal, OR
  b) OMIT that step entirely from your plan (do NOT include it with tool_name="none")
- When you cannot perform a step, simply do NOT include it in the steps array

**MULTI-REPLICATE (rep_num > 1):**
- Call `build_simulation_system` **ONCE** into the shared simsetup directory.
- Do NOT create `rep1/`, `rep2/`, `rep01/`, or any per-replicate setup folders.
- Do NOT call build_simulation_system once per replicate.
- Independent production replicates are created later by the HPC agent
  (copy the same simsetup into `hpc/repXX` with different velocity seeds).

**FILE PATH REQUIREMENTS:**
- In tool_params, you MUST specify only FILENAMES, never full paths
- ❌ WRONG: "output_file": "/path/to/file.gro" or "output_file": "working_dir/simsetup/file.gro"
- ✅ CORRECT: "output_file": "file.gro"
- The system automatically handles directory management - all outputs go to working_dir/simsetup/
- Examples:
  • "topology_file": "topol.top" (not "/home/user/working_dir/simsetup/topol.top")
  • "coordinate_file": "processed.gro" (not "working_dir/simsetup/processed.gro")
  • "output_file": "solvated.gro" (not "./solvated.gro")

Your task: Create a detailed, step-by-step execution plan that follows the planner's instructions above.
The plan should specify which tools to call and in what order to achieve the planner's objectives.
ONLY include steps that use valid tools from the list above.

Output as JSON with this structure:
{{
  "reasoning": "How you'll implement the planner's instructions",
  "overview": "High-level summary",
  "steps": [
    {{
      "name": "step name",
      "description": "what it does",
      "tool_name": "tool to call",
      "tool_params": {{"param": "value"}},
      "reason": "why it's needed per planner's instructions"
    }}
  ],
  "potential_issues": ["issue1"],
  "recommendations": ["rec1"]
}}
"""
    
    def _build_standard_planning_prompt(self, agent_input: SimSetupAgentInput,
                                        analysis: Dict[str, Any]) -> str:
        """Build LLM planning prompt from config template using dynamic tool metadata"""
        config_prompt = self.config.get("llm", {}).get("planning_prompt_template", "")
        
        # Get available tools list dynamically from tool metadata
        tool_metadata = get_tool_metadata()
        tools_list = []
        
        for tool_info in tool_metadata.values():
            tool_entry = f"→ {tool_info['name']}\n"
            tool_entry += f"  {tool_info['description']}\n"
            
            if tool_info['args']:
                tool_entry += "  Parameters:\n"
                for arg_name, arg_details in tool_info['args'].items():
                    required = "required" if arg_details['required'] else "optional"
                    desc = arg_details.get('description', 'No description')
                    tool_entry += f"    • {arg_name} ({required}): {desc}\n"
            
            tools_list.append(tool_entry)
        
        tools_list_str = "\n".join(tools_list)
        
        comp_sel = agent_input.component_selection or {}
        include_ligand = comp_sel.get('ligand', analysis.get('has_ligand', False)) is not False
        include_ions = comp_sel.get('ions', analysis.get('has_ions', False)) is not False

        # Format analysis for prompt and suppress excluded source-PDB components.
        analysis_lines = [
            f"- System Type: {analysis.get('system_type', 'unknown')}",
            f"- Estimated Atoms: {analysis.get('estimated_atoms', 0)}",
        ]
        if include_ligand:
            analysis_lines.extend([
                f"- Has Ligand: {analysis.get('has_ligand', False)}",
                f"- Ligand Residues: {analysis.get('ligand_resnames', [])}",
            ])
            if analysis.get('ligand_params_available'):
                for resname, path in analysis['ligand_params_available'].items():
                    analysis_lines.append(f"- Pre-built params for {resname}: AVAILABLE (no need to generate)")
            for resname in analysis.get('ligand_params_missing', []):
                analysis_lines.append(f"- Pre-built params for {resname}: MISSING (must use generate_ligand_parameters)")
        if include_ions:
            analysis_lines.extend([
                f"- Has Ions: {analysis.get('has_ions', False)}",
                f"- Ion Residues: {analysis.get('ion_resnames', [])}",
            ])
        analysis_str = "\n".join(analysis_lines)
        
        # Use template or build basic prompt
        if config_prompt:
            return config_prompt.format(
                cleaned_pdb=agent_input.cleaned_pdb,
                user_goal=agent_input.user_goal,
                force_field=agent_input.force_field,
                water_model=agent_input.water_model,
                temperature=agent_input.temperature,
                pressure=agent_input.pressure,
                production_ns=(agent_input.production_ns if agent_input.production_ns is not None else "not specified"),
                analysis=analysis_str,
                tools_list=tools_list_str
            )
        else:
            # Fallback prompt
            return f"""
You are a GROMACS molecular dynamics simulation expert. Create a setup plan for:

System: {agent_input.cleaned_pdb}
Goal: {agent_input.user_goal}
Production Length: {agent_input.production_ns if agent_input.production_ns is not None else 'not specified'} ns
Analysis: {analysis_str}

Available tools:
{tools_list_str}

**CRITICAL: You MUST ONLY use the tools listed above. Do NOT invent or suggest non-existent tools.**
- For multi-replicate campaigns: call build_simulation_system ONCE (shared simsetup/).
  Do NOT make per-replicate setup directories (rep1/rep2). HPC stages hpc/repXX later.

Return JSON with: reasoning, overview, steps (name, description, tool_name, tool_params, reason)
"""
    
    def _extract_plan_json(self, content: str) -> Dict[str, Any]:
        """Extract and parse JSON plan from LLM response"""
        import re
        
        # Try to find JSON block in response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: return minimal structure
        return {
            "reasoning": content,
            "overview": "Simulation setup plan",
            "steps": []
        }

    def replan_with_guidance(self, human_recommendation: str, state: dict) -> Optional[dict]:
        """Update the structured setup plan by applying human guidance.

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
            .get("setup")
        )

        if current_plan:
            # Modification mode: keep existing plan, apply targeted changes
            tool_metadata = get_tool_metadata()
            tools_list = [
                f"→ {t['name']}: {t['description']}"
                for t in tool_metadata.values()
            ]
            tools_str = "\n".join(tools_list)
            prompt = (
                f"You are updating a GROMACS MD simulation setup execution plan.\n\n"
                f"CURRENT PLAN (JSON):\n```json\n{_j.dumps(current_plan, indent=2)}\n```\n\n"
                f"HUMAN GUIDANCE (apply ONLY these changes):\n{human_recommendation}\n\n"
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
                .get("setup_agent", "")
            )
            augmented = (
                nl_instructions
                + "\n\n**HUMAN RECOMMENDATION (must be followed):**\n" + human_recommendation
            ) if nl_instructions else human_recommendation
            try:
                temperature = float(state.get("temperature", 300.0))
            except (TypeError, ValueError):
                temperature = 300.0
            try:
                pressure = float(state.get("pressure", 1.0))
            except (TypeError, ValueError):
                pressure = 1.0
            agent_input = SimSetupAgentInput(
                cleaned_pdb=state.get("cleaned_pdb") or state.get("pdb_path", ""),
                working_directory=state.get("working_directory", "."),
                force_field=state.get("force_field", "amber99sb-ildn"),
                water_model=state.get("water_model", "tip3p"),
                temperature=temperature,
                pressure=pressure,
                user_goal=state.get("user_goal", ""),
                additional_instructions=augmented,
                component_selection=state.get("component_selection"),
            )
            prompt = self._build_planning_prompt(agent_input, {}, state)

        try:
            resp = self.llm.prompt(prompt, temperature=0.1)
            return self._extract_plan_json(resp)
        except Exception:
            return None

    def _create_fallback_plan(self, agent_input: SimSetupAgentInput, 
                             analysis: Dict[str, Any]) -> SimSetupPlan:
        """
        Create template-based fallback plan when LLM fails.
        Uses build_simulation_system which handles all system types end-to-end.
        """
        system_type = analysis.get("system_type", "protein_only")

        from src.simsetup.system_options import resolve_simsetup_options

        defaults = self.config.get("defaults", {})
        resolved = resolve_simsetup_options(
            user_text=agent_input.user_goal,
            pdb_path=agent_input.cleaned_pdb,
            label=Path(agent_input.working_directory).name,
            defaults=defaults,
        )

        # Build tool params based on what components are present
        tool_params: Dict[str, Any] = {
            "protein_file": agent_input.cleaned_pdb,
            "force_field": agent_input.force_field,
            "water_model": agent_input.water_model,
            "box_type": resolved.get("box_type") or "cubic",
            "box_distance": resolved.get("box_distance") or 1.0,
            "ion_concentration": resolved.get("ion_concentration") if resolved.get("ion_concentration") is not None else 0.15,
            "temperature": agent_input.temperature,
            "pressure": agent_input.pressure,
            "extended_minimization": agent_input.extended_minimization,
        }
        if hasattr(agent_input, 'production_ns') and agent_input.production_ns:
            tool_params["production_ns"] = agent_input.production_ns

        # Add ligand params if present
        if analysis.get("has_ligand"):
            ligand_files = analysis.get("ligand_files", [])
            if ligand_files:
                tool_params["ligand_file"] = ligand_files[0]
            ligand_resnames = analysis.get("ligand_resnames", [])
            if ligand_resnames:
                tool_params["ligand_resname"] = ligand_resnames[0]
            ligand_itps = analysis.get("ligand_itps", [])
            if ligand_itps:
                tool_params["ligand_itp"] = ligand_itps[0]

        # Add ion params if present
        if analysis.get("has_ions"):
            ion_files = analysis.get("ion_files", [])
            if ion_files:
                tool_params["ion_file"] = ion_files[0]
            ion_resnames = analysis.get("ion_resnames", [])
            if ion_resnames:
                tool_params["ion_resname"] = ion_resnames[0]

        steps = []
        # The single-step fallback used to call build_simulation_system without
        # an ITP, so declared ATP was omitted from system.gro. Parameterize first.
        if analysis.get("has_ligand") and not analysis.get("ligand_itps"):
            ligand_pdb = (analysis.get("ligand_files") or [None])[0]
            ligand_resname = (analysis.get("ligand_resnames") or [None])[0]
            if ligand_pdb:
                steps.append(
                    SimSetupStep(
                        name="Parameterize ligand",
                        description=(
                            f"Generate GAFF2 parameters for "
                            f"{ligand_resname or 'ligand'} before the system build"
                        ),
                        tool_name="generate_ligand_parameters",
                        tool_params={
                            "ligand_pdb": ligand_pdb,
                            "net_charge": -4 if str(ligand_resname or "").upper() == "ATP" else None,
                        },
                        reason="Ligand coordinates exist but no topology ITP is available yet",
                    )
                )
        steps.append(
            SimSetupStep(
                name="Build Complete System",
                description=(
                    f"Build complete {system_type} GROMACS system end-to-end: "
                    "topology → merge → box → solvate → ions → MDP → TPR"
                ),
                tool_name="build_simulation_system",
                tool_params=tool_params,
                reason="Single orchestrator handles all system types with modular delegation"
            ),
        )

        return SimSetupPlan(
            reasoning="Using build_simulation_system orchestrator (LLM fallback)",
            overview=f"Standard {system_type} simulation system setup",
            steps=steps,
            potential_issues=[],
            recommendations=["Verify topology parameters before production run"]
        )
    
    def _execute_plan(self, agent_input: SimSetupAgentInput, plan: SimSetupPlan,
                     state: MDState) -> SimSetupResult:
        """
        Execute setup plan step-by-step using tool executor
        Tracks file outputs and chains them between steps
        """
        execution_log = []
        issues = []
        warnings = []
        generated_files = {}
        
        current_gro = None
        topology_file = None
        mdp_files = {}
        box_dimensions = None
        water_molecules = None
        ion_count = None
        ligand_itp_path = None  # Actual ITP from generate_ligand_parameters
        ligand_gro_path = None  # Actual GRO from generate_ligand_parameters
        
        # Get execution limits from config
        agent_config = self.config.get("agent", {})
        max_retries = agent_config.get("max_tool_retries", 2)
        max_steps = agent_config.get("max_total_steps", 20)
        fail_fast = agent_config.get("fail_fast", False)
        
        # Get simsetup directory
        simsetup_dir = state.get("simsetup_directory", self.tool_executor.working_dir)

        from src.simsetup.system_options import (
            SIMSETUP_OVERRIDE_KEYS,
            coerce_simsetup_tool_value,
            resolve_simsetup_options,
            summarize_override_changes,
        )

        simsetup_tool_opts = resolve_simsetup_options(
            state=state,
            user_text=agent_input.user_goal,
            pdb_path=state.get("raw_pdb") or agent_input.cleaned_pdb,
            label=Path(state.get("working_directory", "working_dir")).name,
            defaults=self.config.get("defaults", {}),
        )
        requested_production_ns = simsetup_tool_opts.get("production_ns")
        if requested_production_ns is not None:
            try:
                requested_production_ns = float(requested_production_ns)
            except (TypeError, ValueError):
                requested_production_ns = None

        simsetup_override_tools = {
            "build_simulation_system",
            "build_simulation_box",
            "generate_mdp_files",
            "solvate_system",
            "add_ions",
        }

        log_agent_action(
            "setup",
            "Resolved simsetup parameters (override LLM plan at execution)",
            {
                "note": (
                    "The LLM plan JSON may show different values; "
                    "these resolved parameters are applied when tools run."
                ),
                "sim_label": Path(state.get("working_directory", "working_dir")).name,
                "resolved": {
                    k: simsetup_tool_opts.get(k) for k in SIMSETUP_OVERRIDE_KEYS
                },
            },
        )
        
        try:
            if not plan.steps:
                return SimSetupResult(
                    success=False,
                    report="Setup failed: planning produced no executable steps",
                    issues=["No executable setup steps were generated"],
                    warnings=warnings,
                    generated_files=generated_files,
                    execution_log="No setup steps available to execute"
                )

            # Enforce max steps limit
            if len(plan.steps) > max_steps:
                warnings.append(f"Plan has {len(plan.steps)} steps, limiting to {max_steps}")
                plan.steps = plan.steps[:max_steps]
            
            for i, step in enumerate(plan.steps):
                # Validate tool name - skip invalid placeholder names
                invalid_tools = ["none", "manual", "skip", "custom", "placeholder", ""]
                if not step.tool_name or step.tool_name.lower() in invalid_tools:
                    skip_msg = f"Skipping step {i+1} '{step.name}': invalid tool name '{step.tool_name}'"
                    execution_log.append(f"\n⚠ {skip_msg}")
                    warnings.append(skip_msg)
                    log_agent_action("setup", f"Step {i+1}/{len(plan.steps)} skipped", {
                        "step": step.name,
                        "reason": f"Invalid tool name: {step.tool_name}"
                    })
                    continue  # Skip this step entirely
                
                # Log step start
                log_agent_action("setup", f"Executing step {i+1}/{len(plan.steps)}", {
                    "step": step.name,
                    "tool": step.tool_name
                })
                
                execution_log.append(f"\n--- Step {i+1}: {step.name} ---")
                execution_log.append(f"Description: {step.description}")
                execution_log.append(f"Tool: {step.tool_name}")
                execution_log.append(f"Reason: {step.reason}")
                
                # Prepare tool parameters
                tool_params = dict(step.tool_params) if step.tool_params else {}
                llm_tool_params = dict(tool_params)
                
                # CRITICAL: Normalize all file paths to ensure directory isolation
                # Tools should only receive filenames, and working_dir handles the rest
                from ..utils import normalize_tool_params_for_agent
                tool_params = normalize_tool_params_for_agent(
                    tool_params,
                    simsetup_dir,
                    param_names=[
                        "pdb_file", "coordinate_file", "topology_file", "mdp_file",
                        "protein_file", "ligand_file", "ion_file", "ligand_itp",
                        "protein_pdb", "ligand_pdb", "ion_pdb",
                        "input_structure", "restraint_file", "output_file", "output_path"
                    ]
                )
                
                # Copy input files to simsetup directory if they exist elsewhere
                path_keys = ["pdb_file", "coordinate_file", "topology_file", "mdp_file",
                            "protein_file", "ligand_file", "ion_file", "ligand_itp",
                            "protein_pdb", "ligand_pdb", "ion_pdb",
                            "input_structure", "restraint_file"]
                            
                for path_key in path_keys:
                    if path_key in tool_params:
                        filename = tool_params[path_key]  # Already normalized to filename
                        if isinstance(filename, str) and filename:
                            simsetup_path = Path(simsetup_dir) / filename
                            
                            if not simsetup_path.exists():
                                # Try to find file from state or other directories
                                # Check preprocess directory
                                preprocess_dir = str(Path(state.get("working_directory", "working_dir")) / "preprocess")
                                preprocess_path = Path(preprocess_dir) / filename
                                
                                if preprocess_path.exists():
                                    shutil.copy2(str(preprocess_path), str(simsetup_path))
                                    logger.info(f"Copied {filename} from preprocess to simsetup")
                                else:
                                    # Check if absolute path was stored in state
                                    for state_key in ["cleaned_pdb", "raw_pdb", "topology", "coordinates"]:
                                        state_value = state.get(state_key)
                                        if state_value and Path(state_value).name == filename and Path(state_value).exists():
                                            shutil.copy2(state_value, str(simsetup_path))
                                            logger.info(f"Copied {filename} from {state_value} to simsetup")
                                            break

                # P0: force protein inputs from cleaned_pdb — never trust LLM-hallucinated
                # paths like domain_clean.pdb that do not exist on disk.
                cleaned = state.get("cleaned_pdb")
                if cleaned and Path(cleaned).is_file():
                    dest_name = Path(cleaned).name
                    dest = Path(simsetup_dir) / dest_name
                    if not dest.exists():
                        shutil.copy2(str(cleaned), str(dest))
                        logger.info(
                            "Copied cleaned_pdb %s → simsetup/%s", cleaned, dest_name
                        )
                    for pk in ("protein_file", "pdb_file", "protein_pdb"):
                        if pk not in tool_params:
                            continue
                        prev = tool_params.get(pk)
                        if Path(str(prev or "")).name != dest_name:
                            logger.warning(
                                "Overriding %s=%r → %s (cleaned_pdb)",
                                pk,
                                prev,
                                dest_name,
                            )
                            tool_params[pk] = dest_name
                
                # ALWAYS force output_dir and working_dir to simsetup directory
                # regardless of what the LLM plan suggests — prevents file leaks
                tool_params["output_dir"] = simsetup_dir
                tool_params["working_dir"] = simsetup_dir

                if step.tool_name in simsetup_override_tools:
                    for key in SIMSETUP_OVERRIDE_KEYS:
                        val = coerce_simsetup_tool_value(
                            key, simsetup_tool_opts.get(key)
                        )
                        if val is not None:
                            tool_params[key] = val

                    override_changes = summarize_override_changes(
                        llm_tool_params, tool_params
                    )
                    if override_changes:
                        log_agent_action(
                            "setup",
                            "Applied deterministic overrides (LLM plan → executed)",
                            {
                                "step": step.name,
                                "tool": step.tool_name,
                                "changes": override_changes,
                            },
                        )
                        execution_log.append(
                            "Deterministic overrides applied: "
                            + ", ".join(
                                f"{k}={override_changes[k]['executed']}"
                                for k in sorted(override_changes)
                            )
                        )
                    else:
                        log_agent_action(
                            "setup",
                            "Tool parameters match resolved goal/state values",
                            {
                                "step": step.name,
                                "tool": step.tool_name,
                                "executed": {
                                    k: tool_params.get(k)
                                    for k in SIMSETUP_OVERRIDE_KEYS
                                    if tool_params.get(k) is not None
                                },
                            },
                        )
                
                # Tool-specific default output filenames (relative paths)
                if step.tool_name == "generate_ligand_parameters":
                    # Acpype creates its own subdirectory ({ligand}.acpype/) inside
                    # output_dir, so output_dir must be simsetup_dir (already set above).
                    # Resolve ligand_pdb to absolute path inside simsetup_dir so
                    # acpype can find the file regardless of the process CWD.
                    lpdb = tool_params.get("ligand_pdb", "")
                    if lpdb and not Path(lpdb).is_absolute():
                        tool_params["ligand_pdb"] = str(Path(simsetup_dir) / Path(lpdb).name)
                    resolved_lpdb = Path(str(tool_params.get("ligand_pdb") or ""))
                    if not resolved_lpdb.is_file():
                        generic = Path(simsetup_dir) / "ligand.pdb"
                        if generic.is_file():
                            tool_params["ligand_pdb"] = str(generic)
                            logger.info(
                                "Ligand PDB %s missing; using %s",
                                resolved_lpdb.name or "(unset)",
                                generic.name,
                            )
                    # ATP is -4. A planned charge of 0 makes antechamber fail or
                    # emit a neutral ligand that never lands in system.gro.
                    declared = {
                        str(r).upper()
                        for r in (state.get("ligand_resnames") or [])
                    }
                    ligand_name = Path(str(tool_params.get("ligand_pdb") or "")).stem.upper()
                    if "ATP" in declared or ligand_name == "ATP":
                        try:
                            planned_charge = int(tool_params.get("net_charge"))
                        except (TypeError, ValueError):
                            planned_charge = None
                        if planned_charge in (None, 0):
                            tool_params["net_charge"] = -4
                            logger.info(
                                "ATP net charge set to -4 (plan had %s)",
                                planned_charge,
                            )
                
                elif step.tool_name == "build_topology" and "output_file" not in tool_params:
                    tool_params["output_file"] = "processed.gro"
                    if "topology_file" not in tool_params:
                        tool_params["topology_file"] = "topol.top"
                
                elif step.tool_name == "build_simulation_system":
                    # Override ligand_itp/ligand_file with the actual acpype
                    # outputs if generate_ligand_parameters ran earlier.
                    # The LLM plan often guesses wrong names (e.g. "ATP.itp"
                    # instead of "ATP_GMX.itp").
                    if ligand_itp_path:
                        tool_params["ligand_itp"] = Path(ligand_itp_path).name
                        logger.info(f"Injecting actual ligand_itp: {Path(ligand_itp_path).name}")
                    if ligand_gro_path:
                        tool_params["ligand_file"] = Path(ligand_gro_path).name
                        logger.info(f"Injecting actual ligand_file: {Path(ligand_gro_path).name}")
                    else:
                        # Fallback and some LLM plans name the ligand ATP.pdb.
                        # Preprocess writes ligand.pdb, so ATP never reaches system.gro.
                        comp = state.get("component_selection") or {}
                        if comp.get("ligand") is not False:
                            generic = Path(simsetup_dir) / "ligand.pdb"
                            current_name = Path(str(tool_params.get("ligand_file") or "")).name
                            current = Path(simsetup_dir) / current_name if current_name else None
                            if generic.is_file() and (current is None or not current.is_file()):
                                tool_params["ligand_file"] = generic.name
                                logger.info("Injecting ligand_file=ligand.pdb")
                            if not tool_params.get("ligand_resname"):
                                names = [
                                    str(r) for r in (state.get("ligand_resnames") or [])
                                    if str(r).strip()
                                ]
                                if names:
                                    tool_params["ligand_resname"] = names[0]
                        
                elif step.tool_name == "build_simulation_box" and "output_file" not in tool_params:
                    tool_params["output_file"] = "boxed.gro"
                    
                elif step.tool_name == "solvate_system" and "output_file" not in tool_params:
                    tool_params["output_file"] = "solvated.gro"
                    # Set defaults if not specified
                    if "temperature" not in tool_params:
                        tool_params["temperature"] = 300.0
                    if "pressure" not in tool_params:
                        tool_params["pressure"] = 1.0
                    if requested_production_ns and "production_ns" not in tool_params:
                        tool_params["production_ns"] = requested_production_ns
                    
                    # Execute the tool (will generate all MDP files: minim, nvt, npt, md, ions)
                    mdp_result = self.tool_executor.execute_tool("generate_mdp_files", tool_params)
                    
                    if mdp_result.get("success"):
                        # Extract generated MDP file paths
                        generated_mdp_files = mdp_result.get("mdp_files", {})
                        for phase, mdp_path in generated_mdp_files.items():
                            mdp_files[phase] = mdp_path
                            generated_files[mdp_path] = f"MDP file for {phase} phase"
                            log_file_operation("setup", "create", mdp_path, True)
                        
                        execution_log.append(f"✓ Generated {len(mdp_files)} MDP files: {list(mdp_files.keys())}")
                        log_agent_action("setup", f"MDP files generated", {
                            "count": len(mdp_files),
                            "phases": list(mdp_files.keys())
                        })
                    else:
                        error_msg = mdp_result.get("error", "Unknown error")
                        execution_log.append(f"✗ MDP generation failed: {error_msg}")
                        issues.append(f"MDP generation failed: {error_msg}")
                        log_agent_action("setup", "MDP generation failed", {"error": error_msg})
                        if fail_fast:
                            break
                    
                    continue  # Skip to next step (don't execute tool again below)
                    
                elif step.tool_name == "add_ions":
                    if current_gro:
                        tool_params["coordinate_file"] = Path(current_gro).name  # Just filename
                    if topology_file:
                        tool_params["topology_file"] = Path(topology_file).name  # Just filename
                    
                    # Use ions.mdp if available, otherwise minim.mdp (just filename)
                    if "ions" in mdp_files:
                        tool_params["mdp_file"] = Path(mdp_files["ions"]).name
                    elif "minim" in mdp_files:
                        tool_params["mdp_file"] = Path(mdp_files["minim"]).name
                    else:
                        # MDP files not generated - log error and skip
                        error_msg = "Cannot add ions: No MDP files available. Run generate_mdp_files first."
                        execution_log.append(f"✗ {error_msg}")
                        issues.append(error_msg)
                        log_agent_action("setup", "add_ions skipped", {"error": error_msg})
                        if fail_fast:
                            break
                        continue  # Skip this step
                    
                    tool_params["output_file"] = "system.gro"  # Relative to working_dir
                    tool_params["working_dir"] = simsetup_dir  # Critical for backup files
                
                elif step.tool_name == "generate_tpr_file":
                    # Set up parameters for TPR generation
                    coord_filename = None
                    if current_gro:
                        coord_filename = Path(current_gro).name  # Just filename
                        tool_params["coordinate_file"] = coord_filename
                    if topology_file:
                        tool_params["topology_file"] = Path(topology_file).name  # Just filename
                    
                    # Use minim.mdp if available, or the mdp_file specified in params
                    if "mdp_file" not in tool_params and "minim" in mdp_files:
                        tool_params["mdp_file"] = Path(mdp_files["minim"]).name
                    elif "mdp_file" in tool_params:
                        # If mdp_file is specified, ensure it's just the filename
                        tool_params["mdp_file"] = Path(tool_params["mdp_file"]).name
                    
                    # Set restraint file (typically same as coordinate file for position restraints)
                    # DEFAULT: Use the same file as coordinate_file (most common for position restraints)
                    if "restraint_file" not in tool_params:
                        if coord_filename:
                            tool_params["restraint_file"] = coord_filename
                    elif "restraint_file" in tool_params:
                        tool_params["restraint_file"] = Path(tool_params["restraint_file"]).name
                    
                    # Default output name
                    if "output_tpr" not in tool_params:
                        mdp_name = Path(tool_params.get("mdp_file", "minim.mdp")).stem
                        tool_params["output_tpr"] = f"{mdp_name}.tpr"
                    
                    tool_params["working_dir"] = simsetup_dir
                
                execution_log.append(f"Parameters: {json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in tool_params.items()}, indent=2)}")
                
                # Execute tool with retry logic
                retry_count = 0
                result = None
                
                while retry_count <= max_retries:
                    result = self.tool_executor.execute_tool(step.tool_name, tool_params)
                    
                    if result.get("success"):
                        break  # Success, exit retry loop
                    
                    retry_count += 1
                    if retry_count <= max_retries:
                        execution_log.append(f"⚠ Retry {retry_count}/{max_retries}: {result.get('error')}")
                    else:
                        execution_log.append(f"✗ Max retries ({max_retries}) exceeded")
                
                # Process results
                if result and result.get("success"):
                    execution_log.append(f"✓ Success: {result.get('message', 'Step completed')}")
                    
                    # --- Post-processing: copy acpype outputs to simsetup dir ---
                    if step.tool_name == "generate_ligand_parameters":
                        for key in ("topology", "coordinates"):
                            src_path = result.get(key)
                            if src_path and Path(src_path).exists():
                                dst_path = Path(simsetup_dir) / Path(src_path).name
                                if str(Path(src_path).resolve()) != str(dst_path.resolve()):
                                    shutil.copy2(src_path, str(dst_path))
                                    logger.info(f"Copied {Path(src_path).name} → {simsetup_dir}")
                                    execution_log.append(f"  Copied {Path(src_path).name} to simsetup/")
                                    generated_files[str(dst_path)] = f"Ligand {key} file"
                                    log_file_operation("setup", "create", str(dst_path), True)
                                # Track actual filenames for chaining to next steps
                                if key == "topology":
                                    ligand_itp_path = str(dst_path)
                                elif key == "coordinates":
                                    ligand_gro_path = str(dst_path)
                        # Also copy posre_{RESNAME}.itp from acpype output dir.
                        # Prefer the dedicated acpype_dir key; fall back to output_dir.
                        # (ligand_topology.py already copies them to output_dir, so
                        # either search dir will work, but acpype_dir is authoritative.)
                        _posre_search_dir = result.get("acpype_dir") or result.get("output_dir")
                        if _posre_search_dir:
                            for posre in Path(_posre_search_dir).glob("posre_*.itp"):
                                dst = Path(simsetup_dir) / posre.name
                                if not dst.exists():
                                    shutil.copy2(str(posre), str(dst))
                                    logger.info(f"Copied {posre.name} → {simsetup_dir}")
                                    execution_log.append(f"  Copied {posre.name} to simsetup/")
                                    generated_files[str(dst)] = "Ligand position restraint file"
                                    log_file_operation("setup", "create", str(dst), True)
                    
                    # Update current state from results (store absolute paths internally)
                    if "output_file" in result:
                        output = result["output_file"]
                        # If relative path, make absolute with simsetup_dir
                        if not Path(output).is_absolute():
                            current_gro = str(Path(simsetup_dir) / output)
                        else:
                            current_gro = output
                        generated_files[current_gro] = step.description
                        log_file_operation("setup", "create", current_gro, True)
                        
                    if "topology_file" in result:
                        topo = result["topology_file"]
                        # If relative path, make absolute with simsetup_dir
                        if not Path(topo).is_absolute():
                            topology_file = str(Path(simsetup_dir) / topo)
                        else:
                            topology_file = topo
                        generated_files[topology_file] = "GROMACS topology file"
                        log_file_operation("setup", "create", topology_file, True)
                    
                    # Handle nested output from build_simulation_system
                    # which returns {"files": {"system_gro": ..., "topology": ...}}
                    if "files" in result and isinstance(result["files"], dict):
                        files_dict = result["files"]
                        if not current_gro and "system_gro" in files_dict:
                            output = files_dict["system_gro"]
                            if not Path(output).is_absolute():
                                current_gro = str(Path(simsetup_dir) / output)
                            else:
                                current_gro = output
                            generated_files[current_gro] = step.description
                            log_file_operation("setup", "create", current_gro, True)
                        if not topology_file and "topology" in files_dict:
                            topo = files_dict["topology"]
                            if not Path(topo).is_absolute():
                                topology_file = str(Path(simsetup_dir) / topo)
                            else:
                                topology_file = topo
                            generated_files[topology_file] = "GROMACS topology file"
                            log_file_operation("setup", "create", topology_file, True)
                        if "minim_tpr" in files_dict:
                            tpr = files_dict["minim_tpr"]
                            if not Path(tpr).is_absolute():
                                tpr_path = str(Path(simsetup_dir) / tpr)
                            else:
                                tpr_path = tpr
                            generated_files[tpr_path] = "GROMACS binary run input file (TPR)"
                            log_file_operation("setup", "create", tpr_path, True)
                        if "chain_residue_map" in files_dict:
                            cmap = files_dict["chain_residue_map"]
                            if not Path(cmap).is_absolute():
                                cmap_path = str(Path(simsetup_dir) / cmap)
                            else:
                                cmap_path = cmap
                            generated_files[cmap_path] = (
                                "PDB chain + resid → trajectory resindex map"
                            )
                            log_file_operation("setup", "create", cmap_path, True)
                        if "mdp_files" in files_dict and isinstance(files_dict["mdp_files"], dict):
                            mdp_files.update(files_dict["mdp_files"])
                        
                    if "box_dimensions" in result:
                        box_dimensions = result["box_dimensions"]
                        
                    if "water_molecules" in result:
                        water_molecules = result["water_molecules"]
                        
                    if "ions_added" in result:
                        ion_count = result["ions_added"]
                    
                    if "output_tpr" in result:
                        tpr = result["output_tpr"]
                        # If relative path, make absolute with simsetup_dir
                        if not Path(tpr).is_absolute():
                            tpr_path = str(Path(simsetup_dir) / tpr)
                        else:
                            tpr_path = tpr
                        generated_files[tpr_path] = "GROMACS binary run input file (TPR)"
                        log_file_operation("setup", "create", tpr_path, True)
                    
                    # Log step completion
                    log_agent_action("setup", f"Step {i+1}/{len(plan.steps)} completed", {
                        "step": step.name,
                        "tool": step.tool_name,
                        "status": "✅ SUCCESS"
                    })
                    
                    if result.get("warning"):
                        warnings.append(f"{step.name}: {result['warning']}")
                else:
                    error_msg = result.get('error', 'Failed to execute') if result else 'Tool execution failed'
                    execution_log.append(f"✗ Failed after {retry_count} attempts: {error_msg}")
                    issues.append(f"{step.name}: {error_msg}")
                    
                    log_agent_action("setup", f"Step {i+1}/{len(plan.steps)} failed", {
                        "step": step.name,
                        "tool": step.tool_name,
                        "status": "❌ FAILED",
                        "error": error_msg
                    })
                    
                    if fail_fast:
                        execution_log.append("⚠ Stopping execution (fail_fast enabled)")
                        break
            
            execution_log_str = "\n".join(execution_log)

            # Treat missing primary artifacts as setup failure even if tools did
            # not return explicit errors.
            if len(issues) == 0:
                if not current_gro:
                    issues.append("Setup did not produce coordinates (.gro) output")
                if not topology_file:
                    issues.append("Setup did not produce topology (.top) output")
            
            return SimSetupResult(
                success=len(issues) == 0,
                coordinates=current_gro,
                topology=topology_file,
                mdp_files=mdp_files,
                box_dimensions=box_dimensions,
                water_molecules=water_molecules,
                ion_count=ion_count,
                report=f"Setup completed: {len(plan.steps)} steps executed",
                issues=issues,
                warnings=warnings,
                generated_files=generated_files,
                execution_log=execution_log_str
            )
            
        except Exception as e:
            return SimSetupResult(
                success=False,
                report=f"Setup failed: {str(e)}",
                issues=[str(e)],
                warnings=warnings,
                generated_files=generated_files,
                execution_log="\n".join(execution_log)
            )
    
    def _update_state(self, state: MDState, agent_output: SimSetupAgentOutput):
        """Update workflow state with setup results"""
        if agent_output.supervisor_update:
            state.update(agent_output.supervisor_update)
        
        state["setup_report"] = agent_output.result.report
        state["setup_issues"] = agent_output.result.issues
        state["setup_warnings"] = agent_output.result.warnings
        state["setup_execution_log"] = agent_output.result.execution_log
        
        # Initialize file registry if not present
        if "file_registry" not in state or state["file_registry"] is None:
            state["file_registry"] = {}
        
        # Register all generated files with metadata
        for file_path, description in agent_output.result.generated_files.items():
            # Determine file type from filename
            file_type = "unknown"
            if ".top" in file_path:
                file_type = "topology"
            elif ".gro" in file_path:
                file_type = "coordinates"
            elif ".mdp" in file_path:
                file_type = "mdp"
            elif ".itp" in file_path:
                file_type = "topology_include"
            elif file_path.endswith("chain_residue_map.json"):
                file_type = "chain_residue_map"
                state["chain_residue_map"] = file_path
            
            state["file_registry"][file_path] = {
                "type": file_type,
                "description": description,
                "stage": "simsetup"
            }
