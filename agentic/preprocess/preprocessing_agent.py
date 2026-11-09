"""
Preprocessing Agent - Orchestrates PDB structure preparation for MD simulations
Focuses on high-level workflow coordination, delegates tool execution to tools.py
"""
import logging
import json
import yaml
import os
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
    PreprocessingPlan, PreprocessingStep,
    PreprocessingResult, PreprocessingAgentInput, PreprocessingAgentOutput,
    StructureRequest,
)
from src.preprocess.structure_request_parser import (
    parse_structure_request,
    build_output_basename,
    build_domain_context_for_agents,
    resolve_domain_residue_range,
)
from src.preprocess.structure_acquisition import acquire_structure_from_request
from src.preprocess.structure_scope import (
    allowed_structure_ids,
    sibling_structure_ids,
    structure_fetch_skip_reason,
)
from src.preprocess.domain_extractor import (
    extract_domain,
    classify_domain_extract,
    _atom_record_count,
)
from .tools import PreprocessingToolExecutor, get_tool_metadata

logger = logging.getLogger(__name__)


class PreprocessingAgent:
    """
    LLM-powered preprocessing agent - orchestrates PDB preparation workflow
    Delegates tool execution to PreprocessingToolExecutor from tools.py
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config_path: Optional[str] = None):
        """
        Initialize preprocessing agent
        
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
        """Load preprocessing configuration from YAML"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def preprocess_node(self, state: MDState) -> MDState:
        """
        Main preprocessing node - entry point from workflow
        Orchestrates the entire preprocessing pipeline
        """
        # Check if we have planner instructions
        execution_plan = state.get("execution_plan", {})
        has_planner_instructions = execution_plan.get("format") == "natural_language"
        
        input_summary = {
            "raw_pdb": state.get("raw_pdb"),
            "force_field": state.get("force_field"),
            "water_model": state.get("water_model")
        }
        
        if has_planner_instructions:
            # Prefer pre-extracted instructions from supervisor (avoids duplication)
            preprocessing_section = state.get("preprocessing_instructions")
            
            if not preprocessing_section:
                # Fallback: Extract from full plan if supervisor didn't provide it
                full_plan = execution_plan.get("full_plan", "")
                preprocessing_section = self._extract_agent_instructions(full_plan, "Preprocessing Agent")
            
            if preprocessing_section:
                input_summary["planner_instructions"] = preprocessing_section
            else:
                input_summary["planner_instructions"] = "[Natural language plan from planner - see above]"
        else:
            # Fallback to user goal if no planner instructions
            input_summary["user_goal"] = state.get("user_goal")
        
        log_agent_start("preprocessing", "PDB Preprocessing with LLM Tool Calling", input_summary)
        
        try:
            # Initialize secure file manager
            base_working_dir = state.get("working_directory", "working_dir")
            file_registry = state.get("file_registry", {})
            
            self.file_manager = SecureFileManager(
                working_dir=base_working_dir,
                agent_name="preprocess",
                file_registry=file_registry
            )
            
            logger.info(f"Preprocessing agent directory: {self.file_manager.agent_dir}")
            
            # Get preprocess directory from file manager (ensures consistency)
            preprocess_dir = self.file_manager.agent_dir
            
            # CRITICAL: Set preprocess_directory in state BEFORE creating tool executor
            # This allows tools to access the directory during execution
            state["preprocess_directory"] = preprocess_dir
            
            self.tool_executor = PreprocessingToolExecutor(preprocess_dir, self.config)
            
            # Prepare agent input from state
            agent_input = self._prepare_agent_input(state)
            
            # Run LLM-guided preprocessing workflow
            agent_output = self._run_preprocessing_workflow(agent_input, state)
            
            # Update state with results
            self._update_state(state, agent_output)
            
            # Determine next workflow node
            if agent_output.success:
                # Clear any previous preprocessing-related errors from earlier retry attempts
                state["errors"] = [
                    e for e in state.get("errors", [])
                    if not (e.startswith("Preprocessing failed:") or e.startswith("Preprocessing error:"))
                ]
                if hitl_should_interact(state):
                    state["next_node"] = "human_preprocess_check"
                else:
                    state["next_node"] = "supervisor"
            else:
                state["errors"].append(f"Preprocessing failed: {agent_output.result.report}")
                # Route to human checkpoint on failure (error-triggered HITL)
                state["next_node"] = "human_preprocess_check"
                state["error_triggered_hitl"] = True
            
            success = agent_output.success and len(agent_output.result.issues) == 0
            log_agent_completion("preprocessing", "PDB Preprocessing", state, success)
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Preprocessing agent failed: {e}")
            logger.error(f"Traceback: {tb}")
            log_error("preprocessing_agent.preprocess_node", e, {"state": state})
            state["errors"].append(f"Preprocessing error: {str(e)}")
            # Error-triggered HITL
            state["next_node"] = "human_preprocess_check"
            state["error_triggered_hitl"] = True
        
        return state
    
    def _extract_agent_instructions(self, full_plan: str, agent_name: str) -> Optional[str]:
        """
        Extract agent-specific detailed instructions from planner's natural language plan.
        
        Args:
            full_plan: Complete natural language plan from planner
            agent_name: Name of the agent section to extract (e.g., "Preprocessing Agent")
            
        Returns:
            Extracted instructions for this specific agent, or full plan as fallback
        """
        import re
        
        # Try multiple patterns to find the agent section (in priority order)
        patterns = [
            # New standardized format: **Agent Name:**
            rf'\*\*{agent_name}:\*\*\s*\n(.*?)(?=\n\s*\*\*(?:Simulation Setup Agent|HPC Agent|Analysis Agent|Expected Outcomes):|$)',
            # Alternative with optional colon
            rf'\*\*{agent_name}\*\*:?\s*\n(.*?)(?=\n\s*\*\*[A-Z]|\Z)',
            # Markdown headings
            rf'###\s*{agent_name}.*?\n(.*?)(?=###|\Z)',
            rf'##\s*{agent_name}.*?\n(.*?)(?=##|\Z)',
            # Fuzzy match patterns (allow for variations like "PDB Preprocessing Agent")
            rf'\*\*(?:PDB\s*)?Preprocess(?:ing)?\s*Agent\*\*:?\s*\n(.*?)(?=\n\s*\*\*[A-Z]|\Z)',
            rf'###\s*(?:PDB\s*)?Preprocess.*?Agent.*?\n(.*?)(?=###|\Z)',
            rf'##\s*(?:PDB\s*)?Preprocess.*?Agent.*?\n(.*?)(?=##|\Z)',
            # Section number patterns
            rf'\d+\..*?(?:PDB\s*)?Preprocess.*?Agent.*?\n(.*?)(?=\d+\.|\Z)',
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
    
    def _prepare_agent_input(self, state: MDState) -> PreprocessingAgentInput:
        """Prepare structured input for preprocessing from workflow state"""
        defaults = self.config.get("defaults", {})
        
        # Check if planner provided detailed instructions for this agent
        # Prefer pre-extracted instructions from supervisor
        planner_instructions = state.get("preprocessing_instructions")
        
        if not planner_instructions:
            # Fallback: Extract from execution_plan if supervisor didn't provide it
            execution_plan = state.get("execution_plan", {})
            if execution_plan.get("format") == "natural_language":
                full_plan = execution_plan.get("full_plan", "")
                planner_instructions = self._extract_agent_instructions(full_plan, "Preprocessing Agent")
        
        # Append human recommendation to instructions so the LLM planner sees it
        human_rec = state.get("human_recommendation")
        if human_rec:
            rec_block = (
                f"\n\n**HUMAN RECOMMENDATION (must be followed):**\n{human_rec}\n"
                "Adjust the preprocessing plan to incorporate this recommendation."
            )
            if planner_instructions:
                planner_instructions += rec_block
            else:
                planner_instructions = rec_block
            
        structure_request = self._resolve_structure_request(state, planner_instructions)

        return PreprocessingAgentInput(
            pdb_path=state.get("raw_pdb", "") or "",
            working_directory=state.get("working_directory", "working_dir"),
            force_field=state.get("force_field", defaults.get("force_field", "amber99sb-ildn")),
            water_model=state.get("water_model", defaults.get("water_model", "tip3p")),
            remove_waters=state.get("remove_waters", defaults.get("remove_waters", True)),
            add_hydrogens=state.get("add_hydrogens", defaults.get("add_hydrogens", True)),
            user_goal=state.get("user_goal", ""),
            additional_instructions=planner_instructions,
            component_selection=state.get("component_selection"),
            structure_request=structure_request,
        )

    def _resolve_structure_request(
        self, state: MDState, planner_instructions: Optional[str]
    ) -> Optional[StructureRequest]:
        """Build StructureRequest from state or parse user/planner text."""
        raw = state.get("structure_request")
        if isinstance(raw, dict) and raw.get("uniprot_id"):
            return StructureRequest(**raw)

        combined_text = "\n".join(
            part for part in (
                state.get("user_goal_original", ""),
                state.get("user_goal", ""),
                state.get("master_enriched_prompt", ""),
                planner_instructions or "",
                state.get("preprocessing_instructions", "") or "",
            )
            if part
        )
        parsed = parse_structure_request(combined_text)
        if not parsed:
            return None

        defaults = self.config.get("structure_acquisition", {})
        start_resid = parsed.get("start_resid")
        end_resid = parsed.get("end_resid")
        domain_label = parsed.get("domain_label")

        domain_lookup: Dict[str, Any] = {}
        if domain_label and parsed.get("uniprot_id") and not (start_resid and end_resid):
            domain_lookup = resolve_domain_residue_range(
                parsed["uniprot_id"], domain_label
            )
            state["domain_lookup"] = domain_lookup
            log_agent_action(
                "preprocessing",
                "Domain residue-range lookup",
                {
                    "uniprot_id": parsed["uniprot_id"],
                    "domain_label": domain_label,
                    "success": bool(domain_lookup.get("domain_lookup_success")),
                    "source": domain_lookup.get("domain_lookup_source"),
                    "range": domain_lookup.get("residue_range"),
                    "description": domain_lookup.get("domain_lookup_description"),
                    "message": domain_lookup.get("domain_lookup_message")
                    or domain_lookup.get("domain_lookup_error"),
                },
            )
            if domain_lookup.get("domain_lookup_success"):
                start_resid = domain_lookup.get("start_resid")
                end_resid = domain_lookup.get("end_resid")
                parsed = {**parsed, **domain_lookup}

        req = StructureRequest(
            uniprot_id=parsed["uniprot_id"],
            protein_name=parsed.get("protein_name"),
            domain_label=domain_label,
            start_resid=start_resid,
            end_resid=end_resid,
            extract_domain=bool(domain_label and start_resid and end_resid),
            needs_download=parsed.get("needs_download", True),
            structure_source=parsed.get("structure_source")
            or defaults.get("default_source", "auto"),
        )
        domain_ctx = build_domain_context_for_agents({**parsed, **req.model_dump()})
        if domain_ctx:
            state["domain_context"] = domain_ctx
        return req

    def _combined_structure_text(
        self, agent_input: PreprocessingAgentInput, state: MDState
    ) -> str:
        return "\n".join(
            part for part in (
                state.get("user_goal_original", ""),
                agent_input.user_goal,
                state.get("master_enriched_prompt", ""),
                agent_input.additional_instructions or "",
                state.get("preprocessing_instructions", "") or "",
            )
            if part
        )

    def _ensure_input_structure(
        self, agent_input: PreprocessingAgentInput, state: MDState
    ) -> PreprocessingAgentInput:
        """
        Download and extract structure when no local PDB exists but the user
        described a UniProt accession / domain in their request.
        """
        req = agent_input.structure_request
        if not req:
            # Still try to infer domain trimming from free text alone
            parsed = parse_structure_request(self._combined_structure_text(agent_input, state))
            if parsed and parsed.get("start_resid") and parsed.get("end_resid"):
                req = StructureRequest(
                    uniprot_id=parsed.get("uniprot_id") or Path(agent_input.pdb_path).stem.upper(),
                    protein_name=parsed.get("protein_name"),
                    domain_label=parsed.get("domain_label"),
                    start_resid=parsed.get("start_resid"),
                    end_resid=parsed.get("end_resid"),
                    extract_domain=True,
                    needs_download=False,
                    structure_source=parsed.get("structure_source", "auto"),
                )
                agent_input.structure_request = req
            else:
                return agent_input

        preprocess_dir = str(self.tool_executor.working_dir)
        combined_text = self._combined_structure_text(agent_input, state)
        pdb_path = agent_input.pdb_path

        if not pdb_path or not os.path.isfile(pdb_path):
            logger.info(
                "No local PDB found — acquiring structure for UniProt %s",
                req.uniprot_id,
            )
            log_agent_action(
                "preprocessing",
                "Acquiring structure from database",
                {"uniprot_id": req.uniprot_id, "source": req.structure_source},
            )

            result = acquire_structure_from_request(
                text=combined_text,
                working_dir=preprocess_dir,
                source=req.structure_source,
            )

            if not result.get("success"):
                logger.error("Structure acquisition failed: %s", result.get("error"))
                state.setdefault("errors", []).append(
                    f"Structure acquisition failed: {result.get('error')}"
                )
                return agent_input

            acquired_pdb = result.get("pdb_file") or result.get("output_file")
            if acquired_pdb and os.path.isfile(acquired_pdb):
                agent_input.pdb_path = acquired_pdb
                state["raw_pdb"] = acquired_pdb
                state["structure_acquisition_log"] = result.get("log", [])
                state["structure_acquisition_result"] = result
                log_file_operation(
                    "preprocessing", "create", acquired_pdb, True,
                    "Downloaded/trimmed structure from database",
                )
                logger.info("Acquired structure: %s", acquired_pdb)

        return self._apply_domain_trim_if_requested(agent_input, state, req)

    def _apply_domain_trim_if_requested(
        self,
        agent_input: PreprocessingAgentInput,
        state: MDState,
        req: StructureRequest,
    ) -> PreprocessingAgentInput:
        """Trim an existing/full PDB to the requested residue range."""
        if not req.extract_domain or not req.start_resid or not req.end_resid:
            return agent_input

        pdb_path = agent_input.pdb_path
        if not pdb_path or not os.path.isfile(pdb_path):
            return agent_input

        preprocess_dir = str(self.tool_executor.working_dir)
        basename = build_output_basename(
            {
                "protein_name": req.protein_name or Path(pdb_path).stem,
                "domain_label": req.domain_label or "domain",
                "extract_domain": True,
                "uniprot_id": req.uniprot_id,
            }
        )
        domain_pdb = str(Path(preprocess_dir) / f"{basename}.pdb")

        if Path(domain_pdb).resolve() == Path(pdb_path).resolve():
            return agent_input

        decision = classify_domain_extract(pdb_path, req.start_resid, req.end_resid)
        if decision in ("already_present", "numbering_mismatch", "empty_input"):
            logger.info(
                "Skipping domain extract (%s) on %s for requested residues %s-%s; "
                "keeping the local PDB",
                decision,
                Path(pdb_path).name,
                req.start_resid,
                req.end_resid,
            )
            return agent_input

        if os.path.isfile(domain_pdb) and _atom_record_count(domain_pdb) > 0:
            agent_input.pdb_path = domain_pdb
            state["raw_pdb"] = domain_pdb
            return agent_input

        logger.info(
            "Extracting domain residues %s-%s from %s",
            req.start_resid,
            req.end_resid,
            Path(pdb_path).name,
        )
        result = extract_domain.func(
            pdb_file=pdb_path,
            start_resid=req.start_resid,
            end_resid=req.end_resid,
            output_file=domain_pdb,
            protein_name=req.protein_name,
            domain_name=req.domain_label or "domain",
        )
        if result.get("skipped"):
            logger.info("Domain extract skipped: %s", result.get("message"))
            return agent_input
        if not result.get("success"):
            state.setdefault("errors", []).append(
                f"Domain extraction failed: {result.get('error')}"
            )
            return agent_input

        agent_input.pdb_path = domain_pdb
        state["raw_pdb"] = domain_pdb
        log_file_operation(
            "preprocessing", "create", domain_pdb, True,
            f"Extracted domain residues {req.start_resid}-{req.end_resid}",
        )
        return agent_input
    
    def _run_preprocessing_workflow(self, agent_input: PreprocessingAgentInput, 
                                    state: MDState) -> PreprocessingAgentOutput:
        """
        Run complete preprocessing workflow:
        1. LLM creates intelligent plan based on PDB analysis
        2. Execute plan step-by-step using tools
        3. Return structured results
        """
        try:
            # Step 0: Download/trim structure when user provided UniProt but no PDB
            agent_input = self._ensure_input_structure(agent_input, state)

            # Step 1: LLM analyzes PDB and creates plan.
            # When human_recommendation is set, replan on top of the existing plan
            # so the recommendation is actually applied (not bypassed).
            exec_plan = state.get("execution_plan")
            human_rec = state.get("human_recommendation")
            if human_rec:
                logger.info("preprocessing: replanning with human guidance: %s", human_rec[:120])
                _updated = self.replan_with_guidance(human_rec, state)
                if _updated:
                    plan = PreprocessingPlan(
                        reasoning=_updated.get("reasoning", ""),
                        overview=_updated.get("overview", ""),
                        steps=[
                            PreprocessingStep(
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
                    plan = self._create_preprocessing_plan(agent_input, state)
            else:
                plan = self._create_preprocessing_plan(agent_input, state)

            log_agent_action("preprocessing", "Generated preprocessing plan", {
                "steps": len(plan.steps),
                "reasoning": plan.reasoning[:200]
            })

            # Persist structured plan to state for HITL inspection/modification
            if exec_plan is not None:
                exec_plan.setdefault("structured_plans", {})["preprocess"] = plan.model_dump()
            
            # Step 2: Execute plan using tool executor
            result = self._execute_plan(agent_input, plan, state)

            ligand_paths = set(result.ligand_files or [])
            ion_paths = set(result.ion_files or [])
            ligand_resnames = {res.lower() for res in (result.ligand_resnames or [])}
            ion_resnames = {res.lower() for res in (result.ion_resnames or [])}
            
            # Step 3: Register all created files using SecureFileManager
            # Register protein file
            if result.cleaned_pdb:
                self.file_manager.register_external_file(
                    file_path=result.cleaned_pdb,
                    file_type="protein",
                    description="Preprocessed protein structure with hydrogens"
                )
            
            # Register all generated files from preprocessing
            for file_path, description in result.generated_files.items():
                if file_path == result.cleaned_pdb:
                    continue  # Already registered above
                
                stem = Path(file_path).stem.lower()
                description_l = (description or "").lower()
                if file_path in ligand_paths or stem in ligand_resnames or description_l.startswith("ligand"):
                    self.file_manager.register_external_file(
                        file_path=file_path,
                        file_type="ligand",
                        description=description or "Preprocessed ligand structure"
                    )
                elif file_path in ion_paths or stem in ion_resnames or description_l.startswith("ion"):
                    self.file_manager.register_external_file(
                        file_path=file_path,
                        file_type="ion",
                        description=description or "Preprocessed ion structure"
                    )
                else:
                    # Generic file registration
                    self.file_manager.register_external_file(
                        file_path=file_path,
                        file_type="other",
                        description=description or "Preprocessed file"
                    )
            
            # Step 4: Prepare supervisor update - preprocessing outputs cleaned PDB and file registry
            from src.simsetup.minimization_options import should_use_extended_minimization

            preprocess_dir = str(Path(self.tool_executor.working_dir))
            extended_minim = should_use_extended_minimization(
                user_text=agent_input.user_goal,
                preprocess_dir=preprocess_dir,
            )
            if not extended_minim:
                extended_minim = any(
                    "merged_missing" in path or "remodel" in (desc or "").lower()
                    for path, desc in result.generated_files.items()
                )
            supervisor_update = {
                "cleaned_pdb": result.cleaned_pdb,
                "preprocessing_report": result.report,
                "file_registry": self.file_manager.file_registry,
                "extended_minimization": extended_minim,
            }
            
            return PreprocessingAgentOutput(
                success=result.success,
                plan=plan,
                result=result,
                supervisor_update=supervisor_update
            )
            
        except Exception as e:
            logger.error(f"Preprocessing workflow failed: {e}")
            return PreprocessingAgentOutput(
                success=False,
                plan=PreprocessingPlan(
                    reasoning=f"Error in planning: {str(e)}",
                    overview="Failed",
                    steps=[]
                ),
                result=PreprocessingResult(
                    success=False,
                    report=f"Error: {str(e)}",
                    issues=[str(e)],
                    warnings=[]
                ),
                supervisor_update={}
            )
    
    def _create_preprocessing_plan(
        self,
        agent_input: PreprocessingAgentInput,
        state: Optional[MDState] = None,
    ) -> PreprocessingPlan:
        """
        Use LLM to analyze PDB and create intelligent preprocessing plan
        Falls back to template-based plan if LLM fails
        """
        # First, analyze the PDB structure
        analysis_result = self.tool_executor.execute_tool(
            "analyze_pdb",
            {"pdb_file": agent_input.pdb_path}
        )
        
        analysis = analysis_result.get("analysis", {}) if analysis_result.get("success") else {}

        if state and state.get("reuse_hpc"):
            logger.info("reuse_hpc: skipping LLM preprocess planning, using deterministic fallback")
            return self._create_fallback_plan(agent_input, analysis)
        
        # Build LLM prompt from config template
        prompt = self._build_planning_prompt(agent_input, analysis)
        
        try:
            response = self.llm.invoke([prompt])
            content = response.content or ""
            
            log_llm_interaction("preprocessing.planning", prompt, content,
                              is_mock=hasattr(self.llm, '_is_mock_mode') and self.llm._is_mock_mode)
            
            # Parse LLM response into structured plan
            plan_dict = self._extract_plan_json(content)

            # Avoid accepting malformed LLM output as a valid no-op run.
            if not isinstance(plan_dict.get("steps"), list) or not plan_dict.get("steps"):
                logger.warning("LLM preprocessing plan had no executable steps; using fallback plan")
                return self._create_fallback_plan(agent_input, analysis)
            
            return PreprocessingPlan(
                reasoning=plan_dict.get("reasoning", content[:500]),
                overview=plan_dict.get("overview", "Processing PDB for MD simulation"),
                steps=[
                    PreprocessingStep(
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
            return self._create_fallback_plan(agent_input, analysis)
    
    def _simulation_scope_note(self, agent_input: PreprocessingAgentInput) -> str:
        """Tell the planner this directory belongs to one protein."""
        req = agent_input.structure_request
        uid = req.uniprot_id if req else ""
        label = Path(agent_input.working_directory).name if agent_input.working_directory else ""
        who = uid or label or "the staged structure"
        return (
            f"**THIS DIRECTORY IS ONE SIMULATION ONLY ({who}):** "
            "The study goal may name other proteins. Do not download, extract, "
            "or write any structure except the one staged for this directory. "
            "Other proteins are preprocessed in their own directories."
        )

    def _build_planning_prompt(self, agent_input: PreprocessingAgentInput, 
                               analysis: Dict[str, Any]) -> str:
        """Build LLM planning prompt - use planner's detailed instructions if available"""
        
        # Check if we have detailed instructions from planner
        if agent_input.additional_instructions:
            logger.info("Using planner's detailed instructions for preprocessing")
            return self._build_prompt_from_planner_instructions(
                agent_input, analysis, agent_input.additional_instructions
            )
        
        # Otherwise use standard config-based prompt
        return self._build_standard_planning_prompt(agent_input, analysis)
    
    def _build_prompt_from_planner_instructions(self, agent_input: PreprocessingAgentInput,
                                                analysis: Dict[str, Any],
                                                planner_instructions: str) -> str:
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
        
        # Format analysis
        analysis_str = "\n".join([
            f"- Atoms: {analysis.get('atom_count', 'unknown')}",
            f"- Chains: {analysis.get('chain_ids', [])}",
            f"- Has Waters: {analysis.get('has_waters', False)}",
            f"- Heteroatoms: {analysis.get('heteroatoms', [])}",
            f"- Alternate Locations: {analysis.get('alternate_locations', False)}"
        ])
        
        # Format component selection context
        comp_sel = agent_input.component_selection or {}
        if comp_sel:
            comp_lines = [
                f"- Include Protein: {comp_sel.get('protein', True)}",
                f"- Include Ligand: {comp_sel.get('ligand', False)}",
                f"- Include Ions: {comp_sel.get('ions', False)}",
                f"- Keep Crystallographic Water: {comp_sel.get('water', False)}",
            ]
            comp_str = "\n".join(comp_lines)
            component_context = f"""
**COMPONENT SELECTION (from user goal):**
{comp_str}

IMPORTANT: After separating complex components, ONLY keep the components marked True above.
- If ligand is True, keep ligand files for handoff to Setup Agent
- If ions is True, keep ion files for handoff to Setup Agent
- If a component is False, do NOT include it in generated_files or pass it downstream
- The Setup Agent will only process components that preprocessing provides"""
        else:
            component_context = ""
        
        return f"""You are a molecular dynamics preprocessing expert executing a detailed plan from the workflow planner.

**PDB Information:**
- File: {agent_input.pdb_path}
- Force Field: {agent_input.force_field}
- Water Model: {agent_input.water_model}

**Structure Analysis:**
{analysis_str}
{component_context}

**DETAILED INSTRUCTIONS FROM PLANNER:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{planner_instructions}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Available Tools:**
{tools_list_str}

**CRITICAL: DO NOT PUT FILE PATHS IN tool_params**
- The executor assigns canonical preprocess filenames (input_experimental.pdb, input_donor.pdb,
  protein.pdb, remodeled.pdb, protein_h.pdb). You choose tool_name and settings only (mode, ph, …).
- Leave tool_params empty or with non-path options only — never pdb_file, target_pdb, donor_pdb, output_file.

{self._simulation_scope_note(agent_input)}

**CRITICAL INSTRUCTIONS:**
- You MUST ONLY use the tools listed above - do NOT invent or suggest non-existent tools
- If a required capability is missing, use the available tools creatively or skip that step
- Every "tool_name" in your plan must match exactly one of the tool names listed above
- Do NOT create placeholder tools like "custom_file_filter", "none", or "manual"
- Do NOT add file concatenation/merging steps - that's handled by the Setup Agent

Your task: Create a detailed, step-by-step execution plan that follows the planner's instructions above.
The plan should specify which tools to call and in what order to achieve the planner's objectives.

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
    
    def _build_standard_planning_prompt(self, agent_input: PreprocessingAgentInput,
                                        analysis: Dict[str, Any]) -> str:
        """Build LLM planning prompt from config template using dynamic tool metadata"""
        config_prompt = self.config.get("llm", {}).get("planning_prompt_template", "")
        
        # Get available tools list dynamically from tool metadata - properly formatted
        tool_metadata = get_tool_metadata()
        tools_list = []
        
        for tool_info in tool_metadata.values():
            tool_entry = f"→ {tool_info['name']}\n"
            tool_entry += f"  {tool_info['description']}\n"
            
            # Format parameters nicely
            if tool_info['args']:
                tool_entry += "  Parameters:\n"
                for arg_name, arg_details in tool_info['args'].items():
                    required = "required" if arg_details['required'] else "optional"
                    desc = arg_details.get('description', 'No description')
                    tool_entry += f"    • {arg_name} ({required}): {desc}\n"
            
            tools_list.append(tool_entry)
        
        tools_list_str = "\n".join(tools_list)
        
        # Format analysis for prompt
        analysis_str = "\n".join([
            f"- Atoms: {analysis.get('atom_count', 'unknown')}",
            f"- Chains: {analysis.get('chain_ids', [])}",
            f"- Has Waters: {analysis.get('has_waters', False)}",
            f"- Heteroatoms: {analysis.get('heteroatoms', [])}",
            f"- Alternate Locations: {analysis.get('alternate_locations', False)}"
        ])
        
        # Use template or build basic prompt
        if config_prompt:
            return config_prompt.format(
                pdb_path=agent_input.pdb_path,
                user_goal=agent_input.user_goal,
                force_field=agent_input.force_field,
                water_model=agent_input.water_model,
                analysis=analysis_str,
                tools_list=tools_list_str
            )
        else:
            # Fallback prompt
            return f"""
You are a molecular dynamics preprocessing expert. Create a preprocessing plan for:

PDB: {agent_input.pdb_path}
Goal: {agent_input.user_goal}
Analysis: {analysis_str}

Available tools:
{tools_list_str}

**CRITICAL: You MUST ONLY use the tools listed above. Do NOT invent or suggest non-existent tools.**

{self._simulation_scope_note(agent_input)}

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
            "overview": "Preprocessing plan",
            "steps": []
        }

    def replan_with_guidance(self, human_recommendation: str, state: dict) -> Optional[dict]:
        """Update the structured preprocessing plan by applying human guidance.

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
            .get("preprocess")
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
                f"You are updating a GROMACS MD preprocessing execution plan.\n\n"
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
                .get("preprocessing_agent", "")
            )
            augmented = (
                nl_instructions
                + "\n\n**HUMAN RECOMMENDATION (must be followed):**\n" + human_recommendation
            ) if nl_instructions else human_recommendation
            agent_input = PreprocessingAgentInput(
                pdb_path=state.get("pdb_path", ""),
                working_directory=state.get("working_directory", "."),
                force_field=state.get("force_field", "amber99sb-ildn"),
                water_model=state.get("water_model", "tip3p"),
                user_goal=state.get("user_goal", ""),
                additional_instructions=augmented,
                component_selection=state.get("component_selection"),
            )
            prompt = self._build_planning_prompt(agent_input, state.get("pdb_analysis") or {})

        try:
            resp = self.llm.prompt(prompt, temperature=0.1)
            return self._extract_plan_json(resp)
        except Exception:
            return None

    def _create_fallback_plan(self, agent_input: PreprocessingAgentInput, 
                             analysis: Dict[str, Any]) -> PreprocessingPlan:
        """
        Create template-based fallback plan when LLM fails
        Uses workflows from config.yaml
        """
        steps = []
        workflows = self.config.get("workflows", {})
        
        # Use from_uniprot workflow when structure was acquired from database
        if agent_input.structure_request and not os.path.isfile(agent_input.pdb_path or ""):
            workflow_key = "from_uniprot"
        elif agent_input.structure_request:
            workflow_key = "full"
        else:
            workflow_key = "full"

        default_workflow = workflows.get(workflow_key, {}).get("steps", [
            "analyze_pdb", "separate_complex_components",
            "add_hydrogens", "validate_structure",
        ])
        
        # Build steps based on workflow and analysis
        for tool_name in default_workflow:
            step = self._create_step_from_tool(tool_name, agent_input, analysis)
            if step:
                steps.append(step)
        
        return PreprocessingPlan(
            reasoning="Using standard preprocessing workflow (LLM fallback)",
            overview="Standard PDB preprocessing for GROMACS MD simulation",
            steps=steps,
            potential_issues=[],
            recommendations=["Review generated topology file before simulation"]
        )
    
    def _auto_normalize_phosphorylation(
        self,
        stage,
        protein_path: str,
        execution_log: list,
        generated_files: dict,
        file_alias_map: dict,
        force_field: str = "amber99sb-ildn",
        user_text: str = "",
    ) -> str:
        """Map phospho residues for GROMACS (CHARMM dianionic SP2/THP2/TP2 by default)."""
        from src.preprocess.phospho_residues import (
            is_charmm_force_field,
            parse_phospho_ionic_state,
        )
        from src.preprocess.phosphorylation_normalizer import needs_gromacs_phospho_mapping

        ionic_state = parse_phospho_ionic_state(user_text, force_field)

        if not protein_path or not needs_gromacs_phospho_mapping(
            protein_path, force_field, ionic_state=ionic_state, user_text=user_text,
        ):
            return protein_path

        if is_charmm_force_field(force_field):
            execution_log.append(
                f"\n--- Auto: CHARMM phospho mapping ({ionic_state}) "
                f"for {force_field} ---"
            )
        else:
            execution_log.append(
                "\n--- Auto: normalize_phosphorylation_for_gromacs (AMBER SP2/THP/TP2) ---"
            )
        params = stage.apply_tool_params(
            "normalize_phosphorylation_for_gromacs",
            {"force_field": force_field, "user_text": user_text},
            protein_path,
        )
        result = self.tool_executor.execute_tool(
            "normalize_phosphorylation_for_gromacs", params,
        )
        if result.get("success"):
            execution_log.append(f"✓ {result.get('message', 'Phosphorylation mapped')}")
            mapped = result.get("output_file")
            final = result.get("protein_file", protein_path)
            if mapped:
                generated_files[mapped] = "Phosphorylation mapped for GROMACS (SP2/THP1/TP2)"
            if final:
                generated_files[final] = "Clean protein (GROMACS phospho names)"
            file_alias_map.update(stage.register_aliases())
            log_agent_action(
                "preprocessing",
                "Auto phosphorylation mapping for GROMACS",
                {
                    "status": "✅ SUCCESS",
                    "output_files": [mapped, final],
                    "resname_changes": result.get("resname_changes"),
                    "atom_changes": result.get("atom_changes"),
                },
            )
            return final or protein_path

        execution_log.append(
            f"✗ Phosphorylation mapping failed: {result.get('error', 'unknown')}"
        )
        log_agent_action(
            "preprocessing",
            "Auto phosphorylation mapping for GROMACS",
            {"status": "❌ FAILED", "error": result.get("error")},
        )
        return protein_path

    def _create_step_from_tool(self, tool_name: str, agent_input: PreprocessingAgentInput,
                               analysis: Dict[str, Any]) -> Optional[PreprocessingStep]:
        """Create preprocessing step from tool name and analysis"""
        tools_config = self.config.get("tools", {})
        tool_info = tools_config.get(tool_name, {})
        
        # Skip if conditions not met
        # Most tools are applied universally now
        
        # Build tool params
        tool_params = {"pdb_file": agent_input.pdb_path}
        
        if tool_name == "add_hydrogens":
            defaults = self.config.get("defaults", {})
            tool_params["method"] = defaults.get("hydrogen_method", "auto")
            tool_params["ph"] = defaults.get("ph", 7.4)
        
        return PreprocessingStep(
            name=tool_name.replace("_", " ").title(),
            description=tool_info.get("description", f"Execute {tool_name}"),
            tool_name=tool_name,
            tool_params=tool_params,
            reason=f"Required for {tool_info.get('description', 'preprocessing')}"
        )
    
    def _execute_plan(self, agent_input: PreprocessingAgentInput, plan: PreprocessingPlan,
                     state: MDState) -> PreprocessingResult:
        """
        Execute preprocessing plan step-by-step using tool executor
        Tracks file outputs and chains them between steps
        Includes retry limits to prevent infinite loops
        """
        execution_log = []
        issues = []
        warnings = []
        generated_files = {}
        ion_files = []  # Track ion files to skip hydrogen addition
        ligand_files = []  # Track ligand files for hydrogen addition
        ligand_resnames = []  # Track ligand residue names (e.g. ATP)
        ion_resnames = []    # Track ion residue names (e.g. MG)
        # Maps generic plan names (e.g. "ligand.pdb", "ions.pdb") and actual
        # basenames to full resolved paths, populated after separate_complex_components.
        file_alias_map: dict = {}
        
        current_pdb = agent_input.pdb_path
        topology_file = None
        processed_coords = None
        
        # Get execution limits from config
        agent_config = self.config.get("agent", {})
        max_retries = agent_config.get("max_tool_retries", 2)
        max_steps = agent_config.get("max_total_steps", 20)
        fail_fast = agent_config.get("fail_fast", False)
        
        # Track retry counts per tool
        tool_retry_counts = {}
        
        try:
            if not plan.steps:
                return PreprocessingResult(
                    success=False,
                    report="Preprocessing failed: planning produced no executable steps",
                    issues=["No executable preprocessing steps were generated"],
                    warnings=warnings,
                    generated_files=generated_files,
                    ligand_files=ligand_files,
                    ligand_resnames=sorted(set(ligand_resnames)),
                    ion_files=ion_files,
                    ion_resnames=sorted(set(ion_resnames)),
                    execution_log="No preprocessing steps available to execute"
                )

            # Enforce max steps limit
            if len(plan.steps) > max_steps:
                warnings.append(f"Plan has {len(plan.steps)} steps, limiting to {max_steps}")
                plan.steps = plan.steps[:max_steps]

            preprocess_dir = self.tool_executor.working_dir
            base_working_dir = str(Path(preprocess_dir).parent)
            from src.preprocess.stage_files import PreprocessStageManager

            goal_text = "\n".join(
                part for part in (
                    agent_input.user_goal,
                    agent_input.additional_instructions or "",
                )
                if part
            )
            stage = PreprocessStageManager(str(preprocess_dir))
            staged_path = stage.initialize(
                experimental_source=agent_input.pdb_path,
                goal_text=goal_text,
                search_dirs=[base_working_dir],
            )
            if staged_path and os.path.isfile(staged_path):
                current_pdb = staged_path
            file_alias_map.update(stage.register_aliases())
            staged_files: dict[str, str] = {}
            if stage._raw_staged:
                staged_files["raw.pdb"] = str(stage.files.raw.resolve())
            if stage._reference_staged:
                staged_files["reference_af3.pdb"] = str(stage.files.reference_af3.resolve())
            execution_log.append(
                f"Canonical preprocess staging: {stage.summary()} "
                f"(working PDB: {Path(current_pdb).name if current_pdb else 'none'})"
            )
            if staged_files:
                for label, path in staged_files.items():
                    execution_log.append(f"  staged {label}: {path}")
                log_agent_action(
                    "preprocessing",
                    "Staged input PDB files",
                    {
                        "status": "✅ SUCCESS",
                        "staged_files": staged_files,
                        "working_pdb": current_pdb,
                        "note": (
                            "reference_af3.pdb is optional — used only when remodel/align "
                            "tools need an AlphaFold or homology model donor"
                        ),
                    },
                )
            
            scope_ids = allowed_structure_ids(
                agent_input.working_directory,
                agent_input.pdb_path,
                uniprot_id=(
                    agent_input.structure_request.uniprot_id
                    if agent_input.structure_request
                    else None
                ),
            )
            scope_siblings = sibling_structure_ids(agent_input.working_directory)
            if scope_ids:
                execution_log.append(
                    "Structure scope: this directory may only fetch "
                    + ", ".join(sorted(scope_ids))
                )

            _METADATA_NOOP_TOOLS = {
                "",
                "noop",
                "none",
                "n/a",
                "na",
                "metadata",
                "note",
                "pass",
                "skip",
            }

            for i, step in enumerate(plan.steps):
                tool_key_name = (step.tool_name or "").strip().lower()
                if tool_key_name in _METADATA_NOOP_TOOLS:
                    execution_log.append(
                        f"\n--- Step {i+1}: {step.name} — skipped (metadata/no-op tool {step.tool_name!r}) ---"
                    )
                    log_agent_action(
                        "preprocessing",
                        f"Step {i+1}/{len(plan.steps)} skipped",
                        {
                            "step": step.name,
                            "tool": step.tool_name or "noop",
                            "reason": "Metadata placeholder — not an executable tool",
                        },
                    )
                    continue

                raw_params = dict(step.tool_params) if step.tool_params else {}
                skip_reason = structure_fetch_skip_reason(
                    step.tool_name,
                    raw_params,
                    allowed_ids=scope_ids,
                    sibling_ids=scope_siblings,
                    structure_already_staged=bool(
                        current_pdb and os.path.isfile(current_pdb)
                    ),
                )
                if skip_reason:
                    execution_log.append(f"\n--- Step {i+1}: {step.name} — skipped ---")
                    execution_log.append(f"⊘ {skip_reason}")
                    warnings.append(f"{step.name}: {skip_reason}")
                    log_agent_action(
                        "preprocessing",
                        f"Step {i+1}/{len(plan.steps)} skipped",
                        {
                            "step": step.name,
                            "tool": step.tool_name,
                            "reason": skip_reason,
                        },
                    )
                    continue

                # Log step start
                log_agent_action("preprocessing", f"Executing step {i+1}/{len(plan.steps)}", {
                    "step": step.name,
                    "tool": step.tool_name
                })
                
                execution_log.append(f"\n--- Step {i+1}: {step.name} ---")
                execution_log.append(f"Description: {step.description}")
                execution_log.append(f"Tool: {step.tool_name}")
                execution_log.append(f"Reason: {step.reason}")
                
                # Deterministic paths: LLM supplies tool + settings only
                tool_params = stage.apply_tool_params(
                    step.tool_name,
                    raw_params,
                    current_pdb,
                )

                # Log actual paths being used (after overrides)
                execution_log.append(f"Parameters: {json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in tool_params.items()}, indent=2)}")
                
                # Skip hydrogen addition for ion files and already-processed ligand files
                if step.tool_name == "add_hydrogens":
                    input_file = tool_params.get("pdb_file", current_pdb)
                    input_stem = Path(input_file).stem
                    
                    # Check if this is an ion file (by name or if it's in our tracked ion files)
                    if "_ion" in input_stem or input_file in ion_files:
                        execution_log.append(f"⊘ Skipping hydrogen addition for ion file: {Path(input_file).name}")
                        log_agent_action("preprocessing", f"Step {i+1}/{len(plan.steps)} skipped", {
                            "step": step.name,
                            "tool": step.tool_name,
                            "reason": "Hydrogen addition not needed for ions"
                        })
                        continue  # Skip to next step

                    # Check if hydrogens were already added (file ends with _h)
                    if input_stem.endswith("_h"):
                        execution_log.append(f"⊘ Skipping hydrogen addition - file already has hydrogens: {Path(input_file).name}")
                        log_agent_action("preprocessing", f"Step {i+1}/{len(plan.steps)} skipped", {
                            "step": step.name,
                            "tool": step.tool_name,
                            "reason": "Hydrogens already added to this file"
                        })
                        continue  # Skip to next step
                
                # Execute tool with retry logic
                retry_count = 0
                result = None
                tool_key = f"{step.tool_name}_{i}"
                
                while retry_count <= max_retries:
                    result = self.tool_executor.execute_tool(step.tool_name, tool_params)
                    
                    if result.get("success"):
                        break  # Success, exit retry loop
                    
                    retry_count += 1
                    if retry_count <= max_retries:
                        execution_log.append(f"⚠ Retry {retry_count}/{max_retries}: {result.get('error')}")
                        tool_retry_counts[tool_key] = retry_count
                    else:
                        execution_log.append(f"✗ Max retries ({max_retries}) exceeded")
                
                # Process results
                if result and result.get("success"):
                    execution_log.append(f"✓ Success: {result.get('message', 'Step completed')}")
                    
                    # Component selection filtering for separate_complex_components
                    comp_sel = agent_input.component_selection or {}
                    
                    # Collect output files for logging
                    output_files = []
                    if step.tool_name == "remodel_structure" and result.get("output_file"):
                        output_files.append(result["output_file"])
                        current_pdb = result["output_file"]
                        generated_files[result["output_file"]] = step.description
                    file_alias_map.update(stage.register_aliases())
                    if "output_file" in result and step.tool_name != "remodel_structure":
                        output_files.append(result["output_file"])
                        current_pdb = result["output_file"]
                        generated_files[result["output_file"]] = step.description
                    if "pdb_file" in result and result["pdb_file"] not in output_files:
                        output_files.append(result["pdb_file"])
                        current_pdb = result["pdb_file"]
                        generated_files[result["pdb_file"]] = step.description
                    if "protein_file" in result:
                        # Protein is always kept (required for MD)
                        output_files.append(result["protein_file"])
                        current_pdb = result["protein_file"]  # Use protein for chaining
                        generated_files[result["protein_file"]] = "Protein component"
                        actual_protein = result["protein_file"]
                        file_alias_map["protein.pdb"] = actual_protein
                        file_alias_map[Path(actual_protein).name] = actual_protein
                        input_stem = Path(agent_input.pdb_path or "").stem
                        if input_stem:
                            file_alias_map[f"{input_stem}_protein.pdb"] = actual_protein
                        file_alias_map.update(stage.register_aliases())
                        current_pdb = self._auto_normalize_phosphorylation(
                            stage,
                            current_pdb,
                            execution_log,
                            generated_files,
                            file_alias_map,
                            force_field=state.get("force_field", "amber99sb-ildn"),
                            user_text=state.get("user_goal", "") or "",
                        )
                    elif step.tool_name == "remodel_structure" and result.get("output_file"):
                        current_pdb = self._auto_normalize_phosphorylation(
                            stage,
                            result["output_file"],
                            execution_log,
                            generated_files,
                            file_alias_map,
                            force_field=state.get("force_field", "amber99sb-ildn"),
                            user_text=state.get("user_goal", "") or "",
                        )
                    if step.tool_name == "normalize_phosphorylation_for_gromacs" and result.get("protein_file"):
                        current_pdb = result["protein_file"]
                    if "ligand_file" in result:
                        from src.preprocess.phospho_residues import is_phospho_protein_resname

                        lig_stats = result.get("statistics", {}).get("ligand", {})
                        lig_resnames = lig_stats.get("resnames", [])
                        phospho_only = lig_resnames and all(
                            is_phospho_protein_resname(r) for r in lig_resnames
                        )
                        if phospho_only:
                            execution_log.append(
                                f"  ⊘ Ignored phospho residue file as ligand: "
                                f"{Path(result['ligand_file']).name}"
                            )
                        elif not comp_sel or comp_sel.get("ligand", True):
                            output_files.append(result["ligand_file"])
                            generated_files[result["ligand_file"]] = "Ligand component"
                            ligand_files.append(result["ligand_file"])
                            # Register generic aliases so subsequent plan steps that use
                            # names like "ligand.pdb" resolve to the actual file.
                            actual_ligand = result["ligand_file"]
                            file_alias_map["ligand.pdb"] = actual_ligand
                            file_alias_map[Path(actual_ligand).name] = actual_ligand
                            # Extract resnames from statistics
                            stats = result.get("statistics", {}).get("ligand", {})
                            if stats.get("resnames"):
                                ligand_resnames.extend(
                                    r for r in stats["resnames"]
                                    if not is_phospho_protein_resname(r)
                                )
                        else:
                            execution_log.append(f"  ⊘ Ligand file excluded by component selection: {Path(result['ligand_file']).name}")
                    if "ion_file" in result:
                        # Only keep ions if component_selection says so (or if no selection specified)
                        if not comp_sel or comp_sel.get("ions", True):
                            output_files.append(result["ion_file"])
                            generated_files[result["ion_file"]] = "Ion component"
                            ion_files.append(result["ion_file"])
                            # Register generic aliases so subsequent plan steps that use
                            # names like "ions.pdb" resolve to the actual file.
                            actual_ion = result["ion_file"]
                            file_alias_map["ions.pdb"] = actual_ion
                            file_alias_map["ion.pdb"] = actual_ion
                            file_alias_map[Path(actual_ion).name] = actual_ion
                            # Extract resnames from statistics
                            stats = result.get("statistics", {}).get("ions", {})
                            if stats.get("resnames"):
                                ion_resnames.extend(stats["resnames"])
                        else:
                            execution_log.append(f"  ⊘ Ion file excluded by component selection: {Path(result['ion_file']).name}")
                    if "topology_file" in result:
                        topology_file = result["topology_file"]
                        generated_files[topology_file] = "GROMACS topology file"
                    
                    # Log step completion status with output files
                    log_data = {
                        "step": step.name,
                        "tool": step.tool_name,
                        "status": "✅ SUCCESS"
                    }
                    if output_files:
                        log_data["output_files"] = output_files
                    log_agent_action("preprocessing", f"Step {i+1}/{len(plan.steps)} completed", log_data)
                    
                    if result.get("warning"):
                        warnings.append(f"{step.name}: {result['warning']}")
                else:
                    error_msg = result.get('error', 'Failed to execute') if result else 'Tool execution failed'
                    execution_log.append(f"✗ Failed after {retry_count} attempts: {error_msg}")
                    issues.append(f"{step.name}: {error_msg}")
                    
                    # Log step failure status
                    log_agent_action("preprocessing", f"Step {i+1}/{len(plan.steps)} failed", {
                        "step": step.name,
                        "tool": step.tool_name,
                        "status": "❌ FAILED",
                        "error": error_msg
                    })
                    
                    # Fail fast option
                    if fail_fast:
                        execution_log.append("⚠ Stopping execution (fail_fast enabled)")
                        break
            
            # Preprocessing complete - no topology verification needed
            # Topology generation is handled by simulation setup agent
            execution_log_str = "\n".join(execution_log)

            comp_sel = agent_input.component_selection or {}
            if len(issues) == 0:
                initial_pdb = agent_input.pdb_path
                same_as_input = (
                    initial_pdb
                    and current_pdb
                    and Path(current_pdb).resolve() == Path(initial_pdb).resolve()
                )
                if not current_pdb or (same_as_input and not any(
                    s.tool_name in ("add_hydrogens", "separate_complex_components")
                    for s in plan.steps
                )):
                    issues.append("Preprocessing did not produce a processed protein output in preprocess/")
                if comp_sel.get("ligand") and not ligand_files:
                    issues.append("Preprocessing did not produce ligand component output")
                if comp_sel.get("ions") and not ion_files:
                    issues.append("Preprocessing did not produce ion component output")
            
            return PreprocessingResult(
                success=len(issues) == 0,
                cleaned_pdb=current_pdb,
                topology=None,  # Topology is NOT created during preprocessing
                processed_coordinates=None,  # Coordinates are created by setup agent
                report=f"Preprocessing completed: {len(plan.steps)} steps executed",
                issues=issues,
                warnings=warnings,
                generated_files=generated_files,
                ligand_files=ligand_files,
                ligand_resnames=sorted(set(ligand_resnames)),
                ion_files=ion_files,
                ion_resnames=sorted(set(ion_resnames)),
                execution_log=execution_log_str
            )
            
        except Exception as e:
            return PreprocessingResult(
                success=False,
                report=f"Preprocessing failed: {str(e)}",
                issues=[str(e)],
                warnings=warnings,
                generated_files=generated_files,
                execution_log="\n".join(execution_log)
            )
    
    def _update_state(self, state: MDState, agent_output: PreprocessingAgentOutput):
        """Update workflow state with preprocessing results"""
        if agent_output.supervisor_update:
            state.update(agent_output.supervisor_update)
        
        state["preprocessing_report"] = agent_output.result.report
        state["preprocessing_issues"] = agent_output.result.issues
        state["preprocessing_warnings"] = agent_output.result.warnings
        state["preprocessing_execution_log"] = agent_output.result.execution_log
        
        # Propagate ligand/ion component tracking to workflow state
        if agent_output.result.ligand_files:
            state["ligand_files"] = agent_output.result.ligand_files
        if agent_output.result.ligand_resnames:
            state["ligand_resnames"] = agent_output.result.ligand_resnames
        if agent_output.result.ion_files:
            state["ion_files"] = agent_output.result.ion_files
        if agent_output.result.ion_resnames:
            state["ion_resnames"] = agent_output.result.ion_resnames
        
        # Build a set of known ligand/ion resnames for type detection
        known_ligand_rn = {r.lower() for r in agent_output.result.ligand_resnames}
        known_ion_rn = {r.lower() for r in agent_output.result.ion_resnames}
        
        # Register all generated files using SecureFileManager
        for file_path, description in agent_output.result.generated_files.items():
            # Determine file type: use resname-based detection first, then fallback
            file_type = "unknown"
            stem = Path(file_path).stem.lower()
            if "protein" in file_path.lower():
                file_type = "protein"
            elif stem in known_ligand_rn or description.lower().startswith("ligand"):
                file_type = "ligand"
            elif stem in known_ion_rn or description.lower().startswith("ion"):
                file_type = "ion"
            
            self.file_manager.register_external_file(
                file_path=file_path,
                file_type=file_type,
                description=description
            )
            
            # Log file operation
            log_file_operation("preprocessing", "create", file_path, True, description)
