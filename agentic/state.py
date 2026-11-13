"""MD Simulation State Definition for LangGraph"""
from typing import TypedDict, Optional, Dict, Any, List


# ── Agent I/O Directory Mapping ──────────────────────────────────────────────
# Each agent has a fixed input directory and output directory relative to
# working_dir/.  Agents decide only *file names*; the framework resolves
# full paths using this mapping.
#
#   input_dir = None  → agent reads user-supplied files (e.g. raw PDB)
#   input_dir = "hpc" → agent reads from working_dir/hpc/
#
AGENT_IO_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "preprocess": {"input_dir": None,          "output_dir": "preprocess"},
    "simsetup":   {"input_dir": "preprocess",  "output_dir": "simsetup"},
    "hpc":        {"input_dir": "simsetup",    "output_dir": "hpc"},
    "analysis":   {"input_dir": "hpc",         "output_dir": "analysis"},
    "reporter":   {"input_dir": "analysis",    "output_dir": "reporter"},
}


class MDState(TypedDict):
    """Central state for MD simulation workflow."""
    
    # User input
    user_goal: str
    user_goal_original: Optional[str]  # The raw --goal text, preserved unchanged throughout
    
    # Metadata & Configuration
    md_engine: str                 # "gromacs"
    force_field: str               # "amber99sb-ildn"
    water_model: str               # "tip3p"
    production_ns: Optional[float] # Requested production simulation length (ns)
    extended_minimization: Optional[bool]  # Two-stage minim for remodelled/strained structures
    human_in_loop: bool
    hitl_mode: Optional[str]  # None (off), "error", or "all"
    
    # Subtask-specific workflow
    subtask_type: Optional[str]    # "analysis_only", "setup_only", "preprocess_only", "hpc_only", "reporter_only", "multi_agent", None
    subtask_type_initialized: Optional[bool]  # Whether subtask type has been initialized
    agent_list: Optional[List[str]]           # Ordered list of agents for multi-agent workflow
    pipeline_agent_list: Optional[List[str]]  # Original CLI agents (survives prep-only filter)
    required_inputs: Optional[Dict[str, bool]]  # What inputs this task requires
    input_validated: Optional[bool]             # Whether unified input validation has completed
    analysis_directory: Optional[str]   # Path to analysis output directory
    trajectory_paths: Optional[Dict[str, Optional[str]]]  # Trajectory, topology, energy paths for analysis-only
    
    # System info (extracted during input validation)
    system_info: Optional[Dict[str, Any]]   # Molecular system metadata: components, atom counts, frames, etc.

    # PDB Analysis (from supervisor validation)
    pdb_analysis: Optional[Dict[str, Any]]  # Output from PDB analyzer
    component_selection: Optional[Dict[str, Any]]  # User-specified component selection
    structured_prompt: Optional[str]  # High-level structured prompt for planner
    
    # Structure acquisition (when user provides UniProt but no local PDB)
    structure_request: Optional[Dict[str, Any]]
    structure_requests: Optional[Dict[str, Dict[str, Any]]]  # keyed by pdb stem / uniprot id
    domain_context: Optional[str]  # Resolved domain/residue info for agent prompts
    sim_case: Optional[Dict[str, Any]]  # Per-sim component case metadata in multi-sim mode
    structure_acquisition_result: Optional[Dict[str, Any]]
    structure_acquisition_log: Optional[List[str]]

    # Resume / retry control (set by --resume / --retry-labels CLI flags)
    resume_failed_only: Optional[bool]   # True → skip already-succeeded sims on re-run
    requeue_failed_sims: Optional[bool]  # One-shot: reopen failed parallel-pool labels on --resume
    retry_labels: Optional[List[str]]    # Labels to force-retry even if previously succeeded
    _resume_succeeded_labels: Optional[List[str]]  # Internal: labels confirmed succeeded on disk
    multisim_resume_applied: Optional[bool]  # True after --resume initial bind (LangGraph-persisted)
    workflow_loop_streak: Optional[int]  # Detect supervisor/input_validation routing loops
    workflow_loop_key: Optional[str]  # Last routing key for loop detection
    combined_only: Optional[bool]        # True → skip per-sim loop; run base-level combined analysis + report only

    # Preprocessing stage
    raw_pdb: Optional[str]
    cleaned_pdb: Optional[str]
    preprocessing_report: Optional[str]
    preprocessing_issues: List[str]
    
    # Ligand / ion component tracking (set by preprocessing, consumed by setup)
    ligand_files: Optional[List[str]]          # Paths to separated ligand PDB files
    ligand_resnames: Optional[List[str]]       # Residue names (e.g. ["ATP", "GTP"])
    ion_files: Optional[List[str]]             # Paths to separated ion PDB files
    ion_resnames: Optional[List[str]]          # Residue names (e.g. ["MG", "ZN"])
    
    # File registry - tracks all files created during workflow
    file_registry: Dict[str, Dict[str, str]]  # {file_path: {"type": "protein", "description": "...", "stage": "preprocess"}}
    
    # Generated files tracking by agent - centralized tracking of important outputs
    generated_files: Dict[str, Dict[str, Dict[str, str]]]  # {agent: {file_key: {"path": "...", "description": "..."}}}
    
    # Setup stage
    topology: Optional[str]
    coordinates: Optional[str]
    chain_residue_map: Optional[str]  # PDB chain+resid → trajectory resindex JSON
    mdp_files: Dict[str, str]      # {"minim": "minim.mdp", "nvt": "nvt.mdp", ...}
    setup_report: Optional[str]
    setup_issues: List[str]
    
    # HPC stage
    hpc_action: Optional[str]
    hpc_output_directory: Optional[str]  # Path to HPC output directory (e.g., working_dir/hpc)
    job_script: Optional[str]
    job_id: Optional[str]
    job_status: Optional[str]
    trajectory_path: Optional[str]
    energy_file: Optional[str]  # Path to energy file (.edr)
    hpc_report: Optional[str]
    extension_ns: Optional[float]  # Requested additional duration for continuation
    target_total_ns: Optional[float]  # Preferred final total duration for continuation
    continuation_simulation_dir: Optional[str]  # Existing dir containing md.tpr/md.cpt
    continuation_source_tpr: Optional[str]
    continuation_checkpoint: Optional[str]
    continuation_current_ns: Optional[float]
    continuation_target_total_ns: Optional[float]
    continuation_tpr: Optional[str]
    continuation_manifest: Optional[str]
    continuation_job_id: Optional[str]
    continuation_status: Optional[str]
    
    # Analysis stage
    analysis_action: Optional[str]
    analysis_request: Optional[str]
    analysis_results: Dict[str, Any]
    figures: List[str]
    conclusions: Optional[str]

    # Reporter stage
    reporter_output: Optional[Dict[str, Any]]  # Output from reporter agent
    report_type: Optional[str]                 # "comprehensive", "executive", "custom"
    include_literature: Optional[bool]         # Whether to search PubMed
    literature_keywords: Optional[List[str]]   # Manual PubMed keywords
    reporter_plan: Optional[Dict[str, Any]]    # Reporter plan from supervisor
    reporter_instructions: Optional[str]       # Reporter-specific instructions from planner
    
    # Control flow & Execution tracking
    current_node: Optional[str]        # Track which node we're currently in
    next_node: Optional[str]
    human_feedback: Optional[str]
    working_directory: Optional[str]   # Root working directory (e.g., ./working_dir)
    
    # Enriched / rephrased prompt (set once after input validation)
    enriched_prompt: Optional[str]     # Unified enriched goal used by planner and agents
    master_enriched_prompt: Optional[str]  # Multi-sim: master supervisor enriched prompt (preserved across per-sim state resets)

    # Agent-specific output directories (hardcoded structure)
    preprocess_dir: Optional[str]      # working_dir/preprocess/
    simsetup_dir: Optional[str]        # working_dir/simsetup/
    hpc_dir: Optional[str]             # working_dir/hpc/ (or hpc/repXX when multi-rep)
    analysis_dir: Optional[str]        # working_dir/analysis/ (or analysis/repXX)
    # Multi-replicate production (prep/simsetup once; N seeded MD runs)
    rep_num: Optional[int]             # N replicates (default 1 = legacy flat layout)
    active_rep_id: Optional[str]       # e.g. rep01 when a pool worker is bound to one rep
    replicate_base_seed: Optional[int] # seed for rep01; rep_k uses base + k - 1
    
    execution_path: List[str]          # Track which nodes have been visited
    execution_plan: Optional[Dict[str, Any]]  # Detailed execution plan from planner
    plan_executed: bool                # Whether execution plan has been created
    rephrased_goal: Optional[str]      # LLM-rephrased user goal

    # Agent-specific instruction sections (extracted from full execution plan)
    preprocessing_instructions: Optional[str]
    setup_instructions: Optional[str]
    hpc_instructions: Optional[str]
    analysis_instructions: Optional[str]

    # Agent retry counters (used by _assign_field_agent_tasks)
    current_agent_idx: Optional[int]
    completed_pipeline_stages: Optional[List[str]]  # Forward-only ticks (preprocess…reporter)
    preprocess_retry_count: Optional[int]
    setup_retry_count: Optional[int]
    hpc_retry_count: Optional[int]
    analysis_retry_count: Optional[int]
    reporter_retry_count: Optional[int]

    # Intermediate validation artifacts
    pdb_summary: Optional[str]              # Human-readable PDB summary (for enrichment)
    file_info: Optional[Dict[str, Any]]     # Trajectory/topology file info (analysis tasks)
    reporter_file_info: Optional[Dict[str, Any]]  # Analysis output file info (reporter tasks)

    # Final report
    final_report: Optional[str]        # LLM-generated or fallback workflow completion report
    workflow_status: Optional[str]     # "completed", "failed"
    human_final_decision: Optional[str]  # "done" | "rerun_reporter" | "rerun_analysis"
    
    # Error & Warning tracking
    errors: List[str]                  # List of errors encountered during workflow
    warnings: List[str]                # List of warnings generated during workflow

    # ── Multi-Simulation Mode ─────────────────────────────────────────────
    is_multi_simulation: Optional[bool]          # True when running multiple PDBs
    multi_sim_phase: Optional[str]               # "planning" | "pre_combined" | "hpc_pool" | "executing_sims" | "combined_analysis"/"post_combined" | "combined_reporter" | None
    pdb_list: Optional[List[str]]                # Original PDB file paths from CLI / goal extraction
    sim_prompts: Optional[List[Dict[str, Any]]]  # Per-sim prompts from master planner [{prompt, pdb, label, working_dir}, ...]
    run_combined_analysis: Optional[bool]        # Legacy alias for run_post_combined (post per-sim loop)
    combined_analysis_plan: Optional[str]        # Legacy alias for post_combined_plan
    run_pre_combined: Optional[bool]             # Optional combined tools BEFORE per-sim traj analysis (n_sims>1)
    pre_combined_attempts: Optional[int]         # Failed shared-setup tries; persisted across supervisor hops
    pre_combined_plan: Optional[str]             # NL plan for pre_combined (pocket/MSA/consensus → base/cross_sim/)
    run_post_combined: Optional[bool]            # Optional combined tools AFTER all per-sim work (n_sims>1)
    post_combined_plan: Optional[str]            # NL plan for post_combined (compare/Ward/overlays)
    campaign_spec: Optional[Dict[str, Any]]      # Compiled campaign program (recipe + contracts)
    hitl_checkpoint_subtype: Optional[str]       # compile | pre_combined | post_combined
    hitl_pause_after_compile: Optional[bool]     # Review compiled protocol before mapping
    allow_partial_combined: Optional[bool]       # If True, Ward may run on a sparse matrix
    current_sim_index: Optional[int]             # Index into sim_prompts (which sim is next)
    completed_sim_states: Optional[List[Dict[str, Any]]]  # Saved state snapshots after each sim completes
    sim_working_dirs: Optional[List[str]]        # Per-sim working directories (e.g., base_dir/1abc/)
    multi_sim_base_dir: Optional[str]            # Base directory for multi-simulation checkpoint mirroring

    # Cross-sim HPC pool (SLURM job queue across simulations)
    hpc_pool: Optional[Dict[str, Any]]
    allowed_hpc_jobs: Optional[int]
    hpc_check_interval_sec: Optional[int]
    hpc_check_interval: Optional[str]
    post_hpc_analysis_only: Optional[bool]
    hpc_pool_phase_complete: Optional[bool]

    # Cross-sim parallel worker pool (local prep / analysis+reporter)
    parallel_pool: Optional[Dict[str, Any]]
    parallel_workers: Optional[Any]              # "auto" or int (1 = sequential)
    parallel_mem_gb_per_job: Optional[float]
    parallel_cpus_per_job: Optional[float]
    parallel_workers_resolved: Optional[int]
    llm_concurrency: Optional[Any]                 # "auto" or int (Ollama parallel slots)
    _allowed_hpc_jobs_explicit: Optional[bool]

    # Human-in-the-loop session (persisted in state.jsonl across checkpoints)
    hitl_active_agent: Optional[str]               # Field agent selected in HITL chat
    hitl_target_sim_label: Optional[str]           # Multi-sim: bound simulation label (e.g. p23458)
    hitl_sim_dirs: Optional[Dict[str, str]]        # Multi-sim: label -> working dir map
    hitl_agent_working_directory: Optional[str]    # HITL view dir (sim root in multi-sim)
    hitl_agent_output_directory: Optional[str]     # HITL tool output dir (e.g. .../analysis)
    hitl_return_checkpoint: Optional[str]          # Return here after delegated agent run
    hitl_delegate_agent: Optional[str]
    hitl_delegate_task: Optional[str]
