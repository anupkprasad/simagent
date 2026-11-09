"""
Submit-only HPC path for multi-sim pool (no blocking monitor loop).

Copies simsetup → hpc, creates SLURM script, sbatch — one job per simulation.
Uses deterministic SLURM defaults from config; walltime stays at the default
unless the user explicitly requests a limit in the goal.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agentic.hpc.submit_params import (
    build_slurm_script_params,
    derive_job_name,
    resolve_production_ns,
)

logger = logging.getLogger(__name__)

# Never overwrite these under --reuse-hpc (frozen production trajectories).
_PROTECTED_HPC_NAMES = frozenset({"md.tpr", "mdWrap.xtc", "md.xtc", "md.trr"})


def prepare_hpc_without_submit(
    sim_working_dir: str,
    sim_label: str,
    production_ns: Optional[float] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
    *,
    hpc_dir: Optional[str] = None,
    rep_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Stage HPC directory for a reuse-hpc demo: copy simsetup → hpc (without
    clobbering existing md.tpr/mdWrap.xtc) and create a SLURM script, but do
    **not** call sbatch.

    Optional *hpc_dir* / *rep_id* / *seed* support nested ``hpc/repXX`` replicates.
    """
    from src.hpc.script_creator import create_slurm_script
    from src.hpc.file_copy import copy_simulation_files

    wd = Path(sim_working_dir)
    simsetup_dir = wd / "simsetup"
    hpc_dir = Path(hpc_dir) if hpc_dir else (wd / "hpc")
    hpc_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot protected production files (hardlinks/inodes) before copy.
    protected: Dict[str, Path] = {}
    for name in _PROTECTED_HPC_NAMES:
        p = hpc_dir / name
        if p.is_file():
            protected[name] = p.resolve()

    ns = production_ns
    if ns is None:
        ns = resolve_production_ns(workflow_state, sim_label, sim_working_dir)

    from agentic.utils.conversation_logger import (
        log_agent_action,
        log_file_operation,
        temporary_log_file,
    )

    log_path = str(wd / "agent_conversation.log")
    with temporary_log_file(log_path):
        log_agent_action(
            "hpc",
            f"HPC pool: reuse-hpc prepare (no sbatch) for {sim_label}",
            {
                "mode": "hpc_pool_reuse",
                "sim_label": sim_label,
                "simsetup_dir": str(simsetup_dir),
                "hpc_dir": str(hpc_dir),
                "production_ns": ns,
                "rep_id": rep_id,
                "seed": seed,
            },
        )

    copy_result: Dict[str, Any] = {"success": True, "files_copied": 0}
    if simsetup_dir.is_dir():
        copy_result = copy_simulation_files.func(
            source_dir=str(simsetup_dir),
            dest_dir=str(hpc_dir),
        )
        if seed is not None:
            from src.analysis.replicate_paths import (
                apply_gen_seed_to_mdp_dir,
                parse_rep_id,
                write_replicate_meta,
            )

            apply_gen_seed_to_mdp_dir(hpc_dir, int(seed))
            if rep_id:
                write_replicate_meta(
                    hpc_dir,
                    rep_id=rep_id,
                    seed=int(seed),
                    parent_label=sim_label,
                    rep_index=int(parse_rep_id(rep_id) or 1),
                    rep_num=int((workflow_state or {}).get("rep_num") or 1),
                )
        # Restore protected trajectories if copy somehow replaced them.
        import os

        for name, src in protected.items():
            dest = hpc_dir / name
            try:
                same = dest.is_file() and dest.resolve() == src
                if same:
                    continue
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                os.link(str(src), str(dest))
            except OSError:
                logger.warning(
                    "reuse-hpc: could not restore protected %s in %s", name, hpc_dir
                )
        with temporary_log_file(log_path):
            log_file_operation(
                "hpc",
                "copy",
                str(hpc_dir),
                bool(copy_result.get("success")),
                f"simsetup → hpc reuse-safe ({copy_result.get('files_copied', 'ok')})",
            )
    else:
        logger.info(
            "reuse-hpc: no simsetup yet for %s — keeping existing hpc/ trajectories",
            sim_label,
        )

    job_name = derive_job_name(
        f"{sim_label}_{rep_id}" if rep_id else sim_label,
        sim_working_dir,
    )
    slurm_params = build_slurm_script_params(
        job_name=job_name,
        hpc_dir=str(hpc_dir),
        simsetup_dir=str(simsetup_dir),
        workflow_state=workflow_state,
    )
    script_result = create_slurm_script.func(**slurm_params)
    script_path = None
    if script_result.get("success"):
        script_path = script_result.get("script_path") or script_result.get("output_path")
    if not script_path:
        candidates = sorted(hpc_dir.glob("*_run.sh")) + sorted(hpc_dir.glob("*.sh"))
        script_path = str(candidates[0]) if candidates else None

    msg = "reuse-hpc: staged HPC files; sbatch skipped"
    with temporary_log_file(log_path):
        log_agent_action(
            "hpc",
            msg,
            {
                "job_script": script_path,
                "job_name": job_name,
                "protected_files": sorted(protected),
            },
        )
    return {
        "success": True,
        "job_id": None,
        "job_script": script_path,
        "job_name": job_name,
        "production_ns": ns,
        "status": "SKIPPED_REUSE_HPC",
        "message": msg,
        "copy_ok": bool(copy_result.get("success")),
    }


def submit_simulation_job(
    sim_working_dir: str,
    sim_label: str,
    production_ns: Optional[float] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
    *,
    hpc_dir: Optional[str] = None,
    rep_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare and submit one SLURM job for *sim_working_dir*.

    Optional *hpc_dir* / *rep_id* / *seed* for nested ``hpc/repXX`` replicates.

    Returns dict with success, job_id, job_script, error.
    """
    from src.hpc.job_submitter import submit_job
    from src.hpc.script_creator import create_slurm_script
    from src.hpc.file_copy import copy_simulation_files

    wd = Path(sim_working_dir)
    simsetup_dir = wd / "simsetup"
    hpc_path = Path(hpc_dir) if hpc_dir else (wd / "hpc")
    hpc_path.mkdir(parents=True, exist_ok=True)
    hpc_dir = hpc_path  # Path below

    ns = production_ns
    if ns is None:
        ns = resolve_production_ns(workflow_state, sim_label, sim_working_dir)

    from agentic.utils.conversation_logger import (
        log_agent_action,
        log_agent_completion,
        log_file_operation,
        temporary_log_file,
    )

    display_label = f"{sim_label}/{rep_id}" if rep_id else sim_label
    log_path = str(wd / "agent_conversation.log")
    with temporary_log_file(log_path):
        log_agent_action(
            "hpc",
            f"HPC pool: prepare and submit SLURM job for {display_label}",
            {
                "mode": "hpc_pool",
                "sim_label": sim_label,
                "simsetup_dir": str(simsetup_dir),
                "hpc_dir": str(hpc_dir),
                "production_ns": ns,
                "rep_id": rep_id,
                "seed": seed,
            },
        )

    copy_result = copy_simulation_files.func(
        source_dir=str(simsetup_dir),
        dest_dir=str(hpc_dir),
    )
    if not copy_result.get("success"):
        with temporary_log_file(log_path):
            log_file_operation("hpc", "copy", str(hpc_dir), False, copy_result.get("error", ""))
        return {
            "success": False,
            "error": copy_result.get("error", "copy_simulation_files failed"),
        }
    # SLURM scripts hardcode system.gro — refuse submit if ions/prep never finished.
    if not (hpc_dir / "system.gro").is_file() and not (simsetup_dir / "system.gro").is_file():
        err = (
            f"Missing system.gro under {simsetup_dir} (and hpc/). "
            "Prep must finish ion addition before HPC submit."
        )
        with temporary_log_file(log_path):
            log_agent_action("hpc", "HPC submit blocked — incomplete prep", {"error": err})
        return {"success": False, "error": err}
    if seed is not None:
        from src.analysis.replicate_paths import (
            apply_gen_seed_to_mdp_dir,
            parse_rep_id,
            write_replicate_meta,
        )

        apply_gen_seed_to_mdp_dir(hpc_dir, int(seed))
        if rep_id:
            write_replicate_meta(
                hpc_dir,
                rep_id=rep_id,
                seed=int(seed),
                parent_label=sim_label,
                rep_index=int(parse_rep_id(rep_id) or 1),
                rep_num=int((workflow_state or {}).get("rep_num") or 1),
            )
    with temporary_log_file(log_path):
        log_file_operation(
            "hpc",
            "copy",
            str(hpc_dir),
            True,
            f"simsetup → hpc ({copy_result.get('files_copied', 'ok')})",
        )

    job_name = derive_job_name(
        f"{sim_label}_{rep_id}" if rep_id else sim_label,
        sim_working_dir,
    )
    slurm_params = build_slurm_script_params(
        job_name=job_name,
        hpc_dir=str(hpc_dir),
        simsetup_dir=str(simsetup_dir),
        workflow_state=workflow_state,
    )
    with temporary_log_file(log_path):
        log_agent_action(
            "hpc",
            "SLURM script parameters (pool mode)",
            {
                "job_name": job_name,
                "time_limit": slurm_params.get("time_limit"),
                "memory": slurm_params.get("memory"),
                "partition": slurm_params.get("partition"),
                "simulation_phases": slurm_params.get("simulation_phases"),
            },
        )

    script_result = create_slurm_script.func(**slurm_params)
    if not script_result.get("success"):
        with temporary_log_file(log_path):
            log_agent_action(
                "hpc",
                "SLURM script creation failed",
                {"error": script_result.get("error", "create_slurm_script failed")},
            )
        return {
            "success": False,
            "error": script_result.get("error", "create_slurm_script failed"),
        }

    script_path = script_result.get("script_path") or script_result.get("output_path")
    if not script_path:
        candidates = sorted(hpc_dir.glob("*_run.sh")) + sorted(hpc_dir.glob("*.sh"))
        script_path = str(candidates[0]) if candidates else None
    if not script_path:
        return {"success": False, "error": "No SLURM script produced"}

    submit_result = submit_job.func(script_path=script_path)
    if not submit_result.get("success"):
        with temporary_log_file(log_path):
            log_agent_action(
                "hpc",
                "SLURM submission failed",
                {"error": submit_result.get("error", "submit_job failed"), "script": script_path},
            )
        return {
            "success": False,
            "error": submit_result.get("error", "submit_job failed"),
            "job_script": script_path,
        }

    job_id = submit_result.get("job_id")
    try:
        from agentic.hpc.job_markers import write_job_marker

        write_job_marker(hpc_dir, job_id)
    except Exception:
        logger.debug("Could not write .agentic_job_id marker", exc_info=True)

    result = {
        "success": True,
        "job_id": job_id,
        "status": submit_result.get("status"),
        "job_script": script_path,
        "job_name": job_name,
        "production_ns": ns,
        "time_limit": slurm_params.get("time_limit"),
        "message": submit_result.get("message"),
    }
    with temporary_log_file(log_path):
        log_agent_action(
            "hpc",
            "Submitted SLURM job",
            {
                "job_id": result["job_id"],
                "job_script": script_path,
                "job_name": job_name,
                "mode": "hpc_pool",
                "production_ns": ns,
                "time_limit": slurm_params.get("time_limit"),
            },
        )
        log_agent_completion(
            "hpc",
            "HPC pool job submission",
            {
                "job_id": result["job_id"],
                "job_script": script_path,
                "job_name": job_name,
                "hpc_dir": str(hpc_dir),
                "production_ns": ns,
                "time_limit": slurm_params.get("time_limit"),
            },
            True,
        )
    return result
