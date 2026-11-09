"""
HPC Job Submitter - Submit SLURM jobs to HPC systems
"""
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from langchain.tools import tool

logger = logging.getLogger(__name__)


def reuse_hpc_blocks_sbatch() -> bool:
    """True when ``--reuse-hpc`` / ``AGENTIC_REUSE_HPC`` forbids calling sbatch."""
    return os.environ.get("AGENTIC_REUSE_HPC", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def find_live_slurm_job_for_dir(workdir: str | Path) -> Optional[str]:
    """Return a queued or running SLURM job whose work directory is *workdir*.

    Completed and failed jobs are not listed by ``squeue``, so a later retry
    of a finished directory is still allowed. Returns None when SLURM cannot
    be queried.
    """
    import shutil

    target = Path(workdir).expanduser()
    try:
        target = target.resolve()
    except OSError:
        target = target.absolute()
    user = os.environ.get("USER", "").strip()
    if not user or shutil.which("squeue") is None:
        return None
    try:
        result = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%i|%T|%Z"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("squeue workdir lookup failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        job_id, _state, raw_dir = (line.split("|", 2) + ["", ""])[:3]
        job_id = job_id.strip()
        raw_dir = raw_dir.strip()
        if not job_id or not raw_dir:
            continue
        try:
            queued = Path(raw_dir).expanduser().resolve()
        except OSError:
            queued = Path(raw_dir)
        if queued == target:
            return job_id
    return None


@tool
def submit_job(
    script_path: str,
    remote_host: Optional[str] = None,
    remote_user: Optional[str] = None,
    remote_dir: Optional[str] = None,
    ssh_key_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Submit SLURM job to HPC system.
    
    Args:
        script_path: Path to SLURM script
        remote_host: Remote HPC hostname (optional, for SSH submission)
        remote_user: Remote username (optional)
        remote_dir: Remote working directory (optional)
        ssh_key_path: SSH key path (optional)
        
    Returns:
        Dict with job_id and submission status
        
    Note:
        If remote_host is not provided, assumes local submission via sbatch.
        For remote submission, uses SSH to transfer script and submit.
        When AGENTIC_REUSE_HPC is set (--reuse-hpc), sbatch is never invoked.
    """
    if reuse_hpc_blocks_sbatch():
        logger.error(
            "submit_job blocked: --reuse-hpc / AGENTIC_REUSE_HPC is active (no sbatch)"
        )
        return {
            "success": False,
            "error": "sbatch blocked: --reuse-hpc is active (reuse existing md.tpr/mdWrap.xtc)",
            "status": "SKIPPED_REUSE_HPC",
        }

    try:
        script_file = Path(script_path)
        if not script_file.exists():
            return {
                "success": False,
                "error": f"Script file not found: {script_path}"
            }

        # Local submission
        if not remote_host:
            live = find_live_slurm_job_for_dir(script_file.parent)
            if live:
                logger.warning(
                    "submit_job: %s already has SLURM job %s — not submitting again",
                    script_file.parent,
                    live,
                )
                try:
                    from agentic.hpc.job_markers import write_job_marker

                    write_job_marker(script_file.parent, live)
                except Exception:
                    logger.debug("Could not write .agentic_job_id marker", exc_info=True)
                return {
                    "success": True,
                    "job_id": live,
                    "status": "ALREADY_QUEUED",
                    "message": f"Job already queued for this directory: {live}",
                    "job_script": str(script_file.resolve()),
                }
            result = subprocess.run(
                ["sbatch", str(script_file)],
                capture_output=True,
                text=True,
                cwd=script_file.parent
            )
            
            if result.returncode == 0:
                # Parse job ID from "Submitted batch job 12345"
                import re
                match = re.search(r"Submitted batch job (\d+)", result.stdout)
                job_id = match.group(1) if match else "UNKNOWN"
                try:
                    from agentic.hpc.job_markers import write_job_marker

                    write_job_marker(script_file.parent, job_id)
                except Exception:
                    logger.debug("Could not write .agentic_job_id marker", exc_info=True)

                return {
                    "success": True,
                    "job_id": job_id,
                    "status": "SUBMITTED",
                    "message": f"Job submitted successfully: {job_id}",
                    "stdout": result.stdout,
                    "job_script": str(script_file.resolve()),
                }
            else:
                return {
                    "success": False,
                    "error": f"sbatch failed: {result.stderr}",
                    "returncode": result.returncode
                }
        
        # Remote submission via SSH
        else:
            try:
                import paramiko
            except ImportError:
                return {
                    "success": False,
                    "error": "paramiko not installed - cannot submit to remote host"
                }
            
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect
            connect_kwargs = {
                "hostname": remote_host,
                "username": remote_user,
                "port": 22
            }
            if ssh_key_path:
                connect_kwargs["key_filename"] = ssh_key_path
            
            client.connect(**connect_kwargs)
            
            # Upload script
            sftp = client.open_sftp()
            remote_script = f"{remote_dir}/{script_file.name}"
            sftp.put(str(script_file), remote_script)
            sftp.close()
            
            # Submit job
            stdin, stdout, stderr = client.exec_command(f"sbatch {remote_script}")
            output = stdout.read().decode("utf-8").strip()
            error = stderr.read().decode("utf-8").strip()
            
            client.close()
            
            # Parse job ID
            import re
            match = re.search(r"Submitted batch job (\d+)", output)
            job_id = match.group(1) if match else "UNKNOWN"
            
            return {
                "success": True,
                "job_id": job_id,
                "status": "SUBMITTED",
                "message": f"Remote job submitted: {job_id}",
                "stdout": output,
                "remote_host": remote_host
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Job submission failed: {e}"
        }
