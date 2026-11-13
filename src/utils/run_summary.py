"""
Build and write base-level workflow run summaries (JSON + Markdown).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sim_status(sim: Dict[str, Any]) -> str:
    if sim.get("skipped"):
        return "skipped"
    if sim.get("success"):
        return "success"
    return "failed"


def _sim_succeeded(sim: Dict[str, Any]) -> bool:
    if sim.get("skipped"):
        return False
    if "success" in sim:
        return bool(sim["success"])
    return bool(
        sim.get("job_id")
        or sim.get("trajectory_path")
        or sim.get("topology")
        or sim.get("coordinates")
        or sim.get("analysis_results")
        or sim.get("reporter_output")
    )


def _analysis_in_pipeline(state: Dict[str, Any]) -> bool:
    """True when this run was supposed to produce per-simulation analysis."""
    names: List[str] = []
    for key in ("agent_list", "pipeline_agent_list"):
        value = state.get(key) or []
        if isinstance(value, str):
            names.append(value)
        else:
            names.extend(str(item) for item in value)
    blob = " ".join(names).lower()
    if "analysis" in blob:
        return True
    sub = str(
        state.get("subtask_type") or state.get("pipeline_subtask_type") or ""
    ).lower()
    if "analysis" in sub:
        return True
    phase = str(state.get("multi_sim_phase") or "")
    pool_phase = str((state.get("parallel_pool") or {}).get("phase") or "")
    return phase.startswith("combined") or pool_phase == "analysis"


def _expected_simulation_count(final_state: Dict[str, Any], pdb_list: List[str]) -> int:
    sim_prompts = final_state.get("sim_prompts") or []
    source_count = len(final_state.get("pdb_list") or pdb_list)
    if sim_prompts and source_count:
        return max(len(sim_prompts), source_count)
    if sim_prompts:
        return len(sim_prompts)
    sim_dirs = final_state.get("sim_working_dirs") or []
    if sim_dirs:
        return len(sim_dirs)
    return len(pdb_list)


def _missing_planned_simulations(
    final_state: Dict[str, Any], pdb_list: List[str]
) -> List[str]:
    """PDB stems requested but absent from ``sim_prompts``."""
    requested = {
        Path(p).stem.lower()
        for p in (final_state.get("pdb_list") or pdb_list or [])
        if p
    }
    planned = {
        str(sp.get("label", "")).lower()
        for sp in (final_state.get("sim_prompts") or [])
        if sp.get("label")
    }
    return sorted(requested - planned)


def _hpc_status_to_summary(hpc_status: Optional[str]) -> str:
    if hpc_status in ("completed", "skipped"):
        return "success"
    if hpc_status == "failed":
        return "failed"
    if hpc_status in ("submitted", "running"):
        return "submitted"
    if hpc_status == "pending":
        return "pending"
    return "unknown"


def _simulations_from_hpc_pool(final_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    pool = final_state.get("hpc_pool") or {}
    pool_sims = pool.get("sims") or {}
    rows: List[Dict[str, Any]] = []
    for sp in final_state.get("sim_prompts") or []:
        label = sp.get("label")
        rec = pool_sims.get(label) or {}
        prep = rec.get("prep_status")
        hpc = rec.get("hpc_status")
        wd = sp.get("working_dir") or rec.get("working_dir") or ""
        health = "pending"
        try:
            from agentic.utils.sim_health import should_skip_downstream_for_sim

            decision = should_skip_downstream_for_sim(
                final_state, label or "", working_dir=wd
            )
            if decision.get("skip") or hpc == "failed" or prep == "failed":
                health = "failed"
            elif decision.get("health_label") == "healthy":
                health = "healthy"
        except Exception:
            if hpc == "failed" or prep == "failed":
                health = "failed"
        if health == "failed":
            status = "failed"
        elif prep in ("done", "skipped") and hpc in ("completed", "skipped"):
            status = "success"
            health = "healthy"
        elif hpc in ("submitted", "running"):
            status = "submitted"
        elif prep == "pending":
            status = "not_started"
        else:
            status = _hpc_status_to_summary(hpc)
        rows.append(
            {
                "label": label,
                "status": status,
                "health": health,
                "success": status == "success",
                "skipped": prep == "skipped" or hpc == "skipped",
                "skip_reason": rec.get("skip_reason"),
                "working_directory": wd,
                "job_id": rec.get("job_id"),
                "job_status": hpc,
                "prep_status": prep,
                "errors": [rec["error"]] if rec.get("error") else [],
            }
        )
    return rows


def _simulations_merged_from_state(final_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge session snapshots with disk-done sims from ``multi_sim_progress``."""
    from agentic.multi_sim_progress import (
        load_local_continuation_job,
        merge_completed_states_from_progress,
    )

    merged_state = dict(final_state)
    merge_completed_states_from_progress(merged_state)
    completed = list(merged_state.get("completed_sim_states") or [])
    progress = final_state.get("multi_sim_progress") or {}
    progress_sims = progress.get("sims") or {}
    sim_order = progress.get("sim_order") or []
    order_index = {label: i for i, label in enumerate(sim_order)}
    terminal = {
        "COMPLETED", "COMPLETE", "CANCELLED", "CANCELED", "FAILED",
        "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
    }

    def _row_from_disk(label: str, rec: Dict[str, Any], sim: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        wd = (sim or {}).get("working_directory") or rec.get("working_dir") or ""
        job_info = load_local_continuation_job(wd) if wd else None
        job_id = (job_info or {}).get("job_id") or (sim or {}).get("job_id") or (sim or {}).get("continuation_job_id")
        job_status = str(
            (job_info or {}).get("status")
            or (sim or {}).get("job_status")
            or (sim or {}).get("continuation_status")
            or ""
        ).upper()
        agents = rec.get("agents") or {}
        health = "pending"
        fatal_reason = ""
        try:
            from agentic.utils.sim_health import (
                production_health,
                is_reuse_hpc,
                should_skip_downstream_for_sim,
            )

            decision = should_skip_downstream_for_sim(
                final_state, label, working_dir=wd
            )
            if decision.get("skip"):
                health = "failed"
                fatal_reason = str(decision.get("reason") or "")
            else:
                ph = decision.get("health") or production_health(
                    wd, reuse_hpc=is_reuse_hpc(final_state)
                )
                health = "healthy" if ph.get("healthy") else "pending"
        except Exception:
            pass

        if rec.get("status") == "failed" or agents.get("hpc") == "failed":
            health = "failed"
            fatal_reason = fatal_reason or rec.get("error") or "simulation failed"

        analysis_requested = _analysis_in_pipeline(final_state)
        analysis_ok = True
        if analysis_requested and wd:
            try:
                from agentic.multi_sim_progress import per_sim_analysis_done_on_disk

                analysis_ok = bool(
                    per_sim_analysis_done_on_disk(wd, state=final_state)
                )
            except Exception:
                analysis_ok = agents.get("analysis") == "done"
        disk_done = (
            rec.get("status") == "done"
            or (
                agents.get("analysis") == "done"
                and agents.get("reporter") == "done"
            )
            or bool(job_info and job_info.get("complete"))
        )
        if analysis_requested:
            disk_done = bool(disk_done and analysis_ok)
        # A finished trajectory is not a successful study when the requested
        # analysis files were never written. Analysis artifacts alone do not
        # make a dead MD "success".
        success = False
        if health == "failed":
            status = "failed"
            success = False
        elif analysis_requested and not analysis_ok:
            status = "failed"
            success = False
            if health == "healthy":
                health = "incomplete"
            fatal_reason = (
                fatal_reason
                or rec.get("error")
                or "analysis incomplete"
            )
        elif job_info and job_info.get("complete"):
            status = "success"
            success = True
            health = "healthy"
        elif job_id and job_status not in terminal:
            status = "submitted"
            success = True
        elif disk_done and health != "failed":
            status = "success"
            success = True
            health = "healthy"
        elif sim:
            status = _sim_status(sim)
            if status == "failed" and job_id and health != "failed":
                status = "submitted"
                success = True
            else:
                success = status == "success"
        elif agents.get("analysis") == "done" and agents.get("reporter") != "done":
            status = "in_progress"
            success = False
        elif rec.get("status") == "failed":
            status = "failed"
            success = False
        else:
            status = "pending"
            success = False
        errors = []
        if not success:
            errors = list((sim or {}).get("errors") or [])
            if fatal_reason and fatal_reason not in errors:
                errors = [fatal_reason] + errors
        return {
            "label": label,
            "status": status,
            "health": health,
            "success": success,
            "skipped": bool((sim or {}).get("skipped")),
            "skip_reason": (sim or {}).get("skip_reason") or rec.get("skip_reason"),
            "working_directory": wd,
            "job_id": job_id,
            "job_status": job_status or (sim or {}).get("job_status"),
            "topology": (sim or {}).get("topology"),
            "trajectory_path": (sim or {}).get("trajectory_path"),
            "errors": errors[:10],
            "warnings": ((sim or {}).get("warnings") or [])[:5],
        }

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for sim in sorted(
        completed,
        key=lambda s: order_index.get(str(s.get("label")), int(s.get("sim_index") or 999)),
    ):
        label = sim.get("label")
        if not label or label in seen:
            continue
        seen.add(label)
        rec = progress_sims.get(label) or {}
        rows.append(_row_from_disk(label, rec, sim))

    for label in sim_order:
        if label in seen:
            continue
        rec = progress_sims.get(label) or {}
        rows.append(_row_from_disk(label, rec, None))
        seen.add(label)
    return rows


def build_run_summary(
    final_state: Dict[str, Any],
    *,
    working_dir: str,
    goal: str,
    pdb_list: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a structured summary of a workflow run."""
    pdb_list = pdb_list or []
    config = config or {}
    completed = list(final_state.get("completed_sim_states") or [])
    is_multi = bool(final_state.get("is_multi_simulation"))
    missing = _missing_planned_simulations(final_state, pdb_list) if is_multi else []
    total = _expected_simulation_count(final_state, pdb_list) if is_multi else 1

    simulations: List[Dict[str, Any]] = []
    if is_multi:
        simulations = _simulations_merged_from_state(final_state)

    n_success = sum(1 for s in simulations if s.get("status") == "success")
    n_skipped = sum(1 for s in simulations if s.get("skipped") or s.get("status") == "skipped")
    n_failed = sum(1 for s in simulations if s.get("status") in ("failed", "not_started"))
    n_submitted = sum(1 for s in simulations if s.get("status") == "submitted")
    n_pending = sum(1 for s in simulations if s.get("status") == "pending")
    n_in_progress = sum(1 for s in simulations if s.get("status") == "in_progress")

    combined_dir = str(Path(working_dir) / "combinedAnalysis")
    analysis_dir = final_state.get("analysis_directory") or str(Path(working_dir) / "analysis")
    n_healthy = sum(1 for s in simulations if s.get("health") == "healthy" or s.get("status") == "success")
    n_unhealthy = sum(1 for s in simulations if s.get("health") == "failed" or s.get("status") == "failed")

    science: Dict[str, Any] = {}
    try:
        from agentic.campaign.contracts import campaign_science_report
        from agentic.campaign.spec import spec_from_state

        spec = spec_from_state(final_state)
        if spec and spec.family_modular:
            science = campaign_science_report(working_dir, spec=spec, state=final_state)
    except Exception:
        science = {}

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "working_directory": working_dir,
        "mode": "multi_simulation" if is_multi else "single_simulation",
        "goal": goal,
        "subtask": (
            config.get("subtask")
            or final_state.get("pipeline_subtask_type")
            or (
                "full_task"
                if is_multi and final_state.get("hpc_pool")
                else final_state.get("subtask_type")
            )
        ),
        "source_pdbs": [Path(p).name for p in (final_state.get("pdb_list") or pdb_list)],
        "domain_context": final_state.get("domain_context"),
        "structure_request": final_state.get("structure_request"),
        "counts": {
            "total_simulations": total,
            "planned_simulations": len(final_state.get("sim_prompts") or []),
            "source_pdbs": len(final_state.get("pdb_list") or pdb_list),
            "missing_from_plan": missing,
            "completed_success": n_success,
            "healthy": n_healthy,
            "failed": n_failed,
            "unhealthy": n_unhealthy,
            "submitted_running": n_submitted,
            "skipped": n_skipped,
            "pending": n_pending,
            "in_progress": n_in_progress,
            "snapshots_recorded": len(completed),
        },
        "simulations": simulations,
        "combined_analysis_dir": combined_dir if Path(combined_dir).exists() else None,
        "analysis_dir": analysis_dir if Path(analysis_dir).exists() else None,
        "reporter_output": final_state.get("reporter_output"),
        "science": science,
        "campaign_spec": final_state.get("campaign_spec"),
        "final_report_present": bool(final_state.get("final_report")),
        "errors": list(final_state.get("errors") or [])[:20],
        "warnings": list(final_state.get("warnings") or [])[:20],
        "log_file": str(Path(working_dir) / "agent_conversation.log"),
        "execution_report": str(Path(working_dir) / "supervisor" / "execution_report.md"),
        "llm_usage_file": str(Path(working_dir) / "llm_usage.json"),
    }


def format_run_summary_terminal(summary: Dict[str, Any]) -> str:
    """Format summary for terminal display."""
    lines = [
        "=" * 60,
        "WORKFLOW RUN SUMMARY",
        "=" * 60,
    ]
    counts = summary.get("counts", {})
    mode = summary.get("mode", "unknown")
    lines.append(f"  Mode: {mode}")
    lines.append(f"  Working directory: {summary.get('working_directory')}")

    if mode == "multi_simulation":
        src = counts.get("source_pdbs", 0)
        total = counts.get("total_simulations", 0)
        planned = counts.get("planned_simulations", total)
        missing = counts.get("missing_from_plan") or []
        if src and total > planned:
            lines.append(
                f"  Simulations: {total} requested ({planned} in master plan, "
                f"{src} source PDB(s))"
            )
        elif src and total > src:
            lines.append(
                f"  Simulations: {total} total "
                f"({src} source structure(s) × component cases)"
            )
        else:
            lines.append(f"  Simulations: {total} total")
        if missing:
            lines.append(f"  Not in master plan: {', '.join(missing)}")
        lines.append(f"  Succeeded: {counts.get('completed_success', 0)}")
        lines.append(f"  Healthy:   {counts.get('healthy', counts.get('completed_success', 0))}")
        pending_n = counts.get("pending", 0)
        in_prog = counts.get("in_progress", 0)
        if pending_n or in_prog:
            lines.append(f"  Remaining: {pending_n + in_prog} ({in_prog} in progress)")
        lines.append(f"  Submitted: {counts.get('submitted_running', 0)}")
        lines.append(f"  Skipped:   {counts.get('skipped', 0)}")
        lines.append(f"  Failed:    {counts.get('failed', 0)}")

        sims = summary.get("simulations") or []
        if sims:
            lines.append("")
            lines.append("  Per-simulation:")
            for sim in sims:
                label = sim.get("label", "?")
                status = sim.get("status", "unknown")
                health = sim.get("health") or ("failed" if status == "failed" else "—")
                icon = {"success": "✓", "skipped": "⊘", "failed": "✗"}.get(status, "?")
                line = f"    {icon} {label} — {status} [{health}]"
                if sim.get("skip_reason"):
                    line += f" ({sim['skip_reason']})"
                elif sim.get("errors"):
                    line += f" ({sim['errors'][0][:80]})"
                elif sim.get("job_id"):
                    line += f" (job {sim['job_id']})"
                lines.append(line)
    else:
        err_n = len(summary.get("errors") or [])
        warn_n = len(summary.get("warnings") or [])
        lines.append(f"  Errors: {err_n}  |  Warnings: {warn_n}")

    if summary.get("domain_context"):
        lines.append("")
        lines.append("  Domain target:")
        for dl in summary["domain_context"].splitlines()[:4]:
            lines.append(f"    {dl}")

    if summary.get("combined_analysis_dir"):
        lines.append(f"  Combined analysis: {summary['combined_analysis_dir']}")
    if summary.get("reporter_output"):
        lines.append(f"  Report: {summary['reporter_output']}")

    usage_path = summary.get("llm_usage_file") or str(
        Path(summary.get("working_directory") or ".") / "llm_usage.json"
    )
    lines.append(f"  LLM usage file: {usage_path}")

    lines.append("")
    lines.append(f"  Summary file: {Path(summary['working_directory']) / 'run_summary.md'}")
    lines.append(f"  Full log: {summary.get('log_file')}")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_run_summary_markdown(summary: Dict[str, Any]) -> str:
    """Render run summary as Markdown."""
    counts = summary.get("counts", {})
    lines = [
        "# SimAgent Workflow Run Summary",
        "",
        f"**Generated:** {summary.get('generated_at')}",
        f"**Mode:** {summary.get('mode')}",
        f"**Working directory:** `{summary.get('working_directory')}`",
        "",
        "## Goal",
        "",
        f"{summary.get('goal', '')}",
        "",
        "## Counts",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total simulations | {counts.get('total_simulations', '—')} |",
        f"| Source PDBs | {counts.get('source_pdbs', '—')} |",
        f"| Succeeded | {counts.get('completed_success', 0)} |",
        f"| Healthy | {counts.get('healthy', counts.get('completed_success', 0))} |",
        f"| Skipped | {counts.get('skipped', 0)} |",
        f"| Failed | {counts.get('failed', 0)} |",
        "",
    ]
    science = summary.get("science") or {}
    if science:
        lines.extend(
            [
                "## Science completeness",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Family modular | {science.get('family_modular')} |",
                f"| Systems with required artifacts | "
                f"{science.get('n_science_complete', 0)}/"
                f"{science.get('n_systems', 0)} |",
                f"| Feature matrix ready | {bool((science.get('matrix') or {}).get('ok'))} |",
                f"| Campaign science OK | {science.get('ok')} |",
                "",
            ]
        )
        incomplete = science.get("incomplete_labels") or []
        if incomplete:
            lines.append("Incomplete labels: " + ", ".join(incomplete))
            lines.append("")
        reason = (science.get("matrix") or {}).get("reason")
        if reason:
            lines.append(f"Matrix: {reason}")
            lines.append("")


    if summary.get("domain_context"):
        lines.extend(["## Domain / Structure", "", summary["domain_context"], ""])

    sims = summary.get("simulations") or []
    if sims:
        lines.extend(["## Simulations", ""])
        for sim in sims:
            label = sim.get("label", "?")
            status = sim.get("status", "unknown")
            lines.append(f"### {label} — {status}")
            if sim.get("health"):
                lines.append(f"- **Health:** {sim['health']}")
            if sim.get("working_directory"):
                lines.append(f"- **Directory:** `{sim['working_directory']}`")
            if sim.get("job_id"):
                lines.append(f"- **Job ID:** {sim['job_id']}")
            if sim.get("topology"):
                lines.append(f"- **Topology:** `{sim['topology']}`")
            if sim.get("skip_reason"):
                lines.append(f"- **Skip reason:** {sim['skip_reason']}")
            if sim.get("errors"):
                lines.append("- **Errors:**")
                for err in sim["errors"][:5]:
                    lines.append(f"  - {err}")
            lines.append("")

    if summary.get("errors"):
        lines.extend(["## Global Errors", ""])
        for err in summary["errors"]:
            lines.append(f"- {err}")
        lines.append("")

    if summary.get("warnings"):
        lines.extend(["## Global Warnings", ""])
        for warn in summary["warnings"][:15]:
            lines.append(f"- {warn}")
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Log: `{summary.get('log_file')}`",
            f"- Execution report: `{summary.get('execution_report')}`",
        ]
    )
    if summary.get("combined_analysis_dir"):
        lines.append(f"- Combined analysis: `{summary['combined_analysis_dir']}`")
    if summary.get("reporter_output"):
        lines.append(f"- Reporter output: `{summary['reporter_output']}`")

    usage_path = summary.get("llm_usage_file") or str(
        Path(summary.get("working_directory") or ".") / "llm_usage.json"
    )
    lines.extend(
        [
            "",
            "## LLM usage",
            "",
            f"Token ledger is kept only in `{usage_path}` (not duplicated here).",
        ]
    )

    return "\n".join(lines)


def write_run_summary(working_dir: str, summary: Dict[str, Any]) -> Dict[str, str]:
    """Write run_summary.json and run_summary.md to the base working directory."""
    base = Path(working_dir)
    base.mkdir(parents=True, exist_ok=True)

    json_path = base / "run_summary.json"
    md_path = base / "run_summary.md"

    json_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    md_path.write_text(format_run_summary_markdown(summary), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}
