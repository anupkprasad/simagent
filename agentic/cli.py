"""SimAgent command line.

`simagent` and `python SimAgent.py` both call `main` in this module.
"""
import argparse
import sys
import os
import asyncio
import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure acpype/antechamber/gmx are on PATH before any simsetup tool import.
try:
    from src.simsetup.md_env import ensure_md_toolchain

    ensure_md_toolchain()
except Exception:
    pass

from agentic.workflow import MDWorkflow
from agentic.llm import LLMClient
from agentic.llm_usage import TokenBudgetExceeded
from agentic.utils import (
    log_user_prompt, set_log_file
)
from src.utils.run_summary import (
    build_run_summary,
    format_run_summary_terminal,
    write_run_summary,
)
from src.utils.chat_tools import (
    BUILTIN_TOOL_NAMES as _BUILTIN_TOOLS,
    TOOL_CALL_PATTERN as _TOOL_PATTERN,
    TOOL_INTENT_PATTERNS,
    MAX_TOOL_ROUNDS as _MAX_TOOL_ROUNDS,
    MAX_TASK_TOOL_ROUNDS as _MAX_TASK_TOOL_ROUNDS,
    TOOL_INSTRUCTIONS as _TOOL_INSTRUCTIONS,
    resolve_path as _resolve_path,
    read_file_tool as _read_file_snippet,
    list_dir_tool as _list_dir_safe,
    write_file_tool as _write_file_safe,
    extract_tool_intent as _extract_tool_intent,
    extract_tool_call_line as _extract_tool_call_line,
    execute_tool_call as _execute_tool_call,
    execute_domain_tool as _execute_domain_tool,
    append_execution_log as _append_execution_log,
)
from agentic.hitl_router import (
    parse_hitl_command,
    format_execute_command,
    bind_agent_context,
    sync_hitl_active_agent_for_checkpoint,
    merge_hitl_context_into_state_jsonl,
    hitl_view_working_directory,
    hitl_agent_output_directory,
    format_hitl_directory_context,
    format_hitl_pwd_lines,
    hitl_agent_dirs,
    agents_in_workflow,
    artifact_summary,
    AGENT_DISPLAY,
)
from agentic.hitl_agent_runner import run_hitl_agent_task

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ---------------------------------------------------------------------------
# Action keywords that end the interactive session and return a decision
# ---------------------------------------------------------------------------
_ACTION_KEYWORDS = {
    "approved", "continue", "retry", "redo", "exit", "quit", "stop"
}

# Friendly display names for agents
_AGENT_DISPLAY_NAMES = {
    "preprocess": "Preprocessing Agent",
    "setup": "Simulation Setup Agent",
    "hpc": "HPC Agent",
    "analysis": "Analysis Agent",
    "reporter": "Reporter Agent",
}

# Natural-language patterns that map to action keywords.
# Each tuple is (list of phrases, action keyword to return).
_NL_ACTION_PATTERNS = [
    # retry / redo patterns
    (["redo simulation", "redo setup", "redo the setup", "redo the simulation",
      "redo preprocessing", "redo preprocess", "redo analysis", "redo hpc",
      "rerun simulation", "rerun setup", "rerun the setup", "rerun the simulation",
      "please redo", "please retry", "please rerun",
      "run it again", "run again", "run setup again", "run the setup again",
      "re-run", "re-do"], "retry"),
    # approve / continue patterns
    (["looks good", "looks fine", "this is fine", "that's fine", "go ahead",
      "proceed", "move on", "move forward", "all good", "no issues",
      "i'm happy", "i am happy", "i am satisfied", "accept"], "approved"),
    # exit patterns
    (["stop the workflow", "cancel", "abort", "stop everything"], "exit"),
]


def _is_action(text: str) -> bool:
    """Return True if the user input is a workflow action (not a question)."""
    lower = text.lower().strip()
    # Exact match
    if lower in _ACTION_KEYWORDS:
        return True
    # "modify: ..." or "recommend: ..." prefix
    if lower.startswith("modify") or lower.startswith("recommend"):
        return True
    # Natural-language action detection
    if _normalize_action(lower):
        return True
    return False


def _normalize_action(text: str) -> str:
    """Map natural-language input to a canonical action keyword, or return empty string.
    
    If the text contains extra context beyond the action intent, wraps it as
    a 'recommend:' so the agent retries with the human's guidance.
    """
    lower = text.lower().strip()
    
    # Exact keyword match — return as-is
    if lower in _ACTION_KEYWORDS:
        return lower
    if lower.startswith("modify") or lower.startswith("recommend"):
        return lower
    
    for phrases, action in _NL_ACTION_PATTERNS:
        for phrase in phrases:
            if phrase in lower:
                # If the user included extra instructions beyond the action word,
                # convert to "recommend: <full text>" so the agent uses the guidance.
                # e.g. "redo setup with CHARMM force field" -> "recommend: redo setup with CHARMM force field"
                stripped = lower
                for p in phrases:
                    stripped = stripped.replace(p, "").strip()
                # Remove filler words
                for filler in ["please", "the", "with", "using", "and"]:
                    stripped = stripped.replace(filler, "").strip()
                stripped = " ".join(stripped.split())  # collapse whitespace
                
                if stripped and action == "retry" and len(stripped) > 3:
                    # User gave extra context like "updated execution plan" — 
                    # treat as recommend so the agent sees the guidance
                    return f"recommend: {text}"
                return action
    
    return ""


def _build_qa_context(summary: Dict[str, Any]) -> str:
    """Build a textual context block from the checkpoint summary for LLM Q&A."""
    parts = [f"Checkpoint: {summary['checkpoint_type']}"]
    
    if summary.get("error_triggered"):
        parts.append("STATUS: Error — agent needs human guidance")
    
    parts.append("\nCurrent state:")
    for k, v in summary.get("current_state", {}).items():
        parts.append(f"  {k}: {v}")
    
    issues = summary.get("issues_found", [])
    if issues:
        parts.append("\nIssues:")
        for i in issues:
            parts.append(f"  - {i}")
    
    recs = summary.get("recommendations", [])
    if recs:
        parts.append("\nRecommendations:")
        for r in recs:
            parts.append(f"  - {r}")

    if summary.get("available_agents"):
        parts.append(f"\nWorkflow field agents: {', '.join(summary['available_agents'])}")
    if summary.get("artifacts"):
        parts.append(f"\nActive agent artifacts: {summary['artifacts']}")
    
    return "\n".join(parts)


def _sync_hitl_chat_context(
    state: Dict[str, Any],
    active_agent: str,
    summary: Dict[str, Any],
) -> Tuple[str, str, str, Dict[str, str], str]:
    """Refresh HITL paths and Q&A context after switch or state reload."""
    working_dir = hitl_view_working_directory(state)
    output_dir = hitl_agent_output_directory(state, active_agent)
    agent_dirs = hitl_agent_dirs(state, working_dir)
    agent_label = AGENT_DISPLAY.get(
        active_agent,
        _AGENT_DISPLAY_NAMES.get(active_agent, active_agent.title() + " Agent"),
    )
    qa_context = (
        _build_qa_context(summary)
        + "\n\n"
        + format_hitl_directory_context(state, active_agent)
    )
    return working_dir, output_dir, agent_dirs, qa_context, agent_label

_TASK_EXECUTION_RE = re.compile(
    r"\b("
    r"calculate|compute|run|perform|execute|plot|generate|create|analyse|analyze|"
    r"dssp|secondary\s+structure|rmsf|rmsd|radius\s+of\s+gyration|rg|sasa|dccm|"
    r"com\s+distance|energy|wrap|heatmap"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_ONLY_RE = re.compile(
    r"^(who are you|what(?:'s| is) your name|what tools|what files|list files|"
    r"show me|explain|describe|why |how many|what did you)\b",
    re.IGNORECASE,
)

_PWD_QUESTION_RE = re.compile(
    r"^(?:what(?:'s| is)\s+(?:your\s+)?(?:the\s+)?(?:current\s+)?(?:working\s+)?"
    r"(?:dir(?:ectory)?|folder|path)|"
    r"(?:your\s+)?(?:current\s+)?(?:working\s+)?(?:dir(?:ectory)?|folder|path)|"
    r"pwd|cwd)\s*\??$",
    re.IGNORECASE,
)


def _is_pwd_question(text: str) -> bool:
    """True for pwd/cwd and natural-language directory questions (typo-tolerant)."""
    raw = text.strip()
    if not raw:
        return False
    t = raw.rstrip("?").lower().strip()
    if t in ("pwd", "cwd", "your current directory", "current directory"):
        return True
    if _PWD_QUESTION_RE.match(raw):
        return True
    if re.search(r"\bwhat\b", t) and re.search(r"\b(dir\w*|folder|path)\b", t):
        return True
    if re.search(r"\b(curr+\w*|working|your)\b", t) and re.search(
        r"\b(dir\w*|folder|path)\w*\b", t
    ):
        return True
    return False

_COT_MARKERS = (
    "we need to decide",
    "according to the rule",
    "thus we should",
    "let's assume",
    "the tool likely requires",
    "we must answer accordingly",
    "we should produce",
    "we can guess",
    "the spec not provided",
)


def _is_task_execution_request(user_input: str, domain_tools: Optional[Dict[str, Any]]) -> bool:
    """True when the user wants the agent to run analysis/tools (not just Q&A)."""
    if not domain_tools:
        return False
    text = user_input.strip()
    if _QUESTION_ONLY_RE.search(text):
        return False
    if parse_hitl_command(text):
        return False
    return bool(_TASK_EXECUTION_RE.search(text))


def _looks_like_chain_of_thought(text: str) -> bool:
    low = text.lower()
    if len(text) > 400 and ">>CALL:" not in text:
        return True
    return any(m in low for m in _COT_MARKERS)


def _strip_chain_of_thought(text: str) -> str:
    """Remove leaked LLM reasoning; keep >>CALL: lines and concise answers."""
    if not text:
        return text
    m = _extract_tool_call_line(text)
    if m:
        return m.group(0).strip()
    lines = []
    for line in text.splitlines():
        ls = line.strip().lower()
        if ls.startswith(">>call:"):
            lines.append(line.strip())
            continue
        if any(m in ls for m in _COT_MARKERS):
            continue
        if ls.startswith(("correct:", "wrong:", "example response", "example:")):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned or text[:300].strip()


def _discover_md_input_paths(working_dir: str) -> Dict[str, str]:
    """Find topology/trajectory paths under a simulation working directory."""
    wd = Path(working_dir)
    hpc = wd / "hpc"
    paths: Dict[str, str] = {"working_dir": str(wd / "analysis")}
    if not hpc.is_dir():
        return paths
    for name in ("md.tpr", "npt.tpr", "em.tpr", "topol.tpr"):
        p = hpc / name
        if p.is_file():
            paths["topology_file"] = str(p.resolve())
            break
    for name in ("md.xtc", "mdWrap.xtc", "md.trr", "npt.xtc", "em.xtc"):
        p = hpc / name
        if p.is_file():
            paths["trajectory_file"] = str(p.resolve())
            break
    return paths


def _hitl_execution_log_path(agent_dirs: Dict[str, str], active_agent: str) -> str:
    agent_dir = agent_dirs.get(active_agent) or agent_dirs.get("analysis") or ""
    if not agent_dir:
        return ""
    return str(Path(agent_dir) / "execution_log.txt")


def _hitl_conversation_log_path(sim_root: str) -> str:
    if not sim_root:
        return ""
    return str(Path(sim_root) / "agent_conversation.log")


def _format_md_paths_for_prompt(working_dir: str) -> str:
    paths = _discover_md_input_paths(working_dir)
    if not paths.get("topology_file") and not paths.get("trajectory_file"):
        return ""
    lines = ["MD INPUT FILES (use these paths in tool calls):"]
    for key in ("topology_file", "trajectory_file", "working_dir"):
        if paths.get(key):
            lines.append(f"  {key}: {paths[key]}")
    return "\n".join(lines)


def _try_direct_task_execution(
    user_input: str,
    domain_tools: Dict[str, Any],
    working_dir: str,
    log_path: str = "",
    output_dir: str = "",
    conversation_log_path: str = "",
) -> Optional[str]:
    """Deterministic fallback when the LLM fails to emit >>CALL: for known tasks."""
    low = user_input.lower()
    paths = _discover_md_input_paths(working_dir)
    topo = paths.get("topology_file")
    traj = paths.get("trajectory_file")
    out_dir = paths.get("working_dir", str(Path(working_dir) / "analysis"))

    tool_name = None
    kwargs: Dict[str, Any] = {}
    if any(k in low for k in ("dssp", "secondary structure", "secondary-structure")):
        tool_name = "analyze_secondary_structure"
        if topo and traj:
            kwargs = {
                "topology_file": topo,
                "trajectory_file": traj,
                "working_dir": out_dir,
                "selection": "protein",
                "output_prefix": "dssp",
            }
    elif "rmsf" in low and "calculate_rmsf" in domain_tools:
        tool_name = "calculate_rmsf"
    elif re.search(r"\brmsd\b", low) and "calculate_rmsd" in domain_tools:
        tool_name = "calculate_rmsd"
    elif re.search(r"\b(rg|radius of gyration|gyration)\b", low):
        tool_name = "calculate_radius_of_gyration"

    if not tool_name or tool_name not in domain_tools:
        return None
    if not kwargs and topo and traj:
        kwargs = {
            "topology_file": topo,
            "trajectory_file": traj,
            "working_dir": out_dir,
        }
    if not kwargs:
        return None

    tool_line = f">>CALL: {tool_name} | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"  [executing: {tool_name} (direct)]", flush=True)
    result = _execute_tool_call(
        tool_line, working_dir, {}, domain_tools,
        log_path=log_path or None,
        user_request=user_input,
        output_dir=output_dir or None,
        conversation_log_path=conversation_log_path or None,
    )
    return f"Executed {tool_name}.\n{result}"


# _read_file_snippet, _resolve_path, _list_dir_safe, _write_file_safe
# are imported from src.utils.chat_tools at the top of this file.


def _persist_state_to_jsonl(state: dict, working_dir: str) -> bool:
    """Write state dict to {working_dir}/supervisor/state.jsonl. Returns True on success."""
    state_path = Path(working_dir) / "supervisor" / "state.jsonl"
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"state": state}, indent=2, default=str), encoding="utf-8")
        return True
    except Exception:
        return False


def _replan_with_guidance(
    llm,
    user_input: str,
    state: dict,
    working_dir: str,
) -> Optional[str]:
    """Apply human guidance to structured plans in state and persist.

    1. Detects which agent's plan(s) the guidance targets.
    2. Instantiates the matching field-agent class with the shared LLM client.
    3. Calls agent.replan_with_guidance(user_input, state) — same prompt infra
       as normal planning: planner NL instructions + available tools + human rec.
    4. Stores updated plan + human_recommendation in state and writes state.jsonl.
    5. Returns a printable summary, or None if nothing was updated.
    """
    exec_plan = state.get("execution_plan") or {}
    structured_plans = exec_plan.get("structured_plans", {})

    # Detect targeted agent(s) from the user's text
    _agent_aliases = {
        "preprocess": ["preprocess", "preprocessing"],
        "setup": ["setup", "simsetup", "simulation setup"],
        "hpc": ["hpc"],
        "analysis": ["analysis"],
    }
    inp_lower = user_input.lower()
    targets = [
        key for key, aliases in _agent_aliases.items()
        if any(a in inp_lower for a in aliases) and key in structured_plans
    ]
    if not targets:
        # No named agent — apply to all available structured plans
        targets = list(structured_plans.keys())
    if not targets:
        return None

    # Lazy-import field agent classes to avoid circular imports at module level
    try:
        from agentic.preprocess.preprocessing_agent import PreprocessingAgent as _PreprocessAgent
        from agentic.simsetup.setup_agent import SimSetupAgent as _SetupAgent
        from agentic.analysis.analysis_agent import MDAnalysisAgent as _AnalysisAgent
        from agentic.hpc.hpc_agent import MDHPCAgent as _HPCAgent
    except ImportError as _ie:
        return f"[Import error — cannot replan: {_ie}]"

    _agent_cls_map = {
        "preprocess": _PreprocessAgent,
        "setup": _SetupAgent,
        "analysis": _AnalysisAgent,
        "hpc": _HPCAgent,
    }

    updated_agents = []
    for target_agent in targets:
        agent_cls = _agent_cls_map.get(target_agent)
        if agent_cls is None:
            continue
        try:
            agent_instance = agent_cls(llm_client=llm)
            updated_plan = agent_instance.replan_with_guidance(user_input, state)
        except Exception:
            updated_plan = None
        if not updated_plan:
            continue
        exec_plan.setdefault("structured_plans", {})[target_agent] = updated_plan
        updated_agents.append(target_agent)

    if not updated_agents:
        return None

    # Store guidance in state so agent's LLM planner sees it on next run
    state["human_recommendation"] = user_input
    state["execution_plan"] = exec_plan

    # Apply parameter overrides (force_field, water_model, temperature, pressure, production_ns)
    _param_patterns = [
        (re.compile(r'\b(?:force.?field|ff)\s*[:=]?\s*([\w-]+)', re.I), "force_field"),
        (re.compile(r'\bwater.?model\s*[:=]?\s*(\w+)', re.I), "water_model"),
        (re.compile(r'\btemperature\s*[:=]?\s*(\d+(?:\.\d+)?)\s*k?\b', re.I), "temperature"),
        (re.compile(r'\bpressure\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:bar)?\b', re.I), "pressure"),
        (re.compile(r'(?:simulation\s*(?:time|length|duration|ns)|production(?:\s*run)?|run\s*(?:time|length)?)\s*[:=of]?\s*(\d+(?:\.\d+)?)\s*(?:ns|nanoseconds?)?\b|\b(\d+(?:\.\d+)?)\s*(?:ns|nanoseconds?)\b', re.I), "production_ns"),
    ]
    for pattern, key in _param_patterns:
        m = pattern.search(user_input)
        if m:
            val = next((g for g in m.groups() if g is not None), None)
            if key in ("temperature", "pressure", "production_ns"):
                try:
                    val = float(val)
                except ValueError:
                    pass
            state[key] = val

    _persist_state_to_jsonl(state, working_dir)

    n_steps = {a: len(exec_plan["structured_plans"][a].get("steps", [])) for a in updated_agents}
    summary_lines = [f"Plan updated for: {', '.join(updated_agents)}"]
    for ag in updated_agents:
        summary_lines.append(f"  • {ag}: {n_steps[ag]} steps")
    summary_lines.append(
        "State saved. Type 'retry' to re-run the agent with the updated plan, "
        "or keep chatting to review further."
    )
    return "\n".join(summary_lines)


def _llm_route_or_replan(
    llm,
    user_input: str,
    state: dict,
    working_dir: str,
    agent_name: str = "MD Workflow Assistant",
    domain_tools: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Use the LLM to decide intent: answer question OR update a structured plan.

    The LLM sees the current structured plans + available tools and responds with
    either a plain answer or a JSON plan update (signalled by a sentinel prefix).

    Returns:
        str  — human-readable message to print (answer OR "Plan updated: ...")
        None — LLM unavailable / mock mode
    """
    import json as _j

    exec_plan = state.get("execution_plan") or {}
    structured_plans = exec_plan.get("structured_plans", {})
    avail_agents = list(structured_plans.keys())

    # Separate full_plan (planner narrative) from structured_plans (agent step lists)
    full_plan = exec_plan.get("full_plan", "")
    full_plan_section = ""
    if full_plan:
        full_plan_section = f"\n\nPLANNER FULL EXECUTION PLAN (full_plan):\n{full_plan}"

    # Snapshot of current plans to give the LLM full context
    plans_snapshot = ""
    if structured_plans:
        try:
            plans_snapshot = (
                "\n\nCURRENT STRUCTURED PLANS — per-agent step lists (structured_plans):\n"
                + _j.dumps(structured_plans, indent=2)
            )
        except Exception:
            plans_snapshot = f"\n\nStructured plans available for: {', '.join(avail_agents)}"

    prompt = f"""You are the {agent_name} in a molecular dynamics simulation workflow.
Your name is '{agent_name}'. When asked your name or role, identify yourself as the {agent_name}.
The human is chatting with you about the current workflow state and plans.

Your job is to decide whether the human's message is:
(A) A QUESTION  — answer it using the context provided.
(B) A PLAN UPDATE REQUEST — the human wants to change something in a structured plan.

PLAN ANSWERING RULES:
- "full plan" / "execution plan" / "planner plan" / "overall plan"
  → reproduce the EXACT text from PLANNER FULL EXECUTION PLAN below. Show it verbatim.
- "agent plan" / "analysis plan" / "structured plan" / "step plan"
  → reproduce the EXACT content from CURRENT STRUCTURED PLANS below.

For (B) plan updates — CRITICAL RULES:
1. Identify which agent's plan to change (preprocess / setup / hpc / analysis).
   If the request mentions temperature, simulation time/ns, pressure, force field, or water
   model, the target agent is ALWAYS "setup".
2. Take the EXISTING plan for that agent from CURRENT STRUCTURED PLANS below.
   DO NOT invent new steps. ONLY change the specific parameter values the human requested.
   Keep every other field (step names, descriptions, tool names, other tool_params) identical.
3. Reproduce the ENTIRE updated plan object as valid JSON — no comments, no trailing commas.
4. Start your response with EXACTLY this sentinel on its own line:
     PLAN_UPDATE:<agent_key>
   Then output ONLY the raw JSON object. Nothing else before or after the JSON.
   agent_key must be one of: preprocess, setup, hpc, analysis.
   Example response for a temperature change:
PLAN_UPDATE:setup
{{"reasoning":"Updated temperature to 310 K","steps":[...full step list...]}}

For (A), just answer the question directly. Do NOT include a sentinel.

Context:
- Simulation directory: {state.get("hitl_agent_working_directory") or state.get("working_directory", ".")}
- Agent output directory: {state.get("hitl_agent_output_directory") or state.get("analysis_dir", "")}
- Multi-sim project base (routing only): {state.get("multi_sim_base_dir") or state.get("working_directory", ".")}
- Force field: {state.get("force_field", "amber99sb-ildn")}
- Water model: {state.get("water_model", "tip3p")}
- Temperature: {state.get("temperature", 300.0)} K
- Available structured plans: {avail_agents if avail_agents else "none yet"}
{full_plan_section}
{plans_snapshot}

AVAILABLE TOOLS (call with >>CALL: tool_name | args):
  Built-in file tools (always available):
    read_file   | <filepath>                      — read file content (max 80 lines)
    list_dir    | <directory>                     — list directory contents
    write_file  | <filepath> | <content>          — write/overwrite a file
    grep_file   | <pattern>  | <filepath_or_dot>  — regex search in file(s)
  {agent_name} domain tools:
    {chr(10).join(('    ' + n) for n in (domain_tools or {}).keys()) or '    (none loaded)'}

TOOL CALL RULE — if you need to call a tool to answer, your ENTIRE response must be ONLY:
  >>CALL: tool_name | argument
No reasoning. No quotes around the call. No explanation before or after.

Human message:
{user_input}
"""

    try:
        resp = (getattr(llm, 'prompt_raw', None) or llm.prompt)(prompt, temperature=0.1)
    except Exception as e:
        return f"[LLM error: {e}]"

    if not resp or resp.startswith("MOCK_LLM_RESPONSE") or resp.startswith("LLM_ERROR"):
        return None

    # Check if LLM decided this is a plan update
    import re as _re
    sentinel_match = _re.search(r'PLAN_UPDATE:(\w+)[ \t]*\n([\s\S]+)', resp.strip())
    if sentinel_match:
        agent_key = sentinel_match.group(1).strip().lower()
        json_body = sentinel_match.group(2).strip()

        # Strip markdown code fences if present
        json_body = _re.sub(r'^```(?:json)?\s*', '', json_body, flags=_re.MULTILINE)
        json_body = _re.sub(r'```\s*$', '', json_body, flags=_re.MULTILINE).strip()

        if agent_key not in ("preprocess", "setup", "hpc", "analysis"):
            # Not a recognised key — treat as plain answer
            return resp.strip()

        # --- Robust JSON extraction ---
        # Try the body as-is first, then fall back to extracting the outermost { } block
        updated_plan = None
        for candidate in [json_body, None]:
            if candidate is None:
                # find the outermost matching { } in json_body
                brace_m = _re.search(r'\{', json_body)
                if not brace_m:
                    break
                depth, start = 0, brace_m.start()
                for ci, ch in enumerate(json_body[start:], start):
                    if ch == '{': depth += 1
                    elif ch == '}': depth -= 1
                    if depth == 0:
                        candidate = json_body[start: ci + 1]
                        break
                if candidate is None:
                    break
            try:
                updated_plan = _j.loads(candidate)
                break
            except _j.JSONDecodeError:
                continue

        if updated_plan is None:
            return (
                "Could not parse updated plan JSON from LLM response. "
                "Please try again or rephrase your request."
            )

        # --- Scalar param extraction from user text ---
        _param_specs = [
            (_re.compile(r'\b(?:force.?field|ff)\s*[:=]?\s*([\w-]+)', _re.I),
             "force_field", ["force_field"], str),
            (_re.compile(r'\bwater.?model\s*[:=]?\s*(\w+)', _re.I),
             "water_model", ["water_model"], str),
            (_re.compile(
                r'(?:temperature|temp)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*k?\b'
                r'|(\d+(?:\.\d+)?)\s*k\b',
                _re.I),
             "temperature", ["temperature"], float),
            (_re.compile(r'\bpressure\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:bar)?\b', _re.I),
             "pressure", ["pressure"], float),
            (_re.compile(
                r'(?:simulation\s*(?:time|length|duration|ns)|production(?:\s*run)?|run\s*(?:time|length)?)'
                r'\s*[:=of]?\s*(\d+(?:\.\d+)?)\s*(?:ns|nanoseconds?)?\b'
                r'|(\d+(?:\.\d+)?)\s*(?:ns|nanoseconds?)\b',
                _re.I),
             "production_ns", ["production_ns"], float),
        ]

        for pat, state_key, tool_keys, cast in _param_specs:
            pm = pat.search(user_input)
            if pm:
                raw = next((g for g in pm.groups() if g is not None), None)
                if raw is None:
                    continue
                try:
                    val: Any = cast(raw)
                except (ValueError, TypeError):
                    val = raw
                state[state_key] = val
                # Also patch tool_params in every step of the updated plan
                for step in updated_plan.get("steps", []):
                    tp = step.get("tool_params")
                    if isinstance(tp, dict):
                        for tk in tool_keys:
                            if tk in tp:
                                tp[tk] = val

        exec_plan.setdefault("structured_plans", {})[agent_key] = updated_plan
        state["execution_plan"] = exec_plan
        state["human_recommendation"] = user_input

        _persist_state_to_jsonl(state, working_dir)
        n_steps = len(updated_plan.get("steps", []))
        return (
            f"Plan updated for '{agent_key}' agent ({n_steps} steps). "
            f"Changes saved to state.\n"
            f"Type 'show {agent_key} plan' to review, or 'retry' to re-run."
        )

    # Plain answer — return as-is
    return resp.strip()


def _load_full_state_from_jsonl(working_dir: str) -> dict:
    """Return the raw state dict from state.jsonl, or empty dict on failure."""
    state_path = Path(working_dir) / "supervisor" / "state.jsonl"
    if not state_path.exists():
        return {}
    try:
        content = state_path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        entry = json.loads(content)
        return entry.get("state", {})
    except Exception:
        return {}


def _load_last_state_snapshot(working_dir: str) -> str:
    """Load the current workflow state from state.jsonl for LLM context.
    Includes execution_plan summary so the LLM can quote plans verbatim.
    """
    st = _load_full_state_from_jsonl(working_dir)
    if not st:
        return ""
    try:
        # Extract the most useful fields
        useful_keys = [
            "user_goal", "force_field", "water_model",
            "raw_pdb", "cleaned_pdb", "preprocessing_report",
            "ligand_resnames", "ion_resnames",
            "topology", "coordinates", "mdp_files", "setup_report",
            "job_script", "job_id", "job_status", "hpc_report",
            "analysis_results", "figures", "conclusions",
            "errors", "warnings", "execution_path",
            "working_directory", "preprocess_dir", "simsetup_dir",
            "hpc_dir", "analysis_dir",
        ]
        filtered = {k: st[k] for k in useful_keys if k in st and st[k]}

        # Include execution_plan info for plan-related questions
        exec_plan = st.get("execution_plan")
        if exec_plan and isinstance(exec_plan, dict):
            plan_summary: dict = {}
            full_plan = exec_plan.get("full_plan", "")
            if full_plan:
                # Include first 1500 chars of full NL plan
                plan_summary["full_plan"] = (
                    full_plan[:1500] + ("...(truncated)" if len(full_plan) > 1500 else "")
                )
            plan_summary["agent_sequence"] = exec_plan.get("agent_sequence", [])
            plan_summary["title"] = exec_plan.get("title", "")
            # Include structured plans as-is (usually compact JSON)
            structured_plans = exec_plan.get("structured_plans", {})
            if structured_plans:
                plan_summary["structured_plans"] = structured_plans
            # Include per-agent NL instruction sections
            agent_plans = exec_plan.get("agent_plans", {})
            if agent_plans:
                plan_summary["agent_nl_instructions"] = {
                    k: (v[:800] + "...(truncated)" if len(v) > 800 else v)
                    for k, v in agent_plans.items()
                }
            filtered["execution_plan"] = plan_summary

        return json.dumps(filtered, indent=2, default=str)
    except Exception:
        return ""


def _load_execution_report(working_dir: str) -> str:
    """Load execution_report.md for LLM context."""
    report_path = Path(working_dir) / "supervisor" / "execution_report.md"
    if not report_path.exists():
        return ""
    try:
        text = report_path.read_text(encoding="utf-8", errors="replace")
        # Truncate if very long
        if len(text) > 3000:
            text = text[:3000] + "\n... (truncated)"
        return text
    except Exception:
        return ""


# Tool constants and helpers are imported from src.utils.chat_tools:
#   TOOL_CALL_PATTERN, BUILTIN_TOOL_NAMES, TOOL_INTENT_PATTERNS, extract_tool_intent
#   execute_domain_tool, TOOL_INSTRUCTIONS, MAX_TOOL_ROUNDS, execute_tool_call

# ---------------------------------------------------------------------------
# Agent domain tool registry — loads StructuredTool objects per checkpoint type
# ---------------------------------------------------------------------------

def _load_agent_domain_tools(checkpoint_type: str, working_dir: str = "") -> Tuple[Dict[str, Any], str]:
    """Load domain-specific tools for the agent at a checkpoint.

    Returns:
        (tool_objects, tool_instructions_text)
        - tool_objects: Dict mapping tool name -> StructuredTool object
        - tool_instructions_text: Formatted text describing the tools for the LLM prompt
    """
    tools_map: Dict[str, Any] = {}
    metadata: Dict[str, Dict[str, Any]] = {}

    try:
        if checkpoint_type == "preprocess":
            from agentic.preprocess.tools import get_preprocessing_tools, get_tool_metadata
            for tool in get_preprocessing_tools():
                tools_map[tool.name] = tool
            metadata = get_tool_metadata()

        elif checkpoint_type == "setup":
            from agentic.simsetup.tools import get_simulation_setup_tools
            from agentic.simsetup.tools import get_tool_metadata as _get_meta
            for tool in get_simulation_setup_tools():
                tools_map[tool.name] = tool
            metadata = _get_meta()

        elif checkpoint_type == "hpc":
            from agentic.hpc.tools import (
                copy_simulation_files, estimate_simulation_time,
                create_slurm_script, submit_job, check_job_status, download_results,
            )
            hpc_tools = [
                copy_simulation_files, estimate_simulation_time,
                create_slurm_script, submit_job, check_job_status, download_results,
            ]
            for tool in hpc_tools:
                tools_map[tool.name] = tool
            # Build metadata manually (HPC has no standalone get_tool_metadata)
            for tool in hpc_tools:
                tool_info: Dict[str, Any] = {"name": tool.name, "description": tool.description, "args": {}}
                if hasattr(tool, "args_schema") and tool.args_schema:
                    schema = tool.args_schema.schema()
                    if "properties" in schema:
                        tool_info["args"] = {
                            k: {
                                "type": v.get("type", "string"),
                                "description": v.get("description", ""),
                                "required": k in schema.get("required", []),
                            }
                            for k, v in schema["properties"].items()
                        }
                metadata[tool.name] = tool_info

        elif checkpoint_type == "analysis":
            from agentic.analysis.tools import get_analysis_tools
            from agentic.analysis.tools import get_tool_metadata as _get_meta_a
            for tool in get_analysis_tools():
                tools_map[tool.name] = tool
            metadata = _get_meta_a(working_directory=working_dir or None)

        elif checkpoint_type == "reporter":
            from agentic.reporter.tools import get_reporter_tools
            from agentic.reporter.tools import get_tool_metadata as _get_meta_r
            for tool in get_reporter_tools():
                tools_map[tool.name] = tool
            metadata = _get_meta_r()

    except Exception as e:
        # Use print instead of logger since this is a utility function
        print(f"Warning: Could not load domain tools for {checkpoint_type}: {e}", file=sys.stderr)
        return {}, ""

    if not metadata:
        return tools_map, ""

    # Build formatted instructions text
    agent_label = _AGENT_DISPLAY_NAMES.get(checkpoint_type, checkpoint_type.title())
    lines = [f"\nIn addition to the built-in file tools above, you have these {agent_label} domain tools:"]
    for name, info in metadata.items():
        desc = info.get("description", "")
        if len(desc) > 200:
            desc = desc[:200] + "..."
        lines.append(f"  - {name}: {desc}")
        args = info.get("args", {})
        if args:
            arg_parts = []
            for arg_name, arg_info in args.items():
                req = " (required)" if arg_info.get("required") else ""
                arg_parts.append(f"{arg_name}{req}")
            lines.append(f"    Args: {', '.join(arg_parts)}")

    lines.append("")
    lines.append("To call a domain tool, use the same >>CALL: format with key=value args separated by |:")
    lines.append("  >>CALL: <tool_name> | key1=value1 | key2=value2")
    lines.append("Examples:")
    # Add 1-2 examples based on checkpoint type
    if checkpoint_type == "preprocess":
        lines.append("  >>CALL: analyze_pdb | pdb_path=protein.pdb")
    elif checkpoint_type == "setup":
        lines.append("  >>CALL: build_simulation_system | working_dir=simsetup | system_type=protein-only")
    elif checkpoint_type == "analysis":
        lines.append("  >>CALL: calculate_rmsd | tpr_file=hpc/md.tpr | xtc_file=hpc/mdWrap.xtc | output_dir=analysis")
        lines.append("  >>CALL: analyze_secondary_structure | topology_file=hpc/md.tpr | trajectory_file=hpc/mdWrap.xtc | working_dir=analysis")
    elif checkpoint_type == "hpc":
        lines.append("  >>CALL: estimate_simulation_time | system_size=50000 | simulation_ns=100")

    return tools_map, "\n".join(lines)


# _execute_domain_tool, _TOOL_INSTRUCTIONS, _MAX_TOOL_ROUNDS, _execute_tool_call
# are imported from src.utils.chat_tools at the top of this file.


def _llm_qa_with_tools(
    llm,
    user_question: str,
    qa_context: str,
    conversation_history: List[Dict[str, str]],
    working_dir: str,
    agent_dirs: Dict[str, str],
    domain_tools: Optional[Dict[str, Any]] = None,
    domain_tool_instructions: str = "",
    agent_name: str = "MD Workflow",
    execution_mode: bool = False,
    log_path: str = "",
    output_dir: str = "",
    conversation_log_path: str = "",
) -> Optional[str]:
    """Run LLM Q&A with tool-calling loop.
    
    Uses prompt_raw() (/api/generate) to bypass Ollama's chat-endpoint
    tool-call parser which would choke on our text-based tool format.
    """
    
    # Pre-load key context files
    state_snapshot = _load_last_state_snapshot(working_dir)
    exec_report = _load_execution_report(working_dir)
    
    key_files_context = ""
    if state_snapshot:
        key_files_context += (
            f"\n\nLATEST WORKFLOW STATE (from state.jsonl — "
            f"prefer HITL ACTIVE CONTEXT above for directory paths):\n{state_snapshot}"
        )
    if exec_report:
        key_files_context += f"\n\nEXECUTION REPORT (execution_report.md):\n{exec_report}"
    
    # Build conversation history
    history_text = ""
    if conversation_history:
        history_text = "\n\nPrevious Q&A in this session:\n"
        for turn in conversation_history[-5:]:
            history_text += f"  User: {turn['q']}\n  Assistant: {turn['a']}\n"
    
    system_prompt = (
        f"You are the {agent_name} in a molecular dynamics simulation workflow. "
        f"Your name is '{agent_name}'. When asked your name or role, identify yourself as the {agent_name}. "
        + (
            "The user wants you to EXECUTE an analysis task using your domain tools. "
            "You MUST call the appropriate domain tool(s) immediately — do NOT explain your reasoning. "
            "Use the MD INPUT FILES paths provided below. After tools succeed, summarize output files created. "
            "Do NOT ask clarifying questions unless topology/trajectory files are truly missing.\n"
            if execution_mode else
            "The user is reviewing results at a human checkpoint in an MD workflow. "
            "IMPORTANT: HITL ACTIVE CONTEXT in the prompt defines your simulation and output directories. "
            "When asked for the current/working directory, report Simulation directory and Agent output directory "
            "from HITL ACTIVE CONTEXT — NOT the multi-sim project base from state.jsonl. "
            "Try to answer from the checkpoint context FIRST. "
            "Only call a tool if the context does not contain enough detail. "
            "Be concise. If the request is ambiguous, ask ONE clear follow-up question. "
            "Do NOT tell the user to approve or continue.\n"
        )
        + "PLAN QUESTIONS — use the following rules:\n"
        "  * 'full plan', 'execution plan', 'planner plan', 'overall plan' "
        "    → reproduce the EXACT text from execution_plan.full_plan in the workflow state. "
        "    This is the planner's complete natural-language narrative. Show it verbatim.\n"
        "  * 'agent plan', 'analysis plan', 'structured plan', 'step plan' "
        "    → reproduce the EXACT content from execution_plan.structured_plans.<agent_key>. "
        "    Show exact tool names, parameters, step descriptions.\n"
        "  * Both full_plan and structured_plans are embedded in PLANNER EXECUTION PLAN below.\n\n"
        + _TOOL_INSTRUCTIONS
        + (("\n" + domain_tool_instructions) if domain_tool_instructions else "")
    )
    
    # Build a compact tool inventory for injection into the prompt body so
    # the LLM sees all available tools regardless of which part of the prompt
    # it focuses on when answering "what tools do you have?"
    _builtin_inventory = (
        "AVAILABLE TOOLS (call with >>CALL: tool_name | args):\n"
        "  Built-in file tools (always available):\n"
        "    read_file   | <filepath>                      — read file content\n"
        "    list_dir    | <directory>                     — list directory contents\n"
        "    write_file  | <filepath> | <content>          — write/overwrite a file\n"
        "    grep_file   | <pattern>  | <filepath_or_dot>  — regex search in file(s)\n"
    )
    _domain_inventory = ""
    if domain_tools:
        _domain_inventory = (
            f"  {agent_name} domain tools:\n"
            + "".join(
                f"    {name}\n"
                for name in domain_tools
            )
        )
    _tool_inventory = _builtin_inventory + _domain_inventory

    # Embed the planner execution plan directly in the prompt body so the LLM
    # always has it available without needing to read state.jsonl via a tool.
    _exec_plan_section = ""
    _live_ep = _load_full_state_from_jsonl(working_dir)
    _ep = (_live_ep or {}).get("execution_plan") if _live_ep else None
    if not _ep:
        # fall back to domain_tools parent state if available
        import json as _j2
        _state_raw = _load_last_state_snapshot(working_dir)
        if _state_raw:
            try:
                _ep = _j2.loads(_state_raw).get("execution_plan") if _state_raw.strip().startswith("{") else None
            except Exception:
                _ep = None
    if _ep:
        _fp = _ep.get("full_plan", "")
        _sp = _ep.get("structured_plans", {})
        _exec_plan_section = "\nPLANNER EXECUTION PLAN:\n"
        if _fp:
            _exec_plan_section += f"  full_plan:\n{_fp}\n"
        if _sp:
            import json as _j3
            _exec_plan_section += f"  structured_plans:\n{_j3.dumps(_sp, indent=4)}\n"

    _md_paths_section = _format_md_paths_for_prompt(working_dir)
    _out_dir_note = ""
    if output_dir:
        _out_dir_note = f"\nAGENT OUTPUT DIRECTORY (write all analysis outputs here): {output_dir}\n"
    prompt_text = f"""CHECKPOINT CONTEXT:
{qa_context}
{key_files_context}
{_exec_plan_section}
{_md_paths_section}
{_out_dir_note}
{history_text}

{_tool_inventory}
USER QUESTION: {user_question}

{"TASK: Execute the requested analysis using domain tools NOW." if execution_mode else "If you can answer from the context above, give a FINAL ANSWER directly."}
TOOL CALL RULE — if you need a tool: your ENTIRE response must be ONLY the bare >>CALL: line.
  CORRECT:   >>CALL: analyze_secondary_structure | topology_file=hpc/md.tpr | trajectory_file=hpc/mdWrap.xtc | working_dir=analysis
  CORRECT:   >>CALL: read_file | rmsf.dat
  WRONG:     Use ">>CALL: read_file | rmsf.dat" or I will call read_file...
No reasoning. No quotes around the call. No explanation. Nothing else."""
    
    llm_call = getattr(llm, 'prompt_raw', None) or llm.prompt
    max_rounds = _MAX_TASK_TOOL_ROUNDS if execution_mode else _MAX_TOOL_ROUNDS
    
    response = ""
    for round_idx in range(max_rounds):
        try:
            response = llm_call(prompt_text, system=system_prompt)
        except Exception as e:
            return f"[LLM error: {e}]"
        
        if response.startswith("MOCK_LLM_RESPONSE"):
            return None  # Signal mock mode
        
        # Check if response contains a tool call
        tool_match = _extract_tool_call_line(response) or _TOOL_PATTERN.search(response)
        if not tool_match:
            synthetic_call = _extract_tool_intent(response, domain_tools)
            if synthetic_call:
                tool_match = _extract_tool_call_line(synthetic_call) or _TOOL_PATTERN.search(synthetic_call)
                if tool_match:
                    response = synthetic_call
            
            if not tool_match:
                if execution_mode and domain_tools:
                    direct = _try_direct_task_execution(
                        user_question, domain_tools, working_dir, log_path,
                        output_dir=output_dir, conversation_log_path=conversation_log_path,
                    )
                    if direct:
                        return direct
                return _strip_chain_of_thought(response.strip())
        
        tool_line = tool_match.group(0)
        tool_result = _execute_tool_call(
            tool_line, working_dir, agent_dirs, domain_tools,
            log_path=log_path or None,
            user_request=user_question,
            output_dir=output_dir or None,
            conversation_log_path=conversation_log_path or None,
        )
        
        # Show the user what tool was called (transparency)
        tool_name = tool_match.group(1)
        tool_arg = tool_match.group(2).strip().split("|")[0].strip()
        if tool_name in _BUILTIN_TOOLS:
            print(f"  [reading: {tool_arg}]", flush=True)
        else:
            print(f"  [executing: {tool_name}({tool_arg})]", flush=True)
        
        # Build follow-up prompt with tool result
        prompt_text = f"""CHECKPOINT CONTEXT:
{qa_context}
{key_files_context}
{history_text}

USER QUESTION: {user_question}

You called: {tool_line}
RESULT:
{tool_result}

Now give a FINAL ANSWER based on the result and context.
If you still need more information, output ONLY a tool call line — nothing else:
>>CALL: <tool_name> | <args>"""
    
    # Exhausted rounds.  If the last response is still a raw >>CALL: line the LLM
    # never gave a plain answer.  Make one final call with tools disabled.
    final_text = response.strip() if response else ""
    if _extract_tool_call_line(final_text) or _TOOL_PATTERN.search(final_text) or not final_text:
        if execution_mode and domain_tools:
            direct = _try_direct_task_execution(
                user_question, domain_tools, working_dir, log_path,
                output_dir=output_dir, conversation_log_path=conversation_log_path,
            )
            if direct:
                return direct
        try:
            no_tool_prompt = (
                f"USER QUESTION: {user_question}\n\n"
                f"You have already read files during this session. "
                f"Now give a concise PLAIN TEXT answer — no tool calls, no >>CALL: lines. "
                f"If the file content wasn't useful, say so directly."
            )
            final_text = llm_call(no_tool_prompt, system=(
                f"You are the {agent_name} in a molecular dynamics simulation workflow. "
                f"Your name is '{agent_name}'. "
                "Answer the user's question in plain text. No tool calls."
            )).strip()
        except Exception:
            pass
    # Strip any residual >>CALL: lines the model insists on emitting
    clean_lines = [l for l in final_text.splitlines()
                   if not l.strip().startswith(">>CALL:")]
    final_text = _strip_chain_of_thought("\n".join(clean_lines).strip())
    return final_text or "[Could not determine answer]"


def interactive_feedback_handler(summary: Dict[str, Any]) -> str:
    """
    Interactive command-line feedback handler for human-in-the-loop.
    
    Supports a conversational loop with LLM-powered Q&A:
      - User can ask questions about the checkpoint results
      - LLM reads files (state.jsonl, topol.top, MDP files, etc.) to answer
      - Supports: show <file>, files, list <dir>, write <file> <content>
      - Loop continues until user gives an action command
    """
    llm = summary.pop("_llm", None)
    state = summary.pop("_state", {})
    
    checkpoint_type = summary["checkpoint_type"]
    active_agent = summary.get("active_agent") or checkpoint_type
    available_agents = summary.get("available_agents") or agents_in_workflow(state)
    agent_label = AGENT_DISPLAY.get(active_agent, _AGENT_DISPLAY_NAMES.get(active_agent, active_agent.title() + " Agent"))
    error_triggered = summary.get("error_triggered", False)

    sync_hitl_active_agent_for_checkpoint(state, checkpoint_type)
    bind_agent_context(
        state,
        active_agent,
        sim_label=state.get("hitl_target_sim_label"),
        for_execution=False,
    )
    sim_label = state.get("hitl_target_sim_label")
    
    # --- Display banner ---
    print("\n" + "="*60, flush=True)
    if error_triggered:
        print(f"  {agent_label} — Human Chat  [ERROR RECOVERY]", flush=True)
    else:
        print(f"  {agent_label} — Human Chat", flush=True)
    print("="*60, flush=True)
    
    if error_triggered:
        print("\n⚠ This agent encountered errors after maximum retries.", flush=True)
        print("  Please review the issues below and provide guidance.", flush=True)
    
    print("\nCurrent State:", flush=True)
    for key, value in summary.get("current_state", {}).items():
        if key == "execution_log_tail" and isinstance(value, str) and "\n" in value:
            # Show multiline log tail in a compact block
            lines = value.strip().splitlines()
            print(f"  {key}: ({len(lines)} lines)", flush=True)
            for line in lines[-15:]:
                print(f"    {line}", flush=True)
        elif isinstance(value, list) and len(value) > 10:
            print(f"  {key}: [{len(value)} items]", flush=True)
        elif isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}...", flush=True)
        else:
            print(f"  {key}: {value}", flush=True)
    
    if summary.get("issues_found"):
        print("\n⚠ Issues Found:", flush=True)
        for issue in summary["issues_found"]:
            print(f"  - {issue}", flush=True)
    else:
        print("\n✓ No issues found.", flush=True)
    
    print("\nRecommendations:", flush=True)
    for rec in summary.get("recommendations", []):
        print(f"  - {rec}", flush=True)
    
    # --- Build persistent context for Q&A ---
    conversation_history: List[Dict[str, str]] = []
    working_dir, output_dir, agent_dirs, qa_context, agent_label = _sync_hitl_chat_context(
        state, active_agent, summary
    )
    domain_tools, domain_tool_instructions = _load_agent_domain_tools(active_agent, working_dir)
    
    print("\n" + "-"*60, flush=True)
    print(f"{agent_label} — ask questions, switch agents, or delegate tasks:", flush=True)
    agents_line = ", ".join(available_agents) if available_agents else "all"
    print(f"  Active agent: {active_agent}  |  Workflow agents: {agents_line}", flush=True)
    if sim_label:
        print(f"  Bound simulation: {sim_label}  (sim: {working_dir})", flush=True)
    print(f"  Agent output directory: {output_dir}", flush=True)
    # Indicate whether state was loaded
    _state_snap = _load_last_state_snapshot(working_dir)
    if _state_snap:
        print("  ✓ Workflow state loaded — LLM will use it to answer questions first", flush=True)
    if domain_tools:
        print(f"  Domain tools available ({len(domain_tools)}): {', '.join(domain_tools.keys())}", flush=True)
    print("  'show <filename>' — display a file", flush=True)
    print("  'files' — list generated files", flush=True)
    print("  'list <dir>' — list directory contents", flush=True)
    print("  'tools' — show detailed tool descriptions", flush=True)
    print("  'agents' — list workflow field agents and artifact directories", flush=True)
    print("  'pwd' — show simulation and agent output directories", flush=True)
    print("  Ask questions freely — the assistant can read state.jsonl, plans, and output files.", flush=True)
    print("  The assistant may ask you clarifying questions before you approve.", flush=True)
    print("  ---", flush=True)
    print("  'switch analysis' / 'switch reporter' — chat as another field agent (reloads tools)", flush=True)
    print("  'switch p23458 analysis' — bind a specific simulation (multi-sim)", flush=True)
    print("  'switch combined analysis' — cross-simulation work at project base", flush=True)
    print("  'run analysis: <task>' — execute task with Analysis agent, return here after", flush=True)
    print("  'run p23458 analysis: <task>' — run on one simulation (multi-sim)", flush=True)
    print("  'run combined analysis: <task>' — combined/cross-sim analysis at base level", flush=True)
    print("  'run: <task>' — execute with the active agent (full plan + execute)", flush=True)
    print("  ---", flush=True)
    print("  'approved' / 'continue' — proceed to next step", flush=True)
    print("  'retry' — redo this step from scratch", flush=True)
    print("  'modify: <instructions>' — redo with specific changes", flush=True)
    if error_triggered:
        print("  'recommend: <your advice>' — agent will retry with your guidance", flush=True)
    print("  'exit' / 'quit' — stop workflow", flush=True)
    print("-"*60 + "\n", flush=True)
    
    while True:
        sys.stdout.flush()
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(interrupted — exiting workflow)", flush=True)
            return "exit"
        
        if not user_input:
            continue

        # --- Agent list ---
        if user_input.lower().strip() == "agents":
            print("\nField agents in this workflow:", flush=True)
            for key in available_agents:
                bind_agent_context(state, key)
                art = artifact_summary(state, key)
                nfiles = len(art.get("files") or [])
                print(
                    f"  • {AGENT_DISPLAY.get(key, key)} ({key}) — "
                    f"{art.get('agent_directory', '?')} [{nfiles} files]",
                    flush=True,
                )
            if state.get("hitl_sim_dirs"):
                print("\n  Per-simulation directories:", flush=True)
                for lbl, path in state["hitl_sim_dirs"].items():
                    print(f"    {lbl}: {path}", flush=True)
            bind_agent_context(state, active_agent)
            domain_tools, domain_tool_instructions = _load_agent_domain_tools(
                active_agent, working_dir
            )
            print("", flush=True)
            continue

        # --- In-chat agent switch / delegated execute (before action detection) ---
        hitl_cmd = parse_hitl_command(user_input, state, default_agent=active_agent)
        if hitl_cmd and hitl_cmd.get("type") == "switch_chat":
            active_agent = hitl_cmd["agent"]
            if hitl_cmd.get("combined"):
                state["hitl_view_combined"] = True
                state.pop("hitl_target_sim_label", None)
            state["hitl_active_agent"] = active_agent
            bind_agent_context(
                state,
                active_agent,
                sim_label=hitl_cmd.get("sim_label"),
                for_execution=False,
            )
            working_dir, output_dir, agent_dirs, qa_context, agent_label = (
                _sync_hitl_chat_context(state, active_agent, summary)
            )
            sim_label = state.get("hitl_target_sim_label")
            domain_tools, domain_tool_instructions = _load_agent_domain_tools(
                active_agent, working_dir
            )
            merge_hitl_context_into_state_jsonl(state)
            sim_note = f" (sim {sim_label})" if sim_label else ""
            print(
                f"\n>> Switched to {agent_label}{sim_note}. Domain tools and artifacts reloaded.",
                flush=True,
            )
            print(f"   Simulation: {working_dir}", flush=True)
            print(f"   Output dir: {output_dir}\n", flush=True)
            print(f"Assistant: I am the {agent_label}.\n", flush=True)
            continue
        if hitl_cmd and hitl_cmd.get("type") == "execute":
            cmd = format_execute_command(
                hitl_cmd["agent"],
                hitl_cmd["task"],
                hitl_cmd.get("sim_label"),
            )
            print(f"\n>> Delegating: {cmd}\n", flush=True)
            return cmd
        
        # --- Check for action commands ---
        if _is_action(user_input):
            normalized = _normalize_action(user_input.lower().strip())
            action = normalized or user_input
            print(f"\n>> Action: {action}\n", flush=True)
            return action
        
        # --- pwd / current directory (deterministic — no LLM) ---
        if _is_pwd_question(user_input):
            print("\n" + "\n".join(format_hitl_pwd_lines(state, active_agent)) + "\n", flush=True)
            continue

        # --- Handle 'files' command ---
        if user_input.lower() == "files":
            print("\nGenerated files by agent:", flush=True)
            for agent_name, agent_dir in agent_dirs.items():
                if agent_dir and Path(agent_dir).is_dir():
                    files = sorted(f.name for f in Path(agent_dir).iterdir() if f.is_file())
                    if files:
                        print(f"  [{agent_name}] {agent_dir}/", flush=True)
                        for f in files:
                            print(f"    - {f}", flush=True)
            # Also show supervisor dir
            sup_dir = Path(working_dir) / "supervisor"
            if sup_dir.is_dir():
                files = sorted(f.name for f in sup_dir.iterdir() if f.is_file())
                if files:
                    print(f"  [supervisor] {sup_dir}/", flush=True)
                    for f in files:
                        print(f"    - {f}", flush=True)
            print("", flush=True)
            continue
        
        # --- Handle 'tools' command — show detailed domain tool descriptions ---
        if user_input.lower().strip() == "tools":
            print(f"\n  Built-in tools: read_file, list_dir, write_file, grep_file", flush=True)
            if domain_tools:
                print(f"\n  {agent_label} domain tools:", flush=True)
                for tname, tobj in domain_tools.items():
                    desc = getattr(tobj, 'description', '') or ''
                    if len(desc) > 200:
                        desc = desc[:200] + "..."
                    print(f"    • {tname}: {desc}", flush=True)
                    if hasattr(tobj, 'args_schema') and tobj.args_schema:
                        try:
                            schema = tobj.args_schema.schema()
                            props = schema.get("properties", {})
                            req = set(schema.get("required", []))
                            if props:
                                arg_strs = []
                                for aname, ainfo in props.items():
                                    marker = " (required)" if aname in req else ""
                                    arg_strs.append(f"{aname}: {ainfo.get('type', '?')}{marker}")
                                print(f"      Args: {', '.join(arg_strs)}", flush=True)
                        except Exception:
                            pass
            else:
                print(f"\n  No domain tools loaded for {agent_label}.", flush=True)
            print("", flush=True)
            continue
        
        # --- Handle 'show <file>' command ---
        if user_input.lower().startswith("show "):
            filename = user_input[5:].strip()
            resolved = _resolve_path(filename, working_dir, agent_dirs)
            if resolved:
                print(f"\n--- {resolved} ---", flush=True)
                print(_read_file_snippet(str(resolved), max_lines=80), flush=True)
                print(f"--- end ---\n", flush=True)
            else:
                print(f"\n  File '{filename}' not found in any agent directory.", flush=True)
                print(f"  Tip: use 'files' to see available files.\n", flush=True)
            continue
        
        # --- Handle 'list <dir>' command ---
        if user_input.lower().startswith("list "):
            dir_path = user_input[5:].strip()
            print(f"\n{_list_dir_safe(dir_path, working_dir)}\n", flush=True)
            continue
        
        # Re-read state.jsonl every iteration so plan data is always fresh.
        # Use .update() (not rebind) so the original LangGraph state dict reference
        # is preserved — the dict IS the state object that run_with_human_feedback
        # holds and will pass forward to the next node on retry.
        _live_state = _load_full_state_from_jsonl(working_dir)
        if _live_state:
            _hitl_preserve = {
                k: state[k] for k in (
                    "hitl_active_agent", "hitl_checkpoint_type", "hitl_target_sim_label",
                    "hitl_last_per_sim_label",
                    "hitl_sim_dirs", "hitl_agent_working_directory",
                    "hitl_agent_output_directory",
                ) if k in state
            }
            state.update(_live_state)
            state.update(_hitl_preserve)
            active_agent = state.get("hitl_active_agent") or active_agent
            bind_agent_context(
                state,
                active_agent,
                sim_label=state.get("hitl_target_sim_label"),
                for_execution=False,
            )
            working_dir, output_dir, agent_dirs, qa_context, agent_label = (
                _sync_hitl_chat_context(state, active_agent, summary)
            )
            domain_tools, domain_tool_instructions = _load_agent_domain_tools(
                active_agent, working_dir
            )

        log_path = _hitl_execution_log_path(agent_dirs, active_agent)
        conversation_log_path = _hitl_conversation_log_path(working_dir)
        is_task = _is_task_execution_request(user_input, domain_tools)

        if not (llm and getattr(llm, 'available', False)):
            print(f"\n  [LLM not available — cannot process requests in mock mode]", flush=True)
            print(f"  Try: 'show <filename>' or 'files' to inspect results manually.\n", flush=True)
            continue

        # Fast-path: if no structured plans exist at all, skip the route+replan step
        # and go straight to Q&A so plan-display questions still work
        _has_structured_plans = bool(
            (state.get("execution_plan") or {}).get("structured_plans")
        )

        if is_task and domain_tools:
            print(f"\n>> Running {agent_label} plan + execute …\n", flush=True)
            hitl_result = run_hitl_agent_task(active_agent, user_input, state, llm)
            answer = hitl_result.message
            if answer is None:
                print(f"\n  [Could not execute task]\n", flush=True)
            else:
                print(f"\nAssistant: {answer}\n", flush=True)
                conversation_history.append({"q": user_input, "a": answer[:300]})
            continue

        if _has_structured_plans:
            route_result = _llm_route_or_replan(llm, user_input, state, working_dir, agent_name=agent_label, domain_tools=domain_tools)
            if route_result is None:
                print(f"\n  [LLM unavailable — cannot process request]", flush=True)
                print(f"  Try: 'show <filename>' or 'files' to inspect results manually.\n", flush=True)
                continue
            if _looks_like_chain_of_thought(route_result) and domain_tools:
                route_result = _llm_qa_with_tools(
                    llm, user_input, qa_context,
                    conversation_history, working_dir, agent_dirs,
                    domain_tools=domain_tools,
                    domain_tool_instructions=domain_tool_instructions,
                    agent_name=agent_label,
                    execution_mode=_is_task_execution_request(user_input, domain_tools),
                    log_path=log_path,
                    output_dir=output_dir,
                    conversation_log_path=conversation_log_path,
                ) or _strip_chain_of_thought(route_result)
            else:
                _has_tool_call = (
                    _extract_tool_call_line(route_result)
                    or _TOOL_PATTERN.search(route_result)
                    or _extract_tool_intent(route_result, domain_tools)
                )
                if _has_tool_call and not route_result.startswith("Plan updated"):
                    route_result = _llm_qa_with_tools(
                        llm, user_input, qa_context,
                        conversation_history, working_dir, agent_dirs,
                        domain_tools=domain_tools,
                        domain_tool_instructions=domain_tool_instructions,
                        agent_name=agent_label,
                        log_path=log_path,
                        output_dir=output_dir,
                        conversation_log_path=conversation_log_path,
                    ) or route_result
            print(f"\nAssistant: {route_result}\n", flush=True)
            conversation_history.append({"q": user_input, "a": route_result[:200]})
        else:
            # No structured plans yet — use standard Q&A (plan display, file reads, etc.)
            answer = _llm_qa_with_tools(
                llm, user_input, qa_context,
                conversation_history, working_dir, agent_dirs,
                domain_tools=domain_tools,
                domain_tool_instructions=domain_tool_instructions,
                agent_name=agent_label,
                log_path=log_path,
                output_dir=output_dir,
                conversation_log_path=conversation_log_path,
            )
            if answer is None:
                print(f"\n  [LLM unavailable — cannot answer questions in mock mode]", flush=True)
                print(f"  Try: 'show <filename>' or 'files' to inspect results manually.\n", flush=True)
            else:
                print(f"\nAssistant: {answer}\n", flush=True)
                conversation_history.append({"q": user_input, "a": answer})

# Remove the old log_workflow_state function since we now use conversation_logger


def _filter_remodel_donor_pdbs(goal: str, paths: list) -> list:
    """
    When the goal is structure remodeling (experimental + model donor), keep only
    the experimental PDB for multi-sim / raw_pdb selection. The donor stays in the
    goal text for the preprocessing agent tools.
    """
    from src.simsetup.system_options import filter_remodel_donor_pdbs

    return filter_remodel_donor_pdbs(goal, paths)


def _extract_pdb_paths_from_goal(goal: str) -> list:
    """Extract unique input PDB paths / filenames from a natural-language goal."""
    import re as _re
    from src.utils.pdb_paths import unique_pdb_paths

    matches = _re.findall(r'[\w./\\-]+\.pdb', goal, _re.IGNORECASE)
    paths = unique_pdb_paths(matches)
    paths = [
        path
        for path in paths
        if not (
            path.lower().endswith(".pdb")
            and _is_likely_output_file_reference(path, goal)
        )
    ]
    return _filter_remodel_donor_pdbs(goal, paths)


def _entries_from_structure_targets(
    targets: List[Dict[str, Any]],
    working_dir: str,
) -> List[Dict[str, Any]]:
    """Map validated structure targets to synthetic PDB path entries."""
    entries: List[Dict[str, Any]] = []
    for t in targets or []:
        uid = t.get("uniprot_id")
        pid = t.get("pdb_id")
        label = (uid or pid or "structure").lower()
        pdb_name = f"{label}.pdb"
        pdb_path = str((Path(working_dir) / pdb_name).resolve())
        domain = t.get("domain_label")
        entries.append(
            {
                "uniprot_id": uid,
                "pdb_id": None if uid else pid,
                "protein_name": t.get("protein_name"),
                "pdb_path": pdb_path,
                "pdb_name": pdb_name,
                "structure_source": t.get("structure_source")
                or ("alphafold" if uid else "rcsb"),
                "domain_label": domain,
                "start_resid": t.get("start_resid"),
                "end_resid": t.get("end_resid"),
                "extract_domain": bool(t.get("extract_domain") or domain),
            }
        )
    return entries


def _build_pdb_list_from_uniprot_goal(
    goal: str,
    working_dir: str,
    llm: Any = None,
) -> List[Dict[str, Any]]:
    """
    Build synthetic PDB entries from UniProt IDs or RCSB PDB IDs when no local
    PDB path is provided in the goal.

    Order of operations:
      1. Deterministic regex / structured parse (primary)
      2. Optional LLM gap-fill when parse is empty but the goal clearly asks
         to download / AlphaFold a structure — every ID is shape-validated

    Each entry maps a placeholder path ({working_dir}/{id}.pdb) to download
    metadata consumed by the preprocess agent and multi-sim orchestration.
    """
    from src.preprocess.structure_request_parser import (
        extract_uniprot_ids,
        extract_pdb_ids,
        extract_protein_name,
        parse_structure_request,
        goal_requests_structure_download,
        interpret_fuzzy_structure_goal,
    )

    parsed = parse_structure_request(goal)
    uniprot_ids = extract_uniprot_ids(goal)
    pdb_ids = extract_pdb_ids(goal)

    # Shared domain / source intent from the goal applies to every UniProt ID
    # (e.g. "download … for both … extract the pseudokinase domain").
    shared_domain = (parsed or {}).get("domain_label") if parsed else None
    shared_source = (parsed or {}).get("structure_source", "auto") if parsed else "auto"
    shared_extract = bool((parsed or {}).get("extract_domain")) if parsed else False

    entries: List[Dict[str, Any]] = []
    for uid in uniprot_ids:
        req = parsed if parsed and parsed.get("uniprot_id") == uid else None
        protein_name = extract_protein_name(goal, uid)
        if req and req.get("protein_name") and str(req.get("protein_name")).upper() not in {
            "TWO", "BOTH", "ALL", "EACH", "THE", "AND",
        }:
            protein_name = req.get("protein_name")
        pdb_name = f"{uid.lower()}.pdb"
        pdb_path = str((Path(working_dir) / pdb_name).resolve())
        entries.append(
            {
                "uniprot_id": uid,
                "pdb_id": None,
                "protein_name": protein_name,
                "pdb_path": pdb_path,
                "pdb_name": pdb_name,
                "structure_source": (req or {}).get("structure_source") or shared_source or "auto",
                "domain_label": (req or {}).get("domain_label") or shared_domain,
                "start_resid": (req or {}).get("start_resid") if req else None,
                "end_resid": (req or {}).get("end_resid") if req else None,
                "extract_domain": bool(
                    (req or {}).get("extract_domain")
                    or (shared_extract and ((req or {}).get("domain_label") or shared_domain))
                ),
            }
        )

    # PDB-ID-only goals (no UniProt) — download coordinates from RCSB
    if not entries and pdb_ids:
        for pid in pdb_ids:
            pdb_name = f"{pid.lower()}.pdb"
            pdb_path = str((Path(working_dir) / pdb_name).resolve())
            entries.append(
                {
                    "uniprot_id": None,
                    "pdb_id": pid,
                    "protein_name": None,
                    "pdb_path": pdb_path,
                    "pdb_name": pdb_name,
                    "structure_source": "rcsb",
                    "domain_label": None,
                    "start_resid": None,
                    "end_resid": None,
                    "extract_domain": False,
                }
            )

    # LLM gap-fill: only when deterministic parse found nothing, the goal
    # asks for a download, and an LLM client is available.
    if not entries and llm is not None and goal_requests_structure_download(goal):
        print(
            "\n  Deterministic structure parse found no IDs — "
            "asking LLM to interpret fuzzy goal (validated before fetch)…",
            flush=True,
        )
        llm_result = interpret_fuzzy_structure_goal(goal, llm)
        targets = llm_result.get("targets") or []
        if targets:
            entries = _entries_from_structure_targets(targets, working_dir)
            labels = ", ".join(
                t.get("uniprot_id") or t.get("pdb_id") or "?" for t in targets
            )
            print(f"  LLM proposed validated structure target(s): {labels}", flush=True)
        else:
            err = llm_result.get("error") or "no validated targets"
            print(f"  LLM gap-fill did not yield usable IDs ({err})", flush=True)

    return entries


def _prefetch_uniprot_structures(
    entries: List[Dict[str, Any]],
    goal: str,
) -> Tuple[List[str], List[str]]:
    """
    Ensure shared structures exist in the base working directory.

    Existing (user-provided) files are reused as-is; only missing structures
    are fetched from the database (UniProt→AlphaFold/RCSB, or direct PDB ID).

    Returns ``(reused, downloaded)`` lists of PDB paths.
    """
    from src.preprocess.structure_acquisition import acquire_structure_from_request
    from src.preprocess.structure_downloader import download_structure, download_pdb_by_id

    reused: List[str] = []
    downloaded: List[str] = []
    for entry in entries:
        pdb_path = entry["pdb_path"]
        if Path(pdb_path).exists():
            reused.append(pdb_path)
            continue

        out_dir = str(Path(pdb_path).parent)
        source = entry.get("structure_source", "auto")
        uid = entry.get("uniprot_id")
        pdb_id = entry.get("pdb_id")

        if pdb_id and not uid:
            result = download_pdb_by_id.func(pdb_id=pdb_id, output_file=pdb_path)
            if result.get("success"):
                downloaded.append(pdb_path)
            continue

        if entry.get("extract_domain") and entry.get("start_resid") and entry.get("end_resid"):
            result = acquire_structure_from_request(goal, out_dir, source=source)
            acquired = result.get("pdb_file") or result.get("output_file")
            if result.get("success") and acquired:
                downloaded.append(acquired)
                entry["pdb_path"] = acquired
                entry["pdb_name"] = Path(acquired).name
            continue

        result = download_structure.func(
            uniprot_id=uid,
            output_file=pdb_path,
            source=source,
        )
        if result.get("success"):
            downloaded.append(pdb_path)

    return reused, downloaded


def _structure_needs_download(
    entry: Dict[str, Any],
    goal: str,
    parsed: Dict[str, Any],
) -> bool:
    """True only when the structure file is missing and the user did not supply a local PDB."""
    pdb_path = entry.get("pdb_path") or ""
    if pdb_path and Path(pdb_path).is_file():
        return False
    if parsed.get("needs_download") is False:
        return False
    goal_lower = (goal or "").lower()
    if re.search(r"\b(?:download|fetch|retrieve)\b.*\b(?:structure|pdb|alphafold|uniprot)\b", goal_lower):
        return True
    if re.search(r"\b(?:structure|pdb).*\b(?:download|fetch|retrieve)\b", goal_lower):
        return True
    if parsed.get("needs_download") is True:
        return True
    return not bool(pdb_path and Path(pdb_path).exists())


def _build_structure_request_config(entry: Dict[str, Any], goal: str) -> Dict[str, Any]:
    """Normalize structure request metadata passed into workflow state."""
    from src.preprocess.structure_request_parser import (
        parse_structure_request,
        resolve_domain_residue_range,
    )

    parsed = parse_structure_request(goal) or {}
    uid = entry.get("uniprot_id") or parsed.get("uniprot_id")
    domain_label = entry.get("domain_label") or parsed.get("domain_label")

    start_resid = entry.get("start_resid") or parsed.get("start_resid")
    end_resid = entry.get("end_resid") or parsed.get("end_resid")

    # Re-resolve from UniProt when domain named but no explicit simulation range
    domain_lookup = {}
    if domain_label and uid and not (start_resid and end_resid):
        domain_lookup = resolve_domain_residue_range(uid, domain_label)
        if domain_lookup.get("domain_lookup_success"):
            start_resid = domain_lookup.get("start_resid")
            end_resid = domain_lookup.get("end_resid")

    config = {
        "uniprot_id": uid,
        "pdb_id": entry.get("pdb_id") or parsed.get("pdb_id"),
        "protein_name": entry.get("protein_name") or parsed.get("protein_name"),
        "structure_source": entry.get("structure_source") or parsed.get("structure_source", "auto"),
        "extract_domain": bool(domain_label and start_resid and end_resid),
        "start_resid": start_resid,
        "end_resid": end_resid,
        "domain_label": domain_label,
        "needs_download": _structure_needs_download(entry, goal, parsed),
    }
    if domain_lookup:
        config.update(
            {k: v for k, v in domain_lookup.items() if k.startswith("domain_lookup")}
        )
    elif parsed:
        config.update(
            {k: v for k, v in parsed.items() if k.startswith("domain_lookup")}
        )
    return config


def _extract_production_ns_from_goal(goal: str) -> float | None:
    """Extract requested production duration in nanoseconds from goal text."""
    import re as _re
    patterns = [
        _re.compile(
            r'(?:simulation\s*(?:time|length|duration|ns)|production(?:\s*run)?|run\s*(?:time|length)?)'
            r'\s*[:=of]?\s*(\d+(?:\.\d+)?)\s*(?:ns|nanoseconds?)?\b',
            _re.IGNORECASE,
        ),
        _re.compile(r'\b(\d+(?:\.\d+)?)\s*(?:ns|nanoseconds?)\b', _re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(goal or "")
        if match:
            raw = next((g for g in match.groups() if g is not None), None)
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val > 0:
                return val
    return None


# Filename suffixes that usually denote generated outputs, not workflow inputs.
_OUTPUT_PDB_SUFFIXES = (
    "_remodeled", "_aligned", "_cleaned", "_fixed", "_merged", "_separated",
    "_trimmed", "_domain", "_extracted", "_processed",
)


def _is_likely_output_file_reference(filename: str, goal: str) -> bool:
    """Return True when a .pdb in --goal is probably an output, not an input."""
    stem = Path(filename).stem.lower()
    if any(stem.endswith(suffix) for suffix in _OUTPUT_PDB_SUFFIXES):
        return True
    # e.g. "Output chain_a_6VC0_remodeled.pdb" or "write to foo.pdb"
    pattern = re.compile(
        r"(?:output|write|save|generate|create|produce)\s+"
        + re.escape(filename),
        re.IGNORECASE,
    )
    return bool(pattern.search(goal))


def _goal_file_exists(
    candidate: str,
    working_dir: Path,
    sim_dirs: Optional[List[str]] = None,
) -> bool:
    """Check whether a goal-referenced file exists in common search locations."""
    path = Path(candidate)
    if path.is_absolute():
        return path.exists()

    search_roots = [Path.cwd(), working_dir]
    rel_parts = path.parts
    candidates = [
        Path.cwd() / path,
        working_dir / path,
    ]
    # Agent subdirs (preprocess/, hpc/, etc.)
    for root in search_roots:
        candidates.append(root / path)
        for sub in ("preprocess", "simsetup", "hpc", "analysis"):
            candidates.append(root / sub / path)
    # If goal uses preprocess/foo.pdb, also try foo.pdb at working_dir root
    if len(rel_parts) >= 2 and rel_parts[0] in {"preprocess", "simsetup", "hpc"}:
        candidates.append(working_dir / Path(*rel_parts[1:]))

    # Multi-sim continuation / analysis: bare names like md.tpr live under
    # each --sim-dirs entry (usually {label}/hpc/md.tpr).
    for raw in sim_dirs or []:
        sim_root = Path(raw).expanduser().resolve()
        candidates.append(sim_root / path)
        candidates.append(sim_root / "hpc" / path)
        if len(rel_parts) >= 2 and rel_parts[0] == "hpc":
            candidates.append(sim_root / Path(*rel_parts[1:]))

    return any(p.exists() for p in candidates)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SimAgent: LLM-powered molecular dynamics workflow"
    )
    parser.add_argument("--goal", required=True, 
                       help="Natural language description of simulation goal")
    parser.add_argument("--pdb-list", default=None, nargs='+', metavar="PDB",
                       help=("Multiple PDB files for multi-simulation mode. "
                             "Each PDB gets its own pipeline in a separate directory. "
                             "Example: --pdb-list 1abc.pdb 2def.pdb 3ghi.pdb"))
    parser.add_argument("--sim-dirs", default=None, nargs='+', metavar="DIR",
                       help=("Per-simulation directories for multi-sim analysis of "
                             "already-completed simulations. Each directory must "
                             "contain an hpc/ sub-folder with trajectory data. "
                             "The directory basename is used as the simulation label. "
                             "Example: --sim-dirs pseudokin/p17612 pseudokin/p24941"))
    parser.add_argument("--subtask", default=None, nargs='+',
                       choices=["preprocess", "simsetup", "hpcjob", "analysis", "reporter"],
                       metavar="AGENT",
                       help=("One or more field agents to run, e.g. --subtask analysis reporter. "
                             "Valid values: preprocess simsetup hpcjob analysis reporter. "
                             "Omit to run the full pipeline."))
    parser.add_argument("--no-llm", action="store_true",
                       help=("Disable LLM planning and use heuristic / registry-based "
                             "plans only. Without this flag, a reachable Ollama server "
                             "or --llm-api-key is required."))
    parser.add_argument("--llm-model", default="gpt-oss:20b",
                       help="LLM model to use")
    parser.add_argument("--llm-base-url", default="http://localhost:11434",
                       help="LLM API base URL")
    parser.add_argument("--llm-api-key", default=None,
                       help="Paid LLM API key (or set LLM_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--llm-token-budget", type=int, default=None,
                       help="Max total LLM tokens for this run (enforced when set)")
    parser.add_argument("--llm-provider", default="auto",
                       choices=["auto", "ollama", "openai"],
                       help="LLM backend when using a paid API key")
    parser.add_argument("--HITL", dest="hitl", default=None,
                       choices=["error", "all"],
                       help=("Human-in-the-loop: 'error' pauses only on failures; "
                             "'all' pauses at every workflow checkpoint. "
                             "Default: fully automatic (no HITL), or campaign.yaml hitl."))
    parser.add_argument(
        "--campaign-yaml",
        default=None,
        metavar="PATH",
        help=(
            "Campaign settings YAML (conservation_metric, pocket_cutoff_A, "
            "retrieval_k, gold_columns, hitl, …). Default: "
            "{working-dir}/campaign.yaml if present, else "
            "agentic/campaign/campaign.yaml. Env AGENTIC_* overrides keys."
        ),
    )
    # Deprecated — kept for backward compatibility with older scripts/docs
    parser.add_argument("--use-llm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-human-loop", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force-field", default="amber99sb-ildn",
                       help="Force field to use")
    parser.add_argument("--water-model", default="tip3p", 
                       help="Water model to use")
    parser.add_argument("--working-dir", default=".",
                       help=("Campaign base directory. Each simulation writes under "
                             "{working-dir}/{label}/ (preprocess, simsetup, hpc, …). "
                             "Campaign provenance stays at the base."))
    parser.add_argument("--max-concurrent", type=int, default=4,
                       help="Maximum concurrent simulations in multi-sim mode (legacy)")
    parser.add_argument("--allowed-hpc-jobs", type=int, default=None,
                       help=(
                           "Max concurrent SLURM jobs in cross-sim HPC pool mode. "
                           "Default: auto from CPU/memory when --parallel-workers is auto."
                       ))
    parser.add_argument(
        "--rep-num",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of independent MD production replicates per system label "
            "(default: 1, or soft-parsed from goal like '3 replicates'). "
            "Prep/simsetup run once; HPC writes hpc/rep01..repNN "
            "and analysis fans out into analysis/rep01..repNN + analysis/avg/."
        ),
    )
    parser.add_argument("--parallel-workers", default="4",
                       help=(
                           "Max parallel local workers for multi-sim prep and "
                           "analysis/reporter: integer (default 4), 'auto' (CPU/mem), "
                           "or 1=sequential"
                       ))
    parser.add_argument("--parallel-mem-gb", type=float, default=None,
                       help="Estimated GiB RAM per parallel worker (default: phase-specific)")
    parser.add_argument("--parallel-cpus", type=float, default=None,
                       help="Estimated CPU cores per parallel worker (default: phase-specific)")
    parser.add_argument("--llm-concurrency", default="auto",
                       help=(
                           "Max concurrent LLM requests for parallel prep/analysis workers. "
                           "'auto' (default) matches OLLAMA_NUM_PARALLEL (4). "
                           "Caps local workers so fewer sims queue on the shared Ollama server."
                       ))
    parser.add_argument(
        "--sim-max-attempts",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max attempts per simulation for parallel prep/analysis before marking "
            "that sim failed and continuing (default: 3, or AGENTIC_SIM_MAX_ATTEMPTS)."
        ),
    )
    parser.add_argument("--hpc-check-interval", default="2h",
                       help=(
                           "SLURM poll interval during HPC pool wait "
                           "(e.g. 2h, 120m, 7200). Default: 2h"
                       ))
    parser.add_argument("--resume", action="store_true",
                       help=(
                           "Resume a multi-sim run by re-running only the simulations that "
                           "previously failed or were not completed. Already-succeeded simulations "
                           "are skipped. The --working-dir must point to the same directory as "
                           "the original run so prior state can be loaded."
                       ))
    parser.add_argument("--retry-labels", default=None, nargs="+", metavar="LABEL",
                       help=(
                           "Force-retry specific simulation labels even if they previously "
                           "succeeded. Useful when a job submitted but ran with wrong parameters. "
                           "Example: --retry-labels p21860_ATP_MG q8nb16_ATP_MG"
                       ))
    parser.add_argument("--combined-only", action="store_true",
                       help=(
                           "Multi-sim only: skip per-simulation analysis and run ONLY the "
                           "combined cross-simulation analysis + report using existing per-sim "
                           "outputs on disk. Use after per-sim analysis is already complete to "
                           "regenerate combined overlays/DCCM/report without re-analysing each "
                           "simulation. Requires the same --working-dir as the original run."
                       ))
    parser.add_argument(
        "--reuse-hpc",
        action="store_true",
        help=(
            "Full-pipeline demo on frozen trajectories: still run preprocess + simsetup "
            "+ HPC staging (copy/SLURM script) + analysis + reporter, but never call "
            "sbatch. Existing hpc/md.tpr and hpc/mdWrap.xtc are preserved (no md.log "
            "required)."
        ),
    )
    parser.add_argument(
        "--skip-hpc-submit",
        action="store_true",
        help=(
            "Same skip-sbatch behavior as --reuse-hpc: run preprocess + simsetup + "
            "staging + analysis normally, but never call sbatch. Prefer this flag for "
            "seeded-trajectory campaigns. Existing hpc[/repXX]/md.tpr + mdWrap.xtc "
            "are required (no md.log)."
        ),
    )

    args = parser.parse_args(argv)

    if args.no_human_loop and args.hitl:
        parser.error("Cannot use both --no-human-loop and --HITL")
    if args.no_human_loop:
        print(
            "Note: --no-human-loop is deprecated (non-HITL is the default).",
            flush=True,
        )
    if args.use_llm:
        print(
            "Note: --use-llm is deprecated (LLM is enabled by default). "
            "Use --no-llm to disable.",
            flush=True,
        )

    from agentic.campaign.config import load_campaign_settings
    from agentic.hitl_config import resolve_hitl_from_cli

    campaign_settings = load_campaign_settings(
        path=getattr(args, "campaign_yaml", None),
        working_dir=args.working_dir,
    )
    hitl_cli = args.hitl or campaign_settings.hitl
    human_in_loop, hitl_mode = resolve_hitl_from_cli(hitl_cli)
    use_llm = not args.no_llm

    # -------------------------------------------------------------------------
    # Early path validation — fail fast with a clear error before any agent runs
    # -------------------------------------------------------------------------
    _validation_errors: List[str] = []

    # 1. Validate --working-dir (must exist if user explicitly specified it)
    if args.working_dir not in (".", "working_dir"):
        _wd_path = Path(args.working_dir)
        if not _wd_path.exists():
            _validation_errors.append(
                f"--working-dir '{args.working_dir}' does not exist. "
                f"Please create it first or check for a typo."
            )
        elif not _wd_path.is_dir():
            _validation_errors.append(
                f"--working-dir '{args.working_dir}' is not a directory."
            )

    # 2. Validate PDB / file paths mentioned in --goal
    import re as _re_val
    _path_candidates = _re_val.findall(
        r'(?:^|\s)([^\s"\']+\.(?:pdb|gro|top|xtc|trr|tpr|itp|mdp))',
        args.goal, _re_val.IGNORECASE
    )
    _wd_for_check = (
        Path(args.working_dir)
        if args.working_dir not in (".", "working_dir")
        else Path.cwd()
    )
    _sim_dirs_for_check = list(getattr(args, "sim_dirs", None) or [])
    for _cand in _path_candidates:
        if _cand.lower().endswith(".pdb") and _is_likely_output_file_reference(
            _cand, args.goal
        ):
            continue
        if not _goal_file_exists(
            _cand, _wd_for_check, sim_dirs=_sim_dirs_for_check
        ):
            _validation_errors.append(
                f"File referenced in --goal not found: '{_cand}' "
                f"(checked cwd, '{_wd_for_check}', agent subdirs"
                + (", and --sim-dirs/*/hpc)" if _sim_dirs_for_check else ")")
            )

    if _validation_errors:
        print("\n" + "=" * 60, flush=True)
        print("  INPUT VALIDATION FAILED — workflow not started", flush=True)
        print("=" * 60, flush=True)
        for _err in _validation_errors:
            print(f"  ERROR: {_err}", flush=True)
        print("=" * 60 + "\n", flush=True)
        return 1
    # -------------------------------------------------------------------------
    if getattr(args, "llm_api_key", None):
        os.environ["LLM_API_KEY"] = args.llm_api_key.strip()
    # Configuration
    config = {
        "force_field": args.force_field,
        "water_model": args.water_model,
        "human_in_loop": human_in_loop,
        "hitl_mode": hitl_mode,
        "use_llm": use_llm,
        "working_directory": args.working_dir,
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url if use_llm else None,
        "llm_token_budget": getattr(args, "llm_token_budget", None),
        "llm_provider": getattr(args, "llm_provider", "auto") or "auto",
        "resume_failed_only": getattr(args, "resume", False),
        # One-shot consumed by parallel pool init — reopen failed sims on --resume.
        "requeue_failed_sims": getattr(args, "resume", False),
        "retry_labels": list(getattr(args, "retry_labels", None) or []),
        "combined_only": getattr(args, "combined_only", False),
        "reuse_hpc": bool(getattr(args, "reuse_hpc", False)),
        "skip_hpc_submit": bool(getattr(args, "skip_hpc_submit", False)),
        "allowed_hpc_jobs": getattr(args, "allowed_hpc_jobs", None),
        "_allowed_hpc_jobs_explicit": getattr(args, "allowed_hpc_jobs", None) is not None,
        "max_concurrent": getattr(args, "max_concurrent", 4),
        "hpc_check_interval": getattr(args, "hpc_check_interval", "2h"),
        "rep_num": 1,
        "replicate_base_seed": 12345,
        "parallel_workers": getattr(args, "parallel_workers", "auto"),
        "parallel_mem_gb_per_job": getattr(args, "parallel_mem_gb", None),
        "parallel_cpus_per_job": getattr(args, "parallel_cpus", None),
        "llm_concurrency": getattr(args, "llm_concurrency", "auto"),
        "sim_max_attempts": getattr(args, "sim_max_attempts", None),
        "campaign_yaml": str(campaign_settings.source_path or ""),
        "campaign_settings": campaign_settings.to_dict(),
    }

    if config["skip_hpc_submit"] and not config["reuse_hpc"]:
        # Same runtime path as reuse-hpc (traj-ready without md.log; no sbatch).
        config["reuse_hpc"] = True
    if config["reuse_hpc"] or config["skip_hpc_submit"]:
        import os as _os_reuse

        _os_reuse.environ["AGENTIC_REUSE_HPC"] = "1"
        if config.get("skip_hpc_submit"):
            _os_reuse.environ["AGENTIC_SKIP_HPC_SUBMIT"] = "1"
        flag = "--skip-hpc-submit" if config.get("skip_hpc_submit") else "--reuse-hpc"
        print(
            f"Note: {flag} active — full pipeline runs (preprocess/simsetup/HPC "
            "staging/analysis/reporter); existing md.tpr+mdWrap.xtc are kept; sbatch "
            "is blocked.",
            flush=True,
        )
    
    # Pass subtask type directly in config
    if args.subtask:
        single_agent_map = {
            "preprocess": "preprocess_only",
            "simsetup": "setup_only",
            "hpcjob": "hpc_only",
            "analysis": "analysis_only",
            "reporter": "reporter_only"
        }
        if len(args.subtask) == 1:
            # Single agent: use existing specific subtask_type
            config["subtask_type"] = single_agent_map[args.subtask[0]]
        else:
            # Multiple agents: multi_agent mode with ordered agent list
            config["subtask_type"] = "multi_agent"
            config["agent_list"] = args.subtask
            config["pipeline_agent_list"] = list(args.subtask)
    
    goal = args.goal
    from src.analysis.replicate_paths import normalize_rep_num, parse_rep_num_from_text

    if getattr(args, "rep_num", None) is not None:
        config["rep_num"] = normalize_rep_num(args.rep_num)
    else:
        config["rep_num"] = parse_rep_num_from_text(goal, default=1)
    if config["rep_num"] > 1:
        print(
            f"Note: rep_num={config['rep_num']} — each label gets "
            f"{config['rep_num']} nested production replicates (hpc/repXX, analysis/repXX + avg/).",
            flush=True,
        )

    production_ns = _extract_production_ns_from_goal(goal)
    if production_ns is not None:
        config["production_ns"] = production_ns

    from src.simsetup.minimization_options import parse_extended_minimization_from_text
    from src.simsetup.system_options import apply_goal_simsetup_config

    if parse_extended_minimization_from_text(goal):
        config["extended_minimization"] = True
    apply_goal_simsetup_config(goal, config)

    from src.hpc.time_options import apply_goal_hpc_time_config

    apply_goal_hpc_time_config(goal, config)
    
    # Resolve working directory (same logic as workflow._initialize_state)
    working_dir = args.working_dir
    if working_dir == ".":
        working_dir = "working_dir"
    if not Path(working_dir).is_absolute():
        working_dir = str(Path.cwd() / working_dir)
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    llm_client = LLMClient(
        model=args.llm_model,
        base_url=args.llm_base_url if use_llm else None,
        api_key=getattr(args, "llm_api_key", None),
        token_budget=getattr(args, "llm_token_budget", None),
        working_dir=working_dir if use_llm else None,
        provider=getattr(args, "llm_provider", "auto") or "auto",
    )
    if use_llm:
        try:
            llm_client.require_ready()
        except RuntimeError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
            sys.exit(2)
        _budget = llm_client.usage.limit
        _usage_path = f"{working_dir}/llm_usage.json"
        if llm_client.usage.billing_enabled and _budget:
            print(
                f"\n  LLM billing: API key set, token budget {_budget:,} "
                f"(usage tracked in {_usage_path})",
                flush=True,
            )
        elif llm_client.usage.billing_enabled:
            print(
                f"\n  LLM billing: API key set, usage tracked in "
                f"{_usage_path} (no budget cap — set --llm-token-budget to enforce)",
                flush=True,
            )
        elif _budget:
            print(
                f"\n  LLM usage: tracking + budget {_budget:,} "
                f"(local/no API key) → {_usage_path}",
                flush=True,
            )
        else:
            print(
                f"\n  LLM usage: tracking enabled (no budget) → {_usage_path}",
                flush=True,
            )
    else:
        print(
            "\n  LLM disabled (--no-llm): using heuristic / registry-based plans.",
            flush=True,
        )    
    # Set up logging inside working_dir (not at project root)
    log_path = str(Path(working_dir) / "agent_conversation.log")
    set_log_file(log_path)

    # ------------------------------------------------------------------
    # Resolve structures (always {base}/{label}/ multi-sim tree)
    # ------------------------------------------------------------------
    pdb_list = getattr(args, 'pdb_list', None) or []

    # --sim-dirs: convert per-sim directories into synthetic pdb_list entries
    # so the existing multi-sim planner logic works unchanged.
    # Each dir's basename becomes the simulation label.  E.g.:
    #   --sim-dirs pseudokin/p17612 pseudokin/p24941
    # produces a pdb_list of ["/abs/pseudokin/p17612/p17612.pdb", ...].
    # The planner uses Path(pdb).stem as the label → "p17612",
    # and builds sim_working_dir = {base_working_dir}/p17612/ which
    # matches the existing directory layout.
    sim_dirs_arg = getattr(args, 'sim_dirs', None) or []
    if sim_dirs_arg and not pdb_list:
        resolved_sim_dirs = [Path(sd).resolve() for sd in sim_dirs_arg]
        # Synthetic PDB entries: {sim_dir}/{label}.pdb (file need not exist)
        pdb_list = [str(d / f"{d.name}.pdb") for d in resolved_sim_dirs]
        print(
            f"\n--sim-dirs: activating campaign for {len(pdb_list)} directories",
            flush=True,
        )

    # Auto-detect PDbs from goal if not already provided via --pdb-list or --sim-dirs
    structure_request_entries: List[Dict[str, Any]] = []
    if not pdb_list:
        _goal_pdbs = _extract_pdb_paths_from_goal(goal)
        if len(_goal_pdbs) > 0:
            pdb_list = _goal_pdbs
            print(
                f"\n  Auto-detected {len(pdb_list)} PDB(s) from goal: "
                f"{', '.join(Path(p).name for p in pdb_list)}",
                flush=True,
            )
        else:
            # No .pdb in goal — try UniProt-based structure acquisition
            # (deterministic parse first; LLM gap-fill only if needed)
            structure_request_entries = _build_pdb_list_from_uniprot_goal(
                goal,
                working_dir,
                llm=llm_client if use_llm else None,
            )
            if structure_request_entries:
                pdb_list = [e["pdb_path"] for e in structure_request_entries]
                labels = ", ".join(
                    e.get("uniprot_id") or e.get("pdb_id") or Path(e["pdb_path"]).stem
                    for e in structure_request_entries
                )
                _n_existing = sum(
                    1 for e in structure_request_entries if Path(e["pdb_path"]).exists()
                )
                if _n_existing == len(structure_request_entries):
                    print(
                        f"\n  Using {_n_existing} existing local structure(s): {labels}",
                        flush=True,
                    )
                else:
                    print(
                        f"\n  Resolving structure(s): {labels} "
                        f"({_n_existing} present locally, "
                        f"{len(structure_request_entries) - _n_existing} to fetch)",
                        flush=True,
                    )
                if len(structure_request_entries) == 1 and structure_request_entries[0].get(
                    "uniprot_id"
                ):
                    print(
                        "  Note: one UniProt ID may expand into component cases "
                        "(e.g. protein-only vs protein+ATP+MG) if requested.",
                        flush=True,
                    )

    if not pdb_list:
        print(
            "\nERROR: No PDB files, UniProt accessions, or PDB IDs found!",
            file=sys.stderr, flush=True
        )
        print(
            "   Please provide structures via:",
            file=sys.stderr, flush=True
        )
        print(
            "   - --pdb-list file1.pdb file2.pdb ...",
            file=sys.stderr, flush=True
        )
        print(
            "   - --sim-dirs dir1 dir2 ...",
            file=sys.stderr, flush=True
        )
        print(
            "   - Mention .pdb files in your --goal",
            file=sys.stderr, flush=True
        )
        print(
            "   - Or request download by UniProt ID in --goal "
            "(e.g. 'UniProt P21860, download from AlphaFold')",
            file=sys.stderr, flush=True
        )
        print(
            "   - Or request download by PDB ID in --goal "
            "(e.g. 'Download PDB 6VC0 and run MD')",
            file=sys.stderr, flush=True
        )
        return 1

    # Ensure UniProt structures exist at base working_dir (shared by cases).
    # User-provided files are reused; only missing ones are fetched.
    if structure_request_entries:
        n_missing = sum(
            1 for e in structure_request_entries if not Path(e["pdb_path"]).exists()
        )
        if n_missing:
            print(
                f"  Fetching {n_missing} missing structure(s) from database...",
                flush=True,
            )
        reused, downloaded = _prefetch_uniprot_structures(
            structure_request_entries, goal
        )
        if reused:
            print(
                f"  Reusing {len(reused)} existing local structure file(s)",
                flush=True,
            )
        if downloaded:
            print(
                f"  Downloaded {len(downloaded)} structure file(s)",
                flush=True,
            )
        if not reused and not downloaded:
            print(
                "  Warning: structure acquisition deferred to preprocessing agent",
                flush=True,
            )
        config["structure_requests"] = {
            Path(entry["pdb_path"]).stem.lower(): _build_structure_request_config(entry, goal)
            for entry in structure_request_entries
        }
        config["structure_request"] = _build_structure_request_config(
            structure_request_entries[0], goal
        )

    num_sims = len(pdb_list)
    print(f"\nCampaign mode: {num_sims} source structure(s) → {{base}}/{{label}}/", flush=True)
    if sim_dirs_arg:
        print("  Source: --sim-dirs", flush=True)
    elif getattr(args, "pdb_list", None):
        print("  Source: --pdb-list", flush=True)
    else:
        print("  Source: goal / UniProt resolution", flush=True)

    print("  Structures:", flush=True)
    for i, pdb in enumerate(pdb_list, 1):
        print(f"    {i}. {Path(pdb).name}", flush=True)

    log_user_prompt(goal, config)

    # Always use the multi-sim tree (even for N=1). Combined analysis is skipped
    # later when len(sim_prompts) <= 1.
    config["is_multi_simulation"] = True
    config["multi_sim_base_dir"] = working_dir
    resolved_pdbs = []
    for p in pdb_list:
        _p = Path(p)
        if _p.is_absolute() and _p.exists():
            resolved_pdbs.append(str(_p))
        elif (Path(working_dir) / p).exists():
            resolved_pdbs.append(str((Path(working_dir) / p).resolve()))
        elif _p.exists():
            resolved_pdbs.append(str(_p.resolve()))
        else:
            resolved_pdbs.append(str((Path(working_dir) / p).resolve()))
    from src.utils.pdb_paths import unique_pdb_paths

    config["pdb_list"] = unique_pdb_paths(resolved_pdbs)

    _goal_pdb_names = {
        Path(p).name.lower() for p in _extract_pdb_paths_from_goal(goal)
    }
    _resolved_names = {Path(p).name.lower() for p in config["pdb_list"]}
    _missing_pdbs = sorted(_goal_pdb_names - _resolved_names)
    if _missing_pdbs:
        print(
            f"\n  Warning: goal mentions {len(_missing_pdbs)} PDB(s) not found under "
            f"{working_dir}: {', '.join(_missing_pdbs)}",
            flush=True,
        )
        print(
            "  Those systems will be omitted from the master plan until the files exist.",
            flush=True,
        )

    feedback_handler = None
    if config["human_in_loop"]:
        _hitl = config.get("hitl_mode") or "all"
        if _hitl == "error":
            print("\n  HITL mode: pause on errors only (--HITL error)", flush=True)
        else:
            print("\n  HITL mode: all checkpoints (--HITL all)", flush=True)
        feedback_handler = interactive_feedback_handler

    workflow = MDWorkflow(llm_client)

    try:
        if config["human_in_loop"] and feedback_handler:
            final_state = workflow.run_with_human_feedback(goal, feedback_handler, config)
        else:
            final_state = workflow.run(goal, config)

        run_summary = build_run_summary(
            final_state,
            working_dir=working_dir,
            goal=goal,
            pdb_list=pdb_list,
            config=config,
        )
        summary_paths = write_run_summary(working_dir, run_summary)
        run_summary["summary_files"] = summary_paths
        print(f"\n{format_run_summary_terminal(run_summary)}", flush=True)

        counts = run_summary.get("counts", {})
        n_fail = counts.get("failed", 0)
        return 0 if n_fail == 0 else 1

    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        return 130
    except TokenBudgetExceeded as e:
        print(f"\nLLM token budget exceeded: {e}", flush=True)
        run_summary = build_run_summary(
            {"errors": [str(e)], "warnings": []},
            working_dir=working_dir,
            goal=goal,
            pdb_list=pdb_list,
            config=config,
        )
        write_run_summary(working_dir, run_summary)
        print(f"\n{format_run_summary_terminal(run_summary)}", flush=True)
        return 1
    except Exception as e:
        print(f"\nWorkflow failed: {e}")
        logging.exception("Workflow execution failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
