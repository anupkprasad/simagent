"""Resolve ligand residue names / MDAnalysis selections from workflow state.

Avoid silent ATP defaults when the campaign did not declare a ligand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _resnames_from_campaign_spec(state: Optional[Dict[str, Any]]) -> List[str]:
    """Ligand names compiled into the campaign spec (goal inference or recipe)."""
    try:
        from agentic.campaign.spec import spec_from_state
    except Exception:
        return []
    spec = spec_from_state(state)
    if spec is None:
        return []
    names: List[str] = []
    raw = (spec.settings or {}).get("ligand_resnames")
    if isinstance(raw, str) and raw.strip():
        names.append(raw.strip())
    elif isinstance(raw, (list, tuple)):
        names.extend(str(x).strip() for x in raw if str(x).strip())
    if names:
        return names
    recipes = (
        getattr(spec, "analysis_recipe", None),
        getattr(spec, "pre_combined_recipe", None),
    )
    for recipe in recipes:
        for step in getattr(recipe, "steps", None) or []:
            params = getattr(step, "tool_params", None) or {}
            sel = str(params.get("ligand_selection") or "").strip()
            if sel.lower().startswith("resname "):
                names.extend(tok for tok in sel.split()[1:] if tok)
    return names


def ligand_resnames_from_state(state: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return declared ligand residue names (may be empty for apo / protein-only)."""
    state = state or {}
    names: List[str] = []
    raw = state.get("ligand_resnames")
    if isinstance(raw, (list, tuple)):
        names.extend(str(x).strip() for x in raw if str(x).strip())
    elif isinstance(raw, str) and raw.strip():
        names.append(raw.strip())
    # Component / case resolver often stores under component_selection
    comp = state.get("component_selection") or state.get("components") or {}
    if isinstance(comp, dict):
        for key in ("ligand_resnames", "ligands"):
            val = comp.get(key)
            if isinstance(val, (list, tuple)):
                names.extend(str(x).strip() for x in val if str(x).strip())
            elif isinstance(val, str) and val.strip():
                names.append(val.strip())
    # Deduplicate preserving order
    if not names:
        names.extend(_resnames_from_campaign_spec(state))
    seen = set()
    out: List[str] = []
    for n in names:
        key = n.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def ligand_selection_from_state(
    state: Optional[Dict[str, Any]] = None,
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    """MDAnalysis selection string, or *default* / None when no ligand declared."""
    names = ligand_resnames_from_state(state)
    if not names:
        return default
    if len(names) == 1:
        return f"resname {names[0]}"
    joined = " ".join(names)
    return f"resname {joined}"


def apply_ligand_selection(
    tool_params: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    *,
    key: str = "ligand_selection",
) -> Dict[str, Any]:
    """Set or clear ``ligand_selection`` from state; never invent ATP."""
    params = dict(tool_params or {})
    sel = ligand_selection_from_state(state, default=None)
    if sel:
        params[key] = sel
    elif key in params and _is_silent_atp_default(params.get(key)):
        # Strip silent ATP default when campaign has no ligand
        params.pop(key, None)
    return params


def _is_silent_atp_default(value: Any) -> bool:
    raw = str(value or "").strip().upper().replace("  ", " ")
    return raw in {"RESNAME ATP", "ATP", "RESNAME ATP AND NOT NAME H*"}


def resolve_ligand_selection(
    explicit: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    *,
    allow_atp_default: bool = False,
) -> Optional[str]:
    """Resolve ligand MDAnalysis selection without inventing ATP for apo runs.

    Priority: non-empty explicit (unless it is silent ATP with no declared ligand)
    → state ligand_resnames → None (caller should skip pocket/ligand tools).
    """
    state_sel = ligand_selection_from_state(state, default=None)
    exp = (explicit or "").strip()
    if exp and not _is_silent_atp_default(exp):
        return exp
    if state_sel:
        return state_sel
    if exp and _is_silent_atp_default(exp) and allow_atp_default:
        return exp
    # Explicit ATP with no declared ligand → treat as unset (apo-safe)
    return None
