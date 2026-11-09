"""Compile a CampaignSpec once from the master goal + planned systems."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from agentic.campaign.family_recipe import (
    REQUIRED_CALCULATIONS,
    build_post_combined_protocol,
    build_shared_analysis_protocol,
    family_contract,
    pack_wants_shared_dyn,
    resolve_feature_pack,
)
from agentic.campaign.pre_combined import (
    REQUIRED_SHARED_SETUP,
    build_shared_setup_protocol,
)
from agentic.campaign.spec import (
    AnalysisRecipe,
    ArtifactContract,
    CampaignSpec,
    RecipeStep,
    spec_from_state,
)

logger = logging.getLogger(__name__)

# Soft per-sim recipes for non-family goals that name classic metrics.
_SOFT_METRIC_STEPS: Dict[str, Dict[str, Any]] = {
    "rmsd": {
        "name": "Backbone RMSD",
        "tool_name": "calculate_rmsd",
        "tool_params": {"output_file": "rmsd.dat"},
    },
    "rmsf": {
        "name": "Per-residue RMSF",
        "tool_name": "calculate_rmsf",
        "tool_params": {"output_file": "rmsf.dat"},
    },
    "rg": {
        "name": "Radius of gyration",
        "tool_name": "calculate_gyration",
        "tool_params": {"output_file": "gyration.dat"},
    },
    "gyration": {
        "name": "Radius of gyration",
        "tool_name": "calculate_gyration",
        "tool_params": {"output_file": "gyration.dat"},
    },
    "sasa": {
        "name": "SASA",
        "tool_name": "calculate_sasa",
        "tool_params": {"output_file": "sasa.dat"},
    },
    "dccm": {
        "name": "DCCM",
        "tool_name": "calculate_dccm",
        "tool_params": {"output_file": "dccm.dat"},
    },
    "energy": {
        "name": "Potential energy",
        "tool_name": "analyze_energy",
        "tool_params": {"output_file": "energy.dat"},
    },
}


def _soft_metric_recipe(goal: str) -> AnalysisRecipe:
    """Compile a soft pin list when the goal names classic metrics.

    Broad dynamics goals return an empty recipe so the LLM still chooses.
    """
    try:
        from agentic.planner.planning_guidelines import detect_requested_metrics

        metrics = detect_requested_metrics(goal)
    except Exception:
        metrics = None
    if not metrics:
        return AnalysisRecipe()
    steps = []
    seen = set()
    for metric in metrics:
        spec = _SOFT_METRIC_STEPS.get(str(metric))
        if not spec or spec["tool_name"] in seen:
            continue
        seen.add(spec["tool_name"])
        steps.append(
            RecipeStep(
                name=spec["name"],
                tool_name=spec["tool_name"],
                tool_params=dict(spec["tool_params"]),
                reason="Soft pin from requested metrics",
            )
        )
    return AnalysisRecipe(steps=steps, metric_groups=sorted(seen))


def _looks_like_apo_holo(labels: List[str], goal: str) -> bool:
    joined = " ".join(labels).lower() + " " + (goal or "").lower()
    has_apo = "apo" in joined
    has_holo = "holo" in joined or "_atp" in joined
    return has_apo and has_holo


def _apo_holo_post_recipe() -> AnalysisRecipe:
    """Paired comparison pins for apo/holo campaigns (overlays, not Ward)."""
    return AnalysisRecipe(
        steps=[
            RecipeStep(
                name="Apo/holo metric overlays",
                tool_name="run_combined_analysis",
                description="Overlay RMSD/RMSF/Rg across apo and holo labels",
                tool_params={"metrics": ["rmsd", "rmsf", "rg"]},
                reason="apo_holo profile: paired overlays",
            ),
        ]
    )


def compile_campaign_spec(state: Optional[Dict[str, Any]] = None) -> CampaignSpec:
    """Build (or refresh) the campaign program from workflow state.

    Family-modular goals get the shared analysis protocol and a hard artifact
    contract. Generic goals leave the protocol empty so LLM/retrieval planning
    still applies.
    """
    state = state or {}
    original = (
        state.get("user_goal_original")
        or state.get("user_goal")
        or ""
    )
    sim_prompts = state.get("sim_prompts") or []
    labels: List[str] = [
        str(sp.get("label") or "") for sp in sim_prompts if sp.get("label")
    ]
    if not labels:
        labels = [
            str(p).rsplit("/", 1)[-1].replace(".pdb", "")
            for p in (state.get("pdb_list") or [])
        ]

    family_mode = "auto"
    pack_name = "from_goal"
    reference_label = ""
    try:
        from agentic.campaign.config import settings_from_state

        settings = settings_from_state(state)
        settings_dict = settings.to_dict()
        gold_override = list(settings.gold_columns or [])
        family_mode = (getattr(settings, "family_modular", None) or "auto").strip().lower()
        pack_name = getattr(settings, "feature_pack", None) or "from_goal"
        reference_label = getattr(settings, "reference_label", None) or ""
    except Exception:
        settings = None
        settings_dict = {}
        gold_override = []

    detect: Dict[str, Any] = {"family": False, "score": 0, "reason": "off", "signals": []}
    if family_mode == "off":
        family = False
        detect = {"family": False, "score": 0, "reason": "family_modular=off", "signals": []}
    elif family_mode == "force":
        family = True
        detect = {"family": True, "score": 6, "reason": "family_modular=force", "signals": []}
    else:
        try:
            from agentic.planner.planning_guidelines import score_family_modular_dynamics

            detect = score_family_modular_dynamics(original)
            family = bool(detect.get("family"))
        except Exception:
            family = False
            detect = {"family": False, "score": 0, "reason": "detector failed", "signals": []}

    from agentic.campaign.family_recipe import build_protocol_from_goal

    pack_key, pack = resolve_feature_pack(pack_name)
    if pack is None:
        pack = build_protocol_from_goal(original, gold_columns=gold_override or None)
    elif gold_override:
        pack = dict(pack)
        pack["gold_columns"] = list(gold_override)
    try:
        from agentic.campaign.feature_plugins import (
            apply_plugins_to_pack,
            load_plugins_from_settings,
        )

        load_plugins_from_settings(
            (settings_dict or {}).get("feature_plugins")
            or getattr(settings, "feature_plugins", None)
        )
        pack = apply_plugins_to_pack(pack)
    except Exception:
        logger.debug("feature plugin merge skipped", exc_info=True)
    gold = list(pack.get("gold_columns") or [])

    settings_dict = dict(settings_dict)
    settings_dict["family_detect"] = {
        "mode": family_mode or "auto",
        "family": family,
        "score": detect.get("score"),
        "signals": list(detect.get("signals") or []),
        "reason": detect.get("reason"),
        "feature_pack": pack_key,
    }
    settings_dict["feature_pack"] = pack_key
    if not reference_label:
        try:
            from agentic.planner.planning_guidelines import _resolve_reference_label

            reference_label = _resolve_reference_label(
                original, labels=labels, default=""
            )
        except Exception:
            reference_label = ""
    settings_dict["reference_label"] = reference_label

    allow_partial = bool(state.get("allow_partial_combined"))
    existing = spec_from_state(state)
    apo_holo = _looks_like_apo_holo(labels, original)
    n_systems = len(labels)

    if family:
        from agentic.campaign.ligand_context import ligand_selection_from_state

        lig_sel = ligand_selection_from_state(state, default=None)
        # Infer ATP from holo-style labels only as a fallback when components
        # have not declared a ligand.
        if not lig_sel:
            blob = (" ".join(labels) + "\n" + str(original or "")).lower()
            # Bare UniProt labels (p17612) do not contain "_atp". The goal
            # still names the ligand ("protein–ATP holo", "ATP COM distance").
            has_atp = bool(
                re.search(r"(?:_atp\b|\batp_mg\b|\bresname\s+atp\b|\batp\b)", blob)
            )
            holo_only = ("holo" in blob) and not re.search(r"\bapo\b", blob)
            if has_atp or holo_only:
                lig_sel = "resname ATP"
        if lig_sel and not state.get("ligand_resnames"):
            token = lig_sel.split()[-1] if lig_sel.lower().startswith("resname ") else lig_sel
            state["ligand_resnames"] = [token]
            settings_dict["ligand_resnames"] = [token]
        elif lig_sel:
            token = lig_sel.split()[-1] if lig_sel.lower().startswith("resname ") else lig_sel
            settings_dict.setdefault("ligand_resnames", [token])
        pin = list(pack.get("pin_tools") or REQUIRED_CALCULATIONS)
        pin_pre = list(REQUIRED_SHARED_SETUP)
        if not lig_sel:
            pin = [t for t in pin if t != "calculate_consensus_pocket_metrics"]
            pin_pre = [
                t
                for t in pin_pre
                if t
                not in {
                    "define_pocket_mapped_residues",
                    "map_pocket_mapped_residues",
                }
            ]
        recipe = build_shared_analysis_protocol(
            gold_columns=gold or None,
            ligand_selection=lig_sel,
            pin_tools=pin,
            metric_groups=list(pack.get("metric_groups") or []),
        )
        pre_recipe = build_shared_setup_protocol(
            settings=settings, ligand_selection=lig_sel
        )
        post_tools = list(pack.get("post_tools") or [])
        if not pack_wants_shared_dyn(pack):
            post_tools = []
        post_recipe = build_post_combined_protocol(
            reference_label=reference_label,
            post_tools=post_tools,
        )
        modular = list(pack.get("modular_dirs") or [])
        if not lig_sel and "reference_pocket" in modular:
            modular = [d for d in modular if d != "reference_pocket"]
        spec = CampaignSpec(
            mode="family_modular",
            n_systems=n_systems,
            labels=labels,
            family_modular=True,
            analysis_recipe=recipe,
            pre_combined_recipe=pre_recipe,
            post_combined_recipe=post_recipe,
            contract=family_contract(
                modular_dirs=modular,
                n_features=len(gold),
            ),
            pin_tools=pin,
            pin_pre_tools=pin_pre,
            pin_post_tools=post_tools,
            allow_partial_combined=allow_partial,
            original_goal=str(original),
            settings=settings_dict,
        )
    else:
        mode = "single_md" if n_systems <= 1 else "multi_protocol"
        if apo_holo and n_systems >= 2:
            mode = "apo_holo"
        soft = _soft_metric_recipe(original)
        spec = CampaignSpec(
            mode=mode,
            n_systems=n_systems,
            labels=labels,
            family_modular=False,
            analysis_recipe=soft,
            pre_combined_recipe=AnalysisRecipe(),
            post_combined_recipe=_apo_holo_post_recipe() if mode == "apo_holo" else AnalysisRecipe(),
            contract=ArtifactContract(),
            pin_tools=[s.tool_name for s in soft.steps],
            pin_pre_tools=[],
            pin_post_tools=(
                [s.tool_name for s in _apo_holo_post_recipe().steps] if mode == "apo_holo" else []
            ),
            allow_partial_combined=True,
            original_goal=str(original),
            settings=settings_dict,
        )
    # Keep an LLM-compiled pre_combined recipe across spec refreshes.
    if existing and existing.pre_combined_compiled and existing.pre_combined_recipe.steps:
        spec.pre_combined_recipe = existing.pre_combined_recipe
        spec.pre_combined_compiled = True
        spec.pin_pre_tools = list(existing.pin_pre_tools or spec.pin_pre_tools)
    logger.info(
        "CampaignSpec compiled: mode=%s n=%d family=%s required_calculations=%s",
        spec.mode,
        spec.n_systems,
        spec.family_modular,
        spec.pin_tools,
    )
    return spec
