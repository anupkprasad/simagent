"""Load one campaign.yaml: shipped default → working-dir → CLI → env."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SHIPPED_CAMPAIGN_YAML = Path(__file__).resolve().parent / "campaign.yaml"

_ENV_MAP = {
    "conservation_metric": "AGENTIC_CONSERVATION_METRIC",
    "min_conservation": "AGENTIC_MIN_CONSERVATION",
    "min_coverage": "AGENTIC_MIN_COVERAGE",
    "pocket_cutoff_A": "AGENTIC_POCKET_CUTOFF_A",
    "lobe_method": "AGENTIC_LOBE_METHOD",
    "retrieval_k": "AGENTIC_RETRIEVAL_K",
    "embed_model": "AGENTIC_EMBED_MODEL",
    "hitl": "AGENTIC_HITL",
}

_DEFAULT_AGENT_RETRIEVAL: Dict[str, Any] = {
    "analysis": {"knowledge_k": 4, "memory_k": 5, "study_k": 4},
    "reporter": {"knowledge_k": 6, "memory_k": 4, "study_k": 6},
    "supervisor": {
        "knowledge_k": 2,
        "memory_k": 5,
        "study_k": 4,
        "route_once": True,
        "allowed_destinations": ["analysis", "reporter", "final_report"],
    },
}


def _merge_agent_retrieval(raw: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        name: dict(profile) for name, profile in _DEFAULT_AGENT_RETRIEVAL.items()
    }
    if not isinstance(raw, dict):
        return merged
    for name, profile in raw.items():
        if not isinstance(profile, dict):
            continue
        base = dict(merged.get(str(name), {}))
        base.update(profile)
        merged[str(name)] = base
    return merged


_DEFAULT_GOLD: List[str] = []


@dataclass
class CampaignSettings:
    """Typed campaign knobs shared by CLI, CampaignSpec, and tools."""

    conservation_metric: str = "similarity"
    min_conservation: float = 0.5
    min_coverage: float = 0.25
    pocket_cutoff_A: float = 15.0
    lobe_method: str = "auto"
    retrieval_k: int = 16
    embed_model: str = ""
    agent_retrieval: Dict[str, Any] = field(
        default_factory=lambda: _merge_agent_retrieval(None)
    )
    gold_columns: List[str] = field(default_factory=lambda: list(_DEFAULT_GOLD))
    hitl: Optional[str] = None
    feature_pack: str = "from_goal"
    family_modular: str = "auto"  # force | auto | off
    reference_label: str = ""
    feature_plugins: List[Dict[str, Any]] = field(default_factory=list)
    source_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "CampaignSettings":
        data = dict(data or {})
        if "gold_columns" in data and data.get("gold_columns") is not None:
            gold = list(data.get("gold_columns") or [])
        else:
            gold = list(_DEFAULT_GOLD)
        hitl = data.get("hitl")
        if isinstance(hitl, str):
            hitl = hitl.strip().lower() or None
            if hitl in ("none", "null", "off", "false"):
                hitl = None
        embed = data.get("embed_model")
        if embed is None:
            embed = ""
        return cls(
            conservation_metric=str(data.get("conservation_metric") or "similarity"),
            min_conservation=float(data.get("min_conservation") if data.get("min_conservation") is not None else 0.5),
            min_coverage=float(data.get("min_coverage") if data.get("min_coverage") is not None else 0.25),
            pocket_cutoff_A=float(data.get("pocket_cutoff_A") if data.get("pocket_cutoff_A") is not None else 15.0),
            lobe_method=str(data.get("lobe_method") or "auto"),
            retrieval_k=int(data.get("retrieval_k") if data.get("retrieval_k") is not None else 16),
            embed_model=str(embed),
            agent_retrieval=_merge_agent_retrieval(data.get("agent_retrieval")),
            gold_columns=[str(c) for c in gold if c],
            hitl=hitl,
            feature_pack=str(data.get("feature_pack") or "from_goal"),
            family_modular=str(data.get("family_modular") or "auto").strip().lower(),
            reference_label=str(data.get("reference_label") or ""),
            feature_plugins=list(data.get("feature_plugins") or []),
            source_path=str(data.get("source_path") or ""),
        )


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def _env_overrides() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, env_name in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        if key in ("min_conservation", "min_coverage", "pocket_cutoff_A"):
            try:
                out[key] = float(raw)
            except ValueError:
                continue
        elif key == "retrieval_k":
            try:
                out[key] = int(raw)
            except ValueError:
                continue
        else:
            out[key] = raw
    return out


def resolve_campaign_yaml_path(
    *,
    explicit: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> Path:
    """CLI path wins, then {working-dir}/campaign.yaml, then the shipped default."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p.resolve()
        logger.warning("campaign.yaml not found at --campaign-yaml %s", explicit)
    if working_dir:
        local = Path(working_dir).expanduser() / "campaign.yaml"
        if local.is_file():
            return local.resolve()
    return SHIPPED_CAMPAIGN_YAML


def load_campaign_settings(
    path: Optional[str] = None,
    *,
    working_dir: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> CampaignSettings:
    """Merge shipped defaults, file, env, then explicit overrides.

    Precedence (later wins): shipped YAML → {working-dir}/campaign.yaml or
    ``path`` → ``AGENTIC_*`` env → ``overrides`` (CLI).
    """
    chosen = resolve_campaign_yaml_path(explicit=path, working_dir=working_dir)
    shipped = _read_yaml(SHIPPED_CAMPAIGN_YAML)
    file_data = _read_yaml(chosen) if chosen != SHIPPED_CAMPAIGN_YAML else shipped
    merged: Dict[str, Any] = {}
    merged.update(shipped)
    if chosen != SHIPPED_CAMPAIGN_YAML:
        merged.update(file_data)
    merged.update(_env_overrides())
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    merged["source_path"] = str(chosen)
    settings = CampaignSettings.from_dict(merged)
    if settings.embed_model and not os.environ.get("AGENTIC_EMBED_MODEL"):
        os.environ["AGENTIC_EMBED_MODEL"] = settings.embed_model
    logger.info(
        "CampaignSettings loaded from %s metric=%s cov=%s cutoff=%s k=%s",
        chosen,
        settings.conservation_metric,
        settings.min_coverage,
        settings.pocket_cutoff_A,
        settings.retrieval_k,
    )
    return settings


def settings_from_state(state: Optional[Dict[str, Any]]) -> CampaignSettings:
    """Read settings already stored on workflow state, or load defaults."""
    if not state:
        return load_campaign_settings()
    raw = state.get("campaign_settings")
    if isinstance(raw, CampaignSettings):
        return raw
    if isinstance(raw, dict) and raw:
        return CampaignSettings.from_dict(raw)
    return load_campaign_settings(
        path=state.get("campaign_yaml"),
        working_dir=str(state.get("multi_sim_base_dir") or state.get("working_directory") or ""),
    )
