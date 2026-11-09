"""Register extra family descriptors without editing the core recipe.

A plugin maps one calculator to an output directory, feature columns, and
the stage that should pin it (``per_sim`` or ``post``). Built-in packs stay
in ``family_recipe.FEATURE_PACKS``. Call :func:`register_feature_plugin` from
campaign code or list entries under ``feature_plugins`` in ``campaign.yaml``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_PLUGINS: Dict[str, Dict[str, Any]] = {}


def register_feature_plugin(
    tool_name: str,
    *,
    output_dir: str = "",
    feature_columns: Optional[List[str]] = None,
    stage: str = "per_sim",
    metric_group: str = "",
    tool_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Register ``tool_name`` as an optional family pin.

    ``stage`` is ``per_sim`` or ``post``. ``output_dir`` is the modular
    directory the science gate should accept when the plugin is pinned.
    """
    name = (tool_name or "").strip()
    if not name:
        raise ValueError("feature plugin requires tool_name")
    stage_name = (stage or "per_sim").strip().lower()
    if stage_name not in {"per_sim", "post", "pre"}:
        stage_name = "per_sim"
    _PLUGINS[name] = {
        "tool_name": name,
        "output_dir": (output_dir or "").strip(),
        "feature_columns": [str(c) for c in (feature_columns or []) if c],
        "stage": stage_name,
        "metric_group": (metric_group or name).strip(),
        "tool_params": dict(tool_params or {}),
    }


def clear_feature_plugins() -> None:
    """Drop registered plugins (tests)."""
    _PLUGINS.clear()


def registered_plugins() -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in _PLUGINS.items()}


def load_plugins_from_settings(raw: Any) -> None:
    """Load ``feature_plugins`` from campaign settings (list of dicts)."""
    if not raw:
        return
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool_name") or item.get("tool") or "").strip()
        if not tool:
            continue
        register_feature_plugin(
            tool,
            output_dir=str(item.get("output_dir") or ""),
            feature_columns=list(item.get("feature_columns") or []),
            stage=str(item.get("stage") or "per_sim"),
            metric_group=str(item.get("metric_group") or ""),
            tool_params=dict(item.get("tool_params") or {}),
        )


def apply_plugins_to_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Return a pack copy with registered plugins merged into pins and columns."""
    out = dict(pack)
    pins = list(out.get("pin_tools") or [])
    posts = list(out.get("post_tools") or [])
    dirs = list(out.get("modular_dirs") or [])
    cols = list(out.get("gold_columns") or [])
    groups = list(out.get("metric_groups") or [])
    for plugin in _PLUGINS.values():
        tool = plugin["tool_name"]
        stage = plugin["stage"]
        if stage == "post":
            if tool not in posts:
                posts.append(tool)
        elif stage == "per_sim":
            if tool not in pins:
                pins.append(tool)
            if plugin["output_dir"] and plugin["output_dir"] not in dirs:
                dirs.append(plugin["output_dir"])
        for col in plugin["feature_columns"]:
            if col not in cols:
                cols.append(col)
        group = plugin["metric_group"]
        if group and group not in groups:
            groups.append(group)
    out["pin_tools"] = pins
    out["post_tools"] = posts
    out["modular_dirs"] = dirs
    out["gold_columns"] = cols
    out["metric_groups"] = groups
    out["plugins"] = registered_plugins()
    return out
