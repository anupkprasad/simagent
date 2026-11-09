"""Keep one simulation directory from fetching another protein.

A multi-protein goal is split into one working directory per system. The
preprocess planner still sees the full goal, and it can plan
``download_structure`` for a protein that belongs in a sibling directory.
That write lands in the wrong folder and the next canonical step
(``separate_complex_components``) then replaces ``protein.pdb``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .structure_request_parser import (
    _VALID_PDB_ID_RE,
    _VALID_UNIPROT_RE,
    extract_uniprot_ids,
)

# Tools that write coordinates into the current preprocess directory.
COORDINATE_DOWNLOAD_TOOLS = frozenset({
    "download_structure",
    "download_pdb_by_id",
    "acquire_protein_structure",
})

# Sequence download does not replace protein.pdb, but a foreign accession
# still does not belong in this directory.
FOREIGN_FETCH_TOOLS = COORDINATE_DOWNLOAD_TOOLS | frozenset({
    "download_protein_sequence",
})


def identifier_from_label(label: str) -> Optional[str]:
    """Return a UniProt accession or PDB id embedded in a path or directory name."""
    if not label:
        return None
    stem = Path(str(label)).stem.upper()
    if _VALID_UNIPROT_RE.match(stem) or _VALID_PDB_ID_RE.match(stem):
        return stem
    for token in re.split(r"[^A-Za-z0-9]+", stem):
        if _VALID_UNIPROT_RE.match(token) or _VALID_PDB_ID_RE.match(token):
            return token
    return None


def allowed_structure_ids(
    *labels: Optional[str],
    uniprot_id: Optional[str] = None,
    pdb_id: Optional[str] = None,
) -> Set[str]:
    """Identifiers this simulation directory is allowed to fetch or write."""
    allowed: Set[str] = set()
    for explicit in (uniprot_id, pdb_id):
        if explicit and str(explicit).strip():
            allowed.add(str(explicit).strip().upper())
    for label in labels:
        ident = identifier_from_label(label or "")
        if ident:
            allowed.add(ident)
    return allowed


def sibling_structure_ids(sim_directory: Optional[str]) -> Set[str]:
    """Accession-like names of other simulation folders next to this one."""
    if not sim_directory:
        return set()
    sim_dir = Path(sim_directory)
    parent = sim_dir.parent
    if not parent.is_dir():
        return set()
    found: Set[str] = set()
    try:
        children = list(parent.iterdir())
    except OSError:
        return set()
    for child in children:
        if not child.is_dir():
            continue
        try:
            if child.resolve() == sim_dir.resolve():
                continue
        except OSError:
            if child.name == sim_dir.name:
                continue
        ident = identifier_from_label(child.name)
        if ident:
            found.add(ident)
    return found


def requested_structure_ids(tool_name: str, params: Optional[Dict[str, Any]]) -> List[str]:
    """Identifiers a download/acquire tool call is asking to fetch."""
    params = params or {}
    found: List[str] = []
    for key in ("uniprot_id", "accession", "pdb_id"):
        value = params.get(key)
        if value and str(value).strip():
            found.append(str(value).strip().upper())
    text = params.get("user_request") or params.get("text") or ""
    if text:
        found.extend(extract_uniprot_ids(str(text)))
    seen: Set[str] = set()
    unique: List[str] = []
    for ident in found:
        if ident not in seen:
            seen.add(ident)
            unique.append(ident)
    return unique


def structure_fetch_skip_reason(
    tool_name: str,
    params: Optional[Dict[str, Any]],
    *,
    allowed_ids: Iterable[str] = (),
    sibling_ids: Iterable[str] = (),
    structure_already_staged: bool = False,
) -> Optional[str]:
    """Why this tool call must not write a structure into the current directory.

    Returns None when the call is allowed.
    """
    name = (tool_name or "").strip().lower()
    if name not in FOREIGN_FETCH_TOOLS:
        return None

    allowed = {str(i).upper() for i in allowed_ids if i}
    siblings = {str(i).upper() for i in sibling_ids if i}
    requested = requested_structure_ids(name, params)
    foreign = [ident for ident in requested if ident in siblings or (allowed and ident not in allowed)]
    if foreign:
        owner = ", ".join(sorted(allowed)) if allowed else "this simulation"
        return (
            f"This directory is only for {owner}. "
            f"Not downloading {', '.join(foreign)} here."
        )

    if name in COORDINATE_DOWNLOAD_TOOLS and structure_already_staged:
        return (
            "This simulation already has its structure. "
            "Not downloading another PDB into this directory."
        )
    return None
