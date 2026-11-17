"""
Resolve human-readable protein names for reports and literature search.

Falls back through master-plan metadata, id:name maps in goals, inline
gene names (e.g. 'IRAK2 holo system (o43187)'), and UniProt lookup when
only an accession is available.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Classic 6-char accessions (O43825, Q9C098) and longer A0A… forms.
# Do NOT treat gene symbols like B3GALT2 / RAB7A as accessions.
_UNIPROT_RE = re.compile(
    r"^(?:"
    r"[OPQABCD][0-9][A-Z0-9]{3}[0-9]"           # classic 6-char
    r"|[A-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{1,5}"  # extended (≥7, e.g. A0A0B4J2F0)
    r")$",
    re.IGNORECASE,
)
_ID_NAME_MAP_RE = re.compile(
    r"\b([A-Za-z0-9_\-]{4,12})\s*:\s*([A-Za-z][A-Za-z0-9_\-]{1,30})"
)

_GENE_SYNONYMS = {
    "ERBB3": ("HER3", "ErbB-3"),
    "TITIN": ("TTN",),
    "MLKL": ("mixed lineage kinase domain-like",),
}

_SKIP_NAMES = frozenset({
    "ATP", "ADP", "MD", "PDB", "HPC", "COM", "FEL", "PCA", "DSSP", "DCCM",
    "RMSD", "RMSF", "SASA", "HTML", "B5R1", "SYSTEM", "PROTEIN", "LIGAND",
    "HOLO", "APO", "FOR", "THE", "AND", "WITH", "USING", "BASED",
    "ANALYZE", "ANALYSE", "COMPUTE", "CALCULATE", "GENERATE", "PREPARE",
    "RUN", "WRITE", "CREATE", "BUILD", "PERFORM", "EXECUTE", "REPORT",
    # Agent-stage verbs from enriched prompts ("q9c098: Preprocess the …")
    "PREPROCESS", "PREPROCESSING", "SIMSETUP", "SETUP", "ANALYSIS",
    "ANALYZER", "REPORTER", "SUPERVISOR", "PLANNER", "PROGRAMMER",
    "DOWNLOAD", "EXTRACT", "SUBMIT", "WRAP", "SOLVATE", "MINIMIZE",
    # Generic planner placeholders — never treat as gene/protein names
    "DOMAIN", "KINASE", "PSEUDOKINASE", "CATALYTIC", "RESIDUE", "RESIDUES",
    "REGION", "LOOP", "CHAIN", "STRUCTURE", "MODEL", "COMPLEX", "UNKNOWN",
    "NA", "NONE", "NULL", "TEST", "EXAMPLE", "TARGET", "QUERY", "GYRATION",
    # Goal verbs and method words — "protein dynamics" is not a gene name
    "DYNAMICS", "DYNAMIC", "BINDING", "EFFECT", "STUDY", "SIMULATION",
    "MOLECULAR", "CONFORMATIONAL", "ACTIVATION", "ALLOSTERIC", "REGULATION",
    "OVERLAY", "COMPARATIVE", "LITERATURE", "FLEXIBILITY", "STABILITY",
    "CORRELATION", "DISTANCE", "POCKET", "CATALYTIC", "BOUND", "HUMAN",
    "WHOLE", "ONLY", "EACH", "THESE", "GIVEN", "AVAILABLE", "CURRENT",
    "NAMED", "FOLLOWING", "INCLUDING", "BETWEEN", "WITHIN", "PRESENT",
})


def _is_placeholder_name(name: Optional[str]) -> bool:
    if not name or not str(name).strip():
        return True
    token = str(name).strip().upper().replace("-", "_")
    if token in _SKIP_NAMES:
        return True
    # e.g. kinase_domain, APO_kinase_domain
    parts = [p for p in re.split(r"[_\s]+", token) if p]
    if parts and all(p in _SKIP_NAMES for p in parts):
        return True
    return False


def _looks_like_uniprot(name: str) -> bool:
    return bool(name and _UNIPROT_RE.match(name.strip()))


def study_proteins_from_goal(text: str) -> List[Tuple[str, str]]:
    """Named proteins in a goal, as ``(id, name)`` pairs.

    Reads lines such as ``p21860: ERBB3``. Generic words and the campaign
    folder name are not proteins.
    """
    found: List[Tuple[str, str]] = []
    seen: set = set()
    for uid, pname in _parse_id_name_map(text or "").items():
        if _is_placeholder_name(pname) or not _looks_like_gene_symbol(pname):
            continue
        key = pname.upper()
        if key in seen:
            continue
        seen.add(key)
        found.append((uid, pname.upper() if pname.isupper() or any(c.isdigit() for c in pname) else pname))
    return found


def _parse_id_name_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _ID_NAME_MAP_RE.finditer(text or ""):
        uid, pname = m.group(1).strip().lower(), m.group(2).strip()
        if any(c.isdigit() for c in uid) and pname[0].isalpha():
            out[uid] = pname
    return out

_ORGANISM_TOKENS = frozenset({
    "human", "mouse", "rat", "yeast", "bovine", "chicken", "zebrafish",
    "arabidopsis", "ecoli", "homo", "sapiens", "mus", "musculus",
})


def _looks_like_gene_symbol(name: str) -> bool:
    """True for compact gene symbols (DCLK3, B3GALT2), not descriptive names."""
    if not name:
        return False
    token = name.strip()
    if " " in token or "," in token:
        return False
    # Hyphenated chemistry / enzyme names (Beta-1,3-…) are not gene symbols
    if "-" in token and re.search(r"-\d", token):
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,14}", token):
        return False
    if _is_placeholder_name(token) or _looks_like_uniprot(token):
        return False
    return True


def _split_uniprot_parenthetical(paren: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse ``DCLK3, human`` or ``Beta-1,3-galactosyltransferase 2, human``.

    Returns ``(gene_symbol_or_None, full_protein_name_or_None)``.
    Only strips a trailing organism clause — chemical commas (``1,3``) stay intact.
    """
    raw = (paren or "").strip()
    if not raw:
        return None, None
    # Strip trailing ", human" / ", Homo sapiens" (last comma-separated organism clause)
    core = re.sub(
        r",\s*(?:homo\s+sapiens|mus\s+musculus|"
        + "|".join(sorted(_ORGANISM_TOKENS, key=len, reverse=True))
        + r")\s*$",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    if not core:
        core = raw
    if _looks_like_gene_symbol(core):
        return core.upper() if re.search(r"\d", core) or core.isupper() else core, None
    # Descriptive protein name (may still start with a gene-like first token)
    first = core.split()[0] if core.split() else core
    gene = first if _looks_like_gene_symbol(first) else None
    return gene, core


def _names_from_goal(label: str, *texts: Optional[str]) -> Dict[str, Optional[str]]:
    """Extract gene symbol and/or full protein name from goal text."""
    combined = " ".join(t for t in texts if t)
    out: Dict[str, Optional[str]] = {"gene": None, "protein_name": None}
    if not combined:
        return out
    label = label or ""

    # UniProt O43825 (Beta-1,3-galactosyltransferase 2, human)
    for m in re.finditer(
        r"\b(?:uniprot\s+)?([A-Za-z][0-9][A-Za-z0-9]{3,9})\s*\(\s*([^)]+?)\s*\)",
        combined,
        re.IGNORECASE,
    ):
        acc, paren = m.group(1), m.group(2)
        if label and acc.lower() != label.lower() and not _looks_like_uniprot(label):
            # Still accept when label is empty / unrelated
            pass
        gene, pname = _split_uniprot_parenthetical(paren)
        if gene and not out["gene"]:
            out["gene"] = gene
        if pname and not out["protein_name"]:
            out["protein_name"] = pname
        if out["gene"] or out["protein_name"]:
            break

    # DCLK3 (Q9C098)
    if not out["gene"]:
        m = re.search(
            r"\b([A-Z][A-Za-z0-9]{2,20})\s*\(\s*([A-Za-z][0-9][A-Za-z0-9]{3,9})\s*\)",
            combined,
        )
        if m and _looks_like_gene_symbol(m.group(1)):
            out["gene"] = m.group(1).upper() if re.search(r"\d", m.group(1)) else m.group(1)

    if not out["gene"]:
        patterns = [
            r"\b(?:protein|gene|kinase)\s+([A-Z][A-Za-z0-9]{2,20})\b",
            r"\b([A-Z][A-Za-z0-9]{2,20})\s+(?:kinase|pseudokinase|protein)\s+domain\b",
        ]
        if label:
            patterns.extend([
                rf"\b([A-Z][A-Za-z0-9]{{2,20}})\s+(?:holo|apo)\s+(?:simulation|system)\s*\(\s*{re.escape(label)}\s*\)",
                rf"of the\s+([A-Z][A-Za-z0-9]{{2,20}})\s+holo\s+system\s*\(\s*{re.escape(label)}\s*\)",
                rf"\b([A-Z][A-Za-z0-9]{{2,20}})\s+(?:holo|apo)\s+(?:simulation|system)[^.\n]{{0,40}}\b{re.escape(label)}\b",
                rf"for the\s+([A-Z][A-Za-z0-9]{{2,20}})\s+(?:holo|apo|kinase|pseudokinase|protein|system)",
                rf"\b([A-Z][A-Za-z0-9]{{2,20}})\b[^.\n]{{0,80}}\(\s*{re.escape(label)}\s*\)",
            ])
        for pat in patterns:
            hit = re.search(pat, combined, re.IGNORECASE)
            if not hit:
                continue
            for candidate in (g for g in hit.groups() if g):
                candidate = candidate.strip()
                if not _looks_like_gene_symbol(candidate):
                    continue
                out["gene"] = (
                    candidate.upper()
                    if candidate.isupper() or re.search(r"\d", candidate)
                    else candidate
                )
                break
            if out["gene"]:
                break
    return out


def _gene_from_goal(label: str, *texts: Optional[str]) -> Optional[str]:
    names = _names_from_goal(label, *texts)
    return names.get("gene") or None


def _lookup_uniprot_gene(accession: str) -> Optional[Dict[str, Any]]:
    try:
        from src.reporter.literature_search import search_uniprot
        result = search_uniprot.invoke({"query": accession, "max_results": 1})
        if not isinstance(result, dict) or not result.get("success"):
            return None
        entries = result.get("entries") or []
        if not entries:
            return None
        entry = entries[0]
        gene = (entry.get("gene_name") or "").strip()
        pname = (entry.get("protein_name") or "").strip()
        organism = (entry.get("organism") or "").strip()
        func = (entry.get("function") or "").strip()
        aliases = list(entry.get("aliases") or [])
        key_refs = list(entry.get("key_references") or [])
        # Prefer gene symbol for short display; keep full UniProt protein name.
        display = gene or pname or accession
        if pname and (not gene or not _looks_like_gene_symbol(gene)):
            display = pname
        if display and not _looks_like_uniprot(str(display)):
            return {
                "gene_name": gene or (display if _looks_like_gene_symbol(str(display)) else ""),
                "protein_name": pname or display,
                "display_name": display,
                "aliases": aliases,
                "key_references": key_refs,
                "organism": organism,
                "function": func,
                "uniprot_id": accession,
            }
    except Exception as exc:
        logger.debug("UniProt lookup failed for %s: %s", accession, exc)
    return None


def _sim_prompt_protein_name(state: Dict[str, Any], sim_label: str) -> Optional[str]:
    for sp in state.get("sim_prompts") or []:
        if (sp.get("label") or "").lower() == sim_label.lower():
            pname = (sp.get("protein_name") or "").strip()
            if pname and not _looks_like_uniprot(pname) and not _is_placeholder_name(pname):
                return pname
            prompt = sp.get("prompt") or sp.get("goal") or ""
            gene = _gene_from_goal(sim_label, prompt)
            if gene:
                return gene
    return None


def resolve_protein_identity(
    state: Optional[Dict[str, Any]] = None,
    user_goal: str = "",
    sim_label: str = "",
) -> Dict[str, Any]:
    """
    Return resolved identity fields for reporting and literature.

    Keys: display_name, gene_name, protein_name, aliases, uniprot_id, organism,
    protein_family, related_terms, key_references, source
    """
    state = state or {}
    goal_text = " ".join(
        t for t in (
            state.get("user_goal_original"),
            state.get("master_enriched_prompt"),
            state.get("enriched_prompt"),
            state.get("user_goal"),
            user_goal,
        )
        if t
    )

    workdir = state.get("working_directory") or ""
    if not sim_label and workdir:
        sim_label = Path(workdir).name

    uniprot_id: Optional[str] = None
    gene_name: Optional[str] = None
    protein_name: Optional[str] = None
    display_name: Optional[str] = None
    aliases: List[str] = []
    key_references: List[Dict[str, Any]] = []
    source = "unknown"
    study_proteins = study_proteins_from_goal(goal_text)
    study_ids = {acc.lower() for acc, _name in study_proteins}
    multi_study = (
        len(study_proteins) >= 2
        and sim_label.lower() not in study_ids
    )
    if multi_study:
        names: List[str] = []
        for _acc, name in study_proteins:
            if name not in names:
                names.append(name)
        display_name = ", ".join(names)
        protein_name = display_name
        gene_name = ""
        aliases = list(names)
        for name in names:
            for extra in _GENE_SYNONYMS.get(name.upper(), ()):
                if extra not in aliases:
                    aliases.append(extra)
        source = "goal_id_map"

    # A label that the goal names explicitly (p21860: ERBB3) wins over a
    # loose "protein <word>" match such as "protein dynamics" or "protein whole".
    id_map = _parse_id_name_map(goal_text)
    mapped = id_map.get((sim_label or "").lower()) if not multi_study else None
    if mapped and not _is_placeholder_name(mapped):
        if _looks_like_gene_symbol(mapped):
            gene_name = mapped.upper() if mapped.isupper() or any(ch.isdigit() for ch in mapped) else mapped
        else:
            protein_name = mapped
        display_name = gene_name or mapped
        source = "goal_id_map"
        goal_names = {}
    elif multi_study:
        goal_names = {}
    else:
        # Enriched plans often contain "q9c098: Preprocess the …", which the
        # id map rejects because the name is an agent-stage verb.
        goal_names = _names_from_goal(sim_label, goal_text)
    if goal_names.get("gene") and not _is_placeholder_name(goal_names["gene"]):
        gene_name = goal_names["gene"]
        display_name = gene_name
        source = "goal_text"
    if goal_names.get("protein_name") and not _is_placeholder_name(goal_names["protein_name"]):
        protein_name = goal_names["protein_name"]
        # Prefer the full descriptive name for report titles when no gene symbol
        if not gene_name or not _looks_like_gene_symbol(gene_name):
            display_name = protein_name
            source = "goal_text"

    id_map = _parse_id_name_map(goal_text)
    if (not display_name or _is_placeholder_name(display_name)) and sim_label:
        mapped = id_map.get(sim_label.lower())
        if mapped and not _is_placeholder_name(mapped):
            if _looks_like_gene_symbol(mapped):
                gene_name = mapped
            else:
                protein_name = protein_name or mapped
            display_name = mapped
            source = "goal_id_map"

    if not display_name or _is_placeholder_name(display_name):
        from_sim = _sim_prompt_protein_name(state, sim_label)
        if from_sim and not _is_placeholder_name(from_sim):
            if _looks_like_gene_symbol(from_sim):
                gene_name = from_sim
            else:
                protein_name = protein_name or from_sim
            display_name = from_sim
            source = "master_plan"

    sys_info = state.get("system_info") if isinstance(state.get("system_info"), dict) else {}
    if (not display_name or _is_placeholder_name(display_name)) and sys_info:
        candidate = (
            sys_info.get("protein_name")
            or sys_info.get("system_name")
            or sys_info.get("gene_name")
        )
        if candidate and not _looks_like_uniprot(str(candidate)) and not _is_placeholder_name(str(candidate)):
            display_name = str(candidate)
            if _looks_like_gene_symbol(str(candidate)):
                gene_name = str(candidate)
            else:
                protein_name = protein_name or str(candidate)
            source = "system_info"

    if not uniprot_id:
        uniprot_id = (
            state.get("uniprot_id")
            or (sys_info.get("uniprot_id") if sys_info else None)
            or (sim_label if _looks_like_uniprot(sim_label) else None)
        )
        # Pull accession from goal text when sim label is not UniProt-like
        if not uniprot_id:
            m = re.search(
                r"\buniprot\s+([A-Za-z][0-9][A-Za-z0-9]{3,9})\b",
                goal_text,
                re.IGNORECASE,
            )
            if m:
                uniprot_id = m.group(1)

    uni_meta: Optional[Dict[str, Any]] = None
    # Always enrich from UniProt when we have an accession — needed for aliases
    # even when goal already supplied a display name.
    need_lookup = True
    if need_lookup:
        acc = uniprot_id or sim_label
        if acc and _looks_like_uniprot(acc):
            uni_meta = _lookup_uniprot_gene(acc)
            if uni_meta:
                gene_name = gene_name or uni_meta.get("gene_name") or gene_name
                protein_name = protein_name or uni_meta.get("protein_name") or protein_name
                aliases = list(uni_meta.get("aliases") or [])
                key_references = list(uni_meta.get("key_references") or [])
                # Prefer full protein name over truncated/non-gene placeholders
                if protein_name and (
                    not display_name
                    or _is_placeholder_name(display_name)
                    or not _looks_like_gene_symbol(str(display_name))
                ):
                    display_name = protein_name
                elif uni_meta.get("display_name") and (
                    not display_name or _is_placeholder_name(display_name)
                ):
                    display_name = uni_meta.get("display_name")
                if source == "unknown":
                    source = "uniprot_lookup"
                if not uniprot_id:
                    uniprot_id = uni_meta.get("uniprot_id")

    if not display_name or _is_placeholder_name(display_name):
        display_name = protein_name or gene_name or sim_label or "Protein"
    # Keep gene_name only when it is a compact symbol; do not overwrite with long names
    if gene_name and not _looks_like_gene_symbol(gene_name):
        gene_name = None
    if not gene_name:
        if _looks_like_gene_symbol(str(display_name)):
            gene_name = str(display_name)
        elif uni_meta and _looks_like_gene_symbol(str(uni_meta.get("gene_name") or "")):
            gene_name = str(uni_meta.get("gene_name"))
        else:
            gene_name = ""
    if not protein_name:
        protein_name = display_name

    # Merge goal/UniProt names into a deduped alias pack
    seen_al: set = set()
    alias_pack: List[str] = []

    def _push_alias(val: Optional[str]) -> None:
        v = (val or "").strip()
        if not v or _is_placeholder_name(v) or "," in v:
            return
        key = v.lower()
        if key in seen_al:
            return
        seen_al.add(key)
        alias_pack.append(v)

    _push_alias(gene_name)
    _push_alias(protein_name)
    _push_alias(display_name)
    for a in aliases:
        _push_alias(a)
    if uniprot_id:
        _push_alias(str(uniprot_id).upper())
        _push_alias(str(uniprot_id).lower())

    related_terms: List[str] = []
    lower = goal_text.lower()
    # Prefer specific enzyme-class terms before broad parents (transferase).
    for term in (
        "pseudokinase", "galactosyltransferase", "glycosyltransferase",
        "kinase", "receptor", "phosphatase", "transferase",
        "neurodegeneration", "autoregulation", "allosteric", "substrate",
    ):
        # Match stem so "neurodegenerative" hits "neurodegeneration"
        stem = term[:12] if len(term) > 12 else term
        if stem in lower or term in lower:
            related_terms.append(term)

    # Enrich related terms from UniProt function text (disease / mechanism cues)
    if uni_meta is None and uniprot_id and _looks_like_uniprot(uniprot_id):
        uni_meta = _lookup_uniprot_gene(uniprot_id)
        if uni_meta and not alias_pack:
            for a in uni_meta.get("aliases") or []:
                _push_alias(a)
            key_references = list(uni_meta.get("key_references") or [])
    function_text = (uni_meta or {}).get("function") or ""
    if function_text:
        func_low = function_text.lower()
        for term in (
            "neurodegeneration", "cancer", "tumor", "autism", "epilepsy",
            "autoregulation", "allosteric", "phosphorylation", "microtubule",
            "doublecortin", "pseudokinase", "kinase",
            "galactosyltransferase", "glycosyltransferase", "glycolipid",
        ):
            stem = term[:10]
            if stem in func_low or term in func_low:
                related_terms.append(term)

    # Specific family first — avoid locking onto bare "transferase"/"kinase"
    protein_family = ""
    for term in (
        "pseudokinase", "galactosyltransferase", "glycosyltransferase",
        "kinase", "receptor", "phosphatase", "transferase",
    ):
        if term in related_terms or term in lower:
            protein_family = term
            break
    if not protein_family and function_text:
        func_low = function_text.lower()
        for term in (
            "pseudokinase", "galactosyltransferase", "glycosyltransferase",
            "kinase", "receptor", "transferase",
        ):
            if term in func_low:
                protein_family = term
                break

    # Deduplicate related terms
    seen_rt: set = set()
    related_deduped: List[str] = []
    for t in related_terms:
        low = t.lower()
        if low not in seen_rt:
            seen_rt.add(low)
            related_deduped.append(t)

    return {
        "display_name": display_name,
        "gene_name": gene_name,
        "protein_name": protein_name or display_name,
        "aliases": alias_pack,
        "key_references": key_references,
        "uniprot_id": (
            str(uniprot_id).lower()
            if uniprot_id and _looks_like_uniprot(str(uniprot_id))
            else ""
        ),
        "study_proteins": [
            {"id": acc, "name": name} for acc, name in study_proteins
        ],
        "organism": (sys_info or {}).get("organism", "") or (uni_meta or {}).get("organism", ""),
        "protein_family": protein_family,
        "related_terms": related_deduped,
        "function": function_text,
        "source": source,
    }
