"""Literature search functionality for scientific reports"""
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.parse import quote_plus, urlencode
from urllib.error import URLError
from langchain.tools import tool

logger = logging.getLogger(__name__)

# Terms that signal dynamics / MD relevance in abstracts and titles
_DYNAMICS_TERMS = (
    "molecular dynamics", "md simulation", "conformational", "flexibility",
    "rmsd", "rmsf", "alloster", "cross-correlation", "dccm", "dynamics",
    "trajectory", "activation loop",
)

_LIGAND_TERMS = ("atp", "ligand", "binding", "holo", "apo", "nucleotide")

_REGION_RE = re.compile(
    r"(?:resid(?:ue)?s?\s*)?(\d+)\s*(?:to|-)\s*(\d+)",
    re.IGNORECASE,
)
_PROTEIN_MAP_RE = re.compile(
    r'\b([A-Za-z0-9_\-]+)\s*:\s*([A-Za-z][A-Za-z0-9_\-]+)',
)


def extract_research_context(
    user_goal: Optional[str] = None,
    protein_name: Optional[str] = None,
    analysis_types: Optional[List[str]] = None,
    analysis_stats: Optional[Dict[str, Any]] = None,
    protein_family: Optional[str] = None,
    related_terms: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None,
    gene_name: Optional[str] = None,
    uniprot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract structured research context from the user goal and analysis outputs."""
    goal = (user_goal or "").strip()
    goal_lower = goal.lower()
    context: Dict[str, Any] = {
        "user_goal": goal,
        "protein_names": [],
        "protein_ids": [],
        "aliases": [],
        "ligand_terms": [],
        "region_terms": [],
        "hypothesis_terms": [],
        "analysis_types": list(analysis_types or []),
        "finding_phrases": [],
        "protein_family": (protein_family or "").strip(),
        "related_terms": list(related_terms or []),
        "primary_gene": "",
    }

    _BAD_PROTEIN_TOKENS = {
        "domain", "kinase", "pseudokinase", "protein", "system", "apo", "holo",
        "preprocess", "preprocessing", "setup", "simsetup", "analysis",
        "reporter", "planner", "supervisor", "gyration", "structure",
        "dynamics", "dynamic", "binding", "effect", "study", "simulation",
        "molecular", "conformational", "activation", "allosteric", "regulation",
        "overlay", "comparative", "literature", "flexibility", "human",
    }

    def _ok_protein_token(part: str) -> bool:
        p = (part or "").strip()
        if len(p) < 2:
            return False
        if p.lower() in _BAD_PROTEIN_TOKENS:
            return False
        return True

    def _add_name(part: str, *, as_id: bool = False) -> None:
        p = (part or "").strip()
        if not p or not _ok_protein_token(p):
            return
        if as_id:
            if p not in context["protein_ids"]:
                context["protein_ids"].append(p)
            return
        if p not in context["protein_names"]:
            context["protein_names"].append(p)
        if p not in context["aliases"]:
            context["aliases"].append(p)

    if gene_name and _ok_protein_token(gene_name):
        _add_name(gene_name)
        context["primary_gene"] = gene_name.strip()

    if protein_name:
        # Keep full descriptive names intact (chemical commas like 1,3 stay).
        # Only split on semicolon / slash list separators — never on commas.
        _add_name(protein_name.strip())
        for part in re.split(r"[;/]+", protein_name):
            part = part.strip()
            if part != protein_name.strip() and _ok_protein_token(part):
                _add_name(part)

    for alias in aliases or []:
        _add_name(str(alias))

    if uniprot_id:
        _add_name(str(uniprot_id), as_id=True)

    # Rescue gene from goal text when caller passed a placeholder name
    if not context["protein_names"]:
        m = re.search(
            r"\b(?:uniprot\s+)?([A-Za-z][0-9][A-Za-z0-9]{3,9})\s*\(\s*([A-Z][A-Za-z0-9]{2,20})\b",
            goal,
            re.IGNORECASE,
        )
        if m and _ok_protein_token(m.group(2)):
            context["protein_ids"].append(m.group(1))
            _add_name(m.group(2).upper() if m.group(2).isupper() else m.group(2))
        else:
            m2 = re.search(
                r"\b([A-Z][A-Za-z0-9]{2,20})\s+(?:kinase|pseudokinase)\s+domain\b",
                goal,
            )
            if m2 and _ok_protein_token(m2.group(1)):
                _add_name(m2.group(1))

    if not context["primary_gene"] and context["protein_names"]:
        # Prefer a compact gene-like token over a long descriptive name
        for cand in context["protein_names"]:
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,14}", cand or ""):
                context["primary_gene"] = cand
                break
        if not context["primary_gene"]:
            context["primary_gene"] = context["protein_names"][0]

    study_genes: List[str] = []
    for match in _PROTEIN_MAP_RE.finditer(goal):
        pid, pname = match.group(1).strip(), match.group(2).strip()
        if len(pid) >= 3 and any(c.isdigit() for c in pid) and pname[0].isalpha() and _ok_protein_token(pname):
            if pid not in context["protein_ids"]:
                context["protein_ids"].append(pid)
            _add_name(pname)
            if pname not in study_genes:
                study_genes.append(pname)
    context["study_genes"] = study_genes
    if len(study_genes) >= 2:
        # Every named protein is a search target. Do not crown one of them,
        # or a generic word such as "dynamics", as the only gene.
        context["primary_gene"] = ""

    apo_only = bool(
        re.search(
            r"\b(?:no|without)\s+ligand\b|\bprotein[- ]only\b"
            r"|\bapo\b(?![^.\n]{0,60}\bholo\b)",
            goal_lower,
        )
    )
    for term in _LIGAND_TERMS:
        if term in goal_lower:
            if apo_only and term in {"apo", "ligand", "atp", "holo", "binding"}:
                continue
            context["ligand_terms"].append(term)

    for match in _REGION_RE.finditer(goal):
        context["region_terms"].append(f"residues {match.group(1)}-{match.group(2)}")

    _HYP_TRIGGERS = (
        "inhibit", "bind", "interact", "affect", "role", "function", "mechanism",
        "pathway", "stability", "flexibility", "alloster", "mutation", "mutant",
        "drug", "therapeutic", "disease", "cancer", "activate", "deactivate",
        "phosphorylat", "fold", "unfold", "aggregate", "dimer", "oligomer",
        "signaling", "effect of", "compare", "versus", "apo", "holo",
    )
    for trig in _HYP_TRIGGERS:
        if trig in goal_lower:
            context["hypothesis_terms"].append(trig)

    if analysis_stats:
        for atype, stats in analysis_stats.items():
            if not isinstance(stats, dict):
                continue
            alow = str(atype).lower()
            if "rmsd" in alow:
                mean = stats.get("mean_rmsd_angstrom") or stats.get("mean")
                if mean is not None:
                    try:
                        val = float(mean)
                        phrase = (
                            "conformational instability"
                            if val > 3.0
                            else "structural stability"
                        )
                        context["finding_phrases"].append(
                            f"{phrase} (mean RMSD {val:.2f} Å)"
                        )
                    except (TypeError, ValueError):
                        pass
            if "rmsf" in alow:
                mean = stats.get("mean_rmsf_angstrom") or stats.get("mean")
                if mean is not None:
                    try:
                        val = float(mean)
                        if val > 2.0:
                            context["finding_phrases"].append(
                                f"elevated backbone flexibility (mean RMSF {val:.2f} Å)"
                            )
                    except (TypeError, ValueError):
                        pass
            if "dccm" in alow or "correlation" in alow:
                context["finding_phrases"].append("correlated residue motions / allosteric coupling")

    # Deduplicate while preserving order
    for key in ("protein_names", "protein_ids", "ligand_terms", "region_terms",
                "hypothesis_terms", "finding_phrases"):
        seen: set = set()
        deduped = []
        for item in context[key]:
            low = str(item).lower()
            if low not in seen:
                seen.add(low)
                deduped.append(item)
        context[key] = deduped

    return context


def build_analysis_summary_text(
    analysis_data: Optional[Dict[str, Any]] = None,
    analysis_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a concise text summary of simulation analysis for LLM prompts."""
    lines: List[str] = []
    stats_map = dict(analysis_stats or {})

    if analysis_data and isinstance(analysis_data, dict):
        for entry in analysis_data.get("entries", []):
            atype = entry.get("analysis_type", "Unknown")
            stats = entry.get("statistics", {})
            if isinstance(stats, dict) and stats:
                stats_map.setdefault(atype, stats)

    for atype, stats in stats_map.items():
        if not isinstance(stats, dict):
            continue
        numeric = {
            k: v for k, v in stats.items()
            if isinstance(v, (int, float)) and not k.startswith("_")
        }
        if numeric:
            stat_line = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in list(numeric.items())[:8])
            lines.append(f"- {atype}: {stat_line}")
        else:
            lines.append(f"- {atype}: completed")

    return "\n".join(lines) if lines else "No quantitative analysis statistics available."


def _confusable_gene_penalty(haystack: str, primary_genes: List[str]) -> int:
    """Penalise papers about a closely related but different gene (e.g. IRAK4 vs IRAK2)."""
    for gene in primary_genes:
        if not gene or len(gene) < 3:
            continue
        g = gene.upper()
        prefix = re.sub(r"\d+$", "", g)
        if len(prefix) < 3:
            continue
        siblings = [
            m.group(1).upper()
            for m in re.finditer(rf"\b({re.escape(prefix)}\d{{1,3}})\b", haystack, re.IGNORECASE)
        ]
        if not siblings:
            continue
        if g in siblings:
            return 0
        return -18
    return 0


def score_literature_ref_relevance(
    ref: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[int, str]:
    """Score one reference against research context; return (score, short note).

    Two-tier model:
      * Identity — any alias / gene / UniProt id in title or abstract
      * Theme — family + dynamics + analysis metrics + finding phrases

    Absence of the gene is **not** a hard penalty when theme evidence is strong.
    UniProt-seeded curated refs get a baseline boost.
    """
    protein_names = list(context.get("protein_names") or [])
    protein_ids = list(context.get("protein_ids") or [])
    aliases = list(context.get("aliases") or [])
    identity_terms = list(dict.fromkeys(
        [t for t in (aliases + protein_names + protein_ids) if t]
    ))
    primary_gene = context.get("primary_gene") or (protein_names[0] if protein_names else "")
    protein_family = (context.get("protein_family") or "").lower()
    hypothesis_terms = context.get("hypothesis_terms") or []
    ligand_terms = context.get("ligand_terms") or []
    region_terms = context.get("region_terms") or []
    finding_phrases = context.get("finding_phrases") or []
    analysis_types = [str(a).lower() for a in (context.get("analysis_types") or [])]

    title = ref.get("title", "") or ""
    abstract = ref.get("abstract") or ""
    haystack = " ".join(filter(None, [
        title,
        abstract,
        ref.get("journal", ""),
        " ".join(ref.get("authors") or []),
    ])).lower()
    title_lower = title.lower()

    score = 0
    matched: List[str] = []
    matched_aliases: List[str] = []

    _GENERIC = {
        "domain", "kinase", "pseudokinase", "protein", "system", "apo", "holo",
        "catalytic", "region", "loop", "structure",
        "preprocess", "preprocessing", "setup", "simsetup", "analysis",
        "reporter", "planner", "supervisor", "gyration",
    }
    # Prefer a real gene symbol; ignore placeholders for scoring boosts
    effective_gene = ""
    if primary_gene and primary_gene.lower() not in _GENERIC:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,14}", primary_gene):
            effective_gene = primary_gene
    if not effective_gene:
        for n in identity_terms:
            if n and n.lower() not in _GENERIC and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,14}", n):
                effective_gene = n
                break

    mentions_protein = False
    identity_hit = False

    # UniProt curated seed — always keep above the ranking floor
    if str(ref.get("source") or "").lower() in {"uniprot_curated", "uniprot"}:
        score += 14
        matched.append("uniprot_curated")
        identity_hit = True

    for name in identity_terms:
        if not name or name.lower() in _GENERIC:
            continue
        nl = name.lower()
        if len(nl) < 3:
            continue
        if nl in title_lower:
            bump = 30 if name == effective_gene else 14
            score += bump
            matched.append(name)
            matched_aliases.append(name)
            mentions_protein = True
            identity_hit = True
        elif nl in haystack:
            bump = 18 if name == effective_gene else 10
            score += bump
            matched.append(name)
            matched_aliases.append(name)
            mentions_protein = True
            identity_hit = True

    if effective_gene:
        score += _confusable_gene_penalty(haystack, [effective_gene])

    _GENERIC_FAMILY = {
        "kinase", "transferase", "receptor", "phosphatase", "enzyme", "protein",
    }

    def _family_term_hit(term: str, text: str) -> bool:
        tl = (term or "").lower().strip()
        if len(tl) < 4:
            return False
        # Broad parents must be whole words (avoid succinyl*transferase* false hits)
        if tl in _GENERIC_FAMILY:
            return bool(re.search(rf"\b{re.escape(tl)}\b", text))
        return tl in text

    family_hit = False
    if protein_family and _family_term_hit(protein_family, haystack):
        # Specific families (galactosyltransferase) outweigh bare transferase
        bump = 12 if protein_family.lower() not in _GENERIC_FAMILY else 5
        score += bump
        matched.append(protein_family)
        family_hit = True
    for term in context.get("related_terms") or []:
        tl = str(term).lower()
        if len(tl) > 4 and _family_term_hit(tl, haystack):
            score += 4 if tl not in _GENERIC_FAMILY else 2
            matched.append(str(term))
            if tl == (protein_family or "").lower() or tl not in _GENERIC_FAMILY:
                if "transferase" in tl or "kinase" in tl or tl == (protein_family or "").lower():
                    family_hit = True

    theme_hits = 0
    for term in hypothesis_terms + ligand_terms:
        if term.lower() in haystack:
            score += 5
            theme_hits += 1

    for term in _DYNAMICS_TERMS:
        if term in haystack:
            score += 3
            theme_hits += 1

    for atype in analysis_types:
        al = (atype or "").lower().strip()
        if al and al not in {"pdb_input_validation", "trajectorybatch", "unknown"} and al in haystack:
            score += 4
            theme_hits += 1

    for region in region_terms:
        if region.lower() in haystack:
            score += 4
            theme_hits += 1

    _FINDING_THEMES = (
        "conformational stability", "structural stability",
        "conformational instability", "backbone flexibility",
        "flexible loop", "activation loop", "correlated residue",
        "allosteric",
    )
    for finding in finding_phrases:
        fl = finding.lower()
        theme_matched = False
        for phrase in _FINDING_THEMES:
            if phrase in fl and phrase in haystack:
                score += 5
                theme_hits += 1
                theme_matched = True
                break
        if theme_matched:
            continue
        for word in fl.replace("(", " ").replace(")", " ").split():
            if len(word) > 5 and word not in {"angstrom", "mean"} and word in haystack:
                score += 3
                theme_hits += 1
                break

    if abstract:
        score += 2
        abstract_lower = abstract.lower()
        for term in _DYNAMICS_TERMS:
            if term in abstract_lower:
                score += 2
                theme_hits += 1
        for term in ligand_terms:
            if term in abstract_lower:
                score += 3
                theme_hits += 1

    # Strong theme needs dynamics + specific family (or many theme hits).
    # Bare "protein dynamics" reviews without family/identity stay weak.
    specific_family = bool(
        protein_family and protein_family.lower() not in _GENERIC_FAMILY and family_hit
    )
    strong_theme = (
        (specific_family and theme_hits >= 2)
        or (family_hit and theme_hits >= 3)
        or theme_hits >= 4
    )

    # Soft demotion only when identity is known AND theme evidence is weak
    if effective_gene and not identity_hit and not strong_theme:
        score -= 10
    elif not identity_hit and strong_theme:
        # Theme / family papers without gene name stay rankable
        score += 6
    elif not identity_hit and not strong_theme:
        score -= 4

    # Soft floor — papers with neither identity nor strong theme stay low
    if not identity_hit and not strong_theme and str(ref.get("source") or "") != "uniprot_curated":
        score = min(score, 8)

    if matched_aliases:
        ref["_matched_aliases"] = list(dict.fromkeys(matched_aliases))

    note = ""
    if matched_aliases and mentions_protein:
        note = f"Mentions: {', '.join(dict.fromkeys(matched_aliases))}"
    elif identity_hit and matched:
        note = f"Identity: {', '.join(dict.fromkeys(matched))}"
    elif strong_theme:
        note = "Matches family / dynamics / analysis themes"
    elif theme_hits >= 2 or score >= 10:
        note = "Matches objective / dynamics / analysis themes"
    elif matched:
        note = f"Related terms: {', '.join(dict.fromkeys(matched))}"
    else:
        note = "Low contextual overlap with study"

    return score, note


def _ref_mentions_study_proteins(
    ref: Dict[str, Any],
    protein_names: List[str],
    protein_ids: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None,
) -> bool:
    """True when title/abstract mentions a named study protein / alias / UniProt id."""
    names = [
        n
        for n in list(protein_names or [])
        + list(protein_ids or [])
        + list(aliases or [])
        if n
    ]
    if not names:
        return False
    haystack = " ".join(
        filter(
            None,
            [
                ref.get("title") or "",
                ref.get("abstract") or "",
                ref.get("journal") or "",
            ],
        )
    ).lower()
    for name in names:
        token = str(name).strip().lower()
        if len(token) >= 3 and token in haystack:
            return True
    return False


def rank_literature_refs(
    refs: List[Dict[str, Any]],
    context: Dict[str, Any],
    max_refs: int = 10,
    min_score: int = 10,
    protein_slot_fraction: float = 0.5,
) -> List[Dict[str, Any]]:
    """Re-rank literature by relevance to protein, user goal, and simulation findings.

    Prefer papers that name a study protein, but always reserve slots for
    high-scoring papers that match the research objective, methods, or
    observed results (even when they do not mention the simulated gene).
    """
    protein_names = list(context.get("protein_names") or [])
    protein_ids = list(context.get("protein_ids") or [])
    aliases = list(context.get("aliases") or [])
    primary = context.get("primary_gene")
    if primary and primary not in protein_names:
        protein_names = [primary] + protein_names

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for ref in refs:
        score, note = score_literature_ref_relevance(ref, context)
        ref_copy = dict(ref)
        ref_copy["_relevance_score"] = score
        ref_copy["relevance_note"] = note
        if ref.get("_matched_aliases"):
            ref_copy["matched_aliases"] = list(ref.get("_matched_aliases") or [])
        scored.append((score, ref_copy))

    scored.sort(key=lambda x: x[0], reverse=True)

    protein_hits = [
        r
        for s, r in scored
        if s >= min_score
        and _ref_mentions_study_proteins(r, protein_names, protein_ids, aliases)
    ]
    # Theme / objective / results papers: stricter when a primary gene is known
    # so unrelated "protein dynamics" or broad transferase hits do not dominate.
    has_primary = bool(primary or protein_names)
    theme_min = max(12, min_score) if has_primary else max(8, min_score - 2)
    theme_hits = [
        r
        for s, r in scored
        if s >= theme_min
        and not _ref_mentions_study_proteins(r, protein_names, protein_ids, aliases)
    ]

    max_refs = max(1, int(max_refs))
    # Prefer protein-named papers (~70%) when identity is known.
    frac = 0.7 if has_primary else protein_slot_fraction
    n_protein_slots = max(1, int(round(max_refs * frac)))
    if protein_hits:
        n_protein_slots = min(len(protein_hits), max(n_protein_slots, min(3, max_refs)))
    else:
        n_protein_slots = 0
    n_theme_slots = max_refs - n_protein_slots
    # A named multi-protein study should cite those proteins, not a generic
    # dynamics review that happens to share a method word.
    if len(context.get("study_genes") or []) >= 2:
        n_theme_slots = 0

    filtered: List[Dict[str, Any]] = []
    filtered.extend(protein_hits[:n_protein_slots])
    filtered.extend(theme_hits[:n_theme_slots])

    # If either bucket under-filled, top up from remaining high scores.
    if len(filtered) < max_refs and len(context.get("study_genes") or []) < 2:
        seen = {id(r) for r in filtered}
        for _, r in scored:
            if id(r) in seen:
                continue
            if r.get("_relevance_score", 0) < theme_min and filtered:
                continue
            filtered.append(r)
            seen.add(id(r))
            if len(filtered) >= max_refs:
                break

    if not filtered and len(context.get("study_genes") or []) < 2:
        filtered = [r for _, r in scored[:max_refs]]

    for r in filtered:
        r.pop("_relevance_score", None)
    return filtered[:max_refs]


_CITE_BRACKET_RE = re.compile(r"\[(\d+)\]")


def citation_first_appearance_order(*texts: Optional[str]) -> List[int]:
    """Return citation numbers in order of first appearance across *texts*."""
    order: List[int] = []
    seen: set = set()
    for text in texts:
        if not text:
            continue
        for match in _CITE_BRACKET_RE.finditer(text):
            num = int(match.group(1))
            if num not in seen:
                seen.add(num)
                order.append(num)
    return order


def remap_citation_brackets(text: Optional[str], mapping: Dict[int, int]) -> Optional[str]:
    """Rewrite ``[old]`` → ``[new]`` using *mapping*; drop unknown brackets."""
    if not text:
        return text

    def _sub(match: re.Match) -> str:
        old = int(match.group(1))
        new = mapping.get(old)
        if new is None:
            return ""
        return f"[{new}]"

    return _CITE_BRACKET_RE.sub(_sub, text)


def sync_cited_literature(
    literature_refs: List[Dict[str, Any]],
    literature_review: Optional[str] = None,
    final_impression: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    """Keep only references cited in review/impression; renumber by first use.

    Reading order is Literature Review → Final Impression.  The first paper
    cited becomes ``[1]``, the next distinct paper ``[2]``, and so on.  Papers
    never cited are dropped from the reference list.
    """
    refs = list(literature_refs or [])
    if not refs:
        return [], literature_review, final_impression

    appearance = citation_first_appearance_order(literature_review, final_impression)
    valid = [n for n in appearance if 1 <= n <= len(refs)]
    if not valid:
        # Cited-only bibliography: if the prose has no [n] markers, drop all
        # retrieved papers so the References section can say none were cited.
        logger.info(
            "sync_cited_literature: no in-text citations; clearing %d uncited refs",
            len(refs),
        )
        return [], literature_review, final_impression

    mapping = {old: new for new, old in enumerate(valid, 1)}
    pruned = [refs[old - 1] for old in valid]
    new_review = remap_citation_brackets(literature_review, mapping)
    new_impression = remap_citation_brackets(final_impression, mapping)
    # Collapse whitespace left by dropped invalid citations.
    if new_review:
        new_review = re.sub(r"[ \t]{2,}", " ", new_review)
        new_review = re.sub(r" +([.,;:])", r"\1", new_review)
    if new_impression:
        new_impression = re.sub(r"[ \t]{2,}", " ", new_impression)
        new_impression = re.sub(r" +([.,;:])", r"\1", new_impression)

    logger.info(
        "sync_cited_literature: kept %d/%d cited refs (order %s → %s)",
        len(pruned),
        len(refs),
        valid,
        list(range(1, len(pruned) + 1)),
    )
    return pruned, new_review, new_impression


def extract_analysis_stats_from_entries(
    analysis_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Pull analysis_type → statistics mapping from analysis_summary entries."""
    stats_map: Dict[str, Any] = {}
    if not isinstance(analysis_data, dict):
        return stats_map
    for entry in analysis_data.get("entries", []):
        atype = entry.get("analysis_type")
        stats = entry.get("statistics")
        if atype and isinstance(stats, dict) and stats:
            stats_map[atype] = stats
    return stats_map


@tool
def search_pubmed(
    query: str,
    max_results: int = 10,
    include_abstracts: bool = True,
    max_age_years: Optional[int] = 10
) -> Dict[str, Any]:
    """
    Search PubMed for scientific literature.
    
    Uses the NCBI E-utilities API to search PubMed and retrieve
    article metadata, abstracts, and citations.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        include_abstracts: Whether to fetch full abstracts
        max_age_years: Limit to papers from last N years (None for no limit)
    
    Returns:
        Dict with search results and metadata
    """
    try:
        from Bio import Entrez
        Entrez.email = "anupkprasad121@gmail.com"  # Required by NCBI
        
        # Build search term with date filter
        search_term = query
        if max_age_years:
            from datetime import datetime
            current_year = datetime.now().year
            min_year = current_year - max_age_years
            search_term += f" AND {min_year}[PDAT]:{current_year}[PDAT]"
        
        logger.info(f"Searching PubMed: {search_term}")

        import time
        from urllib.error import HTTPError

        search_results = None
        last_error: Optional[Exception] = None
        for attempt in range(4):
            try:
                time.sleep(0.4)
                handle = Entrez.esearch(
                    db="pubmed",
                    term=search_term,
                    retmax=max_results,
                    sort="relevance"
                )
                search_results = Entrez.read(handle)
                handle.close()
                break
            except HTTPError as http_exc:
                last_error = http_exc
                if getattr(http_exc, "code", None) != 429 or attempt == 3:
                    raise
                wait = (2, 5, 12)[attempt]
                logger.warning("PubMed rate limit (429); retrying in %ss", wait)
                time.sleep(wait)
        if search_results is None:
            raise last_error or RuntimeError("PubMed search failed")
        
        pmids = search_results.get("IdList", [])
        logger.info(f"Found {len(pmids)} results")
        
        if not pmids:
            return {
                "success": True,
                "query": query,
                "total_found": 0,
                "results": []
            }
        
        # Fetch article details
        if include_abstracts:
            handle = Entrez.efetch(
                db="pubmed",
                id=pmids,
                rettype="medline",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            
            # Article types to skip (not research papers)
            _SKIP_TYPES = {
                "highlights", "reply", "editorial", "comment",
                "erratum", "retraction", "letter", "news",
                "correction", "special issue",
            }

            articles = []
            for article in records.get("PubmedArticle", []):
                try:
                    medline = article.get("MedlineCitation", {})
                    article_data = medline.get("Article", {})

                    # Skip non-research article types
                    raw_title = str(article_data.get("ArticleTitle", "")).strip().rstrip(".")
                    if raw_title.lower() in _SKIP_TYPES:
                        logger.debug("Skipping non-research article: %s", raw_title)
                        continue
                    pub_types = article_data.get("PublicationTypeList", [])
                    pub_type_strs = {str(pt).lower() for pt in pub_types}
                    if pub_type_strs & {"editorial", "comment", "letter", "published erratum", "retraction of publication", "news"}:
                        logger.debug("Skipping by pub-type: %s", pub_type_strs)
                        continue
                    
                    # Extract authors
                    author_list = article_data.get("AuthorList", [])
                    authors = []
                    for author in author_list[:10]:  # Max 10 authors
                        if "LastName" in author and "Initials" in author:
                            authors.append(f"{author['LastName']} {author['Initials']}")
                    
                    # Extract abstract
                    abstract_data = article_data.get("Abstract", {})
                    abstract_texts = abstract_data.get("AbstractText", [])
                    abstract = " ".join([str(text) for text in abstract_texts]) if abstract_texts else None
                    
                    # Extract journal info
                    journal = article_data.get("Journal", {})
                    journal_title = journal.get("Title", "Unknown")
                    pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
                    year = pub_date.get("Year", None)
                    
                    # Extract IDs
                    pmid = medline.get("PMID", None)
                    article_ids = article.get("PubmedData", {}).get("ArticleIdList", [])
                    doi = None
                    for aid in article_ids:
                        if aid.attributes.get("IdType") == "doi":
                            doi = str(aid)
                            break
                    
                    articles.append({
                        "title": article_data.get("ArticleTitle", "Unknown"),
                        "authors": authors,
                        "journal": journal_title,
                        "year": int(year) if year else None,
                        "pmid": str(pmid) if pmid else None,
                        "doi": doi,
                        "abstract": abstract,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
            return {
                "success": True,
                "query": query,
                "total_found": len(pmids),
                "returned": len(articles),
                "results": articles
            }
        else:
            # Just return PMIDs without abstracts
            return {
                "success": True,
                "query": query,
                "total_found": len(pmids),
                "pmids": [str(pmid) for pmid in pmids],
                "results": []
            }
            
    except ImportError:
        logger.error("Biopython not installed - PubMed search unavailable")
        return {
            "success": False,
            "error": "Biopython not installed. Install with: pip install biopython",
            "results": []
        }
    except Exception as e:
        logger.error(f"PubMed search error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


# ---------------------------------------------------------------------------
# bioRxiv / medRxiv preprint search  (via Europe PMC REST API)
# ---------------------------------------------------------------------------

@tool
def search_biorxiv(
    query: str,
    max_results: int = 5,
    max_age_years: Optional[int] = 5
) -> Dict[str, Any]:
    """
    Search bioRxiv/medRxiv preprints via the Europe PMC REST API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_age_years: Limit to preprints from last N years (None for no limit)

    Returns:
        Dict with search results and metadata
    """
    try:
        search_q = query
        if max_age_years:
            from datetime import datetime
            min_year = datetime.now().year - max_age_years
            search_q += f" FIRST_PDATE:[{min_year} TO *]"
        # SRC:PPR restricts to preprints (bioRxiv + medRxiv)
        search_q += " SRC:PPR"

        params = urlencode({
            "query": search_q,
            "format": "json",
            "resultType": "lite",
            "pageSize": min(max_results, 25),
        })
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}"
        logger.info("Searching bioRxiv (Europe PMC): %s", search_q[:120])

        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())

        articles = []
        for item in data.get("resultList", {}).get("result", []):
            doi = item.get("doi")
            articles.append({
                "title": item.get("title", "Unknown"),
                "authors": [a.strip() for a in (item.get("authorString") or "").split(",")][:10],
                "journal": (item.get("bookOrReportDetails") or {}).get("publisher", "bioRxiv"),
                "year": int(item["pubYear"]) if item.get("pubYear") else None,
                "pmid": item.get("pmid"),
                "doi": doi,
                "abstract": item.get("abstractText"),
                "url": f"https://doi.org/{doi}" if doi else None,
                "source": "bioRxiv",
            })

        return {
            "success": True,
            "query": query,
            "total_found": data.get("hitCount", 0),
            "returned": len(articles),
            "results": articles,
        }
    except (URLError, OSError) as e:
        logger.warning("bioRxiv search network error: %s", e)
        return {"success": False, "error": str(e), "results": []}
    except Exception as e:
        logger.error("bioRxiv search error: %s", e, exc_info=True)
        return {"success": False, "error": str(e), "results": []}


def search_europe_pmc(
    query: str,
    max_results: int = 8,
    max_age_years: Optional[int] = 12,
    open_access_only: bool = True,
) -> Dict[str, Any]:
    """
    Search Europe PMC for peer-reviewed literature (includes many open-access journals).

    Complements PubMed with PMC full-text metadata and broader OA coverage.
    """
    try:
        search_q = query
        if max_age_years:
            from datetime import datetime
            min_year = datetime.now().year - max_age_years
            search_q += f" FIRST_PDATE:[{min_year} TO *]"
        if open_access_only:
            search_q += " OPEN_ACCESS:Y"
        # Exclude preprints (handled by search_biorxiv)
        search_q += " NOT SRC:PPR"

        params = urlencode({
            "query": search_q,
            "format": "json",
            "resultType": "core",
            "pageSize": min(max_results, 25),
        })
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}"
        logger.info("Searching Europe PMC: %s", search_q[:120])

        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())

        articles = []
        for item in data.get("resultList", {}).get("result", []):
            doi = item.get("doi")
            pmid = item.get("pmid")
            pmcid = item.get("pmcid")
            articles.append({
                "title": item.get("title", "Unknown"),
                "authors": [a.strip() for a in (item.get("authorString") or "").split(",")][:10],
                "journal": item.get("journalTitle") or item.get("bookOrReportDetails", {}).get("publisher", ""),
                "year": int(item["pubYear"]) if item.get("pubYear") else None,
                "pmid": str(pmid) if pmid else None,
                "pmcid": str(pmcid) if pmcid else None,
                "doi": doi,
                "is_open_access": str(item.get("isOpenAccess", "")).upper() == "Y",
                "abstract": item.get("abstractText"),
                "url": (
                    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    if pmid
                    else (f"https://doi.org/{doi}" if doi else None)
                ),
                "source": "EuropePMC",
            })

        return {
            "success": True,
            "query": query,
            "total_found": data.get("hitCount", 0),
            "returned": len(articles),
            "results": articles,
        }
    except (URLError, OSError) as e:
        logger.warning("Europe PMC search network error: %s", e)
        return {"success": False, "error": str(e), "results": []}
    except Exception as e:
        logger.error("Europe PMC search error: %s", e, exc_info=True)
        return {"success": False, "error": str(e), "results": []}


# ---------------------------------------------------------------------------
# UniProt protein entry search
# ---------------------------------------------------------------------------

def _uniprot_name_value(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("value") or "").strip()
    return str(node or "").strip()


def _parse_uniprot_aliases(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract gene, protein names, synonyms, and alias list from a UniProt entry."""
    accession = item.get("primaryAccession", "") or ""
    prot_desc = item.get("proteinDescription", {}) or {}
    rec_name = prot_desc.get("recommendedName", {}) or {}
    pname = _uniprot_name_value(rec_name.get("fullName")) or accession
    short_names: List[str] = []
    for sn in rec_name.get("shortNames") or []:
        val = _uniprot_name_value(sn)
        if val:
            short_names.append(val)

    alt_names: List[str] = []
    for alt in prot_desc.get("alternativeNames") or []:
        if not isinstance(alt, dict):
            continue
        full = _uniprot_name_value(alt.get("fullName"))
        if full:
            alt_names.append(full)
        for sn in alt.get("shortNames") or []:
            val = _uniprot_name_value(sn)
            if val:
                alt_names.append(val)

    gene_name = ""
    gene_synonyms: List[str] = []
    for gene_block in item.get("genes") or []:
        if not isinstance(gene_block, dict):
            continue
        primary = _uniprot_name_value(gene_block.get("geneName"))
        if primary and not gene_name:
            gene_name = primary
        for syn in gene_block.get("synonyms") or []:
            val = _uniprot_name_value(syn)
            if val:
                gene_synonyms.append(val)

    aliases: List[str] = []
    seen_alias: set = set()

    def _add_alias(val: str) -> None:
        v = (val or "").strip()
        if not v:
            return
        key = v.lower()
        if key in seen_alias:
            return
        seen_alias.add(key)
        aliases.append(v)

    _add_alias(gene_name)
    for g in gene_synonyms:
        _add_alias(g)
    _add_alias(pname)
    for n in short_names + alt_names:
        _add_alias(n)
    if accession:
        _add_alias(accession)

    return {
        "accession": accession,
        "gene_name": gene_name,
        "gene_synonyms": gene_synonyms,
        "protein_name": pname,
        "short_names": short_names,
        "alternative_names": alt_names,
        "aliases": aliases,
    }


@tool
def search_uniprot(
    query: str,
    max_results: int = 3
) -> Dict[str, Any]:
    """
    Search UniProt for reviewed protein entries matching a query.
    Returns functional annotations, gene/protein aliases, and key literature
    references associated with each protein entry.

    Args:
        query: Protein name or keyword to search (e.g. "pseudokinase")
        max_results: Maximum number of UniProt entries to return

    Returns:
        Dict with protein entries containing function, aliases, references, and links
    """
    try:
        params = urlencode({
            "query": f"({query}) AND reviewed:true",
            "format": "json",
            "size": min(max_results, 10),
            "fields": (
                "accession,id,gene_names,protein_name,organism_name,"
                "cc_function,lit_pubmed_id"
            ),
        })
        url = f"https://rest.uniprot.org/uniprotkb/search?{params}"
        logger.info("Searching UniProt: %s", query[:100])

        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())

        entries = []
        for item in data.get("results", []):
            parsed = _parse_uniprot_aliases(item)
            accession = parsed["accession"]
            pname = parsed["protein_name"]

            organism = (item.get("organism") or {}).get("scientificName", "")

            func_text = ""
            for comment in item.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    for txt in comment.get("texts", []):
                        func_text += txt.get("value", "") + " "
            func_text = func_text.strip()

            ref_pmids = []
            ref_details = []
            for ref in item.get("references", []):
                cit = ref.get("citation", {})
                pmid = None
                doi = None
                for xref in cit.get("citationCrossReferences", []):
                    if xref.get("database") == "PubMed":
                        pmid = xref.get("id")
                    elif xref.get("database") == "DOI":
                        doi = xref.get("id")
                if pmid and pmid not in ref_pmids:
                    ref_pmids.append(pmid)
                    ref_details.append({
                        "pmid": pmid,
                        "doi": doi,
                        "title": cit.get("title", ""),
                        "journal": cit.get("journal", ""),
                        "year": int(cit["publicationDate"][:4]) if cit.get("publicationDate") else None,
                        "source": "uniprot_curated",
                    })

            entries.append({
                "accession": accession,
                "gene_name": parsed["gene_name"],
                "gene_synonyms": parsed["gene_synonyms"],
                "protein_name": pname,
                "short_names": parsed["short_names"],
                "alternative_names": parsed["alternative_names"],
                "aliases": parsed["aliases"],
                "organism": organism,
                "function": func_text[:500] if func_text else None,
                "url": f"https://www.uniprot.org/uniprot/{accession}",
                "reference_count": len(ref_pmids),
                "key_references": ref_details[:8],
            })

        return {
            "success": True,
            "query": query,
            "total_entries": len(entries),
            "entries": entries,
        }
    except (URLError, OSError) as e:
        logger.warning("UniProt search network error: %s", e)
        return {"success": False, "error": str(e), "entries": []}
    except Exception as e:
        logger.error("UniProt search error: %s", e, exc_info=True)
        return {"success": False, "error": str(e), "entries": []}


@tool
def generate_literature_queries(
    analysis_types: List[str],
    user_goal: Optional[str] = None,
    protein_name: Optional[str] = None,
    analysis_stats: Optional[Dict[str, Any]] = None,
    protein_family: Optional[str] = None,
    related_terms: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None,
    gene_name: Optional[str] = None,
    uniprot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate PubMed search queries across alias, family, and metric lanes.

    Query priority order:
      1. Alias × dynamics / structure-function (gene + synonyms + full name)
      2. Family / mechanism (papers that never name this gene)
      3. Metric / finding themes linked to observed results
      4. Generic analysis method queries (last resort fallback)

    Args:
        analysis_types: List of analysis types performed (e.g., ["RMSD", "RMSF"])
        user_goal: User's original goal/question (hypothesis extraction)
        protein_name: Explicit protein / system name (e.g. "CDK2", "TP53").
        aliases: Optional UniProt / synonym alias pack for multi-name search.
        gene_name: Optional primary gene symbol.
        uniprot_id: Optional accession for identity context.
        analysis_stats: Optional dict mapping analysis_type → stats.

    Returns:
        Dict with generated queries keyed by priority label
    """
    context = extract_research_context(
        user_goal=user_goal,
        protein_name=protein_name,
        analysis_types=analysis_types,
        analysis_stats=analysis_stats,
        protein_family=protein_family,
        related_terms=related_terms,
        aliases=aliases,
        gene_name=gene_name,
        uniprot_id=uniprot_id,
    )

    # ── 1. Resolve protein / gene name (reject placeholders like "domain") ─
    _PLACEHOLDER = {
        "domain", "kinase", "pseudokinase", "protein", "system", "apo", "holo",
        "catalytic", "region", "loop", "structure", "model", "complex", "unknown",
        "target", "query", "residue", "residues",
        "preprocess", "preprocessing", "setup", "simsetup", "analysis",
        "reporter", "planner", "supervisor", "gyration",
        "dynamics", "dynamic", "binding", "effect", "study", "simulation",
        "molecular", "conformational", "activation", "allosteric",
        "regulation", "overlay", "comparative", "flexibility", "human",
    }

    def _usable_name(name: Optional[str]) -> str:
        n = (name or "").strip()
        if not n:
            return ""
        if n.lower() in _PLACEHOLDER:
            return ""
        parts = [p for p in re.split(r"[_\s]+", n.lower()) if p]
        if parts and all(p in _PLACEHOLDER for p in parts):
            return ""
        return n

    _raw_name = _usable_name(protein_name)
    if not _raw_name:
        for cand in context.get("protein_names") or []:
            _raw_name = _usable_name(cand)
            if _raw_name:
                break
    if not _raw_name:
        _raw_name = _usable_name(context.get("primary_gene"))

    # Known functional-context keywords that add biological specificity
    _FUNC_KEYWORDS = [
        "kinase", "pseudokinase", "protease", "receptor", "channel",
        "transporter", "enzyme", "inhibitor", "ligand", "antibody",
        "chaperone", "transferase", "synthase", "reductase", "oxidase",
        "dehydrogenase", "hydrolase", "lyase", "isomerase", "polymerase",
        "neurodegeneration", "autoregulation", "allosteric", "doublecortin",
    ]

    # Collect functional context words from both protein_name and goal
    goal_lower = (user_goal or "").lower()
    name_lower = _raw_name.lower()
    apo_only_goal = bool(
        re.search(
            r"\b(?:no|without)\s+ligand\b|\bprotein[- ]only\b"
            r"|\bapo\b(?![^.\n]{0,60}\bholo\b)",
            goal_lower,
        )
    )
    func_context = []
    for kw in _FUNC_KEYWORDS:
        if apo_only_goal and kw in {"ligand", "inhibitor", "antibody"}:
            continue
        if kw in name_lower or kw in goal_lower:
            func_context.append(kw)
    for term in context.get("related_terms") or []:
        t = str(term).strip().lower()
        if t and t not in func_context:
            func_context.append(t)
    func_context = list(dict.fromkeys(func_context))

    # Decide the primary search name: never fall back to bare "kinase"/"domain"
    if _raw_name:
        _name = _raw_name
    elif user_goal:
        _STOP = {
            "analysis", "please", "could", "would", "should", "compute",
            "calculate", "analyse", "analyze", "trajectory", "simulation",
            "dynamics", "molecular", "protein", "report", "working",
            "generate", "plots", "values", "steps", "required",
            "additional", "preprocessing", "given", "initial", "structure",
            "already", "finished", "trajectory", "finised", "reporter",
            "agent", "scientific", "based", "comments", "result",
            "those", "data", "secondary", "calculate", "only", "download",
            "alphafold", "uniprot", "human", "extract",
        } | _PLACEHOLDER
        # Prefer gene-like tokens (letters + optional digits, e.g. DCLK3)
        gene_hits = re.findall(r"\b([A-Z][A-Za-z0-9]{2,20})\b", user_goal or "")
        _name = ""
        for g in gene_hits:
            if g.lower() not in _STOP and not re.match(
                r"^[A-Za-z][0-9][A-Za-z0-9]{3,9}$", g
            ):
                _name = g
                break
        if not _name:
            goal_words = [
                w for w in goal_lower.replace(",", " ").replace(".", " ").split()
                if len(w) > 3 and w not in _STOP
            ]
            _name = goal_words[0] if goal_words else ""
    else:
        _name = ""

    # ── 2. Build queries in priority order ───────────────────────────────
    queries = {}  # Ordered dict (Python 3.7+)

    # Alias pack: gene + synonyms + full names (prefer gene-like first)
    alias_candidates: List[str] = []
    for cand in (
        [gene_name or "", _name]
        + list(aliases or [])
        + list(context.get("aliases") or [])
        + list(context.get("protein_names") or [])
    ):
        u = _usable_name(cand)
        if u and u not in alias_candidates:
            alias_candidates.append(u)
    # Prefer compact gene symbols first; demote UniProt accessions and long names
    def _alias_rank(n: str) -> tuple:
        is_acc = bool(re.fullmatch(r"[OPQABCD][0-9][A-Z0-9]{3}[0-9]", n, re.I))
        is_gene = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,14}", n)) and not is_acc
        return (0 if is_gene else 1 if not is_acc else 2, len(n))

    alias_candidates.sort(key=_alias_rank)
    study_genes = [
        n for n in (context.get("study_genes") or []) if _usable_name(n)
    ]
    # A multi-protein study searches each named protein, not the word "dynamics"
    # and not a stack of generic family reviews.
    if len(study_genes) >= 2:
        queries: Dict[str, str] = {}
        for pname in study_genes:
            quoted = f'"{pname}"' if " " in pname else pname
            queries[f"{pname}_md"] = (
                f"{quoted} (pseudokinase OR kinase) "
                f"(\"molecular dynamics\" OR \"activation loop\" OR allosteric OR ATP)"
            )
            queries[f"{pname}_function"] = (
                f"{quoted} (structure OR conformation OR dynamics) "
                f"(activation OR allosteric OR regulation)"
            )
        return {
            "success": True,
            "queries": queries,
            "protein_name": ", ".join(study_genes),
            "aliases": study_genes,
            "research_context": context,
        }

    usable_names = alias_candidates[:6]
    if _name and _name not in usable_names:
        usable_names.insert(0, _name)

    # Lane A — Alias × dynamics / functional / computational study
    for idx, pname in enumerate(usable_names[:5]):
        quoted = f'"{pname}"' if (" " in pname or "-" in pname) else pname
        queries[f"alias_{idx}_dynamics"] = f"{quoted} molecular dynamics"
        queries[f"alias_{idx}_flexibility"] = f"{quoted} conformational flexibility"
        queries[f"alias_{idx}_functional"] = f"{quoted} functional study"
        queries[f"alias_{idx}_computational"] = (
            f"{quoted} (computational OR \"molecular dynamics\" OR simulation)"
        )
        queries[f"protein_{idx}_dynamics"] = f"{pname} protein dynamics molecular dynamics"
        queries[f"protein_{idx}_mechanism"] = (
            f"{pname} mechanism structure function molecular dynamics"
        )
        if func_context:
            queries[f"protein_{idx}_function"] = (
                f"{pname} {' '.join(func_context[:2])} structure dynamics"
            )

    if _name:
        quoted_name = f'"{_name}"' if (" " in _name or "-" in _name) else _name
        queries["protein_dynamics"] = f"{quoted_name} molecular dynamics simulation"
        queries["protein_function"] = f"{quoted_name} structure function"
        queries["protein_functional_study"] = (
            f"{quoted_name} functional study OR deficiency OR knockout"
        )
        queries["protein_computational_study"] = (
            f"{quoted_name} computational study OR molecular dynamics OR simulation"
        )
        if func_context:
            fc = " ".join(func_context[:2])
            queries["protein_context"] = f"{quoted_name} {fc} molecular dynamics"
        hyp = [
            t for t in (context.get("hypothesis_terms") or [])
            if t and len(str(t)) > 3
        ][:3]
        if hyp:
            queries["protein_goal_context"] = f"{_name} {' '.join(hyp)}"
        bio = [
            t for t in func_context
            if t not in {"kinase", "pseudokinase", "enzyme", "protein", "ligand"}
        ][:2]
        if bio:
            queries["protein_biology"] = f"{_name} {' '.join(bio)}"

    family = (context.get("protein_family") or "").strip()
    if _name and family:
        queries["protein_family_review"] = f"{_name} {family} structure function review"

    # Lane B — Family / mechanism (no gene required)
    if family:
        queries["theme_family_dynamics"] = (
            f"{family} conformational dynamics molecular dynamics"
        )
        queries["theme_family_flexibility"] = (
            f"{family} apo flexibility residue dynamics"
        )
    for term in (context.get("related_terms") or [])[:3]:
        t = str(term).strip()
        if t and t.lower() not in {family.lower()} and len(t) > 4:
            queries[f"theme_related_{t[:24].replace(' ', '_')}"] = (
                f"{t} molecular dynamics conformational"
            )

    # Ligand / ATP effect queries from user goal
    if context["ligand_terms"] and _name:
        lig = context["ligand_terms"][0]
        queries["ligand_effect"] = f"{_name} {lig} binding protein dynamics simulation"

    # Region-specific queries (e.g. activation loop residues 150-190)
    if context["region_terms"] and _name:
        region = context["region_terms"][0]
        queries["region_dynamics"] = f"{_name} {region} flexibility dynamics"

    # DCCM / allosteric when requested or computed
    if any("dccm" in a.lower() or "correlation" in a.lower()
           for a in (analysis_types or [])) or "dccm" in goal_lower:
        if _name:
            queries["allosteric_dccm"] = (
                f"{_name} allosteric communication correlated motion molecular dynamics"
            )

    # ── 3. Analysis method queries (protein-preferred + theme fallback) ──
    _METHOD_TEMPLATES = {
        "RMSD": "RMSD stability molecular dynamics",
        "RMSF": "RMSF residue flexibility dynamics",
        "Radius_of_Gyration": "radius of gyration compactness",
        "Energy": "energy stability thermodynamic",
        "SASA": "solvent accessible surface area",
        "Secondary_Structure": "secondary structure DSSP",
        "COM_Analysis": "center of mass trajectory",
    }

    _SKIP_ANALYSIS_TYPES = {
        "pdb_input_validation", "trajectorybatch", "trajectory_batch",
        "unknown", "n/a", "none",
    }
    for atype in analysis_types:
        if not atype or str(atype).lower().replace(" ", "_") in _SKIP_ANALYSIS_TYPES:
            continue
        method_terms = None
        for tkey, tval in _METHOD_TEMPLATES.items():
            if tkey.lower() in atype.lower() or atype.lower() in tkey.lower():
                method_terms = tval
                break
        if method_terms is None:
            # Skip opaque internal labels (CamelCase batch markers, etc.)
            if not any(c.islower() for c in atype) and "_" not in atype and " " not in atype:
                continue
            method_terms = f"{atype} molecular dynamics"

        if _name:
            queries[f"method_{atype}"] = f"{_name} {method_terms}"
        # Lane C — metric / finding themes (optionally scoped by family)
        fam_prefix = f"{family} " if family else ""
        queries[f"theme_method_{atype}"] = f"{fam_prefix}{method_terms}".strip()

    # ── 4. Analysis-result–driven queries ────────────────────────────────
    stats = analysis_stats or {}
    fam_bit = family or (func_context[0] if func_context else "protein")
    if stats:
        for key in stats:
            if "rmsd" in key.lower():
                mean_rmsd = stats[key].get("mean_rmsd_angstrom") or stats[key].get("mean", 0)
                if mean_rmsd and float(mean_rmsd) > 3.0:
                    theme = f"{fam_bit} conformational instability RMSD molecular dynamics"
                else:
                    theme = f"{fam_bit} conformational stability RMSD molecular dynamics"
                if _name:
                    queries["result_rmsd"] = f"{_name} {theme}"
                queries["theme_result_rmsd"] = theme
                break

        for key in stats:
            if "rmsf" in key.lower():
                mean_rmsf = stats[key].get("mean_rmsf_angstrom") or stats[key].get("mean", 0)
                if mean_rmsf and float(mean_rmsf) > 2.0:
                    theme = f"{fam_bit} flexible loop region RMSF molecular dynamics"
                else:
                    theme = f"{fam_bit} residue flexibility RMSF domain molecular dynamics"
                if _name:
                    queries["result_rmsf"] = f"{_name} {theme}"
                queries["theme_result_rmsf"] = theme
                break

        for key in stats:
            if "rg" in key.lower() or "gyration" in key.lower():
                mean_rg = stats[key].get("mean_rg_angstrom") or stats[key].get("mean", 0)
                if mean_rg and float(mean_rg) > 25.0:
                    theme = f"{fam_bit} expanded conformation radius of gyration molecular dynamics"
                elif mean_rg and float(mean_rg) < 15.0:
                    theme = f"{fam_bit} compact folded structure radius of gyration dynamics"
                else:
                    theme = f"{fam_bit} radius of gyration domain stability molecular dynamics"
                if theme:
                    if _name:
                        queries["result_rg"] = f"{_name} {theme}"
                    queries["theme_result_rg"] = theme
                break

    # ── 5. Objective / family / class queries (not gene-limited) ─────────
    hyp_terms = [
        t for t in (context.get("hypothesis_terms") or [])
        if t and len(str(t)) > 3
    ][:4]
    if hyp_terms:
        hyp_str = " ".join(hyp_terms[:3])
        if _name:
            queries["hypothesis_objective"] = f"{_name} {hyp_str} molecular dynamics"
        queries["theme_objective"] = f"{hyp_str} molecular dynamics protein structure"

    # Family / functional-class literature (kinase domain MD, etc.)
    class_bits = []
    if family:
        class_bits.append(family)
    for kw in func_context[:3]:
        if kw not in class_bits:
            class_bits.append(kw)
    if class_bits:
        queries["theme_family_dynamics"] = (
            f"{' '.join(class_bits[:2])} domain molecular dynamics simulation"
        )
        queries["theme_family_structure"] = (
            f"{' '.join(class_bits[:2])} structure function conformational dynamics"
        )

    if context.get("ligand_terms"):
        lig = context["ligand_terms"][0]
        queries["theme_ligand"] = f"{lig} binding protein molecular dynamics simulation"
    if context.get("region_terms"):
        region = context["region_terms"][0]
        queries["theme_region"] = f"protein {region} flexibility molecular dynamics"

    # ── 6. Generic fallback ───────────────────────────────────────────────
    queries["_fallback_general"] = "molecular dynamics protein conformational stability review"

    logger.info(
        "Generated %d literature queries (protein=%r, func_context=%s)",
        len(queries), _name or "unknown", func_context,
    )

    return {
        "success": True,
        "queries": queries,
        "total": len(queries),
        "protein_name": _name or None,
        "aliases": usable_names,
        "research_context": context,
    }
