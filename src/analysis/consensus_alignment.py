"""
Consensus sequence alignment for cross-simulation coordinate mapping.

Builds a global MAFFT MSA, keeps columns that pass physicochemical similarity
and occupancy filters, and exports a compact residue map for transferable
family analyses (RMSF / DCCM / shared-PC / pocket transfer).

Inputs (any one):
  * ``pdb_files`` + ``labels``
  * ``fasta_file``
  * ``sim_dirs`` + ``labels`` (PDB resolved like phylo_tree tools)

Outputs (under ``working_dir``):
  * ``global_msa.fasta`` — full gapped MAFFT alignment (sequence letter
    source for plotting / transfer; plots show consensus/pocket columns only)
  * ``global_consensus_msa.json`` / ``.csv`` — consensus columns with
    occupancy ≥ min_coverage **and** similarity (or identity) ≥ min_conservation
  * Legacy aliases: ``global_mapped.json``, ``consensus_residues.json``,
    ``reference_msa_alignment.*``

Pipeline note: the full unfiltered column JSON (``global_msa.json``) is **not**
required. Pocket transfer is ``15 Å reference shell ∩ consensus columns`` —
mathematically identical to filtering the full map then intersecting.
MSA plots use ``global_consensus_msa`` and ``pocket_mapped`` columns only.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain.tools import tool

from src.preprocess.structure_remodel.sequence_utils import (
    ChainSequence,
    extract_chain_sequences,
    pick_default_chain,
)

logger = logging.getLogger(__name__)

try:
    from Bio.Align import PairwiseAligner, substitution_matrices

    HAS_BIO = True
except Exception:  # pragma: no cover
    HAS_BIO = False

DEFAULT_ALIGNMENT_FASTA = "global_msa.fasta"
DEFAULT_RESIDUE_MAP_CSV = "global_consensus_msa.csv"
DEFAULT_ALIGNMENT_JSON = "global_consensus_msa.json"
# Optional debug dump of every MAFFT column (not needed for paper pipeline).
DEFAULT_FULL_MSA_JSON = "global_msa.json"
# Legacy names still written / accepted so in-flight campaigns keep working.
LEGACY_ALIGNMENT_JSON_ALIASES = (
    "global_mapped.json",
    "reference_msa_alignment.json",
    "consensus_residues.json",
)
LEGACY_RESIDUE_MAP_CSV_ALIASES = (
    "global_mapped.csv",
    "reference_msa_residue_map.csv",
    "reference_residue_map.csv",
)
LEGACY_ALIGNMENT_FASTA = "reference_msa_alignment.fasta"
LEGACY_ALIGNMENT_JSON = "reference_msa_alignment.json"
LEGACY_RESIDUE_MAP_CSV = "reference_msa_residue_map.csv"
LEGACY_CONSENSUS_JSON = "consensus_residues.json"
LEGACY_MAPPED_JSON = "global_mapped.json"
LEGACY_MAPPED_CSV = "global_mapped.csv"
DEFAULT_MSA_METHOD = "mafft"
DEFAULT_CONSERVATION_METRIC = "similarity"
DEFAULT_MIN_CONSERVATION = 0.5
# Paper occupancy floor for physicochemical-group similarity columns.
DEFAULT_MIN_COVERAGE = 0.25
MAPPING_KIND_CONSENSUS = "global_consensus_msa"

# Physicochemical groups for ``similarity`` (paper / ment mapping_residues).
AA_GROUPS: Dict[str, str] = {
    "A": "hydrophobic", "V": "hydrophobic", "L": "hydrophobic", "I": "hydrophobic",
    "M": "hydrophobic", "G": "hydrophobic", "P": "hydrophobic", "C": "hydrophobic",
    "F": "aromatic", "Y": "aromatic", "W": "aromatic",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar",
    "D": "acidic", "E": "acidic",
    "K": "basic", "R": "basic", "H": "basic",
}

_MAFFT_CANDIDATES = (
    "/apps/gb/multi-bacpipe/0.8.0/libexec/t-coffee-13.46.0.919e8c6b-4/plugins/linux/mafft",
    "/apps/gb/multi-bacpipe/0.8.0/lib/t_coffee-11.0.8/plugins/linux/mafft",
)


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON export."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ── Sequence loading ──────────────────────────────────────────────────────────


def _build_aligner() -> "PairwiseAligner":
    aligner = PairwiseAligner()
    aligner.mode = "global"
    try:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception:
        pass
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner


def load_sequences_from_fasta(fasta_file: str) -> Dict[str, ChainSequence]:
    """Parse a FASTA file into label → ChainSequence (1-based pseudo resids)."""
    path = Path(fasta_file)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    chains: Dict[str, ChainSequence] = {}
    label: Optional[str] = None
    chunks: List[str] = []

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label and chunks:
                    seq = "".join(chunks)
                    chains[label] = ChainSequence(
                        chain_id="A",
                        sequence=seq,
                        resids=list(range(1, len(seq) + 1)),
                        resnames=["UNK"] * len(seq),
                    )
                label = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.replace(" ", "").upper())

    if label and chunks:
        seq = "".join(chunks)
        chains[label] = ChainSequence(
            chain_id="A",
            sequence=seq,
            resids=list(range(1, len(seq) + 1)),
            resnames=["UNK"] * len(seq),
        )

    if not chains:
        raise ValueError(f"No sequences parsed from {fasta_file}")
    return chains


def load_sequences_from_pdbs(
    pdb_files: Sequence[str],
    labels: Sequence[str],
    chain_id: Optional[str] = None,
) -> Dict[str, ChainSequence]:
    """Extract the longest (or chosen) protein chain from each PDB."""
    if len(pdb_files) != len(labels):
        raise ValueError("pdb_files and labels must have the same length")

    chains: Dict[str, ChainSequence] = {}
    for pdb, label in zip(pdb_files, labels):
        all_chains = extract_chain_sequences(pdb, chain_id=chain_id)
        cid = chain_id or pick_default_chain(all_chains)
        chains[str(label)] = all_chains[cid]
    return chains


def _resolve_pdbs_from_sims(
    sim_dirs: Sequence[str],
    labels: Sequence[str],
    base_dir: str,
) -> Tuple[List[str], List[str], List[str]]:
    from src.analysis.phylo_tree import resolve_structure_pdb

    pdbs: List[str] = []
    used_labels: List[str] = []
    missing: List[str] = []
    for sim_dir, label in zip(sim_dirs, labels):
        pdb = resolve_structure_pdb(str(label), str(sim_dir), base_dir)
        if pdb:
            pdbs.append(pdb)
            used_labels.append(str(label))
        else:
            missing.append(str(label))
    return pdbs, used_labels, missing


# ── Star MSA to reference ─────────────────────────────────────────────────────


def _pairwise_gapped_strings(
    ref: ChainSequence,
    other: ChainSequence,
    aligner: "PairwiseAligner",
) -> Tuple[str, str, List[Tuple[Optional[int], Optional[int]]]]:
    """Return gapped strings and (ref_idx, other_idx) pairs per alignment column."""
    if not ref.sequence or not other.sequence:
        return "", "", []

    aln = aligner.align(ref.sequence, other.sequence)[0]
    indices = aln.indices
    if indices is None or len(indices) != 2:
        return "", "", []

    ref_path, oth_path = indices[0], indices[1]
    ref_chars: List[str] = []
    oth_chars: List[str] = []
    pairs: List[Tuple[Optional[int], Optional[int]]] = []

    for ri, oi in zip(ref_path, oth_path):
        if ri != -1:
            ref_chars.append(ref.sequence[ri])
        else:
            ref_chars.append("-")
        if oi != -1:
            oth_chars.append(other.sequence[oi])
        else:
            oth_chars.append("-")
        pairs.append(
            (ri if ri != -1 else None, oi if oi != -1 else None)
        )

    return "".join(ref_chars), "".join(oth_chars), pairs


def _reference_indexed_row(
    ref: ChainSequence,
    other: ChainSequence,
    pairs: List[Tuple[Optional[int], Optional[int]]],
) -> str:
    """Pad ``other`` to reference length using matched residue pairs (gaps elsewhere)."""
    chars = ["-"] * len(ref.sequence)
    for ref_idx, oth_idx in pairs:
        if ref_idx is not None and oth_idx is not None:
            chars[ref_idx] = other.sequence[oth_idx]
    return "".join(chars)


def build_star_msa_to_reference(
    chains: Dict[str, ChainSequence],
    reference_label: str,
) -> Dict[str, Any]:
    """
    Build a star alignment anchored on ``reference_label``.

    Each sequence is globally aligned to the reference independently.
    Consensus columns are keyed by **reference residue index** (one column per
    reference residue). FASTA rows are reference-length strings (gap = no
    aligned residue in that protein).
    """
    if reference_label not in chains:
        raise KeyError(f"Reference label {reference_label!r} not in chain set")

    aligner = _build_aligner()
    ref = chains[reference_label]
    other_labels = [k for k in sorted(chains.keys()) if k != reference_label]
    labels = [reference_label] + other_labels

    columns: List[Dict[str, Any]] = []
    for i in range(len(ref.sequence)):
        columns.append({
            "msa_col": i,
            "reference_seq_index": i,
            "reference_resid": ref.resids[i],
            "reference_aa": ref.sequence[i],
            "mappings": {
                reference_label: {
                    "seq_index": i,
                    "resid": ref.resids[i],
                    "aa": ref.sequence[i],
                }
            },
        })

    rows: Dict[str, str] = {reference_label: ref.sequence}

    for label in other_labels:
        other = chains[label]
        ref_gapped, oth_gapped, pairs = _pairwise_gapped_strings(ref, other, aligner)
        if not pairs:
            logger.warning("Empty alignment for %s vs reference", label)
            rows[label] = "-" * len(ref.sequence)
            continue

        rows[label] = _reference_indexed_row(ref, other, pairs)
        for ref_idx, oth_idx in pairs:
            if ref_idx is None or oth_idx is None:
                continue
            columns[ref_idx]["mappings"][label] = {
                "seq_index": int(oth_idx),
                "resid": int(other.resids[oth_idx]),
                "aa": other.sequence[oth_idx],
            }

        # Store full gapped pairwise alignment width for optional inspection
        rows[f"{label}__pairwise_gapped"] = oth_gapped

    width = len(ref.sequence)
    return {
        "reference_label": reference_label,
        "msa_width": width,
        "rows": {k: v for k, v in rows.items() if not k.endswith("__pairwise_gapped")},
        "pairwise_gapped_rows": {k: v for k, v in rows.items() if k.endswith("__pairwise_gapped")},
        "columns": columns,
        "labels": labels,
    }


def normalize_conservation_metric(metric: str) -> str:
    """Map user/LLM aliases onto an internal conservation metric.

    * ``similarity`` / ``group`` — physicochemical-group agreement with the
      modal AA (paper default).
    * ``identity`` — modal AA fraction among non-gaps.
    * ``blosum`` / ``blosum62`` — mean pairwise BLOSUM62, normalized to [0, 1].
    * ``coverage`` / ``none`` — occupancy only.
    """
    m = (metric or DEFAULT_CONSERVATION_METRIC).strip().lower()
    if m in ("similarity", "group", "group_similarity", "physchem", "physicochemical"):
        return "group"
    if m in ("blosum", "blosum62", "pairwise"):
        return "blosum"
    if m in ("identity", "id"):
        return "identity"
    if m in ("coverage", "none", "occupancy", ""):
        return "coverage"
    return m


def select_consensus_positions(
    msa: Dict[str, Any],
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    require_reference: bool = False,
    conservation_metric: str = DEFAULT_CONSERVATION_METRIC,
    min_conservation: float = DEFAULT_MIN_CONSERVATION,
) -> List[Dict[str, Any]]:
    """
    Return MSA columns meeting occupancy **and** conservation filters.

    ``conservation_metric`` (modular; user-selectable):
      * ``similarity`` — fraction of non-gap AAs in the same physicochemical
        group as the modal AA (default; paper method; threshold 0.5)
      * ``identity`` — fraction of non-gap residues matching the modal AA
      * ``blosum`` — mean pairwise BLOSUM62 score normalized to ~[0, 1]
      * ``coverage`` / ``none`` — occupancy only
    """
    labels: List[str] = msa.get("labels") or []
    n_labels = max(len(labels), 1)
    threshold = float(min_coverage) * n_labels
    metric = normalize_conservation_metric(conservation_metric)
    consensus: List[Dict[str, Any]] = []
    blosum = None
    if metric == "blosum" and HAS_BIO:
        try:
            blosum = substitution_matrices.load("BLOSUM62")
        except Exception:
            blosum = None

    for col in msa.get("columns", []):
        if require_reference and col.get("reference_seq_index") is None:
            continue
        mappings = col.get("mappings") or {}
        present = [lab for lab in labels if lab in mappings]
        if len(present) < threshold:
            continue
        aas = [
            str((mappings[lab] or {}).get("aa") or "")[:1]
            for lab in present
            if (mappings.get(lab) or {}).get("aa")
        ]
        cons_score = _column_conservation(aas, metric=metric, blosum=blosum)
        if metric in ("group", "identity", "blosum") and cons_score < float(min_conservation):
            continue
        entry = {
            "consensus_index": len(consensus),
            "msa_col": col.get("msa_col"),
            "reference_seq_index": col.get("reference_seq_index"),
            "reference_resid": col.get("reference_resid"),
            "reference_aa": col.get("reference_aa"),
            "coverage_fraction": len(present) / n_labels,
            "occupancy": len(present) / n_labels,
            "conservation": cons_score,
            "conservation_metric": metric,
            "mappings": mappings,
        }
        consensus.append(entry)

    return consensus


def _column_conservation(
    aas: Sequence[str],
    *,
    metric: str,
    blosum: Any = None,
) -> float:
    letters = [a.upper() for a in aas if a and a not in "-.X"]
    if not letters:
        return 0.0
    metric = normalize_conservation_metric(metric)
    if metric == "coverage":
        return 1.0
    if metric == "identity":
        mode, count = Counter(letters).most_common(1)[0]
        return float(count) / float(len(letters))
    if metric == "group":
        modal, _count = Counter(letters).most_common(1)[0]
        group = AA_GROUPS.get(modal)
        if not group:
            return float(_count) / float(len(letters))
        return float(sum(1 for a in letters if AA_GROUPS.get(a) == group)) / float(len(letters))
    # blosum
    if blosum is None or len(letters) == 1:
        mode, count = Counter(letters).most_common(1)[0]
        return float(count) / float(len(letters))
    scores: List[float] = []
    for i in range(len(letters)):
        for j in range(i + 1, len(letters)):
            a, b = letters[i], letters[j]
            try:
                s = float(blosum[a, b])
                na = float(blosum[a, a])
                nb = float(blosum[b, b])
            except Exception:
                continue
            denom = max(0.5 * (na + nb), 1e-6)
            scores.append(s / denom)
    if not scores:
        return 0.0
    mean = float(np.mean(scores))
    return float(max(0.0, min(1.0, mean)))


def find_mafft_executable() -> Optional[str]:
    """Locate a usable ``mafft`` binary (sets MAFFT_BINARIES when needed)."""
    env_bin = os.environ.get("MAFFT") or os.environ.get("MAFFT_BIN")
    which = shutil.which("mafft")
    for cand in (env_bin, which, *_MAFFT_CANDIDATES):
        if not cand:
            continue
        p = Path(cand)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
    return None


def _star_fallback_or_raise(
    chains: Dict[str, ChainSequence],
    reference_label: str,
    *,
    reason: str,
    allow_star_fallback: bool,
) -> Dict[str, Any]:
    """Star pairwise only when explicitly allowed; otherwise fail loudly."""
    if not allow_star_fallback:
        raise RuntimeError(
            f"{reason}. Install MAFFT (or set MAFFT/MAFFT_BIN), or pass "
            "msa_method='star_pairwise' / allow_star_fallback=True. "
            "Silent star fallback is disabled so family robustness runs cannot "
            "quietly diverge from the paper MAFFT≥0.5 protocol."
        )
    logger.warning("%s — falling back to Biopython star pairwise MSA", reason)
    msa = build_star_msa_to_reference(chains, reference_label)
    msa["msa_method"] = "star_pairwise"
    msa["msa_fallback_from"] = "mafft"
    msa["msa_fallback_reason"] = reason
    return msa


def build_mafft_msa_to_reference(
    chains: Dict[str, ChainSequence],
    reference_label: str,
    *,
    mafft_bin: Optional[str] = None,
    allow_star_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Run MAFFT on all sequences and index columns by the reference row.

    By default does **not** silently fall back to star pairwise (reproducibility).
    Pass ``allow_star_fallback=True`` only when a degraded alignment is acceptable.
    """
    if reference_label not in chains:
        raise KeyError(f"Reference label {reference_label!r} not in chain set")

    exe = mafft_bin or find_mafft_executable()
    if not exe:
        return _star_fallback_or_raise(
            chains,
            reference_label,
            reason="MAFFT not found on PATH or MAFFT/MAFFT_BIN / known candidates",
            allow_star_fallback=allow_star_fallback,
        )

    other_labels = [k for k in sorted(chains.keys()) if k != reference_label]
    labels = [reference_label] + other_labels

    with tempfile.TemporaryDirectory(prefix="mafft_msa_") as tmp:
        tmp_path = Path(tmp)
        in_fa = tmp_path / "input.fasta"
        out_fa = tmp_path / "aligned.fasta"
        with open(in_fa, "w", encoding="utf-8") as fh:
            for lab in labels:
                fh.write(f">{lab}\n")
                seq = chains[lab].sequence
                for i in range(0, len(seq), 80):
                    fh.write(seq[i : i + 80] + "\n")

        env = os.environ.copy()
        # Older mafft wrappers require sibling binaries via MAFFT_BINARIES.
        env.setdefault("MAFFT_BINARIES", str(Path(exe).resolve().parent))
        cmd = [exe, "--auto", "--quiet", str(in_fa)]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=600,
            )
        except Exception as exc:
            return _star_fallback_or_raise(
                chains,
                reference_label,
                reason=f"MAFFT failed ({exc})",
                allow_star_fallback=allow_star_fallback,
            )
        if proc.returncode != 0 or not (proc.stdout or "").strip():
            return _star_fallback_or_raise(
                chains,
                reference_label,
                reason=(
                    f"MAFFT exit={proc.returncode} "
                    f"stderr={(proc.stderr or '')[:300]}"
                ),
                allow_star_fallback=allow_star_fallback,
            )
        out_fa.write_text(proc.stdout, encoding="utf-8")
        aligned = _read_fasta_rows(out_fa)

    if reference_label not in aligned:
        # Case-insensitive header match
        for k in list(aligned.keys()):
            if k.lower() == reference_label.lower():
                aligned[reference_label] = aligned.pop(k)
                break
    if reference_label not in aligned:
        return _star_fallback_or_raise(
            chains,
            reference_label,
            reason="MAFFT output missing reference row",
            allow_star_fallback=allow_star_fallback,
        )

    ref_row = aligned[reference_label]
    width = len(ref_row)
    # Ensure equal width
    for lab in labels:
        row = aligned.get(lab, "-" * width)
        if len(row) < width:
            row = row + "-" * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        aligned[lab] = row

    # Map MSA columns → ungapped sequence indices
    ungapped_idx: Dict[str, List[Optional[int]]] = {}
    for lab in labels:
        idxs: List[Optional[int]] = []
        si = 0
        for ch in aligned[lab]:
            if ch == "-":
                idxs.append(None)
            else:
                idxs.append(si)
                si += 1
        ungapped_idx[lab] = idxs

    columns: List[Dict[str, Any]] = []
    rows: Dict[str, str] = {lab: aligned[lab] for lab in labels}

    for msa_i in range(width):
        ref_seq_i = ungapped_idx[reference_label][msa_i]
        ref_chain = chains[reference_label]
        mappings: Dict[str, Any] = {}
        if ref_seq_i is not None:
            mappings[reference_label] = {
                "seq_index": int(ref_seq_i),
                "resid": int(ref_chain.resids[ref_seq_i]),
                "aa": ref_chain.sequence[ref_seq_i],
            }
        for lab in other_labels:
            oth_i = ungapped_idx[lab][msa_i]
            if oth_i is None:
                continue
            oth = chains[lab]
            mappings[lab] = {
                "seq_index": int(oth_i),
                "resid": int(oth.resids[oth_i]),
                "aa": oth.sequence[oth_i],
            }
        columns.append({
            "msa_col": msa_i,
            "mafft_col": msa_i,
            "reference_seq_index": int(ref_seq_i) if ref_seq_i is not None else None,
            "reference_resid": (
                int(ref_chain.resids[ref_seq_i]) if ref_seq_i is not None else None
            ),
            "reference_aa": (
                ref_chain.sequence[ref_seq_i] if ref_seq_i is not None else "-"
            ),
            "mappings": mappings,
        })

    return {
        "reference_label": reference_label,
        "msa_width": width,
        "mafft_width": width,
        "rows": rows,
        "columns": columns,
        "labels": labels,
        "msa_method": "mafft",
        "mafft_bin": str(exe),
        "full_msa_rows": {lab: aligned[lab] for lab in labels},
    }


def _read_fasta_rows(path: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    label = None
    chunks: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    rows[label] = "".join(chunks)
                label = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.replace(" ", ""))
        if label is not None:
            rows[label] = "".join(chunks)
    return rows


def _mirror_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == src.resolve():
        return
    shutil.copy2(src, dest)


def write_consensus_outputs(
    msa: Dict[str, Any],
    consensus: List[Dict[str, Any]],
    out_dir: Path,
    *,
    fasta_name: str = DEFAULT_ALIGNMENT_FASTA,
    csv_name: str = DEFAULT_RESIDUE_MAP_CSV,
    json_name: str = DEFAULT_ALIGNMENT_JSON,
    full_msa_json_name: str = DEFAULT_FULL_MSA_JSON,
    all_positions: Optional[List[Dict[str, Any]]] = None,
    chains: Optional[Dict[str, ChainSequence]] = None,
    conservation_metric: str = DEFAULT_CONSERVATION_METRIC,
    min_conservation: float = DEFAULT_MIN_CONSERVATION,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    write_full_msa_json: bool = False,
) -> Dict[str, Any]:
    """Write FASTA, consensus CSV/JSON (and optionally unfiltered global_msa.json).

    Paper path writes only ``global_consensus_msa.*`` + ``global_msa.fasta``.
    Unfiltered ``global_msa.json`` is optional debug output — pocket transfer
    uses consensus columns directly (15 Å ∩ consensus), not the full map.
    """
    from src.analysis.cross_sim_artifacts import (
        compact_consensus_from_positions,
        write_json_compact,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / fasta_name
    csv_path = out_dir / csv_name
    json_path = out_dir / json_name

    ref_label = msa["reference_label"]
    row_order = [ref_label] + [l for l in msa.get("labels", []) if l != ref_label]
    with open(fasta_path, "w", encoding="utf-8") as fh:
        for label in row_order:
            seq = msa["rows"].get(label, "")
            if chains and label in chains:
                cid = chains[label].chain_id
                fh.write(f">{label} chain={cid} length={len(seq)}\n")
            else:
                fh.write(f">{label} length={len(seq)}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")

    labels = msa.get("labels") or []
    header = [
        "consensus_index",
        "msa_col",
        "reference_label",
        "reference_seq_index",
        "reference_resid",
        "reference_aa",
        "coverage_fraction",
        "conservation",
    ]
    for lab in labels:
        header.extend([f"{lab}_seq_index", f"{lab}_resid", f"{lab}_aa"])

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for pos in consensus:
            row = [
                pos["consensus_index"],
                pos.get("msa_col"),
                ref_label,
                pos.get("reference_seq_index"),
                pos.get("reference_resid"),
                pos.get("reference_aa"),
                f"{pos.get('coverage_fraction', 0):.4f}",
                f"{float(pos.get('conservation') or 0):.4f}",
            ]
            mappings = pos.get("mappings") or {}
            for lab in labels:
                m = mappings.get(lab)
                if m:
                    row.extend([m["seq_index"], m["resid"], m["aa"]])
                else:
                    row.extend(["", "", ""])
            writer.writerow(row)

    payload = compact_consensus_from_positions(
        reference_label=ref_label,
        labels=labels,
        consensus_positions=consensus,
        msa_width=msa.get("msa_width"),
        msa_method=str(msa.get("msa_method") or DEFAULT_MSA_METHOD),
        conservation_metric=conservation_metric,
        min_conservation=min_conservation,
        min_coverage=min_coverage,
        extra={
            "fasta_file": fasta_name,
            "residue_map_csv": csv_name,
            "mapping_kind": MAPPING_KIND_CONSENSUS,
            "n_global_consensus": len(consensus),
            # Legacy key retained for in-flight consumers.
            "n_global_mapped": len(consensus),
        },
    )
    write_json_compact(json_path, payload)

    result: Dict[str, Any] = {
        "fasta_file": str(out_dir / DEFAULT_ALIGNMENT_FASTA),
        "residue_map_csv": str(out_dir / DEFAULT_RESIDUE_MAP_CSV),
        "json_file": str(out_dir / DEFAULT_ALIGNMENT_JSON),
        "n_global_consensus": len(consensus),
        "n_global_mapped": len(consensus),
    }

    if write_full_msa_json and all_positions is not None:
        full_json_path = out_dir / full_msa_json_name
        full_payload = compact_consensus_from_positions(
            reference_label=ref_label,
            labels=labels,
            consensus_positions=all_positions,
            msa_width=msa.get("msa_width"),
            msa_method=str(msa.get("msa_method") or DEFAULT_MSA_METHOD),
            conservation_metric=conservation_metric,
            min_conservation=0.0,
            min_coverage=0.0,
            extra={
                "fasta_file": fasta_name,
                "mapping_kind": "global_msa",
                "n_global_consensus": len(consensus),
                "global_consensus_msa_json": json_name,
            },
        )
        if full_json_path.resolve() == json_path.resolve():
            full_json_path = out_dir / DEFAULT_FULL_MSA_JSON
        write_json_compact(full_json_path, full_payload)
        _mirror_file(full_json_path, out_dir / DEFAULT_FULL_MSA_JSON)
        result["global_msa_json"] = str(out_dir / DEFAULT_FULL_MSA_JSON)
        result["n_global_msa_columns"] = len(all_positions)

    for alias in (DEFAULT_ALIGNMENT_FASTA, LEGACY_ALIGNMENT_FASTA, "consensus_msa.fasta"):
        _mirror_file(fasta_path, out_dir / alias)
    for alias in (
        DEFAULT_ALIGNMENT_JSON,
        LEGACY_MAPPED_JSON,
        LEGACY_ALIGNMENT_JSON,
        LEGACY_CONSENSUS_JSON,
    ):
        _mirror_file(json_path, out_dir / alias)
    for alias in (
        DEFAULT_RESIDUE_MAP_CSV,
        LEGACY_MAPPED_CSV,
        LEGACY_RESIDUE_MAP_CSV,
        "reference_residue_map.csv",
    ):
        _mirror_file(csv_path, out_dir / alias)

    return result


# ── Public tool ───────────────────────────────────────────────────────────────


@tool
def build_consensus_sequence_alignment(
    working_dir: str,
    reference_label: str,
    labels: Optional[List[str]] = None,
    pdb_files: Optional[List[str]] = None,
    fasta_file: str = "",
    sim_dirs: Optional[List[str]] = None,
    base_dir: str = "",
    chain_id: Optional[str] = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    alignment_fasta: str = DEFAULT_ALIGNMENT_FASTA,
    residue_map_csv: str = DEFAULT_RESIDUE_MAP_CSV,
    alignment_json: str = DEFAULT_ALIGNMENT_JSON,
    consensus_json: str = "",
    msa_method: str = DEFAULT_MSA_METHOD,
    conservation_metric: str = DEFAULT_CONSERVATION_METRIC,
    min_conservation: float = DEFAULT_MIN_CONSERVATION,
    allow_star_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Build a MAFFT global MSA and export ``global_consensus_msa`` columns.

    Default alignment engine is **MAFFT**. Star pairwise is used only when
    ``msa_method='star_pairwise'`` or ``allow_star_fallback=True`` (disabled by
    default so robustness / reproducibility runs cannot silently diverge).

    Paper-style outputs (defaults; all thresholds are user-overridable):
      * ``global_msa.fasta`` — full gapped MAFFT alignment (display / provenance).
      * ``global_consensus_msa.json`` / ``.csv`` — columns with occupancy ≥
        ``min_coverage`` (default 0.25) **and** ``conservation_metric`` ≥
        ``min_conservation`` (default similarity 0.5).

    Unfiltered ``global_msa.json`` is **not** written by default (optional
    debug). Pocket transfer is ``15 Å reference shell ∩ consensus columns``.

    ``conservation_metric`` (modular):
      * ``similarity`` — physicochemical-group agreement with the modal AA
        (default; hydrophobic/aromatic/polar/acidic/basic). This is the paper method.
      * ``identity`` — exact modal-AA fraction among non-gaps.
      * ``blosum`` — mean pairwise BLOSUM62.
      * ``coverage`` — occupancy only.

    Alias of ``build_global_mapped_alignment`` / ``build_global_consensus_msa``.

    Provide sequences via **one** of:
      * ``pdb_files`` + ``labels``
      * ``fasta_file`` (headers become labels; reference must appear in FASTA)
      * ``sim_dirs`` + ``labels`` (PDBs resolved from ``base_dir`` / sim folders)

    Args:
        working_dir: Output directory (e.g. ``{base}/analysis``).
        reference_label: Label of the reference sequence.
        labels: Sequence/simulation labels (required unless only ``fasta_file``).
        pdb_files: Optional list of PDB paths (parallel to ``labels``).
        fasta_file: Optional FASTA path instead of PDBs.
        sim_dirs: Optional per-simulation directories for PDB resolution.
        base_dir: Base multi-simulation directory for ``{label}.pdb`` lookup.
        chain_id: Optional protein chain ID when reading PDBs.
        min_coverage: Min occupancy (non-gap fraction) of a column (default 0.25).
        alignment_fasta: Output MSA FASTA filename.
        residue_map_csv: Output mapping table for manual inspection.
        alignment_json: Output compact consensus JSON.
        consensus_json: Optional alias for ``alignment_json``.
        msa_method: ``mafft`` (default) or ``star_pairwise``.
        conservation_metric: ``similarity`` (default, physchem groups), ``identity``,
            ``blosum``, or ``coverage``.
        min_conservation: Threshold for conservation_metric (default 0.5).
        allow_star_fallback: If True and MAFFT cannot run, use star pairwise
            and record ``msa_fallback_reason`` (default False).

    Returns:
        Dict with ``success``, output paths, ``n_sequences``, ``n_consensus_positions``.
    """
    if not HAS_BIO:
        return {"success": False, "error": "Biopython is required for consensus alignment"}

    # LLM plans often pass consensus_json instead of alignment_json.
    if consensus_json and (
        not alignment_json or alignment_json == DEFAULT_ALIGNMENT_JSON
    ):
        alignment_json = consensus_json

    # Sanitize chain_id from JSON null / "None" / blank.
    if chain_id is not None and str(chain_id).strip().lower() in ("", "none", "null"):
        chain_id = None

    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        chains: Dict[str, ChainSequence] = {}

        if fasta_file:
            chains = load_sequences_from_fasta(fasta_file)
            if labels:
                missing = [l for l in labels if l not in chains]
                if missing:
                    return {
                        "success": False,
                        "error": f"Labels not found in FASTA: {missing}",
                    }
                chains = {l: chains[l] for l in labels}
        elif pdb_files and labels:
            # Drop hallucinated / missing PDB paths and fall back to sim_dirs.
            valid_pairs = [
                (p, lab)
                for p, lab in zip(pdb_files, labels)
                if p and Path(p).is_file()
            ]
            if len(valid_pairs) >= 2:
                pdb_files = [p for p, _ in valid_pairs]
                labels = [lab for _, lab in valid_pairs]
                chains = load_sequences_from_pdbs(pdb_files, labels, chain_id=chain_id)
            elif sim_dirs and labels:
                base = base_dir or str(out_dir.parent)
                pdbs, used_labels, missing = _resolve_pdbs_from_sims(
                    sim_dirs, labels, base
                )
                if len(pdbs) < 2:
                    return {
                        "success": False,
                        "error": "Need >=2 structures for alignment",
                        "missing": missing,
                        "invalid_pdb_files": [
                            p for p in (pdb_files or []) if not Path(p).is_file()
                        ],
                    }
                chains = load_sequences_from_pdbs(pdbs, used_labels, chain_id=chain_id)
                labels = used_labels
            else:
                return {
                    "success": False,
                    "error": (
                        "pdb_files were missing on disk and no sim_dirs were "
                        "provided for fallback resolution"
                    ),
                    "invalid_pdb_files": list(pdb_files or []),
                }
        elif sim_dirs and labels:
            base = base_dir or str(out_dir.parent)
            pdbs, used_labels, missing = _resolve_pdbs_from_sims(sim_dirs, labels, base)
            if len(pdbs) < 2:
                return {
                    "success": False,
                    "error": "Need >=2 structures for alignment",
                    "missing": missing,
                }
            chains = load_sequences_from_pdbs(pdbs, used_labels, chain_id=chain_id)
            labels = used_labels
        else:
            return {
                "success": False,
                "error": (
                    "Provide fasta_file, or pdb_files+labels, or sim_dirs+labels"
                ),
            }

        if reference_label not in chains:
            resolved_ref = None
            ref_l = str(reference_label).lower()
            for key in chains:
                k = str(key).lower()
                if (
                    k == ref_l
                    or k.startswith(ref_l + "_")
                    or ref_l.startswith(k + "_")
                    or ref_l in k
                    or k in ref_l
                ):
                    resolved_ref = key
                    break
            if resolved_ref is None:
                return {
                    "success": False,
                    "error": (
                        f"Reference label {reference_label!r} not found; "
                        f"available: {sorted(chains.keys())}"
                    ),
                }
            reference_label = resolved_ref

        if len(chains) < 2:
            return {"success": False, "error": "Need at least two sequences to align"}

        method = (msa_method or DEFAULT_MSA_METHOD).strip().lower()
        if method in ("star", "star_pairwise", "pairwise", "biopython"):
            msa = build_star_msa_to_reference(chains, reference_label)
            msa["msa_method"] = "star_pairwise"
        else:
            try:
                msa = build_mafft_msa_to_reference(
                    chains,
                    reference_label,
                    allow_star_fallback=bool(allow_star_fallback),
                )
            except RuntimeError as exc:
                return {
                    "success": False,
                    "error": str(exc),
                    "msa_method": "mafft",
                    "mafft_bin": find_mafft_executable(),
                }

        metric = (conservation_metric or DEFAULT_CONSERVATION_METRIC).strip().lower()
        internal = normalize_conservation_metric(metric)
        if internal not in ("group", "identity", "blosum", "coverage"):
            metric = DEFAULT_CONSERVATION_METRIC

        # Paper path: select consensus columns directly (no intermediate
        # unfiltered global_msa.json). That file is optional debug only.
        consensus = select_consensus_positions(
            msa,
            min_coverage=min_coverage,
            conservation_metric=metric,
            min_conservation=min_conservation,
        )
        if len(consensus) < 3:
            return {
                "success": False,
                "error": (
                    f"Only {len(consensus)} consensus positions at "
                    f"min_coverage={min_coverage}, "
                    f"{metric}>={min_conservation}; need >=3"
                ),
                "msa_method": msa.get("msa_method"),
            }

        paths = write_consensus_outputs(
            msa,
            consensus,
            out_dir,
            fasta_name=alignment_fasta,
            csv_name=residue_map_csv,
            json_name=alignment_json,
            all_positions=None,
            chains=chains,
            conservation_metric=metric,
            min_conservation=min_conservation,
            min_coverage=min_coverage,
            write_full_msa_json=False,
        )

        try:
            from src.analysis.summary_logger import append_analysis_summary

            append_analysis_summary(
                working_dir=str(out_dir),
                analysis_type="ConsensusSequenceAlignment",
                statistics={
                    "n_sequences": len(chains),
                    "n_consensus_positions": len(consensus),
                    "min_coverage": min_coverage,
                    "min_conservation": min_conservation,
                    "conservation_metric": metric,
                    "msa_method": msa.get("msa_method"),
                    "reference_label": reference_label,
                },
                files=paths,
                metadata={"labels": list(chains.keys())},
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("append_analysis_summary failed: %s", exc)

        return {
            "success": True,
            "message": (
                f"Global MSA ({msa.get('msa_method')}): {len(chains)} sequences, "
                f"{len(consensus)} global_consensus_msa columns "
                f"({metric}>={min_conservation}, occupancy>={min_coverage})"
            ),
            "reference_label": reference_label,
            "n_sequences": len(chains),
            "n_consensus_positions": len(consensus),
            "n_global_consensus": len(consensus),
            "n_global_mapped": len(consensus),
            "mapping_kind": MAPPING_KIND_CONSENSUS,
            "msa_method": msa.get("msa_method"),
            "conservation_metric": metric,
            "min_conservation": min_conservation,
            "min_coverage": min_coverage,
            "labels": list(chains.keys()),
            **paths,
        }
    except Exception as exc:
        logger.exception("build_global_mapped_alignment failed")
        return {"success": False, "error": str(exc)}


@tool
def build_global_mapped_alignment(
    working_dir: str,
    reference_label: str,
    labels: Optional[List[str]] = None,
    pdb_files: Optional[List[str]] = None,
    fasta_file: str = "",
    sim_dirs: Optional[List[str]] = None,
    base_dir: str = "",
    chain_id: Optional[str] = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    alignment_fasta: str = DEFAULT_ALIGNMENT_FASTA,
    residue_map_csv: str = DEFAULT_RESIDUE_MAP_CSV,
    alignment_json: str = DEFAULT_ALIGNMENT_JSON,
    consensus_json: str = "",
    msa_method: str = DEFAULT_MSA_METHOD,
    conservation_metric: str = DEFAULT_CONSERVATION_METRIC,
    min_conservation: float = DEFAULT_MIN_CONSERVATION,
    allow_star_fallback: bool = False,
) -> Dict[str, Any]:
    """Build a MAFFT global MSA; export ``global_consensus_msa`` (filtered columns).

    Default ``global_consensus_msa`` = MAFFT columns with occupancy ≥ 0.25 and
    physicochemical-group **similarity ≥ 0.5** (paper method). Pass
    ``conservation_metric='identity'`` or ``'blosum'`` and/or different
    ``min_conservation`` / ``min_coverage`` to change the definition.

    Does **not** write unfiltered ``global_msa.json`` by default. Pocket
    mapping is a separate step: 15 Å ligand shell ∩ these consensus columns,
    then transferred to every protein. Does not silently fall back to star
    pairwise unless ``allow_star_fallback=True``.
    """
    return build_consensus_sequence_alignment.func(
        working_dir=working_dir,
        reference_label=reference_label,
        labels=labels,
        pdb_files=pdb_files,
        fasta_file=fasta_file,
        sim_dirs=sim_dirs,
        base_dir=base_dir,
        chain_id=chain_id,
        min_coverage=min_coverage,
        alignment_fasta=alignment_fasta,
        residue_map_csv=residue_map_csv,
        alignment_json=alignment_json,
        consensus_json=consensus_json,
        msa_method=msa_method,
        conservation_metric=conservation_metric,
        min_conservation=min_conservation,
        allow_star_fallback=allow_star_fallback,
    )


# Preferred name; ``build_global_mapped_alignment`` kept for in-flight plans.
build_global_consensus_msa = build_global_mapped_alignment
