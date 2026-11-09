"""
Sequence download tools — UniProt FASTA and RCSB PDB polymer FASTA.
"""
from __future__ import annotations

import os
import re
import urllib.request
from typing import Any, Dict, Optional

from langchain.tools import tool

_UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
_RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdb_id}/download"
_PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
# Loose UniProt accession check (P21860, Q9C098, A0A0B4J2F0, …)
_UNIPROT_RE = re.compile(r"^[A-Z][0-9][A-Z0-9]{3,9}$")


def _http_get_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SimAgent/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


@tool
def download_protein_sequence(
    accession: str,
    source: str = "uniprot",
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download a protein sequence FASTA from UniProt or RCSB PDB.

    Use for sequence-only needs (alignment reference, remodeling target, or
    documentation). MD setup still requires a 3D structure (AlphaFold/RCSB PDB).

    Args:
        accession: UniProt accession (e.g. Q9C098) or PDB ID (e.g. 6VC0)
        source: 'uniprot' (canonical FASTA) or 'pdb' / 'rcsb' (entry polymer FASTA)
        output_file: Output FASTA path (default: {accession}.fasta)

    Returns:
        Dict with success, output_file, source, sequence length, and message/error
    """
    acc = (accession or "").strip().upper()
    if not acc:
        return {"success": False, "error": "accession is required"}

    src = (source or "uniprot").strip().lower()
    if src in ("pdb", "rcsb"):
        if not _PDB_ID_RE.match(acc):
            return {
                "success": False,
                "error": f"Invalid PDB ID for sequence download: {accession!r}",
            }
        url = _RCSB_FASTA_URL.format(pdb_id=acc)
        default_name = f"{acc.lower()}.fasta"
    elif src == "uniprot":
        if not _UNIPROT_RE.match(acc):
            return {
                "success": False,
                "error": f"Invalid UniProt accession for sequence download: {accession!r}",
            }
        url = _UNIPROT_FASTA_URL.format(accession=acc)
        default_name = f"{acc.lower()}.fasta"
    else:
        return {
            "success": False,
            "error": f"Unknown sequence source: {source!r} (use 'uniprot' or 'pdb')",
        }

    out = output_file or default_name
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    try:
        text = _http_get_text(url)
    except Exception as exc:
        return {
            "success": False,
            "source": src,
            "accession": acc,
            "url": url,
            "error": f"Sequence download failed: {exc}",
        }

    if not text.strip().startswith(">"):
        return {
            "success": False,
            "source": src,
            "accession": acc,
            "url": url,
            "error": "Response was not a FASTA record",
        }

    seq_chars = "".join(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith(">")
    )
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")

    return {
        "success": True,
        "source": src,
        "accession": acc,
        "url": url,
        "output_file": out,
        "sequence_length": len(seq_chars),
        "message": f"Downloaded {src} FASTA for {acc} ({len(seq_chars)} aa)",
    }
