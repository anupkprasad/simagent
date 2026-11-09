"""
RCSB PDB source — resolves UniProt accession to experimental PDB entries.

Uses the RCSB search API to find PDB IDs mapped to a UniProt accession,
then downloads the first matching coordinate file. Also supports direct
download by 4-character PDB ID (e.g. 6VC0).
"""
import json
import re
import urllib.error
import urllib.request
from typing import Dict, Any, List, Optional

from .base import StructureSource

_PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


class RCSBSource(StructureSource):
    """Fetch experimental structures from RCSB PDB via UniProt mapping."""

    name = "rcsb"
    description = "RCSB Protein Data Bank (experimental structures)"

    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

    def can_fetch(self, uniprot_id: str) -> bool:
        return bool(uniprot_id and uniprot_id.strip())

    @staticmethod
    def is_pdb_id(value: str) -> bool:
        return bool(value and _PDB_ID_RE.match(value.strip()))

    def download_by_pdb_id(self, pdb_id: str, output_path: str) -> Dict[str, Any]:
        """Download coordinates for an explicit RCSB PDB entry ID."""
        pid = (pdb_id or "").strip().upper()
        if not self.is_pdb_id(pid):
            return {
                "success": False,
                "source": self.name,
                "pdb_id": pid,
                "error": f"Invalid PDB ID (expected 4 chars like 6VC0): {pdb_id!r}",
            }

        url = self.DOWNLOAD_URL.format(pdb_id=pid)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SimAgent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            if not data or len(data) < 100:
                return {
                    "success": False,
                    "source": self.name,
                    "pdb_id": pid,
                    "url": url,
                    "error": f"Empty or too-small response for PDB {pid}",
                }

            with open(output_path, "wb") as fh:
                fh.write(data)

            return {
                "success": True,
                "source": self.name,
                "pdb_id": pid,
                "url": url,
                "output_file": output_path,
                "message": f"Downloaded RCSB structure {pid}",
            }
        except Exception as exc:
            return {
                "success": False,
                "source": self.name,
                "pdb_id": pid,
                "url": url,
                "error": f"RCSB download failed for PDB {pid}: {exc}",
            }

    def _find_pdb_ids(self, uniprot_id: str, limit: int = 5) -> List[str]:
        uid = uniprot_id.strip().upper()
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.uniprot_accession",
                    "operator": "exact_match",
                    "value": uid,
                },
            },
            "return_type": "entry",
            "request_options": {"results_content_type": ["experimental"]},
        }
        payload = json.dumps(query).encode("utf-8")
        req = urllib.request.Request(
            self.SEARCH_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SimAgent/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))

        ids = []
        for hit in result.get("result_set", [])[:limit]:
            pdb_id = hit.get("identifier")
            if pdb_id:
                ids.append(pdb_id.upper())
        return ids

    def download(self, uniprot_id: str, output_path: str) -> Dict[str, Any]:
        uid = uniprot_id.strip().upper()
        errors = []

        try:
            pdb_ids = self._find_pdb_ids(uid)
        except Exception as exc:
            return {
                "success": False,
                "source": self.name,
                "uniprot_id": uid,
                "error": f"RCSB search failed for {uid}: {exc}",
            }

        if not pdb_ids:
            return {
                "success": False,
                "source": self.name,
                "uniprot_id": uid,
                "error": f"No RCSB PDB entries found for UniProt {uid}",
            }

        for pdb_id in pdb_ids:
            url = self.DOWNLOAD_URL.format(pdb_id=pdb_id)
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "SimAgent/1.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    data = response.read()
                if not data or len(data) < 100:
                    errors.append(f"{pdb_id}: empty response")
                    continue

                with open(output_path, "wb") as fh:
                    fh.write(data)

                return {
                    "success": True,
                    "source": self.name,
                    "uniprot_id": uid,
                    "pdb_id": pdb_id,
                    "url": url,
                    "output_file": output_path,
                    "message": f"Downloaded RCSB structure {pdb_id} for UniProt {uid}",
                }
            except Exception as exc:
                errors.append(f"{pdb_id}: {exc}")

        return {
            "success": False,
            "source": self.name,
            "uniprot_id": uid,
            "error": f"RCSB download failed for {uid}",
            "details": errors,
        }
