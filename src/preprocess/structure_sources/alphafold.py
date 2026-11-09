"""
AlphaFold Protein Structure Database source.

Downloads predicted structures from EBI AlphaFold DB.
"""
import urllib.error
import urllib.request
from typing import Dict, Any, List

from .base import StructureSource


class AlphaFoldSource(StructureSource):
    """Fetch structures from the AlphaFold DB (EBI)."""

    name = "alphafold"
    description = "AlphaFold Protein Structure Database (EBI)"

    # Try newest model versions first
    MODEL_VERSIONS = ("v6", "v5", "v4", "v3", "v2")

    def can_fetch(self, uniprot_id: str) -> bool:
        return bool(uniprot_id and uniprot_id.strip())

    def _build_urls(self, uniprot_id: str) -> List[str]:
        uid = uniprot_id.strip().upper()
        return [
            f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_{ver}.pdb"
            for ver in self.MODEL_VERSIONS
        ]

    def download(self, uniprot_id: str, output_path: str) -> Dict[str, Any]:
        uid = uniprot_id.strip().upper()
        errors = []

        for url in self._build_urls(uniprot_id):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "SimAgent/1.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    data = response.read()
                if not data or len(data) < 100:
                    errors.append(f"{url}: empty response")
                    continue

                with open(output_path, "wb") as fh:
                    fh.write(data)

                return {
                    "success": True,
                    "source": self.name,
                    "uniprot_id": uid,
                    "url": url,
                    "output_file": output_path,
                    "message": f"Downloaded AlphaFold structure for {uid}",
                }
            except urllib.error.HTTPError as exc:
                errors.append(f"{url}: HTTP {exc.code}")
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        return {
            "success": False,
            "source": self.name,
            "uniprot_id": uid,
            "error": f"AlphaFold download failed for {uid}",
            "details": errors,
        }
