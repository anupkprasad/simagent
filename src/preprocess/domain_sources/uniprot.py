"""
UniProt domain lookup via REST API.

Fetches feature annotations (Domain, Region) and matches them to requested
domain labels such as kinase_domain or activation_loop.
"""
import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .base import DomainSource


# Map user-facing domain labels to UniProt feature description keywords
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "kinase_domain": [
        "protein kinase",
        "kinase domain",
        "tyrosine-protein kinase",
        "serine/threonine-protein kinase",
        "serine/threonine kinase",
    ],
    "kinase": ["protein kinase", "kinase"],
    "catalytic_domain": ["protein kinase", "catalytic domain", "kinase"],
    "activation_loop": ["activation loop"],
    "sh2_domain": ["sh2"],
    "sh3_domain": ["sh3"],
    "fibronectin_domain": ["fibronectin"],
    "transmembrane": ["transmembrane", "helix"],
}


class UniProtDomainSource(DomainSource):
    """Look up domain boundaries from UniProtKB feature annotations."""

    name = "uniprot"
    description = "UniProtKB protein feature annotations"

    API_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"

    def lookup(
        self,
        uniprot_id: str,
        domain_label: str,
    ) -> Optional[Dict[str, Any]]:
        uid = (uniprot_id or "").strip().upper()
        if not uid or not domain_label:
            return None

        try:
            features = self._fetch_features(uid)
        except Exception as exc:
            return {
                "success": False,
                "source": self.name,
                "uniprot_id": uid,
                "domain_label": domain_label,
                "error": str(exc),
            }

        keywords = self._keywords_for_label(domain_label)
        match = self._best_feature_match(features, keywords)
        if not match:
            return None

        start, end, description, feat_type = match
        return {
            "success": True,
            "source": self.name,
            "uniprot_id": uid,
            "domain_label": domain_label,
            "start_resid": start,
            "end_resid": end,
            "description": description,
            "feature_type": feat_type,
            "confidence": "annotated",
            "message": f"UniProt {feat_type}: {description} ({start}-{end})",
        }

    def _fetch_features(self, uniprot_id: str) -> List[Dict[str, Any]]:
        url = self.API_URL.format(accession=uniprot_id)
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "SimAgent/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        features = []
        for feat in data.get("features", []):
            loc = feat.get("location") or {}
            start = (loc.get("start") or {}).get("value")
            end = (loc.get("end") or {}).get("value")
            if start is None or end is None:
                continue
            features.append(
                {
                    "type": feat.get("type", ""),
                    "description": (feat.get("description") or "").strip(),
                    "start": int(start),
                    "end": int(end),
                }
            )
        return features

    def _keywords_for_label(self, domain_label: str) -> List[str]:
        key = domain_label.lower().replace(" ", "_")
        if key in DOMAIN_KEYWORDS:
            return DOMAIN_KEYWORDS[key]
        if "kinase" in key:
            return DOMAIN_KEYWORDS["kinase_domain"]
        # Generic: use words from label
        words = re.sub(r"[_\-]+", " ", key).strip()
        return [words] if words else [key]

    def _best_feature_match(
        self,
        features: List[Dict[str, Any]],
        keywords: List[str],
    ) -> Optional[Tuple[int, int, str, str]]:
        """Score features by keyword match; prefer Domain over Region."""
        type_priority = {"Domain": 0, "Region": 1, "Repeat": 2, "Motif": 3}
        scored = []

        for feat in features:
            desc_lower = feat["description"].lower()
            for kw in keywords:
                if kw.lower() in desc_lower:
                    priority = type_priority.get(feat["type"], 9)
                    scored.append(
                        (
                            priority,
                            -(feat["end"] - feat["start"]),  # prefer longer match
                            feat["start"],
                            feat["end"],
                            feat["description"],
                            feat["type"],
                        )
                    )
                    break

        if not scored:
            return None

        scored.sort()
        _, _, start, end, description, feat_type = scored[0]
        return start, end, description, feat_type
