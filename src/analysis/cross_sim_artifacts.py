"""Cross-simulation (pre-combined) artifact contract under ``{base}/cross_sim/``.

Pre-combined analysis writes shared artifacts (pocket map, MSA, consensus
residues) here. Per-simulation analysis auto-discovers them so traj tools can
use mapped selections without inventing paths.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

CROSS_SIM_DIRNAME = "cross_sim"
POCKET_MAP_NAME = "pocket_mapped.json"
POCKET_MAP_LEGACY_NAME = "pocket_map.json"
CONSENSUS_RESIDUES_NAME = "global_consensus_msa.json"
CONSENSUS_RESIDUES_LEGACY_NAME = "consensus_residues.json"
CONSENSUS_RESIDUES_ALIASES = (
    "global_consensus_msa.json",
    "global_mapped.json",
    "consensus_residues.json",
    "reference_msa_alignment.json",
    "reference_consensus.json",
    "consensus_alignment.json",
)
MSA_FASTA_NAME = "global_msa.fasta"
MSA_FASTA_LEGACY_NAME = "consensus_msa.fasta"
MSA_ALIGNMENT_NAME = "consensus_alignment.txt"
GLOBAL_MSA_JSON_NAME = "global_msa.json"  # optional debug; not required
PRE_COMPLETE_NAME = "pre_combined_complete.json"
README_NAME = "README.md"
CONSENSUS_SCHEMA_VERSION = "2.0"


def cross_sim_dir(base_dir: str | Path) -> Path:
    return Path(base_dir).resolve() / CROSS_SIM_DIRNAME


def ensure_cross_sim_dir(base_dir: str | Path) -> Path:
    d = cross_sim_dir(base_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def dumps_json_compact_lists(obj: Any, *, indent: int = 2) -> str:
    """Pretty-print JSON but keep homogeneous scalar lists on one line.

    Makes ``resids: [5, 6, 7, ...]`` human-readable without multiline spam.
    """

    def _dump(o: Any, level: int = 0) -> str:
        sp = " " * (indent * level)
        sp1 = " " * (indent * (level + 1))
        if isinstance(o, dict):
            if not o:
                return "{}"
            parts = []
            for k, v in o.items():
                parts.append(f"{sp1}{json.dumps(str(k))}: {_dump(v, level + 1)}")
            return "{\n" + ",\n".join(parts) + "\n" + sp + "}"
        if isinstance(o, list):
            if not o:
                return "[]"
            if all(
                x is None or (isinstance(x, (int, float)) and not isinstance(x, bool))
                for x in o
            ):
                return json.dumps(o, separators=(", ", ": "))
            if all(isinstance(x, str) and len(x) <= 32 for x in o) and len(o) <= 128:
                return json.dumps(o, separators=(", ", ": "))
            parts = [f"{sp1}{_dump(x, level + 1)}" for x in o]
            return "[\n" + ",\n".join(parts) + "\n" + sp + "]"
        return json.dumps(o, ensure_ascii=False)

    return _dump(obj, 0) + "\n"


def write_json_compact(path: Union[str, Path], obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps_json_compact_lists(obj), encoding="utf-8")
    return p


def compact_consensus_from_positions(
    *,
    reference_label: str,
    labels: Sequence[str],
    consensus_positions: Sequence[Dict[str, Any]],
    msa_width: Optional[int] = None,
    msa_method: str = "mafft",
    conservation_metric: str = "similarity",
    min_conservation: float = 0.5,
    min_coverage: float = 0.85,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build pocket_map-like compact consensus residue schema (v2)."""
    labels = [str(l) for l in labels]
    n = len(consensus_positions)
    msa_cols: List[Any] = []
    ref_resids: List[Any] = []
    ref_aas: List[str] = []
    coverage: List[float] = []
    conservation: List[float] = []
    per_sim: Dict[str, Dict[str, List[Any]]] = {
        lab: {"resids": [], "aas": [], "seq_indices": []} for lab in labels
    }

    for pos in consensus_positions:
        msa_cols.append(pos.get("msa_col"))
        ref_resids.append(pos.get("reference_resid"))
        aa = pos.get("reference_aa") or ""
        ref_aas.append(str(aa)[:1] if aa else "")
        coverage.append(float(pos.get("coverage_fraction") or 0.0))
        if "conservation" in pos:
            conservation.append(float(pos.get("conservation") or 0.0))
        mappings = pos.get("mappings") or {}
        for lab in labels:
            m = mappings.get(lab) or {}
            per_sim[lab]["resids"].append(m.get("resid"))
            maa = m.get("aa") or ""
            per_sim[lab]["aas"].append(str(maa)[:1] if maa else "")
            per_sim[lab]["seq_indices"].append(m.get("seq_index"))

    # Prefer compact AA strings (like reference_aas) for human readability.
    for lab in labels:
        aa_list = per_sim[lab]["aas"]
        per_sim[lab]["aas"] = "".join((a if a else "-") for a in aa_list)

    out: Dict[str, Any] = {
        "schema_version": CONSENSUS_SCHEMA_VERSION,
        "reference_label": reference_label,
        "labels": labels,
        "msa_method": msa_method,
        "conservation_metric": conservation_metric,
        "min_conservation": float(min_conservation),
        "min_coverage": float(min_coverage),
        "msa_width": msa_width,
        "n_consensus_positions": n,
        "consensus_indices": list(range(n)),
        "msa_cols": msa_cols,
        "reference_resids": ref_resids,
        "reference_aas": "".join(ref_aas),
        "coverage_fraction": coverage,
        "per_sim": per_sim,
    }
    if conservation and len(conservation) == n:
        out["conservation"] = conservation
    if extra:
        for k, v in extra.items():
            if k not in out:
                out[k] = v
    return out


def expand_consensus_positions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand compact v2 consensus JSON (or pass through legacy list format)."""
    if not isinstance(data, dict):
        return []
    legacy = data.get("consensus_positions")
    if isinstance(legacy, list) and legacy:
        return list(legacy)

    labels = [str(l) for l in (data.get("labels") or [])]
    n = int(data.get("n_consensus_positions") or 0)
    if n <= 0:
        idxs = data.get("consensus_indices") or data.get("reference_resids") or []
        n = len(idxs)
    if n <= 0:
        return []

    msa_cols = list(data.get("msa_cols") or [None] * n)
    ref_resids = list(data.get("reference_resids") or [None] * n)
    ref_aas = data.get("reference_aas") or ""
    if isinstance(ref_aas, list):
        ref_aa_list = [str(x)[:1] if x else "" for x in ref_aas]
    else:
        ref_aa_list = list(str(ref_aas))
    while len(ref_aa_list) < n:
        ref_aa_list.append("")
    coverage = list(data.get("coverage_fraction") or [None] * n)
    conservation = list(data.get("conservation") or [])
    per_sim = data.get("per_sim") or {}
    ref_label = data.get("reference_label")

    positions: List[Dict[str, Any]] = []
    for i in range(n):
        mappings: Dict[str, Any] = {}
        for lab in labels:
            entry = per_sim.get(lab) or {}
            resids = entry.get("resids") or []
            aas = entry.get("aas") or []
            seqs = entry.get("seq_indices") or []
            resid = resids[i] if i < len(resids) else None
            if resid is None or resid == "":
                continue
            aa = ""
            if isinstance(aas, str):
                aa = aas[i] if i < len(aas) else ""
            elif i < len(aas):
                aa = str(aas[i] or "")[:1]
            mappings[lab] = {
                "resid": int(resid) if resid is not None else None,
                "aa": aa,
                "seq_index": seqs[i] if i < len(seqs) else None,
            }
        pos: Dict[str, Any] = {
            "consensus_index": i,
            "msa_col": msa_cols[i] if i < len(msa_cols) else i,
            "reference_seq_index": None,
            "reference_resid": ref_resids[i] if i < len(ref_resids) else None,
            "reference_aa": ref_aa_list[i] if i < len(ref_aa_list) else "",
            "coverage_fraction": coverage[i] if i < len(coverage) else None,
            "mappings": mappings,
        }
        if conservation and i < len(conservation):
            pos["conservation"] = conservation[i]
        if ref_label and ref_label in mappings:
            pos["reference_seq_index"] = mappings[ref_label].get("seq_index")
        positions.append(pos)
    return positions


def pocket_selection_for_label(
    pocket_map: Dict[str, Any],
    label: str,
) -> Optional[str]:
    """Return MDAnalysis selection string for *label* from ``pocket_map.json``."""
    per = (pocket_map or {}).get("per_sim") or {}
    lab = str(label)
    entry = per.get(lab)
    if entry is None:
        for k, v in per.items():
            if str(k).lower() == lab.lower():
                entry = v
                break
    if not isinstance(entry, dict):
        return None
    sel = entry.get("selection")
    if sel:
        return str(sel)
    resids = entry.get("resids") or []
    nums = [str(int(r)) for r in resids if r is not None and str(r).strip() != ""]
    if not nums:
        return None
    return "resid " + " ".join(nums)


def _pocket_label_aliases(label: str) -> List[str]:
    """Folder and bare-id spellings of one simulation label."""
    raw = str(label or "").strip()
    if not raw:
        return []
    aliases = [raw]
    low = raw.lower()
    for suf in ("_atp", "_adp", "_amp"):
        if low.endswith(suf):
            aliases.append(raw[: -len(suf)])
        else:
            aliases.append(raw + suf.upper())
    seen = set()
    out: List[str] = []
    for alias in aliases:
        key = alias.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(alias)
    return out


def pocket_resids_for_label(
    pocket_map: Dict[str, Any],
    label: str,
) -> List[int]:
    """Return mapped PDB residue IDs for *label* from ``pocket_mapped.json``."""
    per = (pocket_map or {}).get("per_sim") or {}
    lab = str(label)
    entry = None
    for alias in _pocket_label_aliases(lab):
        entry = per.get(alias)
        if entry is None:
            for k, v in per.items():
                if str(k).lower() == alias.lower():
                    entry = v
                    break
        if entry is not None:
            break
    if entry is None:
        per_label = (pocket_map or {}).get("per_label_resids") or {}
        raw = None
        for alias in _pocket_label_aliases(lab):
            raw = per_label.get(alias)
            if raw is None:
                for k, v in per_label.items():
                    if str(k).lower() == alias.lower():
                        raw = v
                        break
            if raw is not None:
                break
        if isinstance(raw, (list, tuple)):
            return [int(r) for r in raw if r is not None and str(r).strip() != ""]
        return []
    if not isinstance(entry, dict):
        return []
    resids = entry.get("resids") or []
    return [int(r) for r in resids if r is not None and str(r).strip() != ""]


def write_cross_sim_readme(base_dir: str | Path) -> Path:
    """Write ``cross_sim/README.md`` describing artifact contract."""
    root = ensure_cross_sim_dir(base_dir)
    path = root / README_NAME
    text = """# cross_sim/ — shared pre-combined artifacts

These files are produced **before** per-simulation trajectory analysis so every
system can reuse the same **pocket_mapped** and **global_consensus_msa** residue lists.

## Two maps (independent)

| Map | Meaning |
|-----|---------|
| **pocket_mapped** | Residues of the ATP-binding pocket used for analysis: (reference residues within 15 Å of ligand ATP) **∩ global_consensus_msa**, transferred to every system via the MSA. |
| **global_consensus_msa** | Consensus columns of the common sequence alignment: MAFFT columns with occupancy ≥ 0.25 and physicochemical-group **similarity ≥ 0.5** (paper default; identity / BLOSUM optional). Legacy filename: `global_mapped.json`. |

## Files

| File | Purpose |
|------|---------|
| `pocket_mapped.json` (`pocket_map.json`) | Per-sim pocket residue lists + MDAnalysis `selection` strings. |
| `global_consensus_msa.json` (`global_mapped.json`, `consensus_residues.json`) | Compact similarity-filtered MSA columns (v2). |
| `global_msa.fasta` (`consensus_msa.fasta`) | Gapped MSA (reference first) — letter source for plots/transfer; plots show consensus/pocket columns only. |
| `pocket_mapped.csv` / `reference_pocket_definition.json` | Full pocket definition + audit from the 15 Å ligand filter. |
| `pre_combined_complete.json` | Gate marker (`success: true` required to advance). |

Optional debug (not required for the paper pipeline): `global_msa.json` = every MAFFT column.

## `pocket_mapped.json` (human-friendly)

```json
{
  "reference_label": "p17612_ATP",
  "reference_selection": "resname ATP",
  "mapping_kind": "pocket_mapped",
  "per_sim": {
    "p17612_ATP": {
      "resids": [5, 6, 7, 8],
      "selection": "resid 5 6 7 8"
    }
  }
}
```

Python helper: `pocket_selection_for_label(pocket_map, label)`.

## `global_consensus_msa.json` (compact v2)

Instead of one object per consensus index, arrays are stored together:

- `consensus_indices`, `msa_cols`, `reference_resids`, `reference_aas`
- `per_sim[label].resids` / `.aas` / `.seq_indices` (parallel arrays)

Expand to the legacy list-of-dicts form with:

```python
from src.analysis.cross_sim_artifacts import expand_consensus_positions
positions = expand_consensus_positions(json.load(open("global_consensus_msa.json")))
```

MSA defaults: **MAFFT** alignment; consensus columns kept when
physicochemical-group `similarity` (or `identity` / `blosum` if requested)
≥ `min_conservation` (default **0.5**) and occupancy ≥ `min_coverage`
(default **0.25**). `pocket_mapped` = 15 Å ligand shell ∩ those columns.

## Typical workflow

1. `build_global_mapped_alignment` → `global_msa.fasta` + `global_consensus_msa.*`
2. `define_pocket_mapped_residues` → 15 Å ATP ∩ consensus, transferred via MSA
3. Harvest / normalize → `pocket_mapped.json` + `global_consensus_msa.json` here
4. Per-sim analysis auto-discovers this directory for pocket selections
"""
    path.write_text(text, encoding="utf-8")
    return path


def discover_cross_sim_artifacts(base_dir: str | Path) -> Dict[str, Any]:
    """Return paths / loaded payload for artifacts under ``base/cross_sim/``.

    Keys always present; values may be None when missing.
    """
    root = cross_sim_dir(base_dir)
    out: Dict[str, Any] = {
        "cross_sim_dir": str(root) if root.is_dir() else None,
        "pocket_map_path": None,
        "pocket_map": None,
        "consensus_residues_path": None,
        "consensus_residues": None,
        "msa_fasta_path": None,
        "msa_alignment_path": None,
        "pre_combined_complete": False,
        "artifacts": [],
    }
    if not root.is_dir():
        return out

    pocket_path = None
    for name in (POCKET_MAP_NAME, POCKET_MAP_LEGACY_NAME):
        cand = root / name
        if cand.is_file():
            pocket_path = cand
            break
    if pocket_path is not None:
        out["pocket_map_path"] = str(pocket_path)
        out["artifacts"].append(str(pocket_path))
        try:
            out["pocket_map"] = json.loads(pocket_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", pocket_path, exc)

    cons_path = None
    for alt in CONSENSUS_RESIDUES_ALIASES:
        cand = root / alt
        if cand.is_file():
            cons_path = cand
            break
    if cons_path is not None:
        out["consensus_residues_path"] = str(cons_path)
        out["artifacts"].append(str(cons_path))
        try:
            raw = json.loads(cons_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and not raw.get("consensus_positions"):
                raw = dict(raw)
                raw["consensus_positions"] = expand_consensus_positions(raw)
            out["consensus_residues"] = raw
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", cons_path, exc)

    for names, key in (
        ((MSA_FASTA_NAME, MSA_FASTA_LEGACY_NAME, "reference_msa_alignment.fasta"), "msa_fasta_path"),
        ((MSA_ALIGNMENT_NAME,), "msa_alignment_path"),
    ):
        for name in names:
            p = root / name
            if p.is_file():
                out[key] = str(p)
                out["artifacts"].append(str(p))
                break

    for p in sorted(root.iterdir()):
        if p.is_file() and str(p) not in out["artifacts"]:
            out["artifacts"].append(str(p))

    complete = root / PRE_COMPLETE_NAME
    out["pre_combined_complete"] = complete.is_file()
    return out


def write_pocket_map(
    base_dir: str | Path,
    payload: Dict[str, Any],
) -> Path:
    """Write ``pocket_map.json`` with a minimal validated schema."""
    root = ensure_cross_sim_dir(base_dir)
    path = root / POCKET_MAP_NAME
    data = {
        "schema_version": "1.0",
        "mapping_kind": payload.get("mapping_kind") or "pocket_mapped",
        "reference_label": payload.get("reference_label"),
        "reference_selection": payload.get("reference_selection"),
        "per_sim": payload.get("per_sim") or {},
        "notes": payload.get("notes") or "",
    }
    for k, v in payload.items():
        if k not in data:
            data[k] = v
    write_json_compact(path, data)
    legacy = root / POCKET_MAP_LEGACY_NAME
    if legacy.resolve() != path.resolve():
        shutil.copy2(path, legacy)
    write_cross_sim_readme(base_dir)
    return path


# One retry, then continue into per-sim analysis. The count lives on disk
# because the in-memory counter does not survive the supervisor hop.
MAX_PRE_COMBINED_ATTEMPTS = 2


def pre_combined_attempt_count(base_dir: str | Path) -> int:
    """How many failed pre-combined attempts are recorded in cross_sim/."""
    marker = cross_sim_dir(base_dir) / PRE_COMPLETE_NAME
    if not marker.is_file():
        return 0
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return 0
    try:
        return max(0, int(data.get("attempts") or 0))
    except (TypeError, ValueError):
        return 0


def pre_combined_done_on_disk(base_dir: str | Path) -> bool:
    """True when pre-combined produced artifacts, or failed often enough to move on.

    A single failed marker is not done, so the pipeline can retry the MSA once.
    After ``MAX_PRE_COMBINED_ATTEMPTS`` the campaign continues without consensus
    metrics instead of calling the same step until the graph recursion limit.
    """
    root = cross_sim_dir(base_dir)
    if (root / "pre_combined_give_up.json").is_file():
        return True
    marker = root / PRE_COMPLETE_NAME
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    try:
        attempts = int(data.get("attempts") or 0)
    except (TypeError, ValueError):
        attempts = 0
    if attempts >= MAX_PRE_COMBINED_ATTEMPTS:
        return True
    if data.get("success") is False and not data.get("skipped"):
        return False
    return cross_sim_artifacts_ready(base_dir)


def cross_sim_artifacts_ready(base_dir: str | Path) -> bool:
    """True when MSA JSON (or consensus residues) exists for family tools."""
    root = cross_sim_dir(base_dir)
    if not root.is_dir():
        return False
    artifacts = discover_cross_sim_artifacts(base_dir)
    has_msa = bool(
        artifacts.get("consensus_residues_path")
        or (root / "global_consensus_msa.json").is_file()
        or (root / "global_mapped.json").is_file()
        or (root / "global_msa.json").is_file()
        or (root / "reference_msa_alignment.json").is_file()
        or (root / "reference_consensus.json").is_file()
        or artifacts.get("msa_fasta_path")
    )
    has_pocket = bool(
        artifacts.get("pocket_map_path")
        or (root / "pocket_mapped.json").is_file()
        or (root / "pocket_map.json").is_file()
        or (root / "reference_pocket_definition.json").is_file()
        or (root / "reference_pocket_residue_map.csv").is_file()
        or (root / "pocket_mapped.csv").is_file()
    )
    # MSA is mandatory for consensus_* tools; pocket definition/map for χ1 / COM.
    return bool(has_msa and has_pocket)


def mark_pre_combined_complete(
    base_dir: str | Path,
    *,
    labels: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    root = ensure_cross_sim_dir(base_dir)
    path = root / PRE_COMPLETE_NAME
    previous: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except Exception:
            previous = {}
    payload = {
        "success": True,
        "labels": labels or [],
        **(extra or {}),
    }
    # A later harvest that does not mention attempts must not reset the
    # on-disk retry count back to zero.
    if "attempts" not in (extra or {}) and previous.get("attempts"):
        payload["attempts"] = previous["attempts"]
    if extra is not None and "success" in extra:
        payload["success"] = bool(extra["success"])
    # Never claim success when required artifacts are missing.
    if not cross_sim_artifacts_ready(base_dir):
        payload["success"] = False
        payload.setdefault("artifacts_ready", False)
    else:
        payload["artifacts_ready"] = True
        payload["success"] = True
        payload.pop("skipped", None)
    write_json_compact(path, payload)
    return path


def harvest_pre_artifacts_into_cross_sim(
    base_dir: str | Path,
    search_roots: Optional[List[str | Path]] = None,
) -> List[str]:
    """Copy known pre-combined outputs from analysis/ into ``cross_sim/``."""
    base = Path(base_dir).resolve()
    dest = ensure_cross_sim_dir(base)
    roots = [Path(r) for r in (search_roots or [])]
    roots.extend([base / "analysis", base])
    copied: List[str] = []

    name_map = {
        "pocket_mapped.json": POCKET_MAP_NAME,
        "pocket_map.json": POCKET_MAP_LEGACY_NAME,
        "consensus_pocket_map.json": POCKET_MAP_NAME,
        "mapped_pocket_residues.json": POCKET_MAP_NAME,
        "pocket_mapped_definition.json": "pocket_mapped_definition.json",
        "pocket_mapped.csv": "pocket_mapped.csv",
        "reference_pocket_definition.json": "reference_pocket_definition.json",
        "global_consensus_msa.json": CONSENSUS_RESIDUES_NAME,
        "global_mapped.json": CONSENSUS_RESIDUES_NAME,
        "global_msa.json": GLOBAL_MSA_JSON_NAME,
        "consensus_residues.json": CONSENSUS_RESIDUES_LEGACY_NAME,
        "global_msa.fasta": MSA_FASTA_NAME,
        "consensus_msa.fasta": MSA_FASTA_LEGACY_NAME,
        "msa.fasta": MSA_FASTA_NAME,
        "reference_msa_alignment.fasta": "reference_msa_alignment.fasta",
        "consensus_alignment.txt": MSA_ALIGNMENT_NAME,
        "alignment.fasta": MSA_FASTA_NAME,
        "reference_msa_alignment.json": "reference_msa_alignment.json",
        "global_consensus_msa.csv": "global_consensus_msa.csv",
        "global_mapped.csv": "global_mapped.csv",
        "reference_msa_residue_map.csv": "reference_msa_residue_map.csv",
        "reference_pocket_residue_map.csv": "reference_pocket_residue_map.csv",
    }

    seen_dest: set = set()
    for root in roots:
        if not root.is_dir():
            continue
        for src_name, dest_name in name_map.items():
            if dest_name in seen_dest:
                continue
            candidates = list(root.glob(src_name)) + list(root.rglob(src_name))
            for src in candidates:
                if not src.is_file():
                    continue
                target = dest / dest_name
                if target.resolve() == src.resolve():
                    seen_dest.add(dest_name)
                    if str(target) not in copied:
                        copied.append(str(target))
                    break
                try:
                    shutil.copy2(src, target)
                    copied.append(str(target))
                    seen_dest.add(dest_name)
                    break
                except Exception as exc:
                    logger.warning("harvest %s → %s failed: %s", src, target, exc)
    return copied


def normalize_pre_artifacts_to_contract(base_dir: str | Path) -> Optional[Path]:
    """Build ``pocket_map.json`` / compact consensus from harvested tool outputs."""
    root = ensure_cross_sim_dir(base_dir)
    pocket_path = root / POCKET_MAP_NAME
    if not pocket_path.is_file():
        pocket_path = root / POCKET_MAP_LEGACY_NAME
    def_path = None
    for name in (
        "pocket_mapped_definition.json",
        "reference_pocket_definition.json",
        "pocket_mapped.json",
    ):
        cand = root / name
        if cand.is_file():
            def_path = cand
            break
    if def_path is None:
        def_path = root / "reference_pocket_definition.json"

    # Prefer regenerating pocket_mapped.json from the pocket definition so a
    # stale map cannot shadow a freshly rebuilt definition/CSV.
    if def_path.is_file():
        try:
            definition = json.loads(def_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("normalize: failed to read %s: %s", def_path, exc)
            definition = None
        per_label = (
            (definition.get("per_label_resids") or {})
            if isinstance(definition, dict)
            else {}
        )
        if isinstance(definition, dict) and per_label:
            per_sim: Dict[str, Any] = {}
            for lab, resids in per_label.items():
                if isinstance(resids, list):
                    clean = [int(r) for r in resids if r is not None]
                    per_sim[str(lab)] = {
                        "resids": clean,
                        "selection": (
                            f"resid {' '.join(str(r) for r in clean)}" if clean else ""
                        ),
                    }
            write_pocket_map(
                base_dir,
                {
                    "reference_label": definition.get("reference_label"),
                    "reference_selection": definition.get("reference_selection")
                    or definition.get("ligand_selection"),
                    "per_sim": per_sim,
                    "source": str(def_path.name),
                    "mapping_kind": "pocket_mapped",
                    "usable_labels": definition.get("usable_labels"),
                    "per_label_coverage": definition.get("per_label_coverage"),
                },
            )
        elif pocket_path.is_file():
            try:
                pm = json.loads(pocket_path.read_text(encoding="utf-8"))
                write_json_compact(pocket_path, pm)
            except Exception:
                pass
    elif pocket_path.is_file():
        try:
            pm = json.loads(pocket_path.read_text(encoding="utf-8"))
            write_json_compact(pocket_path, pm)
        except Exception:
            pass

    cons_path = root / CONSENSUS_RESIDUES_NAME
    msa_json = root / "reference_msa_alignment.json"
    for src in (
        cons_path,
        msa_json,
        root / CONSENSUS_RESIDUES_LEGACY_NAME,
    ):
        if not src.is_file():
            continue
        try:
            raw = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("normalize: failed to read %s: %s", src, exc)
            continue
        if not isinstance(raw, dict):
            continue
        if raw.get("schema_version") == CONSENSUS_SCHEMA_VERSION and raw.get("per_sim"):
            compact = raw
        elif raw.get("consensus_positions"):
            compact = compact_consensus_from_positions(
                reference_label=str(raw.get("reference_label") or ""),
                labels=list(raw.get("labels") or []),
                consensus_positions=list(raw.get("consensus_positions") or []),
                msa_width=raw.get("msa_width"),
                msa_method=str(raw.get("msa_method") or "mafft"),
                conservation_metric=str(raw.get("conservation_metric") or "similarity"),
                min_conservation=float(raw.get("min_conservation") or 0.5),
                min_coverage=float(raw.get("min_coverage") or 0.25),
                extra={
                    k: v
                    for k, v in raw.items()
                    if k
                    not in {
                        "consensus_positions",
                        "reference_label",
                        "labels",
                        "msa_width",
                        "n_consensus_positions",
                    }
                },
            )
        else:
            continue
        write_json_compact(cons_path, compact)
        write_json_compact(msa_json, compact)
        write_json_compact(root / CONSENSUS_RESIDUES_LEGACY_NAME, compact)
        break

    if def_path.is_file():
        try:
            definition = json.loads(def_path.read_text(encoding="utf-8"))
            if isinstance(definition, dict):
                write_json_compact(def_path, definition)
        except Exception:
            pass

    write_cross_sim_readme(base_dir)
    return pocket_path if pocket_path.is_file() else None


def format_cross_sim_context_for_prompt(artifacts: Dict[str, Any]) -> str:
    """Short block injected into per-sim analysis planning prompts."""
    if not artifacts.get("artifacts") and not artifacts.get("pocket_map_path"):
        return ""
    lines = [
        "",
        "**PRE-COMBINED CROSS-SIM ARTIFACTS (auto-discovered — prefer these):**",
    ]
    if artifacts.get("cross_sim_dir"):
        lines.append(f"- Directory: `{artifacts['cross_sim_dir']}`")
    if artifacts.get("pocket_map_path"):
        lines.append(
            f"- pocket_mapped (15 Å ATP pocket residues/selections): `{artifacts['pocket_map_path']}`"
        )
        pm = artifacts.get("pocket_map") or {}
        if pm.get("reference_label"):
            lines.append(f"- Reference label: {pm['reference_label']}")
    if artifacts.get("consensus_residues_path"):
        lines.append(
            f"- global_consensus_msa (MAFFT group-similarity ≥ 0.5; use as alignment_json): "
            f"`{artifacts['consensus_residues_path']}`"
        )
    # Prefer explicit MSA alignment JSON when present among artifacts.
    for path in artifacts.get("artifacts") or []:
        name = Path(path).name
        if name in (
            "global_consensus_msa.json",
            "global_mapped.json",
            "global_msa.json",
            "reference_msa_alignment.json",
            "consensus_residues.json",
            "reference_consensus.json",
        ):
            lines.append(f"- alignment_json path: `{path}`")
        if name.endswith("_residue_map.csv") or name.endswith("pocket_residue_map.csv"):
            lines.append(f"- pocket_map_csv path: `{path}`")
    if artifacts.get("msa_fasta_path"):
        lines.append(f"- MSA: `{artifacts['msa_fasta_path']}`")
    lines.append(
        "For calculate_consensus_* / run_independent_dynamics_fel, set "
        "alignment_json to the MSA JSON above and label to this simulation's "
        "folder name (e.g. p17612_ATP). Do not invent alternate pocket/MSA paths "
        "when these files exist."
    )
    return "\n".join(lines) + "\n"
