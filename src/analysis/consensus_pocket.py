"""
Pocket-mapped residue analysis for cross-simulation comparison.

Paper-style maps (defaults; every threshold is a tool argument):

  * **global_consensus_msa** — MAFFT columns with occupancy ≥ min_coverage
    (0.25) and physicochemical-group similarity ≥ min_conservation (0.5).
    (Legacy filename: ``global_mapped.json``.)
  * **pocket_mapped** — (reference residues within 15 Å of the ligand)
    ∩ global_consensus_msa, then transferred to every system via the MSA.

Pass ``pocket_filter='none'`` to map the full 15 Å shell without the
similarity intersection (requires an unfiltered MSA JSON if present).

Workflow:
  1. ``define_pocket_mapped_residues`` — 15 Å ∩ global_consensus_msa (default)
  2. ``map_pocket_mapped_residues`` — per-simulation PDB resid lists + audit CSV
  3. ``calculate_consensus_pocket_metrics`` — one simulation
  4. ``run_consensus_pocket_metrics_batch`` — all simulations + optional re-collect
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain.tools import tool

from src.analysis.combined_analysis import _find_sim_traj_topology
from src.analysis.reference_landscape import load_consensus_alignment

logger = logging.getLogger(__name__)

# Joint pocket–ligand coupling stability thresholds (legacy stable-coupling metric).
STABLE_COUPLING_COM_A = 7.0
STABLE_COUPLING_ANGLE_DEG = 65.0

# Reference-pocket bound-state COM cutoff (Å), calibrated to MLKL:
# MLKL remains pocket-associated over the full trajectory (max COM ≈ 14.7 Å),
# so frames with COM ≤ this envelope are counted as bound.
REFERENCE_POCKET_BOUND_DISTANCE_A = 15.0

DEFAULT_DEFINITION_JSON = "pocket_mapped_definition.json"
DEFAULT_RESIDUE_MAP_CSV = "pocket_mapped.csv"
DEFAULT_ALIGNMENT_JSON = "global_consensus_msa.json"
LEGACY_ALIGNMENT_JSON_NAMES = (
    "global_consensus_msa.json",
    "global_mapped.json",
    "reference_msa_alignment.json",
    "consensus_residues.json",
)
LEGACY_DEFINITION_JSON = "reference_pocket_definition.json"
LEGACY_RESIDUE_MAP_CSV = "reference_pocket_residue_map.csv"
DEFAULT_POCKET_FILTER = "global_consensus_msa"

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances as mda_distances

    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from src.analysis.binding_site_analyzer import (
    _chdir_working,
    _debounce_bound_mask,
    _find_gmx,
    _parse_xvg_two_columns,
    _restore_cwd,
    _write_gmx_index,
    finalize_ligand_residence_result,
    compute_pocket_rmsf_from_universe,
    compute_protein_ligand_contacts_from_universe,
    resolve_pocket_atoms,
)
from src.analysis.com_distance_calculator import clean_pbc_distance_series, minimum_image_distance
from src.analysis.summary_logger import append_analysis_summary

# Formal side-chain charge at physiological pH (~7.4) by residue name.
# HIS is treated as neutral by default (partially protonated in reality).
_RESIDUE_FORMAL_CHARGE: Dict[str, int] = {
    "ARG": +1,
    "LYS": +1,
    "ASP": -1,
    "GLU": -1,
    # Common protonation-state variants written by force fields.
    "HIP": +1,
    "ASH": 0,
    "GLH": 0,
    "HID": 0,
    "HIE": 0,
    "HIS": 0,
    "LYN": 0,
}


def compute_pocket_net_charge(pocket_atomgroup) -> Dict[str, Any]:
    """Net formal charge of the pocket from unique residues (Arg/Lys +1, Asp/Glu -1)."""
    residues = getattr(pocket_atomgroup, "residues", None)
    if residues is None:
        return {"net_charge": None, "n_positive": None, "n_negative": None}
    n_pos = 0
    n_neg = 0
    for resname in residues.resnames:
        charge = _RESIDUE_FORMAL_CHARGE.get(str(resname).upper(), 0)
        if charge > 0:
            n_pos += 1
        elif charge < 0:
            n_neg += 1
    return {
        "net_charge": int(n_pos - n_neg),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
    }


def _n_mapped_positions(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    n = data.get("n_consensus_positions") or data.get("n_global_msa_columns")
    if n:
        return int(n)
    if data.get("consensus_positions"):
        return len(data["consensus_positions"])
    return len(data.get("reference_resids") or [])


def resolve_msa_json_for_pocket(
    out_dir: Path,
    requested: str,
    *,
    prefer_full: bool = False,
) -> Path:
    """Resolve the MSA JSON used to transfer pocket residues.

    Default (paper): ``global_consensus_msa.json`` — 15 Å ∩ similarity-consensus.
    Legacy ``global_mapped.json`` is accepted. ``prefer_full=True`` looks for
    optional ``global_msa.json`` (every aligned column; not written by default).
    """
    out_dir = Path(out_dir)
    req = Path(requested)
    if not req.is_absolute():
        req = out_dir / req.name
    if req.is_file():
        return req
    if prefer_full:
        names = (
            "global_msa.json",
            *LEGACY_ALIGNMENT_JSON_NAMES,
        )
    else:
        names = (
            *LEGACY_ALIGNMENT_JSON_NAMES,
            "global_msa.json",
        )
    for name in names:
        p = out_dir / name
        if p.is_file():
            return p
    return req


def _mirror_artifact(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() == src.resolve():
        return
    shutil.copy2(src, dest)


def _consensus_positions_near_ligand(
    universe: "mda.Universe",
    consensus_positions: Sequence[Dict[str, Any]],
    reference_label: str,
    *,
    ligand_selection: str = "",
    pocket_cutoff_A: float = 15.0,
    chain_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
    """
    Return consensus positions whose reference Cα is within ``pocket_cutoff_A`` of
    the ligand at frame 0, plus the reference PDB resid set from ligand proximity.
    """
    universe.trajectory[0]
    _, proximity_resids, prox_meta = resolve_pocket_atoms(
        universe,
        ligand_selection=ligand_selection,
        pocket_cutoff=pocket_cutoff_A,
    )
    proximity_set = set(proximity_resids)

    selected: List[Dict[str, Any]] = []
    for pos in consensus_positions:
        m = (pos.get("mappings") or {}).get(reference_label)
        if m is None:
            continue
        ref_resid = m.get("resid")
        if ref_resid is None:
            continue
        if int(ref_resid) in proximity_set:
            selected.append(pos)

    audit = {
        **prox_meta,
        "reference_label": reference_label,
        "ligand_selection": ligand_selection,
        "pocket_cutoff_A": float(pocket_cutoff_A),
        "n_proximity_resids": len(proximity_resids),
        "n_consensus_pocket_positions": len(selected),
    }
    return selected, proximity_resids, audit


def _build_per_label_resid_map(
    pocket_positions: Sequence[Dict[str, Any]],
    labels: Sequence[str],
) -> Dict[str, List[int]]:
    """Map each label to sorted unique PDB resids for consensus pocket positions."""
    out: Dict[str, List[int]] = {}
    for label in labels:
        resids: List[int] = []
        for pos in pocket_positions:
            m = (pos.get("mappings") or {}).get(label)
            if not m or m.get("resid") is None:
                continue
            resids.append(int(m["resid"]))
        out[str(label)] = sorted(set(resids))
    return out


def _safe_output_prefix(prefix: str) -> str:
    """Filename stem only. An absolute prefix makes alias copies target the same file."""
    name = Path(str(prefix or "")).name.strip()
    for suffix in (".csv", ".json", ".dat"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    for token in ("_ligand_orientation", "_ligand_distance", "_metrics"):
        if name.endswith(token):
            name = name[: -len(token)]
    name = name.replace("/", "").strip("._")
    return name or "reference_pocket"


def _resolve_pocket_label(requested: str, known: Sequence[str]) -> Optional[str]:
    """Match a display or folder label onto a pocket-definition label.

    ``p17612`` matches ``p17612_ATP``. Gene names such as ``KAPCA_ATP`` do not
    match a UniProt key; callers should pass the simulation directory name.
    """
    req = str(requested or "").strip()
    known_list = [str(k) for k in known if str(k).strip()]
    if not req or not known_list:
        return None
    by_lower = {k.lower(): k for k in known_list}
    candidates = [req]
    low = req.lower()
    for suf in ("_atp", "_adp", "_amp"):
        if low.endswith(suf):
            candidates.append(req[: -len(suf)])
        else:
            candidates.append(req + suf.upper())
    for cand in candidates:
        hit = by_lower.get(cand.lower())
        if hit:
            return hit
    for cand in candidates:
        cl = cand.lower()
        for key, orig in by_lower.items():
            if key.startswith(cl + "_") or cl.startswith(key + "_"):
                return orig
    return None


def _coverage_fraction(
    pocket_positions: Sequence[Dict[str, Any]],
    label: str,
) -> float:
    if not pocket_positions:
        return 0.0
    mapped = sum(
        1
        for pos in pocket_positions
        if ((pos.get("mappings") or {}).get(label) or {}).get("resid") is not None
    )
    return mapped / len(pocket_positions)


def _write_residue_map_csv(
    path: Path,
    pocket_positions: Sequence[Dict[str, Any]],
    labels: Sequence[str],
) -> None:
    """Wide CSV: one row per consensus pocket position with per-label resid/aa."""
    fieldnames = [
        "consensus_index",
        "reference_label",
        "reference_resid",
        "reference_aa",
    ]
    for label in labels:
        fieldnames.extend([f"{label}_resid", f"{label}_aa"])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for pos in pocket_positions:
            ref_lab = pos.get("reference_label", "")
            row: Dict[str, Any] = {
                "consensus_index": pos.get("consensus_index"),
                "reference_label": ref_lab,
                "reference_resid": pos.get("reference_resid"),
                "reference_aa": pos.get("reference_aa"),
            }
            mappings = pos.get("mappings") or {}
            for label in labels:
                m = mappings.get(label) or {}
                row[f"{label}_resid"] = m.get("resid", "")
                row[f"{label}_aa"] = m.get("aa", "")
            writer.writerow(row)


def _principal_axis(positions: np.ndarray) -> Optional[np.ndarray]:
    """First principal component (major axis) of a point cloud."""
    pos = np.asarray(positions, dtype=float)
    mask = np.isfinite(pos).all(axis=1)
    pos = pos[mask]
    if len(pos) < 3:
        return None
    centred = pos - pos.mean(axis=0)
    cov = centred.T @ centred
    _evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, -1]
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return None
    return axis / norm


def _axis_angle_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> Optional[float]:
    """Undirected angle (0–90°) between two 3D axis vectors."""
    a = np.asarray(axis_a, dtype=float)
    b = np.asarray(axis_b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return None
    cos_theta = abs(float(np.dot(a / na, b / nb)))
    cos_theta = min(max(cos_theta, 0.0), 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def summarize_ligand_orientation_series(
    distances: Sequence[float],
    angles: Sequence[float],
    *,
    stable_com_A: float = STABLE_COUPLING_COM_A,
    stable_angle_deg: float = STABLE_COUPLING_ANGLE_DEG,
) -> Dict[str, Optional[float]]:
    """Scalar pocket–ligand coupling descriptors from per-frame COM distance + axis angle."""
    dist_arr = np.asarray(distances, dtype=float)
    ang_arr = np.asarray(angles, dtype=float)
    if dist_arr.size == 0:
        return {
            "mean_distance_A": None,
            "std_distance_A": None,
            "min_distance_A": None,
            "max_distance_A": None,
            "p95_distance_A": None,
            "mean_axis_angle_deg": None,
            "std_axis_angle_deg": None,
            "p95_axis_angle_deg": None,
            "fraction_stable_coupling": None,
        }

    valid_angles = ang_arr[np.isfinite(ang_arr)]
    finite_angle = np.isfinite(ang_arr)
    fraction_stable: Optional[float] = None
    if finite_angle.any():
        stable_mask = (
            (dist_arr <= stable_com_A)
            & finite_angle
            & (ang_arr <= stable_angle_deg)
        )
        fraction_stable = float(np.mean(stable_mask))

    return {
        "mean_distance_A": float(np.mean(dist_arr)),
        "std_distance_A": float(np.std(dist_arr)),
        "min_distance_A": float(np.min(dist_arr)),
        "max_distance_A": float(np.max(dist_arr)),
        "p95_distance_A": float(np.percentile(dist_arr, 95)),
        "mean_axis_angle_deg": (
            float(np.mean(valid_angles)) if valid_angles.size else None
        ),
        "std_axis_angle_deg": (
            float(np.std(valid_angles)) if valid_angles.size else None
        ),
        "p95_axis_angle_deg": (
            float(np.percentile(valid_angles, 95)) if valid_angles.size else None
        ),
        "fraction_stable_coupling": fraction_stable,
    }


def summarize_ligand_orientation_csv(
    orientation_csv: str | Path,
    *,
    stable_com_A: float = STABLE_COUPLING_COM_A,
    stable_angle_deg: float = STABLE_COUPLING_ANGLE_DEG,
) -> Dict[str, Optional[float]]:
    """Load ``reference_pocket_ligand_orientation.csv`` and return coupling scalars."""
    path = Path(orientation_csv)
    if not path.is_file():
        return {"success": False, "error": f"orientation csv not found: {path}"}

    distances: List[float] = []
    angles: List[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                distances.append(float(row["distance_angstrom"]))
            except (KeyError, TypeError, ValueError):
                continue
            raw_angle = (row.get("axis_angle_deg") or "").strip()
            if raw_angle:
                try:
                    angles.append(float(raw_angle))
                except ValueError:
                    angles.append(float("nan"))
            else:
                angles.append(float("nan"))

    if not distances:
        return {"success": False, "error": f"no distance rows in {path}"}

    stats = summarize_ligand_orientation_series(
        distances,
        angles,
        stable_com_A=stable_com_A,
        stable_angle_deg=stable_angle_deg,
    )
    return {"success": True, **stats}


def compute_consensus_pocket_ligand_geometry(
    u,
    *,
    pocket_resids: List[int],
    ligand_selection: str = "",
    protein_selection: str = "protein",
    distance_output: str = "reference_pocket_ligand_distance.csv",
    orientation_output: str = "reference_pocket_ligand_orientation.csv",
    frame_interval: int = 1,
) -> Dict[str, Any]:
    """
    Pocket–ligand COM distance and major-axis angle in one trajectory pass.

    The pocket major axis is the 1st principal component of mapped pocket Cα
    atoms; the ligand major axis uses ATP heavy atoms. The angle between axes
    (0–90°) is rotation-invariant within each frame and complements COM distance
    as a 3D orientation descriptor.
    """
    pocket, resids, meta = resolve_pocket_atoms(
        u, pocket_resids=pocket_resids, protein_selection=protein_selection
    )
    pocket_ca = pocket.select_atoms("name CA")
    ligand = u.select_atoms(f"({ligand_selection}) and not name H*")
    if len(ligand) == 0:
        return {"success": False, "error": f"Ligand selection empty: {ligand_selection}"}
    if len(pocket_ca) < 3:
        return {
            "success": False,
            "error": f"Need >=3 pocket Cα atoms; got {len(pocket_ca)}",
        }

    frames, times, distances, angles = [], [], [], []
    step = max(1, int(frame_interval))
    for ts in u.trajectory[::step]:
        # Do NOT wrap pocket/ligand groups independently before COM.
        # ``AtomGroup.wrap(compound="residues")`` on a large MSA-mapped pocket
        # can move residues into different images so the pocket COM jumps far
        # from the ligand even when ATP is stably bound (and even on an already
        # Protein|ATP-wrapped trajectory). Use coordinates as loaded and apply
        # the minimum-image convention to the two COMs instead.
        com_pocket = pocket.center_of_mass()
        com_ligand = ligand.center_of_mass()
        dist = minimum_image_distance(
            com_pocket, com_ligand, getattr(ts, "dimensions", None)
        )

        pocket_axis = _principal_axis(pocket_ca.positions)
        lig_axis = _principal_axis(ligand.positions)
        angle = (
            _axis_angle_deg(pocket_axis, lig_axis)
            if pocket_axis is not None and lig_axis is not None
            else float("nan")
        )

        frames.append(int(ts.frame))
        times.append(float(ts.time) / 1000.0)
        distances.append(dist)
        angles.append(angle)

    cleaned, _, n_spikes = clean_pbc_distance_series(distances)
    if n_spikes:
        distances = [float(v) for v in cleaned]

    with open(distance_output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "time_ns", "distance_angstrom"])
        for fr, t, d in zip(frames, times, distances):
            writer.writerow([fr, f"{t:.6f}", f"{d:.6f}"])

    with open(orientation_output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["frame", "time_ns", "distance_angstrom", "axis_angle_deg"]
        )
        for fr, t, d, ang in zip(frames, times, distances, angles):
            writer.writerow(
                [
                    fr,
                    f"{t:.6f}",
                    f"{d:.6f}",
                    "" if ang != ang else f"{ang:.6f}",
                ]
            )

    stats = summarize_ligand_orientation_series(distances, angles)
    stats["pocket_residue_count"] = len(resids)
    stats["stable_coupling_com_A"] = STABLE_COUPLING_COM_A
    stats["stable_coupling_angle_deg"] = STABLE_COUPLING_ANGLE_DEG
    return {
        "success": True,
        "distance_output": distance_output,
        "orientation_output": orientation_output,
        "pocket_meta": meta,
        **stats,
    }


def compute_consensus_pocket_com_distance(
    u,
    *,
    pocket_resids: List[int],
    ligand_selection: str = "",
    protein_selection: str = "protein",
    output_file: str = "reference_pocket_ligand_distance.csv",
    frame_interval: int = 1,
) -> Dict[str, Any]:
    """Ligand COM to mapped-pocket COM distance time series."""
    orient_file = str(
        Path(output_file).with_name(
            Path(output_file).name.replace(
                "ligand_distance", "ligand_orientation"
            )
        )
    )
    res = compute_consensus_pocket_ligand_geometry(
        u,
        pocket_resids=pocket_resids,
        ligand_selection=ligand_selection,
        protein_selection=protein_selection,
        distance_output=output_file,
        orientation_output=orient_file,
        frame_interval=frame_interval,
    )
    if not res.get("success"):
        return res
    return {
        "success": True,
        "output_file": output_file,
        "orientation_output": orient_file,
        "pocket_meta": res.get("pocket_meta"),
        "mean_distance_A": res.get("mean_distance_A"),
        "std_distance_A": res.get("std_distance_A"),
        "min_distance_A": res.get("min_distance_A"),
        "max_distance_A": res.get("max_distance_A"),
        "mean_axis_angle_deg": res.get("mean_axis_angle_deg"),
        "std_axis_angle_deg": res.get("std_axis_angle_deg"),
        "p95_axis_angle_deg": res.get("p95_axis_angle_deg"),
        "p95_distance_A": res.get("p95_distance_A"),
        "max_distance_A": res.get("max_distance_A"),
        "fraction_stable_coupling": res.get("fraction_stable_coupling"),
        "pocket_residue_count": res.get("pocket_residue_count"),
    }


def _kabsch_superposition(
    mobile: np.ndarray,
    ref: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return rotation matrix and centroids for superposing mobile onto ref."""
    mobile = np.asarray(mobile, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mask = np.isfinite(mobile).all(axis=1) & np.isfinite(ref).all(axis=1)
    if mask.sum() < 3:
        return None
    mob = mobile[mask]
    refm = ref[mask]
    mob_cent = mob.mean(axis=0)
    ref_cent = refm.mean(axis=0)
    m = mob - mob_cent
    r = refm - ref_cent
    cov = m.T @ r
    v, _s, wt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(wt.T @ v.T))
    rot = wt.T @ np.diag([1.0, 1.0, float(d)]) @ v.T
    return rot, mob_cent, ref_cent


def _apply_superposition(
    coords: np.ndarray,
    rot: np.ndarray,
    mob_cent: np.ndarray,
    ref_cent: np.ndarray,
) -> np.ndarray:
    return (np.asarray(coords, dtype=float) - mob_cent) @ rot + ref_cent


def _consensus_ca_resid_pairs(
    consensus_json: Path,
    reference_display: str,
    mobile_display: str,
) -> List[Tuple[int, int]]:
    loaded = load_consensus_alignment(str(consensus_json))
    if not loaded.get("success"):
        return []
    pairs: List[Tuple[int, int]] = []
    for pos in loaded.get("consensus_positions") or []:
        mappings = pos.get("mappings") or {}
        ref_m = mappings.get(reference_display) or {}
        mob_m = mappings.get(mobile_display) or {}
        ref_res = ref_m.get("resid")
        mob_res = mob_m.get("resid")
        if ref_res in (None, "") or mob_res in (None, ""):
            continue
        pairs.append((int(ref_res), int(mob_res)))
    return pairs


def _select_ca_coords(
    u,
    resids: Sequence[int],
    protein_selection: str = "protein",
) -> np.ndarray:
    """Return Cα coordinates for ``resids`` in order; NaN rows when residue missing."""
    coords = np.full((len(resids), 3), np.nan, dtype=float)
    prot = u.select_atoms(protein_selection)
    resid_to_ca: Dict[int, np.ndarray] = {}
    for res in prot.residues:
        try:
            rid = int(res.resid)
        except (TypeError, ValueError):
            continue
        ca = res.atoms.select_atoms("name CA")
        if len(ca) == 1:
            resid_to_ca[rid] = ca.positions[0].astype(float)
    for i, rid in enumerate(resids):
        if rid in resid_to_ca:
            coords[i] = resid_to_ca[rid]
    return coords


def build_reference_ligand_rmsd_template(
    reference_sim_dir: str,
    *,
    consensus_json: Path,
    reference_display: str = "MLKL",
    protein_selection: str = "protein",
    ligand_selection: str = "",
    reference_frame: int = 0,
) -> Dict[str, Any]:
    """Capture reference Cα pairs and ATP heavy-atom coordinates from frame 0."""
    if not HAS_MDA:
        return {"success": False, "error": "MDAnalysis is required"}

    topo, traj = _find_sim_traj_topology(reference_sim_dir)
    if not topo or not traj:
        return {"success": False, "error": f"No topology/trajectory in {reference_sim_dir}"}

    pairs = _consensus_ca_resid_pairs(
        consensus_json, reference_display, reference_display
    )
    ref_resids = [p[0] for p in pairs]
    if len(ref_resids) < 3:
        return {
            "success": False,
            "error": "Insufficient consensus Cα pairs for reference template",
        }

    u = mda.Universe(topo, traj)
    u.trajectory[reference_frame]
    ref_ca = _select_ca_coords(u, ref_resids, protein_selection=protein_selection)
    lig = u.select_atoms(f"({ligand_selection}) and not name H*")
    if len(lig) == 0:
        return {"success": False, "error": f"Reference ligand empty: {ligand_selection}"}

    return {
        "success": True,
        "reference_display": reference_display,
        "consensus_json": str(consensus_json),
        "ref_resids": ref_resids,
        "mob_resids": ref_resids,
        "ref_ca_positions": ref_ca,
        "ref_ligand_positions": lig.positions.copy(),
        "ligand_atom_names": [str(n) for n in lig.names],
        "n_alignment_atoms": int(np.isfinite(ref_ca).all(axis=1).sum()),
        "n_ligand_atoms": len(lig),
    }


def compute_reference_ligand_rmsd(
    u,
    template: Dict[str, Any],
    *,
    mobile_display: str,
    consensus_json: Path,
    protein_selection: str = "protein",
    ligand_selection: str = "",
    frame_interval: int = 1,
    output_file: str = "reference_pocket_ligand_rmsd.csv",
) -> Dict[str, Any]:
    """
    Ligand heavy-atom RMSD to reference ATP after consensus Cα superposition.

    Protein alignment uses sequence-mapped Cα pairs from ``consensus_json``
    (reference = MLKL by default). Ligand RMSD matches atoms by name.
    """
    if not template.get("success"):
        return {"success": False, "error": template.get("error", "Invalid template")}

    pairs = _consensus_ca_resid_pairs(
        consensus_json,
        template["reference_display"],
        mobile_display,
    )
    if len(pairs) < 3:
        return {
            "success": False,
            "error": f"Insufficient consensus Cα pairs for {mobile_display}",
        }

    ref_resids = [p[0] for p in pairs]
    mob_resids = [p[1] for p in pairs]
    template_ref_resids = template.get("ref_resids") or []
    template_ref_ca = np.asarray(template["ref_ca_positions"], dtype=float)
    ref_resid_to_row = {int(r): i for i, r in enumerate(template_ref_resids)}
    ref_ca = np.array(
        [
            template_ref_ca[ref_resid_to_row[r]]
            if r in ref_resid_to_row
            else np.full(3, np.nan)
            for r in ref_resids
        ],
        dtype=float,
    )

    ref_lig_names = list(template["ligand_atom_names"])
    ref_lig_pos = np.asarray(template["ref_ligand_positions"], dtype=float)
    name_to_ref_idx = {n: i for i, n in enumerate(ref_lig_names)}

    frames, times, rmsds = [], [], []
    step = max(1, int(frame_interval))
    for ts in u.trajectory[::step]:
        mob_ca = _select_ca_coords(u, mob_resids, protein_selection=protein_selection)
        sup = _kabsch_superposition(mob_ca, ref_ca)
        if sup is None:
            continue
        rot, mob_cent, ref_cent = sup

        lig = u.select_atoms(f"({ligand_selection}) and not name H*")
        if len(lig) == 0:
            continue
        mob_name_to_pos = {str(n): p for n, p in zip(lig.names, lig.positions)}
        matched_mobile, matched_ref = [], []
        for name, ref_idx in name_to_ref_idx.items():
            if name in mob_name_to_pos:
                matched_mobile.append(mob_name_to_pos[name])
                matched_ref.append(ref_lig_pos[ref_idx])
        if len(matched_mobile) < 3:
            continue
        mob_arr = _apply_superposition(
            np.asarray(matched_mobile, dtype=float), rot, mob_cent, ref_cent
        )
        ref_arr = np.asarray(matched_ref, dtype=float)
        diff = mob_arr - ref_arr
        rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

        frames.append(int(ts.frame))
        times.append(float(ts.time) / 1000.0)
        rmsds.append(rmsd)

    if not rmsds:
        return {"success": False, "error": "No ligand RMSD frames computed"}

    with open(output_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "time_ns", "ligand_rmsd_angstrom"])
        for fr, t, r in zip(frames, times, rmsds):
            writer.writerow([fr, f"{t:.6f}", f"{r:.6f}"])

    arr = np.asarray(rmsds, dtype=float)
    return {
        "success": True,
        "output_file": output_file,
        "mean_ligand_rmsd_A": float(np.mean(arr)),
        "std_ligand_rmsd_A": float(np.std(arr)),
        "min_ligand_rmsd_A": float(np.min(arr)),
        "max_ligand_rmsd_A": float(np.max(arr)),
        "n_matched_ligand_atoms": len(name_to_ref_idx),
        "n_alignment_atoms": int(np.isfinite(ref_ca).all(axis=1).sum()),
    }


def compute_consensus_residence_from_csvs(
    distance_csv: str,
    contacts_csv: str,
    *,
    bound_distance_A: float = 5.0,
    min_contacts: int = 1,
    output_file: str = "reference_pocket_residence.csv",
    working_dir: Optional[str] = None,
    pocket_meta: Optional[Dict[str, Any]] = None,
    debounce_min_run: int = 3,
    bound_mode: str = "distance",
) -> Dict[str, Any]:
    """
    Derive reference-pocket residence from PBC-cleaned distance + contact CSVs.

    By default, bound state is based on ligand-to-reference-pocket COM distance
    only, so the residence plot is directly interpretable against the COM plot.
    ``bound_mode="contact_or_distance"`` preserves the older, more permissive
    behavior for workflows that explicitly want any atom contact to mark binding.
    """
    dist_path = Path(distance_csv)
    contact_path = Path(contacts_csv)
    if not dist_path.is_file() or not contact_path.is_file():
        return {
            "success": False,
            "error": f"Missing distance or contacts CSV: {dist_path}, {contact_path}",
        }

    with open(dist_path, newline="", encoding="utf-8") as fh:
        dist_rows = list(csv.DictReader(fh))
    with open(contact_path, newline="", encoding="utf-8") as fh:
        contact_rows = list(csv.DictReader(fh))
    if not dist_rows or not contact_rows:
        return {"success": False, "error": "Empty distance or contacts CSV"}

    n = min(len(dist_rows), len(contact_rows))
    times: List[float] = []
    bound_mask: List[bool] = []
    min_dists: List[float] = []
    for i in range(n):
        dr, cr = dist_rows[i], contact_rows[i]
        t = float(dr.get("time_ns") or cr.get("time_ns") or 0.0)
        dist = float(dr.get("distance_angstrom") or dr.get("distance_A") or 0.0)
        n_contacts = int(float(cr.get("n_contacts") or 0))
        if bound_mode == "contact_or_distance":
            is_bound = (n_contacts >= int(min_contacts)) or (dist <= bound_distance_A)
        elif bound_mode == "contact_and_distance":
            is_bound = (n_contacts >= int(min_contacts)) and (dist <= bound_distance_A)
        else:
            is_bound = dist <= bound_distance_A
        times.append(t)
        bound_mask.append(is_bound)
        min_dists.append(dist)

    bound_mask = _debounce_bound_mask(bound_mask, min_run=debounce_min_run)
    meta = dict(pocket_meta or {})
    meta["bound_mode"] = bound_mode
    return finalize_ligand_residence_result(
        times,
        bound_mask,
        min_dists,
        bound_distance_A=bound_distance_A,
        min_contacts=min_contacts,
        pocket_meta=meta,
        output_file=output_file,
        working_dir=working_dir,
    )


def compute_consensus_pocket_sasa(
    u,
    *,
    topology_file: str,
    trajectory_file: str,
    pocket_resids: List[int],
    protein_selection: str = "protein",
    probe_radius: float = 0.14,
    output_file: str = "reference_pocket_sasa.csv",
    ndx_path: str = "reference_pocket_sasa.ndx",
    xvg_path: str = "reference_pocket_sasa.xvg",
) -> Dict[str, Any]:
    """Pocket SASA for mapped consensus residues via GROMACS gmx sasa."""
    import subprocess

    gmx = _find_gmx()
    if not gmx:
        return {"success": False, "error": "GROMACS (gmx) not found in PATH"}
    if not str(topology_file).endswith(".tpr"):
        return {
            "success": False,
            "error": "Consensus pocket SASA requires a .tpr topology file",
        }

    pocket, resids, meta = resolve_pocket_atoms(
        u, pocket_resids=pocket_resids, protein_selection=protein_selection
    )
    _write_gmx_index(ndx_path, "ConsensusPocket", pocket.indices)
    cmd = [
        gmx, "sasa",
        "-f", trajectory_file,
        "-s", topology_file,
        "-n", ndx_path,
        "-o", xvg_path,
        "-probe", str(probe_radius),
    ]
    proc = subprocess.run(
        cmd,
        input="ConsensusPocket\nConsensusPocket\n",
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        return {"success": False, "error": f"gmx sasa failed: {proc.stderr[:400]}"}

    times_ps, sasa_nm2 = _parse_xvg_two_columns(xvg_path)
    if len(sasa_nm2) == 0:
        return {"success": False, "error": "No SASA data in gmx output"}

    with open(output_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_ns", "pocket_sasa_nm2"])
        for t, s in zip(times_ps, sasa_nm2):
            writer.writerow([f"{float(t) / 1000.0:.6f}", f"{float(s):.6f}"])

    return {
        "success": True,
        "output_file": output_file,
        "mean_pocket_sasa_nm2": float(np.mean(sasa_nm2)),
        "std_pocket_sasa_nm2": float(np.std(sasa_nm2)),
        "pocket_meta": meta,
        "pocket_resids": resids,
    }


@tool
def define_reference_consensus_pocket(
    working_dir: str,
    reference_label: str,
    sim_dir: str,
    consensus_json: str = DEFAULT_ALIGNMENT_JSON,
    ligand_selection: str = "",
    pocket_cutoff_A: float = 15.0,
    chain_id: Optional[str] = None,
    output_json: str = DEFAULT_DEFINITION_JSON,
    residue_map_csv: str = DEFAULT_RESIDUE_MAP_CSV,
    pocket_filter: str = DEFAULT_POCKET_FILTER,
) -> Dict[str, Any]:
    """
    Define ``pocket_mapped`` residues: 15 Å ligand shell ∩ global consensus.

    Paper default: KAPCA (or ``reference_label``) residues within
    ``pocket_cutoff_A`` Å of the ligand, **intersected with**
    ``global_consensus_msa`` columns (physicochemical-group similarity ≥ 0.5),
    then transferred to every protein via the MSA.

    Modular override: ``pocket_filter='none'`` maps the full 15 Å shell using
    optional ``global_msa.json`` when present (no similarity intersection).

    Args:
        working_dir: Combined analysis output directory.
        reference_label: Reference simulation label (e.g. ``p17612_ATP``).
        sim_dir: Reference simulation root directory.
        consensus_json: MSA JSON; default ``global_consensus_msa.json``.
        ligand_selection: Ligand MDAnalysis selection string.
        pocket_cutoff_A: Distance cutoff in Å (default 15).
        chain_id: Optional protein chain ID.
        output_json: Output pocket definition JSON filename.
        residue_map_csv: Output wide residue map CSV filename.
        pocket_filter: ``global_consensus_msa`` / ``global_mapped`` (default,
            transferable family pocket) or ``none`` (full proximity shell —
            only when user explicitly requests unfiltered; not for comparative
            clustering features).

    Returns:
        Dict with ``success``, pocket position count, output paths.
    """
    if not HAS_MDA:
        return {"success": False, "error": "MDAnalysis is required"}
    if not (ligand_selection or "").strip():
        return {
            "success": False,
            "error": "ligand_selection is required (no silent ATP default)",
        }

    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filt = (pocket_filter or DEFAULT_POCKET_FILTER).strip().lower()
    prefer_full = filt in ("none", "shell", "proximity", "full", "global_msa")
    json_name = Path(consensus_json).name
    consensus_defaults = {
        "",
        "global_consensus_msa.json",
        "global_mapped.json",
        "reference_msa_alignment.json",
    }
    if prefer_full and json_name in consensus_defaults:
        consensus_json = "global_msa.json"
    align_path = resolve_msa_json_for_pocket(
        out_dir, consensus_json, prefer_full=prefer_full
    )
    loaded = load_consensus_alignment(str(align_path))
    if not loaded.get("success"):
        return loaded

    data = loaded["data"]
    consensus_positions = loaded["consensus_positions"]
    labels = [str(l) for l in (data.get("labels") or [])]
    if reference_label not in labels:
        return {
            "success": False,
            "error": f"reference_label {reference_label!r} not in consensus labels",
        }

    topo, traj = _find_sim_traj_topology(sim_dir)
    if not topo or not traj:
        return {"success": False, "error": f"No topology/trajectory in {sim_dir}"}
    topo = str(Path(topo).resolve())
    traj = str(Path(traj).resolve())

    u = mda.Universe(topo, traj)
    pocket_positions_raw, proximity_resids, audit = _consensus_positions_near_ligand(
        u,
        consensus_positions,
        reference_label,
        ligand_selection=ligand_selection,
        pocket_cutoff_A=pocket_cutoff_A,
        chain_id=chain_id,
    )
    if len(pocket_positions_raw) < 3:
        return {
            "success": False,
            "error": (
                f"Only {len(pocket_positions_raw)} consensus pocket positions found; "
                f"need >=3 (cutoff={pocket_cutoff_A} Å)"
            ),
        }

    pocket_positions: List[Dict[str, Any]] = []
    for pos in pocket_positions_raw:
        ref_m = (pos.get("mappings") or {}).get(reference_label) or {}
        pocket_positions.append({
            "consensus_index": pos.get("consensus_index"),
            "msa_col": pos.get("msa_col"),
            "reference_label": reference_label,
            "reference_resid": ref_m.get("resid"),
            "reference_aa": ref_m.get("aa"),
            "reference_seq_index": ref_m.get("seq_index"),
            "mappings": pos.get("mappings") or {},
        })

    per_label_resids = _build_per_label_resid_map(pocket_positions, labels)
    per_label_coverage = {
        lab: _coverage_fraction(pocket_positions, lab) for lab in labels
    }

    definition = {
        "reference_label": reference_label,
        "ligand_selection": ligand_selection,
        "pocket_cutoff_A": float(pocket_cutoff_A),
        "consensus_json": str(align_path),
        "n_consensus_pocket_positions": len(pocket_positions),
        "n_pocket_mapped": len(pocket_positions),
        "mapping_kind": "pocket_mapped",
        "pocket_filter": filt,
        "proximity_resids_reference": proximity_resids,
        "consensus_pocket_positions": pocket_positions,
        "labels": labels,
        "per_label_resids": per_label_resids,
        "per_label_coverage": per_label_coverage,
        "audit": audit,
    }

    json_path = out_dir / output_json
    csv_path = out_dir / residue_map_csv
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(definition, fh, indent=2)
    _write_residue_map_csv(csv_path, pocket_positions, labels)

    _mirror_artifact(json_path, out_dir / DEFAULT_DEFINITION_JSON)
    _mirror_artifact(json_path, out_dir / LEGACY_DEFINITION_JSON)
    _mirror_artifact(csv_path, out_dir / DEFAULT_RESIDUE_MAP_CSV)
    _mirror_artifact(csv_path, out_dir / LEGACY_RESIDUE_MAP_CSV)
    _mirror_artifact(csv_path, out_dir / "reference_pocket_resid_map.csv")

    # Compact audit contract (no full residue lists) for robustness / reproducibility scoring.
    contract = {
        "reference_label": reference_label,
        "ligand_selection": ligand_selection,
        "pocket_cutoff_A": float(pocket_cutoff_A),
        "pocket_filter": filt,
        "msa_json": str(align_path),
        "msa_json_basename": Path(align_path).name,
        "n_proximity_resids": int(len(proximity_resids)),
        "n_pocket_mapped": int(len(pocket_positions)),
        "n_labels": int(len(labels)),
        "mapping_kind": "pocket_mapped",
        "definition_json": str(out_dir / DEFAULT_DEFINITION_JSON),
        "mean_per_label_coverage": (
            float(np.mean(list(per_label_coverage.values())))
            if per_label_coverage
            else None
        ),
    }
    contract_path = out_dir / "pocket_contract.json"
    with open(contract_path, "w", encoding="utf-8") as fh:
        json.dump(contract, fh, indent=2)
        fh.write("\n")
    _mirror_artifact(contract_path, out_dir / "reference_pocket_contract.json")

    return {
        "success": True,
        "message": (
            f"Defined pocket_mapped: {len(pocket_positions)} residues "
            f"(reference={reference_label}, cutoff={pocket_cutoff_A} Å, "
            f"15Å shell={len(proximity_resids)})"
        ),
        "definition_json": str(out_dir / DEFAULT_DEFINITION_JSON),
        "residue_map_csv": str(out_dir / DEFAULT_RESIDUE_MAP_CSV),
        "pocket_contract_json": str(contract_path),
        "n_pocket_positions": len(pocket_positions),
        "n_pocket_mapped": len(pocket_positions),
        "n_proximity_resids": len(proximity_resids),
        "mapping_kind": "pocket_mapped",
        "msa_json": str(align_path),
        "per_label_coverage": per_label_coverage,
        "pocket_contract": contract,
    }


@tool
def map_consensus_pocket_residues(
    working_dir: str,
    definition_json: str = DEFAULT_DEFINITION_JSON,
    labels: Optional[List[str]] = None,
    min_coverage: float = 0.5,
    output_json: str = DEFAULT_DEFINITION_JSON,
    residue_map_csv: str = DEFAULT_RESIDUE_MAP_CSV,
) -> Dict[str, Any]:
    """
    Reload or filter consensus pocket residue mappings per simulation.

    Reads ``reference_pocket_definition.json`` and returns per-label mapped
    PDB resid lists. Labels below ``min_coverage`` are flagged in the output.

    Args:
        working_dir: Combined analysis directory.
        definition_json: Pocket definition JSON from ``define_reference_consensus_pocket``.
        labels: Optional subset of labels to export.
        min_coverage: Minimum mapped-position fraction to mark a label usable.
        output_json: Path to definition JSON (updated with filter flags).
        residue_map_csv: Re-export wide CSV path.

    Returns:
        Dict with ``per_label_resids``, coverage, and usable labels.
    """
    out_dir = Path(working_dir)
    def_path = Path(definition_json)
    if not def_path.is_absolute():
        # LLM plans often pass ``./analysis/foo.json`` while working_dir is already
        # ``.../analysis`` — always resolve by basename under working_dir.
        def_path = out_dir / Path(definition_json).name
    if not def_path.is_file():
        for name in (
            DEFAULT_DEFINITION_JSON,
            LEGACY_DEFINITION_JSON,
            "filtered_pocket_definition.json",
        ):
            cand = out_dir / name
            if cand.is_file():
                def_path = cand
                break
    if not def_path.is_file():
        return {"success": False, "error": f"Pocket definition not found: {def_path}"}

    with open(def_path, encoding="utf-8") as fh:
        definition = json.load(fh)

    pocket_positions = definition.get("consensus_pocket_positions")
    # An MSA JSON copied onto the pocket-definition path has no pocket positions.
    # Do not rewrite the residue map as a header-only CSV.
    if not isinstance(pocket_positions, list) or len(pocket_positions) < 3:
        return {
            "success": False,
            "error": (
                f"Pocket definition {def_path.name} has "
                f"{0 if not isinstance(pocket_positions, list) else len(pocket_positions)} "
                "pocket positions; need >=3. Re-run define_pocket_mapped_residues."
            ),
            "stage": "map",
        }

    all_labels = [str(l) for l in (definition.get("labels") or [])]
    if not all_labels:
        seen = []
        for pos in pocket_positions:
            for key in (pos.get("mappings") or {}):
                if str(key) not in seen:
                    seen.append(str(key))
        all_labels = seen
    requested = [str(l) for l in (labels or all_labels)]
    resolved: List[str] = []
    for lab in requested:
        hit = _resolve_pocket_label(lab, all_labels)
        if hit and hit not in resolved:
            resolved.append(hit)
    # Display names (KAPCA_ATP) do not match UniProt keys (p17612_ATP).
    # Keep the definition labels instead of writing an all-zero resid map.
    if all_labels and len(resolved) < max(1, (len(requested) + 1) // 2):
        resolved = list(all_labels)
    target_labels = resolved or list(all_labels)

    per_label_resids = _build_per_label_resid_map(pocket_positions, all_labels or target_labels)
    for lab in requested:
        hit = _resolve_pocket_label(lab, all_labels)
        if hit and lab not in per_label_resids:
            per_label_resids[lab] = list(per_label_resids.get(hit) or [])
    per_label_coverage = {
        lab: _coverage_fraction(pocket_positions, _resolve_pocket_label(lab, all_labels) or lab)
        for lab in (all_labels or target_labels)
    }
    usable = [
        lab
        for lab in (all_labels or target_labels)
        if per_label_coverage.get(lab, 0.0) >= float(min_coverage)
        and len(per_label_resids.get(lab, [])) >= 3
    ]
    skipped = [lab for lab in (all_labels or target_labels) if lab not in usable]

    definition["per_label_resids"] = per_label_resids
    definition["per_label_coverage"] = per_label_coverage
    definition["usable_labels"] = usable
    definition["skipped_labels"] = skipped
    definition["min_coverage_filter"] = float(min_coverage)

    with open(def_path, "w", encoding="utf-8") as fh:
        json.dump(definition, fh, indent=2)
    csv_path = out_dir / residue_map_csv
    _write_residue_map_csv(csv_path, pocket_positions, all_labels or target_labels)
    _mirror_artifact(def_path, out_dir / DEFAULT_DEFINITION_JSON)
    _mirror_artifact(def_path, out_dir / LEGACY_DEFINITION_JSON)
    _mirror_artifact(csv_path, out_dir / DEFAULT_RESIDUE_MAP_CSV)
    _mirror_artifact(csv_path, out_dir / LEGACY_RESIDUE_MAP_CSV)
    _mirror_artifact(csv_path, out_dir / "reference_pocket_resid_map.csv")

    # Always emit pocket_mapped.json (per-sim resid lists) — axis-angle / COM
    # resolve this file, not the definition JSON. Honor explicit output_json when
    # the caller asks for pocket_mapped.json; otherwise write the standard name.
    mapped_name = Path(output_json).name if output_json else "pocket_mapped.json"
    if mapped_name in (
        DEFAULT_DEFINITION_JSON,
        LEGACY_DEFINITION_JSON,
        Path(definition_json).name,
    ):
        mapped_name = "pocket_mapped.json"
    per_sim: Dict[str, Any] = {}
    for lab, resids in per_label_resids.items():
        clean = [int(r) for r in (resids or []) if r is not None]
        per_sim[str(lab)] = {
            "resids": clean,
            "selection": (
                f"resid {' '.join(str(r) for r in clean)}" if clean else ""
            ),
        }
    pocket_map_payload = {
        "schema_version": "1.0",
        "mapping_kind": "pocket_mapped",
        "reference_label": definition.get("reference_label"),
        "reference_selection": definition.get("ligand_selection")
        or definition.get("reference_selection"),
        "per_sim": per_sim,
        "notes": "Written by map_consensus_pocket_residues from pocket definition",
        "source": def_path.name,
        "usable_labels": usable,
        "per_label_coverage": per_label_coverage,
    }
    mapped_path = out_dir / mapped_name
    mapped_path.write_text(
        json.dumps(pocket_map_payload, indent=2) + "\n", encoding="utf-8"
    )
    if mapped_name != "pocket_mapped.json":
        _mirror_artifact(mapped_path, out_dir / "pocket_mapped.json")
    _mirror_artifact(mapped_path, out_dir / "pocket_map.json")
    # Keep campaign cross_sim/ in sync when working_dir is analysis/.
    try:
        from src.analysis.cross_sim_artifacts import write_pocket_map

        campaign_root = out_dir.parent if out_dir.name in ("analysis", "cross_sim") else out_dir
        write_pocket_map(campaign_root, pocket_map_payload)
    except Exception:
        pass

    return {
        "success": True,
        "message": f"Mapped pocket_mapped residues to {len(usable)}/{len(target_labels)} simulations",
        "definition_json": str(out_dir / DEFAULT_DEFINITION_JSON),
        "residue_map_csv": str(out_dir / DEFAULT_RESIDUE_MAP_CSV),
        "pocket_map_json": str((out_dir / "pocket_mapped.json").resolve()),
        "per_label_resids": per_label_resids,
        "per_label_coverage": per_label_coverage,
        "usable_labels": usable,
        "skipped_labels": skipped,
    }


@tool
def define_pocket_mapped_residues(
    working_dir: str,
    reference_label: str,
    sim_dir: str,
    consensus_json: str = DEFAULT_ALIGNMENT_JSON,
    ligand_selection: str = "",
    pocket_cutoff_A: float = 15.0,
    chain_id: Optional[str] = None,
    output_json: str = DEFAULT_DEFINITION_JSON,
    residue_map_csv: str = DEFAULT_RESIDUE_MAP_CSV,
    pocket_filter: str = DEFAULT_POCKET_FILTER,
) -> Dict[str, Any]:
    """Map the reference ligand pocket ∩ global_consensus_msa onto every system.

    **Default (family / transferable):** ``pocket_filter='global_consensus_msa'``
    (alias ``global_mapped``) — reference residues within ``pocket_cutoff_A`` of
    the ligand **intersected with** conserved MAFFT columns, then transferred
    via the MSA. This is the correct choice for cross-system clustering /
    comparative MD.

    **Override:** ``pocket_filter='none'`` keeps the full proximity shell
    without conservation filtering (much larger residue lists; use only when
    the user explicitly asks for an unfiltered shell, not for family features).

    Cutoff defaults to 15 Å; parse from the user goal when they specify e.g.
    "within 15 Å of ATP".
    """
    return define_reference_consensus_pocket.func(
        working_dir=working_dir,
        reference_label=reference_label,
        sim_dir=sim_dir,
        consensus_json=consensus_json,
        ligand_selection=ligand_selection,
        pocket_cutoff_A=pocket_cutoff_A,
        chain_id=chain_id,
        output_json=output_json,
        residue_map_csv=residue_map_csv,
        pocket_filter=pocket_filter,
    )


@tool
def map_pocket_mapped_residues(
    working_dir: str,
    definition_json: str = DEFAULT_DEFINITION_JSON,
    labels: Optional[List[str]] = None,
    min_coverage: float = 0.5,
    output_json: str = DEFAULT_DEFINITION_JSON,
    residue_map_csv: str = DEFAULT_RESIDUE_MAP_CSV,
) -> Dict[str, Any]:
    """Export per-simulation ``pocket_mapped`` residue lists from the pocket definition."""
    return map_consensus_pocket_residues.func(
        working_dir=working_dir,
        definition_json=definition_json,
        labels=labels,
        min_coverage=min_coverage,
        output_json=output_json,
        residue_map_csv=residue_map_csv,
    )


def _coerce_resid_list(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
        out: List[int] = []
        for p in parts:
            try:
                out.append(int(p))
            except ValueError:
                continue
        return out
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            try:
                if item is None or str(item).strip() == "":
                    continue
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


def discover_pocket_map_json(
    sim_dir: str = "",
    pocket_map_json: str = "",
) -> Optional[Path]:
    if pocket_map_json:
        p = Path(pocket_map_json)
        if p.is_file():
            return p.resolve()
    if not sim_dir:
        return None
    root = Path(sim_dir)
    try:
        from src.analysis.inventory import discover_mapped_path

        found = discover_mapped_path(root, ("pocket_mapped.json", "pocket_map.json"))
        if found:
            return Path(found)
    except Exception:
        pass
    for folder in (root.parent / "analysis", root.parent / "cross_sim"):
        for name in ("pocket_mapped.json", "pocket_map.json"):
            cand = folder / name
            if cand.is_file():
                return cand.resolve()
    return None


def resolve_pocket_resids_for_sim(
    *,
    pocket_resids: Any = None,
    label: str = "",
    pocket_map_json: str = "",
    sim_dir: str = "",
) -> List[int]:
    """Use explicit resids, else ``pocket_mapped.json`` for this label."""
    parsed = _coerce_resid_list(pocket_resids)
    if len(parsed) >= 3:
        return parsed
    path = discover_pocket_map_json(sim_dir, pocket_map_json)
    if not path:
        return parsed
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return parsed
    from src.analysis.cross_sim_artifacts import pocket_resids_for_label

    lab = label or Path(sim_dir).name
    found = pocket_resids_for_label(data, lab)
    return found or parsed


def _write_pocket_collector_aliases(adir: Path, output_prefix: str) -> Dict[str, str]:
    """Copy COM / orientation products to names collectors already search."""
    prefix = _safe_output_prefix(output_prefix)
    aliases: Dict[str, str] = {}
    dist = adir / f"{prefix}_ligand_distance.csv"
    dest = adir / "ligand_pocket_distance.csv"
    if dist.is_file():
        _mirror_artifact(dist, dest)
        aliases["ligand_pocket_distance_csv"] = str(dest)
    orient = adir / f"{prefix}_ligand_orientation.csv"
    dest_o = adir / "pocket_axis_angle.csv"
    if orient.is_file():
        _mirror_artifact(orient, dest_o)
        aliases["pocket_axis_angle_csv"] = str(dest_o)
    rp_dir = adir / "reference_pocket"
    rp_dir.mkdir(parents=True, exist_ok=True)
    metrics = adir / f"{prefix}_metrics.json"
    if metrics.is_file():
        _mirror_artifact(metrics, rp_dir / "reference_pocket_metrics.json")
        aliases["reference_pocket_metrics_json"] = str(
            rp_dir / "reference_pocket_metrics.json"
        )
    if orient.is_file():
        _mirror_artifact(orient, rp_dir / f"{prefix}_ligand_orientation.csv")
    return aliases


@tool
def calculate_consensus_pocket_metrics(
    sim_dir: str,
    pocket_resids: Optional[List[int]] = None,
    label: str = "",
    ligand_selection: str = "",
    protein_selection: str = "protein",
    contact_cutoff: float = 4.0,
    bound_distance_A: float = REFERENCE_POCKET_BOUND_DISTANCE_A,
    frame_interval: int = 1,
    working_dir: Optional[str] = None,
    output_prefix: str = "reference_pocket",
    reference_ligand_template: Optional[Dict[str, Any]] = None,
    mobile_display: str = "",
    consensus_json: Optional[str] = None,
    topology_file: str = "",
    trajectory_file: str = "",
    hpc_dir: str = "",
    pocket_map_json: str = "",
    sim_directory: str = "",
) -> Dict[str, Any]:
    """
    Compute pocket metrics using a consensus-mapped residue list.

    Writes prefixed outputs under ``{sim_dir}/analysis/`` (or ``working_dir``):
      - ``{prefix}_ligand_distance.csv``
      - ``{prefix}_sasa.csv`` (requires .tpr)
      - ``{prefix}_contacts.csv``
      - ``{prefix}_residence.csv`` / ``{prefix}_residence.json``
      - ``{prefix}_rmsf.dat``
      - ``{prefix}_metrics.json`` summary
      - ``ligand_pocket_distance.csv`` / ``pocket_axis_angle.csv`` aliases

    Args:
        sim_dir: Simulation root directory.
        pocket_resids: Mapped PDB residue IDs for this simulation. Optional when
            ``pocket_map_json`` / campaign ``pocket_mapped.json`` is available.
        label: Simulation label (for logging).
        ligand_selection: Ligand selection string.
        protein_selection: Protein selection string.
        contact_cutoff: Heavy-atom contact cutoff (Å).
        bound_distance_A: Bound-state distance threshold (Å).
        frame_interval: Trajectory frame stride.
        working_dir: Output directory (default: ``{sim_dir}/analysis``).
        output_prefix: Filename prefix for outputs.
        topology_file: Bound production topology (preferred over first-hit).
        trajectory_file: Bound production trajectory.
        hpc_dir: Replica slot (``hpc/repXX``). Required when several reps exist.
        pocket_map_json: Path to ``pocket_mapped.json``.

    Returns:
        Dict with ``success``, output file paths, scalar summaries.
    """
    if not HAS_MDA:
        return {"success": False, "error": "MDAnalysis is required"}

    sim_root = sim_directory or sim_dir
    lab = label or Path(sim_root).name
    output_prefix = _safe_output_prefix(output_prefix)
    resids = resolve_pocket_resids_for_sim(
        pocket_resids=pocket_resids,
        label=lab,
        pocket_map_json=pocket_map_json,
        sim_dir=sim_root,
    )
    if len(resids) < 3:
        return {
            "success": False,
            "error": (
                f"Need >=3 pocket resids; got {len(resids)}. "
                "Pass pocket_resids or pocket_map_json / pocket_mapped.json."
            ),
        }

    adir = Path(working_dir) if working_dir else Path(sim_root) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)

    from src.analysis.traj_resolve import resolve_topology_trajectory

    topo, traj = resolve_topology_trajectory(
        topology_file or "",
        trajectory_file or "",
        sim_directory=sim_root,
        hpc_dir=hpc_dir or None,
    )
    if not topo or not traj:
        return {
            "success": False,
            "error": (
                f"No topology/trajectory for {sim_root} "
                f"(hpc_dir={hpc_dir or 'unset'})"
            ),
        }
    topo = str(Path(topo).resolve())
    traj = str(Path(traj).resolve())

    original = _chdir_working(str(adir))
    try:
        u = mda.Universe(topo, traj)
        resids = [int(r) for r in resids]

        pocket_atoms, _resolved_resids, _pocket_meta = resolve_pocket_atoms(
            u, pocket_resids=resids, protein_selection=protein_selection
        )
        charge_res = compute_pocket_net_charge(pocket_atoms)

        com_res = compute_consensus_pocket_com_distance(
            u,
            pocket_resids=resids,
            ligand_selection=ligand_selection,
            protein_selection=protein_selection,
            output_file=f"{output_prefix}_ligand_distance.csv",
            frame_interval=frame_interval,
        )
        if not com_res.get("success"):
            return com_res

        contact_res = compute_protein_ligand_contacts_from_universe(
            u,
            topology_file=topo,
            trajectory_file=traj,
            ligand_selection=ligand_selection,
            protein_selection=protein_selection,
            contact_cutoff=contact_cutoff,
            output_file=f"{output_prefix}_contacts.csv",
            working_dir=str(adir),
            frame_interval=frame_interval,
            pocket_resids=resids,
        )

        residence_res = compute_consensus_residence_from_csvs(
            f"{output_prefix}_ligand_distance.csv",
            f"{output_prefix}_contacts.csv",
            bound_distance_A=bound_distance_A,
            output_file=f"{output_prefix}_residence.csv",
            working_dir=str(adir),
            pocket_meta=com_res.get("pocket_meta") or {"pocket_residue_count": len(resids)},
        )
        rmsf_res = compute_pocket_rmsf_from_universe(
            u,
            topology_file=topo,
            trajectory_file=traj,
            ligand_selection=ligand_selection,
            protein_selection=protein_selection,
            output_file=f"{output_prefix}_rmsf.dat",
            working_dir=str(adir),
            pocket_resids=resids,
        )

        sasa_res: Dict[str, Any] = {"success": False, "skipped": True}
        if str(topo).endswith(".tpr"):
            sasa_res = compute_consensus_pocket_sasa(
                u,
                topology_file=topo,
                trajectory_file=traj,
                pocket_resids=resids,
                protein_selection=protein_selection,
                output_file=f"{output_prefix}_sasa.csv",
            )

        summary = {
            "label": lab,
            "sim_directory": str(Path(sim_root).resolve()),
            "topology_file": topo,
            "trajectory_file": traj,
            "hpc_dir": hpc_dir or "",
            "pocket_residue_count": len(resids),
            "pocket_resids": resids,
            "ligand_pocket_distance_mean_A": com_res.get("mean_distance_A"),
            "ligand_pocket_distance_std_A": com_res.get("std_distance_A"),
            "mean_hbonds": contact_res.get("mean_hbonds") if contact_res.get("success") else None,
            "mean_pocket_sasa_nm2": sasa_res.get("mean_pocket_sasa_nm2"),
            "fraction_bound": residence_res.get("fraction_bound"),
            "mean_pocket_rmsf_A": rmsf_res.get("mean_pocket_rmsf"),
            "mean_axis_angle_deg": com_res.get("mean_axis_angle_deg"),
            "std_axis_angle_deg": com_res.get("std_axis_angle_deg"),
            "p95_axis_angle_deg": com_res.get("p95_axis_angle_deg"),
            "ligand_pocket_distance_p95_A": com_res.get("p95_distance_A"),
            "ligand_pocket_distance_max_A": com_res.get("max_distance_A"),
            "fraction_stable_coupling": com_res.get("fraction_stable_coupling"),
            "stable_coupling_com_A": com_res.get("stable_coupling_com_A"),
            "stable_coupling_angle_deg": com_res.get("stable_coupling_angle_deg"),
            "bound_distance_A": bound_distance_A,
            "pocket_net_charge": charge_res.get("net_charge"),
            "pocket_n_positive": charge_res.get("n_positive"),
            "pocket_n_negative": charge_res.get("n_negative"),
            "outputs": {
                "ligand_distance_csv": f"{output_prefix}_ligand_distance.csv",
                "ligand_orientation_csv": f"{output_prefix}_ligand_orientation.csv",
                "contacts_csv": f"{output_prefix}_contacts.csv",
                "residence_csv": f"{output_prefix}_residence.csv",
                "residence_json": f"{output_prefix}_residence.json",
                "rmsf_dat": f"{output_prefix}_rmsf.dat",
                "sasa_csv": f"{output_prefix}_sasa.csv" if sasa_res.get("success") else None,
            },
        }
        summary_path = f"{output_prefix}_metrics.json"
        Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        try:
            aliases = _write_pocket_collector_aliases(adir, output_prefix)
        except Exception as alias_exc:
            logger.warning("Pocket alias copy skipped: %s", alias_exc)
            aliases = {}
        if aliases:
            summary["outputs"].update(aliases)
            Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
            rp_metrics = adir / "reference_pocket" / "reference_pocket_metrics.json"
            _mirror_artifact(adir / summary_path, rp_metrics)

        append_analysis_summary(
            working_dir=str(adir),
            analysis_type="ConsensusPocketMetrics",
            statistics={
                k: summary[k]
                for k in (
                    "ligand_pocket_distance_mean_A",
                    "mean_hbonds",
                    "mean_pocket_sasa_nm2",
                    "fraction_bound",
                    "mean_pocket_rmsf_A",
                    "mean_axis_angle_deg",
                    "std_axis_angle_deg",
                    "ligand_pocket_distance_p95_A",
                    "fraction_stable_coupling",
                    "pocket_net_charge",
                )
                if summary.get(k) is not None
            },
            files=summary["outputs"],
            metadata={
                "pocket_residue_count": len(resids),
                "pocket_definition_mode": "consensus_resid_list",
            },
        )

        ok = com_res.get("success") and contact_res.get("success")
        return {
            "success": bool(ok),
            "message": f"Consensus pocket metrics for {lab}",
            "metrics_json": str(adir / summary_path),
            **summary,
        }
    except Exception as exc:
        logger.exception("calculate_consensus_pocket_metrics failed")
        return {"success": False, "error": str(exc)}
    finally:
        _restore_cwd(original)


@tool
def run_consensus_pocket_metrics_batch(
    sim_dirs: List[str],
    labels: List[str],
    working_dir: str,
    reference_label: str = "q8nb16",
    reference_sim_dir: str = "",
    consensus_json: str = DEFAULT_ALIGNMENT_JSON,
    definition_json: str = DEFAULT_DEFINITION_JSON,
    ligand_selection: str = "",
    pocket_cutoff_A: float = 15.0,
    min_coverage: float = 0.5,
    frame_interval: int = 1,
    rebuild_definition: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end consensus pocket workflow for a multi-simulation project.

    1. Define reference consensus pocket (if missing or ``rebuild_definition``)
    2. Map pocket resids to each simulation
    3. Compute mapped reference-pocket metrics under
       ``{working_dir}/reference_pocket/{uniprot}/``

    Args:
        sim_dirs: Per-simulation directories (parallel to ``labels``).
        labels: Simulation labels.
        working_dir: Combined analysis directory.
        reference_label: Reference label (e.g. ``q8nb16`` MLKL).
        reference_sim_dir: Reference sim directory (inferred from labels if empty).
        consensus_json: Consensus alignment JSON path (relative to working_dir).
        definition_json: Pocket definition JSON filename.
        ligand_selection: Ligand selection string.
        pocket_cutoff_A: Reference pocket cutoff in Å.
        min_coverage: Minimum mapping coverage to run metrics.
        frame_interval: Trajectory frame stride.
        rebuild_definition: Force re-definition of reference pocket.

    Returns:
        Dict with definition paths, per-label results, and failure list.
    """
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _match_ref(candidate: str, lab: str, sd: str) -> bool:
        """Match UniProt id, display label, or ``{id}_ATP`` sim folder names."""
        c = str(candidate or "").lower().strip()
        if not c:
            return False
        l = str(lab or "").lower().strip()
        name = Path(sd).name.lower()
        if c in {l, name}:
            return True
        # p17612 ↔ p17612_ATP / KAPCA_ATP folder prefixes
        for other in (l, name):
            if other.startswith(c + "_") or c.startswith(other + "_"):
                return True
            # strip common ligand suffixes before compare
            for suf in ("_atp", "_adp", "_amp"):
                if other.endswith(suf) and other[: -len(suf)] == c:
                    return True
                if c.endswith(suf) and c[: -len(suf)] == other:
                    return True
        return False

    def_path = out_dir / definition_json
    # Prefer an existing pocket definition's reference label when callers pass a
    # mismatched default (e.g. q8nb16) while pre_combined wrote p17612_ATP.
    if def_path.is_file() and not rebuild_definition:
        try:
            with open(def_path, encoding="utf-8") as _dfh:
                _def = json.load(_dfh)
            def_ref = str((_def or {}).get("reference_label") or "").strip()
            if def_ref:
                reference_label = def_ref
        except Exception:
            pass

    ref_sim = reference_sim_dir
    ref_key = str(reference_label).lower()
    if not ref_sim:
        from src.analysis.reference_labels import build_uniprot_display_map

        uid_to_disp = build_uniprot_display_map(out_dir)
        disp_to_uid = {v: k for k, v in uid_to_disp.items()}
        ref_uid = disp_to_uid.get(reference_label, ref_key)
        for sd, lab in zip(sim_dirs, labels):
            if (
                _match_ref(ref_key, lab, sd)
                or _match_ref(reference_label, lab, sd)
                or _match_ref(ref_uid, lab, sd)
            ):
                ref_sim = sd
                break
    if not ref_sim:
        # Last resort: first sim_dir whose folder matches reference_label loosely.
        for sd, lab in zip(sim_dirs, labels):
            if _match_ref(reference_label, lab, sd):
                ref_sim = sd
                break
    if not ref_sim:
        return {
            "success": False,
            "error": f"reference_sim_dir not found for {reference_label!r}",
        }

    if rebuild_definition or not def_path.is_file():
        def_res = define_reference_consensus_pocket.func(
            working_dir=str(out_dir),
            reference_label=reference_label,
            sim_dir=ref_sim,
            consensus_json=consensus_json,
            ligand_selection=ligand_selection,
            pocket_cutoff_A=pocket_cutoff_A,
            output_json=definition_json,
        )
        if not def_res.get("success"):
            return {"success": False, "stage": "define", **def_res}

    # Map with the definition's own labels. Passing gene-name labels
    # (KAPCA_ATP) against UniProt keys (p17612_ATP) zeroed every resid list.
    map_res = map_consensus_pocket_residues.func(
        working_dir=str(out_dir),
        definition_json=definition_json,
        labels=None,
        min_coverage=min_coverage,
    )
    if not map_res.get("success"):
        return {"success": False, "stage": "map", **map_res}

    per_label_resids: Dict[str, List[int]] = map_res.get("per_label_resids") or {}
    usable = set(map_res.get("usable_labels") or [])
    pocket_root = out_dir / "reference_pocket"
    pocket_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    failed: List[str] = []
    known_keys = list(per_label_resids) or list(usable)
    for sd, lab in zip(sim_dirs, labels):
        lab = str(lab)
        sim_uid = Path(sd).name
        canon = (
            _resolve_pocket_label(sim_uid, known_keys)
            or _resolve_pocket_label(lab, known_keys)
        )
        if not canon or canon not in usable:
            failed.append(sim_uid or lab)
            results[sim_uid or lab] = {
                "success": False,
                "error": "insufficient pocket mapping coverage",
                "coverage": (map_res.get("per_label_coverage") or {}).get(canon or lab),
                "requested_label": lab,
            }
            continue
        resids = per_label_resids.get(canon) or per_label_resids.get(sim_uid) or []
        mres = calculate_consensus_pocket_metrics.func(
            sim_dir=sd,
            pocket_resids=resids,
            label=sim_uid or lab,
            ligand_selection=ligand_selection,
            frame_interval=frame_interval,
            working_dir=str(pocket_root / sim_uid),
            output_prefix="reference_pocket",
        )
        results[sim_uid or lab] = mres
        if not mres.get("success"):
            failed.append(sim_uid or lab)

    manifest = {
        "reference_label": reference_label,
        "definition_json": str(def_path),
        "residue_map_csv": str(out_dir / DEFAULT_RESIDUE_MAP_CSV),
        "output_root": str(pocket_root),
        "n_processed": sum(1 for r in results.values() if r.get("success")),
        "n_failed": len(failed),
        "failed_labels": failed,
        "usable_labels": sorted(usable),
        "usable_sim_ids": sorted(
            Path(sd).name
            for sd, lab in zip(sim_dirs, labels)
            if (
                _resolve_pocket_label(Path(sd).name, known_keys)
                or _resolve_pocket_label(str(lab), known_keys)
            )
            in usable
        ),
        "per_label_coverage": map_res.get("per_label_coverage"),
    }
    manifest_path = out_dir / "reference_pocket_batch_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return {
        "success": len(failed) < len(labels),
        "message": (
            f"Consensus pocket metrics: {manifest['n_processed']}/{len(labels)} simulations"
        ),
        "manifest_file": str(manifest_path),
        "definition_json": str(def_path),
        "results": results,
        "failed_labels": failed,
    }
