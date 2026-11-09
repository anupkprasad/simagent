"""Modular PCA/tICA on dihedral or Cartesian features (independent or shared-ref)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _find_dihedral_dirs(sim_analysis: Path) -> List[Path]:
    """Locate dihedral feature dirs under avg/, top-level, or rep*/ (absolute Paths).

    Prefers ``rep*/consensus_dihedrals`` (and nested avg/consensus_dihedrals) that
    carry a ``per_residue_dihedrals.csv`` with ``consensus_index`` columns — required
    for shared-reference PCA. Flat ``avg/phi_deg.npy`` collapse is a last resort.
    """
    root = Path(sim_analysis).resolve()
    primary: List[Path] = []
    fallback: List[Path] = []

    def _has_phi(d: Path) -> bool:
        return (d / "phi_deg.npy").is_file()

    def _has_schema(d: Path) -> bool:
        csv_path = d / "per_residue_dihedrals.csv"
        if not csv_path.is_file():
            return False
        try:
            head = csv_path.read_text(encoding="utf-8", errors="replace")[:200]
        except OSError:
            return False
        return "consensus_index" in head

    for d in (
        root / "avg" / "consensus_dihedrals",
        root / "consensus_dihedrals",
    ):
        if _has_phi(d):
            (primary if _has_schema(d) else fallback).append(d)
    for rep in sorted(root.glob("rep*")):
        d = rep / "consensus_dihedrals"
        if _has_phi(d):
            (primary if _has_schema(d) else fallback).append(d)
        elif _has_phi(rep):
            (primary if _has_schema(rep) else fallback).append(rep)
    avg = root / "avg"
    if _has_phi(avg):
        (primary if _has_schema(avg) else fallback).append(avg)
    # Deduplicate while preserving order.
    seen = set()
    out: List[Path] = []
    for d in primary + fallback:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _is_replicate_dihedral_dir(path: Path) -> bool:
    """True when path lives under a ``rep*`` folder (not pooled ``avg/``)."""
    return any(part.lower().startswith("rep") for part in path.parts)


def _schema_dihedral_dirs(sim_analysis: Path) -> List[Path]:
    """Consensus-schema dihedral dirs, preferring replicates over pooled avg."""
    dirs = [
        d
        for d in _find_dihedral_dirs(sim_analysis)
        if (d / "per_residue_dihedrals.csv").is_file()
        and "consensus_index"
        in (d / "per_residue_dihedrals.csv").read_text(
            encoding="utf-8", errors="replace"
        )[:200]
    ]
    reps = [d for d in dirs if _is_replicate_dihedral_dir(d)]
    return reps if reps else dirs


def _reference_fit_dihedral_dir(sim_analysis: Path) -> Optional[Path]:
    """Paper shared-PCA fit: first replicate only (``rep01`` / ``rep1``)."""
    dirs = _schema_dihedral_dirs(sim_analysis)
    if not dirs:
        return None
    for d in dirs:
        parts_l = [p.lower() for p in d.parts]
        if "rep01" in parts_l or "rep1" in parts_l:
            return d
    return dirs[0]


def _resolve_campaign_base(base_directory: str, working_dir: Optional[str] = None) -> Path:
    """Return the multi-sim campaign root (parent of ``*_ATP`` dirs), not ``analysis/``."""
    base = Path(base_directory).resolve()
    if base.name == "analysis" and any(base.parent.glob("*_ATP")):
        return base.parent
    if any(base.glob("*_ATP")) or any(base.glob("p[0-9]*")) or any(base.glob("q[0-9]*")):
        return base
    if working_dir:
        wd = Path(working_dir).resolve()
        if wd.name == "analysis" and any(wd.parent.glob("*_ATP")):
            return wd.parent
    return base


def _load_dihedral_residue_table(dihedral_dir: Path):
    """Load per-residue consensus_index table for a dihedral output directory."""
    import pandas as pd

    csv_path = Path(dihedral_dir) / "per_residue_dihedrals.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    if "consensus_index" not in df.columns:
        raise ValueError(f"No consensus_index column in {csv_path}")
    df["consensus_index"] = df["consensus_index"].astype(int)
    if "has_chi1" in df.columns:
        df["has_chi1"] = df["has_chi1"].astype(str).str.lower().isin(
            ("1", "true", "yes")
        )
    else:
        df["has_chi1"] = True
    return df


def _shared_dihedral_schema(
    ref_df,
    angles: Sequence[str],
) -> List[Dict[str, Any]]:
    """Fixed feature schema from the reference residue table (paper shared-PCA style)."""
    angles_l = [a.lower() for a in angles]
    schema: List[Dict[str, Any]] = []
    for _, row in ref_df.iterrows():
        ci = int(row["consensus_index"])
        if "phi" in angles_l:
            schema.append({"consensus_index": ci, "kind": "phi_sin"})
            schema.append({"consensus_index": ci, "kind": "phi_cos"})
        if "psi" in angles_l:
            schema.append({"consensus_index": ci, "kind": "psi_sin"})
            schema.append({"consensus_index": ci, "kind": "psi_cos"})
        if "chi1" in angles_l and bool(row.get("has_chi1", True)):
            schema.append({"consensus_index": ci, "kind": "chi1_sin"})
            schema.append({"consensus_index": ci, "kind": "chi1_cos"})
    return schema


def _build_aligned_dihedral_matrix(
    dihedral_dir: Path,
    schema: Sequence[Dict[str, Any]],
    fill: np.ndarray,
) -> np.ndarray:
    """
    Build n_frames × len(schema) features aligned to the reference schema.

    Missing consensus columns (or NaN angles) are filled with ``fill`` (typically
    the reference feature mean) so every system shares the same PCA dimension.
    """
    ddir = Path(dihedral_dir)
    df = _load_dihedral_residue_table(ddir)
    phi = np.load(ddir / "phi_deg.npy")
    psi = np.load(ddir / "psi_deg.npy") if (ddir / "psi_deg.npy").is_file() else None
    chi_path = ddir / "chi1_deg.npy"
    chi = np.load(chi_path) if chi_path.is_file() else None
    ci_to_j = {int(r.consensus_index): int(j) for j, r in df.iterrows()}
    n_frames = int(phi.shape[0])
    X = np.tile(np.asarray(fill, dtype=float).reshape(1, -1), (n_frames, 1))
    for j, col in enumerate(schema):
        ci = int(col["consensus_index"])
        kind = str(col["kind"])
        if ci not in ci_to_j:
            continue
        jj = ci_to_j[ci]
        if kind.startswith("phi"):
            ang = phi[:, jj]
        elif kind.startswith("psi"):
            if psi is None:
                continue
            ang = psi[:, jj]
        elif kind.startswith("chi1"):
            if chi is None:
                continue
            ang = chi[:, jj]
        else:
            continue
        rad = np.deg2rad(ang.astype(float))
        vals = np.sin(rad) if kind.endswith("sin") else np.cos(rad)
        bad = ~np.isfinite(ang.astype(float))
        vals = np.where(bad, fill[j], vals)
        X[:, j] = vals
    return X


def _resolve_dihedral_dir(
    dihedral_dir: str = "",
    *,
    sim_directory: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> Optional[Path]:
    """Resolve dihedral feature dir to an absolute path; never rely on cwd."""
    candidates: List[Path] = []
    if dihedral_dir:
        p = Path(dihedral_dir)
        if p.is_absolute():
            candidates.append(p)
        else:
            if working_dir:
                candidates.append(Path(working_dir).resolve() / p)
            if sim_directory:
                candidates.append(Path(sim_directory).resolve() / "analysis" / p)
                candidates.append(Path(sim_directory).resolve() / p)
            candidates.append(Path.cwd() / p)
    if sim_directory:
        candidates.extend(_find_dihedral_dirs(Path(sim_directory).resolve() / "analysis"))
    if working_dir:
        # project often sets working_dir = {sim}/analysis
        candidates.extend(_find_dihedral_dirs(Path(working_dir).resolve()))
    for c in candidates:
        if (c / "phi_deg.npy").is_file():
            return c.resolve()
    return None


def _resolve_traj(
    topology_file: str,
    trajectory_file: str,
    sim_directory: Optional[str],
    hpc_dir: Optional[str] = None,
):
    from src.analysis.traj_resolve import resolve_topology_trajectory

    return resolve_topology_trajectory(
        topology_file,
        trajectory_file,
        sim_directory=sim_directory,
        hpc_dir=hpc_dir,
    )


def _load_dihedral_matrix(
    dihedral_dir: Path,
    angles: Sequence[str],
) -> np.ndarray:
    from src.analysis.family_dynamics_core import build_dihedral_feature_matrix

    feat = dihedral_dir / "dihedral_sincos.npy"
    if feat.is_file() and set(a.lower() for a in angles) >= {"phi", "psi", "chi1"}:
        # Full matrix already built with default angles; reload if angles subset needed
        if set(a.lower() for a in angles) == {"phi", "psi", "chi1"}:
            return np.load(feat)
    phi = np.load(dihedral_dir / "phi_deg.npy")
    psi = np.load(dihedral_dir / "psi_deg.npy")
    chi_path = dihedral_dir / "chi1_deg.npy"
    chi = np.load(chi_path) if chi_path.is_file() else None
    has_chi = None
    if chi is not None:
        has_chi = [bool(np.isfinite(chi[:, j]).mean() > 0.1) for j in range(chi.shape[1])]
    return build_dihedral_feature_matrix(
        phi, psi, chi, angles=angles, has_chi1_mask=has_chi
    )


def _load_cartesian_matrix(
    topology_file: str,
    trajectory_file: str,
    alignment_json: str,
    label: str,
    selection: str = "protein and name CA",
) -> np.ndarray:
    """Consensus-mapped Cα coordinates, flattened (n_frames × 3n)."""
    import MDAnalysis as mda
    from MDAnalysis.analysis import align as mda_align

    from src.analysis.family_dynamics_core import (
        load_consensus_alignment,
        mapping_for_label,
    )

    alignment = load_consensus_alignment(alignment_json)
    mapping = mapping_for_label(alignment, label)
    if not mapping:
        raise RuntimeError(f"No mapping for {label}")

    u = mda.Universe(str(Path(topology_file).resolve()), str(Path(trajectory_file).resolve()))
    protein = {int(r.resid): r for r in u.select_atoms("protein").residues}
    resids = []
    for m in mapping:
        if not m or m.get("resid") is None:
            continue
        resid = int(m["resid"])
        if resid in protein:
            resids.append(resid)
    if len(resids) < 5:
        raise RuntimeError(f"Too few mapped Cα residues ({len(resids)})")

    sel = " or ".join(f"resid {r}" for r in resids)
    ag = u.select_atoms(f"({selection}) and ({sel})")
    if ag.n_atoms < 5:
        # fallback: name CA on those residues
        ag = u.select_atoms(f"protein and name CA and ({sel})")
    if ag.n_atoms < 5:
        raise RuntimeError(f"Bad Cα selection n={ag.n_atoms}")

    # Align to frame 0
    ref = u.copy()
    aligner = mda_align.AlignTraj(u, ref, select=f"protein and name CA and ({sel})", in_memory=True)
    aligner.run()

    coords = []
    for _ts in u.trajectory:
        coords.append(ag.positions.copy().reshape(-1))
    return np.asarray(coords, dtype=float)


def _fit_reduce(
    X: np.ndarray,
    *,
    method: str,
    n_components: int,
    lag_frames: int,
    whiten_dim: int,
) -> Dict[str, Any]:
    from src.analysis.family_dynamics_core import fit_pca, fit_pca_whiten, fit_tica

    method = method.lower()
    if method == "pca":
        fit = fit_pca(X, n_components=max(n_components, 2))
        return {
            "method": "pca",
            "mean": fit["mean"],
            "components": fit["components"],
            "projections": fit["projections"],
            "explained_variance_fraction": fit["explained_variance_fraction"],
        }
    if method == "tica":
        Y, mean, Vt, scale = fit_pca_whiten(X, dim=whiten_dim)
        tica = fit_tica(Y, lag=lag_frames, n_tics=max(n_components, 2))
        return {
            "method": "tica",
            "pca_mean": mean,
            "pca_components": Vt,
            "pca_scale": scale,
            "eigenvectors": tica["eigenvectors"],
            "eigenvalues": tica["eigenvalues"],
            "timescales_frames": tica["timescales_frames"],
            "projections": tica["projections"],
            "lag_frames": lag_frames,
        }
    raise ValueError(f"method must be pca or tica, got {method}")


def _project_reduce(X: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    from src.analysis.family_dynamics_core import project_pca, project_tica

    method = str(model.get("method", "pca")).lower()
    if method == "pca":
        return project_pca(X, model["mean"], model["components"])
    # tica: whiten then project
    Y = (X - model["pca_mean"]) @ model["pca_components"].T
    scale = model["pca_scale"]
    Y = Y / scale
    return project_tica(Y, model["eigenvectors"])


def _write_dynamics_result(
    out_dir: Path,
    proj: np.ndarray,
    *,
    method: str,
    meta_extra: Dict[str, Any],
) -> Dict[str, Any]:
    from src.analysis.family_dynamics_core import fel_descriptors_from_proj2, write_fel_outputs

    if proj.shape[1] < 2:
        raise RuntimeError("Need ≥2 components for FEL")
    desc = fel_descriptors_from_proj2(proj[:, :2])
    paths = write_fel_outputs(
        out_dir, proj=proj[:, :2], desc=desc, method=method, meta_extra=meta_extra
    )
    return {
        "success": True,
        "grid_entropy": desc["grid_entropy"],
        "major_basin_population": desc["major_basin_population"],
        "landscape_entropy": desc["landscape_entropy"],
        "n_basins": desc["n_basins"],
        **paths,
        **meta_extra,
    }


@tool
def run_independent_dynamics_fel(
    space: str = "dihedral",
    method: str = "pca",
    label: str = "",
    topology_file: str = "",
    trajectory_file: str = "",
    alignment_json: str = "",
    dihedral_dir: str = "consensus_dihedrals",
    dihedral_angles: Optional[List[str]] = None,
    selection: str = "protein and name CA",
    lag_frames: int = 10,
    n_components: int = 10,
    whiten_dim: int = 50,
    output_dir: str = "",
    working_dir: Optional[str] = None,
    sim_directory: Optional[str] = None,
    hpc_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Fit PCA or tICA on one simulation and write a 2D FEL + grid entropy.

    **space:** ``dihedral`` (needs prior ``calculate_consensus_torsions``) or
    ``cartesian`` (consensus-mapped Cα; needs ``alignment_json``).

    **method:** ``pca`` or ``tica``.

    Always uses **independent** (per-simulation) projection. For shared-reference
    mode use ``fit_dynamics_model`` then ``project_dynamics_model``.

    Output directory defaults from (space, method): e.g. ``consensus_PCA``,
    ``consensus_TICA``, ``consensus_cart_PCA``, ``consensus_cart_TICA``.
    """
    from src.analysis.family_dynamics_core import dynamics_output_dirname

    original_dir = None
    try:
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            original_dir = os.getcwd()
            os.chdir(working_dir)

        space_l = space.lower().strip()
        method_l = method.lower().strip()
        out_name = output_dir or dynamics_output_dirname(
            space_l, method_l, "independent"
        )
        out = Path(out_name)
        feat_path = out / "fel_features.json"
        if feat_path.is_file() and not overwrite:
            return {
                "success": True,
                "skipped": True,
                **json.loads(feat_path.read_text(encoding="utf-8")),
                "output_dir": str(out.resolve()),
            }

        angles = list(dihedral_angles or ["phi", "psi", "chi1"])
        if space_l == "dihedral":
            X = _load_dihedral_matrix(Path(dihedral_dir), angles)
        elif space_l == "cartesian":
            if not label or not alignment_json:
                return {
                    "success": False,
                    "error": "cartesian space requires label and alignment_json",
                }
            top, traj = _resolve_traj(
                topology_file, trajectory_file, sim_directory, hpc_dir=hpc_dir
            )
            if not top or not traj:
                return {"success": False, "error": "Missing topology/trajectory"}
            X = _load_cartesian_matrix(top, traj, alignment_json, label, selection)
        else:
            return {"success": False, "error": f"Unknown space={space}"}

        if X.shape[0] < max(lag_frames + 5, 20) or X.shape[1] < 4:
            return {
                "success": False,
                "error": f"Insufficient data shape {X.shape}",
            }

        model = _fit_reduce(
            X,
            method=method_l,
            n_components=n_components,
            lag_frames=lag_frames,
            whiten_dim=whiten_dim,
        )
        proj = model["projections"]
        meta = {
            "label": label,
            "space": space_l,
            "method": method_l,
            "projection_mode": "independent",
            "n_features": int(X.shape[1]),
            "dihedral_angles": angles if space_l == "dihedral" else None,
        }
        result = _write_dynamics_result(out, proj, method=method_l, meta_extra=meta)
        result["output_dir"] = str(out.resolve())
        return result
    except Exception as exc:
        logger.exception("run_independent_dynamics_fel failed")
        return {"success": False, "error": str(exc)}
    finally:
        if original_dir:
            os.chdir(original_dir)


@tool
def fit_dynamics_model(
    space: str = "dihedral",
    method: str = "pca",
    reference_label: str = "",
    topology_file: str = "",
    trajectory_file: str = "",
    alignment_json: str = "",
    dihedral_dir: str = "",
    dihedral_angles: Optional[List[str]] = None,
    selection: str = "protein and name CA",
    lag_frames: int = 10,
    n_components: int = 10,
    whiten_dim: int = 50,
    model_path: str = "dynamics_model",
    working_dir: Optional[str] = None,
    sim_directory: Optional[str] = None,
    base_directory: str = "",
) -> Dict[str, Any]:
    """
    Fit a shared-reference PCA or tICA model on one reference simulation.

    Use with ``project_dynamics_model`` for family-comparable FELs.
    Saves ``{model_path}.npz`` + ``.json`` under ``working_dir`` (or analysis/).
    """
    from src.analysis.family_dynamics_core import save_model_npz

    if not reference_label:
        return {"success": False, "error": "reference_label is required"}

    # Resolve paths absolute BEFORE any chdir so multi-rep layouts work.
    wd_abs = Path(working_dir).resolve() if working_dir else None
    base_abs = Path(base_directory).resolve() if base_directory else None
    sim_abs = Path(sim_directory).resolve() if (sim_directory and Path(sim_directory).exists()) else None
    if sim_abs is None and base_abs is not None:
        cand = base_abs / reference_label
        if cand.is_dir():
            sim_abs = cand

    original_dir = None
    try:
        if wd_abs is not None:
            wd_abs.mkdir(parents=True, exist_ok=True)
            original_dir = os.getcwd()
            os.chdir(str(wd_abs))

        space_l = space.lower().strip()
        method_l = method.lower().strip()
        angles = list(dihedral_angles or ["phi", "psi", "chi1"])

        ref_sim = sim_abs
        if ref_sim is None and base_abs is not None:
            cand = base_abs / reference_label
            if cand.is_dir():
                ref_sim = cand

        if space_l == "dihedral":
            ddir = _resolve_dihedral_dir(
                dihedral_dir,
                sim_directory=str(ref_sim) if ref_sim else None,
                working_dir=str(wd_abs) if wd_abs else None,
            )
            if ddir is None:
                return {
                    "success": False,
                    "error": (
                        "No consensus_dihedrals/phi_deg.npy found under "
                        f"sim={ref_sim} working_dir={wd_abs} dihedral_dir={dihedral_dir!r}. "
                        "Run calculate_consensus_torsions first (avg/ or rep*/)."
                    ),
                }
            X = _load_dihedral_matrix(ddir, angles)
        elif space_l == "cartesian":
            if not alignment_json:
                return {"success": False, "error": "alignment_json required for cartesian"}
            top, traj = topology_file, trajectory_file
            if ref_sim:
                top, traj = _resolve_traj(top, traj, str(ref_sim))
            else:
                top, traj = _resolve_traj(top, traj, sim_directory)
            if not top or not traj:
                return {"success": False, "error": "Missing reference trajectory"}
            align_p = Path(alignment_json)
            if not align_p.is_absolute() and wd_abs is not None:
                align_p = wd_abs / align_p
            X = _load_cartesian_matrix(
                top, traj, str(align_p), reference_label, selection
            )
        else:
            return {"success": False, "error": f"Unknown space={space}"}

        model = _fit_reduce(
            X,
            method=method_l,
            n_components=n_components,
            lag_frames=lag_frames,
            whiten_dim=whiten_dim,
        )
        # Drop large projections from saved model (refit not needed for project)
        save_model = {k: v for k, v in model.items() if k != "projections"}
        save_model.update(
            {
                "space": space_l,
                "method": method_l,
                "projection_mode": "shared_reference",
                "reference_label": reference_label,
                "dihedral_angles": angles if space_l == "dihedral" else None,
                "n_features": int(X.shape[1]),
                "selection": selection if space_l == "cartesian" else None,
            }
        )
        out = Path(model_path)
        if not out.is_absolute() and wd_abs is not None:
            out = wd_abs / out
        save_model_npz(out, save_model)
        return {
            "success": True,
            "model_path": str(out.with_suffix(".npz").resolve()),
            "model_meta": str(out.with_suffix(".json").resolve()),
            "space": space_l,
            "method": method_l,
            "reference_label": reference_label,
            "n_features": int(X.shape[1]),
            "dihedral_dir": str(ddir) if space_l == "dihedral" else None,
        }
    except Exception as exc:
        logger.exception("fit_dynamics_model failed")
        return {"success": False, "error": str(exc)}
    finally:
        if original_dir:
            os.chdir(original_dir)


@tool
def project_dynamics_model(
    model_path: str,
    label: str = "",
    topology_file: str = "",
    trajectory_file: str = "",
    alignment_json: str = "",
    dihedral_dir: str = "consensus_dihedrals",
    output_dir: str = "",
    working_dir: Optional[str] = None,
    sim_directory: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Project one simulation onto a fitted shared-reference dynamics model and write FEL.

    ``model_path`` is the stem or ``.npz`` from ``fit_dynamics_model``.
    """
    from src.analysis.family_dynamics_core import (
        dynamics_output_dirname,
        load_model_npz,
    )

    original_dir = None
    try:
        wd_abs = Path(working_dir).resolve() if working_dir else None
        sim_abs = (
            Path(sim_directory).resolve()
            if (sim_directory and Path(sim_directory).exists())
            else None
        )
        model_p = Path(model_path)
        if not model_p.is_absolute() and wd_abs is not None:
            model_p = wd_abs / model_p

        if wd_abs is not None:
            wd_abs.mkdir(parents=True, exist_ok=True)
            original_dir = os.getcwd()
            os.chdir(str(wd_abs))

        model = load_model_npz(model_p)
        space_l = str(model.get("space", "dihedral")).lower()
        method_l = str(model.get("method", "pca")).lower()
        angles = list(model.get("dihedral_angles") or ["phi", "psi", "chi1"])
        out_name = output_dir or dynamics_output_dirname(
            space_l, method_l, "shared_reference"
        )
        out = Path(out_name)
        if not out.is_absolute() and wd_abs is not None:
            out = wd_abs / out
        if (out / "fel_features.json").is_file() and not overwrite:
            return {
                "success": True,
                "skipped": True,
                **json.loads((out / "fel_features.json").read_text(encoding="utf-8")),
                "output_dir": str(out.resolve()),
            }

        if space_l == "dihedral":
            ddir = _resolve_dihedral_dir(
                dihedral_dir,
                sim_directory=str(sim_abs) if sim_abs else None,
                working_dir=str(wd_abs) if wd_abs else None,
            )
            if ddir is None:
                return {
                    "success": False,
                    "error": (
                        f"No consensus_dihedrals for label={label!r} "
                        f"sim={sim_abs} working_dir={wd_abs}"
                    ),
                }
            X = _load_dihedral_matrix(ddir, angles)
        else:
            if not label or not alignment_json:
                return {
                    "success": False,
                    "error": "cartesian projection needs label and alignment_json",
                }
            top, traj = _resolve_traj(
                topology_file, trajectory_file, str(sim_abs) if sim_abs else sim_directory
            )
            if not top or not traj:
                return {"success": False, "error": "Missing topology/trajectory"}
            align_p = Path(alignment_json)
            if not align_p.is_absolute() and wd_abs is not None:
                align_p = wd_abs / align_p
            X = _load_cartesian_matrix(
                top,
                traj,
                str(align_p),
                label,
                str(model.get("selection") or "protein and name CA"),
            )

        if X.shape[1] != int(model.get("n_features", X.shape[1])):
            # Allow if close — dihedral χ presence can differ; warn via error if far
            if abs(X.shape[1] - int(model["n_features"])) > 0:
                return {
                    "success": False,
                    "error": (
                        f"Feature dim mismatch: data {X.shape[1]} vs model "
                        f"{model.get('n_features')}. Recompute torsions with same angles/scope."
                    ),
                }

        proj = _project_reduce(X, model)
        meta = {
            "label": label,
            "space": space_l,
            "method": method_l,
            "projection_mode": "shared_reference",
            "reference_label": model.get("reference_label"),
            "model_path": str(model_p.resolve()),
            "dihedral_dir": str(ddir) if space_l == "dihedral" else None,
        }
        result = _write_dynamics_result(out, proj, method=method_l, meta_extra=meta)
        result["output_dir"] = str(out.resolve())
        return result
    except Exception as exc:
        logger.exception("project_dynamics_model failed")
        return {"success": False, "error": str(exc)}
    finally:
        if original_dir:
            os.chdir(original_dir)


@tool
def run_shared_dynamics_fel_batch(
    base_directory: str,
    space: str = "dihedral",
    method: str = "pca",
    reference_label: str = "",
    alignment_json: str = "",
    labels: Optional[List[str]] = None,
    dihedral_angles: Optional[List[str]] = None,
    lag_frames: int = 10,
    n_components: int = 10,
    working_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Fit a shared-reference dynamics model on ``reference_label``, then project all labels.

    Convenience batch for family campaigns. Does not force which features enter
    clustering — only writes FEL artifacts for later collection.
    """
    if not reference_label:
        return {"success": False, "error": "reference_label required"}
    base = Path(base_directory)
    out_root = Path(working_dir) if working_dir else base / "analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    skip = {"analysis", "preprocess", "simsetup", "hpc", "reporter", "programmer"}
    labs = labels or sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and d.name.lower() not in skip and not d.name.startswith(".")
    )
    if reference_label not in labs:
        labs = [reference_label] + labs

    model_stem = out_root / f"dynamics_model_{space}_{method}_{reference_label}"
    fit = fit_dynamics_model.invoke(
        {
            "space": space,
            "method": method,
            "reference_label": reference_label,
            "alignment_json": alignment_json,
            "dihedral_angles": dihedral_angles,
            "lag_frames": lag_frames,
            "n_components": n_components,
            "model_path": str(model_stem),
            "working_dir": str(out_root),
            "base_directory": str(base),
            "sim_directory": str(base / reference_label),
        }
    )
    if not fit.get("success"):
        return fit

    results = []
    for lab in labs:
        sim = base / lab
        adir = sim / "analysis"
        adir.mkdir(parents=True, exist_ok=True)
        res = project_dynamics_model.invoke(
            {
                "model_path": fit["model_path"],
                "label": lab,
                "alignment_json": alignment_json,
                "dihedral_dir": "",
                "working_dir": str(adir),
                "sim_directory": str(sim),
                "overwrite": overwrite,
            }
        )
        results.append({"label": lab, **(res if isinstance(res, dict) else {})})

    n_ok = sum(1 for r in results if r.get("success"))
    manifest = {
        "success": n_ok > 0,
        "fit": fit,
        "n_success": n_ok,
        "n_requested": len(labs),
        "results": results,
    }
    (out_root / "shared_dynamics_batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest


def _pc_shared_descriptors(proj: np.ndarray) -> Dict[str, float]:
    """Paper shared-PC scalars: FEL gmin, centroid, and PC RMS about centroid.

    Global-min location uses a Gaussian-smoothed FEL (``SMOOTH_SIGMA``) on the
    occupied grid, matching the ment sharedpcdyn recipe.
    """
    from src.analysis.family_dynamics_core import SMOOTH_SIGMA, fel_descriptors_from_proj2

    pc1 = proj[:, 0].astype(float)
    pc2 = proj[:, 1].astype(float)
    cent1 = float(np.mean(pc1))
    cent2 = float(np.mean(pc2))
    pc_rms = float(np.sqrt(np.mean((pc1 - cent1) ** 2 + (pc2 - cent2) ** 2)))
    desc = fel_descriptors_from_proj2(proj[:, :2])
    F = np.asarray(desc["F"], dtype=float)
    P = np.asarray(desc["P"], dtype=float)
    xc = np.asarray(desc["x_centers"], dtype=float)
    yc = np.asarray(desc["y_centers"], dtype=float)
    mask = np.isfinite(F) & (P > 0)
    if not np.any(mask):
        gmin1, gmin2 = cent1, cent2
    else:
        F_work = F.astype(float).copy()
        if SMOOTH_SIGMA and float(SMOOTH_SIGMA) > 0:
            try:
                from scipy.ndimage import gaussian_filter

                fill = float(np.max(F_work[mask]))
                F_work = gaussian_filter(
                    np.where(mask, F_work, fill), sigma=float(SMOOTH_SIGMA)
                )
            except Exception:
                logger.debug("shared-PC gmin smooth skipped", exc_info=True)
        F_m = np.where(mask, F_work, np.inf)
        jy, ix = np.unravel_index(int(np.argmin(F_m)), F.shape)
        gmin1, gmin2 = float(xc[ix]), float(yc[jy])
    return {
        "pca_pka_ref_shared_gmin_pc1": gmin1,
        "pca_pka_ref_shared_gmin_pc2": gmin2,
        "pca_pka_ref_shared_centroid_pc1": cent1,
        "pca_pka_ref_shared_centroid_pc2": cent2,
        "pca_pka_ref_shared_pc_rms": pc_rms,
    }


def _shared_dyn_from_desc(
    desc: Dict[str, float],
    *,
    ref_gmin: tuple,
    ref_cent: tuple,
) -> float:
    d_g = float(
        np.hypot(
            desc["pca_pka_ref_shared_gmin_pc1"] - ref_gmin[0],
            desc["pca_pka_ref_shared_gmin_pc2"] - ref_gmin[1],
        )
    )
    d_c = float(
        np.hypot(
            desc["pca_pka_ref_shared_centroid_pc1"] - ref_cent[0],
            desc["pca_pka_ref_shared_centroid_pc2"] - ref_cent[1],
        )
    )
    pc_rms = float(desc["pca_pka_ref_shared_pc_rms"])
    return float(np.sqrt(d_g * d_g + d_c * d_c + pc_rms * pc_rms))


def _resolve_sim_dir_label(base: Path, label: str, available: Sequence[str]) -> Optional[str]:
    """Map display or bare labels back to an on-disk simulation directory name."""
    if not label:
        return None
    if (base / label).is_dir():
        return label
    low = label.lower()
    avail = list(available)
    for cand in avail:
        if cand.lower() == low:
            return cand
    # Bare UniProt / prefix match (p17612 → p17612_ATP).
    stem = low.split("_")[0]
    matches = [c for c in avail if c.lower().startswith(stem)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        exact = [c for c in matches if c.lower() == f"{stem}_atp" or c.lower().startswith(f"{stem}_")]
        return exact[0] if exact else matches[0]
    # Scan filesystem when label list used display names (KAPCA_ATP → p17612_ATP).
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if d.name.lower() == low or d.name.lower().startswith(stem + "_") or d.name.lower() == stem:
            return d.name
    return None


@tool
def compute_shared_pka_ref_dyn_features(
    base_directory: str,
    reference_label: str = "",
    alignment_json: str = "",
    labels: Optional[List[str]] = None,
    dihedral_angles: Optional[List[str]] = None,
    n_components: int = 10,
    working_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Fit shared-reference φ/ψ/χ₁ PCA on ``reference_label`` **replicate 1 only**
    (paper sharedpcdyn), project every system (pooling ``rep*/`` frames, not
    ``avg/``), and write transferable dynamics scalar ``pca_pka_ref_shared_dyn``
    plus supporting PC descriptors.

    Alignment uses the reference residue ``consensus_index`` schema: every
    system is projected into the same feature space, with missing MSA columns
    filled by the reference feature mean (paper shared-PCA style). This avoids
    silent dropouts when per-sim dihedral widths differ.

    Modular: any reference label / system set; used when the user goal asks
    for shared-reference dihedral PCA dynamics (family comparative MD).
    Missing dihedrals return a soft failure so clustering can proceed on the
    available feature columns.

    Outputs per system:
      ``{sim}/analysis/avg/shared_pc_features.json``
      ``{sim}/analysis/shared_pc_features.json``
    Combined: ``{working_dir}/shared_pc_features_all.csv``
    """
    if not reference_label:
        return {"success": False, "error": "reference_label required (e.g. p17612_ATP)"}

    from src.analysis.family_dynamics_core import fit_pca, project_pca, save_model_npz

    base = _resolve_campaign_base(base_directory, working_dir)
    out_root = Path(working_dir).resolve() if working_dir else (base / "analysis")
    if out_root.name != "analysis" and (base / "analysis").is_dir():
        # Prefer campaign analysis/ for combined artefacts.
        if out_root == base:
            out_root = base / "analysis"
    out_root.mkdir(parents=True, exist_ok=True)
    angles = list(dihedral_angles or ["phi", "psi", "chi1"])

    skip = {
        "analysis",
        "preprocess",
        "simsetup",
        "hpc",
        "reporter",
        "programmer",
        "planner",
        "supervisor",
        "campaign",
        "cross_sim",
        "logs",
        "artifacts",
    }
    disk_labs = sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and d.name.lower() not in skip and not d.name.startswith(".")
        and (d / "analysis").is_dir()
    )
    requested = list(labels) if labels else list(disk_labs)
    labs: List[str] = []
    for lab in requested:
        resolved = _resolve_sim_dir_label(base, str(lab), disk_labs)
        if resolved and resolved not in labs:
            labs.append(resolved)
    if not labs:
        labs = list(disk_labs)

    ref_lab = _resolve_sim_dir_label(base, reference_label, labs or disk_labs)
    if not ref_lab:
        for cand in disk_labs:
            if _schema_dihedral_dirs(base / cand / "analysis"):
                ref_lab = cand
                break
        if not ref_lab and disk_labs:
            ref_lab = disk_labs[0]
    if not ref_lab:
        return {
            "success": False,
            "error": (
                f"reference_label {reference_label!r} not found under {base} "
                f"(have {disk_labs[:5]}…)"
            ),
        }
    # If callers passed display names, prefer full on-disk cohort.
    if labels and len(labs) < max(1, len(labels) // 2):
        labs = list(disk_labs)
    if ref_lab not in labs:
        labs = [ref_lab] + [l for l in labs if l != ref_lab]

    ref_fit_dir = _reference_fit_dihedral_dir(base / ref_lab / "analysis")
    if ref_fit_dir is None:
        return {
            "success": False,
            "error": (
                f"No consensus_dihedrals with consensus_index schema for reference "
                f"{ref_lab} under {base / ref_lab / 'analysis'}. "
                "Run calculate_consensus_torsions first."
            ),
        }

    try:
        ref_df = _load_dihedral_residue_table(ref_fit_dir)
        schema = _shared_dihedral_schema(ref_df, angles)
        if len(schema) < 8:
            return {
                "success": False,
                "error": f"Reference schema too small ({len(schema)} features) for {ref_lab}",
            }
        fill0 = np.zeros(len(schema), dtype=float)
        X_ref0 = _build_aligned_dihedral_matrix(ref_fit_dir, schema, fill0)
        fill = X_ref0.mean(axis=0)
        X_ref = _build_aligned_dihedral_matrix(ref_fit_dir, schema, fill)
    except Exception as exc:
        logger.exception("shared dyn: failed to build reference matrix")
        return {"success": False, "error": f"Reference dihedral matrix failed: {exc}"}

    pca = fit_pca(X_ref, n_components=n_components)
    model = {
        "mean": pca["mean"],
        "components": pca["components"],
        "n_features": int(X_ref.shape[1]),
        "n_components": int(pca["components"].shape[0]),
        "explained_variance_fraction": pca.get("explained_variance_fraction"),
        "space": "dihedral",
        "method": "pca",
        "reference_label": ref_lab,
        "reference_fit_dir": str(ref_fit_dir),
        "angles": angles,
        "fill": fill,
        "schema": schema,
    }
    model_stem = out_root / f"dynamics_model_dihedral_pca_{ref_lab}"
    try:
        save_model_npz(model_stem, model)
        (out_root / f"dynamics_model_dihedral_pca_{ref_lab}_schema.json").write_text(
            json.dumps(schema, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("shared dyn: could not save model: %s", exc)

    # Collect pooled projections per label (schema-aligned; replicates only).
    proj_by_lab: Dict[str, np.ndarray] = {}
    skipped: List[Dict[str, str]] = []
    for lab in labs:
        dirs = _schema_dihedral_dirs(base / lab / "analysis")
        if not dirs:
            skipped.append({"label": lab, "reason": "no_dihedrals"})
            continue
        chunks = []
        for ddir in dirs:
            try:
                X = _build_aligned_dihedral_matrix(ddir, schema, fill)
                chunks.append(project_pca(X, model["mean"], model["components"]))
            except Exception as exc:
                logger.warning("shared dyn project %s/%s: %s", lab, ddir, exc)
                skipped.append({"label": lab, "reason": str(exc)})
        if chunks:
            proj_by_lab[lab] = np.vstack(chunks)

    if ref_lab not in proj_by_lab:
        return {
            "success": False,
            "error": f"No projections for reference {ref_lab}; check consensus_dihedrals",
            "skipped": skipped,
        }

    ref_desc = _pc_shared_descriptors(proj_by_lab[ref_lab])
    ref_gmin = (
        ref_desc["pca_pka_ref_shared_gmin_pc1"],
        ref_desc["pca_pka_ref_shared_gmin_pc2"],
    )
    ref_cent = (
        ref_desc["pca_pka_ref_shared_centroid_pc1"],
        ref_desc["pca_pka_ref_shared_centroid_pc2"],
    )

    rows: List[Dict[str, Any]] = []
    n_ok = 0
    for lab, proj in proj_by_lab.items():
        desc = _pc_shared_descriptors(proj)
        desc["pca_pka_ref_shared_dyn"] = _shared_dyn_from_desc(
            desc, ref_gmin=ref_gmin, ref_cent=ref_cent
        )
        adir = base / lab / "analysis"
        avg = adir / "avg"
        avg.mkdir(parents=True, exist_ok=True)
        payload = {"label": lab, "reference_label": ref_lab, **desc}
        text = json.dumps(payload, indent=2) + "\n"
        if overwrite or not (avg / "shared_pc_features.json").is_file():
            (avg / "shared_pc_features.json").write_text(text, encoding="utf-8")
        (adir / "shared_pc_features.json").write_text(text, encoding="utf-8")
        # Also stash under consensus_PCA_ref for modular discovery.
        ref_out = avg / "consensus_PCA_ref"
        ref_out.mkdir(parents=True, exist_ok=True)
        np.save(ref_out / "pca_projections.npy", proj[:, :2])
        (ref_out / "fel_features.json").write_text(
            json.dumps(
                {
                    "grid_entropy": None,
                    "pca_pka_ref_shared_dyn": desc["pca_pka_ref_shared_dyn"],
                    **{k: desc[k] for k in desc if k.startswith("pca_pka_ref_shared_")},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(payload)
        n_ok += 1

    import csv

    csv_path = out_root / "shared_pc_features_all.csv"
    fields = [
        "label",
        "reference_label",
        "pca_pka_ref_shared_dyn",
        "pca_pka_ref_shared_pc_rms",
        "pca_pka_ref_shared_gmin_pc1",
        "pca_pka_ref_shared_gmin_pc2",
        "pca_pka_ref_shared_centroid_pc1",
        "pca_pka_ref_shared_centroid_pc2",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["label"]):
            w.writerow(r)

    return {
        "success": n_ok > 0,
        "n_success": n_ok,
        "n_requested": len(labs),
        "n_schema_features": len(schema),
        "reference_label": ref_lab,
        "csv": str(csv_path),
        "model_path": str(model_stem) + ".npz",
        "skipped": skipped,
        "base_directory": str(base),
    }
