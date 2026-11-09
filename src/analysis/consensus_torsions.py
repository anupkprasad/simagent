"""Consensus-mapped φ/ψ/χ₁ torsion time series for family MD."""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _chi1_atomgroup(residue):
    try:
        ag = residue.chi1_selection()
    except Exception:
        return None
    if ag is None or len(ag) != 4:
        return None
    return ag


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


def _pocket_resids_from_json(pocket_map_json: str, label: str) -> Optional[List[int]]:
    """Load per-sim pocket PDB resids from ``pocket_mapped.json`` when present."""
    if not pocket_map_json:
        return None
    path = Path(pocket_map_json)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    from src.analysis.cross_sim_artifacts import pocket_resids_for_label

    resids = pocket_resids_for_label(data, label)
    return resids or None


@tool
def calculate_consensus_torsions(
    topology_file: str = "",
    trajectory_file: str = "",
    alignment_json: str = "",
    label: str = "",
    angles: Optional[List[str]] = None,
    residue_scope: str = "mapped",
    pocket_map_csv: str = "",
    pocket_map_json: str = "",
    output_dir: str = "consensus_dihedrals",
    working_dir: Optional[str] = None,
    sim_directory: Optional[str] = None,
    hpc_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Compute consensus-mapped φ/ψ/χ₁ dihedral time series for one simulation.

    Requires a family consensus alignment JSON (from
    ``build_consensus_sequence_alignment`` or an equivalent ment-style map).

    **residue_scope:** ``mapped`` (all consensus positions present), ``pocket``
    (intersection with pocket_map_csv consensus indices), or ``all`` (alias of mapped).

    **angles:** subset of ``phi``, ``psi``, ``chi1`` (default all three).

    Pocket χ₁ summary prefers ``pocket_map_json`` per-sim PDB resids (same list as
    axis-angle/COM). Falls back to ``pocket_map_csv`` consensus indices.

    Writes under ``{working_dir}/{output_dir}/``:
    ``phi_deg.npy``, ``psi_deg.npy``, ``chi1_deg.npy``, ``dihedral_sincos.npy``,
    ``per_residue_dihedrals.csv``, ``dihedral_features_meta.json``,
    ``torsion_summary.json`` (circular means including pocket χ₁ when available).
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_dihedrals

    from src.analysis.family_dynamics_core import (
        build_dihedral_feature_matrix,
        circ_mean_std_deg,
        consensus_positions_list,
        load_consensus_alignment,
        load_pocket_consensus_indices,
        mapping_for_label,
    )

    angles = list(angles or ["phi", "psi", "chi1"])
    if not alignment_json:
        return {"success": False, "error": "alignment_json is required"}
    if not label:
        return {"success": False, "error": "label is required"}

    original_dir = None
    try:
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            original_dir = os.getcwd()
            os.chdir(working_dir)

        top, traj = _resolve_traj(
            topology_file, trajectory_file, sim_directory, hpc_dir=hpc_dir
        )
        if not top or not traj or not Path(top).is_file() or not Path(traj).is_file():
            return {"success": False, "error": f"Missing trajectory: top={top} traj={traj}"}

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta_path = out / "dihedral_features_meta.json"
        if meta_path.is_file() and not overwrite:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return {"success": True, "skipped": True, **meta}

        alignment = load_consensus_alignment(alignment_json)
        mapping = mapping_for_label(alignment, label)
        positions = consensus_positions_list(alignment)
        if not mapping:
            return {
                "success": False,
                "error": f"No consensus mapping for label={label!r} in {alignment_json}",
            }

        pocket_ci = load_pocket_consensus_indices(pocket_map_csv or None)
        pocket_resids = _pocket_resids_from_json(pocket_map_json or "", label)
        if pocket_resids is None and sim_directory:
            # Discover campaign pocket_mapped.json next to the sim when unset.
            for cand in (
                Path(sim_directory).parent / "analysis" / "pocket_mapped.json",
                Path(sim_directory).parent / "cross_sim" / "pocket_mapped.json",
            ):
                pocket_resids = _pocket_resids_from_json(str(cand), label)
                if pocket_resids:
                    break
        pocket_resid_set = set(pocket_resids) if pocket_resids else None
        scope = (residue_scope or "mapped").lower()
        if scope == "all":
            scope = "mapped"

        u = mda.Universe(str(Path(top).resolve()), str(Path(traj).resolve()))
        protein = {int(r.resid): r for r in u.select_atoms("protein").residues}

        entries: List[Dict[str, Any]] = []
        for ci, m in enumerate(mapping):
            if not m or m.get("resid") is None:
                continue
            if scope == "pocket" and pocket_ci is not None and ci not in pocket_ci:
                continue
            resid = int(m["resid"])
            res = protein.get(resid)
            if res is None:
                continue
            phi_ag = res.phi_selection()
            psi_ag = res.psi_selection()
            chi_ag = _chi1_atomgroup(res)
            if phi_ag is None or psi_ag is None or len(phi_ag) != 4 or len(psi_ag) != 4:
                continue
            msa_col = positions[ci].get("msa_col", ci) if ci < len(positions) else ci
            entries.append(
                {
                    "consensus_index": ci,
                    "msa_col": msa_col,
                    "resid": resid,
                    "aa": m.get("aa") or res.resname,
                    "phi_ag": phi_ag,
                    "psi_ag": psi_ag,
                    "chi_ag": chi_ag,
                    "has_chi1": chi_ag is not None,
                }
            )

        if len(entries) < 3:
            return {
                "success": False,
                "error": f"Too few dihedral residues ({len(entries)}) for label={label}",
            }

        n_frames = len(u.trajectory)
        phi_ts = np.full((n_frames, len(entries)), np.nan)
        psi_ts = np.full((n_frames, len(entries)), np.nan)
        chi_ts = np.full((n_frames, len(entries)), np.nan)

        for fi, _ts in enumerate(u.trajectory):
            for j, e in enumerate(entries):
                p = e["phi_ag"].positions
                q = e["psi_ag"].positions
                phi_ts[fi, j] = float(np.rad2deg(calc_dihedrals(p[0], p[1], p[2], p[3])))
                psi_ts[fi, j] = float(np.rad2deg(calc_dihedrals(q[0], q[1], q[2], q[3])))
                if e["has_chi1"]:
                    c = e["chi_ag"].positions
                    chi_ts[fi, j] = float(
                        np.rad2deg(calc_dihedrals(c[0], c[1], c[2], c[3]))
                    )

        has_chi = [bool(e["has_chi1"]) for e in entries]
        X = build_dihedral_feature_matrix(
            phi_ts, psi_ts, chi_ts, angles=angles, has_chi1_mask=has_chi
        )
        np.save(out / "phi_deg.npy", phi_ts)
        np.save(out / "psi_deg.npy", psi_ts)
        np.save(out / "chi1_deg.npy", chi_ts)
        np.save(out / "dihedral_sincos.npy", X)

        stats_path = out / "per_residue_dihedrals.csv"
        with open(stats_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "consensus_index",
                    "msa_col",
                    "resid",
                    "aa",
                    "has_chi1",
                    "phi_mean_deg",
                    "phi_std_deg",
                    "psi_mean_deg",
                    "psi_std_deg",
                    "chi1_mean_deg",
                    "chi1_std_deg",
                ],
            )
            w.writeheader()
            for j, e in enumerate(entries):
                phi_m, phi_s = circ_mean_std_deg(phi_ts[:, j])
                psi_m, psi_s = circ_mean_std_deg(psi_ts[:, j])
                if e["has_chi1"]:
                    chi_m, chi_s = circ_mean_std_deg(chi_ts[:, j])
                else:
                    chi_m = chi_s = float("nan")
                w.writerow(
                    {
                        "consensus_index": e["consensus_index"],
                        "msa_col": e["msa_col"],
                        "resid": e["resid"],
                        "aa": e["aa"],
                        "has_chi1": int(e["has_chi1"]),
                        "phi_mean_deg": f"{phi_m:.4f}" if np.isfinite(phi_m) else "",
                        "phi_std_deg": f"{phi_s:.4f}" if np.isfinite(phi_s) else "",
                        "psi_mean_deg": f"{psi_m:.4f}" if np.isfinite(psi_m) else "",
                        "psi_std_deg": f"{psi_s:.4f}" if np.isfinite(psi_s) else "",
                        "chi1_mean_deg": f"{chi_m:.4f}" if np.isfinite(chi_m) else "",
                        "chi1_std_deg": f"{chi_s:.4f}" if np.isfinite(chi_s) else "",
                    }
                )

        # Global / pocket χ₁ circular means.
        # Pocket summary uses circular mean/std of *per-residue* χ₁ means
        # (paper / ment add_chi1_mean_features convention), not pooled frames.
        chi_all = chi_ts[:, [j for j, h in enumerate(has_chi) if h]].ravel()
        chi1_global_mean, chi1_global_std = circ_mean_std_deg(chi_all)
        chi1_pocket_mean = chi1_pocket_std = float("nan")
        pocket_cols: List[int] = []
        pocket_filter = "none"
        if pocket_resid_set is not None:
            pocket_cols = [
                j
                for j, e in enumerate(entries)
                if e["has_chi1"] and int(e["resid"]) in pocket_resid_set
            ]
            pocket_filter = "pocket_map_json_resids"
        elif pocket_ci:
            pocket_cols = [
                j
                for j, e in enumerate(entries)
                if e["has_chi1"] and e["consensus_index"] in pocket_ci
            ]
            pocket_filter = "pocket_map_csv_consensus_index"
        if pocket_cols:
            per_res_means = []
            for j in pocket_cols:
                m, _s = circ_mean_std_deg(chi_ts[:, j])
                if np.isfinite(m):
                    per_res_means.append(m)
            if per_res_means:
                chi1_pocket_mean, chi1_pocket_std = circ_mean_std_deg(
                    np.asarray(per_res_means, dtype=float)
                )

        summary = {
            "label": label,
            "chi1_circ_mean_deg": chi1_global_mean,
            "chi1_circ_std_deg": chi1_global_std,
            "chi1_pocket_circ_mean_deg": chi1_pocket_mean,
            "chi1_pocket_circ_std_deg": chi1_pocket_std,
            "n_residues": len(entries),
            "n_with_chi1": int(sum(has_chi)),
            "n_pocket_chi1": len(pocket_cols),
            "pocket_filter": pocket_filter,
            "angles": angles,
            "residue_scope": scope,
        }
        (out / "torsion_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

        meta = {
            "success": True,
            "label": label,
            "n_frames": n_frames,
            "n_residues": len(entries),
            "n_with_chi1": int(sum(has_chi)),
            "n_features": int(X.shape[1]),
            "feature_matrix_path": str((out / "dihedral_sincos.npy").resolve()),
            "per_residue_csv": str(stats_path.resolve()),
            "torsion_summary": str((out / "torsion_summary.json").resolve()),
            "angles": angles,
            "residue_scope": scope,
            "mode": "consensus_phi_psi_chi1",
            "topology_file": str(Path(top).resolve()),
            "trajectory_file": str(Path(traj).resolve()),
            **{k: summary[k] for k in summary if k.startswith("chi1_")},
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        try:
            from src.analysis.feature_matrix_plots import (
                save_dihedral_summary_png,
                save_feature_matrix_png,
            )

            plots = []
            for npy_name, kind in (
                ("phi_deg.npy", "angle_deg"),
                ("psi_deg.npy", "angle_deg"),
                ("chi1_deg.npy", "angle_deg"),
                ("dihedral_sincos.npy", "sincos"),
            ):
                src = out / npy_name
                if src.is_file():
                    png = save_feature_matrix_png(
                        np.load(src),
                        out / f"{src.stem}.png",
                        title=src.stem,
                        kind=kind,
                    )
                    if png:
                        plots.append(png)
            if stats_path.is_file():
                p = save_dihedral_summary_png(
                    stats_path,
                    out / "per_residue_dihedrals.png",
                )
                if p:
                    plots.append(p)
            if plots:
                meta["plots"] = plots
                meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        except Exception:
            logger.debug("consensus torsions plots skipped", exc_info=True)
        return meta
    except Exception as exc:
        logger.exception("calculate_consensus_torsions failed")
        return {"success": False, "error": str(exc)}
    finally:
        if original_dir:
            os.chdir(original_dir)


@tool
def run_consensus_torsions_batch(
    base_directory: str,
    alignment_json: str,
    labels: Optional[List[str]] = None,
    angles: Optional[List[str]] = None,
    residue_scope: str = "mapped",
    pocket_map_csv: str = "",
    working_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Run ``calculate_consensus_torsions`` for each simulation under ``base_directory``.

    Expects ``{base}/{label}/`` with trajectories discoverable by SimAgent
    conventions. Writes per-sim under ``{label}/analysis/consensus_dihedrals/``.
    """
    base = Path(base_directory)
    if not base.is_dir():
        return {"success": False, "error": f"Base not found: {base}"}

    skip = {"analysis", "preprocess", "simsetup", "hpc", "reporter", "programmer"}
    if labels:
        labs = list(labels)
    else:
        labs = sorted(
            d.name
            for d in base.iterdir()
            if d.is_dir() and d.name.lower() not in skip and not d.name.startswith(".")
        )

    results = []
    for lab in labs:
        sim = base / lab
        adir = sim / "analysis"
        adir.mkdir(parents=True, exist_ok=True)
        res = calculate_consensus_torsions.invoke(
            {
                "topology_file": "",
                "trajectory_file": "",
                "alignment_json": alignment_json,
                "label": lab,
                "angles": angles,
                "residue_scope": residue_scope,
                "pocket_map_csv": pocket_map_csv,
                "output_dir": "consensus_dihedrals",
                "working_dir": str(adir),
                "sim_directory": str(sim),
                "overwrite": overwrite,
            }
        )
        results.append({"label": lab, **(res if isinstance(res, dict) else {"raw": res})})

    n_ok = sum(1 for r in results if r.get("success"))
    manifest = {
        "success": n_ok > 0,
        "n_requested": len(labs),
        "n_success": n_ok,
        "results": results,
    }
    out_root = Path(working_dir) if working_dir else base / "analysis"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "consensus_torsions_batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest
