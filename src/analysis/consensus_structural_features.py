"""Consensus-mapped RMSF and DCCM scalar features for family classification."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


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


def _mapped_ca_atomgroup(u, mapping):
    protein = {int(r.resid): r for r in u.select_atoms("protein").residues}
    resids = []
    meta = []
    for ci, m in enumerate(mapping):
        if not m or m.get("resid") is None:
            continue
        resid = int(m["resid"])
        if resid not in protein:
            continue
        resids.append(resid)
        meta.append({"consensus_index": ci, "resid": resid, "aa": m.get("aa")})
    if not resids:
        return None, []
    sel = " or ".join(f"resid {r}" for r in resids)
    ag = u.select_atoms(f"protein and name CA and ({sel})")
    return ag, meta


@tool
def calculate_consensus_rmsf_features(
    topology_file: str = "",
    trajectory_file: str = "",
    alignment_json: str = "",
    label: str = "",
    output_dir: str = "consensus_rmsf",
    working_dir: Optional[str] = None,
    sim_directory: Optional[str] = None,
    hpc_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Consensus-mapped Cα RMSF mean/std for one simulation.

    Aligns mapped consensus Cα atoms to frame 0 before RMSF (same fit used for
    consensus DCCM). Without that fit, rigid-body motion inflates RMSF by an
    order of magnitude and breaks family-scale clustering.

    Writes ``{output_dir}/consensus_rmsf_features.json`` with
    ``consensus_rmsf_mean_A`` and ``consensus_rmsf_std_A``, plus per-residue profile.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis import align as mda_align
    from MDAnalysis.analysis import rms

    from src.analysis.family_dynamics_core import (
        load_consensus_alignment,
        mapping_for_label,
    )

    if not alignment_json or not label:
        return {"success": False, "error": "alignment_json and label required"}

    # Resolve inputs before any chdir so relative campaign paths stay valid.
    alignment_path = str(Path(alignment_json).expanduser().resolve())
    sim_abs = str(Path(sim_directory).expanduser().resolve()) if sim_directory else None
    hpc_abs = str(Path(hpc_dir).expanduser().resolve()) if hpc_dir else None
    top_in = str(Path(topology_file).expanduser().resolve()) if topology_file else ""
    traj_in = str(Path(trajectory_file).expanduser().resolve()) if trajectory_file else ""
    top, traj = _resolve_traj(top_in, traj_in, sim_abs, hpc_dir=hpc_abs)
    if not top or not traj:
        return {"success": False, "error": "Missing trajectory"}

    original_dir = None
    try:
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            original_dir = os.getcwd()
            os.chdir(working_dir)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        feat_path = out / "consensus_rmsf_features.json"
        if feat_path.is_file() and not overwrite:
            return {"success": True, "skipped": True, **json.loads(feat_path.read_text())}

        alignment = load_consensus_alignment(alignment_path)
        mapping = mapping_for_label(alignment, label)
        u = mda.Universe(str(Path(top).resolve()), str(Path(traj).resolve()))
        ag, meta = _mapped_ca_atomgroup(u, mapping)
        if ag is None or ag.n_atoms < 3:
            return {"success": False, "error": "Too few mapped Cα atoms"}

        # Fit mapped consensus Cα to frame 0 (matches consensus DCCM / family_dynamics).
        resids = [m["resid"] for m in meta]
        sel = " or ".join(f"resid {r}" for r in resids)
        align_sel = f"protein and name CA and ({sel})"
        ref = u.copy()
        mda_align.AlignTraj(u, ref, select=align_sel, in_memory=True).run()
        # Re-select after in-memory align so the AtomGroup tracks fitted coords.
        ag, meta = _mapped_ca_atomgroup(u, mapping)
        if ag is None or ag.n_atoms < 3:
            return {"success": False, "error": "Too few mapped Cα atoms after align"}

        R = rms.RMSF(ag).run()
        vals = np.asarray(R.results.rmsf, dtype=float)
        mean_a = float(np.mean(vals))
        std_a = float(np.std(vals))
        profile = out / "per_residue_rmsf.csv"
        with open(profile, "w", encoding="utf-8") as fh:
            fh.write("consensus_index,resid,aa,rmsf_A\n")
            for i, m in enumerate(meta):
                if i < len(vals):
                    fh.write(
                        f"{m['consensus_index']},{m['resid']},{m.get('aa','')},{vals[i]:.6f}\n"
                    )
        feat = {
            "success": True,
            "label": label,
            "consensus_rmsf_mean_A": mean_a,
            "consensus_rmsf_std_A": std_a,
            "n_atoms": int(ag.n_atoms),
            "aligned": True,
            "align_selection": align_sel,
            "profile_csv": str(profile.resolve()),
            "topology_file": str(Path(top).resolve()),
            "trajectory_file": str(Path(traj).resolve()),
        }
        feat_path.write_text(json.dumps(feat, indent=2) + "\n", encoding="utf-8")
        try:
            from src.analysis.feature_matrix_plots import save_rmsf_profile_png

            png = save_rmsf_profile_png(
                profile,
                out / "per_residue_rmsf.png",
                title=f"Consensus Cα RMSF ({label})",
            )
            if png:
                feat["plot"] = png
                feat_path.write_text(json.dumps(feat, indent=2) + "\n", encoding="utf-8")
        except Exception:
            logger.debug("consensus RMSF plot skipped", exc_info=True)
        return feat
    except Exception as exc:
        logger.exception("calculate_consensus_rmsf_features failed")
        return {"success": False, "error": str(exc)}
    finally:
        if original_dir:
            os.chdir(original_dir)


@tool
def calculate_consensus_dccm_features(
    topology_file: str = "",
    trajectory_file: str = "",
    alignment_json: str = "",
    label: str = "",
    n_lobe_ci_max: int = 34,
    c_lobe_ci_min: int = 35,
    lobe_method: str = "auto",
    output_dir: str = "consensus_DCCM",
    working_dir: Optional[str] = None,
    sim_directory: Optional[str] = None,
    hpc_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Consensus-mapped Cα DCCM scalars: mean |corr| and N↔C lobe mean correlation.

    Lobe split (``lobe_method='auto'``): consensus-index cut when the map is
    long enough (paper: ``ci <= 34`` / ``ci >= 35``); otherwise KAPCA/reference
    residue hinge; otherwise midpoint. Override cuts or set ``lobe_method``.

    Writes ``dccm_N_C_mean_corr``, ``mean_abs_dccm`` into
    ``{output_dir}/consensus_dccm_features.json``.
    """
    from src.analysis.consensus_local_fel import _trajectory_coord_matrix_self_aligned
    from src.analysis.dccm_calculator import _compute_dccm_matrix
    from src.analysis.family_dynamics_core import (
        load_consensus_alignment,
        mapping_for_label,
    )
    from src.analysis.lobe_split import assign_n_c_lobe_indices

    if not alignment_json or not label:
        return {"success": False, "error": "alignment_json and label required"}

    alignment_path = str(Path(alignment_json).expanduser().resolve())
    sim_abs = str(Path(sim_directory).expanduser().resolve()) if sim_directory else None
    hpc_abs = str(Path(hpc_dir).expanduser().resolve()) if hpc_dir else None
    top_in = str(Path(topology_file).expanduser().resolve()) if topology_file else ""
    traj_in = str(Path(trajectory_file).expanduser().resolve()) if trajectory_file else ""
    top, traj = _resolve_traj(top_in, traj_in, sim_abs, hpc_dir=hpc_abs)
    if not top or not traj:
        return {"success": False, "error": "Missing trajectory"}

    original_dir = None
    try:
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            original_dir = os.getcwd()
            os.chdir(working_dir)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        feat_path = out / "consensus_dccm_features.json"
        if feat_path.is_file() and not overwrite:
            return {"success": True, "skipped": True, **json.loads(feat_path.read_text())}

        alignment = load_consensus_alignment(alignment_path)
        mapping = mapping_for_label(alignment, label)
        # Paper path: per-frame Kabsch of consensus Cα onto frame 0, atoms kept
        # in consensus-index order (AlignTraj + stale AtomGroup diluted |corr|).
        X_flat, _times, keep, n_atoms = _trajectory_coord_matrix_self_aligned(
            str(Path(top).resolve()),
            str(Path(traj).resolve()),
            mapping,
            frame_interval=1,
        )
        if X_flat.shape[0] < 10 or n_atoms < 5:
            return {
                "success": False,
                "error": f"Too few frames/atoms for DCCM (T={X_flat.shape[0]}, n={n_atoms})",
            }
        positions = X_flat.reshape(X_flat.shape[0], n_atoms, 3)
        C = _compute_dccm_matrix(positions)
        N = int(C.shape[0])
        np.fill_diagonal(C, 1.0)
        np.save(out / "consensus_dccm.npy", C)

        iu = np.triu_indices(N, k=1)
        mean_abs = float(np.mean(np.abs(C[iu])))
        keep_arr = np.asarray(keep, dtype=bool)
        meta: List[Dict[str, Any]] = []
        for ci, m in enumerate(mapping):
            if ci >= len(keep_arr) or not keep_arr[ci]:
                continue
            meta.append(
                {
                    "consensus_index": ci,
                    "resid": int(m["resid"]) if m and m.get("resid") is not None else None,
                    "aa": (m or {}).get("aa") if m else None,
                }
            )
        if len(meta) != N:
            return {
                "success": False,
                "error": f"DCCM meta mismatch {len(meta)} vs {N} atoms",
            }

        n_idx, c_idx, lobe_info = assign_n_c_lobe_indices(
            meta,
            alignment=alignment,
            n_lobe_ci_max=int(n_lobe_ci_max),
            c_lobe_ci_min=int(c_lobe_ci_min),
            lobe_method=lobe_method,
        )
        if n_idx and c_idx:
            block = C[np.ix_(n_idx, c_idx)]
            nc_mean = float(np.mean(block))
        else:
            nc_mean = float("nan")

        feat = {
            "success": True,
            "label": label,
            "mean_abs_dccm": mean_abs,
            "dccm_N_C_mean_corr": nc_mean,
            "n_consensus_atoms": N,
            "N_lobe_n_atoms": len(n_idx),
            "C_lobe_n_atoms": len(c_idx),
            "topology_file": str(Path(top).resolve()),
            "trajectory_file": str(Path(traj).resolve()),
            "align_method": "kabsch_frame0_consensus_ca",
            **lobe_info,
        }
        feat_path.write_text(json.dumps(feat, indent=2) + "\n", encoding="utf-8")
        try:
            from src.analysis.feature_matrix_plots import save_feature_matrix_png

            png = save_feature_matrix_png(
                C,
                out / "consensus_dccm.png",
                title=f"Consensus DCCM ({label})",
                kind="correlation",
            )
            if png:
                feat["plot"] = png
                feat_path.write_text(json.dumps(feat, indent=2) + "\n", encoding="utf-8")
        except Exception:
            logger.debug("consensus DCCM plot skipped", exc_info=True)
        return feat
    except Exception as exc:
        logger.exception("calculate_consensus_dccm_features failed")
        return {"success": False, "error": str(exc)}
    finally:
        if original_dir:
            os.chdir(original_dir)
