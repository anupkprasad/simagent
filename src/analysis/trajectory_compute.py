"""
Dispatch trajectory metric computation through a shared session.

Calculator modules expose ``compute_*_from_universe`` helpers; this module
routes batch execution to the right helper without reloading the trajectory.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .chain_residue_map import ChainSelectionError
from .metric_registry import TrajectoryPassKind
from .trajectory_session import TrajectorySession

logger = logging.getLogger(__name__)


def _params_for_session(session: TrajectorySession, params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate PDB chain/resid selections onto trajectory resindex."""
    try:
        return session.translate_params(params)
    except ChainSelectionError:
        raise
    except Exception as exc:
        logger.warning("Chain-map selection translation failed: %s", exc)
        return dict(params)


def _resolve_align_selection(params: Dict[str, Any], default: str = "protein and name CA") -> str:
    return (
        params.get("align_selection")
        or params.get("selection")
        or default
    )


def compute_metric_from_session(
    tool_name: str,
    session: TrajectorySession,
    pass_kind: TrajectoryPassKind,
    *,
    topology_file: str,
    trajectory_file: str,
    working_dir: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one metric using the appropriate Universe from *session*."""
    try:
        from .proximity_analyzer import apply_selection_from_files

        params = apply_selection_from_files(dict(params), working_dir=working_dir)
    except Exception as exc:
        logger.warning("selection_from_file resolution failed: %s", exc)
        params = dict(params)

    try:
        params = _params_for_session(session, params)
    except ChainSelectionError as exc:
        return {"success": False, "error": str(exc)}
    align_sel = _resolve_align_selection(params)
    skip_align = pass_kind == TrajectoryPassKind.ALIGNED

    if pass_kind == TrajectoryPassKind.ALIGNED:
        universe = session.aligned_universe(align_selection=align_sel)
    else:
        universe = session.raw_universe

    common = {
        "topology_file": topology_file,
        "trajectory_file": trajectory_file,
        "working_dir": working_dir,
    }

    if tool_name == "calculate_rmsd":
        from .rmsd_calculator import compute_rmsd_from_universe

        return compute_rmsd_from_universe(
            universe,
            selection=params.get("selection", "protein and name CA"),
            reference_frame=int(params.get("reference_frame", 0) or 0),
            output_file=params.get("output_file"),
            align_before_rmsd=False if skip_align else bool(params.get("align_before_rmsd", True)),
            pre_aligned=skip_align,
            **common,
        )

    if tool_name == "calculate_rmsf":
        from .rmsf_calculator import compute_rmsf_from_universe
        from .selection_policy import DEFAULT_CA_SELECTION, ensure_ca_selection

        return compute_rmsf_from_universe(
            universe,
            selection=ensure_ca_selection(params.get("selection", DEFAULT_CA_SELECTION)),
            output_file=params.get("output_file"),
            align_trajectory=not skip_align and bool(params.get("align_trajectory", True)),
            align_selection=ensure_ca_selection(
                params.get("align_selection") or params.get("selection") or DEFAULT_CA_SELECTION
            ),
            skip_align=skip_align,
            **common,
        )

    if tool_name == "calculate_radius_of_gyration":
        from .gyration_calculator import compute_rg_from_universe

        return compute_rg_from_universe(
            universe,
            selection=params.get("selection", "protein"),
            output_file=params.get("output_file"),
            align_before_rg=skip_align,
            **common,
        )

    if tool_name == "calculate_dccm":
        from .dccm_calculator import compute_dccm_from_universe
        from .selection_policy import DEFAULT_CA_SELECTION, ensure_ca_selection

        return compute_dccm_from_universe(
            universe,
            selection=ensure_ca_selection(params.get("selection", DEFAULT_CA_SELECTION)),
            output_prefix=params.get("output_prefix"),
            frame_interval=int(params.get("frame_interval", 1) or 1),
            save_matrix_csv=bool(params.get("save_matrix_csv", True)),
            create_heatmap=bool(params.get("create_heatmap", True)),
            vmin=float(params.get("vmin", -1.0)),
            vmax=float(params.get("vmax", 1.0)),
            **common,
        )

    if tool_name == "calculate_com_distance":
        from .com_distance_calculator import compute_com_distance_from_universe

        return compute_com_distance_from_universe(
            universe,
            selection1=params.get("selection1", "protein"),
            selection2=params.get("selection2", "resname LIG"),
            label1=params.get("label1", "group1"),
            label2=params.get("label2", "group2"),
            output_file=params.get("output_file"),
            frame_interval=int(params.get("frame_interval", 1) or 1),
            **common,
        )

    if tool_name == "identify_nearby_residues":
        from .proximity_analyzer import compute_nearby_residues_from_universe

        return compute_nearby_residues_from_universe(
            universe,
            query_selection=params.get("query_selection", "protein"),
            neighbor_selection=params.get("neighbor_selection", "protein"),
            cutoff=float(params.get("cutoff", params.get("proximity_cutoff", 15.0))),
            frame=int(params.get("frame", 0) or 0),
            output_file=params.get("output_file"),
            **common,
        )

    if tool_name == "calculate_min_heavy_atom_distance":
        from .proximity_analyzer import compute_min_heavy_atom_distance_from_universe

        return compute_min_heavy_atom_distance_from_universe(
            universe,
            selection1=params.get("selection1", "protein"),
            selection2=params.get("selection2", "protein"),
            label1=params.get("label1", "group1"),
            label2=params.get("label2", "group2"),
            output_file=params.get("output_file"),
            frame_interval=int(params.get("frame_interval", 1) or 1),
            **common,
        )

    if tool_name == "calculate_hbond_occupancy":
        from .interface_analyzer import compute_hbond_occupancy_from_universe

        return compute_hbond_occupancy_from_universe(
            universe,
            selection1=params.get("selection1", "protein"),
            selection2=params.get("selection2", "protein"),
            label1=params.get("label1", "group1"),
            label2=params.get("label2", "group2"),
            d_a_cutoff=float(params.get("d_a_cutoff", params.get("hbond_distance", 3.5))),
            angle_cutoff=float(params.get("angle_cutoff", params.get("hbond_angle", 150.0))),
            output_file=params.get("output_file"),
            frame_interval=int(params.get("frame_interval", 1) or 1),
            **common,
        )

    if tool_name == "calculate_salt_bridge_distances":
        from .interface_analyzer import compute_salt_bridge_distances_from_universe

        return compute_salt_bridge_distances_from_universe(
            universe,
            selection1=params.get("selection1", "protein"),
            selection2=params.get("selection2", "protein"),
            label1=params.get("label1", "group1"),
            label2=params.get("label2", "group2"),
            cutoff=float(params.get("cutoff", 4.0)),
            output_file=params.get("output_file"),
            frame_interval=int(params.get("frame_interval", 1) or 1),
            **common,
        )

    if tool_name == "calculate_ligand_pocket_distance":
        from .com_distance_calculator import compute_ligand_pocket_distance_from_universe

        lig_sel = (params.get("ligand_selection") or "").strip()
        if not lig_sel:
            return {"success": False, "error": "ligand_selection is required (no silent ATP default)"}
        return compute_ligand_pocket_distance_from_universe(
            universe,
            ligand_selection=lig_sel,
            protein_selection=params.get("protein_selection", "protein"),
            cutoff=float(params.get("cutoff", 5.0)),
            output_file=params.get("output_file"),
            frame_interval=int(params.get("frame_interval", 1) or 1),
            **common,
        )

    if tool_name == "calculate_pocket_rmsf":
        from .binding_site_analyzer import compute_pocket_rmsf_from_universe

        lig_sel = (params.get("ligand_selection") or "").strip()
        if not lig_sel:
            return {"success": False, "error": "ligand_selection is required (no silent ATP default)"}
        return compute_pocket_rmsf_from_universe(
            universe,
            ligand_selection=lig_sel,
            protein_selection=params.get("protein_selection", "protein"),
            pocket_cutoff=float(params.get("pocket_cutoff", 5.0)),
            output_file=params.get("output_file"),
            align_trajectory=not skip_align and bool(params.get("align_trajectory", True)),
            skip_align=skip_align,
            **common,
        )

    if tool_name == "calculate_ligand_rmsf":
        from .binding_site_analyzer import compute_ligand_rmsf_from_universe

        lig_sel = (params.get("ligand_selection") or "").strip()
        if not lig_sel:
            return {"success": False, "error": "ligand_selection is required (no silent ATP default)"}
        return compute_ligand_rmsf_from_universe(
            universe,
            ligand_selection=lig_sel,
            protein_selection=params.get("protein_selection", "protein"),
            output_file=params.get("output_file"),
            output_json=params.get("output_json"),
            align_trajectory=not skip_align and bool(params.get("align_trajectory", True)),
            align_selection=params.get("align_selection"),
            skip_align=skip_align,
            **common,
        )

    if tool_name == "calculate_protein_ligand_contacts":
        from .binding_site_analyzer import compute_protein_ligand_contacts_from_universe

        lig_sel = (params.get("ligand_selection") or "").strip()
        if not lig_sel:
            return {"success": False, "error": "ligand_selection is required (no silent ATP default)"}
        return compute_protein_ligand_contacts_from_universe(
            universe,
            ligand_selection=lig_sel,
            protein_selection=params.get("protein_selection", "protein"),
            contact_cutoff=float(params.get("contact_cutoff", 4.0)),
            hbond_distance=float(params.get("hbond_distance", 3.0)),
            hbond_angle=float(params.get("hbond_angle", 150.0)),
            output_file=params.get("output_file"),
            frame_interval=int(params.get("frame_interval", 1) or 1),
            **common,
        )

    if tool_name == "analyze_ligand_residence":
        from .binding_site_analyzer import compute_ligand_residence_from_universe

        lig_sel = (params.get("ligand_selection") or "").strip()
        if not lig_sel:
            return {"success": False, "error": "ligand_selection is required (no silent ATP default)"}
        return compute_ligand_residence_from_universe(
            universe,
            ligand_selection=lig_sel,
            protein_selection=params.get("protein_selection", "protein"),
            pocket_cutoff=float(params.get("pocket_cutoff", 5.0)),
            bound_distance_A=float(params.get("bound_distance_A", 5.0)),
            min_contacts=int(params.get("min_contacts", 1) or 1),
            output_file=params.get("output_file"),
            frame_interval=int(params.get("frame_interval", 1) or 1),
            **common,
        )

    if tool_name == "analyze_secondary_structure":
        from .dssp_analyzer import compute_dssp_from_universe
        from pathlib import Path as _Path

        out_base = _Path(working_dir) if working_dir else _Path.cwd()
        return compute_dssp_from_universe(
            universe,
            selection=params.get("selection", "protein"),
            output_prefix=params.get("output_prefix"),
            create_heatmap=bool(params.get("create_heatmap", True)),
            create_time_series=bool(params.get("create_time_series", True)),
            save_raw_data=bool(params.get("save_raw_data", True)),
            out_base=out_base,
            **common,
        )

    return {
        "success": False,
        "error": f"No session compute handler for tool: {tool_name}",
    }


FUSED_RAW_STREAMING_TOOLS = frozenset({
    "calculate_radius_of_gyration",
    "calculate_com_distance",
    "calculate_ligand_pocket_distance",
    "analyze_ligand_residence",
    "calculate_min_heavy_atom_distance",
})


def _fused_collect_frame(col, ts, box, mda_distances, np, minimum_image_distance):
    """Record one frame for a fused RAW collector."""
    if col["type"] == "rg":
        col["values"].append(float(col["atoms"].radius_of_gyration()))
        col["frames"].append(int(ts.frame))
        col["times"].append(float(ts.time))
    elif col["type"] == "com":
        d = minimum_image_distance(
            col["group1"].center_of_mass(),
            col["group2"].center_of_mass(),
            box,
        )
        col["values"].append(d)
        col["frames"].append(int(ts.frame))
        col["times"].append(float(ts.time))
    elif col["type"] == "pocket":
        d = minimum_image_distance(
            col["pocket"].center_of_mass(),
            col["ligand"].center_of_mass(),
            box,
        )
        col["values"].append(d)
        col["frames"].append(int(ts.frame))
        col["times"].append(float(ts.time))
    elif col["type"] == "residence":
        pocket_com = col["pocket"].center_of_mass()
        lig_com = col["ligand"].center_of_mass()
        com_dist = minimum_image_distance(pocket_com, lig_com, box)
        dist_arr = mda_distances.distance_array(
            col["protein_heavy"].positions,
            col["ligand_heavy"].positions,
            box=box,
        )
        n_contacts = int(np.sum(dist_arr <= col["bound_distance_A"]))
        min_d = float(np.min(dist_arr)) if dist_arr.size else com_dist
        is_bound = (
            n_contacts >= col["min_contacts"]
            or com_dist <= col["bound_distance_A"]
        )
        col["times"].append(float(ts.time) / 1000.0)
        col["bound_mask"].append(is_bound)
        col["min_dists"].append(min_d)
    elif col["type"] == "min_dist":
        dist_arr = mda_distances.distance_array(
            col["group1"].positions,
            col["group2"].positions,
            box=box,
        )
        if dist_arr.size:
            col["values"].append(float(np.min(dist_arr)))
            col["frames"].append(int(ts.frame))
            col["times"].append(float(ts.time))


def _build_fused_collector(u, tool_name: str, params: Dict[str, Any], step_idx: int):
    """Create a per-frame collector dict for fused RAW streaming."""
    interval = max(1, int(params.get("frame_interval", 1) or 1))
    base = {"tool": tool_name, "step_idx": step_idx, "params": params, "interval": interval}

    if tool_name == "calculate_radius_of_gyration":
        sel = params.get("selection", "protein")
        atoms = u.select_atoms(sel)
        return {
            **base,
            "type": "rg",
            "atoms": atoms,
            "frames": [],
            "times": [],
            "values": [],
        }

    if tool_name == "calculate_com_distance":
        return {
            **base,
            "type": "com",
            "group1": u.select_atoms(params.get("selection1", "protein")),
            "group2": u.select_atoms(params.get("selection2", "resname LIG")),
            "frames": [],
            "times": [],
            "values": [],
        }

    if tool_name == "calculate_ligand_pocket_distance":
        from .binding_site_analyzer import identify_pocket_atoms

        u.trajectory[0]
        ligand_selection = params.get("ligand_selection", "resname LIG")
        protein_selection = params.get("protein_selection", "protein")
        cutoff = float(params.get("cutoff", params.get("pocket_cutoff", 5.0)))
        pocket_frozen, resids, meta = identify_pocket_atoms(
            u, ligand_selection, protein_selection, cutoff
        )
        ligand = u.select_atoms(ligand_selection)
        return {
            **base,
            "type": "pocket",
            "pocket": pocket_frozen,
            "ligand": ligand,
            "pocket_resids": resids,
            "n_pocket_atoms": meta.get("pocket_atom_count", len(pocket_frozen)),
            "frames": [],
            "times": [],
            "values": [],
        }

    if tool_name == "analyze_ligand_residence":
        from .binding_site_analyzer import identify_pocket_atoms

        u.trajectory[0]
        ligand_selection = params.get("ligand_selection", "resname LIG")
        protein_selection = params.get("protein_selection", "protein")
        pocket_cutoff = float(params.get("pocket_cutoff", 5.0))
        pocket_frozen, _, meta = identify_pocket_atoms(
            u, ligand_selection, protein_selection, pocket_cutoff
        )
        ligand = u.select_atoms(ligand_selection)
        return {
            **base,
            "type": "residence",
            "pocket": pocket_frozen,
            "ligand": ligand,
            "protein_heavy": u.select_atoms(f"({protein_selection}) and not name H*"),
            "ligand_heavy": ligand.select_atoms("not name H*"),
            "bound_distance_A": float(params.get("bound_distance_A", 5.0)),
            "min_contacts": int(params.get("min_contacts", 1) or 1),
            "pocket_meta": meta,
            "times": [],
            "bound_mask": [],
            "min_dists": [],
        }

    if tool_name == "calculate_min_heavy_atom_distance":
        sel1 = params.get("selection1", "protein")
        sel2 = params.get("selection2", "protein")
        return {
            **base,
            "type": "min_dist",
            "group1": u.select_atoms(f"({sel1}) and not name H*"),
            "group2": u.select_atoms(f"({sel2}) and not name H*"),
            "frames": [],
            "times": [],
            "values": [],
        }

    return None


def run_fused_raw_streaming_batch(
    session: TrajectorySession,
    steps: list,
    *,
    topology_file: str,
    trajectory_file: str,
    working_dir: Optional[str],
) -> Dict[int, Dict[str, Any]]:
    """
    Single ``for ts in u.trajectory`` loop for compatible RAW streaming metrics.

    *steps* is a list of ``(step_index, tool_name, params)`` tuples.
    """
    from .metric_registry import resolve_effective_pass_kind

    if not steps:
        return {}

    results: Dict[int, Dict[str, Any]] = {}
    fused_steps = [(i, t, p) for i, t, p in steps if t in FUSED_RAW_STREAMING_TOOLS]
    other_steps = [(i, t, p) for i, t, p in steps if t not in FUSED_RAW_STREAMING_TOOLS]

    for idx, tool_name, params in other_steps:
        results[idx] = compute_metric_from_session(
            tool_name,
            session,
            resolve_effective_pass_kind(tool_name, params) or TrajectoryPassKind.RAW,
            topology_file=topology_file,
            trajectory_file=trajectory_file,
            working_dir=working_dir,
            params=params,
        )

    if not fused_steps:
        return results

    if len(fused_steps) == 1:
        idx, tool_name, params = fused_steps[0]
        results[idx] = compute_metric_from_session(
            tool_name,
            session,
            TrajectoryPassKind.RAW,
            topology_file=topology_file,
            trajectory_file=trajectory_file,
            working_dir=working_dir,
            params=params,
        )
        return results

    try:
        import numpy as np
        from MDAnalysis.analysis import distances as mda_distances
        from src.analysis.pbc_utils import minimum_image_distance
    except ImportError:
        for idx, tool_name, params in fused_steps:
            results[idx] = compute_metric_from_session(
                tool_name, session, TrajectoryPassKind.RAW,
                topology_file=topology_file, trajectory_file=trajectory_file,
                working_dir=working_dir, params=params,
            )
        return results

    u = session.raw_universe
    session.record_pass("raw_fused_streaming", n_metrics=len(fused_steps))

    collectors: list = []
    for step_idx, tool_name, params in fused_steps:
        try:
            from .proximity_analyzer import apply_selection_from_files

            params = apply_selection_from_files(dict(params), working_dir=working_dir)
            params = _params_for_session(session, params)
        except ChainSelectionError as exc:
            results[step_idx] = {"success": False, "error": str(exc)}
            continue
        try:
            col = _build_fused_collector(u, tool_name, params, step_idx)
        except Exception as exc:
            # A missing ligand (apo + resname ATP) must not abort RMSD/Rg/DSSP.
            logger.warning("Fused collector %s skipped: %s", tool_name, exc)
            results[step_idx] = {"success": False, "error": str(exc)}
            continue
        if col is None:
            try:
                results[step_idx] = compute_metric_from_session(
                    tool_name, session, TrajectoryPassKind.RAW,
                    topology_file=topology_file, trajectory_file=trajectory_file,
                    working_dir=working_dir, params=params,
                )
            except Exception as exc:
                logger.warning("Fused metric %s failed: %s", tool_name, exc)
                results[step_idx] = {"success": False, "error": str(exc)}
        else:
            collectors.append(col)

    for ts in u.trajectory:
        box = getattr(ts, "dimensions", None)
        for col in collectors:
            if col.get("failed"):
                continue
            if ts.frame % col["interval"] != 0:
                continue
            try:
                _fused_collect_frame(col, ts, box, mda_distances, np, minimum_image_distance)
            except Exception as exc:
                logger.warning(
                    "Fused metric %s failed at frame %s: %s",
                    col.get("tool"), ts.frame, exc,
                )
                col["failed"] = str(exc)
                results[col["step_idx"]] = {"success": False, "error": str(exc)}

    original_dir = os.getcwd()
    try:
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)

        for col in collectors:
            if col.get("failed"):
                continue
            if col["type"] == "rg":
                from .gyration_calculator import finalize_rg_result

                results[col["step_idx"]] = finalize_rg_result(
                    col["times"], col["values"],
                    selection=col["params"].get("selection", "protein"),
                    output_file=col["params"].get("output_file"),
                    topology_file=topology_file,
                    trajectory_file=trajectory_file,
                    working_dir=working_dir,
                    align_before_rg=False,
                )
            elif col["type"] == "com":
                from .com_distance_calculator import finalize_com_distance_result

                results[col["step_idx"]] = finalize_com_distance_result(
                    col["frames"], col["times"], col["values"],
                    selection1=col["params"].get("selection1", "protein"),
                    selection2=col["params"].get("selection2", "resname LIG"),
                    label1=col["params"].get("label1", "group1"),
                    label2=col["params"].get("label2", "group2"),
                    output_file=col["params"].get("output_file"),
                    topology_file=topology_file,
                    trajectory_file=trajectory_file,
                    working_dir=working_dir,
                    frame_interval=col["interval"],
                )
            elif col["type"] == "pocket":
                from .com_distance_calculator import finalize_ligand_pocket_distance_result

                results[col["step_idx"]] = finalize_ligand_pocket_distance_result(
                    col["frames"], col["times"], col["values"],
                    ligand_selection=col["params"].get("ligand_selection", "resname LIG"),
                    protein_selection=col["params"].get("protein_selection", "protein"),
                    cutoff=float(col["params"].get("cutoff", 5.0)),
                    pocket_resids=col["pocket_resids"],
                    n_pocket_atoms=col["n_pocket_atoms"],
                    output_file=col["params"].get("output_file"),
                    topology_file=topology_file,
                    trajectory_file=trajectory_file,
                    working_dir=working_dir,
                )
            elif col["type"] == "residence":
                from .binding_site_analyzer import finalize_ligand_residence_result

                results[col["step_idx"]] = finalize_ligand_residence_result(
                    col["times"], col["bound_mask"], col["min_dists"],
                    bound_distance_A=col["bound_distance_A"],
                    min_contacts=col["min_contacts"],
                    pocket_meta=col["pocket_meta"],
                    output_file=col["params"].get("output_file"),
                    working_dir=working_dir,
                )
            elif col["type"] == "min_dist":
                from .proximity_analyzer import finalize_min_distance_result

                results[col["step_idx"]] = finalize_min_distance_result(
                    col["frames"], col["times"], col["values"],
                    selection1=col["params"].get("selection1", "protein"),
                    selection2=col["params"].get("selection2", "protein"),
                    label1=col["params"].get("label1", "group1"),
                    label2=col["params"].get("label2", "group2"),
                    output_file=col["params"].get("output_file"),
                    topology_file=topology_file,
                    trajectory_file=trajectory_file,
                    working_dir=working_dir,
                    frame_interval=col["interval"],
                )
    finally:
        if working_dir:
            os.chdir(original_dir)

    return results
