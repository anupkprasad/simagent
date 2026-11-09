"""Shared helpers for modular family dynamics (torsions, PCA/tICA, FEL).

General-purpose utilities used by consensus torsion and dynamics tools.
Independent of ment campaign scripts; classical tICA (no deeptime).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TEMPERATURE_K = 310.0
FEL_BINS = 50
SMOOTH_SIGMA = 2.0
MERGE_BARRIER_KBT = 3.0
DEFAULT_PCA_WHITEN_DIM = 50
DEFAULT_N_TICS = 10
DEFAULT_LAG_FRAMES = 10

# Predictable output directory tags for classifier discovery.
DYNAMICS_OUT_DIRS = {
    ("dihedral", "pca", "independent"): "consensus_PCA",
    ("dihedral", "tica", "independent"): "consensus_TICA",
    ("cartesian", "pca", "independent"): "consensus_cart_PCA",
    ("cartesian", "tica", "independent"): "consensus_cart_TICA",
    ("dihedral", "pca", "shared_reference"): "consensus_PCA_ref",
    ("dihedral", "tica", "shared_reference"): "consensus_TICA_ref",
    ("cartesian", "pca", "shared_reference"): "consensus_cart_PCA_ref",
    ("cartesian", "tica", "shared_reference"): "consensus_cart_TICA_ref",
}


def circ_mean_std_deg(deg: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(deg, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rad = np.deg2rad(x)
    c = float(np.mean(np.cos(rad)))
    s = float(np.mean(np.sin(rad)))
    mean = float(np.rad2deg(np.arctan2(s, c)))
    r = float(np.hypot(c, s))
    std = float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-12))))))
    return mean, std


def dynamics_output_dirname(
    space: str, method: str, projection_mode: str
) -> str:
    key = (space.lower(), method.lower(), projection_mode.lower())
    if key not in DYNAMICS_OUT_DIRS:
        raise ValueError(
            f"Unknown dynamics combo space={space!r} method={method!r} "
            f"projection_mode={projection_mode!r}"
        )
    return DYNAMICS_OUT_DIRS[key]


def load_consensus_alignment(alignment_json: str | Path) -> Dict[str, Any]:
    """Load star-MSA or ment-style consensus alignment JSON."""
    path = Path(alignment_json)
    if not path.is_file():
        raise FileNotFoundError(f"Alignment JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def _label_lookup_aliases(label: str, alignment: Dict[str, Any]) -> List[str]:
    """Gene ↔ UniProt aliases for ment-style ``uniprot_to_name`` maps."""
    lab = label.strip()
    aliases = [lab]
    base = lab
    for sfx in ("_ATP", "_ADP", "_AMP", "_atp", "_adp", "_amp"):
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break
    if base not in aliases:
        aliases.append(base)
    u2n = alignment.get("uniprot_to_name") or {}
    # uniprot → gene
    for uid, name in u2n.items():
        if str(uid).lower() == base.lower() or str(uid).lower() == lab.lower():
            if name and str(name) not in aliases:
                aliases.append(str(name))
            break
    # gene → uniprot (+ optional _ATP sim folder name)
    for uid, name in u2n.items():
        if str(name).lower() == base.lower() or str(name).lower() == lab.lower():
            if str(uid) not in aliases:
                aliases.append(str(uid))
            atp = f"{uid}_ATP"
            if atp not in aliases:
                aliases.append(atp)
            break
    # de-dup case-insensitively preserving order
    seen = set()
    out: List[str] = []
    for a in aliases:
        key = a.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def mapping_for_label(
    alignment: Dict[str, Any], label: str
) -> List[Optional[Dict[str, Any]]]:
    """
    Per-consensus-position residue mapping for one simulation label.

    Supports:
    - Compact v2: ``per_sim[label].resids``
    - Framework star MSA: ``consensus_positions[].mappings[label]``
    - Ment-style: ``proteins[label].mapping`` list aligned to consensus
    - Ment gene-keyed maps via ``uniprot_to_name`` (e.g. ``p17612_ATP`` ↔ ``KAPCA``)
    """
    from src.analysis.cross_sim_artifacts import expand_consensus_positions

    lab = label.strip()
    aliases = _label_lookup_aliases(lab, alignment)
    alias_l = {a.lower() for a in aliases}
    # Ment / occupancy style
    proteins = alignment.get("proteins") or {}
    for key, val in proteins.items():
        if str(key).lower() in alias_l:
            mapping = val.get("mapping") if isinstance(val, dict) else None
            if mapping is not None:
                return list(mapping)
    # Compact / legacy positions
    positions = alignment.get("consensus_positions")
    if not positions:
        positions = expand_consensus_positions(alignment)
    if positions is None and "consensus" in alignment:
        positions = alignment["consensus"]
    if not positions:
        msa = alignment.get("msa") or alignment
        positions = msa.get("consensus_positions") or expand_consensus_positions(msa)
    out: List[Optional[Dict[str, Any]]] = []
    for pos in positions:
        maps = (pos or {}).get("mappings") or {}
        hit = None
        for k, v in maps.items():
            if str(k).lower() in alias_l:
                hit = v
                break
        if hit is None:
            out.append(None)
        elif isinstance(hit, dict):
            out.append(
                {
                    "resid": hit.get("resid"),
                    "aa": hit.get("aa") or hit.get("resname"),
                    "seq_index": hit.get("seq_index"),
                }
            )
        else:
            out.append(None)
    return out


def consensus_positions_list(alignment: Dict[str, Any]) -> List[Dict[str, Any]]:
    from src.analysis.cross_sim_artifacts import expand_consensus_positions

    positions = alignment.get("consensus_positions")
    if positions:
        return list(positions)
    expanded = expand_consensus_positions(alignment)
    if expanded:
        return expanded
    if "consensus" in alignment and isinstance(alignment["consensus"], list):
        return list(alignment["consensus"])
    n = alignment.get("n_consensus_positions")
    if n:
        return [{"consensus_index": i, "msa_col": i} for i in range(int(n))]
    mapping_len = 0
    proteins = alignment.get("proteins") or {}
    for val in proteins.values():
        if isinstance(val, dict) and val.get("mapping"):
            mapping_len = max(mapping_len, len(val["mapping"]))
    return [{"consensus_index": i, "msa_col": i} for i in range(mapping_len)]


def load_pocket_consensus_indices(pocket_map_csv: Optional[str]) -> Optional[List[int]]:
    if not pocket_map_csv:
        return None
    path = Path(pocket_map_csv)
    if not path.is_file():
        return None
    idxs: List[int] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key in ("consensus_index", "consensus_idx", "ci"):
                if key in row and str(row[key]).strip() != "":
                    idxs.append(int(row[key]))
                    break
    return sorted(set(idxs))


def sincos_from_deg(deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(np.asarray(deg, dtype=float))
    return np.sin(rad), np.cos(rad)


def build_dihedral_feature_matrix(
    phi: np.ndarray,
    psi: np.ndarray,
    chi1: Optional[np.ndarray],
    *,
    angles: Sequence[str] = ("phi", "psi", "chi1"),
    has_chi1_mask: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """
    Build n_frames × n_features matrix from angle arrays (degrees).

    Column order per residue: requested angles as sin/cos pairs.
    ``phi``/``psi`` shape (n_frames, n_res); ``chi1`` same or None.
    """
    angles_l = [a.lower() for a in angles]
    n_frames, n_res = phi.shape
    blocks: List[np.ndarray] = []
    for j in range(n_res):
        if "phi" in angles_l:
            s, c = sincos_from_deg(phi[:, j])
            blocks.append(np.column_stack([s, c]))
        if "psi" in angles_l:
            s, c = sincos_from_deg(psi[:, j])
            blocks.append(np.column_stack([s, c]))
        if "chi1" in angles_l and chi1 is not None:
            use = True if has_chi1_mask is None else bool(has_chi1_mask[j])
            if use and np.isfinite(chi1[:, j]).mean() > 0.1:
                s, c = sincos_from_deg(chi1[:, j])
                ok = np.isfinite(chi1[:, j])
                s = np.where(ok, s, 0.0)
                c = np.where(ok, c, 0.0)
                blocks.append(np.column_stack([s, c]))
    if not blocks:
        raise ValueError("No dihedral features constructed")
    return np.hstack(blocks)


def fit_pca(
    X: np.ndarray, n_components: int = 2
) -> Dict[str, Any]:
    mean = X.mean(axis=0)
    Xc = X - mean
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(int(n_components), Vt.shape[0], max(X.shape[0] - 1, 1))
    components = Vt[:k]
    proj = Xc @ components.T
    var = (S[:k] ** 2) / max(X.shape[0] - 1, 1)
    total = float(np.sum(S ** 2) / max(X.shape[0] - 1, 1))
    return {
        "mean": mean,
        "components": components,
        "singular_values": S[:k],
        "projections": proj,
        "explained_variance_fraction": (var / total) if total > 0 else var * 0,
    }


def fit_pca_whiten(
    X: np.ndarray, dim: int = DEFAULT_PCA_WHITEN_DIM
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    Xc = X - mean
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    d = min(dim, Vt.shape[0], max(X.shape[0] - 1, 1))
    scale = S[:d] / np.sqrt(max(X.shape[0] - 1, 1))
    scale = np.where(scale > 1e-12, scale, 1.0)
    Y = (Xc @ Vt[:d].T) / scale
    return Y, mean, Vt[:d], scale


def fit_tica(
    Y: np.ndarray, lag: int, n_tics: int
) -> Dict[str, Any]:
    """Classical TICA via symmetrized lag covariance."""
    from scipy import linalg

    T, d = Y.shape
    if T <= lag + 2:
        raise ValueError(f"Need more frames than lag ({T} <= {lag}+2)")
    y0 = Y[: T - lag]
    yt = Y[lag:]
    C0 = (y0.T @ y0) / max(y0.shape[0], 1)
    Ctau = (y0.T @ yt) / max(y0.shape[0], 1)
    Ctau = 0.5 * (Ctau + Ctau.T)
    C0 = 0.5 * (C0 + C0.T)
    tr = float(np.trace(C0))
    C0 = C0 + (1e-6 * max(tr, 1e-12)) * np.eye(d)
    evals, evecs = linalg.eigh(Ctau, C0)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    k = min(n_tics, d)
    evals = evals[:k]
    evecs = evecs[:, :k]
    proj = Y @ evecs
    timescales = []
    for lam in evals:
        lam_f = float(lam)
        if 0.0 < lam_f <= 1.0:
            timescales.append(-float(lag) / np.log(lam_f))
        else:
            timescales.append(float("nan"))
    kin = np.maximum(evals, 0.0)
    kin = kin / (float(np.sum(kin)) + 1e-30)
    return {
        "projections": proj,
        "eigenvalues": evals,
        "timescales_frames": np.asarray(timescales),
        "kinetic_variance_fraction": kin,
        "eigenvectors": evecs,
        "lag_frames": int(lag),
    }


def project_pca(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X - mean) @ components.T


def project_tica(Y: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    return Y @ eigenvectors


def fel_descriptors_from_proj2(
    proj2: np.ndarray,
    *,
    bins: int = FEL_BINS,
    temperature_k: float = TEMPERATURE_K,
) -> Dict[str, Any]:
    from src.analysis.pca_analyzer import (
        DEFAULT_MIN_BASIN_POPULATION,
        analyze_fel_landscape_core,
        _compute_free_energy_grid,
    )

    grid = _compute_free_energy_grid(
        proj2[:, 0], proj2[:, 1], bins=bins, temperature_k=temperature_k
    )
    if not grid.get("success"):
        raise RuntimeError(grid.get("error") or "FEL grid failed")
    F = np.asarray(grid["free_energy"], dtype=float)
    P = np.asarray(grid["probability"], dtype=float)
    xc = np.asarray(grid["x_centers"], dtype=float)
    yc = np.asarray(grid["y_centers"], dtype=float)
    if F.shape == (len(xc), len(yc)):
        F = F.T
        P = P.T
    merge_barrier_kj = MERGE_BARRIER_KBT * 0.00831446261815324 * temperature_k
    feat = analyze_fel_landscape_core(
        F,
        P,
        smooth_sigma=SMOOTH_SIGMA,
        min_basin_population=DEFAULT_MIN_BASIN_POPULATION,
        min_prominence_kj_mol=1.5,
        merge_barrier_kJ_mol=merge_barrier_kj,
    )
    p_flat = P.ravel()
    p_flat = p_flat[p_flat > 0]
    grid_entropy = float(-np.sum(p_flat * np.log(p_flat + 1e-12)))
    return {
        "grid_entropy": grid_entropy,
        "major_basin_population": float(feat["major_basin_population"]),
        "landscape_entropy": float(feat["landscape_entropy"]),
        "n_basins": int(feat["n_basins"]),
        "F": F,
        "P": P,
        "x_centers": xc,
        "y_centers": yc,
        "fel_features": feat,
    }


def write_fel_outputs(
    out_dir: Path,
    *,
    proj: np.ndarray,
    desc: Dict[str, Any],
    method: str,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_path = out_dir / "pca_projections.npy" if method == "pca" else out_dir / "tica_projections.npy"
    # Always also write generic name for discovery
    np.save(out_dir / "projections.npy", proj)
    np.save(proj_path, proj)

    xname = "PC1" if method == "pca" else "IC1"
    yname = "PC2" if method == "pca" else "IC2"
    grid_name = (
        "fel_pc1_pc2_grid.csv" if method == "pca" else "fel_ic1_ic2_grid.csv"
    )
    grid_path = out_dir / grid_name
    with open(grid_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([xname, yname, "free_energy_kJ_mol", "probability"])
        F, P, xc, yc = desc["F"], desc["P"], desc["x_centers"], desc["y_centers"]
        for j in range(F.shape[0]):
            for i in range(F.shape[1]):
                w.writerow(
                    [
                        f"{xc[i]:.6f}",
                        f"{yc[j]:.6f}",
                        f"{F[j, i]:.6f}",
                        f"{P[j, i]:.6e}",
                    ]
                )

    fel_feat = {
        "grid_entropy": desc["grid_entropy"],
        "major_basin_population": desc["major_basin_population"],
        "landscape_entropy": desc["landscape_entropy"],
        "n_basins": desc["n_basins"],
    }
    (out_dir / "fel_features.json").write_text(
        json.dumps(fel_feat, indent=2) + "\n", encoding="utf-8"
    )
    meta = {
        "success": True,
        "method": method,
        "n_frames": int(proj.shape[0]),
        "n_components_written": int(proj.shape[1]),
        **fel_feat,
        **(meta_extra or {}),
    }
    (out_dir / "dynamics_features.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    paths = {
        "projections": str(proj_path),
        "fel_grid": str(grid_path),
        "fel_features": str(out_dir / "fel_features.json"),
        "dynamics_features": str(out_dir / "dynamics_features.json"),
    }
    try:
        from src.analysis.feature_matrix_plots import save_feature_matrix_png

        png = save_feature_matrix_png(
            proj[:, :2] if proj.shape[1] >= 2 else proj,
            out_dir / "projections.png",
            title=f"{method.upper()} projections",
            kind="projections",
        )
        if png:
            paths["projections_plot"] = png
            # Optional FEL surface from grid
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                F, xc, yc = desc["F"], desc["x_centers"], desc["y_centers"]
                fig, ax = plt.subplots(figsize=(6, 5))
                extent = [float(xc[0]), float(xc[-1]), float(yc[0]), float(yc[-1])]
                im = ax.imshow(
                    F,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    cmap="inferno",
                )
                fig.colorbar(im, ax=ax, label="ΔG (kJ/mol)")
                ax.set_xlabel("PC1" if method == "pca" else "IC1")
                ax.set_ylabel("PC2" if method == "pca" else "IC2")
                ax.set_title(f"{method.upper()} FEL")
                fel_png = out_dir / (
                    "fel_pc1_pc2.png" if method == "pca" else "fel_ic1_ic2.png"
                )
                fig.tight_layout()
                fig.savefig(fel_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                paths["fel_plot"] = str(fel_png)
            except Exception:
                logger.debug("FEL surface plot skipped", exc_info=True)
    except Exception:
        logger.debug("projection plot skipped", exc_info=True)
    return paths


def save_model_npz(path: Path, model: Dict[str, Any]) -> None:
    path = Path(path)
    stem = path.with_suffix("") if path.suffix in {".npz", ".json"} else path
    stem.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta = {}
    for k, v in model.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            meta[k] = v
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    stem.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )


def load_model_npz(path: Path) -> Dict[str, Any]:
    path = Path(path)
    stem = path.with_suffix("") if path.suffix in {".npz", ".json"} else path
    npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")
    if not npz_path.is_file():
        raise FileNotFoundError(f"Model npz not found: {npz_path}")
    data = dict(np.load(npz_path, allow_pickle=False))
    if json_path.is_file():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        data.update(meta)
    return data
