"""
Unsupervised clustering on classification feature matrices.

Reads ``classification_features_zscore.csv`` (recommended) or raw CSV,
runs hierarchical (default) or k-means clustering, and writes assignments
plus plots labeled with protein / simulation names.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain.tools import tool
from scipy.cluster import hierarchy, vq
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None  # type: ignore

from src.reporter.combined_reporter import _parse_label_name_map, resolve_display_label

_META_COLUMNS = frozenset({"label", "sim_directory", "n_features_present"})

# Reference FEL + reference-mapped pocket clustering outputs (combined analysis).
REFERENCE_CLUSTER_METRIC_GROUPS = frozenset(
    {"reference_pocket", "reference_fel", "reference_pca",
     "reference_pocket_archetype", "reference_fel_archetype"}
)
REFERENCE_CLUSTERING_OUTPUT_FILES: Dict[str, str] = {
    "features_csv": "ref_fel_pock_features.csv",
    "zscore_csv": "ref_fel_pock_features_zscore.csv",
    "manifest_json": "ref_fel_pock_features.json",
    "xlsx": "ref_fel_pock_features.xlsx",
    "assignments_csv": "ref_fel_pock_cluster_assignments.csv",
    "summary_json": "ref_fel_pock_clusters.json",
    "pca_png": "ref_fel_pock_clusters_pca.png",
    "dendrogram_png": "ref_fel_pock_dendrogram.png",
    "phylo_png": "ref_fel_pock_phylo_tree.png",
    "heatmap_png": "ref_fel_pock_features_heatmap.png",
    "panel_png": "ref_fel_pock_dendrogram_heatmap.png",
}

# Parallel hybrid: local FEL + reference-pocket archetype (separate output dir).
LOCAL_FEL_REF_POCK_CLUSTERING_OUTPUT_FILES: Dict[str, str] = {
    "features_csv": "local_fel_ref_pock_features.csv",
    "zscore_csv": "local_fel_ref_pock_features_zscore.csv",
    "manifest_json": "local_fel_ref_pock_features.json",
    "xlsx": "local_fel_ref_pock_features.xlsx",
    "assignments_csv": "local_fel_ref_pock_cluster_assignments.csv",
    "summary_json": "local_fel_ref_pock_clusters.json",
    "pca_png": "local_fel_ref_pock_clusters_pca.png",
    "dendrogram_png": "local_fel_ref_pock_dendrogram.png",
    "phylo_png": "local_fel_ref_pock_phylo_tree.png",
    "heatmap_png": "local_fel_ref_pock_features_heatmap.png",
    "panel_png": "local_fel_ref_pock_dendrogram_heatmap.png",
    "mds_png": "local_fel_ref_pock_mds_map.png",
}

# Consensus-mapped Cα local FEL + reference pocket (sibling of local_fel_ref_pock).
CONSENSUS_LOCAL_FEL_REF_POCK_CLUSTERING_OUTPUT_FILES: Dict[str, str] = {
    "features_csv": "consensus_local_fel_ref_pock_features.csv",
    "zscore_csv": "consensus_local_fel_ref_pock_features_zscore.csv",
    "manifest_json": "consensus_local_fel_ref_pock_features.json",
    "xlsx": "consensus_local_fel_ref_pock_features.xlsx",
    "assignments_csv": "consensus_local_fel_ref_pock_cluster_assignments.csv",
    "summary_json": "consensus_local_fel_ref_pock_clusters.json",
    "pca_png": "consensus_local_fel_ref_pock_clusters_pca.png",
    "dendrogram_png": "consensus_local_fel_ref_pock_dendrogram.png",
    "phylo_png": "consensus_local_fel_ref_pock_phylo_tree.png",
    "heatmap_png": "consensus_local_fel_ref_pock_features_heatmap.png",
    "panel_png": "consensus_local_fel_ref_pock_dendrogram_heatmap.png",
    "mds_png": "consensus_local_fel_ref_pock_mds_map.png",
}

REF_FEL_CLUSTERING_OUTPUT_FILES: Dict[str, str] = {
    "features_table": "ref_fel_features_table.json",
    "assignments_csv": "ref_fel_cluster_assignments.csv",
    "summary_json": "ref_fel_clusters.json",
    "dendrogram_png": "ref_fel_dendrogram.png",
    "phylo_png": "ref_fel_phylo_tree.png",
}


def is_reference_structure_clustering(
    metric_groups: Optional[Sequence[str]],
) -> bool:
    """True when clustering should use reference_* outputs only (not classification_*).

    Modular family dynamics (``consensus_rmsf`` / torsions / DCCM / dihedral PCA)
    is
    **not** reference-structure clustering even though it uses mapped pocket
    COM/angle columns from the consensus pocket batch.
    """
    if not metric_groups:
        return False
    groups = set(metric_groups)
    # Modular family dynamics → classification_* path (not reference archetype).
    if groups & {
        "paper_ward4",  # legacy tag; prefer consensus_* / dihedral_pca
        "consensus_rmsf",
        "consensus_torsions",
        "consensus_dccm",
        "dihedral_pca",
        "dihedral_tica",
        "cart_pca",
        "cart_tica",
    }:
        return False
    ref_groups = {g for g in groups if g in REFERENCE_CLUSTER_METRIC_GROUPS}
    return bool(ref_groups) and (
        "reference_pocket" in ref_groups
        or "reference_fel" in ref_groups
        or "reference_pocket_archetype" in ref_groups
        or "reference_fel_archetype" in ref_groups
    )


def reference_archetype_metric_groups(
    metric_groups: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    """Use curated archetype features when reference pocket+FEL clustering is requested."""
    from src.analysis.classification_collector import REFERENCE_ARCHETYPE_METRIC_GROUPS

    if not metric_groups:
        return tuple()
    groups = set(metric_groups)
    if groups & {
        "paper_ward4",
        "consensus_rmsf",
        "consensus_torsions",
        "consensus_dccm",
        "dihedral_pca",
    }:
        return tuple(g for g in metric_groups if g)
    if groups & {
        "reference_pocket",
        "reference_fel",
        "reference_pocket_archetype",
        "reference_fel_archetype",
    }:
        return REFERENCE_ARCHETYPE_METRIC_GROUPS
    return tuple(g for g in metric_groups if g in REFERENCE_CLUSTER_METRIC_GROUPS)


# Reference archetype clustering: up-weight COM/angle deviation so low-variation
# binders group together (Ward uses column * sqrt(weight) on z-scored features).
REFERENCE_ARCHETYPE_COUPLING_DEVIATION_WEIGHT = 2.0
REFERENCE_ARCHETYPE_COUPLING_DEVIATION_FEATURES: Tuple[str, ...] = (
    "reference_pocket_ligand_distance_std_A",
    "reference_pocket_ligand_distance_p95_A",
    "reference_pocket_std_ligand_axis_angle_deg",
    "reference_pocket_ligand_axis_angle_p95_deg",
)


def reference_archetype_feature_weights(
    coupling_deviation_weight: float = REFERENCE_ARCHETYPE_COUPLING_DEVIATION_WEIGHT,
) -> Dict[str, float]:
    """Per-feature Ward weights for reference pocket + FEL archetype clustering."""
    from src.analysis.classification_collector import CLASSIFICATION_FEATURE_GROUPS

    weights: Dict[str, float] = {}
    for group in ("reference_pocket_archetype", "reference_fel_archetype"):
        for col in CLASSIFICATION_FEATURE_GROUPS.get(group, ()):
            weights[col] = 1.0
    for col in REFERENCE_ARCHETYPE_COUPLING_DEVIATION_FEATURES:
        if col in weights:
            weights[col] = float(coupling_deviation_weight)
    return weights


def _apply_feature_weights(
    X: np.ndarray,
    feature_cols: Sequence[str],
    feature_weights: Optional[Dict[str, float]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Scale z-score columns by sqrt(weight) for weighted Euclidean / Ward clustering."""
    if not feature_weights:
        return X, {}
    Xw = np.array(X, dtype=float, copy=True)
    applied: Dict[str, float] = {}
    for j, col in enumerate(feature_cols):
        w = float(feature_weights.get(col, 1.0))
        if w != 1.0:
            Xw[:, j] *= np.sqrt(w)
            applied[col] = w
    return Xw, applied


def _default_n_clusters(n_samples: int) -> int:
    """Heuristic cluster count when the user does not specify one."""
    if n_samples <= 2:
        return max(1, n_samples)
    return max(2, min(8, int(round(np.sqrt(n_samples)))))


def _parse_n_clusters_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\bk\s*=\s*(\d{1,2})\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{1,2})\s+clusters?\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bcluster(?:ing)?\s+(?:into|with|using)\s+(\d{1,2})\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(?:default|use|with)\s+k\s*[=:]?\s*(\d{1,2})\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def resolve_n_clusters_for_goal(
    n_samples: int,
    user_goal_text: str = "",
    *,
    reference_based: bool = False,
) -> int:
    """Resolve k from goal text only; otherwise use a gentle default.

    Do **not** force a fixed cluster count (e.g. paper-style k=4). Humans
    interpret dendrogram cuts; auto-k is only for optional coloring.
    """
    del reference_based  # retained for call-site compatibility
    parsed = _parse_n_clusters_from_text(user_goal_text)
    if parsed is not None:
        return max(1, min(parsed, n_samples))
    return _default_n_clusters(n_samples)


def clustering_method_for_goal(*goal_texts: str) -> str:
    """Infer clustering method from goal text.

    Returns:
      - ``kmeans`` when k-means is explicitly requested
      - ``consensus_hierarchical`` when the goal asks for consensus / bootstrap /
        leave-one-out co-association clustering (paper family-scale recipe)
      - ``hierarchical`` otherwise (single Ward/average cut)
    """
    for text in goal_texts:
        if not text:
            continue
        normalized = text.lower().replace("\u2011", "-")
        if re.search(r"\bk[-\s]?means?\b", normalized):
            return "kmeans"
        if re.search(
            r"\bconsensus\b.*\b(hierarch|cluster)|"
            r"\b(bootstrap|leave[- ]one[- ]out|loo)\b.*\b(cluster|ward|co-?assoc)|"
            r"\bco-?association\b",
            normalized,
            flags=re.DOTALL,
        ):
            return "consensus_hierarchical"
        if "consensus hierarchical" in normalized or "consensus clustering" in normalized:
            return "consensus_hierarchical"
    return "hierarchical"


def _accumulate_coassociation(
    idxs: Sequence[int],
    part_ids: Sequence[int],
    co_sum: np.ndarray,
    co_count: np.ndarray,
) -> None:
    idxs_l = list(idxs)
    part = {i: int(c) for i, c in zip(idxs_l, part_ids)}
    for a in range(len(idxs_l)):
        ia = idxs_l[a]
        for b in range(a + 1, len(idxs_l)):
            ib = idxs_l[b]
            co_count[ia, ib] += 1.0
            co_count[ib, ia] += 1.0
            if part[ia] == part[ib]:
                co_sum[ia, ib] += 1.0
                co_sum[ib, ia] += 1.0


def _coassociation_matrix(
    scaled: np.ndarray,
    n_clusters: int,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    linkage_method: str = "ward",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Leave-one-out + bootstrap Ward co-association (paper consensus recipe)."""
    n = int(scaled.shape[0])
    co_sum = np.zeros((n, n), dtype=float)
    co_count = np.zeros((n, n), dtype=float)
    rng = np.random.default_rng(int(seed))
    k = max(1, min(int(n_clusters), n))

    for leave in range(n):
        keep = [i for i in range(n) if i != leave]
        if len(keep) < k:
            continue
        ids, _ = _cluster_hierarchical(
            scaled[keep], k, linkage_method=linkage_method
        )
        _accumulate_coassociation(keep, ids, co_sum, co_count)

    for _ in range(max(0, int(n_bootstrap))):
        draw = rng.integers(0, n, size=n)
        uniq = sorted({int(i) for i in draw})
        if len(uniq) < k + 1:
            continue
        ids, _ = _cluster_hierarchical(
            scaled[uniq], k, linkage_method=linkage_method
        )
        _accumulate_coassociation(uniq, ids, co_sum, co_count)

    with np.errstate(divide="ignore", invalid="ignore"):
        co = np.divide(
            co_sum, co_count, out=np.zeros_like(co_sum), where=co_count > 0
        )
    np.fill_diagonal(co, 1.0)
    tri = co_count[np.triu_indices(n, 1)]
    meta: Dict[str, Any] = {
        "method": f"LOO + bootstrap Ward k={k}",
        "n_loo": n,
        "n_bootstrap": int(n_bootstrap),
        "pair_obs_min": int(tri.min()) if tri.size else 0,
        "pair_obs_median": float(np.median(tri)) if tri.size else 0.0,
    }
    return co, meta


def _cluster_consensus_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
    linkage_method: str = "ward",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Consensus tree: Ward linkage of distance ``1 − co-association``.

    Returns ``(cluster_ids, linkage_matrix, coassociation, meta)``.
    ``X`` should already be robust-/z-scaled (as in classification_features_zscore.csv).
    """
    co, meta = _coassociation_matrix(
        X,
        n_clusters,
        n_bootstrap=n_bootstrap,
        seed=seed,
        linkage_method=linkage_method,
    )
    dist = np.clip(1.0 - co, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = hierarchy.linkage(condensed, method="ward")
    k = max(1, min(int(n_clusters), X.shape[0]))
    labels = hierarchy.fcluster(Z, t=k, criterion="maxclust")
    return labels.astype(int), Z, co, meta


def _label_base(label: str) -> str:
    """Strip common suffixes (``.pdb``) for name-map lookup."""
    low = label.lower()
    if low.endswith(".pdb"):
        return label[:-4]
    return label


def _load_feature_matrix(
    csv_path: Path,
    min_features_present: int = 1,
) -> Tuple[List[str], List[str], List[str], np.ndarray, List[str]]:
    """
    Load feature matrix from classification CSV.

    Returns (labels, display_names, sim_dirs, X, feature_columns).
    Rows with fewer than ``min_features_present`` non-NaN features are dropped.
    Rows are sorted by label for reproducible clustering order.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"No columns in {csv_path}")
        feature_cols = [c for c in reader.fieldnames if c not in _META_COLUMNS]

        labels: List[str] = []
        display_names: List[str] = []
        sim_dirs: List[str] = []
        rows: List[List[float]] = []

        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                continue
            vals: List[float] = []
            present = 0
            for col in feature_cols:
                raw = (row.get(col) or "").strip()
                if raw == "" or raw.lower() in {"nan", "none", "null", "inf", "-inf"}:
                    vals.append(np.nan)
                else:
                    try:
                        v = float(raw)
                        vals.append(v)
                        if np.isfinite(v):
                            present += 1
                    except ValueError:
                        vals.append(np.nan)
            if present < min_features_present:
                logger.warning(
                    "Skipping %s: only %d/%d features present",
                    label,
                    present,
                    len(feature_cols),
                )
                continue
            labels.append(label)
            display_names.append(label)
            sim_dirs.append((row.get("sim_directory") or "").strip())
            rows.append(vals)

    if not rows:
        raise ValueError(
            f"No usable rows in {csv_path} (need ≥{min_features_present} features per sim)"
        )

    order = sorted(range(len(labels)), key=lambda i: labels[i].lower())
    labels = [labels[i] for i in order]
    display_names = [display_names[i] for i in order]
    sim_dirs = [sim_dirs[i] for i in order]
    rows = [rows[i] for i in order]

    return labels, display_names, sim_dirs, np.array(rows, dtype=float), feature_cols


def _filter_usable_feature_columns(
    X: np.ndarray,
    feature_cols: Sequence[str],
    max_missing_fraction: float = 0.25,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Drop feature columns that are mostly missing or have <2 observed values.

    Returns (filtered_X, kept_columns, dropped_columns).
    """
    keep_indices: List[int] = []
    dropped: List[str] = []
    n_rows = X.shape[0]
    for j, col in enumerate(feature_cols):
        col_vals = X[:, j]
        n_missing = int(np.isnan(col_vals).sum())
        n_present = n_rows - n_missing
        if n_present < 2:
            dropped.append(col)
            continue
        if n_missing / n_rows > max_missing_fraction:
            dropped.append(col)
            continue
        keep_indices.append(j)

    if not keep_indices:
        raise ValueError(
            "No usable feature columns remain after missing-data filtering "
            f"(max_missing_fraction={max_missing_fraction})"
        )

    kept = [feature_cols[j] for j in keep_indices]
    return X[:, keep_indices], kept, dropped


def _impute_column_means(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Replace NaN with column mean; all-NaN columns become zeros."""
    out = X.copy()
    imputed_cells = 0
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = np.isnan(col)
        if not mask.any():
            continue
        mean = np.nanmean(col)
        if np.isnan(mean):
            out[:, j] = 0.0
            imputed_cells += int(mask.sum())
        else:
            col[mask] = mean
            out[:, j] = col
            imputed_cells += int(mask.sum())
    return out, {"imputed_cells": imputed_cells, "total_cells": out.size}


def _pca_2d(X: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """First two principal components (SVD). Returns coords, var PC1, var PC2."""
    Xc = X - X.mean(axis=0)
    if Xc.shape[0] < 2 or Xc.shape[1] < 1:
        return np.zeros((X.shape[0], 2)), 0.0, 0.0
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ vt[:2].T
    n = max(X.shape[0] - 1, 1)
    var = (s ** 2) / n
    total = float(var.sum()) or 1.0
    pc1_var = float(var[0] / total) if len(var) > 0 else 0.0
    pc2_var = float(var[1] / total) if len(var) > 1 else 0.0
    return coords, pc1_var, pc2_var


def _cluster_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    linkage_method: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    Z = hierarchy.linkage(X, method=linkage_method)
    labels = hierarchy.fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels.astype(int), Z


def _cluster_representatives(
    X: np.ndarray,
    labels: Sequence[str],
    display_names: Sequence[str],
    cluster_ids: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Pick medoid (closest to centroid) per cluster in z-score feature space."""
    reps: Dict[str, Dict[str, Any]] = {}
    for cid in sorted({int(c) for c in cluster_ids}):
        idx = [i for i, c in enumerate(cluster_ids) if int(c) == cid]
        if not idx:
            continue
        sub = X[idx]
        centroid = sub.mean(axis=0)
        dists = np.linalg.norm(sub - centroid, axis=1)
        best_local = int(np.argmin(dists))
        best_idx = idx[best_local]
        reps[str(cid)] = {
            "label": labels[best_idx],
            "display_name": display_names[best_idx],
            "distance_to_centroid": float(dists[best_local]),
        }
    return reps


def _load_representative_overrides(out_dir: Path) -> Dict[str, str]:
    """Load optional ``{cluster_id: label_or_display_name}`` overrides from analysis dir."""
    for fname in (
        "ref_fel_pock_representative_overrides.json",
        "representative_overrides.json",
    ):
        path = out_dir / fname
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                continue
            return {
                str(k): str(v).strip()
                for k, v in data.items()
                if not str(k).startswith("_") and v
            }
        except Exception as exc:
            logger.warning("Could not read representative overrides %s: %s", path, exc)
    return {}


def _apply_representative_overrides(
    reps: Dict[str, Dict[str, Any]],
    overrides: Dict[str, str],
    labels: Sequence[str],
    display_names: Sequence[str],
    cluster_ids: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    """Replace medoid representatives when overrides name a cluster member."""
    if not overrides:
        return reps

    label_to_idx = {lbl.lower(): i for i, lbl in enumerate(labels)}
    display_to_idx = {disp.lower(): i for i, disp in enumerate(display_names)}

    out = dict(reps)
    for cid_key, target in overrides.items():
        try:
            cid = int(cid_key)
        except (TypeError, ValueError):
            continue
        key = target.lower()
        idx = label_to_idx.get(key)
        if idx is None:
            idx = display_to_idx.get(key)
        if idx is None:
            logger.warning(
                "Representative override %s → %r not found among cluster members",
                cid_key,
                target,
            )
            continue
        if int(cluster_ids[idx]) != cid:
            logger.warning(
                "Representative override %s → %r belongs to cluster %s, not %s; skipping",
                cid_key,
                target,
                cluster_ids[idx],
                cid,
            )
            continue
        out[str(cid)] = {
            "label": labels[idx],
            "display_name": display_names[idx],
            "distance_to_centroid": None,
            "override": True,
            "override_reason": f"Manual override: {target}",
        }
    return out


def _cluster_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if X.shape[0] <= n_clusters:
        return np.arange(1, X.shape[0] + 1, dtype=int)
    _, labels = vq.kmeans2(
        X,
        n_clusters,
        minit="++",
        seed=rng,
    )
    return labels.astype(int) + 1


def _apply_display_names(
    labels: Sequence[str],
    user_goal: str = "",
    label_name_map: Optional[Dict[str, str]] = None,
    base_analysis_dir: Optional[Path] = None,
) -> List[str]:
    name_map: Dict[str, str] = dict(label_name_map or {})
    parsed = _parse_label_name_map(user_goal)
    for key, val in parsed.items():
        name_map.setdefault(key.lower(), val)
    if base_analysis_dir is not None:
        try:
            from src.analysis.reference_labels import build_uniprot_display_map

            for uid, disp in build_uniprot_display_map(Path(base_analysis_dir)).items():
                name_map.setdefault(uid.lower(), disp)
        except Exception:
            logger.debug(
                "Could not load reference display-name map from %s",
                base_analysis_dir,
                exc_info=True,
            )
    out: List[str] = []
    for label in labels:
        base = _label_base(label)
        resolved = resolve_display_label(base, name_map)
        out.append(resolved)
    return out


def _plot_cluster_scatter(
    coords: np.ndarray,
    display_names: Sequence[str],
    cluster_ids: Sequence[int],
    output_path: Path,
    title: str,
    pc1_var: float,
    pc2_var: float,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    clusters = sorted(set(cluster_ids))
    cmap = plt.get_cmap("tab10", max(len(clusters), 1))
    for i, cid in enumerate(clusters):
        mask = np.array(cluster_ids) == cid
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=120,
            c=[cmap(i)],
            label=f"Cluster {cid}",
            edgecolors="black",
            linewidths=0.6,
            zorder=2,
        )
    for idx, name in enumerate(display_names):
        ax.annotate(
            name,
            (coords[idx, 0], coords[idx, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            fontweight="bold",
            zorder=3,
        )
    ax.set_xlabel(xlabel or f"PC1 ({pc1_var * 100:.1f}% var)")
    ax.set_ylabel(ylabel or f"PC2 ({pc2_var * 100:.1f}% var)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cmdscale_mds_2d(X: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Classical multidimensional scaling (2D) from row-wise feature vectors.

    Preserves pairwise Euclidean distances in z-score feature space — appropriate
    for simulation archetype similarity maps (not evolutionary trees).
    """
    n = X.shape[0]
    if n < 2:
        return np.zeros((n, 2)), 0.0, 0.0
    dist_sq = squareform(pdist(X, metric="euclidean")) ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ dist_sq @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    pos = evals > 1e-10
    if int(pos.sum()) < 1:
        return np.zeros((n, 2)), 0.0, 0.0
    n_keep = min(2, int(pos.sum()))
    vals = np.maximum(evals[:n_keep], 0.0)
    coords = evecs[:, :n_keep] * np.sqrt(vals)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(n)])
    total = float(vals.sum()) or 1.0
    mds1_frac = float(vals[0] / total)
    mds2_frac = float(vals[1] / total) if len(vals) > 1 else 0.0
    return coords, mds1_frac, mds2_frac


def _plot_mds_cluster_map(
    X: np.ndarray,
    display_names: Sequence[str],
    cluster_ids: Sequence[int],
    output_path: Path,
    title: str,
) -> None:
    """2D MDS map of z-score feature distances, colored by cluster assignment."""
    coords, mds1_frac, mds2_frac = _cmdscale_mds_2d(X)
    _plot_cluster_scatter(
        coords,
        display_names,
        cluster_ids,
        output_path,
        title=title,
        pc1_var=mds1_frac,
        pc2_var=mds2_frac,
        xlabel=f"MDS 1 ({mds1_frac * 100:.1f}% of distance variance)",
        ylabel=f"MDS 2 ({mds2_frac * 100:.1f}% of distance variance)",
    )


def _short_feature_label(col: str) -> str:
    """Compact axis label for classification heatmaps."""
    aliases = {
        "delta_F_major_minus_global_kJ_mol": "ΔF major−global (kJ/mol)",
        "landscape_entropy": "landscape entropy",
        "major_basin_population": "major basin population",
        "consensus_rmsf_mean_A": r"global$_{\mathrm{cons}}$ RMSF (mean)",
        "consensus_rmsf_std_A": r"global$_{\mathrm{cons}}$ RMSF (std)",
        "reference_pocket_ligand_distance_p95_A": "pocket–ATP COM (p95)",
        "reference_pocket_ligand_distance_mean_A": "pocket–ATP COM (mean)",
        "reference_pocket_ligand_distance_std_A": "pocket–ATP COM (std)",
        "reference_pocket_fraction_bound": "Fraction bound",
        "reference_pocket_ligand_axis_angle_p95_deg": "pocket–ATP angle (p95)",
        "reference_pocket_ligand_axis_angle_mean_deg": "pocket–ATP angle (mean)",
        "reference_pocket_std_ligand_axis_angle_deg": "pocket–ATP angle (std)",
        "mean_abs_corr_offdiag": r"DCCM $\langle|C|\rangle$",
        "dccm_mean_abs_corr_offdiag": r"DCCM $\langle|C|\rangle$",
        "dccm_top_eigenvalue": r"$\lambda_1$",
        "N_C_mean_corr": r"global$_{\mathrm{cons}}$ DCCM",
        "dccm_N_C_mean_corr": r"global$_{\mathrm{cons}}$ DCCM",
        "pocket_C_mean_corr": r"pocket$\leftrightarrow$C",
        "C_C_mean_corr": r"C$\leftrightarrow$C",
        "timescale_ic1_ns": r"$\tau_1$ (ns)",
        "timescale_ic2_ns": r"$\tau_2$ (ns)",
        "tica_timescale_ic1_ns": r"TICA $\tau_1$ (ns)",
        "tica_timescale_ic2_ns": r"TICA $\tau_2$ (ns)",
        "grid_entropy": "grid entropy",
        "tica_grid_entropy": "TICA grid entropy",
        "tica_major_basin_population": "TICA major basin pop.",
        "pca_grid_entropy": r"PCA grid entropy ($\phi$/$\psi$/$\chi_1$)",
        "pca_phipsi_grid_entropy": r"PCA grid entropy ($\phi$/$\psi$)",
        "chi1_circ_mean_deg": r"global torsion $\chi_1$ (mean)",
        "chi1_circ_std_deg": r"global torsion $\chi_1$ (std)",
        "chi1_pocket_circ_mean_deg": r"pocket torsion $\chi_1$ (mean)",
        "chi1_pocket_circ_std_deg": r"pocket torsion $\chi_1$ (std)",
        "phi_circ_mean_deg": r"Torsion $\phi$ (circ. mean)",
        "pca_pka_ref_shared_dyn": r"global$_{\mathrm{cons}}$ PCA (PKA ref)",
        "pca_pka_ref_shared_pc_rms": r"global$_{\mathrm{cons}}$ PCA PC RMS (PKA-ref)",
        "pca_pka_ref_shared_gmin_pc1": r"global$_{\mathrm{cons}}$ PCA gmin PC1 (PKA-ref)",
        "pca_pka_ref_shared_gmin_pc2": r"global$_{\mathrm{cons}}$ PCA gmin PC2 (PKA-ref)",
        "pca_pka_ref_shared_centroid_pc1": r"global$_{\mathrm{cons}}$ PCA centroid PC1 (PKA-ref)",
        "pca_pka_ref_shared_centroid_pc2": r"global$_{\mathrm{cons}}$ PCA centroid PC2 (PKA-ref)",
        "frac_frames_within_3kBT_of_minF": r"frac $\leq$ 3 kBT",
    }
    if col in aliases:
        return aliases[col]
    try:
        from src.reporter.combined_reporter import _METRIC_LABELS

        if col in _METRIC_LABELS:
            return _METRIC_LABELS[col][0]
    except Exception:
        pass
    return (
        col.replace("reference_pocket_", "")
        .replace("ref_", "")
        .replace("_", " ")
    )


# High-contrast palette for k=4 reference / classification dendrograms and phylo trees.
CLUSTER_COLORS_K4: Tuple[str, ...] = (
    "#E41A1C",  # cluster 1 — red
    "#377EB8",  # cluster 2 — blue
    "#4DAF4A",  # cluster 3 — green
    "#FF7F00",  # cluster 4 — orange
)

# Ward k=4 archetype labels (reference pocket + reference FEL clustering).
REFERENCE_CLUSTER_ARCHETYPE_NAMES: Dict[int, str] = {
    1: "Dissociated",
    2: "Dominant-basin coupling",
    3: "Canonical-like",
    4: "Tight rigid",
}


def _cluster_legend_label(
    cid: int,
    count: int,
    archetype_names: Optional[Dict[int, str]] = None,
) -> str:
    names = archetype_names if archetype_names is not None else REFERENCE_CLUSTER_ARCHETYPE_NAMES
    name = names.get(int(cid), "")
    if name:
        return f"C{cid}: {name} (n={count})"
    return f"Cluster {cid} (n={count})"


def _cluster_color_map(
    cluster_ids: Sequence[int],
    *,
    palette: Optional[Sequence[str]] = None,
) -> Tuple[List[int], Dict[int, Any]]:
    """Return sorted cluster ids and a stable color per cluster id (1→red, 2→blue, …)."""
    from matplotlib.colors import to_hex

    clusters = sorted(set(int(c) for c in cluster_ids))
    # Stable ColorBrewer/Set1-like palette covering k≤5 (matches K4 prefix).
    default_pal = CLUSTER_COLORS_K4 + ("#984EA3",)
    pal = palette or (default_pal if len(clusters) <= len(default_pal) else None)
    if pal is not None:
        colors = {cid: pal[(int(cid) - 1) % len(pal)] for cid in clusters}
        return clusters, colors
    cmap = plt.get_cmap("tab10", max(len(clusters), 1))
    colors = {cid: to_hex(cmap(i)) for i, cid in enumerate(clusters)}
    return clusters, colors


def _link_color_func_for_clusters(
    linkage_matrix: np.ndarray,
    cluster_ids: Sequence[int],
    cluster_colors: Dict[int, str],
) -> Any:
    """Color dendrogram branches by final cluster assignment (not scipy distance threshold)."""
    from matplotlib.colors import to_hex

    n = len(cluster_ids)
    node_cluster: Dict[int, int] = {i: int(cluster_ids[i]) for i in range(n)}
    for i, row in enumerate(linkage_matrix):
        a, b = int(row[0]), int(row[1])
        ca, cb = node_cluster[a], node_cluster[b]
        node_cluster[n + i] = ca if ca == cb else -1

    def link_color_func(k: int) -> str:
        cid = node_cluster.get(int(k), -1)
        if cid < 0:
            return "#B0B0B0"
        c = cluster_colors[cid]
        # Dendrogram requires hex/named colors — never str(rgba_tuple).
        return c if isinstance(c, str) else to_hex(c)

    return link_color_func


def _align_subplot_heights(anchor_ax, *axes) -> None:
    """Match vertical position and height of axes to a reference subplot."""
    pos = anchor_ax.get_position()
    for ax in axes:
        p = ax.get_position()
        ax.set_position([p.x0, pos.y0, p.width, pos.height])


def _normalize_highlight_names(names: Optional[Sequence[str]]) -> set:
    """Case-insensitive set of display names / labels to highlight."""
    if not names:
        return set()
    return {str(n).strip().lower() for n in names if str(n).strip()}


def _draw_simulation_label_bar(
    ax,
    names_ord: Sequence[str],
    cids_ord: Sequence[int],
    cluster_colors: Dict[int, str],
    *,
    fontsize: float = 11.0,
    highlight_names: Optional[Sequence[str]] = None,
) -> None:
    """Colored cluster strip with black protein names centered on each row.

    Names in ``highlight_names`` (e.g. ground-truth kinases) are prefixed with
    a large filled star (★) on this label bar only (no border styling).
    """
    from matplotlib.patches import Rectangle

    hi = _normalize_highlight_names(highlight_names)
    n = len(names_ord)
    ax.set_xlim(0, 1)
    ax.set_ylim(n - 0.5, -0.5)
    for i, (name, cid) in enumerate(zip(names_ord, cids_ord)):
        is_hi = str(name).strip().lower() in hi
        ax.add_patch(
            Rectangle(
                (0.02, i - 0.46),
                0.96,
                0.92,
                facecolor=cluster_colors[int(cid)],
                edgecolor="white",
                linewidth=0.6,
            )
        )
        if is_hi:
            # Star in left gutter; names share one left-aligned column.
            ax.scatter(
                [0.06],
                [i],
                marker="*",
                s=9 ** 2,
                c="black",
                zorder=6,
                clip_on=False,
                linewidths=0.2,
                edgecolors="black",
            )
        ax.text(
            0.22,
            i,
            str(name),
            ha="left",
            va="center",
            color="black",
            fontsize=fontsize,
            fontweight="bold",
            clip_on=True,
            zorder=5,
        )
    ax.axis("off")


def _plot_dendrogram_heatmap_panel(
    linkage_matrix: np.ndarray,
    X: np.ndarray,
    feature_cols: Sequence[str],
    display_names: Sequence[str],
    cluster_ids: Sequence[int],
    output_path: Path,
    *,
    dendrogram_title: str = "Hierarchical clustering dendrogram",
    heatmap_title: str = "Classification features (z-score)",
    colorbar_label: str = "z-score",
    archetype_names: Optional[Dict[int, str]] = None,
    legend_ncol: Optional[int] = None,
    highlight_display_names: Optional[Sequence[str]] = None,
    colorbar_symmetric: bool = True,
    highlight_dominant_features: bool = True,
    dominant_top_k: int = 3,
    dominant_min_abs: float = 0.65,
    legend_row_major: bool = False,
    layout_tight: bool = False,
    simple_panel: bool = False,
) -> Optional[List[int]]:
    """
    Single figure: dendrogram (branches right→left) + label bar + feature heatmap.

    Rows share the same simulation order top→bottom. Protein names appear in black
    on the colored cluster bar; the heatmap has no y-axis labels.
    Optional ``highlight_display_names`` (e.g. ground-truth kinases) are shown with
    a leading ★ on the label bar only.
    When ``colorbar_symmetric`` is False, the heatmap uses the data min/max (or
    [0, 1] if values already lie in that range) instead of a ±vmax diverging scale.
    When ``highlight_dominant_features`` is True, thin rectangles outline the
    strongest-magnitude feature blocks within each cluster's row span.
    When ``legend_row_major`` is True, legend entries fill left→right then top→bottom
    (matplotlib's default is column-major).
    When ``layout_tight`` is True, reduce figure margins / whitespace around the legend.
    When ``simple_panel`` is True, draw a plain gray dendrogram (no Ward-cut / cluster
    branch colors), neutral name labels, no archetype legend, and no feature boxes.
    """
    if not HAS_MPL:
        return None
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch, Rectangle

    n = len(display_names)
    clusters, cluster_colors = _cluster_color_map(cluster_ids)
    if simple_panel:
        highlight_dominant_features = False
        highlight_display_names = None
        # Uniform gray: scipy still paints Ward cuts when color_threshold is
        # omitted/None; force every link color explicitly.
        link_colors = lambda _k: "#555555"
        label_colors = {cid: "#E8E8E8" for cid in clusters}
    else:
        link_colors = _link_color_func_for_clusters(
            linkage_matrix, cluster_ids, cluster_colors
        )
        label_colors = cluster_colors

    # Slightly taller landscape panel for readability.
    # Match original Fig. 3 panel footprint (≈15.75"×11.64" at 300 dpi after tight bbox).
    fig_w = 16.0
    fig_h = max(10.8, min(13.8, n * 0.34))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.10, 0.42, 1.60],
        wspace=0.05 if layout_tight else 0.06,
        left=0.03,
        right=0.90,
        top=0.92 if layout_tight else 0.91,
        bottom=0.10 if simple_panel else 0.16,
    )
    ax_dend = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_hm = fig.add_subplot(gs[0, 2])

    dendro_kwargs: Dict[str, Any] = {
        "labels": [""] * n,
        "orientation": "left",
        "leaf_font_size": 11,
        "ax": ax_dend,
        "above_threshold_color": "#555555",
        "link_color_func": link_colors,
        "color_threshold": 0,
    }
    ddata = hierarchy.dendrogram(linkage_matrix, **dendro_kwargs)
    ax_dend.invert_yaxis()
    leaf_order_tb = [int(i) for i in ddata["leaves"]]

    ax_dend.set_title(dendrogram_title, fontsize=13, pad=8, loc="left", fontweight="bold")
    ax_dend.set_xlabel("Distance", fontsize=12)
    ax_dend.set_ylabel("")
    ax_dend.tick_params(axis="x", labelsize=11)
    ax_dend.tick_params(axis="y", left=False, labelleft=False)

    _align_subplot_heights(ax_dend, ax_bar, ax_hm)

    order = leaf_order_tb
    X_ord = X[order, :]
    names_ord = [display_names[i] for i in order]
    cids_ord = [int(cluster_ids[i]) for i in order]
    feat_labels = [_short_feature_label(c) for c in feature_cols]

    _draw_simulation_label_bar(
        ax_bar,
        names_ord,
        cids_ord,
        label_colors,
        fontsize=9.5,
        highlight_names=highlight_display_names,
    )

    finite = X_ord[np.isfinite(X_ord)]
    if colorbar_symmetric:
        from matplotlib.colors import LinearSegmentedColormap

        vmax = float(np.nanmax(np.abs(X_ord))) if finite.size else 2.5
        vmax = max(vmax, 1.0)
        vmin, vmax = -vmax, vmax
        # Softened diverging scale with moderately dark ends.
        cmap = LinearSegmentedColormap.from_list(
            "soft_blue_white_red",
            ["#4292C6", "#9ECAE1", "#FFFFFF", "#FCBBA1", "#EF6548"],
        )
    else:
        from matplotlib.colors import LinearSegmentedColormap

        if finite.size and float(np.nanmin(finite)) >= -1e-9 and float(np.nanmax(finite)) <= 1.0 + 1e-9:
            vmin, vmax = 0.0, 1.0
        elif finite.size:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if abs(vmax - vmin) < 1e-12:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        # Light blue → white → light red (midpoint at (vmin+vmax)/2).
        cmap = LinearSegmentedColormap.from_list(
            "light_blue_white_red",
            ["#6BAED6", "#FFFFFF", "#FC9272"],
        )
    im = ax_hm.imshow(
        X_ord,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_hm.set_xticks(np.arange(len(feature_cols)))
    ax_hm.set_xticklabels(
        feat_labels,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
        fontsize=9.5 if layout_tight else 11,
    )
    for tick in ax_hm.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
    ax_hm.tick_params(axis="x", pad=6 if layout_tight else 6, labelsize=9.5 if layout_tight else 11)
    ax_hm.set_yticks([])
    ax_hm.set_ylabel("")
    ax_hm.set_title(heatmap_title, fontsize=13, pad=6 if layout_tight else 8, loc="left", fontweight="bold")

    if highlight_dominant_features and X_ord.size:
        # Contiguous row blocks already follow dendrogram leaf order.
        for cid in clusters:
            rows = [i for i, c in enumerate(cids_ord) if c == cid]
            if not rows:
                continue
            r0, r1 = min(rows), max(rows)
            block = X_ord[r0 : r1 + 1, :]
            # Cluster-mean score per feature; large |mean| = dominant for this cluster.
            mean_z = np.nanmean(block, axis=0)
            score = np.abs(mean_z)
            order_f = np.argsort(-score)
            selected: List[int] = []
            for j in order_f:
                if float(score[j]) < float(dominant_min_abs) and selected:
                    break
                if float(score[j]) < 0.40:
                    break
                selected.append(int(j))
                if len(selected) >= max(1, int(dominant_top_k)):
                    break
            if not selected:
                selected = [int(order_f[0])]
            selected = sorted(set(selected))
            # Thin boxes around contiguous dominant-feature runs (keeps neighbors together).
            start = selected[0]
            prev = selected[0]
            ranges: List[Tuple[int, int]] = []
            for j in selected[1:]:
                if j == prev + 1:
                    prev = j
                else:
                    ranges.append((start, prev))
                    start = prev = j
            ranges.append((start, prev))
            edge = "#111111"
            for c0, c1 in ranges:
                ax_hm.add_patch(
                    Rectangle(
                        (c0 - 0.5, r0 - 0.5),
                        (c1 - c0 + 1),
                        (r1 - r0 + 1),
                        fill=False,
                        edgecolor=edge,
                        linestyle="--",
                        linewidth=1.6,
                        zorder=6,
                    )
                )

    if not simple_panel:
        counts: Dict[int, int] = {}
        for cid in cluster_ids:
            counts[int(cid)] = counts.get(int(cid), 0) + 1
        handles = [
            Patch(
                facecolor=cluster_colors[cid],
                edgecolor="black",
                label=_cluster_legend_label(
                    cid, counts.get(cid, 0), archetype_names=archetype_names
                ),
            )
            for cid in clusters
        ]
        hi = _normalize_highlight_names(highlight_display_names)
        if hi:
            from matplotlib.lines import Line2D

            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="black",
                    markerfacecolor="black",
                    markersize=9,
                    linestyle="None",
                    label="ground-truth kinase",
                )
            )
        ncol = int(legend_ncol) if legend_ncol is not None else 3
        if legend_row_major and ncol > 1 and len(handles) > 1:
            # Matplotlib fills legends column-major; permute so display is row-major
            # (left→right, then next row): e.g. C1 C2 C3 / C4 *
            n_h = len(handles)
            nrow = (n_h + ncol - 1) // ncol
            padded: List[Optional[object]] = list(handles) + [None] * (nrow * ncol - n_h)
            grid = [padded[r * ncol : (r + 1) * ncol] for r in range(nrow)]
            reordered = []
            for c in range(ncol):
                for r in range(nrow):
                    h = grid[r][c]
                    if h is not None:
                        reordered.append(h)
            handles = reordered
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=ncol,
            fontsize=9.5 if layout_tight else 11,
            framealpha=0.95,
            bbox_to_anchor=(0.5, 0.02),
            borderaxespad=0.0,
            columnspacing=2.0 if layout_tight else 1.5,
            handletextpad=0.7,
            labelspacing=0.7 if layout_tight else 0.5,
            borderpad=0.6,
            markerscale=1.0,
        )

    # Explicit half-height colorbar (after legend so layout is final).
    fig.canvas.draw()
    _bbox = ax_hm.get_position()
    _cax = fig.add_axes(
        [
            _bbox.x1 + 0.012,
            _bbox.y0 + 0.25 * _bbox.height,
            0.014,
            0.50 * _bbox.height,
        ]
    )
    cbar = fig.colorbar(im, cax=_cax)
    if colorbar_label:
        cbar_lab = str(colorbar_label)
        for tok in (" (k=5)", " (k = 5)", f" (k={len(clusters)})"):
            cbar_lab = cbar_lab.replace(tok, "")
        cbar.set_label(cbar_lab, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return leaf_order_tb


def _plot_dendrogram(
    linkage_matrix: np.ndarray,
    display_names: Sequence[str],
    output_path: Path,
    title: str,
    cluster_ids: Optional[Sequence[int]] = None,
    n_clusters: Optional[int] = None,
    leaf_order_tb: Optional[Sequence[int]] = None,
    highlight_display_names: Optional[Sequence[str]] = None,
) -> Optional[List[int]]:
    """Horizontal dendrogram (left→right); returns leaf order top→bottom.

    Names in ``highlight_display_names`` (e.g. ground-truth kinases) get a ★
    marker to the left of the leaf label.
    """
    if not HAS_MPL or cluster_ids is None:
        return None
    from matplotlib.patches import Patch

    n = len(display_names)
    clusters, cluster_colors = _cluster_color_map(cluster_ids)
    link_colors = _link_color_func_for_clusters(
        linkage_matrix, cluster_ids, cluster_colors
    )
    hi = _normalize_highlight_names(highlight_display_names)

    fig, ax = plt.subplots(figsize=(7.5, max(6, n * 0.38)))
    ddata = hierarchy.dendrogram(
        linkage_matrix,
        labels=list(display_names),
        orientation="right",
        leaf_font_size=9,
        ax=ax,
        link_color_func=link_colors,
        above_threshold_color="#B0B0B0",
        color_threshold=0,
    )
    ax.invert_yaxis()
    leaf_order = (
        [int(i) for i in leaf_order_tb]
        if leaf_order_tb is not None
        else [int(i) for i in ddata["leaves"]]
    )

    for tick, leaf_idx in zip(ax.get_ymajorticklabels(), leaf_order):
        cid = int(cluster_ids[leaf_idx])
        name = str(display_names[leaf_idx])
        tick.set_color(cluster_colors[cid])
        tick.set_fontweight("bold")
        if name.strip().lower() in hi:
            # Place a filled star just left of the leaf label.
            y = tick.get_position()[1]
            ax.scatter(
                [-0.02],
                [y],
                marker="*",
                s=11 ** 2,
                c="black",
                zorder=6,
                clip_on=False,
                transform=ax.get_yaxis_transform(),
                linewidths=0.2,
                edgecolors="black",
            )

    counts: Dict[int, int] = {}
    for cid in cluster_ids:
        counts[int(cid)] = counts.get(int(cid), 0) + 1
    handles = [
        Patch(
            facecolor=cluster_colors[cid],
            edgecolor="black",
            label=_cluster_legend_label(cid, counts.get(cid, 0)),
        )
        for cid in clusters
    ]
    if hi:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=11,
                linestyle="None",
                label="ground-truth kinase",
            )
        )
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return leaf_order


def _plot_cluster_feature_heatmap(
    X: np.ndarray,
    feature_cols: Sequence[str],
    display_names: Sequence[str],
    cluster_ids: Sequence[int],
    leaf_order: Sequence[int],
    output_path: Path,
    title: str,
    colorbar_label: str = "z-score",
) -> None:
    """Z-scored feature heatmap; rows top→bottom match dendrogram leaf order."""
    if not HAS_MPL or not len(leaf_order):
        return
    from matplotlib.cm import ScalarMappable

    order = [int(i) for i in leaf_order]
    X_ord = X[order, :]
    names_ord = [display_names[i] for i in order]
    cids_ord = [int(cluster_ids[i]) for i in order]
    _, cluster_colors = _cluster_color_map(cluster_ids)

    feat_labels = [_short_feature_label(c) for c in feature_cols]
    fig_h = max(5.5, len(order) * 0.34)
    fig_w = max(9.0, len(feature_cols) * 1.05 + 3.5)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.22, 1],
        wspace=0.04,
        left=0.08,
        right=0.92,
        top=0.94,
        bottom=0.12,
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])

    _draw_simulation_label_bar(ax_bar, names_ord, cids_ord, cluster_colors)

    vmax = float(np.nanmax(np.abs(X_ord))) if np.isfinite(X_ord).any() else 2.5
    vmax = max(vmax, 1.0)
    im = ax_hm.imshow(
        X_ord,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_hm.set_xticks(np.arange(len(feature_cols)))
    ax_hm.set_xticklabels(
        feat_labels, rotation=20, ha="right", rotation_mode="anchor", fontsize=9
    )
    ax_hm.tick_params(axis="x", pad=2)
    ax_hm.set_yticks([])
    ax_hm.set_title(title)
    cbar = fig.colorbar(
        ScalarMappable(norm=im.norm, cmap=im.cmap),
        ax=ax_hm,
        fraction=0.046,
        pad=0.04,
    )
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _tree_leaf_count(node) -> int:
    if node.is_leaf():
        return 1
    return _tree_leaf_count(node.get_left()) + _tree_leaf_count(node.get_right())


def _assign_circular_angles(node, start: float, span: float, angles: Dict[int, float]) -> None:
    """Assign polar angles proportional to subtree leaf counts."""
    if node.is_leaf():
        angles[node.id] = start + span / 2.0
        return
    left = node.get_left()
    right = node.get_right()
    left_count = _tree_leaf_count(left)
    right_count = _tree_leaf_count(right)
    total = left_count + right_count
    left_span = span * left_count / total if total else span / 2.0
    _assign_circular_angles(left, start, left_span, angles)
    _assign_circular_angles(right, start + left_span, span - left_span, angles)
    angles[node.id] = start + span / 2.0


def _assign_radial_depths(node, radius: float, radii: Dict[int, float]) -> None:
    """Assign radial depth from the tree root (center) using branch lengths."""
    radii[node.id] = radius
    if node.is_leaf():
        return
    _assign_radial_depths(node.get_left(), radius + node.get_left().dist, radii)
    _assign_radial_depths(node.get_right(), radius + node.get_right().dist, radii)


def _polar_to_xy(theta: float, radius: float) -> Tuple[float, float]:
    return radius * np.cos(theta), radius * np.sin(theta)


def _plot_unrooted_phylo_tree(
    linkage_matrix: np.ndarray,
    display_names: Sequence[str],
    cluster_ids: Sequence[int],
    output_path: Path,
    title: str,
) -> None:
    """
    Circular unrooted phylogram from a scipy linkage matrix.

    Leaf tips and terminal branches are colored by cluster assignment.
    """
    if not HAS_MPL:
        return

    n_samples = len(display_names)
    if n_samples < 2:
        return

    root = hierarchy.to_tree(linkage_matrix, rd=False)
    angles: Dict[int, float] = {}
    radii: Dict[int, float] = {}
    _assign_circular_angles(root, 0.0, 2.0 * np.pi, angles)
    _assign_radial_depths(root, 0.0, radii)
    max_radius = max(radii.values()) or 1.0
    label_radius = max_radius * 1.1

    clusters, cluster_colors = _cluster_color_map(cluster_ids)
    leaf_cluster = {idx: int(cluster_ids[idx]) for idx in range(n_samples)}

    fig_size = max(8.0, min(14.0, 6.0 + n_samples * 0.35))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.set_aspect("equal")

    def _draw_edges(node) -> None:
        if node.is_leaf():
            return
        parent_angle = angles[node.id]
        parent_radius = radii[node.id]
        parent_x, parent_y = _polar_to_xy(parent_angle, parent_radius)
        for child in (node.get_left(), node.get_right()):
            child_angle = angles[child.id]
            child_radius = radii[child.id]
            child_x, child_y = _polar_to_xy(child_angle, child_radius)
            junction_x, junction_y = _polar_to_xy(child_angle, parent_radius)
            if child.is_leaf():
                color = cluster_colors[leaf_cluster[child.id]]
                lw = 2.0
            else:
                color = "#666666"
                lw = 1.3
            ax.plot(
                [child_x, junction_x, parent_x],
                [child_y, junction_y, parent_y],
                color=color,
                linewidth=lw,
                solid_capstyle="round",
                zorder=1,
            )
            _draw_edges(child)

    _draw_edges(root)

    for idx, name in enumerate(display_names):
        cid = leaf_cluster[idx]
        color = cluster_colors[cid]
        theta = angles[idx]
        tip_x, tip_y = _polar_to_xy(theta, radii[idx])
        ax.scatter(
            [tip_x],
            [tip_y],
            s=140,
            c=[color],
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )
        label_x, label_y = _polar_to_xy(theta, label_radius)
        ha = "left" if np.cos(theta) >= 0 else "right"
        ax.text(
            label_x,
            label_y,
            name,
            ha=ha,
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color,
            zorder=4,
        )

    for cid in clusters:
        ax.scatter(
            [],
            [],
            c=[cluster_colors[cid]],
            s=80,
            edgecolors="black",
            linewidths=0.6,
            label=f"Cluster {cid}",
        )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(title)
    margin = max_radius * 0.28
    ax.set_xlim(-max_radius - margin, max_radius + margin)
    ax.set_ylim(-max_radius - margin, max_radius + margin)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@tool
def cluster_classification_features(
    working_dir: str = "",
    features_file: str = "classification_features_zscore.csv",
    method: str = "hierarchical",
    n_clusters: Optional[int] = None,
    linkage_method: str = "ward",
    user_goal: str = "",
    label_name_map: Optional[Dict[str, str]] = None,
    min_features_present: int = 1,
    max_column_missing_fraction: float = 0.25,
    assignments_file: str = "classification_cluster_assignments.csv",
    scatter_plot_file: str = "classification_clusters_pca.png",
    dendrogram_file: str = "classification_dendrogram.png",
    phylo_tree_file: str = "classification_phylo_tree.png",
    heatmap_file: str = "classification_features_heatmap.png",
    panel_file: Optional[str] = None,
    summary_file: str = "classification_clusters.json",
    feature_weights: Optional[Dict[str, float]] = None,
    cluster_archetype_names: Optional[Dict[int, str]] = None,
    legend_ncol: Optional[int] = None,
    feature_scale_label: str = "z-score",
    highlight_display_names: Optional[List[str]] = None,
    colorbar_symmetric: bool = True,
    simple_panel: bool = False,
) -> Dict[str, Any]:
    """
    Cluster simulations from a classification feature table (z-score CSV recommended).

    Default method is hierarchical agglomerative clustering (Ward linkage).
    Use method='kmeans' for k-means, or method='consensus_hierarchical' for the
    paper family-scale recipe (leave-one-out + bootstrap Ward co-association,
    then Ward linkage of ``1 − co-association``). Writes cluster assignments and
    a PCA scatter plot with protein/simulation name labels; hierarchical runs
    also get a horizontal dendrogram, circular similarity tree, and z-score
    feature heatmap.

    Args:
        working_dir: Directory containing the features CSV (usually base/analysis).
        features_file: Input CSV (prefer classification_features_zscore.csv).
        method: 'hierarchical' (default), 'consensus_hierarchical', or 'kmeans'.
        n_clusters: Number of clusters; auto-chosen from sample size if omitted.
        linkage_method: scipy linkage for hierarchical ('ward', 'average', 'complete').
        user_goal: Optional goal text to parse id:name labels (e.g. p23458:JAK1).
        label_name_map: Optional explicit {uniprot_id: protein_name} map (preferred).
        min_features_present: Skip sims with fewer non-NaN feature values.
        max_column_missing_fraction: Drop feature columns missing in more than
            this fraction of simulations (default 0.25) before clustering.
        assignments_file: Output CSV with label, display_name, cluster_id.
        scatter_plot_file: PCA scatter colored by cluster with name annotations.
        dendrogram_file: Horizontal dendrogram PNG (hierarchical only).
        phylo_tree_file: Unrooted circular similarity tree PNG (hierarchical only).
        heatmap_file: Z-score feature heatmap per simulation (hierarchical only).
        summary_file: JSON summary of clustering parameters and results.
        feature_scale_label: Label describing the feature scale in generated plots.
        highlight_display_names: Optional protein display names to outline on the
            dendrogram–heatmap panel (e.g. ground-truth kinases).
        colorbar_symmetric: If True (default), heatmap uses ±vmax diverging scale;
            if False, uses data range (or [0, 1] for min–max features).
        feature_weights: Optional map of feature column → relative weight (default 1).
            Columns are multiplied by sqrt(weight) after z-scoring so Ward distance
            emphasizes selected features (e.g. 2.0 doubles COM/angle deviation influence).
        cluster_archetype_names: Optional {cluster_id: short name} for dendrogram/
            panel legends (overrides REFERENCE_CLUSTER_ARCHETYPE_NAMES when set).
        legend_ncol: Optional legend column count.
        simple_panel: If True with ``panel_file``, draw plain dendrogram+heatmap
            (no cluster branch colors, archetype legend, or feature boxes).
    """
    original_dir = os.getcwd()
    try:
        feat_path = Path(features_file)
        if not feat_path.is_absolute():
            base = Path(working_dir) if working_dir else Path(".")
            feat_path = (base / features_file).resolve()
        if not feat_path.is_file():
            return {"success": False, "error": f"Features file not found: {feat_path}"}

        method_norm = (method or "hierarchical").lower().replace("-", "").replace("_", "")
        if method_norm in ("kmeans", "kmean"):
            method_key = "kmeans"
        elif method_norm in (
            "consensushierarchical",
            "consensus",
            "bootstrapconsensus",
            "coassociation",
        ):
            method_key = "consensus_hierarchical"
        elif method_norm in ("hierarchical", "hier", "agglomerative"):
            method_key = "hierarchical"
        else:
            return {
                "success": False,
                "error": (
                    f"Unknown method {method!r}; use 'hierarchical', "
                    "'consensus_hierarchical', or 'kmeans'"
                ),
            }

        labels, _, sim_dirs, X_raw, feature_cols = _load_feature_matrix(
            feat_path, min_features_present=min_features_present
        )
        X_filtered, feature_cols, dropped_cols = _filter_usable_feature_columns(
            X_raw,
            feature_cols,
            max_missing_fraction=max_column_missing_fraction,
        )
        if dropped_cols:
            logger.warning(
                "Dropped %d feature column(s) with excessive missing data: %s",
                len(dropped_cols),
                ", ".join(dropped_cols),
            )

        out_dir = feat_path.parent
        display_names = _apply_display_names(
            labels,
            user_goal=user_goal,
            label_name_map=label_name_map,
            base_analysis_dir=out_dir,
        )
        n_samples = X_filtered.shape[0]
        k = n_clusters or _parse_n_clusters_from_text(user_goal) or _default_n_clusters(n_samples)
        k = max(1, min(k, n_samples))

        X, impute_stats = _impute_column_means(X_filtered)
        X, applied_weights = _apply_feature_weights(X, feature_cols, feature_weights)
        coords, pc1_var, pc2_var = _pca_2d(X)

        linkage_matrix = None
        consensus_meta: Optional[Dict[str, Any]] = None
        coassociation: Optional[np.ndarray] = None
        if method_key == "hierarchical":
            cluster_ids, linkage_matrix = _cluster_hierarchical(
                X, k, linkage_method=linkage_method
            )
        elif method_key == "consensus_hierarchical":
            cluster_ids, linkage_matrix, coassociation, consensus_meta = (
                _cluster_consensus_hierarchical(
                    X,
                    k,
                    n_bootstrap=200,
                    seed=42,
                    linkage_method=linkage_method,
                )
            )
        else:
            cluster_ids = _cluster_kmeans(X, k)

        out_dir = feat_path.parent
        assign_path = out_dir / assignments_file
        with open(assign_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["label", "display_name", "sim_directory", "cluster_id", "method"],
            )
            writer.writeheader()
            for i, label in enumerate(labels):
                writer.writerow(
                    {
                        "label": label,
                        "display_name": display_names[i],
                        "sim_directory": sim_dirs[i],
                        "cluster_id": int(cluster_ids[i]),
                        "method": method_key,
                    }
                )

        scatter_path = out_dir / scatter_plot_file
        if method_key == "consensus_hierarchical":
            method_title = "Consensus hierarchical"
        elif method_key == "hierarchical":
            method_title = "Hierarchical"
        else:
            method_title = "K-means"
        scale_phrase = (
            f" — {feature_scale_label} features" if feature_scale_label else ""
        )
        if HAS_MPL:
            _plot_cluster_scatter(
                coords,
                display_names,
                cluster_ids,
                scatter_path,
                title=f"{method_title} clustering (k={k}){scale_phrase}",
                pc1_var=pc1_var,
                pc2_var=pc2_var,
            )
        else:
            scatter_path = None

        dendro_path = None
        phylo_path = None
        heatmap_path = None
        panel_path = None
        hier_like = method_key in ("hierarchical", "consensus_hierarchical")
        if hier_like and linkage_matrix is not None and HAS_MPL:
            leaf_order: Optional[List[int]] = None
            dendro_title = (
                "A. Consensus hierarchical clustering (1 − co-association)"
                if method_key == "consensus_hierarchical"
                else "A. Hierarchical clustering dendrogram"
            )
            if panel_file:
                panel_path = out_dir / panel_file
                leaf_order = _plot_dendrogram_heatmap_panel(
                    linkage_matrix,
                    X,
                    feature_cols,
                    display_names,
                    cluster_ids,
                    panel_path,
                    dendrogram_title=dendro_title,
                    heatmap_title="B. Classification features by simulation",
                    colorbar_label=feature_scale_label,
                    archetype_names=cluster_archetype_names,
                    legend_ncol=legend_ncol if legend_ncol is not None else (
                        3 if k >= 5 else None
                    ),
                    highlight_display_names=highlight_display_names,
                    colorbar_symmetric=colorbar_symmetric,
                    simple_panel=simple_panel,
                )
            else:
                dendro_path = out_dir / dendrogram_file
                leaf_order = _plot_dendrogram(
                    linkage_matrix,
                    display_names,
                    dendro_path,
                    title=(
                        f"Consensus hierarchical dendrogram (k={k})"
                        if method_key == "consensus_hierarchical"
                        else f"Hierarchical clustering dendrogram (k={k})"
                    ),
                    cluster_ids=cluster_ids,
                    n_clusters=k,
                    highlight_display_names=highlight_display_names,
                )
                if leaf_order:
                    heatmap_path = out_dir / heatmap_file
                    _plot_cluster_feature_heatmap(
                        X,
                        feature_cols,
                        display_names,
                        cluster_ids,
                        leaf_order,
                        heatmap_path,
                        title=f"Classification features by simulation (k={k})",
                        colorbar_label=feature_scale_label,
                    )
            phylo_path = out_dir / phylo_tree_file
            _plot_unrooted_phylo_tree(
                linkage_matrix,
                display_names,
                cluster_ids,
                phylo_path,
                title=f"Unrooted similarity tree (k={k}){scale_phrase}",
            )

        cluster_representatives = _cluster_representatives(
            X, labels, display_names, cluster_ids
        )
        rep_overrides = _load_representative_overrides(out_dir)
        if rep_overrides:
            cluster_representatives = _apply_representative_overrides(
                cluster_representatives,
                rep_overrides,
                labels,
                display_names,
                cluster_ids,
            )
        if method_key == "consensus_hierarchical":
            repro_note = (
                "Consensus hierarchical clustering: leave-one-out + bootstrap Ward "
                "co-association on the scaled feature matrix, then Ward linkage of "
                "distance 1 − co-association (seed=42, n_bootstrap=200). Re-runs "
                "differ when the feature table changes."
            )
        else:
            repro_note = (
                "Hierarchical Ward clustering is deterministic for a fixed z-score "
                "matrix. Re-runs differ when the feature table changes (metric "
                "groups, missing per-sim outputs, or re-computed analysis values)."
            )
        if applied_weights:
            repro_note += (
                " Feature weights scale columns by sqrt(weight) before linkage."
            )
        summary = {
            "method": method_key,
            "n_clusters": k,
            "n_simulations": n_samples,
            "feature_columns": feature_cols,
            "feature_weights": applied_weights or None,
            "dropped_feature_columns": dropped_cols,
            "imputation": impute_stats,
            "row_order": labels,
            "reproducibility_note": repro_note,
            "consensus": consensus_meta,
            "features_file": str(feat_path),
            "assignments_file": str(assign_path),
            "scatter_plot": str(scatter_path) if scatter_path else None,
            "dendrogram_plot": str(dendro_path) if dendro_path else None,
            "phylo_tree_plot": str(phylo_path) if phylo_path else None,
            "heatmap_plot": str(heatmap_path) if heatmap_path else None,
            "panel_plot": str(panel_path) if panel_path else None,
            "linkage_method": linkage_method if hier_like else None,
            "labels": labels,
            "display_names": display_names,
            "cluster_assignments": {
                labels[i]: int(cluster_ids[i]) for i in range(len(labels))
            },
            "cluster_representatives": cluster_representatives,
            "cluster_archetype_names": (
                {str(k): v for k, v in sorted((cluster_archetype_names or {}).items())}
                or None
            ),
        }
        if coassociation is not None:
            co_path = out_dir / "classification_coassociation.csv"
            with open(co_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["label"] + labels)
                for i, lab in enumerate(labels):
                    writer.writerow(
                        [lab] + [f"{float(coassociation[i, j]):.6f}" for j in range(n_samples)]
                    )
            summary["coassociation_file"] = str(co_path)

        summary_path = out_dir / summary_file
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        plot_note = ""
        if not HAS_MPL:
            plot_note = " (matplotlib unavailable — plots skipped)"

        return {
            "success": True,
            "message": (
                f"{method_title} clustering: {n_samples} simulations → {k} clusters"
                f"{plot_note}"
            ),
            "method": method_key,
            "n_clusters": k,
            "assignments_file": str(assign_path),
            "scatter_plot": str(scatter_path) if scatter_path else None,
            "dendrogram_plot": str(dendro_path) if dendro_path else None,
            "phylo_tree_plot": str(phylo_path) if phylo_path else None,
            "heatmap_plot": str(heatmap_path) if heatmap_path else None,
            "panel_plot": str(panel_path) if panel_path else None,
            "summary_file": str(summary_path),
            "cluster_assignments": summary["cluster_assignments"],
            "cluster_representatives": cluster_representatives,
            "display_names": display_names,
            "feature_columns": feature_cols,
            "dropped_feature_columns": dropped_cols,
            "imputation": impute_stats,
        }
    except Exception as exc:
        logger.exception("cluster_classification_features failed")
        return {"success": False, "error": str(exc)}
    finally:
        if original_dir:
            os.chdir(original_dir)


def _resolve_two_stage_k(n_clusters: int, n_samples: int) -> Tuple[int, int]:
    """Split a target cluster count into pocket-stage k and FEL sub-stage k."""
    total = max(1, min(int(n_clusters), n_samples))
    if total <= 1:
        return 1, 1
    k_pocket = max(2, min(int(round(np.sqrt(total))), n_samples))
    k_fel = max(1, min(total // k_pocket, n_samples))
    return k_pocket, k_fel


def _split_feature_groups(
    feature_cols: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Partition z-score columns into reference pocket vs FEL archetype groups."""
    from src.analysis.classification_collector import CLASSIFICATION_FEATURE_GROUPS

    pocket_set = set(CLASSIFICATION_FEATURE_GROUPS.get("reference_pocket_archetype", ()))
    fel_set = set(CLASSIFICATION_FEATURE_GROUPS.get("reference_fel_archetype", ()))
    pocket_cols = [c for c in feature_cols if c in pocket_set]
    fel_cols = [c for c in feature_cols if c in fel_set]
    return pocket_cols, fel_cols


def _two_stage_cluster_assignments(
    X_pocket: np.ndarray,
    X_fel: np.ndarray,
    *,
    k_pocket: int,
    k_fel: int,
    linkage_method: str = "ward",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 1: pocket features → k_pocket groups.
    Stage 2: FEL features → k_fel sub-groups within each pocket group.

    Returns (final_cluster_ids, pocket_cluster_ids, fel_subcluster_ids, pocket_linkage).
    """
    n_samples = X_pocket.shape[0]
    k_pocket = max(1, min(int(k_pocket), n_samples))
    k_fel = max(1, int(k_fel))

    X_p, _ = _impute_column_means(X_pocket)
    pocket_ids, pocket_linkage = _cluster_hierarchical(
        X_p, k_pocket, linkage_method=linkage_method
    )

    final_ids = np.zeros(n_samples, dtype=int)
    fel_sub_ids = np.zeros(n_samples, dtype=int)
    combo_to_final: Dict[Tuple[int, int], int] = {}
    next_final = 1

    for pocket_cid in sorted({int(c) for c in pocket_ids}):
        group_idx = [i for i, c in enumerate(pocket_ids) if int(c) == pocket_cid]
        n_group = len(group_idx)
        if n_group == 1:
            local_fel = np.array([1], dtype=int)
        else:
            k_sub = min(k_fel, n_group)
            X_f, _ = _impute_column_means(X_fel[np.asarray(group_idx)])
            local_fel, _ = _cluster_hierarchical(
                X_f, k_sub, linkage_method=linkage_method
            )
        for local_i, global_i in enumerate(group_idx):
            fel_cid = int(local_fel[local_i])
            fel_sub_ids[global_i] = fel_cid
            key = (int(pocket_cid), fel_cid)
            if key not in combo_to_final:
                combo_to_final[key] = next_final
                next_final += 1
            final_ids[global_i] = combo_to_final[key]

    return final_ids, pocket_ids.astype(int), fel_sub_ids.astype(int), pocket_linkage


@tool
def cluster_reference_archetype_two_stage(
    working_dir: str = "",
    features_file: str = "ref_fel_pock_features_zscore.csv",
    n_clusters: Optional[int] = None,
    k_pocket: Optional[int] = None,
    k_fel: Optional[int] = None,
    linkage_method: str = "ward",
    user_goal: str = "",
    label_name_map: Optional[Dict[str, str]] = None,
    min_features_present: int = 1,
    max_column_missing_fraction: float = 0.25,
    assignments_file: str = "ref_fel_pock_cluster_assignments.csv",
    scatter_plot_file: str = "ref_fel_pock_clusters_pca.png",
    dendrogram_file: str = "ref_fel_pock_dendrogram.png",
    phylo_tree_file: str = "ref_fel_pock_phylo_tree.png",
    summary_file: str = "ref_fel_pock_clusters.json",
) -> Dict[str, Any]:
    """
    Two-stage reference archetype clustering: pocket coupling first, then FEL within groups.

    Stage 1 uses ``reference_pocket_archetype`` z-scores (COM/angle/bound).
    Stage 2 sub-clusters each pocket group on ``reference_fel_archetype`` features.
    Default k=4 → k_pocket=2 and k_fel=2 (up to four pocket×FEL combinations).
    """
    original_dir = os.getcwd()
    try:
        feat_path = Path(features_file)
        if not feat_path.is_absolute():
            base = Path(working_dir) if working_dir else Path(".")
            feat_path = (base / features_file).resolve()
        if not feat_path.is_file():
            return {"success": False, "error": f"Features file not found: {feat_path}"}

        labels, _, sim_dirs, X_raw, feature_cols = _load_feature_matrix(
            feat_path, min_features_present=min_features_present
        )
        X_filtered, feature_cols, dropped_cols = _filter_usable_feature_columns(
            X_raw,
            feature_cols,
            max_missing_fraction=max_column_missing_fraction,
        )
        pocket_cols, fel_cols = _split_feature_groups(feature_cols)
        if not pocket_cols:
            return {"success": False, "error": "No reference pocket archetype columns found"}
        if not fel_cols:
            return {"success": False, "error": "No reference FEL archetype columns found"}

        pocket_idx = [feature_cols.index(c) for c in pocket_cols]
        fel_idx = [feature_cols.index(c) for c in fel_cols]
        X_pocket = X_filtered[:, pocket_idx]
        X_fel = X_filtered[:, fel_idx]

        out_dir = feat_path.parent
        display_names = _apply_display_names(
            labels,
            user_goal=user_goal,
            label_name_map=label_name_map,
            base_analysis_dir=out_dir,
        )
        n_samples = X_filtered.shape[0]
        total_k = n_clusters or _parse_n_clusters_from_text(user_goal) or 4
        total_k = max(1, min(total_k, n_samples))
        if k_pocket is None or k_fel is None:
            resolved_pocket, resolved_fel = _resolve_two_stage_k(total_k, n_samples)
            k_pocket = k_pocket if k_pocket is not None else resolved_pocket
            k_fel = k_fel if k_fel is not None else resolved_fel

        cluster_ids, pocket_ids, fel_sub_ids, pocket_linkage = _two_stage_cluster_assignments(
            X_pocket,
            X_fel,
            k_pocket=int(k_pocket),
            k_fel=int(k_fel),
            linkage_method=linkage_method,
        )
        k_final = len({int(c) for c in cluster_ids})

        X_all, impute_stats = _impute_column_means(X_filtered)
        coords, pc1_var, pc2_var = _pca_2d(X_all)

        assign_path = out_dir / assignments_file
        with open(assign_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "label",
                    "display_name",
                    "sim_directory",
                    "cluster_id",
                    "pocket_cluster_id",
                    "fel_subcluster_id",
                    "method",
                ],
            )
            writer.writeheader()
            for i, label in enumerate(labels):
                writer.writerow(
                    {
                        "label": label,
                        "display_name": display_names[i],
                        "sim_directory": sim_dirs[i],
                        "cluster_id": int(cluster_ids[i]),
                        "pocket_cluster_id": int(pocket_ids[i]),
                        "fel_subcluster_id": int(fel_sub_ids[i]),
                        "method": "two_stage_hierarchical",
                    }
                )

        scatter_path = dendro_path = phylo_path = None
        if HAS_MPL:
            scatter_path = out_dir / scatter_plot_file
            _plot_cluster_scatter(
                coords,
                display_names,
                cluster_ids,
                scatter_path,
                title=f"Two-stage clustering (k={k_final}, pocket k={k_pocket}, FEL k={k_fel})",
                pc1_var=pc1_var,
                pc2_var=pc2_var,
            )
            dendro_path = out_dir / dendrogram_file
            _plot_dendrogram(
                pocket_linkage,
                display_names,
                dendro_path,
                title=f"Stage 1 pocket dendrogram (k={k_pocket}) — final k={k_final}",
                cluster_ids=cluster_ids,
                n_clusters=k_final,
            )
            phylo_path = out_dir / phylo_tree_file
            _plot_unrooted_phylo_tree(
                pocket_linkage,
                display_names,
                cluster_ids,
                phylo_path,
                title=f"Stage 1 pocket tree — final two-stage clusters (k={k_final})",
            )

        cluster_representatives = _cluster_representatives(
            X_all, labels, display_names, cluster_ids
        )
        rep_overrides = _load_representative_overrides(out_dir)
        if rep_overrides:
            cluster_representatives = _apply_representative_overrides(
                cluster_representatives,
                rep_overrides,
                labels,
                display_names,
                cluster_ids,
            )

        summary = {
            "method": "two_stage_hierarchical",
            "n_clusters": k_final,
            "n_clusters_target": total_k,
            "stage1_k_pocket": int(k_pocket),
            "stage2_k_fel": int(k_fel),
            "n_simulations": n_samples,
            "pocket_feature_columns": pocket_cols,
            "fel_feature_columns": fel_cols,
            "feature_columns": feature_cols,
            "feature_weights": None,
            "dropped_feature_columns": dropped_cols,
            "imputation": impute_stats,
            "row_order": labels,
            "reproducibility_note": (
                "Two-stage Ward clustering: stage 1 on pocket z-scores, stage 2 on "
                "FEL z-scores within each pocket group. Dendrogram/phylo reflect stage 1 only."
            ),
            "features_file": str(feat_path),
            "assignments_file": str(assign_path),
            "scatter_plot": str(scatter_path) if scatter_path else None,
            "dendrogram_plot": str(dendro_path) if dendro_path else None,
            "phylo_tree_plot": str(phylo_path) if phylo_path else None,
            "linkage_method": linkage_method,
            "labels": labels,
            "display_names": display_names,
            "pocket_cluster_assignments": {
                labels[i]: int(pocket_ids[i]) for i in range(len(labels))
            },
            "fel_subcluster_assignments": {
                labels[i]: int(fel_sub_ids[i]) for i in range(len(labels))
            },
            "cluster_assignments": {
                labels[i]: int(cluster_ids[i]) for i in range(len(labels))
            },
            "cluster_representatives": cluster_representatives,
        }
        summary_path = out_dir / summary_file
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        plot_note = "" if HAS_MPL else " (matplotlib unavailable — plots skipped)"
        return {
            "success": True,
            "message": (
                f"Two-stage clustering: {n_samples} simulations → {k_final} clusters "
                f"(pocket k={k_pocket}, FEL k={k_fel}){plot_note}"
            ),
            "method": "two_stage_hierarchical",
            "n_clusters": k_final,
            "stage1_k_pocket": int(k_pocket),
            "stage2_k_fel": int(k_fel),
            "assignments_file": str(assign_path),
            "scatter_plot": str(scatter_path) if scatter_path else None,
            "dendrogram_plot": str(dendro_path) if dendro_path else None,
            "phylo_tree_plot": str(phylo_path) if phylo_path else None,
            "summary_file": str(summary_path),
            "cluster_assignments": summary["cluster_assignments"],
            "cluster_representatives": cluster_representatives,
            "display_names": display_names,
            "feature_columns": feature_cols,
            "dropped_feature_columns": dropped_cols,
            "imputation": impute_stats,
        }
    except Exception as exc:
        logger.exception("cluster_reference_archetype_two_stage failed")
        return {"success": False, "error": str(exc)}
    finally:
        if original_dir:
            os.chdir(original_dir)


# Trajectory metrics that have per-frame time series (not per-residue scalars).
CLUSTER_TRAJECTORY_METRIC_GROUPS: Tuple[str, ...] = (
    "com",
    "contacts",
    "pocket_sasa",
    "residence",
)

CLUSTER_REFERENCE_POCKET_TRAJECTORY_METRIC_GROUPS: Tuple[str, ...] = (
    "reference_pocket_com",
    "reference_pocket_hbonds",
    "reference_pocket_sasa",
    "reference_pocket_residence",
    "reference_pocket_ligand_axis_angle",
)

_CLUSTER_TRAJECTORY_CONFIG: Dict[str, Dict[str, Any]] = {
    "com": {
        "title": "Ligand–Pocket COM Distance",
        "ylabel": "COM Distance (Å)",
        "output_file": "com_distance_by_cluster.png",
        "x_col": 1,
        "y_col": 2,
    },
    "contacts": {
        "title": "Protein–Ligand Contacts",
        "ylabel": "Heavy-Atom Contacts",
        "output_file": "contacts_by_cluster.png",
        "file_pattern": "protein_ligand_contacts",
        "x_col": 1,
        "y_col": 3,
    },
    "pocket_sasa": {
        "title": "Pocket SASA",
        "ylabel": "Pocket SASA (nm²)",
        "output_file": "pocket_sasa_by_cluster.png",
        "file_pattern": "pocket_sasa",
        "x_col": 0,
        "y_col": 1,
    },
    "residence": {
        "title": "Ligand Bound State",
        "ylabel": "Bound (0/1)",
        "output_file": "ligand_residence_by_cluster.png",
        "file_pattern": "ligand_residence",
        "x_col": 0,
        "y_col": 1,
    },
    "reference_pocket_com": {
        "title": "Reference Pocket — Ligand COM Distance",
        "ylabel": "COM Distance (Å)",
        "output_file": "ref_fel_pock_com_distance_by_cluster.png",
        "file_pattern": "reference_pocket_ligand_distance",
        "x_col": 1,
        "y_col": 2,
    },
    "reference_pocket_hbonds": {
        "title": "Reference Pocket — Protein–Ligand H-bonds",
        "ylabel": "H-bonds",
        "output_file": "ref_fel_pock_hbonds_by_cluster.png",
        "file_pattern": "reference_pocket_contacts",
        "x_col": 1,
        "y_col": 2,
    },
    "reference_pocket_sasa": {
        "title": "Reference Pocket SASA",
        "ylabel": "Pocket SASA (nm²)",
        "output_file": "ref_fel_pock_sasa_by_cluster.png",
        "file_pattern": "reference_pocket_sasa",
        "x_col": 0,
        "y_col": 1,
    },
    "reference_pocket_residence": {
        "title": "Reference Pocket — Ligand Bound State",
        "ylabel": "Bound (0/1)",
        "output_file": "ref_fel_pock_ligand_residence_by_cluster.png",
        "file_pattern": "reference_pocket_residence",
        "x_col": 0,
        "y_col": 1,
    },
    "reference_pocket_ligand_axis_angle": {
        "title": "Reference Pocket — Ligand Major-Axis Angle",
        "ylabel": "Pocket–ligand axis angle (°)",
        "output_file": "ref_fel_pock_ligand_axis_angle_by_cluster.png",
        "file_pattern": "reference_pocket_ligand_orientation",
        "x_col": 1,
        "y_col": 3,
    },
}


def _load_cluster_assignments(
    assignments_path: Path,
) -> Tuple[Dict[int, List[Dict[str, str]]], List[int]]:
    """Group assignment rows by cluster_id."""
    from src.analysis.reference_labels import load_reference_pca_manifest

    sim_dirs_by_label: Dict[str, str] = {}
    manifest_path = (
        assignments_path.parent / "reference_fel" / "reference_pca_manifest.json"
    )
    if manifest_path.is_file():
        try:
            manifest = load_reference_pca_manifest(assignments_path.parent)
            sim_dirs_by_label = {
                str(label).lower(): str(sim_dir)
                for label, sim_dir in (
                    manifest.get("sim_directories") or {}
                ).items()
                if sim_dir
            }
            for uid, sim_dir in sim_dirs_by_label.items():
                sim_dirs_by_label.setdefault(
                    Path(sim_dir).name.lower(), sim_dir
                )
        except Exception:
            logger.debug(
                "Could not load simulation directories from %s",
                manifest_path,
                exc_info=True,
            )

    by_cluster: Dict[int, List[Dict[str, str]]] = {}
    with open(assignments_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = int(row["cluster_id"])
            label = row.get("label", "")
            display_name = row.get("display_name") or label
            if display_name.lower() == label.lower():
                from src.analysis.reference_labels import build_uniprot_display_map

                disp_map = build_uniprot_display_map(assignments_path.parent)
                display_name = disp_map.get(label.lower(), display_name)
            sim_directory = row.get("sim_directory", "")
            if not sim_directory:
                sim_directory = (
                    sim_dirs_by_label.get(label.lower())
                    or sim_dirs_by_label.get(display_name.lower())
                    or ""
                )
            by_cluster.setdefault(cid, []).append(
                {
                    "label": label,
                    "display_name": display_name,
                    "sim_directory": sim_directory,
                }
            )
    cluster_ids = sorted(by_cluster)
    return by_cluster, cluster_ids


def _find_trajectory_file(sim_directory: str, metric_group: str) -> Optional[str]:
    """Locate a time-series CSV/dat for one simulation and metric group."""
    from src.analysis.combined_analysis import (
        _find_com_distance_file,
        _find_metric_file,
    )

    sim_path = Path(sim_directory)
    if metric_group.startswith("reference_pocket"):
        from src.analysis.reference_labels import is_reference_pocket_usable

        manifest = (
            sim_path.parent / "analysis" / "reference_pocket_batch_manifest.json"
        )
        if manifest.is_file():
            try:
                with open(manifest, encoding="utf-8") as fh:
                    pocket_manifest = json.load(fh)
                if not is_reference_pocket_usable(
                    sim_path.name,
                    pocket_manifest,
                    base_analysis=sim_path.parent / "analysis",
                ):
                    return None
            except Exception:
                pass
    search_roots = [
        *[p for p in sorted((sim_path / "analysis").glob("rep*")) if p.is_dir()],
        sim_path / "analysis",
        sim_path.parent / "analysis" / "reference_pocket" / sim_path.name,
        sim_path,
    ]
    # Prefer real per-rep series; avg/ headerless dumps are last-resort only.
    if metric_group == "com":
        for root in search_roots:
            if root.is_dir():
                hit = _find_com_distance_file(str(root))
                if hit:
                    return hit
        return None

    cfg = _CLUSTER_TRAJECTORY_CONFIG.get(metric_group, {})
    pattern = cfg.get("file_pattern", metric_group)
    for root in search_roots:
        if root.is_dir():
            hit = _find_metric_file(str(root), pattern)
            if hit:
                return hit
    return None


def _plot_metric_by_cluster_panels(
    by_cluster: Dict[int, List[Dict[str, str]]],
    cluster_ids: Sequence[int],
    metric_group: str,
    output_path: Path,
    n_clusters_total: int,
    figsize_per_panel: Tuple[float, float] = (10.0, 3.0),
    dpi: int = 150,
) -> bool:
    """Draw one subplot per cluster with trajectories for sims in that cluster."""
    if not HAS_MPL:
        return False

    from src.analysis.combined_analysis import _read_two_column_file

    cfg = _CLUSTER_TRAJECTORY_CONFIG[metric_group]
    x_col = int(cfg["x_col"])
    y_col = int(cfg["y_col"])

    all_names = sorted(
        {
            row["display_name"]
            for members in by_cluster.values()
            for row in members
        }
    )
    cmap = plt.get_cmap("tab10", max(len(all_names), 1))
    name_colors = {name: cmap(i) for i, name in enumerate(all_names)}

    n_panels = len(cluster_ids)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * n_panels),
        squeeze=False,
    )

    any_data = False
    for idx, cid in enumerate(cluster_ids):
        ax = axes[idx, 0]
        members = by_cluster.get(cid, [])
        plotted = 0
        for row in members:
            sim_dir = row.get("sim_directory") or ""
            display_name = row.get("display_name") or row.get("label", "")
            if not sim_dir:
                continue
            fpath = _find_trajectory_file(sim_dir, metric_group)
            if not fpath:
                logger.warning(
                    "No trajectory file for %s (%s)", display_name, metric_group
                )
                continue
            try:
                xs, ys = _read_two_column_file(fpath, x_col=x_col, y_col=y_col)
                color = name_colors.get(display_name, "#333333")
                ax.plot(xs, ys, label=display_name, color=color, linewidth=1.1, alpha=0.88)
                plotted += 1
                any_data = True
            except Exception as exc:
                logger.warning(
                    "Skipping %s trajectory (%s): %s", display_name, fpath, exc
                )

        ax.set_ylabel(cfg["ylabel"], fontsize=9)
        ax.set_title(
            f"Cluster {cid} (n={len(members)}) — {cfg['title']}",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(loc="best", fontsize=8, ncol=min(plotted, 3))
        else:
            ax.text(
                0.5,
                0.5,
                "No trajectory data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )

    axes[-1, 0].set_xlabel("Time (ns)", fontsize=10)
    fig.suptitle(
        f"{cfg['title']} by cluster (k={n_clusters_total})",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    if not any_data:
        plt.close(fig)
        return False
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


@tool
def plot_cluster_feature_trajectories(
    working_dir: str = "",
    assignments_file: str = "classification_cluster_assignments.csv",
    metric_groups: Optional[List[str]] = None,
    figsize_per_panel: Tuple[float, float] = (10.0, 3.0),
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Plot major classification feature trajectories with one subplot per cluster.

    After ``cluster_classification_features`` assigns simulations to k groups,
    this tool reads per-simulation time series (pocket SASA, COM distance,
    contacts, ligand residence) and writes one PNG per metric. Each PNG has
    k panels — one per cluster — with all proteins in that cluster overlaid.

    Per-residue metrics (pocket RMSF, ligand RMSF, FEL scalars) are skipped
    because they are not time series.

    Args:
        working_dir: Directory containing ``classification_cluster_assignments.csv``.
        assignments_file: Cluster assignment CSV from clustering step.
        metric_groups: Subset of ``com``, ``contacts``, ``pocket_sasa``,
            ``residence``. Defaults to all trajectory-capable groups.
        figsize_per_panel: Width and height (inches) for each cluster panel.
        dpi: Output image resolution.

    Returns:
        Dict with ``success``, ``plots`` (list of PNG paths), and ``metrics_plotted``.
    """
    try:
        assign_path = Path(assignments_file)
        if not assign_path.is_absolute():
            base = Path(working_dir) if working_dir else Path(".")
            assign_path = (base / assignments_file).resolve()
        if not assign_path.is_file():
            return {
                "success": False,
                "error": f"Assignments file not found: {assign_path}",
            }

        groups = tuple(
            g
            for g in (metric_groups or CLUSTER_TRAJECTORY_METRIC_GROUPS)
            if g in _CLUSTER_TRAJECTORY_CONFIG
        )
        if not groups:
            return {
                "success": False,
                "error": (
                    "No trajectory metric groups requested. "
                    f"Choose from: {', '.join(CLUSTER_TRAJECTORY_METRIC_GROUPS)}"
                ),
            }

        by_cluster, cluster_ids = _load_cluster_assignments(assign_path)
        if not cluster_ids:
            return {"success": False, "error": "No cluster assignments found"}

        out_dir = assign_path.parent
        plots: List[str] = []
        skipped: List[str] = []

        for metric_group in groups:
            cfg = _CLUSTER_TRAJECTORY_CONFIG[metric_group]
            output_path = out_dir / cfg["output_file"]
            ok = _plot_metric_by_cluster_panels(
                by_cluster,
                cluster_ids,
                metric_group,
                output_path,
                n_clusters_total=len(cluster_ids),
                figsize_per_panel=figsize_per_panel,
                dpi=dpi,
            )
            if ok:
                plots.append(str(output_path))
            else:
                skipped.append(metric_group)

        if not plots:
            return {
                "success": False,
                "error": "No trajectory plots generated (missing data or matplotlib)",
                "skipped_metrics": skipped,
            }

        return {
            "success": True,
            "message": (
                f"Cluster trajectory plots: {len(plots)} metric(s), "
                f"k={len(cluster_ids)} clusters"
            ),
            "plots": plots,
            "metrics_plotted": [
                g for g in groups if g not in skipped
            ],
            "skipped_metrics": skipped,
            "n_clusters": len(cluster_ids),
            "assignments_file": str(assign_path),
        }
    except Exception as exc:
        logger.exception("plot_cluster_feature_trajectories failed")
        return {"success": False, "error": str(exc)}


CLUSTER_RMSF_PROFILE_GROUPS: Tuple[str, ...] = (
    "pocket_rmsf",
    "ligand_rmsf",
    "reference_pocket_rmsf",
)

_CLUSTER_RMSF_CONFIG: Dict[str, Dict[str, str]] = {
    "pocket_rmsf": {
        "file_pattern": "pocket_rmsf",
        "output_file": "pocket_rmsf_by_cluster.png",
        "title": "Pocket RMSF",
        "xlabel": "Pocket residue index",
        "ylabel": "RMSF (Å)",
    },
    "ligand_rmsf": {
        "file_pattern": "ligand_rmsf",
        "output_file": "ligand_rmsf_by_cluster.png",
        "title": "Ligand RMSF",
        "xlabel": "Ligand atom index",
        "ylabel": "RMSF (Å)",
    },
    "reference_pocket_rmsf": {
        "file_pattern": "reference_pocket_rmsf",
        "output_file": "ref_fel_pock_rmsf_by_cluster.png",
        "title": "Reference Pocket RMSF",
        "xlabel": "Reference-aligned pocket index",
        "ylabel": "RMSF (Å)",
        "align_consensus_index": True,
    },
}


def _find_rmsf_profile_file(sim_directory: str, profile_type: str) -> Optional[str]:
    if profile_type == "reference_pocket_rmsf":
        from src.analysis.reference_labels import is_reference_pocket_usable

        sim_path = Path(sim_directory)
        manifest = (
            sim_path.parent / "analysis" / "reference_pocket_batch_manifest.json"
        )
        if manifest.is_file():
            try:
                with open(manifest, encoding="utf-8") as fh:
                    pocket_manifest = json.load(fh)
                if not is_reference_pocket_usable(
                    sim_path.name,
                    pocket_manifest,
                    base_analysis=sim_path.parent / "analysis",
                ):
                    return None
            except Exception:
                pass
        candidate = (
            sim_path.parent
            / "analysis"
            / "reference_pocket"
            / sim_path.name
            / "reference_pocket_rmsf.dat"
        )
        if candidate.is_file():
            return str(candidate)
    from src.analysis.combined_analysis import _find_binding_rmsf_file
    return _find_binding_rmsf_file(sim_directory, profile_type)


def _plot_rmsf_profiles_by_cluster_panels(
    by_cluster: Dict[int, List[Dict[str, str]]],
    cluster_ids: Sequence[int],
    profile_type: str,
    output_path: Path,
    n_clusters_total: int,
    figsize_per_panel: Tuple[float, float] = (10.0, 3.5),
    dpi: int = 150,
    consensus_residue_map: Optional[str] = None,
) -> bool:
    """One subplot per cluster with pocket/ligand RMSF profiles overlaid."""
    if not HAS_MPL:
        return False

    from src.analysis.data_plotter import (
        _parse_reference_pocket_rmsf_aligned,
        _parse_rmsf_profile_dat,
    )

    cfg = _CLUSTER_RMSF_CONFIG[profile_type]
    align_consensus = bool(cfg.get("align_consensus_index"))
    residue_map = consensus_residue_map
    if align_consensus and not residue_map:
        candidate = output_path.parent / "reference_pocket_residue_map.csv"
        if candidate.is_file():
            residue_map = str(candidate)

    n_panels = len(cluster_ids)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * n_panels),
        squeeze=False,
    )

    default_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    any_data = False

    for idx, cid in enumerate(cluster_ids):
        ax = axes[idx, 0]
        members = by_cluster.get(cid, [])
        plotted = 0
        for mi, row in enumerate(members):
            sim_dir = row.get("sim_directory") or ""
            display_name = row.get("display_name") or row.get("label", "")
            if not sim_dir:
                continue
            fpath = _find_rmsf_profile_file(sim_dir, profile_type)
            if not fpath:
                continue
            if align_consensus and residue_map:
                profile = _parse_reference_pocket_rmsf_aligned(
                    fpath, residue_map, display_name
                )
            else:
                profile = _parse_rmsf_profile_dat(fpath)
            if not profile:
                continue
            color = default_colors[mi % len(default_colors)]
            ax.plot(
                profile["x_positions"],
                profile["y_values"],
                label=display_name,
                color=color,
                linewidth=1.2,
                alpha=0.88,
            )
            plotted += 1
            any_data = True

        ax.set_ylabel(cfg["ylabel"], fontsize=9)
        ax.set_title(
            f"Cluster {cid} (n={len(members)}) — {cfg['title']}",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(loc="best", fontsize=8, ncol=min(plotted, 3))
        else:
            ax.text(
                0.5, 0.5, "No RMSF data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#666666",
            )

    axes[-1, 0].set_xlabel(cfg["xlabel"], fontsize=10)
    fig.suptitle(
        f"{cfg['title']} by cluster (k={n_clusters_total})",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    if any_data:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return any_data


@tool
def plot_cluster_rmsf_profiles(
    working_dir: str = "",
    assignments_file: str = "classification_cluster_assignments.csv",
    profile_types: Optional[List[str]] = None,
    figsize_per_panel: Tuple[float, float] = (10.0, 3.5),
    dpi: int = 150,
    consensus_residue_map: str = "reference_pocket_residue_map.csv",
) -> Dict[str, Any]:
    """
    Plot pocket and/or ligand RMSF profiles with one subplot per cluster.

    After clustering, overlays each protein's pocket/ligand RMSF curve within
    its assigned cluster panel (ordinal x-axis; legend uses display names).

    Args:
        working_dir: Directory with ``classification_cluster_assignments.csv``.
        assignments_file: Cluster assignment CSV.
        profile_types: ``pocket_rmsf``, ``ligand_rmsf``, or both (default).
        figsize_per_panel: Panel size in inches.
        dpi: Output resolution.

    Standard outputs: ``pocket_rmsf_by_cluster.png``, ``ligand_rmsf_by_cluster.png``.
    """
    try:
        assign_path = Path(assignments_file)
        if not assign_path.is_absolute():
            base = Path(working_dir) if working_dir else Path(".")
            assign_path = (base / assignments_file).resolve()
        if not assign_path.is_file():
            return {
                "success": False,
                "error": f"Assignments file not found: {assign_path}",
            }

        types = tuple(
            t for t in (profile_types or CLUSTER_RMSF_PROFILE_GROUPS)
            if t in _CLUSTER_RMSF_CONFIG
        )
        if not types:
            return {
                "success": False,
                "error": (
                    "No RMSF profile types requested. "
                    f"Choose from: {', '.join(CLUSTER_RMSF_PROFILE_GROUPS)}"
                ),
            }

        by_cluster, cluster_ids = _load_cluster_assignments(assign_path)
        if not cluster_ids:
            return {"success": False, "error": "No cluster assignments found"}

        out_dir = assign_path.parent
        residue_map_path = Path(consensus_residue_map)
        if not residue_map_path.is_absolute():
            residue_map_path = (out_dir / consensus_residue_map).resolve()
        plots: List[str] = []
        skipped: List[str] = []

        for profile_type in types:
            cfg = _CLUSTER_RMSF_CONFIG[profile_type]
            output_path = out_dir / cfg["output_file"]
            ok = _plot_rmsf_profiles_by_cluster_panels(
                by_cluster,
                cluster_ids,
                profile_type,
                output_path,
                n_clusters_total=len(cluster_ids),
                figsize_per_panel=figsize_per_panel,
                dpi=dpi,
                consensus_residue_map=str(residue_map_path)
                if residue_map_path.is_file()
                else None,
            )
            if ok:
                plots.append(str(output_path))
            else:
                skipped.append(profile_type)

        if not plots:
            return {
                "success": False,
                "error": "No cluster RMSF plots generated",
                "skipped_profiles": skipped,
            }

        return {
            "success": True,
            "message": (
                f"Cluster RMSF plots: {len(plots)} profile(s), k={len(cluster_ids)} clusters"
            ),
            "plots": plots,
            "profiles_plotted": [t for t in types if t not in skipped],
            "skipped_profiles": skipped,
            "n_clusters": len(cluster_ids),
            "assignments_file": str(assign_path),
        }
    except Exception as exc:
        logger.exception("plot_cluster_rmsf_profiles failed")
        return {"success": False, "error": str(exc)}

