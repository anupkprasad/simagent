"""
Combined Analysis Tools - Cross-simulation comparison utilities

Provides tools to collect metric data from multiple simulation directories,
produce overlay plots, and compute summary statistics tables.  Designed
for the "combined_analysis" phase of multi-simulation workflows.

Each function follows the same @tool convention used by the other modules
in src/analysis/ (returns a Dict[str, Any] with at least a "success" key).
"""
import os
import re
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from langchain.tools import tool

logger = logging.getLogger(__name__)

# Optional heavy deps
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available — overlay plots will be skipped")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_EXTS = {".xvg", ".dat", ".csv", ".txt", ".xmgr", ".edr", ".xlsx"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".webp"}


def _find_metric_file(directory: str, filename_pattern: str) -> Optional[str]:
    """
    Search *directory* recursively for the best file matching *filename_pattern*
    (case-insensitive).

    *filename_pattern* may be a basename (``rmsd.dat``) or a stem (``rmsd``).
    Prefers an exact basename match, then an exact stem match, then stem prefix.
    Among those, prefers numeric data files (``.xvg``, ``.dat``, …) over plot
    images (``.png``) so ``energy.xvg`` wins over ``energy.png`` for overlays.
    """
    d = Path(directory)
    if not d.is_dir():
        return None
    pat_name = filename_pattern.lower()
    pat_stem = Path(filename_pattern).stem.lower()
    matches: List[Path] = []
    for p in d.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        stem = p.stem.lower()
        if name == pat_name or stem == pat_stem or stem.startswith(pat_stem):
            matches.append(p)
    if not matches:
        return None

    def _priority(p: Path) -> Tuple[int, int, str]:
        name = p.name.lower()
        stem = p.stem.lower()
        if name == pat_name:
            name_tier = 0
        elif stem == pat_stem:
            name_tier = 1
        else:
            name_tier = 2
        ext = p.suffix.lower()
        if ext in _DATA_EXTS:
            data_tier = 0
        elif ext in _IMAGE_EXTS:
            data_tier = 2
        else:
            data_tier = 1
        return (name_tier, data_tier, str(p))

    matches.sort(key=_priority)
    return str(matches[0])


def _read_two_column_file(
    filepath: str,
    y_col: int = 1,
    x_col: int = 0,
) -> Tuple[List[float], List[float]]:
    """
    Read a whitespace-or-comma-separated data file.

    Lines starting with '#' or '@' are treated as comments / metadata.
    Returns (x_values, y_values) as lists of floats.

    Time-unit auto-conversion
    -------------------------
    If the comment header (lines starting with ``#``) mentions ``"(ps)"``
    or ``"label \"Time (ps)\""`` (GROMACS xvg style), the first column
    is divided by 1000 to convert ps → ns so all combined plots share the
    same x-axis unit.

    y_col: index of the column to use as y (default 1 = second column).
           Energy xvg files have columns [time, Potential, Kinetic, Temp];
           use y_col=1 to get Potential.

    Raises ValueError when the file cannot be parsed.
    """
    xs, ys = [], []
    convert_x_to_ns = False  # will be set True when header says ps

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue

            # ── Header / metadata lines ───────────────────────────────
            if raw.startswith("#") or raw.startswith("@"):
                low = raw.lower()
                # Detect ps time axis from comment or xvg @xaxis label
                if "(ps)" in low or 'label "time (ps)"' in low or "time(ps)" in low:
                    convert_x_to_ns = True
                continue

            # ── Data lines ────────────────────────────────────────────
            parts = raw.replace(",", " ").split()
            if len(parts) > max(x_col, y_col):
                try:
                    xs.append(float(parts[x_col]))
                    ys.append(float(parts[y_col]))
                except ValueError:
                    continue

    if not xs:
        raise ValueError(f"No numeric data found in {filepath}")

    if convert_x_to_ns:
        xs = [x / 1000.0 for x in xs]

    return xs, ys


def _float_stats(values: List[float]) -> Dict[str, float]:
    """Return mean/std/min/max rounded to 3 d.p."""
    if not values:
        return {}
    if HAS_NUMPY:
        arr = np.array(values)
        return {
            "mean": round(float(arr.mean()), 3),
            "std":  round(float(arr.std()),  3),
            "min":  round(float(arr.min()),  3),
            "max":  round(float(arr.max()),  3),
        }
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {
        "mean": round(mean, 3),
        "std":  round(variance ** 0.5, 3),
        "min":  round(min(values), 3),
        "max":  round(max(values), 3),
    }


# ---------------------------------------------------------------------------
# Public @tool functions
# ---------------------------------------------------------------------------

@tool
def collect_metric_files(
    sim_dirs: List[str],
    metric_filename: str,
    search_subdir: str = "analysis",
) -> Dict[str, Any]:
    """
    Find a metric data file in each simulation directory.

    Searches ``{sim_dir}/{search_subdir}/`` (and all sub-folders) for the
    first file whose name contains *metric_filename* (e.g. ``"rmsd.dat"``).

    Args:
        sim_dirs: List of per-simulation working directories to search.
        metric_filename: Filename pattern to look for, e.g. ``"rmsd.dat"``.
        search_subdir: Sub-directory name to search inside each sim_dir
            (default: ``"analysis"``).

    Returns:
        Dict with keys:

        * ``found`` – list of ``{"sim_dir": ..., "file": ...}`` dicts
        * ``missing`` – list of sim_dirs where the file was not found
        * ``success`` – True when every directory yielded a match
    """
    found: List[Dict[str, str]] = []
    missing: List[str] = []

    for sim_dir in sim_dirs:
        hit = None
        try:
            from src.analysis.replicate_paths import (
                analysis_avg_dir,
                parse_rep_id,
                preferred_analysis_metric_dirs,
            )

            avg = analysis_avg_dir(sim_dir)
            if avg.is_dir():
                hit = _find_metric_file(str(avg), metric_filename)
            if not hit:
                flat = Path(sim_dir) / search_subdir if search_subdir else Path(sim_dir)
                # Flat analysis/ only (not nested repXX)
                if flat.is_dir() and parse_rep_id(flat.name) is None:
                    hit = _find_metric_file(str(flat), metric_filename)
                    if hit and parse_rep_id(Path(hit).parent.name) is not None:
                        # Recursive search drifted into repXX — ignore for now
                        hit = None
                        for child in flat.iterdir():
                            if child.is_file() and metric_filename in child.name:
                                hit = str(child)
                                break
            if not hit:
                for d in preferred_analysis_metric_dirs(sim_dir):
                    if parse_rep_id(d.name) is None:
                        continue
                    hit = _find_metric_file(str(d), metric_filename)
                    if hit:
                        break
        except Exception:
            search_root = Path(sim_dir) / search_subdir if search_subdir else Path(sim_dir)
            hit = _find_metric_file(str(search_root), metric_filename)

        if not hit:
            hit = _find_metric_file(sim_dir, metric_filename)

        if hit:
            found.append({"sim_dir": sim_dir, "file": hit})
        else:
            missing.append(sim_dir)

    return {
        "success": len(missing) == 0,
        "found": found,
        "missing": missing,
        "n_found": len(found),
        "n_missing": len(missing),
    }


@tool
def plot_combined_overlay(
    data_files: List[str],
    labels: List[str],
    output_file: str,
    working_dir: str,
    title: str = "",
    xlabel: str = "Time (ns)",
    ylabel: str = "Value",
    x_col: int = 0,
    y_col: int = 1,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Overlay multiple 2-column data files on a single plot.

    All files must be parseable by ``_read_two_column_file`` (whitespace/CSV,
    ``#``-commented headers).  One line is drawn per file, coloured and labelled
    according to *labels*.

    Args:
        data_files: Ordered list of data file paths (one per simulation).
        labels: Human-readable legend labels, one per file.
        output_file: Output image filename (e.g. ``"rmsd_overlay.png"``).
            Saved inside *working_dir*.
        working_dir: Directory where the image is written.
        title: Plot title (optional).
        xlabel: X-axis label (default ``"Time (ns)"``).
        ylabel: Y-axis label (default ``"Value"``).
        colors: Optional list of Matplotlib colour strings.
        figsize: Figure size tuple (width, height) in inches.
        dpi: Image resolution.

    Returns:
        Dict with ``success``, ``output_path``, and per-file ``stats``.
    """
    if not HAS_MATPLOTLIB:
        return {"success": False, "error": "matplotlib not available"}
    if len(data_files) != len(labels):
        return {"success": False, "error": "data_files and labels must have the same length"}

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(working_dir) / output_file)

    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                      "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    if colors is None:
        colors = [default_colors[i % len(default_colors)] for i in range(len(data_files))]

    fig, ax = plt.subplots(figsize=figsize)
    per_file_stats: Dict[str, Dict] = {}
    errors: List[str] = []

    for fpath, label, color in zip(data_files, labels, colors):
        try:
            xs, ys = _read_two_column_file(fpath, x_col=x_col, y_col=y_col)
            ax.plot(xs, ys, label=label, color=color, linewidth=1.2, alpha=0.85)
            per_file_stats[label] = _float_stats(ys)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            logger.warning(f"plot_combined_overlay: skipping {fpath}: {exc}")

    if not per_file_stats:
        plt.close(fig)
        return {
            "success": False,
            "error": "No plottable data in any input file",
            "errors": errors,
        }

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved overlay plot → {output_path}")
    return {
        "success": True,
        "output_path": output_path,
        "output_file": output_file,
        "stats": per_file_stats,
        "errors": errors or None,
    }


@tool
def compute_comparison_table(
    data_files: List[str],
    labels: List[str],
    output_csv: str,
    working_dir: str,
) -> Dict[str, Any]:
    """
    Compute mean/std/min/max for each simulation's metric file and write a CSV.

    Args:
        data_files: Ordered list of data file paths (column 1 = time/x, column 2 = metric).
        labels: Row labels (one per file), e.g. ``["1A", "2B", "3C"]``.
        output_csv: Output CSV filename (saved in *working_dir*).
        working_dir: Directory where the CSV is written.

    Returns:
        Dict with ``success``, ``output_path``, and ``table`` (list-of-dicts).
    """
    if len(data_files) != len(labels):
        return {"success": False, "error": "data_files and labels must have the same length"}

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(working_dir) / output_csv)

    rows: List[Dict[str, Any]] = []
    for fpath, label in zip(data_files, labels):
        row: Dict[str, Any] = {"simulation": label}
        try:
            _, ys = _read_two_column_file(fpath)
            row.update(_float_stats(ys))
        except Exception as exc:
            row.update({"mean": "N/A", "std": "N/A", "min": "N/A", "max": "N/A", "error": str(exc)})
            logger.warning(f"compute_comparison_table: cannot read {fpath}: {exc}")
        rows.append(row)

    if rows:
        fieldnames: List[str] = []
        seen: set = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    logger.info(f"Saved comparison table → {output_path}")
    return {
        "success": True,
        "output_path": output_path,
        "output_file": output_csv,
        "table": rows,
    }


@tool
def run_combined_analysis(
    sim_dirs: List[str],
    labels: List[str],
    working_dir: str,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    High-level convenience tool: collect metric files from all simulation
    directories, produce overlay plots and statistics CSV files for each metric.

    For every metric the tool will:
    1. Search each sim's ``analysis/`` sub-directory for a matching file.
    2. Create an overlay plot saved to ``{working_dir}/{metric}_overlay.png``.
    3. Write a stats CSV to ``{working_dir}/{metric}_stats.csv``.

    If a metric is missing in some simulations a warning is logged and the
    remaining simulations are still plotted.

    Args:
        sim_dirs: List of per-simulation root directories (e.g. ``["multi_run/1A",
            "multi_run/2B", "multi_run/3C"]``).
        labels: Human-readable labels for each simulation (same order as
            *sim_dirs*).
        working_dir: Output directory for all combined plots and tables.
        metrics: Metrics to process.  Defaults to
            ``["rmsd", "rmsf", "rg", "energy"]``.
            Each item is used as a filename pattern (e.g. ``"rmsd"`` matches
            ``rmsd.dat``, ``rmsd_ca.dat``, etc.).

    Returns:
        Dict with ``success``, ``plots`` (list of created image paths),
        ``tables`` (list of created CSV paths), and ``skipped`` (metrics
        where no data was found).
    """
    if metrics is None:
        metrics = ["rmsd", "rmsf", "rg", "energy"]

    # Metric → (filename pattern, x-label, y-label)
    _metric_meta: Dict[str, Tuple[str, str, str]] = {
        "rmsd":   ("rmsd",   "Time (ns)", "RMSD (Å)"),
        "rmsf":   ("rmsf",   "Residue",   "RMSF (Å)"),
        "rg":     ("gyration", "Time (ns)", "Rg (Å)"),
        "energy": ("energy", "Time (ns)", "Energy (kJ/mol)"),
        "sasa":   ("sasa",   "Time (ns)", "SASA (nm²)"),
        "hbond":  ("hbond",  "Time (ns)", "H-bonds"),
    }

    Path(working_dir).mkdir(parents=True, exist_ok=True)

    # Always create summary file so downstream reporter has a predictable input.
    try:
        from src.analysis.summary_logger import initialize_summary_file
        initialize_summary_file(working_dir)
    except Exception as init_exc:
        logger.warning(f"run_combined_analysis: could not initialize analysis_summary.jsonl: {init_exc}")

    plots: List[str] = []
    tables: List[str] = []
    skipped: List[str] = []
    # Track (metric, plot_path, table_path) for summary writing
    _processed: List[tuple] = []
    all_ok = True

    for metric in metrics:
        pattern, xlabel, ylabel = _metric_meta.get(
            metric.lower(), (metric, "X", metric.upper())
        )

        # Collect files
        collection = collect_metric_files.func(
            sim_dirs=sim_dirs,
            metric_filename=pattern,
            search_subdir="analysis",
        )

        found = collection.get("found", [])
        if len(found) < 2:
            logger.warning(f"run_combined_analysis: not enough data files for '{metric}' "
                           f"(found {len(found)}) — skipping")
            skipped.append(metric)
            continue

        # Keep only sims that have data; re-align labels
        active_files = [e["file"] for e in found]
        active_labels = []
        for e in found:
            idx = sim_dirs.index(e["sim_dir"]) if e["sim_dir"] in sim_dirs else len(active_labels)
            active_labels.append(labels[idx] if idx < len(labels) else e["sim_dir"])

        # Overlay plot
        plot_result = plot_combined_overlay.func(
            data_files=active_files,
            labels=active_labels,
            output_file=f"{metric}_overlay.png",
            working_dir=working_dir,
            title=f"{metric.upper()} Comparison",
            xlabel=xlabel,
            ylabel=ylabel,
        )
        plot_path = ""
        if plot_result.get("success"):
            plot_path = plot_result["output_path"]
            plots.append(plot_path)
        else:
            all_ok = False

        # Stats table
        table_result = compute_comparison_table.func(
            data_files=active_files,
            labels=active_labels,
            output_csv=f"{metric}_stats.csv",
            working_dir=working_dir,
        )
        table_path = ""
        if table_result.get("success"):
            table_path = table_result["output_path"]
            tables.append(table_path)
        else:
            all_ok = False

        _processed.append((metric, plot_path, table_path))

    # Write analysis_summary.jsonl so the reporter can read structured results
    try:
        from src.analysis.summary_logger import append_analysis_summary

        for metric, plot_path, table_path in _processed:
            # Re-read the CSV we just wrote to pull stats per simulation
            stats_per_sim: Dict[str, Any] = {}
            if table_path:
                try:
                    with open(table_path, newline="", encoding="utf-8") as fcsv:
                        import csv as _csv
                        reader = _csv.DictReader(fcsv)
                        for row in reader:
                            sim_label = row.pop("simulation", None) or row.pop("label", "?")
                            stats_per_sim[sim_label] = {
                                k: (float(v) if v not in ("N/A", "") else v)
                                for k, v in row.items()
                                if k not in ("error",)
                            }
                except Exception as csv_exc:
                    logger.warning(f"run_combined_analysis: could not re-read {table_path}: {csv_exc}")

            append_analysis_summary(
                working_dir=working_dir,
                analysis_type=f"Combined {metric.upper()}",
                statistics=stats_per_sim,
                files={k: v for k, v in {"overlay_plot": plot_path, "stats_csv": table_path}.items() if v},
                metadata={
                    "simulations": labels,
                    "n_simulations": len(labels),
                    "metric": metric,
                },
            )

        # Always add a run-level summary entry, even if every metric was skipped.
        append_analysis_summary(
            working_dir=working_dir,
            analysis_type="Combined_Analysis_Overview",
            statistics={
                "n_plots": len(plots),
                "n_tables": len(tables),
                "n_skipped_metrics": len(skipped),
            },
            files={},
            metadata={
                "simulations": labels,
                "metrics_requested": metrics,
                "metrics_skipped": skipped,
            },
        )
    except Exception as summary_exc:
        logger.warning(f"run_combined_analysis: could not write analysis_summary.jsonl: {summary_exc}")

    return {
        "success": all_ok,
        "plots": plots,
        "tables": tables,
        "skipped": skipped,
        "working_dir": working_dir,
        "summary": (
            f"Combined analysis complete: {len(plots)} plots, "
            f"{len(tables)} tables, {len(skipped)} metrics skipped."
        ),
    }


@tool
def run_combined_dccm_analysis(
    sim_dirs: List[str],
    labels: List[str],
    working_dir: str,
    output_file: str = "dccm_comparison.png",
    vmin: float = -1.0,
    vmax: float = 1.0,
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Collect per-simulation DCCM CSV files and generate comparison heatmaps
    for multi-simulation reports (3×3 grid; paginated as ``*_1.png``,
    ``*_2.png``, … when more than 9 panels).

    Searches each simulation's ``analysis/`` sub-directory for a file whose
    name starts with ``"dccm"`` and ends with ``".csv"``.

    Args:
        sim_dirs: List of per-simulation root directories (same order as
            *labels*).
        labels: Human-readable labels for each simulation (one per sim_dir).
        working_dir: Output directory where the comparison figure(s) are saved.
        output_file: Base filename for the comparison PNG (default:
            ``"dccm_comparison.png"``).
        vmin: Colour scale minimum (default: −1.0).
        vmax: Colour scale maximum (default: +1.0).
        dpi: Image resolution (default: 200).

    Returns:
        Dict with ``success``, ``output_path``, ``output_paths``,
        ``found_files``, ``missing_sims``, and ``message``.
    """
    from src.analysis.dccm_calculator import plot_dccm_comparison as _plot_dccm

    Path(working_dir).mkdir(parents=True, exist_ok=True)

    found_files: List[str] = []
    found_labels: List[str] = []
    missing_sims: List[str] = []

    for sim_dir, label in zip(sim_dirs, labels):
        analysis_dir = Path(sim_dir) / "analysis"
        hit = None
        # Search analysis/ first, then the full sim_dir tree
        for search_root in [str(analysis_dir), sim_dir]:
            for p in sorted(Path(search_root).rglob("dccm*.csv")):
                hit = str(p)
                break
            if hit:
                break
        if hit:
            found_files.append(hit)
            found_labels.append(label)
        else:
            missing_sims.append(sim_dir)
            logger.warning(f"run_combined_dccm_analysis: no dccm CSV in {sim_dir}")

    if len(found_files) < 2:
        msg = (
            f"Not enough DCCM CSV files found "
            f"(need ≥2, found {len(found_files)}). "
            f"Missing: {missing_sims}"
        )
        logger.warning(msg)
        return {
            "success": False,
            "found_files": found_files,
            "missing_sims": missing_sims,
            "message": msg,
        }

    result = _plot_dccm.func(
        dccm_files=found_files,
        labels=found_labels,
        output_file=output_file,
        working_dir=working_dir,
        vmin=vmin,
        vmax=vmax,
        dpi=dpi,
    )

    # Write to analysis_summary.jsonl
    try:
        from src.analysis.summary_logger import append_analysis_summary
        append_analysis_summary(
            working_dir=working_dir,
            analysis_type="Combined_DCCM",
            statistics={
                "n_simulations_compared": len(found_files),
                "n_pages": result.get("n_pages", 1),
            },
            files={
                "comparison_figure": result.get("output_path", ""),
                "comparison_figures": result.get("output_paths")
                or ([result.get("output_path")] if result.get("output_path") else []),
                "dccm_csv_files": found_files,
            },
            metadata={
                "simulations": found_labels,
                "missing_simulations": missing_sims,
            },
        )
    except Exception as se:
        logger.warning(f"run_combined_dccm_analysis: could not write summary: {se}")

    result["found_files"] = found_files
    result["missing_sims"] = missing_sims
    return result


@tool
def run_combined_dccm_difference(
    reference_sim_dir: str,
    compare_sim_dir: str,
    reference_label: str,
    compare_label: str,
    working_dir: str,
    output_prefix: str = "dccm_difference",
    plot_mode: str = "difference_only",
    diff_threshold: float = 0.3,
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Build a DCCM difference map between two simulations (e.g. protein-only vs protein+ATP).

    Locates ``dccm*.csv`` under each simulation's ``analysis/`` folder (from
    ``calculate_dccm``), then calls ``plot_dccm_difference`` with
    Δ = compare − reference.

    Args:
        reference_sim_dir: Baseline simulation root (protein-only).
        compare_sim_dir: Perturbed simulation root (e.g. protein + ATP + Mg).
        reference_label: Plot label for baseline.
        compare_label: Plot label for perturbed system.
        working_dir: Where difference CSV/PNG are written.
        output_prefix: Output filename prefix (default: ``dccm_difference``).
        plot_mode: ``difference_only`` or ``with_matrices``.
        diff_threshold: |ΔC| cutoff for reporting changed pairs.
        dpi: Figure resolution.

    Returns:
        Dict from ``plot_dccm_difference`` plus ``reference_csv`` / ``compare_csv`` paths.
    """
    from src.analysis.dccm_calculator import plot_dccm_difference as _plot_diff

    Path(working_dir).mkdir(parents=True, exist_ok=True)

    def _find_dccm_csv(sim_dir: str) -> Optional[str]:
        analysis_dir = Path(sim_dir) / "analysis"
        for search_root in [str(analysis_dir), sim_dir]:
            for p in sorted(Path(search_root).rglob("dccm*.csv")):
                return str(p)
        return None

    ref_csv = _find_dccm_csv(reference_sim_dir)
    cmp_csv = _find_dccm_csv(compare_sim_dir)
    if not ref_csv:
        return {
            "success": False,
            "message": f"No dccm CSV in reference sim: {reference_sim_dir}",
        }
    if not cmp_csv:
        return {
            "success": False,
            "message": f"No dccm CSV in compare sim: {compare_sim_dir}",
        }

    result = _plot_diff.func(
        reference_dccm_file=ref_csv,
        compare_dccm_file=cmp_csv,
        reference_label=reference_label,
        compare_label=compare_label,
        output_prefix=output_prefix,
        working_dir=working_dir,
        plot_mode=plot_mode,
        diff_threshold=diff_threshold,
        dpi=dpi,
    )
    result["reference_csv"] = ref_csv
    result["compare_csv"] = cmp_csv

    try:
        from src.analysis.summary_logger import append_analysis_summary
        append_analysis_summary(
            working_dir=working_dir,
            analysis_type="Combined_DCCM_Difference",
            statistics=result.get("delta_matrix_stats", {}),
            files=result.get("output_files", {}),
            metadata={
                "reference_sim_dir": reference_sim_dir,
                "compare_sim_dir": compare_sim_dir,
                "reference_label": reference_label,
                "compare_label": compare_label,
                "reference_csv": ref_csv,
                "compare_csv": cmp_csv,
            },
        )
    except Exception as se:
        logger.warning(f"run_combined_dccm_difference: summary log failed: {se}")

    return result


# ---------------------------------------------------------------------------
# Apo / holo pairing helpers
# ---------------------------------------------------------------------------

_HOLO_SUFFIXES = ("_atp_mg", "_atp", "-atp-mg", "_holo")
_HOLO_KEYWORDS = {"_atp_mg", "_atp_", "atp_mg", "holo", "ligand", "bound", "_mg_"}


def _base_protein_id(name: str) -> str:
    """Extract base protein ID from a simulation label or directory name."""
    base = Path(name).name.lower()
    for suffix in _HOLO_SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def is_holo_simulation(sim_dir: str, label: str) -> bool:
    """Return True when the simulation directory/label denotes a holo (ligand-bound) system."""
    key = f"{sim_dir} {label}".lower()
    return any(kw in key for kw in _HOLO_KEYWORDS)


def label_is_holo(label: str) -> bool:
    """True when this simulation's own name is ligand-bound.

    A campaign folder such as ``pseudo_apo_holo`` contains "holo" and must not
    make every protein-only run look ligand-bound.
    """
    name = Path(str(label or "")).name
    return is_holo_simulation(name, name)


def _get_label_name_map(
    label_name_map: Optional[Dict[str, str]] = None,
    user_goal: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve {uniprot_id: protein_name} from explicit map or goal text."""
    if label_name_map:
        return {k.lower(): v for k, v in label_name_map.items()}
    if user_goal:
        from src.reporter.combined_reporter import _parse_label_name_map
        return _parse_label_name_map(user_goal)
    return {}


def _protein_display_name(
    protein_id: str,
    apo_label: str,
    name_map: Optional[Dict[str, str]] = None,
) -> str:
    """Human-readable protein name for plot titles and legends."""
    from src.reporter.combined_reporter import resolve_display_label

    if name_map:
        for candidate in (apo_label, protein_id):
            display = resolve_display_label(candidate, name_map)
            if display != candidate:
                return display
    # Labels may already be mapped (e.g. ERBB3 instead of p21860).
    base = _base_protein_id(apo_label)
    if base and not re.match(r"^[pq]\d", base, re.IGNORECASE):
        return apo_label.split("_")[0].split("-")[0]
    return (apo_label or protein_id).upper()


def _apo_holo_legend_labels(
    protein_id: str,
    apo_label: str,
    name_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Return (apo_legend, holo_legend) using protein name when available."""
    display = _protein_display_name(protein_id, apo_label, name_map)
    return f"{display} (Apo)", f"{display} (Holo)"


def pair_apo_holo_simulations(
    sim_dirs: List[str],
    labels: List[str],
    label_name_map: Optional[Dict[str, str]] = None,
    user_goal: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Group simulation directories into apo/holo pairs by base protein ID.

    Example: ``p21860`` + ``p21860_ATP_MG`` → one pair with protein_id ``p21860``.
    """
    by_id: Dict[str, Dict[str, str]] = {}

    for sim_dir, label in zip(sim_dirs, labels):
        protein_id = _base_protein_id(label) or _base_protein_id(sim_dir)
        if protein_id not in by_id:
            by_id[protein_id] = {"protein_id": protein_id}
        entry = by_id[protein_id]
        if is_holo_simulation(sim_dir, label):
            entry["holo_dir"] = sim_dir
            entry["holo_label"] = label
        else:
            entry["apo_dir"] = sim_dir
            entry["apo_label"] = label

    name_map = _get_label_name_map(label_name_map, user_goal)
    pairs: List[Dict[str, str]] = []
    for protein_id in sorted(by_id):
        entry = by_id[protein_id]
        if entry.get("apo_dir") and entry.get("holo_dir"):
            apo_label = entry.get("apo_label", protein_id)
            holo_label = entry.get("holo_label", f"{protein_id}_ATP_MG")
            protein_display = _protein_display_name(protein_id, apo_label, name_map)
            apo_legend, holo_legend = _apo_holo_legend_labels(
                protein_id, apo_label, name_map
            )
            pairs.append({
                "protein_id": protein_id,
                "protein_display": protein_display,
                "apo_dir": entry["apo_dir"],
                "apo_label": apo_label,
                "holo_dir": entry["holo_dir"],
                "holo_label": holo_label,
                "apo_legend": apo_legend,
                "holo_legend": holo_legend,
            })
    return pairs


# ---------------------------------------------------------------------------
# RMSF segment bar plots & COM distance overlay
# ---------------------------------------------------------------------------

def parse_rmsf_segments_from_goal(text: str) -> List[Dict[str, Any]]:
    """
    Extract named protein residue ranges from a user goal string.

    Recognises patterns such as:
      - ``activation loop: 150 to 190``
      - ``residue range 150-190``
      - ``RMSF for residues 150 to 190``
    """
    if not text:
        return []

    segments: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(name: str, start: int, end: int) -> None:
        if start > end:
            start, end = end, start
        key = (start, end)
        if key in seen or end - start < 1:
            return
        low_name = name.lower()
        if any(skip in low_name for skip in ("uniprot", "pdb:", "protein +")):
            return
        segments.append({
            "name": name.strip(),
            "residue_start": start,
            "residue_end": end,
        })
        seen.add(key)

    for m in re.finditer(
        r'([A-Za-z][A-Za-z0-9\s\-]{2,40}?)\s*:\s*(\d+)\s*(?:to|–|-|—)\s*(\d+)',
        text,
        re.IGNORECASE,
    ):
        _add(m.group(1), int(m.group(2)), int(m.group(3)))

    for m in re.finditer(
        r'(?:residue(?:s)?|resid(?:ue)?s?)\s+(?:range\s+)?(\d+)\s*(?:to|–|-|—)\s*(\d+)',
        text,
        re.IGNORECASE,
    ):
        _add("Residue segment", int(m.group(1)), int(m.group(2)))

    for m in re.finditer(
        r'(activation[\s\-]*loop|αC[\s\-]*helix|A[\s\-]*loop|'
        r'activation[\s\-]*segment|flexible[\s\-]*region)\s+'
        r'(\d+)\s*(?:to|–|-|—)\s*(\d+)',
        text,
        re.IGNORECASE,
    ):
        _add(m.group(1), int(m.group(2)), int(m.group(3)))

    return segments


def _read_rmsf_file(filepath: str) -> Tuple[List[int], List[float]]:
    """Read per-residue RMSF from .dat (tab/space) or two-column file."""
    residues: List[int] = []
    values: List[float] = []

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            raw = line.strip()
            if not raw or raw.startswith("#") or raw.startswith("@"):
                continue
            parts = raw.replace(",", "\t").split()
            if len(parts) < 2:
                continue
            try:
                residues.append(int(float(parts[0])))
                values.append(float(parts[1]))
            except ValueError:
                continue

    if not residues:
        raise ValueError(f"No RMSF data found in {filepath}")
    return residues, values


def _slugify_segment_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "segment"


@tool
def plot_combined_rmsf_segment_bars(
    sim_dirs: List[str],
    labels: List[str],
    residue_start: int,
    residue_end: int,
    working_dir: str,
    segment_name: str = "Segment",
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Grouped bar chart of per-residue RMSF within a residue range for all simulations.
    """
    if not HAS_MATPLOTLIB:
        return {"success": False, "error": "matplotlib not available"}

    if residue_start > residue_end:
        residue_start, residue_end = residue_end, residue_start

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    slug = _slugify_segment_name(segment_name)
    out_name = output_file or f"rmsf_segment_{slug}_{residue_start}_{residue_end}.png"
    output_path = str(Path(working_dir) / out_name)

    series: List[Dict[str, Any]] = []
    missing: List[str] = []

    for sim_dir, label in zip(sim_dirs, labels):
        rmsf_file = _find_metric_file(str(Path(sim_dir) / "analysis"), "rmsf")
        if not rmsf_file:
            rmsf_file = _find_metric_file(sim_dir, "rmsf")
        if not rmsf_file:
            missing.append(label)
            continue
        try:
            res_ids, rmsf_vals = _read_rmsf_file(rmsf_file)
            filtered = {
                r: v for r, v in zip(res_ids, rmsf_vals)
                if residue_start <= r <= residue_end
            }
            if not filtered:
                missing.append(label)
                continue
            series.append({"label": label, "data": filtered})
        except Exception as exc:
            logger.warning(f"plot_combined_rmsf_segment_bars: {label}: {exc}")
            missing.append(label)

    if len(series) < 1:
        return {
            "success": False,
            "error": f"No RMSF data in residues {residue_start}-{residue_end}",
            "missing": missing,
        }

    all_residues = sorted({r for s in series for r in s["data"]})
    if not all_residues:
        return {"success": False, "error": "No residues in requested range"}

    n_res = len(all_residues)
    n_sims = len(series)
    x = list(range(n_res))
    width = 0.8 / max(n_sims, 1)
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                      "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, s in enumerate(series):
        offset = (i - (n_sims - 1) / 2) * width
        ys = [s["data"].get(r, 0.0) for r in all_residues]
        ax.bar(
            [xi + offset for xi in x], ys, width=width,
            label=s["label"],
            color=default_colors[i % len(default_colors)],
            alpha=0.88,
        )

    step = max(1, n_res // 12)
    tick_pos = list(range(0, n_res, step))
    tick_labels = [str(all_residues[i]) for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Residue", fontsize=10)
    ax.set_ylabel("RMSF (Å)", fontsize=10)
    ax.set_title(
        f"RMSF — {segment_name} (residues {residue_start}–{residue_end})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "success": True,
        "output_path": output_path,
        "segment_name": segment_name,
        "residue_start": residue_start,
        "residue_end": residue_end,
        "n_simulations": len(series),
        "missing": missing,
        "message": (
            f"RMSF segment bar plot for {segment_name} "
            f"({residue_start}–{residue_end}) saved to {output_path}"
        ),
    }


@tool
def run_combined_rmsf_segment_analysis(
    sim_dirs: List[str],
    labels: List[str],
    working_dir: str,
    user_goal: Optional[str] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate RMSF bar plots for residue segments specified in the user goal."""
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    seg_list = segments or parse_rmsf_segments_from_goal(user_goal or "")
    if not seg_list:
        return {
            "success": False,
            "plots": [],
            "segments": [],
            "message": "No RMSF residue segments found in user goal",
        }

    plots: List[str] = []
    processed: List[Dict[str, Any]] = []

    for seg in seg_list:
        result = plot_combined_rmsf_segment_bars.func(
            sim_dirs=sim_dirs,
            labels=labels,
            residue_start=int(seg["residue_start"]),
            residue_end=int(seg["residue_end"]),
            working_dir=working_dir,
            segment_name=seg.get("name", "Segment"),
        )
        entry = {
            "segment": seg,
            "success": result.get("success", False),
            "output_path": result.get("output_path", ""),
        }
        processed.append(entry)
        if result.get("success"):
            plots.append(result["output_path"])

    try:
        from src.analysis.summary_logger import append_analysis_summary
        append_analysis_summary(
            working_dir=working_dir,
            analysis_type="Combined_RMSF_Segments",
            statistics={"n_segments": len(processed), "n_plots": len(plots)},
            files={"segment_plots": plots},
            metadata={"segments": seg_list, "results": processed},
        )
    except Exception as se:
        logger.warning(f"run_combined_rmsf_segment_analysis: summary log failed: {se}")

    return {
        "success": len(plots) > 0,
        "plots": plots,
        "segments": seg_list,
        "results": processed,
        "message": f"Generated {len(plots)} RMSF segment bar plot(s)",
    }


_COM_DISTANCE_FILE_PATTERNS = (
    "ligand_pocket_distance",
    "pocket_distance",
    "com_distance",
)

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
_ION_RESNAMES = {"NA", "CL", "MG", "K", "CA", "ZN", "FE", "MN", "CU"}
_SOLVENT_RESNAMES = {"SOL", "WAT", "HOH", "TIP3", "TIP4", "SPC", "SPCE", "W"}
_KNOWN_LIGAND_RESNAMES = ("ATP", "ADP", "AMP", "GTP", "GDP", "NAD", "UNL", "LIG")


def _infer_ligand_selection(topology_file: str, trajectory_file: str) -> Optional[str]:
    """Best-effort MDAnalysis selection for the bound ligand in a holo system."""
    try:
        import MDAnalysis as mda
    except ImportError:
        return None

    try:
        u = mda.Universe(topology_file, trajectory_file)
        for resname in _KNOWN_LIGAND_RESNAMES:
            sel = f"resname {resname}"
            if len(u.select_atoms(sel)) > 0:
                return sel

        candidates: List[Tuple[str, int]] = []
        for res in u.residues:
            rn = res.resname.upper()
            if rn in _STANDARD_AA or rn in _ION_RESNAMES or rn in _SOLVENT_RESNAMES:
                continue
            n_atoms = len(res.atoms)
            if 5 <= n_atoms <= 80:
                candidates.append((rn, n_atoms))
        if candidates:
            candidates.sort(key=lambda item: item[1])
            return f"resname {candidates[0][0]}"
    except Exception as exc:
        logger.warning(f"_infer_ligand_selection failed: {exc}")
    return None


def _find_com_distance_file(directory: str) -> Optional[str]:
    """Locate a ligand-pocket COM distance CSV under *directory*."""
    for pattern in _COM_DISTANCE_FILE_PATTERNS:
        hit = _find_metric_file(directory, pattern)
        if hit:
            return hit
    return None


def _ensure_ligand_pocket_distance_csv(sim_dir: str) -> Optional[str]:
    """Return ligand-pocket COM distance CSV for *sim_dir*, computing it if absent.

    Per-simulation analysis may omit ``calculate_ligand_pocket_distance`` or write
    the metric under an alternate filename (e.g. ``pocket_distance.csv``).
    """
    analysis_dir = Path(sim_dir) / "analysis"
    for root in (analysis_dir, Path(sim_dir)):
        if root.is_dir():
            hit = _find_com_distance_file(str(root))
            if hit:
                return hit

    topo, traj = _find_sim_traj_topology(sim_dir)
    if not topo or not traj:
        logger.warning(
            f"_ensure_ligand_pocket_distance_csv: no topology/trajectory for {sim_dir}"
        )
        return None

    try:
        from src.analysis.com_distance_calculator import calculate_ligand_pocket_distance

        analysis_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"_ensure_ligand_pocket_distance_csv: computing for {sim_dir} "
            f"from {Path(traj).name}"
        )
        ligand_selection = _infer_ligand_selection(topo, traj)
        if not ligand_selection:
            logger.info(
                "_ensure_ligand_pocket_distance_csv: no ligand in %s; skipping",
                sim_dir,
            )
            return None
        res = calculate_ligand_pocket_distance.func(
            topology_file=topo,
            trajectory_file=traj,
            ligand_selection=ligand_selection,
            output_file="ligand_pocket_distance.csv",
            working_dir=str(analysis_dir),
        )
        if not res.get("success"):
            logger.warning(
                f"_ensure_ligand_pocket_distance_csv: failed for {sim_dir}: "
                f"{res.get('error') or res.get('message')}"
            )
            return None

        out = res.get("output_file") or "ligand_pocket_distance.csv"
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = analysis_dir / out_path
        if out_path.is_file():
            return str(out_path)
    except Exception as exc:
        logger.warning(
            f"_ensure_ligand_pocket_distance_csv: error for {sim_dir}: {exc}"
        )
    return None


@tool
def run_combined_com_distance_analysis(
    sim_dirs: List[str],
    labels: List[str],
    working_dir: str,
    output_file: str = "com_distance_overlay.png",
) -> Dict[str, Any]:
    """
    Overlay ATP–catalytic-pocket COM distance time series across holo simulations.
    """
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    found_files: List[str] = []
    found_labels: List[str] = []
    missing: List[str] = []

    for sim_dir, label in zip(sim_dirs, labels):
        hit = None
        analysis_dir = Path(sim_dir) / "analysis"
        if analysis_dir.is_dir():
            hit = _find_com_distance_file(str(analysis_dir))

        # Skip apo-only labels when no COM data exists; include any sim that
        # already has per-sim ligand_pocket_distance output (UniProt labels like
        # p29597 are holo but do not contain holo/atp in the directory name).
        if not hit and not is_holo_simulation(sim_dir, label):
            missing.append(label)
            continue

        if not hit:
            hit = _ensure_ligand_pocket_distance_csv(sim_dir)
        if hit:
            found_files.append(hit)
            found_labels.append(label)
        else:
            missing.append(label)

    if len(found_files) < 1:
        return {
            "success": False,
            "message": (
                "No ligand pocket / COM distance CSV found in any simulation. "
                f"Missing: {missing}"
            ),
            "missing": missing,
        }

    # De-spike each per-sim COM distance series (PBC imaging artifacts) into a
    # scratch dir so the overlay is clean without mutating per-sim outputs.
    plot_files = found_files
    try:
        from src.analysis.pbc_utils import clean_distance_csv

        clean_dir = Path(working_dir) / "_com_clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        cleaned_files: List[str] = []
        total_removed = 0
        for src, label in zip(found_files, found_labels):
            dst = clean_dir / f"{label}_com_clean.csv"
            ok, n_removed = clean_distance_csv(str(src), str(dst))
            cleaned_files.append(str(dst) if ok else str(src))
            total_removed += n_removed
        plot_files = cleaned_files
        if total_removed:
            logger.info(
                "run_combined_com_distance_analysis: removed %d PBC spike(s) "
                "across %d simulation(s) before overlay",
                total_removed, len(found_files),
            )
    except Exception as exc:
        logger.warning(
            "run_combined_com_distance_analysis: PBC cleaning skipped (%s); "
            "plotting raw series", exc,
        )
        plot_files = found_files

    result = plot_combined_overlay.func(
        data_files=plot_files,
        labels=found_labels,
        output_file=output_file,
        working_dir=working_dir,
        title="ATP–Catalytic Pocket COM Distance",
        xlabel="Time (ns)",
        ylabel="COM Distance (Å)",
        x_col=1,
        y_col=2,
    )

    if result.get("success"):
        try:
            from src.analysis.summary_logger import append_analysis_summary
            append_analysis_summary(
                working_dir=working_dir,
                analysis_type="Combined_COM_Distance",
                statistics=result.get("stats", {}),
                files={
                    "overlay_plot": result.get("output_path", ""),
                    "source_csv_files": found_files,
                },
                metadata={
                    "simulations": found_labels,
                    "missing_simulations": missing,
                },
            )
        except Exception as se:
            logger.warning(f"run_combined_com_distance_analysis: summary failed: {se}")

    result["missing"] = missing
    result["found_files"] = found_files
    return result


# ---------------------------------------------------------------------------
# Per-protein apo vs holo comparison plots
# ---------------------------------------------------------------------------

@tool
def plot_rmsf_apo_holo_comparison(
    apo_rmsf_file: str,
    holo_rmsf_file: str,
    apo_label: str,
    holo_label: str,
    protein_label: str,
    output_file: str,
    working_dir: str,
    figsize: Tuple[int, int] = (10, 4),
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Overlay apo and holo RMSF curves for a single protein on one panel.
    """
    if not HAS_MATPLOTLIB:
        return {"success": False, "error": "matplotlib not available"}

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(working_dir) / output_file)

    try:
        apo_res, apo_vals = _read_rmsf_file(apo_rmsf_file)
        holo_res, holo_vals = _read_rmsf_file(holo_rmsf_file)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(apo_res, apo_vals, label=apo_label, color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax.plot(holo_res, holo_vals, label=holo_label, color="#d62728", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Residue", fontsize=10)
    ax.set_ylabel("RMSF (Å)", fontsize=10)
    ax.set_title(f"RMSF — {protein_label} (Apo vs Holo)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "success": True,
        "output_path": output_path,
        "protein_label": protein_label,
        "message": f"RMSF apo/holo comparison saved to {output_path}",
    }


@tool
def run_combined_rmsf_apo_holo_analysis(
    sim_dirs: List[str],
    labels: List[str],
    working_dir: str,
    output_file: str = "rmsf_apo_holo_comparison.png",
    per_protein_files: bool = True,
    figsize_per_panel: Tuple[int, int] = (10, 3),
    dpi: int = 200,
    label_name_map: Optional[Dict[str, str]] = None,
    user_goal: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate per-protein RMSF comparisons (apo vs holo overlaid).

    Produces:
    - One multi-row summary figure (``rmsf_apo_holo_comparison.png``) with one
      panel per protein.
    - Optional individual PNGs per protein (``rmsf_apo_holo_<protein_id>.png``).
    """
    if not HAS_MATPLOTLIB:
        return {"success": False, "error": "matplotlib not available", "plots": []}

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    pairs = pair_apo_holo_simulations(
        sim_dirs, labels,
        label_name_map=label_name_map,
        user_goal=user_goal,
    )
    if not pairs:
        return {
            "success": False,
            "plots": [],
            "pairs": [],
            "message": "No apo/holo simulation pairs detected",
        }

    plots: List[str] = []
    processed: List[Dict[str, Any]] = []

    n_pairs = len(pairs)
    fig, axes = plt.subplots(
        n_pairs, 1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * n_pairs),
        squeeze=False,
    )

    for row, pair in enumerate(pairs):
        protein_id = pair["protein_id"]
        protein_display = pair.get("protein_display", protein_id.upper())
        apo_legend = pair.get("apo_legend", pair["apo_label"])
        holo_legend = pair.get("holo_legend", pair["holo_label"])
        apo_file = _find_metric_file(str(Path(pair["apo_dir"]) / "analysis"), "rmsf")
        if not apo_file:
            apo_file = _find_metric_file(pair["apo_dir"], "rmsf")
        holo_file = _find_metric_file(str(Path(pair["holo_dir"]) / "analysis"), "rmsf")
        if not holo_file:
            holo_file = _find_metric_file(pair["holo_dir"], "rmsf")

        if not apo_file or not holo_file:
            processed.append({
                "protein_id": protein_id,
                "success": False,
                "error": "Missing RMSF file for apo or holo",
            })
            axes[row, 0].set_visible(False)
            continue

        try:
            apo_res, apo_vals = _read_rmsf_file(apo_file)
            holo_res, holo_vals = _read_rmsf_file(holo_file)
        except Exception as exc:
            processed.append({"protein_id": protein_id, "success": False, "error": str(exc)})
            axes[row, 0].set_visible(False)
            continue

        ax = axes[row, 0]
        ax.plot(apo_res, apo_vals, label=apo_legend, color="#1f77b4", linewidth=1.2)
        ax.plot(holo_res, holo_vals, label=holo_legend, color="#d62728", linewidth=1.2)
        ax.set_ylabel("RMSF (Å)", fontsize=9)
        ax.set_title(f"{protein_display} — Apo vs Holo", fontsize=10, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == n_pairs - 1:
            ax.set_xlabel("Residue", fontsize=10)

        entry = {
            "protein_id": protein_id,
            "success": True,
            "apo_file": apo_file,
            "holo_file": holo_file,
        }
        processed.append(entry)

        if per_protein_files:
            indiv = plot_rmsf_apo_holo_comparison.func(
                apo_rmsf_file=apo_file,
                holo_rmsf_file=holo_file,
                apo_label=apo_legend,
                holo_label=holo_legend,
                protein_label=protein_display,
                output_file=f"rmsf_apo_holo_{protein_id}.png",
                working_dir=working_dir,
            )
            if indiv.get("success"):
                plots.append(indiv["output_path"])
                entry["individual_plot"] = indiv["output_path"]

    fig.suptitle("Per-Residue RMSF — Apo vs Holo by Pseudokinase", fontsize=13, fontweight="bold")
    fig.tight_layout()
    summary_path = str(Path(working_dir) / output_file)
    fig.savefig(summary_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    plots.insert(0, summary_path)

    try:
        from src.analysis.summary_logger import append_analysis_summary
        append_analysis_summary(
            working_dir=working_dir,
            analysis_type="Combined_RMSF_Apo_Holo",
            statistics={"n_pairs": len(pairs), "n_plots": len(plots)},
            files={"summary_plot": summary_path, "per_protein_plots": plots[1:]},
            metadata={"pairs": pairs, "results": processed},
        )
    except Exception as se:
        logger.warning(f"run_combined_rmsf_apo_holo_analysis: summary log failed: {se}")

    return {
        "success": any(p.get("success") for p in processed),
        "plots": plots,
        "pairs": pairs,
        "results": processed,
        "message": f"Generated {len(plots)} RMSF apo/holo plot(s) for {len(pairs)} protein(s)",
    }


def _find_sim_traj_topology(
    sim_dir: str,
    hpc_dir: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Locate (topology, trajectory) for a simulation directory.

    Prefers a wrapped trajectory (``mdWrap.xtc``) and ``md.tpr``/``md.gro``
    under ``hpc/repXX`` (or flat ``hpc/``). Avoids fresh simsetup ``system.gro``
    when production topologies exist (atom-count mismatch under reuse-hpc).

    When several ``hpc/repXX`` slots exist, ``hpc_dir`` is required — we do
    not silently return the first replica.
    """
    from src.analysis.replicate_paths import (
        discover_hpc_rep_dirs,
        resolve_production_topology,
        resolve_production_trajectory,
    )

    if hpc_dir:
        slot = Path(hpc_dir)
        if slot.is_dir():
            found_t = resolve_production_topology(slot)
            found_x = resolve_production_trajectory(slot)
            if found_t is not None and found_x is not None:
                return str(found_t), str(found_x)

    nested = list(discover_hpc_rep_dirs(sim_dir))
    if len(nested) > 1 and not hpc_dir:
        logger.warning(
            "_find_sim_traj_topology: %d replica dirs under %s without hpc_dir — "
            "using %s as a representative slot (per-rep tools must pass hpc_dir)",
            len(nested),
            sim_dir,
            nested[0],
        )

    search_roots: list[Path] = list(nested)
    flat = Path(sim_dir) / "hpc"
    if flat.is_dir() and flat not in search_roots:
        search_roots.append(flat)
    search_roots.append(Path(sim_dir))

    topo: Optional[str] = None
    traj: Optional[str] = None
    for root in search_roots:
        if not root.is_dir():
            continue
        if traj is None:
            found = resolve_production_trajectory(root)
            if found is not None:
                traj = str(found)
            else:
                for name in ("mdWrap.xtc", "md.xtc", "md.trr"):
                    cand = root / name
                    if cand.is_file():
                        traj = str(cand)
                        break
        if topo is None:
            found_t = resolve_production_topology(root)
            if found_t is not None:
                topo = str(found_t)
        if traj and topo:
            break
    return topo, traj


def _ensure_dccm_csv(sim_dir: str) -> Optional[str]:
    """Return a DCCM CSV for *sim_dir*, computing it from the trajectory if absent.

    Per-simulation DCCM can be missing when the per-sim analysis step failed
    (e.g. the LLM omitted required arguments to ``calculate_dccm``).  To keep
    the combined apo/holo ΔDCCM robust, this falls back to computing DCCM
    directly from the simulation's wrapped trajectory + topology.
    """
    analysis_dir = Path(sim_dir) / "analysis"
    for root in (analysis_dir, Path(sim_dir)):
        if root.is_dir():
            for p in sorted(root.rglob("dccm*.csv")):
                return str(p)

    topo, traj = _find_sim_traj_topology(sim_dir)
    if not topo or not traj:
        logger.warning(f"_ensure_dccm_csv: no topology/trajectory found for {sim_dir}")
        return None
    try:
        from src.analysis.dccm_calculator import calculate_dccm
        analysis_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"_ensure_dccm_csv: per-sim DCCM CSV missing for {sim_dir} — "
            f"computing from {Path(traj).name}"
        )
        res = calculate_dccm.func(
            topology_file=topo,
            trajectory_file=traj,
            selection="protein and name CA",
            output_prefix="dccm",
            working_dir=str(analysis_dir),
        )
        if res.get("success"):
            csv_path = (res.get("output_files") or {}).get("csv")
            if csv_path and Path(csv_path).is_file():
                return csv_path
        logger.warning(
            f"_ensure_dccm_csv: calculate_dccm failed for {sim_dir}: "
            f"{res.get('error') or res.get('message')}"
        )
    except Exception as exc:
        logger.warning(f"_ensure_dccm_csv: DCCM computation error for {sim_dir}: {exc}")
    return None


@tool
def run_combined_dccm_apo_holo_analysis(
    sim_dirs: List[str],
    labels: List[str],
    working_dir: str,
    plot_mode: str = "with_matrices",
    diff_threshold: float = 0.3,
    dpi: int = 200,
    label_name_map: Optional[Dict[str, str]] = None,
    user_goal: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For each apo/holo protein pair, generate a 3-panel DCCM figure:
    apo DCCM | holo DCCM | Δ(holo − apo).

    Missing per-simulation DCCM matrices are computed on demand from the
    wrapped trajectory so the ΔDCCM comparison is always produced when the
    trajectories are available.
    """
    from src.analysis.dccm_calculator import plot_dccm_difference as _plot_diff

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    pairs = pair_apo_holo_simulations(
        sim_dirs, labels,
        label_name_map=label_name_map,
        user_goal=user_goal,
    )
    if not pairs:
        return {
            "success": False,
            "plots": [],
            "pairs": [],
            "message": "No apo/holo simulation pairs detected",
        }

    plots: List[str] = []
    processed: List[Dict[str, Any]] = []

    for pair in pairs:
        protein_id = pair["protein_id"]
        ref_csv = _ensure_dccm_csv(pair["apo_dir"])
        cmp_csv = _ensure_dccm_csv(pair["holo_dir"])
        if not ref_csv or not cmp_csv:
            missing = []
            if not ref_csv:
                missing.append("apo")
            if not cmp_csv:
                missing.append("holo")
            processed.append({
                "protein_id": protein_id,
                "success": False,
                "error": f"Missing DCCM CSV (could not compute) for: {', '.join(missing)}",
            })
            continue

        apo_legend = pair.get("apo_legend", f"{pair['apo_label']} (Apo)")
        holo_legend = pair.get("holo_legend", f"{pair['holo_label']} (Holo)")
        result = _plot_diff.func(
            reference_dccm_file=ref_csv,
            compare_dccm_file=cmp_csv,
            reference_label=apo_legend,
            compare_label=holo_legend,
            output_prefix=f"dccm_apo_holo_{protein_id}",
            working_dir=working_dir,
            plot_mode=plot_mode,
            diff_threshold=diff_threshold,
            dpi=dpi,
        )
        entry = {
            "protein_id": protein_id,
            "success": result.get("success", False),
            "output_files": result.get("output_files", {}),
            "mean_abs_delta": result.get("delta_matrix_stats", {}).get("mean_abs_delta"),
        }
        processed.append(entry)
        if result.get("success"):
            panels = result.get("output_files", {}).get("panels")
            heatmap = result.get("output_files", {}).get("heatmap")
            if panels:
                plots.append(panels)
            elif heatmap:
                plots.append(heatmap)

    try:
        from src.analysis.summary_logger import append_analysis_summary
        append_analysis_summary(
            working_dir=working_dir,
            analysis_type="Combined_DCCM_Apo_Holo",
            statistics={
                "n_pairs": len(pairs),
                "n_plots": len(plots),
            },
            files={"per_protein_dccm_plots": plots},
            metadata={"pairs": pairs, "results": processed},
        )
    except Exception as se:
        logger.warning(f"run_combined_dccm_apo_holo_analysis: summary log failed: {se}")

    return {
        "success": len(plots) > 0,
        "plots": plots,
        "pairs": pairs,
        "results": processed,
        "message": f"Generated {len(plots)} per-protein DCCM apo/holo plot(s)",
    }


@tool
def run_combined_rmsf_segment_apo_holo_analysis(
    sim_dirs: List[str],
    labels: List[str],
    working_dir: str,
    user_goal: Optional[str] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    label_name_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    RMSF segment bar plots per protein — apo vs holo only (two bar series per residue).
    """
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    pairs = pair_apo_holo_simulations(
        sim_dirs, labels,
        label_name_map=label_name_map,
        user_goal=user_goal,
    )
    seg_list = segments or parse_rmsf_segments_from_goal(user_goal or "")
    if not pairs:
        return {"success": False, "plots": [], "message": "No apo/holo pairs detected"}
    if not seg_list:
        return {"success": False, "plots": [], "message": "No RMSF residue segments found"}

    plots: List[str] = []
    processed: List[Dict[str, Any]] = []

    for pair in pairs:
        protein_display = pair.get("protein_display", pair["protein_id"].upper())
        apo_legend = pair.get("apo_legend", pair["apo_label"])
        holo_legend = pair.get("holo_legend", pair["holo_label"])
        for seg in seg_list:
            result = plot_combined_rmsf_segment_bars.func(
                sim_dirs=[pair["apo_dir"], pair["holo_dir"]],
                labels=[apo_legend, holo_legend],
                residue_start=int(seg["residue_start"]),
                residue_end=int(seg["residue_end"]),
                working_dir=working_dir,
                segment_name=f"{protein_display} — {seg.get('name', 'Segment')}",
                output_file=(
                    f"rmsf_segment_{_slugify_segment_name(protein_display)}_"
                    f"{_slugify_segment_name(seg.get('name', 'segment'))}_"
                    f"{seg['residue_start']}_{seg['residue_end']}.png"
                ),
            )
            processed.append({
                "protein_id": pair["protein_id"],
                "segment": seg,
                "success": result.get("success", False),
                "output_path": result.get("output_path", ""),
            })
            if result.get("success"):
                plots.append(result["output_path"])

    try:
        from src.analysis.summary_logger import append_analysis_summary
        append_analysis_summary(
            working_dir=working_dir,
            analysis_type="Combined_RMSF_Segment_Apo_Holo",
            statistics={"n_plots": len(plots), "n_pairs": len(pairs)},
            files={"segment_plots": plots},
            metadata={"segments": seg_list, "results": processed},
        )
    except Exception as se:
        logger.warning(f"run_combined_rmsf_segment_apo_holo_analysis: summary failed: {se}")

    return {
        "success": len(plots) > 0,
        "plots": plots,
        "segments": seg_list,
        "pairs": pairs,
        "results": processed,
        "message": f"Generated {len(plots)} per-protein apo/holo segment bar plot(s)",
    }


_BINDING_RMSF_CONFIG: Dict[str, Dict[str, str]] = {
    "pocket_rmsf": {
        "file_pattern": "pocket_rmsf",
        "output_file": "pocket_rmsf_overlay.png",
        "title": "Pocket RMSF — all simulations",
        "xlabel": "Pocket residue index",
        "ylabel": "RMSF (Å)",
    },
    "ligand_rmsf": {
        "file_pattern": "ligand_rmsf",
        "output_file": "ligand_rmsf_overlay.png",
        "title": "Ligand RMSF — all simulations",
        "xlabel": "Ligand atom index",
        "ylabel": "RMSF (Å)",
    },
    "reference_pocket_rmsf": {
        "file_pattern": "reference_pocket_rmsf",
        "output_file": "reference_pocket_rmsf_overlay.png",
        "title": "Reference pocket RMSF — all simulations",
        "xlabel": "Reference-aligned pocket index",
        "ylabel": "RMSF (Å)",
    },
}


def _find_binding_rmsf_file(sim_dir: str, profile_type: str) -> Optional[str]:
    cfg = _BINDING_RMSF_CONFIG.get(profile_type, {})
    pattern = cfg.get("file_pattern", profile_type)
    for root in (str(Path(sim_dir) / "analysis"), sim_dir):
        hit = _find_metric_file(root, pattern)
        if hit:
            return hit
    return None


def _plot_rmsf_profiles_overlay(
    profiles: List[Tuple[str, Dict[str, Any]]],
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Tuple[int, int] = (10, 5),
    dpi: int = 200,
) -> bool:
    """Overlay pocket/ligand RMSF profiles (ordinal x) for multiple simulations."""
    if not HAS_MATPLOTLIB or not profiles:
        return False

    fig, ax = plt.subplots(figsize=figsize)
    default_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    for i, (label, profile) in enumerate(profiles):
        color = default_colors[i % len(default_colors)]
        ax.plot(
            profile["x_positions"],
            profile["y_values"],
            label=label,
            color=color,
            linewidth=1.4,
            alpha=0.88,
        )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=8, ncol=min(len(profiles), 2))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


@tool
def run_combined_binding_rmsf_overlay(
    sim_dirs: List[str],
    labels: List[str],
    working_dir: str,
    profile_type: str = "pocket_rmsf",
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Overlay pocket or ligand RMSF profiles across holo simulations.

    Each simulation contributes one line (ordinal pocket-residue or ligand-atom
    index on x). Residue IDs / atom names differ per protein, so x is the
    index within that pocket/ligand selection, not a shared sequence number.

    Args:
        sim_dirs: Per-simulation working directories.
        labels: Legend labels (prefer protein display names).
        working_dir: Combined analysis output directory.
        profile_type: ``pocket_rmsf`` or ``ligand_rmsf``.
        output_file: Optional PNG filename override.

    Standard outputs: ``pocket_rmsf_overlay.png``, ``ligand_rmsf_overlay.png``.
    """
    from src.analysis.data_plotter import _parse_rmsf_profile_dat

    if profile_type not in _BINDING_RMSF_CONFIG:
        return {
            "success": False,
            "error": f"Unknown profile_type {profile_type!r}; use pocket_rmsf or ligand_rmsf",
        }

    cfg = _BINDING_RMSF_CONFIG[profile_type]
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    out_name = output_file or cfg["output_file"]
    output_path = Path(working_dir) / out_name

    profiles: List[Tuple[str, Dict[str, Any]]] = []
    missing: List[str] = []

    for sim_dir, label in zip(sim_dirs, labels):
        hit = _find_binding_rmsf_file(sim_dir, profile_type)
        if not hit and not is_holo_simulation(sim_dir, label):
            missing.append(label)
            continue
        if not hit:
            missing.append(label)
            continue
        profile = _parse_rmsf_profile_dat(hit)
        if not profile:
            missing.append(label)
            continue
        profiles.append((label, profile))

    if len(profiles) < 1:
        return {
            "success": False,
            "message": f"No {profile_type} data found for overlay. Missing: {missing}",
            "missing": missing,
        }

    ok = _plot_rmsf_profiles_overlay(
        profiles,
        output_path,
        title=cfg["title"],
        xlabel=cfg["xlabel"],
        ylabel=cfg["ylabel"],
    )
    if not ok:
        return {"success": False, "error": "matplotlib unavailable or plot failed"}

    try:
        from src.analysis.summary_logger import append_analysis_summary
        append_analysis_summary(
            working_dir=working_dir,
            analysis_type=f"Combined_{profile_type}",
            statistics={"n_simulations": len(profiles)},
            files={"overlay_plot": str(output_path.resolve())},
            metadata={"simulations": [p[0] for p in profiles], "missing": missing},
        )
    except Exception as exc:
        logger.warning("run_combined_binding_rmsf_overlay: summary failed: %s", exc)

    return {
        "success": True,
        "message": f"{profile_type} overlay: {len(profiles)} simulation(s) → {out_name}",
        "output_path": str(output_path.resolve()),
        "output_file": out_name,
        "missing": missing,
        "profile_type": profile_type,
        "n_simulations": len(profiles),
    }

