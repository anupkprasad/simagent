"""
Data Plotter - Visualization tools for scientific data analysis

Creates publication-quality 2D and 3D plots for RMSD, RMSF, Rg, Energy, COM,
and other metrics.  Supports single-panel, multi-panel (subplot), combined-column,
and 3D scatter/trajectory layouts.
"""
import csv
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from langchain.tools import tool

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - plotting will be disabled")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Hardcoded analysis type inference map
_ANALYSIS_TYPE_MAP = {
    "rmsd": "RMSD",
    "rmsf": "RMSF",
    "rg": "Radius_of_Gyration",
    "gyration": "Radius_of_Gyration",
    "energy": "Energy",
    "sasa": "SASA",
    "dssp": "DSSP_SecondaryStructure",
    "com": "COM_Analysis",
    "min_distance": "Min_Heavy_Atom_Distance",
    "nearby": "Nearby_Residues",
}


def _infer_analysis_type(output_file: str, data_files: List[str], working_dir: Optional[str] = None) -> Optional[str]:
    """Infer analysis type from file names, falling back to summary file lookup."""
    # Strategy 1: match data files against existing summary entries (most precise)
    if working_dir:
        try:
            from .summary_logger import infer_analysis_type_from_files
            result = infer_analysis_type_from_files(working_dir, data_files)
            if result:
                return result
        except Exception:
            pass

    # Strategy 2: keyword matching against file names
    candidates = [output_file.lower()] + [f.lower() for f in data_files if f]
    for name in candidates:
        for keyword, atype in _ANALYSIS_TYPE_MAP.items():
            if keyword in name:
                return atype

    return None


_NON_NUMERIC_HEADER_TOKENS = frozenset({
    "resname", "atomname", "name", "chain", "segid", "element",
})


def _try_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _is_numeric_header(name: str) -> bool:
    """Return True when a header column should be treated as numeric."""
    normalized = name.strip().lower().replace("(", " ").replace(")", " ")
    tokens = normalized.split()
    if not tokens:
        return False
    if any(t in _NON_NUMERIC_HEADER_TOKENS for t in tokens):
        return False
    if "name" in normalized and "index" not in normalized:
        return False
    return True


def _parse_mixed_numeric_dat(
    file_path: str,
    header_cols: List[str],
    row_parts: List[List[str]],
) -> Tuple[List[List[float]], List[str]]:
    """
    Parse tab/space .dat files with string columns (ResName, AtomName, etc.).

    Keeps only numeric columns for plotting (e.g. Residue + RMSF).
    """
    if not row_parts:
        return [], header_cols

    numeric_indices: List[int] = []
    if header_cols:
        for i, name in enumerate(header_cols):
            if _is_numeric_header(name):
                numeric_indices.append(i)
    if not numeric_indices:
        # Infer from first data row: keep fields that parse as float.
        numeric_indices = [
            i for i, token in enumerate(row_parts[0])
            if _try_float(token) is not None
        ]

    if not numeric_indices:
        return [], header_cols

    data_columns = [[] for _ in numeric_indices]
    column_names = [
        header_cols[i] if i < len(header_cols) else f"Column {i + 1}"
        for i in numeric_indices
    ]

    for parts in row_parts:
        for col_idx, src_idx in enumerate(numeric_indices):
            if src_idx >= len(parts):
                continue
            val = _try_float(parts[src_idx])
            if val is not None:
                data_columns[col_idx].append(val)

    return data_columns, column_names


def _split_data_line(line: str, is_csv: bool) -> List[str]:
    if is_csv:
        return [p.strip() for p in line.split(",")]
    if "\t" in line:
        return [p.strip() for p in line.split("\t")]
    return line.split()


def _header_index(header_cols: List[str], *keywords: str) -> Optional[int]:
    lower = [h.lower() for h in header_cols]
    for kw in keywords:
        for i, name in enumerate(lower):
            if kw in name:
                return i
    return None


def _parse_rmsf_profile_dat(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse pocket/ligand RMSF .dat files that mix numeric and string columns.

    Returns profile metadata for categorical x-axis plotting.
    """
    header_cols: List[str] = []
    rows: List[List[str]] = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("@"):
                continue
            if line.startswith("#"):
                header_text = line[1:].strip()
                header_cols = _split_data_line(header_text, is_csv=False)
                continue
            parts = _split_data_line(line, is_csv=False)
            if parts:
                rows.append(parts)

    if not rows:
        return None

    stem = Path(file_path).stem.lower()
    if not header_cols:
        return None

    rmsf_idx = _header_index(header_cols, "rmsf")
    if rmsf_idx is None:
        rmsf_idx = len(header_cols) - 1

    is_pocket = (
        "pocket_rmsf" in stem
        or _header_index(header_cols, "residue") is not None
    )
    is_ligand = (
        "ligand_rmsf" in stem
        or _header_index(header_cols, "atomname") is not None
    )

    if is_pocket and not is_ligand:
        resid_idx = _header_index(header_cols, "residue")
        resname_idx = _header_index(header_cols, "resname")
        if resid_idx is None or resname_idx is None:
            return None
        x_tick_labels: List[str] = []
        y_values: List[float] = []
        for parts in rows:
            resid = _try_float(parts[resid_idx])
            rmsf = _try_float(parts[rmsf_idx]) if rmsf_idx < len(parts) else None
            if resid is None or rmsf is None:
                continue
            resname = parts[resname_idx] if resname_idx < len(parts) else ""
            x_tick_labels.append(f"{int(resid)}{resname}")
            y_values.append(rmsf)
        if not y_values:
            return None
        x_positions = list(range(len(y_values)))
        return {
            "kind": "pocket",
            "x_positions": x_positions,
            "x_tick_labels": x_tick_labels,
            "y_values": y_values,
            "default_plot_type": "line",
        }

    if is_ligand:
        atomname_idx = _header_index(header_cols, "atomname")
        if atomname_idx is None:
            return None
        x_tick_labels = []
        y_values = []
        for parts in rows:
            if atomname_idx >= len(parts):
                continue
            rmsf = _try_float(parts[rmsf_idx]) if rmsf_idx < len(parts) else None
            if rmsf is None:
                continue
            x_tick_labels.append(parts[atomname_idx])
            y_values.append(rmsf)
        if not y_values:
            return None
        x_positions = list(range(len(y_values)))
        return {
            "kind": "ligand",
            "x_positions": x_positions,
            "x_tick_labels": x_tick_labels,
            "y_values": y_values,
            "default_plot_type": "bar",
        }

    return None


def _parse_reference_pocket_rmsf_aligned(
    file_path: str,
    residue_map_path: str,
    display_label: str,
) -> Optional[Dict[str, Any]]:
    """
    Parse reference pocket RMSF with x-axis = reference-alignment index.

    Maps each simulation's PDB residue IDs to the shared consensus index
    from ``reference_pocket_residue_map.csv`` so mutants/deletions align across
    proteins in cluster overlay plots.
    """
    base_profile = _parse_rmsf_profile_dat(file_path)
    if not base_profile:
        return None

    map_path = Path(residue_map_path)
    if not map_path.is_file():
        return base_profile

    resid_to_rmsf: Dict[int, float] = {}
    for tick, y in zip(base_profile["x_tick_labels"], base_profile["y_values"]):
        resid_str = "".join(ch for ch in str(tick) if ch.isdigit())
        if not resid_str:
            continue
        resid_to_rmsf[int(resid_str)] = float(y)

    resid_col = f"{display_label}_resid"
    aa_col = f"{display_label}_aa"
    consensus_x: List[int] = []
    y_values: List[float] = []
    x_tick_labels: List[str] = []

    with open(map_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if resid_col not in (reader.fieldnames or []):
            return base_profile
        for row in reader:
            raw_idx = (row.get("consensus_index") or "").strip()
            raw_resid = (row.get(resid_col) or "").strip()
            if not raw_idx or not raw_resid:
                continue
            try:
                cidx = int(float(raw_idx))
                resid = int(float(raw_resid))
            except ValueError:
                continue
            if resid not in resid_to_rmsf:
                continue
            aa = (row.get(aa_col) or "").strip()
            ref_aa = (row.get("reference_aa") or "").strip()
            consensus_x.append(cidx)
            y_values.append(resid_to_rmsf[resid])
            x_tick_labels.append(f"{cidx}:{aa or ref_aa}")

    if not y_values:
        return base_profile

    order = sorted(range(len(consensus_x)), key=lambda i: consensus_x[i])
    consensus_x = [consensus_x[i] for i in order]
    y_values = [y_values[i] for i in order]
    x_tick_labels = [x_tick_labels[i] for i in order]

    return {
        "kind": "reference_pocket",
        "x_positions": consensus_x,
        "x_tick_labels": x_tick_labels,
        "y_values": y_values,
        "default_plot_type": "line",
    }


def _plot_rmsf_profile_on_axes(
    ax,
    profile: Dict[str, Any],
    *,
    plot_type: str,
    color: Optional[str],
    label: str,
) -> None:
    """Draw pocket or ligand RMSF with categorical / residue-id x ticks."""
    x_pos = profile["x_positions"]
    y_vals = profile["y_values"]
    tick_labels = profile["x_tick_labels"]
    kind = profile["kind"]
    use_bar = plot_type == "bar" or (plot_type == "line" and kind == "ligand" and profile.get("default_plot_type") == "bar")

    if use_bar:
        ax.bar(x_pos, y_vals, label=label, color=color or "#1f77b4", alpha=0.85)
    else:
        ax.plot(
            x_pos,
            y_vals,
            label=label,
            color=color or "#1f77b4",
            linewidth=2,
            marker="o",
            markersize=4,
        )

    ax.set_xticks(x_pos)
    if kind == "pocket":
        rotation = 45
        ha = "center"
    elif kind == "ligand":
        rotation = 90
        ha = "right"
    else:
        rotation = 0
        ha = "center"
    fontsize = 7 if len(tick_labels) > 20 else 8
    ax.set_xticklabels(
        tick_labels,
        rotation=rotation,
        ha=ha,
        fontsize=fontsize,
    )
    if kind == "pocket":
        ax.set_xlim(-0.5, len(x_pos) - 0.5)


_SERIES_LABELS: Dict[str, str] = {
    "n_hbonds": "H-bonds",
    "n_contacts": "Heavy-atom contact pairs",
    "min_contact_distance_a": "Min contact distance (Å)",
    "bound": "Bound (1=bound, 0=unbound)",
}


def _series_label(column_names: List[str], y_idx: int, file_label: str) -> str:
    if y_idx < len(column_names):
        col_key = column_names[y_idx].strip().lower()
        if col_key in _SERIES_LABELS:
            return _SERIES_LABELS[col_key]
        return f"{file_label} ({column_names[y_idx]})"
    return file_label


def _is_dccm_difference_table(data_file: str, column_names: List[str]) -> bool:
    """True for a flattened apo/holo ΔDCCM table, not a time series."""
    name = Path(data_file).name.lower()
    if "dccm" not in name:
        return False
    if any(tok in name for tok in ("diff", "delta", "difference")):
        return True
    lower = {n.strip().lower() for n in column_names}
    return "correlation" in lower and "residue_i" in lower


def _running_average(values: List[float], window: int) -> List[float]:
    """Centered moving average. Edges use the points that exist."""
    n = len(values)
    if n == 0 or window < 3 or n < 3:
        return list(values)
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 else n - 1)
    half = window // 2
    cumulative = [0.0]
    for value in values:
        cumulative.append(cumulative[-1] + float(value))
    averaged = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        averaged.append((cumulative[end] - cumulative[start]) / (end - start))
    return averaged


def _dccm_running_window(n_points: int) -> int:
    """Wide enough to stand out from the raw pair-by-pair trace."""
    window = max(51, n_points // 120)
    if window % 2 == 0:
        window += 1
    return window


def _time_axis_is_picoseconds(column_name: str) -> bool:
    """True when a time column header indicates picoseconds."""
    low = (column_name or "").strip().lower()
    if not low:
        return False
    compact = low.replace(" ", "").replace("_", "")
    if "(ps)" in low or "time(ps)" in compact:
        return True
    if compact in {"timeps", "tps"} or compact.endswith("timeps"):
        return True
    return False


def _normalize_time_x_to_ns(
    x_data: List[float],
    column_names: List[str],
    xi: int,
    xlabel: Optional[str] = None,
) -> Tuple[List[float], Optional[str]]:
    """Convert picosecond time axes to nanoseconds for consistent ``Time (ns)`` plots.

    Triggers when the file header is ``Time(ps)`` / ``time_ps``, or when the
    caller requested an ns label while the column is clearly time-in-ps
    (large values, no ``ns`` in the column name).
    """
    if not x_data:
        return x_data, xlabel
    col = column_names[xi] if xi < len(column_names) else ""
    col_low = col.lower()
    want_ns = bool(xlabel) and "ns" in xlabel.lower() and "(ps)" not in xlabel.lower()
    is_ps_header = _time_axis_is_picoseconds(col)
    looks_like_ps = (
        want_ns
        and "time" in col_low
        and "ns" not in col_low
        and max(x_data) > 500.0
    )
    if is_ps_header or looks_like_ps:
        return [float(v) / 1000.0 for v in x_data], "Time (ns)"
    return x_data, xlabel


def _detect_plot_columns(
    column_names: List[str],
    data_columns: List[List[float]],
    x_col: Optional[int] = None,
    y_col: Optional[int] = None,
) -> Tuple[int, List[int]]:
    """Pick x and one-or-more y column indices for trajectory CSVs / .dat files.

    Handles the common MD layout ``frame, time_ns, metric(s)`` and named columns
    such as ``distance_angstrom``, ``n_contacts``, and ``n_hbonds``.
    """
    if x_col is not None:
        xi = int(x_col)
        if y_col is not None:
            return xi, [int(y_col)]
        fallback = 1 if xi == 0 and len(data_columns) > 1 else 0
        return xi, [fallback]

    lower = [n.strip().lower() for n in column_names]

    xi: Optional[int] = None
    for i, name in enumerate(lower):
        if name in ("time_ns", "time (ns)", "t_ns") or (
            "time" in name and "frame" not in name
        ):
            xi = i
            break
    if xi is None and len(data_columns) >= 3 and lower:
        if lower[0] in ("frame", "frames", "frame_index", "index"):
            xi = 1
    if xi is None:
        xi = 0

    skip_y = {"frame", "frames", "frame_index", "index", "time_ns", "time", "t_ns"}

    # Contacts CSV: plot H-bonds and contact counts on the same axes.
    if y_col is None and "n_contacts" in lower:
        y_indices: List[int] = []
        if "n_hbonds" in lower:
            y_indices.append(lower.index("n_hbonds"))
        y_indices.append(lower.index("n_contacts"))
        return xi, y_indices

    # Residence CSV: min heavy-atom distance is more informative than a flat bound mask.
    if y_col is None and "min_contact_distance_a" in lower:
        return xi, [lower.index("min_contact_distance_a")]

    if y_col is not None:
        return xi, [int(y_col)]

    y_priority = (
        "distance_angstrom",
        "distance_a",
        "distance",
        "sasa_nm2",
        "sasa",
        "fraction_bound",
        "residence",
        "rmsd",
        "rmsf",
        "rg",
        "gyration",
        "energy",
        "value",
        "n_contacts",
        "contacts",
        "n_hbonds",
        "hbonds",
        "bound",
    )
    for pat in y_priority:
        for i, name in enumerate(lower):
            if i == xi or name in skip_y:
                continue
            if name == pat or (len(pat) > 4 and pat in name):
                return xi, [i]

    candidates = [i for i in range(len(data_columns)) if i != xi]
    if candidates:
        return xi, [candidates[-1]]
    return xi, [1 if len(data_columns) > 1 else 0]


def parse_data_file(file_path: str) -> Tuple[List[List[float]], List[str]]:
    """
    Parse data file (.xvg, .dat, .csv) and extract columns.
    
    Args:
        file_path: Path to data file
        
    Returns:
        Tuple of (data_columns, column_names)
    """
    data_columns = []
    column_names = []
    mixed_row_parts: List[List[str]] = []
    mixed_header: List[str] = []
    
    # Detect delimiter from file extension
    is_csv = file_path.lower().endswith('.csv')
    csv_header_read = False
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse headers (XVG format)
            if line.startswith('@'):
                if 'xaxis label' in line.lower():
                    # Extract x-axis label
                    label = line.split('"')[1] if '"' in line else "X"
                    if not column_names:
                        column_names.append(label)
                elif 'yaxis label' in line.lower():
                    # Extract y-axis label
                    label = line.split('"')[1] if '"' in line else "Y"
                    if len(column_names) == 1:
                        column_names.append(label)
                continue
            
            # Skip comment lines
            if line.startswith('#'):
                # Try to extract column names from header
                header_text = line[1:].strip()
                if any(k in header_text for k in ("Frame", "Time", "Residue", "Atom", "RMSF")):
                    parts = header_text.split('\t') if '\t' in header_text else header_text.split()
                    mixed_header = [p.strip() for p in parts]
                    column_names = list(mixed_header)
                continue
            
            parts = _split_data_line(line, is_csv)
            
            # Parse data
            try:
                values = [float(p) for p in parts]
                
                # Initialize columns on first data row
                if not data_columns:
                    data_columns = [[] for _ in values]
                
                # Append values to respective columns
                for i, val in enumerate(values):
                    if i < len(data_columns):
                        data_columns[i].append(val)
                        
            except ValueError:
                # CSV files: first non-comment line is often the column header.
                if is_csv and not data_columns and not csv_header_read:
                    column_names = [p.strip() for p in parts]
                    csv_header_read = True
                    continue
                mixed_row_parts.append(parts)
                continue

    if not data_columns and mixed_row_parts:
        data_columns, column_names = _parse_mixed_numeric_dat(
            file_path, mixed_header or column_names, mixed_row_parts
        )
    
    # Set default column names if not found
    if not column_names:
        column_names = [f"Column {i+1}" for i in range(len(data_columns))]
    elif len(column_names) < len(data_columns):
        # Add missing column names
        for i in range(len(column_names), len(data_columns)):
            column_names.append(f"Column {i+1}")
    
    return data_columns, column_names


@tool
def plot_data(
    data_files: List[str],
    output_file: str,
    plot_type: str = "line",
    x_col: Optional[int] = None,
    y_col: Optional[int] = None,
    titles: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    working_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plot analysis data from one or more data files (single 2D panel).
    
    Creates a single-panel 2D plot for visualizing RMSD, RMSF, Rg, energy, or
    other time-series / per-residue data.  Supports overlay of multiple datasets
    on the same axes.
    
    Args:
        data_files: List of data file paths to plot (.xvg, .dat, .csv)
        output_file: Output image filename only (e.g., "plot.png") - saved in working_dir
        plot_type: Type of plot ("line", "scatter", "bar") - default: "line"
        x_col: Column index for x-axis data (default: 0 = first column)
        y_col: Column index for y-axis data (default: 1 = second column)
        titles: Plot title (optional)
        xlabel: X-axis label (optional, auto-detected from file if available)
        ylabel: Y-axis label (optional, auto-detected from file if available)
        labels: Legend labels for each data file (optional)
        colors: Line/marker colors for each dataset (optional)
        figsize: Figure size as (width, height) in inches - default: (10, 6)
        dpi: Resolution in dots per inch - default: 300
        working_dir: Working directory for analysis (files will be written here)
        
    Returns:
        Dict with plotting results
    """
    try:
        if not HAS_MATPLOTLIB:
            return {
                "success": False,
                "error": "matplotlib not available - cannot create plots"
            }
        
        # Convert file paths to absolute before changing directory
        abs_data_files = []
        original_dir = os.getcwd()
        
        for data_file in data_files:
            if os.path.isabs(data_file):
                abs_data_files.append(data_file)
            else:
                # If working_dir is specified, resolve relative paths from there
                if working_dir:
                    abs_data_files.append(os.path.abspath(os.path.join(working_dir, data_file)))
                else:
                    abs_data_files.append(os.path.abspath(data_file))
        
        # Validate that all files exist
        for abs_path, rel_path in zip(abs_data_files, data_files):
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"Data file not found: {rel_path}"
                }
        
        # Setup working directory for output
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)
        
        # -------------------------------------------------------------------
        # Auto-detect geometric (x, y, z) data → delegate to plot_3d
        # -------------------------------------------------------------------
        first_cols, first_names = parse_data_file(abs_data_files[0])
        lower_names = [n.strip().lower() for n in first_names]
        # FEL grids / feature tables are not 1D XY series — refuse line plots.
        if (
            "free_energy_kj_mol" in lower_names
            or (
                any(n.startswith("pc") for n in lower_names)
                and any("free_energy" in n or n == "probability" for n in lower_names)
            )
            or Path(abs_data_files[0]).name.lower() in {
                "fel_features.csv",
                "fel_basins.csv",
                "fel_pc1_pc2_grid.csv",
            }
            or (
                "fel" in Path(abs_data_files[0]).name.lower()
                and "grid" in Path(abs_data_files[0]).name.lower()
            )
        ):
            return {
                "success": False,
                "error": (
                    "Refusing plot_md_data on FEL grid/feature tables. "
                    "Use fel_basins.png from analyze_fel_landscape_features; "
                    "feature info is in fel_features.json/csv only."
                ),
            }
        detected_xyz = _detect_xyz_columns(first_names, first_cols)
        if detected_xyz is not None:
            logger.info("Detected geometric (x,y,z) columns – switching to 3D plot")
            if working_dir:
                os.chdir(original_dir)
            return plot_3d.invoke({
                "data_files": data_files,
                "output_file": output_file,
                "x_col": detected_xyz["x"],
                "y_col": detected_xyz["y"],
                "z_col": detected_xyz["z"],
                "plot_type": "scatter" if plot_type == "scatter" else "line",
                "titles": titles,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "labels": labels,
                "colors": colors,
                "figsize": figsize,
                "dpi": dpi,
                "working_dir": working_dir,
            })
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Default colors if not provided
        if colors is None:
            colors = [f"C{i}" for i in range(len(data_files))]
        
        # Plot each data file (use absolute paths)
        n_plotted = 0
        multi_series = False
        for idx, (abs_file, rel_file) in enumerate(zip(abs_data_files, data_files)):
            profile = _parse_rmsf_profile_dat(abs_file)
            if profile:
                if labels is not None and idx < len(labels):
                    label = labels[idx]
                else:
                    label = Path(rel_file).stem
                color = colors[idx] if idx < len(colors) else None
                effective_plot_type = plot_type
                if plot_type == "line" and profile.get("default_plot_type") == "bar":
                    effective_plot_type = "bar"
                _plot_rmsf_profile_on_axes(
                    ax,
                    profile,
                    plot_type=effective_plot_type,
                    color=color,
                    label=label,
                )
                n_plotted += 1
                if idx == 0 and not xlabel:
                    xlabel = "Residue" if profile["kind"] == "pocket" else "Atom"
                if idx == 0 and not ylabel:
                    ylabel = "RMSF (Å)"
                continue

            # Parse data
            data_columns, column_names = parse_data_file(abs_file)
            
            xi, y_indices = _detect_plot_columns(
                column_names, data_columns, x_col=x_col, y_col=y_col
            )
            if len(y_indices) > 1:
                multi_series = True
            
            if len(data_columns) <= xi:
                logger.warning(
                    f"Insufficient data columns in {rel_file} (need x column {xi}, "
                    f"have {len(data_columns)}), skipping"
                )
                continue
            
            x_data = data_columns[xi]
            if not x_data:
                logger.warning(f"No plottable numeric data in {rel_file}, skipping")
                continue

            x_data, xlabel = _normalize_time_x_to_ns(
                x_data, column_names, xi, xlabel=xlabel
            )
            
            # Determine label base for this file
            if labels is not None and idx < len(labels):
                file_label = labels[idx]
            else:
                file_label = Path(rel_file).stem
            
            plotted_any = False
            for series_i, y_idx in enumerate(y_indices):
                if len(data_columns) <= y_idx:
                    logger.warning(
                        f"Insufficient data columns in {rel_file} (need y column {y_idx}, "
                        f"have {len(data_columns)}), skipping y series"
                    )
                    continue
                y_data = data_columns[y_idx]
                if not y_data:
                    continue

                if len(y_indices) > 1 and y_idx < len(column_names):
                    label = _series_label(column_names, y_idx, file_label)
                else:
                    label = file_label

                if len(y_indices) > 1:
                    series_color = f"C{series_i}"
                else:
                    series_color = colors[idx] if idx < len(colors) else None

                if plot_type == "line" and _is_dccm_difference_table(abs_file, column_names):
                    # Raw ΔC is a dense vertical band. Keep that trace light
                    # and draw the running average on top so the trend can be read.
                    ax.plot(
                        x_data,
                        y_data,
                        label="_nolegend_",
                        color="#9ecae1",
                        linewidth=0.45,
                        alpha=0.18,
                        zorder=1,
                    )
                    averaged = _running_average(y_data, _dccm_running_window(len(y_data)))
                    ax.plot(
                        x_data,
                        averaged,
                        label="_nolegend_",
                        color="white",
                        linewidth=5.0,
                        solid_capstyle="round",
                        zorder=2,
                    )
                    ax.plot(
                        x_data,
                        averaged,
                        label="_nolegend_",
                        color="#a50f15",
                        linewidth=2.2,
                        solid_capstyle="round",
                        zorder=3,
                    )
                    from matplotlib.lines import Line2D
                    ax.legend(
                        handles=[
                            Line2D([0], [0], color="#9ecae1", linewidth=2.0, label="ΔC"),
                            Line2D([0], [0], color="#a50f15", linewidth=2.4, label="Running average"),
                        ],
                        frameon=True,
                        fontsize=10,
                    )
                    multi_series = True
                    if not xlabel or "time" in str(xlabel).lower():
                        xlabel = "Residue"
                    if not ylabel or str(ylabel).lower() in {"value", "correlation"}:
                        ylabel = "Δ correlation"
                elif plot_type == "line":
                    ax.plot(x_data, y_data, label=label, color=series_color, linewidth=2)
                elif plot_type == "scatter":
                    ax.scatter(x_data, y_data, label=label, color=series_color, alpha=0.6)
                elif plot_type == "bar":
                    ax.bar(x_data, y_data, label=label, color=series_color, alpha=0.7)
                plotted_any = True

                if idx == 0 and not ylabel and y_idx < len(column_names):
                    col_key = column_names[y_idx].strip().lower()
                    ylabel = _SERIES_LABELS.get(col_key, column_names[y_idx])
                    if len(y_indices) > 1:
                        ylabel = "Count"

            if not plotted_any:
                continue
            n_plotted += 1
            
            # Auto-detect axis labels from first file (after time-unit normalize)
            if idx == 0 and not xlabel and xi < len(column_names):
                xlabel = column_names[xi]
                if _time_axis_is_picoseconds(xlabel):
                    xlabel = "Time (ns)"
            if idx == 0 and not ylabel and len(y_indices) == 1 and y_indices[0] < len(column_names):
                ylabel = column_names[y_indices[0]]

        if n_plotted == 0:
            plt.close(fig)
            if working_dir:
                os.chdir(original_dir)
            return {
                "success": True,
                "skipped": True,
                "warning": (
                    "No plottable numeric data found in input file(s); "
                    "skipped empty/odd table (e.g. H-bond occupancy)."
                ),
            }
        
        # Set labels
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        # Set title
        if titles is not None and len(titles) > 0:
            ax.set_title(titles[0], fontsize=14, fontweight='bold')
        
        # Add legend when multiple series or multiple files are plotted
        if len(data_files) > 1 or multi_series:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=True, shadow=True, fontsize=10)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure (use filename only, already in working_dir)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {output_file}")
        
        # Update analysis summary with plot file info
        if working_dir:
            try:
                from .summary_logger import update_analysis_summary_with_files
                
                analysis_type = _infer_analysis_type(output_file, data_files, working_dir)
                
                if analysis_type:
                    update_analysis_summary_with_files(
                        working_dir=working_dir,
                        analysis_type=analysis_type,
                        additional_files={"plot": output_file}
                    )
                    logger.info(f"Updated {analysis_type} summary with plot: {output_file}")
            except Exception as e:
                logger.warning(f"Failed to update summary with plot info: {e}")
            
            os.chdir(original_dir)
        
        return {
            "success": True,
            "output_file": output_file,
            "n_datasets": len(data_files),
            "plot_type": plot_type,
            "message": f"Successfully created {plot_type} plot with {len(data_files)} dataset(s)"
        }
        
    except Exception as e:
        logger.exception(f"Plotting failed: {e}")
        if working_dir and 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "success": False,
            "error": f"Plotting failed: {str(e)}"
        }


@tool
def plot_multipanel(
    data_files: List[str],
    output_file: str,
    layout: str = "vertical",
    titles: Optional[List[str]] = None,
    xlabels: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
    plot_types: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    working_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create multi-panel plots for comparing multiple analyses.
    
    Creates a figure with multiple subplots, one for each data file.
    Useful for comparing RMSD, RMSF, Rg, energy, or any 2D data side-by-side.
    
    Args:
        data_files: List of data file paths, one per panel (.xvg, .dat, .csv)
        output_file: Output image file path (.png, .pdf, .svg)
        layout: Subplot layout ("vertical", "horizontal", "grid") - default: "vertical"
        titles: List of subplot titles (optional)
        xlabels: List of x-axis labels for each subplot (optional)
        ylabels: List of y-axis labels for each subplot (optional)
        plot_types: List of plot types for each panel (optional, default: "line")
        colors: List of colors for each panel (optional)
        figsize: Figure size as (width, height) in inches (auto-sized if None)
        dpi: Resolution in dots per inch - default: 300
        working_dir: Working directory for analysis
        
    Returns:
        Dict with plotting results
    """
    try:
        if not HAS_MATPLOTLIB:
            return {
                "success": False,
                "error": "matplotlib not available - cannot create plots"
            }
        
        # Convert file paths to absolute before changing directory
        # This handles cases where paths include working_dir prefix
        abs_data_files = []
        original_dir = os.getcwd()
        
        for data_file in data_files:
            # If path is already absolute, use it
            if os.path.isabs(data_file):
                abs_data_files.append(data_file)
            else:
                # If working_dir is specified, resolve relative paths from there
                if working_dir:
                    abs_data_files.append(os.path.abspath(os.path.join(working_dir, data_file)))
                else:
                    abs_data_files.append(os.path.abspath(data_file))
        
        # Validate that all files exist (before any chdir)
        for abs_path, rel_path in zip(abs_data_files, data_files):
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"Data file not found: {rel_path}"
                }
        
        # Setup working directory for output
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)
        
        n_panels = len(data_files)
        
        # Determine subplot layout
        if layout == "vertical":
            nrows, ncols = n_panels, 1
            if not figsize:
                figsize = (10, 4 * n_panels)
        elif layout == "horizontal":
            nrows, ncols = 1, n_panels
            if not figsize:
                figsize = (6 * n_panels, 5)
        elif layout == "grid":
            ncols = int(np.ceil(np.sqrt(n_panels)))
            nrows = int(np.ceil(n_panels / ncols))
            if not figsize:
                figsize = (6 * ncols, 4 * nrows)
        else:
            if working_dir:
                os.chdir(original_dir)
            return {
                "success": False,
                "error": f"Invalid layout: {layout}. Use 'vertical', 'horizontal', or 'grid'"
            }
        
        # Create figure with subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
        
        # Ensure axes is iterable
        if n_panels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Default plot types
        if plot_types is None:
            plot_types = ["line"] * n_panels
        
        # Default colors
        if colors is None:
            colors = [f"C{i}" for i in range(n_panels)]
        
        # Plot each panel (use absolute paths for reading)
        for idx, (abs_file, rel_file, ax) in enumerate(zip(abs_data_files, data_files, axes)):
            # Parse data
            data_columns, column_names = parse_data_file(abs_file)
            
            if len(data_columns) < 2:
                logger.warning(f"Insufficient data columns in {rel_file}, skipping panel {idx+1}")
                ax.text(0.5, 0.5, f"No data\n{Path(rel_file).name}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            x_data = list(data_columns[0])
            y_data = data_columns[1]

            # New gyration.dat is Time(ns); legacy files may still be Time(ps).
            # Normalize so combined_metrics always plots Time (ns).
            xlabel_hint = None
            if xlabels is not None and idx < len(xlabels):
                xlabel_hint = xlabels[idx]
            elif len(column_names) > 0:
                xlabel_hint = column_names[0]
            x_data, xlabel_resolved = _normalize_time_x_to_ns(
                x_data, column_names, 0, xlabel=xlabel_hint
            )
            
            # Determine plot type
            plot_type = plot_types[idx] if idx < len(plot_types) else "line"
            
            # Determine color
            color = colors[idx] if idx < len(colors) else None
            
            # Plot based on type
            if plot_type == "line":
                ax.plot(x_data, y_data, color=color, linewidth=2)
            elif plot_type == "scatter":
                ax.scatter(x_data, y_data, color=color, alpha=0.6)
            elif plot_type == "bar":
                ax.bar(x_data, y_data, color=color, alpha=0.7)
            
            # Set labels
            if xlabel_resolved:
                ax.set_xlabel(xlabel_resolved, fontsize=11, fontweight='bold')
            elif xlabels is not None and idx < len(xlabels):
                ax.set_xlabel(xlabels[idx], fontsize=11, fontweight='bold')
            elif len(column_names) > 0:
                ax.set_xlabel(column_names[0], fontsize=11, fontweight='bold')
            
            if ylabels is not None and idx < len(ylabels):
                ax.set_ylabel(ylabels[idx], fontsize=11, fontweight='bold')
            elif len(column_names) > 1:
                ax.set_ylabel(column_names[1], fontsize=11, fontweight='bold')
            
            # Set title
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx], fontsize=12, fontweight='bold')
            else:
                ax.set_title(Path(rel_file).stem.replace('_', ' ').title(), 
                           fontsize=12, fontweight='bold')
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Hide extra subplots if grid layout
        for idx in range(n_panels, len(axes)):
            axes[idx].set_visible(False)
        
        # Adjust spacing
        plt.tight_layout()
        
        # Save figure (use filename only, already in working_dir)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Multi-panel plot saved to {output_file}")
        
        # Update analysis summary with plot file (try to infer type from data files)
        if working_dir:
            try:
                from .summary_logger import update_analysis_summary_with_files
                
                analysis_type = _infer_analysis_type(output_file, data_files, working_dir)
                
                if analysis_type:
                    update_analysis_summary_with_files(
                        working_dir=working_dir,
                        analysis_type=analysis_type,
                        additional_files={"multipanel_plot": output_file}
                    )
                    logger.info(f"Updated {analysis_type} summary with multipanel plot: {output_file}")
            except Exception as e:
                logger.warning(f"Could not update analysis summary with plot: {e}")
        
        if working_dir:
            os.chdir(original_dir)
        
        return {
            "success": True,
            "output_file": output_file,
            "n_panels": n_panels,
            "layout": layout,
            "message": f"Successfully created {n_panels}-panel {layout} plot"
        }
        
    except Exception as e:
        logger.exception(f"Multi-panel plotting failed: {e}")
        if working_dir and 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "success": False,
            "error": f"Multi-panel plotting failed: {str(e)}"
        }


@tool
def plot_combined_data(
    data_files: List[str],
    output_file: str,
    data_columns: Optional[List[int]] = None,
    subplot_arrangement: Optional[Tuple[int, int]] = None,
    shared_xaxis: bool = True,
    titles: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    working_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create combined plots from a single data file with multiple columns.
    
    Useful for plotting multiple energy terms, multiple properties, or
    comparing different metrics from the same simulation.
    
    Args:
        data_files: List containing one data file with multiple columns
        output_file: Output image file path (.png, .pdf, .svg)
        data_columns: List of column indices to plot (e.g., [1, 2, 3] for columns after x-axis)
        subplot_arrangement: (rows, cols) for subplot grid (auto if None)
        shared_xaxis: Whether to share x-axis across subplots - default: True
        titles: List of subplot titles (optional)
        xlabel: Common x-axis label (optional)
        ylabels: List of y-axis labels for each subplot (optional)
        colors: List of colors for each subplot (optional)
        figsize: Figure size as (width, height) in inches - default: (12, 8)
        dpi: Resolution in dots per inch - default: 300
        working_dir: Working directory for analysis
        
    Returns:
        Dict with plotting results
    """
    try:
        if not HAS_MATPLOTLIB:
            return {
                "success": False,
                "error": "matplotlib not available - cannot create plots"
            }
        
        # Convert file path to absolute before changing directory
        original_dir = os.getcwd()
        
        # Validate input file
        if not data_files or len(data_files) == 0:
            return {
                "success": False,
                "error": "No data files provided"
            }
        
        data_file = data_files[0]
        if os.path.isabs(data_file):
            abs_data_file = data_file
        else:
            # If working_dir is specified, resolve relative paths from there
            if working_dir:
                abs_data_file = os.path.abspath(os.path.join(working_dir, data_file))
            else:
                abs_data_file = os.path.abspath(data_file)
        
        if not os.path.exists(abs_data_file):
            return {
                "success": False,
                "error": f"Data file not found: {data_file}"
            }
        
        # Setup working directory for output
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)
        
        # Parse data file (use absolute path)
        data_columns_all, column_names = parse_data_file(abs_data_file)
        
        if len(data_columns_all) < 2:
            if working_dir:
                os.chdir(original_dir)
            return {
                "success": False,
                "error": "Insufficient data columns in file"
            }
        
        # Determine which columns to plot
        x_data = data_columns_all[0]
        
        if data_columns:
            y_data_list = [data_columns_all[i] for i in data_columns if i < len(data_columns_all)]
        else:
            # Plot all columns except the first (x-axis)
            y_data_list = data_columns_all[1:]
        
        n_plots = len(y_data_list)
        
        # Determine subplot arrangement
        if subplot_arrangement:
            nrows, ncols = subplot_arrangement
        else:
            ncols = 2 if n_plots > 2 else 1
            nrows = int(np.ceil(n_plots / ncols))
        
        # Create subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, 
                                sharex=shared_xaxis)
        
        # Ensure axes is iterable
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Default colors
        if colors is None:
            colors = [f"C{i}" for i in range(n_plots)]
        
        # Plot each y-column
        for idx, (y_data, ax) in enumerate(zip(y_data_list, axes)):
            color = colors[idx] if idx < len(colors) else None
            
            ax.plot(x_data, y_data, color=color, linewidth=2)
            
            # Set ylabel
            if ylabels is not None and idx < len(ylabels):
                ax.set_ylabel(ylabels[idx], fontsize=11, fontweight='bold')
            elif idx + 1 < len(column_names):
                ax.set_ylabel(column_names[idx + 1], fontsize=11, fontweight='bold')
            
            # Set title
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx], fontsize=12, fontweight='bold')
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Set xlabel for bottom row or if not sharing x-axis
            if not shared_xaxis or idx >= (nrows - 1) * ncols:
                if xlabel:
                    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
                elif len(column_names) > 0:
                    ax.set_xlabel(column_names[0], fontsize=11, fontweight='bold')
        
        # Hide extra subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        # Add common xlabel if shared x-axis
        if shared_xaxis and xlabel:
            fig.text(0.5, 0.02, xlabel, ha='center', fontsize=12, fontweight='bold')
        
        # Adjust spacing
        plt.tight_layout()
        
        # Save figure (use filename only, already in working_dir)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Combined plot saved to {output_file}")
        
        # Update analysis summary with plot file (try to infer type from data file)
        if working_dir:
            try:
                from .summary_logger import update_analysis_summary_with_files
                
                input_files = [data_file] if data_file else data_files
                analysis_type = _infer_analysis_type(output_file, input_files, working_dir)
                
                if analysis_type:
                    update_analysis_summary_with_files(
                        working_dir=working_dir,
                        analysis_type=analysis_type,
                        additional_files={"combined_plot": output_file}
                    )
                    logger.info(f"Updated {analysis_type} summary with combined plot: {output_file}")
            except Exception as e:
                logger.warning(f"Could not update analysis summary with plot: {e}")
        
        if working_dir:
            os.chdir(original_dir)
        
        return {
            "success": True,
            "output_file": output_file,
            "n_plots": n_plots,
            "message": f"Successfully created combined plot with {n_plots} subplots"
        }
        
    except Exception as e:
        logger.exception(f"Combined plotting failed: {e}")
        if working_dir and 'original_dir' in locals():
            os.chdir(original_dir)
        return {
            "success": False,
            "error": f"Combined plotting failed: {str(e)}"
        }


# ---------------------------------------------------------------------------
# Geometric column detection helpers
# ---------------------------------------------------------------------------

_GEOMETRIC_PATTERNS = {
    "x": {"x", "x_coord", "com_x", "pos_x"},
    "y": {"y", "y_coord", "com_y", "pos_y"},
    "z": {"z", "z_coord", "com_z", "pos_z"},
}


def _detect_xyz_columns(
    column_names: List[str], data_columns: List[List[float]]
) -> Optional[Dict[str, int]]:
    """Return {'x': idx, 'y': idx, 'z': idx} if three geometric columns are found."""
    name_lower = [n.lower().strip() for n in column_names]
    result: Dict[str, int] = {}
    for axis, patterns in _GEOMETRIC_PATTERNS.items():
        for i, name in enumerate(name_lower):
            if name in patterns:
                result[axis] = i
                break
    if len(result) == 3 and all(result[a] < len(data_columns) for a in "xyz"):
        return result
    return None


@tool
def plot_3d(
    data_files: List[str],
    output_file: str,
    x_col: Optional[int] = None,
    y_col: Optional[int] = None,
    z_col: Optional[int] = None,
    plot_type: str = "scatter",
    titles: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a 3D plot for geometric or spatial data (x, y, z).

    Automatically detects columns named x, y, z (case-insensitive) when
    column indices are not provided.  Useful for center-of-mass trajectories,
    spatial distributions, or any 3-variable dataset.

    Args:
        data_files: List of data file paths to plot (.csv, .dat, .xvg)
        output_file: Output image filename only (e.g., "com_3d.png") - saved in working_dir
        x_col: Column index for x-axis data (auto-detected if None)
        y_col: Column index for y-axis data (auto-detected if None)
        z_col: Column index for z-axis data (auto-detected if None)
        plot_type: "scatter" or "line" - default: "scatter"
        titles: Plot title (optional)
        xlabel: X-axis label (optional, auto-detected from column name)
        ylabel: Y-axis label (optional, auto-detected from column name)
        zlabel: Z-axis label (optional, auto-detected from column name)
        labels: Legend labels for each data file (optional)
        colors: Colors for each dataset (optional)
        figsize: Figure size as (width, height) in inches - default: (10, 8)
        dpi: Resolution in dots per inch - default: 300
        working_dir: Working directory for analysis (files will be written here)

    Returns:
        Dict with plotting results
    """
    try:
        if not HAS_MATPLOTLIB:
            return {"success": False, "error": "matplotlib not available"}

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        abs_data_files = []
        original_dir = os.getcwd()

        for data_file in data_files:
            if os.path.isabs(data_file):
                abs_data_files.append(data_file)
            elif working_dir:
                abs_data_files.append(os.path.abspath(os.path.join(working_dir, data_file)))
            else:
                abs_data_files.append(os.path.abspath(data_file))

        for abs_path, rel_path in zip(abs_data_files, data_files):
            if not os.path.exists(abs_path):
                return {"success": False, "error": f"Data file not found: {rel_path}"}

        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        if colors is None:
            colors = [f"C{i}" for i in range(len(data_files))]

        for idx, (abs_file, rel_file) in enumerate(zip(abs_data_files, data_files)):
            data_cols, col_names = parse_data_file(abs_file)

            # Resolve column indices -------------------------------------------
            # Handle string column names: convert to index by matching header
            def _resolve_col(val, names):
                if val is None:
                    return None
                if isinstance(val, int):
                    return val
                if isinstance(val, str):
                    # Try matching column header name (case-insensitive)
                    low = val.strip().lower()
                    for i, name in enumerate(names):
                        if name.lower() == low:
                            return i
                    # Try parsing as integer string
                    try:
                        return int(val)
                    except ValueError:
                        pass
                return None

            xi = _resolve_col(x_col, col_names)
            yi = _resolve_col(y_col, col_names)
            zi = _resolve_col(z_col, col_names)

            if xi is None or yi is None or zi is None:
                detected = _detect_xyz_columns(col_names, data_cols)
                if detected:
                    xi = xi if xi is not None else detected["x"]
                    yi = yi if yi is not None else detected["y"]
                    zi = zi if zi is not None else detected["z"]
                else:
                    # Fallback: last 3 columns (skip frame/time if present)
                    n = len(data_cols)
                    if n >= 3:
                        xi, yi, zi = n - 3, n - 2, n - 1
                    else:
                        if working_dir:
                            os.chdir(original_dir)
                        return {
                            "success": False,
                            "error": f"Need at least 3 data columns for 3D plot, found {n} in {rel_file}",
                        }

            if max(xi, yi, zi) >= len(data_cols):
                if working_dir:
                    os.chdir(original_dir)
                return {
                    "success": False,
                    "error": f"Column index out of range in {rel_file} (has {len(data_cols)} columns)",
                }

            x_data = data_cols[xi]
            y_data = data_cols[yi]
            z_data = data_cols[zi]

            label = labels[idx] if labels and idx < len(labels) else Path(rel_file).stem
            color = colors[idx] if idx < len(colors) else None

            if plot_type == "line":
                ax.plot(x_data, y_data, z_data, label=label, color=color, linewidth=2)
            else:
                ax.scatter(x_data, y_data, z_data, label=label, color=color, alpha=0.6, s=20)

            # Auto-detect axis labels from first file
            if idx == 0:
                if not xlabel and xi < len(col_names):
                    xlabel = col_names[xi]
                if not ylabel and yi < len(col_names):
                    ylabel = col_names[yi]
                if not zlabel and zi < len(col_names):
                    zlabel = col_names[zi]

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        if zlabel:
            ax.set_zlabel(zlabel, fontsize=12, fontweight="bold")

        if titles and len(titles) > 0:
            ax.set_title(titles[0], fontsize=14, fontweight="bold")

        if len(data_files) > 1:
            ax.legend(frameon=True, shadow=True, fontsize=10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"3D plot saved to {output_file}")

        if working_dir:
            try:
                from .summary_logger import update_analysis_summary_with_files

                analysis_type = _infer_analysis_type(output_file, data_files, working_dir)
                if analysis_type:
                    update_analysis_summary_with_files(
                        working_dir=working_dir,
                        analysis_type=analysis_type,
                        additional_files={"plot_3d": output_file},
                    )
            except Exception as e:
                logger.warning(f"Failed to update summary with 3D plot info: {e}")
            os.chdir(original_dir)

        return {
            "success": True,
            "output_file": output_file,
            "n_datasets": len(data_files),
            "plot_type": f"3d_{plot_type}",
            "message": f"Successfully created 3D {plot_type} plot with {len(data_files)} dataset(s)",
        }

    except Exception as e:
        logger.exception(f"3D plotting failed: {e}")
        if working_dir and "original_dir" in locals():
            os.chdir(original_dir)
        return {"success": False, "error": f"3D plotting failed: {str(e)}"}


# ---------------------------------------------------------------------------
# Backward-compatible aliases (old names → new names)
# ---------------------------------------------------------------------------
plot_md_data = plot_data
plot_md_multipanel = plot_multipanel
