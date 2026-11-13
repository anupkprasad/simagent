"""
HPC File Copy Tool - Copy simulation files to HPC working directory
"""
import os
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from langchain.tools import tool

from src.hpc.force_field_bundle import bundle_force_fields_for_topology

logger = logging.getLogger(__name__)


@tool
def copy_simulation_files(
    source_dir: str,
    dest_dir: str,
    file_patterns: Optional[Union[str, List[str]]] = None,
    create_dest: bool = True
) -> Dict[str, Any]:
    """
    Copy simulation files from setup directory to HPC working directory.
    
    Args:
        source_dir: Source directory (e.g., working_dir/simsetup)
        dest_dir: Destination directory (e.g., working_dir/hpc)
        file_patterns: File patterns to copy (default: *.gro, *.top, *.mdp, *.itp)
        create_dest: Create destination directory if it doesn't exist
        
    Returns:
        Dict with success status and list of copied files
    """
    try:
        source_path = Path(source_dir)
        dest_path = Path(dest_dir)
        
        if not source_path.exists():
            return {
                "success": False,
                "error": f"Source directory does not exist: {source_dir}"
            }
        
        if create_dest:
            dest_path.mkdir(parents=True, exist_ok=True)
        
        file_patterns = coerce_file_patterns(file_patterns)
        if file_patterns is None:
            file_patterns = ["*.gro", "*.top", "*.mdp", "*.itp"]
        
        copied_files = []
        for pattern in file_patterns:
            for file_path in source_path.glob(pattern):
                dest_file = dest_path / file_path.name
                shutil.copy2(file_path, dest_file)
                copied_files.append({
                    "source": str(file_path),
                    "destination": str(dest_file),
                    "size": file_path.stat().st_size
                })
                logger.info(f"Copied: {file_path.name} -> {dest_dir}")
        
        ff_copied = bundle_force_fields_for_topology(source_path, dest_path)
        for entry in ff_copied:
            copied_files.append(entry)

        if not copied_files:
            return {
                "success": False,
                "error": f"No files matching patterns {file_patterns} found in {source_dir}"
            }

        msg = f"Copied {len(copied_files)} file(s) to {dest_dir}"
        if ff_copied:
            msg += f" (including {len(ff_copied)} force-field bundle(s))"

        return {
            "success": True,
            "files_copied": len(copied_files),
            "copied_files": copied_files,
            "force_fields_bundled": [e.get("force_field") for e in ff_copied],
            "destination": str(dest_path),
            "message": msg,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"File copy failed: {e}"
        }


def coerce_file_patterns(file_patterns: Any) -> Optional[List[str]]:
    """Accept a pattern string from an LLM plan as well as a list.

    ``copy_simulation_files`` is a LangChain tool. A plan that passes
    ``file_patterns='*'`` used to fail validation before the copy ran.
    """
    if file_patterns is None:
        return None
    if isinstance(file_patterns, str):
        parts = [part.strip() for part in file_patterns.split(",") if part.strip()]
        return parts or None
    if isinstance(file_patterns, (list, tuple)):
        patterns: List[str] = []
        for item in file_patterns:
            if isinstance(item, str):
                patterns.extend(
                    part.strip() for part in item.split(",") if part.strip()
                )
            elif item is not None and str(item).strip():
                patterns.append(str(item).strip())
        return patterns or None
    text = str(file_patterns).strip()
    return [text] if text else None
