"""
HPC Results Downloader - Download simulation results from HPC systems
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from langchain.tools import tool

logger = logging.getLogger(__name__)


@tool
def download_results(
    remote_dir: str,
    local_dir: str,
    file_patterns: Optional[Union[str, List[str]]] = None,
    remote_host: Optional[str] = None,
    remote_user: Optional[str] = None,
    ssh_key_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download simulation results from HPC system.
    
    Args:
        remote_dir: Remote directory path
        local_dir: Local download directory
        file_patterns: File patterns to download (default: *.xtc, *.gro, *.edr, *.log, *.cpt)
        remote_host: Remote HPC hostname
        remote_user: Remote username
        ssh_key_path: SSH key path
        
    Returns:
        Dict with downloaded file information
    """
    try:
        from src.hpc.file_copy import coerce_file_patterns

        file_patterns = coerce_file_patterns(file_patterns)
        if file_patterns is None:
            file_patterns = ["*.xtc", "*.gro", "*.edr", "*.log", "*.cpt", "*.xvg"]
        
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        if not remote_host:
            return {
                "success": False,
                "error": "remote_host is required for download"
            }
        
        try:
            import paramiko
        except ImportError:
            return {
                "success": False,
                "error": "paramiko not installed - cannot download from remote"
            }
        
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        connect_kwargs = {
            "hostname": remote_host,
            "username": remote_user,
            "port": 22
        }
        if ssh_key_path:
            connect_kwargs["key_filename"] = ssh_key_path
        
        client.connect(**connect_kwargs)
        sftp = client.open_sftp()
        
        downloaded_files = []
        
        try:
            # List remote files
            remote_files = sftp.listdir(remote_dir)
            
            for pattern in file_patterns:
                import fnmatch
                matching_files = [f for f in remote_files if fnmatch.fnmatch(f, pattern)]
                
                for filename in matching_files:
                    remote_path = f"{remote_dir}/{filename}"
                    local_file = local_path / filename
                    
                    sftp.get(remote_path, str(local_file))
                    file_size = local_file.stat().st_size
                    
                    downloaded_files.append({
                        "filename": filename,
                        "local_path": str(local_file),
                        "size": file_size
                    })
                    logger.info(f"Downloaded: {filename} ({file_size} bytes)")
        
        finally:
            sftp.close()
            client.close()
        
        if not downloaded_files:
            return {
                "success": False,
                "error": f"No files matching patterns {file_patterns} found in {remote_dir}"
            }
        
        return {
            "success": True,
            "files_downloaded": len(downloaded_files),
            "downloaded_files": downloaded_files,
            "local_directory": str(local_path),
            "message": f"Downloaded {len(downloaded_files)} files to {local_dir}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Download failed: {e}"
        }
