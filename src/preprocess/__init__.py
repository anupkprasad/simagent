"""
Preprocessing module for SimAgent

Contains tools for:
- Structure download from AlphaFold / RCSB by UniProt ID
- Domain extraction and validation
- PDB preprocessing and cleaning
- Ligand preprocessing and hydrogen addition
- Structure validation and fixing
"""

from .structure_sources import download_structure, list_structure_sources
from .structure_request_parser import parse_structure_request
from .structure_downloader import download_structure as download_structure_tool  # noqa: F401
from .domain_extractor import extract_domain
from .acquire_structure_tool import acquire_protein_structure

__all__ = [
    "download_structure",
    "list_structure_sources",
    "parse_structure_request",
    "extract_domain",
    "acquire_protein_structure",
]