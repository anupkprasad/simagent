"""
Ligand Topology Generation with Acpype/Antechamber
Generates GAFF/GAFF2 parameters for small molecules
"""
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from langchain.tools import tool
import shutil


class LigandTopologyGenerator:
    """Generate ligand topology using acpype/antechamber."""
    
    def __init__(self):
        try:
            from src.simsetup.md_env import ensure_md_toolchain

            ensure_md_toolchain()
        except Exception:
            pass
        self.acpype_available = shutil.which("acpype") is not None
        self.antechamber_available = shutil.which("antechamber") is not None
        
    def check_dependencies(self) -> Dict[str, bool]:
        """Check which parameterization tools are available."""
        # Re-check after possible late PATH bootstrap.
        try:
            from src.simsetup.md_env import ensure_md_toolchain

            ensure_md_toolchain()
        except Exception:
            pass
        self.acpype_available = shutil.which("acpype") is not None
        self.antechamber_available = shutil.which("antechamber") is not None
        return {
            "acpype": self.acpype_available,
            "antechamber": self.antechamber_available
        }

    def _sanitize_ligand_pdb(self, ligand_pdb: str, output_dir: Path) -> str:
        """
        Create a sanitized PDB for Amber-family tools.

        Some extracted ligand PDBs contain nonstandard altLoc values (e.g. "1")
        which can cause ACPYPE/antechamber parsing failures. This method keeps
        coordinates unchanged while normalizing problematic record columns.
        """
        src = Path(ligand_pdb)
        sanitized = output_dir / f"{src.stem}_acpype_input.pdb"

        with src.open("r", encoding="utf-8", errors="ignore") as fin, sanitized.open("w", encoding="utf-8") as fout:
            for line in fin:
                if line.startswith(("ATOM", "HETATM")) and len(line) >= 17:
                    chars = list(line.rstrip("\n"))
                    # Ensure minimum width before positional edits.
                    if len(chars) < 80:
                        chars.extend([" "] * (80 - len(chars)))
                    # altLoc is column 17 (index 16). Keep letters; clear digits/other tokens.
                    altloc = chars[16]
                    if altloc not in (" ",) and not altloc.isalpha():
                        chars[16] = " "
                    fout.write("".join(chars).rstrip() + "\n")
                else:
                    fout.write(line)

        return str(sanitized)
    
    def generate_with_acpype(
        self,
        ligand_pdb: str,
        output_dir: str,
        charge_method: str = "bcc",
        net_charge: Optional[int] = None,
        atom_type: str = "gaff2"
    ) -> Dict[str, Any]:
        """
        Generate ligand topology using acpype.
        
        Args:
            ligand_pdb: Path to ligand PDB file
            output_dir: Output directory (if empty, uses ligand PDB's directory)
            charge_method: Charge calculation method (bcc, gas, etc.)
            net_charge: Net charge of molecule (auto-detect if None)
            atom_type: GAFF version (gaff, gaff2)
            
        Returns:
            Dict with generated files and status
        """
        # Handle empty output_dir
        if not output_dir or output_dir.strip() == "":
            output_dir = Path(ligand_pdb).parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)

        ligand_input_path = Path(ligand_pdb)
        # Normalize PDB formatting quirks before running ACPYPE.
        if ligand_input_path.suffix.lower() == ".pdb":
            ligand_input = self._sanitize_ligand_pdb(str(ligand_input_path), output_dir)
        else:
            ligand_input = str(ligand_input_path)
        
        # Resolve ligand_pdb to absolute path (acpype runs from output_dir)
        ligand_pdb_abs = str(Path(ligand_input).resolve())
        
        try:
            from src.simsetup.md_env import acpype_subprocess_env, ensure_md_toolchain

            ensure_md_toolchain()
        except Exception:
            acpype_subprocess_env = None  # type: ignore[assignment]

        acpype_exe = shutil.which("acpype") or "acpype"
        cmd = [
            acpype_exe,
            "-i", ligand_pdb_abs,
            "-a", atom_type,
            "-c", charge_method
        ]
        
        # Guard: treat empty string / non-integer as auto-detect
        if net_charge is not None:
            try:
                net_charge = int(net_charge)
                cmd.extend(["-n", str(net_charge)])
            except (ValueError, TypeError):
                pass  # auto-detect
        
        try:
            run_env = acpype_subprocess_env(acpype_exe) if acpype_subprocess_env else None
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(output_dir),
                env=run_env,
            )
            
            # ACPYPE may derive output names from sanitized inputs. Probe both
            # expected directories and fallback to any recent *.acpype directory.
            ligand_name = Path(ligand_pdb).stem
            input_name = Path(ligand_input).stem
            acpype_dirs = []
            for candidate in [
                output_dir / f"{ligand_name}.acpype",
                output_dir / f"{input_name}.acpype",
            ]:
                if candidate not in acpype_dirs:
                    acpype_dirs.append(candidate)
            for candidate in sorted(output_dir.glob("*.acpype"), key=lambda p: p.stat().st_mtime, reverse=True):
                if candidate not in acpype_dirs:
                    acpype_dirs.append(candidate)

            # Check for output files regardless of exit code —
            # acpype can return non-zero even on success (e.g. warnings)
            for acpype_dir in acpype_dirs:
                if not acpype_dir.exists():
                    continue

                topology_file = None
                for stem in [ligand_name, input_name]:
                    candidate = acpype_dir / f"{stem}_GMX.itp"
                    if candidate.exists():
                        topology_file = candidate
                        break
                if topology_file is None:
                    itp_candidates = sorted(acpype_dir.glob("*_GMX.itp"))
                    topology_file = itp_candidates[0] if itp_candidates else None
                if topology_file is None:
                    continue

                coordinate_file = topology_file.with_suffix(".gro")
                if not coordinate_file.exists():
                    gro_candidates = sorted(acpype_dir.glob("*_GMX.gro"))
                    coordinate_file = gro_candidates[0] if gro_candidates else None

                # Copy output files to output_dir using canonical ligand-based names
                # so downstream tools do not depend on ACPYPE internal naming.
                dest_itp = output_dir / f"{ligand_name}_GMX.itp"
                dest_gro = output_dir / f"{ligand_name}_GMX.gro" if coordinate_file else None
                shutil.copy2(str(topology_file), str(dest_itp))
                if dest_gro and coordinate_file:
                    shutil.copy2(str(coordinate_file), str(dest_gro))

                # Copy position-restraint files (posre_<RESNAME>.itp) — needed
                # by topol.top #ifdef POSRES_LIG blocks during NPT/NVT/MD runs
                posre_files = []
                for posre in acpype_dir.glob("posre_*.itp"):
                    dest_posre = output_dir / posre.name
                    shutil.copy2(str(posre), str(dest_posre))
                    posre_files.append(str(dest_posre))
                return {
                    "success": True,
                    "topology": str(dest_itp),
                    "coordinates": str(dest_gro) if dest_gro else None,
                    "posre_files": posre_files,
                    "output_dir": str(output_dir),
                    "acpype_dir": str(acpype_dir),
                    "method": "acpype",
                    "atom_type": atom_type,
                    "charge_method": charge_method,
                    "stdout": result.stdout[-500:] if result.stdout else "",
                    "stderr": result.stderr[-500:] if result.stderr else ""
                }
            
            # If we get here, acpype didn't produce expected output
            combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
            return {
                "success": False,
                "error": f"Acpype finished (exit {result.returncode}) but did not produce expected files. Output: {combined_output[-800:]}"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Acpype execution timed out (>600s)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def _generate_with_acpype_fallbacks(
        self,
        ligand_pdb: str,
        output_dir: str,
        charge_method: str = "bcc",
        net_charge: Optional[int] = None,
        atom_type: str = "gaff2"
    ) -> Dict[str, Any]:
        """
        Try ACPYPE with progressively safer options.

        Rationale: some ligands (notably nucleotide-like molecules) can fail
        with specific atom type / charge combinations. Retrying with alternate
        ACPYPE options often succeeds without requiring user intervention.
        """
        attempts = []
        # Try requested settings first, then conservative fallbacks.
        candidates = [
            (atom_type, charge_method),
            ("gaff", charge_method),
            (atom_type, "gas"),
            ("gaff", "gas"),
        ]

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)

        for atype, cmethod in unique_candidates:
            res = self.generate_with_acpype(
                ligand_pdb=ligand_pdb,
                output_dir=output_dir,
                charge_method=cmethod,
                net_charge=net_charge,
                atom_type=atype,
            )
            attempts.append({
                "atom_type": atype,
                "charge_method": cmethod,
                "success": bool(res.get("success")),
                "error": res.get("error", ""),
            })
            if res.get("success"):
                # Keep traceability when fallback options were needed.
                if atype != atom_type or cmethod != charge_method:
                    res["warning"] = (
                        f"ACPYPE fallback succeeded with atom_type={atype}, "
                        f"charge_method={cmethod}"
                    )
                res["attempts"] = attempts
                return res

        # Final fallback: PDB -> MOL2 via antechamber, then retry ACPYPE on MOL2.
        # This often helps when direct ACPYPE parsing of PDB fails.
        if self.antechamber_available:
            ante = self.generate_with_antechamber(
                ligand_pdb=ligand_pdb,
                output_dir=output_dir,
                charge_method=charge_method,
                net_charge=net_charge,
                atom_type=atom_type,
            )
            mol2_file = ante.get("mol2_file") if ante.get("success") else None
            if mol2_file and Path(mol2_file).exists():
                for atype, cmethod in unique_candidates:
                    res = self.generate_with_acpype(
                        ligand_pdb=mol2_file,
                        output_dir=output_dir,
                        charge_method=cmethod,
                        net_charge=net_charge,
                        atom_type=atype,
                    )
                    attempts.append({
                        "atom_type": atype,
                        "charge_method": cmethod,
                        "success": bool(res.get("success")),
                        "error": res.get("error", ""),
                        "input": "mol2-preconvert",
                    })
                    if res.get("success"):
                        res["warning"] = (
                            "ACPYPE direct PDB input failed; succeeded after "
                            "antechamber MOL2 pre-conversion."
                        )
                        res["attempts"] = attempts
                        return res

        last_error = attempts[-1]["error"] if attempts else "Unknown ACPYPE failure"
        return {
            "success": False,
            "error": f"All ACPYPE attempts failed. Last error: {last_error}",
            "attempts": attempts,
        }
    
    def generate_with_antechamber(
        self,
        ligand_pdb: str,
        output_dir: str,
        charge_method: str = "bcc",
        net_charge: Optional[int] = None,
        atom_type: str = "gaff2"
    ) -> Dict[str, Any]:
        """
        Generate ligand topology using antechamber directly.
        
        Args:
            ligand_pdb: Path to ligand PDB file
            output_dir: Output directory (if empty, uses ligand PDB's directory)
            charge_method: Charge calculation method (bcc, gas, etc.)
            net_charge: Net charge of molecule (auto-detect if None)
            atom_type: GAFF version (gaff, gaff2)
            
        Returns:
            Dict with generated files and status
        """
        # Handle empty output_dir
        if not output_dir or output_dir.strip() == "":
            output_dir = Path(ligand_pdb).parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize PDB formatting quirks before running antechamber.
        ligand_input = self._sanitize_ligand_pdb(ligand_pdb, output_dir)
        
        ligand_name = Path(ligand_pdb).stem
        mol2_file = output_dir / f"{ligand_name}.mol2"
        frcmod_file = output_dir / f"{ligand_name}.frcmod"
        
        # Step 1: Run antechamber
        cmd_ante = [
            "antechamber",
            "-i", str(ligand_input),
            "-fi", "pdb",
            "-o", str(mol2_file),
            "-fo", "mol2",
            "-c", charge_method,
            "-at", atom_type,
            "-pf", "y"
        ]
        
        # Guard: treat empty string / non-integer as auto-detect
        if net_charge is not None:
            try:
                net_charge = int(net_charge)
                cmd_ante.extend(["-nc", str(net_charge)])
            except (ValueError, TypeError):
                pass  # auto-detect
        
        try:
            result_ante = subprocess.run(
                cmd_ante,
                capture_output=True,
                text=True,
                timeout=600,
                check=True
            )
            
            # Step 2: Run parmchk2 to generate missing parameters
            cmd_parmchk = [
                "parmchk2",
                "-i", str(mol2_file),
                "-f", "mol2",
                "-o", str(frcmod_file),
                "-s", atom_type
            ]
            
            result_parmchk = subprocess.run(
                cmd_parmchk,
                capture_output=True,
                text=True,
                timeout=300,
                check=True
            )
            
            return {
                "success": True,
                "mol2_file": str(mol2_file),
                "frcmod_file": str(frcmod_file),
                "output_dir": str(output_dir),
                "method": "antechamber",
                "atom_type": atom_type,
                "charge_method": charge_method,
                "stdout": result_ante.stdout + "\n" + result_parmchk.stdout,
                "stderr": result_ante.stderr + "\n" + result_parmchk.stderr,
                "note": "Convert to GROMACS format with gmx x2top or acpype"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Antechamber execution timed out"
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Antechamber failed: {e.stderr}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def generate_ligand_topology(
        self,
        ligand_pdb: str,
        output_dir: str,
        charge_method: str = "bcc",
        net_charge: Optional[int] = None,
        atom_type: str = "gaff2",
        preferred_tool: str = "acpype"
    ) -> Dict[str, Any]:
        """
        Generate ligand topology with automatic tool selection.
        
        Args:
            ligand_pdb: Path to ligand PDB file
            output_dir: Output directory
            charge_method: Charge calculation method (bcc, gas, etc.)
            net_charge: Net charge of molecule (auto-detect if None)
            atom_type: GAFF version (gaff, gaff2)
            preferred_tool: Preferred tool (acpype or antechamber)
            
        Returns:
            Dict with generated files and status
        """
        deps = self.check_dependencies()
        
        if not deps["acpype"] and not deps["antechamber"]:
            return {
                "success": False,
                "error": (
                    "Neither acpype nor antechamber available on PATH. "
                    "Activate conda env SimAgentEnv (has acpype) or "
                    "`module load AmberTools`, then retry."
                ),
                "dependencies": deps
            }

        # Prefer requested tool when present; otherwise use whatever is available.
        if preferred_tool == "acpype" and not deps["acpype"] and deps["antechamber"]:
            preferred_tool = "antechamber"
        elif preferred_tool == "antechamber" and not deps["antechamber"] and deps["acpype"]:
            preferred_tool = "acpype"
        
        # Try preferred tool first
        if preferred_tool == "acpype" and deps["acpype"]:
            acpype_result = self._generate_with_acpype_fallbacks(
                ligand_pdb, output_dir, charge_method, net_charge, atom_type
            )
            return acpype_result

        elif preferred_tool == "antechamber" and deps["antechamber"]:
            return self.generate_with_antechamber(
                ligand_pdb, output_dir, charge_method, net_charge, atom_type
            )
        
        # Fallback to available tool
        if deps["acpype"]:
            return self._generate_with_acpype_fallbacks(
                ligand_pdb, output_dir, charge_method, net_charge, atom_type
            )
        else:
            return self.generate_with_antechamber(
                ligand_pdb, output_dir, charge_method, net_charge, atom_type
            )


# LangChain tool wrapper
@tool
def generate_ligand_parameters(
    ligand_pdb: str,
    output_dir: Optional[str] = None,
    charge_method: str = "bcc",
    net_charge: Optional[int] = None,
    atom_type: str = "gaff2",
    preferred_tool: str = "acpype"
) -> Dict[str, Any]:
    """
    Generate missing ligand topology parameters using acpype or antechamber.
    Uses GAFF/GAFF2 force field for small molecules.
    
    Args:
        ligand_pdb: Path to ligand PDB file (e.g., ATP_cleaned.pdb)
        output_dir: Directory for output files (if None or empty, uses ligand PDB's directory)
        charge_method: Charge calculation method (bcc=AM1-BCC, gas=Gasteiger)
        net_charge: Net molecular charge (auto-detect if None)
        atom_type: GAFF version (gaff or gaff2, recommended: gaff2)
        preferred_tool: Preferred tool (acpype or antechamber)
        
    Returns:
        Dict with paths to generated topology files and success status
        
    Example:
        >>> result = generate_ligand_parameters(
        ...     ligand_pdb="working_dir/preprocess/ATP.pdb",
        ...     output_dir="working_dir/simsetup/ligand_params"
        ... )
        >>> print(result["topology"])  # ATP_GMX.itp
    """
    generator = LigandTopologyGenerator()
    
    # Handle empty or None output_dir
    if not output_dir or (isinstance(output_dir, str) and output_dir.strip() == ""):
        output_dir = str(Path(ligand_pdb).parent)
    
    # Sanitize net_charge: empty string or non-integer means auto-detect (None)
    if net_charge is not None:
        try:
            net_charge = int(net_charge) if str(net_charge).strip() != "" else None
        except (ValueError, TypeError):
            net_charge = None

    result = generator.generate_ligand_topology(
        ligand_pdb=ligand_pdb,
        output_dir=output_dir,
        charge_method=charge_method,
        net_charge=net_charge,
        atom_type=atom_type,
        preferred_tool=preferred_tool
    )
    
    return result
