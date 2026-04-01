"""
AutoDock4 Utilities
===================
Wrappers for AutoGrid4 and AutoDock4 execution, parameter file
generation, DLG parsing, and sequential MMSD merge operations.

Reference: Rajpal et al., Sci. Rep. 2024 — MMSD protocol
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── AutoGrid4 ──────────────────────────────────────────────────

def generate_gpf(receptor_pdbqt: Path, ligand_pdbqt: Path,
                 center: tuple, npts: tuple,
                 spacing: float = 0.375,
                 output_dir: Path = None) -> Path:
    """
    Generate AutoGrid4 parameter file (.gpf).

    Parameters
    ----------
    receptor_pdbqt : receptor PDBQT file
    ligand_pdbqt : ligand PDBQT file (to detect atom types)
    center : (x, y, z) grid center coordinates
    npts : (nx, ny, nz) grid points per axis
    spacing : grid spacing in Angstroms
    output_dir : where to write the GPF file
    """
    receptor_pdbqt = Path(receptor_pdbqt)
    ligand_pdbqt = Path(ligand_pdbqt)
    if output_dir is None:
        output_dir = receptor_pdbqt.parent
    output_dir = Path(output_dir)

    # Extract ligand atom types from PDBQT
    ligand_types = _extract_atom_types(ligand_pdbqt)
    receptor_types = _extract_atom_types(receptor_pdbqt)

    map_types = sorted(set(ligand_types))
    if not map_types:
        map_types = ["A", "C", "HD", "N", "NA", "OA", "SA"]

    # Check for non-standard atom types (e.g., Si in silane monomers)
    # AutoDock4 doesn't know Si — add custom parameter file
    has_custom_types = any(t not in _AD4_STANDARD_TYPES for t in map_types)
    custom_param_file = None
    if has_custom_types:
        custom_param_file = _write_custom_params(output_dir)
        # Keep custom types that have parameters in our file; remap rest to C
        remapped = []
        for t in map_types:
            if t in _AD4_STANDARD_TYPES:
                remapped.append(t)
            elif t in ("Si", "B"):
                remapped.append(t)  # handled by custom params
            else:
                remapped.append("C")
                logger.warning(f"Remapping unknown atom type '{t}' → C")
        map_types = sorted(set(remapped))

    gpf_name = f"{receptor_pdbqt.stem}_{ligand_pdbqt.stem}"
    gpf_path = output_dir / f"{gpf_name}.gpf"

    lines = []
    # Custom parameter file must come FIRST in GPF
    if custom_param_file:
        lines.append(f"parameter_file {custom_param_file.name}")
    lines.extend([
        f"npts {npts[0]} {npts[1]} {npts[2]}",
        f"gridfld {gpf_name}.maps.fld",
        f"spacing {spacing:.3f}",
        f"receptor_types {' '.join(sorted(set(receptor_types)))}",
        f"ligand_types {' '.join(map_types)}",
        f"receptor {receptor_pdbqt.name}",
        f"gridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}",
        f"smooth 0.5",
    ])
    # Add map line for each ligand atom type
    for atype in map_types:
        lines.append(f"map {gpf_name}.{atype}.map")
    lines.append(f"elecmap {gpf_name}.e.map")
    lines.append(f"dsolvmap {gpf_name}.d.map")
    lines.append("dielectric -0.1465")

    gpf_path.write_text("\n".join(lines) + "\n")
    logger.info(f"GPF written → {gpf_path}")
    return gpf_path


def run_autogrid(gpf_path: Path, timeout: int = 300) -> dict:
    """
    Execute AutoGrid4 with the given parameter file.

    Returns dict with 'success', 'glg_path', 'stderr'.
    """
    from .config import AUTOGRID4_BIN
    gpf_path = Path(gpf_path)
    glg_path = gpf_path.with_suffix(".glg")

    cmd = [AUTOGRID4_BIN, "-p", gpf_path.name, "-l", glg_path.name]
    try:
        result = subprocess.run(
            cmd, cwd=str(gpf_path.parent),
            capture_output=True, text=True, timeout=timeout,
        )
        success = result.returncode == 0
        if not success:
            logger.warning(f"AutoGrid4 failed: {result.stderr[:500]}")
        return {"success": success, "glg_path": glg_path,
                "stderr": result.stderr}
    except FileNotFoundError:
        logger.error(f"AutoGrid4 not found: {AUTOGRID4_BIN}")
        return {"success": False, "glg_path": None,
                "stderr": "autogrid4 not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "glg_path": None,
                "stderr": "timeout"}


# ── AutoDock4 ──────────────────────────────────────────────────

def generate_dpf(receptor_pdbqt: Path, ligand_pdbqt: Path,
                 map_prefix: str,
                 ga_runs: int = 50,
                 ga_pop_size: int = 150,
                 ga_num_evals: int = 2500000,
                 output_dir: Path = None) -> Path:
    """
    Generate AutoDock4 docking parameter file (.dpf).
    Uses Lamarckian Genetic Algorithm (LGA).
    """
    receptor_pdbqt = Path(receptor_pdbqt)
    ligand_pdbqt = Path(ligand_pdbqt)
    if output_dir is None:
        output_dir = receptor_pdbqt.parent
    output_dir = Path(output_dir)

    ligand_types = _extract_atom_types(ligand_pdbqt)
    map_types = sorted(set(ligand_types))

    # Handle non-standard types (Si etc.)
    has_custom = any(t not in _AD4_STANDARD_TYPES for t in map_types)
    custom_param = None
    if has_custom:
        custom_param = _write_custom_params(output_dir)
        remapped = []
        for t in map_types:
            if t in _AD4_STANDARD_TYPES or t in ("Si", "B"):
                remapped.append(t)
            else:
                remapped.append("C")
        map_types = sorted(set(remapped))

    dpf_name = f"{receptor_pdbqt.stem}_{ligand_pdbqt.stem}"
    dpf_path = output_dir / f"{dpf_name}.dpf"

    lines = []
    if custom_param:
        lines.append(f"parameter_file {custom_param.name}")
    lines.extend([
        f"autodock_parameter_version 4.2",
        f"outlev 1",
        f"intelec",
        f"seed pid time",
        f"ligand_types {' '.join(map_types)}",
        f"fld {map_prefix}.maps.fld",
    ])

    for atype in map_types:
        lines.append(f"map {map_prefix}.{atype}.map")
    lines.append(f"elecmap {map_prefix}.e.map")
    lines.append(f"desolvmap {map_prefix}.d.map")

    lines.extend([
        f"move {ligand_pdbqt.name}",
        f"about 0.0 0.0 0.0",
        f"tran0 random",
        f"quaternion0 random",
        f"dihe0 random",
        f"torsdof {_count_torsions(ligand_pdbqt)}",
        f"rmstol 2.0",
        f"extnrg 1000.0",
        f"e0max 0.0 10000",
        f"ga_pop_size {ga_pop_size}",
        f"ga_num_evals {ga_num_evals}",
        f"ga_num_generations 27000",
        f"ga_elitism 1",
        f"ga_mutation_rate 0.02",
        f"ga_crossover_rate 0.8",
        f"ga_window_size 10",
        f"ga_cauchy_alpha 0.0",
        f"ga_cauchy_beta 1.0",
        f"set_ga",
        f"sw_max_its 300",
        f"sw_max_succ 4",
        f"sw_max_fail 4",
        f"sw_rho 1.0",
        f"sw_lb_rho 0.01",
        f"ls_search_freq 0.06",
        f"set_psw1",
        f"unbound_model bound",
        f"ga_run {ga_runs}",
        f"analysis",
    ])

    dpf_path.write_text("\n".join(lines) + "\n")
    logger.info(f"DPF written → {dpf_path}")
    return dpf_path


def run_autodock(dpf_path: Path, timeout: int = 3600) -> dict:
    """
    Execute AutoDock4 (or AutoDock-GPU if available).

    AutoDock-GPU uses identical force field and scoring as AD4
    but runs ~100-350x faster on GPU (Santos-Martins et al.,
    J. Chem. Theory Comput. 2021).

    Returns dict with 'success', 'dlg_path', 'stderr', 'engine'.
    """
    from .config import AUTODOCK4_BIN, USE_AUTODOCK_GPU, AUTODOCK_GPU_BIN
    dpf_path = Path(dpf_path)
    dlg_path = dpf_path.with_suffix(".dlg")

    # Try AutoDock-GPU first (same results, ~100-350x faster)
    # Si handling: same UFF-based AD4_parameters_Si.dat as AutoDock4 CPU,
    # loaded via --import_dpf which reads 'parameter_file' from DPF.
    if USE_AUTODOCK_GPU and AUTODOCK_GPU_BIN:
        fld_files = list(dpf_path.parent.glob("*.maps.fld"))
        ligand_name = _get_ligand_from_dpf(dpf_path)
        if fld_files and ligand_name:
            cmd = [
                AUTODOCK_GPU_BIN,
                "--ffile", fld_files[0].name,
                "--lfile", ligand_name,
                "--resnam", dlg_path.stem,
                "--dlgoutput", "1",     # produce DLG (AD4-compatible)
                "--xmloutput", "0",     # skip XML
                "--clustering", "1",    # cluster analysis in DLG
            ]

            # Si atom handling: --derivtype tells AD-GPU that "Si" is
            # a derivative of "S", so it creates the Si atom type.
            # The actual vdW parameters come from the custom
            # parameter_file (AD4_parameters_Si.dat) in the DPF,
            # which --import_dpf reads — identical to AD4 CPU.
            # Declare non-standard atom types for AD-GPU
            ligand_pdbqt = dpf_path.parent / ligand_name
            derivtypes = []
            if ligand_pdbqt.exists():
                ligand_text = ligand_pdbqt.read_text()
                if " Si" in ligand_text or "\tSi" in ligand_text:
                    derivtypes.append("Si=S")
                if " B\n" in ligand_text or " B " in ligand_text or "\tB" in ligand_text:
                    derivtypes.append("B=C")
            if derivtypes:
                cmd.extend(["--derivtype", "/".join(derivtypes)])

            # Import DPF: reads parameter_file, GA params, etc.
            # AD-GPU --import_dpf has "partial support" — some AD4
            # tokens are unsupported. Write a filtered DPF for GPU.
            gpu_dpf = _write_gpu_compatible_dpf(dpf_path)
            cmd.extend(["--import_dpf", gpu_dpf.name])

            try:
                result = subprocess.run(
                    cmd, cwd=str(dpf_path.parent),
                    capture_output=True, text=True, timeout=timeout,
                )
                gpu_dlg = dpf_path.parent / f"{dlg_path.stem}.dlg"
                if gpu_dlg.exists() and gpu_dlg.stat().st_size > 100:
                    logger.info(f"AutoDock-GPU completed: {gpu_dlg.name}")
                    return {"success": True, "dlg_path": gpu_dlg,
                            "stderr": "", "engine": "AutoDock-GPU"}
                else:
                    logger.warning(
                        f"AutoDock-GPU produced no output, falling back to AD4. "
                        f"stderr: {result.stderr[:300]}"
                    )
            except FileNotFoundError:
                logger.warning("AutoDock-GPU binary not found, using AD4 CPU")
            except subprocess.TimeoutExpired:
                logger.warning("AutoDock-GPU timed out, using AD4 CPU")

    # AutoDock4 CPU fallback
    cmd = [AUTODOCK4_BIN, "-p", dpf_path.name, "-l", dlg_path.name]
    try:
        result = subprocess.run(
            cmd, cwd=str(dpf_path.parent),
            capture_output=True, text=True, timeout=timeout,
        )
        success = result.returncode == 0
        if not success:
            logger.warning(f"AutoDock4 failed: {result.stderr[:500]}")
        return {"success": success, "dlg_path": dlg_path,
                "stderr": result.stderr, "engine": "AutoDock4"}
    except FileNotFoundError:
        logger.error(f"AutoDock4 not found: {AUTODOCK4_BIN}")
        return {"success": False, "dlg_path": None,
                "stderr": "autodock4 not found", "engine": None}
    except subprocess.TimeoutExpired:
        return {"success": False, "dlg_path": None,
                "stderr": "timeout", "engine": None}


def _get_ligand_from_dpf(dpf_path: Path) -> str:
    """Extract ligand filename from DPF 'move' command."""
    for line in Path(dpf_path).read_text().split("\n"):
        if line.startswith("move "):
            return line.split()[1]
    return ""


def _write_gpu_compatible_dpf(dpf_path: Path) -> Path:
    """
    Write a filtered DPF that AutoDock-GPU --import_dpf can parse.

    AD-GPU only supports a subset of AD4 DPF tokens. Unsupported
    tokens cause fatal errors. We keep only what AD-GPU needs:
    parameter_file, ga_* params, unbound_model, etc.
    """
    # Tokens that AutoDock-GPU --import_dpf supports
    _SUPPORTED_TOKENS = {
        "parameter_file", "ligand_types", "fld", "map", "elecmap",
        "desolvmap", "move", "about", "tran0", "quaternion0", "dihe0",
        "torsdof", "rmstol", "ga_pop_size", "ga_num_evals",
        "ga_num_generations", "ga_elitism", "ga_mutation_rate",
        "ga_crossover_rate", "ga_window_size", "ga_run",
        "sw_max_its", "sw_max_succ", "sw_max_fail", "sw_rho",
        "sw_lb_rho", "ls_search_freq", "unbound_model",
        "set_ga", "set_psw1", "analysis",
        "ga_cauchy_alpha", "ga_cauchy_beta",
        "extnrg", "e0max", "seed",
    }

    dpf_path = Path(dpf_path)
    gpu_dpf = dpf_path.parent / f"{dpf_path.stem}_gpu.dpf"

    filtered_lines = []
    for line in dpf_path.read_text().split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        token = stripped.split()[0]
        if token in _SUPPORTED_TOKENS:
            filtered_lines.append(stripped)

    gpu_dpf.write_text("\n".join(filtered_lines) + "\n")
    return gpu_dpf


# ── DLG Parsing ────────────────────────────────────────────────

def parse_dlg(dlg_path: Path) -> list:
    """
    Parse AutoDock4 DLG (docking log) file.

    Returns list of clusters, each dict with:
      - rank, binding_energy, cluster_size, rmsd_from_ref
      - mean_binding_energy, best_binding_energy
      - coords (PDBQT lines of best pose)
    """
    dlg_path = Path(dlg_path)
    if not dlg_path.exists():
        logger.error(f"DLG not found: {dlg_path}")
        return []

    text = dlg_path.read_text()
    clusters = []

    # Parse CLUSTERING HISTOGRAM
    cluster_pattern = re.compile(
        r"RANKING\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)"
    )

    # Parse ranked results table
    # Format: Rank | Sub-Rank | Run | Binding Energy | Cluster RMSD | ...
    result_block = False
    current_results = []

    for line in text.split("\n"):
        # Look for DOCKED keyword to extract poses
        if "CLUSTERING HISTOGRAM" in line:
            result_block = True
            continue

        m = cluster_pattern.search(line)
        if m:
            clusters.append({
                "rank": int(m.group(1)),
                "binding_energy": float(m.group(2)),
                "rmsd_from_ref": float(m.group(3)),
                "cluster_size": int(m.group(4)),
            })

    # If no clustering info found, parse individual runs
    if not clusters:
        clusters = _parse_dlg_individual_runs(text)

    # Extract best pose coordinates
    best_pose_lines = _extract_best_pose(text)
    if clusters and best_pose_lines:
        clusters[0]["pose_pdbqt_lines"] = best_pose_lines

    logger.info(f"Parsed {len(clusters)} clusters from {dlg_path.name}")
    return clusters


def get_best_energy(dlg_path: Path) -> float:
    """Return the best (most negative) binding energy from DLG."""
    clusters = parse_dlg(dlg_path)
    if not clusters:
        return 0.0
    return min(c["binding_energy"] for c in clusters)


def get_mean_best_cluster_energy(dlg_path: Path) -> float:
    """
    Return mean binding energy of the best cluster (rank 1).
    This is the metric used by Rajpal et al. 2024 for comparison.
    """
    clusters = parse_dlg(dlg_path)
    if not clusters:
        return 0.0
    # Rank 1 cluster
    best = [c for c in clusters if c.get("rank") == 1]
    if best:
        return best[0]["binding_energy"]
    return clusters[0]["binding_energy"]


def extract_best_pose_pdbqt(dlg_path: Path, output_path: Path) -> Path:
    """Extract best-energy docked pose from DLG as a PDBQT file."""
    text = Path(dlg_path).read_text()
    pose_lines = _extract_best_pose(text)
    if not pose_lines:
        logger.warning(f"No docked pose found in {dlg_path}")
        return None
    Path(output_path).write_text("\n".join(pose_lines) + "\n")
    return Path(output_path)


# ── MMSD Merge Operations ──────────────────────────────────────

def merge_ligand_into_receptor(receptor_pdb: Path,
                                ligand_pdbqt: Path,
                                output_pdb: Path) -> Path:
    """
    Merge a docked ligand into the receptor PDB for sequential MMSD.

    The merged file becomes the new "receptor" for the next docking round.
    Ligand coordinates are extracted from PDBQT and appended as HETATM.
    """
    receptor_pdb = Path(receptor_pdb)
    ligand_pdbqt = Path(ligand_pdbqt)
    output_pdb = Path(output_pdb)

    receptor_lines = []
    for line in receptor_pdb.read_text().split("\n"):
        if line.startswith(("ATOM", "HETATM", "TER")):
            receptor_lines.append(line)

    # Extract ligand coordinates from PDBQT
    ligand_lines = []
    for line in ligand_pdbqt.read_text().split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            # Convert PDBQT to PDB format (strip last 2 columns)
            pdb_line = line[:66].ljust(66)
            # Mark as HETATM
            if pdb_line.startswith("ATOM"):
                pdb_line = "HETATM" + pdb_line[6:]
            ligand_lines.append(pdb_line)

    all_lines = receptor_lines + ["TER"] + ligand_lines + ["END"]
    output_pdb.write_text("\n".join(all_lines) + "\n")
    logger.info(f"Merged receptor+ligand → {output_pdb}")
    return output_pdb


# ── Full Docking Workflow ──────────────────────────────────────

def dock_single(receptor_pdbqt: Path, ligand_pdbqt: Path,
                center: tuple, npts: tuple,
                work_dir: Path,
                ga_runs: int = 50) -> dict:
    """
    Complete single docking: AutoGrid → AutoDock → parse results.

    Returns dict with:
      - binding_energy (best cluster)
      - mean_cluster_energy (rank-1 cluster mean)
      - n_clusters
      - dlg_path
      - best_pose_path (PDBQT)
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy input files to work_dir
    import shutil
    rec_local = work_dir / receptor_pdbqt.name
    lig_local = work_dir / ligand_pdbqt.name
    if not rec_local.exists():
        shutil.copy2(str(receptor_pdbqt), str(rec_local))
    if not lig_local.exists():
        shutil.copy2(str(ligand_pdbqt), str(lig_local))

    # 1. AutoGrid
    gpf = generate_gpf(rec_local, lig_local, center, npts,
                        output_dir=work_dir)
    grid_result = run_autogrid(gpf)
    if not grid_result["success"]:
        return {"binding_energy": None, "mean_cluster_energy": None,
                "error": grid_result["stderr"], "success": False}

    # 2. AutoDock
    map_prefix = f"{rec_local.stem}_{lig_local.stem}"
    from .config import AUTODOCK4_GA_POP_SIZE, AUTODOCK4_GA_NUM_EVALS
    dpf = generate_dpf(
        rec_local, lig_local, map_prefix,
        ga_runs=ga_runs,
        ga_pop_size=AUTODOCK4_GA_POP_SIZE,
        ga_num_evals=AUTODOCK4_GA_NUM_EVALS,
        output_dir=work_dir,
    )
    dock_result = run_autodock(dpf)
    if not dock_result["success"]:
        return {"binding_energy": None, "mean_cluster_energy": None,
                "error": dock_result["stderr"], "success": False}

    # 3. Parse results
    dlg = dock_result["dlg_path"]
    clusters = parse_dlg(dlg)
    best_e = get_best_energy(dlg)
    mean_e = get_mean_best_cluster_energy(dlg)

    # 4. Extract best pose
    best_pose = work_dir / f"{lig_local.stem}_best.pdbqt"
    extract_best_pose_pdbqt(dlg, best_pose)

    return {
        "binding_energy": best_e,
        "mean_cluster_energy": mean_e,
        "n_clusters": len(clusters),
        "dlg_path": str(dlg),
        "best_pose_path": str(best_pose) if best_pose.exists() else None,
        "clusters": clusters[:5],  # top-5 clusters for reporting
    }


# ── AutoDock4 Standard Atom Types ──────────────────────────────

_AD4_STANDARD_TYPES = {
    "H", "HD", "HS", "C", "A", "N", "NA", "NS", "OA", "OS",
    "F", "Mg", "MN", "P", "SA", "S", "Cl", "CL", "Ca", "Mn",
    "Fe", "Zn", "Br", "BR", "I",
}


def _write_custom_params(output_dir: Path) -> Path:
    """
    Write AutoDock4 custom parameter file with non-standard atom types.

    Per autodock.scripps.edu/how-to-add-new-atom-types:
    - Si: UFF (Rappe et al., JACS 1992) Si_3 tetrahedral.
    - B:  UFF B_3 trigonal (for boronic acid APBA monomer).
    """
    param_path = Path(output_dir) / "AD4_parameters_custom.dat"
    if param_path.exists():
        return param_path

    # Si: UFF Si_3 (tetrahedral silicon)
    #   Rii = 4.295 A, epsii = 0.402 kcal/mol
    # B:  UFF B_3 (trigonal boron)
    #   Rii = 4.083 A, epsii = 0.180 kcal/mol
    #   B in boronic acid B(OH)2 is sp2 trigonal → B_2 in UFF
    #   Rij_hb = 0 (no H-bond), hbond = 0
    content = """\
# AutoDock4 custom parameters for non-standard atom types
# Source: UFF (Rappe et al., JACS 1992, 114, 10024-10035)
# Per: autodock.scripps.edu/how-to-add-new-atom-types-to-the-autodock-force-field/
#
# Si: for silane monomers (PTES, APTES, APTMS, TEOS, etc.)
# B:  for boronic acid monomer (APBA — glycan recognition)
#
# atom_par  Rii    epsii   vol      solpar    Rij_hb epsij_hb hbond rec map
atom_par Si  4.295  0.402  12.175  -0.00143  0.0  0.0  0  -1  -1
atom_par B   4.083  0.180  11.000  -0.00110  0.0  0.0  0  -1  -1
"""
    param_path.write_text(content)
    logger.info(f"Custom AD4 parameters (Si) written: {param_path}")
    return param_path


# ── Internal Helpers ───────────────────────────────────────────

def _extract_atom_types(pdbqt_path: Path) -> list:
    """Extract unique atom types from a PDBQT file."""
    types = []
    for line in Path(pdbqt_path).read_text().split("\n"):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 78:
            atype = line[77:79].strip()
            if atype:
                types.append(atype)
    return types if types else ["A", "C", "HD", "N", "OA"]


def _count_torsions(pdbqt_path: Path) -> int:
    """Count active torsions from PDBQT TORSDOF line."""
    for line in Path(pdbqt_path).read_text().split("\n"):
        if line.startswith("TORSDOF"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    return 0


def _parse_dlg_individual_runs(text: str) -> list:
    """Fallback parser: extract binding energies from individual runs."""
    results = []
    pattern = re.compile(
        r"DOCKED.*?Estimated Free Energy of Binding\s*=\s*(-?\d+\.\d+)",
        re.DOTALL,
    )
    for i, m in enumerate(pattern.finditer(text)):
        results.append({
            "rank": i + 1,
            "binding_energy": float(m.group(1)),
            "cluster_size": 1,
        })
    # Sort by energy
    results.sort(key=lambda x: x["binding_energy"])
    # Re-rank
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


def _extract_best_pose(text: str) -> list:
    """Extract the coordinates of the lowest-energy docked pose."""
    best_energy = float("inf")
    best_lines = []
    current_energy = None
    current_lines = []
    in_model = False

    for line in text.split("\n"):
        if "DOCKED: MODEL" in line:
            in_model = True
            current_lines = []
            current_energy = None
        elif "DOCKED: ENDMDL" in line:
            in_model = False
            if current_energy is not None and current_energy < best_energy:
                best_energy = current_energy
                best_lines = current_lines[:]
        elif in_model:
            stripped = line.replace("DOCKED: ", "", 1)
            if "Estimated Free Energy of Binding" in line:
                m = re.search(r"=\s*(-?\d+\.\d+)", line)
                if m:
                    current_energy = float(m.group(1))
            if stripped.startswith(("ATOM", "HETATM")):
                current_lines.append(stripped)

    return best_lines
