"""
AutoDock4 Utilities
===================
Wrappers for AutoGrid4 and AutoDock4 execution, parameter file
generation, DLG parsing, and sequential MMSD merge operations.

Reference: Rajpal et al., Sci. Rep. 2024 — MMSD protocol
"""

import hashlib
import logging
import math
import os
import re
import shutil
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
            elif t in _CUSTOM_VDW:
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
    Generate the AutoDock4 **CPU** docking parameter file (.dpf).
    Uses the Lamarckian Genetic Algorithm (LGA).

    SCOPE. This file is consumed by the autodock4 CPU binary. AutoDock-GPU does
    NOT read it — run_autodock() re-reads the GA settings from here and passes
    them to AD-GPU as command-line flags instead, because --import_dpf drops
    `parameter_file` and `seed` without saying so. The `parameter_file` line
    below therefore reaches the GPU path only indirectly, via the AutoGrid maps
    the GPF built with the same file.

    `seed pid time` deliberately stays for the CPU engine's own use; the GPU
    path is seeded explicitly from config.AUTODOCK_SEED.
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
            if t in _AD4_STANDARD_TYPES or t in _CUSTOM_VDW:
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


# ── AutoDock-GPU settings transfer (BLOCKER 04) ────────────────
#
# THE DPF IS NOT A CHANNEL TO AutoDock-GPU. --import_dpf has "partial support"
# and what it drops it drops SILENTLY. Verified against this build
# (v1.6-14-g28a6139):
#   * `parameter_file AD4_parameters_custom.dat` -> IGNORED. Replacing it with
#     `parameter_file NO_SUCH_FILE_XYZ.dat` runs to completion, rc=0, and
#     returns the same energy. The Si/B parameters never reach the engine
#     through this route.
#   * `seed pid time` -> IGNORED. The DLG header of an --import_dpf run reads
#     "Random seed: 1786527439, 458862", i.e. AD-GPU's own time+pid default.
#     Three repeats of one identical docking gave -3.89 / -3.49 / -3.68.
#   * `ls_search_freq 0.06`, `set_psw1`, `sw_*` -> IGNORED. AD-GPU keeps its
#     own ADADELTA local search at 100% rate.
#   * ga_run / ga_num_evals / ga_pop_size ARE read — but only as long as
#     --heuristics is off, and heuristics defaults to ON.
# So every setting AD-GPU honours is now passed as an EXPLICIT FLAG, and
# --import_dpf is not used at all. What still cannot be transferred is logged
# once per run rather than assumed applied.

# AD4.1 vdW parameters (Rii Å, epsii kcal/mol) — the same table AutoDock-GPU
# compiles in (host/src/processligand.cpp: reqm[]/eps[]). Needed to build
# --modpair arguments; keep in sync with _write_custom_params below.
_AD4_VDW = {
    "H": (2.00, 0.020), "HD": (2.00, 0.020), "HS": (2.00, 0.020),
    "C": (4.00, 0.150), "A": (4.00, 0.150),
    "N": (3.50, 0.160), "NA": (3.50, 0.160), "NS": (3.50, 0.160),
    "OA": (3.20, 0.200), "OS": (3.20, 0.200),
    "F": (3.09, 0.080), "MG": (1.30, 0.875), "Mg": (1.30, 0.875),
    "P": (4.20, 0.200), "SA": (4.00, 0.200), "S": (4.00, 0.200),
    "CL": (4.09, 0.276), "Cl": (4.09, 0.276),
    "CA": (1.98, 0.550), "Ca": (1.98, 0.550),
    "MN": (1.30, 0.875), "Mn": (1.30, 0.875),
    "FE": (1.30, 0.010), "Fe": (1.30, 0.010),
    "ZN": (1.48, 0.550), "Zn": (1.48, 0.550),
    "BR": (4.33, 0.389), "Br": (4.33, 0.389),
    "I": (4.72, 0.550),
    "CG": (4.00, 0.150), "G0": (0.00, 0.000),
}

# Our non-standard types. AutoDock-GPU has BUILT-IN entries for SI (4.10,
# 0.200) and B (3.84, 0.155) which are NOT the UFF values this project chose,
# so the ligand-internal vdW would silently use the wrong ones. --modpair
# overrides them pair by pair.
_CUSTOM_VDW = {
    "Si": (4.295, 0.402),    # UFF Si_3, Rappe et al. JACS 1992
    "B": (4.083, 0.180),     # UFF B_2 (trigonal, boronic acid)
}
_AD4_COEFF_VDW = 0.1662      # AD4.2 bound-model vdW weight

# Fallback base type per non-standard type, used ONLY when AutoGrid failed to
# produce that type's map (which means the custom parameters were lost anyway).
_DERIV_FALLBACK = {"Si": ("S", "SA"), "B": ("C", "A")}


def _fld_map_types(fld_path: Path) -> set:
    """Atom types for which the .maps.fld actually declares an affinity map."""
    types = set()
    for line in Path(fld_path).read_text().split("\n"):
        m = re.match(r"\s*label=(\S+)-affinity", line)
        if m:
            types.add(m.group(1))
    return types


def _resolve_seed(dpf_path: Path, seed=None) -> tuple:
    """
    Three AD-GPU random seeds, DETERMINISTIC for a given job.

    Base seed from config.AUTODOCK_SEED (falls back to a fixed constant so
    determinism never depends on a config edit landing). The job name is mixed
    in so two different dockings do not share an identical random stream while
    each one stays exactly reproducible across repeats.
    """
    if seed is not None:
        s = tuple(int(v) for v in seed)
        return (s + s[-1:] * 3)[:3]
    # getattr, not `from .config import AUTODOCK_SEED`: the key is not in
    # config yet and test_config_regression.test_import_contract AST-walks
    # from-imports and demands every name resolve under both experiments.
    # This picks the key up the moment it is added; until then the constant
    # below IS the seed, so the run is reproducible either way.
    try:
        from . import config as _cfg
        base = int(getattr(_cfg, "AUTODOCK_SEED", 20260812))
    except Exception:                                 # noqa: BLE001
        base = 20260812
    h = int(hashlib.sha256(Path(dpf_path).stem.encode()).hexdigest()[:8], 16)
    return (base % 2147483647,
            (base ^ h) % 2147483647,
            (base + h) % 2147483647)


def _dpf_settings(dpf_path: Path) -> dict:
    """Pull the GA settings out of a DPF so they can be passed as CLI flags."""
    want = {"ga_run": None, "ga_num_evals": None, "ga_pop_size": None,
            "rmstol": None, "ga_mutation_rate": None,
            "ga_crossover_rate": None, "unbound_model": None}
    for line in Path(dpf_path).read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 2 and parts[0] in want:
            want[parts[0]] = parts[1]
    return want


def _modpair_args(types_present, ligand_types) -> str:
    """
    --modpair string putting THIS project's Si/B vdW parameters into the
    ligand-internal energy AutoDock-GPU computes on the fly.

    AD-GPU's modpair branch takes (reqm_ij, eps_ij) already combined AND
    already weighted by the AD4 vdW coefficient — its default branch computes
    eps_ij = 0.1662 * sqrt(eps_i * eps_j) itself, the modpair branch does not.
    Getting that factor wrong scales the internal clash term by 6x, so it is
    applied here explicitly.
    """
    args = []
    for t in sorted(types_present):
        r_t, e_t = _CUSTOM_VDW[t]
        for x in sorted(set(ligand_types)):
            if x in _CUSTOM_VDW:
                r_x, e_x = _CUSTOM_VDW[x]
            elif x in _AD4_VDW:
                r_x, e_x = _AD4_VDW[x]
            else:
                continue
            # is_mod_pair is symmetric, so a pair of two CUSTOM types would
            # otherwise be emitted twice (Si:B and B:Si). Only that case is
            # deduplicated — skipping on plain string order would have dropped
            # Si:C, Si:HD, Si:OA and Si:SA, i.e. every pair that matters.
            if x in types_present and x < t:
                continue
            reqm = 0.5 * (r_t + r_x)
            eps = _AD4_COEFF_VDW * math.sqrt(e_t * e_x)
            args.append(f"{t}:{x},{reqm:.4f},{eps:.6f}")
    return "/".join(args)


def run_autodock(dpf_path: Path, timeout: int = 3600, seed=None) -> dict:
    """
    Execute AutoDock-GPU (preferred) or AutoDock4 CPU.

    AutoDock-GPU uses the same force field and scoring as AD4 but runs
    ~100-350x faster on GPU (Santos-Martins et al., JCTC 2021).

    THE DPF IS NOT PASSED TO AutoDock-GPU. It is written for the CPU engine and
    read here only as a settings source; everything AD-GPU honours goes in as
    an explicit flag (--nrun, --nev with --heuristics 0, --psize, --rmstol,
    --mrat, --crat, --ubmod, --seed, --modpair). See the module comment above
    for what --import_dpf silently dropped. The DPF settings that AD-GPU has no
    equivalent for are logged, not assumed.

    Returns dict with 'success', 'dlg_path', 'stderr', 'engine', and — for the
    GPU path — 'gpu_cmd', 'seed', 'settings_not_transferred'.
    """
    from .config import AUTODOCK4_BIN, USE_AUTODOCK_GPU, AUTODOCK_GPU_BIN
    dpf_path = Path(dpf_path)
    dlg_path = dpf_path.with_suffix(".dlg")

    if USE_AUTODOCK_GPU and AUTODOCK_GPU_BIN:
        fld_files = list(dpf_path.parent.glob("*.maps.fld"))
        ligand_name = _get_ligand_from_dpf(dpf_path)
        if fld_files and ligand_name:
            fld = fld_files[0]
            cfg = _dpf_settings(dpf_path)
            seeds = _resolve_seed(dpf_path, seed)

            cmd = [
                AUTODOCK_GPU_BIN,
                "--ffile", fld.name,
                "--lfile", ligand_name,
                "--resnam", dlg_path.stem,
                "--dlgoutput", "1",     # produce DLG (AD4-compatible)
                "--xmloutput", "0",     # skip XML
                "--clustering", "1",    # cluster analysis in DLG
                # REPRODUCIBILITY. Without this AD-GPU seeds from time+pid and
                # an identical docking returns a different number every time.
                "--seed", ",".join(str(s) for s in seeds),
                # --heuristics defaults to 1, and when it is on it OVERRIDES
                # --nev with a torsion-count estimate — that is why the
                # configured evaluation budget was inert.
                "--heuristics", "0",
            ]
            if cfg["ga_num_evals"]:
                cmd += ["--nev", str(int(float(cfg["ga_num_evals"])))]
            if cfg["ga_run"]:
                cmd += ["--nrun", str(int(float(cfg["ga_run"])))]
            if cfg["ga_pop_size"]:
                cmd += ["--psize", str(int(float(cfg["ga_pop_size"])))]
            if cfg["rmstol"]:
                cmd += ["--rmstol", str(float(cfg["rmstol"]))]
            if cfg["ga_mutation_rate"]:
                cmd += ["--mrat", f"{float(cfg['ga_mutation_rate']) * 100:g}"]
            if cfg["ga_crossover_rate"]:
                cmd += ["--crat", f"{float(cfg['ga_crossover_rate']) * 100:g}"]
            if (cfg["unbound_model"] or "bound") == "bound":
                cmd += ["--ubmod", "0"]
            # AD-GPU's convergence stop, made EXPLICIT so it is recorded in the
            # DLG header. AD4's DPF has no equivalent, so --nev is a CEILING on
            # the GPU: a converged run reports far fewer evaluations than
            # ga_num_evals and that is not a misconfiguration.
            cmd += ["--autostop", "1", "--stopstd", "0.15"]

            # ── non-standard atom types ──
            ligand_pdbqt = dpf_path.parent / ligand_name
            lig_types = (_extract_atom_types(ligand_pdbqt)
                         if ligand_pdbqt.exists() else [])
            lig_type_set = set(lig_types)
            map_types = _fld_map_types(fld)
            custom_present = sorted(lig_type_set & set(_CUSTOM_VDW))

            # A type that HAS its own AutoGrid map must NOT be sent through
            # --derivtype. The previous code passed `--derivtype Si=S`
            # unconditionally, which makes AD-GPU read sulfur's affinity map
            # for every silicon atom and throws away the UFF Si map AutoGrid
            # built from AD4_parameters_custom.dat. Measured on CD9/MPTMS with
            # a fixed seed: -3.17 kcal/mol with the real Si map, -3.43 with
            # `--derivtype Si=S`, -2.74 with `--derivtype Si=OA`.
            derivtypes = []
            for t in custom_present:
                if t in map_types:
                    continue
                base = next((b for b in _DERIV_FALLBACK.get(t, ())
                             if b in map_types), None)
                if base is None:
                    return {"success": False, "dlg_path": None,
                            "engine": None,
                            "stderr": (f"ligand type {t} has no map in "
                                       f"{fld.name} and no fallback base type "
                                       f"is mapped either — AutoGrid did not "
                                       f"build it. Refusing to dock: the "
                                       f"result would silently use the wrong "
                                       f"element.")}
                derivtypes.append(f"{t}={base}")
                logger.error(
                    "AutoGrid produced NO %s map in %s — falling back to "
                    "--derivtype %s=%s. THE CUSTOM %s PARAMETERS ARE NOT "
                    "APPLIED on the receptor side of this docking.",
                    t, fld.name, t, base, t)
            if derivtypes:
                cmd += ["--derivtype", "/".join(derivtypes)]

            if custom_present:
                mp = _modpair_args(custom_present, lig_type_set)
                if mp:
                    cmd += ["--modpair", mp]

            # ── say out loud what could NOT be handed over ──
            not_transferred = []
            if "parameter_file" in dpf_path.read_text():
                not_transferred.append(
                    "parameter_file: AD-GPU ignores it. Receptor-side Si/B "
                    "parameters reach the run only through the AutoGrid maps; "
                    "ligand-internal ones only through --modpair"
                    + (" (applied)" if custom_present else " (not needed)"))
            not_transferred.append(
                "ls_search_freq / set_psw1 / sw_*: AD-GPU uses ADADELTA local "
                "search at 100% rate; the DPF's Solis-Wets block does not "
                "apply and the GPU search protocol is NOT identical to AD4 CPU")
            not_transferred.append(
                "ga_num_evals is a CEILING on the GPU: --autostop 1 ends a run "
                "on convergence, so the DLG's 'evaluations performed' is "
                "normally well below the configured budget")
            logger.info("AutoDock-GPU settings not transferable from DPF: %s",
                        " | ".join(not_transferred))

            # THE STALE-DLG TRAP.  The DLG path is deterministic and was never
            # cleaned, and success was inferred purely from "a file exists and
            # is > 100 bytes".  So an AutoDock-GPU invocation that CRASHED on
            # run K silently returned run K-1's DLG, and Phase 2 scored it as a
            # fresh measurement — of a different receptor conformer.  Deleting
            # the file up front removes the possibility entirely, rather than
            # relying on the receiving end to notice.
            gpu_dlg = dpf_path.parent / f"{dlg_path.stem}.dlg"
            if gpu_dlg.exists():
                logger.debug(f"Removing pre-existing {gpu_dlg.name} before launch "
                             f"so a crash cannot return the previous run's DLG")
                try:
                    gpu_dlg.unlink()
                except OSError as e:
                    logger.error(f"could not remove stale {gpu_dlg}: {e} — a "
                                 f"failed run could return this file")

            try:
                result = subprocess.run(
                    cmd, cwd=str(dpf_path.parent),
                    capture_output=True, text=True, timeout=timeout,
                )
                # Three independent conditions, all required:
                #   (1) the process actually succeeded — `result` was captured
                #       and then never consulted before this fix;
                #   (2) a DLG exists (it cannot be a leftover: unlinked above);
                #   (3) it contains a docking RESULT, not merely bytes.
                rc_ok = result.returncode == 0
                dlg_ok = gpu_dlg.exists() and gpu_dlg.stat().st_size > 100
                has_result = dlg_ok and (
                    "Estimated Free Energy of Binding"
                    in gpu_dlg.read_text(errors="replace"))
                if rc_ok and dlg_ok and has_result:
                    logger.info(f"AutoDock-GPU completed: {gpu_dlg.name} "
                                f"(seed {seeds})")
                    return {"success": True, "dlg_path": gpu_dlg,
                            "stderr": "", "engine": "AutoDock-GPU",
                            "gpu_cmd": " ".join(cmd), "seed": list(seeds),
                            "settings_not_transferred": not_transferred}
                else:
                    logger.error(
                        f"AutoDock-GPU FAILED (rc={result.returncode}, "
                        f"dlg_present={dlg_ok}, has_binding_energy={has_result}); "
                        f"falling back to AD4 CPU. NOTE the two engines do NOT "
                        f"search identically — see settings_not_transferred.\n"
                        f"cmd: {' '.join(cmd)}\n"
                        f"stdout: {result.stdout[-500:]}\n"
                        f"stderr: {result.stderr[:500]}"
                    )
            except FileNotFoundError:
                logger.error("AutoDock-GPU binary not found, using AD4 CPU")
            except subprocess.TimeoutExpired:
                logger.error("AutoDock-GPU timed out, using AD4 CPU")
        else:
            logger.warning(
                "AutoDock-GPU skipped for %s: fld=%s ligand=%r",
                dpf_path.name, [f.name for f in fld_files], ligand_name)

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


# _write_gpu_compatible_dpf() used to live here. It filtered the DPF down to
# the tokens believed to be "supported by --import_dpf" and the caller then
# claimed the custom parameter file had been applied. --import_dpf is no longer
# used at all (see the module comment above run_autodock), so the filtered DPF
# would only be a second, misleading artefact on disk claiming to be the
# settings that were used. Removed rather than left dead.


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
    # AD4 DLG RANKING format:
    #   rank  sub  run    BE     clus_rmsd  ref_rmsd    RANKING
    #   1     1    45    -3.50    0.00       0.00        RANKING
    # or AD-GPU:
    #   33    1    32    +0.60    0.00       13.92       RANKING
    cluster_pattern = re.compile(
        r"^\s*(\d+)\s+\d+\s+\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+[\d.]+\s+RANKING",
        re.MULTILINE,
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
                "cluster_size": 1,
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


def get_best_energy(dlg_path: Path):
    """Return the best (most negative) binding energy from DLG, or None.

    Returns None — NOT 0.0 — when the DLG yields no clusters. See
    get_mean_best_cluster_energy for why.
    """
    clusters = parse_dlg(dlg_path)
    if not clusters:
        logger.error(
            f"NO DOCKING RESULT in {dlg_path}: parse_dlg found zero clusters. "
            f"Returning None (a missing measurement), not 0.0.")
        return None
    return min(c["binding_energy"] for c in clusters)


def get_mean_best_cluster_energy(dlg_path: Path):
    """
    Return mean binding energy of the best cluster (rank 1), or None.

    This is the metric used by Rajpal et al. 2024 for comparison.

    RETURNS None ON FAILURE, NOT 0.0 (REVIEW FINDING 12).

    0.0 is a value ON the kcal/mol measurement scale — it is the weakest
    possible binding, not a sentinel. A DLG that satisfies phase 2's
    freshness+content gate but whose model records are unusable (truncated
    file, killed job) parsed to zero clusters and returned 0.0, which then
    entered be_matrix and mmsd_sum as a MEASUREMENT: every consumer guards
    only `is None` (phase2_smd.py:161, phase3_mmsd.py:1083 and :1137), so 0.0
    sailed through and shifted every rank below it.
    """
    clusters = parse_dlg(dlg_path)
    if not clusters:
        logger.error(
            f"NO DOCKING RESULT in {dlg_path}: parse_dlg found zero clusters "
            f"(truncated or unparseable DLG). Returning None so the monomer is "
            f"recorded as MISSING, not as 0.0 kcal/mol — 0.0 is the weakest "
            f"real binding energy on this scale, not 'no data'.")
        return None
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

    # Extract ligand coordinates from PDBQT.
    # The AD4 TYPE is dropped (it is not a PDB field) but the ELEMENT is
    # written into cols 77-78, which is a real PDB field. Without it the
    # receptor-prep step has to guess the element from the atom NAME, and for
    # a silane that guess is "S" — the same Si->S confusion the ligand prep
    # goes to some trouble to avoid.
    _TYPE_TO_ELEMENT = {"A": "C", "CG": "C", "CX": "C", "G0": "C",
                        "NA": "N", "NS": "N", "NX": "N",
                        "OA": "O", "OS": "O", "OX": "O",
                        "SA": "S", "HD": "H", "HS": "H"}
    ligand_lines = []
    for line in ligand_pdbqt.read_text().split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            atype = line[77:79].strip() if len(line) >= 78 else ""
            elem = _TYPE_TO_ELEMENT.get(atype, atype)
            # Convert PDBQT to PDB format (strip charge + AD4 type columns)
            pdb_line = line[:66].ljust(76) + f"{elem:>2s}"
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
                ga_runs: int = 50,
                expected_smiles: str = None,
                seed=None) -> dict:
    """
    Complete single docking: AutoGrid → AutoDock → parse results.

    expected_smiles
        When given, the ligand PDBQT's SMILES stamp must match it or the
        docking is refused. This is the check that stops a corrected library
        entry from being scored as its retired predecessor.

    Returns dict with:
      - binding_energy (best cluster)
      - mean_cluster_energy (rank-1 cluster mean)
      - n_clusters
      - dlg_path
      - best_pose_path (PDBQT)
    """
    from .utils_structure import (copy_smiles_stamp, verify_smiles_stamp)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdbqt = Path(receptor_pdbqt)
    ligand_pdbqt = Path(ligand_pdbqt)

    if expected_smiles:
        # Check the SOURCE before it is copied anywhere.
        verify_smiles_stamp(ligand_pdbqt, ligand_pdbqt.stem, expected_smiles,
                            require_stamp=True)

    # Copy input files to work_dir.
    # `if not local.exists()` is a CACHE, and it used to be a cache keyed on
    # the file NAME only: a work dir left over from an earlier run kept its old
    # MONOMER.pdbqt and the new source was never copied in. Content is compared
    # now, and a differing copy is replaced (loudly) instead of reused.
    rec_local = work_dir / receptor_pdbqt.name
    lig_local = work_dir / ligand_pdbqt.name
    for src, dst, what in ((receptor_pdbqt, rec_local, "receptor"),
                           (ligand_pdbqt, lig_local, "ligand")):
        if dst.exists():
            if dst.read_bytes() == src.read_bytes():
                continue
            logger.warning(
                "%s: cached %s %s differs from %s — replacing the stale copy",
                work_dir.name, what, dst.name, src)
        shutil.copy2(str(src), str(dst))

    # carry the provenance into the work dir with the file
    copy_smiles_stamp(ligand_pdbqt, lig_local, ligand_pdbqt.stem)

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
    dock_result = run_autodock(dpf, seed=seed)
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

    # An engine that exited 0 but produced an unparseable DLG has not docked
    # anything (REVIEW FINDING 12). Record that as a FAILURE rather than
    # returning success alongside a None energy — the two used to be
    # indistinguishable from a successful docking whose energy happened to be
    # 0.0, because that is what the getters returned.
    parse_ok = mean_e is not None
    if not parse_ok:
        logger.error(
            f"Docking produced a DLG that yielded no clusters: {dlg}. "
            f"Recorded as a FAILED docking (binding_energy=None); the monomer "
            f"will be reported as MISSING, not as 0.0 kcal/mol.")

    return {
        "binding_energy": best_e,
        "mean_cluster_energy": mean_e,
        "success": parse_ok,
        "error": None if parse_ok else "DLG produced no parseable clusters",
        "n_clusters": len(clusters),
        "dlg_path": str(dlg),
        "best_pose_path": str(best_pose) if best_pose.exists() else None,
        "clusters": clusters[:5],  # top-5 clusters for reporting
        # provenance: which engine, which seed, what could not be transferred
        "engine": dock_result.get("engine"),
        "seed": dock_result.get("seed"),
        "settings_not_transferred": dock_result.get(
            "settings_not_transferred"),
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

    # Si: UFF Si_3 (tetrahedral silicon)  Rii = 4.295 A, epsii = 0.402
    # B:  UFF B_2 (trigonal boron, boronic acid B(OH)2)
    #                                     Rii = 4.083 A, epsii = 0.180
    #   Rij_hb = 0 (no H-bond), hbond = 0
    #
    # Rii/epsii come from _CUSTOM_VDW so this file and the --modpair arguments
    # run_autodock builds for AutoDock-GPU cannot drift apart. AutoGrid reads
    # this file; AutoDock-GPU never does.
    _vol_solpar = {"Si": (12.175, -0.00143), "B": (11.000, -0.00110)}
    lines = [
        "# AutoDock4 custom parameters for non-standard atom types",
        "# Source: UFF (Rappe et al., JACS 1992, 114, 10024-10035)",
        "# Per: autodock.scripps.edu/how-to-add-new-atom-types-to-the-"
        "autodock-force-field/",
        "#",
        "# Si: for silane monomers (PTES, APTES, APTMS, TEOS, etc.)",
        "# B:  for boronic acid monomers (APBA/VPBA/FPBA/AAPBA)",
        "#",
        "# NOTE: AutoDock-GPU does NOT read this file. Its built-in table has",
        "#       SI (4.10, 0.200) and B (3.84, 0.155), which are NOT these",
        "#       values; run_autodock overrides them with --modpair.",
        "#",
        "# atom_par  Rii    epsii   vol      solpar    Rij_hb epsij_hb "
        "hbond rec map",
    ]
    for t, (rii, eps) in sorted(_CUSTOM_VDW.items()):
        vol, solpar = _vol_solpar[t]
        lines.append(f"atom_par {t:<3s} {rii:6.3f} {eps:6.3f} {vol:8.3f} "
                     f"{solpar:9.5f}  0.0  0.0  0  -1  -1")
    param_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Custom AD4 parameters ({', '.join(sorted(_CUSTOM_VDW))}) "
                f"written: {param_path}")
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
