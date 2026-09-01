"""EV (extracellular vesicle) template + fresh-EV builder for membrane MIP protocol.

This module consumes CHARMM-GUI Membrane Builder outputs living under
    structures/membrane/<target>/
        step5_input.gro
        step5_input.top
        step5_input.pdb
        toppar/
            forcefield.itp
            PROA.itp                     (protein)
            {POPC,POPE,PSM,POPS,CHL1}.itp (HEK293T EV mimic lipids)
            {SOD,CLA}.itp                 (Na+/Cl- CHARMM naming)
            TIP3.itp                      (water)
        step6.{1,2,3}_equilibration.mdp   (CHARMM-GUI recommended schedule)

Two public entry points:

    load_template_ev(target, out_dir) -> dict
        Copies a target's CHARMM-GUI outputs into a Phase-4 working directory
        and returns {'gro','top','toppar_dir','pdb','box_nm'}. Fresh-run only
        (skip if the working dir already has step5_input.*).

    build_fresh_ev(target, seed, out_path) -> Path
        Emits a rotated, decoupled copy of the CHARMM-GUI system suitable for
        placement above a Phase-5 cavity. Removes solvent + ions (they are
        rebuilt during Phase 5 setup around the placement) but retains
        protein + lipid bilayer (this IS the fresh EV).

Both functions FAIL CLEAN if the CHARMM-GUI inputs are absent — the caller
should raise a Phase-4 blocking error rather than silently building an
aqueous-only system when membrane mode is on.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MEMBRANE_LIPID_RESNAMES", "GLYCAN_RESNAMES_CHARMM",
    "SOLVENT_RESNAMES", "ION_RESNAMES_CHARMM",
    "load_template_ev", "build_fresh_ev",
    "prepare_independent_fresh_evs",
    "read_gro_box_nm", "MembraneInputMissingError",
]


MEMBRANE_LIPID_RESNAMES = ("POPC", "POPE", "PSM", "POPS", "CHL1")

# N-glycan sugar residue names emitted by CHARMM-GUI Glycan Reader. Names
# here are the FULL CHARMM residue identifiers; the GROMACS .gro format
# truncates the residue name to 5 characters (cols 6-10), so downstream
# matching (see `_split_atoms_by_class`) compares against the 5-char prefix
# of each entry — e.g. BGLCNA in the .top matches "BGLCN" in the .gro.
GLYCAN_RESNAMES_CHARMM = (
    "BGLCNA", "BGLC", "AGLC",       # β/α-glucose, β-N-acetylglucosamine
    "BMAN", "AMAN",                 # β/α-mannose
    "BGAL", "AGAL",                 # β/α-galactose
    "ANE5", "SIA",                  # N-acetylneuraminic acid / sialic acid
    "AFUC", "FUC",                  # α-fucose
    "NAG", "MAN",                   # legacy PDB names
)

# Precomputed 5-char keys for .gro-column matching (BGLCNA -> "BGLCN").
_GLYCAN_GRO_KEYS = frozenset(name[:5] for name in GLYCAN_RESNAMES_CHARMM)

SOLVENT_RESNAMES = ("TIP3", "SOL", "HOH")
ION_RESNAMES_CHARMM = ("SOD", "CLA")   # CHARMM-GUI naming for Na+ / Cl-

# 20 standard + common his tautomer resnames. Anything outside this set
# and outside the lipid/solvent/ion sets is treated as a MONOMER.
STANDARD_AMINO_ACIDS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
    # CHARMM his tautomers + terminal + patched forms occasionally seen
    "HSD", "HSE", "HSP", "HID", "HIE", "HIP",
})


class MembraneInputMissingError(FileNotFoundError):
    """Raised when structures/membrane/<target>/ is absent or incomplete."""


def _membrane_dir(target: str) -> Path:
    """Return the on-disk directory for a target's CHARMM-GUI outputs.

    Read from pipeline.config.PHASE4_MEMBRANE_INPUT_DIR (relative to project
    root, resolved via PROJECT_ROOT) so a caller can override the location.
    """
    from . import config as cfg
    root = Path(getattr(cfg, "PROJECT_ROOT", "."))
    subdir = getattr(cfg, "PHASE4_MEMBRANE_INPUT_DIR", "structures/membrane")
    return root / subdir / target


def _require_charmm_gui_output(target: str) -> Path:
    d = _membrane_dir(target)
    missing = [f for f in ("step5_input.gro", "step5_input.top", "toppar")
               if not (d / f).exists()]
    if missing:
        raise MembraneInputMissingError(
            f"CHARMM-GUI membrane outputs incomplete for {target} at {d}: "
            f"missing {missing}. Run CHARMM-GUI Membrane Builder for {target} "
            f"and copy step5_input.{{gro,top,pdb}} + toppar/ into that directory.")
    return d


def read_gro_box_nm(gro_path: Path) -> tuple[float, float, float]:
    """Return (x, y, z) box vectors in nm from a .gro file's last line."""
    with open(gro_path) as fh:
        last = fh.readlines()[-1]
    parts = last.split()
    if len(parts) < 3:
        raise ValueError(f"{gro_path}: unrecognised box line {last!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def load_template_ev(target: str, out_dir: Path) -> dict:
    """Copy CHARMM-GUI target outputs into `out_dir` as the Phase-4 starting
    system.

    Idempotent: skips if out_dir already contains step5_input.gro. Returns
    {'gro','top','toppar_dir','pdb','box_nm'}.
    """
    src = _require_charmm_gui_output(target)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_gro = out_dir / "step5_input.gro"
    dst_top = out_dir / "step5_input.top"
    dst_pdb = out_dir / "step5_input.pdb"
    dst_toppar = out_dir / "toppar"

    if not dst_gro.exists():
        shutil.copy2(src / "step5_input.gro", dst_gro)
    if not dst_top.exists():
        shutil.copy2(src / "step5_input.top", dst_top)
    if not dst_pdb.exists() and (src / "step5_input.pdb").exists():
        shutil.copy2(src / "step5_input.pdb", dst_pdb)
    if not dst_toppar.exists():
        shutil.copytree(src / "toppar", dst_toppar)
    # CHARMM-GUI equilibration mdps — copy the six-step schedule for reuse.
    for mdp in src.glob("step6.*_equilibration.mdp"):
        target_mdp = out_dir / mdp.name
        if not target_mdp.exists():
            shutil.copy2(mdp, target_mdp)

    return {
        "gro":        dst_gro,
        "top":        dst_top,
        "toppar_dir": dst_toppar,
        "pdb":        dst_pdb if dst_pdb.exists() else None,
        "box_nm":     read_gro_box_nm(dst_gro),
    }


def _parse_gro_lines(gro_path: Path) -> tuple[str, list[str], str]:
    """Return (title_line, atom_lines, box_line)."""
    with open(gro_path) as fh:
        lines = fh.readlines()
    title = lines[0]
    natoms = int(lines[1].strip())
    atoms = lines[2:2 + natoms]
    box = lines[2 + natoms]
    return title, atoms, box


def _split_atoms_by_class(atoms: list[str]) -> dict:
    """Bucket atom lines by molecule class using resname (cols 5-10 in .gro).

    Returns {'protein', 'lipid', 'glycan', 'solvent', 'ion', 'monomer'}.
    Anything not matching the known lipid/glycan/solvent/ion resnames and
    not a standard amino-acid residue name falls into 'monomer'.

    CHARMM-GUI Glycan Reader emits sugars (BGLCNA, BMAN, AMAN, …) as
    covalently-anchored residues on the protein N-glycosylation sites.
    These must be kept in the 'glycan' bucket so `build_fresh_ev` can
    retain them alongside the protein — dropping them would strip
    CD63's N130/N150/N172 glycans, breaking APBA boronate-diol
    recognition in Phase 5 rebinding.
    """
    out = {"protein": [], "lipid": [], "glycan": [],
           "solvent": [], "ion": [], "monomer": []}
    for ln in atoms:
        if len(ln) < 10:
            continue
        resname = ln[5:10].strip()
        if resname in MEMBRANE_LIPID_RESNAMES:
            out["lipid"].append(ln)
        elif resname in _GLYCAN_GRO_KEYS:
            out["glycan"].append(ln)
        elif resname in SOLVENT_RESNAMES:
            out["solvent"].append(ln)
        elif resname in ION_RESNAMES_CHARMM or resname in ("NA", "CL", "K"):
            out["ion"].append(ln)
        elif resname in STANDARD_AMINO_ACIDS:
            out["protein"].append(ln)
        else:
            out["monomer"].append(ln)
    return out


def build_fresh_ev(target: str, seed: int, out_path: Path,
                    rotate_z_deg: float | None = None,
                    drop_solvent: bool = True) -> dict:
    """Emit a fresh CD-in-EV coordinate file for Phase 5 rebinding.

    Parameters
    ----------
    target : str
        Target name (CD9/CD81/CD63) — must have a corresponding CHARMM-GUI
        output under structures/membrane/<target>/.
    seed : int
        Deterministic seed. Applied via numpy.random.default_rng — controls
        the random z-axis rotation used to place independent EV replicas
        differently.
    out_path : Path
        Where to write the fresh-EV .gro.
    rotate_z_deg : float, optional
        Explicit rotation angle in degrees around the z-axis. If None, drawn
        from the seed.
    drop_solvent : bool
        Strip water + ions from the fresh EV (they are re-solvated around
        the composite MIP + fresh-EV system by the Phase 5 setup step).

    Returns
    -------
    dict with keys:
        path : Path — the written .gro file (also present as `out_path`
               for callers that pattern-match on the input argument name)
        n_atoms          : int — total atoms retained
        n_protein_atoms  : int — protein atoms retained
        n_lipid_atoms    : int — bilayer atoms retained
        n_glycan_atoms   : int — covalent N-glycan atoms retained
                                 (D2 verification: MUST be > 0 for CD63,
                                  ≈ 354 for Man3GlcNAc2 × 3 sequons)
        n_solvent_atoms  : int — waters kept (0 when drop_solvent=True)
        n_ion_atoms      : int — ions kept (0 when drop_solvent=True)
        rotate_z_deg     : float — z-rotation actually applied
    """
    src = _require_charmm_gui_output(target)
    src_gro = src / "step5_input.gro"

    title, atoms, box = _parse_gro_lines(src_gro)
    box_nm = read_gro_box_nm(src_gro)

    # ── select which molecule classes to retain ─────────────────
    # CRITICAL (D2 fix): 'glycan' bucket is ALWAYS retained. Fresh CD-in-EV
    # must carry CD63's covalent N-glycans (N130/N150/N172, Man3GlcNAc2 core
    # ≈ 3 sites × ~118 atoms). Stripping them silently breaks APBA
    # boronate-diol recognition downstream in Phase 5 rebinding.
    buckets = _split_atoms_by_class(atoms)
    if drop_solvent:
        keep = buckets["protein"] + buckets["lipid"] + buckets["glycan"]
    else:
        keep = (buckets["protein"] + buckets["lipid"] + buckets["glycan"]
                + buckets["solvent"] + buckets["ion"])
    if not keep:
        raise ValueError(f"build_fresh_ev({target}): retained zero atoms — "
                          f"resname parsing likely wrong for this CHARMM-GUI "
                          f"output; inspect step5_input.gro")

    # ── random z-axis rotation (independent orientation per replica) ────
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(rotate_z_deg if rotate_z_deg is not None
                        else rng.uniform(0, 360))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy, cz = box_nm[0] / 2, box_nm[1] / 2, 0.0   # rotate around box XY-centre

    def _rotate(x, y, z):
        dx, dy = x - cx, y - cy
        return (cx + dx * cos_t - dy * sin_t,
                cy + dx * sin_t + dy * cos_t,
                z)

    rotated_lines = []
    for i, ln in enumerate(keep, start=1):
        # .gro fixed columns: resid(5) resname(5) atomname(5) atomid(5) x(8) y(8) z(8)
        try:
            x = float(ln[20:28]); y = float(ln[28:36]); z = float(ln[36:44])
        except ValueError:
            rotated_lines.append(ln)
            continue
        nx, ny, nz = _rotate(x, y, z)
        prefix = ln[:15]
        rotated_lines.append(f"{prefix}{i:5d}{nx:8.3f}{ny:8.3f}{nz:8.3f}\n")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(f"fresh-EV {target} seed={seed} rot_z={np.rad2deg(theta):.1f}deg\n")
        fh.write(f"{len(rotated_lines):>5d}\n")
        fh.writelines(rotated_lines)
        fh.write(box)
    return {
        "path":            out_path,
        "n_atoms":         len(rotated_lines),
        "n_protein_atoms": len(buckets["protein"]),
        "n_lipid_atoms":   len(buckets["lipid"]),
        "n_glycan_atoms":  len(buckets["glycan"]),
        "n_solvent_atoms": len(buckets["solvent"]) if not drop_solvent else 0,
        "n_ion_atoms":     len(buckets["ion"]) if not drop_solvent else 0,
        "rotate_z_deg":    float(np.rad2deg(theta)),
    }


# ── BLOCKER C4: independent fresh-EV ensemble via NVT perturbations ──

def _resolve_gmx_bin() -> str | None:
    """Return the configured gmx binary path, or None if gmx is unavailable.

    Deferred import so this module can be imported (and stub-tested) in
    environments where GROMACS is not installed. Returns None cleanly when the
    binary cannot be located — the caller is expected to fall back to a stub
    ensemble.
    """
    try:
        from . import config as cfg
        gmx = getattr(cfg, "GMX_BIN", "gmx")
    except Exception:
        gmx = "gmx"
    # If it's a bare name, resolve via PATH; if an absolute path, existence-check.
    if Path(gmx).is_absolute():
        return gmx if Path(gmx).exists() else None
    return shutil.which(gmx)


def _external_replica_dir(target: str, replica_id: int) -> Path:
    """Path to a user-supplied CHARMM-GUI replica for a target.

    Layout expected under structures/membrane/<target>/replicas/replica_<i>/:
        step5_input.gro / step5_input.top / toppar/  (+ optional pdb / mdps)
    """
    return _membrane_dir(target) / "replicas" / f"replica_{int(replica_id)}"


def prepare_independent_fresh_evs(target: str, n_replicas: int,
                                   out_dir: Path,
                                   seed_base: int = 42,
                                   nvt_time_ps: float = 200.0,
                                   mode: str = "nvt_perturbation",
                                   gmx_bin: str | None = None) -> list[dict]:
    """Produce N genuinely independent fresh-EV configurations.

    Addresses BLOCKER C4: the legacy path used N z-rotations of the SAME
    CHARMM-GUI structure — statistically N=1 with rotational sampling. This
    helper builds N configurations that differ in lipid registry and
    protein side-chain conformation, giving a real ensemble.

    Modes
    -----
    "nvt_perturbation" (default)
        For each replica i:
          1. Copy CHARMM-GUI step5_input.{gro,top,pdb} + toppar/ to
             <out_dir>/replica_<i>/
          2. Run EM + a short NVT (nvt_time_ps ps) using gen_seed =
             seed_base + i*10007 for Maxwell-Boltzmann velocity draw.
          3. Retain nvt.gro as the replica's fresh-EV starting coordinates.
        Cached: an existing replica dir with a valid nvt.gro is skipped.
        If GROMACS is not on the PATH, the actual MD runs are SKIPPED and
        the returned dict has status="stub" — safe for import-only tests.

    "external_replicas"
        Locates structures/membrane/<target>/replicas/replica_<i>/step5_input.gro
        for each i in range(n_replicas). Errors if any replica dir is missing.
        Each replica is a truly independent CHARMM-GUI build (different Step 3
        lipid-packing seed) — the manuscript-grade path.

    "rotation_only"
        Legacy behaviour: returns N stub records pointing at the SAME
        CHARMM-GUI step5_input.gro; caller is responsible for applying
        distinct rotations via build_fresh_ev(seed=...). Preserved for
        backwards-compatible dispatch.

    Returns
    -------
    list of dict, one per replica:
        {
          'replica_id':    int,               # 0..n_replicas-1
          'seed':          int,               # gen_seed used
          'gro':           Path,              # coordinates to use for placement
          'top':           Path or None,      # matching topology
          'toppar_dir':    Path or None,
          'mode':          str,               # echoes the input `mode`
          'status':        'ready' | 'stub' | 'cached',
          'nvt_time_ps':   float,             # 0 when mode != 'nvt_perturbation'
          'source':        str,               # where the coordinates came from
        }
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_replicas = max(1, int(n_replicas))
    records: list[dict] = []

    # ── external_replicas: locate pre-built CHARMM-GUI outputs ──
    if mode == "external_replicas":
        for i in range(n_replicas):
            rdir = _external_replica_dir(target, i)
            gro = rdir / "step5_input.gro"
            top = rdir / "step5_input.top"
            if not gro.exists() or not top.exists():
                raise MembraneInputMissingError(
                    f"external_replicas mode: replica {i} missing at {rdir} "
                    f"(need step5_input.gro + step5_input.top). See "
                    f"structures/membrane/README.md for how to build them.")
            records.append({
                "replica_id":  i,
                "seed":        seed_base + i * 10007,
                "gro":         gro,
                "top":         top,
                "toppar_dir":  (rdir / "toppar") if (rdir / "toppar").exists() else None,
                "mode":        "external_replicas",
                "status":      "ready",
                "nvt_time_ps": 0.0,
                "source":      str(rdir),
            })
        return records

    # ── rotation_only: legacy stubs — one per replica, pointing at the ──
    #    same CHARMM-GUI source. Caller applies rotations via build_fresh_ev.
    if mode == "rotation_only":
        src = _require_charmm_gui_output(target)
        for i in range(n_replicas):
            records.append({
                "replica_id":  i,
                "seed":        seed_base + i * 10007,
                "gro":         src / "step5_input.gro",
                "top":         src / "step5_input.top",
                "toppar_dir":  src / "toppar",
                "mode":        "rotation_only",
                "status":      "stub",
                "nvt_time_ps": 0.0,
                "source":      str(src),
            })
        return records

    if mode != "nvt_perturbation":
        raise ValueError(
            f"prepare_independent_fresh_evs: unknown mode {mode!r} — "
            f"expected one of 'rotation_only', 'nvt_perturbation', "
            f"'external_replicas'")

    # ── nvt_perturbation: run N short NVTs with distinct gen_vel seeds ──
    src = _require_charmm_gui_output(target)

    gmx = gmx_bin or _resolve_gmx_bin()
    can_run_md = gmx is not None
    if not can_run_md:
        logger.warning(
            "prepare_independent_fresh_evs(%s): gmx binary not found; "
            "returning %d stub replicas (no NVT actually run). Set GMX_BIN or "
            "install GROMACS to produce a genuine independent ensemble.",
            target, n_replicas)

    # Deferred imports so this module stays importable without utils_gromacs.
    if can_run_md:
        try:
            from .utils_gromacs import (run_energy_minimization,
                                         run_nvt_equilibration,
                                         _write_membrane_index)
        except Exception as e:
            logger.warning(
                "prepare_independent_fresh_evs(%s): unable to import gromacs "
                "helpers (%s); falling back to stub replicas.", target, e)
            can_run_md = False

    for i in range(n_replicas):
        rdir = out_dir / f"replica_{i}"
        rdir.mkdir(parents=True, exist_ok=True)
        seed_i = int(seed_base + i * 10007)

        # Copy CHARMM-GUI inputs into the replica working dir (cached).
        for name in ("step5_input.gro", "step5_input.top", "step5_input.pdb"):
            src_f = src / name
            dst_f = rdir / name
            if src_f.exists() and not dst_f.exists():
                shutil.copy2(src_f, dst_f)
        if not (rdir / "toppar").exists() and (src / "toppar").is_dir():
            shutil.copytree(src / "toppar", rdir / "toppar")

        # Grompp / mdrun expects the working topology as `topol.top` and the
        # starting coords as `ionized.gro` (to match utils_gromacs's mdp
        # template file discovery — see run_energy_minimization).
        top_working = rdir / "topol.top"
        gro_working = rdir / "ionized.gro"
        if not top_working.exists():
            shutil.copy2(rdir / "step5_input.top", top_working)
        if not gro_working.exists():
            shutil.copy2(rdir / "step5_input.gro", gro_working)

        nvt_gro = rdir / "nvt.gro"

        # ── cached: replica already built ──
        if nvt_gro.exists() and nvt_gro.stat().st_size > 50:
            records.append({
                "replica_id":  i,
                "seed":        seed_i,
                "gro":         nvt_gro,
                "top":         top_working,
                "toppar_dir":  rdir / "toppar",
                "mode":        "nvt_perturbation",
                "status":      "cached",
                "nvt_time_ps": float(nvt_time_ps),
                "source":      str(rdir),
            })
            continue

        # ── stub: gmx unavailable, log the plan and move on ──
        if not can_run_md:
            records.append({
                "replica_id":  i,
                "seed":        seed_i,
                "gro":         rdir / "step5_input.gro",  # fallback: original
                "top":         top_working,
                "toppar_dir":  rdir / "toppar",
                "mode":        "nvt_perturbation",
                "status":      "stub",
                "nvt_time_ps": 0.0,
                "source":      str(rdir),
                "note":        "gmx unavailable — NVT skipped; using CHARMM-GUI "
                                "start coords (statistically equivalent to N=1)",
            })
            continue

        # ── actually run EM + short NVT ──
        try:
            logger.info(
                "prepare_independent_fresh_evs(%s): replica %d/%d EM+NVT "
                "gen_seed=%d nvt_time_ps=%.0f in %s",
                target, i + 1, n_replicas, seed_i, nvt_time_ps, rdir)
            run_energy_minimization(rdir)
            # Build TC1/TC2 index for the two-group membrane thermostat and
            # run the perturbation NVT with -DPOSRES_MEMBRANE + tau_t=1.0 ps,
            # so the "independent" replica reflects the equilibrium ensemble
            # (perturbing velocities only) rather than a deformed bilayer.
            # No monomer is merged into the fresh EV at this stage, so
            # monomer_resnames is empty; TC1 gets protein+glycan, TC2 gets
            # bilayer+water+ions.
            index_ndx = rdir / "index.ndx"
            group_counts = _write_membrane_index(
                gro_working, index_ndx,
                lipid_resnames=MEMBRANE_LIPID_RESNAMES,
                monomer_resnames=())
            if group_counts["TC1"] == 0 or group_counts["TC2"] == 0:
                raise RuntimeError(
                    f"prepare_independent_fresh_evs: empty tc-grp "
                    f"({group_counts}) for replica {i} in {rdir}")
            run_nvt_equilibration(rdir, time_ps=float(nvt_time_ps),
                                   gen_seed=seed_i,
                                   define='-DPOSRES_MEMBRANE',
                                   tc_grps='TC1 TC2',
                                   tau_t='1.0 1.0',
                                   index=index_ndx)
            records.append({
                "replica_id":  i,
                "seed":        seed_i,
                "gro":         nvt_gro,
                "top":         top_working,
                "toppar_dir":  rdir / "toppar",
                "mode":        "nvt_perturbation",
                "status":      "ready",
                "nvt_time_ps": float(nvt_time_ps),
                "source":      str(rdir),
            })
        except Exception as e:
            logger.error(
                "prepare_independent_fresh_evs(%s): replica %d NVT FAILED "
                "(%s) — retaining CHARMM-GUI start coords as fallback so the "
                "leg can still run; ensemble is degraded for this replica.",
                target, i, e)
            records.append({
                "replica_id":  i,
                "seed":        seed_i,
                "gro":         rdir / "step5_input.gro",
                "top":         top_working,
                "toppar_dir":  rdir / "toppar",
                "mode":        "nvt_perturbation",
                "status":      "stub",
                "nvt_time_ps": 0.0,
                "source":      str(rdir),
                "error":       str(e),
            })
    return records
