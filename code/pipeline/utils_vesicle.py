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

import shutil
from pathlib import Path

import numpy as np

__all__ = [
    "MEMBRANE_LIPID_RESNAMES", "SOLVENT_RESNAMES", "ION_RESNAMES_CHARMM",
    "load_template_ev", "build_fresh_ev",
    "read_gro_box_nm", "MembraneInputMissingError",
]


MEMBRANE_LIPID_RESNAMES = ("POPC", "POPE", "PSM", "POPS", "CHL1")
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

    Returns {'protein', 'lipid', 'solvent', 'ion', 'monomer'}.
    Anything not matching the known lipid/solvent/ion resnames and not a
    standard amino-acid residue name falls into 'monomer'.
    """
    out = {"protein": [], "lipid": [], "solvent": [], "ion": [], "monomer": []}
    for ln in atoms:
        if len(ln) < 10:
            continue
        resname = ln[5:10].strip()
        if resname in MEMBRANE_LIPID_RESNAMES:
            out["lipid"].append(ln)
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
                    drop_solvent: bool = True) -> Path:
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
    Path : the written file path.
    """
    src = _require_charmm_gui_output(target)
    src_gro = src / "step5_input.gro"

    title, atoms, box = _parse_gro_lines(src_gro)
    box_nm = read_gro_box_nm(src_gro)

    # ── select which molecule classes to retain ─────────────────
    buckets = _split_atoms_by_class(atoms)
    if drop_solvent:
        keep = buckets["protein"] + buckets["lipid"]
    else:
        keep = (buckets["protein"] + buckets["lipid"]
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
    return out_path
