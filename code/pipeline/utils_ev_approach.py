"""EV-templated MIP rebinding — fresh CD-in-EV approach onto a Triton-lysed cavity.

This module implements the Phase 5 rebinding path used when the pipeline is
running the EV-templated protocol:

    Phase 4 (this file NOT involved): CD-in-nanovesicle + monomers polymerised.
    Between Phase 4 and Phase 5 (utils_triton_removal.finalize_triton_removal):
        lipid + template CD deleted from Phase 4 final system → 'cavity.gro'.
    Phase 5 (this file):
        For each snapshot (= replica) and each fresh-EV placement seed:
            (1) build_ev_approach_system(): merge cavity.gro + build_fresh_ev(),
                extend box in +Z, solvate added slab, neutralise ions.
            (2) EM + NVT + NPT + production MD (monomers restrained, fresh EV free).
            (3) analyse persistent contacts on the FRESH EV's protein selection
                only (Triton-lysed cavity has no residual template CD atoms).

The classical `_run_rebinding_md` in phase5_rebinding.py is preserved for the
non-membrane pipeline. `run_ev_approach_leg` here is the entry point the
dispatcher in phase5_rebinding invokes when both PHASE4_MEMBRANE_MODE and
PHASE5_TRITON_REMOVAL_MODE are True.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .utils_vesicle import (build_fresh_ev, read_gro_box_nm,
                              MEMBRANE_LIPID_RESNAMES,
                              GLYCAN_RESNAMES_CHARMM,
                              SOLVENT_RESNAMES,
                              ION_RESNAMES_CHARMM,
                              STANDARD_AMINO_ACIDS)

logger = logging.getLogger(__name__)


# ── coordinate merge + box extension ─────────────────────────

def _read_gro(gro_path: Path):
    """Return (title, atoms, box_line) for a .gro file."""
    with open(gro_path) as fh:
        lines = fh.readlines()
    title = lines[0].rstrip("\n")
    natoms = int(lines[1].strip())
    atoms = lines[2:2 + natoms]
    box_line = lines[2 + natoms]
    return title, atoms, box_line


def _atom_z_nm(atom_line: str) -> float:
    """Extract z-coord (nm) from a .gro atom line. Returns 0.0 on parse failure."""
    try:
        return float(atom_line[36:44])
    except (ValueError, IndexError):
        return 0.0


def _shift_atom(atom_line: str, dx: float = 0.0, dy: float = 0.0,
                dz: float = 0.0, renumber: int | None = None) -> str:
    """Return the same .gro atom line with coordinates shifted by (dx,dy,dz).

    `renumber` (if given) replaces the atom index column (fixed cols 15-20).
    """
    try:
        x = float(atom_line[20:28]); y = float(atom_line[28:36]); z = float(atom_line[36:44])
    except (ValueError, IndexError):
        return atom_line
    prefix = atom_line[:15]
    idx = f"{renumber:5d}" if renumber is not None else atom_line[15:20]
    coord = f"{x + dx:8.3f}{y + dy:8.3f}{z + dz:8.3f}"
    trailer = atom_line[44:]         # velocities if present
    return f"{prefix}{idx}{coord}{trailer}".rstrip("\n") + "\n"


def _strip_solvent_from_gro(in_gro: Path, out_gro: Path) -> dict:
    """Remove SOL/HOH/TIP3 waters and monatomic ions from a .gro file in-place.

    BUG #15 fix: prepare_independent_fresh_evs returns a solvated NVT-relaxed
    fresh-EV coordinate file (nvt.gro), so waters wrapped through +Z PBC would
    otherwise touch the Triton-lysed cavity after the merge. This helper keeps
    the fresh-EV protein + glycans + lipids only; the composite system is
    re-solvated after the merge by `gmx solvate`.
    """
    with open(in_gro) as fh:
        lines = fh.readlines()
    title = lines[0]
    natoms = int(lines[1].strip())
    atoms = lines[2:2 + natoms]
    box = lines[2 + natoms]

    drop_resnames = set(SOLVENT_RESNAMES) | set(ION_RESNAMES_CHARMM) | {
        "NA", "CL", "K", "POT", "MG", "CA2", "ZN", "ZN2", "WAT"}

    kept = []
    n_dropped_solv = 0
    n_dropped_ion = 0
    ion_set = set(ION_RESNAMES_CHARMM) | {"NA", "CL", "K", "POT", "MG",
                                            "CA2", "ZN", "ZN2"}
    for ln in atoms:
        if len(ln) < 10:
            kept.append(ln)
            continue
        rn = ln[5:10].strip()
        if rn in drop_resnames:
            if rn in ion_set:
                n_dropped_ion += 1
            else:
                n_dropped_solv += 1
            continue
        kept.append(ln)

    # Renumber atom indices sequentially (cols 15-20) to keep .gro consistent.
    renumbered = []
    for i, ln in enumerate(kept, start=1):
        if len(ln) < 20:
            renumbered.append(ln)
            continue
        renumbered.append(f"{ln[:15]}{i:5d}{ln[20:]}")

    out_gro = Path(out_gro)
    out_gro.parent.mkdir(parents=True, exist_ok=True)
    with open(out_gro, "w") as fh:
        fh.write(title if title.endswith("\n") else title + "\n")
        fh.write(f"{len(renumbered):>5d}\n")
        fh.writelines(renumbered)
        fh.write(box if box.endswith("\n") else box + "\n")

    return {"n_kept": len(renumbered),
            "n_dropped_solvent": n_dropped_solv,
            "n_dropped_ion": n_dropped_ion}


def _merge_cavity_and_fresh_ev(cavity_gro: Path, fresh_ev_gro: Path,
                                 out_gro: Path,
                                 approach_gap_nm: float,
                                 box_z_extend_nm: float) -> dict:
    """Concatenate cavity + fresh EV into one .gro, extend Z, return diagnostics.

    The cavity sits at its original Z; the fresh EV is translated so its BOTTOM
    is `approach_gap_nm` above the cavity's TOP. The resulting box is Z-extended
    by `box_z_extend_nm` to make room for solvation.
    """
    _, cav_atoms, cav_box = _read_gro(cavity_gro)
    _, fresh_atoms, fresh_box = _read_gro(fresh_ev_gro)

    if not cav_atoms:
        raise ValueError(f"cavity {cavity_gro} has zero atoms")
    if not fresh_atoms:
        raise ValueError(f"fresh EV {fresh_ev_gro} has zero atoms")

    # z-extent of each system
    cav_z = np.fromiter((_atom_z_nm(a) for a in cav_atoms), dtype=float)
    fresh_z = np.fromiter((_atom_z_nm(a) for a in fresh_atoms), dtype=float)
    cav_top = float(cav_z.max())
    fresh_bot = float(fresh_z.min())

    # Translate fresh EV up so its bottom sits `approach_gap_nm` above cavity top.
    dz_fresh = (cav_top + approach_gap_nm) - fresh_bot
    logger.info("EV-approach placement: cavity_top=%.2f nm, fresh_EV_bottom=%.2f nm, "
                "shift=+%.2f nm, gap=%.2f nm",
                cav_top, fresh_bot, dz_fresh, approach_gap_nm)

    # Rebuild atom lines with sequential atom-index and shifted fresh coordinates.
    merged = []
    idx = 1
    for a in cav_atoms:
        merged.append(_shift_atom(a, renumber=idx))
        idx += 1
    for a in fresh_atoms:
        merged.append(_shift_atom(a, dz=dz_fresh, renumber=idx))
        idx += 1

    # Extend Z box vector for solvation slab.
    box_parts = cav_box.split()
    if len(box_parts) < 3:
        raise ValueError(f"cavity box line unparseable: {cav_box!r}")
    cav_x, cav_y, box_z = float(box_parts[0]), float(box_parts[1]), float(box_parts[2])
    # BUG #17 fix: the fresh EV's XY box (after semi-iso NPT shrink) may exceed
    # the Triton-lysed cavity's XY, so merging into the cavity box would wrap
    # fresh-EV atoms across ±X / ±Y PBC and clash. Take the max of both XY box
    # vectors plus a 0.5 nm buffer so the merged box safely contains both.
    fresh_box_parts = fresh_box.split()
    if len(fresh_box_parts) >= 2:
        try:
            fresh_x = float(fresh_box_parts[0])
            fresh_y = float(fresh_box_parts[1])
        except ValueError:
            fresh_x = fresh_y = 0.0
    else:
        fresh_x = fresh_y = 0.0
    box_x = max(cav_x, fresh_x) + 0.5
    box_y = max(cav_y, fresh_y) + 0.5
    new_z = box_z + box_z_extend_nm
    extra = " " + " ".join(box_parts[3:]) if len(box_parts) > 3 else ""
    new_box_line = f"{box_x:10.5f}{box_y:11.5f}{new_z:11.5f}{extra}\n"

    out_gro = Path(out_gro)
    out_gro.parent.mkdir(parents=True, exist_ok=True)
    with open(out_gro, "w") as fh:
        fh.write("cavity + fresh EV (approach system)\n")
        fh.write(f"{len(merged):>5d}\n")
        fh.writelines(merged)
        fh.write(new_box_line)

    return {
        "n_cavity_atoms":  len(cav_atoms),
        "n_fresh_atoms":   len(fresh_atoms),
        "n_total_atoms":   len(merged),
        "cavity_top_nm":   cav_top,
        "fresh_bottom_original_nm": fresh_bot,
        "fresh_shift_nm":  dz_fresh,
        "box_z_original_nm": box_z,
        "box_z_extended_nm": new_z,
    }


# ── topology merge: cavity.top + fresh EV toppar/*.itp ────────

def _count_resname_blocks(gro_atoms: list) -> dict:
    """Return {resname: count} in .gro atom-order-preserving blocks.

    A 'block' is a run of consecutive residues sharing the same resname.
    The count is the number of RESIDUES in each block (not atoms). Used to
    reconstruct the [ molecules ] section.
    """
    blocks = []   # list of (resname, n_residues)
    prev_resname = None
    prev_resid = None
    for ln in gro_atoms:
        if len(ln) < 10:
            continue
        resname = ln[5:10].strip()
        try:
            resid = int(ln[0:5].strip())
        except ValueError:
            continue
        if resname != prev_resname:
            blocks.append([resname, 1])
        elif resid != prev_resid:
            blocks[-1][1] += 1
        prev_resname = resname
        prev_resid = resid
    return blocks


def _extract_topol_molecules_block(top_text: str) -> list:
    """Parse the [ molecules ] section into [(name, count), ...]."""
    lines = top_text.splitlines()
    out = []
    in_mol = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("[") and "molecules" in s:
            in_mol = True; continue
        if s.startswith("[") and in_mol:
            break
        if in_mol and s and not s.startswith(";"):
            parts = s.split()
            if len(parts) >= 2:
                try:
                    out.append((parts[0], int(parts[1])))
                except ValueError:
                    pass
    return out


def _build_merged_topology(cavity_top: Path, fresh_ev_top: Path,
                             out_top: Path,
                             fresh_toppar_dir: Path,
                             out_toppar_dir: Path,
                             cavity_atoms: list,
                             fresh_atoms: list) -> dict:
    """Merge topology headers + [ molecules ] blocks.

    Strategy: use the fresh EV's topology as the base (it has the full
    CHARMM-GUI toppar chain including PROA + lipid itps + ions + water), then
    APPEND cavity's monomer itps + rewrite the [ molecules ] block to reflect
    the actual atom ordering in the merged .gro (cavity atoms first, then
    fresh EV atoms).
    """
    fresh_text = Path(fresh_ev_top).read_text()
    cavity_text = Path(cavity_top).read_text()

    # ── copy fresh EV toppar to output dir ──
    if not out_toppar_dir.exists() and fresh_toppar_dir.is_dir():
        shutil.copytree(fresh_toppar_dir, out_toppar_dir)

    # ── extract cavity monomer itps (anything under `#include "toppar/…"`
    #    not already present in the fresh EV toppar) ──
    # The fresh EV loads CHARMM36 as its base force field via its own
    # forcefield.itp / ions.itp / SPC water. The cavity was built with a
    # different FF (typically amber99sb-ildn from pdb2gmx, with GAFF2 monomer
    # itps). We MUST NOT append another FF definition on top of CHARMM36 —
    # GROMACS would silently load two [ defaults ] / atomtypes blocks and
    # produce nonsense forces. Drop every FF/water/ions include from the
    # cavity side; keep only per-monomer itps.
    #
    # NOTE (open): monomer itps parameterised under GAFF2 reference atom types
    # ('c3', 'hc', 'oh', ...) that are not defined by CHARMM36. For a truly
    # unified system the monomers should be re-parameterised with CGenFF via
    # CHARMM-GUI Ligand Reader. Until then the merged topology is only sound
    # if the monomer itps embed a self-contained [ atomtypes ] block.
    import re as _re
    _FF_INCLUDE_PATTERNS = (
        _re.compile(r'\.ff/forcefield\.itp"'),
        _re.compile(r'\.ff/ions\.itp"'),
        _re.compile(r'\.ff/spc\w*\.itp"'),
        _re.compile(r'\.ff/tip\w*\.itp"'),
    )
    fresh_includes = {
        ln.strip() for ln in fresh_text.splitlines()
        if ln.lstrip().startswith("#include")
    }
    extra_includes = []
    dropped_ff_includes = []
    cavity_toppar_dir = Path(cavity_top).parent / "toppar"
    for ln in cavity_text.splitlines():
        if not ln.lstrip().startswith("#include"):
            continue
        if ln.strip() in fresh_includes:
            continue
        if any(pat.search(ln) for pat in _FF_INCLUDE_PATTERNS):
            dropped_ff_includes.append(ln.strip())
            continue
        # Non-standard include (usually a monomer itp) — carry it over.
        extra_includes.append(ln.rstrip())
        # Copy the referenced file if it exists next to cavity.top
        m = _re.search(r'"([^"]+)"', ln)
        if m and cavity_toppar_dir.is_dir():
            src_itp = cavity_toppar_dir / Path(m.group(1)).name
            dst_itp = out_toppar_dir / Path(m.group(1)).name
            if src_itp.is_file() and not dst_itp.exists():
                shutil.copy2(src_itp, dst_itp)
    if dropped_ff_includes:
        logger.warning(
            "Dropped %d force-field include(s) from cavity topology to avoid "
            "double-loading on top of the fresh-EV CHARMM36 base: %s. "
            "Monomer itps must carry their own [ atomtypes ] block, or the "
            "monomers must be re-parameterised with CGenFF for the merger to "
            "be physically consistent.",
            len(dropped_ff_includes), dropped_ff_includes)

    # ── build the merged molecule list from the ATOM ORDER ──
    #   cavity atoms first (their [ molecules ] entries), then fresh EV atoms
    cav_mol_blocks = _count_resname_blocks(cavity_atoms)
    fresh_mol_blocks = _count_resname_blocks(fresh_atoms)
    # Convert 'block of residues with resname X, count N' → topology entry.
    # For polymers (protein, lipid), N residues typically belong to ONE
    # moleculetype ('PROA' has N residues but 1 molecule). We can't infer that
    # from resname alone; fall back to the ORIGINAL [ molecules ] blocks of
    # cavity + fresh EV concatenated in that order.
    cav_topol_mols = _extract_topol_molecules_block(cavity_text)
    fresh_topol_mols = _extract_topol_molecules_block(fresh_text)
    merged_mols = cav_topol_mols + fresh_topol_mols

    # ── strip fresh EV's [ molecules ] block, append ours ──
    lines = fresh_text.splitlines(keepends=True)
    out_lines = []
    in_mol = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("[") and "molecules" in s:
            in_mol = True
            continue
        if in_mol:
            # skip old molecules content until next section or EOF
            if s.startswith("[") and "molecules" not in s:
                in_mol = False
                out_lines.append(ln)
            continue
        out_lines.append(ln)

    # Append extra includes (cavity monomer itps) BEFORE the new [ molecules ]
    # section, right after the fresh EV's include block. We insert them just
    # after the last existing #include line.
    last_include_i = max(
        (i for i, l in enumerate(out_lines)
         if l.lstrip().startswith("#include")),
        default=-1)
    if last_include_i >= 0 and extra_includes:
        insert = "\n".join(extra_includes) + "\n"
        out_lines.insert(last_include_i + 1, insert)

    # Append new [ molecules ] block
    out_lines.append("\n[ molecules ]\n; Compound\t#mols\n")
    for name, count in merged_mols:
        out_lines.append(f"{name:8s}\t{count:10d}\n")

    out_top = Path(out_top)
    out_top.write_text("".join(out_lines))
    return {
        "cavity_mol_entries":  cav_topol_mols,
        "fresh_mol_entries":   fresh_topol_mols,
        "merged_mol_entries":  merged_mols,
        "extra_includes":      extra_includes,
    }


# ── entry point ──────────────────────────────────────────────

def build_ev_approach_system(cavity_gro: Path, cavity_top: Path,
                               target: str, seed: int,
                               md_dir: Path,
                               approach_gap_nm: float | None = None,
                               box_z_extend_nm: float | None = None,
                               solvate: bool = True,
                               neutralise: bool = True,
                               gmx_bin: str | None = None,
                               fresh_ev_gro: Path | None = None) -> dict:
    """Assemble a Phase 5 EV-approach system: cavity + fresh EV + solvent.

    Parameters
    ----------
    cavity_gro, cavity_top : Path
        Triton-lysed cavity from utils_triton_removal.finalize_triton_removal.
    target : str
        Target name (CD63/CD81/CD9) — resolved to the fresh EV CHARMM-GUI
        outputs under structures/membrane/<target>/.
    seed : int
        Deterministic seed for fresh EV rotation (independent placement per
        Phase 5 replica).
    md_dir : Path
        Where to write the assembled system files.
    approach_gap_nm, box_z_extend_nm : float, optional
        Override the config defaults PHASE5_FRESH_EV_APPROACH_GAP_NM and
        PHASE5_BOX_Z_EXTEND_NM.
    solvate, neutralise : bool
        Whether to run `gmx solvate` and `gmx genion` after coordinate merge.
        Skip only for dry-run testing.
    gmx_bin : str, optional
        Path to the GROMACS binary. Defaults to config.GMX_BIN.
    fresh_ev_gro : Path, optional
        BLOCKER C4 fix. When supplied, this pre-built .gro is used AS-IS for
        the fresh EV, BYPASSING the internal build_fresh_ev() call. Intended
        for callers that produced an independent ensemble via
        utils_vesicle.prepare_independent_fresh_evs — that path gives a real
        set of independent structural realisations rather than N rotations of
        the same CHARMM-GUI equilibrated system. When None (the default), the
        legacy behaviour is preserved: build_fresh_ev(target, seed=seed) is
        called to produce a rotated copy of the CHARMM-GUI EV.

    Returns
    -------
    dict with keys 'success', 'md_dir', 'system_gro', 'ionized_gro',
    'system_top', 'placement_diag', 'topology_diag', plus per-step diagnostics.
    """
    from . import config as cfg
    md_dir = Path(md_dir)
    md_dir.mkdir(parents=True, exist_ok=True)
    gap = float(approach_gap_nm if approach_gap_nm is not None
                else getattr(cfg, "PHASE5_FRESH_EV_APPROACH_GAP_NM", 4.0))
    z_ext = float(box_z_extend_nm if box_z_extend_nm is not None
                   else getattr(cfg, "PHASE5_BOX_Z_EXTEND_NM", 15.0))
    gmx = gmx_bin or getattr(cfg, "GMX_BIN", "gmx")

    # ── 1. Fresh EV coordinates ──
    # C4 fix: prefer a pre-built independent-ensemble replica when supplied;
    # otherwise fall back to the legacy rotation-only build for backwards
    # compatibility.
    if fresh_ev_gro is not None:
        fresh_ev_gro = Path(fresh_ev_gro)
        if not fresh_ev_gro.exists():
            raise FileNotFoundError(
                f"build_ev_approach_system: fresh_ev_gro was supplied but "
                f"does not exist on disk: {fresh_ev_gro}")
        # BUG #15 fix: prepare_independent_fresh_evs returns a solvated
        # NVT-relaxed .gro; if we merge it verbatim the fresh-EV waters wrap
        # through +Z PBC and touch the Triton-lysed cavity. Strip solvent
        # + ions into a local copy before the merge — the composite system
        # is re-solvated after the merge by `gmx solvate`.
        local_fresh = md_dir / f"fresh_ev_{target}_replica_seed{seed}.gro"
        strip_diag = _strip_solvent_from_gro(fresh_ev_gro, local_fresh)
        logger.info("EV-approach using pre-built independent fresh EV: %s "
                    "(bypassing rotation-only build_fresh_ev). "
                    "Stripped %d solvent + %d ion atoms before merge; kept %d.",
                    fresh_ev_gro,
                    strip_diag["n_dropped_solvent"],
                    strip_diag["n_dropped_ion"],
                    strip_diag["n_kept"])
        fresh_ev_gro = local_fresh
    else:
        fresh_ev_gro = md_dir / f"fresh_ev_{target}_seed{seed}.gro"
        build_fresh_ev(target, seed=seed, out_path=fresh_ev_gro,
                        drop_solvent=True)

    # ── 2. Merge cavity + fresh EV, extend box ──
    system_gro = md_dir / "ev_approach_system.gro"
    placement = _merge_cavity_and_fresh_ev(
        Path(cavity_gro), fresh_ev_gro, system_gro,
        approach_gap_nm=gap, box_z_extend_nm=z_ext)
    logger.info("  merged system: %d atoms, box_z %.2f → %.2f nm",
                placement["n_total_atoms"],
                placement["box_z_original_nm"], placement["box_z_extended_nm"])

    # ── 3. Merge topology ──
    from .utils_vesicle import load_template_ev
    fresh_info = load_template_ev(target, md_dir / f"_fresh_ev_{target}_toppar")
    system_top = md_dir / "topol.top"
    topol_diag = _build_merged_topology(
        cavity_top=Path(cavity_top),
        fresh_ev_top=Path(fresh_info["top"]),
        out_top=system_top,
        fresh_toppar_dir=Path(fresh_info["toppar_dir"]),
        out_toppar_dir=md_dir / "toppar",
        cavity_atoms=_read_gro(Path(cavity_gro))[1],
        fresh_atoms=_read_gro(fresh_ev_gro)[1],
    )

    # ── 4. gmx solvate — fill the added slab with water ──
    if solvate:
        solvated_gro = md_dir / "solvated.gro"
        try:
            subprocess.run(
                [gmx, "solvate", "-cp", str(system_gro), "-cs", "spc216.gro",
                 "-p", str(system_top), "-o", str(solvated_gro)],
                check=True, capture_output=True, cwd=str(md_dir), timeout=600)
            logger.info("  gmx solvate: added slab filled with water")
            current_gro = solvated_gro
        except subprocess.CalledProcessError as e:
            logger.error("  gmx solvate FAILED: %s", e.stderr.decode()[-500:])
            return {"success": False, "error": "solvate_failed",
                    "placement_diag": placement, "topology_diag": topol_diag}
    else:
        current_gro = system_gro

    # ── 5. gmx genion — neutralise ──
    ionized_gro = md_dir / "ionized.gro"
    if neutralise and solvate:
        genion_mdp = md_dir / "_genion.mdp"
        genion_mdp.write_text(
            "integrator = steep\nnsteps = 0\ncutoff-scheme = Verlet\n")
        try:
            subprocess.run(
                [gmx, "grompp", "-f", str(genion_mdp), "-c", str(current_gro),
                 "-p", str(system_top), "-o", str(md_dir / "_genion.tpr"),
                 "-maxwarn", "20"],
                check=True, capture_output=True, cwd=str(md_dir), timeout=120)
            subprocess.run(
                [gmx, "genion", "-s", str(md_dir / "_genion.tpr"),
                 "-o", str(ionized_gro), "-p", str(system_top),
                 "-pname", "SOD", "-nname", "CLA", "-neutral"],
                input=b"SOL\n", check=True, capture_output=True,
                cwd=str(md_dir), timeout=120)
            logger.info("  gmx genion: system neutralised (SOD/CLA)")
        except subprocess.CalledProcessError as e:
            logger.warning("  gmx genion failed (system may already be neutral): %s",
                            e.stderr.decode()[-300:])
            shutil.copy2(current_gro, ionized_gro)
    else:
        shutil.copy2(current_gro, ionized_gro)

    # ── 6. Write TC1/TC2 membrane index (BUG #8 fix): the Phase 5 NVT / NPT /
    # production MDPs must reference two-bath tc-grps (protein+glycan+monomer
    # vs bilayer+water+ion), otherwise the default single 'Protein Non-Protein'
    # bath couples lipids and water together and imbalances temperatures.
    # Derive the monomer resnames from the cavity's atom lines (anything not
    # protein / lipid / glycan / solvent / ion).
    try:
        from .utils_gromacs import _write_membrane_index
        cav_atoms_for_idx = _read_gro(Path(cavity_gro))[1]
        _known = (set(STANDARD_AMINO_ACIDS)
                  | set(MEMBRANE_LIPID_RESNAMES)
                  | {n[:5] for n in GLYCAN_RESNAMES_CHARMM}
                  | set(SOLVENT_RESNAMES)
                  | set(ION_RESNAMES_CHARMM)
                  | {"NA", "CL", "K", "POT", "MG", "CA2", "ZN", "ZN2", "WAT"})
        monomer_resnames = set()
        for _ln in cav_atoms_for_idx:
            if len(_ln) < 10:
                continue
            _rn = _ln[5:10].strip()
            if _rn and _rn not in _known:
                monomer_resnames.add(_rn)
        index_ndx = md_dir / "index.ndx"
        idx_counts = _write_membrane_index(
            ionized_gro, index_ndx,
            lipid_resnames=MEMBRANE_LIPID_RESNAMES,
            monomer_resnames=monomer_resnames)
        logger.info("  TC groups: TC1 = %d atoms, TC2 = %d atoms "
                    "(monomers detected: %s)",
                    idx_counts.get("TC1", 0), idx_counts.get("TC2", 0),
                    sorted(monomer_resnames))
        if idx_counts.get("TC1", 0) == 0 or idx_counts.get("TC2", 0) == 0:
            logger.warning("EV-approach index has an empty TC group: %s",
                            idx_counts)
        index_path_str = str(index_ndx)
    except Exception as _e:
        logger.warning("Could not write membrane index for EV-approach: %s", _e)
        index_path_str = None
        idx_counts = {}

    return {
        "success":         True,
        "md_dir":          str(md_dir),
        "system_gro":      str(system_gro),
        "ionized_gro":     str(ionized_gro),
        "system_top":      str(system_top),
        "fresh_ev_gro":    str(fresh_ev_gro),
        "index_ndx":       index_path_str,
        "index_counts":    idx_counts,
        "placement_diag":  placement,
        "topology_diag":   topol_diag,
    }


def run_ev_approach_leg(cavity_gro: Path, cavity_top: Path,
                          target: str, seed: int,
                          output_dir: Path,
                          time_ns: int = 20,
                          is_own_target: bool = True,
                          gmx_bin: str | None = None,
                          fresh_ev_gro: Path | None = None) -> dict:
    """Full EV-approach rebinding: build → EM → NVT → NPT → production → analyse.

    Mirrors the return shape of `_run_rebinding_md` in phase5_rebinding.py so
    the dispatcher there can call it in-place. When PHASE5 EV-approach mode is
    off, the caller should NOT invoke this — it will still run, but the
    Triton-lysed cavity is a prerequisite.

    fresh_ev_gro : Path, optional
        BLOCKER C4 fix — see build_ev_approach_system. When supplied, the
        rotation-only fresh-EV construction is bypassed and this pre-built
        replica coordinate file is used instead. Intended for callers that
        prepared an independent ensemble via
        utils_vesicle.prepare_independent_fresh_evs.
    """
    from . import config as cfg
    from .utils_gromacs import (run_energy_minimization,
                                 run_nvt_equilibration,
                                 run_npt_equilibration,
                                 run_production_md)
    from .utils_persistent_contacts import compute_persistent_contacts_fast

    output_dir = Path(output_dir)
    md_dir = output_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build ──
    try:
        build = build_ev_approach_system(
            cavity_gro=cavity_gro, cavity_top=cavity_top,
            target=target, seed=seed, md_dir=md_dir, gmx_bin=gmx_bin,
            fresh_ev_gro=fresh_ev_gro)
    except Exception as e:
        logger.error("EV-approach build failed for %s: %s", target, e)
        return {"success": False, "error": f"build_failed: {e}",
                "protocol": "ev_approach", "is_own_target": bool(is_own_target)}
    if not build["success"]:
        return {"success": False, "error": build.get("error", "build_failed"),
                "protocol": "ev_approach", "is_own_target": bool(is_own_target),
                "build_diag": build}

    ionized = Path(build["ionized_gro"])

    # ── 2. EM ──
    try:
        run_energy_minimization(md_dir)
    except Exception as e:
        logger.error("EM failed in EV-approach %s seed=%s: %s", target, seed, e)
        return {"success": False, "error": f"em_failed: {e}",
                "protocol": "ev_approach", "build_diag": build}

    # ── 3. NVT / NPT / Production ──
    # BUG #13 fix: run_nvt/npt_equilibration expect `time_ps=`, not `time_ns=`.
    # BUG #4  fix: bilayer systems require semiisotropic P-coupling with the
    #              two-value compressibility / ref_p, else LINCS explodes.
    # BUG #7  fix: keep the membrane restrained during NVT/NPT so it does not
    #              drift during monomer / fresh-EV insertion (define =
    #              -DPOSRES_MEMBRANE, deactivated for production).
    # BUG #8  fix: pass tc_grps='TC1 TC2' together with the .ndx that defines
    #              those groups; the default 'Protein Non-Protein' couples
    #              lipids + water in one bath and imbalances T.
    index_ndx = build.get("index_ndx")
    index_path = Path(index_ndx) if index_ndx else None
    try:
        run_nvt_equilibration(md_dir, time_ps=100.0,
                                temperature=cfg.MD_TEMPERATURE_K,
                                define='-DPOSRES_MEMBRANE',
                                tc_grps='TC1 TC2',
                                index=index_path)
        run_npt_equilibration(md_dir, time_ps=100.0,
                                pressure=cfg.MD_PRESSURE_BAR,
                                temperature=cfg.MD_TEMPERATURE_K,
                                define='-DPOSRES_MEMBRANE',
                                tc_grps='TC1 TC2',
                                pcoupltype='semiisotropic',
                                compressibility='4.5e-5 4.5e-5',
                                ref_p='1.0 1.0',
                                index=index_path)
        run_production_md(md_dir, time_ns=time_ns,
                            temperature=cfg.MD_TEMPERATURE_K,
                            pressure=cfg.MD_PRESSURE_BAR,
                            gpu_id=cfg.MD_GPU_ID,
                            tc_grps='TC1 TC2',
                            pcoupltype='semiisotropic',
                            compressibility='4.5e-5 4.5e-5',
                            ref_p='1.0 1.0',
                            index=index_path)
    except Exception as e:
        logger.error("MD failed in EV-approach %s seed=%s: %s", target, seed, e)
        return {"success": False, "error": f"md_failed: {e}",
                "protocol": "ev_approach", "build_diag": build}

    # ── 4. Analyse — persistent contacts on the fresh EV's protein only.
    # (Triton removal already dropped the template CD; the ONLY protein in
    # the system is the fresh EV's, so `protein_sel='protein'` is safe.)
    md_tpr = md_dir / "md.tpr"
    md_xtc = md_dir / "md.xtc"
    contacts_result = None
    if md_tpr.exists() and md_xtc.exists():
        try:
            # BUG #16 fix: glycosylated residues (Man3GlcNAc2 sugars) are
            # always in contact with the protein they are covalently bound to,
            # so leaving them in the binder selection makes them score
            # freq=1.0 as "persistent" contacts. Exclude the full CHARMM
            # glycan resname list from the binder side.
            _glycan_excl = " ".join(GLYCAN_RESNAMES_CHARMM)
            binder_sel = ("not protein and not resname SOL HOH NA CL TIP3 "
                          "WAT SOD CLA POPC POPE PSM POPS CHL1 "
                          f"{_glycan_excl}")
            freq, n_persistent, meta = compute_persistent_contacts_fast(
                traj_path=md_xtc, top_path=md_tpr,
                return_meta=True,
                protein_sel="protein",
                binder_sel=binder_sel)
            contacts_result = {
                "available":            True,
                "n_persistent_residues": int(n_persistent),
                "total_protein_residues": len(freq),
                "fraction_persistent":  (n_persistent / len(freq)
                                          if freq else 0.0),
                "mean_contact_frequency": float(np.mean(list(freq.values())))
                                          if freq else 0.0,
                "meta":                 meta,
            }
        except Exception as e:
            logger.warning("PCSI analysis failed for EV-approach leg: %s", e)
            contacts_result = {"available": False, "error": str(e)}

    return {
        "success":          True,
        "protocol":         "ev_approach",
        "is_own_target":    bool(is_own_target),
        "target":           target,
        "placement_seed":   int(seed),
        "time_ns":          time_ns,
        "build_diag":       build,
        "contacts":         contacts_result,
        "n_persistent_residues": (contacts_result or {}).get(
            "n_persistent_residues", 0),
        "fraction_persistent":   (contacts_result or {}).get(
            "fraction_persistent", 0.0),
        # C4: record where the fresh-EV coordinates came from so the
        # replica-level breakdown in verify_phase5 can flag rotation-only
        # (legacy, statistically N=1) versus a real independent ensemble.
        "fresh_ev_source":  (str(fresh_ev_gro) if fresh_ev_gro is not None
                              else "rotation_only(build_fresh_ev)"),
    }
