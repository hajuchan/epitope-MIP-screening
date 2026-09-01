"""Triton X-100 lysis simulation — remove template EV (lipids + CD) from a
Phase 4 final system, leaving the monomer network as the imprinted cavity.

Bridges Phase 4 (CD-in-EV + monomers polymerised) and Phase 5 (fresh EV
approaches empty cavity). Modelling choice per user protocol: opt 1 —
explicit removal via atom deletion, not explicit detergent MD.

Public entry point:

    finalize_triton_removal(phase4_md_dir, cavity_out_dir) -> dict

Removes every atom whose residue name is in the lipid or standard
amino-acid set (drops both the phospholipid bilayer and the template
tetraspanin) and rewrites the .gro + .top + a short-relaxation em/nvt to
let the vacated cavity relax before Phase 5 rebinding.

Returns {'cavity_gro','cavity_top','em_gro','n_atoms_removed'}.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from .utils_vesicle import (MEMBRANE_LIPID_RESNAMES, SOLVENT_RESNAMES,
                              ION_RESNAMES_CHARMM, STANDARD_AMINO_ACIDS,
                              GLYCAN_RESNAMES_CHARMM)

# .gro resname column is 5 chars wide, so match glycans by 5-char prefix
# (e.g. "BGLCNA" in [ molecules ] appears as "BGLCN" in the .gro column).
_GLYCAN_GRO_KEYS = frozenset(name[:5] for name in GLYCAN_RESNAMES_CHARMM)

logger = logging.getLogger(__name__)


def _classify_gro_lines(gro_path: Path):
    """Return {'title','box','keep','removed'} — keep is monomer+solvent+ion,
    removed is lipid+protein (=EV template)."""
    with open(gro_path) as fh:
        lines = fh.readlines()
    title = lines[0]
    natoms = int(lines[1].strip())
    atoms = lines[2:2 + natoms]
    box = lines[2 + natoms]

    keep, removed = [], []
    for ln in atoms:
        if len(ln) < 10:
            keep.append(ln); continue
        resname = ln[5:10].strip()
        # Glycans are covalently bonded to PROA in the CHARMM-GUI setup —
        # dropping the protein without also dropping its glycan atoms leaves
        # them in the .gro while the [ molecules ] block loses PROA, causing
        # grompp to abort on a coordinate/topology atom-count mismatch.
        if (resname in MEMBRANE_LIPID_RESNAMES
                or resname in STANDARD_AMINO_ACIDS
                or resname in _GLYCAN_GRO_KEYS):
            removed.append(ln)
        else:
            # Everything else — monomers (functional + crosslinker), waters,
            # ions — stays. The monomers form the cavity; water/ions maintain
            # aqueous phase for the short relaxation.
            keep.append(ln)
    return {"title": title, "box": box, "keep": keep, "removed": removed}


def _write_gro(out_path: Path, title: str, atoms: list, box: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(f"{title.rstrip()} (triton-lysed cavity)\n")
        fh.write(f"{len(atoms):>5d}\n")
        # Renumber atoms 1..N (residue numbers left as-is; the .top's
        # [ molecules ] section is what actually gates topology).
        for i, ln in enumerate(atoms, start=1):
            prefix = ln[:15]        # residue + resname + atomname (fixed)
            coord = ln[20:]         # x/y/z (+velocities if present)
            fh.write(f"{prefix}{i:5d}{coord}")
        fh.write(box)


def _count_moleculetype_atoms(itp_path: Path) -> dict:
    """Parse an .itp/.top and return {moleculetype_name: n_atoms_in_it}.

    Reads sequentially, tracking the last `[ moleculetype ]` name and counting
    non-comment/non-blank rows inside the following `[ atoms ]` section until
    the next `[ … ]` directive.
    """
    counts: dict[str, int] = {}
    try:
        with open(itp_path) as fh:
            lines = fh.readlines()
    except OSError:
        return counts
    current_name: str | None = None
    section: str | None = None
    expect_moltype = False
    for raw in lines:
        s = raw.split(";", 1)[0].strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            section = s.strip("[] ").strip().lower()
            if section == "moleculetype":
                expect_moltype = True
            continue
        if expect_moltype and section == "moleculetype":
            tok = s.split()
            if tok:
                current_name = tok[0]
                counts.setdefault(current_name, 0)
                expect_moltype = False
            continue
        if section == "atoms" and current_name is not None:
            # Each atom row starts with an integer atom-index.
            tok = s.split()
            if tok and tok[0].isdigit():
                counts[current_name] += 1
    return counts


def _count_topol_atoms(top_path: Path, src_dir: Path) -> int:
    """Sum moleculetype_size * count over the [ molecules ] block of `top_path`,
    resolving moleculetype sizes from the .top itself and every `#include`-d
    file (searched relative to `top_path.parent` and the original `src_dir`)."""
    with open(top_path) as fh:
        lines = fh.readlines()

    search_roots = [top_path.parent]
    if src_dir is not None:
        search_roots.append(Path(src_dir))

    # Build moleculetype -> n_atoms map from the top itself + any #includes.
    mol_sizes: dict[str, int] = dict(_count_moleculetype_atoms(top_path))
    for ln in lines:
        s = ln.strip()
        if not s.startswith("#include"):
            continue
        # e.g. #include "toppar/POPC.itp"
        parts = s.split('"')
        if len(parts) < 2:
            continue
        rel = parts[1]
        for root in search_roots:
            cand = root / rel
            if cand.is_file():
                for k, v in _count_moleculetype_atoms(cand).items():
                    mol_sizes.setdefault(k, v)
                break

    # Sum the [ molecules ] section.
    total = 0
    in_molecules = False
    for ln in lines:
        s = ln.split(";", 1)[0].strip()
        if s.startswith("[") and s.endswith("]"):
            in_molecules = (s.strip("[] ").strip().lower() == "molecules")
            continue
        if not in_molecules or not s:
            continue
        tok = s.split()
        if len(tok) < 2 or not tok[-1].isdigit():
            continue
        name, count = tok[0], int(tok[-1])
        size = mol_sizes.get(name, 0)
        total += size * count
    return total


def _rewrite_topology(top_in: Path, top_out: Path,
                       removed_resnames: set):
    """Drop `#include` of lipid + protein itp lines and their [ molecules ]
    entries. Everything else (monomers, water, ions, forcefield) stays.
    """
    # PROA and any lipid resname listed → drop
    drop_includes = {f'"toppar/{r}.itp"' for r in removed_resnames}
    drop_includes.add('"toppar/PROA.itp"')
    drop_mol_names = removed_resnames | {"PROA"}

    with open(top_in) as fh:
        lines = fh.readlines()
    out_lines = []
    in_molecules = False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("[") and "molecules" in stripped:
            in_molecules = True
            out_lines.append(ln); continue
        if stripped.startswith("[") and in_molecules:
            in_molecules = False
        if ln.lstrip().startswith("#include"):
            if any(d in ln for d in drop_includes):
                continue
        if in_molecules and stripped and not stripped.startswith(";"):
            tokens = stripped.split()
            if tokens and tokens[0] in drop_mol_names:
                continue
        out_lines.append(ln)
    top_out.write_text("".join(out_lines))


def finalize_triton_removal(phase4_md_dir: Path = None,
                              cavity_out_dir: Path = None,
                              relax_em: bool = True,
                              gmx_bin: str | None = None,
                              input_gro: Path | None = None,
                              input_top: Path | None = None) -> dict:
    """Delete lipids + template protein from the Phase 4 final system.

    Two call modes:
      (a) directory-based: pass `phase4_md_dir` — reads md.gro + topol.top
          from that dir (original interface).
      (b) explicit-path: pass `input_gro` + `input_top` — Phase 5's snapshot
          layout uses frame.gro instead of md.gro, so callers there use this
          mode. Both `phase4_md_dir` and (input_gro, input_top) resolve to
          the same downstream logic.

    Parameters
    ----------
    phase4_md_dir : Path, optional
        Directory containing md.gro + topol.top + toppar/. Not modified.
    cavity_out_dir : Path
        Where cavity.gro / cavity.top / (optional em.gro) are written.
    relax_em : bool
        Run a short (steepest-descents) EM on the stripped system to
        relieve any lipid-vacated stress on the monomer network.
    gmx_bin : str, optional
        Path to `gmx`. Defaults to config.GMX_BIN.
    input_gro, input_top : Path, optional
        Explicit coordinate + topology paths. Take precedence over
        phase4_md_dir when provided.

    Returns
    -------
    dict : {'cavity_gro','cavity_top','em_gro','n_atoms_removed', ...}
    """
    if input_gro is not None and input_top is not None:
        src_gro = Path(input_gro)
        src_top = Path(input_top)
        phase4_md_dir = src_gro.parent
    elif phase4_md_dir is not None:
        phase4_md_dir = Path(phase4_md_dir)
        src_gro = phase4_md_dir / "md.gro"
        src_top = phase4_md_dir / "topol.top"
    else:
        raise ValueError("finalize_triton_removal requires either "
                          "phase4_md_dir or (input_gro + input_top)")
    if cavity_out_dir is None:
        raise ValueError("finalize_triton_removal requires cavity_out_dir")
    cavity_out_dir = Path(cavity_out_dir)
    cavity_out_dir.mkdir(parents=True, exist_ok=True)

    if not src_gro.is_file():
        raise FileNotFoundError(f"input .gro missing at {src_gro}")
    if not src_top.is_file():
        raise FileNotFoundError(f"input .top missing at {src_top}")

    logger.info("Triton removal: stripping lipids + template CD from %s",
                phase4_md_dir)
    parsed = _classify_gro_lines(src_gro)
    n_removed = len(parsed["removed"])
    n_keep = len(parsed["keep"])
    logger.info("  → removed %d atoms (lipid+protein), kept %d "
                "(monomer+water+ion)", n_removed, n_keep)

    cavity_gro = cavity_out_dir / "cavity.gro"
    _write_gro(cavity_gro, parsed["title"], parsed["keep"], parsed["box"])

    # Rewrite topology — drop lipid AND glycan moleculetype entries in
    # addition to PROA (glycans go with the protein; see _classify_gro_lines).
    cavity_top = cavity_out_dir / "cavity.top"
    _rewrite_topology(src_top, cavity_top,
                       removed_resnames=(set(MEMBRANE_LIPID_RESNAMES)
                                          | set(GLYCAN_RESNAMES_CHARMM)))

    # Post-write consistency check: .gro atom count must equal the sum of
    # moleculetype * count in topol.top's [ molecules ] block, or grompp
    # will fail with a coordinate/topology atom-count mismatch.
    with open(cavity_gro) as _fh:
        _gro_head = [next(_fh) for _ in range(2)]
    n_atoms_in_gro = int(_gro_head[1].strip())
    n_atoms_in_topol = _count_topol_atoms(cavity_top, phase4_md_dir)
    assert n_atoms_in_gro == n_atoms_in_topol, (
        f"MISMATCH: gro={n_atoms_in_gro} vs topol={n_atoms_in_topol}")
    # Copy toppar so #include paths resolve
    src_toppar = phase4_md_dir / "toppar"
    if src_toppar.is_dir():
        dst_toppar = cavity_out_dir / "toppar"
        if not dst_toppar.exists():
            shutil.copytree(src_toppar, dst_toppar)

    result = {
        "cavity_gro": str(cavity_gro),
        "cavity_top": str(cavity_top),
        "n_atoms_removed": n_removed,
        "n_atoms_kept":    n_keep,
    }

    if relax_em:
        em_gro = _run_relaxation(cavity_out_dir, cavity_gro, cavity_top, gmx_bin)
        result["em_gro"] = str(em_gro) if em_gro else None
    return result


def _run_relaxation(work_dir: Path, cavity_gro: Path, cavity_top: Path,
                     gmx_bin: str | None):
    """Short EM (steepest descents) + 100 ps NVT to relax the vacated cavity.

    Uses the CHARMM-GUI-style em/nvt mdp inputs already generated by
    utils_gromacs when the pipeline sets up an aqueous system. If those
    templates are not available in this working dir, writes minimal defaults.
    """
    from . import config as cfg
    gmx = gmx_bin or getattr(cfg, "GMX_BIN", "gmx")

    em_mdp = work_dir / "cavity_em.mdp"
    if not em_mdp.exists():
        em_mdp.write_text(_MIN_EM_MDP)

    em_tpr = work_dir / "cavity_em.tpr"
    em_gro = work_dir / "cavity_em.gro"
    try:
        subprocess.run(
            [gmx, "grompp", "-f", str(em_mdp), "-c", str(cavity_gro),
             "-p", str(cavity_top), "-o", str(em_tpr), "-maxwarn", "10"],
            check=True, capture_output=True, cwd=str(work_dir), timeout=300)
        subprocess.run(
            [gmx, "mdrun", "-deffnm", str(em_tpr.with_suffix("")),
             "-c", str(em_gro), "-nt", "4"],
            check=True, capture_output=True, cwd=str(work_dir), timeout=600)
    except subprocess.CalledProcessError as e:
        # Do not swallow gmx failures — the caller must decide whether to
        # abort Phase 5. Attach stderr for post-mortem.
        stderr_tail = (e.stderr or b"").decode("utf-8", "replace")[-2000:]
        raise RuntimeError(
            f"Triton relaxation EM failed (rc={e.returncode}): "
            f"{' '.join(map(str, e.cmd))}\n{stderr_tail}"
        ) from e
    logger.info("  → cavity_em.gro written (relaxed cavity)")
    return em_gro


_MIN_EM_MDP = """; Minimal energy minimisation for Triton-lysed cavity.
integrator      = steep
nsteps          = 5000
emtol           = 1000
emstep          = 0.01
nstlist         = 10
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.2
rvdw            = 1.2
constraints     = h-bonds
"""
