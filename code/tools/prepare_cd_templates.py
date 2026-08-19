#!/usr/bin/env python3
"""Build membrane-framed, repaired receptor templates for CD63 / CD81 / CD9.

WHAT THIS IS FOR
----------------
The lab no longer imprints a synthetic 16-mer.  It imprints intact
nanovesicles carrying each tetraspanin in a lipid bilayer, so the only surface
that matters is the one a polymer can reach from the extracellular side.  That
makes three things load-bearing that the pipeline's current templates get
wrong:

  * the LEL must be in the OPEN (upright) conformer, or a quarter of the
    scored surface is really facing lipid;
  * the extracellular loop must actually be present — 6K4J is missing five
    CD9 residues that were DELETED FROM THE CONSTRUCT, and they turn out to
    sit at the apex of the domain;
  * engineered mutations must be reverted, or CD63 is scored as an aglycosyl
    triple mutant it never is on a vesicle.

This script performs those repairs and writes one prepared template per target
plus one closed-conformer control for CD81.  It NEVER writes into results/ and
NEVER re-runs any pipeline phase.

PROVENANCE IS IN THE FILE, NOT JUST THE LOG
-------------------------------------------
Every prepared template carries a REMARK 999 block naming its source entries,
every splice, every reverted mutation and every rebuilt atom, and the
OCCUPANCY COLUMN encodes provenance per atom:

    1.00  experimental, primary entry
    0.75  experimental, spliced donor entry
    0.50  side chain rebuilt (truncated in the deposition, or a reverted mutation)
    0.25  backbone built de novo (loop closure) — PREDICTED
    0.00  AlphaFold scaffold — PREDICTED

so `awk '$1=="ATOM" && substr($0,55,6)+0 < 0.75'` lists every invented atom.

USAGE
-----
    conda activate MIPscreen
    python3 code/tools/prepare_cd_templates.py --all
    python3 code/tools/prepare_cd_templates.py --target CD63 --no-minimize
    python3 code/tools/prepare_cd_templates.py --selfcheck   # verify only

Author: prepared-template session 2026-08-12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cd_pdbio import (  # noqa: E402
    AA1TO3, AA3TO1, BACKBONE, YCM_KEEP, Model, Residue,
    OCC_DENOVO_BB, OCC_DONOR, OCC_EXPERIMENTAL, OCC_PREDICTED, OCC_REBUILT_SC,
    frame_from_tm, kabsch, read_pdb, write_pdb,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "structures" / "raw"
PREPARED = ROOT / "structures" / "prepared"
LOGS = ROOT / "structures" / "logs"

# How many times to re-attempt the restrained minimisation before giving up.
# The step is NOT reproducible: pdbfixer/Modeller place invented atoms afresh on
# every call, and an unlucky placement puts two atoms close enough for the GBn2
# Born-radius term to diverge (E ~ -2e7 kJ/mol, Fmax ~ 1e13) and NaN out. Same
# model, measured: 1 failure then 6/6 successes, so retrying is the fix.
MINIMISE_ATTEMPTS = 8

# ── target facts (every one traceable to a header record or a UniProt field) ──
UNIPROT = {"CD63": "P08962", "CD81": "P60033", "CD9": "P21926"}

TOPOLOGY = {
    # TM spans, EC1, EC2 — UniProt topological domains, confirmed live 2026-08-12
    "CD63": dict(tm=[(12, 32), (52, 72), (82, 102), (204, 224)], ec1=(33, 51), ec2=(103, 203)),
    "CD81": dict(tm=[(13, 33), (64, 84), (90, 112), (202, 224)], ec1=(34, 63), ec2=(113, 201)),
    "CD9":  dict(tm=[(13, 33), (56, 76), (90, 111), (196, 220)], ec1=(34, 55), ec2=(112, 195)),
}

# Disulfide PAIRING, with the citation for each pair.  CD63 is the reason this
# table exists: UniProt P08962 carries NO DISULFID annotation at all, so the
# pairing can only be cited to the SSBOND records of the two 2025 depositions.
DISULFIDE_PAIRS = {
    "CD63": [(145, 191), (146, 169), (170, 177)],
    "CD81": [(156, 190), (157, 175)],
    "CD9":  [(152, 181), (153, 167)],
}
DISULFIDE_CITATION = {
    "CD63": "SSBOND records of PDB 9HUR (1.65 A, P41212) and 9HQ5 (2.60 A, P43212), "
            "two independent crystal forms; UniProt P08962 has NO DISULFID feature.",
    "CD81": "UniProt P60033 DISULFID 156..190 and 157..175, plus SSBOND in all 16 "
            "human CD81 entries.",
    "CD9":  "SSBOND records of PDB 6K4J, 6Z1V, 6Z20, 6RLO, 6RLR (all five agree).",
}

N_GLYCAN_SEQUONS = {"CD63": [130, 150, 172], "CD81": [], "CD9": [52, 53]}

# CD63 HAS NO OPM ENTRY, so its hydrophobic half-thickness cannot be fitted.
# Do NOT estimate it from the TM helices' own z-extent: measured this session,
# that estimator returns 14.7 A for 5TCX where OPM fits 17.1 A, and the
# under-estimate loses 14 of the 25 membrane-occluded EC2 residues.  What the
# TM-SVD construction DOES get right is the orientation and the origin - against
# 5TCX its bilayer mid-plane sits 0.07 +/- 0.12 A from OPM's, and fed the true
# 17.1 A it reproduces the OPM occluded set exactly plus one residue (26 vs 25).
# So the normal is derived and the thickness is asserted: 15.5 A, the mean of the
# two OPM-fitted OPEN tetraspanin membranes (7JIC 15.3, 6K4J 15.7).  The
# sensitivity across 15.3-17.1 A is reported with every template.
# Reproduce: python3 code/tools/prepare_cd_templates.py --validate-frame
CD63_HALF_THICKNESS = 15.5
HALF_THICKNESS_BRACKET = (15.3, 15.7, 17.1)

# ── small helpers ───────────────────────────────────────────────────────────
_LOG_LINES = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr)
    _LOG_LINES.append(line)


def uniprot_seq(acc: str) -> str:
    p = RAW / "uniprot" / f"{acc}.fasta"
    if not p.is_file():
        raise SystemExit(f"missing {p} — re-fetch it before running")
    return "".join(l.strip() for l in open(p) if not l.startswith(">"))


def assert_wt(model: Model, acc: str, nums, where: str) -> None:
    """Hard stop if a residue we are about to trust is not the wild-type one.

    This is the guard that catches expression tags whose numbering collides
    with real residues — 6Z1V numbers its GS tag 112/113 while CD9 P21926 has
    SER112/HIS113 there, so a naive "take 112-191 from 6Z1V" silently
    substitutes two wrong residues into the scored surface.
    """
    seq = uniprot_seq(acc)
    bad = []
    for i in nums:
        r = model.res.get(i)
        if r is None:
            continue
        if not (1 <= i <= len(seq)) or r.one != seq[i - 1]:
            bad.append(f"{i}:{r.resname}({r.one})!=UNP{seq[i-1] if 1<=i<=len(seq) else '?'}")
    if bad:
        raise SystemExit(f"WILD-TYPE CHECK FAILED at {where}: {', '.join(bad)}")


def strip_sidechain(res: Residue, keep=BACKBONE + ("CB",)) -> None:
    for name in [n for n in res.atoms if n not in keep]:
        del res.atoms[name]


def demodify(res: Residue) -> None:
    """Reduce a modified residue to its parent amino acid (YCM -> CYS)."""
    if res.resname == "YCM":
        for name in [n for n in res.atoms if n not in YCM_KEEP]:
            del res.atoms[name]
        res.resname = "CYS"
        for a in res.atoms.values():
            a.resname = "CYS"
    elif res.resname == "MSE":
        res.resname = "MET"
        for a in res.atoms.values():
            a.resname = "MET"
            if a.name == "SE":
                a.name, a.element = "SD", "S"
        res.atoms = OrderedDict((a.name, a) for a in res.atoms.values())


def mark(prov: dict, nums, **kw) -> None:
    for i in nums:
        prov.setdefault(i, {}).update(kw)


# ── pdbfixer stage ──────────────────────────────────────────────────────────
def pdbfixer_pass(model: Model, mutations, insert, chain_id="A"):
    """Apply mutations, build the listed missing residues, complete side chains.

    `mutations` : list of "SER-130-ASN" strings, in the model's own numbering.
    `insert`    : {after_resnum: [resname, ...]} internal gaps to close.

    Returns (new_model, added_atoms, added_residues, tool_versions).
    Residue numbering is restored positionally afterwards and asserted against
    the expected residue list, because pdbfixer renumbers inserted residues.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    import pdbfixer as _pf
    import openmm as _mm

    # Expected residue list AFTER insertion, in order.
    expected = []
    for i in model.nums():
        expected.append((i, model.res[i].resname))
        if i in insert:
            for k, name in enumerate(insert[i], start=1):
                expected.append((i + k, name))
    # apply mutations to the expectation
    mut_map = {}
    for m in mutations:
        old, num, new = m.split("-")
        mut_map[int(num)] = (old, new)
    expected = [(n, mut_map[n][1] if n in mut_map else rn) for n, rn in expected]

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.pdb"
        write_pdb(model, src, chain=chain_id)
        fixer = PDBFixer(filename=str(src))

        if mutations:
            fixer.applyMutations(list(mutations), chain_id)

        fixer.missingResidues = {}
        if insert:
            chain = list(fixer.topology.chains())[0]
            residues = list(chain.residues())
            idx_of = {int(r.id): k for k, r in enumerate(residues)}
            for after, names in insert.items():
                if after not in idx_of:
                    raise SystemExit(f"cannot insert after residue {after}: not present")
                fixer.missingResidues[(0, idx_of[after] + 1)] = list(names)

        fixer.findMissingAtoms()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.addMissingAtoms()

        out = Path(td) / "out.pdb"
        with open(out, "w") as fh:
            PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=False)
        fixed = read_pdb(out, chain_id, keep_hetatm=())

    got = [(i, fixed.res[i].resname) for i in fixed.nums()]
    if len(got) != len(expected) or [r for _, r in got] != [r for _, r in expected]:
        raise SystemExit(
            "pdbfixer returned an unexpected residue list.\n"
            f"  expected {len(expected)}: {''.join(AA3TO1.get(r,'X') for _, r in expected)}\n"
            f"  got      {len(got)}: {''.join(AA3TO1.get(r,'X') for _, r in got)}")

    # restore the true numbering
    renum = Model(source="pdbfixer", chain=chain_id)
    for (want, _name), (have, _) in zip(expected, got):
        r = fixed.res[have].copy()
        r.resnum = want
        for a in r.atoms.values():
            a.resnum = want
        renum.res[want] = r
    renum.res = OrderedDict(sorted(renum.res.items()))

    # OpenMM drops the B-factor column, so restore it from the input.  It is not
    # decoration: for the AlphaFold part it IS the per-residue pLDDT, and for the
    # experimental part it is the deposited B, and both were load-bearing in the
    # structure survey.  Atoms this script invented get B = 0 and are identified
    # by the occupancy channel instead.
    for i in renum.nums():
        src = model.res.get(i)
        for name, a in renum.res[i].atoms.items():
            a.b = src.atoms[name].b if (src and name in src.atoms) else 0.0

    before = {(i, n) for i in model.nums() for n in model.res[i].atoms}
    inserted_nums = {n for after, names in insert.items()
                     for n in range(after + 1, after + 1 + len(names))}
    added_atoms, added_res = [], sorted(inserted_nums)
    for i in renum.nums():
        for n in renum.res[i].atoms:
            if n == "OXT":
                continue
            if (i, n) not in before:
                added_atoms.append((i, n))

    # drop pdbfixer's terminal OXT so the template stays a clean fragment
    for i in renum.nums():
        renum.res[i].atoms.pop("OXT", None)

    versions = {"pdbfixer": getattr(_pf, "__version__", "unknown"),
                "openmm": _mm.version.version}
    return renum, added_atoms, added_res, versions


def minimize(model: Model, restrain_keys, chain_id="A", free_nums=()):
    """Restrained vacuum/implicit minimisation: only invented atoms may move.

    Every atom in `restrain_keys` (a set of (resnum, atom_name)) is held by a
    10 kJ/mol/A^2 harmonic restraint, so only atoms this script invented are
    free to move.  The AlphaFold scaffold is restrained too even though it is
    predicted: it defines the membrane frame, which was measured before the
    minimisation, and letting the TM bundle re-fold here would silently
    invalidate every z-height in the report.

    `free_nums` releases the restraint on the residues immediately flanking a
    splice junction.  Without it a stretched junction peptide bond fights the
    restraints instead of closing, and the minimiser converges to a strained
    structure with the break still in it.
    """
    import openmm as mm
    from openmm import app, unit

    from pdbfixer import PDBFixer

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "min_in.pdb"
        # Fragments become separate chains so each gets a proper terminal group;
        # residue numbers stay globally unique, so mapping back is unambiguous.
        write_pdb(model, src, chain=chain_id, split_at_gaps=True)
        capped = Path(td) / "min_capped.pdb"
        fx = PDBFixer(filename=str(src))
        fx.missingResidues = {}
        fx.findMissingAtoms()
        fx.addMissingAtoms()          # adds OXT to each fragment C-terminus
        with open(capped, "w") as fh:
            app.PDBFile.writeFile(fx.topology, fx.positions, fh, keepIds=True)
        pdb = app.PDBFile(str(capped))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        ff = None
        for combo in (("amber14-all.xml", "implicit/gbn2.xml"), ("amber14-all.xml",)):
            try:
                ff = app.ForceField(*combo)
                modeller2 = app.Modeller(pdb.topology, pdb.positions)
                modeller2.addHydrogens(ff, pH=7.4)
                modeller = modeller2
                break
            except Exception as e:              # pragma: no cover
                last = e
                ff = None
        if ff is None:
            raise RuntimeError(f"forcefield/addHydrogens failed: {last}")

        system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                                 constraints=None, rigidWater=False)

        # positional restraints, matched by (resnum, atom name)
        free = set(free_nums)
        want = {}
        for i in model.nums():
            if i in free:
                continue
            for name, a in model.res[i].atoms.items():
                if (i, name) in restrain_keys:
                    want[(i, name)] = a.xyz
        force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", 1000.0 * unit.kilojoule_per_mole /
                                 unit.nanometer ** 2)
        for p in ("x0", "y0", "z0"):
            force.addPerParticleParameter(p)
        n_restrained = 0
        for atom in modeller.topology.atoms():
            key = (int(atom.residue.id), atom.name)
            if key in want:
                xyz = want[key] / 10.0        # A -> nm
                force.addParticle(atom.index, [float(xyz[0]), float(xyz[1]), float(xyz[2])])
                n_restrained += 1
        system.addForce(force)

        integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                           0.002 * unit.picoseconds)
        sim = app.Simulation(modeller.topology, system, integrator,
                             mm.Platform.getPlatformByName("CPU"))
        sim.context.setPositions(modeller.positions)
        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
        sim.minimizeEnergy(maxIterations=2000)
        st = sim.context.getState(getPositions=True, getEnergy=True)
        e1 = st.getPotentialEnergy()
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

        out = Model(source="minimised", chain=chain_id)
        for i in model.nums():
            out.res[i] = model.res[i].copy()
        moved = 0
        for atom in modeller.topology.atoms():
            if atom.element is not None and atom.element.symbol == "H":
                continue
            i, name = int(atom.residue.id), atom.name
            r = out.res.get(i)
            if r and name in r.atoms:
                old = r.atoms[name].xyz
                r.atoms[name].set_xyz(pos[atom.index])
                if np.linalg.norm(old - r.atoms[name].xyz) > 0.1:
                    moved += 1
    return out, {"e_before_kJ": float(e0.value_in_unit(unit.kilojoule_per_mole)),
                 "e_after_kJ": float(e1.value_in_unit(unit.kilojoule_per_mole)),
                 "n_restrained": n_restrained, "n_atoms_moved_gt_0.1A": moved}


# ── membrane frame ──────────────────────────────────────────────────────────
def apply_opm_frame(model: Model, target: str):
    """OPM files are already in the membrane frame; verify the sign and record."""
    if model.dum_z is None:
        raise SystemExit(f"{model.source} carries no DUM atoms — not an OPM file")
    half = float(max(abs(model.dum_z[0]), abs(model.dum_z[1])))
    ec2 = TOPOLOGY[target]["ec2"]
    zs = [model.res[i].atoms["CA"].z for i in model.nums()
          if ec2[0] <= i <= ec2[1] and "CA" in model.res[i].atoms]
    if np.mean(zs) < 0:
        raise SystemExit(f"{model.source}: EC2 centroid is at negative z — "
                         f"extracellular side is not +z, refusing to proceed")
    return half, f"OPM ({model.source}), DUM planes at z = +/-{half:.1f} A"


def apply_derived_frame(model: Model, target: str):
    """No OPM entry exists for CD63; derive the frame from the TM bundle."""
    tm = TOPOLOGY[target]["tm"]
    R, t, extent_half = frame_from_tm(model, tm, signs=[+1, -1, +1, -1])
    model.transform(R, t)
    ec2 = TOPOLOGY[target]["ec2"]
    zs = [model.res[i].atoms["CA"].z for i in model.nums()
          if ec2[0] <= i <= ec2[1] and "CA" in model.res[i].atoms]
    if np.mean(zs) < 0:                       # flip if the sign convention inverted
        R2 = np.diag([1.0, -1.0, -1.0])
        model.transform(R2, np.zeros(3))
    half = CD63_HALF_THICKNESS
    return half, (
        f"NORMAL AND ORIGIN DERIVED from the TM bundle (SVD of the TM1-4 CA axes, "
        f"signs +,-,+,-); THICKNESS ASSERTED at {half:.1f} A. No OPM entry exists for "
        f"{target}, so nothing here is fitted to a bilayer. Validation of the "
        f"construction on 5TCX, which does have an OPM entry: the derived mid-plane "
        f"sits 0.07 +/- 0.12 A from OPM's and, given OPM's own 17.1 A, the derived "
        f"frame returns 26 occluded EC2 residues against OPM's 25 (the extra one is "
        f"147). The TM z-extent estimator was REJECTED: it returns {extent_half:.1f} A "
        f"here and 14.7 A on 5TCX where the fitted value is 17.1 A, which would have "
        f"discarded 14 of 25 occluded residues. 15.5 A is the mean of the two "
        f"OPM-fitted OPEN tetraspanin membranes (7JIC 15.3, 6K4J 15.7). This is the "
        f"least trustworthy of the three membrane frames, and its dominant "
        f"uncertainty is the LEL-TM hinge, not the thickness.")


def occluded(model: Model, lo: int, hi: int, half: float, margin: float = 6.0):
    """Residues whose CA sits in or below the lipid headgroup band."""
    return [i for i in model.nums()
            if lo <= i <= hi and "CA" in model.res[i].atoms
            and model.res[i].atoms["CA"].z < half + margin]


# ══════════════════════════════════════════════════════════════════════════
# BUILDERS
# ══════════════════════════════════════════════════════════════════════════
def build_cd63():
    """CD63_open_v1 = 9HUR LEL + 9HQ5 104-107, deglyco-reverted, on AF v6 TM."""
    acc = UNIPROT["CD63"]
    prov, steps = {}, []

    af = read_pdb(RAW / "alphafold" / "AF-P08962-F1-model_v6.pdb", "A")
    assert af.span() == (1, 238) and not af.gaps(), "AF CD63 model is not 1-238 ungapped"
    half, frame_desc = apply_derived_frame(af, "CD63")
    mark(prov, af.nums(), src="AF-P08962-F1-model_v6 (AlphaFold DB v6, 2025-08-01)",
         occ=OCC_PREDICTED)

    hur = read_pdb(RAW / "pdb" / "9HUR.pdb", "A")
    assert hur.span() == (108, 209), f"9HUR:A span changed: {hur.span()}"
    hur.drop(range(204, 210))          # ENLYFQ TEV tag (SEQADV EXPRESSION TAG)
    steps.append("9HUR chain B (sybody LA4, 154 aa) and all HETATM (SO4, NA, HOH) "
                 "dropped; C-terminal ENLYFQ tag 204-209 dropped (SEQADV EXPRESSION TAG). "
                 "16 CD63 side chains lay inside the LA4 4.5 A epitope (E114, F115, N118, "
                 "F119, Q122, N129, H131, I135, L136, R138, M139, D142, F143, I195, W198, "
                 "L199) and are now presented as free-surface rotamers although they were "
                 "selected under a bound binder; treat a hit driven only by them with "
                 "suspicion.")

    # place the experimental LEL into the AlphaFold membrane frame
    common = [i for i in range(108, 204) if i in hur.res and i in af.res]
    R, t, rms_lel = kabsch(hur.coords(common), af.coords(common))
    hur.transform(R, t)

    # PLANNED-BUT-REJECTED: splicing 104-107 from 9HQ5.  Measured this session:
    # after fitting 9HQ5 onto 9HUR on CA 108-125 (RMSD 1.73 A) and moving into the
    # AF frame, 9HQ5's stalk terminus lands 9.3-19.5 A from AF's own 104-107 and
    # leaves a 17.8 A break at the 103/104 junction.  That terminus is flexible,
    # has no TM anchor, and is engaged by the SB3 sybody (epitope covers S103, G104,
    # Y105, V106) with Y105's whole ring truncated per REMARK 470.  Grafting it
    # would have produced a physically disconnected molecule, so AF supplies
    # 104-108 instead and 9HQ5 is used only as independent confirmation of the fold
    # (CEalign 0.90 A vs 9HUR) and of the disulfide pairing.
    exp_lo, exp_hi = 109, 198
    lel = hur.subset(exp_lo, exp_hi)
    lel.set_occ(OCC_EXPERIMENTAL)
    mark(prov, range(exp_lo, exp_hi + 1), src="9HUR chain A (X-ray 1.65 A)",
         occ=OCC_EXPERIMENTAL)
    steps.append(
        f"Experimental LEL block is {exp_lo}-{exp_hi} (90 of the 101 EC2 residues), "
        f"placed into the AlphaFold membrane frame by CA superposition on the whole "
        f"common loop 108-203 (n={len(common)}, RMSD {rms_lel:.2f} A, which is the "
        f"honest measure of how far AlphaFold's CD63 LEL is from the crystal). The "
        f"boundaries were chosen by measuring peptide-bond closure, not by taste: "
        f"across every candidate start 108-120 the AF(i)C-9HUR(i+1)N distance is "
        f"1.05 A at 109 and 1.9-4.9 A everywhere else, and at the C-terminal end the "
        f"9HUR(i)C-AF(i+1)N distance is 1.45 A at 198 but 4.7 A at 199 and 9.5 A at "
        f"203. The last five EC2 residues 199-203 therefore come from AlphaFold: the "
        f"isolated-loop crystal has no TM helix to constrain its stalk exit, and those "
        f"five residues are membrane-occluded and excluded from the scored surface "
        f"anyway.")
    steps.append("The LEL-to-TM hinge angle is AlphaFold's, NOT experimental. Neither "
                 "CD63 entry contains a TM helix, a lipid, a detergent or an OPM entry, "
                 "so no experiment constrains it. Transplanting the 9HUR LEL into the "
                 "5TCX and 6K4J membrane frames brackets the membrane-occluded fraction "
                 "of the CD63 LEL at 10-28%; report CD63 accessibility as that range, "
                 "not as a point.")

    merged = Model(source="CD63_open_v1", chain="A")
    merged.merge(af)
    merged.merge(lel)                      # overwrite 109-198 with experimental
    for r in merged.res.values():
        demodify(r)
    junctions = [(exp_lo - 1, exp_lo), (exp_hi, exp_hi + 1)]

    # revert the triple glycan knockout
    muts = []
    for pos in (130, 150, 172):
        r = merged.res[pos]
        assert r.resname == "SER", f"9HUR residue {pos} is {r.resname}, expected SER"
        strip_sidechain(r)                 # keep backbone + CB, rebuild as ASN
        muts.append(f"SER-{pos}-ASN")
    steps.append("Glycan knockout reverted: S130N, S150N, S172N (SEQADV ENGINEERED "
                 "MUTATION in both 9HUR and 9HQ5). No wild-type-sequence CD63 structure "
                 "exists in the PDB at any resolution, so these three Asn side chains "
                 "are REBUILT ROTAMERS, not measurements — and they are the anchors for "
                 "the glycan-shadow mask.")

    truncated = sorted(set(hur.truncated.get("A", {})) & set(range(exp_lo, exp_hi + 1)))
    fixed, added_atoms, added_res, versions = pdbfixer_pass(merged, muts, {})
    steps.append(f"Truncated side chains rebuilt by pdbfixer {versions['pdbfixer']} "
                 f"(REMARK 470 of 9HUR inside the kept block: {truncated}); six of them "
                 f"are surface Lys/Glu, i.e. exactly the chemistry a charged-monomer "
                 f"screen scores on. A rebuilt Lys is a rotamer guess with ~4 A of "
                 f"freedom at NZ; any charge-driven hit resting on one of these should "
                 f"be re-tested over a rotamer ensemble.")

    return dict(name="CD63_open_v1", target="CD63", model=fixed, prov=prov,
                steps=steps, added_atoms=added_atoms, added_res=added_res,
                versions=versions, half=half, frame=frame_desc, junctions=junctions,
                primary="9HUR chain A (X-ray 1.65 A, released 2025-11-19) -> EC2 109-198",
                donors=["AF-P08962-F1 v6 (AlphaFold DB, 2025-08-01) -> 1-108, 199-238: "
                        "TM bundle, EC1, both stalk exits, and the membrane frame",
                        "9HQ5 chain A -> NOT used for coordinates (see repair 3); "
                        "used only to corroborate the fold and the disulfide pairing"],
                conformer="OPEN/UPRIGHT (AlphaFold hinge - PREDICTED, not measured)")


def build_cd81_open():
    """CD81_open_v1 = 7JIC chain B, CD19/Fab stripped, gaps repaired, WT reverted."""
    acc = UNIPROT["CD81"]
    prov, steps = {}, []

    jic = read_pdb(RAW / "opm" / "7jic_opm.pdb", "B")
    assert jic.span() == (9, 226), f"7JIC:B span changed: {jic.span()}"
    half, frame_desc = apply_opm_frame(jic, "CD81")
    steps.append("Chains A (CD19, ectodomain + its own TM helix) and H/L (coltuximab "
                 "Fab) deleted. The Fab is a pure fiducial: at a 5.0 A all-atom cutoff "
                 "it contacts zero CD81 residues. CD19 does contact CD81 (22-34, 77-80 "
                 "intramembrane; 157-189 on EC2) and stripping it is correct for a "
                 "CD19-free vesicle — but the open conformer and the EC2 head trace are "
                 "CD19-STABILISED, so the resting state on a vesicle is a hypothesis, "
                 "not a measurement. Use CD81_closed_5tcx_v1 as the paired control.")
    steps.append(f"Membrane frame taken from OPM directly: {frame_desc}. 7JIC carries no "
                 f"HETATM of any kind (no lipid, no cholesterol, no water).")

    jic.set_occ(OCC_EXPERIMENTAL)
    mark(prov, jic.nums(), src="7JIC chain B (cryo-EM 3.80 A)", occ=OCC_EXPERIMENTAL)

    # EC2 gap 1: D137-A140 spliced from 5M33 chain A (1.28 A)
    m33 = read_pdb(RAW / "pdb" / "5M33.pdb", "A")
    flank = [i for i in list(range(130, 137)) + list(range(141, 148))]
    R, t, rms = kabsch(m33.coords(flank), jic.coords(flank))
    m33.transform(R, t)
    donor = m33.subset(137, 140)
    assert_wt(donor, acc, range(137, 141), "5M33 donor 137-140")
    donor.set_occ(OCC_DONOR)
    jic.merge(donor)
    mark(prov, range(137, 141), src="5M33 chain A (X-ray 1.28 A), spliced", occ=OCC_DONOR)
    steps.append(f"EC2 gap D137-A140 (REMARK 465; disorder, not deletion — present in "
                 f"SEQRES) SPLICED from 5M33 chain A after CA superposition on the flanks "
                 f"130-136+141-147 (n={len(flank)}, RMSD {rms:.2f} A). The same acidic "
                 f"stretch is disordered in 1G8Q:B, 5M33:B, 6EJM and 5M4R, so a single "
                 f"conformer understates its real flexibility.")

    # EC2 gaps 2 and 3 built de novo
    insert = {115: ["LYS"], 166: ["THR", "SER", "VAL", "LEU"]}
    steps.append("K116 built de novo (CA115-CA117 = "
                 f"{np.linalg.norm(jic.res[115].ca()-jic.res[117].ca()):.1f} A against a "
                 "~7.6 A maximum for one residue, so the closure is essentially "
                 "determined).")
    steps.append("T167-S168-V169-L170 built DE NOVO — NOT spliced. 5M33 cannot donate "
                 "here: fitting on the flanks 160-166+171-177 gives 4.4 A, confirming the "
                 "7JIC head is genuinely rearranged in this region, so a splice would "
                 "import the wrong local fold. These four residues are PREDICTED, they "
                 "sit next to the reverted K171, and they are inside the former CD19 "
                 "interface. Exclude 167-172 from any primary scored surface.")

    muts = ["THR-171-LYS", "SER-172-ASN"]
    for pos, exp in ((171, "THR"), (172, "SER")):
        assert jic.res[pos].resname == exp, f"7JIC {pos} is {jic.res[pos].resname}"
        strip_sidechain(jic.res[pos])
    steps.append("Construct substitutions reverted: T171K and N172S -> wild type "
                 "(SEQADV flags both as CONFLICT, not ENGINEERED MUTATION, so their "
                 "provenance is ambiguous — but SEQRES chain B confirms they are real). "
                 "K171 is a surface lysine: leaving it as Thr would have removed a "
                 "positive charge from the scored surface and biased every acidic-monomer "
                 "result for CD81.")

    fixed, added_atoms, added_res, versions = pdbfixer_pass(jic, muts, insert)
    steps.append(f"Truncated side chains rebuilt by pdbfixer {versions['pdbfixer']}; in "
                 f"EC2 only four were truncated (F113, K121, N173, K193 per REMARK 470), "
                 f"i.e. 85/89 intact despite the nominal 3.8 A. EC1 34-63 is COMPLETE in "
                 f"this entry — the only CD81 structure where that is true — but carries "
                 f"six side-chain stubs (R36, K52, N56, V60, I62, Y63).")

    return dict(name="CD81_open_v1", target="CD81", model=fixed, prov=prov,
                steps=steps, added_atoms=added_atoms, added_res=added_res,
                versions=versions, half=half, frame=frame_desc,
                junctions=[(136, 137), (140, 141), (115, 116), (116, 117),
                           (166, 167), (170, 171)],
                primary="7JIC chain B (cryo-EM 3.80 A, CD19-CD81-Fab complex)",
                donors=["5M33 chain A (X-ray 1.28 A) -> 137-140"],
                conformer="OPEN/UPRIGHT (experimental, but CD19-stabilised)")


def build_cd81_closed():
    """CD81_closed_5tcx_v1 — the mandatory closed-conformer control arm."""
    prov, steps = {}, []
    tcx = read_pdb(RAW / "opm" / "5tcx_opm.pdb", "A")
    assert tcx.span() == (6, 232), f"5TCX:A span changed: {tcx.span()}"
    half, frame_desc = apply_opm_frame(tcx, "CD81")
    tcx.set_occ(OCC_EXPERIMENTAL)
    mark(prov, tcx.nums(), src="5TCX chain A (X-ray 2.955 A)", occ=OCC_EXPERIMENTAL)

    for r in tcx.res.values():
        demodify(r)
    steps.append("Cholesterol CLR A301 and waters deleted (CLR sits entirely inside the "
                 "hydrophobic core, z -3.1..+12.8, and contacts TM residues only — it "
                 "occludes no extracellular surface). YCM at Cys80/Cys89 "
                 "(iodoacetamide-alkylated Cys, both intramembrane/cytoplasmic) reduced "
                 "to CYS.")

    muts = []
    for pos in (6, 9, 227, 228):
        r = tcx.res.get(pos)
        if r is not None and r.resname == "SER":
            strip_sidechain(r)
            muts.append(f"SER-{pos}-CYS")
    steps.append(f"Engineered palmitoylation-site mutations reverted: "
                 f"{', '.join(muts) or 'none present in the modelled range'} "
                 f"(SEQADV ENGINEERED MUTATION C6S/C9S/C227S/C228S). All four are "
                 f"cytoplasmic and irrelevant to the extracellular surface.")
    steps.append("EC1 IS LEFT BROKEN ON PURPOSE. REMARK 465 removes D38-A54 "
                 "(DPQTTNLLYLELGDKPA), 17 of the 30 EC1 residues, and that block is "
                 "exactly the solvent-facing apex. Building a 17-residue de novo loop "
                 "would be fabrication; this arm exists to test the EC2 conformer, and "
                 "EC1 is unscored for every target anyway.")

    fixed, added_atoms, added_res, versions = pdbfixer_pass(tcx, muts, {})
    return dict(name="CD81_closed_5tcx_v1", target="CD81", model=fixed, prov=prov,
                steps=steps, added_atoms=added_atoms, added_res=added_res,
                versions=versions, half=half, frame=frame_desc, junctions=[],
                primary="5TCX chain A (X-ray 2.955 A, cholesterol-bound)",
                donors=[],
                conformer="CLOSED/FLAT — control arm, NOT for the cross-target contrast")


def build_cd9(variant="a"):
    """CD9_open_v1 (6Z1V apex) / CD9_open_v1b (6RLO apex) on the 6K4J membrane frame."""
    acc = UNIPROT["CD9"]
    prov, steps = {}, []

    k4j = read_pdb(RAW / "opm" / "6k4j_opm.pdb", "A")
    assert k4j.span() == (1, 225) and k4j.gaps() == [(174, 180)], \
        f"6K4J:A changed: span={k4j.span()} gaps={k4j.gaps()}"
    half, frame_desc = apply_opm_frame(k4j, "CD9")
    k4j.set_occ(OCC_EXPERIMENTAL)
    mark(prov, k4j.nums(), src="6K4J chain A (X-ray 2.70 A, lipidic cubic phase)",
         occ=OCC_EXPERIMENTAL)
    for r in k4j.res.values():
        demodify(r)
    steps.append("HETATM removed: PLM A301 (palmitate — modelled 1.27 A from CYS219 SG "
                 "with NO LINK record, which is not a credible covalent bond and is a "
                 "modelling assertion rather than a measurement), OLC A302 (monoolein, "
                 "the LCP host lipid, sitting at the EC1/headgroup boundary against "
                 "TYR61 and ARG36), NI A303 (chelated by GLU174 and GLU187 — a "
                 "crystallisation artefact clamping the residue immediately before the "
                 "construct deletion). The 5-residue N-terminal GSREF tag and residues "
                 "226-228 are absent from the deposition already.")
    steps.append("N-terminal expression-tag residues (SEQADV EXPRESSION TAG, numbered "
                 "-4..0) are not present in the ATOM records and so never enter the "
                 "template.")

    if variant == "a":
        donor_file, donor_chain, donor_id = RAW / "pdb" / "6Z1V.pdb", "A", \
            "6Z1V chain A (X-ray 1.33 A, nanobody 4E8 complex)"
        apex_note = ("apex conformer from 6Z1V (nanobody 4E8). Epitope at 4.0 A covers "
                     "129,133,155,157-161,164,169,173-178, i.e. the nanobody sits "
                     "directly on the segment being imported.")
    else:
        donor_file, donor_chain, donor_id = RAW / "pdb" / "6RLO.pdb", "J", \
            "6RLO chain J (X-ray 2.20 A, AT1412dm Fab complex)"
        apex_note = ("apex conformer from 6RLO (AT1412dm Fab), a genuinely DIFFERENT "
                     "D-loop: after fitting on 114-150 it deviates from 6Z1V by up to "
                     "10.5 A at residue 172. This is the second ensemble member; a "
                     "CD9-selective monomer hit must survive both.")

    dz = read_pdb(donor_file, donor_chain)
    scaffold = [i for i in list(range(114, 152)) + list(range(181, 191))
                if i in dz.res and i in k4j.res]
    R, t, rms = kabsch(dz.coords(scaffold), k4j.coords(scaffold))
    dz.transform(R, t)

    # 6Z1V numbers its GS expression tag 112/113, where CD9 P21926 has SER112/HIS113.
    # Import 114-190 only: the donors' last residue 191 lands 2.2 A (6Z1V) / 3.4 A
    # (6RLO) from 6K4J's own CA191 and leaves a 3.6 A / 6.4 A peptide-bond break at
    # the 191/192 junction, whereas ending the graft at 190 closes it to 2.2 / 2.7 A.
    graft_lo, graft_hi = 114, 190
    apex = dz.subset(graft_lo, graft_hi)
    assert_wt(apex, acc, range(graft_lo, graft_hi + 1), f"{donor_id} {graft_lo}-{graft_hi}")
    apex.set_occ(OCC_DONOR)

    k4j.drop([i for i in range(graft_lo, graft_hi + 1)])
    k4j.merge(apex)
    mark(prov, range(graft_lo, graft_hi + 1), src=f"{donor_id}, grafted", occ=OCC_DONOR)
    junctions = [(graft_lo - 1, graft_lo), (graft_hi, graft_hi + 1)]
    steps.append(
        f"EC2 {graft_lo}-{graft_hi} REPLACED WHOLESALE by {donor_id}, grafted on the "
        f"helix A/B/E scaffold 114-151+181-190 (n={len(scaffold)}, RMSD {rms:.2f} A). "
        f"6K4J does not "
        f"merely lack T175-K179: SEQADV lists them as DELETION, they are absent from "
        f"SEQRES and absent from REMARK 465, and the model welds E174 straight onto S180 "
        f"(CA-CA 3.79 A), so the whole C/D apex 156-174 is deformed by 2.3-11.1 A, not "
        f"just the five deleted residues. A 5-residue insertion would not have repaired "
        f"it. {apex_note}")
    steps.append("Residues 112-113 are taken from 6K4J, NOT from the donor: both LEL-only "
                 "CD9 entries number an expression tag at 112/113 (6Z1V has GLY112/SER113) "
                 "while P21926 has SER112/HIS113 there, so importing them would have put "
                 "two wrong residues into the scored surface. 191-195 also come from 6K4J. "
                 "This is why the builder runs a hard wild-type check against the UniProt "
                 "sequence on every donor block before splicing it.")
    steps.append("T175-F176-T177-V178-K179 are now PRESENT. In this membrane frame they "
                 "sit 18-24 A above the hydrophobic-core boundary — the most "
                 "polymer-accessible point on the whole protein, and precisely where "
                 "three independent antibodies chose to bind. Every calculation this "
                 "project has run so far omitted the tip of its own target.")

    truncated = sorted(set(k4j.truncated.get("A", {})))
    fixed, added_atoms, added_res, versions = pdbfixer_pass(k4j, [], {})
    steps.append(f"Truncated side chains rebuilt by pdbfixer {versions['pdbfixer']} "
                 f"(6K4J REMARK 470: {truncated}). N51 and N52 matter most: both lack "
                 f"CG/OD1/ND2, they are the glycosylation-relevant asparagines, and they "
                 f"are therefore the least well determined atoms anchoring CD9's "
                 f"glycan-shadow mask.")

    name = "CD9_open_v1" if variant == "a" else "CD9_open_v1b"
    return dict(name=name, target="CD9", model=fixed, prov=prov, steps=steps,
                added_atoms=added_atoms, added_res=added_res, versions=versions,
                half=half, frame=frame_desc, junctions=junctions,
                primary=("6K4J chain A (X-ray 2.70 A, lipidic cubic phase) -> membrane frame, "
                         "TM1-4, EC1, and residues 112-113 + 191-195"),
                donors=[f"{donor_id} -> EC2 {graft_lo}-{graft_hi}"],
                conformer="OPEN/UPRIGHT (experimental, membrane-embedded, no partner)")


BUILDERS = {
    "CD63_open_v1": build_cd63,
    "CD81_open_v1": build_cd81_open,
    "CD81_closed_5tcx_v1": build_cd81_closed,
    "CD9_open_v1": lambda: build_cd9("a"),
    "CD9_open_v1b": lambda: build_cd9("b"),
}
BY_TARGET = {"CD63": ["CD63_open_v1"],
             "CD81": ["CD81_open_v1", "CD81_closed_5tcx_v1"],
             "CD9": ["CD9_open_v1", "CD9_open_v1b"]}


# ══════════════════════════════════════════════════════════════════════════
# REMARK BLOCK + VERIFICATION
# ══════════════════════════════════════════════════════════════════════════
def build_remarks(b, report) -> list:
    t = b["target"]
    R = ["=" * 68,
         f"PREPARED RECEPTOR TEMPLATE  {b['name']}   target {t} ({UNIPROT[t]})",
         "=" * 68,
         f"Built {time.strftime('%Y-%m-%d %H:%M')} by code/tools/prepare_cd_templates.py",
         "Purpose: extracellular surface of the tetraspanin AS IT SITS IN A VESICLE",
         "MEMBRANE. Membrane-embedded and lumen-facing surface is irrelevant here.",
         "",
         "THIS FILE IS A CHIMERA / REPAIRED MODEL. IT IS NOT A DEPOSITED STRUCTURE.",
         "Do not cite it as one.",
         "",
         "-- SOURCES ------------------------------------------------------",
         f"PRIMARY : {b['primary']}"]
    for d in b["donors"]:
        R.append(f"DONOR   : {d}")
    R += ["",
          f"CONFORMER: {b['conformer']}",
          f"MEMBRANE FRAME: {b['frame']}",
          f"  hydrophobic half-thickness {b['half']:.1f} A; bilayer normal = +z;",
          f"  mid-plane z = 0; extracellular side = +z.",
          "",
          "-- PROVENANCE CHANNEL (occupancy column) ------------------------",
          "  1.00 experimental, primary entry",
          "  0.75 experimental, spliced/grafted donor entry",
          "  0.50 side chain REBUILT (truncated in the deposition, or a reverted",
          "       mutation) - a predicted rotamer",
          "  0.25 backbone built DE NOVO (loop closure) - predicted",
          "  0.00 AlphaFold scaffold - predicted",
          f"  atom counts: {report['occ_counts']}",
          "",
          "-- REPAIRS PERFORMED --------------------------------------------"]
    for i, s in enumerate(b["steps"], 1):
        R.append(f"({i}) {s}")
    R += ["",
          "-- REBUILT / DE NOVO RESIDUES -----------------------------------",
          f"de novo backbone residues : {b['added_res'] or 'none'}",
          f"residues with >=1 rebuilt atom : {report['rebuilt_residues'] or 'none'}",
          f"total rebuilt atoms : {len(b['added_atoms'])}",
          "",
          "-- DISULFIDES ---------------------------------------------------",
          f"pairing : {DISULFIDE_PAIRS[t]}",
          f"citation: {DISULFIDE_CITATION[t]}"]
    for (a, c), d in zip(DISULFIDE_PAIRS[t], report["ss_dist"]):
        R.append(f"  CYS {a} - CYS {c}  SG-SG = "
                 f"{'%.2f A' % d if d is not None else 'NOT MEASURABLE (residue absent)'}")
    R += ["",
          "-- TOOLS --------------------------------------------------------",
          f"pdbfixer {b['versions'].get('pdbfixer')}, OpenMM {b['versions'].get('openmm')}, "
          f"numpy {np.__version__}",
          f"minimisation: {report['minimisation']}",
          "",
          "-- COVERAGE -----------------------------------------------------",
          f"modelled span {report['span'][0]}-{report['span'][1]}, "
          f"{report['n_res']} residues, internal gaps: {report['gaps'] or 'none'}",
          f"EC1 {TOPOLOGY[t]['ec1']}: {report['ec1_modelled']} residues modelled "
          f"(EC1 IS CARRIED AS STERIC CONTEXT AND IS NOT SCORED, for all three targets)",
          f"EC2 {TOPOLOGY[t]['ec2']}: {report['ec2_modelled']} residues modelled",
          f"EC2 membrane-occluded (CA below core+6 A): {report['occluded']}",
          f"EC2 polymer-accessible (scored_surface): {report['scored']}",
          f"occluded-count sensitivity to the assumed half-thickness: "
          f"{report['occluded_sensitivity']}",
          f"backbone C-N bonds longer than 1.5 A: "
          f"{report['stretched_bonds'] or 'none - the chain is continuous'}",
          "",
          "-- WHAT IS STILL MISSING ----------------------------------------"]
    R += [f"  {x}" for x in report["caveats"]]
    R.append("=" * 68)
    return R


def peptide_bonds(m: Model, tol: float = 1.5, lo: float = 1.2):
    """Every i -> i+1 C-N distance outside [`lo`, `tol`] A. A prepared template
    with a broken backbone bond must not be docked.

    THE LOWER BOUND IS NOT COSMETIC. This check used to test `d > tol` only, and
    a splice is just as likely to COMPRESS a junction as to stretch it: CD63's
    108/109 seam sat at 1.045 A -- a larger absolute error than the 1.5 A limit
    this function does test for -- and was reported as "none", i.e. as a
    continuous chain. A one-sided test on a two-sided failure mode reports
    success on half of its own domain.

    THE C-N DISTANCE ALONE IS NOT ENOUGH EITHER. It is one scalar with no
    direction, so a junction can satisfy it while the peptide unit is folded
    flat: CD63's 198/199 seam sat at C-N 1.45 A (in range) with CA(198)...N(199)
    at 0.985 A and a CA-C-N angle of 38 deg instead of 116 deg. The two 1-3
    distances are therefore checked as well -- they are what actually detects a
    collapsed junction, and they are the reason a template can pass "the chain
    is continuous" while being unusable.
    """
    bad = {}
    nums = m.nums()
    for a, c in zip(nums, nums[1:]):
        if c != a + 1:
            continue
        ra, rc = m.res[a], m.res[c]
        if "C" in ra.atoms and "N" in rc.atoms:
            d = float(np.linalg.norm(ra.atoms["C"].xyz - rc.atoms["N"].xyz))
            if d > tol or d < lo:
                bad[f"{a}-{c}"] = round(d, 2)
        # 1-3 distances across the junction; ideal ~2.43-2.46 A for both.
        for nm_a, nm_c, key in (("CA", "N", "CA..N"), ("C", "CA", "C..CA")):
            if nm_a in ra.atoms and nm_c in rc.atoms:
                d13 = float(np.linalg.norm(ra.atoms[nm_a].xyz - rc.atoms[nm_c].xyz))
                if not 2.2 <= d13 <= 2.7:
                    bad[f"{a}-{c} {key}"] = round(d13, 2)
    return bad


def stereo_problems(m: Model) -> dict:
    """D-amino acids and non-proline cis peptide bonds.

    Neither was checked before, and both were present in the shipped set:
    CD81_open_v1 carried three D-residues (K116, T167, L170) and one non-Pro cis
    bond (L170-K171) out of pdbfixer's de novo loop, and a restrained
    minimisation that had to pull apart a 0.985 A junction overlap inverted
    CD63's TRP198 -- an EXPERIMENTAL, scored-surface residue. Both are invisible
    to a C-N distance test: a cis bond has a perfectly normal 1.33 A C-N, and an
    inverted CA changes no bond length at all. An unnoticed D-residue silently
    changes the shape of the surface a docking screen scores.
    """
    d_res, cis = [], []
    nums = m.nums()
    for i in nums:
        r = m.res[i]
        if r.resname == "GLY":
            continue                      # no CB: not a stereocentre
        a = r.atoms
        if not all(n in a for n in ("N", "CA", "C", "CB")):
            continue
        ca = a["CA"].xyz
        # L-amino acids give a POSITIVE (CA->N) x (CA->C) . (CA->CB) triple product
        trip = float(np.dot(np.cross(a["N"].xyz - ca, a["C"].xyz - ca),
                            a["CB"].xyz - ca))
        if trip < 0:
            d_res.append(f"{r.resname}{i}")
    for x, y in zip(nums, nums[1:]):
        if y != x + 1:
            continue
        ra, rc = m.res[x], m.res[y]
        if not ({"CA", "C"} <= set(ra.atoms) and {"N", "CA"} <= set(rc.atoms)):
            continue
        p0, p1 = ra.atoms["CA"].xyz, ra.atoms["C"].xyz
        p2, p3 = rc.atoms["N"].xyz, rc.atoms["CA"].xyz
        b1 = p2 - p1
        b1 /= np.linalg.norm(b1)
        v = (p0 - p1) - np.dot(p0 - p1, b1) * b1
        w = (p3 - p2) - np.dot(p3 - p2, b1) * b1
        omega = float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w),
                                            np.dot(v, w))))
        if abs(omega) < 30.0 and rc.resname != "PRO":
            cis.append(f"{ra.resname}{x}-{rc.resname}{y}({omega:.0f}deg)")
    return {"d_amino_acids": d_res, "cis_nonpro": cis}


def analyse(b, minim_note) -> dict:
    m, t = b["model"], b["target"]
    ec1, ec2 = TOPOLOGY[t]["ec1"], TOPOLOGY[t]["ec2"]

    occ_counts = {}
    for i in m.nums():
        for a in m.res[i].atoms.values():
            occ_counts[f"{a.occ:.2f}"] = occ_counts.get(f"{a.occ:.2f}", 0) + 1

    ss = []
    for a, c in DISULFIDE_PAIRS[t]:
        ra, rc = m.res.get(a), m.res.get(c)
        if ra and rc and "SG" in ra.atoms and "SG" in rc.atoms:
            ss.append(float(np.linalg.norm(ra.atoms["SG"].xyz - rc.atoms["SG"].xyz)))
        else:
            ss.append(None)

    occ_res = occluded(m, ec2[0], ec2[1], b["half"])
    ec2_res = [i for i in m.nums() if ec2[0] <= i <= ec2[1]]
    scored = [i for i in ec2_res if i not in occ_res]

    caveats = []
    if b["added_res"]:
        caveats.append(f"residues {b['added_res']} have DE NOVO backbone (predicted)")
    unmodelled_ec2 = [i for i in range(ec2[0], ec2[1] + 1) if i not in m.res]
    if unmodelled_ec2:
        caveats.append(f"EC2 residues still absent: {unmodelled_ec2}")
    unmodelled_ec1 = [i for i in range(ec1[0], ec1[1] + 1) if i not in m.res]
    if unmodelled_ec1:
        caveats.append(f"EC1 residues still absent: {unmodelled_ec1} (EC1 is unscored)")
    if N_GLYCAN_SEQUONS[t]:
        caveats.append(
            f"NO GLYCANS ARE MODELLED at the N-glycan sequons {N_GLYCAN_SEQUONS[t]}. "
            f"Apply the glycan-shadow mask instead. Consequence to state in any "
            f"write-up: boronic-acid monomers (VPBA, AAPBA) will score identically on "
            f"all three aglycosyl templates while being plausibly the most "
            f"CD63-selective chemistry available, so a null result for them from this "
            f"screen is uninformative.")
    else:
        caveats.append("CD81 has ZERO extracellular N-glycan sequons (the only N-X-S/T "
                       "in P60033 is the cytoplasmic N232) — nothing is missing here, "
                       "this is a real biological asymmetry and makes CD81 the negative "
                       "control on the glycan axis.")

    sensitivity = {f"half={h}": len(occluded(m, ec2[0], ec2[1], h))
                   for h in HALF_THICKNESS_BRACKET}

    return dict(occ_counts=occ_counts, ss_dist=ss, span=m.span(), n_res=len(m.res),
                gaps=m.gaps(), occluded=occ_res, scored=scored,
                occluded_sensitivity=sensitivity,
                stretched_bonds=peptide_bonds(m),
                ec1_modelled=sum(1 for i in m.nums() if ec1[0] <= i <= ec1[1]),
                ec2_modelled=len(ec2_res),
                rebuilt_residues=sorted({i for i, _ in b["added_atoms"]}),
                minimisation=minim_note, caveats=caveats)


def verify(path: Path, target: str, half: float) -> dict:
    """Independent re-read of the written file. Nothing is trusted in memory."""
    m = read_pdb(path, "A", keep_hetatm=())
    seq = uniprot_seq(UNIPROT[target])
    mismatched = [f"{i}{m.res[i].one}" for i in m.nums()
                  if not (1 <= i <= len(seq)) or m.res[i].one != seq[i - 1]]
    ss = {}
    for a, c in DISULFIDE_PAIRS[target]:
        ra, rc = m.res.get(a), m.res.get(c)
        ss[f"{a}-{c}"] = (round(float(np.linalg.norm(ra.atoms["SG"].xyz - rc.atoms["SG"].xyz)), 3)
                          if ra and rc and "SG" in ra.atoms and "SG" in rc.atoms else None)
    chains = set()
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            chains.add(line[21:22])
    return dict(file=str(path), span=m.span(), n_res=len(m.res), gaps=m.gaps(),
                chains=sorted(chains), sequence_mismatches=mismatched,
                ss_sg_sg=ss,
                n_hetatm=sum(1 for l in open(path) if l.startswith("HETATM")),
                # WAS: {f"{a.occ:.2f}": 0 for ...} -- a dict comprehension that
                # built the right KEYS and pinned every value to the literal 0.
                # This is the one field produced by independently re-reading the
                # written file, so the provenance channel reported "0 atoms" in
                # every class and could never contradict anything. Now counted.
                occ_histogram=dict(sorted(Counter(
                    f"{a.occ:.2f}" for i in m.nums()
                    for a in m.res[i].atoms.values()).items())),
                stereo=stereo_problems(m),
                )


# ══════════════════════════════════════════════════════════════════════════
def build_one(name: str, do_min: bool) -> dict:
    log(f"=== building {name} ===")
    b = BUILDERS[name]()

    # apply the provenance channel: residue-level default, then atom-level overrides
    m = b["model"]
    for i in m.nums():
        occ = b["prov"].get(i, {}).get("occ", OCC_EXPERIMENTAL)
        for a in m.res[i].atoms.values():
            a.occ = occ
    denovo = set(b["added_res"])
    for i, atom in b["added_atoms"]:
        r = m.res.get(i)
        if r and atom in r.atoms:
            r.atoms[atom].occ = OCC_DENOVO_BB if i in denovo else OCC_REBUILT_SC
    for i in denovo:
        for a in m.res[i].atoms.values():
            a.occ = OCC_DENOVO_BB

    # Held during minimisation: every atom this script did NOT invent.
    added = {(i, n) for i, n in b["added_atoms"]}
    restrain = {(i, n) for i in m.nums() for n in m.res[i].atoms
                if (i, n) not in added}
    # Released: de novo residues, plus the two residues each side of a junction.
    free = set(b["added_res"])
    for a, c in b.get("junctions", []):
        free.update(range(a - 2, c + 3))
    free &= set(m.nums())

    minim_note = "not attempted (--no-minimize)"
    minim_ok = None if not do_min else False
    pre_bad = peptide_bonds(m)
    if do_min:
        # RETRY, because this step is not reproducible. pdbfixer.addMissingAtoms
        # and Modeller.addHydrogens place invented atoms differently on every
        # call, so the same input can start at E=+6e5 kJ/mol (fine) or at
        # E=-2e7 with Fmax=1e13 (a GB Born-radius singularity from two atoms
        # landing on top of each other), and the latter NaNs out immediately.
        # Measured on CD63_open_v1: 1 failure, then 6/6 successes on the SAME
        # model. So a single attempt is a lottery every template plays -- CD81
        # and CD9 minimised on the first try by luck, not by being easier.
        last_err = None
        for attempt in range(1, MINIMISE_ATTEMPTS + 1):
            try:
                log(f"    restrained minimisation ({name}, attempt "
                    f"{attempt}/{MINIMISE_ATTEMPTS}); {len(free)} residues released, "
                    f"stretched bonds before: {pre_bad or 'none'}")
                m2, stats = minimize(m, restrain, free_nums=free)
                for i in m2.nums():                       # keep occupancies
                    for n2, a2 in m2.res[i].atoms.items():
                        if n2 in m.res[i].atoms:
                            a2.occ = m.res[i].atoms[n2].occ
                # A minimisation that buys geometry with stereochemistry is a
                # FAILED minimisation, not a successful one. Measured: relaxing
                # CD63's 0.985 A junction overlap flipped TRP198 -- an
                # experimental, scored-surface residue -- into a D-amino acid,
                # and every existing check passed it, because an inverted CA
                # changes no bond length. The step is stochastic, so rejecting
                # the attempt and drawing again is a real fix, not a retry-loop
                # that hides the problem.
                _s_pre, _s_post = stereo_problems(m), stereo_problems(m2)
                _new_d = sorted(set(_s_post["d_amino_acids"]) - set(_s_pre["d_amino_acids"]))
                _new_c = sorted(set(_s_post["cis_nonpro"]) - set(_s_pre["cis_nonpro"]))
                if _new_d or _new_c:
                    raise RuntimeError(
                        f"minimisation introduced stereochemistry errors: "
                        f"D-amino acids {_new_d or 'none'}, non-Pro cis bonds "
                        f"{_new_c or 'none'}")
                # ACCEPTANCE IS "A VALID STRUCTURE CAME OUT", NOT "NOTHING
                # THREW". A run that stops at E = -7.0e5 kJ/mol (-193 per atom,
                # ~25x below anything physical for this system in implicit
                # solvent) having moved ZERO atoms and left the 1.04 A junction
                # untouched is a diverged GB evaluation, and the first version
                # of this gate accepted it because it only looked at
                # stereochemistry. Both symptoms are now disqualifying.
                _n_at = sum(len(m2.res[i].atoms) for i in m2.nums())
                _e_at = stats["e_after_kJ"] / max(1, _n_at)
                if not np.isfinite(stats["e_after_kJ"]) or _e_at < -50.0:
                    raise RuntimeError(
                        f"minimisation converged to an unphysical energy: "
                        f"{stats['e_after_kJ']:.0f} kJ/mol over {_n_at} atoms "
                        f"= {_e_at:.1f} kJ/mol/atom")
                _bad_post = peptide_bonds(m2)
                if _bad_post:
                    raise RuntimeError(
                        f"minimisation left backbone bond(s) out of range: "
                        f"{_bad_post} (before: {pre_bad or 'none'})")
                b["model"] = m = m2
                minim_note = (f"OpenMM restrained minimisation, 10 kJ/mol/A^2 on every atom "
                              f"this script did not invent ({stats['n_restrained']} atoms held; "
                              f"{len(free)} residues released, namely the de novo ones and the "
                              f"two residues flanking each splice junction); "
                              f"E {stats['e_before_kJ']:.0f} -> {stats['e_after_kJ']:.0f} kJ/mol; "
                              f"{stats['n_atoms_moved_gt_0.1A']} heavy atoms moved > 0.1 A; "
                              f"stretched backbone bonds {pre_bad or 'none'} -> "
                              f"{peptide_bonds(m) or 'none'}"
                              + (f"; succeeded on attempt {attempt} of "
                                 f"{MINIMISE_ATTEMPTS}." if attempt > 1 else "."))
                minim_ok = True
                log(f"    {minim_note}")
                break
            except Exception as e:
                last_err = e
                log(f"    minimisation attempt {attempt}/{MINIMISE_ATTEMPTS} "
                    f"FAILED: {e}")
        if not minim_ok:
            minim_note = (f"FAILED on all {MINIMISE_ATTEMPTS} attempts and therefore NOT "
                          f"APPLIED: {type(last_err).__name__}: {last_err}. Rebuilt side "
                          f"chains and de novo loops are raw pdbfixer output.")
            log(f"    minimisation FAILED after {MINIMISE_ATTEMPTS} attempts: {last_err}")

    report = analyse(b, minim_note)
    remarks = build_remarks(b, report)
    out = PREPARED / f"{b['name']}.pdb"
    write_pdb(m, out, remarks=remarks, chain="A")
    log(f"    wrote {out}")

    v = verify(out, b["target"], b["half"])
    v.update(name=b["name"], target=b["target"], conformer=b["conformer"],
             half_thickness=b["half"], frame=b["frame"], primary=b["primary"],
             donors=b["donors"], steps=b["steps"],
             de_novo_residues=b["added_res"],
             rebuilt_atoms=[f"{i}:{n}" for i, n in b["added_atoms"]],
             occluded_ec2=report["occluded"], scored_surface=report["scored"],
             occluded_sensitivity=report["occluded_sensitivity"],
             stretched_bonds=report["stretched_bonds"],
             minimisation=minim_note,
             # Machine-readable twin of `minimisation`, on the dict build_one
             # actually RETURNS, so main() can refuse to exit 0 on a template
             # that was never relaxed. The prose field already told the truth;
             # the PROCESS did not, and prose is not checkable.
             minimisation_ok=minim_ok,
             tools=b["versions"])
    (PREPARED / f"{b['name']}_provenance.json").write_text(json.dumps(v, indent=2))

    log(f"    span={v['span']} n={v['n_res']} gaps={v['gaps']} chains={v['chains']} "
        f"seq_mismatch={v['sequence_mismatches'] or 'NONE'} SS={v['ss_sg_sg']}")
    return v


def validate_frame() -> dict:
    """Is the TM-derived frame (CD63's only option) trustworthy?

    Run the derivation on 5TCX, which HAS an OPM fit, and compare.  This is the
    evidence behind CD63_HALF_THICKNESS being asserted rather than estimated.
    """
    m = read_pdb(RAW / "pdb" / "5TCX.pdb", "A")
    R, t, extent_half = frame_from_tm(m, TOPOLOGY["CD81"]["tm"], signs=[+1, -1, +1, -1])
    m.transform(R, t)
    if np.mean([m.res[i].atoms["CA"].z for i in m.nums() if 113 <= i <= 201]) < 0:
        m.transform(np.diag([1.0, -1.0, -1.0]), np.zeros(3))
    opm = read_pdb(RAW / "opm" / "5tcx_opm.pdb", "A")
    common = [i for i in m.nums()
              if i in opm.res and "CA" in m.res[i].atoms and "CA" in opm.res[i].atoms]
    dz = [opm.res[i].atoms["CA"].z - m.res[i].atoms["CA"].z for i in common]
    ref = occluded(opm, 113, 201, 17.1)
    with_extent = occluded(m, 113, 201, extent_half)
    with_opm_half = occluded(m, 113, 201, 17.1)
    return {
        "opm_half_thickness": 17.1,
        "tm_extent_estimate": round(extent_half, 2),
        "midplane_offset_mean_A": round(float(np.mean(dz)), 3),
        "midplane_offset_sd_A": round(float(np.std(dz)), 3),
        "opm_occluded": ref,
        "derived_with_extent_estimate": with_extent,
        "derived_with_opm_half": with_opm_half,
        "verdict": ("orientation and origin are sound (mid-plane within 0.1 A of OPM); "
                    "the TM z-extent thickness estimator is NOT sound and is therefore "
                    "not used - CD63_HALF_THICKNESS is asserted instead"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--target", choices=sorted(BY_TARGET))
    ap.add_argument("--name", choices=sorted(BUILDERS))
    ap.add_argument("--no-minimize", action="store_true")
    ap.add_argument("--selfcheck", action="store_true",
                    help="re-verify already-written templates, build nothing")
    ap.add_argument("--validate-frame", action="store_true",
                    help="check the TM-derived membrane frame against OPM on 5TCX")
    args = ap.parse_args()

    if args.validate_frame:
        print(json.dumps(validate_frame(), indent=2))
        return

    PREPARED.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    if args.selfcheck:
        out = {}
        for name in BUILDERS:
            p = PREPARED / f"{name}.pdb"
            if not p.is_file():
                out[name] = "MISSING"
                continue
            meta = json.loads((PREPARED / f"{name}_provenance.json").read_text())
            out[name] = verify(p, meta["target"], meta["half_thickness"])
        print(json.dumps(out, indent=2))
        return

    names = ([args.name] if args.name else
             BY_TARGET[args.target] if args.target else list(BUILDERS))
    results = {}
    for n in names:
        results[n] = build_one(n, do_min=not args.no_minimize)

    (LOGS / "prepare_cd_templates.log").write_text("\n".join(_LOG_LINES) + "\n")
    (PREPARED / "SUMMARY.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {"span": v["span"], "gaps": v["gaps"],
                          "seq_mismatches": v["sequence_mismatches"],
                          "ss": v["ss_sg_sg"], "chains": v["chains"],
                          # WAS ABSENT. A template whose minimisation NaN'd was
                          # written anyway and reported through a summary that
                          # did not mention minimisation at all, under exit 0 --
                          # indistinguishable from a relaxed one.
                          "minimisation_ok": v.get("minimisation_ok"),
                          "stereo": v.get("stereo")}
                      for k, v in results.items()}, indent=2))

    rc = 0
    unrelaxed = sorted(k for k, v in results.items()
                       if v.get("minimisation_ok") is False)
    if unrelaxed:
        print(f"\nMINIMISATION FAILED for {len(unrelaxed)} template(s): "
              f"{', '.join(unrelaxed)}", file=sys.stderr)
        print("Their de novo loops and rebuilt side chains are RAW pdbfixer output "
              "and were never relaxed. Do not dock or run MD against them; re-run "
              "this script (the step is stochastic) or investigate.", file=sys.stderr)
        rc = 1

    bad_stereo = {k: v["stereo"] for k, v in results.items()
                  if (v.get("stereo") or {}).get("d_amino_acids")
                  or (v.get("stereo") or {}).get("cis_nonpro")}
    if bad_stereo:
        print(f"\nSTEREOCHEMISTRY DEFECTS in {len(bad_stereo)} template(s):",
              file=sys.stderr)
        for k, s in sorted(bad_stereo.items()):
            print(f"  {k}: D-amino acids {s['d_amino_acids'] or 'none'}; "
                  f"non-Pro cis bonds {s['cis_nonpro'] or 'none'}", file=sys.stderr)
        print("A D-residue changes the shape of the surface a docking screen "
              "scores, and no force field will restore it during MD. Rebuild "
              "(pdbfixer's loop geometry is stochastic) or repair by hand.",
              file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
