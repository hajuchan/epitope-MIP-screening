#!/usr/bin/env python3
"""
head_integrity.py — audit a 16-mer head (epitope) against a candidate template
==============================================================================

WHY THIS EXISTS
---------------
The selectivity index (SI) this project reports is NOT computed on the EC2
docking receptor.  It is computed in Phase 5 by re-binding a 16-mer *head*
peptide into the cavity and taking cross_rmsd / own_rmsd:

    code/pipeline/phase5_rebinding.py:2109   own  -> _run_rebinding_md(..., head, ...)
    code/pipeline/phase5_rebinding.py:2117   cross_head = phase1_results[ot]["head_pdb"]
    code/pipeline/phase5_rebinding.py:2121   cross -> _run_rebinding_md(..., cross_head, ...)

So a template swap that improves EC2 surface accessibility can still *destroy*
the number the project actually publishes, if the new template does not resolve
the head cleanly.  Ranking head candidates on SASA + GRAVY alone (which is what
phase1_epitope_prep.evaluate_epitope_candidates did) cannot see this: an
unmodelled residue simply drops out of the extracted sequence and the candidate
is scored on the survivors.

This module makes the damage measurable BEFORE a template is adopted.  For a
given (template, chain, head range) it classifies every residue of the head as

    ok                    present, all heavy atoms, sequence == UniProt, no binder
    construct_deleted     removed from the expression construct (SEQADV DELETION,
                          or simply absent from SEQRES) — a real piece of protein
                          that no amount of modelling recovers honestly
    unmodelled            in SEQRES, absent from ATOM (REMARK 465) — must be BUILT
    incomplete_sidechain  REMARK 470 — side-chain atoms missing
    seqadv_conflict       SEQADV CONFLICT / ENGINEERED MUTATION — residue identity
                          differs from the UniProt the SI is supposed to be about
    binder_contact        within `cutoff` A of a non-target polymer chain (Fab,
                          sybody, co-receptor).  Its conformation is the *bound*
                          one, not the free-vesicle one the polymer will see.

and reports the fractions.  `ok_fraction` is the one that matters: it is the
fraction of the head that the template supplies without invention.

USAGE
-----
    python3 code/tools/head_integrity.py --report          # full published table
    python3 code/tools/head_integrity.py --json out.json

    from head_integrity import audit_head
    audit_head(pdb, chain="A", head_range=(168,183), uniprot="P60033")

Every number this module prints is derived from the raw PDB text (SEQRES,
SEQADV, REMARK 465, REMARK 470, DBREF, ATOM) and the UniProt FASTA on disk.
Nothing is taken from a summary page or from memory.
"""
from __future__ import annotations

import argparse
import json
import re as _re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_STRUCT = _REPO / "structures"
_PDB_DIR = _STRUCT / "raw" / "pdb"
_UNP_DIR = _STRUCT / "raw" / "uniprot"
_OPM_DIR = _STRUCT / "raw" / "opm"

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # modified residues that still occupy a sequence position
    "MSE": "M", "YCM": "C", "CSO": "C", "SEP": "S", "TPO": "T",
}

# Heavy atoms expected per residue type (standard, no OXT, no hydrogens).
_SIDECHAIN = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CB", "CG", "CD"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}
_BACKBONE = ["N", "CA", "C", "O"]

# ── severity tiering ──────────────────────────────────────────────────────
# FATAL   the template cannot supply this residue without inventing chemistry
#         the vesicle does not have.  A head containing one of these is not a
#         head *of this protein* any more, so the SI computed from it is not
#         about the protein either.
# REPAIR  routinely rebuilt by the protonation/minimisation prep step
#         (utils_structure.assign_protonation_states).  Worth recording,
#         not worth disqualifying — every crystal structure has some.
FATAL_STATUSES = frozenset({
    "construct_deleted",    # residue deleted from the expression construct
    "unmodelled",           # backbone must be invented de novo
    "seqadv_conflict",      # residue identity differs from UniProt
    "binder_contact",       # conformation is the bound one, not the free one
    "outside_construct",    # template does not span this residue at all
})
REPAIRABLE_STATUSES = frozenset({"incomplete_sidechain"})


# ──────────────────────────────────────────────────────────────────────────
# raw PDB header parsing — no Bio.PDB, so nothing is silently repaired
# ──────────────────────────────────────────────────────────────────────────

def _read(pdb_path) -> list[str]:
    with open(pdb_path) as fh:
        return fh.readlines()


def parse_seqres(lines, chain) -> list[str]:
    """SEQRES residue names for `chain`, in order."""
    out = []
    for ln in lines:
        if ln.startswith("SEQRES") and ln[11] == chain:
            out.extend(ln[19:70].split())
    return out


# The column header that opens the residue list differs between entries and
# between remark types: "M RES C SSSEQI" (7JIC/5TCX REMARK 465), "M RES CSSEQI"
# (REMARK 470), sometimes "M RES C SSEQI".  Matching the literal string missed
# 7JIC's 465 block entirely and silently reclassified its unmodelled residues as
# construct deletions, so match the header by pattern instead.
_HDR_RE = _re.compile(r"M\s+RES\s+C\s*SS+EQI")


def parse_remark465(lines, chain) -> set[int]:
    """Residues declared missing (unmodelled) for `chain`."""
    missing, in_block = set(), False
    for ln in lines:
        if not ln.startswith("REMARK 465"):
            continue
        if _HDR_RE.search(ln):
            in_block = True
            continue
        if not in_block:
            continue
        body = ln[10:].rstrip()
        parts = body.split()
        # forms: "RES C SSEQI"  or  "M RES C SSEQI"
        if len(parts) >= 3 and parts[-2] == chain:
            try:
                missing.add(int(parts[-1]))
            except ValueError:
                pass
    return missing


def parse_remark470(lines, chain) -> dict[int, list[str]]:
    """Residues with missing atoms for `chain` -> list of absent atom names."""
    out, in_block = {}, False
    for ln in lines:
        if not ln.startswith("REMARK 470"):
            continue
        if _HDR_RE.search(ln):
            in_block = True
            continue
        if not in_block:
            continue
        parts = ln[10:].split()
        if len(parts) < 3:
            continue
        # forms: "RES C SSEQI ATOM..."  or  "M RES C SSEQI ATOM..."
        for i in range(len(parts) - 1):
            if parts[i] == chain:
                try:
                    resnum = int(parts[i + 1])
                except ValueError:
                    continue
                out.setdefault(resnum, []).extend(parts[i + 2:])
                break
    return out


def parse_seqadv(lines, chain) -> dict[int, dict]:
    """SEQADV records for `chain` -> {resnum: {pdb_res, unp_res, kind}}."""
    out = {}
    for ln in lines:
        if not ln.startswith("SEQADV"):
            continue
        if len(ln) < 40 or ln[16] != chain:
            continue
        pdb_res = ln[12:15].strip()
        try:
            resnum = int(ln[18:22])
        except ValueError:
            continue
        unp_res = ln[39:42].strip()
        try:
            unp_num = int(ln[43:48])
        except ValueError:
            unp_num = None
        kind = ln[49:].strip()
        out[resnum] = {"pdb_res": pdb_res, "unp_res": unp_res,
                       "unp_num": unp_num, "kind": kind}
    return out


def parse_dbref(lines, chain) -> dict | None:
    for ln in lines:
        if ln.startswith("DBREF") and len(ln) > 20 and ln[12] == chain:
            return {
                "pdb_begin": int(ln[14:18]), "pdb_end": int(ln[20:24]),
                "db": ln[26:32].strip(), "accession": ln[33:41].strip(),
                "db_begin": int(ln[55:60]), "db_end": int(ln[62:67]),
            }
    return None


def parse_atoms(lines, chain) -> dict[int, dict]:
    """Modelled residues of `chain` -> {resname, atoms:set, coords:list}."""
    res = {}
    for ln in lines:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        if ln[21] != chain:
            continue
        resname = ln[17:20].strip()
        if resname not in AA3TO1:
            continue                      # ligand / water / lipid, not a residue
        altloc = ln[16]
        if altloc not in (" ", "A"):
            continue
        elem = ln[76:78].strip()
        if elem == "H":
            continue
        try:
            resnum = int(ln[22:26])
        except ValueError:
            continue
        atom = ln[12:16].strip()
        e = res.setdefault(resnum, {"resname": resname, "atoms": set(),
                                    "coords": []})
        e["atoms"].add(atom)
        e["coords"].append((float(ln[30:38]), float(ln[38:46]),
                            float(ln[46:54])))
    return res


def parse_all_chain_atoms(lines) -> dict[str, list]:
    """Every polymer chain -> [(resnum, atomname, x, y, z)]."""
    out = {}
    for ln in lines:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        resname = ln[17:20].strip()
        if resname not in AA3TO1:
            continue
        altloc = ln[16]
        if altloc not in (" ", "A"):
            continue
        if ln[76:78].strip() == "H":
            continue
        ch = ln[21]
        try:
            resnum = int(ln[22:26])
        except ValueError:
            continue
        out.setdefault(ch, []).append(
            (resnum, ln[12:16].strip(),
             float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return out


def opm_frame(opm_pdb, chain) -> tuple:
    """(half_thickness, {resnum: CA z}) read from the OPM file's OWN frame.

    OPM re-orients the deposited coordinates so the membrane normal is z and
    the bilayer mid-plane is z=0, then adds DUM pseudo-atoms at +/- the
    hydrophobic half-thickness.  Mixing a half-thickness taken from the OPM
    file with CA z values taken from the *deposited* PDB compares two different
    frames and is meaningless, so both come from the OPM file here.
    """
    zs, ca = set(), {}
    try:
        lines = _read(opm_pdb)
    except OSError:
        return None, {}
    for ln in lines:
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        if ln[17:20].strip() == "DUM":
            zs.add(round(float(ln[46:54]), 3))
        elif ln[21] == chain and ln[12:16].strip() == "CA":
            try:
                ca[int(ln[22:26])] = float(ln[46:54])
            except ValueError:
                pass
    return (max(abs(z) for z in zs) if zs else None), ca


# ──────────────────────────────────────────────────────────────────────────
# UniProt
# ──────────────────────────────────────────────────────────────────────────

def uniprot_seq(accession: str) -> str:
    fa = _UNP_DIR / f"{accession}.fasta"
    seq = "".join(l.strip() for l in _read(fa) if not l.startswith(">"))
    return seq


# ──────────────────────────────────────────────────────────────────────────
# the audit
# ──────────────────────────────────────────────────────────────────────────

def _contacts(lines, chain, cutoff=4.5) -> dict[int, list[str]]:
    """Residues of `chain` within `cutoff` A of any OTHER polymer chain."""
    chains = parse_all_chain_atoms(lines)
    if chain not in chains or len(chains) < 2:
        return {}
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError:
        return {}
    tgt = chains[chain]
    tgt_tree = cKDTree(np.array([a[2:5] for a in tgt]))
    out: dict[int, set] = {}
    for other, atoms in chains.items():
        if other == chain:
            continue
        oth_tree = cKDTree(np.array([a[2:5] for a in atoms]))
        for i, hits in enumerate(tgt_tree.query_ball_tree(oth_tree, r=cutoff)):
            if hits:
                out.setdefault(tgt[i][0], set()).add(other)
    return {k: sorted(v) for k, v in out.items()}


def audit_head(pdb_path, chain: str, head_range: tuple, accession: str,
               cutoff: float = 4.5, opm_pdb=None,
               membrane_margin: float = 6.0) -> dict:
    """Classify every residue of `head_range` on this template.

    Returns a dict with per-residue status and aggregate fractions.  See the
    module docstring for what each status means.
    """
    pdb_path = Path(pdb_path)
    lines = _read(pdb_path)
    lo, hi = int(head_range[0]), int(head_range[1])
    span = list(range(lo, hi + 1))

    seqres = parse_seqres(lines, chain)
    missing = parse_remark465(lines, chain)
    incomplete = parse_remark470(lines, chain)
    seqadv = parse_seqadv(lines, chain)
    dbref = parse_dbref(lines, chain)
    atoms = parse_atoms(lines, chain)
    contacts = _contacts(lines, chain, cutoff)
    unp = uniprot_seq(accession)

    # Residue numbers that SEQRES accounts for.  A construct deletion shows up
    # as a residue that is neither modelled nor declared missing, yet lies
    # inside the DBREF span — the 6K4J CD9 175-179 case.
    seqres_len = len(seqres)
    modelled = set(atoms)
    declared_missing = set(missing)

    half_t, ca_z = opm_frame(opm_pdb, chain) if opm_pdb else (None, {})

    per_res, counts = [], {}
    for r in span:
        unp_aa = unp[r - 1] if 0 < r <= len(unp) else "?"
        entry = {"resnum": r, "uniprot_aa": unp_aa}
        adv = seqadv.get(r)

        if adv and "DELETION" in adv["kind"].upper():
            status = "construct_deleted"
        elif r not in modelled and r not in declared_missing:
            # inside the DBREF span but neither built nor declared absent ->
            # it was removed from the construct entirely
            inside = dbref and dbref["pdb_begin"] <= r <= dbref["pdb_end"]
            status = "construct_deleted" if inside else "outside_construct"
        elif r not in modelled:
            status = "unmodelled"
        elif adv and ("CONFLICT" in adv["kind"].upper()
                      or "MUTATION" in adv["kind"].upper()):
            status = "seqadv_conflict"
            entry["seqadv"] = f'{adv["pdb_res"]} vs UNP {adv["unp_res"]}: {adv["kind"]}'
        elif r in contacts:
            status = "binder_contact"
            entry["contact_chains"] = contacts[r]
        elif r in incomplete:
            status = "incomplete_sidechain"
            entry["missing_atoms"] = incomplete[r]
        else:
            rn = atoms[r]["resname"]
            need = set(_BACKBONE) | set(_SIDECHAIN.get(rn, []))
            gap = sorted(need - atoms[r]["atoms"])
            if gap:
                status = "incomplete_sidechain"
                entry["missing_atoms"] = gap
            elif AA3TO1.get(rn, "X") != unp_aa:
                status = "seqadv_conflict"
                entry["seqadv"] = f"ATOM {rn} vs UNP {unp_aa} (undeclared)"
            else:
                status = "ok"

        if half_t is not None and r in ca_z:
            entry["ca_z"] = round(ca_z[r], 2)
            entry["membrane_proximal"] = abs(ca_z[r]) < half_t + membrane_margin

        entry["status"] = status
        counts[status] = counts.get(status, 0) + 1
        per_res.append(entry)

    n = len(span)
    fatal = [e["resnum"] for e in per_res if e["status"] in FATAL_STATUSES]
    repairable = [e["resnum"] for e in per_res
                  if e["status"] in REPAIRABLE_STATUSES]
    seq_modelled = "".join(
        AA3TO1.get(atoms[r]["resname"], "X") if r in modelled else "-"
        for r in span)
    seq_unp = "".join(unp[r - 1] if 0 < r <= len(unp) else "?" for r in span)

    return {
        "template": pdb_path.name,
        "chain": chain,
        "accession": accession,
        "head_range": [lo, hi],
        "n_residues": n,
        "dbref": dbref,
        "seqres_len": seqres_len,
        "uniprot_seq": seq_unp,
        "modelled_seq": seq_modelled,
        "counts": counts,
        "ok_fraction": round(counts.get("ok", 0) / n, 4),
        "modelled_fraction": round(
            sum(1 for r in span if r in modelled) / n, 4),
        "binder_contacted_fraction": round(
            sum(1 for r in span if r in contacts) / n, 4),
        "damaged_residues": [e["resnum"] for e in per_res if e["status"] != "ok"],
        "fatal_residues": fatal,
        "repairable_residues": repairable,
        "fatal_fraction": round(len(fatal) / n, 4),
        "usable": not fatal,
        "opm_half_thickness": half_t,
        "residues": per_res,
    }


def accession_for(pdb_path, chain: str) -> str | None:
    """UniProt accession that DBREF assigns to `chain`."""
    ref = parse_dbref(_read(pdb_path), chain)
    return ref["accession"] if ref and ref["db"].startswith("UNP") else None


# ──────────────────────────────────────────────────────────────────────────
# the published table
# ──────────────────────────────────────────────────────────────────────────

# (target, accession) -> candidate templates that have been proposed for it.
# `role` records what the template is actually being asked to supply.
TEMPLATE_MATRIX = {
    "CD63": {
        "accession": "P08962",
        "head_range": (157, 172),          # results/phase1 selected_head_range
        "templates": [
            ("AF-P08962-F1-model_v6.pdb", "A", "alphafold", None),
            ("9HUR.pdb", None, "pdb", None),
            ("9HQ5.pdb", None, "pdb", None),
        ],
    },
    "CD81": {
        "accession": "P60033",
        "head_range": (168, 183),
        "templates": [
            ("5TCX.pdb", "A", "pdb", "5tcx_opm.pdb"),
            ("7JIC.pdb", "B", "pdb", "7jic_opm.pdb"),
        ],
    },
    "CD9": {
        "accession": "P21926",
        "head_range": (156, 171),
        "templates": [
            ("6K4J.pdb", "A", "pdb", "6k4j_opm.pdb"),
            ("AF-P21926-F1-model_v6.pdb", "A", "alphafold", None),
        ],
    },
}


def _resolve(fname, kind):
    if kind == "alphafold":
        return _STRUCT / "raw" / "alphafold" / fname
    return _PDB_DIR / fname


def _guess_chain(pdb_path, accession) -> str | None:
    """Chain whose DBREF points at `accession`."""
    for ln in _read(pdb_path):
        if ln.startswith("DBREF") and accession in ln:
            return ln[12]
    return None


def build_report() -> dict:
    report = {}
    for target, spec in TEMPLATE_MATRIX.items():
        acc = spec["accession"]
        rng = spec["head_range"]
        rows = []
        for fname, chain, kind, opm in spec["templates"]:
            path = _resolve(fname, kind)
            if not path.exists():
                rows.append({"template": fname, "error": "file not on disk"})
                continue
            ch = chain or _guess_chain(path, acc)
            if ch is None:
                rows.append({"template": fname,
                             "error": f"no DBREF chain for {acc}"})
                continue
            opm_path = (_OPM_DIR / opm) if opm else None
            rows.append(audit_head(path, ch, rng, acc, opm_pdb=opm_path))
        report[target] = {"accession": acc, "head_range": list(rng),
                          "templates": rows}
    return report


def print_report(report):
    W = 78
    print("=" * W)
    print("HEAD INTEGRITY — what each candidate template supplies for the")
    print("16-mer head that the Phase-5 selectivity index is computed from")
    print("=" * W)
    for target, spec in report.items():
        lo, hi = spec["head_range"]
        print(f"\n### {target}   head {lo}-{hi}   UniProt {spec['accession']}")
        print(f"{'template':<30} {'ok':>7} {'modelled':>9} {'binder':>8} "
              f"{'verdict':>9}  fatal")
        print("-" * W)
        for row in spec["templates"]:
            if "error" in row:
                print(f"{row['template']:<30} {'--':>7} {'--':>9} {'--':>8} "
                      f"{'--':>9}  {row['error']}")
                continue
            n = row["n_residues"]
            ok = row["counts"].get("ok", 0)
            mod = int(row["modelled_fraction"] * n + 0.5)
            bind = int(row["binder_contacted_fraction"] * n + 0.5)
            fat = row["fatal_residues"]
            verdict = "USABLE" if row["usable"] else "REJECT"
            label = f"{row['template']}:{row['chain']}"
            print(f"{label:<30} {ok:>3}/{n:<3} {mod:>5}/{n:<3} {bind:>4}/{n:<3} "
                  f"{verdict:>9}  {_compress(fat) if fat else 'none'}")
        print()
        for row in spec["templates"]:
            if "error" in row:
                continue
            print(f"  {row['template']}:{row['chain']}")
            print(f"     UniProt  {row['uniprot_seq']}")
            print(f"     modelled {row['modelled_seq']}")
            bad = [e for e in row["residues"] if e["status"] != "ok"]
            for e in bad:
                extra = (e.get("seqadv") or
                         (",".join(e["missing_atoms"]) if e.get("missing_atoms") else "") or
                         (f"contacts {'/'.join(e['contact_chains'])}"
                          if e.get("contact_chains") else ""))
                tier = "FATAL " if e["status"] in FATAL_STATUSES else "repair"
                print(f"       {e['resnum']:>4} {e['uniprot_aa']}  {tier} "
                      f"{e['status']:<21} {extra}")
            if not bad:
                print("       (every residue ok)")
            print()


def _compress(nums):
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for x in nums[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = x
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


# ──────────────────────────────────────────────────────────────────────────
# imprintable surface — the metric a template swap must actually be judged on
# ──────────────────────────────────────────────────────────────────────────

# Templates that have been proposed for the CD81 EC2 docking receptor.  The
# closed/cholesterol-bound 5TCX lies EC2-parallel to the membrane, so a swap to
# the open CD19-bound 7JIC was proposed to expose more surface.
SURFACE_MATRIX = {
    "CD81": {
        "accession": "P60033", "ec2": (113, 201),
        "templates": [("5TCX.pdb", "A", "5tcx_opm.pdb"),
                      ("7JIC.pdb", "B", "7jic_opm.pdb")],
    },
    "CD9": {
        "accession": "P21926", "ec2": (112, 195),
        "templates": [("6K4J.pdb", "A", "6k4j_opm.pdb")],
    },
}


def imprintable_surface(pdb_path, chain, ec2_range, accession, opm_pdb,
                        margin=6.0):
    """EC2 residues that are BOTH defect-free AND reachable from outside.

    On an intact vesicle the polymer only ever touches extracellular surface
    that clears the lipid headgroup band, and it can only be imprinted against
    atoms the template actually supplies.  Counting either criterion alone
    flatters a template: a wide-open conformer full of unmodelled loops and
    co-receptor contacts is not more imprintable than a closed one that is
    complete.  This is the intersection.
    """
    a = audit_head(pdb_path, chain, ec2_range, accession)
    half_t, ca = opm_frame(opm_pdb, chain)
    lo, hi = ec2_range
    ok = {e["resnum"] for e in a["residues"] if e["status"] == "ok"}
    reachable = {r for r in range(lo, hi + 1)
                 if r in ca and abs(ca[r]) >= half_t + margin}
    both = sorted(ok & reachable)
    n = hi - lo + 1
    return {
        "template": Path(pdb_path).name, "chain": chain,
        "ec2_range": [lo, hi], "n_ec2": n,
        "opm_half_thickness": half_t, "membrane_margin": margin,
        "n_defect_free": len(ok),
        "n_membrane_distal": len(reachable),
        "n_imprintable": len(both),
        "imprintable_residues": both,
        "imprintable_compressed": _compress(both),
        "membrane_proximal_residues": sorted(
            r for r in range(lo, hi + 1)
            if r in ca and abs(ca[r]) < half_t + margin),
    }


def build_surface_report() -> dict:
    out = {}
    for target, spec in SURFACE_MATRIX.items():
        rows = []
        for fname, chain, opm in spec["templates"]:
            rows.append(imprintable_surface(
                _PDB_DIR / fname, chain, tuple(spec["ec2"]),
                spec["accession"], _OPM_DIR / opm))
        out[target] = {"accession": spec["accession"],
                       "ec2_range": list(spec["ec2"]), "templates": rows}
    return out


def print_surface_report(report):
    W = 78
    print("=" * W)
    print("IMPRINTABLE EC2 SURFACE — defect-free AND clear of the headgroup band")
    print("=" * W)
    for target, spec in report.items():
        lo, hi = spec["ec2_range"]
        print(f"\n### {target}   EC2 {lo}-{hi}   UniProt {spec['accession']}")
        print(f"{'template':<16} {'half-t':>7} {'defect-free':>12} "
              f"{'memb-distal':>12} {'IMPRINTABLE':>12}")
        print("-" * W)
        for r in spec["templates"]:
            n = r["n_ec2"]
            print(f"{r['template']}:{r['chain']:<10} {r['opm_half_thickness']:>7} "
                  f"{r['n_defect_free']:>8}/{n:<3} "
                  f"{r['n_membrane_distal']:>8}/{n:<3} "
                  f"{r['n_imprintable']:>8}/{n:<3}")
        print()
        for r in spec["templates"]:
            print(f"  {r['template']}:{r['chain']} imprintable -> "
                  f"{r['imprintable_compressed']}")
            print(f"  {r['template']}:{r['chain']} membrane-proximal -> "
                  f"{_compress(r['membrane_proximal_residues'])}")
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--report", action="store_true", help="print the head table")
    ap.add_argument("--surface", action="store_true",
                    help="print the imprintable-EC2-surface table")
    ap.add_argument("--json", metavar="PATH", help="write the full audit JSON")
    ap.add_argument("--pdb", help="audit one file instead of the matrix")
    ap.add_argument("--chain")
    ap.add_argument("--range", help="e.g. 168-183")
    ap.add_argument("--accession")
    args = ap.parse_args(argv)

    if args.pdb:
        lo, hi = (int(x) for x in args.range.split("-"))
        res = audit_head(args.pdb, args.chain, (lo, hi), args.accession)
        print(json.dumps(res, indent=2))
        return 0

    report = {"heads": build_report(), "surface": build_surface_report()}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    if args.surface:
        print_surface_report(report["surface"])
    if args.report or not (args.json or args.surface):
        print_report(report["heads"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
