#!/usr/bin/env python3
"""
Head-integrity gate tests
=========================
The Phase-5 selectivity index is computed by re-binding 16-mer HEAD peptides,
not the EC2 docking receptor:

    phase5_rebinding.py:2109  own   -> _run_rebinding_md(..., head, ...)
    phase5_rebinding.py:2117  cross_head = phase1_results[ot]["head_pdb"]

so any template decision has to be judged on what it does to those 16-mers.
These tests pin that judgement.

G1  FROZEN SELECTION SURVIVES  — the gate must re-select exactly the heads that
                                 results/phase1 already contains.  A gate that
                                 changes the frozen selection would silently
                                 invalidate ~500 GB of downstream MD.
G2  7JIC REJECTED FOR THE HEAD — the proposed CD81 5TCX->7JIC swap must be
                                 refused: 15 of 16 head residues are built,
                                 reverted, or CD19-contacting.
G3  GLYCAN KNOCKOUTS REJECTED  — 9HUR/9HQ5 carry N130S/N150S/N172S; N172 is
                                 inside the CD63 head, so they cannot source it.
G4  CONSTRUCT DELETION SEEN    — 6K4J CD9 175-179 (TFTVK) is a construct
                                 deletion, not disorder: absent from SEQRES and
                                 from REMARK 465.  The auditor must not confuse
                                 it with an unmodelled residue.
G5  NO EMPTY CANDIDATE SET     — every target must retain at least one usable
                                 head candidate on its adopted template.  A plan
                                 that disqualifies a target's only candidate
                                 leaves it with no SI at all.

    python3 code/tests/test_head_integrity_gate.py
    pytest  code/tests/test_head_integrity_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
_REPO = _CODE.parent
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "tools"))

from head_integrity import (audit_head, accession_for, build_report,  # noqa: E402
                            parse_remark465, parse_seqres, _read)

_PDB = _REPO / "structures" / "raw" / "pdb"
_AF = _REPO / "structures" / "raw" / "alphafold"

# The templates results/phase1 actually used, and the heads it actually selected
# (read from results/phase1/phase1_results.json — see FROZEN below).
ADOPTED = {
    "CD63": (_AF / "AF-P08962-F1-model_v6.pdb", "A", (157, 172), "P08962"),
    "CD81": (_PDB / "5TCX.pdb", "A", (168, 183), "P60033"),
    "CD9":  (_PDB / "6K4J.pdb", "A", (156, 171), "P21926"),
}
FROZEN_SEQ = {
    "CD63": "IPSMSKNRVPDSCCIN",
    "CD81": "SVLKNNLCPSGSNIIS",
    "CD9":  "AGGVEQFISDICPKKD",
}


def _fail(msg):
    raise AssertionError(msg)


def test_frozen_selection_survives():
    """G1: the gate must pass every head results/phase1 already committed to."""
    for tgt, (pdb, chain, rng, acc) in ADOPTED.items():
        a = audit_head(pdb, chain, rng, acc)
        if not a["usable"]:
            _fail(f"G1 {tgt}: gate rejects the head already in results/phase1 "
                  f"({rng} on {pdb.name}); fatal={a['fatal_residues']}. "
                  f"Re-calibrate the gate, do not re-run the pipeline.")
        if a["modelled_seq"] != FROZEN_SEQ[tgt]:
            _fail(f"G1 {tgt}: template supplies {a['modelled_seq']!r}, "
                  f"results/phase1 recorded {FROZEN_SEQ[tgt]!r}")
        if a["uniprot_seq"] != FROZEN_SEQ[tgt]:
            _fail(f"G1 {tgt}: UniProt {acc} {rng} is {a['uniprot_seq']!r}, "
                  f"results/phase1 recorded {FROZEN_SEQ[tgt]!r}")
    return "G1 PASS  gate re-selects all 3 frozen heads; sequences == UniProt"


def test_7jic_rejected_for_head():
    """G2: 7JIC must not be adoptable as the source of the CD81 head."""
    a = audit_head(_PDB / "7JIC.pdb", "B", (168, 183), "P60033")
    if a["usable"]:
        _fail("G2: 7JIC passed the gate for the CD81 head; it must not")
    if len(a["fatal_residues"]) != 15:
        _fail(f"G2: expected 15 fatally damaged head residues on 7JIC, "
              f"got {len(a['fatal_residues'])}: {a['fatal_residues']}")
    if a["fatal_residues"] != list(range(168, 183)):
        _fail(f"G2: expected 168-182 damaged, got {a['fatal_residues']}")
    # and the alternative must be clean, or the swap would be a wash
    b = audit_head(_PDB / "5TCX.pdb", "A", (168, 183), "P60033")
    if not b["usable"] or b["damaged_residues"]:
        _fail(f"G2: 5TCX is not clean over 168-183: {b['damaged_residues']}")
    return ("G2 PASS  7JIC:B head 168-183 REJECT (15/16 fatal); "
            "5TCX:A USABLE (16/16 ok)")


def test_glycan_knockouts_rejected():
    """G3: 9HUR/9HQ5 mutate N172, which sits inside the CD63 head."""
    for pid in ("9HUR", "9HQ5"):
        path = _PDB / f"{pid}.pdb"
        chain = None
        for ln in _read(path):
            if ln.startswith("DBREF") and "P08962" in ln:
                chain = ln[12]
                break
        if chain is None:
            _fail(f"G3 {pid}: no DBREF chain for P08962")
        a = audit_head(path, chain, (157, 172), "P08962")
        if a["usable"]:
            _fail(f"G3 {pid}: passed the gate despite the N172S knockout")
        if 172 not in a["fatal_residues"]:
            _fail(f"G3 {pid}: N172 not flagged; fatal={a['fatal_residues']}")
    return "G3 PASS  9HUR and 9HQ5 both REJECT for the CD63 head (N172S)"


def test_construct_deletion_distinguished():
    """G4: 6K4J CD9 175-179 is a construct deletion, not disorder."""
    a = audit_head(_PDB / "6K4J.pdb", "A", (172, 187), "P21926")
    deleted = [e["resnum"] for e in a["residues"]
               if e["status"] == "construct_deleted"]
    if deleted != [175, 176, 177, 178, 179]:
        _fail(f"G4: expected 175-179 construct_deleted, got {deleted}")
    unmodelled = [e["resnum"] for e in a["residues"]
                  if e["status"] == "unmodelled"]
    if unmodelled:
        _fail(f"G4: 175-179 must not be reported as unmodelled; got {unmodelled}")
    lines = _read(_PDB / "6K4J.pdb")
    # ground truth 1: SEQADV names them as DELETION, not as anything milder
    seqadv_del = [ln for ln in lines
                  if ln.startswith("SEQADV") and "DELETION" in ln]
    got = [(ln[39:42].strip(), int(ln[43:48])) for ln in seqadv_del]
    want = [("THR", 175), ("PHE", 176), ("THR", 177), ("VAL", 178), ("LYS", 179)]
    if got != want:
        _fail(f"G4: SEQADV DELETION records changed: {got}")
    # ground truth 2: absent from REMARK 465 — a merely disordered residue
    # would be listed there, so this is what separates deletion from disorder
    if any(r in parse_remark465(lines, "A") for r in range(175, 180)):
        _fail("G4: 175-179 appear in REMARK 465 — reclassify as unmodelled")
    # ground truth 3: SEQRES is 228 long and DBREF claims P21926 1-228, which
    # LOOKS full length — the 5 deleted residues are exactly offset by a
    # 5-residue GSREF expression tag.  That coincidence is why this deletion
    # went unnoticed: neither length nor DBREF span reveals it.
    seqres = parse_seqres(lines, "A")
    if len(seqres) != 228 or seqres[:5] != ["GLY", "SER", "ARG", "GLU", "PHE"]:
        _fail(f"G4: 6K4J chain A SEQRES changed: len={len(seqres)} "
              f"head={seqres[:5]}")
    return ("G4 PASS  6K4J 175-179 TFTVK = construct_deleted (SEQADV DELETION, "
            "absent from REMARK 465); SEQRES 228 only matches P21926 because a "
            "5-residue GSREF tag offsets the 5 deleted residues")


def test_no_target_left_without_a_head():
    """G5: every target keeps >=1 usable candidate on its adopted template."""
    sys.path.insert(0, str(_CODE / "pipeline"))
    import ast
    src = (_CODE / "pipeline" / "config.py").read_text()
    tree = ast.parse(src)
    targets = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "TARGETS" for t in node.targets):
            targets = ast.literal_eval(node.value)
            break
    if targets is None:
        _fail("G5: could not read TARGETS from config.py")

    lines = []
    for tgt, (pdb, chain, _rng, acc) in ADOPTED.items():
        cands = targets[tgt].get("head_candidates") or [
            {"range": targets[tgt]["head_residues"], "name": "head_canonical"}]
        usable = [c for c in cands
                  if audit_head(pdb, chain, tuple(c["range"]), acc)["usable"]]
        if not usable:
            _fail(f"G5 {tgt}: all {len(cands)} head candidates disqualified on "
                  f"{pdb.name}. {tgt} would have no epitope and therefore no "
                  f"selectivity index.")
        lines.append(f"{tgt} {len(usable)}/{len(cands)}")
    return "G5 PASS  usable candidates per target: " + ", ".join(lines)


TESTS = [
    test_frozen_selection_survives,
    test_7jic_rejected_for_head,
    test_glycan_knockouts_rejected,
    test_construct_deletion_distinguished,
    test_no_target_left_without_a_head,
]


def main():
    failed = 0
    for t in TESTS:
        try:
            print(t())
        except AssertionError as e:
            print(f"FAIL  {e}")
            failed += 1
    print("ALL PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
