#!/usr/bin/env python3
"""Independent verification of the prepared CD templates.

Deliberately does NOT import prepare_cd_templates' build logic: it re-reads the
written PDB files and re-derives every claim from the file plus the UniProt
FASTA, so a bug in the builder cannot hide behind the builder's own report.

    conda activate MIPscreen
    python3 code/tools/check_cd_templates.py            # all checks, table + exit code
    python3 code/tools/check_cd_templates.py --json     # machine-readable

Checks
  C1 residue span, internal gaps, single chain, zero HETATM
  C2 every modelled residue matches the UniProt sequence  (this is what proves
     the reverted mutations really read as wild type, and that no expression-tag
     residue was imported under a colliding residue number)
  C3 disulfide SG-SG distances lie in 2.0-2.1 A with the configured PAIRING
  C4 backbone continuity: no i->i+1 C-N bond longer than 1.5 A
  C5 provenance channel: occupancy histogram, and the residues carrying
     predicted coordinates
  C6 membrane cut: EC2 residues below hydrophobic core + 6 A
  C7 head coverage of the RETIRED 16-mer heads -- reported, never enforced.
     The lab now imprints intact nanovesicles, so there is no peptide to order
     and no head to immobilise; this number exists only so the change is
     visible rather than silent.

Author: prepared-template session 2026-08-12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cd_pdbio import OCC_LABEL, read_pdb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "structures" / "prepared"
RAW = ROOT / "structures" / "raw"

UNIPROT = {"CD63": "P08962", "CD81": "P60033", "CD9": "P21926"}
EC2 = {"CD63": (103, 203), "CD81": (113, 201), "CD9": (112, 195)}
EC1 = {"CD63": (33, 51), "CD81": (34, 63), "CD9": (34, 55)}
SS = {"CD63": [(145, 191), (146, 169), (170, 177)],
      "CD81": [(156, 190), (157, 175)],
      "CD9": [(152, 181), (153, 167)]}

# The 16-mer heads results/phase1 froze, plus config.py's head_residues default.
# RETIRED: kept here as a reported number only.
RETIRED_HEADS = {
    "CD63": {"phase1_frozen": (157, 172), "config_default": (155, 170)},
    "CD81": {"phase1_frozen": (168, 183), "config_default": (168, 183)},
    "CD9":  {"phase1_frozen": (156, 171), "config_default": (156, 171)},
}

TEMPLATES = {
    "CD63_open_v1": "CD63",
    "CD81_open_v1": "CD81",
    "CD81_closed_5tcx_v1": "CD81",
    "CD9_open_v1": "CD9",
    "CD9_open_v1b": "CD9",
}


def uniprot_seq(acc: str) -> str:
    return "".join(l.strip() for l in open(RAW / "uniprot" / f"{acc}.fasta")
                   if not l.startswith(">"))


def check(name: str, target: str) -> dict:
    path = PREPARED / f"{name}.pdb"
    meta_path = PREPARED / f"{name}_provenance.json"
    if not path.is_file():
        return {"template": name, "FAIL": ["file missing"]}
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    m = read_pdb(path, "A", keep_hetatm=())
    fails, notes = [], {}

    # C1 ---------------------------------------------------------------
    chains, n_het = set(), 0
    for line in open(path):
        if line.startswith("ATOM"):
            chains.add(line[21:22])
        elif line.startswith("HETATM"):
            n_het += 1
    notes["span"] = list(m.span())
    notes["n_residues"] = len(m.res)
    notes["internal_gaps"] = m.gaps()
    notes["chains"] = sorted(chains)
    notes["n_hetatm"] = n_het
    if sorted(chains) != ["A"]:
        fails.append(f"C1 expected a single chain A, got {sorted(chains)}")
    if n_het:
        fails.append(f"C1 {n_het} HETATM records survived (binders/ligands must be stripped)")

    # C2 ---------------------------------------------------------------
    seq = uniprot_seq(UNIPROT[target])
    bad = [f"{i}{m.res[i].resname}" for i in m.nums()
           if not (1 <= i <= len(seq)) or m.res[i].one != seq[i - 1]]
    notes["sequence_mismatches_vs_uniprot"] = bad
    if bad:
        fails.append(f"C2 {len(bad)} residues differ from UniProt {UNIPROT[target]}: {bad[:12]}")

    # C3 ---------------------------------------------------------------
    dists = {}
    for a, c in SS[target]:
        ra, rc = m.res.get(a), m.res.get(c)
        if not (ra and rc and "SG" in ra.atoms and "SG" in rc.atoms):
            fails.append(f"C3 CYS{a}-CYS{c}: SG missing")
            dists[f"{a}-{c}"] = None
            continue
        d = float(np.linalg.norm(ra.atoms["SG"].xyz - rc.atoms["SG"].xyz))
        dists[f"{a}-{c}"] = round(d, 3)
        if not (2.0 <= d <= 2.1):
            fails.append(f"C3 CYS{a}-CYS{c} SG-SG = {d:.2f} A, outside 2.0-2.1")
    notes["ss_sg_sg"] = dists

    # C4 ---------------------------------------------------------------
    stretched = {}
    nums = m.nums()
    for a, c in zip(nums, nums[1:]):
        if c != a + 1:
            continue
        ra, rc = m.res[a], m.res[c]
        if "C" in ra.atoms and "N" in rc.atoms:
            d = float(np.linalg.norm(ra.atoms["C"].xyz - rc.atoms["N"].xyz))
            if d > 1.5:
                stretched[f"{a}-{c}"] = round(d, 2)
    notes["stretched_backbone_bonds"] = stretched
    if stretched:
        fails.append(f"C4 backbone C-N bonds longer than 1.5 A: {stretched}")

    # C5 ---------------------------------------------------------------
    hist, predicted = {}, []
    for i in m.nums():
        occs = {a.occ for a in m.res[i].atoms.values()}
        for a in m.res[i].atoms.values():
            k = f"{a.occ:.2f}"
            hist[k] = hist.get(k, 0) + 1
        if max(occs) < 0.75:
            predicted.append(i)
    notes["occupancy_histogram"] = {k: hist[k] for k in sorted(hist)}
    notes["occupancy_legend"] = {f"{k:.2f}": v for k, v in OCC_LABEL.items()}
    notes["fully_predicted_residues"] = predicted
    ec2 = EC2[target]
    notes["predicted_residues_inside_EC2"] = [i for i in predicted if ec2[0] <= i <= ec2[1]]

    # C6 ---------------------------------------------------------------
    half = meta.get("half_thickness")
    if half:
        occl = [i for i in m.nums() if ec2[0] <= i <= ec2[1]
                and "CA" in m.res[i].atoms and m.res[i].atoms["CA"].z < half + 6.0]
        modelled = [i for i in m.nums() if ec2[0] <= i <= ec2[1]]
        notes["membrane_half_thickness"] = half
        notes["ec2_modelled"] = len(modelled)
        notes["ec2_occluded"] = occl
        notes["ec2_scored_surface"] = [i for i in modelled if i not in occl]
        notes["ec2_occluded_fraction"] = round(len(occl) / max(len(modelled), 1), 3)
        if meta.get("occluded_ec2") is not None and meta["occluded_ec2"] != occl:
            fails.append("C6 occluded set disagrees with the builder's own report")

    # C7 ---------------------------------------------------------------
    heads = {}
    for label, (lo, hi) in RETIRED_HEADS[target].items():
        present = [i for i in range(lo, hi + 1) if i in m.res]
        exp = [i for i in present
               if max(a.occ for a in m.res[i].atoms.values()) >= 0.75]
        heads[label] = {
            "range": [lo, hi],
            "modelled": f"{len(present)}/{hi - lo + 1}",
            "experimental_backbone": f"{len(exp)}/{hi - lo + 1}",
            "sequence": "".join(m.res[i].one if i in m.res else "-"
                                for i in range(lo, hi + 1)),
        }
    notes["retired_16mer_head_coverage"] = heads

    notes["ec1_modelled"] = (f"{sum(1 for i in m.nums() if EC1[target][0] <= i <= EC1[target][1])}"
                             f"/{EC1[target][1] - EC1[target][0] + 1}  (EC1 is carried as "
                             f"steric context and is NOT scored, for all three targets)")
    return {"template": name, "target": target, "PASS": not fails,
            "FAIL": fails, **notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = [check(n, t) for n, t in TEMPLATES.items()]
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for r in out:
            print("=" * 74)
            print(f"{r['template']:24s} target={r.get('target','?')}  "
                  f"{'PASS' if r.get('PASS') else 'FAIL'}")
            if r.get("FAIL"):
                for f in r["FAIL"]:
                    print(f"   !! {f}")
            for k in ("span", "n_residues", "internal_gaps", "chains", "n_hetatm",
                      "sequence_mismatches_vs_uniprot", "ss_sg_sg",
                      "stretched_backbone_bonds", "occupancy_histogram",
                      "predicted_residues_inside_EC2", "membrane_half_thickness",
                      "ec2_modelled", "ec2_occluded", "ec2_occluded_fraction",
                      "ec1_modelled"):
                if k in r:
                    print(f"   {k:32s} {r[k]}")
            for label, h in r.get("retired_16mer_head_coverage", {}).items():
                print(f"   RETIRED head {label:16s} {h['range']} modelled "
                      f"{h['modelled']}, experimental {h['experimental_backbone']}  "
                      f"{h['sequence']}")
        print("=" * 74)
    return 0 if all(r.get("PASS") for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
