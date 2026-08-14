"""Freeze / inspect the content-addressed PCSI* regression anchor (code/anchor.py).

    python code/freeze_anchor.py --freeze          # write code/anchor_v2.json
    python code/freeze_anchor.py --freeze --force  # overwrite an existing anchor
    python code/freeze_anchor.py --check           # three-stage verification
    python code/freeze_anchor.py --show            # print the frozen record
    python code/freeze_anchor.py --headline        # what the published figure is
                                                   #   worth on CURRENT data

Read-only with respect to results/. The only file written is code/anchor_v2.json.
Freezing is deliberately manual and never happens as a side effect of a run:
re-freezing is how a stale reference gets blessed, so it must be a human act.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anchor as A


def print_headline(rec):
    d = rec.get("derived_headline") or {}
    v1 = rec["superseded_anchor_v1"]["counts"]
    print("\n=== THE PUBLISHED HEADLINE, RECOMPUTED ON CURRENT TRAJECTORIES ===")
    print("    published : CD63 dual-imprinting PCSI 2.00 STRONG")
    print(f"                from own k={v1['CD63']['k']}, CD81 k={v1['CD81']['k']}, "
          f"CD9 k={v1['CD9']['k']}  ->  6 / max(0,3) = 2.00")
    print("                (those counts are NOT reproducible from any trajectory")
    print("                 now on disk -- see code/anchor_forensics.py)")
    rows = []
    for lab, leg in rec["legs"].items():
        if leg["expect"] == "present":
            c = leg["counts"]
            rows.append(f"{lab} k={c['k']} n={c['n']} f={c['f']:.6f}")
        else:
            rows.append(f"{lab} NO TRAJECTORY "
                        f"({leg['absence_evidence']['classification']})")
    print("    today     : " + " | ".join(rows))
    if d:
        print(f"                legacy PCSI (counts)  = {d['legacy_pcsi_counts']}")
        for arm, v in d["pcsi_star_per_arm"].items():
            print(f"                PCSI* vs {arm:5s}        = {v:+.4f}")
        print(f"                PCSI* worst case      = "
              f"{d['pcsi_star_worst_case']:+.4f}")
        if d.get("caveat"):
            print(f"    CAVEAT    : {d['caveat']}")
    print("    CONSEQUENCE: the sign flips. On current data this cavity is")
    print("    cross-reactive, not selective, and the published figure must be")
    print("    revisited independently of the PCSI* redesign.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--show", action="store_true")
    g.add_argument("--headline", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="allow --freeze to overwrite an existing anchor")
    ap.add_argument("--no-hash", action="store_true",
                    help="size+mtime only; NOT sufficient for a real freeze")
    a = ap.parse_args()

    if a.show or a.headline:
        rec = A.load()
        if rec is None:
            print(f"NO ANCHOR at {A.ANCHOR_PATH}")
            return 1
        if a.show:
            print(json.dumps(rec, indent=2))
        else:
            print_headline(rec)
        return 0

    if a.check:
        ok, rep = A.verify(full_hash=not a.no_hash)
        print(json.dumps(rep["stages"], indent=2) if "stages" in rep
              else json.dumps(rep, indent=2))
        print("ANCHOR VERIFIED" if ok else "ANCHOR FAILED")
        return 0 if ok else 1

    # --freeze
    old = A.load()
    if old is not None and not a.force:
        print(f"An anchor already exists at {A.ANCHOR_PATH}")
        print(f"  frozen_at {old['frozen_at']}  commit {old['git_commit']}")
        print("Re-freezing BLESSES whatever is on disk today as the new reference.")
        print("If the current anchor is failing, first establish WHY (stage 1 =")
        print("the trajectory changed, stage 2 = the code changed). Only stage-1")
        print("failures are ever legitimately re-frozen. Re-run with --force.")
        return 2

    print(f"Freezing anchor from {A.ANCHOR_DIR.relative_to(A.ROOT)}")
    print("  (reads 2 trajectories; sha256 over ~3.2 GB)")
    rec = A.freeze(full_hash=not a.no_hash)
    if a.no_hash:
        rec["status"] = "FROZEN_WITHOUT_HASH"
        print("\nWARNING: frozen without sha256. This anchor cannot detect a "
              "re-run trajectory that preserves size, which is the exact failure "
              "that killed v1. Re-freeze properly before relying on it.")
    path = A.save(rec)
    print_headline(rec)
    print(f"\nwrote {path}")

    ok, rep = A.verify(full_hash=not a.no_hash, verbose=True)
    print(f"\nself-verification: {'PASS' if ok else 'FAIL'}  {rep.get('stages')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
