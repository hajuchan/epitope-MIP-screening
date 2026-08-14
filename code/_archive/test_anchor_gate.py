"""Negative tests for the anchor gate: prove it FAILS when it should.

A gate that has only ever been seen to pass is not evidence of anything. The v1
anchor "passed" for weeks in the sense that nobody ran it; when it finally ran it
could not say WHY it failed. These tests corrupt a throwaway copy of the anchor
one field at a time and assert that the right stage fails with the right message.

  python code/test_anchor_gate.py            # all scenarios (~40 s, 2 trajectories)
  python code/test_anchor_gate.py --fast     # skip the stage-2 recompute scenarios

The real code/anchor_v2.json is never written to: every scenario runs against a
temp copy selected with PCSI_ANCHOR_PATH. results/ is read-only throughout.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
REAL = Path(__file__).resolve().parent / "anchor_v2.json"
FAILS = []


def ck(name, cond, note=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")
    if not cond:
        FAILS.append(name)
    return cond


def run_verify(rec, tmpdir, recompute=True):
    """Verify a (possibly corrupted) anchor record in an isolated process-local
    module state. Returns (ok, report)."""
    p = Path(tmpdir) / "anchor_tamper.json"
    p.write_text(json.dumps(rec, indent=2, default=str))
    os.environ["PCSI_ANCHOR_PATH"] = str(p)
    import importlib
    import anchor as A
    importlib.reload(A)
    try:
        return A.verify(full_hash=True, recompute=recompute, verbose=False)
    finally:
        os.environ.pop("PCSI_ANCHOR_PATH", None)
        importlib.reload(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    if not REAL.is_file():
        print(f"no anchor at {REAL}; run python code/freeze_anchor.py --freeze")
        return 1
    base = json.loads(REAL.read_text())
    before = REAL.read_bytes()

    with tempfile.TemporaryDirectory() as td:
        print("\n=== S0 clean anchor verifies ===")
        ok, rep = run_verify(base, td)
        ck("S0 ok", ok)
        ck("S0 all three stages true", all(rep["stages"].values()), str(rep["stages"]))

        print("\n=== S1 trajectory changed (sha256 differs, size identical) ===")
        print("    THE v1 FAILURE MODE. v1 could not see this at all; it stored no")
        print("    provenance, so a re-run trajectory was indistinguishable from a")
        print("    code bug. Stage 1 must fail and stage 2 must not even be trusted.")
        t = copy.deepcopy(base)
        t["legs"]["CD63"]["provenance"]["md.xtc"]["sha256"] = "0" * 64
        ok, rep = run_verify(t, td)
        ck("S1 gate fails", not ok)
        ck("S1 stage 1 provenance FALSE", rep["stages"]["1_provenance"] is False)
        ck("S1 stage 2 not credited", rep["stages"]["2_counts"] is False)
        ck("S1 counts skipped, not silently recomputed",
           rep["legs"]["CD63"]["counts"].get("skipped") is not None,
           str(rep["legs"]["CD63"]["counts"].get("skipped")))
        ck("S1 names the file",
           rep["legs"]["CD63"]["provenance"]["md.xtc"]["sha256_match"] is False)

        print("\n=== S1b size changed too (a genuinely different file) ===")
        t = copy.deepcopy(base)
        t["legs"]["CD9"]["provenance"]["md.xtc"]["bytes"] = 123
        t["legs"]["CD9"]["provenance"]["md.xtc"]["sha256"] = "1" * 64
        ok, rep = run_verify(t, td)
        ck("S1b gate fails", not ok)
        ck("S1b bytes mismatch reported",
           rep["legs"]["CD9"]["provenance"]["md.xtc"]["bytes_match"] is False)

        print("\n=== S1c mtime differs but content identical (a touch, not a re-run) ===")
        print("    Must PASS: sha256 is the authority, mtime is only reported.")
        t = copy.deepcopy(base)
        t["legs"]["CD63"]["provenance"]["md.xtc"]["mtime_epoch"] = 1
        t["legs"]["CD63"]["provenance"]["md.xtc"]["mtime"] = "1970-01-01T00:00:01"
        ok, rep = run_verify(t, td, recompute=not a.fast)
        ck("S1c gate still passes", ok, str(rep["stages"]))
        ck("S1c mtime drift is reported, not fatal",
           rep["legs"]["CD63"]["provenance"]["md.xtc"]["mtime_match"] is False)

        if not a.fast:
            print("\n=== S2 code changed (files identical, counts differ) ===")
            print("    Stage 1 passes, so the files are provably the same bytes:")
            print("    the only remaining explanation is the code. Must NOT re-freeze.")
            t = copy.deepcopy(base)
            t["legs"]["CD63"]["counts"]["k"] = 6          # the dead v1 value
            t["legs"]["CD63"]["counts"]["f"] = 6 / 101
            ok, rep = run_verify(t, td)
            ck("S2 gate fails", not ok)
            ck("S2 stage 1 provenance TRUE", rep["stages"]["1_provenance"] is True)
            ck("S2 stage 2 counts FALSE", rep["stages"]["2_counts"] is False)
            ck("S2 recomputed value is the live one",
               rep["legs"]["CD63"]["counts"]["now"]["k"] == 7)

            print("\n=== S2b same count, different residue SET ===")
            print("    k alone is a weak fingerprint: 7 residues out of 101 can be")
            print("    the wrong 7. The identity of the set is part of the anchor.")
            t = copy.deepcopy(base)
            t["legs"]["CD63"]["persistent_resids"] = [1, 2, 3, 4, 5, 6, 7]
            ok, rep = run_verify(t, td)
            ck("S2b gate fails on the resid set alone", not ok)
            ck("S2b k still matches", rep["legs"]["CD63"]["counts"]["now"]["k"] == 7)
            ck("S2b resids_match False",
               rep["legs"]["CD63"]["counts"]["resids_match"] is False)

        print("\n=== S3 an absent leg reappears / changes character ===")
        t = copy.deepcopy(base)
        t["legs"]["CD81"]["absence_evidence"]["classification"] = "size_excluded"
        ok, rep = run_verify(t, td, recompute=not a.fast)
        ck("S3 gate fails", not ok)
        ck("S3 stage 3 absence FALSE", rep["stages"]["3_absence"] is False)
        ck("S3 reports both classifications",
           rep["legs"]["CD81"]["frozen_classification"] == "size_excluded"
           and rep["legs"]["CD81"]["now_classification"] == "md_failed_nvt_crash")
        print("    (this matters: a crash imputed as size exclusion would score +1,")
        print("     i.e. perfect selectivity manufactured out of a GROMACS failure)")

        print("\n=== S4 no anchor at all ===")
        os.environ["PCSI_ANCHOR_PATH"] = str(Path(td) / "does_not_exist.json")
        import importlib
        import anchor as A
        importlib.reload(A)
        ok, rep = A.verify()
        os.environ.pop("PCSI_ANCHOR_PATH", None)
        importlib.reload(A)
        ck("S4 gate fails when the anchor is missing", not ok)
        ck("S4 says how to fix it", "freeze" in rep.get("fix", ""))

        print("\n=== S5 the DRIVER refuses to launch on a failing anchor ===")
        t = copy.deepcopy(base)
        t["legs"]["CD63"]["provenance"]["md.xtc"]["sha256"] = "0" * 64
        p = Path(td) / "anchor_driver_test.json"
        p.write_text(json.dumps(t, indent=2, default=str))
        env = dict(os.environ, PCSI_ANCHOR_PATH=str(p))
        r = subprocess.run(
            [sys.executable, str(ROOT / "code/run_pcsi_star.py"),
             "--only", "CD63|dual|0|", "--workers", "1",
             "--out-dir", str(Path(td) / "out")],
            capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=900)
        ck("S5 driver exit code 3", r.returncode == 3, f"(got {r.returncode})")
        ck("S5 driver says it refuses", "refusing to launch" in r.stdout)
        ck("S5 driver points at the forensics",
           "anchor_forensics" in r.stdout)
        ck("S5 driver wrote no summary",
           not (Path(td) / "out/pcsi_star_summary.json").exists())

        print("\n=== S6 --skip-anchor-gate is loud and self-recording ===")
        r = subprocess.run(
            [sys.executable, str(ROOT / "code/run_pcsi_star.py"),
             "--only", "NO_SUCH_LEG", "--workers", "1", "--skip-anchor-gate",
             "--no-sweep", "--out-dir", str(Path(td) / "out2")],
            capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=900)
        ck("S6 driver runs", r.returncode == 0, f"(got {r.returncode})")
        ck("S6 prints UNANCHORED warning", "UNANCHORED" in r.stdout)
        s = json.loads((Path(td) / "out2/pcsi_star_summary.json").read_text())
        ck("S6 records the skip in metadata",
           s["meta"]["anchor_gate"].get("reason") == "--skip-anchor-gate",
           str(s["meta"]["anchor_gate"]))
        ck("S6 the human report carries the UNANCHORED label",
           "UNANCHORED" in (Path(td) / "out2/pcsi_star_report.txt").read_text())

    ck("the real anchor file was never modified", REAL.read_bytes() == before)

    print("\n" + "=" * 70)
    if FAILS:
        print(f"ANCHOR GATE TESTS: {len(FAILS)} FAILURE(S): {FAILS}")
        return 1
    print("ANCHOR GATE TESTS: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
