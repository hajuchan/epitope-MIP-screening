"""Content-addressed regression anchor for the PCSI* persistent-contact code.

WHY THIS FILE EXISTS
--------------------
The v1 anchor (results/reports/phase5_persistent_contacts_trial.json, "own 6 /
CD81 0 / CD9 3") is DEAD. It was written 2026-05-30 01:01:09; every one of the 88
md.xtc files on disk was written AFTER that date (oldest 2026-06-07, newest
2026-06-30). The analysis code never changed (one commit, ff83853, 2026-05-29,
byte-identical to today), and no (window, cutoff, persistence) cell in a ~19,000
cell grid search reproduces the pair own=6 & CD9=3 -- see code/anchor_forensics.py.
The anchor therefore describes trajectories that no longer exist. It could not
detect that, because it recorded only three integers and no provenance.

v2 fixes exactly that failure mode: the anchor is CONTENT-ADDRESSED. Every
reference count is stored next to the sha256, byte size and mtime of the md.xtc
and md.tpr it was computed from, plus the full parameter set and the code commit.
Verification is three staged gates, and a stale trajectory now fails at stage 1
with "the FILE changed" instead of at stage 2 with an ambiguous "the number
changed" that cannot distinguish a code bug from a re-run trajectory.

  stage 1  PROVENANCE   sha256 of each referenced file == the frozen sha256
  stage 2  COUNTS       recomputed (k, n, f) and the persistent-resid SET match
  stage 3  ABSENCE      arms frozen as absent are still absent, same evidence

sha256 is the authority, not mtime: a `touch` must not fail the gate and a
re-run that happens to land on the same size must not pass it.

USAGE
    python code/freeze_anchor.py --freeze     # (re)freeze from current disk
    python code/validate_pcsi_star.py anchor  # verify (the hard gate)
Both are read-only with respect to results/.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# PCSI_ANCHOR_PATH exists so the gate itself can be tested against deliberately
# corrupted anchors (code/test_anchor_gate.py) without ever touching the real one.
ANCHOR_PATH = Path(os.environ.get(
    "PCSI_ANCHOR_PATH", str(Path(__file__).resolve().parent / "utils_anchor_v2.json")))
SCHEMA = "pcsi_star_anchor/2"

# The cavity the anchor pins. Chosen because it carries the published headline.
ANCHOR_CAVITY, ANCHOR_MODE, ANCHOR_SNAPSHOT = "CD63", "dual", 0
ANCHOR_DIR = ROOT / "results/phase5/CD63/snapshot_0/dual_imprinting"
ANCHOR_LEGS = {"CD63": "rebind_own", "CD81": "rebind_CD81", "CD9": "rebind_CD9"}

# The v1 anchor, kept verbatim so the historical number is never quietly lost.
ANCHOR_V1 = {
    "source": "results/reports/phase5_persistent_contacts_trial.json",
    "counts": {"CD63": {"k": 6, "n": 101, "f": 0.0594059405940594},
               "CD81": {"k": 0, "n": 89, "f": 0.0},
               "CD9": {"k": 3, "n": 79, "f": 0.0379746835443038}},
    "status": "DEAD -- describes trajectories that no longer exist on disk",
}


# ------------------------------------------------------------- provenance ----

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def file_provenance(path, full_hash=True):
    """Identity of one file: size, mtime and (by default) full sha256."""
    path = Path(path)
    st = path.stat()
    rec = {
        "path": str(path.relative_to(ROOT)),
        "bytes": st.st_size,
        "mtime_epoch": int(st.st_mtime),
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
    }
    if full_hash:
        t0 = time.perf_counter()
        rec["sha256"] = sha256_file(path)
        rec["hash_wall_s"] = round(time.perf_counter() - t0, 2)
    return rec


def dir_absence_evidence(md_dir):
    """Why an arm has no trajectory, from disk evidence only.

    Distinguishes a pre-MD size exclusion (no em.gro) from a GROMACS crash
    (em.gro present, nvt.log present, nvt.xtc zero bytes). Imputing selectivity
    for a crash would manufacture a result out of a failure, so the anchor
    records which one it is and refuses to let it change silently.
    """
    md_dir = Path(md_dir)
    ev = {"md_dir": str(md_dir.relative_to(ROOT)), "exists": md_dir.is_dir()}
    if not ev["exists"]:
        ev["classification"] = "absent"
        return ev
    names = sorted(p.name for p in md_dir.iterdir())
    ev["has_md_xtc"] = "md.xtc" in names
    ev["has_md_tpr"] = "md.tpr" in names
    ev["has_em_gro"] = "em.gro" in names
    ev["has_nvt_log"] = "nvt.log" in names
    ev["zero_byte_files"] = sorted(
        n for n in names if (md_dir / n).is_file() and (md_dir / n).stat().st_size == 0)
    ev["n_files"] = len(names)
    nvt = md_dir / "nvt.xtc"
    ev["nvt_xtc_bytes"] = nvt.stat().st_size if nvt.is_file() else None
    if ev["has_md_xtc"] and ev["has_md_tpr"]:
        ev["classification"] = "present"
    elif ev["has_em_gro"] and ev["has_nvt_log"] and ev["nvt_xtc_bytes"] == 0:
        ev["classification"] = "md_failed_nvt_crash"
    elif not ev["has_em_gro"]:
        ev["classification"] = "size_excluded_or_never_started"
    else:
        ev["classification"] = "md_failed"
    return ev


def dependent_scan():
    """ADVISORY (never gates): where the dead v1 headline is still hardcoded.

    The published figure does not read the trajectories -- code/make_presentation
    _figures.py carries `"CD63_dual": 2.00` as a literal. Nothing about a PCSI*
    redesign touches that number, so it would survive the whole exercise unless
    someone is told, on every gate run, that it is still there.
    """
    hits = []
    for rel, needles in {
        "code/make_presentation_figures.py": ('"CD63_dual"', "2.00"),
        "code/make_process_figures.py": ('"CD63_dual"', "2.00"),
    }.items():
        p = ROOT / rel
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if all(n in line for n in needles):
                hits.append({"file": rel, "line": i, "source": line.strip()})
    return {
        "note": "advisory only -- these are hardcoded literals, not computed "
                "from the trajectories, and they still carry the dead v1 value",
        "hits": hits,
    }


def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30
                              ).stdout.strip() or None
    except Exception:
        return None


def _env_record():
    import numpy as np
    import MDAnalysis as mda
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "mdanalysis": mda.__version__, "platform": sys.platform}


# ------------------------------------------------------------------ freeze ----

def freeze(full_hash=True, verbose=True):
    """Recompute the anchor legs from CURRENT disk and return the v2 record.

    Reads exactly the 2 trajectories that exist under the anchor directory.
    Writes nothing -- the caller decides whether to persist it.
    """
    from pipeline.utils_pcsi_star import (analyze_leg, leg_summary, enable_readonly_mode,
                           CUTOFF_A, PERSISTENCE_FRAC, LAST_FRAC, PROTEIN_SEL,
                           MONOMER_SEL, contrast, legacy_pcsi)
    enable_readonly_mode()

    legs = {}
    for lab, sub in ANCHOR_LEGS.items():
        md = ANCHOR_DIR / sub / "md"
        xtc, tpr = md / "md.xtc", md / "md.tpr"
        if not (xtc.is_file() and tpr.is_file()):
            legs[lab] = {"role": "own" if lab == ANCHOR_CAVITY else "cross",
                         "expect": "absent",
                         "absence_evidence": dir_absence_evidence(md)}
            if verbose:
                print(f"  {lab:5s} ABSENT   {legs[lab]['absence_evidence']['classification']}")
            continue
        t0 = time.perf_counter()
        res = analyze_leg(xtc, tpr, cutoffs=(CUTOFF_A,))
        s = leg_summary(res, CUTOFF_A, PERSISTENCE_FRAC)
        legs[lab] = {
            "role": "own" if lab == ANCHOR_CAVITY else "cross",
            "expect": "present",
            "md_dir": str(md.relative_to(ROOT)),
            "provenance": {"md.xtc": file_provenance(xtc, full_hash),
                           "md.tpr": file_provenance(tpr, full_hash)},
            "counts": {"k": s["k"], "n": s["n"], "f": s["f"],
                       "mean_contact_freq": s["mean_contact_freq"]},
            "persistent_resids": s["persistent_resids"],
            "window": {k: res[k] for k in
                       ("n_frames", "start_frame", "n_analyzed", "dt_ps",
                        "t_first_ps", "t_last_ps", "window_ns")},
            "system": {k: res[k] for k in
                       ("n_protein_atoms", "n_protein_residues",
                        "n_monomer_atoms", "monomer_resnames", "n_atoms_total")},
            "wall_s": round(time.perf_counter() - t0, 2),
        }
        if verbose:
            print(f"  {lab:5s} k={s['k']:3d} n={s['n']:3d} f={s['f']:.6f}  "
                  f"sha256 {legs[lab]['provenance']['md.xtc'].get('sha256','-')[:16]}...  "
                  f"({legs[lab]['wall_s']}s)")

    present = {k: v for k, v in legs.items() if v["expect"] == "present"}
    own = present.get(ANCHOR_CAVITY)
    derived = {}
    if own:
        cross = {k: v for k, v in present.items() if k != ANCHOR_CAVITY}
        missing = [k for k, v in legs.items() if v["expect"] == "absent"]
        Ds = {k: contrast(own["counts"]["f"], v["counts"]["f"])
              for k, v in cross.items()}
        derived = {
            "arms_used": sorted(cross),
            "arms_missing_no_data": missing,
            "pcsi_star_per_arm": Ds,
            "pcsi_star_worst_case": min(Ds.values()) if Ds else None,
            "legacy_pcsi_counts": legacy_pcsi(
                own["counts"]["k"], [v["counts"]["k"] for v in cross.values()]),
            "caveat": (
                f"worst-case is over {len(cross)} of 2 off-target arms; "
                f"{missing} has no trajectory, so this is NOT a complete "
                f"selectivity statement and gate (e) must fail." if missing else None),
        }

    rec = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_by": "code/freeze_anchor.py --freeze",
        "git_commit": _git_commit(),
        "env": _env_record(),
        "cavity": ANCHOR_CAVITY, "mode": ANCHOR_MODE, "snapshot": ANCHOR_SNAPSHOT,
        "anchor_dir": str(ANCHOR_DIR.relative_to(ROOT)),
        "params": {
            "cutoff_A": CUTOFF_A, "persistence_frac": PERSISTENCE_FRAC,
            "last_frac": LAST_FRAC, "protein_sel": PROTEIN_SEL,
            "monomer_sel": MONOMER_SEL, "method": "capped",
            "predicate": "residue is in contact iff min over (all its atoms) x "
                         "(all monomer atoms) < cutoff_A, STRICT '<'",
            "persistence_predicate": "persistent iff contact_freq > "
                                     "persistence_frac, STRICT '>'",
            "window": "start = int(n_frames * (1 - last_frac)); denominator = "
                      "n_frames - start (the legacy off-by-one is preserved: "
                      "5001 frames gives 2501 analysed, i.e. 50.01%)",
            "selections_are_all_atom": True,
        },
        "legs": legs,
        "derived_headline": derived,
        "known_dependents_on_the_dead_v1_value": dependent_scan(),
        "superseded_anchor_v1": ANCHOR_V1,
        "verification": {
            "command": "python code/validate_pcsi_star.py anchor",
            "stages": ["provenance sha256", "recomputed counts + resid set",
                       "absent arms still absent"],
            "on_failure": "STOP. Do not launch the 88-leg run and do not edit "
                          "the reference to match the code. Determine whether "
                          "the trajectory changed (stage 1) or the code changed "
                          "(stage 2), then re-freeze deliberately.",
        },
    }
    return rec


def load():
    if not ANCHOR_PATH.is_file():
        return None
    return json.loads(ANCHOR_PATH.read_text())


def save(rec):
    ANCHOR_PATH.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    return ANCHOR_PATH


# ------------------------------------------------------------------ verify ----

def verify(full_hash=True, recompute=True, verbose=True):
    """Three-stage gate. Returns (ok: bool, report: dict). Never raises on a
    mismatch -- a mismatch is data, and the caller must be able to print it."""
    rec = load()
    if rec is None:
        return False, {"error": "NO ANCHOR", "anchor_path": str(ANCHOR_PATH),
                       "fix": "python code/freeze_anchor.py --freeze"}
    if rec.get("schema") != SCHEMA:
        return False, {"error": f"anchor schema {rec.get('schema')!r} != {SCHEMA!r}"}

    from pipeline.utils_pcsi_star import analyze_leg, leg_summary, enable_readonly_mode
    enable_readonly_mode()
    p = rec["params"]

    out = {"anchor_path": str(ANCHOR_PATH), "frozen_at": rec["frozen_at"],
           "anchor_git_commit": rec["git_commit"], "legs": {}, "stages": {}}
    prov_ok = counts_ok = absence_ok = True

    for lab, leg in rec["legs"].items():
        row = {"expect": leg["expect"]}
        if leg["expect"] == "absent":
            ev = dir_absence_evidence(ROOT / leg["absence_evidence"]["md_dir"])
            want = leg["absence_evidence"]
            same = (ev["classification"] == want["classification"]
                    and ev.get("has_md_xtc") == want.get("has_md_xtc"))
            row.update({"stage": "absence", "ok": same,
                        "frozen_classification": want["classification"],
                        "now_classification": ev["classification"],
                        "evidence": ev})
            absence_ok &= same
            out["legs"][lab] = row
            if verbose:
                print(f"  [{'PASS' if same else 'FAIL'}] {lab:5s} stage 3 ABSENCE: "
                      f"frozen {want['classification']!r}, now {ev['classification']!r}")
                if not same:
                    print("         a leg reappearing on disk is NEW DATA -- the "
                          "anchor must be re-frozen deliberately, not ignored.")
            continue

        # ---- stage 1: provenance
        md = ROOT / leg["md_dir"]
        prov_now, prov_rows, leg_prov_ok = {}, {}, True
        for fname, want in leg["provenance"].items():
            f = md / fname
            if not f.is_file():
                prov_rows[fname] = {"ok": False, "reason": "FILE MISSING"}
                leg_prov_ok = False
                continue
            got = file_provenance(f, full_hash)
            prov_now[fname] = got
            sha_ok = (not full_hash) or (got["sha256"] == want["sha256"])
            size_ok = got["bytes"] == want["bytes"]
            prov_rows[fname] = {
                "ok": bool(sha_ok and size_ok),
                "sha256_match": sha_ok if full_hash else None,
                "bytes_match": size_ok,
                "mtime_match": got["mtime_epoch"] == want["mtime_epoch"],
                "frozen": {k: want.get(k) for k in ("bytes", "mtime", "sha256")},
                "now": {k: got.get(k) for k in ("bytes", "mtime", "sha256")},
            }
            leg_prov_ok &= prov_rows[fname]["ok"]
        prov_ok &= leg_prov_ok
        row["provenance"] = prov_rows
        if verbose:
            for fname, r in prov_rows.items():
                extra = "" if r.get("mtime_match", True) else \
                    "  (mtime differs, content identical -- a touch, not a re-run)"
                print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {lab:5s} stage 1 "
                      f"PROVENANCE {fname}: sha256={r.get('sha256_match')} "
                      f"bytes={r.get('bytes_match')}{extra}")
                if not r["ok"]:
                    print(f"         frozen {r.get('frozen')}")
                    print(f"         now    {r.get('now')}")
                    print("         THE TRAJECTORY CHANGED. This is the v1 failure "
                          "mode. Do not adjust the anchor to match; find out why.")

        # ---- stage 2: counts
        if recompute and leg_prov_ok:
            t0 = time.perf_counter()
            res = analyze_leg(md / "md.xtc", md / "md.tpr", cutoffs=(p["cutoff_A"],),
                              last_frac=p["last_frac"])
            s = leg_summary(res, p["cutoff_A"], p["persistence_frac"])
            want = leg["counts"]
            same = (s["k"] == want["k"] and s["n"] == want["n"]
                    and abs(s["f"] - want["f"]) < 1e-12
                    and list(s["persistent_resids"]) == list(leg["persistent_resids"]))
            counts_ok &= same
            row["counts"] = {"ok": same, "frozen": want,
                             "now": {"k": s["k"], "n": s["n"], "f": s["f"]},
                             "resids_match":
                                 list(s["persistent_resids"]) == list(leg["persistent_resids"]),
                             "now_resids": s["persistent_resids"],
                             "wall_s": round(time.perf_counter() - t0, 2)}
            if verbose:
                print(f"  [{'PASS' if same else 'FAIL'}] {lab:5s} stage 2 COUNTS: "
                      f"k={s['k']} n={s['n']} f={s['f']:.6f} "
                      f"(frozen k={want['k']} n={want['n']} f={want['f']:.6f}) "
                      f"resid-set={'match' if row['counts']['resids_match'] else 'DIFFER'} "
                      f"({row['counts']['wall_s']}s)")
                if not same:
                    print("         Provenance passed, so the FILES are identical: "
                          "the CODE changed. Fix the code, do not re-freeze.")
        elif not leg_prov_ok:
            row["counts"] = {"ok": False, "skipped": "provenance failed first"}
            counts_ok = False
        out["legs"][lab] = row

    out["stages"] = {"1_provenance": prov_ok, "2_counts": counts_ok,
                     "3_absence": absence_ok}
    ok = bool(prov_ok and counts_ok and absence_ok)
    out["anchor_verified"] = ok
    out["v1_anchor"] = dict(ANCHOR_V1, note=(
        "HISTORICAL ONLY. The published 'CD63 dual-imprinting PCSI 2.00 STRONG' "
        "rests on these counts and they do not come back from the trajectories "
        "now on disk. See derived_headline in the anchor for what the same "
        "cavity gives today."))
    out["derived_headline"] = rec.get("derived_headline")
    out["dependents_advisory"] = dependent_scan()
    if verbose and out["dependents_advisory"]["hits"]:
        print("\n  ADVISORY (does not gate): the dead v1 headline is still a "
              "hardcoded literal in")
        for h in out["dependents_advisory"]["hits"]:
            print(f"    {h['file']}:{h['line']}  {h['source']}")
        print("  Those figures are drawn from constants, not from the "
              "trajectories, so no recompute will ever correct them.")
    return ok, out
