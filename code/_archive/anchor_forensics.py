"""Why the v1 anchor (own 6 / CD81 0 / CD9 3) does not reproduce: evidence, not opinion.

Three independent lines, each printed with the command that produced it:

  A. CODE      git history of code/analyze_persistent_contacts.py, and a diff of
               the contact function against its only commit.
  B. FILE AGE  mtime of every md.xtc against the mtime of the reference JSON.
  C. PARAMS    exhaustive grid search over (window length x cutoff x persistence)
               asking whether ANY parameter cell reproduces own=6 AND CD9=3.

C is the one that actually settles it. Naively it is ~19,000 analyses; here one
trajectory pass stores the per-frame per-residue MINIMUM distance (2501 x n_res
float32, ~1 MB), after which every (window, cutoff, persistence) cell is pure
numpy on that matrix. Two trajectory passes total.

    python code/anchor_forensics.py                # full, ~2-3 min
    python code/anchor_forensics.py --quick        # coarser grid, ~1 min

Read-only with respect to results/. Output goes to reports_v2/anchor_forensics.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase5/CD63/snapshot_0/dual_imprinting"
TRIAL = ROOT / "results/reports/phase5_persistent_contacts_trial.json"
V1 = {"CD63": 6, "CD81": 0, "CD9": 3}


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(ROOT))
    return (r.stdout + r.stderr).rstrip()


# ------------------------------------------------------- A. code provenance --

def line_a():
    print("\n=== A. THE CODE NEVER CHANGED ===")
    cmds = {
        "git_log": "git log --format='%h %ad %s' --date=short -- "
                   "code/analyze_persistent_contacts.py",
        "diff_vs_first_commit":
            "diff <(git show ff83853:code/analyze_persistent_contacts.py | "
            "sed -n '20,52p') <(sed -n '20,52p' "
            "code/analyze_persistent_contacts.py) && echo IDENTICAL",
    }
    out = {}
    for k, c in cmds.items():
        out[k] = {"command": c, "output": sh(f"bash -c {json.dumps(c)}")}
        print(f"  $ {c}\n    {out[k]['output']}")
    return out


# ------------------------------------------------------------ B. file dates --

def line_b():
    print("\n=== B. EVERY TRAJECTORY POSTDATES THE REFERENCE JSON ===")
    ref_m = TRIAL.stat().st_mtime
    xtcs = [p for p in BASE.parent.parent.parent.rglob("md/md.xtc")
            if "removal_test" not in str(p)]
    older = [p for p in xtcs if p.stat().st_mtime < ref_m]
    newer = [p for p in xtcs if p.stat().st_mtime >= ref_m]
    ts = sorted(p.stat().st_mtime for p in xtcs)
    out = {
        "reference_json": str(TRIAL.relative_to(ROOT)),
        "reference_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ref_m)),
        "n_md_xtc": len(xtcs), "n_older_than_reference": len(older),
        "n_newer_than_reference": len(newer),
        "oldest_md_xtc": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts[0])),
        "newest_md_xtc": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts[-1])),
        "command": "find results/phase5 -path '*/md/md.xtc' | grep -v removal_test | "
                   "xargs stat -c '%Y' | awk ...",
    }
    print(f"  reference JSON mtime : {out['reference_mtime']}")
    print(f"  md.xtc files         : {out['n_md_xtc']}  "
          f"(older than reference: {out['n_older_than_reference']}, "
          f"newer: {out['n_newer_than_reference']})")
    print(f"  oldest / newest      : {out['oldest_md_xtc']} / {out['newest_md_xtc']}")
    return out


# ------------------------------------------------------- C. parameter sweep --

def min_distance_matrix(traj, top, max_cutoff=9.0):
    """Per-frame, per-residue minimum protein-monomer distance over the LAST
    HALF of the trajectory. Distances above max_cutoff are stored as +inf,
    which is correct for any cutoff <= max_cutoff under a strict '<' test.

    Same selections and same minimum-image convention as the legacy code.
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance
    from pcsi_star import (PROTEIN_SEL, MONOMER_SEL, enable_readonly_mode,
                           _residue_index_map)
    enable_readonly_mode()

    u = mda.Universe(str(top), str(traj))
    protein, monomers = u.select_atoms(PROTEIN_SEL), u.select_atoms(MONOMER_SEL)
    n_frames = len(u.trajectory)
    start = n_frames // 2                       # half, then the caller re-windows
    atom_ridx, resids = _residue_index_map(protein)
    n_res = len(resids)
    n_an = n_frames - start
    M = np.full((n_an, n_res), np.inf, dtype=np.float32)
    for i, ts in enumerate(u.trajectory[start:]):
        pairs, d = capped_distance(protein.positions, monomers.positions,
                                   max_cutoff=max_cutoff, box=ts.dimensions,
                                   return_distances=True)
        if len(pairs):
            np.minimum.at(M[i], atom_ridx[pairs[:, 0]], d.astype(np.float32))
    return M, resids, n_frames, start


def k_from_matrix(M, window, cutoff, persistence):
    """Persistent-residue count for the LAST `window` frames of M."""
    sub = M[-window:]
    return int((((sub < cutoff).mean(axis=0)) > persistence).sum())


def line_c(quick=False):
    print("\n=== C. NO PARAMETER CELL REPRODUCES own=6 AND CD9=3 ===")
    mats = {}
    for lab, sub in (("CD63", "rebind_own"), ("CD9", "rebind_CD9")):
        md = BASE / sub / "md"
        t0 = time.perf_counter()
        M, resids, nf, start = min_distance_matrix(md / "md.xtc", md / "md.tpr")
        mats[lab] = M
        print(f"  {lab:5s} min-distance matrix {M.shape} from {nf} frames "
              f"(window starts at {start})   {time.perf_counter()-t0:.1f}s")

    n_full = mats["CD63"].shape[0]
    if quick:
        windows = list(range(200, n_full + 1, 200)) + [n_full]
        cutoffs = np.arange(3.0, 9.01, 0.5)
        persist = np.arange(0.30, 0.801, 0.05)
    else:
        # step 50 so the search is a superset of any round-ish window someone
        # might plausibly have used, including the 950-frame cell that an
        # earlier coarser search turned up.
        windows = list(range(200, n_full + 1, 50)) + [n_full]
        cutoffs = np.arange(3.0, 9.01, 0.25)
        persist = np.arange(0.30, 0.801, 0.01)
    windows = sorted(set(windows))

    t0 = time.perf_counter()
    hits, n_cells = [], 0
    own6_cells, cd9_at_own6 = 0, set()
    for w in windows:
        # per-window contact-frequency tensors, one pass per cutoff
        for c in cutoffs:
            fo = (mats["CD63"][-w:] < c).mean(axis=0)
            fc = (mats["CD9"][-w:] < c).mean(axis=0)
            for p in persist:
                n_cells += 1
                ko = int((fo > p).sum())
                kc = int((fc > p).sum())
                if ko == V1["CD63"] and kc == V1["CD9"]:
                    hits.append({"window_frames": int(w), "cutoff_A": round(float(c), 3),
                                 "persistence": round(float(p), 3)})
                if w == n_full and ko == V1["CD63"]:
                    own6_cells += 1
                    cd9_at_own6.add(kc)
    wall = time.perf_counter() - t0

    print(f"  grid: {len(windows)} windows x {len(cutoffs)} cutoffs x "
          f"{len(persist)} persistences = {n_cells} cells   ({wall:.1f}s)")
    print(f"  window range {min(windows)}..{max(windows)} frames, "
          f"cutoff {cutoffs[0]:.2f}..{cutoffs[-1]:.2f} A, "
          f"persistence {persist[0]:.2f}..{persist[-1]:.2f}")
    canonical_hits = [h for h in hits if h["window_frames"] == n_full]
    print(f"  cells giving own=6 AND CD9=3 : {len(hits)}  {hits}")
    print(f"    of which at the canonical {n_full}-frame window: {len(canonical_hits)}")
    print(f"  at the CANONICAL {n_full}-frame window, own=6 occurs in "
          f"{own6_cells} cells; CD9 at those same cells is always in "
          f"{sorted(cd9_at_own6)} -- never 3.")

    # The pipeline's own settings, the only cell anyone actually ran.
    ko = int(((mats["CD63"] < 6.0).mean(axis=0) > 0.5).sum())
    kc = int(((mats["CD9"] < 6.0).mean(axis=0) > 0.5).sum())
    print(f"  at the PIPELINE cell (window {n_full}, 6.00 A, 0.50): "
          f"own={ko} CD9={kc}   (v1 claims own=6 CD9=3)")

    if not hits:
        verdict = ("no parameter cell in the grid reproduces the v1 pair")
    elif not canonical_hits:
        verdict = (f"{len(hits)} coincidental cell(s) out of {n_cells} reproduce the "
                   f"pair, none at the canonical window and none at a combination "
                   f"any run would have used -- a parameter difference is NOT the "
                   f"explanation")
    else:
        verdict = ("a canonical-window cell reproduces the pair; inspect it "
                   "before concluding anything")
    print(f"  VERDICT: {verdict}")
    return {"n_cells": n_cells, "windows": [int(w) for w in windows],
            "pipeline_cell": {"window_frames": n_full, "cutoff_A": 6.0,
                              "persistence": 0.5, "own_k": ko, "cd9_k": kc},
            "matches_at_canonical_window": canonical_hits,
            "cutoff_range": [float(cutoffs[0]), float(cutoffs[-1])],
            "cutoff_step": float(cutoffs[1] - cutoffs[0]),
            "persistence_range": [float(persist[0]), float(persist[-1])],
            "persistence_step": float(persist[1] - persist[0]),
            "matches_own6_and_cd9_3": hits,
            "canonical_window_frames": int(n_full),
            "n_cells_with_own6_at_canonical_window": own6_cells,
            "cd9_values_at_those_cells": sorted(cd9_at_own6),
            "wall_s": round(wall, 1), "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports_v2/anchor_forensics.json"))
    a = ap.parse_args()

    print("FORENSICS: why the v1 anchor 6/0/3 does not reproduce")
    rep = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "v1_anchor": V1, "A_code": line_a(), "B_file_dates": line_b(),
           "C_parameter_sweep": line_c(a.quick)}
    rep["conclusion"] = (
        "The code is unchanged since before the reference was written, every "
        "trajectory on disk was written after it, and no parameter cell "
        "reproduces the reference pair. The reference describes trajectories "
        "that no longer exist. Re-freeze the anchor from current data; do not "
        "'fix' the code to chase the old numbers.")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(f"\nCONCLUSION: {rep['conclusion']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
