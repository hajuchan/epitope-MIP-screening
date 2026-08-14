"""Validation for pcsi_star.py. Four independent checks, each exits non-zero on failure.

  python code/validate_pcsi_star.py edge         # synthetic edge cases, no MD
  python code/validate_pcsi_star.py anchor       # content-addressed anchor gate
                                                 #   (code/anchor_v2.json)
  python code/validate_pcsi_star.py equiv  <md_dir> [--frames N]   # loop vs vectorised
  python code/validate_pcsi_star.py timing <md_dir>                # one leg, project 88

`edge` touches no trajectory. `anchor`, `equiv` and `timing` read from results/ only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FAILURES = []


def check(name, got, want, note=""):
    if isinstance(want, float) and math.isfinite(want):
        ok = got is not None and isinstance(got, (int, float)) \
            and math.isfinite(got) and abs(got - want) < 1e-12
    else:
        ok = got == want          # exact: covers inf, None, str, int, bool, list
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r} {note}")
    if not ok:
        FAILURES.append(name)
    return ok


def check_true(name, cond, note=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")
    if not cond:
        FAILURES.append(name)
    return cond


# ============================================================ 1. EDGE CASES ====

def cmd_edge(_a):
    from pcsi_star import (contrast, snapshot_value, bootstrap_ci,
                           aggregate_cavity, evaluate_gate, power_requirement,
                           legacy_pcsi, legacy_summary)

    print("\n=== EDGE CASE E1: f_own > 0, f_cross = 0 (size exclusion / non-binding) ===")
    print("    spec: D = +1 exactly. No special case, no infinity, no clamping.")
    check("E1 D(0.06, 0.0)", contrast(0.0594, 0.0), 1.0)
    check("E1 D(1/101, 0)", contrast(1 / 101, 0.0), 1.0)
    check("E1 D(40/101, 0)", contrast(40 / 101, 0.0), 1.0)
    print("    CAVEAT (spec): +1 is attained by ANY a>0 — 1/101 vs 0 scores exactly")
    print("    the same as 40/101 vs 0. PCSI* measures CONTRAST, not affinity;")
    print("    mitigated by reporting raw k_L and by the separate binding gate (c).")

    print("\n=== EDGE CASE E2: f_own = 0, f_cross > 0 (binds the wrong protein) ===")
    print("    spec: D = -1 exactly. Scored, kept, NEVER excluded.")
    check("E2 D(0, 0.112)", contrast(0.0, 0.11235955056179775), -1.0)
    r = snapshot_value(0.0, {"CD81": {"f": 10 / 89, "status": "ok"},
                             "CD9": {"f": 27 / 79, "status": "ok"}},
                       ["CD81", "CD9"])
    check("E2 real case CD63_silane_predual S", r["S"], -1.0,
          "(own=0/101, CD81=10/89, CD9=27/79)")
    check("E2 status", r["status"], "defined")

    print("\n=== EDGE CASE E3: f_own = 0 AND every f_cross = 0 (nothing bound) ===")
    print("    spec: 0/0 UNDEFINED. Do NOT score 0. Classify no_binding, EXCLUDE")
    print("    from theta_hat and from the bootstrap pool, count it, gate on (c).")
    check("E3 contrast(0,0) is None (never 0.0)", contrast(0.0, 0.0), None)
    r = snapshot_value(0.0, {"CD81": {"f": 0.0, "status": "ok"},
                             "CD9": {"f": 0.0, "status": "ok"}}, ["CD81", "CD9"])
    check("E3 S", r["S"], None)
    check("E3 status", r["status"], "no_binding")
    print("    sub-case: f_own = 0 but SOME f_cross > 0 -> snapshot IS defined,")
    print("    only the 0/0 pair is dropped, min over the rest is -1.")
    r = snapshot_value(0.0, {"CD81": {"f": 0.0, "status": "ok"},
                             "CD9": {"f": 0.05, "status": "ok"}}, ["CD81", "CD9"])
    check("E3b status", r["status"], "defined")
    check("E3b S", r["S"], -1.0)
    check("E3b n_cross_used (0/0 pair dropped)", r["n_cross_used"], 1)

    print("\n=== EDGE CASE E4: leg absent, genuinely SIZE_EXCLUDED before MD ran ===")
    print("    spec: impute D = +1, flag imputed, report n_imputed, and report")
    print("    theta_hat BOTH with and without the imputed legs (zero variance).")
    r = snapshot_value(0.10, {"CD63": {"f": None, "status": "size_excluded"},
                              "CD81": {"f": 0.05, "status": "ok"}},
                       ["CD63", "CD81"])
    check("E4 D(size-excluded arm)", r["cross"]["CD63"]["D"], 1.0)
    check("E4 imputed flag", r["cross"]["CD63"]["imputed"], True)
    check("E4 n_imputed", r["n_imputed"], 1)
    check("E4 S = min, so the imputed +1 does NOT carry it",
          round(r["S"], 12), round((0.10 - 0.05) / 0.15, 12))

    print("\n=== EDGE CASE E5: leg absent for ANY other reason (NVT crash) ===")
    print("    spec: genuinely missing. Do NOT impute, do NOT score. Drop the pair,")
    print("    record n_missing, keep J_C homogeneous.")
    r = snapshot_value(0.10, {"CD81": {"f": None, "status": "md_failed"},
                              "CD9": {"f": 0.20, "status": "ok"}},
                       ["CD81", "CD9"])
    check("E5 D(crashed arm) is None (NOT +1)", r["cross"]["CD81"]["D"], None)
    check("E5 n_missing", r["n_missing"], 1)
    check("E5 n_cross_used", r["n_cross_used"], 1)
    check("E5 S from the surviving arm only",
          round(r["S"], 12), round((0.10 - 0.20) / 0.30, 12))
    check_true("E5 a crashed arm is NEVER imputed +1",
               r["cross"]["CD81"]["D"] != 1.0)

    print("\n    J_C homogeneity: PRIMARY uses only arms present in EVERY snapshot.")
    snaps = {}
    for s in range(6):
        snaps[s] = {"own": {"k": 10, "n": 101, "status": "ok"},
                    "cross": {"CD81": {"k": None, "n": None, "status": "md_failed"},
                              "CD9": {"k": 5, "n": 79, "status": "ok"}}}
    cav = aggregate_cavity("CD63", "dual", snaps, ["CD81", "CD9"], n_boot=2000)
    check("E5 J_C excludes the everywhere-missing arm", cav["bookkeeping"]["J_C"], ["CD9"])
    check("E5 worst_case_over_n", cav["bookkeeping"]["worst_case_over_n"], 1)
    check("E5 gate (e) completeness fails", cav["gate"]["conditions"]["e_completeness"], False)
    check_true("E5 missing arm named in the verdict reasons",
               any("CD81" in x for x in cav["gate"]["reasons"]))

    print("\n=== EDGE CASE E6: unequal frame counts / window lengths ===")
    print("    spec: hard-error (not warn) if usable window < 20 ns or stride > 20 ps.")
    from pcsi_star import MIN_WINDOW_NS, MAX_STRIDE_PS
    check("E6 MIN_WINDOW_NS", MIN_WINDOW_NS, 20.0)
    check("E6 MAX_STRIDE_PS", MAX_STRIDE_PS, 20.0)
    print("    (enforced inside analyze_leg(check_window=True); exercised on real")
    print("     trajectories by the `anchor` check, which records t_first/t_last/stride)")

    print("\n=== EDGE CASE E7: J_C is a singleton ===")
    print("    spec: compute normally, set worst_case_over_n=1, print it, and never")
    print("    call it 'worst-case across off-targets'.")
    check("E7 flag present", cav["bookkeeping"]["worst_case_over_n"], 1)

    print("\n=== EDGE CASE E8: degenerate bootstrap, all S[s] identical ===")
    print("    spec: detect s==0, emit [theta,theta], degenerate=True, and do NOT")
    print("    report it as a passing CI even though the lower bound exceeds 0.")
    ci = bootstrap_ci([1.0] * 8, n_boot=2000)
    check("E8 degenerate", ci["degenerate"], True)
    check("E8 bca_lo == theta", ci["bca_lo"], 1.0)
    check("E8 sd", ci["sd"], 0.0)
    g = evaluate_gate(ci, 8, 8, ["CD81", "CD9"], ["CD81", "CD9"])
    check("E8 lower bound > 0 but verdict is NOT PASS", g["verdict"], "UNDECIDABLE")
    check("E8 gate (d) stability fails", g["conditions"]["d_stability"], False)
    check("E8 gate (a) formally true", g["conditions"]["a_ci_lower_gt_0"], True,
          "-> proves the conjunction, not (a) alone, is what blocks it")

    print("\n=== GATE CONJUNCTION (a)-(e) ===")
    good = [0.30, 0.35, 0.28, 0.40, 0.33, 0.31, 0.36, 0.29, 0.34, 0.32]
    ci = bootstrap_ci(good, n_boot=10000)
    g = evaluate_gate(ci, 10, 10, ["CD81", "CD9"], ["CD81", "CD9"])
    check("gate PASS on a clean cavity", g["verdict"], "PASS")
    check("label STRONG (theta > 0.20)", g["label"], "STRONG")
    check_true("gate (a) is the CI lower bound, not the point estimate",
               ci["bca_lo"] > 0)
    g = evaluate_gate(bootstrap_ci(good[:4], n_boot=5000), 4, 4,
                      ["CD81", "CD9"], ["CD81", "CD9"])
    check("(b) n_def=4 < 5 -> UNDECIDABLE not PASS", g["verdict"], "UNDECIDABLE")
    g = evaluate_gate(ci, 10, 2, ["CD81", "CD9"], ["CD81", "CD9"])
    check("(c) n_bound=2 < 3 -> UNDECIDABLE", g["verdict"], "UNDECIDABLE")
    g = evaluate_gate(ci, 10, 10, ["CD9"], ["CD81", "CD9"])
    check("(e) missing arm -> UNDECIDABLE even with a clean CI", g["verdict"],
          "UNDECIDABLE")
    bad = [-0.40, -0.35, -0.42, -0.38, -0.30, -0.45, -0.36, -0.41, -0.33, -0.39]
    g = evaluate_gate(bootstrap_ci(bad, n_boot=10000), 10, 10,
                      ["CD81", "CD9"], ["CD81", "CD9"])
    check("FAIL (upper bound < 0) is distinct from UNDECIDABLE", g["verdict"], "FAIL")
    straddle = [0.30, -0.30, 0.25, -0.20, 0.10, -0.15, 0.35, -0.35, 0.05, 0.0]
    g = evaluate_gate(bootstrap_ci(straddle, n_boot=10000), 10, 10,
                      ["CD81", "CD9"], ["CD81", "CD9"])
    check("CI straddles 0 -> UNDECIDABLE, not FAIL", g["verdict"], "UNDECIDABLE")

    print("\n=== SCALE EQUIVALENCE to legacy PCSI on fractions: D = (R-1)/(R+1) ===")
    for R, want in [(1.0, 0.0), (1.2, 0.0909090909090909), (1.5, 0.20),
                    (2.0, 1 / 3)]:
        d = contrast(R, 1.0)
        check(f"  R={R} -> D", round(d, 10), round(want, 10))
    check("  R=inf (f_cross=0) -> D", contrast(1.0, 0.0), 1.0)

    print("\n=== BOUNDEDNESS ===")
    rng = np.random.default_rng(1)
    vs = [contrast(a, b) for a, b in rng.random((2000, 2)) if a + b > 0]
    check_true("all D in [-1,+1]", all(-1.0 <= v <= 1.0 for v in vs))

    print("\n=== MEAN-OF-MIN <= MIN-OF-MEAN (Jensen; the gate takes the safe one) ===")
    snaps = {}
    rng = np.random.default_rng(7)
    for s in range(10):
        snaps[s] = {"own": {"k": 12, "n": 101, "status": "ok"},
                    "cross": {"CD81": {"k": int(rng.integers(2, 10)), "n": 89,
                                       "status": "ok"},
                              "CD9": {"k": int(rng.integers(2, 10)), "n": 79,
                                      "status": "ok"}}}
    cav = aggregate_cavity("CD63", "main", snaps, ["CD81", "CD9"], n_boot=5000)
    check_true(f"mean_of_min ({cav['mean_of_min']:.4f}) <= min_of_mean "
               f"({cav['min_of_mean']:.4f})",
               cav["mean_of_min"] <= cav["min_of_mean"] + 1e-12)

    print("\n=== BOOKKEEPING: snapshot categories are exhaustive and disjoint ===")
    snaps = {0: {"own": {"k": 10, "n": 101, "status": "ok"},
                 "cross": {"CD81": {"k": 3, "n": 89, "status": "ok"},
                           "CD9": {"k": 4, "n": 79, "status": "ok"}}},
             1: {"own": {"k": 0, "n": 101, "status": "ok"},          # no_binding
                 "cross": {"CD81": {"k": 0, "n": 89, "status": "ok"},
                           "CD9": {"k": 0, "n": 79, "status": "ok"}}},
             2: {"own": {"k": None, "n": None, "status": "md_failed"},  # own absent
                 "cross": {"CD81": {"k": 3, "n": 89, "status": "ok"},
                           "CD9": {"k": 4, "n": 79, "status": "ok"}}}}
    cav = aggregate_cavity("CD63", "main", snaps, ["CD81", "CD9"], n_boot=1000)
    b = cav["bookkeeping"]
    check("n_snapshots", b["n_snapshots"], 3)
    check("n_def", b["n_def"], 1)
    check("n_nobind", b["n_nobind"], 1)
    check("n_own_absent", b["n_own_absent"], 1)
    check("partition sums to n_snapshots",
          b["n_def"] + b["n_nobind"] + b["n_own_absent"] + b["n_no_cross_data"],
          b["n_snapshots"])
    check("no_binding snapshot excluded from the bootstrap pool",
          len(cav["S_values"]), 1)
    check_true("no_binding was NOT scored as 0.0",
               0.0 not in [v for v in cav["S_values"]] or cav["S_values"] != [0.0])

    print("\n=== LEGACY PCSI reproduction (small-count fragility, made visible) ===")
    check("legacy 6 / max(0,3)", legacy_pcsi(6, [0, 3]), 2.0)
    check("  if CD9 were 2", legacy_pcsi(6, [0, 2]), 3.0)
    check("  if CD9 were 4", legacy_pcsi(6, [0, 4]), 1.5)
    print("    -> one residue moves the verdict across two threshold bands.")
    check("legacy all-cross-zero -> inf", legacy_pcsi(6, [0, 0]), math.inf)
    ls = legacy_summary([2.0, 1.1, math.inf, 0.9, math.inf])
    check("legacy mean over FINITE only (midrun_pcsi_check.py:67)",
          round(ls["mean_finite_only"], 6), round((2.0 + 1.1 + 0.9) / 3, 6))
    check("legacy n_inf", ls["n_inf"], 2)
    check("legacy n_pass counts over ALL 5 incl. infinities", ls["n_pass_gt_1_2"], 3)
    print("    -> mean over 3, counts over 5: two statistics, two populations.")
    check("PCSI* has no infinities to drop",
          all(v is None or math.isfinite(v)
              for v in [contrast(0.06, 0.0), contrast(0.0, 0.06), contrast(0.0, 0.0)]),
          True)

    print("\n=== POWER FORMULA ===")
    for d, want80 in [(0.5, 32), (0.75, 14), (1.0, 8), (1.5, 4), (2.0, 2)]:
        p = power_requirement(d * 1.0, 1.0, 10)   # theta0=d, s0=1 -> standardised d
        check(f"  d={d} n(80%)", p["at_s0"]["n_power80"], want80)
    p = power_requirement(0.30, 0.30, 10)
    check("  n_sig uses z=1.959964 (the GATE quantile), not 1.645",
          p["at_s0"]["n_sig"], int(math.ceil(3.8415 / 1.0)))
    check_true("  s_upper > s0 (variance-uncertainty inflation is mandatory)",
               p["s_upper_80pct"] > p["s0"])
    check_true("  n at s_upper >= n at s0",
               p["at_s_upper"]["n_power80"] >= p["at_s0"]["n_power80"])
    check("  budget floored at gate (b) n>=5", power_requirement(2.0, 1.0, 10)
          ["budget_n_snapshots"], 5)
    p0 = power_requirement(0.0, 0.3, 10)
    check_true("  theta0 <= 0 -> no n, an explanation instead", "note" in p0)

    print("\n=== AUTOCORRELATION DIAGNOSTIC ===")
    from pcsi_star import lag1_autocorrelation
    ac = lag1_autocorrelation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    check_true(f"monotone series -> strong positive rho1={ac['rho1']:.3f}",
               ac["rho1"] > 0.5)
    check_true("  n_effective < n when positively correlated",
               ac["n_effective"] < 10)
    ac = lag1_autocorrelation(list(np.random.default_rng(3).normal(size=200)))
    check_true(f"  white noise -> rho1 ~ 0 ({ac['rho1']:.3f})", abs(ac["rho1"]) < 0.2)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"EDGE CASES: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("EDGE CASES: ALL PASS")
    return 0


# ====================================================== 2. REFERENCE ANCHOR ====

ANCHOR_V1 = {                    # results/reports/phase5_persistent_contacts_trial.json
    "CD63": {"k": 6, "n": 101, "f": 0.0594059405940594},
    "CD81": {"k": 0, "n": 89, "f": 0.0},
    "CD9": {"k": 3, "n": 79, "f": 0.0379746835443038},
}


def cmd_anchor(a):
    """HARD GATE against the CONTENT-ADDRESSED anchor, code/anchor_v2.json.

    v1 pinned three integers with no provenance, so when the trajectories were
    re-run 15-18 days after the reference was written it failed with an
    ambiguous "the number changed" that could not distinguish a code bug from
    new data. v2 pins the sha256 of every file the numbers came from and checks
    in three stages, so the two causes now report differently:

      stage 1 fails -> the FILE changed  (re-freeze deliberately, after asking why)
      stage 2 fails -> the CODE changed  (fix the code; never re-freeze)
      stage 3 fails -> an absent leg reappeared, i.e. new data

    The dead v1 counts are still printed on every run so the published headline
    that rests on them is never quietly forgotten.
    """
    import anchor as A

    print("\n=== HARD GATE: content-addressed anchor ===")
    rec = A.load()
    if rec is None:
        print(f"  [FAIL] no anchor at {A.ANCHOR_PATH}")
        print("         freeze one with: python code/freeze_anchor.py --freeze")
        FAILURES.append("anchor missing")
        return 1
    print(f"    anchor  {A.ANCHOR_PATH.relative_to(ROOT)}")
    print(f"    frozen  {rec['frozen_at']}  commit {rec['git_commit']}  "
          f"MDAnalysis {rec['env']['mdanalysis']}")
    print(f"    pins    {rec['cavity']} {rec['mode']} snapshot_{rec['snapshot']}  "
          f"cutoff {rec['params']['cutoff_A']} A  persistence "
          f"{rec['params']['persistence_frac']}  last_frac {rec['params']['last_frac']}")
    print(f"    hashing {'ON (sha256 over ~3.2 GB)' if not a.no_hash else 'OFF (--no-hash)'}")

    ok, rep = A.verify(full_hash=not a.no_hash, verbose=True)
    for stage, sok in rep["stages"].items():
        check_true(f"stage {stage}", sok)

    print("\n=== v1 ANCHOR (DEAD) — kept visible, never used as a gate ===")
    print("    results/reports/phase5_persistent_contacts_trial.json, mtime "
          + time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(
              (ROOT / "results/reports/phase5_persistent_contacts_trial.json")
              .stat().st_mtime)))
    for lab, want in ANCHOR_V1.items():
        leg = rec["legs"].get(lab, {})
        if leg.get("expect") == "present":
            got = leg["counts"]
            print(f"    {lab:5s} v1 k={want['k']:3d} n={want['n']:3d}   |   "
                  f"today k={got['k']:3d} n={got['n']:3d}   "
                  f"{'AGREES' if got['k'] == want['k'] else 'DIFFERS'}")
        else:
            print(f"    {lab:5s} v1 k={want['k']:3d} n={want['n']:3d}   |   "
                  f"today NO TRAJECTORY "
                  f"({leg.get('absence_evidence', {}).get('classification')})")
    d = rec.get("derived_headline") or {}
    if d:
        print(f"    published headline 'PCSI 2.00 STRONG' -> on current data "
              f"legacy PCSI {d['legacy_pcsi_counts']}, PCSI* "
              f"{d['pcsi_star_worst_case']:+.4f} (sign flip). The figure needs "
              f"revisiting independently of this redesign.")

    out = ROOT / "reports_v2/anchor_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"schema": "anchor_check/2", "anchor_verified": ok,
         "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "full_hash": not a.no_hash, "report": rep,
         "v1_anchor_dead": ANCHOR_V1}, indent=2, default=str))
    print(f"\n    wrote {out}")

    if not ok:
        print("\n  *** HARD GATE FAILED — STOPPING. The reference is NOT adjusted")
        print("  *** to match the code. Read the stage that failed:")
        print("  ***   stage 1 = the trajectory changed (new data, re-freeze on purpose)")
        print("  ***   stage 2 = the code changed (fix the code, do NOT re-freeze)")
        print("  ***   stage 3 = a leg that had no trajectory now has one")
        print("  *** The 88-leg run is blocked until this passes.")
        return 1
    print("\n  ANCHOR VERIFIED — files unchanged, counts reproduce, absences intact.")
    return 0


# ==================================================== 3. LOOP vs VECTORISED ====

def _reference_loop(traj, top, cutoff_A=6.0, last_frac=0.5, max_frames=None):
    """The LEGACY per-residue loop, copied verbatim from
    code/analyze_persistent_contacts.py:41-48 (that file is NOT modified).
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array
    u = mda.Universe(str(top), str(traj))
    protein = u.select_atoms("protein")
    monomers = u.select_atoms("not protein and not resname SOL HOH NA CL TIP3 WAT")
    n_frames = len(u.trajectory)
    start = int(n_frames * (1 - last_frac))
    if max_frames:
        start = max(start, n_frames - max_frames)
    n_analyzed = n_frames - start
    res_contact_count = {res.resid: 0 for res in protein.residues}
    for ts in u.trajectory[start:]:
        for res in protein.residues:
            d = distance_array(res.atoms.positions, monomers.positions,
                               box=ts.dimensions)
            if d.min() < cutoff_A:
                res_contact_count[res.resid] += 1
    freq = {rid: c / n_analyzed for rid, c in res_contact_count.items()}
    return freq, sum(1 for f in freq.values() if f > 0.5), n_analyzed


def cmd_equiv(a):
    from pcsi_star import analyze_leg, enable_readonly_mode
    enable_readonly_mode()
    md = Path(a.md_dir)
    if not md.is_absolute():
        md = ROOT / md
    traj, top = md / "md.xtc", md / "md.tpr"

    import MDAnalysis as mda
    nf = len(mda.Universe(str(top), str(traj)).trajectory)
    lf = (a.frames / nf) if a.frames else 0.5
    print(f"\n=== LOOP vs VECTORISED, per-residue frequencies ===")
    print(f"    {md.relative_to(ROOT)}  n_frames={nf}  window=last {a.frames or nf//2} frames")

    t0 = time.perf_counter()
    ref_freq, ref_k, n_an = _reference_loop(traj, top, last_frac=lf,
                                            max_frames=a.frames)
    t_ref = time.perf_counter() - t0
    print(f"    reference loop      : k={ref_k:3d}  n_res={len(ref_freq):3d}  "
          f"n_analyzed={n_an}  {t_ref:8.2f}s")

    results = {}
    for method in ("capped", "dense"):
        t0 = time.perf_counter()
        res = analyze_leg(traj, top, cutoffs=(6.0,), last_frac=lf, method=method,
                          check_window=False)
        t = time.perf_counter() - t0
        b = res["by_cutoff"]["6"]
        got = {int(k): v for k, v in b["contact_freq"].items()}
        k = b["n_persistent"]["0.5"]
        keys_eq = set(got) == set(ref_freq)
        diffs = {r: (ref_freq[r], got[r]) for r in ref_freq
                 if got.get(r) != ref_freq[r]}
        maxd = max((abs(got[r] - ref_freq[r]) for r in ref_freq), default=0.0)
        results[method] = got
        print(f"    vectorised {method:7s} : k={k:3d}  n_res={len(got):3d}  "
              f"{t:8.2f}s   speedup {t_ref/t:5.2f}x")
        check(f"{method}: n_persistent matches loop", k, ref_k)
        check_true(f"{method}: residue key sets identical", keys_eq)
        check(f"{method}: max |delta freq|", maxd, 0.0)
        check(f"{method}: n residues differing", len(diffs), 0)
        if diffs:
            print(f"        first diffs: {dict(list(diffs.items())[:5])}")
    check_true("capped == dense exactly", results["capped"] == results["dense"])

    print(f"\n    n_analyzed={n_an}, so each freq is a multiple of 1/{n_an};")
    print(f"    bit-for-bit equality of every residue frequency is the check, not "
          f"just the count.")
    return 1 if FAILURES else 0


# =============================================================== 4. TIMING =====

def cmd_timing(a):
    from pcsi_star import analyze_leg, enable_readonly_mode, CUTOFF_GRID
    enable_readonly_mode()
    md = Path(a.md_dir)
    if not md.is_absolute():
        md = ROOT / md
    traj, top = md / "md.xtc", md / "md.tpr"

    print(f"\n=== TIMING, full production window ===")
    print(f"    {md.relative_to(ROOT)}")
    rows = {}
    for tag, cutoffs in (("single cutoff 6 A", (6.0,)),
                         ("sweep 5/6/7 A (production default)", CUTOFF_GRID)):
        t0 = time.perf_counter()
        res = analyze_leg(traj, top, cutoffs=cutoffs, keep_frames=(cutoffs != (6.0,)))
        t = time.perf_counter() - t0
        rows[tag] = t
        print(f"    {tag:38s} {t:7.2f} s   "
              f"({res['n_analyzed']} frames, {res['n_protein_atoms']} prot atoms, "
              f"{res['n_monomer_atoms']} monomer atoms)")

    per_leg = rows["sweep 5/6/7 A (production default)"]
    print(f"\n    PROJECTION for 88 legs at {per_leg:.1f} s/leg:")
    for w in (1, 4, 6, 8, 12):
        eff = {1: 1.0, 4: 3.45, 6: 4.49, 8: 5.3, 12: 6.28}[w]
        print(f"      {w:2d} worker(s): {88*per_leg/eff/60:6.1f} min "
              f"(measured parallel efficiency {eff:.2f}x)")
    print(f"\n    Reference-loop equivalent would be ~{88*35/60:.0f} min serial.")
    print(f"    Bounded below by ~122 GB of XTC reads regardless of core count.")
    return 0


# ================================================================== dispatch ===

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("edge")
    p = sub.add_parser("anchor")
    p.add_argument("--no-hash", action="store_true",
                   help="skip the sha256 provenance check (size+mtime only); "
                        "faster, but blind to exactly the re-run that killed v1")
    p = sub.add_parser("equiv")
    p.add_argument("md_dir")
    p.add_argument("--frames", type=int, default=None,
                   help="use only the last N frames (fast check)")
    p = sub.add_parser("timing")
    p.add_argument("md_dir")
    a = ap.parse_args()
    rc = {"edge": cmd_edge, "anchor": cmd_anchor, "equiv": cmd_equiv,
          "timing": cmd_timing}[a.cmd](a)
    sys.exit(rc)


if __name__ == "__main__":
    main()
