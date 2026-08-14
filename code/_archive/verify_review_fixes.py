"""Verification that each review finding is actually fixed, as executable checks.

    python code/verify_review_fixes.py            # all checks
    python code/verify_review_fixes.py 3          # one finding

Every check re-creates the SITUATION described in the finding's evidence and
asserts the corrected behaviour. Read-only with respect to results/. Uses the
two anchor trajectories already exercised elsewhere and no others.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcsi_star as P                                             # noqa: E402
import run_pcsi_star as R                                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FAILURES = []


def check(name, got, want, ok=None):
    good = (got == want) if ok is None else ok
    print(f"  [{'PASS' if good else 'FAIL'}] {name}\n"
          f"         got={got!r}\n         want={want!r}")
    if not good:
        FAILURES.append(name)
    return good


def leg(k, n, st="ok", window=None):
    d = {"k": k, "n": n, "status": st}
    if window:
        d["window"] = window
    return d


W = {"n_analyzed": 2501, "dt_ps": 10.0, "t_first_ps": 25000.0,
     "t_last_ps": 50000.0, "window_ns": 25.0}


# ---------------------------------------------------------------- finding 1 --
def f1():
    print("\n=== FINDING 1: gate (a) anti-conservative; docstring claimed 97.5% ===")
    rng = np.random.default_rng(0)
    v = list(rng.normal(0.30, 0.15, 10))
    ci = P.bootstrap_ci(v, n_boot=4000, seed=1)
    check("gate_lo is the MINIMUM of the three lower bounds",
          round(ci["gate_lo"], 10),
          round(min(ci["bca_lo"], ci["perc_lo"], ci["t_lo"]), 10))
    check("gate_lo <= bca_lo (never looser than BCa)",
          (ci["gate_lo"], ci["bca_lo"]), "gate_lo <= bca_lo",
          ok=ci["gate_lo"] <= ci["bca_lo"])
    check("studentized computed but EXCLUDED from the gate methods",
          ("stud_lo present", "studentized" in P.GATE_LB_METHODS),
          ("stud_lo present", False),
          ok=("stud_lo" in ci) and ("studentized" not in P.GATE_LB_METHODS))

    # the docstring must no longer assert conservativeness
    doc = P.bootstrap_ci.__doc__
    check("bootstrap_ci docstring no longer calls the bound conservative",
          "a conservative 97.5% one-sided bound" in doc, False)
    check("bootstrap_ci docstring states the MEASURED error rates",
          all(s in doc for s in ("0.0862", "0.0535", "it is NOT")), True)

    # Measured type-I on an exact null, at the sizes the gate permits.
    # Averaged over independent pilots: a SINGLE pilot's estimate has sd
    # 0.02-0.055, so one draw cannot certify calibration (that is precisely why
    # MAX_MEASURED_TYPE1 is a gross-failure guard and not a fine threshold).
    print("  measured type-I of the gate under an exact null, 25 pilots each:")
    for n in (5, 10):
        est = [P.gate_type1_error(list(rng.normal(0.0, 0.26, n)),
                                  n_sim=300, n_boot=400,
                                  seed=int(rng.integers(1 << 30)))["type1"]
               for _ in range(25)]
        med, mean = float(np.median(est)), float(np.mean(est))
        print(f"    n={n:2d}  median {med:.4f}  mean {mean:.4f}  "
              f"max {max(est):.4f}   (nominal 0.025)")
        check(f"n={n}: median measured type-I at or near nominal 0.025",
              round(med, 4), "<= 0.05", ok=med <= 0.05)

    # a cavity whose measured type-I is bad must fail gate (d)
    bad = {"type1": 0.20, "nominal": 0.025, "exceeds_limit": True, "n": 10}
    ci2 = P.bootstrap_ci([0.3] * 5 + [0.35] * 5, n_boot=2000, seed=1)
    g = P.evaluate_gate(ci2, 10, 10, ["A", "B"], ["A", "B"], type1=bad)
    check("inflated measured type-I blocks the gate via (d)",
          (g["conditions"]["d_stability"], g["verdict"]), (False, "UNDECIDABLE"))


# ---------------------------------------------------------------- finding 2 --
def f2():
    print("\n=== FINDING 2: with/without-imputed split could never fire ===")
    xs = np.linspace(-1, 1, 4001)
    check("sup|min(1,x)-x| over [-1,1] is still 0 (the old diagnostic IS null)",
          float(np.max(np.abs(np.minimum(1.0, xs) - xs))), 0.0)

    # CD9-main's real pattern: arm A measured in 0-3, size-excluded in 4-9.
    snaps = {}
    rng = np.random.default_rng(3)
    for s in range(10):
        cross = {"B": leg(int(rng.integers(2, 8)), 89, window=W)}
        cross["A"] = (leg(int(rng.integers(2, 8)), 79, window=W) if s < 4
                      else leg(None, 79, "size_excluded"))
        snaps[s] = {"own": leg(int(rng.integers(8, 14)), 101, window=W),
                    "cross": cross}
    r = P.aggregate_cavity("T", "m", snaps, ["A", "B"], n_boot=2000, seed=1)

    d = r["imputation_diagnostic"]
    check("new diagnostic splits snapshots into clean vs imputed-arm",
          (d["n_clean"], d["n_with_imputed_arm"]), (4, 6))
    check("clean subset is snapshots 0-3", d["clean_snapshot_ids"], [0, 1, 2, 3])
    check("imputed subset is snapshots 4-9", d["imputed_snapshot_ids"],
          [4, 5, 6, 7, 8, 9])
    check("the split produces DIFFERENT thetas (i.e. it can fire)",
          (round(d["theta_clean"], 4), round(d["theta_with_imputed"], 4),
           round(d["difference"], 4)),
          "two distinct numbers with a non-zero difference",
          ok=abs(d["difference"]) > 1e-9)
    print(f"         theta_clean={d['theta_clean']:+.4f}  "
          f"theta_with_imputed={d['theta_with_imputed']:+.4f}  "
          f"diff={d['difference']:+.4f}")

    lnd = r["without_imputed"]["legacy_null_diagnostic"]
    check("the superseded diagnostic is reported AS null, with its proof",
          (lnd["max_abs_change_in_S"], "identically null" in
           r["without_imputed"]["legacy_null_diagnostic"]["proof"]
           or "never change a value" in lnd["proof"]),
          (0.0, True))


# ---------------------------------------------------------------- finding 3 --
def f3():
    print("\n=== FINDING 3: size-exclusion smuggled heterogeneous J_C into CD9 ===")
    snaps = {}
    rng = np.random.default_rng(3)
    for s in range(10):
        cross = {"CD81": leg(int(rng.integers(2, 8)), 89, window=W)}
        cross["CD63"] = (leg(int(rng.integers(2, 8)), 101, window=W) if s < 4
                         else leg(None, 101, "size_excluded"))
        snaps[s] = {"own": leg(int(rng.integers(8, 14)), 79, window=W),
                    "cross": cross}
    r = P.aggregate_cavity("CD9", "main", snaps, ["CD63", "CD81"],
                           n_boot=2000, seed=1)
    b = r["bookkeeping"]

    check("size-excluded arm EXCLUDED from the primary functional",
          b["J_C"], ["CD81"])
    check("worst_case_over_n no longer claims 2", b["worst_case_over_n"], 1)
    check("no imputed legs reach the primary estimator", b["n_imputed_legs"], 0)
    check("gate (e) now fails and names the arm",
          (r["gate"]["conditions"]["e_completeness"],
           r["gate"]["missing_arms"], r["gate"]["verdict"]),
          (False, ["CD63"], "UNDECIDABLE"))
    check("S is the SAME functional in every snapshot (one arm count)",
          sorted({rec["n_cross_used"] for rec in r["snapshots"].values()
                  if rec.get("S") is not None}), [1])

    see = r["size_exclusion_evidence"]
    check("size-exclusion evidence retained and flagged separately",
          (see["CD63"]["n_snapshots_size_excluded"],
           see["CD63"]["in_primary_J_C"]), (6, False))

    st = r["stratified"]
    strata = {(tuple(x["measured_arms"]), tuple(x["size_excluded_arms"])): x
              for x in st}
    check("stratified table reports the two designs separately",
          sorted((len(k[0]), v["n"]) for k, v in strata.items()),
          [(1, 6), (2, 4)])
    for k, v in sorted(strata.items(), key=lambda kv: -len(kv[0][0])):
        print(f"         measured={list(k[0])} excluded={list(k[1])} "
              f"n={v['n']} theta={v['theta']:+.4f} "
              f"worst_case_over_n={v['worst_case_over_n']}")


# ---------------------------------------------------------------- finding 4 --
def f4():
    print("\n=== FINDING 4: budget was an uncapped point estimate ===")
    print("  the blow-up cases from the finding, now capped and labelled:")
    for th in (0.005, 1e-06):
        p = P.power_requirement(th, 0.28, 10)
        print(f"    theta={th:<9g} uncapped={p['budget_n_snapshots_uncapped']:<15d} "
              f"budget={p['budget_n_snapshots']:<4d} infeasible={p['budget_infeasible']}")
        check(f"theta={th}: budget capped at MAX_FEASIBLE_N",
              p["budget_n_snapshots"], P.MAX_FEASIBLE_N)
        check(f"theta={th}: flagged infeasible with an explanation",
              (p["budget_infeasible"], "effect too small" in p.get("note", "")),
              (True, True))

    rng = np.random.default_rng(0)
    v = list(rng.normal(0.30, 0.28, 10))
    bi = P.budget_interval(v, n_boot=800, seed=1)
    check("budget reported as an interval, not one integer",
          sorted(k for k in ("n_median", "n_q25", "n_q75",
                             "n_at_quantile_uncapped", "n_max") if k in bi),
          ["n_at_quantile_uncapped", "n_max", "n_median", "n_q25", "n_q75"])
    print(f"         n: median {bi['n_median']}  IQR {bi['n_q25']}-{bi['n_q75']}  "
          f"80th {bi['n_at_quantile_uncapped']}  max {bi['n_max']}  "
          f"-> recommend {bi['n_recommended']}")
    check("the recommendation is the upper quantile, not the median",
          bi["n_recommended"] >= bi["n_median"], True)

    # a cavity whose theta CI straddles 0 must say so AT the budget
    rr = np.random.default_rng(11)
    straddle = {s: {"own": leg(int(rr.integers(4, 10)), 101, window=W),
                    "cross": {"A": leg(int(rr.integers(4, 10)), 89, window=W)}}
                for s in range(10)}
    r = P.aggregate_cavity("T", "m", straddle, ["A"], n_boot=4000, seed=1)
    bd = r["budget"]
    check("theta's own CI is carried on the budget record",
          all(k in bd for k in ("theta_hat", "theta_ci_gate_lo",
                                "theta_ci_gate_hi", "theta_ci_straddles_zero")),
          True)
    print(f"         theta={bd['theta_hat']:+.4f} "
          f"CI=[{bd['theta_ci_gate_lo']:+.4f},{bd['theta_ci_gate_hi']:+.4f}] "
          f"straddles_zero={bd['theta_ci_straddles_zero']}")
    check("a straddling CI raises an explicit warning at the budget",
          bd["theta_ci_straddles_zero"] and "straddles 0" in bd.get("warning", ""),
          True)


# ---------------------------------------------------------------- finding 5 --
def f5():
    print("\n=== FINDING 5: power curve endpoint was one deterministic draw ===")
    v = [0.5575, 0.7512, 0.1557, 0.0279, -0.2042, 0.0797, 0.6487,
         -0.2605, -0.2605, 1.0]
    c = P.empirical_power_curve(v, n_boot=500, n_draws=100, seed=0)
    print("  curve =", {k: round(x, 3) for k, x in c.items()})
    check("n'=n endpoint is no longer exactly 0.0 or 1.0",
          c[10], "a rate strictly between 0 and 1",
          ok=0.0 < c[10] < 1.0)
    check("every point carries a Monte-Carlo standard error",
          sorted(c.mc_se) == sorted(c), True)
    check("the ~0.27 per-pilot sd is stated on the object",
          "0.27" in c.caveat and "CANNOT adjudicate" in c.caveat, True)
    check("still a plain n'->rate mapping for existing callers",
          (isinstance(c, dict), sorted(c)), (True, list(range(3, 11))))


# ---------------------------------------------------------------- finding 7 --
def f7():
    print("\n=== FINDING 7: unequal analysis windows never compared ===")
    W2 = {"n_analyzed": 2006, "dt_ps": 10.0, "t_first_ps": 29950.0,
          "t_last_ps": 50000.0, "window_ns": 20.05}
    same = {s: {"own": leg(10, 101, window=W),
                "cross": {"A": leg(5, 89, window=W)}} for s in range(10)}
    r_ok = P.aggregate_cavity("T", "m", same, ["A"], n_boot=2000, seed=1)
    check("identical windows -> gate (f) passes",
          (r_ok["window_check"]["consistent"],
           r_ok["gate"]["conditions"]["f_window_consistent"]), (True, True))

    mixed = {s: {"own": leg(10, 101, window=W),
                 "cross": {"A": leg(5, 89, window=W if s < 5 else W2)}}
             for s in range(10)}
    r_bad = P.aggregate_cavity("T", "m", mixed, ["A"], n_boot=2000, seed=1)
    check("mismatched windows -> gate (f) fails, verdict UNDECIDABLE",
          (r_bad["window_check"]["consistent"],
           r_bad["gate"]["conditions"]["f_window_consistent"],
           r_bad["gate"]["verdict"]), (False, False, "UNDECIDABLE"))
    check("the mismatch is described, both windows named",
          r_bad["window_check"]["n_distinct"], 2)
    reason = [x for x in r_bad["gate"]["reasons"] if "UNEQUAL ANALYSIS WINDOWS" in x]
    check("a reason line names the offending windows", bool(reason), True)
    if reason:
        print(f"         {reason[0][:150]}")
    check("the subsampling rule is stated for the future re-run",
          "coarsest common" in r_bad["window_check"]["rule"], True)


# --------------------------------------------------------------- finding 13 --
def f13():
    print("\n=== FINDING 13: variance decomposition depended on checkpoint order ===")
    ck = ROOT / "reports_v2/pcsi_star_legs_v2.jsonl"
    legs = []
    for line in ck.read_text().splitlines():
        if line.strip():
            try:
                legs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    legs = [r for r in legs if r.get("ok") and r.get("frames_npz")]
    if len(legs) < 2:
        print("  SKIP: fewer than 2 frame-storing legs in the checkpoint")
        return
    agg = {"CD63_dual": {"S_values": [0.10, -0.29],
                         "snapshots": {"0": {"f_own": 0.0693,
                                             "cross": {"CD9": {"f_cross": 0.1266}},
                                             "S": -0.29},
                                       "1": {"f_own": 0.0700,
                                             "cross": {"CD9": {"f_cross": 0.0600}},
                                             "S": 0.10}}}}
    outs = []
    for perm in ([0, 1], [1, 0]):
        ordered = [legs[i] for i in perm]
        outs.append(R.variance_decomposition(ordered, "CD63_dual", agg, 0.5,
                                             max_legs=None))
    check("var_within_f identical under checkpoint reordering",
          [o["var_within_f"] for o in outs],
          "both orders equal",
          ok=outs[0]["var_within_f"] == outs[1]["var_within_f"])
    check("legs_used identical and sorted by key",
          [o["legs_used"] for o in outs],
          [sorted(o["legs_used"]) for o in outs])
    check("every frame-storing leg used, not an arbitrary 3",
          outs[0]["n_legs_used"], len(legs))
    check("per-leg variances recorded so the mean is auditable",
          sorted(outs[0]["var_within_f_per_leg"]), sorted(outs[0]["legs_used"]))
    print(f"         legs_used={outs[0]['legs_used']}")
    print(f"         var_within_f={outs[0]['var_within_f']:.6g} (both orders)")


ALL = {1: f1, 2: f2, 3: f3, 4: f4, 5: f5, 7: f7, 13: f13}

if __name__ == "__main__":
    want = [int(x) for x in sys.argv[1:]] or sorted(ALL)
    for k in want:
        ALL[k]()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")
