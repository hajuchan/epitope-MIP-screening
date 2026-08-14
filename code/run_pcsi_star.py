"""Driver: recompute PCSI* over all Phase 5 rebinding legs and report the budget.

    python code/run_pcsi_star.py                      # full run, 6 workers
    python code/run_pcsi_star.py --workers 8
    python code/run_pcsi_star.py --aggregate-only     # re-stat from checkpoint, free
    python code/run_pcsi_star.py --limit 2            # smoke test, 2 legs

Read-only with respect to results/. Every output goes to --out-dir (reports_v2/).
An interrupted run resumes from the per-leg JSONL checkpoint: only legs that are
absent or recorded ok=False are recomputed.

Emits, per cavity: PCSI* point estimate, BCa + percentile bootstrap CI, the
five-condition gate verdict, the legacy PCSI (counts and fractions) for
comparison, the snapshot-to-snapshot variance, and the snapshot count implied by
the power formula.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent

# EXPERIMENT AWARENESS (REVIEW FINDINGS 2 + 3).
#
# This driver used to hardcode `ROOT / "results/phase5"` and
# `TARGETS = ["CD63", "CD81", "CD9"]`, and defaulted --out-dir to
# `ROOT / "reports_v2"`. Three consequences, all silent:
#
#   1. verify_phase5 blocks unless <results_root>/reports/pcsi_star_summary.json
#      exists, but this script wrote to reports_v2/ — a DIFFERENT directory.
#      The documented unblocking step could not clear the gate.
#   2. reports_v2/ is the FROZEN PCSI* recompute and already holds files of
#      exactly these names, so running the documented command overwrote a
#      protected artifact.
#   3. Under MIP_EXPERIMENT=BSA the results tree is results_BSA_intact/, so the
#      driver scanned the CD tree and verify_phase5 read a summary nothing
#      could ever produce — an unsatisfiable block.
#
# Everything is now derived from pipeline.config, which is experiment-aware.
# Under CD the values are IDENTICAL to the hardcoded ones (verified), so this
# is behaviour-preserving there.
from pipeline.config import OUTPUT_DIRS as _OUTPUT_DIRS, TARGETS as _CFG_TARGETS

TARGETS = list(_CFG_TARGETS)
PHASE5_DIR = Path(_OUTPUT_DIRS["phase5"])
DEFAULT_OUT_DIR = Path(_OUTPUT_DIRS["reports"])

# Each worker must stay single-threaded or N workers fight over the BLAS pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


# ------------------------------------------------------------ leg inventory --

def leg_key(cav, mode, snap, lig, replica=0):
    # replica is part of the key: with snapshots spread across Phase 4 replicas,
    # (cavity, mode, snapshot index, ligand) is no longer unique — rep0_snapshot_0
    # and rep1_snapshot_0 are different structures. Replica 0 renders as the old
    # 4-field key so legacy single-trajectory inventories keep their identifiers.
    return (f"{cav}|{mode}|{snap}|{lig}" if not replica
            else f"{cav}|{mode}|r{replica}s{snap}|{lig}")


def _pipeline_status(pipeline, cav, snap, lig, replica=0):
    """Recorded pipeline status for one leg, or None. Used ONLY to confirm
    size-exclusion — never to conclude a leg ran.

    The snapshots list is positional, but with snapshots spread across Phase 4
    replicas the position is no longer the snapshot index: rep0_snapshot_0 and
    rep1_snapshot_0 both have index 0. Match on the recorded (replica, index)
    when Phase 5 wrote them, and fall back to positional indexing for a legacy
    single-trajectory result file.
    """
    try:
        snaps = pipeline[cav]["snapshots"]
    except (KeyError, TypeError):
        return None
    rec = None
    if isinstance(snaps, list):
        for r in snaps:
            if not isinstance(r, dict):
                continue
            if r.get("replica") is not None and \
                    (int(r.get("replica")), int(r.get("replica_snapshot_index", -1))) \
                    == (int(replica), int(snap)):
                rec = r
                break
        if rec is None and replica == 0:
            try:
                rec = snaps[snap]
            except (IndexError, TypeError):
                return None
    if not isinstance(rec, dict):
        return None
    return rec.get("rebind_own" if lig == cav else f"rebind_{lig}")


def classify(md_dir, pipe_rec):
    """Status from DISK EVIDENCE. Missing md.xtc does NOT mean size-excluded.

    32 legs lack md.xtc but only 6 are size-excluded; the other 26 are NVT
    crashes. phase5_rebinding_results.json is no help — it records the crashed
    legs as time_ns=50, rebound=None, status=None, identical in shape to a
    completed leg. Only the filesystem distinguishes them. Imputing +1 for a
    crash would manufacture perfect selectivity out of a GROMACS failure.
    """
    if (md_dir / "md.xtc").exists() and (md_dir / "md.tpr").exists():
        return "ok"
    # Size exclusion is decided PRE-MD (compute_steric_clash > 30,
    # phase5_rebinding.py:866-895 returns BEFORE energy minimisation), so the
    # signature is: recorded status SIZE_EXCLUDED *and* no em.gro.
    said_excluded = bool(pipe_rec) and (
        pipe_rec.get("status") == "SIZE_EXCLUDED" or pipe_rec.get("size_excluded") is True)
    has_em = (md_dir / "em.gro").exists()
    if said_excluded and not has_em:
        return "size_excluded"
    if said_excluded and has_em:
        return "conflict_excluded_but_em_present"   # fail closed, never imputed
    nvt_xtc = md_dir / "nvt.xtc"
    if (md_dir / "nvt.log").exists() and nvt_xtc.exists() and nvt_xtc.stat().st_size == 0:
        return "md_failed"                          # NVT crash: NOT data
    if has_em:
        return "md_failed"
    if not md_dir.exists():
        return "absent"
    return "absent"


_SNAP_DIR_RE = re.compile(r"^(?:rep(?P<rep>\d+)_)?snapshot_(?P<idx>\d+)$")


def _discover_snapshots(cav_dir):
    """[(replica, index, dir)] for a cavity, understanding BOTH Phase 5 layouts.

    Phase 5 used to write snapshot_<i> for i in 0..REBINDING_N_SNAPSHOTS-1, all
    carved out of ONE Phase 4 trajectory, and this function used to assume
    exactly that by iterating range(10). Phase 4 now runs PHASE4_N_REPLICAS
    independent trajectories and Phase 5 spreads its snapshots across them,
    naming them rep<r>_snapshot_<i> so the source replica stays recoverable.
    Iterating range(10) against that layout finds NOTHING — and PCSI* would
    report an empty inventory rather than an error, which is precisely the
    silent-failure mode this pipeline is being audited for.

    The replica is carried into every row. Snapshots from ONE trajectory are
    autocorrelated (measured rho1 = 0.23, so 10 snapshots carry ~6.3 independent
    samples); snapshots from different replicas are independent. The statistic
    itself is UNCHANGED here — bootstrap_ci still resamples snapshots as the
    exchangeable unit — but the grouping needed to correct that is now present
    in the data instead of being unrecoverable. See the note in bootstrap_ci.
    """
    out = []
    if not cav_dir.is_dir():
        return out
    for d in cav_dir.iterdir():
        if not d.is_dir():
            continue
        m = _SNAP_DIR_RE.match(d.name)
        if m:
            out.append((int(m.group("rep") or 0), int(m.group("idx")), d))
    return sorted(out, key=lambda t: (t[0], t[1]))


def build_inventory(pipeline):
    """The full expected grid: every cavity x discovered snapshot x 3 legs, + CD63 dual."""
    rows = []
    for cav in TARGETS:
        others = [o for o in TARGETS if o != cav]
        snaps = _discover_snapshots(PHASE5_DIR / cav)
        if not snaps:
            print(f"[pcsi*] WARNING: no snapshot directories found for {cav} "
                  f"under {PHASE5_DIR / cav} (looked for snapshot_<i> and "
                  f"rep<r>_snapshot_<i>)", file=sys.stderr)
        for replica, snap, base in snaps:
            modes = [("main", base)]
            if cav == "CD63":
                modes.append(("dual", base / "dual_imprinting"))
            for mode, prefix in modes:
                for lig, sub in [(cav, "rebind_own")] + [(o, f"rebind_{o}") for o in others]:
                    md = prefix / sub / "md"
                    pr = _pipeline_status(pipeline, cav, snap, lig, replica)
                    st = classify(md, pr)
                    row = {"cavity": cav, "mode": mode, "snapshot": snap,
                           # Source Phase 4 trajectory. Snapshots sharing a
                           # replica are correlated; across replicas they are not.
                           "replica": replica,
                           "snapshot_dir": base.name,
                           "ligand": lig, "is_own": lig == cav,
                           "md_dir": str(md.relative_to(ROOT)), "status": st,
                           "key": leg_key(cav, mode, snap, lig, replica)}
                    if st == "ok":
                        x = md / "md.xtc"
                        stt = x.stat()
                        row["xtc_bytes"] = stt.st_size
                        row["xtc_mtime"] = time.strftime(
                            "%Y-%m-%dT%H:%M:%S", time.localtime(stt.st_mtime))
                    if pr and pr.get("status"):
                        row["pipeline_status"] = pr["status"]
                    rows.append(row)
    return rows


# ------------------------------------------------------------- worker (child) --

def _frame_sha(path, nbytes=1 << 20):
    """Cheap provenance fingerprint: sha256 of the first and last 1 MB."""
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
        if size > 2 * nbytes:
            fh.seek(-nbytes, os.SEEK_END)
            h.update(fh.read(nbytes))
    return h.hexdigest()[:24]


def _safe_rel(p):
    """Path recorded in the checkpoint, ROOT-relative WHEN POSSIBLE.

    REVIEW FINDING 9: this used Path.relative_to(ROOT) unguarded, so any
    --out-dir outside the repo root (or any relative one) raised ValueError
    inside the worker — AFTER the leg had already paid its full ~10 s of
    trajectory reading. Every leg failed that way, the report came out empty,
    and because failures are stored as ok=False every later resume retried them
    forever. A 122 GB job on a filesystem that is already 78% full is exactly
    the situation that invites --out-dir /mnt/somewhere-else.
    """
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def _resolve_under_root(p):
    """Inverse of _safe_rel: accept either a ROOT-relative or an absolute path."""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def _work(row, last_frac, keep_frames, frames_dir):
    """Runs in a child process. NEVER raises — a failure comes back as a record
    so one bad leg cannot abort the other 87."""
    import numpy as np
    from pipeline.utils_pcsi_star import analyze_leg, enable_readonly_mode, leg_summary, CUTOFF_A
    enable_readonly_mode()
    t0 = time.perf_counter()
    try:
        md = ROOT / row["md_dir"]
        res = analyze_leg(md / "md.xtc", md / "md.tpr", last_frac=last_frac,
                          keep_frames=keep_frames)
        fr = res.pop("_frames", None)
        if fr is not None and frames_dir:
            fd = Path(frames_dir)
            fd.mkdir(parents=True, exist_ok=True)
            npz = fd / (row["key"].replace("|", "_") + ".npz")
            np.savez_compressed(
                npz,
                **{f"c{c:g}": np.packbits(m, axis=1) for c, m in fr.items()},
                n_res=np.array([fr[list(fr)[0]].shape[1]]))
            res["frames_npz"] = _safe_rel(npz)
        res["xtc_sha_head_tail"] = _frame_sha(md / "md.xtc")
        res["summary_at_default"] = leg_summary(res, CUTOFF_A)
        return {**row, **res, "ok": True,
                "wall_s": round(time.perf_counter() - t0, 2)}
    except Exception as e:
        return {**row, "ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:],
                "wall_s": round(time.perf_counter() - t0, 2)}


# ------------------------------------------------------------------ assembly --

def assemble(inventory, legs, pipeline, cutoff, persistence, n_boot, seed):
    from pipeline.utils_pcsi_star import aggregate_cavity, leg_summary

    by_key = {r["key"]: r for r in legs if r.get("ok")}
    inv = {r["key"]: r for r in inventory}
    frame_idx = {}
    for cav in TARGETS:
        try:
            frame_idx[cav] = {i: s["frame_idx"]
                              for i, s in enumerate(pipeline[cav]["snapshots"])}
        except Exception:
            frame_idx[cav] = {}

    def leg_cell(cav, mode, snap, lig, replica=0):
        k = leg_key(cav, mode, snap, lig, replica)
        st = inv.get(k, {}).get("status", "absent")
        rec = by_key.get(k)
        if rec is None:
            # A leg that is on disk but not yet computed is NOT "ok" — fail closed.
            return {"k": None, "n": None, "status": ("uncomputed" if st == "ok" else st)}
        s = leg_summary(rec, cutoff, persistence)
        return {"k": s["k"], "n": s["n"], "f": s["f"], "status": "ok",
                "mean_contact_freq": s["mean_contact_freq"],
                "persistent_resids": s["persistent_resids"],
                # REVIEW FINDING 7: carried so aggregate_cavity can ASSERT that
                # every leg of a cavity shares one analysis window. These four
                # fields were recorded in the provenance list and never compared.
                "window": {"n_analyzed": rec.get("n_analyzed"),
                           "dt_ps": rec.get("dt_ps"),
                           "t_first_ps": rec.get("t_first_ps"),
                           "t_last_ps": rec.get("t_last_ps"),
                           "window_ns": rec.get("window_ns")}}

    # The (replica, snapshot) pairs that ACTUALLY EXIST, taken from the
    # inventory rather than assumed to be range(10).
    #
    # This loop used to iterate range(10) and build its keys with an implicit
    # replica 0. With snapshots spread across Phase 4 replicas, every leg from
    # replica >= 1 has a key this loop never asks for, so it would be dropped
    # from the aggregation SILENTLY — the cavity would be summarised from a
    # third of its data with no error anywhere.
    pairs_by_cav = {}
    for row in (inventory or []):
        pairs_by_cav.setdefault(row.get("cavity"), set()).add(
            (int(row.get("replica", 0) or 0), int(row.get("snapshot", 0) or 0)))

    out = {}
    for cav in TARGETS:
        others = [o for o in TARGETS if o != cav]
        pairs = sorted(pairs_by_cav.get(cav, set()))
        for mode in (["main", "dual"] if cav == "CD63" else ["main"]):
            snaps = {}
            for ordinal, (replica, snap) in enumerate(pairs):
                # The key must be a UNIQUE, SORTABLE INT: aggregate_cavity does
                # sorted(snapshots), and the snapshot index alone is only unique
                # within a replica (rep0_snapshot_0 and rep1_snapshot_0 both
                # have index 0). A mixed int/str key set would raise TypeError
                # in sorted(); an ordinal over the sorted (replica, index) pairs
                # is unique, ordered, and keeps the legacy single-replica case
                # numbered 0..n-1 exactly as before.
                snaps[ordinal] = {
                    "replica": replica,
                    "snapshot_index": snap,
                    "own": leg_cell(cav, mode, snap, cav, replica),
                    "cross": {o: leg_cell(cav, mode, snap, o, replica)
                              for o in others},
                }
            out[f"{cav}_{mode}"] = aggregate_cavity(
                cav, mode, snaps, others, n_boot=n_boot, seed=seed,
                frame_idx=frame_idx.get(cav))
    return out


def threshold_sweep(inventory, legs, pipeline, n_boot, seed):
    """All 9 (cutoff, persistence) cells, free from the checkpoint — no re-read.

    A verdict that survives only at 6 A / 0.5 is not a result.
    """
    from pipeline.utils_pcsi_star import CUTOFF_GRID, PERSISTENCE_GRID
    cells = {}
    for c in CUTOFF_GRID:
        for p in PERSISTENCE_GRID:
            agg = assemble(inventory, legs, pipeline, c, p, max(2000, n_boot // 5), seed)
            cells[f"cutoff{c:g}_pers{p:g}"] = {
                k: {"theta_hat": v["theta_hat"], "n_def": v["bookkeeping"]["n_def"],
                    "bca_lo": v["ci"].get("bca_lo"), "verdict": v["gate"]["verdict"]}
                for k, v in agg.items()}
    flips = {}
    base = cells["cutoff6_pers0.5"]
    for key in base:
        verdicts = {cell: cells[cell][key]["verdict"] for cell in cells}
        flips[key] = {"base_verdict": base[key]["verdict"],
                      "distinct_verdicts": sorted(set(verdicts.values())),
                      "flips": sorted(set(verdicts.values())) != [base[key]["verdict"]],
                      "by_cell": verdicts}
    return {"cells": cells, "verdict_stability": flips}


def variance_decomposition(legs, cavity_key, agg, persistence, max_legs=None):
    """Var_within vs Var_between — the number that says whether the budget goes
    to LONGER snapshots or MORE snapshots. Diagnostic only; the reported CI is
    the one-level snapshot bootstrap (adding a frames level would double-count).

    REVIEW FINDING 13: this took "the first 3 legs with stored frames in
    CHECKPOINT ORDER", which in a parallel run is worker completion order — i.e.
    a race. The same 12 real CD63 legs, with the checkpoint shuffled three ways,
    gave var_within_f = 6.606e-4 / 7.258e-4 / 2.068e-3: a 3.1x swing from line
    ordering alone, on the number that decides the budget lever. Two changes:
    candidates are sorted by leg key, and max_legs defaults to None (use EVERY
    leg of the cavity) so the answer does not depend on a cutoff either. The
    legs actually used are recorded and printed.
    """
    import numpy as np
    from pipeline.utils_pcsi_star import block_bootstrap_within_variance, delta_method_var_D

    cav = agg[cavity_key]
    v = cav.get("S_values") or []
    if len(v) < 2:
        return {"error": "n_def < 2"}
    var_total = float(np.var(v, ddof=1))
    cands = sorted(
        (r for r in legs
         if r.get("ok") and r.get("frames_npz")
         and f"{r['cavity']}|{r['mode']}" == cavity_key.replace("_", "|")),
        key=lambda r: r["key"])                 # DETERMINISTIC, not arrival order
    if max_legs is not None:
        cands = cands[:max_legs]
    within = []
    for rec in cands:
        d = np.load(_resolve_under_root(rec["frames_npz"]))
        nres = int(d["n_res"][0])
        M = np.unpackbits(d["c6"], axis=1)[:, :nres].astype(bool)
        within.append({"key": rec["key"],
                       **block_bootstrap_within_variance(M, persistence)})
    if not within:
        return {"error": "no per-frame matrices stored (rerun with --keep-frames)",
                "var_total_S": var_total}
    vw = float(np.mean([w["var_within_f"] for w in within]))
    # propagate Var(f) -> Var(D) by the delta method at the observed working point
    snaps = cav["snapshots"]
    pts = [(s["f_own"], min((c["f_cross"] for c in s["cross"].values()
                             if c["f_cross"] is not None), default=None))
           for s in snaps.values() if s.get("S") is not None]
    dv = [delta_method_var_D(a, b, vw, vw) for a, b in pts
          if a is not None and b is not None]
    dv = [x for x in dv if x is not None]
    var_within_S = float(np.mean(dv)) if dv else None
    return {
        "legs_used": [w["key"] for w in within],
        "n_legs_used": len(within),
        "leg_selection": "ALL frame-storing legs of the cavity, sorted by key "
                         "(deterministic; not checkpoint/completion order)",
        "var_within_f_per_leg": {w["key"]: w["var_within_f"] for w in within},
        "tau_int_frames": [w["tau_int_frames"] for w in within],
        "block_len_frames": [w["block_len_frames"] for w in within],
        "var_within_f": vw,
        "var_within_S": var_within_S,
        "var_total_S": var_total,
        "var_between_S": (var_total - var_within_S) if var_within_S is not None else None,
        "budget_lever": (None if var_within_S is None else
                         "LONGER snapshots (within-snapshot sampling dominates)"
                         if var_within_S > 0.5 * var_total else
                         "MORE snapshots (between-snapshot variability dominates)"),
    }


# -------------------------------------------------------------------- report --

def render(agg, inventory, meta, sweep=None, vardec=None):
    L = []
    A = L.append
    A("=" * 100)
    A("PCSI*  —  bounded persistent-contact selectivity contrast")
    A(f"  D[s,j] = (f_own - f_j)/(f_own + f_j) in [-1,+1];  S[s] = min_j D[s,j];  "
      f"theta = mean_s S[s]")
    A(f"  GATE = CONSERVATIVE {int(meta['level']*100)}% CI LOWER BOUND > 0, where that bound is")
    A(f"         gate_lo = min(BCa, percentile, Student-t) lower endpoints.")
    A(f"         It is NOT a certified 97.5% one-sided bound: BCa alone measured a 5.4% "
      f"false-positive")
    A(f"         rate at n=10 and 8.6% at n=5 against a nominal 2.5%. Each cavity below "
      f"prints the")
    A(f"         SIMULATED type-I error for its own n and shape; read that, not the "
      f"nominal level.")
    A(f"  scale equivalence: legacy PCSI 1.0->0  1.2->0.0909  1.5->0.20  2.0->0.333  inf->1.0")
    am = meta.get("anchor_gate") or {}
    if am.get("verified"):
        A(f"  ANCHOR: VERIFIED against {am.get('anchor_path')} frozen {am.get('frozen_at')} "
          f"(sha256={am.get('full_hash')}) — files unchanged, counts reproduce")
    elif am.get("enforced") is False and am.get("reason"):
        A(f"  ANCHOR: NOT CHECKED ({am['reason']}) — these numbers are UNANCHORED")
    else:
        A("  ANCHOR: state unknown — treat these numbers as UNANCHORED")
    A("  The v1 anchor (own 6 / CD81 0 / CD9 3, published as 'PCSI 2.00 STRONG') is")
    A("  DEAD: its trajectories were re-run after it was written. See code/anchor.py.")
    A("=" * 100)

    tally = {}
    for r in inventory:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    A("")
    A("LEG INVENTORY (status from disk evidence, not inference)")
    for k in sorted(tally):
        A(f"    {k:<12s} {tally[k]:3d}")
    A(f"    {'TOTAL':<12s} {sum(tally.values()):3d}   (expected grid = 120)")

    for key, v in agg.items():
        b = v["bookkeeping"]
        ci = v["ci"]
        g = v["gate"]
        A("")
        A("-" * 100)
        A(f"### {key}")
        A(f"  bookkeeping: n_snapshots={b['n_snapshots']} n_def={b['n_def']} "
          f"n_nobind={b['n_nobind']} n_own_absent={b['n_own_absent']} "
          f"n_no_cross={b['n_no_cross_data']} n_bound={b['n_bound']} "
          f"n_imputed_legs={b['n_imputed_legs']} n_missing_legs={b['n_missing_legs']}")
        A(f"  J_C (MEASURED arms) = {b['J_C'] or '(none)'}   "
          f"worst_case_over_n = {b['worst_case_over_n']}"
          + ("   *** NOT 'worst-case across off-targets' ***"
             if b["worst_case_over_n"] < 2 else ""))
        see = v.get("size_exclusion_evidence") or {}
        sx = {j: d["n_snapshots_size_excluded"] for j, d in see.items()
              if d["n_snapshots_size_excluded"]}
        if sx:
            A(f"  size-excluded arms (NOT in the primary min; D=+1 is never the "
              f"argmin so they would have made S a min-over-fewer-arms): {sx}")
        if v["theta_hat"] is None:
            A("  NO DEFINED SNAPSHOTS — no estimate.")
        else:
            A(f"  PCSI* theta_hat = {v['theta_hat']:+.4f}   median = {v['median']:+.4f}   "
              f"sd = {v['sd'] if v['sd'] is None else round(v['sd'],4)}   "
              f"cv = {v['cv'] if v['cv'] is None else round(v['cv'],3)}")
            A(f"  BCa  95% CI = [{ci.get('bca_lo'):+.4f}, {ci.get('bca_hi'):+.4f}]"
              if ci.get("bca_lo") is not None else "  BCa  95% CI = n/a")
            A(f"  perc 95% CI = [{ci.get('perc_lo'):+.4f}, {ci.get('perc_hi'):+.4f}]"
              if ci.get("perc_lo") is not None else "  perc 95% CI = n/a")
            A(f"  t    95% CI = [{ci.get('t_lo'):+.4f}, {ci.get('t_hi'):+.4f}]"
              if ci.get("t_lo") is not None else "  t    95% CI = n/a")
            if ci.get("gate_lo") is not None:
                A(f"  >> GATE BOUND gate_lo = {ci['gate_lo']:+.4f}  "
                  f"(most conservative of the three; binding method = "
                  f"{ci.get('gate_lo_method')})")
            te = v.get("type1_error") or {}
            if te.get("type1") is not None:
                A(f"  SIMULATED type-I error of this gate at n={te.get('n')}: "
                  f"{te['type1']:.3f} +- {te.get('mc_se',0):.3f} "
                  f"(nominal {te.get('nominal'):.3f}, "
                  f"{te.get('ratio_to_nominal',0):.1f}x)"
                  + ("   *** EXCEEDS LIMIT -> gate (d) fails ***"
                     if te.get("exceeds_limit") else ""))
            if ci.get("lb_disagreement") is not None:
                A(f"  |LB_BCa - LB_perc| = {ci['lb_disagreement']:.4f} "
                  f"(limit {0.05}) {'OK' if ci['lb_disagreement']<=0.05 else 'UNSTABLE'}")
            A(f"  mean-of-min = {v['mean_of_min']:+.4f}   min-of-mean = "
              f"{'n/a' if v['min_of_mean'] is None else format(v['min_of_mean'],'+.4f')}"
              f"   (mean-of-min is the conservative one; it gates)")
        A(f"  VERDICT: {g['verdict']}   label={g['label']}   "
          f"conditions={ {k: ('Y' if x else 'N') for k, x in g['conditions'].items()} }")
        for r in g["reasons"]:
            A(f"      ! {r}")
        A("  per cross target:")
        for j, pc in v["per_cross_target"].items():
            c = pc["ci"] or {}
            A(f"      {j:5s} n={pc['n']:2d} imputed={pc['n_imputed']} "
              f"mean D={pc['mean'] if pc['mean'] is None else round(pc['mean'],4)} "
              f"BCa=[{c.get('bca_lo')}, {c.get('bca_hi')}]")
        lc, lf = v["legacy_pcsi_counts"], v["legacy_pcsi_fractions"]
        A(f"  LEGACY PCSI (counts)   : n={lc['n']} mean_finite_only="
          f"{lc['mean_finite_only']} n_inf={lc['n_inf']} "
          f"n_pass={lc['n_pass_gt_1_2']} n_strong={lc['n_strong_gt_1_5']}")
        A(f"      {lc['bias_note']}")
        A(f"  LEGACY PCSI (fractions): n={lf['n']} mean_finite_only="
          f"{lf['mean_finite_only']} n_inf={lf['n_inf']}")
        wc = v.get("window_check") or {}
        A(f"  window check: {'CONSISTENT' if wc.get('consistent') else 'MISMATCH'} "
          f"({wc.get('n_distinct')} distinct window(s) among the legs used)"
          + ("" if wc.get("consistent") else "   *** gate (f) FAILS ***"))
        for grp in (wc.get("groups") or [])[:4]:
            A(f"      n_analyzed={grp['n_analyzed']} dt={grp['dt_ps']} ps "
              f"t=[{grp['t_first_ps']},{grp['t_last_ps']}] ps  "
              f"x{grp['n_legs']} legs")
        idg = v.get("imputation_diagnostic") or {}
        if idg.get("n_with_imputed_arm"):
            A(f"  IMPUTATION SPLIT (functional {idg.get('J_functional')}): "
              f"clean n={idg.get('n_clean')} theta="
              f"{'n/a' if idg.get('theta_clean') is None else format(idg['theta_clean'],'+.4f')}"
              f"   |   with-imputed-arm n={idg.get('n_with_imputed_arm')} theta="
              f"{'n/a' if idg.get('theta_with_imputed') is None else format(idg['theta_with_imputed'],'+.4f')}"
              f"   diff="
              f"{'n/a' if idg.get('difference') is None else format(idg['difference'],'+.4f')}")
            A(f"      {idg.get('interpretation')}")
        wi = v["without_imputed"]
        lnd = wi.get("legacy_null_diagnostic") or {}
        A(f"  (superseded) drop-imputed-terms diagnostic: n_def={wi['n_def']} "
          f"theta={wi['ci'].get('theta_hat')} verdict={wi['gate']['verdict']}; "
          f"max |change in S| = {lnd.get('max_abs_change_in_S')} — this comparison "
          f"is identically null by construction, see legacy_null_diagnostic.proof")
        st = v.get("stratified") or []
        if len(st) > 1:
            A("  STRATIFIED by which arms were MEASURED (strata are different "
              "functionals; never averaged together):")
            for s_ in st:
                A(f"      measured={s_['measured_arms']} excluded={s_['size_excluded_arms']} "
                  f"n={s_['n']} theta="
                  f"{'n/a' if s_.get('theta') is None else format(s_['theta'],'+.4f')} "
                  f"snapshots={s_['snapshots']}")
        ac = v["autocorrelation"]
        A(f"  snapshot autocorrelation (ordered by Phase 4 frame_idx): "
          f"S rho1={ac['S_lag1'].get('rho1')} n_eff={ac['S_lag1'].get('n_effective')}; "
          f"f_own rho1={ac['f_own_lag1'].get('rho1')}")
        A(f"      {ac['S_lag1'].get('interpretation','')}")
        p = v["power"]
        if "at_s_upper" in p:
            A(f"  POWER: d={p['at_s0']['d']:.3f} (s0={p['s0']:.4f}) -> n_sig="
              f"{p['at_s0']['n_sig']} n80={p['at_s0']['n_power80']} n90={p['at_s0']['n_power90']}")
            A(f"         variance-inflated s_upper={p['s_upper_80pct']:.4f} -> n_sig="
              f"{p['at_s_upper']['n_sig']} n80={p['at_s_upper']['n_power80']} "
              f"n90={p['at_s_upper']['n_power90']}   <-- BUDGET FIGURE")
            if p.get("budget_infeasible"):
                A(f"         {p.get('note')}")
            else:
                A(f"         budget: {p['budget_n_snapshots']} snapshots/cavity x 3 "
                  f"cavities x 160 ns = {p['budget_ns_phase5_three_cavities']} ns "
                  f"of Phase 5")
        else:
            A(f"  POWER: {p.get('error') or p.get('note')}")
        bd = v.get("budget") or {}
        if "error" not in bd and bd.get("n_median") is not None:
            A(f"  BUDGET INTERVAL (pilot bootstrap through the power formula) "
              f"<-- READ THIS, NOT THE POINT ESTIMATE")
            A(f"      n: median {bd['n_median']}  IQR {bd['n_q25']}-{bd['n_q75']}  "
              f"{int(bd['quantile']*100)}th pct {bd['n_at_quantile_uncapped']}  "
              f"max {bd['n_max']}   (cap {bd['cap']})")
            A(f"      theta_hat={bd['theta_hat']:+.4f} CI=[{bd['theta_ci_gate_lo']:+.4f},"
              f"{bd['theta_ci_gate_hi']:+.4f}]  "
              f"resamples with theta<=0: {bd['frac_undefined_theta_le_0']:.1%}")
            A(f"      -> {bd['interpretation']}")
            if bd.get("warning"):
                A(f"      !! {bd['warning']}")
        elif bd.get("error"):
            A(f"  BUDGET INTERVAL: {bd['error']}")
        epc = v.get("empirical_power_curve") or {}
        if epc.get("curve"):
            A(f"  empirical Pr[gate_lo>0] vs n': "
              + "  ".join(f"{k}:{x:.2f}" for k, x in sorted(epc["curve"].items())))
            A(f"      {epc.get('caveat')}")
        for n in v["notes"][:12]:
            A(f"  note: {n}")

    A("")
    A("ASSUMPTIONS PRINTED NEXT TO THE BUDGET NUMBER")
    A("  1. Snapshots i.i.d. — FALSE as the pipeline stands. _select_equilibrium_frames")
    A("     (phase5_rebinding.py:476-522) takes EVENLY SPACED frames from the last 50% of")
    A("     ONE Phase 4 trajectory; frame_idx [191..335] at 16 ns spacing, IDENTICAL for all")
    A("     three targets. If slow modes correlate over >~16 ns the CI is anti-conservative")
    A("     and more snapshots from the same run buy sub-sqrt(n) information — the lever is")
    A("     INDEPENDENT Phase 4 replicas (+350 ns each). See the rho1 lines above.")
    A("  2. CLT normality of the mean at the target n — poor for a bounded statistic that")
    A("     saturates at +1; hence the empirical Pr[LB>0] curve.")
    A("  3. Imputed size-excluded legs are CONSTANTS contributing zero variance; if they")
    A("     carry the signal, s0 is understated. They are therefore EXCLUDED from the")
    A("     primary functional entirely (an imputed +1 can never be the argmin of a min,")
    A("     so including them silently made S a min over fewer arms in some snapshots),")
    A("     and appear only in the stratified table and the imputation split above.")
    A("  4. Effect size stable between pilot and re-run (same monomers, cavity, window).")
    A("  5. The CI is NOT certified at its nominal level at these n. Every cavity above")
    A("     carries a SIMULATED type-I error for its own realised n and shape; a cavity")
    A("     whose measured rate exceeds 0.05 fails gate (d) whatever its interval says.")
    A("  6. The budget is a RATIO estimator (n ~ s^2/theta^2) and is heavy-tailed: on a")
    A("     10-snapshot pilot its IQR spans 5-8x and its tail runs five orders of")
    A("     magnitude. Read the BUDGET INTERVAL, never the single integer.")
    if sweep:
        A("")
        A("THRESHOLD SENSITIVITY (9 cells: cutoff 5/6/7 A x persistence 0.4/0.5/0.6)")
        for k, f in sweep["verdict_stability"].items():
            A(f"  {k:12s} base={f['base_verdict']:12s} "
              f"{'VERDICT FLIPS: ' + ','.join(f['distinct_verdicts']) if f['flips'] else 'stable'}")
    if vardec:
        A("")
        A("VARIANCE DECOMPOSITION (diagnostic block bootstrap)")
        for k, d in vardec.items():
            A(f"  {k}: {json.dumps(d, default=str)[:300]}")
    return "\n".join(L)


# ---------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    # CHANGED DEFAULT (REVIEW FINDING 2): was `ROOT / "reports_v2"`.
    # verify_phase5 reads <results_root>/reports/pcsi_star_summary.json, so the
    # old default wrote where the gate does not look AND clobbered the frozen
    # reports_v2/ recompute. Now defaults to this experiment's reports dir.
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help=f"where to write pcsi_star_summary.json "
                         f"(default: {DEFAULT_OUT_DIR}, which is exactly where "
                         f"verify_phase5 looks)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="only N legs (smoke test)")
    ap.add_argument("--only", default=None, help="substring filter on the leg key")
    ap.add_argument("--last-frac", type=float, default=0.5)
    # REVIEW FINDING 12: these were free `type=float`, but the per-leg records
    # only ever store the CUTOFF_GRID / PERSISTENCE_GRID cells. --cutoff 8 used
    # to crash the aggregation with an uncaught KeyError AFTER the whole 88-leg
    # compute had been paid for. `choices` rejects it in argparse, before any
    # trajectory is opened.
    from pipeline.utils_pcsi_star import CUTOFF_GRID, PERSISTENCE_GRID
    ap.add_argument("--cutoff", type=float, default=6.0, choices=list(CUTOFF_GRID),
                    help=f"contact cutoff in A; one of {list(CUTOFF_GRID)} "
                         f"(the grid the per-leg records store)")
    ap.add_argument("--persistence", type=float, default=0.5,
                    choices=list(PERSISTENCE_GRID),
                    help=f"persistence fraction; one of {list(PERSISTENCE_GRID)}")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-frames", action="store_true", default=True,
                    help="store packed per-frame contact matrices (needed for the "
                         "variance decomposition); ~30 kB/leg")
    ap.add_argument("--no-keep-frames", dest="keep_frames", action="store_false")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--skip-anchor-gate", action="store_true",
                    help="launch WITHOUT verifying the regression anchor. The "
                         "run is then unanchored: nothing proves the code still "
                         "reproduces known counts on unchanged files. Recorded "
                         "in the output metadata.")
    ap.add_argument("--anchor-no-hash", action="store_true",
                    help="anchor gate checks size+mtime only, not sha256")
    a = ap.parse_args()

    # ---- LAUNCH INTERLOCK -------------------------------------------------
    # The v1 anchor died silently: it pinned three integers with no provenance,
    # the trajectories were re-run underneath it, and nothing in the pipeline
    # noticed. A 1600 ns/target budget decision was about to be taken on top of
    # that. Nothing that reads a trajectory starts until the anchor verifies.
    anchor_meta = {"enforced": not a.skip_anchor_gate}
    if a.aggregate_only:
        anchor_meta = {"enforced": False,
                       "reason": "--aggregate-only reads no trajectory"}
        print("anchor gate: skipped (--aggregate-only reads no trajectory)")
    elif a.skip_anchor_gate:
        anchor_meta["reason"] = "--skip-anchor-gate"
        print("=" * 70)
        print("WARNING: --skip-anchor-gate. This run is UNANCHORED. No evidence")
        print("that the code reproduces known counts on files known unchanged.")
        print("Any number it produces must be labelled unanchored downstream.")
        print("=" * 70)
    else:
        from pipeline import utils_anchor as _A
        try:                                  # may be overridden out of tree
            _ap = str(_A.ANCHOR_PATH.relative_to(ROOT))
        except ValueError:
            _ap = str(_A.ANCHOR_PATH)
        print(f"anchor gate: verifying {_ap} "
              f"({'sha256' if not a.anchor_no_hash else 'size+mtime only'}) ...")
        ok, rep = _A.verify(full_hash=not a.anchor_no_hash, verbose=True)
        anchor_meta.update({
            "anchor_path": _ap,
            "verified": ok, "stages": rep.get("stages"),
            "frozen_at": rep.get("frozen_at"),
            "anchor_git_commit": rep.get("anchor_git_commit"),
            "full_hash": not a.anchor_no_hash})
        if not ok:
            print("\nANCHOR GATE FAILED — refusing to launch.")
            print(json.dumps(rep.get("stages", rep), indent=2))
            print("\n  stage 1 = the trajectory changed -> re-freeze on purpose:")
            print("            python code/freeze_anchor.py --freeze --force")
            print("  stage 2 = the code changed -> fix the code, do NOT re-freeze")
            print("  stage 3 = a leg that had no trajectory now has one")
            print("\n  Diagnosis of a stage-1 failure: python code/anchor_forensics.py")
            print("  Override (records itself in the output): --skip-anchor-gate")
            return 3
        print(f"anchor gate: PASS  {rep['stages']}\n")

    # REVIEW FINDING 9: resolve to an absolute path here so a relative --out-dir
    # cannot reach the workers and fail 88 legs one by one.
    out_dir = Path(a.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Distinct from the earlier prototype's pcsi_star_legs.jsonl, whose per-leg
    # record schema is incompatible with assemble().
    ckpt = out_dir / "pcsi_star_legs_v2.jsonl"
    frames_dir = out_dir / "frames"

    # ---- REVIEW FINDING 11: single-instance lock --------------------------
    # Two drivers on one --out-dir were undetected: no lock, no PID file. Both
    # recomputed every leg (100% duplicated I/O and CPU on a 122 GB job) and
    # both raced to write summary/report — the likely way a checkpoint line gets
    # torn. The docs invite it ("re-running the same command recomputes only
    # legs that are absent"), so the collision has to be refused, not documented.
    import fcntl
    lock_path = out_dir / ".pcsi_star.lock"
    # "a+", never "w": mode "w" truncates on OPEN, i.e. before flock is even
    # attempted, so the loser of the race would erase the winner's identity and
    # the refusal message would name nobody. Truncate only after acquiring.
    lock_fh = open(lock_path, "a+")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            lock_fh.seek(0)
            holder = lock_fh.read().strip() or "(holder has not identified itself yet)"
        except Exception:
            holder = "(unknown)"
        print(f"ERROR: another run_pcsi_star.py already holds {lock_path}\n"
              f"       holder: {holder}\n"
              f"       Two instances on one --out-dir duplicate all work and can "
              f"tear the checkpoint.\n"
              f"       Wait for it, or use a different --out-dir.", file=sys.stderr)
        return 4
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(f"pid={os.getpid()} host={os.uname().nodename} "
                  f"started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    lock_fh.flush()

    pipeline = json.loads(
        (PHASE5_DIR / "phase5_rebinding_results.json").read_text())
    inventory = build_inventory(pipeline)
    (out_dir / "pcsi_star_inventory.json").write_text(json.dumps(inventory, indent=2))

    present = [r for r in inventory if r["status"] == "ok"]
    if a.only:
        present = [r for r in present if a.only in r["key"]]
    if a.limit:
        present = present[:a.limit]

    done = {}
    if ckpt.exists() and not a.restart:
        # REVIEW FINDING 10: json.loads was unguarded here, so ONE torn line
        # anywhere in the checkpoint made the driver refuse to start at all —
        # including --aggregate-only. The JSONL that IS the crash-recovery
        # mechanism was itself a single point of failure, and its records are
        # ~8 kB, well above the 4 kB atomic-append size, so a SIGKILL mid-flush
        # reaches it. An unparseable line just means that leg is recomputed.
        n_bad = 0
        for ln, line in enumerate(ckpt.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                key = r["key"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                n_bad += 1
                print(f"  WARNING: checkpoint line {ln} unreadable "
                      f"({type(e).__name__}: {str(e)[:80]}) — skipped, that leg "
                      f"will be recomputed")
                continue
            done[key] = r                       # last write wins on retry
        print(f"checkpoint: {len(done)} legs on file ({ckpt})"
              + (f", {n_bad} unreadable line(s) skipped" if n_bad else ""))

    if not a.aggregate_only:
        todo = [r for r in present
                if r["key"] not in done or not done[r["key"]].get("ok")]
        print(f"{len(present)} legs present, {len(todo)} to compute, "
              f"{a.workers} workers")
        t0 = time.perf_counter()
        # ---- REVIEW FINDING 8: survive a worker dying ---------------------
        # fut.result() was unguarded, so a single OOM-killed / segfaulted worker
        # raised BrokenProcessPool out of main(): exit 1, NO report, and every
        # pending leg cancelled rather than just the one in flight. Verified by
        # kill -9 on one child of a 6-leg run. The checkpoint bounded the loss,
        # but a 122 GB job that cannot finish unattended is not usable.
        # Now: per-future failures are recorded as ok=False and the run
        # continues; a broken pool is REBUILT and the outstanding legs resubmitted.
        by_key = {r["key"]: r for r in todo}
        pending = [r["key"] for r in todo]
        n_done, attempt, pool_deaths = 0, 0, 0
        with open(ckpt, "a") as fh:
            def _record(rec):
                nonlocal n_done
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())            # survive a hard kill
                done[rec["key"]] = rec
                n_done += 1
                el = time.perf_counter() - t0
                k = (rec.get("summary_at_default") or {}).get("k")
                print(f"[{n_done}/{len(todo)}] {rec['key']:28s} "
                      f"{'ok  ' if rec['ok'] else 'FAIL'} k={k} "
                      f"{rec['wall_s']:6.1f}s | elapsed {el/60:5.1f}m "
                      f"eta {el/max(n_done,1)*(len(todo)-n_done)/60:5.1f}m",
                      flush=True)

            while pending and attempt <= len(todo) + 2:
                attempt += 1
                batch, pending = pending, []
                try:
                    with ProcessPoolExecutor(max_workers=a.workers) as ex:
                        futs = {ex.submit(_work, by_key[k], a.last_frac,
                                          a.keep_frames, str(frames_dir)): k
                                for k in batch}
                        for fut in as_completed(futs):
                            key = futs[fut]
                            try:
                                _record(fut.result())
                            except BrokenProcessPool:
                                raise
                            except Exception as e:
                                # One leg blew up in a way the worker could not
                                # catch. Record it and keep going.
                                _record({**by_key[key], "ok": False, "wall_s": 0.0,
                                         "error": f"{type(e).__name__}: {e}",
                                         "traceback": traceback.format_exc()[-2000:]})
                except BrokenProcessPool as e:
                    pool_deaths += 1
                    lost = [k for k in batch if k not in done or not done[k].get("ok")]
                    print(f"  WARNING: worker pool died ({e}); "
                          f"{len(lost)} leg(s) unfinished — rebuilding the pool "
                          f"and resubmitting (pool death #{pool_deaths})",
                          flush=True)
                    pending = lost
                    if pool_deaths > 3:
                        print("  ERROR: pool died more than 3 times; giving up on "
                              f"{len(pending)} leg(s) and reporting on the rest.")
                        for k in pending:
                            _record({**by_key[k], "ok": False, "wall_s": 0.0,
                                     "error": "BrokenProcessPool x4 — leg abandoned"})
                        pending = []
        print(f"compute wall: {(time.perf_counter()-t0)/60:.2f} min"
              + (f"  ({pool_deaths} worker-pool death(s) recovered)"
                 if pool_deaths else ""))

    legs = list(done.values())
    fails = [r for r in legs if not r.get("ok")]
    if fails:
        # FAIL-CLOSED: any leg that errors makes its cavity UNDECIDABLE, it never
        # silently drops out of the denominator.
        print(f"WARNING: {len(fails)} legs failed: {[r['key'] for r in fails]}")

    agg = assemble(inventory, legs, pipeline, a.cutoff, a.persistence,
                   a.n_boot, a.seed)

    sweep = None
    if not a.no_sweep and len(legs) > 3:
        sweep = threshold_sweep(inventory, legs, pipeline, a.n_boot, a.seed)

    vardec = {}
    for key in agg:
        if agg[key]["bookkeeping"]["n_def"] >= 2:
            # max_legs=None: every frame-storing leg of the cavity, sorted by key.
            vardec[key] = variance_decomposition(legs, key, agg, a.persistence,
                                                 max_legs=None)

    # empirical power curve, per cavity
    from pipeline.utils_pcsi_star import empirical_power_curve
    for key, v in agg.items():
        if v["bookkeeping"]["n_def"] >= 3:
            v["empirical_power_curve"] = empirical_power_curve(
                v["S_values"], seed=a.seed).meta()

    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    import numpy as np
    import MDAnalysis as mda
    meta = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": commit, "mdanalysis": mda.__version__,
            "numpy": np.__version__, "python": sys.version.split()[0],
            "seed": a.seed, "n_boot": a.n_boot, "level": 0.95,
            "cutoff_A": a.cutoff, "persistence_frac": a.persistence,
            "last_frac": a.last_frac,
            "n_legs_ok": len(legs) - len(fails), "n_legs_failed": len(fails),
            "failed_keys": [r["key"] for r in fails],
            "anchor_gate": anchor_meta}

    payload = {"meta": meta, "inventory_tally": {}, "cavities": agg,
               "threshold_sweep": sweep, "variance_decomposition": vardec,
               "per_leg": [{k: r.get(k) for k in
                            ("key", "cavity", "mode", "snapshot", "ligand",
                             "status", "ok", "n_frames", "n_analyzed", "dt_ps",
                             "t_first_ps", "t_last_ps", "window_ns",
                             "n_protein_residues", "n_monomer_atoms",
                             "monomer_resnames", "xtc_bytes", "xtc_mtime",
                             "xtc_sha_head_tail", "summary_at_default", "wall_s")}
                           for r in legs]}
    for r in inventory:
        payload["inventory_tally"][r["status"]] = \
            payload["inventory_tally"].get(r["status"], 0) + 1

    (out_dir / "pcsi_star_summary.json").write_text(
        json.dumps(payload, indent=2, default=str))
    report = render(agg, inventory, meta, sweep, vardec)
    (out_dir / "pcsi_star_report.txt").write_text(report)
    print("\n" + report)
    print(f"\nwrote {out_dir/'pcsi_star_summary.json'}")
    print(f"wrote {out_dir/'pcsi_star_report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
