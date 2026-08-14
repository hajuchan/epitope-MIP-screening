"""PCSI* — bounded persistent-contact selectivity contrast for Phase 5 rebinding.

This module is the authoritative implementation of the replacement statistic.
It contains, in one place:

  LEVEL 1  vectorised persistent-contact analysis for a single rebinding leg
           (bit-for-bit identical to the legacy per-residue loop in
            code/analyze_persistent_contacts.py, which is NOT modified)
  LEVEL 2  snapshot-level bounded contrast
               D[s,j] = (f_own - f_j) / (f_own + f_j)      in [-1, +1]
               S[s]   = min_j D[s,j]                       worst-case off-target
  LEVEL 3  cavity-level point estimate, BCa + percentile bootstrap CI,
           the five-condition gate, and the power / budget formula.

Every edge case named in the specification is implemented below and carries a
comment of the form `# EDGE CASE Ex:` so the handling can be audited against the
spec line by line.

READ-ONLY GUARANTEE: nothing in this module writes under results/. MDAnalysis
itself does (offset caches + filelocks next to every .xtc); enable_readonly_mode()
suppresses that and is called by every entry point.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import capped_distance

# ----------------------------------------------------------------- constants --

PROTEIN_SEL = "protein"              # ALL atoms, hydrogens INCLUDED — verbatim legacy.
MONOMER_SEL = "not protein and not resname SOL HOH NA CL TIP3 WAT"

CUTOFF_A = 6.0                       # contact distance, strict "<"
PERSISTENCE_FRAC = 0.5               # persistent iff freq_r > this, STRICT
LAST_FRAC = 0.5                      # analysis window = last 50% of frames

# Threshold-sensitivity sweep grid (spec: 9 cells). Computed in ONE trajectory
# pass by searching at max(CUTOFF_GRID) and filtering; the persistence grid is
# then free because per-residue frequencies are stored.
CUTOFF_GRID = (5.0, 6.0, 7.0)
PERSISTENCE_GRID = (0.4, 0.5, 0.6)

# Window sanity limits (spec: hard-error, not warn).
MIN_WINDOW_NS = 20.0
MAX_STRIDE_PS = 20.0

# Gate constants, sourced from the pipeline's own config so they cannot drift.
BOOTSTRAP_N_RESAMPLES = 10000        # code/pipeline/config.py:122
BOOTSTRAP_CI = 0.95                  # code/pipeline/config.py:123
REBINDING_N_SNAPSHOTS = 10           # code/pipeline/config.py:627
# code/pipeline/phase5_rebinding.py:427 — reuse, do not invent a new threshold.
BINDING_MIN_SNAPSHOTS = max(1, REBINDING_N_SNAPSHOTS // 3)      # = 3
MIN_N_DEF = 5                        # gate (b): BCa jackknife unusable below 5
MAX_LB_DISAGREEMENT = 0.05           # gate (d): BCa vs percentile
STRONG_THETA = 0.20                  # label only; == legacy PCSI 1.5
MODERATE_THETA = 1.0 / 11.0          # label only; == legacy PCSI 1.2 (0.0909...)

TARGETS = ("CD63", "CD81", "CD9")

# --- gate calibration (REVIEW FINDING 1) -------------------------------------
# The BCa lower bound is NOT a conservative 97.5% bound at n=5..10. Measured
# false-positive rate of "LB_BCa > 0" at an exact null (scratchpad/adv3):
#     n= 5 -> 0.0862   n= 8 -> 0.0668   n=10 -> 0.0535   n=20 -> 0.0385
# against a nominal 0.025, i.e. 2.1x-3.4x anti-conservative, and the percentile
# interval shares the failure mode (0.056 vs 0.054 at n=10) so gate (d) cannot
# catch it. A Student-t interval measured 0.030-0.034 over the same scenarios.
# Two changes follow:
#   (iii) the gate runs on the MOST CONSERVATIVE of the lower bounds below,
#         not on BCa alone;
#   (i)   every cavity carries a SIMULATED type-I error for its own realised
#         (n, shape), and a cavity whose measured rate exceeds
#         MAX_MEASURED_TYPE1 fails gate (d) however clean its interval looks.
GATE_LB_METHODS = ("bca", "percentile", "t")
# The conservative gate_lo IS the fix, and it is validated by DIRECT simulation
# from the generating process (scratchpad/adv3): gate PASS rate at an exact null
# fell from 0.0858 to 0.0257 at n=5 and 0.0535 to 0.0295 at n=10, against a
# nominal 0.025. gate_type1_error() below is a per-cavity MONITOR on top of that,
# not the protection itself, and it must not be mistaken for a sharp instrument:
# across 120 independent null pilots its own spread is
#     n= 5  mean 0.0471  sd 0.0551  median 0.0325  max 0.3200
#     n=10  mean 0.0314  sd 0.0203  median 0.0275  max 0.1225
# so it CANNOT resolve 0.025 from 0.05. A 0.05 threshold would block 23% (n=5)
# and 15% (n=10) of perfectly well-calibrated cavities on noise alone. 0.10 is
# chosen as a GROSS-failure guard: measured spurious block rate 8.3% at n=5 and
# 1.7% at n=10. The number is printed next to every verdict regardless.
MAX_MEASURED_TYPE1 = 0.10            # gate (d): gross miscalibration only
TYPE1_N_SIM = 1000                   # smoothed-null simulations per cavity
TYPE1_N_BOOT = 1000                  # inner resamples inside each simulation

# --- budget feasibility (REVIEW FINDING 4) -----------------------------------
# n scales as s^2/theta^2 and theta from an n0=10 pilot can land arbitrarily
# close to 0, so the point estimate is an unbounded heavy-tailed ratio (measured
# IQR spread 5-8x, tail to 1e12). Budgets are reported as an INTERVAL and capped.
MAX_FEASIBLE_N = 40                  # 40 snapshots x 3 cavities x 160 ns = 19.2 us
BUDGET_QUANTILE = 0.80               # upper quantile of the bootstrapped n


# ------------------------------------------------------------ read-only mode --

def enable_readonly_mode():
    """Stop MDAnalysis writing offset caches / filelocks under results/.

    XDRBaseReader._load_offsets takes a filelock next to every .xtc and rewrites
    the .npz offset cache whenever ctime/size/n_atoms disagree. results/ holds
    504 GB of irreplaceable MD output and is off limits for writes. This patch
    keeps MDAnalysis' own staleness validation verbatim — a cache that fails
    validation is IGNORED and offsets are rebuilt in memory, never written back.
    """
    from os.path import isfile, getctime, getsize
    from MDAnalysis.coordinates.XDR import (XDRBaseReader, offsets_filename,
                                            read_numpy_offsets)

    if getattr(XDRBaseReader._load_offsets, "_readonly_patched", False):
        return

    def _load_offsets(self):
        fname = offsets_filename(self.filename)
        if isfile(fname):
            data = read_numpy_offsets(fname)
            if data:
                try:
                    if (getctime(self.filename) == data["ctime"]
                            and getsize(self.filename) == data["size"]
                            and self._xdr.n_atoms == data["n_atoms"]):
                        self._xdr.set_offsets(data["offsets"])
                        return
                except KeyError:
                    pass
        self._read_offsets(store=False)          # in memory only

    _load_offsets._readonly_patched = True
    XDRBaseReader._load_offsets = _load_offsets

    orig_read = XDRBaseReader._read_offsets
    if not getattr(orig_read, "_no_store_patched", False):
        def _read(self, store=False):
            return orig_read(self, store=False)
        _read._no_store_patched = True
        XDRBaseReader._read_offsets = _read


# ================================================== LEVEL 1: contact analysis ==

def _residue_index_map(protein):
    """Map every protein atom to a compact residue slot 0..n_res-1.

    Uses the AtomGroup's own residue ordering (residues.ix / atoms.resindices),
    NOT resid arithmetic, so non-contiguous, repeated or negative resids are safe.
    """
    residues = protein.residues
    res_ix = residues.ix
    order = np.argsort(res_ix, kind="stable")
    lookup = np.empty(int(res_ix.max()) + 1, dtype=np.int64)
    lookup[res_ix[order]] = order
    return lookup[protein.atoms.resindices].astype(np.int64), residues.resids.astype(np.int64)


def analyze_leg(traj_path, top_path, cutoffs=CUTOFF_GRID, last_frac=LAST_FRAC,
                method="capped", keep_frames=False, check_window=True):
    """Persistent-contact analysis of ONE rebinding leg, all cutoffs in one pass.

    Returns a dict with, for each cutoff: per-residue contact frequency, the
    persistent-residue count at every persistence threshold in PERSISTENCE_GRID,
    and mean_contact_freq. Plus window/provenance metadata.

    Semantics are the legacy ones, exactly:
      * selections ALL-ATOM (hydrogens included) — do NOT switch to heavy atoms,
        measured to change CD63-own 7->5 and CD9 10->8, i.e. a different statistic
      * window start = int(n_frames * (1 - last_frac));  denominator n_analyzed
      * contact iff min over (all atoms of r) x (all monomer atoms) < cutoff, STRICT
      * minimum image via box = ts.dimensions, passed per frame
      * per-residue counts folded into a dict keyed by resid, SUMMING duplicates
    """
    cutoffs = tuple(float(c) for c in cutoffs)
    cmax = max(cutoffs)

    u = mda.Universe(str(top_path), str(traj_path))
    protein = u.select_atoms(PROTEIN_SEL)
    monomers = u.select_atoms(MONOMER_SEL)
    if len(protein) == 0 or len(monomers) == 0:
        raise ValueError(f"empty selection: protein={len(protein)} monomer={len(monomers)}")

    n_frames = len(u.trajectory)
    start = int(n_frames * (1 - last_frac))
    n_analyzed = n_frames - start
    if n_analyzed <= 0:
        raise ValueError("empty analysis window")

    atom_ridx, resids = _residue_index_map(protein)
    n_res = len(resids)

    hits = {c: np.zeros(n_res, dtype=np.int64) for c in cutoffs}
    frames = ({c: np.zeros((n_analyzed, n_res), dtype=bool) for c in cutoffs}
              if keep_frames else None)

    t_first = t_last = None
    for i, ts in enumerate(u.trajectory[start:]):
        if i == 0:
            t_first = float(ts.time)
        t_last = float(ts.time)

        if method == "capped":
            pairs, dists = capped_distance(
                protein.positions, monomers.positions,
                max_cutoff=cmax, box=ts.dimensions, return_distances=True)
            for c in cutoffs:
                if len(pairs) == 0:
                    continue
                # capped_distance is INCLUSIVE ("<="); the legacy predicate is
                # STRICT ("<"). Re-filter or a residue sitting at exactly the
                # cutoff would be counted by one implementation and not the other.
                keep = dists < c
                if not keep.any():
                    continue
                hit_res = np.unique(atom_ridx[pairs[keep, 0]])
                hits[c][hit_res] += 1
                if frames is not None:
                    frames[c][i, hit_res] = True
        elif method == "dense":
            # The literal "one distance_array per frame" form. Kept as an
            # independent cross-check on the capped grid search.
            d = distance_array(protein.positions, monomers.positions,
                               box=ts.dimensions)
            row_min = d.min(axis=1)                       # per protein ATOM
            for c in cutoffs:
                in_cut = row_min < c
                if not in_cut.any():
                    continue
                per_res = np.bincount(atom_ridx[in_cut], minlength=n_res)
                sel = per_res > 0
                hits[c][sel] += 1
                if frames is not None:
                    frames[c][i, sel] = True
        else:
            raise ValueError(f"unknown method {method!r}")

    dt = float(u.trajectory.dt)
    window_ns = (t_last - t_first) / 1000.0 if n_analyzed > 1 else 0.0
    if check_window:
        # EDGE CASE E6: legs with different frame counts / lengths. f_L is
        # normalised by n_analyzed so the MEAN frequency is window-invariant, but
        # persistence is a STEP function of freq_r whose sampling noise scales as
        # 1/sqrt(n_analyzed) — unequal windows are different estimators. Hard-error.
        if window_ns < MIN_WINDOW_NS:
            raise ValueError(f"usable window {window_ns:.2f} ns < {MIN_WINDOW_NS} ns "
                             f"({traj_path})")
        if dt > MAX_STRIDE_PS:
            raise ValueError(f"stride {dt} ps > {MAX_STRIDE_PS} ps ({traj_path})")

    out = {
        "n_frames": int(n_frames), "start_frame": int(start),
        "n_analyzed": int(n_analyzed), "dt_ps": dt,
        "t_first_ps": t_first, "t_last_ps": t_last, "window_ns": window_ns,
        "total_ns": float(n_frames * dt / 1000.0),
        "n_protein_atoms": int(len(protein)),
        "n_protein_residues": int(n_res),
        "n_monomer_atoms": int(len(monomers)),
        "monomer_resnames": sorted(set(map(str, monomers.resnames))),
        "n_atoms_total": int(len(u.atoms)),
        "method": method,
        "cutoffs": list(cutoffs),
        "persistence_grid": list(PERSISTENCE_GRID),
        "by_cutoff": {},
    }
    for c in cutoffs:
        # Fold slots into a resid-keyed dict, SUMMING duplicates — legacy
        # semantics (the legacy dict is keyed by res.resid, so two residue
        # objects sharing a resid collapse into one key). Not triggered on the
        # current data (resids unique and contiguous 1..N) but preserved.
        counts = {}
        for slot, rid in enumerate(resids):
            rid = int(rid)
            counts[rid] = counts.get(rid, 0) + int(hits[c][slot])
        freq = {rid: n / n_analyzed for rid, n in counts.items()}
        out["by_cutoff"][f"{c:g}"] = {
            "cutoff_A": c,
            "contact_freq": {str(k): float(v) for k, v in freq.items()},
            "total_residues": len(freq),
            "mean_contact_freq": float(np.mean(list(freq.values()))),
            "n_persistent": {f"{p:g}": int(sum(1 for v in freq.values() if v > p))
                             for p in PERSISTENCE_GRID},
            "persistent_resids": {
                f"{p:g}": sorted(int(k) for k, v in freq.items() if v > p)
                for p in PERSISTENCE_GRID},
        }
    if frames is not None:
        out["_frames"] = frames          # not JSON-serialisable; driver packs it
    return out


def leg_summary(res, cutoff=CUTOFF_A, persistence=PERSISTENCE_FRAC):
    """Pull the (k_L, n_L, f_L) triple for one (cutoff, persistence) cell."""
    b = res["by_cutoff"][f"{cutoff:g}"]
    k = b["n_persistent"][f"{persistence:g}"]
    n = b["total_residues"]
    return {
        "k": int(k), "n": int(n),
        "f": (k / n) if n else 0.0,
        "mean_contact_freq": b["mean_contact_freq"],
        "persistent_resids": b["persistent_resids"][f"{persistence:g}"],
    }


def compute_persistent_contacts_fast(traj_path, top_path, cutoff_A=CUTOFF_A,
                                     persistence_frac=PERSISTENCE_FRAC,
                                     last_frac=LAST_FRAC, method="capped"):
    """Drop-in signature match for the legacy function: (freq, n_persistent)."""
    res = analyze_leg(traj_path, top_path, cutoffs=(cutoff_A,), last_frac=last_frac,
                      method=method, check_window=False)
    b = res["by_cutoff"][f"{cutoff_A:g}"]
    freq = {int(k): v for k, v in b["contact_freq"].items()}
    return freq, int(sum(1 for v in freq.values() if v > persistence_frac))


# ============================================ LEVEL 2: snapshot-level contrast ==

def contrast(f_own, f_cross):
    """D = (f_own - f_cross) / (f_own + f_cross), bounded [-1, +1].

    Returns None when UNDEFINED. Never returns 0.0 for a 0/0 input.
    """
    if f_own is None or f_cross is None:
        return None
    den = f_own + f_cross
    if den == 0.0:
        # EDGE CASE E3 (pair half): 0/0 is UNDEFINED. Returning 0.0 here would
        # be the single most damaging shortcut in the whole design — it asserts
        # "own and cross bound equally", a positive claim of equal affinity,
        # when in fact NOTHING bound. The caller drops the pair and counts it.
        return None
    # EDGE CASE E1: f_own > 0, f_cross == 0 -> +1 exactly. No special case, no
    #   infinity, no clamping. This is the size-exclusion / non-binding endpoint
    #   the legacy metric wrote as inf and then silently dropped from its mean.
    # EDGE CASE E2: f_own == 0, f_cross > 0 -> -1 exactly. Genuine
    #   anti-selectivity evidence; kept, never excluded, or theta_hat is biased
    #   upward exactly where the cavity is worst.
    return (f_own - f_cross) / den


def snapshot_value(f_own, cross, J_C):
    """S[s] = min over j in J_C of D[s,j], plus full bookkeeping.

    `cross` maps target -> {"f": float|None, "status": str}. `status` is disk
    evidence: "ok" | "size_excluded" | "md_failed" | "absent".

    Returns a record. `S` is None when the snapshot contributes nothing.
    """
    per_j, used, n_imputed, n_missing = {}, [], 0, 0
    for j in J_C:
        info = cross.get(j) or {"f": None, "status": "absent"}
        st = info.get("status", "absent")
        fj = info.get("f")

        if st == "size_excluded":
            # EDGE CASE E4: leg absent because the cross target was genuinely
            # size-excluded BEFORE MD ran (recorded status SIZE_EXCLUDED,
            # corroborated by no em.gro). f_cross = 0 by construction, whose
            # contrast limit is exactly +1. This is EVIDENCE, not missing data —
            # but it is a CONSTANT and contributes ZERO variance, so it is
            # flagged and theta_hat is reported with and without it.
            d = contrast(f_own, 0.0)
            per_j[j] = {"f_cross": 0.0, "D": d, "source": "size_excluded",
                        "imputed": True}
            n_imputed += 1
            if d is not None:
                used.append(d)
        elif st == "ok" and fj is not None:
            d = contrast(f_own, fj)
            per_j[j] = {"f_cross": fj, "D": d, "source": "md", "imputed": False}
            if d is None:
                # EDGE CASE E3 (pair half): f_own == 0 AND this f_cross == 0.
                # Drop this pair only; the snapshot may still be defined via
                # another cross target.
                per_j[j]["note"] = "0/0 undefined -> pair dropped"
            else:
                used.append(d)
        else:
            # EDGE CASE E5: leg absent for any other reason (NVT crash,
            # incomplete run). Genuinely MISSING data. Do NOT impute, do NOT
            # score. Imputing +1 here would manufacture perfect selectivity out
            # of a GROMACS crash — 26 of the 32 absent legs are exactly this.
            per_j[j] = {"f_cross": None, "D": None, "source": st, "imputed": False}
            n_missing += 1

    rec = {"f_own": f_own, "cross": per_j, "n_cross_used": len(used),
           "n_imputed": n_imputed, "n_missing": n_missing,
           "S": None, "status": None}

    if used:
        # The min is taken AFTER the ratio, never before: all D[s,j] are computed
        # and stored, because the per-cross-target breakdown is a required output
        # and cannot be reconstructed from the min afterwards.
        rec["S"] = min(used)
        rec["status"] = "defined"
        return rec

    # Nothing usable. Distinguish the two reasons — they are different budget facts.
    all_zero = (f_own == 0.0) and all(
        (v["f_cross"] == 0.0) for v in per_j.values() if v["f_cross"] is not None)
    any_present = any(v["f_cross"] is not None for v in per_j.values())
    if any_present and all_zero:
        # EDGE CASE E3 (snapshot half): f_own == 0 AND every available f_cross
        # == 0 -> nothing bound at all. Classify no_binding, EXCLUDE from
        # theta_hat and from the bootstrap pool, and report n_nobind as a
        # first-class number gated by condition (c). Excluding silently would
        # let a cavity that fails in 8 of 10 snapshots be reported as selective
        # on the 2 that worked.
        rec["status"] = "no_binding"
    else:
        rec["status"] = "no_cross_data"
    return rec


# ========================================== LEVEL 3: cavity-level inference ====

def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam), so scipy is not a hard dependency."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


_T_975 = {                            # Student-t inverse CDF at p=0.975 by df
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
    7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788,
    13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009,
    19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639,
    25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423,
    35: 2.0301, 40: 2.0211, 50: 2.0086, 60: 2.0003, 120: 1.9799,
}


def _t_ppf(q, df):
    """Student-t quantile. scipy when available; exact table at the 0.975 point
    the gate actually uses; Cornish-Fisher expansion otherwise."""
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(q, df))
    except Exception:
        pass
    if abs(q - 0.975) < 1e-12:
        if df in _T_975:
            return _T_975[df]
        ks = sorted(_T_975)
        if df > ks[-1]:
            return 1.959964
        lo = max(k for k in ks if k <= df)
        hi = min(k for k in ks if k >= df)
        if lo == hi:
            return _T_975[lo]
        w = (df - lo) / (hi - lo)
        return _T_975[lo] * (1 - w) + _T_975[hi] * w
    z = _norm_ppf(q)
    return z + (z ** 3 + z) / (4.0 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96.0 * df ** 2)


def bootstrap_ci(values, n_boot=BOOTSTRAP_N_RESAMPLES, level=BOOTSTRAP_CI, seed=0):
    """BCa (primary) + percentile (mandatory cross-check) CI for the MEAN.

    RESAMPLING UNIT IS THE SNAPSHOT, one level, i.i.d. with replacement. Frames
    are never resampled here: within-leg frames are strongly autocorrelated
    (2501 frames at 10 ps, contact correlation times in the ns range, so ~10-25
    effective samples per leg) and a frame bootstrap would report an interval
    roughly sqrt(2501/15) ~ 13x too narrow. More fundamentally the inferential
    question is "would a DIFFERENT snapshot also give a positive contrast", so
    the snapshot is the exchangeable unit.

    STILL OPEN — SNAPSHOTS ARE NOT FULLY EXCHANGEABLE ACROSS REPLICAS.
    The i.i.d. assumption above is exactly true only for snapshots drawn from
    INDEPENDENT trajectories. Within one Phase 4 trajectory the snapshot
    observable has measured lag-1 autocorrelation rho1 = 0.23, so 10 snapshots
    from one leg carry n_eff ~ 6.3 independent samples and this interval is
    correspondingly too narrow — in the same anti-conservative direction as the
    small-n BCa behaviour documented below, and compounding with it.

    Phase 4 now runs PHASE4_N_REPLICAS independent trajectories and Phase 5
    spreads its snapshots across them (directories named rep<r>_snapshot_<i>),
    which REDUCES this error: with the budget split over 3 replicas, most pairs
    of snapshots are genuinely independent. The `replica` field is carried
    through run_pcsi_star's inventory so a hierarchical (cluster) bootstrap
    resampling replicas first and snapshots within them can be implemented
    without re-running anything.

    THAT CORRECTION IS NOT IMPLEMENTED HERE. This function still treats all
    snapshots as one flat i.i.d. sample. Deliberate: changing the estimator is a
    change to the statistic, not an integration fix, and the gate's calibration
    was characterised against this estimator. Treat the interval as a lower
    bound on the true width until the cluster bootstrap lands.

    NOTE ON THE GATE — CORRECTED, REVIEW FINDING 1. The lower endpoint of the
    two-sided `level` interval is a NOMINAL 97.5% one-sided bound and it is NOT
    conservative at the sample sizes this gate permits. Simulation at an exact
    null (scratchpad/adv3_null_and_scipy.py, 4000 cavities per n) measured the
    false-positive rate of "LB_BCa > 0" as

        n= 5  0.0862      n= 8  0.0668      n=10  0.0535      n=20  0.0385

    against the nominal 0.025 — 2.1x to 3.4x anti-conservative. This is intrinsic
    small-n BCa behaviour, not an implementation error: the same estimator from
    scipy.stats.bootstrap(method='BCa') agrees to |dLB| <= 0.0054. The percentile
    interval misses at essentially the same rate (0.056 vs 0.054 at n=10), so
    gate (d) — which compares the two — cannot detect it.

    Three lower bounds are therefore computed and the GATE USES THE MINIMUM
    (`gate_lo`, methods in GATE_LB_METHODS): BCa, percentile, and Student-t.
    The t bound measured 0.030-0.034 at n=5-10 in the same scenarios and is
    usually the binding one. The studentized (bootstrap-t) interval is computed
    as a diagnostic but is EXCLUDED from the gate minimum by default: its
    endpoints blow up when a resample's SE approaches 0, which at n=5-10 makes
    it erratic rather than conservative.

    Do not "fix" the endpoint to the 5% quantile; that loosens the gate further.
    Read `gate_lo` and the cavity's measured `type1_error`, never `bca_lo` alone.

    Nothing is clipped. A lower bound at -1 or an upper at +1 is real saturation.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    alpha = (1.0 - level) / 2.0
    base = {"n": int(n), "n_boot": int(n_boot), "seed": int(seed), "level": level,
            "alpha_lo": alpha}
    if n == 0:
        return {**base, "theta_hat": None, "degenerate": True,
                "note": "no defined snapshots"}
    theta = float(v.mean())
    out = {**base, "theta_hat": theta,
           "median": float(np.median(v)),
           "sd": float(v.std(ddof=1)) if n > 1 else None,
           "cv": (float(v.std(ddof=1) / abs(v.mean()))
                  if n > 1 and v.mean() != 0 else None)}

    # EDGE CASE E8: degenerate resample — every S[s] identical (e.g. every
    # snapshot an imputed +1). The BCa acceleration is a jackknife ratio that is
    # 0/0 here and the percentile interval collapses to a point. A zero-width
    # interval is an artefact of imputed constants, NOT evidence of precision,
    # so it is marked degenerate and must not be reported as a passing CI even
    # though its lower bound formally exceeds 0.
    if n == 1 or np.all(v == v[0]):
        return {**out, "bca_lo": theta, "bca_hi": theta,
                "perc_lo": theta, "perc_hi": theta,
                "t_lo": theta, "t_hi": theta,
                "gate_lo": theta, "gate_lo_method": "degenerate",
                "degenerate": True,
                "note": "all snapshot values identical (or n=1): zero-width interval"}

    # --- Student-t interval on the mean -------------------------------------
    # Better calibrated than either bootstrap interval at n=5-20 (measured
    # one-sided LB error 0.030-0.034 vs BCa 0.054-0.086). Usually the binding
    # term in gate_lo, which is the point.
    sd_v = float(v.std(ddof=1))
    se = sd_v / math.sqrt(n)
    tq = _t_ppf(1 - alpha, n - 1)
    out["se"] = se
    out["t_crit"] = tq
    out["t_lo"], out["t_hi"] = theta - tq * se, theta + tq * se

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_samples = v[idx]
    boot = boot_samples.mean(axis=1)

    p_lo, p_hi = np.quantile(boot, [alpha, 1 - alpha])
    out["perc_lo"], out["perc_hi"] = float(p_lo), float(p_hi)

    # --- studentized (bootstrap-t), DIAGNOSTIC ONLY --------------------------
    # Excluded from gate_lo: se_b -> 0 on a degenerate resample sends t* to
    # +-inf, so at n=5-10 this interval is erratic rather than conservative.
    with np.errstate(invalid="ignore", divide="ignore"):
        se_b = boot_samples.std(axis=1, ddof=1) / math.sqrt(n)
        tstar = (boot - theta) / se_b
    tstar = tstar[np.isfinite(tstar)]
    if se > 0 and tstar.size >= max(50, n_boot // 20):
        q_hi, q_lo = np.quantile(tstar, [1 - alpha, alpha])
        out["stud_lo"] = float(theta - q_hi * se)
        out["stud_hi"] = float(theta - q_lo * se)
        out["stud_n_finite"] = int(tstar.size)
    else:
        out["stud_lo"] = out["stud_hi"] = None
        out["stud_n_finite"] = int(tstar.size)

    # --- BCa -----------------------------------------------------------------
    prop = float(np.mean(boot < theta))
    if prop <= 0.0 or prop >= 1.0:
        out.update(bca_lo=float(p_lo), bca_hi=float(p_hi), degenerate=True,
                   note="z0 undefined (all bootstrap means on one side); "
                        "BCa fell back to percentile")
        return _finalize_gate_lo(out)
    z0 = _norm_ppf(prop)

    jack = (v.sum() - v) / (n - 1)           # leave-one-out means
    dev = jack.mean() - jack
    den = 6.0 * (float(np.sum(dev ** 2)) ** 1.5)
    a = float(np.sum(dev ** 3)) / den if den > 0 else 0.0

    def _end(q):
        z = _norm_ppf(q)
        adj = z0 + (z0 + z) / (1.0 - a * (z0 + z))
        return float(np.quantile(boot, min(max(_norm_cdf(adj), 0.0), 1.0)))

    out.update(z0=z0, acceleration=a, bca_lo=_end(alpha), bca_hi=_end(1 - alpha),
               degenerate=False)
    # BCa's acceleration is jackknife-estimated and unstable for n < 15-20. At
    # n=10 it is at the edge of validity, hence gate condition (d).
    out["lb_disagreement"] = abs(out["bca_lo"] - out["perc_lo"])
    out["bca_unstable_small_n"] = bool(n < 15)
    return _finalize_gate_lo(out)


def _finalize_gate_lo(out):
    """gate_lo = min over GATE_LB_METHODS of the available lower bounds.

    REVIEW FINDING 1. BCa alone is 2.1x-3.4x anti-conservative at n=5-10, so the
    gate takes the most conservative of the three bounds rather than trusting
    any single one. The upper bound (used only for the FAIL verdict, i.e. the
    claim "positively cross-reactive") is made symmetrically conservative by
    taking the MAXIMUM, so neither direction is claimed too readily.
    """
    key = {"bca": ("bca_lo", "bca_hi"), "percentile": ("perc_lo", "perc_hi"),
           "t": ("t_lo", "t_hi"), "studentized": ("stud_lo", "stud_hi")}
    los, his = {}, {}
    for m in GATE_LB_METHODS:
        klo, khi = key[m]
        vlo, vhi = out.get(klo), out.get(khi)
        if vlo is not None and math.isfinite(vlo):
            los[m] = float(vlo)
        if vhi is not None and math.isfinite(vhi):
            his[m] = float(vhi)
    if los:
        m = min(los, key=los.get)
        out["gate_lo"], out["gate_lo_method"] = los[m], m
    else:
        out["gate_lo"], out["gate_lo_method"] = None, None
    if his:
        m = max(his, key=his.get)
        out["gate_hi"], out["gate_hi_method"] = his[m], m
    else:
        out["gate_hi"], out["gate_hi_method"] = None, None
    out["lower_bounds"] = los
    return out


def gate_type1_error(values, n=None, n_sim=TYPE1_N_SIM, n_boot=TYPE1_N_BOOT,
                     seed=12345, level=BOOTSTRAP_CI):
    """MEASURED false-positive rate of gate condition (a) for this cavity.

    REVIEW FINDING 1, fix option (i): "calibrate the decision threshold by
    simulation under the fitted null and print the simulated type-I error next
    to every verdict".

    The null is the cavity's OWN distribution shifted to mean zero, so the
    simulation inherits the realised n, spread, skew and saturation mass at +1
    rather than assuming a shape. Under it theta = 0 exactly, so every gate PASS
    is a false positive by construction and the pass rate IS the type-I error of
    the gate as configured (gate_lo over GATE_LB_METHODS).

    SMOOTHED, not the naive empirical bootstrap. Resampling n points from n
    atoms with replacement produces many repeated values, which shrinks the
    within-sample sd, narrows the t interval and INFLATES the apparent error
    rate — the estimator would then indict the gate for an artefact of its own
    discreteness. Measured against direct simulation from the generating
    process (the exact null of scratchpad/adv3, p_own=0.064624 vs 0.05/0.05):

        n      direct truth    naive atoms         smoothed (this)
        5         0.0260       0.0407 (sd 0.045)   0.0263 (sd 0.018)
        10        0.0293       0.0383 (sd 0.026)   0.0322 (sd 0.020)

    A Gaussian kernel of Silverman bandwidth h is added and the draw is divided
    by sqrt(1 + h^2/s^2) so total variance is preserved. Samples are NOT
    recentred individually — the POPULATION has mean 0 and each simulated
    sample's mean must be free to vary, or theta_hat would be 0 by construction
    and the measured rate would be a meaningless 0.

    Read the per-pilot sd of ~0.02 alongside the estimate: this quantity is
    itself noisy at n=10, which is why MAX_MEASURED_TYPE1 is a coarse 2x-nominal
    guard rail rather than a fine threshold.

    Interpretation: nominal is (1-level)/2 = 0.025. A measured rate materially
    above that means the CI is optimistic at this n and the honest response is
    more snapshots, not a relabelled verdict — hence gate (d) consumes this.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return {"error": "n < 2", "type1": None, "nominal": (1.0 - level) / 2.0}
    n = int(n or v.size)
    if np.all(v == v[0]):
        return {"type1": 0.0, "mc_se": 0.0, "n": n, "n_sim": 0,
                "nominal": (1.0 - level) / 2.0, "degenerate": True,
                "note": "zero-variance pilot: null cannot be simulated"}
    centred = v - v.mean()                       # population mean exactly 0
    s = float(centred.std(ddof=1))
    iqr = float(np.subtract(*np.percentile(centred, [75, 25])))
    spread = min(s, iqr / 1.34) if iqr > 0 else s
    h = 0.9 * spread * (len(centred) ** -0.2)    # Silverman
    scale = math.sqrt(1.0 + (h * h) / (s * s)) if s > 0 else 1.0
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(n_sim):
        sub = (rng.choice(centred, size=n, replace=True)
               + h * rng.standard_normal(n)) / scale
        ci = bootstrap_ci(sub, n_boot=n_boot, level=level, seed=int(seed + 1 + i))
        lo = ci.get("gate_lo")
        if lo is not None and lo > 0 and not ci.get("degenerate"):
            hits += 1
    p = hits / n_sim
    nominal = (1.0 - level) / 2.0
    return {"type1": p, "mc_se": math.sqrt(max(p * (1 - p), 1e-12) / n_sim),
            "n": n, "n_sim": n_sim, "n_boot": n_boot, "nominal": nominal,
            "ratio_to_nominal": p / nominal if nominal else None,
            "methods": list(GATE_LB_METHODS),
            "bandwidth": h, "variance_scale": scale,
            "exceeds_limit": bool(p > MAX_MEASURED_TYPE1),
            "per_pilot_sd_note": "this estimate carries a per-pilot sd of ~0.02 "
                                 "at n=10; treat MAX_MEASURED_TYPE1 as a coarse "
                                 "guard rail, not a fine threshold",
            "note": "null = smoothed bootstrap of the pilot shifted to mean 0; "
                    "every PASS under it is a false positive by construction"}


# ------------------------------------------------------------- power / budget --

_CHI2_020_LOWER = {                  # chi2 inverse CDF at p=0.20, df = n0-1
    2: 0.4463, 3: 1.0052, 4: 1.6488, 5: 2.3425, 6: 3.0701, 7: 3.8223,
    8: 4.5936, 9: 5.3801, 10: 6.1791, 11: 6.9887, 12: 7.8073, 13: 8.6339,
    14: 9.4673, 15: 10.3070, 16: 11.1521, 17: 12.0023, 18: 12.8570,
    19: 13.7158, 20: 14.5784,
}


def _chi2_ppf_020(df):
    if df in _CHI2_020_LOWER:
        return _CHI2_020_LOWER[df]
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(0.20, df))
    except Exception:                      # Wilson-Hilferty fallback
        z = _norm_ppf(0.20)
        return df * (1 - 2.0 / (9 * df) + z * math.sqrt(2.0 / (9 * df))) ** 3


def power_requirement(theta0, s0, n0):
    """Snapshots required for the CI lower bound to clear 0.

    Normal approximation to the BCa lower bound: LB ~= theta - z*s/sqrt(n),
    z = 1.959964 (the 2.5% point of the two-sided 95% interval, MATCHING THE
    GATE — not 1.645).

    n_sig is the significance-only number and is a TRAP: at n_sig a fresh run
    clears 0 only about half the time. Budget on n(80%) or n(90%).

    s0 is itself estimated from n0 points with SE(s)/s = 1/sqrt(2(n0-1)) — 24%
    at n0=10, on a quantity that gets SQUARED. The mandatory inflation
    substitutes the upper 80% one-sided confidence limit on sigma; quote THAT
    number as the budget figure.
    """
    z = 1.959964
    out = {"theta0": theta0, "s0": s0, "n0": n0, "z": z}
    if s0 is None or theta0 is None or n0 is None or n0 < 2:
        return {**out, "error": "insufficient pilot data"}
    if s0 == 0:
        # Zero observed variance: usually every value imputed/identical.
        return {**out, "d": None, "degenerate": True,
                "note": "s0 = 0 (all snapshot values identical); n_required "
                        "undefined — variance is an artefact, not precision"}
    if theta0 <= 0:
        return {**out, "d": theta0 / s0, "note":
                "theta0 <= 0: no positive effect to power for; the question is "
                "not 'how many snapshots' but whether the cavity is selective at all"}

    s_up = s0 * math.sqrt((n0 - 1) / _chi2_ppf_020(n0 - 1))
    res = {**out, "s_upper_80pct": s_up}
    for tag, s in (("at_s0", s0), ("at_s_upper", s_up)):
        d = theta0 / s
        res[tag] = {
            "s": s, "d": d,
            "n_sig": int(math.ceil(z ** 2 / d ** 2)),
            "n_power80": int(math.ceil((z + 0.84162) ** 2 / d ** 2)),
            "n_power90": int(math.ceil((z + 1.28155) ** 2 / d ** 2)),
        }
    # Gate (b) floors any recommendation at MIN_N_DEF.
    raw = max(MIN_N_DEF, res["at_s_upper"]["n_power80"])
    # REVIEW FINDING 4: n ~ s^2/theta^2 is unbounded and heavy-tailed. Emitting
    # 1e12 snapshots as a bare integer is worse than refusing: cap it and say so.
    res["budget_n_snapshots_uncapped"] = raw
    res["budget_infeasible"] = bool(raw > MAX_FEASIBLE_N)
    res["budget_n_snapshots"] = min(raw, MAX_FEASIBLE_N)
    if res["budget_infeasible"]:
        res["note"] = (
            f"effect too small to power at any feasible n: the formula asks for "
            f"{raw} snapshots/cavity (> cap {MAX_FEASIBLE_N} = "
            f"{MAX_FEASIBLE_N*3*160} ns). d={res['at_s_upper']['d']:.3f}. Do not "
            f"budget this cavity from this number; the effect, not the sample "
            f"size, is the problem.")
    # 3 rebinding legs x 50 ns + 1 removal test x 10 ns per snapshot per cavity.
    res["ns_per_snapshot_per_cavity"] = 160
    res["budget_ns_phase5_three_cavities"] = res["budget_n_snapshots"] * 3 * 160
    return res


def budget_interval(values, n_boot=1000, seed=0, quantile=BUDGET_QUANTILE):
    """The budget as an INTERVAL, by propagating pilot uncertainty through the
    power formula. REVIEW FINDING 4.

    The mandated s_upper inflation corrects only the DENOMINATOR's uncertainty.
    The dominant error term is theta_hat itself, which a 10-snapshot pilot
    estimates with an IQR spread of 5-8x and a tail five orders of magnitude
    long. Each bootstrap resample of the pilot is pushed through
    power_requirement, giving a distribution over n rather than one draw from it.

    `n_recommended` is the upper `quantile` of that distribution: budget so that
    the run is adequately powered for that fraction of plausible pilots.
    `frac_undefined` is the fraction of resamples with theta <= 0, where the
    budget question does not have an answer at all — if that is material the
    cavity is not ready to be budgeted.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n < 2:
        return {"error": f"n={n} < 2"}
    rng = np.random.default_rng(seed)
    ns, undefined, infeasible = [], 0, 0
    for b in range(n_boot):
        s = rng.choice(v, size=n, replace=True)
        sd = float(s.std(ddof=1))
        th = float(s.mean())
        if sd == 0 or th <= 0:
            undefined += 1
            continue
        r = power_requirement(th, sd, n)
        if "at_s_upper" not in r:
            undefined += 1
            continue
        if r["budget_infeasible"]:
            infeasible += 1
        ns.append(r["budget_n_snapshots_uncapped"])
    out = {"n_pilot": int(n), "n_boot": int(n_boot), "quantile": quantile,
           "frac_undefined_theta_le_0": undefined / n_boot,
           "frac_infeasible_gt_cap": infeasible / n_boot,
           "cap": MAX_FEASIBLE_N}
    if not ns:
        return {**out, "error": "every resample gave theta <= 0 or zero variance: "
                                "the budget question has no answer for this cavity"}
    arr = np.asarray(ns, dtype=float)
    out.update({
        "n_median": int(np.median(arr)),
        "n_q25": int(np.quantile(arr, 0.25)),
        "n_q75": int(np.quantile(arr, 0.75)),
        "n_at_quantile_uncapped": int(np.quantile(arr, quantile)),
        "n_max": int(arr.max()),
        "spread_iqr_ratio": float(np.quantile(arr, 0.75) / max(np.quantile(arr, 0.25), 1)),
    })
    rec = max(MIN_N_DEF, out["n_at_quantile_uncapped"])
    out["n_recommended"] = min(rec, MAX_FEASIBLE_N)
    out["recommendation_infeasible"] = bool(rec > MAX_FEASIBLE_N
                                            or out["frac_undefined_theta_le_0"] > 0.10)
    out["interpretation"] = (
        f"budget {out['n_recommended']} snapshots/cavity "
        f"(pilot-bootstrap {int(quantile*100)}th pct; median {out['n_median']}, "
        f"IQR {out['n_q25']}-{out['n_q75']}, max {out['n_max']})"
        if not out["recommendation_infeasible"] else
        f"NOT BUDGETABLE from this pilot: "
        f"{out['frac_undefined_theta_le_0']:.0%} of resamples have theta<=0 and "
        f"the {int(quantile*100)}th pct asks for {out['n_at_quantile_uncapped']} "
        f"snapshots (cap {MAX_FEASIBLE_N})")
    return out


class PowerCurve(dict):
    """{n' -> Pr[gate_lo > 0]}, with its Monte-Carlo band carried alongside.

    Subclasses dict so it stays a plain n'->rate mapping for callers and for
    json.dumps, while `mc_se` / `caveat` / `n_draws` ride along as attributes.
    """
    __slots__ = ("mc_se", "caveat", "n_draws", "n_pilot", "statistic")

    def meta(self):
        return {"curve": dict(self), "mc_se": self.mc_se, "caveat": self.caveat,
                "n_draws": self.n_draws, "n_pilot": self.n_pilot,
                "statistic": self.statistic}


def empirical_power_curve(values, n_boot=2000, n_draws=200, seed=0,
                          level=BOOTSTRAP_CI):
    """Pr[gate lower bound > 0] vs n', by RESAMPLING the pilot with replacement.

    The normal formula is not trustworthy for a bounded statistic that piles mass
    at +1; quote whichever n is LARGER, this or the formula — but read the
    Monte-Carlo band first.

    REVIEW FINDING 5, two defects fixed:

      1. The endpoint n'=n used draws=1, i.e. ONE deterministic CI on the pilot
         itself, so it reported exactly 0.0 or 1.0 — not a probability — and made
         the curve non-monotone. Every n' now uses n_draws resamples WITH
         replacement, so the endpoint is a rate like every other point. (Without
         replacement at n'=n can only return the pilot itself; that is the bug.)

      2. The curve carried no uncertainty. Across 300 independent pilots its
         per-pilot sd was 0.27 while tracking truth well ON AVERAGE (n'=5: mean
         0.748 sd 0.272 vs true 0.725). On the SINGLE real 10-snapshot pilot the
         spec tells you to read it from, that sd makes it nearly uninformative.
         `mc_se` below is only the binomial error of the resampling; the
         dominant 0.27 is pilot-to-pilot and is reported in `caveat`.

    Returns a PowerCurve: a plain {n' -> rate} mapping carrying .mc_se and
    .caveat as attributes.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    rng = np.random.default_rng(seed)
    curve, mc_se = {}, {}
    for np_ in range(3, n + 1):
        hits = 0
        for k in range(n_draws):
            # WITH replacement at every n', including n'=n.
            sub = rng.choice(v, size=np_, replace=True)
            ci = bootstrap_ci(sub, n_boot=n_boot, level=level, seed=int(seed + k))
            lo = ci.get("gate_lo")
            if lo is not None and not ci.get("degenerate") and lo > 0:
                hits += 1
        p = hits / n_draws
        curve[np_] = p
        mc_se[np_] = math.sqrt(max(p * (1 - p), 1e-12) / n_draws)
    out = PowerCurve(curve)
    out.mc_se = mc_se
    out.n_draws = int(n_draws)
    out.n_pilot = int(n)
    out.statistic = ("Pr[gate_lo > 0]  (gate_lo = min over "
                     + ", ".join(GATE_LB_METHODS) + ")")
    out.caveat = ("per-pilot sd of this curve is ~0.27 (measured over 300 "
                  "independent pilots); it CANNOT adjudicate between the "
                  "formula's n and its own on a single 10-snapshot pilot. "
                  "mc_se is resampling error only and understates that.")
    return out


def lag1_autocorrelation(series):
    """Lag-1 autocorrelation and the implied n_effective = n*(1-rho)/(1+rho).

    Snapshots are NOT i.i.d. as the pipeline stands: _select_equilibrium_frames
    (phase5_rebinding.py:476-522) takes EVENLY SPACED frames from the last 50% of
    ONE Phase 4 trajectory — frame_idx [191..335] at 16 ns spacing, identical for
    all three targets. If the cavity's slow modes correlate over >~16 ns the
    snapshot bootstrap UNDERSTATES variance and the CI is anti-conservative, and
    the correct budget lever is INDEPENDENT Phase 4 replicas, not more snapshots.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 3:
        return {"n": int(n), "rho1": None, "n_effective": None,
                "note": "n < 3, autocorrelation not estimable"}
    xc = x - x.mean()
    den = float(np.sum(xc ** 2))
    if den == 0:
        return {"n": int(n), "rho1": None, "n_effective": None,
                "note": "zero variance"}
    rho = float(np.sum(xc[:-1] * xc[1:]) / den)
    n_eff = n * (1 - rho) / (1 + rho) if rho > -1 else None
    return {"n": int(n), "rho1": rho,
            "n_effective": (min(float(n_eff), float(n)) if n_eff else None),
            "interpretation": ("positively correlated: bootstrap CI is "
                               "anti-conservative, budget more Phase 4 replicas"
                               if rho > 0.2 else
                               "no material positive correlation at lag 1")}


def integrated_autocorr_time(series):
    """Integrated autocorrelation time tau_int by the initial-positive-sequence rule."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    xc = x - x.mean()
    den = float(np.sum(xc ** 2))
    if n < 10 or den == 0:
        return 1.0
    tau, k = 0.5, 1
    while k < n // 3:
        r = float(np.sum(xc[:-k] * xc[k:]) / den)
        if r <= 0:
            break
        tau += r
        k += 1
    return float(2 * tau)


def block_bootstrap_within_variance(frame_matrix, persistence=PERSISTENCE_FRAC,
                                    n_boot=400, seed=0):
    """Var_within(f_L) by moving-block bootstrap over FRAMES of one leg.

    Diagnostic only — NOT used to build the reported CI. The snapshot-level
    variance already contains the within-snapshot sampling component; adding a
    frames-within-snapshot level to the interval would double-count it. This is
    run to DECOMPOSE the variance: if Var_within dominates, the budget lever is
    LONGER snapshots; if Var_between dominates, it is MORE snapshots.

    Block length = 5 x the integrated autocorrelation time of the per-frame
    persistent-residue count.
    """
    M = np.asarray(frame_matrix, dtype=bool)      # (n_frames, n_res)
    nf, nr = M.shape
    counts = M.sum(axis=1).astype(float)
    tau = integrated_autocorr_time(counts)
    L = max(1, min(nf // 4, int(round(5 * tau))))
    nblk = int(math.ceil(nf / L))
    rng = np.random.default_rng(seed)
    fs = np.empty(n_boot)
    starts_max = nf - L
    for b in range(n_boot):
        st = rng.integers(0, starts_max + 1, size=nblk)
        idx = (st[:, None] + np.arange(L)[None, :]).ravel()[:nf]
        freq = M[idx].mean(axis=0)
        fs[b] = float(np.sum(freq > persistence)) / nr
    return {"tau_int_frames": tau, "block_len_frames": int(L),
            "n_blocks": int(nblk), "n_boot": int(n_boot),
            "f_mean": float(fs.mean()), "var_within_f": float(fs.var(ddof=1)),
            "sd_within_f": float(fs.std(ddof=1))}


def delta_method_var_D(f_own, f_cross, var_own, var_cross):
    """Var(D) from Var(f) by the delta method on D = (a-b)/(a+b).

    dD/da = 2b/(a+b)^2,  dD/db = -2a/(a+b)^2.
    """
    den = (f_own + f_cross) ** 2
    if den == 0:
        return None
    ga = 2.0 * f_cross / den
    gb = -2.0 * f_own / den
    return float(ga ** 2 * var_own + gb ** 2 * var_cross)


# ------------------------------------------------------------- legacy metrics --

def legacy_pcsi(k_own, k_cross_list):
    """own / max(cross) on RAW COUNTS — the metric being replaced, unchanged.

    Reproduced verbatim (including its use of max()) so the published numbers stay
    traceable. Returns inf when every cross count is 0, exactly as before.
    """
    vals = [k for k in k_cross_list if k is not None]
    if not vals:
        return None
    mx = max(vals)
    return (k_own / mx) if mx > 0 else float("inf")


def legacy_summary(pcsis):
    """Legacy aggregation, reproducing code/midrun_pcsi_check.py:67 EXACTLY.

    The defect being replaced is visible right here: `mean_finite` averages only
    the FINITE values — an infinite value means every cross target had zero
    persistent contacts, i.e. the MOST selective snapshots — while n_pass and
    n_strong count over ALL snapshots including the infinities. Two statistics
    over two different populations. Both are emitted, plus n_inf, so the
    down-bias is a NUMBER rather than an argument.
    """
    vals = [p for p in pcsis if p is not None]
    fin = [p for p in vals if math.isfinite(p)]
    n_inf = sum(1 for p in vals if p == math.inf)
    return {
        "n": len(vals), "n_inf": n_inf,
        "mean_finite_only": float(np.mean(fin)) if fin else None,   # legacy line 67
        "std_finite_only": float(np.std(fin)) if len(fin) > 1 else None,
        "median_all_inf_ok": (float(np.median([p if math.isfinite(p) else np.inf
                                               for p in vals]))
                              if vals else None),
        "n_pass_gt_1_2": sum(1 for p in vals if p > 1.2),           # over ALL n
        "n_strong_gt_1_5": sum(1 for p in vals if p > 1.5),         # over ALL n
        "bias_note": ("mean_finite_only excludes the %d most-selective snapshot(s); "
                      "n_pass/n_strong include them. Different populations."
                      % n_inf) if n_inf else "no infinities in this cavity",
    }


# ------------------------------------------------------------------- the gate --

def evaluate_gate(ci, n_def, n_bound, J_C, full_cross_set, n_nobind=0,
                  n_imputed=0, n_missing=0, notes=None, type1=None,
                  window_consistent=True, window_detail=None):
    """The six-condition conjunction. No single number can carry it.

    (a) CONSERVATIVE 95% CI lower bound > 0 strictly — `gate_lo`, the minimum of
        the BCa, percentile and Student-t lower bounds. NOT BCa alone: BCa is
        2.1x-3.4x anti-conservative at n=5-10 (REVIEW FINDING 1). The point
        estimate does not gate. n_pass / n_strong do not gate and are not
        computed as gate inputs at all.
    (b) n_def >= 5, else UNDECIDABLE (BCa jackknife unusable below 5).
    (c) n_bound >= max(1, REBINDING_N_SNAPSHOTS//3) = 3.
    (d) STABILITY: |LB_BCa - LB_percentile| <= 0.05, not degenerate, AND the
        cavity's simulated type-I error <= MAX_MEASURED_TYPE1. The last clause
        is new: the two bootstrap intervals share their failure mode, so their
        agreement was never evidence that either was calibrated. It is a
        GROSS-failure guard only — the estimator's own per-pilot sd is 0.02-0.055
        at these n, so it cannot resolve 0.025 from 0.05. The actual protection
        is that (a) runs on the conservative gate_lo.
    (e) J_C is the full cross-target set, or every absent arm is named and
        worst_case_over_n is carried.
    (f) WINDOW CONSISTENCY: every leg feeding the cavity shares an identical
        (n_analyzed, dt_ps, t_first_ps, t_last_ps). REVIEW FINDING 7: the window
        is a FRACTION of frames, and legs differing only within the per-leg
        guards (>=20 ns, <=20 ps) were measured to move the cavity statistic by
        36%. Unequal windows are different estimators, so this fails closed.
    FAIL-CLOSED: anything unexplained gives UNDECIDABLE, never PASS.
    """
    reasons, conds = [], {}
    missing_arms = sorted(set(full_cross_set) - set(J_C))

    lo_b, lo_p = ci.get("bca_lo"), ci.get("perc_lo")
    lo_g = ci.get("gate_lo")
    hi_b = ci.get("gate_hi", ci.get("bca_hi"))
    theta = ci.get("theta_hat")
    t1 = (type1 or {}).get("type1")

    conds["a_ci_lower_gt_0"] = bool(lo_g is not None and lo_g > 0)
    conds["b_sufficiency"] = bool(n_def >= MIN_N_DEF)
    conds["c_binding"] = bool(n_bound >= BINDING_MIN_SNAPSHOTS)
    conds["d_stability"] = bool(
        lo_b is not None and lo_p is not None
        and abs(lo_b - lo_p) <= MAX_LB_DISAGREEMENT and not ci.get("degenerate")
        and (t1 is None or t1 <= MAX_MEASURED_TYPE1))
    conds["e_completeness"] = not missing_arms
    conds["f_window_consistent"] = bool(window_consistent)

    if not conds["b_sufficiency"]:
        reasons.append(f"n_def={n_def} < {MIN_N_DEF}: BCa jackknife not usable")
    if not conds["c_binding"]:
        reasons.append(f"n_bound={n_bound} < {BINDING_MIN_SNAPSHOTS}: cavity mostly "
                       f"fails to bind; cannot pass on the minority that worked")
    if not conds["d_stability"]:
        if ci.get("degenerate"):
            reasons.append("degenerate interval (zero width / undefined z0): "
                           "artefact of constants, not precision")
        elif lo_b is None or lo_p is None:
            reasons.append("interval not computable")
        elif abs(lo_b - lo_p) > MAX_LB_DISAGREEMENT:
            reasons.append(f"BCa and percentile lower bounds disagree by "
                           f"{abs(lo_b-lo_p):.3f} > {MAX_LB_DISAGREEMENT}: "
                           f"interval untrustworthy at n={n_def}, need more snapshots")
        if t1 is not None and t1 > MAX_MEASURED_TYPE1:
            reasons.append(f"simulated type-I error {t1:.3f} > {MAX_MEASURED_TYPE1} "
                           f"at n={n_def} (nominal {(type1 or {}).get('nominal')}): "
                           f"the interval is optimistic at this sample size; the "
                           f"answer is more snapshots, not a relabelled verdict")
    if not conds["f_window_consistent"]:
        reasons.append(f"UNEQUAL ANALYSIS WINDOWS across the legs of this cavity: "
                       f"{window_detail}. The window is a FRACTION of frames, not a "
                       f"physical time; unequal n_analyzed means the legs are "
                       f"different estimators and f_own vs f_cross is not a like-for-"
                       f"like comparison. Fail closed.")
    # EDGE CASE E7: J_C is a singleton. Compute normally, but carry
    # worst_case_over_n and print it in the verdict line, and NEVER describe the
    # result as "worst-case across off-targets" — the min over a singleton is
    # just that one contrast, and calling it worst-case is a claim the data does
    # not support. This is CD63's present state (CD81 arm absent in all 20 legs),
    # and it is the same cavity that carries the published headline.
    if len(J_C) < 2:
        reasons.append(f"worst_case_over_n={len(J_C)}: the min is over a SINGLE "
                       f"cross target {J_C}; this is NOT 'worst-case across "
                       f"off-targets'")
    if missing_arms:
        reasons.append(f"cross-target arm(s) ABSENT: {', '.join(missing_arms)} — "
                       f"worst_case_over_n={len(J_C)}; a cavity whose off-target "
                       f"arm never ran is not a selectivity result")
    if notes:
        reasons.extend(notes)

    hard_ok = (conds["b_sufficiency"] and conds["c_binding"]
               and conds["d_stability"] and conds["f_window_consistent"])

    if hard_ok and conds["a_ci_lower_gt_0"] and conds["e_completeness"]:
        verdict = "PASS"
    elif hard_ok and hi_b is not None and hi_b < 0:
        # "We showed it is NOT selective" — positively cross-reactive. This is a
        # different budget decision from "we could not tell" and the distinction
        # must survive into the report.
        verdict = "FAIL"
    else:
        verdict = "UNDECIDABLE"

    if verdict == "PASS":
        label = ("STRONG" if theta > STRONG_THETA else
                 "MODERATE" if theta > MODERATE_THETA else "WEAK")
    elif verdict == "FAIL":
        label = "CROSS-REACTIVE"
    else:
        label = "UNDECIDABLE"

    return {
        "verdict": verdict, "label": label, "conditions": conds,
        "reasons": reasons, "missing_arms": missing_arms,
        "worst_case_over_n": len(J_C), "J_C": list(J_C),
        "n_def": n_def, "n_bound": n_bound, "n_nobind": n_nobind,
        "n_imputed": n_imputed, "n_missing": n_missing,
        "gate_lo": lo_g, "gate_lo_method": ci.get("gate_lo_method"),
        "type1_error": type1,
        "window_consistent": bool(window_consistent),
        "gate_bound_note": "condition (a) uses gate_lo = min(BCa, percentile, "
                           "Student-t) lower bounds; it is NOT a certified 97.5% "
                           "bound — see the measured type1_error for this n",
    }


# ================================================ cavity-level assembly ========

def aggregate_cavity(cavity, mode, snapshots, full_cross_set,
                     n_boot=BOOTSTRAP_N_RESAMPLES, seed=0, level=BOOTSTRAP_CI,
                     frame_idx=None):
    """Assemble one cavity (= cavity x mode) from its per-leg (k, n, status) data.

    `snapshots` : {snap_index: {"own": leg, "cross": {target: leg}}} where a leg
                  is {"k": int|None, "n": int|None, "status": str}.
                  status is DISK EVIDENCE: "ok" | "size_excluded" | "md_failed" |
                  "absent". A size-excluded leg legitimately has k=None.
    `frame_idx` : Phase 4 frame index per snapshot, for the autocorrelation
                  diagnostic (snapshots are ordered by it).

    Returns the full cavity record: PCSI* point estimate, BCa + percentile CIs,
    gate verdict, per-cross-target breakdown with its own CI, legacy PCSI in both
    count and fraction form, variance, autocorrelation, and the power/budget
    requirement — each with and without the imputed size-excluded legs.
    """
    snap_ids = sorted(snapshots)
    n_snapshots = len(snap_ids)

    def _f(leg):
        if leg is None:
            return None
        k, n = leg.get("k"), leg.get("n")
        if k is None or not n:
            return None
        return k / n

    # ---- J_C: choose the cross-target set ONCE per cavity ------------------
    # PRIMARY uses only cross targets MEASURED in EVERY snapshot that has an own
    # leg, so S[s] is the SAME functional in every snapshot. min over 1 cross
    # target is stochastically >= min over 2, so mixing snapshot minima taken
    # over different J_C sizes gives a biased, non-comparable mean.
    #
    # REVIEW FINDING 3 — this rule was enforced for md_failed arms and BYPASSED
    # for size_excluded ones, which silently reintroduced exactly the bias it
    # exists to prevent, on CD9-main, the only cavity that could reach PASS.
    # An imputed arm contributes D=+1, the maximum of the range, and S is a MIN,
    # so the imputed term is never the argmin: those snapshots were effectively
    # min-over-ONE while the rest were min-over-two, yet the report printed
    # worst_case_over_n=2. Measured gap on a binomial DGP: the 6 imputed CD9
    # snapshots sit +0.121 above the 4 measured ones in expectation, theta_hat
    # estimates a 0.4/0.6 mixture (+0.2339) rather than E[min over both arms]
    # (+0.1611), and BCa coverage of the estimand a reader assumes falls to 0.796.
    #
    # Size exclusion is REAL EVIDENCE and is not discarded — but it changes the
    # FUNCTIONAL, not merely the value, so it is reported separately
    # (`size_exclusion_evidence`) and in a stratified secondary estimator, never
    # folded into the primary min.
    own_ok = [s for s in snap_ids if _f(snapshots[s].get("own")) is not None]

    def _measured(s, j):
        leg = snapshots[s].get("cross", {}).get(j)
        return bool(leg and leg.get("status") == "ok" and _f(leg) is not None)

    def _usable(s, j):                      # measured OR size-excluded (imputable)
        leg = snapshots[s].get("cross", {}).get(j)
        if leg is None:
            return False
        st = leg.get("status")
        return st == "size_excluded" or (st == "ok" and _f(leg) is not None)

    J_primary = [j for j in full_cross_set
                 if own_ok and all(_measured(s, j) for s in own_ok)]
    J_imputable = [j for j in full_cross_set
                   if own_ok and all(_usable(s, j) for s in own_ok)]
    J_all = [j for j in full_cross_set if any(_usable(s, j) for s in snap_ids)]

    def _build(J_C):
        recs, notes = {}, []
        for s in snap_ids:
            own = snapshots[s].get("own")
            f_own = _f(own)
            if f_own is None:
                st = (own or {}).get("status", "absent")
                recs[s] = {"S": None, "status": f"own_leg_{st}", "f_own": None,
                           "cross": {}, "n_cross_used": 0, "n_imputed": 0,
                           "n_missing": 0}
                notes.append(f"snap{s}: own leg {st} -> snapshot has no value")
                continue
            cross = {}
            for j in J_C:
                leg = snapshots[s].get("cross", {}).get(j) or {"status": "absent"}
                cross[j] = {"f": _f(leg), "status": leg.get("status", "absent"),
                            "k": leg.get("k"), "n": leg.get("n")}
            r = snapshot_value(f_own, cross, J_C)
            r["k_own"] = own.get("k")
            r["n_own"] = own.get("n")
            recs[s] = r
            for j, c in r["cross"].items():
                if c["source"] not in ("md", "size_excluded"):
                    notes.append(f"snap{s}: cross {j} unusable ({c['source']})"
                                 f" -> dropped from min(), NOT imputed")
            if r["status"] == "no_binding":
                notes.append(f"snap{s}: NO BINDING (f_own=0 and all f_cross=0)"
                             f" -> excluded from theta_hat and bootstrap pool")
        return recs, notes

    def _score(J_C, drop_imputed=False):
        recs, notes = _build(J_C)
        if drop_imputed:
            # Re-take the min over only the non-imputed cross terms so the
            # constant +1 legs cannot carry the signal or deflate the variance.
            for s, r in recs.items():
                keep = [c["D"] for c in r["cross"].values()
                        if c["D"] is not None and not c.get("imputed")]
                r = dict(r)
                r["S"] = min(keep) if keep else None
                r["n_cross_used"] = len(keep)
                if not keep and r["status"] == "defined":
                    r["status"] = "all_cross_imputed"
                recs[s] = r
        vals = [recs[s]["S"] for s in snap_ids if recs[s]["S"] is not None]
        ids = [s for s in snap_ids if recs[s]["S"] is not None]
        return recs, notes, vals, ids

    recs, notes, vals, ids = _score(J_primary)

    # ---- snapshot-level partition; must be exhaustive ----------------------
    n_def = len(vals)
    n_nobind = sum(1 for s in snap_ids if recs[s]["status"] == "no_binding")
    n_own_absent = sum(1 for s in snap_ids
                       if str(recs[s]["status"]).startswith("own_leg_"))
    n_no_cross = sum(1 for s in snap_ids if recs[s]["status"] == "no_cross_data")
    # Every reported quantity states its n and its inclusion rule; the snapshot
    # categories must be exhaustive and disjoint or we have reintroduced the
    # two-statistics-two-populations defect.
    assert n_def + n_nobind + n_own_absent + n_no_cross == n_snapshots, (
        f"snapshot partition does not sum: {n_def}+{n_nobind}+{n_own_absent}"
        f"+{n_no_cross} != {n_snapshots}")

    n_imputed_legs = sum(recs[s]["n_imputed"] for s in snap_ids)
    n_missing_legs = sum(recs[s]["n_missing"] for s in snap_ids)
    # (c) BINDING: a snapshot counts as bound if its own leg formed any
    # persistent contact. +1 saturates on CONTRAST, not affinity — f_own=1/101
    # vs 0 scores identically to 40/101 vs 0 — so this gate is separate and
    # non-negotiable, and raw k_L is reported alongside every value.
    n_bound = sum(1 for s in snap_ids
                  if (recs[s].get("k_own") or 0) > 0)

    # ---- (f) WINDOW CONSISTENCY across the legs of this cavity -------------
    # REVIEW FINDING 7. The analysis window is `start = int(n_frames*(1-last_frac))`
    # — a FRACTION of frames, not a physical time. The only guards are per-leg
    # (window_ns >= 20, dt <= 20 ps); neither compares legs. Legs at 2501 and
    # 2006 analysed frames both pass every guard, and on the real CD9 leg that
    # difference moves k from 10 to 8 and the cavity statistic from -0.2924 to
    # -0.1874, a 36% shift. f_L is normalised by n_analyzed so the MEAN
    # frequency is window-invariant, but persistence is a STEP function of
    # freq_r whose sampling noise scales as 1/sqrt(n_analyzed) — unequal windows
    # are different estimators. Latent on today's corpus (all 88 legs identical)
    # and precisely the thing a future re-run can silently change, which is what
    # this whole recompute exists to size.
    win_groups = {}
    for s in snap_ids:
        legs_here = [("own", snapshots[s].get("own"))]
        legs_here += [(j, snapshots[s].get("cross", {}).get(j)) for j in J_primary]
        for role, leg in legs_here:
            if not leg or leg.get("status") != "ok":
                continue
            w = leg.get("window")
            if not w:
                continue
            kw = (w.get("n_analyzed"), w.get("dt_ps"),
                  w.get("t_first_ps"), w.get("t_last_ps"))
            win_groups.setdefault(kw, []).append(f"snap{s}:{role}")
    window_consistent = len(win_groups) <= 1
    window_check = {
        "consistent": window_consistent,
        "n_distinct": len(win_groups),
        "rule": "all legs of a cavity must share identical (n_analyzed, dt_ps, "
                "t_first_ps, t_last_ps); if a future re-run violates this, "
                "subsample every leg onto the coarsest common time grid BEFORE "
                "computing freq_r — do not compare fractions across windows",
        "groups": [{"n_analyzed": k[0], "dt_ps": k[1], "t_first_ps": k[2],
                    "t_last_ps": k[3], "n_legs": len(v), "legs": sorted(v)[:8]}
                   for k, v in win_groups.items()],
    }
    if not window_consistent:
        notes.append(f"WINDOW MISMATCH: {len(win_groups)} distinct analysis "
                     f"windows among the legs of this cavity -> gate (f) fails")
    window_detail = "; ".join(
        f"n_analyzed={g['n_analyzed']} dt={g['dt_ps']} "
        f"[{g['t_first_ps']},{g['t_last_ps']}] ps x{g['n_legs']} legs"
        for g in window_check["groups"]) or "no window metadata recorded"

    ci = bootstrap_ci(vals, n_boot=n_boot, level=level, seed=seed)
    # REVIEW FINDING 1: the gate's own false-positive rate at this cavity's
    # realised n and shape, simulated under a recentred null. Printed next to
    # the verdict and consumed by gate condition (d).
    t1 = (gate_type1_error(vals, seed=seed + 7919) if n_def >= 2
          else {"type1": None, "error": f"n_def={n_def} < 2"})
    gate = evaluate_gate(ci, n_def, n_bound, J_primary, full_cross_set,
                         n_nobind=n_nobind, n_imputed=n_imputed_legs,
                         n_missing=n_missing_legs, type1=t1,
                         window_consistent=window_consistent,
                         window_detail=window_detail)

    # ---- mean-of-min vs min-of-mean ----------------------------------------
    # E[min_j D_j] <= min_j E[D_j] (Jensen): mean-of-min is CONSERVATIVE. Gate on
    # mean-of-min; report both so nobody later "discovers" the discrepancy and
    # switches to min-of-mean, quietly loosening the gate.
    per_cross = {}
    for j in J_primary:
        dv = [recs[s]["cross"][j]["D"] for s in snap_ids
              if j in recs[s]["cross"] and recs[s]["cross"][j]["D"] is not None]
        per_cross[j] = {
            "n": len(dv), "values": dv,
            "mean": float(np.mean(dv)) if dv else None,
            "ci": bootstrap_ci(dv, n_boot=n_boot, level=level, seed=seed) if dv else None,
            "n_imputed": sum(1 for s in snap_ids
                             if j in recs[s]["cross"]
                             and recs[s]["cross"][j].get("imputed")),
        }
    min_of_mean = min((v["mean"] for v in per_cross.values()
                       if v["mean"] is not None), default=None)

    # ---- imputation diagnostic --------------------------------------------
    # REVIEW FINDING 2. The mandated with/without-imputed split was
    # MATHEMATICALLY INCAPABLE OF FIRING. Imputation sets D=+1, the maximum of
    # the range, and the snapshot aggregation is a min, so an imputed term is
    # never the argmin unless every other term is +1 too: sup|min(1,x) - x| = 0
    # over x in [-1,1]. A randomised search over 4000 partially-imputed cavities
    # found 1194 identical, 2805 differing only by dropped all-imputed
    # snapshots, and ZERO with a changed value. The report line "with/without
    # verdicts DIFFER" could essentially never trigger, so its silence was
    # false reassurance rather than evidence.
    #
    # Two things replace it:
    #   (1) `legacy_null_diagnostic` — the old comparison, still computed, with
    #       the measured maximum effect, so the null result is visible AS null.
    #   (2) `imputation_diagnostic` — the comparison that can actually vary:
    #       theta over snapshots with NO imputed arm vs theta over snapshots
    #       WITH one, both taken over the SAME functional J_imputable. For
    #       CD9-main that is snapshots 0-3 against 4-9.
    recs_ni, _, vals_ni, ids_ni = _score(J_primary, drop_imputed=True)
    ci_ni = bootstrap_ci(vals_ni, n_boot=n_boot, level=level, seed=seed)
    gate_ni = evaluate_gate(ci_ni, len(vals_ni),
                            sum(1 for s in snap_ids if (recs_ni[s].get("k_own") or 0) > 0),
                            J_primary, full_cross_set,
                            window_consistent=window_consistent,
                            window_detail=window_detail)
    legacy_null_diag = {
        "max_abs_change_in_S": max(
            (abs(recs_ni[s]["S"] - recs[s]["S"]) for s in snap_ids
             if recs_ni[s]["S"] is not None and recs[s]["S"] is not None),
            default=0.0),
        "n_snapshots_dropped": sum(
            1 for s in snap_ids if recs[s]["S"] is not None and recs_ni[s]["S"] is None),
        "proof": "sup|min(1,x) - x| = 0 for x in [-1,1]; an imputed +1 can never "
                 "be the argmin of a min, so this diagnostic can only ever drop "
                 "whole snapshots, never change a value. Its agreement is NOT "
                 "evidence that imputation is carrying the signal.",
    }

    # The diagnostic that CAN vary: split snapshots by whether any arm of the
    # imputable functional was imputed, and compare theta across the split.
    imp_diag = {"J_functional": J_imputable}
    if J_imputable:
        recs_i, _, _, _ = _score(J_imputable)
        clean_ids = [s for s in snap_ids
                     if recs_i[s].get("S") is not None
                     and all(not c.get("imputed") for c in recs_i[s]["cross"].values())]
        dirty_ids = [s for s in snap_ids
                     if recs_i[s].get("S") is not None
                     and any(c.get("imputed") for c in recs_i[s]["cross"].values())]
        cv = [recs_i[s]["S"] for s in clean_ids]
        dv = [recs_i[s]["S"] for s in dirty_ids]
        imp_diag.update({
            "n_clean": len(cv), "n_with_imputed_arm": len(dv),
            "clean_snapshot_ids": clean_ids, "imputed_snapshot_ids": dirty_ids,
            "theta_clean": float(np.mean(cv)) if cv else None,
            "theta_with_imputed": float(np.mean(dv)) if dv else None,
            "difference": (float(np.mean(dv) - np.mean(cv)) if cv and dv else None),
            "ci_clean": (bootstrap_ci(cv, n_boot=n_boot, level=level, seed=seed)
                         if len(cv) >= 2 else None),
            "interpretation": None,
        })
        if cv and dv:
            imp_diag["interpretation"] = (
                f"snapshots carrying an imputed arm score "
                f"{imp_diag['difference']:+.4f} vs the measured-only snapshots; "
                f"a large positive gap means the apparent selectivity is coming "
                f"from size-exclusion constants, not from measured contrast")
        elif not dv:
            imp_diag["interpretation"] = "no imputed arms in this cavity"
        else:
            imp_diag["interpretation"] = (
                "EVERY snapshot carries an imputed arm: there is no measured-only "
                "subset to compare against, so imputation cannot be ruled out as "
                "the source of the signal")

    # ---- stratified estimator: the size-exclusion evidence, kept separate ---
    # REVIEW FINDING 3, second half of the fix. Group snapshots by WHICH arms
    # were actually measured; each stratum is a different functional and gets
    # its own theta. Nothing is averaged across strata.
    strata = {}
    for s in snap_ids:
        if _f(snapshots[s].get("own")) is None:
            continue
        meas = tuple(sorted(j for j in full_cross_set if _measured(s, j)))
        exc = tuple(sorted(j for j in full_cross_set
                           if (snapshots[s].get("cross", {}).get(j) or {}
                               ).get("status") == "size_excluded"))
        strata.setdefault((meas, exc), []).append(s)
    stratified = []
    for (meas, exc), ss in sorted(strata.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        if not meas:
            stratified.append({"measured_arms": list(meas), "size_excluded_arms": list(exc),
                               "snapshots": ss, "n": len(ss), "theta": None,
                               "note": "no measured cross arm; min-over-zero is undefined"})
            continue
        r_st, _, v_st, _ = _score(list(meas))
        vv = [r_st[s]["S"] for s in ss if r_st[s]["S"] is not None]
        stratified.append({
            "measured_arms": list(meas), "size_excluded_arms": list(exc),
            "snapshots": ss, "n": len(vv),
            "worst_case_over_n": len(meas),
            "theta": float(np.mean(vv)) if vv else None,
            "sd": float(np.std(vv, ddof=1)) if len(vv) > 1 else None,
        })
    size_exclusion_evidence = {
        j: {"n_snapshots_size_excluded":
                sum(1 for s in snap_ids
                    if (snapshots[s].get("cross", {}).get(j) or {}).get("status")
                    == "size_excluded"),
            "in_primary_J_C": j in J_primary,
            "imputed_D_if_used": 1.0}
        for j in full_cross_set}

    # ---- secondary estimator over ALL available arms ------------------------
    sec = None
    if set(J_all) != set(J_primary):
        r2, n2, v2, _ = _score(J_all)
        sec = {"J_C": J_all, "n_def": len(v2), "values": v2,
               "ci": bootstrap_ci(v2, n_boot=n_boot, level=level, seed=seed),
               "note": "all-available-legs estimator; J_C differs per snapshot, "
                       "NOT comparable with the primary"}

    # ---- legacy comparators -------------------------------------------------
    legacy_counts, legacy_fracs = [], []
    for s in snap_ids:
        own = snapshots[s].get("own")
        if not own or own.get("k") is None:
            continue
        kc, fc = [], []
        # J_imputable, not J_primary: the legacy metric DID fold size-excluded
        # arms in as zero counts, and this comparator has to stay faithful to
        # the thing it is comparing against.
        for j in J_imputable:
            leg = snapshots[s].get("cross", {}).get(j) or {}
            if leg.get("status") == "ok" and leg.get("k") is not None:
                kc.append(leg["k"])
                fc.append(leg["k"] / leg["n"])
            elif leg.get("status") == "size_excluded":
                kc.append(0)
                fc.append(0.0)
        p = legacy_pcsi(own["k"], kc)
        if p is not None:
            legacy_counts.append(p)
        if fc:
            mx = max(fc)
            legacy_fracs.append((own["k"] / own["n"]) / mx if mx > 0 else math.inf)

    # ---- variance / autocorrelation diagnostics -----------------------------
    order = ids
    if frame_idx:
        order = sorted(ids, key=lambda s: frame_idx.get(s, s))
    acf_S = lag1_autocorrelation([recs[s]["S"] for s in order])
    acf_f = lag1_autocorrelation([recs[s]["f_own"] for s in order])

    theta = ci.get("theta_hat")
    sd = ci.get("sd")
    power = power_requirement(theta, sd, n_def) if n_def >= 2 else \
        {"error": f"n_def={n_def} < 2"}
    power_ni = power_requirement(ci_ni.get("theta_hat"), ci_ni.get("sd"),
                                 len(vals_ni)) if len(vals_ni) >= 2 else None
    # REVIEW FINDING 4: the budget as an interval, plus theta's own CI right
    # beside it — a theta CI straddling 0 makes the budget number undefined,
    # and that must be visible at the point of reading the budget.
    budget = budget_interval(vals, seed=seed) if n_def >= 2 else \
        {"error": f"n_def={n_def} < 2"}
    # theta's own CI rides on the budget record ALWAYS, including when the
    # budget itself is undefined — a CI straddling 0 is exactly the case where
    # a reader must not take a snapshot count away from this block.
    budget["theta_hat"] = theta
    budget["theta_ci_gate_lo"] = ci.get("gate_lo")
    budget["theta_ci_gate_hi"] = ci.get("gate_hi")
    budget["theta_ci_straddles_zero"] = bool(
        ci.get("gate_lo") is not None and ci.get("gate_hi") is not None
        and ci["gate_lo"] <= 0 <= ci["gate_hi"])
    if budget["theta_ci_straddles_zero"]:
        budget["warning"] = (
            "theta's own CI straddles 0: the sign of the effect is not "
            "established, so ANY snapshot count derived from it is undefined. "
            "Budget to establish the sign, not to reach a target power.")

    return {
        "cavity": cavity, "mode": mode, "key": f"{cavity}_{mode}",
        "bookkeeping": {
            "n_snapshots": n_snapshots, "n_def": n_def, "n_nobind": n_nobind,
            "n_own_absent": n_own_absent, "n_no_cross_data": n_no_cross,
            "n_bound": n_bound, "n_imputed_legs": n_imputed_legs,
            "n_missing_legs": n_missing_legs,
            "J_C": J_primary, "J_C_all_available": J_all,
            "J_C_imputable": J_imputable,
            "worst_case_over_n": len(J_primary),
            "worst_case_note":
                "worst_case_over_n counts only MEASURED arms. Arms usable solely "
                "via size-exclusion imputation are excluded from the primary "
                "functional (REVIEW FINDING 3) and appear in "
                "size_exclusion_evidence / stratified instead.",
        },
        "snapshots": {str(s): recs[s] for s in snap_ids},
        "S_values": vals, "S_snapshot_ids": ids,
        "theta_hat": theta, "median": ci.get("median"),
        "sd": sd, "cv": ci.get("cv"), "var": (sd ** 2 if sd is not None else None),
        "ci": ci, "gate": gate,
        "mean_of_min": theta, "min_of_mean": min_of_mean,
        "aggregation_bias_note": "mean-of-min <= min-of-mean (Jensen); the gate "
                                 "runs on mean-of-min, the conservative one",
        "per_cross_target": per_cross,
        "without_imputed": {"n_def": len(vals_ni), "values": vals_ni,
                            "ci": ci_ni, "gate": gate_ni, "power": power_ni,
                            "legacy_null_diagnostic": legacy_null_diag},
        "imputation_diagnostic": imp_diag,
        "stratified": stratified,
        "size_exclusion_evidence": size_exclusion_evidence,
        "window_check": window_check,
        "type1_error": t1,
        "budget": budget,
        "secondary_all_arms": sec,
        "legacy_pcsi_counts": legacy_summary(legacy_counts),
        "legacy_pcsi_fractions": legacy_summary(legacy_fracs),
        "autocorrelation": {"S_lag1": acf_S, "f_own_lag1": acf_f},
        "power": power,
        "notes": sorted(set(notes)),
    }
