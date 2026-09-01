"""
Phase 4: Pre-polymerization MD + Optimal Monomer Ratio Determination
=====================================================================
Run pre-polymerization MD with experimental monomer ratios, analyze contact
frequency to determine optimal synthesis ratios, and validate binding stability.

Workflow:
  Step A: Build system (sol-gel box built to a target Q mole fraction) → MD
  Step B: Analyze per-monomer-type contact frequency with epitope
  Step C: Derive optimal synthesis ratio from occupancy
  Step D: Pool INDEPENDENT replicas; reject legs that never equilibrated

Reference:
  Sullivan et al., J. Phys. Chem. B 2019 — MM-GBSA for MIP
  Sehit/Altintas, ACS Sensors 2024 — epitope:monomer = 1:20
  Yuan et al. 2024 — EBN / HBNMax
  Polania & Jimenez 2024 — block-averaged MD convergence

════════════════════════════════════════════════════════════════════
2026-08 AUDIT FIXES — behaviour changes are flagged  # BEHAVIOUR CHANGE
════════════════════════════════════════════════════════════════════
 1. MM-GBSA frame window is now derived from the ACTUAL trajectory dt and
    asserted to lie inside the trajectory (was: startframe=1, endframe=100,
    i.e. the FIRST 1 ns of a 350 ns run, silently ignoring the config keys).
 2. The `md.gro exists → skip parameterisation` guard is now conditional on an
    input fingerprint (SMILES, copy numbers, template, seed, ...) still
    matching. A stale directory FAILS LOUDLY instead of silently reusing a
    trajectory made with retired force-field parameters.
 3. HBNMax donor/acceptor selections are now DISJOINT (protein→monomer and
    monomer→protein), so a molecule with no H-bond donors can no longer score
    168-195 off the protein's own internal H-bonds.
 4. Residence time is reported in ps/ns, not in stride units. Runs that are
    still in contact at the end of the window are now flushed.
 5. Block-averaged convergence is now a RECORDED, PROPAGATED FAILURE: a
    non-converged replica is not accepted and is not pooled.
 6. Sol-gel boxes are built to SOLGEL_Q_MOLE_FRACTION (Q = tetraalkoxysilane
    crosslinker) instead of `total // (n_functional + 1)`, which gave 72 mol%
    T units. total_monomers is UNCHANGED, so GPU cost is unchanged.
 7. optimal_ratio[crosslinker] is derived from the target crosslinker mole
    fraction instead of being hardcoded to sum(functional) (= exactly 50 mol%
    for every recipe, regardless of the EBN data it claimed to derive from).
 8. PHASE4_N_REPLICAS INDEPENDENT trajectories per (target, pc_id), each in its
    own rep_<i>/ directory with a distinct, reproducible placement seed.

════════════════════════════════════════════════════════════════════
2026-08 SAMPLING AUDIT — what the numbers can and cannot support
════════════════════════════════════════════════════════════════════
Measured on the archived 350 ns CD63 / CD9 legs and on this machine's GROMACS:
 A. The per-copy contact observable has an integrated autocorrelation time of
    1.5-33 ns.  The old analysis window (last 25% of a 30 ns leg = 7.5 ns) held
    under ONE correlation time, and window-to-window noise falls as T^-0.31,
    not T^-0.50.  -> PHASE4_ANALYSIS_WINDOW_FRACTION, default 0.50, and every
    leg now reports its own tau_int / n_independent_samples / sub-window
    ranking stability in occupancy_analysis.sampling.
 B. Splitting ONE 350 ns trajectory into 50 non-overlapping 7 ns windows gives
    19 DISTINCT rankings of its 4 monomers.  -> the grid ranks with a
    two-sample Welch test plus a pre-registered MDD, reports a `decision` of
    optimum / trend_only / flat, and refuses to rank at all at n_replicas=1.
 C. The 4 within-window block means understate the true window-to-window SD by
    a median 2.8x (up to 7.5x).  -> they stay a DRIFT diagnostic and are
    flagged `block_means_are_not_an_error_bar`; only independent replicas count.
 D. A present species with zero contacts is a MEASUREMENT, not a failed
    analysis.  Rejecting on it biased rejection toward the scarce-species end
    of the grid.  -> PHASE4_ZERO_CONTACT_IS_A_MEASUREMENT; an all-zero box is
    still rejected.
 E. `quick=(time_ns <= 20)` silently replaced any leg at or below 20 ns with
    MD_QUICK_NS.  -> quick is never inferred; the effective length is recorded
    and asserted, and the MM-GBSA window is pre-flighted against the leg length.
 F. MD_SOLVENT_PH had no physical effect.  -> utils_structure.titration_model +
    utils_gromacs.setup_protein_topology drive pdb2gmx, and `_ph_preflight`
    refuses to start while the fixed-charge residual is unacknowledged.

CONFIG KEYS THIS MODULE NEEDS (add to config.py's baseline body — config_CD.py
is a pure-documentation delta, and a key added only there would be invisible to
every other experiment).  Until they exist, `_cfg()` logs an ERROR per key and
falls back to the documented default listed here:
    SOLGEL_Q_MOLE_FRACTION            = 0.60
    RADICAL_CROSSLINKER_MOLE_FRACTION = 0.50   # documents legacy behaviour
    PHASE4_N_REPLICAS                 = 3
    PHASE4_CONVERGENCE_TOL_PCT        = 10.0
    PHASE4_RMSD_DRIFT_TOL_NM          = 0.05
    PHASE4_ALLOW_STALE_RESUME         = False

POST-BASELINE KEYS (`_CFG_POST_BASELINE`) are deliberately NOT in config.py:
config_baseline_CD.json is a frozen pre-dispatch contract and
test_config_regression T1 asserts CD's namespace gains nothing.  config_BSA.py
declares them; under CD they resolve silently to the defaults above and their
effective values are stamped into ratio_grid_summary.json.
"""

import hashlib
import inspect
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Config access ─────────────────────────────────────────────
# Keys introduced by the 2026-08 audit may not have landed in config.py yet.
# Missing key → ERROR-level log + documented default (never a silent guess).

_CFG_FALLBACK_WARNED = set()

_CFG_DEFAULTS = {
    # Mole fraction of Q units (tetraalkoxysilane crosslinker, e.g. TEOS/TMOS)
    # in a sol-gel pre-polymerisation box. Closest published analogue for an
    # APTES/TEOS protein-imprinted sol-gel is ~60 mol% Q.
    "SOLGEL_Q_MOLE_FRACTION": 0.60,
    # Crosslinker mole fraction for free-radical (vinyl/catechol) systems.
    # 0.50 reproduces the pre-audit hardcoded behaviour exactly.
    "RADICAL_CROSSLINKER_MOLE_FRACTION": 0.50,
    # Independent Phase 4 trajectories per (target, pc_id).
    "PHASE4_N_REPLICAS": 3,
    # Block-averaged contact-frequency convergence tolerance (percent).
    "PHASE4_CONVERGENCE_TOL_PCT": 10.0,
    # Max allowed backbone-RMSD drift between the 3rd and 4th quarter (nm).
    "PHASE4_RMSD_DRIFT_TOL_NM": 0.05,
    # Escape hatch for resuming a directory whose inputs no longer match.
    "PHASE4_ALLOW_STALE_RESUME": False,
    # BLOCKER 11. md_centered.xtc is a full-size duplicate of md.xtc (measured
    # 11.19 GB vs 11.33 GB on one 350 ns leg) consumed by a single analysis
    # pass. False = delete it once the analysis has read it. Set True only to
    # keep it for manual inspection, and expect ~1 GB per leg per replica.
    "PHASE4_KEEP_CENTERED_TRAJECTORY": False,

    # ── 2026-08 SAMPLING AUDIT (findings 3, 7-12) ─────────────────────
    # Fraction of the trajectory, measured from the END, that the occupancy
    # analysis reads.  WAS effectively 0.25 (`use_q4=True`, the last quarter).
    # MEASURED: the integrated autocorrelation time of the per-copy contact
    # signal is 8-33 ns on the archived legs, so the Q4 window of a 30 ns leg
    # (7.5 ns) is UNDER ONE correlation time and window-to-window noise falls
    # only as T^-0.31 rather than T^-0.50.  Reading the last HALF doubles the
    # sample at zero GPU cost.  Anything earlier is pre-equilibration.
    "PHASE4_ANALYSIS_WINDOW_FRACTION": 0.50,
    # A species that is PRESENT in the box but records zero contacts in both
    # comparison windows has produced a MEASUREMENT (contact frequency 0 at
    # this composition), not a failed analysis.  True = record it, rank the
    # point at zero for that species, and do not reject the replica for it.
    # False = restore the pre-fix behaviour (reject).  Zero contacts for EVERY
    # species is still a rejection either way — that is a dead box.
    "PHASE4_ZERO_CONTACT_IS_A_MEASUREMENT": True,
    # How the grid decides that two points are indistinguishable.
    #   "ci"    — declare a tie whenever the runner-up's mean lies inside the
    #             top point's replicate confidence interval (default).
    #   "mdd"   — declare a tie whenever the relative margin is below
    #             PHASE4_RANK_MDD_PCT.
    #   "exact" — the retired behaviour: float equality (fires ~0.1% of the
    #             time under a completely flat truth).
    "PHASE4_RANK_TIE_RULE": "ci",
    # Pre-registered minimum detectable difference, percent of the top score.
    # A margin below this is a tie whatever the CI says.  MEASURED basis: the
    # single-molecule window CV is ~1.8, so a point with n functional copies
    # and R replicas has CV ~ 1.8/sqrt(n*R); at n=12, R=3 that is 30%.
    "PHASE4_RANK_MDD_PCT": 30.0,
    # Two-sided confidence level for the replicate CI used by the tie rule.
    "PHASE4_RANK_CI_LEVEL": 0.95,
    # The loading (n_total) axis may only be run on a composition that is
    # either explicitly pinned or QUALIFIED (its CI excludes the runner-up).
    # False = allow the old behaviour of conditioning it on a bare argmax.
    "PHASE4_LOADING_SWEEP_REQUIRE_QUALIFIED_WINNER": True,
    # How the within-leg block-convergence number is used.
    #   "blocking" — a leg that fails it is REJECTED and never pooled (the
    #                behaviour the 2026-08 audit introduced; kept for CD).
    #   "advisory" — the number is still computed and reported, but it does not
    #                reject the leg. Uncertainty is then carried by the
    #                replicate CI (`_mean_ci`), which is what the ranking
    #                already consumes.
    # WHY "advisory" EXISTS. The gate compares two sub-window block means and
    # demands they agree within PHASE4_CONVERGENCE_TOL_PCT. MEASURED on the BSA
    # ratio grid: tau of the contact count is 0.45-0.78 ns, so a 3.75 ns block
    # holds ~4 independent samples and the block-to-block difference expected
    # from NOISE ALONE is 17-20% — above the 10% threshold. The gate therefore
    # rejected legs at close to random, which is statistically indistinguishable
    # from the cherry-picking the LiveCoMS best-practices checklist forbids
    # (Grossfield et al., LiveCoMS 1(1):5067, 2018: "Use all of the available
    # data unless there is an objective and compelling reason not to... sampling
    # metrics should be applied uniformly to all simulations"). No published
    # standard defines a block-to-block percentage threshold at all.
    "PHASE4_CONVERGENCE_GATE": "blocking",
}


# Keys introduced by the 2026-08 SAMPLING audit.  They are deliberately NOT in
# config.py's baseline body: config_baseline_CD.json is a frozen pre-dispatch
# contract and test_config_regression T1 asserts that CD's namespace gains
# nothing but the four dispatch names.  config_BSA.py declares them, so under
# BSA they are real, live, listed config keys; under CD they resolve to the
# documented defaults above WITHOUT an ERROR, because a missing key here is the
# designed state rather than an omission.  Their effective values are stamped
# into every ratio_grid_summary.json (`effective_sampling_config`) so no run can
# depend on a default nobody can see.
_CFG_POST_BASELINE = frozenset({
    "PHASE4_ANALYSIS_WINDOW_FRACTION",
    "PHASE4_ZERO_CONTACT_IS_A_MEASUREMENT",
    "PHASE4_RANK_TIE_RULE",
    "PHASE4_RANK_MDD_PCT",
    "PHASE4_RANK_CI_LEVEL",
    "PHASE4_LOADING_SWEEP_REQUIRE_QUALIFIED_WINNER",
    "PHASE4_CONVERGENCE_GATE",
})


def _cfg(name):
    """Read a config symbol, falling back LOUDLY to a documented default."""
    from . import config as _c
    if hasattr(_c, name):
        return getattr(_c, name)
    default = _CFG_DEFAULTS[name]
    if name in _CFG_POST_BASELINE:
        return default
    if name not in _CFG_FALLBACK_WARNED:
        _CFG_FALLBACK_WARNED.add(name)
        logger.error(
            "CONFIG KEY MISSING: %s is not defined in pipeline.config. "
            "Falling back to the documented default %r. Add it to config.py's "
            "baseline body (see phase4_md_validation.py header).", name, default)
    return default


def _effective_sampling_config() -> dict:
    """The values of the post-baseline sampling keys that this run is using."""
    from . import config as _c
    return {k: {"value": _cfg(k),
                "source": ("config" if hasattr(_c, k) else
                           "phase4 documented default (not declared in config)")}
            for k in sorted(_CFG_POST_BASELINE)}


def _cfg_get(name, default=None):
    """Read an OPTIONAL config symbol that only some experiments declare.

    Distinct from `_cfg()`: `_cfg` is for keys every experiment must have (a
    miss is an ERROR + documented default).  `_cfg_get` is for experiment-local
    declarations such as BSA_RATIO_GRID, which legitimately do not exist under
    CD.  Reading them by getattr — instead of `from .config import ...` — is
    what lets the ratio-grid driver consume the BSA keys BY THEIR REAL NAMES
    without breaking the CD import contract (test_config_regression T3).
    """
    from . import config as _c
    return getattr(_c, name, default)


def _env_int(name: str):
    """A positive int from the environment, or None. Refuses garbage loudly."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        v = int(raw.strip())
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer")
    if v < 1:
        raise ValueError(f"{name}={v} must be >= 1")
    logger.info("  %s=%d taken from the environment", name, v)
    return v


def _ratio_grid_enabled() -> bool:
    """PHASE4_RATIO_GRID_ENABLED, overridable by env for a one-off screening run."""
    env = os.environ.get("PHASE4_RATIO_GRID_ENABLED")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(_cfg_get("PHASE4_RATIO_GRID_ENABLED", False))


def _loading_sweep_enabled() -> bool:
    """PHASE4_LOADING_SWEEP_ENABLED / BSA_LOADING_SWEEP_ENABLED, env-overridable."""
    env = os.environ.get("PHASE4_LOADING_SWEEP_ENABLED")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(_cfg_get("PHASE4_LOADING_SWEEP_ENABLED",
                         _cfg_get("BSA_LOADING_SWEEP_ENABLED", False)))


def _allow_stale_resume() -> bool:
    """PHASE4_ALLOW_STALE_RESUME, overridable by env for one-off recoveries."""
    env = os.environ.get("PHASE4_ALLOW_STALE_RESUME")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(_cfg("PHASE4_ALLOW_STALE_RESUME"))


def _analysis_window_fraction() -> float:
    """Fraction of the trajectory, from the END, that the analysis reads."""
    f = float(_cfg("PHASE4_ANALYSIS_WINDOW_FRACTION"))
    if not (0.0 < f <= 1.0):
        raise ValueError(
            f"PHASE4_ANALYSIS_WINDOW_FRACTION={f} is not in (0, 1]")
    return f


# ── Sampling statistics (2026-08 sampling audit) ───────────────
#
# THE OBSERVABLE IS AUTOCORRELATED AND THE ANALYSIS WINDOW IS SHORT.
# Measured on the archived 350 ns CD63 / CD9 legs: the integrated
# autocorrelation time of the per-copy contact signal is 1.5-33 ns, so a 7.5 ns
# Q4 window holds well under one independent sample per molecule.  Everything
# below exists so the pipeline REPORTS that fact per leg instead of implying a
# precision it does not have.

def _tau_int(x, max_lag: int = None) -> float:
    """Integrated autocorrelation time of a 1-D series, in FRAMES.

    Initial-positive-sequence truncation (Geyer): sum until the normalised
    autocorrelation first goes non-positive.  Returns 1.0 for a constant or
    trivially short series, which is the honest "one sample per frame" answer.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return 1.0
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return 1.0
    if max_lag is None:
        max_lag = max(2, min(n // 2, 500))
    tau = 1.0
    for k in range(1, max_lag):
        a = float((x[:n - k] * x[k:]).sum()) / denom
        if a <= 0:
            break
        tau += 2.0 * a * (1.0 - k / n)
    return float(max(1.0, tau))


def _t_crit(df: int, level: float = 0.95) -> float:
    """Two-sided Student-t critical value; falls back to a table without scipy."""
    if df < 1:
        return float("inf")
    try:
        from scipy import stats as _st
        return float(_st.t.ppf(0.5 + level / 2.0, df))
    except Exception:
        table95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                   6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                   15: 2.131, 20: 2.086, 30: 2.042}
        if abs(level - 0.95) > 1e-6:
            logger.error("scipy unavailable; t-table only covers level=0.95, "
                         "got %s — using the 95%% value", level)
        for k in sorted(table95):
            if df <= k:
                return table95[k]
        return 1.960


def _welch_p(a, b) -> float:
    """Two-sided Welch t-test p-value between two replicate samples.

    Used to decide whether the top grid point is SEPARATED from the runner-up.
    The obvious alternative — "is the runner-up's mean inside the top point's
    own CI?" — is not the right test: it demands roughly 2.8 standard errors of
    separation where a two-sample test needs about 2.0, and simulated on this
    grid it left even a 2x effect unresolved 85% of the time at R=6.  Returns
    1.0 (no evidence of a difference) whenever either sample is too small.
    """
    a = [float(v) for v in (a or []) if isinstance(v, (int, float))]
    b = [float(v) for v in (b or []) if isinstance(v, (int, float))]
    if len(a) < 2 or len(b) < 2:
        return 1.0
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return 0.0 if abs(np.mean(a) - np.mean(b)) > 0 else 1.0
    t = (np.mean(a) - np.mean(b)) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from scipy import stats as _st
        return float(2 * _st.t.sf(abs(t), df))
    except Exception:
        # No scipy: fall back to the 95% critical value, i.e. a binary answer.
        return 0.04 if abs(t) >= _t_crit(max(1, int(df)), 0.95) else 0.5


def _mean_ci(values, level: float = 0.95) -> dict:
    """Mean + Student-t CI of INDEPENDENT REPLICATE values.

    n is the number of independent trajectories, never the number of frames or
    of within-window blocks — see `_replica_acceptance` and the block-mean
    warning in `_analyze_monomer_occupancy`.
    """
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "sem": None,
                "ci_lo": None, "ci_hi": None, "ci_level": level,
                "ci_is_degenerate": True}
    mean = float(np.mean(vals))
    if n == 1:
        return {"n": 1, "mean": round(mean, 6), "sd": None, "sem": None,
                "ci_lo": None, "ci_hi": None, "ci_level": level,
                "ci_is_degenerate": True,
                "note": "one replica — no error bar exists. n=1 cannot rank."}
    sd = float(np.std(vals, ddof=1))
    sem = sd / math.sqrt(n)
    t = _t_crit(n - 1, level)
    return {"n": n, "mean": round(mean, 6), "sd": round(sd, 6),
            "sem": round(sem, 6), "ci_lo": round(mean - t * sem, 6),
            "ci_hi": round(mean + t * sem, 6), "ci_level": level,
            "ci_is_degenerate": False}


# ── MM-GBSA window pre-flight ─────────────────────────────────

def _mmpbsa_window_status(time_ns: float) -> dict:
    """Does the configured MM-GBSA window fit inside a leg of `time_ns`?

    MD_MMPBSA_START_NS / MD_MMPBSA_END_NS are re-derived by hand for
    MD_PRODUCTION_NS.  A sweep that runs at a DIFFERENT leg length therefore
    kills MM-GBSA on every point — and did so silently, after the trajectory
    already existed.  Checked BEFORE the first leg instead.
    """
    start = _cfg_get("MD_MMPBSA_START_NS")
    end = _cfg_get("MD_MMPBSA_END_NS")
    rec = {"time_ns": time_ns, "mmpbsa_start_ns": start, "mmpbsa_end_ns": end}
    if start is None or end is None:
        rec.update({"fits": None, "reason": "MD_MMPBSA_START_NS/END_NS not set"})
        return rec
    fits = (0.0 <= float(start) < float(end) <= float(time_ns) + 1e-9)
    rec["fits"] = bool(fits)
    if not fits:
        rec["reason"] = (
            f"MM-GBSA window {start}-{end} ns does not fit inside a {time_ns:g} ns "
            f"leg. Every point in this sweep will record mmpbsa.window_valid=False. "
            f"Set MD_MMPBSA_START_NS/MD_MMPBSA_END_NS for THIS leg length "
            f"(e.g. {time_ns * 2 / 3:.3g}-{time_ns:g} ns) before spending GPU time.")
        rec["suggested_start_ns"] = round(time_ns * 2 / 3, 3)
        rec["suggested_end_ns"] = round(time_ns, 3)
    return rec


# ── pH pre-flight ─────────────────────────────────────────────

def _ph_preflight(epitope_pdb) -> dict:
    """What charge state will actually be simulated, and does anyone know?

    THE DEFECT THIS CLOSES.  MD_SOLVENT_PH was consumed by Phase 1 and recorded
    as protonation.ph, and had NO physical effect: propka absent, pdb2gmx run
    with no protonation flags.  BSA was built at its pH-7 census (-16 e, 16 Na+)
    while the protocol runs at pH ~9.5, where the same census titrates to about
    -28 e.  setup_protein_topology now drives the states, but a FIXED-CHARGE
    force field still cannot reach -28: Lys is 91% protonated at pH 9.5 and
    tyrosinate has no residue type in amber99sb-ildn, so the best representable
    state is about -18 e.  That residual is a real physical shortfall in the
    quantity this experiment optimises, not an error bar on it.

    This function computes it BEFORE the grid runs and refuses to spend GPU time
    until it has been acknowledged in config (MD_PH_CHARGE_RESIDUAL_ACK), so the
    decision is made by a person rather than by a default.
    """
    ph = _cfg_get("MD_SOLVENT_PH")
    rec = {"ph": ph, "epitope_pdb": str(epitope_pdb)}
    if ph is None:
        rec["status"] = "no MD_SOLVENT_PH declared"
        return rec
    try:
        from .utils_structure import titration_model
        from .utils_gromacs import _ff_terminus_states_available
        unavailable = set()
        if not _ff_terminus_states_available():
            unavailable |= {("NTERM", "deprotonated"), ("CTERM", "protonated")}
        # Honour the configured protonation rule. Declaring PH_CHARGE_ASSIGNMENT
        # and PH_USE_PROPKA without reading them here would leave the pre-flight
        # judging a charge state the run will not actually build — the exact
        # "declared but unconsumed" failure this audit was called to remove.
        m = titration_model(Path(epitope_pdb), float(ph),
                            unavailable_states=unavailable,
                            assignment=_cfg_get("PH_CHARGE_ASSIGNMENT", "majority"),
                            use_propka=bool(_cfg_get("PH_USE_PROPKA", False)))
    except Exception as e:
        rec.update({"status": "model_failed", "error": f"{type(e).__name__}: {e}"})
        logger.error("  pH pre-flight FAILED (%s) — the charge state that will "
                     "be simulated is unknown.", e)
        return rec

    residual = float(m["charge_residual_vs_hh"])
    tol = float(_cfg_get("MD_PH_CHARGE_RESIDUAL_TOL_E", 2.0))
    ack = bool(_cfg_get("MD_PH_CHARGE_RESIDUAL_ACK", False))
    rec.update({
        "status": "ok",
        "hh_continuum_charge_e": m["hh_continuum_charge"],
        "discrete_charge_e": m["discrete_charge"],
        "will_be_simulated_charge_e": m["representable_charge"],
        "charge_residual_vs_hh_e": residual,
        "tolerance_e": tol,
        "acknowledged": ack,
        "unrepresentable_sites": m["unrepresentable_sites"],
        "n_disulfide_cysteines": m["n_disulfide_cysteines"],
    })
    logger.info(
        "  pH %.2f pre-flight: the box will carry a protein of %+d e. "
        "Henderson-Hasselbalch expects %+.2f e (residual %+.2f e).",
        float(ph), m["representable_charge"], m["hh_continuum_charge"], residual)
    # GATE 0 — is the pH itself known? This is PRIOR to the residual question.
    # The residual test asks "is the rounding we did acceptable?"; the LYN
    # net-charge assignment now answers that with +0.08 e. It says nothing about
    # whether the pH we rounded TO is the pH in the beaker, and no amount of
    # careful rounding fixes a wrong target. MD_SOLVENT_PH also drives
    # MONOMER_PROTONATION_SPLIT, so an assumed pH now sets how many APTES
    # molecules are cationic — the recognition partner count runs 11% to 94%
    # across the literature bracket for this pot (pH 7.5-10.3). Refuse until a
    # measurement is recorded.
    if not bool(_cfg_get("MD_SOLVENT_PH_MEASURED", True)):
        rec["status"] = "ph_unmeasured"
        raise RuntimeError(
            f"pH PRE-FLIGHT REFUSED THE RUN: MD_SOLVENT_PH = {ph} is ASSUMED, "
            f"not measured (MD_SOLVENT_PH_MEASURED is False).\n"
            f"This value sets TWO things, not one:\n"
            f"  * the protein's titration state — currently "
            f"{m['representable_charge']:+d} e\n"
            f"  * how many APTES are cationic, via MONOMER_PROTONATION_SPLIT — "
            f"at pKa 9.2 the protonated fraction is 0.94 at pH 8.0, 0.50 at "
            f"9.2 and 0.11 at 10.1\n"
            f"No published pH exists for any aqueous TEOS/APTES mixture, and "
            f"calculation only brackets this pot to 7.5-10.3 at t = 2 h. "
            f"Measure the working mixture at t = 0 and t = 2 h, set "
            f"MD_SOLVENT_PH from it, then set MD_SOLVENT_PH_MEASURED = True.")
    if abs(residual) > tol and not ack:
        raise RuntimeError(
            f"pH PRE-FLIGHT REFUSED THE RUN.\n"
            f"  MD_SOLVENT_PH = {ph}\n"
            f"  charge that will actually be simulated : {m['representable_charge']:+d} e\n"
            f"  Henderson-Hasselbalch expectation      : {m['hh_continuum_charge']:+.2f} e\n"
            f"  residual                               : {residual:+.2f} e "
            f"(tolerance {tol} e)\n"
            f"The gap is carried by groups titrating within ~1 pH unit of this pH "
            f"(Lys, pKa 10.5, is 91% protonated at pH 9.5 — a fixed-charge model "
            f"must make it 100%) and by tyrosinate, for which amber99sb-ildn has "
            f"no residue type. This MOVES the optimised quantity, it does not "
            f"merely widen its error bar: the whole imprinting driving force here "
            f"is APTES ammonium against BSA carboxylate.\n"
            f"DECIDE, then record the decision in config:\n"
            f"  * accept the fixed-charge approximation -> "
            f"MD_PH_CHARGE_RESIDUAL_ACK = True\n"
            f"  * simulate the pH you can build -> set MD_SOLVENT_PH to a value "
            f"whose majority states the force field can represent\n"
            f"  * do it properly -> install propka for site-specific pKa, or use "
            f"constant-pH MD, neither of which this pipeline currently has.")
    if abs(residual) > tol:
        logger.error(
            "  pH residual %+.2f e is above the %s e tolerance but "
            "MD_PH_CHARGE_RESIDUAL_ACK is set: proceeding with an "
            "ACKNOWLEDGED approximation. The simulated protein is %+d e, not "
            "the %+.2f e the titration curve gives.",
            residual, tol, m["representable_charge"], m["hh_continuum_charge"])
    return rec


# ── Ratio-grid identity ───────────────────────────────────────

def _guard_resumed_target_against_grid(target: str, target_results: dict) -> None:
    """Refuse a per-target RESUME whose recorded grid is not the declared grid.

    THE BUG THIS CLOSES.  `run_phase4` skips a target whose entry is already in
    phase4_md_results.json, and that skip runs BEFORE the ratio grid is
    consulted.  So editing BSA_RATIO_GRID and re-running into the same output
    directory returned the PREVIOUS grid, in 0.0 s, reported as complete, with
    no warning that the declared point set and the recorded one differ.  The
    per-POINT resume inside the grid was correctly fingerprinted; only this
    enclosing skip was not.

    Raises RuntimeError on a mismatch — the same way `_check_resume` and the
    per-point `records copies` check already do.
    """
    enabled = _ratio_grid_enabled()
    cfg = _ratio_grid_from_config() if enabled else None
    recorded = {pc: (e or {}).get("ratio_grid")
                for pc, e in (target_results or {}).items()
                if isinstance(e, dict)}

    if not enabled:
        if any(recorded.values()):
            logger.error(
                "  %s: the recorded Phase 4 result came from a RATIO GRID but "
                "PHASE4_RATIO_GRID_ENABLED is now off. The resumed entry is the "
                "winning grid point re-keyed to the Phase 3 pc_id, not a "
                "default-composition leg. Nothing is recomputed.", target)
        return

    if not cfg:
        return

    for pc_id, rg in recorded.items():
        if not rg:
            raise RuntimeError(
                f"RESUME REFUSED for {target}/{pc_id}: a ratio grid is enabled "
                f"({cfg['source_keys']}) but the recorded Phase 4 result carries "
                f"no ratio_grid block — it is a single-composition leg from a "
                f"previous configuration. Resuming it would report the old leg "
                f"as the sweep. Re-run with --fresh, or move "
                f"phase4_md_results.json aside.")
        try:
            points = _grid_points_to_copies(
                cfg["grid"], rg.get("crosslinker"),
                rg.get("functional_monomers") or [],
                cfg.get("max_monomers_in_shell"))
        except Exception as e:
            raise RuntimeError(
                f"RESUME REFUSED for {target}/{pc_id}: the declared ratio grid "
                f"cannot even be normalised against the recorded composition "
                f"({rg.get('functional_monomers')} + {rg.get('crosslinker')!r}): "
                f"{e}") from e
        declared_fp = _grid_points_fingerprint(points)
        recorded_fp = rg.get("grid_fingerprint")
        if recorded_fp != declared_fp:
            declared_labels = [p["label"] for p in points]
            recorded_labels = sorted((rg.get("per_point_summary") or {}).keys())
            raise RuntimeError(
                f"RESUME REFUSED for {target}/{pc_id}: the DECLARED ratio grid "
                f"is not the grid that is recorded on disk.\n"
                f"  declared ({cfg['source_keys']}): {declared_labels} "
                f"[fingerprint {declared_fp}]\n"
                f"  recorded in phase4_md_results.json: {recorded_labels} "
                f"[fingerprint {recorded_fp}]\n"
                f"The per-target resume runs before the grid is consulted, so "
                f"continuing would silently return the OLD grid and report it as "
                f"complete. Re-run with --fresh, point --output-dir somewhere "
                f"else, or restore the recorded grid.")
        logger.info(
            "  %s/%s: resumed ratio grid matches the declared grid "
            "(fingerprint %s, %d points)", target, pc_id, declared_fp,
            len(points))


def _grid_points_fingerprint(points: list) -> str:
    """Order-independent sha256 of a normalised grid's compositions.

    Stamped into the Phase 4 result so a per-target RESUME can tell whether the
    grid it is about to skip is the grid that is now declared.
    """
    payload = sorted(
        json.dumps({m: int(n) for m, n in p["copies"].items()}, sort_keys=True)
        for p in points)
    return hashlib.sha256("|".join(payload).encode()).hexdigest()[:16]


# ── Replica bookkeeping ───────────────────────────────────────

def _replica_seed(target: str, pc_id: str, replica_index: int) -> int:
    """Deterministic, distinct, GROMACS-legal (1 .. 2^31-2) seed per replica."""
    h = hashlib.sha256(f"{target}|{pc_id}|rep{replica_index}".encode()).digest()
    return (int.from_bytes(h[:4], "big") % (2 ** 31 - 2)) + 1


def _replica_dir(pc_dir: Path, replica_index: int) -> Path:
    return Path(pc_dir) / f"rep_{replica_index}"


def run_phase4(phase1_results: dict = None,
               phase3_results: dict = None,
               target_names: list = None,
               output_dir: str = None,
               quick: bool = False,
               cross_reactivity: bool = True,
               n_replicas: int = None,
               fresh: bool = False) -> dict:
    """
    Phase 4 entry point.

    For each target's top PCs from Phase 3, run PHASE4_N_REPLICAS INDEPENDENT
    pre-polymerisation trajectories, accept/reject each replica on its own
    equilibration + convergence criteria, and pool only the accepted ones.
    """
    from .config import (MMSD_TOP_PC, MD_PRODUCTION_NS,
                         MD_QUICK_NS, EPITOPE_MONOMER_MOLAR_RATIO,
                         get_output_path)

    if output_dir is None:
        output_dir = str(get_output_path("phase4"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # A1: membrane-mode input validation. Fail fast if the user set the flag
    # but never uploaded CHARMM-GUI outputs — otherwise Phase 4 spends hours
    # solvating an ECL2-only box that will silently be marked as "membrane" in
    # the manifest. We check INPUT presence here; the actual consumption of
    # step5_input.gro is done downstream when solvation is set up (marked
    # NotImplementedError at the current wiring point until the user commits
    # their bilayer files).
    try:
        from .config import (PHASE4_MEMBRANE_MODE, PHASE4_TEMPLATE_MODE,
                              PHASE4_MEMBRANE_INPUT_DIR, PROJECT_ROOT)
    except Exception:
        PHASE4_MEMBRANE_MODE = False
        PHASE4_TEMPLATE_MODE = "ecl2"
        PHASE4_MEMBRANE_INPUT_DIR = ""
        PROJECT_ROOT = "."
    if PHASE4_MEMBRANE_MODE or PHASE4_TEMPLATE_MODE == "membrane":
        if not (PHASE4_MEMBRANE_MODE and PHASE4_TEMPLATE_MODE == "membrane"):
            raise ValueError(
                "PHASE4_MEMBRANE_MODE and PHASE4_TEMPLATE_MODE='membrane' must "
                "be set TOGETHER. Membrane mode requires both flags.")
        missing_inputs = []
        expected_files = ("step5_input.gro", "step5_input.top",
                           "step5_input.pdb")
        for t in target_names:
            tdir = Path(PROJECT_ROOT) / PHASE4_MEMBRANE_INPUT_DIR / t
            for fn in expected_files:
                if not (tdir / fn).is_file():
                    missing_inputs.append(f"{tdir/fn}")
        if missing_inputs:
            raise FileNotFoundError(
                f"PHASE4_MEMBRANE_MODE=True but the required CHARMM-GUI "
                f"Membrane Builder outputs are absent:\n  "
                + "\n  ".join(missing_inputs) + "\n\n"
                f"Upload each target's full-length PDB to CHARMM-GUI "
                f"Membrane Builder → OPM orient → POPC bilayer (or your lipid "
                f"mix) → run step5, then copy step5_input.{{gro,top,pdb}} into "
                f"{PROJECT_ROOT}/{PHASE4_MEMBRANE_INPUT_DIR}/<target>/.")
        logger.info(
            "A1: membrane mode ON — will consume CHARMM-GUI Membrane Builder "
            "outputs from %s. Lipid POSRES k=%s during eq.",
            PHASE4_MEMBRANE_INPUT_DIR,
            getattr(__import__("pipeline.config", fromlist=["_"]),
                    "PHASE4_MEMBRANE_POSRES_LIPID_K", 200))
        # The downstream membrane-mode consumer lives in
        # utils_gromacs.setup_from_charmm_gui_membrane() and is called by
        # _setup_membrane_leg() below (in place of the standard box/solvate
        # path). The check here just fails fast on missing inputs — the
        # actual coordinate + topology loading happens per-leg.
        try:
            from .utils_vesicle import load_template_ev  # sanity import
            del load_template_ev
        except Exception as e:
            raise RuntimeError(
                f"PHASE4_MEMBRANE_MODE=True but utils_vesicle failed to "
                f"import ({e}). Membrane-mode setup cannot proceed.")

    if phase1_results is None:
        with open(get_output_path("phase1") / "phase1_results.json") as f:
            phase1_results = json.load(f)

    if phase3_results is None:
        with open(get_output_path("phase3") / "phase3_mmsd_results.json") as f:
            phase3_results = json.load(f)

    if target_names is None:
        target_names = [t for t in phase3_results.keys()
                        if not isinstance(phase3_results[t], str)]

    # Single source of truth for the Phase 3 output contract — imported rather
    # than duplicated, so a future Phase 3 schema bump cannot leave this gate
    # silently checking a stale string. Imported here (not at module scope):
    # phase3_mmsd is a peer module and a top-level import would be circular.
    from .phase3_mmsd import _PHASE3_SCHEMA as _PHASE3_REQUIRED_SCHEMA

    time_ns = MD_QUICK_NS if quick else MD_PRODUCTION_NS
    total_monomers = EPITOPE_MONOMER_MOLAR_RATIO

    if n_replicas is None:
        n_replicas = int(_cfg("PHASE4_N_REPLICAS"))
    n_replicas = max(1, int(n_replicas))
    logger.info(f"Phase 4: {n_replicas} independent replica(s) per (target, pc_id)")

    # Per-target resume: load partial results if previous run was interrupted
    partial_json = output_dir / "phase4_md_results.json"
    results = {}
    if fresh and partial_json.exists():
        # --fresh: honoured by the phase itself, not only by run_pipeline having
        # archived the directory. NOTE this only disarms the TOP-LEVEL per-target
        # skip; per-replica reuse of an existing md/ directory is separately
        # guarded by the phase4_inputs.json fingerprint, which refuses a
        # directory whose inputs changed.
        logger.warning("--fresh: IGNORING existing phase4_md_results.json; "
                       "every requested target will be recomputed")
    elif partial_json.exists():
        try:
            with open(partial_json) as f:
                results = json.load(f)
            done_targets = [t for t in results
                            if isinstance(results.get(t), dict) and results[t]]
            if done_targets:
                logger.info(f"Resume: loaded existing Phase 4 results for "
                            f"{done_targets} — skipping these targets")
        except Exception as e:
            logger.warning(f"Could not parse existing phase4_md_results.json: {e}")
            results = {}

    for target in target_names:
        # Skip if already completed in a prior run.
        # GUARDED: the skip used to run before the ratio grid was consulted, so
        # an edited BSA_RATIO_GRID was silently ignored and the OLD grid was
        # reported as complete. The guard raises on a grid mismatch.
        if isinstance(results.get(target), dict) and results[target]:
            _guard_resumed_target_against_grid(target, results[target])
            logger.info(f"\n{'='*20} Phase 4: {target} (RESUMED — already done) {'='*20}")
            continue

        p3_data = phase3_results.get(target, {})
        if "error" in p3_data:
            continue

        # ── PHASE 3 PROVENANCE GATE ────────────────────────────────
        # Phase 4 is where the GPU month is spent. It used to accept any dict
        # carrying a pc_id and a monomer list — including a pre-BLOCKER-05
        # Phase 3 entry, whose smd_sum was 0.0 for every record, whose synergy
        # flag was therefore true for every record, and whose NSGA-II Pareto
        # front had mmsd_result=null throughout (so the "Pareto" pick had
        # collapsed to argmin of a single scalar). Phase 3 now stamps its
        # output; refuse anything that is not stamped.
        _p3_schema = p3_data.get("schema")
        if _p3_schema != _PHASE3_REQUIRED_SCHEMA:
            raise RuntimeError(
                f"Phase 3 result for {target} carries schema={_p3_schema!r}, not "
                f"{_PHASE3_REQUIRED_SCHEMA!r}. This is a PRE-BLOCKER-05 result: its "
                f"smd_sum is 0.0 for every composition (a target-keyed matrix was "
                f"passed into the monomer-keyed slot), every record therefore reports "
                f"synergy, and its Pareto front carries no mmsd_result so the final "
                f"pick was argmin of one scalar rather than the three objectives. "
                f"Re-run Phase 3 before spending MD time on its output.")

        top_pcs = p3_data.get("top_pcs", [])

        # A composition picked by the scalar fallback rather than from the
        # Pareto front is usable but is NOT what the NSGA-II run was for.
        for _pc in top_pcs[:MMSD_TOP_PC]:
            _sel = str((_pc or {}).get("selected_from", ""))
            if _sel.startswith("fallback_"):
                logger.error(
                    "  %s/%s was selected by %s — the NSGA-II Pareto front was "
                    "unusable and the pick collapsed to argmin(bo_objective). "
                    "The MD below is valid; the COMPOSITION CHOICE is not "
                    "multi-objective. See phase3 pareto_selection.fallback_reason.",
                    target, (_pc or {}).get("pc_id"), _sel)
        # Use full ECL2 (not 16-mer head) for pre-polymerization MD.
        # Rationale: actual MIP synthesis uses the whole protein (~90 aa ECL2 is
        # the only solvent-accessible face of tetraspanin TM4 proteins). Head-only
        # imprinting (epitope MIP) fails because isolated 16-mer is too flexible —
        # cavity formed against unstable peptide doesn't discriminate other heads
        # at rebinding time (observed in initial head-template run: SI=0.85, 1/10
        # rebound). Whole-ECL2 imprinting is closer to the experimental reality
        # (surface MIP) and is what the adenovirus eIP paper (PMC11059108) and
        # other 2024 epitope-MIP work converged on after similar failures.
        # Protein Cα backbone is restrained in MD (k=1000 kJ/mol/nm²) to mimic
        # solid-phase / surface immobilization during polymerization.
        from .config import resolve_path
        epitope_pdb = resolve_path(phase1_results[target].get("ecl2_pdb",
                                   phase1_results[target].get("head_pdb",
                                   phase1_results[target]["epitope_pdb"])))

        logger.info(f"\n{'='*20} Phase 4: {target} "
                    f"({len(top_pcs)} PCs, {time_ns}ns, "
                    f"1:{total_monomers} ratio, {n_replicas} replicas) {'='*20}")

        target_results = {}
        for pc_data in top_pcs[:MMSD_TOP_PC]:
            pc_id = pc_data["pc_id"]
            monomers = pc_data["monomers"]
            # Separate functional monomers from crosslinker
            # Crosslinker is stored in pc_data by Phase 3, or detect from monomers list
            from .config import CROSSLINKER_LIBRARY
            crosslinker = pc_data.get("crosslinker")
            if crosslinker is None:
                # Fallback: last monomer in MMSD list is the crosslinker
                for m in reversed(monomers):
                    if m in CROSSLINKER_LIBRARY:
                        crosslinker = m
                        break
            functional = [m for m in monomers if m != crosslinker]

            logger.info(f"\n--- {target}/{pc_id}: {functional} + {crosslinker} ---")

            pc_dir = output_dir / target / pc_id

            # ── RATIO GRID vs SINGLE COMPOSITION ──────────────────
            # When a ratio grid is declared AND enabled, the grid REPLACES the
            # single-composition leg rather than running beside it. Running both
            # would spend a full leg on a composition nobody chose: with one
            # functional monomer the default composition is
            # SOLGEL_Q_MOLE_FRACTION=0.60 -> 36 TEOS / 24 APTES, i.e. x=0.40,
            # which is not a member of BSA_RATIO_GRID at all.
            _grid_cfg = _ratio_grid_from_config() if _ratio_grid_enabled() else None
            if _grid_cfg:
                logger.info(f"  PHASE4_RATIO_GRID_ENABLED — Phase 4 for "
                            f"{target}/{pc_id} IS the ratio grid "
                            f"({_grid_cfg['source_keys']})")
                grid_summary = run_phase4_ratio_grid(
                    target=target, pc_id=pc_id,
                    functional_monomers=functional,
                    crosslinker=crosslinker,
                    epitope_pdb=epitope_pdb,
                    work_dir=pc_dir / "ratio_grid",
                )
                best = grid_summary.get("best_point")
                if best:
                    md_result = dict(grid_summary["per_point"][best]["pooled"])
                    # Phases 5/6 look this entry up by the PHASE 3 pc_id, so the
                    # winning leg has to answer to it. Its own grid-qualified id
                    # (pc_id@label) is kept beside it, not thrown away.
                    md_result["grid_point_pc_id"] = md_result.get("pc_id")
                    md_result["grid_point_label"] = best
                    md_result["pc_id"] = pc_id
                    md_result["ratio_grid_decision"] = grid_summary.get("decision")
                    md_result["ratio_grid_best_point_qualified"] = (
                        grid_summary.get("best_point_qualified"))
                    if not grid_summary.get("best_point_qualified"):
                        logger.error(
                            "  %s/%s: the grid point handed to Phases 5/6 (%s) is "
                            "NOT a qualified optimum — decision=%s, margin over "
                            "the runner-up %s%%, tied with %s. Phases 5/6 will "
                            "consume it as a representative composition. Do not "
                            "publish it as 'the optimal ratio'.",
                            target, pc_id, best, grid_summary.get("decision"),
                            grid_summary.get("margin_over_runner_up_pct"),
                            grid_summary.get("tied_points"))
                else:
                    # No point ranked. Do NOT invent a representative leg —
                    # Phases 5/6 must see a failed Phase 4, not a silent pick.
                    md_result = {
                        "target": target, "pc_id": pc_id,
                        "functional_monomers": functional,
                        "crosslinker": crosslinker,
                        "time_ns": grid_summary.get("time_ns"),
                        "success": False, "accepted": False,
                        "error": ("ratio grid produced no complete + accepted "
                                  "point; see ratio_grid_summary.json"),
                    }
                md_result["ratio_grid"] = {
                    k: v for k, v in grid_summary.items() if k != "per_point"}
                md_result["ratio_grid"]["per_point_summary"] = {
                    lbl: {k: v for k, v in e.items() if k != "pooled"}
                    for lbl, e in grid_summary.get("per_point", {}).items()}

                # SECONDARY axis, off by default: a FIXED composition re-run at
                # several n_total values, to separate a composition effect from
                # a crowding artefact. Gated separately because it is another
                # 3 x n_replicas legs on top of the grid.
                #
                # IT IS NO LONGER CONDITIONED ON A BARE ARGMAX.
                # # BEHAVIOUR CHANGE. Under a flat truth the grid names
                # n_APTES=3 about 30% of the time, and with a real +30% effect
                # at R=3 it names the true optimum only ~28% of the time. Making
                # the second axis follow that pick spends ~2.7 GPU-days
                # resolving the loading dependence of a composition that is,
                # with ~72% probability, not the optimum. The composition is
                # therefore taken from BSA_LOADING_SWEEP_COMPOSITION when it is
                # pinned, and otherwise only from a QUALIFIED winner.
                if _loading_sweep_enabled():
                    pinned = (_cfg_get("PHASE4_LOADING_SWEEP_COMPOSITION")
                              or _cfg_get("BSA_LOADING_SWEEP_COMPOSITION"))
                    require_qual = bool(
                        _cfg("PHASE4_LOADING_SWEEP_REQUIRE_QUALIFIED_WINNER"))
                    if pinned:
                        _copies = dict(pinned)
                        _src = "pinned (BSA_LOADING_SWEEP_COMPOSITION)"
                    elif grid_summary.get("best_copies") and (
                            grid_summary.get("best_point_qualified")
                            or not require_qual):
                        _copies = grid_summary["best_copies"]
                        _src = ("qualified grid winner"
                                if grid_summary.get("best_point_qualified")
                                else "UNQUALIFIED grid winner "
                                     "(PHASE4_LOADING_SWEEP_REQUIRE_QUALIFIED_"
                                     "WINNER is False)")
                    else:
                        _copies = None
                        _src = None
                    if _copies:
                        logger.info(f"  loading sweep composition source: {_src}")
                        try:
                            md_result["loading_sweep"] = run_phase4_loading_sweep(
                                target=target, pc_id=pc_id,
                                functional_monomers=functional,
                                crosslinker=crosslinker,
                                epitope_pdb=epitope_pdb,
                                work_dir=pc_dir / "loading_sweep",
                                winning_copies=_copies,
                                composition_source=_src)
                        except Exception as e:
                            logger.error(f"  loading sweep failed: {e}")
                            md_result["loading_sweep"] = {"error": str(e),
                                                          "success": False}
                    else:
                        _why = (
                            f"the grid's top point {grid_summary.get('best_point')!r} "
                            f"is NOT qualified (margin "
                            f"{grid_summary.get('margin_over_runner_up_pct')}%, tied "
                            f"with {grid_summary.get('tied_points')})"
                            if grid_summary.get("best_copies")
                            else "the ratio grid named no winner")
                        logger.error(
                            "  LOADING SWEEP SKIPPED: %s, and no composition is "
                            "pinned in BSA_LOADING_SWEEP_COMPOSITION. Running the "
                            "loading axis on an unqualified argmax would spend a "
                            "second GPU-week resolving the loading dependence of a "
                            "composition that is probably not the optimum. Pin the "
                            "composition, or raise PHASE4_N_REPLICAS and re-run the "
                            "x-sweep first.", _why)
                        md_result["loading_sweep"] = {
                            "skipped": True, "reason": _why, "success": False}
                target_results[pc_id] = md_result
            else:
                replica_results = []
                for rep in range(n_replicas):
                    logger.info(f"\n  >>> {target}/{pc_id} replica {rep + 1}/{n_replicas} "
                                f"(seed={_replica_seed(target, pc_id, rep)}) <<<")
                    rep_res = _run_prepolymerization_md(
                        target=target,
                        pc_id=pc_id,
                        functional_monomers=functional,
                        crosslinker=crosslinker,
                        epitope_pdb=epitope_pdb,
                        work_dir=_replica_dir(pc_dir, rep),
                        time_ns=time_ns,
                        total_monomers=total_monomers,
                        replica_index=rep,
                        n_replicas=n_replicas,
                    )
                    replica_results.append(rep_res)

                md_result = _pool_replicas(target, pc_id, functional, crosslinker,
                                           replica_results, time_ns, pc_dir,
                                           output_dir)
                target_results[pc_id] = md_result

            # A5/B7: Solvent sweep (porogen MD)
            from .config import PHASE4_SOLVENT_SWEEP, PHASE4_RATIO_SWEEP
            if PHASE4_SOLVENT_SWEEP:
                logger.info(f"  A5/B7: Solvent sweep for {pc_id}...")
                try:
                    solv_result = run_phase4_solvent_sweep(
                        target=target, pc_id=pc_id,
                        functional_monomers=functional,
                        crosslinker=crosslinker,
                        epitope_pdb=epitope_pdb,
                        work_dir=pc_dir / "solvent_sweep",
                        time_ns=30,
                    )
                    md_result["solvent_sweep"] = solv_result
                except Exception as e:
                    logger.warning(f"  A5/B7 solvent sweep failed: {e}")

            # B6: Legacy preset ratio sweep. Mutually exclusive with the grid —
            # the grid already IS the ratio sweep, and running both would build
            # the same compositions twice under two different labels.
            if PHASE4_RATIO_SWEEP and not _grid_cfg:
                logger.info(f"  B6: Monomer ratio sweep for {pc_id}...")
                try:
                    ratio_result = run_phase4_ratio_sweep(
                        target=target, pc_id=pc_id,
                        functional_monomers=functional,
                        crosslinker=crosslinker,
                        epitope_pdb=epitope_pdb,
                        work_dir=pc_dir / "ratio_sweep",
                        time_ns=30,
                    )
                    md_result["ratio_sweep"] = ratio_result
                except Exception as e:
                    logger.warning(f"  B6 ratio sweep failed: {e}")

        results[target] = target_results

        # Per-target incremental save — survives crashes
        try:
            with open(output_dir / "phase4_md_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"  Phase 4: {target} complete → saved partial results")
        except Exception as e:
            logger.warning(f"  Failed to save partial Phase 4 results: {e}")

    # Cross-reactivity removed — Phase 6 cavity rebinding handles selectivity

    # Final save (in case all 3 targets ran in one go)
    with open(output_dir / "phase4_md_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Quality report goes in its OWN file. It must NOT become a top-level key of
    # phase4_md_results.json: phase5/phase6 do
    #     target_names = [t for t in phase4_results if t != "cross_reactivity"]
    # and would treat any extra key as a target name.
    _write_quality_report(results, output_dir)

    _print_phase4_summary(results, target_names)

    return results


# ── Box composition ───────────────────────────────────────────

def _polymerization_type(combo: list) -> str:
    """'sol-gel' (silane / Q-T network) or 'radical' (vinyl, catechol)."""
    try:
        from .config import is_polymerization_compatible
        ok, dom, _ = is_polymerization_compatible(combo)
        if ok and dom:
            return "sol-gel" if dom == "silane" else "radical"
    except Exception as e:
        # Raised from DEBUG (REVIEW FINDING 19, same class): the fallback below
        # returns "radical" for anything it cannot classify, and the
        # polymerization type selects the crosslinker mole fraction
        # (SOLGEL_Q_MOLE_FRACTION 0.60 vs RADICAL_CROSSLINKER_MOLE_FRACTION
        # 0.50). Misclassifying a sol-gel box as radical silently changes what
        # gets simulated AND the synthesis ratio derived from it.
        logger.error(f"is_polymerization_compatible unavailable ({e}) — "
                     f"falling back to composition inspection for {combo}")
    try:
        from .config import ALL_MONOMERS
        types = {(ALL_MONOMERS.get(m) or {}).get("polymerization")
                 or (ALL_MONOMERS.get(m) or {}).get("type") for m in combo}
        if "silane" in types:
            return "sol-gel"
    except Exception as e:
        logger.error(
            f"could not inspect monomer chemistry for {combo} ({e}) — "
            f"defaulting to 'radical', which sets the crosslinker mole "
            f"fraction to RADICAL_CROSSLINKER_MOLE_FRACTION. If this box is "
            f"actually a silane sol-gel, its composition and its synthesis "
            f"ratio are both wrong.")
    return "radical"


def _monomer_copy_numbers(functional_monomers: list, crosslinker: str,
                          total_monomers: int) -> dict:
    """Copy numbers for the pre-polymerisation box.

    # BEHAVIOUR CHANGE (2026-08 audit) — SOL-GEL SYSTEMS ONLY.
    Old code:  copies_per_functional = total // (n_functional + 1)
               → with total=25, n_functional=3 that is 6/6/6 T + 7 Q,
                 i.e. 72 mol% organosilane (T units). The closest published
                 protein-imprinted sol-gel analogue runs ~60 mol% Q.
    New code:  n_crosslinker = round(total * SOLGEL_Q_MOLE_FRACTION), the
               remainder split as evenly as possible over the functional
               monomers.  `total_monomers` is UNCHANGED, so box size, atom
               count and therefore GPU cost are unchanged.

    Free-radical systems keep the legacy formula exactly (it is not what the
    audit flagged, and changing it would move an unrelated published baseline).
    Returns a dict with copies + the composition metadata that is stored in the
    run manifest and in the result JSON.
    """
    functional_monomers = list(functional_monomers)
    n_func = len(functional_monomers)
    if n_func == 0:
        raise ValueError("no functional monomers")
    if crosslinker in functional_monomers:
        raise ValueError(f"crosslinker {crosslinker!r} also listed as functional")

    combo = functional_monomers + [crosslinker]
    poly_type = _polymerization_type(combo)

    if poly_type == "sol-gel":
        q_target = float(_cfg("SOLGEL_Q_MOLE_FRACTION"))
        if not 0.0 < q_target < 1.0:
            raise ValueError(
                f"SOLGEL_Q_MOLE_FRACTION={q_target!r} must be in (0, 1)")
        n_cross = int(round(total_monomers * q_target))
        # Every functional monomer needs at least one copy.
        n_cross = min(max(n_cross, 1), total_monomers - n_func)
        rest = total_monomers - n_cross
        if rest < n_func:
            raise ValueError(
                f"total_monomers={total_monomers} cannot hold {n_func} functional "
                f"monomers at Q={q_target:.2f} (only {rest} non-Q slots). "
                f"Raise EPITOPE_MONOMER_MOLAR_RATIO or lower SOLGEL_Q_MOLE_FRACTION.")
        base, rem = divmod(rest, n_func)
        copies = {m: base + (1 if i < rem else 0)
                  for i, m in enumerate(functional_monomers)}
        copies[crosslinker] = n_cross
    else:
        q_target = float(_cfg("RADICAL_CROSSLINKER_MOLE_FRACTION"))
        # LEGACY formula, preserved verbatim for free-radical systems.
        copies_per_functional = max(1, total_monomers // (n_func + 1))
        copies = {m: copies_per_functional for m in functional_monomers}
        copies[crosslinker] = total_monomers - (copies_per_functional * n_func)

    total = sum(copies.values())
    if total != total_monomers:
        raise ValueError(
            f"copy-number bug: {copies} sums to {total}, expected {total_monomers}")
    if any(v < 1 for v in copies.values()):
        raise ValueError(f"copy-number bug: non-positive copies in {copies}")

    return {
        "copies": copies,
        "requested_copies": dict(copies),
        "polymerization_type": poly_type,
        "crosslinker_mole_fraction_target": round(q_target, 4),
        "crosslinker_mole_fraction_actual": round(copies[crosslinker] / total, 4),
        "total_monomers": total,
        # Species that are actually IN the box. Identical to the requested set
        # here (this path can never produce a zero copy), but present so every
        # caller can read the same two keys whichever path built the box.
        "present_functional": list(functional_monomers),
        "present_crosslinker": crosslinker,
        "composition_source": ("SOLGEL_Q_MOLE_FRACTION" if poly_type == "sol-gel"
                               else "RADICAL_CROSSLINKER_MOLE_FRACTION"),
    }


def _protonation_split_spec() -> dict:
    """{parent_monomer: {"protonated_name":..., "pka":...}} or {}.

    THE DEFECT THIS CLOSES.  Every leg modelled APTES as a NEUTRAL primary
    amine — the freshly generated topology has N typed n8 with two `hn`
    hydrogens at +0.3493 and qtot 0.000000 — while at pH 9.5 the aminopropyl
    amine is substantially -NH3+, and that ammonium against BSA carboxylate
    (pI 4.7) IS the imprinting driving force this grid exists to measure.  At
    24 copies that was the whole cationic population missing from the box.
    (An earlier version of this docstring quoted pKa 10.6 / f = 0.926 / ~+22 e.
    The measured aminopropylsilanetriol pKa is 9.2, so the fraction — and hence
    that charge — depends steeply on the pH and is no longer a fixed number.
    See MONOMER_PROTONATION_SPLIT in config_BSA.py.)  The root cause
    (acpype's hardcoded `-n 0`) is fixed in utils_gromacs; this is the other
    half — actually PUTTING the protonated species in the box, in the
    Henderson-Hasselbalch proportion, instead of only being able to.
    """
    spec = _cfg_get("MONOMER_PROTONATION_SPLIT") or {}
    if not spec:
        return {}
    ph = _cfg_get("MD_SOLVENT_PH")
    if ph is None:
        logger.error("MONOMER_PROTONATION_SPLIT is declared but MD_SOLVENT_PH "
                     "is not — no split can be computed; every monomer stays "
                     "neutral.")
        return {}
    out = {}
    for parent, s in spec.items():
        pka = float(s["pka"])
        f = 1.0 / (1.0 + 10.0 ** (float(ph) - pka))
        out[parent] = {"protonated_name": s["protonated_name"],
                       "pka": pka, "ph": float(ph),
                       "fraction_protonated": round(f, 4)}
    return out


def _apply_protonation_split(built: dict, spec: dict) -> tuple:
    """Split each parent species into (neutral, protonated) at its HH fraction.

    Returns (new_copies, record).  Rounding is deterministic (round-half-up on
    the protonated count) and the parent is DROPPED when the split leaves it at
    zero, so `_verify_built_composition` still compares like with like.
    """
    if not spec:
        return dict(built), {}
    out, record = {}, {}
    for m, n in built.items():
        s = spec.get(m)
        if not s or n <= 0:
            out[m] = out.get(m, 0) + n
            continue
        n_prot = int(math.floor(n * s["fraction_protonated"] + 0.5 + 1e-9))
        n_neut = n - n_prot
        if n_neut > 0:
            out[m] = out.get(m, 0) + n_neut
        if n_prot > 0:
            out[s["protonated_name"]] = out.get(s["protonated_name"], 0) + n_prot
        record[m] = {**s, "n_requested": n, "n_neutral": n_neut,
                     "n_protonated": n_prot,
                     "protonated_name": s["protonated_name"]}
        logger.info(
            "  pH %.2f: %d %s split into %d neutral + %d %s "
            "(pKa %.2f -> %.1f%% protonated). The +%d e enters genion's "
            "neutralisation — against a %s protein it simply removes that many "
            "Na+; only a net-positive box draws Cl-. MD_IONIC_STRENGTH stays "
            "0.0, so DI water is still counter-ions only.",
            s["ph"], n, m, n_neut, n_prot, s["protonated_name"], s["pka"],
            s["fraction_protonated"] * 100, n_prot, "negative BSA")
    return out, record


def _merge_protonation_forms(values: dict, parent_of: dict) -> dict:
    """Fold a protonated form's value back onto its parent, for the RECIPE only.

    The BENCH buys one bottle of APTES. Reporting "APTES 1 part : APTESH 3
    parts : TEOS 2 parts" would be a recipe nobody can weigh out — the split is
    a solution equilibrium, not a formulation. The per-species numbers are kept
    untouched everywhere else; only the synthesis ratio is merged.
    """
    if not parent_of:
        return dict(values or {})
    out = {}
    for m, v in (values or {}).items():
        key = parent_of.get(m, m)
        if isinstance(v, (int, float)):
            out[key] = out.get(key, 0) + v
        else:
            out.setdefault(key, v)
    return out


def _composition_from_copies(functional_monomers: list, crosslinker: str,
                             copies: dict) -> dict:
    """Composition for an EXPLICITLY REQUESTED per-monomer copy-number dict.

    This is the entry point that makes a ratio grid real.  `_monomer_copy_numbers`
    derives the split from SOLGEL_Q_MOLE_FRACTION and therefore returns THE SAME
    box for every grid point; this function imposes the caller's numbers instead.

    ZERO IS ALLOWED AND MEANS ZERO.  The grid's two control points — TEOS-only
    (no functional monomer) and APTES-only (no crosslinker) — are the points
    that say whether the observable is imprinting or amine count, and both were
    previously refused by three separate guards.  A species requested at 0 is
    DROPPED from the box and from `present_functional`/`present_crosslinker`, so
    the downstream analysis is told about the species that are actually there
    rather than scoring an absent monomer as "measured 0 contacts".
    """
    functional_monomers = list(functional_monomers)
    if crosslinker is not None and crosslinker in functional_monomers:
        raise ValueError(f"crosslinker {crosslinker!r} also listed as functional")

    known = functional_monomers + ([crosslinker] if crosslinker else [])
    unknown = [m for m in copies if m not in known]
    if unknown:
        raise ValueError(
            f"explicit copies name {unknown}, which are neither functional "
            f"monomers {functional_monomers} nor the crosslinker {crosslinker!r}")

    requested = {}
    for m in known:
        v = copies.get(m, 0)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"explicit copy count for {m!r} is {v!r}, not an int")
        v = int(v)
        if v < 0:
            raise ValueError(f"explicit copy count for {m!r} is negative ({v})")
        requested[m] = v

    built = {m: n for m, n in requested.items() if n > 0}
    total = sum(built.values())
    if total == 0:
        raise ValueError(
            f"explicit copies {requested} would build an EMPTY box (no monomer "
            f"has a positive copy count)")

    # ── pH SPECIATION ────────────────────────────────────────
    # A monomer with a declared pKa is put in the box as a MIXTURE of its
    # neutral and protonated forms in the Henderson-Hasselbalch proportion at
    # MD_SOLVENT_PH, not as 100% of the neutral form. Applied AFTER the
    # requested counts are fixed, so the grid's bookkeeping (labels,
    # requested_copies, the n_total assertion) is untouched.
    split_spec = _protonation_split_spec()
    built_unsplit = dict(built)
    built, split_record = _apply_protonation_split(built, split_spec)
    total = sum(built.values())

    # The species that are actually in the box, INCLUDING the protonated forms:
    # a split parent contributes both of its forms to present_functional, so the
    # occupancy analysis, the RDF and the ranking see the whole population.
    def _forms(m):
        s = split_record.get(m)
        if not s:
            return [m] if built.get(m, 0) > 0 else []
        return [n for n in (m, s["protonated_name"]) if built.get(n, 0) > 0]

    present_functional = [n for m in functional_monomers for n in _forms(m)]
    present_crosslinker = crosslinker if built.get(crosslinker, 0) > 0 else None

    combo = present_functional + ([present_crosslinker] if present_crosslinker else [])
    poly_type = _polymerization_type(combo)
    q_key = ("SOLGEL_Q_MOLE_FRACTION" if poly_type == "sol-gel"
             else "RADICAL_CROSSLINKER_MOLE_FRACTION")
    x_actual = (built.get(crosslinker, 0) / total) if crosslinker else 0.0

    return {
        "copies": built,
        "requested_copies": requested,
        "polymerization_type": poly_type,
        # The TARGET is the composition the caller asked for, so target and
        # actual agree by construction. The library constant is recorded beside
        # it as `crosslinker_mole_fraction_config` — it is NOT what built this
        # box, and reporting it as the target is how a sweep silently becomes
        # eight copies of one composition.
        "crosslinker_mole_fraction_target": round(x_actual, 4),
        "crosslinker_mole_fraction_actual": round(x_actual, 4),
        "crosslinker_mole_fraction_config": round(float(_cfg(q_key)), 4),
        "total_monomers": total,
        "present_functional": present_functional,
        "present_crosslinker": present_crosslinker,
        "dropped_monomers": sorted(m for m, n in requested.items() if n == 0),
        "composition_source": "explicit_copies",
        # pH speciation: what was requested vs what the box holds, and the
        # parent -> protonated-form map the recipe has to merge back over.
        "copies_before_protonation_split": built_unsplit,
        "protonation_split": split_record,
        "protonation_parent_of": {
            s["protonated_name"]: m for m, s in split_record.items()},
    }


# ── H-bond selection strings ──────────────────────────────────
#
# THESE ARE THE PARAMETERS THAT MAKE A SILANOL VISIBLE.  # BEHAVIOUR CHANGE
#
# MDAnalysis' HydrogenBondAnalysis guesses BOTH sides when they are not given,
# and both guesses were wrong for this system:
#
#   hydrogens_sel unset -> guess_hydrogens(min_charge=0.3).  On the pre-fix
#       Gasteiger silane topologies NO monomer hydrogen reached 0.3, so the
#       monomer-as-DONOR direction returned identically zero — the exact
#       observable the intact-vs-hydrolysed comparison exists to measure.  It
#       also drops the PROTEIN side: an amber99sb-ildn backbone amide H carries
#       +0.2719, below the threshold, so ~10 of every 13 protein polar
#       hydrogens were silently excluded.  And because the guess returns a
#       (resname, name) selection STRING while acpype names every monomer
#       residue UNL, a charge that cleared the threshold on species A admitted
#       the same-named atom of species B — in a two-monomer box that is exactly
#       the mixture a ratio sweep is made of.
#
#   acceptors_sel given -> guess_acceptors() is NOT called and EVERY atom in the
#       selection counts as an acceptor, carbons and hydrogens included.  The
#       spurious count scales with monomer atom count (intact TEOS 33 atoms vs
#       Si(OH)4 9), so it biased the species comparison as well as inflating it.
#
# Both sides are therefore pinned explicitly, by MASS rather than by atom name
# or type: amber protein names (N, OD1, SG) and GAFF2 monomer types (oh, n8, hn)
# follow no shared convention, but a TPR always carries correct masses.
#   H  1.008 | N 14.007 | O 15.999 | S 32.06   (Si 28.085 and P 30.974 excluded)
_HB_HYDROGEN_SEL = "(prop mass > 0.9 and prop mass < 1.6)"
_HB_DONOR_ACCEPTOR_SEL = ("((prop mass > 13.9 and prop mass < 14.1) or "
                          "(prop mass > 15.9 and prop mass < 16.1) or "
                          "(prop mass > 31.9 and prop mass < 32.2))")


# ── Built-composition verification ────────────────────────────

# Protonation-state residue names. pdb2gmx -lys/-his/-asp/-glu rename titratable
# residues to encode the state the pH model chose, and MDAnalysis's `protein`
# selection does not recognise all of them -- LYSN (neutral lysine) in
# particular. In the pH 7.87 BSA model two lysines are neutral, so they fell OUT
# of `protein` and INTO the monomer selection: 102 "monomer" residues against a
# 100-molecule topology, and the positional-mapping guard then refused the whole
# occupancy analysis. The MD was fine; only the selection was wrong. Every grid
# point with a live pH model failed this way.
_PH_PROTEIN_RESNAMES = {
    "LYSN", "LYN",                                  # neutral lysine
    "HISD", "HISE", "HISH", "HID", "HIE", "HIP",    # histidine states
    "ASPH", "ASH", "GLUH", "GLH",                   # protonated Asp / Glu
    "CYM", "CYX", "CYS2",                           # deprotonated / SS cysteine
    "ARGN", "TYRO", "LYSH",                         # neutral Arg, tyrosinate, +Lys
}

_TOPOLOGY_SOLVENT_IONS = {"SOL", "WAT", "TIP3", "TIP3P", "HOH", "T3P",
                          "NA", "CL", "K", "MG", "CA", "ZN", "NA+", "CL-"}

# One set, two readers. The occupancy analysis used to keep its own private copy
# of this list, so the [ molecules ] scan and the trajectory selection could
# drift apart silently -- which is exactly how a residue can be "not solvent" to
# one and "not a monomer" to the other.
_SOLVENT_IONS = _TOPOLOGY_SOLVENT_IONS


def _read_topology_molecule_counts(top_file: Path) -> dict:
    """{molecule_name: count} from a GROMACS [ molecules ] block.

    Protein_* entries and solvent/ions are dropped, so what comes back is the
    monomer census of the box that was ACTUALLY built.
    """
    top_file = Path(top_file)
    if not top_file.exists():
        raise FileNotFoundError(f"{top_file} does not exist")
    counts = {}
    in_molecules = False
    for line in top_file.read_text().split("\n"):
        if "[ molecules ]" in line:
            in_molecules = True
            continue
        if not in_molecules:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith("["):
            break
        parts = stripped.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            n = int(parts[1])
        except ValueError:
            continue
        if name.startswith("Protein") or name.upper() in _TOPOLOGY_SOLVENT_IONS:
            continue
        counts[name] = counts.get(name, 0) + n
    return counts


def _verify_built_composition(md_dir: Path, expected_copies: dict) -> dict:
    """Assert the built topology contains EXACTLY the requested copy numbers.

    Without this the copies= plumbing is unfalsifiable: `_include_monomers_in_topology`
    places monomers from a list, and a placement failure, a truncated list or a
    resumed-but-stale directory would all produce a box whose composition nobody
    ever compared against the request.  The grid's whole claim is that point k
    differs from point k-1, so the composition is the one thing that MUST be
    verified after the fact rather than assumed.

    Returns the verification record; RAISES on any mismatch.
    """
    md_dir = Path(md_dir)
    top_file = md_dir / "topol.top"
    if not top_file.exists():
        raise RuntimeError(
            f"cannot verify the built composition: {top_file} does not exist. "
            f"The system build did not get as far as writing a topology, so the "
            f"MD step above failed for a reason of its own — read ITS error "
            f"first; this check is reporting the consequence, not the cause.")
    built = _read_topology_molecule_counts(top_file)
    expected = {m: int(n) for m, n in expected_copies.items() if int(n) > 0}
    ok = built == expected
    record = {
        "topology": str(top_file),
        "expected_copies": expected,
        "built_copies": built,
        "verified": ok,
    }
    if not ok:
        raise RuntimeError(
            f"BUILT COMPOSITION DOES NOT MATCH THE REQUEST. {top_file} carries "
            f"{built} but this leg asked for {expected}. Every per-point number "
            f"in a ratio sweep is meaningless if the box is not the box that was "
            f"requested — refusing to analyse it. Delete {md_dir} and re-run.")
    logger.info(f"    Composition verified against {top_file.name}: {built}")
    return record


def _derive_optimal_ratio(ebn: dict, functional_monomers: list,
                          crosslinker: str, poly_type: str,
                          _crosslinker_mole_fraction: float = None) -> dict:
    """Synthesis ratio: functional parts from EBN, crosslinker from mole fraction.

    `_crosslinker_mole_fraction` overrides the library constant with the
    AS-BUILT crosslinker mole fraction of the box this EBN came from. Pass it
    for any explicitly-composed box (ratio grid): otherwise every grid point
    publishes the same SOLGEL_Q_MOLE_FRACTION-derived crosslinker term
    regardless of what was simulated.

    # BEHAVIOUR CHANGE (2026-08 audit).
    Old code:  optimal_ratio[crosslinker] = sum(optimal_ratio.values())
               → crosslinker is ALWAYS exactly 50 mol% of the recipe, for every
                 monomer set, no matter what the EBN data said.
    New code:  crosslinker parts = round(sum(functional parts) * x / (1 - x))
               where x is SOLGEL_Q_MOLE_FRACTION (sol-gel) or
               RADICAL_CROSSLINKER_MOLE_FRACTION (radical, default 0.50 =
               unchanged legacy value). Functional parts still come from EBN.
    """
    # DO NOT CLAMP AWAY A MISSING MEASUREMENT (REVIEW FINDING 13).
    #
    # This was `max(int(ebn.get(m, 1) or 0), 1)`, which mapped BOTH a measured
    # EBN of 0 (the monomer never contacted the protein) AND an absent EBN key
    # (never measured at all) to 1 — and then published the CLAMPED value as
    # optimal_ratio_basis.functional_EBN. The provenance field a reviewer would
    # read to audit the recipe therefore erased the fact that the measurement
    # was zero or missing. Combined with an all-zero occupancy analysis, that
    # produced a clean-looking 1:1:1:4 wet-lab recipe from no data at all.
    ebn = ebn or {}
    if not functional_monomers:
        # A crosslinker-only control box (the grid's (60,0) TEOS-only point).
        # There is no functional monomer to derive parts FROM, so there is no
        # synthesis ratio — say so instead of inventing a 1:1.
        raise ValueError(
            "cannot derive a synthesis ratio: this box contains no functional "
            "monomer (crosslinker-only control). A control point is a baseline "
            "for the observable, not a recipe.")
    raw_ebn = {m: ebn.get(m, None) for m in functional_monomers}
    absent = [m for m, v in raw_ebn.items() if v is None]
    zero = [m for m, v in raw_ebn.items() if v is not None and int(v) <= 0]

    if absent:
        raise ValueError(
            f"cannot derive a synthesis ratio: {absent} have NO EBN entry "
            f"(measured EBN keys: {sorted(ebn)}). A monomer with no measured "
            f"binding cannot be assigned a ratio — the old code silently gave "
            f"it 1 part and reported that 1 as if it were the measurement.")
    if zero:
        raise ValueError(
            f"cannot derive a synthesis ratio: {zero} have EBN=0, i.e. they "
            f"never made a single simultaneous contact with the epitope over "
            f"the analysed window. Either the occupancy analysis failed (see "
            f"the residue->monomer mapping guard) or these monomers do not "
            f"belong in the composition. The old code clamped them to 1 part "
            f"and published that 1 as the EBN.")

    functional_ebn = {m: int(v) for m, v in raw_ebn.items()}
    if functional_ebn:
        min_ebn = min(functional_ebn.values())
        ratio = {m: max(1, round(v / min_ebn)) for m, v in functional_ebn.items()}
    else:
        ratio = {m: 1 for m in functional_monomers}

    if crosslinker is None:
        # Functional-only control box (the grid's (0,60) APTES-only point).
        # No crosslinker was simulated, so none is derived. The recipe is
        # honest about being a control rather than silently re-inserting the
        # library's crosslinker mole fraction.
        total_parts = sum(ratio.values())
        return {
            "optimal_ratio": ratio,
            "optimal_ratio_basis": {
                "functional_parts_from": "EBN (max simultaneous contacts)",
                "crosslinker_parts_from": "NONE — crosslinker-free control box",
                "crosslinker_mole_fraction_target": 0.0,
                "crosslinker_mole_fraction_achieved": 0.0,
                "functional_EBN": dict(raw_ebn),
                "functional_EBN_is_raw": True,
                "is_control_box": True,
            },
        }

    key = ("SOLGEL_Q_MOLE_FRACTION" if poly_type == "sol-gel"
           else "RADICAL_CROSSLINKER_MOLE_FRACTION")
    x = _crosslinker_mole_fraction if _crosslinker_mole_fraction is not None \
        else float(_cfg(key))
    x = min(max(float(x), 1e-6), 1 - 1e-6)
    sum_f = sum(ratio.values())
    # FLOAT EDGE CASE (2026-08 audit): x=0.60 gives x/(1-x) =
    # 1.4999999999999998 in IEEE754, and round() then returned 1 — publishing a
    # 1:1 recipe (50 mol% crosslinker) from a 60 mol% target, contradicting the
    # box that was simulated. Nudge by one ulp-scale epsilon before rounding
    # half-up explicitly (round() also does banker's rounding at exact .5).
    parts_x = sum_f * x / (1.0 - x)
    ratio[crosslinker] = max(1, int(math.floor(parts_x + 0.5 + 1e-9)))

    total_parts = sum(ratio.values())
    return {
        "optimal_ratio": ratio,
        "optimal_ratio_basis": {
            "functional_parts_from": "EBN (max simultaneous contacts)",
            "crosslinker_parts_from": (
                "as-built box composition" if _crosslinker_mole_fraction is not None
                else key),
            "crosslinker_mole_fraction_target": round(x, 4),
            "crosslinker_mole_fraction_achieved":
                round(ratio[crosslinker] / total_parts, 4),
            # The RAW measurement, exactly as the occupancy analysis reported
            # it. Nothing is clamped or defaulted into this field, so it can
            # never disagree with the number the ratio was derived from.
            "functional_EBN": dict(raw_ebn),
            "functional_EBN_is_raw": True,
        },
    }


# ── Run manifest / resume guard ───────────────────────────────

_MANIFEST_NAME = "phase4_inputs.json"

# Fields that define "the same simulation". A change in ANY of these means the
# existing trajectory was produced with different inputs and must not be reused.
_FINGERPRINT_FIELDS = (
    "target", "pc_id", "replica_index", "placement_seed",
    "epitope_pdb_sha256", "monomer_copies", "monomer_smiles",
    "total_monomers", "time_ns", "protein_restrained", "template_mode",
    "polymerization_type", "crosslinker_mole_fraction_actual",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(manifest: dict) -> str:
    payload = {k: manifest.get(k) for k in _FINGERPRINT_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _check_resume(md_dir: Path, manifest: dict) -> bool:
    """Return True iff an existing run in `md_dir` may be resumed.

    # BEHAVIOUR CHANGE (2026-08 audit).
    Old code:  `if (md_dir / "md.gro").exists(): skip parameterisation`.
               A directory built with retired SMILES / wrong copy numbers was
               silently accepted and its trajectory analysed as if current.
    New code:  resumption requires a matching input fingerprint. A mismatch (or
               a missing manifest next to existing simulation output) raises,
               and the failure is recorded per (target, pc_id, replica).
               PHASE4_ALLOW_STALE_RESUME=1 downgrades it to a warning.
    """
    md_dir = Path(md_dir)
    produced = [n for n in ("md.gro", "md.xtc", "md.tpr", "complex.gro")
                if (md_dir / n).exists()]
    if not produced:
        return False

    manifest_path = md_dir / _MANIFEST_NAME
    prev = None
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text())
        except Exception as e:
            logger.error(f"    {manifest_path} is unreadable ({e})")

    if prev is not None and prev.get("fingerprint") == manifest["fingerprint"]:
        if (md_dir / "md.gro").exists():
            logger.info("    Resume: inputs match recorded manifest → "
                        "reusing existing trajectory")
            return True
        logger.info("    Resume: inputs match recorded manifest → "
                    "continuing an incomplete run")
        return True

    if prev is None:
        why = (f"{md_dir} already contains {produced} but has no "
               f"{_MANIFEST_NAME}, so the inputs that produced it are unknown")
        diff = {}
    else:
        diff = {k: {"recorded": prev.get(k), "now": manifest.get(k)}
                for k in _FINGERPRINT_FIELDS
                if prev.get(k) != manifest.get(k)}
        why = (f"{md_dir} contains {produced} produced with DIFFERENT inputs; "
               f"changed fields: {json.dumps(diff, default=str)}")

    msg = ("STALE PHASE 4 DIRECTORY — refusing to reuse it. " + why +
           ". This is exactly the failure the md.gro skip guard used to hide "
           "(corrected force fields never regenerating). Delete the directory "
           "to re-run, or set PHASE4_ALLOW_STALE_RESUME=1 to reuse it anyway.")

    if _allow_stale_resume():
        logger.error("    PHASE4_ALLOW_STALE_RESUME is set — " + msg)
        return True
    raise RuntimeError(msg)


# ── Pre-polymerization MD with Ratio Analysis ─────────────────

def _run_prepolymerization_md(target: str, pc_id: str,
                                functional_monomers: list,
                                crosslinker: str,
                                epitope_pdb: Path,
                                work_dir: Path,
                                time_ns: float = 50.0,
                                total_monomers: int = 20,
                                replica_index: int = 0,
                                n_replicas: int = 1,
                                copies: dict = None,
                                seed_label: str = None) -> dict:
    """
    Run ONE pre-polymerization MD replica and analyze monomer-epitope interactions.

    Step A: Build system — monomers randomly placed from a per-replica seed
    Step B: Run MD — monomers find binding sites by free diffusion
    Step C: Contact / EBN / HBNMax analysis → per-replica acceptance

    `copies`  EXPLICIT per-monomer copy numbers, e.g. {"TEOS": 48, "APTES": 12}.
              When given they are IMPOSED on the box and `total_monomers` is
              ignored; zero is honoured and drops that species entirely.  When
              omitted the composition is derived from SOLGEL_Q_MOLE_FRACTION /
              RADICAL_CROSSLINKER_MOLE_FRACTION exactly as before — which is
              the same box at every ratio, and is why the ratio sweep was
              declaration-only.  The composition that is actually BUILT is
              verified against this request in `topol.top` before any number is
              read off the trajectory.
    `seed_label` disambiguates the placement seed between grid points that share
              a (target, pc_id).  Without it two grid points would place their
              monomers with the identical RNG stream.
    """
    from .utils_gromacs import (run_full_md_pipeline,
                                  run_full_md_pipeline_membrane,
                                  parameterize_monomer)
    try:
        from .config import PHASE4_MEMBRANE_MODE as _P4_MEMBRANE_MODE
    except Exception:
        _P4_MEMBRANE_MODE = False
    from .utils_structure import smiles_to_mol2
    from .config import ALL_MONOMERS

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "target": target,
        "pc_id": pc_id,
        "replica_index": replica_index,
        "n_replicas": n_replicas,
        "functional_monomers": functional_monomers,
        "crosslinker": crosslinker,
        "time_ns": time_ns,
        "work_dir": str(work_dir),
    }

    # A box with no functional monomer is only nonsense when NOBODY asked for
    # it. With explicit copies it is the grid's crosslinker-only control point,
    # which is one of the two points that say whether the observable is
    # imprinting or amine count. Refuse the accidental case, allow the
    # deliberate one.
    if len(functional_monomers) == 0 and not copies:
        result.update({"error": "No functional monomers", "success": False,
                       "accepted": False})
        return result

    try:
        # Step A: Determine copy numbers.
        # explicit `copies` → imposed verbatim (the ratio grid);
        # otherwise      → derived from the library crosslinker mole fraction.
        if copies:
            comp = _composition_from_copies(functional_monomers, crosslinker,
                                            copies)
        else:
            comp = _monomer_copy_numbers(functional_monomers, crosslinker,
                                         total_monomers)
        monomer_copies = comp["copies"]
        # The species that are actually in this box. EVERYTHING downstream —
        # the occupancy analysis, the EBN→ratio derivation, the RDF — is keyed
        # on these, not on the nominal argument lists, so a dropped species is
        # never scored as "measured zero contacts".
        present_functional = comp.get("present_functional", functional_monomers)
        present_crosslinker = comp.get("present_crosslinker", crosslinker)
        result["initial_copies"] = monomer_copies
        result["requested_copies"] = comp.get("requested_copies", monomer_copies)
        result["present_functional"] = present_functional
        result["present_crosslinker"] = present_crosslinker
        result["box_composition"] = {k: v for k, v in comp.items() if k != "copies"}

        logger.info(
            f"  System: epitope + {comp['total_monomers']} monomers "
            f"({', '.join(f'{m}x{n}' for m, n in monomer_copies.items())}) — "
            f"{comp['polymerization_type']}, crosslinker "
            f"{comp['crosslinker_mole_fraction_actual']*100:.0f} mol% "
            f"(target {comp['crosslinker_mole_fraction_target']*100:.0f} mol%, "
            f"source {comp.get('composition_source')})")
        if comp.get("dropped_monomers"):
            logger.info(f"  Requested at ZERO copies, dropped from the box: "
                        f"{comp['dropped_monomers']}")

        from .config import PHASE4_TEMPLATE_MODE
        protein_restrained = (PHASE4_TEMPLATE_MODE == "ecl2")

        # seed_label keeps two grid points that share a (target, pc_id) from
        # drawing the identical monomer placement + velocity stream.
        seed = _replica_seed(target, seed_label or pc_id, replica_index)
        result["placement_seed"] = seed
        result["seed_label"] = seed_label or pc_id

        md_dir = work_dir / "md"
        md_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema": 2,
            "target": target,
            "pc_id": pc_id,
            "replica_index": replica_index,
            "n_replicas": n_replicas,
            "placement_seed": seed,
            "epitope_pdb": str(epitope_pdb),
            "epitope_pdb_sha256": (_sha256_file(Path(epitope_pdb))
                                   if Path(epitope_pdb).exists() else None),
            "monomer_copies": monomer_copies,
            "requested_copies": comp.get("requested_copies", monomer_copies),
            "composition_source": comp.get("composition_source"),
            "seed_label": seed_label or pc_id,
            "monomer_smiles": {m: (ALL_MONOMERS.get(m) or {}).get("smiles")
                               for m in monomer_copies},
            "total_monomers": comp["total_monomers"],
            "time_ns": time_ns,
            "protein_restrained": protein_restrained,
            "template_mode": PHASE4_TEMPLATE_MODE,
            "polymerization_type": comp["polymerization_type"],
            "crosslinker_mole_fraction_actual":
                comp["crosslinker_mole_fraction_actual"],
        }
        manifest["fingerprint"] = _fingerprint(manifest)
        result["inputs_fingerprint"] = manifest["fingerprint"]

        # Resume guard — raises on a stale directory instead of silently reusing
        resumable = _check_resume(md_dir, manifest)

        if resumable and (md_dir / "md.gro").exists():
            logger.info("  MD already completed with matching inputs, "
                        "skipping to analysis")
            monomer_itps = []  # topology already built; not needed for analysis
        else:
            # Step A: Parameterize monomers (random placement, no docked poses)
            logger.info(f"  Parameterizing monomers...")
            monomer_itps = []
            missing = []
            for m_name, n_copies in monomer_copies.items():
                m_info = ALL_MONOMERS.get(m_name)
                if m_info is None:
                    logger.error(f"  {m_name} not in library — cannot build box")
                    missing.append(m_name)
                    continue

                param_dir = work_dir / "monomer_params"
                mol2 = smiles_to_mol2(m_info["smiles"], m_name, param_dir)
                param = parameterize_monomer(mol2, m_name, param_dir)

                if param.get("itp"):
                    for copy_i in range(n_copies):
                        monomer_itps.append(param)
                else:
                    logger.error(f"  GAFF2 failed for {m_name}")
                    missing.append(m_name)

            if missing:
                # Was: a warning, then MD ran with a box silently missing that
                # monomer type — and its EBN/ratio came out as 0 with no trace.
                raise RuntimeError(
                    f"could not parameterise {missing}; refusing to run MD on a "
                    f"box that silently omits a monomer type")

            if len(monomer_itps) != comp["total_monomers"]:
                raise RuntimeError(
                    f"built {len(monomer_itps)} monomer molecules but the "
                    f"composition asks for {comp['total_monomers']}")

            logger.info(f"  Total monomer molecules: {len(monomer_itps)}")

            # C2 · Force-field consistency pre-flight. When PHASE4_MEMBRANE_MODE
            # is on, monomer itps will be MERGED with CHARMM-GUI's CHARMM36
            # base. If they still carry a GAFF [ defaults ] block or GAFF atom
            # types, GROMACS silently uses the base's fudge factors and mixes
            # two force fields in one box — internal energies and monomer-
            # protein contacts come out wrong with no error. Refuse to launch.
            if _P4_MEMBRANE_MODE:
                try:
                    from .config import PHASE4_FF_STRATEGY as _ff_strategy
                except Exception:
                    _ff_strategy = "cgenff_uniform"
                from .utils_gromacs import _preflight_ff_consistency_check
                _preflight = _preflight_ff_consistency_check(
                    monomer_itps, _ff_strategy)
                if not _preflight["ok"]:
                    raise RuntimeError(
                        f"Phase 4 pre-flight FAILED (C2 FF consistency, "
                        f"strategy={_ff_strategy!r}, checked "
                        f"{_preflight['checked']} itp(s)): {_preflight['issues']}. "
                        f"Recommendation: {_preflight['recommendation']}")
                logger.info(
                    "  C2 pre-flight PASS (strategy=%s, %d monomer itps clean)",
                    _preflight["strategy"], _preflight["checked"])

        # Record the manifest BEFORE the MD so a crashed run is still traceable
        (md_dir / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2,
                                                        default=str))

        # Step B: Run MD with protein backbone restrained (whole-ECL2 imprinting).
        # Mimics solid-phase / surface immobilization during polymerization —
        # protein cannot diffuse, monomers polymerize around its exposed face.
        # Reference: adenovirus eIP (PMC11059108), Pluhar/Battaglia 2021.
        #
        # SEEDING: run_full_md_pipeline(seed=...) now forwards this seed to BOTH
        # stochastic steps — _include_monomers_in_topology(seed=) for monomer
        # placement and NVT gen_seed for Maxwell velocities — so a replica is
        # fully reproducible (integration pass; the two utils_gromacs cross-file
        # requests landed).  The global-RNG seeding below is kept as a belt-and-
        # braces fallback for the pre-fix signature and for any other module-level
        # random use inside the call; the previous state is restored afterwards so
        # nothing else in the process inherits the seeded stream.
        # QUICK IS NEVER INFERRED FROM time_ns.  # BEHAVIOUR CHANGE
        #
        # WAS: quick=(time_ns <= 20).  run_full_md_pipeline then does
        # `if quick: time_ns = MD_QUICK_NS`, so ANY requested length at or below
        # 20 ns was silently replaced by MD_QUICK_NS (5 ns under the BSA config)
        # while the REQUESTED value was what the grid logged as its GPU budget
        # and stamped into ratio_grid_summary.json and every point_result.json.
        # A 1-replica 10 ns screening pass ran 5 ns and was recorded as 10.
        # The caller already applied `quick` when it chose time_ns
        # (run_phase4: time_ns = MD_QUICK_NS if quick else MD_PRODUCTION_NS), so
        # applying it again here was a double application as well as a silent one.
        md_kwargs = dict(time_ns=time_ns, quick=False,
                         protein_restrained=protein_restrained)
        result["requested_time_ns"] = float(time_ns)
        # A1: pick the aqueous vs CHARMM-GUI-membrane pipeline. Membrane mode
        # ignores `epitope_pdb` (that path uses pdb2gmx on a naked peptide and
        # would silently rebuild the aqueous system) and instead consumes
        # structures/membrane/<target>/step5_input.gro directly, placing the
        # monomer swarm only in the aqueous slab above the upper leaflet.
        _pipeline_fn = (run_full_md_pipeline_membrane if _P4_MEMBRANE_MODE
                        else run_full_md_pipeline)
        sig = inspect.signature(_pipeline_fn).parameters
        if "seed" in sig:
            md_kwargs["seed"] = seed
            result["thermostat_seed_reproducible"] = True
        else:
            # gen_seed = -1 in MDP_NVT → GROMACS picks its own velocity seed.
            # Replicas are still INDEPENDENT (distinct placements + distinct
            # velocities); they are just not bit-reproducible.
            result["thermostat_seed_reproducible"] = False
            logger.error(
                "  %s() does not accept seed= yet, so MDP_NVT keeps "
                "gen_seed=-1: replica %d is independent but NOT reproducible. "
                "See cross-file request for utils_gromacs.py.",
                _pipeline_fn.__name__, replica_index)

        rng_state = random.getstate()
        random.seed(seed)
        try:
            if _P4_MEMBRANE_MODE:
                # Membrane pipeline takes (target, monomer_itps, work_dir, …);
                # `epitope_pdb` and `protein_restrained` do not apply — the
                # protein is the CHARMM-GUI PROA + covalently attached glycans
                # and the bilayer has been equilibrated by CHARMM-GUI already.
                md_kwargs.pop("protein_restrained", None)
                logger.info("  PHASE4_MEMBRANE_MODE=True → using "
                            "run_full_md_pipeline_membrane(%s)", target)
                md_result = run_full_md_pipeline_membrane(
                    target, monomer_itps, md_dir, **md_kwargs)
            else:
                md_result = run_full_md_pipeline(epitope_pdb, monomer_itps,
                                                  md_dir, **md_kwargs)
        finally:
            random.setstate(rng_state)
        result.update(md_result)
        result["md_completed"] = bool((md_dir / "md.gro").exists())

        # THE LENGTH THAT RAN, not the length that was asked for. Recorded
        # separately and asserted, so no future `quick`-style override can
        # substitute a different leg length behind the recorded one.
        _eff = md_result.get("time_ns", time_ns)
        result["effective_time_ns"] = (float(_eff)
                                       if isinstance(_eff, (int, float)) else None)
        result["time_ns"] = float(time_ns)
        if (result["effective_time_ns"] is not None
                and abs(result["effective_time_ns"] - float(time_ns)) > 1e-6):
            raise RuntimeError(
                f"run_full_md_pipeline ran {result['effective_time_ns']:g} ns but "
                f"this leg asked for {float(time_ns):g} ns. Something between the "
                f"request and mdrun substituted a different length (historically "
                f"`quick` -> MD_QUICK_NS={_cfg_get('MD_QUICK_NS')}). Refusing to "
                f"record a leg under a length it did not run.")

        # ── COMPOSITION VERIFICATION — the check that makes `copies=` falsifiable.
        # Runs on BOTH paths (fresh build and resumed directory) and RAISES on a
        # mismatch, so a grid point can never report numbers for a box other
        # than the one it asked for.
        result["composition_verification"] = _verify_built_composition(
            md_dir, monomer_copies)
        result["built_copies"] = result["composition_verification"]["built_copies"]

        # Step C: Contact frequency analysis → optimal ratio
        xtc = md_dir / "md.xtc"
        gro = md_dir / "npt.gro"
        if xtc.exists() and gro.exists():
            logger.info(f"  Analyzing contact frequency...")
            ratio_result = _analyze_monomer_occupancy(
                xtc, gro, present_functional, present_crosslinker,
                target=target,
            )
            result["occupancy_analysis"] = ratio_result

            # An explicitly-composed box publishes ITS OWN crosslinker mole
            # fraction, not the library constant — otherwise every grid point
            # emits the same recipe regardless of what was simulated.
            _x_built = (comp["crosslinker_mole_fraction_actual"]
                        if comp.get("composition_source") == "explicit_copies"
                        else None)
            # The synthesis ratio is reported per BOTTLE, so the pH speciation
            # of a monomer is folded back onto its parent first.
            _parent_of = comp.get("protonation_parent_of") or {}
            _recipe_ebn = _merge_protonation_forms(
                ratio_result.get("EBN", {}), _parent_of)
            _recipe_functional = sorted(
                {_parent_of.get(m, m) for m in present_functional})
            try:
                ratio_pack = _derive_optimal_ratio(
                    _recipe_ebn, _recipe_functional,
                    present_crosslinker, comp["polymerization_type"],
                    _crosslinker_mole_fraction=_x_built)
            except ValueError as e:
                # A control box (no functional monomer, or no crosslinker) has
                # no synthesis ratio by construction. Record why; do NOT fail
                # the leg — its observable is the point of running it.
                logger.info(f"  No synthesis ratio for this box: {e}")
                ratio_pack = {
                    "optimal_ratio": None,
                    "optimal_ratio_basis": {
                        "error": str(e),
                        "functional_EBN": dict(ratio_result.get("EBN", {})),
                        "functional_EBN_is_raw": True,
                        "is_control_box": not present_functional
                                          or present_crosslinker is None,
                    },
                }
            ratio_result.update(ratio_pack)
            result["optimal_ratio"] = ratio_pack["optimal_ratio"]
            result["optimal_ratio_basis"] = ratio_pack["optimal_ratio_basis"]

            if ratio_pack["optimal_ratio"] is None:
                logger.info("  Optimal synthesis ratio: NONE (control box)")
            else:
                logger.info(f"  Optimal synthesis ratio:")
                for m, ratio in ratio_pack["optimal_ratio"].items():
                    occ = ratio_result.get("contact_freq_6A", {}).get(m, 0)
                    logger.info(f"    {m}: {ratio} parts (contact freq={occ:.2f})")

            # Per-atom RDF analysis (Yuan et al. 2024)
            logger.info(f"  Per-atom RDF analysis...")
            rdf_result = _compute_per_atom_rdf(
                xtc, gro, present_functional, present_crosslinker)
            result["per_atom_rdf"] = rdf_result

            # Per-residue MM-GBSA decomposition (Kumar et al. 2024)
            if (md_dir / "md.tpr").exists():
                logger.info(f"  Per-residue MM-GBSA decomposition...")
                result["mmpbsa_decomp"] = _run_mmgbsa_window(md_dir, decomp=True)

            # Audit the MAIN MM-GBSA (run inside run_full_md_pipeline)
            result["mmpbsa_window_audit"] = _audit_mmpbsa_window(
                md_dir, result.get("mmpbsa", {}))

        # ── Per-replica acceptance ──
        acc = _replica_acceptance(result, md_dir)
        result["acceptance"] = acc
        result["accepted"] = acc["accepted"]
        # `success` now means "this replica produced a usable, accepted result".
        # `md_completed` is the old technical meaning.  # BEHAVIOUR CHANGE
        result["success"] = bool(result.get("md_completed")) and acc["accepted"]
        if not acc["accepted"]:
            logger.error(
                "  REPLICA %d REJECTED for %s/%s — failed: %s. It will NOT be "
                "pooled.", replica_index, target, pc_id,
                ", ".join(acc["failed_criteria"]) or "unknown")

    except Exception as e:
        logger.error(f"  MD failed: {e}")
        result["success"] = False
        result["accepted"] = False
        result["error"] = str(e)
        result.setdefault("acceptance", {
            "accepted": False, "criteria": {},
            "failed_criteria": ["exception"], "notes": [str(e)]})

    return result


# ── MM-GBSA frame window ──────────────────────────────────────

def _trajectory_time_axis(md_dir: Path) -> tuple:
    """(n_frames, dt_ps, t0_ps) of md.xtc, read from the trajectory itself."""
    import MDAnalysis as mda
    md_dir = Path(md_dir)
    xtc = md_dir / "md.xtc"
    top = md_dir / "md.tpr"
    if not top.exists():
        top = md_dir / "npt.gro"
    if not (xtc.exists() and top.exists()):
        raise FileNotFoundError(f"need {xtc} and a topology in {md_dir}")
    u = mda.Universe(str(top), str(xtc))
    n = len(u.trajectory)
    if n == 0:
        raise ValueError(f"{xtc} has no frames")
    t0 = float(u.trajectory[0].time)
    if n >= 2:
        dt = float(u.trajectory[1].time) - t0
    else:
        dt = float(getattr(u.trajectory, "dt", 0.0) or 0.0)
    return n, dt, t0


def _mmpbsa_frame_window(md_dir: Path, start_ns: float, end_ns: float,
                         n_frames_target: int = 100) -> dict:
    """Convert an MM-GBSA ns window into 1-indexed gmx_MMPBSA frame indices.

    # BEHAVIOUR CHANGE (2026-08 audit) — this is the BLOCKER fix.
    utils_gromacs.run_mmpbsa() accepted start_ns/end_ns and then wrote
    `startframe=1, endframe=<n_frames>` regardless, so MM-GBSA was computed on
    the FIRST 1 ns of a 350 ns production run. This converts the requested
    window with the ACTUAL trajectory dt and ASSERTS it lies inside the
    trajectory; an out-of-range window raises instead of quietly sliding to
    frame 1.
    """
    n, dt_ps, t0_ps = _trajectory_time_axis(md_dir)
    if dt_ps <= 0:
        raise ValueError(
            f"cannot determine frame spacing for {md_dir}/md.xtc "
            f"(n_frames={n}, dt={dt_ps} ps) — refusing to guess an MM-GBSA window")

    traj_start_ns = t0_ps / 1000.0
    traj_end_ns = (t0_ps + (n - 1) * dt_ps) / 1000.0
    tol = dt_ps / 2000.0  # half a frame, in ns

    if not (end_ns > start_ns):
        raise ValueError(f"MM-GBSA window end_ns={end_ns} must exceed "
                         f"start_ns={start_ns}")
    # A TRAJECTORY THAT STOPS ONE FRAME SHORT IS NOT A CONFIG ERROR.
    # mdrun's last written frame need not land exactly on time_ns: with
    # nstxout-compressed=5000 and dt=2 fs the final frame can be up to one
    # output interval early (measured: a 0.1 ns leg ended at 0.094 ns, and the
    # correctly-configured 0.06-0.1 ns window then failed by 6 ps and lost
    # MM-GBSA on the leg). Clamp a shortfall of up to two output intervals to
    # the last frame and SAY SO; anything larger is still a real mismatch.
    clamp_tol_ns = 2.0 * dt_ps / 1000.0
    clamped_end_ns = None
    if traj_end_ns < end_ns <= traj_end_ns + clamp_tol_ns:
        clamped_end_ns = end_ns
        logger.info(
            "    MM-GBSA end %.4f ns is %.1f ps past the last written frame "
            "(%.4f ns, dt=%.1f ps) — clamping to the last frame rather than "
            "discarding the window.",
            end_ns, (end_ns - traj_end_ns) * 1000.0, traj_end_ns, dt_ps)
        end_ns = traj_end_ns

    if start_ns < traj_start_ns - tol or end_ns > traj_end_ns + tol:
        raise ValueError(
            f"MM-GBSA window {start_ns}-{end_ns} ns lies outside the trajectory "
            f"{traj_start_ns:.3f}-{traj_end_ns:.3f} ns "
            f"({n} frames, dt={dt_ps:.1f} ps) in {md_dir}. Fix MD_MMPBSA_START_NS "
            f"/ MD_MMPBSA_END_NS or MD_PRODUCTION_NS.")

    # IN-RANGE IS NOT THE SAME AS EQUILIBRATED (REVIEW FINDING 7).
    #
    # The range check above only asks whether the window exists in the
    # trajectory. It happily passed MD_MMPBSA_START_NS=30 / END=50 against a
    # 350 ns run — the first 8.6%-14.3%, i.e. the equilibration transient —
    # and _audit_mmpbsa_window then stamped that ΔG window_valid=True. A stale
    # config pair (these were written for a 50 ns production run and never
    # re-derived when it became 350) must fail LOUDLY, not acquire a validity
    # stamp.
    #
    # A binding free energy is only meaningful on the equilibrated portion, so
    # the window must begin no earlier than the trajectory midpoint.
    traj_span_ns = traj_end_ns - traj_start_ns
    midpoint_ns = traj_start_ns + 0.5 * traj_span_ns
    if traj_span_ns > 0 and start_ns < midpoint_ns - tol:
        raise ValueError(
            f"MM-GBSA window {start_ns}-{end_ns} ns starts in the FIRST HALF of "
            f"the {traj_start_ns:.3f}-{traj_end_ns:.3f} ns trajectory "
            f"(covers {100.0 * (start_ns - traj_start_ns) / traj_span_ns:.1f}%"
            f"-{100.0 * (end_ns - traj_start_ns) / traj_span_ns:.1f}% of it). "
            f"That is the equilibration transient, not the equilibrated tail, "
            f"so the binding free energy computed there is not a converged "
            f"number. MD_MMPBSA_START_NS must be >= {midpoint_ns:.1f} ns for "
            f"this run. Re-derive MD_MMPBSA_START_NS / MD_MMPBSA_END_NS from "
            f"MD_PRODUCTION_NS — they are NOT automatically updated when the "
            f"production length changes.")

    startframe = int(round((start_ns * 1000.0 - t0_ps) / dt_ps)) + 1
    endframe = int(round((end_ns * 1000.0 - t0_ps) / dt_ps)) + 1
    startframe = min(max(1, startframe), n)
    endframe = min(max(1, endframe), n)
    if endframe <= startframe:
        raise ValueError(
            f"MM-GBSA window {start_ns}-{end_ns} ns collapses to frames "
            f"{startframe}..{endframe} of {n} (dt={dt_ps:.1f} ps): too narrow")

    span = endframe - startframe + 1
    interval = max(1, span // max(1, int(n_frames_target)))
    return {
        "startframe": startframe,
        "endframe": endframe,
        "interval": interval,
        "n_frames_used": len(range(startframe, endframe + 1, interval)),
        "traj_n_frames": n,
        "traj_dt_ps": dt_ps,
        "traj_start_ns": round(traj_start_ns, 4),
        "traj_end_ns": round(traj_end_ns, 4),
        "requested_start_ns": start_ns,
        "requested_end_ns": clamped_end_ns if clamped_end_ns is not None else end_ns,
        "end_clamped_to_last_frame": clamped_end_ns is not None,
    }


def _run_mmgbsa_window(md_dir: Path, decomp: bool = False) -> dict:
    """Call utils_gromacs.run_mmpbsa on the CORRECT frame window.

    run_mmpbsa() must accept startframe/endframe/interval for this to be
    correct. Until it does (cross-file request), this refuses to run rather
    than producing a first-1-ns number that looks like a converged ΔG.
    """
    from .utils_gromacs import run_mmpbsa
    from .config import MD_MMPBSA_START_NS, MD_MMPBSA_END_NS, MD_MMPBSA_INTERVAL

    try:
        win = _mmpbsa_frame_window(md_dir, MD_MMPBSA_START_NS, MD_MMPBSA_END_NS,
                                   MD_MMPBSA_INTERVAL)
    except Exception as e:
        logger.error(f"  MM-GBSA window invalid: {e}")
        return {"error": f"invalid MM-GBSA window: {e}", "window_valid": False}

    logger.info(f"    MM-GBSA window {MD_MMPBSA_START_NS}-{MD_MMPBSA_END_NS} ns "
                f"→ frames {win['startframe']}..{win['endframe']} "
                f"step {win['interval']} (dt={win['traj_dt_ps']:.1f} ps, "
                f"{win['traj_n_frames']} frames total)")

    sig = inspect.signature(run_mmpbsa).parameters
    if "startframe" not in sig:
        msg = ("utils_gromacs.run_mmpbsa() does not accept startframe/endframe "
               "yet; it hardcodes startframe=1, endframe=n_frames, i.e. the "
               "FIRST ~1 ns of the run. Refusing to compute a number that would "
               "be silently wrong. See the cross-file request for "
               "utils_gromacs.py:run_mmpbsa.")
        logger.error("  " + msg)
        return {"error": msg, "window_valid": False, "window": win}

    try:
        out = run_mmpbsa(md_dir,
                         start_ns=MD_MMPBSA_START_NS,
                         end_ns=MD_MMPBSA_END_NS,
                         n_frames=MD_MMPBSA_INTERVAL,
                         startframe=win["startframe"],
                         endframe=win["endframe"],
                         interval=win["interval"],
                         decomp=decomp)
    except Exception as e:
        logger.error(f"  MM-GBSA failed: {e}")
        return {"error": str(e), "window_valid": False, "window": win}

    if isinstance(out, dict):
        out["window"] = win
        out["window_valid"] = True
    return out


def _audit_mmpbsa_window(md_dir: Path, mmpbsa: dict) -> dict:
    """Check the MM-GBSA that run_full_md_pipeline computed used the right frames.

    run_full_md_pipeline calls run_mmpbsa itself; until that call is fixed, its
    ΔG is the first ~1 ns. Recording the audit here makes the wrongness visible
    in phase4_md_results.json instead of invisible.
    """
    from .config import MD_MMPBSA_START_NS, MD_MMPBSA_END_NS, MD_MMPBSA_INTERVAL
    audit = {"requested_start_ns": MD_MMPBSA_START_NS,
             "requested_end_ns": MD_MMPBSA_END_NS}
    try:
        win = _mmpbsa_frame_window(md_dir, MD_MMPBSA_START_NS, MD_MMPBSA_END_NS,
                                   MD_MMPBSA_INTERVAL)
        audit["expected_window"] = win
    except Exception as e:
        audit["error"] = str(e)
        audit["window_valid"] = False
        logger.error(f"  MM-GBSA window audit failed: {e}")
        return audit

    reported = (mmpbsa or {}).get("window") or {}
    if reported.get("startframe") == win["startframe"] and \
       reported.get("endframe") == win["endframe"]:
        audit["window_valid"] = True
    else:
        audit["window_valid"] = False
        audit["reported_window"] = reported or None
        logger.error(
            "  MM-GBSA in run_full_md_pipeline did NOT record the expected "
            "frame window (expected %d..%d, recorded %s). Its ΔG is the "
            "first-frames value, not the %s-%s ns window. See cross-file "
            "request for utils_gromacs.py.",
            win["startframe"], win["endframe"], reported or "nothing",
            MD_MMPBSA_START_NS, MD_MMPBSA_END_NS)
    return audit


# ── Per-replica acceptance ────────────────────────────────────

def _rmsd_drift_nm(md_dir: Path):
    """Backbone-RMSD drift between the 3rd and 4th quarter of the run, in nm."""
    try:
        from .utils_gromacs import _parse_xvg
    except Exception:
        return None
    data = _parse_xvg(Path(md_dir) / "rmsd.xvg")
    if data is None or len(data) < 8:
        return None
    n = len(data)
    q3 = data[(n * 2) // 4:(n * 3) // 4, 1]
    q4 = data[(n * 3) // 4:, 1]
    if len(q3) == 0 or len(q4) == 0:
        return None
    return float(abs(np.mean(q4) - np.mean(q3)))


def _replica_acceptance(result: dict, md_dir: Path) -> dict:
    """Decide whether ONE replica may be pooled.

    # BEHAVIOUR CHANGE (2026-08 audit). The block-averaged convergence number
    used to be computed and then IGNORED — every leg reported success even with
    window differences of 73%. Convergence and RMSD equilibration are now
    BLOCKING criteria; a rejected replica is excluded from the pooled result.
    """
    occ = result.get("occupancy_analysis", {}) or {}
    crit = {}
    advisory = {}          # computed and reported, but never rejects the leg
    notes = []

    crit["md_completed"] = bool(result.get("md_completed"))
    crit["analysis_ok"] = bool(occ) and "error" not in occ
    if not crit["analysis_ok"] and occ.get("error"):
        notes.append(f"occupancy analysis error: {occ['error']}")

    # A partially-failed H-bond analysis fails the replica (REVIEW FINDING 19).
    # HBNMax is None (not 0) for a monomer whose HBA raised; pooling a replica
    # with holes in it would average over a metric that is missing for some
    # monomers and present for others.
    _hb_failed = occ.get("hbond_failed_monomers") or []
    crit["hbond_analysis_complete"] = not _hb_failed
    if _hb_failed:
        notes.append(f"H-bond analysis failed for {_hb_failed} — their HBNMax "
                     f"is None (not 0), so this replica's H-bond metrics are "
                     f"incomplete")

    # CONVERGENCE: BLOCKING OR ADVISORY (see PHASE4_CONVERGENCE_GATE).
    # Under "advisory" the number is computed, reported and visible in every
    # summary, but it does not reject the leg — uncertainty is carried by the
    # replicate CI instead. A leg is still rejected for the things that are
    # actual FAILURES (md_completed, analysis_ok, hbond_analysis_complete,
    # rmsd_equilibrated); only the sub-window drift heuristic is demoted.
    conv = occ.get("convergence") or {}
    _gate_mode = str(_cfg("PHASE4_CONVERGENCE_GATE")).strip().lower()
    if conv.get("converged") is None:
        _conv_ok = False
        _conv_note = ("convergence undetermined: "
                      + str(conv.get("reason", "no convergence block")))
    else:
        _conv_ok = bool(conv["converged"])
        _conv_note = (None if _conv_ok else
                      f"block-averaged contact frequency not converged "
                      f"(tol={conv.get('tolerance_pct')}%, "
                      f"diffs={conv.get('window_diff_pct')})")
    # The key stays in `criteria` in BOTH modes so the record shape is stable
    # for anything that reads it; "advisory" only removes it from the set that
    # can REJECT the replica. Moving the key instead was a mistake — it turned a
    # configuration choice into a schema change and broke consumers that index
    # criteria["converged"].
    crit["converged"] = _conv_ok
    # ADVISORY DEMOTES DRIFT, NOT ABSENCE OF DATA. `converged` carries two very
    # different failures: False means the species HAD contacts and the block
    # means disagreed (the noisy heuristic this mode is meant to stop rejecting
    # on), while None means the analysis could not determine it at all — a DEAD
    # BOX where nothing touched the protein, or a missing convergence block.
    # A dead box is a real failure with no data behind it and must still reject
    # in either mode; demoting it would let an empty leg into the pooled result.
    _conv_is_drift = conv.get("converged") is False
    if _gate_mode == "advisory" and _conv_is_drift:
        advisory["converged"] = _conv_ok
        if _conv_note:
            notes.append(
                _conv_note + " — ADVISORY ONLY (PHASE4_CONVERGENCE_GATE="
                "'advisory'): this leg is still pooled, and the uncertainty it "
                "carries is reported as the replicate CI, not as a pass/fail.")
    elif _conv_note:
        notes.append(_conv_note)
    # A present species that recorded zero contacts is DATA (see the
    # convergence block). Recorded as a note, never as a failed criterion.
    _zero = conv.get("monomers_without_contacts") or []
    if _zero and crit.get("converged"):
        notes.append(
            f"{_zero} recorded ZERO contacts in both comparison windows. That "
            f"is a measured zero at this composition, not a failed analysis — "
            f"this replica is accepted and the point ranks with that zero in it.")

    drift = _rmsd_drift_nm(md_dir)
    tol_nm = float(_cfg("PHASE4_RMSD_DRIFT_TOL_NM"))
    if drift is None:
        crit["rmsd_equilibrated"] = False
        notes.append("rmsd.xvg missing or too short — equilibration unverifiable")
    else:
        crit["rmsd_equilibrated"] = drift <= tol_nm
        if not crit["rmsd_equilibrated"]:
            notes.append(f"backbone RMSD still drifting: |Q4-Q3| = {drift:.3f} nm "
                         f"> {tol_nm} nm")
    result["rmsd_drift_q3q4_nm"] = None if drift is None else round(drift, 4)

    # NON-BLOCKING quality flags (their fixes live in other agents' files).
    warn = {}
    dssp = result.get("dssp") or {}
    # utils_analysis.analyze_dssp_changes currently divides by ALL residues
    # INCLUDING WATER, so structure_preserved cannot be False. Until it reports
    # `selection == "protein"`, its verdict is not trusted here.
    warn["dssp_protein_only"] = (dssp.get("selection") == "protein")
    if dssp and not warn["dssp_protein_only"]:
        notes.append("DSSP fractions include non-protein residues — "
                     "structure_preserved is not trustworthy (see cross-file "
                     "request for utils_analysis.py)")
    audit = result.get("mmpbsa_window_audit") or {}
    warn["mmpbsa_window_valid"] = bool(audit.get("window_valid"))
    if audit and not warn["mmpbsa_window_valid"]:
        notes.append("MM-GBSA in run_full_md_pipeline used the wrong frame window")
    warn["thermostat_seed_reproducible"] = bool(
        result.get("thermostat_seed_reproducible"))

    # Criteria demoted by configuration are reported but never reject.
    failed = [k for k, v in crit.items() if not v and k not in advisory]
    return {
        "accepted": not failed,
        "criteria": crit,
        # Criteria that were EVALUATED but deliberately not allowed to reject.
        # Kept separate from `warnings` so it is obvious which checks were
        # demoted by configuration rather than merely informational.
        "advisory_criteria": advisory,
        "convergence_gate_mode": _gate_mode,
        "failed_criteria": failed,
        "warnings": warn,
        "notes": notes,
        "rmsd_drift_q3q4_nm": None if drift is None else round(drift, 4),
        "rmsd_drift_tol_nm": tol_nm,
    }


# ── Replica pooling ───────────────────────────────────────────

def _pool_replicas(target: str, pc_id: str, functional_monomers: list,
                   crosslinker: str, replica_results: list, time_ns: float,
                   pc_dir: Path, output_dir: Path) -> dict:
    """Combine independent replicas into the one dict Phases 5/6 consume.

    Only ACCEPTED replicas contribute to the pooled numbers. If none is
    accepted the pooled entry still carries the data (so nothing is lost) but
    is marked success=False / accepted=False and lists why.
    """
    accepted = [r for r in replica_results if r.get("accepted")]
    pool_src = accepted if accepted else [r for r in replica_results
                                          if r.get("occupancy_analysis")]
    basis = ("accepted_replicas" if accepted else
             ("all_replicas_UNACCEPTED" if pool_src else "none"))

    pooled = {
        "target": target,
        "pc_id": pc_id,
        "functional_monomers": functional_monomers,
        "crosslinker": crosslinker,
        "time_ns": time_ns,
        "n_replicas": len(replica_results),
        "n_replicas_accepted": len(accepted),
        "accepted_replica_indices": [r.get("replica_index") for r in accepted],
        "pooling_basis": basis,
        "replica_md_dirs": [
            str(Path(r.get("work_dir", _replica_dir(pc_dir, i))) / "md")
            for i, r in enumerate(replica_results)],
        "replicas": [_compact_replica(r) for r in replica_results],
    }

    # Representative replica = first accepted (or first with a trajectory).
    rep = (accepted or pool_src or replica_results)[0]
    pooled["representative_replica"] = rep.get("replica_index")
    pooled["md_dir"] = str(Path(rep.get("work_dir",
                                        _replica_dir(pc_dir, 0))) / "md")
    for k in ("initial_copies", "requested_copies", "built_copies",
              "composition_verification", "present_functional",
              "present_crosslinker", "box_composition", "rmsd_mean_nm",
              "rmsd_last50ns_mean_nm", "rmsf_mean_nm", "rmsf_max_nm",
              "hbond_mean", "hbond_max", "rg_mean_nm", "dssp",
              "mmpbsa", "mmpbsa_decomp", "mmpbsa_window_audit",
              "per_atom_rdf"):
        if k in rep:
            pooled[k] = rep[k]

    # EVERY pooled replica must have been built from the same composition, or
    # the pooled mean is an average over different experiments. The per-replica
    # fingerprint already refuses a mismatched RESUME; this catches a mismatch
    # introduced within one run (e.g. a caller passing different copies per
    # replica) before the numbers are averaged together.
    _built = [tuple(sorted((r.get("built_copies") or
                            r.get("initial_copies") or {}).items()))
              for r in replica_results if r.get("md_completed")]
    if len(set(_built)) > 1:
        raise RuntimeError(
            f"replicas of {target}/{pc_id} were built from DIFFERENT "
            f"compositions {sorted(set(_built))} — pooling them would average "
            f"over different boxes. Refusing.")
    pooled["pooled_composition"] = dict(_built[0]) if _built else None

    # The species that were actually in the box, taken from the replica that is
    # being pooled — not from the nominal argument lists, which still name a
    # species that a control point requested at zero copies.
    functional_monomers = rep.get("present_functional", functional_monomers)
    crosslinker = rep.get("present_crosslinker", crosslinker)
    pooled["functional_monomers"] = list(functional_monomers)
    pooled["crosslinker"] = crosslinker

    all_mons = list(functional_monomers) + ([crosslinker] if crosslinker else [])
    occs = [r["occupancy_analysis"] for r in pool_src
            if isinstance(r.get("occupancy_analysis"), dict)
            and "error" not in r["occupancy_analysis"]]

    if occs:
        def _pool_mean(key):
            out = {}
            for m in all_mons:
                vals = [o.get(key, {}).get(m) for o in occs]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if vals:
                    out[m] = round(float(np.mean(vals)), 3)
            return out

        def _pool_std(key):
            out = {}
            for m in all_mons:
                vals = [o.get(key, {}).get(m) for o in occs]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if len(vals) > 1:
                    out[m] = round(float(np.std(vals, ddof=1)), 3)
                elif vals:
                    out[m] = 0.0
            return out

        ebn_pool = {}
        for m in all_mons:
            vals = [o.get("EBN", {}).get(m) for o in occs]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                ebn_pool[m] = int(max(vals))

        poly_type = (rep.get("box_composition", {})
                        .get("polymerization_type")
                     or _polymerization_type(all_mons))
        # _derive_optimal_ratio now RAISES on a zero or absent EBN rather than
        # clamping it to 1 part (REVIEW FINDING 13). Catch it here so the
        # failure is recorded against THIS leg instead of aborting Phase 4 for
        # every other target: no ratio is emitted, and the reason travels into
        # the result where phase6's bench-safety gate and verify_phase4 see it.
        _x_built = None
        _bc = rep.get("box_composition") or {}
        if _bc.get("composition_source") == "explicit_copies":
            _x_built = _bc.get("crosslinker_mole_fraction_actual")
        # Fold pH speciation back onto the parent monomer for the RECIPE only.
        _parent_of = (rep.get("box_composition") or {}).get(
            "protonation_parent_of") or {}
        _recipe_ebn = _merge_protonation_forms(ebn_pool, _parent_of)
        _recipe_functional = sorted(
            {_parent_of.get(m, m) for m in functional_monomers})
        try:
            ratio_pack = _derive_optimal_ratio(
                _recipe_ebn, _recipe_functional, crosslinker, poly_type,
                _crosslinker_mole_fraction=_x_built)
        except ValueError as e:
            logger.error(
                f"  NO SYNTHESIS RATIO for this leg: {e} "
                f"(pooled EBN = {ebn_pool})")
            ratio_pack = {
                "optimal_ratio": None,
                "optimal_ratio_basis": {
                    "error": str(e),
                    "functional_EBN": dict(ebn_pool),
                    "functional_EBN_is_raw": True,
                },
            }

        pooled_occ = {
            "n_replicas_pooled": len(occs),
            "pooling_basis": basis,
            "contact_freq_6A": _pool_mean("contact_freq_6A"),
            "contact_freq_6A_sd": _pool_std("contact_freq_6A"),
            "mean_min_dist_A": _pool_mean("mean_min_dist_A"),
            "proximity_fraction": _pool_mean("proximity_fraction"),
            "residence_ns": _pool_mean("residence_ns"),
            "residence_ps": _pool_mean("residence_ps"),
            "residence_steps": _pool_mean("residence_steps"),
            # legacy key that generate_report.py reads; still analysed-frame
            # counts, with residence_ns/_ps beside it as the real time
            "residence_frames": _pool_mean("residence_steps"),
            "HBNMax": _pool_mean("HBNMax"),
            "hbond_mean_per_frame": _pool_mean("hbond_mean_per_frame"),
            "hbond_lifetime_ps": _pool_mean("hbond_lifetime_ps"),
            "hbond_lifetime_avg": _pool_mean("hbond_mean_per_frame"),
            "crosslinker_proximity_A": _pool_mean("crosslinker_proximity_A"),
            "EBN": ebn_pool,
            # SAMPLING EVIDENCE. These are per-replica, not averaged: the number
            # of 6 A crossings is the SAMPLE SIZE behind an occupancy, so it
            # adds across replicas, and a block-average SE is a property of one
            # trajectory. Averaging either would destroy the thing they are for.
            # (They were computed and logged from the first run but dropped
            # here, because this dict is assembled from an explicit key list.)
            "n_binding_events_per_replica": [
                (o.get("n_binding_events") or {}) for o in occs],
            "n_binding_events_total": {
                m: sum((o.get("n_binding_events") or {}).get(m, 0) for o in occs)
                for m in all_mons},
            "block_average_se_per_replica": [
                (o.get("block_average_se") or {}) for o in occs],
            "pair_distances_md_A": {},
            "convergence": {
                "converged": all(bool((o.get("convergence") or {}).get("converged"))
                                 for o in occs),
                "per_replica": [(o.get("convergence") or {}).get("converged")
                                for o in occs],
                # worst (largest) block difference seen across pooled replicas
                # None values (e.g. when convergence analysis errored out) are
                # coerced to 0 rather than crashing the max().
                "window_diff_pct": {
                    m: max([(v if isinstance(v, (int, float)) else 0)
                            for o in occs
                            for v in [(o.get("convergence") or {})
                                       .get("window_diff_pct", {}).get(m, 0)]] or [0])
                    for m in all_mons},
                "tolerance_pct": float(_cfg("PHASE4_CONVERGENCE_TOL_PCT")),
            },
        }
        # Pair distances: average over pooled replicas
        pair_keys = set()
        for o in occs:
            pair_keys |= set((o.get("pair_distances_md_A") or {}).keys())
        for k in sorted(pair_keys):
            vals = [o.get("pair_distances_md_A", {}).get(k) for o in occs]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                pooled_occ["pair_distances_md_A"][k] = round(float(np.mean(vals)), 2)
        pooled_occ["total_contact_mean"] = round(
            float(sum(pooled_occ["contact_freq_6A"].values())), 3)
        pooled_occ["occupancy"] = pooled_occ["contact_freq_6A"]  # back-compat
        pooled_occ.update(ratio_pack)

        pooled["occupancy_analysis"] = pooled_occ
        pooled["optimal_ratio"] = ratio_pack["optimal_ratio"]
        pooled["optimal_ratio_basis"] = ratio_pack["optimal_ratio_basis"]

    # ── Propagated failure ──
    failures = []
    if not replica_results:
        failures.append("no replicas were run")
    if not accepted:
        failures.append(
            f"0 / {len(replica_results)} replicas passed acceptance")
    for r in replica_results:
        acc = r.get("acceptance", {})
        if not acc.get("accepted"):
            failures.append(
                f"rep_{r.get('replica_index')}: "
                f"{', '.join(acc.get('failed_criteria', [])) or r.get('error', 'unknown')}")

    pooled["quality_failures"] = failures
    pooled["accepted"] = bool(accepted)
    # `success` is now honest: it means "at least one replica equilibrated,
    # converged and was pooled".  # BEHAVIOUR CHANGE
    pooled["success"] = bool(accepted)
    pooled["md_completed"] = any(r.get("md_completed") for r in replica_results)

    if accepted:
        logger.info(f"  {target}/{pc_id}: {len(accepted)}/{len(replica_results)} "
                    f"replicas accepted and pooled")
    else:
        logger.error(
            "  %s/%s: NO REPLICA PASSED ACCEPTANCE — phase 4 result for this "
            "(target, pc_id) is marked success=False. Reasons: %s",
            target, pc_id, "; ".join(failures))
    return pooled


def _compact_replica(r: dict) -> dict:
    """Per-replica record kept inside the pooled entry (full detail stays on disk)."""
    occ = r.get("occupancy_analysis", {}) or {}
    return {
        "replica_index": r.get("replica_index"),
        "work_dir": r.get("work_dir"),
        "md_dir": (str(Path(r["work_dir"]) / "md") if r.get("work_dir") else None),
        "placement_seed": r.get("placement_seed"),
        "inputs_fingerprint": r.get("inputs_fingerprint"),
        "md_completed": r.get("md_completed"),
        # Requested vs EFFECTIVE leg length. They can only differ if something
        # between the request and mdrun substituted a length (historically
        # `quick` -> MD_QUICK_NS); _run_prepolymerization_md now raises on that,
        # and both numbers are kept so the record proves which one ran.
        "requested_time_ns": r.get("requested_time_ns"),
        "effective_time_ns": r.get("effective_time_ns"),
        # Sampling diagnostics: the observable's own correlation time and the
        # number of INDEPENDENT samples the analysis window actually held.
        "sampling": (r.get("occupancy_analysis") or {}).get("sampling"),
        # The composition ACTUALLY built, read back out of topol.top, kept per
        # replica so a pooled number can always be traced to the box it came
        # from without opening the trajectory directory.
        "requested_copies": r.get("requested_copies"),
        "built_copies": r.get("built_copies"),
        "composition_verified": (r.get("composition_verification") or {}).get(
            "verified"),
        "accepted": r.get("accepted"),
        "acceptance": r.get("acceptance"),
        "error": r.get("error"),
        "rmsd_mean_nm": r.get("rmsd_mean_nm"),
        "rmsd_drift_q3q4_nm": r.get("rmsd_drift_q3q4_nm"),
        "hbond_mean": r.get("hbond_mean"),
        "mmpbsa": r.get("mmpbsa"),
        "contact_freq_6A": occ.get("contact_freq_6A"),
        "EBN": occ.get("EBN"),
        "HBNMax": occ.get("HBNMax"),
        "residence_ns": occ.get("residence_ns"),
        "convergence": occ.get("convergence"),
        "optimal_ratio": r.get("optimal_ratio"),
    }


def _write_quality_report(results: dict, output_dir: Path):
    """Standalone quality/acceptance report (never a key of the results JSON)."""
    report = {"targets": {}, "rejected_legs": [], "accepted_legs": []}
    for target, tdata in results.items():
        if not isinstance(tdata, dict):
            continue
        report["targets"][target] = {}
        for pc_id, pdata in tdata.items():
            if not isinstance(pdata, dict):
                continue
            entry = {
                "success": pdata.get("success"),
                "accepted": pdata.get("accepted"),
                "n_replicas": pdata.get("n_replicas"),
                "n_replicas_accepted": pdata.get("n_replicas_accepted"),
                "pooling_basis": pdata.get("pooling_basis"),
                "quality_failures": pdata.get("quality_failures", []),
            }
            report["targets"][target][pc_id] = entry
            (report["accepted_legs"] if pdata.get("accepted")
             else report["rejected_legs"]).append(f"{target}/{pc_id}")
    try:
        (Path(output_dir) / "phase4_quality_report.json").write_text(
            json.dumps(report, indent=2, default=str))
    except Exception as e:
        logger.warning(f"Could not write phase4_quality_report.json: {e}")
    if report["rejected_legs"]:
        logger.error("PHASE 4 QUALITY: %d leg(s) REJECTED: %s",
                     len(report["rejected_legs"]), report["rejected_legs"])
    return report


def _ensure_centered_trajectory(traj_path: Path, tpr_path: Path) -> Path:
    """Generate PBC-centered trajectory (gmx trjconv -pbc mol -center) if missing.

    Returns path to centered xtc, written as md_centered.xtc beside the input.

    BLOCKER 11 — md_centered.xtc IS A TRANSIENT, NOT A CACHE.
    Measured on one 350 ns leg, this file was 11.19 GB against md.xtc's 11.33 GB:
    a byte-for-byte-sized duplicate of the trajectory, kept forever, for the sake
    of one analysis pass that reads the last 25% at a stride.  With replicas
    (PHASE4_N_REPLICAS) it multiplies by the replica count.

    We deliberately still centre the FULL-density md.xtc rather than the ~500-frame
    md_reduced.xtc: _analyze_monomer_occupancy's contact frequencies feed the Q3-vs-Q4
    convergence criterion, which is now a BLOCKING acceptance gate at a 10% tolerance,
    and halving the analysed sample would add noise to a test that rejects expensive
    MD.  Statistics are preserved; the DUPLICATE is what goes away, via
    _discard_centered_trajectory() once the caller has finished reading it.

    Set PHASE4_KEEP_CENTERED_TRAJECTORY = True to retain it for manual inspection.
    """
    from .config import GMX_BIN
    import subprocess
    traj_path = Path(traj_path)
    centered = traj_path.with_name("md_centered.xtc")
    if centered.exists() and centered.stat().st_mtime > traj_path.stat().st_mtime:
        return centered
    if not (traj_path.exists() and Path(tpr_path).exists()):
        return traj_path
    logger.info(f"    PBC-centering trajectory → {centered.name}")
    subprocess.run(
        [GMX_BIN, "trjconv", "-f", str(traj_path), "-s", str(tpr_path),
         "-o", str(centered), "-pbc", "mol", "-center"],
        input="1\n0\n", capture_output=True, text=True, timeout=1800)
    if not centered.exists():
        logger.warning("    trjconv centering failed, using raw trajectory")
        return traj_path
    return centered


def _discard_centered_trajectory(centered: Path, source: Path) -> None:
    """Delete the transient md_centered.xtc produced by _ensure_centered_trajectory.

    No-ops when centering was skipped (centered is the source trajectory itself),
    when the file is not ours, or when PHASE4_KEEP_CENTERED_TRAJECTORY is set.
    Failure to delete is never fatal — it costs disk, not correctness.
    """
    try:
        centered, source = Path(centered), Path(source)
        if centered == source or centered.name != "md_centered.xtc":
            return                                   # nothing of ours to remove
        if _cfg("PHASE4_KEEP_CENTERED_TRAJECTORY"):
            return
        if centered.exists():
            freed_mb = centered.stat().st_size / 1e6
            centered.unlink()
            logger.info(f"    Removed transient {centered.name} "
                        f"({freed_mb:.0f} MB; BLOCKER 11 — it duplicates md.xtc)")
    except Exception as e:
        logger.warning(f"    could not remove {centered}: {e}")


def _analyze_monomer_occupancy(traj_path: Path, top_path: Path,
                                 functional_monomers: list,
                                 crosslinker: str,
                                 target: str = None,
                                 cutoff_nm: float = 0.60,
                                 use_q4: bool = None) -> dict:
    """
    Analyze monomer-epitope interactions from MD trajectory.

    Metrics (literature-based):
      1. Contact frequency at 6Å cutoff (vdW interactions included)
      2. RDF g(r) peak height per monomer type
      3. Coordination number (avg monomers within 6Å)
      4. Residence time — reported in ps/ns, NOT in stride units

    Frame selection.  # BEHAVIOUR CHANGE (2026-08 sampling audit)
      `use_q4=None` (the default now) reads PHASE4_ANALYSIS_WINDOW_FRACTION,
      which ships at 0.50 — the last HALF of the trajectory.  It used to
      default to the last QUARTER.  Measured reason: the integrated
      autocorrelation time of the per-copy contact signal is 8-33 ns on the
      archived legs, so the Q4 window of a 30 ns leg (7.5 ns) holds well under
      one independent sample per molecule and window-to-window noise falls only
      as T^-0.31 instead of T^-0.50.  Throwing away three quarters of a
      trajectory that has already been paid for costs precision and saves
      nothing.  `use_q4=True/False` still force 0.25 / 0.50 explicitly.

    Trajectory is PBC-centered first (gmx trjconv -pbc mol -center) to handle
    proteins that crossed periodic boundaries during long MD.
    """
    # BLOCKER 11: tracked so the finally-block can drop the transient duplicate
    # on EVERY exit path, including the two exception handlers.
    centered_traj = None
    try:
        import MDAnalysis as mda

        # PBC centering for analysis (writes the transient md_centered.xtc)
        tpr_for_center = Path(top_path)
        if not str(tpr_for_center).endswith('.tpr'):
            tpr_candidate = Path(traj_path).with_name("md.tpr")
            if tpr_candidate.exists():
                tpr_for_center = tpr_candidate
        centered_traj = _ensure_centered_trajectory(Path(traj_path), tpr_for_center)

        u = mda.Universe(str(top_path), str(centered_traj))

        protein = u.select_atoms("protein")
        logger.info(f"    Template: {len(protein)} atoms")

        if len(protein) == 0:
            return {"error": "No protein atoms found in template region"}

        cutoff_A = cutoff_nm * 10  # nm to Angstrom

        # Frame selection. Default comes from PHASE4_ANALYSIS_WINDOW_FRACTION
        # (0.50 = the last half); use_q4=True/False force 0.25/0.50.
        n_frames = len(u.trajectory)
        if use_q4 is None:
            window_fraction = _analysis_window_fraction()
            window_source = "PHASE4_ANALYSIS_WINDOW_FRACTION"
        else:
            window_fraction = 0.25 if use_q4 else 0.50
            window_source = f"use_q4={use_q4} (explicit caller override)"
        start_frame = int(n_frames * (1.0 - window_fraction))
        start_frame = max(0, min(start_frame, max(0, n_frames - 1)))
        stride = max(1, (n_frames - start_frame) // 200)
        window_label = (f"last {window_fraction*100:.0f}% "
                        f"[{window_source}]")

        # Real time per analysed step — needed to report residence in ps, not
        # in "stride units".  # BEHAVIOUR CHANGE (2026-08 audit)
        try:
            t0 = float(u.trajectory[0].time)
            frame_dt_ps = (float(u.trajectory[1].time) - t0) if n_frames >= 2 else 0.0
        except Exception:
            frame_dt_ps = float(getattr(u.trajectory, "dt", 0.0) or 0.0)
        step_ps = frame_dt_ps * stride

        logger.info(f"    Frames: {start_frame}-{n_frames} {window_label} "
                    f"(stride={stride}, {step_ps:.1f} ps/step), "
                    f"cutoff={cutoff_nm*10:.0f}Å")

        # Build residue → monomer type mapping from topology
        # Excludes solvent/ions AND the protonation-variant protein names --
        # see _PH_PROTEIN_RESNAMES. The old selection was
        # "not protein and not resname SOL NA CL", which (a) missed renamed
        # titratable residues and (b) used a narrower notion of "solvent" than
        # the [ molecules ] scan below, so the two could disagree by
        # construction. Both now read the same sets.
        _excl = " ".join(sorted(_SOLVENT_IONS | _PH_PROTEIN_RESNAMES))
        non_protein = u.select_atoms(f"not protein and not resname {_excl}")
        # crosslinker may legitimately be None: a functional-only control
        # box (the grid's APTES-only point) has no crosslinker to score.
        all_monomers_list = list(functional_monomers) + (
            [crosslinker] if crosslinker else [])

        top_file = Path(top_path).with_name("topol.top")

        # THE MAPPING IS LOAD-BEARING — AN EMPTY ONE IS NOT A MEASUREMENT
        # (REVIEW FINDING 9).
        #
        # Every contact/EBN/residence/HBNMax number below is keyed on
        # res_to_monomer. If it comes up empty or partial, EVERY affected
        # monomer scores contact_freq 0.0 and EBN 0 even when it is 2 A from
        # the protein in every frame — and the convergence gate then reads
        # 0-vs-0 as converged=True, so the replica is ACCEPTED and pooled. The
        # only signal was logger.info("Monomer mapping: 0 residues").
        #
        # Three ways it used to come up short, all silent:
        #   (a) topol.top absent            -> {} (the `if exists()` had no else)
        #   (b) SOL listed before the monomers -> `break` cut the scan short
        #   (c) a monomer missing from [ molecules ] -> partial mapping
        # (b) is fixed here by skipping solvent/ion lines instead of breaking;
        # (a) and (c) now raise.
        if not top_file.exists():
            return {"error": (
                f"topol.top not found at {top_file} — the residue->monomer "
                f"mapping cannot be built, and without it every monomer scores "
                f"contact_freq 0.0 / EBN 0 while appearing converged. Refusing "
                f"to report zeros as measurements.")}

        res_to_monomer = {}
        in_molecules = False
        mol_idx = 0
        for line in top_file.read_text().split("\n"):
            if "[ molecules ]" in line:
                in_molecules = True
                continue
            if in_molecules and line.strip() and not line.startswith(";"):
                if line.strip().startswith("["):
                    break               # next section: [ molecules ] is over
                parts = line.split()
                if len(parts) >= 2:
                    mol_name = parts[0]
                    try:
                        mol_count = int(parts[1])
                    except ValueError:
                        continue
                    if mol_name.startswith("Protein"):
                        continue
                    # `continue`, NOT `break`: a topology that lists SOL before
                    # the monomers used to truncate the scan and leave the
                    # mapping empty. Solvent/ion order is not ours to assume.
                    if mol_name.upper() in _SOLVENT_IONS:
                        continue
                    for _ci in range(mol_count):
                        res_to_monomer[mol_idx] = mol_name
                        mol_idx += 1

        mapped_types = set(res_to_monomer.values())
        expected_types = set(all_monomers_list)
        logger.info(f"    Monomer mapping: {len(res_to_monomer)} residues "
                    f"({len(mapped_types)} types)")

        if not res_to_monomer:
            return {"error": (
                f"residue->monomer mapping is EMPTY: no monomer entries found "
                f"in the [ molecules ] section of {top_file}. Every monomer "
                f"would score contact_freq 0.0 / EBN 0 and the convergence "
                f"gate would read 0-vs-0 as converged. Refusing.")}

        missing_types = expected_types - mapped_types
        if missing_types:
            return {"error": (
                f"residue->monomer mapping is INCOMPLETE: "
                f"{sorted(missing_types)} appear in the requested composition "
                f"but not in the [ molecules ] section of {top_file} "
                f"(mapped: {sorted(mapped_types)}). Those monomers would score "
                f"contact_freq 0.0 / EBN 0 — indistinguishable from a monomer "
                f"that genuinely never touched the protein. Refusing.")}

        # The mapping indexes into non_protein.residues, so its length must
        # match or the residue->monomer attribution is off by an offset.
        n_mon_res = len(non_protein.residues)
        if len(res_to_monomer) != n_mon_res:
            return {"error": (
                f"residue->monomer mapping covers {len(res_to_monomer)} "
                f"molecules but the trajectory has {n_mon_res} non-solvent "
                f"residues. The mapping is positional, so a mismatch "
                f"misattributes contacts to the wrong monomer type. Refusing.")}

        # Per-frame analysis
        mon_residues = non_protein.residues
        contact_per_frame = {m: [] for m in all_monomers_list}
        min_dist_per_frame = {m: [] for m in all_monomers_list}
        # For residence time: track per-residue contact state
        res_contact_durations = {m: [] for m in all_monomers_list}
        current_duration = {ri: 0 for ri in range(len(mon_residues))}
        # A residue whose distance computation raises contributes ZERO contacts
        # to that frame, indistinguishable from a residue that was genuinely far
        # away (REVIEW FINDING 19). The old handler was a bare `pass`. Count
        # them and abort rather than silently under-counting contacts.
        n_dist_errors = 0
        first_dist_error = None

        for ts in u.trajectory[start_frame::stride]:
            head_pos = protein.positions
            frame_contacts = {m: 0 for m in all_monomers_list}
            frame_min_dists = {m: [] for m in all_monomers_list}

            for ri, res in enumerate(mon_residues):
                m_name = res_to_monomer.get(ri)
                if m_name not in frame_contacts:
                    continue
                try:
                    dists = np.linalg.norm(
                        res.atoms.positions[:, np.newaxis, :] -
                        head_pos[np.newaxis, :, :], axis=2)
                    min_dist = float(np.min(dists))
                    frame_min_dists[m_name].append(min_dist)

                    in_contact = min_dist < cutoff_A
                    if in_contact:
                        frame_contacts[m_name] += 1
                        current_duration[ri] += 1
                    else:
                        if current_duration[ri] > 0:
                            res_contact_durations[m_name].append(current_duration[ri])
                        current_duration[ri] = 0
                except Exception as e:
                    n_dist_errors += 1
                    if first_dist_error is None:
                        first_dist_error = f"{type(e).__name__}: {e}"

            for m in all_monomers_list:
                contact_per_frame[m].append(frame_contacts[m])
                if frame_min_dists[m]:
                    min_dist_per_frame[m].append(min(frame_min_dists[m]))

        if n_dist_errors:
            # Do not report an under-counted contact frequency as a
            # measurement: the shortfall is invisible in the output and biases
            # every downstream number (EBN, the synthesis ratio, convergence).
            return {"error": (
                f"{n_dist_errors} per-residue distance computation(s) failed "
                f"during the occupancy analysis (first: {first_dist_error}). "
                f"Each failure silently contributes zero contacts for that "
                f"residue in that frame, so contact_freq_6A would be "
                f"under-counted by an unknown amount. Refusing to report it.")}

        # Flush residues still in contact at the end of the window — previously
        # these runs were dropped entirely, biasing residence time downwards.
        for ri, dur in current_duration.items():
            if dur > 0:
                m_name = res_to_monomer.get(ri)
                if m_name in res_contact_durations:
                    res_contact_durations[m_name].append(dur)

        # ── Metrics ──
        results = {}

        # 1. Contact frequency (6Å): avg contacts per frame
        contact_freq = {}
        for m, counts in contact_per_frame.items():
            contact_freq[m] = round(float(np.mean(counts)), 3) if counts else 0.0
        results["contact_freq_6A"] = contact_freq
        results["total_contact_mean"] = round(float(sum(contact_freq.values())), 3)
        logger.info(f"    Contact freq (6Å): {contact_freq}")

        # 2. Mean minimum distance per monomer type
        mean_min_dist = {}
        for m, dists in min_dist_per_frame.items():
            mean_min_dist[m] = round(float(np.mean(dists)), 2) if dists else 99.0
        results["mean_min_dist_A"] = mean_min_dist
        logger.info(f"    Mean min dist (Å): {mean_min_dist}")

        # 3. Residence time — REPORTED IN TIME, not in stride units.
        # `residence_steps` is the raw count of analysed frames; multiply by
        # step_ps (= trajectory dt x stride) to get real time.
        residence_steps, residence_ps, residence_ns = {}, {}, {}
        for m, durations in res_contact_durations.items():
            if durations:
                mean_steps = float(np.mean(durations))
                residence_steps[m] = round(mean_steps, 1)
                residence_ps[m] = round(mean_steps * step_ps, 1)
                residence_ns[m] = round(mean_steps * step_ps / 1000.0, 4)
            else:
                residence_steps[m] = 0.0
                residence_ps[m] = 0.0
                residence_ns[m] = 0.0
        results["residence_steps"] = residence_steps
        results["residence_ps"] = residence_ps
        results["residence_ns"] = residence_ns
        results["analysis_step_ps"] = round(step_ps, 3)
        results["trajectory_dt_ps"] = round(frame_dt_ps, 3)
        # Legacy key kept, but now explicitly labelled so nobody reads it as time
        results["residence_frames"] = residence_steps
        logger.info(f"    Residence (ns): {residence_ns} "
                    f"[{residence_steps} analysed steps x {step_ps:.1f} ps]")

        # 3b. THE REAL SAMPLE SIZE: how many times the 6 A shell was crossed.
        # An occupancy average is not made of frames and not of nanoseconds; it
        # is made of independent association/dissociation EVENTS, which is the
        # unit PyLipID uses for exactly this class of observable (Song et al.,
        # JCTC 18:1188, 2022). `res_contact_durations` already holds one entry
        # per completed contact episode, so the count is free — it was simply
        # never reported, and it is the number that says whether an occupancy is
        # worth a CI at all. A leg with 3 events has no error bar no matter how
        # many frames it wrote.
        n_events = {m: len(d) for m, d in res_contact_durations.items()}
        results["n_binding_events"] = n_events
        # Same quantity per copy, so compositions with different monomer counts
        # are comparable.
        _ncop = {}
        for _ri, _m in res_to_monomer.items():
            _ncop[_m] = _ncop.get(_m, 0) + 1
        results["n_binding_events_per_copy"] = {
            m: round(n / _ncop[m], 3) for m, n in n_events.items() if _ncop.get(m)}
        logger.info(f"    Binding events (6 A crossings): {n_events} "
                    f"= {results['n_binding_events_per_copy']} per copy")

        # 3c. WITHIN-LEG ERROR BAR BY BLOCK AVERAGING (Flyvbjerg & Petersen,
        # JCP 91:461, 1989; Frenkel & Smit App. D.3; LiveCoMS 1(1):5067 Eqs.
        # 13-14). BSE(n) = s(block means)/sqrt(M) is computed over a ladder of
        # block sizes; once a block is longer than the correlation time the
        # block means decorrelate and BSE stops rising, and that PLATEAU is the
        # honest standard error of this leg's mean. If no plateau appears the
        # leg cannot support an error bar, and saying so is the result — that is
        # the whole point of the method. This REPLACES nothing: the block
        # DIFFERENCE heuristic stays where it is, but it is a drift diagnostic,
        # while this is the statistic a reviewer will ask for.
        block_se = {}
        for m, counts in contact_per_frame.items():
            arr = np.asarray(counts, dtype=float)
            N = len(arr)
            if N < 20 or arr.std() == 0:
                block_se[m] = {"plateau_se": None, "reason":
                               "too few frames or a constant signal"}
                continue
            ladder = []
            nb = 1
            while nb * 5 <= N:          # need >= 5 blocks for s() to mean anything
                M = N // nb
                means = arr[:M * nb].reshape(M, nb).mean(axis=1)
                ladder.append({"block_frames": nb,
                               "block_ps": round(nb * step_ps, 1),
                               "n_blocks": M,
                               "bse": round(float(means.std(ddof=1) /
                                                  math.sqrt(M)), 4)})
                nb *= 2
            # Plateau = the largest BSE on the ladder. It is an estimate FROM
            # BELOW: BSE rises toward the true SE, so if it is still climbing at
            # the last usable block size the value is a lower bound, and that is
            # flagged rather than hidden.
            if ladder:
                best = max(ladder, key=lambda d: d["bse"])
                still_rising = best is ladder[-1] and len(ladder) > 1
                block_se[m] = {
                    "plateau_se": best["bse"],
                    "plateau_block_ps": best["block_ps"],
                    "n_blocks_at_plateau": best["n_blocks"],
                    "still_rising_at_largest_block": bool(still_rising),
                    "ladder": ladder,
                }
        results["block_average_se"] = block_se
        logger.info("    Block-average SE (F&P plateau): " + ", ".join(
            f"{m}={v.get('plateau_se')}"
            + ("[rising]" if v.get("still_rising_at_largest_block") else "")
            for m, v in block_se.items()))

        # 4. RDF-like: fraction of time each monomer type is "close" (< 6Å)
        proximity_frac = {}
        for m, dists in min_dist_per_frame.items():
            if dists:
                proximity_frac[m] = round(sum(1 for d in dists if d < cutoff_A) / len(dists), 3)
            else:
                proximity_frac[m] = 0.0
        results["proximity_fraction"] = proximity_frac
        logger.info(f"    Proximity frac: {proximity_frac}")

        # 5. Monomer-monomer min distance (cavity shape analysis)
        logger.info("    Cavity analysis: monomer-monomer min distances...")
        pair_mindists_md = {}

        for ts in u.trajectory[start_frame::stride]:
            for i, m1 in enumerate(all_monomers_list):
                for m2 in all_monomers_list[i+1:]:
                    key = f"{m1}-{m2}"
                    atoms1 = []
                    atoms2 = []
                    for ri, res in enumerate(mon_residues):
                        mn = res_to_monomer.get(ri)
                        if mn == m1:
                            atoms1.append(res.atoms.positions)
                        elif mn == m2:
                            atoms2.append(res.atoms.positions)
                    if atoms1 and atoms2:
                        pos1 = np.vstack(atoms1)
                        pos2 = np.vstack(atoms2)
                        dists = np.linalg.norm(
                            pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :], axis=2)
                        md = float(np.min(dists))
                        if key not in pair_mindists_md:
                            pair_mindists_md[key] = []
                        pair_mindists_md[key].append(md)

        # Average min distance over frames
        pair_dists_md = {}
        for key, dists in pair_mindists_md.items():
            pair_dists_md[key] = round(float(np.mean(dists)), 2) if dists else 99.0
        results["pair_distances_md_A"] = pair_dists_md
        logger.info(f"    MD pair min-dists (Å): {pair_dists_md}")

        # ── 6. EBN: Effective Binding Number (Yuan et al. 2024) ──
        ebn = {}
        for m, counts in contact_per_frame.items():
            ebn[m] = int(max(counts)) if counts else 0
        results["EBN"] = ebn
        logger.info(f"    EBN (max simultaneous): {ebn}")

        # ── 7. HBNMax: Maximum H-bond Number (Yuan et al. 2024) ──
        # PROTEIN <-> MONOMER ONLY.  # BEHAVIOUR CHANGE (2026-08 audit)
        # The old call passed donors_sel="protein or (monomer)" AND
        # acceptors_sel="protein or (monomer)", so every intra-protein backbone
        # H-bond was counted as a monomer H-bond. That is why DVB and TMOS —
        # which have no H-bond donors at all — scored 168-195. The two sides are
        # now DISJOINT and both directions are summed per frame:
        #     (protein donor -> monomer acceptor) + (monomer donor -> protein acceptor)
        try:
            from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
                HydrogenBondAnalysis as HBA)

            # Load universe with TPR (has charges/bonds) for H-bond analysis
            tpr_path = Path(top_path).with_name("md.tpr")
            if tpr_path.exists():
                u_hb = mda.Universe(str(tpr_path), str(centered_traj))
            else:
                u_hb = u  # fallback to GRO (may fail — no bond/charge info)

            hbnmax = {}
            hb_mean = {}
            hb_lifetime_ps = {}
            hb_failed = []          # monomers whose HBA raised (REVIEW FINDING 19)
            # Same exclusion as the occupancy selection above — this copy was
            # missed when that one was fixed, so HBNMax alone still counted the
            # two neutral lysines as monomers and skipped itself.
            mon_residues_hb = u_hb.select_atoms(
                f"not protein and not resname {_excl}").residues
            if len(mon_residues_hb) != len(mon_residues):
                logger.error(
                    "    HBNMax: monomer residue count differs between the "
                    "analysis universe (%d) and the TPR universe (%d) — the "
                    "residue→monomer mapping cannot be trusted, skipping HBNMax",
                    len(mon_residues), len(mon_residues_hb))
                mon_residues_hb = []

            for m in all_monomers_list:
                m_resids = [res.resid for ri, res in enumerate(mon_residues_hb)
                            if res_to_monomer.get(ri) == m]
                if not m_resids:
                    continue
                m_sel = "(" + " or ".join(f"resid {r}" for r in m_resids) + ")"
                m_sel = (f"(not protein and not resname SOL WAT HOH NA CL "
                         f"and {m_sel})")
                try:
                    per_frame = {}
                    triples = {}   # (d,h,a) -> list of frame indices
                    for grp_d, grp_a in (("protein", m_sel), (m_sel, "protein")):
                        hb = HBA(u_hb,
                                 donors_sel=f"({grp_d}) and {_HB_DONOR_ACCEPTOR_SEL}",
                                 hydrogens_sel=f"({grp_d}) and {_HB_HYDROGEN_SEL}",
                                 acceptors_sel=f"({grp_a}) and {_HB_DONOR_ACCEPTOR_SEL}",
                                 d_a_cutoff=3.5, d_h_a_angle_cutoff=150,
                                 update_selections=False)
                        hb.run(start=start_frame, step=stride, verbose=False)
                        for row in hb.results.hbonds:
                            fr = int(row[0])
                            per_frame[fr] = per_frame.get(fr, 0) + 1
                            key = (int(row[1]), int(row[2]), int(row[3]))
                            triples.setdefault(key, []).append(fr)

                    hbnmax[m] = int(max(per_frame.values())) if per_frame else 0
                    hb_mean[m] = (round(float(np.mean(list(per_frame.values()))), 2)
                                  if per_frame else 0.0)
                    # REAL lifetime: mean length of consecutive-frame runs for a
                    # given donor-H-acceptor triple, converted to ps.
                    runs = []
                    for frames in triples.values():
                        fs = sorted(set(frames))
                        run = 1
                        for a, b in zip(fs, fs[1:]):
                            if b - a <= stride:
                                run += 1
                            else:
                                runs.append(run)
                                run = 1
                        runs.append(run)
                    hb_lifetime_ps[m] = (round(float(np.mean(runs)) * step_ps, 1)
                                         if runs else 0.0)
                except Exception as e:
                    # A FAILED H-BOND ANALYSIS IS NOT ZERO H-BONDS
                    # (REVIEW FINDING 19). This wrote 0 / 0.0 / 0.0 and logged
                    # at DEBUG, which the pipeline (INFO) never emits — so the
                    # failure produced exactly the DVB/TMOS "no H-bond donors
                    # yet HBNMax 168" signature the audit was chasing, in
                    # reverse: a real donor silently scoring zero.
                    logger.error(
                        f"    HBNMax FAILED for {m}: {type(e).__name__}: {e} — "
                        f"recorded as None (not 0), so the replica's analysis "
                        f"is marked incomplete rather than reporting no "
                        f"H-bonds.")
                    hbnmax[m] = None
                    hb_mean[m] = None
                    hb_lifetime_ps[m] = None
                    hb_failed.append(m)

            results["HBNMax"] = hbnmax
            results["hbond_mean_per_frame"] = hb_mean
            results["hbond_lifetime_ps"] = hb_lifetime_ps
            # Legacy key: used to hold the MEAN COUNT PER FRAME, not a lifetime.
            results["hbond_lifetime_avg"] = hb_mean
            results["hbond_selection"] = (
                "protein<->monomer only (disjoint sides); donors/acceptors "
                "restricted to N/O/S by mass; hydrogens_sel passed explicitly "
                "(no min_charge guess)")
            results["hbond_selection_strings"] = {
                "donor_acceptor": _HB_DONOR_ACCEPTOR_SEL,
                "hydrogens": _HB_HYDROGEN_SEL,
            }
            results["hbond_failed_monomers"] = hb_failed
            results["hbond_analysis_complete"] = not hb_failed
            logger.info(f"    HBNMax (protein<->monomer): {hbnmax}")
            logger.info(f"    H-bond mean/frame: {hb_mean}")
            logger.info(f"    H-bond lifetime (ps): {hb_lifetime_ps}")
            if hb_failed:
                logger.error(f"    H-bond analysis INCOMPLETE for {hb_failed} — "
                             f"their HBNMax is None, not 0")
        except ImportError as e:
            # Raised from DEBUG: MDAnalysis' HBA missing means HBNMax was never
            # computed for ANY monomer, which is worth more than a hidden line.
            logger.error(
                f"HydrogenBondAnalysis unavailable ({e}) — HBNMax / "
                f"hbond_lifetime were NOT computed for this replica")
            results["hbond_analysis_complete"] = False
            results["hbond_failed_monomers"] = list(all_monomers_list)

        # ── 7b. SAMPLING DIAGNOSTICS ─────────────────────────────
        #
        # WHY THIS EXISTS.  Splitting one archived 350 ns CD63 trajectory into
        # 50 non-overlapping 7 ns windows (= the Q4 window of a 30 ns leg) and
        # recomputing the 4-monomer ranking in each window gave 19 DISTINCT
        # orderings; the most common accounted for 6 of 50, and the winner
        # identity was IBTES 20 / TTMS 14 / PTES 14 / TMOS 2.  A ranking that
        # unstable is not a measurement.  Every leg therefore now REPORTS its
        # own correlation time, its effective sample size, and whether its own
        # ranking survives being recomputed in sub-windows — so a bad run is
        # visible from point 1 instead of at the end of the grid.
        n_copies_present = {}
        for _ri, _m in res_to_monomer.items():
            n_copies_present[_m] = n_copies_present.get(_m, 0) + 1
        window_ns = round(
            (len(next(iter(contact_per_frame.values()), [])) * step_ps) / 1000.0, 4)
        sampling = {
            "analysis_window_fraction": round(window_fraction, 4),
            "analysis_window_source": window_source,
            "analysis_window_ns": window_ns,
            "analysis_step_ps": round(step_ps, 3),
            "n_copies_present": n_copies_present,
            "tau_int_ns": {},
            "n_independent_samples": {},
            "per_copy_contact": {},
            "sub_window_means": {},
            "sub_window_cv": {},
        }
        for m, counts in contact_per_frame.items():
            n_m = max(1, n_copies_present.get(m, 1))
            per_copy = np.asarray(counts, dtype=float) / n_m
            tau_frames = _tau_int(per_copy)
            tau_ns = round(tau_frames * step_ps / 1000.0, 3)
            sampling["tau_int_ns"][m] = tau_ns
            sampling["n_independent_samples"][m] = (
                round(window_ns / tau_ns, 2) if tau_ns > 0 else None)
            sampling["per_copy_contact"][m] = round(float(per_copy.mean()), 4) \
                if len(per_copy) else None
        # Sub-window recomputation of the SAME observable the grid ranks on.
        n_sub = 4
        n_pts = min((len(v) for v in contact_per_frame.values()), default=0)
        if n_pts >= n_sub * 2:
            edge = n_pts // n_sub
            for m, counts in contact_per_frame.items():
                n_m = max(1, n_copies_present.get(m, 1))
                pc = np.asarray(counts, dtype=float) / n_m
                sub = [round(float(pc[i * edge:(i + 1) * edge].mean()), 4)
                       for i in range(n_sub)]
                sampling["sub_window_means"][m] = sub
                mu = float(np.mean(sub))
                sampling["sub_window_cv"][m] = (
                    round(float(np.std(sub, ddof=1)) / mu, 3) if mu > 0 else None)
            sampling["sub_window_ns"] = round(edge * step_ps / 1000.0, 4)
            # Does THIS leg's own ranking of its monomer types survive being
            # recomputed in each sub-window?
            orders = []
            for i in range(n_sub):
                vals = []
                for m, counts in contact_per_frame.items():
                    n_m = max(1, n_copies_present.get(m, 1))
                    pc = np.asarray(counts, dtype=float) / n_m
                    vals.append((float(pc[i * edge:(i + 1) * edge].mean()), m))
                vals.sort(reverse=True)
                orders.append(tuple(m for _, m in vals))
            sampling["sub_window_orders"] = ["|".join(o) for o in orders]
            sampling["sub_window_ranking_stable"] = (len(set(orders)) == 1)
            if len(all_monomers_list) > 1 and not sampling["sub_window_ranking_stable"]:
                logger.error(
                    "    SUB-WINDOW RANKING IS UNSTABLE: this leg's own monomer "
                    "ordering changes %d times across %d sub-windows of "
                    "%.2f ns (%s). The leg's mean is still reported, but it is "
                    "one draw from a distribution this wide — see "
                    "occupancy_analysis.sampling.",
                    len(set(orders)), n_sub, sampling["sub_window_ns"],
                    sampling["sub_window_orders"])
        else:
            sampling["sub_window_ranking_stable"] = None
            sampling["sub_window_note"] = (
                f"only {n_pts} analysed frames — fewer than {n_sub * 2}, so the "
                f"sub-window stability check was not run")
        # THE HONEST HEADLINE. n_eff < ~5 means the leg mean is essentially one
        # configuration and cannot separate compositions on its own.
        _neff = [v for v in sampling["n_independent_samples"].values()
                 if isinstance(v, (int, float))]
        sampling["min_n_independent_samples"] = min(_neff) if _neff else None
        if _neff and min(_neff) < 5:
            logger.error(
                "    UNDER-SAMPLED LEG: the analysis window is %.2f ns but the "
                "observable's correlation time is up to %.2f ns, i.e. only %.2f "
                "independent samples. Noise on this leg falls as T^-0.31, not "
                "T^-0.50; spend the budget on REPLICAS, not nanoseconds.",
                window_ns, max(sampling["tau_int_ns"].values() or [0]),
                min(_neff))
        results["sampling"] = sampling

        # ── 8. Block-averaged convergence (Polania & Jimenez 2024) ──
        # Four equal blocks of the ANALYSED window; the last two must agree to
        # within PHASE4_CONVERGENCE_TOL_PCT for every monomer type.
        #
        # THE BLOCK MEANS ARE A DRIFT DIAGNOSTIC, NOT AN ERROR BAR.
        # MEASURED: the SEM built from these 4 blocks understates the true
        # window-to-window SD by a median factor of 2.8x (up to 7.5x) on the
        # archived legs, because the blocks are ~2 ns apart while tau is
        # 8-33 ns — they are sub-correlation-time slices of ONE configuration,
        # not quasi-independent samples. Nothing downstream may use them as n.
        tol_pct = float(_cfg("PHASE4_CONVERGENCE_TOL_PCT"))
        n_analyzed = min((len(v) for v in contact_per_frame.values()), default=0)
        if n_analyzed >= 4:
            q1 = n_analyzed // 4
            window_freqs = []
            for w in range(4):
                w_start = w * q1
                w_end = (w + 1) * q1
                w_freq = {}
                for m, counts in contact_per_frame.items():
                    w_counts = counts[w_start:w_end]
                    w_freq[m] = round(float(np.mean(w_counts)), 3) if w_counts else 0
                window_freqs.append(w_freq)

            conv_details = {}
            # ZERO-vs-ZERO IS NOT CONVERGENCE (REVIEW FINDING 9) — BUT IT IS
            # ALSO NOT A FAILED ANALYSIS.  # BEHAVIOUR CHANGE (2026-08 sampling)
            #
            # Two different things used to collapse into converged=None:
            #   (a) a species that is PRESENT and recorded zero contacts. That
            #       is a MEASUREMENT — contact frequency 0 at this composition —
            #       and rejecting the whole leg for it biases rejection toward
            #       the scarce-species end of the grid, which is exactly the end
            #       where the imprinting question is being asked. Observed:
            #       both smoke legs (2 and 10 APTES) were rejected on this
            #       criterion alone, and the declared grid contains points with
            #       3 and 6 APTES.
            #   (b) EVERY species recording zero contacts, or an unusable
            #       analysis. That is a dead box and stays a rejection.
            # `converged` now answers only "did the species WITH signal drift?".
            # The zero-contact species are reported separately so the acceptance
            # gate can treat them as data rather than as a hole.
            drifting, tested, no_data = [], [], []
            for m in all_monomers_list:
                f3 = window_freqs[2].get(m, 0)
                f4 = window_freqs[3].get(m, 0)
                if max(f3, f4) > 0:
                    diff_pct = abs(f4 - f3) / max(f3, f4) * 100
                    conv_details[m] = round(diff_pct, 1)
                    tested.append(m)
                    if diff_pct > tol_pct:
                        drifting.append(m)
                else:
                    # No contacts in either comparison window: there is nothing
                    # to be converged ABOUT. Report it as such, never as 0%.
                    conv_details[m] = None
                    no_data.append(m)

            zero_is_data = bool(_cfg("PHASE4_ZERO_CONTACT_IS_A_MEASUREMENT"))
            if not tested:
                # Nothing anywhere touched the protein: not a composition
                # result, a dead box.
                converged = None
                reason = ("no monomer species recorded a single contact in "
                          "either comparison window — this is a dead box, not a "
                          "composition with a low contact frequency")
            elif drifting:
                converged = False
                reason = (f"{drifting} still drifting between block 3 and "
                          f"block 4 by more than {tol_pct}%")
            elif no_data and not zero_is_data:
                converged = None
                reason = (f"{no_data} recorded zero contacts in both comparison "
                          f"windows and PHASE4_ZERO_CONTACT_IS_A_MEASUREMENT is "
                          f"False")
            else:
                converged = True
                reason = None

            results["convergence"] = {
                "converged": converged,
                "window_diff_pct": conv_details,
                "tolerance_pct": tol_pct,
                # KEPT FOR DRIFT DIAGNOSIS ONLY. Never an error bar: the SEM
                # built from these blocks understates the true window-to-window
                # SD by a median 2.8x (measured), because the blocks sit inside
                # one correlation time.
                "block_means": window_freqs,
                "block_means_are_not_an_error_bar": True,
                "n_blocks": 4,
                "n_frames_per_block": q1,
                "monomers_tested_for_drift": tested,
                "monomers_drifting": drifting,
                "monomers_without_contacts": no_data,
                "zero_contact_is_a_measurement": zero_is_data,
                "reason": reason,
            }
            if converged and no_data:
                logger.info(
                    f"    Convergence: YES for {tested} (window diff%: "
                    f"{ {k: v for k, v in conv_details.items() if v is not None} }, "
                    f"tol={tol_pct}%). {no_data} recorded ZERO contacts — "
                    f"recorded as a measured zero at this composition, NOT as a "
                    f"failed analysis (PHASE4_ZERO_CONTACT_IS_A_MEASUREMENT).")
            elif converged:
                logger.info(f"    Convergence: YES (window diff%: {conv_details}, "
                            f"tol={tol_pct}%)")
            elif converged is None:
                logger.error(f"    Convergence: UNDECIDABLE — {reason}. Replica "
                             f"will be REJECTED.")
            else:
                # Was: logged and ignored. Now it fails the replica.
                logger.error(f"    Convergence: NO — window diff%: {conv_details} "
                             f"exceeds tol={tol_pct}%. This replica will be "
                             f"REJECTED and not pooled.")
        else:
            results["convergence"] = {
                "converged": None,
                "reason": f"only {n_analyzed} analysed frames (<4 blocks)",
                "tolerance_pct": tol_pct,
            }
            logger.error(f"    Convergence: UNDETERMINED — only {n_analyzed} "
                         f"analysed frames. Replica will be REJECTED.")

        # ── 9. Crosslinker-Monomer Proximity (Rajpal 2023 review) ──
        xl_proximity = {}
        for m in (functional_monomers if crosslinker else []):
            key = f"{crosslinker}-{m}"
            rev_key = f"{m}-{crosslinker}"
            dist = pair_dists_md.get(key) or pair_dists_md.get(rev_key)
            if dist:
                xl_proximity[m] = dist
        results["crosslinker_proximity_A"] = xl_proximity
        xl_close = all(d < 10.0 for d in xl_proximity.values()) if xl_proximity else False
        results["crosslinker_well_positioned"] = xl_close
        logger.info(f"    Crosslinker proximity: {xl_proximity} → "
                    f"{'OK' if xl_close else 'WARNING: crosslinker too far from some monomers'}")

        results["occupancy"] = contact_freq  # backward compatibility
        results["n_frames_analyzed"] = n_analyzed
        results["stride"] = stride
        results["cutoff_nm"] = cutoff_nm

        return results

    except ImportError:
        logger.warning("MDAnalysis not available")
        return {"error": "MDAnalysis not installed"}
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")
        return {"error": str(e)}
    finally:
        # The centred trajectory has now been fully consumed (u.trajectory is
        # out of scope for every return path above). Drop the duplicate.
        if centered_traj is not None:
            _discard_centered_trajectory(centered_traj, Path(traj_path))


def _compute_per_atom_rdf(traj_path: Path, top_path: Path,
                           functional_monomers: list,
                           crosslinker: str) -> dict:
    """Per-atom RDF between monomer functional groups and epitope (Yuan 2024).

    Identifies which functional groups (OH, NH2, COOH, etc.) drive binding.
    Returns dict of {monomer: {atom_pair: peak_height}} for top interactions.
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.rdf import InterRDF

        u = mda.Universe(str(top_path), str(traj_path))
        protein = u.select_atoms("protein")
        n_frames = len(u.trajectory)
        start_frame = n_frames // 2

        # Key functional group atoms to track
        fg_atoms = {
            "OH": "name OH or name OG or name OG1",
            "NH": "name N or name NZ or name NH1 or name NH2",
            "CO": "name O or name OD1 or name OD2 or name OE1 or name OE2",
        }

        rdf_results = {}
        rdf_failures = []
        all_monomers = list(functional_monomers) + (
            [crosslinker] if crosslinker else [])
        # Protonation-variant residue names count as protein — see
        # _PH_PROTEIN_RESNAMES.
        non_prot = u.select_atoms(
            "not protein and not resname "
            + " ".join(sorted(_SOLVENT_IONS | _PH_PROTEIN_RESNAMES)))

        for m in all_monomers:
            # Select all atoms of this monomer type
            m_atoms = non_prot.select_atoms(f"resname {m} or resname UNL")
            if len(m_atoms) == 0:
                continue
            rdf_results[m] = {}
            for fg_name, fg_sel in fg_atoms.items():
                prot_fg = protein.select_atoms(fg_sel)
                if len(prot_fg) == 0:
                    continue
                try:
                    irdf = InterRDF(m_atoms, prot_fg,
                                    range=(0, 10), nbins=100)
                    irdf.run(start=start_frame, step=max(1, (n_frames - start_frame) // 50),
                             verbose=False)
                    peak = float(np.max(irdf.results.rdf))
                    peak_r = float(irdf.results.bins[np.argmax(irdf.results.rdf)])
                    if peak > 1.5:  # significant enrichment
                        rdf_results[m][fg_name] = {
                            "peak_g_r": round(peak, 2),
                            "peak_r_A": round(peak_r, 2),
                        }
                except Exception as e:
                    # Same class as REVIEW FINDING 19: a bare `pass` here made
                    # "no enrichment peak" and "the RDF crashed" identical in
                    # the output. RDF is informational and gates nothing, so
                    # this records the failure rather than aborting.
                    rdf_failures.append(f"{m}/{fg_name}: {type(e).__name__}: {e}")

        if rdf_failures:
            logger.error(f"    Per-atom RDF: {len(rdf_failures)} "
                         f"monomer/group pair(s) FAILED (absent from the "
                         f"results, not zero-enrichment): {rdf_failures[:5]}")
            rdf_results["_failures"] = rdf_failures
        logger.info(f"    Per-atom RDF: {len(rdf_results)} monomers analyzed")
        return rdf_results

    except Exception as e:
        # Raised from DEBUG: at INFO (the pipeline's level) a total RDF
        # failure produced an empty dict and no trace of why.
        logger.error(f"RDF analysis failed entirely: {type(e).__name__}: {e}")
        return {"_error": f"{type(e).__name__}: {e}"}


# ── Cross-Reactivity ──────────────────────────────────────────

def _run_cross_reactivity(md_results: dict, phase1_results: dict,
                           target_names: list, work_dir: Path,
                           time_ns: float = 20.0) -> dict:
    """Test top PC from each target against other epitopes."""
    xr_results = {}

    for source_target in target_names:
        source_data = md_results.get(source_target, {})
        if not source_data:
            continue

        best_pc_id = next(iter(source_data), None)
        if best_pc_id is None:
            continue

        best_pc = source_data[best_pc_id]
        functional = best_pc.get("functional_monomers", [])
        crosslinker = best_pc.get("crosslinker", "TEOS")

        for test_target in target_names:
            if test_target == source_target:
                continue

            key = f"{source_target}_PC_on_{test_target}"
            logger.info(f"  {key}")

            from .config import resolve_path
            test_epitope = resolve_path(phase1_results[test_target].get("head_pdb",
                                        phase1_results[test_target]["epitope_pdb"]))

            try:
                xr_result = _run_prepolymerization_md(
                    target=test_target,
                    pc_id=f"XR_{source_target}",
                    functional_monomers=functional,
                    crosslinker=crosslinker,
                    epitope_pdb=test_epitope,
                    work_dir=work_dir / key,
                    time_ns=time_ns,
                    total_monomers=10,  # fewer for cross-reactivity
                )
                xr_results[key] = {
                    "source": source_target,
                    "test": test_target,
                    "monomers": functional + [crosslinker],
                    "mmpbsa": xr_result.get("mmpbsa", {}),
                    "occupancy": xr_result.get("occupancy_analysis", {}).get("occupancy", {}),
                    "success": xr_result.get("success", False),
                }
            except Exception as e:
                xr_results[key] = {"error": str(e)}

    return xr_results


# ── Summary ───────────────────────────────────────────────────

def _print_phase4_summary(results: dict, target_names: list):
    """Print Phase 4 results summary."""
    logger.info(f"\n{'='*70}")
    logger.info("Phase 4: Pre-polymerization MD + Optimal Ratio Summary")
    logger.info(f"{'='*70}")

    n_rejected = 0
    for target in target_names:
        target_data = results.get(target, {})
        if not target_data:
            continue

        logger.info(f"\n[{target}]")
        for pc_id, data in target_data.items():
            if not isinstance(data, dict):
                continue
            status = "OK" if data.get("success") else "FAIL"
            monomers = data.get("functional_monomers", [])

            n_acc = data.get("n_replicas_accepted")
            n_rep = data.get("n_replicas")
            rep_str = (f" [{n_acc}/{n_rep} replicas accepted]"
                       if n_rep is not None else "")
            logger.info(f"  {pc_id} ({'+'.join(monomers)}): {status}{rep_str}")

            if not data.get("success"):
                n_rejected += 1
                for why in data.get("quality_failures", []):
                    logger.info(f"    ! {why}")

            # Optimal ratio
            ratio = data.get("optimal_ratio", {})
            if ratio:
                ratio_str = " : ".join(f"{m}={r}" for m, r in ratio.items())
                logger.info(f"    Optimal ratio: {ratio_str}")

            comp = data.get("box_composition", {})
            if comp:
                logger.info(f"    Box: {comp.get('polymerization_type')}, "
                            f"crosslinker "
                            f"{comp.get('crosslinker_mole_fraction_actual')} "
                            f"mol fraction")

            # MD metrics
            rmsd = data.get("rmsd_mean_nm", "N/A")
            hbond = data.get("hbond_mean", "N/A")
            dg = data.get("mmpbsa", {}).get("delta_total_kcal", "N/A")
            logger.info(f"    RMSD={rmsd}, H-bond={hbond}, ΔG={dg}")

    if n_rejected:
        logger.error(f"\n  PHASE 4 DID NOT PASS: {n_rejected} (target, pc_id) "
                     f"leg(s) have no accepted replica. See "
                     f"phase4_quality_report.json.")

    # Cross-reactivity
    xr = results.get("cross_reactivity", {})
    if xr:
        logger.info(f"\n  Cross-Reactivity:")
        for key, data in xr.items():
            dg = data.get("mmpbsa", {}).get("delta_total_kcal", "N/A")
            logger.info(f"    {key}: ΔG={dg}")

    logger.info(f"{'='*70}")


# ════════════════════════════════════════════════════════════════
# A5/B7: Solvent (Porogen) Sweep
# ════════════════════════════════════════════════════════════════

def run_phase4_solvent_sweep(target: str, pc_id: str,
                              functional_monomers: list, crosslinker: str,
                              epitope_pdb, work_dir, solvents: list = None,
                              time_ns: float = 30) -> dict:
    """A5/B7: Re-run Phase 4 pre-poly MD in multiple solvents (porogens).

    Real MIP polymerization is rarely in pure water. Sol-gel uses EtOH/H2O,
    free-radical uses ACN, CHCl3, etc. The same monomer set can give very
    different binding affinities depending on porogen.

    Reference: Cormack 2019 Chromatographia; MDPI Polymers 17:1057 (2025).
    """
    from .config import PHASE4_SOLVENTS, is_polymerization_compatible

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Determine compatible solvents from polymerization type
    combo = functional_monomers + [crosslinker]
    ok, dom, _ = is_polymerization_compatible(combo)
    if not ok:
        return {"error": "incompatible monomers — skip solvent sweep"}

    poly_type = "sol-gel" if dom == "silane" else "radical"
    if solvents is None:
        solvents = [s for s, cfg in PHASE4_SOLVENTS.items()
                     if poly_type in cfg.get("use_for", [])]
        if not solvents:
            solvents = ["water"]

    logger.info(f"\n=== Solvent sweep for {target}/{pc_id} ({poly_type}) ===")
    logger.info(f"  Testing solvents: {solvents}")

    sweep_results = {}
    for solv in solvents:
        solv_dir = work_dir / f"solvent_{solv}"
        solv_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n  → {solv} ({time_ns} ns)...")
        try:
            md_result = _run_prepolymerization_md(
                target=target, pc_id=f"{pc_id}_{solv}",
                functional_monomers=functional_monomers,
                crosslinker=crosslinker,
                epitope_pdb=epitope_pdb,
                work_dir=solv_dir,
                time_ns=time_ns,
            )
            sweep_results[solv] = {
                "dielectric": PHASE4_SOLVENTS.get(solv, {}).get("dielectric"),
                "contact_mean": (md_result.get("occupancy_analysis", {})
                                  .get("total_contact_mean")),
                "n_bound_monomers": (md_result.get("occupancy_analysis", {})
                                      .get("n_bound_monomers")),
                "accepted": md_result.get("accepted"),
                "result": md_result,
            }
        except Exception as e:
            logger.warning(f"  {solv} failed: {e}")
            sweep_results[solv] = {"error": str(e)}

    # Rank by imprinting affinity (more contacts = better imprinting).
    # Only ACCEPTED legs may win the ranking.
    valid = {s: r for s, r in sweep_results.items()
             if "error" not in r and r.get("contact_mean") is not None
             and r.get("accepted")}
    if valid:
        best_solvent = max(valid, key=lambda s: valid[s]["contact_mean"])
    else:
        best_solvent = None
        logger.error("  Solvent sweep: no solvent leg passed acceptance — "
                     "no best_solvent selected")
    logger.info(f"  Best solvent: {best_solvent}")

    return {
        "polymerization_type": poly_type,
        "solvents_tested": list(sweep_results.keys()),
        "best_solvent": best_solvent,
        "per_solvent": sweep_results,
    }



# ════════════════════════════════════════════════════════════════
# B6 / BSA: EXPLICIT MONOMER RATIO GRID
# ════════════════════════════════════════════════════════════════
#
# WHAT CHANGED (2026-08 audit follow-up).  # BEHAVIOUR CHANGE
# The old run_phase4_ratio_sweep() computed a `copies` dict, logged it, and then
# called _run_prepolymerization_md() WITHOUT it — that function had no copies
# parameter and always re-derived a uniform composition internally.  Every
# "ratio" therefore built the IDENTICAL box.  It additionally hardcoded
# total=20 (ignoring EPITOPE_MONOMER_MOLAR_RATIO), truncated every preset to
# n_functional elements (so a one-functional-monomer system collapsed all
# presets onto one point), ran a single replica with no pooling and therefore no
# SD, and ranked on total_contact_mean — the raw SUM over all monomer types,
# which is mechanically confounded with composition.
#
# The driver below imposes the copy numbers, runs PHASE4_N_REPLICAS replicas per
# point through _pool_replicas (so every point gets the pooled SD), verifies the
# built topology against the request, writes each point's result to its own JSON
# the moment it finishes, and skips completed points on a re-run.

_GRID_INDEX_NAME = "ratio_grid_index.json"
_GRID_POINT_NAME = "point_result.json"


def _grid_point_label(copies: dict, order: list) -> str:
    """Filesystem-safe, composition-derived, order-stable point label."""
    return "_".join(f"{m}{int(copies.get(m, 0))}" for m in order)


def _ratio_grid_from_config() -> dict:
    """The ratio grid this experiment declares, or None.

    Reads the BSA keys BY NAME (BSA_RATIO_GRID, BSA_RATIO_TOTAL_MONOMERS,
    BSA_RATIO_SWEEP_MD_NS, BSA_MAX_MONOMERS_IN_SHELL, BSA_RATIO_GRID_PROVISIONAL)
    so that what the config declares is literally what the engine consumes.
    PHASE4_RATIO_GRID / PHASE4_RATIO_GRID_* are the experiment-neutral aliases;
    they win if both are set, so a non-BSA experiment can use this driver
    without inheriting a BSA-shaped name.
    """
    grid = _cfg_get("PHASE4_RATIO_GRID") or _cfg_get("BSA_RATIO_GRID")
    if not grid:
        return None
    total = (_cfg_get("PHASE4_RATIO_GRID_TOTAL_MONOMERS")
             or _cfg_get("BSA_RATIO_TOTAL_MONOMERS"))
    ns = (_cfg_get("PHASE4_RATIO_GRID_MD_NS")
          or _cfg_get("BSA_RATIO_SWEEP_MD_NS"))
    cap = (_cfg_get("PHASE4_RATIO_GRID_MAX_MONOMERS")
           or _cfg_get("BSA_MAX_MONOMERS_IN_SHELL"))
    return {
        "grid": [tuple(int(v) for v in pt) for pt in grid],
        "total_monomers": None if total is None else int(total),
        "time_ns": None if ns is None else float(ns),
        "max_monomers_in_shell": None if cap is None else int(cap),
        "provisional": bool(_cfg_get("PHASE4_RATIO_GRID_PROVISIONAL",
                                     _cfg_get("BSA_RATIO_GRID_PROVISIONAL",
                                              False))),
        # Env-overridable: the staged design (screen at R=4, top the survivors
        # up to R=12) changes only this number between stages, and requiring a
        # config edit between two runs of the same grid is how the wrong R ends
        # up recorded against a leg.
        "n_replicas": (_env_int("PHASE4_RATIO_GRID_N_REPLICAS")
                       or _cfg_get("PHASE4_RATIO_GRID_N_REPLICAS")
                       or _cfg_get("BSA_RATIO_GRID_N_REPLICAS")),
        "loading_sweep": list(_cfg_get("PHASE4_LOADING_SWEEP")
                              or _cfg_get("BSA_LOADING_SWEEP") or []),
        "source_keys": [k for k in ("PHASE4_RATIO_GRID", "BSA_RATIO_GRID")
                        if _cfg_get(k)],
    }


def _grid_points_to_copies(grid: list, crosslinker: str,
                           functional_monomers: list,
                           max_monomers: int = None) -> list:
    """Normalise declared grid entries into explicit copy-number dicts.

    Accepted entry shapes:
      * dict            {"TEOS": 48, "APTES": 12}          — fully explicit
      * (n_xl, n_func)  (48, 12) with ONE functional monomer — the BSA form,
                        ordered (crosslinker, functional) to match
                        BSA_RATIO_GRID's documented "(n_TEOS, n_APTES)".
      * sequence        one entry per monomer in [crosslinker] + functional.
    A point that exceeds `max_monomers` is REFUSED here rather than discovered
    as a 250 nm box at editconf time.
    """
    order = ([crosslinker] if crosslinker else []) + list(functional_monomers)
    out = []
    for i, pt in enumerate(grid):
        if isinstance(pt, dict):
            copies = {m: int(pt.get(m, 0)) for m in order}
            unknown = sorted(set(pt) - set(order))
            if unknown:
                raise ValueError(
                    f"ratio grid point {i} names {unknown}, which are not in "
                    f"this composition {order}")
        else:
            vals = list(pt)
            if len(vals) != len(order):
                raise ValueError(
                    f"ratio grid point {i} = {pt} has {len(vals)} entries but "
                    f"the composition has {len(order)} monomers {order}. "
                    f"Declare it as a dict to be unambiguous.")
            copies = {m: int(v) for m, v in zip(order, vals)}
        if any(v < 0 for v in copies.values()):
            raise ValueError(f"ratio grid point {i} = {copies} has a negative "
                             f"copy count")
        n_total = sum(copies.values())
        if n_total == 0:
            raise ValueError(f"ratio grid point {i} = {copies} is an empty box")
        if max_monomers is not None and n_total > max_monomers:
            raise ValueError(
                f"ratio grid point {i} = {copies} asks for {n_total} monomers, "
                f"above the shell placement cap of {max_monomers} "
                f"(BSA_MAX_MONOMERS_IN_SHELL). The placer would fall back to "
                f"r_outer + 0.3*i nm and editconf would build an enormous box. "
                f"Raise the cap deliberately or lower the point.")
        out.append({"index": i, "copies": copies,
                    "label": _grid_point_label(copies, order),
                    "n_total": n_total,
                    "x_functional": round(
                        sum(copies[m] for m in functional_monomers) / n_total, 4)})
    labels = [p["label"] for p in out]
    dupes = sorted({l for l in labels if labels.count(l) > 1})
    if dupes:
        raise ValueError(f"ratio grid contains duplicate compositions {dupes}")
    return out


def _grid_point_metrics(pooled: dict, copies: dict,
                        functional_monomers: list, crosslinker: str) -> dict:
    """Ranking-relevant numbers for one grid point, all normalised per molecule.

    A RAW contact/H-bond count is monotone in the number of copies of the
    monomer that makes the contact, so ranking grid points on it measures the
    composition back to itself.  Every quantity here is therefore divided by the
    copy number of the species that produced it, and the raw values are kept
    beside them so the ranking can be redone offline without re-running MD.
    """
    occ = pooled.get("occupancy_analysis") or {}
    cf = occ.get("contact_freq_6A") or {}
    cf_sd = occ.get("contact_freq_6A_sd") or {}
    hb = occ.get("HBNMax") or {}

    def _per_copy(d):
        return {m: (round(float(v) / copies[m], 4)
                    if copies.get(m) and isinstance(v, (int, float)) else None)
                for m, v in (d or {}).items()}

    n_func = sum(copies.get(m, 0) for m in functional_monomers)
    func_contacts = sum(float(cf.get(m) or 0.0) for m in functional_monomers)

    # PER-REPLICA values of the ranking metric. These — and ONLY these — are
    # the independent samples this point has. The 4 within-window block means
    # are NOT: measured, their SEM understates the true window-to-window SD by
    # a median 2.8x because the blocks sit inside one correlation time.
    per_replica = []
    for r in (pooled.get("replicas") or []):
        if not r.get("accepted"):
            continue
        rcf = r.get("contact_freq_6A") or {}
        if not rcf:
            continue
        v = sum(float(rcf.get(m) or 0.0) for m in functional_monomers)
        if n_func:
            per_replica.append(round(v / n_func, 6))
    ci = _mean_ci(per_replica, float(_cfg("PHASE4_RANK_CI_LEVEL")))

    # Sampling health, carried up from the representative replica so a grid
    # summary shows at a glance whether the legs were long enough to rank.
    samp = {}
    for r in (pooled.get("replicas") or []):
        if r.get("sampling"):
            samp = r["sampling"]
            break

    return {
        "contact_freq_6A": cf,
        "contact_freq_6A_sd": cf_sd,
        "contact_freq_per_copy": _per_copy(cf),
        "HBNMax": hb,
        "HBNMax_per_copy": _per_copy(hb),
        "residence_ns": occ.get("residence_ns"),
        "proximity_fraction": occ.get("proximity_fraction"),
        "EBN": occ.get("EBN"),
        # PRIMARY RANKING METRIC. Contacts made by the functional monomers,
        # per functional molecule present. Normalising removes the trivial
        # "more APTES -> more APTES contacts" scaling that makes a raw count
        # rise monotonically to the all-APTES endpoint.
        "functional_contact_per_copy": (round(func_contacts / n_func, 4)
                                        if n_func else None),
        # The replicate-level sample behind that mean, and its Student-t CI.
        # The tie rule and the loading-sweep gate read THESE, not the mean.
        "functional_contact_per_copy_replicas": per_replica,
        "functional_contact_per_copy_ci": ci,
        "total_contact_mean": occ.get("total_contact_mean"),
        "n_replicas_pooled": occ.get("n_replicas_pooled"),
        "converged": ((occ.get("convergence") or {}).get("converged")),
        "monomers_without_contacts": (
            (occ.get("convergence") or {}).get("monomers_without_contacts")),
        "sampling": samp,
    }


# ── Ranking, ties and trend ───────────────────────────────────

def _rank_grid_points(per_point: dict, rank_metric: str,
                      x_key: str = "x_functional") -> dict:
    """Rank complete+accepted points, and say honestly whether the top is real.

    THREE THINGS THIS DOES THAT `sorted()` DID NOT.

    1. TIE BY CI, NOT BY FLOAT EQUALITY.  The retired guard was
       `v == ranked[0][0]` on values rounded to 4 dp.  Simulated against the
       pipeline's exact arithmetic under a completely FLAT truth (p=0.15,
       CV_1=1.8, 20,000 trials) it fired 0.11% of the time at R=1 and 0.14% at
       R=3 — so a flat sweep announced a unique "best point" in 99.9% of runs,
       with a median winning margin of 18%.  A tie is now declared whenever the
       runner-up's mean lies inside the top point's replicate CI, or whenever
       the relative margin is below the pre-registered MDD.

    2. TREND, NOT ONLY ARGMAX.  argmax names a winner in 100% of runs whatever
       the truth is.  A regression of the metric on the composition axis has a
       calibrated false-positive rate and is reported beside the argmax.

    3. IT REFUSES TO RANK ON n=1.  With one replica there is no error bar, so
       no tie rule can run; the grid says so instead of implying precision.
    """
    rule = str(_cfg("PHASE4_RANK_TIE_RULE")).lower()
    mdd_pct = float(_cfg("PHASE4_RANK_MDD_PCT"))
    ranked, unrankable = [], []
    for label, e in per_point.items():
        if e.get("status") != "done" or not e.get("accepted"):
            continue
        v = (e.get("metrics") or {}).get(rank_metric)
        if isinstance(v, (int, float)):
            ranked.append((float(v), label))
        else:
            # Not a loss — a crosslinker-only control HAS no
            # functional_contact_per_copy. It is a baseline for the observable,
            # not a candidate composition.
            unrankable.append(label)
    ranked.sort(reverse=True)

    out = {
        "rank_metric": rank_metric,
        "tie_rule": rule,
        "mdd_pct": mdd_pct,
        "ranking": [{"label": l, rank_metric: v,
                     "ci": ((per_point[l].get("metrics") or {})
                            .get(f"{rank_metric}_ci")),
                     "replicas": ((per_point[l].get("metrics") or {})
                                  .get(f"{rank_metric}_replicas"))}
                    for v, l in ranked],
        "unrankable_points": unrankable,
        "best_point": ranked[0][1] if ranked else None,
        "best_point_is_tied": False,
        "tied_points": [],
        "best_point_qualified": False,
        "margin_over_runner_up_pct": None,
        "decision": "no_rank",
        "decision_reason": None,
    }
    if not ranked:
        out["decision_reason"] = "no grid point is both complete and accepted"
        return out
    if len(ranked) == 1:
        out["decision"] = "single_point"
        out["decision_reason"] = "only one point is complete and accepted"
        return out

    top_v, top_l = ranked[0]
    second_v, _ = ranked[1]
    margin = ((top_v - second_v) / top_v * 100.0) if top_v > 0 else 0.0
    out["margin_over_runner_up_pct"] = round(margin, 2)
    top_ci = ((per_point[top_l].get("metrics") or {})
              .get(f"{rank_metric}_ci")) or {}

    if rule == "exact":
        tied = [l for v, l in ranked if v == top_v]
    elif rule == "mdd":
        tied = [l for v, l in ranked if top_v <= 0 or
                (top_v - v) / top_v * 100.0 < mdd_pct]
    else:  # "ci" — the default
        top_reps = ((per_point[top_l].get("metrics") or {})
                    .get(f"{rank_metric}_replicas")) or []
        if len(top_reps) < 2:
            # n=1, or a degenerate sample. There is no error bar, so nothing can
            # be separated: every point is tied with the top one.
            tied = [l for _, l in ranked]
            out["decision_reason"] = (
                f"the top point has {len(top_reps)} replicate value(s); one "
                f"replica cannot rank compositions")
        else:
            # A point is TIED with the top unless a two-sample Welch test
            # separates it AND the margin clears the pre-registered MDD. The
            # MDD is a floor under the test: a statistically clean 4%
            # difference is still not a difference worth choosing a recipe on.
            tied, pvals = [], {}
            for v, l in ranked:
                if l == top_l:
                    tied.append(l)
                    continue
                reps = ((per_point[l].get("metrics") or {})
                        .get(f"{rank_metric}_replicas")) or []
                p = _welch_p(top_reps, reps)
                pvals[l] = round(p, 5)
                margin_l = ((top_v - v) / top_v * 100.0) if top_v > 0 else 0.0
                if not (p < (1.0 - float(_cfg("PHASE4_RANK_CI_LEVEL")))
                        and margin_l >= mdd_pct):
                    tied.append(l)
            out["welch_p_vs_best"] = pvals
    # dedupe, keep rank order
    seen, tied_ordered = set(), []
    for _, l in ranked:
        if l in tied and l not in seen:
            seen.add(l)
            tied_ordered.append(l)
    out["best_point_is_tied"] = len(tied_ordered) > 1
    out["tied_points"] = tied_ordered if len(tied_ordered) > 1 else []
    # QUALIFICATION IS ABOUT THE RUNNER-UP, not about every point on the grid.
    # "The winner is identified" means the top point is separated from the
    # SECOND-best by a two-sample test and by more than the pre-registered MDD.
    # A lower-mean but noisier point further down the ranking staying inside
    # `tied_points` is a caveat worth reporting — it is not a reason to withhold
    # a winner that beat its nearest rival cleanly.
    _p_runner = (out.get("welch_p_vs_best") or {}).get(
        ranked[1][1], 1.0) if rule == "ci" else (
            0.0 if margin >= mdd_pct else 1.0)
    out["welch_p_vs_runner_up"] = _p_runner
    out["best_point_qualified"] = bool(
        top_ci.get("n", 0) >= 2
        and margin >= mdd_pct
        and _p_runner < (1.0 - float(_cfg("PHASE4_RANK_CI_LEVEL"))))

    # Trend across the composition axis, with a calibrated false-positive rate.
    xs, ys = [], []
    for v, l in ranked:
        x = per_point[l].get(x_key)
        if isinstance(x, (int, float)):
            xs.append(float(x))
            ys.append(v)
    if len(set(xs)) >= 3:
        try:
            from scipy import stats as _st
            lr = _st.linregress(xs, ys)
            out["trend"] = {"x": x_key, "n_points": len(xs),
                            "slope": round(float(lr.slope), 6),
                            "r": round(float(lr.rvalue), 4),
                            "p_value": round(float(lr.pvalue), 5),
                            "significant_at_0.05": bool(lr.pvalue < 0.05)}
        except Exception as e:
            out["trend"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        out["trend"] = {"error": f"only {len(set(xs))} distinct {x_key} values"}

    if out["best_point_qualified"]:
        out["decision"] = "optimum"
        out["decision_reason"] = (
            f"{top_l} beats the runner-up by {margin:.1f}% (>= MDD {mdd_pct}%) "
            f"and every other point is separated from it by a two-sample Welch "
            f"test at alpha={1 - float(_cfg('PHASE4_RANK_CI_LEVEL')):.2f} "
            f"(p vs runner-up = {(out.get('welch_p_vs_best') or {}).get(ranked[1][1])})")
    elif (out.get("trend") or {}).get("significant_at_0.05"):
        out["decision"] = "trend_only"
        out["decision_reason"] = (
            f"no single point is separated (top margin {margin:.1f}%, tied with "
            f"{out['tied_points']}), but the metric trends with {x_key} "
            f"(slope {out['trend']['slope']}, p={out['trend']['p_value']}). "
            f"Read the TREND, not the argmax.")
    else:
        out["decision"] = "flat"
        out["decision_reason"] = out["decision_reason"] or (
            f"the top point is not separated from {out['tied_points']} "
            f"(margin {margin:.1f}% < MDD {mdd_pct}%, or the runner-up lies "
            f"inside its CI) and no trend is significant. A FLAT SWEEP IS A "
            f"RESULT — it is not an optimum.")
    return out


def run_phase4_ratio_grid(target: str, pc_id: str,
                          functional_monomers: list, crosslinker: str,
                          epitope_pdb, work_dir,
                          grid: list = None,
                          total_monomers: int = None,
                          time_ns: float = None,
                          n_replicas: int = None,
                          max_monomers_in_shell: int = None,
                          rank_metric: str = "functional_contact_per_copy",
                          provisional: bool = False) -> dict:
    """Run one MD leg per declared (crosslinker, functional) composition.

    RESUMABILITY.  Each point owns a directory and writes `point_result.json`
    the instant its replicas are pooled; `ratio_grid_index.json` is rewritten
    after every point.  A re-run reads those files back and skips completed
    points, so a crash at point 5 keeps points 1-4.  A point that RAISES is
    recorded with its traceback and the grid continues — one bad composition
    must not cost the other seven — but it is counted in `n_failed` and the
    grid reports success=False, so a partial grid can never be mistaken for a
    complete one.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = _ratio_grid_from_config() or {}
    if grid is None:
        grid = cfg.get("grid")
    if not grid:
        raise ValueError(
            "run_phase4_ratio_grid called with no grid, and neither "
            "PHASE4_RATIO_GRID nor BSA_RATIO_GRID is declared in the active "
            "config")
    if time_ns is None:
        time_ns = cfg.get("time_ns") or float(_cfg_get("MD_PRODUCTION_NS", 30))
    if max_monomers_in_shell is None:
        max_monomers_in_shell = cfg.get("max_monomers_in_shell")
    if n_replicas is None:
        n_replicas = cfg.get("n_replicas") or int(_cfg("PHASE4_N_REPLICAS"))
    n_replicas = max(1, int(n_replicas))
    provisional = bool(provisional or cfg.get("provisional"))

    points = _grid_points_to_copies(grid, crosslinker, functional_monomers,
                                    max_monomers_in_shell)

    # BSA_RATIO_TOTAL_MONOMERS is an ASSERTION on the primary sweep, not a
    # second source of truth for n_total. Every declared point already carries
    # its own copy numbers; if the grid is edited so a point no longer sums to
    # the declared total, that point silently runs at a different silane
    # molarity than the rest and the sweep is no longer a pure composition axis.
    # (On the loading axis `total_monomers` is passed explicitly and the
    # assertion is deliberately skipped — rescaling IS the point there.)
    _declared_total = cfg.get("total_monomers")
    if total_monomers is None and _declared_total:
        off = [(p["label"], p["n_total"]) for p in points
               if p["n_total"] != _declared_total]
        if off:
            raise ValueError(
                f"ratio grid point(s) {off} do not sum to the declared "
                f"BSA_RATIO_TOTAL_MONOMERS = {_declared_total}. The x-sweep is "
                f"meant to vary COMPOSITION at fixed loading; a point at a "
                f"different n_total also changes the silane concentration, so "
                f"its difference cannot be attributed to the ratio. Fix the "
                f"grid, or pass total_monomers= to rescale every point on "
                f"purpose (that is the loading axis).")

    # `total_monomers` is the SECONDARY (loading) axis. When given it rescales
    # every point to that box size while holding the composition fixed, which
    # is what BSA_LOADING_SWEEP means. When absent each point keeps its own
    # declared n_total.
    if total_monomers is not None:
        rescaled = []
        for p in points:
            n_now = p["n_total"]
            scaled = {m: int(round(n * total_monomers / n_now))
                      for m, n in p["copies"].items()}
            # Preserve zeros exactly (a control point must stay a control
            # point) and repair the rounding residue on the largest species.
            for m, n in p["copies"].items():
                if n == 0:
                    scaled[m] = 0
            drift = total_monomers - sum(scaled.values())
            if drift:
                biggest = max((m for m in scaled if scaled[m] > 0),
                              key=lambda m: scaled[m])
                scaled[biggest] += drift
            if any(v < 0 for v in scaled.values()):
                raise ValueError(
                    f"rescaling point {p['label']} to n_total={total_monomers} "
                    f"produced {scaled}")
            order = ([crosslinker] if crosslinker else []) + list(functional_monomers)
            rescaled.append({**p, "copies": scaled,
                             "n_total": sum(scaled.values()),
                             "label": _grid_point_label(scaled, order)})
        points = rescaled

    grid_fp = _grid_points_fingerprint(points)

    # PRE-FLIGHT 1: what protein charge state will actually be simulated?
    # Raises unless the fixed-charge shortfall at MD_SOLVENT_PH has been
    # acknowledged in config.
    ph_status = _ph_preflight(epitope_pdb)

    # PRE-FLIGHT, before a single ns is spent: does the configured MM-GBSA
    # window fit inside a leg of THIS length? MD_MMPBSA_START_NS/END_NS are
    # re-derived by hand for MD_PRODUCTION_NS, so a sweep at any other leg
    # length used to kill MM-GBSA on every point and only say so afterwards.
    # PRE-FLIGHT 2: does the configured MM-GBSA window fit this leg length?
    mmpbsa_status = _mmpbsa_window_status(time_ns)
    if mmpbsa_status.get("fits") is False:
        logger.error("  MM-GBSA PRE-FLIGHT FAILED: %s", mmpbsa_status["reason"])
    elif mmpbsa_status.get("fits"):
        logger.info("  MM-GBSA window %.3g-%.3g ns fits inside a %g ns leg",
                    mmpbsa_status["mmpbsa_start_ns"],
                    mmpbsa_status["mmpbsa_end_ns"], time_ns)

    logger.info(
        f"\n  RATIO GRID for {target}/{pc_id}: {len(points)} composition(s) x "
        f"{n_replicas} replica(s) x {time_ns:g} ns = "
        f"{len(points) * n_replicas * time_ns:g} ns of production MD "
        f"[fingerprint {grid_fp}]")
    if n_replicas < 2:
        logger.error(
            "  n_replicas=1: THIS GRID CANNOT RANK. One trajectory per point "
            "has no error bar, so the tie rule cannot separate any two points "
            "and every point will be reported as tied. Use it to shake the "
            "pipeline out, never to choose a ratio. "
            "MEASURED basis: one 350 ns leg re-analysed in 50 non-overlapping "
            "7 ns windows gave 19 distinct rankings of its 4 monomers.")
    if provisional:
        logger.warning(
            "  BSA_RATIO_GRID_PROVISIONAL is True — this grid is centred on the "
            "protein-imprinted sol-gel LITERATURE, not on the user's measured "
            "TEOS/APTES volumes. Its optimum is an optimum of a literature grid "
            "until BSA_PROTOCOL_MEASURED is filled in and the grid re-centred.")
    for p in points:
        logger.info(f"    [{p['index']}] {p['label']}  n_total={p['n_total']} "
                    f"x_functional={p['x_functional']}")

    index_path = work_dir / _GRID_INDEX_NAME
    per_point = {}

    def _write_index(done_state):
        payload = {
            "schema": "phase4/ratio-grid/v2",
            "target": target,
            "pc_id": pc_id,
            "functional_monomers": list(functional_monomers),
            "crosslinker": crosslinker,
            "time_ns": time_ns,
            "mmpbsa_window": mmpbsa_status,
            "ph_preflight": ph_status,
            "n_replicas": n_replicas,
            "grid_fingerprint": grid_fp,
            "grid_source_keys": cfg.get("source_keys"),
            "provisional": provisional,
            "rank_metric": rank_metric,
            "points": [
                {"index": p["index"], "label": p["label"],
                 "copies": p["copies"], "n_total": p["n_total"],
                 "x_functional": p["x_functional"],
                 "dir": str(work_dir / p["label"]),
                 "result_json": str(work_dir / p["label"] / _GRID_POINT_NAME),
                 "status": done_state.get(p["label"], "pending")}
                for p in points],
        }
        index_path.write_text(json.dumps(payload, indent=2, default=str))

    state = {}
    _write_index(state)

    n_failed = 0
    for p in points:
        label, copies = p["label"], p["copies"]
        pdir = work_dir / label
        pdir.mkdir(parents=True, exist_ok=True)
        point_json = pdir / _GRID_POINT_NAME

        # ── RESUME: a completed point is read back, never recomputed. ──
        if point_json.exists():
            try:
                prev = json.loads(point_json.read_text())
            except Exception as e:
                logger.error(f"  Grid point {label}: {point_json} is unreadable "
                             f"({e}) — recomputing")
                prev = None
            if prev and prev.get("copies") == copies:
                _prev_R = int(prev.get("n_replicas") or 0)
                # `status` is set to "done" whenever pooling returns without
                # RAISING — including the case where every replica was rejected
                # and md_completed is false for all of them, which is exactly
                # what an aborted mdrun leaves behind. Such a point holds no
                # trajectory at all, so resuming it would freeze an EMPTY sweep
                # into place and report it as complete. Only treat the replica
                # records as proof of absence when they are actually present:
                # if the file predates them we cannot tell, so keep the old
                # behaviour rather than recomputing good points.
                # Only records that actually CARRY the key are evidence. A
                # replica dict without `md_completed` is UNKNOWN, not known-bad;
                # treating absence as failure would retry — and so re-run from
                # scratch — every point written by a producer that omits it,
                # which silently doubles the cost of the staged R=1 -> R=4
                # top-up this design depends on.
                _prev_reps = ((prev.get("pooled") or {}).get("replicas") or [])
                _known = [r for r in _prev_reps if "md_completed" in r]
                _no_data = (bool(_known)
                            and not any(r.get("md_completed") for r in _known))
                if prev.get("status") == "done" and _no_data:
                    logger.warning(
                        f"  Grid point {label}: recorded status='done' but NO "
                        f"replica finished its MD (md_completed=false for all "
                        f"{len(_prev_reps)}) — the point holds no trajectory. "
                        f"RETRYING rather than resuming an empty result.")
                elif prev.get("status") == "done" and _prev_R >= n_replicas:
                    logger.info(f"  Grid point {label}: RESUMED from {point_json}")
                    per_point[label] = prev
                    state[label] = "done"
                    _write_index(state)
                    continue
                elif prev.get("status") == "done":
                    # STAGED DESIGN. Asking for more replicas than the recorded
                    # run EXTENDS the point instead of resuming it. Replica
                    # seeds are sha256(target|pc_id@label|rep<i>), so reps
                    # 0..prev-1 keep their seeds, their manifests still match
                    # and `_check_resume` reuses their trajectories; only the
                    # new reps cost GPU time. This is what makes "screen at
                    # R=4, then top up the survivors to R=10" affordable.
                    logger.info(
                        f"  Grid point {label}: recorded {_prev_R} replica(s), "
                        f"{n_replicas} now requested — EXTENDING (reps "
                        f"0-{_prev_R - 1} will be reused from disk)")
                else:
                    # A point that FAILED is retried, not resumed. Resuming it
                    # would make a transient failure (a full disk, a killed
                    # mdrun) permanent and silently shrink the grid on every
                    # subsequent run.
                    logger.warning(
                        f"  Grid point {label}: previous attempt recorded "
                        f"status={prev.get('status')!r} ({prev.get('error')}) — "
                        f"RETRYING")
            elif prev:
                raise RuntimeError(
                    f"{point_json} records copies {prev.get('copies')} but this "
                    f"grid point asks for {copies}. The grid was edited between "
                    f"runs and the two cannot be pooled. Move or delete {pdir}.")

        logger.info(f"\n  ── GRID POINT {p['index'] + 1}/{len(points)}: {label} "
                    f"({', '.join(f'{m}x{n}' for m, n in copies.items() if n)}) ──")
        entry = {
            "index": p["index"], "label": label, "copies": copies,
            "n_total": p["n_total"], "x_functional": p["x_functional"],
            "target": target, "pc_id": pc_id, "time_ns": time_ns,
            "n_replicas": n_replicas, "dir": str(pdir),
        }
        try:
            replica_results = []
            for rep in range(n_replicas):
                replica_results.append(_run_prepolymerization_md(
                    target=target,
                    pc_id=f"{pc_id}@{label}",
                    functional_monomers=functional_monomers,
                    crosslinker=crosslinker,
                    epitope_pdb=epitope_pdb,
                    work_dir=_replica_dir(pdir, rep),
                    time_ns=time_ns,
                    copies=copies,
                    replica_index=rep,
                    n_replicas=n_replicas,
                    seed_label=f"{pc_id}@{label}",
                ))
            pooled = _pool_replicas(target, f"{pc_id}@{label}",
                                    functional_monomers, crosslinker,
                                    replica_results, time_ns, pdir, work_dir)
            _eff = sorted({r.get("effective_time_ns")
                           for r in (pooled.get("replicas") or [])
                           if isinstance(r.get("effective_time_ns"), (int, float))})
            entry.update({
                "status": "done",
                "accepted": bool(pooled.get("accepted")),
                "n_replicas_accepted": pooled.get("n_replicas_accepted"),
                # The length the legs of this point ACTUALLY ran, beside the
                # requested `time_ns` already in `entry`. A quick-mode override
                # can no longer hide behind the requested number.
                "effective_time_ns": (_eff[0] if len(_eff) == 1 else _eff or None),
                "built_copies": pooled.get("pooled_composition"),
                # THE METRIC IS KEYED ON THE BOX, NOT ON THE ARGUMENT LIST.
                # With pH speciation the box holds APTESH where the caller said
                # APTES, so summing contact_freq over the nominal
                # `functional_monomers` returned 0.0 for a point whose ammonium
                # was in contact in every frame. `pooled` carries the species
                # that were actually present and the composition actually built.
                "metrics": _grid_point_metrics(
                    pooled,
                    pooled.get("pooled_composition") or copies,
                    pooled.get("functional_monomers") or functional_monomers,
                    pooled.get("crosslinker", crosslinker)),
                "optimal_ratio": pooled.get("optimal_ratio"),
                "pooled": pooled,
            })
        except Exception as e:
            import traceback
            n_failed += 1
            logger.error(f"  GRID POINT {label} FAILED: {type(e).__name__}: {e}")
            entry.update({"status": "failed", "accepted": False,
                          "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()})

        # Written IMMEDIATELY: a crash at point k+1 must not cost point k.
        point_json.write_text(json.dumps(entry, indent=2, default=str))
        per_point[label] = entry
        state[label] = entry["status"]
        _write_index(state)
        logger.info(f"  Grid point {label} → {entry['status']} "
                    f"(saved {point_json})")

    # ── Ranking ──
    rank = _rank_grid_points(per_point, rank_metric)
    ranked = [(e[rank_metric], e["label"]) for e in rank["ranking"]]
    unrankable = rank["unrankable_points"]
    best = rank["best_point"]
    tied = rank["tied_points"]
    n_done = sum(1 for e in per_point.values() if e.get("status") == "done")
    n_accepted = sum(1 for e in per_point.values() if e.get("accepted"))

    if best is None:
        logger.error(
            f"  RATIO GRID: no grid point is both complete and accepted "
            f"({n_done}/{len(points)} complete, {n_accepted} accepted, "
            f"{n_failed} failed) — NO optimum is reported. A grid that does not "
            f"rank is a result; a grid ranked on rejected legs is not.")
    elif rank["decision"] == "optimum":
        logger.info(f"  RATIO GRID OPTIMUM: {best} ({rank_metric}="
                    f"{ranked[0][0]}) — {rank['decision_reason']}")
    elif rank["decision"] == "trend_only":
        logger.error(f"  RATIO GRID DID NOT SEPARATE A SINGLE POINT: "
                     f"{rank['decision_reason']}")
    elif rank["decision"] == "single_point":
        logger.info(f"  RATIO GRID: one point only ({best}) — nothing to rank "
                    f"against. {rank['decision_reason']}")
    else:
        logger.error(
            f"  RATIO GRID DID NOT DISCRIMINATE: {best!r} is top by "
            f"{rank['margin_over_runner_up_pct']}% but ties with {tied} under "
            f"the {rank['tie_rule']!r} rule (MDD {rank['mdd_pct']}%). "
            f"{rank['decision_reason']} Do NOT read best_point as an optimum.")
    if unrankable:
        logger.info(f"  Complete but not rankable on {rank_metric} "
                    f"(control points with no functional monomer): {unrankable}")

    # Sampling health of the whole grid, so an under-sampled sweep is visible
    # in one line of the summary rather than buried per replica.
    _neffs = [((e.get("metrics") or {}).get("sampling") or {})
              .get("min_n_independent_samples") for e in per_point.values()]
    _neffs = [v for v in _neffs if isinstance(v, (int, float))]
    _unstable = [l for l, e in per_point.items()
                 if ((e.get("metrics") or {}).get("sampling") or {})
                 .get("sub_window_ranking_stable") is False]

    summary = {
        "schema": "phase4/ratio-grid/v2",
        "target": target,
        "pc_id": pc_id,
        "functional_monomers": list(functional_monomers),
        "crosslinker": crosslinker,
        "time_ns": time_ns,
        "mmpbsa_window": mmpbsa_status,
        "ph_preflight": ph_status,
        "n_replicas": n_replicas,
        "n_points": len(points),
        "n_points_complete": n_done,
        "n_points_accepted": n_accepted,
        "n_points_failed": n_failed,
        "provisional_grid": provisional,
        "grid_fingerprint": grid_fp,
        "effective_sampling_config": _effective_sampling_config(),
        "grid_points_declared": [
            {"label": p["label"], "copies": p["copies"]} for p in points],
        "rank_metric": rank_metric,
        "ranking": rank["ranking"],
        "unrankable_points": unrankable,
        "best_point": best,
        "best_copies": per_point[best]["copies"] if best else None,
        # True when the top point is NOT separated from the runner-up by the
        # active tie rule. The caller MUST NOT read best_point as an optimum in
        # that case — see the ERROR logged above.
        "best_point_is_tied": rank["best_point_is_tied"],
        "tied_points": tied,
        # QUALIFIED = separated from the runner-up by more than the
        # pre-registered MDD *and* by a replicate CI. Only a qualified winner
        # may condition the loading axis.
        "best_point_qualified": rank["best_point_qualified"],
        "margin_over_runner_up_pct": rank["margin_over_runner_up_pct"],
        "tie_rule": rank["tie_rule"],
        "mdd_pct": rank["mdd_pct"],
        "decision": rank["decision"],
        "decision_reason": rank["decision_reason"],
        "trend": rank.get("trend"),
        "sampling_health": {
            "min_n_independent_samples_over_points": (
                min(_neffs) if _neffs else None),
            "points_with_unstable_sub_window_ranking": _unstable,
        },
        "index_json": str(index_path),
        "per_point": per_point,
        "success": bool(best) and n_failed == 0 and n_done == len(points),
    }
    (work_dir / "ratio_grid_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    _write_index(state)
    return summary


def run_phase4_loading_sweep(target: str, pc_id: str,
                             functional_monomers: list, crosslinker: str,
                             epitope_pdb, work_dir,
                             winning_copies: dict,
                             loadings: list = None,
                             time_ns: float = None,
                             n_replicas: int = None,
                             composition_source: str = None) -> dict:
    """SECONDARY axis: ONE FIXED composition re-run at several n_total values.

    This is the control that says whether the x-optimum is a real composition
    effect or a crowding artefact: the ratio is held FIXED and only the number
    of monomers in the shell changes (BSA_LOADING_SWEEP = [30, 60, 100], roughly
    17 / 35 / 58 mM).  It is deliberately NOT crossed with the full grid —
    8 compositions x 3 loadings x 3 replicas is GPU-weeks on the wrong axis.

    Runs through the same driver, so each loading point gets the same per-point
    JSON, the same resume behaviour and the same built-composition assertion.
    """
    cfg = _ratio_grid_from_config() or {}
    if loadings is None:
        loadings = cfg.get("loading_sweep") or []
    if not loadings:
        raise ValueError(
            "run_phase4_loading_sweep called with no loadings, and neither "
            "PHASE4_LOADING_SWEEP nor BSA_LOADING_SWEEP is declared")
    if not winning_copies:
        raise ValueError(
            "the loading sweep re-runs ONE FIXED composition; it cannot be run "
            "without one. Pin BSA_LOADING_SWEEP_COMPOSITION, or let the x-sweep "
            "produce a QUALIFIED winner (one whose CI excludes the runner-up).")
    logger.info(f"  loading sweep composition = {dict(winning_copies)} "
                f"(source: {composition_source or 'caller'})")

    n_now = sum(int(v) for v in winning_copies.values())
    work_dir = Path(work_dir)
    results = {}
    for n_total in loadings:
        n_total = int(n_total)
        logger.info(f"\n  ── LOADING POINT n_total={n_total} "
                    f"(composition held at {winning_copies}) ──")
        # One-composition grid, rescaled by the driver so the rounding and the
        # zero-preservation logic are shared rather than re-implemented here.
        results[str(n_total)] = run_phase4_ratio_grid(
            target=target, pc_id=f"{pc_id}#n{n_total}",
            functional_monomers=functional_monomers, crosslinker=crosslinker,
            epitope_pdb=epitope_pdb,
            work_dir=work_dir / f"n{n_total}",
            grid=[dict(winning_copies)],
            total_monomers=n_total,
            time_ns=time_ns, n_replicas=n_replicas,
            max_monomers_in_shell=cfg.get("max_monomers_in_shell"))
    summary = {
        "schema": "phase4/loading-sweep/v2",
        "target": target, "pc_id": pc_id,
        "winning_copies": dict(winning_copies),
        "composition_source": composition_source,
        "winning_n_total": n_now,
        "loadings": [int(n) for n in loadings],
        "per_loading": results,
        "success": all(r.get("success") for r in results.values()),
    }
    (work_dir / "loading_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return summary


def run_phase4_ratio_sweep(target: str, pc_id: str,
                           functional_monomers: list, crosslinker: str,
                           epitope_pdb, work_dir, ratio_presets: list = None,
                           time_ns: float = 30) -> dict:
    """LEGACY preset sweep — now IMPOSES its copy numbers instead of reporting them.

    Kept for the free-radical CD compositions, whose presets are per-functional
    part ratios rather than absolute copy numbers.  It is a thin adapter: the
    presets are converted to explicit copy-number dicts (respecting
    EPITOPE_MONOMER_MOLAR_RATIO instead of the old hardcoded total=20, and
    WITHOUT the preset[:n_func] truncation that collapsed every preset onto one
    point for a single-functional-monomer system) and handed to the grid driver,
    which runs replicas, pools them and writes per-point JSON.

    Reference: Lutz 2019 Macromol Rapid Commun (sequence-controlled polymers).
    """
    from .config import PHASE4_RATIO_PRESETS

    if ratio_presets is None:
        ratio_presets = PHASE4_RATIO_PRESETS
    n_func = len(functional_monomers)
    total = int(_cfg_get("EPITOPE_MONOMER_MOLAR_RATIO", 20))

    poly_type = _polymerization_type(list(functional_monomers) + [crosslinker])
    x_key = ("SOLGEL_Q_MOLE_FRACTION" if poly_type == "sol-gel"
             else "RADICAL_CROSSLINKER_MOLE_FRACTION")
    x_cross = min(max(float(_cfg(x_key)), 1e-6), 1 - 1e-6)

    grid, seen = [], set()
    for preset in ratio_presets:
        # PAD, never truncate: a preset longer than the functional list used to
        # be cut down to one element, which is how five distinct presets became
        # five identical boxes.
        parts = list(preset)[:n_func] if len(preset) >= n_func else \
            list(preset) + [1] * (n_func - len(preset))
        if len(preset) > n_func:
            logger.warning(
                f"  preset {preset} has more entries than the {n_func} "
                f"functional monomer(s) {functional_monomers}; using the first "
                f"{n_func}. Distinct presets that differ only past that point "
                f"collapse onto the same box and are de-duplicated below.")
        n_cross = max(1, int(math.floor(total * x_cross + 0.5 + 1e-9)))
        rest = total - n_cross
        if rest < n_func:
            raise ValueError(
                f"total {total} cannot hold {n_func} functional monomer(s) at "
                f"{x_key}={x_cross}")
        scale = rest / sum(parts)
        copies = {m: max(1, int(round(r * scale)))
                  for m, r in zip(functional_monomers, parts)}
        copies[crosslinker] = total - sum(copies.values())
        if copies[crosslinker] < 1:
            raise ValueError(
                f"preset {preset} leaves no room for the crosslinker at "
                f"total={total} (copies={copies})")
        key = tuple(sorted(copies.items()))
        if key in seen:
            logger.info(f"  preset {preset} duplicates an earlier composition "
                        f"{copies} — skipped")
            continue
        seen.add(key)
        grid.append(dict(copies))

    return run_phase4_ratio_grid(
        target=target, pc_id=pc_id,
        functional_monomers=functional_monomers, crosslinker=crosslinker,
        epitope_pdb=epitope_pdb, work_dir=work_dir,
        grid=grid, time_ns=time_ns)
