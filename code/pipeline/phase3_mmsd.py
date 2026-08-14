"""
Phase 3: Greedy Forward Selection + MMSD
=========================================
Greedy forward selection with swap refinement to find the optimal
monomer combination (type + number), evaluated by MMSD sequential
docking (Rajpal et al. 2024).

Phase 2 SMD results are used to rank candidates (best BE first).
Forward selection iteratively adds monomers while avg BE improves,
then swap refinement tests replacements at each position.

Objective: mmsd_per_monomer + interference_penalty
  (size-normalized — fair comparison across 2-6종 combinations)

Key concept: MMSD sum vs SMD sum reveals synergy/interference.
  MMSD sum < SMD sum → synergy (cooperative binding)
  MMSD sum > SMD sum → interference (steric clash)

Reference:
  Rajpal et al., Sci. Rep. 2024 — MMSD protocol
  Sullivan et al., J. Phys. Chem. B 2019 — non-competitive binding
"""

import hashlib
import json
import logging
import random
from itertools import combinations
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Bump whenever the meaning of a published Phase 3 field changes. Result
# entries stamped with anything else are NOT resumed — see run_phase3.
_PHASE3_SCHEMA = "phase3/v2-blocker05"


# ════════════════════════════════════════════════════════════════
# BLOCKER 05(a) — dict-shape guards
#
# The v1 run passed phase2_results["be_matrix"] — which is keyed by TARGET —
# positionally into the `smd_be` slot, which expects a dict keyed by MONOMER.
# `smd_be.get(monomer)` therefore returned None for every monomer,
# `sum([]) == 0.0`, so smd_sum was 0.0 for all 213 evaluated compositions,
# delta_sum (= mmsd_sum - smd_sum) was always negative, and `"synergy": true`
# was published for all 213 combinations having never been computed.
#
# The positional call is what let one dict masquerade as the other, so smd_be
# and the full matrix are now KEYWORD-ONLY everywhere and both shapes are
# asserted at every entry point.
# ════════════════════════════════════════════════════════════════

def _assert_monomer_keyed_be(smd_be, where: str) -> dict:
    """Assert `smd_be` is a non-empty {monomer_name: binding_energy} dict.

    Raises loudly rather than letting a target-keyed matrix through, because
    the failure mode it guards is silent (smd_sum -> 0.0, synergy -> always
    true) and is only visible many phases downstream.
    """
    if not isinstance(smd_be, dict):
        raise TypeError(
            f"{where}: smd_be must be a dict {{monomer: BE}}, got "
            f"{type(smd_be).__name__}. Pass smd_be=phase2_results"
            f"['be_matrix'][target], by keyword.")
    if not smd_be:
        raise ValueError(
            f"{where}: smd_be is EMPTY. Phase 2 produced no per-monomer binding "
            f"energies for this target, so the MMSD-vs-SMD synergy statistic "
            f"cannot be computed. Refusing to publish delta_sum=mmsd_sum-0.0.")
    nested = [k for k, v in smd_be.items() if isinstance(v, dict)]
    if nested:
        raise TypeError(
            f"{where}: smd_be looks TARGET-keyed, not monomer-keyed — entries "
            f"{nested[:3]} map to dicts. This is exactly BLOCKER 05(a): pass "
            f"smd_be=phase2_results['be_matrix'][target] (one target's row), "
            f"and the full matrix separately as be_matrix_all=, by keyword.")
    numeric = [v for v in smd_be.values()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric:
        raise ValueError(
            f"{where}: smd_be has {len(smd_be)} entries but not one numeric "
            f"binding energy among them — values look like "
            f"{[type(v).__name__ for v in list(smd_be.values())[:3]]}.")
    return smd_be


def _assert_target_keyed_be(be_matrix_all, where: str) -> dict:
    """Assert `be_matrix_all` is the FULL {target: {monomer: BE}} matrix."""
    if not isinstance(be_matrix_all, dict):
        raise TypeError(
            f"{where}: be_matrix_all must be the full {{target: {{monomer: BE}}}} "
            f"matrix, got {type(be_matrix_all).__name__}.")
    if not be_matrix_all:
        raise ValueError(
            f"{where}: be_matrix_all is EMPTY — the NSGA-II selectivity "
            f"objective would silently score every combination identically.")
    flat = [k for k, v in be_matrix_all.items() if not isinstance(v, dict)]
    if flat:
        raise TypeError(
            f"{where}: be_matrix_all looks MONOMER-keyed, not target-keyed — "
            f"entries {flat[:3]} map to scalars. Pass the full "
            f"phase2_results['be_matrix'], not one target's row.")
    return be_matrix_all


# ════════════════════════════════════════════════════════════════
# MMSD docking-order policy
#
# MMSD is a SEQUENTIAL protocol: monomer k is docked into a receptor that
# already carries the best poses of monomers 0..k-1. The per-monomer BE
# therefore depends on WHICH STEP the monomer occupies — up to ~2 kcal/mol in
# the v1 run. In v1 the step order was the genotype's arbitrary gene order,
# while every evaluation cache was keyed on the SORTED SET, so exactly one
# arbitrary order per combination was ever measured and the resulting spread
# was read as chemistry.
#
# Default policy is now "smd_rank": a CANONICAL order derived from Phase 2
# SMD affinity, strongest binder first (the strongest binder claims the
# highest-affinity subsite, matching the sequential rationale of Rajpal 2024).
# Ties break on monomer name, so the order is a pure function of the SET and
# the cache key is the order actually measured.
#
# This is explicitly an APPROXIMATION: the reported mmsd_sum is one point on
# the order distribution, not its mean. Set MMSD_ORDER_REPLICATES > 1 to
# average over several orders instead (cost multiplies by that factor) — the
# result then carries mmsd_sum_std_over_orders so the spread is visible.
# ════════════════════════════════════════════════════════════════

_ORDER_POLICIES = ("smd_rank", "as_given")


def _order_policy() -> tuple:
    """(policy, n_replicates, seed) from config, with documented defaults."""
    from . import config as _cfg
    policy = getattr(_cfg, "MMSD_DOCKING_ORDER", "smd_rank")
    n_rep = int(getattr(_cfg, "MMSD_ORDER_REPLICATES", 1))
    seed = int(getattr(_cfg, "MMSD_ORDER_SEED", 42))
    if policy not in _ORDER_POLICIES:
        raise ValueError(
            f"MMSD_DOCKING_ORDER={policy!r} is not one of {_ORDER_POLICIES}. "
            f"Refusing to guess a docking order.")
    if n_rep < 1:
        raise ValueError(f"MMSD_ORDER_REPLICATES must be >= 1, got {n_rep}")
    return policy, n_rep, seed


def _canonical_mmsd_order(monomers: list, smd_be: dict,
                          policy: str = "smd_rank") -> list:
    """Return the docking order for `monomers` under the active policy.

    "smd_rank" (default): ascending Phase 2 SMD BE (most negative = strongest
        binder first), ties broken alphabetically. A pure function of the SET.
    "as_given": legacy — keep the caller's order. Callers must then key their
        caches on the ORDERED tuple (they do), so different orders are measured
        separately instead of one arbitrary order standing in for all of them.
    """
    if policy == "as_given":
        return list(monomers)
    return sorted(
        monomers,
        key=lambda m: (smd_be.get(m) if isinstance(smd_be.get(m), (int, float))
                       and not isinstance(smd_be.get(m), bool) else 0.0, m),
    )


def _order_replicates(monomers: list, smd_be: dict, policy: str,
                      n_rep: int, seed: int) -> list:
    """Distinct docking orders to average over. Element 0 is always canonical."""
    canonical = _canonical_mmsd_order(monomers, smd_be, policy)
    if n_rep <= 1 or len(canonical) < 2:
        return [canonical]
    orders = [canonical]
    rng = random.Random(seed)
    # bounded, deterministic search for distinct permutations
    for _ in range(n_rep * 20):
        if len(orders) >= n_rep:
            break
        cand = canonical[:]
        rng.shuffle(cand)
        if cand not in orders:
            orders.append(cand)
    return orders


def _cache_key(monomers: list, smd_be: dict, policy: str) -> tuple:
    """Cache key = the ORDERED tuple actually docked (not the sorted set).

    Under "smd_rank" this is the canonical order, so it still collapses to one
    entry per combination — but the entry now corresponds to the order that was
    measured. Under "as_given" two gene orders of the same set get two keys and
    two measurements, instead of one masquerading as the other.
    """
    return tuple(_canonical_mmsd_order(monomers, smd_be, policy))


def run_phase3(phase1_results: dict = None,
               phase2_results: dict = None,
               target_names: list = None,
               output_dir: str = None,
               fresh: bool = False) -> dict:
    """
    Phase 3 entry point: Greedy Forward Selection + MMSD for all targets.

    For each target:
    1. Forward selection: iteratively add best monomer by avg BE
    2. Swap refinement: try replacing each monomer with alternatives
    3. Automatically determine optimal combination size (2-6)
    4. Rank and select top polymer compositions (PCs)
    """
    from .config import (TARGETS, FUNCTIONAL_MONOMERS,
                         MMSD_TOP_PC, MMSD_HIGH_AFFINITY_THRESHOLD,
                         AUTODOCK4_GA_RUNS, get_output_path)

    if output_dir is None:
        output_dir = str(get_output_path("phase3"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load previous phase results
    if phase1_results is None:
        p1_path = get_output_path("phase1") / "phase1_results.json"
        with open(p1_path) as f:
            phase1_results = json.load(f)

    if phase2_results is None:
        p2_path = get_output_path("phase2") / "phase2_smd_results.json"
        with open(p2_path) as f:
            phase2_results = json.load(f)

    if target_names is None:
        target_names = list(phase1_results.keys())

    results = {}

    from .config import resolve_path, MMSD_SELECTIVITY_AWARE

    # Build per-target receptor metadata (used for cross-docking selectivity)
    target_meta = {}
    for t in target_names:
        if t not in phase1_results:
            continue
        p1t = phase1_results[t]
        target_meta[t] = {
            "receptor": resolve_path(p1t["receptor_pdbqt"]),
            "epitope": resolve_path(p1t["epitope_pdb"]),
            "center": tuple(p1t["grid_center"]),
            "npts": tuple(p1t["grid_npts"]),
        }

    # Per-target resume: load partial results if previous run was interrupted.
    # Only entries stamped with the CURRENT schema may be resumed — a v1 entry
    # carries smd_sum=0.0, synergy=true and a null-mmsd_result Pareto front
    # (BLOCKER 05), and resuming it would republish exactly those numbers.
    partial_json = output_dir / "phase3_mmsd_results.json"
    if fresh and partial_json.exists():
        # --fresh: do not read the partial results at all. Previously the only
        # thing making --fresh meaningful here was run_pipeline having archived
        # the directory first; now the flag is honoured by the phase itself.
        logger.warning("--fresh: IGNORING existing phase3_mmsd_results.json; "
                       "every requested target will be recomputed")
    elif partial_json.exists():
        try:
            with open(partial_json) as f:
                results = json.load(f)
            stale = [t for t, v in results.items()
                     if isinstance(v, dict) and v.get("top_pcs")
                     and v.get("schema") != _PHASE3_SCHEMA]
            for t in stale:
                logger.error(
                    f"Resume: DISCARDING pre-{_PHASE3_SCHEMA} Phase 3 result for "
                    f"{t} (schema={results[t].get('schema')!r}). It was produced "
                    f"before BLOCKER 05 was fixed — smd_sum=0.0, synergy always "
                    f"true, Pareto front unusable. {t} will be recomputed.")
                results.pop(t)
            done_targets = [t for t, v in results.items()
                            if isinstance(v, dict) and v.get("top_pcs")
                            and v.get("schema") == _PHASE3_SCHEMA]
            if done_targets:
                logger.info(f"Resume: loaded existing Phase 3 results for "
                            f"{done_targets} — skipping these targets")
        except Exception as e:
            logger.warning(f"Could not parse existing phase3_mmsd_results.json: {e}")
            results = {}

    for target in target_names:
        # Skip if already completed in a prior run (current schema only)
        if (isinstance(results.get(target), dict) and results[target].get("top_pcs")
                and results[target].get("schema") == _PHASE3_SCHEMA):
            logger.info(f"\n{'='*20} Phase 3: {target} (RESUMED — already done) {'='*20}")
            continue

        logger.info(f"\n{'='*20} Phase 3 Greedy+MMSD: {target} {'='*20}")

        # BLOCKER 05(a): keep the two shapes under two clearly different names
        # and never pass either positionally.
        #   smd_be       {monomer: BE}          — this target's Phase 2 row
        #   be_matrix_all{target: {monomer: BE}} — full matrix, selectivity only
        be_matrix_all = _assert_target_keyed_be(
            phase2_results["be_matrix"], f"run_phase3({target})")
        smd_be = be_matrix_all.get(target, {})
        if not smd_be:
            logger.error(f"[{target}] Phase 2 be_matrix has no row for this "
                         f"target — Phase 3 cannot compute the MMSD-vs-SMD "
                         f"synergy statistic. Skipping target.")
            results[target] = {"error": "No Phase 2 be_matrix row for target"}
            continue
        _assert_monomer_keyed_be(smd_be, f"run_phase3({target})")

        filtered = phase2_results["filtered"].get(target, [])

        if not filtered:
            logger.warning(f"[{target}] No filtered monomers from Phase 2")
            results[target] = {"error": "No candidates"}
            continue

        p1 = phase1_results[target]
        receptor_pdbqt = resolve_path(p1["receptor_pdbqt"])
        epitope_pdb = resolve_path(p1["epitope_pdb"])
        center = tuple(p1["grid_center"])
        npts = tuple(p1["grid_npts"])

        # Available monomers for BO (exclude all crosslinkers)
        from .config import CROSSLINKERS
        available = [m for m in filtered if m not in CROSSLINKERS]
        if not available:
            logger.warning(f"[{target}] No monomers passed Phase 2 filter")
            results[target] = {"error": "No candidates from Phase 2"}
            continue

        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        # Off-target receptors (for cross-MMSD ΔΔG selectivity penalty)
        off_targets = ({k: v for k, v in target_meta.items() if k != target}
                       if MMSD_SELECTIVITY_AWARE else None)
        if off_targets:
            logger.info(f"  Selectivity-aware MMSD enabled: "
                        f"cross-docking vs {list(off_targets.keys())}")

        # A4/C2: Choose optimizer — nsga2 (default) / bayesian / greedy
        from .config import MMSD_OPTIMIZER, BAYESIAN_N_CALLS

        # Auto-fallback: nsga2 → bayesian → greedy if deps missing
        active_optimizer = MMSD_OPTIMIZER
        if active_optimizer == "nsga2":
            try:
                import pymoo  # noqa
            except ImportError:
                logger.warning("pymoo missing — falling back to BO")
                active_optimizer = "bayesian"
        if active_optimizer == "bayesian":
            try:
                import skopt  # noqa
            except ImportError:
                logger.warning("skopt missing — falling back to greedy")
                active_optimizer = "greedy"

        nsga_pareto_front = None  # populated only by NSGA-II

        if active_optimizer == "nsga2":
            logger.info(f"\n  Using NSGA-II 3-objective optimization "
                        f"(Affinity + Selectivity + Synthesizability)")
            nsga_result = _run_nsga2_mmsd(
                target=target,
                available_monomers=available,
                receptor_pdbqt=receptor_pdbqt,
                epitope_pdb=epitope_pdb,
                center=center, npts=npts,
                # BOTH by keyword, and they are DIFFERENT objects:
                smd_be=smd_be,                # {monomer: BE} for THIS target
                be_matrix_all=be_matrix_all,  # {target: {monomer: BE}}
                work_dir=target_dir,
                ga_runs=min(AUTODOCK4_GA_RUNS, 25),
                all_targets=target_names,
                pop_size=20,
                n_gen=15,
                off_targets=off_targets,
            )
            if nsga_result is not None:
                all_pc_results = nsga_result["all_evaluated"]
                nsga_pareto_front = nsga_result["selected_pareto_front"]
            else:  # pymoo missing → fallback to greedy
                all_pc_results = _run_greedy_mmsd(
                    target=target, available_monomers=available,
                    smd_be=smd_be, receptor_pdbqt=receptor_pdbqt,
                    epitope_pdb=epitope_pdb, center=center, npts=npts,
                    work_dir=target_dir, ga_runs=min(AUTODOCK4_GA_RUNS, 25),
                    off_targets=off_targets,
                )
        elif active_optimizer == "bayesian":
            logger.info(f"\n  Using Bayesian Optimization (n_calls={BAYESIAN_N_CALLS})")
            bo_result = _run_bayesian_mmsd(
                target=target,
                available_monomers=available,
                receptor_pdbqt=receptor_pdbqt,
                epitope_pdb=epitope_pdb,
                center=center, npts=npts,
                smd_be=smd_be,
                work_dir=target_dir,
                ga_runs=min(AUTODOCK4_GA_RUNS, 25),
                n_calls=BAYESIAN_N_CALLS,
                off_targets=off_targets,
            )
            if bo_result is not None:
                all_pc_results = bo_result["all_results"]
            else:  # skopt missing → fallback
                all_pc_results = _run_greedy_mmsd(
                    target=target, available_monomers=available,
                    smd_be=smd_be, receptor_pdbqt=receptor_pdbqt,
                    epitope_pdb=epitope_pdb, center=center, npts=npts,
                    work_dir=target_dir, ga_runs=min(AUTODOCK4_GA_RUNS, 25),
                    off_targets=off_targets,
                )
        else:  # greedy (default, legacy)
            all_pc_results = _run_greedy_mmsd(
                target=target,
                available_monomers=available,
                smd_be=smd_be,
                receptor_pdbqt=receptor_pdbqt,
                epitope_pdb=epitope_pdb,
                center=center, npts=npts,
                work_dir=target_dir,
                ga_runs=min(AUTODOCK4_GA_RUNS, 25),
                off_targets=off_targets,
            )

        # Remove excluded PCs
        all_pc_results = [r for r in all_pc_results if not r.get("excluded")]

        # Rank by bo_objective only (competition is informational, not filtering)
        all_pc_results.sort(key=lambda x:
            x.get("bo_objective") if x.get("bo_objective") is not None else 0,
        )

        # C2 fix: when NSGA-II used, take FULL combo (functional + crosslinker)
        # and verify compatibility — auto-picked crosslinker may break otherwise
        # compatible functional set.
        from .config import (is_polymerization_compatible,
                             MMSD_REQUIRE_CHEMISTRY_DIVERSITY)
        from .utils_analysis import check_chemistry_diversity

        def _full_compat(r):
            """Check compatibility AND chemistry diversity of full combo."""
            full = list(r.get("functional_monomers", []) or r.get("monomers", []))
            xl = r.get("crosslinker")
            if xl and xl not in full:
                full.append(xl)
            ok, _, _ = is_polymerization_compatible(full)
            if not ok:
                return False
            # Rule 1 + Rule 2 hard check (entropy bonus already in NSGA-II obj)
            if MMSD_REQUIRE_CHEMISTRY_DIVERSITY:
                div = check_chemistry_diversity(full)
                if not div["valid"]:
                    return False
            return True

        # ── Final pick ───────────────────────────────────────────────
        # BLOCKER 05(b): when NSGA-II ran, the pick MUST come from the
        # 3-objective Pareto front, ranked by the documented rule in
        # _PARETO_RULE_DOC — not from argmin(bo_objective), which is a single
        # scalar and throws away selectivity and synthesizability entirely.
        # The front is already ordered by _rank_pareto_front.
        pareto_selection = None
        if nsga_pareto_front is not None:
            n_front = len(nsga_pareto_front)
            n_null = sum(1 for p_ in nsga_pareto_front
                         if not (p_.get("mmsd_result") or {}))
            if n_null:
                logger.error(
                    f"[{target}] {n_null}/{n_front} Pareto entries carry "
                    f"mmsd_result=null and cannot be selected from.")
            top_pcs = []
            for p_ in nsga_pareto_front:
                m_res = p_.get("mmsd_result") or {}
                if not m_res or not _full_compat(m_res):
                    continue
                m_res = dict(m_res)
                m_res["selected_from"] = "nsga2_pareto_front"
                m_res["pareto_rank_score"] = p_.get("pareto_rank_score")
                m_res["pareto_objectives"] = p_.get("objectives")
                top_pcs.append(m_res)
                if len(top_pcs) >= MMSD_TOP_PC:
                    break
            pareto_selection = {
                "rule": _PARETO_RULE_DOC,
                "weights": list(_pareto_weights()),
                "n_front": n_front,
                "n_front_null_mmsd_result": n_null,
                "n_selected_from_front": len(top_pcs),
                "fallback_used": False,
            }
            if not top_pcs:
                # LOUD: falling back to the scalar objective is exactly the v1
                # failure. It is allowed, but it is recorded in the JSON.
                logger.error(
                    f"[{target}] NO usable Pareto solution "
                    f"({n_front} entries, {n_null} with null mmsd_result, rest "
                    f"failed compatibility/diversity). FALLING BACK to "
                    f"argmin(bo_objective) — a SINGLE-objective pick. The "
                    f"composition handed to Phases 4/5/6 is NOT Pareto-selected.")
                top_pcs = [dict(r, selected_from="fallback_argmin_bo_objective")
                           for r in all_pc_results if _full_compat(r)][:MMSD_TOP_PC]
                pareto_selection["fallback_used"] = True
                pareto_selection["fallback_reason"] = (
                    "no compatible Pareto entry with a populated mmsd_result")
        else:
            top_pcs = [dict(r, selected_from="argmin_bo_objective")
                       for r in all_pc_results if _full_compat(r)][:MMSD_TOP_PC] \
                     or [dict(r, selected_from="argmin_bo_objective_incompatible")
                         for r in all_pc_results[:MMSD_TOP_PC]]

        # B5: DFT validation hook for top PCs
        dft_refined = None
        try:
            dft_refined = _dft_validate_top_combinations(top_pcs, target, target_dir)
        except Exception as _e:
            logger.debug(f"DFT validation skipped: {_e}")

        sel_tag = " + cross-MMSD ΔΔG" if off_targets else ""
        method_name = (
            f"NSGA-II 3-objective (Affinity+Selectivity+Synthesizability{sel_tag})"
            if active_optimizer == "nsga2"
            else f"Bayesian Optimization (GP{sel_tag})"
            if active_optimizer == "bayesian"
            else f"Greedy Forward Selection + MMSD{sel_tag}"
        )
        results[target] = {
            "schema": _PHASE3_SCHEMA,
            "method": method_name,
            "n_evaluations": len(all_pc_results),
            "available_monomers": available,
            "top_pcs": top_pcs,
            "pareto_front": nsga_pareto_front,  # C2: only set if NSGA-II
            "pareto_selection": pareto_selection,
            "docking_order_policy": {
                "policy": _order_policy()[0],
                "replicates": _order_policy()[1],
                "note": ("MMSD is order-dependent. Compositions are docked in a "
                         "canonical order and caches are keyed on that ordered "
                         "tuple, so the measured value is a reproducible "
                         "function of the SET — an approximation of the mean "
                         "over orders, not the mean itself."),
            },
            "all_results": all_pc_results,
            "dft_validation": dft_refined,
            "synergy_computed_count": sum(
                1 for r in all_pc_results if r.get("delta_sum") is not None),
            "synergy_true_count": sum(
                1 for r in all_pc_results if r.get("synergy") is True),
            "high_affinity_count": sum(
                1 for r in all_pc_results
                if r.get("mmsd_sum") is not None
                and r["mmsd_sum"] <= MMSD_HIGH_AFFINITY_THRESHOLD
            ),
        }

        # Log top results
        logger.info(f"\n[{target}] Top {min(MMSD_TOP_PC, len(top_pcs))} PCs "
                    f"(from {len(all_pc_results)} BO evaluations):")
        for j, pc in enumerate(top_pcs, 1):
            logger.info(
                f"  {j}. {pc['monomers']} "
                f"(XL={pc.get('crosslinker', '?')}): "
                f"obj={pc.get('bo_objective', 'N/A')}, "
                f"avg={pc.get('mmsd_per_monomer', 'N/A')}, "
                f"sum={pc.get('mmsd_sum', 'N/A')}"
            )

        # Per-target incremental save — survives crashes
        try:
            with open(output_dir / "phase3_mmsd_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"  Phase 3: {target} complete → saved partial results")
        except Exception as e:
            logger.warning(f"  Failed to save partial Phase 3 results: {e}")

    # Final save
    with open(output_dir / "phase3_mmsd_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    _plot_mmsd_comparison(results, output_dir)

    return results


# ── Crosslinker Compatibility ─────────────────────────────────

def _get_compatible_crosslinkers(monomers: list) -> list:
    """
    Return list of compatible crosslinkers based on monomer POLYMERIZATION TYPE
    (not just dict membership).

    Polymerization type metadata is the source of truth:
      - silane functional → silane crosslinker (TEOS, TMOS)
      - vinyl/catechol functional → vinyl crosslinker (MBAAm, EGDMA, DVB, TRIM)
      - surface-only (e.g., APBA, FPBA) → does NOT influence crosslinker choice
        (must be combined with another polymerization-active monomer)
    """
    from .config import ALL_MONOMERS, CROSSLINKER_LIBRARY

    has_silane = any(
        ALL_MONOMERS.get(m, {}).get("polymerization") == "silane"
        for m in monomers
    )
    has_radical = any(
        ALL_MONOMERS.get(m, {}).get("polymerization") in ("vinyl", "catechol")
        for m in monomers
    )

    if has_silane and not has_radical:
        return [k for k, v in CROSSLINKER_LIBRARY.items()
                if v.get("polymerization", v.get("type")) == "silane"]
    elif has_radical and not has_silane:
        return [k for k, v in CROSSLINKER_LIBRARY.items()
                if v.get("polymerization", v.get("type")) == "vinyl"]
    elif has_silane and has_radical:
        # Mixed functional set — cannot synthesize in one pot.
        # Return empty so this combo is skipped or flagged.
        return []
    else:
        # All surface-only (APBA/FPBA only) — invalid without polymerization matrix
        return []


# ── Selectivity-Aware MMSD (cross-docking ΔΔG penalty) ────────

def _run_mmsd_with_order_policy(target: str, pc_id: str,
                                functional_monomers: list,
                                compatible_crosslinkers: list,
                                receptor_pdbqt: Path, epitope_pdb: Path,
                                center: tuple, npts: tuple,
                                work_dir: Path, *, smd_be: dict,
                                ga_runs: int = 25) -> dict:
    """Run MMSD under the active docking-order policy.

    MMSD_ORDER_REPLICATES == 1 (default): one canonical order, and the result
    says which order that was and that it is an approximation of the mean over
    orders.

    MMSD_ORDER_REPLICATES == K > 1: dock K distinct orders and AVERAGE. The
    crosslinker is then picked on its mean BE across orders, and the returned
    metrics are re-derived from the averaged per-monomer energies by the same
    _finalize_mmsd_metrics used for the single-order path. Cost is K x.
    """
    policy, n_rep, seed = _order_policy()
    orders = _order_replicates(functional_monomers, smd_be, policy, n_rep, seed)

    if len(orders) == 1:
        res = _run_single_mmsd(
            target, pc_id, functional_monomers, compatible_crosslinkers,
            receptor_pdbqt, epitope_pdb, center, npts, work_dir,
            smd_be=smd_be, ga_runs=ga_runs, docking_order=orders[0],
        )
        res["n_orders"] = 1
        res["mmsd_sum_std_over_orders"] = None
        res["order_dependence_note"] = (
            "Single canonical order (MMSD_DOCKING_ORDER=%s). MMSD is "
            "order-dependent; this is one point on the order distribution, not "
            "its mean. Set MMSD_ORDER_REPLICATES>1 to average." % policy)
        return res

    runs = []
    for k, order in enumerate(orders):
        r = _run_single_mmsd(
            target, f"{pc_id}_ord{k}", functional_monomers,
            compatible_crosslinkers, receptor_pdbqt, epitope_pdb, center, npts,
            Path(work_dir) / f"order{k}",
            smd_be=smd_be, ga_runs=ga_runs, docking_order=order,
        )
        if r.get("excluded"):
            r["pc_id"] = pc_id
            r["n_orders"] = k + 1
            r["order_dependence_note"] = (
                f"Order replicate {k} was excluded; the whole composition is "
                f"excluded rather than averaging over a biased subset of orders.")
            return r
        runs.append(r)

    # Mean per-monomer energies across orders (functional monomers only:
    # each run may have chosen a different crosslinker).
    func_names = list(functional_monomers)
    mean_energies = {}
    for m in func_names:
        vals = [r["monomer_energies"][m] for r in runs
                if isinstance(r["monomer_energies"].get(m), (int, float))]
        if not vals:
            continue
        mean_energies[m] = float(np.mean(vals))

    # Crosslinker chosen on its MEAN BE across orders.
    xl_means = {}
    for xl in compatible_crosslinkers:
        vals = [r["crosslinker_comparison"].get(xl) for r in runs]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            xl_means[xl] = float(np.mean(vals))
    if not xl_means:
        base = runs[0]
        base["pc_id"] = pc_id
        base["excluded"] = True
        base["order_dependence_note"] = "No crosslinker scored in any order."
        return base
    best_xl = min(xl_means, key=xl_means.get)
    mean_energies[best_xl] = xl_means[best_xl]

    all_monomers = func_names + [best_xl]
    metrics = _finalize_mmsd_metrics(
        pc_id=pc_id, all_monomers=all_monomers,
        monomer_energies=mean_energies, smd_be=smd_be,
    )
    per_order_sums = [r["mmsd_sum"] for r in runs]
    out = {
        "pc_id": pc_id,
        "monomers": all_monomers,
        "functional_monomers": func_names,
        "docking_order": None,          # averaged — no single order applies
        "docking_order_rule": f"{policy}+average",
        "n_functional": len(func_names),
        "crosslinker": best_xl,
        "crosslinker_comparison": {k: round(v, 3) for k, v in xl_means.items()},
        "monomer_energies": {k: round(v, 3) for k, v in mean_energies.items()},
        "competition": runs[0].get("competition", {}),
        "is_uniform": runs[0].get("is_uniform", True),
        "n_orders": len(runs),
        "orders_docked": orders,
        "per_order_mmsd_sum": per_order_sums,
        "mmsd_sum_std_over_orders": round(float(np.std(per_order_sums)), 3),
        "order_dependence_note": (
            f"Averaged over {len(runs)} distinct docking orders; "
            f"mmsd_sum_std_over_orders quantifies the protocol spread."),
    }
    out.update(metrics)
    return out


def _evaluate_with_selectivity(target: str, pc_id: str,
                                functional_monomers: list,
                                compatible_crosslinkers: list,
                                receptor_pdbqt: Path, epitope_pdb: Path,
                                center: tuple, npts: tuple,
                                work_dir: Path, *,
                                smd_be: dict,
                                ga_runs: int = 25,
                                off_targets: dict = None) -> dict:
    """Run MMSD on own target + each off-target receptor, compute ΔΔG penalty.

    bo_objective_with_sel = bo_objective + w * max(0, DDG - threshold)
      DDG = mmsd_sum_own - mean(mmsd_sum_off)   (kcal/mol; negative = own stronger)
      threshold = -1.0 means "own must be ≥ 1 kcal/mol stronger than off-mean"
      DDG ≤ threshold (very negative) → no penalty (selective enough)
      DDG > threshold → penalty grows with how non-selective the combo is

    off_targets is dict: {target_name: {"receptor": Path, "epitope": Path,
                                        "center": tuple, "npts": tuple}}.
    If None or empty, behaves identically to _run_single_mmsd (no penalty).

    `smd_be` is KEYWORD-ONLY: it must be the monomer-keyed row
    phase2_results["be_matrix"][target], never the full target-keyed matrix.

    Reference: Garcia-Ortegon 2022 DOCKSTRING JCIM 62:3486.
    """
    from .config import (MMSD_SELECTIVITY_AWARE, SELECTIVITY_WEIGHT,
                         SELECTIVITY_DDG_THRESHOLD)

    _assert_monomer_keyed_be(smd_be, f"_evaluate_with_selectivity({pc_id})")

    own_result = _run_mmsd_with_order_policy(
        target, pc_id, functional_monomers, compatible_crosslinkers,
        receptor_pdbqt, epitope_pdb, center, npts, work_dir,
        smd_be=smd_be, ga_runs=ga_runs,
    )

    # "NOT MEASURED" AND "NO PENALTY" ARE DIFFERENT (REVIEW FINDING 16).
    #
    # All three early exits used to write selectivity_penalty = 0.0, so a
    # composition whose selectivity could not be measured received exactly the
    # same objective contribution as one measured to be perfectly selective.
    # The only distinguishing field was DDG_selectivity is None, which nothing
    # gated on.
    #
    # Now: an unmeasurable selectivity is None (with selectivity_measured
    # False and a reason), while a DELIBERATELY disabled feature keeps 0.0 —
    # because "the user turned selectivity scoring off" really does mean "add
    # nothing to the objective".
    if own_result.get("excluded") or own_result.get("mmsd_sum") is None:
        own_result["DDG_selectivity"] = None
        own_result["selectivity_penalty"] = None
        own_result["selectivity_measured"] = False
        own_result["selectivity_status"] = "own_mmsd_unusable"
        own_result["cross_target_be"] = {}
        return own_result

    if not MMSD_SELECTIVITY_AWARE or not off_targets:
        own_result["DDG_selectivity"] = None
        own_result["selectivity_penalty"] = 0.0      # deliberate: feature off
        own_result["selectivity_measured"] = False
        own_result["selectivity_status"] = (
            "selectivity_aware_disabled" if not MMSD_SELECTIVITY_AWARE
            else "no_off_targets_configured")
        own_result["cross_target_be"] = {}
        return own_result

    # Cross-target MMSD: same combo, different receptors
    off_mmsd_sums = {}
    cross_failures = {}
    for off_name, off_meta in off_targets.items():
        if off_name == target:
            continue
        off_dir = work_dir / f"cross_{off_name}"
        try:
            # Same composition, same docking-order policy, different receptor —
            # so ΔΔG is a receptor difference and not an order artefact.
            off_result = _run_mmsd_with_order_policy(
                target, f"{pc_id}_vs_{off_name}",
                functional_monomers, compatible_crosslinkers,
                off_meta["receptor"], off_meta["epitope"],
                off_meta["center"], off_meta["npts"], off_dir,
                smd_be=smd_be, ga_runs=ga_runs,
            )
            if off_result.get("mmsd_sum") is not None:
                off_mmsd_sums[off_name] = off_result["mmsd_sum"]
        except Exception as e:
            # Raised from DEBUG (REVIEW FINDING 16). The pipeline runs at INFO,
            # so this line was never emitted: every cross-target MMSD could
            # fail and the run would look clean.
            cross_failures[off_name] = f"{type(e).__name__}: {e}"
            logger.error(f"  Cross-MMSD {pc_id} vs {off_name} FAILED: {e}")

    if not off_mmsd_sums:
        own_result["DDG_selectivity"] = None
        # None, NOT 0.0 — selectivity was not measured, so there is no penalty
        # to apply and no evidence of selectivity either. Scoring this 0.0 made
        # an unmeasurable composition tie with a perfectly selective one.
        own_result["selectivity_penalty"] = None
        own_result["selectivity_measured"] = False
        own_result["selectivity_status"] = "all_cross_target_mmsd_failed"
        own_result["cross_target_failures"] = cross_failures
        own_result["cross_target_be"] = {}
        logger.error(
            f"    {pc_id}: NO cross-target MMSD succeeded "
            f"({list(off_targets)}) — selectivity is UNMEASURED for this "
            f"composition. selectivity_penalty is None, not 0.0; do not read "
            f"its bo_objective as selectivity-corrected. Failures: "
            f"{cross_failures}")
        return own_result

    own_sum = own_result["mmsd_sum"]
    off_mean = float(np.mean(list(off_mmsd_sums.values())))
    ddg = own_sum - off_mean  # negative = own preferred
    # Penalize when ΔΔG exceeds threshold (i.e., own not selective enough).
    # threshold = -1.0 ⇒ require own ≤ off_mean − 1 kcal/mol for zero penalty.
    penalty = SELECTIVITY_WEIGHT * max(0.0, ddg - SELECTIVITY_DDG_THRESHOLD)

    own_result["DDG_selectivity"] = round(ddg, 3)
    own_result["selectivity_penalty"] = round(penalty, 3)
    own_result["selectivity_measured"] = True
    own_result["selectivity_status"] = (
        "measured" if not cross_failures else "measured_partial")
    own_result["cross_target_be"] = {k: round(v, 3) for k, v in off_mmsd_sums.items()}
    if cross_failures:
        # Measured, but against fewer off-targets than configured. The ΔΔG is
        # real; its denominator is not what a reader would assume.
        own_result["cross_target_failures"] = cross_failures
        logger.warning(
            f"    {pc_id}: selectivity measured against "
            f"{len(off_mmsd_sums)}/{len(off_targets)} off-targets; "
            f"failed: {list(cross_failures)}")
    if own_result.get("bo_objective") is not None:
        own_result["bo_objective"] = round(own_result["bo_objective"] + penalty, 3)

    logger.info(f"    {pc_id} ΔΔG={ddg:+.2f} kcal/mol, "
                f"penalty={penalty:.2f} → obj={own_result.get('bo_objective')}")
    return own_result


# ── Greedy Forward Selection + Swap Refinement ──────────────

# COMBINATION-SIZE BOUNDS COME FROM CONFIG (REVIEW FINDING 18).
#
# All three optimizers used the hardcoded defaults `min_size=2, max_size=6`,
# which happen to equal MMSD_MIN_COMBO_SIZE / MMSD_MAX_COMBO_SIZE today — so
# the config keys looked authoritative while being completely inert. Changing
# either knob did nothing, and the coincidence would have hidden the drift the
# moment someone edited config. _run_greedy_mmsd even imported
# MMSD_MIN_COMBO_SIZE and never referenced it.
def _combo_size_bounds():
    """(min_size, max_size) for a monomer combination, from config."""
    from .config import MMSD_MIN_COMBO_SIZE, MMSD_MAX_COMBO_SIZE
    lo, hi = int(MMSD_MIN_COMBO_SIZE), int(MMSD_MAX_COMBO_SIZE)
    if lo < 1 or hi < lo:
        raise ValueError(
            f"MMSD_MIN_COMBO_SIZE={lo} / MMSD_MAX_COMBO_SIZE={hi} are not a "
            f"usable range (need 1 <= min <= max)")
    return lo, hi


def _run_greedy_mmsd(target: str, available_monomers: list,
                      receptor_pdbqt: Path, epitope_pdb: Path,
                      center: tuple, npts: tuple,
                      work_dir: Path, *,
                      smd_be: dict,
                      ga_runs: int = 25,
                      min_size: int = None, max_size: int = None,
                      off_targets: dict = None) -> list:
    """
    Greedy forward selection + swap refinement for optimal monomer combination.

    Phase A — Forward Selection:
      Round 1: try each monomer alone → pick best by avg BE
      Round 2: try adding each remaining monomer → pick best pair
      ...continue until avg BE stops improving or max_size reached

    Phase B — Swap Refinement:
      Try replacing each selected monomer with every unselected one.
      Accept if bo_objective improves. Repeat until no more swaps help.

    Uses Phase 2 SMD BE (`smd_be`, monomer-keyed, KEYWORD-ONLY) to sort
    candidates — best SMD monomers are tried first for efficiency.
    """
    # MMSD_MIN_COMBO_SIZE used to be imported HERE and never referenced.
    _lo, _hi = _combo_size_bounds()
    min_size = _lo if min_size is None else int(min_size)
    max_size = _hi if max_size is None else int(max_size)

    _assert_monomer_keyed_be(smd_be, f"_run_greedy_mmsd({target})")

    all_results = []
    selected = []
    best_avg = float("inf")
    best_obj = float("inf")
    best_result = None

    # Sort available monomers by Phase 2 SMD BE (best first)
    smd_sorted = sorted(available_monomers,
                        key=lambda m: smd_be.get(m, 0) or 0)

    logger.info(f"[{target}] Greedy Forward Selection: "
                f"{len(available_monomers)} monomers, max_size={max_size}")
    logger.info(f"  Phase 2 SMD ranking: {smd_sorted}")

    # ── Phase A: Forward Selection ──────────────────────────────
    # Optional: filter out polymerization-incompatible combinations
    # (Liu 2017 Nat. Protoc.: silane + radical cannot be mixed in one pot)
    from .config import (MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY,
                         is_polymerization_compatible)

    for round_k in range(max_size):
        candidates = [m for m in smd_sorted if m not in selected]
        if not candidates:
            break

        logger.info(f"\n  Round {round_k+1}: testing {len(candidates)} "
                    f"candidates (current: {selected})")

        round_best_candidate = None
        round_best_obj = float("inf")
        round_best_result = None

        for candidate in candidates:
            trial = selected + [candidate]

            # Polymerization compatibility filter
            if MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY:
                ok, _, conflicts = is_polymerization_compatible(trial)
                if not ok:
                    logger.debug(f"    skip {candidate}: {conflicts[0]}")
                    continue

            compatible_xls = _get_compatible_crosslinkers(trial)
            pc_id = f"FWD{round_k+1}_{candidate}"

            result = _evaluate_with_selectivity(
                target, pc_id, trial, compatible_xls,
                receptor_pdbqt, epitope_pdb,
                center, npts, work_dir / pc_id,
                smd_be=smd_be,
                ga_runs=ga_runs,
                off_targets=off_targets,
            )
            all_results.append(result)

            obj = result.get("bo_objective")
            avg = result.get("mmsd_per_monomer")
            xl = result.get("crosslinker", "?")
            if obj is not None:
                logger.info(f"    {pc_id}: obj={obj:.3f} avg={avg:.3f} XL={xl}")
                if obj < round_best_obj:
                    round_best_obj = obj
                    round_best_candidate = candidate
                    round_best_result = result

        if round_best_candidate is None:
            logger.warning(f"  Round {round_k+1}: all candidates failed")
            break

        new_avg = round_best_result["mmsd_per_monomer"]

        # Stop if adding monomer worsens avg BE (after minimum size reached)
        if len(selected) >= min_size and new_avg >= best_avg:
            logger.info(f"  STOP at {len(selected)} monomers: "
                        f"adding {round_best_candidate} worsens avg "
                        f"({new_avg:.3f} vs {best_avg:.3f})")
            break

        selected.append(round_best_candidate)
        best_avg = new_avg
        best_obj = round_best_obj
        best_result = round_best_result
        logger.info(f"  → Selected: {selected} "
                    f"(avg={best_avg:.3f}, obj={best_obj:.3f})")

    if not selected:
        logger.warning(f"[{target}] Forward selection failed — no valid combos")
        return all_results

    logger.info(f"\n  Forward selection result: {selected} "
                f"(obj={best_obj:.3f})")

    # ── Phase B: Swap Refinement ────────────────────────────────
    logger.info(f"\n  Swap refinement: testing alternatives for each position")
    swap_round = 0
    improved = True

    while improved:
        improved = False
        swap_round += 1
        logger.info(f"  Swap round {swap_round}:")

        for i, current_mono in enumerate(selected):
            alternatives = [m for m in smd_sorted
                            if m not in selected]

            for alt in alternatives:
                trial = selected[:i] + [alt] + selected[i+1:]
                compatible_xls = _get_compatible_crosslinkers(trial)
                pc_id = f"SWAP{swap_round}_{current_mono}to{alt}"

                result = _evaluate_with_selectivity(
                    target, pc_id, trial, compatible_xls,
                    receptor_pdbqt, epitope_pdb,
                    center, npts, work_dir / pc_id,
                    smd_be=smd_be, ga_runs=ga_runs,
                    off_targets=off_targets,
                )
                all_results.append(result)

                obj = result.get("bo_objective")
                if obj is not None and obj < best_obj:
                    logger.info(f"    SWAP: {current_mono}→{alt} "
                                f"improves obj {best_obj:.3f}→{obj:.3f}")
                    selected[i] = alt
                    best_obj = obj
                    best_result = result
                    improved = True
                    break  # restart position loop
            if improved:
                break

        if not improved:
            logger.info(f"  No more improvements found")

    # ── Summary ─────────────────────────────────────────────────
    logger.info(f"\n[{target}] Greedy result: {selected} + "
                f"{best_result.get('crosslinker', '?')}")
    logger.info(f"  obj={best_obj:.3f}, "
                f"avg={best_result.get('mmsd_per_monomer', '?')}, "
                f"sum={best_result.get('mmsd_sum', '?')}")
    logger.info(f"  Total evaluations: {len(all_results)}")

    return all_results


# ── MMSD Sequential Docking (unchanged from Rajpal 2024) ──────

def _run_single_mmsd(target: str, pc_id: str,
                      functional_monomers: list,
                      compatible_crosslinkers: list,
                      receptor_pdbqt: Path,
                      epitope_pdb: Path,
                      center: tuple, npts: tuple,
                      work_dir: Path,
                      *,
                      smd_be: dict,
                      ga_runs: int = 25,
                      docking_order: list = None) -> dict:
    """
    Run sequential MMSD for a single polymer composition.

    Steps 1~N-1: dock functional monomers sequentially (once).
    Step N: try each compatible crosslinker → pick best by BE.

    `smd_be` is KEYWORD-ONLY and must be the monomer-keyed row
    phase2_results["be_matrix"][target] — see _assert_monomer_keyed_be.

    `docking_order` overrides the step order; when None the canonical order
    for the active MMSD_DOCKING_ORDER policy is used. The order actually
    docked is echoed back in the result as "docking_order".
    """
    from .utils_autodock import dock_single, merge_ligand_into_receptor
    from .utils_structure import prepare_receptor_pdbqt
    # ALL_MONOMERS merges the functional and crosslinker libraries, so one
    # lookup covers both dock_single call sites below (expected_smiles).
    from .config import ALL_MONOMERS

    _assert_monomer_keyed_be(smd_be, f"_run_single_mmsd({pc_id})")

    policy, _, _ = _order_policy()
    if docking_order is None:
        docking_order = _canonical_mmsd_order(functional_monomers, smd_be, policy)
    elif sorted(docking_order) != sorted(functional_monomers):
        raise ValueError(
            f"{pc_id}: docking_order {docking_order} is not a permutation of "
            f"{functional_monomers}")
    # The set is what defines the composition; the order is the protocol knob.
    composition = list(functional_monomers)
    functional_monomers = list(docking_order)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase A: Sequential docking of functional monomers ──────
    monomer_energies = {}
    current_receptor_pdb = epitope_pdb
    current_receptor_pdbqt = receptor_pdbqt

    for step, monomer_name in enumerate(functional_monomers):
        step_dir = work_dir / f"step{step}_{monomer_name}"

        monomer_pdbqt = _get_monomer_pdbqt(monomer_name, work_dir.parent)
        if monomer_pdbqt is None:
            logger.warning(f"  {pc_id} step {step}: {monomer_name} PDBQT "
                           "not found, skipping")
            monomer_energies[monomer_name] = None
            continue

        # expected_smiles: the monomer PDBQT is COPIED into this per-step work
        # directory, and that copy is keyed on name only. dock_single's stale-copy
        # replacement and stamp check only enforce when the expected SMILES is
        # supplied; without it a retired-SMILES copy left by an earlier run is
        # docked and scored as the current molecule.
        dock_result = dock_single(
            current_receptor_pdbqt, monomer_pdbqt,
            center, npts, step_dir,
            ga_runs=ga_runs,
            expected_smiles=(ALL_MONOMERS.get(monomer_name) or {}).get("smiles"),
        )

        be = dock_result.get("mean_cluster_energy")
        if be is None:
            logger.warning(f"    Step {step} {monomer_name}: docking failed")
            monomer_energies[monomer_name] = None
            continue
        monomer_energies[monomer_name] = be

        best_pose = dock_result.get("best_pose_path")
        if best_pose and Path(best_pose).exists():
            merged_pdb = work_dir / f"receptor_step{step}.pdb"
            merge_ligand_into_receptor(
                current_receptor_pdb, Path(best_pose), merged_pdb
            )
            merged_pdbqt = merged_pdb.with_suffix(".pdbqt")
            prepare_receptor_pdbqt(merged_pdb, merged_pdbqt)
            current_receptor_pdb = merged_pdb
            current_receptor_pdbqt = merged_pdbqt

    # Check functional monomer failures
    valid_func = {m: e for m, e in monomer_energies.items() if e is not None}
    if len(valid_func) < len(functional_monomers):
        n_fail = len(functional_monomers) - len(valid_func)
        logger.warning(f"    {pc_id}: {n_fail} functional monomer(s) failed — excluded")
        return {
            "pc_id": pc_id, "monomers": functional_monomers,
            "functional_monomers": composition,
            "docking_order": list(functional_monomers),
            "monomer_energies": monomer_energies,
            "crosslinker": None, "crosslinker_comparison": {},
            "mmsd_sum": None, "smd_sum": None, "delta_sum": None,
            "synergy": None, "is_uniform": False, "excluded": True,
        }

    # ── Phase B: Try each compatible crosslinker at last step ───
    xl_step = len(functional_monomers)
    xl_results = {}

    for xl_name in compatible_crosslinkers:
        xl_step_dir = work_dir / f"step{xl_step}_{xl_name}"

        xl_pdbqt = _get_monomer_pdbqt(xl_name, work_dir.parent)
        if xl_pdbqt is None:
            logger.warning(f"    {pc_id}: {xl_name} PDBQT not found")
            xl_results[xl_name] = None
            continue

        xl_dock = dock_single(
            current_receptor_pdbqt, xl_pdbqt,
            center, npts, xl_step_dir,
            ga_runs=ga_runs,
            expected_smiles=(ALL_MONOMERS.get(xl_name) or {}).get("smiles"),
        )
        xl_be = xl_dock.get("mean_cluster_energy")
        xl_results[xl_name] = xl_be
        if xl_be is not None:
            logger.info(f"    XL {xl_name}: BE={xl_be:.2f}")
        else:
            logger.warning(f"    XL {xl_name}: docking failed")

    # Pick best crosslinker
    valid_xls = {k: v for k, v in xl_results.items() if v is not None}
    if not valid_xls:
        logger.warning(f"    {pc_id}: all crosslinkers failed — excluded")
        return {
            "pc_id": pc_id, "monomers": functional_monomers,
            "functional_monomers": composition,
            "docking_order": list(functional_monomers),
            "monomer_energies": monomer_energies,
            "crosslinker": None, "crosslinker_comparison": xl_results,
            "mmsd_sum": None, "smd_sum": None, "delta_sum": None,
            "synergy": None, "is_uniform": False, "excluded": True,
        }

    best_xl = min(valid_xls, key=valid_xls.get)
    best_xl_be = valid_xls[best_xl]
    logger.info(f"    Best XL: {best_xl} (BE={best_xl_be:.2f})")

    # ── Compute final MMSD metrics ──────────────────────────────
    all_monomers = functional_monomers + [best_xl]
    monomer_energies[best_xl] = best_xl_be

    metrics = _finalize_mmsd_metrics(
        pc_id=pc_id, all_monomers=all_monomers,
        monomer_energies=monomer_energies, smd_be=smd_be,
    )

    # Competition analysis (Sullivan 2019)
    from .config import MMSD_PENALIZE_COMPETITION, MMSD_COMPETITION_DISTANCE
    competition = {"is_uniform": True, "n_competing": 0}
    monomer_centers = {}

    if MMSD_PENALIZE_COMPETITION:
        for step, m_name in enumerate(functional_monomers):
            pose_path = work_dir / f"step{step}_{m_name}" / f"{m_name}_best.pdbqt"
            if pose_path.exists():
                from .utils_structure import compute_grid_center
                monomer_centers[m_name] = compute_grid_center(pose_path)

        if len(monomer_centers) >= 2:
            from .utils_analysis import analyze_competition
            competition = analyze_competition(
                monomer_centers,
                distance_threshold=MMSD_COMPETITION_DISTANCE,
            )

    result = {
        "pc_id": pc_id,
        "monomers": all_monomers,
        # BLOCKER 05(b): every downstream matcher (_full_compat, the BO
        # best-result lookup, the NSGA-II Pareto-front join) looked up
        # "functional_monomers" — a key _run_single_mmsd never emitted. The
        # Pareto join therefore matched nothing and published mmsd_result:null
        # for every front entry. It is emitted now.
        "functional_monomers": composition,
        "docking_order": list(functional_monomers),
        "docking_order_rule": _order_policy()[0],
        "n_functional": len(functional_monomers),
        "crosslinker": best_xl,
        "crosslinker_comparison": {k: round(v, 3) if v is not None else None
                                   for k, v in xl_results.items()},
        "monomer_energies": monomer_energies,
        "competition": competition,
        "is_uniform": competition.get("is_uniform", True),
    }
    result.update(metrics)
    return result


# ── MMSD metric finalisation (single source of the formulas) ───

def _finalize_mmsd_metrics(pc_id: str, all_monomers: list,
                           monomer_energies: dict, smd_be: dict) -> dict:
    """Derive mmsd_sum / smd_sum / delta_sum / synergy / bo_objective.

    BLOCKER 05(a), second half. `smd_sum` is a LIKE-FOR-LIKE reference:
    Phase 2 only docks FUNCTIONAL monomers, so crosslinkers (TEOS/TMOS/DVB/…)
    have no SMD entry. The v1 formula summed MMSD over all monomers including
    the crosslinker but summed SMD only over the monomers it found, which
    biases delta_sum negative by the whole crosslinker BE (3-5 kcal/mol) and
    manufactures "synergy" on its own — independently of the dict-shape bug.

    delta_sum is therefore computed over the COVERED subset only: the monomers
    that have both an MMSD step energy and an SMD reference. mmsd_sum keeps its
    original meaning (full composition, incl. crosslinker) because that is the
    affinity term the objective ranks on.

    CHANGED DEFAULT (loudly): delta_sum can now be POSITIVE, so the
    BO_INTERFERENCE_PENALTY term in bo_objective is live for the first time.
    In v1 delta_sum was negative for 213/213 records, so the penalty never fired.
    """
    from .config import BO_INTERFERENCE_PENALTY

    def _num(v):
        return (isinstance(v, (int, float)) and not isinstance(v, bool))

    mmsd_sum = sum(v for m, v in monomer_energies.items()
                   if m in all_monomers and _num(v))

    covered = [m for m in all_monomers
               if _num(monomer_energies.get(m)) and _num(smd_be.get(m))]
    missing = [m for m in all_monomers if m not in covered]

    if covered:
        mmsd_covered = sum(monomer_energies[m] for m in covered)
        smd_covered = sum(smd_be[m] for m in covered)
        delta_sum = mmsd_covered - smd_covered
        synergy = bool(delta_sum < 0)
        penalty = BO_INTERFERENCE_PENALTY * max(0.0, delta_sum)
    else:
        # No SMD reference at all — refuse to invent one. delta/synergy are
        # unavailable (None, not False), and the interference penalty is
        # explicitly recorded as NOT applied instead of silently being 0.
        logger.error(
            f"    {pc_id}: no Phase 2 SMD reference for ANY of {all_monomers} — "
            f"delta_sum/synergy reported as null, interference penalty NOT applied.")
        mmsd_covered = smd_covered = None
        delta_sum = None
        synergy = None
        penalty = 0.0

    if missing:
        logger.debug(f"    {pc_id}: no SMD reference for {missing} "
                     f"(expected for crosslinkers) — excluded from delta_sum")

    n_mono = len(all_monomers)
    mmsd_per_monomer = mmsd_sum / n_mono if n_mono else None
    bo_objective = mmsd_per_monomer + penalty if mmsd_per_monomer is not None else None

    return {
        "mmsd_sum": round(mmsd_sum, 3),
        "mmsd_sum_covered": round(mmsd_covered, 3) if mmsd_covered is not None else None,
        "smd_sum": round(smd_covered, 3) if smd_covered is not None else None,
        "smd_reference_monomers": covered,
        "smd_reference_missing": missing,
        "smd_reference_available": bool(covered),
        "delta_sum": round(delta_sum, 3) if delta_sum is not None else None,
        "mmsd_per_monomer": round(mmsd_per_monomer, 3),
        "interference_penalty": round(penalty, 3),
        "interference_penalty_applied": bool(covered),
        "bo_objective": round(bo_objective, 3),
        "synergy": synergy,
    }


# ── Helpers ────────────────────────────────────────────────────

def _library_stamp(name: str) -> str:
    """SHA-256 (16 hex) of the CURRENT library SMILES for `name`.

    Any cached PDBQT that does not carry this stamp was built from a different
    SMILES and must not be reused. Nine silanes, APBA, AAPBA and TRIM all had
    their SMILES corrected this cycle; 753 stale PDBQTs of the retired strings
    were sitting in the results tree.
    """
    from .config import ALL_MONOMERS
    entry = ALL_MONOMERS.get(name)
    if not entry or not entry.get("smiles"):
        return ""
    return hashlib.sha256(entry["smiles"].encode("utf-8")).hexdigest()[:16]


def _stamp_path(pdbqt: Path) -> Path:
    return Path(pdbqt).with_suffix(".pdbqt.smiles.json")


def _read_stamp(pdbqt: Path) -> str:
    """Stamp recorded beside a cached PDBQT, or "" if unstamped."""
    sp = _stamp_path(pdbqt)
    if not sp.exists():
        return ""
    try:
        return json.loads(sp.read_text(encoding="utf-8")).get("smiles_sha256", "")
    except Exception:
        return ""


def _write_stamp(pdbqt: Path, name: str):
    from .config import ALL_MONOMERS
    try:
        _stamp_path(pdbqt).write_text(json.dumps({
            "name": name,
            "smiles": ALL_MONOMERS.get(name, {}).get("smiles"),
            "smiles_sha256": _library_stamp(name),
        }, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"    Could not write SMILES stamp for {name}: {e}")


# Per-process memo so the stamp check runs once per monomer, not once per step.
_PDBQT_RESOLVED = {}


def _get_monomer_pdbqt(name: str, search_dir: Path) -> Path:
    """Resolve a monomer PDBQT with an EXPLICIT, stamp-checked search order.

    v1 called `Path(search_dir).rglob(f"{name}.pdbqt")` and returned
    `candidates[0]` — an arbitrary filesystem-order hit anywhere under the
    results tree. That preferred 753 stale copies built from retired SMILES
    over regenerating from the corrected library.

    Order is now fixed and every hit must carry a SMILES stamp matching the
    CURRENT library entry:
      1. Phase 3 monomer cache   (results/phase3/monomers/NAME.pdbqt)
      2. Phase 2 monomer dir     (results/phase2/monomers/NAME.pdbqt)
      3. this run's work tree    (rglob under search_dir, sorted — deterministic)
      4. regenerate from ALL_MONOMERS[name]["smiles"] into the Phase 3 cache

    A hit whose stamp is absent or different is REJECTED, not used: an
    unstamped file is indistinguishable from a retired-SMILES file, and
    regenerating costs seconds while docking the wrong molecule costs the run.
    If regeneration then fails, this returns None so the caller excludes the
    composition — a loud miss, never a silent wrong ligand.
    """
    from .config import get_output_path, ALL_MONOMERS

    expected = _library_stamp(name)
    memo_key = (name, expected, str(search_dir))
    hit = _PDBQT_RESOLVED.get(memo_key)
    if hit is not None and Path(hit).exists():
        return Path(hit)

    if not expected:
        logger.error(f"    {name}: no SMILES in the monomer library — cannot "
                     f"verify or regenerate a PDBQT. Refusing to guess.")
        return None

    p3_monomers = get_output_path("phase3") / "monomers"
    ordered = [p3_monomers / f"{name}.pdbqt",
               get_output_path("phase2") / "monomers" / f"{name}.pdbqt"]
    ordered += sorted(Path(search_dir).rglob(f"{name}.pdbqt"))

    seen = set()
    rejected = []
    for cand in ordered:
        cand = Path(cand)
        if cand in seen or not cand.exists():
            continue
        seen.add(cand)
        found = _read_stamp(cand)
        if found == expected:
            _PDBQT_RESOLVED[memo_key] = str(cand)
            return cand
        rejected.append((cand, found or "UNSTAMPED"))

    if rejected:
        logger.warning(
            f"    {name}: rejected {len(rejected)} cached PDBQT(s) whose SMILES "
            f"stamp != current library ({expected}); regenerating. "
            f"First: {rejected[0][0]} [{rejected[0][1]}]")

    if name not in ALL_MONOMERS:
        logger.error(f"    {name} is not in ALL_MONOMERS — cannot regenerate.")
        return None

    from .utils_structure import smiles_to_pdbqt
    logger.info(f"    Generating PDBQT for {name} from current library SMILES")
    p3_monomers.mkdir(parents=True, exist_ok=True)
    try:
        out = Path(smiles_to_pdbqt(ALL_MONOMERS[name]["smiles"], name, p3_monomers))
    except Exception as e:
        logger.error(f"    Failed to generate {name} PDBQT: {e} — composition "
                     f"will be EXCLUDED rather than docked with a stale file.")
        return None
    _write_stamp(out, name)
    _PDBQT_RESOLVED[memo_key] = str(out)
    return out


def _plot_mmsd_comparison(results: dict, output_dir: Path):
    """Generate BO optimization and ranking plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for target, data in results.items():
            if "error" in data:
                continue

            all_results = data.get("all_results", [])
            valid = [r for r in all_results
                     if r.get("mmsd_sum") is not None]
            if not valid:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # BO convergence plot
            mmsd_values = [r["mmsd_sum"] for r in valid]
            best_so_far = []
            current_best = float("inf")
            for v in mmsd_values:
                current_best = min(current_best, v)
                best_so_far.append(current_best)

            ax1.plot(range(1, len(mmsd_values) + 1), mmsd_values,
                     "o", alpha=0.5, label="MMSD sum", color="steelblue")
            ax1.plot(range(1, len(best_so_far) + 1), best_so_far,
                     "-", color="red", linewidth=2, label="Best so far")
            ax1.set_xlabel("BO Iteration")
            ax1.set_ylabel("MMSD Sum (kcal/mol)")
            ax1.set_title(f"{target}: Greedy Selection Convergence")
            ax1.legend()

            # Top 10 bar chart
            sorted_valid = sorted(valid, key=lambda x: x["mmsd_sum"])[:10]
            names = ["+".join(r["monomers"][:-1]) for r in sorted_valid]
            values = [r["mmsd_sum"] for r in sorted_valid]
            n_monos = [r.get("n_functional", len(r["monomers"])-1)
                       for r in sorted_valid]

            colors = ["#2ecc71" if n <= 3 else "#3498db" if n <= 4
                      else "#e74c3c" for n in n_monos]

            ax2.barh(range(len(sorted_valid)), values, color=colors)
            ax2.set_yticks(range(len(sorted_valid)))
            ax2.set_yticklabels([f"{n} ({nm}種)" for n, nm in zip(names, n_monos)],
                                fontsize=7)
            ax2.set_xlabel("MMSD Sum (kcal/mol)")
            ax2.set_title(f"{target}: Top 10 PCs (green=3, blue=4, red=5+)")
            ax2.invert_yaxis()

            plt.tight_layout()
            plt.savefig(output_dir / f"phase3_{target}_bo.png", dpi=150)
            plt.close()

        logger.info(f"BO plots saved → {output_dir}")
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")


# ════════════════════════════════════════════════════════════════
# A4: Bayesian Optimization (alternative to greedy)
# ════════════════════════════════════════════════════════════════

def _run_bayesian_mmsd(target, available_monomers, receptor_pdbqt,
                       epitope_pdb, center, npts, work_dir,
                       ga_runs, *, smd_be, n_calls=30, min_size=None, max_size=None,
                       off_targets=None):
    """A4: Gaussian Process Bayesian Optimization over monomer combinations.

    Replaces greedy forward+swap with skopt.gp_minimize. Sample-efficient:
    30 GP evaluations approximate 720 greedy trials.

    `smd_be` is KEYWORD-ONLY: the monomer-keyed Phase 2 row for `target`.

    Reference: Garcia-Ortegon 2022 DOCKSTRING JCIM; Frazier 2018 arXiv.
    """
    from .config import (BAYESIAN_N_CALLS, BAYESIAN_ACQUISITION,
                         MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY,
                         is_polymerization_compatible)

    _lo, _hi = _combo_size_bounds()      # REVIEW FINDING 18
    min_size = _lo if min_size is None else int(min_size)
    max_size = _hi if max_size is None else int(max_size)

    _assert_monomer_keyed_be(smd_be, f"_run_bayesian_mmsd({target})")
    policy, _, _ = _order_policy()

    try:
        from skopt import gp_minimize
        from skopt.space import Categorical
    except ImportError:
        logger.warning("skopt not installed; falling back to greedy. "
                       "pip install scikit-optimize")
        return None

    # Search space: 6 positions, each can be one of monomers or "NONE"
    monomer_choices = available_monomers + ["NONE"]
    space = [Categorical(monomer_choices, name=f"pos_{i}") for i in range(max_size)]

    all_results = []
    eval_cache = {}  # avoid re-evaluating same combo

    def objective(combo_raw):
        # Remove duplicates and NONE; require min_size
        combo = []
        for m in combo_raw:
            if m != "NONE" and m not in combo:
                combo.append(m)
        if len(combo) < min_size:
            return 0.0  # penalty for too small
        if MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY:
            ok, _, _ = is_polymerization_compatible(combo)
            if not ok:
                return 0.0
        # Cache on the ORDERED tuple actually docked, not the sorted set.
        key = _cache_key(combo, smd_be, policy)
        if key in eval_cache:
            return eval_cache[key]

        compatible_xls = _get_compatible_crosslinkers(combo)
        pc_id = f"BO_{len(all_results)+1}_{combo[0]}"
        result = _evaluate_with_selectivity(
            target, pc_id, combo, compatible_xls,
            receptor_pdbqt, epitope_pdb, center, npts,
            work_dir / pc_id, smd_be=smd_be, ga_runs=ga_runs,
            off_targets=off_targets,
        )
        all_results.append(result)
        obj = result.get("bo_objective", 0.0) or 0.0
        eval_cache[key] = obj
        logger.info(f"  BO[{len(all_results)}/{n_calls}]: {combo} → obj={obj:.3f}")
        return obj

    logger.info(f"\n  Bayesian Optimization ({n_calls} GP evaluations):")
    res = gp_minimize(
        objective, space,
        n_calls=n_calls,
        acq_func=BAYESIAN_ACQUISITION,
        random_state=42,
        n_initial_points=8,  # random Sobol points before GP fit
    )
    best_combo_raw = res.x
    best_combo = list(dict.fromkeys([m for m in best_combo_raw if m != "NONE"]))
    best_obj = res.fun
    logger.info(f"  BO BEST: {best_combo} (obj={best_obj:.3f})")

    # Find the matching result. `functional_monomers` is now actually emitted
    # by _run_single_mmsd — before BLOCKER 05(b) this key never existed, so the
    # lookup silently returned best_result=None.
    best_result = None
    want = tuple(sorted(best_combo))
    for r in all_results:
        if tuple(sorted(r.get("functional_monomers") or [])) == want:
            best_result = r
            break
    if best_result is None:
        logger.error(
            f"  BO best combo {best_combo} has no matching evaluation record — "
            f"downstream phases would receive best_result=None.")
    return {"selected": best_combo, "best_result": best_result,
            "all_results": all_results, "n_evaluations": len(all_results)}


# ════════════════════════════════════════════════════════════════
# B5: DFT Validation Hook (Top-N PC refinement)
# ════════════════════════════════════════════════════════════════

def _dft_validate_top_combinations(top_pcs, target, work_dir,
                                   level=None, top_n=None):
    """B5: DFT refinement of top Phase 3 PCs (M06-2X by default).

    Currently a stub interface — full DFT requires Psi4 or Gaussian installation.
    Records computed DFT energies for ranking validation; falls back to a
    GFN2-xTB single-point estimate if Psi4 unavailable (much faster, lower accuracy).

    Reference: Khan 2024 (J Mol Graph Model); Boulanger 2019 (JCTC).
    """
    from .config import DFT_VALIDATION_TOP_N, DFT_LEVEL, DFT_SOLVENT
    level = level or DFT_LEVEL
    top_n = top_n or DFT_VALIDATION_TOP_N

    out_dir = Path(work_dir) / "dft_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    refined = []

    # Try Psi4 first (full DFT), else fall back to xTB
    use_psi4 = False
    try:
        import psi4  # noqa
        use_psi4 = True
    except ImportError:
        pass

    for i, pc in enumerate(top_pcs[:top_n]):
        pc_dir = out_dir / pc.get("pc_id", f"top_{i+1}")
        pc_dir.mkdir(exist_ok=True)
        entry = {
            "pc_id": pc.get("pc_id"),
            "monomers": pc.get("monomers"),
            "phase3_mmsd_sum": pc.get("mmsd_sum"),
            "phase3_bo_objective": pc.get("bo_objective"),
            "dft_method": level if use_psi4 else "GFN2-xTB (xtb)",
            "dft_solvent": DFT_SOLVENT,
            "dft_energy_kcal_mol": None,
            "status": "stub" if not use_psi4 else "psi4_available",
        }
        if use_psi4:
            entry["status"] = "implement_psi4_run"
            # TODO: load docked pose → Psi4 sp at M06-2X/def2-TZVP
        else:
            # xTB fallback would need atomic coordinates — placeholder for now
            entry["status"] = "psi4_missing_fallback_to_xtb_stub"
            entry["note"] = (
                "DFT validation hook installed. Full DFT requires "
                "`pip install psi4` or Gaussian. Currently a structured stub "
                "— interface ready for downstream integration."
            )
        refined.append(entry)
    out_file = out_dir / "dft_validation_summary.json"
    out_file.write_text(json.dumps(refined, indent=2), encoding="utf-8")
    logger.info(f"DFT validation hook → {out_file} ({len(refined)} entries, status: stub)")
    return refined


# ════════════════════════════════════════════════════════════════
# C2: NSGA-II Multi-Objective Optimization
# (Affinity + Selectivity + Synthesizability)
# ════════════════════════════════════════════════════════════════

def _run_nsga2_mmsd(target, available_monomers, receptor_pdbqt,
                    epitope_pdb, center, npts, work_dir,
                    ga_runs, all_targets, *,
                    smd_be, be_matrix_all,
                    pop_size=20, n_gen=15, min_size=None, max_size=None,
                    off_targets=None):
    """C2: NSGA-II multi-objective optimization.

    Three objectives (all minimized):
      1. Affinity:        mmsd_per_monomer (more negative = better)
      2. Selectivity:     -selectivity_score (higher = better)
      3. Synthesizability: -synth_score/10 (higher = better)

    Two DISTINCT binding-energy structures are needed here, and conflating
    them is BLOCKER 05(a). Both are KEYWORD-ONLY and shape-asserted:
      smd_be       — phase2_results["be_matrix"][target], {monomer: BE}.
                     The SMD reference for the MMSD-vs-SMD synergy statistic.
      be_matrix_all— phase2_results["be_matrix"], {target: {monomer: BE}}.
                     Needed by compute_3obj_for_combo for cross-target ΔΔG.
    v1 passed the FULL matrix into the smd_be slot positionally.

    Returns Pareto front of non-dominated monomer combinations, each entry
    carrying its full mmsd_result (BLOCKER 05(b)).
    Reference: Deb 2002 NSGA-II; Garcia-Ortegon 2022 DOCKSTRING.
    """
    from .config import (MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY,
                         is_polymerization_compatible,
                         MMSD_REQUIRE_CHEMISTRY_DIVERSITY,
                         MMSD_CHEMISTRY_ENTROPY_WEIGHT)

    _lo, _hi = _combo_size_bounds()      # REVIEW FINDING 18
    min_size = _lo if min_size is None else int(min_size)
    max_size = _hi if max_size is None else int(max_size)

    _assert_monomer_keyed_be(smd_be, f"_run_nsga2_mmsd({target}) smd_be")
    _assert_target_keyed_be(be_matrix_all, f"_run_nsga2_mmsd({target}) be_matrix_all")
    policy, _, _ = _order_policy()
    from .utils_analysis import (compute_3obj_for_combo,
                                  compute_synthesizability_score,
                                  compute_selectivity_score,
                                  check_chemistry_diversity)

    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.operators.crossover.pntx import PointCrossover
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.repair.rounding import RoundingRepair
        from pymoo.optimize import minimize
    except ImportError:
        logger.warning("pymoo not installed; falling back to greedy. "
                       "pip install pymoo")
        return None

    monomer_pool = available_monomers + ["NONE"]
    n_choices = len(monomer_pool)
    eval_cache = {}
    pareto_results = []
    # key (ordered docking tuple) -> the mmsd_result recorded for it, so the
    # Pareto front can be joined back to real evaluations instead of scanning
    # a list for a key that did not exist.
    result_by_key = {}

    class MIPMultiObjProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=max_size,
                n_obj=3,
                xl=0,
                xu=n_choices - 1,
                vtype=int,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            # Decode integer array → monomer combo
            combo = []
            for idx in x:
                m = monomer_pool[int(idx)]
                if m != "NONE" and m not in combo:
                    combo.append(m)

            # Invalid: too few or too many
            if len(combo) < min_size:
                out["F"] = [1e6, 1e6, 1e6]
                return

            # Polymerization compatibility filter
            if MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY:
                ok, _, _ = is_polymerization_compatible(combo)
                if not ok:
                    out["F"] = [1e6, 1e6, 1e6]
                    return

            # Chemistry diversity filter (Mavliutova 2021, Liu 2017, Cleland 2022)
            # Rule 1: ≥3 chemistry classes among non-XL monomers
            # Rule 2: single class ≤50% of non-XL monomers
            if MMSD_REQUIRE_CHEMISTRY_DIVERSITY:
                div = check_chemistry_diversity(combo)
                if not div["valid"]:
                    out["F"] = [1e6, 1e6, 1e6]
                    return

            # Cache key = the ORDERED tuple actually docked (see _cache_key).
            key = _cache_key(combo, smd_be, policy)
            if key in eval_cache:
                out["F"] = eval_cache[key]
                return

            # Evaluate via MMSD run (with cross-target ΔΔG selectivity penalty)
            compatible_xls = _get_compatible_crosslinkers(combo)
            pc_id = f"NSGA_{len(pareto_results)+1}_{combo[0]}"
            try:
                mmsd_result = _evaluate_with_selectivity(
                    target, pc_id, combo, compatible_xls,
                    receptor_pdbqt, epitope_pdb, center, npts,
                    work_dir / pc_id, smd_be=smd_be, ga_runs=ga_runs,
                    off_targets=off_targets,
                )
            except Exception as e:
                # Shape/plumbing errors must not be swallowed as "bad combo" —
                # that is how BLOCKER 05(a) would have hidden itself again.
                if isinstance(e, (TypeError, ValueError)):
                    raise
                logger.warning(f"NSGA eval failed for {combo}: {e}")
                out["F"] = [1e6, 1e6, 1e6]
                eval_cache[key] = [1e6, 1e6, 1e6]
                return

            # Compute 3 objectives (needs the FULL target-keyed matrix)
            objs = compute_3obj_for_combo(
                combo, target, mmsd_result, be_matrix_all, all_targets)
            f_values = list(objs["objectives"])

            # Override selectivity objective with cross-MMSD ΔΔG when available
            # (more accurate than Phase 2 per-monomer ΔΔG)
            ddg = mmsd_result.get("DDG_selectivity")
            if ddg is not None:
                # Clamp [-5, 5] → score [0, 5] (mirrors compute_selectivity_score)
                sel_score_cross = max(0.0, min(5.0, -ddg))
                f_values[1] = -sel_score_cross  # minimize negative score
                objs["raw"]["selectivity_score_cross_mmsd"] = round(sel_score_cross, 2)
                objs["raw"]["selectivity_mean_ddg_cross"] = round(ddg, 2)

            # Chemistry entropy bonus (Rule 3 — Cleland 2022, Mavliutova 2021)
            # Higher diversity → better recognition specificity → reward in selectivity obj
            if MMSD_REQUIRE_CHEMISTRY_DIVERSITY:
                div = check_chemistry_diversity(combo)
                ent_bonus = MMSD_CHEMISTRY_ENTROPY_WEIGHT * div["normalized_entropy"]
                f_values[1] -= ent_bonus  # bonus to selectivity objective (minimize)
                objs["raw"]["chemistry_classes"] = div["classes"]
                objs["raw"]["chemistry_entropy"] = div["entropy"]
                objs["raw"]["chemistry_normalized_entropy"] = div["normalized_entropy"]

            eval_cache[key] = f_values
            out["F"] = f_values

            # Record full result
            mmsd_result.update({
                "objectives": f_values,
                "objective_details": objs["raw"],
                "pc_id": pc_id,
            })
            pareto_results.append(mmsd_result)
            result_by_key[key] = mmsd_result

            ddg_str = f", ΔΔG={ddg:+.2f}" if ddg is not None else ""
            logger.info(f"  NSGA[{len(pareto_results)}]: {combo} → "
                        f"aff={f_values[0]:.2f}, sel={-f_values[1]:.2f}, "
                        f"synth={-f_values[2]*10:.1f}/10{ddg_str}")

    logger.info(f"\n  NSGA-II Multi-Objective Optimization:")
    logger.info(f"    Pop size={pop_size}, Generations={n_gen}, "
                f"Total evals ≤ {pop_size * n_gen}")

    problem = MIPMultiObjProblem()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=PointCrossover(n_points=2),
        mutation=PolynomialMutation(prob=0.15, eta=15, vtype=int,
                                     repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem, algorithm,
        termination=("n_gen", n_gen),
        seed=42, verbose=False,
    )

    # Extract Pareto front
    pareto_X = res.X  # Pareto-optimal integer arrays
    pareto_F = res.F  # Pareto-optimal objective values

    # ── BLOCKER 05(b): populate every front entry with its evaluation ──
    # v1 joined on r["functional_monomers"], a key _run_single_mmsd never
    # emitted, so `matching` was None for every entry, every front entry was
    # published with mmsd_result:null, run_phase3 skipped them all, and the
    # final pick silently collapsed to argmin(bo_objective) over all_results —
    # i.e. a single scalar instead of the three objectives NSGA-II optimised.
    pareto_combos = []
    n_unmatched = 0
    if pareto_X is None or pareto_F is None or len(np.atleast_2d(pareto_F)) == 0:
        logger.error(f"[{target}] NSGA-II returned an EMPTY Pareto front.")
        pareto_X, pareto_F = [], []
    pareto_X = np.atleast_2d(pareto_X) if len(pareto_X) else []
    pareto_F = np.atleast_2d(pareto_F) if len(pareto_F) else []

    for x, f in zip(pareto_X, pareto_F):
        combo = list(dict.fromkeys(
            [monomer_pool[int(idx)] for idx in x
             if monomer_pool[int(idx)] != "NONE"]))
        if len(combo) < min_size:
            continue
        key = _cache_key(combo, smd_be, policy)
        matching = result_by_key.get(key)
        if matching is None:
            # Second chance: join on the functional-monomer SET.
            want = tuple(sorted(combo))
            for r in pareto_results:
                if tuple(sorted(r.get("functional_monomers") or [])) == want:
                    matching = r
                    break
        if matching is None:
            n_unmatched += 1
            logger.error(
                f"[{target}] Pareto entry {combo} has NO evaluation record — "
                f"it cannot be selected from and is dropped. This is the "
                f"BLOCKER 05(b) signature; do not ignore it.")
            continue
        pareto_combos.append({
            "monomers": combo,
            "objectives": {
                "affinity_mmsd_per": float(f[0]),
                "selectivity_score": float(-f[1]),
                "synthesizability_score": float(-f[2] * 10),
            },
            # minimisation-form objectives, kept raw for the selection rule
            "objectives_min_form": [float(f[0]), float(f[1]), float(f[2])],
            "mmsd_result": matching,
        })

    if n_unmatched:
        logger.error(f"[{target}] {n_unmatched}/{len(pareto_F)} Pareto entries "
                     f"were unmatched and dropped.")

    pareto_combos = _rank_pareto_front(pareto_combos, target)

    logger.info(f"\n  Pareto front: {len(pareto_combos)} non-dominated solutions "
                f"(all carry a populated mmsd_result)")
    for i, c in enumerate(pareto_combos[:10]):
        o = c["objectives"]
        logger.info(f"    P{i+1}: {c['monomers']} → "
                    f"affinity={o['affinity_mmsd_per']:.2f}, "
                    f"sel={o['selectivity_score']:.2f}, "
                    f"synth={o['synthesizability_score']:.1f}/10, "
                    f"rank_score={c['pareto_rank_score']:.4f}")

    return {
        "selected_pareto_front": pareto_combos,
        "all_evaluated": pareto_results,
        "n_evaluations": len(pareto_results),
        "n_pareto": len(pareto_combos),
        "n_pareto_unmatched_dropped": n_unmatched,
        "pareto_selection_rule": _PARETO_RULE_DOC,
        "method": "NSGA-II 3-objective",
    }


# ── Pareto-front selection rule (documented, BLOCKER 05(b)) ────

_PARETO_RULE_DOC = (
    "min-max normalised weighted sum (compromise programming). Each of the "
    "three MINIMISATION-form objectives f0=affinity(mmsd_per_monomer), "
    "f1=-selectivity_score, f2=-synth_score/10 is normalised across the front "
    "to n_j=(f_j-min_j)/(max_j-min_j) (0 when the front is degenerate in that "
    "objective), then scored as sum_j w_j*n_j with weights "
    "MMSD_PARETO_WEIGHTS (default (1,1,1) = equal). Lowest score wins; ties "
    "break on raw affinity then pc_id, so the pick is deterministic. "
    "Normalisation is the point: the v1 sort added raw affinity (kcal/mol, "
    "spread ~1) to a 0-5 selectivity score and a 0-10 synthesizability score, "
    "so synthesizability dominated the ordering by construction."
)


def _pareto_weights() -> tuple:
    """(w_affinity, w_selectivity, w_synth); equal weights unless configured."""
    from . import config as _cfg
    w = getattr(_cfg, "MMSD_PARETO_WEIGHTS", (1.0, 1.0, 1.0))
    w = tuple(float(x) for x in w)
    if len(w) != 3:
        raise ValueError(f"MMSD_PARETO_WEIGHTS must have 3 entries, got {w}")
    if sum(abs(x) for x in w) == 0:
        raise ValueError("MMSD_PARETO_WEIGHTS are all zero — no ranking possible")
    return w


def _rank_pareto_front(pareto_combos: list, target: str) -> list:
    """Rank a populated Pareto front by the documented rule (_PARETO_RULE_DOC).

    CHANGED DEFAULT — stated loudly: this replaces v1's
    `affinity - selectivity - synthesizability` raw-unit sum, which mixed
    kcal/mol with two dimensionless 0-5 / 0-10 scores. The ordering of the
    front changes as a result.
    """
    if not pareto_combos:
        return []
    w = _pareto_weights()
    F = np.array([c["objectives_min_form"] for c in pareto_combos], dtype=float)
    lo, hi = F.min(axis=0), F.max(axis=0)
    span = hi - lo
    norm = np.zeros_like(F)
    for j in range(F.shape[1]):
        norm[:, j] = 0.0 if span[j] <= 0 else (F[:, j] - lo[j]) / span[j]
    for c, n in zip(pareto_combos, norm):
        c["objectives_normalized"] = [round(float(v), 4) for v in n]
        c["pareto_rank_score"] = round(float(np.dot(w, n)), 4)
        c["pareto_rank_weights"] = list(w)
        c["pareto_rank_rule"] = _PARETO_RULE_DOC
    pareto_combos.sort(key=lambda c: (
        c["pareto_rank_score"],
        c["objectives"]["affinity_mmsd_per"],
        str((c.get("mmsd_result") or {}).get("pc_id", "")),
    ))
    return pareto_combos
