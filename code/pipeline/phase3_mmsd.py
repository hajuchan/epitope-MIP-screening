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

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)




def run_phase3(phase1_results: dict = None,
               phase2_results: dict = None,
               target_names: list = None,
               output_dir: str = None) -> dict:
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

    for target in target_names:
        logger.info(f"\n{'='*20} Phase 3 Greedy+MMSD: {target} {'='*20}")

        be_matrix = phase2_results["be_matrix"].get(target, {})
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
                be_matrix=phase2_results["be_matrix"],  # full matrix for selectivity
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
                    be_matrix=be_matrix, receptor_pdbqt=receptor_pdbqt,
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
                be_matrix=be_matrix,
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
                    be_matrix=be_matrix, receptor_pdbqt=receptor_pdbqt,
                    epitope_pdb=epitope_pdb, center=center, npts=npts,
                    work_dir=target_dir, ga_runs=min(AUTODOCK4_GA_RUNS, 25),
                    off_targets=off_targets,
                )
        else:  # greedy (default, legacy)
            all_pc_results = _run_greedy_mmsd(
                target=target,
                available_monomers=available,
                be_matrix=be_matrix,
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
        from .config import is_polymerization_compatible

        def _full_compat(r):
            """Check compatibility of full combo (functional + crosslinker)."""
            full = list(r.get("functional_monomers", []) or r.get("monomers", []))
            xl = r.get("crosslinker")
            if xl and xl not in full:
                full.append(xl)
            ok, _, _ = is_polymerization_compatible(full)
            return ok

        if nsga_pareto_front:
            # Pareto front items have format {monomers, objectives, mmsd_result}.
            top_pcs = []
            for p_ in nsga_pareto_front:
                m_res = p_.get("mmsd_result") or {}
                if not m_res:
                    continue
                if not _full_compat(m_res):
                    continue
                top_pcs.append(m_res)
                if len(top_pcs) >= MMSD_TOP_PC:
                    break
            if not top_pcs:
                # No compatible Pareto solutions — fall back to all_pc_results
                # with full compatibility check
                top_pcs = [r for r in all_pc_results
                            if _full_compat(r)
                           ][:MMSD_TOP_PC]
        else:
            top_pcs = [r for r in all_pc_results
                        if _full_compat(r)][:MMSD_TOP_PC] \
                     or all_pc_results[:MMSD_TOP_PC]

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
            "method": method_name,
            "n_evaluations": len(all_pc_results),
            "available_monomers": available,
            "top_pcs": top_pcs,
            "pareto_front": nsga_pareto_front,  # C2: only set if NSGA-II
            "all_results": all_pc_results,
            "dft_validation": dft_refined,
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

    # Save results
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

def _evaluate_with_selectivity(target: str, pc_id: str,
                                functional_monomers: list,
                                compatible_crosslinkers: list,
                                receptor_pdbqt: Path, epitope_pdb: Path,
                                center: tuple, npts: tuple,
                                smd_be: dict, work_dir: Path,
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

    Reference: Garcia-Ortegon 2022 DOCKSTRING JCIM 62:3486.
    """
    from .config import (MMSD_SELECTIVITY_AWARE, SELECTIVITY_WEIGHT,
                         SELECTIVITY_DDG_THRESHOLD)

    own_result = _run_single_mmsd(
        target, pc_id, functional_monomers, compatible_crosslinkers,
        receptor_pdbqt, epitope_pdb, center, npts, smd_be,
        work_dir, ga_runs=ga_runs,
    )

    if own_result.get("excluded") or own_result.get("mmsd_sum") is None:
        own_result["DDG_selectivity"] = None
        own_result["selectivity_penalty"] = 0.0
        own_result["cross_target_be"] = {}
        return own_result

    if not MMSD_SELECTIVITY_AWARE or not off_targets:
        own_result["DDG_selectivity"] = None
        own_result["selectivity_penalty"] = 0.0
        own_result["cross_target_be"] = {}
        return own_result

    # Cross-target MMSD: same combo, different receptors
    off_mmsd_sums = {}
    for off_name, off_meta in off_targets.items():
        if off_name == target:
            continue
        off_dir = work_dir / f"cross_{off_name}"
        try:
            off_result = _run_single_mmsd(
                target, f"{pc_id}_vs_{off_name}",
                functional_monomers, compatible_crosslinkers,
                off_meta["receptor"], off_meta["epitope"],
                off_meta["center"], off_meta["npts"], smd_be,
                off_dir, ga_runs=ga_runs,
            )
            if off_result.get("mmsd_sum") is not None:
                off_mmsd_sums[off_name] = off_result["mmsd_sum"]
        except Exception as e:
            logger.debug(f"  Cross-MMSD {pc_id} vs {off_name} failed: {e}")

    if not off_mmsd_sums:
        own_result["DDG_selectivity"] = None
        own_result["selectivity_penalty"] = 0.0
        own_result["cross_target_be"] = {}
        return own_result

    own_sum = own_result["mmsd_sum"]
    off_mean = float(np.mean(list(off_mmsd_sums.values())))
    ddg = own_sum - off_mean  # negative = own preferred
    # Penalize when ΔΔG exceeds threshold (i.e., own not selective enough).
    # threshold = -1.0 ⇒ require own ≤ off_mean − 1 kcal/mol for zero penalty.
    penalty = SELECTIVITY_WEIGHT * max(0.0, ddg - SELECTIVITY_DDG_THRESHOLD)

    own_result["DDG_selectivity"] = round(ddg, 3)
    own_result["selectivity_penalty"] = round(penalty, 3)
    own_result["cross_target_be"] = {k: round(v, 3) for k, v in off_mmsd_sums.items()}
    if own_result.get("bo_objective") is not None:
        own_result["bo_objective"] = round(own_result["bo_objective"] + penalty, 3)

    logger.info(f"    {pc_id} ΔΔG={ddg:+.2f} kcal/mol, "
                f"penalty={penalty:.2f} → obj={own_result.get('bo_objective')}")
    return own_result


# ── Greedy Forward Selection + Swap Refinement ──────────────

def _run_greedy_mmsd(target: str, available_monomers: list,
                      be_matrix: dict,
                      receptor_pdbqt: Path, epitope_pdb: Path,
                      center: tuple, npts: tuple,
                      work_dir: Path, ga_runs: int = 25,
                      min_size: int = 2, max_size: int = 6,
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

    Uses Phase 2 SMD BE (be_matrix) to sort candidates — best SMD monomers
    are tried first for efficiency.
    """
    from .config import MMSD_MIN_COMBO_SIZE

    all_results = []
    selected = []
    best_avg = float("inf")
    best_obj = float("inf")
    best_result = None

    # Sort available monomers by Phase 2 SMD BE (best first)
    smd_sorted = sorted(available_monomers,
                        key=lambda m: be_matrix.get(m, 0) or 0)

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
                center, npts, be_matrix,
                work_dir / pc_id,
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
                    center, npts, be_matrix,
                    work_dir / pc_id, ga_runs=ga_runs,
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
                      smd_be: dict,
                      work_dir: Path,
                      ga_runs: int = 25) -> dict:
    """
    Run sequential MMSD for a single polymer composition.

    Steps 1~N-1: dock functional monomers sequentially (once).
    Step N: try each compatible crosslinker → pick best by BE.
    """
    from .utils_autodock import dock_single, merge_ligand_into_receptor
    from .utils_structure import prepare_receptor_pdbqt

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

        dock_result = dock_single(
            current_receptor_pdbqt, monomer_pdbqt,
            center, npts, step_dir,
            ga_runs=ga_runs,
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
            "monomer_energies": monomer_energies,
            "crosslinker": None, "crosslinker_comparison": {},
            "mmsd_sum": None, "smd_sum": None, "delta_sum": None,
            "synergy": False, "is_uniform": False, "excluded": True,
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
            "monomer_energies": monomer_energies,
            "crosslinker": None, "crosslinker_comparison": xl_results,
            "mmsd_sum": None, "smd_sum": None, "delta_sum": None,
            "synergy": False, "is_uniform": False, "excluded": True,
        }

    best_xl = min(valid_xls, key=valid_xls.get)
    best_xl_be = valid_xls[best_xl]
    logger.info(f"    Best XL: {best_xl} (BE={best_xl_be:.2f})")

    # ── Compute final MMSD metrics ──────────────────────────────
    all_monomers = functional_monomers + [best_xl]
    monomer_energies[best_xl] = best_xl_be

    mmsd_sum = sum(valid_func.values()) + best_xl_be
    smd_sum = sum(smd_be.get(m, 0.0) for m in all_monomers
                  if smd_be.get(m) is not None)
    delta_sum = mmsd_sum - smd_sum

    # Size-normalized BO objective
    from .config import BO_INTERFERENCE_PENALTY
    n_mono = len(all_monomers)
    mmsd_per_monomer = mmsd_sum / n_mono
    bo_objective = (mmsd_per_monomer
                    + BO_INTERFERENCE_PENALTY * max(0, delta_sum))

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

    return {
        "pc_id": pc_id,
        "monomers": all_monomers,
        "n_functional": len(functional_monomers),
        "crosslinker": best_xl,
        "crosslinker_comparison": {k: round(v, 3) if v is not None else None
                                   for k, v in xl_results.items()},
        "monomer_energies": monomer_energies,
        "mmsd_sum": round(mmsd_sum, 3),
        "smd_sum": round(smd_sum, 3),
        "delta_sum": round(delta_sum, 3),
        "mmsd_per_monomer": round(mmsd_per_monomer, 3),
        "bo_objective": round(bo_objective, 3),
        "synergy": delta_sum < 0,
        "competition": competition,
        "is_uniform": competition.get("is_uniform", True),
    }


# ── Helpers ────────────────────────────────────────────────────

def _get_monomer_pdbqt(name: str, search_dir: Path) -> Path:
    """Find monomer PDBQT file, or generate it if not found."""
    # 1. Search in work directory tree
    candidates = list(Path(search_dir).rglob(f"{name}.pdbqt"))
    if candidates:
        return candidates[0]

    # 2. Check Phase 2 monomer directory
    from .config import get_output_path
    p2_monomers = get_output_path("phase2") / "monomers"
    candidate = p2_monomers / f"{name}.pdbqt"
    if candidate.exists():
        return candidate

    # 3. Check Phase 3 monomer directory (crosslinker cache)
    p3_monomers = get_output_path("phase3") / "monomers"
    candidate = p3_monomers / f"{name}.pdbqt"
    if candidate.exists():
        return candidate

    # 4. Generate on-the-fly (for crosslinkers like TEOS not in Phase 2)
    from .config import ALL_MONOMERS
    if name in ALL_MONOMERS:
        from .utils_structure import smiles_to_pdbqt
        logger.info(f"    Generating PDBQT for {name} (not in Phase 2 monomers)")
        gen_dir = get_output_path("phase3") / "monomers"
        gen_dir.mkdir(parents=True, exist_ok=True)
        try:
            return smiles_to_pdbqt(ALL_MONOMERS[name]["smiles"], name, gen_dir)
        except Exception as e:
            logger.error(f"    Failed to generate {name} PDBQT: {e}")

    return None


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
                       epitope_pdb, center, npts, be_matrix, work_dir,
                       ga_runs, n_calls=30, min_size=2, max_size=6,
                       off_targets=None):
    """A4: Gaussian Process Bayesian Optimization over monomer combinations.

    Replaces greedy forward+swap with skopt.gp_minimize. Sample-efficient:
    30 GP evaluations approximate 720 greedy trials.

    Reference: Garcia-Ortegon 2022 DOCKSTRING JCIM; Frazier 2018 arXiv.
    """
    from .config import (BAYESIAN_N_CALLS, BAYESIAN_ACQUISITION,
                         MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY,
                         is_polymerization_compatible)

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
        key = tuple(sorted(combo))
        if key in eval_cache:
            return eval_cache[key]

        compatible_xls = _get_compatible_crosslinkers(combo)
        pc_id = f"BO_{len(all_results)+1}_{combo[0]}"
        result = _evaluate_with_selectivity(
            target, pc_id, combo, compatible_xls,
            receptor_pdbqt, epitope_pdb, center, npts, be_matrix,
            work_dir / pc_id, ga_runs=ga_runs,
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

    # Find the matching result
    best_result = None
    for r in all_results:
        if tuple(sorted(r.get("functional_monomers", []))) == tuple(sorted(best_combo)):
            best_result = r
            break
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
                    epitope_pdb, center, npts, be_matrix, work_dir,
                    ga_runs, all_targets,
                    pop_size=20, n_gen=15, min_size=2, max_size=6,
                    off_targets=None):
    """C2: NSGA-II multi-objective optimization.

    Three objectives (all minimized):
      1. Affinity:        mmsd_per_monomer (more negative = better)
      2. Selectivity:     -selectivity_score (higher = better)
      3. Synthesizability: -synth_score/10 (higher = better)

    Returns Pareto front of non-dominated monomer combinations.
    Reference: Deb 2002 NSGA-II; Garcia-Ortegon 2022 DOCKSTRING.
    """
    from .config import (MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY,
                         is_polymerization_compatible)
    from .utils_analysis import (compute_3obj_for_combo,
                                  compute_synthesizability_score,
                                  compute_selectivity_score)

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

            # Cache key
            key = tuple(sorted(combo))
            if key in eval_cache:
                out["F"] = eval_cache[key]
                return

            # Evaluate via MMSD run (with cross-target ΔΔG selectivity penalty)
            compatible_xls = _get_compatible_crosslinkers(combo)
            pc_id = f"NSGA_{len(pareto_results)+1}_{combo[0]}"
            try:
                mmsd_result = _evaluate_with_selectivity(
                    target, pc_id, combo, compatible_xls,
                    receptor_pdbqt, epitope_pdb, center, npts, be_matrix,
                    work_dir / pc_id, ga_runs=ga_runs,
                    off_targets=off_targets,
                )
            except Exception as e:
                logger.debug(f"NSGA eval failed: {e}")
                out["F"] = [1e6, 1e6, 1e6]
                eval_cache[key] = [1e6, 1e6, 1e6]
                return

            # Compute 3 objectives
            objs = compute_3obj_for_combo(
                combo, target, mmsd_result, be_matrix, all_targets)
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

            eval_cache[key] = f_values
            out["F"] = f_values

            # Record full result
            mmsd_result.update({
                "objectives": f_values,
                "objective_details": objs["raw"],
                "pc_id": pc_id,
            })
            pareto_results.append(mmsd_result)

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

    pareto_combos = []
    for x, f in zip(pareto_X, pareto_F):
        combo = list(dict.fromkeys(
            [monomer_pool[int(idx)] for idx in x
             if monomer_pool[int(idx)] != "NONE"]))
        if len(combo) >= min_size:
            # Find matching cached result
            matching = None
            for r in pareto_results:
                if tuple(sorted(r.get("functional_monomers", []))) == tuple(sorted(combo)):
                    matching = r
                    break
            pareto_combos.append({
                "monomers": combo,
                "objectives": {
                    "affinity_mmsd_per": float(f[0]),
                    "selectivity_score": float(-f[1]),
                    "synthesizability_score": float(-f[2] * 10),
                },
                "mmsd_result": matching,
            })

    # Sort by composite quality (lower aff + higher sel + higher synth)
    def composite_score(c):
        o = c["objectives"]
        return o["affinity_mmsd_per"] - o["selectivity_score"] - o["synthesizability_score"]
    pareto_combos.sort(key=composite_score)

    logger.info(f"\n  Pareto front: {len(pareto_combos)} non-dominated solutions")
    for i, c in enumerate(pareto_combos[:10]):
        o = c["objectives"]
        logger.info(f"    P{i+1}: {c['monomers']} → "
                    f"affinity={o['affinity_mmsd_per']:.2f}, "
                    f"sel={o['selectivity_score']:.2f}, "
                    f"synth={o['synthesizability_score']:.1f}/10")

    return {
        "selected_pareto_front": pareto_combos,
        "all_evaluated": pareto_results,
        "n_evaluations": len(pareto_results),
        "n_pareto": len(pareto_combos),
        "method": "NSGA-II 3-objective",
    }
