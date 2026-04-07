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

    for target in target_names:
        logger.info(f"\n{'='*20} Phase 3 Greedy+MMSD: {target} {'='*20}")

        be_matrix = phase2_results["be_matrix"].get(target, {})
        filtered = phase2_results["filtered"].get(target, [])

        if not filtered:
            logger.warning(f"[{target}] No filtered monomers from Phase 2")
            results[target] = {"error": "No candidates"}
            continue

        p1 = phase1_results[target]
        from .config import resolve_path
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

        # Run Greedy Forward Selection + MMSD
        all_pc_results = _run_greedy_mmsd(
            target=target,
            available_monomers=available,
            be_matrix=be_matrix,
            receptor_pdbqt=receptor_pdbqt,
            epitope_pdb=epitope_pdb,
            center=center, npts=npts,
            work_dir=target_dir,
            ga_runs=min(AUTODOCK4_GA_RUNS, 25),
        )

        # Remove excluded PCs
        all_pc_results = [r for r in all_pc_results if not r.get("excluded")]

        # Rank by bo_objective only (competition is informational, not filtering)
        all_pc_results.sort(key=lambda x:
            x.get("bo_objective") if x.get("bo_objective") is not None else 0,
        )

        top_pcs = all_pc_results[:MMSD_TOP_PC]

        results[target] = {
            "method": "Greedy Forward Selection + MMSD",
            "n_evaluations": len(all_pc_results),
            "available_monomers": available,
            "top_pcs": top_pcs,
            "all_results": all_pc_results,
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
    Return list of compatible crosslinkers based on monomer chemistry.
    - All silane → ["TEOS"]
    - All vinyl/acrylic → ["MBAAm", "EGDMA", "DVB", "TRIM"]
    - Mixed → all 5
    """
    from .config import SILANE_MONOMERS, VINYL_MONOMERS, CROSSLINKER_LIBRARY

    has_silane = any(m in SILANE_MONOMERS for m in monomers)
    has_vinyl = any(m in VINYL_MONOMERS for m in monomers)

    if has_silane and not has_vinyl:
        return [k for k, v in CROSSLINKER_LIBRARY.items() if v["type"] == "silane"]
    elif has_vinyl and not has_silane:
        return [k for k, v in CROSSLINKER_LIBRARY.items() if v["type"] == "vinyl"]
    else:
        return list(CROSSLINKER_LIBRARY.keys())


# ── Greedy Forward Selection + Swap Refinement ──────────────

def _run_greedy_mmsd(target: str, available_monomers: list,
                      be_matrix: dict,
                      receptor_pdbqt: Path, epitope_pdb: Path,
                      center: tuple, npts: tuple,
                      work_dir: Path, ga_runs: int = 25,
                      min_size: int = 2, max_size: int = 6) -> list:
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
            compatible_xls = _get_compatible_crosslinkers(trial)
            pc_id = f"FWD{round_k+1}_{candidate}"

            result = _run_single_mmsd(
                target, pc_id, trial, compatible_xls,
                receptor_pdbqt, epitope_pdb,
                center, npts, be_matrix,
                work_dir / pc_id,
                ga_runs=ga_runs,
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

                result = _run_single_mmsd(
                    target, pc_id, trial, compatible_xls,
                    receptor_pdbqt, epitope_pdb,
                    center, npts, be_matrix,
                    work_dir / pc_id,
                    ga_runs=ga_runs,
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

    # 3. Generate on-the-fly (for crosslinkers like TEOS not in Phase 2)
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
