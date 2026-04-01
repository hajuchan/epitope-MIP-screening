"""
Phase 3: Bayesian Optimization + MMSD
======================================
Gryffin Bayesian Optimization (Hase et al. 2021) to find the optimal
monomer combination (type + number), evaluated by MMSD sequential
docking (Rajpal et al. 2024).

Instead of exhaustive search of fixed 4-monomer combinations (~36-330),
Gryffin explores 2-6 monomer combinations with physicochemical
descriptors, finding near-optimal solutions in ~30-50 evaluations.

Key concept: MMSD sum vs SMD sum reveals synergy/interference.
  MMSD sum < SMD sum → synergy (cooperative binding)
  MMSD sum > SMD sum → interference (steric clash)

Reference:
  Hase F et al., Appl. Phys. Rev. 2021;8:031406 — Gryffin BO
  Rajpal et al., Sci. Rep. 2024 — MMSD protocol
  Sullivan et al., J. Phys. Chem. B 2019 — non-competitive binding
"""

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Monomer Physicochemical Descriptors for Gryffin ────────────
# Computed from RDKit: [MW, LogP, HBD, HBA, TPSA, RotatableBonds, AromaticRings, HeavyAtoms]
# All values are continuous and well-distributed → no normalization issues.
# Enables GPR to generalize: "H-bond monomers were good → try similar ones"

MONOMER_DESCRIPTORS = {
    # Silane monomers
    "PTES":   [240.4, 1.94, 0, 3, 27.7, 7, 1, 16],
    "APTES":  [237.4, 0.90, 1, 5, 62.9, 10, 0, 15],
    "APTMS":  [195.3, -0.27, 1, 5, 62.9, 7, 0, 12],
    "UPTMS":  [238.3, -0.56, 2, 5, 92.0, 8, 0, 15],
    "MPTMS":  [212.3, 0.70, 1, 5, 36.9, 7, 0, 12],
    "IBTES":  [236.4, 2.20, 0, 4, 36.9, 9, 0, 15],
    "MTMS":   [136.2, 0.49, 0, 3, 27.7, 3, 0, 8],
    "EDTMS":  [238.4, -0.68, 2, 6, 75.0, 10, 0, 15],
    "ICTES":  [263.4, 1.27, 0, 6, 66.4, 11, 0, 17],
    "VTMS":   [148.2, 0.59, 0, 3, 27.7, 4, 0, 9],
    "GPTMS":  [208.3, 0.17, 0, 5, 49.5, 7, 0, 13],
    "DIDMS":  [120.2, 0.98, 0, 2, 18.5, 2, 0, 7],
    "CETES":  [247.4, 1.85, 0, 5, 60.7, 10, 0, 16],
    "TTMS":   [212.3, 1.08, 0, 3, 27.7, 4, 1, 14],
    # Vinyl/acrylic monomers
    "AA":     [72.1, 0.26, 1, 1, 37.3, 1, 0, 5],
    "MAA":    [86.1, 0.65, 1, 1, 37.3, 1, 0, 6],
    "AAm":    [71.1, -0.34, 1, 1, 43.1, 1, 0, 5],
    "NIPAm":  [113.2, 0.70, 1, 1, 29.1, 2, 0, 8],
    "4VIm":   [94.1, 1.05, 1, 1, 28.7, 1, 1, 7],
    "HEMA":   [130.1, 0.10, 1, 3, 46.5, 3, 0, 9],
    "DA":     [153.2, 0.60, 3, 3, 66.5, 2, 1, 11],
    "NE":     [169.2, 0.09, 4, 4, 86.7, 2, 1, 12],
    "TBAm":   [127.2, 1.09, 1, 1, 29.1, 1, 0, 9],
    "APBA":   [136.9, -1.05, 3, 3, 66.5, 1, 1, 10],
}


def run_phase3(phase1_results: dict = None,
               phase2_results: dict = None,
               target_names: list = None,
               output_dir: str = None) -> dict:
    """
    Phase 3 entry point: Gryffin BO + MMSD for all targets.

    For each target:
    1. Configure Gryffin with monomer descriptors
    2. BO loop: suggest combination → run MMSD → observe result
    3. Automatically determine optimal combination size (2-6)
    4. Rank and select top polymer compositions (PCs)
    """
    from .config import (TARGETS, FUNCTIONAL_MONOMERS,
                         MMSD_DEFAULT_CROSSLINKER, MMSD_TOP_PC,
                         MMSD_HIGH_AFFINITY_THRESHOLD,
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
        logger.info(f"\n{'='*20} Phase 3 BO+MMSD: {target} {'='*20}")

        be_matrix = phase2_results["be_matrix"].get(target, {})
        filtered = phase2_results["filtered"].get(target, [])

        if not filtered:
            logger.warning(f"[{target}] No filtered monomers from Phase 2")
            results[target] = {"error": "No candidates"}
            continue

        p1 = phase1_results[target]
        receptor_pdbqt = Path(p1["receptor_pdbqt"])
        epitope_pdb = Path(p1["epitope_pdb"])
        center = tuple(p1["grid_center"])
        npts = tuple(p1["grid_npts"])

        crosslinker = MMSD_DEFAULT_CROSSLINKER

        # Available monomers for BO:
        # Use filtered list if sufficient, otherwise expand to top-N by BE
        from .config import MMSD_MIN_POOL_SIZE
        available = [m for m in filtered if m != crosslinker]
        if len(available) < MMSD_MIN_POOL_SIZE:
            # Expand: top monomers by BE (regardless of ΔΔG filter)
            sorted_by_be = sorted(
                [(m, e) for m, e in be_matrix.items()
                 if e is not None and m != crosslinker],
                key=lambda x: x[1]
            )
            available = [m for m, _ in sorted_by_be[:max(MMSD_MIN_POOL_SIZE, len(available))]]
            logger.info(f"[{target}] Expanded monomer pool to {len(available)} "
                        f"(filtered {len(filtered)} was too small for BO)")

        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        # Run Gryffin BO + MMSD
        all_pc_results = _run_bo_mmsd(
            target=target,
            available_monomers=available,
            crosslinker=crosslinker,
            be_matrix=be_matrix,
            receptor_pdbqt=receptor_pdbqt,
            epitope_pdb=epitope_pdb,
            center=center, npts=npts,
            work_dir=target_dir,
            ga_runs=min(AUTODOCK4_GA_RUNS, 25),
        )

        # Remove excluded PCs
        all_pc_results = [r for r in all_pc_results if not r.get("excluded")]

        # Rank: uniform first, then by MMSD sum
        all_pc_results.sort(key=lambda x: (
            not x.get("is_uniform", True),
            x.get("mmsd_sum") if x.get("mmsd_sum") is not None else 0,
        ))

        top_pcs = all_pc_results[:MMSD_TOP_PC]

        results[target] = {
            "method": "Gryffin BO + MMSD",
            "n_evaluations": len(all_pc_results),
            "available_monomers": available,
            "crosslinker": crosslinker,
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
                f"  {j}. {pc['monomers']} ({len(pc['monomers'])}종): "
                f"MMSD={pc.get('mmsd_sum', 'N/A')}, "
                f"synergy={pc.get('synergy', '?')}"
            )

    # Save results
    with open(output_dir / "phase3_mmsd_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    _plot_mmsd_comparison(results, output_dir)

    return results


# ── Gryffin BO + MMSD Loop ────────────────────────────────────

def _run_bo_mmsd(target: str, available_monomers: list,
                  crosslinker: str, be_matrix: dict,
                  receptor_pdbqt: Path, epitope_pdb: Path,
                  center: tuple, npts: tuple,
                  work_dir: Path, ga_runs: int = 25,
                  n_initial: int = 15, max_iter: int = 50,
                  min_size: int = 2, max_size: int = 6) -> list:
    """
    Gryffin Bayesian Optimization loop for monomer combination search.

    Each iteration:
    1. Gryffin suggests a monomer combination (type + size)
    2. Run MMSD sequential docking
    3. Report MMSD sum back to Gryffin
    4. Repeat until convergence or max iterations
    """
    from gryffin import Gryffin

    n_slots = max_size  # max monomer slots

    # Build Gryffin config with descriptor-informed categorical variables
    # Each slot is a categorical variable: which monomer (or "empty")
    options_with_empty = available_monomers + ["_empty_"]

    # Build descriptor dict for each option
    descriptors = {}
    for m in available_monomers:
        if m in MONOMER_DESCRIPTORS:
            descriptors[m] = MONOMER_DESCRIPTORS[m]
        else:
            descriptors[m] = [0, 0, 0, 0, 0, 100]
    # _empty_ = "no monomer in this slot"
    # Values set to minimum of each descriptor column across real monomers
    # This ensures _empty_ is at the edge of the distribution, not an outlier
    import numpy as np
    all_desc = np.array(list(descriptors.values()))
    empty_desc = list(np.min(all_desc, axis=0))
    descriptors["_empty_"] = empty_desc

    config = {
        "parameters": [
            {
                "name": f"monomer_{i}",
                "type": "categorical",
                "category_details": descriptors,
            }
            for i in range(n_slots)
        ],
        "objectives": [
            {"name": "mmsd_sum", "goal": "min"},
        ],
    }

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            module="gryffin")
    gryffin = Gryffin(config_dict=config, silent=True)

    all_results = []
    observations = []
    evaluated_combos = set()
    best_mmsd = float("inf")
    no_improvement_count = 0

    logger.info(f"[{target}] Gryffin BO: {len(available_monomers)} monomers, "
                f"slots={n_slots}, max_iter={max_iter}")

    import random as _rng
    _rng.seed(42)

    for iteration in range(max_iter):
        # Phase 1: Random diverse exploration (first n_initial iterations)
        # Phase 2: Gryffin-guided exploitation (after sufficient observations)
        use_gryffin = len(observations) >= n_initial

        combo = None

        if use_gryffin:
            try:
                suggestions = gryffin.recommend(
                    observations=observations,
                    num_batches=1,
                )
                if suggestions:
                    suggestion = suggestions[0]
                    combo = []
                    for i in range(n_slots):
                        m = suggestion[f"monomer_{i}"]
                        if m != "_empty_" and m not in combo:
                            combo.append(m)
            except Exception as e:
                logger.warning(f"  Gryffin recommend failed: {e}")

        # Random sampling: during initial phase, or if Gryffin gave duplicate
        combo_key = tuple(sorted(combo)) if combo else ()
        if not combo or combo_key in evaluated_combos:
            # Generate random unique combination
            for _ in range(50):
                k = _rng.randint(min_size, max_size)
                combo = _rng.sample(available_monomers, min(k, len(available_monomers)))
                combo_key = tuple(sorted(combo))
                if combo_key not in evaluated_combos:
                    break
            else:
                logger.info(f"  Exhausted unique combinations after "
                            f"{len(evaluated_combos)} evaluations")
                break

        evaluated_combos.add(combo_key)

        # Add crosslinker at the end
        monomers_with_xl = combo + [crosslinker]
        pc_id = f"BO{iteration+1:03d}"

        logger.info(f"  [{pc_id}] {monomers_with_xl} ({len(combo)}+XL)")

        # Run MMSD
        pc_result = _run_single_mmsd(
            target, pc_id, monomers_with_xl,
            receptor_pdbqt, epitope_pdb,
            center, npts, be_matrix,
            work_dir / pc_id,
            ga_runs=ga_runs,
        )
        all_results.append(pc_result)

        # Report to Gryffin
        mmsd_sum = pc_result.get("mmsd_sum")
        if mmsd_sum is not None:
            obs = {f"monomer_{i}": "_empty_" for i in range(n_slots)}
            for i, m in enumerate(combo):
                if i < n_slots:
                    obs[f"monomer_{i}"] = m
            obs["mmsd_sum"] = mmsd_sum
            observations.append(obs)

            # Convergence check
            if mmsd_sum < best_mmsd - 0.3:
                best_mmsd = mmsd_sum
                no_improvement_count = 0
                logger.info(f"    NEW BEST: {mmsd_sum:.2f} kcal/mol")
            else:
                no_improvement_count += 1

            if no_improvement_count >= 8 and len(observations) >= n_initial:
                logger.info(f"  Converged after {iteration+1} iterations "
                            f"(no improvement in 8 steps)")
                break
        else:
            logger.warning(f"    {pc_id}: MMSD failed, skipping observation")

    logger.info(f"[{target}] BO completed: {len(all_results)} evaluations, "
                f"best MMSD={best_mmsd:.2f}")

    return all_results


# ── MMSD Sequential Docking (unchanged from Rajpal 2024) ──────

def _run_single_mmsd(target: str, pc_id: str,
                      monomers: list,
                      receptor_pdbqt: Path,
                      epitope_pdb: Path,
                      center: tuple, npts: tuple,
                      smd_be: dict,
                      work_dir: Path,
                      ga_runs: int = 25) -> dict:
    """
    Run sequential MMSD for a single polymer composition.
    Unchanged from Rajpal 2024 protocol.
    """
    from .utils_autodock import dock_single, merge_ligand_into_receptor
    from .utils_structure import prepare_receptor_pdbqt

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    monomer_energies = {}
    current_receptor_pdb = epitope_pdb
    current_receptor_pdbqt = receptor_pdbqt

    for step, monomer_name in enumerate(monomers):
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

    # Compute MMSD sum — skip if any monomer failed
    valid_energies = {m: e for m, e in monomer_energies.items() if e is not None}
    if len(valid_energies) < len(monomers):
        n_fail = len(monomers) - len(valid_energies)
        logger.warning(f"    {pc_id}: {n_fail} monomer(s) failed — excluded")
        return {
            "pc_id": pc_id, "monomers": monomers,
            "monomer_energies": monomer_energies,
            "mmsd_sum": None, "smd_sum": None, "delta_sum": None,
            "synergy": False, "is_uniform": False, "excluded": True,
        }

    mmsd_sum = sum(valid_energies.values())
    smd_sum = sum(smd_be.get(m, 0.0) for m in monomers
                  if smd_be.get(m) is not None)
    delta_sum = mmsd_sum - smd_sum

    # Competition analysis (Sullivan 2019)
    from .config import MMSD_PENALIZE_COMPETITION, MMSD_COMPETITION_DISTANCE
    competition = {"is_uniform": True, "n_competing": 0}
    monomer_centers = {}

    if MMSD_PENALIZE_COMPETITION:
        for step, m_name in enumerate(monomers):
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
        "monomers": monomers,
        "n_functional": len(monomers) - 1,  # exclude crosslinker
        "monomer_energies": monomer_energies,
        "mmsd_sum": round(mmsd_sum, 3),
        "smd_sum": round(smd_sum, 3),
        "delta_sum": round(delta_sum, 3),
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
            ax1.set_title(f"{target}: Gryffin BO Convergence")
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
