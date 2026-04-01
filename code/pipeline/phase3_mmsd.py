"""
Phase 3: Multi-Monomer Simultaneous Docking (MMSD)
===================================================
Sequential docking of 4-monomer combinations to evaluate
cooperative binding effects, per Rajpal et al. 2024 protocol.

Key concept: MMSD sum vs SMD sum reveals synergy/interference.
  MMSD sum > SMD sum → synergy (cooperative binding)
  MMSD sum < SMD sum → interference (steric clash)

Reference:
  Rajpal et al., Sci. Rep. 2024 — Tables 2-3, Fig. 3
  Rajpal & Mizaikoff, J. Mater. Chem. B 2022 — bi/tri-monomer docking
"""

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_phase3(phase1_results: dict = None,
               phase2_results: dict = None,
               target_names: list = None,
               output_dir: str = None) -> dict:
    """
    Phase 3 entry point: MMSD for all targets.

    For each target:
    1. Fix 2 monomers: best-BE monomer + cross-linker (TEOS)
    2. Generate 4-monomer combinations
    3. Run sequential AutoDock4 docking
    4. Compare MMSD sum vs SMD sum
    5. Rank and select top polymer compositions (PCs)

    Returns
    -------
    dict : {target: {top_pcs: [...], all_results: [...]}}
    """
    from .config import (TARGETS, FUNCTIONAL_MONOMERS,
                         MMSD_COMBO_SIZE, MMSD_N_FIXED,
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
        logger.info(f"\n{'='*20} MMSD: {target} {'='*20}")

        be_matrix = phase2_results["be_matrix"].get(target, {})
        filtered = phase2_results["filtered"].get(target, [])

        if not filtered:
            logger.warning(f"[{target}] No filtered monomers from Phase 2")
            results[target] = {"error": "No candidates"}
            continue

        # 1. Select fixed monomers
        best_monomer = min(be_matrix, key=be_matrix.get)
        crosslinker = MMSD_DEFAULT_CROSSLINKER
        fixed = [best_monomer, crosslinker]
        logger.info(f"[{target}] Fixed: {fixed[0]} (best BE: "
                    f"{be_matrix.get(best_monomer, 0):.2f}) + {crosslinker}")

        # 2. Generate variable monomer combinations
        variable_pool = [m for m in filtered
                         if m not in fixed and m != crosslinker]
        n_variable = MMSD_COMBO_SIZE - MMSD_N_FIXED
        combos = list(combinations(variable_pool, n_variable))

        if not combos:
            # If not enough filtered monomers, use all available
            variable_pool = [m for m in FUNCTIONAL_MONOMERS
                             if m not in fixed]
            combos = list(combinations(variable_pool, n_variable))

        logger.info(f"[{target}] {len(combos)} combinations to screen")

        # 3. Run MMSD for each combination
        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        p1 = phase1_results[target]
        receptor_pdbqt = Path(p1["receptor_pdbqt"])
        epitope_pdb = Path(p1["epitope_pdb"])
        center = tuple(p1["grid_center"])
        npts = tuple(p1["grid_npts"])

        all_pc_results = []
        for i, variable_monomers in enumerate(combos):
            pc_monomers = list(fixed) + list(variable_monomers)
            pc_id = f"PC{i+1:03d}"
            pc_name = "_".join(pc_monomers)

            logger.info(f"  [{pc_id}] {pc_name}")
            pc_result = _run_single_mmsd(
                target, pc_id, pc_monomers,
                receptor_pdbqt, epitope_pdb,
                center, npts,
                be_matrix,  # for SMD sum comparison
                target_dir / pc_id,
                ga_runs=min(AUTODOCK4_GA_RUNS, 25),  # fewer runs for speed
            )
            all_pc_results.append(pc_result)

        # Remove excluded PCs (incomplete docking)
        all_pc_results = [r for r in all_pc_results if not r.get("excluded")]

        # 4. Rank by MMSD sum (Sullivan 2019: penalize competing PCs)
        # Uniform (non-competing) PCs get priority at equal MMSD sum
        all_pc_results.sort(key=lambda x: (
            not x.get("is_uniform", True),  # uniform first
            x.get("mmsd_sum", 0),           # then by energy
        ))

        # 5. Select top PCs
        top_pcs = all_pc_results[:MMSD_TOP_PC]

        results[target] = {
            "fixed_monomers": fixed,
            "n_combinations": len(combos),
            "top_pcs": top_pcs,
            "all_results": all_pc_results,
            "high_affinity_count": sum(
                1 for r in all_pc_results
                if r.get("mmsd_sum", 0) <= MMSD_HIGH_AFFINITY_THRESHOLD
            ),
        }

        # Log top results
        logger.info(f"\n[{target}] Top {MMSD_TOP_PC} PCs:")
        for j, pc in enumerate(top_pcs, 1):
            logger.info(
                f"  {j}. {pc['pc_id']} {pc['monomers']}: "
                f"MMSD={pc.get('mmsd_sum', 0):.2f}, "
                f"SMD={pc.get('smd_sum', 0):.2f}, "
                f"Δ={pc.get('delta_sum', 0):.2f} kcal/mol"
            )

    # Save results
    with open(output_dir / "phase3_mmsd_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate comparison plots
    _plot_mmsd_comparison(results, output_dir)

    return results


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

    Protocol (Rajpal 2024):
    1. Dock monomer-1 onto epitope → best pose
    2. Merge monomer-1 into receptor
    3. Dock monomer-2 onto (epitope + monomer-1) → best pose
    4. Merge monomer-2 into receptor
    5. ... repeat for all monomers
    6. MMSD sum = Σ(each monomer's best BE)
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

        # Get monomer PDBQT
        monomer_pdbqt = _get_monomer_pdbqt(monomer_name, work_dir.parent)
        if monomer_pdbqt is None:
            logger.warning(f"  {pc_id} step {step}: {monomer_name} PDBQT "
                           "not found, skipping")
            monomer_energies[monomer_name] = 0.0
            continue

        # Dock
        dock_result = dock_single(
            current_receptor_pdbqt, monomer_pdbqt,
            center, npts, step_dir,
            ga_runs=ga_runs,
        )

        be = dock_result.get("mean_cluster_energy")
        if be is None:
            logger.warning(f"    Step {step} {monomer_name}: docking failed, skipping")
            monomer_energies[monomer_name] = None
            continue
        monomer_energies[monomer_name] = be

        # Merge docked ligand into receptor for next step
        best_pose = dock_result.get("best_pose_path")
        if best_pose and Path(best_pose).exists():
            merged_pdb = work_dir / f"receptor_step{step}.pdb"
            merge_ligand_into_receptor(
                current_receptor_pdb, Path(best_pose), merged_pdb
            )
            # Re-prepare as PDBQT
            merged_pdbqt = merged_pdb.with_suffix(".pdbqt")
            prepare_receptor_pdbqt(merged_pdb, merged_pdbqt)
            current_receptor_pdb = merged_pdb
            current_receptor_pdbqt = merged_pdbqt

    # Compute MMSD sum and SMD sum — skip if any monomer failed
    valid_energies = {m: e for m, e in monomer_energies.items() if e is not None}
    if len(valid_energies) < len(monomers):
        n_fail = len(monomers) - len(valid_energies)
        logger.warning(f"    {pc_id}: {n_fail} monomer(s) failed — PC excluded")
        return {
            "pc_id": pc_id, "monomers": monomers,
            "monomer_energies": monomer_energies,
            "mmsd_sum": None, "smd_sum": None, "delta_sum": None,
            "synergy": False, "is_uniform": False, "excluded": True,
        }

    mmsd_sum = sum(valid_energies.values())
    smd_sum = sum(smd_be.get(m, 0.0) for m in monomers if smd_be.get(m) is not None)
    delta_sum = mmsd_sum - smd_sum  # negative = synergy

    # Sullivan 2019: analyze competition (monomers at same binding site)
    from .config import MMSD_PENALIZE_COMPETITION, MMSD_COMPETITION_DISTANCE
    competition = {"is_uniform": True, "n_competing": 0}
    monomer_centers = {}

    if MMSD_PENALIZE_COMPETITION:
        # Extract pose centers for each monomer
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
            if not competition["is_uniform"]:
                logger.info(
                    f"    {pc_id}: {competition['n_competing']} competing "
                    f"pair(s) detected (Sullivan 2019 penalty)"
                )

    return {
        "pc_id": pc_id,
        "monomers": monomers,
        "monomer_energies": monomer_energies,
        "mmsd_sum": round(mmsd_sum, 3),
        "smd_sum": round(smd_sum, 3),
        "delta_sum": round(delta_sum, 3),
        "synergy": delta_sum < 0,
        "competition": competition,
        "is_uniform": competition.get("is_uniform", True),
    }


def _get_monomer_pdbqt(name: str, search_dir: Path) -> Path:
    """Find monomer PDBQT file in the search directory tree."""
    candidates = list(Path(search_dir).rglob(f"{name}.pdbqt"))
    if candidates:
        return candidates[0]

    # Also check phase2 monomer directory
    from .config import get_output_path
    p2_monomers = get_output_path("phase2") / "monomers"
    candidate = p2_monomers / f"{name}.pdbqt"
    if candidate.exists():
        return candidate

    return None


def _plot_mmsd_comparison(results: dict, output_dir: Path):
    """Generate MMSD sum vs SMD sum comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for target, data in results.items():
            if "error" in data:
                continue

            all_results = data.get("all_results", [])
            if not all_results:
                continue

            mmsd_sums = [r["mmsd_sum"] for r in all_results]
            smd_sums = [r["smd_sum"] for r in all_results]
            labels = [r["pc_id"] for r in all_results]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Scatter: MMSD vs SMD
            ax1.scatter(smd_sums, mmsd_sums, alpha=0.7, c="steelblue")
            # Diagonal line (no change)
            lims = [min(min(smd_sums), min(mmsd_sums)) - 1,
                    max(max(smd_sums), max(mmsd_sums)) + 1]
            ax1.plot(lims, lims, "k--", alpha=0.3, label="No change")
            ax1.set_xlabel("SMD Sum (kcal/mol)")
            ax1.set_ylabel("MMSD Sum (kcal/mol)")
            ax1.set_title(f"{target}: MMSD vs SMD Sum")
            ax1.legend()

            # Highlight top PCs
            top_pcs = data.get("top_pcs", [])[:5]
            for pc in top_pcs:
                idx = next((i for i, r in enumerate(all_results)
                            if r["pc_id"] == pc["pc_id"]), None)
                if idx is not None:
                    ax1.annotate(pc["pc_id"],
                                 (smd_sums[idx], mmsd_sums[idx]),
                                 fontsize=7, color="red")

            # Bar chart: top 10 PCs by MMSD sum
            top10 = sorted(all_results,
                           key=lambda x: x["mmsd_sum"])[:10]
            names = ["_".join(r["monomers"][2:]) for r in top10]
            values = [r["mmsd_sum"] for r in top10]

            ax2.barh(range(len(top10)), values, color="steelblue")
            ax2.set_yticks(range(len(top10)))
            ax2.set_yticklabels(names, fontsize=8)
            ax2.set_xlabel("MMSD Sum (kcal/mol)")
            ax2.set_title(f"{target}: Top 10 Polymer Compositions")
            ax2.invert_yaxis()

            plt.tight_layout()
            plt.savefig(output_dir / f"phase3_{target}_comparison.png",
                        dpi=150)
            plt.close()

        logger.info(f"MMSD plots → {output_dir}")
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
