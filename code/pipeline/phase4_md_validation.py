"""
Phase 4: MD Validation with GROMACS + MM-PBSA
==============================================
Run 50ns all-atom MD simulations for top polymer compositions,
compute binding free energies via MM-PBSA, and evaluate
cross-reactivity between CD63/CD81/CD9.

Reference:
  Sullivan et al., J. Phys. Chem. B 2019 — MM-PBSA for MIP
  Rebelo et al., Int. J. Mol. Sci. 2023 — GROMACS + gmx_MMPBSA
  Sehit/Altintas, ACS Sensors 2024 — MD-based monomer validation
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_phase4(phase1_results: dict = None,
               phase3_results: dict = None,
               target_names: list = None,
               output_dir: str = None,
               quick: bool = False,
               cross_reactivity: bool = True) -> dict:
    """
    Phase 4 entry point: MD validation of top polymer compositions.

    Parameters
    ----------
    phase1_results : Phase 1 output (epitope structures)
    phase3_results : Phase 3 output (top PCs per target)
    target_names : filter to specific targets
    quick : if True, run 20ns instead of 50ns (debugging)
    cross_reactivity : if True, test CD63 PCs against CD81/CD9

    Returns
    -------
    dict : {target: {pc_id: {md_results, mmpbsa, ...}}}
    """
    from .config import (TARGETS, MMSD_TOP_PC, MD_PRODUCTION_NS,
                         MD_QUICK_NS, get_output_path)

    if output_dir is None:
        output_dir = str(get_output_path("phase4"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load previous results
    if phase1_results is None:
        p1_path = get_output_path("phase1") / "phase1_results.json"
        with open(p1_path) as f:
            phase1_results = json.load(f)

    if phase3_results is None:
        p3_path = get_output_path("phase3") / "phase3_mmsd_results.json"
        with open(p3_path) as f:
            phase3_results = json.load(f)

    if target_names is None:
        target_names = list(phase3_results.keys())

    time_ns = MD_QUICK_NS if quick else MD_PRODUCTION_NS
    results = {}

    # 1. MD for each target's top PCs
    for target in target_names:
        p3_data = phase3_results.get(target, {})
        if "error" in p3_data:
            continue

        top_pcs = p3_data.get("top_pcs", [])
        epitope_pdb = Path(phase1_results[target]["epitope_pdb"])

        logger.info(f"\n{'='*20} Phase 4 MD: {target} "
                    f"({len(top_pcs)} PCs, {time_ns}ns) {'='*20}")

        target_results = {}
        for pc_data in top_pcs:
            pc_id = pc_data["pc_id"]
            monomers = pc_data["monomers"]

            logger.info(f"\n--- {target}/{pc_id}: {monomers} ---")
            md_result = _run_md_for_pc(
                target, pc_id, monomers, epitope_pdb,
                output_dir / target / pc_id,
                time_ns=time_ns,
            )
            target_results[pc_id] = md_result

        results[target] = target_results

    # 2. Cross-reactivity test
    if cross_reactivity and len(target_names) > 1:
        logger.info(f"\n{'='*20} Cross-Reactivity Test {'='*20}")
        xr_results = _run_cross_reactivity(
            results, phase1_results, target_names,
            output_dir / "cross_reactivity",
            time_ns=min(time_ns, 50),  # shorter for cross-reactivity
        )
        results["cross_reactivity"] = xr_results

    # Save results
    with open(output_dir / "phase4_md_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    _print_phase4_summary(results, target_names)

    return results


def _run_md_for_pc(target: str, pc_id: str,
                    monomers: list, epitope_pdb: Path,
                    work_dir: Path, time_ns: float = 200.0) -> dict:
    """Run GROMACS MD for one (target, PC) combination."""
    from .utils_gromacs import run_full_md_pipeline
    from .utils_structure import smiles_to_mol2
    from .utils_gromacs import parameterize_monomer
    from .config import ALL_MONOMERS

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "target": target,
        "pc_id": pc_id,
        "monomers": monomers,
        "time_ns": time_ns,
    }

    try:
        # 1. Parameterize monomers with GAFF2
        logger.info(f"  Parameterizing {len(monomers)} monomers...")
        monomer_itps = []
        for m_name in monomers:
            m_info = ALL_MONOMERS.get(m_name)
            if m_info is None:
                logger.warning(f"  Monomer {m_name} not found in library")
                continue

            param_dir = work_dir / "monomer_params"
            mol2 = smiles_to_mol2(m_info["smiles"], m_name, param_dir)
            param = parameterize_monomer(mol2, m_name, param_dir)

            if param.get("itp"):
                monomer_itps.append(param)
            else:
                logger.warning(f"  GAFF2 parameterization failed for {m_name}")

        # 2. Run full MD pipeline
        md_result = run_full_md_pipeline(
            epitope_pdb, monomer_itps,
            work_dir / "md",
            time_ns=time_ns,
            quick=(time_ns <= 20),
        )
        result.update(md_result)

    except Exception as e:
        logger.error(f"  MD failed for {target}/{pc_id}: {e}")
        result["success"] = False
        result["error"] = str(e)

    return result


def _run_cross_reactivity(md_results: dict, phase1_results: dict,
                           target_names: list, work_dir: Path,
                           time_ns: float = 50.0) -> dict:
    """
    Test cross-reactivity: run top PC from each target against
    the other two targets' epitopes.

    E.g., CD63's best PC docked/MD against CD81 and CD9 epitopes.
    """
    xr_results = {}

    for source_target in target_names:
        source_data = md_results.get(source_target, {})
        if not source_data:
            continue

        # Get the best PC (first one, already sorted by MMSD sum)
        best_pc_id = next(iter(source_data), None)
        if best_pc_id is None:
            continue

        best_pc = source_data[best_pc_id]
        monomers = best_pc.get("monomers", [])

        for test_target in target_names:
            if test_target == source_target:
                continue

            key = f"{source_target}_PC_on_{test_target}"
            logger.info(f"  Cross-reactivity: {key}")

            test_epitope = Path(phase1_results[test_target]["epitope_pdb"])

            try:
                xr_result = _run_md_for_pc(
                    test_target, f"XR_{source_target}_{best_pc_id}",
                    monomers, test_epitope,
                    work_dir / key,
                    time_ns=time_ns,
                )
                xr_results[key] = {
                    "source": source_target,
                    "test": test_target,
                    "pc_monomers": monomers,
                    "mmpbsa": xr_result.get("mmpbsa", {}),
                    "success": xr_result.get("success", False),
                }
            except Exception as e:
                xr_results[key] = {"error": str(e)}

    return xr_results


def _print_phase4_summary(results: dict, target_names: list):
    """Print summary table of Phase 4 results."""
    logger.info(f"\n{'='*60}")
    logger.info("Phase 4 MD Validation Summary")
    logger.info(f"{'='*60}")

    header = (f"{'Target':<8} {'PC':<8} {'Monomers':<30} "
              f"{'RMSD(nm)':<10} {'H-bond':<8} "
              f"{'ΔG(kcal)':<10} {'Status':<8}")
    logger.info(header)
    logger.info("-" * len(header))

    for target in target_names:
        target_data = results.get(target, {})
        for pc_id, data in target_data.items():
            monomers = "+".join(data.get("monomers", [])[:2]) + "+..."
            rmsd = data.get("rmsd_mean_nm", float("nan"))
            hbond = data.get("hbond_mean", float("nan"))
            dg = data.get("mmpbsa", {}).get("delta_total_kcal", float("nan"))
            status = "OK" if data.get("success") else "FAIL"

            logger.info(
                f"{target:<8} {pc_id:<8} {monomers:<30} "
                f"{rmsd:<10.3f} {hbond:<8.1f} "
                f"{dg:<10.2f} {status:<8}"
            )

    # Cross-reactivity
    xr = results.get("cross_reactivity", {})
    if xr:
        logger.info(f"\nCross-Reactivity:")
        for key, data in xr.items():
            dg = data.get("mmpbsa", {}).get("delta_total_kcal", "N/A")
            logger.info(f"  {key}: ΔG = {dg} kcal/mol")
