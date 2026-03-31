"""
Validation Runner
=================
Runs the actual pipeline (Phase 1-3) against the Rajpal 2024
benchmark system (SARS-CoV-2 spike epitope + 11 silane monomers)
to verify that our code reproduces the published results.

This imports and calls the pipeline modules directly — it is NOT
an independent re-implementation. If validation passes, we can
trust the pipeline for novel CD63/CD81/CD9 targets.

Usage:
    conda activate GROMACS
    cd "/home/chan/Research/Monomer screening in Bio"
    python -m code.validation.run_validation              # full validation
    python -m code.validation.run_validation --smd-only    # Phase 2 only
    python -m code.validation.run_validation --check-only  # check existing results
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

VALIDATION_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "validation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate pipeline against Rajpal 2024 benchmark"
    )
    parser.add_argument("--smd-only", action="store_true",
                        help="Run only Phase 2 SMD validation")
    parser.add_argument("--mmsd-only", action="store_true",
                        help="Run only Phase 3 MMSD validation")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check existing results, don't recompute")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (fewer GA runs)")
    return parser.parse_args()


# ── Step 1: Prepare Rajpal 2024 Benchmark System ──────────────

def step_prepare_benchmark(output_dir: Path) -> dict:
    """
    Download the SARS-CoV-2 spike epitope and prepare it as a
    validation target, using our Phase 1 pipeline code.
    """
    from .config_validation import (
        RAJPAL_EPITOPE_PDB_ID, RAJPAL_EPITOPE_CHAIN,
        RAJPAL_EPITOPE_RESIDUES, RAJPAL_EPITOPE_SEQUENCE,
    )
    from pipeline.utils_structure import (
        download_pdb, extract_epitope, analyze_peptide,
        prepare_receptor_pdbqt, compute_grid_center, compute_grid_size,
    )

    logger.info("Step 1: Preparing Rajpal 2024 benchmark epitope...")
    epitope_dir = output_dir / "epitope"
    epitope_dir.mkdir(parents=True, exist_ok=True)

    # Download PDB
    pdb_path = download_pdb(RAJPAL_EPITOPE_PDB_ID, epitope_dir)

    # Extract epitope region
    epitope_pdb = epitope_dir / "rajpal_epitope.pdb"
    extract_epitope(pdb_path, RAJPAL_EPITOPE_CHAIN,
                    RAJPAL_EPITOPE_RESIDUES, epitope_pdb)

    # Analyze properties
    props = analyze_peptide(epitope_pdb)
    logger.info(f"  Sequence: {props.get('sequence', 'N/A')}")
    logger.info(f"  Expected: {RAJPAL_EPITOPE_SEQUENCE[:20]}...")

    # Prepare receptor PDBQT
    receptor_pdbqt = epitope_dir / "rajpal_receptor.pdbqt"
    prepare_receptor_pdbqt(epitope_pdb, receptor_pdbqt)

    # Grid parameters
    center = compute_grid_center(epitope_pdb)
    npts = compute_grid_size(epitope_pdb)

    result = {
        "epitope_pdb": str(epitope_pdb),
        "receptor_pdbqt": str(receptor_pdbqt),
        "grid_center": center,
        "grid_npts": npts,
        "properties": props,
    }

    with open(output_dir / "step1_benchmark.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Step 2: Run SMD with Pipeline Code ─────────────────────────

def step_run_smd(benchmark: dict, output_dir: Path,
                  quick: bool = False) -> dict:
    """
    Run Phase 2 SMD using the actual pipeline code against the
    Rajpal epitope with the 11 silane monomers from the paper.
    """
    from .config_validation import RAJPAL_SMD_REFERENCE
    from pipeline.config import SILANE_MONOMERS, AUTODOCK4_GA_RUNS
    from pipeline.utils_structure import smiles_to_pdbqt
    from pipeline.utils_autodock import dock_single

    logger.info("Step 2: Running SMD (calling pipeline.utils_autodock)...")
    smd_dir = output_dir / "smd"
    smd_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdbqt = Path(benchmark["receptor_pdbqt"])
    center = tuple(benchmark["grid_center"])
    npts = tuple(benchmark["grid_npts"])
    ga_runs = 10 if quick else AUTODOCK4_GA_RUNS

    # Only dock monomers that exist in both our library and Rajpal's
    monomers_to_test = {k: v for k, v in SILANE_MONOMERS.items()
                        if k in RAJPAL_SMD_REFERENCE}

    # Prepare monomer PDBQTs
    monomer_dir = smd_dir / "monomers"
    monomer_pdbqts = {}
    for name, info in monomers_to_test.items():
        try:
            pdbqt = smiles_to_pdbqt(info["smiles"], name, monomer_dir)
            monomer_pdbqts[name] = pdbqt
        except Exception as e:
            logger.warning(f"  Failed to prepare {name}: {e}")

    # Run docking using pipeline code
    results = {}
    for name, pdbqt in monomer_pdbqts.items():
        work = smd_dir / f"dock_{name}"
        logger.info(f"  Docking {name}...")
        try:
            dock_result = dock_single(
                receptor_pdbqt, pdbqt, center, npts, work,
                ga_runs=ga_runs,
            )
            results[name] = dock_result.get("mean_cluster_energy", 0.0)
            logger.info(f"    {name}: {results[name]:.2f} kcal/mol "
                        f"(ref: {RAJPAL_SMD_REFERENCE.get(name, 'N/A')})")
        except Exception as e:
            logger.error(f"    {name} failed: {e}")
            results[name] = 0.0

    # Save in pipeline-compatible format for validation checks
    smd_output = {
        "be_matrix": {"rajpal_epitope": results},
        "filtered": {"rajpal_epitope": list(results.keys())},
    }
    with open(output_dir / "phase2_smd_results.json", "w") as f:
        json.dump(smd_output, f, indent=2)

    return results


# ── Step 3: Run MMSD with Pipeline Code ────────────────────────

def step_run_mmsd(benchmark: dict, smd_results: dict,
                   output_dir: Path, quick: bool = False) -> dict:
    """
    Run Phase 3 MMSD using pipeline code for selected 4-monomer
    combinations matching Rajpal 2024 Table 3.
    """
    from .config_validation import RAJPAL_EXPERIMENTAL_IF
    from pipeline.utils_autodock import dock_single, merge_ligand_into_receptor
    from pipeline.utils_structure import prepare_receptor_pdbqt
    from pipeline.config import AUTODOCK4_GA_RUNS

    logger.info("Step 3: Running MMSD (calling pipeline.utils_autodock)...")
    mmsd_dir = output_dir / "mmsd"
    mmsd_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdbqt = Path(benchmark["receptor_pdbqt"])
    epitope_pdb = Path(benchmark["epitope_pdb"])
    center = tuple(benchmark["grid_center"])
    npts = tuple(benchmark["grid_npts"])
    ga_runs = 5 if quick else min(AUTODOCK4_GA_RUNS, 25)

    all_pcs = []
    for pc_id, pc_data in RAJPAL_EXPERIMENTAL_IF.items():
        monomers = pc_data["monomers"]
        logger.info(f"  MMSD {pc_id}: {monomers}")

        pc_dir = mmsd_dir / pc_id
        pc_dir.mkdir(parents=True, exist_ok=True)

        current_receptor_pdb = epitope_pdb
        current_receptor_pdbqt = receptor_pdbqt
        monomer_energies = {}

        for step, m_name in enumerate(monomers):
            # Get monomer PDBQT
            m_pdbqt = mmsd_dir.parent / "smd" / "monomers" / f"{m_name}.pdbqt"
            if not m_pdbqt.exists():
                from pipeline.config import SILANE_MONOMERS
                from pipeline.utils_structure import smiles_to_pdbqt
                if m_name in SILANE_MONOMERS:
                    m_pdbqt = smiles_to_pdbqt(
                        SILANE_MONOMERS[m_name]["smiles"], m_name,
                        mmsd_dir.parent / "smd" / "monomers",
                    )
                else:
                    monomer_energies[m_name] = 0.0
                    continue

            step_dir = pc_dir / f"step{step}_{m_name}"
            try:
                dock_result = dock_single(
                    current_receptor_pdbqt, m_pdbqt,
                    center, npts, step_dir,
                    ga_runs=ga_runs,
                )
                be = dock_result.get("mean_cluster_energy", 0.0)
                monomer_energies[m_name] = be

                # Merge for next step
                best_pose = dock_result.get("best_pose_path")
                if best_pose and Path(best_pose).exists():
                    merged = pc_dir / f"receptor_step{step}.pdb"
                    merge_ligand_into_receptor(
                        current_receptor_pdb, Path(best_pose), merged
                    )
                    merged_pdbqt = merged.with_suffix(".pdbqt")
                    prepare_receptor_pdbqt(merged, merged_pdbqt)
                    current_receptor_pdb = merged
                    current_receptor_pdbqt = merged_pdbqt
            except Exception as e:
                logger.error(f"    Step {step} {m_name} failed: {e}")
                monomer_energies[m_name] = 0.0

        mmsd_sum = sum(monomer_energies.values())
        smd_sum = sum(smd_results.get(m, 0.0) for m in monomers)

        pc_result = {
            "pc_id": pc_id,
            "monomers": monomers,
            "monomer_energies": monomer_energies,
            "mmsd_sum": round(mmsd_sum, 3),
            "smd_sum": round(smd_sum, 3),
            "delta_sum": round(mmsd_sum - smd_sum, 3),
            "synergy": (mmsd_sum - smd_sum) < 0,
            "is_uniform": True,
            "experimental_if": pc_data.get("IF"),
        }
        all_pcs.append(pc_result)
        logger.info(f"    MMSD sum: {mmsd_sum:.2f}, "
                    f"SMD sum: {smd_sum:.2f}, "
                    f"IF: {pc_data.get('IF', 'N/A')}")

    all_pcs.sort(key=lambda x: x["mmsd_sum"])

    mmsd_output = {
        "rajpal_epitope": {
            "fixed_monomers": ["PTES", "TEOS"],
            "n_combinations": len(all_pcs),
            "top_pcs": all_pcs[:8],
            "all_results": all_pcs,
        }
    }
    with open(output_dir / "phase3_mmsd_results.json", "w") as f:
        json.dump(mmsd_output, f, indent=2, default=str)

    return mmsd_output


# ── Step 4: Check Results Against Reference ────────────────────

def step_run_sullivan(output_dir: Path, quick: bool = False) -> dict:
    """
    Step 5: Run Sullivan 2019 validation — Myoglobin + 5 acrylamide monomers.
    Tests whether computed BE predicts experimental IF ranking.
    """
    from .validate_sullivan import run_sullivan_validation
    logger.info("\nStep 5: Sullivan 2019 validation (Myoglobin)...")
    sullivan_dir = output_dir / "sullivan"
    return run_sullivan_validation(sullivan_dir, quick=quick)


def step_check_results(output_dir: Path) -> dict:
    """Run all validation checks on computed results."""
    from .validate_smd import validate_smd
    from .validate_mmsd import validate_mmsd
    from .validate_ranking import validate_ranking

    logger.info("\nChecking results against reference data...")

    reports = {}

    # Rajpal SMD check
    smd_path = output_dir / "phase2_smd_results.json"
    if smd_path.exists():
        smd_report = validate_smd(smd_path)
        reports["rajpal_smd"] = smd_report
        _log_report(smd_report)
    else:
        reports["rajpal_smd"] = {"status": "SKIP", "reason": "No SMD results"}

    # Rajpal MMSD check
    mmsd_path = output_dir / "phase3_mmsd_results.json"
    if mmsd_path.exists():
        mmsd_report = validate_mmsd(mmsd_path)
        reports["rajpal_mmsd"] = mmsd_report
        _log_report(mmsd_report)

        ranking_report = validate_ranking(mmsd_path)
        reports["rajpal_ranking"] = ranking_report
        _log_report(ranking_report)
    else:
        reports["rajpal_mmsd"] = {"status": "SKIP"}
        reports["rajpal_ranking"] = {"status": "SKIP"}

    # Sullivan check
    sullivan_path = output_dir / "sullivan" / "sullivan_validation.json"
    if sullivan_path.exists():
        with open(sullivan_path) as f:
            sullivan_report = json.load(f)
        reports["sullivan"] = sullivan_report
        logger.info(f"\n  Sullivan 2019: {sullivan_report.get('status', '?')}")
    else:
        reports["sullivan"] = {"status": "SKIP", "reason": "Not yet run"}

    # Save validation report
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(reports, f, indent=2)

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*50}")
    for name, report in reports.items():
        status = report.get("status", "UNKNOWN")
        icon = {"PASS": "✓", "WARN": "△", "FAIL": "✗", "SKIP": "—"}.get(status, "?")
        logger.info(f"  {icon} {name}: {status}")

    overall = "PASS"
    for r in reports.values():
        if r.get("status") == "FAIL":
            overall = "FAIL"
            break
        if r.get("status") == "WARN":
            overall = "WARN"

    logger.info(f"\n  Overall: {overall}")
    return reports


# ── Main ───────────────────────────────────────────────────────

def main():
    args = parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Validation] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(VALIDATION_DIR / "validation.log")),
        ],
    )

    logger.info("Epitope-MIP Pipeline Validation")
    logger.info(f"Benchmark: Rajpal et al., Sci. Rep. 2024")
    logger.info(f"Output: {VALIDATION_DIR}")

    t0 = time.time()

    if args.check_only:
        step_check_results(VALIDATION_DIR)
        return

    # ── Rajpal 2024 Validation ──
    if not args.mmsd_only:
        # Step 1: Prepare benchmark
        benchmark = step_prepare_benchmark(VALIDATION_DIR)

        # Step 2: SMD
        smd_results = step_run_smd(benchmark, VALIDATION_DIR, quick=args.quick)

        # Step 3: MMSD (unless smd-only)
        if not args.smd_only:
            step_run_mmsd(benchmark, smd_results, VALIDATION_DIR,
                          quick=args.quick)

    # ── Sullivan 2019 Validation ──
    if not args.smd_only and not args.mmsd_only:
        step_run_sullivan(VALIDATION_DIR, quick=args.quick)

    # ── Check all results ──
    step_check_results(VALIDATION_DIR)

    elapsed = time.time() - t0
    logger.info(f"\nValidation completed in {elapsed:.0f}s")


def _log_report(report: dict):
    """Pretty-print a validation report."""
    logger.info(f"\n  {report.get('test', 'Unknown test')}: {report.get('status', '?')}")
    for check in report.get("checks", []):
        icon = {"PASS": "✓", "WARN": "△", "FAIL": "✗", "SKIP": "—"}.get(
            check.get("status"), "?"
        )
        logger.info(f"    {icon} {check.get('name', '')}: {check.get('status', '')}")
        for k, v in check.items():
            if k not in ("name", "status") and not isinstance(v, (dict, list)):
                logger.info(f"      {k}: {v}")


if __name__ == "__main__":
    main()
