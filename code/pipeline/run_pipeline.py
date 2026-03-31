"""
Epitope-MIP Screening Pipeline — Main Entry Point
==================================================
5-Phase computational screening of functional monomers for
epitope-imprinted MIPs targeting exosome tetraspanins (CD63/CD81/CD9).

Phase 1: Epitope extraction + structure preparation
Phase 2: Single Monomer Docking (SMD) with AutoDock4
Phase 3: Multi-Monomer Simultaneous Docking (MMSD)
Phase 4: GROMACS MD validation + MM-PBSA
Phase 5: Synthesis recipe generation

Additional options:
  --report     Generate HTML report from existing results

Reference:
  Rajpal et al., Sci. Rep. 2024 — MMSD methodology
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import OUTPUT_DIR, TARGETS

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Epitope-MIP Screening Pipeline for Exosome Tetraspanins"
    )
    parser.add_argument(
        "--target", type=str, default="all",
        choices=list(TARGETS.keys()) + ["all"],
        help="Which target to screen (default: all)"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "3", "4", "5", "all"],
        help="Which phase to run (default: all)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate HTML report from existing results"
    )
    parser.add_argument(
        "--skip-md", action="store_true",
        help="Skip all intermediate MD (Phase 1 stability + Phase 2 contact). "
             "NOT recommended — use only for debugging docking logic."
    )
    parser.add_argument(
        "--quick-md", action="store_true",
        help="Run 20ns instead of 50ns MD in Phase 4 (debugging only)"
    )
    parser.add_argument(
        "--no-cross-reactivity", action="store_true",
        help="Skip cross-reactivity test in Phase 4"
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from last completed phase (DEFAULT: on). "
             "Skips phases whose result files already exist."
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Force re-run all phases from scratch, ignoring existing results."
    )
    return parser.parse_args()


def _phase_dir(root_dir: str, phase_key: str) -> str:
    """Return phase-specific output subdirectory, creating if needed."""
    p = Path(root_dir) / phase_key
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# Result file that marks each phase as completed
_PHASE_RESULT_FILES = {
    1: "phase1/phase1_results.json",
    2: "phase2/phase2_smd_results.json",
    3: "phase3/phase3_mmsd_results.json",
    4: "phase4/phase4_md_results.json",
    5: "phase5/phase5_recipes.json",
}


def _check_phase_completed(phase_num: int, output_dir: str) -> bool:
    """Check if a phase's result file exists (= already completed)."""
    result_file = _PHASE_RESULT_FILES.get(phase_num)
    if result_file is None:
        return False
    return (Path(output_dir) / result_file).exists()


def _load_phase_result(phase_num: int, output_dir: str) -> dict:
    """Load existing phase result from disk."""
    result_file = _PHASE_RESULT_FILES.get(phase_num)
    if result_file is None:
        return None
    path = Path(output_dir) / result_file
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def run_phase(phase_num: int, target_names: list, output_dir: str,
              prev_results: dict, args) -> dict:
    """Run a single pipeline phase and return its results."""
    t0 = time.time()

    if phase_num == 1:
        from .phase1_epitope_prep import run_phase1
        result = run_phase1(
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase1"),
            skip_stability_md=args.skip_md,
        )

    elif phase_num == 2:
        from .phase2_smd import run_phase2
        phase1_results = prev_results.get("phase1")
        result = run_phase2(
            phase1_results=phase1_results,
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase2"),
        )

    elif phase_num == 3:
        from .phase3_mmsd import run_phase3
        result = run_phase3(
            phase1_results=prev_results.get("phase1"),
            phase2_results=prev_results.get("phase2"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase3"),
        )

    elif phase_num == 4:
        from .phase4_md_validation import run_phase4
        result = run_phase4(
            phase1_results=prev_results.get("phase1"),
            phase3_results=prev_results.get("phase3"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase4"),
            quick=args.quick_md,
            cross_reactivity=not args.no_cross_reactivity,
        )

    elif phase_num == 5:
        from .phase5_recipe import run_phase5
        result = run_phase5(
            phase3_results=prev_results.get("phase3"),
            phase4_results=prev_results.get("phase4"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase5"),
        )

    elapsed = time.time() - t0
    logger.info(f"Phase {phase_num} completed in {elapsed:.1f}s")
    return {"result": result, "elapsed_s": round(elapsed, 1)}


def print_summary(output_dir: str):
    """Print final summary of pipeline results."""
    out = Path(output_dir)
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY — Epitope-MIP Screening Results")
    logger.info(f"{'='*60}")

    # Phase 2: SMD summary
    p2 = out / "phase2" / "phase2_smd_results.json"
    if p2.exists():
        with open(p2) as f:
            smd = json.load(f)
        logger.info("\n[Phase 2] SMD Filtered Monomers:")
        for target, monomers in smd.get("filtered", {}).items():
            logger.info(f"  {target}: {monomers}")

    # Phase 3: MMSD top PCs
    p3 = out / "phase3" / "phase3_mmsd_results.json"
    if p3.exists():
        with open(p3) as f:
            mmsd = json.load(f)
        logger.info("\n[Phase 3] Top Polymer Compositions:")
        for target, data in mmsd.items():
            if isinstance(data, dict) and "top_pcs" in data:
                for pc in data["top_pcs"][:3]:
                    logger.info(
                        f"  {target} {pc['pc_id']}: "
                        f"{pc['monomers']} "
                        f"(MMSD={pc.get('mmsd_sum', 'N/A'):.2f})"
                    )

    # Phase 5: Recipes
    p5 = out / "phase5" / "phase5_recipes.json"
    if p5.exists():
        with open(p5) as f:
            recipes = json.load(f)
        logger.info("\n[Phase 5] Recommended Recipes:")
        for target, recipe in recipes.items():
            monomers = list(recipe.get("monomers", {}).keys())
            logger.info(f"  {target}: {monomers} "
                        f"({recipe.get('polymerization_type', 'N/A')})")

    logger.info(f"\n{'='*60}")


def main():
    args = parse_args()

    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "reports").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Pipeline] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                str(Path(out_dir) / "reports" / "pipeline.log")
            ),
        ],
    )

    # Report-only mode
    if args.report:
        from .generate_report import generate_report
        path = generate_report(output_dir=out_dir)
        logger.info(f"Report generated: {path}")
        return

    # Determine targets
    if args.target == "all":
        target_names = list(TARGETS.keys())
    else:
        target_names = [args.target]

    logger.info(f"Targets: {target_names}")
    logger.info(f"Output: {out_dir}")

    # Determine phases
    if args.phase == "all":
        phases = [1, 2, 3, 4, 5]
    else:
        phases = [int(args.phase)]

    t_total = time.time()
    timings = {}
    prev_results = {}

    # --resume (default) / --fresh
    if args.fresh:
        args.resume = False
        logger.info("Fresh run: ignoring existing results")

    if args.resume:
        for phase_num in phases:
            if _check_phase_completed(phase_num, out_dir):
                loaded = _load_phase_result(phase_num, out_dir)
                prev_results[f"phase{phase_num}"] = loaded
                logger.info(f"Phase {phase_num}: LOADED from existing results "
                            f"({_PHASE_RESULT_FILES[phase_num]})")
            else:
                break  # run this phase and all subsequent

    for phase_num in phases:
        # Skip if already loaded via --resume
        if f"phase{phase_num}" in prev_results:
            timings[f"phase{phase_num}"] = 0.0
            continue

        logger.info(f"\n{'='*20} PHASE {phase_num} {'='*20}")
        phase_output = run_phase(
            phase_num, target_names, out_dir, prev_results, args
        )
        timings[f"phase{phase_num}"] = phase_output["elapsed_s"]
        prev_results[f"phase{phase_num}"] = phase_output["result"]

    total_time = time.time() - t_total

    # Timing summary
    logger.info(f"\n--- Timing Summary ---")
    for name, t in timings.items():
        logger.info(f"  {name}: {t:.1f}s")
    logger.info(f"  Total: {total_time:.1f}s")

    # Final summary
    print_summary(out_dir)

    # Auto-generate report
    if args.phase == "all":
        try:
            from .generate_report import generate_report
            path = generate_report(output_dir=out_dir)
            logger.info(f"Report: {path}")
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")


if __name__ == "__main__":
    main()
