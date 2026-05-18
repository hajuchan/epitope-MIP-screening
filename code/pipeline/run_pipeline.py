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
        "--target", type=str, nargs="+", default=["all"],
        help="Which target(s) to screen (default: all). E.g. --target CD63 CD81"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "3", "4", "5", "6", "all"],
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
    parser.add_argument(
        "--extend-drifters", action="store_true",
        help="Extend Phase 5 rebinding MDs that show RMSD drift Q1→Q4 > 1.5 Å. "
             "Uses gmx convert-tpr -extend + mdrun -cpi to continue MD by 100 ns. "
             "Run after standard Phase 5 completes."
    )
    parser.add_argument(
        "--extend-ns", type=int, default=100,
        help="Extension length in ns for --extend-drifters (default: 100)"
    )
    parser.add_argument(
        "--multirestart", action="store_true",
        help="Multi-restart ensemble: re-run Phase 5 rebinding with N=3 perturbed "
             "starting head positions per snapshot. Adds extra reps to existing run."
    )
    parser.add_argument(
        "--n-reps", type=int, default=3,
        help="Number of replicates for --multirestart (default: 3)"
    )
    parser.add_argument(
        "--reanalyze", action="store_true",
        help="Re-analyze existing Phase 5 trajectories with PBC centering "
             "(gmx trjconv -pbc mol -center) and Q4 (last 25%%) RMSD. No new MD. "
             "Recomputes selectivity matrix from corrected analysis."
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
    5: "phase5/phase5_rebinding_results.json",
    6: "phase6/phase6_recipes.json",
}


def _check_phase_completed(phase_num: int, output_dir: str,
                           target_names: list = None) -> bool:
    """Check if a phase's result file exists AND covers all requested targets.

    If target_names is given, also verify each target has non-empty entry
    in the JSON. Phases that save partial results per-target (e.g., Phase 4)
    will return False here if any target is missing, so the phase re-runs
    and the per-target resume logic inside the phase skips completed targets.
    """
    result_file = _PHASE_RESULT_FILES.get(phase_num)
    if result_file is None:
        return False
    path = Path(output_dir) / result_file
    if not path.exists():
        return False

    if target_names:
        try:
            with open(path) as f:
                data = json.load(f)
            for t in target_names:
                entry = data.get(t)
                if not isinstance(entry, dict) or not entry:
                    return False  # missing or empty target → re-run
        except Exception:
            return False
    return True


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
        from .phase5_rebinding import run_phase6 as run_phase5_rebinding
        result = run_phase5_rebinding(
            phase4_results=prev_results.get("phase4"),
            phase1_results=prev_results.get("phase1"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase5"),
        )

    elif phase_num == 6:
        from .phase6_recipe import run_phase5 as run_phase6_recipe
        result = run_phase6_recipe(
            phase3_results=prev_results.get("phase3"),
            phase4_results=prev_results.get("phase4"),
            phase5_results=prev_results.get("phase5"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase6"),
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
    if "all" in args.target:
        target_names = list(TARGETS.keys())
    else:
        target_names = args.target

    # Convergence-improvement modes (run instead of standard pipeline)
    if args.extend_drifters:
        from .phase5_rebinding import extend_drifting_mds
        logger.info(f"Extension mode: +{args.extend_ns} ns for drifting Phase 5 MDs")
        # output_dir=None lets the function auto-detect phase5_extended over phase5
        extend_drifting_mds(
            target_names=target_names,
            output_dir=None,
            extend_ns=args.extend_ns,
        )
        return

    if args.multirestart:
        from .phase5_rebinding import run_multirestart
        logger.info(f"Multi-restart mode: N={args.n_reps} replicates per snapshot")
        run_multirestart(
            target_names=target_names,
            n_reps=args.n_reps,
            output_dir=None,
        )
        return

    if args.reanalyze:
        from .phase5_rebinding import reanalyze_phase5
        logger.info("Re-analysis mode: PBC centering + Q4 RMSD on existing trajectories")
        reanalyze_phase5(
            target_names=target_names,
            output_dir=None,
        )
        return

    logger.info(f"Targets: {target_names}")
    logger.info(f"Output: {out_dir}")

    # Determine phases
    if args.phase == "all":
        phases = [1, 2, 3, 4, 5, 6]
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
            if _check_phase_completed(phase_num, out_dir, target_names):
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
