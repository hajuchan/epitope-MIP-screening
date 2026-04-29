"""
Extended Phase 5 rebinding: N=10 snapshots × 50 ns × 3 targets.

Uses updated config (REBINDING_N_SNAPSHOTS=10, REBINDING_MD_NS=50).
Outputs to results/phase5_extended/ to preserve existing phase5/ data.

Compute estimate: 10 snap × 3 targets × (50 ns own + 50 ns CD81 + 50 ns CD9
+ 10 ns removal) ≈ 4800 ns total → ~5-8 days on RTX 4070 Ti.

Recommended: run on stable GPU, monitor periodically.
"""
import json
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import get_output_path, OUTPUT_DIRS

# Override Phase 5 output to extended dir
EXTENDED_DIR = Path(OUTPUT_DIRS["phase5"]).parent / "phase5_extended"
EXTENDED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("phase5_ext")


def main():
    p4_path = get_output_path("phase4") / "phase4_md_results.json"
    p1_path = get_output_path("phase1") / "phase1_results.json"

    if not p4_path.exists():
        logger.error("Phase 4 results not found")
        return

    with open(p4_path) as f:
        p4 = json.load(f)
    with open(p1_path) as f:
        p1 = json.load(f)

    # Only re-run targets with weak selectivity at n=5; CD81 is already excellent
    targets = [t for t in p4 if t in ("CD63", "CD9")]
    logger.info(f"Targets: {targets} (CD81 skipped — already SI>3 with p<0.05)")
    logger.info(f"Output: {EXTENDED_DIR}")

    from pipeline.phase5_rebinding import run_phase6 as run_phase5
    from pipeline.config import REBINDING_N_SNAPSHOTS, REBINDING_MD_NS
    logger.info(f"Config: N_SNAP={REBINDING_N_SNAPSHOTS}, MD_NS={REBINDING_MD_NS}")

    t0 = time.time()
    result = run_phase5(
        phase4_results=p4,
        phase1_results=p1,
        target_names=targets,
        output_dir=str(EXTENDED_DIR),
    )
    elapsed = time.time() - t0
    logger.info(f"Phase 5 extended done in {elapsed/3600:.1f} hours")

    # Save
    out = EXTENDED_DIR / "phase5_extended_summary.json"
    with open(out, "w") as f:
        json.dump({
            "n_snapshots": REBINDING_N_SNAPSHOTS,
            "rebinding_ns": REBINDING_MD_NS,
            "elapsed_hours": elapsed / 3600,
            "results": result,
        }, f, indent=2, default=str)
    logger.info(f"Saved {out}")


if __name__ == "__main__":
    main()
