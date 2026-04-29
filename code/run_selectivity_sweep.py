"""
Selectivity weight sweep for CD63 only.
Runs Phase 3 with w_sel = {0.0, 0.3, 0.5, 1.0} and compares DDG.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import get_output_path

# Load Phase 1/2 results (shared across all sweeps)
with open(get_output_path("phase1") / "phase1_results.json") as f:
    p1 = json.load(f)
with open(get_output_path("phase2") / "phase2_smd_results.json") as f:
    p2 = json.load(f)

w_sel_values = [0.0, 0.3, 0.5, 1.0]
results_summary = {}

for w_sel in w_sel_values:
    print(f"\n{'='*60}")
    print(f"  Phase 3 Sweep: w_sel = {w_sel}")
    print(f"{'='*60}")

    # Set config dynamically
    import pipeline.config as cfg
    cfg.SELECTIVITY_WEIGHT = w_sel

    # Output to separate directory
    out_dir = get_output_path("phase3") / f"sweep_wsel_{w_sel}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    from pipeline.phase3_mmsd import run_phase3
    result = run_phase3(
        phase1_results=p1,
        phase2_results=p2,
        target_names=["CD63"],
        output_dir=str(out_dir),
    )

    elapsed = time.time() - t0

    # Extract CD63 top PC
    cd63 = result.get("CD63", {})
    top_pcs = cd63.get("top_pcs", [])
    if top_pcs:
        best = top_pcs[0]
        results_summary[w_sel] = {
            "pc_id": best["pc_id"],
            "monomers": best["monomers"],
            "bo_objective": best.get("bo_objective"),
            "mmsd_sum": best.get("mmsd_sum"),
            "DDG_selectivity": best.get("DDG_selectivity"),
            "selectivity_penalty": best.get("selectivity_penalty"),
            "cross_target_be": best.get("cross_target_be"),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"\n  Result: {best['pc_id']} {best['monomers']}")
        print(f"  obj={best.get('bo_objective')}, DDG={best.get('DDG_selectivity')}")
    else:
        results_summary[w_sel] = {"error": "no results"}

    # Force reimport to pick up config change next iteration
    del sys.modules["pipeline.phase3_mmsd"]

# Summary
print(f"\n{'='*60}")
print("SELECTIVITY WEIGHT SWEEP SUMMARY — CD63")
print(f"{'='*60}")
print(f"{'w_sel':>6} {'PC':<22} {'Obj':>8} {'MMSD':>8} {'DDG':>8} {'Penalty':>8}")
for w, r in sorted(results_summary.items()):
    if "error" in r:
        print(f"{w:>6} ERROR")
        continue
    print(f"{w:>6} {r['pc_id']:<22} {r.get('bo_objective','?'):>8} "
          f"{r.get('mmsd_sum','?'):>8} {r.get('DDG_selectivity','?'):>8} "
          f"{r.get('selectivity_penalty','?'):>8}")

# Save
with open(get_output_path("phase3") / "selectivity_sweep_results.json", "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"\nResults saved to: {get_output_path('phase3')}/selectivity_sweep_results.json")
