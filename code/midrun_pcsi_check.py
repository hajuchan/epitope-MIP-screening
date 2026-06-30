"""Mid-run PCSI check across all completed CD63 (10-snap × main + dual) and CD81 snapshots.

For each snapshot leg, compute persistent residue counts. Aggregate to per-snapshot
PCSI and report ensemble mean ± std + the fraction of snapshots above the
PASS (1.2) / STRONG (1.5) thresholds.
"""
import os, json
from pathlib import Path
os.chdir(Path(__file__).parent.parent)
import sys; sys.path.insert(0, "code")
import numpy as np
from analyze_persistent_contacts import compute_persistent_contacts

TARGETS = ["CD63", "CD81", "CD9"]


def pcsi_snapshot(base_dir, target, others, mode="main"):
    """Return dict {target_or_other: n_persistent} for one snapshot dir."""
    if mode == "dual":
        prefix = base_dir / "dual_imprinting"
    else:
        prefix = base_dir
    out = {}
    for label, sub in [(target, "rebind_own")] + [(o, f"rebind_{o}") for o in others]:
        traj = prefix / sub / "md" / "md.xtc"
        top = prefix / sub / "md" / "md.tpr"
        if not (traj.exists() and top.exists()):
            out[label] = None
            continue
        try:
            freq, n_pers = compute_persistent_contacts(traj, top)
            out[label] = n_pers
        except Exception as e:
            out[label] = f"err: {e}"
    return out


def aggregate(target, mode="main"):
    others = [t for t in TARGETS if t != target]
    per_snap = []
    for i in range(10):
        snap_dir = Path(f"results/phase5/{target}/snapshot_{i}")
        if not snap_dir.exists():
            continue
        res = pcsi_snapshot(snap_dir, target, others, mode=mode)
        own = res.get(target)
        if not isinstance(own, int):
            continue
        cross_vals = [v for k, v in res.items() if k != target and isinstance(v, int)]
        if not cross_vals:
            pcsi = float("inf")  # size-excluded all cross
            size_ex = True
        else:
            mx = max(cross_vals)
            pcsi = own / mx if mx > 0 else float("inf")
            size_ex = False
        per_snap.append({"snap": i, "own": own, "cross": res, "pcsi": pcsi,
                          "size_excluded_all_cross": size_ex})
        print(f"    snap_{i}: own={own:3d} cross={ {k:v for k,v in res.items() if k!=target} } "
              f"PCSI={'inf' if pcsi==float('inf') else f'{pcsi:.2f}'}")
    return per_snap


def summarize(target, snaps, label):
    if not snaps:
        print(f"  {target} {label}: NO DATA"); return None
    finite = [s["pcsi"] for s in snaps if s["pcsi"] != float("inf")]
    n_inf = sum(1 for s in snaps if s["pcsi"] == float("inf"))
    pass_n = sum(1 for s in snaps if s["pcsi"] > 1.2)
    strong_n = sum(1 for s in snaps if s["pcsi"] > 1.5)
    print(f"  {target} {label}:  n={len(snaps)}  "
          f"PCSI mean={np.mean(finite) if finite else 'N/A':.2f} ± "
          f"{np.std(finite) if len(finite)>1 else 0:.2f}  "
          f"(inf={n_inf})  PASS(>1.2)={pass_n}/{len(snaps)}  STRONG(>1.5)={strong_n}/{len(snaps)}")
    return {"n": len(snaps), "mean": float(np.mean(finite)) if finite else None,
            "std": float(np.std(finite)) if len(finite)>1 else None,
            "n_inf": n_inf, "n_pass": pass_n, "n_strong": strong_n}


def main():
    results = {}
    for target in ["CD63", "CD81"]:
        print(f"\n===== {target} main rebinding =====")
        main_snaps = aggregate(target, mode="main")
        results[f"{target}_main"] = summarize(target, main_snaps, "MAIN")
    print("\n===== CD63 dual-imprinting =====")
    dual_snaps = aggregate("CD63", mode="dual")
    results["CD63_dual"] = summarize("CD63", dual_snaps, "DUAL")
    out = Path("results/reports/midrun_pcsi_check.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
