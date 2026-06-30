"""Compute PCSI for CD63 dual-imprinting (APBA) trajectories."""
import os, json
from pathlib import Path
os.chdir(Path(__file__).parent.parent)
import sys; sys.path.insert(0, "code")
from analyze_persistent_contacts import compute_persistent_contacts

base = Path("results/phase5/CD63/snapshot_0/dual_imprinting")
ligands = {"CD63": "rebind_own", "CD81": "rebind_CD81", "CD9": "rebind_CD9"}

res = {}
for lbl, sub in ligands.items():
    traj = base / sub / "md" / "md.xtc"
    top = base / sub / "md" / "md.tpr"
    if not (traj.exists() and top.exists()):
        print(f"  {lbl}: MISSING trajectory")
        res[lbl] = None
        continue
    freq, n_pers = compute_persistent_contacts(traj, top)
    total = len(freq)
    res[lbl] = {
        "n_persistent_residues": n_pers,
        "total_residues": total,
        "fraction_persistent": n_pers / total if total else 0,
        "mean_contact_freq": float(sum(freq.values()) / total) if total else 0,
    }
    print(f"  {lbl}: persistent={n_pers}/{total} "
          f"({res[lbl]['fraction_persistent']*100:.1f}%), "
          f"mean_freq={res[lbl]['mean_contact_freq']:.3f}")

own = res["CD63"]["n_persistent_residues"]
cross = [res[t]["n_persistent_residues"] for t in ("CD81", "CD9") if res.get(t)]
max_cross = max(cross) if cross else 0
pcsi = own / max_cross if max_cross > 0 else float("inf")
print(f"\nCD63 DUAL PCSI = {own}/{max_cross} = {pcsi:.2f}")
verdict = ("STRONG" if pcsi > 1.5 else "MODERATE" if pcsi > 1.2
           else "WEAK" if pcsi > 1.0 else "CROSS-REACTIVE")
print(f"Verdict: {verdict}")

out = Path("results/reports/phase5_cd63_dual_pcsi.json")
json.dump({"CD63_dual": {"target": "CD63", **res,
                          "PCSI": pcsi if pcsi != float('inf') else None,
                          "verdict": verdict}},
          open(out, "w"), indent=2, default=str)
print(f"Saved: {out}")
