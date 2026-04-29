"""
Crosslinker ratio sweep for CD63/FWD4_CETES.

Runs pre-polymerization MD with TRIM crosslinker fractions of
3%, 5%, 8%, 10% and compares cavity stability metrics.

Compute estimate: 4 ratios × 30 ns ≈ 6-10 hours on RTX 4070 Ti.

Output:
  results/phase4_crosslinker_sweep/CD63/xl_{frac}/
    md/ — trajectory
    cavity_metrics.json — RMSF, contacts, H-bonds
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import (get_output_path, ALL_MONOMERS,
                              CROSSLINKER_RATIO_SWEEP, CROSSLINKER_SWEEP_MD_NS,
                              resolve_path)
from pipeline.utils_gromacs import run_full_md_pipeline, parameterize_monomer
from pipeline.utils_structure import smiles_to_mol2

TARGET = "CD63"
PC_ID = "FWD4_CETES"  # Best PC from Phase 3
FUNCTIONAL = ["NE", "GPTMS", "TTMS", "CETES"]
CROSSLINKER = "TRIM"
TOTAL_MONOMERS = 40  # fits in shell (avoids placement fallback to r>10nm)
# At TOTAL=40, ratios become: 3%→2/40=5%, 5%→2/40=5%, 8%→3/40=7.5%, 10%→4/40=10%
# We bypass round-to-nearest by using explicit n_xl values


def build_copies(crosslinker_fraction: float):
    """Compute integer copy numbers approximating target ratio.
    Explicit table for clean ratios with TOTAL=40."""
    # Hand-pick n_xl to match target ratio approximately
    table = {0.03: 1, 0.05: 2, 0.08: 3, 0.10: 4}
    n_xl = table.get(round(crosslinker_fraction, 2), max(1, round(TOTAL_MONOMERS * crosslinker_fraction)))
    n_func_total = TOTAL_MONOMERS - n_xl
    per_func = max(1, n_func_total // len(FUNCTIONAL))
    func_copies = {m: per_func for m in FUNCTIONAL}
    remainder = n_func_total - per_func * len(FUNCTIONAL)
    for i, m in enumerate(FUNCTIONAL):
        if i < remainder:
            func_copies[m] += 1
    return n_xl, func_copies


def analyze_cavity(md_dir: Path, time_ns: float) -> dict:
    """Quick cavity metric: epitope-monomer mindist, contacts, RMSF."""
    import subprocess
    from pipeline.config import GMX_BIN
    tpr = md_dir / "md.tpr"
    xtc = md_dir / "md.xtc"
    if not (tpr.exists() and xtc.exists()):
        return {"error": "MD outputs missing"}
    # Build group: Protein vs Other
    work = md_dir / "analysis"
    work.mkdir(exist_ok=True)
    # mindist Protein vs Other in last 50% of trajectory
    half = (time_ns * 1000) / 2
    end = time_ns * 1000
    res = {}
    try:
        # mindist
        mp = work / "mindist.xvg"
        cp = work / "ncontacts.xvg"
        cmd = [GMX_BIN, "mindist", "-f", str(xtc), "-s", str(tpr),
               "-od", str(mp), "-on", str(cp), "-d", "0.5",
               "-b", str(half), "-e", str(end), "-group"]
        proc = subprocess.run(cmd, input="1\n12\n", capture_output=True,
                              text=True, timeout=600)
        if mp.exists():
            import numpy as np
            md = np.loadtxt(mp, comments=['#', '@'])
            nc = np.loadtxt(cp, comments=['#', '@'])
            res["mindist_nm_mean"] = float(md[:, 1].mean())
            res["mindist_nm_std"] = float(md[:, 1].std())
            res["contacts_lt_5A_mean"] = float(nc[:, 1].mean())
            res["contacts_lt_5A_std"] = float(nc[:, 1].std())
        # Protein RMSF
        rf = work / "rmsf.xvg"
        cmd = [GMX_BIN, "rmsf", "-f", str(xtc), "-s", str(tpr),
               "-o", str(rf), "-b", str(half), "-e", str(end)]
        proc = subprocess.run(cmd, input="1\n", capture_output=True,
                              text=True, timeout=300)
        if rf.exists():
            import numpy as np
            arr = np.loadtxt(rf, comments=['#', '@'])
            res["rmsf_protein_A_mean"] = float(arr[:, 1].mean() * 10)
            res["rmsf_protein_A_max"] = float(arr[:, 1].max() * 10)
    except Exception as e:
        res["error"] = str(e)
    return res


def run_sweep():
    p1_path = get_output_path("phase1") / "phase1_results.json"
    with open(p1_path) as f:
        p1 = json.load(f)
    epitope_pdb = resolve_path(
        p1[TARGET].get("head_pdb", p1[TARGET]["epitope_pdb"]))

    output_root = Path(get_output_path("phase4")).parent / "phase4_crosslinker_sweep" / TARGET
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for frac in CROSSLINKER_RATIO_SWEEP:
        label = f"xl_{int(frac*100):02d}pct"
        print(f"\n{'='*60}\n  Crosslinker sweep: {label} (target {frac*100:.0f}%)\n{'='*60}")

        n_xl, func_copies = build_copies(frac)
        actual_frac = n_xl / (n_xl + sum(func_copies.values()))
        print(f"  Composition: {func_copies}, {CROSSLINKER}={n_xl} → actual frac {actual_frac:.3f}")

        work_dir = output_root / label
        work_dir.mkdir(parents=True, exist_ok=True)

        # Parameterize monomers
        param_dir = work_dir / "monomer_params"
        monomer_itps = []

        all_copies = {**func_copies, CROSSLINKER: n_xl}
        for m_name, n_copies in all_copies.items():
            m_info = ALL_MONOMERS.get(m_name)
            if m_info is None:
                continue
            mol2 = smiles_to_mol2(m_info["smiles"], m_name, param_dir)
            param = parameterize_monomer(mol2, m_name, param_dir)
            if param.get("itp"):
                for _ in range(n_copies):
                    monomer_itps.append(param)

        print(f"  Total monomer molecules: {len(monomer_itps)}")

        t0 = time.time()
        md_dir = work_dir / "md"
        if (md_dir / "md.gro").exists():
            print("  MD already done, skipping")
        else:
            md_result = run_full_md_pipeline(
                epitope_pdb, monomer_itps, md_dir,
                time_ns=CROSSLINKER_SWEEP_MD_NS,
                quick=False,
            )
            print(f"  MD elapsed: {time.time()-t0:.1f} s")

        # Analyze
        metrics = analyze_cavity(md_dir, CROSSLINKER_SWEEP_MD_NS)
        summary[label] = {
            "target_fraction": frac,
            "actual_fraction": actual_frac,
            "n_crosslinker": n_xl,
            "func_copies": func_copies,
            "time_ns": CROSSLINKER_SWEEP_MD_NS,
            "metrics": metrics,
        }
        print(f"  Metrics: {metrics}")

        with open(work_dir / "cavity_metrics.json", "w") as f:
            json.dump(summary[label], f, indent=2)

    # Final summary
    print(f"\n{'='*60}\n  SWEEP SUMMARY\n{'='*60}")
    for label, data in summary.items():
        m = data["metrics"]
        print(f"  {label} ({data['actual_fraction']:.3f}): "
              f"contacts={m.get('contacts_lt_5A_mean','?')}, "
              f"mindist={m.get('mindist_nm_mean','?')}, "
              f"RMSF_max={m.get('rmsf_protein_A_max','?')}")

    out_path = output_root / "sweep_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved to {out_path}")
    return summary


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    run_sweep()
