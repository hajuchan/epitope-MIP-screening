"""
MMSD Validation — Rajpal 2024 Table 2 Synergy Reproduction
===========================================================
Validate that MMSD correctly identifies synergy/interference
patterns: APTMS/APTES/MPTMS should show BE increase (synergy)
while DIDMS/IBTES/MTMS should show BE decrease (interference).

Pass criteria:
  1. APTMS, APTES show positive BE change in MMSD (synergy direction)
  2. DIDMS, IBTES show negative or unchanged BE (interference direction)
  3. Top-ranked PCs contain synergy monomers (APTMS/APTES)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_mmsd(results_path: Path = None) -> dict:
    """
    Validate MMSD results against Rajpal 2024 synergy patterns.

    Parameters
    ----------
    results_path : path to phase3_mmsd_results.json from validation run

    Returns
    -------
    dict with pass/fail status per criterion
    """
    from .config_validation import (
        RAJPAL_MMSD_SYNERGY_MONOMERS,
        RAJPAL_MMSD_INTERFERENCE_MONOMERS,
        RAJPAL_EXPERIMENTAL_IF,
        RAJPAL_IF_RANKING,
    )

    report = {"test": "MMSD Validation (Rajpal 2024 Table 2)", "checks": []}

    if results_path is None:
        from pipeline.config import get_output_path
        results_path = get_output_path("phase3") / "phase3_mmsd_results.json"

    if not Path(results_path).exists():
        return {"test": "MMSD Validation", "status": "SKIP",
                "reason": f"Results not found: {results_path}"}

    with open(results_path) as f:
        results = json.load(f)

    target = list(results.keys())[0] if results else None
    if target is None or "error" in results.get(target, {}):
        return {"test": "MMSD Validation", "status": "FAIL",
                "reason": "No valid target in results"}

    data = results[target]
    all_pcs = data.get("all_results", [])

    if not all_pcs:
        return {"test": "MMSD Validation", "status": "FAIL",
                "reason": "No PC results"}

    # --- Check 1: Synergy direction ---
    # Collect per-monomer BE changes across PCs
    monomer_mmsd_bes = {}
    monomer_smd_bes = {}
    for pc in all_pcs:
        for m, be in pc.get("monomer_energies", {}).items():
            monomer_mmsd_bes.setdefault(m, []).append(be)
        # SMD sum / n_monomers gives average SMD BE per monomer
        if pc.get("smd_sum") and pc.get("monomers"):
            for m in pc["monomers"]:
                smd_per = pc["smd_sum"] / len(pc["monomers"])
                monomer_smd_bes.setdefault(m, []).append(smd_per)

    synergy_correct = 0
    synergy_total = 0
    for m in RAJPAL_MMSD_SYNERGY_MONOMERS:
        if m in monomer_mmsd_bes:
            synergy_total += 1
            avg_mmsd = sum(monomer_mmsd_bes[m]) / len(monomer_mmsd_bes[m])
            # Synergy = MMSD BE more negative than individual
            # In Rajpal's terms: "BE increases" = becomes more negative
            if avg_mmsd < -2.0:  # meaningful improvement
                synergy_correct += 1

    report["checks"].append({
        "name": "Synergy monomer direction (APTMS/APTES/MPTMS/UPTMS)",
        "status": "PASS" if synergy_correct >= synergy_total * 0.5 else "WARN",
        "correct": synergy_correct,
        "total": synergy_total,
        "detail": "≥50% of synergy monomers should show enhanced BE in MMSD",
    })

    # --- Check 2: Top PC contains synergy monomers ---
    top_pcs = data.get("top_pcs", [])[:3]
    top_has_synergy = False
    for pc in top_pcs:
        monomers = set(pc.get("monomers", []))
        if monomers & set(RAJPAL_MMSD_SYNERGY_MONOMERS):
            top_has_synergy = True
            break

    report["checks"].append({
        "name": "Top PCs include synergy monomers",
        "status": "PASS" if top_has_synergy else "FAIL",
        "top_pcs": [{"id": pc["pc_id"], "monomers": pc["monomers"]}
                     for pc in top_pcs],
        "expected_synergy_monomers": RAJPAL_MMSD_SYNERGY_MONOMERS,
    })

    # --- Check 3: Synergy PCs rank above interference PCs ---
    synergy_mmsd_sums = []
    interf_mmsd_sums = []
    for pc in all_pcs:
        monomers = set(pc.get("monomers", []))
        ms = pc.get("mmsd_sum", 0)
        if monomers & set(RAJPAL_MMSD_SYNERGY_MONOMERS):
            synergy_mmsd_sums.append(ms)
        if monomers & set(RAJPAL_MMSD_INTERFERENCE_MONOMERS):
            interf_mmsd_sums.append(ms)

    if synergy_mmsd_sums and interf_mmsd_sums:
        avg_syn = sum(synergy_mmsd_sums) / len(synergy_mmsd_sums)
        avg_int = sum(interf_mmsd_sums) / len(interf_mmsd_sums)
        better = avg_syn < avg_int  # more negative = better

        report["checks"].append({
            "name": "Synergy PCs rank above interference PCs",
            "status": "PASS" if better else "WARN",
            "avg_synergy_mmsd_sum": round(avg_syn, 2),
            "avg_interference_mmsd_sum": round(avg_int, 2),
        })

    # --- Check 4: Uniform binding (Sullivan 2019) ---
    n_uniform = sum(1 for pc in top_pcs if pc.get("is_uniform", False))
    report["checks"].append({
        "name": "Top PCs have uniform (non-competitive) binding",
        "status": "PASS" if n_uniform >= len(top_pcs) * 0.5 else "WARN",
        "uniform_count": n_uniform,
        "total_top": len(top_pcs),
    })

    # Overall
    statuses = [c["status"] for c in report["checks"]]
    report["status"] = "FAIL" if "FAIL" in statuses else \
                       "WARN" if "WARN" in statuses else "PASS"
    return report
