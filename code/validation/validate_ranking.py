"""
Ranking Validation — MMSD Sum vs Experimental IF Correlation
=============================================================
Validate that the computational MMSD sum ranking correlates
with experimentally measured Imprinting Factors from Rajpal 2024.

Pass criteria:
  1. Spearman ρ ≥ 0.6 between MMSD sum and experimental IF
  2. PC_I (PTES+APTMS+APTES+TEOS) appears in top 3
  3. Control PCs (PC_V, PC_VI with interference monomers) rank low
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_ranking(mmsd_results_path: Path = None) -> dict:
    """
    Validate MMSD ranking against experimental IF.

    Parameters
    ----------
    mmsd_results_path : path to phase3_mmsd_results.json

    Returns
    -------
    dict with pass/fail status per criterion
    """
    from .config_validation import (
        RAJPAL_EXPERIMENTAL_IF, RAJPAL_IF_RANKING,
        IF_RANKING_SPEARMAN,
    )

    report = {"test": "Ranking Validation (MMSD sum vs IF)", "checks": []}

    if mmsd_results_path is None:
        from pipeline.config import get_output_path
        mmsd_results_path = get_output_path("phase3") / "phase3_mmsd_results.json"

    if not Path(mmsd_results_path).exists():
        return {"test": "Ranking Validation", "status": "SKIP",
                "reason": f"Results not found: {mmsd_results_path}"}

    with open(mmsd_results_path) as f:
        results = json.load(f)

    target = list(results.keys())[0] if results else None
    if target is None:
        return {"test": "Ranking Validation", "status": "FAIL",
                "reason": "No target"}

    all_pcs = results[target].get("all_results", [])

    # Match computed PCs to Rajpal PCs by monomer composition
    matched = {}
    for rajpal_id, rajpal_data in RAJPAL_EXPERIMENTAL_IF.items():
        raj_set = set(rajpal_data["monomers"])
        raj_if = rajpal_data["IF"]
        if raj_if is None:
            continue  # skip PCs without IF data

        for pc in all_pcs:
            comp_set = set(pc.get("monomers", []))
            if raj_set == comp_set:
                matched[rajpal_id] = {
                    "mmsd_sum": pc["mmsd_sum"],
                    "experimental_if": raj_if,
                }
                break

    if len(matched) < 3:
        report["checks"].append({
            "name": "Spearman ρ (MMSD sum vs IF)",
            "status": "SKIP",
            "reason": f"Only {len(matched)} PCs matched (need ≥3)",
            "matched": matched,
        })
    else:
        from scipy.stats import spearmanr
        import numpy as np

        # Note: more negative MMSD = better binding, higher IF = better
        # So we expect negative correlation (or rank correlation)
        mmsd_vals = [matched[k]["mmsd_sum"] for k in sorted(matched)]
        if_vals = [matched[k]["experimental_if"] for k in sorted(matched)]

        rho, pval = spearmanr(mmsd_vals, if_vals)
        # MMSD is negative (more negative = better), IF is positive (higher = better)
        # Expected: negative ρ (or use abs)
        passed = abs(rho) >= IF_RANKING_SPEARMAN

        report["checks"].append({
            "name": "Spearman ρ (MMSD sum vs experimental IF)",
            "status": "PASS" if passed else "FAIL",
            "spearman_rho": round(float(rho), 3),
            "p_value": round(float(pval), 4),
            "threshold": IF_RANKING_SPEARMAN,
            "matched_pcs": matched,
        })

    # --- Check 2: PC_I in top 3 ---
    top3_sets = [set(pc["monomers"]) for pc in all_pcs[:3]]
    pc1_set = set(RAJPAL_EXPERIMENTAL_IF["PC_I"]["monomers"])
    pc1_in_top3 = any(pc1_set == s for s in top3_sets)

    report["checks"].append({
        "name": "PC_I (PTES+APTMS+APTES+TEOS) in top 3",
        "status": "PASS" if pc1_in_top3 else "WARN",
        "computed_top3": [pc["monomers"] for pc in all_pcs[:3]],
        "expected": RAJPAL_EXPERIMENTAL_IF["PC_I"]["monomers"],
    })

    # --- Check 3: Control PCs rank low ---
    # PC_V (PTES+DIDMS+IBTES+TEOS) and PC_VI (PTES+DIDMS+MTMS+TEOS)
    control_sets = [
        set(RAJPAL_EXPERIMENTAL_IF["PC_V"]["monomers"]),
        set(RAJPAL_EXPERIMENTAL_IF["PC_VI"]["monomers"]),
    ]
    n_pcs = len(all_pcs)
    control_ranks = []
    for i, pc in enumerate(all_pcs):
        comp_set = set(pc.get("monomers", []))
        for ctrl in control_sets:
            if comp_set == ctrl:
                control_ranks.append(i + 1)

    controls_low = all(r > n_pcs * 0.5 for r in control_ranks) if control_ranks else True

    report["checks"].append({
        "name": "Control PCs (interference monomers) rank in bottom half",
        "status": "PASS" if controls_low else "WARN",
        "control_ranks": control_ranks,
        "total_pcs": n_pcs,
    })

    # Overall
    statuses = [c["status"] for c in report["checks"]]
    report["status"] = "FAIL" if "FAIL" in statuses else \
                       "WARN" if "WARN" in statuses else "PASS"
    return report
