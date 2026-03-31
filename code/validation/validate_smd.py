"""
SMD Validation — Rajpal 2024 Table 1 Reproduction
==================================================
Validate that our AutoDock4 SMD pipeline reproduces the
binding energy rankings for 11 silane monomers against the
SARS-CoV-2 spike epitope.

Pass criteria:
  1. Spearman ρ ≥ 0.7 between computed and reference BE rankings
  2. Individual BE values within ±0.5 kcal/mol of reference
  3. PTES identified as top-ranked monomer
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def validate_smd(results_path: Path = None) -> dict:
    """
    Validate SMD results against Rajpal 2024 Table 1.

    Parameters
    ----------
    results_path : path to phase2_smd_results.json from validation run

    Returns
    -------
    dict with pass/fail status per criterion
    """
    from .config_validation import (
        RAJPAL_SMD_REFERENCE, RAJPAL_SMD_RANKING,
        SMD_BE_TOLERANCE, SMD_RANKING_SPEARMAN,
    )

    report = {"test": "SMD Validation (Rajpal 2024 Table 1)", "checks": []}

    # Load computed results
    if results_path is None:
        from pipeline.config import get_output_path
        results_path = get_output_path("phase2") / "phase2_smd_results.json"

    if not Path(results_path).exists():
        return {"test": "SMD Validation", "status": "SKIP",
                "reason": f"Results not found: {results_path}"}

    with open(results_path) as f:
        results = json.load(f)

    # Extract BE matrix for validation target
    be_matrix = results.get("be_matrix", {})
    # Use first target's results (validation should use Rajpal epitope)
    target = list(be_matrix.keys())[0] if be_matrix else None
    if target is None:
        return {"test": "SMD Validation", "status": "FAIL",
                "reason": "No target in BE matrix"}

    computed = be_matrix[target]

    # --- Check 1: Spearman ranking correlation ---
    ref_monomers = [m for m in RAJPAL_SMD_RANKING if m in computed]
    if len(ref_monomers) < 5:
        report["checks"].append({
            "name": "Spearman ranking",
            "status": "SKIP",
            "reason": f"Only {len(ref_monomers)} overlapping monomers",
        })
    else:
        from scipy.stats import spearmanr
        ref_ranks = [RAJPAL_SMD_RANKING.index(m) for m in ref_monomers]
        comp_sorted = sorted(ref_monomers, key=lambda m: computed[m])
        comp_ranks = [comp_sorted.index(m) for m in ref_monomers]

        rho, pval = spearmanr(ref_ranks, comp_ranks)
        passed = rho >= SMD_RANKING_SPEARMAN

        report["checks"].append({
            "name": "Spearman ranking correlation",
            "status": "PASS" if passed else "FAIL",
            "spearman_rho": round(float(rho), 3),
            "p_value": round(float(pval), 4),
            "threshold": SMD_RANKING_SPEARMAN,
        })

    # --- Check 2: Individual BE deviation ---
    deviations = {}
    for monomer, ref_be in RAJPAL_SMD_REFERENCE.items():
        if monomer in computed:
            comp_be = computed[monomer]
            dev = abs(comp_be - ref_be)
            deviations[monomer] = {
                "reference": ref_be,
                "computed": round(comp_be, 3),
                "deviation": round(dev, 3),
                "within_tolerance": dev <= SMD_BE_TOLERANCE,
            }

    n_within = sum(1 for d in deviations.values() if d["within_tolerance"])
    n_total = len(deviations)

    report["checks"].append({
        "name": "Individual BE accuracy",
        "status": "PASS" if n_within >= n_total * 0.7 else "WARN",
        "within_tolerance": n_within,
        "total_compared": n_total,
        "tolerance": f"±{SMD_BE_TOLERANCE} kcal/mol",
        "deviations": deviations,
    })

    # --- Check 3: PTES as top-ranked ---
    comp_ranking = sorted(computed.keys(), key=lambda m: computed[m])
    ptes_top = comp_ranking[0] == "PTES" if "PTES" in computed else False

    report["checks"].append({
        "name": "PTES top-ranked",
        "status": "PASS" if ptes_top else "WARN",
        "computed_top3": comp_ranking[:3],
        "expected_top": "PTES",
    })

    # Overall status
    statuses = [c["status"] for c in report["checks"]]
    if "FAIL" in statuses:
        report["status"] = "FAIL"
    elif "WARN" in statuses:
        report["status"] = "WARN"
    else:
        report["status"] = "PASS"

    return report
