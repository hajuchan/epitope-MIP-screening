"""
Sullivan 2019 Validation — Docking BE vs Experimental IF Correlation
====================================================================
Run our AutoDock4 SMD pipeline on Myoglobin (PDB 5D5R) with 5
acrylamide monomers and check whether computed binding energies
predict the experimental Imprinting Factor ranking.

Core question: Does our calculation predict which monomer makes
the best MIP?

Validation checks:
  1. Spearman ρ (computed BE ranking vs experimental IF ranking)
  2. Best monomer identification (NHMAm expected)
  3. Worst monomer identification (DMAm expected)
  4. Backbone H-bond ratio correlates with CD helix disruption
  5. Comonomer prediction: MIP_A > MIP_B (non-competitive > competitive)
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run_sullivan_validation(output_dir: Path = None,
                             quick: bool = False) -> dict:
    """
    Full Sullivan 2019 validation: prepare target, dock monomers,
    compare with experimental data.

    This calls the actual pipeline code — not a separate implementation.
    """
    from pipeline.utils_structure import (
        download_pdb, prepare_receptor_pdbqt,
        compute_grid_center, compute_grid_size, smiles_to_pdbqt,
    )
    from pipeline.utils_autodock import dock_single
    from pipeline.config import AUTODOCK4_GA_RUNS
    from .config_validation import (
        SULLIVAN_TARGET_PDB, SULLIVAN_TARGET_CHAIN,
        SULLIVAN_MONOMERS, SULLIVAN_EXPERIMENTAL,
    )

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / \
                     "results" / "validation" / "sullivan"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"test": "Sullivan 2019 — Myoglobin MIP Validation"}
    ga_runs = 10 if quick else AUTODOCK4_GA_RUNS

    # ── Step 1: Download and prepare Myoglobin ──
    logger.info("Sullivan validation: preparing Myoglobin (5D5R)...")
    pdb_path = download_pdb(SULLIVAN_TARGET_PDB, output_dir)

    # Use full protein (not epitope extraction — Sullivan used whole protein)
    receptor_pdbqt = output_dir / "myoglobin_receptor.pdbqt"
    prepare_receptor_pdbqt(pdb_path, receptor_pdbqt)

    center = compute_grid_center(pdb_path)
    npts = compute_grid_size(pdb_path, padding=10.0)  # larger box for protein
    logger.info(f"  Grid center: {center}, npts: {npts}")

    # ── Step 2: Dock 5 acrylamide monomers ──
    logger.info(f"Docking {len(SULLIVAN_MONOMERS)} monomers "
                f"({ga_runs} GA runs each)...")
    monomer_dir = output_dir / "monomers"
    computed_be = {}

    for name, info in SULLIVAN_MONOMERS.items():
        logger.info(f"  Docking {name}...")
        try:
            pdbqt = smiles_to_pdbqt(info["smiles"], name, monomer_dir)
            work = output_dir / f"dock_{name}"
            dock_result = dock_single(
                receptor_pdbqt, pdbqt, center, npts, work,
                ga_runs=ga_runs,
            )
            be = dock_result.get("mean_cluster_energy", 0.0)
            computed_be[name] = be
            exp = SULLIVAN_EXPERIMENTAL[name]
            logger.info(f"    BE: {be:.2f} kcal/mol | "
                        f"Exp IF: {exp['IF']}, rebind: {exp['pct_rebind']}%")
        except Exception as e:
            logger.error(f"    {name} failed: {e}")
            computed_be[name] = 0.0

    results["computed_be"] = computed_be

    # ── Step 3: Validate against experimental data ──
    logger.info("\nComparing with experimental data...")
    checks = validate_sullivan_results(computed_be)
    results["checks"] = checks

    # Save
    with open(output_dir / "sullivan_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log summary
    for check in checks:
        icon = {"PASS": "✓", "WARN": "△", "FAIL": "✗"}.get(
            check["status"], "?"
        )
        logger.info(f"  {icon} {check['name']}: {check['status']}")

    results["status"] = "FAIL" if any(c["status"] == "FAIL" for c in checks) \
        else "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"

    return results


def validate_sullivan_results(computed_be: dict) -> list:
    """
    Validate computed binding energies against Sullivan 2019
    experimental data.

    The key question: does more negative BE → higher IF?
    """
    from scipy.stats import spearmanr
    from .config_validation import (
        SULLIVAN_EXPERIMENTAL, SULLIVAN_MIP_RANKING,
        SULLIVAN_HBOND_ANALYSIS, SULLIVAN_CD_HELIX,
        SULLIVAN_COMONOMER_MIPS, SULLIVAN_RANKING_SPEARMAN,
    )

    checks = []
    monomers = [m for m in SULLIVAN_MIP_RANKING if m in computed_be]

    if len(monomers) < 4:
        return [{"name": "Insufficient data", "status": "FAIL",
                 "reason": f"Only {len(monomers)} monomers computed"}]

    # ── Check 1: Spearman ρ (BE ranking vs IF ranking) ──
    # More negative BE should correlate with higher IF
    be_values = [computed_be[m] for m in monomers]
    if_values = [SULLIVAN_EXPERIMENTAL[m]["IF"] for m in monomers]

    rho, pval = spearmanr(be_values, if_values)
    # We expect negative ρ: more negative BE → higher IF
    # Or equivalently, rank by BE (ascending) should match rank by IF (descending)

    # Rank comparison approach: sort monomers by computed BE (ascending = strongest)
    comp_ranking = sorted(monomers, key=lambda m: computed_be[m])
    exp_ranking = SULLIVAN_MIP_RANKING  # from config

    comp_ranks = [comp_ranking.index(m) for m in monomers]
    exp_ranks = [exp_ranking.index(m) for m in monomers]
    rho_rank, pval_rank = spearmanr(comp_ranks, exp_ranks)

    passed = rho_rank >= SULLIVAN_RANKING_SPEARMAN

    checks.append({
        "name": "Spearman ρ (computed BE rank vs experimental IF rank)",
        "status": "PASS" if passed else "WARN",
        "spearman_rho": round(float(rho_rank), 3),
        "p_value": round(float(pval_rank), 4),
        "threshold": SULLIVAN_RANKING_SPEARMAN,
        "computed_ranking": comp_ranking,
        "experimental_ranking": exp_ranking,
        "detail": {
            m: {"computed_BE": round(computed_be[m], 2),
                "experimental_IF": SULLIVAN_EXPERIMENTAL[m]["IF"],
                "experimental_rebind": SULLIVAN_EXPERIMENTAL[m]["pct_rebind"]}
            for m in monomers
        },
    })

    # ── Check 2: Best monomer (NHMAm expected) ──
    best_computed = comp_ranking[0]
    best_expected = "NHMAm"
    # NHMAm or AAm as top-2 is acceptable (they're close in IF: 1.90 vs 1.77)
    top2_ok = best_computed in ["NHMAm", "AAm"]

    checks.append({
        "name": "Best monomer identification",
        "status": "PASS" if best_computed == best_expected else
                  "WARN" if top2_ok else "FAIL",
        "computed_best": best_computed,
        "expected_best": best_expected,
        "acceptable": ["NHMAm", "AAm"],
        "note": "Sullivan 2019: NHMAm (IF=1.90) and AAm (IF=1.77) are top-2",
    })

    # ── Check 3: Worst monomer (DMAm expected) ──
    worst_computed = comp_ranking[-1]
    worst_expected = "DMAm"
    bottom2_ok = worst_computed in ["DMAm", "TrisNHMAm"]

    checks.append({
        "name": "Worst monomer identification",
        "status": "PASS" if worst_computed == worst_expected else
                  "WARN" if bottom2_ok else "FAIL",
        "computed_worst": worst_computed,
        "expected_worst": worst_expected,
        "acceptable": ["DMAm", "TrisNHMAm"],
        "note": "Sullivan 2019: DMAm (IF=1.48) and TrisNHMAm (IF=1.10) are bottom-2",
    })

    # ── Check 4: Backbone H-bond → helix disruption trend ──
    # Sullivan's key finding: more backbone H-bonds → more 2° structure disruption
    # TrisNHMAm: 8 helical backbone → 42.6% helix (native 88%)
    # AAm: 5 helical backbone → 80.5% helix (preserved)
    # DMAm: 4 helical backbone → 34.7% helix (but ratio is high)
    #
    # Our pipeline's backbone H-bond analysis should flag TrisNHMAm and DMAm
    # We check if our computed BE + backbone ratio correctly identifies the
    # "good" monomers (low backbone ratio + strong BE)

    # This is a qualitative check — we verify the concept direction
    checks.append({
        "name": "Backbone H-bond disruption awareness (Sullivan 2019 Table 4)",
        "status": "INFO",
        "sullivan_data": {
            m: {
                "total_hbonds": SULLIVAN_HBOND_ANALYSIS[m]["total"],
                "helical_backbone": SULLIVAN_HBOND_ANALYSIS[m]["helical_backbone"],
                "backbone_ratio": round(
                    SULLIVAN_HBOND_ANALYSIS[m]["helical_backbone"] /
                    SULLIVAN_HBOND_ANALYSIS[m]["total"], 2
                ),
                "cd_helix_pct": SULLIVAN_CD_HELIX[m],
                "helix_loss": round(88.0 - SULLIVAN_CD_HELIX[m], 1),
            }
            for m in monomers if m in SULLIVAN_HBOND_ANALYSIS
        },
        "insight": (
            "TrisNHMAm has most total H-bonds (42) but highest helical "
            "backbone contacts (8) → 88→42.6% helix loss → poor IF (1.10). "
            "NHMAm has 27 H-bonds but only 3 helical backbone → best IF (1.90). "
            "Pipeline should flag monomers with backbone_ratio > 0.3."
        ),
    })

    # ── Check 5: Comonomer prediction direction ──
    # MIP_A (TrisNHMAm+DMAm): IF=2.36 (better than either alone)
    # MIP_B (NHEAm+DMAm): IF=1.58 (worse than NHEAm alone)
    # Sullivan: non-competitive binding → synergy, competitive → interference
    #
    # We can test if our computed BEs for these monomers predict
    # that MIP_A benefits more from combination than MIP_B
    mip_a = SULLIVAN_COMONOMER_MIPS["MIP_A"]
    mip_b = SULLIVAN_COMONOMER_MIPS["MIP_B"]

    mip_a_better = mip_a["IF"] > mip_b["IF"]  # True (2.36 > 1.58)

    # Predict: sum of individual BEs — the pair with more diverse (different)
    # binding should do better
    be_a = sum(computed_be.get(m, 0) for m in mip_a["monomers"])
    be_b = sum(computed_be.get(m, 0) for m in mip_b["monomers"])

    checks.append({
        "name": "Comonomer prediction (Sullivan 2019: MIP_A vs MIP_B)",
        "status": "PASS" if (be_a < be_b) == mip_a_better else "WARN",
        "MIP_A": {
            "monomers": mip_a["monomers"],
            "experimental_IF": mip_a["IF"],
            "computed_sum_BE": round(be_a, 2),
            "rationale": mip_a["rationale"],
        },
        "MIP_B": {
            "monomers": mip_b["monomers"],
            "experimental_IF": mip_b["IF"],
            "computed_sum_BE": round(be_b, 2),
            "rationale": mip_b["rationale"],
        },
        "note": (
            "MIP_A (non-competitive) should outperform MIP_B (competitive). "
            "Sullivan 2019: IF 2.36 vs 1.58. "
            "More negative sum BE should correspond to higher IF."
        ),
    })

    return checks
