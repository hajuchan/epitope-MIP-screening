"""
Phase 5: Synthesis Recipe Generation
=====================================
Compile final polymer compositions and generate synthesis protocols
for sol-gel or free-radical polymerization of epitope-imprinted MIPs.

Reference:
  Rajpal et al., Sci. Rep. 2024 — experimental PC synthesis
  Bhakta et al., ACS Appl. Mater. Interfaces 2015 — sol-gel protocol
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_phase5(phase3_results: dict = None,
               phase4_results: dict = None,
               target_names: list = None,
               output_dir: str = None) -> dict:
    """
    Phase 5 entry point: generate synthesis recipes.

    Returns dict with recipes per target, including monomer ratios,
    synthesis protocol, and characterization recommendations.
    """
    from .config import (TARGETS, ALL_MONOMERS, CROSSLINKERS,
                         SILANE_MONOMERS, VINYL_MONOMERS,
                         POLYMERIZATION_SILANE, POLYMERIZATION_VINYL,
                         get_output_path)

    if output_dir is None:
        output_dir = str(get_output_path("phase5"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    if phase3_results is None:
        p3_path = get_output_path("phase3") / "phase3_mmsd_results.json"
        with open(p3_path) as f:
            phase3_results = json.load(f)

    if phase4_results is None:
        p4_path = get_output_path("phase4") / "phase4_md_results.json"
        if p4_path.exists():
            with open(p4_path) as f:
                phase4_results = json.load(f)
        else:
            phase4_results = {}

    if target_names is None:
        target_names = list(phase3_results.keys())

    recipes = {}

    for target in target_names:
        p3 = phase3_results.get(target, {})
        p4 = phase4_results.get(target, {})

        if "error" in p3:
            continue

        logger.info(f"\n{'='*20} Recipe: {target} {'='*20}")

        # Select best PC (from Phase 4 if available, else Phase 3)
        best_pc = _select_best_pc(p3, p4)
        if best_pc is None:
            logger.warning(f"[{target}] No valid PC found")
            continue

        monomers = best_pc["monomers"]

        # Determine polymerization type
        silane_count = sum(1 for m in monomers if m in SILANE_MONOMERS)
        vinyl_count = sum(1 for m in monomers if m in VINYL_MONOMERS)
        poly_type = POLYMERIZATION_SILANE if silane_count >= vinyl_count \
            else POLYMERIZATION_VINYL

        # Generate recipe
        recipe = _generate_recipe(
            target, monomers, best_pc, poly_type,
            ALL_MONOMERS, TARGETS[target],
        )
        recipes[target] = recipe

        # Log recipe
        _log_recipe(target, recipe)

    # Save recipes
    with open(output_dir / "phase5_recipes.json", "w") as f:
        json.dump(recipes, f, indent=2)

    # Generate human-readable protocol
    protocol_path = output_dir / "synthesis_protocol.txt"
    _write_protocol(recipes, protocol_path)
    logger.info(f"\nProtocol → {protocol_path}")

    return recipes


def _select_best_pc(p3_data: dict, p4_data: dict) -> dict:
    """
    Select best polymer composition.
    Priority: Phase 4 MM-PBSA > Phase 3 MMSD sum.
    """
    top_pcs = p3_data.get("top_pcs", [])
    if not top_pcs:
        return None

    # If Phase 4 results available, rank by MM-PBSA
    if p4_data:
        best_dg = float("inf")
        best_pc = None
        for pc in top_pcs:
            pc_id = pc["pc_id"]
            md_data = p4_data.get(pc_id, {})
            dg = md_data.get("mmpbsa", {}).get("delta_total_kcal", 0)
            if dg < best_dg:
                best_dg = dg
                best_pc = pc
        if best_pc:
            return best_pc

    # Fallback: Phase 3 best MMSD sum
    return top_pcs[0]


def _generate_recipe(target: str, monomers: list,
                      pc_data: dict, poly_type: str,
                      monomer_lib: dict, target_cfg: dict) -> dict:
    """Generate a complete synthesis recipe."""
    from .config import CROSSLINKERS

    # Categorize monomers
    functional = [m for m in monomers if m not in CROSSLINKERS]
    crosslinker = [m for m in monomers if m in CROSSLINKERS]

    # Molar ratios (Rajpal 2024: equal molar for functional, 10x for TEOS)
    ratios = {}
    for m in functional:
        ratios[m] = 1.0
    for m in crosslinker:
        ratios[m] = 10.0  # cross-linker excess

    # Interaction profile
    interactions = {}
    for m in monomers:
        info = monomer_lib.get(m, {})
        interactions[m] = info.get("interaction", "unknown")

    # Special notes
    notes = []
    if "APBA" in monomers:
        notes.append(
            "APBA (boronic acid) provides glycan-selective recognition. "
            f"Target {target} has {target_cfg.get('n_glycan_sites', 0)} "
            "N-glycan sites. Ensure pH 7.4 for optimal boronate-diol binding."
        )
    if target_cfg.get("n_glycan_sites", 0) == 0 and "APBA" in monomers:
        notes.append(
            f"WARNING: {target} is non-glycosylated. APBA may not provide "
            "additional selectivity."
        )
    # Teixeira 2021: dual-epitope for glycosylated targets
    from .config import DUAL_EPITOPE_CD63, GLYCAN_EPITOPE
    if target == "CD63" and DUAL_EPITOPE_CD63:
        notes.append(
            f"DUAL-EPITOPE STRATEGY (Teixeira 2021): CD63 has 3 N-glycan "
            f"sites. Consider dual imprinting with peptide epitope + "
            f"{GLYCAN_EPITOPE} (glycan layer) for maximum CD63 selectivity "
            "over non-glycosylated CD81."
        )

    recipe = {
        "target": target,
        "target_description": target_cfg.get("description", ""),
        "pc_id": pc_data.get("pc_id", ""),
        "polymerization_type": poly_type,
        "monomers": {
            m: {
                "full_name": monomer_lib.get(m, {}).get("name", m),
                "smiles": monomer_lib.get(m, {}).get("smiles", ""),
                "role": "cross-linker" if m in CROSSLINKERS else "functional",
                "molar_ratio": ratios.get(m, 1.0),
                "interaction": interactions.get(m, ""),
            }
            for m in monomers
        },
        "mmsd_sum": pc_data.get("mmsd_sum"),
        "mmpbsa_dg": pc_data.get("mmpbsa", {}).get(
            "delta_total_kcal") if "mmpbsa" in pc_data else None,
        "protocol": _get_protocol(poly_type, target),
        "characterization": _get_characterization_plan(),
        "notes": notes,
    }

    return recipe


def _get_protocol(poly_type: str, target: str) -> dict:
    """Generate synthesis protocol steps."""
    from .config import EPITOPE_MONOMER_MOLAR_RATIO

    if poly_type == "sol-gel":
        return {
            "steps": [
                "1. Prepare SiO₂ nanoparticles (200-600 nm) in ethanol",
                "2. Functionalize NP surface with APTES (amino-coating)",
                "3. Activate with glutaraldehyde (cross-linker for peptide)",
                f"4. Immobilize {target} epitope peptide "
                "(C-terminal Cys + maleimide coupling)",
                f"5. Prepare monomer solution (epitope:monomer = 1:{EPITOPE_MONOMER_MOLAR_RATIO}, "
                "equal molar functional + 10x TEOS) in PBS pH 7.4",
                "6. Add monomer solution to template-NPs",
                "7. Sol-gel polymerization: RT, 16h, gentle stirring",
                "8. Template removal: 10% AcOH + 10% SDS (3× washes)",
                "9. DI water wash (5×)",
                "10. Characterize by FTIR, SEM, zeta potential",
            ],
            "conditions": {
                "temperature": "RT (25°C)",
                "time": "16 hours",
                "solvent": "PBS buffer, pH 7.4",
                "stirring": "Gentle orbital shaking",
                "epitope_monomer_ratio": f"1:{EPITOPE_MONOMER_MOLAR_RATIO}",
            },
        }
    elif poly_type == "solid-phase":
        # Sehit/Altintas 2024: solid-phase synthesis on glass beads
        return {
            "steps": [
                "1. Boil 60g glass beads in 2M NaOH (15 min) — surface activation",
                "2. Silanize with 2% APTES in dry toluene (overnight)",
                "3. Functionalize with 7% glutaraldehyde in PBS (2.5h)",
                f"4. Immobilize {target} epitope (10mg in 5mL MeOH → 40mL PBS)",
                f"5. Pre-complexation: epitope + monomers (1:{EPITOPE_MONOMER_MOLAR_RATIO}) in PBS, 1h",
                "6. Add cross-linker + initiator (APS/TEMED)",
                "7. Polymerize at RT under N₂ (overnight)",
                "8. Collect high-affinity nanoMIPs by elution at 60°C",
                "9. Wash: cold water (remove low-affinity), hot water (collect nanoMIPs)",
                "10. Characterize: DLS, FTIR, zeta potential, HRTEM",
            ],
            "conditions": {
                "temperature": "RT polymerization, 60°C elution",
                "time": "Overnight polymerization",
                "solvent": "PBS buffer, pH 7.4",
                "glass_beads": "60g, 70-100 µm",
                "epitope_monomer_ratio": f"1:{EPITOPE_MONOMER_MOLAR_RATIO}",
            },
        }
    else:  # free-radical
        return {
            "steps": [
                f"1. Dissolve {target} epitope peptide in DMSO/water",
                f"2. Add functional monomers (1:{EPITOPE_MONOMER_MOLAR_RATIO}, "
                "pre-complexation 30 min)",
                "3. Add cross-linker (EGDMA or MBAAm, 10% w/w)",
                "4. Add initiator (APS + TEMED for RT, AIBN for 60°C)",
                "5. Polymerization under N₂: 6-12h",
                "6. Template removal: methanol/acetic acid (9:1, v/v)",
                "7. Soxhlet extraction (24h) to remove residual template",
                "8. Dry under vacuum",
            ],
            "conditions": {
                "temperature": "RT (APS) or 60°C (AIBN)",
                "time": "6-12 hours",
                "solvent": "DMSO/water (4:1)",
                "initiator": "APS/TEMED or AIBN",
                "epitope_monomer_ratio": f"1:{EPITOPE_MONOMER_MOLAR_RATIO}",
            },
        }


def _get_characterization_plan() -> list:
    """
    Comprehensive characterization and validation plan.

    Kowalczyk 2023: SPR + QCM-D for CD9/CD63/CD81 quantification
    Sullivan 2019: CD spectroscopy for 2° structure verification
    Teixeira 2021: KD and IF as primary performance metrics
    """
    from .config import (TARGET_KD_NM, TARGET_IF_MIN,
                         VALIDATION_SPR, VALIDATION_QCM_D,
                         VALIDATION_CD_SPECTROSCOPY)

    plan = [
        "── Material Characterization ──",
        "FTIR spectroscopy — confirm monomer incorporation and functional groups",
        "SEM — morphology and surface roughness (MIP vs NIP comparison)",
        "Zeta potential — surface charge (each PC should differ, per Rajpal 2024)",
        "DLS — hydrodynamic diameter of nanoMIPs (target: 50-200 nm)",
    ]

    if VALIDATION_CD_SPECTROSCOPY:
        plan.append(
            "── Secondary Structure Verification (Sullivan 2019) ──"
        )
        plan.append(
            "CD spectroscopy — compare epitope 2° structure with/without monomers. "
            "Monomers that cause >10% α-helix loss should be avoided. "
            "Use 1:1081 protein:monomer ratio (prepolymerization conditions)."
        )

    plan.extend([
        "── Binding Performance ──",
        "Fluorescence binding assay — FITC-labeled epitope (Rajpal 2024: 96-well plate)",
        f"Target: IF > {TARGET_IF_MIN} (MIP binding / NIP binding)",
        f"Target: KD < {TARGET_KD_NM} nM",
    ])

    if VALIDATION_SPR:
        plan.extend([
            "── SPR Binding Kinetics (Kowalczyk 2023) ──",
            "SPR — immobilize MIP on sensor chip, flow epitope/protein solutions",
            "Fit with two-state reaction model: Ab+Ag ↔ [Ab·Ag] ↔ [Ab·Ag]*",
            "Determine ka1, kd1, ka2, kd2, total KD",
            "Compare: CD9 MIP vs CD63 MIP vs CD81 MIP on same SPR chip",
        ])

    if VALIDATION_QCM_D:
        plan.extend([
            "── QCM-D Validation (Kowalczyk 2023) ──",
            "QCM-D — Au crystal + cysteamine + MIP conjugation",
            "Monitor ΔF (frequency) and ΔD (dissipation) vs EV concentration",
            "Calibration range: 6.1×10⁴ to 6.1×10⁷ particles/mL",
            "ΔD vs ΔF plot — assess layer rigidity and binding mechanism",
        ])

    plan.extend([
        "── Cross-Reactivity (Critical) ──",
        "Test each MIP against all 3 epitopes (CD63, CD81, CD9)",
        "Test against non-specific proteins: HSA, BSA, lysozyme",
        "Calculate selectivity coefficient: BC(target)/BC(non-target)",
        "Kowalczyk 2023 reference: EV affinity order CD9 > CD63 > CD81",
        "── Virus-Like Particle / EV Binding ──",
        "Test with EV isolate from cell culture (ultracentrifugation)",
        "Confirm epitope-MIP recognizes full-length tetraspanin on EV surface",
    ])

    return plan


def _log_recipe(target: str, recipe: dict):
    """Log recipe summary."""
    logger.info(f"[{target}] Best PC: {recipe['pc_id']}")
    logger.info(f"  Type: {recipe['polymerization_type']}")
    for m_name, m_info in recipe["monomers"].items():
        logger.info(f"  {m_name} ({m_info['role']}): "
                    f"ratio={m_info['molar_ratio']}, "
                    f"{m_info['interaction']}")
    if recipe.get("mmsd_sum"):
        logger.info(f"  MMSD sum: {recipe['mmsd_sum']:.2f} kcal/mol")
    for note in recipe.get("notes", []):
        logger.info(f"  NOTE: {note}")


def _write_protocol(recipes: dict, output_path: Path):
    """Write human-readable synthesis protocol."""
    lines = [
        "=" * 70,
        "EPITOPE-IMPRINTED MIP SYNTHESIS PROTOCOL",
        "Computational Screening Results",
        "=" * 70,
        "",
    ]

    for target, recipe in recipes.items():
        lines.append(f"\n{'─'*50}")
        lines.append(f"TARGET: {target}")
        lines.append(f"Description: {recipe.get('target_description', '')}")
        lines.append(f"Polymerization: {recipe['polymerization_type']}")
        lines.append(f"PC ID: {recipe['pc_id']}")
        lines.append("")
        lines.append("MONOMERS:")
        for m_name, m_info in recipe["monomers"].items():
            lines.append(
                f"  {m_name:<10} {m_info['full_name']:<40} "
                f"ratio={m_info['molar_ratio']:<5} "
                f"[{m_info['role']}]"
            )
        lines.append("")
        lines.append("PROTOCOL:")
        for step in recipe.get("protocol", {}).get("steps", []):
            lines.append(f"  {step}")
        lines.append("")
        if recipe.get("notes"):
            lines.append("NOTES:")
            for note in recipe["notes"]:
                lines.append(f"  • {note}")

    lines.append(f"\n{'='*70}")
    lines.append("CHARACTERIZATION PLAN:")
    for item in _get_characterization_plan():
        lines.append(f"  • {item}")

    Path(output_path).write_text("\n".join(lines) + "\n")
