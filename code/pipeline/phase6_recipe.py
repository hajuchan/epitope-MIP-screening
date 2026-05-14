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
               phase5_results: dict = None,
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
        output_dir = str(get_output_path("phase6"))
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

    if phase5_results is None:
        p5_path = get_output_path("phase5") / "phase5_rebinding_results.json"
        if p5_path.exists():
            with open(p5_path) as f:
                phase5_results = json.load(f)
        else:
            phase5_results = {}

    if target_names is None:
        target_names = list(phase3_results.keys())

    recipes = {}

    for target in target_names:
        p3 = phase3_results.get(target, {})
        p4 = phase4_results.get(target, {})

        if "error" in p3:
            continue

        # Check Phase 5 rebinding results (if available)
        rebinding = {}
        if phase5_results and target in phase5_results:
            rebinding = phase5_results[target]
            n_rebound = rebinding.get("n_rebound", 0)
            n_snap = rebinding.get("n_snapshots", 0)
            if n_snap > 0 and n_rebound == 0:
                logger.warning(f"[{target}] Rebinding FAILED (0/{n_snap}) — skipping recipe")
                continue
            logger.info(f"[{target}] Rebinding: {rebinding.get('success_rate', 'N/A')}")

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

        # Get Phase 4 optimal ratio if available
        p4_ratio = {}
        best_pc_id = best_pc.get("pc_id", "")
        if best_pc_id in p4:
            p4_ratio = p4[best_pc_id].get("optimal_ratio", {})

        # Generate recipe
        recipe = _generate_recipe(
            target, monomers, best_pc, poly_type,
            ALL_MONOMERS, TARGETS[target],
            md_ratio=p4_ratio,
        )
        recipes[target] = recipe

        # A7: NIP (Non-Imprinted Polymer) control recipe — same as MIP minus template
        from .config import PHASE6_GENERATE_NIP
        if PHASE6_GENERATE_NIP:
            nip_recipe = _generate_nip_recipe(recipe)
            recipes[target + "_NIP"] = nip_recipe

        # Log recipe
        _log_recipe(target, recipe)

    # Save recipes (one combined JSON)
    with open(output_dir / "phase6_recipes.json", "w") as f:
        json.dump(recipes, f, indent=2)

    # A7: Per-target recipe file (MIP + NIP in SAME file per CD)
    for target in target_names:
        if target not in recipes:
            continue
        target_recipes = {target: recipes[target]}
        if PHASE6_GENERATE_NIP and (target + "_NIP") in recipes:
            target_recipes[target + "_NIP"] = recipes[target + "_NIP"]
        per_target_path = output_dir / f"{target}_recipe.txt"
        _write_protocol(target_recipes, per_target_path)
        logger.info(f"  Per-target recipe ({target} MIP + NIP) → {per_target_path}")

    # Combined protocol (all targets)
    protocol_path = output_dir / "synthesis_protocol.txt"
    _write_protocol(recipes, protocol_path)
    logger.info(f"\nCombined protocol → {protocol_path}")

    return recipes


def _generate_nip_recipe(mip_recipe: dict) -> dict:
    """A7: Generate NIP (Non-Imprinted Polymer) control from MIP recipe.

    NIP is identical synthesis BUT WITHOUT template addition.
    Used as denominator for IF (Imprinting Factor = MIP/NIP).

    Reference: Vasapollo 2011 (Int J Mol Sci); Mosbach 1994 (Trends Biotechnol).
    """
    import copy
    nip = copy.deepcopy(mip_recipe)
    nip["target"] = nip.get("target", "Unknown") + "_NIP"
    nip["target_description"] = (
        (nip.get("target_description") or "") +
        " — NIP control (no template)"
    )
    nip["template"] = None
    nip["is_nip"] = True

    # Filter out template-related steps from protocol
    if "protocol" in nip and "steps" in nip["protocol"]:
        filtered_steps = []
        for step in nip["protocol"]["steps"]:
            s_lower = step.lower()
            # Skip steps that explicitly add/remove template
            if any(kw in s_lower for kw in [
                "epitope", "template", "glycopeptide", "boronate-reversal",
                "boronate reversal", "pba competition", "phenylboronic acid",
                "fructose competition",
            ]):
                continue
            filtered_steps.append(step)
        nip["protocol"]["steps"] = filtered_steps

    # Update notes
    nip_note = (
        "NIP CONTROL — Non-Imprinted Polymer:\n"
        "  • Identical synthesis to MIP BUT WITHOUT template addition\n"
        "  • Glutaraldehyde activation + NaBH3CN reduction still performed\n"
        "    (gives identical surface chemistry to MIP)\n"
        "  • Use as denominator for Imprinting Factor: IF = Binding_MIP / Binding_NIP\n"
        "  • Target IF > 3 (good MIP); IF > 5-10 (excellent)\n"
        "  • Vasapollo 2011 / Mosbach 1994 — standard MIP validation"
    )
    if "notes" not in nip or not isinstance(nip.get("notes"), list):
        nip["notes"] = []
    nip["notes"].insert(0, nip_note)
    return nip


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
                      monomer_lib: dict, target_cfg: dict,
                      md_ratio: dict = None) -> dict:
    """Generate a complete synthesis recipe.
    Uses Phase 4 MD optimal ratio if available, else default equal molar."""
    from .config import CROSSLINKERS

    # Categorize monomers
    functional = [m for m in monomers if m not in CROSSLINKERS]
    crosslinker = [m for m in monomers if m in CROSSLINKERS]

    # Molar ratios: from Phase 4 MD contact analysis, or default
    ratios = {}
    if md_ratio:
        # Use Phase 4 optimal ratio (inverse contact frequency)
        for m in functional:
            ratios[m] = float(md_ratio.get(m, 1))
        for m in crosslinker:
            ratios[m] = float(md_ratio.get(m, sum(ratios.values()) or 10))
        logger.info(f"  Using Phase 4 MD ratio: {ratios}")
    else:
        # Default: equal molar functional, 10x crosslinker
        for m in functional:
            ratios[m] = 1.0
        for m in crosslinker:
            ratios[m] = 10.0

    # Interaction profile
    interactions = {}
    for m in monomers:
        info = monomer_lib.get(m, {})
        interactions[m] = info.get("interaction", "unknown")

    # Special notes
    notes = []

    # ── Polymerization compatibility check (Liu 2017 Nat. Protoc.) ──
    from .config import (is_polymerization_compatible, synthesis_pH_window,
                         reactivity_ratio_product, QE_PARAMS)
    compatible, dominant, conflicts = is_polymerization_compatible(monomers)
    if not compatible:
        notes.append(
            "❌ POLYMERIZATION COMPATIBILITY ISSUE: " + " | ".join(conflicts) +
            " Choose ONE polymerization chemistry path per pot: "
            "(a) sol-gel for silane monomers (Liu 2017 boronate-anchored "
            "oriented imprinting); (b) free-radical for vinyl monomers."
        )

    # ── pH stability check (intersection of monomer pH ranges) ──
    pH_min, pH_max, pH_rec = synthesis_pH_window(monomers)
    if pH_min is not None:
        if pH_rec is None:
            notes.append(
                f"❌ NO COMMON pH RANGE: monomers' stability windows do not "
                f"overlap (max-min = {pH_min:.1f}–{pH_max:.1f}). "
                "Some monomers must be removed or substituted."
            )
        else:
            notes.append(
                f"✓ Recommended synthesis pH: {pH_rec:.1f} "
                f"(common stability range: {pH_min:.1f}–{pH_max:.1f})"
            )

    # ── Vinyl reactivity ratio check (Q-e scheme, radical only) ──
    vinyl_ms = [m for m in monomers if m in QE_PARAMS]
    if dominant == "radical" and len(vinyl_ms) >= 2:
        # Pairwise r1·r2 product
        bad_pairs = []
        ideal_pairs = []
        for i, m1 in enumerate(vinyl_ms):
            for m2 in vinyl_ms[i + 1:]:
                rr = reactivity_ratio_product(m1, m2)
                if rr is None: continue
                if rr < 0.05 or rr > 20:
                    bad_pairs.append((m1, m2, rr))
                elif 0.1 < rr < 10:
                    ideal_pairs.append((m1, m2, rr))
        if bad_pairs:
            pair_str = ", ".join(f"{a}/{b} (r1·r2={r:.2f})" for a, b, r in bad_pairs)
            notes.append(
                f"⚠ REACTIVITY MISMATCH (Q-e scheme): {pair_str}. "
                f"r1·r2 outside [0.1, 10] → block-like or homopolymer instead of "
                f"random copolymer (Alfrey-Price 1947). Consider monomer substitution."
            )
        if ideal_pairs:
            pair_str = ", ".join(f"{a}/{b} (r1·r2={r:.2f})" for a, b, r in ideal_pairs)
            notes.append(
                f"✓ Vinyl co-polymerization OK for: {pair_str}"
            )

    # ── Serum cross-reactivity warning (Tier 5 critical) ──
    has_boronate = any(m in {"APBA", "VPBA", "FPBA", "AAPBA"} for m in monomers)
    if has_boronate:
        notes.append(
            "⚠ SERUM CROSS-REACTIVITY (boronate MIPs): Human serum contains "
            "highly sialylated proteins that compete with target binding. "
            "Major interferents (per liter): transferrin (3 g, 4 sialic), "
            "α1-acid glycoprotein (1 g, 14 sialic!), haptoglobin (1-3 g, 5-7 "
            "sialic), fetuin (<0.1 g, 8 sialic). Mandatory pretreatment: "
            "(a) ultracentrifugation (100,000 g) for EV isolation BEFORE MIP "
            "capture; OR (b) pre-clearance with anti-transferrin/fetuin "
            "antibody beads; OR (c) size-exclusion chromatography. "
            "Direct serum capture without pretreatment will give false positives."
        )

    # ── EV-specific consideration ──
    if target.startswith("CD") and target in ("CD63", "CD81", "CD9"):
        notes.append(
            f"EV CONTEXT: {target} is a transmembrane tetraspanin on EV surface "
            "(~100 nm vesicle). MIP cavity designed from soluble head epitope "
            "may not match steric access to native membrane-anchored protein. "
            "Validate with intact EVs (not just recombinant protein) via "
            "QCM-D and TIRF microscopy."
        )

    # ── N-glycan heterogeneity caveat (for dual-imprinting targets) ──
    if has_boronate and target_cfg.get("n_glycan_sites", 0) > 0:
        notes.append(
            "GLYCAN HETEROGENEITY: N-glycans on the same site can be "
            "high-mannose (no sialic acid → no boronate binding), complex "
            "(with sialic acid → binds), or hybrid. Only ~50-70% of CD63 "
            "N-glycans are sialylated complex-type. Cancer-derived EVs often "
            "have altered sialylation (Cancer Cell 2020). Calibrate MIP "
            "against the specific EV source intended for diagnostic use."
        )

    # ── Initiator suggestion ──
    if dominant == "silane":
        notes.append(
            "Synthesis: sol-gel polycondensation. Catalyst: 1 M HCl (acid-catalyzed) "
            "or 1 M NH3 (base-catalyzed). Polymerization at RT, 16-24 h, no initiator needed."
        )
    elif dominant == "radical":
        # B10: Compute initiator mass from monomer mol total
        from .config import INITIATOR_MOL_PERCENT
        from .utils_analysis import calculate_initiator_mass
        # Estimate total monomer mol (~5 mmol for 100 mg NP scale, default)
        total_mol = 0.005  # 5 mmol for standard 100 mg NP synthesis
        if any(m in {"VPBA", "AAPBA"} for m in monomers):
            mass_info = calculate_initiator_mass(
                total_mol, "Irgacure_2959", INITIATOR_MOL_PERCENT)
            notes.append(
                f"Synthesis: free-radical polymerization. Initiator: "
                f"{mass_info['initiator_mass_mg']} mg Irgacure 2959 "
                f"({INITIATOR_MOL_PERCENT} mol% of {total_mol*1000:.1f} mmol "
                f"total monomer; Odian 2004 standard 0.5-2 mol%). "
                "UV 365 nm, RT — RT required for boronate stability. "
                "Solvent: acetonitrile/PBS (3:1). N2 atmosphere."
            )
        else:
            mass_aibn = calculate_initiator_mass(
                total_mol, "AIBN", INITIATOR_MOL_PERCENT)
            mass_irg = calculate_initiator_mass(
                total_mol, "Irgacure_2959", INITIATOR_MOL_PERCENT)
            notes.append(
                f"Synthesis: free-radical polymerization. Initiator options: "
                f"({mass_aibn['initiator_mass_mg']} mg AIBN, thermal 60°C 16h) "
                f"or ({mass_irg['initiator_mass_mg']} mg Irgacure 2959, "
                f"UV 365 nm RT). Both at {INITIATOR_MOL_PERCENT} mol% of "
                f"{total_mol*1000:.1f} mmol total monomer. "
                "Solvent: acetonitrile (porogen). N2 atmosphere."
            )

    # APBA polymerizability warning (APBA itself is NOT polymerizable)
    if "APBA" in monomers:
        notes.append(
            "⚠️ APBA NOTE: 3-Aminophenylboronic acid has no vinyl group and "
            "cannot undergo free-radical co-polymerization. Use FPBA (surface "
            "grafted via Schiff base) for sol-gel pathways, or VPBA "
            "(4-vinylphenylboronic acid) / AAPBA for radical pathways. "
            "Phase 3 scored APBA on binding affinity only — chemistry "
            "manually validated downstream."
        )

    # Glycan layer rationale
    n_gly = target_cfg.get("n_glycan_sites", 0)
    if n_gly > 0:
        notes.append(
            f"Glycan recognition layer: {target} has {n_gly} N-glycan site(s). "
            f"For optimal recognition, use FPBA-grafted NP support + glycopeptide "
            f"template at pH 8.5 (boronate-diol cyclic ester optimum). "
            f"GFN2-xTB L1 prediction: {n_gly}-valent ΔΔG ≈ "
            f"-{3.29*n_gly:.1f} kcal/mol vs glycan-free target."
        )
    if n_gly == 0 and ("APBA" in monomers or "VPBA" in monomers or "FPBA" in monomers):
        notes.append(
            f"WARNING: {target} has 0 N-glycan sites → boronate-glycan "
            "recognition not applicable. Boronate monomer acts only as "
            "π-stacking + electrostatic functional monomer."
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
