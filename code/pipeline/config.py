"""
Epitope-MIP Screening Pipeline Configuration
=============================================
Computational screening of functional monomers for epitope-imprinted
MIPs targeting exosome tetraspanin ECL2 (CD63 / CD81 / CD9).

Reference:
  Rajpal et al., Sci. Rep. 2024 — MMSD methodology
  Sehit/Altintas, ACS Sensors 2024 — epitope MD stability + monomer contact MD
  Sullivan et al., J. Phys. Chem. B 2019 — SiteMap + MM-GBSA + comonomer design
  Teixeira et al., Science Advances 2021 — epitope selection criteria
  Kowalczyk et al., Anal. Chem. 2023 — SPR/QCM-D tetraspanin validation
"""
from pathlib import Path as _Path

# ── Targets (Tetraspanin ECL2 epitopes) ─────────────────────────
TARGETS = {
    "CD63": {
        "uniprot_id": "P08962",
        "source": "alphafold",
        "chain": "A",
        "full_length": 238,
        "ecl2_range": (103, 203),
        # A1: head_candidates — multiple epitope regions to evaluate
        "head_residues": (155, 170),  # Default canonical 16-mer (back-compat)
        "head_candidates": [
            {"range": (155, 170), "name": "head_canonical",      "glycan_in": []},
            {"range": (157, 172), "name": "head_glyco_N172",     "glycan_in": [172]},
            {"range": (143, 158), "name": "head_glyco_N150",     "glycan_in": [150]},
            {"range": (123, 138), "name": "head_glyco_N130",     "glycan_in": [130]},
        ],
        "n_glycan_sites": 3,
        "n_glycan_positions": [130, 150, 172],
        "ccg_position": 145,
        "disulfide_cys": [145, 146, 169, 170, 177, 191],
        "description": "CD63 LEL — heavy glycosylation, largest head",
    },
    "CD81": {
        "pdb_id": "5TCX",
        "source": "pdb",
        "chain": "A",
        "full_length": 236,
        "ecl2_range": (113, 201),
        "head_residues": (168, 183),
        "head_candidates": [
            {"range": (168, 183), "name": "head_canonical", "glycan_in": []},
        ],
        "n_glycan_sites": 0,
        "n_glycan_positions": [],
        "ccg_position": 156,
        "disulfide_cys": [156, 157, 175, 190],
        "description": "CD81 LEL — non-glycosylated, hydrophobic head",
    },
    "CD9": {
        "pdb_id": "6K4J",
        "source": "pdb",
        "chain": "A",
        "full_length": 228,
        "ecl2_range": (112, 195),
        "head_residues": (156, 171),
        "head_candidates": [
            {"range": (156, 171), "name": "head_canonical", "glycan_in": []},
            # CD9 single N-glycan position is uncertain; if known, add here
        ],
        "n_glycan_sites": 1,
        "n_glycan_positions": [],
        "ccg_position": 152,
        "disulfide_cys": [152, 153, 167, 181],
        "description": "CD9 LEL — light glycosylation, smallest head",
    },
}

# A1/B1/B2: Epitope quality (DEFAULT ON)
PHASE1_EVALUATE_MULTI_EPITOPE = True   # A1: rank head_candidates
PHASE1_COMPUTE_SASA = True             # B1: SASA per residue
PHASE1_COMPUTE_GRAVY = True            # B2: hydrophobicity balance
EPITOPE_SASA_MIN_A2 = 50.0
EPITOPE_GRAVY_MIN = -2.0
EPITOPE_GRAVY_MAX = +2.0
# REMOVED (REVIEW FINDING 18): EPITOPE_RMSD_MAX = 3.0 lived here as a dead
# exact duplicate of EPITOPE_RMSD_THRESHOLD (also 3.0, defined below and
# ACTUALLY enforced in phase1_epitope_prep.py:831 and verify_phase.py:363).
# Two names for one concept means an editor can tighten the inert one and
# believe the pipeline got stricter. Use EPITOPE_RMSD_THRESHOLD.

# A2: K-medoids clustering for conformer extraction (DEFAULT ON)
ENSEMBLE_CLUSTERING_METHOD = "kmedoids"  # "uniform" (legacy) or "kmedoids"

# A3: Multi-pose docking + clustering (DEFAULT ON)
PHASE2_POSE_CLUSTERING = True
POSE_CLUSTERING_RMSD_A = 2.0
POSE_CLUSTERING_MIN_SIZE = 3
POSE_CLUSTERING_MAX_CLUSTERS = 5

# A4/C2: MMSD optimizer
#   "nsga2"    — NSGA-II 3-objective Pareto front (C2, DEFAULT)
#               Affinity + Selectivity + Synthesizability simultaneously
#   "bayesian" — GP-based Bayesian Optimization (A4: single-obj, 20x fewer evals)
#   "greedy"   — Forward selection + swap (legacy, deterministic)
MMSD_OPTIMIZER = "nsga2"          # DEFAULT — most informative multi-objective
BAYESIAN_N_CALLS = 30
BAYESIAN_ACQUISITION = "EI"
NSGA2_POP_SIZE = 20              # NSGA-II population size
NSGA2_N_GEN = 15                 # NSGA-II generations (× pop = ~300 evals; ~80 unique with cache)
NSGA2_OBJECTIVES = ["affinity", "selectivity", "synthesizability"]
# Auto-fallback chain if pymoo missing → BO if skopt missing → greedy
MMSD_OPTIMIZER_FALLBACK = ["nsga2", "bayesian", "greedy"]

# Selectivity-aware MMSD (Garcia-Ortegon 2022 DOCKSTRING JCIM)
# Cross-target docking ΔΔG penalty: re-docks same combo on off-target receptors
# and penalizes if own-target advantage < DDG_threshold (i.e., not selective enough)
MMSD_SELECTIVITY_AWARE = True       # If True, run cross-MMSD per evaluation (3x cost)
SELECTIVITY_WEIGHT = 0.5            # Penalty coefficient (Garcia-Ortegon 2022 ≈ affinity)
SELECTIVITY_DDG_THRESHOLD = -1.0    # kcal/mol — own must be ≤ other-mean by this much

# ── A3 · Membrane calibration factor (2026-08) ────────────────
# Reported literature comparison of isolated-ECL2 MD vs full-length in-membrane
# MD (Zimmerman 2016 CD81; Umeda 2020 CD9; TEM cluster effect) predicts a 20-
# 30% down-adjust in selectivity metrics when the cavity is challenged with
# actual membrane-embedded target. Verify_phase5 displays BOTH raw and
# calibrated PCSI so the reader sees the honest expected experimental value.
# Set to 1.0 to disable (raw only).
SELECTIVITY_MEMBRANE_CALIBRATION = 0.75

# A5/B7: Polymerization solvent variation (DEFAULT ON)
PHASE4_SOLVENT_SWEEP = True     # If True, run Phase 4 in multiple solvents
PHASE4_SOLVENTS = {              # GROMACS template names + dielectric
    "water":           {"template": "spc216.gro", "dielectric": 78.5,  "use_for": ["sol-gel", "radical"]},
    "ethanol_water_3_1": {"template": "ethanol_water.gro", "dielectric": 49.0, "use_for": ["sol-gel"]},
    "methanol":        {"template": "methanol.gro", "dielectric": 32.6, "use_for": ["sol-gel"]},
    "acetonitrile":    {"template": "acn.gro", "dielectric": 36.6, "use_for": ["radical"]},
}

# A6: Bootstrap CI
BOOTSTRAP_N_RESAMPLES = 10000    # Number of bootstrap iterations
BOOTSTRAP_CI = 0.95              # Confidence level

# A7: NIP recipe auto-generation
PHASE6_GENERATE_NIP = True       # Always generate NIP recipe alongside MIP

# B3: Decoy monomer evaluation (DEFAULT ON)
PHASE2_DECOY_BASELINE = True
DECOY_MONOMERS = {
    "acetate":  {"smiles": "CC(=O)O", "name": "Acetate",
                  "polymerization": "none", "interaction": "small anion (decoy)"},
    "methanol_dec": {"smiles": "CO", "name": "Methanol (decoy)",
                  "polymerization": "none", "interaction": "small polar (decoy)"},
    "ethane":   {"smiles": "CC", "name": "Ethane",
                  "polymerization": "none", "interaction": "small nonpolar (decoy)"},
    "phenol_dec":   {"smiles": "c1ccc(O)cc1", "name": "Phenol (decoy)",
                  "polymerization": "none", "interaction": "aromatic (decoy)"},
    "dimethyl_ether": {"smiles": "COC", "name": "Dimethyl ether",
                  "polymerization": "none", "interaction": "ether (decoy)"},
}

# B5: DFT validation
DFT_VALIDATION_TOP_N = 3         # Top-N PCs to DFT refine
DFT_LEVEL = "M06-2X/def2-TZVP"   # DFT functional/basis
DFT_SOLVENT = "water"            # implicit solvent

# B6: Variable monomer ratio sweep (DISABLED — redundant with EBN-based optimal_ratio)
# Rationale: optimal_ratio is computed from the main production MD via EBN
# (Yuan 2024 standard). The B6 sweep ran 5 preset ratios × 30 ns separately and
# was not consumed by Phase 6 recipe — pure overhead, removed.
PHASE4_RATIO_SWEEP = False
PHASE4_RATIO_PRESETS = [
    (1, 1, 1, 1),
    (2, 1, 1, 1),
    (3, 1, 1, 1),
    (1, 2, 1, 1),
    (1, 1, 2, 1),
]

# B8: Multi-pose rebinding (DEFAULT ON)
REBINDING_MULTI_POSE = True     # If True, multiple head conformers × replicates
REBINDING_N_HEAD_CONFORMERS = 5

# B9: FEP framework (stubs — full setup requires manual GROMACS-FEP config)
FEP_ENABLED = False
FEP_LAMBDA_WINDOWS = 21
FEP_NS_PER_WINDOW = 5

# B10: Initiator mole percent
INITIATOR_MOL_PERCENT = 1.0      # 0.5-2 mol% typical (Odian 2004)
INITIATOR_MW = {                 # for mass calculation
    "Irgacure_2959": 224.25,
    "AIBN": 164.21,
    "KPS": 270.32,
    "BPO": 242.23,
}

# ── Monomer Libraries ───────────────────────────────────────────
# Polymerization type taxonomy:
#   "silane"  : Si(OR)x — sol-gel polycondensation (Si–O–Si network)
#   "vinyl"   : C=C — free-radical chain polymerization
#   "catechol": polyhydroxyphenyl — oxidative auto-polymerization at pH > 7
#   "surface" : NOT polymerizable; surface-grafted via amine/aldehyde coupling
#   "epoxy"   : epoxide ring — covalent reaction with nucleophiles (Lys, Cys)
# CRITICAL: Different polymerization types CANNOT be mixed in one pot.

# A. Silane monomers (Sol-gel epitope imprinting, per Rajpal 2024)
SILANE_MONOMERS = {
    "PTES":   {"smiles": "CCO[Si](OCC)(OCC)c1ccccc1",
               "name": "Phenyltriethoxysilane",
               "polymerization": "silane",
               "interaction": "π-π stacking, hydrophobic"},
    # NOTE: alkyl-substituted silanes must bond the linker carbon DIRECTLY to Si
    # (Si-C), not through an ether oxygen. Writing "...CO[Si]" makes a
    # tetraalkoxysilane whose organic group hydrolyses off — the functional
    # group would never stay tethered to the silica network.
    "APTES":  {"smiles": "NCCC[Si](OCC)(OCC)OCC",
               "name": "(3-Aminopropyl)triethoxysilane",
               "polymerization": "silane",
               "interaction": "H-bond donor, electrostatic"},
    "APTMS":  {"smiles": "NCCC[Si](OC)(OC)OC",
               "name": "(3-Aminopropyl)trimethoxysilane",
               "polymerization": "silane",
               "interaction": "H-bond, electrostatic"},
    "UPTMS":  {"smiles": "O=C(N)NCCC[Si](OC)(OC)OC",
               "name": "3-Ureidopropyltrimethoxysilane",
               "polymerization": "silane",
               "interaction": "Multi H-bond (urea D+A)"},
    "MPTMS":  {"smiles": "SCCC[Si](OC)(OC)OC",
               "name": "(3-Mercaptopropyl)trimethoxysilane",
               "polymerization": "silane",
               "interaction": "Thiol-Cys, H-bond"},
    "IBTES":  {"smiles": "CC(C)C[Si](OCC)(OCC)OCC",
               "name": "Isobutyltriethoxysilane",
               "polymerization": "silane",
               "interaction": "Hydrophobic, alkyl"},
    "MTMS":   {"smiles": "C[Si](OC)(OC)OC",
               "name": "Methyltrimethoxysilane",
               "polymerization": "silane",
               "interaction": "Hydrophobic"},
    "TEOS":   {"smiles": "CCO[Si](OCC)(OCC)OCC",
               "name": "Tetraethyl orthosilicate",
               "polymerization": "silane",
               "interaction": "Cross-linker"},
    "EDTMS":  {"smiles": "NCCNCCC[Si](OC)(OC)OC",
               "name": "N-[3-(Trimethoxysilyl)propyl]ethylenediamine",
               "polymerization": "silane",
               "interaction": "Chelate-type H-bond"},
    "ICTES":  {"smiles": "O=C=NCCC[Si](OCC)(OCC)OCC",
               "name": "3-(Triethoxysilyl)propyl isocyanate",
               "polymerization": "silane",
               "interaction": "Covalent (Lys ε-NH₂)"},
    "VTMS":   {"smiles": "C=C[Si](OC)(OC)OC",
               "name": "Vinyltrimethoxysilane",
               "polymerization": "silane",  # Si-OR dominant; vinyl can crosslink with radical comonomer
               "interaction": "Hydrophobic + crosslinkable"},
    "GPTMS":  {"smiles": "C1OC1COCCC[Si](OC)(OC)OC",
               "name": "(3-Glycidyloxypropyl)trimethoxysilane",
               "polymerization": "silane",  # silane primary; epoxy is covalent side group
               "interaction": "Epoxy covalent (nucleophile)"},
    "DIDMS":  {"smiles": "C[Si](C)(OC)OC",
               "name": "Dimethyldimethoxysilane",
               "polymerization": "silane",
               "interaction": "Strong hydrophobic"},
    "CETES":  {"smiles": "N#CCCC[Si](OCC)(OCC)OCC",
               "name": "3-Cyanopropyltriethoxysilane",
               "polymerization": "silane",
               "interaction": "Weak H-bond acceptor (CN)"},
    "TTMS":   {"smiles": "Cc1ccc(cc1)[Si](OC)(OC)OC",
               "name": "p-Tolyltrimethoxysilane",
               "polymerization": "silane",
               "interaction": "π-π + mild hydrophobic"},
}

# B. Vinyl/acrylic monomers (Free-radical polymerization)
VINYL_MONOMERS = {
    "AA":     {"smiles": "C=CC(=O)O",
               "name": "Acrylic acid",
               "polymerization": "vinyl",
               "interaction": "Electrostatic (Lys, Arg, His)"},
    "MAA":    {"smiles": "CC(=C)C(=O)O",
               "name": "Methacrylic acid",
               "polymerization": "vinyl",
               "interaction": "Electrostatic + mild hydrophobic"},
    "AAm":    {"smiles": "C=CC(N)=O",
               "name": "Acrylamide",
               "polymerization": "vinyl",
               "interaction": "H-bond (D+A)"},
    "NIPAm":  {"smiles": "CC(C)NC(=O)C=C",
               "name": "N-Isopropylacrylamide",
               "polymerization": "vinyl",
               "interaction": "H-bond + hydrophobic"},
    "4VIm":   {"smiles": "C=Cc1cnc[nH]1",
               "name": "4(5)-Vinylimidazole",
               "polymerization": "vinyl",
               "interaction": "π-π, H-bond, His mimic"},
    "HEMA":   {"smiles": "C=C(C)C(=O)OCCO",
               "name": "2-Hydroxyethyl methacrylate",
               "polymerization": "vinyl",
               "interaction": "H-bond (OH)"},
    "MBAAm":  {"smiles": "C=CC(=O)NCNC(=O)C=C",
               "name": "N,N'-Methylenebisacrylamide",
               "polymerization": "vinyl",
               "interaction": "Cross-linker"},
    "DA":     {"smiles": "NCCc1ccc(O)c(O)c1",
               "name": "Dopamine hydrochloride",
               "polymerization": "catechol",  # auto-oxidation at pH > 7
               "interaction": "Multi H-bond, catechol"},
    "NE":     {"smiles": "NCC(O)c1ccc(O)c(O)c1",
               "name": "Norepinephrine",
               "polymerization": "catechol",
               "interaction": "DA-like + extra OH"},
    "TBAm":   {"smiles": "C=CC(=O)NC(C)(C)C",
               "name": "N-tert-Butylacrylamide",
               "polymerization": "vinyl",
               "interaction": "Hydrophobic"},
    # meta (3-) substitution, per the name. The para isomer here previously made
    # this 4-aminophenylboronic acid — a different molecule with a different
    # boronic-acid pKa and diol-binding geometry. Molecular formula cannot catch
    # this: isomers share it. Verified by ring substitution pattern.
    "APBA":   {"smiles": "Nc1cccc(B(O)O)c1",
               "name": "3-Aminophenylboronic acid",
               "polymerization": "surface",  # NOT polymerizable; surface-grafted via NH2 to aldehyde
               "interaction": "Glycan (diol) recognition; ONLY usable as surface anchor"},
    "VPBA":   {"smiles": "C=Cc1ccc(B(O)O)cc1",
               "name": "4-Vinylphenylboronic acid",
               "polymerization": "vinyl",  # POLYMERIZABLE boronate (literature standard for radical MIP)
               "interaction": "Glycan (diol) recognition + radical polymerization"},
    "FPBA":   {"smiles": "O=Cc1ccc(B(O)O)cc1",
               "name": "4-Formylphenylboronic acid",
               "polymerization": "surface",  # surface anchor via aldehyde-amine Schiff base
               "interaction": "Glycan recognition; Liu 2017 standard for NP grafting"},
    "AAPBA":  {"smiles": "C=CC(=O)Nc1cccc(B(O)O)c1",   # meta, per the name (was para)
               "name": "N-Acryloyl-3-aminophenylboronic acid",
               "polymerization": "vinyl",
               "interaction": "Glycan recognition + radical polymerization (APBA-acrylate)"},
    "EGDMA":  {"smiles": "C=C(C)C(=O)OCCOC(=O)C(=C)C",
               "name": "Ethylene glycol dimethacrylate",
               "polymerization": "vinyl",
               "interaction": "Cross-linker"},
}

# Cross-linkers (auto-selected by monomer chemistry compatibility)
# Note: `type` field doubles as `polymerization` taxonomy
CROSSLINKER_LIBRARY = {
    "TEOS":  {"smiles": "CCO[Si](OCC)(OCC)OCC",
              "name": "Tetraethyl orthosilicate",
              "type": "silane", "polymerization": "silane", "functionality": 4,
              "interaction": "Sol-gel Si-O-Si network, slow hydrolysis (~16h)"},
    "TMOS":  {"smiles": "CO[Si](OC)(OC)OC",
              "name": "Tetramethyl orthosilicate",
              "type": "silane", "polymerization": "silane", "functionality": 4,
              "interaction": "Sol-gel Si-O-Si network, fast hydrolysis (~6x TEOS)"},
    "MBAAm": {"smiles": "C=CC(=O)NCNC(=O)C=C",
              "name": "N,N'-Methylenebisacrylamide",
              "type": "vinyl", "polymerization": "vinyl", "functionality": 2,
              "interaction": "Free-radical, flexible"},
    "EGDMA": {"smiles": "C=C(C)C(=O)OCCOC(=O)C(=C)C",
              "name": "Ethylene glycol dimethacrylate",
              "type": "vinyl", "polymerization": "vinyl", "functionality": 2,
              "interaction": "Free-radical, semi-rigid"},
    "DVB":   {"smiles": "C=Cc1ccc(C=C)cc1",
              "name": "Divinylbenzene",
              "type": "vinyl", "polymerization": "vinyl", "functionality": 2,
              "interaction": "Free-radical, rigid aromatic"},
    # Trimethylolpropane has THREE CH2-O arms; the third was written without its
    # CH2, giving C17H24O6 for a compound that is C18H26O6.
    "TRIM":  {"smiles": "C=C(C)C(=O)OCC(CC)(COC(=O)C(=C)C)COC(=O)C(=C)C",
              "name": "Trimethylolpropane trimethacrylate",
              "type": "vinyl", "polymerization": "vinyl", "functionality": 3,
              "interaction": "Free-radical, tri-functional high crosslink"},
}


# ── Polymerization Compatibility Helper ─────────────────────────
# Define which polymerization types can co-exist in one-pot synthesis.
# CRITICAL: do NOT mix radical (vinyl) and condensation (silane) in single pot.
POLYMERIZATION_COMPATIBILITY = {
    "silane":   {"silane"},                       # sol-gel only
    "vinyl":    {"vinyl", "catechol"},            # radical + catechol (catechol can co-radical-polymerize)
    "catechol": {"vinyl", "catechol"},
    "surface":  {"silane", "vinyl", "catechol"},  # surface-grafted; compatible with any matrix
    "epoxy":    {"silane", "vinyl", "catechol"},  # side-chain covalent; compatible
}


# ── Monomer pH Stability Ranges ─────────────────────────────────
# Each monomer is stable in a pH range; outside this range the monomer
# hydrolyzes (silane, epoxide, NCO) or auto-polymerizes (catechol).
# Synthesis pH must be in the INTERSECTION of all monomers' ranges.
# References:
#   - Silanes: typical sol-gel range pH 2-10 (Brinker & Scherer 1990)
#   - Boronates: pH > 7.5 for diol binding (Lorand & Edwards 1959; APBA pKa 8.8)
#   - Catechol (NE, DA): pH < 7 to avoid auto-oxidation (Lee 2007 Sci Adv)
#   - Acrylates/methacrylates: pH 4-12 (carboxylate pKa ~4.5)
#   - Isocyanate (ICTES): pH < 5 only (reacts with water at neutral pH)
PH_STABILITY = {
    # Silanes — hydrolyze in strong acid/base, stable in moderate pH
    "PTES":   (3.0, 9.5),  "APTES": (3.0, 10.0), "APTMS": (3.0, 10.0),
    "UPTMS":  (3.0, 9.5),  "MPTMS": (3.0, 9.0),  "IBTES": (3.0, 10.0),
    "MTMS":   (3.0, 9.5),  "TEOS":  (3.0, 10.0), "EDTMS": (3.0, 9.5),
    "ICTES":  (1.0, 4.5),  # isocyanate hydrolyzes pH > 5
    "VTMS":   (3.0, 9.5),  "DIDMS": (3.0, 9.5),  "CETES": (3.0, 9.5),
    "TTMS":   (3.0, 9.5),
    "GPTMS":  (5.5, 10.0),  # epoxide ring acid-hydrolyzes pH < 5.5
    # Vinyl monomers — broad stability
    "AA":     (2.0, 12.0), "MAA": (2.0, 12.0), "AAm": (3.0, 12.0),
    "NIPAm":  (3.0, 12.0), "HEMA": (3.0, 11.0), "TBAm": (3.0, 12.0),
    "MBAAm":  (3.0, 11.0), "EGDMA": (3.0, 11.0),
    "4VIm":   (4.0, 8.0),   # imidazole protonation pKa 6
    # Catechols — auto-oxidize above pH 7
    "DA":     (4.0, 7.0),   "NE":   (4.0, 7.0),
    # Boronates — pKa 8.8 for APBA, optimal binding at pH > 7.5
    "APBA":   (3.0, 11.0),  # stable but only binds diol at pH > 7.5
    "VPBA":   (3.0, 11.0),
    "FPBA":   (3.0, 11.0),
    "AAPBA":  (3.0, 11.0),
    # Crosslinkers (vinyl)
    "DVB":    (1.0, 13.0), "TRIM": (3.0, 12.0), "TMOS": (3.0, 10.0),
}


def synthesis_pH_window(monomers: list) -> tuple:
    """Compute the intersection of pH stability ranges across selected monomers.

    Returns (pH_min, pH_max, recommended_pH) or (None, None, None) if no
    intersection exists. The recommended pH defaults to the midpoint, or
    biased toward 8.5 if any boronate is present (boronate optimum).
    """
    ranges = [PH_STABILITY[m] for m in monomers if m in PH_STABILITY]
    if not ranges:
        return None, None, None
    pH_min = max(r[0] for r in ranges)
    pH_max = min(r[1] for r in ranges)
    if pH_min >= pH_max:
        return pH_min, pH_max, None  # No overlap
    # Bias recommendation toward 8.5 if boronate present
    boronate_present = any(m in {"APBA", "VPBA", "FPBA", "AAPBA"}
                           for m in monomers)
    if boronate_present and 8.0 <= pH_max:
        recommended = min(8.5, pH_max - 0.3)
    else:
        recommended = (pH_min + pH_max) / 2
    return pH_min, pH_max, recommended


# ── Vinyl Co-polymerization Reactivity (Q-e scheme, Alfrey 1947) ─
# Q = resonance stabilization of the propagating radical
# e = polar effect of the substituent
# Reactivity ratio r1 = (Q1/Q2) × exp(-e1*(e1-e2))
# Successful random copolymer: r1·r2 in range [0.1, 10]
# Reference: Polymer Handbook (Wiley) IV/2 (Brandrup et al.)
QE_PARAMS = {  # (Q, e)
    "AA":     (0.83, 0.88),   "MAA":   (2.34, 0.65),
    "AAm":    (1.18, 1.30),   "NIPAm": (0.85, 0.79),
    "HEMA":   (0.74, 0.40),   "TBAm":  (1.00, 1.00),
    "MBAAm":  (1.20, 1.20),   "EGDMA": (0.78, 0.40),
    "DVB":    (1.05, -0.80),  "TRIM":  (0.85, 0.40),
    "4VIm":   (2.50, -2.00),  # electron-rich, can give alternating with electrophilic
    "DA":     (1.00, 0.30),   "NE":    (1.00, 0.30),  # catechol (estimated)
    "VPBA":   (1.00, -0.50),  # styrene-like, mild electron-donor
    "AAPBA":  (0.85, 0.85),   # acrylate-like
    # APBA/FPBA are NOT polymerizable (surface) — no Q-e
}


def reactivity_ratio_product(monomer_a: str, monomer_b: str) -> float:
    """Q-e scheme reactivity ratio product r1·r2 between two vinyl monomers.

    Returns r1·r2; ideal random copolymer: 0.1 < r1·r2 < 10.
    Returns None if either monomer lacks Q-e (non-polymerizable).
    """
    import math
    if monomer_a not in QE_PARAMS or monomer_b not in QE_PARAMS:
        return None
    Q1, e1 = QE_PARAMS[monomer_a]
    Q2, e2 = QE_PARAMS[monomer_b]
    if Q1 == 0 or Q2 == 0: return None
    r1 = (Q1 / Q2) * math.exp(-e1 * (e1 - e2))
    r2 = (Q2 / Q1) * math.exp(-e2 * (e2 - e1))
    return r1 * r2


def is_polymerization_compatible(monomer_names: list) -> tuple:
    """Check if a list of monomers can be co-polymerized in one pot.

    Returns (compatible: bool, dominant_chemistry: str, conflicts: list).
    dominant_chemistry indicates which polymerization mechanism is required.
    """
    # Gather polymerization types from any monomer/crosslinker library
    types = []
    for m in monomer_names:
        if m in ALL_MONOMERS:
            t = ALL_MONOMERS[m].get("polymerization") or ALL_MONOMERS[m].get("type", "unknown")
            types.append((m, t))

    surface_only = [m for m, t in types if t == "surface"]
    matrix_types = {t for m, t in types if t not in ("surface",)}

    # All-surface alone has no matrix → invalid
    if matrix_types == set() and surface_only:
        return False, None, [(f"{','.join(surface_only)} are surface-only; no polymerization matrix")]

    # silane and (vinyl/catechol) cannot coexist
    has_silane = "silane" in matrix_types
    has_radical = bool(matrix_types & {"vinyl", "catechol"})
    if has_silane and has_radical:
        silane_ms = [m for m, t in types if t == "silane"]
        radical_ms = [m for m, t in types if t in ("vinyl", "catechol")]
        return False, None, [(
            f"INCOMPATIBLE: silane ({','.join(silane_ms)}) + radical ({','.join(radical_ms)}) "
            f"cannot be polymerized in one pot. Choose one chemistry."
        )]

    dominant = "silane" if has_silane else "radical"
    return True, dominant, []
CROSSLINKERS = set(CROSSLINKER_LIBRARY.keys())

# ── Chemistry primary class for diversity constraint (Mavliutova 2021, Liu 2017)
# Each functional monomer maps to ONE primary recognition mode. NSGA-II requires
# top combos to span ≥3 classes (Rule 1) with ≤50% in single class (Rule 2).
# References: Liu 2017 (Acc.Chem.Res. — boronate class-selectivity),
#  Mavliutova 2021 (PMC8154165 — sialic-acid MIP needs ≥3 modes),
#  Cleland 2022 (PMC9460123 — dual-monomer 2× IF over single)
# Crosslinkers (XL) are STRUCTURAL — do NOT count toward recognition diversity.
PRIMARY_CHEM_CLASS = {
    # Boronate-diol (glycan / Ser-Thr-Tyr OH)
    "APBA":  "boronate",  "VPBA":   "boronate",
    "AAPBA": "boronate",  "FPBA":   "boronate",
    # Catechol (multi H-bond + π-π)
    "DA":    "catechol",  "NE":     "catechol",
    # Aromatic π-π (Phe/Tyr/Trp stacking)
    "PTES":  "pi_stack",  "TTMS":   "pi_stack",
    "4VIm":  "pi_stack",
    # H-bond donor (amine/urea — Asp/Glu/backbone NH)
    "APTES": "hbond_donor",   "APTMS":  "hbond_donor",
    "UPTMS": "hbond_donor",   "EDTMS":  "hbond_donor",
    "AAm":   "hbond_donor",   "NIPAm":  "hbond_donor",
    # H-bond acceptor (CN, ester carbonyl)
    "CETES": "hbond_accept",  "HEMA":   "hbond_accept",
    # Hydrophobic alkyl (Leu/Ile/Val packing)
    "IBTES": "hydrophobic",   "MTMS":   "hydrophobic",
    "DIDMS": "hydrophobic",   "TBAm":   "hydrophobic",
    # Covalent reactive (Lys ε-NH2 — epoxy/isocyanate/thiol)
    "GPTMS": "covalent",      "ICTES":  "covalent",
    "MPTMS": "covalent",      "VTMS":   "covalent",
    # Electrostatic (COOH — Lys/Arg)
    "AA":    "electrostatic", "MAA":    "electrostatic",
    # Crosslinkers — STRUCTURAL only (excluded from diversity count)
    "TEOS":  "xl_structural", "TMOS":   "xl_structural",
    "MBAAm": "xl_structural", "EGDMA":  "xl_structural",
    "DVB":   "xl_structural", "TRIM":   "xl_structural",
}

# Diversity constraint thresholds
# Calibration: must reject CD9 (3 boronates / 4 functional) but accept CD81
# (2 boronates / 3 functional, observed PCSI=1.90 in trial).
MMSD_REQUIRE_CHEMISTRY_DIVERSITY = True
MMSD_MIN_CHEMISTRY_CLASSES = 2       # Rule 1: ≥2 classes (literature ≥3 too strict)
MMSD_MAX_PER_CLASS_COUNT = 2         # Rule 2: max 2 monomers per single class (absolute)
MMSD_CHEMISTRY_ENTROPY_WEIGHT = 0.3  # Rule 3: soft bonus weight in NSGA-II

# Combined libraries
ALL_MONOMERS = {**SILANE_MONOMERS, **VINYL_MONOMERS, **CROSSLINKER_LIBRARY}

# Functional monomers only (for SMD/MMSD screening)
FUNCTIONAL_MONOMERS = {k: v for k, v in ALL_MONOMERS.items()
                       if k not in CROSSLINKERS}

# ── Phase 1: Epitope Preparation ───────────────────────────────
EPITOPE_MIN_LENGTH = 9            # minimum residues (Teixeira 2021: nonapeptide)
EPITOPE_MAX_LENGTH = 16           # Teixeira 2021: >16 causes intramolecular folding
EPITOPE_MD_TIME_NS = 20           # 16-mer peptide: 20ns sufficient for RMSD convergence
EPITOPE_RMSD_THRESHOLD = 3.0      # Å — max RMSD for "stable" epitope
EPITOPE_PLDDT_THRESHOLD = 70      # AlphaFold confidence cutoff
EPITOPE_MONOMER_MOLAR_RATIO = 25  # 5 copies per type — compact system for fast MD
EPITOPE_STABILITY_MD = True       # Sehit 2024: mandatory stability MD
# Ensemble docking: extract N conformers from Phase 1 MD, dock to each
ENSEMBLE_DOCKING = True           # dock to multiple receptor conformations
ENSEMBLE_N_CONFORMERS = 5         # number of MD snapshots to extract

# ── Ensemble symmetry (cross-target comparability) ─────────────
# Conformer extraction used to be gated on the epitope passing the RMSD
# stability check. That made the ENSEMBLE SIZE TARGET-DEPENDENT: in the
# 2026-05-13 CD run CD81 missed the 3.0 Å threshold by 0.02 Å (0.3019 nm vs
# CD63 0.2773 and CD9 0.1934) and so was docked against 1 receptor while the
# other two got 6 each. Since Phase 2 kept the best score across conformers,
# the own-vs-cross contrast partly measured ensemble size: re-parsed from the
# .dlg files the ensemble gain was CD63 -0.072, CD9 -0.139, CD81 0.000
# kcal/mol, i.e. 22% of the CD63 contrast and 53% of the CD9 contrast, with
# 4/27 monomers changing sign on CD63-vs-CD81 once removed.
ENSEMBLE_EXTRACT_REQUIRES_STABLE = False  # False: every target gets an ensemble
# EXCEPTION for PHASE1_GLYCAN_MODE="explicit": CD63 glyco receptor is a
# single CHARMM-GUI equilibrated frame (n=1) while CD9/CD81 keep their
# 20 ns MD ensembles (n=6). Setting this False accepts the asymmetry;
# the bias direction is CONSERVATIVE (CD9/CD81 get more conformational
# averaging → their scores are noise-reduced, so CD63 winning boronate
# specificity remains a strong claim). The exchangeability caveat is
# recorded in Phase 2 output.
ENSEMBLE_REQUIRE_EQUAL_N = False   # was True; relaxed for glyco asymmetry
ENSEMBLE_MERGE = "boltzmann"      # "boltzmann" | "mean" | "min" (min = legacy,
                                  # best-of-N, biased under unequal N)
ENSEMBLE_BOLTZMANN_T_K = 298.15   # K, for the Boltzmann-weighted merge

# Minimum fraction of a conformer's atoms that must lie inside the docking
# box for that conformer to be docked at all.
# THIS GUARDS A BUG THAT SILENTLY RUINED EVERY ENSEMBLE DOCKING SO FAR.
# `gmx trjconv -pbc mol` emits frames in the simulation-box frame, and the
# extracted conformers were never superposed back onto the crystal structure
# the AutoGrid box was built from. Measured on the committed CD run:
#   CD63 conf_0..conf_4  48-70 Å from the grid centre,  0%, 0%, 0%, 0%, 5%
#                        of atoms inside the box
#   CD9  conf_0..conf_4  21-35 Å from the grid centre, 81%, 52%, 48%, 17%, 50%
# CD63's md_conf1-3 consequently returned IDENTICAL statistics over all 27
# monomers (mean +1.61 kcal/mol — POSITIVE, i.e. no binding at all), because
# all three were docking into empty solvent. Phase 1 now superposes each
# frame onto the reference and drops any conformer that still fails this
# check. min-over-conformers hid the damage by always falling back to the
# crystal pose; any averaging estimator would have propagated it.
ENSEMBLE_MIN_BOX_COVERAGE = 0.90

# ── Phase 2: Single Monomer Docking (SMD) — AutoDock4 ──────────
AUTODOCK4_GA_RUNS = 50            # Lamarckian GA runs per docking
AUTODOCK4_GA_POP_SIZE = 150       # GA population size
AUTODOCK4_GA_NUM_EVALS = 2500000  # max energy evaluations
AUTODOCK4_NPTS = (60, 60, 60)     # grid points (x, y, z)
AUTODOCK4_SPACING = 0.375         # grid spacing (Å)
SMD_BE_THRESHOLD = -2.0           # kcal/mol — minimum meaningful binding
SMD_TOP_N_FOR_PHASE3 = 12         # pass top N monomers by BE to Phase 3 (per target)

# ── Docking reproducibility and receptor-prep policy (2026-08 audit) ──
# AUTODOCK_SEED
#   Consumed by utils_autodock._resolve_seed(), which mixes it with the job name
#   to give each docking its own deterministic AutoDock-GPU seed. Without a seed
#   AD-GPU seeds from time+pid: three repeats of ONE identical docking measured
#   -3.89 / -3.49 / -3.68 kcal/mol, a 0.40 kcal/mol spread on a scale where the
#   whole selectivity signal is ~1 kcal/mol.
AUTODOCK_SEED = 20260812
# RECEPTOR_PDBQT_ALLOW_NEUTRAL_FALLBACK
#   Consumed by utils_structure._allow_neutral_receptor_fallback(). Receptor
#   PDBQTs are built by ADFR prepare_receptor4 at pH 7.4. obabel has NO pH model
#   and produces a NEUTRAL protein (measured +0.058 / +0.044 / +0.048 e against
#   formal charges of +4 / -2 / -1), so a silent fallback would erase exactly the
#   target-asymmetric electrostatics the screen is trying to measure.
#   True = accept the obabel receptor anyway; results are then NOT comparable
#   across targets. Keep False.
# SMOKE TEST: MD-extracted conformer PDBQTs can drift from formal charge by
# up to ~2 e due to Gasteiger sensitivity to slight residue-position changes
# at pH 9 (N-term amine pKa 9, His pKa 6 near His-B pathway). For a smoke
# test validating end-to-end plumbing, accepting the fallback is OK — the
# absolute BE numbers carry ~kT electrostatic noise but relative rankings
# survive. Set back to False for publishable selectivity claims.
RECEPTOR_PDBQT_ALLOW_NEUTRAL_FALLBACK = True
# PHASE2_REQUIRE_FRESH_DLG
#   Consumed by phase2_smd._validate_dock_result(). The .dlg path is deterministic
#   and was never cleaned, and "success" used to be inferred from the file merely
#   existing — so a crashed docking silently returned the PREVIOUS run's DLG and it
#   was scored as a fresh measurement of a different receptor conformer. True
#   rejects a DLG older than the current docking batch. False downgrades to a
#   warning.
PHASE2_REQUIRE_FRESH_DLG = True
# PHASE2_EF_THRESHOLD
#   Consumed by phase2_smd.run_phase2()'s enrichment block. Enrichment factor of
#   real monomers over the decoy set. The decoy baseline never actually docked
#   anything before the audit (decoy_be was a None placeholder), so this criterion
#   had never executed on any target.
PHASE2_EF_THRESHOLD = 1.5
# PROTONATION_APPLY_STATES
#   Consumed by utils_structure._protonation_apply_enabled(). When True, the
#   HIP/CYM renames PROPKA predicts are WRITTEN INTO the structure; when False
#   they are predicted, logged at ERROR, recorded in the *_protonation.json
#   sidecar, and NOT applied.
#   Default False on purpose: the same PDB is the docking receptor, and nobody has
#   yet verified what ADFR prepare_receptor4 does with a HIP residue. Turning this
#   on shifts the receptor net charge by +1 per HIP, which verify_receptor_charge
#   WILL flag — that is the intended way to validate it before trusting it.
#   NOTE: propka is NOT installed in the MIPscreen environment as of 2026-08-12,
#   so this flag is currently moot and the sidecar reports
#   status="propka_not_installed".
PROTONATION_APPLY_STATES = False

# ── Fixed-charge protonation: how discrete states are chosen ───────────
# A fixed-charge force field must give each titratable site an INTEGER charge,
# but at a pH near several side-chain pKa values the true states are
# fractional. Two defensible rules, and they disagree:
#
#   "majority"    each site independently takes its more probable state
#                 (pKa > pH -> protonated). Maximum-likelihood per site, but
#                 the per-site errors do NOT cancel: most lysines sit above the
#                 working pH so every one rounds the same way. On BSA at pH 9.5
#                 this leaves the protein ~7 e too positive.
#   "net_charge"  flip the least-confident sites (fraction nearest 0.5) until
#                 the TOTAL matches the Henderson-Hasselbalch expectation.
#                 Reproduces the net charge and hence the long-range field, at
#                 the cost of putting a few surface sites in their MINORITY
#                 state, where their local charge is then wrong.
#
# Which is right depends on the physics: long-range Coulomb (a charged monomer
# drawn to a polyanion across nanometres, especially in low salt where the
# Debye length is long) is governed by the net charge; contact-level
# recognition is governed by the local charge. Only sites whose flipped state
# the force field can actually BUILD are eligible, so e.g. tyrosinate is never
# used to balance the books.
PH_CHARGE_ASSIGNMENT = "majority"   # "majority" | "net_charge"
# Site-specific pKa from PROPKA 3 instead of the textbook table. Degrades to
# the table with a log line when propka3 is not installed.
PH_USE_PROPKA = False

# ── Selectivity metric ─────────────────────────────────────────
# NEVER compare a raw docking score across targets. The three receptors
# differ in size (SASA), in provenance (one predicted, two crystallographic)
# and in evidential quality, and each of those shifts a target's whole energy
# scale; a raw own-minus-cross subtraction passes the shift through
# undiluted. What survives a per-target monotone rescaling is RANK ORDER, so
# the primary metric is rank-based and the raw ΔΔG is kept only as a flagged
# diagnostic. One of:
#   "rank_delta"   mean(cross rank) - own rank; positive = selective  [default]
#   "z_ddg"        own-vs-cross after per-target standardisation
#   "own_cross_ratio"  BE(own)/mean(BE(cross)); >1 = selective
#   "ddg_raw"      LEGACY absolute difference — only valid when the receptor
#                  exchangeability check in Phase 2 reports exchangeable=True
SELECTIVITY_METRIC = "rank_delta"

# Define the docking box on the polymer-accessible surface (config key
# `scored_surface`) rather than on the whole ECL2, when the target declares
# one. Membrane-occluded residues face lipid on an intact vesicle and the
# amount of such surface is target-dependent (CD81-closed 25/89 vs CD9 9/79),
# so leaving it in the box is a cross-target confound. Targets without a
# `scored_surface` key (e.g. BSA) keep the legacy head-centred box.
RECEPTOR_BOX_ON_SCORED_SURFACE = True
# ── Membrane-accessibility filter on DOCKED POSES ──────────────
# Consumed by phase2_smd.apply_membrane_accessibility_filter().
#
# RECEPTOR_BOX_ON_SCORED_SURFACE above centres and sizes the AutoGrid box on the
# polymer-accessible surface, and that was believed to keep the membrane-facing
# residues out of the search. IT DOES NOT. Measured on the prepared templates,
# an axis-aligned box around `scored_surface` still contains
#     CD63 100/100, CD81 92/97, CD9 96/96
# membrane-occluded atoms, because those residues sit at the BASE of the same
# LEL, interleaved with the scored ones. A rectangular envelope cannot separate
# them — only a per-residue test can, and that has to happen per POSE.
#
# PHASE2_ENFORCE_MEMBRANE_ACCESSIBILITY
#   True: after docking, re-score each monomer from its best pose cluster whose
#   contacts are NOT predominantly with membrane-occluded (or unscored-EC1)
#   residues; a monomer with no accessible cluster is recorded as MISSING rather
#   than credited with binding it cannot do on an intact vesicle. Every decision
#   is logged at ERROR and recorded per monomer (pose_occlusion,
#   occluded_fraction_best_pose, binding_energy_occluded_pose).
#   False: measure and record the occluded fraction but change no score.
PHASE2_ENFORCE_MEMBRANE_ACCESSIBILITY = True
# Fraction of a pose's receptor contacts that may be with occluded residues
# before the pose is rejected. 0.5 = "more than half its contacts are on lipid".
# A STATED DEFAULT, NOT FITTED — results/ is archived, so there is nothing to
# calibrate against; the continuous occluded_fraction is always recorded so this
# can be re-cut later without re-docking.
PHASE2_MAX_OCCLUDED_POSE_FRACTION = 0.5
# Heavy-atom distance defining a ligand-receptor contact for the filter above.
PHASE2_POSE_CONTACT_CUTOFF_A = 4.0
# Sullivan 2019: use fpocket/SiteMap to identify binding pockets first
USE_BINDING_SITE_PREDICTION = True  # focused docking per site vs blind
BINDING_SITE_TOOL = "fpocket"       # "fpocket" (free) or "sitemap" (Schrödinger)
# Sullivan 2019: backbone H-bond penalty — flag monomers that disrupt 2° structure
BACKBONE_HBOND_PENALTY = True       # analyze backbone vs sidechain H-bonds
MAX_BACKBONE_HBOND_RATIO = 0.3     # >30% backbone = structural disruption risk
# Sehit 2024: monomer-epitope contact MD (10-20ns) before MMSD — mandatory
MONOMER_CONTACT_MD = True           # mandatory: run short MD per monomer-epitope pair
MONOMER_CONTACT_MD_NS = 10          # simulation time for contact frequency

# ── Phase 3: Greedy Forward Selection + MMSD ───────────────────
MMSD_MIN_COMBO_SIZE = 2           # minimum functional monomers (excl. crosslinker)
MMSD_MAX_COMBO_SIZE = 6           # maximum functional monomers
MMSD_HIGH_AFFINITY_THRESHOLD = -11.0  # kcal/mol — high-affinity PC threshold (informational)
MMSD_TOP_PC = 1                   # top PCs to pass to Phase 4 (greedy finds 1 optimal)
# BO objective weights (size-normalized scoring)
BO_INTERFERENCE_PENALTY = 0.3     # weight for interference (delta_sum > 0) penalty
# Selectivity-aware MMSD (Garcia-Ortegon 2022, Mestres 2011)
# Sullivan 2019: non-competitive binding
MMSD_COMPETITION_DISTANCE = 5.0   # Å — same-site competition check (informational)
MMSD_PENALIZE_COMPETITION = False # disabled — competition info recorded but not used for ranking
# Polymerization compatibility filter (Liu 2017 Nat. Protoc.):
# When True, Phase 3 rejects monomer combinations that mix incompatible
# polymerization chemistries (silane + radical). Strongly recommended ON
# to avoid generating chemically non-synthesizable recipes.
MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY = True

# ── MMSD docking order and Pareto selection (2026-08 audit, BLOCKER 05) ──
# MMSD_DOCKING_ORDER
#   Consumed by phase3_mmsd._order_policy() / _canonical_mmsd_order().
#   Sequential MMSD docks monomers one at a time into a growing cavity, so the
#   result DEPENDS ON THE ORDER — measured spread up to 2.02 kcal/mol for one
#   monomer across step positions (CD81/APBA). The cache used to be keyed on the
#   SORTED SET while the docking used the genotype's arbitrary gene order, so
#   exactly one arbitrary order per combination was ever measured and the
#   protocol asymmetry was read as chemistry.
#     "smd_rank"  dock strongest Phase 2 binder first, ties alphabetical. A pure
#                 function of the SET, so the measurement is reproducible and the
#                 cache entry corresponds to the order actually docked. [default]
#     "as_given"  keep the genotype order, but key the cache on the ORDERED tuple
#                 so different orders get separate measurements.
#   This is an APPROXIMATION of the order-averaged quantity, labelled as such in
#   every result record (order_dependence_note).
MMSD_DOCKING_ORDER = "smd_rank"
# MMSD_ORDER_REPLICATES
#   K > 1 docks K deterministic distinct orders and AVERAGES them, publishing
#   mmsd_sum_std_over_orders. Multiplies Phase 3 docking cost by K. Default 1.
MMSD_ORDER_REPLICATES = 1
# MMSD_ORDER_SEED
#   Seed for choosing the K distinct orders when MMSD_ORDER_REPLICATES > 1.
MMSD_ORDER_SEED = 42
# MMSD_PARETO_WEIGHTS
#   Consumed by phase3_mmsd._rank_pareto_front(): (affinity, selectivity,
#   synthesizability) weights applied AFTER min-max normalising each objective
#   across the front. The pre-audit code summed the three in RAW UNITS —
#   kcal/mol (spread ~1) plus a 0-5 score plus a 0-10 score — so synthesizability
#   dominated the ordering purely by unit mismatch. Equal weighting is a stated
#   value judgement, not a neutral choice; it lives here so it is reviewable.
MMSD_PARETO_WEIGHTS = (1.0, 1.0, 1.0)

# ── Phase 4: MD Validation — GROMACS ───────────────────────────
# Trial mode for whole-ECL2 imprinting validation: 100 ns is sufficient for
# protein-surface cavity equilibration (smaller system relative to head-template).
# Set back to 350 ns for production after method validation.
MD_PRODUCTION_NS = 200            # pre-polymerization MD (200 ns full production;
                                  # Hoshino 2008 / Sehit 2024 precedent; loop
                                  # + monomer occupancy converge by ~150 ns.
                                  # Was 350 (with sparse MM-GBSA sampling); 200
                                  # is the defensible speed/precision compromise
                                  # for a 3-target Phase 4-5 sweep on 1 GPU.
# Integration timestep. Set to 2 fs for standard H mass (1.008 Da) or 4 fs
# under HMR (H mass 4.032 Da). PHASE4_HMR_MODE at module scope overrides:
# when HMR is on, we set 4 fs here so downstream mdp writers pick it up.
MD_TIMESTEP_FS = 2.0              # integration timestep
MD_TEMPERATURE_K = 300.0          # K
MD_PRESSURE_BAR = 1.0             # bar
MD_FF_PROTEIN = "amber99sb-ildn"  # protein force field
MD_FF_MONOMER = "gaff2"           # monomer force field (via acpype)
MD_WATER_MODEL = "tip3p"
MD_BOX_TYPE = "dodecahedron"
MD_BOX_DISTANCE = 1.2             # nm — minimum distance to box edge
MD_IONIC_STRENGTH = 0.15          # mol/L — PBS condition (0.15 M NaCl)
#
# C1 fix (2026-08): pH 7.4 → 9.0. The wet-lab MIP synthesis is a sol-gel run at
# pH 9-10 (basic catalysis for TEOS/TTMS hydrolysis). Simulating at PBS pH 7.4
# built the box at the WRONG ionisation census on three separate axes:
#   * APBA boronate (pKa 8.8) — trigonal/neutral at 7.4 (0.4% tetrahedral);
#     at pH 9 it is 50/50 trigonal/tetrahedral. Diol binding is only accessible
#     to the tetrahedral form, so a pH-7 box measures binding on 0.4% of the
#     population and calls it zero. See note on APBA SMILES below (§ Vinyl).
#   * Aminopropyl silanes (APTES/APTMS/EDTMS pKa ~10.6) — 99.8% [NH3+] at 7.4,
#     ~97.5% [NH3+] at pH 9. Same functional form, but the extra 2% of neutral
#     amine at pH 9 is what condenses onto silica during sol-gel — at pH 7.4
#     the amines simply cannot condense at all.
#   * Aspartate/Glutamate (pKa ~4) — deprotonated at both pH, no change.
#   * Histidine (pKa ~6) — 90%+ neutral at pH 9 vs 20% at pH 7.4.
# The SMILES of the aminopropyl-containing library monomers (APTES, APTMS, DA,
# UPTMS, EDTMS) are the NEUTRAL form; CD does not run a protonation-split like
# BSA does (see APTESH in config_BSA.py), so no SMILES change is required.
# APBA's SMILES stays the neutral trigonal form for a first-pass simplification;
# a fair rebinding score should carry a 50% multiplicative penalty on APBA-diol
# contacts at pH 9, OR a proper CGenFF split into APBA-neutral + APBA-anion.
MD_SOLVENT_PH = 9.0               # sol-gel basic catalysis (was 7.4; wet-lab is pH 9-10)
# MM-GBSA ANALYSIS WINDOW — CHANGED (REVIEW FINDING 7): was 30-50 ns.
#
# The old values, and their comment "(last 20ns of 50ns)", were written when
# MD_PRODUCTION_NS was 50. Production is now 350 ns, so 30-50 ns covered only
# 8.6%-14.3% of the trajectory — the EQUILIBRATION transient, not the
# equilibrated tail — and the binding free energy was computed there. The
# BLOCKER fix that made the frame arithmetic correct did not help: 30-50 lies
# comfortably inside 0-350, so the range assertion passed and the number was
# stamped window_valid=True.
#
# These must be re-derived whenever MD_PRODUCTION_NS changes. They are the
# LAST 50 ns of the production run. utils_gromacs._mmpbsa_frame_window now
# refuses a window that starts in the first half of the trajectory, so a stale
# pair fails loudly instead of silently reporting the transient.
MD_MMPBSA_START_NS = 150          # last 50 ns of the 200 ns production run (25% Q4)
MD_MMPBSA_END_NS = 200            # == MD_PRODUCTION_NS (end of trajectory)
MD_MMPBSA_INTERVAL = 100          # number of frames for MM-GBSA
MD_GPU_ID = "0"                   # GROMACS GPU device ID
MD_QUICK_NS = 20                  # quick mode for debugging (20ns)
# MD_NSTXOUT_COMPRESSED — production trajectory write rate, in STEPS.
#   Consumed by utils_gromacs.run_production_md() (read via getattr, default
#   50000; values below 1000 are rejected with a ValueError).
#   CHANGED DEFAULT (2026-08 audit): was 5000 steps = 10 ps/frame at dt=2 fs.
#   50000 = 100 ps/frame. Measured effect, same system and length: md.xtc went
#   from 1158.5 kB / 51 frames to 136.3 kB / 6 frames, which scales to
#   11.33 GB -> ~1.1 GB for a 350 ns leg. A 350 ns run still yields 3,500 frames,
#   ~17x what any downstream analysis reads (analyze_trajectory strides to ~500,
#   occupancy to ~200). Lower this only if sub-100 ps kinetics are needed.
MD_NSTXOUT_COMPRESSED = 50000
# Sullivan 2019: MM-GBSA is faster and more suitable for protein-monomer
MMPBSA_METHOD = "GBSA"            # "PBSA" or "GBSA" — Sullivan used GBSA
# Sullivan 2019 / Sehit 2024: DSSP 2° structure analysis (computational CD)
DSSP_ANALYSIS = True              # track helix/sheet/coil changes during MD

# ── Phase 4: Whole-protein imprinting mode ────────────────────
# Use ECL2 (not head 16-mer) as template — matches actual MIP synthesis where
# whole protein is used. Protein backbone heavy atoms restrained during MD
# to mimic solid-phase / surface immobilization (Pluhar/Battaglia 2021 review,
# adenovirus eIP PMC11059108 protocol).
PHASE4_TEMPLATE_MODE = "ecl2"     # "head" (16-mer, legacy) | "ecl2" (whole loop) | "membrane"
PHASE4_PROTEIN_RESTRAINT_K = 1000 # kJ/mol/nm² — Cα/heavy-atom restraint during MD

# ── A1 · MEMBRANE MODE (2026-08 methodology extension) ────────
# Realistic tetraspanin condition: full-length CD (TM1-4 + ECL2 + ECL1) embedded
# in a POPC/POPS/cholesterol lipid bilayer built by CHARMM-GUI. When enabled
# and template mode == "membrane", Phase 4 skips its own box-build/solvate step
# and consumes the CHARMM-GUI Membrane Builder output (step5_input.{gro,top,pdb}).
#
# User workflow:
#   1. Upload each target's full-length PDB (or ECL1+TM1..TM4+ECL2 chimera)
#      to CHARMM-GUI Membrane Builder → OPM oriented → POPC bilayer (or your
#      lipid mix) → run through step5.
#   2. Copy step5_input.gro / step5_input.top / step5_input.pdb into
#      PHASE4_MEMBRANE_INPUT_DIR/<target>/ (or symlink).
#   3. Set PHASE4_MEMBRANE_MODE = True and PHASE4_TEMPLATE_MODE = "membrane".
#   4. Run --phase 4.
#
# Phase 4 will:
#   - Fail cleanly if any target's step5_input.gro is missing.
#   - Use CHARMM-GUI's box + solvation as-is (no re-solvate).
#   - Attach the monomer swarm around the ECL2 face (extracellular half of
#     the bilayer).
#   - Add lipid position restraints (POSRES_LIPID) at k = LIPID_POSRES_K
#     during equilibration; release them for production.
PHASE4_MEMBRANE_MODE = False       # opt-in — Off by default (fresh runs stay ECL2)
PHASE4_MEMBRANE_INPUT_DIR = "structures/membrane"  # per-target subdirs expected
PHASE4_MEMBRANE_POSRES_LIPID_K = 200  # kJ/mol/nm² on lipid P atoms during eq
# HEK293T EV lipidome mimic (Skotland 2020). Names match CHARMM-GUI toppar
# residue codes 1:1 so utils_vesicle can recognise them without translation.
PHASE4_MEMBRANE_LIPIDS = ("POPC", "POPE", "PSM", "POPS", "CHL1")
# When True, add ECL1 + TM helices to the restraint set so only ECL2 loop
# fluctuates (matches solid-phase MIP + membrane composite condition).
PHASE4_MEMBRANE_RESTRAIN_TM = True
# Where to place the monomer swarm in the CHARMM-GUI membrane box:
#   "upper_only" (default) — pack monomers into the aqueous slab above the
#                            upper leaflet only. Matches ECL2-facing extracellular
#                            imprinting; preserves current dry-run behaviour.
#   "symmetric"           — split monomers 50/50 between the upper slab and
#                            the lower slab (below the lower leaflet). Use when
#                            the target protein exposes recognisable epitopes on
#                            both faces (intracellular + extracellular) and the
#                            experimenter wants a symmetric imprint.
# Consumed by utils_gromacs.setup_from_charmm_gui_membrane and its gmx
# insert-molecules integration; ignored in the aqueous (non-membrane) mode.
PHASE4_MEMBRANE_MONOMER_PLACEMENT = "upper_only"

# ── HMR (Hydrogen Mass Repartitioning) — 4 fs timestep ────────
# CHARMM-GUI's "Hydrogen mass repartitioning" checkbox rebalances H → 4 Da /
# heavy → correspondingly reduced, allowing dt = 4 fs (vs standard 2 fs) with
# LINCS-all-bonds constraints. Set True ONLY if the CHARMM-GUI outputs at
# structures/membrane/<target>/ were generated with HMR on — otherwise the
# topology H masses stay 2 Da and 4 fs will fly apart.
# When True this forces MD_TIMESTEP_FS=4.0 downstream and the mdp templates
# switch `constraints = h-bonds` → `all-bonds` (required for HMR stability).
# HMR was tried with dt=4 fs but produced LINCS explosion + CUDA
# cudaErrorIllegalAddress crashes on this system (both aqueous protein-only
# Phase 1 stability MD and CD-in-EV Phase 4 monomer MD). CHARMM-GUI outputs
# at structures/membrane/<target>/ may not have been generated with HMR on
# (default is off), and the mix of HMR-repartitioned monomer H (3.024) with
# non-repartitioned CHARMM36m protein/lipid H (1.008) is the likely trigger.
# Falling back to dt=2 fs which is stable and adds ~30 min per 100 ns of
# Phase 4 production — acceptable cost for reliability.
PHASE4_HMR_MODE = False

# ── C2 fix (2026-08): Phase 4/5 force-field consistency strategy ──
# Phase 4 and Phase 5 (EV-approach) MERGE a monomer topology with a base
# topology (protein + optional lipid bilayer). GROMACS honours exactly ONE
# [ defaults ] block per system, and the winner is whichever include comes
# first. Two failure modes this closes:
#
#   * acpype/GAFF2 monomer itps SHIP their own [ defaults ] block with
#     fudgeLJ=0.5 and fudgeQQ=0.8333. CHARMM-GUI's charmm36.itp ships
#     fudgeLJ=1.0 and fudgeQQ=1.0. Merged silently — the base's block wins —
#     and every GAFF-parameterised monomer's 1-4 interactions are then scaled
#     wrong.  A first-pass rebinding score built on that is wrong on the
#     monomer's internal energy and on every monomer-protein contact.
#
#   * The corresponding monomer atom types (`c3`, `hc`, `oh`, ...) are GAFF's;
#     CHARMM36's [ atomtypes ] does not define them. `grompp` refuses on
#     "Atomtype X not found" unless the monomer itp embeds a full [ atomtypes ]
#     block. It does, but the LJ parameters were derived for the GAFF combining
#     rule (`comb-rule 2`), NOT CHARMM's arithmetic-mean-on-sigma. Merging
#     changes the effective LJ well depth by tens of percent.
#
# The only clean fix is to re-parameterise each monomer with CGenFF (via
# CHARMM-GUI's Ligand Reader or paramchem.org) so the whole box is one FF.
#
# 'gaff_uniform'   — pdb2gmx amber99sb-ildn, acpype GAFF2 monomers. Legacy
#                    aqueous pipeline. No CHARMM base anywhere — safe.
# 'cgenff_uniform' — CHARMM36 base + CGenFF monomers. REQUIRED for Phase 4
#                    membrane mode + Phase 5 EV-approach. Pipeline REFUSES to
#                    run unless every monomer itp is a self-contained CGenFF
#                    build (no [ defaults ] block of its own). Drop the CGenFF
#                    itps at CGENFF_MONOMER_DIR/<monomer>.itp.
# 'strict_error'   — always refuse on a mixed force field, regardless of the
#                    membrane toggle. Development-only.
# Option B (Jorge 2021 PolCA + Rappe 1992 UFF-B + Wang 2004 GAFF2 + MacKerell 1998
# CHARMM36 base). Silane monomers (Si-containing) cannot be parameterised with
# CGenFF (SilcBio 2024: "element (SI) not supported"), so cgenff_uniform is not
# achievable for a silane sol-gel MIP. The pipeline's `_apply_polca_si_overrides`
# path patches acpype/GAFF2 silicon with published PolCA Lennard-Jones values,
# and `_apply_boron_uff_overrides` patches boron with UFF Si_3 / B_3 values. FF
# mixing with the CHARMM36m protein/lipid base introduces 1-4 fudge-factor
# discrepancy (GAFF: 0.5/0.833 vs CHARMM: 1.0/1.0); the base wins for cross-
# interactions. Documented limitation; standard for MIP MD literature.
PHASE4_FF_STRATEGY = "gaff_uniform"
CGENFF_MONOMER_DIR = "structures/monomers/cgenff"

# ── Phase 5 · EV-templated MIP with Triton lysis ──────────────
# Between Phase 4 (CD-in-EV + monomers polymerised) and Phase 5 (fresh EV
# rebinding), an explicit removal step deletes lipids + template CD from the
# system, leaving the monomer cavity in aqueous phase — this models Triton
# X-100 detergent lysis of the template EV.
PHASE5_TRITON_REMOVAL_MODE = True
# Number of independent fresh-EV placements per snapshot (each with a
# different z-axis rotation seed). N=3 gives feasibility-priority first-run
# statistics; bootstrap CI needs N>=3, publication rigour prefers N=5.
PHASE5_FRESH_EV_PLACEMENTS = 3
# Extracellular gap (nm) between the top of the MIP cavity and the bottom of
# the incoming fresh EV at t=0. Small gap → faster approach kinetics but risks
# initial-condition bias; large gap → longer simulation before contact.
PHASE5_FRESH_EV_APPROACH_GAP_NM = 4.0
# Z-extension (nm) added to the simulation box to accommodate fresh EV +
# solvation slab above the cavity. Applied via `gmx editconf -box` before
# `gmx solvate` fills the new volume.
PHASE5_BOX_Z_EXTEND_NM = 15.0
# BLOCKER C4 (2026-08-20): PHASE5_FRESH_EV_PLACEMENTS previously produced N
# z-axis ROTATIONS of the same CHARMM-GUI equilibrated system — statistically
# N=1 with rotational sampling only. A reviewer will reject that as inadequate
# replicates. This knob selects how the N fresh-EV placements are made
# independent from each other:
#
#   "rotation_only"     — LEGACY. N rotations of ONE CHARMM-GUI EV. Preserved
#                          for reproducibility of pre-C4 runs; do NOT ship.
#   "nvt_perturbation"  — (RECOMMENDED default) pre-run N short NVTs from the
#                          equilibrated CHARMM-GUI EV with DIFFERENT gen_vel
#                          seeds; each final .gro is one independent draw from
#                          the equilibrium ensemble (different lipid registries
#                          + protein side-chain conformations).
#   "external_replicas" — expects the user to have produced N distinct
#                          CHARMM-GUI outputs under
#                          structures/membrane/<target>/replicas/replica_<i>/
#                          — the gold standard for a manuscript. Errors if
#                          any replica directory is missing.
PHASE5_FRESH_EV_INDEPENDENCE = "nvt_perturbation"
# Length (ps) of each NVT perturbation run when
# PHASE5_FRESH_EV_INDEPENDENCE == "nvt_perturbation". 100-200 ps is enough for
# lipid-headgroup registry decorrelation and side-chain reshuffling; longer
# runs give more independent samples but each replica costs ~30 min on a GPU
# for a typical CD-in-EV system.
PHASE5_FRESH_EV_NVT_TIME_PS = 200.0

# ── A2 · N-GLYCAN TREE (2026-08 methodology extension) ────────
# Attach Man3GlcNAc2 core (or user-supplied glycan) to N-X-S/T sequons
# in each target's ECL2/full structure.
#
# Modes:
#   "none"     — no glycan (default, current behaviour)
#   "external" — Phase 1 reads a pre-glycosylated PDB from
#                PHASE1_GLYCAN_INPUT_DIR/<target>_glyco.pdb (user builds via
#                CHARMM-GUI Glycan Reader & Modeler, GLYCAM, or similar).
#                Fails cleanly if the file is missing.
#   "auto"     — future stub: attach Man3GlcNAc2 automatically from RDKit +
#                a rotamer library. Not yet implemented → falls back to
#                "external" with a warning if selected.
#   "explicit" — Phase 1 CARVES the ECL2 protein + covalent N-glycans out of
#                the CHARMM-GUI Membrane Builder output at
#                structures/membrane/<target>/step5_input.gro and builds a
#                SECOND receptor PDBQT (CD63_receptor_glyco.pdbqt) alongside
#                the naked one.  Currently gated to CD63 only inside Phase 1
#                (the only target with ECL2-internal N-glycans); CD9 and CD81
#                keep the naked receptor whether accessed as own or cross
#                target.  Phase 2/3 read receptor_pdbqt_glyco when it exists;
#                setting this back to "none" gives byte-identical behaviour
#                to the pre-glyco pipeline (nothing is written, nothing is
#                read).  This mode enables the glycan-aware differentiation
#                needed to break the near-degenerate CD63/CD9/CD81 monomer
#                picks that appeared with naked ECL2 receptors.
PHASE1_GLYCAN_MODE = "explicit"
PHASE1_GLYCAN_INPUT_DIR = "structures/glycosylated"
# Which residues to consider as glycosylation sites — dropdown mirroring
# properties.n_glycan_sites_known. Overridden per target via
# TARGETS[t]["glycan_sites"] if present.
PHASE1_GLYCAN_SEQUON_REGEX = r"N[^P][ST]"

# ── Phase 4: replicas, acceptance and box composition (2026-08 audit) ──
# All consumed via phase4_md_validation._cfg(), which falls back LOUDLY (an
# ERROR log naming the key) if any of them is absent.
#
# SOLGEL_Q_MOLE_FRACTION
#   Mole fraction of Q units (tetraalkoxysilane crosslinker: TEOS/TMOS) in a
#   sol-gel pre-polymerisation box. The old formula
#   `copies = total // (n_functional + 1)` produced 72 mol% organosilane T units
#   where the closest published APTES/TEOS protein-imprinted sol-gel uses ~60
#   mol% Q. total_monomers is UNCHANGED, so atom count, box size and GPU cost
#   are unchanged — only the composition moves.
SOLGEL_Q_MOLE_FRACTION = 0.60
# RADICAL_CROSSLINKER_MOLE_FRACTION
#   Same quantity for free-radical (vinyl/catechol) systems. 0.50 reproduces the
#   pre-audit hardcoded behaviour EXACTLY, so radical boxes are unchanged.
RADICAL_CROSSLINKER_MOLE_FRACTION = 0.50
# PHASE4_N_REPLICAS
#   Independent trajectories per (target, pc_id), each with its own deterministic
#   seed for monomer placement AND velocity generation, written to
#   <target>/<pc_id>/rep_<i>/md/. Phase 5 spreads its snapshots across them:
#   within ONE trajectory the snapshot observable has lag-1 autocorrelation
#   rho1 = 0.23 (10 snapshots = 6.3 effective samples), so extra snapshots from a
#   single leg are nearly free of information while extra replicas are not.
#   MULTIPLIES PHASE 4 GPU COST BY THIS FACTOR.
# 3, per the user's decision (2026-08-12): "keep 10 snapshots, run Phase 4 as
# 2-3 independent replicas". A repair pass had moved this to 5 with
# REBINDING_N_SNAPSHOTS dropped to 5 ("one per replica"); that alternative is
# defensible — for a cluster bootstrap the number of REPLICAS is what carries
# the degrees of freedom, so 5x5 gives 5 independent clusters against 3x10's 3 —
# but it was not what was asked for, and it silently halved the snapshot count.
# Restored. Revisit deliberately if the variance pilot argues for it.
PHASE4_N_REPLICAS = 3
# PHASE4_CONVERGENCE_TOL_PCT
#   Max Q3→Q4 block-averaged contact-frequency difference for a replica to be
#   ACCEPTED. This number was computed and then acted on by nothing: every leg
#   reported success while window differences ran up to 73%.
PHASE4_CONVERGENCE_TOL_PCT = 10.0
# PHASE4_CONVERGENCE_GATE — "advisory" (report but do not block leg acceptance)
# vs "blocking" (reject leg if any monomer's Q3→Q4 contact-freq window exceeds
# PHASE4_CONVERGENCE_TOL_PCT). At 20 ns quick-md, per-monomer window differences
# routinely hit 50-100% (natural binding-unbinding turnover) so "blocking" rejects
# every replica. Under 350 ns full production the number is more informative but
# still often exceeds 10% for a single replica; the REPLICATE CI (across N=3
# replicas) is the proper uncertainty measure, not intra-replica block variance.
# BSA experiment also uses "advisory" (see config_BSA.py:832) — CD aligned.
PHASE4_CONVERGENCE_GATE = "advisory"
# PHASE4_RMSD_DRIFT_TOL_NM
#   Max |mean backbone RMSD(Q4) - mean(Q3)| for a replica to be ACCEPTED.
PHASE4_RMSD_DRIFT_TOL_NM = 0.05
# PHASE4_ALLOW_STALE_RESUME
#   Each replica writes md/phase4_inputs.json fingerprinting its epitope hash,
#   monomer SMILES, copy numbers and seed. Resumption requires a MATCH. True (or
#   the PHASE4_ALLOW_STALE_RESUME env var) downgrades the refusal to a warning —
#   only for a deliberate one-off recovery.
PHASE4_ALLOW_STALE_RESUME = False
# PHASE4_KEEP_CENTERED_TRAJECTORY
#   md_centered.xtc is a full-size duplicate of md.xtc (measured 11.19 GB against
#   11.33 GB on one 350 ns leg) consumed by a single analysis pass. False deletes
#   it once _analyze_monomer_occupancy has read it. True retains it for manual
#   inspection at ~1 GB per leg per replica.
PHASE4_KEEP_CENTERED_TRAJECTORY = False

# ── Phase 5/6: VIP Cavity Rebinding (Zink 2018, two-tier restraint) ───────
REBINDING_MD_NS = 50              # rebinding simulation time per snapshot (extended from 20ns)
# A4 (2026-08): switch from 10-snap-per-replica × 1 replica statistics to
# 1-snap-per-replica × N_REPLICAS. Within one 350-ns trajectory the snapshot
# observable has lag-1 autocorrelation rho1 ≈ 0.23 → 10 correlated snapshots ≈
# 6.3 effective samples, whereas N replicas each seeded independently gives N
# genuinely independent draws. Set both to 5 for a balanced N × 1 layout;
# earlier trial with N_SNAPSHOTS=10 remains valid data but confounds intra-
# trajectory correlation with real variance.
REBINDING_N_SNAPSHOTS = 10        # per replica; user decision 2026-08-12
REBINDING_RMSD_THRESHOLD = 5.0    # Å — template stays in cavity if RMSD < this
REBINDING_RESTRAINT_K = 1000      # kJ/mol/nm² — position restraint on FUNCTIONAL monomers
# Two-tier restraint (Yuan 2024, adenovirus eIP protocol):
#  - Crosslinker = irreversible C-C covalent network → very stiff
#  - Functional monomer = non-covalent H-bond anchor → moderate (allows recognition)
REBINDING_CROSSLINKER_RESTRAINT_K = 5000  # kJ/mol/nm² — rigid matrix
REBINDING_PROTEIN_RESTRAINED = True       # keep ECL2 Cα restrained during rebinding

# Trial / quick mode: 1 snapshot per target for method validation
REBINDING_TRIAL_MODE = False      # if True: N_SNAPSHOTS=1, MD_NS=30 (override above)

# Steric Complementarity Score (SCS) — size-exclusion selectivity.
# When a cross-target protein is too large for the cavity, it clashes severely
# with the restrained monomers → this IS selectivity (shape/size exclusion,
# Hoshino 2008, Shea group). Instead of running MD that explodes, we count
# heavy-atom clashes after rigid placement: high clash = REJECTED = selective.
REBINDING_CLASH_CUTOFF_A = 2.0    # Å — heavy-atom pair below this = clash
REBINDING_CLASH_THRESHOLD = 30    # >this many clashes → size-excluded (selective)

# ── Phase 5: symmetric placement and the CONTACT observable (2026-08 audit) ──
# All consumed via phase5_rebinding._cfg(name, default).
#
# REBINDING_SYMMETRIC_PLACEMENT
#   BLOCKER 07. The own and cross legs used to be DIFFERENT PROTOCOLS: the own
#   leg continued from the equilibrated Phase 4 pose while the cross leg was
#   COM-shifted in, and waters were deleted around the OLD protein's position —
#   the one place guaranteed to be empty. True runs ONE procedure for both legs
#   (strip template -> isolated pdb2gmx -> principal-axis placement scored by
#   Chamfer distance to the vacated volume -> clash-based solvent deletion ->
#   topology surgery -> genion charge rebalance -> EM with a convergence gate).
#   CHANGED DEFAULT: own-leg numbers WILL move relative to results_v1; that is
#   the point. False restores the old asymmetric path and stamps results
#   `legacy_continuation`, whose SI is not a valid comparison.
REBINDING_SYMMETRIC_PLACEMENT = True
# The persistent-contact observable. These four define "bound" for Phase 5 AND
# must stay in step with code/pipeline/utils_pcsi_star.py — the whole reason Phase 5 switched to
# this statistic is so the two cannot disagree.
REBINDING_CONTACT_CUTOFF_A = 6.0        # Å — residue-monomer heavy-atom contact
REBINDING_CONTACT_PERSISTENCE = 0.5     # fraction of analysed frames in contact
REBINDING_CONTACT_LAST_FRAC = 0.5       # analyse the last 50% of the trajectory
# Minimum persistent residues for a leg to count as `rebound`. 1 is exactly
# pcsi_star's gate (c). DELIBERATELY PERMISSIVE — it means "formed at least one
# persistent contact", NOT "bound well". The continuous fraction_persistent is
# what feeds SI, the contrast D and the statistics.
REBINDING_MIN_PERSISTENT_RESIDUES = 1
# REBINDING_EM_FMAX_TOL
#   kJ/mol/nm. A rebinding leg whose EM stalls above this without GROMACS
#   reporting convergence is ABORTED with status EM_NOT_CONVERGED rather than
#   silently producing a number. Equal to MDP_EM's emtol, deliberately strict.
REBINDING_EM_FMAX_TOL = 1000.0
# REBINDING_WATER_CLASH_CUTOFF_A
#   Å. Solvent/ion residues within this of the PLACED template are deleted whole.
#   2.4 is phase5_rebinding's own built-in default and the value its build was
#   verified against (89 waters removed on the own leg, 98 on the cross leg, both
#   reaching EM convergence). It is deliberately LARGER than
#   REBINDING_CLASH_CUTOFF_A = 2.0, which is a heavy-atom clash test on the
#   protein; this one must also clear a water's hydrogens.
REBINDING_WATER_CLASH_CUTOFF_A = 2.4
# REBINDING_REMOVAL_CONTACT_DROP_FRAC
#   Template-removal verdict: escaped if k_persistent(Q4) == 0 OR
#   k_persistent(Q4)/k_persistent(Q1) <= this. Replaces `rmsd_end > 5.0 Å`, which
#   made every target read STUCK BY CONSTRUCTION — the observed range of that
#   statistic across all completed legs was 2.04-3.87 Å against a 5.0 Å threshold,
#   so `escaped` was False for every leg that ever ran.
#   A STATED DEFAULT, NOT FITTED: results/ is archived cold storage, so there is
#   no data to calibrate against. The k_q4 == 0 branch is knob-independent and the
#   continuous `retention` is always reported. Calibrate on the first replica set.
REBINDING_REMOVAL_CONTACT_DROP_FRAC = 0.25
# Crosslinker ratio sweep (Phase 4 supplementary; Phase 6 default 5%)
CROSSLINKER_RATIO_SWEEP = [0.03, 0.05, 0.08, 0.10]
CROSSLINKER_SWEEP_MD_NS = 30      # short MD per ratio for cavity stability check (reduced for faster sweep)

# ── Phase 5: Recipe ────────────────────────────────────────────
POLYMERIZATION_SILANE = "sol-gel"
POLYMERIZATION_VINYL = "free-radical"
POLYMERIZATION_SOLIDPHASE = "solid-phase"  # Sehit 2024: glass bead + solid-phase
# Teixeira 2021: dual-epitope for glycoprotein targets
DUAL_EPITOPE_CD63 = True          # peptide + glycan epitope for CD63
GLYCAN_EPITOPE = "N-acetylneuraminic acid"  # sialic acid for CD63 glycan layer
# Teixeira 2021: performance targets
TARGET_KD_NM = 50                 # target KD < 50 nM
TARGET_IF_MIN = 3                 # minimum IF > 3 (good: >5)
# Kowalczyk 2023: validation protocols
VALIDATION_SPR = True             # SPR binding kinetics (two-state model)
VALIDATION_QCM_D = True           # QCM-D mass/viscoelastic
VALIDATION_CD_SPECTROSCOPY = True # CD spectroscopy for 2° structure check

# ── Pipeline Parameters ────────────────────────────────────────
N_WORKERS = 4                     # parallel docking processes (GPU 1개에 4 이하 권장)
USE_GPU = True
PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "results")
OUTPUT_DIRS = {
    "phase1":   f"{OUTPUT_DIR}/phase1",
    "phase2":   f"{OUTPUT_DIR}/phase2",
    "phase3":   f"{OUTPUT_DIR}/phase3",
    "phase4":   f"{OUTPUT_DIR}/phase4",
    "phase5":   f"{OUTPUT_DIR}/phase5",   # rebinding
    "phase6":   f"{OUTPUT_DIR}/phase6",   # recipe
    "reports":  f"{OUTPUT_DIR}/reports",
}


def get_output_path(phase_key: str) -> _Path:
    """Return Path for a phase output directory, creating it if needed."""
    p = _Path(OUTPUT_DIRS[phase_key])
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path_str: str) -> _Path:
    """Resolve a path from result JSONs. Handles relative paths portably."""
    p = _Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to project root
    resolved = PROJECT_ROOT / p
    if resolved.exists():
        return resolved
    return p  # return as-is, let caller handle missing file


# ── Physical Constants ──────────────────────────────────────────
KB_KCAL = 0.001987204             # Boltzmann constant kcal/(mol·K)
TEMPERATURE = 298.15              # K
HARTREE_TO_KCAL = 627.509         # 1 Hartree in kcal/mol

# ── External Tool Paths (auto-detected or user-set) ────────────
import shutil as _shutil
import os as _os

# Add GROMACS conda env bin to PATH for tool discovery
_GROMACS_BIN = _Path(_os.path.expanduser("~/anaconda3/envs/GROMACS/bin"))
if _GROMACS_BIN.exists():
    _os.environ["PATH"] = str(_GROMACS_BIN) + _os.pathsep + _os.environ.get("PATH", "")

AUTODOCK4_BIN = _shutil.which("autodock4") or str(_GROMACS_BIN / "autodock4")
AUTOGRID4_BIN = _shutil.which("autogrid4") or str(_GROMACS_BIN / "autogrid4")
# AutoDock-GPU: same force field as AD4 but ~100-350x faster on GPU
# Falls back to AutoDock4 CPU if not available
# AutoDock-GPU: check PATH first, then known install locations
_ADGPU_SEARCH = [
    _shutil.which("autodock_gpu_128wi"),
    _shutil.which("autodock_gpu_64wi"),
    _shutil.which("autodock_gpu"),
    str(_Path(_os.path.expanduser("~/Research/AutoDock-GPU/bin/autodock_gpu_128wi"))),
    str(_GROMACS_BIN / "autodock_gpu_128wi") if _GROMACS_BIN.exists() else None,
]
AUTODOCK_GPU_BIN = next((p for p in _ADGPU_SEARCH
                         if p and _Path(p).exists()), None)
USE_AUTODOCK_GPU = USE_GPU and AUTODOCK_GPU_BIN is not None
PREPARE_RECEPTOR = (_shutil.which("prepare_receptor4")
                    or _shutil.which("prepare_receptor4.py")
                    or _shutil.which("prepare_receptor")
                    or "prepare_receptor4.py")
PREPARE_LIGAND = (_shutil.which("prepare_ligand4")
                  or _shutil.which("prepare_ligand4.py")
                  or _shutil.which("prepare_ligand")
                  or "prepare_ligand4.py")
# Use GROMACS GPU build (2025.2) — /usr/bin/gmx is old 2021.4
GMX_BIN = "/usr/local/gromacs-gpu/bin/gmx"
ACPYPE_BIN = _shutil.which("acpype") or "acpype"

# ═══════════════════════════════════════════════════════════════════════════
# ── EXPERIMENT OVERRIDE LAYER ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# 여기 윗줄까지가 CD(tetraspanin ECL2) baseline 이고, results/ 를 만들어 낸
# 바로 그 코드다. 새 실험을 추가할 때 위쪽을 고치지 말 것 —
# code/pipeline/config_<NAME>.py 를 새로 만들고 KNOWN_EXPERIMENTS 에 등록한다.
#
#     MIP_EXPERIMENT=BSA python3 -m pipeline.run_pipeline --target BSA --phase 4
#     (unset  ->  "CD"  ->  today's behaviour, object for object)
#
# WHY exec() AND NOT `from .config_BSA import *`:
#   1. CLOSURE TRAP. The five functions above (get_output_path, resolve_path,
#      is_polymerization_compatible, synthesis_pH_window,
#      reactivity_ratio_product — 30 of the ~115 import sites) read OUTPUT_DIRS /
#      PROJECT_ROOT / ALL_MONOMERS / PH_STABILITY / QE_PARAMS as bare globals
#      through __globals__, i.e. THIS dict. exec(..., globals()) means rebinding
#      those tables below redirects the functions. A star-import copies the
#      function OBJECT, whose __globals__ still points here, so
#      is_polymerization_compatible would keep reading CD's 24-monomer library
#      while every constant around it said BSA — and get_output_path would
#      return CD paths and then mkdir them.
#   2. _Path / _shutil / _os / _GROMACS_BIN / _ADGPU_SEARCH are already bound and
#      usable by override files. `import *` silently drops all five.
#   3. __file__ inside the exec'd code is THIS file, so the PROJECT_ROOT
#      three-.parent-hop cannot break on an override file's directory depth.
#   4. Override files are DELTAS. All 148 public symbols are already bound, so a
#      name an override forgets keeps its baseline value. The "config_BSA is
#      missing a symbol -> ImportError six hours into an MD run" failure class
#      cannot occur.
# The tool-detection block (L706-739) and its NON-IDEMPOTENT PATH prepend run
# exactly ONCE, above, in original order. Both experiments inherit identical
# binary paths.
#
# pipeline.config stays an ordinary module with ordinary attributes: no
# __getattr__, no sys.modules swap, no proxy. So
#     import pipeline.config as cfg; cfg.SELECTIVITY_WEIGHT = w
# (run_selectivity_sweep.py:30) keeps working, and the deferred
# `from .config import SELECTIVITY_WEIGHT` at phase3_mmsd.py:406 keeps seeing the
# mutation across the `del sys.modules["pipeline.phase3_mmsd"]` reload at
# run_selectivity_sweep.py:69.


def _rederive():
    """Recompute every import-time-derived symbol from its CURRENT inputs.

    Call as the LAST statement of an override file that touched any of:
        SILANE_MONOMERS | VINYL_MONOMERS | CROSSLINKER_LIBRARY
        OUTPUT_DIR | USE_GPU
    Anything that must override a DERIVED value goes AFTER the call.

    Mirrors, in order, config.py lines 492 / 539 / 542 / 670-678 / 728.
    >>> IF YOU EDIT THOSE FIVE PLACES, EDIT THIS FUNCTION. <<<
    Idempotent. test_config_regression.py asserts it is a no-op for CD, which is
    what catches drift — the CD path never calls it, so a divergence would break
    only BSA, silently.
    """
    g = globals()
    g["CROSSLINKERS"] = set(g["CROSSLINKER_LIBRARY"].keys())                 # L492
    g["ALL_MONOMERS"] = {**g["SILANE_MONOMERS"],                             # L539
                         **g["VINYL_MONOMERS"],
                         **g["CROSSLINKER_LIBRARY"]}
    g["FUNCTIONAL_MONOMERS"] = {k: v for k, v in g["ALL_MONOMERS"].items()   # L542
                                if k not in g["CROSSLINKERS"]}
    g["OUTPUT_DIRS"] = {k: f"{g['OUTPUT_DIR']}/{k}" for k in                 # L670-678
                        ("phase1", "phase2", "phase3",
                         "phase4", "phase5", "phase6", "reports")}
    g["USE_AUTODOCK_GPU"] = (g["USE_GPU"]                                    # L728
                             and g["AUTODOCK_GPU_BIN"] is not None)


KNOWN_EXPERIMENTS = ("CD", "BSA")
EXPERIMENT = _os.environ.get("MIP_EXPERIMENT", "CD").strip()
_EXPERIMENT_SOURCE = "MIP_EXPERIMENT" if _os.environ.get("MIP_EXPERIMENT") else "default"

if EXPERIMENT not in KNOWN_EXPERIMENTS:
    # Fail LOUD, never fall back. warn-and-continue would run CD and write into
    # results/. `MIP_EXPERIMENT=bsa` is a typo, not a request for CD, and a
    # stderr warning is invisible in a multi-hour GROMACS log.
    raise ImportError(
        f"MIP_EXPERIMENT={EXPERIMENT!r} is not one of {KNOWN_EXPERIMENTS} "
        f"(case-sensitive). Unset MIP_EXPERIMENT for the default CD run.")

_OVERRIDE = _Path(__file__).resolve().parent / f"config_{EXPERIMENT}.py"

__CONFIG_DISPATCH__ = True          # sentinel the override files assert on
# Which names the delta file ASSIGNS, read from its source. Comparing VALUES
# cannot answer this: config_BSA.py sets MD_TIMESTEP_FS = 2.0 and the baseline
# is also 2.0, so "the experiment chose 2 fs" and "nobody touched it" are the
# same number. The HMR guard below needs the distinction. Empty for CD, whose
# delta is pure documentation.
_DELTA_ASSIGNED = frozenset()
if _OVERRIDE.is_file():
    # compile(..., str(_OVERRIDE), ...) keeps tracebacks pointing at the real
    # file and line inside config_BSA.py rather than at "<string>".
    _DELTA_SRC = _OVERRIDE.read_text(encoding="utf-8")
    import ast as _ast
    _DELTA_ASSIGNED = frozenset(
        _t.id
        for _n in _ast.walk(_ast.parse(_DELTA_SRC, str(_OVERRIDE)))
        if isinstance(_n, _ast.Assign)
        for _t in _n.targets
        if isinstance(_t, _ast.Name))
    exec(compile(_DELTA_SRC, str(_OVERRIDE), "exec"), globals())
elif EXPERIMENT == "CD":
    # PACKAGING RESILIENCE. config_CD.py is a PURE DOCUMENTATION delta — it
    # contains zero value assignments, because config.py's own body IS the CD
    # experiment (see that file's docstring). Making it mandatory would give the
    # baseline experiment a hard runtime dependency on a file that a
    # `git commit -am`, a fresh clone, `git worktree add`, `git clean -fd` or any
    # tracked-files-only deploy can leave behind — and CD would then fail to
    # import AT ALL. So a missing CD delta degrades to an empty delta.
    # Every other experiment still hard-fails: their deltas carry OUTPUT_DIR and
    # the pinned libraries, so silently running them without their delta would
    # execute CD's values under their name and write into results/.
    EXPERIMENT_LABEL = "CD63/CD81/CD9 tetraspanin ECL2 epitope imprinting"
    UNCONSUMED_KEYS = frozenset()
    import sys as _sys
    print(f"[config] NOTE: {_OVERRIDE.name} is absent; CD is running from "
          f"config.py's own body (this is a safe, fully-equivalent fallback, "
          f"but the file should be restored/committed).", file=_sys.stderr)
else:
    raise ImportError(
        f"experiment {EXPERIMENT!r} selected but {_OVERRIDE} is missing. Only the "
        f"CD baseline may run without its delta file (config.py's body IS CD); "
        f"every other experiment's OUTPUT_DIR and monomer libraries live in its "
        f"delta, so running without it would execute CD's values.")

# ── GUARD 1 (STRUCTURAL, the important one) ────────────────────────────────
# Any non-CD experiment MUST write outside the CD tree. This is the single
# mechanical block on the ~500 GB data-loss scenario: it does not depend on env
# var hygiene, on stamp files, or on anyone remembering to set OUTPUT_DIR.
#
# IT CHECKS OUTPUT_DIRS, NOT JUST OUTPUT_DIR. Consumers never read the scalar —
# they call get_output_path(key), which indexes OUTPUT_DIRS (and mkdirs). An
# override file that follows the documented "derived-symbol overrides go AFTER
# the _rederive() call" recipe can therefore retarget any individual phase dir
# into results/ while OUTPUT_DIR still points somewhere harmless. Validate every
# path this module can hand out.
if EXPERIMENT != "CD":
    _cd_tree = (PROJECT_ROOT / "results").resolve()

    def _reject_cd_tree(_label, _value):
        _p = _Path(_value).resolve()
        if _p == _cd_tree or _cd_tree in _p.parents:
            raise ImportError(
                f"experiment {EXPERIMENT!r} resolved {_label}={str(_value)!r}, which is "
                f"inside the CD results tree {str(_cd_tree)!r}. Refusing to start: this "
                f"would overwrite the existing CD results. Fix the path in "
                f"{_OVERRIDE.name}.")
        return _p

    _reject_cd_tree("OUTPUT_DIR", OUTPUT_DIR)
    for _k in sorted(OUTPUT_DIRS):
        _reject_cd_tree(f"OUTPUT_DIRS[{_k!r}]", OUTPUT_DIRS[_k])
    del _k, _reject_cd_tree

# ── GUARD 2 (READ-ONLY, defense in depth) ──────────────────────────────────
# Never creates or writes anything — config.py has no import-time writes today
# and must keep none. Create stamps deliberately:
#     python3 code/tools/stamp_experiment.py
# Same widening as Guard 1: check the stamp of every distinct tree this config
# can write into, not only OUTPUT_DIR's.
for _root in dict.fromkeys([str(OUTPUT_DIR)] + [str(v) for v in OUTPUT_DIRS.values()]):
    _stamp = _Path(_root) / ".experiment"
    if _stamp.is_file():
        _owner = _stamp.read_text().strip()
        if _owner != EXPERIMENT:
            raise ImportError(
                f"{_root} is stamped experiment={_owner!r} but this process "
                f"resolved {EXPERIMENT!r}. Refusing to mix result trees.")
del _root

# ── GUARD 3 (ENTRY-POINT PINNING) ──────────────────────────────────────────
# Some top-level scripts import config (so they silently adopt the ACTIVE
# experiment's monomer libraries and chemistry tables) but write to their own
# HARDCODED `results/...` literals that Guards 1 and 2 structurally cannot see.
# Under a leaked MIP_EXPERIMENT they would run the wrong experiment's values
# straight into the protected CD tree — e.g. make_process_figures.py applies
# PRIMARY_CHEM_CLASS to CD's phase3 JSONs and overwrites results/presentation.
#
# Enforced HERE, in config, so that not one line of those scripts is edited
# (the user's hard constraint). sys.argv[0] is the only thing readable this
# early: run_pipeline.py:27 and verify_phase.py:24 bind config values at module
# scope, long before any parse_args().
#
# NOT LISTED, and correctly so: code/analyze_*.py and code/midrun_pcsi_check.py
# hardcode results/ too but import NOTHING from config, so MIP_EXPERIMENT cannot
# change what they compute. Verified with
#     grep -L "pipeline.config" code/analyze_*.py code/midrun_pcsi_check.py
_CD_PINNED_SCRIPTS = frozenset({
    "make_presentation_figures.py",   # writes results/presentation/*.png
    "make_process_figures.py",        # writes results/presentation/*.png
    "run_membrane_bacteria.py",       # writes results/membrane_bacteria/
    "run_validation.py",              # writes results/validation/ (+ sullivan/)
})
if EXPERIMENT != "CD":
    import sys as _sys
    _argv0 = _Path(_sys.argv[0]).name if _sys.argv and _sys.argv[0] else ""
    if _argv0 in _CD_PINNED_SCRIPTS:
        raise ImportError(
            f"{_argv0} is CD-PINNED: it imports config (so it would pick up "
            f"experiment {EXPERIMENT!r}'s monomer libraries) but writes to hardcoded "
            f"'results/...' paths inside the CD tree. Refusing to run it under "
            f"MIP_EXPERIMENT={EXPERIMENT!r}. Unset MIP_EXPERIMENT (do not `export` "
            f"it — use `python3 run_BSA.py ...` or a one-shot `VAR=x cmd` prefix).")
    del _argv0

# ── HMR override — MUST be after all symbol definitions ──────
# When PHASE4_HMR_MODE=True the H atoms are 4 Da (repartitioned) and dt=4 fs
# is safe with LINCS all-bonds. Consumers that write mdp files (utils_gromacs
# _write_production_mdp) branch on both MD_TIMESTEP_FS and the constraint
# keyword; here we set the timestep alone — constraint switch lives in the
# mdp writer.
if globals().get("PHASE4_HMR_MODE", False):
    # REFUSE TO CLOBBER AN EXPERIMENT'S OWN CHOICE. This assignment runs AFTER
    # the delta is exec'd, so `MD_TIMESTEP_FS = 2.0` in config_<X>.py used to be
    # silently overwritten with 4.0 -- and 4 fs on a topology whose hydrogens
    # were never repartitioned is not a slower answer, it is a wrong one. An
    # experiment that wants a different timestep must say so by turning the flag
    # off (PHASE4_HMR_MODE = False), which is also the truthful statement about
    # its topology; setting only the timestep is ambiguous, so it is an error.
    if "MD_TIMESTEP_FS" in _DELTA_ASSIGNED:
        raise ImportError(
            f"config_{EXPERIMENT}.py sets MD_TIMESTEP_FS = {MD_TIMESTEP_FS} while "
            f"PHASE4_HMR_MODE is True, and this block would overwrite it with 4.0. "
            f"HMR is only valid where the TOPOLOGY carries repartitioned hydrogen "
            f"masses (CD gets that from CHARMM-GUI; a pdb2gmx/acpype topology does "
            f"NOT, and nothing in this repo repartitions one). Set "
            f"PHASE4_HMR_MODE = False in that file instead.")
    MD_TIMESTEP_FS = 4.0

if not _os.environ.get("MIP_CONFIG_QUIET"):
    import sys as _sys
    print(f"[config] experiment={EXPERIMENT} ({_EXPERIMENT_SOURCE})  "
          f"output={OUTPUT_DIR}  "
          f"unconsumed_keys={len(globals().get('UNCONSUMED_KEYS', ()))}",
          file=_sys.stderr)
