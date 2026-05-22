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
EPITOPE_RMSD_MAX = 3.0

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
    "APTES":  {"smiles": "NCCCO[Si](OCC)(OCC)OCC",
               "name": "(3-Aminopropyl)triethoxysilane",
               "polymerization": "silane",
               "interaction": "H-bond donor, electrostatic"},
    "APTMS":  {"smiles": "NCCCO[Si](OC)(OC)OC",
               "name": "(3-Aminopropyl)trimethoxysilane",
               "polymerization": "silane",
               "interaction": "H-bond, electrostatic"},
    "UPTMS":  {"smiles": "O=C(N)NCCCO[Si](OC)(OC)OC",
               "name": "3-Ureidopropyltrimethoxysilane",
               "polymerization": "silane",
               "interaction": "Multi H-bond (urea D+A)"},
    "MPTMS":  {"smiles": "SCCCO[Si](OC)(OC)OC",
               "name": "(3-Mercaptopropyl)trimethoxysilane",
               "polymerization": "silane",
               "interaction": "Thiol-Cys, H-bond"},
    "IBTES":  {"smiles": "CC(C)CO[Si](OCC)(OCC)OCC",
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
    "EDTMS":  {"smiles": "NCCNCCCO[Si](OC)(OC)OC",
               "name": "N-[3-(Trimethoxysilyl)propyl]ethylenediamine",
               "polymerization": "silane",
               "interaction": "Chelate-type H-bond"},
    "ICTES":  {"smiles": "O=C=NCCCO[Si](OCC)(OCC)OCC",
               "name": "3-(Triethoxysilyl)propyl isocyanate",
               "polymerization": "silane",
               "interaction": "Covalent (Lys ε-NH₂)"},
    "VTMS":   {"smiles": "C=C[Si](OC)(OC)OC",
               "name": "Vinyltrimethoxysilane",
               "polymerization": "silane",  # Si-OR dominant; vinyl can crosslink with radical comonomer
               "interaction": "Hydrophobic + crosslinkable"},
    "GPTMS":  {"smiles": "C(CO[Si](OC)(OC)OC)C1CO1",
               "name": "(3-Glycidyloxypropyl)trimethoxysilane",
               "polymerization": "silane",  # silane primary; epoxy is covalent side group
               "interaction": "Epoxy covalent (nucleophile)"},
    "DIDMS":  {"smiles": "C[Si](C)(OC)OC",
               "name": "Dimethyldimethoxysilane",
               "polymerization": "silane",
               "interaction": "Strong hydrophobic"},
    "CETES":  {"smiles": "N#CCCCO[Si](OCC)(OCC)OCC",
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
    "APBA":   {"smiles": "Nc1ccc(B(O)O)cc1",
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
    "AAPBA":  {"smiles": "C=CC(=O)Nc1ccc(B(O)O)cc1",
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
    "TRIM":  {"smiles": "C=C(C)C(=O)OCC(CC)(COC(=O)C(=C)C)OC(=O)C(=C)C",
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

# ── Phase 2: Single Monomer Docking (SMD) — AutoDock4 ──────────
AUTODOCK4_GA_RUNS = 50            # Lamarckian GA runs per docking
AUTODOCK4_GA_POP_SIZE = 150       # GA population size
AUTODOCK4_GA_NUM_EVALS = 2500000  # max energy evaluations
AUTODOCK4_NPTS = (60, 60, 60)     # grid points (x, y, z)
AUTODOCK4_SPACING = 0.375         # grid spacing (Å)
SMD_BE_THRESHOLD = -2.0           # kcal/mol — minimum meaningful binding
SMD_TOP_N_FOR_PHASE3 = 12         # pass top N monomers by BE to Phase 3 (per target)
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

# ── Phase 4: MD Validation — GROMACS ───────────────────────────
# Trial mode for whole-ECL2 imprinting validation: 100 ns is sufficient for
# protein-surface cavity equilibration (smaller system relative to head-template).
# Set back to 350 ns for production after method validation.
MD_PRODUCTION_NS = 100            # pre-polymerization MD (trial mode for ECL2; default 350 ns)
MD_TIMESTEP_FS = 2.0              # integration timestep
MD_TEMPERATURE_K = 300.0          # K
MD_PRESSURE_BAR = 1.0             # bar
MD_FF_PROTEIN = "amber99sb-ildn"  # protein force field
MD_FF_MONOMER = "gaff2"           # monomer force field (via acpype)
MD_WATER_MODEL = "tip3p"
MD_BOX_TYPE = "dodecahedron"
MD_BOX_DISTANCE = 1.2             # nm — minimum distance to box edge
MD_IONIC_STRENGTH = 0.15          # mol/L — PBS condition (0.15 M NaCl)
MD_SOLVENT_PH = 7.4               # PBS pH (for protonation state reference)
MD_MMPBSA_START_NS = 30           # MM-GBSA window start (last 20ns of 50ns)
MD_MMPBSA_END_NS = 50             # MM-GBSA window end
MD_MMPBSA_INTERVAL = 100          # number of frames for MM-GBSA
MD_GPU_ID = "0"                   # GROMACS GPU device ID
MD_QUICK_NS = 20                  # quick mode for debugging (20ns)
# Sullivan 2019: MM-GBSA is faster and more suitable for protein-monomer
MMPBSA_METHOD = "GBSA"            # "PBSA" or "GBSA" — Sullivan used GBSA
# Sullivan 2019 / Sehit 2024: DSSP 2° structure analysis (computational CD)
DSSP_ANALYSIS = True              # track helix/sheet/coil changes during MD

# ── Phase 4: Whole-protein imprinting mode ────────────────────
# Use ECL2 (not head 16-mer) as template — matches actual MIP synthesis where
# whole protein is used. Protein backbone heavy atoms restrained during MD
# to mimic solid-phase / surface immobilization (Pluhar/Battaglia 2021 review,
# adenovirus eIP PMC11059108 protocol).
PHASE4_TEMPLATE_MODE = "ecl2"     # "head" (16-mer, legacy) | "ecl2" (whole loop)
PHASE4_PROTEIN_RESTRAINT_K = 1000 # kJ/mol/nm² — Cα/heavy-atom restraint during MD

# ── Phase 5/6: VIP Cavity Rebinding (Zink 2018, two-tier restraint) ───────
REBINDING_MD_NS = 50              # rebinding simulation time per snapshot (extended from 20ns)
REBINDING_N_SNAPSHOTS = 10        # top contact frames (n=10 for statistical power)
REBINDING_RMSD_THRESHOLD = 5.0    # Å — template stays in cavity if RMSD < this
REBINDING_RESTRAINT_K = 1000      # kJ/mol/nm² — position restraint on FUNCTIONAL monomers
# Two-tier restraint (Yuan 2024, adenovirus eIP protocol):
#  - Crosslinker = irreversible C-C covalent network → very stiff
#  - Functional monomer = non-covalent H-bond anchor → moderate (allows recognition)
REBINDING_CROSSLINKER_RESTRAINT_K = 5000  # kJ/mol/nm² — rigid matrix
REBINDING_PROTEIN_RESTRAINED = True       # keep ECL2 Cα restrained during rebinding

# Trial / quick mode: 1 snapshot per target for method validation
REBINDING_TRIAL_MODE = True       # if True: N_SNAPSHOTS=1, MD_NS=30 (override above)
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
