"""
Validation Configuration — Reference Data
==========================================
Experimental data from Rajpal et al. Sci. Rep. 2024 for validating
the SMD/MMSD docking pipeline against known results.

The SARS-CoV-2 spike epitope + 11 silane monomers serve as the
benchmark system. If our pipeline reproduces these rankings,
we can trust it for the novel CD63/CD81/CD9 targets.
"""

# ── Rajpal 2024 Validation Target ──────────────────────────────
# Epitope from SARS-CoV-2 Spike protein RBD (PDB: 7JMO)
RAJPAL_EPITOPE_SEQUENCE = "YQAGSTPCNGVEGFNCYFPLQSYGF"
RAJPAL_EPITOPE_PDB_ID = "7JMO"
RAJPAL_EPITOPE_CHAIN = "A"
RAJPAL_EPITOPE_RESIDUES = (473, 497)  # RBD epitope (verified in PDB 7JMO Chain A)

# ── SMD Reference Data (Rajpal 2024, Table 1) ─────────────────
# Mean binding energy (kcal/mol) of highest-ranked cluster per monomer
RAJPAL_SMD_REFERENCE = {
    "PTES":   -3.31,
    "DIDMS":  -2.93,
    "IBTES":  -2.68,
    "MTMS":   -2.65,
    "CETES":  -2.55,
    "TEOS":   -2.45,
    "UPTMS":  -2.39,
    "TMOS":   -2.26,  # Trimethoxy(octyl)silane — not in our library
    "APTES":  -2.05,
    "APTMS":  -1.95,
    "MPTMS":  -1.95,
}

# Expected SMD ranking order (best to worst)
RAJPAL_SMD_RANKING = [
    "PTES", "DIDMS", "IBTES", "MTMS", "CETES",
    "TEOS", "UPTMS", "TMOS", "APTES", "APTMS", "MPTMS",
]

# ── MMSD BE Change (Rajpal 2024, Table 2) ─────────────────────
# Qualitative change in binding energy when applied in MMSD vs SMD
RAJPAL_MMSD_CHANGES = {
    "PTES":   {"change": "same",     "pct": 0},
    "DIDMS":  {"change": "decrease", "pct": -6},
    "IBTES":  {"change": "decrease", "pct": -5},
    "MTMS":   {"change": "decrease", "pct": -7},
    "CETES":  {"change": "decrease", "pct": -6},
    "TEOS":   {"change": "decrease", "pct": -7},
    "UPTMS":  {"change": "increase", "pct": 16},
    "TMOS":   {"change": "same",     "pct": 0},
    "APTES":  {"change": "increase", "pct": 44},   # substantial
    "APTMS":  {"change": "increase", "pct": 60},   # substantial
    "MPTMS":  {"change": "increase", "pct": 30},   # substantial
}

# Key insight: monomers showing BE increase in MMSD (APTMS, APTES, MPTMS)
# performed BETTER experimentally than their individual SMD rankings suggested.
RAJPAL_MMSD_SYNERGY_MONOMERS = ["APTMS", "APTES", "MPTMS", "UPTMS"]
RAJPAL_MMSD_INTERFERENCE_MONOMERS = ["DIDMS", "IBTES", "MTMS"]

# ── Experimental IF Data (Rajpal 2024, Table 3) ───────────────
# Polymer Composition (PC) → Imprinting Factor
RAJPAL_EXPERIMENTAL_IF = {
    "PC_I":    {"monomers": ["PTES", "APTMS", "APTES", "TEOS"],   "IF": 1.73},
    "PC_II":   {"monomers": ["PTES", "APTMS", "MPTMS", "TEOS"],   "IF": 1.50},
    "PC_III":  {"monomers": ["PTES", "APTES", "MPTMS", "TEOS"],   "IF": 1.42},
    "PC_IV":   {"monomers": ["PTES", "UPTMS", "APTES", "TEOS"],   "IF": 1.12},
    "PC_V":    {"monomers": ["PTES", "DIDMS", "IBTES", "TEOS"],   "IF": None},  # low (control)
    "PC_VI":   {"monomers": ["PTES", "DIDMS", "MTMS", "TEOS"],    "IF": None},  # low (control)
    "PC_VII":  {"monomers": ["PTES", "APTES", "MPTMS", "TEOS"],   "IF": None},  # with MPTMS+APTES
    "PC_VIII": {"monomers": ["APTMS", "APTES", "MPTMS", "TEOS"],  "IF": None},  # no PTES (control)
}

# Expected ranking: PC_I > PC_II > PC_III > PC_IV > PC_V, PC_VI
RAJPAL_IF_RANKING = ["PC_I", "PC_II", "PC_III", "PC_IV"]

# ══════════════════════════════════════════════════════════════
# Sullivan et al., J. Phys. Chem. B 2019 — Myoglobin Validation
# ══════════════════════════════════════════════════════════════
# Target: equine skeletal myoglobin (PDB: 5D5R)
# 5 acrylamide-based monomers docked to 14 predicted binding sites
# Experimental validation: IF and % rebind from bulk hydrogel MIPs

SULLIVAN_TARGET_PDB = "5D5R"
SULLIVAN_TARGET_CHAIN = "A"
SULLIVAN_TARGET_NAME = "Myoglobin"

# Monomer definitions (SMILES for 5 acrylamide derivatives)
SULLIVAN_MONOMERS = {
    "AAm":        {"smiles": "C=CC(N)=O",
                   "name": "Acrylamide"},
    "NHMAm":      {"smiles": "C=CC(=O)NCO",
                   "name": "N-(Hydroxymethyl)acrylamide"},
    "NHEAm":      {"smiles": "C=CC(=O)NCCO",
                   "name": "N-(Hydroxyethyl)acrylamide"},
    "DMAm":       {"smiles": "C=CC(=O)N(C)C",
                   "name": "N,N-Dimethylacrylamide"},
    "TrisNHMAm":  {"smiles": "C=CC(=O)NC(CO)(CO)CO",
                   "name": "N-[Tris(hydroxymethyl)methyl]acrylamide"},
}

# Experimental MIP performance (Sullivan 2019, Table 1)
SULLIVAN_EXPERIMENTAL = {
    "AAm":       {"pct_rebind": 87.1, "IF": 1.77, "pct_nip": 49.2},
    "NHMAm":     {"pct_rebind": 98.9, "IF": 1.90, "pct_nip": 52.1},
    "NHEAm":     {"pct_rebind": 77.2, "IF": 1.77, "pct_nip": 43.6},
    "DMAm":      {"pct_rebind": 72.0, "IF": 1.48, "pct_nip": 48.7},
    "TrisNHMAm": {"pct_rebind": 79.9, "IF": 1.10, "pct_nip": 72.5},
}

# Expected MIP efficacy ranking (best to worst)
SULLIVAN_MIP_RANKING = ["NHMAm", "AAm", "NHEAm", "TrisNHMAm", "DMAm"]

# Sullivan Table 3: average number of monomers bound to protein
SULLIVAN_AVG_BOUND = {
    "AAm":       9.17,
    "NHMAm":     10.0,
    "NHEAm":     10.5,
    "DMAm":      10.2,
    "TrisNHMAm": 11.5,
}

# Sullivan Table 4: H-bond interaction analysis
# total H-bonds and helical backbone H-bonds per monomer
SULLIVAN_HBOND_ANALYSIS = {
    "AAm":       {"total": 22, "helical_backbone": 5},
    "NHMAm":     {"total": 27, "helical_backbone": 3},
    "NHEAm":     {"total": 22, "helical_backbone": 1},
    "DMAm":      {"total": 14, "helical_backbone": 4},
    "TrisNHMAm": {"total": 42, "helical_backbone": 8},
}

# CD spectroscopy — α-helix % (native Mb: 88.0%)
# Sullivan 2019: monomers disrupting 2° structure → poor MIP
SULLIVAN_CD_HELIX = {
    "native":    88.0,
    "AAm":       80.5,   # minimal disruption → good MIP
    "NHMAm":     55.5,   # moderate disruption but best MIP
    "NHEAm":     50.2,   # moderate disruption
    "DMAm":      34.7,   # severe disruption → worst MIP
    "TrisNHMAm": 42.6,   # severe disruption → poor IF
}

# Sullivan key insight: backbone H-bond ratio predicts disruption
# High backbone ratio → poor selectivity (except NHMAm which compensates
# with excellent site-specific binding)
SULLIVAN_BACKBONE_RATIO_THRESHOLD = 0.3  # >30% = risk

# Comonomer validation (Sullivan 2019, Table 1 bottom)
SULLIVAN_COMONOMER_MIPS = {
    "MIP_A": {
        "monomers": ["TrisNHMAm", "DMAm"],  # designed to perform BETTER
        "pct_rebind": 86.2, "IF": 2.36,
        "rationale": "Non-competitive binding at different sites",
    },
    "MIP_B": {
        "monomers": ["NHEAm", "DMAm"],  # designed to perform WORSE
        "pct_rebind": 68.3, "IF": 1.58,
        "rationale": "Competitive binding at overlapping sites",
    },
}

# ── Validation Thresholds ──────────────────────────────────────
SMD_BE_TOLERANCE = 2.0        # kcal/mol — acceptable deviation (systematic offset expected
                              # due to different PDBQT charges, grid box, preprocessing)
SMD_RANKING_SPEARMAN = 0.7    # minimum Spearman ρ for SMD ranking
MMSD_SYNERGY_DIRECTION = True # APTMS, APTES must show BE increase in MMSD
IF_RANKING_SPEARMAN = 0.6     # minimum Spearman ρ for IF ranking
SULLIVAN_RANKING_SPEARMAN = 0.6  # ρ for Sullivan BE vs IF correlation
