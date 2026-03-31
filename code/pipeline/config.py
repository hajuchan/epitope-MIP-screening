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
        "source": "alphafold",        # No experimental crystal structure
        "chain": "A",
        "full_length": 238,
        "ecl2_range": (113, 201),     # ECL2 (LEL) approximate boundaries
        "head_residues": (150, 195),  # Head subdomain (helix C-D + loops)
        "n_glycan_sites": 3,          # N130, N150, N172
        "description": "CD63 LEL — heavy glycosylation, largest head",
    },
    "CD81": {
        "pdb_id": "5TCX",
        "source": "pdb",
        "chain": "A",
        "full_length": 236,
        "ecl2_range": (113, 201),
        "head_residues": (157, 190),  # Helix C-D variable region
        "n_glycan_sites": 0,          # No glycosylation
        "description": "CD81 LEL — non-glycosylated, hydrophobic head",
    },
    "CD9": {
        "pdb_id": "6K4J",
        "source": "pdb",
        "chain": "A",
        "full_length": 228,
        "ecl2_range": (112, 195),
        "head_residues": (152, 183),  # Equivalent head region
        "n_glycan_sites": 1,          # 1 N-glycan in LEL
        "description": "CD9 LEL — light glycosylation, smallest head",
    },
}

# ── Monomer Libraries ───────────────────────────────────────────
# A. Silane monomers (Sol-gel epitope imprinting, per Rajpal 2024)
SILANE_MONOMERS = {
    "PTES":   {"smiles": "CCO[Si](OCC)(OCC)c1ccccc1",
               "name": "Phenyltriethoxysilane",
               "interaction": "π-π stacking, hydrophobic"},
    "APTES":  {"smiles": "NCCCO[Si](OCC)(OCC)OCC",
               "name": "(3-Aminopropyl)triethoxysilane",
               "interaction": "H-bond donor, electrostatic"},
    "APTMS":  {"smiles": "NCCCO[Si](OC)(OC)OC",
               "name": "(3-Aminopropyl)trimethoxysilane",
               "interaction": "H-bond, electrostatic"},
    "UPTMS":  {"smiles": "O=C(N)NCCCO[Si](OC)(OC)OC",
               "name": "3-Ureidopropyltrimethoxysilane",
               "interaction": "Multi H-bond (urea D+A)"},
    "MPTMS":  {"smiles": "SCCCO[Si](OC)(OC)OC",
               "name": "(3-Mercaptopropyl)trimethoxysilane",
               "interaction": "Thiol-Cys, H-bond"},
    "IBTES":  {"smiles": "CC(C)CO[Si](OCC)(OCC)OCC",
               "name": "Isobutyltriethoxysilane",
               "interaction": "Hydrophobic, alkyl"},
    "MTMS":   {"smiles": "C[Si](OC)(OC)OC",
               "name": "Methyltrimethoxysilane",
               "interaction": "Hydrophobic"},
    "TEOS":   {"smiles": "CCO[Si](OCC)(OCC)OCC",
               "name": "Tetraethyl orthosilicate",
               "interaction": "Cross-linker"},
    "EDTMS":  {"smiles": "NCCNCCCO[Si](OC)(OC)OC",
               "name": "N-[3-(Trimethoxysilyl)propyl]ethylenediamine",
               "interaction": "Chelate-type H-bond"},
    "ICTES":  {"smiles": "O=C=NCCCO[Si](OCC)(OCC)OCC",
               "name": "3-(Triethoxysilyl)propyl isocyanate",
               "interaction": "Covalent (Lys ε-NH₂)"},
    "VTMS":   {"smiles": "C=C[Si](OC)(OC)OC",
               "name": "Vinyltrimethoxysilane",
               "interaction": "Hydrophobic + crosslinkable"},
    "GPTMS":  {"smiles": "C(CO[Si](OC)(OC)OC)C1CO1",
               "name": "(3-Glycidyloxypropyl)trimethoxysilane",
               "interaction": "Epoxy covalent (nucleophile)"},
    "DIDMS":  {"smiles": "C[Si](C)(OC)OC",
               "name": "Dimethyldimethoxysilane",
               "interaction": "Strong hydrophobic"},
    "CETES":  {"smiles": "N#CCCCO[Si](OCC)(OCC)OCC",
               "name": "3-Cyanopropyltriethoxysilane",
               "interaction": "Weak H-bond acceptor (CN)"},
    "TTMS":   {"smiles": "Cc1ccc(cc1)[Si](OC)(OC)OC",
               "name": "p-Tolyltrimethoxysilane",
               "interaction": "π-π + mild hydrophobic"},
}

# B. Vinyl/acrylic monomers (Free-radical polymerization)
VINYL_MONOMERS = {
    "AA":     {"smiles": "C=CC(=O)O",
               "name": "Acrylic acid",
               "interaction": "Electrostatic (Lys, Arg, His)"},
    "MAA":    {"smiles": "CC(=C)C(=O)O",
               "name": "Methacrylic acid",
               "interaction": "Electrostatic + mild hydrophobic"},
    "AAm":    {"smiles": "C=CC(N)=O",
               "name": "Acrylamide",
               "interaction": "H-bond (D+A)"},
    "NIPAm":  {"smiles": "CC(C)NC(=O)C=C",
               "name": "N-Isopropylacrylamide",
               "interaction": "H-bond + hydrophobic"},
    "4VIm":   {"smiles": "C=Cc1cnc[nH]1",
               "name": "4(5)-Vinylimidazole",
               "interaction": "π-π, H-bond, His mimic"},
    "HEMA":   {"smiles": "C=C(C)C(=O)OCCO",
               "name": "2-Hydroxyethyl methacrylate",
               "interaction": "H-bond (OH)"},
    "MBAAm":  {"smiles": "C=CC(=O)NCNC(=O)C=C",
               "name": "N,N'-Methylenebisacrylamide",
               "interaction": "Cross-linker"},
    "DA":     {"smiles": "NCCc1ccc(O)c(O)c1",
               "name": "Dopamine hydrochloride",
               "interaction": "Multi H-bond, catechol"},
    "NE":     {"smiles": "NCC(O)c1ccc(O)c(O)c1",
               "name": "Norepinephrine",
               "interaction": "DA-like + extra OH"},
    "TBAm":   {"smiles": "C=CC(=O)NC(C)(C)C",
               "name": "N-tert-Butylacrylamide",
               "interaction": "Hydrophobic"},
    "APBA":   {"smiles": "Nc1ccc(B(O)O)cc1",
               "name": "3-Aminophenylboronic acid",
               "interaction": "Glycan (diol) recognition — CD63 specific"},
    "EGDMA":  {"smiles": "C=C(C)C(=O)OCCOC(=O)C(=C)C",
               "name": "Ethylene glycol dimethacrylate",
               "interaction": "Cross-linker"},
}

# Combined libraries
ALL_MONOMERS = {**SILANE_MONOMERS, **VINYL_MONOMERS}

# Cross-linkers (excluded from functional monomer screening)
CROSSLINKERS = {"TEOS", "MBAAm", "EGDMA"}

# Functional monomers only (for SMD/MMSD screening)
FUNCTIONAL_MONOMERS = {k: v for k, v in ALL_MONOMERS.items()
                       if k not in CROSSLINKERS}

# ── Phase 1: Epitope Preparation ───────────────────────────────
EPITOPE_MIN_LENGTH = 9            # minimum residues (Teixeira 2021: nonapeptide)
EPITOPE_MAX_LENGTH = 16           # Teixeira 2021: >16 causes intramolecular folding
EPITOPE_MD_TIME_NS = 100          # implicit-solvent MD for stability check
EPITOPE_RMSD_THRESHOLD = 3.0      # Å — max RMSD for "stable" epitope
EPITOPE_PLDDT_THRESHOLD = 70      # AlphaFold confidence cutoff
EPITOPE_MONOMER_MOLAR_RATIO = 20  # Sehit 2024: epitope:monomer = 1:20

# ── Phase 2: Single Monomer Docking (SMD) — AutoDock4 ──────────
AUTODOCK4_GA_RUNS = 50            # Lamarckian GA runs per docking
AUTODOCK4_GA_POP_SIZE = 150       # GA population size
AUTODOCK4_GA_NUM_EVALS = 2500000  # max energy evaluations
AUTODOCK4_NPTS = (60, 60, 60)     # grid points (x, y, z)
AUTODOCK4_SPACING = 0.375         # grid spacing (Å)
SMD_BE_THRESHOLD = -2.0           # kcal/mol — minimum meaningful binding
SMD_DDG_THRESHOLD = -0.5          # kcal/mol — minimum selectivity
# Sullivan 2019: use fpocket/SiteMap to identify binding pockets first
USE_BINDING_SITE_PREDICTION = True  # focused docking per site vs blind
BINDING_SITE_TOOL = "fpocket"       # "fpocket" (free) or "sitemap" (Schrödinger)
# Sullivan 2019: backbone H-bond penalty — flag monomers that disrupt 2° structure
BACKBONE_HBOND_PENALTY = True       # analyze backbone vs sidechain H-bonds
MAX_BACKBONE_HBOND_RATIO = 0.3     # >30% backbone = structural disruption risk
# Sehit 2024: monomer-epitope contact MD (10-20ns) before MMSD
MONOMER_CONTACT_MD = True           # run short MD per monomer-epitope pair
MONOMER_CONTACT_MD_NS = 10          # simulation time for contact frequency

# ── Phase 3: Multi-Monomer Simultaneous Docking (MMSD) ─────────
MMSD_COMBO_SIZE = 4               # monomers per combination (Rajpal 2024)
MMSD_N_FIXED = 2                  # fixed slots: best-BE + cross-linker
MMSD_DEFAULT_CROSSLINKER = "TEOS" # cross-linker identity
MMSD_HIGH_AFFINITY_THRESHOLD = -11.0  # kcal/mol — MMSD sum threshold
MMSD_TOP_PC = 8                   # top polymer compositions to advance
# Sullivan 2019: non-competitive binding — monomers should occupy different sites
MMSD_COMPETITION_DISTANCE = 5.0   # Å — same-site threshold for competition check
MMSD_PENALIZE_COMPETITION = True  # penalize PCs with competing monomers

# ── Phase 4: MD Validation — GROMACS ───────────────────────────
MD_PRODUCTION_NS = 200            # production MD time
MD_TIMESTEP_FS = 2.0              # integration timestep
MD_TEMPERATURE_K = 300.0          # K
MD_PRESSURE_BAR = 1.0             # bar
MD_FF_PROTEIN = "amber99sb-ildn"  # protein force field
MD_FF_MONOMER = "gaff2"           # monomer force field (via acpype)
MD_WATER_MODEL = "tip3p"
MD_BOX_TYPE = "dodecahedron"
MD_BOX_DISTANCE = 1.2             # nm — minimum distance to box edge
MD_MMPBSA_START_NS = 150          # MM-PBSA window start
MD_MMPBSA_END_NS = 200            # MM-PBSA window end
MD_MMPBSA_INTERVAL = 100          # number of frames for MM-PBSA
MD_GPU_ID = "0"                   # GROMACS GPU device ID
MD_QUICK_NS = 50                  # quick mode for preliminary screening
# Sullivan 2019: MM-GBSA is faster and more suitable for protein-monomer
MMPBSA_METHOD = "GBSA"            # "PBSA" or "GBSA" — Sullivan used GBSA
# Sullivan 2019 / Sehit 2024: DSSP 2° structure analysis (computational CD)
DSSP_ANALYSIS = True              # track helix/sheet/coil changes during MD

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
N_WORKERS = 16                    # CPU parallel processes
USE_GPU = True
OUTPUT_DIR = str(_Path(__file__).resolve().parent.parent.parent / "results")
OUTPUT_DIRS = {
    "phase1":   f"{OUTPUT_DIR}/phase1",
    "phase2":   f"{OUTPUT_DIR}/phase2",
    "phase3":   f"{OUTPUT_DIR}/phase3",
    "phase4":   f"{OUTPUT_DIR}/phase4",
    "phase5":   f"{OUTPUT_DIR}/phase5",
    "reports":  f"{OUTPUT_DIR}/reports",
}


def get_output_path(phase_key: str) -> _Path:
    """Return Path for a phase output directory, creating it if needed."""
    p = _Path(OUTPUT_DIRS[phase_key])
    p.mkdir(parents=True, exist_ok=True)
    return p


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
PREPARE_RECEPTOR = (_shutil.which("prepare_receptor4")
                    or _shutil.which("prepare_receptor4.py")
                    or _shutil.which("prepare_receptor")
                    or "prepare_receptor4.py")
PREPARE_LIGAND = (_shutil.which("prepare_ligand4")
                  or _shutil.which("prepare_ligand4.py")
                  or _shutil.which("prepare_ligand")
                  or "prepare_ligand4.py")
GMX_BIN = _shutil.which("gmx") or "gmx"
ACPYPE_BIN = _shutil.which("acpype") or "acpype"
