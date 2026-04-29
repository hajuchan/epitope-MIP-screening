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
        "ecl2_range": (103, 203),     # LEL (UniProt topology)
        "head_residues": (155, 170),  # 16-mer: WEKIPSMSKNRVPDSC
        # Between disulfides C145-C169/C146-C170, variable region
        # Near N172 glycan sequon — key for CD63 selectivity
        "n_glycan_sites": 3,          # N130, N150, N172
        "ccg_position": 145,          # LEL CCG motif
        "disulfide_cys": [145, 146, 169, 170, 177, 191],
        "description": "CD63 LEL — heavy glycosylation, largest head",
    },
    "CD81": {
        "pdb_id": "5TCX",
        "source": "pdb",
        "chain": "A",
        "full_length": 236,
        "ecl2_range": (113, 201),     # LEL (UniProt topology)
        "head_residues": (168, 183),  # 16-mer: SVLKNNLCPSGSNIIS
        # Helix C (165-172) + Helix D (180-186) variable region
        # Kitadokoro 2001 (1IV5), Drummer 2006 (HCV E2 binding)
        "n_glycan_sites": 0,          # No glycosylation
        "ccg_position": 156,          # LEL CCG motif
        "disulfide_cys": [156, 157, 175, 190],
        "description": "CD81 LEL — non-glycosylated, hydrophobic head",
    },
    "CD9": {
        "pdb_id": "6K4J",
        "source": "pdb",
        "chain": "A",
        "full_length": 228,
        "ecl2_range": (112, 195),     # LEL (UniProt topology)
        "head_residues": (156, 171),  # 16-mer: AGGVEQFISDICPKKD
        # Variable region between disulfides C152-C181/C153-C167
        # Kitamura 2020: T175-K179 truncated for crystallization
        "n_glycan_sites": 1,          # 1 N-glycan in LEL
        "ccg_position": 152,          # LEL CCG motif
        "disulfide_cys": [152, 153, 167, 181],
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

# Cross-linkers (auto-selected by monomer chemistry compatibility)
CROSSLINKER_LIBRARY = {
    "TEOS":  {"smiles": "CCO[Si](OCC)(OCC)OCC",
              "name": "Tetraethyl orthosilicate",
              "type": "silane", "functionality": 4,
              "interaction": "Sol-gel Si-O-Si network, slow hydrolysis (~16h)"},
    "TMOS":  {"smiles": "CO[Si](OC)(OC)OC",
              "name": "Tetramethyl orthosilicate",
              "type": "silane", "functionality": 4,
              "interaction": "Sol-gel Si-O-Si network, fast hydrolysis (~6x TEOS)"},
    "MBAAm": {"smiles": "C=CC(=O)NCNC(=O)C=C",
              "name": "N,N'-Methylenebisacrylamide",
              "type": "vinyl", "functionality": 2,
              "interaction": "Free-radical, flexible"},
    "EGDMA": {"smiles": "C=C(C)C(=O)OCCOC(=O)C(=C)C",
              "name": "Ethylene glycol dimethacrylate",
              "type": "vinyl", "functionality": 2,
              "interaction": "Free-radical, semi-rigid"},
    "DVB":   {"smiles": "C=Cc1ccc(C=C)cc1",
              "name": "Divinylbenzene",
              "type": "vinyl", "functionality": 2,
              "interaction": "Free-radical, rigid aromatic"},
    "TRIM":  {"smiles": "C=C(C)C(=O)OCC(CC)(COC(=O)C(=C)C)OC(=O)C(=C)C",
              "name": "Trimethylolpropane trimethacrylate",
              "type": "vinyl", "functionality": 3,
              "interaction": "Free-radical, tri-functional high crosslink"},
}
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

# ── Phase 4: MD Validation — GROMACS ───────────────────────────
MD_PRODUCTION_NS = 350            # pre-polymerization MD (Polania 2024: 350ns for convergence)
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

# ── Phase 6: VIP Cavity Rebinding (Zink 2018) ─────────────────
REBINDING_MD_NS = 50              # rebinding simulation time per snapshot (extended from 20ns)
REBINDING_N_SNAPSHOTS = 10        # top contact frames (n=10 for statistical power)
REBINDING_RMSD_THRESHOLD = 5.0    # Å — template stays in cavity if RMSD < this
REBINDING_RESTRAINT_K = 1000      # kJ/mol/nm² — position restraint on monomer heavy atoms
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
