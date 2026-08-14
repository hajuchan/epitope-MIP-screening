"""BSA whole-protein sol-gel imprinting — TEOS + APTES, 1% Tween20 / DI water.

Bench protocol the user actually runs (3 steps, 실제 실험 프로토콜):
  1. TEOS 를 1% Tween20 / DI water 에 넣고 잘 섞는다.
  2. APTES 를 BSA 용액(1 mg/mL)에 넣고 잘 섞는다.
  3. 용액 2 를 용액 1 에 넣고 magnetic bar 로 2 h 교반.

So: BSA = template (fixed), TEOS + APTES = the ONLY monomers (fixed, no
screening).  The ONE optimisation variable is the TEOS:APTES:BSA ratio.
Solvent is a FIXED CONDITION and is NOT swept.  Selectivity is deferred
(단계적 — 우선 결합만): this round optimises BSA binding/imprinting only.

DELTA FILE.  This is exec'd into config.py's globals() AFTER the whole CD
baseline has already been bound, so it only needs to contain what DIFFERS.
Anything not mentioned here keeps its config.py value — which is precisely why
no `from .config import X` anywhere in the codebase can fail under BSA.

ORDER MATTERS:
    §1 SILANE_SPECIES toggle   (feeds §4)
    §2 OUTPUT_DIR              (feeds _rederive's OUTPUT_DIRS)
    §3-§9 plain overrides
    §10 _rederive()            <-- MUST BE THE LAST STATEMENT
Anything that must override a DERIVED symbol goes AFTER the _rederive() call.
"""
if "__CONFIG_DISPATCH__" not in globals():
    raise ImportError(
        "config_BSA.py is a DELTA file exec'd into config.py's namespace by the "
        "dispatcher at the bottom of config.py. It is NOT importable on its own — "
        "every name it relies on (PROJECT_ROOT, synthesis_pH_window, _rederive, …) "
        "lives in config.py's globals. Select it with:\n"
        "    MIP_EXPERIMENT=BSA python3 -m pipeline.run_pipeline ...\n"
        "    python3 run_BSA.py ...")


# ═══════════════════════════════════════════════════════════════════════════
# §0  IDENTITY
# ═══════════════════════════════════════════════════════════════════════════
EXPERIMENT_LABEL = ("BSA whole-protein sol-gel imprinting "
                    "(TEOS + APTES, 1% Tween20 / DI water)")


# ═══════════════════════════════════════════════════════════════════════════
# §1  OPEN CHEMISTRY TOGGLE — intact ethoxysilane vs hydrolysed silanol
# ═══════════════════════════════════════════════════════════════════════════
# THE CHEMISTRY (unchanged, and still true):  in the real sol-gel reaction
# TEOS/APTES hydrolyse before they ever reach BSA, so over 2 h of stirring in
# water the species that actually contacts the protein is the SILANOL, not the
# ethoxysilane.  On chemistry alone, "hydrolyzed" is the right answer.
#
# *** BUT "hydrolyzed" IS BLOCKED ON THREE ENGINE DEFECTS — SEE §11(i). ***
# The whole point of the hydrolysed form is that Si-OH DONATES hydrogen bonds to
# BSA's polar surface.  That donation is unrepresentable AND unmeasurable in this
# pipeline today, verified on real generated files (/tmp/bsa_md/mp/*_polca/*.itp):
#   1. TYPING   — utils_gromacs.py:1258 `etype = {..., 1: "h1", ...}` types EVERY
#                 hydrogen h1 by ELEMENT, never `ho`.  h1 sigma = 0.2422 nm vs
#                 ho sigma = 0.0538 nm, so the LJ wall alone excludes H-bond
#                 geometry: the H···O minimum moves 0.196 nm -> 0.302 nm.
#   2. GEOMETRY — utils_gromacs.py:1301-1315 keys _std_bond_len as (8,1)/(7,1)/
#                 (6,1) but looks up (min,max) = (1,8)/(1,7)/(1,6).  Every X-H
#                 bond misses and falls back to 0.1500 nm.  Confirmed:
#                 TEOS.itp bonds 1-6/3-7/4-8/5-9 = 0.1500 (O-H should be 0.0960).
#                 Si-O 0.1640 and C-Si 0.1860 are correct — only H is affected.
#   3. COUNTING — phase4_md_validation.py:608 builds HBA(...) with no
#                 `hydrogens_sel`, so MDAnalysis 2.10 guesses with min_charge=0.3.
#                 The largest monomer H charge produced by _generate_silane_itp is
#                 0.2337 (TEOS H) / 0.2313 (APTES silanol H).  ZERO monomer
#                 hydrogens qualify, so HBNMax can never score a silanol or an
#                 amine as a DONOR — for EITHER species.
# Net effect: the intact-vs-hydrolysed comparison this toggle exists to enable
# returns the SAME (zero) donor H-bond signal either way.  It cannot answer the
# question it was added for.  This is NEW exposure — no CD silane contains an
# O-H, so the hydroxyl path had never been exercised before BSA.
#
# DEFAULT IS THEREFORE "intact" — the conservative choice, not the chemically
# preferred one:
#   * "intact" is exactly the SMILES config.py already ships and CD already ran,
#     so it adds no new, unexercised code path.  Intact TEOS has no polar H at
#     all, so defect 2 cannot touch it, and "a hydrophobic sphere" is a fair
#     description of what the force field will actually produce.
#   * "hydrolyzed" would ship a molecule whose defining feature is silently
#     deleted by the engine, while the config's own comments claimed it was the
#     measured advantage.  That is worse than being slightly wrong on chemistry.
# Monomer-as-ACCEPTOR H-bonds (protein donor -> silanol/ethoxy O) are still
# counted for both species; only the DONOR direction is lost.  Contact counts,
# EBN and molecular volume still differ between the two species, so the toggle
# is not useless — it just cannot support the H-bond argument.
#
# TO RE-ENABLE: fix the three defects in §11(i), then set "hydrolyzed" here.
# Until then it is refused at import unless you deliberately opt in with
# MIP_ALLOW_BLOCKED_SPECIES=1 (a stderr warning would be invisible in a
# multi-hour GROMACS log, so this fails loud like every other guard here).
SILANE_SPECIES = "intact"            # "intact" | "hydrolyzed" (BLOCKED, see §11(i))

SILANE_SPECIES_BLOCKED = frozenset({"hydrolyzed"})
if SILANE_SPECIES in SILANE_SPECIES_BLOCKED and not _os.environ.get(  # noqa: F821
        "MIP_ALLOW_BLOCKED_SPECIES"):
    raise ImportError(
        f"SILANE_SPECIES={SILANE_SPECIES!r} is BLOCKED-ON-ENGINE-FIX. The Si-OH "
        f"hydrogen bond that motivates it is deleted three separate ways by the "
        f"current engine (H typed h1 not ho; every X-H bond forced to 0.150 nm by "
        f"a (min,max) key mismatch in utils_gromacs._std_bond_len; HBA guesses "
        f"donors at min_charge=0.3 while no monomer H exceeds 0.234). See §11(i) "
        f"in config_BSA.py for the exact three-part fix. Use SILANE_SPECIES="
        f"'intact', or set MIP_ALLOW_BLOCKED_SPECIES=1 to run it anyway knowing "
        f"the H-bond observable will be identically zero.")

_SILANE_SMILES = {
    # Intact ethoxysilanes.  These are exactly the config.py baseline strings —
    # APTES here is the ALREADY-FIXED Si-C form (was "NCCCO[Si]…", an ether
    # oxygen between propyl and Si, i.e. a tetraalkoxysilane with NO Si-C bond).
    # RDKit-verified: C9H23NO3Si, MW 221.37.  DO NOT revert.
    "intact": {
        "TEOS":  "CCO[Si](OCC)(OCC)OCC",      # C8H20O4Si — always was correct
        "APTES": "NCCC[Si](OCC)(OCC)OCC",     # C9H23NO3Si, MW 221.37
    },
    # Fully hydrolysed forms.
    "hydrolyzed": {
        "TEOS":  "O[Si](O)(O)O",              # silicic acid, H4O4Si, 9 atoms
        "APTES": "NCCC[Si](O)(O)O",           # aminopropylsilanetriol, C3H11NO3Si
    },
}[SILANE_SPECIES]

# RESOLVED AT MODULE LOAD — it CANNOT be flipped after import, because
# _rederive() (§10) builds ALL_MONOMERS from the dicts in §4 and nothing in the
# codebase ever re-derives ALL_MONOMERS again.  The SMILES reach acpype/PolCA
# through ALL_MONOMERS at utils_gromacs.py:145 and :218.
#
# NOT MODELLED (declaration only): the protonated ammonium form
# "[NH3+]CCC[Si](O)(O)O" is the species that actually drives imprinting at
# pH 9-10... except acpype is invoked with "-n 0" (utils_gromacs.py:169) and the
# PolCA path has no formal-charge handling, so a +1 monomer is not expressible
# today.  See §11(a) — it is only usable after the ITP charge fix.


# ═══════════════════════════════════════════════════════════════════════════
# §2  OUTPUT TREE — must be the first path assignment, feeds _rederive()
# ═══════════════════════════════════════════════════════════════════════════
# Species-tagged on purpose: acpype / PolCA topology output is cached on disk
# keyed by monomer NAME ("APTES.itp", "TEOS.itp").  A shared tree would silently
# reuse the other species' .itp/.gro and the SILANE_SPECIES toggle would appear
# to do nothing.  Guard 1 in config.py additionally refuses to start if this
# ever resolves inside the CD results/ tree.
OUTPUT_DIR = str(PROJECT_ROOT / f"results_BSA_{SILANE_SPECIES}")  # noqa: F821


# ═══════════════════════════════════════════════════════════════════════════
# §3  TARGET — whole-protein BSA replaces the three tetraspanins
# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA CONSTRAINT: phase1_epitope_prep.py reads cfg["ecl2_range"] (:115) and
# cfg["head_residues"] (:124) as BARE SUBSCRIPTS — both keys are mandatory or
# Phase 1 KeyErrors.  extract_epitope() (utils_structure.py:90-125) is nothing
# but a chain filter + an inclusive residue-number filter — no loop detection,
# no secondary-structure logic — so (1, 583) emits the complete chain A and
# head_pdb comes out byte-identical to ecl2_pdb.  "ecl2" is a misnomer for BSA
# but NO code inspects its semantics; see PHASE4_TEMPLATE_MODE in §7 for why the
# literal string must stay "ecl2".
#
# 4F5S chosen over 3V03 / AlphaFold:
#   - 4F5S chain A: all 583 mature residues, zero gaps, zero REMARK 465,
#     2.47 Å, defatted (only waters + 1 PGE cryoprotectant nearby).
#   - 3V03: 2.70 Å, missing Asp1-Thr2, carries 6 Ca2+ and 4 acetate.
#   - AlphaFold P02769 is the 607-aa PRECURSOR (signal 1-18 + propeptide 19-24),
#     so every index is +24 off from all PDB/literature numbering, AND
#     source="alphafold" re-enables the check_plddt(ecl2_range) branch at
#     phase1:95-104 which is meaningless for a whole protein.
TARGETS = {
    "BSA": {
        "uniprot_id": "P02769",     # documentation only — never read (source="pdb")
        "source": "pdb",
        "pdb_id": "4F5S",           # Bujacz, Acta Cryst D 2012, 68:1278
        "chain": "A",               # 4F5S has 2 chains; A is the complete one
        "full_length": 583,         # mature BSA; P02769 precursor is 607 aa
        "ecl2_range": (1, 583),     # MANDATORY (phase1:115 bare subscript)
        "head_residues": (1, 583),  # MANDATORY (phase1:124 bare subscript)
        "head_candidates": [],      # empty short-circuits the phase1:125 A1 block
        "n_glycan_sites": 0,        # TRUE, not a bypass — BSA is not N-glycosylated
        "n_glycan_positions": [],
        "description": (
            "Bovine serum albumin, whole-protein template — 583 aa, 66.5 kDa, "
            "pI 4.7, 17 S-S + free Cys34, non-glycosylated. Sol-gel TEOS/APTES "
            "imprinting in 1% Tween20 / DI water."),
        # ccg_position / disulfide_cys DELIBERATELY OMITTED: grepped every .py —
        # neither key is read by ANY module.  gmx pdb2gmx detects all 17 BSA
        # disulfides geometrically from 4F5S (34 SSBOND records) and correctly
        # leaves Cys34 as free SH.  Enumerating them here would buy nothing.
    },
}
# CAVEAT for whoever reads phase1_results.json: pI 4.7 above is the EXPERIMENTAL
# electrophoretic value (the one that matters for the APTES-ammonium ↔ BSA-
# carboxylate argument).  Bio.SeqUtils ProteinAnalysis will report 5.60 for the
# same sequence — a naive Henderson-Hasselbalch estimate.  They disagree; 4.7 is
# the physically meaningful one.


# ═══════════════════════════════════════════════════════════════════════════
# §4  MONOMER LIBRARIES — pinned to exactly {APTES, TEOS}
# ═══════════════════════════════════════════════════════════════════════════
# Built FRESH (never mutated in place) so the CD libraries above are untouched.
#
# CRITICAL TRAP: TEOS exists in BOTH SILANE_MONOMERS (config.py:222) and
# CROSSLINKER_LIBRARY (config.py:323), and ALL_MONOMERS is
# {**SILANE, **VINYL, **CROSSLINKER} — the CROSSLINKER record WINS the merge.
# So the SILANE_SPECIES toggle MUST patch the crosslinker copy too, or
# ALL_MONOMERS["TEOS"]["smiles"] silently keeps the intact ethoxysilane and
# acpype parameterises the wrong molecule.
SILANE_MONOMERS = {
    "APTES": {"smiles": _SILANE_SMILES["APTES"],
              "name": "(3-Aminopropyl)triethoxysilane",
              "polymerization": "silane",
              "interaction": "H-bond donor, electrostatic"},
}
CROSSLINKER_LIBRARY = {
    "TEOS": {"smiles": _SILANE_SMILES["TEOS"],
             "name": "Tetraethyl orthosilicate",
             "type": "silane", "polymerization": "silane", "functionality": 4,
             "interaction": "Sol-gel Si-O-Si network"},
}
VINYL_MONOMERS = {}     # no radical chemistry in this protocol at all
DECOY_MONOMERS = {}     # Phase 2 is skipped; decoy baseline has nothing to run

# NOT overridden on purpose: PRIMARY_CHEM_CLASS, PH_STABILITY, QE_PARAMS,
# POLYMERIZATION_COMPATIBILITY, INITIATOR_MW.  All are keyed lookup tables —
# extra keys are inert, and keeping the full CD tables guarantees no KeyError in
# check_chemistry_diversity() / synthesis_pH_window() / is_polymerization_
# compatible().  Relevant existing rows:
#   PRIMARY_CHEM_CLASS: APTES -> "hbond_donor", TEOS -> "xl_structural"
#   PH_STABILITY:       APTES -> (3.0, 10.0),   TEOS -> (3.0, 10.0)
# QE_PARAMS has no silane rows (silanes have no radical Q-e), so
# reactivity_ratio_product("APTES","TEOS") returns None — callers tolerate it.


# ═══════════════════════════════════════════════════════════════════════════
# §5  PHYSICS — DI water, no added salt; pH 9-10
# ═══════════════════════════════════════════════════════════════════════════
# DECISION (physics), NOT MEASUREMENT.  The CD baseline is 0.15 M PBS.  This
# protocol uses DI WATER with no salt whatsoever.  It matters: the imprinting
# driving force is APTES ammonium ↔ BSA carboxylate (BSA pI 4.7, strongly
# anionic), and 0.15 M NaCl cuts the Debye length to ~0.8 nm and masks exactly
# that interaction.
#
# THIS KEY IS GENUINELY LIVE (rare!).  utils_gromacs.py:348-355 already passes
# BOTH "-neutral" AND "-conc <MD_IONIC_STRENGTH>" to genion, so 0.0 gives
# counterions-only with ZERO engine edits.  Read a second time at :634 for the
# MM-GBSA salt concentration, which therefore stays consistent automatically.
MD_IONIC_STRENGTH = 0.0             # mol/L — DI water, neutralize-only

# DECISION (assumption), NOT MEASUREMENT, and UNCONSUMED-IN-PRACTICE.
# DI water + APTES (a basic aminosilane) gives pH ~9-10, not the 7.4 that the CD
# config assumes.  BSA is strongly negative there.  Setting 9.5 DOCUMENTS the
# condition but does NOT change a single protonation state — see §11(b) for the
# five independent reasons.  Do not read this number as "the MD ran at pH 9.5".
MD_SOLVENT_PH = 9.5


# ═══════════════════════════════════════════════════════════════════════════
# §6  SIMULATION BUDGET — BSA is ~290k atoms, not a 16-mer
# ═══════════════════════════════════════════════════════════════════════════
# Measured from 4F5S chain A: 4,653 heavy atoms, max radius 4.86 nm from the
# centroid, monomer shell r = 5.16-6.16 nm, editconf cubic -d 0.5 -> ~14.2 nm box
# = ~2,860 nm³ -> ~92,300 TIP3P waters -> ~287,000-290,000 atoms.  Box size is set
# by the SHELL RADIUS, not by monomer count, so every ratio point costs the same.
# At ~30-60 ns/day on one GPU, MD_PRODUCTION_NS = 350 would be 6-12 GPU-days PER
# SYSTEM (2-3 months for an 8-point grid).  30 ns per point ≈ 1 week total.
# DECISION: reserve 350 ns for the single winning ratio only.
MD_PRODUCTION_NS = 30               # baseline 350 — unaffordable per ratio point
MD_QUICK_NS = 5                     # baseline 20
MD_MMPBSA_START_NS = 20             # re-derived for a 30 ns run (baseline 30)
MD_MMPBSA_END_NS = 30               # baseline 50 — last 10 ns of the 30 ns run
MD_MMPBSA_INTERVAL = 100            # unchanged; restated here for locality
EPITOPE_MD_TIME_NS = 5              # Phase 1 stability MD; prefer --skip-md anyway
# (Pre-existing baseline inconsistency deliberately NOT normalised here:
#  TEMPERATURE=298.15 vs MD_TEMPERATURE_K=300.0.  Touching either would change
#  CD numbers if the value ever migrated upward.)


# ═══════════════════════════════════════════════════════════════════════════
# §7  PHASE TOGGLES
# ═══════════════════════════════════════════════════════════════════════════
PHASE1_EVALUATE_MULTI_EPITOPE = False  # BSA has no candidate epitope regions to rank
ENSEMBLE_DOCKING = False               # Phase 2 is skipped -> conformers are pure cost

# MAGIC STRING — must stay the literal "ecl2".
# phase4_md_validation.py:285 (protein_restrained = mode=="ecl2"),
# phase5_rebinding.py:254, :370 and :1397 (_tkey = "ecl2_pdb" if mode=="ecl2"
# else "head_pdb") all branch on this exact string.  A descriptive new value like
# "whole_protein" would silently fall through to the legacy head branch and look
# for a head_pdb that does not exist.  Whole-protein-ness is expressed by
# TARGETS["BSA"]["ecl2_range"] = (1, 583), not by this string.
#
# *** THIS FREEZES THE ENTIRE PROTEIN — SEE §11(j). ***  phase4:285 sets
# protein_restrained = (mode == "ecl2") and utils_gromacs.py:806 turns that into
# `define = -DPOSRES`.  pdb2gmx's posre.itp restrains ALL HEAVY ATOMS, not the
# backbone that phase4's own comment claims: measured on 4F5S chain A it is 4,653
# entries at 1000/1000/1000 kJ/mol/nm².  For a 16-residue ECL2 loop that is a
# reasonable "solid-phase immobilisation" mimic; for a 583-residue template it
# means every Asp/Glu carboxylate is nailed in place, i.e. the induced-fit
# reorientation that §5 names as the imprinting driving force is suppressed by
# construction.  BSA Phase 4 measures monomer diffusion against a RIGID surface.
PHASE4_TEMPLATE_MODE = "ecl2"
PHASE4_PROTEIN_RESTRAINT_K = 1000   # kJ/mol/nm² (DEAD KEY — never imported; the
                                    # -DPOSRES value is set inside phase4)

# Solvent is a FIXED CONDITION (1% Tween20 / DI water) — the user explicitly
# rejected an ethanol co-solvent suggestion; Tween20 is what solubilises TEOS.
# AND the sweep is a STUB anyway: _run_prepolymerization_md takes no solvent
# argument and utils_gromacs.py:331 hardcodes "-cs spc216.gro", so all four CD
# "solvents" ran in plain water.  Off, and reduced to the single real entry.
PHASE4_SOLVENT_SWEEP = False
PHASE4_SOLVENTS = {                 # NOTE: this is a DICT keyed by solvent name
    "water": {"template": "spc216.gro", "dielectric": 78.5, "use_for": ["sol-gel"]},
}

# ── UNCONSUMED / STUB WARNING ──────────────────────────────────────────────
# PHASE4_RATIO_SWEEP stays False.  run_phase4_ratio_sweep()
# (phase4_md_validation.py:972) reads PHASE4_RATIO_PRESETS, computes a `copies`
# dict, then NEVER PASSES IT to _run_prepolymerization_md() — that function has
# no copies parameter and always computes uniform copies internally.  Worse, for
# a one-functional-monomer system it truncates every preset to a single element
# (line 990: preset = list(preset[:n_func])), so all presets run the IDENTICAL
# 50/50 box.  Turning this True would produce N indistinguishable runs labelled
# as N different ratios.  The real declaration is BSA_RATIO_GRID in §9.
PHASE4_RATIO_SWEEP = False
# PHASE4_RATIO_PRESETS is deliberately NOT overridden — it must still exist for
# the import at phase4:972, and overloading it with (n_TEOS, n_APTES) pairs would
# just get truncated to one element by that same preset[:n_func] line.

# n_total monomers placed in the shell.  phase4:230-236 computes
#   copies_per_functional = max(1, total // (n_functional + 1))
#   copies_crosslinker    = total - copies_per_functional * n_functional
# With APTES the only functional monomer and TEOS a crosslinker, n_functional==1,
# so this ALWAYS splits exactly 50/50 (n_APTES = 30, n_TEOS = 30).  That is the
# degeneracy, and it bites at the copy-number level, not just at the EBN level.
# 60 monomers in the ~2,860 nm³ box ≈ 35 mM total silane — squarely inside the
# real sol-gel range (typical preparations are 10-100 mM).
EPITOPE_MONOMER_MOLAR_RATIO = 60

# Phase 2/3 diversity gates: DISABLE, do not retune.
# PRIMARY_CHEM_CLASS puts APTES in "hbond_donor" and TEOS in "xl_structural",
# and crosslinkers are excluded from the diversity count — so the BSA system has
# exactly ONE recognition class.  MMSD_MIN_CHEMISTRY_CLASSES=2 would reject the
# user's own bench recipe.  MMSD_MIN_COMBO_SIZE=2 likewise: there is only one
# functional monomer.  MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY stays True —
# both species are "silane", so is_polymerization_compatible passes cleanly.
MMSD_SELECTIVITY_AWARE = False      # single target — nothing to cross-dock against
MMSD_REQUIRE_CHEMISTRY_DIVERSITY = False
MMSD_MIN_CHEMISTRY_CLASSES = 1
MMSD_MIN_COMBO_SIZE = 1
# MMSD_TOP_PC not overridden — baseline is already 1.
# SELECTIVITY_WEIGHT / SELECTIVITY_DDG_THRESHOLD not overridden — kept
# present-but-inert as the clearly-marked switch-on point (see §9).

DUAL_EPITOPE_CD63 = False           # CD63-only glycan layer; must not fire for BSA

# Phase 5 is DEFERRED (§9).  These merely make it expensive rather than absurd
# if someone switches it on: 10 snapshots × 50 ns × ~290k atoms is 50-100
# GPU-days.  REBINDING_RMSD_THRESHOLD 5.0 Å was calibrated on a 16-mer / ECL2
# loop and is not meaningful for a 66 kDa template.
REBINDING_N_SNAPSHOTS = 3
REBINDING_MD_NS = 30
REBINDING_RMSD_THRESHOLD = 8.0      # Å — DECISION (scaled for a 583-mer), unvalidated

INITIATOR_MOL_PERCENT = 0.0         # sol-gel: no radical initiator exists
TARGET_KD_NM = 1000                 # DECISION — restated for a BSA-imprinted silica;
TARGET_IF_MIN = 2                   # the 50 nM / IF>3 goals were tetraspanin-era


# ═══════════════════════════════════════════════════════════════════════════
# §8  pH RECOMMENDATION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════
# synthesis_pH_window({"TEOS","APTES"}) intersects (3.0,10.0) with (3.0,10.0) and,
# with no boronate present, returns the plain MIDPOINT 6.5.  phase6_recipe.py:261
# prints that straight into the recipe — and 6.5 flatly contradicts the actual
# pH ~9-10 of DI water + a basic aminosilane.  Wrapping the baseline function
# here fixes the printed recipe with ZERO edits to config.py or phase6.
# (This is a DECISION: 9.5 is the assumed protocol pH, not a measured one.)
_base_synthesis_pH_window = synthesis_pH_window  # noqa: F821


def synthesis_pH_window(monomers, _base=_base_synthesis_pH_window):
    """BSA/DI-water override: keep the stability window, fix the recommendation."""
    lo, hi, _mid = _base(monomers)
    if lo is None:
        return lo, hi, _mid
    return lo, hi, 9.5


# ═══════════════════════════════════════════════════════════════════════════
# §9  NEW BSA-ONLY KEYS — EVERY ONE OF THESE IS CURRENTLY UNCONSUMED
# ═══════════════════════════════════════════════════════════════════════════
# 아래 키들은 전부 "선언"이지 "동작"이 아니다.  No module imports any of them
# today.  They exist so the ratio-sweep engine work (and the later selectivity
# round) has a single, reviewed place to read from instead of inventing new
# names — and so nobody mistakes a declaration for a running sweep.
# UNCONSUMED_KEYS at the bottom of this section is the machine-readable list;
# the startup banner prints its length on every run.

MONOMER_SET_PINNED = True           # UNCONSUMED — "monomer set is fixed, skip the search"
PINNED_PC = {                       # UNCONSUMED — run_phase4 reads top_pcs from
    "pc_id": "BSA_TEOS_APTES",      #   phase3_mmsd_results.json ON DISK (phase4:57-59),
    "monomers": ["APTES", "TEOS"],  #   so this must be hand-written as a stub JSON;
    "crosslinker": "TEOS",          #   see §11(e).
}
RUN_PHASES = {1: True, 2: False, 3: False, 4: True, 5: False, 6: False}
# ^ UNCONSUMED — run_pipeline.py:341-344 builds the phase list purely from the
#   --phase CLI argument; there is no config gate.  Run `--phase 1` then
#   `--phase 4` by hand.

# ── THE RATIO GRID ─────────────────────────────────────────────────────────
# *** PROVISIONAL — LITERATURE-CENTRED, NOT PROTOCOL-CENTRED. ***
# 사용자가 아직 실제 TEOS / APTES 사용량을 주지 않았다.  Until BSA_PROTOCOL_MEASURED
# below is filled in, this grid is derived from the protein-imprinted sol-gel
# literature, NOT from the user's bench sheet.
#
# Literature basis: Shiomi/Matsui/Mizukami/Nakanishi, Biomaterials 2005, 26, 5564
# (haemoglobin surface-imprinted silica via aminopropylsilane + TEOS) is the
# canonical protocol and the origin of the aminosilane-minority convention.
# Across the BSA/HSA-imprinted Stöber-type silica literature TEOS:APTES clusters
# between 10:1 and 2:1, i.e. x_APTES ≈ 0.09-0.33 — four of the eight points below
# sit inside that window.  The upper bound is mechanistic, not arbitrary: the
# propylamine of APTES is itself a condensation base catalyst, so above ~25-30
# mol% the sol gels prematurely into a poorly-condensed amine-rich network with
# degraded cavity fidelity.
#
# Entries are (n_TEOS, n_APTES) per ONE BSA molecule, at fixed n_total = 60.
# x_APTES = n_APTES / (n_TEOS + n_APTES) is the primary axis.
BSA_RATIO_GRID = [                  # UNCONSUMED (engine is a stub — see §7)
    (60,  0),   # x=0.00  TEOS-only control: no ammonium at all. Isolates the
                #         non-electrostatic contribution (silanol H-bond +
                #         hydrophobic patch contact). Chemistry-side NIP baseline.
    (57,  3),   # x=0.05  19:1
    (54,  6),   # x=0.10  10:1  — low end of the literature window
    (48, 12),   # x=0.20   4:1
    (40, 20),   # x=0.33   2:1  — high end of the literature window
    (30, 30),   # x=0.50   1:1  — the ONLY point today's engine can actually
                #         build (phase4:230-236 with n_functional==1), so it
                #         doubles as the regression anchor for any engine fix.
    (20, 40),   # x=0.67   1:2  — deliberately PAST the literature window so the
                #         optimum is bracketed rather than sitting at a grid edge.
    ( 0, 60),   # x=1.00  APTES-only control: electrostatic ceiling. If the
                #         contact observable rises monotonically to here, it is
                #         measuring amine COUNT, not imprinting.
]
BSA_RATIO_GRID_PROVISIONAL = True   # flip to False after re-centring on real volumes
BSA_RATIO_TOTAL_MONOMERS = 60       # n_total for the primary x-sweep (~35 mM)
BSA_LOADING_SWEEP = [30, 60, 100]   # secondary n_total axis AT THE WINNING x
                                    # ≈ 17 / 35 / 58 mM — a local-concentration
                                    # control that checks the answer is not a
                                    # crowding artefact.
BSA_RATIO_SWEEP_MD_NS = 30          # per sweep point; NOT MD_PRODUCTION_NS.
                                    # Matches the existing CROSSLINKER_SWEEP_MD_NS
                                    # precedent. Reserve x=0.10 and x=0.20 for a
                                    # 350 ns re-run if the sweep comes out flat
                                    # (a flat sweep is itself a result).
BSA_MAX_MONOMERS_IN_SHELL = 100     # HARD GUARD. The monomer shell is only
    # (4/3)π(6.16³ − 5.16³) = 403 nm³ and _include_monomers_in_topology tries 500
    # random placements per molecule with a 1.0 nm exclusion. Past ~100 it starts
    # hitting the else-branch fallback, which puts molecule i at r_outer + 0.3·i nm
    # — molecule #120 lands 42 nm out and editconf then builds an ~85 nm box
    # (>20 M atoms). Do not raise this without fixing the placer.

# ── THE RE-CENTRING HOOK ───────────────────────────────────────────────────
# *** ALL None UNTIL THE USER SUPPLIES REAL NUMBERS. ***  This is the one place
# to drop the bench measurements in.  Conversion is mechanical:
#     n_TEOS_mol  = teos_uL  * 1e-3 * 0.933 / 208.33   # ρ 0.933 g/mL, M 208.33
#     n_APTES_mol = aptes_uL * 1e-3 * 0.946 / 221.37   # ρ 0.946 g/mL, M 221.37
#     x_APTES_measured = n_APTES / (n_TEOS + n_APTES)
# Then RE-CENTRE BSA_RATIO_GRID so x_measured is a grid POINT sitting mid-sweep
# with at least half a decade of bracket either side — an optimum found at a grid
# BOUNDARY is not an optimum.  Also set n_total from the measured molarity
# (n_total = C_silane_mM / 0.58, since one molecule ≈ 0.58 mM in this box)
# instead of defaulting to 60.
BSA_PROTOCOL_MEASURED = {           # UNCONSUMED — documentation + re-centring input
    "teos_uL":         None,        # <-- FILL IN
    "aptes_uL":        None,        # <-- FILL IN
    "bsa_mg_per_mL":   1.0,         # given in the protocol
    "bsa_solution_mL": None,        # <-- FILL IN
    "tween20_pct_w_v": 1.0,         # given in the protocol (1% Tween20)
    "di_water_mL":     None,        # <-- FILL IN
    "stir_hours":      2.0,         # given in the protocol (2 h magnetic stir)
}
# NOTE on the third term of "TEOS:APTES:BSA": it is pinned at 1 BY CONSTRUCTION.
# One BSA sits in the box, so the box stoichiometry is 60:1 while the flask is
# ~700-6600:1 (1 mg/mL BSA = 15.1 µM vs 10-100 mM silane), and the box's BSA
# concentration is ~570 µM, ~38× the flask.  Neither is fixable in a
# single-protein simulation.  What DOES map faithfully is (i) the dimensionless
# TEOS:APTES ratio and (ii) the silane concentration.  Do not try to encode the
# BSA term as a sweep variable.

SOLVENT_CONDITION = {               # UNCONSUMED — documents the fixed condition
    "medium": "1% (w/v) Tween 20 in DI water",
    "swept": False,
    "md_representation": ("TIP3P water only — polysorbate micelle not modelled "
                          "(all-atom Tween20 judged too expensive)"),
}
MD_EXPLICIT_SURFACTANT = False      # UNCONSUMED — records the Tween20 omission as
                                    # a DECISION, not an oversight

# ── DEFERRED SELECTIVITY: the clearly-marked switch-on point ───────────────
# User decision: 단계적 — 우선 결합만.  This round optimises BSA binding only.
# TO SWITCH ON LATER:
#   1. SELECTIVITY_DEFERRED = False
#   2. populate BSA_COMPETITOR_TARGETS with same-schema TARGETS entries
#      (candidates: lysozyme 1DPX, ovalbumin 1OVA, human serum albumin 1AO6)
#   3. MMSD_SELECTIVITY_AWARE = True
#   4. drop REBINDING_N_SNAPSHOTS further — Phase 5 at ~290k atoms is brutal.
# SELECTIVITY_WEIGHT and SELECTIVITY_DDG_THRESHOLD keep their CD baseline values,
# present-but-inert, so step 3 is genuinely a one-line change.
SELECTIVITY_DEFERRED = True         # UNCONSUMED
BSA_COMPETITOR_TARGETS = []         # UNCONSUMED — empty BY DECISION, not by omission

# Machine-readable honesty list.  Delete entries as the engine work lands.
# config.py's startup banner prints len(UNCONSUMED_KEYS) on every single run so
# that "declared" can never quietly be mistaken for "running".
UNCONSUMED_KEYS = frozenset({
    "MONOMER_SET_PINNED", "PINNED_PC", "RUN_PHASES", "BSA_RATIO_GRID",
    "BSA_RATIO_GRID_PROVISIONAL", "BSA_RATIO_TOTAL_MONOMERS", "BSA_LOADING_SWEEP",
    "BSA_RATIO_SWEEP_MD_NS", "BSA_MAX_MONOMERS_IN_SHELL", "BSA_PROTOCOL_MEASURED",
    "SOLVENT_CONDITION", "MD_EXPLICIT_SURFACTANT", "SELECTIVITY_DEFERRED",
    "BSA_COMPETITOR_TARGETS", "MD_SOLVENT_PH", "PHASE4_PROTEIN_RESTRAINT_K",
})


# ═══════════════════════════════════════════════════════════════════════════
# §10  RE-DERIVE — MUST BE THE LAST STATEMENT IN THIS FILE
# ═══════════════════════════════════════════════════════════════════════════
# Rebuilds, from the §2/§4 overrides above:
#   CROSSLINKERS        = {"TEOS"}
#   ALL_MONOMERS        = {"APTES", "TEOS"}   (TEOS = the crosslinker record)
#   FUNCTIONAL_MONOMERS = {"APTES"}           (TEOS excluded by definition)
#   OUTPUT_DIRS         = results_BSA_<species>/phase1..6, reports
#   USE_AUTODOCK_GPU
# No post-hoc FUNCTIONAL_MONOMERS override is needed — pinning the libraries
# makes it fall out structurally.  That single-functional-monomer condition is
# also exactly why the EBN-derived optimal_ratio is degenerate: it is
# round(EBN/min(EBN)) over functional monomers, which for one monomer is always
# 1, and crosslinker = sum = 1, so Phase 6 would print a constant APTES:TEOS =
# 1:1 no matter what the MD says.  See §11(e).
_rederive()  # noqa: F821


# ═══════════════════════════════════════════════════════════════════════════
# §11  HONESTY NOTES — read these before trusting any BSA output
# ═══════════════════════════════════════════════════════════════════════════
# (a) MD_IONIC_STRENGTH = 0.0 IS NECESSARY BUT NOT SUFFICIENT — AND IT MAKES THE
#     FABRICATED-CHARGE ARTEFACT WORSE, NOT NEUTRAL TO IT.
#     _generate_silane_itp (utils_gromacs.py:1241-1250) computes Gasteiger
#     charges on an Si→S proxy molecule and then OVERWRITES the Si charge with a
#     hardcoded +0.9 WITHOUT renormalising, so every silane ITP carries a
#     non-integer net charge: intact TEOS +0.676, hydrolysed TEOS +0.687, intact
#     APTES +0.811, hydrolysed APTES +0.819 (all should be 0.000).  Verified
#     against a real production file (results/phase4/CD63/NSGA_31_TTMS/
#     monomer_params/TMOS_polca/TMOS.itp = +0.677) and against freshly generated
#     BSA-path files (/tmp/bsa_md/mp/TEOS_polca/TEOS.itp = +0.6868,
#     mp/APTES_polca/APTES.itp = +0.8190).
#
#     MEASURED, not estimated, on a real 30 APTES + 30 TEOS + BSA box:
#       silane charge   30(+0.819) + 30(+0.687) = +45.17 e
#       BSA chain A     -16 e (gmx pdb2gmx qtot; ProteinAnalysis says -15.97)
#       grompp          "System has non-zero total charge: 29.174"
#       genion          "Will try to add 0 NA ions and 29 CL ions"
#       box volume      2602.29 nm³  ->  29 Cl⁻ = 18.5 mM
#     So the correct numbers are 29 Cl⁻ / 18.5 mM.  (An earlier draft of this
#     note said "~45 Cl⁻, ~25-30 mM"; that forgot BSA's own -16 and is WRONG.)
#
#     THE SHARPER POINT: setting MD_IONIC_STRENGTH = 0.0 AMPLIFIES this defect.
#     Debye length = 0.304/sqrt(I) nm:  150 mM -> 0.78 nm, 18.5 mM -> 2.24 nm.
#     The +45.17 e of fabricated silane charge therefore acts over ~3× the range
#     in this DI-water condition than it does under the CD/PBS baseline — and it
#     acts on a pI-4.7 protein, manufacturing part of the exact electrostatic
#     signal the experiment exists to measure.
#     >>> Charge renormalisation in _generate_silane_itp is a PREREQUISITE
#     >>> phase-code fix, and it must land BEFORE (not after) any production BSA
#     >>> run, precisely because the DI-water setting magnifies the error.
#
# (b) MD_SOLVENT_PH = 9.5 CHANGES NO PROTONATION STATE.  Five reasons, any one
#     sufficient: (i) propka is not installed in the MIPscreen env, so
#     assign_protonation_states takes the ImportError branch and reduces to
#     shutil.copy2; (ii) even with propka it only LOGS predicted HIP/CYM changes
#     and never renames a residue; (iii) its output filename is hardcoded to
#     stem + "_pH74.pdb" regardless of the ph argument; (iv) pdbfixer's
#     addMissingHydrogens is hardcoded to 7.4 (utils_gromacs.py:882); (v)
#     pdb2gmx runs with no -asp/-glu/-lys/-his flags.  BSA WILL be simulated
#     with pH-7 protonation whatever this key says.
#
# (c) DEAD KEYS — deliberately NOT set here, because setting them changes
#     nothing and would imply otherwise: MD_BOX_TYPE, MD_BOX_DISTANCE (box is
#     hardcoded cubic / 0.5 nm at utils_gromacs.py:313), MD_FF_PROTEIN,
#     MD_FF_MONOMER, MD_WATER_MODEL (setup_protein_topology hardcodes
#     amber99sb-ildn / tip3p as default args at utils_gromacs.py:283-284 and all
#     call sites pass only two positional args).  Happily the desired BSA values
#     — TIP3P water — are the baseline values anyway.
#
# (d) verify_phase5's cross-target loop is `for o in TARGETS if o != t`.  With a
#     single target it is EMPTY, so the selectivity check produces nothing and
#     reports a VACUOUS PASS.  Read that as SKIP, never as success.
#
# (e) PHASES 2/3 CANNOT ACTUALLY BE SKIPPED BY CONFIG.  run_pipeline.py:341-344
#     builds the phase list from --phase only, and run_phase4 loads top_pcs from
#     phase3_mmsd_results.json on disk (phase4:57-59, :112).  Practical recipe:
#         MIP_EXPERIMENT=BSA python3 -m pipeline.run_pipeline --target BSA \
#             --phase 1 --skip-md
#         # then hand-write <OUTPUT_DIR>/phase3/phase3_mmsd_results.json as
#         #   {"BSA": {"top_pcs": [PINNED_PC]}}
#         MIP_EXPERIMENT=BSA python3 -m pipeline.run_pipeline --target BSA --phase 4
#
# (f) make_presentation_figures.py / make_process_figures.py are CD-PINNED and
#     OUT OF SCOPE.  They import PRIMARY_CHEM_CLASS from config but read
#     hardcoded "results/phase3/..." literals after os.chdir(ROOT) and write to
#     results/presentation.  Under MIP_EXPERIMENT=BSA they would apply BSA's
#     chemistry table to CD's JSONs and OVERWRITE CD's figures.  NEVER run them
#     for BSA.
#
# (g) Phase 2 could not run for BSA even if it were wanted: compute_grid_size on
#     a 583-residue receptor returns (290, 208, 244) grid points per axis, an
#     order of magnitude past AutoGrid4's per-axis limit.
#
# (h) Phase 6 output is FOR-INTERNAL-USE-ONLY this round.  Beyond the degenerate
#     optimal_ratio in §10, _get_protocol("sol-gel", target) emits a canned
#     SiO2-nanoparticle / glutaraldehyde / epitope-immobilisation protocol that
#     is not the user's three-step Tween20 procedure at all.
#
# (i) NO MONOMER H-BOND DONOR IS REPRESENTABLE OR COUNTABLE.  This is why §1
#     defaults to "intact" and refuses "hydrolyzed".  Three independent defects,
#     each sufficient on its own, all verified on real generated files:
#
#       1. TYPING (utils_gromacs.py:1258)
#            etype = {6: "c3", 1: "h1", 8: "oh", 7: "n3", 14: si_type, ...}
#          keys on ELEMENT, so every H is GAFF2 `h1` regardless of neighbour.
#          Observed in mp/TEOS_polca/TEOS.itp: H6..H9 are all `h1`.
#          gaff2.dat MOD4:  ho 0.3019/0.0047 -> sigma 0.0538 nm
#                           h1 1.3593/0.0208 -> sigma 0.2422 nm  (what is emitted)
#          Against a protein carboxylate O (sigma 0.29599) the H···O LJ minimum
#          moves from 0.196 nm to 0.302 nm — the LJ wall alone forbids H-bond
#          geometry.  (The same table types ETHER oxygens `oh` instead of `os`;
#          milder, and it affects the intact species, so it is noted not fixed.)
#
#       2. GEOMETRY (utils_gromacs.py:1301-1315)
#          _std_bond_len is keyed (8,1)/(7,1)/(6,1)/(16,1) but the lookup builds
#          key = (min(e1,e2), max(e1,e2)) = (1,8)/(1,7)/(1,6)/(1,16).  EVERY X-H
#          bond misses and takes the 0.1500 nm default.  Observed:
#            TEOS.itp  1-6, 3-7, 4-8, 5-9        = 0.1500  (O-H, want 0.0960)
#            APTES.itp 1-9, 1-10                 = 0.1500  (N-H, want 0.1010)
#                      2-11 .. 4-16              = 0.1500  (C-H, want 0.1090)
#                      6-17, 7-18, 8-19          = 0.1500  (O-H, want 0.0960)
#          Si-O (0.1640) and C-Si (0.1860) are CORRECT — the bug is X-H only.
#          With O-H = 0.150 nm and H···O >= 0.30 nm from defect 1, a linear
#          donor···acceptor heavy-atom distance is ~0.45 nm, well past the
#          d_a_cutoff = 3.5 Å used at phase4_md_validation.py:609.
#
#       3. COUNTING (phase4_md_validation.py:608-611)
#          HBA(u_hb, donors_sel=..., acceptors_sel=..., d_a_cutoff=3.5,
#              d_h_a_angle_cutoff=150, update_selections=False)
#          passes NO hydrogens_sel, so MDAnalysis 2.10 falls back to
#          guess_hydrogens(select='all', max_mass=1.1, min_charge=0.3,
#          min_mass=0.9).  Monomer H charges from _generate_silane_itp are
#          {0.029, 0.0426, 0.0433, 0.1184, 0.2313, 0.2337} — the maximum, 0.2337,
#          is below min_charge=0.3.  On the real box: 0 of 450 monomer hydrogens
#          are guessed (462 protein H are).  HBNMax therefore cannot score a
#          silanol or an amine as a DONOR, for EITHER species.
#          (Monomer-as-ACCEPTOR still works: protein H -> monomer O.)
#
#     THE THREE-PART FIX, if/when the engine is opened up:
#       (1) in _generate_silane_itp, type H by its bonded neighbour instead of by
#           element — O -> "ho" (sigma 0.05379, eps 0.01966), N -> "hn"
#           (0.11065, 0.04184), C -> "h1".  Both types are ALREADY present in the
#           _gaff2_lj table at utils_gromacs.py:1272-1281, so only the lookup
#           changes; add them to `used_types` so the [ atomtypes ] block emits them.
#       (2) re-key _std_bond_len as (1,6)/(1,7)/(1,8)/(1,16) (or normalise the key
#           at build time) so X-H bonds stop defaulting to 0.1500 nm.
#       (3) pass an explicit hydrogens_sel to HBA at phase4:609 (e.g.
#           "name H* and bonded (name N* O* S*)") instead of relying on the
#           min_charge=0.3 guess.
#     Do (1)+(2) together with the §11(a) charge renormalisation — they touch the
#     same function and all three must be re-validated against a reference ITP.
#
# (j) BSA PHASE 4 RUNS A RIGID TEMPLATE.  See the block above PHASE4_TEMPLATE_MODE
#     in §7: -DPOSRES + pdb2gmx posre.itp restrains all 4,653 BSA heavy atoms at
#     1000 kJ/mol/nm² for the whole production run, side chains included.  Read
#     contact counts, EBN and HBNMax as "monomer behaviour against a frozen BSA
#     surface", never as induced fit.  A backbone-only restraint would need a
#     `gmx genrestr` on a C-alpha/backbone index group and a phase-code knob;
#     that is engine work, deliberately not attempted from config.
