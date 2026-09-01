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
# unbuffered DI water at pH ~9.5 the species that actually contacts the protein
# is the SILANOL, not the ethoxysilane.  On chemistry alone, "hydrolyzed" is the
# right answer, and it always was.
#
# *** THE THREE ENGINE DEFECTS THAT BLOCKED IT ARE GONE.  DEFAULT IS NOW
#     "hydrolyzed".  *** (2026-08 audit follow-up — see §11(i) for the old text.)
# Re-measured on the topologies the PRODUCTION path actually generates
# (parameterize_monomer -> acpype/GAFF2 + PolCA Si override), for both species:
#   1. TYPING   — WAS: every hydrogen typed h1 by element (sigma 0.2422 nm), so
#                 the LJ wall alone excluded H-bond geometry.
#                 NOW: hydrogens are typed by bonded neighbour.  Hydrolysed TEOS
#                 emits four `ho`; the aminopropylsilanetriol emits three `ho`
#                 plus two `hn`, with h1/hc correct on the propyl chain.  ZERO
#                 h1 hydrogens sit on an O or an N in any of the four species.
#   2. GEOMETRY — WAS: a (min,max) key mismatch in _std_bond_len missed every
#                 X-H pair and fell back to 0.1500 nm.
#                 NOW: O-H = 0.09725 nm, N-H = 0.10192 nm, C-H = 0.1096 nm, and
#                 0 of 4 (TEOS) / 0 of 11 (APTES) X-H bonds are left at 0.1500.
#                 Both silanes still renormalise to qtot 0.00000 and both carry
#                 [dihedrals] and [pairs].
#   3. COUNTING — WAS: HBA(...) built with no `hydrogens_sel`, so MDAnalysis
#                 guessed donors at min_charge=0.3 while no monomer H exceeded
#                 0.2337.  The monomer-as-DONOR direction was identically zero
#                 for BOTH species — the one observable this toggle exists to
#                 measure.
#                 NOW: phase4_md_validation passes `hydrogens_sel` explicitly and
#                 restricts donors AND acceptors to N/O/S by mass, so the guess
#                 is not consulted at all.  (Two further defects were found in
#                 the same call and fixed with it: acceptors_sel was passed
#                 verbatim, which made every carbon and hydrogen in the selection
#                 an "acceptor" with a bias that scaled with monomer atom count;
#                 and the guessed selection is a (resname, name) string while
#                 acpype names every monomer UNL, so in a two-monomer box —
#                 i.e. every box a ratio sweep builds — a qualifying hydrogen of
#                 species A admitted the same-named hydrogen of species B.)
#                 The charges are now comfortable rather than marginal: silanol
#                 H = +0.4500 (TEOS) / +0.4407 (APTES), amine H = +0.3493.
#
# WHY THIS MATTERS FOR THE RATIO ANSWER, not just for tidiness:  intact TEOS has
# NO polar hydrogen at all (its largest H charge is +0.0415) — it is a bulky
# hydrophobic sphere, 33 atoms, ~0.96 nm span.  Si(OH)4 is 9 atoms, ~0.44 nm,
# and donates four hydrogen bonds.  The question being asked is "how much APTES
# do I need to line the cavity against how much TEOS do I need to build the
# wall", and the answer plausibly MOVES when TEOS stops being inert and starts
# competing with APTES for BSA's polar surface.  Running the sweep on the intact
# species would answer it for a molecule that is not in the beaker at t = 2 h.
#
# WHAT IS STILL NOT MODELLED, and applies to BOTH species equally, so it is not
# an argument either way:  the PROTONATED ammonium form "[NH3+]CCC[Si](O)(O)O",
# which is the majority species at pH ~9.5 (propylamine pKa ~10) and is the
# actual electrostatic driving force.  acpype is invoked with a hardcoded
# "-n 0" (utils_gromacs.py), so a +1 monomer fails there and degrades to the
# hand-built path.  APTES is therefore modelled as a NEUTRAL amine in this
# round, and the DI-water electrostatic term is under-represented.
#
# THE GUARD ITSELF IS KEPT, with an EMPTY blocked set.  It is the mechanism that
# stops a chemically-motivated toggle from shipping ahead of the engine that has
# to represent it, and the next species to be added (the ammonium form above)
# will need it.  Adding a name to SILANE_SPECIES_BLOCKED re-arms it.
SILANE_SPECIES = "hydrolyzed"        # "intact" | "hydrolyzed"

# EMPTY BY MEASUREMENT, not by omission.  Every reason the block existed for was
# re-tested and is gone (see above).  Re-arm by adding a species name here.
SILANE_SPECIES_BLOCKED = frozenset()
if SILANE_SPECIES in SILANE_SPECIES_BLOCKED and not _os.environ.get(  # noqa: F821
        "MIP_ALLOW_BLOCKED_SPECIES"):
    raise ImportError(
        f"SILANE_SPECIES={SILANE_SPECIES!r} is BLOCKED-ON-ENGINE-FIX: it is "
        f"listed in SILANE_SPECIES_BLOCKED, which means the engine cannot yet "
        f"represent or measure the feature that motivates it. See §1 and §11(i) "
        f"in config_BSA.py for what has to land first. Set MIP_ALLOW_BLOCKED_"
        f"SPECIES=1 to run it anyway, knowing the observable it exists to "
        f"produce is not trustworthy.")

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

# THE AMMONIUM FORM — NOW BUILDABLE AND NOW BUILT.  # BEHAVIOUR CHANGE 2026-08
# It used to be a declaration-only note here ("a +1 monomer is not expressible
# today") because _run_acpype hardcoded "-n 0". That is gone: the net charge is
# taken from the SMILES' RDKit formal charge, and the ammonium silanetriol was
# generated on the PRODUCTION path (smiles_to_mol2 -> parameterize_monomer ->
# acpype/GAFF2 -> PolCA Si override) and audited:
#     APTESH  qtot=+1.00000  atoms=20  X-H@0.1500=0/12  O-H 0.09725  N-H 0.10271
#             polar H types {hn, ho}   max polar H +0.4611   dihedrals+pairs: yes
# compared with the neutral amine from the same command:
#     APTESn  qtot=-0.00000  atoms=19  X-H@0.1500=0/11  O-H 0.09725  N-H 0.10192
# MONOMER_PROTONATION_SPLIT (below) is what actually puts it in the box.
_APTES_PROTONATED_SMILES = {
    "intact":     "[NH3+]CCC[Si](OCC)(OCC)OCC",
    "hydrolyzed": "[NH3+]CCC[Si](O)(O)O",
}[SILANE_SPECIES]

# NOTE ON THE OLD "NOT MODELLED" PARAGRAPH THAT USED TO SIT BELOW: it said the
# ammonium form "is not expressible today" because acpype was invoked with
# "-n 0" and the PolCA path had no formal-charge handling. Both are fixed; see
# _APTES_PROTONATED_SMILES above and MONOMER_PROTONATION_SPLIT in §4.
#
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
# The pH-9.5 MAJORITY form of the SAME bottle. It is deliberately NOT in
# SILANE_MONOMERS: that would put it in FUNCTIONAL_MONOMERS and Phase 3 would
# enumerate it as a third reagent to screen, which it is not. It is injected
# into ALL_MONOMERS AFTER _rederive() (see §10) so parameterize_monomer can
# resolve its SMILES, and _merge_protonation_forms folds it back onto APTES
# before any synthesis ratio is published.
_APTES_PROTONATED_RECORD = {
    "smiles": _APTES_PROTONATED_SMILES,   # noqa: F821
    "name": "(3-Aminopropyl)silanetriol, ammonium form",
    "polymerization": "silane",
    "interaction": "cation, electrostatic (BSA carboxylate)",
    "protonated_form_of": "APTES",
}
# ── pH SPECIATION OF THE MONOMER POPULATION ───────────────────────────────
# CONSUMED by phase4_md_validation._protonation_split_spec() ->
# _apply_protonation_split(), which is called from _composition_from_copies, so
# it applies to EVERY grid point.  n copies of APTES become
# round(n * f) APTESH + the rest APTES, with f from Henderson-Hasselbalch at
# MD_SOLVENT_PH.  At pH 9.5 and pKa 10.6 that is f = 0.926, i.e. a 12-APTES
# point becomes 1 neutral + 11 ammonium and the box carries +11 e, neutralised
# by genion Cl- (MD_IONIC_STRENGTH stays 0.0: "-neutral -conc 0.0" is
# counter-ions only, so DI water is still honoured).
#
# WAS: every leg modelled APTES as 100% neutral primary amine, so the ammonium-
# carboxylate driving force — the entire mechanism this grid measures — was
# absent. At 24 copies that was ~+22 e missing from the box.
#
# pKa CORRECTED 2026-08-13: 9.2, not 10.6.
#   10.6 is n-propylamine — a free alkyl amine with nothing else on the chain.
#   HYDROLYSED APTES IS NOT THAT MOLECULE. H2N-C3H6-Si(OH)3 is an internal
#   ampholyte: it carries its own base (the amine) and its own acid (three
#   silanols) three carbons apart, and the silanol depresses the ammonium pKa
#   to a MEASURED 9.1-9.3 in aqueous hydrolysed APTES. The same self-buffering
#   is why the pH of aqueous APTES barely moves over a 40x concentration range
#   (10.0-10.5 from 40 mM to 1.9 M), and why a 1 wt% (45.2 mM) solution — within
#   6% of this experiment's 42.7 mM — measures pH 10.12, not the 11.6 that
#   pKa 10.6 predicts.
#   Source: Campora et al., Polymers 2022;14(9):1820 (PMC9099464),
#   doi:10.3390/polym14091820 — acid-base titration of 1 wt% APTES in water,
#   calibrated meter, temperature-corrected; pH read at zero added acid.
#
# WHY THIS MATTERS MORE THAN IT LOOKS. The protonated fraction is what decides
# how many CATIONIC recognition partners exist, and it is brutally sensitive to
# the pH-minus-pKa gap:
#       pH  8.0   f = 0.94    12 APTES -> 11 ammonium +  1 neutral
#       pH  9.2   f = 0.50    12 APTES ->  6 ammonium +  6 neutral
#       pH 10.1   f = 0.11    12 APTES ->  1 ammonium + 11 neutral
# The superseded pKa 10.6 at the assumed pH 9.5 gave f = 0.926 — i.e. it built
# ~9x more cationic monomer than the measured chemistry supports if the pot
# really sits near 10.1. That is a larger and better-established error than the
# BSA integer-charge rounding this file spends §9c on.
#
# MD_SOLVENT_PH IS NOW THE BINDING UNKNOWN. There is no published pH for ANY
# aqueous TEOS/APTES mixture at a stated composition; the honest bracket for
# this pot at t = 2 h is 7.5-10.3, and f spans 0.94 to 0.11 across it. The pH
# pre-flight must therefore keep refusing to run until a measured value is
# recorded here — an assumed pH now propagates into the count of the
# recognition monomer, not merely into the protein's net charge.
#
# Set MONOMER_PROTONATION_SPLIT to {} to model the neutral amine only (the
# pre-2026-08 behaviour); the split is recorded in every box_composition and in
# point_result.json either way.
MONOMER_PROTONATION_SPLIT = {
    "APTES": {"protonated_name": "APTESH", "pka": 9.2},
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

# DECISION (assumption), NOT MEASUREMENT — but NO LONGER INERT.
# DI water + APTES (a basic aminosilane) gives pH ~9-10, not the 7.4 that the CD
# config assumes.  BSA is strongly negative there.
#
# WHAT THIS KEY NOW DOES (2026-08; §11(b) rewritten):
#   utils_structure.titration_model computes per-residue states at this pH
#   (standard pKa + Henderson-Hasselbalch, propka in preference when present,
#   disulfide-bridged cysteines excluded by SG-SG distance), and
#   utils_gromacs.setup_protein_topology drives pdb2gmx with them and ASSERTS
#   the built protein charge against the model.  Measured on 4F5S chain A:
#       pH 7.4 -> flags []            -> built -16 e  (unchanged; CD path safe)
#       pH 9.5 -> flags ['-his'] +
#                 CYS34 renamed CYM   -> built -18 e
#   The Henderson-Hasselbalch expectation at 9.5 is -28.27 e.  The -10.3 e gap
#   is the honest limit of a fixed-charge force field here; §9c gates the run on
#   acknowledging it.  Do NOT read this number as "the MD ran at a -28 e BSA".
# MEASURED 2026-08-13 on the working mixture, meter, two time points:
#     t = 0    pH 7.91        t = 2 h   pH 7.83        drift -0.08
# The mean of the two is used. A 0.08 drift over the whole stir means one value
# covers the entire simulated window — there is no time-dependence to model.
#
# THIS LANDED AT THE BOTTOM OF THE LITERATURE BRACKET (7.5-10.3) AND 1.6 UNITS
# BELOW THE 9.5 THAT WAS ASSUMED. Two things follow, and both matter.
#
# 1. TEOS IS NOT STALLED — it has already finished. APTES alone at 43 mM gives
#    pH 10-11.6; reaching 7.91 requires ~95% of the amine to be protonated,
#    i.e. >40 mM of acid equivalents already present at the FIRST reading.
#    Inverting Henderson-Hasselbalch on 43 mM base against 179-43 mM excess
#    acid gives an effective acid pKa of 8.41, not the 9.7 of monomeric silicic
#    acid — that is the signature of CONDENSED oligomer/particle-surface
#    silanols (literature pKa ~7-8.5). So hydrolysis AND condensation are
#    substantially complete before t = 0, which is consistent with TEOS-alone
#    turning white. The "2 h is too short" hypothesis is weakened accordingly:
#    the acid-base chemistry is over long before the stir ends.
#
# 2. THE PREVIOUS NUMBER WAS RIGHT FOR THE WRONG REASON. The superseded model
#    used pKa 10.6 at pH 9.5 and got f = 0.93 (11 of 12 APTES cationic). The
#    corrected pKa 9.2 at the MEASURED 7.87 gives f = 0.95 — the same 11. The
#    two errors pointed opposite ways and cancelled. Correcting the pKa ALONE
#    would have moved it to f = 0.33 and made the model wrong; both had to move
#    together. Noted because it is exactly the kind of coincidence that makes a
#    partial fix worse than no fix.
MD_SOLVENT_PH = 7.87


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
# MD_NSTXOUT_COMPRESSED — BSA MUST override the baseline here.
# The baseline 50000 steps (100 ps/frame at dt=2 fs) was sized for CD's 350 ns
# legs, where it still yields 3,500 frames. BSA legs are 30 ns, so the same
# setting would give only 300 frames — and the occupancy analysis reads the last
# 25%, i.e. 75 frames, which is too thin for the Q3-vs-Q4 convergence criterion
# that now BLOCKS acceptance. 5000 steps = 10 ps/frame = 3,000 frames per leg,
# ~100 MB, which is affordable precisely because the legs are short.
MD_NSTXOUT_COMPRESSED = 5000

# MD_TIMESTEP_FS — BSA MUST override the baseline here.
# The baseline turned on PHASE4_HMR_MODE (2026-08-14, "membrane batch"), which
# forces MD_TIMESTEP_FS = 4.0 at module scope.  That is correct for the CD legs
# and ONLY for them: their systems come from CHARMM-GUI, whose topologies are
# genuinely hydrogen-mass-repartitioned (toppar/PROA.itp carries H = 3.024 Da
# and N = 7.959 Da instead of 1.008 / 14.007).  BSA has no membrane and no
# CHARMM-GUI step: its protein topology is built here by pdb2gmx with
# amber99sb-ildn and its monomers by acpype, both of which give H = 1.008 Da,
# and NOTHING in this repository repartitions them -- there is no HMR code at
# all, and the mdp writers all emit `constraints = h-bonds` rather than the
# `all-bonds` the baseline comment claims HMR switches them to.
#
# So the inherited flag would integrate a 1.008 Da hydrogen at 4 fs, which is
# exactly the failure the baseline warns about two lines above its own
# declaration ("otherwise the topology H masses stay 2 Da and 4 fs will fly
# apart").  Overriding the TIMESTEP rather than the FLAG keeps the override
# narrow: PHASE4_HMR_MODE stays True for CD, and no code branches on it.
#
# COST: 30 ns is 15,000,000 steps instead of 7,500,000, so a grid leg is ~7 h
# instead of ~3.6 h.  This is also what the EM/NVT/NPT setup already on disk was
# built with, so those stay reusable.  To buy the 2x back, implement real HMR
# for the pipeline-built path (pdb2gmx -heavyh + repartitioning the acpype
# monomer itps, mass-conserving) -- then this override can go.
#
# OVERRIDE THE FLAG, NOT THE TIMESTEP. config.py re-forces MD_TIMESTEP_FS = 4.0
# at line ~1368, AFTER this file is exec'd into its globals ("MUST be after all
# symbol definitions"), so assigning MD_TIMESTEP_FS here alone is silently
# undone. Turning the flag off is also the honest statement: BSA's hydrogens
# really are 1.008 Da. MD_TIMESTEP_FS is restated below so the value this
# experiment runs at is readable here rather than inherited.
PHASE4_HMR_MODE = False
MD_TIMESTEP_FS = 2.0
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

# ── LEGACY PRESET SWEEP: STAYS OFF, AND IS NO LONGER THE MECHANISM ─────────
# PHASE4_RATIO_SWEEP is the LEGACY preset sweep (relative parts among functional
# monomers).  It stays False for BSA: with APTES the only functional monomer
# every preset collapses onto one composition, so it can express nothing.  The
# ratio question is answered by BSA_RATIO_GRID + PHASE4_RATIO_GRID_ENABLED in
# §9, which are absolute per-monomer copy numbers and ARE imposed on the box.
# (run_phase4 treats the two as mutually exclusive: when the grid is enabled the
# legacy sweep is skipped, so turning both on cannot double-build.)
PHASE4_RATIO_SWEEP = False
# PHASE4_RATIO_PRESETS is deliberately NOT overridden — it must still exist for
# the import inside run_phase4_ratio_sweep, and overloading it with
# (n_TEOS, n_APTES) pairs would mean the wrong thing there.

# n_total monomers placed in the shell, for a leg that is NOT a grid point.
# CORRECTED 2026-08: this used to claim the engine "ALWAYS splits exactly 50/50
# (n_APTES = 30, n_TEOS = 30)".  That has not been true since the audit replaced
# the legacy `total // (n_functional + 1)` formula with SOLGEL_Q_MOLE_FRACTION
# for sol-gel systems.  With SOLGEL_Q_MOLE_FRACTION = 0.60 the default leg is
# n_TEOS = 36, n_APTES = 24, i.e. x_APTES = 0.40 — which is NOT a member of
# BSA_RATIO_GRID (it sits between (40,20) at x=0.33 and (30,30) at x=0.50).
# So (30,30) is NOT a regression anchor for the grid, and the default leg is not
# a grid point.  This is exactly why the grid REPLACES the default leg rather
# than running beside it: otherwise a full leg is spent on a composition nobody
# chose.  With the grid enabled this key is used only by non-grid callers.
# 60 monomers in the box is ~0.21 monomers/nm^2 over BSA's 284.6 nm^2 of SASA,
# i.e. ~8-10% areal coverage — inside the real sol-gel range.
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
# §9  BSA-ONLY KEYS — CONSUMPTION IS MARKED PER KEY
# ═══════════════════════════════════════════════════════════════════════════
# 2026-08: the ratio-sweep engine landed, so this section is no longer uniformly
# declaration-only.  Each key below is marked CONSUMED (with its consumer) or
# UNCONSUMED.  UNCONSUMED_KEYS at the bottom is the machine-readable list and the
# startup banner prints its length on every run — a key that is declared but not
# read is worse than absent, because it reads as configured when it is not.

MONOMER_SET_PINNED = True           # UNCONSUMED — "monomer set is fixed, skip the search"
PINNED_PC = {                       # CONSUMED by code/tools/write_pinned_phase3.py,
    "pc_id": "BSA_TEOS_APTES",      #   which writes the Phase 3 stub run_phase4
    "monomers": ["APTES", "TEOS"],  #   loads from disk (it also stamps the schema
    "crosslinker": "TEOS",          #   imported from phase3_mmsd). See §11(e).
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
BSA_RATIO_GRID = [                  # CONSUMED by phase4_md_validation
    # ._ratio_grid_from_config() -> .run_phase4_ratio_grid(), which turns each
    # entry into an EXPLICIT copies dict and passes it to
    # _run_prepolymerization_md(copies=...).  (n_TEOS, n_APTES), n_total = 100.
    #
    # RE-CENTRED 2026-08-14 on the lab's MEASURED composition. TEOS 0.4 mL and
    # APTES 0.1 mL in 10 mL is 1.791 vs 0.427 mmol, i.e. TEOS:APTES = 4.19:1 and
    # x_APTES = 0.193 — which lands on the (80,20) point. The sweep therefore
    # brackets the real formulation rather than a literature guess, and the
    # optimum cannot sit at a grid edge.
    #
    # n_total 100 = ~58 mM total silane. The BENCH runs at 222 mM (5% v/v), so
    # this is 3.8x dilute. That is a deliberate compromise, not an oversight:
    # matching 222 mM needs ~383 monomers and a larger box. The literature says
    # 5% v/v is unremarkable for de novo silica (Stober 2.2-11.2%), so the
    # experiment is NOT to be diluted to suit the model — the model is the one
    # making the approximation, and it is recorded here as such.
    (100,   0),  # x=0.00  TEOS-only control: no ammonium at all. Isolates the
                 #         non-electrostatic contribution. Chemistry-side NIP
                 #         baseline. pH CAVEAT below.
    ( 95,   5),  # x=0.05  19:1
    ( 90,  10),  # x=0.10   9:1  — low end of the literature window
    ( 80,  20),  # x=0.20   4:1  — THE LAB'S ACTUAL COMPOSITION (x=0.193)
    ( 67,  33),  # x=0.33   2:1  — high end of the literature window
    (  0, 100),  # x=1.00  APTES-only control: electrostatic ceiling. If the
                 #         contact observable rises monotonically to here it is
                 #         measuring amine COUNT, not imprinting. pH CAVEAT below.
]
# pH CAVEAT ON THE TWO ENDPOINTS. MD_SOLVENT_PH = 7.87 was measured on the
# WORKING mixture only. pH is a function of the ratio being swept — the amine is
# a base and the condensed silanol an acid — so the endpoints are NOT covered by
# that measurement. Calculated expectations: x=0.00 near pH 5-6 (silicic acid
# alone), x=1.00 near pH 10 (aqueous APTES measures 10.1 at this concentration).
# At pH 5-6 BSA is close to its pI 4.7 and carries almost no charge; at pH 10
# only ~11% of the amine is protonated. BOTH endpoints therefore simulate a
# different electrostatic regime from the interior points, which is acceptable
# for a CONTROL but would be wrong to read as part of a smooth x-trend. Measure
# them, or interpret them qualitatively.
# THE SWITCH.  The grid is machinery now, but machinery that is OFF by default:
# 8 points x PHASE4_N_REPLICAS(3) x BSA_RATIO_SWEEP_MD_NS(30) = 720 ns of
# production MD, which is days of GPU time and ~2.7 GB of trajectory per leg.
# Turn it on deliberately — here, or with PHASE4_RATIO_GRID_ENABLED=1 in the
# environment for a one-off.  When True the grid REPLACES Phase 4's single
# composition leg (which sits at x_APTES = 0.40, not on this grid); when False
# Phase 4 behaves exactly as it did before.
# The driver logs the total ns and the point list BEFORE the first leg starts.
PHASE4_RATIO_GRID_ENABLED = False   # CONSUMED by phase4_md_validation._ratio_grid_enabled
# Screening pass: set to 1 ONLY to shake the pipeline out. One trajectory per
# point has NO error bar, the tie rule cannot separate any two points, and
# run_phase4_ratio_grid logs an ERROR saying so and reports every point as
# tied. MEASURED basis for that refusal: one archived 350 ns leg, re-analysed
# in 50 non-overlapping 7 ns windows, gave 19 DISTINCT rankings of its four
# monomers — the most common accounted for 6 of 50. None -> PHASE4_N_REPLICAS.
BSA_RATIO_GRID_N_REPLICAS = None    # CONSUMED by run_phase4_ratio_grid
# CLEARED 2026-08-14. The grid is now centred on the MEASURED composition
# (x_APTES = 0.193 lands on the (80,20) point), so the warning this flag used to
# print — "centred on the literature, not on the user's measured volumes" — is
# no longer true and would be a false caveat in every report.
# What remains approximate is the CONCENTRATION, not the ratio: the grid runs at
# ~58 mM against the bench's 222 mM. That is recorded in the BSA_RATIO_GRID
# comment and in BSA_PROTOCOL_MEASURED, not as a provisional-grid flag.
BSA_RATIO_GRID_PROVISIONAL = False  # CONSUMED — run_phase4_ratio_grid logs a WARNING
                                    # banner and stamps `provisional_grid` into
                                    # ratio_grid_summary.json while this is True.
                                    # Flip to False after re-centring on real volumes.
#
# ── THE GRID THE 2026-08 SAMPLING AUDIT RECOMMENDS INSTEAD ────────────────
# The 8-point / n_total=60 / R=3 grid above CANNOT ANSWER THE QUESTION, and the
# arithmetic is not close: with the MEASURED single-molecule window CV of 1.8,
# a point with n_APTES copies and R replicas has CV = 1.8*(W/7ns)^-0.31/sqrt(nR),
# so at R=3 and 30 ns the smallest difference the 3-vs-6-APTES pair can resolve
# (two-sample, alpha 0.05, power 0.80) is 216% — a 3x effect. The whole grid:
#     3 vs 6   216% | 6 vs 12  153% | 12 vs 20 111% | 20 vs 30  88%
#    30 vs 40   74% | 40 vs 60  62%
# Raising n_total is the cheapest lever there is: the box is sized by the SHELL
# RADIUS, not by the monomer count (measured, §6), so 100 monomers cost the same
# per ns as 60 while every point gains sqrt(100/60) = 1.29x precision. Dropping
# the three points past the literature window frees the budget for replicas.
#   BSA_RATIO_GRID = [(100, 0), (95, 5), (90, 10), (80, 20), (67, 33), (0, 100)]
#   BSA_RATIO_TOTAL_MONOMERS = 100
# with R=4 to screen and R=12 on the survivors:
#     5 vs 10  130% -> 66% | 10 vs 20  92% -> 47% | 20 vs 33  68% -> 34%
#    33 vs 100  48% -> 24%
# Cost at the MEASURED 68.7-149.5 ns/day on this machine (RTX 4070 Ti,
# gmx 2025.2, ~210k-atom BSA box): stage 1 = 24 legs x 30 ns = 5-10 GPU-days;
# stage 2 = 24 more legs = another 5-10. Requesting more replicas EXTENDS a
# completed point rather than recomputing it, so stage 2 only pays for the new
# legs. Set R per stage with PHASE4_RATIO_GRID_N_REPLICAS in the environment.
BSA_RATIO_TOTAL_MONOMERS = 100      # CONSUMED by run_phase4_ratio_grid as an
                                    # ASSERTION, not as a second source of truth:
                                    # every point of the x-sweep must sum to it,
                                    # or the driver refuses the grid. A point at
                                    # a different n_total also changes the silane
                                    # concentration, so its difference could not
                                    # be attributed to the ratio. (~58 mM here;
                                    # the bench is 222 mM — see the grid note.)
                                    # The loading axis passes n_total explicitly
                                    # and skips the assertion on purpose.
BSA_LOADING_SWEEP = [30, 60, 100]   # CONSUMED by run_phase4_loading_sweep, which
                                    # re-runs ONE FIXED composition at each of
                                    # these n_total values (~17 / 35 / 58 mM).
                                    # This is the control that separates a real
                                    # composition optimum from a crowding
                                    # artefact. It is NOT crossed with the grid:
                                    # 8 compositions x 3 loadings x 3 replicas is
                                    # GPU-weeks on the wrong axis.
# WHICH composition the loading axis holds fixed.  # BEHAVIOUR CHANGE 2026-08.
# It used to follow the x-sweep's bare argmax. Simulated: under a FLAT truth the
# grid names n_APTES=3 about 30% of the time, and against a real +30% effect at
# R=3 it names the true optimum only ~28% of the time — so the second axis was
# ~72% likely to be resolving the loading dependence of the wrong composition,
# for a further 12 legs (~2.7 GPU-days, ~32 GB). Now:
#   * a dict here PINS it (recommended — use the protocol-centred composition
#     once BSA_PROTOCOL_MEASURED is filled in), e.g. {"TEOS": 48, "APTES": 12};
#   * None falls back to the x-sweep winner, but ONLY if that winner is
#     QUALIFIED (margin >= the MDD and the runner-up outside its replicate CI).
BSA_LOADING_SWEEP_COMPOSITION = None   # CONSUMED by run_phase4 (loading gate)
BSA_LOADING_SWEEP_ENABLED = False   # CONSUMED by _loading_sweep_enabled. Its own
                                    # switch because it is another 3 x n_replicas
                                    # legs ON TOP of the grid, and it is only
                                    # meaningful once a composition is fixed.
BSA_RATIO_SWEEP_MD_NS = 30          # CONSUMED as the per-grid-point time_ns; NOT
                                    # MD_PRODUCTION_NS. Reserve x=0.10 and x=0.20
                                    # for a 350 ns re-run if the sweep comes out
                                    # flat (a flat sweep is itself a result).
                                    # NOTE the value is now honoured VERBATIM.
                                    # It used to be silently replaced by
                                    # MD_QUICK_NS for any value <= 20 ns
                                    # (run_full_md_pipeline's `quick` branch,
                                    # which _run_prepolymerization_md turned on
                                    # with `quick=(time_ns <= 20)`), while the
                                    # REQUESTED value was what got logged and
                                    # stamped into every point_result.json.
                                    # If you change it, change
                                    # MD_MMPBSA_START_NS/END_NS with it — the
                                    # grid now pre-flights that window and says
                                    # so BEFORE the first leg instead of after.
BSA_MAX_MONOMERS_IN_SHELL = 300     # CONSUMED — run_phase4_ratio_grid REFUSES any
    # RAISED 100 -> 300 (2026-09) to run the LOADING axis. The rationale note
    # below already measured the real limit: n = 100, 120, 150, 200 and 300 all
    # place with ZERO fallbacks against BSA's shell, and the first fallback is
    # at n = 400. 300 is the largest MEASURED-SAFE value, so the cap now sits at
    # the evidence rather than below it. Do NOT raise further without re-running
    # the placer: past the real limit the else-branch puts molecule i at
    # r_outer + 0.3*i nm and editconf builds a >20 M atom box.
    # WHY THE LOADING AXIS NEEDS IT: at n_total = 100 the box holds 20 APTES
    # while the BSA surface alone takes ~25 at saturation, so the x-sweep was
    # run in an amine-STARVED regime that the bench (222 mM silane, ~7,400 per
    # BSA) is nowhere near. n_total = 300 puts the box at ~211 mM, i.e. 95% of
    # the bench concentration, which is what makes the ratio conclusion
    # transferable at all.
    # grid point above this before building anything, instead of discovering it
    # as a 250 nm box at editconf time.
    # RATIONALE CORRECTED 2026-08: the stated failure mode is real but the
    # threshold was measured on a small ECL2 shell, not on BSA's. Replaying the
    # placer against BSA's measured shell (r 5.157-6.157 nm, min_sep 1.0 nm),
    # n = 100, 120, 150, 200 and 300 all place with ZERO fallbacks; the first
    # fallbacks appear at n = 400. 100 is therefore conservative, not a cliff —
    # but do not raise it casually, because past the real limit the else-branch
    # puts molecule i at r_outer + 0.3*i nm and editconf builds a >20 M atom box.


# ── §9b  SAMPLING & DECISION RULE (2026-08 sampling audit) ─────────────────
# These six keys are declared HERE rather than in config.py's baseline body on
# purpose: config_baseline_CD.json is a frozen pre-dispatch contract and
# test_config_regression T1 asserts CD's namespace gains nothing.  Under BSA
# they are live config; under CD phase4 falls back to the SAME documented
# defaults and stamps them into ratio_grid_summary.json.effective_sampling_config.
#
# WHY THEY EXIST.  Measured on the archived 350 ns CD63 and CD9 legs:
#   * the integrated autocorrelation time of the per-copy contact signal is
#     8-33 ns (CD63 IBTES 7.9, PTES 8.2, TTMS 9.2, TMOS 1.5; CD9 AAPBA 33.3,
#     FPBA 9.7, UNL 10.2);
#   * so window-to-window noise falls as T^-0.31, not T^-0.50 — halving it by
#     lengthening a leg costs ~9x the GPU time, while replicates still buy
#     R^-0.50;
#   * splitting ONE 350 ns trajectory into 50 non-overlapping 7 ns windows (the
#     Q4 window of a 30 ns leg) gives 19 DISTINCT rankings of its 4 monomers;
#   * the 4 within-window block means the convergence gate computes understate
#     the true window-to-window SD by a median 2.8x (up to 7.5x), so they are
#     NOT an error bar and nothing may propagate them as one.
PHASE4_ANALYSIS_WINDOW_FRACTION = 0.50   # CONSUMED by _analyze_monomer_occupancy
    # Fraction of the trajectory, from the END, that the occupancy analysis
    # reads. WAS 0.25 (`use_q4=True`). At 30 ns that was a 7.5 ns window, i.e.
    # under one correlation time. 0.50 doubles the sample for free; anything
    # earlier is pre-equilibration.
PHASE4_ZERO_CONTACT_IS_A_MEASUREMENT = True   # CONSUMED by the convergence block
    # A species that is PRESENT and records zero contacts has MEASURED a zero at
    # this composition. It used to make `converged` undecidable and reject the
    # whole replica — which biases rejection toward the TEOS-rich end, exactly
    # where this grid asks its question. Both 12-monomer smoke legs were
    # rejected on this criterion alone, and the production grid contains points
    # with 3 and 6 APTES. A box where NO species contacts anything is still a
    # rejection: that is a dead box, not a composition.
PHASE4_RANK_TIE_RULE = "ci"          # CONSUMED by _rank_grid_points
    # "ci" (default) | "mdd" | "exact". The retired guard was float equality on
    # 4-dp values: simulated against the pipeline's own arithmetic under a
    # completely flat truth it fired 0.11% of the time at R=1 and 0.14% at R=3,
    # so a flat sweep announced a unique winner in 99.9% of runs with a median
    # margin of 18%.
PHASE4_RANK_MDD_PCT = 30.0           # CONSUMED by _rank_grid_points
    # Pre-registered minimum detectable difference, percent of the top score.
    # Basis: single-molecule window CV ~1.8, so a point with n functional copies
    # and R replicas has CV ~ 1.8/sqrt(n*R); at n=12, R=3 that is 30%.
PHASE4_RANK_CI_LEVEL = 0.95          # CONSUMED by _mean_ci / _rank_grid_points

# PHASE4_CONVERGENCE_GATE — "advisory", not "blocking", for BSA.
# CONSUMED by _replica_acceptance.
#
# The within-leg gate asks "did these two 3.75 ns block means agree to 10%?".
# MEASURED on this grid's own trajectories: tau of the contact count is
# 0.45-0.78 ns, so a block holds ~4 independent samples and the block-to-block
# difference expected from NOISE ALONE is 17-20%. A 10% threshold sits BELOW
# that noise floor, so it accepted and rejected legs at close to random: of the
# first two completed legs, one failed at 28% and one passed at 5.7% while the
# expected noise was 20% and 17% respectively. Filtering on a coin flip and then
# ranking the survivors biases the grid.
#
# There is no published threshold to fall back on. The field's designated
# best-practices document (Grossfield, Patrone, Roe, Schultz, Siderius &
# Zuckerman, LiveCoMS 1(1):5067, 2018) deliberately declines to give blanket
# numeric acceptance criteria, and its checklist requires that any sampling
# metric used for exclusion be applied uniformly and for "an objective and
# compelling reason". Metrology says the same thing more bluntly (NIST IR 8526,
# quoting Mandel 1991): excluding on purely statistical grounds "sharply reduced
# the field to which the inferences from the study apply".
#
# So the leg is no longer judged converged-or-not. Uncertainty is carried where
# it belongs — the Student-t CI over INDEPENDENT REPLICAS (_mean_ci), which the
# ranking already consumes, and which is honest that n=1 cannot rank at all.
# Replicas here re-draw the monomer placement as well as the velocities
# (_replica_seed feeds both), so that CI measures the variable that actually
# dominates: where the monomers started, not the restrained protein.
#
# NOT PRE-REGISTERED. This was changed AFTER seeing that all six legs failed the
# old gate. It is a change of statistic, not a selection of legs — the new rule
# keeps every leg, so it cannot cherry-pick — but the sequence must be stated in
# any write-up rather than presented as the original plan.
PHASE4_CONVERGENCE_GATE = "advisory"
PHASE4_LOADING_SWEEP_REQUIRE_QUALIFIED_WINNER = True   # CONSUMED by run_phase4

# ── §9c  pH REALISM GATE ───────────────────────────────────────────────────
# MD_SOLVENT_PH is now GENUINELY LIVE (see §11(b), rewritten): pdb2gmx is driven
# with states from utils_structure.titration_model, and setup_protein_topology
# asserts the built protein charge against them.  Measured on 4F5S chain A:
#     pH 7.4  HH -16.65 e | discrete -17 e | built -16 e (pdb2gmx default path)
#     pH 9.5  HH -28.27 e | discrete -19 e | built -18 e  <- what will run
# The remaining -10.3 e is not a bug and not an error bar: Lys (pKa 10.5) is 91%
# protonated at pH 9.5 and a fixed-charge model must round it to 100%, and
# amber99sb-ildn has no tyrosinate residue type at all (nor a neutral N-terminus
# — its aminoacids.n.tdb / .c.tdb ship EMPTY, so `pdb2gmx -ter` has nothing to
# offer). run_phase4_ratio_grid REFUSES TO START while this exceeds the
# tolerance and has not been acknowledged, so the approximation is chosen by a
# person rather than by a default.
MD_PH_CHARGE_RESIDUAL_TOL_E = 2.0    # CONSUMED by phase4_md_validation._ph_preflight
MD_PH_CHARGE_RESIDUAL_ACK = False    # CONSUMED — set True ONLY after deciding
    # that a -18 e BSA is an acceptable stand-in for a -28 e one for THIS
    # question. The alternatives are: lower MD_SOLVENT_PH to a value the force
    # field can represent, install propka for site-specific pKa, or move to
    # constant-pH MD (which this pipeline does not have).

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
BSA_PROTOCOL_MEASURED = {           # CONSUMED as provenance; FILLED IN 2026-08-14 from the bench
    "teos_uL":         400.0,       # 0.4 mL -> 373 mg -> 1.791 mmol -> 179 mM
    "aptes_uL":        100.0,       # 0.1 mL ->  95 mg -> 0.427 mmol ->  43 mM
    "bsa_mg_per_mL":   1.0,
    "bsa_solution_mL": None,        # not separately reported; total is 10 mL
    "tween20_pct_w_v": 1.0,
    "di_water_mL":     10.0,        # total working volume
    "stir_hours":      2.0,
    # DERIVED — the two numbers the grid is centred on:
    "teos_aptes_molar_ratio": 4.19,
    "x_aptes":                0.193,   # -> the (80, 20) grid point, x = 0.20
    "total_silane_mM":        222.0,   # 5.0% v/v
    # MEASURED pH of this exact mixture: t=0 7.91, t=2h 7.83 -> MD_SOLVENT_PH 7.87
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
#
# REMOVED 2026-08, because they are now genuinely read by code (each one's
# consumer is named at its declaration above):
#   BSA_RATIO_GRID, BSA_RATIO_GRID_PROVISIONAL, BSA_RATIO_GRID_N_REPLICAS,
#   BSA_RATIO_TOTAL_MONOMERS, BSA_LOADING_SWEEP, BSA_RATIO_SWEEP_MD_NS,
#   BSA_MAX_MONOMERS_IN_SHELL   -> phase4_md_validation.run_phase4_ratio_grid
#   PINNED_PC                   -> code/tools/write_pinned_phase3.py
#   MD_SOLVENT_PH               -> it was NEVER unconsumed. phase1_epitope_prep
#                                  from-imports it by name and passes it to
#                                  assign_protonation_states / _verify_protonation,
#                                  and it lands in phase1_results.json as
#                                  protonation.ph = 9.5. The honest label is
#                                  "consumed, NO PHYSICAL EFFECT (propka absent)",
#                                  which is what §11(b) already explains. Leaving
#                                  it on this list was actively dangerous: a
#                                  future cleanup of "unconsumed" keys would have
#                                  broken Phase 1 with an ImportError.
UNCONSUMED_KEYS = frozenset({
    "MONOMER_SET_PINNED", "RUN_PHASES", "BSA_PROTOCOL_MEASURED",
    "SOLVENT_CONDITION", "MD_EXPLICIT_SURFACTANT", "SELECTIVITY_DEFERRED",
    "BSA_COMPETITOR_TARGETS", "PHASE4_PROTEIN_RESTRAINT_K",
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
# makes it fall out structurally.
#
# THE EBN-DERIVED optimal_ratio IS STILL DEGENERATE FOR THIS SYSTEM, and no
# engine fix changes that: it is round(EBN/min(EBN)) over FUNCTIONAL monomers,
# which for a single functional monomer is always exactly 1, whatever the MD
# measured.  Two things about it DID change in 2026-08 and are worth knowing:
#   * the crosslinker term used to come out 1 as well — 0.6/(1-0.6) evaluates to
#     1.4999999999999998 in IEEE754 and round() gave 1 — so Phase 6 published
#     "APTES:TEOS = 1:1, crosslinker 50 mol%" from a 60 mol% target, contradicting
#     the box it claimed to derive from.  That rounding is fixed.
#   * for a box built from EXPLICIT copies (every ratio-grid point), the
#     crosslinker term is now derived from THAT BOX's own crosslinker mole
#     fraction rather than from the library constant, so the emitted recipe at
#     least agrees with the leg it came from.
# The ratio ANSWER therefore does not come from optimal_ratio at all.  It comes
# from comparing grid points — ratio_grid_summary.json / per-point
# point_result.json — which is the entire reason BSA_RATIO_GRID exists.  Read
# Phase 6's optimal_ratio for this experiment as provenance, not as the answer.
_rederive()  # noqa: F821

# ── AFTER _rederive(): pH SPECIATION FORMS ────────────────────────────────
# §10's rule is that anything overriding a DERIVED symbol goes after the call.
# ALL_MONOMERS is derived; FUNCTIONAL_MONOMERS and CROSSLINKERS are derived
# FROM it, and they must stay pinned to {APTES} / {TEOS} — a speciation form is
# not a reagent to screen, and putting it in SILANE_MONOMERS would make Phase 3
# enumerate combinations of a molecule with itself.  So it is added to
# ALL_MONOMERS only, which is the map utils_gromacs.parameterize_monomer reads.
ALL_MONOMERS["APTESH"] = _APTES_PROTONATED_RECORD    # noqa: F821
assert sorted(FUNCTIONAL_MONOMERS) == ["APTES"], sorted(FUNCTIONAL_MONOMERS)  # noqa: F821
assert sorted(CROSSLINKERS) == ["TEOS"], sorted(CROSSLINKERS)                 # noqa: F821


# ═══════════════════════════════════════════════════════════════════════════
# §11  HONESTY NOTES — read these before trusting any BSA output
# ═══════════════════════════════════════════════════════════════════════════
# (a) THE FABRICATED-SILANE-CHARGE ARTEFACT IS FIXED.  RESOLVED 2026-08 — this
#     note used to read as a standing blocker on any BSA production run.  It is
#     not one any more, and it is left here (rather than deleted) because the
#     REASON it was blocking is the reason MD_IONIC_STRENGTH = 0.0 is safe.
#     WAS: _generate_silane_itp computed Gasteiger charges on an Si->S proxy and
#     then overwrote the Si charge with a hardcoded +0.9 WITHOUT renormalising,
#     so every silane ITP carried a non-integer net charge (intact TEOS +0.676,
#     hydrolysed TEOS +0.687, intact APTES +0.811, hydrolysed APTES +0.819).  On
#     a 30+30 box that is +45.17 e of charge that does not exist, which grompp
#     balanced with 29 Cl- (18.5 mM in a 2602 nm^3 box).
#     NOW: verified on freshly generated production-path topologies for all four
#     species — qtot = +0.00000 / -0.00000 for intact TEOS, intact APTES,
#     hydrolysed TEOS and hydrolysed APTES alike.  The built system carries BSA's
#     own counter-ions and NOTHING else (16 Na+, 0 Cl-), i.e. the DI-water
#     condition is genuinely honoured.
#     WHY IT MATTERED SO MUCH HERE: Debye length = 0.304/sqrt(I) nm, so 150 mM
#     screens at 0.78 nm but 18.5 mM screens at 2.24 nm.  A DI-water condition
#     AMPLIFIES any fabricated charge ~3x in range, acting on a pI-4.7 protein —
#     it would have manufactured part of the exact electrostatic signal this
#     experiment exists to measure.  Keep this paragraph in mind before any
#     future change to monomer charge assignment.
#
# (b) MD_SOLVENT_PH — RESOLVED 2026-08, PARTIALLY.  This note used to read
#     "CHANGES NO PROTONATION STATE ... BSA WILL be simulated with pH-7
#     protonation whatever this key says", for five reasons.  Four are fixed and
#     the fifth is now a measured, gated approximation.
#
#     WAS (all verified at the time): (i) propka absent -> ImportError branch ->
#     shutil.copy2; (ii) even with propka the predicted HIP/CYM were logged and
#     discarded; (iii) the output filename was hardcoded "_pH74.pdb" whatever
#     `ph` was; (iv) pdbfixer addMissingHydrogens hardcoded to 7.4; (v) pdb2gmx
#     ran with no -asp/-glu/-lys/-his/-ter flags.  Measured consequence: the
#     built topology summed to -16 e (ASP -39, GLU -59, LYS +59, ARG +23,
#     HIS +1) and genion added 16 Na+, at a pH where the census titrates to -28.
#
#     NOW: (i)/(iii) fixed — utils_structure.titration_model always runs, propka
#     or not, the output is named from the actual pH, and the model is written
#     to the *_protonation.json sidecar; (ii)/(v) fixed —
#     utils_gromacs.setup_protein_topology builds a pdb2gmx plan from the model
#     (-lys/-asp/-glu/-his where the majority state differs from pdb2gmx's
#     default, plus a CYS->CYM rename because pdb2gmx has no -cys flag) and then
#     ASSERTS the built protein charge against it.  Verified end to end on
#     4F5S chain A with GROMACS 2025.2:
#         pH 7.4 -> no flags       -> built -16.000 e   (CD path bit-identical)
#         pH 9.5 -> ['-his'] + CYM34 -> built -18.000 e (assertion passes)
#     Disulfides are detected by SG-SG distance (34 of BSA's 35 cysteines are
#     bridged; only Cys34 titrates) — without that the standard-pKa model
#     deprotonates all 35 above pH 8.5 and predicts a charge ~34 e too negative.
#
#     WHAT IS STILL AN APPROXIMATION, and why it is a GATE rather than a note:
#     the Henderson-Hasselbalch expectation at pH 9.5 is -28.27 e and the best
#     representable fixed-charge state is -18 e.  The -10.3 e is carried by
#     Lys (pKa 10.5, 91% protonated at 9.5 — a fixed-charge model must round to
#     100%: 59 x 0.09 = 5.3 e) and by tyrosinate (20 Tyr, pKa 10.1, 20%
#     deprotonated = 4.0 e), for which amber99sb-ildn has NO residue type.  A
#     neutral N-terminus is unrepresentable too: this GROMACS port ships EMPTY
#     aminoacids.n.tdb / .c.tdb, so `-ter` prints no menu and silently keeps
#     NH3+ (measured — that is the 1 e between "discrete -19" and "built -18").
#     phase4_md_validation._ph_preflight refuses to start the grid while
#     |residual| > MD_PH_CHARGE_RESIDUAL_TOL_E unless MD_PH_CHARGE_RESIDUAL_ACK
#     is True.  (iv) pdbfixer's hardcoded 7.4 is untouched and now harmless: it
#     only adds hydrogens, and pdb2gmx runs with -ignh and rebuilds them.
#
# (c) DEAD KEYS — PARTIALLY RESOLVED.  MD_BOX_TYPE and MD_BOX_DISTANCE are NO
#     LONGER DEAD: setup_simulation_box (utils_gromacs.py:579) now defaults both
#     to the config values and grows the padding until the SOLUTE clears
#     2*rvdw + MIN_IMAGE_MARGIN_NM, verified on the box editconf actually built.
#     The live values are dodecahedron / 1.2 nm.  Anything in §6 that costs the
#     run from a cubic -d 0.5 box is therefore stale — the real box is smaller
#     (~256k atoms, not ~290k) AND §6 omits PHASE4_N_REPLICAS = 3, so a grid
#     point is 3 x MD_PRODUCTION_NS, not one.  Re-budget before committing GPU
#     time; see §9's grid banner, which prints the total ns up front.
#     STILL DEAD: MD_FF_PROTEIN, MD_FF_MONOMER, MD_WATER_MODEL —
#     setup_protein_topology hardcodes amber99sb-ildn / tip3p as default args and
#     the call sites pass only two positional arguments.  Happily the desired BSA
#     values are the baseline values anyway.
#
# (d) verify_phase5's cross-target loop is `for o in TARGETS if o != t`.  With a
#     single target it is EMPTY, so the selectivity check produces nothing and
#     reports a VACUOUS PASS.  Read that as SKIP, never as success.
#
# (e) PHASES 2/3 CANNOT BE SKIPPED BY CONFIG, AND THE OLD HAND-WRITTEN STUB NO
#     LONGER WORKS.  run_pipeline builds the phase list from --phase only, and
#     run_phase4 loads top_pcs from phase3_mmsd_results.json ON DISK.  A stub of
#     the old shape {"BSA": {"top_pcs": [PINNED_PC]}} is now refused at TWO
#     gates: run_pipeline._check_phase_completed wants a manifest completion
#     record with a matching sha256 and input fingerprint, and run_phase4 raises
#     unless the entry carries schema == phase3_mmsd._PHASE3_SCHEMA.
#     THE WORKING RECIPE, and the tool that writes it (it reads PINNED_PC and
#     imports the schema constant, so neither can drift):
#         MIP_EXPERIMENT=BSA python3 run_BSA.py --target BSA --phase 1 --skip-md
#         MIP_EXPERIMENT=BSA python3 code/tools/write_pinned_phase3.py
#         MIP_EXPERIMENT=BSA python3 run_BSA.py --target BSA --phase 4 \
#             --adopt-existing-tree
#     (write_pinned_phase3.py also drops reports/run_manifest.json, which is what
#     --adopt-existing-tree then re-adopts under the current fingerprint.)
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
# (i) MONOMER H-BOND DONORS ARE NOW REPRESENTABLE AND COUNTABLE.  RESOLVED
#     2026-08.  This note is the reason §1 used to default to "intact"; all
#     three defects it described have been re-tested and are gone, which is why
#     §1 now defaults to "hydrolyzed".  Kept as the record of what was measured.
#
#       1. TYPING — WAS: an element-keyed table typed every H as GAFF2 `h1`
#          (sigma 0.2422 nm vs `ho` 0.0538 nm), so the LJ wall alone moved the
#          H...O minimum from 0.196 nm to 0.302 nm and forbade H-bond geometry.
#          NOW: H is typed by bonded neighbour.  Re-generated through
#          parameterize_monomer: hydrolysed TEOS emits 4x `ho`; hydrolysed APTES
#          emits 3x `ho` + 2x `hn`; zero h1 hydrogens sit on any O or N in any of
#          the four species (intact TEOS/APTES included).
#
#       2. GEOMETRY — WAS: _std_bond_len keyed (8,1)/(7,1)/(6,1) but looked up
#          (min,max) = (1,8)/(1,7)/(1,6), so EVERY X-H bond missed and took the
#          0.1500 nm default (38-56% too long, and constrained there).
#          NOW: O-H = 0.09725 nm, N-H = 0.10192 nm, C-H = 0.1096 nm, and 0 of 4
#          (hydrolysed TEOS) / 0 of 11 (hydrolysed APTES) / 0 of 20 (intact TEOS)
#          / 0 of 23 (intact APTES) X-H bonds remain at 0.1500.  All four carry
#          [dihedrals] and [pairs] and renormalise to qtot 0.00000.
#
#       3. COUNTING — WAS: HBA(...) was built with no `hydrogens_sel`, so
#          MDAnalysis 2.10 fell back to guess_hydrogens(min_charge=0.3) against a
#          maximum monomer H charge of 0.2337.  ZERO monomer hydrogens qualified,
#          so the monomer-as-DONOR direction was identically zero for BOTH
#          species — the one observable the §1 toggle exists to measure.
#          NOW: phase4_md_validation passes `hydrogens_sel` explicitly and
#          restricts donors AND acceptors to N/O/S BY MASS (see
#          _HB_HYDROGEN_SEL / _HB_DONOR_ACCEPTOR_SEL), so the guess is never
#          consulted.  Two further defects in the same call were found and fixed
#          with it, both of which mattered more than the original:
#            * acceptors_sel was passed VERBATIM.  MDAnalysis only calls
#              guess_acceptors() when acceptors_sel is None, so "protein" made
#              every carbon and hydrogen in the selection an acceptor.  The
#              spurious count scales with monomer ATOM COUNT (intact TEOS 33 vs
#              Si(OH)4 9), so it biased the very species comparison this toggle
#              exists to make.
#            * the guessed selection is a (resname, name) STRING, and acpype
#              names every monomer residue UNL.  In a two-monomer box — i.e.
#              every box the ratio grid builds — a hydrogen of species A that
#              cleared the threshold admitted the same-named hydrogen of species
#              B, including carbon-bonded ones.
#          The min_charge threshold also silently deleted the PROTEIN side: an
#          amber99sb-ildn backbone amide H carries +0.2719, below 0.3.  Verified
#          directly against MDAnalysis 2.10: guess_hydrogens at min_charge=0.3
#          returns only the monomer silanol H and DROPS the backbone amide H;
#          at 0.2 it returns both.  The explicit selections return both
#          unconditionally.
#
#     THE RESIDUAL IS NOW GONE TOO (2026-08 sampling audit).  This paragraph
#     used to read: "the acpype call hardcodes '-n 0', so the protonated
#     ammonium form [NH3+]CCC[Si](O)(O)O — the majority species at pH ~9.5 and
#     the actual imprinting driving force — still cannot be parameterised on the
#     production path.  APTES is modelled as a NEUTRAL amine this round."
#     Both halves are fixed:
#       * utils_gromacs._run_acpype now derives the net charge from the library
#         SMILES' RDKit formal charge instead of hardcoding 0, and
#         _assert_itp_net_charge checks the built topology against it.  Measured
#         on the production path: APTESH qtot=+1.00000, 20 atoms, 0/12 X-H at
#         0.1500 nm, O-H 0.09725, N-H 0.10271, polar H {hn, ho}, max +0.4611,
#         [dihedrals] and [pairs] present.
#       * MONOMER_PROTONATION_SPLIT (§4) puts it IN THE BOX: every grid point's
#         APTES copies are split into neutral + ammonium at the
#         Henderson-Hasselbalch fraction (0.926 at pH 9.5, pKa 10.6), so a
#         12-APTES point builds 1 APTES + 11 APTESH.  Verified in a real build:
#         [ molecules ] = Protein_chain_A 1 / APTESH 4 / TEOS 8 / SOL 59637 /
#         NA 14 — i.e. the +4 e from the ammonium simply reduced the Na+ count.
#     A further silane guard landed with it: an acpype failure on a silane now
#     RAISES instead of falling through to the hand-built Gasteiger path, whose
#     polar-H charges are ~0.15-0.19 e against acpype's ~0.35-0.46 e.  Two
#     silanes from different charge models in one box compares charge models,
#     not compositions.  MIP_ALLOW_HANDBUILT_SILANE=1 opts back in for debugging.
#
# (j) BSA PHASE 4 RUNS A RIGID TEMPLATE.  See the block above PHASE4_TEMPLATE_MODE
#     in §7: -DPOSRES + pdb2gmx posre.itp restrains all 4,653 BSA heavy atoms at
#     1000 kJ/mol/nm² for the whole production run, side chains included.  Read
#     contact counts, EBN and HBNMax as "monomer behaviour against a frozen BSA
#     surface", never as induced fit.  A backbone-only restraint would need a
#     `gmx genrestr` on a C-alpha/backbone index group and a phase-code knob;
#     that is engine work, deliberately not attempted from config.

# ── Protonation rule for the BSA experiment ────────────────────────────
# BSA at the working pH is a polyanion and the recognition monomer (APTES) is
# a cation, so the imprinting driving force is long-range Coulomb — and the
# solvent is DI water, so there is no salt to screen it and the Debye length
# is long. That makes the NET charge the quantity to get right, which is why
# this experiment uses "net_charge" where the CD experiment keeps "majority".
#
# Measured on 4F5S chain A at pH 9.5 (propka3 3.5.1):
#     majority + textbook pKa   built -19 e   residual -9.27 e
#     majority + PROPKA         built -25 e   residual -6.92 e
#     net_charge + PROPKA       built -32 e   residual +0.08 e   <- shipped
# The seven sites flipped are all lysines with pKa within ~0.4 of the working
# pH (LYS20, 114, 350, 471, 474, 523, 524), i.e. the ones the majority rule
# was least sure about. Their LOCAL charge is now wrong; the net charge and
# the long-range field are right. Run one grid point under "majority" as a
# control and record whether the ranking moves — if it does not, the choice is
# immaterial and that is worth stating; if it does, that is the sensitivity.
PH_CHARGE_ASSIGNMENT = "net_charge"
PH_USE_PROPKA = True

# ── pH provenance gate ─────────────────────────────────────────────────
# Set to True ONLY after MD_SOLVENT_PH has been set from a MEASUREMENT of this
# pot (meter, at the composition and time point being simulated). This is a
# separate gate from MD_PH_CHARGE_RESIDUAL_ACK: that one asks "is the residual
# after rounding acceptable?", which the LYN net-charge assignment now answers
# (+0.08 e). This one asks the prior question — "is the pH we rounded TO the
# right pH?" — and no amount of careful rounding fixes a wrong target.
# Recommended measurement: pH at t = 0, 2, 5, 15, 30, 60, 120 min for four
# beakers at matched molarity — TEOS-only (179 mM), APTES-only (43 mM), the
# working mixture, and DI water with 1% Tween20 as the blank. The mixture pH is
# the readout of how much emulsified TEOS has actually hydrolysed, which is the
# single unknown behind the whole 7.5-10.3 bracket.
# Released 2026-08-13 against the measurement recorded at MD_SOLVENT_PH above.
# Set back to False if the composition, the scale or the protocol changes — the
# pH is a property of THIS pot, and the grid deliberately varies the ratio that
# sets it. Grid points far from the measured composition (in particular the
# APTES-only control, expected near pH 10) are NOT covered by this measurement
# and should carry their own value or their own acknowledgement.
MD_SOLVENT_PH_MEASURED = True
