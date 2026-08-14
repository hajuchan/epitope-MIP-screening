"""CD experiment overrides — exosome tetraspanin ECL2 (CD63 / CD81 / CD9).

Exec'd into config.py's globals() by the dispatcher at the bottom of that file.
Read the comment block there before editing this one.

THIS FILE IS DELIBERATELY EMPTY OF VALUE ASSIGNMENTS.
config.py's body IS the CD experiment. The CD run must stay identical to what
produced the ~500 GB in results/, so there is nothing to override, and
_rederive() is deliberately NOT called — the symbols built at lines
492/539/542/670/728 are already correct, and re-running would rebuild
equal-but-new objects for no reason.

Put any FUTURE CD-only change here rather than in config.py, so BSA and later
experiments keep inheriting an unchanged shared baseline.
"""
if "__CONFIG_DISPATCH__" not in globals():
    raise ImportError(
        "config_CD.py is a DELTA file exec'd into config.py's namespace by the "
        "dispatcher at the bottom of config.py. It is NOT importable on its own — "
        "every name it relies on (PROJECT_ROOT, synthesis_pH_window, _rederive, …) "
        "lives in config.py's globals. Use `import pipeline.config` instead; CD is "
        "the default experiment.")

EXPERIMENT_LABEL = "CD63/CD81/CD9 tetraspanin ECL2 epitope imprinting"
# Declared here, read by nothing yet. The startup banner prints the count so a
# declaration is never mistaken for a live setting — the 2026-08 audit found 16
# such keys on the BSA side alone, several of which read as configured when the
# machinery behind them did not exist. The EV_* group is vesicle provenance and
# the glycan assumption that follows from it; they become consumed when Route P
# (glycan modelling) is implemented.
UNCONSUMED_KEYS = frozenset({
    "EV_SOURCE_CELL_LINE", "EV_SOURCE_CATALOGUE",
    "EV_GLYCAN_TYPE", "EV_GLYCAN_SIALYLATED",
})
# OUTPUT_DIR stays PROJECT_ROOT/"results" exactly as defined at config.py:669.

# ═══════════════════════════════════════════════════════════════════════════
# THE 2026-08 AUDIT KEYS LIVE IN config.py's BODY, NOT HERE — AND WHY
# ═══════════════════════════════════════════════════════════════════════════
# The audit added 25 configuration symbols (docking seed, receptor-charge
# policy, MMSD docking order and Pareto weights, Phase 4 replicas and
# acceptance tolerances, box composition, the Phase 5 contact observable,
# trajectory write rate, protonation policy). Every one of them is consumed by
# a SHARED module — utils_gromacs, utils_autodock, utils_structure,
# phase3_mmsd, phase4_md_validation, phase5_rebinding — that both experiments
# execute. A key defined only in this file would be invisible under
# MIP_EXPERIMENT=BSA, and the consuming modules would silently fall back to
# their built-in defaults on the BSA path only: the single worst failure mode
# this dispatch design exists to prevent, because it is invisible in CD testing.
#
# So they are in config.py's body, grouped under the phase that consumes them,
# each with a comment naming the consuming function and the measurement that
# justifies its value. CD therefore inherits them, exactly as it inherits every
# other baseline value, and this file stays what its header promises: a delta.
#
# ONE experiment-specific override was needed, and it is in config_BSA.py, not
# here: MD_NSTXOUT_COMPRESSED. The baseline 50000 steps (100 ps/frame) is sized
# for CD's 350 ns legs (3,500 frames); BSA's 30 ns legs would get 300, which is
# too thin for the Q3-vs-Q4 convergence gate. BSA overrides it to 5000.
#
# If a CD-only value is ever needed for one of these keys, override it BELOW
# this block — not by editing config.py, which is the shared baseline.
#
# REGRESSION NOTE: those 25 additions change the CD namespace, so
# code/tests/test_config_regression.py::test_cd_unchanged (T1) required
# config_baseline_CD.json to be regenerated:
#     python3 code/tools/config_baseline.py --write
# T2-T10 are unaffected.

# ═══════════════════════════════════════════════════════════════════════════
# STATE OF results/phase2 — READ BEFORE QUOTING ANY CD PHASE-2 NUMBER
# ═══════════════════════════════════════════════════════════════════════════
# 2026-08-12 12:07, during the silane-SMILES fix session, 18 files (the 9
# corrected silanes × .pdb/.pdbqt) were MOVED out of
#     results/phase2/monomers   ->   results/phase2/monomers_stale_wrong_smiles
# Affected monomers: APTES APTMS CETES EDTMS GPTMS IBTES ICTES MPTMS UPTMS.
# (An earlier hand-off note claimed "newest mtime in results/ is 2026-07-28 …
#  zero files created, modified or deleted under results/". That claim is FALSE.
#  Its stat probe sampled results, results/phase1, results/phase6 and
#  results/reports — and results/phase2, the tree that had actually changed, was
#  not among them. Full audit of 2026-08-12 activity under results/:
#    12:07  phase2/{monomers, monomers_stale_wrong_smiles}   the 18-file move below
#    12:44  phase1/{CD63,CD81,CD9}/stability_md/_verify_centered.gro   created, and
#           reports/phase1_verification.json + reports/phase1_summary.md rewritten
#           — side effects of a `verify_phase.py 1` run; harmless, verifier output
#           only, no primary data.
#    13:09  results/ dir mtime only — a stray results/validation/ (created 13:08 by
#           a `run_validation.py --check-only` probe, birth-stamped 13:08, nothing
#           pre-existed) was removed again. Content-wise results/ is back as it was.)
#
# CONSEQUENCE: results/phase2/phase2_smd_results.json is still the 2026-05-13
# file, computed from the OLD (Si-C-less, tetraalkoxysilane) SMILES for those 9
# monomers, while their docking-input cache has been renamed away. The CD Phase-2
# tree is therefore INTERNALLY INCONSISTENT: 9 of 27 binding energies are
# old-SMILES numbers with no reproducible inputs beside them.
#
# NOT REPAIRED HERE, deliberately — this session is config-only and must not
# touch results/. Two defensible resolutions, USER'S CALL:
#   (A) re-dock the 9 silanes and regenerate phase2_smd_results.json, then let
#       Phase 3 re-run. Correct, and the only way CD Phase-2 numbers become
#       self-consistent with the fixed SMILES. Costly.
#   (B) restore monomers_stale_wrong_smiles/* back into monomers/ and treat the
#       whole CD result set as a frozen historical artefact of the old SMILES.
#       Cheap, but then the 9 fixed SMILES in config.py no longer describe what
#       results/ contains.
# Until one is chosen, do NOT present CD Phase-2 binding energies for those 9
# silanes as products of the corrected structures.

# ═══════════════════════════════════════════════════════════════════════════
# TEMPLATE DECISION, SUPERSEDED — the "TARGETS IS UNCHANGED" block that stood
# here from earlier on 2026-08-12 has been overturned. Its measurements were
# right; its decision criterion no longer applies.
# ═══════════════════════════════════════════════════════════════════════════
# That block rejected the CD81 5TCX -> 7JIC swap because 7JIC supplies only
# 1 of the 16 residues of the frozen head 168-183 as clean experimental
# coordinates. That was a correct measurement of the wrong quantity. THE
# 16-MER HEADS ARE RETIRED: the lab now receives nanovesicles with each CD
# protein overexpressed in a lipid bilayer and imprints the intact vesicle, so
# there is no peptide to order and nothing to immobilise. The imprinted
# surface is the extracellular face of the protein in a membrane, which is
# exactly what the rejected swap was optimising.
#
# For the record, and measurable with
#     python3 code/tools/check_cd_templates.py
# the heads survive anyway once the templates are actually REPAIRED rather
# than used raw — which is what the earlier block did not evaluate:
#     CD63 157-172  16/16 modelled, 16/16 experimental backbone
#     CD81 168-183  16/16 modelled, 13/16 experimental (168-170 rebuilt de novo)
#     CD9  156-171  16/16 modelled, 16/16 experimental
# So the head-loss objection was an artefact of comparing a repaired template
# against an unrepaired one. It is recorded here, not enforced anywhere.
#
# Two facts from that block are worth keeping because they were verified and
# are still true: 5TCX is the CLOSED, cholesterol-bound conformer with 25 of
# its 89 EC2 residues in the lipid headgroup band, and its cholesterol touches
# no extracellular residue.

# ═══════════════════════════════════════════════════════════════════════════
# PREPARED TEMPLATE SET v2 — built 2026-08-12
# ═══════════════════════════════════════════════════════════════════════════
# Built by      code/tools/prepare_cd_templates.py --all
# Verified by   code/tools/check_cd_templates.py          (all five PASS)
# Frame check   code/tools/prepare_cd_templates.py --validate-frame
# Raw inputs    structures/raw/{pdb,alphafold,opm,uniprot}/   (kept untouched)
# Outputs       structures/prepared/*.pdb + *_provenance.json
#
# Each prepared file carries a REMARK 999 block naming every source entry,
# splice, reverted mutation and rebuilt atom, and encodes provenance in the
# OCCUPANCY column: 1.00 experimental primary / 0.75 experimental donor /
# 0.50 rebuilt side chain / 0.25 de novo backbone / 0.00 AlphaFold. So
#     awk '$1=="ATOM" && substr($0,55,6)+0 < 0.75' structures/prepared/X.pdb
# lists every atom of X that is NOT an experimental measurement — 1123 of 1786
# for CD63 (mostly the AlphaFold scaffold), 328 of 1610 for CD81, 36 of 1758
# for CD9. The B-factor column is carried through from
# the source files, so it is per-residue pLDDT over the AlphaFold part of CD63
# and deposited B elsewhere — MIXED UNITS in one column, which is precisely why
# check_plddt() must not be pointed at a prepared template unscoped. Invented
# atoms carry B = 0.
#
# WHY EACH TEMPLATE CHANGED
# --------------------------
# CD63  AlphaFold-only  ->  9HUR LEL (1.65 A) on the AlphaFold TM scaffold.
#       Two experimental CD63 LEL structures were released 2025-11-19; the
#       pipeline was scoring a prediction where a 1.65 A measurement exists.
#       Both are N130S/N150S/N172S glycan knockouts and both have been
#       reverted to wild type here. There is NO wild-type-sequence CD63
#       structure in the PDB at any resolution.
# CD81  5TCX (closed)   ->  7JIC (open) + the 5TCX arm kept as a control.
#       5TCX put 25 of 89 EC2 residues in the headgroup band and left EC1 57%
#       unresolved. 7JIC is the only open CD81 conformer and the only entry
#       with a complete EC1 (30/30).
# CD9   6K4J raw        ->  6K4J membrane frame + 6Z1V EC2 grafted on.
#       6K4J does not merely fail to resolve T175-K179: SEQADV lists them as
#       DELETION, they are absent from SEQRES, and the model welds E174 onto
#       S180 (CA-CA 3.79 A), which deforms the whole C/D apex 156-174 by
#       2.3-11.1 A. Those five residues sit 18-24 A above the hydrophobic
#       core - the most polymer-accessible point on CD9, and where three
#       independent antibodies bind. Every calculation this project has run
#       omitted the tip of its own target.
#
# WIRED INTO THE PIPELINE as of 2026-08-12 (this supersedes the "NOT WIRED,
# DELIBERATELY" note that stood here). utils_structure.download_structure now
# checks `prepared_path` FIRST and copies the chimera into the run directory —
# a chimera has no accession, so it can never be re-downloaded. If the key is
# present but the file is missing it raises rather than falling through to the
# raw deposition: a silent fallback would dock the very structures the
# prepared set exists to replace (closed CD81 with a quarter of its surface in
# lipid; CD9 missing T175-K179) and nothing downstream would record it.
# source/pdb_id/uniprot_id are LEFT IN PLACE — they still document where each
# template came from, and they remain the fallback for any target with no
# prepared_path (e.g. BSA).
#
# Two consequences that were handled at the same time, because wiring without
# them would have been worse than not wiring:
#   1. phase1 called check_plddt() whenever source == "alphafold". On a
#      prepared template the B-factor column is MIXED UNITS — pLDDT over the
#      AlphaFold scaffold, deposited crystallographic B over the spliced 9HUR
#      block — so the average is meaningless and would flag the 1.65 Å graft,
#      the best-determined part of the model, as low confidence. The check is
#      now scoped to genuinely predicted residues via the occupancy column
#      (only_predicted=True; 0.00 predicted / 0.50 rebuilt / 0.75 donor /
#      1.00 experimental).
#   2. the docking box is now defined on `scored_surface` rather than on the
#      whole ECL2 (RECEPTOR_BOX_ON_SCORED_SURFACE in config.py), so the
#      membrane-occluded residues are not searched. Leaving them in was a
#      cross-target confound, not a uniform inefficiency: 25/89 of the docked
#      CD81 surface was lipid-facing against 9/79 for CD9.
#
# THIS INVALIDATES results/phase1 ONWARD. The existing tree was produced from
# the raw depositions with a head-centred box and must be re-run before any
# number in it is quoted alongside these templates. Re-run it TOGETHER with
# the ensemble fix (ENSEMBLE_EXTRACT_REQUIRES_STABLE / ENSEMBLE_MERGE in
# config.py) — one re-run settles both, two settle them twice.
#
# REGRESSION NOTE: adding the keys below makes
# code/tests/test_config_regression.py::test_cd_unchanged (T1) fail, by design
# — T1 asserts the CD namespace is byte-identical to the committed baseline and
# this is a deliberate change to it. Refresh the contract with
#     python3 code/tools/config_baseline.py --write
# after reviewing the diff. T2-T10 are unaffected.

CD_TEMPLATE_SET_VERSION = "v2-2026-08-12"
CD_PREPARED_DIR = PROJECT_ROOT / "structures" / "prepared"
CD_TEMPLATES_WIRED = True         # see "WIRED INTO THE PIPELINE" above.
                                  # download_structure() reads prepared_path;
                                  # results/ predates this and must be re-run.

# EC1 POLICY, ONE RULE FOR ALL THREE TARGETS: EC1 is carried in the receptor as
# steric and electrostatic context and is EXCLUDED from scoring. The evidential
# quality of EC1 differs by an order of magnitude between the targets — CD9 has
# all 22 residues experimentally (2.70 A), CD81 all 30 (3.80 A trace, mean CA
# B 150.8), CD63 has none at all (AlphaFold, mean pLDDT 66.5, residues 44-48 at
# 43.8-51.3) — so scoring it would turn part of the selectivity signal into a
# readout of template provenance. Deleting it instead would leave an artificial
# cavity at the TM1/TM2 exit for monomers to score in.
#
# GLYCAN POLICY, ALSO ONE RULE: no glycans are modelled on any template; apply
# a shadow mask at the sequons instead. This is consistent but NOT neutral, and
# the bias is target-dependent: CD63 has 3 sequons on EC2, CD9 has 2 on EC1,
# CD81 has ZERO anywhere extracellular. Concretely — boronic-acid monomers
# (VPBA, AAPBA) will score identically on all three aglycosyl templates while
# being plausibly the most CD63-selective chemistry available, so a null result
# for that class from this screen is uninformative and must be reported as
# such rather than as a finding.

_CD_TEMPLATE_V2 = {
    "CD63": {
        # ---- prepared template -------------------------------------------
        "prepared_template": "CD63_open_v1",
        "prepared_path": PROJECT_ROOT / "structures" / "prepared" / "CD63_open_v1.pdb",
        "template_primary": "9HUR chain A (X-ray 1.65 A, released 2025-11-19) -> EC2 109-198",
        "template_donors": ["AF-P08962-F1 v6 (2025-08-01) -> 1-108, 199-238: TM1-4, "
                            "EC1, both stalk exits, and the membrane frame"],
        "template_conformer": "OPEN/UPRIGHT — but the LEL-TM hinge is AlphaFold's, "
                              "NOT experimental. No CD63 structure contains a TM helix, "
                              "a lipid, a detergent or an OPM entry, so nothing "
                              "constrains it. Transplanting the 9HUR LEL into the 5TCX "
                              "and 6K4J membrane frames brackets the occluded fraction "
                              "at 10-28%; report CD63 accessibility as a range.",
        "template_provenance_json": "structures/prepared/CD63_open_v1_provenance.json",
        # ---- membrane frame ----------------------------------------------
        "membrane_frame": "TM-SVD derived (no OPM entry exists for CD63); normal and "
                          "origin validated against OPM on 5TCX to 0.07 +/- 0.12 A, "
                          "thickness ASSERTED not fitted",
        "membrane_half_thickness_A": 15.5,
        # ---- residue ranges ----------------------------------------------
        "ec1_range": (33, 51),
        "ec1_scored": False,
        "ec2_experimental_range": (109, 198),   # the rest of EC2 is AlphaFold
        "predicted_residues_in_ec2": [103, 104, 105, 106, 107, 108,
                                      199, 200, 201, 202, 203],
        "membrane_occluded": [103, 104, 105, 106, 107, 108, 111,
                              199, 200, 201, 202, 203],
        "membrane_occluded_sensitivity": {15.3: 12, 15.7: 13, 17.1: 14},
        "scored_surface": [109, 110] + list(range(112, 199)),   # 89 of 101
        # ---- disulfides: PAIRING, not just a residue list ------------------
        "disulfide_pairs": [(145, 191), (146, 169), (170, 177)],
        "disulfide_pairing_citation": (
            "SSBOND records of PDB 9HUR (1.65 A, P41212) and 9HQ5 (2.60 A, P43212) — "
            "two independent crystal forms in different space groups, measured SG-SG "
            "2.02-2.05 A with the next-closest non-bonded SG pair at 7.66/7.70 A. "
            "UniProt P08962 carries NO DISULFID annotation, so the PDB entries are the "
            "only citable source. Independently reproduced by AlphaFold v6 (SG-SG "
            "2.03-2.04 A). CD63 has exactly 6 LEL cysteines; the 8-cysteine tetraspanin "
            "subclass does not apply."),
        # ---- glycans -------------------------------------------------------
        "glycans_modelled": False,
        "glycan_shadow_anchors": [130, 150, 172],
        "glycan_note": ("N130/N150/N172 were ENGINEERED TO SER in both 9HUR and 9HQ5 and "
                        "have been reverted here. The three Asn side chains are rebuilt "
                        "rotamers, not measurements — and S150 has no OG at all in 9HUR, "
                        "so there is no experimental side-chain information at that "
                        "position. They anchor the shadow mask, so their error "
                        "propagates into it."),
        "head_status": "RETIRED (nanovesicle imprinting); coverage on this template "
                       "157-172 = 16/16 modelled, 16/16 experimental",
    },
    "CD81": {
        "prepared_template": "CD81_open_v1",
        "prepared_path": PROJECT_ROOT / "structures" / "prepared" / "CD81_open_v1.pdb",
        "template_primary": "7JIC chain B (cryo-EM 3.80 A, CD19-CD81-Fab complex)",
        "template_donors": ["5M33 chain A (X-ray 1.28 A) -> EC2 137-140"],
        "template_conformer": "OPEN/UPRIGHT (experimental) but CD19-STABILISED. 7JIC's "
                              "LEL head is a conformational singleton: 13.0-15.8 A from "
                              "all 18 other CD81 LELs after stalk superposition, where "
                              "every other pair falls in 0.35-5.10 A. The vesicles carry "
                              "no CD19. 7JIC proves the open state is ACCESSIBLE; it "
                              "does not prove it is POPULATED without CD19, and no PDB "
                              "entry can settle that.",
        "template_provenance_json": "structures/prepared/CD81_open_v1_provenance.json",
        # The closed arm is MANDATORY, not optional: it is the only handle on the
        # conformational uncertainty above. Report CD81 as a range across the two.
        "conformer_arms": {
            "open": PROJECT_ROOT / "structures" / "prepared" / "CD81_open_v1.pdb",
            "closed": PROJECT_ROOT / "structures" / "prepared" / "CD81_closed_5tcx_v1.pdb",
        },
        "closed_arm_note": ("CD81_closed_5tcx_v1 (5TCX, cholesterol-bound) is a "
                            "WITHIN-CD81 sensitivity arm ONLY. Do NOT put it in the "
                            "cross-target contrast: it would make CD81 the one closed "
                            "template of three and the contrast would partly measure "
                            "that. Its EC1 is 13/30 and is left broken on purpose — "
                            "REMARK 465 removes D38-A54, the solvent-facing apex, and "
                            "building 17 de novo residues would be fabrication."),
        "membrane_frame": "OPM 7jic (DUM planes at z = +/-15.3 A); the opening is "
                          "intrinsic to the coordinates, not to OPM's placement — a "
                          "TM-only superposition onto 5TCX still drops the headgroup-band "
                          "count from 24/80 to 6/80",
        "membrane_half_thickness_A": 15.3,
        "ec1_range": (34, 63),
        "ec1_scored": False,
        "ec2_experimental_range": (113, 201),
        "predicted_residues_in_ec2": [116, 167, 168, 169, 170],
        "membrane_occluded": [113, 114, 115, 116, 117, 119,
                              196, 197, 198, 199, 200, 201],
        # 118 and 119 straddle the cut (CA within 0.04 A of it) and swap places
        # between rebuilds; the listing above is what the committed file has.
        "membrane_cut_boundary_residues": [118, 119],
        "membrane_occluded_closed_arm": [113, 114, 115, 116, 119, 123, 141, 142, 143,
                                         144, 145, 146, 148, 149, 150, 151, 152, 153,
                                         154, 177, 197, 198, 199, 200, 201],
        "scored_surface": [118] + list(range(120, 196)),        # 77 of 89
        "scored_surface_caution": [167, 168, 169, 170, 171, 172, 173],
        "scored_surface_caution_reason": (
            "167-170 are DE NOVO backbone (5M33 cannot donate: fitting on the flanks "
            "160-166+171-177 gives 4.4 A, so the 7JIC head really is rearranged there "
            "and a splice would import the wrong fold). 171-172 are reverted construct "
            "substitutions (SEQADV CONFLICT K171T/N172S; K171 is a surface lysine whose "
            "loss would have biased every acidic-monomer result). The whole block sits "
            "inside the former CD19 interface. Build it for sterics, exclude it from "
            "primary scoring, and report how many top-ranked monomers touch it."),
        "disulfide_pairs": [(156, 190), (157, 175)],
        "disulfide_pairing_citation": (
            "UniProt P60033 DISULFID 156..190 and 157..175 (CD81 IS annotated, unlike "
            "CD63), independently reproduced by SSBOND in all 16 human CD81 entries at "
            "SG-SG 1.97-2.09 A. Both bonds are ideal at 2.03 A in 7JIC, which is the "
            "strongest internal argument that its unusual head trace is not a "
            "modelling collapse."),
        "glycans_modelled": False,
        "glycan_shadow_anchors": [],
        "glycan_note": ("CD81 has ZERO extracellular N-glycan sequons. UniProt P60033 "
                        "lists no Glycosylation feature and the only N-X-S/T in the "
                        "236-aa sequence is the cytoplasmic N232; no carbohydrate "
                        "appears on any CD81 chain in any of the 20 surveyed structures. "
                        "Nothing is missing here — this is a real biological asymmetry "
                        "and makes CD81 the negative control on the glycan axis."),
        "head_status": "RETIRED (nanovesicle imprinting); coverage on this template "
                       "168-183 = 16/16 modelled, 13/16 experimental (168-170 de novo)",
    },
    "CD9": {
        "prepared_template": "CD9_open_v1",
        "prepared_path": PROJECT_ROOT / "structures" / "prepared" / "CD9_open_v1.pdb",
        "template_primary": "6K4J chain A (X-ray 2.70 A, lipidic cubic phase) — "
                            "membrane frame, TM1-4, EC1, and 112-113 + 191-195",
        "template_donors": ["6Z1V chain A (X-ray 1.33 A, nanobody 4E8) -> EC2 114-190"],
        "template_conformer": "OPEN/UPRIGHT, experimental, membrane-embedded, no protein "
                              "partner — the best-founded of the three, corroborated "
                              "independently at 8.6 A by EMD-11053 (CD9-EWI-F).",
        "template_provenance_json": "structures/prepared/CD9_open_v1_provenance.json",
        # The apex is demonstrably plastic, and it is the most exposed point on the
        # protein: 6Z1V/6Z20 give one D-loop and 6RLO another, differing by up to
        # 10.5 A at residue 172. Every compact CD9 LEL in the PDB has a binder sitting
        # on 175-178, and the one binder-free crystal (6RLR) came out domain-swapped
        # and extended. So the apex conformation is binder-shaped in EVERY source.
        "conformer_arms": {
            "apex_6z1v": PROJECT_ROOT / "structures" / "prepared" / "CD9_open_v1.pdb",
            "apex_6rlo": PROJECT_ROOT / "structures" / "prepared" / "CD9_open_v1b.pdb",
        },
        "apex_ensemble_note": ("Any CD9-selective monomer hit must survive BOTH arms. "
                               "If they agree the binder-shaped-loop worry is retired "
                               "cheaply; if they disagree, CD9 needs a larger apex "
                               "ensemble than two crystal conformers."),
        "membrane_frame": "OPM 6k4j (DUM planes at z = +/-15.7 A)",
        "membrane_half_thickness_A": 15.7,
        "ec1_range": (34, 55),
        "ec1_scored": False,
        "ec2_experimental_range": (112, 195),    # fully experimental, two entries
        "predicted_residues_in_ec2": [],
        "membrane_occluded": [112, 113, 114, 117, 141, 142, 145, 149,
                              192, 193, 194, 195],
        "scored_surface": [115, 116] + list(range(118, 141)) + [143, 144] + \
                          list(range(146, 149)) + list(range(150, 192)),   # 72 of 84
        "restored_residues": [175, 176, 177, 178, 179],
        "restored_residues_note": (
            "T175-F176-T177-V178-K179 were DELETED FROM THE 6K4J CONSTRUCT (SEQADV "
            "DELETION, absent from SEQRES, absent from REMARK 465) and are now present "
            "from 6Z1V. In this membrane frame they sit 18-24 A above the hydrophobic "
            "core — the single most polymer-accessible point on CD9, and precisely "
            "where nanobody 4E8, nanobody 4C8 and the AT1412dm Fab all bind."),
        "disulfide_pairs": [(152, 181), (153, 167)],
        "disulfide_pairing_citation": (
            "SSBOND records of PDB 6K4J, 6Z1V, 6Z20, 6RLO and 6RLR — all five agree at "
            "SG-SG 1.96-2.10 A. Every bond is INTRA-chain even in 6RLR, so that entry's "
            "domain swap is not a disulfide-shuffling artefact and does not undermine "
            "the pairing."),
        "glycans_modelled": False,
        "glycan_shadow_anchors": [52, 53],
        "glycan_note": ("N52/N53 sit on EC1 at CA heights 26.7 and 24.4 A, i.e. a low "
                        "collar around the base of the LEL rather than a second "
                        "presenting surface. CD9's mask has a WIDER positional error "
                        "than CD63's: N51 and N52 both lack CG/OD1/ND2 in 6K4J and the "
                        "only rotamer donor is AlphaFold at pLDDT 54.7-66 — its worst "
                        "EC1 stretch is exactly the sequon loop."),
        "head_status": "RETIRED (nanovesicle imprinting); coverage on this template "
                       "156-171 = 16/16 modelled, 16/16 experimental",
    },
}

# Merge into TARGETS without removing a single existing key. Every legacy key
# (uniprot_id, pdb_id, source, chain, full_length, ecl2_range, head_residues,
# head_candidates, n_glycan_sites, n_glycan_positions, ccg_position,
# disulfide_cys, description) keeps its current value, so the ~119 import sites
# and everything that describes results/ are untouched.
for _t, _extra in _CD_TEMPLATE_V2.items():
    TARGETS[_t].update(_extra)
del _t, _extra

# ── COMPARABILITY, AND THE FOUR CONFOUNDS THAT SURVIVE ─────────────────────
# Measured on one yardstick (fraction of modelled EC2 CA below hydrophobic
# core + 6 A, each in its own membrane frame):
#     CD63_open_v1          12/101 = 11.9%
#     CD81_open_v1          12/89  = 13.5%
#     CD9_open_v1(b)        12/84  = 14.3%
#     CD81_closed_5tcx_v1   25/89  = 28.1%   <- the arm, not the contrast
#
# The three primaries are now the same conformational class. Had 5TCX stayed
# primary, a cross-target contrast would have been measuring, more than
# anything else, that one of the three LELs was lying flat on the bilayer.
#
# THE MEMBRANE CUT IS A HARD THRESHOLD ON A CONTINUOUS QUANTITY, so residues
# sitting within ~1 A of core+6 A flip between rebuilds. Rebuilding the set
# twice moved CD81 118/119 across the line — their CA sit 0.04 and 0.02 A from
# the cut, i.e. a coin flip, and CD9 has 13 residues within 1.5 A of its cut
# (114,116,117,120,121,138,141,142,145,146,149,188,189). Treat the boundary
# residues as soft: they are listed in scored_surface as the build found them,
# not because 0.02 A of z is meaningful. Nothing downstream should hinge on a
# single boundary residue, and a monomer whose ranking does should be re-tested
# with the cut moved +/-1 A. Re-verify the lists against the files after any
# rebuild — the tail of code/tools/check_cd_templates.py prints them.
#
# 1. PROVENANCE OF "OPEN" IS NOT UNIFORM, and this is the serious one. CD9's
#    open state is experimental, from a full-length protein in a membrane with
#    no partner. CD81's is experimental but CD19-induced. CD63's is
#    AlphaFold's, which is not evidence. Same class, wildly different
#    confidence. Mitigation: the mandatory closed arm for CD81, and reporting
#    CD81 and CD63 as ranges.
# 2. GLYCANS: uniform treatment of a non-uniform reality (3 EC2 / 0 / 2 EC1).
#    The aglycosyl bias is target-dependent, not noise.
# 3. TEMPLATE EVIDENTIAL QUALITY cannot be equalised: CD63's scored EC2 is
#    1.65 A but carries 47 rebuilt atoms and 3 reverted engineered mutations at
#    the glycan anchors; CD81's is a 3.8 A trace with one spliced and one de
#    novo loop and two reverted substitutions; CD9's is 1.33 A with a
#    binder-shaped apex and a ~2 A graft seam. PRACTICAL RULE THAT FOLLOWS:
#    never compare a raw docking score across targets. Compare monomer RANK
#    ORDER within a target, and compare targets only through own-vs-cross
#    RATIOS, where the per-template quality term partially cancels.
# 4. MEMBRANE-FRAME PROVENANCE: CD81 and CD9 have real OPM fits; CD63's normal
#    is derived and its thickness asserted.
#
# NON-STRUCTURAL, BUT IT BELONGS HERE. The NTA mode diameters differ (CD63 115,
# CD81 105, CD9 139, NC 102 nm). Assuming identical surface density cancels in
# the own-vs-cross contrast, as intended — but it does NOT cancel in anything
# computed per vesicle: a 139 nm CD9 vesicle has ~1.75x the surface of a 105 nm
# CD81 one. If any downstream quantity is normalised per vesicle rather than
# per unit area, or if curvature enters the imprinting geometry, that asymmetry
# re-enters the comparison independently of every structural point above.
#
# ── STILL OPEN ─────────────────────────────────────────────────────────────
# * Is CD81 open or closed on a CD19-free vesicle? No PDB entry can say. The
#   cheapest decisive evidence would be cryo-ET / subtomogram averaging of
#   tetraspanin-positive vesicles — and note that BOTH structure surveys
#   reported the EMDB free-text search API returning empty for every query
#   syntax tried, so map-only depositions of exactly that kind are the one
#   class of data neither survey could rule out.
# * Which cell line did the collaborator use for the overexpression? HEK293 vs
#   HEK293 GnTI- vs CHO changes CD63 from three complex biantennary (possibly
#   sialylated) glycans to three Man5 stubs, which changes both the shadow-mask
#   radius and whether boronic-acid or cationic monomers have anything to bind.
#   Free to ask, and it gates the whole stage-2 glycan decision.
# * Are further CD63 depositions embargoed? 9HQ5/9HUR/9HT6/9HV7 were released
#   together from one unpublished study (Nagarathinam/Krey, "Evolutionary
#   relationship of tetraspanins", TO BE PUBLISHED). A wild-type-sequence CD63
#   LEL from that project would remove the worst repair in this set — reverting
#   a triple glycan knockout. Only 19,291 of 35,007 unreleased PDB entries
#   expose a title, so a title-suppressed entry is invisible to search. Re-run
#   the P08962 accession query before committing to these templates.
# * Should the NC (no-overexpression, 102 nm) vesicle have a structural
#   counterpart? It is the only experimental control on non-specific polymer
#   binding and currently has no computational one. A bare-lipid reference
#   would separate "binds CD63" from "binds membrane" — a different question
#   from own-vs-cross, and one none of these three templates answers.

# ═══════════════════════════════════════════════════════════════════════════
# SOURCE OF THE VESICLES, AND WHAT IT FIXES ABOUT THE GLYCANS
# ═══════════════════════════════════════════════════════════════════════════
# The templates are no longer synthetic peptides. The lab receives nanovesicles
# from a collaborating lab, each CD protein overexpressed in a lipid bilayer,
# and imprints the intact vesicle.
#
# PROVENANCE OF THIS ENTRY: a project DECISION taken 2026-08-13, not a
# measurement and not a document from the producing lab. The line was given
# verbally, first as "G293T" and then settled as HEK293T; no catalogue number
# was supplied. Recorded as an ASSUMPTION so that a later contradiction is
# visible rather than silently absorbed. If the producing lab ever supplies an
# ATCC/supplier number, replace this block with it.
EV_SOURCE_CELL_LINE = "HEK293T"           # ASSUMED — see provenance note above
EV_SOURCE_CATALOGUE = None                # not supplied; HEK293T is ATCC CRL-3216
#
# WHY THE LINE DECIDES THE GLYCAN CHEMISTRY. The "T" in 293T is SV40 large T
# antigen — an episomal-replication marker that has nothing to do with
# glycosylation. It is NOT GnTI, and 293T is NOT a glyco-mutant. The glyco-
# engineered line is a different cell, HEK293S GnTI-minus (Reeves et al., PNAS
# 2002), in which N-acetylglucosaminyltransferase I is knocked out so processing
# stalls at Man5GlcNAc2. On HEK293T the pathway runs to completion:
#     HEK293T          complex-type biantennary, commonly sialylated
#     HEK293S GnTI-    Man5GlcNAc2 only, neutral, small
EV_GLYCAN_TYPE = "complex_biantennary"    # follows from HEK293T
EV_GLYCAN_SIALYLATED = True               # ASSUMED; degree not measured
#
# CONSEQUENCE FOR THE SCREEN, and it is the largest discriminator this project
# has. CD63 carries three N-glycan sequons on the imprinted face (N130/N150/
# N172); CD81 carries ZERO extracellular sequons; CD9 carries two, and both sit
# on EC1 down at the membrane collar. With complex sialylated glycans that is
# not a subtle difference in residue composition — it is three bulky, ANIONIC,
# diol-rich trees present on one target and absent on another. Boronate
# monomers (APBA/AAPBA/VPBA/FPBA) bind cis-diols, so this is presence-versus-
# absence of their binding partner, not a few percent of contact frequency.
# It is why the recommended reframing is CD63 versus a pooled off-target rather
# than three-way discrimination.
#
# STILL UNMODELLED. glycans_modelled is False on all three targets and no
# available structure supplies coordinates. Complex biantennary glycans are
# large and conformationally heterogeneous, so building them is real work — see
# Route P in the plan. Until then the boronate monomers score identically on
# three aglycosyl templates and a null result from that class is uninformative.
#
# CAVEATS TO CARRY. EV-surface glycosylation can differ from the parent cell
# surface, and an overexpressed protein may be incompletely processed, so the
# real vesicle may present a mixture rather than uniform complex biantennary.
# Resolving that needs lectin blotting or glycan analysis and is out of scope
# now; it is recorded here so the assumption is not mistaken for a measurement.
