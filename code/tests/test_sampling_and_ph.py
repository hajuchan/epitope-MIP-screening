#!/usr/bin/env python3
"""
Sampling, decision-rule and pH-realism tests (2026-08 sampling audit)
====================================================================
One test per finding the audit raised, each asserting the FIXED behaviour and
naming the measurement that motivated it.  No MD is run; the two tests that
touch GROMACS are skipped when `gmx` is absent.

    python3 code/tests/test_sampling_and_ph.py     # standalone
    pytest code/tests/test_sampling_and_ph.py      # or under pytest

S1  GRID RESUME GUARD    — editing BSA_RATIO_GRID and re-running into the same
                           tree used to return the PREVIOUS grid in 0.0 s and
                           report it complete: the per-TARGET skip ran before the
                           grid was consulted. It now raises on a fingerprint
                           mismatch, and still resumes an unchanged grid.
S2  NO QUICK OVERRIDE    — `quick=(time_ns <= 20)` silently replaced any leg at
                           or below 20 ns with MD_QUICK_NS (=5 under BSA) while
                           the REQUESTED value was what got recorded. quick is
                           now always False from this path, and a leg whose
                           effective length differs from the request raises.
S2b MM-GBSA PRE-FLIGHT   — the window is checked against the leg length BEFORE
                           the first leg, not discovered afterwards.
S3  ZERO CONTACT IS DATA — a PRESENT species with zero contacts is a measured
                           zero, not a failed analysis. It no longer rejects the
                           replica (which biased rejection toward the TEOS-rich
                           end); an all-zero box still does.
S4  pH IS LIVE           — titration_model reproduces the audit's numbers on BSA
                           (-16.65 e at 7.4, -28.27 e at 9.5), excludes the 34
                           disulfide-bridged cysteines, and separates the HH
                           continuum charge from the charge a fixed-charge force
                           field can actually build.
S4b pH PRE-FLIGHT GATE   — Phase 4 refuses to start while the fixed-charge
                           residual is unacknowledged.
S5  ACPYPE NET CHARGE    — the hardcoded `-n 0` is gone; the net charge comes
                           from the library SMILES' formal charge, so the
                           ammonium form of APTES is expressible.
S6  POLCA Si LJ PINNED   — eps 0.108 is kJ/mol per Jorge et al. 2021 Table 6.
                           It is NOT the un-converted kcal value it resembles;
                           pinned so nobody "fixes" it by multiplying by 4.184.
S7  ANALYSIS WINDOW      — defaults to the last 50%, not the last 25%.
S8  SAMPLING STATS       — tau_int recovers a known correlation time; the CI
                           helper refuses to invent an error bar at n=1.
S9  TIE RULE             — a 12% margin at R=3 is a TIE under the CI rule and
                           was a unique "winner" under the retired float-equality
                           guard; a large, consistent margin still qualifies.
S10 BLOCK MEANS          — recorded, and explicitly flagged as not an error bar.
S12 LOADING GATE         — the loading axis will not follow an unqualified
                           argmax.
S13 STAGED REPLICATION   — asking for more replicas EXTENDS a completed grid
                           point instead of resuming it, so 'screen at R=4, top
                           the survivors up to R=10' costs only the new legs.
S14 METRIC KEYING        — pH speciation renames the species in the box, so the
                           ranking metric is keyed on the pooled record's own
                           present-species list and as-built composition, never
                           on the caller's nominal argument list.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent
sys.path.insert(0, str(_CODE_DIR))

import pipeline.phase4_md_validation as p4          # noqa: E402
import pipeline.utils_structure as us               # noqa: E402
import pipeline.utils_gromacs as ug                 # noqa: E402

_BSA_PDB = None
for _cand in (Path("/tmp/claude-1000/-home-chan-Research-Monomer-screening-in-Bio"
                   "/7c469218-1b27-4230-b8ca-cb430896a81e/scratchpad/bsa_p1"
                   "/phase1/BSA/BSA_ecl2.pdb"),
              _CODE_DIR.parent / "structures" / "raw" / "4F5S.pdb"):
    if _cand.exists():
        _BSA_PDB = _cand
        break


# ── S1 ──────────────────────────────────────────────────────────
def test_per_target_resume_refuses_a_changed_grid():
    """The enclosing per-target skip is now fingerprinted like the per-point one."""
    grid_a = [(10, 2), (2, 10)]
    grid_b = [(8, 4), (4, 8)]
    pts_a = p4._grid_points_to_copies(grid_a, "TEOS", ["APTES"], 100)
    pts_b = p4._grid_points_to_copies(grid_b, "TEOS", ["APTES"], 100)
    fp_a = p4._grid_points_fingerprint(pts_a)
    fp_b = p4._grid_points_fingerprint(pts_b)
    assert fp_a != fp_b

    # Fingerprint is order-independent — a reordered grid is the same grid.
    assert p4._grid_points_fingerprint(
        p4._grid_points_to_copies(list(reversed(grid_a)), "TEOS",
                                  ["APTES"], 100)) == fp_a

    recorded = {"BSA_TEOS_APTES": {"ratio_grid": {
        "grid_fingerprint": fp_a,
        "functional_monomers": ["APTES"], "crosslinker": "TEOS",
        "per_point_summary": {p["label"]: {} for p in pts_a}}}}

    _cfg = p4._ratio_grid_from_config
    _en = p4._ratio_grid_enabled
    try:
        p4._ratio_grid_enabled = lambda: True

        # (a) declared grid == recorded grid -> resume is allowed
        p4._ratio_grid_from_config = lambda: {
            "grid": grid_a, "max_monomers_in_shell": 100,
            "source_keys": ["BSA_RATIO_GRID"]}
        p4._guard_resumed_target_against_grid("BSA", recorded)

        # (b) declared grid != recorded grid -> RAISES (was: silent, 0.0 s,
        #     "RESUMED — already done", old grid reported as complete)
        p4._ratio_grid_from_config = lambda: {
            "grid": grid_b, "max_monomers_in_shell": 100,
            "source_keys": ["BSA_RATIO_GRID"]}
        try:
            p4._guard_resumed_target_against_grid("BSA", recorded)
            raise AssertionError("a changed grid was silently resumed")
        except RuntimeError as e:
            assert "RESUME REFUSED" in str(e)
            assert "TEOS8_APTES4" in str(e) and "TEOS10_APTES2" in str(e)

        # (c) a recorded entry with NO ratio_grid block while the grid is on
        try:
            p4._guard_resumed_target_against_grid(
                "BSA", {"BSA_TEOS_APTES": {"pc_id": "BSA_TEOS_APTES"}})
            raise AssertionError("a single-composition leg was resumed as a grid")
        except RuntimeError as e:
            assert "no ratio_grid block" in str(e)
    finally:
        p4._ratio_grid_from_config = _cfg
        p4._ratio_grid_enabled = _en
    print("S1 PASS  per-target resume is fingerprinted against the declared grid")


# ── S2 ──────────────────────────────────────────────────────────
def test_leg_length_is_not_silently_overridden():
    src = Path(p4.__file__).read_text()
    # Check the CODE, not the prose: the retired expression is quoted verbatim
    # in the module docstring and in the comment that explains the fix, so a
    # plain substring search on the file would always fire.
    import ast
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "quick" and not (
                    isinstance(kw.value, ast.Constant) and kw.value.value is False):
                bad.append((getattr(node.func, "id", None)
                            or getattr(node.func, "attr", None),
                            node.lineno, ast.unparse(kw.value)))
    assert not bad, \
        f"a `quick=` argument is computed rather than pinned to False: {bad} — " \
        f"any leg <= 20 ns would silently become MD_QUICK_NS again"
    assert "md_kwargs = dict(time_ns=time_ns, quick=False," in src
    # and the mismatch is a hard error, not a note
    assert "record a leg under a length it did not run" in src
    print("S2 PASS  quick is never inferred from time_ns; effective length asserted")


def test_mmpbsa_window_preflight():
    _get = p4._cfg_get
    try:
        p4._cfg_get = lambda n, d=None: {"MD_MMPBSA_START_NS": 20,
                                         "MD_MMPBSA_END_NS": 30}.get(n, d)
        assert p4._mmpbsa_window_status(30)["fits"] is True
        bad = p4._mmpbsa_window_status(10)
        assert bad["fits"] is False
        assert "does not fit" in bad["reason"]
        assert bad["suggested_start_ns"] == 6.667 and bad["suggested_end_ns"] == 10
    finally:
        p4._cfg_get = _get
    print("S2b PASS  MM-GBSA window is checked before the first leg")


# ── S3 ──────────────────────────────────────────────────────────
def test_zero_contact_species_is_a_measurement_not_a_rejection():
    """A scarce species with no contacts must not reject the whole grid point."""
    # Shape of the record the convergence block now emits.
    conv_ok_with_zero = {
        "converged": True, "monomers_tested_for_drift": ["TEOS"],
        "monomers_drifting": [], "monomers_without_contacts": ["APTES"],
        "window_diff_pct": {"TEOS": 0.0, "APTES": None},
        "tolerance_pct": 10.0, "zero_contact_is_a_measurement": True,
        "block_means_are_not_an_error_bar": True, "reason": None}
    res = {"md_completed": True,
           "occupancy_analysis": {"convergence": conv_ok_with_zero},
           "rmsd_drift_q3q4_nm": 0.0001}
    with tempfile.TemporaryDirectory() as td:
        # rmsd.xvg absent -> that criterion fails; check ONLY the converged one.
        acc = p4._replica_acceptance(res, Path(td))
    assert acc["criteria"]["converged"] is True, acc
    assert "converged" not in acc["failed_criteria"], acc["failed_criteria"]
    assert any("measured zero" in n for n in acc["notes"]), acc["notes"]

    # A DEAD box (nothing anywhere contacted the protein) is still rejected.
    conv_dead = {"converged": None,
                 "reason": "no monomer species recorded a single contact in "
                           "either comparison window",
                 "monomers_without_contacts": ["TEOS", "APTES"]}
    res2 = {"md_completed": True,
            "occupancy_analysis": {"convergence": conv_dead}}
    with tempfile.TemporaryDirectory() as td:
        acc2 = p4._replica_acceptance(res2, Path(td))
    assert acc2["criteria"]["converged"] is False
    assert "converged" in acc2["failed_criteria"]

    # A DRIFTING species is still rejected.
    conv_drift = {"converged": False, "monomers_drifting": ["APTES"],
                  "monomers_without_contacts": [],
                  "window_diff_pct": {"APTES": 73.0}, "tolerance_pct": 10.0}
    res3 = {"md_completed": True,
            "occupancy_analysis": {"convergence": conv_drift}}
    with tempfile.TemporaryDirectory() as td:
        acc3 = p4._replica_acceptance(res3, Path(td))
    assert "converged" in acc3["failed_criteria"]
    print("S3 PASS  measured zero != failed analysis; dead and drifting still fail")


# ── S4 ──────────────────────────────────────────────────────────
def test_ph_titration_model_on_bsa():
    if _BSA_PDB is None:
        print("S4 SKIP  no BSA structure available")
        return
    ss = us.detect_disulfide_cysteines(_BSA_PDB)
    assert len(ss) == 34, f"expected 34 bridged cysteines in BSA, got {len(ss)}"

    m74 = us.titration_model(_BSA_PDB, 7.4)
    m95 = us.titration_model(_BSA_PDB, 9.5)
    # The audit's own Henderson-Hasselbalch numbers, reproduced.
    assert abs(m74["hh_continuum_charge"] - (-16.65)) < 0.05, m74["hh_continuum_charge"]
    assert abs(m95["hh_continuum_charge"] - (-28.27)) < 0.05, m95["hh_continuum_charge"]
    # ... and the three charges are kept DISTINCT, which is the whole point.
    assert m95["discrete_charge"] == -19, m95["discrete_charge"]
    assert m95["hh_continuum_charge"] != m95["discrete_charge"]
    assert abs(m95["charge_residual_vs_hh"] + 9.27) < 0.05

    # With the engine's real limits (no neutral N-terminus in this ff port).
    m95ff = us.titration_model(
        _BSA_PDB, 9.5,
        unavailable_states={("NTERM", "deprotonated"), ("CTERM", "protonated")})
    assert m95ff["representable_charge"] == -18, m95ff["representable_charge"]
    assert len(m95ff["unrepresentable_sites"]) == 1
    print("S4 PASS  pH model: HH -16.65 / -28.27 e, buildable -18 e, 34 SS-Cys")


def test_ph_preflight_gates_the_grid():
    if _BSA_PDB is None:
        print("S4b SKIP  no BSA structure available")
        return
    _get = p4._cfg_get
    try:
        p4._cfg_get = lambda n, d=None: {
            "MD_SOLVENT_PH": 9.5, "MD_PH_CHARGE_RESIDUAL_TOL_E": 2.0,
            "MD_PH_CHARGE_RESIDUAL_ACK": False}.get(n, d)
        try:
            p4._ph_preflight(_BSA_PDB)
            raise AssertionError("the pH pre-flight let an unacknowledged "
                                 "10 e charge shortfall through")
        except RuntimeError as e:
            assert "pH PRE-FLIGHT REFUSED THE RUN" in str(e)
            assert "-28.27" in str(e)

        p4._cfg_get = lambda n, d=None: {
            "MD_SOLVENT_PH": 9.5, "MD_PH_CHARGE_RESIDUAL_TOL_E": 2.0,
            "MD_PH_CHARGE_RESIDUAL_ACK": True}.get(n, d)
        rec = p4._ph_preflight(_BSA_PDB)
        assert rec["acknowledged"] is True
        assert rec["will_be_simulated_charge_e"] == -18
    finally:
        p4._cfg_get = _get
    print("S4b PASS  Phase 4 refuses to start on an unacknowledged pH residual")


# ── S5 ──────────────────────────────────────────────────────────
def test_acpype_net_charge_comes_from_the_smiles():
    src = Path(ug.__file__).read_text()
    assert '"-n", "0",' not in src, "acpype's net charge is hardcoded to 0 again"
    assert '"-n", str(int(net_charge))' in src

    class _Stub(dict):
        pass
    _all = None
    import pipeline.config as cfg
    _all = getattr(cfg, "ALL_MONOMERS", None)
    try:
        cfg.ALL_MONOMERS = {
            "APTES_H": {"smiles": "[NH3+]CCC[Si](O)(O)O"},
            "APTES":   {"smiles": "NCCC[Si](O)(O)O"},
            "TEOS":    {"smiles": "O[Si](O)(O)O"},
        }
        assert ug._monomer_formal_charge("APTES_H") == 1
        assert ug._monomer_formal_charge("APTES") == 0
        assert ug._monomer_formal_charge("TEOS") == 0
        assert ug._monomer_formal_charge("NOT_IN_LIBRARY") == 0
    finally:
        if _all is not None:
            cfg.ALL_MONOMERS = _all
    print("S5 PASS  acpype net charge is the SMILES formal charge, not 0")


# ── S6 ──────────────────────────────────────────────────────────
def test_polca_si_lj_is_pinned_to_the_published_table():
    """Jorge et al., ACS Phys. Chem. Au 2021, 1(1), 54-69, Table 6.

    Verbatim: 'atom  sigma (nm)  eps (kJ/mol)' / Si0 0.580 0.108 / Si1 0.551
    0.108 / Si2 0.522 0.108 / Si3 0.493 0.108 / Si4 0.464 0.108, and in the body
    'This returned values of sigma = 0.58 nm and eps = 0.108 kJ/mol for the Si
    atom in alkylsilanes.'  The value LOOKS like GAFF2 c3's 0.1078 kcal/mol
    (parmchk writes 'Si 1.9069 0.1078 same as c3') but is not: this test exists
    so a future reader does not multiply it by 4.184.
    """
    want = {"Si0": 0.580, "Si1": 0.551, "Si2": 0.522, "Si3": 0.493, "Si4": 0.464}
    for k, sigma in want.items():
        assert abs(ug._POLCA_SI_LJ[k]["sigma"] - sigma) < 1e-9, k
        assert abs(ug._POLCA_SI_LJ[k]["eps"] - 0.108) < 1e-9, k
    # The paper's own sigma rule, and it is CUMULATIVE-LINEAR, not compounding:
    # "we reduced the value of sigma for silicon by 5% for each oxygen-containing
    # substituent group -- e.g. the value of sigma was scaled by 20% for
    # tetramethoxysilane, which has 4 alkoxy groups". 20%, not 1-0.95^4 = 18.5%.
    for i, k in enumerate(("Si0", "Si1", "Si2", "Si3", "Si4")):
        assert abs(ug._POLCA_SI_LJ[k]["sigma"] - 0.580 * (1 - 0.05 * i)) < 1e-9, k
    src = ug._POLCA_SI_LJ_SOURCE
    assert src["eps_units"] == "kJ/mol"
    assert "acsphyschemau.1c00014" in src["reference"]
    # the kcal reading would be 0.108 * 4.184 = 0.4518; assert it is NOT that
    assert abs(ug._POLCA_SI_LJ["Si4"]["eps"] - 0.108 * 4.184) > 0.3
    print("S6 PASS  PolCA Si eps 0.108 kJ/mol pinned to Table 6 (not a unit error)")


# ── S7 ──────────────────────────────────────────────────────────
def test_analysis_window_defaults_to_the_last_half():
    assert p4._CFG_DEFAULTS["PHASE4_ANALYSIS_WINDOW_FRACTION"] == 0.50
    assert abs(p4._analysis_window_fraction() - 0.50) < 1e-9
    src = Path(p4.__file__).read_text()
    assert "start_frame = (3 * n_frames) // 4 if use_q4 else n_frames // 2" not in src
    assert "start_frame = int(n_frames * (1.0 - window_fraction))" in src
    print("S7 PASS  analysis window is the last 50% (tau_int is 8-33 ns; Q4 of a "
          "30 ns leg was under one correlation time)")


# ── S8 ──────────────────────────────────────────────────────────
def test_sampling_statistics():
    import numpy as np
    rng = np.random.default_rng(0)
    # White noise -> tau ~ 1 frame.
    assert p4._tau_int(rng.standard_normal(4000)) < 2.0
    # AR(1) with phi=0.9 -> tau = (1+phi)/(1-phi) = 19 frames.
    x = np.zeros(20000)
    for i in range(1, len(x)):
        x[i] = 0.9 * x[i - 1] + rng.standard_normal()
    tau = p4._tau_int(x)
    assert 12 < tau < 26, tau

    # n=1 must NOT produce an error bar.
    ci1 = p4._mean_ci([0.4])
    assert ci1["ci_lo"] is None and ci1["ci_is_degenerate"] is True
    ci3 = p4._mean_ci([0.10, 0.12, 0.14])
    assert ci3["n"] == 3 and ci3["ci_lo"] < 0.12 < ci3["ci_hi"]
    print(f"S8 PASS  tau_int recovers AR(1) tau=19 as {tau:.1f}; n=1 has no CI")


# ── S9 ──────────────────────────────────────────────────────────
def _pt(label, value, replicas, x):
    return {"status": "done", "accepted": True, "label": label,
            "x_functional": x,
            "metrics": {"functional_contact_per_copy": value,
                        "functional_contact_per_copy_replicas": replicas,
                        "functional_contact_per_copy_ci": p4._mean_ci(replicas)}}


def test_tie_rule_is_a_ci_not_float_equality():
    # A 12% margin — the MEASURED median winning margin under a flat truth at
    # R=3 was 12.4% — with overlapping replicate spread. Retired guard: unique
    # winner (0.1120 != 0.1000). New guard: a tie.
    per_point = {
        "A": _pt("A", 0.1120, [0.100, 0.112, 0.124], 0.20),
        "B": _pt("B", 0.1000, [0.090, 0.100, 0.110], 0.33),
        "C": _pt("C", 0.0900, [0.080, 0.090, 0.100], 0.50),
    }
    r = p4._rank_grid_points(per_point, "functional_contact_per_copy")
    assert r["best_point"] == "A"
    assert r["best_point_is_tied"] is True, r
    assert set(r["tied_points"]) >= {"A", "B"}, r["tied_points"]
    assert r["best_point_qualified"] is False
    assert r["decision"] in ("flat", "trend_only"), r["decision"]
    # the retired rule would have called this a unique winner
    assert len([l for l, e in per_point.items()
                if e["metrics"]["functional_contact_per_copy"] == 0.1120]) == 1

    # A large, consistent margin still qualifies.
    per_point2 = {
        "A": _pt("A", 0.300, [0.295, 0.300, 0.305], 0.20),
        "B": _pt("B", 0.100, [0.095, 0.100, 0.105], 0.33),
        "C": _pt("C", 0.090, [0.085, 0.090, 0.095], 0.50),
    }
    r2 = p4._rank_grid_points(per_point2, "functional_contact_per_copy")
    assert r2["best_point"] == "A" and r2["best_point_qualified"] is True, r2
    assert r2["decision"] == "optimum"
    assert r2["margin_over_runner_up_pct"] > 30.0

    # R=1: no error bar exists, so NOTHING can be separated.
    per_point3 = {
        "A": _pt("A", 0.300, [0.300], 0.20),
        "B": _pt("B", 0.100, [0.100], 0.33),
    }
    r3 = p4._rank_grid_points(per_point3, "functional_contact_per_copy")
    assert r3["best_point_qualified"] is False, r3
    assert r3["best_point_is_tied"] is True, r3
    print("S9 PASS  ties are declared by CI+MDD, not by float equality; R=1 "
          "cannot rank")


# ── S10 ─────────────────────────────────────────────────────────
def test_block_means_are_labelled_as_not_an_error_bar():
    src = Path(p4.__file__).read_text()
    assert '"block_means_are_not_an_error_bar": True' in src
    # the value the audit measured, kept in the code so the rationale cannot
    # drift away from the flag
    assert "2.8x" in src and "7.5x" in src
    # and the emitted record carries the flag itself
    assert '"block_means": window_freqs,' in src
    print("S10 PASS  4-block means are a drift diagnostic only (they understate "
          "the true SD by a median 2.8x)")


# ── S12 ─────────────────────────────────────────────────────────
def test_loading_sweep_requires_a_pinned_or_qualified_composition():
    src = Path(p4.__file__).read_text()
    assert "PHASE4_LOADING_SWEEP_REQUIRE_QUALIFIED_WINNER" in src
    assert "BSA_LOADING_SWEEP_COMPOSITION" in src
    assert "LOADING SWEEP SKIPPED" in src
    assert p4._CFG_DEFAULTS[
        "PHASE4_LOADING_SWEEP_REQUIRE_QUALIFIED_WINNER"] is True
    # and the driver itself refuses an empty composition with the new message
    try:
        p4.run_phase4_loading_sweep(
            target="BSA", pc_id="X", functional_monomers=["APTES"],
            crosslinker="TEOS", epitope_pdb="x.pdb", work_dir="/tmp/x",
            winning_copies={}, loadings=[30])
        raise AssertionError("the loading sweep ran with no composition")
    except ValueError as e:
        assert "ONE FIXED composition" in str(e)
    print("S12 PASS  the loading axis will not follow an unqualified argmax")


# ── S13 ─────────────────────────────────────────────────────────
def test_grid_extends_a_point_when_more_replicas_are_requested():
    """The staged design: screen at R=2, top the survivors up to R=4.

    Replica seeds are sha256(target|pc_id@label|rep<i>), so reps 0..k-1 keep
    their seeds and their manifests, `_check_resume` reuses their trajectories,
    and only the NEW replicas cost GPU time. Without this the per-point resume
    short-circuits on the recorded run and a staged design is impossible.
    """
    calls = []

    def fake_md(**kw):
        calls.append((kw["pc_id"], kw["replica_index"]))
        return {"replica_index": kw["replica_index"], "md_completed": True,
                "accepted": True, "built_copies": dict(kw["copies"]),
                "occupancy_analysis": {"contact_freq_6A": {"TEOS": 1.0,
                                                           "APTES": 0.4}},
                "work_dir": str(kw["work_dir"])}

    def fake_pool(target, pc_id, func, xl, reps, time_ns, pc_dir, out_dir):
        return {"target": target, "pc_id": pc_id, "accepted": True,
                "n_replicas": len(reps), "n_replicas_accepted": len(reps),
                "pooled_composition": dict(reps[0]["built_copies"]),
                "replicas": [{"accepted": True, "effective_time_ns": time_ns,
                              "contact_freq_6A": {"TEOS": 1.0, "APTES": 0.4}}
                             for _ in reps],
                "occupancy_analysis": {"contact_freq_6A": {"TEOS": 1.0,
                                                           "APTES": 0.4}}}

    _md, _pool, _pre = (p4._run_prepolymerization_md, p4._pool_replicas,
                        p4._ph_preflight)
    _cfgfn = p4._ratio_grid_from_config
    try:
        p4._run_prepolymerization_md = fake_md
        p4._pool_replicas = fake_pool
        p4._ph_preflight = lambda pdb: {"status": "stubbed"}
        # isolate from BSA_RATIO_TOTAL_MONOMERS, which is an assertion on the
        # PRODUCTION grid and would refuse this 12-monomer toy point
        p4._ratio_grid_from_config = lambda: {"grid": [(8, 4)],
                                              "max_monomers_in_shell": 100,
                                              "source_keys": ["TEST"]}
        with tempfile.TemporaryDirectory() as td:
            kw = dict(target="BSA", pc_id="P", functional_monomers=["APTES"],
                      crosslinker="TEOS", epitope_pdb="x.pdb",
                      work_dir=td, grid=[(8, 4)], time_ns=30,
                      total_monomers=None, max_monomers_in_shell=100)
            p4.run_phase4_ratio_grid(n_replicas=2, **kw)
            assert len(calls) == 2, calls
            calls.clear()
            # same R -> pure resume, no MD at all
            p4.run_phase4_ratio_grid(n_replicas=2, **kw)
            assert calls == [], calls
            # more R -> only the NEW replicas run
            p4.run_phase4_ratio_grid(n_replicas=4, **kw)
            assert len(calls) == 4, calls
            assert [i for _, i in calls] == [0, 1, 2, 3], calls
            rec = json.loads((Path(td) / "TEOS8_APTES4" /
                              p4._GRID_POINT_NAME).read_text())
            assert rec["n_replicas"] == 4
    finally:
        p4._run_prepolymerization_md, p4._pool_replicas = _md, _pool
        p4._ph_preflight = _pre
        p4._ratio_grid_from_config = _cfgfn
    print("S13 PASS  requesting more replicas EXTENDS a completed grid point "
          "(reps 0..k-1 are reused from disk)")


# ── S14 ─────────────────────────────────────────────────────────
def test_metric_is_keyed_on_the_box_not_the_argument_list():
    """pH speciation renames the species in the box; the metric must follow it.

    Caught on a real 100 ps BSA leg: the box held APTESH (the pH-9.5 ammonium)
    in contact with the protein in every frame, and
    functional_contact_per_copy came out 0.0000 because it was summed over the
    nominal ['APTES'] and divided by the REQUESTED copies. Keyed on the pooled
    record's own present-species list and as-built composition it is 0.25.
    """
    pooled = {
        "functional_monomers": ["APTESH"], "crosslinker": "TEOS",
        "pooled_composition": {"APTESH": 4, "TEOS": 8},
        "replicas": [],
        "occupancy_analysis": {"contact_freq_6A": {"APTESH": 1.0, "TEOS": 0.038}},
    }
    # the WRONG keying — nominal names, requested copies — reproduces the bug
    bad = p4._grid_point_metrics(pooled, {"TEOS": 8, "APTES": 4},
                                 ["APTES"], "TEOS")
    assert bad["functional_contact_per_copy"] == 0.0
    assert bad["contact_freq_per_copy"]["APTESH"] is None
    # the production keying
    good = p4._grid_point_metrics(pooled, pooled["pooled_composition"],
                                  pooled["functional_monomers"], "TEOS")
    assert good["functional_contact_per_copy"] == 0.25, good
    assert good["contact_freq_per_copy"] == {"APTESH": 0.25, "TEOS": 0.0047}
    # and the driver passes the production keying
    src = Path(p4.__file__).read_text()
    assert 'pooled.get("pooled_composition") or copies' in src
    assert 'pooled.get("functional_monomers") or functional_monomers' in src
    print("S14 PASS  the ranking metric follows the species actually in the box")


_TESTS = [
    test_per_target_resume_refuses_a_changed_grid,
    test_leg_length_is_not_silently_overridden,
    test_mmpbsa_window_preflight,
    test_zero_contact_species_is_a_measurement_not_a_rejection,
    test_ph_titration_model_on_bsa,
    test_ph_preflight_gates_the_grid,
    test_acpype_net_charge_comes_from_the_smiles,
    test_polca_si_lj_is_pinned_to_the_published_table,
    test_analysis_window_defaults_to_the_last_half,
    test_sampling_statistics,
    test_tie_rule_is_a_ci_not_float_equality,
    test_block_means_are_labelled_as_not_an_error_bar,
    test_loading_sweep_requires_a_pinned_or_qualified_composition,
    test_grid_extends_a_point_when_more_replicas_are_requested,
    test_metric_is_keyed_on_the_box_not_the_argument_list,
]

if __name__ == "__main__":
    fails = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'ALL PASS' if not fails else 'FAILURES'} "
          f"{len(_TESTS) - fails}/{len(_TESTS)}")
    sys.exit(1 if fails else 0)
