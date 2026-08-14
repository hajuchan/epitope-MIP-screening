#!/usr/bin/env python3
"""
Phase 4 explicit-composition / ratio-grid unit tests
====================================================
These cover the machinery that turns BSA_RATIO_GRID from a declaration into a
sweep.  NO MD is run: `_run_prepolymerization_md` and `_pool_replicas` are
stubbed, so the whole file executes in well under a second.

    python3 code/tests/test_ratio_grid.py        # standalone
    pytest code/tests/test_ratio_grid.py         # or under pytest

G1  EXPLICIT COPIES     — copies= is imposed verbatim; a species requested at 0
                          is dropped from the box AND from present_functional /
                          present_crosslinker, so the analysis is never told to
                          measure an absent monomer.
G2  COPY VALIDATION     — negative / non-int / unknown / all-zero are refused.
G3  DEFAULT PATH        — _monomer_copy_numbers still derives 36 TEOS / 24 APTES
                          at n=60 from SOLGEL_Q_MOLE_FRACTION=0.6, i.e. x=0.40,
                          which is NOT a BSA_RATIO_GRID member. This is the
                          regression anchor config_BSA.py §7 used to claim (30,30)
                          was.
G4  TOPOLOGY VERIFY     — the built [ molecules ] block is parsed and asserted
                          against the request; a mismatch RAISES.
G5  RATIO ROUNDING      — x=0.6 must give a 2-part crosslinker, not 1
                          (0.6/0.4 == 1.4999999999999998 in IEEE754).
G6  CONTROL BOXES       — no functional monomer / no crosslinker produce an
                          explicit "no recipe" instead of an invented 1:1.
G7  GRID NORMALISATION  — entry shapes, the shell cap, duplicate points.
G8  GRID RESUME         — a completed point is read back, not recomputed; a
                          point that RAISES does not cost the points before it;
                          the index and per-point JSON are written as it goes.
G9  H-BOND SELECTIONS   — donors/hydrogens/acceptors are element-restricted and
                          explicit; the min_charge=0.3 guess this replaces drops
                          a backbone amide H (+0.2719), and an un-restricted
                          acceptors_sel would admit carbons.
G10 LOADING SWEEP      — the winning composition is rescaled to each n_total with
                          the ratio held fixed, and a zero stays zero.
G11 LOADING ASSERTION  — BSA_RATIO_TOTAL_MONOMERS is asserted against every point
                          of the x-sweep instead of silently rescaling it.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent
sys.path.insert(0, str(_CODE_DIR))

import pipeline.phase4_md_validation as p4  # noqa: E402


# ── G1 ──────────────────────────────────────────────────────────
def test_explicit_copies_are_imposed():
    comp = p4._composition_from_copies(["APTES"], "TEOS",
                                       {"TEOS": 48, "APTES": 12})
    assert comp["copies"] == {"TEOS": 48, "APTES": 12}
    assert comp["total_monomers"] == 60
    assert comp["composition_source"] == "explicit_copies"
    # The box's OWN crosslinker mole fraction, not the library constant.
    assert comp["crosslinker_mole_fraction_actual"] == 0.8
    assert comp["crosslinker_mole_fraction_target"] == 0.8
    assert comp["present_functional"] == ["APTES"]
    assert comp["present_crosslinker"] == "TEOS"

    # (60, 0) — the TEOS-only control. Previously refused three ways.
    ctrl = p4._composition_from_copies(["APTES"], "TEOS",
                                       {"TEOS": 60, "APTES": 0})
    assert ctrl["copies"] == {"TEOS": 60}, ctrl["copies"]
    assert ctrl["present_functional"] == []
    assert ctrl["present_crosslinker"] == "TEOS"
    assert ctrl["dropped_monomers"] == ["APTES"]
    assert ctrl["requested_copies"] == {"TEOS": 60, "APTES": 0}

    # (0, 60) — the APTES-only control.
    ctrl2 = p4._composition_from_copies(["APTES"], "TEOS",
                                        {"TEOS": 0, "APTES": 60})
    assert ctrl2["copies"] == {"APTES": 60}
    assert ctrl2["present_functional"] == ["APTES"]
    assert ctrl2["present_crosslinker"] is None
    assert ctrl2["crosslinker_mole_fraction_actual"] == 0.0
    print("G1 PASS  explicit copies imposed; zero drops the species cleanly")


# ── G2 ──────────────────────────────────────────────────────────
def test_copy_validation():
    bad = [
        ({"TEOS": -1, "APTES": 10}, "negative"),
        ({"TEOS": 1.5, "APTES": 10}, "not an int"),
        ({"TEOS": 10, "APTES": 10, "MAA": 5}, "neither functional"),
        ({"TEOS": 0, "APTES": 0}, "EMPTY box"),
    ]
    for copies, needle in bad:
        try:
            p4._composition_from_copies(["APTES"], "TEOS", copies)
        except ValueError as e:
            assert needle in str(e), (copies, str(e))
        else:
            raise AssertionError(f"{copies} was accepted")
    # True is an int subclass in Python — it must not slip through as 1.
    try:
        p4._composition_from_copies(["APTES"], "TEOS", {"TEOS": True, "APTES": 5})
    except ValueError:
        pass
    else:
        raise AssertionError("bool copy count accepted")
    print("G2 PASS  negative / float / unknown / empty / bool copies refused")


# ── G3 ──────────────────────────────────────────────────────────
def test_default_composition_is_not_a_grid_point():
    comp = p4._monomer_copy_numbers(["APTES"], "TEOS", 60)
    assert comp["copies"] == {"APTES": 24, "TEOS": 36}, comp["copies"]
    assert comp["crosslinker_mole_fraction_actual"] == 0.6
    assert comp["composition_source"] == "SOLGEL_Q_MOLE_FRACTION"
    x_aptes = comp["copies"]["APTES"] / comp["total_monomers"]
    assert abs(x_aptes - 0.40) < 1e-9
    # …and 0.40 is not a member of the declared BSA grid (0, .05, .1, .2,
    # .333, .5, .667, 1.0). This is why the grid REPLACES the default leg.
    for x in (0.0, 0.05, 0.10, 0.20, 1 / 3, 0.50, 2 / 3, 1.0):
        assert abs(x_aptes - x) > 1e-6
    print("G3 PASS  default leg is x_APTES=0.40 (36 TEOS / 24 APTES), off-grid")


# ── G4 ──────────────────────────────────────────────────────────
_TOP = """; generated
[ system ]
Protein in water

[ molecules ]
; Compound        #mols
Protein_chain_A     1
APTES              12
TEOS               48
SOL             75461
NA                 16
"""


def test_topology_composition_verification():
    with tempfile.TemporaryDirectory() as td:
        md = Path(td)
        (md / "topol.top").write_text(_TOP)
        counts = p4._read_topology_molecule_counts(md / "topol.top")
        assert counts == {"APTES": 12, "TEOS": 48}, counts

        rec = p4._verify_built_composition(md, {"TEOS": 48, "APTES": 12})
        assert rec["verified"] is True
        assert rec["built_copies"] == {"APTES": 12, "TEOS": 48}

        # A zero-copy request must compare against the species that are there.
        (md / "topol.top").write_text(_TOP.replace("APTES              12\n", ""))
        rec2 = p4._verify_built_composition(md, {"TEOS": 48, "APTES": 0})
        assert rec2["built_copies"] == {"TEOS": 48}

        # …and a real mismatch must RAISE, not warn.
        for wrong in ({"TEOS": 48, "APTES": 12}, {"TEOS": 47}):
            try:
                p4._verify_built_composition(md, wrong)
            except RuntimeError as e:
                assert "DOES NOT MATCH THE REQUEST" in str(e)
            else:
                raise AssertionError(f"mismatch {wrong} accepted")
    print("G4 PASS  built [ molecules ] parsed and asserted; mismatch raises")


# ── G5 ──────────────────────────────────────────────────────────
def test_crosslinker_parts_rounding():
    # THE BUG: 0.6 / (1 - 0.6) == 1.4999999999999998, and round() gave 1 —
    # publishing a 1:1 recipe (50 mol%) from a 60 mol% target.
    assert 0.6 / (1 - 0.6) != 1.5          # the float fact this guards
    pack = p4._derive_optimal_ratio({"APTES": 7}, ["APTES"], "TEOS", "sol-gel",
                                    _crosslinker_mole_fraction=0.6)
    assert pack["optimal_ratio"] == {"APTES": 1, "TEOS": 2}, pack["optimal_ratio"]
    basis = pack["optimal_ratio_basis"]
    assert basis["crosslinker_mole_fraction_target"] == 0.6
    assert basis["crosslinker_mole_fraction_achieved"] == round(2 / 3, 4)
    # An as-built box publishes ITS OWN fraction, not the library constant.
    pack80 = p4._derive_optimal_ratio({"APTES": 7}, ["APTES"], "TEOS", "sol-gel",
                                      _crosslinker_mole_fraction=0.8)
    assert pack80["optimal_ratio"] == {"APTES": 1, "TEOS": 4}
    assert pack80["optimal_ratio_basis"]["crosslinker_parts_from"] == \
        "as-built box composition"
    print("G5 PASS  crosslinker parts round half-up (0.6 -> 2 parts, 0.8 -> 4)")


# ── G6 ──────────────────────────────────────────────────────────
def test_control_boxes_get_no_invented_recipe():
    # Crosslinker-only: nothing to derive parts FROM.
    try:
        p4._derive_optimal_ratio({}, [], "TEOS", "sol-gel")
    except ValueError as e:
        assert "no functional monomer" in str(e)
    else:
        raise AssertionError("crosslinker-only box produced a recipe")

    # Functional-only: no crosslinker was simulated, so none is reported.
    pack = p4._derive_optimal_ratio({"APTES": 5}, ["APTES"], None, "sol-gel")
    assert pack["optimal_ratio"] == {"APTES": 1}
    assert "TEOS" not in pack["optimal_ratio"]
    b = pack["optimal_ratio_basis"]
    assert b["is_control_box"] is True
    assert b["crosslinker_mole_fraction_achieved"] == 0.0
    print("G6 PASS  control boxes report no recipe instead of an invented 1:1")


# ── G7 ──────────────────────────────────────────────────────────
def test_grid_point_normalisation():
    grid = [(60, 0), (48, 12), {"TEOS": 30, "APTES": 30}]
    pts = p4._grid_points_to_copies(grid, "TEOS", ["APTES"])
    assert [p["copies"] for p in pts] == [
        {"TEOS": 60, "APTES": 0},
        {"TEOS": 48, "APTES": 12},
        {"TEOS": 30, "APTES": 30}]
    assert [p["label"] for p in pts] == [
        "TEOS60_APTES0", "TEOS48_APTES12", "TEOS30_APTES30"]
    assert [p["x_functional"] for p in pts] == [0.0, 0.2, 0.5]

    # The shell cap is enforced BEFORE anything is built.
    try:
        p4._grid_points_to_copies([(300, 300)], "TEOS", ["APTES"], 100)
    except ValueError as e:
        assert "shell placement cap" in str(e)
    else:
        raise AssertionError("over-cap point accepted")

    # Two points that are the same composition are a silent halving of the grid.
    try:
        p4._grid_points_to_copies([(48, 12), (48, 12)], "TEOS", ["APTES"])
    except ValueError as e:
        assert "duplicate compositions" in str(e)
    else:
        raise AssertionError("duplicate grid point accepted")

    # A wrong-length tuple must not be silently zero-padded.
    try:
        p4._grid_points_to_copies([(1, 2, 3)], "TEOS", ["APTES"])
    except ValueError as e:
        assert "entries but" in str(e)
    else:
        raise AssertionError("wrong-length grid point accepted")
    print("G7 PASS  grid entry shapes, shell cap and duplicate points handled")


# ── G8 ──────────────────────────────────────────────────────────
def _stub_grid_run(tmp: Path, fail_labels=(), calls=None, grid=None):
    """Run the grid driver with the MD replaced by a deterministic stub."""
    grid = grid or [(60, 0), (48, 12), (30, 30)]
    real_md, real_pool = p4._run_prepolymerization_md, p4._pool_replicas

    def fake_md(**kw):
        label = kw["seed_label"].split("@", 1)[1]
        if calls is not None:
            calls.append(label)
        if label in fail_labels:
            raise RuntimeError(f"boom at {label}")
        comp = p4._composition_from_copies(kw["functional_monomers"],
                                           kw["crosslinker"], kw["copies"])
        return {"replica_index": kw["replica_index"], "md_completed": True,
                "accepted": True, "success": True,
                "built_copies": comp["copies"],
                "initial_copies": comp["copies"],
                "present_functional": comp["present_functional"],
                "present_crosslinker": comp["present_crosslinker"],
                "box_composition": {k: v for k, v in comp.items()
                                    if k != "copies"},
                "work_dir": str(kw["work_dir"]),
                # Contacts scale with APTES count, so a RAW ranking would pick
                # the all-APTES endpoint every time; the per-copy metric must not.
                "occupancy_analysis": {
                    "contact_freq_6A": {m: 0.5 * n
                                        for m, n in comp["copies"].items()},
                    "EBN": {m: 3 for m in comp["copies"]},
                    "convergence": {"converged": True}},
                }

    def fake_pool(target, pc_id, functional, crosslinker, reps, time_ns,
                  pc_dir, output_dir):
        rep = reps[0]
        return {"target": target, "pc_id": pc_id, "accepted": True,
                "success": True, "n_replicas_accepted": len(reps),
                "pooled_composition": rep["built_copies"],
                "occupancy_analysis": rep["occupancy_analysis"],
                "optimal_ratio": None}

    p4._run_prepolymerization_md, p4._pool_replicas = fake_md, fake_pool
    try:
        return p4.run_phase4_ratio_grid(
            target="BSA", pc_id="PC", functional_monomers=["APTES"],
            crosslinker="TEOS", epitope_pdb=tmp / "x.pdb",
            work_dir=tmp / "ratio_grid", grid=grid,
            time_ns=1.0, n_replicas=1)
    finally:
        p4._run_prepolymerization_md, p4._pool_replicas = real_md, real_pool


def test_grid_is_resumable_and_per_point_readable():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ── first run: point 2 of 3 explodes ──
        calls1 = []
        s1 = _stub_grid_run(tmp, fail_labels={"TEOS48_APTES12"}, calls=calls1)
        assert s1["n_points"] == 3
        assert s1["n_points_failed"] == 1
        assert s1["success"] is False, "a partial grid must not report success"

        gdir = tmp / "ratio_grid"
        # Every point is INDEPENDENTLY readable from its own file.
        for label in ("TEOS60_APTES0", "TEOS48_APTES12", "TEOS30_APTES30"):
            pj = gdir / label / "point_result.json"
            assert pj.exists(), pj
            e = json.loads(pj.read_text())
            assert e["label"] == label
            assert e["copies"] == s1["per_point"][label]["copies"]
        failed = json.loads(
            (gdir / "TEOS48_APTES12" / "point_result.json").read_text())
        assert failed["status"] == "failed"
        assert "boom" in failed["error"] and failed["traceback"]

        idx = json.loads((gdir / "ratio_grid_index.json").read_text())
        assert {p["label"]: p["status"] for p in idx["points"]} == {
            "TEOS60_APTES0": "done", "TEOS48_APTES12": "failed",
            "TEOS30_APTES30": "done"}

        # ── second run: completed points are NOT recomputed ──
        calls2 = []
        _stub_grid_run(tmp, fail_labels=set(), calls=calls2)
        assert "TEOS60_APTES0" not in calls2, "a completed point was re-run"
        assert "TEOS30_APTES30" not in calls2, "a completed point was re-run"
        assert calls2 == ["TEOS48_APTES12"], calls2
        s2 = json.loads((gdir / "ratio_grid_summary.json").read_text())
        assert s2["n_points_complete"] == 3 and s2["n_points_failed"] == 0
        assert s2["success"] is True

        # ── a point directory whose recorded composition disagrees with the
        # grid must RAISE, not be reused. (Labels are composition-derived, so
        # this needs a tampered/renamed file to reproduce — which is exactly the
        # case a silent reuse would be most damaging in.) ──
        pj = gdir / "TEOS30_APTES30" / "point_result.json"
        tampered = json.loads(pj.read_text())
        tampered["copies"] = {"TEOS": 31, "APTES": 29}
        pj.write_text(json.dumps(tampered))
        try:
            _stub_grid_run(tmp)
        except RuntimeError as e:
            assert "records copies" in str(e), str(e)
        else:
            raise AssertionError("a mismatched point result was silently reused")
    print("G8 PASS  per-point JSON + index; resume skips done points; "
          "a failed point costs only itself")


def test_loading_sweep_rescales_and_preserves_controls():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        real_md, real_pool = p4._run_prepolymerization_md, p4._pool_replicas
        seen = {}

        def fake_md(**kw):
            seen[kw["pc_id"]] = dict(kw["copies"])
            comp = p4._composition_from_copies(kw["functional_monomers"],
                                               kw["crosslinker"], kw["copies"])
            return {"replica_index": 0, "md_completed": True, "accepted": True,
                    "success": True, "built_copies": comp["copies"],
                    "initial_copies": comp["copies"],
                    "present_functional": comp["present_functional"],
                    "present_crosslinker": comp["present_crosslinker"],
                    "box_composition": {k: v for k, v in comp.items()
                                        if k != "copies"},
                    "work_dir": str(kw["work_dir"]),
                    "occupancy_analysis": {
                        "contact_freq_6A": {m: 1.0 for m in comp["copies"]},
                        "EBN": {m: 2 for m in comp["copies"]},
                        "convergence": {"converged": True}}}

        def fake_pool(target, pc_id, functional, crosslinker, reps, time_ns,
                      pc_dir, output_dir):
            return {"target": target, "pc_id": pc_id, "accepted": True,
                    "success": True, "n_replicas_accepted": 1,
                    "pooled_composition": reps[0]["built_copies"],
                    "occupancy_analysis": reps[0]["occupancy_analysis"],
                    "optimal_ratio": None}

        p4._run_prepolymerization_md, p4._pool_replicas = fake_md, fake_pool
        try:
            s = p4.run_phase4_loading_sweep(
                target="BSA", pc_id="PC", functional_monomers=["APTES"],
                crosslinker="TEOS", epitope_pdb=tmp / "x.pdb",
                work_dir=tmp / "loading", winning_copies={"TEOS": 48, "APTES": 12},
                loadings=[30, 60, 100], time_ns=1.0, n_replicas=1)
        finally:
            p4._run_prepolymerization_md, p4._pool_replicas = real_md, real_pool

        built = {sum(v.values()): v for v in seen.values()}
        assert set(built) == {30, 60, 100}, sorted(built)
        # The RATIO is held fixed while n_total moves — that is the whole point.
        for n, cp in built.items():
            assert abs(cp["APTES"] / n - 0.2) < 0.02, (n, cp)
        assert s["success"] is True
        assert json.loads(
            (tmp / "loading" / "loading_sweep_summary.json").read_text()
        )["loadings"] == [30, 60, 100]

        # A control point must stay a control point through the rescale.
        pts = p4._grid_points_to_copies([(60, 0)], "TEOS", ["APTES"])
        assert pts[0]["copies"]["APTES"] == 0
    print("G10 PASS  loading sweep rescales n_total at fixed composition")


def test_total_monomers_is_asserted_not_silently_rescaled():
    # Simulates a grid edited so one point drifts off the declared loading:
    # the difference between that point and the rest would then be part ratio
    # and part concentration, which is exactly what must not be attributable.
    real = p4._ratio_grid_from_config
    p4._ratio_grid_from_config = lambda: {
        "grid": None, "total_monomers": 60, "time_ns": 1.0,
        "max_monomers_in_shell": 100, "provisional": False, "n_replicas": 1,
        "loading_sweep": [], "source_keys": ["TEST"]}
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                p4.run_phase4_ratio_grid(
                    target="BSA", pc_id="PC", functional_monomers=["APTES"],
                    crosslinker="TEOS", epitope_pdb=Path(td) / "x.pdb",
                    work_dir=Path(td) / "g", grid=[(48, 12), (40, 30)],
                    time_ns=1.0, n_replicas=1)
            except ValueError as e:
                assert "BSA_RATIO_TOTAL_MONOMERS" in str(e), str(e)
                assert "TEOS40_APTES30" in str(e), str(e)
            else:
                raise AssertionError("off-loading grid point accepted")
    finally:
        p4._ratio_grid_from_config = real
    print("G11 PASS  BSA_RATIO_TOTAL_MONOMERS asserted against every grid point")


def test_grid_ranking_is_normalised():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        s = _stub_grid_run(tmp, grid=[(54, 6), (30, 30), (0, 60)])
        # The stub makes raw contacts strictly proportional to copy number, so a
        # raw-count ranking would ALWAYS name the all-APTES endpoint. Per-copy,
        # every point ties — which is the honest answer for that stub, and the
        # point of the normalisation.
        m = {l: e["metrics"]["functional_contact_per_copy"]
             for l, e in s["per_point"].items()}
        assert set(m.values()) == {0.5}, m
        raw = {l: e["metrics"]["contact_freq_6A"]["APTES"]
               for l, e in s["per_point"].items()}
        assert raw["TEOS0_APTES60"] > raw["TEOS54_APTES6"]
        assert s["best_point"] is not None
        assert s["rank_metric"] == "functional_contact_per_copy"
        # An all-ties grid must SAY it did not discriminate rather than letting
        # sort order pass for an optimum.
        assert s["best_point_is_tied"] is True
        assert len(s["tied_points"]) == 3, s["tied_points"]
    print("G8b PASS  ranking metric is per-functional-copy; ties are declared")


def test_legacy_preset_sweep_imposes_distinct_boxes():
    """The old sweep collapsed every preset onto one box; it must not any more."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        real_md, real_pool, real_cfg = (p4._run_prepolymerization_md,
                                        p4._pool_replicas,
                                        p4._ratio_grid_from_config)
        built = []

        def fake_md(**kw):
            built.append(dict(kw["copies"]))
            comp = p4._composition_from_copies(kw["functional_monomers"],
                                               kw["crosslinker"], kw["copies"])
            return {"replica_index": 0, "md_completed": True, "accepted": True,
                    "success": True, "built_copies": comp["copies"],
                    "initial_copies": comp["copies"],
                    "present_functional": comp["present_functional"],
                    "present_crosslinker": comp["present_crosslinker"],
                    "box_composition": {k: v for k, v in comp.items()
                                        if k != "copies"},
                    "work_dir": str(kw["work_dir"]),
                    "occupancy_analysis": {
                        "contact_freq_6A": {m: 1.0 for m in comp["copies"]},
                        "EBN": {m: 2 for m in comp["copies"]},
                        "convergence": {"converged": True}}}

        p4._run_prepolymerization_md = fake_md
        p4._pool_replicas = lambda t, p, f, c, r, ns, pd, od: {
            "target": t, "pc_id": p, "accepted": True, "success": True,
            "pooled_composition": r[0]["built_copies"],
            "occupancy_analysis": r[0]["occupancy_analysis"],
            "optimal_ratio": None}
        p4._ratio_grid_from_config = lambda: None      # no declared grid
        try:
            p4.run_phase4_ratio_sweep(
                target="CD63", pc_id="PC",
                functional_monomers=["MAA", "HEMA"], crosslinker="EGDMA",
                epitope_pdb=tmp / "x.pdb", work_dir=tmp / "sweep",
                ratio_presets=[(1, 1), (2, 1), (3, 1), (1, 2)], time_ns=1.0)
        finally:
            (p4._run_prepolymerization_md, p4._pool_replicas,
             p4._ratio_grid_from_config) = real_md, real_pool, real_cfg

        uniq = {tuple(sorted(b.items())) for b in built}
        assert len(uniq) == 4, f"presets collapsed onto {len(uniq)} box(es): {uniq}"
        # Every preset honours the crosslinker mole fraction and the real total,
        # not the old hardcoded total=20.
        for b in built:
            assert sum(b.values()) == 20 or sum(b.values()) == \
                p4._cfg_get("EPITOPE_MONOMER_MOLAR_RATIO", 20), b
    print("G12 PASS  legacy preset sweep now builds one distinct box per preset")


# ── G9 ──────────────────────────────────────────────────────────
def _toy_universe():
    import numpy as np
    import MDAnalysis as mda
    names = ["N", "H", "CA", "C", "O", "O1", "H1", "SI1", "O2"]
    masses = [14.007, 1.008, 12.011, 12.011, 15.999,
              15.999, 1.008, 28.085, 15.999]
    # amber99sb-ildn backbone amide H is +0.2719 — BELOW MDAnalysis' 0.3 guess.
    charges = [-0.4157, 0.2719, 0.0337, 0.5973, -0.5679,
               -0.6, 0.45, 1.0, -0.6]
    u = mda.Universe.empty(len(names), n_residues=2,
                           atom_resindex=[0, 0, 0, 0, 0, 1, 1, 1, 1],
                           trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("type", ["N", "H", "CT", "C", "O", "oh", "ho", "Si", "oh"])
    u.add_TopologyAttr("mass", masses)
    u.add_TopologyAttr("charge", charges)
    u.add_TopologyAttr("resname", ["ALA", "UNL"])
    u.add_TopologyAttr("resid", [1, 2])
    u.atoms.positions = np.array([
        [0, 0, 0], [0.0, 1.0, 0.0], [1.5, 0, 0], [2.0, 1.4, 0], [1.6, 2.5, 0],
        [0.0, 3.0, 0.0], [0.6, 3.7, 0.3], [0.0, 4.6, 0.0], [1.2, 5.2, 0.0],
    ], dtype="float32")
    u.dimensions = [50, 50, 50, 90, 90, 90]
    return u


def test_hbond_selections_are_explicit_and_element_restricted():
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
        HydrogenBondAnalysis as HBA)
    u = _toy_universe()
    prot, mon = "protein", "(not protein and resid 2)"

    # (1) The guess this replaces DROPS the protein backbone amide H.
    probe = HBA(u, donors_sel="protein", hydrogens_sel="name H",
                acceptors_sel="resid 2")
    guessed = probe.guess_hydrogens("all")                 # min_charge=0.3
    assert "resname ALA and name H" not in guessed, guessed
    assert "resname ALA and name H" in probe.guess_hydrogens("all", min_charge=0.2)

    # (2) The explicit selections recover it, and admit no carbon.
    h_prot = u.select_atoms(f"({prot}) and {p4._HB_HYDROGEN_SEL}")
    da_prot = u.select_atoms(f"({prot}) and {p4._HB_DONOR_ACCEPTOR_SEL}")
    da_mon = u.select_atoms(f"({mon}) and {p4._HB_DONOR_ACCEPTOR_SEL}")
    h_mon = u.select_atoms(f"({mon}) and {p4._HB_HYDROGEN_SEL}")
    assert list(h_prot.names) == ["H"]
    assert list(da_prot.names) == ["N", "O"], list(da_prot.names)
    assert list(da_mon.names) == ["O1", "O2"], list(da_mon.names)
    assert list(h_mon.names) == ["H1"]
    # The un-restricted acceptors_sel the old code passed would have made every
    # carbon and hydrogen an acceptor.
    assert set(u.select_atoms(prot).names) - set(da_prot.names) >= {"CA", "C", "H"}

    # (3) End to end: a real H-bond is found through the production selections.
    hb = HBA(u,
             donors_sel=f"({prot}) and {p4._HB_DONOR_ACCEPTOR_SEL}",
             hydrogens_sel=f"({prot}) and {p4._HB_HYDROGEN_SEL}",
             acceptors_sel=f"({mon}) and {p4._HB_DONOR_ACCEPTOR_SEL}",
             d_a_cutoff=3.5, d_h_a_angle_cutoff=150, update_selections=False)
    hb.run(verbose=False)
    assert len(hb.results.hbonds) == 1, hb.results.hbonds
    assert list(hb._donors.names) == ["N"] and list(hb._hydrogens.names) == ["H"]
    print("G9 PASS  H-bond donors/hydrogens/acceptors explicit + N/O/S-restricted")


def _stub_grid_run_no_md(tmp: Path, md_ok: bool, calls=None, grid=None):
    """Grid driver where every replica's mdrun ABORTS but pooling still returns.

    This is the real Stage-0 failure shape: build/EM/NVT/NPT succeed, production
    mdrun dies, `_pool_replicas` returns a result object with
    md_completed=False for every replica, and nothing RAISES — so the point is
    recorded status='done'.
    """
    grid = grid or [(60, 0), (48, 12)]
    real_md, real_pool = p4._run_prepolymerization_md, p4._pool_replicas

    def fake_md(**kw):
        label = kw["seed_label"].split("@", 1)[1]
        if calls is not None:
            calls.append(label)
        comp = p4._composition_from_copies(kw["functional_monomers"],
                                           kw["crosslinker"], kw["copies"])
        return {"replica_index": kw["replica_index"], "md_completed": md_ok,
                "accepted": md_ok, "success": md_ok,
                "built_copies": comp["copies"],
                "initial_copies": comp["copies"],
                "present_functional": comp["present_functional"],
                "present_crosslinker": comp["present_crosslinker"],
                "box_composition": {k: v for k, v in comp.items()
                                    if k != "copies"},
                "work_dir": str(kw["work_dir"]),
                "occupancy_analysis": {
                    "contact_freq_6A": {m: 0.5 * n
                                        for m, n in comp["copies"].items()},
                    "EBN": {m: 3 for m in comp["copies"]},
                    "convergence": {"converged": True}},
                }

    def fake_pool(target, pc_id, functional, crosslinker, reps, time_ns,
                  pc_dir, output_dir):
        rep = reps[0]
        return {"target": target, "pc_id": pc_id,
                "accepted": md_ok, "success": md_ok,
                "md_completed": md_ok,
                "n_replicas_accepted": len(reps) if md_ok else 0,
                "accepted_replica_indices": [0] if md_ok else [],
                "replicas": reps,
                "pooled_composition": rep["built_copies"],
                "occupancy_analysis": rep["occupancy_analysis"],
                "optimal_ratio": None}

    p4._run_prepolymerization_md, p4._pool_replicas = fake_md, fake_pool
    try:
        return p4.run_phase4_ratio_grid(
            target="BSA", pc_id="PC", functional_monomers=["APTES"],
            crosslinker="TEOS", epitope_pdb=tmp / "x.pdb",
            work_dir=tmp / "ratio_grid", grid=grid,
            time_ns=1.0, n_replicas=1)
    finally:
        p4._run_prepolymerization_md, p4._pool_replicas = real_md, real_pool


def test_point_with_no_trajectory_is_retried_not_resumed():
    """G12 — status='done' is NOT proof the point holds data.

    Stage 0 hit this for real: all 6 points aborted in production mdrun, every
    one was written status='done' accepted=False with zero accepted replicas,
    and the resume gate keyed on status alone.  Re-running would have read the
    six EMPTY points straight back and reported a complete sweep, so the whole
    ratio answer would have come from trajectories that never existed.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        gdir = tmp / "ratio_grid"

        # ── run 1: mdrun aborts everywhere, but pooling returns normally ──
        s1 = _stub_grid_run_no_md(tmp, md_ok=False)
        for label in ("TEOS60_APTES0", "TEOS48_APTES12"):
            e = json.loads((gdir / label / "point_result.json").read_text())
            # This is the trap: 'done' yet empty.
            assert e["status"] == "done", e["status"]
            assert e["accepted"] is False
            assert e["n_replicas_accepted"] == 0
            assert all(r["md_completed"] is False for r in e["pooled"]["replicas"])
        assert s1["n_points_accepted"] == 0
        assert s1["success"] is False, "a grid with no trajectories is not a success"

        # ── run 2: both points must be RE-RUN, not read back ──
        calls2 = []
        s2 = _stub_grid_run_no_md(tmp, md_ok=True, calls=calls2)
        assert sorted(calls2) == ["TEOS48_APTES12", "TEOS60_APTES0"], (
            f"an empty point was resumed instead of retried: {calls2}")
        assert s2["n_points_accepted"] == 2
        assert s2["success"] is True

        # ── run 3: now that they hold trajectories, resume DOES skip them ──
        calls3 = []
        _stub_grid_run_no_md(tmp, md_ok=True, calls=calls3)
        assert calls3 == [], f"a point with real data was recomputed: {calls3}"
    print("G12 PASS  status='done' with no finished MD is retried, not resumed; "
          "a point that really ran is still skipped")


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
    print(f"\nALL PASS  {len(_TESTS)}/{len(_TESTS)}")
