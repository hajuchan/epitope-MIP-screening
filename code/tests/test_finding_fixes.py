"""
Regression tests for the Phase-1/Phase-2 comparability defects fixed on
2026-08-12.

Each test names the defect it pins. Run with:
    MIP_CONFIG_QUIET=1 PYTHONPATH=code python3 code/tests/test_finding_fixes.py

Tests marked [DATA] read results/ read-only and skip if it is absent. Nothing
here writes to results/.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MIP_CONFIG_QUIET", "1")

ROOT = Path(__file__).resolve().parents[2]
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"\n          {detail}" if detail else ""))


# ── F3: prepared templates are actually read ───────────────────────────────
def test_prepared_template_wiring():
    print("\nF3  prepared templates are wired into download_structure")
    from pipeline.utils_structure import download_structure

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "CD63_open_v1.pdb"
        src.write_text("ATOM      1  CA  ALA A 103      "
                       "0.000   0.000   0.000  1.00 50.00           C\nEND\n")
        out = Path(td) / "run"
        got = download_structure({"source": "alphafold", "uniprot_id": "P08962",
                                  "prepared_path": src}, out)
        check("prepared_path wins over source=alphafold",
              got.exists() and got.name == "CD63_open_v1.pdb",
              f"returned {got}")
        check("chimera is copied, not re-downloaded",
              got.read_text() == src.read_text())

        # A declared-but-missing template must NOT fall through to the raw entry
        try:
            download_structure({"source": "pdb", "pdb_id": "5TCX",
                                "prepared_path": Path(td) / "gone.pdb"}, out)
            check("missing prepared_path raises instead of falling back", False)
        except FileNotFoundError:
            check("missing prepared_path raises instead of falling back", True)

        # targets with no prepared_path keep the legacy behaviour
        cfg = {"source": "nonsense"}
        try:
            download_structure(cfg, out)
            check("no prepared_path -> legacy branch still reached", False)
        except ValueError:
            check("no prepared_path -> legacy branch still reached", True)

    from pipeline import config
    if config.EXPERIMENT == "CD":
        check("CD_TEMPLATES_WIRED is True", config.CD_TEMPLATES_WIRED is True)
        for t in ("CD63", "CD81", "CD9"):
            p = Path(config.TARGETS[t]["prepared_path"])
            check(f"{t} prepared template exists on disk", p.is_file(), str(p))


# ── F3b: the pLDDT gate must not average pLDDT with crystallographic B ─────
def test_plddt_scoping():
    print("\nF3b pLDDT gate is scoped on a mixed-provenance template")
    from pipeline import config
    from pipeline.utils_structure import check_plddt
    if config.EXPERIMENT != "CD":
        print("      (skipped: CD only)")
        return
    cfg = config.TARGETS["CD63"]
    p = Path(cfg["prepared_path"])
    if not p.is_file():
        print("      (skipped: prepared template absent)")
        return

    naive = check_plddt(p, cfg["ecl2_range"], threshold=70)
    scoped = check_plddt(p, cfg["ecl2_range"], threshold=70, only_predicted=True)
    check("naive and scoped disagree on a chimera",
          naive.get("mean_plddt") != scoped.get("mean_plddt"),
          f"all-residues mean={naive.get('mean_plddt')} "
          f"vs predicted-only mean={scoped.get('mean_plddt')} "
          f"over {scoped.get('n_predicted_residues')} predicted / "
          f"{scoped.get('n_experimental_residues')} experimental residues")
    check("scoped result labels its scope",
          scoped.get("scope") == "predicted-only")
    check("a fully experimental EC2 does not fail the gate",
          scoped.get("pass") is True or scoped.get("n_predicted_residues", 0) > 0,
          scoped.get("message", ""))


# ── F2: the merge estimator must not reward having more conformers ────────
def test_merge_estimator_unbiased():
    print("\nF2  ensemble merge is not a best-of-N statistic")
    from pipeline.phase2_smd import _merge_across_conformers
    import pipeline.config as config

    rng = np.random.default_rng(0)
    trials = 4000
    got = {}
    for how in ("min", "boltzmann", "mean"):
        config.ENSEMBLE_MERGE = how
        means = {}
        for n in (1, 6):
            vals = []
            for _ in range(trials):
                es = rng.normal(-5.0, 0.5, size=n)
                pc = {"M": [(f"c{i}", {"mean_cluster_energy": float(e)})
                            for i, e in enumerate(es)]}
                vals.append(_merge_across_conformers("T", pc)["M"]
                            ["mean_cluster_energy"])
            means[n] = float(np.mean(vals))
        got[how] = means[6] - means[1]
        print(f"       {how:>10}: E[N=1]={means[1]:+.3f}  E[N=6]={means[6]:+.3f}"
              f"  shift={got[how]:+.3f} kcal/mol")

    config.ENSEMBLE_MERGE = "boltzmann"
    check("min() is strongly N-biased (the defect)", got["min"] < -0.5,
          f"best-of-6 is {abs(got['min']):.2f} kcal/mol stronger than best-of-1 "
          f"for identical binding")
    check("boltzmann is far less N-biased than min",
          abs(got["boltzmann"]) < 0.5 * abs(got["min"]),
          f"|{got['boltzmann']:.3f}| vs |{got['min']:.3f}|")
    check("mean is N-unbiased", abs(got["mean"]) < 0.05)


def test_merge_records_provenance():
    print("\nF2b merged results record how many conformers produced them")
    from pipeline.phase2_smd import _merge_across_conformers
    pc = {"AA": [("crystal", {"mean_cluster_energy": -4.0}),
                 ("md_conf1", {"mean_cluster_energy": -3.0})]}
    r = _merge_across_conformers("CD63", pc)["AA"]
    check("ensemble_n_conformers recorded", r["ensemble_n_conformers"] == 2)
    check("per-conformer energies retained",
          set(r["ensemble_conformer_energies"]) == {"crystal", "md_conf1"})
    check("merged score lies within the conformer range",
          -4.0 <= r["mean_cluster_energy"] <= -3.0,
          f"score={r['mean_cluster_energy']}")


# ── F1: selectivity must not subtract raw scores across receptors ─────────
def test_selectivity_is_normalised():
    print("\nF1  selectivity is rank-based, not a raw cross-receptor difference")
    from pipeline.phase2_smd import _compute_selectivity

    # Two targets, identical RANK ORDER, but target B's whole energy scale is
    # shifted by -2 kcal/mol — exactly what a larger or better-resolved
    # receptor does. A sound metric must report no selectivity here.
    mons = [f"M{i}" for i in range(10)]
    be = {"A": {m: -3.0 - 0.1 * i for i, m in enumerate(mons)},
          "B": {m: -5.0 - 0.1 * i for i, m in enumerate(mons)}}
    sel = _compute_selectivity(be, ["A", "B"])
    d = sel["_detail"]

    ddg = [d["A"][m]["ddg_raw"] for m in mons]
    rd = [d["A"][m]["rank_delta"] for m in mons]
    zd = [d["A"][m]["z_ddg"] for m in mons]
    check("raw ddG is fooled by a pure scale shift (the defect)",
          all(abs(v - 2.0) < 1e-6 for v in ddg),
          f"every monomer reports ddg_raw=+2.0 though rank order is identical")
    check("rank_delta correctly reports no selectivity",
          all(abs(v) < 1e-9 for v in rd))
    check("z_ddg correctly reports no selectivity",
          all(abs(v) < 1e-9 for v in zd))

    # And it must still SEE a real, rank-order selectivity difference.
    be2 = {"A": {m: -3.0 - 0.1 * i for i, m in enumerate(mons)},
           "B": {m: -3.0 - 0.1 * (9 - i) for i, m in enumerate(mons)}}
    d2 = _compute_selectivity(be2, ["A", "B"])["_detail"]
    check("rank_delta detects a genuine rank inversion",
          d2["A"]["M9"]["rank_delta"] > 5 and d2["A"]["M0"]["rank_delta"] < -5,
          f"M9 rank_delta={d2['A']['M9']['rank_delta']}, "
          f"M0 rank_delta={d2['A']['M0']['rank_delta']}")

    check("primary metric is not ddg_raw by default",
          _compute_selectivity(be, ["A", "B"])["_metric"] != "ddg_raw")


def test_exchangeability_gate():
    print("\nF1b unequal ensembles / unequal surface are flagged, not hidden")
    from pipeline.phase2_smd import _check_receptor_exchangeability
    p1 = {t: {"receptor_descriptor": {"sasa_scored_surface_A2": a}}
          for t, a in (("CD63", 5600.0), ("CD81", 5600.0), ("CD9", 5600.0))}
    ok = _check_receptor_exchangeability(["CD63", "CD81", "CD9"],
                                         {"CD63": 6, "CD81": 6, "CD9": 6}, p1)
    check("equal N + equal area -> exchangeable", ok["exchangeable"] is True)

    bad = _check_receptor_exchangeability(["CD63", "CD81", "CD9"],
                                          {"CD63": 6, "CD81": 1, "CD9": 6}, p1)
    check("the committed run's 6/1/6 is flagged NOT exchangeable",
          bad["exchangeable"] is False and bad["equal_sampling"] is False)

    p2 = json.loads(json.dumps(p1))
    p2["CD63"]["receptor_descriptor"]["sasa_scored_surface_A2"] = 6968.3
    p2["CD9"]["receptor_descriptor"]["sasa_scored_surface_A2"] = 5528.3
    area = _check_receptor_exchangeability(["CD63", "CD81", "CD9"],
                                           {"CD63": 6, "CD81": 6, "CD9": 6}, p2)
    check("the measured 26% CD63/CD9 surface gap is flagged",
          area["comparable_area"] is False,
          f"area ratio {area['area_max_min_ratio']}")


# ── F4 [DATA]: MD conformers must live in the docking box ─────────────────
def test_conformers_are_superposed():
    print("\nF4  MD conformers are superposed into the docking-box frame")
    from pipeline.utils_structure import (superpose_onto_reference,
                                          fraction_atoms_in_box)
    from pipeline import config
    p1 = ROOT / "results" / "phase1" / "phase1_results.json"
    if not p1.is_file() or config.EXPERIMENT != "CD":
        print("      (skipped: results/phase1 absent)")
        return
    r = json.loads(p1.read_text())

    worst_before, best_after = 1.0, 1.0
    with tempfile.TemporaryDirectory() as td:
        for t in ("CD63", "CD9"):
            gc, npts = tuple(r[t]["grid_center"]), tuple(r[t]["grid_npts"])
            ref = ROOT / "results" / "phase1" / t / f"{t}_ecl2.pdb"
            for i in range(5):
                src = ROOT / "results" / "phase1" / t / "conformers" / f"conf_{i}.pdb"
                if not src.is_file():
                    continue
                dst = Path(td) / f"{t}_{i}.pdb"
                shutil.copy2(src, dst)          # results/ is never modified
                worst_before = min(worst_before,
                                   fraction_atoms_in_box(dst, gc, npts))
                superpose_onto_reference(dst, ref, dst)
                best_after = min(best_after, fraction_atoms_in_box(dst, gc, npts))

    check("committed conformers really were outside their box (the defect)",
          worst_before < 0.20,
          f"worst coverage before superposition: {worst_before:.1%}")
    check("superposition puts every conformer back in the box",
          best_after >= config.ENSEMBLE_MIN_BOX_COVERAGE,
          f"worst coverage after superposition: {best_after:.1%} "
          f"(threshold {config.ENSEMBLE_MIN_BOX_COVERAGE:.0%})")


def test_box_coverage_guard_exists():
    print("\nF4b a misplaced conformer is dropped rather than docked")
    from pipeline import config
    check("ENSEMBLE_MIN_BOX_COVERAGE is configured",
          0.0 < config.ENSEMBLE_MIN_BOX_COVERAGE <= 1.0,
          str(config.ENSEMBLE_MIN_BOX_COVERAGE))
    src = (ROOT / "code" / "pipeline" / "phase1_epitope_prep.py").read_text()
    check("extraction superposes onto the reference",
          "superpose_onto_reference" in src)
    check("extraction validates box coverage",
          "fraction_atoms_in_box" in src and "ENSEMBLE_MIN_BOX_COVERAGE" in src)
    check("extraction is no longer gated on the stability verdict",
          "ENSEMBLE_EXTRACT_REQUIRES_STABLE" in src)


def test_config_knobs_resolve():
    print("\nCfg new knobs resolve under the active experiment")
    from pipeline import config
    for k in ("ENSEMBLE_MERGE", "ENSEMBLE_BOLTZMANN_T_K",
              "ENSEMBLE_REQUIRE_EQUAL_N", "ENSEMBLE_EXTRACT_REQUIRES_STABLE",
              "ENSEMBLE_MIN_BOX_COVERAGE", "SELECTIVITY_METRIC",
              "RECEPTOR_BOX_ON_SCORED_SURFACE"):
        check(f"{k} defined", hasattr(config, k), f"= {getattr(config, k, None)}")
    check("ENSEMBLE_MERGE is a known estimator",
          config.ENSEMBLE_MERGE in ("boltzmann", "mean", "min"))
    check("SELECTIVITY_METRIC is a known metric",
          config.SELECTIVITY_METRIC in ("rank_delta", "z_ddg",
                                        "own_cross_ratio", "ddg_raw"))


if __name__ == "__main__":
    from pipeline import config
    print(f"experiment = {config.EXPERIMENT}")
    for fn in (test_prepared_template_wiring, test_plddt_scoping,
               test_merge_estimator_unbiased, test_merge_records_provenance,
               test_selectivity_is_normalised, test_exchangeability_gate,
               test_conformers_are_superposed, test_box_coverage_guard_exists,
               test_config_knobs_resolve):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            FAIL.append(f"{fn.__name__} raised {e!r}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1 if FAIL else 0)
