#!/usr/bin/env python3
"""
Config dispatch regression tests
================================
The whole experiment-selection design rests on ONE promise: selecting CD (the
default) must be indistinguishable from the pre-dispatch config.py.  These five
tests are what enforce it.  Run them after ANY edit to config.py,
config_CD.py or config_BSA.py.

    python3 code/tests/test_config_regression.py        # standalone
    pytest code/tests/test_config_regression.py         # or under pytest

T1  CD UNCHANGED       — snapshot == committed config_baseline_CD.json
T2  _rederive NO-OP    — catches drift between _rederive() and config.py's five
                         derived expressions (L492/539/542/670/728).  This is
                         the design's one asymmetric hazard: the CD path never
                         CALLS _rederive(), so a divergence would break only the
                         less-exercised BSA path, silently.
T3  IMPORT CONTRACT    — every name any module imports from config must exist
                         under BOTH experiments.
T4  BSA ISOLATED       — output tree, pinned libraries, the TEOS-crosslinker-
                         record SMILES trap, DI water, pH recommendation.
T5  BAD SELECTOR       — a typo'd MIP_EXPERIMENT must raise, never fall back.
T6  BLOCKED SPECIES    — SILANE_SPECIES="hydrolyzed" must be refused at import
                         (no measurable Si-OH donor on this engine, §11(i)) and
                         the MIP_ALLOW_BLOCKED_SPECIES opt-in must still work.
T7  GUARD 1 WIDTH      — a CD-tree path injected into OUTPUT_DIRS *after*
                         _rederive() must be rejected; consumers read OUTPUT_DIRS
                         via get_output_path (which mkdirs), never OUTPUT_DIR.
T8  ENTRY-POINT PIN    — scripts that import config but write to hardcoded
                         results/ must refuse a non-CD experiment.
T9  PACKAGING          — CD must import with config_CD.py absent (it is a pure
                         documentation delta); every other experiment must not.
T10 BASELINE WRITE     — config_baseline.py --write must scrub MIP_EXPERIMENT and
                         refuse to fill the CD contract from another experiment.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent                      # …/code
_REPO_ROOT = _CODE_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR / "tools"))

from config_baseline import snapshot, BASELINE_JSON  # noqa: E402

# Names the dispatch layer legitimately ADDS to the CD namespace.  Nothing else
# may appear, disappear, or change value.
_NEW_NAMES = {
    "EXPERIMENT", "KNOWN_EXPERIMENTS", "EXPERIMENT_LABEL", "UNCONSUMED_KEYS",
}


def _fresh_env(**extra) -> dict:
    env = dict(os.environ)
    env.pop("MIP_EXPERIMENT", None)
    env["MIP_CONFIG_QUIET"] = "1"
    env["PYTHONPATH"] = str(_CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra)
    return env


def _run_probe(code: str, experiment: str | None = None):
    """Run a snippet in a FRESH interpreter with the given experiment."""
    env = _fresh_env()
    if experiment is not None:
        env["MIP_EXPERIMENT"] = experiment
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, cwd=str(_REPO_ROOT))


# ── T1 ──────────────────────────────────────────────────────────
def test_cd_unchanged():
    """CD (default) must equal the committed pre-dispatch baseline."""
    os.environ.pop("MIP_EXPERIMENT", None)
    os.environ["MIP_CONFIG_QUIET"] = "1"
    import pipeline.config as cfg
    got = snapshot(cfg)
    want = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))

    added = set(got) - set(want)
    removed = set(want) - set(got)
    changed = sorted(k for k in set(got) & set(want) if got[k] != want[k])

    assert not removed, f"symbols DISAPPEARED from CD: {sorted(removed)}"
    assert added <= _NEW_NAMES, f"unexpected new CD symbols: {sorted(added - _NEW_NAMES)}"
    assert not changed, f"CD values CHANGED: {changed}"
    print(f"T1 PASS  CD identical ({len(want)} keys, +{len(added)} dispatch names)")


# ── T2 ──────────────────────────────────────────────────────────
def test_rederive_is_noop_for_cd():
    """_rederive() must reproduce config.py L492/539/542/670/728 exactly."""
    os.environ.pop("MIP_EXPERIMENT", None)
    os.environ["MIP_CONFIG_QUIET"] = "1"
    import pipeline.config as cfg
    before = snapshot(cfg)
    cfg._rederive()
    after = snapshot(cfg)
    drift = sorted(k for k in before if before[k] != after[k])
    assert not drift, (
        f"_rederive() drifted from config.py's derived expressions: {drift}. "
        f"Edit _rederive() to match lines 492/539/542/670/728.")
    print("T2 PASS  _rederive() is a verified no-op on the CD path")


# ── T3 ──────────────────────────────────────────────────────────
def _imported_config_names() -> set[str]:
    """AST-walk every .py outside results/ for `from (.|pipeline.)config import X`."""
    names: set[str] = set()
    for py in _REPO_ROOT.rglob("*.py"):
        if "results" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                # relative `.config` shows up as module="config", level=1
                if mod in ("config", "pipeline.config"):
                    names.update(a.name for a in node.names if a.name != "*")
    return names


def test_import_contract():
    """Every imported config name must resolve under BOTH experiments."""
    wanted = _imported_config_names()
    assert len(wanted) > 100, f"AST walk found only {len(wanted)} names — suspicious"
    code = (
        "import json,sys;import pipeline.config as c;"
        "print(json.dumps([n for n in sys.argv[1:] if not hasattr(c,n)]))"
    )
    for exp in ("CD", "BSA"):
        env = _fresh_env(MIP_EXPERIMENT=exp)
        r = subprocess.run([sys.executable, "-c", code, *sorted(wanted)],
                           env=env, capture_output=True, text=True,
                           cwd=str(_REPO_ROOT))
        assert r.returncode == 0, f"{exp}: config import failed\n{r.stderr}"
        missing = json.loads(r.stdout.strip().splitlines()[-1])
        assert not missing, f"{exp}: config is MISSING imported names {missing}"
    print(f"T3 PASS  all {len(wanted)} imported names resolve under CD and BSA")


# ── T4 ──────────────────────────────────────────────────────────
def test_bsa_isolated():
    code = r"""
import json, pathlib
import pipeline.config as c
out = {
 "output_dir": c.OUTPUT_DIR,
 "cd_tree": str((c.PROJECT_ROOT / "results").resolve()),
 "all_monomers": sorted(c.ALL_MONOMERS),
 "functional": sorted(c.FUNCTIONAL_MONOMERS),
 "crosslinkers": sorted(c.CROSSLINKERS),
 "teos_smiles": c.ALL_MONOMERS["TEOS"]["smiles"],
 "aptes_smiles": c.ALL_MONOMERS["APTES"]["smiles"],
 "ionic": c.MD_IONIC_STRENGTH,
 "ph": c.synthesis_pH_window(["APTES", "TEOS"])[2],
 "gop": str(c.get_output_path("phase4")),
 "targets": list(c.TARGETS),
 "tmpl_mode": c.PHASE4_TEMPLATE_MODE,
 "ratio_sweep": c.PHASE4_RATIO_SWEEP,
 "solvent_sweep": c.PHASE4_SOLVENT_SWEEP,
 "species": c.SILANE_SPECIES,
 "blocked": sorted(c.SILANE_SPECIES_BLOCKED),
}
print("@@" + json.dumps(out))
"""
    r = _run_probe(code, "BSA")
    assert r.returncode == 0, r.stderr
    d = json.loads([l for l in r.stdout.splitlines() if l.startswith("@@")][0][2:])

    cd = Path(d["cd_tree"])
    mine = Path(d["output_dir"]).resolve()
    assert mine != cd and cd not in mine.parents, \
        f"BSA OUTPUT_DIR {mine} is inside the CD tree {cd}"
    assert d["all_monomers"] == ["APTES", "TEOS"], d["all_monomers"]
    assert d["functional"] == ["APTES"], d["functional"]
    assert d["crosslinkers"] == ["TEOS"], d["crosslinkers"]
    # THE trap: TEOS lives in both SILANE_MONOMERS and CROSSLINKER_LIBRARY and
    # the crosslinker record wins the ALL_MONOMERS merge.  If the toggle only
    # patched SILANE_MONOMERS, ALL_MONOMERS["TEOS"] would keep the OTHER species'
    # string.  Asserted against the species actually selected, so the test stays
    # honest whichever way §1 is set.
    _SPECIES_SMILES = {
        "intact":     {"TEOS": "CCO[Si](OCC)(OCC)OCC", "APTES": "NCCC[Si](OCC)(OCC)OCC"},
        "hydrolyzed": {"TEOS": "O[Si](O)(O)O",         "APTES": "NCCC[Si](O)(O)O"},
    }
    want = _SPECIES_SMILES[d["species"]]
    assert d["teos_smiles"] == want["TEOS"], (d["species"], d["teos_smiles"])
    assert d["aptes_smiles"] == want["APTES"], (d["species"], d["aptes_smiles"])
    # The output tree must be species-tagged, else the acpype/PolCA topology
    # cache (keyed by monomer NAME) makes the toggle appear to do nothing.
    assert mine.name.endswith(d["species"]), mine
    # A blocked species must never be the shipped default.
    assert d["species"] not in d["blocked"], (
        f"SILANE_SPECIES={d['species']!r} is in SILANE_SPECIES_BLOCKED but loaded "
        f"anyway — the §1 import guard did not fire")
    assert d["ionic"] == 0.0, d["ionic"]
    assert d["ph"] == 9.5, d["ph"]
    assert cd not in Path(d["gop"]).resolve().parents, "get_output_path leaked to CD tree"
    assert d["targets"] == ["BSA"], d["targets"]
    assert d["tmpl_mode"] == "ecl2", "PHASE4_TEMPLATE_MODE must stay the literal 'ecl2'"
    assert d["ratio_sweep"] is False and d["solvent_sweep"] is False
    print(f"T4 PASS  BSA isolated; {d['species']} SMILES reached the crosslinker record")


# ── T6 ──────────────────────────────────────────────────────────
def test_blocked_species_refused():
    """SILANE_SPECIES in SILANE_SPECIES_BLOCKED must raise, not warn."""
    src = _CODE_DIR / "pipeline" / "config_BSA.py"
    original = src.read_text(encoding="utf-8")
    marker = 'SILANE_SPECIES = "intact"'
    assert marker in original, "config_BSA.py no longer defaults to intact"
    try:
        src.write_text(original.replace(marker, 'SILANE_SPECIES = "hydrolyzed"', 1),
                       encoding="utf-8")
        r = _run_probe("import pipeline.config", "BSA")
        assert r.returncode != 0, "hydrolyzed loaded without the opt-in — guard dead"
        assert "BLOCKED-ON-ENGINE-FIX" in r.stderr, r.stderr
        # …and the documented opt-in must actually work.
        env = _fresh_env(MIP_EXPERIMENT="BSA", MIP_ALLOW_BLOCKED_SPECIES="1")
        r2 = subprocess.run(
            [sys.executable, "-c",
             "import pipeline.config as c;print('@@'+c.SILANE_SPECIES+'|'+c.OUTPUT_DIR)"],
            env=env, capture_output=True, text=True, cwd=str(_REPO_ROOT))
        assert r2.returncode == 0, r2.stderr
        assert "@@hydrolyzed|" in r2.stdout and "results_BSA_hydrolyzed" in r2.stdout, \
            r2.stdout
    finally:
        src.write_text(original, encoding="utf-8")
    assert src.read_text(encoding="utf-8") == original
    print("T6 PASS  blocked species refused; MIP_ALLOW_BLOCKED_SPECIES opt-in works")


# ── T7 ──────────────────────────────────────────────────────────
def test_guard1_covers_output_dirs():
    """Guard 1 must reject a CD-tree path in OUTPUT_DIRS, not only OUTPUT_DIR.

    The documented 'add a third experiment' recipe says derived-symbol overrides
    go AFTER _rederive(); following it with OUTPUT_DIRS is exactly how a late
    write into results/ sneaks past a scalar-only guard.
    """
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sbx = Path(td) / "sbx"
        (sbx / "code" / "pipeline").mkdir(parents=True)
        for f in ("__init__.py", "config.py", "config_CD.py"):
            shutil.copy2(_CODE_DIR / "pipeline" / f, sbx / "code" / "pipeline" / f)
        cfg = sbx / "code" / "pipeline" / "config.py"
        cfg.write_text(cfg.read_text(encoding="utf-8").replace(
            'KNOWN_EXPERIMENTS = ("CD", "BSA")',
            'KNOWN_EXPERIMENTS = ("CD", "BSA", "LATE")', 1), encoding="utf-8")
        (sbx / "code" / "pipeline" / "config_LATE.py").write_text(
            'OUTPUT_DIR = str(PROJECT_ROOT / "results_LATE")\n'
            '_rederive()\n'
            'OUTPUT_DIRS["phase1"] = str(PROJECT_ROOT / "results" / "phase1")\n',
            encoding="utf-8")
        env = _fresh_env(MIP_EXPERIMENT="LATE")
        env["PYTHONPATH"] = str(sbx / "code")
        r = subprocess.run(
            [sys.executable, "-c",
             "import pipeline.config as c;print('LOADED', c.get_output_path('phase1'))"],
            env=env, capture_output=True, text=True, cwd=str(sbx))
        assert r.returncode != 0, f"late OUTPUT_DIRS override was ACCEPTED:\n{r.stdout}"
        assert "OUTPUT_DIRS['phase1']" in r.stderr, r.stderr
        assert not (sbx / "results").exists(), "guard fired but the CD tree was mkdir'd"
    print("T7 PASS  Guard 1 validates every OUTPUT_DIRS entry, no dir created")


# ── T8 ──────────────────────────────────────────────────────────
def test_cd_pinned_entry_points_refused():
    """CD-pinned scripts (hardcoded results/ + config import) must refuse non-CD."""
    pinned = ["make_presentation_figures.py", "make_process_figures.py",
              "run_membrane_bacteria.py", "run_validation.py"]
    for name in pinned:
        env = _fresh_env(MIP_EXPERIMENT="BSA")
        # argv[0] is what config.py inspects; -c cannot set it, so use a stub
        # file with the pinned basename that does nothing but import config.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            stub = Path(td) / name
            stub.write_text("import pipeline.config\nprint('LOADED')\n")
            r = subprocess.run([sys.executable, str(stub)], env=env,
                               capture_output=True, text=True, cwd=str(_REPO_ROOT))
        assert r.returncode != 0, f"{name} was allowed to run under BSA:\n{r.stdout}"
        assert "CD-PINNED" in r.stderr, (name, r.stderr)
    # …and the same names must still work under CD.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / pinned[0]
        stub.write_text("import pipeline.config\nprint('LOADED')\n")
        r = subprocess.run([sys.executable, str(stub)], env=_fresh_env(),
                           capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert r.returncode == 0 and "LOADED" in r.stdout, r.stderr
    print(f"T8 PASS  {len(pinned)} CD-pinned entry points refuse non-CD, still run under CD")


# ── T9 ──────────────────────────────────────────────────────────
def test_missing_cd_delta_is_survivable():
    """A tracked-files-only deploy without config_CD.py must still import CD…

    …while a non-CD experiment without ITS delta must still hard-fail (its
    OUTPUT_DIR and pinned libraries live there; running it would execute CD's
    values under another name and write into results/).
    """
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sbx = Path(td) / "sbx"
        (sbx / "code" / "pipeline").mkdir(parents=True)
        for f in ("__init__.py", "config.py"):          # NOTE: no config_CD.py
            shutil.copy2(_CODE_DIR / "pipeline" / f, sbx / "code" / "pipeline" / f)
        env = _fresh_env()
        env["PYTHONPATH"] = str(sbx / "code")
        r = subprocess.run(
            [sys.executable, "-c",
             "import pipeline.config as c;print('@@', c.EXPERIMENT, len(c.ALL_MONOMERS))"],
            env=env, capture_output=True, text=True, cwd=str(sbx))
        assert r.returncode == 0, f"CD died without its delta file:\n{r.stderr}"
        assert "@@ CD 33" in r.stdout, r.stdout
        env["MIP_EXPERIMENT"] = "BSA"
        r2 = subprocess.run([sys.executable, "-c", "import pipeline.config"],
                            env=env, capture_output=True, text=True, cwd=str(sbx))
        assert r2.returncode != 0, "BSA loaded without config_BSA.py — would run CD values"
        assert "is missing" in r2.stderr, r2.stderr
    print("T9 PASS  CD survives a missing delta; non-CD still hard-fails")


# ── T10 ─────────────────────────────────────────────────────────
def test_baseline_write_guard():
    """config_baseline.py --write must refuse to snapshot a non-CD experiment."""
    tool = _CODE_DIR / "tools" / "config_baseline.py"
    before = BASELINE_JSON.read_bytes()
    # (1) stale export must be scrubbed, not obeyed
    env = dict(os.environ, MIP_EXPERIMENT="BSA", MIP_CONFIG_QUIET="1")
    r = subprocess.run([sys.executable, str(tool)], env=env,
                       capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["EXPERIMENT"] == "CD", "stale MIP_EXPERIMENT not scrubbed"
    # (2) explicit --experiment + --write must be refused
    r2 = subprocess.run([sys.executable, str(tool), "--experiment", "BSA", "--write"],
                        env=dict(os.environ, MIP_CONFIG_QUIET="1"),
                        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert r2.returncode != 0 and "refusing to write" in r2.stderr, (r2.returncode, r2.stderr)
    # (3) a stale export + --write must ABORT (not scrub-and-write): --write is
    #     the destructive mode, and a dirty env means look before you leap.
    r3 = subprocess.run([sys.executable, str(tool), "--write"], env=env,
                        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert r3.returncode != 0 and "Refusing" in r3.stderr, (r3.returncode, r3.stderr)
    assert BASELINE_JSON.read_bytes() == before, "BASELINE_JSON was overwritten!"
    print("T10 PASS  baseline --write scrubs/aborts on MIP_EXPERIMENT and refuses non-CD")


# ── T5 ──────────────────────────────────────────────────────────
def test_bad_selector_raises():
    for bad in ("bsa", "BSA_v2", "cd"):
        r = _run_probe("import pipeline.config", bad)
        assert r.returncode != 0, f"MIP_EXPERIMENT={bad!r} did NOT raise"
        assert "MIP_EXPERIMENT" in r.stderr and "ImportError" in r.stderr, r.stderr
    print("T5 PASS  typo'd MIP_EXPERIMENT fails loud, never falls back to CD")


def main() -> int:
    fails = 0
    for fn in (test_cd_unchanged, test_rederive_is_noop_for_cd,
               test_import_contract, test_bsa_isolated, test_bad_selector_raises,
               test_blocked_species_refused, test_guard1_covers_output_dirs,
               test_cd_pinned_entry_points_refused,
               test_missing_cd_delta_is_survivable, test_baseline_write_guard):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print(f"{fn.__name__} FAIL: {e}")
    print("ALL PASS" if not fails else f"{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
