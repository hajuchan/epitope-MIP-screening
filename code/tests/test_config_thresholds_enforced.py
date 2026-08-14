"""Every configured THRESHOLD/TOL/MIN/MAX/CUTOFF must actually be enforced.

REVIEW FINDING 18. Several config keys were defined, documented with a
citation, and referenced NOWHERE in the code — or imported into a function and
then never used. They read as authoritative knobs while being completely
inert:

  * MMSD_MIN_COMBO_SIZE   imported into _run_greedy_mmsd, never referenced;
                          all three optimizers used a hardcoded `min_size=2`
  * MMSD_MAX_COMBO_SIZE   never referenced; `max_size=6` hardcoded
  * EPITOPE_MIN_LENGTH    never referenced (Teixeira 2021 nonapeptide floor)
  * EPITOPE_MAX_LENGTH    never referenced
  * EPITOPE_RMSD_MAX      a dead exact duplicate of EPITOPE_RMSD_THRESHOLD
  * MAX_BACKBONE_HBOND_RATIO  imported by phase2_smd, but analyze_hbond_types
                          hardcoded 0.3 internally, so the knob did nothing

The values happened to coincide with the hardcoded literals, so nothing looked
wrong — until someone edited config and the pipeline silently ignored them.

This test fails when a NEW such key appears, or when an existing one loses its
last real use. Keys that are inert ON PURPOSE go in _DELIBERATELY_INERT with
the reason; that list is the place to argue about them, not the code.
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

_PATTERN = re.compile(r"THRESHOLD|TOL|MIN|MAX|CUTOFF")

# Inert by deliberate decision, each with the reason it is not enforced.
# Adding a name here is a claim that a human decided this; keep the reason.
_DELIBERATELY_INERT = {
    "POSE_CLUSTERING_MIN_SIZE": (
        "A3 pose clustering only started producing output after the DLG parser "
        "was fixed; applying the min-size filter at the same moment would mean "
        "the first real output anyone sees is already filtered. Documented at "
        "phase2_smd.py:1397. Every cluster carries its own cluster_size."),
    "REBINDING_RMSD_THRESHOLD": (
        "Retired as a decision threshold when the Phase 5 observable moved to "
        "persistent contacts — the old 5.0 A cut sat entirely outside the "
        "observed 2.04-3.87 A range, so `escaped` was False for every leg that "
        "ever ran. Still imported so the config contract and the BSA override "
        "stay valid; explicitly ignored, see phase5_rebinding.py:31."),
}


def _config_keys():
    import pipeline.config as config
    return {k for k in dir(config) if k.isupper() and _PATTERN.search(k)}


def _used_keys(keys):
    """Names genuinely READ somewhere in code/, resolving import aliases."""
    used = set()
    for path in (ROOT / "code").rglob("*.py"):
        if "config" in path.name or path.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue

        alias = {}                     # local name -> config key
        import_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.add(node.lineno)
                # multi-line from-imports occupy a range
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    import_lines.add(ln)
            if isinstance(node, ast.ImportFrom) and node.module \
                    and "config" in node.module:
                for a in node.names:
                    if a.name in keys:
                        alias[a.asname or a.name] = a.name

        for node in ast.walk(tree):
            # a real read: the local name used outside its own import statement
            if isinstance(node, ast.Name) and node.id in alias \
                    and node.lineno not in import_lines:
                used.add(alias[node.id])
            # getattr(config, "KEY", default) and _cfg("KEY", default)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value in keys:
                used.add(node.value)
    return used


def test_every_threshold_key_is_enforced():
    keys = _config_keys()
    assert keys, "no threshold-like config keys found — the scan is broken"

    unenforced = sorted(keys - _used_keys(keys) - set(_DELIBERATELY_INERT))
    assert not unenforced, (
        "these config keys are defined but NEVER READ, so editing them changes "
        "nothing:\n  " + "\n  ".join(unenforced) +
        "\n\nEither enforce each at the point it is imported, delete it from "
        "config (and regenerate config_baseline_CD.json), or add it to "
        "_DELIBERATELY_INERT in this file WITH the reason.")


def test_deliberately_inert_keys_still_exist():
    """Stale entries in the allowlist hide a key that was genuinely removed."""
    import pipeline.config as config
    missing = [k for k in _DELIBERATELY_INERT if not hasattr(config, k)]
    assert not missing, (
        f"_DELIBERATELY_INERT names keys that no longer exist in config: "
        f"{missing} — remove them from the allowlist")


if __name__ == "__main__":
    test_every_threshold_key_is_enforced()
    test_deliberately_inert_keys_still_exist()
    print("config threshold enforcement: OK")
