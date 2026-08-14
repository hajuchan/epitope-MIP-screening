#!/usr/bin/env python3
"""
Config regression baseline generator
====================================
Dumps a canonicalized snapshot of EVERY public symbol of ``pipeline.config``
plus a set of *closure probes* (the five functions defined inside config.py
that read module globals through ``__globals__``).

Why the probes matter
---------------------
``get_output_path`` / ``resolve_path`` / ``is_polymerization_compatible`` /
``synthesis_pH_window`` / ``reactivity_ratio_product`` do NOT take their tables
as arguments — they read ``OUTPUT_DIRS`` / ``PROJECT_ROOT`` / ``ALL_MONOMERS`` /
``PH_STABILITY`` / ``QE_PARAMS`` as bare globals of the module in which the
``def`` executed.  Comparing constants alone would therefore MISS the single
worst regression class: a function silently answering with the wrong
experiment's tables.  So we call them and snapshot the answers too.

Usage
-----
    # write / refresh the committed CD contract
    python3 code/tools/config_baseline.py --write

    # print (never write) the snapshot of another experiment
    python3 code/tools/config_baseline.py --experiment BSA

    # snapshot an arbitrary config.py living outside the repo
    python3 code/tools/config_baseline.py --config-file /tmp/head_config.py

The ``--config-file`` mode is what the CD byte-identity check uses: it loads
``git show HEAD:code/pipeline/config.py`` from a temp file, so nothing in the
repo (and certainly nothing in results/) is touched.

NOTE ON SIDE EFFECTS: ``get_output_path`` mkdirs.  All seven CD phase dirs
already exist, so probing creates nothing.  For a fresh experiment tree it
WILL create the (empty) phase directories — that is the pre-existing behaviour
of config.py, not something this tool adds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent          # …/code
_REPO_ROOT = _CODE_DIR.parent
BASELINE_JSON = _CODE_DIR / "pipeline" / "config_baseline_CD.json"


# ── canonicalization ────────────────────────────────────────────
# repr() of a dict-of-set (POLYMERIZATION_COMPATIBILITY) is NOT stable across
# runs — set iteration order depends on hash randomization for str keys only
# when PYTHONHASHSEED varies, but tuples/frozensets can reorder regardless.
# So: sets are sorted, sequences keep order, everything exotic degrades to str.
def canon(v):
    if isinstance(v, (set, frozenset)):
        return {"__set__": sorted(map(str, v))}
    if isinstance(v, dict):
        return {str(k): canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return {"__seq__": [canon(x) for x in v]}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def snapshot(cfg) -> dict:
    """Canonical dict of every public non-callable symbol + closure probes."""
    snap = {}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        val = getattr(cfg, name)
        if callable(val):
            continue                     # functions are probed, not dumped
        snap[name] = canon(val)

    # ── closure probes: prove the five in-config functions resolve their
    #    tables against the CURRENTLY-active experiment, not a stale one.
    snap["@gop"] = {k: str(cfg.get_output_path(k)) for k in sorted(cfg.OUTPUT_DIRS)}
    snap["@rp"] = str(cfg.resolve_path("results/phase1"))
    snap["@ipc1"] = canon(cfg.is_polymerization_compatible(["APTES", "TEOS"]))
    snap["@ipc2"] = canon(cfg.is_polymerization_compatible(["AA", "EGDMA", "APBA"]))
    snap["@ipc3"] = canon(cfg.is_polymerization_compatible(["APTES", "AA"]))
    snap["@sph"] = canon(cfg.synthesis_pH_window(["APTES", "TEOS"]))
    snap["@rrp"] = canon(cfg.reactivity_ratio_product("AA", "EGDMA"))
    return snap


def serialize(snap: dict) -> str:
    return json.dumps(snap, sort_keys=True, indent=1, ensure_ascii=False)


def load_config_module(config_file: str | None):
    """Import pipeline.config, or an arbitrary config.py given by path.

    MIP_EXPERIMENT IS SCRUBBED IN BOTH BRANCHES.  This tool exists to snapshot
    the CD contract; a stale `export MIP_EXPERIMENT=BSA` in the caller's shell
    would otherwise silently produce a BSA snapshot under a file named
    config_baseline_CD.json.  Scrubbing is done here, before the import, because
    config.py reads the variable at import time.  Pass --experiment to opt in to
    snapshotting a non-CD experiment (never combined with --write; see main()).
    """
    if config_file is None:
        sys.path.insert(0, str(_CODE_DIR))
        return importlib.import_module("pipeline.config")
    # Standalone file (e.g. git-HEAD copy in /tmp).  It must still be able to
    # compute PROJECT_ROOT = __file__.parent.parent.parent, so the caller is
    # responsible for placing it three levels below a plausible root.
    spec = importlib.util.spec_from_file_location("_cfg_probe", config_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help=f"write the snapshot to {BASELINE_JSON}")
    ap.add_argument("--config-file", default=None,
                    help="snapshot this config.py instead of pipeline.config")
    ap.add_argument("--experiment", default=None,
                    help="snapshot this experiment instead of CD (never with --write)")
    args = ap.parse_args()

    # ── ENV SCRUB ───────────────────────────────────────────────────
    # Do this BEFORE load_config_module(): config.py reads MIP_EXPERIMENT at
    # import time.  Without the scrub, a stale `export MIP_EXPERIMENT=BSA` in the
    # caller's shell makes this tool emit a BSA snapshot — and the documented
    # remedy for a failing T1 is to re-run with --write, so the stale export
    # would re-poison the CD contract on the second attempt too.
    stale = os.environ.pop("MIP_EXPERIMENT", None)
    if stale and args.write:
        # --write is the destructive mode; a dirty environment is a reason to
        # STOP and let the caller look, not to quietly do something else.
        sys.exit(f"MIP_EXPERIMENT={stale!r} is set in this shell and --write was "
                 f"requested. Refusing: unset it (or use a one-shot `VAR=x cmd` "
                 f"prefix elsewhere) and re-run. {BASELINE_JSON.name} is the CD "
                 f"contract and must only ever be written from a clean CD import.")
    if stale:
        print(f"[baseline] NOTE: ignoring MIP_EXPERIMENT={stale!r} from the "
              f"environment; snapshotting CD.", file=sys.stderr)
    if args.experiment:
        os.environ["MIP_EXPERIMENT"] = args.experiment

    cfg = load_config_module(args.config_file)
    text = serialize(snapshot(cfg))

    if args.write:
        # ── WRITE GUARD ─────────────────────────────────────────────
        # BASELINE_JSON is the single artifact that proves CD is unchanged.  Both
        # its filename and this constant say "CD"; refuse to fill it with
        # anything else, whichever way the experiment got selected.
        got = getattr(cfg, "EXPERIMENT", "CD")
        if args.experiment or got != "CD":
            sys.exit(f"refusing to write {BASELINE_JSON.name} from experiment "
                     f"{(args.experiment or got)!r} — it is the CD contract. "
                     f"Drop --experiment (and any exported MIP_EXPERIMENT).")
        if args.config_file:
            sys.exit(f"refusing to write {BASELINE_JSON.name} from an arbitrary "
                     f"--config-file ({args.config_file}); that mode is for "
                     f"comparison only.")
        # PROVENANCE WARNING. The committed contract is the 150-key PRE-DISPATCH
        # snapshot (config.py as of git HEAD, before the experiment layer). This
        # tool now also picks up the 4 names the dispatcher adds — EXPERIMENT,
        # KNOWN_EXPERIMENTS, EXPERIMENT_LABEL, UNCONSUMED_KEYS — which T1 merely
        # WHITELISTS. Rewriting the file bakes them in and quietly retires that
        # whitelist as a check. Only refresh on an intentional CD change.
        if len(json.loads(text)) != len(json.loads(BASELINE_JSON.read_text("utf-8"))):
            print(f"[baseline] WARNING: key count changes "
                  f"{len(json.loads(BASELINE_JSON.read_text('utf-8')))} -> "
                  f"{len(json.loads(text))}. If you did not intend a CD change, "
                  f"abort and `git checkout -- {BASELINE_JSON.name}`.", file=sys.stderr)
        BASELINE_JSON.write_text(text, encoding="utf-8")
        print(f"[baseline] wrote {BASELINE_JSON}", file=sys.stderr)

    print(text)
    print(f"[baseline] keys={len(json.loads(text))} bytes={len(text)} "
          f"sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
