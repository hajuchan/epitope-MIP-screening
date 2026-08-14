#!/usr/bin/env python3
"""
Write the pinned Phase 3 stub so Phase 4 can run without Phases 2-3
===================================================================
For BSA the monomer set is FIXED (TEOS + APTES, the user's own bench recipe), so
there is nothing for Phase 2 (docking) or Phase 3 (MMSD/NSGA-II) to search.  But
run_phase4 does not read the pinned composition from config — it loads top_pcs
from ``<OUTPUT_DIR>/phase3/phase3_mmsd_results.json`` on disk.  So the stub has
to exist.

config_BSA.py §11(e) used to describe hand-writing::

    {"BSA": {"top_pcs": [PINNED_PC]}}

That recipe is stale twice over and now fails at TWO gates:

  1. ``run_pipeline._check_phase_completed`` wants a manifest completion record
     with a matching sha256 and a matching input fingerprint, which a
     hand-written file can never satisfy.
  2. ``run_phase4`` raises unless the entry carries
     ``schema == phase3_mmsd._PHASE3_SCHEMA``.

This tool writes a stub that satisfies (2) by IMPORTING the schema constant
(so it cannot drift when Phase 3's contract is bumped) and clears (1) by
removing the run manifest, which ``--adopt-existing-tree`` then rebuilds from
the current inputs.

It reads the composition from ``config.PINNED_PC``, which is what that key is
for.  Nothing is invented here: if PINNED_PC is missing or malformed the tool
refuses rather than guessing a composition for a GPU-week of MD.

Usage
-----
    MIP_EXPERIMENT=BSA python3 code/tools/write_pinned_phase3.py
    MIP_EXPERIMENT=BSA python3 code/tools/write_pinned_phase3.py --dry-run
    MIP_EXPERIMENT=BSA python3 code/tools/write_pinned_phase3.py --keep-manifest

Then::

    MIP_EXPERIMENT=BSA python3 run_BSA.py --target BSA --phase 4 \
        --adopt-existing-tree
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent          # …/code
sys.path.insert(0, str(_CODE_DIR))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--target", default=None,
                    help="target name (default: every entry of TARGETS)")
    ap.add_argument("--output-dir", default=None,
                    help="phase3 directory (default: config's phase3 tree)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the stub, write nothing")
    ap.add_argument("--keep-manifest", action="store_true",
                    help="do NOT remove reports/run_manifest.json (the stub "
                         "will then be refused by _check_phase_completed)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing phase3_mmsd_results.json")
    args = ap.parse_args(argv)

    if not os.environ.get("MIP_EXPERIMENT"):
        print("REFUSING: MIP_EXPERIMENT is not set. This tool writes into the "
              "SELECTED experiment's output tree; running it under the default "
              "(CD) would drop a pinned BSA composition into CD's results/.",
              file=sys.stderr)
        return 2

    import pipeline.config as cfg
    from pipeline.phase3_mmsd import _PHASE3_SCHEMA

    pinned = getattr(cfg, "PINNED_PC", None)
    if not isinstance(pinned, dict):
        print(f"REFUSING: config.PINNED_PC is {pinned!r}, not a dict. This tool "
              f"exists to write the composition the config PINS; it will not "
              f"invent one.", file=sys.stderr)
        return 2
    missing = [k for k in ("pc_id", "monomers", "crosslinker") if not pinned.get(k)]
    if missing:
        print(f"REFUSING: config.PINNED_PC is missing {missing}. A Phase 4 leg "
              f"needs all three to split functional monomers from the "
              f"crosslinker.", file=sys.stderr)
        return 2

    unknown = [m for m in pinned["monomers"] if m not in cfg.ALL_MONOMERS]
    if unknown:
        print(f"REFUSING: PINNED_PC names {unknown}, which are not in "
              f"ALL_MONOMERS ({sorted(cfg.ALL_MONOMERS)}). Phase 4 would fail "
              f"at parameterisation.", file=sys.stderr)
        return 2
    if pinned["crosslinker"] not in pinned["monomers"]:
        print(f"REFUSING: PINNED_PC crosslinker {pinned['crosslinker']!r} is not "
              f"in its own monomer list {pinned['monomers']}.", file=sys.stderr)
        return 2

    targets = [args.target] if args.target else list(cfg.TARGETS)
    entry = {
        "schema": _PHASE3_SCHEMA,
        "top_pcs": [{
            "pc_id": pinned["pc_id"],
            "monomers": list(pinned["monomers"]),
            "crosslinker": pinned["crosslinker"],
            # NOT "fallback_*": this composition did not come from a collapsed
            # NSGA-II front, it was pinned by the experimenter. run_phase4 logs
            # an ERROR for any selected_from starting with "fallback_", and that
            # warning would be false here.
            "selected_from": "manual_pin (config.PINNED_PC)",
            "pinned_by": "code/tools/write_pinned_phase3.py",
            "experiment": getattr(cfg, "EXPERIMENT", None),
            "silane_species": getattr(cfg, "SILANE_SPECIES", None),
        }],
    }
    stub = {t: entry for t in targets}

    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(cfg.get_output_path("phase3")))
    out_path = out_dir / "phase3_mmsd_results.json"

    print(json.dumps(stub, indent=2))
    print(f"\n-> {out_path}")
    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    if out_path.exists() and not args.force:
        print(f"REFUSING: {out_path} already exists. It may be a REAL Phase 3 "
              f"result — overwriting it would replace a searched composition "
              f"with a pinned one silently. Pass --force if you mean it.",
              file=sys.stderr)
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stub, indent=2))
    print(f"WROTE {out_path}")

    if not args.keep_manifest:
        manifest = Path(cfg.get_output_path("reports")) / "run_manifest.json"
        if manifest.exists():
            manifest.unlink()
            print(f"REMOVED {manifest} — re-run Phase 4 with "
                  f"--adopt-existing-tree so it is rebuilt from the CURRENT "
                  f"inputs (this also re-adopts Phase 1).")
        else:
            print(f"(no {manifest} to remove)")

    print("\nNext:\n"
          f"    MIP_EXPERIMENT={os.environ['MIP_EXPERIMENT']} python3 "
          f"run_{os.environ['MIP_EXPERIMENT']}.py --target {targets[0]} "
          f"--phase 4 --adopt-existing-tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
