#!/usr/bin/env python3
"""
Write the <OUTPUT_DIR>/.experiment guard stamp — deliberately, once.
====================================================================
config.py's Guard 2 REFUSES to start if the output tree it resolved is already
stamped for a DIFFERENT experiment.  That guard is strictly read-only: config
never creates the stamp itself, because config.py has no import-time writes
today and must keep none.  This tool is the only writer.

    python3 code/tools/stamp_experiment.py            # show what it would do
    python3 code/tools/stamp_experiment.py --write    # actually write

Each stamp is a 3-4 byte file containing the experiment name.  Nothing else is
created or modified.  By default the CD tree is SKIPPED — stamping it means
writing into the ~500 GB results/ directory the user asked not to touch, and
Guard 1 (the structural OUTPUT_DIR assert in config.py) already blocks the
catastrophic case without any write at all.  Pass --include-cd if you
explicitly want belt-and-braces protection on the CD tree too.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent

_PROBE = (
    "import pipeline.config as c;"
    "print('@@' + c.OUTPUT_DIR)"
)


def resolve_output_dir(experiment: str) -> str:
    """Ask a FRESH interpreter what OUTPUT_DIR that experiment resolves to."""
    env = dict(os.environ)
    env["MIP_EXPERIMENT"] = experiment
    env["MIP_CONFIG_QUIET"] = "1"
    env["PYTHONPATH"] = str(_CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[stamp] {experiment}: config import failed\n{r.stderr}")
    return [l for l in r.stdout.splitlines() if l.startswith("@@")][0][2:]


def known_experiments() -> tuple[str, ...]:
    env = dict(os.environ)
    env.pop("MIP_EXPERIMENT", None)
    env["MIP_CONFIG_QUIET"] = "1"
    env["PYTHONPATH"] = str(_CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-c",
         "import pipeline.config as c;print('@@' + ','.join(c.KNOWN_EXPERIMENTS))"],
        env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    return tuple([l for l in r.stdout.splitlines()
                  if l.startswith("@@")][0][2:].split(","))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="actually write the stamps")
    ap.add_argument("--include-cd", action="store_true",
                    help="also stamp the CD results/ tree (writes into results/)")
    args = ap.parse_args()

    for exp in known_experiments():
        out = Path(resolve_output_dir(exp))
        stamp = out / ".experiment"
        if exp == "CD" and not args.include_cd:
            print(f"[stamp] SKIP  {exp:4s} -> {stamp}  (pass --include-cd to write "
                  f"into the protected CD tree)")
            continue
        if stamp.is_file():
            cur = stamp.read_text().strip()
            print(f"[stamp] {'OK  ' if cur == exp else 'WARN'}  {exp:4s} -> {stamp} "
                  f"already says {cur!r}")
            continue
        if not args.write:
            print(f"[stamp] WOULD WRITE  {exp:4s} -> {stamp}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        stamp.write_text(exp + "\n", encoding="utf-8")
        print(f"[stamp] WROTE  {exp:4s} -> {stamp}")
    if not args.write:
        print("[stamp] dry run — nothing written. Re-run with --write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
