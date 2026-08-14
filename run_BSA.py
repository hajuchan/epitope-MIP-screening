#!/usr/bin/env python3
"""Run the BSA sol-gel imprinting experiment (TEOS + APTES, 1% Tween20 / DI water).

The experiment name is in the FILENAME you type, so nobody has to remember —
or, worse, `export` — MIP_EXPERIMENT.  An exported variable leaks into every
later command in that shell, every tmux pane and every nohup child, and
--resume defaults to True, so a stale export can silently resume the wrong
experiment days later.  This sets it for THIS PROCESS ONLY.

    python3 run_BSA.py --target BSA --phase 1 --skip-md
    python3 run_BSA.py --target BSA --phase 4

Equivalent to:
    MIP_EXPERIMENT=BSA python3 -m pipeline.run_pipeline ...
"""
import os
import pathlib
import sys

# Re-exec in UTF-8 mode BEFORE anything else.  .mdp/.top/.itp/report files are
# written with Python text I/O, which encodes using the *ambient locale*.  Under
# LC_ALL=C with PYTHONUTF8=0 and PYTHONCOERCECLOCALE=0 that resolves to ASCII,
# and then a single non-ASCII character in any template aborts the run mid-MD.
# That is exactly how a full 6-point Stage 0 sweep died after NPT.  Fixing the
# ~235 individual write sites is unreliable; forcing the interpreter default is
# not.  The sentinel makes the re-exec strictly once, so it cannot loop.
if not sys.flags.utf8_mode and not os.environ.get("_MIP_UTF8_REEXEC"):
    os.environ["_MIP_UTF8_REEXEC"] = "1"
    os.execve(sys.executable,
              [sys.executable, "-X", "utf8", *sys.argv], os.environ)

# Must be set BEFORE the first pipeline import — config.py reads it at import
# time and 7 sites bind config values at module scope (run_pipeline.py:27 bakes
# OUTPUT_DIR into the --output-dir argparse default), so it is far too late by
# the time parse_args() runs.
os.environ["MIP_EXPERIMENT"] = "BSA"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "code"))

from pipeline.run_pipeline import main  # noqa: E402

# PROPAGATE THE EXIT CODE — see run_CD.py for the full note.
sys.exit(main() or 0)
