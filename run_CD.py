#!/usr/bin/env python3
"""Run the CD experiment (CD63 / CD81 / CD9 tetraspanin ECL2 epitope imprinting).

CD is the DEFAULT — `python3 run_pipeline.py ...` and
`cd code && python3 -m pipeline.run_pipeline ...` already do exactly this with
no environment variable at all.  This wrapper exists so the two experiments are
symmetric on the command line, and so that it still selects CD correctly even
in a shell where someone has left `MIP_EXPERIMENT` exported to something else.

    python3 run_CD.py --target CD63 CD81 CD9 --phase all
"""
import os
import pathlib
import sys

# Force UTF-8 text I/O regardless of the launching locale — see run_BSA.py for
# why (an ASCII locale killed a Stage 0 sweep at the .mdp write).
if not sys.flags.utf8_mode and not os.environ.get("_MIP_UTF8_REEXEC"):
    os.environ["_MIP_UTF8_REEXEC"] = "1"
    os.execve(sys.executable,
              [sys.executable, "-X", "utf8", *sys.argv], os.environ)

# Explicit, and defensive against a stale export in the calling shell.
os.environ["MIP_EXPERIMENT"] = "CD"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "code"))

from pipeline.run_pipeline import main  # noqa: E402

# PROPAGATE THE EXIT CODE. main() returns 0 and calls sys.exit() itself for
# every refusal (2 = a phase failed verification, 3 = --resume refused,
# 4 = --fresh refused). A bare main() would have this wrapper exit 0 on a
# return value it discarded, so `run_CD.py ... && next_step` would chain off
# a blocked pipeline.
sys.exit(main() or 0)
