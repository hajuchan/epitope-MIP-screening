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

# Must be set BEFORE the first pipeline import — config.py reads it at import
# time and 7 sites bind config values at module scope (run_pipeline.py:27 bakes
# OUTPUT_DIR into the --output-dir argparse default), so it is far too late by
# the time parse_args() runs.
os.environ["MIP_EXPERIMENT"] = "BSA"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "code"))

from pipeline.run_pipeline import main  # noqa: E402

main()
