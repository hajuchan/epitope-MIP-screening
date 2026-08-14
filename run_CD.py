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

# Explicit, and defensive against a stale export in the calling shell.
os.environ["MIP_EXPERIMENT"] = "CD"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "code"))

from pipeline.run_pipeline import main  # noqa: E402

main()
