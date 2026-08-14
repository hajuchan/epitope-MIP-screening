#!/usr/bin/env python
"""
Epitope-MIP Screening Pipeline — Project Root Entry Point
==========================================================
Usage:
    conda activate GROMACS
    cd "/home/chan/Research/Monomer_screening_in_Bio"

    # Full pipeline (Phase 1→2→3→4→5)
    python run_pipeline.py

    # Specific target
    python run_pipeline.py --target CD63

    # Specific phase
    python run_pipeline.py --phase 1        # Epitope preparation
    python run_pipeline.py --phase 2        # SMD docking
    python run_pipeline.py --phase 3        # MMSD docking
    python run_pipeline.py --phase 4        # MD validation
    python run_pipeline.py --phase 5        # Recipe generation

    # Options
    python run_pipeline.py --skip-stability-md   # Skip Phase 1 MD
    python run_pipeline.py --quick-md            # 50ns instead of 200ns
    python run_pipeline.py --report              # Generate HTML report

Configuration:
    All settings are in code/pipeline/config.py
"""
import os
import sys
from pathlib import Path

# Force UTF-8 text I/O regardless of the launching locale — see run_BSA.py for
# why (an ASCII locale killed a Stage 0 sweep at the .mdp write).
if not sys.flags.utf8_mode and not os.environ.get("_MIP_UTF8_REEXEC"):
    os.environ["_MIP_UTF8_REEXEC"] = "1"
    os.execve(sys.executable,
              [sys.executable, "-X", "utf8", *sys.argv], os.environ)

# code/ 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent / "code"))

from pipeline.run_pipeline import main

# Propagate the exit code. main() returns 0 on a clean finish and uses
#   2 = a phase FAILED VERIFICATION and the run was stopped
#   3 = resume refused (no manifest, input drift, or --no-resume without --fresh)
#   4 = --fresh refused (could not dispose of state, or an incompatible flag)
# Refusals call sys.exit() internally so they already propagate, but a bare
# `main()` discarded the RETURN value, so `python run_pipeline.py && next_step`
# chained off a blocked pipeline. run_CD.py / run_BSA.py already do this.
sys.exit(main() or 0)
