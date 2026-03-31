#!/usr/bin/env python
"""
Epitope-MIP Screening Pipeline — Project Root Entry Point
==========================================================
Usage:
    conda activate GROMACS
    cd "/home/chan/Research/Monomer screening in Bio"

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
import sys
from pathlib import Path

# code/ 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent / "code"))

from pipeline.run_pipeline import main
main()
