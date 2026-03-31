"""
Phase 1: Epitope Extraction and Structure Preparation
=====================================================
Download PDB/AlphaFold structures, extract ECL2 head region,
analyze physicochemical properties, and validate stability.

Reference:
  Sehit/Altintas, ACS Sensors 2024 — epitope MD stability protocol
  Bossi et al., Anal. Bioanal. Chem. 2021 — epitope selection criteria
  Canfarotta et al., Science Advances 2021 — design principles
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_phase1(target_names: list = None,
               output_dir: str = None,
               skip_stability_md: bool = True) -> dict:
    """
    Phase 1 entry point: prepare epitope structures for all targets.

    Parameters
    ----------
    target_names : list of target names (default: all from config)
    output_dir : output directory
    skip_stability_md : if True, skip MD stability check (fast mode)

    Returns
    -------
    dict : {target_name: {epitope_pdb, receptor_pdbqt, properties, ...}}
    """
    from .config import TARGETS, get_output_path

    if output_dir is None:
        output_dir = str(get_output_path("phase1"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_names is None:
        target_names = list(TARGETS.keys())

    results = {}
    for name in target_names:
        if name not in TARGETS:
            logger.warning(f"Unknown target: {name}, skipping")
            continue
        logger.info(f"\n{'='*20} Phase 1: {name} {'='*20}")
        try:
            result = _prepare_single_target(
                name, TARGETS[name], output_dir, skip_stability_md
            )
            results[name] = result
        except Exception as e:
            logger.error(f"Phase 1 failed for {name}: {e}")
            results[name] = {"error": str(e)}

    # Save results
    results_path = output_dir / "phase1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Phase 1 results → {results_path}")

    return results


def _prepare_single_target(name: str, cfg: dict,
                            output_dir: Path,
                            skip_stability_md: bool) -> dict:
    """Prepare a single target epitope."""
    from .utils_structure import (
        download_structure, extract_epitope, analyze_peptide,
        get_epitope_sequence, check_plddt,
        prepare_receptor_pdbqt, compute_grid_center, compute_grid_size,
    )
    from .config import EPITOPE_PLDDT_THRESHOLD

    target_dir = output_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    result = {"target": name}

    # 1. Download structure
    logger.info(f"[{name}] Downloading structure ({cfg['source']})...")
    full_pdb = download_structure(cfg, target_dir)
    result["full_pdb"] = str(full_pdb)

    # 2. pLDDT check for AlphaFold models
    if cfg["source"] == "alphafold":
        logger.info(f"[{name}] Checking AlphaFold pLDDT scores...")
        plddt = check_plddt(full_pdb, cfg["head_residues"],
                            threshold=EPITOPE_PLDDT_THRESHOLD)
        result["plddt"] = plddt
        if not plddt["pass"]:
            logger.warning(
                f"[{name}] Low pLDDT in head region: {plddt['message']}. "
                "Consider homology modeling from CD81 (5TCX)."
            )

    # 3. Extract epitope (head region)
    logger.info(f"[{name}] Extracting epitope head region "
                f"(residues {cfg['head_residues']})...")
    epitope_pdb = target_dir / f"{name}_epitope.pdb"
    extract_epitope(
        full_pdb, cfg.get("chain", "A"),
        cfg["head_residues"], epitope_pdb
    )
    result["epitope_pdb"] = str(epitope_pdb)

    # 4. Sequence and property analysis
    logger.info(f"[{name}] Analyzing peptide properties...")
    seq = get_epitope_sequence(epitope_pdb)
    result["sequence"] = seq
    result["properties"] = analyze_peptide(
        epitope_pdb, n_glycan_sites=cfg.get("n_glycan_sites", 0)
    )
    _log_properties(name, result["properties"])

    # 5. Grid center and size for AutoDock4
    center = compute_grid_center(epitope_pdb)
    npts = compute_grid_size(epitope_pdb)
    result["grid_center"] = center
    result["grid_npts"] = npts

    # 6. Prepare receptor PDBQT for Phase 2
    logger.info(f"[{name}] Preparing receptor PDBQT...")
    receptor_pdbqt = target_dir / f"{name}_receptor.pdbqt"
    prepare_receptor_pdbqt(epitope_pdb, receptor_pdbqt)
    result["receptor_pdbqt"] = str(receptor_pdbqt)

    # 7. MD stability check (optional)
    if not skip_stability_md:
        logger.info(f"[{name}] Running MD stability check...")
        stability = _run_stability_md(epitope_pdb, target_dir)
        result["stability_md"] = stability
    else:
        logger.info(f"[{name}] Skipping MD stability check")
        result["stability_md"] = {"skipped": True}

    return result


def _run_stability_md(epitope_pdb: Path, work_dir: Path) -> dict:
    """
    Short GROMACS MD to verify epitope conformational stability.

    Per Sehit/Altintas 2024: Run 100ns implicit or 200ns explicit,
    check RMSD < 3.0 Å for the last 50ns.
    """
    from .config import EPITOPE_MD_TIME_NS, EPITOPE_RMSD_THRESHOLD
    from .utils_gromacs import (
        setup_protein_topology, setup_simulation_box,
        run_energy_minimization, run_nvt_equilibration,
        run_npt_equilibration, run_production_md,
        analyze_trajectory,
    )

    md_dir = work_dir / "stability_md"
    md_dir.mkdir(parents=True, exist_ok=True)

    try:
        setup_protein_topology(epitope_pdb, md_dir)
        setup_simulation_box(md_dir / "protein.gro", md_dir)
        run_energy_minimization(md_dir)
        run_nvt_equilibration(md_dir, time_ps=100.0)
        run_npt_equilibration(md_dir, time_ps=100.0)
        run_production_md(md_dir, time_ns=EPITOPE_MD_TIME_NS)
        analysis = analyze_trajectory(md_dir)

        # Stability check
        rmsd_ok = analysis.get("rmsd_last50ns_mean_nm", 999) < (
            EPITOPE_RMSD_THRESHOLD / 10.0  # Å → nm
        )
        analysis["stable"] = rmsd_ok
        if not rmsd_ok:
            logger.warning(
                f"Epitope RMSD > {EPITOPE_RMSD_THRESHOLD} Å — "
                "consider trimming flexible termini"
            )
        return analysis
    except Exception as e:
        logger.error(f"Stability MD failed: {e}")
        return {"error": str(e), "stable": False}


def _log_properties(name: str, props: dict):
    """Log key epitope properties."""
    logger.info(f"[{name}] Epitope properties:")
    logger.info(f"  Sequence: {props.get('sequence', 'N/A')}")
    logger.info(f"  Length:   {props.get('length', 'N/A')} residues")
    logger.info(f"  MW:       {props.get('molecular_weight', 'N/A')} Da")
    logger.info(f"  pI:       {props.get('isoelectric_point', 'N/A')}")
    logger.info(f"  GRAVY:    {props.get('gravy', 'N/A')}")
    logger.info(f"  H-bond D: {props.get('hbond_donors', 'N/A')}")
    logger.info(f"  H-bond A: {props.get('hbond_acceptors', 'N/A')}")
    logger.info(f"  Aromatic: {props.get('aromatic_residues', 'N/A')}")
    logger.info(f"  N-glycan: {props.get('n_glycan_sites_known', 'N/A')} "
                f"(sequons: {props.get('n_glyco_sequons_detected', 'N/A')})")
