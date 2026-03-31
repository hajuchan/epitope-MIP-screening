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
               skip_stability_md: bool = False) -> dict:
    """
    Phase 1 entry point: prepare epitope structures for all targets.

    Sehit 2024: MD stability check is mandatory to ensure the epitope
    maintains its conformation during imprinting.

    Parameters
    ----------
    target_names : list of target names (default: all from config)
    output_dir : output directory
    skip_stability_md : if True, skip MD stability check (not recommended)

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
        assign_protonation_states,
    )
    from .config import EPITOPE_PLDDT_THRESHOLD, MD_SOLVENT_PH

    target_dir = output_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    result = {"target": name}

    # 1. Download structure
    logger.info(f"[{name}] Downloading structure ({cfg['source']})...")
    full_pdb = download_structure(cfg, target_dir)
    result["full_pdb"] = str(full_pdb)

    # 2. pLDDT check for AlphaFold models (check ECL2 region)
    if cfg["source"] == "alphafold":
        logger.info(f"[{name}] Checking AlphaFold pLDDT scores...")
        plddt = check_plddt(full_pdb, cfg["ecl2_range"],
                            threshold=EPITOPE_PLDDT_THRESHOLD)
        result["plddt"] = plddt
        if not plddt["pass"]:
            logger.warning(
                f"[{name}] Low pLDDT in ECL2 region: {plddt['message']}. "
                "Consider homology modeling from CD81 (5TCX)."
            )

    # 3a. Extract ECL2 for docking receptor (~90 residues)
    # Uses full ECL2 (stalk + head) to preserve disulfide bonds
    # and structural context. Monomers that bind conserved stalk
    # will show low selectivity and be filtered by ΔΔG.
    logger.info(f"[{name}] Extracting ECL2 for docking receptor "
                f"(residues {cfg['ecl2_range']})...")
    ecl2_pdb = target_dir / f"{name}_ecl2.pdb"
    extract_epitope(
        full_pdb, cfg.get("chain", "A"),
        cfg["ecl2_range"], ecl2_pdb
    )
    # 3a-2. Assign protonation states at pH 7.4 (PROPKA)
    logger.info(f"[{name}] Assigning protonation states at pH {MD_SOLVENT_PH}...")
    ecl2_pdb = assign_protonation_states(ecl2_pdb, ph=MD_SOLVENT_PH)
    result["ecl2_pdb"] = str(ecl2_pdb)

    # 3b. Extract head region for synthesis template (16-mer)
    logger.info(f"[{name}] Extracting head region for synthesis template "
                f"(residues {cfg['head_residues']})...")
    head_pdb = target_dir / f"{name}_head.pdb"
    extract_epitope(
        full_pdb, cfg.get("chain", "A"),
        cfg["head_residues"], head_pdb
    )
    result["head_pdb"] = str(head_pdb)
    result["epitope_pdb"] = str(ecl2_pdb)  # docking uses ECL2

    # 4. Sequence and property analysis (both ECL2 and head)
    logger.info(f"[{name}] Analyzing properties...")
    ecl2_seq = get_epitope_sequence(ecl2_pdb)
    head_seq = get_epitope_sequence(head_pdb)
    result["ecl2_sequence"] = ecl2_seq
    result["head_sequence"] = head_seq
    result["sequence"] = head_seq  # for recipe/report
    result["properties"] = analyze_peptide(
        head_pdb, n_glycan_sites=cfg.get("n_glycan_sites", 0)
    )
    _log_properties(name, result["properties"])
    logger.info(f"[{name}] ECL2: {len(ecl2_seq)} residues (docking receptor)")
    logger.info(f"[{name}] Head: {len(head_seq)} residues (synthesis template)")

    # 5. BLAST uniqueness check (Bossi 2021)
    from .utils_structure import check_epitope_uniqueness
    logger.info(f"[{name}] Checking epitope uniqueness (BLAST)...")
    blast_result = check_epitope_uniqueness(head_seq, name)
    result["blast_uniqueness"] = blast_result
    if blast_result.get("status") == "WARN":
        logger.warning(
            f"[{name}] Epitope may cross-react with other human proteins! "
            "Consider adjusting head_residues in config.py."
        )

    # 6. Grid center and size — centered on HEAD within ECL2
    # Monomers should preferentially dock to head region
    from .utils_structure import compute_grid_center as _cgc
    from .utils_structure import compute_grid_size as _cgs
    # Grid center on head (variable region), but grid covers full ECL2
    center = _cgc(head_pdb)  # center on head
    npts = _cgs(ecl2_pdb)    # size covers full ECL2
    result["grid_center"] = center
    result["grid_npts"] = npts

    # 6. Prepare receptor PDBQT from ECL2 (not head)
    logger.info(f"[{name}] Preparing ECL2 receptor PDBQT...")
    receptor_pdbqt = target_dir / f"{name}_receptor.pdbqt"
    prepare_receptor_pdbqt(ecl2_pdb, receptor_pdbqt)
    result["receptor_pdbqt"] = str(receptor_pdbqt)

    # 8. MD stability check on ECL2 (includes disulfide bonds)
    if not skip_stability_md:
        logger.info(f"[{name}] Running ECL2 MD stability check (20ns)...")
        stability = _run_stability_md(ecl2_pdb, target_dir)
        result["stability_md"] = stability

        # 9. Ensemble docking: extract conformers from MD trajectory
        from .config import ENSEMBLE_DOCKING, ENSEMBLE_N_CONFORMERS
        if ENSEMBLE_DOCKING and stability.get("stable", False):
            logger.info(f"[{name}] Extracting {ENSEMBLE_N_CONFORMERS} "
                        "conformers for ensemble docking...")
            conformers = _extract_md_conformers(
                target_dir / "stability_md",
                target_dir / "conformers",
                n_conformers=ENSEMBLE_N_CONFORMERS,
            )
            result["ensemble_conformers"] = conformers

            # Prepare PDBQT for each conformer
            if conformers:
                conf_pdbqts = []
                for i, conf_pdb in enumerate(conformers):
                    conf_pdbqt = target_dir / "conformers" / f"conf_{i}.pdbqt"
                    prepare_receptor_pdbqt(Path(conf_pdb), conf_pdbqt)
                    conf_pdbqts.append(str(conf_pdbqt))
                result["ensemble_receptor_pdbqts"] = conf_pdbqts
                logger.info(f"[{name}] {len(conf_pdbqts)} ensemble receptors ready")
    else:
        logger.info(f"[{name}] Skipping MD stability check")
        result["stability_md"] = {"skipped": True}

    return result


def _extract_md_conformers(md_dir: Path, output_dir: Path,
                            n_conformers: int = 5) -> list:
    """
    Extract representative conformers from MD trajectory by
    evenly sampling the equilibrated portion.

    Returns list of PDB file paths.
    """
    from .utils_gromacs import _gmx

    md_dir = Path(md_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xtc = md_dir / "md.xtc"
    tpr = md_dir / "md.tpr"
    if not xtc.exists() or not tpr.exists():
        logger.warning("MD trajectory not found for conformer extraction")
        return []

    conformers = []
    try:
        # Get trajectory length
        from .config import EPITOPE_MD_TIME_NS
        total_ps = EPITOPE_MD_TIME_NS * 1000  # ns to ps
        # Sample from last 75% of trajectory (skip equilibration)
        start_ps = int(total_ps * 0.25)
        interval = (total_ps - start_ps) / (n_conformers + 1)

        for i in range(n_conformers):
            time_ps = int(start_ps + (i + 1) * interval)
            conf_pdb = output_dir / f"conf_{i}.pdb"

            # Extract single frame using gmx trjconv
            _gmx(["trjconv",
                   "-f", str(xtc),
                   "-s", str(tpr),
                   "-o", str(conf_pdb),
                   "-dump", str(time_ps),
                   "-pbc", "mol"],
                  md_dir, input_text="Protein\n")

            if conf_pdb.exists() and conf_pdb.stat().st_size > 100:
                conformers.append(str(conf_pdb))

    except Exception as e:
        logger.warning(f"Conformer extraction failed: {e}")

    return conformers


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
