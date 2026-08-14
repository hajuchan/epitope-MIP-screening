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
               skip_stability_md: bool = False,
               fresh: bool = False) -> dict:
    """
    Phase 1 entry point: prepare epitope structures for all targets.

    Sehit 2024: MD stability check is mandatory to ensure the epitope
    maintains its conformation during imprinting.

    `fresh` is accepted for uniform threading from the runner; Phase 1's
    outputs are rebuilt end-to-end from the source structure files each
    call, so the flag is currently a no-op but harmless.

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

    # M9: `fresh=True` wipes the per-target subdirs (structure/PDBQT cache
    # + stability MD outputs) inside output_dir. The runner's --fresh already
    # archives the whole phase1/ directory before Phase 1 starts, so this
    # matters only when Phase 1 is invoked directly with fresh=True.
    if fresh:
        import shutil as _sh
        for name in target_names:
            t_dir = output_dir / name
            if t_dir.is_dir():
                _sh.rmtree(t_dir, ignore_errors=True)
                logger.info(f"[fresh] cleared {t_dir}")
        for stale in (output_dir / "phase1_results.json",):
            if stale.exists():
                stale.unlink()
                logger.info(f"[fresh] removed {stale}")

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
    #
    # NOTE: `results` is keyed BY TARGET NAME ONLY. Phase 2 does
    # `target_names = list(phase1_results.keys())`, so any extra bookkeeping
    # key placed in here would be treated as a target and crash on
    # t_result["receptor_pdbqt"]. The ensemble audit therefore goes to its
    # own file, and per-target copies live inside each target's dict.
    results_path = output_dir / "phase1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Phase 1 results → {results_path}")

    audit = _audit_ensemble_symmetry(results)
    audit_path = output_dir / "phase1_ensemble_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    logger.info(f"Phase 1 ensemble audit → {audit_path}")
    _enforce_ensemble_symmetry(audit, skip_stability_md)

    return results


def _audit_ensemble_symmetry(results: dict) -> dict:
    """Collect, per target, how big its receptor ensemble is and how it was sampled.

    BLOCKER 12. Phase 2 keeps a statistic over each target's receptor
    conformers, so ensemble size is part of every cross-target number it
    produces. In the committed CD run CD63 and CD9 had 6 receptors and CD81
    had 1; measured on the .dlg files that accounted for 22% of the
    CD63-vs-CD81 contrast and 53% of the CD9-vs-CD81 contrast and flipped the
    selectivity sign for 4 of 27 monomers, methacrylic acid and acrylic acid
    among them. Nothing in the output said so. This function makes the
    asymmetry a first-class, machine-readable fact.
    """
    per_target = {}
    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        if "error" in r:
            per_target[name] = {"status": "phase1_error",
                                "error": str(r.get("error"))[:300],
                                "n_receptors": None}
            continue
        samp = r.get("ensemble_sampling") or {}
        n_conf = len(r.get("ensemble_receptor_pdbqts") or [])
        per_target[name] = {
            "status": "ok",
            # +1 for the original (crystal/prepared) receptor, which Phase 2
            # always docks against in addition to the MD conformers.
            "n_receptors": n_conf + 1,
            "n_md_conformers": n_conf,
            "sampling_method": samp.get("method"),
            "requested": samp.get("requested"),
            "shortfall": samp.get("shortfall"),
            "sampling_errors": samp.get("errors") or [],
            "stability_verdict": samp.get("stability_verdict"),
        }

    usable = {k: v for k, v in per_target.items() if v["status"] == "ok"}
    sizes = {k: v["n_receptors"] for k, v in usable.items()}
    methods = {k: v["sampling_method"] for k, v in usable.items()}
    distinct_sizes = sorted(set(sizes.values()))
    distinct_methods = sorted({m for m in methods.values() if m is not None})
    return {
        "per_target": per_target,
        "n_receptors_per_target": sizes,
        "sampling_method_per_target": methods,
        "equal_ensemble_size": len(distinct_sizes) <= 1,
        "equal_sampling_method": len(distinct_methods) <= 1,
        "distinct_sizes": distinct_sizes,
        "distinct_methods": distinct_methods,
        "note": (
            "Phase 2 merges docking scores over these receptors. Unequal "
            "ensemble size makes the cross-target contrast partly a function "
            "of how many receptors each target was docked against; unequal "
            "sampling METHOD (k-medoids vs uniform-time fallback) makes it "
            "partly a function of how diverse each ensemble is. Neither is "
            "interpretable as selectivity."),
    }


def _enforce_ensemble_symmetry(audit: dict, skip_stability_md: bool):
    """Fail loudly on an asymmetric ensemble instead of shipping a biased contrast.

    Controlled by ENSEMBLE_REQUIRE_EQUAL_N (config, default True). With it
    False the audit is still written and still logged at ERROR — the run just
    proceeds with the bias on the record. There is no configuration under
    which the asymmetry is silent.
    """
    from .config import ENSEMBLE_DOCKING, ENSEMBLE_REQUIRE_EQUAL_N

    if skip_stability_md or not ENSEMBLE_DOCKING:
        return
    if len(audit["n_receptors_per_target"]) < 2:
        return
    if audit["equal_ensemble_size"] and audit["equal_sampling_method"]:
        logger.info(
            f"Ensemble symmetry OK: {audit['n_receptors_per_target']} "
            f"receptors per target, sampling={audit['distinct_methods']}")
        return

    msg = (
        f"ASYMMETRIC RECEPTOR ENSEMBLE ACROSS TARGETS.\n"
        f"  receptors per target : {audit['n_receptors_per_target']}\n"
        f"  sampling method      : {audit['sampling_method_per_target']}\n"
        f"Phase 2 computes one score per target from these receptors, so an "
        f"unequal count or an unequal sampling method biases every "
        f"cross-target number by construction. On the previous CD run this "
        f"same asymmetry (6/1/6) supplied 22% of the CD63-vs-CD81 contrast, "
        f"53% of the CD9-vs-CD81 contrast, and flipped the selectivity sign "
        f"for 4 of 27 monomers including methacrylic acid and acrylic acid.\n"
        f"Fix the target(s) with a shortfall (see "
        f"phase1_ensemble_audit.json -> per_target[*].sampling_errors) or set "
        f"ENSEMBLE_REQUIRE_EQUAL_N=False to proceed with the bias recorded.")
    logger.error(msg)
    if ENSEMBLE_REQUIRE_EQUAL_N:
        raise RuntimeError(msg)


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
    #
    # SCOPE, NOT SOURCE. A prepared template is a chimera, so cfg["source"]
    # no longer tells you what the B-factor column contains: on CD63 it is
    # pLDDT over the AlphaFold scaffold and deposited crystallographic B over
    # the spliced 9HUR block. Averaging those two units yields a meaningless
    # number that would flag the 1.65 Å graft — the best-determined part of
    # the model — as low confidence. So when a prepared template is in use we
    # restrict the statistic to the residues that are genuinely predicted,
    # which the builder marks with occupancy 0.00.
    _is_prepared = bool(cfg.get("prepared_path"))
    if cfg["source"] == "alphafold" or _is_prepared:
        logger.info(f"[{name}] Checking AlphaFold pLDDT scores"
                    f"{' (predicted residues only)' if _is_prepared else ''}...")
        plddt = check_plddt(full_pdb, cfg["ecl2_range"],
                            threshold=EPITOPE_PLDDT_THRESHOLD,
                            only_predicted=_is_prepared)
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
    _pre_protonation_pdb = ecl2_pdb
    ecl2_pdb = assign_protonation_states(ecl2_pdb, ph=MD_SOLVENT_PH)
    result["protonation"] = _verify_protonation(
        name, _pre_protonation_pdb, ecl2_pdb, MD_SOLVENT_PH)
    result["ecl2_pdb"] = str(ecl2_pdb)

    # 3b-A1: Multi-epitope candidate evaluation (if enabled and candidates listed)
    from .config import PHASE1_EVALUATE_MULTI_EPITOPE
    selected_head_range = cfg["head_residues"]  # legacy default
    if PHASE1_EVALUATE_MULTI_EPITOPE and cfg.get("head_candidates"):
        logger.info(f"[{name}] A1: Evaluating {len(cfg['head_candidates'])} "
                    f"epitope candidates...")
        candidates = evaluate_epitope_candidates(name, cfg, full_pdb, target_dir,
                                                  skip_md=True)
        result["epitope_candidates"] = candidates
        # Pick highest quality_score candidate
        if candidates and "error" not in candidates[0]:
            best = candidates[0]
            selected_head_range = tuple(best["range"])
            logger.info(f"[{name}]   → Best: '{best['name']}' "
                        f"{selected_head_range} score={best['quality_score']}")
            for c in candidates:
                logger.info(f"[{name}]     • {c.get('name','?')}: "
                            f"score={c.get('quality_score','?')}, "
                            f"GRAVY={c.get('gravy',0):.2f}, "
                            f"glycan_in_head={c.get('glycan_in_head', [])}")
            result["selected_head_candidate"] = best["name"]
            result["selected_head_range"] = list(selected_head_range)

    # 3b. Extract head region for synthesis template (16-mer)
    logger.info(f"[{name}] Extracting head region for synthesis template "
                f"(residues {selected_head_range})...")
    head_pdb = target_dir / f"{name}_head.pdb"
    extract_epitope(
        full_pdb, cfg.get("chain", "A"),
        selected_head_range, head_pdb
    )
    result["head_pdb"] = str(head_pdb)
    result["epitope_pdb"] = str(ecl2_pdb)  # docking uses ECL2

    # A2: N-glycan attachment hook. When configured to "external", the
    # extracted ECL2 is REPLACED with a user-provided glycosylated PDB
    # (built via CHARMM-GUI Glycan Reader, GLYCAM, etc.). We do not
    # attempt automatic attachment here — carbohydrate residue geometry
    # is force-field-specific and worth doing right in a dedicated tool.
    try:
        from .config import (PHASE1_GLYCAN_MODE, PHASE1_GLYCAN_INPUT_DIR,
                              PROJECT_ROOT)
    except Exception:
        PHASE1_GLYCAN_MODE = "none"
        PHASE1_GLYCAN_INPUT_DIR = ""
        PROJECT_ROOT = "."
    if PHASE1_GLYCAN_MODE == "external":
        gpath = (Path(PROJECT_ROOT) / PHASE1_GLYCAN_INPUT_DIR
                 / f"{name}_glyco.pdb")
        if not gpath.is_file():
            raise FileNotFoundError(
                f"PHASE1_GLYCAN_MODE=external but {gpath} is missing. "
                f"Build the glycosylated structure via CHARMM-GUI Glycan "
                f"Reader (Man3GlcNAc2 core minimum) and drop the PDB at "
                f"that path, or set PHASE1_GLYCAN_MODE = 'none'.")
        result["ecl2_pdb_bare"] = str(ecl2_pdb)
        result["epitope_pdb"] = str(gpath)   # docking + Phase 4/5 uses glyco
        result["ecl2_pdb"] = str(gpath)
        result["glycan_input"] = str(gpath)
        logger.info(f"[{name}] A2: using glycosylated ECL2 {gpath}")
    elif PHASE1_GLYCAN_MODE == "auto":
        logger.warning(
            f"[{name}] A2: PHASE1_GLYCAN_MODE='auto' is a stub — automatic "
            f"glycan attachment not yet implemented. Falling back to 'none'.")

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

    # B1/B2: Compute SASA + GRAVY for selected head
    from .config import (PHASE1_COMPUTE_SASA, PHASE1_COMPUTE_GRAVY,
                          EPITOPE_GRAVY_MIN, EPITOPE_GRAVY_MAX,
                          EPITOPE_SASA_MIN_A2)
    from .utils_analysis import (gravy_index, compute_sasa_per_residue)

    quality = {}
    if PHASE1_COMPUTE_GRAVY:
        gravy = gravy_index(head_seq)
        balance = ("too_hydrophobic" if gravy > EPITOPE_GRAVY_MAX
                   else "too_hydrophilic" if gravy < EPITOPE_GRAVY_MIN
                   else "balanced")
        quality["gravy"] = round(gravy, 2)
        quality["balance"] = balance
        logger.info(f"[{name}] B2: GRAVY={gravy:+.2f} ({balance})")

    if PHASE1_COMPUTE_SASA:
        try:
            sasa = compute_sasa_per_residue(head_pdb, selected_head_range)
            quality["sasa"] = sasa
            if sasa.get("mean_sasa_A2") is not None:
                accessible = sasa["mean_sasa_A2"] >= EPITOPE_SASA_MIN_A2
                quality["surface_accessible"] = accessible
                logger.info(f"[{name}] B1: SASA mean={sasa['mean_sasa_A2']:.1f} Å² "
                            f"(threshold {EPITOPE_SASA_MIN_A2}, "
                            f"accessible: {'✓' if accessible else '✗'})")
        except Exception as e:
            logger.warning(f"[{name}] B1 SASA failed: {e}")

    result["epitope_quality"] = quality

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

    # 6. Grid center and size
    #
    # The box is defined on the POLYMER-ACCESSIBLE surface when the config
    # declares one (`scored_surface`), and on the head/full-ECL2 otherwise.
    # On an intact vesicle the membrane-occluded residues face lipid, so
    # surface a polymer can never reach should not be inside the search box —
    # and the amount of such surface is strongly target-dependent (CD81
    # closed carries 25/89 lipid-facing residues against CD9's 9/79), which
    # makes it a cross-target confound rather than a uniform inefficiency.
    from .utils_structure import (
        compute_grid_center as _cgc, compute_grid_size as _cgs,
        compute_grid_center_residues as _cgcr,
        compute_grid_size_residues as _cgsr,
        compute_surface_area as _sasa,
    )
    from .config import RECEPTOR_BOX_ON_SCORED_SURFACE
    scored = cfg.get("scored_surface")
    if RECEPTOR_BOX_ON_SCORED_SURFACE and scored:
        center = _cgcr(ecl2_pdb, scored)
        npts = _cgsr(ecl2_pdb, scored)
        result["grid_basis"] = "scored_surface"
        logger.info(f"[{name}] Grid box on scored surface "
                    f"({len(scored)} residues, {len(cfg.get('membrane_occluded', []))} "
                    f"membrane-occluded excluded)")
    else:
        # Legacy: centre on head (variable region), size covers full ECL2.
        center = _cgc(head_pdb)
        npts = _cgs(ecl2_pdb)
        result["grid_basis"] = "head_center_ecl2_size"
    result["grid_center"] = center
    result["grid_npts"] = npts

    # 6b. RECEPTOR DESCRIPTOR — the per-target scale that Phase 2 needs to
    # avoid subtracting docking scores across receptors of different size.
    # Recorded even when it is not used, so the asymmetry can never again be
    # invisible in the output.
    _all_res = sorted({r.get_id()[1] for r in __import__(
        "Bio.PDB", fromlist=["PDBParser"]).PDBParser(QUIET=True)
        .get_structure("e", str(ecl2_pdb)).get_residues()})
    result["receptor_descriptor"] = {
        "n_residues_in_receptor": len(_all_res),
        "n_scored_surface": len(scored) if scored else None,
        "n_membrane_occluded": len(cfg.get("membrane_occluded", []) or []),
        "sasa_total_A2": _sasa(ecl2_pdb),
        "sasa_scored_surface_A2": _sasa(ecl2_pdb, scored) if scored else None,
        "template": cfg.get("prepared_template"),
        "grid_basis": result["grid_basis"],
    }

    # 6. Prepare receptor PDBQT from ECL2 (not head)
    logger.info(f"[{name}] Preparing ECL2 receptor PDBQT...")
    receptor_pdbqt = target_dir / f"{name}_receptor.pdbqt"
    prepare_receptor_pdbqt(ecl2_pdb, receptor_pdbqt)
    result["receptor_pdbqt"] = str(receptor_pdbqt)

    # 6c. RECEPTOR CHARGE — recorded, not merely logged.
    #
    # prepare_receptor_pdbqt() already ENFORCES this (BLOCKER 02: it raises
    # when the PDBQT net charge does not match the pH-7.4 formal charge), but
    # the number only reached the log.  The three targets' formal charges are
    # strongly asymmetric (+4 / -2 / -1 on these templates), and that asymmetry
    # is the whole point of BLOCKER 02, so it belongs in the run record where a
    # reviewer sees it.  Never fatal here — the enforcement is upstream.
    from .utils_structure import verify_receptor_charge
    try:
        result["receptor_charge"] = verify_receptor_charge(ecl2_pdb, receptor_pdbqt)
        _rc = result["receptor_charge"]
        logger.info(f"[{name}] Receptor charge: net={_rc.get('pdbqt_net_charge')} "
                    f"expected={_rc.get('expected_charge')} ok={_rc.get('ok')}")
    except Exception as e:                       # reporting must not fail the phase
        logger.error(f"[{name}] receptor charge check could not be recorded: {e}")
        result["receptor_charge"] = {"error": str(e)}

    # 8. MD stability check on ECL2 (includes disulfide bonds)
    if not skip_stability_md:
        logger.info(f"[{name}] Running ECL2 MD stability check (20ns)...")
        stability = _run_stability_md(ecl2_pdb, target_dir)
        result["stability_md"] = stability

        # 9. Ensemble docking: extract conformers from MD trajectory
        #
        # NOT GATED ON stability["stable"] — that coupling was a cross-target
        # confound, not a safeguard. `stable` is a hard threshold on a
        # continuous quantity (RMSD of the last 50 ns < 3.0 Å), and in the
        # committed CD run CD81 returned 0.3019 nm against CD63's 0.2773 and
        # CD9's 0.1934. A 0.02 Å miss gave CD81 ZERO receptor conformers
        # while the other two got five each; Phase 2 then took the best score
        # over each target's conformers, so the cross-target contrast was
        # partly a best-of-6 vs best-of-1 sampling artefact. The trajectory
        # CD81 needed existed the whole time (results/phase1/CD81/
        # stability_md/md.xtc, 116 MB) — only the extraction was skipped.
        #
        # A flexible epitope is a REASON to sample more conformers, not
        # fewer. The stability verdict stays in the output as QC; it no
        # longer decides how many receptors a target gets to be docked
        # against. See ENSEMBLE_EXTRACT_REQUIRES_STABLE in config.py.
        from .config import (ENSEMBLE_DOCKING, ENSEMBLE_N_CONFORMERS,
                             ENSEMBLE_EXTRACT_REQUIRES_STABLE)
        _stable = bool(stability.get("stable", False))
        if ENSEMBLE_DOCKING and not _stable:
            logger.warning(
                f"[{name}] Epitope flagged UNSTABLE "
                f"(RMSD_last50ns={stability.get('rmsd_last50ns_mean_nm')} nm) — "
                f"extracting conformers anyway so every target gets the same "
                f"ensemble size; treat this target's poses with more caution.")
        if ENSEMBLE_DOCKING and (_stable or not ENSEMBLE_EXTRACT_REQUIRES_STABLE):
            logger.info(f"[{name}] Extracting {ENSEMBLE_N_CONFORMERS} "
                        "conformers for ensemble docking...")
            ext = _extract_md_conformers(
                target_dir / "stability_md",
                target_dir / "conformers",
                n_conformers=ENSEMBLE_N_CONFORMERS,
                reference_pdb=ecl2_pdb,   # the structure the grid box is on
                grid_center=center,
                grid_npts=npts,
            )
            conformers = ext["conformers"]
            result["ensemble_conformers"] = conformers
            # The sampling PROVENANCE travels with the target from here on.
            # Phase 2 keeps a statistic over the ensemble, so "how many
            # conformers and chosen how" is part of the measurement, not a
            # log line. Two targets sampled by different methods, or to
            # different depths, do not produce comparable scores.
            result["ensemble_sampling"] = {
                "method": ext["method"],
                "requested": ext["requested"],
                "n_extracted": ext["n_extracted"],
                "shortfall": ext["shortfall"],
                "frame_indices": ext["frame_indices"],
                "frame_interval_ps": ext["frame_interval_ps"],
                "time_points_ps": ext["time_points_ps"],
                "dropped": ext["dropped"],
                "errors": ext["errors"],
                "stability_verdict": _stable,
            }

            # Prepare PDBQT for each conformer
            if conformers:
                conf_pdbqts = []
                conf_charges = []
                for i, conf_pdb in enumerate(conformers):
                    conf_pdbqt = target_dir / "conformers" / f"conf_{i}.pdbqt"
                    prepare_receptor_pdbqt(Path(conf_pdb), conf_pdbqt)
                    conf_pdbqts.append(str(conf_pdbqt))
                    # Same rationale as the primary receptor above: the charge
                    # is ENFORCED inside prepare_receptor_pdbqt; recorded here
                    # so an ensemble member that drifted is visible in the JSON.
                    try:
                        from .utils_structure import verify_receptor_charge as _vrc
                        conf_charges.append(_vrc(Path(conf_pdb), conf_pdbqt))
                    except Exception as e:
                        logger.error(f"[{name}] conf_{i} charge check failed: {e}")
                        conf_charges.append({"error": str(e)})
                result["ensemble_receptor_pdbqts"] = conf_pdbqts
                result["ensemble_receptor_charges"] = conf_charges
                logger.info(f"[{name}] {len(conf_pdbqts)} ensemble receptors ready")
            else:
                result["ensemble_receptor_pdbqts"] = []
                logger.error(
                    f"[{name}] ZERO ensemble receptors. This target would be "
                    f"docked against 1 receptor while its comparators are "
                    f"docked against {ENSEMBLE_N_CONFORMERS + 1}. run_phase1() "
                    f"checks this across targets before returning.")
    else:
        logger.info(f"[{name}] Skipping MD stability check")
        result["stability_md"] = {"skipped": True}

    return result


def _extract_md_conformers(md_dir: Path, output_dir: Path,
                            n_conformers: int = 5,
                            reference_pdb: Path = None,
                            grid_center: tuple = None,
                            grid_npts: tuple = None) -> dict:
    """
    Extract representative conformers from MD trajectory.

    RETURN TYPE CHANGED (BLOCKER 12). This used to return a bare list of PDB
    paths, which made every way of getting FEWER conformers than asked for
    indistinguishable from success: a k-medoids failure, a dropped
    out-of-box frame and a missing trajectory all just produced a shorter
    list, and the caller had no way to tell. It now returns

        {"conformers": [...], "method": str, "requested": int,
         "n_extracted": int, "shortfall": int, "errors": [...], ...}

    so the CALLER can refuse to ship an ensemble that is smaller, or sampled
    differently, than another target's. The only caller is
    _prepare_single_target() in this file.

    reference_pdb
        Structure the AutoGrid box was computed from. Every extracted frame
        is rigid-body superposed onto it. THIS IS NOT COSMETIC: trjconv
        writes frames in the simulation-box frame, so without this step the
        conformers are translated and rotated away from the docking box and
        the "ensemble" docks into solvent. See superpose_onto_reference().
    grid_center, grid_npts
        If given, every conformer is checked against the actual box and one
        that still falls outside is DROPPED rather than docked.
    """
    from .utils_gromacs import _gmx

    md_dir = Path(md_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"conformers": [], "method": None, "requested": int(n_conformers),
              "n_extracted": 0, "shortfall": int(n_conformers),
              "frame_indices": None, "frame_interval_ps": None,
              "time_points_ps": None, "dropped": [], "errors": []}

    xtc = md_dir / "md.xtc"
    tpr = md_dir / "md.tpr"
    if not xtc.exists() or not tpr.exists():
        msg = (f"MD trajectory not found for conformer extraction "
               f"(xtc={xtc.exists()}, tpr={tpr.exists()} in {md_dir})")
        logger.error("  " + msg + " — this target will get ZERO receptor "
                     "conformers while other targets get a full ensemble. "
                     "That asymmetry is exactly BLOCKER 12.")
        report["method"] = "none"
        report["errors"].append(msg)
        return report

    conformers = []
    try:
        # Get trajectory length
        from .config import EPITOPE_MD_TIME_NS, ENSEMBLE_CLUSTERING_METHOD
        total_ps = EPITOPE_MD_TIME_NS * 1000  # ns to ps

        # A2: Choose conformer selection method.
        #
        # BLOCKER 12 ROOT CAUSE. cluster_conformers_kmedoids() does not RAISE
        # when it cannot cluster — it RETURNS {"error": ..., "frames": None}.
        # So the try/except around it never fired, `frames` was falsy, and the
        # function fell through to uniform time sampling without emitting a
        # single log line. The "diverse ensemble" feature reported as ON while
        # silently degrading, and on a target where the degradation went all
        # the way to zero frames the ensemble vanished. Both the returned
        # error and a raised exception are now handled, both are logged at
        # ERROR, and the method that was actually used travels out in the
        # return value so Phase 2 can refuse to mix sampling methods.
        time_points_ps = []
        method = "uniform"
        if ENSEMBLE_CLUSTERING_METHOD == "kmedoids":
            kmed = None
            try:
                from .utils_analysis import cluster_conformers_kmedoids
                kmed = cluster_conformers_kmedoids(
                    str(xtc), str(tpr), n_clusters=n_conformers, stride=10)
            except Exception as e:
                # ValueError, ImportError, anything — all of it used to be
                # invisible or half-caught.
                kmed = {"error": f"{type(e).__name__}: {e}", "frames": None}
            kmed = kmed or {"error": "k-medoids returned None", "frames": None}
            frames = kmed.get("frames")
            kerr = kmed.get("error")

            # Frame index → simulation time. Derived from the trajectory
            # itself where possible instead of the hardcoded "10 ps per
            # frame", which is only true for MD_TIMESTEP_FS=2 fs and
            # nstxout-compressed=5000 and would silently mis-place every
            # conformer if either ever changed.
            n_total = kmed.get("n_total_frames")
            if isinstance(n_total, int) and n_total > 1:
                frame_interval_ps = total_ps / float(n_total - 1)
            else:
                from .config import MD_TIMESTEP_FS
                frame_interval_ps = (MD_TIMESTEP_FS / 1000.0) * 5000  # mdp
            report["frame_interval_ps"] = round(float(frame_interval_ps), 4)

            if frames:
                time_points_ps = [int(f * frame_interval_ps) for f in frames]
                report["frame_indices"] = list(frames)
                if kerr:
                    method = "kmedoids_degraded"
                    logger.error(
                        f"  A2: K-medoids DEGRADED ({kerr}); it returned "
                        f"{len(frames)} fallback frame(s) instead of true "
                        f"medoids. This target is NOT sampled the same way as "
                        f"one where clustering succeeded.")
                    report["errors"].append(f"kmedoids degraded: {kerr}")
                else:
                    method = "kmedoids"
                    logger.info(f"  A2: K-medoids selected {len(time_points_ps)} "
                                f"diverse conformers (frame indices: {frames}, "
                                f"{frame_interval_ps:.1f} ps/frame)")
            else:
                method = "uniform_after_kmedoids_failure"
                logger.error(
                    f"  A2: K-MEDOIDS PRODUCED NO CONFORMERS "
                    f"({kerr or 'no frames returned'}). Falling back to "
                    f"uniform time sampling. This is NOT a free fallback: the "
                    f"ensemble is no longer a diverse one, and a target that "
                    f"falls back is not sampled comparably with one that does "
                    f"not. The fallback is recorded and Phase 2 refuses to "
                    f"mix sampling methods across targets.")
                report["errors"].append(
                    f"kmedoids produced no frames: {kerr or 'unknown'}")

        if not time_points_ps:
            # Legacy uniform sampling (fallback)
            start_ps = int(total_ps * 0.25)
            interval = (total_ps - start_ps) / (n_conformers + 1)
            time_points_ps = [int(start_ps + (i + 1) * interval)
                              for i in range(n_conformers)]

        report["method"] = method
        report["time_points_ps"] = list(time_points_ps)

        for i, time_ps in enumerate(time_points_ps):
            conf_pdb = output_dir / f"conf_{i}.pdb"

            # Extract single frame using gmx trjconv
            #
            # check=False IS DELIBERATE and is the ONLY _gmx call site outside
            # utils_gromacs.py that tolerates a non-zero return.  _gmx now
            # RAISES by default (BLOCKER 09a); here a failed extraction is a
            # recoverable, per-conformer event — the very next lines detect the
            # missing/short conf_pdb, log it, record it in report["dropped"] and
            # continue.  Letting it raise would abort the whole phase because a
            # single frame could not be dumped.
            _gmx(["trjconv",
                   "-f", str(xtc),
                   "-s", str(tpr),
                   "-o", str(conf_pdb),
                   "-dump", str(time_ps),
                   "-pbc", "mol"],
                  md_dir, input_text="Protein\n", check=False)

            if not (conf_pdb.exists() and conf_pdb.stat().st_size > 100):
                logger.error(f"  conf_{i}: trjconv produced no usable frame at "
                             f"{time_ps} ps — DROPPED")
                report["dropped"].append({"i": i, "time_ps": time_ps,
                                          "reason": "trjconv_no_output"})
                continue

            # Bring the frame back into the reference (docking box) frame.
            if reference_pdb is not None:
                from .utils_structure import superpose_onto_reference
                try:
                    fit = superpose_onto_reference(conf_pdb, reference_pdb,
                                                   conf_pdb)
                    logger.info(f"  conf_{i}: superposed onto reference "
                                f"(CA RMSD {fit['rmsd']} Å over "
                                f"{fit['n_atoms']} atoms)")
                except Exception as e:
                    logger.error(f"  conf_{i}: superposition FAILED ({e}) — "
                                 f"dropping this conformer rather than "
                                 f"docking it in the wrong frame")
                    report["dropped"].append({"i": i, "time_ps": time_ps,
                                              "reason": f"superpose: {e}"})
                    continue

            # Refuse to ship a conformer that does not sit in its own box.
            if grid_center is not None and grid_npts is not None:
                from .utils_structure import fraction_atoms_in_box
                from .config import ENSEMBLE_MIN_BOX_COVERAGE
                cov = fraction_atoms_in_box(conf_pdb, grid_center, grid_npts)
                if cov < ENSEMBLE_MIN_BOX_COVERAGE:
                    logger.error(
                        f"  conf_{i}: only {cov:.1%} of atoms inside the "
                        f"docking box (need {ENSEMBLE_MIN_BOX_COVERAGE:.0%}) "
                        f"— DROPPED. Docking it would score monomers against "
                        f"empty solvent and return near-zero or positive "
                        f"binding energies.")
                    report["dropped"].append(
                        {"i": i, "time_ps": time_ps,
                         "reason": f"box_coverage={cov:.3f}"})
                    continue
                logger.info(f"  conf_{i}: {cov:.1%} of atoms inside box")

            conformers.append(str(conf_pdb))

    except Exception as e:
        logger.error(f"Conformer extraction failed: {type(e).__name__}: {e}")
        report["errors"].append(f"{type(e).__name__}: {e}")

    report["conformers"] = conformers
    report["n_extracted"] = len(conformers)
    report["shortfall"] = int(n_conformers) - len(conformers)
    if report["shortfall"] > 0:
        logger.error(
            f"  ENSEMBLE SHORTFALL: asked for {n_conformers} conformers, got "
            f"{len(conformers)}. Ensemble size is a per-target quantity that "
            f"leaks straight into the cross-target contrast, so a shortfall "
            f"here is a scientific defect, not a performance detail.")
    return report


def _verify_protonation(name: str, src_pdb: Path, out_pdb: Path,
                        ph: float) -> dict:
    """Report what assign_protonation_states() actually did — not what it claims.

    HISTORY (integration pass 2026-08-12).
    assign_protonation_states() used to end every one of its four branches with
    the SAME `shutil.copy2(pdb_path, output_path)`, so a PROPKA that ran, a
    PROPKA that was not installed and a PROPKA that crashed all produced a
    byte-identical file named `*_pH74.pdb` — and every downstream stage treated
    that NAME as evidence that pH 7.4 protonation had been applied. It never
    had been: no residue was renamed in any branch.

    utils_structure now writes a `<stem>_protonation.json` sidecar recording
    which branch it took, what PROPKA predicted, and how many states were
    actually written into the structure. This function prefers that sidecar and
    falls back to its own probe (PROPKA .pka present? output byte-identical to
    input?) for files produced by the old code. Either way the verdict lands in
    phase1_results.json, so the filename is never again the evidence.

    NOTE the sidecar can legitimately report status="propka_ran_but_no_states_
    written": actually renaming HIS->HIP is gated behind
    PROTONATION_APPLY_STATES (default False) because the same PDB is the
    docking receptor. That is a recorded, deliberate omission, not a silent one.
    """
    import hashlib

    src_pdb, out_pdb = Path(src_pdb), Path(out_pdb)
    info = {"ph": ph, "input": str(src_pdb), "output": str(out_pdb)}

    # A sidecar written by a fixed assign_protonation_states() wins if present.
    sidecar = out_pdb.with_name(out_pdb.stem + "_protonation.json")
    if sidecar.is_file():
        try:
            info.update(json.loads(sidecar.read_text()))
            info["source"] = "sidecar"
            return info
        except Exception as e:                       # pragma: no cover
            logger.warning(f"[{name}] unreadable protonation sidecar: {e}")

    pka = src_pdb.with_suffix(".pka")
    info["pka_file"] = str(pka) if pka.is_file() else None
    info["propka_ran"] = bool(pka.is_file())

    try:
        h_in = hashlib.sha256(src_pdb.read_bytes()).hexdigest()
        h_out = hashlib.sha256(out_pdb.read_bytes()).hexdigest()
        info["identical_to_input"] = (h_in == h_out)
    except Exception as e:                           # pragma: no cover
        info["identical_to_input"] = None
        info["error"] = str(e)

    info["states_applied"] = bool(info["propka_ran"]) and (
        info["identical_to_input"] is False)
    info["status"] = ("applied" if info["states_applied"]
                      else "propka_ran_but_no_states_written"
                      if info["propka_ran"]
                      else "propka_did_not_run")

    if not info["propka_ran"]:
        logger.error(
            f"[{name}] PROPKA DID NOT RUN (no {pka.name}). "
            f"'{out_pdb.name}' is a verbatim copy of '{src_pdb.name}' at "
            f"default protonation — it is NOT a pH {ph} structure, despite "
            f"the file name. Recorded as protonation.status="
            f"'propka_did_not_run'.")
    elif info["identical_to_input"]:
        logger.error(
            f"[{name}] PROPKA ran ({pka.name} exists) but "
            f"'{out_pdb.name}' is byte-identical to its input: no HIP/CYM "
            f"assignment was written into the structure. Downstream GROMACS "
            f"and PDBQT preparation use DEFAULT protonation. Recorded as "
            f"protonation.status='propka_ran_but_no_states_written'.")
    return info


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


# ════════════════════════════════════════════════════════════════
# A1: Multi-epitope candidate evaluation + B1/B2 quality scoring
# ════════════════════════════════════════════════════════════════

def evaluate_epitope_candidates(target: str, target_cfg: dict,
                                 pdb_path, output_dir: Path,
                                 skip_md: bool = True) -> dict:
    """A1+B1+B2: Evaluate all head_candidates for a target.

    For each candidate: extract sequence, compute GRAVY, SASA, stability
    proxy. Rank by composite quality score; select best.

    STRUCTURAL INTEGRITY GATE
    -------------------------
    GRAVY and SASA are computed from the sequence that was actually EXTRACTED
    from the template.  If the template does not resolve part of the candidate,
    those residues silently drop out and the candidate is scored on the
    survivors — a 16-mer that the template supplies as 13 residues scores like a
    good 13-mer, not like a broken 16-mer.  That blindness is what let a
    proposed CD81 template swap (5TCX -> 7JIC) look like an improvement while
    destroying 15 of the 16 residues of the head that Phase 5 computes the
    selectivity index from.

    So before scoring, every candidate is audited against the template by
    code/tools/head_integrity.py and any candidate carrying FATAL damage
    (residue deleted from the construct, unmodelled backbone, sequence
    conflict vs UniProt, or a residue whose only modelled conformation is one
    bound to a co-crystallised binder) is disqualified rather than ranked.
    Missing side-chain atoms are recorded but NOT disqualifying — the
    protonation/minimisation prep step rebuilds those routinely.

    The gate is deliberately calibrated so that it re-selects exactly the heads
    already in results/phase1 (CD63 157-172 on AF-P08962, CD81 168-183 on 5TCX,
    CD9 156-171 on 6K4J all pass); see
    code/tests/test_head_integrity_gate.py::test_frozen_selection_survives.

    Returns list of evaluated candidates sorted by quality_score (desc).
    Disqualified candidates are returned too, marked `usable: False`, and sort
    below every usable one so `candidates[0]` is never a broken head.
    """
    from .utils_analysis import check_epitope_quality
    # EPITOPE LENGTH BOUNDS ARE NOW ENFORCED (REVIEW FINDING 18).
    #
    # EPITOPE_MIN_LENGTH / EPITOPE_MAX_LENGTH (9-16, Teixeira 2021: a
    # nonapeptide floor, and >16 residues start folding intramolecularly so the
    # linear epitope stops being the thing being imprinted) were defined in
    # config, documented with a citation, and referenced NOWHERE in the code.
    #
    # This is the right place for the gate: it runs only under
    # PHASE1_EVALUATE_MULTI_EPITOPE, which the BSA intact-protein experiment
    # disables — BSA's "head" is the whole 583-residue chain by design and must
    # not be measured against a linear-epitope length window.
    from .config import EPITOPE_MIN_LENGTH, EPITOPE_MAX_LENGTH

    candidates = target_cfg.get("head_candidates", [])
    if not candidates:
        # Fallback to legacy single head_residues
        candidates = [{
            "range": target_cfg["head_residues"],
            "name": "head_canonical",
            "glycan_in": [],
        }]

    chain = target_cfg.get("chain", "A")
    audit = _head_auditor(pdb_path, chain, target_cfg)

    evaluated = []
    for cand in candidates:
        rng = tuple(cand["range"])

        # Length gate, before any scoring: a candidate outside the imprintable
        # window is disqualified, not ranked. Same treatment as the structural
        # integrity gate below — sorted beneath every usable candidate so
        # candidates[0] can never be one of these.
        cand_len = rng[1] - rng[0] + 1
        if not (EPITOPE_MIN_LENGTH <= cand_len <= EPITOPE_MAX_LENGTH):
            logger.error(
                f"[{target}] candidate '{cand.get('name','?')}' {rng[0]}-{rng[1]} "
                f"DISQUALIFIED on length: {cand_len} residues is outside the "
                f"imprintable window {EPITOPE_MIN_LENGTH}-{EPITOPE_MAX_LENGTH} "
                f"(Teixeira 2021 — below the nonapeptide floor there is too "
                f"little to imprint; above 16 the peptide folds "
                f"intramolecularly and the linear epitope is no longer what is "
                f"being imprinted).")
            evaluated.append({**cand, "length": cand_len, "usable": False,
                              "error": (f"length {cand_len} outside "
                                        f"{EPITOPE_MIN_LENGTH}-{EPITOPE_MAX_LENGTH}"),
                              "quality_score": -float("inf")})
            continue

        try:
            sequence = _extract_sequence(pdb_path, rng)
        except Exception as e:
            evaluated.append({**cand, "error": f"extraction failed: {e}"})
            continue
        integrity = audit(rng)
        if integrity and not integrity["usable"]:
            logger.error(
                f"[{target}] candidate '{cand.get('name','?')}' {rng[0]}-{rng[1]} "
                f"DISQUALIFIED on {Path(pdb_path).name}: "
                f"{integrity['n_fatal']} of {rng[1]-rng[0]+1} residues carry "
                f"fatal damage ({integrity['fatal_summary']}). This template "
                f"cannot supply this head; the selectivity index computed from "
                f"it would not be about this protein.")
            evaluated.append({**cand, "sequence": sequence,
                              "integrity": integrity, "usable": False,
                              "quality_score": -float("inf")})
            continue
        # Quality check (GRAVY + SASA)
        quality = check_epitope_quality(sequence, pdb_path=pdb_path,
                                         residue_range=rng)
        # Glycan colocalization bonus
        glycan_in_head = [g for g in target_cfg.get("n_glycan_positions", [])
                          if rng[0] <= g <= rng[1]]
        glycan_score = len(glycan_in_head) * 0.5  # +0.5 per glycan in head
        composite_score = quality["quality_score"] + glycan_score

        entry = {
            **cand,
            "sequence": sequence,
            "gravy": quality["gravy"],
            "balance": quality["balance"],
            "sasa": quality.get("sasa"),
            "surface_accessible": quality.get("surface_accessible"),
            "glycan_in_head": glycan_in_head,
            "glycan_bonus": glycan_score,
            "quality_score": composite_score,
            "integrity": integrity,
            "usable": True,
        }
        evaluated.append(entry)

    # Sort by score (desc).  Unusable candidates carry -inf and so land last;
    # sorting on (usable, score) keeps that true even if a future scorer ever
    # produces -inf for a usable candidate.
    evaluated.sort(key=lambda e: (e.get("usable", False),
                                  e.get("quality_score", -float("inf"))),
                   reverse=True)

    if evaluated and not evaluated[0].get("usable", False):
        logger.error(
            f"[{target}] EVERY head candidate is disqualified on this "
            f"template. Phase 5 cannot produce a selectivity index for "
            f"{target}. Fix the template or the candidate ranges in "
            f"config_CD.py — do not fall through to a broken head.")
    return evaluated


def _head_auditor(pdb_path, chain, target_cfg):
    """Return audit(range)->integrity dict, or a no-op if the audit can't run.

    The accession is taken from the template's own DBREF where there is one
    (5TCX/6K4J/7JIC are PDB entries), falling back to the configured
    uniprot_id for AlphaFold models, which carry no DBREF.
    """
    import sys
    tools = Path(__file__).resolve().parent.parent / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        from head_integrity import audit_head, accession_for, FATAL_STATUSES
    except ImportError as e:                       # pragma: no cover
        logger.warning(f"head integrity gate unavailable ({e}); "
                       f"candidates will NOT be checked for structural damage")
        return lambda rng: None

    acc = accession_for(pdb_path, chain) or target_cfg.get("uniprot_id")
    if not acc:
        logger.warning(f"no UniProt accession for {pdb_path}; "
                       f"head integrity gate skipped")
        return lambda rng: None

    def audit(rng):
        try:
            a = audit_head(pdb_path, chain, rng, acc)
        except Exception as e:                     # pragma: no cover
            logger.warning(f"head integrity audit failed for {rng}: {e}")
            return None
        by_status = {}
        for e in a["residues"]:
            if e["status"] in FATAL_STATUSES:
                by_status.setdefault(e["status"], []).append(e["resnum"])
        return {
            "template": a["template"], "chain": a["chain"],
            "accession": acc,
            "ok_fraction": a["ok_fraction"],
            "modelled_fraction": a["modelled_fraction"],
            "binder_contacted_fraction": a["binder_contacted_fraction"],
            "fatal_residues": a["fatal_residues"],
            "repairable_residues": a["repairable_residues"],
            "n_fatal": len(a["fatal_residues"]),
            "fatal_summary": "; ".join(
                f"{k}={v}" for k, v in sorted(by_status.items())) or "none",
            "usable": a["usable"],
        }
    return audit


def _extract_sequence(pdb_path, residue_range: tuple) -> str:
    """Extract amino acid sequence from PDB within residue range."""
    aa3to1 = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
        "GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I",
        "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
        "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }
    seen = set()
    seq = []
    start, end = residue_range
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
            except (ValueError, IndexError):
                continue
            if start <= resnum <= end and (resnum, resname) not in seen:
                seen.add((resnum, resname))
                seq.append(aa3to1.get(resname, "X"))
    return "".join(seq)
