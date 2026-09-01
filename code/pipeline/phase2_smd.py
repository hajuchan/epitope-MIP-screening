"""
Phase 2: Single Monomer Docking (SMD)
=====================================
Screen all functional monomers against each epitope target
using AutoDock4, compute selectivity matrix, and filter candidates.

Reference:
  Rajpal et al., Sci. Rep. 2024 — Table 1 (SMD screening)
  Rajpal & Mizaikoff, J. Mater. Chem. B 2022 — MMSD methodology
"""

import json
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_phase2(phase1_results: dict = None,
               target_names: list = None,
               output_dir: str = None,
               fresh: bool = False) -> dict:
    """
    Phase 2 entry point: SMD screening of all monomers × all targets.

    `fresh` is accepted for uniform threading from the runner; Phase 2 does
    per-(target, monomer) docking and reuses any per-pair result JSONs it
    finds on disk. Currently the flag is a no-op — a future improvement is
    to have it discard per-pair results, but on-disk state is already
    invalidated by the runner archiving `phase2/` when --fresh is used.

    Parameters
    ----------
    phase1_results : output from Phase 1 (epitope structures)
    target_names : filter to specific targets
    output_dir : output directory

    Returns
    -------
    dict : {
        "be_matrix": {target: {monomer: energy}},
        "selectivity": {target: {monomer: <SELECTIVITY_METRIC value>},
                        "_detail": {target: {monomer: {all metrics}}},
                        "_metric": str, "_exchangeability": {...}},
        "receptor_exchangeability": {...},   # may cross-target scores be
                                             # subtracted at all?
        "n_conformers_per_target": {target: N},
        "filtered": {target: [monomer_names]},
    }
    """
    from .config import (TARGETS, FUNCTIONAL_MONOMERS, N_WORKERS,
                         get_output_path, AUTODOCK4_GA_RUNS,
                         SMD_BE_THRESHOLD)

    if output_dir is None:
        output_dir = str(get_output_path("phase2"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # M9: real fresh — wipe every per-target smd_* subdir + top-level result
    # JSON so cached per-(target, monomer) docking outputs cannot leak.
    if fresh:
        import shutil as _sh
        for sub in output_dir.glob("smd_*"):
            if sub.is_dir():
                _sh.rmtree(sub, ignore_errors=True)
                logger.info(f"[fresh] cleared {sub}")
        for stale in (output_dir / "phase2_smd_results.json",):
            if stale.exists():
                stale.unlink()
                logger.info(f"[fresh] removed {stale}")

    # Load Phase 1 results if not provided
    if phase1_results is None:
        p1_path = get_output_path("phase1") / "phase1_results.json"
        if p1_path.exists():
            with open(p1_path) as f:
                phase1_results = json.load(f)
        else:
            raise FileNotFoundError(
                "Phase 1 results not found. Run Phase 1 first."
            )

    if target_names is None:
        # Underscore-prefixed keys are bookkeeping, never targets. Phase 1
        # keeps its ensemble audit in a separate file for this reason, but a
        # hand-edited or older phase1_results.json may carry one, and
        # treating it as a target would crash on t_result["receptor_pdbqt"].
        target_names = [t for t in phase1_results.keys()
                        if not str(t).startswith("_")]

    # 1. Prepare all monomer PDBQTs
    logger.info("Preparing monomer PDBQT files...")
    monomer_dir = output_dir / "monomers"
    monomer_pdbqts = _prepare_all_monomers(FUNCTIONAL_MONOMERS, monomer_dir)

    # 2. Run SMD for all (target, monomer) pairs
    logger.info(f"Starting SMD: {len(target_names)} targets × "
                f"{len(monomer_pdbqts)} monomers = "
                f"{len(target_names) * len(monomer_pdbqts)} dockings")

    be_matrix = {}  # {target: {monomer: binding_energy}}
    all_dock_results = {}
    n_conformers_used = {}  # {target: N} — recorded so unequal ensemble
                            # sizes can never again be invisible downstream
    ensembles_used = {}     # {target: [receptor pdbqt paths]} — reused for
                            # the decoy baseline so the enrichment factor is
                            # computed over the same receptors
    missing_scores = {}     # {target: [monomers with no usable docking]}

    # Glycan-aware receptor selection.  When PHASE1_GLYCAN_MODE == "explicit"
    # and Phase 1 built a CD63_receptor_glyco.pdbqt alongside the naked one,
    # Phase 2 docks against the GLYCO receptor for CD63 so the boronate-diol
    # selectivity mechanism is measured.  CD9/CD81 keep the naked receptor
    # (they carry no ECL2-internal glycans, and swapping only CD63 preserves
    # the cross-target contrast).  The glyco path is gated on the CONFIG
    # value AND on the presence of "receptor_pdbqt_glyco" in phase1_results —
    # a Phase 1 run without the glyco branch (e.g. resumed from before this
    # patch) transparently falls back to the naked path.
    from .config import PHASE1_GLYCAN_MODE as _PH1_GLYCAN_MODE
    _use_glyco_for = {t for t in target_names
                      if _PH1_GLYCAN_MODE == "explicit"
                      and t == "CD63"
                      and (phase1_results.get(t) or {}).get("receptor_pdbqt_glyco")}
    if _use_glyco_for:
        logger.info(f"Phase 2: glycosylated receptor active for "
                    f"{sorted(_use_glyco_for)} (PHASE1_GLYCAN_MODE="
                    f"{_PH1_GLYCAN_MODE!r})")

    for target in target_names:
        t_result = phase1_results.get(target, {})
        if "error" in t_result:
            logger.warning(f"Skipping {target} (Phase 1 error)")
            continue

        # Glyco or naked?  A single decision, recorded on the per-target dict
        # so downstream reports can tell which receptor scored each row.
        if target in _use_glyco_for:
            receptor_pdbqt = Path(t_result["receptor_pdbqt_glyco"])
            # Glyco grid is task-mandated: mean Asn Cα + max glycan distance
            # + 5 Å.  Falls back to the naked grid if Phase 1 could not
            # compute it (e.g. extraction succeeded but Bio.PDB unavailable).
            center = tuple(t_result.get("grid_center_glyco")
                            or t_result["grid_center"])
            npts = tuple(t_result.get("grid_npts_glyco")
                          or t_result["grid_npts"])
            logger.info(f"  [{target}] using GLYCOSYLATED receptor "
                        f"{receptor_pdbqt.name}, grid center={center}, "
                        f"npts={npts}")
        else:
            receptor_pdbqt = Path(t_result["receptor_pdbqt"])
            center = tuple(t_result["grid_center"])
            npts = tuple(t_result["grid_npts"])

        # Ensemble docking: collect all receptor PDBQTs (original + MD conformers).
        # NOTE: MD conformers are naked ECL2 (Phase 1 stability MD does not carry
        # glycans), so we do NOT add them to the glyco ensemble — mixing glyco +
        # naked receptors would score the same monomer against two different
        # chemical environments and merge them by target.
        ensemble_pdbqts = [receptor_pdbqt]
        if "ensemble_receptor_pdbqts" in t_result and target not in _use_glyco_for:
            ensemble_pdbqts.extend(
                Path(p) for p in t_result["ensemble_receptor_pdbqts"]
                if Path(p).exists()
            )

        # M8 FIX (audit): Boltzmann-average conformer merge
        #   BE_merged(m,T) = -kT * ln( mean_i(exp(-BE_i(m,T)/kT)) )
        # has a Jensen bias that skews averaged energies DOWNWARD (more
        # favourable). The bias scales with the number of conformers
        # averaged: CD9/CD81 had n=6 MD conformers while CD63 had n=1
        # (glyco receptor + no glyco MD ensemble), so CD9/CD81 got a
        # systematic n=6-vs-n=1 Boltzmann "bonus" that artificially
        # INFLATED their selectivity vs CD63.
        #
        # Fix (simplest, task-mandated): in explicit-glyco mode force EVERY
        # target to n=1 (keep only the crystal conformer). This eliminates
        # the unequal-N bias entirely and keeps the naked-mode path
        # (PHASE1_GLYCAN_MODE=="none") byte-identical.
        if _PH1_GLYCAN_MODE == "explicit" and len(ensemble_pdbqts) > 1:
            dropped = len(ensemble_pdbqts) - 1
            ensemble_pdbqts = [ensemble_pdbqts[0]]
            logger.warning(
                f"  [{target}] M8: PHASE1_GLYCAN_MODE=='explicit' and "
                f"CD63 has n=1 glyco receptor. Forcing this target to "
                f"n=1 as well (dropped {dropped} MD conformer(s)) to "
                f"eliminate the Boltzmann-merge Jensen bias that would "
                f"otherwise skew this target's BE more negative than "
                f"CD63 purely by having more conformers averaged.")

        logger.info(f"\n--- SMD for {target} "
                    f"({len(ensemble_pdbqts)} receptor conformer(s)) ---")

        # Sullivan 2019: predict binding sites for focused docking
        from .config import USE_BINDING_SITE_PREDICTION, BINDING_SITE_TOOL
        binding_sites = None
        if USE_BINDING_SITE_PREDICTION:
            from .utils_analysis import predict_binding_sites
            epitope_pdb = Path(t_result["epitope_pdb"])
            binding_sites = predict_binding_sites(
                epitope_pdb, method=BINDING_SITE_TOOL,
                output_dir=output_dir / f"sites_{target}",
            )
            logger.info(f"  [{target}] {len(binding_sites)} binding sites identified")

        # Dock to each conformer, then merge across conformers.
        per_conformer = {}          # monomer -> [(conf_label, result), ...]
        for ci, conf_pdbqt in enumerate(ensemble_pdbqts):
            conf_label = "crystal" if ci == 0 else f"md_conf{ci}"
            if len(ensemble_pdbqts) > 1:
                logger.info(f"  [{target}] Ensemble conformer {ci+1}/"
                            f"{len(ensemble_pdbqts)} ({conf_label})")

            # The .pdb the PDBQT was built from — needed for the
            # membrane-accessibility filter, which tests pose contacts against
            # RESIDUE NUMBERS and so cannot use the PDBQT (no residue records
            # survive prepare_receptor4's type column reliably). Ensemble
            # conformers keep a .pdb sibling; the crystal receptor's source is
            # the epitope PDB itself. extract_epitope preserves the original
            # numbering, so config's scored_surface / membrane_occluded lists
            # line up with it directly.
            _rec_pdb = Path(conf_pdbqt).with_suffix(".pdb")
            if not _rec_pdb.exists():
                _rec_pdb = Path(t_result["epitope_pdb"])

            conf_results = _run_smd_for_target(
                target, conf_pdbqt, monomer_pdbqts,
                center, npts, output_dir / f"smd_{target}_{conf_label}",
                ga_runs=AUTODOCK4_GA_RUNS,
                n_workers=N_WORKERS,
                binding_sites=binding_sites,
                receptor_pdb=_rec_pdb,
                target_cfg=TARGETS.get(target, {}),
            )

            for m, r in conf_results.items():
                if r.get("mean_cluster_energy") is None:
                    continue        # skip failed docking
                per_conformer.setdefault(m, []).append((conf_label, r))

        target_results = _merge_across_conformers(target, per_conformer)
        n_conformers_used[target] = len(ensemble_pdbqts)
        ensembles_used[target] = list(ensemble_pdbqts)

        # A monomer that produced no usable score against ANY conformer never
        # reaches _merge_across_conformers, so it would otherwise vanish from
        # both the matrix and the log. Name it.
        never_scored = sorted(set(monomer_pdbqts) - set(target_results))
        if never_scored:
            logger.error(
                f"  [{target}] {len(never_scored)} monomer(s) produced NO "
                f"usable docking against ANY of the {len(ensemble_pdbqts)} "
                f"receptor conformers and are ABSENT from the matrix (not "
                f"zero-filled): {never_scored}")
        missing_scores[target] = never_scored

        # Sullivan 2019: analyze backbone vs sidechain H-bonds
        from .config import BACKBONE_HBOND_PENALTY
        if BACKBONE_HBOND_PENALTY:
            target_results = _analyze_hbond_types_for_target(
                target, target_results, t_result, output_dir)

        # Only include monomers with successful docking (non-None BE)
        failed = [m for m, r in target_results.items()
                  if r.get("mean_cluster_energy") is None]
        if failed:
            logger.warning(f"  [{target}] {len(failed)} monomer(s) failed docking: {failed}")

        be_matrix[target] = {
            m: r["mean_cluster_energy"]
            for m, r in target_results.items()
            if r.get("mean_cluster_energy") is not None
        }
        all_dock_results[target] = target_results

    # 2b. Sehit 2024: short monomer-epitope contact MD
    from .config import MONOMER_CONTACT_MD, MONOMER_CONTACT_MD_NS
    contact_scores = {}
    if MONOMER_CONTACT_MD:
        logger.info("\nRunning monomer-epitope contact MD (Sehit 2024)...")
        contact_scores = _run_contact_md(
            phase1_results, target_names, monomer_pdbqts,
            output_dir, time_ns=MONOMER_CONTACT_MD_NS,
        )

    # B3: Decoy baseline for enrichment factor
    from .config import PHASE2_DECOY_BASELINE
    decoy_results = {}
    if PHASE2_DECOY_BASELINE:
        logger.info("\nB3: Running decoy monomer baseline (enrichment factor)...")
        for target in target_names:
            t_result = phase1_results.get(target, {})
            if "error" in t_result:
                continue
            try:
                d = evaluate_decoy_baseline(
                    target,
                    Path(t_result["receptor_pdbqt"]),
                    tuple(t_result["grid_center"]),
                    tuple(t_result["grid_npts"]),
                    output_dir,
                    ensemble_pdbqts=ensembles_used.get(target),
                    # Same membrane-accessibility filter as the real monomers:
                    # EF compares the two arms, so both must be measured on the
                    # polymer-accessible surface or the ratio is meaningless.
                    receptor_pdb=Path(t_result["epitope_pdb"]),
                    target_cfg=TARGETS.get(target, {}),
                )
                decoy_results[target] = d
                logger.info(f"  {target}: {d.get('n_valid_decoys', 0)}/"
                            f"{len(d.get('decoy_monomers_tested', []))} decoys "
                            f"actually docked")
            except Exception as e:
                logger.error(f"  {target} decoy baseline failed "
                             f"({type(e).__name__}: {e}) — no enrichment "
                             f"factor for this target")
                decoy_results[target] = {"error": f"{type(e).__name__}: {e}",
                                         "n_valid_decoys": 0, "docked": False,
                                         "decoy_be_matrix": {}}

    # 3. Compute selectivity matrix
    #
    # Receptor exchangeability is established BEFORE any cross-target
    # quantity is computed, and travels with the numbers into the JSON and
    # the CSV. An unequal ensemble size or a >15% difference in scored
    # surface does not stop the calculation — it labels every cross-target
    # number produced from it.
    exchangeability = _check_receptor_exchangeability(
        target_names, n_conformers_used, phase1_results)
    logger.info(f"\nReceptor exchangeability: {exchangeability['n_conformers']} "
                f"conformer(s), equal_sampling="
                f"{exchangeability['equal_sampling']}, equal_method="
                f"{exchangeability['equal_sampling_method']} "
                f"({exchangeability['ensemble_sampling_method']}), area ratio="
                f"{exchangeability['area_max_min_ratio']}")
    from .config import ENSEMBLE_REQUIRE_EQUAL_N
    if ENSEMBLE_REQUIRE_EQUAL_N and not (
            exchangeability["equal_sampling"]
            and exchangeability["equal_sampling_method"]):
        raise RuntimeError(
            f"Receptor ensembles are not equivalent across targets.\n"
            f"  conformers per target : {exchangeability['n_conformers']}\n"
            f"  sampling method       : "
            f"{exchangeability['ensemble_sampling_method']}\n"
            f"Keeping a statistic over N conformers makes N part of the "
            f"measurement, so unequal N — or the same N reached by different "
            f"sampling procedures — biases the cross-target contrast by "
            f"construction (measured on the previous CD run: 22% of the CD63 "
            f"contrast, 53% of the CD9 contrast, sign flip for 4 of 27 "
            f"monomers including methacrylic and acrylic acid). "
            f"Give every target the same ensemble, or set "
            f"ENSEMBLE_REQUIRE_EQUAL_N=False to proceed with the bias "
            f"recorded in phase2_smd_results.json.")

    logger.info("Computing selectivity matrix...")
    selectivity = _compute_selectivity(be_matrix, target_names, exchangeability)

    # B3 (cont.): Enrichment factor from be_matrix vs decoy_results.
    #
    # This is the pipeline's only test of whether the docking scores carry any
    # signal at all: do the real functional monomers separate from small inert
    # molecules on this receptor? It has never run, because the decoy stage
    # never docked anything. It runs now, and if it still cannot run the
    # RESULT SAYS SO — the criterion is not quietly dropped.
    from . import config as _cfg
    ef_threshold = float(getattr(_cfg, "PHASE2_EF_THRESHOLD", 1.5))
    enrichment = {}
    if PHASE2_DECOY_BASELINE:
        from .utils_analysis import enrichment_factor
        for target in target_names:
            t_be = {k: v for k, v in (be_matrix.get(target) or {}).items()
                    if v is not None}
            d_be = (decoy_results.get(target) or {}).get("decoy_be_matrix", {})
            valid_decoy = {k: v for k, v in (d_be or {}).items()
                           if v is not None}
            if not t_be or not valid_decoy:
                enrichment[target] = {
                    "status": "NOT_EVALUATED",
                    "n_monomers": len(t_be), "n_decoys": len(valid_decoy),
                    "ef_threshold": ef_threshold,
                    "criterion_met": None,
                    "reason": ("no decoy was successfully docked"
                               if not valid_decoy
                               else "no monomer was successfully docked"),
                }
                logger.error(
                    f"  {target}: ENRICHMENT FACTOR NOT EVALUATED "
                    f"({enrichment[target]['reason']}). Nothing in this run "
                    f"demonstrates that {target}'s docking scores distinguish "
                    f"functional monomers from inert decoys. Do not cite the "
                    f"EF>{ef_threshold} criterion for this target.")
                continue
            ef = dict(enrichment_factor(t_be, valid_decoy))
            ef["status"] = "OK"
            ef["n_monomers"] = len(t_be)
            ef["n_decoys"] = len(valid_decoy)
            ef["ef_threshold"] = ef_threshold
            ef["criterion_met"] = (ef.get("enrichment_factor") is not None
                                   and ef["enrichment_factor"] > ef_threshold)
            enrichment[target] = ef
            log = logger.info if ef["criterion_met"] else logger.warning
            log(f"  {target}: Enrichment Factor = "
                f"{ef.get('enrichment_factor', '?')} over {len(valid_decoy)} "
                f"decoys (threshold {ef_threshold}, "
                f"criterion_met={ef['criterion_met']})")
        if enrichment and all(e.get("status") == "NOT_EVALUATED"
                              for e in enrichment.values()):
            logger.error(
                "ENRICHMENT FACTOR WAS NOT EVALUATED ON ANY TARGET. The "
                "EF>%.1f validity criterion is unmet — not failed, ABSENT. "
                "Phase 2's absolute binding energies are unvalidated.",
                ef_threshold)

    # 4. Filter candidates
    from .config import SMD_TOP_N_FOR_PHASE3
    filtered = _filter_monomers(
        be_matrix, selectivity, target_names,
        be_threshold=SMD_BE_THRESHOLD,
        top_n=SMD_TOP_N_FOR_PHASE3,
    )

    # 5. Generate outputs
    results = {
        "be_matrix": be_matrix,
        "selectivity": selectivity,
        "receptor_exchangeability": exchangeability,
        "n_conformers_per_target": n_conformers_used,
        "filtered": filtered,
        "contact_md_scores": contact_scores,
        "decoy_baseline": decoy_results if PHASE2_DECOY_BASELINE else None,
        "enrichment_factor": enrichment if PHASE2_DECOY_BASELINE else None,
        # Monomers with NO usable docking on a target. They are absent from
        # be_matrix rather than present as 0.0; this is where they are named
        # so their absence is auditable instead of merely invisible.
        "monomers_without_score": missing_scores,
        "dock_details": {
            t: {m: {k: v for k, v in r.items() if k != "clusters"}
                for m, r in tresults.items()}
            for t, tresults in all_dock_results.items()
        },
    }

    # Save JSON
    with open(output_dir / "phase2_smd_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save CSV
    _save_selectivity_csv(be_matrix, selectivity, filtered,
                           output_dir / "phase2_selectivity.csv",
                           dock_details=all_dock_results,
                           enrichment=enrichment)

    # Generate heatmap
    _plot_heatmap(be_matrix, output_dir / "phase2_heatmap.png")

    # Log summary
    for target in target_names:
        filt = filtered.get(target, [])
        logger.info(f"[{target}] Filtered monomers ({len(filt)}): {filt}")

    return results


def _write_phase2_smiles_sidecar(pdbqt: Path, name: str, smiles: str) -> None:
    """Write <NAME>.pdbqt.smiles.json beside a monomer PDBQT.

    BYTE-COMPATIBLE with phase3_mmsd._write_stamp / _library_stamp — Phase 3
    REJECTS an unstamped cached PDBQT by design (an unstamped file is
    indistinguishable from one built with a since-retired SMILES, which is
    exactly the 753-stale-file problem BLOCKER 05 found). Without this sidecar
    Phase 3 regenerates every monomer on first use instead of reusing Phase 2's,
    and Phase 2's own docking cache stays unprotected against the same failure.
    """
    import hashlib
    try:
        Path(f"{pdbqt}.smiles.json").write_text(json.dumps({
            "name": name,
            "smiles": smiles,
            "smiles_sha256": hashlib.sha256(
                smiles.encode("utf-8")).hexdigest()[:16],
        }, indent=1), encoding="utf-8")
    except Exception as e:                       # provenance must not kill a run
        logger.error(f"{name}: could not write SMILES sidecar beside {pdbqt}: {e}")


def _prepare_all_monomers(monomer_lib: dict, output_dir: Path) -> dict:
    """Prepare PDBQT files for all monomers. Returns {name: pdbqt_path}.

    # BEHAVIOUR CHANGE (2026-08 audit), two parts:
    #
    # 1. A CACHE HIT IS NO LONGER TAKEN ON THE FILENAME ALONE.  The cache is
    #    keyed on <NAME>.pdbqt, so a corrected SMILES — nine silanes, APBA,
    #    AAPBA and TRIM were all corrected this cycle — leaves the RETIRED
    #    molecule sitting on disk under the current name, and every docking
    #    against it silently measures the wrong ligand.  Each hit is now
    #    verified against the library SMILES and regenerated on mismatch.
    #
    # 2. A PREPARATION FAILURE NOW FAILS THE PHASE.  smiles_to_pdbqt() used to
    #    cache a zero-byte file on failure; it now raises.  The old handler
    #    here turned that into a warning and carried on with a SMALLER LIBRARY,
    #    which changes the ranking denominator and the cross-target comparison
    #    without appearing anywhere in the output.  Missing monomers are
    #    collected and raised as one error naming all of them.
    """
    from .utils_structure import smiles_to_pdbqt, verify_smiles_stamp

    output_dir = Path(output_dir)
    pdbqts = {}
    failed = {}
    regenerated = []
    for name, info in monomer_lib.items():
        smiles = info["smiles"]
        pdbqt = output_dir / f"{name}.pdbqt"

        if pdbqt.exists():
            try:
                verify_smiles_stamp(pdbqt, name, smiles, require_stamp=True)
                pdbqts[name] = pdbqt
                continue
            except Exception as e:
                logger.warning(f"{name}: {e} — regenerating")
                pdbqt.unlink(missing_ok=True)
                regenerated.append(name)

        try:
            pdbqt_path = smiles_to_pdbqt(smiles, name, output_dir)
            _write_phase2_smiles_sidecar(Path(pdbqt_path), name, smiles)
            pdbqts[name] = pdbqt_path
        except Exception as e:
            logger.error(f"FAILED to prepare {name} ({smiles}): {e}")
            failed[name] = str(e)

    if regenerated:
        logger.warning(f"Regenerated {len(regenerated)} monomer PDBQT(s) whose "
                       f"cached copy did not match the current library SMILES: "
                       f"{sorted(regenerated)}")
    logger.info(f"Prepared {len(pdbqts)}/{len(monomer_lib)} monomer PDBQTs")

    if failed:
        raise RuntimeError(
            f"Phase 2 could not prepare {len(failed)} of {len(monomer_lib)} "
            f"monomers: {failed}. Refusing to continue with a smaller library — "
            f"dropping a monomer silently changes the ranking denominator and "
            f"every cross-target comparison built on it. Fix the SMILES (or the "
            f"PDBQT writers) and re-run.")
    return pdbqts


def _analyze_hbond_types_for_target(target: str, target_results: dict,
                                     p1_result: dict,
                                     output_dir: Path) -> dict:
    """Sullivan 2019: analyze backbone vs sidechain H-bond ratios."""
    from .utils_analysis import analyze_hbond_types
    from .config import MAX_BACKBONE_HBOND_RATIO

    receptor_pdb = Path(p1_result["epitope_pdb"])
    for monomer_name, result in target_results.items():
        dlg = result.get("dlg_path")
        if dlg and Path(dlg).exists():
            try:
                # MAX_BACKBONE_HBOND_RATIO is now PASSED, not merely imported
                # (REVIEW FINDING 19): analyze_hbond_types used to hardcode 0.3
                # internally, so this module's import of the config knob had no
                # effect whatsoever on the threshold it appeared to control.
                hb = analyze_hbond_types(Path(dlg), receptor_pdb,
                                         max_backbone_ratio=MAX_BACKBONE_HBOND_RATIO)
                result["hbond_analysis"] = hb
                if hb.get("structural_disruption_risk"):
                    logger.warning(
                        f"  {target}-{monomer_name}: HIGH backbone H-bond "
                        f"ratio ({hb['backbone_ratio']:.0%} > "
                        f"{MAX_BACKBONE_HBOND_RATIO:.0%}) — "
                        "potential 2° structure disruption (Sullivan 2019)"
                    )
            except Exception as e:
                # Raised from DEBUG (REVIEW FINDING 15). The pipeline runs at
                # INFO, so a DEBUG line is never emitted: this analysis could
                # fail for every monomer of every target and leave no trace.
                logger.error(
                    f"H-bond type analysis FAILED for {target}-{monomer_name}: "
                    f"{type(e).__name__}: {e} — no backbone/sidechain "
                    f"breakdown recorded for this monomer")
    return target_results


def _run_contact_md(phase1_results: dict, target_names: list,
                     monomer_pdbqts: dict, output_dir: Path,
                     time_ns: float = 10.0) -> dict:
    """
    Sehit 2024: run short MD per (target, monomer) pair and compute
    contact frequency. Monomers with more epitope contacts rank higher.

    Each simulation: epitope + 1 monomer in TIP3P + 0.15M NaCl, 10ns.
    Contact frequency = fraction of frames where monomer is within
    3.5A of any epitope residue.
    """
    from .utils_gromacs import (
        setup_protein_topology, setup_simulation_box,
        run_energy_minimization, run_nvt_equilibration,
        run_npt_equilibration, run_production_md,
        parameterize_monomer, _gmx,
    )
    from .utils_structure import smiles_to_mol2
    from .utils_analysis import compute_contact_frequency
    from .config import ALL_MONOMERS, MD_TEMPERATURE_K, MD_GPU_ID

    contact_scores = {}
    for target in target_names:
        p1 = phase1_results.get(target, {})
        if "error" in p1:
            continue
        epitope_pdb = Path(p1["epitope_pdb"])
        target_scores = {}

        logger.info(f"  [{target}] Contact MD: {len(monomer_pdbqts)} monomers "
                    f"x {time_ns}ns each")

        for monomer_name in monomer_pdbqts:
            m_info = ALL_MONOMERS.get(monomer_name)
            if m_info is None:
                continue

            md_dir = output_dir / f"contact_md_{target}" / monomer_name
            md_dir.mkdir(parents=True, exist_ok=True)

            # Check if already done (resume support)
            result_file = md_dir / "contact_result.json"
            if result_file.exists():
                import json as _json
                with open(result_file) as f:
                    target_scores[monomer_name] = _json.load(f)
                logger.info(f"    {monomer_name}: loaded existing result")
                continue

            try:
                # 1. Parameterize monomer
                param_dir = md_dir / "params"
                mol2 = smiles_to_mol2(m_info["smiles"], monomer_name, param_dir)
                param = parameterize_monomer(mol2, monomer_name, param_dir)

                if not param.get("itp"):
                    logger.warning(f"    {monomer_name}: parameterization failed")
                    target_scores[monomer_name] = {"error": "param failed"}
                    continue

                # 2. Setup GROMACS system (epitope + monomer)
                setup_protein_topology(epitope_pdb, md_dir)
                _include_monomer_in_topology(
                    md_dir, param["itp"], param["gro"], monomer_name)
                setup_simulation_box(md_dir / "complex.gro", md_dir)

                # 3. Quick MD (EM → NVT → short production)
                run_energy_minimization(md_dir)
                run_nvt_equilibration(md_dir, time_ps=50.0,
                                       temperature=MD_TEMPERATURE_K)
                run_npt_equilibration(md_dir, time_ps=50.0,
                                       temperature=MD_TEMPERATURE_K)
                run_production_md(md_dir, time_ns=time_ns,
                                   temperature=MD_TEMPERATURE_K,
                                   gpu_id=MD_GPU_ID)

                # 4. Compute contact frequency
                xtc = md_dir / "md.xtc"
                gro = md_dir / "npt.gro"
                if xtc.exists() and gro.exists():
                    contacts = compute_contact_frequency(xtc, gro)
                    target_scores[monomer_name] = contacts
                    logger.info(f"    {monomer_name}: contact_score="
                                f"{contacts.get('total_contact_score', 'N/A')}")
                else:
                    target_scores[monomer_name] = {"error": "MD output missing"}

                # Save individual result for resume
                import json as _json
                with open(result_file, "w") as f:
                    _json.dump(target_scores[monomer_name], f, indent=2)

            except Exception as e:
                logger.warning(f"    {monomer_name} contact MD failed: {e}")
                target_scores[monomer_name] = {"error": str(e)}

        contact_scores[target] = target_scores

    return contact_scores


def _include_monomer_in_topology(work_dir: Path, itp_path: str,
                                  gro_path: str, name: str):
    """
    Merge monomer into GROMACS system:
    1. Add #include monomer.itp to topol.top
    2. Merge monomer.gro coordinates into protein.gro
    3. Add molecule entry to [ molecules ] section
    """
    import shutil
    work_dir = Path(work_dir)

    # Copy ITP to work directory
    itp_src = Path(itp_path)
    itp_dst = work_dir / f"{name}.itp"
    shutil.copy2(str(itp_src), str(itp_dst))

    # Edit topol.top: add #include before [ molecules ]
    top_path = work_dir / "topol.top"
    if top_path.exists():
        content = top_path.read_text()
        include_line = f'#include "{name}.itp"\n'
        if include_line not in content:
            # Insert before [ molecules ] section
            if "[ molecules ]" in content:
                content = content.replace(
                    "[ molecules ]",
                    f'{include_line}\n[ molecules ]'
                )
            else:
                content += f"\n{include_line}\n"
            # Add molecule to [ molecules ] section
            content += f"{name}     1\n"
            top_path.write_text(content)

    # Merge coordinates: append monomer GRO to protein GRO
    prot_gro = work_dir / "protein.gro"
    mon_gro = Path(gro_path)
    complex_gro = work_dir / "complex.gro"

    if prot_gro.exists() and mon_gro.exists():
        prot_lines = prot_gro.read_text().strip().split("\n")
        mon_lines = mon_gro.read_text().strip().split("\n")

        # GRO format: line 1=title, line 2=natoms, lines 3..N=coords, last=box
        prot_natoms = int(prot_lines[1].strip())
        mon_natoms = int(mon_lines[1].strip())
        total = prot_natoms + mon_natoms

        out_lines = [prot_lines[0]]  # title
        out_lines.append(f" {total}")
        out_lines.extend(prot_lines[2:2+prot_natoms])  # protein coords
        out_lines.extend(mon_lines[2:2+mon_natoms])     # monomer coords
        out_lines.append(prot_lines[-1])                 # box vector

        complex_gro.write_text("\n".join(out_lines) + "\n")
    else:
        # If can't merge, just use protein
        if prot_gro.exists():
            shutil.copy2(str(prot_gro), str(complex_gro))


def _validate_dock_result(target: str, name: str, result: dict,
                          started_at: float) -> dict:
    """Turn a CLAIMED docking success into a verified one, or into missing data.

    run_autodock() infers success for the AutoDock-GPU path from nothing but
    "a file bigger than 100 bytes exists at the deterministic .dlg path"; the
    subprocess return code is not consulted and the path is never cleaned. So
    a docking that crashes on run K hands back run K-1's .dlg and it is scored
    as a fresh measurement of a different receptor conformer. (The return-code
    check itself lives in utils_autodock.py, which this agent does not own —
    a cross-file request is filed. This is the receiving end of that check,
    and it stands on its own: it verifies the ARTEFACT rather than trusting
    the engine.)

    Three things must hold for a score to count:
      1. dock_single did not report success=False;
      2. the .dlg exists and was WRITTEN DURING THIS CALL (mtime >= start) —
         this is what catches a stale DLG surviving a failed run;
      3. the .dlg actually contains a binding-energy record.
    Anything else becomes missing data (mean_cluster_energy=None), never a
    number.
    """
    from . import config as _cfg
    require_fresh = getattr(_cfg, "PHASE2_REQUIRE_FRESH_DLG", True)

    result = dict(result or {})
    result.setdefault("dock_verified", True)
    problems = []

    if result.get("success") is False:
        problems.append(f"engine reported failure: "
                        f"{str(result.get('error'))[:200]}")

    dlg = result.get("dlg_path")
    dlg_p = Path(dlg) if dlg else None
    if dlg_p is None or not dlg_p.is_file():
        problems.append(f"no DLG produced (dlg_path={dlg!r})")
    else:
        age = dlg_p.stat().st_mtime
        result["dlg_mtime"] = age
        # 2 s of slack for filesystem timestamp granularity.
        if age < started_at - 2.0:
            msg = (f"STALE DLG: {dlg_p.name} was last written "
                   f"{started_at - age:.0f} s BEFORE this docking started, so "
                   f"it is a previous run's output, not this run's result")
            if require_fresh:
                problems.append(msg)
            else:
                logger.warning(f"  {target}-{name}: {msg} "
                               f"(PHASE2_REQUIRE_FRESH_DLG=False: accepted)")
        try:
            text = dlg_p.read_text(errors="replace")
        except Exception as e:                       # pragma: no cover
            text = ""
            problems.append(f"DLG unreadable: {e}")
        if text and "Estimated Free Energy of Binding" not in text:
            problems.append("DLG contains no 'Estimated Free Energy of "
                            "Binding' record — the run produced a file but "
                            "no docking result")

    if result.get("mean_cluster_energy") is None and not problems:
        problems.append("no rank-1 cluster energy could be parsed from the DLG")

    if problems:
        # MISSING DATA, NOT A MEASUREMENT. Every numeric field is cleared so
        # that nothing downstream can read a score out of a failed docking.
        logger.error(f"  {target}-{name}: DOCKING REJECTED — "
                     + "; ".join(problems))
        result["mean_cluster_energy"] = None
        result["binding_energy"] = None
        result["dock_verified"] = False
        result["dock_problems"] = problems
        result["success"] = False
    return result


def _run_smd_for_target(target: str, receptor_pdbqt: Path,
                         monomer_pdbqts: dict,
                         center: tuple, npts: tuple,
                         output_dir: Path,
                         ga_runs: int = 50,
                         n_workers: int = 4,
                         binding_sites: list = None,
                         pose_clustering: bool = None,
                         receptor_pdb: Path = None,
                         target_cfg: dict = None) -> dict:
    """Run SMD docking for one target against all monomers.

    receptor_pdb / target_cfg
        When BOTH are supplied the membrane-accessibility filter runs (see
        apply_membrane_accessibility_filter): poses sitting predominantly on the
        lipid-facing surface are not accepted as the monomer's score. The DECOY
        baseline passes them too — the enrichment factor compares the two arms,
        so filtering only one would bias it.
    """
    from .utils_autodock import dock_single

    results = {}
    dock_dir = output_dir / f"smd_{target}"
    dock_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()

    # Parallel docking
    # expected_smiles: dock_single re-checks the ligand's SMILES stamp against
    # the LIBRARY entry before docking, and refuses a mismatch. Without the
    # argument the check is skipped entirely. This catches a stale PDBQT that an
    # earlier run copied into the per-monomer work directory — the work dirs are
    # keyed on name, not on content, so they survive a SMILES correction.
    from .config import ALL_MONOMERS as _ALL_MONOMERS
    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for name, pdbqt in monomer_pdbqts.items():
            work = dock_dir / f"{target}_{name}"
            future = executor.submit(
                dock_single,
                receptor_pdbqt, pdbqt,
                center, npts, work,
                ga_runs=ga_runs,
                expected_smiles=(_ALL_MONOMERS.get(name) or {}).get("smiles"),
            )
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                # MISSING DATA, NOT ZERO. This used to write
                # binding_energy=0.0 / mean_cluster_energy=0.0, which then
                # entered be_matrix as a MEASUREMENT. 0.0 kcal/mol is not a
                # neutral placeholder: it is the weakest value the scale
                # produces, so a crashed docking rendered as "binds nothing",
                # sorted to the bottom of _filter_monomers and pushed every
                # real monomer below it up one rank. A failure must be
                # absent from the matrix, not present as a strong claim.
                logger.error(f"  {target}-{name} docking raised "
                             f"{type(e).__name__}: {e} — recorded as MISSING "
                             f"(mean_cluster_energy=None), not as 0.0")
                results[name] = {"mean_cluster_energy": None,
                                 "binding_energy": None,
                                 "error": f"{type(e).__name__}: {e}",
                                 "success": False, "dock_verified": False,
                                 "dock_problems": ["exception in dock_single"]}
                continue
            result = _validate_dock_result(target, name, result, started_at)
            results[name] = result
            be = result.get("mean_cluster_energy")
            logger.info(f"  {target}-{name}: BE = "
                        + (f"{be:.2f} kcal/mol" if be is not None
                           else "MISSING (docking not usable)"))

    n_missing = sum(1 for r in results.values()
                    if r.get("mean_cluster_energy") is None)
    if n_missing:
        logger.error(f"  [{target}] {n_missing}/{len(results)} dockings "
                     f"produced NO usable score against this receptor. They "
                     f"are excluded from the matrix, not zero-filled.")

    # Membrane accessibility: reject poses on the lipid-facing surface.
    # Runs BEFORE pose clustering is published so the clusters reported reflect
    # the same poses the score now comes from.
    from .config import PHASE2_ENFORCE_MEMBRANE_ACCESSIBILITY
    if receptor_pdb is not None and target_cfg is not None:
        if PHASE2_ENFORCE_MEMBRANE_ACCESSIBILITY:
            results = apply_membrane_accessibility_filter(
                target, target_cfg, receptor_pdb, results, dock_dir)
        else:
            logger.warning(
                f"  [{target}] PHASE2_ENFORCE_MEMBRANE_ACCESSIBILITY is False — "
                f"binding energies may come from poses on the membrane-occluded "
                f"surface, which is a TARGET-DEPENDENT confound, not a uniform one.")

    # A3: Pose clustering for each monomer (if dlg files available)
    from .config import PHASE2_POSE_CLUSTERING
    if pose_clustering is None:
        pose_clustering = PHASE2_POSE_CLUSTERING
    if pose_clustering:
        n_clustered = 0
        for name in results:
            work = dock_dir / f"{target}_{name}"
            dlg_files = list(Path(work).rglob("*.dlg"))
            if not dlg_files:
                results[name]["pose_clustering"] = "no_dlg"
                continue
            try:
                clusters = cluster_poses_from_dlg(dlg_files[0])
            except Exception as e:
                # Was logger.debug — so the TypeError from the DLG-prefix bug
                # was invisible while A3 reported as enabled. A feature that
                # silently produces nothing is worse than one that is off.
                logger.error(f"  {target}-{name}: A3 pose clustering FAILED "
                             f"({type(e).__name__}: {e}) on "
                             f"{dlg_files[0].name}")
                results[name]["pose_clustering"] = f"error: {type(e).__name__}"
                continue
            if clusters:
                n_clustered += 1
                results[name]["pose_clusters"] = [
                    {
                        "cluster_size": c["cluster_size"],
                        "binding_energy": c["binding_energy"],
                        "energies_in_cluster": c.get(
                            "binding_energies_in_cluster", []),
                    }
                    for c in clusters
                ]
                results[name]["pose_clustering"] = "ok"
                logger.info(f"  {target}-{name}: A3 found "
                            f"{len(clusters)} pose clusters")
            else:
                results[name]["pose_clustering"] = "no_poses_parsed"
        if results and n_clustered == 0:
            logger.error(
                f"  [{target}] A3 pose clustering is ENABLED but produced "
                f"ZERO clusters for all {len(results)} monomers. Treat "
                f"pose-cluster output as unavailable for this receptor "
                f"rather than as evidence of a single binding mode.")

    return results


def _merge_across_conformers(target: str, per_conformer: dict) -> dict:
    """
    Collapse per-conformer docking results to one score per monomer.

    WHY NOT min(). Taking the best score over a target's receptor conformers
    is a best-of-N statistic, and the expectation of a minimum falls as N
    grows even when nothing about the binding changes. That is harmless when
    every target has the same N and fatal when they do not: in the committed
    CD run CD63 and CD9 each had 6 conformers and CD81 had 1, so the
    own-vs-cross contrast partly measured how many receptors each target was
    docked against. Re-parsed from the .dlg files, the ensemble gain was
    CD63 -0.072, CD9 -0.139 and CD81 0.000 kcal/mol — 22% of the CD63
    contrast and 53% of the CD9 contrast.

    BOLTZMANN (default) averages the conformers with a 1/N prior:
        E_eff = -kT ln( (1/N) Σ exp(-E_i / kT) )
    The 1/N keeps the estimator N-invariant in expectation, so adding
    conformers sharpens it instead of shifting it, while still letting a
    genuinely better-binding conformer dominate the average as it should.
    MEAN is the plain arithmetic average; MIN reproduces the legacy
    behaviour and is kept only so the old numbers can be regenerated.
    """
    from .config import ENSEMBLE_MERGE, ENSEMBLE_BOLTZMANN_T_K
    kT = 0.0019872041 * ENSEMBLE_BOLTZMANN_T_K       # kcal/mol

    merged = {}
    for m, entries in per_conformer.items():
        energies = np.array([r["mean_cluster_energy"] for _, r in entries],
                            dtype=float)
        best_i = int(np.argmin(energies))
        # The pose-level fields (clusters, contacts, H-bonds) must come from a
        # real docking run, so the best conformer carries them; only the score
        # is replaced by the ensemble estimator.
        rec = dict(entries[best_i][1])

        if ENSEMBLE_MERGE == "min":
            score = float(energies.min())
        elif ENSEMBLE_MERGE == "mean":
            score = float(energies.mean())
        elif ENSEMBLE_MERGE == "boltzmann":
            e0 = float(energies.min())               # shift for stability
            score = e0 - kT * float(np.log(np.mean(np.exp(-(energies - e0) / kT))))
        else:
            raise ValueError(
                f"ENSEMBLE_MERGE={ENSEMBLE_MERGE!r} not in "
                f"('boltzmann', 'mean', 'min')")

        rec["mean_cluster_energy"] = round(score, 4)
        rec["ensemble_merge"] = ENSEMBLE_MERGE
        rec["ensemble_n_conformers"] = len(entries)
        rec["ensemble_conformer_energies"] = {
            label: round(float(r["mean_cluster_energy"]), 4)
            for label, r in entries}
        rec["ensemble_best_conformer"] = entries[best_i][0]
        rec["ensemble_min_energy"] = round(float(energies.min()), 4)
        rec["ensemble_spread"] = round(float(energies.max() - energies.min()), 4)
        # ALL THREE ESTIMATORS ARE ALWAYS REPORTED, whichever one
        # ENSEMBLE_MERGE selects as the score. best-of-N (min) is biased even
        # at equal N — E[min] drifts downward with N for identical binding —
        # so the N-invariant arithmetic mean travels beside it and any reader
        # can see how much of a contrast is estimator choice. ensemble_sd is
        # the conformer-to-conformer scatter: a monomer whose score moves 2
        # kcal/mol across receptor conformers has not been measured to 0.1.
        rec["ensemble_mean_energy"] = round(float(energies.mean()), 4)
        rec["ensemble_sd_energy"] = round(float(energies.std(ddof=0)), 4)
        e0 = float(energies.min())
        rec["ensemble_boltzmann_energy"] = round(
            e0 - kT * float(np.log(np.mean(np.exp(-(energies - e0) / kT)))), 4)
        merged[m] = rec

    if merged:
        ns = sorted({r["ensemble_n_conformers"] for r in merged.values()})
        logger.info(f"  [{target}] merged {len(merged)} monomers over "
                    f"{ns} conformer(s) using ENSEMBLE_MERGE="
                    f"{ENSEMBLE_MERGE}")
        if len(ns) > 1:
            # Even within one target, a monomer merged over 6 conformers is
            # not comparable with one merged over 2 — same best-of-N problem,
            # one rank order down.
            per_n = {}
            for mm, r in merged.items():
                per_n.setdefault(r["ensemble_n_conformers"], []).append(mm)
            logger.error(
                f"  [{target}] UNEQUAL CONFORMER COUNT WITHIN THE TARGET: "
                f"{ {k: len(v) for k, v in sorted(per_n.items())} } "
                f"(monomers per N). Some dockings failed on some conformers, "
                f"so this target's monomers were not all scored over the same "
                f"receptor set; ranks within the target are affected. "
                f"Smallest group: N={min(per_n)} -> {sorted(per_n[min(per_n)])}")
    return merged


def _check_receptor_exchangeability(target_names: list, n_conformers: dict,
                                    phase1_results: dict) -> dict:
    """
    Decide whether absolute docking scores may be compared across targets.

    Three receptors are exchangeable only if they were sampled the same way
    and offer a comparable amount of surface. This returns the evidence and a
    verdict; it never silently downgrades a number, it labels it.
    """
    ns = {t: n_conformers.get(t) for t in target_names}
    distinct_n = {v for v in ns.values() if v is not None}
    equal_sampling = len(distinct_n) <= 1

    areas, residues, templates, methods = {}, {}, {}, {}
    for t in target_names:
        p1t = phase1_results.get(t) or {}
        d = p1t.get("receptor_descriptor") or {}
        areas[t] = d.get("sasa_scored_surface_A2") or d.get("sasa_total_A2")
        residues[t] = d.get("n_scored_surface") or d.get("n_residues_in_receptor")
        templates[t] = d.get("template")
        # How the ensemble was CHOSEN, not just how big it is. A k-medoids
        # ensemble spans the trajectory's conformational clusters; the
        # uniform-time fallback spans wall-clock. Same N, different quantity.
        methods[t] = ((p1t.get("ensemble_sampling") or {}).get("method"))
    distinct_methods = {m for m in methods.values() if m is not None}
    equal_method = len(distinct_methods) <= 1

    known = [a for a in areas.values() if a]
    area_ratio = (round(max(known) / min(known), 3)
                  if len(known) == len(target_names) and min(known) > 0 else None)
    comparable_area = area_ratio is not None and area_ratio <= 1.15

    return {
        "n_conformers": ns,
        "equal_sampling": equal_sampling,
        "ensemble_sampling_method": methods,
        "equal_sampling_method": equal_method,
        "scored_surface_A2": areas,
        "scored_surface_residues": residues,
        "templates": templates,
        "area_max_min_ratio": area_ratio,
        "comparable_area": comparable_area,
        "exchangeable": bool(equal_sampling and equal_method
                             and comparable_area),
        "note": (
            "Absolute cross-target ΔΔG is only meaningful when receptors are "
            "exchangeable. equal_sampling=False means one target was docked "
            "against more receptor conformers than another (best-of-N bias). "
            "equal_sampling_method=False means the ensembles were chosen by "
            "different procedures (k-medoids vs uniform-time fallback), so "
            "they span different things at the same N. comparable_area=False "
            "means the receptors differ in scored surface by >15%, so a "
            "bigger target wins on opportunity alone."),
    }


def _compute_selectivity(be_matrix: dict, target_names: list,
                         exchangeability: dict = None) -> dict:
    """
    Selectivity of each monomer for each target.

    THE RAW ΔΔG = BE(own) - mean(BE(cross)) IS NO LONGER THE PRIMARY METRIC.
    It is a difference of absolute docking scores taken on three receptors
    that are not exchangeable: different size (the CD63 receptor is 26%
    larger by SASA than CD9's), different provenance (one wholly predicted,
    two crystallographic at 2.7-3.0 Å), and, before the ensemble fix,
    different numbers of receptor conformers. Every one of those differences
    shifts a target's whole energy scale, and subtracting across scales
    passes the shift straight into the "selectivity" number undiluted.

    What survives a per-target monotone rescaling is RANK ORDER, so the
    default primary metric is rank-based:

      rank_within_target   1 = strongest binder of this target's monomers
      rank_delta           mean(rank on cross-targets) - rank on own target;
                           POSITIVE = ranks better on its own target than on
                           the others, which is what selectivity means
      z_ddg                own-vs-cross difference after standardising each
                           target's scores to zero mean / unit SD — keeps
                           the magnitude information, removes the scale
      own_cross_ratio      BE(own) / mean(BE(cross)), the ratio form; >1 is
                           selective. Per-template quality terms partially
                           cancel in a ratio, which is why the template
                           documentation prescribes it
      ddg_raw              retained for continuity, FLAGGED, never primary

    The scalar returned in selectivity[target][monomer] is whichever metric
    SELECTIVITY_METRIC names; the full breakdown is in the "_detail" key.
    """
    from .config import SELECTIVITY_METRIC

    detail = {}
    ranks, zs = {}, {}

    # Per-target standardisation and ranking — computed WITHIN a target, so
    # no cross-target quantity enters before this point.
    for t in target_names:
        row = {m: v for m, v in (be_matrix.get(t) or {}).items()
               if v is not None}
        if not row:
            ranks[t], zs[t] = {}, {}
            continue
        ordered = sorted(row.items(), key=lambda kv: kv[1])   # most negative 1st
        ranks[t] = {m: i + 1 for i, (m, _) in enumerate(ordered)}
        vals = np.array(list(row.values()), dtype=float)
        mu, sd = float(vals.mean()), float(vals.std())
        zs[t] = {m: ((v - mu) / sd if sd > 1e-9 else 0.0)
                 for m, v in row.items()}

    selectivity = {}
    for target in target_names:
        non_targets = [t for t in target_names if t != target]
        selectivity[target] = {}
        detail[target] = {}

        for monomer in be_matrix.get(target, {}):
            be_target = be_matrix[target].get(monomer)
            be_others = [be_matrix.get(t, {}).get(monomer)
                         for t in non_targets]
            be_others = [v for v in be_others if v is not None]
            if be_target is None or not be_others:
                continue
            mean_others = float(np.mean(be_others))

            r_own = ranks[target].get(monomer)
            r_others = [ranks[t][monomer] for t in non_targets
                        if monomer in ranks.get(t, {})]
            rank_delta = (float(np.mean(r_others)) - r_own
                          if r_own is not None and r_others else None)

            z_own = zs[target].get(monomer)
            z_others = [zs[t][monomer] for t in non_targets
                        if monomer in zs.get(t, {})]
            z_ddg = (z_own - float(np.mean(z_others))
                     if z_own is not None and z_others else None)

            ratio = (be_target / mean_others
                     if abs(mean_others) > 1e-6 else None)

            n_t = len(ranks[target]) or 1
            d = {
                "ddg_raw": round(float(be_target - mean_others), 3),
                "rank_within_target": r_own,
                "n_monomers_ranked": n_t,
                "percentile_within_target": (round(100.0 * (n_t - r_own) / n_t, 1)
                                             if r_own else None),
                "rank_delta": (round(rank_delta, 2)
                               if rank_delta is not None else None),
                "z_within_target": (round(z_own, 3) if z_own is not None else None),
                "z_ddg": round(z_ddg, 3) if z_ddg is not None else None,
                "own_cross_ratio": round(float(ratio), 3) if ratio else None,
            }
            detail[target][monomer] = d

            primary = d.get(SELECTIVITY_METRIC, d["ddg_raw"])
            selectivity[target][monomer] = (
                round(float(primary), 3) if primary is not None else None)

    selectivity["_detail"] = detail
    selectivity["_metric"] = SELECTIVITY_METRIC
    selectivity["_exchangeability"] = exchangeability or {}
    if exchangeability and not exchangeability.get("exchangeable", False):
        logger.warning(
            "  Receptors are NOT exchangeable "
            f"(equal_sampling={exchangeability.get('equal_sampling')}, "
            f"comparable_area={exchangeability.get('comparable_area')}, "
            f"area ratio={exchangeability.get('area_max_min_ratio')}). "
            "Raw ΔΔG between these targets is not interpretable; use the "
            "rank-based columns.")
    return selectivity


def _filter_monomers(be_matrix: dict, selectivity: dict,
                      target_names: list,
                      be_threshold: float = -2.0,
                      top_n: int = 12) -> dict:
    """
    Select top N monomers by BE for Phase 3.

    Selectivity (ΔΔG) is computed for reporting but NOT used for filtering.
    Selectivity is assessed in Phase 3 (MMSD synergy) and Phase 4
    (MD cross-reactivity) where it can be evaluated more accurately.
    """
    filtered = {}
    for target in target_names:
        candidates = []
        for monomer in be_matrix.get(target, {}):
            be = be_matrix[target].get(monomer, None)
            if be is None:
                continue
            if be <= be_threshold:
                candidates.append((monomer, be))

        candidates.sort(key=lambda x: x[1])
        filtered[target] = [c[0] for c in candidates[:top_n]]

        logger.info(f"  [{target}] {len(filtered[target])} monomers selected "
                    f"(BE ≤ {be_threshold}, top {top_n})")
    return filtered


def _save_selectivity_csv(be_matrix: dict, selectivity: dict,
                           filtered: dict, output_path: Path,
                           dock_details: dict = None,
                           enrichment: dict = None):
    """
    Save combined results as CSV.

    `selectivity_ddg` is KEPT, with its original meaning (raw own-minus-cross
    difference of absolute docking scores) so older readers are not silently
    handed a different quantity under a familiar name. It is a diagnostic:
    it is only interpretable when receptors_exchangeable is True. The
    rank-based columns beside it are the ones to rank monomers by.

    `be_ensemble_mean` / `be_ensemble_min` / `be_ensemble_sd` are the same
    docking data reduced by the three estimators. best-of-N (min) is biased
    downward as N grows even for identical binding, so the N-invariant mean is
    printed beside it: if a cross-target difference lives in the min column
    and not in the mean column, it is an artefact of ensemble size.
    """
    detail = selectivity.get("_detail", {})
    metric = selectivity.get("_metric", "ddg_raw")
    exch = selectivity.get("_exchangeability", {}) or {}
    dock_details = dock_details or {}
    enrichment = enrichment or {}
    rows = []
    for target in be_matrix:
        ef = enrichment.get(target) or {}
        for monomer in be_matrix[target]:
            d = detail.get(target, {}).get(monomer, {})
            dd = (dock_details.get(target) or {}).get(monomer, {}) or {}
            rows.append({
                "target": target,
                "monomer": monomer,
                "binding_energy": be_matrix[target][monomer],
                "be_estimator": dd.get("ensemble_merge"),
                "be_ensemble_mean": dd.get("ensemble_mean_energy"),
                "be_ensemble_min": dd.get("ensemble_min_energy"),
                "be_ensemble_sd": dd.get("ensemble_sd_energy"),
                "n_conformers_scored": dd.get("ensemble_n_conformers"),
                # legacy column, unchanged definition
                "selectivity_ddg": d.get("ddg_raw", 0),
                # normalised, cross-target-safe metrics
                "selectivity_primary": selectivity.get(target, {}).get(monomer),
                "selectivity_metric": metric,
                "rank_within_target": d.get("rank_within_target"),
                "percentile_within_target": d.get("percentile_within_target"),
                "rank_delta": d.get("rank_delta"),
                "z_within_target": d.get("z_within_target"),
                "z_ddg": d.get("z_ddg"),
                "own_cross_ratio": d.get("own_cross_ratio"),
                "n_receptor_conformers": (exch.get("n_conformers") or {}).get(target),
                "ensemble_sampling_method": (
                    exch.get("ensemble_sampling_method") or {}).get(target),
                "receptors_exchangeable": exch.get("exchangeable"),
                # Did anything in this run show that these scores separate
                # real monomers from inert decoys? Blank/NOT_EVALUATED means
                # no — the criterion did not fail, it never ran.
                "enrichment_factor": ef.get("enrichment_factor"),
                "enrichment_status": ef.get("status", "NOT_EVALUATED"),
                "enrichment_criterion_met": ef.get("criterion_met"),
                "passed_filter": monomer in filtered.get(target, []),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Selectivity CSV → {output_path}")


def _plot_heatmap(be_matrix: dict, output_path: Path):
    """Generate binding energy heatmap (targets × monomers)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        targets = list(be_matrix.keys())
        monomers = sorted(set(
            m for t in targets for m in be_matrix[t]
        ))

        # MISSING IS NaN, NOT 0.0. Zero-filling a monomer that failed to dock
        # painted it as the weakest binder on the map — an absence rendered as
        # a measurement. NaN leaves the cell blank instead.
        data = np.full((len(targets), len(monomers)), np.nan)
        for i, t in enumerate(targets):
            for j, m in enumerate(monomers):
                v = be_matrix[t].get(m)
                if v is not None:
                    data[i, j] = float(v)
        n_missing = int(np.isnan(data).sum())

        try:                       # matplotlib >= 3.6
            cmap = matplotlib.colormaps["RdYlBu"].copy()
        except Exception:          # pragma: no cover — older matplotlib
            cmap = matplotlib.cm.get_cmap("RdYlBu").copy()
        cmap.set_bad(color="0.85")
        fig, ax = plt.subplots(figsize=(max(12, len(monomers) * 0.6), 4))
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(monomers)))
        ax.set_xticklabels(monomers, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets)
        ax.set_xlabel("Monomer")
        ax.set_ylabel("Target")
        ax.set_title("SMD Binding Energy (kcal/mol)"
                     + (f"  —  grey = no usable docking ({n_missing} cells)"
                        if n_missing else ""))
        plt.colorbar(im, ax=ax, label="BE (kcal/mol)")

        # Annotate values
        for i in range(len(targets)):
            for j in range(len(monomers)):
                v = data[i, j]
                ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.1f}",
                        ha="center", va="center", fontsize=6)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Heatmap → {output_path}"
                    + (f" ({n_missing} missing cell(s) drawn as n/a)"
                       if n_missing else ""))
    except ImportError:
        logger.warning("matplotlib not available, skipping heatmap")


# ════════════════════════════════════════════════════════════════
# A3: Pose Clustering — extract multiple binding modes from AutoDock
# ════════════════════════════════════════════════════════════════

# AutoDock4 / AutoDock-GPU write the docked models as PDB records prefixed
# with "DOCKED: ". The record body is standard PDB, i.e. the whitespace INSIDE
# it is fixed-width, not a fixed number of spaces after "USER". Matching on a
# literal "DOCKED: USER  Estimated..." (two spaces) against a file that
# contains "DOCKED: USER    Estimated..." (four) is what left every pose with
# binding_energy=None; cluster_docking_poses() then sorted on None and raised
# TypeError, which _run_smd_for_target logged at DEBUG. Result: A3 reported as
# ON and produced zero clusters on every docking ever run. Regexes with \s+
# now do the matching, so the space count cannot matter again.
_DLG_RUN_RE = re.compile(r"^DOCKED:\s*USER\s+Run\s*=\s*(\d+)")
_DLG_BE_RE = re.compile(
    r"^DOCKED:\s*USER\s+Estimated\s+Free\s+Energy\s+of\s+Binding\s*=\s*"
    r"(-?\d+(?:\.\d+)?)")
_DLG_ATOM_RE = re.compile(r"^DOCKED:\s*(ATOM|HETATM)")


def parse_autodock_dlg_poses(dlg_path) -> list:
    """A3: Parse AutoDock4 .dlg file to extract all GA-run poses.

    Returns list of {binding_energy, run, coords}. Poses without a parsed
    binding energy or without coordinates are DISCARDED rather than returned
    with None fields — a pose that cannot be scored cannot be clustered, and
    passing it on is what produced the TypeError.
    """
    poses = []
    current = None
    n_dropped = 0

    def _close(p):
        nonlocal n_dropped
        if p is None:
            return
        if p["binding_energy"] is None or len(p["coords"]) < 3:
            n_dropped += 1
            return
        poses.append(p)

    with open(dlg_path, errors="replace") as f:
        for line in f:
            if not line.startswith("DOCKED:"):
                continue
            m = _DLG_RUN_RE.match(line)
            if m:
                _close(current)
                current = {"run": int(m.group(1)), "coords": [],
                           "binding_energy": None}
                continue
            if current is None:
                continue
            m = _DLG_BE_RE.match(line)
            if m:
                current["binding_energy"] = float(m.group(1))
                continue
            m = _DLG_ATOM_RE.match(line)
            if m:
                # PDB columns are counted from the start of the ATOM record,
                # NOT from the start of the "DOCKED: " line. Slicing the raw
                # line at [30:38] read 8 columns to the left of x — i.e. the
                # residue-number field — so even the poses that did get a
                # binding energy carried nonsense coordinates.
                body = line[m.start(1):]
                try:
                    current["coords"].append([float(body[30:38]),
                                              float(body[38:46]),
                                              float(body[46:54])])
                except (ValueError, IndexError):
                    pass
    _close(current)

    if n_dropped:
        logger.warning(f"  {Path(dlg_path).name}: {n_dropped} pose block(s) "
                       f"had no parsable binding energy or coordinates and "
                       f"were discarded ({len(poses)} usable)")
    return poses


def cluster_poses_from_dlg(dlg_path, rmsd_cutoff: float = None,
                            max_clusters: int = None) -> list:
    """A3: Cluster AutoDock poses by RMSD.

    NOTE ON config.POSE_CLUSTERING_MIN_SIZE (=3): it is deliberately NOT
    applied here. cluster_docking_poses() defaults to min_cluster_size=1 and
    that is the behaviour that has always been in effect; since A3 has never
    actually produced a cluster (see parse_autodock_dlg_poses), turning on a
    size filter at the same moment the feature starts working would make the
    first real output a filtered one that nobody has ever seen unfiltered.
    Every returned cluster carries its own `cluster_size`, so a caller that
    wants the filter can apply it. The knob is inert, not silently honoured.
    """
    from .config import POSE_CLUSTERING_RMSD_A, POSE_CLUSTERING_MAX_CLUSTERS
    from .utils_analysis import cluster_docking_poses
    rmsd_cutoff = rmsd_cutoff or POSE_CLUSTERING_RMSD_A
    max_clusters = max_clusters or POSE_CLUSTERING_MAX_CLUSTERS

    poses = parse_autodock_dlg_poses(dlg_path)
    if not poses:
        return []
    # All poses are the same ligand, so all coordinate lists must be the same
    # length; a truncated block would make the RMSD comparison broadcast
    # against the wrong shape instead of failing.
    n_atoms = max(len(p["coords"]) for p in poses)
    poses = [p for p in poses if len(p["coords"]) == n_atoms]
    if not poses:
        return []
    clusters = cluster_docking_poses(poses, rmsd_cutoff=rmsd_cutoff)
    # Sort by binding_energy and keep top-N. `or 0.0` guards a None slipping
    # through: sorting a mix of float and None raises TypeError in py3.
    clusters.sort(key=lambda c: (c.get("binding_energy") if
                                 c.get("binding_energy") is not None else 0.0))
    return clusters[:max_clusters]


# ════════════════════════════════════════════════════════════════
# MEMBRANE ACCESSIBILITY OF A DOCKED POSE
# ════════════════════════════════════════════════════════════════
#
# WHY THIS EXISTS, AND WHAT IT IS NOT.
#
# config declares `scored_surface` (polymer-accessible) and `membrane_occluded`
# (lipid-facing on an intact vesicle) per target, and Phase 1 already centres
# and sizes the AutoGrid box on `scored_surface`. That was believed to keep the
# membrane-facing surface out of the search. IT DOES NOT, and the measurement is
# not close: an axis-aligned box drawn around the scored surface contains
#     CD63  100/100 occluded atoms   (100%)
#     CD81   92/97                    (95%)
#     CD9    96/96                   (100%)
# because the occluded residues sit at the BASE of the same LEL, spatially
# interleaved with the scored ones, not off to one side. A rectangular envelope
# cannot separate them; only a per-residue criterion can.
#
# What this filter does: after docking, it measures what fraction of a pose's
# receptor contacts are with membrane-occluded residues. A pose that binds
# predominantly to the lipid-facing surface is not reachable by polymer on an
# intact vesicle, so it is not a valid measurement of vesicle-accessible
# binding, and the pipeline re-scores from the best ACCESSIBLE pose cluster
# instead. This matters asymmetrically across targets — the occluded fraction of
# the docked surface differs by target — so it is a cross-target confound, not a
# uniform inefficiency.
#
# What this filter is NOT: it is not a modified force field and not a membrane
# simulation. The receptor keeps every atom, so sterics and the electrostatic
# environment are unchanged (deleting the occluded residues would carve an
# artificial cavity at the TM exit — the same reason EC1 is carried but not
# scored). The honest description is a POST-HOC ACCESSIBILITY FILTER on poses.

def _occluded_resids(target_cfg: dict) -> set:
    """Membrane-occluded residue numbers, plus EC1 when it is carried unscored.

    EC1 is present in the receptor as steric and electrostatic context but is
    excluded from scoring by config's `ec1_scored: False` — the evidential
    quality of EC1 differs by an order of magnitude between the three targets
    (CD9 fully experimental, CD81 a 3.8 Å trace, CD63 pure AlphaFold), so
    scoring it would turn part of the selectivity signal into a readout of
    template provenance. A pose sitting on EC1 is therefore just as unusable as
    one sitting on lipid, for a different reason.
    """
    occl = set(target_cfg.get("membrane_occluded") or [])
    if not target_cfg.get("ec1_scored", True):
        rng = target_cfg.get("ec1_range")
        if rng:
            occl |= set(range(int(rng[0]), int(rng[1]) + 1))
    return occl


@lru_cache(maxsize=16)
def _receptor_heavy_atoms(receptor_pdb: str, occluded_key: tuple):
    """(xyz, is_occluded) for the receptor's heavy atoms. Cached.

    Called once per pose otherwise — 27 monomers x ~5 clusters x 6 ensemble
    conformers is ~800 re-parses of the same PDB per target. occluded_key is
    part of the cache key because the occluded set decides the boolean mask.
    """
    from Bio.PDB import PDBParser
    import numpy as _np
    occl = set(occluded_key)
    st = PDBParser(QUIET=True).get_structure("r", receptor_pdb)
    xyz, mask = [], []
    for res in st.get_residues():
        is_occl = res.get_id()[1] in occl
        for a in res:
            if a.element == "H":
                continue
            xyz.append(a.get_coord())
            mask.append(is_occl)
    if not xyz:
        return None, None
    return _np.asarray(xyz, dtype=float), _np.asarray(mask, dtype=bool)


def pose_occluded_fraction(receptor_pdb, pose_coords, occluded_resids,
                           cutoff_A: float = 4.0):
    """Fraction of a pose's receptor contacts that are with occluded residues.

    Returns (fraction, n_contacts). fraction is None when the pose touches
    nothing within cutoff_A — an unscoreable geometry rather than a clean one,
    so callers must not read None as "accessible".
    """
    import numpy as _np

    coords = _np.asarray(pose_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return None, 0

    rec_xyz, rec_occl = _receptor_heavy_atoms(
        str(receptor_pdb), tuple(sorted(occluded_resids)))
    if rec_xyz is None:
        return None, 0

    # Receptor atoms within cutoff of ANY ligand atom = the contact shell.
    d2 = ((rec_xyz[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    touched = (d2 <= cutoff_A ** 2).any(axis=1)
    n = int(touched.sum())
    if n == 0:
        return None, 0
    return float(rec_occl[touched].sum()) / n, n


def apply_membrane_accessibility_filter(target: str, target_cfg: dict,
                                        receptor_pdb, results: dict,
                                        dock_dir: Path) -> dict:
    """Re-score each monomer from its best MEMBRANE-ACCESSIBLE pose cluster.

    Adds per monomer:
      occluded_fraction_best_pose  what the reported BE was actually sitting on
      membrane_accessible          verdict for the pose the BE now comes from
      be_reselected_from_cluster   index of the cluster used, when re-scored
      binding_energy_occluded_pose the discarded value, kept for comparison

    A monomer whose EVERY pose cluster is predominantly occluded is recorded as
    MISSING (mean_cluster_energy=None) rather than scored: on an intact vesicle
    the polymer cannot reach that site, so there is no accessible affinity to
    report, and carrying the lipid-facing number forward would credit the
    monomer with binding it cannot do.
    """
    from .config import PHASE2_MAX_OCCLUDED_POSE_FRACTION as _MAXOCC
    from .config import PHASE2_POSE_CONTACT_CUTOFF_A as _CUT

    occl = _occluded_resids(target_cfg)
    if not occl:
        logger.info(f"  [{target}] no membrane_occluded/EC1 residues declared — "
                    f"accessibility filter not applicable")
        return results
    if not Path(receptor_pdb).exists():
        logger.error(f"  [{target}] receptor PDB {receptor_pdb} missing — CANNOT "
                     f"run the membrane-accessibility filter. Binding energies "
                     f"for this target may sit on the lipid-facing surface.")
        return results

    n_reselected = n_dropped = n_checked = 0
    for name, res in results.items():
        if not isinstance(res, dict) or res.get("mean_cluster_energy") is None:
            continue
        work = Path(dock_dir) / f"{target}_{name}"
        dlgs = list(work.rglob("*.dlg"))
        if not dlgs:
            res["membrane_accessible"] = None
            res["occlusion_check"] = "no_dlg"
            continue
        try:
            clusters = cluster_poses_from_dlg(dlgs[0])
        except Exception as e:
            res["membrane_accessible"] = None
            res["occlusion_check"] = f"error: {type(e).__name__}: {e}"
            logger.error(f"  {target}-{name}: accessibility filter could not "
                         f"read poses ({type(e).__name__}: {e})")
            continue
        if not clusters:
            res["membrane_accessible"] = None
            res["occlusion_check"] = "no_poses_parsed"
            continue

        n_checked += 1
        scored = []
        for ci, c in enumerate(clusters):
            # cluster_docking_poses() publishes the cluster's lowest-energy
            # member under 'representative_pose'; that pose carries the coords.
            rep = c.get("representative_pose") or {}
            coords = rep.get("coords") if isinstance(rep, dict) else None
            if not coords:
                continue
            frac, ncon = pose_occluded_fraction(receptor_pdb, coords, occl, _CUT)
            scored.append({"cluster": ci, "binding_energy": c.get("binding_energy"),
                           "occluded_fraction": frac, "n_contacts": ncon,
                           "cluster_size": c.get("cluster_size")})
        if not scored:
            res["membrane_accessible"] = None
            res["occlusion_check"] = "no_pose_coordinates"
            continue

        res["pose_occlusion"] = scored
        best = scored[0]                      # clusters are sorted by energy
        res["occluded_fraction_best_pose"] = best["occluded_fraction"]

        def _ok(s):
            return s["occluded_fraction"] is not None and \
                   s["occluded_fraction"] <= _MAXOCC

        if _ok(best):
            res["membrane_accessible"] = True
            res["occlusion_check"] = "ok"
            continue

        # Defensive: if occluded_fraction could not be computed for ANY pose
        # (e.g. glyco receptor whose "occluded" set is empty because occl is
        # keyed to naked-receptor residue coords in a different frame), skip
        # the filter rather than dropping every monomer as fully occluded —
        # the drop branch below would then try to f-format None with :.0%.
        if all(s.get("occluded_fraction") is None for s in scored):
            res["membrane_accessible"] = None
            res["occlusion_check"] = "occlusion_undetermined"
            continue

        accessible = [s for s in scored if _ok(s)]
        if accessible:
            new = min(accessible,
                      key=lambda s: (s["binding_energy"] is None,
                                     s["binding_energy"]))
            n_reselected += 1
            res["binding_energy_occluded_pose"] = res.get("mean_cluster_energy")
            res["mean_cluster_energy"] = new["binding_energy"]
            res["binding_energy"] = new["binding_energy"]
            res["be_reselected_from_cluster"] = new["cluster"]
            res["membrane_accessible"] = True
            res["occlusion_check"] = "reselected"
            best_frac = best.get("occluded_fraction")
            best_frac_str = f"{best_frac:.0%}" if best_frac is not None else "N/A"
            new_frac = new.get("occluded_fraction")
            new_frac_str = f"{new_frac:.0%}" if new_frac is not None else "N/A"
            logger.error(
                f"  {target}-{name}: best pose sat {best_frac_str} "
                f"on MEMBRANE-OCCLUDED surface (limit {_MAXOCC:.0%}). Re-scored "
                f"from cluster {new['cluster']} "
                f"({new_frac_str} occluded): BE "
                f"{res['binding_energy_occluded_pose']} -> {new['binding_energy']} "
                f"kcal/mol.")
        else:
            n_dropped += 1
            res["binding_energy_occluded_pose"] = res.get("mean_cluster_energy")
            res["mean_cluster_energy"] = None
            res["binding_energy"] = None
            res["membrane_accessible"] = False
            res["occlusion_check"] = "all_poses_occluded"
            best_frac = best.get("occluded_fraction")
            best_frac_str = f"{best_frac:.0%}" if best_frac is not None else "N/A"
            logger.error(
                f"  {target}-{name}: EVERY pose cluster binds predominantly to "
                f"the membrane-occluded surface (best {best_frac_str} "
                f"> {_MAXOCC:.0%}). Recorded as MISSING, not as "
                f"{res['binding_energy_occluded_pose']} kcal/mol — on an intact "
                f"vesicle the polymer cannot reach that site.")

    logger.info(f"  [{target}] membrane-accessibility filter: {n_checked} monomers "
                f"checked, {n_reselected} re-scored from an accessible cluster, "
                f"{n_dropped} dropped as entirely lipid-facing "
                f"({len(occl)} occluded/EC1 residues)")
    return results


# ════════════════════════════════════════════════════════════════
# B3: Decoy Monomer Evaluation — Enrichment Factor
# ════════════════════════════════════════════════════════════════

def evaluate_decoy_baseline(target: str, receptor_pdbqt, center, npts,
                             output_dir: Path,
                             ensemble_pdbqts: list = None,
                             receptor_pdb=None, target_cfg: dict = None) -> dict:
    """B3: Dock decoy monomers (negative control) for enrichment factor.

    Reference: Mysinger 2012 (JCIM) DUD-E benchmark.

    THIS NOW ACTUALLY DOCKS. It used to call smiles_to_mol2() and then assign
    `decoy_be[name] = None  # placeholder`, run no docking at all, and return
    n_valid_decoys=0 with a note saying integration was still required. Every
    caller then found an empty decoy matrix and skipped the enrichment factor,
    so the EF > 1.5 criterion — the ONLY check in the pipeline that asks
    whether the docking scores mean anything at all — has never executed on
    any target, while the config advertised PHASE2_DECOY_BASELINE = True.

    The decoys are docked against the SAME receptor ensemble and merged with
    the SAME estimator as the real monomers. Comparing an ensemble-merged
    monomer score against a single-receptor decoy score would put part of the
    ensemble effect into the enrichment factor.
    """
    from .config import DECOY_MONOMERS, AUTODOCK4_GA_RUNS, N_WORKERS
    from .utils_structure import smiles_to_pdbqt

    decoy_dir = Path(output_dir) / "decoy_baseline" / target
    decoy_dir.mkdir(parents=True, exist_ok=True)

    if not DECOY_MONOMERS:
        logger.warning(f"  [{target}] DECOY_MONOMERS is empty — no decoy "
                       f"baseline, so no enrichment factor for this target.")
        return {"decoy_monomers_tested": [], "decoy_be_matrix": {},
                "n_valid_decoys": 0, "docked": False,
                "error": "DECOY_MONOMERS is empty"}

    # 1. Ligand preparation (same converter as the real monomers).
    decoy_pdbqts, prep_errors = {}, {}
    for name, info in DECOY_MONOMERS.items():
        try:
            decoy_pdbqts[name] = smiles_to_pdbqt(info["smiles"], name,
                                                 decoy_dir / "ligands")
        except Exception as e:
            logger.error(f"  [{target}] decoy {name}: PDBQT preparation "
                         f"failed ({type(e).__name__}: {e})")
            prep_errors[name] = f"{type(e).__name__}: {e}"
    if not decoy_pdbqts:
        return {"decoy_monomers_tested": list(DECOY_MONOMERS),
                "decoy_be_matrix": {n: None for n in DECOY_MONOMERS},
                "n_valid_decoys": 0, "docked": False,
                "prep_errors": prep_errors,
                "error": "no decoy ligand could be prepared"}

    # 2. Dock against every receptor conformer the real monomers saw.
    receptors = list(ensemble_pdbqts) if ensemble_pdbqts else [receptor_pdbqt]
    per_conformer = {}
    for ci, rec in enumerate(receptors):
        label = "crystal" if ci == 0 else f"md_conf{ci}"
        # The decoy arm gets the SAME membrane-accessibility filter as the
        # real monomers. It has to: the enrichment factor asks whether real
        # monomers outscore inert molecules AT THE ACCESSIBLE SITE, and
        # filtering only the real arm would push its energies up while leaving
        # the decoy scale untouched — systematically UNDERSTATING EF, i.e.
        # making the one check that asks whether these scores mean anything
        # fail for a reason that has nothing to do with the chemistry.
        _dec_rec_pdb = Path(rec).with_suffix(".pdb")
        if not _dec_rec_pdb.exists():
            _dec_rec_pdb = receptor_pdb        # may be None -> filter skipped
        conf_results = _run_smd_for_target(
            f"{target}_decoy", Path(rec), decoy_pdbqts, center, npts,
            decoy_dir / label,
            ga_runs=AUTODOCK4_GA_RUNS, n_workers=N_WORKERS,
            binding_sites=None,
            pose_clustering=False,      # decoys are a scale, not a mechanism
            receptor_pdb=_dec_rec_pdb,
            target_cfg=target_cfg,
        )
        for m, r in conf_results.items():
            if r.get("mean_cluster_energy") is None:
                continue
            per_conformer.setdefault(m, []).append((label, r))

    merged = _merge_across_conformers(f"{target}_decoy", per_conformer)
    decoy_be = {n: (merged[n]["mean_cluster_energy"] if n in merged else None)
                for n in DECOY_MONOMERS}
    valid = {k: v for k, v in decoy_be.items() if v is not None}

    if not valid:
        logger.error(
            f"  [{target}] EVERY decoy docking failed. The enrichment factor "
            f"cannot be computed, so nothing validates that this target's "
            f"docking scores separate real monomers from small inert "
            f"molecules. Reported as enrichment_status='no_decoys_docked'.")
    else:
        logger.info(f"  [{target}] decoy baseline: {len(valid)}/"
                    f"{len(DECOY_MONOMERS)} decoys docked over "
                    f"{len(receptors)} receptor(s), mean BE = "
                    f"{float(np.mean(list(valid.values()))):.2f} kcal/mol")

    return {
        "decoy_monomers_tested": list(DECOY_MONOMERS),
        "decoy_be_matrix": decoy_be,
        "n_valid_decoys": len(valid),
        "docked": True,
        "n_receptor_conformers": len(receptors),
        "prep_errors": prep_errors,
        "decoy_details": {
            m: {k: v for k, v in r.items() if k != "clusters"}
            for m, r in merged.items()},
    }
