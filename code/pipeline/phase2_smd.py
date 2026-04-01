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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_phase2(phase1_results: dict = None,
               target_names: list = None,
               output_dir: str = None) -> dict:
    """
    Phase 2 entry point: SMD screening of all monomers × all targets.

    Parameters
    ----------
    phase1_results : output from Phase 1 (epitope structures)
    target_names : filter to specific targets
    output_dir : output directory

    Returns
    -------
    dict : {
        "be_matrix": {target: {monomer: energy}},
        "selectivity": {target: {monomer: ddg}},
        "filtered": {target: [monomer_names]},
    }
    """
    from .config import (TARGETS, FUNCTIONAL_MONOMERS, N_WORKERS,
                         get_output_path, AUTODOCK4_GA_RUNS,
                         SMD_BE_THRESHOLD, SMD_DDG_THRESHOLD)

    if output_dir is None:
        output_dir = str(get_output_path("phase2"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        target_names = list(phase1_results.keys())

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

    for target in target_names:
        t_result = phase1_results.get(target, {})
        if "error" in t_result:
            logger.warning(f"Skipping {target} (Phase 1 error)")
            continue

        receptor_pdbqt = Path(t_result["receptor_pdbqt"])
        center = tuple(t_result["grid_center"])
        npts = tuple(t_result["grid_npts"])

        # Ensemble docking: collect all receptor PDBQTs (original + MD conformers)
        ensemble_pdbqts = [receptor_pdbqt]
        if "ensemble_receptor_pdbqts" in t_result:
            ensemble_pdbqts.extend(
                Path(p) for p in t_result["ensemble_receptor_pdbqts"]
                if Path(p).exists()
            )

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

        # Dock to each conformer, take best BE per monomer
        all_conf_results = {}
        for ci, conf_pdbqt in enumerate(ensemble_pdbqts):
            conf_label = "crystal" if ci == 0 else f"md_conf{ci}"
            if len(ensemble_pdbqts) > 1:
                logger.info(f"  [{target}] Ensemble conformer {ci+1}/"
                            f"{len(ensemble_pdbqts)} ({conf_label})")

            conf_results = _run_smd_for_target(
                target, conf_pdbqt, monomer_pdbqts,
                center, npts, output_dir / f"smd_{target}_{conf_label}",
                ga_runs=AUTODOCK4_GA_RUNS,
                n_workers=N_WORKERS,
                binding_sites=binding_sites,
            )

            # Merge: keep best BE per monomer across conformers (skip failures)
            for m, r in conf_results.items():
                if r.get("mean_cluster_energy") is None:
                    continue  # skip failed docking
                prev = all_conf_results.get(m)
                if prev is None or prev.get("mean_cluster_energy") is None \
                        or r["mean_cluster_energy"] < prev["mean_cluster_energy"]:
                    all_conf_results[m] = r

        target_results = all_conf_results

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

    # 3. Compute selectivity matrix
    logger.info("\nComputing selectivity matrix...")
    selectivity = _compute_selectivity(be_matrix, target_names)

    # 4. Filter candidates
    filtered = _filter_monomers(
        be_matrix, selectivity, target_names,
        be_threshold=SMD_BE_THRESHOLD,
        ddg_threshold=SMD_DDG_THRESHOLD,
    )

    # 5. Generate outputs
    results = {
        "be_matrix": be_matrix,
        "selectivity": selectivity,
        "filtered": filtered,
        "contact_md_scores": contact_scores,
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
                           output_dir / "phase2_selectivity.csv")

    # Generate heatmap
    _plot_heatmap(be_matrix, output_dir / "phase2_heatmap.png")

    # Log summary
    for target in target_names:
        filt = filtered.get(target, [])
        logger.info(f"[{target}] Filtered monomers ({len(filt)}): {filt}")

    return results


def _prepare_all_monomers(monomer_lib: dict, output_dir: Path) -> dict:
    """Prepare PDBQT files for all monomers. Returns {name: pdbqt_path}."""
    from .utils_structure import smiles_to_pdbqt

    output_dir = Path(output_dir)
    pdbqts = {}
    for name, info in monomer_lib.items():
        smiles = info["smiles"]
        pdbqt = output_dir / f"{name}.pdbqt"
        if pdbqt.exists():
            pdbqts[name] = pdbqt
            continue
        try:
            pdbqt_path = smiles_to_pdbqt(smiles, name, output_dir)
            pdbqts[name] = pdbqt_path
        except Exception as e:
            logger.warning(f"Failed to prepare {name}: {e}")
    logger.info(f"Prepared {len(pdbqts)}/{len(monomer_lib)} monomer PDBQTs")
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
                hb = analyze_hbond_types(Path(dlg), receptor_pdb)
                result["hbond_analysis"] = hb
                if hb.get("structural_disruption_risk"):
                    logger.warning(
                        f"  {target}-{monomer_name}: HIGH backbone H-bond "
                        f"ratio ({hb['backbone_ratio']:.0%}) — "
                        "potential 2° structure disruption (Sullivan 2019)"
                    )
            except Exception as e:
                logger.debug(f"H-bond analysis failed for {monomer_name}: {e}")
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


def _run_smd_for_target(target: str, receptor_pdbqt: Path,
                         monomer_pdbqts: dict,
                         center: tuple, npts: tuple,
                         output_dir: Path,
                         ga_runs: int = 50,
                         n_workers: int = 4,
                         binding_sites: list = None) -> dict:
    """Run SMD docking for one target against all monomers."""
    from .utils_autodock import dock_single

    results = {}
    dock_dir = output_dir / f"smd_{target}"
    dock_dir.mkdir(parents=True, exist_ok=True)

    # Parallel docking
    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for name, pdbqt in monomer_pdbqts.items():
            work = dock_dir / f"{target}_{name}"
            future = executor.submit(
                dock_single,
                receptor_pdbqt, pdbqt,
                center, npts, work,
                ga_runs=ga_runs,
            )
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results[name] = result
                be = result.get("mean_cluster_energy", 0.0)
                logger.info(f"  {target}-{name}: BE = {be:.2f} kcal/mol")
            except Exception as e:
                logger.error(f"  {target}-{name} failed: {e}")
                results[name] = {"mean_cluster_energy": 0.0,
                                 "binding_energy": 0.0, "error": str(e)}

    return results


def _compute_selectivity(be_matrix: dict, target_names: list) -> dict:
    """
    Compute selectivity ΔΔG for each (target, monomer) pair.
    ΔΔG = BE(target) - mean(BE(non-targets))
    More negative = more selective for target.
    """
    selectivity = {}
    for target in target_names:
        non_targets = [t for t in target_names if t != target]
        selectivity[target] = {}

        for monomer in be_matrix.get(target, {}):
            be_target = be_matrix[target].get(monomer, 0.0)
            be_others = [be_matrix.get(t, {}).get(monomer, 0.0)
                         for t in non_targets]
            mean_others = np.mean(be_others) if be_others else 0.0
            ddg = be_target - mean_others
            selectivity[target][monomer] = round(float(ddg), 3)

    return selectivity


def _filter_monomers(be_matrix: dict, selectivity: dict,
                      target_names: list,
                      be_threshold: float = -2.0,
                      ddg_threshold: float = -0.5) -> dict:
    """Filter monomers by binding energy and selectivity thresholds."""
    filtered = {}
    for target in target_names:
        candidates = []
        for monomer in be_matrix.get(target, {}):
            be = be_matrix[target].get(monomer, 0.0)
            ddg = selectivity.get(target, {}).get(monomer, 0.0)
            if be <= be_threshold and ddg <= ddg_threshold:
                candidates.append((monomer, be, ddg))

        # Sort by binding energy (most negative first)
        candidates.sort(key=lambda x: x[1])
        filtered[target] = [c[0] for c in candidates]
    return filtered


def _save_selectivity_csv(be_matrix: dict, selectivity: dict,
                           filtered: dict, output_path: Path):
    """Save combined results as CSV."""
    rows = []
    for target in be_matrix:
        for monomer in be_matrix[target]:
            rows.append({
                "target": target,
                "monomer": monomer,
                "binding_energy": be_matrix[target][monomer],
                "selectivity_ddg": selectivity.get(target, {}).get(monomer, 0),
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

        data = np.zeros((len(targets), len(monomers)))
        for i, t in enumerate(targets):
            for j, m in enumerate(monomers):
                data[i, j] = be_matrix[t].get(m, 0.0)

        fig, ax = plt.subplots(figsize=(max(12, len(monomers) * 0.6), 4))
        im = ax.imshow(data, cmap="RdYlBu", aspect="auto")
        ax.set_xticks(range(len(monomers)))
        ax.set_xticklabels(monomers, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets)
        ax.set_xlabel("Monomer")
        ax.set_ylabel("Target")
        ax.set_title("SMD Binding Energy (kcal/mol)")
        plt.colorbar(im, ax=ax, label="BE (kcal/mol)")

        # Annotate values
        for i in range(len(targets)):
            for j in range(len(monomers)):
                ax.text(j, i, f"{data[i, j]:.1f}",
                        ha="center", va="center", fontsize=6)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Heatmap → {output_path}")
    except ImportError:
        logger.warning("matplotlib not available, skipping heatmap")
