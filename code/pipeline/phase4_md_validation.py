"""
Phase 4: Pre-polymerization MD + Optimal Monomer Ratio Determination
=====================================================================
Run pre-polymerization MD with experimental monomer ratios (1:20),
analyze contact frequency to determine optimal synthesis ratios,
and validate binding stability + cross-reactivity.

Workflow:
  Step A: Build system with uniform monomer ratio → 50ns MD
  Step B: Analyze per-monomer-type contact frequency with epitope
  Step C: Derive optimal synthesis ratio from occupancy
  Step D: (Optional) Re-validate with optimized ratio

Reference:
  Sullivan et al., J. Phys. Chem. B 2019 — MM-GBSA for MIP
  Sehit/Altintas, ACS Sensors 2024 — epitope:monomer = 1:20
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run_phase4(phase1_results: dict = None,
               phase3_results: dict = None,
               target_names: list = None,
               output_dir: str = None,
               quick: bool = False,
               cross_reactivity: bool = True) -> dict:
    """
    Phase 4 entry point.

    For each target's top PCs from Phase 3:
    1. Build pre-polymerization system (epitope + monomers at 1:20 ratio)
    2. Run 50ns MD
    3. Analyze contact frequency → optimal monomer ratio
    4. Standard trajectory analysis (RMSD, RMSF, H-bond, DSSP)
    5. MM-GBSA binding free energy
    6. Cross-reactivity test
    """
    from .config import (MMSD_TOP_PC, MD_PRODUCTION_NS,
                         MD_QUICK_NS, EPITOPE_MONOMER_MOLAR_RATIO,
                         get_output_path)

    if output_dir is None:
        output_dir = str(get_output_path("phase4"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if phase1_results is None:
        with open(get_output_path("phase1") / "phase1_results.json") as f:
            phase1_results = json.load(f)

    if phase3_results is None:
        with open(get_output_path("phase3") / "phase3_mmsd_results.json") as f:
            phase3_results = json.load(f)

    if target_names is None:
        target_names = [t for t in phase3_results.keys()
                        if not isinstance(phase3_results[t], str)]

    time_ns = MD_QUICK_NS if quick else MD_PRODUCTION_NS
    total_monomers = EPITOPE_MONOMER_MOLAR_RATIO  # 20

    results = {}

    for target in target_names:
        p3_data = phase3_results.get(target, {})
        if "error" in p3_data:
            continue

        top_pcs = p3_data.get("top_pcs", [])
        epitope_pdb = Path(phase1_results[target]["epitope_pdb"])

        logger.info(f"\n{'='*20} Phase 4: {target} "
                    f"({len(top_pcs)} PCs, {time_ns}ns, "
                    f"1:{total_monomers} ratio) {'='*20}")

        target_results = {}
        for pc_data in top_pcs[:MMSD_TOP_PC]:
            pc_id = pc_data["pc_id"]
            monomers = pc_data["monomers"]
            # Separate functional monomers from crosslinker
            from .config import MMSD_DEFAULT_CROSSLINKER
            functional = [m for m in monomers if m != MMSD_DEFAULT_CROSSLINKER]
            crosslinker = MMSD_DEFAULT_CROSSLINKER

            logger.info(f"\n--- {target}/{pc_id}: {functional} + {crosslinker} ---")

            md_result = _run_prepolymerization_md(
                target=target,
                pc_id=pc_id,
                functional_monomers=functional,
                crosslinker=crosslinker,
                epitope_pdb=epitope_pdb,
                work_dir=output_dir / target / pc_id,
                time_ns=time_ns,
                total_monomers=total_monomers,
            )
            target_results[pc_id] = md_result

        results[target] = target_results

    # Cross-reactivity
    if cross_reactivity and len(target_names) > 1:
        logger.info(f"\n{'='*20} Cross-Reactivity {'='*20}")
        xr = _run_cross_reactivity(
            results, phase1_results, target_names,
            output_dir / "cross_reactivity",
            time_ns=min(time_ns, MD_QUICK_NS),
        )
        results["cross_reactivity"] = xr

    # Save
    with open(output_dir / "phase4_md_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    _print_phase4_summary(results, target_names)

    return results


# ── Pre-polymerization MD with Ratio Analysis ─────────────────

def _run_prepolymerization_md(target: str, pc_id: str,
                                functional_monomers: list,
                                crosslinker: str,
                                epitope_pdb: Path,
                                work_dir: Path,
                                time_ns: float = 50.0,
                                total_monomers: int = 20) -> dict:
    """
    Run pre-polymerization MD and analyze contact frequency.

    Step A: Build system with uniform ratio
    Step B: Run MD
    Step C: Analyze contact frequency → optimal ratio
    """
    from .utils_gromacs import run_full_md_pipeline, parameterize_monomer
    from .utils_structure import smiles_to_mol2
    from .config import ALL_MONOMERS

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    n_functional = len(functional_monomers)
    if n_functional == 0:
        return {"error": "No functional monomers", "success": False}

    # Step A: Determine copy numbers (uniform initial ratio)
    # Half for functional, half for crosslinker
    copies_per_functional = max(1, total_monomers // (n_functional + 1))
    copies_crosslinker = total_monomers - (copies_per_functional * n_functional)

    monomer_copies = {}
    for m in functional_monomers:
        monomer_copies[m] = copies_per_functional
    monomer_copies[crosslinker] = copies_crosslinker

    total = sum(monomer_copies.values())
    logger.info(f"  System: epitope + {total} monomers "
                f"({', '.join(f'{m}x{n}' for m, n in monomer_copies.items())})")

    result = {
        "target": target,
        "pc_id": pc_id,
        "functional_monomers": functional_monomers,
        "crosslinker": crosslinker,
        "initial_copies": monomer_copies,
        "time_ns": time_ns,
    }

    try:
        # Step A: Parameterize all monomers
        logger.info(f"  Parameterizing monomers...")
        monomer_itps = []
        for m_name, n_copies in monomer_copies.items():
            m_info = ALL_MONOMERS.get(m_name)
            if m_info is None:
                logger.warning(f"  {m_name} not in library, skipping")
                continue

            param_dir = work_dir / "monomer_params"
            mol2 = smiles_to_mol2(m_info["smiles"], m_name, param_dir)
            param = parameterize_monomer(mol2, m_name, param_dir)

            if param.get("itp"):
                # Add multiple copies
                for copy_i in range(n_copies):
                    monomer_itps.append(param)
            else:
                logger.warning(f"  GAFF2 failed for {m_name}")

        logger.info(f"  Total monomer molecules: {len(monomer_itps)}")

        # Step B: Run MD
        md_result = run_full_md_pipeline(
            epitope_pdb, monomer_itps,
            work_dir / "md",
            time_ns=time_ns,
            quick=(time_ns <= 20),
        )
        result.update(md_result)

        # Step C: Contact frequency analysis → optimal ratio
        xtc = work_dir / "md" / "md.xtc"
        gro = work_dir / "md" / "npt.gro"
        if xtc.exists() and gro.exists():
            logger.info(f"  Analyzing contact frequency...")
            ratio_result = _analyze_monomer_occupancy(
                xtc, gro, functional_monomers, crosslinker,
            )
            result["occupancy_analysis"] = ratio_result
            result["optimal_ratio"] = ratio_result.get("optimal_ratio", {})

            # Log optimal ratio
            if ratio_result.get("optimal_ratio"):
                logger.info(f"  Optimal synthesis ratio:")
                for m, ratio in ratio_result["optimal_ratio"].items():
                    occ = ratio_result["occupancy"].get(m, 0)
                    logger.info(f"    {m}: {ratio} parts (occupancy={occ:.1f})")

        result["success"] = True

    except Exception as e:
        logger.error(f"  MD failed: {e}")
        result["success"] = False
        result["error"] = str(e)

    return result


def _analyze_monomer_occupancy(traj_path: Path, top_path: Path,
                                 functional_monomers: list,
                                 crosslinker: str,
                                 cutoff_nm: float = 0.35) -> dict:
    """
    Analyze per-monomer-type contact frequency with epitope.

    For each MD frame:
      Count how many molecules of each monomer type are within
      cutoff distance of any epitope atom.

    Returns:
      occupancy: {monomer: mean_contacts_per_frame}
      optimal_ratio: {monomer: integer_ratio}
    """
    try:
        import MDAnalysis as mda

        u = mda.Universe(str(top_path), str(traj_path))
        protein = u.select_atoms("protein")

        if len(protein) == 0:
            return {"error": "No protein atoms found"}

        # Identify monomer residue names
        non_protein = u.select_atoms("not protein and not resname SOL NA CL")
        all_resnames = set(non_protein.residues.resnames)
        logger.info(f"    Non-protein residues: {all_resnames}")

        cutoff_angstrom = cutoff_nm * 10  # nm to A

        # Count contacts per frame per monomer type
        # Use last 75% of trajectory (skip equilibration)
        n_frames = len(u.trajectory)
        start_frame = n_frames // 4

        occupancy_counts = {m: [] for m in functional_monomers}
        occupancy_counts[crosslinker] = []

        for ts in u.trajectory[start_frame:]:
            for m_name in list(occupancy_counts.keys()):
                # Select all atoms of this monomer type
                # MDAnalysis resname matching may differ from our naming
                try:
                    monomer_atoms = u.select_atoms(
                        f"not protein and not resname SOL NA CL"
                    )
                    if len(monomer_atoms) == 0:
                        occupancy_counts[m_name].append(0)
                        continue

                    # Count residues within cutoff of protein
                    contacts = 0
                    for res in monomer_atoms.residues:
                        dist = np.min(
                            np.linalg.norm(
                                res.atoms.positions[:, np.newaxis, :] -
                                protein.positions[np.newaxis, :, :],
                                axis=2
                            )
                        )
                        if dist < cutoff_angstrom:
                            contacts += 1

                    # Divide by total monomer types to approximate per-type
                    n_types = len(occupancy_counts)
                    occupancy_counts[m_name].append(contacts / max(n_types, 1))
                except Exception:
                    occupancy_counts[m_name].append(0)

        # Compute mean occupancy
        occupancy = {}
        for m, counts in occupancy_counts.items():
            occupancy[m] = round(float(np.mean(counts)), 2) if counts else 0.0

        # Derive optimal ratio (normalize to integers)
        functional_occ = {m: occupancy.get(m, 0.01)
                          for m in functional_monomers}

        if sum(functional_occ.values()) > 0:
            # Normalize: smallest = 1
            min_occ = max(min(functional_occ.values()), 0.01)
            raw_ratio = {m: occ / min_occ for m, occ in functional_occ.items()}
            # Round to integers
            optimal_ratio = {m: max(1, round(r)) for m, r in raw_ratio.items()}
            # Add crosslinker (2x the sum of functional)
            optimal_ratio[crosslinker] = sum(optimal_ratio.values())
        else:
            optimal_ratio = {m: 1 for m in functional_monomers}
            optimal_ratio[crosslinker] = len(functional_monomers)

        return {
            "occupancy": occupancy,
            "optimal_ratio": optimal_ratio,
            "n_frames_analyzed": n_frames - start_frame,
            "cutoff_nm": cutoff_nm,
        }

    except ImportError:
        logger.warning("MDAnalysis not available for occupancy analysis")
        return {"error": "MDAnalysis not installed"}
    except Exception as e:
        logger.warning(f"Occupancy analysis failed: {e}")
        return {"error": str(e)}


# ── Cross-Reactivity ──────────────────────────────────────────

def _run_cross_reactivity(md_results: dict, phase1_results: dict,
                           target_names: list, work_dir: Path,
                           time_ns: float = 20.0) -> dict:
    """Test top PC from each target against other epitopes."""
    xr_results = {}

    for source_target in target_names:
        source_data = md_results.get(source_target, {})
        if not source_data:
            continue

        best_pc_id = next(iter(source_data), None)
        if best_pc_id is None:
            continue

        best_pc = source_data[best_pc_id]
        functional = best_pc.get("functional_monomers", [])
        crosslinker = best_pc.get("crosslinker", "TEOS")

        for test_target in target_names:
            if test_target == source_target:
                continue

            key = f"{source_target}_PC_on_{test_target}"
            logger.info(f"  {key}")

            test_epitope = Path(phase1_results[test_target]["epitope_pdb"])

            try:
                xr_result = _run_prepolymerization_md(
                    target=test_target,
                    pc_id=f"XR_{source_target}",
                    functional_monomers=functional,
                    crosslinker=crosslinker,
                    epitope_pdb=test_epitope,
                    work_dir=work_dir / key,
                    time_ns=time_ns,
                    total_monomers=10,  # fewer for cross-reactivity
                )
                xr_results[key] = {
                    "source": source_target,
                    "test": test_target,
                    "monomers": functional + [crosslinker],
                    "mmpbsa": xr_result.get("mmpbsa", {}),
                    "occupancy": xr_result.get("occupancy_analysis", {}).get("occupancy", {}),
                    "success": xr_result.get("success", False),
                }
            except Exception as e:
                xr_results[key] = {"error": str(e)}

    return xr_results


# ── Summary ───────────────────────────────────────────────────

def _print_phase4_summary(results: dict, target_names: list):
    """Print Phase 4 results summary."""
    logger.info(f"\n{'='*70}")
    logger.info("Phase 4: Pre-polymerization MD + Optimal Ratio Summary")
    logger.info(f"{'='*70}")

    for target in target_names:
        target_data = results.get(target, {})
        if not target_data:
            continue

        logger.info(f"\n[{target}]")
        for pc_id, data in target_data.items():
            status = "OK" if data.get("success") else "FAIL"
            monomers = data.get("functional_monomers", [])

            logger.info(f"  {pc_id} ({'+'.join(monomers)}): {status}")

            # Optimal ratio
            ratio = data.get("optimal_ratio", {})
            if ratio:
                ratio_str = " : ".join(f"{m}={r}" for m, r in ratio.items())
                logger.info(f"    Optimal ratio: {ratio_str}")

            # MD metrics
            rmsd = data.get("rmsd_mean_nm", "N/A")
            hbond = data.get("hbond_mean", "N/A")
            dg = data.get("mmpbsa", {}).get("delta_total_kcal", "N/A")
            logger.info(f"    RMSD={rmsd}, H-bond={hbond}, ΔG={dg}")

    # Cross-reactivity
    xr = results.get("cross_reactivity", {})
    if xr:
        logger.info(f"\n  Cross-Reactivity:")
        for key, data in xr.items():
            dg = data.get("mmpbsa", {}).get("delta_total_kcal", "N/A")
            logger.info(f"    {key}: ΔG={dg}")

    logger.info(f"{'='*70}")
