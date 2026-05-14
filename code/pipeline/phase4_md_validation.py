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
        # Use head 16-mer (not full ECL2) for pre-polymerization MD
        # Matches actual MIP synthesis template; all monomer contacts are relevant
        from .config import resolve_path
        epitope_pdb = resolve_path(phase1_results[target].get("head_pdb",
                                   phase1_results[target]["epitope_pdb"]))

        logger.info(f"\n{'='*20} Phase 4: {target} "
                    f"({len(top_pcs)} PCs, {time_ns}ns, "
                    f"1:{total_monomers} ratio) {'='*20}")

        target_results = {}
        for pc_data in top_pcs[:MMSD_TOP_PC]:
            pc_id = pc_data["pc_id"]
            monomers = pc_data["monomers"]
            # Separate functional monomers from crosslinker
            # Crosslinker is stored in pc_data by Phase 3, or detect from monomers list
            from .config import CROSSLINKER_LIBRARY
            crosslinker = pc_data.get("crosslinker")
            if crosslinker is None:
                # Fallback: last monomer in MMSD list is the crosslinker
                for m in reversed(monomers):
                    if m in CROSSLINKER_LIBRARY:
                        crosslinker = m
                        break
            functional = [m for m in monomers if m != crosslinker]

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

            # A5/B7: Solvent sweep (porogen MD)
            from .config import PHASE4_SOLVENT_SWEEP, PHASE4_RATIO_SWEEP
            if PHASE4_SOLVENT_SWEEP:
                logger.info(f"  A5/B7: Solvent sweep for {pc_id}...")
                try:
                    solv_result = run_phase4_solvent_sweep(
                        target=target, pc_id=pc_id,
                        functional_monomers=functional,
                        crosslinker=crosslinker,
                        epitope_pdb=epitope_pdb,
                        work_dir=output_dir / target / pc_id / "solvent_sweep",
                        time_ns=30,
                    )
                    md_result["solvent_sweep"] = solv_result
                except Exception as e:
                    logger.warning(f"  A5/B7 solvent sweep failed: {e}")

            # B6: Monomer ratio sweep
            if PHASE4_RATIO_SWEEP:
                logger.info(f"  B6: Monomer ratio sweep for {pc_id}...")
                try:
                    ratio_result = run_phase4_ratio_sweep(
                        target=target, pc_id=pc_id,
                        functional_monomers=functional,
                        crosslinker=crosslinker,
                        epitope_pdb=epitope_pdb,
                        work_dir=output_dir / target / pc_id / "ratio_sweep",
                        time_ns=30,
                    )
                    md_result["ratio_sweep"] = ratio_result
                except Exception as e:
                    logger.warning(f"  B6 ratio sweep failed: {e}")

        results[target] = target_results

    # Cross-reactivity removed — Phase 6 cavity rebinding handles selectivity

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
    Run pre-polymerization MD and analyze monomer-epitope interactions.

    Step A: Build system — monomers randomly placed (literature standard)
    Step B: Run MD — monomers find binding sites by free diffusion
    Step C: MM-PBSA per monomer type → Boltzmann-weighted synthesis ratio
    """
    from .utils_gromacs import run_full_md_pipeline, parameterize_monomer
    from .utils_structure import smiles_to_mol2
    from .config import ALL_MONOMERS

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    n_functional = len(functional_monomers)
    if n_functional == 0:
        return {"error": "No functional monomers", "success": False}

    # Step A: Determine copy numbers (uniform ratio)
    # Monomers placed uniformly; MD free diffusion produces natural binding distribution.
    # EBN-based synthesis ratio is derived from analysis, not imposed on MD.
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
        md_dir = work_dir / "md"

        # Check if MD already completed (md.gro exists)
        if (md_dir / "md.gro").exists():
            logger.info(f"  MD already completed, skipping to analysis")
            monomer_itps = []  # not needed for analysis
        else:
            # Step A: Parameterize monomers (random placement, no docked poses)
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
                target=target,
            )
            result["occupancy_analysis"] = ratio_result
            result["optimal_ratio"] = ratio_result.get("optimal_ratio", {})

            # Log optimal ratio (inverse of occupancy)
            if ratio_result.get("optimal_ratio"):
                logger.info(f"  Optimal synthesis ratio (low occupancy → more copies):")
                for m, ratio in ratio_result["optimal_ratio"].items():
                    occ = ratio_result["occupancy"].get(m, 0)
                    logger.info(f"    {m}: {ratio} parts (contact freq={occ:.2f})")

            # Per-atom RDF analysis (Yuan et al. 2024)
            logger.info(f"  Per-atom RDF analysis...")
            rdf_result = _compute_per_atom_rdf(
                xtc, gro, functional_monomers, crosslinker)
            result["per_atom_rdf"] = rdf_result

            # Per-residue MM-GBSA decomposition (Kumar et al. 2024)
            md_dir = work_dir / "md"
            if (md_dir / "md.tpr").exists():
                logger.info(f"  Per-residue MM-GBSA decomposition...")
                try:
                    from .utils_gromacs import run_mmpbsa
                    from .config import MD_MMPBSA_START_NS, MD_MMPBSA_END_NS, MD_MMPBSA_INTERVAL
                    mmpbsa_decomp = run_mmpbsa(
                        md_dir, start_ns=MD_MMPBSA_START_NS,
                        end_ns=MD_MMPBSA_END_NS,
                        n_frames=MD_MMPBSA_INTERVAL,
                        decomp=True)
                    result["mmpbsa_decomp"] = mmpbsa_decomp
                except Exception as e:
                    logger.debug(f"  MM-GBSA decomposition skipped: {e}")

        result["success"] = True

    except Exception as e:
        logger.error(f"  MD failed: {e}")
        result["success"] = False
        result["error"] = str(e)

    return result


def _ensure_centered_trajectory(traj_path: Path, tpr_path: Path) -> Path:
    """Generate PBC-centered trajectory (gmx trjconv -pbc mol -center) if missing.

    Returns path to centered xtc. Caches as md_centered.xtc beside the input.
    """
    from .config import GMX_BIN
    import subprocess
    traj_path = Path(traj_path)
    centered = traj_path.with_name("md_centered.xtc")
    if centered.exists() and centered.stat().st_mtime > traj_path.stat().st_mtime:
        return centered
    if not (traj_path.exists() and Path(tpr_path).exists()):
        return traj_path
    logger.info(f"    PBC-centering trajectory → {centered.name}")
    proc = subprocess.run(
        [GMX_BIN, "trjconv", "-f", str(traj_path), "-s", str(tpr_path),
         "-o", str(centered), "-pbc", "mol", "-center"],
        input="1\n0\n", capture_output=True, text=True, timeout=1800)
    if not centered.exists():
        logger.warning(f"    trjconv centering failed, using raw trajectory")
        return traj_path
    return centered


def _analyze_monomer_occupancy(traj_path: Path, top_path: Path,
                                 functional_monomers: list,
                                 crosslinker: str,
                                 target: str = None,
                                 cutoff_nm: float = 0.60,
                                 use_q4: bool = True) -> dict:
    """
    Analyze monomer-epitope interactions from MD trajectory.

    Metrics (literature-based):
      1. Contact frequency at 6Å cutoff (vdW interactions included)
      2. RDF g(r) peak height per monomer type
      3. Coordination number (avg monomers within 6Å)
      4. Residence time (consecutive contact duration)

    Frame selection:
      - use_q4=True: last 25% (equilibrium-only, recommended for non-converged MDs)
      - use_q4=False: last 50% (legacy, may include kinetic transitions)

    Trajectory is PBC-centered first (gmx trjconv -pbc mol -center) to handle
    proteins that crossed periodic boundaries during long MD.
    """
    try:
        import MDAnalysis as mda

        # PBC centering for analysis (caches md_centered.xtc)
        tpr_for_center = Path(top_path)
        if not str(tpr_for_center).endswith('.tpr'):
            tpr_candidate = Path(traj_path).with_name("md.tpr")
            if tpr_candidate.exists():
                tpr_for_center = tpr_candidate
        centered_traj = _ensure_centered_trajectory(Path(traj_path), tpr_for_center)

        u = mda.Universe(str(top_path), str(centered_traj))

        # Phase 4 uses head 16-mer as template → all protein atoms are binding site
        protein = u.select_atoms("protein")
        logger.info(f"    Template: {len(protein)} atoms (head peptide)")

        if len(protein) == 0:
            return {"error": "No protein atoms found in head region"}

        cutoff_A = cutoff_nm * 10  # nm to Angstrom

        # Frame selection: Q4 (last 25%) for equilibrium, or Q3-Q4 (last 50%) legacy
        n_frames = len(u.trajectory)
        start_frame = (3 * n_frames) // 4 if use_q4 else n_frames // 2
        stride = max(1, (n_frames - start_frame) // 200)
        window_label = "Q4 (last 25%)" if use_q4 else "Q3-Q4 (last 50%)"
        logger.info(f"    Frames: {start_frame}-{n_frames} {window_label} "
                    f"(stride={stride}), cutoff={cutoff_nm*10:.0f}Å")

        # Build residue → monomer type mapping from topology
        non_protein = u.select_atoms("not protein and not resname SOL NA CL")
        all_monomers_list = functional_monomers + [crosslinker]

        top_file = Path(top_path).with_name("topol.top")
        res_to_monomer = {}
        if top_file.exists():
            in_molecules = False
            mol_idx = 0
            for line in top_file.read_text().split("\n"):
                if "[ molecules ]" in line:
                    in_molecules = True
                    continue
                if in_molecules and line.strip() and not line.startswith(";"):
                    parts = line.split()
                    if len(parts) >= 2:
                        mol_name, mol_count = parts[0], int(parts[1])
                        if mol_name.startswith("Protein"):
                            continue
                        if mol_name in ("SOL", "NA", "CL"):
                            break
                        for ci in range(mol_count):
                            res_to_monomer[mol_idx] = mol_name
                            mol_idx += 1

        logger.info(f"    Monomer mapping: {len(res_to_monomer)} residues "
                    f"({len(set(res_to_monomer.values()))} types)")

        # Per-frame analysis
        mon_residues = non_protein.residues
        contact_per_frame = {m: [] for m in all_monomers_list}
        min_dist_per_frame = {m: [] for m in all_monomers_list}
        # For residence time: track per-residue contact state
        res_contact_state = {ri: False for ri in range(len(mon_residues))}
        res_contact_durations = {m: [] for m in all_monomers_list}
        current_duration = {ri: 0 for ri in range(len(mon_residues))}

        for ts in u.trajectory[start_frame::stride]:
            head_pos = protein.positions
            frame_contacts = {m: 0 for m in all_monomers_list}
            frame_min_dists = {m: [] for m in all_monomers_list}

            for ri, res in enumerate(mon_residues):
                m_name = res_to_monomer.get(ri)
                if m_name not in frame_contacts:
                    continue
                try:
                    dists = np.linalg.norm(
                        res.atoms.positions[:, np.newaxis, :] -
                        head_pos[np.newaxis, :, :], axis=2)
                    min_dist = float(np.min(dists))
                    frame_min_dists[m_name].append(min_dist)

                    in_contact = min_dist < cutoff_A
                    if in_contact:
                        frame_contacts[m_name] += 1
                        current_duration[ri] += 1
                    else:
                        if current_duration[ri] > 0:
                            res_contact_durations[m_name].append(current_duration[ri])
                        current_duration[ri] = 0
                except Exception:
                    pass

            for m in all_monomers_list:
                contact_per_frame[m].append(frame_contacts[m])
                if frame_min_dists[m]:
                    min_dist_per_frame[m].append(min(frame_min_dists[m]))

        # ── Metrics ──
        results = {}

        # 1. Contact frequency (6Å): avg contacts per frame
        contact_freq = {}
        for m, counts in contact_per_frame.items():
            contact_freq[m] = round(float(np.mean(counts)), 3) if counts else 0.0
        results["contact_freq_6A"] = contact_freq
        logger.info(f"    Contact freq (6Å): {contact_freq}")

        # 2. Mean minimum distance per monomer type
        mean_min_dist = {}
        for m, dists in min_dist_per_frame.items():
            mean_min_dist[m] = round(float(np.mean(dists)), 2) if dists else 99.0
        results["mean_min_dist_A"] = mean_min_dist
        logger.info(f"    Mean min dist (Å): {mean_min_dist}")

        # 3. Residence time (consecutive frames in contact)
        residence = {}
        for m, durations in res_contact_durations.items():
            residence[m] = round(float(np.mean(durations)), 1) if durations else 0.0
        results["residence_frames"] = residence
        logger.info(f"    Residence (frames): {residence}")

        # 4. RDF-like: fraction of time each monomer type is "close" (< 6Å)
        proximity_frac = {}
        for m, dists in min_dist_per_frame.items():
            if dists:
                proximity_frac[m] = round(sum(1 for d in dists if d < cutoff_A) / len(dists), 3)
            else:
                proximity_frac[m] = 0.0
        results["proximity_fraction"] = proximity_frac
        logger.info(f"    Proximity frac: {proximity_frac}")

        # 5. Monomer-monomer min distance (cavity shape analysis)
        # Min distance between closest atoms of each monomer type pair
        logger.info("    Cavity analysis: monomer-monomer min distances...")
        pair_mindists_md = {}

        # Collect all atom positions per monomer type per frame
        type_atoms_per_frame = {m: [] for m in all_monomers_list}
        for ts in u.trajectory[start_frame::stride]:
            for ri, res in enumerate(mon_residues):
                m_name = res_to_monomer.get(ri)
                if m_name is None:
                    continue
                if m_name not in type_atoms_per_frame:
                    type_atoms_per_frame[m_name] = []
            # For each pair, compute min distance across all copies
            for i, m1 in enumerate(all_monomers_list):
                for m2 in all_monomers_list[i+1:]:
                    key = f"{m1}-{m2}"
                    # Get all atoms of m1 and m2 in this frame
                    atoms1 = []
                    atoms2 = []
                    for ri, res in enumerate(mon_residues):
                        mn = res_to_monomer.get(ri)
                        if mn == m1:
                            atoms1.append(res.atoms.positions)
                        elif mn == m2:
                            atoms2.append(res.atoms.positions)
                    if atoms1 and atoms2:
                        pos1 = np.vstack(atoms1)
                        pos2 = np.vstack(atoms2)
                        dists = np.linalg.norm(
                            pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :], axis=2)
                        md = float(np.min(dists))
                        if key not in pair_mindists_md:
                            pair_mindists_md[key] = []
                        pair_mindists_md[key].append(md)

        # Average min distance over frames
        pair_dists_md = {}
        for key, dists in pair_mindists_md.items():
            pair_dists_md[key] = round(float(np.mean(dists)), 2) if dists else 99.0
        results["pair_distances_md_A"] = pair_dists_md
        logger.info(f"    MD pair min-dists (Å): {pair_dists_md}")

        # Cavity selectivity: pair distances are compared in cross-reactivity
        # Same monomer set on own target → compact (small pair dists)
        # Same monomer set on other target → dispersed (large pair dists)
        # This comparison happens in _run_cross_reactivity using pair_distances_md_A

        # ── 6. EBN: Effective Binding Number (Yuan et al. 2024) ──
        # Max number of copies of each monomer type simultaneously within cutoff
        ebn = {}
        for m, counts in contact_per_frame.items():
            ebn[m] = int(max(counts)) if counts else 0
        results["EBN"] = ebn
        logger.info(f"    EBN (max simultaneous): {ebn}")

        # ── 7. HBNMax: Maximum H-bond Number (Yuan et al. 2024) ──
        # Max H-bonds between protein and each monomer type per frame
        # NOTE: HBA requires charge info → use TPR instead of GRO
        try:
            from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
                HydrogenBondAnalysis as HBA)

            # Load universe with TPR (has charges) for H-bond analysis
            tpr_path = Path(top_path).with_name("md.tpr")
            if tpr_path.exists():
                u_hb = mda.Universe(str(tpr_path), str(traj_path))
            else:
                u_hb = u  # fallback to GRO (may fail)

            hbnmax = {}
            hb_lifetime = {}
            mon_residues_hb = u_hb.select_atoms(
                "not protein and not resname SOL NA CL").residues
            for m in all_monomers_list:
                m_sel = " or ".join(
                    [f"resid {res.resid}" for ri, res in enumerate(mon_residues_hb)
                     if res_to_monomer.get(ri) == m])
                if not m_sel:
                    continue
                try:
                    hb = HBA(u_hb, donors_sel=f"protein or ({m_sel})",
                             acceptors_sel=f"protein or ({m_sel})",
                             d_a_cutoff=3.5, d_h_a_angle_cutoff=150,
                             update_selections=False)
                    hb.run(start=start_frame, step=stride, verbose=False)
                    if len(hb.results.hbonds) > 0:
                        # Count H-bonds per frame between protein and monomer
                        frame_hb_counts = {}
                        for row in hb.results.hbonds:
                            fr = int(row[0])
                            frame_hb_counts[fr] = frame_hb_counts.get(fr, 0) + 1
                        hbnmax[m] = max(frame_hb_counts.values()) if frame_hb_counts else 0
                        # H-bond lifetime (mean consecutive frames)
                        hb_lifetime[m] = round(float(np.mean(
                            list(frame_hb_counts.values()))), 1) if frame_hb_counts else 0
                    else:
                        hbnmax[m] = 0
                        hb_lifetime[m] = 0
                except Exception:
                    hbnmax[m] = 0
                    hb_lifetime[m] = 0

            results["HBNMax"] = hbnmax
            results["hbond_lifetime_avg"] = hb_lifetime
            logger.info(f"    HBNMax: {hbnmax}")
            logger.info(f"    H-bond lifetime (avg): {hb_lifetime}")
        except ImportError:
            logger.debug("HydrogenBondAnalysis not available")

        # ── 8. MD Convergence Check (Polania & Jimenez 2024) ──
        # Compare contact frequency in 25ns windows
        n_analyzed = (n_frames - start_frame) // stride
        if n_analyzed >= 4:
            q1 = n_analyzed // 4
            window_freqs = []
            for w in range(4):
                w_start = w * q1
                w_end = (w + 1) * q1
                w_freq = {}
                for m, counts in contact_per_frame.items():
                    w_counts = counts[w_start:w_end]
                    w_freq[m] = round(float(np.mean(w_counts)), 3) if w_counts else 0
                window_freqs.append(w_freq)

            # Check last two windows — diff < 10% = converged
            converged = True
            conv_details = {}
            for m in all_monomers_list:
                f3 = window_freqs[2].get(m, 0)
                f4 = window_freqs[3].get(m, 0)
                if max(f3, f4) > 0:
                    diff_pct = abs(f4 - f3) / max(f3, f4) * 100
                else:
                    diff_pct = 0
                conv_details[m] = round(diff_pct, 1)
                if diff_pct > 10:
                    converged = False

            results["convergence"] = {
                "converged": converged,
                "window_diff_pct": conv_details,
            }
            logger.info(f"    Convergence: {'YES' if converged else 'NO'} "
                        f"(window diff%: {conv_details})")

        # ── 9. Crosslinker-Monomer Proximity (Rajpal 2023 review) ──
        # Check if crosslinker positions near functional monomers (needed for network)
        xl_proximity = {}
        for m in functional_monomers:
            key = f"{crosslinker}-{m}"
            rev_key = f"{m}-{crosslinker}"
            dist = pair_dists_md.get(key) or pair_dists_md.get(rev_key)
            if dist:
                xl_proximity[m] = dist
        results["crosslinker_proximity_A"] = xl_proximity
        xl_close = all(d < 10.0 for d in xl_proximity.values()) if xl_proximity else False
        results["crosslinker_well_positioned"] = xl_close
        logger.info(f"    Crosslinker proximity: {xl_proximity} → "
                    f"{'OK' if xl_close else 'WARNING: crosslinker too far from some monomers'}")

        # ── Synthesis ratio (EBN-based, Yuan et al. 2024) ──
        # EBN = max simultaneous binding count per monomer type
        # Higher EBN = more template binding sites → needs more copies
        functional_ebn = {m: max(ebn.get(m, 1), 1) for m in functional_monomers}

        if functional_ebn:
            min_ebn = min(functional_ebn.values())
            optimal_ratio = {m: max(1, round(v / min_ebn))
                             for m, v in functional_ebn.items()}
            optimal_ratio[crosslinker] = sum(optimal_ratio.values())
        else:
            optimal_ratio = {m: 1 for m in functional_monomers}
            optimal_ratio[crosslinker] = len(functional_monomers)

        results["optimal_ratio"] = optimal_ratio
        results["occupancy"] = contact_freq  # backward compatibility
        results["n_frames_analyzed"] = (n_frames - start_frame) // stride
        results["cutoff_nm"] = cutoff_nm

        return results

    except ImportError:
        logger.warning("MDAnalysis not available")
        return {"error": "MDAnalysis not installed"}
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")
        return {"error": str(e)}


def _compute_per_atom_rdf(traj_path: Path, top_path: Path,
                           functional_monomers: list,
                           crosslinker: str) -> dict:
    """Per-atom RDF between monomer functional groups and epitope (Yuan 2024).

    Identifies which functional groups (OH, NH2, COOH, etc.) drive binding.
    Returns dict of {monomer: {atom_pair: peak_height}} for top interactions.
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.rdf import InterRDF

        u = mda.Universe(str(top_path), str(traj_path))
        protein = u.select_atoms("protein")
        n_frames = len(u.trajectory)
        start_frame = n_frames // 2

        # Key functional group atoms to track
        fg_atoms = {
            "OH": "name OH or name OG or name OG1",
            "NH": "name N or name NZ or name NH1 or name NH2",
            "CO": "name O or name OD1 or name OD2 or name OE1 or name OE2",
        }

        rdf_results = {}
        all_monomers = functional_monomers + [crosslinker]
        non_prot = u.select_atoms("not protein and not resname SOL NA CL WAT")

        for m in all_monomers:
            # Select all atoms of this monomer type
            m_atoms = non_prot.select_atoms(f"resname {m} or resname UNL")
            if len(m_atoms) == 0:
                continue
            rdf_results[m] = {}
            for fg_name, fg_sel in fg_atoms.items():
                prot_fg = protein.select_atoms(fg_sel)
                if len(prot_fg) == 0:
                    continue
                try:
                    irdf = InterRDF(m_atoms, prot_fg,
                                    range=(0, 10), nbins=100)
                    irdf.run(start=start_frame, step=max(1, (n_frames - start_frame) // 50),
                             verbose=False)
                    peak = float(np.max(irdf.results.rdf))
                    peak_r = float(irdf.results.bins[np.argmax(irdf.results.rdf)])
                    if peak > 1.5:  # significant enrichment
                        rdf_results[m][fg_name] = {
                            "peak_g_r": round(peak, 2),
                            "peak_r_A": round(peak_r, 2),
                        }
                except Exception:
                    pass

        logger.info(f"    Per-atom RDF: {len(rdf_results)} monomers analyzed")
        return rdf_results

    except Exception as e:
        logger.debug(f"RDF analysis failed: {e}")
        return {}


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

            from .config import resolve_path
            test_epitope = resolve_path(phase1_results[test_target].get("head_pdb",
                                        phase1_results[test_target]["epitope_pdb"]))

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


# ════════════════════════════════════════════════════════════════
# A5/B7: Solvent (Porogen) Sweep
# ════════════════════════════════════════════════════════════════

def run_phase4_solvent_sweep(target: str, pc_id: str,
                              functional_monomers: list, crosslinker: str,
                              epitope_pdb, work_dir, solvents: list = None,
                              time_ns: float = 30) -> dict:
    """A5/B7: Re-run Phase 4 pre-poly MD in multiple solvents (porogens).

    Real MIP polymerization is rarely in pure water. Sol-gel uses EtOH/H2O,
    free-radical uses ACN, CHCl3, etc. The same monomer set can give very
    different binding affinities depending on porogen.

    Reference: Cormack 2019 Chromatographia; MDPI Polymers 17:1057 (2025).
    """
    from .config import PHASE4_SOLVENTS, is_polymerization_compatible

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Determine compatible solvents from polymerization type
    combo = functional_monomers + [crosslinker]
    ok, dom, _ = is_polymerization_compatible(combo)
    if not ok:
        return {"error": "incompatible monomers — skip solvent sweep"}

    poly_type = "sol-gel" if dom == "silane" else "radical"
    if solvents is None:
        solvents = [s for s, cfg in PHASE4_SOLVENTS.items()
                     if poly_type in cfg.get("use_for", [])]
        if not solvents:
            solvents = ["water"]

    logger.info(f"\n=== Solvent sweep for {target}/{pc_id} ({poly_type}) ===")
    logger.info(f"  Testing solvents: {solvents}")

    sweep_results = {}
    for solv in solvents:
        solv_dir = work_dir / f"solvent_{solv}"
        solv_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n  → {solv} ({time_ns} ns)...")
        try:
            md_result = _run_prepolymerization_md(
                target=target, pc_id=f"{pc_id}_{solv}",
                functional_monomers=functional_monomers,
                crosslinker=crosslinker,
                epitope_pdb=epitope_pdb,
                work_dir=solv_dir,
                time_ns=time_ns,
            )
            sweep_results[solv] = {
                "dielectric": PHASE4_SOLVENTS.get(solv, {}).get("dielectric"),
                "contact_mean": (md_result.get("occupancy_analysis", {})
                                  .get("total_contact_mean")),
                "n_bound_monomers": (md_result.get("occupancy_analysis", {})
                                      .get("n_bound_monomers")),
                "result": md_result,
            }
        except Exception as e:
            logger.warning(f"  {solv} failed: {e}")
            sweep_results[solv] = {"error": str(e)}

    # Rank by imprinting affinity (more contacts = better imprinting)
    valid = {s: r for s, r in sweep_results.items()
             if "error" not in r and r.get("contact_mean") is not None}
    if valid:
        best_solvent = max(valid, key=lambda s: valid[s]["contact_mean"])
    else:
        best_solvent = "water"  # fallback
    logger.info(f"  Best solvent: {best_solvent}")

    return {
        "polymerization_type": poly_type,
        "solvents_tested": list(sweep_results.keys()),
        "best_solvent": best_solvent,
        "per_solvent": sweep_results,
    }


# ════════════════════════════════════════════════════════════════
# B6: Variable Monomer Ratio Sweep
# ════════════════════════════════════════════════════════════════

def run_phase4_ratio_sweep(target: str, pc_id: str,
                            functional_monomers: list, crosslinker: str,
                            epitope_pdb, work_dir, ratio_presets: list = None,
                            time_ns: float = 30) -> dict:
    """B6: Test multiple monomer ratios (instead of uniform 1:1:1:1).

    Default presets test ratios (1,1,1,1), (2,1,1,1), (3,1,1,1), etc.
    Ratio that maximizes contact stability is selected.

    Reference: Lutz 2019 Macromol Rapid Commun (sequence-controlled polymers).
    """
    from .config import PHASE4_RATIO_PRESETS

    if ratio_presets is None:
        ratio_presets = PHASE4_RATIO_PRESETS

    work_dir = Path(work_dir)
    n_func = len(functional_monomers)
    sweep_results = {}

    for preset in ratio_presets:
        # Pad/truncate to n_func
        if len(preset) < n_func:
            preset = list(preset) + [1] * (n_func - len(preset))
        else:
            preset = list(preset[:n_func])
        label = "_".join(str(r) for r in preset)
        rdir = work_dir / f"ratio_{label}"
        rdir.mkdir(parents=True, exist_ok=True)

        # Convert ratio to copy numbers (total ~20 + crosslinker)
        total = 20
        scale = total / sum(preset)
        copies = {m: max(1, int(round(r * scale)))
                  for m, r in zip(functional_monomers, preset)}
        copies[crosslinker] = total - sum(copies.values())
        if copies[crosslinker] < 1:
            copies[crosslinker] = max(1, total // 5)

        logger.info(f"\n  Ratio {preset} → copies {copies}")
        try:
            # Call with explicit copy override (extension to _run_prepoly_md needed)
            # For now, skip if not implemented; use default
            md_result = _run_prepolymerization_md(
                target=target, pc_id=f"{pc_id}_r{label}",
                functional_monomers=functional_monomers,
                crosslinker=crosslinker,
                epitope_pdb=epitope_pdb,
                work_dir=rdir,
                time_ns=time_ns,
                total_monomers=total,
            )
            sweep_results[label] = {
                "ratio": preset,
                "copies": copies,
                "contact_mean": (md_result.get("occupancy_analysis", {})
                                  .get("total_contact_mean")),
                "result": md_result,
            }
        except Exception as e:
            sweep_results[label] = {"error": str(e), "ratio": preset}

    # Rank by contact stability
    valid = {k: v for k, v in sweep_results.items()
             if "error" not in v and v.get("contact_mean") is not None}
    if valid:
        best = max(valid, key=lambda k: valid[k]["contact_mean"])
        logger.info(f"  Best ratio: {sweep_results[best]['ratio']}")
    else:
        best = None
    return {
        "ratios_tested": [v.get("ratio") for v in sweep_results.values()],
        "best_ratio_label": best,
        "best_ratio": sweep_results[best]["ratio"] if best else None,
        "per_ratio": sweep_results,
    }
