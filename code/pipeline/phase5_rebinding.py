"""
Phase 5: VIP Cavity Formation + Rebinding Validation
=====================================================
Virtually Imprinted Polymer (VIP) approach (Zink & Moura, PCCP 2018):
1. Select equilibrium frames from Phase 4 MD
2. Freeze monomers (position restraint) → approximate polymerization
3. Template removal test → can template escape? (too strong = bad)
4. Rebind own template → validate cavity recognition
5. Rebind other templates → validate selectivity

Reference:
  Zink S et al., Phys. Chem. Chem. Phys. 2018;20:13145-13152
"""

import json
import logging
import shutil
import subprocess
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def _gmx_rmsd(tpr_path: Path, xtc_path: Path, work_dir: Path,
              xvg_name: str = "rmsd_protein.xvg") -> tuple:
    """Calculate protein RMSD using gmx rms (handles PBC correctly).

    Returns (rmsd_mean_second_half_A, rmsd_final_A) or (None, None).
    """
    from .config import GMX_BIN
    xvg = work_dir / xvg_name
    try:
        subprocess.run(
            [GMX_BIN, "rms", "-s", str(tpr_path), "-f", str(xtc_path),
             "-o", str(xvg), "-tu", "ns"],
            input="Protein\nProtein\n", capture_output=True, text=True,
            cwd=str(work_dir), timeout=300,
        )
    except Exception as e:
        logger.warning(f"gmx rms failed: {e}")
        return None, None

    if not xvg.exists():
        return None, None

    # Parse XVG — columns: time(ns) rmsd(nm)
    rmsds = []
    with open(xvg) as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rmsds.append(float(parts[1]) * 10.0)  # nm → Å

    if not rmsds:
        return None, None

    n = len(rmsds)
    # Q4-only (last 25%) for equilibrium analysis — handles slow binding/unbinding kinetics
    last_quarter = rmsds[3 * n // 4:]
    rmsd_mean = round(float(np.mean(last_quarter)), 2) if last_quarter else None
    rmsd_final = round(rmsds[-1], 2)
    # Convergence diagnostic: drift from Q1 to Q4
    q1_mean = float(np.mean(rmsds[:n // 4])) if n >= 4 else 0.0
    drift = round(rmsd_mean - q1_mean, 2) if rmsd_mean else None
    q4_std = round(float(np.std(last_quarter)), 2) if last_quarter else None
    # Attach drift info via tuple expansion via attribute on locals — simplest: log
    if drift is not None and abs(drift) > 1.5:
        logger.warning(f"  Non-converged RMSD: drift Q1→Q4 = {drift:+.2f} Å, Q4 std = {q4_std} Å")
    return rmsd_mean, rmsd_final


def _gmx_hbond(tpr_path: Path, xtc_path: Path, work_dir: Path) -> float:
    """Count template-monomer H-bonds using MDAnalysis HBA (second half avg).

    Uses TPR for charge info (required for hydrogen identification).
    Returns mean H-bond count or None.
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
            HydrogenBondAnalysis as HBA)

        tpr = work_dir / "md.tpr"
        xtc = work_dir / "md.xtc"
        if not (tpr.exists() and xtc.exists()):
            return None

        u = mda.Universe(str(tpr), str(xtc))
        n = len(u.trajectory)
        start = n // 2
        stride = max(1, (n - start) // 50)

        hb = HBA(u,
                 donors_sel="protein",
                 acceptors_sel="not protein and not resname SOL NA CL WAT",
                 d_a_cutoff=3.5, d_h_a_angle_cutoff=150,
                 update_selections=False)
        hb.run(start=start, step=stride, verbose=False)

        # Also check reverse direction (monomer donors → protein acceptors)
        hb2 = HBA(u,
                  donors_sel="not protein and not resname SOL NA CL WAT",
                  acceptors_sel="protein",
                  d_a_cutoff=3.5, d_h_a_angle_cutoff=150,
                  update_selections=False)
        hb2.run(start=start, step=stride, verbose=False)

        # Combine both directions, count per frame
        frame_counts = {}
        for row in list(hb.results.hbonds) + list(hb2.results.hbonds):
            fr = int(row[0])
            frame_counts[fr] = frame_counts.get(fr, 0) + 1

        if not frame_counts:
            return 0.0

        return round(float(np.mean(list(frame_counts.values()))), 1)

    except Exception as e:
        logger.debug(f"H-bond analysis failed: {e}")
        return None


def _contact_count(tpr_path: Path, xtc_path: Path, work_dir: Path,
                   cutoff_A: float = 6.0) -> float:
    """Count template-monomer contacts (< cutoff) using MDAnalysis (second half avg).

    Returns mean contact count or None.
    """
    try:
        import MDAnalysis as mda
    except ImportError:
        return None

    # Prefer TPR (has all info), fallback to GRO
    tpr = work_dir / "md.tpr"
    gro = work_dir / "npt.gro"
    top = str(tpr) if tpr.exists() else str(gro)
    xtc = work_dir / "md.xtc"
    if not xtc.exists():
        return None

    try:
        u = mda.Universe(top, str(xtc))
        protein = u.select_atoms("protein")
        monomers = u.select_atoms("not protein and not resname SOL NA CL WAT")
        if len(protein) == 0 or len(monomers) == 0:
            return None

        n = len(u.trajectory)
        start = n // 2
        stride = max(1, (n - start) // 50)
        counts = []
        cutoff_nm = cutoff_A  # MDAnalysis uses Å

        from MDAnalysis.lib.distances import distance_array
        for ts in u.trajectory[start::stride]:
            # Count monomer atoms within cutoff of any protein atom
            dists = distance_array(protein.positions, monomers.positions)
            n_within = int(np.sum(np.min(dists, axis=0) < cutoff_A))
            counts.append(n_within)

        return round(float(np.mean(counts)), 1) if counts else None
    except Exception as e:
        logger.debug(f"Contact count failed: {e}")
        return None


def run_phase6(phase4_results: dict = None,
               phase1_results: dict = None,
               target_names: list = None,
               output_dir: str = None) -> dict:
    """
    Phase 6: VIP cavity rebinding validation.

    For each target's top PC:
    1. Select top N contact frames from Phase 4 trajectory
    2. For each frame: freeze → remove template → rebind → analyze
    3. Test selectivity with other targets' heads
    """
    from .config import (TARGETS, REBINDING_MD_NS, REBINDING_N_SNAPSHOTS,
                         REBINDING_RMSD_THRESHOLD, REBINDING_TRIAL_MODE,
                         get_output_path, resolve_path)

    # Trial mode: 1 snapshot per target, 30 ns rebinding MD
    # Used to validate method works (ECL2 discrimination) before full 10-snapshot run
    if REBINDING_TRIAL_MODE:
        REBINDING_N_SNAPSHOTS = 1
        REBINDING_MD_NS = 30
        logger.warning("REBINDING_TRIAL_MODE active: 1 snapshot × 30 ns per target")

    if output_dir is None:
        output_dir = str(get_output_path("phase6"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 4 results
    if phase4_results is None:
        p4_path = get_output_path("phase4") / "phase4_md_results.json"
        if p4_path.exists():
            with open(p4_path) as f:
                phase4_results = json.load(f)
        else:
            logger.error("Phase 4 results not found")
            return {}

    if phase1_results is None:
        p1_path = get_output_path("phase1") / "phase1_results.json"
        with open(p1_path) as f:
            phase1_results = json.load(f)

    if target_names is None:
        target_names = [t for t in phase4_results if t != "cross_reactivity"]

    # Per-target resume: load partial results if previous run was interrupted
    partial_json = output_dir / "phase5_rebinding_results.json"
    results = {}
    if partial_json.exists():
        try:
            with open(partial_json) as f:
                results = json.load(f)
            done_targets = [t for t in results
                            if isinstance(results.get(t), dict) and results[t]]
            if done_targets:
                logger.info(f"Resume: loaded existing Phase 5 results for "
                            f"{done_targets} — skipping these targets")
        except Exception as e:
            logger.warning(f"Could not parse existing phase5_rebinding_results.json: {e}")
            results = {}

    for target in target_names:
        # Skip if already completed in a prior run
        if isinstance(results.get(target), dict) and results[target]:
            logger.info(f"\n{'='*20} Phase 5: {target} (RESUMED — already done) {'='*20}")
            continue

        p4 = phase4_results.get(target, {})
        if not p4:
            continue

        # Get top PC
        best_pc_id = next(iter(p4), None)
        if not best_pc_id:
            continue

        pc_data = p4[best_pc_id]
        # Template for rebinding: use the same template used in Phase 4.
        # In whole-protein imprinting mode (PHASE4_TEMPLATE_MODE="ecl2"),
        # rebind the entire ECL2 (~90 residues) rather than just the 16-mer head.
        from .config import PHASE4_TEMPLATE_MODE
        if PHASE4_TEMPLATE_MODE == "ecl2":
            head_pdb = resolve_path(phase1_results[target].get("ecl2_pdb",
                                    phase1_results[target].get("head_pdb",
                                    phase1_results[target]["epitope_pdb"])))
            logger.info(f"  Whole-ECL2 rebinding mode (template: {head_pdb.name})")
        else:
            head_pdb = resolve_path(phase1_results[target].get("head_pdb",
                                    phase1_results[target]["epitope_pdb"]))

        # Phase 4 MD directory
        p4_md_dir = get_output_path("phase4") / target / best_pc_id / "md"
        traj = p4_md_dir / "md_reduced.xtc"
        if not traj.exists():
            traj = p4_md_dir / "md.xtc"
        top = p4_md_dir / "npt.gro"

        if not traj.exists() or not top.exists():
            logger.warning(f"[{target}] Phase 4 trajectory not found")
            continue

        logger.info(f"\n{'='*20} Phase 6 Rebinding: {target}/{best_pc_id} {'='*20}")

        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Select evenly spaced equilibrium frames
        logger.info(f"  Step 1: Selecting {REBINDING_N_SNAPSHOTS} equilibrium frames...")
        top_frames = _select_equilibrium_frames(
            traj, top, p4_md_dir / "topol.top",
            n_frames=REBINDING_N_SNAPSHOTS)

        if not top_frames:
            logger.warning(f"  No suitable frames found")
            continue

        logger.info(f"  Selected frames: {[f['frame_idx'] for f in top_frames]}")
        logger.info(f"  Contact counts: {[f['total_contacts'] for f in top_frames]}")

        # Steps 2-4: For each snapshot
        snapshot_results = []
        for si, frame_info in enumerate(top_frames):
            snap_dir = target_dir / f"snapshot_{si}"
            snap_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\n  --- Snapshot {si+1}/{len(top_frames)} "
                        f"(frame {frame_info['frame_idx']}, "
                        f"contacts={frame_info['total_contacts']}) ---")

            # Step 2: Extract frame + freeze monomers
            logger.info(f"  Step 2: Extracting frame and freezing monomers...")
            cavity_result = _create_cavity(
                traj, top, p4_md_dir / "topol.top",
                frame_info['frame_idx'],
                snap_dir)

            if not cavity_result.get("success"):
                logger.warning(f"  Cavity creation failed: {cavity_result.get('error')}")
                snapshot_results.append({"success": False})
                continue

            # Step 3: Template removal test — can template escape?
            logger.info(f"  Step 3: Template removal test...")
            removal_result = _run_template_removal_md(
                cavity_result["cavity_gro"],
                cavity_result["cavity_top"],
                snap_dir / "removal_test",
                time_ns=min(REBINDING_MD_NS, 10),  # shorter test
                p4_md_dir=p4_md_dir)

            snap_result_removal = removal_result

            # Step 4: Rebind own template
            logger.info(f"  Step 4: Rebinding {target} head...")
            rebind_own = _run_rebinding_md(
                cavity_result["cavity_gro"],
                cavity_result["cavity_top"],
                head_pdb,
                snap_dir / "rebind_own",
                time_ns=REBINDING_MD_NS,
                p4_md_dir=p4_md_dir,
                is_own_target=True)

            snap_result = {
                "frame_idx": frame_info["frame_idx"],
                "total_contacts": frame_info["total_contacts"],
                "removal_test": snap_result_removal,
                "rebind_own": rebind_own,
            }

            # B8: Multi-pose rebinding for ensemble averaging (only first snap to save compute)
            from .config import (REBINDING_MULTI_POSE,
                                  REBINDING_N_HEAD_CONFORMERS)
            if REBINDING_MULTI_POSE and si == 0:
                phase1_target = phase1_results.get(target, {})
                conformer_files = phase1_target.get("conformer_pdbs", [])
                if conformer_files and len(conformer_files) >= 2:
                    head_confs = conformer_files[:REBINDING_N_HEAD_CONFORMERS]
                    logger.info(f"  B8: Multi-pose rebinding "
                                f"({len(head_confs)} head conformers × 1 rep)...")
                    try:
                        mp_result = run_multipose_rebinding(
                            target, cavity_result["cavity_gro"],
                            cavity_result["cavity_top"],
                            head_confs, n_replicates=1,
                            time_ns=min(REBINDING_MD_NS, 20),
                            work_dir=snap_dir / "multipose",
                            p4_md_dir=p4_md_dir)
                        snap_result["multipose_ensemble"] = mp_result
                    except Exception as e:
                        logger.warning(f"  B8 multi-pose failed: {e}")

            # Step 5: Rebind other targets' heads (selectivity)
            for other_target in target_names:
                if other_target == target:
                    continue
                # Cross-rebinding template: match Phase 4 mode (ECL2 or head)
                if PHASE4_TEMPLATE_MODE == "ecl2":
                    other_head = resolve_path(phase1_results[other_target].get(
                        "ecl2_pdb", phase1_results[other_target].get(
                            "head_pdb", phase1_results[other_target]["epitope_pdb"])))
                else:
                    other_head = resolve_path(phase1_results[other_target].get(
                        "head_pdb", phase1_results[other_target]["epitope_pdb"]))

                logger.info(f"  Step 5: Rebinding {other_target} head (selectivity)...")
                rebind_other = _run_rebinding_md(
                    cavity_result["cavity_gro"],
                    cavity_result["cavity_top"],
                    other_head,
                    snap_dir / f"rebind_{other_target}",
                    time_ns=REBINDING_MD_NS,
                    p4_md_dir=p4_md_dir,
                    is_own_target=False)

                snap_result[f"rebind_{other_target}"] = rebind_other

            snap_result["success"] = True
            snapshot_results.append(snap_result)

        # Step 6: Analyze results
        results[target] = _analyze_rebinding_results(
            target, target_names, snapshot_results, REBINDING_RMSD_THRESHOLD)

        # Step 7: Auto dual-imprinting if weak selectivity + has N-glycan
        target_result = results[target]
        n_glycan = phase1_results[target].get("properties", {}).get(
            "n_glycan_sites_known", 0)
        sel = target_result.get("selectivity", {})
        # Dual-imprinting criteria:
        # 1. Any cross-target SI < 1.5 AND p > 0.05 (not statistically selective)
        # 2. N-glycan ≥ 1 (APBA boronate-diol target exists) [Teixeira 2021]
        # 3. Rebinding ≥ 3/5 (cavity works; if <3, monomer combo itself is the issue)
        any_not_significant = any(
            s.get("selectivity_label") in ("weak", "cross-reactive")
            and (s.get("p_value") is None or s.get("p_value") > 0.05)
            for s in sel.values())
        n_rebound = target_result.get("n_rebound", 0)
        # Adaptive rebound threshold: ≥3 in full run (10 snaps), ≥1 in trial (1 snap)
        rebound_threshold = max(1, REBINDING_N_SNAPSHOTS // 3)

        if any_not_significant and n_glycan > 0 and n_rebound >= rebound_threshold:
            logger.info(f"\n  *** Dual-imprinting triggered for {target} ***")
            logger.info(f"      Reason: non-significant selectivity (SI<1.5, p>0.05) "
                        f"+ {n_glycan} N-glycan sites + rebinding {n_rebound}/{REBINDING_N_SNAPSHOTS}")
            logger.info(f"      Action: adding APBA (boronic acid) to cavity for glycan recognition")

            dual_results = _run_dual_imprinting_vip(
                target, target_names, snapshot_results,
                phase1_results, p4_md_dir, output_dir / target,
                n_glycan=n_glycan,
            )
            target_result["dual_imprinting"] = dual_results
            target_result["dual_imprinting_reason"] = (
                f"SI weak + {n_glycan} N-glycan sites → APBA layer 2")
        elif any_not_significant and n_glycan == 0:
            target_result["dual_imprinting"] = None
            target_result["dual_imprinting_reason"] = (
                "Weak selectivity but no N-glycan sites — dual-imprinting not applicable")
            logger.info(f"  {target}: weak selectivity but no N-glycan → "
                        f"dual-imprinting not applicable")

        # Per-target incremental save — survives crashes
        try:
            with open(output_dir / "phase5_rebinding_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"  Phase 5: {target} complete → saved partial results")
        except Exception as e:
            logger.warning(f"  Failed to save partial Phase 5 results: {e}")

    # Final save
    with open(output_dir / "phase5_rebinding_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    _print_phase6_summary(results)
    return results


# ── Frame Selection ──────────────────────────────────────────

def _select_equilibrium_frames(traj_path, top_path, topol_path,
                                n_frames=5, cutoff_A=6.0):
    """Select evenly spaced frames from equilibrated (last 50%) trajectory.
    No cherry-picking — represents random polymerization timing.
    Also reports contact count per frame for reference."""
    try:
        import MDAnalysis as mda

        u = mda.Universe(str(top_path), str(traj_path))
        protein = u.select_atoms("protein")
        non_protein = u.select_atoms("not protein and not resname SOL NA CL")

        if len(protein) == 0 or len(non_protein) == 0:
            return []

        # Last 50% of trajectory, evenly spaced
        n_total = len(u.trajectory)
        start = n_total // 2
        interval = (n_total - start) // (n_frames + 1)

        selected = []
        for i in range(1, n_frames + 1):
            frame_idx = start + i * interval
            if frame_idx >= n_total:
                frame_idx = n_total - 1

            u.trajectory[frame_idx]
            head_pos = protein.positions

            # Count contacts for reference (not selection criterion)
            total = 0
            for res in non_protein.residues:
                try:
                    min_dist = np.min(np.linalg.norm(
                        res.atoms.positions[:, np.newaxis, :] -
                        head_pos[np.newaxis, :, :], axis=2))
                    if min_dist < cutoff_A:
                        total += 1
                except Exception:
                    pass

            selected.append({
                "frame_idx": frame_idx,
                "total_contacts": total,
            })

        return selected

    except Exception as e:
        logger.warning(f"Frame selection failed: {e}")
        return []


# ── Cavity Creation ──────────────────────────────────────────

def _create_cavity(traj_path, top_path, topol_path, frame_idx, output_dir):
    """
    Extract specific frame → create position restraints for monomers.
    Keep full system (protein + monomers + water) — topology unchanged.
    For rebinding: replace protein coordinates with new template.
    """
    from .config import (REBINDING_RESTRAINT_K,
                          REBINDING_CROSSLINKER_RESTRAINT_K,
                          CROSSLINKER_LIBRARY)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import MDAnalysis as mda
        from .utils_gromacs import _gmx
        from .config import GMX_BIN
        import subprocess

        # Step 1: Use gmx trjconv to extract centered, PBC-corrected frame
        # -pbc mol: keep molecules whole (no bond crossing PBC)
        # -center: center protein in box
        # This fixes "protein at edge" issue from Phase 4 PBC split
        frame_gro = output_dir / "frame.gro"
        frame_time_ps = frame_idx * 10  # frames are at 10 ps intervals
        try:
            # Need a tpr; use the topology if it's a tpr, else regenerate
            top_for_trjconv = str(top_path) if str(top_path).endswith('.tpr') else None
            if top_for_trjconv is None:
                # Find tpr in same dir as topol.top
                tpr_candidate = Path(topol_path).parent / "md.tpr"
                if tpr_candidate.exists():
                    top_for_trjconv = str(tpr_candidate)
            if top_for_trjconv:
                # Build index group for centering: use Protein
                # gmx trjconv: select Protein (1) for centering, System (0) for output
                cmd = [GMX_BIN, "trjconv",
                       "-f", str(traj_path),
                       "-s", top_for_trjconv,
                       "-o", str(frame_gro),
                       "-pbc", "mol",
                       "-center",
                       "-dump", str(frame_time_ps)]
                proc = subprocess.run(cmd, input="1\n0\n", capture_output=True,
                                      text=True, timeout=300)
                if not frame_gro.exists():
                    raise RuntimeError(f"trjconv failed: {proc.stderr[-300:]}")
                logger.info(f"    Frame {frame_idx} centered via gmx trjconv (-pbc mol -center)")
                # Load centered frame for downstream
                u = mda.Universe(str(frame_gro))
            else:
                raise RuntimeError("No tpr available for trjconv")
        except Exception as ce:
            logger.warning(f"    trjconv centering failed ({ce}); falling back to MDAnalysis")
            u = mda.Universe(str(top_path), str(traj_path))
            u.trajectory[frame_idx]
            # Manual centering via MDAnalysis transformations
            try:
                from MDAnalysis.transformations import unwrap, center_in_box, wrap
                prot = u.select_atoms("protein")
                workflow = [unwrap(u.atoms), center_in_box(prot, center='geometry'),
                            wrap(u.atoms, compound='residues')]
                u.trajectory.add_transformations(*workflow)
            except Exception as te:
                logger.warning(f"    MDA centering also failed: {te}")
            with mda.Writer(str(frame_gro), n_atoms=u.atoms.n_atoms) as w:
                w.write(u.atoms)

        # Create position restraints by modifying each monomer ITP
        # GROMACS requires [ position_restraints ] inside each [ moleculetype ]
        # We add it to each monomer ITP file, guarded by #ifdef POSRES_MONOMER
        topol_src = Path(topol_path)
        cavity_top = output_dir / "topol.top"
        shutil.copy2(str(topol_src), str(cavity_top))

        # Find and modify monomer ITP files
        p4_md = topol_src.parent
        for itp_file in p4_md.glob("*.itp"):
            if itp_file.name in ("posre.itp", "posre_monomers.itp"):
                continue
            content = itp_file.read_text()
            # Skip if already has position_restraints or is not a monomer ITP
            if "position_restraints" in content:
                shutil.copy2(str(itp_file), str(output_dir / itp_file.name))
                continue
            if "[ moleculetype ]" not in content:
                shutil.copy2(str(itp_file), str(output_dir / itp_file.name))
                continue

            # Count atoms in this moleculetype
            n_atoms = 0
            in_atoms = False
            for line in content.split("\n"):
                if "[ atoms ]" in line:
                    in_atoms = True
                    continue
                if in_atoms and line.strip().startswith("["):
                    break
                if in_atoms and line.strip() and not line.strip().startswith(";"):
                    n_atoms += 1

            # Two-tier restraint (Yuan 2024, adenovirus eIP protocol):
            #  - Crosslinker = irreversible C-C network → stiff (k=5000)
            #  - Functional monomer = non-covalent anchor → moderate (k=1000)
            mol_name = itp_file.stem  # e.g. "TTMS", "DVB"
            is_crosslinker = mol_name in CROSSLINKER_LIBRARY
            k = (REBINDING_CROSSLINKER_RESTRAINT_K if is_crosslinker
                 else REBINDING_RESTRAINT_K)
            tier_label = "CROSSLINKER stiff" if is_crosslinker else "functional moderate"

            posre_block = "\n#ifdef POSRES_MONOMER\n[ position_restraints ]\n"
            posre_block += f"; {tier_label} (k = {k} kJ/mol/nm^2)\n"
            posre_block += "; ai  funct  fcx    fcy    fcz\n"
            in_atoms = False
            atom_idx = 0
            for line in content.split("\n"):
                if "[ atoms ]" in line:
                    in_atoms = True
                    continue
                if in_atoms and line.strip().startswith("["):
                    break
                if in_atoms and line.strip() and not line.strip().startswith(";"):
                    atom_idx += 1
                    parts = line.split()
                    # Check mass (column 8 in GROMACS ITP)
                    try:
                        mass = float(parts[7]) if len(parts) > 7 else 12.0
                    except (ValueError, IndexError):
                        mass = 12.0
                    if mass > 2.0:  # heavy atoms only
                        posre_block += (f"  {atom_idx}    1  {k}  {k}  {k}\n")
            posre_block += "#endif\n"

            # Append to ITP content
            modified = content.rstrip() + "\n" + posre_block
            (output_dir / itp_file.name).write_text(modified)
            logger.debug(f"    Added position restraints to {itp_file.name} ({atom_idx} atoms)")

        # Copy posre.itp for protein (if exists)
        posre_protein = p4_md / "posre.itp"
        if posre_protein.exists():
            shutil.copy2(str(posre_protein), str(output_dir / "posre.itp"))

        protein_n = u.select_atoms("protein").n_atoms
        logger.info(f"    Cavity created: {u.atoms.n_atoms} atoms "
                    f"(template {protein_n} atoms kept for rebinding)")

        return {
            "success": True,
            "cavity_gro": frame_gro,  # full system with protein
            "cavity_top": cavity_top,
            "frame_gro": frame_gro,
        }

    except Exception as e:
        logger.error(f"Cavity creation failed: {e}")
        return {"success": False, "error": str(e)}


# ── Rebinding MD ─────────────────────────────────────────────

def _run_rebinding_md(cavity_gro, cavity_top, template_pdb,
                       output_dir, time_ns=20, p4_md_dir=None,
                       is_own_target=True):
    """
    Place template near cavity center, run MD with monomers restrained.
    Analyze if template stays in cavity (RMSD).
    """
    from .utils_gromacs import (_gmx, run_full_md_pipeline)
    from .config import (MD_TEMPERATURE_K, MD_PRESSURE_BAR, MD_GPU_ID,
                         REBINDING_RMSD_THRESHOLD, GMX_BIN)

    output_dir = Path(output_dir)
    md_dir = output_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    try:
        import MDAnalysis as mda

        # Copy full system as starting point
        shutil.copy2(str(cavity_gro), str(md_dir / "rebind_system.gro"))

        # Copy topology and all ITP files
        shutil.copy2(str(cavity_top), str(md_dir / "topol.top"))
        for src_dir in [Path(cavity_top).parent]:
            for itp in src_dir.glob("*.itp"):
                dst = md_dir / itp.name
                if not dst.exists():
                    shutil.copy2(str(itp), str(dst))
        if p4_md_dir:
            for itp in Path(p4_md_dir).glob("*.itp"):
                dst = md_dir / itp.name
                if not dst.exists():
                    shutil.copy2(str(itp), str(dst))

        if is_own_target:
            # Own rebinding: use system as-is (protein already correct)
            logger.info(f"    Own rebinding: using Phase 4 frame directly")
        else:
            # Selectivity: copy cavity topology, replace protein with different head
            logger.info(f"    Selectivity rebinding: rebuilding with different head...")
            try:
                from .utils_gromacs import setup_protein_topology

                u_sys = mda.Universe(str(cavity_gro))

                # 1. Generate new protein GRO + posre from different head
                setup_protein_topology(Path(template_pdb), md_dir)
                new_prot_gro = md_dir / "protein.gro"

                # 1b. Align new head to old head's center of mass
                old_protein = u_sys.select_atoms("protein")
                old_com = old_protein.center_of_mass()

                u_new = mda.Universe(str(new_prot_gro))
                new_protein = u_new.select_atoms("all")
                new_com = new_protein.center_of_mass()
                shift = old_com - new_com
                new_protein.translate(shift)
                with mda.Writer(str(new_prot_gro), n_atoms=new_protein.n_atoms) as w:
                    w.write(new_protein)
                logger.info(f"    Aligned new head to old COM (shift={np.linalg.norm(shift):.1f} Å)")

                # 2. Remove old protein + nearby water that would clash
                # Remove water within 3Å of old protein position to make room
                old_prot_near_water = u_sys.select_atoms(
                    "resname SOL and around 3.0 protein")
                # Get residue IDs to remove whole water molecules
                remove_resids = set(old_prot_near_water.residues.resids)
                keep = u_sys.select_atoms(
                    f"not protein and not (resname SOL and resid {' '.join(str(r) for r in remove_resids)})")
                n_removed = len(old_prot_near_water.residues)

                monomers_gro = md_dir / "monomers_only.gro"
                with mda.Writer(str(monomers_gro), n_atoms=keep.n_atoms) as w:
                    w.write(keep)
                logger.info(f"    Removed {n_removed} waters near old protein position")

                # 3. Merge new protein + cleaned monomers/water/ions
                prot_lines = new_prot_gro.read_text().strip().split("\n")
                mon_lines = monomers_gro.read_text().strip().split("\n")
                prot_natoms = int(prot_lines[1].strip())
                mon_natoms = int(mon_lines[1].strip())
                total = prot_natoms + mon_natoms

                merged = [prot_lines[0]]
                merged.append(f" {total}")
                merged.extend(prot_lines[2:2+prot_natoms])
                merged.extend(mon_lines[2:2+mon_natoms])
                merged.append(mon_lines[-1])  # Use cavity box, not protein box
                rebind_gro = md_dir / "rebind_system.gro"
                rebind_gro.write_text("\n".join(merged) + "\n")

                # 4. Build topology: copy cavity topology, replace protein section
                cavity_top_text = Path(cavity_top).read_text()

                # Extract new protein [ moleculetype ] block from pdb2gmx topology
                # Include everything up to and including the #endif after posre.itp
                new_pdb2gmx_top = (md_dir / "topol.top").read_text()
                mt_start = new_pdb2gmx_top.find("[ moleculetype ]")
                # Find the #endif that closes the POSRES block (after posre.itp)
                posre_marker = '#include "posre.itp"'
                posre_idx = new_pdb2gmx_top.find(posre_marker, mt_start)
                if posre_idx >= 0:
                    endif_idx = new_pdb2gmx_top.find("#endif", posre_idx)
                    if endif_idx >= 0:
                        new_prot_end = endif_idx + len("#endif")
                    else:
                        new_prot_end = posre_idx + len(posre_marker)
                else:
                    water_idx = new_pdb2gmx_top.find('#include "amber99sb-ildn.ff/tip3p.itp"', mt_start)
                    new_prot_end = water_idx if water_idx >= 0 else new_pdb2gmx_top.find("[ system ]")

                new_protein_block = new_pdb2gmx_top[mt_start:new_prot_end].rstrip() + "\n\n"

                # In cavity topology, find protein section boundaries
                # Start: [ moleculetype ]
                cav_mt_start = cavity_top_text.find("[ moleculetype ]")
                # End: just before #include "amber99sb-ildn.ff/tip3p.itp"
                water_marker = '#include "amber99sb-ildn.ff/tip3p.itp"'
                cav_water_idx = cavity_top_text.find(water_marker)

                if cav_mt_start >= 0 and cav_water_idx > cav_mt_start:
                    final_top = (cavity_top_text[:cav_mt_start]
                                 + new_protein_block
                                 + cavity_top_text[cav_water_idx:])
                else:
                    final_top = cavity_top_text

                # Update SOL count in topology (we removed some waters)
                if n_removed > 0:
                    import re
                    final_top = re.sub(
                        r'(SOL\s+)(\d+)',
                        lambda m: f"{m.group(1)}{int(m.group(2)) - n_removed}",
                        final_top, count=1)

                (md_dir / "topol.top").write_text(final_top)

                # 5. Copy ITP files
                for src_dir in [Path(cavity_top).parent]:
                    for itp in src_dir.glob("*.itp"):
                        dst = md_dir / itp.name
                        if not dst.exists():
                            shutil.copy2(str(itp), str(dst))
                if p4_md_dir:
                    for itp in Path(p4_md_dir).glob("*.itp"):
                        dst = md_dir / itp.name
                        if not dst.exists():
                            shutil.copy2(str(itp), str(dst))

                # Use rebuilt system as ionized.gro
                ionized = md_dir / "ionized.gro"
                if not ionized.exists():
                    shutil.copy2(str(rebind_gro), str(ionized))

                logger.info(f"    Rebuilt system: {total} atoms (new head + monomers)")

            except Exception as e:
                logger.error(f"    Selectivity rebuild failed: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "time_ns": time_ns,
                    "rmsd_mean_A": None,
                    "rmsd_final_A": None,
                    "rebound": None,
                    "note": f"rebuild failed: {e}",
                }

        # System already has water + ions → copy as ionized.gro
        from .utils_gromacs import (run_energy_minimization,
                                     run_nvt_equilibration, run_npt_equilibration,
                                     run_production_md)

        # ── Steric Complementarity check (size-exclusion selectivity) ──
        # For cross-target rebinding: if the protein is too large for the cavity,
        # severe clashes mean it's physically REJECTED (selective). Detect this
        # BEFORE EM (which would explode with Max-force=inf), record as
        # size-excluded, and skip the doomed MD.
        if not is_own_target:
            from .config import (REBINDING_CLASH_CUTOFF_A,
                                 REBINDING_CLASH_THRESHOLD)
            from .utils_analysis import compute_steric_clash
            try:
                clash = compute_steric_clash(
                    md_dir / "rebind_system.gro",
                    clash_cutoff_A=REBINDING_CLASH_CUTOFF_A)
                logger.info(f"    Steric clash: {clash['clash_count']} "
                            f"({clash['clash_per_residue']}/residue)")
                if clash["clash_count"] > REBINDING_CLASH_THRESHOLD:
                    logger.info(f"    → SIZE-EXCLUDED: cross-protein too large "
                                f"for cavity ({clash['clash_count']} clashes "
                                f"> {REBINDING_CLASH_THRESHOLD}) — REJECTED "
                                f"(size-exclusion selectivity ✓)")
                    return {
                        "time_ns": 0,
                        "rmsd_mean_A": None,
                        "rmsd_final_A": None,
                        "rebound": False,
                        "size_excluded": True,
                        "steric_clash": clash,
                        "hbond_mean": 0.0,
                        "contact_mean": 0.0,
                        "mmpbsa_dG": None,
                        "status": "SIZE_EXCLUDED",
                    }
            except Exception as e:
                logger.warning(f"    Steric clash check failed: {e}, proceeding to MD")

        ionized = md_dir / "ionized.gro"
        if not ionized.exists():
            shutil.copy2(str(md_dir / "rebind_system.gro"), str(ionized))

        if not (md_dir / "em.gro").exists():
            logger.info(f"    Energy minimization...")
            run_energy_minimization(md_dir)

        posres_define = "define = -DPOSRES_MONOMER"

        if not (md_dir / "nvt.gro").exists():
            logger.info(f"    NVT equilibration (monomers restrained)...")
            run_nvt_equilibration(md_dir, time_ps=100.0,
                                  define=posres_define,
                                  temperature=MD_TEMPERATURE_K)

        if not (md_dir / "npt.gro").exists():
            logger.info(f"    NPT equilibration (monomers restrained)...")
            run_npt_equilibration(md_dir, time_ps=100.0,
                                  define=posres_define,
                                  temperature=MD_TEMPERATURE_K,
                                  pressure=MD_PRESSURE_BAR)

        if not (md_dir / "md.gro").exists():
            logger.info(f"    Production MD ({time_ns}ns, monomers restrained)...")
            run_production_md(md_dir, time_ns=time_ns,
                              define=posres_define,
                              temperature=MD_TEMPERATURE_K,
                              pressure=MD_PRESSURE_BAR,
                              gpu_id=MD_GPU_ID)
        else:
            logger.info(f"    Production MD: FOUND")

        # 7. Analyze template RMSD using gmx rms (PBC-safe)
        xtc = md_dir / "md.xtc"
        tpr = md_dir / "md.tpr"

        rmsd_mean, rmsd_final = _gmx_rmsd(tpr, xtc, md_dir, "rmsd_rebind.xvg")

        # H-bond and contact analysis
        hbond_mean = _gmx_hbond(tpr, xtc, md_dir)
        contact_mean = _contact_count(tpr, xtc, md_dir)

        # MM-GBSA binding energy (Kumar et al. 2024)
        mmpbsa_dG = None
        try:
            from .utils_gromacs import run_mmpbsa
            mmpbsa_result = run_mmpbsa(
                md_dir, start_ns=time_ns * 0.5, end_ns=time_ns, n_frames=50)
            if "delta_total_kcal" in mmpbsa_result:
                mmpbsa_dG = mmpbsa_result.get("delta_total_kcal")
                if mmpbsa_dG is not None:
                    mmpbsa_dG = round(float(mmpbsa_dG), 2)
            logger.info(f"    MM-GBSA ΔG: {mmpbsa_dG} kcal/mol")
        except Exception as e:
            logger.debug(f"    MM-GBSA skipped: {e}")

        rebound = rmsd_mean < REBINDING_RMSD_THRESHOLD if rmsd_mean else None
        status = "REBOUND" if rebound else ("ESCAPED" if rebound is False else "N/A")

        hb_str = f", H-bonds={hbond_mean}" if hbond_mean is not None else ""
        ct_str = f", contacts={contact_mean}" if contact_mean is not None else ""
        dg_str = f", ΔG={mmpbsa_dG}" if mmpbsa_dG is not None else ""
        logger.info(f"    Result: RMSD={rmsd_mean} Å → {status}{hb_str}{ct_str}{dg_str}")

        return {
            "time_ns": time_ns,
            "rmsd_mean_A": rmsd_mean,
            "rmsd_final_A": rmsd_final,
            "rebound": rebound,
            "hbond_mean": hbond_mean,
            "contact_mean": contact_mean,
            "mmpbsa_dG": mmpbsa_dG,
        }

    except Exception as e:
        logger.error(f"Rebinding MD failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ── Results Analysis ─────────────────────────────────────────

# ── Template Removal Test ─────────────────────────────────────

def _run_template_removal_md(cavity_gro, cavity_top, output_dir,
                              time_ns=10, p4_md_dir=None):
    """
    Test if template can escape from cavity (monomers restrained, template free).

    If template RMSD increases significantly → template can be removed → good MIP.
    If template stays put (RMSD stable) → binding too strong → template removal difficult.

    Optimal: template escapes within 10ns → moderate binding → good IF.
    """
    output_dir = Path(output_dir)
    md_dir = output_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    try:
        import MDAnalysis as mda
        from .utils_gromacs import (run_energy_minimization,
                                     run_nvt_equilibration, run_npt_equilibration,
                                     run_production_md)
        from .config import MD_TEMPERATURE_K, MD_PRESSURE_BAR, MD_GPU_ID, REBINDING_RMSD_THRESHOLD

        # System as-is: monomers restrained, template + water free
        # Template should drift away if binding is moderate
        shutil.copy2(str(cavity_gro), str(md_dir / "rebind_system.gro"))
        shutil.copy2(str(cavity_top), str(md_dir / "topol.top"))

        # Copy ITP files
        for src_dir in [Path(cavity_top).parent]:
            for itp in src_dir.glob("*.itp"):
                dst = md_dir / itp.name
                if not dst.exists():
                    shutil.copy2(str(itp), str(dst))
        if p4_md_dir:
            for itp in Path(p4_md_dir).glob("*.itp"):
                dst = md_dir / itp.name
                if not dst.exists():
                    shutil.copy2(str(itp), str(dst))

        ionized = md_dir / "ionized.gro"
        if not ionized.exists():
            shutil.copy2(str(md_dir / "rebind_system.gro"), str(ionized))

        # Run MD with monomer restraints
        posres_define = "define = -DPOSRES_MONOMER"

        if not (md_dir / "em.gro").exists():
            logger.info(f"    Removal test: EM...")
            run_energy_minimization(md_dir)

        if not (md_dir / "nvt.gro").exists():
            logger.info(f"    Removal test: NVT (monomers restrained)...")
            run_nvt_equilibration(md_dir, time_ps=100.0, define=posres_define,
                                  temperature=MD_TEMPERATURE_K)

        if not (md_dir / "npt.gro").exists():
            logger.info(f"    Removal test: NPT (monomers restrained)...")
            run_npt_equilibration(md_dir, time_ps=100.0, define=posres_define,
                                  temperature=MD_TEMPERATURE_K, pressure=MD_PRESSURE_BAR)

        if not (md_dir / "md.gro").exists():
            logger.info(f"    Removal test: {time_ns}ns MD (monomers restrained)...")
            run_production_md(md_dir, time_ns=time_ns, define=posres_define,
                              temperature=MD_TEMPERATURE_K,
                              pressure=MD_PRESSURE_BAR, gpu_id=MD_GPU_ID)

        # Analyze: template RMSD over time using gmx rms (PBC-safe)
        xtc = md_dir / "md.xtc"
        tpr = md_dir / "md.tpr"

        rmsd_start = None
        rmsd_end = None
        escaped = None

        xvg = md_dir / "rmsd_removal.xvg"
        from .config import GMX_BIN
        try:
            subprocess.run(
                [GMX_BIN, "rms", "-s", str(tpr), "-f", str(xtc),
                 "-o", str(xvg), "-tu", "ns"],
                input="Protein\nProtein\n", capture_output=True, text=True,
                cwd=str(md_dir), timeout=300,
            )
        except Exception as e:
            logger.warning(f"    gmx rms failed: {e}")

        if xvg.exists():
            rmsds = []
            with open(xvg) as f:
                for line in f:
                    if line.startswith(("#", "@")):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        rmsds.append(float(parts[1]) * 10.0)  # nm → Å

            if rmsds:
                n = len(rmsds)
                rmsd_start = round(float(np.mean(rmsds[:n//4])), 2)
                rmsd_end = round(float(np.mean(rmsds[-n//4:])), 2)
                escaped = rmsd_end > REBINDING_RMSD_THRESHOLD

        if escaped is True:
            status = "REMOVABLE (moderate binding — good MIP)"
        elif escaped is False:
            status = "STUCK (too strong binding — template removal difficult)"
        else:
            status = "N/A"

        logger.info(f"    Removal test: RMSD {rmsd_start}→{rmsd_end} Å → {status}")

        return {
            "rmsd_start_A": rmsd_start,
            "rmsd_end_A": rmsd_end,
            "escaped": escaped,
            "status": status,
        }

    except Exception as e:
        logger.error(f"    Template removal test failed: {e}")
        return {"error": str(e)}


# ── Dual-Imprinting VIP ──────────────────────────────────────

def _run_dual_imprinting_vip(target, target_names, snapshot_results,
                              phase1_results, p4_md_dir, target_dir,
                              n_glycan=1):
    """
    Dual-imprinting: physically add APBA molecules to cavity, then re-run VIP.

    1. Parameterize APBA (acpype/GAFF2)
    2. Place n_glycan APBA molecules near protein COM in each snapshot cavity
    3. Update topology (add APBA ITP + molecules + position restraint)
    4. Re-run rebinding MD with APBA-enhanced cavity
    5. APBA forms boronate-diol bond with glycan → glycosylated targets bind better

    Teixeira 2021 [1]: dual epitope+glycan imprinting for CD63.
    """
    import MDAnalysis as mda
    from .config import (REBINDING_MD_NS, REBINDING_RMSD_THRESHOLD,
                         REBINDING_RESTRAINT_K, resolve_path, ALL_MONOMERS)
    from .utils_structure import smiles_to_mol2
    from .utils_gromacs import parameterize_monomer

    logger.info(f"    Dual-imprinting VIP: physically adding {n_glycan} APBA to cavity")

    # 1. Parameterize APBA
    apba_info = ALL_MONOMERS.get("APBA")
    if not apba_info:
        logger.error("    APBA not found in monomer library")
        return {"error": "APBA not in library"}

    param_dir = target_dir / "dual_apba_param"
    param_dir.mkdir(parents=True, exist_ok=True)
    try:
        mol2 = smiles_to_mol2(apba_info["smiles"], "APBA", param_dir)
        apba_param = parameterize_monomer(mol2, "APBA", param_dir)
        apba_itp = apba_param.get("itp")
        apba_gro = apba_param.get("gro")
        if not apba_itp or not apba_gro:
            logger.error("    APBA parameterization failed")
            return {"error": "APBA parameterization failed"}
        logger.info(f"    APBA parameterized: {apba_itp}")
    except Exception as e:
        logger.error(f"    APBA parameterization failed: {e}")
        return {"error": str(e)}

    dual_snapshot_results = []

    for i, snap in enumerate(snapshot_results):
        if not snap.get("success"):
            continue

        snap_dir = target_dir / f"snapshot_{i}" / "dual_imprinting"
        snap_dir.mkdir(parents=True, exist_ok=True)

        orig_snap_dir = target_dir / f"snapshot_{i}"
        cavity_gro = orig_snap_dir / "frame.gro"
        cavity_top = orig_snap_dir / "topol.top"

        if not cavity_gro.exists():
            logger.warning(f"    snap{i}: cavity files missing, skip")
            continue

        # 2. Add APBA molecules to cavity GRO
        try:
            u_cav = mda.Universe(str(cavity_gro))
            protein = u_cav.select_atoms("protein")
            prot_com = protein.center_of_mass()

            # Find APBA docked pose from Phase 2 (actual binding position)
            from .config import get_output_path
            apba_docked_pdbqt = None
            p2_base = get_output_path("phase2")
            for p2_dir in p2_base.glob(f"smd_{target}*/smd_{target}/{target}_APBA"):
                best = p2_dir / "APBA_best.pdbqt"
                if best.exists():
                    apba_docked_pdbqt = best
                    break

            # Get APBA coordinates: prefer docked pose, fallback to GRO + COM offset
            cav_lines = Path(cavity_gro).read_text().strip().split("\n")
            cav_natoms = int(cav_lines[1].strip())
            box_line = cav_lines[-1]

            apba_gro_text = Path(apba_gro).read_text().strip().split("\n")
            apba_natoms = int(apba_gro_text[1].strip())
            apba_coord_lines = apba_gro_text[2:2 + apba_natoms]

            if apba_docked_pdbqt:
                # Use Phase 2 docked position — APBA at its actual binding site
                logger.info(f"    Using Phase 2 docked APBA pose: {apba_docked_pdbqt}")
                try:
                    u_docked = mda.Universe(str(apba_docked_pdbqt))
                    docked_com = u_docked.select_atoms("all").center_of_mass()
                    # Docked coords are in epitope frame; need to align with cavity
                    # APBA GRO COM → shift to docked COM position
                    u_apba = mda.Universe(str(apba_gro))
                    apba_com = u_apba.select_atoms("all").center_of_mass()
                    # Shift in nm (GRO) — docked is in Å (PDB)
                    shift_x = docked_com[0] / 10.0 - apba_com[0] / 10.0
                    shift_y = docked_com[1] / 10.0 - apba_com[1] / 10.0
                    shift_z = docked_com[2] / 10.0 - apba_com[2] / 10.0
                    docked_offsets = [(shift_x, shift_y, shift_z)]
                    # For additional copies, add small perturbations (±0.5nm)
                    for di in range(1, n_glycan):
                        perturb = [(0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0),
                                   (0, -0.5, 0)][di - 1] if di < 5 else (0, 0, 0)
                        docked_offsets.append((
                            shift_x + perturb[0],
                            shift_y + perturb[1],
                            shift_z + perturb[2]))
                except Exception as e:
                    logger.warning(f"    Docked pose parsing failed: {e}, using COM offset")
                    docked_offsets = None
            else:
                docked_offsets = None

            if not docked_offsets:
                # Fallback: place near protein COM
                logger.info(f"    No docked pose found, placing APBA near protein COM")
                docked_offsets = []
                offsets_nm = [(1.5, 0, 0), (-1.5, 0, 0), (0, 1.5, 0)]
                for ci in range(n_glycan):
                    ox, oy, oz = offsets_nm[ci % len(offsets_nm)]
                    docked_offsets.append((
                        prot_com[0] / 10.0 + ox,
                        prot_com[1] / 10.0 + oy,
                        prot_com[2] / 10.0 + oz))

            new_atom_lines = []
            for copy_i in range(min(n_glycan, len(docked_offsets))):
                sx, sy, sz = docked_offsets[copy_i]
                for line in apba_coord_lines:
                    if len(line) >= 44:
                        x = float(line[20:28]) + sx
                        y = float(line[28:36]) + sy
                        z = float(line[36:44]) + sz
                        res_num = cav_natoms // max(apba_natoms, 1) + copy_i + 100
                        new_line = f"{res_num:5d}APBA {line[10:15]}{cav_natoms + len(new_atom_lines) + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
                        new_atom_lines.append(new_line)

            total_atoms = cav_natoms + len(new_atom_lines)
            dual_gro = snap_dir / "dual_cavity.gro"

            # Insert APBA before SOL (topology order must match GRO order)
            atom_lines = cav_lines[2:2 + cav_natoms]
            sol_start = None
            for li, line in enumerate(atom_lines):
                if len(line) >= 10 and "SOL" in line[5:10]:
                    sol_start = li
                    break

            merged = [cav_lines[0]]
            merged.append(f" {total_atoms}")
            if sol_start is not None:
                merged.extend(atom_lines[:sol_start])  # protein + monomers
                merged.extend(new_atom_lines)            # APBA
                merged.extend(atom_lines[sol_start:])    # SOL + ions
            else:
                merged.extend(atom_lines)
                merged.extend(new_atom_lines)
            merged.append(box_line)
            dual_gro.write_text("\n".join(merged) + "\n")

            # 3. Update topology: add APBA ITP + molecules
            shutil.copy2(str(apba_itp), str(snap_dir / Path(apba_itp).name))
            # Copy all existing ITPs
            for itp in orig_snap_dir.glob("*.itp"):
                dst = snap_dir / itp.name
                if not dst.exists():
                    shutil.copy2(str(itp), str(dst))
            if p4_md_dir:
                for itp in Path(p4_md_dir).glob("*.itp"):
                    dst = snap_dir / itp.name
                    if not dst.exists():
                        shutil.copy2(str(itp), str(dst))

            top_text = Path(cavity_top).read_text()
            apba_itp_name = Path(apba_itp).name

            # Add APBA include after forcefield
            if f'#include "{apba_itp_name}"' not in top_text:
                top_text = top_text.replace(
                    "[ moleculetype ]",
                    f'#include "{apba_itp_name}"\n\n[ moleculetype ]', 1)

            # Add APBA to [ molecules ] before SOL
            n_apba_added = min(n_glycan, len(docked_offsets))
            top_text = top_text.replace(
                "SOL", f"APBA     {n_apba_added}\nSOL", 1)

            # Remove [ atomtypes ] from APBA ITP (already in main topology)
            apba_itp_path = snap_dir / apba_itp_name
            apba_itp_text = apba_itp_path.read_text()
            cleaned_lines = []
            skip_atomtypes = False
            for line in apba_itp_text.split("\n"):
                if "[ atomtypes ]" in line:
                    skip_atomtypes = True
                    continue
                if skip_atomtypes:
                    if line.strip().startswith("[") and "atomtypes" not in line:
                        skip_atomtypes = False
                    else:
                        continue
                cleaned_lines.append(line)
            apba_itp_text = "\n".join(cleaned_lines)
            apba_itp_path.write_text(apba_itp_text)

            # Also add APBA atomtypes to main topology's [ atomtypes ] section
            # Extract from original ITP
            orig_itp_text = Path(apba_itp).read_text()
            apba_atomtypes = ""
            in_at = False
            for line in orig_itp_text.split("\n"):
                if "[ atomtypes ]" in line:
                    in_at = True
                    continue
                if in_at:
                    if line.strip().startswith("[") and "atomtypes" not in line:
                        break
                    if line.strip() and not line.startswith(";"):
                        apba_atomtypes += line + "\n"

            if apba_atomtypes and "[ atomtypes ]" in top_text:
                # Insert APBA atomtypes INSIDE the [ atomtypes ] block, right after
                # the existing atomtype entries and BEFORE any #include (which pull
                # in moleculetypes). Inserting before the next "[" section would
                # place them after the silane #includes + APBA #include → GROMACS
                # would parse APBA's [ atoms ] (using n3) before n3 is defined.
                lines = top_text.split("\n")
                out_lines = []
                inserted = False
                in_at = False
                for ln in lines:
                    if ln.strip().startswith("[ atomtypes ]"):
                        in_at = True
                        out_lines.append(ln)
                        continue
                    if in_at and not inserted:
                        # End of atomtype entries: blank line, #include, or new [ section
                        stripped = ln.strip()
                        if (stripped == "" or stripped.startswith("#")
                                or stripped.startswith("[")):
                            out_lines.append(apba_atomtypes.rstrip("\n"))
                            inserted = True
                            in_at = False
                    out_lines.append(ln)
                if not inserted:  # fallback: append at very end of atomtypes header
                    out_lines.append(apba_atomtypes)
                top_text = "\n".join(out_lines)

            # Add position restraint to APBA ITP
            if "#ifdef POSRES_MONOMER" not in apba_itp_text:
                # Count heavy atoms
                heavy_atoms = []
                in_atoms = False
                for line in apba_itp_text.split("\n"):
                    if "[ atoms ]" in line:
                        in_atoms = True
                        continue
                    if in_atoms and line.strip().startswith("["):
                        break
                    if in_atoms and line.strip() and not line.startswith(";"):
                        parts = line.split()
                        if len(parts) >= 2 and not parts[1].startswith("h"):
                            heavy_atoms.append(parts[0])
                posre = "\n#ifdef POSRES_MONOMER\n[ position_restraints ]\n"
                posre += "; ai  funct  fcx    fcy    fcz\n"
                for ai in heavy_atoms:
                    posre += f"  {ai}    1  {REBINDING_RESTRAINT_K}  {REBINDING_RESTRAINT_K}  {REBINDING_RESTRAINT_K}\n"
                posre += "#endif\n"
                apba_itp_path.write_text(apba_itp_text.rstrip() + "\n" + posre)

            dual_top = snap_dir / "topol.top"
            dual_top.write_text(top_text)

            logger.info(f"    snap{i}: added {n_apba_added} APBA to cavity "
                        f"({total_atoms} atoms)")

        except Exception as e:
            logger.error(f"    snap{i}: APBA insertion failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 4. Run rebinding MD with APBA-enhanced cavity.
        # Use SAME template type as Phase 4 (ECL2 in whole-protein mode).
        from .config import PHASE4_TEMPLATE_MODE
        _tkey = "ecl2_pdb" if PHASE4_TEMPLATE_MODE == "ecl2" else "head_pdb"
        own_head = resolve_path(phase1_results[target].get(
            _tkey, phase1_results[target].get(
                "head_pdb", phase1_results[target]["epitope_pdb"])))

        logger.info(f"    snap{i}: rebinding own ({target}) with APBA cavity...")
        rebind_own = _run_rebinding_md(
            str(dual_gro), str(dual_top), own_head,
            snap_dir / "rebind_own",
            time_ns=REBINDING_MD_NS,
            p4_md_dir=str(snap_dir),
            is_own_target=True)

        dual_snap = {"rebind_own": rebind_own}

        # Rebind other targets
        for other_t in target_names:
            if other_t == target:
                continue
            other_n_glycan = phase1_results[other_t].get(
                "properties", {}).get("n_glycan_sites_known", 0)
            other_head = resolve_path(phase1_results[other_t].get(
                _tkey, phase1_results[other_t].get(
                    "head_pdb", phase1_results[other_t]["epitope_pdb"])))

            logger.info(f"    snap{i}: rebinding {other_t} "
                        f"(glycan={other_n_glycan}) with APBA cavity...")
            rebind_other = _run_rebinding_md(
                str(dual_gro), str(dual_top), other_head,
                snap_dir / f"rebind_{other_t}",
                time_ns=REBINDING_MD_NS,
                p4_md_dir=str(snap_dir),
                is_own_target=False)
            dual_snap[f"rebind_{other_t}"] = rebind_other

        dual_snap["success"] = True
        dual_snapshot_results.append(dual_snap)

    # Analyze
    if dual_snapshot_results:
        dual_analysis = _analyze_rebinding_results(
            target, target_names, dual_snapshot_results,
            REBINDING_RMSD_THRESHOLD)
        dual_analysis["n_glycan_sites"] = n_glycan
        dual_analysis["n_apba_added"] = min(n_glycan, 5)
        dual_analysis["apba_added"] = True
        dual_analysis["note"] = (
            f"Dual-imprinting: {min(n_glycan, 5)} APBA molecules physically added "
            f"to cavity with position restraints. "
            f"APBA boronic acid provides glycan recognition layer.")

        sel = dual_analysis.get("selectivity", {})
        for ot, s in sel.items():
            other_glycan = phase1_results.get(ot, {}).get(
                "properties", {}).get("n_glycan_sites_known", "?")
            logger.info(
                f"    Dual SI vs {ot} (glycan={other_glycan}): "
                f"SI={s.get('selectivity_index')} [{s.get('selectivity_label')}]")

        return dual_analysis

    return {"error": "No snapshots for dual-imprinting"}


# ── Results Analysis ─────────────────────────────────────────

def _analyze_rebinding_results(target, all_targets, snapshot_results, threshold):
    """Compute rebinding success rate, selectivity index, and statistics."""
    from scipy import stats

    n_success = 0
    n_total = 0
    own_rmsds = []
    other_rmsds = {t: [] for t in all_targets if t != target}
    # Per-snapshot H-bond and contact data
    own_hbonds = []
    other_hbonds = {t: [] for t in all_targets if t != target}
    own_contacts = []
    other_contacts = {t: [] for t in all_targets if t != target}
    # Size-exclusion: count snapshots where cross-protein was rejected by clash
    other_size_excluded = {t: 0 for t in all_targets if t != target}

    for snap in snapshot_results:
        if not snap.get("success"):
            continue
        n_total += 1

        own = snap.get("rebind_own", {})
        rmsd = own.get("rmsd_mean_A")
        if rmsd is not None:
            own_rmsds.append(rmsd)
            if rmsd < threshold:
                n_success += 1
        # Collect H-bond and contact data
        if own.get("hbond_mean") is not None:
            own_hbonds.append(own["hbond_mean"])
        if own.get("contact_mean") is not None:
            own_contacts.append(own["contact_mean"])

        for other_t in all_targets:
            if other_t == target:
                continue
            other = snap.get(f"rebind_{other_t}", {})
            # Size-exclusion: cross-protein rejected by steric clash
            if other.get("size_excluded"):
                other_size_excluded[other_t] += 1
                continue
            r = other.get("rmsd_mean_A")
            if r is not None:
                other_rmsds[other_t].append(r)
            if other.get("hbond_mean") is not None:
                other_hbonds[other_t].append(other["hbond_mean"])
            if other.get("contact_mean") is not None:
                other_contacts[other_t].append(other["contact_mean"])

    own_mean = float(np.mean(own_rmsds)) if own_rmsds else None
    own_std = float(np.std(own_rmsds)) if len(own_rmsds) > 1 else None

    result = {
        "target": target,
        "n_snapshots": n_total,
        "n_rebound": n_success,
        "success_rate": f"{n_success}/{n_total}" if n_total > 0 else "0/0",
        "own_rmsd_mean": round(own_mean, 2) if own_mean else None,
        "own_rmsd_std": round(own_std, 2) if own_std else None,
        "own_hbond_mean": round(float(np.mean(own_hbonds)), 1) if own_hbonds else None,
        "own_contact_mean": round(float(np.mean(own_contacts)), 1) if own_contacts else None,
        "snapshots": snapshot_results,
        "selectivity": {},
    }

    # Selectivity Index (SI) and statistical tests for each other target
    for other_t, rmsds in other_rmsds.items():
        if not rmsds:
            continue
        other_mean = float(np.mean(rmsds))
        other_std = float(np.std(rmsds)) if len(rmsds) > 1 else None

        # Selectivity Index: SI = RMSD_other / RMSD_own
        # SI > 1.5 = selective, 1.0-1.5 = weak, < 1.0 = cross-reactive
        si = round(other_mean / own_mean, 2) if own_mean and own_mean > 0 else None

        # Welch's t-test: own_rmsds vs other_rmsds
        p_value = None
        if len(own_rmsds) >= 2 and len(rmsds) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(
                    rmsds, own_rmsds, equal_var=False)
                p_value = round(float(p_value), 4)
            except Exception:
                pass

        if si is not None:
            if si > 1.5:
                sel_label = "selective"
            elif si > 1.0:
                sel_label = "weak"
            else:
                sel_label = "cross-reactive"
        else:
            sel_label = "N/A"

        sel_entry = {
            "other_rmsd_mean": round(other_mean, 2),
            "other_rmsd_std": round(other_std, 2) if other_std else None,
            "selectivity_index": si,
            "selectivity_label": sel_label,
            "p_value": p_value,
            "significant": p_value < 0.05 if p_value is not None else None,
        }

        # A6: Bootstrap 95% CI for SI
        try:
            from .utils_analysis import bootstrap_selectivity_index
            from .config import BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI
            boot = bootstrap_selectivity_index(
                own_rmsds, rmsds,
                n_bootstrap=BOOTSTRAP_N_RESAMPLES, ci=BOOTSTRAP_CI)
            if 'error' not in boot:
                sel_entry["si_ci_lower"] = round(boot["si_ci_lower"], 2)
                sel_entry["si_ci_upper"] = round(boot["si_ci_upper"], 2)
                sel_entry["si_bootstrap_mean"] = round(boot["si_mean"], 2)
                sel_entry["is_selective_at_95CI"] = boot["is_selective_at_ci"]
        except Exception as _e:
            logger.debug(f"Bootstrap CI failed for {other_t}: {_e}")

        # H-bond selectivity
        other_hb = other_hbonds.get(other_t, [])
        if own_hbonds and other_hb:
            sel_entry["own_hbond_mean"] = round(float(np.mean(own_hbonds)), 1)
            sel_entry["other_hbond_mean"] = round(float(np.mean(other_hb)), 1)

        # Contact selectivity
        other_ct = other_contacts.get(other_t, [])
        if own_contacts and other_ct:
            sel_entry["own_contact_mean"] = round(float(np.mean(own_contacts)), 1)
            sel_entry["other_contact_mean"] = round(float(np.mean(other_ct)), 1)

        result["selectivity"][other_t] = sel_entry
        # Keep backward compatibility
        result[f"other_{other_t}_rmsd_mean"] = round(other_mean, 2)

    # Size-excluded cross-targets → strong selectivity (shape/size exclusion).
    # These had no MD (rejected by steric clash) so aren't in other_rmsds.
    for other_t, n_excl in other_size_excluded.items():
        if n_excl > 0 and other_t not in result["selectivity"]:
            result["selectivity"][other_t] = {
                "other_rmsd_mean": None,
                "selectivity_index": None,
                "selectivity_label": "size-excluded",
                "size_excluded_snapshots": n_excl,
                "p_value": None,
                "significant": True,  # physical rejection = significant
                "mechanism": "size/shape exclusion (cross-protein too large for cavity)",
            }

    return result


def _print_phase6_summary(results):
    """Print Phase 6 summary with selectivity index."""
    logger.info(f"\n{'='*60}")
    logger.info("Phase 6: VIP Cavity Rebinding Summary")
    logger.info(f"{'='*60}")

    for target, data in results.items():
        logger.info(f"\n[{target}]")
        own_std = data.get('own_rmsd_std')
        std_str = f" ± {own_std}" if own_std else ""
        logger.info(f"  Rebinding: {data.get('success_rate', 'N/A')}")
        logger.info(f"  Own RMSD: {data.get('own_rmsd_mean', 'N/A')}{std_str} Å")

        if data.get("own_hbond_mean") is not None:
            logger.info(f"  Own H-bonds: {data['own_hbond_mean']}")
        if data.get("own_contact_mean") is not None:
            logger.info(f"  Own contacts: {data['own_contact_mean']}")

        sel = data.get("selectivity", {})
        for other_t, s in sel.items():
            si = s.get("selectivity_index", "N/A")
            label = s.get("selectivity_label", "")
            p = s.get("p_value")
            sig = "*" if s.get("significant") else ""
            p_str = f" (p={p}{sig})" if p is not None else ""
            logger.info(
                f"  vs {other_t}: {s.get('other_rmsd_mean')} Å  "
                f"SI={si} [{label}]{p_str}")

    logger.info(f"{'='*60}")


# ── Re-analysis with PBC centering + Q4 ────────────────────────

def _ensure_centered_xtc(md_dir: Path) -> Path:
    """Generate PBC-centered xtc (md_centered.xtc) for analysis. Caches result."""
    from .config import GMX_BIN
    import subprocess
    md_dir = Path(md_dir)
    xtc = md_dir / "md.xtc"
    tpr = md_dir / "md.tpr"
    centered = md_dir / "md_centered.xtc"
    if not (xtc.exists() and tpr.exists()):
        return None
    if centered.exists() and centered.stat().st_mtime > xtc.stat().st_mtime:
        return centered
    proc = subprocess.run(
        [GMX_BIN, "trjconv", "-f", str(xtc), "-s", str(tpr),
         "-o", str(centered), "-pbc", "mol", "-center"],
        input="1\n0\n", capture_output=True, text=True, timeout=1800)
    if not centered.exists():
        logger.warning(f"  trjconv centering failed for {md_dir}")
        return None
    return centered


def _q4_rmsd_from_centered(md_dir: Path) -> dict:
    """Compute Q4 (last 25%) RMSD from PBC-centered trajectory.

    First centers via gmx trjconv, then runs gmx rms on centered xtc, parses
    last 25% of the time series.
    """
    from .config import GMX_BIN, REBINDING_RMSD_THRESHOLD
    import subprocess
    centered = _ensure_centered_xtc(md_dir)
    if centered is None:
        return {"error": "trajectory missing or centering failed"}
    tpr = md_dir / "md.tpr"
    xvg = md_dir / "rmsd_centered.xvg"
    proc = subprocess.run(
        [GMX_BIN, "rms", "-s", str(tpr), "-f", str(centered),
         "-o", str(xvg), "-tu", "ns"],
        input="Protein\nProtein\n", capture_output=True, text=True,
        cwd=str(md_dir), timeout=300)
    if not xvg.exists():
        return {"error": "gmx rms failed"}
    rmsds = []
    for line in xvg.read_text(encoding="utf-8").split("\n"):
        if line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try: rmsds.append(float(parts[1]) * 10.0)
            except ValueError: pass
    if len(rmsds) < 4:
        return {"error": "trajectory too short"}
    rmsds = np.array(rmsds)
    n = len(rmsds)
    q1, q4 = rmsds[:n // 4], rmsds[3 * n // 4:]
    drift = float(q4.mean()) - float(q1.mean())
    return {
        "n_frames": n,
        "q1_mean_A": round(float(q1.mean()), 2),
        "q4_mean_A": round(float(q4.mean()), 2),
        "q4_std_A": round(float(q4.std()), 2),
        "rmsd_final_A": round(float(rmsds[-1]), 2),
        "drift_A": round(drift, 2),
        "converged": abs(drift) < 1.0 and float(q4.std()) < 1.0,
        "rebound_q4": float(q4.mean()) < REBINDING_RMSD_THRESHOLD,
    }


def reanalyze_phase5(target_names: list = None,
                     output_dir: str = None) -> dict:
    """Re-analyze existing Phase 5 rebinding MDs with PBC centering + Q4.

    Does NOT re-run MD. Applies gmx trjconv -pbc mol -center to existing
    md.xtc, then computes Q4 RMSD on centered trajectories, then derives
    selectivity statistics.

    Auto-detects phase5_extended over phase5 if both exist.
    """
    from .config import (REBINDING_RMSD_THRESHOLD as _RT,
                         get_output_path)
    from scipy.stats import ttest_ind

    if output_dir is None:
        phase5_ext = get_output_path("phase5").parent / "phase5_extended"
        if phase5_ext.exists() and any(phase5_ext.glob("*/snapshot_*")):
            output_dir = str(phase5_ext)
            logger.info(f"Re-analyzing: {output_dir}")
        else:
            output_dir = str(get_output_path("phase5"))
    output_dir = Path(output_dir)

    if target_names is None:
        target_names = sorted({d.name for d in output_dir.iterdir()
                                if d.is_dir() and d.name in ("CD63", "CD81", "CD9")})

    summary = {}
    for target in target_names:
        target_dir = output_dir / target
        if not target_dir.exists():
            continue
        cross_targets = [t for t in target_names if t != target]

        snap_data = []
        for snap_dir in sorted(target_dir.glob("snapshot_*")):
            si = int(snap_dir.name.split("_")[-1])
            entry = {"snap_idx": si}
            for sub_name in ["rebind_own"] + [f"rebind_{t}" for t in cross_targets]:
                sub_dir = snap_dir / sub_name / "md"
                if sub_dir.exists():
                    diag = _q4_rmsd_from_centered(sub_dir)
                    entry[sub_name] = diag
            snap_data.append(entry)

        # Aggregate per cross target
        own_q4 = [s["rebind_own"]["q4_mean_A"] for s in snap_data
                  if "rebind_own" in s and "q4_mean_A" in s["rebind_own"]]
        own_q4_conv = [s["rebind_own"]["q4_mean_A"] for s in snap_data
                       if "rebind_own" in s and s["rebind_own"].get("converged")]

        sel = {}
        for ot in cross_targets:
            key = f"rebind_{ot}"
            cross_q4 = [s[key]["q4_mean_A"] for s in snap_data
                        if key in s and "q4_mean_A" in s[key]]
            cross_q4_conv = [s[key]["q4_mean_A"] for s in snap_data
                             if key in s and s[key].get("converged")]

            si_all = (round(float(np.mean(cross_q4)) / float(np.mean(own_q4)), 2)
                      if own_q4 and cross_q4 else None)
            si_conv = (round(float(np.mean(cross_q4_conv)) / float(np.mean(own_q4_conv)), 2)
                       if own_q4_conv and cross_q4_conv else None)
            try:
                p_all = round(float(ttest_ind(cross_q4, own_q4, equal_var=False).pvalue), 4) if len(own_q4) >= 3 and len(cross_q4) >= 3 else None
            except Exception:
                p_all = None
            try:
                p_conv = round(float(ttest_ind(cross_q4_conv, own_q4_conv, equal_var=False).pvalue), 4) if len(own_q4_conv) >= 3 and len(cross_q4_conv) >= 3 else None
            except Exception:
                p_conv = None

            sel[ot] = {
                "n_cross": len(cross_q4),
                "n_cross_converged": len(cross_q4_conv),
                "cross_q4_mean": round(float(np.mean(cross_q4)), 2) if cross_q4 else None,
                "cross_q4_std": round(float(np.std(cross_q4)), 2) if cross_q4 else None,
                "selectivity_index_all": si_all,
                "selectivity_index_converged": si_conv,
                "p_value_all": p_all,
                "p_value_converged": p_conv,
                "selectivity_label": (
                    "selective" if (si_all and si_all > 1.5 and p_all and p_all < 0.05)
                    else "weak" if (si_all and 1.0 < si_all <= 1.5)
                    else "cross-reactive" if (si_all and si_all <= 1.0)
                    else "n/a"
                ),
            }

        summary[target] = {
            "n_total": len(snap_data),
            "n_own_converged": len(own_q4_conv),
            "n_rebound_q4": sum(1 for s in snap_data
                                if s.get("rebind_own", {}).get("rebound_q4")),
            "own_q4_mean": round(float(np.mean(own_q4)), 2) if own_q4 else None,
            "own_q4_std": round(float(np.std(own_q4)), 2) if own_q4 else None,
            "own_q4_converged_only_mean": (
                round(float(np.mean(own_q4_conv)), 2) if own_q4_conv else None),
            "selectivity": sel,
            "snapshots": snap_data,
        }

        s = summary[target]
        logger.info(f"\n=== {target} (PBC-centered + Q4) ===")
        logger.info(f"  N total: {s['n_total']}, own converged: {s['n_own_converged']}")
        logger.info(f"  Q4 own: {s['own_q4_mean']} ± {s['own_q4_std']} Å "
                    f"(rebound: {s['n_rebound_q4']}/{s['n_total']})")
        for ot, ss in sel.items():
            logger.info(f"  vs {ot}: SI={ss['selectivity_index_all']} "
                        f"(p={ss['p_value_all']}), "
                        f"conv-only SI={ss['selectivity_index_converged']} "
                        f"(p={ss['p_value_converged']})")

    out_file = output_dir / "phase5_reanalyzed_centered.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"\nSaved: {out_file}")
    return summary


# ── Convergence Diagnostic & Extension ──────────────────────────

def _is_md_drifting(rebind_md_dir: Path, drift_threshold_A: float = 1.5) -> dict:
    """Check if RMSD trajectory shows drift (Q1→Q4 difference) > threshold."""
    xvg = Path(rebind_md_dir) / "rmsd_rebind.xvg"
    if not xvg.exists():
        return {"available": False}
    rmsds = []
    for line in xvg.read_text(encoding="utf-8").split("\n"):
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                rmsds.append(float(parts[1]) * 10.0)  # nm → Å
            except ValueError:
                pass
    if len(rmsds) < 100:
        return {"available": False}
    rmsds = np.array(rmsds)
    n = len(rmsds)
    q1 = float(rmsds[:n // 4].mean())
    q4 = float(rmsds[3 * n // 4:].mean())
    q4_std = float(rmsds[3 * n // 4:].std())
    drift = q4 - q1
    return {
        "available": True,
        "q1_mean_A": round(q1, 2),
        "q4_mean_A": round(q4, 2),
        "q4_std_A": round(q4_std, 2),
        "drift_A": round(drift, 2),
        "drifting": abs(drift) > drift_threshold_A,
        "converged": abs(drift) < 1.0 and q4_std < 1.0,
    }


def extend_drifting_mds(target_names: list = None,
                         output_dir: str = None,
                         extend_ns: int = 100,
                         drift_threshold_A: float = 1.5) -> dict:
    """Extend non-converged Phase 5 rebinding MDs by `extend_ns` ns each.

    Uses gmx convert-tpr -extend + gmx mdrun -cpi state.cpt to continue
    from existing checkpoint. Operates on existing output directory in-place.

    Targets snapshots with |Q1→Q4 drift| > drift_threshold_A.

    Auto-detects best Phase 5 source: prefers phase5_extended (n=10/50 ns)
    over phase5 (n=5/20 ns) if both exist.
    """
    from .config import GMX_BIN, get_output_path
    import subprocess

    if output_dir is None:
        # Prefer phase5_extended (n=10/50ns) if available
        phase5_ext = get_output_path("phase5").parent / "phase5_extended"
        if phase5_ext.exists() and any(phase5_ext.glob("*/snapshot_*")):
            output_dir = str(phase5_ext)
            logger.info(f"Using phase5_extended source: {output_dir}")
        else:
            output_dir = str(get_output_path("phase5"))
    output_dir = Path(output_dir)

    if target_names is None:
        target_names = [d.name for d in output_dir.iterdir() if d.is_dir()]

    extended, skipped = [], []
    for target in target_names:
        target_dir = output_dir / target
        if not target_dir.exists():
            continue
        logger.info(f"\n=== {target}: drift detection (threshold {drift_threshold_A} Å) ===")
        for snap_dir in sorted(target_dir.glob("snapshot_*")):
            for sub_dir in snap_dir.iterdir():
                if not sub_dir.is_dir() or not sub_dir.name.startswith("rebind_"):
                    continue
                md_dir = sub_dir / "md"
                diag = _is_md_drifting(md_dir, drift_threshold_A)
                if not diag.get("available"):
                    continue
                if not diag.get("drifting"):
                    logger.info(f"  {snap_dir.name}/{sub_dir.name}: converged "
                                f"(drift={diag['drift_A']:+.2f} Å)")
                    continue

                logger.info(f"  {snap_dir.name}/{sub_dir.name}: DRIFT "
                            f"(Q1={diag['q1_mean_A']}, Q4={diag['q4_mean_A']}, "
                            f"drift={diag['drift_A']:+.2f}) → extending {extend_ns} ns")

                tpr = md_dir / "md.tpr"
                cpt = md_dir / "md.cpt"
                if not (tpr.exists() and cpt.exists()):
                    skipped.append(f"{target}/{snap_dir.name}/{sub_dir.name} (missing tpr/cpt)")
                    continue

                # convert-tpr to extend
                new_tpr = md_dir / "md_extended.tpr"
                proc = subprocess.run(
                    [GMX_BIN, "convert-tpr", "-s", str(tpr),
                     "-o", str(new_tpr), "-extend", str(extend_ns * 1000)],
                    capture_output=True, text=True, timeout=60)
                if not new_tpr.exists():
                    skipped.append(f"{target}/{snap_dir.name}/{sub_dir.name} (convert-tpr failed)")
                    continue

                # Backup original tpr, replace with extended
                if not (md_dir / "md_orig.tpr").exists():
                    tpr.rename(md_dir / "md_orig.tpr")
                else:
                    tpr.unlink()
                new_tpr.rename(tpr)

                # Continue MD from checkpoint
                proc = subprocess.run(
                    [GMX_BIN, "mdrun", "-deffnm", "md", "-cpi", "md.cpt",
                     "-nb", "gpu", "-pme", "gpu", "-bonded", "gpu",
                     "-update", "gpu", "-gpu_id", "0"],
                    cwd=str(md_dir), capture_output=True, text=True,
                    timeout=86400 * 2)
                extended.append(f"{target}/{snap_dir.name}/{sub_dir.name}")

    summary = {"extended": extended, "skipped": skipped, "extend_ns": extend_ns,
               "drift_threshold_A": drift_threshold_A}
    out_file = output_dir / "extension_log.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"\nExtension summary: {len(extended)} MDs extended, "
                f"{len(skipped)} skipped → {out_file}")
    return summary


# ── Multi-Restart Ensemble ──────────────────────────────────────

def _recenter_gro_via_trjconv(src_gro: Path, tpr_path: Path, dst_gro: Path) -> bool:
    """Use gmx trjconv -pbc mol -center to recenter protein in box.

    Required because Phase 4 trajectories may have protein crossing PBC boundaries.
    Without this, downstream MD inherits the split state.
    """
    from .config import GMX_BIN
    import subprocess
    try:
        proc = subprocess.run(
            [GMX_BIN, "trjconv", "-f", str(src_gro), "-s", str(tpr_path),
             "-o", str(dst_gro), "-pbc", "mol", "-center"],
            input="1\n0\n", capture_output=True, text=True, timeout=300)
        return dst_gro.exists()
    except Exception:
        return False


def _perturb_head_position(src_gro: Path, dst_gro: Path, rep_idx: int,
                            tpr_for_center: Path = None):
    """Generate perturbed starting structure: random rotation + small COM offset.

    First recenters input via gmx trjconv (-pbc mol -center) if tpr_for_center
    provided — handles PBC-split proteins from Phase 4 trajectories.
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.transformations import rotation_matrix

    # Step 0: recenter source if tpr available (handles PBC split)
    work = Path(dst_gro).parent
    centered_src = work / "src_centered.gro"
    if tpr_for_center and Path(tpr_for_center).exists():
        ok = _recenter_gro_via_trjconv(Path(src_gro), Path(tpr_for_center),
                                        centered_src)
        if ok:
            src_gro = centered_src
            logger.info(f"  Source recentered (trjconv -pbc mol -center)")

    rng = np.random.default_rng(seed=42 + rep_idx)
    u = mda.Universe(str(src_gro))
    prot = u.select_atoms("protein")
    com = prot.center_of_mass()
    angle = rng.uniform(30, 90)
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    R = rotation_matrix(np.deg2rad(angle), axis, com)[:3, :3]
    new_pos = (prot.positions - com) @ R.T + com
    prot.positions = new_pos
    offset = rng.uniform(-1.0, 1.0, size=3)
    prot.translate(offset)
    with mda.Writer(str(dst_gro), n_atoms=u.atoms.n_atoms) as w:
        w.write(u.atoms)
    return {"angle_deg": float(angle), "axis": axis.tolist(),
            "offset_A": offset.tolist(), "recentered": tpr_for_center is not None}


def run_multirestart(target_names: list,
                      n_reps: int = 3,
                      output_dir: str = None,
                      phase1_results: dict = None,
                      phase4_results: dict = None) -> dict:
    """Multi-restart ensemble for Phase 5 rebinding.

    For each snapshot, generate (n_reps - 1) additional starting structures
    with random head rotation + offset, run rebinding MD, ensemble-average results.

    Existing rep_0 (from standard Phase 5 run) is preserved.
    """
    from .config import (REBINDING_MD_NS, get_output_path, resolve_path)

    if output_dir is None:
        phase5_ext = get_output_path("phase5").parent / "phase5_extended"
        if phase5_ext.exists() and any(phase5_ext.glob("*/snapshot_*")):
            output_dir = str(phase5_ext)
            logger.info(f"Using phase5_extended source: {output_dir}")
        else:
            output_dir = str(get_output_path("phase5"))
    output_dir = Path(output_dir)
    multi_dir = output_dir.parent / "phase5_multirestart"
    multi_dir.mkdir(parents=True, exist_ok=True)

    if phase1_results is None:
        with open(get_output_path("phase1") / "phase1_results.json") as f:
            phase1_results = json.load(f)
    if phase4_results is None:
        with open(get_output_path("phase4") / "phase4_md_results.json") as f:
            phase4_results = json.load(f)

    # For cross-rebinding selectivity, always use ALL standard tetraspanin targets
    # as cross targets, regardless of what subset was given for multi-restart.
    from .config import TARGETS as _ALL_TARGETS
    all_targets = list(_ALL_TARGETS.keys())

    summary = {"n_reps": n_reps, "targets": {}}
    for target in target_names:
        # cross_targets = all standard tetraspanins except own (so CD63 → [CD81, CD9])
        cross_targets = [t for t in all_targets if t != target]
        head = resolve_path(phase1_results[target].get(
            "head_pdb", phase1_results[target]["epitope_pdb"]))

        # Phase 4 dir for ITPs
        p4 = phase4_results.get(target, {})
        best_pc = next(iter(p4), None)
        if not best_pc:
            continue
        p4_md_dir = get_output_path("phase4") / target / best_pc / "md"

        target_summary = []
        for snap_dir in sorted((output_dir / target).glob("snapshot_*")):
            snap_idx = int(snap_dir.name.split("_")[-1])
            cavity_gro = snap_dir / "frame.gro"
            cavity_top = snap_dir / "topol.top"
            if not (cavity_gro.exists() and cavity_top.exists()):
                continue

            logger.info(f"\n--- {target} snap_{snap_idx} multi-restart ---")
            snap_results = {"snap_idx": snap_idx, "reps": [{"rep": 0, "source": str(snap_dir)}]}

            for rep in range(1, n_reps):
                rep_dir = multi_dir / target / f"snapshot_{snap_idx}" / f"rep_{rep}"
                rep_dir.mkdir(parents=True, exist_ok=True)
                pgro = rep_dir / "frame_perturbed.gro"
                if not pgro.exists():
                    # Pass Phase 4 tpr so source is PBC-recentered before perturbation
                    p4_tpr = p4_md_dir / "md.tpr"
                    pinfo = _perturb_head_position(
                        cavity_gro, pgro, rep,
                        tpr_for_center=p4_tpr if p4_tpr.exists() else None)
                    snap_results.setdefault("perturbations", {})[str(rep)] = pinfo
                # Copy ITPs and topology
                for itp in snap_dir.glob("*.itp"):
                    dst = rep_dir / itp.name
                    if not dst.exists():
                        shutil.copy2(str(itp), str(dst))
                rep_top = rep_dir / "topol.top"
                if not rep_top.exists():
                    shutil.copy2(str(cavity_top), str(rep_top))

                logger.info(f"  Rep {rep}: own rebinding (50 ns)")
                own_res = _run_rebinding_md(
                    pgro, rep_top, head,
                    rep_dir / "rebind_own",
                    time_ns=REBINDING_MD_NS,
                    p4_md_dir=p4_md_dir,
                    is_own_target=True)

                rep_entry = {"rep": rep, "rebind_own": own_res}
                for ot in cross_targets:
                    cross_head = resolve_path(phase1_results[ot].get(
                        "head_pdb", phase1_results[ot]["epitope_pdb"]))
                    logger.info(f"  Rep {rep}: cross rebinding {ot}")
                    cross_res = _run_rebinding_md(
                        pgro, rep_top, cross_head,
                        rep_dir / f"rebind_{ot}",
                        time_ns=REBINDING_MD_NS,
                        p4_md_dir=p4_md_dir,
                        is_own_target=False)
                    rep_entry[f"rebind_{ot}"] = cross_res
                snap_results["reps"].append(rep_entry)
            target_summary.append(snap_results)

        summary["targets"][target] = target_summary

    out_file = multi_dir / "multirestart_summary.json"
    out_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"\nMulti-restart summary saved → {out_file}")
    return summary


# ════════════════════════════════════════════════════════════════
# A6: Bootstrap CI integration into selectivity analysis
# ════════════════════════════════════════════════════════════════

def compute_bootstrap_ci_for_selectivity(snapshot_results: list,
                                         cross_target: str,
                                         n_bootstrap: int = None,
                                         ci: float = None):
    """A6: Compute SI 95% CI via bootstrap from existing snapshot results."""
    from .config import BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI
    from .utils_analysis import bootstrap_selectivity_index

    n_bootstrap = n_bootstrap or BOOTSTRAP_N_RESAMPLES
    ci = ci or BOOTSTRAP_CI

    own_rmsds = []
    cross_rmsds = []
    for snap in snapshot_results:
        own = snap.get("rebind_own", {})
        cross = snap.get(f"rebind_{cross_target}", {})
        if own.get("rmsd_mean_A") is not None:
            own_rmsds.append(own["rmsd_mean_A"])
        if cross.get("rmsd_mean_A") is not None:
            cross_rmsds.append(cross["rmsd_mean_A"])

    return bootstrap_selectivity_index(own_rmsds, cross_rmsds,
                                       n_bootstrap=n_bootstrap, ci=ci)


# ════════════════════════════════════════════════════════════════
# B8: Multi-pose rebinding (multiple head conformers × replicates)
# ════════════════════════════════════════════════════════════════

def run_multipose_rebinding(target: str, cavity_gro, cavity_top,
                             head_conformers: list, n_replicates: int = 3,
                             time_ns: float = 50.0,
                             work_dir: Path = None,
                             p4_md_dir: Path = None) -> dict:
    """B8: For each head conformer × n replicates, run rebinding and aggregate.

    Reference: Hospital 2015 (Adv Bioinform) — ensemble docking robustness.
    """
    work_dir = Path(work_dir or ".")
    work_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for ci_, conf in enumerate(head_conformers):
        for rep in range(n_replicates):
            sub = work_dir / f"conf_{ci_}_rep_{rep}"
            sub.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Multi-pose rebinding: conformer={ci_}, rep={rep}")
            try:
                r = _run_rebinding_md(
                    cavity_gro, cavity_top, conf, sub,
                    time_ns=time_ns, p4_md_dir=p4_md_dir,
                    is_own_target=True,
                )
                r["conformer_idx"] = ci_
                r["rep"] = rep
                all_results.append(r)
            except Exception as e:
                all_results.append({"conformer_idx": ci_, "rep": rep,
                                     "error": str(e)})
    # Ensemble statistics
    valid_rmsds = [r["rmsd_mean_A"] for r in all_results
                   if r.get("rmsd_mean_A") is not None]
    if valid_rmsds:
        import numpy as np
        rmsd_mean = float(np.mean(valid_rmsds))
        rmsd_std = float(np.std(valid_rmsds))
    else:
        rmsd_mean = rmsd_std = None
    return {
        "ensemble_rmsd_mean": rmsd_mean,
        "ensemble_rmsd_std": rmsd_std,
        "n_total": len(all_results),
        "n_valid": len(valid_rmsds),
        "per_run": all_results,
    }


# ════════════════════════════════════════════════════════════════
# B9: FEP framework (Free Energy Perturbation) — stub
# ════════════════════════════════════════════════════════════════

def setup_fep_calculation(cavity_gro, cavity_top, template_pdb,
                           work_dir: Path, lambda_windows: int = None,
                           ns_per_window: int = None) -> dict:
    """B9: Set up Free Energy Perturbation calculation.

    Computes absolute binding free energy via N λ-windows of decoupling.
    Far more accurate than MM-GBSA but ~30 days/replicate.

    Reference: Mey 2020 LiveCoMS; Cournia 2020 JCTC.

    NOTE: Full FEP requires GROMACS-FEP setup (lambda topology + alchemical
    free energy mdp files). This function generates a SCAFFOLD for manual
    completion — the lambda windows are prepared but not auto-run, since
    they take 1-3 weeks per system.
    """
    from .config import FEP_LAMBDA_WINDOWS, FEP_NS_PER_WINDOW
    import numpy as np

    lambda_windows = lambda_windows or FEP_LAMBDA_WINDOWS
    ns_per_window = ns_per_window or FEP_NS_PER_WINDOW
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Generate λ values: 21 evenly spaced [0, 1]
    lambdas = np.linspace(0, 1, lambda_windows)

    summary = {
        "status": "scaffold_only",
        "method": "FEP / Alchemical Free Energy",
        "lambda_windows": int(lambda_windows),
        "ns_per_window": int(ns_per_window),
        "total_ns": float(lambda_windows * ns_per_window),
        "estimated_runtime_days": float(lambda_windows * ns_per_window / 100),  # ~100 ns/day
        "lambda_values": lambdas.tolist(),
        "work_dir": str(work_dir),
        "next_steps": [
            "Generate λ-topology files using `gmx grompp` per window with "
            "free_energy=yes, init_lambda_state=i in fep.mdp",
            "Run mdrun for each λ window in parallel",
            "Use `gmx bar` or alchemlyb (BAR/MBAR) for free energy estimation",
            "Estimated runtime: ~1-3 weeks per system on single GPU",
        ],
        "alternative_for_quick_estimate": "MM-GBSA via gmx_MMPBSA (already in pipeline)",
    }

    # Write FEP setup template
    fep_mdp_template = f"""\
; FEP/alchemical mdp (stub — fill λ-windows for production)
free-energy              = yes
init-lambda-state        = 0
delta-lambda             = 0
calc-lambda-neighbors    = 1
fep-lambdas              = {' '.join(f'{x:.4f}' for x in lambdas)}
nstdhdl                  = 100
dhdl-print-energy        = total
; ... standard MD settings (T, P, time, etc.) ...
nsteps                   = {ns_per_window * 500000}  ; {ns_per_window} ns at 2 fs
"""
    (work_dir / "fep_template.mdp").write_text(fep_mdp_template, encoding="utf-8")
    (work_dir / "README_FEP.txt").write_text(
        "FEP setup scaffold generated. To run:\n"
        "1. Copy cavity_gro/cavity_top to this dir\n"
        "2. Run grompp + mdrun per λ window (21 windows × 5 ns)\n"
        "3. Analyze with `gmx bar` or alchemlyb\n"
        f"4. Total compute: {summary['estimated_runtime_days']:.1f} days/replicate\n",
        encoding="utf-8")

    return summary
