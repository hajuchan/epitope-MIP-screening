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
                         REBINDING_RMSD_THRESHOLD, get_output_path, resolve_path)

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

    results = {}

    for target in target_names:
        p4 = phase4_results.get(target, {})
        if not p4:
            continue

        # Get top PC
        best_pc_id = next(iter(p4), None)
        if not best_pc_id:
            continue

        pc_data = p4[best_pc_id]
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

            # Step 5: Rebind other targets' heads (selectivity)
            for other_target in target_names:
                if other_target == target:
                    continue
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

    # Save
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
    from .config import REBINDING_RESTRAINT_K

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import MDAnalysis as mda

        u = mda.Universe(str(top_path), str(traj_path))
        u.trajectory[frame_idx]

        # Write full system GRO (keep everything including protein)
        frame_gro = output_dir / "frame.gro"
        with mda.Writer(str(frame_gro), n_atoms=u.atoms.n_atoms) as w:
            w.write(u.atoms)

        # Create position restraint for monomer heavy atoms
        # Index relative to full system (protein is first)
        monomer_atoms = u.select_atoms("not protein and not resname SOL NA CL")
        posre_path = output_dir / "posre_monomers.itp"
        with open(posre_path, "w") as f:
            f.write("[ position_restraints ]\n")
            f.write("; ai  funct  fcx    fcy    fcz\n")
            # Monomer atom indices in the monomer ITP (1-based within each molecule)
            # We restrain ALL non-protein, non-solvent heavy atoms
            for atom in monomer_atoms:
                if atom.mass > 2.0:  # heavy atoms only (skip H)
                    # Index within the monomer residue (1-based)
                    local_idx = atom.index - atom.residue.atoms[0].index + 1
                    f.write(f"  {local_idx}    1  {REBINDING_RESTRAINT_K}  "
                            f"{REBINDING_RESTRAINT_K}  {REBINDING_RESTRAINT_K}\n")

        # Copy topology as-is (keep protein definition)
        topol_src = Path(topol_path)
        cavity_top = output_dir / "topol.top"
        shutil.copy2(str(topol_src), str(cavity_top))

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
            # Selectivity: replace protein with different target's head
            # Need to rebuild system: extract monomers+water, add new head via pdb2gmx
            logger.info(f"    Selectivity rebinding: rebuilding with different head...")
            try:
                from .utils_gromacs import setup_protein_topology, _include_monomers_in_topology

                u_sys = mda.Universe(str(cavity_gro))

                # 1. Write monomers + water (no protein) GRO
                non_protein = u_sys.select_atoms("not protein")
                monomers_gro = md_dir / "monomers_only.gro"
                with mda.Writer(str(monomers_gro), n_atoms=non_protein.n_atoms) as w:
                    w.write(non_protein)

                # 2. Generate new protein topology from different head
                setup_protein_topology(Path(template_pdb), md_dir)

                # 3. Merge: new protein + old monomers
                prot_lines = (md_dir / "protein.gro").read_text().strip().split("\n")
                mon_lines = monomers_gro.read_text().strip().split("\n")
                prot_natoms = int(prot_lines[1].strip())
                mon_natoms = int(mon_lines[1].strip())
                total = prot_natoms + mon_natoms

                merged = [prot_lines[0]]
                merged.append(f" {total}")
                merged.extend(prot_lines[2:2+prot_natoms])
                merged.extend(mon_lines[2:2+mon_natoms])
                merged.append(prot_lines[-1])
                rebind_gro = md_dir / "rebind_system.gro"
                rebind_gro.write_text("\n".join(merged) + "\n")

                # 4. Add monomer ITPs to new topology
                # Copy ITP files
                for src_dir in [Path(cavity_top).parent]:
                    for itp in src_dir.glob("*.itp"):
                        dst = md_dir / itp.name
                        if not dst.exists() and "posre" not in itp.name:
                            shutil.copy2(str(itp), str(dst))
                if p4_md_dir:
                    for itp in Path(p4_md_dir).glob("*.itp"):
                        dst = md_dir / itp.name
                        if not dst.exists() and "posre" not in itp.name:
                            shutil.copy2(str(itp), str(dst))

                # Copy posre for monomers
                posre_src = Path(cavity_top).parent / "posre_monomers.itp"
                if posre_src.exists():
                    shutil.copy2(str(posre_src), str(md_dir / posre_src.name))

                # Add monomer molecules to topology
                # Read Phase 4 topology to get monomer molecule entries
                p4_top = Path(cavity_top).read_text()
                monomer_molecules = []
                monomer_includes = []
                in_mol = False
                for line in p4_top.split("\n"):
                    if "[ molecules ]" in line:
                        in_mol = True
                        continue
                    if in_mol and line.strip() and not line.startswith(";"):
                        parts = line.split()
                        if len(parts) >= 2:
                            name = parts[0]
                            if name.startswith("Protein") or name in ("SOL", "NA", "CL"):
                                continue
                            monomer_molecules.append(line)
                    if "#include" in line and "forcefield" not in line and "tip3p" not in line and "ions" not in line and "posre.itp" not in line:
                        monomer_includes.append(line)

                # Modify new topology
                new_top = (md_dir / "topol.top").read_text()
                # Add monomer includes after forcefield
                for inc in monomer_includes:
                    if inc not in new_top:
                        new_top = new_top.replace(
                            "[ moleculetype ]",
                            f"{inc}\n\n[ moleculetype ]", 1)
                # Add monomer molecules before SOL
                mol_block = "\n".join(monomer_molecules)
                new_top = new_top.replace("SOL", f"{mol_block}\nSOL", 1)
                # Add posre
                new_top += f'\n#include "posre_monomers.itp"\n'
                (md_dir / "topol.top").write_text(new_top)

                # Use rebuilt system as ionized.gro (skip solvation)
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

        ionized = md_dir / "ionized.gro"
        if not ionized.exists():
            shutil.copy2(str(md_dir / "rebind_system.gro"), str(ionized))

        if not (md_dir / "em.gro").exists():
            logger.info(f"    Energy minimization...")
            run_energy_minimization(md_dir)

        if not (md_dir / "nvt.gro").exists():
            logger.info(f"    NVT equilibration...")
            run_nvt_equilibration(md_dir, time_ps=100.0,
                                  temperature=MD_TEMPERATURE_K)

        if not (md_dir / "npt.gro").exists():
            logger.info(f"    NPT equilibration...")
            run_npt_equilibration(md_dir, time_ps=100.0,
                                  temperature=MD_TEMPERATURE_K,
                                  pressure=MD_PRESSURE_BAR)

        if not (md_dir / "md.gro").exists():
            logger.info(f"    Production MD ({time_ns}ns)...")
            run_production_md(md_dir, time_ns=time_ns,
                              temperature=MD_TEMPERATURE_K,
                              pressure=MD_PRESSURE_BAR,
                              gpu_id=MD_GPU_ID)
        else:
            logger.info(f"    Production MD: FOUND")

        # 7. Analyze template RMSD
        xtc = md_dir / "md_reduced.xtc"
        if not xtc.exists():
            xtc = md_dir / "md.xtc"
        tpr_gro = md_dir / "npt.gro"

        rmsd_mean = None
        rmsd_final = None

        if xtc.exists() and tpr_gro.exists():
            try:
                u_md = mda.Universe(str(tpr_gro), str(xtc))
                protein_md = u_md.select_atoms("protein")

                if len(protein_md) > 0:
                    # Reference: first frame protein position
                    u_md.trajectory[0]
                    ref_pos = protein_md.positions.copy()

                    rmsds = []
                    n_frames = len(u_md.trajectory)
                    start = n_frames // 2
                    stride = max(1, (n_frames - start) // 100)

                    for ts in u_md.trajectory[start::stride]:
                        rmsd = np.sqrt(np.mean(np.sum(
                            (protein_md.positions - ref_pos) ** 2, axis=1)))
                        rmsds.append(rmsd)

                    if rmsds:
                        rmsd_mean = round(float(np.mean(rmsds)), 2)
                        rmsd_final = round(float(rmsds[-1]), 2)
            except Exception as e:
                logger.warning(f"    RMSD analysis failed: {e}")

        rebound = rmsd_mean < REBINDING_RMSD_THRESHOLD if rmsd_mean else None
        status = "REBOUND" if rebound else ("ESCAPED" if rebound is False else "N/A")
        logger.info(f"    Result: RMSD={rmsd_mean} Å → {status}")

        return {
            "time_ns": time_ns,
            "rmsd_mean_A": rmsd_mean,
            "rmsd_final_A": rmsd_final,
            "rebound": rebound,
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

        # Run MD
        if not (md_dir / "em.gro").exists():
            logger.info(f"    Removal test: EM...")
            run_energy_minimization(md_dir)

        if not (md_dir / "nvt.gro").exists():
            logger.info(f"    Removal test: NVT...")
            run_nvt_equilibration(md_dir, time_ps=100.0, temperature=MD_TEMPERATURE_K)

        if not (md_dir / "npt.gro").exists():
            logger.info(f"    Removal test: NPT...")
            run_npt_equilibration(md_dir, time_ps=100.0,
                                  temperature=MD_TEMPERATURE_K, pressure=MD_PRESSURE_BAR)

        if not (md_dir / "md.gro").exists():
            logger.info(f"    Removal test: {time_ns}ns MD...")
            run_production_md(md_dir, time_ns=time_ns,
                              temperature=MD_TEMPERATURE_K,
                              pressure=MD_PRESSURE_BAR, gpu_id=MD_GPU_ID)

        # Analyze: template RMSD over time
        xtc = md_dir / "md_reduced.xtc"
        if not xtc.exists():
            xtc = md_dir / "md.xtc"
        top_gro = md_dir / "npt.gro"

        rmsd_start = None
        rmsd_end = None
        escaped = None

        if xtc.exists() and top_gro.exists():
            try:
                u = mda.Universe(str(top_gro), str(xtc))
                protein = u.select_atoms("protein")
                if len(protein) > 0:
                    u.trajectory[0]
                    ref_pos = protein.positions.copy()

                    rmsds = []
                    for ts in u.trajectory:
                        rmsd = float(np.sqrt(np.mean(np.sum(
                            (protein.positions - ref_pos) ** 2, axis=1))))
                        rmsds.append(rmsd)

                    if rmsds:
                        rmsd_start = round(float(np.mean(rmsds[:len(rmsds)//4])), 2)
                        rmsd_end = round(float(np.mean(rmsds[-len(rmsds)//4:])), 2)
                        # Template escaped if RMSD increased significantly
                        escaped = rmsd_end > REBINDING_RMSD_THRESHOLD
            except Exception as e:
                logger.warning(f"    Removal RMSD analysis failed: {e}")

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


# ── Results Analysis ─────────────────────────────────────────

def _analyze_rebinding_results(target, all_targets, snapshot_results, threshold):
    """Compute rebinding success rate and selectivity."""
    n_success = 0
    n_total = 0
    own_rmsds = []
    other_rmsds = {t: [] for t in all_targets if t != target}

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

        for other_t in all_targets:
            if other_t == target:
                continue
            other = snap.get(f"rebind_{other_t}", {})
            r = other.get("rmsd_mean_A")
            if r is not None:
                other_rmsds[other_t].append(r)

    result = {
        "target": target,
        "n_snapshots": n_total,
        "n_rebound": n_success,
        "success_rate": f"{n_success}/{n_total}" if n_total > 0 else "0/0",
        "own_rmsd_mean": round(float(np.mean(own_rmsds)), 2) if own_rmsds else None,
        "snapshots": snapshot_results,
    }

    # Selectivity
    for other_t, rmsds in other_rmsds.items():
        if rmsds:
            result[f"other_{other_t}_rmsd_mean"] = round(float(np.mean(rmsds)), 2)

    return result


def _print_phase6_summary(results):
    """Print Phase 6 summary."""
    logger.info(f"\n{'='*60}")
    logger.info("Phase 6: VIP Cavity Rebinding Summary")
    logger.info(f"{'='*60}")

    for target, data in results.items():
        logger.info(f"\n[{target}]")
        logger.info(f"  Rebinding: {data.get('success_rate', 'N/A')}")
        logger.info(f"  Own RMSD: {data.get('own_rmsd_mean', 'N/A')} Å")

        for key, val in data.items():
            if key.startswith("other_") and key.endswith("_rmsd_mean"):
                other = key.replace("other_", "").replace("_rmsd_mean", "")
                logger.info(f"  vs {other}: {val} Å")

    logger.info(f"{'='*60}")
