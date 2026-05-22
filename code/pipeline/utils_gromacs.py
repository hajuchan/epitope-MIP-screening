"""
GROMACS Utilities
=================
Wrappers for GROMACS MD simulation setup, execution,
trajectory analysis, and MM-PBSA binding free energy calculation.

Reference:
  Sullivan et al., J. Phys. Chem. B 2019 — MM-PBSA for MIP
  Rebelo et al., Int. J. Mol. Sci. 2023 — GROMACS + gmx_MMPBSA protocol
"""

import logging
import os
import subprocess
from pathlib import Path
from textwrap import dedent

logger = logging.getLogger(__name__)


# ── MDP Templates ──────────────────────────────────────────────

MDP_EM = dedent("""\
    ; Energy Minimization
    integrator  = steep
    emtol       = 1000.0
    emstep      = 0.01
    nsteps      = 50000
    nstlist     = 10
    cutoff-scheme = Verlet
    ns_type     = grid
    coulombtype = PME
    rcoulomb    = 1.0
    rvdw        = 1.0
    pbc         = xyz
""")

MDP_NVT = dedent("""\
    ; NVT Equilibration
    {define}
    integrator  = md
    nsteps      = {nsteps}
    dt          = {dt}
    nstxout-compressed = 5000
    nstenergy   = 5000
    nstlog      = 5000
    continuation = no
    constraint_algorithm = lincs
    constraints = h-bonds
    lincs_iter  = 1
    lincs_order = 4
    cutoff-scheme = Verlet
    ns_type     = grid
    nstlist     = 10
    coulombtype = PME
    rcoulomb    = 1.0
    rvdw        = 1.0
    pbc         = xyz
    tcoupl      = V-rescale
    tc-grps     = Protein Non-Protein
    tau_t       = 0.1 0.1
    ref_t       = {temperature} {temperature}
    pcoupl      = no
    gen_vel     = yes
    gen_temp    = {temperature}
    gen_seed    = -1
""")

MDP_NPT = dedent("""\
    ; NPT Equilibration
    {define}
    integrator  = md
    nsteps      = {nsteps}
    dt          = {dt}
    nstxout-compressed = 5000
    nstenergy   = 5000
    nstlog      = 5000
    continuation = yes
    constraint_algorithm = lincs
    constraints = h-bonds
    lincs_iter  = 1
    lincs_order = 4
    cutoff-scheme = Verlet
    ns_type     = grid
    nstlist     = 10
    coulombtype = PME
    rcoulomb    = 1.0
    rvdw        = 1.0
    pbc         = xyz
    tcoupl      = V-rescale
    tc-grps     = Protein Non-Protein
    tau_t       = 0.1 0.1
    ref_t       = {temperature} {temperature}
    pcoupl      = Parrinello-Rahman
    pcoupltype  = isotropic
    tau_p       = 2.0
    ref_p       = {pressure}
    compressibility = 4.5e-5
    refcoord_scaling = com
""")

MDP_PRODUCTION = dedent("""\
    ; Production MD
    {define}
    integrator  = md
    nsteps      = {nsteps}
    dt          = {dt}
    nstxout-compressed = 5000
    nstenergy   = 5000
    nstlog      = 5000
    continuation = yes
    constraint_algorithm = lincs
    constraints = h-bonds
    lincs_iter  = 1
    lincs_order = 4
    cutoff-scheme = Verlet
    ns_type     = grid
    nstlist     = 10
    coulombtype = PME
    rcoulomb    = 1.0
    rvdw        = 1.0
    pbc         = xyz
    tcoupl      = V-rescale
    tc-grps     = Protein Non-Protein
    tau_t       = 0.1 0.1
    ref_t       = {temperature} {temperature}
    pcoupl      = Parrinello-Rahman
    pcoupltype  = isotropic
    tau_p       = 2.0
    ref_p       = {pressure}
    compressibility = 4.5e-5
""")


# ── Monomer Parameterization ──────────────────────────────────

def parameterize_monomer(mol2_path: Path, name: str,
                          output_dir: Path,
                          charge_method: str = "bcc") -> dict:
    """
    Generate topology for a monomer.
    Si/B-containing monomers → PolCA force field (Jorge 2021).
    Others → acpype GAFF2.
    """
    from .config import ACPYPE_BIN, ALL_MONOMERS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mol2_path = Path(mol2_path)

    # Si/B monomers: always use PolCA (acpype generates zero Si parameters)
    m_info = ALL_MONOMERS.get(name)
    if m_info:
        from rdkit import Chem
        mol_check = Chem.MolFromSmiles(m_info["smiles"])
        if mol_check and any(a.GetAtomicNum() in (14, 5) for a in mol_check.GetAtoms()):
            logger.info(f"  {name}: Si/B detected → using PolCA force field")
            return _try_polca_fallback(name, mol2_path, output_dir)
    work_dir = output_dir / f"acpype_{name}"

    # Run acpype via current Python interpreter to ensure openbabel is available
    # (acpype shebang may point to system python without openbabel)
    import sys
    cmd = [
        sys.executable, ACPYPE_BIN,
        "-i", str(mol2_path),
        "-b", name,
        "-c", charge_method,
        "-n", "0",           # net charge
        "-a", "gaff2",
        "-o", "gmx",
    ]

    # Ensure conda env bin is in PATH (antechamber, sqm, tleap etc.)
    env = os.environ.copy()
    conda_bin = str(Path(sys.executable).parent)
    if conda_bin not in env.get("PATH", ""):
        env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")

    try:
        result = subprocess.run(
            cmd, cwd=str(output_dir),
            capture_output=True, text=True, timeout=600,  # 10min for large molecules
            env=env,
        )
        if result.returncode != 0:
            # Si/B monomers are expected to fail GAFF2 → PolCA fallback
            err_msg = result.stderr[:300] or result.stdout[-300:]
            logger.debug(f"acpype GAFF2 failed for {name} (rc={result.returncode}), trying fallback")
            return _try_polca_fallback(name, mol2_path, output_dir)
    except FileNotFoundError:
        logger.error(f"acpype not found: {ACPYPE_BIN}")
        return _try_polca_fallback(name, mol2_path, output_dir)

    # Find output files
    acpype_dir = output_dir / f"{name}.acpype"
    if not acpype_dir.exists():
        candidates = list(output_dir.glob(f"*{name}*acpype*"))
        acpype_dir = candidates[0] if candidates else output_dir

    itp_files = list(acpype_dir.glob("*_GMX.itp")) + \
                list(acpype_dir.glob("*.itp"))
    gro_files = list(acpype_dir.glob("*_GMX.gro")) + \
                list(acpype_dir.glob("*.gro"))

    return {
        "itp": str(itp_files[0]) if itp_files else None,
        "gro": str(gro_files[0]) if gro_files else None,
        "acpype_dir": str(acpype_dir),
    }


def _try_polca_fallback(name: str, mol2_path: Path, output_dir: Path) -> dict:
    """
    Fallback for Si/B-containing monomers: use PolCA force field.
    Jorge et al., ACS Phys. Chem. Au 2021 — organosilane parameters.
    """
    from .config import ALL_MONOMERS
    m_info = ALL_MONOMERS.get(name)
    if m_info is None:
        return {"error": f"Monomer {name} not in library"}

    smiles = m_info["smiles"]

    # Check if molecule contains Si or B
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES for {name}"}

    has_si = any(a.GetAtomicNum() == 14 for a in mol.GetAtoms())
    has_b = any(a.GetAtomicNum() == 5 for a in mol.GetAtoms())

    if has_si or has_b:
        try:
            logger.info(f"  {name}: using PolCA Si force field (acpype fallback)")
            return _generate_silane_itp(name, smiles, output_dir / f"{name}_polca")
        except Exception as e:
            logger.error(f"  PolCA fallback failed for {name}: {e}")
            return {"error": str(e)}
    else:
        return {"error": f"acpype failed and no fallback available for {name}"}


# ── System Setup ───────────────────────────────────────────────

def _gmx(cmd_args: list, work_dir: Path, input_text: str = None,
          timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a gmx command. Converts absolute paths to relative to handle spaces."""
    from .config import GMX_BIN
    work_dir = Path(work_dir)

    # Convert absolute paths in args to relative (avoids GROMACS space-in-path issues)
    rel_args = []
    for arg in cmd_args:
        if str(work_dir) in str(arg):
            try:
                rel_args.append(str(Path(arg).relative_to(work_dir)))
            except ValueError:
                rel_args.append(arg)
        else:
            rel_args.append(arg)

    # Ensure grompp writes mdout.mdp to work_dir (not project root)
    if rel_args and rel_args[0] == "grompp" and "-po" not in rel_args:
        rel_args.extend(["-po", "mdout.mdp"])

    full_cmd = [GMX_BIN] + rel_args
    result = subprocess.run(
        full_cmd, cwd=str(work_dir),
        input=input_text, capture_output=True, text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        logger.debug(f"gmx {rel_args[0]} rc={result.returncode}: "
                     f"{result.stderr[-200:] if result.stderr else ''}")

    return result


def setup_protein_topology(pdb_path: Path, work_dir: Path,
                            forcefield: str = "amber99sb-ildn",
                            water: str = "tip3p") -> Path:
    """
    Generate GROMACS topology for the protein epitope.
    Automatically fixes missing heavy atoms before pdb2gmx.
    Returns path to processed .gro file.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Fix missing heavy atoms (e.g., side chains not resolved in X-ray)
    fixed_pdb = _fix_missing_atoms(pdb_path, work_dir)

    result = _gmx([
        "pdb2gmx",
        "-f", str(fixed_pdb),
        "-o", str(work_dir / "protein.gro"),
        "-p", str(work_dir / "topol.top"),
        "-ignh",
        "-ff", forcefield,
        "-water", water,
    ], work_dir)

    if result.returncode != 0:
        logger.error(f"pdb2gmx failed: {result.stderr[:500]}")
        raise RuntimeError(f"pdb2gmx failed: {result.stderr[:300]}")

    return work_dir / "protein.gro"


def setup_simulation_box(gro_path: Path, work_dir: Path,
                          box_type: str = "cubic",
                          distance: float = 0.5) -> Path:
    """Create simulation box, solvate, and add ions.
    Uses cubic box with 0.5nm padding (compact, PBC handles periodicity)."""
    work_dir = Path(work_dir)

    # Define box — center system in box
    _gmx(["editconf",
           "-f", str(gro_path),
           "-o", str(work_dir / "boxed.gro"),
           "-c",                    # center in box
           "-d", str(distance),     # min distance to box edge
           "-bt", box_type], work_dir)

    # Solvate
    _gmx(["solvate",
           "-cp", str(work_dir / "boxed.gro"),
           "-cs", "spc216.gro",
           "-o", str(work_dir / "solvated.gro"),
           "-p", str(work_dir / "topol.top")], work_dir)

    # Add ions (neutralize)
    # First generate .tpr for genion
    mdp_path = work_dir / "ions.mdp"
    mdp_path.write_text(MDP_EM)
    _gmx(["grompp",
           "-f", str(mdp_path),
           "-c", str(work_dir / "solvated.gro"),
           "-p", str(work_dir / "topol.top"),
           "-o", str(work_dir / "ions.tpr"),
           "-maxwarn", "10"], work_dir)

    # Replace solvent with ions — PBS condition (0.15 M NaCl)
    from .config import MD_IONIC_STRENGTH
    _gmx(["genion",
           "-s", str(work_dir / "ions.tpr"),
           "-o", str(work_dir / "ionized.gro"),
           "-p", str(work_dir / "topol.top"),
           "-pname", "NA", "-nname", "CL",
           "-neutral",
           "-conc", str(MD_IONIC_STRENGTH)],  # 0.15 M NaCl (PBS)
          work_dir, input_text="SOL\n")

    return work_dir / "ionized.gro"


# ── MD Execution ───────────────────────────────────────────────

def run_energy_minimization(work_dir: Path) -> Path:
    """Run energy minimization."""
    work_dir = Path(work_dir)
    mdp_path = work_dir / "em.mdp"
    mdp_path.write_text(MDP_EM)

    _gmx(["grompp",
           "-f", str(mdp_path),
           "-c", str(work_dir / "ionized.gro"),
           "-p", str(work_dir / "topol.top"),
           "-o", str(work_dir / "em.tpr"),
           "-maxwarn", "10"], work_dir)

    result = _gmx(["mdrun", "-deffnm", "em"],
                   work_dir, timeout=1800)
    if result.returncode != 0:
        logger.warning(f"EM mdrun issue: {result.stderr[:300]}")

    return work_dir / "em.gro"


def run_nvt_equilibration(work_dir: Path, time_ps: float = 100.0, define: str = "",
                           temperature: float = 300.0) -> Path:
    """Run NVT equilibration."""
    work_dir = Path(work_dir)
    from .config import MD_TIMESTEP_FS
    dt = MD_TIMESTEP_FS / 1000.0  # fs to ps
    nsteps = int(time_ps / dt)

    mdp_path = work_dir / "nvt.mdp"
    mdp_path.write_text(MDP_NVT.format(define=define,
        nsteps=nsteps, dt=dt, temperature=temperature))

    _gmx(["grompp",
           "-f", str(mdp_path),
           "-c", str(work_dir / "em.gro"),
           "-r", str(work_dir / "em.gro"),
           "-p", str(work_dir / "topol.top"),
           "-o", str(work_dir / "nvt.tpr"),
           "-maxwarn", "10"], work_dir)

    _gmx(["mdrun", "-deffnm", "nvt"], work_dir, timeout=3600)
    # Extract final frame from checkpoint/trajectory if gro not created
    nvt_gro = work_dir / "nvt.gro"
    if not nvt_gro.exists() and (work_dir / "nvt.cpt").exists():
        _gmx(["trjconv", "-f", "nvt.cpt", "-s", "nvt.tpr",
               "-o", "nvt.gro", "-dump", "0"],
              work_dir, input_text="0\n")
    return nvt_gro


def run_npt_equilibration(work_dir: Path, time_ps: float = 100.0, define: str = "",
                           temperature: float = 300.0,
                           pressure: float = 1.0) -> Path:
    """Run NPT equilibration."""
    work_dir = Path(work_dir)
    from .config import MD_TIMESTEP_FS
    dt = MD_TIMESTEP_FS / 1000.0
    nsteps = int(time_ps / dt)

    mdp_path = work_dir / "npt.mdp"
    mdp_path.write_text(MDP_NPT.format(define=define,
        nsteps=nsteps, dt=dt, temperature=temperature, pressure=pressure))

    _gmx(["grompp",
           "-f", str(mdp_path),
           "-c", str(work_dir / "nvt.gro"),
           "-r", str(work_dir / "nvt.gro"),
           "-t", str(work_dir / "nvt.cpt"),
           "-p", str(work_dir / "topol.top"),
           "-o", str(work_dir / "npt.tpr"),
           "-maxwarn", "10"], work_dir)

    _gmx(["mdrun", "-deffnm", "npt"], work_dir, timeout=3600)
    npt_gro = work_dir / "npt.gro"
    if not npt_gro.exists() and (work_dir / "npt.cpt").exists():
        _gmx(["trjconv", "-f", "npt.cpt", "-s", "npt.tpr",
               "-o", "npt.gro", "-dump", "0"],
              work_dir, input_text="0\n")
    return npt_gro


def run_production_md(work_dir: Path, time_ns: float = 200.0, define: str = "",
                       temperature: float = 300.0,
                       pressure: float = 1.0,
                       gpu_id: str = "0") -> Path:
    """Run production MD simulation."""
    work_dir = Path(work_dir)
    from .config import MD_TIMESTEP_FS
    dt = MD_TIMESTEP_FS / 1000.0
    nsteps = int(time_ns * 1000.0 / dt)  # ns → ps → steps

    mdp_path = work_dir / "md.mdp"
    mdp_path.write_text(MDP_PRODUCTION.format(define=define,
        nsteps=nsteps, dt=dt, temperature=temperature, pressure=pressure))

    _gmx(["grompp",
           "-f", str(mdp_path),
           "-c", str(work_dir / "npt.gro"),
           "-r", str(work_dir / "npt.gro"),
           "-t", str(work_dir / "npt.cpt"),
           "-p", str(work_dir / "topol.top"),
           "-o", str(work_dir / "md.tpr"),
           "-maxwarn", "10"], work_dir)

    md_cmd = ["mdrun", "-deffnm", "md", "-v"]

    # Resume from checkpoint if available (interrupted run)
    md_cpt = work_dir / "md.cpt"
    if md_cpt.exists():
        md_cmd.extend(["-cpi", "md.cpt", "-append"])
        logger.info(f"Resuming production MD from checkpoint in {work_dir}")
    else:
        logger.info(f"Starting {time_ns}ns production MD in {work_dir}")

    from .config import USE_GPU
    if USE_GPU:
        md_cmd.extend([
            "-nb", "gpu",
            "-pme", "gpu",
            "-bonded", "gpu",
            "-update", "gpu",
            "-gpu_id", gpu_id,
        ])

    # Production MD: show real-time progress (-v output)
    from .config import GMX_BIN
    full_cmd = [GMX_BIN] + md_cmd
    subprocess.run(full_cmd, cwd=str(work_dir), timeout=int(time_ns * 3600))
    return work_dir / "md.xtc"


# ── Trajectory Analysis ────────────────────────────────────────

def analyze_trajectory(work_dir: Path) -> dict:
    """
    Analyze production MD trajectory.
    Returns dict with RMSD, RMSF, H-bond, Rg metrics.
    """
    work_dir = Path(work_dir)
    results = {}

    import numpy as np

    # First, create a stride-reduced trajectory for analysis (every 100th frame)
    reduced_xtc = work_dir / "md_reduced.xtc"
    if not reduced_xtc.exists() and (work_dir / "md.xtc").exists():
        logger.info("  [1/5] Creating reduced trajectory (stride 100)...")
        _gmx(["trjconv", "-f", "md.xtc", "-s", "md.tpr",
               "-o", "md_reduced.xtc", "-skip", "100"],
              work_dir, input_text="0\n", timeout=300)
    else:
        logger.info("  [1/5] Reduced trajectory: FOUND")

    xtc_for_analysis = str(reduced_xtc) if reduced_xtc.exists() else "md.xtc"

    # RMSD
    if not (work_dir / "rmsd.xvg").exists():
        logger.info("  [2/5] RMSD analysis...")
        try:
            _gmx(["rms", "-f", xtc_for_analysis, "-s", "md.tpr",
                   "-o", "rmsd.xvg"],
                  work_dir, input_text="Backbone\nBackbone\n")
        except Exception as e:
            logger.warning(f"  [2/5] RMSD failed: {e}")
    else:
        logger.info("  [2/5] RMSD: FOUND")

    rmsd_data = _parse_xvg(work_dir / "rmsd.xvg")
    if rmsd_data is not None and len(rmsd_data) > 0:
        results["rmsd_mean_nm"] = float(np.mean(rmsd_data[:, 1]))
        half_time = rmsd_data[-1, 0] / 2
        last_half = rmsd_data[rmsd_data[:, 0] > half_time]
        if len(last_half) > 0:
            results["rmsd_last50ns_mean_nm"] = float(np.mean(last_half[:, 1]))

    # RMSF
    if not (work_dir / "rmsf.xvg").exists():
        logger.info("  [3/5] RMSF analysis...")
        try:
            _gmx(["rmsf", "-f", xtc_for_analysis, "-s", "md.tpr",
                   "-o", "rmsf.xvg", "-res"],
                  work_dir, input_text="Backbone\n")
        except Exception as e:
            logger.warning(f"  [3/5] RMSF failed: {e}")
    else:
        logger.info("  [3/5] RMSF: FOUND")

    rmsf_data = _parse_xvg(work_dir / "rmsf.xvg")
    if rmsf_data is not None and len(rmsf_data) > 0:
        results["rmsf_mean_nm"] = float(np.mean(rmsf_data[:, 1]))
        results["rmsf_max_nm"] = float(np.max(rmsf_data[:, 1]))

    # H-bonds
    if not (work_dir / "hbond.xvg").exists():
        logger.info("  [4/5] H-bond analysis...")
        try:
            _gmx(["hbond", "-f", xtc_for_analysis, "-s", "md.tpr",
                   "-r", "group 1", "-t", "not group 1",
                   "-num", "hbond.xvg"],
                  work_dir, timeout=600)
        except Exception as e:
            logger.warning(f"  [4/5] H-bond failed: {e}")
    else:
        logger.info("  [4/5] H-bond: FOUND")

    hb_data = _parse_xvg(work_dir / "hbond.xvg")
    if hb_data is not None and len(hb_data) > 0:
        results["hbond_mean"] = float(np.mean(hb_data[:, 1]))
        results["hbond_max"] = float(np.max(hb_data[:, 1]))

    # Radius of gyration
    if not (work_dir / "gyrate.xvg").exists():
        logger.info("  [5/5] Radius of gyration...")
        try:
            _gmx(["gyrate", "-f", xtc_for_analysis, "-s", "md.tpr",
                   "-o", "gyrate.xvg"],
                  work_dir, input_text="Protein\n")
        except Exception as e:
            logger.warning(f"  [5/5] Rg failed: {e}")
    else:
        logger.info("  [5/5] Rg: FOUND")

    rg_data = _parse_xvg(work_dir / "gyrate.xvg")
    if rg_data is not None and len(rg_data) > 0:
        results["rg_mean_nm"] = float(np.mean(rg_data[:, 1]))

    # Sullivan 2019 / Sehit 2024: DSSP secondary structure
    from .config import DSSP_ANALYSIS
    if DSSP_ANALYSIS:
        try:
            logger.info("  DSSP secondary structure analysis...")
            from .utils_analysis import analyze_dssp_changes
            # Use npt.gro as topology (avoid editconf hang on large tpr)
            gro = work_dir / "npt.gro"
            if not gro.exists():
                gro = work_dir / "em.gro"
            if gro.exists() and reduced_xtc.exists():
                dssp = analyze_dssp_changes(reduced_xtc, gro)
                results["dssp"] = dssp
                if dssp.get("structure_preserved") is False:
                    logger.warning(
                        f"  2° structure NOT preserved "
                        f"(helix change: {dssp.get('helix_change', 'N/A')})")
                else:
                    logger.info("  DSSP: structure preserved")
            else:
                logger.info("  DSSP: skipped (missing files)")
        except ImportError:
            logger.info("  DSSP: mdtraj not available, skipped")
        except Exception as e:
            logger.warning(f"  DSSP failed: {e}")

    logger.info("  Trajectory analysis complete")

    return results


# ── MM-PBSA ────────────────────────────────────────────────────

def run_mmpbsa(work_dir: Path, start_ns: float = 150.0,
               end_ns: float = 200.0,
               n_frames: int = 100,
               decomp: bool = False) -> dict:
    """
    Run gmx_MMPBSA for binding free energy calculation.

    Sullivan 2019: MM-GBSA is preferred for protein-monomer systems.
    Supports both PBSA and GBSA modes via config.MMPBSA_METHOD.

    Requires gmx_MMPBSA to be installed (pip install gmx_MMPBSA).
    """
    from .config import MMPBSA_METHOD, MD_IONIC_STRENGTH
    work_dir = Path(work_dir)

    # Create MMPBSA input file — GBSA (Sullivan 2019) or PBSA
    # Ionic strength matches MD simulation (PBS 0.15 M)
    mmpbsa_in = work_dir / "mmpbsa.in"
    # Per-residue decomposition (Kumar et al. 2024)
    decomp_block = ""
    if decomp:
        decomp_block = dedent("""\
            &decomp
              idecomp=2, dec_verbose=1,
              print_res="within 6"
            /
        """)

    if MMPBSA_METHOD == "GBSA":
        mmpbsa_in.write_text(dedent(f"""\
            &general
              startframe=1, endframe={n_frames}, interval=1,
              verbose=2,
            /
            &gb
              igb=5, saltcon={MD_IONIC_STRENGTH},
            /
        """) + decomp_block)
    else:
        mmpbsa_in.write_text(dedent(f"""\
            &general
              startframe=1, endframe={n_frames}, interval=1,
              verbose=2,
            /
            &pb
              istrng={MD_IONIC_STRENGTH}, fillratio=4.0,
            /
        """) + decomp_block)

    # Generate index file: need "Protein" and "Other" (monomers only) groups
    ndx_path = work_dir / "index.ndx"
    if not ndx_path.exists():
        _gmx(["make_ndx", "-f", "md.tpr", "-o", "index.ndx"],
              work_dir, input_text="q\n")

    # Find the "Other" group number (monomers, not water/ions)
    # Default GROMACS: Protein=1, Other=12 (may vary)
    ligand_group = "12"
    if ndx_path.exists():
        ndx_text = ndx_path.read_text()
        # Parse group names to find "Other"
        import re as _re
        for match in _re.finditer(r'\[\s*(\S+)\s*\]', ndx_text):
            pass  # just checking existence
        # Use make_ndx output to find group number
        result = _gmx(["make_ndx", "-f", "md.tpr", "-n", "index.ndx"],
                       work_dir, input_text="q\n")
        if result.stdout:
            for line in result.stdout.split("\n"):
                if "Other" in line and line.strip()[0].isdigit():
                    ligand_group = line.strip().split()[0]
                    break

    # gmx_MMPBSA command
    cmd = [
        "gmx_MMPBSA",
        "-O",
        "-i", str(mmpbsa_in),
        "-cs", str(work_dir / "md.tpr"),
        "-ct", str(work_dir / "md.xtc"),
        "-ci", str(ndx_path),
        "-cg", "1", ligand_group,
        "-cp", str(work_dir / "topol.top"),
        "-eo", str(work_dir / "FINAL_RESULTS_MMPBSA.csv"),
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(work_dir),
            capture_output=True, text=True, timeout=7200,
        )
        if result.returncode != 0:
            logger.warning(f"gmx_MMPBSA issue: {result.stderr[:500]}")

        # Parse results
        final_dat = work_dir / "FINAL_RESULTS_MMPBSA.dat"
        if final_dat.exists():
            return _parse_mmpbsa_results(final_dat)
        return {"error": "No results file produced",
                "stderr": result.stderr[:300]}
    except FileNotFoundError:
        return {"error": "gmx_MMPBSA not found"}
    except subprocess.TimeoutExpired:
        return {"error": "gmx_MMPBSA timeout"}


# ── Full MD Pipeline ───────────────────────────────────────────

def run_full_md_pipeline(protein_pdb: Path, monomer_itps: list,
                          work_dir: Path,
                          time_ns: float = 200.0,
                          quick: bool = False,
                          protein_restrained: bool = False) -> dict:
    """
    Complete GROMACS MD pipeline:
    pdb2gmx → solvate → EM → NVT → NPT → production → analysis → MM-PBSA
    """
    from .config import (MD_TEMPERATURE_K, MD_PRESSURE_BAR,
                         MD_GPU_ID, MD_QUICK_NS,
                         MD_MMPBSA_START_NS, MD_MMPBSA_END_NS,
                         MD_MMPBSA_INTERVAL)

    if quick:
        time_ns = MD_QUICK_NS

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results = {"work_dir": str(work_dir), "time_ns": time_ns}

    try:
        # Each step checks if output exists → skip if already done

        # 1. Protein topology
        if not (work_dir / "topol.top").exists():
            logger.info("Setting up protein topology...")
            setup_protein_topology(protein_pdb, work_dir)
        else:
            logger.info("Protein topology: FOUND, skipping")

        # 1b. Include monomer ITP/GRO in topology
        if not (work_dir / "complex.gro").exists():
            if monomer_itps:
                logger.info(f"Including {len(monomer_itps)} monomer(s) in topology...")
                _include_monomers_in_topology(work_dir, monomer_itps)
                system_gro = work_dir / "complex.gro"
            else:
                system_gro = work_dir / "protein.gro"
        else:
            logger.info("Complex GRO: FOUND, skipping")
            system_gro = work_dir / "complex.gro"

        # 2. Solvate & ionize
        if not (work_dir / "ionized.gro").exists():
            logger.info("Setting up simulation box...")
            setup_simulation_box(system_gro, work_dir)
        else:
            logger.info("Ionized system: FOUND, skipping")

        # 3. Energy minimization
        if not (work_dir / "em.gro").exists():
            logger.info("Running energy minimization...")
            run_energy_minimization(work_dir)
        else:
            logger.info("EM: FOUND, skipping")

        # 4. NVT equilibration (100 ps)
        if not (work_dir / "nvt.gro").exists():
            logger.info("NVT equilibration...")
            run_nvt_equilibration(work_dir, time_ps=100.0,
                                   temperature=MD_TEMPERATURE_K)
        else:
            logger.info("NVT: FOUND, skipping")

        # 5. NPT equilibration (100 ps)
        if not (work_dir / "npt.gro").exists():
            logger.info("NPT equilibration...")
            run_npt_equilibration(work_dir, time_ps=100.0,
                                   temperature=MD_TEMPERATURE_K,
                                   pressure=MD_PRESSURE_BAR)
        else:
            logger.info("NPT: FOUND, skipping")

        # 6. Production MD
        if not (work_dir / "md.gro").exists():
            define = "define = -DPOSRES" if protein_restrained else ""
            if protein_restrained:
                logger.info(f"Production MD ({time_ns} ns) with -DPOSRES "
                            "(protein heavy atoms restrained, surface MIP mode)...")
            else:
                logger.info(f"Production MD ({time_ns} ns)...")
            run_production_md(work_dir, time_ns=time_ns,
                               temperature=MD_TEMPERATURE_K,
                               pressure=MD_PRESSURE_BAR,
                               gpu_id=MD_GPU_ID,
                               define=define)
        else:
            logger.info(f"Production MD: FOUND, skipping")

        # 7. Trajectory analysis
        logger.info("Analyzing trajectory...")
        analysis = analyze_trajectory(work_dir)
        results.update(analysis)

        # 8. MM-PBSA
        logger.info("Running MM-PBSA...")
        mmpbsa = run_mmpbsa(
            work_dir,
            start_ns=MD_MMPBSA_START_NS if not quick else time_ns - 10,
            end_ns=MD_MMPBSA_END_NS if not quick else time_ns,
            n_frames=MD_MMPBSA_INTERVAL,
        )
        results["mmpbsa"] = mmpbsa

        results["success"] = True
    except Exception as e:
        logger.error(f"MD pipeline failed: {e}")
        results["success"] = False
        results["error"] = str(e)

    return results


# ── Internal Helpers ───────────────────────────────────────────

def _fix_missing_atoms(pdb_path: Path, work_dir: Path) -> Path:
    """
    Fix missing heavy atoms in PDB (e.g., unresolved side chains in X-ray).
    Uses pdbfixer (OpenMM) if available, otherwise returns original PDB.
    """
    fixed_path = work_dir / f"{Path(pdb_path).stem}_fixed.pdb"

    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        fixer = PDBFixer(filename=str(pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)  # pH 7.4

        with open(str(fixed_path), "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        n_added = len(fixer.missingAtoms)
        if n_added > 0:
            logger.info(f"Fixed {n_added} residue(s) with missing atoms → {fixed_path.name}")
        return fixed_path

    except ImportError:
        logger.warning("pdbfixer not installed (pip install pdbfixer openmm). "
                       "Using original PDB — pdb2gmx may fail if atoms are missing.")
        return pdb_path
    except Exception as e:
        logger.warning(f"pdbfixer failed: {e}. Using original PDB.")
        return pdb_path


def _parse_xvg(xvg_path: Path):
    """Parse GROMACS .xvg file, returns numpy array or None."""
    import numpy as np
    xvg_path = Path(xvg_path)
    if not xvg_path.exists():
        return None
    data = []
    for line in xvg_path.read_text().split("\n"):
        if line.startswith(("#", "@")) or not line.strip():
            continue
        parts = line.split()
        try:
            data.append([float(x) for x in parts])
        except ValueError:
            continue
    return np.array(data) if data else None


def _include_monomers_in_topology(work_dir: Path, monomer_itps: list):
    """
    Include monomer ITP files and coordinates in GROMACS topology.

    For each monomer:
    1. Copy .itp to work_dir
    2. Add #include to topol.top (before [ molecules ])
    3. Add molecule name to [ molecules ] section
    4. Merge .gro coordinates into system (protein.gro → complex.gro)
    """
    import shutil

    work_dir = Path(work_dir)
    top_path = work_dir / "topol.top"
    prot_gro = work_dir / "protein.gro"

    if not top_path.exists() or not prot_gro.exists():
        logger.warning("topol.top or protein.gro not found, skipping monomer inclusion")
        return

    # Read protein GRO
    prot_lines = prot_gro.read_text().strip().split("\n")
    prot_natoms = int(prot_lines[1].strip())
    coord_lines = prot_lines[2:2+prot_natoms]
    box_line = prot_lines[-1]

    # Compute protein center and radius for monomer placement
    prot_center = _get_gro_center(coord_lines)
    prot_radius = _get_gro_radius(coord_lines, prot_center)
    logger.info(f"  Protein center: ({prot_center[0]:.2f}, {prot_center[1]:.2f}, "
                f"{prot_center[2]:.2f}) nm, radius: {prot_radius:.2f} nm")

    # Pre-compute non-overlapping monomer positions (PACKMOL-style)
    import random, math
    n_monomers = len(monomer_itps)
    min_dist_nm = 1.0  # minimum distance between monomer centers
    r_inner = prot_radius + 0.3  # start just outside protein surface
    r_outer = r_inner + 1.0      # thin shell → 9nm box target (Rajpal 2024)

    placed_centers = []
    monomer_positions = []  # pre-computed (x, y, z) for each monomer

    for mi in range(n_monomers):
        # Try to place without overlapping other monomers
        for attempt in range(500):
            angle1 = random.uniform(0, 2 * math.pi)
            angle2 = random.uniform(-math.pi/2, math.pi/2)
            r = random.uniform(r_inner, r_outer)
            x = prot_center[0] + r * math.cos(angle1) * math.cos(angle2)
            y = prot_center[1] + r * math.sin(angle1) * math.cos(angle2)
            z = prot_center[2] + r * math.sin(angle2)

            # Check distance to all already placed monomers
            too_close = False
            for px, py, pz in placed_centers:
                d = math.sqrt((x-px)**2 + (y-py)**2 + (z-pz)**2)
                if d < min_dist_nm:
                    too_close = True
                    break

            if not too_close:
                placed_centers.append((x, y, z))
                monomer_positions.append((x, y, z))
                break
        else:
            # Fallback: expand radius and place
            r = r_outer + mi * 0.3
            angle1 = random.uniform(0, 2 * math.pi)
            angle2 = random.uniform(-math.pi/2, math.pi/2)
            x = prot_center[0] + r * math.cos(angle1) * math.cos(angle2)
            y = prot_center[1] + r * math.sin(angle1) * math.cos(angle2)
            z = prot_center[2] + r * math.sin(angle2)
            placed_centers.append((x, y, z))
            monomer_positions.append((x, y, z))

    logger.info(f"  Placed {len(monomer_positions)} monomers in shell "
                f"r={r_inner:.1f}-{r_outer:.1f} nm (min_sep={min_dist_nm} nm)")

    # Collect monomer coordinates and topology edits
    seen_itps = {}   # itp_name → (mol_name, itp_src)
    molecule_counts = {}  # mol_name → count
    all_mon_coords = []

    for i, param in enumerate(monomer_itps):
        itp_path = param.get("itp")
        gro_path = param.get("gro")
        if not itp_path or not Path(itp_path).exists():
            continue

        itp_src = Path(itp_path)
        mol_name = itp_src.stem.replace("_GMX", "")

        # Copy ITP once, count molecules
        if itp_src.name not in seen_itps:
            shutil.copy2(str(itp_src), str(work_dir / itp_src.name))
            seen_itps[itp_src.name] = (mol_name, itp_src)
        molecule_counts[mol_name] = molecule_counts.get(mol_name, 0) + 1

        # Place monomer at pre-computed non-overlapping position
        if gro_path and Path(gro_path).exists():
            mon_lines = Path(gro_path).read_text().strip().split("\n")
            mon_natoms = int(mon_lines[1].strip())
            mon_coords = mon_lines[2:2+mon_natoms]
            mon_center = _get_gro_center(mon_coords)

            tx, ty, tz = monomer_positions[i]
            xo = tx - mon_center[0]
            yo = ty - mon_center[1]
            zo = tz - mon_center[2]

            offset_coords = _offset_gro_coords(mon_coords,
                                                x_offset=xo, y_offset=yo, z_offset=zo)
            all_mon_coords.extend(offset_coords)

    # Edit topol.top — proper directive ordering:
    # 1. Extract [ atomtypes ] from ITPs → insert after forcefield.itp
    # 2. Strip [ atomtypes ] from ITPs, include after forcefield.itp
    # 3. Add molecule counts to [ molecules ]
    content = top_path.read_text()
    lines = content.split("\n")

    # Collect all atomtypes from monomer ITPs
    all_atomtypes = []
    for itp_name, (mol_name, itp_src) in seen_itps.items():
        itp_content = (work_dir / itp_name).read_text()
        # Extract [ atomtypes ] section
        in_atomtypes = False
        cleaned_lines = []
        for line in itp_content.split("\n"):
            if line.strip() == "[ atomtypes ]":
                in_atomtypes = True
                continue
            elif line.strip().startswith("[") and in_atomtypes:
                in_atomtypes = False
            if in_atomtypes:
                if line.strip() and not line.strip().startswith(";"):
                    all_atomtypes.append(line)
            else:
                cleaned_lines.append(line)
        # Fix Si atoms with mass=0 (acpype doesn't know Si mass)
        fixed_lines = []
        for line in cleaned_lines:
            if "[ atoms ]" not in line and "Si" in line and "0.00000" in line:
                # Check if this is an atoms line with mass=0 for Si
                parts = line.split()
                if len(parts) >= 8 and parts[1] == "Si" and float(parts[7]) == 0:
                    parts[7] = "28.08600"
                    line = "  ".join(parts)
            fixed_lines.append(line)
        (work_dir / itp_name).write_text("\n".join(fixed_lines))

    # Build atomtypes block (deduplicate by atom name)
    # Fix Si atom: acpype generates Si with mass=0, sigma=0 — replace with PolCA
    seen_atoms = set()
    unique_atomtypes = []
    for line in all_atomtypes:
        parts = line.split()
        atom_name = parts[0] if parts else ""
        if not atom_name or atom_name in seen_atoms:
            continue
        seen_atoms.add(atom_name)
        # Replace broken Si atomtype with PolCA parameters
        if atom_name == "Si" and len(parts) >= 7:
            sigma = float(parts[5]) if parts[5] != "0.00000e+00" else 0
            if sigma == 0:  # acpype generated empty Si params
                line = (f" Si  Si  {28.086:.3f}  0.000  A  "
                        f"4.29500e-01  4.02000e-01")  # UFF Si_3
                logger.info("  Fixed Si atomtype: acpype → UFF Si_3 parameters")
        unique_atomtypes.append(line)

    atomtypes_block = ""
    if unique_atomtypes:
        atomtypes_block = "\n[ atomtypes ]\n" + "\n".join(unique_atomtypes) + "\n"

    # Insert atomtypes after forcefield.itp include
    include_block = "\n".join(f'#include "{name}"' for name in seen_itps)
    molecule_block = "\n".join(f"{mol}     {cnt}"
                               for mol, cnt in molecule_counts.items())

    # Find insertion point: after #include "...forcefield.itp"
    new_lines = []
    ff_inserted = False
    for line in lines:
        new_lines.append(line)
        if not ff_inserted and "forcefield.itp" in line:
            new_lines.append(atomtypes_block)
            new_lines.append(include_block)
            ff_inserted = True

    content = "\n".join(new_lines)

    # Add molecules to [ molecules ] section
    if "[ molecules ]" in content:
        content = content.rstrip() + "\n" + molecule_block + "\n"
    else:
        content += f"\n[ molecules ]\n{molecule_block}\n"

    top_path.write_text(content)

    # Write complex.gro (protein + all monomers)
    total_atoms = prot_natoms + len(all_mon_coords)
    complex_gro = work_dir / "complex.gro"
    out_lines = [prot_lines[0]]  # title
    out_lines.append(f" {total_atoms}")
    out_lines.extend(coord_lines)
    out_lines.extend(all_mon_coords)
    out_lines.append(box_line)
    complex_gro.write_text("\n".join(out_lines) + "\n")

    logger.info(f"Topology updated: {len(monomer_itps)} monomers, "
                f"{total_atoms} total atoms → {complex_gro}")


def _pdbqt_to_gro_coords(pdbqt_path: Path, mol_name: str) -> list:
    """Extract coordinates from PDBQT docked pose → GRO format lines.
    PDBQT coordinates are in Angstroms, GRO in nm."""
    lines = Path(pdbqt_path).read_text().strip().split("\n")
    gro_lines = []
    atom_idx = 0
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_idx += 1
            atom_name = line[12:16].strip()
            x = float(line[30:38]) / 10.0  # Å → nm
            y = float(line[38:46]) / 10.0
            z = float(line[46:54]) / 10.0
            gro_lines.append(
                f"{1:5d}{mol_name:>5s}{atom_name:>5s}{atom_idx:5d}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}")
    return gro_lines if gro_lines else None


def _get_pdbqt_center(pdbqt_path: Path) -> tuple:
    """Get center of mass from PDBQT file (returns nm)."""
    lines = Path(pdbqt_path).read_text().strip().split("\n")
    xs, ys, zs = [], [], []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            xs.append(float(line[30:38]) / 10.0)
            ys.append(float(line[38:46]) / 10.0)
            zs.append(float(line[46:54]) / 10.0)
    if xs:
        return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
    return None


def _get_gro_center(coord_lines: list) -> tuple:
    """Get center of mass from GRO coordinate lines (nm)."""
    xs, ys, zs = [], [], []
    for line in coord_lines:
        try:
            xs.append(float(line[20:28]))
            ys.append(float(line[28:36]))
            zs.append(float(line[36:44]))
        except (ValueError, IndexError):
            pass
    if xs:
        return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
    return (0, 0, 0)


def _get_gro_radius(coord_lines: list, center: tuple) -> float:
    """Get max distance from center to any atom (nm)."""
    import math
    max_r = 0
    for line in coord_lines:
        try:
            x = float(line[20:28]) - center[0]
            y = float(line[28:36]) - center[1]
            z = float(line[36:44]) - center[2]
            r = math.sqrt(x*x + y*y + z*z)
            if r > max_r:
                max_r = r
        except (ValueError, IndexError):
            pass
    return max_r


def _offset_gro_coords(coord_lines: list, x_offset: float = 2.0,
                        y_offset: float = 0.0, z_offset: float = 0.0) -> list:
    """Offset GRO coordinate lines by xyz offsets (nm)."""
    shifted = []
    for line in coord_lines:
        try:
            prefix = line[:20]
            x = float(line[20:28]) + x_offset
            y = float(line[28:36]) + y_offset
            z = float(line[36:44]) + z_offset
            rest = line[44:] if len(line) > 44 else ""
            shifted.append(f"{prefix}{x:8.3f}{y:8.3f}{z:8.3f}{rest}")
        except (ValueError, IndexError):
            shifted.append(line)
    return shifted


# ══════════════════════════════════════════════════════════════
# PolCA Organosilane Force Field (Jorge et al., ACS Phys. Chem. Au 2021)
# ══════════════════════════════════════════════════════════════
# GAFF2/acpype cannot handle Si. PolCA provides GROMACS-compatible
# parameters for organosilane molecules.

_POLCA_SI_LJ = {
    "Si0": {"sigma": 0.580, "eps": 0.108},  # 4 alkyl
    "Si1": {"sigma": 0.551, "eps": 0.108},  # 3 alkyl, 1 O
    "Si2": {"sigma": 0.522, "eps": 0.108},  # 2 alkyl, 2 O
    "Si3": {"sigma": 0.493, "eps": 0.108},  # 1 alkyl, 3 O
    "Si4": {"sigma": 0.464, "eps": 0.108},  # 0 alkyl, 4 O (TEOS-like)
}


def _classify_si_type(smiles: str) -> str:
    """Determine Si type from SMILES by counting O neighbors."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Si4"
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 14:
            n_o = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 8)
            return f"Si{n_o}" if f"Si{n_o}" in _POLCA_SI_LJ else "Si4"
    return "Si4"


def _generate_silane_itp(name: str, smiles: str, output_dir: Path) -> dict:
    """Generate GROMACS .itp/.gro for silane using PolCA Si + OPLS-AA organic."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.useRandomCoords = True; p.randomSeed = 42
    if AllChem.EmbedMolecule(mol, p) != 0:
        return {"error": "3D embedding failed"}
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    # Charges: Si→S proxy for Gasteiger
    rw = Chem.RWMol(mol)
    si_idx = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 14]
    for idx in si_idx:
        rw.GetAtomWithIdx(idx).SetAtomicNum(16)
    AllChem.ComputeGasteigerCharges(rw)
    charges = [float(rw.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
               for i in range(rw.GetNumAtoms())]
    for idx in si_idx:
        charges[idx] = 0.9

    si_type = _classify_si_type(smiles)
    si_lj = _POLCA_SI_LJ[si_type]
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    # Use GAFF2 atom types (compatible with amber99sb-ildn for MD)
    etype = {6: "c3", 1: "h1", 8: "oh",
             7: "n3", 14: si_type, 16: "ss", 5: "c3"}
    mmap = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999,
            14: 28.086, 16: 32.065, 5: 10.811}

    alines = []
    for i in range(n_atoms):
        a = mol.GetAtomWithIdx(i)
        e = a.GetAtomicNum()
        alines.append(
            f"    {i+1:5d} {etype.get(e,'opls_135'):>10s} 1    {name:>5s} "
            f"{a.GetSymbol()+str(i+1):>5s} {i+1:5d} {charges[i]:10.4f} "
            f"{mmap.get(e,12.011):10.4f}")

    itp_path = output_dir / f"{name}.itp"
    # Include [ atomtypes ] for Si + all GAFF2 types used by this molecule
    _gaff2_lj = {
        "c3": (0.33977, 0.45104), "h1": (0.24220, 0.08703),
        "oh": (0.32429, 0.38911), "ho": (0.05379, 0.01966),
        "n3": (0.33210, 0.41236), "hn": (0.11065, 0.04184),
        "os": (0.31561, 0.30376), "ha": (0.26255, 0.06736),
        "hc": (0.26002, 0.08703), "ss": (0.35636, 1.04600),
        "ca": (0.33152, 0.41338), "c1": (0.34790, 0.66777),
        "n1": (0.32735, 0.45940), "c2": (0.33152, 0.41338),
    }
    # Collect unique atom types used in this molecule
    used_types = set()
    for i in range(n_atoms):
        e = mol.GetAtomWithIdx(i).GetAtomicNum()
        used_types.add(etype.get(e, "c3"))
    at_lines = [f"; name  bond_type  mass    charge  ptype  sigma       epsilon"]
    at_lines.append(f"  {si_type}  {si_type}  {28.086:.3f}  0.000  A  "
                    f"{si_lj['sigma']:.5e}  {si_lj['eps']:.5e}")
    for t in sorted(used_types):
        if t != si_type and t in _gaff2_lj:
            s, e = _gaff2_lj[t]
            m = {"c3": 12.011, "h1": 1.008, "oh": 15.999, "ho": 1.008,
                 "n3": 14.007, "hn": 1.008, "os": 15.999, "ha": 1.008,
                 "hc": 1.008, "ss": 32.065, "ca": 12.011, "c1": 12.011,
                 "n1": 14.007, "c2": 12.011}.get(t, 12.011)
            at_lines.append(f"  {t}  {t}  {m:.3f}  0.000  A  {s:.5e}  {e:.5e}")
    atomtypes_section = "[ atomtypes ]\n" + "\n".join(at_lines) + "\n\n"
    # Generate [ bonds ] from RDKit connectivity with standard bond lengths
    _std_bond_len = {  # nm, standard covalent bond lengths
        (6, 6): 0.1529, (6, 1): 0.1090, (6, 8): 0.1430,
        (6, 7): 0.1470, (6, 14): 0.1860, (8, 14): 0.1640,
        (8, 1): 0.0960, (7, 1): 0.1010, (14, 14): 0.2340,
        (6, 16): 0.1810, (16, 1): 0.1340, (6, 5): 0.1560,
        (5, 8): 0.1370, (5, 7): 0.1420,
    }
    blines = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx() + 1
        j = bond.GetEndAtomIdx() + 1
        e1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum()
        e2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum()
        key = (min(e1, e2), max(e1, e2))
        dist = _std_bond_len.get(key, 0.1500)  # default 0.15nm
        blines.append(f"  {i:5d}  {j:5d}    1    {dist:.4f}  500000.0")

    # Generate [ angles ] from RDKit
    aangle_lines = []
    from rdkit.Chem import rdMolTransforms
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for ni in range(len(neighbors)):
            for nj in range(ni+1, len(neighbors)):
                a1, a2, a3 = neighbors[ni]+1, idx+1, neighbors[nj]+1
                try:
                    angle = rdMolTransforms.GetAngleDeg(conf, neighbors[ni], idx, neighbors[nj])
                    aangle_lines.append(f"  {a1:5d}  {a2:5d}  {a3:5d}    1    {angle:.2f}  500.0")
                except Exception:
                    pass

    bonds_section = "[ bonds ]\n" + "\n".join(blines) + "\n" if blines else ""
    angles_section = "\n[ angles ]\n" + "\n".join(aangle_lines) + "\n" if aangle_lines else ""

    itp_path.write_text(
        f"; {name} - PolCA Si (Jorge 2021) + GAFF2\n"
        f"{atomtypes_section}"
        f"[ moleculetype ]\n{name}    3\n\n[ atoms ]\n"
        + "\n".join(alines) + "\n\n"
        + bonds_section + angles_section,
        encoding="utf-8")

    gro_path = output_dir / f"{name}.gro"
    gl = [f"{name} silane", f" {n_atoms}"]
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        a = mol.GetAtomWithIdx(i)
        gl.append(f"{1:5d}{name:>5s}{a.GetSymbol()+str(i+1):>5s}{i+1:5d}"
                  f"{pos.x/10:8.3f}{pos.y/10:8.3f}{pos.z/10:8.3f}")
    gl.append("   5.00000   5.00000   5.00000")
    gro_path.write_text("\n".join(gl) + "\n", encoding="utf-8")

    logger.info(f"  {name}: PolCA topology ({si_type}, "
                f"sigma={si_lj['sigma']}, eps={si_lj['eps']})")
    return {"itp": str(itp_path), "gro": str(gro_path),
            "si_type": si_type, "method": "PolCA"}


def _parse_mmpbsa_results(dat_path: Path) -> dict:
    """Parse gmx_MMPBSA FINAL_RESULTS file."""
    text = Path(dat_path).read_text(encoding="utf-8", errors="replace")
    results = {}

    # gmx_MMPBSA uses Δ (unicode delta) prefix
    for line in text.split("\n"):
        if "TOTAL" in line and ("DELTA" in line or "Δ" in line):
            parts = line.split()
            try:
                # ΔTOTAL  -0.71  41.65  2.63  4.17  0.26
                for i, p in enumerate(parts):
                    if "TOTAL" in p:
                        results["delta_total_kcal"] = float(parts[i+1])
                        if len(parts) > i + 2:
                            results["delta_total_std"] = float(parts[i+4])
                        break
            except (IndexError, ValueError):
                pass
        elif ("Δ" in line or "DELTA" in line) and "---" not in line:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].replace("Δ", "delta_")
                try:
                    results[key.lower()] = float(parts[1])
                except (IndexError, ValueError):
                    pass

    return results
