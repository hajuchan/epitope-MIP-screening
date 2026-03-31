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
    Generate GAFF2 topology for a monomer using acpype.

    Returns dict with paths to .itp and .gro files.
    """
    from .config import ACPYPE_BIN
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mol2_path = Path(mol2_path)
    work_dir = output_dir / f"acpype_{name}"

    cmd = [
        ACPYPE_BIN,
        "-i", str(mol2_path),
        "-b", name,
        "-c", charge_method,
        "-n", "0",           # net charge
        "-a", "gaff2",
        "-o", "gmx",
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(output_dir),
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.warning(f"acpype failed for {name}: {result.stderr[:300]}")
            return {"error": result.stderr}
    except FileNotFoundError:
        logger.error(f"acpype not found: {ACPYPE_BIN}")
        return {"error": "acpype not found"}

    # Find output files
    acpype_dir = output_dir / f"{name}.acpype"
    if not acpype_dir.exists():
        # Try alternate naming
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


# ── System Setup ───────────────────────────────────────────────

def _gmx(cmd_args: list, work_dir: Path, input_text: str = None,
          timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a gmx command."""
    from .config import GMX_BIN
    full_cmd = [GMX_BIN] + cmd_args
    return subprocess.run(
        full_cmd, cwd=str(work_dir),
        input=input_text, capture_output=True, text=True,
        timeout=timeout,
    )


def setup_protein_topology(pdb_path: Path, work_dir: Path,
                            forcefield: str = "amber99sb-ildn",
                            water: str = "tip3p") -> Path:
    """
    Generate GROMACS topology for the protein epitope.
    Returns path to processed .gro file.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    result = _gmx([
        "pdb2gmx",
        "-f", str(pdb_path),
        "-o", str(work_dir / "protein.gro"),
        "-p", str(work_dir / "topol.top"),
        "-ignh",
        "-ff", forcefield,
        "-water", water,
    ], work_dir, input_text="1\n")  # select force field

    if result.returncode != 0:
        logger.error(f"pdb2gmx failed: {result.stderr[:500]}")
        raise RuntimeError(f"pdb2gmx failed: {result.stderr[:300]}")

    return work_dir / "protein.gro"


def setup_simulation_box(gro_path: Path, work_dir: Path,
                          box_type: str = "dodecahedron",
                          distance: float = 1.2) -> Path:
    """Create simulation box, solvate, and add ions."""
    work_dir = Path(work_dir)

    # Define box
    _gmx(["editconf",
           "-f", str(gro_path),
           "-o", str(work_dir / "boxed.gro"),
           "-c", "-d", str(distance),
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
           "-maxwarn", "2"], work_dir)

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
           "-maxwarn", "2"], work_dir)

    result = _gmx(["mdrun", "-deffnm", "em"], work_dir, timeout=1800)
    if result.returncode != 0:
        logger.warning(f"EM mdrun issue: {result.stderr[:300]}")

    return work_dir / "em.gro"


def run_nvt_equilibration(work_dir: Path, time_ps: float = 100.0,
                           temperature: float = 300.0) -> Path:
    """Run NVT equilibration."""
    work_dir = Path(work_dir)
    from .config import MD_TIMESTEP_FS
    dt = MD_TIMESTEP_FS / 1000.0  # fs to ps
    nsteps = int(time_ps / dt)

    mdp_path = work_dir / "nvt.mdp"
    mdp_path.write_text(MDP_NVT.format(
        nsteps=nsteps, dt=dt, temperature=temperature))

    _gmx(["grompp",
           "-f", str(mdp_path),
           "-c", str(work_dir / "em.gro"),
           "-r", str(work_dir / "em.gro"),
           "-p", str(work_dir / "topol.top"),
           "-o", str(work_dir / "nvt.tpr"),
           "-maxwarn", "2"], work_dir)

    _gmx(["mdrun", "-deffnm", "nvt"], work_dir, timeout=3600)
    return work_dir / "nvt.gro"


def run_npt_equilibration(work_dir: Path, time_ps: float = 100.0,
                           temperature: float = 300.0,
                           pressure: float = 1.0) -> Path:
    """Run NPT equilibration."""
    work_dir = Path(work_dir)
    from .config import MD_TIMESTEP_FS
    dt = MD_TIMESTEP_FS / 1000.0
    nsteps = int(time_ps / dt)

    mdp_path = work_dir / "npt.mdp"
    mdp_path.write_text(MDP_NPT.format(
        nsteps=nsteps, dt=dt, temperature=temperature, pressure=pressure))

    _gmx(["grompp",
           "-f", str(mdp_path),
           "-c", str(work_dir / "nvt.gro"),
           "-r", str(work_dir / "nvt.gro"),
           "-t", str(work_dir / "nvt.cpt"),
           "-p", str(work_dir / "topol.top"),
           "-o", str(work_dir / "npt.tpr"),
           "-maxwarn", "2"], work_dir)

    _gmx(["mdrun", "-deffnm", "npt"], work_dir, timeout=3600)
    return work_dir / "npt.gro"


def run_production_md(work_dir: Path, time_ns: float = 200.0,
                       temperature: float = 300.0,
                       pressure: float = 1.0,
                       gpu_id: str = "0") -> Path:
    """Run production MD simulation."""
    work_dir = Path(work_dir)
    from .config import MD_TIMESTEP_FS
    dt = MD_TIMESTEP_FS / 1000.0
    nsteps = int(time_ns * 1000.0 / dt)  # ns → ps → steps

    mdp_path = work_dir / "md.mdp"
    mdp_path.write_text(MDP_PRODUCTION.format(
        nsteps=nsteps, dt=dt, temperature=temperature, pressure=pressure))

    _gmx(["grompp",
           "-f", str(mdp_path),
           "-c", str(work_dir / "npt.gro"),
           "-t", str(work_dir / "npt.cpt"),
           "-p", str(work_dir / "topol.top"),
           "-o", str(work_dir / "md.tpr"),
           "-maxwarn", "2"], work_dir)

    md_cmd = ["mdrun", "-deffnm", "md", "-v"]
    from .config import USE_GPU
    if USE_GPU:
        md_cmd.extend([
            "-nb", "gpu",
            "-pme", "gpu",
            "-bonded", "gpu",
            "-update", "gpu",
            "-gpu_id", gpu_id,
        ])

    logger.info(f"Starting {time_ns}ns production MD in {work_dir}")
    _gmx(md_cmd, work_dir, timeout=int(time_ns * 3600))  # generous timeout
    return work_dir / "md.xtc"


# ── Trajectory Analysis ────────────────────────────────────────

def analyze_trajectory(work_dir: Path) -> dict:
    """
    Analyze production MD trajectory.
    Returns dict with RMSD, RMSF, H-bond, Rg metrics.
    """
    work_dir = Path(work_dir)
    results = {}

    # RMSD
    try:
        _gmx(["rms",
               "-f", "md.xtc", "-s", "md.tpr",
               "-o", "rmsd.xvg"],
              work_dir, input_text="Backbone\nBackbone\n")
        rmsd_data = _parse_xvg(work_dir / "rmsd.xvg")
        if rmsd_data:
            import numpy as np
            results["rmsd_mean_nm"] = float(np.mean(rmsd_data[:, 1]))
            results["rmsd_last50ns_mean_nm"] = float(
                np.mean(rmsd_data[rmsd_data[:, 0] > rmsd_data[-1, 0] - 50000, 1])
            )
    except Exception as e:
        logger.warning(f"RMSD analysis failed: {e}")

    # RMSF
    try:
        _gmx(["rmsf",
               "-f", "md.xtc", "-s", "md.tpr",
               "-o", "rmsf.xvg", "-res"],
              work_dir, input_text="Backbone\n")
        rmsf_data = _parse_xvg(work_dir / "rmsf.xvg")
        if rmsf_data is not None:
            import numpy as np
            results["rmsf_mean_nm"] = float(np.mean(rmsf_data[:, 1]))
            results["rmsf_max_nm"] = float(np.max(rmsf_data[:, 1]))
    except Exception as e:
        logger.warning(f"RMSF analysis failed: {e}")

    # H-bonds (protein to non-protein)
    try:
        _gmx(["hbond",
               "-f", "md.xtc", "-s", "md.tpr",
               "-num", "hbond.xvg"],
              work_dir, input_text="Protein\nNon-Protein\n")
        hb_data = _parse_xvg(work_dir / "hbond.xvg")
        if hb_data is not None:
            import numpy as np
            results["hbond_mean"] = float(np.mean(hb_data[:, 1]))
            results["hbond_max"] = float(np.max(hb_data[:, 1]))
    except Exception as e:
        logger.warning(f"H-bond analysis failed: {e}")

    # Radius of gyration
    try:
        _gmx(["gyrate",
               "-f", "md.xtc", "-s", "md.tpr",
               "-o", "gyrate.xvg"],
              work_dir, input_text="Protein\n")
        rg_data = _parse_xvg(work_dir / "gyrate.xvg")
        if rg_data is not None:
            import numpy as np
            results["rg_mean_nm"] = float(np.mean(rg_data[:, 1]))
    except Exception as e:
        logger.warning(f"Rg analysis failed: {e}")

    # Sullivan 2019 / Sehit 2024: DSSP secondary structure (computational CD)
    from .config import DSSP_ANALYSIS
    if DSSP_ANALYSIS:
        try:
            from .utils_analysis import analyze_dssp_changes
            xtc = work_dir / "md.xtc"
            tpr = work_dir / "md.tpr"
            # Convert tpr to pdb for mdtraj topology
            gro = work_dir / "md_start.gro"
            _gmx(["editconf", "-f", str(tpr), "-o", str(gro)], work_dir)
            if xtc.exists() and gro.exists():
                dssp = analyze_dssp_changes(xtc, gro)
                results["dssp"] = dssp
                if dssp.get("structure_preserved") is False:
                    logger.warning(
                        f"2° structure NOT preserved during MD "
                        f"(helix change: {dssp.get('helix_change', 'N/A')})"
                    )
        except Exception as e:
            logger.warning(f"DSSP analysis failed: {e}")

    return results


# ── MM-PBSA ────────────────────────────────────────────────────

def run_mmpbsa(work_dir: Path, start_ns: float = 150.0,
               end_ns: float = 200.0,
               n_frames: int = 100) -> dict:
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
    if MMPBSA_METHOD == "GBSA":
        mmpbsa_in.write_text(dedent(f"""\
            &general
              startframe=1, endframe={n_frames}, interval=1,
              verbose=2,
            /
            &gb
              igb=5, saltcon={MD_IONIC_STRENGTH},
            /
        """))
    else:
        mmpbsa_in.write_text(dedent(f"""\
            &general
              startframe=1, endframe={n_frames}, interval=1,
              verbose=2,
            /
            &pb
              istrng={MD_IONIC_STRENGTH}, fillratio=4.0,
            /
        """))

    # gmx_MMPBSA command
    cmd = [
        "gmx_MMPBSA",
        "-O",
        "-i", str(mmpbsa_in),
        "-cs", str(work_dir / "md.tpr"),
        "-ct", str(work_dir / "md.xtc"),
        "-ci", str(work_dir / "index.ndx"),
        "-cg", "1", "13",  # receptor and ligand groups (adjust per system)
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
                          quick: bool = False) -> dict:
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
        # 1. Protein topology
        logger.info("Setting up protein topology...")
        setup_protein_topology(protein_pdb, work_dir)

        # 1b. Include monomer ITP/GRO in topology
        if monomer_itps:
            logger.info(f"Including {len(monomer_itps)} monomer(s) in topology...")
            _include_monomers_in_topology(work_dir, monomer_itps)
            system_gro = work_dir / "complex.gro"
        else:
            system_gro = work_dir / "protein.gro"

        # 2. Solvate & ionize
        logger.info("Setting up simulation box...")
        setup_simulation_box(system_gro, work_dir)

        # 3. Energy minimization
        logger.info("Running energy minimization...")
        run_energy_minimization(work_dir)

        # 4. NVT equilibration (100 ps)
        logger.info("NVT equilibration...")
        run_nvt_equilibration(work_dir, time_ps=100.0,
                               temperature=MD_TEMPERATURE_K)

        # 5. NPT equilibration (100 ps)
        logger.info("NPT equilibration...")
        run_npt_equilibration(work_dir, time_ps=100.0,
                               temperature=MD_TEMPERATURE_K,
                               pressure=MD_PRESSURE_BAR)

        # 6. Production MD
        logger.info(f"Production MD ({time_ns} ns)...")
        run_production_md(work_dir, time_ns=time_ns,
                           temperature=MD_TEMPERATURE_K,
                           pressure=MD_PRESSURE_BAR,
                           gpu_id=MD_GPU_ID)

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

    # Collect monomer coordinates and topology edits
    include_lines = []
    molecule_lines = []
    all_mon_coords = []

    for i, param in enumerate(monomer_itps):
        itp_path = param.get("itp")
        gro_path = param.get("gro")
        if not itp_path or not Path(itp_path).exists():
            continue

        # Derive molecule name from ITP
        itp_src = Path(itp_path)
        mol_name = itp_src.stem.replace("_GMX", "")

        # Copy ITP
        itp_dst = work_dir / itp_src.name
        shutil.copy2(str(itp_src), str(itp_dst))
        include_lines.append(f'#include "{itp_src.name}"')
        molecule_lines.append(f"{mol_name}     1")

        # Read monomer GRO coordinates
        if gro_path and Path(gro_path).exists():
            mon_lines = Path(gro_path).read_text().strip().split("\n")
            mon_natoms = int(mon_lines[1].strip())
            mon_coords = mon_lines[2:2+mon_natoms]

            # Offset monomer position to avoid overlap with protein
            # Place each monomer at +2nm offset in x direction
            offset_coords = _offset_gro_coords(mon_coords, x_offset=2.0 + i * 1.5)
            all_mon_coords.extend(offset_coords)

    # Edit topol.top
    content = top_path.read_text()
    include_block = "\n".join(include_lines)
    molecule_block = "\n".join(molecule_lines)

    if "[ molecules ]" in content:
        content = content.replace(
            "[ molecules ]",
            f"{include_block}\n\n[ molecules ]"
        )
        content = content.rstrip() + "\n" + molecule_block + "\n"
    else:
        content += f"\n{include_block}\n\n[ molecules ]\n{molecule_block}\n"
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


def _offset_gro_coords(coord_lines: list, x_offset: float = 2.0) -> list:
    """Offset GRO coordinate lines by x_offset nm to avoid steric clash."""
    shifted = []
    for line in coord_lines:
        try:
            # GRO format: resid(5) resname(5) atomname(5) atomnr(5) x(8.3) y(8.3) z(8.3)
            prefix = line[:20]
            x = float(line[20:28]) + x_offset
            y = float(line[28:36])
            z = float(line[36:44])
            rest = line[44:] if len(line) > 44 else ""
            shifted.append(f"{prefix}{x:8.3f}{y:8.3f}{z:8.3f}{rest}")
        except (ValueError, IndexError):
            shifted.append(line)
    return shifted


def _parse_mmpbsa_results(dat_path: Path) -> dict:
    """Parse gmx_MMPBSA FINAL_RESULTS file."""
    import re
    text = Path(dat_path).read_text()
    results = {}

    # Look for DELTA TOTAL line
    for line in text.split("\n"):
        if "DELTA TOTAL" in line:
            parts = line.split()
            try:
                results["delta_total_kcal"] = float(parts[-2])
                results["delta_total_std"] = float(parts[-1])
            except (IndexError, ValueError):
                pass
        elif "DELTA" in line and "TOTAL" not in line:
            parts = line.split()
            if len(parts) >= 3:
                key = parts[0] + "_" + parts[1]
                try:
                    results[key.lower()] = float(parts[-2])
                except (IndexError, ValueError):
                    pass

    return results
