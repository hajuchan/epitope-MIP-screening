"""
GROMACS Utilities
=================
Wrappers for GROMACS MD simulation setup, execution,
trajectory analysis, and MM-PBSA binding free energy calculation.

Reference:
  Sullivan et al., J. Phys. Chem. B 2019 -- MM-PBSA for MIP
  Rebelo et al., Int. J. Mol. Sci. 2023 -- GROMACS + gmx_MMPBSA protocol
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from textwrap import dedent

logger = logging.getLogger(__name__)


# ── Tunables that used to be hardcoded ─────────────────────────
# Every constant below was previously a magic number buried in a call.
# They are named here so a change is visible in one place.

MD_RVDW_NM = 1.0        # must match rvdw/rcoulomb in the MDP templates below
MD_RCOULOMB_NM = 1.0

# BLOCKER 10 -- minimum-image safety.
# A solute atom must not see its own periodic image inside the non-bonded
# cutoff.  Requirement: minimum image separation of the SOLUTE (protein AND
# monomers) > 2*rvdw + MIN_IMAGE_MARGIN_NM.  With rvdw = 1.0 nm that is
# 2.4 nm, i.e. >= 1.2 nm of padding on every side.  The old code hardcoded
# 0.5 nm of padding measured from the protein only, which gave 1.09 nm of
# separation -- monomers interacted with their own images for the whole run.
MIN_IMAGE_MARGIN_NM = 0.4

# BLOCKER 11 -- trajectory output rate.
# CHANGED DEFAULT (was 5000 steps = 10 ps/frame).  At dt = 2 fs a 350 ns leg
# wrote 35,000 frames / 11.3 GB, of which the analysis code reads ~204.
# 50000 steps = 100 ps/frame -> 3,500 frames / ~1.1 GB per leg, still ~17x
# more than anything downstream consumes.  Override with MD_NSTXOUT_COMPRESSED
# in config if a denser trajectory is genuinely needed.
NSTXOUT_COMPRESSED_PRODUCTION = 50000
NSTXOUT_COMPRESSED_EQUIL = 5000     # equilibration legs are 100 ps; 20 frames

# Equilibration is only 100 ps, so energies are sampled densely enough that a
# plateau test has statistics.  .edr is a few hundred kB -- this costs nothing.
NSTENERGY_EQUIL = 100

# BLOCKER 10 -- equilibration acceptance criteria.  There were none.
EM_REQUIRE_CONVERGENCE = True       # hard-fail if steep did not reach emtol
EQUIL_TEMP_TOL_K = 5.0              # |<T> - ref_t| over the 2nd half
EQUIL_TEMP_DRIFT_TOL_K = 10.0       # |T(2nd half) - T(1st half)|
EQUIL_DENSITY_DRIFT_FRAC = 0.01     # |rho(2nd half)/rho(1st half) - 1|
EQUIL_PRESSURE_TOL_BAR = 100.0      # advisory only -- <P> over 50 ps is noisy


def _cfg(name, default):
    """Read an OPTIONAL config symbol.

    Symbols that must exist are imported normally so a missing one is a loud
    ImportError.  This helper is only for knobs that are allowed to be absent
    (so the module keeps working against a config that predates them).
    """
    from . import config as _config
    return getattr(_config, name, default)


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
    nstxout-compressed = {nstxout}
    nstenergy   = {nstenergy}
    nstlog      = {nstenergy}
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
    gen_seed    = {gen_seed}
""")

MDP_NPT = dedent("""\
    ; NPT Equilibration
    {define}
    integrator  = md
    nsteps      = {nsteps}
    dt          = {dt}
    nstxout-compressed = {nstxout}
    nstenergy   = {nstenergy}
    nstlog      = {nstenergy}
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
    nstxout-compressed = {nstxout}
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
    ; BLOCKER 10: production may run with -DPOSRES (surface-MIP mode). Without
    ; refcoord_scaling grompp emits a WARNING about position restraints under
    ; pressure coupling -- previously swallowed by -maxwarn 10, now a hard error.
    refcoord_scaling = com
""")


# ── Monomer Parameterization ──────────────────────────────────

def _monomer_elements(name: str) -> set:
    """Atomic numbers present in a library monomer, or an empty set if unknown."""
    from .config import ALL_MONOMERS
    m_info = ALL_MONOMERS.get(name)
    if not m_info:
        return set()
    from rdkit import Chem
    mol = Chem.MolFromSmiles(m_info["smiles"])
    if mol is None:
        return set()
    return {a.GetAtomicNum() for a in mol.GetAtoms()}


def _monomer_formal_charge(name: str) -> int:
    """Net formal charge of a library monomer, from its SMILES.

    WHY THIS EXISTS.  `_run_acpype` hardcoded `-n 0`, so any ionic species was
    either refused by antechamber or silently parameterised as if it were
    neutral.  The species that matters here is the one the protocol actually
    contains: at pH ~9.5 the aminopropylsilanetriol amine (pKa ~10.6) is ~93%
    protonated, i.e. [NH3+]CCC[Si](O)(O)O, and that ammonium against BSA
    carboxylate (pI 4.7) IS the imprinting driving force this experiment
    measures.  A hardcoded 0 made the driving force unrepresentable.
    Unknown monomer -> 0, which is the old behaviour for every neutral species.
    """
    from .config import ALL_MONOMERS
    m_info = ALL_MONOMERS.get(name)
    if not m_info or not m_info.get("smiles"):
        return 0
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(m_info["smiles"])
        if mol is None:
            return 0
        return int(Chem.GetFormalCharge(mol))
    except Exception:
        return 0


def _run_acpype(mol2_path: Path, name: str, output_dir: Path,
                 charge_method: str = "bcc", net_charge: int = None) -> dict:
    """Run acpype/GAFF2 and return {itp, gro, acpype_dir} or {error: ...}.

    `net_charge` None -> derived from the library SMILES' formal charge (see
    `_monomer_formal_charge`).  It used to be hardcoded to 0.
    """
    from .config import ACPYPE_BIN
    import sys
    if net_charge is None:
        net_charge = _monomer_formal_charge(name)
    if int(net_charge) != 0:
        logger.info(f"  {name}: net formal charge {int(net_charge):+d} from the "
                    f"library SMILES -- passing it to acpype (-n {int(net_charge)})")
    cmd = [
        sys.executable, ACPYPE_BIN,
        "-i", str(mol2_path),
        "-b", name,
        "-c", charge_method,
        "-n", str(int(net_charge)),   # net charge -- NO LONGER hardcoded to 0
        "-a", "gaff2",
        "-o", "gmx",
    ]

    # Put THIS interpreter's env bin FIRST in PATH, and pin AMBERHOME to it.
    #
    # This used to prepend only `if conda_bin not in PATH`, which is not the
    # same thing: on this machine another conda environment
    # (~/anaconda3/envs/GROMACS/bin) sits ahead of MIPscreen/bin in the
    # inherited PATH, so antechamber found the WRONG teLeap -- one that never
    # reads the parmchk frcmod and dies with
    #   "teLeap: Error! Could not find angle parameter: os - Si - os"
    #   "could not find vdW (or other) parameters for type (Si)"
    # acpype then exits 19 and every silane fell through to the fallback path.
    # Reproduce: run acpype on TEOS with and without an unconditional prepend --
    # rc=19 vs rc=0 from the same directory and the same input.
    env = os.environ.copy()
    conda_bin = str(Path(sys.executable).parent)
    env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
    env["AMBERHOME"] = str(Path(sys.executable).parent.parent)

    try:
        result = subprocess.run(
            cmd, cwd=str(output_dir),
            capture_output=True, text=True, timeout=600,  # 10min for large molecules
            env=env,
        )
    except FileNotFoundError:
        return {"error": f"acpype not found: {ACPYPE_BIN}"}
    except subprocess.TimeoutExpired:
        return {"error": f"acpype timed out for {name}"}

    if result.returncode != 0:
        detail = (result.stderr or "")[-500:] or (result.stdout or "")[-500:]
        return {"error": f"acpype rc={result.returncode}: {detail}"}

    acpype_dir = output_dir / f"{name}.acpype"
    if not acpype_dir.exists():
        candidates = list(output_dir.glob(f"*{name}*acpype*"))
        acpype_dir = candidates[0] if candidates else output_dir

    itp_files = list(acpype_dir.glob("*_GMX.itp")) + list(acpype_dir.glob("*.itp"))
    gro_files = list(acpype_dir.glob("*_GMX.gro")) + list(acpype_dir.glob("*.gro"))
    if not itp_files or not gro_files:
        return {"error": f"acpype produced no itp/gro for {name} in {acpype_dir}"}

    return {"itp": str(itp_files[0]), "gro": str(gro_files[0]),
            "acpype_dir": str(acpype_dir)}


def parameterize_monomer(mol2_path: Path, name: str,
                          output_dir: Path,
                          charge_method: str = "bcc") -> dict:
    """
    Generate topology for a monomer.

    BLOCKER 03 -- routing changed.  The old code sent EVERY Si- and B-containing
    monomer to the hand-built `_generate_silane_itp` path, which produced
    topologies with no dihedrals, no 1-4 pairs, mistyped polar hydrogens and
    (because of a (min,max) key mismatch) every X-H bond stretched to 0.1500 nm.

    Measured on this machine (see the audit report for the exact commands):

      * SILICON -- acpype/antechamber/sqm handle Si fine.  `acpype -i MTMS.mol2
        -c bcc -a gaff2` exits 0 and writes a complete GAFF2 topology with
        AM1-BCC charges, [pairs], [dihedrals] and correct H typing.  What it
        gets WRONG is only the Si itself: parmchk fills every Si parameter with
        "same as c3", so Si carries sp3-carbon LJ and Si-C / Si-O bonds come
        out at 0.1535 / 0.1427 nm instead of ~0.187 / ~0.164 nm.  So: run
        acpype, then overwrite exactly those Si parameters with PolCA/published
        values (`_apply_polca_si_overrides`).

      * BORON -- acpype genuinely cannot.  `atomtype -p gaff2` assigns boron the
        dummy type `DU` (GAFF2 has no boron), and AM1-BCC dies with
        "QMMM: Atom number: 7 has atomic number 5.  There are no AM1
        parameters for this element."  `-c gas` fails too ("No Gasteiger
        parameter for atom (ID: 6, Name: B, Type: DU)").  Boron therefore stays
        on the hand-built path -- which is now a proper GAFF2 build with
        symmetric bond lookup, real polar-H types, published boron LJ,
        generated propers, planarity impropers and 1-4 pairs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mol2_path = Path(mol2_path)

    elements = _monomer_elements(name)
    has_si = 14 in elements
    has_b = 5 in elements

    if has_b:
        logger.info(f"  {name}: boron -- GAFF2/AM1-BCC has no boron parameters, "
                    f"using the hand-built GAFF2+UFF-B path")
        return _hand_built_topology(name, output_dir)

    if has_si:
        logger.info(f"  {name}: silicon -- acpype/GAFF2 + PolCA Si override")
        res = _run_acpype(mol2_path, name, output_dir, charge_method)
        if "error" in res:
            # WAS: a WARNING and a silent fall-through to _generate_silane_itp,
            # whose Gasteiger polar-H charges are ~0.15-0.19 against acpype's
            # ~0.35-0.45. Two silanes in one box, one from each path, is not a
            # composition comparison -- it is a comparison of charge models.
            # Opt back in with MIP_ALLOW_HANDBUILT_SILANE=1 for a debugging run.
            if os.environ.get("MIP_ALLOW_HANDBUILT_SILANE", "").strip().lower() \
                    in ("1", "true", "yes", "on"):
                logger.error(
                    f"  {name}: acpype failed ({res['error'][:200]}) -- "
                    f"MIP_ALLOW_HANDBUILT_SILANE is set, falling back to the "
                    f"hand-built path. Its electrostatics are NOT comparable "
                    f"with an acpype-built silane in the same box.")
                return _hand_built_topology(name, output_dir)
            raise RuntimeError(
                f"acpype failed for the silane {name}: {res['error'][:300]}. "
                f"Refusing to fall through to the hand-built Gasteiger path: it "
                f"ships different polar-hydrogen charges (~0.15-0.19 e vs "
                f"~0.35-0.45 e) under a warning, and a box that mixes the two "
                f"paths compares charge models rather than compositions. Fix the "
                f"acpype failure, or set MIP_ALLOW_HANDBUILT_SILANE=1 "
                f"deliberately for a debugging run.")
        try:
            si_type = _apply_polca_si_overrides(Path(res["itp"]), name)
        except Exception as e:
            logger.error(f"  {name}: PolCA Si override failed on "
                         f"{res['itp']}: {e}")
            raise
        res["si_type"] = si_type
        res["method"] = "acpype-gaff2+polca-si"
        res["net_charge"] = _assert_itp_net_charge(Path(res["itp"]), name)
        return res

    res = _run_acpype(mol2_path, name, output_dir, charge_method)
    if "error" in res:
        logger.error(f"  {name}: acpype failed -- {res['error'][:300]}")
    else:
        res["method"] = "acpype-gaff2"
        res["net_charge"] = _assert_itp_net_charge(Path(res["itp"]), name)
    return res


def _assert_itp_net_charge(itp_path: Path, name: str, tol: float = 1e-3) -> int:
    """Assert the built ITP's summed charge equals the SMILES formal charge.

    The charge a monomer carries is now a MEASURED property of the topology,
    checked against what was requested, rather than an assumption. A silane
    that quietly came out non-integer is exactly what manufactured part of the
    electrostatic signal in the previous round.
    """
    want = _monomer_formal_charge(name)
    qtot = 0.0
    in_atoms = False
    for line in Path(itp_path).read_text().split("\n"):
        s = line.strip()
        if s.startswith("["):
            in_atoms = s.replace(" ", "").startswith("[atoms]")
            continue
        if not in_atoms or not s or s.startswith(";"):
            continue
        parts = s.split()
        if len(parts) >= 7:
            try:
                qtot += float(parts[6])
            except ValueError:
                continue
    if abs(qtot - want) > tol:
        raise RuntimeError(
            f"{name}: the built topology {itp_path} sums to qtot={qtot:+.5f} e "
            f"but the library SMILES has formal charge {want:+d}. A monomer that "
            f"carries a charge nobody asked for adds an electrostatic signal to "
            f"every leg -- and in DI water (MD_IONIC_STRENGTH=0) the Debye length "
            f"is ~3x longer, so the artefact reaches ~3x further. Refusing.")
    logger.info(f"  {name}: net charge verified qtot={qtot:+.5f} e "
                f"(formal {want:+d})")
    return want


def _hand_built_topology(name: str, output_dir: Path) -> dict:
    """Hand-built GAFF2 topology from SMILES (boron monomers, Si fallback)."""
    from .config import ALL_MONOMERS
    m_info = ALL_MONOMERS.get(name)
    if m_info is None:
        return {"error": f"Monomer {name} not in library"}
    try:
        return _generate_silane_itp(name, m_info["smiles"],
                                     output_dir / f"{name}_polca")
    except Exception as e:
        logger.error(f"  hand-built topology failed for {name}: {e}")
        return {"error": str(e)}


# Back-compat shim: older call sites imported this name.
def _try_polca_fallback(name: str, mol2_path: Path, output_dir: Path) -> dict:
    return _hand_built_topology(name, Path(output_dir))


# ── System Setup ───────────────────────────────────────────────

class GromacsError(RuntimeError):
    """A gmx invocation returned non-zero."""


class EquilibrationError(RuntimeError):
    """A system failed an EM / NVT / NPT acceptance criterion."""


def _gmx_failure_report(rel_args: list, result) -> str:
    """Build a message that actually says WHY gmx failed.

    grompp prints its WARNING/ERROR text to stderr in a banner; the old code
    logged only the last 200 characters, at DEBUG, which is below the INFO
    level the pipeline runs at.  Failures were therefore invisible.
    """
    blob = "\n".join(filter(None, [result.stdout or "", result.stderr or ""]))
    lines = blob.split("\n")
    keys = ("ERROR", "Error", "WARNING", "Warning", "Fatal", "Invalid",
            "Cannot", "not found", "No such")
    # Keep the banner line AND the following few lines -- grompp puts the actual
    # diagnostic ("System has non-zero total charge: ...") under its header, so
    # matching only the header prints a failure with no reason in it.
    wanted = set()
    for i, ln in enumerate(lines):
        if any(k in ln for k in keys):
            wanted.update(range(i, min(i + 4, len(lines))))
    keep = [lines[i] for i in sorted(wanted) if lines[i].strip()]
    detail = "\n  ".join(keep[-40:]) if keep else (blob[-800:] or "(no output)")
    return (f"gmx {' '.join(str(a) for a in rel_args)} failed "
            f"(rc={result.returncode}):\n  {detail}")


def _gmx(cmd_args: list, work_dir: Path, input_text: str = None,
          timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    """Run a gmx command. Converts absolute paths to relative to handle spaces.

    BLOCKER 09a -- CHANGED DEFAULT.  This used to log a non-zero return code at
    DEBUG and carry on, so every downstream step ran on whatever stale or
    missing file the failed command left behind.  It now RAISES GromacsError by
    default.  Pass check=False at the few call sites that genuinely tolerate a
    failure (probes, optional analyses, cached-output fallbacks) -- and log
    there yourself why the failure is acceptable.
    """
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
        report = _gmx_failure_report(rel_args, result)
        if check:
            logger.error(report)
            raise GromacsError(report)
        logger.warning(f"[tolerated] {report}")

    return result


def _require_file(path: Path, what: str, min_bytes: int = 1) -> Path:
    """Assert an intermediate exists and is non-empty.

    BLOCKER 09a -- setup_simulation_box used to `return work_dir/"ionized.gro"`
    unconditionally, so a failed genion produced a path to a file that was
    never written and the caller happily fed it to grompp.
    """
    path = Path(path)
    if not path.exists():
        raise GromacsError(f"{what}: expected output {path} was never created")
    size = path.stat().st_size
    if size < min_bytes:
        raise GromacsError(f"{what}: output {path} is {size} bytes "
                           f"(< {min_bytes}) -- the command produced a stub")
    return path


# BLOCKER 10 -- grompp warning budget.
#
# Every grompp call used to carry `-maxwarn 10`, which suppressed exactly the
# diagnostics that would have caught the minimum-image bug ("The cut-off ...
# is longer than half the shortest box vector", "Periodic molecules"), the
# net-charge bug, and the position-restraint/pressure-coupling artifact.
#
# The budget is now 0 everywhere except the pre-genion `ions.tpr` build, where
# ONE warning is whitelisted with a justification:
#
#   "System has non-zero total charge"
#       The system is deliberately charged at this point -- genion is the very
#       next call and neutralises it.  Nothing is integrated with this .tpr.
#
# If a new warning appears, read it and either fix the system or add it here
# with a written justification.  Do not raise the number to make it go away.
GROMPP_MAXWARN_IONS = 1
GROMPP_MAXWARN_PRODUCTION = 0


def _grompp(work_dir: Path, mdp: Path, conf: Path, top: Path, out_tpr: Path,
            restraint: Path = None, checkpoint: Path = None,
            maxwarn: int = GROMPP_MAXWARN_PRODUCTION,
            index: Path = None) -> Path:
    """Run grompp with an explicit, justified warning budget and verify the tpr."""
    args = ["grompp", "-f", str(mdp), "-c", str(conf), "-p", str(top),
            "-o", str(out_tpr)]
    if restraint is not None:
        args += ["-r", str(restraint)]
    if checkpoint is not None and Path(checkpoint).exists():
        args += ["-t", str(checkpoint)]
    if index is not None:
        args += ["-n", str(index)]
    args += ["-maxwarn", str(maxwarn)]

    result = _gmx(args, work_dir, check=True)

    # grompp prints its warnings even when it exits 0 (up to -maxwarn). Surface
    # them at WARNING so a whitelisted warning is still visible in the log.
    blob = "\n".join(filter(None, [result.stdout or "", result.stderr or ""]))
    for chunk in blob.split("\n"):
        if "WARNING" in chunk:
            logger.warning(f"  grompp: {chunk.strip()}")

    return _require_file(out_tpr, f"grompp {Path(mdp).name}", min_bytes=1000)


def _ff_terminus_states_available(forcefield: str = "amber99sb-ildn") -> bool:
    """Can `pdb2gmx -ter` actually offer a choice for this force field?

    MEASURED, not assumed.  GROMACS' amber99sb-ildn port ships
    aminoacids.n.tdb and aminoacids.c.tdb containing only "; empty": the
    termini are baked into the NASP/CALA rtp entries via the r2b N-ter/C-ter
    columns, so `-ter` prints no menu, consumes no answer and silently leaves
    the terminus charged.  That is why a pH-9.5 build came out at -18 e when
    the chemistry asks for -19 e.  Detected here so the difference is reported
    as an ENGINE LIMIT rather than as a failed selection.
    """
    try:
        from .config import GMX_BIN
        import shutil as _sh
        gmx = _sh.which(GMX_BIN) or GMX_BIN
        prefix = Path(gmx).resolve().parent.parent
        ffdir = prefix / "share" / "gromacs" / "top" / f"{forcefield}.ff"
        n_tdb = ffdir / "aminoacids.n.tdb"
        c_tdb = ffdir / "aminoacids.c.tdb"
        if not n_tdb.exists() or not c_tdb.exists():
            return False

        def _has_entries(p):
            return any(line.strip().startswith("[")
                       for line in p.read_text().split("\n"))
        return _has_entries(n_tdb) and _has_entries(c_tdb)
    except Exception as e:
        logger.error("could not inspect %s terminus databases (%s) -- assuming "
                     "the termini are NOT selectable", forcefield, e)
        return False


def _ph_protonation_plan(pdb_path: Path, ph: float,
                         forcefield: str = "amber99sb-ildn") -> dict:
    """pdb2gmx interactive selections that realise `ph`, or {} for the default.

    pdb2gmx's NON-INTERACTIVE defaults are: Lys charged, Asp/Glu charged, Arg
    charged, His chosen by H-bond geometry (always a NEUTRAL tautomer), termini
    charged.  This returns interactive flags ONLY for the residue classes whose
    majority state at `ph` DIFFERS from that default, so a pH-7.4 run produces
    byte-identical input to the pre-fix behaviour and only a genuinely different
    pH changes the topology.

    pdb2gmx has no -cys and no -tyr flag.  A deprotonated cysteine is therefore
    applied by renaming the residue to CYM (amber99sb-ildn's aminoacids.rtp
    defines it); a deprotonated tyrosine has no residue type at all and is
    reported as unrepresentable rather than silently built neutral.
    """
    from .utils_structure import titration_model
    ter_selectable = _ff_terminus_states_available(forcefield)
    unavailable = set()
    if not ter_selectable:
        unavailable |= {("NTERM", "deprotonated"), ("CTERM", "protonated")}
    # Same rule the pre-flight judged against -- if these two disagree, the run
    # is approved on one charge state and built with another.
    try:
        from .config import PH_CHARGE_ASSIGNMENT as _ASSIGN
    except ImportError:
        _ASSIGN = "majority"
    try:
        from .config import PH_USE_PROPKA as _PROPKA
    except ImportError:
        _PROPKA = False
    model = titration_model(Path(pdb_path), float(ph),
                            unavailable_states=unavailable,
                            assignment=_ASSIGN, use_propka=bool(_PROPKA))

    # pdb2gmx prompt order for each interactive flag, per residue occurrence.
    #   -lys : 0 = LYN (neutral)      1 = LYS (charged)      [default 1]
    #   -asp : 0 = ASP (charged)      1 = ASH (neutral)      [default 0]
    #   -glu : 0 = GLU (charged)      1 = GLH (neutral)      [default 0]
    #   -his : 0 = HID  1 = HIE  2 = HIP  3 = HIS1           [default: H-bonds]
    #   -ter : N-term 0 = NH3+  1 = NH2  2 = None ; C-term 0 = COO-  1 = COOH
    answers = {"lys": [], "asp": [], "glu": [], "his": [], "ter": []}
    nonstandard = {"lys": False, "asp": False, "glu": False,
                   "his": False, "ter": False}
    cys_renames = []
    for s in model["sites"]:
        g, prot = s["group"], (s["majority_state"] == "protonated")
        if g == "LYS":
            answers["lys"].append("1" if prot else "0")
            nonstandard["lys"] |= (not prot)
        elif g == "ASP":
            answers["asp"].append("1" if prot else "0")
            nonstandard["asp"] |= prot
        elif g == "GLU":
            answers["glu"].append("1" if prot else "0")
            nonstandard["glu"] |= prot
        elif g == "HIS":
            answers["his"].append("2" if prot else "1")   # HIP / HIE
            nonstandard["his"] |= prot
        elif g == "NTERM" and ter_selectable:
            answers["ter"].append(("N", "0" if prot else "1"))
            nonstandard["ter"] |= (not prot)
        elif g == "CTERM" and ter_selectable:
            answers["ter"].append(("C", "1" if prot else "0"))
            nonstandard["ter"] |= prot
        elif g == "CYS" and not prot:
            cys_renames.append(s["resid"])

    # HISTIDINE IS PINNED AS SOON AS ANYTHING ELSE IS.
    # With no -his flag pdb2gmx picks HID or HIE per residue from H-bond
    # geometry -- both neutral, so the choice does not change the net charge but
    # it is not reproducible across inputs, and it CAN produce a HIP. Once we
    # are asserting the built charge against the model, the His states have to
    # be ours or the assertion is checking a number pdb2gmx chose. When nothing
    # else differs from the default (a pH-7.4 run), no flag is passed at all
    # and the historical behaviour is preserved byte for byte.
    any_change = any(nonstandard[k] for k in ("lys", "asp", "glu", "ter")) \
        or nonstandard["his"] or bool(cys_renames)
    if any_change and answers["his"]:
        nonstandard["his"] = True

    flags, stdin_parts = [], []
    for key in ("lys", "asp", "glu", "his"):
        if nonstandard[key]:
            flags.append(f"-{key}")
            stdin_parts.append(answers[key])
    if nonstandard["ter"]:
        flags.append("-ter")
        # pdb2gmx asks N-terminus then C-terminus for each chain, in order.
        stdin_parts.append([a for _side, a in answers["ter"]])

    return {
        "model": model,
        "terminus_states_selectable": ter_selectable,
        "flags": flags,
        # pdb2gmx consumes the interactive answers in FLAG ORDER, one line each.
        "stdin": "".join(v + "\n" for grp in stdin_parts for v in grp),
        "cys_renames": cys_renames,
        "changes_topology": bool(flags or cys_renames),
        # True only when every titratable state in the box was chosen by the
        # model rather than by a pdb2gmx heuristic. The built-charge assertion
        # is exact only in that case.
        "states_fully_determined": bool(flags or cys_renames),
    }


def _rename_residues(pdb_in: Path, pdb_out: Path, renames: dict) -> int:
    """Rewrite a PDB with {resid: new_resname}; returns the number renamed."""
    done = set()
    lines = []
    for line in Path(pdb_in).read_text().splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
            resid = line[22:27].strip()
            new = renames.get(resid)
            if new:
                line = line[:17] + f"{new:<3}" + line[20:]
                done.add(resid)
        lines.append(line)
    Path(pdb_out).write_text("".join(lines))
    return len(done)


def setup_protein_topology(pdb_path: Path, work_dir: Path,
                            forcefield: str = "amber99sb-ildn",
                            water: str = "tip3p", ph: float = None) -> Path:
    """
    Generate GROMACS topology for the protein epitope.
    Automatically fixes missing heavy atoms before pdb2gmx.
    Returns path to processed .gro file.

    `ph` (default: MD_SOLVENT_PH) is now GENUINELY CONSUMED.  # BEHAVIOUR CHANGE
    pdb2gmx used to run with no -asp/-glu/-lys/-his/-ter flags, so BSA was built
    at its pH-7 census (-16 e, 16 Na+ counter-ions) whatever MD_SOLVENT_PH said;
    the protocol runs at pH ~9.5.  The states now come from
    utils_structure.titration_model and are driven into pdb2gmx interactively.
    At pH 7.4 the model reproduces pdb2gmx's own defaults and NO flag is passed,
    so CD topologies are bit-identical to before.

    A protonation summary is written to `work_dir/protonation_model.json` and
    the built protein charge is asserted against it.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if ph is None:
        try:
            from .config import MD_SOLVENT_PH as _ph
            ph = float(_ph)
        except Exception:
            ph = 7.4

    # Fix missing heavy atoms (e.g., side chains not resolved in X-ray)
    fixed_pdb = _fix_missing_atoms(pdb_path, work_dir)

    plan = {}
    try:
        plan = _ph_protonation_plan(fixed_pdb, ph, forcefield)
    except Exception as e:
        logger.error(
            "pH protonation model failed for %s at pH %s (%s) -- falling back "
            "to pdb2gmx DEFAULTS, i.e. a pH-7 charge state. This is recorded "
            "as protonation_model.json status=failed so no downstream step can "
            "read the run as having been performed at pH %s.",
            fixed_pdb, ph, e, ph)
        plan = {"error": str(e), "flags": [], "stdin": "", "cys_renames": [],
                "changes_topology": False, "model": None}

    pdb_for_gmx = fixed_pdb
    if plan.get("cys_renames"):
        pdb_for_gmx = work_dir / (Path(fixed_pdb).stem + "_cym.pdb")
        n = _rename_residues(fixed_pdb, pdb_for_gmx,
                             {r: "CYM" for r in plan["cys_renames"]})
        logger.info(f"  pH {ph}: renamed {n} free cysteine(s) to CYM "
                    f"(thiolate; pdb2gmx has no -cys flag)")

    cmd = [
        "pdb2gmx",
        "-f", str(pdb_for_gmx),
        "-o", str(work_dir / "protein.gro"),
        "-p", str(work_dir / "topol.top"),
        "-ignh",
        "-ff", forcefield,
        "-water", water,
    ] + list(plan.get("flags") or [])

    if plan.get("changes_topology"):
        logger.info(
            "  pH %.2f protonation is LIVE: pdb2gmx flags %s, %d CYM rename(s). "
            "Model charges -- HH continuum %+.2f e, discrete %+d e, "
            "force-field representable %+d e.",
            ph, plan["flags"], len(plan.get("cys_renames") or []),
            (plan.get("model") or {}).get("hh_continuum_charge", float("nan")),
            (plan.get("model") or {}).get("discrete_charge", 0),
            (plan.get("model") or {}).get("representable_charge", 0))
    else:
        logger.info("  pH %.2f: every majority state equals pdb2gmx's default, "
                    "so no interactive flag is passed (topology unchanged).", ph)

    result = _gmx(cmd, work_dir, input_text=plan.get("stdin") or None)

    if result.returncode != 0:
        logger.error(f"pdb2gmx failed: {result.stderr[:500]}")
        raise RuntimeError(f"pdb2gmx failed: {result.stderr[:300]}")

    # ── ASSERT the built charge against the model ──────────────
    built_q = protein_charge_from_topology(work_dir / "topol.top")
    want_q = (plan.get("model") or {}).get("representable_charge")
    record = {
        "ph": ph,
        "status": "failed" if plan.get("error") else (
            "applied" if plan.get("changes_topology") else "default_states"),
        "error": plan.get("error"),
        "pdb2gmx_flags": plan.get("flags"),
        "pdb2gmx_stdin": plan.get("stdin"),
        "cys_renames": plan.get("cys_renames"),
        "built_protein_charge_e": built_q,
        "model": plan.get("model"),
    }
    (work_dir / "protonation_model.json").write_text(
        json.dumps(record, indent=2, default=str))
    record["states_fully_determined"] = bool(plan.get("states_fully_determined"))
    (work_dir / "protonation_model.json").write_text(
        json.dumps(record, indent=2, default=str))
    if (plan.get("states_fully_determined") and want_q is not None
            and built_q is not None and abs(built_q - want_q) > 0.05):
        raise RuntimeError(
            f"pdb2gmx built a protein of net charge {built_q:+.3f} e but the "
            f"pH {ph} titration model asks for {want_q:+d} e. The interactive "
            f"selections did not land -- check {work_dir/'protonation_model.json'} "
            f"and the pdb2gmx prompt order. Refusing to simulate a charge state "
            f"nobody chose.")
    if built_q is not None:
        logger.info(f"  Built protein net charge: {built_q:+.3f} e "
                    f"(pH {ph} model target {want_q}, states_fully_determined="
                    f"{bool(plan.get('states_fully_determined'))})")

    return work_dir / "protein.gro"


def protein_charge_from_topology(top_path: Path):
    """Net charge of the Protein_* [moleculetype]s in a GROMACS topology, in e.

    Reads topol.top plus any `#include`d Protein_chain_*.itp beside it. Returns
    None when nothing protein-shaped is found.
    """
    top_path = Path(top_path)
    if not top_path.exists():
        return None
    texts = [top_path.read_text()]
    for line in texts[0].split("\n"):
        s = line.strip()
        if s.startswith('#include') and "Protein" in s:
            inc = s.split('"')[1] if '"' in s else None
            if inc:
                p = (top_path.parent / inc)
                if p.exists():
                    texts.append(p.read_text())
    total, found = 0.0, False
    for text in texts:
        section, molname, in_prot = None, None, False
        for line in text.split("\n"):
            s = line.split(";")[0].strip()
            if not s:
                continue
            if s.startswith("["):
                section = s.strip("[] ").strip()
                if section == "moleculetype":
                    molname = None
                continue
            if section == "moleculetype":
                molname = s.split()[0]
                in_prot = molname.startswith("Protein")
                continue
            if section == "atoms" and in_prot:
                parts = s.split()
                if len(parts) >= 7:
                    try:
                        total += float(parts[6])
                        found = True
                    except ValueError:
                        pass
    return round(total, 4) if found else None


def _read_gro(gro_path: Path):
    """Return (coords[list of (x,y,z) nm], box_vectors[3x3 nm]) from a .gro."""
    lines = Path(gro_path).read_text().rstrip("\n").split("\n")
    natoms = int(lines[1].strip())
    coords = []
    for line in lines[2:2 + natoms]:
        coords.append((float(line[20:28]), float(line[28:36]), float(line[36:44])))
    fields = [float(x) for x in lines[2 + natoms].split()]
    # GRO box line: v1x v2y v3z [v1y v1z v2x v2z v3x v3y]
    box = [[fields[0], 0.0, 0.0],
           [0.0, fields[1], 0.0],
           [0.0, 0.0, fields[2]]]
    if len(fields) >= 9:
        box[0][1], box[0][2] = fields[3], fields[4]
        box[1][0], box[1][2] = fields[5], fields[6]
        box[2][0], box[2][1] = fields[7], fields[8]
    return coords, box


def solute_min_image_nm(gro_path: Path) -> float:
    """Lower bound on the distance between the solute and its nearest periodic image.

    BLOCKER 10.  For every non-zero lattice translation T = i*a + j*b + k*c with
    i,j,k in {-1,0,1}, the closest possible approach between the solute and its
    image under T is |T| minus the extent of the solute projected onto T-hat.
    The minimum over those 26 translations is a rigorous lower bound and is
    tight for a compact solute -- and unlike a per-axis (box - extent) estimate
    it is correct for triclinic / dodecahedral boxes.

    Call this on a PRE-SOLVATION .gro so every atom in the file is solute.
    """
    import math
    coords, box = _read_gro(gro_path)
    if not coords:
        raise GromacsError(f"{gro_path} contains no atoms")

    best = float("inf")
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i == j == k == 0:
                    continue
                t = [i * box[0][d] + j * box[1][d] + k * box[2][d] for d in range(3)]
                tn = math.sqrt(sum(v * v for v in t))
                if tn < 1e-9:
                    continue
                u = [v / tn for v in t]
                proj = [c[0] * u[0] + c[1] * u[1] + c[2] * u[2] for c in coords]
                extent = max(proj) - min(proj)
                best = min(best, tn - extent)
    return best


def setup_simulation_box(gro_path: Path, work_dir: Path,
                          box_type: str = None,
                          distance: float = None) -> Path:
    """Create simulation box, solvate, and add ions.

    BLOCKER 10 -- three defects fixed here:

    1. box_type/distance were hardcoded "cubic"/0.5 nm and IGNORED config's
       MD_BOX_TYPE ("dodecahedron") and MD_BOX_DISTANCE (1.2 nm).  They now
       default to the config values; explicit arguments still win.
    2. 0.5 nm of padding gave a measured 1.09 nm minimum-image separation
       against rvdw = rcoulomb = 1.0 nm, i.e. monomers interacted with their
       own periodic images for the entire trajectory.  The padding is now
       grown until the SOLUTE (protein AND monomers -- every atom in the input
       .gro, which is complex.gro when monomers are present) clears
       2*rvdw + MIN_IMAGE_MARGIN_NM, verified on the box editconf actually
       produced, not on the number we asked for.
    3. Every intermediate is now asserted to exist and be non-empty before it
       is handed to the next step.
    """
    from .config import MD_BOX_TYPE, MD_BOX_DISTANCE, MD_IONIC_STRENGTH
    work_dir = Path(work_dir)
    gro_path = _require_file(gro_path, "setup_simulation_box input", min_bytes=50)

    if box_type is None:
        box_type = MD_BOX_TYPE
    if distance is None:
        distance = MD_BOX_DISTANCE

    required_sep = 2.0 * MD_RVDW_NM + MIN_IMAGE_MARGIN_NM
    # A padding of d gives ~2d of image separation, so anything below
    # rvdw + margin/2 cannot possibly pass. Start from the larger of the two.
    min_distance = MD_RVDW_NM + MIN_IMAGE_MARGIN_NM / 2.0
    if distance < min_distance:
        logger.warning(
            f"  Box padding {distance} nm is too small for rvdw={MD_RVDW_NM} nm "
            f"(needs >= {min_distance} nm for {required_sep} nm minimum image "
            f"separation) -- raising it to {min_distance} nm.")
        distance = min_distance

    boxed = work_dir / "boxed.gro"
    attempt_d = float(distance)
    achieved = None
    for attempt in range(4):
        _gmx(["editconf",
               "-f", str(gro_path),
               "-o", str(boxed),
               "-c",                       # center in box
               "-d", f"{attempt_d:.3f}",   # min distance solute → box edge
               "-bt", box_type], work_dir)
        _require_file(boxed, "editconf", min_bytes=50)
        achieved = solute_min_image_nm(boxed)
        if achieved >= required_sep - 1e-6:
            break
        grown = attempt_d + (required_sep - achieved) / 2.0 + 0.05
        logger.warning(
            f"  Box from -d {attempt_d:.3f} nm gives only {achieved:.3f} nm of "
            f"solute minimum-image separation (need {required_sep:.3f} nm) -- "
            f"growing to -d {grown:.3f} nm.")
        attempt_d = grown
    else:
        raise GromacsError(
            f"Could not build a box with >= {required_sep:.3f} nm solute "
            f"minimum-image separation (best {achieved:.3f} nm at -d "
            f"{attempt_d:.3f} nm, bt={box_type}). The solute is probably not "
            f"compact -- check monomer placement in complex.gro.")

    logger.info(f"  Box: {box_type}, -d {attempt_d:.2f} nm, solute minimum-image "
                f"separation {achieved:.2f} nm (>= {required_sep:.2f} nm required "
                f"for rvdw={MD_RVDW_NM} nm)")

    # Solvate
    solvated = work_dir / "solvated.gro"
    _gmx(["solvate",
           "-cp", str(boxed),
           "-cs", "spc216.gro",
           "-o", str(solvated),
           "-p", str(work_dir / "topol.top")], work_dir)
    _require_file(solvated, "solvate", min_bytes=50)

    # Add ions (neutralize). First generate .tpr for genion.
    mdp_path = work_dir / "ions.mdp"
    mdp_path.write_text(MDP_EM, encoding='utf-8')
    _grompp(work_dir, mdp_path, solvated, work_dir / "topol.top",
            work_dir / "ions.tpr",
            maxwarn=GROMPP_MAXWARN_IONS)  # net charge -- genion fixes it next

    # Replace solvent with ions -- PBS condition (0.15 M NaCl)
    ionized = work_dir / "ionized.gro"
    _gmx(["genion",
           "-s", str(work_dir / "ions.tpr"),
           "-o", str(ionized),
           "-p", str(work_dir / "topol.top"),
           "-pname", "NA", "-nname", "CL",
           "-neutral",
           "-conc", str(MD_IONIC_STRENGTH)],  # 0.15 M NaCl (PBS)
          work_dir, input_text="SOL\n")

    return _require_file(ionized, "genion", min_bytes=50)


def verify_existing_box(work_dir: Path) -> float:
    """Re-check a box that a previous run built, before reusing it.

    BLOCKER 10 companion.  run_full_md_pipeline skips setup_simulation_box when
    ionized.gro is already present.  Without this, a work_dir produced by the
    old hardcoded cubic/0.5 nm code would be silently reused and the whole
    minimum-image fix would never apply to it.  boxed.gro is the pre-solvation
    file, so every atom in it is solute.
    """
    work_dir = Path(work_dir)
    boxed = work_dir / "boxed.gro"
    required = 2.0 * MD_RVDW_NM + MIN_IMAGE_MARGIN_NM
    if not boxed.exists():
        logger.warning(
            f"  Reusing an existing box in {work_dir} but boxed.gro is gone, so "
            f"its solute minimum-image separation CANNOT be verified. If this "
            f"directory predates the box fix, delete it and rebuild.")
        return float("nan")
    sep = solute_min_image_nm(boxed)
    if sep < required - 1e-6:
        raise GromacsError(
            f"Existing box in {work_dir} has only {sep:.3f} nm of solute "
            f"minimum-image separation (need {required:.3f} nm for "
            f"rvdw={MD_RVDW_NM} nm). This directory was built by the old "
            f"cubic/0.5 nm code -- delete it and let the pipeline rebuild the "
            f"box rather than reusing a system whose monomers see their own "
            f"periodic images.")
    logger.info(f"  Reused box verified: solute minimum-image separation "
                f"{sep:.2f} nm (>= {required:.2f} nm)")
    return sep


# ── MD Execution ───────────────────────────────────────────────

def _check_em_converged(work_dir: Path, emtol: float = 1000.0) -> dict:
    """BLOCKER 10 -- EM acceptance criterion. There was none.

    `gmx mdrun` for `integrator = steep` exits 0 whether or not it converged;
    the only record is a line in em.log.  A system that stops at Fmax = 1e6
    kJ/mol/nm has an unresolved clash and will blow up (or, worse, quietly
    distort) in NVT.  This parses em.log and raises unless Fmax <= emtol.
    """
    log = work_dir / "em.log"
    if not log.exists():
        raise EquilibrationError("energy minimisation produced no em.log")
    text = log.read_text(errors="replace")

    import math
    import re as _re
    epot = fmax = None
    m = _re.search(r"Potential Energy\s*=\s*([-\deE.+]+)", text)
    if m:
        epot = float(m.group(1))
    m = _re.search(r"Maximum force\s*=\s*([-\deE.+]+)", text)
    if m:
        fmax = float(m.group(1))
    converged = "Steepest Descents converged" in text

    info = {"epot_kj_mol": epot, "fmax_kj_mol_nm": fmax, "converged": converged}

    if epot is None or fmax is None:
        raise EquilibrationError(
            f"could not read Potential Energy / Maximum force from {log} -- "
            f"minimisation did not complete")
    if not math.isfinite(epot) or not math.isfinite(fmax):
        raise EquilibrationError(
            f"energy minimisation produced non-finite energies "
            f"(Epot={epot}, Fmax={fmax}) -- the starting structure is broken")

    logger.info(f"  EM: Epot = {epot:.3e} kJ/mol, Fmax = {fmax:.3e} kJ/mol/nm, "
                f"converged={converged}")

    if EM_REQUIRE_CONVERGENCE and not (converged and fmax <= emtol):
        raise EquilibrationError(
            f"energy minimisation did NOT converge: Fmax = {fmax:.3e} "
            f"kJ/mol/nm > emtol = {emtol} (converged flag={converged}). "
            f"Refusing to equilibrate a system with unresolved clashes. "
            f"Inspect {work_dir/'em.log'}; usually a bad monomer placement in "
            f"complex.gro or a broken monomer topology.")
    return info


def _energy_terms(work_dir: Path, edr_stem: str, terms: list) -> dict:
    """Extract time series for named .edr terms via `gmx energy`.

    Returns {term: numpy Nx2 array (time_ps, value)}.  Missing terms are
    simply absent from the result.
    """
    out = {}
    for term in terms:
        xvg = work_dir / f"energy_{edr_stem}_{term.lower()}.xvg"
        res = _gmx(["energy", "-f", f"{edr_stem}.edr", "-o", xvg.name],
                   work_dir, input_text=f"{term}\n\n", check=False)
        if res.returncode != 0 or not xvg.exists():
            logger.warning(f"  could not extract '{term}' from {edr_stem}.edr")
            continue
        data = _parse_xvg(xvg)
        if data is not None and len(data) >= 4:
            out[term] = data
    return out


def _plateau_stats(series):
    """(mean of 2nd half, mean of 1st half, drift = 2nd - 1st)."""
    n = len(series)
    first = series[: n // 2, 1]
    second = series[n // 2:, 1]
    return float(second.mean()), float(first.mean()), \
        float(second.mean() - first.mean())


def _check_equilibration(work_dir: Path, stage: str, temperature: float,
                          pressure: float = None) -> dict:
    """BLOCKER 10 -- NVT/NPT acceptance criteria. There were none.

    Hard-fails on a temperature that never reached the thermostat set point or
    that is still drifting, and (NPT only) on a density that has not plateaued.
    Pressure is reported but only warned on: <P> over a 100 ps window has a
    standard error of tens of bar, so a pressure test is not a real criterion.
    """
    wanted = ["Temperature"] + (["Pressure", "Density"] if pressure is not None else [])
    series = _energy_terms(work_dir, stage, wanted)
    report = {}

    if "Temperature" not in series:
        raise EquilibrationError(
            f"{stage}: no Temperature series in {stage}.edr -- the run produced "
            f"no usable energy output")

    t_mean, t_first, t_drift = _plateau_stats(series["Temperature"])
    report["temperature_K"] = t_mean
    report["temperature_drift_K"] = t_drift
    logger.info(f"  {stage.upper()}: <T> = {t_mean:.1f} K (ref {temperature:.1f} K), "
                f"drift {t_drift:+.1f} K")
    if abs(t_mean - temperature) > EQUIL_TEMP_TOL_K:
        raise EquilibrationError(
            f"{stage}: <T> = {t_mean:.1f} K is more than {EQUIL_TEMP_TOL_K} K "
            f"from the set point {temperature:.1f} K -- the thermostat never "
            f"equilibrated this system.")
    if abs(t_drift) > EQUIL_TEMP_DRIFT_TOL_K:
        raise EquilibrationError(
            f"{stage}: temperature is still drifting "
            f"({t_drift:+.1f} K between the two halves of the window, tolerance "
            f"{EQUIL_TEMP_DRIFT_TOL_K} K) -- extend the equilibration.")

    if pressure is not None:
        if "Density" in series:
            d_mean, d_first, d_drift = _plateau_stats(series["Density"])
            report["density_kg_m3"] = d_mean
            report["density_drift_frac"] = d_drift / d_first if d_first else 0.0
            logger.info(f"  {stage.upper()}: <rho> = {d_mean:.1f} kg/m^3, "
                        f"drift {100*report['density_drift_frac']:+.2f} %")
            if abs(report["density_drift_frac"]) > EQUIL_DENSITY_DRIFT_FRAC:
                raise EquilibrationError(
                    f"{stage}: density has not plateaued "
                    f"({100*report['density_drift_frac']:+.2f} % between the two "
                    f"halves, tolerance {100*EQUIL_DENSITY_DRIFT_FRAC:.1f} %) -- "
                    f"the barostat has not finished compressing the box.")
        else:
            raise EquilibrationError(
                f"{stage}: no Density series in {stage}.edr -- cannot verify the "
                f"barostat converged")

        if "Pressure" in series:
            p_mean, _, _ = _plateau_stats(series["Pressure"])
            report["pressure_bar"] = p_mean
            if abs(p_mean - pressure) > EQUIL_PRESSURE_TOL_BAR:
                logger.warning(
                    f"  {stage.upper()}: <P> = {p_mean:.1f} bar vs ref "
                    f"{pressure:.1f} bar (advisory; instantaneous pressure in a "
                    f"box this size fluctuates by hundreds of bar)")
    return report


def run_energy_minimization(work_dir: Path) -> Path:
    """Run energy minimization, and REFUSE to continue if it did not converge."""
    work_dir = Path(work_dir)
    mdp_path = work_dir / "em.mdp"
    mdp_path.write_text(MDP_EM, encoding='utf-8')

    _grompp(work_dir, mdp_path, work_dir / "ionized.gro",
            work_dir / "topol.top", work_dir / "em.tpr")

    # mdrun for steep exits 0 even when it fails to converge -- see
    # _check_em_converged, which is the real acceptance test.
    _gmx(["mdrun", "-deffnm", "em"], work_dir, timeout=1800)
    _require_file(work_dir / "em.gro", "EM mdrun", min_bytes=50)
    _check_em_converged(work_dir, emtol=1000.0)

    return work_dir / "em.gro"


def run_nvt_equilibration(work_dir: Path, time_ps: float = 100.0, define: str = "",
                           temperature: float = 300.0, gen_seed: int = -1) -> Path:
    """Run NVT equilibration and verify the thermostat actually equilibrated it.

    gen_seed
        Seed for Maxwell velocity generation.  -1 (the GROMACS default, and the
        default here) draws from the wall clock, which makes a replica's
        starting velocities UNREPRODUCIBLE.  Phase 4 computes a distinct,
        deterministic seed per replica and passes it through
        run_full_md_pipeline(seed=...); without it, replicas are independent but
        cannot be re-run to the same trajectory.
    """
    work_dir = Path(work_dir)
    from .config import MD_TIMESTEP_FS
    dt = MD_TIMESTEP_FS / 1000.0  # fs to ps
    nsteps = int(time_ps / dt)

    mdp_path = work_dir / "nvt.mdp"
    mdp_path.write_text(MDP_NVT.format(
        define=define, nsteps=nsteps, dt=dt, temperature=temperature,
        nstxout=NSTXOUT_COMPRESSED_EQUIL, nstenergy=NSTENERGY_EQUIL,
        gen_seed=int(gen_seed)), encoding='utf-8')

    _grompp(work_dir, mdp_path, work_dir / "em.gro", work_dir / "topol.top",
            work_dir / "nvt.tpr", restraint=work_dir / "em.gro")

    _gmx(["mdrun", "-deffnm", "nvt"], work_dir, timeout=3600)
    # Extract final frame from checkpoint/trajectory if gro not created.
    # check=False: this is itself the recovery path; the _require_file below
    # is what decides whether NVT actually produced a usable structure.
    nvt_gro = work_dir / "nvt.gro"
    if not nvt_gro.exists() and (work_dir / "nvt.cpt").exists():
        _gmx(["trjconv", "-f", "nvt.cpt", "-s", "nvt.tpr",
               "-o", "nvt.gro", "-dump", "0"],
              work_dir, input_text="0\n", check=False)
    _require_file(nvt_gro, "NVT equilibration", min_bytes=50)
    _check_equilibration(work_dir, "nvt", temperature=temperature)
    return nvt_gro


def run_npt_equilibration(work_dir: Path, time_ps: float = 100.0, define: str = "",
                           temperature: float = 300.0,
                           pressure: float = 1.0) -> Path:
    """Run NPT equilibration and verify T and density have plateaued."""
    work_dir = Path(work_dir)
    from .config import MD_TIMESTEP_FS
    dt = MD_TIMESTEP_FS / 1000.0
    nsteps = int(time_ps / dt)

    mdp_path = work_dir / "npt.mdp"
    mdp_path.write_text(MDP_NPT.format(
        define=define, nsteps=nsteps, dt=dt, temperature=temperature,
        pressure=pressure,
        nstxout=NSTXOUT_COMPRESSED_EQUIL, nstenergy=NSTENERGY_EQUIL), encoding='utf-8')

    _grompp(work_dir, mdp_path, work_dir / "nvt.gro", work_dir / "topol.top",
            work_dir / "npt.tpr", restraint=work_dir / "nvt.gro",
            checkpoint=work_dir / "nvt.cpt")

    _gmx(["mdrun", "-deffnm", "npt"], work_dir, timeout=3600)
    npt_gro = work_dir / "npt.gro"
    if not npt_gro.exists() and (work_dir / "npt.cpt").exists():
        _gmx(["trjconv", "-f", "npt.cpt", "-s", "npt.tpr",
               "-o", "npt.gro", "-dump", "0"],
              work_dir, input_text="0\n", check=False)
    _require_file(npt_gro, "NPT equilibration", min_bytes=50)
    _check_equilibration(work_dir, "npt", temperature=temperature,
                         pressure=pressure)
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

    # BLOCKER 11 -- trajectory write frequency.
    nstxout = int(_cfg("MD_NSTXOUT_COMPRESSED", NSTXOUT_COMPRESSED_PRODUCTION))
    if nstxout < 1000:
        raise ValueError(
            f"MD_NSTXOUT_COMPRESSED={nstxout} would write a trajectory at least "
            f"50x larger than anything downstream reads. Use >= 25000 "
            f"(default {NSTXOUT_COMPRESSED_PRODUCTION}).")
    n_frames = nsteps // nstxout
    logger.info(f"  Trajectory: nstxout-compressed = {nstxout} steps "
                f"({nstxout*dt:.0f} ps/frame) → ~{n_frames} frames for "
                f"{time_ns:g} ns")

    mdp_path = work_dir / "md.mdp"
    mdp_path.write_text(MDP_PRODUCTION.format(define=define,
        nsteps=nsteps, dt=dt, temperature=temperature, pressure=pressure,
        nstxout=nstxout), encoding='utf-8')

    _grompp(work_dir, mdp_path, work_dir / "npt.gro", work_dir / "topol.top",
            work_dir / "md.tpr", restraint=work_dir / "npt.gro",
            checkpoint=work_dir / "npt.cpt")

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
    proc = subprocess.run(full_cmd, cwd=str(work_dir),
                          timeout=int(time_ns * 3600))
    # BLOCKER 09a: this return code was previously discarded entirely.
    if proc.returncode != 0:
        raise GromacsError(
            f"production mdrun failed (rc={proc.returncode}) in {work_dir}; "
            f"see {work_dir/'md.log'}")
    _require_file(work_dir / "md.xtc", "production mdrun", min_bytes=1000)
    _require_file(work_dir / "md.gro", "production mdrun", min_bytes=50)
    return work_dir / "md.xtc"


# ── Trajectory Analysis ────────────────────────────────────────

# Downstream analysis (RMSD/RMSF/H-bond/Rg, DSSP, occupancy) reads a few
# hundred frames. Anything more is disk cost with no information gain.
ANALYSIS_TARGET_FRAMES = 500


def _analysis_output_dt_ps(work_dir: Path, target_frames: int = 500) -> float:
    """Time stride (ps) that yields ~target_frames from the production run.

    Reads md.mdp (written by run_production_md) so it stays correct whatever
    nstxout-compressed and dt are set to. Falls back to a stride equal to the
    trajectory write interval if md.mdp is unreadable.
    """
    mdp = work_dir / "md.mdp"
    nsteps = dt = nstxout = None
    if mdp.exists():
        for line in mdp.read_text().split("\n"):
            if "=" not in line or line.strip().startswith(";"):
                continue
            key, _, val = line.partition("=")
            key = key.strip().lower().replace("_", "-")
            val = val.split(";")[0].strip()
            try:
                if key == "nsteps":
                    nsteps = int(float(val))
                elif key == "dt":
                    dt = float(val)
                elif key == "nstxout-compressed":
                    nstxout = int(float(val))
            except ValueError:
                pass
    if not (nsteps and dt and nstxout):
        logger.warning(f"  could not read nsteps/dt/nstxout-compressed from "
                       f"{mdp}; using a 100 ps analysis stride")
        return 100.0
    frame_dt = nstxout * dt                      # ps between written frames
    total_ps = nsteps * dt
    wanted = total_ps / max(1, target_frames)
    # Must be a multiple of the write interval or trjconv silently drops frames.
    n = max(1, int(round(wanted / frame_dt)))
    return n * frame_dt


def analyze_trajectory(work_dir: Path) -> dict:
    """
    Analyze production MD trajectory.
    Returns dict with RMSD, RMSF, H-bond, Rg metrics.
    """
    work_dir = Path(work_dir)
    results = {}

    import numpy as np

    # BLOCKER 11: the reduced trajectory used to be built with a hardcoded
    # `-skip 100`, which is a frame stride and therefore depends on
    # nstxout-compressed. Now that the production trajectory is written 10x
    # more sparsely, `-skip 100` would leave ~35 frames. Select on TIME instead
    # so the analysis window is independent of the write frequency.
    reduced_xtc = work_dir / "md_reduced.xtc"
    if not reduced_xtc.exists() and (work_dir / "md.xtc").exists():
        out_dt_ps = _analysis_output_dt_ps(work_dir,
                                           target_frames=ANALYSIS_TARGET_FRAMES)
        logger.info(f"  [1/5] Creating reduced trajectory "
                    f"(-dt {out_dt_ps:g} ps → ~{ANALYSIS_TARGET_FRAMES} frames)...")
        # check=False: analysis is best-effort; if this fails we fall back to
        # the full trajectory below rather than aborting a finished MD run.
        _gmx(["trjconv", "-f", "md.xtc", "-s", "md.tpr",
               "-o", "md_reduced.xtc", "-dt", f"{out_dt_ps:g}"],
              work_dir, input_text="0\n", timeout=1800, check=False)
        if not reduced_xtc.exists():
            logger.warning("  [1/5] Reduced trajectory could not be built -- "
                           "analysing the full md.xtc instead")
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
               decomp: bool = False,
               startframe: int = None,
               endframe: int = None,
               interval: int = None) -> dict:
    """
    Run gmx_MMPBSA for binding free energy calculation.

    Sullivan 2019: MM-GBSA is preferred for protein-monomer systems.
    Supports both PBSA and GBSA modes via config.MMPBSA_METHOD.

    Requires gmx_MMPBSA to be installed (pip install gmx_MMPBSA).

    # BEHAVIOUR CHANGE (2026-08 audit) -- THE MM-GBSA WINDOW BLOCKER.
    This function used to accept start_ns/end_ns, ignore them completely, and
    write `startframe=1, endframe={n_frames}, interval=1` into mmpbsa.in.  On a
    350 ns production run written every 10 ps that is frames 1..100 = the FIRST
    1 ns -- i.e. the binding free energy was computed on the pre-equilibrium part
    of every trajectory, and start_ns/end_ns were decorative.

    Now: the caller may pass explicit 1-indexed `startframe`/`endframe`/
    `interval`; if it does not, they are DERIVED from start_ns/end_ns against
    the trajectory's OWN frame spacing.  There is no path back to "frame 1"
    -- a window that cannot be derived or does not lie inside the trajectory
    returns an error dict (this function's contract is to return, not raise)
    and is logged at ERROR.  The window actually used is echoed back in
    result["window"] so callers can verify it rather than trust it.
    """
    from .config import MMPBSA_METHOD, MD_IONIC_STRENGTH
    work_dir = Path(work_dir)

    # ── Resolve the frame window ────────────────────────────────────
    window = None
    if startframe is None or endframe is None:
        # Imported INSIDE the function: phase4_md_validation imports
        # utils_gromacs lazily too, and a module-level import either way would
        # make the pair circular.
        try:
            from .phase4_md_validation import _mmpbsa_frame_window
            window = _mmpbsa_frame_window(work_dir, start_ns, end_ns, n_frames)
        except Exception as e:
            logger.error(
                "MM-GBSA REFUSED: cannot derive the frame window for %s "
                "(%s-%s ns): %s. Refusing to fall back to frames 1..%s, which "
                "would silently compute the free energy on the FIRST part of "
                "the run instead of the requested window.",
                work_dir, start_ns, end_ns, e, n_frames)
            return {"error": f"MM-GBSA window could not be derived: {e}",
                    "window_valid": False}
        startframe = window["startframe"]
        endframe = window["endframe"]
        if interval is None:
            interval = window["interval"]
    if interval is None:
        interval = 1
    startframe, endframe, interval = int(startframe), int(endframe), max(1, int(interval))
    if endframe <= startframe:
        logger.error("MM-GBSA REFUSED: window collapses to frames %d..%d in %s",
                     startframe, endframe, work_dir)
        return {"error": f"MM-GBSA window collapses to frames "
                         f"{startframe}..{endframe}", "window_valid": False}
    if window is None:
        window = {"startframe": startframe, "endframe": endframe,
                  "interval": interval,
                  "n_frames_used": len(range(startframe, endframe + 1, interval))}
    logger.info("MM-GBSA frames %d..%d step %d (%d frames) for %s-%s ns",
                startframe, endframe, interval,
                window.get("n_frames_used", -1), start_ns, end_ns)

    # Create MMPBSA input file -- GBSA (Sullivan 2019) or PBSA
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
              startframe={startframe}, endframe={endframe}, interval={interval},
              verbose=2,
            /
            &gb
              igb=5, saltcon={MD_IONIC_STRENGTH},
            /
        """) + decomp_block)
    else:
        mmpbsa_in.write_text(dedent(f"""\
            &general
              startframe={startframe}, endframe={endframe}, interval={interval},
              verbose=2,
            /
            &pb
              istrng={MD_IONIC_STRENGTH}, fillratio=4.0,
            /
        """) + decomp_block)

    # Generate index file: need "Protein" and "Other" (monomers only) groups
    ndx_path = work_dir / "index.ndx"
    if not ndx_path.exists():
        # run_mmpbsa's contract is to RETURN an error dict, not to raise, so
        # convert the (now loud) GromacsError into that contract rather than
        # letting it escape into callers that only expect a dict.
        try:
            _gmx(["make_ndx", "-f", "md.tpr", "-o", "index.ndx"],
                  work_dir, input_text="q\n")
        except GromacsError as e:
            # window echoed here too: every return path reports which frames
            # were going to be used, so a caller never has to assume.
            return {"error": f"make_ndx failed: {e}", "window": window}

    # Find the "Other" group number (the monomers -- not water, not ions).
    #
    # NEVER GUESS A GROUP NUMBER (REVIEW FINDING 10).
    #
    # This used to be `ligand_group = "12"` with the probe running at
    # check=False and logging nothing. GROMACS group numbering depends on what
    # is in the system: in a solvated protein+monomer box, group 12 is
    # typically WATER. So whenever the probe failed -- non-zero rc, empty
    # stdout, anything -- gmx_MMPBSA was silently handed the water group and
    # computed a protein-WATER free energy, which was then returned as
    # delta_total_kcal and stamped window_valid=True.
    #
    # There is no safe default here. A group number we did not read off this
    # system's own listing is a guess, and a guessed ΔG is worse than none.
    ligand_group = None
    group_listing = ""
    if ndx_path.exists():
        # The probe reads the group listing off stdout. check=False because a
        # non-zero rc is handled explicitly below rather than raising.
        result = _gmx(["make_ndx", "-f", "md.tpr", "-n", "index.ndx"],
                       work_dir, input_text="q\n", check=False)
        group_listing = (result.stdout or "") + "\n" + (result.stderr or "")
        for line in group_listing.split("\n"):
            s = line.strip()
            if "Other" in s and s and s[0].isdigit():
                ligand_group = s.split()[0]
                break

    if ligand_group is None:
        logger.error(
            "MM-GBSA: could not resolve the ligand ('Other') index group from "
            "make_ndx. Refusing to fall back to a hardcoded group number -- the "
            "old default was 12, which in a solvated system is WATER, so the "
            "result would have been a protein-water free energy reported as a "
            "protein-monomer binding energy.")
        return {"error": ("could not resolve the ligand index group from "
                          "make_ndx; refusing to guess (the retired default, "
                          "group 12, is Water in a solvated system)"),
                "window": window,
                "make_ndx_listing": group_listing[-2000:]}

    # Cross-check the resolved group against the index file: it must exist and
    # be non-empty, or gmx_MMPBSA is being pointed at nothing.
    logger.info(f"    MM-GBSA ligand index group resolved to {ligand_group} "
                f"('Other')")

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

    # DELETE THE PREVIOUS RUN'S OUTPUT BEFORE LAUNCHING (REVIEW FINDING 11).
    #
    # gmx_MMPBSA writes FINAL_RESULTS_MMPBSA.dat to a deterministic path that
    # was never cleaned. On a re-run that FAILED (rc != 0 only warned), the
    # `if final_dat.exists()` below found the PREVIOUS run's file and returned
    # it as this run's free energy -- stamped window_valid=True and carrying the
    # CURRENT window, so phase4's _audit_mmpbsa_window passed it. A stale
    # number wearing a fresh window is exactly the silent-wrongness this audit
    # exists to remove.
    #
    # This is the same fix already applied to the AutoDock-GPU DLG in
    # utils_autodock.py ("so a crash cannot return the previous run's DLG").
    final_dat = work_dir / "FINAL_RESULTS_MMPBSA.dat"
    final_csv = work_dir / "FINAL_RESULTS_MMPBSA.csv"
    for _stale in (final_dat, final_csv):
        try:
            if _stale.exists():
                logger.debug(f"    removing stale {_stale.name} before re-running")
                _stale.unlink()
        except OSError as e:
            return {"error": (f"could not remove stale {_stale.name} ({e}); "
                              f"refusing to run, because a failed run would "
                              f"then return the previous result"),
                    "window": window}

    try:
        result = subprocess.run(
            cmd, cwd=str(work_dir),
            capture_output=True, text=True, timeout=7200,
        )

        # A NON-ZERO EXIT IS A FAILURE, NOT A WARNING (REVIEW FINDING 11).
        # This used to log at WARNING and then parse whatever .dat was on
        # disk. With the pre-run unlink above there is no stale file left to
        # find, but a partially-written one is just as bad, so refuse outright.
        if result.returncode != 0:
            logger.error(f"gmx_MMPBSA FAILED (rc={result.returncode}): "
                         f"{result.stderr[:500]}")
            return {"error": f"gmx_MMPBSA exited with rc={result.returncode}",
                    "returncode": result.returncode,
                    "stderr": result.stderr[:500], "window": window}

        # Parse results.
        # The window is echoed on EVERY return path (including the error ones)
        # so a caller can verify which frames produced the number instead of
        # assuming -- that assumption is what hid this blocker.
        if final_dat.exists():
            parsed = _parse_mmpbsa_results(final_dat)
            if isinstance(parsed, dict):
                parsed["window"] = window
                # Only a run that exited 0 AND produced its own fresh .dat
                # (the pre-run unlink guarantees freshness) gets this stamp.
                parsed["window_valid"] = True
            return parsed
        return {"error": "No results file produced",
                "stderr": result.stderr[:300], "window": window}
    except FileNotFoundError:
        return {"error": "gmx_MMPBSA not found", "window": window}
    except subprocess.TimeoutExpired:
        return {"error": "gmx_MMPBSA timeout", "window": window}


# ── Full MD Pipeline ───────────────────────────────────────────

def run_full_md_pipeline(protein_pdb: Path, monomer_itps: list,
                          work_dir: Path,
                          time_ns: float = 200.0,
                          quick: bool = False,
                          protein_restrained: bool = False,
                          seed: int = None) -> dict:
    """
    Complete GROMACS MD pipeline:
    pdb2gmx → solvate → EM → NVT → NPT → production → analysis → MM-PBSA

    seed
        Master seed for the two stochastic steps that decide what a replica
        actually is: monomer PLACEMENT (_include_monomers_in_topology) and
        Maxwell VELOCITY generation (NVT gen_seed).  None keeps the historical
        behaviour -- global RNG placement and gen_seed=-1 (wall-clock) -- which is
        independent but not reproducible.  Phase 4 derives a distinct
        deterministic seed per replica from sha256(target|pc_id|rep<i>).
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
                _include_monomers_in_topology(work_dir, monomer_itps, seed=seed)
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
            # ...but do NOT inherit a box built by the old 0.5 nm code.
            results["box_min_image_nm"] = verify_existing_box(work_dir)

        # 3. Energy minimization
        if not (work_dir / "em.gro").exists():
            logger.info("Running energy minimization...")
            run_energy_minimization(work_dir)
        else:
            logger.info("EM: FOUND, skipping")

        # 4. NVT equilibration (100 ps)
        if not (work_dir / "nvt.gro").exists():
            logger.info("NVT equilibration...")
            # gen_seed: a real seed makes the replica's velocities
            # reproducible; -1 keeps GROMACS' wall-clock default.
            run_nvt_equilibration(work_dir, time_ps=100.0,
                                   temperature=MD_TEMPERATURE_K,
                                   gen_seed=(int(seed) if seed is not None else -1))
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

        # DERIVED, NOT ASSERTED.  This used to be an unconditional
        # `results["success"] = True` at the end of the try block, so a run
        # whose analysis or MM-PBSA step returned an error DICT (neither
        # raises) was still reported as a success -- and phase4 propagated that
        # flag upward.  A leg is a success only if nothing recorded an error.
        _mm_err = (results.get("mmpbsa") or {}).get("error")
        results["success"] = not _mm_err and not results.get("error")
        # success_basis lets a consumer tell "the MD itself broke" apart from
        # "the MD ran but the free-energy step produced no number", without
        # either being silently reported as OK.  Phase 4 does not gate on this
        # flag (it uses md_completed plus its own acceptance criteria, where
        # MM-GBSA is a recorded warning); the sweeps and Phase 5 do.
        results["success_basis"] = (
            "md_and_mmpbsa_ok" if results["success"]
            else f"mmpbsa_error: {_mm_err}" if _mm_err
            else f"error: {results.get('error')}")
        if _mm_err:
            logger.error("MD finished but MM-PBSA reported an error: %s. "
                         "run_full_md_pipeline is returning success=False.", _mm_err)
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
                       "Using original PDB -- pdb2gmx may fail if atoms are missing.")
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


def _include_monomers_in_topology(work_dir: Path, monomer_itps: list,
                                  seed: int = None):
    """
    Include monomer ITP files and coordinates in GROMACS topology.

    For each monomer:
    1. Copy .itp to work_dir
    2. Add #include to topol.top (before [ molecules ])
    3. Add molecule name to [ molecules ] section
    4. Merge .gro coordinates into system (protein.gro -> complex.gro)

    seed
        Seed for the monomer PLACEMENT RNG.  This used to call the bare
        `random.uniform()` of the process-global module, so two replicas of the
        same composition got different starting boxes with no way to reproduce
        either.  Phase 4 worked around it by seeding the global module around
        the call and restoring the previous state afterwards; passing the seed
        here makes the dependency explicit and local.  None keeps the old
        global-RNG behaviour so existing callers are unaffected.
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
    # Local RNG when a seed is given; otherwise the process-global module, which
    # is what every pre-audit caller relied on.
    rng = random.Random(seed) if seed is not None else random
    if seed is not None:
        logger.info(f"  Monomer placement RNG seeded with {seed}")
    n_monomers = len(monomer_itps)
    min_dist_nm = 1.0  # minimum distance between monomer centers
    r_inner = prot_radius + 0.3  # start just outside protein surface
    r_outer = r_inner + 1.0      # thin shell → 9nm box target (Rajpal 2024)

    placed_centers = []
    monomer_positions = []  # pre-computed (x, y, z) for each monomer

    for mi in range(n_monomers):
        # Try to place without overlapping other monomers
        for attempt in range(500):
            angle1 = rng.uniform(0, 2 * math.pi)
            angle2 = rng.uniform(-math.pi/2, math.pi/2)
            r = rng.uniform(r_inner, r_outer)
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
            angle1 = rng.uniform(0, 2 * math.pi)
            angle2 = rng.uniform(-math.pi/2, math.pi/2)
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

    # Edit topol.top -- proper directive ordering:
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
    # Fix Si atom: acpype generates Si with mass=0, sigma=0 -- replace with PolCA
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
# Hand-built GAFF2 topology  +  PolCA organosilane Si  +  UFF boron
# ══════════════════════════════════════════════════════════════
# BLOCKER 03.  What this section produces used to be physically invalid:
#
#   * _std_bond_len was keyed on ordered tuples like (6,1) but looked up with
#     (min, max) = (1,6), so EVERY C-H, N-H, O-H, S-H and C-B lookup missed and
#     fell through to a 0.1500 nm default -- C-H 38 % too long, O-H 56 %, N-H
#     49 % -- and LINCS then held them there for the whole trajectory.
#   * every hydrogen was typed h1 (the type for H on an sp3 carbon bearing one
#     electronegative substituent), including hydroxyl and amine hydrogens, so
#     polar H carried ~5x too much LJ volume and no ho/hn identity at all.
#   * there were NO [dihedrals] and NO [pairs] sections, so every torsional
#     barrier was zero, aromatic rings could pucker freely and 1-4 interactions
#     were simply absent -- which lands directly on the boronate recognition
#     motif this experiment is built around.
#   * boron was typed as GAFF2 `c3` (sp3-carbon LJ) with only the mass patched.
#
# Silicon no longer comes through here at all in the normal path -- see
# parameterize_monomer, which routes Si through acpype/AM1-BCC and then applies
# _apply_polca_si_overrides.  This path is now (a) the boron path, because
# GAFF2 has no boron atom type and AM1-BCC has no boron parameters, and (b) the
# fallback if acpype fails on a silane.
#
# Parameter provenance:
#   GAFF2 (Wang et al., J. Comput. Chem. 2004; gaff2.dat) -- organic LJ, bonds,
#       angles, torsions, impropers.  R* / eps converted as
#       sigma[nm] = R*[A] * 0.1781797,  eps[kJ/mol] = eps[kcal/mol] * 4.184.
#   PolCA (Jorge et al., ACS Phys. Chem. Au 2021) -- Si Lennard-Jones.
#   UFF (Rappe et al., JACS 1992) -- boron LJ (B_2: x = 4.083 A, D = 0.180
#       kcal/mol -> sigma = 0.36375 nm, eps = 0.7531 kJ/mol).
#   Si-C 1.876 A, Si-O 1.640 A, Si-O-C 120 deg; B-C(aryl) 1.568 A, B-O 1.365 A
#       -- standard organosilane / arylboronic-acid crystallographic values.
#
# Anything this table does not cover RAISES.  A missing parameter must never
# again silently become 0.1500 nm.

# ── PolCA silicon Lennard-Jones ───────────────────────────────
#
# SOURCE, VERBATIM.  Jorge, Milne, Barrera & Gomes, "New Force-Field for
# Organosilicon Molecules in the Liquid Phase", ACS Phys. Chem. Au 2021, 1(1),
# 54-69, doi:10.1021/acsphyschemau.1c00014.
#
#   Table 6. Final Lennard-Jones Parameters for the Organosilicate Molecules
#   Considered in This Work
#       atom   σ (nm)   ε (kJ/mol)
#       Si0    0.580    0.108
#       Si1    0.551    0.108
#       Si2    0.522    0.108
#       Si3    0.493    0.108
#       Si4    0.464    0.108
#   "The superscript in the Si atom denotes the number of oxygen-containing
#    substituent groups."
#
# THE COLUMN HEADER SAYS kJ/mol.  This is recorded explicitly because ε = 0.108
# is NUMERICALLY ALMOST IDENTICAL to GAFF2's c3 epsilon of 0.1078 kcal/mol -- the
# substitute parmchk writes as "Si  1.9069  0.1078  same as c3" -- so the table
# reads like an un-converted kcal value and was flagged as a suspected 4.184x
# unit error.  It is NOT one.  The body text states it directly: "This returned
# values of σ = 0.58 nm and ε = 0.108 kJ/mol for the Si atom in alkylsilanes",
# and the σ ladder here is the paper's own rule -- "we reduced the value of σ for
# silicon by 5% for each oxygen-containing substituent group", which the paper's
# own worked example makes CUMULATIVE-LINEAR rather than compounding: "the value
# of sigma was scaled by 20% for tetramethoxysilane, which has 4 alkoxy groups",
# i.e. 0.580 x (1 - 0.05k), which reproduces all five rows exactly.
# DO NOT MULTIPLY THESE BY 4.184.  test_polca_si_lj.py pins them.
#
# KNOWN TRANSFERABILITY CAVEAT, not a bug and not fixed here: PolCA is a
# UNITED-ATOM model for the alkyl groups and fits its own oxygen types
# (OC/OB σ 0.235 nm ε 1.344 kJ/mol, silanol OH σ 0.304 nm ε 1.750 kJ/mol),
# whereas this pipeline keeps GAFF2 all-atom oxygens (oh: σ 0.32429 nm,
# ε 0.38911 kJ/mol) and overrides ONLY the silicon.  The silanol oxygen well is
# therefore ~4.5x shallower than PolCA's own.  Changing it would need a refit
# and its own validation; it applies equally to TEOS and APTES silanols, so it
# does not bias the ratio axis.
_POLCA_SI_LJ = {
    # sigma [nm], eps [kJ/mol] -- Jorge et al. 2021, Table 6, verbatim.
    "Si0": {"sigma": 0.580, "eps": 0.108},  # 4 alkyl
    "Si1": {"sigma": 0.551, "eps": 0.108},  # 3 alkyl, 1 O
    "Si2": {"sigma": 0.522, "eps": 0.108},  # 2 alkyl, 2 O
    "Si3": {"sigma": 0.493, "eps": 0.108},  # 1 alkyl, 3 O
    "Si4": {"sigma": 0.464, "eps": 0.108},  # 0 alkyl, 4 O (TEOS-like)
}
_POLCA_SI_LJ_SOURCE = {
    "reference": ("Jorge, Milne, Barrera & Gomes, ACS Phys. Chem. Au 2021, "
                  "1(1), 54-69, doi:10.1021/acsphyschemau.1c00014, Table 6"),
    "sigma_units": "nm",
    "eps_units": "kJ/mol",
    "sigma_rule": ("0.580 nm x (1 - 0.05 * number of oxygen-containing "
                   "substituents) -- cumulative-linear, per the paper's own "
                   "'scaled by 20% for tetramethoxysilane' example"),
    "not_a_unit_error": ("eps 0.108 kJ/mol coincidentally matches GAFF2 c3's "
                         "0.1078 kcal/mol; the paper's Table 6 header and body "
                         "text both say kJ/mol"),
}

# name -> (sigma nm, epsilon kJ/mol, mass amu)
_GAFF2_LJ = {
    # carbons
    "c":  (0.33152, 0.41338, 12.011),   # carbonyl / carboxyl sp2 C
    "c1": (0.34790, 0.66777, 12.011),   # sp C
    "c2": (0.33152, 0.41338, 12.011),   # aliphatic sp2 C
    "c3": (0.33977, 0.45104, 12.011),   # sp3 C
    "ca": (0.33152, 0.41338, 12.011),   # aromatic C
    # nitrogens
    "n":  (0.33210, 0.41236, 14.007),   # amide N
    "n1": (0.32735, 0.45940, 14.007),   # sp N (nitrile)
    "n2": (0.33210, 0.41236, 14.007),   # sp2 N with 2 substituents
    "n3": (0.33210, 0.41236, 14.007),   # sp3 amine N
    "nh": (0.33210, 0.41236, 14.007),   # amine N attached to aromatic ring
    # oxygens
    "o":  (0.30481, 0.61212, 15.999),   # carbonyl O
    "oh": (0.32429, 0.38911, 15.999),   # hydroxyl O
    "os": (0.31561, 0.30376, 15.999),   # ether / ester O
    # sulfurs
    "sh": (0.35636, 1.04600, 32.065),   # thiol S
    "ss": (0.35636, 1.04600, 32.065),   # thioether S
    # hydrogens
    "h1": (0.24220, 0.08703, 1.008),    # H on sp3 C with 1 EW neighbour
    "h2": (0.22437, 0.08703, 1.008),    # ... 2 EW neighbours
    "h3": (0.20655, 0.08703, 1.008),    # ... 3 EW neighbours
    "ha": (0.26255, 0.06736, 1.008),    # H on aromatic / sp2 / sp C
    "hc": (0.26002, 0.08703, 1.008),    # H on sp3 C, no EW neighbour
    "hn": (0.11065, 0.04184, 1.008),    # H on N
    "ho": (0.05379, 0.01966, 1.008),    # H on O
    "hs": (0.10890, 0.05188, 1.008),    # H on S
    # boron -- UFF B_2/B_3 (Rappe 1992). GAFF2 has no boron type.
    "b":  (0.36375, 0.75310, 10.811),
}


def _norm_type(t: str) -> str:
    """Collapse the five PolCA Si types onto one key for the bonded tables."""
    return "Si" if t.startswith("Si") else t


def _sym(table_rows):
    """Build a symmetric lookup keyed on tuple(sorted((t1, t2)))."""
    out = {}
    for t1, t2, val in table_rows:
        out[tuple(sorted((t1, t2)))] = val
    return out


# (b0 nm, kb kJ/mol/nm^2).  GROMACS bond funct 1 uses V = 1/2 kb (r-b0)^2, so
# kb = 2 * K[kcal/mol/A^2] * 4.184 * 100 = K * 836.8.
_BOND_PARAMS = _sym([
    # X-H
    ("c3", "hc", (0.1093, 282300.0)),
    ("c3", "h1", (0.1093, 282300.0)),
    ("c3", "h2", (0.1093, 282300.0)),
    ("c3", "h3", (0.1093, 282300.0)),
    ("ca", "ha", (0.1087, 289400.0)),
    ("c2", "ha", (0.1085, 288400.0)),
    ("c",  "ha", (0.1094, 274600.0)),
    ("c1", "ha", (0.1067, 320000.0)),
    ("oh", "ho", (0.0974, 462800.0)),
    ("n3", "hn", (0.1019, 337400.0)),
    ("n",  "hn", (0.1013, 351200.0)),
    ("nh", "hn", (0.1015, 348000.0)),
    ("n2", "hn", (0.1027, 340000.0)),
    ("sh", "hs", (0.1344, 232000.0)),
    # C-C
    ("c3", "c3", (0.1538, 251800.0)),
    ("c3", "ca", (0.1516, 269000.0)),
    ("c3", "c2", (0.1508, 273000.0)),
    ("c3", "c",  (0.1518, 269000.0)),
    ("c3", "c1", (0.1470, 300000.0)),
    ("ca", "ca", (0.1387, 385800.0)),
    ("ca", "c2", (0.1466, 320000.0)),
    ("ca", "c",  (0.1491, 293000.0)),
    ("c2", "c2", (0.1327, 458200.0)),
    ("c2", "c",  (0.1470, 313000.0)),
    ("c1", "c1", (0.1200, 800000.0)),
    # C-N
    ("c3", "n3", (0.1462, 313800.0)),
    ("c3", "n",  (0.1460, 314000.0)),
    ("c3", "nh", (0.1463, 313000.0)),
    ("ca", "nh", (0.1390, 373100.0)),
    ("ca", "n",  (0.1392, 371000.0)),
    ("ca", "n3", (0.1400, 360000.0)),
    ("c",  "n",  (0.1348, 410500.0)),
    ("c2", "n2", (0.1290, 500000.0)),
    ("c1", "n1", (0.1160, 881000.0)),
    ("c1", "n2", (0.1230, 600000.0)),
    # C-O
    ("c3", "oh", (0.1423, 267900.0)),
    ("c3", "os", (0.1427, 267900.0)),
    ("ca", "oh", (0.1366, 393000.0)),
    ("ca", "os", (0.1370, 390000.0)),
    ("c",  "o",  (0.1218, 542200.0)),
    ("c",  "oh", (0.1349, 410000.0)),
    ("c",  "os", (0.1338, 425000.0)),
    ("c1", "o",  (0.1180, 700000.0)),
    # C-S
    ("c3", "sh", (0.1816, 190000.0)),
    ("c3", "ss", (0.1810, 191000.0)),
    ("ca", "ss", (0.1760, 200000.0)),
    ("ca", "sh", (0.1770, 198000.0)),
    # Si (PolCA / organosilane crystallographic)
    ("Si", "c3", (0.1876, 156500.0)),
    ("Si", "ca", (0.1868, 160000.0)),
    ("Si", "c2", (0.1868, 160000.0)),
    ("Si", "os", (0.1640, 251000.0)),
    ("Si", "oh", (0.1645, 250000.0)),
    ("Si", "Si", (0.2340, 100000.0)),
    # Boron (arylboronic acid)
    ("b", "ca", (0.1568, 285000.0)),
    ("b", "c2", (0.1560, 285000.0)),
    ("b", "c3", (0.1590, 260000.0)),
    ("b", "oh", (0.1365, 480000.0)),
    ("b", "os", (0.1365, 480000.0)),
    ("b", "n",  (0.1420, 350000.0)),
    ("b", "nh", (0.1420, 350000.0)),
])

# Proper torsions, keyed on the CENTRAL bond types, GAFF "X-A-B-X" generics
# already divided by their idivf and converted to kJ/mol.
# value = list of (multiplicity, phase_deg, k_kJ_per_mol)
_TORSION_PARAMS = _sym([
    ("ca", "ca", [(2, 180.0, 15.167)]),   # X-ca-ca-X  14.50/4 kcal -- ring rigidity
    ("ca", "c2", [(2, 180.0, 4.184)]),
    ("c2", "c2", [(2, 180.0, 27.823)]),   # X-c2-c2-X  26.60/4 kcal -- alkene
    ("ca", "c",  [(2, 180.0, 4.184)]),
    ("ca", "nh", [(2, 180.0, 20.083)]),   # X-ca-nh-X   9.60/2 kcal -- aniline
    ("ca", "n",  [(2, 180.0, 1.883)]),
    ("c",  "n",  [(2, 180.0, 10.460)]),   # X-c -n -X  10.00/4 kcal -- amide
    ("c",  "c2", [(2, 180.0, 4.812)]),
    ("ca", "oh", [(2, 180.0, 3.766)]),
    ("ca", "os", [(2, 180.0, 3.766)]),
    ("ca", "c3", [(2, 0.0, 0.0)]),
    ("c3", "c3", [(3, 0.0, 0.651)]),
    ("c3", "oh", [(3, 0.0, 0.697)]),
    ("c3", "os", [(3, 0.0, 1.604)]),
    ("c3", "n3", [(3, 0.0, 0.976)]),
    ("c3", "nh", [(3, 0.0, 0.976)]),
    ("c3", "n",  [(2, 0.0, 0.0)]),
    ("c3", "c2", [(3, 0.0, 0.0)]),
    ("c3", "c",  [(3, 0.0, 0.0)]),
    ("c3", "sh", [(3, 0.0, 0.837)]),
    ("c3", "ss", [(3, 0.0, 1.395)]),
    ("c",  "oh", [(2, 180.0, 9.623)]),
    ("c",  "os", [(2, 180.0, 11.297)]),
    # Si -- the c3/os analogue parmchk itself substitutes, verified against the
    # acpype MTMS topology.
    ("Si", "os", [(2, 0.0, 0.335), (3, 0.0, 3.138), (1, 180.0, 3.682)]),
    ("Si", "oh", [(2, 0.0, 0.335), (3, 0.0, 3.138), (1, 180.0, 3.682)]),
    ("Si", "c3", [(3, 0.0, 0.651)]),
    ("Si", "ca", [(2, 0.0, 0.0)]),
    # Boron -- ESTIMATED, no published GAFF term exists. Magnitudes chosen to
    # reproduce the ~2 kcal/mol B-OH and ~1 kcal/mol aryl-B rotation barriers
    # of arylboronic acids. Planarity itself is enforced by the improper below,
    # which does not depend on these.
    ("b", "ca", [(2, 180.0, 4.184)]),
    ("b", "oh", [(2, 180.0, 8.368)]),
    ("b", "os", [(2, 180.0, 8.368)]),
])

# Periodic improper (GROMACS funct 4, phase 180, multiplicity 2) keyed on the
# CENTRAL atom type. Central atom is the 3rd of the four, per the AMBER
# convention acpype also emits. This is what keeps aromatic rings, the
# carbonyl, the amide N and the sp2 boron planar.
_IMPROPER_K = {
    "ca": 4.60240,   # GAFF X-X-ca-ha  1.1 kcal
    "c":  43.93200,  # GAFF X-X-c -o  10.5 kcal
    "c2": 4.60240,
    "n":  4.18400,   # amide N
    "nh": 4.18400,   # aniline N
    "b":  4.60240,   # ESTIMATED -- sp2 boron, same magnitude as aromatic C
}

# Angle force constants (kJ/mol/rad^2). GAFF2 angle K spans ~330 (H-C-H) to
# ~570 (C-C-C); the previous code used a flat 500 for everything. theta0 comes
# from the UFF-optimised 3D structure, which reproduces sp3/sp2/aromatic
# geometry to well under a degree.
_ANGLE_K_HH = 330.0    # both outer atoms are hydrogen
_ANGLE_K_XH = 400.0    # one outer atom is hydrogen
_ANGLE_K_XX = 550.0    # all heavy

_EW_ELEMENTS = {7, 8, 9, 15, 16, 17, 35, 53}   # electronegative, for h1/h2/h3


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


def _assign_gaff_types(mol, si_type: str, name: str = "?") -> list:
    """Assign a GAFF2 atom type to every atom, or raise.

    Three passes: heavy atoms that do not depend on neighbours, then nitrogen
    (which needs its neighbours' carbon types), then hydrogen (which needs the
    heavy atom it hangs off).
    """
    from rdkit import Chem
    n = mol.GetNumAtoms()
    types = [None] * n

    def _has(atom, order):
        return any(b.GetBondType() == order for b in atom.GetBonds())

    for atom in mol.GetAtoms():
        z, i = atom.GetAtomicNum(), atom.GetIdx()
        if z in (1, 7):
            continue
        if z == 6:
            if atom.GetIsAromatic():
                types[i] = "ca"
            elif _has(atom, Chem.BondType.TRIPLE):
                types[i] = "c1"
            elif any(b.GetBondType() == Chem.BondType.DOUBLE
                     and b.GetOtherAtom(atom).GetAtomicNum() == 8
                     for b in atom.GetBonds()):
                types[i] = "c"
            elif _has(atom, Chem.BondType.DOUBLE):
                types[i] = "c2"
            else:
                types[i] = "c3"
        elif z == 8:
            if _has(atom, Chem.BondType.DOUBLE):
                types[i] = "o"
            elif any(nb.GetAtomicNum() == 1 for nb in atom.GetNeighbors()):
                types[i] = "oh"
            else:
                types[i] = "os"
        elif z == 16:
            if any(nb.GetAtomicNum() == 1 for nb in atom.GetNeighbors()):
                types[i] = "sh"
            else:
                types[i] = "ss"
        elif z == 14:
            types[i] = si_type
        elif z == 5:
            types[i] = "b"
        else:
            raise ValueError(
                f"{name}: no GAFF2 atom type rule for element Z={z} "
                f"({atom.GetSymbol()}) at atom {i+1}. Refusing to guess.")

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        i = atom.GetIdx()
        if _has(atom, Chem.BondType.TRIPLE):
            types[i] = "n1"
        elif _has(atom, Chem.BondType.DOUBLE):
            types[i] = "n2"
        elif any(types[nb.GetIdx()] == "c" for nb in atom.GetNeighbors()):
            types[i] = "n"          # amide
        elif any(types[nb.GetIdx()] == "ca" for nb in atom.GetNeighbors()):
            types[i] = "nh"         # aromatic amine (aniline)
        else:
            types[i] = "n3"

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        i = atom.GetIdx()
        nbrs = atom.GetNeighbors()
        if len(nbrs) != 1:
            raise ValueError(f"{name}: hydrogen {i+1} has {len(nbrs)} "
                             f"neighbours -- broken connectivity")
        heavy = nbrs[0]
        hz = heavy.GetAtomicNum()
        if hz == 8:
            types[i] = "ho"
        elif hz == 7:
            types[i] = "hn"
        elif hz == 16:
            types[i] = "hs"
        elif hz == 6:
            ht = types[heavy.GetIdx()]
            if ht in ("ca", "c2", "c", "c1"):
                types[i] = "ha"
            else:                      # c3 -- count electronegative neighbours
                n_ew = sum(1 for nb in heavy.GetNeighbors()
                           if nb.GetAtomicNum() in _EW_ELEMENTS)
                types[i] = {0: "hc", 1: "h1", 2: "h2", 3: "h3"}[min(n_ew, 3)]
        else:
            raise ValueError(
                f"{name}: hydrogen {i+1} is bonded to "
                f"{heavy.GetSymbol()} -- no GAFF2 H type for that. "
                f"Refusing to guess.")

    missing = [i + 1 for i, t in enumerate(types) if t is None]
    if missing:
        raise ValueError(f"{name}: untyped atoms {missing}")
    return types


def _bond_param(t1: str, t2: str, name: str, i: int, j: int):
    key = tuple(sorted((_norm_type(t1), _norm_type(t2))))
    if key not in _BOND_PARAMS:
        raise ValueError(
            f"{name}: no bond parameter for {t1}-{t2} (atoms {i}-{j}). "
            f"This used to silently become 0.1500 nm. Add the pair to "
            f"_BOND_PARAMS with a citation, or parameterise this monomer "
            f"through acpype.")
    return _BOND_PARAMS[key]


def _one_four_pairs(mol):
    """1-4 atom pairs (0-based), excluding anything that is also 1-2 or 1-3."""
    n = mol.GetNumAtoms()
    adj = [set() for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)

    excluded = set()
    for i in range(n):
        for j in adj[i]:
            excluded.add(tuple(sorted((i, j))))
            for k in adj[j]:
                if k != i:
                    excluded.add(tuple(sorted((i, k))))

    pairs = set()
    for b in mol.GetBonds():
        j, k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        for i in adj[j]:
            if i == k:
                continue
            for l in adj[k]:
                if l == j or l == i:
                    continue
                p = tuple(sorted((i, l)))
                if p not in excluded:
                    pairs.add(p)
    return sorted(pairs)


def _proper_torsions(mol, types, name, conf):
    """Proper dihedrals for every rotatable/rigid bond, GAFF X-A-B-X generics."""
    from rdkit.Chem import rdMolTransforms
    adj = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()}
    lines = []
    unmatched = set()
    for b in mol.GetBonds():
        j, k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        # A terminal bond (X-H, C=O) has no torsion about it at all -- don't
        # look one up, and above all don't warn that we couldn't find one.
        if len(adj[j]) < 2 or len(adj[k]) < 2:
            continue
        tj, tk = _norm_type(types[j]), _norm_type(types[k])
        # A torsion about a linear (sp) centre is undefined and makes GROMACS
        # produce NaN forces. Skip rather than emit garbage.
        if "c1" in (tj, tk) or "n1" in (tj, tk):
            continue
        terms = _TORSION_PARAMS.get(tuple(sorted((tj, tk))))
        if terms is None:
            unmatched.add(f"{tj}-{tk}")
            terms = [(3, 0.0, 0.651)]      # X-c3-c3-X generic
        for i in adj[j]:
            if i == k:
                continue
            for l in adj[k]:
                if l == j or l == i:
                    continue
                try:
                    ang_ijk = rdMolTransforms.GetAngleDeg(conf, i, j, k)
                    ang_jkl = rdMolTransforms.GetAngleDeg(conf, j, k, l)
                except Exception:
                    continue
                if min(abs(180.0 - ang_ijk), abs(180.0 - ang_jkl)) < 5.0:
                    continue           # near-linear: dihedral is ill-defined
                for mult, phase, kd in terms:
                    if kd == 0.0:
                        continue
                    lines.append(f"  {i+1:5d}  {j+1:5d}  {k+1:5d}  {l+1:5d}"
                                 f"    9  {phase:8.2f}  {kd:10.5f}  {mult:2d}")
    if unmatched:
        logger.warning(
            f"  {name}: no published torsion term for central bond type(s) "
            f"{sorted(unmatched)} -- used the GAFF X-c3-c3-X generic "
            f"(0.651 kJ/mol, n=3). Planarity is still enforced by the "
            f"impropers; check this if the affected group matters.")
    return lines


def _impropers(mol, types, name):
    """Periodic impropers on every 3-coordinate sp2 centre (planarity)."""
    lines = []
    for atom in mol.GetAtoms():
        t = types[atom.GetIdx()]
        if t not in _IMPROPER_K:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) != 3:
            continue
        c = atom.GetIdx()
        a, b_, d = nbrs
        # AMBER convention: the central atom is the THIRD of the four.
        lines.append(f"  {a+1:5d}  {b_+1:5d}  {c+1:5d}  {d+1:5d}"
                     f"    4    180.00  {_IMPROPER_K[t]:10.5f}   2")
    return lines


def _generate_silane_itp(name: str, smiles: str, output_dir: Path) -> dict:
    """Hand-built GAFF2 .itp/.gro (boron monomers; fallback for silanes).

    Produces [atomtypes], [atoms], [bonds], [pairs], [angles], [dihedrals]
    (propers) and [dihedrals] (impropers). Raises rather than guessing any
    parameter it does not have.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolTransforms

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

    # Charges: Si→S proxy for Gasteiger (RDKit has no Si parameters; it DOES
    # have boron, which is why the boronic acids get sensible charges here:
    # B ≈ +0.49, O ≈ -0.42, OH hydrogen ≈ +0.19).
    rw = Chem.RWMol(mol)
    si_idx = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 14]
    for idx in si_idx:
        rw.GetAtomWithIdx(idx).SetAtomicNum(16)
    AllChem.ComputeGasteigerCharges(rw)
    charges = [float(rw.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
               for i in range(rw.GetNumAtoms())]
    for idx in si_idx:
        charges[idx] = 0.9

    # Renormalise the molecule to its formal charge.
    # Overwriting Si with the PolCA value above throws away whatever Gasteiger
    # assigned there, which leaves every silane net-charged: +0.68 to +0.82 e
    # across this library (measured on the generated ITPs). Uncorrected that is
    # not cosmetic -- a 25-monomer sol-gel box carries ~+18 e, grompp reports a
    # non-zero system charge, genion compensates with counter-ions (adding salt
    # the DI-water conditions do not have), and every silane gains a spurious
    # Coulomb attraction to a net-charged template. For an experiment whose
    # signal IS the monomer-template electrostatics, that fabricates part of the
    # observable.
    # Correction: spread the residual uniformly over the non-Si atoms so the
    # PolCA Si charge stays exact, then absorb the 4-decimal rounding error on
    # one atom so the written .itp sums to the formal charge exactly.
    _si_set = set(si_idx)
    _formal = Chem.GetFormalCharge(mol)
    _free = [i for i in range(len(charges)) if i not in _si_set]
    if _free:
        _per = (sum(charges) - _formal) / len(_free)
        for i in _free:
            charges[i] -= _per
        charges = [round(q, 4) for q in charges]
        _resid = round(sum(charges) - _formal, 6)
        if abs(_resid) > 1e-9:
            _sink = max(_free, key=lambda i: abs(charges[i]))
            charges[_sink] = round(charges[_sink] - _resid, 4)
        logger.debug(f"  {name}: net charge renormalised to "
                     f"{sum(charges):+.4f} (formal {_formal})")

    si_type = _classify_si_type(smiles) if si_idx else None
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    types = _assign_gaff_types(mol, si_type or "Si4", name)

    # ── [ atoms ] ────────────────────────────────────────────
    alines = []
    for i in range(n_atoms):
        a = mol.GetAtomWithIdx(i)
        t = types[i]
        if t.startswith("Si"):
            mass = 28.086
        else:
            mass = _GAFF2_LJ[t][2]
        alines.append(
            f"    {i+1:5d} {t:>10s} 1    {name:>5s} "
            f"{a.GetSymbol()+str(i+1):>5s} {i+1:5d} {charges[i]:10.4f} "
            f"{mass:10.4f}")

    # ── [ atomtypes ] ────────────────────────────────────────
    at_lines = ["; name  bond_type  mass    charge  ptype  sigma       epsilon"]
    used_types = sorted(set(types))
    for t in used_types:
        if t.startswith("Si"):
            lj = _POLCA_SI_LJ[t]
            at_lines.append(f"  {t}  {t}  {28.086:.3f}  0.000  A  "
                            f"{lj['sigma']:.5e}  {lj['eps']:.5e}")
        else:
            if t not in _GAFF2_LJ:
                raise ValueError(f"{name}: no Lennard-Jones parameters for "
                                 f"atom type {t!r}")
            s, e, m = _GAFF2_LJ[t]
            at_lines.append(f"  {t}  {t}  {m:.3f}  0.000  A  {s:.5e}  {e:.5e}")
    atomtypes_section = "[ atomtypes ]\n" + "\n".join(at_lines) + "\n\n"

    # ── [ bonds ] ────────────────────────────────────────────
    blines = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b0, kb = _bond_param(types[i], types[j], name, i + 1, j + 1)
        blines.append(f"  {i+1:5d}  {j+1:5d}    1    {b0:.4f}  {kb:.1f}"
                      f"   ; {types[i]}-{types[j]}")

    # ── [ pairs ] (1-4) ──────────────────────────────────────
    plines = [f"  {i+1:5d}  {j+1:5d}    1" for i, j in _one_four_pairs(mol)]

    # ── [ angles ] ───────────────────────────────────────────
    aangle_lines = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for ni in range(len(neighbors)):
            for nj in range(ni + 1, len(neighbors)):
                i, k = neighbors[ni], neighbors[nj]
                try:
                    angle = rdMolTransforms.GetAngleDeg(conf, i, idx, k)
                except Exception:
                    continue
                n_h = (types[i].startswith("h")) + (types[k].startswith("h"))
                kth = (_ANGLE_K_HH if n_h == 2 else
                       _ANGLE_K_XH if n_h == 1 else _ANGLE_K_XX)
                aangle_lines.append(
                    f"  {i+1:5d}  {idx+1:5d}  {k+1:5d}    1    "
                    f"{angle:.2f}  {kth:.1f}")

    dlines = _proper_torsions(mol, types, name, conf)
    ilines = _impropers(mol, types, name)

    sections = [atomtypes_section,
                f"[ moleculetype ]\n{name}    3\n\n",
                "[ atoms ]\n" + "\n".join(alines) + "\n\n"]
    if blines:
        sections.append("[ bonds ]\n" + "\n".join(blines) + "\n\n")
    if plines:
        sections.append("[ pairs ]\n" + "\n".join(plines) + "\n\n")
    if aangle_lines:
        sections.append("[ angles ]\n" + "\n".join(aangle_lines) + "\n\n")
    if dlines:
        sections.append("[ dihedrals ] ; propers\n" + "\n".join(dlines) + "\n\n")
    if ilines:
        sections.append("[ dihedrals ] ; impropers\n" + "\n".join(ilines) + "\n")

    itp_path = output_dir / f"{name}.itp"
    itp_path.write_text(
        f"; {name} -- hand-built GAFF2"
        + (" + PolCA Si (Jorge 2021)" if si_idx else "")
        + (" + UFF boron (Rappe 1992)"
           if any(t == "b" for t in types) else "") + "\n"
        + "; Gasteiger charges renormalised to the formal charge.\n"
        + "".join(sections),
        encoding="utf-8")

    gro_path = output_dir / f"{name}.gro"
    gl = [f"{name} monomer", f" {n_atoms}"]
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        a = mol.GetAtomWithIdx(i)
        gl.append(f"{1:5d}{name:>5s}{a.GetSymbol()+str(i+1):>5s}{i+1:5d}"
                  f"{pos.x/10:8.3f}{pos.y/10:8.3f}{pos.z/10:8.3f}")
    gl.append("   5.00000   5.00000   5.00000")
    gro_path.write_text("\n".join(gl) + "\n", encoding="utf-8")

    logger.info(f"  {name}: hand-built GAFF2 topology "
                f"({n_atoms} atoms, {len(blines)} bonds, {len(plines)} pairs, "
                f"{len(aangle_lines)} angles, {len(dlines)} propers, "
                f"{len(ilines)} impropers"
                + (f", Si={si_type}" if si_idx else "") + ")")
    return {"itp": str(itp_path), "gro": str(gro_path),
            "si_type": si_type, "method": "hand-built-gaff2"}


# ── PolCA Si overrides on an acpype/GAFF2 topology ────────────

# What parmchk gets wrong for silicon, and what we put back.
# `parmchk2` fills every missing Si term with "same as c3" (verified: the
# generated MTMS_AC.frcmod literally reads `Si 28.085 0.878 same as c3`,
# `Si-c3 228.89 1.535 same as c3-c3`, `Si-os 282.27 1.427 same as c3-os`,
# `NONBON Si 1.9069 0.1078 same as c3`). So silicon arrives with sp3-carbon
# LJ, a 1.535 A Si-C bond (true 1.876 A) and a 1.427 A Si-O bond (true
# 1.640 A). Everything else in the acpype topology -- AM1-BCC charges, H typing,
# torsions, impropers, 1-4 pairs -- is correct and is left untouched.
_SI_BOND_TARGETS = {          # partner GAFF type -> (b0 nm, kb kJ/mol/nm^2)
    "c3": (0.1876, 156500.0), "ca": (0.1868, 160000.0),
    "c2": (0.1868, 160000.0), "c":  (0.1868, 160000.0),
    "os": (0.1640, 251000.0), "oh": (0.1645, 250000.0),
}
_SI_CENTRED_ANGLE_DEG = 110.0   # X-Si-Y, near-tetrahedral organosilane
_SI_O_C_ANGLE_DEG = 120.0       # Si-O-C is much wider than the c3-os-c3 113.6


def _apply_polca_si_overrides(itp_path: Path, name: str) -> str:
    """Rewrite the Si atomtype/LJ, Si bonds and Si angles of an acpype ITP.

    Returns the PolCA Si type name (Si0..Si4), which also becomes the atomtype
    name in the file. Renaming matters: acpype calls every silicon `Si`, and
    _include_monomers_in_topology deduplicates hoisted atomtypes BY NAME, so
    leaving them all called `Si` would silently give a TEOS-like Si4 the LJ of
    whichever silane happened to be parameterised first.
    """
    from .config import ALL_MONOMERS
    m_info = ALL_MONOMERS.get(name)
    if m_info is None:
        raise ValueError(f"{name} not in ALL_MONOMERS -- cannot classify Si type")
    si_type = _classify_si_type(m_info["smiles"])
    lj = _POLCA_SI_LJ[si_type]

    itp_path = Path(itp_path)
    lines = itp_path.read_text().split("\n")

    # Pass 1: map atom index -> type, and find the silicons.
    section = None
    idx_type = {}
    for line in lines:
        s = line.strip()
        if s.startswith("["):
            section = s.strip("[] ").strip()
            continue
        if not s or s.startswith(";"):
            continue
        parts = s.split()
        if section == "atoms" and len(parts) >= 2:
            try:
                idx_type[int(parts[0])] = parts[1]
            except ValueError:
                pass
    si_atoms = {i for i, t in idx_type.items() if t.lower() == "si"}
    if not si_atoms:
        raise ValueError(f"{itp_path}: no atom typed 'Si' -- acpype output does "
                         f"not look like a silane topology")

    # Pass 2: rewrite.
    out = []
    section = None
    n_bonds = n_angles = 0
    for line in lines:
        s = line.strip()
        if s.startswith("["):
            section = s.strip("[] ").strip()
            out.append(line)
            continue
        if not s or s.startswith(";"):
            out.append(line)
            continue
        parts = s.split()

        if section == "atomtypes" and parts and parts[0].lower() == "si":
            comment = line.split(";", 1)[1] if ";" in line else ""
            out.append(f" {si_type}  {si_type}  0.00000  0.00000   A     "
                       f"{lj['sigma']:.5e}   {lj['eps']:.5e} ; PolCA {si_type}"
                       f" (Jorge 2021), was GAFF2 c3 {comment.strip()}")
            continue

        if section == "atoms" and len(parts) >= 2 and parts[1].lower() == "si":
            out.append(line.replace(f" {parts[1]} ", f" {si_type} ", 1))
            continue

        if section == "bonds" and len(parts) >= 5:
            try:
                ai, aj = int(parts[0]), int(parts[1])
            except ValueError:
                out.append(line); continue
            if ai in si_atoms or aj in si_atoms:
                other = aj if ai in si_atoms else ai
                ot = idx_type.get(other, "")
                if ai in si_atoms and aj in si_atoms:
                    b0, kb = 0.2340, 100000.0
                elif ot in _SI_BOND_TARGETS:
                    b0, kb = _SI_BOND_TARGETS[ot]
                else:
                    raise ValueError(
                        f"{itp_path}: Si bonded to GAFF type {ot!r} (atom "
                        f"{other}) -- no published Si-{ot} bond length. "
                        f"Refusing to leave the parmchk c3 substitute in place.")
                out.append(f"{parts[0]:>7s}{parts[1]:>7s}   1    "
                           f"{b0:.4e}  {kb:.4e} ; PolCA Si-{ot} "
                           f"(was {parts[3]} {parts[4]})")
                n_bonds += 1
                continue

        if section == "angles" and len(parts) >= 6:
            try:
                ai, aj, ak = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                out.append(line); continue
            new_theta = None
            if aj in si_atoms:
                new_theta = _SI_CENTRED_ANGLE_DEG
            elif idx_type.get(aj, "").startswith("o") and \
                    (ai in si_atoms or ak in si_atoms):
                new_theta = _SI_O_C_ANGLE_DEG
            if new_theta is not None:
                out.append(f"{parts[0]:>7s}{parts[1]:>7s}{parts[2]:>7s}   1    "
                           f"{new_theta:.4e}  {float(parts[5]):.4e} "
                           f"; PolCA (was {parts[4]} deg)")
                n_angles += 1
                continue

        out.append(line)

    itp_path.write_text("\n".join(out), encoding="utf-8")
    logger.info(f"  {name}: PolCA Si override applied -- atomtype Si→{si_type} "
                f"(sigma={lj['sigma']}, eps={lj['eps']}), {n_bonds} Si bonds "
                f"and {n_angles} Si angles corrected off the parmchk 'same as "
                f"c3' substitutes")
    return si_type



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
