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

AUDIT-FIX NOTES (read before changing an observable or a verdict here)
---------------------------------------------------------------------
* BLOCKER 06 — a trajectory index is NEVER converted to a time with an assumed
  dt. `_select_equilibrium_frames` records the real `ts.time`; `_create_cavity`
  verifies the frame trjconv actually dumped against the `t=` stamp in the .gro
  title and returns a coordinate fingerprint so `run_phase6` can assert the
  snapshots are distinct.
* BLOCKER 07 — own and cross rebinding legs go through ONE placement procedure
  (`_build_rebinding_system`): strip template, pdb2gmx the incoming template,
  principal-axis placement into the vacated volume, delete clashing
  solvent/ions, genion charge rebalance, energy minimise, VERIFY convergence.
* OBSERVABLE — the binding verdict is the PERSISTENT-CONTACT FRACTION
  (code/pipeline/utils_persistent_contacts.py), not RMSD. `gmx rms` with its default
  least-squares fit superimposes the template onto itself and cannot see it
  leave the cavity; over 68 completed legs it correlated with actual contact at
  rho = 0.07. `rmsd_mean_A` now carries a FIT-FREE, monomer-centred number and
  `rmsd_selffit_A` preserves the retired statistic. Neither gates anything.
* REBINDING_RMSD_THRESHOLD is still imported in a few places to keep the config
  import contract stable, but NOTHING in this module reads its value any more.
  Do not reintroduce an RMSD threshold as a verdict.
"""

import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import numpy as np
from pathlib import Path

# Shared with Phase 4: a pH model renames titratable residues (LYSN, HISE, ...)
# and MDAnalysis's `protein` selection does not know all of them, so an
# exclusion-based monomer selection silently picks up protein residues. Phase 4
# owns the canonical sets; importing them keeps the two phases from drifting.
from .phase4_md_validation import _PH_PROTEIN_RESNAMES, _SOLVENT_IONS

_NON_MONOMER_RESNAMES = " ".join(sorted(_SOLVENT_IONS | _PH_PROTEIN_RESNAMES))

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Config access
# ══════════════════════════════════════════════════════════════════════════
# Symbols added by this audit-fix pass are read through _cfg() rather than
# `from .config import X`, so the module keeps importing (and
# tests/test_config_regression.py keeps passing) whether or not the
# integration agent has landed them in config_CD.py yet. Once they exist the
# config value wins; until then the default below is used and announced.
_CFG_DEFAULTS_ANNOUNCED = set()


def _cfg(name, default):
    """Read a config symbol, falling back to `default` if it is not defined."""
    from . import config as _c
    if hasattr(_c, name):
        return getattr(_c, name)
    if name not in _CFG_DEFAULTS_ANNOUNCED:
        _CFG_DEFAULTS_ANNOUNCED.add(name)
        logger.warning(f"config symbol {name} is not defined — Phase 5 is using "
                       f"its built-in default {default!r}")
    return default


def _import_contact_module():
    """Return the persistent-contacts module.

    Historical: this used to reach up to code/pipeline/utils_persistent_contacts.py via a
    sys.path shim. The module has since been consolidated into
    code/pipeline/utils_persistent_contacts.py, so a normal package import
    suffices. Function kept as an indirection point in case future code needs
    to swap the backend (e.g. a numba/GPU implementation).
    """
    from . import utils_persistent_contacts as _pcf
    return _pcf


def _import_pcsi_star():
    """Return the PCSI* module (was code/pcsi_star.py, now consolidated)."""
    from . import utils_pcsi_star as _ps
    return _ps


# ══════════════════════════════════════════════════════════════════════════
# BLOCKER (Phase 5 observable): contact-based binding readout
# ══════════════════════════════════════════════════════════════════════════
# The pipeline's binding readout used to be `gmx rms -s md.tpr` with the
# DEFAULT least-squares fit, i.e. the template was superimposed onto itself
# before its displacement was measured. That removes rigid-body escape — the
# only thing the observable exists to detect — and over 68 completed legs it
# correlated with actual template/monomer contact at rho = 0.07.
#
# The primary observable is now the persistent-contact fraction computed by
# code/pipeline/utils_persistent_contacts.py (the same statistic PCSI* is built on):
#     f = (# template residues in contact with a monomer in > 50% of the
#          analysed frames) / (# template residues)
# A fit-free RMSD (monomer-centred, `gmx rms -fit none`) is kept alongside it
# as a DIAGNOSTIC only. Nothing downstream may key a verdict off the
# self-fitted number again.

def _contact_metrics(md_dir: Path, tag: str = "") -> dict:
    """Persistent-contact observable for one rebinding leg.

    Returns {"available": bool, ...}. Never raises.
    """
    md_dir = Path(md_dir).resolve()
    xtc = md_dir / "md.xtc"
    tpr = md_dir / "md.tpr"
    if not (xtc.exists() and tpr.exists()):
        return {"available": False, "reason": "md.xtc / md.tpr missing"}

    cutoff = float(_cfg("REBINDING_CONTACT_CUTOFF_A", 6.0))
    persistence = float(_cfg("REBINDING_CONTACT_PERSISTENCE", 0.5))
    last_frac = float(_cfg("REBINDING_CONTACT_LAST_FRAC", 0.5))
    try:
        pcf = _import_contact_module()
        freq, n_pers, meta = pcf.compute_persistent_contacts_fast(
            xtc, tpr, cutoff_A=cutoff, persistence_frac=persistence,
            last_frac=last_frac, return_meta=True)
    except Exception as e:
        logger.warning(f"    contact observable failed{tag}: {e}")
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}

    total = len(freq)
    if total == 0:
        return {"available": False, "reason": "empty template selection"}

    return {
        "available": True,
        "n_persistent_residues": int(n_pers),
        "total_residues": int(total),
        "fraction_persistent": round(n_pers / total, 4),
        "mean_contact_freq": round(float(np.mean(list(freq.values()))), 4),
        "cutoff_A": cutoff,
        "persistence_frac": persistence,
        "last_frac": last_frac,
        # window provenance — PCSI* gate (f) needs these to be equal across legs
        "n_analyzed": meta.get("n_analyzed"),
        "dt_ps": meta.get("dt_ps"),
        "n_frames": meta.get("n_frames"),
    }


def _contact_quartiles(md_dir: Path) -> dict:
    """Persistent contacts over Q1 vs Q4 of one leg — for the removal test.

    Uses pcsi_star.analyze_leg(keep_frames=True) so the per-frame contact
    matrix is built ONCE and both windows come out of the same pass with
    exactly the legacy contact semantics.
    """
    md_dir = Path(md_dir).resolve()
    xtc = md_dir / "md.xtc"
    tpr = md_dir / "md.tpr"
    if not (xtc.exists() and tpr.exists()):
        return {"available": False, "reason": "md.xtc / md.tpr missing"}

    cutoff = float(_cfg("REBINDING_CONTACT_CUTOFF_A", 6.0))
    persistence = float(_cfg("REBINDING_CONTACT_PERSISTENCE", 0.5))
    try:
        ps = _import_pcsi_star()
        res = ps.analyze_leg(xtc, tpr, cutoffs=(cutoff,), last_frac=1.0,
                             keep_frames=True, check_window=False)
        frames = res["_frames"][cutoff]          # (n_frames, n_res) bool
    except Exception as e:
        logger.warning(f"    contact quartile analysis failed: {e}")
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}

    n = frames.shape[0]
    n_res = frames.shape[1]
    if n < 4 or n_res == 0:
        return {"available": False, "reason": f"too few frames ({n})"}

    q1 = frames[:n // 4]
    q4 = frames[3 * n // 4:]
    k_q1 = int(np.sum(q1.mean(axis=0) > persistence))
    k_q4 = int(np.sum(q4.mean(axis=0) > persistence))
    return {
        "available": True,
        "n_frames": int(n),
        "total_residues": int(n_res),
        "k_persistent_q1": k_q1,
        "k_persistent_q4": k_q4,
        "f_persistent_q1": round(k_q1 / n_res, 4),
        "f_persistent_q4": round(k_q4 / n_res, 4),
        "retention": (round(k_q4 / k_q1, 3) if k_q1 > 0 else None),
        "cutoff_A": cutoff,
        "persistence_frac": persistence,
    }


def _make_group_ndx(tpr_path: Path, work_dir: Path,
                    ndx_name: str = "p5_groups.ndx") -> Path:
    """Index file with group 0 = MONOMERS, 1 = TEMPLATE, 2 = SYSTEM.

    Written directly from MDAnalysis rather than via `gmx make_ndx` /
    `gmx select`: make_ndx group NUMBERING depends on which default groups
    GROMACS happens to generate for a given system, and feeding a wrong number
    to trjconv silently centres on the wrong thing. Here the ordering is fixed
    by construction, so `-n <this file>` + "0\\n2\\n" always means
    "centre on the monomers, write the whole system".
    """
    import MDAnalysis as mda
    # resolve(): every gmx call below runs with cwd=work_dir, so a relative
    # output path would land somewhere else entirely (and silently produce
    # "analysis unavailable" instead of an error).
    work_dir = Path(work_dir).resolve()
    tpr_path = Path(tpr_path).resolve()
    ndx = work_dir / ndx_name
    if ndx.exists():
        return ndx
    try:
        u = mda.Universe(str(tpr_path))
    except Exception as e:
        logger.warning(f"    could not read {tpr_path} for index generation: {e}")
        return None

    groups = [
        ("MONOMERS", u.select_atoms(_MONOMER_SEL)),
        ("TEMPLATE", u.select_atoms("protein")),
        ("SYSTEM", u.atoms),
    ]
    for name, ag in groups:
        if len(ag) == 0:
            logger.warning(f"    index group {name} is EMPTY — cannot build a "
                           f"cavity-frame trajectory for {tpr_path}")
            return None

    out = []
    for name, ag in groups:
        out.append(f"[ {name} ]")
        idx = ag.indices + 1                      # GROMACS .ndx is 1-based
        for i in range(0, len(idx), 15):
            out.append(" ".join(f"{j:d}" for j in idx[i:i + 15]))
    ndx.write_text("\n".join(out) + "\n")
    return ndx


def _gmx_rmsd_nofit(tpr_path: Path, xtc_path: Path, work_dir: Path,
                    xvg_name: str = "rmsd_nofit.xvg") -> tuple:
    """FIT-FREE template RMSD in the cavity frame.

    The monomers are position-restrained, so the cavity is fixed in the box
    frame: centring the trajectory on the MONOMER group and running
    `gmx rms -fit none` measures how far the template moved RELATIVE TO THE
    CAVITY. This is the honest version of the old `_gmx_rmsd`, which fitted the
    template onto itself and therefore could not see it leave.

    Returns (q4_mean_A, final_A) or (None, None). DIAGNOSTIC ONLY — the binding
    verdict comes from _contact_metrics().
    """
    from .config import GMX_BIN
    work_dir = Path(work_dir).resolve()      # gmx runs with cwd=work_dir
    tpr_path, xtc_path = Path(tpr_path).resolve(), Path(xtc_path).resolve()
    if not (tpr_path.exists() and xtc_path.exists()):
        return None, None

    ndx = _make_group_ndx(tpr_path, work_dir)
    if ndx is None:
        return None, None

    centered = work_dir / "md_cavframe.xtc"
    if not centered.exists():
        try:
            subprocess.run(
                [GMX_BIN, "trjconv", "-f", str(xtc_path), "-s", str(tpr_path),
                 "-n", str(ndx), "-o", str(centered), "-pbc", "mol", "-center"],
                input="0\n2\n", capture_output=True, text=True,
                cwd=str(work_dir), timeout=1800)
        except Exception as e:
            logger.warning(f"    cavity-frame centering failed: {e}")
            return None, None
    if not centered.exists():
        logger.warning("    cavity-frame centering produced no trajectory")
        return None, None

    xvg = work_dir / xvg_name
    try:
        subprocess.run(
            [GMX_BIN, "rms", "-s", str(tpr_path), "-f", str(centered),
             "-n", str(ndx), "-fit", "none", "-o", str(xvg), "-tu", "ns"],
            input="1\n1\n", capture_output=True, text=True,
            cwd=str(work_dir), timeout=600)
    except Exception as e:
        logger.warning(f"    gmx rms -fit none failed: {e}")
        return None, None

    rmsds = _parse_xvg_rmsd(xvg)
    if not rmsds:
        return None, None
    n = len(rmsds)
    q4 = rmsds[3 * n // 4:]
    return round(float(np.mean(q4)), 2), round(float(rmsds[-1]), 2)


def _parse_xvg_rmsd(xvg: Path) -> list:
    """Parse column 2 of an xvg (nm) into a list of Å."""
    xvg = Path(xvg)
    if not xvg.exists():
        return []
    out = []
    for line in xvg.read_text(errors="replace").split("\n"):
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                out.append(float(parts[1]) * 10.0)
            except ValueError:
                pass
    return out


def _gmx_rmsd(tpr_path: Path, xtc_path: Path, work_dir: Path,
              xvg_name: str = "rmsd_protein.xvg") -> tuple:
    """SELF-FITTED protein RMSD (`gmx rms` default fit). DIAGNOSTIC ONLY.

    RETAINED FOR PROVENANCE, NOT FOR VERDICTS. This superimposes the template
    onto its own reference before measuring displacement, so a template that
    walks out of the cavity intact reads ~0 Å. Use _gmx_rmsd_nofit() for a
    displacement number and _contact_metrics() for the binding verdict.

    Returns (rmsd_mean_second_half_A, rmsd_final_A) or (None, None).
    """
    from .config import GMX_BIN
    work_dir = Path(work_dir).resolve()       # gmx runs with cwd=work_dir
    tpr_path, xtc_path = Path(tpr_path).resolve(), Path(xtc_path).resolve()
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
                 acceptors_sel=f"not protein and not resname {_NON_MONOMER_RESNAMES}",
                 d_a_cutoff=3.5, d_h_a_angle_cutoff=150,
                 update_selections=False)
        hb.run(start=start, step=stride, verbose=False)

        # Also check reverse direction (monomer donors → protein acceptors)
        hb2 = HBA(u,
                  donors_sel=f"not protein and not resname {_NON_MONOMER_RESNAMES}",
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
        monomers = u.select_atoms(f"not protein and not resname {_NON_MONOMER_RESNAMES}")
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


# ── Phase 4 replica layout (integration pass) ─────────────────
#
# Phase 4 changed from ONE trajectory per (target, pc_id) to PHASE4_N_REPLICAS
# independent ones, writing <target>/<pc_id>/rep_<i>/md/ instead of
# <target>/<pc_id>/md/. Phase 5 hardcoded the old path in two places and would
# simply not have found a trajectory. These two helpers are the single point
# where that layout is interpreted.

def _resolve_p4_md_dirs(target: str, pc_id: str, pc_data: dict) -> list:
    """Return [(replica_index, md_dir)] of USABLE Phase 4 trajectories.

    Preference order:
      1. pc_data["replica_md_dirs"], restricted to accepted_replica_indices
         when Phase 4 published them (a rejected replica did not converge, so
         its frames are not equilibrium frames);
      2. pc_data["md_dir"], the representative replica;
      3. the legacy <target>/<pc_id>/md path, with a loud warning.
    A directory counts only if it actually holds a trajectory and an npt.gro.
    """
    def _usable(d):
        d = Path(d)
        has_traj = (d / "md_reduced.xtc").exists() or (d / "md.xtc").exists()
        return has_traj and (d / "npt.gro").exists()

    def _index_of(d, position):
        """Replica index from the rep_<i> directory name, not list position.

        Phase 4 builds replica_md_dirs in order, so position usually equals the
        index — but only usually. The directory name is the authoritative label
        (phase4._replica_dir writes rep_<i>), and getting this wrong would
        mis-attribute snapshots to the wrong source trajectory, which is exactly
        the grouping the autocorrelation correction depends on.
        """
        m = re.search(r"rep_(\d+)", str(d))
        return int(m.group(1)) if m else position

    out = []
    rep_dirs = pc_data.get("replica_md_dirs") or []
    accepted = pc_data.get("accepted_replica_indices")
    if rep_dirs:
        pairs = [(_index_of(d, i), Path(d)) for i, d in enumerate(rep_dirs)]
        if accepted:
            keep = [p for p in pairs if p[0] in set(accepted)]
            if keep and len(keep) < len(pairs):
                logger.info(f"[{target}] using {len(keep)}/{len(pairs)} Phase 4 "
                            f"replicas (accepted: {sorted(set(accepted))})")
            if not keep:
                logger.error(f"[{target}] NO Phase 4 replica passed acceptance; "
                             f"falling back to all {len(pairs)} so the failure "
                             f"is visible downstream rather than silent.")
            pairs = keep or pairs
        out = [(i, d) for i, d in pairs if _usable(d)]
        if out:
            return out

    md_dir = pc_data.get("md_dir")
    if md_dir and _usable(md_dir):
        logger.warning(f"[{target}] replica_md_dirs unusable; falling back to the "
                       f"representative md_dir {md_dir}")
        return [(0, Path(md_dir))]

    from .config import get_output_path          # not a module-level import here
    legacy = get_output_path("phase4") / target / pc_id / "md"
    if _usable(legacy):
        logger.warning(f"[{target}] using LEGACY single-trajectory Phase 4 layout "
                       f"{legacy} — this run predates PHASE4_N_REPLICAS, so all "
                       f"snapshots come from one correlated trajectory.")
        return [(0, legacy)]
    return []


def _plan_snapshot_jobs(p4_md_dirs: list, n_snapshots: int, target: str) -> list:
    """Spread n_snapshots over the available replicas, as evenly as possible.

    Returns [{replica, index, md_dir, traj, top, frame}] where `index` is the
    snapshot's position WITHIN its replica, so the directory name
    rep<r>_snapshot_<index> is unique and the source replica stays recoverable.
    """
    n_rep = len(p4_md_dirs)
    if n_rep == 0:
        return []
    # e.g. 10 snapshots over 3 replicas -> 4, 3, 3
    per = [n_snapshots // n_rep + (1 if i < n_snapshots % n_rep else 0)
           for i in range(n_rep)]

    jobs = []
    for (rep_idx, md_dir), want in zip(p4_md_dirs, per):
        if want <= 0:
            continue
        traj = md_dir / "md_reduced.xtc"
        if not traj.exists():
            traj = md_dir / "md.xtc"
        top = md_dir / "npt.gro"
        frames = _select_equilibrium_frames(traj, top, md_dir / "topol.top",
                                            n_frames=want)
        if not frames:
            logger.error(f"[{target}] replica {rep_idx}: no suitable equilibrium "
                         f"frames in {md_dir} — contributing 0 snapshots")
            continue
        for k, fr in enumerate(frames):
            jobs.append({"replica": rep_idx, "index": k, "md_dir": md_dir,
                         "traj": traj, "top": top, "frame": fr})

    if len(jobs) < n_snapshots:
        logger.error(f"[{target}] planned {len(jobs)} snapshots but "
                     f"{n_snapshots} were requested — the shortfall is recorded "
                     f"rather than back-filled from one trajectory, because "
                     f"duplicating a replica would inflate the apparent sample.")
    return jobs


_SNAPSHOT_DIR_RE = re.compile(r"^(?:rep(?P<rep>\d+)_)?snapshot_(?P<idx>\d+)$")


def _iter_snapshot_dirs(target_dir: Path) -> list:
    """Yield (replica, index, dir) for every snapshot directory under a target.

    Understands BOTH layouts:
      snapshot_<i>            legacy, one Phase 4 trajectory  -> replica 0
      rep<r>_snapshot_<i>     current, snapshots spread over Phase 4 replicas

    The three re-analysis entry points below globbed "snapshot_*", which stops
    matching the moment snapshots carry their source replica in the name. Sorted
    by (replica, index) so ordering is deterministic.
    """
    out = []
    target_dir = Path(target_dir)
    if not target_dir.is_dir():
        return out
    for d in target_dir.iterdir():
        if not d.is_dir():
            continue
        m = _SNAPSHOT_DIR_RE.match(d.name)
        if m:
            out.append((int(m.group("rep") or 0), int(m.group("idx")), d))
    return sorted(out, key=lambda t: (t[0], t[1]))


def _has_snapshot_dirs(root: Path) -> bool:
    """True if any <target>/<snapshot dir> exists under root, either layout."""
    root = Path(root)
    if not root.is_dir():
        return False
    return any(_iter_snapshot_dirs(d) for d in root.iterdir() if d.is_dir())


def run_phase6(phase4_results: dict = None,
               phase1_results: dict = None,
               target_names: list = None,
               output_dir: str = None,
               fresh: bool = False) -> dict:
    """
    Phase 5 in the pipeline order: VIP cavity rebinding validation.

    NOTE ON NAMING: this function is HISTORICALLY named `run_phase6` because an
    early draft numbered the phases differently. The pipeline in run_pipeline.py
    dispatches this as Phase 5 (VIP rebinding). `run_phase5` below is the
    semantically-correct alias — prefer it in new code; keep `run_phase6` for
    back-compat with callers that already reference it.

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
    if fresh and partial_json.exists():
        # --fresh: honoured by the phase itself, not only by run_pipeline having
        # archived the directory first.
        logger.warning("--fresh: IGNORING existing phase5_rebinding_results.json; "
                       "every requested target will be recomputed")
    elif partial_json.exists():
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

        # ── PHASE 4 QUALITY GATE ───────────────────────────────────
        # Phase 4 now records per-replica acceptance (convergence, RMSD
        # equilibration, completed MD). A leg whose MD never equilibrated is
        # not a cavity worth rebinding against, and rebinding it silently
        # launders a rejected trajectory into a selectivity number.
        if pc_data.get("accepted") is False:
            logger.error(
                f"[{target}] Phase 4 leg {best_pc_id} FAILED acceptance "
                f"({pc_data.get('n_replicas_accepted', 0)}/"
                f"{pc_data.get('n_replicas', '?')} replicas accepted; "
                f"failures: {pc_data.get('quality_failures')}). "
                f"Every Phase 5 result for this target is marked PROVISIONAL.")

        # ── PHASE 4 TRAJECTORY DIRECTORIES (replica-aware) ─────────
        # Phase 4 writes <target>/<pc_id>/rep_<i>/md/, not <target>/<pc_id>/md/.
        p4_md_dirs = _resolve_p4_md_dirs(target, best_pc_id, pc_data)
        if not p4_md_dirs:
            logger.warning(f"[{target}] Phase 4 trajectory not found "
                           f"(looked for replica_md_dirs / md_dir / legacy path)")
            continue

        logger.info(f"\n{'='*20} Phase 6 Rebinding: {target}/{best_pc_id} {'='*20}")

        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Select equilibrium frames, DISTRIBUTED ACROSS REPLICAS.
        #
        # This is the whole point of Phase 4 running replicas. Within ONE
        # trajectory the measured lag-1 autocorrelation of the snapshot
        # observable is rho1 = 0.23, i.e. 10 snapshots carry n_eff = 6.3
        # independent samples. Snapshots drawn from DIFFERENT replicas are
        # independent by construction, so spreading the same budget across
        # replicas buys real sample size for no extra Phase 5 cost.
        logger.info(f"  Step 1: Selecting {REBINDING_N_SNAPSHOTS} equilibrium "
                    f"frames across {len(p4_md_dirs)} replica(s)...")
        jobs = _plan_snapshot_jobs(p4_md_dirs, REBINDING_N_SNAPSHOTS, target)

        if not jobs:
            logger.warning("  No suitable frames found")
            continue

        logger.info(f"  Snapshot plan: "
                    + ", ".join(f"rep{j['replica']}@{j['frame']['time_ps']}ps"
                                for j in jobs))

        # BLOCKER 06: selected times must be distinct BEFORE any extraction.
        # Distinctness is required WITHIN a replica; the same time in two
        # different replicas is a different structure and is legitimate.
        _dup = None
        for _rep in sorted({j["replica"] for j in jobs}):
            _times = [j["frame"].get("time_ps") for j in jobs
                      if j["replica"] == _rep]
            if len(set(_times)) != len(_times):
                _dup = (_rep, _times)
                break
        if _dup is not None:
            logger.error(f"[{target}] replica {_dup[0]} snapshot times are not "
                         f"distinct: {_dup[1]} — refusing to build a replica set "
                         f"out of repeated frames.")
            results[target] = {"error": "non-distinct snapshot times",
                               "replica": _dup[0], "selected_times_ps": _dup[1]}
            continue

        # Steps 2-4: For each snapshot
        snapshot_results = []
        _fingerprints = {}     # fingerprint -> snapshot index (distinctness proof)
        for si, job in enumerate(jobs):
            frame_info = job["frame"]
            p4_md_dir = job["md_dir"]
            traj, top = job["traj"], job["top"]
            # Directory name carries the SOURCE REPLICA so the downstream
            # statistics (PCSI*) can group snapshots by trajectory and sum
            # n_eff per replica instead of pooling correlated samples.
            snap_dir = target_dir / f"rep{job['replica']}_snapshot_{job['index']}"
            snap_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\n  --- Snapshot {si+1}/{len(jobs)} "
                        f"(replica {job['replica']}, frame {frame_info['frame_idx']}, "
                        f"contacts={frame_info['total_contacts']}) ---")

            # Step 2: Extract frame + freeze monomers
            logger.info(f"  Step 2: Extracting frame and freezing monomers...")
            cavity_result = _create_cavity(
                traj, top, p4_md_dir / "topol.top",
                frame_info['frame_idx'],
                snap_dir,
                frame_time_ps=frame_info.get("time_ps"))

            if not cavity_result.get("success"):
                logger.warning(f"  Cavity creation failed: {cavity_result.get('error')}")
                snapshot_results.append({"success": False,
                                          "error": cavity_result.get("error")})
                continue

            # BLOCKER 06: assert the extracted structures really are distinct.
            fp = cavity_result.get("fingerprint")
            if fp is None:
                logger.error("  Could not fingerprint the extracted frame — "
                             "cannot prove snapshots are distinct; skipping.")
                snapshot_results.append({"success": False,
                                          "error": "fingerprint unavailable"})
                continue
            if fp in _fingerprints:
                logger.error(
                    f"  Snapshot {si} (frame {frame_info['frame_idx']}, "
                    f"t={cavity_result['frame_time_ps']:g} ps) is byte-identical to "
                    f"snapshot {_fingerprints[fp]} — the trajectory index -> time "
                    f"mapping is still wrong. Refusing to run a duplicate leg.")
                snapshot_results.append({
                    "success": False,
                    "error": f"duplicate of snapshot {_fingerprints[fp]}",
                    "frame_idx": frame_info["frame_idx"],
                    "frame_time_ps": cavity_result["frame_time_ps"]})
                continue
            _fingerprints[fp] = si

            # ── EV-templated protocol: Triton X-100 lysis ──
            # When PHASE4_MEMBRANE_MODE + PHASE5_TRITON_REMOVAL_MODE are both
            # on, strip lipids + template CD from the snapshot frame BEFORE
            # rebinding starts. The resulting `cavity.gro` becomes the
            # rebinding cavity — replaces the raw Phase 4 frame which still
            # carries the template that the rebinding legs would clash with.
            # Also skips the classical template-removal test (Step 3): with a
            # membrane+EV template the "does the head escape?" question is
            # replaced by "does a fresh EV dock?" downstream.
            from .config import (PHASE4_MEMBRANE_MODE as _MEMB,
                                  PHASE5_TRITON_REMOVAL_MODE as _TRIT)
            snap_result_removal = None
            ev_mode = bool(_MEMB and _TRIT)
            if ev_mode:
                try:
                    from .utils_triton_removal import finalize_triton_removal
                    logger.info(f"  Step 3 (EV-mode): Triton lysis — "
                                f"stripping lipids + template CD from frame")
                    lysed = finalize_triton_removal(
                        cavity_out_dir=snap_dir / "triton",
                        relax_em=False,
                        input_gro=cavity_result["cavity_gro"],
                        input_top=cavity_result["cavity_top"])
                    cavity_result["cavity_gro"] = lysed["cavity_gro"]
                    cavity_result["cavity_top"] = lysed["cavity_top"]
                    snap_result_removal = {
                        "protocol": "triton_lysis",
                        "n_atoms_removed": lysed["n_atoms_removed"],
                        "n_atoms_kept":    lysed["n_atoms_kept"],
                        "cavity_gro":      lysed["cavity_gro"],
                    }
                    logger.info(
                        f"    → removed {lysed['n_atoms_removed']} atoms, "
                        f"kept {lysed['n_atoms_kept']} — cavity ready")
                except Exception as e:
                    logger.error(f"  Triton lysis failed: {e}")
                    snap_result_removal = {"protocol": "triton_lysis",
                                            "error": str(e)}
                    snapshot_results.append({"success": False,
                                              "error": f"triton_lysis: {e}"})
                    continue
            else:
                # Step 3: Template removal test — can template escape?
                logger.info(f"  Step 3: Template removal test...")
                removal_result = _run_template_removal_md(
                    cavity_result["cavity_gro"],
                    cavity_result["cavity_top"],
                    snap_dir / "removal_test",
                    time_ns=min(REBINDING_MD_NS, 10),  # shorter test
                    p4_md_dir=p4_md_dir)

                snap_result_removal = removal_result

            # Step 4: Rebind own template.
            # `target=` is passed explicitly so the EV-approach dispatcher in
            # _run_rebinding_md can resolve the fresh-EV CHARMM-GUI outputs
            # deterministically when PHASE5_TRITON_REMOVAL_MODE is on.
            logger.info(f"  Step 4: Rebinding {target} head...")
            rebind_own = _run_rebinding_md(
                cavity_result["cavity_gro"],
                cavity_result["cavity_top"],
                head_pdb,
                snap_dir / "rebind_own",
                time_ns=REBINDING_MD_NS,
                p4_md_dir=p4_md_dir,
                is_own_target=True,
                target=target,
                placement_seed=si)   # 1 seed per snapshot → N=n_snapshots placements

            snap_result = {
                "snapshot": si,
                # SOURCE REPLICA. Snapshots from one trajectory are correlated
                # (measured rho1 = 0.23, n_eff 6.3 of 10); snapshots from
                # different replicas are independent. Downstream statistics
                # must group on this rather than pooling all snapshots.
                "replica": job["replica"],
                "replica_snapshot_index": job["index"],
                "p4_md_dir": str(p4_md_dir),
                "phase4_accepted": pc_data.get("accepted"),
                "frame_idx": frame_info["frame_idx"],
                "frame_time_ps": cavity_result.get("frame_time_ps"),
                "traj_dt_ps": cavity_result.get("traj_dt_ps"),
                "frame_time_source": cavity_result.get("time_source"),
                "frame_fingerprint": fp,
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
                    is_own_target=False,
                    target=other_target,
                    placement_seed=si)

                snap_result[f"rebind_{other_target}"] = rebind_other

            # DERIVED, NOT ASSERTED.  This was an unconditional
            # `snap_result["success"] = True` executed after the legs were
            # gathered.  _run_rebinding_md RETURNS {"success": False, "error": …}
            # on ~nine distinct failure paths (build failed, EM not converged,
            # missing topology, …) instead of raising, so a snapshot in which
            # EVERY leg failed was still recorded as a success.
            _legs = {k: v for k, v in snap_result.items()
                     if k.startswith("rebind_") and isinstance(v, dict)}
            snap_result["success"] = bool(_legs) and all(
                not v.get("error") and v.get("success", True)
                for v in _legs.values())
            snap_result["legs_total"] = len(_legs)
            snap_result["legs_failed"] = sorted(
                k for k, v in _legs.items()
                if v.get("error") or not v.get("success", True))
            if not snap_result["success"]:
                logger.error(
                    f"  snapshot {snap_result.get('snapshot', '?')}: "
                    f"{len(snap_result['legs_failed'])}/{len(_legs)} rebinding "
                    f"leg(s) FAILED: {snap_result['legs_failed']}")
            snapshot_results.append(snap_result)

        # Step 6: Analyze results
        # threshold=None: the RMSD threshold no longer decides anything (see
        # _analyze_rebinding_results docstring).
        results[target] = _analyze_rebinding_results(
            target, target_names, snapshot_results, threshold=None)

        # ── Provenance of the sample this target's numbers rest on ──
        _reps_used = sorted({j["replica"] for j in jobs})
        results[target]["phase4_pc_id"] = best_pc_id
        results[target]["phase4_accepted"] = pc_data.get("accepted")
        results[target]["phase4_quality_failures"] = pc_data.get("quality_failures")
        results[target]["source_replicas"] = _reps_used
        results[target]["n_source_replicas"] = len(_reps_used)
        results[target]["snapshots_per_replica"] = {
            str(r): sum(1 for j in jobs if j["replica"] == r) for r in _reps_used}
        if pc_data.get("accepted") is False:
            # Loud, machine-readable, and carried into every report that reads
            # this JSON — a recipe must not be built on a rejected MD without
            # the reader being told.
            results[target]["provisional"] = True
            results[target]["provisional_reason"] = (
                f"Phase 4 leg {best_pc_id} failed acceptance "
                f"({pc_data.get('quality_failures')}); the cavity these numbers "
                f"describe came from an MD that did not meet the convergence / "
                f"equilibration criteria.")

        # Step 7: Auto dual-imprinting if weak selectivity + has N-glycan IN ECL2
        target_result = results[target]
        # Count N-X-S/T sequons in the ECL2 sequence (X != P). The total-protein
        # `n_glycan_sites_known` counts glycans anywhere (including ECL1 / intracellular)
        # — but APBA grafted into our ECL2-imprinted cavity can only reach glycans
        # that actually sit on the ECL2 surface. Counting ECL2-local sequons here
        # prevents false-positive triggers (e.g. CD9: 1 total glycan at N52 in ECL1,
        # 0 in ECL2 → dual must NOT trigger).
        ecl2_seq = phase1_results[target].get("ecl2_sequence", "") or ""
        n_glycan = sum(
            1 for i in range(len(ecl2_seq) - 2)
            if ecl2_seq[i] == "N" and ecl2_seq[i + 1] != "P"
            and ecl2_seq[i + 2] in ("S", "T"))
        sel = target_result.get("selectivity", {})
        # Dual-imprinting criteria:
        # 1. Any cross-target SI < 1.5 AND p > 0.05 (not statistically selective)
        # 2. N-glycan ≥ 1 *in ECL2* (APBA needs accessible diol on ECL2 surface)
        # 3. Rebinding ≥ N/3 (cavity works; if too low, monomer combo itself is bad)
        # 4. NO cross-target is already size-excluded. Size-exclusion is a stronger,
        #    mechanism-level selectivity than APBA glycan recognition — if it's already
        #    in effect, adding APBA only introduces background.
        any_size_excluded = any(
            s.get("selectivity_label") == "size-excluded"
            for s in sel.values())
        any_not_significant = any(
            s.get("selectivity_label") in ("weak", "cross-reactive")
            and (s.get("p_value") is None or s.get("p_value") > 0.05)
            for s in sel.values())
        n_rebound = target_result.get("n_rebound", 0)
        # Adaptive rebound threshold: ≥3 in full run (10 snaps), ≥1 in trial (1 snap)
        rebound_threshold = max(1, REBINDING_N_SNAPSHOTS // 3)

        # Skip dual-imprinting entirely in EV-approach mode: fresh EV template
        # already carries its glycans in the CHARMM-GUI Membrane Builder + Glycan
        # Reader outputs, so adding APBA to the cavity would double-count the
        # recognition signal and would also try to rebuild via the naked-template
        # path (_run_dual_imprinting_vip) which doesn't understand the fresh-EV
        # rebinding return shape.
        from .config import (PHASE4_MEMBRANE_MODE as _MEMB_MODE,
                              PHASE5_TRITON_REMOVAL_MODE as _TRIT_MODE)
        _in_ev_mode = bool(_MEMB_MODE and _TRIT_MODE)

        if _in_ev_mode:
            target_result["dual_imprinting"] = None
            target_result["dual_imprinting_reason"] = (
                "EV-approach mode active: fresh CD-in-EV already carries "
                "glycans from CHARMM-GUI Glycan Reader outputs; no APBA layer "
                "added.")
            logger.info(f"  {target}: EV-approach mode → dual-imprinting skipped")
        elif (any_not_significant and n_glycan > 0 and n_rebound >= rebound_threshold
                and not any_size_excluded):
            logger.info(f"\n  *** Dual-imprinting triggered for {target} ***")
            logger.info(f"      Reason: non-significant selectivity (SI<1.5, p>0.05) "
                        f"+ {n_glycan} ECL2 N-glycan sites + rebinding {n_rebound}/{REBINDING_N_SNAPSHOTS}")
            logger.info(f"      Action: adding APBA (boronic acid) to cavity for glycan recognition")

            dual_results = _run_dual_imprinting_vip(
                target, target_names, snapshot_results,
                phase1_results, p4_md_dir, output_dir / target,
                n_glycan=n_glycan,
            )
            target_result["dual_imprinting"] = dual_results
            target_result["dual_imprinting_reason"] = (
                f"SI weak + {n_glycan} ECL2 N-glycan sites → APBA layer 2")
        elif any_not_significant and n_glycan == 0:
            target_result["dual_imprinting"] = None
            target_result["dual_imprinting_reason"] = (
                "Weak selectivity but no N-glycan sites in ECL2 — dual-imprinting not applicable")
            logger.info(f"  {target}: weak selectivity but no ECL2 N-glycan → "
                        f"dual-imprinting not applicable")
        elif any_size_excluded:
            target_result["dual_imprinting"] = None
            target_result["dual_imprinting_reason"] = (
                "Cross-target already size-excluded — size/shape selectivity is "
                "mechanism-level; APBA layer would only add background, skipped")
            logger.info(f"  {target}: size-exclusion already provides selectivity → "
                        f"dual-imprinting skipped")

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


# Semantic alias — this module IS Phase 5 (VIP rebinding) in the pipeline.
# `run_phase6` is the historical name; `run_phase5_rebinding` is what new
# callers should reach for. Both bind the same object.
run_phase5_rebinding = run_phase6


# ── Frame Selection ──────────────────────────────────────────

def _select_equilibrium_frames(traj_path, top_path, topol_path,
                                n_frames=5, cutoff_A=6.0):
    """Select evenly spaced frames from equilibrated (last 50%) trajectory.
    No cherry-picking — represents random polymerization timing.
    Also reports contact count per frame for reference.

    BLOCKER 06: every selected frame now carries the REAL simulation time read
    off the trajectory (`u.trajectory[i].time`) plus the trajectory's own dt.
    Downstream must use `time_ps`; deriving a time from the index with an
    assumed dt is what turned the ten "evenly spaced equilibrium snapshots"
    into two frames from t = 2 ns and 3 ns of a 350 ns run.
    """
    try:
        import MDAnalysis as mda

        u = mda.Universe(str(top_path), str(traj_path))
        protein = u.select_atoms("protein")
        non_protein = u.select_atoms(f"not protein and not resname {_NON_MONOMER_RESNAMES}")

        if len(protein) == 0 or len(non_protein) == 0:
            return []

        # Last 50% of trajectory, evenly spaced
        n_total = len(u.trajectory)
        start = n_total // 2
        interval = (n_total - start) // (n_frames + 1)
        if interval < 1:
            logger.warning(
                f"  Only {n_total} frames available for {n_frames} snapshots — "
                f"spacing collapses to <1 frame; snapshots would repeat.")
            interval = 1

        dt_ps = float(u.trajectory.dt)

        selected = []
        seen_idx = set()
        for i in range(1, n_frames + 1):
            frame_idx = start + i * interval
            if frame_idx >= n_total:
                frame_idx = n_total - 1
            if frame_idx in seen_idx:
                # Do not emit the same frame twice under a different snapshot
                # label — that is exactly the silent duplication BLOCKER 06 hid.
                logger.warning(f"  Frame {frame_idx} already selected — "
                               f"dropping duplicate snapshot request {i}")
                continue
            seen_idx.add(frame_idx)

            ts = u.trajectory[frame_idx]
            time_ps = float(ts.time)
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
                "time_ps": time_ps,
                "traj_dt_ps": dt_ps,
                "n_frames_total": int(n_total),
                "total_contacts": total,
            })

        return selected

    except Exception as e:
        logger.warning(f"Frame selection failed: {e}")
        return []


# ── Snapshot provenance / distinctness ───────────────────────────

_GRO_TITLE_TIME_RE = re.compile(r"t\s*=\s*([-+0-9.eE]+)")


def _gro_title_time_ps(gro_path: Path):
    """Read the `t= <ps>` stamp GROMACS writes into a .gro title line."""
    try:
        with open(gro_path) as f:
            title = f.readline()
    except Exception:
        return None
    m = _GRO_TITLE_TIME_RE.search(title)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _gro_fingerprint(gro_path: Path):
    """md5 over the coordinate columns of the non-solvent atoms of a .gro.

    Used to ASSERT that two "different" snapshots really are different
    structures. Solvent is excluded so that the fingerprint tracks the
    template + monomer configuration rather than water bookkeeping.
    """
    try:
        lines = Path(gro_path).read_text().split("\n")
        n_atoms = int(lines[1].strip())
    except Exception:
        return None
    h = hashlib.md5()
    for line in lines[2:2 + n_atoms]:
        if len(line) < 44:
            continue
        resname = line[5:10].strip()
        if resname in ("SOL", "HOH", "WAT", "TIP3", "NA", "CL", "NA+", "CL-"):
            continue
        h.update(line[20:44].encode())
    return h.hexdigest()


# ── Cavity Creation ──────────────────────────────────────────

def _create_cavity(traj_path, top_path, topol_path, frame_idx, output_dir,
                   frame_time_ps=None):
    """
    Extract specific frame → create position restraints for monomers.
    Keep full system (protein + monomers + water) — topology unchanged.
    For rebinding: replace protein coordinates with new template.

    BLOCKER 06 FIX. This function used to compute the dump time as
    `frame_idx * 10` ps. The reduced Phase 4 trajectory it is handed is stored
    at 1000 ps/frame, so indices 191-335 (the documented last half of a
    351-frame / 350 ns run) were dumped at t = 1910-3350 ps — i.e. every
    "equilibrium snapshot" came from the first 1% of the trajectory, and
    trjconv's nearest-frame `-dump` collapsed several of them onto the SAME
    two frames. An index is now NEVER converted to a time with an assumed dt:
    the real time is read from the trajectory (`u.trajectory[i].time`), the
    time actually written by trjconv is verified against it via the `t=` stamp
    in the .gro title, and a coordinate fingerprint is returned so the caller
    can assert the snapshots are distinct.
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

        # ── Step 0: the REAL time of frame_idx, read off the trajectory ──
        traj_dt_ps = None
        try:
            u_probe = mda.Universe(str(top_path), str(traj_path))
            n_traj = len(u_probe.trajectory)
            if not (0 <= frame_idx < n_traj):
                raise IndexError(f"frame {frame_idx} outside trajectory "
                                 f"(0..{n_traj - 1})")
            ts_probe = u_probe.trajectory[frame_idx]
            real_time_ps = float(ts_probe.time)
            traj_dt_ps = float(u_probe.trajectory.dt)
        except Exception as te:
            # NO fallback to an assumed dt. Silently guessing the timebase is
            # the defect being fixed; refuse the snapshot instead.
            return {"success": False,
                    "error": f"could not read real frame time for frame "
                             f"{frame_idx} from {traj_path}: {te}"}

        if frame_time_ps is not None and abs(float(frame_time_ps) - real_time_ps) > 1e-6:
            logger.warning(
                f"    Caller-supplied time {frame_time_ps} ps disagrees with the "
                f"trajectory's own time {real_time_ps} ps for frame {frame_idx}; "
                f"using the trajectory value.")
        frame_time_ps = real_time_ps
        logger.info(f"    Frame {frame_idx} -> t = {frame_time_ps:g} ps "
                    f"(trajectory dt = {traj_dt_ps:g} ps, {n_traj} frames)")

        # Step 1: Use gmx trjconv to extract centered, PBC-corrected frame
        # -pbc mol: keep molecules whole (no bond crossing PBC)
        # -center: center protein in box
        # This fixes "protein at edge" issue from Phase 4 PBC split
        frame_gro = output_dir / "frame.gro"
        time_source = None
        try:
            # Need a tpr; use the topology if it's a tpr, else regenerate
            top_for_trjconv = str(top_path) if str(top_path).endswith('.tpr') else None
            if top_for_trjconv is None:
                # Find tpr in same dir as topol.top
                tpr_candidate = Path(topol_path).parent / "md.tpr"
                if tpr_candidate.exists():
                    top_for_trjconv = str(tpr_candidate)
            if top_for_trjconv:
                # -pbc modes:
                #   aqueous system: `-pbc mol -center` (Protein as reference)
                #     works fine; Protein is a single moleculetype in one place.
                #   membrane system: centering on Protein while the bilayer
                #     wraps around it can slice the bilayer at the box edge.
                #     `-pbc mol -boxcenter tric` first puts everything in the
                #     unit cell around the box centre, THEN a second pass
                #     `-pbc mol -center` on Protein places the protein at the
                #     box centre without touching bilayer wrapping.
                from .config import (PHASE4_MEMBRANE_MODE as _MEMB,
                                      PHASE5_TRITON_REMOVAL_MODE as _TRIT)
                _memb = bool(_MEMB and _TRIT)
                if _memb:
                    # Two-step trjconv: whole-molecules first (no centering),
                    # then centre Protein on the whole-molecules output.
                    frame_whole = frame_gro.with_suffix(".whole.gro")
                    subprocess.run(
                        [GMX_BIN, "trjconv", "-f", str(traj_path),
                         "-s", top_for_trjconv, "-o", str(frame_whole),
                         "-pbc", "mol", "-ur", "compact",
                         "-dump", f"{frame_time_ps:.6f}"],
                        input="0\n", capture_output=True, text=True, timeout=300)
                    cmd = [GMX_BIN, "trjconv",
                           "-f", str(frame_whole),
                           "-s", top_for_trjconv,
                           "-o", str(frame_gro),
                           "-pbc", "mol",
                           "-center"]
                else:
                    cmd = [GMX_BIN, "trjconv",
                           "-f", str(traj_path),
                           "-s", top_for_trjconv,
                           "-o", str(frame_gro),
                           "-pbc", "mol",
                           "-center",
                           "-dump", f"{frame_time_ps:.6f}"]
                proc = subprocess.run(cmd, input="1\n0\n", capture_output=True,
                                      text=True, timeout=300)
                if not frame_gro.exists():
                    raise RuntimeError(f"trjconv failed: {proc.stderr[-300:]}")
                # VERIFY the frame trjconv actually dumped. `-dump` snaps to the
                # nearest frame, so a wrong time silently yields a wrong (and
                # possibly duplicate) structure — the BLOCKER 06 failure mode.
                dumped_t = _gro_title_time_ps(frame_gro)
                if dumped_t is None:
                    raise RuntimeError(
                        "trjconv wrote no `t=` stamp in the .gro title — cannot "
                        "verify which frame was extracted")
                tol = max(0.5 * (traj_dt_ps or 0.0), 1e-3)
                if abs(dumped_t - frame_time_ps) > tol:
                    raise RuntimeError(
                        f"trjconv dumped t={dumped_t} ps but frame {frame_idx} "
                        f"is at t={frame_time_ps} ps (tolerance {tol} ps)")
                time_source = "trjconv"
                logger.info(f"    Frame {frame_idx} extracted at t={dumped_t:g} ps, "
                            f"centered via gmx trjconv (-pbc mol -center)")
                # Load centered frame for downstream
                u = mda.Universe(str(frame_gro))
            else:
                raise RuntimeError("No tpr available for trjconv")
        except Exception as ce:
            logger.warning(f"    trjconv centering failed ({ce}); falling back to MDAnalysis")
            u = mda.Universe(str(top_path), str(traj_path))
            u.trajectory[frame_idx]          # index directly — no time arithmetic
            time_source = "mdanalysis_index"
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
            # BLOCKER 06 provenance — the caller asserts distinctness on these
            "frame_idx": int(frame_idx),
            "frame_time_ps": float(frame_time_ps),
            "traj_dt_ps": traj_dt_ps,
            "time_source": time_source,
            "fingerprint": _gro_fingerprint(frame_gro),
        }

    except Exception as e:
        logger.error(f"Cavity creation failed: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# BLOCKER 07: symmetric own / cross template placement
# ══════════════════════════════════════════════════════════════════════════
# The own and cross legs used to be different experiments:
#   own   — rebind_system.gro was md5-identical to frame.gro. Nothing was
#           re-placed; the "rebinding" leg was a continuation of the already
#           equilibrated Phase 4 pose.
#   cross — a differently shaped protein was dropped into the old protein's
#           hole with a COM shift only: no rotational alignment, waters removed
#           around the OLD protein's position rather than the new one, and no
#           ion rebalancing. 22 of 90 legs died in energy minimisation.
# The selectivity index is this pipeline's deliverable and its numerator and
# denominator came from those two different protocols. Everything below is ONE
# procedure, run identically for own and cross:
#   strip template -> pdb2gmx the new template -> principal-axis placement with
#   an orientation search -> delete clashing solvent/ions -> rebalance charge
#   with genion -> energy minimise -> VERIFY EM converged before proceeding.

_SOLVENT_RESNAMES = frozenset({"SOL", "HOH", "WAT", "TIP3", "TIP4", "SPC"})
_ION_RESNAMES = frozenset({"NA", "CL", "NA+", "CL-", "K", "MG", "CA", "ZN"})
_MONOMER_SEL = "not protein and not resname SOL HOH WAT TIP3 NA CL"

_WATER_INCLUDE_RE = re.compile(
    r'^[ \t]*#include[ \t]+"[^"]*\.ff/(?:tip3p|tip4p|tip5p|spc|spce)\.itp"[ \t]*$',
    re.M | re.I)


def _protein_block_span(top_text: str):
    """(start, end) of the inline protein [ moleculetype ] region of a .top.

    Starts at the first `[ moleculetype ]` (monomer topologies are #include'd
    above it, so the first inline moleculetype is the protein) and ends at the
    water-topology #include.
    """
    start = top_text.find("[ moleculetype ]")
    if start < 0:
        return None
    m = _WATER_INCLUDE_RE.search(top_text, start)
    if m:
        return start, m.start()
    for marker in ("; Include water topology", "[ system ]"):
        i = top_text.find(marker, start)
        if i >= 0:
            return start, i
    return None


def _moleculetype_names(block: str) -> list:
    """Names declared by every [ moleculetype ] directive in a topology block."""
    names, expect = [], False
    for line in block.split("\n"):
        s = line.strip()
        if s.startswith("[") and "moleculetype" in s:
            expect = True
            continue
        if expect:
            if not s or s.startswith((";", "#")):
                continue
            names.append(s.split()[0])
            expect = False
    return names


def _parse_molecules_entries(top_text: str):
    """(molecules_start_index, lines, entries) for the final [ molecules ] block.

    entries is a list of [line_index, name, count].
    """
    idx = top_text.rfind("[ molecules ]")
    if idx < 0:
        return None
    lines = top_text[idx:].split("\n")
    entries = []
    for k, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith((";", "[", "#")):
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                entries.append([k, parts[0], int(parts[1])])
            except ValueError:
                continue
    return idx, lines, entries


def _rebuild_molecules_section(top_text: str, old_protein_names: list,
                               new_protein_entries: list, gro_counts: dict):
    """Swap the protein entries and reconcile the rest against the real .gro.

    Returns (new_top_text, report) or (None, error_string). Monomer counts are
    NEVER silently adjusted — a mismatch there means the merge dropped a
    monomer and is a hard error.
    """
    parsed = _parse_molecules_entries(top_text)
    if parsed is None:
        return None, "no [ molecules ] section in cavity topology"
    idx, lines, entries = parsed

    report = {"protein_replaced": [], "counts_updated": {}, "dropped": []}
    out_lines = list(lines)
    handled = set()
    first_protein_line = None

    for k, name, count in entries:
        if name in old_protein_names:
            if first_protein_line is None:
                first_protein_line = k
            out_lines[k] = None                      # remove; re-inserted below
            report["protein_replaced"].append(name)
            handled.add(k)
            continue
        real = gro_counts.get(name)
        if real is None:
            continue                                  # not resolvable from the gro
        if real == count:
            continue
        if name in _SOLVENT_RESNAMES or name in _ION_RESNAMES:
            if real == 0:
                out_lines[k] = None
                report["dropped"].append(name)
            else:
                out_lines[k] = f"{name}     {real}"
                report["counts_updated"][name] = [count, real]
        else:
            return None, (f"monomer molecule {name} count changed "
                          f"{count} -> {real} during template replacement; the "
                          f"cavity was damaged, refusing to continue")

    if first_protein_line is None:
        return None, (f"none of the cavity's protein moleculetypes "
                      f"{old_protein_names} appear in [ molecules ]")

    new_block = [f"{n}     {c}" for n, c in new_protein_entries]
    out_lines[first_protein_line] = "\n".join(new_block)

    rebuilt = "\n".join(l for l in out_lines if l is not None)
    return top_text[:idx] + rebuilt, report


def _heavy_mask(atomgroup):
    """Heavy-atom boolean mask by NAME (no mass guessing) — matches
    utils_analysis.compute_steric_clash's `not name H*` convention."""
    names = np.array([n.strip().upper() for n in atomgroup.names])
    return np.array([not n.startswith("H") for n in names], dtype=bool)


def _gyration_axes(positions):
    """Unit-weighted principal axes of a point cloud, largest spread first.

    Returns (centroid, axes, eigenvalues) with axes[:, k] the k-th principal
    direction. The frame is forced RIGHT-HANDED (det = +1) so that composing
    two of them always yields a proper rotation — otherwise every candidate
    orientation is a reflection and placement fails outright.
    """
    centroid = positions.mean(axis=0)
    x = positions - centroid
    tensor = x.T @ x
    evals, evecs = np.linalg.eigh(tensor)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]
    return centroid, axes, evals[order]


def _place_template(old_positions, new_positions, monomer_positions,
                    box=None, clash_cutoff_A=2.0, contact_cutoff_A=6.0):
    """Rigid-body placement of a template into the cavity the old one vacated.

    Principal axes of the NEW template are aligned onto the principal axes of
    the OLD template, and the centroids are matched. Principal axes are only
    defined up to sign, so all four proper-rotation sign combinations are
    enumerated and scored by how well the placed template FILLS THE VOLUME THE
    OLD TEMPLATE VACATED — a symmetric Chamfer distance between the two point
    clouds. That is the literal statement of the task ("put it in the hole"),
    it is deterministic, and it is identical for own and cross legs, which is
    the whole point of BLOCKER 07. For an own leg (same molecule) the score is
    exactly zero at the true orientation, so the own template lands back in its
    equilibrated pose and the two legs differ ONLY in which template is placed.

    Clash counts against the restrained monomers are computed for every
    candidate and reported, but they do NOT choose the orientation: minimum
    clash was measured to pick a 180-degree-flipped pose on chiral templates.
    Residual clashes are what energy minimisation (and the EM gate) is for.

    NOT A DOCKING ALGORITHM. One rigid placement per leg is one draw, not a
    search over binding modes.

    Returns (placed_positions, diagnostics).
    """
    from MDAnalysis.lib.distances import distance_array

    c_old, A_old, ev_old = _gyration_axes(old_positions)
    c_new, A_new, ev_new = _gyration_axes(new_positions)
    centred = new_positions - c_new

    # Near-degenerate principal moments => the axis frame is not well defined
    # and the four-orientation search is not a meaningful sample of SO(3).
    # Say so; a silently arbitrary orientation is exactly the class of defect
    # this pass exists to remove.
    degenerate = []
    for lbl, ev in (("old", ev_old), ("new", ev_new)):
        ev = np.asarray(ev, dtype=float)
        if ev[0] > 0 and (ev[0] - ev[2]) / ev[0] < 0.15:
            degenerate.append(lbl)
    if degenerate:
        logger.warning(
            f"    Principal axes are near-degenerate for the {'/'.join(degenerate)} "
            f"template (moments old={np.round(ev_old, 1).tolist()}, "
            f"new={np.round(ev_new, 1).tolist()}): the placement orientation is "
            f"poorly determined and this leg's pose should be treated as one "
            f"arbitrary draw, not a docked pose.")

    candidates = []
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        A = A_new * np.array(signs, dtype=float)      # scale columns
        R = A_old @ A.T
        if np.linalg.det(R) < 0:                      # keep it a proper rotation
            continue
        placed = centred @ R.T + c_old
        # symmetric Chamfer distance to the vacated volume
        d_old = distance_array(placed, old_positions, box=box)
        overlap = float(d_old.min(axis=1).mean() + d_old.min(axis=0).mean())
        if len(monomer_positions):
            d = distance_array(placed, monomer_positions, box=box)
            n_clash = int(np.sum(d < clash_cutoff_A))
            n_contact = int(np.sum(d.min(axis=1) < contact_cutoff_A))
        else:
            n_clash, n_contact = 0, 0
        candidates.append({"signs": signs, "positions": placed,
                           "overlap": overlap,
                           "n_clash": n_clash, "n_contact": n_contact})

    if not candidates:
        raise RuntimeError("principal-axis placement produced no proper rotation")

    best = min(candidates, key=lambda c: (round(c["overlap"], 6), c["n_clash"]))
    diag = {
        "chosen_signs": list(best["signs"]),
        "vacated_volume_chamfer_A": round(best["overlap"], 3),
        "placement_clashes": best["n_clash"],
        "placement_contacts": best["n_contact"],
        "orientations_tried": [
            {"signs": list(c["signs"]), "chamfer_A": round(c["overlap"], 3),
             "n_clash": c["n_clash"], "n_contact": c["n_contact"]}
            for c in candidates],
        "centroid_shift_A": round(float(np.linalg.norm(c_old - c_new)), 2),
        "principal_moments_old": np.round(ev_old, 1).tolist(),
        "principal_moments_new": np.round(ev_new, 1).tolist(),
        "axes_near_degenerate": degenerate,
    }
    return best["positions"], diag


def _em_convergence(md_dir: Path) -> dict:
    """Parse em.log. `ok` is False unless GROMACS actually converged."""
    md_dir = Path(md_dir).resolve()
    log = md_dir / "em.log"
    gro = md_dir / "em.gro"
    out = {"ok": False, "em_gro": gro.exists(), "fmax": None, "epot": None,
           "reported_converged": False, "reason": None}
    if not log.exists():
        out["reason"] = "em.log missing"
        return out
    text = log.read_text(errors="replace")
    out["reported_converged"] = "converged to Fmax" in text
    # \S+ so that `inf` / `nan` are captured too — a blown-up minimisation must
    # be READ, not skipped as an unparseable line.
    m = re.findall(r"Maximum force\s*=\s*(\S+)", text)
    if m:
        try:
            out["fmax"] = float(m[-1])
        except ValueError:
            out["fmax"] = float("nan")
    m = re.findall(r"Potential Energy\s*=\s*(\S+)", text)
    if m:
        try:
            out["epot"] = float(m[-1])
        except ValueError:
            out["epot"] = float("nan")

    tol = float(_cfg("REBINDING_EM_FMAX_TOL", 1000.0))
    if not out["em_gro"]:
        out["reason"] = "em.gro not produced (mdrun failed or blew up)"
    elif out["epot"] is None or not np.isfinite(out["epot"]):
        out["reason"] = f"potential energy not finite ({out['epot']})"
    elif out["fmax"] is None:
        out["reason"] = "no Maximum force reported in em.log"
    elif not np.isfinite(out["fmax"]):
        out["reason"] = f"Fmax not finite ({out['fmax']})"
    elif out["reported_converged"] or out["fmax"] <= tol:
        out["ok"] = True
    else:
        out["reason"] = (f"Fmax {out['fmax']:.3g} kJ/mol/nm > tolerance {tol:g} "
                         f"and GROMACS did not report convergence")
    return out


def _build_rebinding_system(cavity_gro, cavity_top, template_pdb, md_dir,
                            p4_md_dir=None, leg_label="") -> dict:
    """THE symmetric placement procedure. Run for own AND cross legs alike.

    Leaves md_dir/ionized.gro + md_dir/topol.top ready for energy minimisation.
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance
    from .utils_gromacs import setup_protein_topology, _gmx, MDP_EM

    md_dir = Path(md_dir)
    md_dir.mkdir(parents=True, exist_ok=True)
    diag = {"protocol": "symmetric_replacement", "leg": leg_label}

    # ── 1. read the cavity ───────────────────────────────────────
    u_sys = mda.Universe(str(cavity_gro))
    old_prot = u_sys.select_atoms("protein")
    if len(old_prot) == 0:
        return {"success": False, "error": "cavity contains no protein to strip"}
    monomers = u_sys.select_atoms(_MONOMER_SEL)
    if len(monomers) == 0:
        return {"success": False, "error": "cavity contains no monomers"}
    box = u_sys.dimensions

    # ── 2. build the incoming template from scratch (pdb2gmx) ────
    build_dir = md_dir / "template_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        new_gro = setup_protein_topology(Path(template_pdb), build_dir)
    except Exception as e:
        return {"success": False, "error": f"pdb2gmx on {Path(template_pdb).name} "
                                           f"failed: {e}"}
    new_top_text = (build_dir / "topol.top").read_text()
    u_new = mda.Universe(str(new_gro))
    new_atoms = u_new.atoms

    # ── 3. principal-axis placement with orientation search ──────
    old_heavy = old_prot[_heavy_mask(old_prot)]
    new_heavy_mask = _heavy_mask(new_atoms)
    mon_heavy = monomers[_heavy_mask(monomers)]
    clash_cut = float(_cfg("REBINDING_CLASH_CUTOFF_A", 2.0))
    try:
        placed_heavy, place_diag = _place_template(
            old_heavy.positions, new_atoms.positions[new_heavy_mask],
            mon_heavy.positions, box=box, clash_cutoff_A=clash_cut)
    except Exception as e:
        return {"success": False, "error": f"template placement failed: {e}"}
    # apply the same rigid transform to ALL atoms (H included)
    c_old, A_old, _ = _gyration_axes(old_heavy.positions)
    c_new, A_new, _ = _gyration_axes(new_atoms.positions[new_heavy_mask])
    A = A_new * np.array(place_diag["chosen_signs"], dtype=float)
    R = A_old @ A.T
    new_atoms.positions = (new_atoms.positions - c_new) @ R.T + c_old
    diag.update(place_diag)

    # ── 4. delete solvent/ions clashing with the PLACED template ──
    #     (the old code removed water around the OLD protein's position, which
    #      is the one place there is guaranteed to be no new protein)
    water_cut = float(_cfg("REBINDING_WATER_CLASH_CUTOFF_A", 2.4))
    non_prot = u_sys.select_atoms("not protein")
    solv_ion = non_prot.select_atoms(
        "resname " + " ".join(sorted(_SOLVENT_RESNAMES | _ION_RESNAMES)))
    drop_resix = set()
    if len(solv_ion):
        pairs = capped_distance(new_atoms.positions, solv_ion.positions,
                                max_cutoff=water_cut, box=box,
                                return_distances=False)
        if len(pairs):
            drop_resix = set(np.unique(solv_ion.resindices[pairs[:, 1]]).tolist())
    keep = non_prot
    if drop_resix:
        mask = ~np.isin(non_prot.resindices, list(drop_resix))
        keep = non_prot[mask]
    diag["solvent_residues_removed"] = len(drop_resix)
    diag["water_clash_cutoff_A"] = water_cut

    # ── 5. merge: new template first, then monomers/solvent/ions in order ──
    merged = mda.Merge(new_atoms, keep)
    merged.dimensions = box
    placed_gro = md_dir / "placed_system.gro"
    merged.atoms.write(str(placed_gro))
    shutil.copy2(str(placed_gro), str(md_dir / "rebind_system.gro"))
    diag["n_atoms_placed_system"] = int(len(merged.atoms))

    # ── 6. topology surgery ──────────────────────────────────────
    cav_text = Path(cavity_top).read_text()
    cav_span = _protein_block_span(cav_text)
    new_span = _protein_block_span(new_top_text)
    if cav_span is None or new_span is None:
        return {"success": False,
                "error": "could not locate the protein moleculetype block "
                         "(cavity or pdb2gmx topology)"}
    old_names = _moleculetype_names(cav_text[cav_span[0]:cav_span[1]])
    new_block = new_top_text[new_span[0]:new_span[1]].rstrip() + "\n\n"
    new_parsed = _parse_molecules_entries(new_top_text)
    if new_parsed is None or not new_parsed[2]:
        return {"success": False,
                "error": "pdb2gmx topology has no [ molecules ] entries"}
    new_prot_entries = [(n, c) for _, n, c in new_parsed[2]]

    gro_counts = {}
    for rn in set(keep.residues.resnames):
        gro_counts[str(rn)] = int(np.sum(keep.residues.resnames == rn))

    merged_top = cav_text[:cav_span[0]] + new_block + cav_text[cav_span[1]:]
    merged_top, report = _rebuild_molecules_section(
        merged_top, old_names, new_prot_entries, gro_counts)
    if merged_top is None:
        return {"success": False, "error": report}
    diag["topology"] = report
    (md_dir / "topol.top").write_text(merged_top)

    # posre.itp belongs to the OLD protein — replace it, force-overwriting the
    # copies the ITP loops made. (#ifdef POSRES is off in rebinding, but a stale
    # restraint file with the wrong atom count is a landmine.)
    for posre in build_dir.glob("posre*.itp"):
        shutil.copy2(str(posre), str(md_dir / posre.name))

    # ── 7. rebalance ions for the NEW template's net charge ──────
    # `check=False` where utils_gromacs supports it: this function does its own
    # explicit output-file checks and returns a specific error string, which is
    # more useful than a generic GromacsError. If the failure is real it is
    # still reported — just with the reason attached.
    def _run_gmx(args, **kw):
        try:
            return _gmx(args, md_dir, check=False, **kw)
        except TypeError:
            return _gmx(args, md_dir, **kw)          # older signature
        except Exception as e:                        # raised despite check
            class _R:
                returncode, stdout, stderr = 1, "", str(e)
            return _R()

    (md_dir / "ions.mdp").write_text(MDP_EM)
    r = _run_gmx(["grompp", "-f", str(md_dir / "ions.mdp"),
                  "-c", str(placed_gro), "-p", str(md_dir / "topol.top"),
                  "-o", str(md_dir / "ions.tpr"), "-maxwarn", "10"])
    if not (md_dir / "ions.tpr").exists():
        return {"success": False,
                "error": f"grompp on the re-placed system failed "
                         f"(topology/coordinate mismatch): "
                         f"{(r.stderr or '')[-600:]}"}
    def _mol_totals(top_path):
        """name -> TOTAL count. genion appends a SECOND `NA` line rather than
        incrementing the existing one, so duplicates must be SUMMED — a dict
        comprehension would silently report 'no ions added'."""
        parsed = _parse_molecules_entries(Path(top_path).read_text())
        totals = {}
        for _, n, c in (parsed[2] if parsed else []):
            totals[n] = totals.get(n, 0) + c
        return totals

    before = _mol_totals(md_dir / "topol.top")
    r = _run_gmx(["genion", "-s", str(md_dir / "ions.tpr"),
                  "-o", str(md_dir / "ionized.gro"), "-p", str(md_dir / "topol.top"),
                  "-pname", "NA", "-nname", "CL", "-neutral"],
                 input_text="SOL\n")
    if not (md_dir / "ionized.gro").exists():
        return {"success": False,
                "error": f"genion charge rebalance failed: {(r.stderr or '')[-600:]}"}
    after = _mol_totals(md_dir / "topol.top")
    diag["ions_rebalanced"] = {
        k: [before.get(k, 0), after.get(k, 0)]
        for k in sorted(set(before) | set(after))
        if k in _ION_RESNAMES and before.get(k, 0) != after.get(k, 0)}
    diag["net_charge_rebalance_needed"] = bool(diag["ions_rebalanced"])
    diag["solvent_after_rebalance"] = {k: after.get(k) for k in ("SOL",)
                                       if k in after}

    return {"success": True, **diag}


# ── Rebinding MD ─────────────────────────────────────────────

def _aggregate_ev_placements(legs: list, target: str,
                              is_own_target: bool) -> dict:
    """Combine N fresh-EV placement legs into one rebinding-leg dict.

    Return shape mirrors `_run_rebinding_md` legacy output so downstream
    `_analyze_rebinding_results` sees the same keys. Median is used across
    placements to be outlier-resistant (a single bad initial rotation should
    not dominate the leg's verdict).
    """
    ok = [L for L in legs if L and L.get("success")]
    n_ok = len(ok)
    if not ok:
        first_err = next(
            (L.get("error") for L in legs if L and L.get("error")),
            "all placements failed")
        return {
            "success":         False,
            "protocol":        "ev_approach",
            "is_own_target":   bool(is_own_target),
            "target":          target,
            "n_placements":    len(legs),
            "n_placements_ok": 0,
            "error":           f"all placements failed: {first_err}",
            "per_placement":   legs,
        }
    n_persist = [int(L.get("n_persistent_residues", 0)) for L in ok]
    fpers = [float(L.get("fraction_persistent", 0.0)) for L in ok]
    n_persist_med = float(np.median(n_persist))
    fpers_med = float(np.median(fpers))
    return {
        "success":              True,
        "protocol":             "ev_approach",
        "is_own_target":        bool(is_own_target),
        "target":               target,
        "n_placements":         len(legs),
        "n_placements_ok":      n_ok,
        # PROMOTED to leg top-level so _analyze_rebinding_results (and the
        # verify_phase5 PCSI branch) can consume them without a special-case
        # protocol check.
        "n_persistent_residues": int(round(n_persist_med)),
        "fraction_persistent":  fpers_med,
        "n_persistent_min":     int(min(n_persist)),
        "n_persistent_max":     int(max(n_persist)),
        "n_persistent_all":     list(n_persist),
        "per_placement":        legs,
    }


def _run_rebinding_md(cavity_gro, cavity_top, template_pdb,
                       output_dir, time_ns=20, p4_md_dir=None,
                       is_own_target=True, target=None, placement_seed=None):
    """
    Place template near cavity center, run MD with monomers restrained.
    Analyze if template stays in cavity (RMSD).

    DISPATCH: when PHASE4_MEMBRANE_MODE + PHASE5_TRITON_REMOVAL_MODE are both
    True and a target name is available (either explicit `target` kwarg, or
    inferred from `template_pdb` basename), the legacy naked-template
    rebinding is bypassed and utils_ev_approach.run_ev_approach_leg is called
    instead — it consumes the Triton-lysed cavity + a fresh CD-in-EV assembly
    from CHARMM-GUI outputs. Callers that pass `target=` explicitly get
    deterministic seeding across PHASE5_FRESH_EV_PLACEMENTS replicas.
    """
    from .utils_gromacs import (_gmx, run_full_md_pipeline)
    from .config import (MD_TEMPERATURE_K, MD_PRESSURE_BAR, MD_GPU_ID,
                         REBINDING_RMSD_THRESHOLD, GMX_BIN,
                         PHASE4_MEMBRANE_MODE, PHASE5_TRITON_REMOVAL_MODE)

    output_dir = Path(output_dir)
    md_dir = output_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    # ── EV-approach dispatch ──
    # Fire only when BOTH toggles are on AND we can resolve a target name.
    # Runs PHASE5_FRESH_EV_PLACEMENTS independent placements per (target,
    # snapshot) with different rotation seeds and aggregates. Falling back to
    # the legacy path is deliberate: pre-existing callers that don't pass
    # target= remain functionally identical.
    if PHASE4_MEMBRANE_MODE and PHASE5_TRITON_REMOVAL_MODE:
        resolved_target = target
        if resolved_target is None:
            # Best-effort: extract 'CD9' etc. from template_pdb filename.
            import re as _re
            m = _re.search(r"(CD\d+)", str(template_pdb))
            resolved_target = m.group(1) if m else None
        if resolved_target is not None:
            from .utils_ev_approach import run_ev_approach_leg
            from .config import PHASE5_FRESH_EV_PLACEMENTS
            base_seed = (int(placement_seed) if placement_seed is not None
                          else abs(hash((str(output_dir), resolved_target))) % (2**31))
            n_placements = max(1, int(PHASE5_FRESH_EV_PLACEMENTS))
            legs = []
            for i in range(n_placements):
                seed_i = (base_seed + i * 10007) % (2**31)  # deterministic offset
                logger.info(
                    "EV-approach dispatch [%d/%d]: target=%s seed=%d cavity=%s",
                    i + 1, n_placements, resolved_target, seed_i,
                    Path(cavity_gro).name)
                leg = run_ev_approach_leg(
                    cavity_gro=cavity_gro, cavity_top=cavity_top,
                    target=resolved_target, seed=seed_i,
                    output_dir=output_dir / f"placement_{i}",
                    time_ns=time_ns,
                    is_own_target=is_own_target)
                legs.append(leg)
            return _aggregate_ev_placements(
                legs, target=resolved_target, is_own_target=is_own_target)
        else:
            logger.warning(
                "PHASE5 EV-approach flags active but target could not be "
                "resolved from %s — falling back to legacy naked-template "
                "rebinding. Pass target= explicitly to force EV-approach mode.",
                Path(template_pdb).name)

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

        # ── BLOCKER 07: ONE placement procedure for own AND cross ──
        # DEFAULT CHANGED. Previously `is_own_target=True` skipped placement
        # entirely and the own leg continued the equilibrated Phase 4 pose,
        # making the SI numerator and denominator incomparable. Now both legs
        # are stripped and re-placed identically. Set
        # REBINDING_SYMMETRIC_PLACEMENT=False in config to restore the old
        # asymmetric behaviour (results are then stamped as such).
        symmetric = bool(_cfg("REBINDING_SYMMETRIC_PLACEMENT", True))
        placement_diag = None

        if symmetric:
            logger.info(f"    Symmetric re-placement "
                        f"({'own' if is_own_target else 'cross'} leg): "
                        f"stripping template, re-placing {Path(template_pdb).name}")
            build = _build_rebinding_system(
                cavity_gro, cavity_top, template_pdb, md_dir,
                p4_md_dir=p4_md_dir,
                leg_label=("own" if is_own_target else "cross"))
            if not build.get("success"):
                logger.error(f"    Symmetric rebuild FAILED: {build.get('error')}")
                return {
                    "time_ns": 0,
                    "placement_protocol": "symmetric_replacement",
                    "rmsd_mean_A": None, "rmsd_final_A": None,
                    "rebound": None,
                    "status": "BUILD_FAILED",
                    "error": build.get("error"),
                }
            placement_diag = {k: v for k, v in build.items() if k != "success"}
            logger.info(
                f"    Placed: clashes={build.get('placement_clashes')}, "
                f"contacts={build.get('placement_contacts')}, "
                f"waters removed={build.get('solvent_residues_removed')}, "
                f"ions rebalanced={build.get('ions_rebalanced')}")
        elif is_own_target:
            # LEGACY ASYMMETRIC PATH — own leg is a continuation, not a rebinding.
            logger.warning(
                "    REBINDING_SYMMETRIC_PLACEMENT=False: own leg is a "
                "CONTINUATION of the Phase 4 pose, NOT a rebinding. Its RMSD / "
                "contacts are not comparable with the cross legs and any "
                "selectivity index built from them is invalid.")
            placement_diag = {"protocol": "legacy_continuation"}
        else:
            # Selectivity: copy cavity topology, replace protein with different head
            logger.info(f"    Selectivity rebinding: rebuilding with different head...")
            placement_diag = {"protocol": "legacy_com_shift"}
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
        # If the protein is too large for the cavity, severe clashes mean it is
        # physically REJECTED (selective). Detect this BEFORE EM (which would
        # explode with Max-force=inf), record as size-excluded, skip the MD.
        # BLOCKER 07: this now runs for the OWN leg too. Applying a rejection
        # filter to only one arm of a ratio biases the ratio; if the own
        # template no longer fits its own cavity that is a finding, not an
        # exemption.
        from .config import (REBINDING_CLASH_CUTOFF_A,
                             REBINDING_CLASH_THRESHOLD)
        from .utils_analysis import compute_steric_clash
        steric_clash = None
        try:
            steric_clash = compute_steric_clash(
                md_dir / "rebind_system.gro",
                clash_cutoff_A=REBINDING_CLASH_CUTOFF_A)
            logger.info(f"    Steric clash: {steric_clash['clash_count']} "
                        f"({steric_clash['clash_per_residue']}/residue)")
            if steric_clash["clash_count"] > REBINDING_CLASH_THRESHOLD:
                logger.info(f"    → SIZE-EXCLUDED: template too large "
                            f"for cavity ({steric_clash['clash_count']} clashes "
                            f"> {REBINDING_CLASH_THRESHOLD}) — REJECTED "
                            f"(size-exclusion selectivity)")
                return {
                    "time_ns": 0,
                    "placement": placement_diag,
                    "rmsd_mean_A": None,
                    "rmsd_final_A": None,
                    "rmsd_selffit_A": None,
                    "contacts": {"available": True, "n_persistent_residues": 0,
                                 "fraction_persistent": 0.0,
                                 "reason": "size-excluded before MD"},
                    "fraction_persistent": 0.0,
                    "n_persistent_residues": 0,
                    "rebound": False,
                    "size_excluded": True,
                    "is_own_target": bool(is_own_target),
                    "steric_clash": steric_clash,
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
            # utils_gromacs.run_energy_minimization RAISES when EM fails. Catch
            # it so the failure is recorded as a structured EM_NOT_CONVERGED leg
            # (with Fmax/Epot) by the gate below, rather than collapsing into a
            # generic {"error": ...} that downstream cannot distinguish from a
            # missing file.
            try:
                run_energy_minimization(md_dir)
            except Exception as _eme:
                logger.error(f"    Energy minimisation raised: {_eme}")

        # ── BLOCKER 07: VERIFY EM converged before spending GPU-days on MD ──
        # 22 of 90 legs previously died here and the failure propagated as a
        # missing/garbage number rather than a refusal.
        em = _em_convergence(md_dir)
        if not em["ok"]:
            logger.error(f"    ENERGY MINIMISATION DID NOT CONVERGE "
                         f"({em['reason']}; Fmax={em['fmax']}, "
                         f"Epot={em['epot']}). Refusing to run "
                         f"{'own' if is_own_target else 'cross'} leg MD — a leg "
                         f"started from an unminimised system is not a "
                         f"measurement.")
            return {
                "time_ns": 0,
                "placement": placement_diag,
                "rmsd_mean_A": None, "rmsd_final_A": None,
                "rmsd_selffit_A": None,
                "rebound": None,
                "is_own_target": bool(is_own_target),
                "steric_clash": steric_clash,
                "em": em,
                "status": "EM_NOT_CONVERGED",
                "error": em["reason"],
            }
        logger.info(f"    EM converged (Fmax={em['fmax']}, Epot={em['epot']})")

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

        # ── 7. Binding readout ──────────────────────────────────────
        xtc = md_dir / "md.xtc"
        tpr = md_dir / "md.tpr"

        # PRIMARY: persistent-contact fraction (code/pipeline/utils_persistent_contacts.py,
        # the statistic PCSI* is built on). This is what decides `rebound`.
        contacts = _contact_metrics(md_dir,
                                    tag=f" [{Path(output_dir).name}]")

        # DIAGNOSTIC: fit-free, cavity-frame displacement.
        rmsd_mean, rmsd_final = _gmx_rmsd_nofit(tpr, xtc, md_dir,
                                                "rmsd_rebind_nofit.xvg")
        # DIAGNOSTIC (provenance only): the old self-fitted number, so the two
        # can be compared on the same trajectories. NEVER gates anything.
        rmsd_selffit, _ = _gmx_rmsd(tpr, xtc, md_dir, "rmsd_rebind.xvg")

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

        # ── The verdict comes from CONTACTS, not from RMSD ──────────
        # A leg counts as rebound iff the template retained at least
        # REBINDING_MIN_PERSISTENT_RESIDUES residues in contact with a monomer
        # for more than half the analysed frames. This is exactly pcsi_star's
        # binding condition (c) ("a snapshot counts as bound if its own leg
        # formed any persistent contact"), so Phase 5 and PCSI* can no longer
        # disagree about what "bound" means.
        min_k = int(_cfg("REBINDING_MIN_PERSISTENT_RESIDUES", 1))
        if contacts.get("available"):
            k = contacts["n_persistent_residues"]
            rebound = bool(k >= min_k)
            status = "REBOUND" if rebound else "ESCAPED"
        else:
            k = None
            rebound = None
            status = "NO_OBSERVABLE"

        hb_str = f", H-bonds={hbond_mean}" if hbond_mean is not None else ""
        ct_str = f", contacts={contact_mean}" if contact_mean is not None else ""
        dg_str = f", ΔG={mmpbsa_dG}" if mmpbsa_dG is not None else ""
        logger.info(
            f"    Result: persistent contacts k={k}"
            f"/{contacts.get('total_residues')} "
            f"(f={contacts.get('fraction_persistent')}) → {status}; "
            f"RMSD_nofit={rmsd_mean} Å, RMSD_selffit={rmsd_selffit} Å"
            f"{hb_str}{ct_str}{dg_str}")

        return {
            "time_ns": time_ns,
            "is_own_target": bool(is_own_target),
            "placement": placement_diag,
            "steric_clash": steric_clash,
            "em": em,
            # PRIMARY observable
            "contacts": contacts,
            "n_persistent_residues": k,
            "fraction_persistent": contacts.get("fraction_persistent"),
            "mean_contact_freq": contacts.get("mean_contact_freq"),
            "rebound": rebound,
            "status": status,
            # DIAGNOSTICS. rmsd_mean_A now carries the FIT-FREE number so old
            # readers get a displacement that can actually detect escape;
            # rmsd_selffit_A preserves the retired self-fitted statistic.
            "rmsd_mean_A": rmsd_mean,
            "rmsd_final_A": rmsd_final,
            "rmsd_selffit_A": rmsd_selffit,
            "rmsd_is_fit_free": True,
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
            try:
                run_energy_minimization(md_dir)
            except Exception as _eme:
                logger.error(f"    Removal test: EM raised: {_eme}")

        em = _em_convergence(md_dir)
        if not em["ok"]:
            logger.error(f"    Removal test: EM did not converge ({em['reason']}) "
                         f"— refusing to run the removal MD.")
            return {"status": "EM_NOT_CONVERGED", "em": em,
                    "escaped": None, "error": em["reason"]}

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

        # ── Analyse: does the template lose its cavity contacts? ──────────
        # THRESHOLD REMOVED. The verdict used to be
        #     escaped = mean_RMSD_last_quarter > REBINDING_RMSD_THRESHOLD (5.0 Å)
        # on the SELF-FITTED RMSD, whose observed range across every completed
        # leg was 2.04-3.87 Å — entirely below the threshold. Every target read
        # STUCK by construction, and the statistic could not have detected an
        # escape anyway (it fits the template onto itself first).
        # The verdict is now a contact verdict: the template has left the cavity
        # when it holds no persistent monomer contact in the last quarter of the
        # run. A template that never made contact in Q1 gives NO verdict — that
        # is a failed cavity, not a removable one.
        xtc = md_dir / "md.xtc"
        tpr = md_dir / "md.tpr"

        cq = _contact_quartiles(md_dir)
        escaped = None
        verdict_basis = "persistent_contacts_q1_vs_q4"
        if cq.get("available"):
            k1, k4 = cq["k_persistent_q1"], cq["k_persistent_q4"]
            if k1 == 0:
                status = ("UNDEFINED (template held no persistent contact even at "
                          "the start — nothing to remove; the cavity failed)")
            else:
                drop = float(_cfg("REBINDING_REMOVAL_CONTACT_DROP_FRAC", 0.25))
                escaped = bool(k4 == 0 or (k4 / k1) <= drop)
                status = ("REMOVABLE (contacts decayed "
                          f"{k1}→{k4} — moderate binding, good MIP)" if escaped
                          else f"STUCK (contacts held {k1}→{k4} — template "
                               f"removal difficult)")
        else:
            status = f"N/A ({cq.get('reason')})"

        # Fit-free displacement, kept as a diagnostic next to the verdict.
        rmsd_end, _ = _gmx_rmsd_nofit(tpr, xtc, md_dir, "rmsd_removal_nofit.xvg")
        rmsds = _parse_xvg_rmsd(md_dir / "rmsd_removal_nofit.xvg")
        rmsd_start = (round(float(np.mean(rmsds[:max(1, len(rmsds) // 4)])), 2)
                      if rmsds else None)

        logger.info(f"    Removal test: persistent contacts "
                    f"{cq.get('k_persistent_q1')}→{cq.get('k_persistent_q4')}, "
                    f"RMSD_nofit {rmsd_start}→{rmsd_end} Å → {status}")

        return {
            "escaped": escaped,
            "status": status,
            "verdict_basis": verdict_basis,
            "contacts_q1_q4": cq,
            # diagnostics — NOT the verdict
            "rmsd_start_A": rmsd_start,
            "rmsd_end_A": rmsd_end,
            "rmsd_is_fit_free": True,
            "em": em,
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

        # DERIVED, NOT ASSERTED — same defect and same fix as the primary
        # snapshot loop above: _run_rebinding_md signals failure by RETURNING
        # {"success": False, "error": …}, so an unconditional True here recorded
        # an all-failed dual-imprinting snapshot as a success.
        _dlegs = {k: v for k, v in dual_snap.items()
                  if k.startswith("rebind_") and isinstance(v, dict)}
        dual_snap["success"] = bool(_dlegs) and all(
            not v.get("error") and v.get("success", True) for v in _dlegs.values())
        dual_snap["legs_total"] = len(_dlegs)
        dual_snap["legs_failed"] = sorted(
            k for k, v in _dlegs.items()
            if v.get("error") or not v.get("success", True))
        if not dual_snap["success"]:
            logger.error(f"    snap{i} (dual): {len(dual_snap['legs_failed'])}/"
                         f"{len(_dlegs)} leg(s) FAILED: {dual_snap['legs_failed']}")
        dual_snapshot_results.append(dual_snap)

    # Analyze
    if dual_snapshot_results:
        dual_analysis = _analyze_rebinding_results(
            target, target_names, dual_snapshot_results, threshold=None)
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

def _analyze_rebinding_results(target, all_targets, snapshot_results, threshold=None):
    """Compute rebinding success rate, selectivity index, and statistics.

    OBSERVABLE CHANGED (BLOCKER — Phase 5 observable). The selectivity index is
    now built on the PERSISTENT-CONTACT FRACTION f, not on the self-fitted
    RMSD:
        SI = mean(f_own) / mean(f_cross)          (polarity unchanged:
                                                   >1.5 selective, 1.0-1.5 weak,
                                                   <=1.0 cross-reactive)
        D  = (f_own - f_cross) / (f_own + f_cross)  — pcsi_star.contrast, bounded
    `threshold` is accepted for signature compatibility and IGNORED: the
    per-leg `rebound` flag already carries the contact criterion, and the old
    RMSD threshold (5.0 Å) sat entirely outside the observed 2.04-3.87 Å range.
    """
    from scipy import stats

    if threshold is not None:
        logger.debug("_analyze_rebinding_results: RMSD threshold %r ignored — "
                     "verdicts come from persistent contacts", threshold)

    n_success = 0
    n_total = 0
    own_f = []                    # PRIMARY: persistent-contact fraction
    own_rmsds = []                # diagnostic: fit-free RMSD
    other_f = {t: [] for t in all_targets if t != target}
    other_rmsds = {t: [] for t in all_targets if t != target}
    # Per-snapshot H-bond and contact data
    own_hbonds = []
    other_hbonds = {t: [] for t in all_targets if t != target}
    own_contacts = []
    other_contacts = {t: [] for t in all_targets if t != target}
    # Size-exclusion: count snapshots where cross-protein was rejected by clash
    other_size_excluded = {t: 0 for t in all_targets if t != target}
    # Legs that never produced a usable observable (EM failure, build failure)
    own_failed = 0
    other_failed = {t: 0 for t in all_targets if t != target}

    def _f_of(leg):
        """Persistent-contact fraction of a leg, or None if it has no readout."""
        if not isinstance(leg, dict):
            return None
        if leg.get("size_excluded"):
            return 0.0          # physically rejected: zero contact by construction
        f = leg.get("fraction_persistent")
        return float(f) if f is not None else None

    # INTEGRATION FIX — this gate used to read `if not snap.get("success"): continue`,
    # which was harmless only because snap["success"] was an unconditional True.
    # Now that it is DERIVED from the legs (a snapshot is "successful" only if
    # every leg succeeded), that gate would throw away an entire snapshot — its
    # good own leg included — because one cross leg failed to build. That is a
    # silent shrinking of the sample, the exact failure class this audit exists
    # to remove.
    #
    # Per-LEG failure is already handled correctly below: _f_of() returns None
    # for a leg with no readout and the leg is counted in own_failed /
    # other_failed rather than scored as zero binding. So the only snapshot worth
    # skipping is one that produced no usable leg at all.
    n_partial = 0
    n_no_usable_leg = 0
    for snap in snapshot_results:
        _legs = {k: v for k, v in snap.items()
                 if k.startswith("rebind_") and isinstance(v, dict)}
        if not _legs:
            logger.error(f"  {target}: snapshot {snap.get('snapshot', '?')} has no "
                         f"rebinding legs at all — EXCLUDED from the analysis")
            n_no_usable_leg += 1
            continue

        # AN INFRASTRUCTURE FAILURE IS NOT A NON-REBINDING EVENT
        # (REVIEW FINDING 5). A snapshot whose every leg returned
        # {"success": False, "error": "BUILD_FAILED"} carries NO observable at
        # all. Counting it in n_total diluted the denominator, so
        # success_rate fell from 10/10 to 10/11 purely because a build
        # crashed — reporting a cavity as less effective because the software
        # failed. This contradicts the module's own stated contract ("legs
        # with no observable are EXCLUDED ... never scored as zero binding"),
        # which the per-leg path already honours via _f_of() -> None.
        #
        # Note this is NOT the same as the `success` flag: a snapshot with one
        # good own leg and one failed cross leg is PARTIAL and is kept, because
        # its surviving leg is a real measurement.
        if all(_f_of(v) is None for v in _legs.values()):
            logger.error(
                f"  {target}: snapshot {snap.get('snapshot', '?')} produced NO "
                f"usable observable on ANY leg "
                f"({sorted(k for k in _legs)}) — EXCLUDED from the analysis "
                f"rather than counted as a failure to rebind. Leg errors: "
                f"{ {k: v.get('error') for k, v in _legs.items()} }")
            n_no_usable_leg += 1
            continue

        if not snap.get("success", True):
            n_partial += 1          # kept, but its failed legs are counted below
        n_total += 1

        own = snap.get("rebind_own", {})
        f = _f_of(own)
        if f is not None:
            own_f.append(f)
        else:
            own_failed += 1
        if own.get("rebound"):
            n_success += 1
        rmsd = own.get("rmsd_mean_A")
        if rmsd is not None:
            own_rmsds.append(rmsd)
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
            if isinstance(other, dict) and other.get("size_excluded"):
                other_size_excluded[other_t] += 1
                continue
            fo = _f_of(other)
            if fo is not None:
                other_f[other_t].append(fo)
            else:
                other_failed[other_t] += 1
            r = other.get("rmsd_mean_A") if isinstance(other, dict) else None
            if r is not None:
                other_rmsds[other_t].append(r)
            if isinstance(other, dict) and other.get("hbond_mean") is not None:
                other_hbonds[other_t].append(other["hbond_mean"])
            if isinstance(other, dict) and other.get("contact_mean") is not None:
                other_contacts[other_t].append(other["contact_mean"])

    own_f_mean = float(np.mean(own_f)) if own_f else None
    own_mean = float(np.mean(own_rmsds)) if own_rmsds else None
    own_std = float(np.std(own_rmsds)) if len(own_rmsds) > 1 else None

    result = {
        "target": target,
        "observable": "persistent_contact_fraction",
        "observable_note": (
            "rebound / success_rate / selectivity_index are computed from the "
            "persistent-contact fraction (persistent_contacts_fast.py). RMSD "
            "fields are fit-free diagnostics and gate nothing."),
        "n_snapshots": n_total,
        # Snapshots ANALYSED but carrying at least one failed leg. Kept in the
        # sample (their surviving legs are real measurements); surfaced so a
        # reader can see the analysis rested on partial snapshots.
        "n_snapshots_partial": n_partial,
        # Snapshots EXCLUDED because no leg produced an observable at all
        # (build/EM failure on every leg). These are infrastructure failures,
        # not measurements of non-binding, so they are out of n_snapshots —
        # but they are reported so the loss is visible rather than silent.
        "n_snapshots_no_usable_leg": n_no_usable_leg,
        "n_snapshots_submitted": len(snapshot_results),
        "n_rebound": n_success,
        "success_rate": f"{n_success}/{n_total}" if n_total > 0 else "0/0",
        "own_fraction_persistent_mean": round(own_f_mean, 4) if own_f_mean is not None else None,
        "own_fraction_persistent_std": (round(float(np.std(own_f)), 4)
                                        if len(own_f) > 1 else None),
        "n_own_legs_with_observable": len(own_f),
        "n_own_legs_failed": own_failed,
        "own_rmsd_mean": round(own_mean, 2) if own_mean else None,
        "own_rmsd_std": round(own_std, 2) if own_std else None,
        "own_hbond_mean": round(float(np.mean(own_hbonds)), 1) if own_hbonds else None,
        "own_contact_mean": round(float(np.mean(own_contacts)), 1) if own_contacts else None,
        "snapshots": snapshot_results,
        "selectivity": {},
    }

    # Selectivity Index (SI) and statistical tests for each other target
    for other_t, fvals in other_f.items():
        if not fvals:
            continue
        other_f_mean = float(np.mean(fvals))
        rmsds = other_rmsds.get(other_t, [])
        other_mean = float(np.mean(rmsds)) if rmsds else None
        other_std = float(np.std(rmsds)) if len(rmsds) > 1 else None

        # Selectivity Index on the CONTACT observable.
        # SI > 1.5 = selective, 1.0-1.5 = weak, <= 1.0 = cross-reactive.
        if own_f_mean is None:
            si = None
        elif other_f_mean > 0:
            si = round(own_f_mean / other_f_mean, 2)
        elif own_f_mean > 0:
            si = None            # cross target formed NO persistent contact
        else:
            si = None            # neither bound: undefined, not "selective"

        # Bounded contrast (pcsi_star.contrast) — defined where the ratio is not
        try:
            _ps = _import_pcsi_star()
            contrast_D = _ps.contrast(own_f_mean, other_f_mean)
        except Exception:
            den = (own_f_mean or 0.0) + other_f_mean
            contrast_D = ((own_f_mean - other_f_mean) / den) if den else None

        # Welch's t-test on the contact fractions (own vs cross)
        p_value = None
        if len(own_f) >= 2 and len(fvals) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(own_f, fvals, equal_var=False)
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
        elif own_f_mean and other_f_mean == 0:
            sel_label = "selective"       # own bound, cross formed no contact
        else:
            sel_label = "N/A"

        sel_entry = {
            "own_fraction_persistent_mean": (round(own_f_mean, 4)
                                             if own_f_mean is not None else None),
            "other_fraction_persistent_mean": round(other_f_mean, 4),
            "contrast_D": (round(contrast_D, 4) if contrast_D is not None else None),
            "n_cross_legs_with_observable": len(fvals),
            "n_cross_legs_failed": other_failed.get(other_t, 0),
            "other_rmsd_mean": round(other_mean, 2) if other_mean is not None else None,
            "other_rmsd_std": round(other_std, 2) if other_std else None,
            "selectivity_index": si,
            "selectivity_basis": "persistent_contact_fraction",
            "selectivity_label": sel_label,
            "p_value": p_value,
            "significant": p_value < 0.05 if p_value is not None else None,
        }

        # A6: Bootstrap 95% CI for SI — on the contact observable.
        # bootstrap_selectivity_index(own, other) was written for RMSD, where
        # SI = mean(other)/mean(own). Contact SI is mean(own)/mean(cross), so
        # the argument order is SWAPPED here to keep the same ratio orientation.
        try:
            from .utils_analysis import bootstrap_selectivity_index
            from .config import BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI
            boot = bootstrap_selectivity_index(
                fvals, own_f,
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
        if other_mean is not None:
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
        if data.get("error"):
            logger.info(f"  ERROR: {data['error']}")
            continue
        own_std = data.get('own_rmsd_std')
        std_str = f" ± {own_std}" if own_std else ""
        logger.info(f"  Rebinding (persistent contacts): "
                    f"{data.get('success_rate', 'N/A')}")
        logger.info(f"  Own persistent-contact fraction: "
                    f"{data.get('own_fraction_persistent_mean', 'N/A')} "
                    f"± {data.get('own_fraction_persistent_std')}")
        if data.get("n_own_legs_failed"):
            logger.info(f"  Own legs WITHOUT an observable (build/EM failure): "
                        f"{data['n_own_legs_failed']}")
        logger.info(f"  Own RMSD (fit-free, diagnostic): "
                    f"{data.get('own_rmsd_mean', 'N/A')}{std_str} Å")

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
                f"  vs {other_t}: f={s.get('other_fraction_persistent_mean')}  "
                f"SI={si} D={s.get('contrast_D')} [{label}]{p_str}")
            if s.get("n_cross_legs_failed"):
                logger.info(f"      ({s['n_cross_legs_failed']} cross legs had "
                            f"no observable and were EXCLUDED, not counted as "
                            f"non-binding)")

    logger.info(f"{'='*60}")


# ── Re-analysis with PBC centering + Q4 ────────────────────────

# BLOCKER 11 — _ensure_centered_xtc() WAS REMOVED HERE (integration pass).
#
# It wrote a full-size md_centered.xtc duplicate of md.xtc (measured 11.19 GB
# against 11.33 GB for one 350 ns leg) once per rebinding snapshot, and Phase 5
# runs REBINDING_N_SNAPSHOTS snapshots per target, so it multiplied faster than
# Phase 4's copy did.
#
# Its only consumer was _q4_rmsd_from_centered(), which no longer needs it: that
# function was rewritten to take its observable from persistent contacts and its
# RMSD from `gmx rms -fit none` with the monomer group centred by index, both
# read straight off md.xtc.  Removing the function rather than shrinking it is
# deliberate — leaving a duplicate-producer in the module invites a future caller
# to reintroduce the cost.  Phase 4's equivalent still centres (its occupancy
# analysis genuinely requires PBC-whole molecules) but now deletes the transient
# afterwards; see phase4_md_validation._discard_centered_trajectory.


def _q4_rmsd_from_centered(md_dir: Path) -> dict:
    """Per-leg re-analysis: contact observable + FIT-FREE Q4 RMSD.

    OBSERVABLE CHANGED. This used to run `gmx rms` with the default
    least-squares fit on a protein-centred trajectory and set
    `rebound_q4 = q4_mean < REBINDING_RMSD_THRESHOLD`. Both halves of that were
    wrong: the fit removes the escape, and the threshold sat outside the
    observed range. `rebound_q4` now comes from persistent contacts; the RMSD
    fields are fit-free and are convergence diagnostics only.
    """
    md_dir = Path(md_dir).resolve()
    tpr, xtc = md_dir / "md.tpr", md_dir / "md.xtc"
    if not (tpr.exists() and xtc.exists()):
        return {"error": "trajectory missing"}

    contacts = _contact_metrics(md_dir)
    min_k = int(_cfg("REBINDING_MIN_PERSISTENT_RESIDUES", 1))

    # Fit-free, monomer-centred displacement (the cavity is fixed in the box
    # frame because the monomers are position-restrained).
    ndx = _make_group_ndx(tpr, md_dir)
    rmsds = []
    if ndx is not None:
        _gmx_rmsd_nofit(tpr, xtc, md_dir, "rmsd_centered_nofit.xvg")
        rmsds = _parse_xvg_rmsd(md_dir / "rmsd_centered_nofit.xvg")

    out = {"observable": "persistent_contact_fraction",
           "contacts": contacts,
           "rmsd_is_fit_free": True}
    if contacts.get("available"):
        out["n_persistent_residues"] = contacts["n_persistent_residues"]
        out["fraction_persistent"] = contacts["fraction_persistent"]
        out["rebound_q4"] = bool(contacts["n_persistent_residues"] >= min_k)
    else:
        out["rebound_q4"] = None

    if len(rmsds) < 4:
        out["error_rmsd"] = "fit-free RMSD unavailable"
        return out
    arr = np.array(rmsds)
    n = len(arr)
    q1, q4 = arr[:n // 4], arr[3 * n // 4:]
    drift = float(q4.mean()) - float(q1.mean())
    out.update({
        "n_frames": n,
        "q1_mean_A": round(float(q1.mean()), 2),
        "q4_mean_A": round(float(q4.mean()), 2),
        "q4_std_A": round(float(q4.std()), 2),
        "rmsd_final_A": round(float(arr[-1]), 2),
        "drift_A": round(drift, 2),
        "converged": abs(drift) < 1.0 and float(q4.std()) < 1.0,
    })
    return out


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
        if phase5_ext.exists() and _has_snapshot_dirs(phase5_ext):
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
        for _rep, si, snap_dir in _iter_snapshot_dirs(target_dir):
            # replica is carried through so downstream statistics can group
            # correlated snapshots by their source Phase 4 trajectory.
            entry = {"snap_idx": si, "replica": _rep,
                     "snapshot_dir": snap_dir.name}
            for sub_name in ["rebind_own"] + [f"rebind_{t}" for t in cross_targets]:
                sub_dir = snap_dir / sub_name / "md"
                if sub_dir.exists():
                    diag = _q4_rmsd_from_centered(sub_dir)
                    entry[sub_name] = diag
            snap_data.append(entry)

        # Aggregate per cross target — on the CONTACT observable.
        # (This used to divide two self-fitted RMSDs. See _q4_rmsd_from_centered.)
        own_f = [s["rebind_own"]["fraction_persistent"] for s in snap_data
                 if "fraction_persistent" in s.get("rebind_own", {})]
        own_f_conv = [s["rebind_own"]["fraction_persistent"] for s in snap_data
                      if "fraction_persistent" in s.get("rebind_own", {})
                      and s["rebind_own"].get("converged")]
        own_q4 = [s["rebind_own"]["q4_mean_A"] for s in snap_data
                  if "rebind_own" in s and "q4_mean_A" in s["rebind_own"]]
        own_q4_conv = [s["rebind_own"]["q4_mean_A"] for s in snap_data
                       if "rebind_own" in s and s["rebind_own"].get("converged")]

        def _si(own_vals, cross_vals):
            """SI = mean(f_own)/mean(f_cross); None when undefined."""
            if not own_vals or not cross_vals:
                return None
            mo, mc = float(np.mean(own_vals)), float(np.mean(cross_vals))
            return round(mo / mc, 2) if mc > 0 else None

        sel = {}
        for ot in cross_targets:
            key = f"rebind_{ot}"
            cross_f = [s[key]["fraction_persistent"] for s in snap_data
                       if "fraction_persistent" in s.get(key, {})]
            cross_f_conv = [s[key]["fraction_persistent"] for s in snap_data
                            if "fraction_persistent" in s.get(key, {})
                            and s[key].get("converged")]
            cross_q4 = [s[key]["q4_mean_A"] for s in snap_data
                        if key in s and "q4_mean_A" in s[key]]
            cross_q4_conv = [s[key]["q4_mean_A"] for s in snap_data
                             if key in s and s[key].get("converged")]

            si_all = _si(own_f, cross_f)
            si_conv = _si(own_f_conv, cross_f_conv)
            # t-tests on the CONTACT observable (own vs cross), not on RMSD
            try:
                p_all = (round(float(ttest_ind(own_f, cross_f,
                                               equal_var=False).pvalue), 4)
                         if len(own_f) >= 3 and len(cross_f) >= 3 else None)
            except Exception:
                p_all = None
            try:
                p_conv = (round(float(ttest_ind(own_f_conv, cross_f_conv,
                                                equal_var=False).pvalue), 4)
                          if len(own_f_conv) >= 3 and len(cross_f_conv) >= 3 else None)
            except Exception:
                p_conv = None

            sel[ot] = {
                "basis": "persistent_contact_fraction",
                "n_cross": len(cross_f),
                "n_cross_converged": len(cross_f_conv),
                "cross_fraction_persistent_mean": (round(float(np.mean(cross_f)), 4)
                                                   if cross_f else None),
                "cross_fraction_persistent_std": (round(float(np.std(cross_f)), 4)
                                                  if cross_f else None),
                # fit-free RMSD, diagnostic only
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
                    # SI is undefined when the cross target formed NO persistent
                    # contact; that is the selective endpoint, not "n/a".
                    else "selective" if (si_all is None and own_f and cross_f
                                         and float(np.mean(own_f)) > 0
                                         and float(np.mean(cross_f)) == 0)
                    else "n/a"
                ),
            }

        summary[target] = {
            "observable": "persistent_contact_fraction",
            "observable_note": ("SI and rebound_q4 come from persistent contacts; "
                                "q4 RMSD fields are fit-free convergence "
                                "diagnostics and gate nothing."),
            "n_total": len(snap_data),
            "n_own_converged": len(own_q4_conv),
            "n_rebound_q4": sum(1 for s in snap_data
                                if s.get("rebind_own", {}).get("rebound_q4")),
            "own_fraction_persistent_mean": (round(float(np.mean(own_f)), 4)
                                             if own_f else None),
            "own_fraction_persistent_std": (round(float(np.std(own_f)), 4)
                                            if own_f else None),
            "own_q4_mean": round(float(np.mean(own_q4)), 2) if own_q4 else None,
            "own_q4_std": round(float(np.std(own_q4)), 2) if own_q4 else None,
            "own_q4_converged_only_mean": (
                round(float(np.mean(own_q4_conv)), 2) if own_q4_conv else None),
            "selectivity": sel,
            "snapshots": snap_data,
        }

        s = summary[target]
        logger.info(f"\n=== {target} (contact observable + fit-free Q4 RMSD) ===")
        logger.info(f"  N total: {s['n_total']}, own converged: {s['n_own_converged']}")
        logger.info(f"  Own persistent-contact fraction: "
                    f"{s['own_fraction_persistent_mean']} ± "
                    f"{s['own_fraction_persistent_std']} "
                    f"(rebound: {s['n_rebound_q4']}/{s['n_total']})")
        logger.info(f"  Own Q4 RMSD (fit-free, diagnostic): "
                    f"{s['own_q4_mean']} ± {s['own_q4_std']} Å")
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
    """Check if the RMSD trajectory shows drift (Q1→Q4 difference) > threshold.

    Prefers the FIT-FREE series (rmsd_rebind_nofit.xvg). The self-fitted
    rmsd_rebind.xvg is only used if no fit-free series exists, and the choice is
    reported in `source` — a convergence verdict taken from a statistic that
    cannot see rigid-body motion is not a convergence verdict.
    """
    md = Path(rebind_md_dir)
    nofit = md / "rmsd_rebind_nofit.xvg"
    selffit = md / "rmsd_rebind.xvg"
    if nofit.exists():
        xvg, source = nofit, "fit_free"
    elif selffit.exists():
        xvg, source = selffit, "self_fitted_LEGACY"
        logger.warning(f"  {md}: no fit-free RMSD series; falling back to the "
                       f"retired self-fitted one for drift detection only")
    else:
        return {"available": False}
    rmsds = _parse_xvg_rmsd(xvg)
    if len(rmsds) < 100:
        return {"available": False, "source": source}
    rmsds = np.array(rmsds)
    n = len(rmsds)
    q1 = float(rmsds[:n // 4].mean())
    q4 = float(rmsds[3 * n // 4:].mean())
    q4_std = float(rmsds[3 * n // 4:].std())
    drift = q4 - q1
    return {
        "available": True,
        "source": source,
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
        if phase5_ext.exists() and _has_snapshot_dirs(phase5_ext):
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
        for _rep, _si, snap_dir in _iter_snapshot_dirs(target_dir):
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
        if phase5_ext.exists() and _has_snapshot_dirs(phase5_ext):
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
        # Phase 4 no longer writes <target>/<pc>/md — it writes rep_<i>/md per
        # replica. Resolve through the same helper run_phase6 uses; only the
        # first usable directory is needed here (it is read for *.itp and
        # md.tpr, which are identical across replicas of one composition).
        _p4_dirs = _resolve_p4_md_dirs(target, best_pc, p4.get(best_pc, {}))
        if not _p4_dirs:
            logger.warning(f"[{target}] no usable Phase 4 md dir for {best_pc}")
            continue
        p4_md_dir = _p4_dirs[0][1]

        target_summary = []
        for _rep_src, snap_idx, snap_dir in _iter_snapshot_dirs(output_dir / target):
            cavity_gro = snap_dir / "frame.gro"
            cavity_top = snap_dir / "topol.top"
            if not (cavity_gro.exists() and cavity_top.exists()):
                continue

            logger.info(f"\n--- {target} snap_{snap_idx} multi-restart ---")
            # source_replica = the PHASE 4 trajectory this snapshot came from;
            # "rep" below is the multi-restart index, a different axis entirely.
            snap_results = {"snap_idx": snap_idx, "source_replica": _rep_src,
                            "snapshot_dir": snap_dir.name,
                            "reps": [{"rep": 0, "source": str(snap_dir)}]}

            for rep in range(1, n_reps):
                rep_dir = (multi_dir / target /
                           f"rep{_rep_src}_snapshot_{snap_idx}" / f"rep_{rep}")
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
