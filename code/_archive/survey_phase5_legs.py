"""Enumerate the Phase 5 rebinding legs and survey every trajectory.

Read-only with respect to the phase tree. Writes ONE json into this
experiment's reports dir (default: <results_root>/reports/phase5_leg_survey.json).
Classifies every expected leg as: present / size_excluded / md_failed / missing.
"""
import argparse, json, sys, warnings
import re
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent

# REVIEW FINDING 3: derived from the experiment-aware config rather than
# hardcoded to the CD tree, for the same reason as run_pcsi_star.py. Under CD
# these resolve to exactly the previous hardcoded values.
from pipeline.config import OUTPUT_DIRS as _OUTPUT_DIRS, TARGETS as _CFG_TARGETS

TARGETS = list(_CFG_TARGETS)
PHASE5_DIR = Path(_OUTPUT_DIRS["phase5"])


_SNAP_DIR_RE = re.compile(r"^(?:rep(?P<rep>\d+)_)?snapshot_(?P<idx>\d+)$")


def _snapshot_dirs(cav_dir):
    """[(replica, index, dir)] understanding both Phase 5 layouts.

    Legacy: snapshot_<i>, all carved from ONE Phase 4 trajectory.
    Current: rep<r>_snapshot_<i>, spread across Phase 4 replicas so the source
    trajectory stays recoverable (snapshots sharing a replica are correlated).
    Iterating range(10) finds nothing under the current layout.
    """
    if not Path(cav_dir).is_dir():
        return []
    out = []
    for d in Path(cav_dir).iterdir():
        m = _SNAP_DIR_RE.match(d.name) if d.is_dir() else None
        if m:
            out.append((int(m.group("rep") or 0), int(m.group("idx")), d))
    return sorted(out, key=lambda t: (t[0], t[1]))


def expected_legs():
    """Yield (cavity, mode, snapshot, ligand, md_dir) for every discovered leg."""
    for cav in TARGETS:
        others = [o for o in TARGETS if o != cav]
        for _replica, snap, base in _snapshot_dirs(PHASE5_DIR / cav):
            modes = [("main", base)]
            if cav == "CD63":
                modes.append(("dual", base / "dual_imprinting"))
            for mode, prefix in modes:
                for lig, sub in [(cav, "rebind_own")] + [(o, f"rebind_{o}") for o in others]:
                    yield cav, mode, snap, lig, prefix / sub / "md"


def classify(md_dir):
    """Why is md.xtc absent? Distinguish size-exclusion from a crashed MD run."""
    if (md_dir / "md.xtc").exists() and (md_dir / "md.tpr").exists():
        return "present"
    if not md_dir.exists():
        return "missing_dir"
    has_em = (md_dir / "em.gro").exists()
    nvt_log = md_dir / "nvt.log"
    nvt_xtc = md_dir / "nvt.xtc"
    if not has_em and (md_dir / "rebind_system.gro").exists():
        # skipped before energy minimisation -> steric pre-screen rejected it
        return "size_excluded"
    if nvt_log.exists() and nvt_xtc.exists() and nvt_xtc.stat().st_size == 0:
        return "md_failed_nvt"
    if has_em:
        return "md_incomplete"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    # CHANGED DEFAULT (REVIEW FINDING 2, same class): was
    # ROOT/"reports_v2/phase5_leg_survey.json", which is the FROZEN PCSI*
    # recompute directory and already holds a file of that name. Writing there
    # by default overwrites a protected artifact.
    ap.add_argument("--out",
                    default=str(Path(_OUTPUT_DIRS["reports"]) /
                                "phase5_leg_survey.json"))
    ap.add_argument("--no-open", action="store_true",
                    help="file-existence classification only, do not open trajectories")
    a = ap.parse_args()

    from persistent_contacts_fast import survey_leg, enable_readonly_mode
    enable_readonly_mode()

    rows = []
    for cav, mode, snap, lig, md in expected_legs():
        st = classify(md)
        row = {"cavity": cav, "mode": mode, "snapshot": snap, "ligand": lig,
               "is_own": lig == cav, "md_dir": str(md.relative_to(ROOT)),
               "status": st}
        if st == "present" and not a.no_open:
            row.update(survey_leg(md / "md.xtc", md / "md.tpr"))
        rows.append(row)
        print(f"{cav:5s} {mode:4s} snap{snap} {'own' if lig==cav else lig:5s} "
              f"{st:15s} " + (f"frames={row.get('n_frames')} dt={row.get('dt_ps')} "
                              f"ns={row.get('total_ns')} prot_res={row.get('n_protein_residues')} "
                              f"ok={row.get('ok')}" if st == "present" and not a.no_open else ""),
              flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}  ({len(rows)} expected legs)")


if __name__ == "__main__":
    main()
