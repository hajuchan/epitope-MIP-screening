#!/usr/bin/env python3
"""
Per-phase verification script — Level 1/2/3 checks.

Usage:
    python verify_phase.py 1              # verify Phase 1
    python verify_phase.py 4 --targets CD63 CD81
    python verify_phase.py all            # verify every phase that has results

Exit code:
    0  every enforced check passed  (report["blocked"] is False)
    1  at least one enforced check failed (report["blocked"] is True)
    2  usage error

Saves:
    <results>/reports/phaseN_verification.json   (machine-readable)
    <results>/reports/phaseN_summary.md          (human-readable)

──────────────────────────────────────────────────────────────────────────
BLOCKER 08 — WHAT CHANGED AND WHY (2026-08, audit fix)
──────────────────────────────────────────────────────────────────────────
Before this rewrite the file could not fail:

  * verify_phase3/4/6 ended with an UNCONDITIONAL literal
        report["decision"] = "proceed_to_phase_4" / "_5" / "all_phases_complete"
    so the computed level1/2/3 dicts were decoration. A Phase 4 in which all
    three legs were non-converged (window differences up to 73%) returned
    issues: [] and proceed_to_phase_5.
  * No threshold defined anywhere in the file was ever compared against.
  * verify_phase5 computed  pcsi = own / max_cross if max_cross > 0 else inf
    and then  pcsi_pass = pcsi > 1.2 — so a target with NO measurable
    cross-target arm at all scored +inf and PASSED. "We never measured the
    off-target" was reported as "perfectly selective".
  * main() always returned 0 and run_pipeline.py never called it.

Now: EVERY decision is DERIVED from the computed report by _finalize(), which
blocks whenever report["issues"] is non-empty. Nothing may assign
report["decision"] directly. The enforced (blocking) criteria are listed in
BLOCKING_CRITERIA below; anything advisory goes to report["warnings"], which
is printed but never blocks.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    np = None
    _NP_AVAILABLE = False

# OUTPUT_DIRS, not get_output_path(): the latter mkdirs, so merely IMPORTING
# this module used to conjure an empty results/ tree — including when the
# caller is about to point --results-root somewhere else entirely.
# NOTE: this file used to live at code/verify_phase.py and inserted its parent
# into sys.path so `from .config import ...` worked. Now that it lives
# INSIDE the pipeline package, the same import resolves via normal package
# path — no sys.path hack required.
from .config import OUTPUT_DIRS, TARGETS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("verify")

REPORTS_DIR = Path(OUTPUT_DIRS["reports"])
RESULTS_ROOT = Path(OUTPUT_DIRS["phase1"]).parent

# Phase 3's output contract. Imported from the producer so a schema bump cannot
# leave this verifier checking a stale string; the literal is only a fallback
# for a tree whose phase3_mmsd module is unavailable.
try:
    from .phase3_mmsd import _PHASE3_SCHEMA as _PHASE3_EXPECTED_SCHEMA
except Exception:                                    # pragma: no cover
    _PHASE3_EXPECTED_SCHEMA = "phase3/v2-blocker05"

# Phase 5 snapshot directory names, BOTH layouts: the legacy snapshot_<i> (all
# snapshots carved from one Phase 4 trajectory) and the current
# rep<r>_snapshot_<i> (spread across Phase 4 replicas).
import re as _re
_SNAPSHOT_DIR_RE = _re.compile(r"^(?:rep(?P<rep>\d+)_)?snapshot_(?P<idx>\d+)$")


BLOCKING_CRITERIA = """\
Enforced (a violation BLOCKS the pipeline):
  all    : the phase result JSON is missing or unparseable
  all    : a requested target is absent from level1 (never verified)
  1      : the target's phase 1 result carries an "error" (every structure for
           it failed to build) — note an ABSENT path key now reads as missing,
           because Path("") is PosixPath('.') and used to read as present
  1      : head/ECL2/receptor file recorded in the JSON does not exist on disk
  1      : true PBC split (span > 0.95 x box) or a centering check that errored
  1      : stability-MD mean RMSD > EPITOPE_RMSD_THRESHOLD
  1      : conformer sampling degraded away from ENSEMBLE_CLUSTERING_METHOD —
           including when it degraded for EVERY target, which the symmetry and
           exchangeability checks pass by construction
  2      : zero monomers docked, or zero monomers filtered through to Phase 3
  2      : binding energies outside the physically plausible range (-15, 0) —
           BOTH ends: a non-negative BE is not a docking result, and 0.0 is
           the retired parse-failure default
  2      : decoy enrichment factor <= 1 (real monomers score no better than
           decoys, so the docking carries no discriminating signal)
  3      : target reported an error, or produced zero top PCs
  3      : NSGA-II selected but the Pareto front is empty
  3      : the top PC is not polymerization-compatible
  3      : any Pareto entry with mmsd_result=null (BLOCKER 05b — the front was
           never joined to its MMSD results, so the pick was one scalar)
  3      : smd_sum == 0 for every record (BLOCKER 05a — the Phase 2 reference
           never resolved), or synergy == true for every record
  3      : schema != the Phase 3 output contract (pre-BLOCKER-05 result)
  3      : a pareto_front entry with no "objectives" object (a malformed front
           cannot have been ranked; this used to be an uncaught KeyError that
           aborted the verifier before it emitted any Phase 3 finding)
  3      : the SELECTED composition had every cross-target MMSD fail, so its
           selectivity is unmeasured (penalty None, not 0.0)
  4      : zero PCs, missing MD results, a leg carrying success=False/error
  4      : every monomer recorded zero contacts, or every EBN is 0 — a failed
           occupancy analysis (typically an empty residue->monomer mapping),
           not a measurement of non-binding
  4      : convergence absent, or convergence.converged == False
  4      : a leg with accepted == False (no replica met the acceptance criteria)
  4      : any leg listed in phase4_quality_report.json's rejected_legs
  4      : MM-GBSA missing, carrying an "error", or without delta_total_kcal
  4      : cross-reactivity leg carrying an error
  4      : true PBC split or a centering check that errored
  5      : zero rebound snapshots
  5      : UNMEASURED CROSS-TARGET arm — a cavity whose expected off-target
           legs were neither measured nor physically rejected is not a
           selectivity result and can never PASS. Size-excluded off-targets
           (selectivity_label='size-excluded', clash-count pre-EM rejection)
           are NOT unmeasured — they are mechanism-level selectivity evidence
           (Hoshino 2008); a cavity with all cross-targets size-excluded
           PASSes as size/shape exclusion selectivity.
  5      : own template rebinding null (own_n == 0) — the cavity does not
           bind the ligand it was imprinted against, so no selectivity ratio
           has a valid numerator regardless of what happens on cross-targets
  5      : no selectivity metric computed at all — EXCEPT in a single-target
           experiment (BSA), where PCSI* is undefined for want of an off-target
           and demanding it would be an unsatisfiable block; there the gate
           degrades to "did the cavity rebind?" and warns
  5      : no target passes the selectivity gate
  6      : recipe/NIP/per-target file missing, or not polymerization-compatible
  6      : Phase 5 cleared NO target, so the Phase 6 scope is empty — an empty
           scope used to finalise as "all_phases_complete" with exit 0
"""


# ── decision machinery ──────────────────────────────────────────
def _new_report(phase: int, targets) -> dict:
    return {
        "phase": phase,
        "targets_requested": list(targets),
        "level1": {}, "level2": {}, "level3": {},
        "issues": [], "warnings": [],
    }


def _finalize(report: dict, ok_decision: str) -> dict:
    """THE only place a decision is assigned.

    A decision is a FUNCTION of report["issues"]. Do not assign
    report["decision"] anywhere else — that is exactly the bug this rewrite
    exists to remove.
    """
    blocked = bool(report.get("issues"))
    report["blocked"] = blocked
    report["n_issues"] = len(report.get("issues", []))
    report["decision"] = (
        f"BLOCKED_phase{report['phase']}" if blocked else ok_decision)
    return report


def _path_exists(value) -> bool:
    """True only when `value` is a NON-EMPTY path string that exists on disk.

    Do NOT write `Path(str(d.get(k, ""))).exists()`. `Path("")` is
    `PosixPath('.')` — the current directory — which ALWAYS exists, so an
    ABSENT key reads as a present file. That is how a phase1_results.json in
    which every target crashed (`{"CD63": {"error": ...}}`, written at
    phase1_epitope_prep.py:62) scored head_pdb_exists=True and sailed through
    verification with decision=proceed_to_phase_2.
    """
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    return Path(s).exists()


def _require_targets_present(report: dict, targets) -> None:
    """Block on a requested target that never made it into level1.

    A target that silently vanished (crashed, skipped, filtered out upstream)
    must not be readable as "nothing to report, therefore fine".
    """
    for t in targets:
        if t not in report["level1"]:
            report["issues"].append(
                f"{t}: requested but ABSENT from level1 — the phase produced no "
                f"verifiable record for it")


def _load_json(path: Path, report: dict, label: str):
    """Load a phase JSON; append an issue and return None on any failure."""
    if not path.exists():
        report["issues"].append(f"{label} missing: {path}")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        report["issues"].append(f"{label} unparseable ({path}): {e}")
        return None


def _requested_targets(targets=None):
    """Targets this verification must account for.

    Priority: explicit argument > run manifest written by run_pipeline >
    every target in TARGETS. The manifest matters because
    `--target CD63` must not be judged against a 3-target contract.
    """
    if targets:
        return list(targets)
    manifest = REPORTS_DIR / "run_manifest.json"
    if manifest.is_file():
        try:
            with open(manifest) as f:
                mt = json.load(f).get("targets")
            if mt:
                return list(mt)
        except Exception:
            pass
    return list(TARGETS)


def _check_md_centering(gro_path, apply_trjconv: bool = True):
    """Level 3: Check protein centered + no real PBC split.

    Uses trjconv -pbc mol -center to first fix any wrapping artifacts,
    then checks: (a) COM near box center, (b) span much less than box.

    PBC split criterion is span > 0.95 × box (very strict — only when
    protein wraps across full box). Tight-box artifacts (span 0.7-0.9×box)
    are NOT flagged as splits since they're common for elongated proteins.
    """
    try:
        import MDAnalysis as mda
        import numpy as np
        import subprocess
        from .config import GMX_BIN

        gro_path = Path(gro_path)
        # Optional: apply trjconv -pbc mol -center to get true geometry
        if apply_trjconv:
            tpr = gro_path.with_suffix('.tpr')
            if not tpr.exists():
                tpr = gro_path.parent / "md.tpr"
            centered = gro_path.parent / f"_verify_centered.gro"
            try:
                proc = subprocess.run(
                    [GMX_BIN, "trjconv", "-s", str(tpr), "-f", str(gro_path),
                     "-o", str(centered), "-pbc", "mol", "-center"],
                    input="1\n0\n", capture_output=True, text=True, timeout=120,
                )
                if centered.exists():
                    gro_path = centered
            except Exception:
                pass

        u = mda.Universe(str(gro_path))
        prot = u.select_atoms("protein")
        if len(prot) == 0:
            return {"error": "no protein"}
        box = u.dimensions[:3] / 10  # nm
        com = prot.center_of_mass() / 10
        span = (prot.positions.max(axis=0) - prot.positions.min(axis=0)) / 10
        offset = com - box / 2

        # Cleanup
        cf = Path(str(gro_path.parent / "_verify_centered.gro"))
        if cf.exists() and cf != gro_path:
            try: cf.unlink()
            except Exception: pass

        return {
            "box_nm": [round(float(b), 2) for b in box],
            "protein_span_nm": [round(float(s), 2) for s in span],
            "protein_com_offset_nm": [round(float(o), 2) for o in offset],
            "span_to_box_ratio_max": round(float((span / box).max()), 2),
            "pbc_split": bool(any(span > box * 0.95)),  # strict: only true wrapping
            "tight_box": bool(any(span > box * 0.7) and not any(span > box * 0.95)),
            "centered": bool(np.linalg.norm(offset) < 1.0),
            "com_distance_nm": round(float(np.linalg.norm(offset)), 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _judge_centering(report: dict, label: str, cen: dict) -> None:
    """Turn a _check_md_centering() dict into issues/warnings.

    An 'error' BLOCKS: a check that did not run is not a check that passed.
    """
    if not cen:
        return
    if cen.get("error"):
        report["issues"].append(
            f"{label}: PBC/centering check could not run ({cen['error']}) — "
            f"unverified geometry")
        return
    if cen.get("pbc_split"):
        report["issues"].append(
            f"{label}: protein is PBC-SPLIT "
            f"(span/box = {cen.get('span_to_box_ratio_max')})")
    if cen.get("tight_box"):
        report["warnings"].append(
            f"{label}: tight box (span/box = {cen.get('span_to_box_ratio_max')})")
    if cen.get("centered") is False:
        report["warnings"].append(
            f"{label}: protein COM {cen.get('com_distance_nm')} nm off box centre")


# ── Phase 1 ─────────────────────────────────────────────────────
def verify_phase1(targets=None):
    """Phase 1: Epitope preparation"""
    targets = _requested_targets(targets)
    p = RESULTS_ROOT / "phase1"
    report = _new_report(1, targets)

    from .config import EPITOPE_RMSD_THRESHOLD

    data = _load_json(p / "phase1_results.json", report, "phase1_results.json")
    if data is None:
        return _finalize(report, "proceed_to_phase_2")

    for t in targets:
        if t not in data:
            continue                       # caught by _require_targets_present
        tr = data[t]

        # A per-target exception is recorded as {"error": str(e)} and NOTHING
        # else (phase1_epitope_prep.py:62). That shape has no head_pdb key at
        # all, so it must be caught here before the file checks — otherwise the
        # most common partial-failure shape is the one the verifier misses.
        if isinstance(tr, dict) and tr.get("error"):
            report["issues"].append(
                f"{t}: phase 1 recorded an error and produced no usable "
                f"structures: {tr['error']}")
        if not isinstance(tr, dict):
            report["issues"].append(
                f"{t}: phase 1 result is {type(tr).__name__}, not an object")
            continue

        files = {
            # _path_exists, NOT Path(str(...)) — see its docstring. An absent
            # key must read False, and Path("") is PosixPath('.').
            "head_pdb_exists": _path_exists(tr.get("head_pdb")),
            "ecl2_pdb_exists": _path_exists(tr.get("epitope_pdb")),
            "receptor_pdbqt_exists": _path_exists(tr.get("receptor_pdbqt")),
        }
        report["level1"][t] = files
        for k, ok in files.items():
            if not ok:
                report["issues"].append(f"{t}: {k} is False — structure not on disk")

        # L2: A1 multi-epitope
        report["level2"][t] = {
            "A1_selected_head": tr.get("selected_head_candidate"),
            "A1_selected_range": tr.get("selected_head_range"),
            "A1_candidates_evaluated": len(tr.get("epitope_candidates", [])),
            "B1_sasa_mean": tr.get("epitope_quality", {}).get("sasa", {}).get("mean_sasa_A2"),
            "B1_accessible": tr.get("epitope_quality", {}).get("surface_accessible"),
            "B2_gravy": tr.get("epitope_quality", {}).get("gravy"),
            "B2_balance": tr.get("epitope_quality", {}).get("balance"),
        }
        if report["level2"][t]["B1_accessible"] is False:
            report["warnings"].append(f"{t}: epitope not surface-accessible (B1)")

        # L3: MD centering
        md_dir = p / t / "stability_md"
        if (md_dir / "md.gro").exists():
            cen = _check_md_centering(md_dir / "md.gro")
            report["level3"][t] = cen
            _judge_centering(report, f"{t} stability_md", cen)
            # Stability MD RMSD
            rmsd_xvg = md_dir / "rmsd.xvg"
            if rmsd_xvg.exists():
                try:
                    rmsds = []
                    for line in rmsd_xvg.read_text().split("\n"):
                        if line and not line.startswith(("#", "@")):
                            parts = line.split()
                            if len(parts) >= 2:
                                rmsds.append(float(parts[1]) * 10)
                    if rmsds:
                        mean_rmsd = round(sum(rmsds) / len(rmsds), 2)
                        report["level3"][t]["stability_rmsd_mean_A"] = mean_rmsd
                        report["level3"][t]["stability_rmsd_max_A"] = round(max(rmsds), 2)
                        # THRESHOLD NOW ENFORCED (was computed and ignored)
                        if mean_rmsd > EPITOPE_RMSD_THRESHOLD:
                            report["issues"].append(
                                f"{t}: stability-MD mean RMSD {mean_rmsd} A > "
                                f"EPITOPE_RMSD_THRESHOLD {EPITOPE_RMSD_THRESHOLD} A — "
                                f"epitope does not hold its conformation")
                except Exception as e:
                    report["warnings"].append(f"{t}: rmsd.xvg unreadable ({e})")
        else:
            report["warnings"].append(
                f"{t}: no stability MD (md.gro absent) — epitope conformational "
                f"stability is UNVERIFIED (--skip-md?)")

    # A SYMMETRIC DEGRADATION IS STILL A DEGRADATION (REVIEW FINDING 20).
    #
    # The ensemble-symmetry gate and _check_receptor_exchangeability both
    # compare targets to EACH OTHER, so when the configured conformer sampling
    # fails identically for all of them they both PASS and the run proceeds
    # with the headline A2 k-medoids feature inoperative — every target on
    # uniform-time sampling, which is exactly what A2 exists to replace.
    _methods = {}
    for t in targets:
        tr = data.get(t)
        if isinstance(tr, dict):
            samp = tr.get("ensemble_sampling") or {}
            if samp.get("method"):
                _methods[t] = samp["method"]
    if _methods:
        report["level2"].setdefault("_ensemble", {})["sampling_method"] = _methods
        degraded = {t: m for t, m in _methods.items()
                    if "failure" in m or m.startswith("uniform_after")}
        try:
            from .config import ENSEMBLE_CLUSTERING_METHOD as _cfg_method
        except Exception:
            _cfg_method = None
        if degraded and len(degraded) == len(_methods):
            report["issues"].append(
                f"EVERY target fell back from the configured conformer "
                f"sampling ({_cfg_method!r}) to {sorted(set(degraded.values()))} "
                f"— the ensemble is uniform-time sampled for all of them, so "
                f"the symmetry and exchangeability checks pass while the "
                f"feature they are guarding never ran")
        elif degraded:
            report["issues"].append(
                f"conformer sampling degraded for {sorted(degraded)} but not "
                f"all targets ({_methods}) — the ensembles are not comparable")

    _require_targets_present(report, targets)
    return _finalize(report, "proceed_to_phase_2")


# ── Phase 2 ─────────────────────────────────────────────────────
def verify_phase2(targets=None):
    targets = _requested_targets(targets)
    p = RESULTS_ROOT / "phase2"
    report = _new_report(2, targets)

    data = _load_json(p / "phase2_smd_results.json", report, "phase2 JSON")
    if data is None:
        return _finalize(report, "proceed_to_phase_3")

    for t in targets:
        be = data.get("be_matrix", {}).get(t, {})
        sel = data.get("selectivity", {}).get(t, {})
        decoy = data.get("decoy_baseline", {}).get(t, {}) if data.get("decoy_baseline") else {}
        ef = data.get("enrichment_factor", {}).get(t, {}) if data.get("enrichment_factor") else {}
        filt = data.get("filtered", {}).get(t, [])
        if not be and not filt:
            continue                       # caught by _require_targets_present
        report["level1"][t] = {
            "n_monomers_docked": len(be),
            "n_filtered_for_phase3": len(filt),
            "has_selectivity": len(sel) > 0,
        }
        if len(be) == 0:
            report["issues"].append(f"{t}: zero monomers docked")
        if len(filt) == 0:
            report["issues"].append(
                f"{t}: zero monomers passed the SMD filter — Phase 3 has no input")
        if len(sel) == 0:
            report["warnings"].append(f"{t}: no selectivity metric recorded")

        report["level2"][t] = {
            "B3_decoy_active": bool(decoy),
            "B3_enrichment_factor": ef.get("enrichment_factor"),
            "B3_enriched": ef.get("enriched"),
            # A3 pose clustering check (dlg parsing)
            "A3_some_monomers_have_clusters": any(
                "pose_clusters" in (data.get("dock_details", {}).get(t, {}).get(m, {}))
                for m in be
            ),
        }
        # AN UNENRICHED DOCKING CARRIES NO SIGNAL (REVIEW FINDING 16).
        # The decoy baseline exists to answer one question: do these scores
        # mean anything? EF <= 1 says the real monomers score no better than
        # chemically irrelevant decoys, i.e. the ranking Phase 3 is about to
        # optimise is noise. That is a blocking condition, not a warning.
        if ef.get("enriched") is False:
            report["issues"].append(
                f"{t}: decoy enrichment factor {ef.get('enrichment_factor')} "
                f"(<= 1) — real monomers score NO BETTER than decoys, so the "
                f"docking carries no discriminating signal and the Phase 3 "
                f"ranking built on it is not meaningful")
        elif ef.get("status") == "NOT_EVALUATED":
            report["warnings"].append(
                f"{t}: decoy enrichment NOT EVALUATED "
                f"({ef.get('reason')}) — the EF criterion must not be cited")

        # L3: BE range sanity — THRESHOLD NOW ENFORCED ON BOTH ENDS.
        #
        # This used to be `-15.0 < min(bes) < 0.0`: only the MINIMUM was
        # tested. BE_max was computed, published, and compared to nothing, so
        # a matrix containing 0.0 (the retired parse-failure default) and
        # physically impossible POSITIVE binding energies passed with
        # plausible_range=True and zero issues.
        bes = [v for v in be.values() if v is not None]
        if bes:
            lo, hi = min(bes), max(bes)
            too_strong = lo <= -15.0
            non_binding = hi >= 0.0
            plausible = not (too_strong or non_binding)
            nonneg = {m: v for m, v in be.items()
                      if isinstance(v, (int, float)) and v >= 0.0}
            report["level3"][t] = {
                "BE_min": round(lo, 2),
                "BE_max": round(hi, 2),
                "BE_mean": round(sum(bes) / len(bes), 2),
                "plausible_range": plausible,
                "n_non_negative_BE": len(nonneg),
            }
            if too_strong:
                report["issues"].append(
                    f"{t}: binding energy {lo:.2f} kcal/mol is below the "
                    f"plausible floor of -15 — implausibly strong for a small "
                    f"monomer, so the docking or the parameters are wrong")
            if non_binding:
                report["issues"].append(
                    f"{t}: {len(nonneg)} monomer(s) have a NON-NEGATIVE "
                    f"binding energy {nonneg} — 0.0 is the retired "
                    f"parse-failure default and a positive value is "
                    f"repulsive; neither is a docking result and both shift "
                    f"every rank below them")

    _require_targets_present(report, targets)
    return _finalize(report, "proceed_to_phase_3")


# ── Phase 3 ─────────────────────────────────────────────────────
def verify_phase3(targets=None):
    targets = _requested_targets(targets)
    p = RESULTS_ROOT / "phase3"
    report = _new_report(3, targets)

    from .config import is_polymerization_compatible, MMSD_OPTIMIZER

    data = _load_json(p / "phase3_mmsd_results.json", report, "phase3 JSON")
    if data is None:
        return _finalize(report, "proceed_to_phase_4")

    for t in targets:
        td = data.get(t, {})
        if not isinstance(td, dict) or not td:
            continue                       # caught by _require_targets_present
        if "error" in td:
            report["issues"].append(f"{t}: {td['error']}")
            continue
        method = td.get("method", "")
        pareto = td.get("pareto_front") or []
        top_pcs = td.get("top_pcs", []) or []
        report["level1"][t] = {
            "method": method,
            "n_evaluations": td.get("n_evaluations", 0),
            "n_pareto": len(pareto),
            "n_top_pcs": len(top_pcs),
        }
        if len(top_pcs) == 0:
            report["issues"].append(
                f"{t}: zero top PCs — Phase 4 has nothing to simulate")

        nsga2_active = "NSGA-II" in method

        # A MALFORMED FRONT MUST BLOCK, NOT CRASH (REVIEW FINDING 8).
        #
        # This was `p_["objectives"].get(...)`, which raised an uncaught
        # KeyError on any Pareto entry without an "objectives" key — aborting
        # the whole verifier BEFORE it could emit any of its four Phase 3
        # BLOCKING findings. A component whose entire job is to refuse bad
        # input must not fall over on bad input. Real HEAD-produced fronts do
        # carry "objectives", but a hand-edited, truncated or third-party
        # results file trips it.
        n_malformed = sum(1 for p_ in pareto
                          if not isinstance(p_, dict)
                          or not isinstance(p_.get("objectives"), dict))
        try:
            best_synth = max(
                [(p_.get("objectives") or {}).get("synthesizability_score", 0)
                 for p_ in pareto if isinstance(p_, dict)],
                default=0)
        except Exception as e:          # non-numeric objective values, etc.
            best_synth = None
            report["issues"].append(
                f"{t}: pareto_front objectives are unreadable ({e}) — the "
                f"front cannot have been ranked on the three objectives")

        report["level2"][t] = {
            "C2_nsga2_active": nsga2_active,
            "pareto_front_size": len(pareto),
            "n_pareto_malformed": n_malformed,
            "best_synth_score": best_synth,
        }
        if n_malformed:
            report["issues"].append(
                f"{t}: {n_malformed}/{len(pareto)} pareto_front entries are "
                f"MALFORMED (no 'objectives' object) — the front cannot be "
                f"ranked on affinity/selectivity/synthesizability, so the "
                f"composition Phase 4 consumes was not chosen by the "
                f"documented rule")
        # THRESHOLD NOW ENFORCED: nsga2 configured => a Pareto front must exist
        if str(MMSD_OPTIMIZER).lower() in ("nsga2", "nsga-ii") and not nsga2_active:
            report["warnings"].append(
                f"{t}: MMSD_OPTIMIZER={MMSD_OPTIMIZER} but method={method!r} "
                f"(fallback optimizer was used)")
        if nsga2_active and len(pareto) == 0:
            report["issues"].append(
                f"{t}: NSGA-II reported but the Pareto front is EMPTY")

        # ── BLOCKER 05 SIGNATURES ───────────────────────────────────
        # These three are exactly how BLOCKER 05 presented itself in the v1
        # output, and this verifier inspected pareto_front and top_pcs without
        # noticing any of them. Each is one line because Phase 3 now publishes
        # the fields specifically so the check is cheap.
        schema = td.get("schema")
        sel = td.get("pareto_selection") or {}
        report["level1"][t]["schema"] = schema
        report["level1"][t]["pareto_fallback_used"] = sel.get("fallback_used")

        # (1) Pareto entries carrying no mmsd_result: the join between the
        #     NSGA-II front and the MMSD results silently produced None for
        #     every entry, so the "Pareto" pick was argmin of one scalar.
        n_null_mmsd = sum(1 for e in pareto
                          if isinstance(e, dict) and e.get("mmsd_result") is None)
        if pareto and n_null_mmsd:
            report["issues"].append(
                f"{t}: {n_null_mmsd}/{len(pareto)} Pareto entries have "
                f"mmsd_result=null — the front cannot have been ranked on the "
                f"three objectives (BLOCKER 05b)")

        all_results = td.get("all_results") or []
        smd_vals = {r.get("smd_sum") for r in all_results if isinstance(r, dict)}

        # (2) smd_sum identically zero: a target-keyed BE matrix was passed
        #     into the monomer-keyed slot, so no SMD reference was ever found.
        # M5: `None` (missing key) must be treated as "did not observe a
        # non-zero", not "unknown, keep looking" — otherwise one stray record
        # with smd_sum=None hides that every OTHER record is 0. Drop None from
        # the value set before comparing.
        non_null_smd = smd_vals - {None}
        if all_results and non_null_smd and non_null_smd <= {0, 0.0}:
            n_null = sum(1 for r in all_results
                         if isinstance(r, dict) and r.get("smd_sum") is None)
            report["issues"].append(
                f"{t}: every observed smd_sum in {len(all_results)} records "
                f"is 0 ({n_null} were None/absent) — the Phase 2 reference "
                f"never resolved, so delta_sum and every synergy verdict built "
                f"on it are meaningless (BLOCKER 05a)")

        # (3) synergy true everywhere: the arithmetic consequence of (2).
        if all_results and all(r.get("synergy") is True
                               for r in all_results if isinstance(r, dict)):
            report["issues"].append(
                f"{t}: all {len(all_results)} records report synergy=true — "
                f"an impossible unanimity that follows from smd_sum=0 "
                f"(BLOCKER 05a)")

        # (4) Selectivity that was never measured must not read as "no penalty"
        #     (REVIEW FINDING 16). Phase 3 now marks these explicitly.
        unmeasured = [r.get("pc_id") for r in all_results
                      if isinstance(r, dict)
                      and r.get("selectivity_status") == "all_cross_target_mmsd_failed"]
        if unmeasured:
            report["level2"][t]["n_selectivity_unmeasured"] = len(unmeasured)
            msg = (f"{t}: {len(unmeasured)}/{len(all_results)} compositions had "
                   f"EVERY cross-target MMSD fail, so their selectivity is "
                   f"UNMEASURED (penalty=None, not 0.0): {unmeasured[:5]}")
            # Blocking only if it is the top pick or the majority; otherwise a
            # few failed cross-dockings are a known, survivable cost.
            top_ids = {p_.get("pc_id") for p_ in top_pcs if isinstance(p_, dict)}
            if top_ids & set(unmeasured):
                report["issues"].append(
                    msg + " — this includes the SELECTED composition, so the "
                          "pick was not selectivity-corrected")
            else:
                report["warnings"].append(msg)

        if sel.get("fallback_used"):
            report["warnings"].append(
                f"{t}: the final composition was NOT taken from the Pareto "
                f"front (fallback: {sel.get('fallback_reason')})")

        if schema != _PHASE3_EXPECTED_SCHEMA:
            report["issues"].append(
                f"{t}: Phase 3 result carries schema={schema!r}, expected "
                f"{_PHASE3_EXPECTED_SCHEMA!r} — this output predates the "
                f"BLOCKER 05 fix and Phase 4 will refuse it")

        # L3: top PC is polymerization-compatible — NOW ENFORCED
        if top_pcs:
            tp = top_pcs[0]
            ok, _chem, conflicts = is_polymerization_compatible(tp.get("monomers", []))
            report["level3"][t] = {
                "top_pc_polymerization_compatible": bool(ok),
                "top_pc_monomers": tp.get("monomers"),
                "selected_from": tp.get("selected_from"),
            }
            if not ok:
                report["issues"].append(
                    f"{t}: top PC {tp.get('monomers')} is NOT polymerization-"
                    f"compatible ({conflicts})")

    _require_targets_present(report, targets)
    return _finalize(report, "proceed_to_phase_4")


# ── Phase 4 ─────────────────────────────────────────────────────
def _judge_mmgbsa(report: dict, label: str, mm, required: bool = True) -> dict:
    """MM-GBSA must exist, carry no error, and report a ΔG."""
    out = {"present": bool(mm), "error": None, "delta_total_kcal": None}
    if not isinstance(mm, dict) or not mm:
        if required:
            report["issues"].append(f"{label}: MM-GBSA result MISSING")
        return out
    if mm.get("error"):
        out["error"] = mm["error"]
        report["issues"].append(f"{label}: MM-GBSA carries an error — {mm['error']}")
        return out
    dg = mm.get("delta_total_kcal")
    out["delta_total_kcal"] = dg
    if dg is None:
        report["issues"].append(
            f"{label}: MM-GBSA produced no delta_total_kcal (no ΔG was computed)")
    return out


def verify_phase4(targets=None):
    targets = _requested_targets(targets)
    p = RESULTS_ROOT / "phase4"
    report = _new_report(4, targets)

    data = _load_json(p / "phase4_md_results.json", report, "phase4 JSON")
    if data is None:
        return _finalize(report, "proceed_to_phase_5")

    for t in targets:
        td = data.get(t, {})
        if not isinstance(td, dict) or not td:
            continue                       # caught by _require_targets_present
        pcs = {k: v for k, v in td.items() if isinstance(v, dict)}
        report["level1"][t] = {"n_pcs": len(pcs)}
        if not pcs:
            report["issues"].append(f"{t}: zero PCs simulated in Phase 4")
            continue

        for pc_id, pcd in pcs.items():
            label = f"{t}/{pc_id}"
            occ = pcd.get("occupancy_analysis", {})
            conv = occ.get("convergence") if isinstance(occ, dict) else None
            converged = conv.get("converged") if isinstance(conv, dict) else None
            win = conv.get("window_diff_pct", {}) if isinstance(conv, dict) else {}
            worst_win = max([v for v in win.values()
                             if isinstance(v, (int, float))], default=None)

            mm = _judge_mmgbsa(report, label, pcd.get("mmpbsa"))
            if "mmpbsa_decomp" in pcd:
                _judge_mmgbsa(report, f"{label} decomp",
                              pcd.get("mmpbsa_decomp"), required=False)

            has_md = ("occupancy_analysis" in pcd) or ("rmsd" in pcd) \
                or ("rmsd_mean_nm" in pcd)
            report["level2"][label] = {
                "has_md_results": has_md,
                "success": pcd.get("success"),
                "error": pcd.get("error"),
                "converged": converged,
                "worst_window_diff_pct": worst_win,
                # Phase 4 now runs PHASE4_N_REPLICAS independent trajectories
                # and accepts them one by one; a leg pooled from zero accepted
                # replicas is not a validated leg.
                "accepted": pcd.get("accepted"),
                "n_replicas": pcd.get("n_replicas"),
                "n_replicas_accepted": pcd.get("n_replicas_accepted"),
                "pooling_basis": pcd.get("pooling_basis"),
                "mmgbsa_delta_total_kcal": mm["delta_total_kcal"],
                "mmgbsa_error": mm["error"],
                "A5_solvent_sweep_done": bool(pcd.get("solvent_sweep")),
                "A5_best_solvent": (pcd.get("solvent_sweep") or {}).get("best_solvent"),
                "B6_ratio_sweep_done": bool(pcd.get("ratio_sweep")),
                "B6_best_ratio": (pcd.get("ratio_sweep") or {}).get("best_ratio"),
            }

            # AN ALL-ZERO OCCUPANCY ANALYSIS IS NOT A MEASUREMENT
            # (REVIEW FINDING 9). A broken residue->monomer mapping gives every
            # monomer contact_freq 0.0 and EBN 0 while the convergence test
            # reads 0-vs-0 as converged, so the leg was accepted, pooled, and
            # passed here with zero issues. Phase 4 raises on the mapping
            # failures it can detect; this is the backstop for an all-zero
            # result arriving from any source (including an older run).
            cf = occ.get("contact_freq_6A") if isinstance(occ, dict) else None
            if isinstance(cf, dict) and cf:
                vals = [v for v in cf.values() if isinstance(v, (int, float))]
                if vals and not any(v > 0 for v in vals):
                    report["issues"].append(
                        f"{label}: EVERY monomer recorded zero contacts "
                        f"({cf}) — no monomer ever approached the epitope. "
                        f"This is a failed analysis (typically an empty "
                        f"residue->monomer mapping), not a measurement of "
                        f"non-binding; EBN and the derived synthesis ratio "
                        f"are meaningless")
            ebn = occ.get("EBN") if isinstance(occ, dict) else None
            if isinstance(ebn, dict) and ebn:
                evals = [v for v in ebn.values() if isinstance(v, (int, float))]
                if evals and not any(v > 0 for v in evals):
                    report["issues"].append(
                        f"{label}: EVERY monomer has EBN=0 ({ebn}) — the "
                        f"Phase 6 monomer ratio would be derived from no data")

            if not has_md:
                report["issues"].append(f"{label}: no MD results recorded")
            if pcd.get("error"):
                report["issues"].append(f"{label}: leg carries an error — {pcd['error']}")
            if pcd.get("success") is False:
                report["issues"].append(f"{label}: success=False")
            # THE Phase-4 blocker: non-converged MD is not a validated MD.
            if conv is None:
                report["issues"].append(
                    f"{label}: convergence was never computed (too few frames, or "
                    f"occupancy analysis did not run) — cannot certify this MD")
            elif converged is False:
                report["issues"].append(
                    f"{label}: MD NOT CONVERGED — worst Q3→Q4 contact-frequency "
                    f"window difference {worst_win}% (limit 10%); per-monomer {win}")

            if pcd.get("accepted") is False:
                report["issues"].append(
                    f"{label}: leg REJECTED by Phase 4 acceptance "
                    f"({pcd.get('n_replicas_accepted', 0)}/"
                    f"{pcd.get('n_replicas', '?')} replicas accepted; "
                    f"failures: {pcd.get('quality_failures')})")

            # L3: centering check.
            # Phase 4 writes <target>/<pc>/rep_<i>/md/ now, not <target>/<pc>/md.
            # Prefer the representative replica the pooled result names, and
            # fall back to the legacy single-trajectory path.
            md_dir = None
            for cand in ([pcd.get("md_dir")] if pcd.get("md_dir") else []) + \
                        list(pcd.get("replica_md_dirs") or []) + \
                        [p / t / pc_id / "md"]:
                if cand and (Path(cand) / "md.gro").exists():
                    md_dir = Path(cand)
                    break
            if md_dir is not None:
                cen = _check_md_centering(md_dir / "md.gro")
                cen["md_dir"] = str(md_dir)
                report["level3"][label] = cen
                _judge_centering(report, label, cen)

    # Cross-reactivity arm
    xr = data.get("cross_reactivity", {})
    if isinstance(xr, dict) and xr:
        report["level2"]["cross_reactivity"] = {
            k: {"success": v.get("success"), "error": v.get("error")}
            for k, v in xr.items() if isinstance(v, dict)
        }
        for k, v in xr.items():
            if not isinstance(v, dict):
                continue
            if v.get("error"):
                report["issues"].append(f"cross_reactivity/{k}: {v['error']}")
            elif v.get("success") is False:
                report["issues"].append(f"cross_reactivity/{k}: success=False")

    # ── Phase 4's own acceptance verdict ────────────────────────────
    # phase4_quality_report.json is written separately from
    # phase4_md_results.json (deliberately: phase5/phase6 iterate the results
    # dict's keys as targets, so an extra top-level key there would be read as
    # a fourth target). Without reading it, a run in which every replica was
    # rejected would still pass verification.
    qpath = p / "phase4_quality_report.json"
    if qpath.exists():
        q = _load_json(qpath, report, "phase4 quality report")
        if isinstance(q, dict):
            rejected = q.get("rejected_legs") or []
            report["level2"]["quality_report"] = {
                "n_rejected_legs": len(rejected),
                "rejected_legs": rejected,
            }
            for leg in rejected:
                report["issues"].append(
                    f"phase4 quality report: leg {leg} was REJECTED "
                    f"(no replica met the acceptance criteria)")
    else:
        # M8: previously a warning-only path — but if ANY per-leg record is
        # ALSO missing the `accepted` field (pre-per-replica-acceptance),
        # then convergence cannot be judged AT ALL and passing is unsafe.
        any_leg_missing_accepted = any(
            pcd.get("accepted") is None
            for _, pcs in (data or {}).items() if isinstance(pcs, dict)
            for _, pcd in pcs.items() if isinstance(pcd, dict)
        )
        if any_leg_missing_accepted:
            report["issues"].append(
                "phase4_quality_report.json is absent AND at least one leg has "
                "no `accepted` field — convergence verdict cannot be recovered "
                "from either source, so acceptance is UNKNOWN. Re-run Phase 4 "
                "on the current code to produce the quality report.")
        else:
            report["warnings"].append(
                "phase4_quality_report.json is absent — this Phase 4 run "
                "predates per-replica acceptance, and every leg carries its "
                "own `accepted` field, so the report is not strictly required")

    _require_targets_present(report, targets)
    return _finalize(report, "proceed_to_phase_5")


# ── Phase 5 ─────────────────────────────────────────────────────
def _load_pcsi_star():
    """PCSI* summary for THIS run's reports dir only.

    Deliberately does NOT search reports_v2/ or any sibling tree: importing a
    selectivity verdict computed against a different (archived) results tree is
    precisely the silent-wrongness this file exists to stop. Point
    MIP_PCSI_STAR_JSON at a file to override.
    """
    env = os.environ.get("MIP_PCSI_STAR_JSON")
    cands = [Path(env)] if env else []
    cands.append(REPORTS_DIR / "pcsi_star_summary.json")
    for c in cands:
        if c.is_file():
            try:
                with open(c) as f:
                    return json.load(f), str(c)
            except Exception:
                continue
    return None, None


def verify_phase5(targets=None):
    """Phase 5 verification — selectivity gate.

    Metric precedence:
      1. PCSI*  (code/run_pcsi_star.py → reports/pcsi_star_summary.json).
         A target passes only if its cavity gate verdict is PASS.
      2. legacy PCSI (reports/phase5_persistent_contacts.json).
      3. nothing → BLOCK. "No metric" is not "passed".

    FIXED (BLOCKER 08): an empty cross-target list used to give
    pcsi = own/0 → +inf → PASS. A cavity whose off-target arm was never
    measured now BLOCKS instead.
    """
    targets = _requested_targets(targets)
    p = RESULTS_ROOT / "phase5"
    report = _new_report(5, targets)

    data = _load_json(p / "phase5_rebinding_results.json", report, "phase5 JSON")
    if data is None:
        return _finalize(report, "proceed_to_phase_6")

    star, star_path = _load_pcsi_star()
    pcsi_data = {}
    legacy_path = REPORTS_DIR / "phase5_persistent_contacts.json"
    if star is None and legacy_path.exists():
        try:
            with open(legacy_path) as f:
                pcsi_data = json.load(f)
        except Exception as e:
            report["issues"].append(f"legacy PCSI JSON unparseable: {e}")
    # M4: single-target experiments (e.g. BSA) leave any legacy
    # phase5_persistent_contacts.json stale (it was produced by an earlier
    # multi-target run). Ignoring it means the cavity is judged on rebinding
    # alone, which is the right gate for a single-target run.
    try:
        from .config import TARGETS as _CFG_TARGETS
        _n_experiment_targets = len(list(_CFG_TARGETS))
    except Exception:
        _n_experiment_targets = 0
    if pcsi_data and _n_experiment_targets < 2:
        report["warnings"].append(
            f"legacy PCSI JSON ({legacy_path.name}) ignored: this experiment "
            f"has {_n_experiment_targets} target(s), so the file is stale "
            f"from a previous multi-target run and cannot judge this one")
        pcsi_data = {}
    report["metric"] = ("PCSI*" if star is not None
                        else "PCSI(legacy)" if pcsi_data else "none")
    report["metric_source"] = star_path or (str(legacy_path) if pcsi_data else None)

    # SELECTIVITY IS ONLY DEFINED WITH AN OFF-TARGET (REVIEW FINDING 3).
    #
    # PCSI* is a cross-target ratio: it needs at least one off-target to
    # challenge the cavity with. A single-target experiment (BSA runs
    # TARGETS == ["BSA"]) has no off-target arm even in principle, so
    # demanding a PCSI* summary there is an UNSATISFIABLE block — nothing
    # could ever produce it and Phase 5 could never clear.
    #
    # The test is on the EXPERIMENT's full target list, not the requested
    # subset: `verify_phase.py 5 --targets CD63` under CD must still be gated,
    # because CD81/CD9 exist in the data as genuine off-targets.
    try:
        from .config import TARGETS as _ALL_TARGETS
        _n_targets = len(list(_ALL_TARGETS))
    except Exception:
        _n_targets = 0
    selectivity_definable = _n_targets >= 2
    report["selectivity_definable"] = selectivity_definable
    report["n_experiment_targets"] = _n_targets

    if star is None and not pcsi_data:
        if selectivity_definable:
            report["issues"].append(
                "NO selectivity metric computed for this results tree — run "
                "`python code/run_pcsi_star.py` (it defaults to this "
                "experiment's reports dir, which is exactly where this check "
                "looks). Phase 5 cannot be judged without it, and "
                "'not measured' is not 'passed'.")
        else:
            report["warnings"].append(
                f"selectivity NOT EVALUATED: this experiment has "
                f"{_n_targets} target(s), so there is no off-target arm and "
                f"PCSI* is undefined. Phase 5 is judged on rebinding alone; "
                f"do NOT cite a selectivity claim for this run.")

    passing = []
    for t in targets:
        td = data.get(t, {})
        if not isinstance(td, dict) or not td:
            continue                       # caught by _require_targets_present
        sel = td.get("selectivity", {}) or {}
        a6_count = sum(1 for v in sel.values()
                       if isinstance(v, dict) and "si_ci_lower" in v)
        snaps = td.get("snapshots", []) or []
        b8_count = sum(1 for s in snaps if isinstance(s, dict)
                       and "multipose_ensemble" in s)
        n_rebound = td.get("n_rebound", 0)
        n_snaps = td.get("n_snapshots", 0)

        report["level1"][t] = {
            "n_rebound": n_rebound,
            "n_snapshots": n_snaps,
            "n_cross_selectivity": len(sel),
        }
        if not n_rebound:
            report["issues"].append(f"{t}: zero rebound snapshots")
        report["level2"][t] = {
            "A6_bootstrap_CI_active": a6_count > 0,
            "A6_n_with_CI": a6_count,
            "B8_multipose_done": b8_count > 0,
        }

        # Level 3: cavity centering.
        # Phase 5 names snapshots rep<r>_snapshot_<i> when they are spread over
        # Phase 4 replicas, so a hardcoded "snapshot_0" finds nothing and the
        # check silently passes on an empty dict. Take the first snapshot
        # directory that actually exists, whichever layout produced it.
        lvl3 = {}
        _snap = None
        for cand in sorted((p / t).iterdir()) if (p / t).is_dir() else []:
            if cand.is_dir() and _SNAPSHOT_DIR_RE.match(cand.name) \
                    and (cand / "frame.gro").exists():
                _snap = cand
                break
        if _snap is not None:
            lvl3 = _check_md_centering(_snap / "frame.gro")
            lvl3["snapshot_dir"] = _snap.name
            _judge_centering(report, f"{t} {_snap.name}", lvl3)
        report["level3"][t] = lvl3

        # ── selectivity gate ────────────────────────────────────
        # ── EV-approach branch (fires first) ──────────────────
        # Detect the EV-templated protocol by looking at any snapshot's
        # rebind_own leg: `run_ev_approach_leg` stamps protocol='ev_approach'
        # and aggregates N placements per leg. When present, compute PCSI
        # directly from the leg-level `n_persistent_residues` (median across
        # placements, promoted to the leg top-level by _aggregate_ev_placements).
        _ev_snaps = [s for s in snaps if isinstance(s, dict)
                     and isinstance(s.get("rebind_own"), dict)
                     and s["rebind_own"].get("protocol") == "ev_approach"]
        if _ev_snaps:
            def _median_leg_np(leg_key: str) -> int | None:
                vals = []
                for s in _ev_snaps:
                    leg = s.get(leg_key)
                    if isinstance(leg, dict) and leg.get("success") \
                            and "n_persistent_residues" in leg:
                        vals.append(int(leg["n_persistent_residues"]))
                if not vals:
                    return None
                return int(np.median(vals)) if _NP_AVAILABLE \
                    else int(round(sum(vals) / len(vals)))

            own_n = _median_leg_np("rebind_own")
            cross_ns = {}
            for other in targets:
                if other == t:
                    continue
                v = _median_leg_np(f"rebind_{other}")
                if v is not None:
                    cross_ns[other] = v
            lvl3["protocol"] = "ev_approach"
            lvl3["own_persistent_contacts"] = own_n
            lvl3["cross_persistent_contacts"] = cross_ns
            lvl3["n_placements_per_leg"] = (
                _ev_snaps[0].get("rebind_own", {}).get("n_placements", 1))
            if own_n is None or own_n <= 0:
                report["issues"].append(
                    f"{t}: EV-approach — zero persistent contacts with own "
                    f"fresh CD-in-EV; cavity does not rebind its own target")
                lvl3["PCSI_PASS"] = False
                continue
            if not cross_ns:
                report["issues"].append(
                    f"{t}: EV-approach — no cross-target EV legs measured; "
                    f"selectivity cannot be defined")
                lvl3["PCSI_PASS"] = False
                continue
            max_cross = max(cross_ns.values())
            pcsi = own_n / max_cross if max_cross > 0 else float("inf")
            lvl3["PCSI"] = (round(pcsi, 2) if pcsi != float("inf")
                            else "inf")
            lvl3["max_cross_persistent_contacts"] = max_cross
            pcsi_pass = (pcsi > 1.2) if pcsi != float("inf") else True
            lvl3["PCSI_PASS"] = pcsi_pass
            if pcsi_pass:
                passing.append(t)
            else:
                report["warnings"].append(
                    f"{t}: EV-approach PCSI {pcsi:.2f} <= 1.2 (own={own_n}, "
                    f"max_cross={max_cross})")
            report["level3"][t] = lvl3
            continue

        if star is not None:
            cav = {k: v for k, v in (star.get("cavities") or {}).items()
                   if isinstance(v, dict) and v.get("cavity") == t}
            if not cav:
                report["issues"].append(
                    f"{t}: no PCSI* cavity entry in {star_path}")
                lvl3["PCSI_STAR_PASS"] = False
                continue
            verdicts, missing_arms = {}, set()
            for key, cv in cav.items():
                gate = cv.get("gate", {}) or {}
                verdicts[key] = gate.get("verdict")
                missing_arms.update(gate.get("missing_arms") or [])
                # M6: worst_case_over_n may be a non-numeric string or absent
                # in older producer versions. Coerce safely — a bad value must
                # block cleanly, not raise ValueError and kill the verifier
                # before any Phase 5 finding is emitted.
                try:
                    wc = int(gate.get("worst_case_over_n") or 0)
                except (TypeError, ValueError):
                    report["issues"].append(
                        f"{t}/{key}: PCSI* gate['worst_case_over_n'] is "
                        f"non-numeric ({gate.get('worst_case_over_n')!r}) — "
                        f"treating as EMPTY CROSS-TARGET ARM")
                    wc = 0
                if wc < 1:
                    report["issues"].append(
                        f"{t}/{key}: EMPTY CROSS-TARGET ARM — worst-case min is"
                        f"over zero off-targets; not a selectivity result")
            lvl3["PCSI_STAR_verdicts"] = verdicts
            lvl3["PCSI_STAR_missing_arms"] = sorted(missing_arms)
            ok = any(v == "PASS" for v in verdicts.values())
            lvl3["PCSI_STAR_PASS"] = ok
            if missing_arms:
                report["issues"].append(
                    f"{t}: cross-target arm(s) never measured: "
                    f"{sorted(missing_arms)} — the cavity was never challenged "
                    f"with those off-targets")
            if ok:
                passing.append(t)
            else:
                report["warnings"].append(
                    f"{t}: PCSI* verdicts {verdicts} (no PASS)")

        elif pcsi_data:
            if t not in pcsi_data:
                report["issues"].append(
                    f"{t}: no legacy-PCSI entry — selectivity never computed")
                lvl3["PCSI_PASS"] = False
                continue
            tgt_pcsi = pcsi_data[t]
            own_d = tgt_pcsi.get(t, {})
            own_n = own_d.get("n_persistent_residues", 0) if isinstance(own_d, dict) else 0
            # Size-exclusion selectivity: cross-targets physically rejected
            # pre-EM (clash_count > REBINDING_CLASH_THRESHOLD) are labelled
            # "size-excluded" by phase5_rebinding._analyze_rebinding_results.
            # That IS mechanism-level selectivity (Hoshino 2008, Shea group) —
            # the cavity is too small/wrong-shape for that off-target to enter.
            # An off-target absent for OTHER reasons (crashed MD, never run) is
            # different: it's absence of evidence. Distinguish them.
            size_excluded = {o for o, v in sel.items()
                             if isinstance(v, dict)
                             and v.get("selectivity_label") == "size-excluded"}
            cross_ns = [tgt_pcsi.get(o, {}).get("n_persistent_residues", 0)
                        for o in targets if o != t
                        and o not in size_excluded
                        and isinstance(tgt_pcsi.get(o, {}), dict)
                        and "n_persistent_residues" in tgt_pcsi.get(o, {})]
            # Off-targets that were expected but neither measured nor size-
            # excluded — genuine data gap.
            unmeasured = [o for o in targets if o != t
                          and o not in size_excluded
                          and (o not in tgt_pcsi
                               or not isinstance(tgt_pcsi.get(o, {}), dict)
                               or "n_persistent_residues"
                               not in tgt_pcsi.get(o, {}))]
            max_cross = max(cross_ns) if cross_ns else None
            lvl3["own_persistent_contacts"] = own_n
            lvl3["max_cross_persistent_contacts"] = max_cross
            lvl3["n_measured_cross_arms"] = len(cross_ns)
            if size_excluded:
                lvl3["size_excluded_targets"] = sorted(size_excluded)
            if unmeasured:
                lvl3["unmeasured_cross_targets"] = sorted(unmeasured)

            if own_n <= 0:
                report["issues"].append(
                    f"{t}: zero persistent contacts with its OWN template — the "
                    f"cavity does not rebind what it was imprinted against")
                lvl3["PCSI"] = None
                lvl3["PCSI_PASS"] = False
                continue

            if unmeasured:
                # Genuine missing data — cannot claim selectivity we didn't test.
                report["issues"].append(
                    f"{t}: cross-target arm(s) {unmeasured} missing without "
                    f"size-exclusion — selectivity vs those targets was never "
                    f"measured")
                lvl3["PCSI"] = None
                lvl3["PCSI_PASS"] = False
                continue

            if not cross_ns and size_excluded:
                # All off-targets size-excluded, own binds → mechanism-level
                # selectivity. PASS, mark with mechanism attribution.
                lvl3["PCSI"] = "size-exclusion"
                lvl3["PCSI_PASS"] = True
                lvl3["selectivity_mechanism"] = "size/shape exclusion"
                passing.append(t)
                report["warnings"].append(
                    f"{t}: PASS via size-exclusion mechanism (own={own_n}, "
                    f"all cross-targets {sorted(size_excluded)} physically "
                    f"rejected pre-EM) — no numeric PCSI ratio")
                continue

            pcsi = own_n / max_cross if max_cross and max_cross > 0 else 0.0
            pcsi_pass = (pcsi > 1.2) and own_n > 0
            lvl3["PCSI"] = round(pcsi, 2)
            lvl3["PCSI_PASS"] = pcsi_pass
            # A3: membrane calibration. Display alongside raw PCSI so the
            # reader sees what to expect experimentally on membrane-embedded
            # target. Does NOT change PASS/FAIL — that stays on raw simulation
            # numbers (moving the threshold instead would compress uncertainty
            # into one non-transparent knob).
            try:
                from .config import SELECTIVITY_MEMBRANE_CALIBRATION as _CAL
            except Exception:
                _CAL = 1.0
            if _CAL != 1.0:
                lvl3["PCSI_calibrated"] = round(pcsi * _CAL, 2)
                lvl3["calibration_factor"] = _CAL
                lvl3["calibration_note"] = (
                    f"Membrane-context calibration ×{_CAL} applied to raw "
                    f"PCSI. Represents expected experimental value on "
                    f"membrane-embedded target (Zimmerman 2016, Umeda 2020).")
            if size_excluded:
                # Mixed: some cross size-excluded, some measured → both mechanisms
                lvl3["selectivity_mechanism"] = (
                    "persistent-contact + size/shape exclusion")
            if pcsi_pass:
                passing.append(t)
            else:
                report["warnings"].append(
                    f"{t}: PCSI {pcsi:.2f} <= 1.2 (own={own_n}, "
                    f"max_cross={max_cross})")
        elif not selectivity_definable:
            # Single-target experiment: there is no off-target, so the gate
            # degrades to "did the cavity rebind its own template at all?".
            # n_rebound == 0 has already been raised as a blocking issue above.
            lvl3["PCSI_PASS"] = None
            lvl3["SELECTIVITY_NOT_APPLICABLE"] = True
            if n_rebound:
                passing.append(t)
        else:
            lvl3["PCSI_PASS"] = False

    _require_targets_present(report, targets)
    report["targets_pass"] = passing
    report["n_targets_pass"] = len(passing)
    report["n_targets_total"] = len(report["level1"])
    if not passing:
        report["issues"].append(
            "NO target passed the selectivity gate — Phase 6 would emit a "
            "synthesis recipe for a polymer with no demonstrated selectivity"
            if selectivity_definable else
            "NO target rebound its own template — Phase 6 would emit a "
            "synthesis recipe for a polymer that binds nothing")
    return _finalize(
        report, f"proceed_to_phase_6_with_{len(passing)}_target(s)")


# ── Phase 6 ─────────────────────────────────────────────────────
def verify_phase6(targets=None):
    targets = _requested_targets(targets)
    p = RESULTS_ROOT / "phase6"
    report = _new_report(6, targets)

    from .config import is_polymerization_compatible, PHASE6_GENERATE_NIP

    # Phase 6 only owes recipes for the targets Phase 5 cleared. If the Phase 5
    # verification exists, use its pass list; otherwise every requested target.
    p5v = REPORTS_DIR / "phase5_verification.json"
    scoped_by_phase5 = False
    if p5v.is_file():
        try:
            with open(p5v) as f:
                p5 = json.load(f)
            if isinstance(p5.get("targets_pass"), list):
                targets = [t for t in targets if t in p5["targets_pass"]]
                report["targets_requested"] = list(targets)
                report["scoped_by"] = "phase5_verification.targets_pass"
                scoped_by_phase5 = True
        except Exception:
            pass

    # AN EMPTY SCOPE IS NOT A SUCCESS (REVIEW FINDING 4).
    #
    # Narrowing to targets_pass means an empty pass list leaves nothing to
    # loop over: the per-target checks never run, _require_targets_present has
    # nothing to require, and the vacuous report was finalised as
    # "all_phases_complete" with blocked=False and exit 0. That is BLOCKER
    # 08b's "empty arm scores PASS" reappearing at the FINAL gate — the last
    # line a human reads from `verify_phase.py all` said the pipeline
    # completed, when Phase 5 had cleared no target at all.
    if scoped_by_phase5 and not targets:
        report["issues"].append(
            "Phase 5 cleared NO target, so there is nothing for Phase 6 to be "
            "complete about — an empty scope is not a passing Phase 6")
        report["n_targets_scoped"] = 0
        return _finalize(report, "all_phases_complete")

    data = _load_json(p / "phase6_recipes.json", report, "phase6 recipes JSON")
    if data is None:
        return _finalize(report, "all_phases_complete")

    for t in targets:
        mip = data.get(t, {}) or {}
        nip = data.get(t + "_NIP", {}) or {}
        per_target_file = p / f"{t}_recipe.txt"
        if not mip and not nip and not per_target_file.exists():
            continue                       # caught by _require_targets_present
        report["level1"][t] = {
            "mip_present": bool(mip),
            "nip_present": bool(nip),  # A7
            "per_target_file_exists": per_target_file.exists(),  # A7
        }
        if not mip:
            report["issues"].append(f"{t}: no MIP recipe generated")
        if PHASE6_GENERATE_NIP and not nip:
            report["issues"].append(
                f"{t}: PHASE6_GENERATE_NIP is on but no NIP control recipe exists")
        if not per_target_file.exists():
            report["issues"].append(f"{t}: {per_target_file.name} not written")

        notes_str = " ".join(mip.get("notes", []) if isinstance(mip.get("notes"), list)
                              else [str(mip.get("notes", ""))])
        has_initiator = "mg Irgacure" in notes_str or "mg AIBN" in notes_str
        report["level2"][t] = {
            "A7_nip_recipe": bool(nip),
            "B10_initiator_mass_in_notes": has_initiator,
            "serum_warning_present": "SERUM CROSS-REACTIVITY" in notes_str,
            "ev_context_note": "EV CONTEXT" in notes_str,
            "ph_recommended_present": "pH" in notes_str,
        }
        if not has_initiator and mip.get("polymerization_type") != "silane":
            report["warnings"].append(
                f"{t}: no initiator mass in recipe notes (B10)")

        # L3: polymerization compat — NOW ENFORCED
        if mip.get("monomers"):
            ok, _chem, conflicts = is_polymerization_compatible(
                list(mip["monomers"].keys()))
            report["level3"][t] = {"polymerization_compatible": bool(ok)}
            if not ok:
                report["issues"].append(
                    f"{t}: recipe monomers {list(mip['monomers'])} are NOT "
                    f"polymerization-compatible ({conflicts})")
        elif mip:
            report["issues"].append(f"{t}: recipe carries no monomers")

    _require_targets_present(report, targets)
    return _finalize(report, "all_phases_complete")


VERIFY_FUNCS = {
    1: verify_phase1, 2: verify_phase2, 3: verify_phase3,
    4: verify_phase4, 5: verify_phase5, 6: verify_phase6,
}


def set_results_root(root) -> None:
    """Point the verifier at a specific results tree.

    run_pipeline.py honours --output-dir, so the verifier must not stay pinned
    to whatever get_output_path() resolved at import time.
    """
    global RESULTS_ROOT, REPORTS_DIR
    RESULTS_ROOT = Path(root)
    REPORTS_DIR = RESULTS_ROOT / "reports"


def write_summary_md(report):
    """Human-readable summary."""
    n = report["phase"]
    lines = [
        f"# Phase {n} Verification Summary",
        "",
        f"**Decision**: `{report.get('decision', 'unknown')}`  "
        f"(blocked: `{report.get('blocked')}`)",
        "",
    ]
    if report.get("issues"):
        lines.append("## ⛔ BLOCKING issues")
        for issue in report["issues"]:
            lines.append(f"- {issue}")
        lines.append("")
    if report.get("warnings"):
        lines.append("## ⚠ Warnings (non-blocking)")
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    for level, name in [("level1", "L1 Code Integrity"),
                        ("level2", "L2 Algorithm Correctness"),
                        ("level3", "L3 Physics/Chemistry Validity")]:
        lines.append(f"## {name}")
        for target, checks in report.get(level, {}).items():
            lines.append(f"### {target}")
            if not isinstance(checks, dict):
                lines.append(f"- `{checks}`")
                continue
            for k, v in checks.items():
                icon = ("✓" if v in (True, "True") else "✗" if v in (False, None) else "—")
                lines.append(f"- {icon} {k}: `{v}`")
        lines.append("")
    out = REPORTS_DIR / f"phase{n}_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(phase_num, targets=None, results_root=None, quiet=False):
    """Verify one phase. Returns the report dict (report["blocked"] is the verdict)."""
    if results_root is not None:
        set_results_root(results_root)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    func = VERIFY_FUNCS.get(phase_num)
    if func is None:
        raise ValueError(f"Unknown phase: {phase_num}")
    if not quiet:
        print(f"Verifying Phase {phase_num} ({RESULTS_ROOT})...")
    report = func(targets)
    out_json = REPORTS_DIR / f"phase{phase_num}_verification.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md = write_summary_md(report)
    if not quiet:
        print(f"  → {out_json}")
        print(f"  → {out_md}")
        for issue in report.get("issues", []):
            print(f"  BLOCKING: {issue}")
        for w in report.get("warnings", []):
            print(f"  warning : {w}")
        print(f"  Decision: {report.get('decision')}  "
              f"(blocked={report.get('blocked')})")
    return report


def _cli():
    ap = argparse.ArgumentParser(
        description="Verify one pipeline phase (exit 1 = BLOCKED).",
        epilog=BLOCKING_CRITERIA,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", nargs="?", default=None,
                    help="1..6, or 'all'")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="targets this phase owes results for "
                         "(default: run_manifest.json, else every TARGETS key)")
    ap.add_argument("--results-root", default=None,
                    help="results tree to verify (default: config OUTPUT_DIR)")
    ap.add_argument("--criteria", action="store_true",
                    help="print the enforced blocking criteria and exit")
    args = ap.parse_args()

    if args.criteria:
        print(BLOCKING_CRITERIA)
        return 0
    if args.phase is None:
        ap.error("a phase (1..6 or 'all') is required unless --criteria")

    if str(args.phase).lower() == "all":
        blocked_any = False
        for n in sorted(VERIFY_FUNCS):
            rep = main(n, args.targets, args.results_root)
            blocked_any |= bool(rep.get("blocked"))
        return 1 if blocked_any else 0

    try:
        n = int(args.phase)
    except ValueError:
        print(f"Unknown phase: {args.phase}")
        return 2
    if n not in VERIFY_FUNCS:
        print(f"Unknown phase: {n}")
        return 2
    rep = main(n, args.targets, args.results_root)
    return 1 if rep.get("blocked") else 0


if __name__ == "__main__":
    sys.exit(_cli())
