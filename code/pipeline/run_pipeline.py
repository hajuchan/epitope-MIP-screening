"""
Epitope-MIP Screening Pipeline — Main Entry Point
==================================================
6-Phase computational screening of functional monomers for
epitope-imprinted MIPs targeting exosome tetraspanins (CD63/CD81/CD9).

Phase 1: Epitope extraction + structure preparation
Phase 2: Single Monomer Docking (SMD) with AutoDock
Phase 3: Multi-Monomer Simultaneous Docking (MMSD) + NSGA-II
Phase 4: GROMACS pre-polymerisation MD + MM-GBSA
Phase 5: VIP cavity rebinding
Phase 6: Synthesis recipe generation

Additional options:
  --report     Generate HTML report from existing results

Exit codes
----------
  0  pipeline finished, every verified phase passed
  2  a phase FAILED VERIFICATION (verify_phase.py blocked) — see
     <results>/reports/phaseN_summary.md
  3  --resume refused: the stored run inputs no longer match this process
     (structures / monomer SMILES / config fingerprint drifted), or the tree
     has no run manifest at all
  4  --fresh refused: the existing tree was not archived or deleted

──────────────────────────────────────────────────────────────────────────
BLOCKER 01 — WHAT CHANGED AND WHY (2026-08, audit fix)
──────────────────────────────────────────────────────────────────────────
Before this rewrite:
  * --resume defaulted to True and _check_phase_completed() answered True for
    all six phases against any existing results tree, so the documented re-run
    command executed ZERO phases and only regenerated the HTML report.
  * --fresh deleted nothing. It set resume=False and was never passed down to
    Phases 3, 4 or 5 — all three resume internally from their own per-target
    JSONs, so a "fresh" run silently reused every stale per-target result.
  * "does the file exist" was the entire resume test. A results tree written
    from different structures, a different monomer library or a different
    config was indistinguishable from a matching one.

Now:
  * --fresh REFUSES to run unless it has archived (default, an os.rename) or
    explicitly deleted (--fresh-mode delete --yes-delete) the phase directories
    it is about to rewrite. It never silently reuses.
  * --fresh is threaded into every phase call whose signature accepts it, and
    LOUDLY reports the phases that do not accept it yet (their stale state is
    already gone with the archived directory, which is the structural fix).
  * --resume validates a run manifest: experiment, per-target structure
    identity (sha256 of the prepared template), the monomer name→SMILES set,
    and a per-key config fingerprint. Any drift refuses with exit 3.
  * A results tree with phase results but NO manifest cannot be resumed at all
    (--adopt-existing-tree to opt in deliberately).
  * verify_phase.py is invoked after every phase — including phases loaded from
    disk — and a blocking verdict stops the pipeline with exit 2.
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import OUTPUT_DIR, TARGETS

logger = logging.getLogger(__name__)

_CODE_DIR = Path(__file__).resolve().parent.parent          # …/code

# Exit codes (see module docstring)
EXIT_VERIFY_BLOCKED = 2
EXIT_RESUME_REFUSED = 3
EXIT_FRESH_REFUSED = 4

MANIFEST_NAME = "run_manifest.json"
MANIFEST_VERSION = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Epitope-MIP Screening Pipeline for Exosome Tetraspanins"
    )
    parser.add_argument(
        "--target", type=str, nargs="+", default=["all"],
        help="Which target(s) to screen (default: all). E.g. --target CD63 CD81"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        help="Which phase to run (default: all)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate HTML report from existing results"
    )
    parser.add_argument(
        "--skip-md", action="store_true",
        help="Skip all intermediate MD (Phase 1 stability + Phase 2 contact). "
             "NOT recommended — use only for debugging docking logic."
    )
    parser.add_argument(
        "--quick-md", action="store_true",
        help="Run 20ns instead of 50ns MD in Phase 4 (debugging only)"
    )
    parser.add_argument(
        "--no-cross-reactivity", action="store_true",
        help="Skip cross-reactivity test in Phase 4"
    )

    # ── state / re-run control ──────────────────────────────────
    g = parser.add_argument_group(
        "state control",
        "Resume is only honoured when the STORED INPUTS still match this "
        "process. Fresh only runs after the target directories are gone.")
    g.add_argument(
        "--resume", action="store_true", default=None,
        help="Resume from completed phases (DEFAULT: on). A phase counts as "
             "complete only if the run manifest records it, the result file "
             "still hashes to the recorded value, it covers every requested "
             "target, and the structure/monomer/config fingerprint matches."
    )
    g.add_argument(
        "--no-resume", action="store_true",
        help="REFUSED unless combined with --fresh: Phases 3/4/5 resume "
             "internally from their own per-target JSONs, so merely disabling "
             "the top-level skip re-uses stale per-target results. Use --fresh."
    )
    g.add_argument(
        "--fresh", action="store_true",
        help="Re-run from scratch. ARCHIVES (default) or deletes the phase "
             "directories being run before starting. Refuses to run if it "
             "cannot."
    )
    g.add_argument(
        "--fresh-mode", choices=["archive", "delete"], default="archive",
        help="--fresh disposal: archive (os.rename, reversible, default) or "
             "delete (irreversible, needs --yes-delete)."
    )
    g.add_argument(
        "--archive-dir", type=str, default=None,
        help="Where --fresh archives to (default: "
             "<output-dir>.archive_<UTC timestamp> next to the output dir)."
    )
    g.add_argument(
        "--yes-delete", action="store_true",
        help="Required confirmation for --fresh-mode delete. Irreversible."
    )
    g.add_argument(
        "--adopt-existing-tree", action="store_true",
        help="Write a run manifest for a results tree that predates manifests "
             "and resume from it. You are asserting the tree was produced by "
             "the CURRENT structures, monomer SMILES and config. Requires "
             "--yes-adopt confirmation because it makes the tree look "
             "reproducible without evidence — this can permanently launder "
             "mixed-input results into 'clean provenance' in later resumes."
    )
    g.add_argument(
        "--yes-adopt", action="store_true",
        help="Required confirmation for --adopt-existing-tree. Adopted phases "
             "are permanently marked with provenance='adopted' in the manifest."
    )
    g.add_argument(
        "--accept-input-drift", action="store_true",
        help="UNSAFE. Resume even though the stored inputs no longer match. "
             "Every mismatching key is logged. Mixed-input results follow."
    )
    g.add_argument(
        "--skip-verify", action="store_true",
        help="UNSAFE. Do not run verify_phase.py between phases, so a failed "
             "phase does not stop the pipeline."
    )

    parser.add_argument(
        "--extend-drifters", action="store_true",
        help="Extend Phase 5 rebinding MDs that show RMSD drift Q1→Q4 > 1.5 Å. "
             "Uses gmx convert-tpr -extend + mdrun -cpi to continue MD by 100 ns. "
             "Run after standard Phase 5 completes."
    )
    parser.add_argument(
        "--extend-ns", type=int, default=100,
        help="Extension length in ns for --extend-drifters (default: 100)"
    )
    parser.add_argument(
        "--multirestart", action="store_true",
        help="Multi-restart ensemble: re-run Phase 5 rebinding with N=3 perturbed "
             "starting head positions per snapshot. Adds extra reps to existing run."
    )
    parser.add_argument(
        "--n-reps", type=int, default=3,
        help="Number of replicates for --multirestart (default: 3)"
    )
    parser.add_argument(
        "--reanalyze", action="store_true",
        help="Re-analyze existing Phase 5 trajectories with PBC centering "
             "(gmx trjconv -pbc mol -center) and Q4 (last 25%%) RMSD. No new MD. "
             "Recomputes selectivity matrix from corrected analysis."
    )
    return parser.parse_args(argv)


def _phase_dir(root_dir: str, phase_key: str) -> str:
    """Return phase-specific output subdirectory, creating if needed."""
    p = Path(root_dir) / phase_key
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# Result file that marks each phase as completed
_PHASE_RESULT_FILES = {
    1: "phase1/phase1_results.json",
    2: "phase2/phase2_smd_results.json",
    3: "phase3/phase3_mmsd_results.json",
    4: "phase4/phase4_md_results.json",
    5: "phase5/phase5_rebinding_results.json",
    6: "phase6/phase6_recipes.json",
}


# ════════════════════════════════════════════════════════════════
# Run fingerprint + manifest
# ════════════════════════════════════════════════════════════════
# Environment/machine-specific config symbols. These change what the run COSTS,
# not what it COMPUTES, so they must not invalidate a resume.
_VOLATILE_CONFIG_KEYS = frozenset({
    "PROJECT_ROOT", "OUTPUT_DIR", "OUTPUT_DIRS",
    "AUTODOCK4_BIN", "AUTOGRID4_BIN", "AUTODOCK_GPU_BIN",
    "PREPARE_RECEPTOR", "PREPARE_LIGAND", "GMX_BIN", "ACPYPE_BIN",
    "USE_GPU", "USE_AUTODOCK_GPU", "MD_GPU_ID", "N_WORKERS",
    "EXPERIMENT_LABEL", "UNCONSUMED_KEYS", "KNOWN_EXPERIMENTS",
})

# Config-level symbols that are ALREADY hashed by dedicated fingerprint
# sections (structures / targets / monomers). Hashing them a second time in
# the `config` section means an edit to a single monomer SMILES trips drift
# in TWO sections — one detected mismatch reported as two, without adding
# any information.
_DUPLICATE_SECTION_KEYS = frozenset({
    "TARGETS",         # covered by sections['targets'] + sections['structures']
    "ALL_MONOMERS",    # covered by sections['monomers']
    "FUNCTIONAL_MONOMERS", "SILANE_MONOMERS", "VINYL_MONOMERS",
    "CROSSLINKERS", "CROSSLINKER_LIBRARY",
    "PRIMARY_CHEM_CLASS",  # derived from ALL_MONOMERS metadata
})

# Per-target structure identity fields. Anything here changing means the run
# was pointed at a different molecule or a different region of it.
_STRUCTURE_ID_KEYS = (
    "source", "pdb_id", "uniprot_id", "chain", "full_length",
    "ecl2_range", "head_residues", "head_candidates",
    "n_glycan_positions", "disulfide_cys", "ccg_position",
    "prepared_template", "prepared_path",
)


def _canon(v):
    """Order-stable, JSON-safe canonicalisation (sets sorted, tuples as lists)."""
    if isinstance(v, (set, frozenset)):
        return {"__set__": sorted(map(str, v))}
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(_canon(obj), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _section(per_key: dict) -> dict:
    return {"digest": _digest(per_key), "per_key": per_key}


def compute_run_fingerprint(target_names) -> dict:
    """Identity of everything this run's numbers depend on.

    Four sections, each with a per-key digest map so a mismatch can name the
    exact target / monomer / config symbol that moved:

      structures — sha256 of each target's prepared template file (or its
                   PDB/UniProt id when no prepared chimera is declared)
      targets    — the epitope-defining fields of TARGETS
      monomers   — the name→SMILES map of ALL_MONOMERS
      config     — every non-volatile public config symbol
    """
    from . import config as cfg

    # M1: iterate the FULL TARGETS set, not just this CLI's `target_names`.
    # Otherwise `--target CD63` after an earlier `--target CD63 CD81 CD9`
    # produces different structures/targets sections purely because the CLI
    # narrowed, and resume refuses as drift when nothing actually drifted.
    # The subset run's coverage is separately enforced by _phase_targets_covered.
    structures, targets_id = {}, {}
    for t in sorted(TARGETS.keys()):
        tcfg = TARGETS.get(t)
        if tcfg is None:
            structures[t] = "ABSENT_FROM_TARGETS"
            targets_id[t] = "ABSENT_FROM_TARGETS"
            continue
        targets_id[t] = _digest({k: tcfg.get(k) for k in _STRUCTURE_ID_KEYS})
        prepared = tcfg.get("prepared_path")
        if prepared:
            pp = Path(str(prepared))
            structures[t] = (f"sha256:{_file_sha256(pp)}" if pp.is_file()
                             else f"MISSING:{pp}")
        else:
            structures[t] = "id:" + str(
                tcfg.get("pdb_id") or tcfg.get("uniprot_id") or "unknown")

    monomers = {name: hashlib.sha256(
        str(info.get("smiles", "")).encode("utf-8")).hexdigest()[:16]
        for name, info in sorted(getattr(cfg, "ALL_MONOMERS", {}).items())}

    # Fingerprint only "plain data" values that actually parameterize the
    # computation — strings/ints/floats/bools/None + std containers + Path.
    # This prevents unrelated import churn (e.g. a top-level `from collections
    # import defaultdict` added to config.py) from invalidating every resume:
    # the imported class/module is no longer walked through `dir(cfg)`.
    _FINGERPRINT_TYPES = (
        bool, int, float, str, bytes, type(None),
        list, tuple, dict, set, frozenset,
        Path,
    )
    config_keys = {}
    for name in dir(cfg):
        if (name.startswith("_")
                or name in _VOLATILE_CONFIG_KEYS
                or name in _DUPLICATE_SECTION_KEYS):
            continue
        val = getattr(cfg, name)
        if callable(val) or inspect.ismodule(val) or inspect.isclass(val):
            continue
        if not isinstance(val, _FINGERPRINT_TYPES):
            continue
        config_keys[name] = _digest(val)[:16]

    sections = {
        "structures": _section(structures),
        "targets": _section(targets_id),
        "monomers": _section(monomers),
        "config": _section(config_keys),
    }
    return {
        "experiment": getattr(cfg, "EXPERIMENT", "CD"),
        "sections": sections,
        "digest": _digest({k: v["digest"] for k, v in sections.items()}),
    }


def _fingerprint_mismatches(stored: dict, current: dict, max_keys: int = 12):
    """Human-readable list of what drifted between two fingerprints."""
    out = []
    if not isinstance(stored, dict) or "sections" not in stored:
        return ["stored fingerprint is missing or malformed"]
    if stored.get("experiment") != current.get("experiment"):
        out.append(f"experiment: stored={stored.get('experiment')!r} "
                   f"now={current.get('experiment')!r}")
    for name, cur in current["sections"].items():
        old = stored["sections"].get(name)
        if old is None:
            out.append(f"{name}: absent from the stored manifest")
            continue
        if old.get("digest") == cur.get("digest"):
            continue
        o, c = old.get("per_key", {}), cur.get("per_key", {})
        added = sorted(set(c) - set(o))
        removed = sorted(set(o) - set(c))
        changed = sorted(k for k in set(o) & set(c) if o[k] != c[k])
        detail = []
        if changed:
            detail.append(f"changed={changed[:max_keys]}"
                          + (f" (+{len(changed)-max_keys} more)"
                             if len(changed) > max_keys else ""))
        if added:
            detail.append(f"added={added[:max_keys]}")
        if removed:
            detail.append(f"removed={removed[:max_keys]}")
        out.append(f"{name}: " + "; ".join(detail or ["digest differs"]))
    return out


def _manifest_path(out_dir) -> Path:
    return Path(out_dir) / "reports" / MANIFEST_NAME


def _manifest_lock_path(out_dir) -> Path:
    return Path(out_dir) / "reports" / (MANIFEST_NAME + ".lock")


def _manifest_lock(out_dir, exclusive: bool = True):
    """OS-level advisory lock so two concurrent --phase N runs don't interleave
    manifest writes.

    Falls back to a no-op context on Windows (no fcntl). tmp-then-replace still
    protects a single write's atomicity; the lock protects the read-modify-
    write sequence across the load → mutate → save cycle.
    """
    import contextlib
    try:
        import fcntl  # POSIX only
    except ImportError:
        return contextlib.nullcontext()
    lock_path = _manifest_lock_path(out_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _ctx():
        fh = open(lock_path, "a+")
        try:
            fcntl.flock(fh.fileno(),
                        fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()
    return _ctx()


def _load_manifest(out_dir):
    p = _manifest_path(out_dir)
    if not p.is_file():
        return None
    with _manifest_lock(out_dir, exclusive=False):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"run manifest unreadable ({p}): {e}")
            return None


def _save_manifest(out_dir, manifest: dict) -> None:
    p = _manifest_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock(out_dir, exclusive=True):
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, default=str),
                        encoding="utf-8")
        tmp.replace(p)


def _new_manifest(out_dir, target_names, fingerprint) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": now,
        "updated_utc": now,
        "experiment": fingerprint["experiment"],
        "output_dir": str(Path(out_dir).resolve()),
        "targets": list(target_names),
        "fingerprint": fingerprint,
        "phases": {},
    }


def _record_phase_completion(manifest, out_dir, phase_num, target_names,
                             fingerprint, provenance: str = "computed") -> None:
    """Record a phase as completed, pinning the hash of the file it produced.

    Storing the result-file hash is what makes a later resume able to notice a
    hand-edited or truncated JSON, which "the file exists" never could.

    `provenance` distinguishes phases actually computed under this manifest
    ("computed") from phases inherited by --adopt-existing-tree ("adopted").
    Adopted records preserve a permanent audit trail that this phase's fingerprint
    match is the USER'S ASSERTION, not a verified computation — otherwise one
    accidental --adopt-existing-tree launders every subsequent resume into
    looking cleanly reproducible.
    """
    rel = _PHASE_RESULT_FILES.get(phase_num)
    path = Path(out_dir) / rel if rel else None
    if path is None or not path.is_file():
        logger.error(
            f"Phase {phase_num} finished but produced no {rel} — NOT recording it "
            f"as complete; the next run will re-run it.")
        manifest["phases"].pop(str(phase_num), None)
        _save_manifest(out_dir, manifest)
        return
    manifest["phases"][str(phase_num)] = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "result_file": rel,
        "result_sha256": _file_sha256(path),
        "targets": list(target_names),
        "fingerprint_digest": fingerprint["digest"],
        "provenance": provenance,
    }
    _save_manifest(out_dir, manifest)


# ════════════════════════════════════════════════════════════════
# Resume validation
# ════════════════════════════════════════════════════════════════
def _phase_targets_covered(phase_num: int, path: Path, target_names) -> str:
    """Return '' if the JSON covers every requested target, else the reason.

    Handles known layouts:
      * flat: {target: entry, ...}                — Phase 1, 3, 4, 5, 6
      * Phase 2: {"filtered": {target: entry, ...}, ...}
    Also falls back to a one-level-deep search so a future phase that nests
    per-target payload under a different key (e.g. "results") is detected as
    covering the targets rather than silently reading as flat-and-empty.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        return f"result JSON unparseable: {e}"
    if not isinstance(data, dict):
        return "result JSON is not an object"
    # Phase 2 is the only phase that CURRENTLY nests targets under a specific
    # key. If any target is missing at top level, look one level deeper for a
    # dict whose keys include the target — that catches nested-payload phases
    # without special-casing each one.
    candidate_containers = [data]
    if phase_num == 2 and isinstance(data.get("filtered"), dict):
        candidate_containers.insert(0, data["filtered"])
    for v in data.values():
        if isinstance(v, dict) and any(t in v for t in target_names):
            candidate_containers.append(v)

    missing = list(target_names)
    for cont in candidate_containers:
        still_missing = []
        for t in missing:
            entry = cont.get(t)
            if entry is None or (isinstance(entry, (dict, list)) and not entry):
                still_missing.append(t)
        missing = still_missing
        if not missing:
            break
    if missing:
        return f"target(s) {missing} missing or empty in result JSON"
    return ""


def _check_phase_completed(phase_num: int, output_dir: str, target_names,
                           manifest, fingerprint) -> tuple:
    """(completed, reason). Existence alone is NEVER sufficient.

    A phase is complete only when ALL of these hold:
      1. the result file exists,
      2. the manifest records the phase,
      3. the file still hashes to the recorded value (no hand edits, no
         truncated write from a killed job),
      4. the recorded run covered every target this run asks for,
      5. the phase was produced under the current input fingerprint,
      6. the JSON actually carries a non-empty entry for each target.
    """
    rel = _PHASE_RESULT_FILES.get(phase_num)
    if rel is None:
        return False, "no result file defined for this phase"
    path = Path(output_dir) / rel
    if not path.exists():
        return False, f"{rel} does not exist"
    if not manifest:
        return False, "no run manifest"
    rec = (manifest.get("phases") or {}).get(str(phase_num))
    if not rec:
        return False, f"manifest has no completion record for phase {phase_num}"
    try:
        actual = _file_sha256(path)
    except Exception as e:
        return False, f"cannot hash {rel}: {e}"
    if rec.get("result_sha256") != actual:
        return False, (f"{rel} has changed since it was recorded "
                       f"(sha256 {actual[:12]}… != {str(rec.get('result_sha256'))[:12]}…)")
    missing = [t for t in target_names if t not in (rec.get("targets") or [])]
    if missing:
        return False, f"recorded run did not cover {missing}"
    if rec.get("fingerprint_digest") != fingerprint["digest"]:
        return False, "produced under a different input fingerprint"
    reason = _phase_targets_covered(phase_num, path, target_names)
    if reason:
        return False, reason
    return True, ""


# Which earlier phases each phase's entry point actually consumes — mirrors
# the kwargs in run_phase() exactly.
_PHASE_DEPS = {1: (), 2: (1,), 3: (1, 2), 4: (1, 3), 5: (1, 4), 6: (3, 4, 5)}


def _load_upstream(prev_results: dict, phases, out_dir, target_names,
                   manifest, fingerprint, args=None) -> None:
    """Load the results of phases this run will NOT execute but does consume.

    `--phase 4` used to hand run_phase4() phase1_results=None and
    phase3_results=None: the old resume loop only ever considered the phases
    in `phases`, and the run "worked" solely because Phase 4 itself was found
    on disk and skipped. The moment it actually had to run, it ran against
    None. Upstream results are now loaded and validated, or the run refuses.

    Also runs the verify_phase.py gate on every loaded upstream — without this,
    `--phase 5` would consume Phase 1/2/3/4 results without ever re-checking
    that they still pass their own verifier (e.g. a Phase 4 whose all three
    legs were non-converged would silently feed Phase 5).
    """
    to_run = set(phases)
    needed = sorted({d for n in to_run for d in _PHASE_DEPS[n]} - to_run)
    for n in needed:
        if f"phase{n}" in prev_results:
            continue
        ok, reason = _check_phase_completed(
            n, out_dir, target_names, manifest, fingerprint)
        if not ok:
            _die(EXIT_RESUME_REFUSED,
                 f"phase(s) {sorted(to_run)} consume Phase {n}'s results, but "
                 f"they are not usable: {reason}.",
                 f"Run Phase {n} first (e.g. --phase all), or point "
                 f"--output-dir at the tree that has it.")
        prev_results[f"phase{n}"] = _load_phase_result(n, out_dir)
        logger.info(f"Phase {n}: LOADED as an input to {sorted(to_run)}")
        # Verify the loaded upstream — same gate that resumed phases pass.
        if args is not None:
            _verify_or_stop(n, target_names, out_dir, args)


def _load_phase_result(phase_num: int, output_dir: str):
    """Load existing phase result from disk."""
    result_file = _PHASE_RESULT_FILES.get(phase_num)
    if result_file is None:
        return None
    path = Path(output_dir) / result_file
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _tree_has_results(out_dir) -> list:
    return [rel for rel in _PHASE_RESULT_FILES.values()
            if (Path(out_dir) / rel).is_file()]


def _die(code: int, *lines) -> None:
    """Log the refusal and exit non-zero. Never returns.

    Falls back to stderr when logging is not configured yet — several refusals
    fire before basicConfig(), and a refusal nobody can read is a silent one.
    """
    configured = bool(logging.getLogger().handlers)
    for line in lines:
        if configured:
            logger.error(line)
        else:
            # logging's lastResort handler already writes to stderr, so calling
            # logger.error() here too would double every line of the refusal.
            print(line, file=sys.stderr)
    sys.exit(code)


# ════════════════════════════════════════════════════════════════
# --fresh: archive or delete before running
# ════════════════════════════════════════════════════════════════
def _phase_dirs_for(out_dir, phases, include_reports: bool) -> list:
    dirs = [Path(out_dir) / f"phase{n}" for n in phases]
    if include_reports:
        dirs.append(Path(out_dir) / "reports")
    return [d for d in dirs if d.exists()]


def _do_fresh(out_dir, phases, args) -> dict:
    """Archive (default) or delete the directories --fresh is about to rewrite.

    THIS IS THE POINT OF --fresh. The old implementation set resume=False and
    left every byte on disk, so Phases 3/4/5 — which resume from their own
    per-target JSONs regardless of any flag — silently reused stale results.
    Removing the directory is the only disposal that all three phases obey,
    because none of them can resume from a directory that is not there.

    Refuses (exit 4) rather than proceeding on top of state it could not move.
    """
    include_reports = len(phases) == len(_PHASE_RESULT_FILES)
    victims = _phase_dirs_for(out_dir, phases, include_reports)
    plan = {"mode": args.fresh_mode, "dirs": [str(d) for d in victims],
            "archive_dir": None}
    if not victims:
        logger.info("--fresh: nothing to clear (no existing phase directories)")
        return plan

    if args.fresh_mode == "delete":
        if not args.yes_delete:
            _die(EXIT_FRESH_REFUSED,
                 "--fresh --fresh-mode delete REFUSED: irreversible deletion of",
                 *[f"    {d}" for d in plan["dirs"]],
                 "Re-run with --yes-delete to confirm, or drop --fresh-mode "
                 "delete to archive instead (the default, an os.rename).")
        for d in victims:
            logger.warning(f"--fresh: DELETING {d}")
            shutil.rmtree(d)
        logger.warning(f"--fresh: deleted {len(victims)} director(ies)")
        return plan

    # archive (default): os.rename is atomic and instant on the same filesystem,
    # which matters when the tree is hundreds of GB.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    root = Path(args.archive_dir) if args.archive_dir else Path(
        str(Path(out_dir).resolve()) + f".archive_{stamp}")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        _die(EXIT_FRESH_REFUSED,
             f"--fresh REFUSED: cannot create archive directory {root}: {e}")
    for d in victims:
        dest = root / d.name
        if dest.exists():
            _die(EXIT_FRESH_REFUSED,
                 f"--fresh REFUSED: archive destination already exists: {dest}")
        try:
            os.rename(d, dest)
        except OSError as e:
            _die(EXIT_FRESH_REFUSED,
                 f"--fresh REFUSED: could not archive {d} → {dest}: {e}",
                 "Nothing was cleared, so the run would have re-used stale "
                 "state. Move the tree by hand (e.g. `mv` to another disk) and "
                 "re-run, or use --archive-dir on the SAME filesystem.")
        logger.warning(f"--fresh: archived {d} → {dest}")
    plan["archive_dir"] = str(root)
    logger.warning(f"--fresh: archived {len(victims)} director(ies) to {root}")
    return plan


# ════════════════════════════════════════════════════════════════
# Phase execution
# ════════════════════════════════════════════════════════════════
def _call_phase(fn, kwargs: dict, fresh: bool, phase_num: int):
    """Call a phase entry point, passing fresh= when its signature accepts it.

    Phases 3/4/5 own an internal per-target resume that no flag currently
    reaches. Until their signatures grow `fresh`, the guarantee comes from
    _do_fresh() having removed the directory they resume from — so this
    reports the gap loudly instead of implying a flag was honoured.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "fresh" in params:
        kwargs["fresh"] = fresh
    elif fresh:
        logger.warning(
            f"--fresh: {fn.__module__}.{fn.__name__}() does not accept fresh=; "
            f"Phase {phase_num}'s internal per-target resume is disarmed ONLY "
            f"because its output directory was archived/deleted above.")
    return fn(**kwargs)


def run_phase(phase_num: int, target_names: list, output_dir: str,
              prev_results: dict, args) -> dict:
    """Run a single pipeline phase and return its results."""
    t0 = time.time()
    fresh = bool(getattr(args, "fresh", False))

    if phase_num == 1:
        from .phase1_epitope_prep import run_phase1
        result = _call_phase(run_phase1, dict(
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase1"),
            skip_stability_md=args.skip_md,
        ), fresh, 1)

    elif phase_num == 2:
        from .phase2_smd import run_phase2
        result = _call_phase(run_phase2, dict(
            phase1_results=prev_results.get("phase1"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase2"),
        ), fresh, 2)

    elif phase_num == 3:
        from .phase3_mmsd import run_phase3
        result = _call_phase(run_phase3, dict(
            phase1_results=prev_results.get("phase1"),
            phase2_results=prev_results.get("phase2"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase3"),
        ), fresh, 3)

    elif phase_num == 4:
        from .phase4_md_validation import run_phase4
        result = _call_phase(run_phase4, dict(
            phase1_results=prev_results.get("phase1"),
            phase3_results=prev_results.get("phase3"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase4"),
            quick=args.quick_md,
            cross_reactivity=not args.no_cross_reactivity,
        ), fresh, 4)

    elif phase_num == 5:
        from .phase5_rebinding import run_phase5_rebinding
        result = _call_phase(run_phase5_rebinding, dict(
            phase4_results=prev_results.get("phase4"),
            phase1_results=prev_results.get("phase1"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase5"),
        ), fresh, 5)

    elif phase_num == 6:
        from .phase6_recipe import run_phase6_recipe
        result = _call_phase(run_phase6_recipe, dict(
            phase3_results=prev_results.get("phase3"),
            phase4_results=prev_results.get("phase4"),
            phase5_results=prev_results.get("phase5"),
            target_names=target_names,
            output_dir=_phase_dir(output_dir, "phase6"),
        ), fresh, 6)

    else:
        raise ValueError(f"unknown phase {phase_num}")

    elapsed = time.time() - t0
    logger.info(f"Phase {phase_num} completed in {elapsed:.1f}s")
    return {"result": result, "elapsed_s": round(elapsed, 1)}


# ════════════════════════════════════════════════════════════════
# Verification gate
# ════════════════════════════════════════════════════════════════
def _note_unverified(out_dir, phase_num: int) -> None:
    """Persist the fact that a phase was skipped by --skip-verify.

    The flag lives in one shell invocation; the results live for months. The
    tree itself has to carry the record that it was never checked.
    """
    manifest = _load_manifest(out_dir)
    if manifest is None:
        return
    rec = manifest.setdefault("unverified_phases", [])
    if phase_num not in rec:
        rec.append(phase_num)
        _save_manifest(out_dir, manifest)


def _verify_or_stop(phase_num: int, target_names: list, out_dir: str,
                    args) -> None:
    """Run verify_phase.py for a phase and STOP the pipeline on a block.

    Runs for resumed phases too: a tree that failed verification yesterday must
    not become acceptable merely by being loaded from disk today.
    """
    if args.skip_verify:
        logger.warning(
            f"--skip-verify: Phase {phase_num} NOT verified. A failed phase "
            f"will not stop this run.")
        _note_unverified(out_dir, phase_num)
        return
    try:
        from . import verify_phase
    except Exception as e:
        _die(EXIT_VERIFY_BLOCKED,
             f"Phase {phase_num}: pipeline.verify_phase could not be imported "
             f"({e}). Refusing to continue unverified — pass --skip-verify to override.")
    try:
        report = verify_phase.main(phase_num, targets=target_names,
                                   results_root=out_dir, quiet=True)
    except Exception as e:
        _die(EXIT_VERIFY_BLOCKED,
             f"Phase {phase_num}: verification itself failed ({e!r}). Refusing "
             f"to continue unverified — pass --skip-verify to override.")

    for w in report.get("warnings", []):
        logger.warning(f"[verify P{phase_num}] {w}")
    if report.get("blocked"):
        lines = [f"Phase {phase_num} FAILED VERIFICATION "
                 f"(decision={report.get('decision')}):"]
        lines += [f"    - {i}" for i in report.get("issues", [])]
        lines.append(f"    see {Path(out_dir)/'reports'/f'phase{phase_num}_summary.md'}")
        lines.append("    Fix the phase, or re-run with --skip-verify to "
                     "proceed on results that did not pass.")
        _die(EXIT_VERIFY_BLOCKED, *lines)
    logger.info(f"[verify P{phase_num}] PASS — {report.get('decision')}")


def print_summary(output_dir: str):
    """Print final summary of pipeline results."""
    out = Path(output_dir)
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY — Epitope-MIP Screening Results")
    logger.info(f"{'='*60}")

    # Phase 2: SMD summary
    p2 = out / "phase2" / "phase2_smd_results.json"
    if p2.exists():
        with open(p2) as f:
            smd = json.load(f)
        logger.info("\n[Phase 2] SMD Filtered Monomers:")
        for target, monomers in smd.get("filtered", {}).items():
            logger.info(f"  {target}: {monomers}")

    # Phase 3: MMSD top PCs
    p3 = out / "phase3" / "phase3_mmsd_results.json"
    if p3.exists():
        with open(p3) as f:
            mmsd = json.load(f)
        logger.info("\n[Phase 3] Top Polymer Compositions:")
        for target, data in mmsd.items():
            if isinstance(data, dict) and "top_pcs" in data:
                for pc in data["top_pcs"][:3]:
                    logger.info(
                        f"  {target} {pc['pc_id']}: "
                        f"{pc['monomers']} "
                        f"(MMSD={pc.get('mmsd_sum', 'N/A'):.2f})"
                    )

    # Phase 6: Recipes.
    # FIXED: this used to read phase5/phase5_recipes.json — a path no phase has
    # ever written, so the recipe summary was silently always empty.
    p6 = out / "phase6" / "phase6_recipes.json"
    if p6.exists():
        with open(p6) as f:
            recipes = json.load(f)
        logger.info("\n[Phase 6] Recommended Recipes:")
        for target, recipe in recipes.items():
            if not isinstance(recipe, dict):
                continue
            monomers = list(recipe.get("monomers", {}).keys())
            logger.info(f"  {target}: {monomers} "
                        f"({recipe.get('polymerization_type', 'N/A')})")

    logger.info(f"\n{'='*60}")


# ════════════════════════════════════════════════════════════════
def _resolve_state_flags(args) -> None:
    """Turn the raw flags into an effective, non-contradictory state policy."""
    if args.fresh and args.resume:
        _die(EXIT_FRESH_REFUSED,
             "--fresh and --resume are mutually exclusive: one re-runs from "
             "scratch, the other re-uses what is on disk. Pick one.")
    if args.no_resume and not args.fresh:
        _die(EXIT_RESUME_REFUSED,
             "--no-resume REFUSED without --fresh.",
             "It only disables the TOP-LEVEL phase skip. Phases 3, 4 and 5 each "
             "resume from their own per-target JSON inside their output "
             "directory, and no flag reaches that logic — so --no-resume alone "
             "silently re-uses stale per-target results while claiming a "
             "re-run. Use --fresh (archives or deletes those directories).")
    # Effective policy: resume unless a fresh run was requested.
    args.resume = not args.fresh


def main(argv=None):
    args = parse_args(argv)

    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Determine targets/phases early — --fresh disposal depends on them.
    if "all" in args.target:
        target_names = list(TARGETS.keys())
    else:
        target_names = args.target
    unknown = [t for t in target_names if t not in TARGETS]
    if unknown:
        _die(EXIT_RESUME_REFUSED,
             f"unknown target(s) {unknown}; known targets are {list(TARGETS)}")
    phases = [1, 2, 3, 4, 5, 6] if args.phase == "all" else [int(args.phase)]

    mode_only = (args.report or args.extend_drifters or args.multirestart
                 or args.reanalyze)
    if mode_only and (args.fresh or args.no_resume):
        _die(EXIT_FRESH_REFUSED,
             "--fresh/--no-resume have no meaning with --report / "
             "--extend-drifters / --multirestart / --reanalyze: those modes "
             "read an existing tree. Refusing rather than ignoring the flag.")
    if not mode_only:
        _resolve_state_flags(args)
        # Disposal happens BEFORE the log file is opened, so reports/ can be
        # archived along with everything else.
        if args.fresh:
            _do_fresh(out_dir, phases, args)

    (Path(out_dir) / "reports").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Pipeline] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                str(Path(out_dir) / "reports" / "pipeline.log")
            ),
        ],
    )

    # Convergence / report modes still MUTATE (extend/multirestart/reanalyze
    # rewrite Phase 5 data; --report synthesizes over Phase 4-5). They must
    # therefore run against a matching-input tree — the exact case the
    # manifest exists to protect. Validate now, before dispatching.
    if mode_only:
        current_fp = compute_run_fingerprint(target_names)
        stored_manifest = _load_manifest(out_dir)
        if stored_manifest is None:
            _die(EXIT_RESUME_REFUSED,
                 f"{out_dir} has no {MANIFEST_NAME}: cannot run a convergence "
                 f"/ report mode against an un-fingerprinted tree.",
                 "Run the pipeline once (`--phase all`) to establish a "
                 "manifest, or --adopt-existing-tree to assert this tree "
                 "matches the current inputs.")
        drift = _fingerprint_mismatches(
            stored_manifest.get("fingerprint", {}), current_fp)
        if drift and not args.accept_input_drift:
            lines = [f"{out_dir} was built from DIFFERENT inputs:"]
            lines += [f"    - {d}" for d in drift]
            _die(EXIT_RESUME_REFUSED, *lines,
                 "Refusing to run --report / --extend-drifters / --multirestart "
                 "/ --reanalyze on a mixed-input tree.",
                 "Choose one:",
                 "  --fresh                 archive the tree and re-run",
                 "  --accept-input-drift    proceed anyway (UNSAFE)")
        elif drift:
            for line in ["--accept-input-drift active for convergence mode:"] + drift:
                logger.warning(line)

    # M11: smoke test — every convergence/report mode consumes Phase 4 (and
    # 5) outputs. If those artefacts are absent, the mode's own error will
    # bury the real reason. Fail early with an actionable message.
    if mode_only:
        needed_files = {
            "report":            [_PHASE_RESULT_FILES[5]],  # generate_report reads P5
            "extend_drifters":   [_PHASE_RESULT_FILES[4], _PHASE_RESULT_FILES[5]],
            "multirestart":      [_PHASE_RESULT_FILES[4], _PHASE_RESULT_FILES[5]],
            "reanalyze":         [_PHASE_RESULT_FILES[5]],
        }
        active_mode = next(m for m in needed_files if getattr(args, m, False))
        missing = [rel for rel in needed_files[active_mode]
                   if not (Path(out_dir) / rel).is_file()]
        if missing:
            _die(EXIT_RESUME_REFUSED,
                 f"--{active_mode.replace('_', '-')} needs upstream outputs "
                 f"that are absent: " + ", ".join(missing),
                 "Run the standard pipeline (`--phase all`) far enough to "
                 "produce those files before invoking a convergence mode.")

    # Report-only mode
    if args.report:
        from .generate_report import generate_report
        path = generate_report(output_dir=out_dir)
        logger.info(f"Report generated: {path}")
        return 0

    # Convergence-improvement modes (run instead of standard pipeline)
    if args.extend_drifters:
        from .phase5_rebinding import extend_drifting_mds
        logger.info(f"Extension mode: +{args.extend_ns} ns for drifting Phase 5 MDs")
        # output_dir=None lets the function auto-detect phase5_extended over phase5
        extend_drifting_mds(
            target_names=target_names,
            output_dir=None,
            extend_ns=args.extend_ns,
        )
        return 0

    if args.multirestart:
        from .phase5_rebinding import run_multirestart
        logger.info(f"Multi-restart mode: N={args.n_reps} replicates per snapshot")
        run_multirestart(
            target_names=target_names,
            n_reps=args.n_reps,
            output_dir=None,
        )
        return 0

    if args.reanalyze:
        from .phase5_rebinding import reanalyze_phase5
        logger.info("Re-analysis mode: PBC centering + Q4 RMSD on existing trajectories")
        reanalyze_phase5(
            target_names=target_names,
            output_dir=None,
        )
        return 0

    logger.info(f"Targets: {target_names}")
    logger.info(f"Output: {out_dir}")

    # ── run manifest: what these results are allowed to be resumed from ──
    fingerprint = compute_run_fingerprint(target_names)
    manifest = _load_manifest(out_dir)
    existing = _tree_has_results(out_dir)

    if manifest is None:
        if existing and not args.adopt_existing_tree:
            _die(EXIT_RESUME_REFUSED,
                 f"{out_dir} already holds phase results but has no "
                 f"{MANIFEST_NAME}: " + ", ".join(existing),
                 "There is no record of the structures, monomer SMILES or "
                 "config those results were produced from, so resuming them "
                 "cannot be checked — and 'the file exists' is exactly the test "
                 "that let a stale tree pass as a completed run.",
                 "Choose one:",
                 "  --fresh                             archive them and re-run (safe)",
                 "  --adopt-existing-tree --yes-adopt   assert they match CURRENT inputs")
        if existing and args.adopt_existing_tree and not args.yes_adopt:
            _die(EXIT_RESUME_REFUSED,
                 "--adopt-existing-tree requires --yes-adopt.",
                 "Adopting stamps every pre-existing phase file with the "
                 "CURRENT input fingerprint, so all future resumes will look "
                 "clean without any evidence that the tree was actually built "
                 "from these inputs. Confirm with --yes-adopt if that is what "
                 "you want; the adoption is recorded permanently in the manifest.")
        manifest = _new_manifest(out_dir, target_names, fingerprint)
        if existing and args.adopt_existing_tree:
            logger.warning(
                "--adopt-existing-tree --yes-adopt: adopting %d pre-existing "
                "result file(s) under the CURRENT input fingerprint %s. "
                "Recording provenance='adopted' so this stays visible in the "
                "manifest for every future audit.",
                len(existing), fingerprint["digest"][:12])
            manifest.setdefault("adoption_events", []).append({
                "utc": datetime.now(timezone.utc).isoformat(),
                "adopted_files": list(existing),
                "fingerprint_digest": fingerprint["digest"],
            })
            for rel in existing:
                n = next(k for k, v in _PHASE_RESULT_FILES.items() if v == rel)
                _record_phase_completion(manifest, out_dir, n, target_names,
                                         fingerprint, provenance="adopted")
        elif not existing:
            # H7: silent-fresh path. Log clearly that this is a new run.
            logger.info(
                "No %s and no phase results on disk — treating as a NEW run. "
                "A fresh manifest with fingerprint %s will be written after "
                "Phase 1 completes.", MANIFEST_NAME, fingerprint["digest"][:12])
        _save_manifest(out_dir, manifest)
    else:
        drift = _fingerprint_mismatches(manifest.get("fingerprint", {}), fingerprint)
        if drift:
            lines = [f"{out_dir} was built from DIFFERENT inputs:"]
            lines += [f"    - {d}" for d in drift]
            if not args.accept_input_drift:
                _die(EXIT_RESUME_REFUSED, *lines,
                     "Resuming would mix results computed from different "
                     "structures/monomers/settings into one report.",
                     "Choose one:",
                     "  --fresh                 archive the tree and re-run",
                     "  --accept-input-drift    proceed anyway (UNSAFE)")
            for line in lines:
                logger.warning("--accept-input-drift: " + line)
            manifest["fingerprint"] = fingerprint
            manifest.setdefault("input_drift_accepted", []).append({
                "utc": datetime.now(timezone.utc).isoformat(),
                "mismatches": drift,
            })
        missing_t = [t for t in target_names if t not in (manifest.get("targets") or [])]
        if missing_t:
            manifest["targets"] = sorted(set(manifest.get("targets", [])) | set(target_names))
        _save_manifest(out_dir, manifest)

    t_total = time.time()
    timings = {}
    prev_results = {}

    if args.fresh:
        # Completion records for the cleared phases are now lies. Also drop
        # every downstream phase — a re-computed Phase 4 invalidates the
        # Phase 5 that consumed it, even if Phase 5's own manifest entry
        # looks fine. Without this cascade, `--phase 4 --fresh` leaves
        # `phases['5']` in manifest with a fingerprint that no longer matches
        # its upstream Phase 4.
        min_cleared = min(phases)
        stale = [n for n in _PHASE_RESULT_FILES.keys() if n >= min_cleared]
        for n in stale:
            manifest["phases"].pop(str(n), None)
        _save_manifest(out_dir, manifest)
        downstream = sorted(set(stale) - set(phases))
        if downstream:
            logger.warning(
                f"Fresh run: phases {phases} cleared, downstream {downstream} "
                f"also dropped from manifest (they depend on the re-run phases)")
        else:
            logger.warning(f"Fresh run: phases {phases} cleared and will be recomputed")

    resumed = set()
    if args.resume:
        for phase_num in phases:
            ok, reason = _check_phase_completed(
                phase_num, out_dir, target_names, manifest, fingerprint)
            if ok:
                prev_results[f"phase{phase_num}"] = _load_phase_result(
                    phase_num, out_dir)
                resumed.add(phase_num)
                logger.info(f"Phase {phase_num}: LOADED from existing results "
                            f"({_PHASE_RESULT_FILES[phase_num]})")
            else:
                logger.info(f"Phase {phase_num}: will RUN — {reason}")
                break  # run this phase and all subsequent

    # Anything about to RUN needs its upstream inputs, even when those phases
    # are not part of this invocation (`--phase 4`). Passing args so verify
    # runs on loaded upstreams too — a stale-but-hash-matching Phase 3 would
    # otherwise silently feed Phase 5.
    _load_upstream(prev_results, [n for n in phases if n not in resumed],
                   out_dir, target_names, manifest, fingerprint, args=args)

    reports_dir = Path(out_dir) / "reports"
    for phase_num in phases:
        if phase_num in resumed:
            timings[f"phase{phase_num}"] = 0.0
        else:
            # M7: delete any stale verify artefacts for this phase before it
            # runs. Otherwise a phase that BLOCKED yesterday and is being
            # re-run today leaves the old BLOCKED JSON on disk until the new
            # verify overwrites it — anyone who reads results/reports/ in
            # between sees a lie.
            for stale in (
                    reports_dir / f"phase{phase_num}_verification.json",
                    reports_dir / f"phase{phase_num}_summary.md"):
                if stale.exists():
                    stale.unlink()
                    logger.info(f"Cleared stale {stale.name} before re-running Phase {phase_num}")

            logger.info(f"\n{'='*20} PHASE {phase_num} {'='*20}")
            phase_output = run_phase(
                phase_num, target_names, out_dir, prev_results, args
            )
            timings[f"phase{phase_num}"] = phase_output["elapsed_s"]
            prev_results[f"phase{phase_num}"] = phase_output["result"]
            _record_phase_completion(manifest, out_dir, phase_num,
                                     target_names, fingerprint)
        # Verification gate — resumed phases included.
        _verify_or_stop(phase_num, target_names, out_dir, args)

    total_time = time.time() - t_total

    # Timing summary
    logger.info(f"\n--- Timing Summary ---")
    for name, t in timings.items():
        logger.info(f"  {name}: {t:.1f}s")
    logger.info(f"  Total: {total_time:.1f}s")

    # Final summary
    print_summary(out_dir)

    # Auto-generate report
    if args.phase == "all":
        try:
            from .generate_report import generate_report
            path = generate_report(output_dir=out_dir)
            logger.info(f"Report: {path}")
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
    return 0


if __name__ == "__main__":
    # `python -m pipeline.run_pipeline` bypasses the run_BSA.py / run_CD.py
    # wrappers, so it needs its own UTF-8 guarantee — .mdp/.top writes encode
    # with the ambient locale, and an ASCII one aborts the run mid-MD.  Re-exec
    # with -m (NOT as a plain script path): this module uses relative imports,
    # so running it by path would break `from . import config`.
    import os as _os
    if not sys.flags.utf8_mode and not _os.environ.get("_MIP_UTF8_REEXEC"):
        _os.environ["_MIP_UTF8_REEXEC"] = "1"
        _os.execve(sys.executable,
                   [sys.executable, "-X", "utf8", "-m", "pipeline.run_pipeline",
                    *sys.argv[1:]], _os.environ)
    sys.exit(main() or 0)
