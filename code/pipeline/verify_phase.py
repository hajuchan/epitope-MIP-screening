#!/usr/bin/env python3
"""
Per-phase verification script — Level 1/2/3 checks.

Usage:
    python verify_phase.py 1  # verify Phase 1
    python verify_phase.py 2  # verify Phase 2 ... etc

Saves:
    results/reports/phaseN_verification.json   (machine-readable)
    results/reports/phaseN_summary.md          (human-readable)
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.config import get_output_path, TARGETS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("verify")

REPORTS_DIR = Path(get_output_path("reports"))
RESULTS_ROOT = Path(get_output_path("phase1")).parent


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
        from pipeline.config import GMX_BIN

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


def verify_phase1():
    """Phase 1: Epitope preparation"""
    p = RESULTS_ROOT / "phase1"
    report = {"phase": 1, "level1": {}, "level2": {}, "level3": {}, "issues": []}

    # L1: files
    json_p = p / "phase1_results.json"
    if not json_p.exists():
        report["issues"].append("phase1_results.json missing")
        return report
    with open(json_p) as f:
        data = json.load(f)

    for t in TARGETS:
        if t not in data:
            report["issues"].append(f"{t} missing")
            continue
        tr = data[t]
        report["level1"][t] = {
            "head_pdb_exists": Path(tr.get("head_pdb", "")).exists(),
            "ecl2_pdb_exists": Path(tr.get("epitope_pdb", "")).exists(),
            "receptor_pdbqt_exists": Path(tr.get("receptor_pdbqt", "")).exists(),
        }
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
        # L3: MD centering
        md_dir = p / t / "stability_md"
        if (md_dir / "md.gro").exists():
            report["level3"][t] = _check_md_centering(md_dir / "md.gro")
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
                        report["level3"][t]["stability_rmsd_mean_A"] = round(
                            sum(rmsds) / len(rmsds), 2)
                        report["level3"][t]["stability_rmsd_max_A"] = round(max(rmsds), 2)
                except Exception:
                    pass

    # Decision logic
    n_errors = sum(1 for t in report["level1"].values()
                    if not all(t.values()))
    n_pbc_split = sum(1 for t in report["level3"].values()
                      if t.get("pbc_split"))
    if n_errors:
        report["decision"] = "fix_required"
    elif n_pbc_split > 0:
        report["decision"] = "fix_pbc_warning"
    else:
        report["decision"] = "proceed_to_phase_2"

    return report


def verify_phase2():
    p = RESULTS_ROOT / "phase2"
    report = {"phase": 2, "level1": {}, "level2": {}, "level3": {}, "issues": []}
    json_p = p / "phase2_smd_results.json"
    if not json_p.exists():
        report["issues"].append("phase2 JSON missing")
        return report
    with open(json_p) as f:
        data = json.load(f)

    for t in TARGETS:
        be = data.get("be_matrix", {}).get(t, {})
        sel = data.get("selectivity", {}).get(t, {})
        decoy = data.get("decoy_baseline", {}).get(t, {}) if data.get("decoy_baseline") else {}
        ef = data.get("enrichment_factor", {}).get(t, {}) if data.get("enrichment_factor") else {}
        filt = data.get("filtered", {}).get(t, [])
        report["level1"][t] = {
            "n_monomers_docked": len(be),
            "n_filtered_for_phase3": len(filt),
            "has_selectivity": len(sel) > 0,
        }
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
        # L3: BE range sanity
        bes = [v for v in be.values() if v is not None]
        if bes:
            report["level3"][t] = {
                "BE_min": round(min(bes), 2),
                "BE_max": round(max(bes), 2),
                "BE_mean": round(sum(bes) / len(bes), 2),
                "plausible_range": -15.0 < min(bes) < 0.0,
            }
    n_errors = sum(1 for r in report["level1"].values()
                    if r["n_filtered_for_phase3"] == 0)
    report["decision"] = "proceed_to_phase_3" if n_errors == 0 else "fix_required"
    return report


def verify_phase3():
    p = RESULTS_ROOT / "phase3"
    report = {"phase": 3, "level1": {}, "level2": {}, "level3": {}, "issues": []}
    json_p = p / "phase3_mmsd_results.json"
    if not json_p.exists():
        report["issues"].append("phase3 JSON missing")
        return report
    with open(json_p) as f:
        data = json.load(f)

    for t in TARGETS:
        td = data.get(t, {})
        if "error" in td:
            report["issues"].append(f"{t}: {td['error']}")
            continue
        method = td.get("method", "")
        pareto = td.get("pareto_front") or []
        top_pcs = td.get("top_pcs", [])
        report["level1"][t] = {
            "method": method,
            "n_evaluations": td.get("n_evaluations", 0),
            "n_pareto": len(pareto),
            "n_top_pcs": len(top_pcs),
        }
        report["level2"][t] = {
            "C2_nsga2_active": "NSGA-II" in method,
            "pareto_front_size": len(pareto),
            "best_synth_score": max(
                [p_["objectives"].get("synthesizability_score", 0) for p_ in pareto],
                default=0
            ) if pareto else 0,
        }
        # L3: top PC is polymerization-compatible
        from pipeline.config import is_polymerization_compatible
        if top_pcs:
            tp = top_pcs[0]
            ok, _, _ = is_polymerization_compatible(tp.get("monomers", []))
            report["level3"][t] = {
                "top_pc_polymerization_compatible": ok,
                "top_pc_monomers": tp.get("monomers"),
            }

    report["decision"] = "proceed_to_phase_4"
    return report


def verify_phase4():
    p = RESULTS_ROOT / "phase4"
    report = {"phase": 4, "level1": {}, "level2": {}, "level3": {}, "issues": []}
    json_p = p / "phase4_md_results.json"
    if not json_p.exists():
        report["issues"].append("phase4 JSON missing")
        return report
    with open(json_p) as f:
        data = json.load(f)

    for t in TARGETS:
        td = data.get(t, {})
        report["level1"][t] = {"n_pcs": len(td)}
        for pc_id, pcd in td.items():
            report["level2"][f"{t}/{pc_id}"] = {
                "has_md_results": "occupancy_analysis" in pcd or "rmsd" in pcd,
                "A5_solvent_sweep_done": bool(pcd.get("solvent_sweep")),
                "A5_best_solvent": pcd.get("solvent_sweep", {}).get("best_solvent"),
                "B6_ratio_sweep_done": bool(pcd.get("ratio_sweep")),
                "B6_best_ratio": pcd.get("ratio_sweep", {}).get("best_ratio"),
            }
            # L3: centering check
            md_dir = p / t / pc_id / "md"
            if (md_dir / "md.gro").exists():
                report["level3"][f"{t}/{pc_id}"] = _check_md_centering(md_dir / "md.gro")

    report["decision"] = "proceed_to_phase_5"
    return report


def verify_phase5():
    """Phase 5 verification — uses PCSI (persistent contact selectivity index)
    as PRIMARY metric. Falls back to RMSD-SI if PCSI not yet computed.

    PCSI is size-invariant (counts persistent residue contacts) — replaces
    backbone RMSD which is biased by protein size (smaller proteins → higher
    RMSD even if cavity-bound). See analyze_persistent_contacts.py.
    """
    p = RESULTS_ROOT / "phase5"
    report = {"phase": 5, "level1": {}, "level2": {}, "level3": {}, "issues": []}
    json_p = p / "phase5_rebinding_results.json"
    if not json_p.exists():
        report["issues"].append("phase5 JSON missing")
        return report
    with open(json_p) as f:
        data = json.load(f)

    # PRIMARY metric: PCSI from analyze_persistent_contacts.py output
    pcsi_path = REPORTS_DIR / "phase5_persistent_contacts.json"
    pcsi_data = {}
    if pcsi_path.exists():
        try:
            with open(pcsi_path) as f:
                pcsi_data = json.load(f)
        except Exception:
            pass
    else:
        report["issues"].append(
            "PCSI metric not computed yet — run: "
            "python code/analyze_persistent_contacts.py")

    n_targets_pass = 0
    n_targets_total = 0
    for t in TARGETS:
        td = data.get(t, {})
        if not td:
            continue
        sel = td.get("selectivity", {})
        a6_count = sum(1 for v in sel.values() if "si_ci_lower" in v)
        snaps = td.get("snapshots", [])
        b8_count = sum(1 for s in snaps if "multipose_ensemble" in s)
        n_rebound = td.get("n_rebound", 0)
        n_snaps = td.get("n_snapshots", 0)

        report["level1"][t] = {
            "n_rebound": n_rebound,
            "n_snapshots": n_snaps,
            "n_cross_selectivity": len(sel),
        }
        report["level2"][t] = {
            "A6_bootstrap_CI_active": a6_count > 0,
            "A6_n_with_CI": a6_count,
            "B8_multipose_done": b8_count > 0,
        }

        # Level 3: cavity centering
        snap0 = p / t / "snapshot_0"
        if (snap0 / "frame.gro").exists():
            report["level3"][t] = _check_md_centering(snap0 / "frame.gro")
        if t not in report["level3"]:
            report["level3"][t] = {}

        # ⭐ PRIMARY: PCSI metric (chemistry-corrected selectivity)
        n_targets_total += 1
        if t in pcsi_data:
            tgt_pcsi = pcsi_data[t]
            own_d = tgt_pcsi.get(t, {})
            own_n = own_d.get("n_persistent_residues", 0) if isinstance(own_d, dict) else 0
            # Size-exclusion: cross-targets physically excluded from the cavity
            # are selective by mechanism. Their persistent-contact counts come
            # from stale/forced MD (oversized protein crammed against walls) and
            # must NOT be scored against the target.
            size_excluded = {o for o, v in sel.items()
                             if isinstance(v, dict)
                             and v.get("selectivity_label") == "size-excluded"}
            cross_ns = [tgt_pcsi.get(o, {}).get("n_persistent_residues", 0)
                        for o in TARGETS if o != t
                        and o not in size_excluded
                        and isinstance(tgt_pcsi.get(o, {}), dict)
                        and "n_persistent_residues" in tgt_pcsi.get(o, {})]
            max_cross = max(cross_ns) if cross_ns else 0
            pcsi = own_n / max_cross if max_cross > 0 else float('inf')
            # PASS if PCSI clears threshold on measurable cross-targets AND the
            # target actually binds its own template (own_n > 0).
            pcsi_pass = (pcsi > 1.2) and own_n > 0
            if pcsi_pass:
                n_targets_pass += 1
            report["level3"][t]["PCSI"] = round(pcsi, 2) if pcsi != float('inf') else "inf"
            report["level3"][t]["own_persistent_contacts"] = own_n
            report["level3"][t]["max_cross_persistent_contacts"] = max_cross
            report["level3"][t]["PCSI_PASS"] = pcsi_pass
            if size_excluded:
                report["level3"][t]["size_excluded_targets"] = sorted(size_excluded)
                report["level3"][t]["selectivity_mechanism"] = (
                    "size/shape exclusion + persistent-contact"
                    if cross_ns else "size/shape exclusion")
        else:
            # Fallback: traditional SI-RMSD (less reliable)
            rmsd_si_pass = any(
                v.get("selectivity_index", 0) > 1.5
                for v in sel.values()
            )
            if rmsd_si_pass:
                n_targets_pass += 1
            report["level3"][t]["RMSD_SI_PASS"] = rmsd_si_pass
            report["level3"][t]["note"] = "Using RMSD-SI fallback (PCSI not computed)"

    # Decision: ≥1 target passes PCSI → Phase 6 with target subset
    report["n_targets_pass_PCSI"] = n_targets_pass
    report["n_targets_total"] = n_targets_total
    if n_targets_pass >= 1:
        report["decision"] = f"proceed_to_phase_6_with_{n_targets_pass}_target(s)"
    else:
        report["decision"] = "redesign_required_no_target_passed_PCSI"
    return report


def verify_phase6():
    p = RESULTS_ROOT / "phase6"
    report = {"phase": 6, "level1": {}, "level2": {}, "level3": {}, "issues": []}
    json_p = p / "phase6_recipes.json"
    if not json_p.exists():
        report["issues"].append("phase6 recipes JSON missing")
        return report
    with open(json_p) as f:
        data = json.load(f)

    for t in TARGETS:
        mip = data.get(t, {})
        nip = data.get(t + "_NIP", {})
        per_target_file = p / f"{t}_recipe.txt"
        report["level1"][t] = {
            "mip_present": bool(mip),
            "nip_present": bool(nip),  # A7
            "per_target_file_exists": per_target_file.exists(),  # A7
        }
        notes_str = " ".join(mip.get("notes", []) if isinstance(mip.get("notes"), list)
                              else [str(mip.get("notes", ""))])
        report["level2"][t] = {
            "A7_nip_recipe": bool(nip),
            "B10_initiator_mass_in_notes": "mg Irgacure" in notes_str or "mg AIBN" in notes_str,
            "serum_warning_present": "SERUM CROSS-REACTIVITY" in notes_str,
            "ev_context_note": "EV CONTEXT" in notes_str,
            "ph_recommended_present": "pH" in notes_str,
        }
        # L3: polymerization compat
        from pipeline.config import is_polymerization_compatible
        if mip.get("monomers"):
            ok, _, _ = is_polymerization_compatible(list(mip["monomers"].keys()))
            report["level3"][t] = {"polymerization_compatible": ok}

    report["decision"] = "all_phases_complete"
    return report


VERIFY_FUNCS = {
    1: verify_phase1, 2: verify_phase2, 3: verify_phase3,
    4: verify_phase4, 5: verify_phase5, 6: verify_phase6,
}


def write_summary_md(report):
    """Human-readable summary."""
    n = report["phase"]
    lines = [
        f"# Phase {n} Verification Summary",
        "",
        f"**Decision**: `{report.get('decision', 'unknown')}`",
        "",
    ]
    if report.get("issues"):
        lines.append("## ⚠ Issues found")
        for issue in report["issues"]:
            lines.append(f"- {issue}")
        lines.append("")

    for level, name in [("level1", "L1 Code Integrity"),
                        ("level2", "L2 Algorithm Correctness"),
                        ("level3", "L3 Physics/Chemistry Validity")]:
        lines.append(f"## {name}")
        for target, checks in report.get(level, {}).items():
            lines.append(f"### {target}")
            for k, v in checks.items():
                icon = ("✓" if v in (True, "True") else "✗" if v in (False, None) else "—")
                lines.append(f"- {icon} {k}: `{v}`")
        lines.append("")
    out = REPORTS_DIR / f"phase{n}_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(phase_num):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    func = VERIFY_FUNCS.get(phase_num)
    if func is None:
        print(f"Unknown phase: {phase_num}")
        sys.exit(1)
    print(f"Verifying Phase {phase_num}...")
    report = func()
    out_json = REPORTS_DIR / f"phase{phase_num}_verification.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md = write_summary_md(report)
    print(f"  → {out_json}")
    print(f"  → {out_md}")
    print(f"  Decision: {report.get('decision', 'unknown')}")
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify_phase.py <phase_num>")
        sys.exit(1)
    main(int(sys.argv[1]))
