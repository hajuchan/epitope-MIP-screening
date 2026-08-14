"""Driver: recompute persistent contacts for all Phase 5 rebinding legs and
aggregate the bounded contrast PCSI* with a bootstrap CI.

    PCSI*_j = (f_own - f_j) / (f_own + f_j)      f_L = n_persistent(L) / n_residues(L)
    snapshot value = min_j PCSI*_j               (worst-case cross target)
    gate           = bootstrap 95% CI lower bound over snapshots > 0

Read-only with respect to results/. All output goes to --out-dir (default reports_v2/).

  python code/recompute_pcsi_star.py --workers 6                 # full run
  python code/recompute_pcsi_star.py --workers 6 --limit 3       # smoke test
  python code/recompute_pcsi_star.py --aggregate-only            # re-aggregate checkpoint
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["CD63", "CD81", "CD9"]

# ---- threads: each worker must stay single-threaded or 12 workers thrash -----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


# --------------------------------------------------------------- leg worker --

def _leg_key(r):
    return f"{r['cavity']}|{r['mode']}|{r['snapshot']}|{r['ligand']}"


def _work(row):
    """Runs in a child process. Never raises: failures come back as a record."""
    from persistent_contacts_fast import leg_metrics, enable_readonly_mode
    enable_readonly_mode()
    key = _leg_key(row)
    t0 = time.perf_counter()
    try:
        md = ROOT / row["md_dir"]
        m = leg_metrics(md / "md.xtc", md / "md.tpr")
        return {**row, **m, "key": key, "ok": True,
                "wall_s": round(time.perf_counter() - t0, 2)}
    except Exception as e:
        return {**row, "key": key, "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:],
                "wall_s": round(time.perf_counter() - t0, 2)}


# ------------------------------------------------------------------ metrics --

def pcsi_star(f_own, f_cross):
    denom = f_own + f_cross
    if denom == 0:
        return 0.0, True            # (value, degenerate) — both legs made no contact
    return (f_own - f_cross) / denom, False


def bootstrap_ci(values, n_boot=20000, alpha=0.05, seed=0):
    import numpy as np
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None
    if len(v) == 1:
        return {"n": 1, "mean": float(v[0]), "lo": None, "hi": None,
                "note": "n=1, no CI"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n": int(len(v)), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "lo": float(lo), "hi": float(hi),
            "n_boot": n_boot, "seed": seed}


def aggregate(leg_records, survey_rows, n_boot, seed):
    """Combine per-leg fractions into per-snapshot PCSI* and per-cavity CIs."""
    by_key = {r["key"]: r for r in leg_records if r.get("ok")}
    status = {_leg_key(r): r["status"] for r in survey_rows}

    # LAYOUT GUARD (integration pass 2026-08-12).
    # This driver is PINNED TO THE v1 PHASE 5 LAYOUT: exactly 10 snapshots per
    # cavity, named snapshot_0..snapshot_9, all carved from ONE Phase 4
    # trajectory. That is the layout the archived run has and the layout
    # reports_v2/ was computed from, so the range(10) below is left alone
    # deliberately — changing it would alter a frozen artifact's provenance.
    #
    # Phase 5 now spreads snapshots across Phase 4 replicas and names them
    # rep<r>_snapshot_<i>. Pointed at such a tree this driver would match NO
    # keys and emit an empty summary WITHOUT ERROR. Refuse instead; the
    # forward-looking driver is code/run_pcsi_star.py, which discovers both
    # layouts.
    if by_key and not any(
            k.split("|")[2].isdigit() for k in by_key if k.count("|") >= 3):
        sys.exit(
            "recompute_pcsi_star.py is pinned to the v1 Phase 5 layout "
            "(snapshot_0..9 from a single Phase 4 trajectory), but the legs it "
            "was given carry replica-qualified snapshot ids (rep<r>_snapshot_<i>). "
            "It would silently aggregate zero snapshots. Use "
            "code/run_pcsi_star.py for a replica-spread tree.")

    out = {}
    for cav in TARGETS:
        others = [o for o in TARGETS if o != cav]
        for mode in (["main", "dual"] if cav == "CD63" else ["main"]):
            snaps, notes = [], []
            for snap in range(10):
                ok = by_key.get(f"{cav}|{mode}|{snap}|{cav}")
                if ok is None:
                    st = status.get(f"{cav}|{mode}|{snap}|{cav}", "absent")
                    why = "not yet computed" if st == "present" else st
                    notes.append(f"snap{snap}: own leg {why} -> snapshot dropped")
                    continue
                f_own = ok["fraction_persistent"]
                per_cross, degen = {}, False
                for o in others:
                    k = f"{cav}|{mode}|{snap}|{o}"
                    st = status.get(k)
                    if st == "size_excluded":
                        # DATA: the cross protein cannot enter the cavity at all.
                        # f_cross = 0 by construction -> PCSI* = +1.
                        v, d = pcsi_star(f_own, 0.0)
                        per_cross[o] = {"f_cross": 0.0, "pcsi_star": v,
                                        "source": "size_excluded"}
                        degen |= d
                    elif k in by_key:
                        fc = by_key[k]["fraction_persistent"]
                        v, d = pcsi_star(f_own, fc)
                        per_cross[o] = {"f_cross": fc, "pcsi_star": v,
                                        "source": "md"}
                        degen |= d
                    else:
                        # NOT data. A crashed/absent MD run is unknown, not zero.
                        per_cross[o] = {"f_cross": None, "pcsi_star": None,
                                        "source": st or "missing"}
                        notes.append(f"snap{snap}: cross {o} unusable ({st}) -> excluded from min()")
                usable = [c["pcsi_star"] for c in per_cross.values()
                          if c["pcsi_star"] is not None]
                if not usable:
                    notes.append(f"snap{snap}: no usable cross target -> snapshot dropped")
                    continue
                snaps.append({
                    "snapshot": snap, "f_own": f_own,
                    "n_persistent_own": ok["n_persistent_residues"],
                    "total_residues_own": ok["total_residues"],
                    "cross": per_cross,
                    "pcsi_star": min(usable),
                    "n_cross_used": len(usable),
                    "degenerate_zero_zero": degen,
                })
            vals = [s["pcsi_star"] for s in snaps]
            ci = bootstrap_ci(vals, n_boot=n_boot, seed=seed) if vals else None
            out[f"{cav}_{mode}"] = {
                "cavity": cav, "mode": mode, "snapshots": snaps,
                "pcsi_star_values": vals, "ci": ci,
                "decision": (None if not ci or ci.get("lo") is None
                             else ("SELECTIVE" if ci["lo"] > 0 else "NOT DECIDABLE")),
                # label is descriptive only — the gate is ci["lo"] > 0
                "label": (None if not ci else
                          ("STRONG" if ci["mean"] > 0.20 else
                           "MODEST" if ci["mean"] > 0 else
                           "CROSS-REACTIVE")),
                "notes": notes,
            }
    return out


# --------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", default=str(ROOT / "reports_v2/phase5_leg_survey.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports_v2"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="only N legs (smoke test)")
    ap.add_argument("--last-frac", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore existing checkpoint")
    a = ap.parse_args()

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "pcsi_star_legs.jsonl"          # append-only checkpoint
    survey = json.loads(Path(a.survey).read_text())
    todo_all = [r for r in survey if r["status"] == "present"]
    if a.limit:
        todo_all = todo_all[:a.limit]

    done = {}
    if ckpt.exists() and not a.restart:
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r                    # last write wins on retry
        print(f"checkpoint: {len(done)} legs already done ({ckpt})")

    if not a.aggregate_only:
        todo = [r for r in todo_all if _leg_key(r) not in done or not done[_leg_key(r)].get("ok")]
        print(f"{len(todo)} legs to compute, {a.workers} workers")
        t0 = time.perf_counter()
        if todo:
            with ProcessPoolExecutor(max_workers=a.workers) as ex, \
                 open(ckpt, "a") as fh:
                futs = {ex.submit(_work, r): _leg_key(r) for r in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    os.fsync(fh.fileno())             # survive a hard kill
                    done[rec["key"]] = rec
                    el = time.perf_counter() - t0
                    print(f"[{i}/{len(todo)}] {rec['key']:35s} "
                          f"{'ok' if rec['ok'] else 'FAIL'} "
                          f"n_pers={rec.get('n_persistent_residues')} "
                          f"{rec['wall_s']:6.1f}s | elapsed {el/60:5.1f}m "
                          f"eta {el/i*(len(todo)-i)/60:5.1f}m", flush=True)
        print(f"compute wall: {(time.perf_counter()-t0)/60:.2f} min")

    legs = list(done.values())
    fails = [r for r in legs if not r.get("ok")]
    if fails:
        print(f"WARNING: {len(fails)} legs failed: {[r['key'] for r in fails]}")

    agg = aggregate(legs, survey, a.n_boot, a.seed)
    (out_dir / "pcsi_star_summary.json").write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "params": {"cutoff_A": 6.0, "persistence_frac": 0.5,
                    "last_frac": a.last_frac, "n_boot": a.n_boot, "seed": a.seed},
         "n_legs_ok": len(legs) - len(fails), "n_legs_failed": len(fails),
         "cavities": agg}, indent=2))
    print("\n" + "=" * 78)
    for k, v in agg.items():
        ci = v["ci"]
        if not ci:
            print(f"{k:12s}  NO DATA"); continue
        sd = ci.get("sd")
        ciw = (f"[{ci['lo']:+.3f},{ci['hi']:+.3f}]" if ci.get("lo") is not None
               else "[n<2, no CI]")
        print(f"{k:12s}  n={ci['n']:2d}  PCSI*={ci['mean']:+.3f}  "
              f"sd={'  n/a' if sd is None else f'{sd:.3f}'}  "
              f"95%CI={ciw}  {v['decision']}  {v['label']}")
        for n in v["notes"]:
            print(f"                 note: {n}")
    print(f"\nwrote {out_dir/'pcsi_star_summary.json'}")


if __name__ == "__main__":
    main()
