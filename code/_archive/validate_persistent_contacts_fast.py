"""Bit-for-bit + wall-clock comparison: reference loop vs vectorised backends.

Usage:
  python code/validate_persistent_contacts_fast.py <leg_md_dir> [--last-frac F] [--skip-ref]

Prints per-residue frequency max |delta| and both wall times. Exit 1 on mismatch.
"""
import argparse, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from analyze_persistent_contacts import compute_persistent_contacts as ref_impl
from persistent_contacts_fast import compute_persistent_contacts_fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md_dir")
    ap.add_argument("--last-frac", type=float, default=0.5)
    ap.add_argument("--skip-ref", action="store_true")
    a = ap.parse_args()

    md = Path(a.md_dir)
    xtc, tpr = md / "md.xtc", md / "md.tpr"
    print(f"leg: {md}")
    print(f"last_frac={a.last_frac}")

    out = {}
    if not a.skip_ref:
        t0 = time.perf_counter()
        f_ref, n_ref = ref_impl(xtc, tpr, last_frac=a.last_frac)
        t_ref = time.perf_counter() - t0
        print(f"  REFERENCE loop      : n_persistent={n_ref:4d}  "
              f"total_res={len(f_ref):4d}  wall={t_ref:8.2f} s")
        out["ref"] = (f_ref, n_ref, t_ref)

    for method in ("dense", "capped"):
        t0 = time.perf_counter()
        f_new, n_new, meta = compute_persistent_contacts_fast(
            xtc, tpr, last_frac=a.last_frac, method=method, return_meta=True)
        t_new = time.perf_counter() - t0
        print(f"  VECTORISED {method:<8}: n_persistent={n_new:4d}  "
              f"total_res={len(f_new):4d}  wall={t_new:8.2f} s")
        out[method] = (f_new, n_new, t_new)
        if "ref" in out:
            f_ref, n_ref, t_ref = out["ref"]
            assert set(f_ref) == set(f_new), "resid key sets differ"
            delta = max(abs(f_ref[k] - f_new[k]) for k in f_ref)
            ident = all(f_ref[k] == f_new[k] for k in f_ref)
            print(f"      max|dfreq|={delta:.3e}  exactly_equal={ident}  "
                  f"n_persistent_match={n_ref == n_new}  speedup={t_ref / t_new:.1f}x")
            if not ident or n_ref != n_new:
                print("MISMATCH"); sys.exit(1)

    if "dense" in out and "capped" in out:
        fd, fc = out["dense"][0], out["capped"][0]
        print(f"  dense vs capped exactly_equal={all(fd[k] == fc[k] for k in fd)}")

    print(json.dumps({"meta": meta}, indent=2))


if __name__ == "__main__":
    main()
