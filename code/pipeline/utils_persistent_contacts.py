"""Vectorised drop-in replacement for analyze_persistent_contacts.compute_persistent_contacts.

The reference implementation (code/analyze_persistent_contacts.py:41-48) calls
distance_array once PER PROTEIN RESIDUE PER FRAME, i.e. n_res * n_frames_analyzed
calls (~2.5e5 for a 101-residue, 5001-frame leg). This module does the same
physics with one neighbour search per frame.

Semantics preserved EXACTLY from the reference so the published counts reproduce:
  * protein selection   : "protein"                       (ALL atoms, H included)
  * monomer selection   : "not protein and not resname SOL HOH NA CL TIP3 WAT"
  * frame window        : u.trajectory[int(n_frames*(1-last_frac)):]
  * denominator         : n_analyzed = n_frames - start
  * contact predicate   : min over (res_atoms x monomer_atoms) distance < cutoff_A
                          STRICT less-than, minimum-image via box=ts.dimensions
  * accumulator         : dict keyed by res.resid (residues sharing a resid are
                          folded into one key by SUMMATION, exactly as the
                          reference dict-increment does)
  * persistent          : freq > persistence_frac   (strict)

Two backends, both verified bit-identical to the reference:
  method="capped" (default) — MDAnalysis.lib.distances.capped_distance grid search,
                              returns only pairs within the cutoff. Fastest.
  method="dense"            — one distance_array per frame over all protein atoms
                              vs all monomer atoms, then per-residue reduction.
                              Kept as the literal "one distance_array per frame"
                              implementation and as a cross-check on "capped".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import capped_distance

def enable_readonly_mode():
    """Keep results/ byte-for-byte untouched.

    MDAnalysis 2.10 writes TWO things next to every .xtc it opens:
      * .md.xtc_offsets.lock  — a filelock, created and removed on every open
      * .md.xtc_offsets.npz   — the offset cache, rewritten if ctime/size/n_atoms
                                disagree with the cache
    Both land inside results/, which we are forbidden to modify (and which would
    hard-fail on a read-only mount). Replace _load_offsets with a lock-free,
    write-free equivalent that keeps MDAnalysis' own staleness checks intact:
    a cache that does not validate is simply ignored and offsets are rebuilt in
    memory (slow but correct), never rewritten to disk.
    """
    from os.path import isfile, getctime, getsize
    from MDAnalysis.coordinates.XDR import (XDRBaseReader, offsets_filename,
                                            read_numpy_offsets)

    if getattr(XDRBaseReader._load_offsets, "_readonly_patched", False):
        return

    def _load_offsets(self):
        fname = offsets_filename(self.filename)
        if isfile(fname):
            data = read_numpy_offsets(fname)
            if data:
                try:
                    if (getctime(self.filename) == data["ctime"]
                            and getsize(self.filename) == data["size"]
                            and self._xdr.n_atoms == data["n_atoms"]):
                        self._xdr.set_offsets(data["offsets"])
                        return
                except KeyError:
                    pass
        self._read_offsets(store=False)      # rebuild in memory, write nothing

    _load_offsets._readonly_patched = True
    XDRBaseReader._load_offsets = _load_offsets

    orig_read = XDRBaseReader._read_offsets
    if not getattr(orig_read, "_no_store_patched", False):
        def _read(self, store=False):
            return orig_read(self, store=False)
        _read._no_store_patched = True
        XDRBaseReader._read_offsets = _read


PROTEIN_SEL = "protein"
# Exclude standard solvents + ions + HEK293T EV lipidome. If two proteins are
# present (e.g. Phase 5 EV-approach mode has a fresh CD *and* possibly a
# residual template CD), pass a specific `binder_sel` to
# compute_persistent_contacts_fast() rather than defaulting to `not protein`
# — that lets the caller name the fresh EV's chain/segid.
MONOMER_SEL = ("not protein "
               "and not resname SOL HOH NA CL TIP3 WAT "
               "and not resname SOD CLA "               # CHARMM-GUI ion names
               "and not resname POPC POPE PSM POPS CHL1")  # HEK293T EV lipidome


def _residue_index_map(protein):
    """Map every protein atom to a 0..n_res-1 residue slot.

    Uses the AtomGroup's own residue ordering (protein.residues), NOT resid
    arithmetic, so non-contiguous / repeated / negative resids are all safe.
    Returns (atom_ridx, resids) where resids[k] is the resid of slot k.
    """
    residues = protein.residues
    # position of each atom's residue within protein.residues
    res_ix = residues.ix                      # global residue indices, in order
    order = np.argsort(res_ix, kind="stable")
    lookup = np.empty(res_ix.max() + 1, dtype=np.int64)
    lookup[res_ix[order]] = order
    atom_ridx = lookup[protein.atoms.resindices]
    return atom_ridx.astype(np.int64), residues.resids.astype(np.int64)


def compute_persistent_contacts_fast(traj_path, top_path, cutoff_A=6.0,
                                     persistence_frac=0.5, last_frac=0.5,
                                     method="capped", return_meta=False,
                                     protein_sel=None, binder_sel=None):
    """Return (freq, n_persistent[, meta]) — identical to the reference function.

    Parameters
    ----------
    protein_sel : str, optional
        Override the default `'protein'` selection. Use to name a specific
        chain/segid when multiple proteins are present (Phase 5 EV-approach
        mode has the FRESH EV's CD alongside possible template residuals).
        Example: `'protein and chainID F'` — F for "fresh".
    binder_sel : str, optional
        Override the default MONOMER_SEL. Use to distinguish MIP monomers
        from the fresh EV's own lipids when the two coexist.
    """
    u = mda.Universe(str(top_path), str(traj_path))
    protein = u.select_atoms(protein_sel or PROTEIN_SEL)
    monomers = u.select_atoms(binder_sel or MONOMER_SEL)

    if len(protein) == 0 or len(monomers) == 0:
        empty = ({}, 0)
        return empty + ({"n_frames": len(u.trajectory), "empty_selection": True},) \
            if return_meta else empty

    n_frames = len(u.trajectory)
    start = int(n_frames * (1 - last_frac))
    n_analyzed = n_frames - start

    atom_ridx, resids = _residue_index_map(protein)
    n_res = len(resids)

    hit_count = np.zeros(n_res, dtype=np.int64)

    if method == "capped":
        for ts in u.trajectory[start:]:
            pairs, dists = capped_distance(
                protein.positions, monomers.positions,
                max_cutoff=cutoff_A, box=ts.dimensions,
                return_distances=True)
            if len(pairs) == 0:
                continue
            # strict "<" to match `d.min() < cutoff_A`; capped is inclusive "<="
            keep = dists < cutoff_A
            if not keep.any():
                continue
            hit_res = np.unique(atom_ridx[pairs[keep, 0]])
            hit_count[hit_res] += 1
    elif method == "dense":
        for ts in u.trajectory[start:]:
            d = distance_array(protein.positions, monomers.positions,
                               box=ts.dimensions)
            row_min = d.min(axis=1)                       # per protein ATOM
            in_cut = row_min < cutoff_A
            if not in_cut.any():
                continue
            # any-atom-in-contact -> residue in contact
            per_res = np.bincount(atom_ridx[in_cut], minlength=n_res)
            hit_count[per_res > 0] += 1
    else:
        raise ValueError(f"unknown method {method!r}")

    # Fold slots into a dict keyed by resid, SUMMING duplicates (reference
    # semantics: the dict is keyed by resid and each residue object increments it).
    freq = {}
    for slot, rid in enumerate(resids):
        rid = int(rid)
        freq[rid] = freq.get(rid, 0) + int(hit_count[slot])
    freq = {rid: c / n_analyzed for rid, c in freq.items()}
    n_persistent = sum(1 for f in freq.values() if f > persistence_frac)

    if not return_meta:
        return freq, n_persistent

    meta = {
        "n_frames": int(n_frames),
        "start_frame": int(start),
        "n_analyzed": int(n_analyzed),
        "dt_ps": float(u.trajectory.dt),
        "total_ns": float(n_frames * u.trajectory.dt / 1000.0),
        "n_protein_atoms": int(len(protein)),
        "n_protein_residues": int(len(resids)),
        "n_unique_resids": int(len(freq)),
        "n_monomer_atoms": int(len(monomers)),
        "monomer_resnames": sorted(set(map(str, monomers.resnames))),
        "method": method,
    }
    return freq, n_persistent, meta


# ---------------------------------------------------------------- leg metric --

def leg_metrics(traj_path, top_path, **kw):
    """Persistent-contact summary for ONE rebinding leg, fractions included."""
    freq, n_pers, meta = compute_persistent_contacts_fast(
        traj_path, top_path, return_meta=True, **kw)
    total = len(freq)
    return {
        "n_persistent_residues": int(n_pers),
        "total_residues": int(total),
        "fraction_persistent": (n_pers / total) if total else 0.0,
        "mean_contact_freq": float(np.mean(list(freq.values()))) if freq else 0.0,
        "contact_freq": {str(k): float(v) for k, v in freq.items()},
        **meta,
    }


def survey_leg(traj_path, top_path):
    """Cheap header-only survey: frames, dt, atom counts. No distance work."""
    info = {"xtc": str(traj_path), "tpr": str(top_path)}
    try:
        tp, xp = Path(top_path), Path(traj_path)
        info["xtc_bytes"] = tp.stat().st_size and xp.stat().st_size
        info["xtc_bytes"] = xp.stat().st_size
        info["tpr_bytes"] = tp.stat().st_size
        u = mda.Universe(str(top_path), str(traj_path))
        prot = u.select_atoms(PROTEIN_SEL)
        mon = u.select_atoms(MONOMER_SEL)
        nf = len(u.trajectory)
        u.trajectory[-1]                       # force read of the LAST frame:
        last_t = float(u.trajectory.time)      # a truncated xtc dies here
        u.trajectory[0]
        info.update(
            ok=True, n_frames=int(nf), dt_ps=float(u.trajectory.dt),
            last_frame_time_ps=last_t,
            total_ns=float(nf * u.trajectory.dt / 1000.0),
            n_protein_atoms=int(len(prot)),
            n_protein_residues=int(len(prot.residues)),
            n_monomer_atoms=int(len(mon)),
            monomer_resnames=sorted(set(map(str, mon.resnames))),
            n_atoms_total=int(len(u.atoms)),
        )
    except Exception as e:
        info.update(ok=False, error=f"{type(e).__name__}: {e}")
    return info
