"""Vectorised persistent-contact core for PCSI* (design validation).

Drop-in replacement for analyze_persistent_contacts.compute_persistent_contacts,
but one distance_array per frame over ALL protein atoms instead of one per
residue per frame (~101x fewer calls).

IMPORTANT: to stay bit-for-bit identical to the legacy loop, the protein
selection must be "protein" (ALL atoms, hydrogens included) -- NOT heavy atoms.
`heavy_only=True` is provided only to quantify the difference; it is a
DIFFERENT statistic and must not be used for the regression anchor.
"""
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

MONOMER_SEL = "not protein and not resname SOL HOH NA CL TIP3 WAT"


def compute_persistent_contacts_vec(traj_path, top_path, cutoff_A=6.0,
                                    persistence_frac=0.5, last_frac=0.5,
                                    heavy_only=False):
    u = mda.Universe(str(top_path), str(traj_path))
    protein = u.select_atoms("protein and not name H*" if heavy_only else "protein")
    monomers = u.select_atoms(MONOMER_SEL)
    if len(protein) == 0 or len(monomers) == 0:
        return {}, 0, {}

    n_frames = len(u.trajectory)
    start = int(n_frames * (1 - last_frac))
    n_analyzed = n_frames - start

    resids = np.array([r.resid for r in protein.residues])
    n_res = len(resids)
    # map each protein atom -> compact residue index 0..n_res-1
    atom_ridx = protein.resindices
    uniq, atom_ridx = np.unique(atom_ridx, return_inverse=True)
    assert len(uniq) == n_res

    counts = np.zeros(n_res, dtype=np.int64)
    for ts in u.trajectory[start:]:
        d = distance_array(protein.positions, monomers.positions,
                           box=ts.dimensions)
        per_atom_min = d.min(axis=1)
        # per-residue minimum via unbuffered scatter-min
        per_res_min = np.full(n_res, np.inf)
        np.minimum.at(per_res_min, atom_ridx, per_atom_min)
        counts += (per_res_min < cutoff_A)

    freq = {int(rid): int(c) / n_analyzed for rid, c in zip(resids, counts)}
    persistent = {rid: f for rid, f in freq.items() if f > persistence_frac}
    meta = {"n_frames": n_frames, "start_frame": start, "n_analyzed": n_analyzed,
            "n_protein_atoms": len(protein), "n_monomer_atoms": len(monomers),
            "total_residues": n_res,
            "persistent_resids": sorted(persistent)}
    return freq, len(persistent), meta
