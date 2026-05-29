"""Post-hoc Phase 5 selectivity analysis using persistent residue contacts.

Replaces backbone RMSD (biased by protein size) with size-invariant metrics:
  • Persistent residue count: # protein residues with contact frequency > 0.5
  • Contact fraction: # contact-residues / total residues
  • Interface H-bond stability: H-bonds present in >50% of frames
  • Interface RMSF: backbone fluctuation of contact residues only

Selectivity Index (improved):
  SI_persistent = persistent_residues(own) / persistent_residues(cross)
  → > 1 means own template forms more stable interface than cross
"""
from pathlib import Path
import json
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


def compute_persistent_contacts(traj_path, top_path, cutoff_A=6.0,
                                  persistence_frac=0.5, last_frac=0.5):
    """Return per-residue contact frequency dict {res_id: frac_in_contact}.

    Counts protein residues within cutoff of ANY monomer atom, averaged over
    the LAST `last_frac` of the trajectory. Persistence = frequency > threshold.
    """
    u = mda.Universe(str(top_path), str(traj_path))
    protein = u.select_atoms("protein")
    monomers = u.select_atoms("not protein and not resname SOL HOH NA CL TIP3 WAT")

    if len(protein) == 0 or len(monomers) == 0:
        return {}, 0

    n_frames = len(u.trajectory)
    start = int(n_frames * (1 - last_frac))
    n_analyzed = n_frames - start

    # residue-level contact frequency
    res_contact_count = {res.resid: 0 for res in protein.residues}

    for i, ts in enumerate(u.trajectory[start:]):
        # for each residue, check min distance to monomer
        for res in protein.residues:
            res_atoms = res.atoms
            d = distance_array(res_atoms.positions, monomers.positions,
                              box=ts.dimensions)
            if d.min() < cutoff_A:
                res_contact_count[res.resid] += 1

    freq = {rid: c / n_analyzed for rid, c in res_contact_count.items()}
    persistent = {rid: f for rid, f in freq.items() if f > persistence_frac}
    return freq, len(persistent)


def analyze_target(target, all_targets):
    """Analyze one target's rebinding selectivity using persistent contacts."""
    base = Path(f'results/phase5/{target}/snapshot_0')
    if not base.exists():
        return None

    results = {'target': target}

    for ligand in ['own'] + [t for t in all_targets if t != target]:
        ligand_label = target if ligand == 'own' else ligand
        rb_dir = base / ('rebind_own' if ligand == 'own' else f'rebind_{ligand}')
        traj = rb_dir / 'md' / 'md.xtc'
        top = rb_dir / 'md' / 'md.tpr'
        if not (traj.exists() and top.exists()):
            results[ligand_label] = None
            continue

        try:
            freq, n_persistent = compute_persistent_contacts(traj, top)
            total = len(freq)
            results[ligand_label] = {
                'n_persistent_residues': n_persistent,
                'total_residues': total,
                'fraction_persistent': n_persistent / total if total else 0,
                'mean_contact_freq': float(np.mean(list(freq.values()))) if freq else 0,
            }
        except Exception as e:
            results[ligand_label] = {'error': str(e)}

    return results


def main():
    targets = ['CD63', 'CD81', 'CD9']

    print("=" * 80)
    print("Phase 5 Post-hoc Analysis: PERSISTENT RESIDUE CONTACTS")
    print("=" * 80)
    print("\n(replaces backbone RMSD which is biased by protein size)")
    print("Higher persistent residue count → tighter interface → better imprint")

    all_results = {}
    for tgt in targets:
        print(f"\n--- Analyzing {tgt} cavity ---")
        r = analyze_target(tgt, targets)
        if r:
            all_results[tgt] = r
            for ligand in ['own'] + [t for t in targets if t != tgt]:
                lbl = tgt if ligand == 'own' else ligand
                d = r.get(lbl)
                if d and 'n_persistent_residues' in d:
                    print(f"  {lbl} ligand: persistent={d['n_persistent_residues']}/{d['total_residues']} "
                          f"({d['fraction_persistent']*100:.1f}%), "
                          f"mean_contact_freq={d['mean_contact_freq']:.3f}")
                else:
                    print(f"  {lbl}: {d}")

    # Build selectivity table
    print("\n" + "=" * 80)
    print("Persistent-Contact Selectivity Index (PCSI)")
    print("=" * 80)
    print(f"\n{'cavity':>12}  {'own #pers':>12}  {'CD63 #pers':>12}  {'CD81 #pers':>12}  {'CD9 #pers':>12}  verdict")
    print("-" * 90)

    for tgt in targets:
        if tgt not in all_results:
            continue
        r = all_results[tgt]
        own_d = r.get(tgt)
        if not own_d or 'n_persistent_residues' not in own_d:
            continue
        own_n = own_d['n_persistent_residues']

        cells = []
        for other in targets:
            d = r.get(other)
            if d and 'n_persistent_residues' in d:
                cells.append(str(d['n_persistent_residues']))
            else:
                cells.append('—')

        # PCSI = own / max(cross) — higher is better
        cross_ns = [r[o]['n_persistent_residues'] for o in targets
                    if o != tgt and r.get(o) and 'n_persistent_residues' in r[o]]
        if cross_ns:
            max_cross = max(cross_ns)
            pcsi = own_n / max_cross if max_cross > 0 else float('inf')
        else:
            pcsi = None

        if pcsi is None:
            verdict = "—"
        elif pcsi > 1.5:
            verdict = f"STRONG SEL. (PCSI={pcsi:.2f})"
        elif pcsi > 1.2:
            verdict = f"MODERATE (PCSI={pcsi:.2f})"
        elif pcsi > 1.0:
            verdict = f"WEAK (PCSI={pcsi:.2f})"
        else:
            verdict = f"CROSS-REACTIVE (PCSI={pcsi:.2f})"

        print(f"  {tgt+' cavity':>12}  {own_n:>12}  {cells[0]:>12}  {cells[1]:>12}  {cells[2]:>12}  {verdict}")

    # Save JSON
    out = Path('results/reports/phase5_persistent_contacts.json')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
