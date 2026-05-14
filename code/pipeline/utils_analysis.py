"""
Analysis Utilities
==================
Binding site prediction (fpocket), hydrogen bond analysis
(backbone vs sidechain), contact frequency from MD, DSSP
secondary structure analysis, and competition scoring.

Reference:
  Sullivan et al., J. Phys. Chem. B 2019 — SiteMap, H-bond analysis
  Sehit/Altintas, ACS Sensors 2024 — contact frequency from MD
"""

import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Binding Site Prediction ────────────────────────────────────

def predict_binding_sites(pdb_path: Path, method: str = "fpocket",
                           output_dir: Path = None) -> list:
    """
    Predict binding pockets on epitope surface.

    Sullivan 2019 used SiteMap (Schrödinger, commercial).
    Here we use fpocket (free) as default, with geometric fallback.

    Returns list of dicts: [{center: (x,y,z), residues: [...], score: float}]
    """
    pdb_path = Path(pdb_path)
    if output_dir is None:
        output_dir = pdb_path.parent

    if method == "fpocket":
        return _fpocket_sites(pdb_path, output_dir)
    elif method == "geometric":
        return _geometric_sites(pdb_path)
    else:
        logger.warning(f"Unknown method {method}, using geometric fallback")
        return _geometric_sites(pdb_path)


def _fpocket_sites(pdb_path: Path, output_dir: Path) -> list:
    """Run fpocket to identify binding pockets."""
    try:
        result = subprocess.run(
            ["fpocket", "-f", str(pdb_path)],
            cwd=str(output_dir),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning(f"fpocket failed: {result.stderr[:200]}")
            return _geometric_sites(pdb_path)

        # Parse fpocket output
        out_dir = output_dir / f"{pdb_path.stem}_out"
        info_file = out_dir / f"{pdb_path.stem}_info.txt"
        pockets = []

        if info_file.exists():
            pocket_data = _parse_fpocket_info(info_file)
            # Read pocket PDB files for center coordinates
            for i, pdata in enumerate(pocket_data):
                pocket_pdb = out_dir / "pockets" / f"pocket{i}_atm.pdb"
                if pocket_pdb.exists():
                    center = _pdb_center(pocket_pdb)
                    pdata["center"] = center
                pockets.append(pdata)
        else:
            # Try parsing pocket PDB files directly
            pocket_dir = out_dir / "pockets"
            if pocket_dir.exists():
                for pocket_pdb in sorted(pocket_dir.glob("pocket*_atm.pdb")):
                    center = _pdb_center(pocket_pdb)
                    pockets.append({
                        "center": center,
                        "score": 0.0,
                        "residues": [],
                    })

        if not pockets:
            return _geometric_sites(pdb_path)

        logger.info(f"fpocket found {len(pockets)} binding pockets")
        return pockets

    except FileNotFoundError:
        logger.warning("fpocket not installed, using geometric fallback")
        return _geometric_sites(pdb_path)


def _geometric_sites(pdb_path: Path, n_sites: int = 5) -> list:
    """
    Fallback: divide epitope surface into clusters based on
    residue positions (geometric partitioning).
    """
    from Bio.PDB import PDBParser
    from scipy.cluster.hierarchy import fcluster, linkage

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pep", str(pdb_path))

    # Get CA positions
    ca_coords = []
    residues = []
    for residue in structure.get_residues():
        if "CA" in residue:
            ca_coords.append(residue["CA"].get_vector().get_array())
            residues.append(residue.get_id()[1])

    if len(ca_coords) < 2:
        center = np.mean(ca_coords, axis=0) if ca_coords else [0, 0, 0]
        return [{"center": tuple(center), "residues": residues, "score": 1.0}]

    ca_coords = np.array(ca_coords)
    n_clusters = min(n_sites, len(ca_coords))

    Z = linkage(ca_coords, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    sites = []
    for c_id in range(1, n_clusters + 1):
        mask = labels == c_id
        cluster_coords = ca_coords[mask]
        cluster_residues = [r for r, m in zip(residues, mask) if m]
        center = np.mean(cluster_coords, axis=0)
        sites.append({
            "center": tuple(float(x) for x in center),
            "residues": cluster_residues,
            "score": float(len(cluster_residues)) / len(residues),
        })

    logger.info(f"Geometric partitioning: {len(sites)} binding sites")
    return sites


# ── H-bond Analysis (backbone vs sidechain) ───────────────────

def analyze_hbond_types(dlg_path: Path, receptor_pdb: Path) -> dict:
    """
    Analyze whether docked monomer H-bonds are with protein
    backbone or sidechain atoms.

    Sullivan 2019: high backbone H-bond ratio → 2° structure disruption.

    Returns dict with backbone_count, sidechain_count, ratio,
    helical_backbone_count.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import dssp_dict_from_pdb_file

    # Identify backbone vs sidechain atoms, and secondary structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", str(receptor_pdb))

    backbone_atoms = {"N", "CA", "C", "O", "H", "HA"}
    residue_info = {}  # resid -> {backbone_atoms: set, ss: str}

    for residue in structure.get_residues():
        resid = residue.get_id()[1]
        bb = set()
        sc = set()
        for atom in residue.get_atoms():
            if atom.get_name() in backbone_atoms:
                bb.add(atom.get_name())
            else:
                sc.add(atom.get_name())
        residue_info[resid] = {"backbone": bb, "sidechain": sc}

    # Get DSSP secondary structure
    try:
        dssp = dssp_dict_from_pdb_file(str(receptor_pdb))
        for key, val in dssp[0].items():
            resid = key[1][1]
            if resid in residue_info:
                residue_info[resid]["ss"] = val[2]  # H, E, C, etc.
    except Exception:
        # DSSP not available, assume all helix for epitope
        for resid in residue_info:
            residue_info[resid]["ss"] = "H"

    # Parse docked pose contacts
    from .utils_autodock import parse_dlg, _extract_best_pose
    text = Path(dlg_path).read_text()
    pose_lines = _extract_best_pose(text)

    # Simple contact analysis: find receptor atoms within 3.5Å of ligand
    ligand_coords = _extract_coords_from_pdbqt_lines(pose_lines)
    receptor_coords = _get_receptor_atom_info(structure)

    bb_contacts = 0
    sc_contacts = 0
    helical_bb_contacts = 0

    for lig_coord in ligand_coords:
        for rec_info in receptor_coords:
            dist = np.linalg.norm(lig_coord - rec_info["coord"])
            if dist < 3.5:  # H-bond distance cutoff
                resid = rec_info["resid"]
                if rec_info["atom_name"] in backbone_atoms:
                    bb_contacts += 1
                    ss = residue_info.get(resid, {}).get("ss", "C")
                    if ss in ("H", "G", "I"):  # helical
                        helical_bb_contacts += 1
                else:
                    sc_contacts += 1

    total = bb_contacts + sc_contacts
    ratio = bb_contacts / total if total > 0 else 0.0

    return {
        "backbone_contacts": bb_contacts,
        "sidechain_contacts": sc_contacts,
        "total_contacts": total,
        "backbone_ratio": round(ratio, 3),
        "helical_backbone_contacts": helical_bb_contacts,
        "structural_disruption_risk": ratio > 0.3,  # Sullivan threshold
    }


# ── Contact Frequency from MD (Sehit 2024) ────────────────────

def compute_contact_frequency(traj_path: Path, top_path: Path,
                               protein_selection: str = "protein",
                               ligand_selection: str = "not protein and not water",
                               cutoff_nm: float = 0.35) -> dict:
    """
    Compute contact frequency between protein residues and ligand
    from an MD trajectory.

    Sehit 2024: "number of contacts" between epitope and monomer
    was used to rank monomers.
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis import contacts

        u = mda.Universe(str(top_path), str(traj_path))
        protein = u.select_atoms(protein_selection)
        ligand = u.select_atoms(ligand_selection)

        # Per-residue contact frequency
        residue_contacts = {}
        n_frames = 0

        for ts in u.trajectory:
            n_frames += 1
            for res in protein.residues:
                res_atoms = res.atoms
                for lig_atom in ligand.atoms:
                    dists = np.linalg.norm(
                        res_atoms.positions - lig_atom.position, axis=1
                    )
                    if np.any(dists < cutoff_nm * 10):  # nm to Å
                        resid = res.resid
                        residue_contacts[resid] = \
                            residue_contacts.get(resid, 0) + 1
                        break  # count once per residue per frame

        # Normalize by frame count
        freq = {resid: count / n_frames
                for resid, count in residue_contacts.items()}

        total_contacts = sum(freq.values())
        n_contact_residues = sum(1 for f in freq.values() if f > 0.1)

        return {
            "residue_frequencies": freq,
            "total_contact_score": round(total_contacts, 2),
            "n_contact_residues": n_contact_residues,
            "n_frames": n_frames,
        }
    except ImportError:
        logger.warning("MDAnalysis not available for contact analysis")
        return {"error": "MDAnalysis not installed"}
    except Exception as e:
        logger.warning(f"Contact frequency analysis failed: {e}")
        return {"error": str(e)}


# ── DSSP Secondary Structure (computational CD) ───────────────

def analyze_dssp_changes(traj_path: Path, top_path: Path) -> dict:
    """
    Track secondary structure changes during MD (computational
    equivalent of CD spectroscopy).

    Sullivan 2019: monomers that disrupt α-helix → poor MIP performance.
    """
    try:
        import mdtraj as md

        traj = md.load(str(traj_path), top=str(top_path))

        # DSSP at first and last frames
        dssp_first = md.compute_dssp(traj[0])
        dssp_last = md.compute_dssp(traj[-1])

        # Compute fraction over entire trajectory
        dssp_all = md.compute_dssp(traj)  # (n_frames, n_residues)

        # Count fractions
        def _count_ss(dssp_array):
            total = dssp_array.size
            return {
                "helix": float(np.sum(dssp_array == "H")) / total,
                "sheet": float(np.sum(dssp_array == "E")) / total,
                "coil": float(np.sum(np.isin(dssp_array, ["C", " ", "NA"]))) / total,
            }

        ss_initial = _count_ss(dssp_first)
        ss_final = _count_ss(dssp_last)
        ss_average = _count_ss(dssp_all)

        helix_change = ss_final["helix"] - ss_initial["helix"]

        return {
            "initial": ss_initial,
            "final": ss_final,
            "average": ss_average,
            "helix_change": round(helix_change, 3),
            "structure_preserved": abs(helix_change) < 0.1,
        }
    except ImportError:
        logger.warning("mdtraj not available for DSSP analysis")
        return {"error": "mdtraj not installed"}
    except Exception as e:
        logger.warning(f"DSSP analysis failed: {e}")
        return {"error": str(e)}


# ── MMSD Competition Analysis (Sullivan 2019) ─────────────────

def analyze_competition(monomer_poses: dict,
                         distance_threshold: float = 5.0) -> dict:
    """
    Check if monomers in an MMSD combination compete for
    the same binding site.

    Sullivan 2019: "uniform, noncompetitive binding around the
    protein surface is favorable."

    Parameters
    ----------
    monomer_poses : {monomer_name: (x, y, z) center of docked pose}
    distance_threshold : Å — below this = same site = competition

    Returns dict with competition_pairs, competition_score, is_uniform.
    """
    names = list(monomer_poses.keys())
    centers = [np.array(monomer_poses[n]) for n in names]

    competition_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < distance_threshold:
                competition_pairs.append({
                    "monomer_1": names[i],
                    "monomer_2": names[j],
                    "distance": round(float(dist), 2),
                })

    n_pairs = len(names) * (len(names) - 1) / 2
    competition_score = len(competition_pairs) / n_pairs if n_pairs > 0 else 0

    return {
        "competition_pairs": competition_pairs,
        "n_competing": len(competition_pairs),
        "competition_score": round(competition_score, 3),
        "is_uniform": len(competition_pairs) == 0,
    }


# ── Internal Helpers ───────────────────────────────────────────

def _parse_fpocket_info(info_path: Path) -> list:
    """Parse fpocket info file."""
    pockets = []
    current = {}
    for line in Path(info_path).read_text().split("\n"):
        if line.startswith("Pocket"):
            if current:
                pockets.append(current)
            current = {"score": 0.0, "residues": []}
        elif "Score" in line and ":" in line:
            try:
                current["score"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
    if current:
        pockets.append(current)
    return pockets


def _pdb_center(pdb_path: Path) -> tuple:
    """Get geometric center of a PDB file."""
    coords = []
    for line in Path(pdb_path).read_text().split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue
    if not coords:
        return (0.0, 0.0, 0.0)
    center = np.mean(coords, axis=0)
    return tuple(round(float(c), 3) for c in center)


def _extract_coords_from_pdbqt_lines(lines: list) -> list:
    """Extract coordinates from PDBQT lines."""
    coords = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append(np.array([x, y, z]))
            except ValueError:
                continue
    return coords


def _get_receptor_atom_info(structure) -> list:
    """Extract atom info from BioPython structure."""
    atoms = []
    for residue in structure.get_residues():
        resid = residue.get_id()[1]
        for atom in residue.get_atoms():
            atoms.append({
                "coord": atom.get_vector().get_array(),
                "atom_name": atom.get_name(),
                "resid": resid,
            })
    return atoms

# ════════════════════════════════════════════════════════════════
# Epitope Quality Analysis (A1, B1, B2)
# ════════════════════════════════════════════════════════════════

KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


def gravy_index(sequence: str) -> float:
    """Grand Average of Hydropathy (Kyte & Doolittle 1982 J Mol Biol).

    GRAVY > +2 = too hydrophobic, < -2 = too hydrophilic, in between balanced.
    """
    if not sequence:
        return 0.0
    return sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in sequence) / len(sequence)


def compute_sasa_per_residue(pdb_path, residue_range: tuple = None) -> dict:
    """Solvent-Accessible Surface Area per residue using mdtraj (Shrake-Rupley).

    Returns mean SASA, min SASA, list of fully-buried residues (SASA<10 Å²).

    Indexing logic:
    - mdtraj's `shrake_rupley(mode='residue')` returns a (1, N_residues) array
      where the index 0..N-1 corresponds to the residues in the PDB file
      in their order of appearance, NOT to original protein residue numbers.
    - If `residue_range` provided AND it falls within the array bounds, use it.
    - If `residue_range` provided BUT exceeds array bounds (e.g., extracted
      head PDB only has 16 residues but range=(157,172)), use the entire
      array since the extracted PDB already contains only the residues of
      interest (offset by start for buried-residue reporting).

    Reference: Lee & Richards 1971 J Mol Biol.
    """
    try:
        import mdtraj as md
        traj = md.load(str(pdb_path))
        sasa = md.shrake_rupley(traj, mode='residue')  # nm² per residue
        n_residues = sasa.shape[1]
        # Try original numbering first; if out of range, fall back to entire array
        if residue_range is None:
            window = sasa[0]
            start_for_label = 1
        else:
            start, end = residue_range
            # If the requested range fits within the array, use absolute indexing
            if 0 <= start - 1 < n_residues and end <= n_residues:
                window = sasa[0, start - 1:end]
                start_for_label = start
            else:
                # PDB likely contains only the head region (renumbered or not)
                # → use the entire array, treat first residue as `start`
                window = sasa[0]
                start_for_label = start

        if len(window) == 0:
            return {
                'error': f'empty SASA window (n_residues={n_residues}, '
                          f'requested={residue_range})',
                'mean_sasa_A2': None,
            }

        window_A2 = window * 100  # nm² → Å²
        return {
            'mean_sasa_A2': round(float(window_A2.mean()), 2),
            'min_sasa_A2': round(float(window_A2.min()), 2),
            'max_sasa_A2': round(float(window_A2.max()), 2),
            'n_residues_used': int(len(window)),
            'fully_buried_residues': [
                int(start_for_label + i) for i, s in enumerate(window_A2) if s < 10
            ],
        }
    except Exception as e:
        return {'error': str(e), 'mean_sasa_A2': None}


def check_epitope_quality(sequence: str, pdb_path=None,
                          residue_range: tuple = None) -> dict:
    """A1/B1/B2: Combined epitope quality scoring.

    Returns gravy, SASA, balance category, recommendation.
    Used by Phase 1 to rank multi-epitope candidates.
    """
    from .config import (EPITOPE_SASA_MIN_A2, EPITOPE_GRAVY_MIN, EPITOPE_GRAVY_MAX)

    result = {
        'sequence': sequence,
        'length': len(sequence),
        'gravy': gravy_index(sequence),
    }
    # Hydrophilicity balance
    if result['gravy'] > EPITOPE_GRAVY_MAX:
        result['balance'] = 'too_hydrophobic'
    elif result['gravy'] < EPITOPE_GRAVY_MIN:
        result['balance'] = 'too_hydrophilic'
    else:
        result['balance'] = 'balanced'

    # SASA (if PDB given)
    if pdb_path and residue_range:
        sasa = compute_sasa_per_residue(pdb_path, residue_range)
        result['sasa'] = sasa
        result['surface_accessible'] = (sasa.get('mean_sasa_A2', 0) or 0) >= EPITOPE_SASA_MIN_A2

    # Overall score (higher = better epitope candidate)
    # +1 for balance, +1 for accessibility, +0.5 for not having buried residues
    score = 0
    if result['balance'] == 'balanced':
        score += 1
    if result.get('surface_accessible'):
        score += 1
    if pdb_path and not result.get('sasa', {}).get('fully_buried_residues'):
        score += 0.5
    result['quality_score'] = score
    return result



# ════════════════════════════════════════════════════════════════
# Conformer Clustering (A2) — K-medoids on RMSD matrix
# ════════════════════════════════════════════════════════════════

def cluster_conformers_kmedoids(traj_path, top_path, n_clusters: int = 5,
                                  stride: int = 10):
    """A2: K-medoids clustering of MD frames by pairwise RMSD.

    Returns list of medoid frame indices and per-cluster info.
    Reference: Shao 2007 JCTC; Daura 1999 Angew Chem.
    """
    try:
        import mdtraj as md
        from sklearn_extra.cluster import KMedoids
        import numpy as np
    except ImportError as e:
        return {'error': f'Missing dependency: {e}', 'frames': None}

    try:
        traj = md.load(str(traj_path), top=str(top_path))
        if traj.n_frames < n_clusters * 2:
            return {'error': f'Too few frames ({traj.n_frames}) for {n_clusters} clusters',
                    'frames': list(range(0, traj.n_frames,
                                          max(1, traj.n_frames // n_clusters)))[:n_clusters]}
        # Stride to reduce computation
        traj_stride = traj[::stride]
        # Compute pairwise RMSD matrix (CA-only for speed)
        ca_indices = traj_stride.topology.select('name CA')
        if len(ca_indices) == 0:
            ca_indices = traj_stride.topology.select('protein and name C')
        traj_ca = traj_stride.atom_slice(ca_indices)
        n = traj_ca.n_frames
        rmsd_matrix = np.zeros((n, n))
        for i in range(n):
            rmsd_matrix[i] = md.rmsd(traj_ca, traj_ca, i) * 10  # nm → Å

        # K-medoids
        km = KMedoids(n_clusters=n_clusters, metric='precomputed',
                       method='pam', random_state=42).fit(rmsd_matrix)
        medoid_strided = km.medoid_indices_
        medoid_frames = [int(i * stride) for i in medoid_strided]
        # Cluster sizes
        cluster_sizes = [int(sum(km.labels_ == k)) for k in range(n_clusters)]
        return {
            'frames': medoid_frames,
            'cluster_sizes': cluster_sizes,
            'method': 'kmedoids_PAM',
            'n_total_frames': traj.n_frames,
            'n_stride_frames': n,
        }
    except Exception as e:
        return {'error': str(e), 'frames': None}


# ════════════════════════════════════════════════════════════════
# Pose Clustering (A3) — RMSD-based clustering of AutoDock poses
# ════════════════════════════════════════════════════════════════

def cluster_docking_poses(pdbqt_lines_list, rmsd_cutoff: float = 2.0,
                          min_cluster_size: int = 1):
    """A3: Cluster AutoDock poses by ligand RMSD.

    Input: list of (pose_dict) where each pose has 'binding_energy' and 'coords'.
    Output: list of (representative pose, cluster_size, binding_energy).
    Reference: Trott & Olson 2010 (AutoDock Vina).
    """
    import numpy as np

    def rmsd(coords_a, coords_b):
        ca = np.array(coords_a)
        cb = np.array(coords_b)
        return float(np.sqrt(np.mean((ca - cb) ** 2)))

    sorted_poses = sorted(pdbqt_lines_list, key=lambda p: p.get('binding_energy', 0))
    clusters = []  # each: list of poses
    for pose in sorted_poses:
        added = False
        for c in clusters:
            if rmsd(pose['coords'], c[0]['coords']) < rmsd_cutoff:
                c.append(pose)
                added = True
                break
        if not added:
            clusters.append([pose])

    representatives = []
    for c in clusters:
        if len(c) < min_cluster_size:
            continue
        rep = c[0]
        representatives.append({
            'representative_pose': rep,
            'cluster_size': len(c),
            'binding_energy': rep.get('binding_energy'),
            'binding_energies_in_cluster': [p.get('binding_energy') for p in c],
        })
    return representatives


# ════════════════════════════════════════════════════════════════
# Bootstrap Confidence Interval (A6) — for SI, p-value, etc.
# ════════════════════════════════════════════════════════════════

def bootstrap_selectivity_index(own_rmsds, cross_rmsds,
                                 n_bootstrap: int = 10000,
                                 ci: float = 0.95):
    """A6: Bootstrap CI for selectivity index (SI = cross/own).

    Reference: Efron 1979 (Ann Stat).
    """
    import numpy as np
    own = np.array([x for x in own_rmsds if x is not None])
    cross = np.array([x for x in cross_rmsds if x is not None])
    if len(own) < 2 or len(cross) < 2:
        return {'error': 'insufficient data'}

    si_samples = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        own_s = rng.choice(own, size=len(own), replace=True)
        cross_s = rng.choice(cross, size=len(cross), replace=True)
        if own_s.mean() > 0:
            si_samples.append(cross_s.mean() / own_s.mean())
    si_samples = np.array(si_samples)
    alpha = (1 - ci) / 2
    return {
        'si_mean': float(si_samples.mean()),
        'si_median': float(np.median(si_samples)),
        'si_std': float(si_samples.std()),
        'si_ci_lower': float(np.percentile(si_samples, alpha * 100)),
        'si_ci_upper': float(np.percentile(si_samples, (1 - alpha) * 100)),
        'ci_level': ci,
        'n_bootstrap': n_bootstrap,
        'is_selective_at_ci': float(np.percentile(si_samples, alpha * 100)) > 1.5,
    }


# ════════════════════════════════════════════════════════════════
# Enrichment Factor (B3) — vs decoy monomers
# ════════════════════════════════════════════════════════════════

def enrichment_factor(target_be_matrix: dict, decoy_be_matrix: dict) -> dict:
    """B3: Enrichment of real monomers over decoys (negative control).

    Higher value (e.g., 3-10) = real monomers genuinely bind better than random.
    Reference: Mysinger 2012 (JCIM) — DUD-E benchmark concept.
    """
    import numpy as np
    target_bes = [v for v in target_be_matrix.values() if v is not None]
    decoy_bes = [v for v in decoy_be_matrix.values() if v is not None]
    if not target_bes or not decoy_bes:
        return {'error': 'empty matrix'}
    t_mean = np.mean(target_bes)
    d_mean = np.mean(decoy_bes)
    # Lower (more negative) BE = stronger binding.
    # EF = |target| / |decoy| → > 1 means real monomers bind genuinely stronger.
    ef = abs(t_mean) / abs(d_mean) if d_mean != 0 else float('nan')
    return {
        'target_mean_be': float(t_mean),
        'decoy_mean_be': float(d_mean),
        'enrichment_factor': float(ef),
        'enriched': ef > 1.5,  # target binds at least 1.5x stronger than decoys
    }


# ════════════════════════════════════════════════════════════════
# Initiator Quantity Calculation (B10)
# ════════════════════════════════════════════════════════════════

def calculate_initiator_mass(total_monomer_mol: float,
                              initiator: str = "Irgacure_2959",
                              mol_percent: float = 1.0) -> dict:
    """B10: Calculate initiator mass for given monomer amount.

    Reference: Odian 2004 Polymer Synthesis textbook (0.5–2 mol% standard).
    """
    from .config import INITIATOR_MW
    mw = INITIATOR_MW.get(initiator, 224.25)
    init_mol = total_monomer_mol * mol_percent / 100.0
    init_mass_mg = init_mol * mw * 1000.0
    return {
        'initiator': initiator,
        'mol_percent': mol_percent,
        'initiator_mass_mg': round(init_mass_mg, 2),
        'initiator_mol': init_mol,
        'note': f'{mol_percent} mol% of total monomer ({total_monomer_mol*1000:.2f} mmol)',
    }


# ════════════════════════════════════════════════════════════════
# C2: Multi-Objective Optimization Helpers
# ════════════════════════════════════════════════════════════════

def compute_synthesizability_score(combo: list) -> dict:
    """C2: Synthesizability score 0-10 for a monomer combination.

    Combines:
      • Polymerization compatibility (3 pts) — silane vs vinyl one-pot
      • pH stability window (2 pts) — wider = more flexible synthesis
      • Vinyl Q-e reactivity ratios (2.5 pts) — random copolymer ideal
      • No "surface-only" pseudo-monomer issues (1.5 pts)
      • Bonus for matching boronate chemistry to glycan target (1 pt)

    Reference: Liu 2017 Nat Protoc (compatibility), Alfrey-Price 1947 (Q-e).
    """
    import itertools
    from .config import (is_polymerization_compatible, synthesis_pH_window,
                          reactivity_ratio_product, QE_PARAMS, ALL_MONOMERS)

    score = 0.0
    breakdown = {}

    # 1. Polymerization compatibility (3 pts)
    ok, dominant, _ = is_polymerization_compatible(combo)
    if ok:
        score += 3.0
        breakdown["polymerization"] = 3.0
    else:
        breakdown["polymerization"] = 0.0

    # 2. pH stability window (2 pts max)
    pH_min, pH_max, pH_rec = synthesis_pH_window(combo)
    if pH_min is not None and pH_max is not None and pH_max > pH_min:
        window = pH_max - pH_min
        pH_pts = min(2.0, (window / 5.0) * 2.0)  # 5+ pH range → 2 pts
        score += pH_pts
        breakdown["ph_window"] = round(pH_pts, 2)
    else:
        breakdown["ph_window"] = 0.0

    # 3. Vinyl reactivity ratios (2.5 pts max, only for radical)
    if dominant == "radical":
        vinyl_ms = [m for m in combo if m in QE_PARAMS]
        if len(vinyl_ms) >= 2:
            n_good_pairs = 0
            n_total_pairs = 0
            for m1, m2 in itertools.combinations(vinyl_ms, 2):
                rr = reactivity_ratio_product(m1, m2)
                if rr is None:
                    continue
                n_total_pairs += 1
                if 0.1 < rr < 10:
                    n_good_pairs += 1
            rxr_pts = (n_good_pairs / max(n_total_pairs, 1)) * 2.5
            score += rxr_pts
            breakdown["reactivity"] = round(rxr_pts, 2)
        else:
            breakdown["reactivity"] = 2.5  # not applicable, full credit
            score += 2.5
    else:
        # Silane sol-gel doesn't have reactivity ratio issue
        breakdown["reactivity"] = 2.5
        score += 2.5

    # 4. Surface-only monomer handling (1.5 pts)
    surface_ms = [m for m in combo
                   if ALL_MONOMERS.get(m, {}).get("polymerization") == "surface"]
    if not surface_ms:
        # All monomers polymerizable in-mix — clean
        score += 1.5
        breakdown["surface_clean"] = 1.5
    elif len(surface_ms) == len(combo):
        # All surface-only — invalid (no matrix)
        breakdown["surface_clean"] = 0.0
    else:
        # Mixed surface + polymerizable — requires extra synthesis step
        # (grafting before polymerization) but doable
        score += 0.7
        breakdown["surface_clean"] = 0.7

    # 5. Boronate chemistry bonus (1 pt, for glycan targets)
    has_boronate = any(m in {"APBA", "VPBA", "FPBA", "AAPBA"} for m in combo)
    if has_boronate:
        score += 1.0
        breakdown["boronate_bonus"] = 1.0
    else:
        breakdown["boronate_bonus"] = 0.0

    return {
        "synth_score": round(score, 2),  # 0-10
        "breakdown": breakdown,
        "polymerization_type": dominant,
        "synthesizable": ok and (pH_min is not None) and (pH_max > pH_min if pH_max else False),
    }


def compute_selectivity_score(combo: list, target: str,
                              be_matrix: dict, all_targets: list) -> dict:
    """C2: Selectivity score for monomer combination (using Phase 2 ΔΔG).

    Higher value → combo monomers preferentially bind to target over others.
    """
    import numpy as np
    other_targets = [t for t in all_targets if t != target]
    ddg_per_monomer = []

    for m in combo:
        be_target = be_matrix.get(target, {}).get(m)
        if be_target is None:
            continue
        # Mean BE on other targets
        other_bes = [be_matrix.get(t, {}).get(m)
                     for t in other_targets if be_matrix.get(t, {}).get(m) is not None]
        if not other_bes:
            continue
        # ΔΔG = BE_target - mean(BE_other)
        # Negative ΔΔG = target preferred
        ddg = be_target - float(np.mean(other_bes))
        ddg_per_monomer.append(ddg)

    if not ddg_per_monomer:
        return {"selectivity_score": 0.0, "mean_ddg": None, "n_evaluated": 0}

    mean_ddg = float(np.mean(ddg_per_monomer))
    # Score: more negative ΔΔG = higher selectivity score
    # ΔΔG = -2.0 kcal/mol → score 1.0; ΔΔG = -4.0 → 2.0; clamped [0, 5]
    score = max(0.0, min(5.0, -mean_ddg))
    return {
        "selectivity_score": round(score, 2),
        "mean_ddg_kcal_mol": round(mean_ddg, 2),
        "n_evaluated": len(ddg_per_monomer),
    }


def compute_3obj_for_combo(combo: list, target: str,
                            mmsd_result: dict,
                            be_matrix: dict, all_targets: list) -> dict:
    """C2: Compute 3 objectives for NSGA-II.

    All in MINIMIZATION form (smaller = better):
      • affinity: -mmsd_per_monomer (more negative MMSD = better affinity)
      • selectivity: -selectivity_score (higher selectivity = better)
      • synthesizability: -(synth_score / 10)  (higher synth = better)

    Returns dict with raw + normalized values.
    """
    # Affinity from Phase 3 MMSD
    mmsd_per = mmsd_result.get("mmsd_per_monomer")
    affinity_obj = mmsd_per if mmsd_per is not None else 0.0  # lower MMSD = stronger

    # Selectivity from Phase 2 ΔΔG
    sel = compute_selectivity_score(combo, target, be_matrix, all_targets)
    selectivity_obj = -sel["selectivity_score"]  # negate to minimize

    # Synthesizability
    synth = compute_synthesizability_score(combo)
    synth_obj = -synth["synth_score"] / 10.0  # normalize [-1, 0], minimize

    return {
        "objectives": [affinity_obj, selectivity_obj, synth_obj],
        "raw": {
            "affinity_mmsd_per_monomer": affinity_obj,
            "selectivity_score": sel["selectivity_score"],
            "selectivity_mean_ddg": sel.get("mean_ddg_kcal_mol"),
            "synth_score": synth["synth_score"],
            "synth_breakdown": synth["breakdown"],
            "synthesizable": synth["synthesizable"],
        },
    }
