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
