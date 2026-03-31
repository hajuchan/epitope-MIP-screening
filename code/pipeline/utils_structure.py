"""
Structure Utilities
===================
PDB/AlphaFold download, epitope extraction, peptide analysis,
and PDBQT preparation for AutoDock4 docking.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── PDB / AlphaFold Download ───────────────────────────────────

def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    """Download PDB file from RCSB."""
    from Bio.PDB import PDBList
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdbl = PDBList(verbose=False)
    fname = pdbl.retrieve_pdb_file(
        pdb_id, pdir=str(output_dir), file_format="pdb"
    )
    # PDBList saves as pdb{id}.ent — rename to {ID}.pdb
    src = Path(fname)
    dst = output_dir / f"{pdb_id.upper()}.pdb"
    if src.exists():
        src.rename(dst)
    logger.info(f"Downloaded PDB {pdb_id} → {dst}")
    return dst


def download_alphafold(uniprot_id: str, output_dir: Path) -> Path:
    """Download AlphaFold predicted structure from EBI."""
    import urllib.request
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    url = (f"https://alphafold.ebi.ac.uk/files/"
           f"AF-{uniprot_id}-F1-model_v4.pdb")
    dst = output_dir / f"AF_{uniprot_id}.pdb"
    urllib.request.urlretrieve(url, str(dst))
    logger.info(f"Downloaded AlphaFold {uniprot_id} → {dst}")
    return dst


def download_structure(target_cfg: dict, output_dir: Path) -> Path:
    """Download structure based on target config (PDB or AlphaFold)."""
    if target_cfg["source"] == "pdb":
        return download_pdb(target_cfg["pdb_id"], output_dir)
    elif target_cfg["source"] == "alphafold":
        return download_alphafold(target_cfg["uniprot_id"], output_dir)
    else:
        raise ValueError(f"Unknown source: {target_cfg['source']}")


# ── Epitope Extraction ─────────────────────────────────────────

def extract_epitope(pdb_path: Path, chain_id: str,
                    residue_range: tuple, output_path: Path) -> Path:
    """
    Extract a peptide region from a PDB file.

    Parameters
    ----------
    pdb_path : Path to full PDB
    chain_id : Chain identifier (e.g., 'A')
    residue_range : (start_resid, end_resid) inclusive
    output_path : Where to save the extracted peptide PDB
    """
    from Bio.PDB import PDBParser, PDBIO, Select

    class ResidueSelect(Select):
        def accept_chain(self, chain):
            return chain.get_id() == chain_id

        def accept_residue(self, residue):
            resid = residue.get_id()[1]
            return residue_range[0] <= resid <= residue_range[1]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_path))

    io = PDBIO()
    io.set_structure(structure)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(output_path), ResidueSelect())

    # Count extracted residues
    structure2 = parser.get_structure("epitope", str(output_path))
    n_res = sum(1 for _ in structure2.get_residues())
    logger.info(f"Extracted {n_res} residues → {output_path}")
    return output_path


def get_epitope_sequence(pdb_path: Path) -> str:
    """Extract amino acid sequence from a peptide PDB file."""
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pep", str(pdb_path))
    seq = []
    for residue in structure.get_residues():
        resname = residue.get_resname().strip()
        if resname in protein_letters_3to1:
            seq.append(protein_letters_3to1[resname])
    return "".join(seq)


# ── Peptide Property Analysis ──────────────────────────────────

def analyze_peptide(pdb_path: Path, n_glycan_sites: int = 0) -> dict:
    """
    Compute physicochemical properties of an epitope peptide.

    Returns dict with: sequence, length, MW, pI, GRAVY,
    H-bond donors/acceptors, charge at pH 7, glycosylation info.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    seq = get_epitope_sequence(pdb_path)
    if not seq:
        return {"error": "No amino acid residues found"}

    pa = ProteinAnalysis(seq)

    # H-bond potential (side-chain donors/acceptors)
    hbd = sum(1 for aa in seq if aa in "STNQYWKRH")
    hba = sum(1 for aa in seq if aa in "STDENQY")

    # N-glycosylation sequons (NxS/T where x != P)
    glyco_sequons = []
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST":
            glyco_sequons.append(i)

    return {
        "sequence": seq,
        "length": len(seq),
        "molecular_weight": round(pa.molecular_weight(), 1),
        "isoelectric_point": round(pa.isoelectric_point(), 2),
        "gravy": round(pa.gravy(), 3),
        "aromaticity": round(pa.aromaticity(), 3),
        "charge_at_pH7": round(pa.charge_at_pH(7.0), 2),
        "hbond_donors": hbd,
        "hbond_acceptors": hba,
        "aromatic_residues": sum(1 for aa in seq if aa in "FWY"),
        "charged_residues": sum(1 for aa in seq if aa in "DEKRH"),
        "n_glycan_sites_known": n_glycan_sites,
        "n_glyco_sequons_detected": len(glyco_sequons),
        "glyco_sequon_positions": glyco_sequons,
    }


# ── AlphaFold pLDDT Check ──────────────────────────────────────

def check_plddt(pdb_path: Path, residue_range: tuple,
                threshold: float = 70.0) -> dict:
    """
    Check AlphaFold pLDDT scores for the epitope region.
    pLDDT is stored in the B-factor column of AlphaFold PDBs.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("af", str(pdb_path))
    scores = []
    for residue in structure.get_residues():
        resid = residue.get_id()[1]
        if residue_range[0] <= resid <= residue_range[1]:
            bfactors = [a.get_bfactor() for a in residue.get_atoms()]
            if bfactors:
                scores.append(np.mean(bfactors))

    if not scores:
        return {"mean_plddt": 0.0, "min_plddt": 0.0,
                "pass": False, "message": "No residues found"}

    mean_plddt = float(np.mean(scores))
    min_plddt = float(np.min(scores))
    low_confidence = [i for i, s in enumerate(scores) if s < threshold]

    return {
        "mean_plddt": round(mean_plddt, 1),
        "min_plddt": round(min_plddt, 1),
        "n_low_confidence": len(low_confidence),
        "pass": mean_plddt >= threshold,
        "message": ("OK" if mean_plddt >= threshold
                     else f"Low confidence: mean pLDDT={mean_plddt:.1f}"),
    }


# ── PDBQT Preparation ──────────────────────────────────────────

def prepare_receptor_pdbqt(pdb_path: Path, output_path: Path = None,
                           add_h: bool = True) -> Path:
    """
    Convert receptor PDB to rigid PDBQT for AutoDock4.

    Strategy:
    1. Try OpenBabel (best: adds Gasteiger charges + AD4 types)
    2. Try ADFR prepare_receptor4 (Python 2 only)
    3. Fallback: manual PDB→PDBQT with Gasteiger charges via RDKit
    """
    pdb_path = Path(pdb_path)
    if output_path is None:
        output_path = pdb_path.with_suffix(".pdbqt")
    output_path = Path(output_path)

    # Method 1: OpenBabel (best for receptor PDBQT)
    try:
        cmd = ["obabel", str(pdb_path), "-O", str(output_path),
               "-xr",  # rigid receptor mode (no torsions)
               "--partialcharge", "gasteiger"]
        if add_h:
            cmd.append("-h")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and output_path.exists():
            # Verify it has charges
            content = output_path.read_text()
            if "ATOM" in content or "HETATM" in content:
                logger.info(f"Receptor PDBQT (obabel) → {output_path}")
                return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: ADFR prepare_receptor4.py via Python 2.7
    # ADFR Suite scripts are Python 2 — must use python2.7 explicitly
    _gromacs_bin = Path(__file__).resolve().parent.parent.parent
    _py27 = Path("~/anaconda3/envs/GROMACS/bin/python2.7").expanduser()
    _prep_script = Path("~/anaconda3/envs/GROMACS/bin/prepare_receptor4.py").expanduser()
    if _py27.exists() and _prep_script.exists():
        try:
            cmd = [str(_py27), str(_prep_script),
                   "-r", str(pdb_path), "-o", str(output_path)]
            if add_h:
                cmd.extend(["-A", "hydrogens"])
            result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=60)
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Receptor PDBQT (ADFR python2.7) → {output_path}")
                return output_path
            else:
                logger.warning(f"prepare_receptor4: {result.stderr[:200]}")
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"prepare_receptor4 failed: {e}")

    # Method 3: OpenBabel pybel fallback
    try:
        from openbabel import pybel
        mol = next(pybel.readfile("pdb", str(pdb_path)))
        if add_h:
            mol.addh()
        mol.write("pdbqt", str(output_path), overwrite=True)
        logger.info(f"Receptor PDBQT (pybel) → {output_path}")
        return output_path
    except ImportError:
        pass

    logger.error("No PDBQT converter available for receptor. "
                 "Install OpenBabel or ensure ADFR python2.7 works.")
    # Copy PDB as-is (docking will fail, but won't crash pipeline)
    import shutil
    shutil.copy2(str(pdb_path), str(output_path))
    return output_path


def smiles_to_pdbqt(smiles: str, name: str, output_dir: Path) -> Path:
    """
    Convert SMILES to 3D-optimized PDBQT for AutoDock4 ligand.

    Steps: SMILES → RDKit 3D → MMFF94 optimize → PDB → PDBQT
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # ETKDGv3 embedding
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # fallback to less strict embedding
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # MMFF94 optimization
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    # Write PDB
    pdb_path = output_dir / f"{name}.pdb"
    Chem.MolToPDBFile(mol, str(pdb_path))

    # ── Convert to PDBQT ──
    # AutoDock4 has no Gasteiger parameters for Si.
    # Official workaround (autodock.scripps.edu/how-to-add-new-atom-types):
    #   1. Substitute Si→S for PDBQT generation (Gasteiger charges)
    #   2. Restore Si atom type in the PDBQT file
    #   3. Provide custom parameter_file with Si params (UFF-based)
    #      in GPF/DPF — handled by utils_autodock.py
    si_atom_indices = [a.GetIdx() for a in mol.GetAtoms()
                       if a.GetAtomicNum() == 14]
    has_si = len(si_atom_indices) > 0

    # Make a copy with Si→S for Gasteiger-compatible PDBQT generation
    mol_for_pdbqt = mol
    if has_si:
        rw = Chem.RWMol(mol)
        for idx in si_atom_indices:
            rw.GetAtomWithIdx(idx).SetAtomicNum(16)  # S as proxy
        mol_for_pdbqt = rw.GetMol()

    pdbqt_path = output_dir / f"{name}.pdbqt"
    pdbqt_text = None

    # Try meeko first
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        preparator = MoleculePreparation()
        mol_setup_list = preparator.prepare(mol_for_pdbqt)
        pdbqt_text = PDBQTWriterLegacy.write_string(mol_setup_list[0])[0]
    except Exception as e:
        logger.warning(f"meeko failed for {name}: {e}")

    # Fallback: obabel
    if pdbqt_text is None:
        sdf_path = output_dir / f"{name}_tmp.sdf"
        Chem.MolToMolFile(mol_for_pdbqt, str(sdf_path))
        try:
            result = subprocess.run(
                ["obabel", str(sdf_path), "-O", str(pdbqt_path),
                 "--partialcharge", "gasteiger"],
                capture_output=True, text=True,
            )
            if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
                pdbqt_text = pdbqt_path.read_text()
        except FileNotFoundError:
            pass

    # Fallback: prepare_ligand4.py via python2.7
    if pdbqt_text is None:
        _py27 = Path("~/anaconda3/envs/GROMACS/bin/python2.7").expanduser()
        _prep = Path("~/anaconda3/envs/GROMACS/bin/prepare_ligand4.py").expanduser()
        # prepare_ligand4 needs file in cwd
        import shutil
        tmp_pdb = output_dir / f"{name}_tmp.pdb"
        Chem.MolToPDBFile(mol_for_pdbqt, str(tmp_pdb))
        if _py27.exists() and _prep.exists():
            try:
                subprocess.run(
                    [str(_py27), str(_prep),
                     "-l", tmp_pdb.name, "-o", pdbqt_path.name],
                    cwd=str(output_dir),
                    capture_output=True, text=True, timeout=30,
                )
                if pdbqt_path.exists():
                    pdbqt_text = pdbqt_path.read_text()
            except Exception:
                pass

    if pdbqt_text is None:
        logger.error(f"All PDBQT methods failed for {name}")
        pdbqt_path.write_text("")
        return pdbqt_path

    # Restore Si atom type in PDBQT (replace proxy S/SA → Si)
    if has_si:
        pdbqt_text = _restore_si_in_pdbqt(pdbqt_text, len(si_atom_indices))

    pdbqt_path.write_text(pdbqt_text)

    logger.info(f"Monomer {name} → {pdbqt_path}")
    return pdbqt_path


def smiles_to_mol2(smiles: str, name: str, output_dir: Path) -> Path:
    """Convert SMILES to mol2 (for acpype GAFF2 parameterization)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    pdb_path = output_dir / f"{name}.pdb"
    Chem.MolToPDBFile(mol, str(pdb_path))

    # PDB → mol2 via obabel
    mol2_path = output_dir / f"{name}.mol2"
    try:
        subprocess.run(
            ["obabel", str(pdb_path), "-O", str(mol2_path), "--gen3d"],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        # If obabel not available, use RDKit mol2 writer
        Chem.MolToMolFile(mol, str(output_dir / f"{name}.mol"))
        mol2_path = output_dir / f"{name}.mol"
        logger.warning(f"obabel not found, saved as .mol: {mol2_path}")

    return mol2_path


# ── Grid Center Calculation ────────────────────────────────────

def compute_grid_center(pdb_path: Path) -> tuple:
    """Compute geometric center of a PDB structure."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", str(pdb_path))
    coords = [atom.get_vector().get_array()
              for atom in structure.get_atoms()]
    if not coords:
        return (0.0, 0.0, 0.0)
    center = np.mean(coords, axis=0)
    return tuple(round(float(c), 3) for c in center)


def compute_grid_size(pdb_path: Path, padding: float = 8.0) -> tuple:
    """Compute grid box size that encompasses the structure + padding."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", str(pdb_path))
    coords = np.array([atom.get_vector().get_array()
                        for atom in structure.get_atoms()])
    if len(coords) == 0:
        return (60, 60, 60)

    span = coords.max(axis=0) - coords.min(axis=0)
    # Convert to grid points at 0.375 Å spacing
    from .config import AUTODOCK4_SPACING
    npts = tuple(
        int(np.ceil((s + 2 * padding) / AUTODOCK4_SPACING))
        for s in span
    )
    # Ensure even number (AutoGrid requirement)
    npts = tuple(n + (n % 2) for n in npts)
    return npts


# ── Internal Helpers ───────────────────────────────────────────

def _restore_si_in_pdbqt(pdbqt_text: str, n_si: int) -> str:
    """
    Restore Si atom type in PDBQT after Si→S proxy generation.

    In PDBQT format, atom type occupies columns 77-78.
    S or SA atoms that were originally Si are replaced back to Si.
    We match by count (same order as RDKit atom indices).
    """
    restored = []
    si_restored = 0
    for line in pdbqt_text.split("\n"):
        if (line.startswith(("ATOM", "HETATM"))
                and len(line) >= 78
                and si_restored < n_si):
            atype = line[77:79].strip()
            if atype in ("S", "SA"):
                # Pad to 2 chars right-justified in columns 77-78
                line = line[:77] + "Si" + line[79:]
                si_restored += 1
        restored.append(line)
    if si_restored > 0:
        logger.info(f"  Restored {si_restored} Si atom(s) in PDBQT")
    return "\n".join(restored)


