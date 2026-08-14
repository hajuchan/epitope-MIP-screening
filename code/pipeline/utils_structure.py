"""
Structure Utilities
===================
PDB/AlphaFold download, epitope extraction, peptide analysis,
and PDBQT preparation for AutoDock4 docking.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── ADFR (AutoDockTools) locations ─────────────────────────────
# prepare_receptor4.py is a PYTHON 2 script. It must be run with the python2.7
# that ships in the GROMACS conda env; running it under the MIPscreen python3
# is a SyntaxError, which is how it silently lost to obabel before.
_ADFR_PY27 = Path("~/anaconda3/envs/GROMACS/bin/python2.7").expanduser()
_ADFR_PREPARE_RECEPTOR = Path(
    "~/anaconda3/envs/GROMACS/bin/prepare_receptor4.py").expanduser()


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
    """
    Download AlphaFold predicted structure from EBI.
    Uses the API to get the latest version URL (v4→v6 migration).
    """
    import json
    import urllib.request
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / f"AF_{uniprot_id}.pdb"

    # Method 1: API query for latest PDB URL
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        with urllib.request.urlopen(api_url) as resp:
            data = json.loads(resp.read())
        if data and isinstance(data, list):
            pdb_url = data[0].get("pdbUrl")
            if pdb_url:
                urllib.request.urlretrieve(pdb_url, str(dst))
                version = pdb_url.split("_v")[-1].split(".")[0]
                logger.info(f"Downloaded AlphaFold {uniprot_id} v{version} → {dst}")
                return dst
    except Exception as e:
        logger.warning(f"AlphaFold API failed: {e}, trying direct URL")

    # Method 2: Try versions 6, 4, 3 in order
    for ver in [6, 4, 3]:
        url = (f"https://alphafold.ebi.ac.uk/files/"
               f"AF-{uniprot_id}-F1-model_v{ver}.pdb")
        try:
            urllib.request.urlretrieve(url, str(dst))
            logger.info(f"Downloaded AlphaFold {uniprot_id} v{ver} → {dst}")
            return dst
        except Exception:
            continue

    raise RuntimeError(f"Failed to download AlphaFold structure for {uniprot_id}")


def download_structure(target_cfg: dict, output_dir: Path) -> Path:
    """
    Obtain the target structure named by the config.

    PRECEDENCE: a `prepared_path` wins over source/pdb_id/uniprot_id.

    A prepared template is a CHIMERA (e.g. CD63 = 9HUR EC2 spliced into an
    AlphaFold membrane frame). It has no accession, so it can never be
    re-downloaded — it is COPIED from structures/prepared/ into the run
    directory. If the file is declared but missing we fail LOUD rather than
    silently falling through to the raw deposition: falling through would
    dock the very structure the prepared template exists to replace (a closed
    CD81 with a quarter of its surface in lipid, a CD9 missing T175-K179),
    and nothing downstream would record that the swap did not happen.
    """
    prepared = target_cfg.get("prepared_path")
    if prepared:
        prepared = Path(prepared)
        if not prepared.is_file():
            raise FileNotFoundError(
                f"prepared_path is declared but absent: {prepared}. "
                f"Rebuild with code/tools/prepare_cd_templates.py, or remove "
                f"the key to fall back to the raw deposition. Refusing to "
                f"silently dock the unprepared structure.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dst = output_dir / prepared.name
        shutil.copy2(prepared, dst)      # never re-download a chimera
        logger.info(f"Prepared template {prepared.name} → {dst} "
                    f"(source/pdb_id/uniprot_id bypassed)")
        return dst

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
                threshold: float = 70.0,
                only_predicted: bool = False) -> dict:
    """
    Check AlphaFold pLDDT scores for the epitope region.
    pLDDT is stored in the B-factor column of AlphaFold PDBs.

    only_predicted
        On a PREPARED template the B-factor column is MIXED UNITS: pLDDT
        (0-100, higher better) over the AlphaFold scaffold, but deposited
        crystallographic B (Å², lower better) over any spliced experimental
        block. Averaging the two produces a number that means nothing — on
        CD63 the 1.65 Å 9HUR graft carries B ≈ 20-40, which reads as
        "catastrophically low pLDDT" for the single best-determined part of
        the structure. Set True to restrict the statistic to the residues
        that really are predicted, identified by the occupancy convention
        the template builder writes: 0.00 predicted, 0.50 rebuilt, 0.75
        donor-spliced, 1.00 experimental.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("af", str(pdb_path))
    scores = []
    n_skipped = 0
    for residue in structure.get_residues():
        resid = residue.get_id()[1]
        if residue_range[0] <= resid <= residue_range[1]:
            atoms = list(residue.get_atoms())
            if only_predicted:
                atoms = [a for a in atoms if a.get_occupancy() is not None
                         and a.get_occupancy() < 0.25]
                if not atoms:
                    n_skipped += 1
                    continue
            bfactors = [a.get_bfactor() for a in atoms]
            if bfactors:
                scores.append(np.mean(bfactors))

    if not scores:
        # On a fully experimental prepared template this is the CORRECT
        # outcome, not a failure: there is no predicted residue to score.
        if only_predicted:
            return {"mean_plddt": None, "min_plddt": None,
                    "n_predicted_residues": 0,
                    "n_experimental_residues": n_skipped,
                    "scope": "predicted-only",
                    "pass": True,
                    "message": (f"No predicted residues in {residue_range} — "
                                f"{n_skipped} residue(s) are experimental or "
                                f"rebuilt; pLDDT does not apply")}
        return {"mean_plddt": 0.0, "min_plddt": 0.0,
                "pass": False, "message": "No residues found"}

    mean_plddt = float(np.mean(scores))
    min_plddt = float(np.min(scores))
    low_confidence = [i for i, s in enumerate(scores) if s < threshold]

    return {
        "mean_plddt": round(mean_plddt, 1),
        "min_plddt": round(min_plddt, 1),
        "n_low_confidence": len(low_confidence),
        "n_predicted_residues": len(scores),
        "n_experimental_residues": n_skipped,
        "scope": "predicted-only" if only_predicted else "all-residues",
        "pass": mean_plddt >= threshold,
        "message": ("OK" if mean_plddt >= threshold
                     else f"Low confidence: mean pLDDT={mean_plddt:.1f}"),
    }


# ── pH titration model ─────────────────────────────────────────
#
# WHY THIS EXISTS.  MD_SOLVENT_PH was wired all the way through Phase 1 and
# then had NO physical effect: propka is not installed, the ImportError branch
# reduced to shutil.copy2, and pdb2gmx ran with no -asp/-glu/-lys/-his/-ter
# flags.  Measured consequence on BSA: the built topology summed the pH-7 census
# (ASP -39, GLU -59, LYS +59, ARG +23, HIS +1) to a protein charge of -16 e and
# genion added 16 Na+, while the protocol runs at pH ~9.5 where the same census
# titrates to about -28 e.  The pH knob is now LIVE: this model decides the
# states, setup_protein_topology drives pdb2gmx with them, and Phase 4 asserts
# the built charge against the prediction.
#
# Standard model pKa values (Nozaki & Tanford; Grimsley/Pace 2009 for the
# free-cysteine and terminal values).  Site-specific shifts need propka, which
# this model uses IN PREFERENCE when it is importable.
_MODEL_PKA = {
    "ASP": 3.9, "GLU": 4.3, "HIS": 6.0, "CYS": 8.5,
    "TYR": 10.1, "LYS": 10.5, "ARG": 12.5,
    "NTERM": 8.0, "CTERM": 3.6,
}
# Charge of the PROTONATED and DEPROTONATED form of each titratable group.
_TITRATION_CHARGE = {                      # (q_protonated, q_deprotonated)
    "ASP": (0, -1), "GLU": (0, -1), "HIS": (+1, 0), "CYS": (0, -1),
    "TYR": (0, -1), "LYS": (+1, 0), "ARG": (+1, 0),
    "NTERM": (+1, 0), "CTERM": (0, -1),
}
# Which non-default states GROMACS' amber99sb-ildn port can actually build.
# aminoacids.rtp defines ASH, GLH, HID/HIE/HIP, LYN, CYM, CYX — but NOT a
# deprotonated tyrosine, and ARGN maps to "-" in aminoacids.r2b.  A residue
# whose majority state is not on this list is a KNOWN, RECORDED gap, never a
# silent substitution.
_FF_REPRESENTABLE_STATES = {
    "ASP": {"protonated": "ASH", "deprotonated": "ASP"},
    "GLU": {"protonated": "GLH", "deprotonated": "GLU"},
    "HIS": {"protonated": "HIP", "deprotonated": "HIE"},
    "LYS": {"protonated": "LYS", "deprotonated": "LYN"},
    "CYS": {"protonated": "CYS", "deprotonated": "CYM"},
    "NTERM": {"protonated": "NH3+", "deprotonated": "NH2"},
    "CTERM": {"protonated": "COOH", "deprotonated": "COO-"},
    "TYR": {"protonated": "TYR", "deprotonated": None},   # no TYD in this ff
    "ARG": {"protonated": "ARG", "deprotonated": None},   # ARGN unavailable
}


def _hh_fraction_protonated(pka: float, ph: float) -> float:
    """Henderson-Hasselbalch fraction of a group that is protonated."""
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def detect_disulfide_cysteines(pdb_path: Path, cutoff_A: float = 2.5) -> set:
    """Residue ids of cysteines whose SG is within `cutoff_A` of another SG.

    A bridged cysteine does not titrate — it has no thiol proton.  Without this
    the standard-pKa model deprotonates all 35 of BSA's cysteines above pH 8.5
    and predicts a protein charge ~34 e too negative; BSA has 17 disulfides and
    exactly ONE free cysteine (Cys34).  pdb2gmx detects the same bonds by
    distance and builds them as CYS2/CYX, so the two agree by construction.
    """
    sg = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        if line[12:16].strip() != "SG" or line[17:20].strip() not in ("CYS", "CYX", "CYM"):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        sg.append((line[22:27].strip(), xyz))
    bridged = set()
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            dx = sg[i][1][0] - sg[j][1][0]
            dy = sg[i][1][1] - sg[j][1][1]
            dz = sg[i][1][2] - sg[j][1][2]
            if (dx * dx + dy * dy + dz * dz) ** 0.5 <= cutoff_A:
                bridged.add(sg[i][0])
                bridged.add(sg[j][0])
    return bridged


def propka_pkas(pdb_path: Path, work_dir: Path = None) -> dict:
    """Site-specific pKa values from PROPKA 3, as {(chain, resid): pka}.

    Returns {} when propka is not installed or the run fails — the caller then
    falls back to the model pKa table, which is what `titration_model` does by
    default. Never raises: a missing pKa predictor must degrade to the textbook
    values, not stop the pipeline.

    Why this matters here: on BSA at pH 9.5 the model table and PROPKA disagree
    by a lot at individual sites. PROPKA puts Lys221 at 6.61 and Lys106 at 7.51
    — both fully deprotonated at the working pH — where the textbook 10.5 would
    call them protonated. Those shifts are most of the difference between a
    net charge of -18 e and the -31.9 e PROPKA computes for the ensemble.
    """
    import shutil as _sh
    import subprocess as _sp
    import tempfile as _tf

    exe = _sh.which("propka3")
    if exe is None:
        logger.info("  propka3 not on PATH — falling back to model pKa values")
        return {}
    pdb_path = Path(pdb_path)
    work_dir = Path(work_dir) if work_dir else Path(_tf.mkdtemp(prefix="propka_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / pdb_path.name
    if local.resolve() != pdb_path.resolve():
        _sh.copy2(pdb_path, local)
    try:
        r = _sp.run([exe, local.name], cwd=str(work_dir),
                    capture_output=True, text=True, timeout=900)
    except Exception as e:                       # noqa: BLE001 - degrade, never stop
        logger.warning("  propka3 failed to run (%s) — using model pKa values", e)
        return {}
    pka_file = local.with_suffix(".pka")
    if r.returncode != 0 or not pka_file.exists():
        logger.warning("  propka3 exited %d / no .pka written — using model pKa "
                       "values", r.returncode)
        return {}

    # SUMMARY block: "RESNAME  RESID CHAIN  pKa  model-pKa"
    out, in_summary = {}, False
    for line in pka_file.read_text(errors="replace").splitlines():
        if line.startswith("SUMMARY"):
            in_summary = True
            continue
        if in_summary:
            if line.startswith("-") or not line.strip():
                if out:
                    break
                continue
            p = line.split()
            if len(p) >= 4 and p[0] in _MODEL_PKA:
                try:
                    # resid as STRING: _iter_pdb_residues yields the raw PDB
                    # column, so an int key here would never match and the
                    # overrides would be silently ignored.
                    out[(p[2], str(int(p[1])))] = float(p[3])
                except (ValueError, IndexError):
                    continue
    logger.info("  PROPKA: %d site-specific pKa values from %s",
                len(out), pka_file.name)
    return out


def _net_charge_assignment(per_site_fracs, majority_states, hh_q, disc_q,
                           flippable):
    """Rule B: choose the discrete microstate whose TOTAL charge matches the
    ensemble average, instead of putting every site in its own majority state.

    Rule A (majority) is the maximum-likelihood microstate: each site
    independently takes whichever state is more probable. Its flaw is that the
    per-site errors do not cancel. Most lysines sit above the working pH, so
    every one of them is rounded the same way (up), and the errors accumulate:
    on BSA at pH 9.5 the sum of Henderson-Hasselbalch fractions over 59 lysines
    is 46.58 while the majority rule counts 59, a systematic +12.4 e.

    Rule B flips the least-confident sites — those whose fraction is closest to
    0.5, i.e. the ones the majority rule was least sure about — until the total
    matches. Each flip moves the total by exactly 1 e for every group here.

    The trade is explicit and is not free: the flipped sites end up in their
    MINORITY state, so their LOCAL charge is wrong where rule A had it right.
    Rule B buys a correct net charge and long-range field at the cost of a few
    wrong surface patches; rule A buys locally-likely sites at the cost of a
    net charge that is systematically off. Which is better depends on whether
    the physics of interest is long-range (net charge dominates) or contact-
    level (local charge dominates). See PH_CHARGE_ASSIGNMENT in config.
    """
    need = int(round(hh_q - disc_q))          # negative => must shed charge
    if need == 0:
        return {}, 0
    want_less = need < 0
    # Candidates are sites currently in the state we need to leave AND whose
    # flipped state the force field can actually build. Skipping that second
    # test is a trap: flipping a tyrosine to tyrosinate changes the arithmetic
    # but not the topology, because amber99sb-ildn has no tyrosinate residue
    # and pdb2gmx silently builds the neutral form anyway. The flip would then
    # move `discrete_charge` while `representable_charge` — the one the MD
    # actually gets — stayed put.
    cands = []
    for key, f in per_site_fracs.items():
        if not flippable.get(key, False):
            continue
        is_prot = majority_states[key]
        if want_less and is_prot:
            cands.append((f, key))            # least protonated first
        elif (not want_less) and (not is_prot):
            cands.append((-f, key))           # most nearly protonated first
    cands.sort()
    flips = {k: (not majority_states[k]) for _, k in cands[:abs(need)]}
    return flips, len(flips)


def titration_model(pdb_path: Path, ph: float, pka_overrides: dict = None,
                    disulfide_cys: set = None,
                    unavailable_states: set = None,
                    assignment: str = "majority",
                    use_propka: bool = False) -> dict:
    """Per-residue protonation states at `ph`, plus what the force field can build.

    Returns three charges that must never be conflated:

      `hh_continuum_charge`  the Henderson-Hasselbalch expectation, in which a
                             group can be 91% protonated.  This is the number a
                             titration curve gives and the one people quote.
      `discrete_charge`      the best a FIXED-CHARGE force field can do: every
                             group takes its majority state.  This is what the
                             MD actually simulates.
      `representable_charge` the discrete charge after dropping the states this
                             force field has no residue type for (tyrosinate,
                             arginine base).  This is what pdb2gmx will build.

    The gap between them is not a rounding error — on BSA at pH 9.5 it is about
    10 e — and it is reported rather than hidden, because it is the honest
    limit of a fixed-charge model at a pH near several side-chain pKa values.
    """
    if assignment not in ("majority", "net_charge"):
        raise ValueError(
            f"assignment must be 'majority' or 'net_charge', got {assignment!r}")
    pka_overrides = dict(pka_overrides or {})
    # PROPKA supplies site-specific pKa values; explicit overrides still win.
    propka_used = False
    if use_propka:
        _p = propka_pkas(pdb_path)
        if _p:
            propka_used = True
            for k, v in _p.items():
                pka_overrides.setdefault(k, v)
    # (group, state) pairs the ENGINE cannot build, on top of the ones this
    # module already knows are missing from amber99sb-ildn. The caller supplies
    # them because they are a property of the force-field FILES, not of the
    # chemistry: e.g. GROMACS' amber99sb-ildn port ships EMPTY aminoacids.n.tdb
    # and .c.tdb, so `pdb2gmx -ter` has nothing to offer and the termini are
    # fixed by the NASP/CALA rtp entries — measured, not assumed.
    unavailable_states = set(unavailable_states or ())
    if disulfide_cys is None:
        disulfide_cys = detect_disulfide_cysteines(pdb_path)
    disulfide_cys = set(disulfide_cys)
    residues = [r for r in _iter_pdb_residues(pdb_path)
                if r[2] not in _SOLVENT_RESNAMES]

    sites, seen_chains, last_of_chain = [], set(), {}
    for chain, resseq, resname, atoms in residues:
        if resname not in _STANDARD_AA:
            continue
        if chain not in seen_chains:
            seen_chains.add(chain)
            sites.append({"chain": chain, "resid": resseq, "resname": resname,
                          "group": "NTERM"})
        last_of_chain[chain] = (chain, resseq, resname, atoms)
        if resname in ("ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG"):
            if resname == "CYS" and resseq in disulfide_cys:
                continue                      # a disulfide cysteine does not titrate
            sites.append({"chain": chain, "resid": resseq, "resname": resname,
                          "group": resname})
    for chain, (c, resseq, resname, atoms) in last_of_chain.items():
        if "OXT" in atoms:
            sites.append({"chain": c, "resid": resseq, "resname": resname,
                          "group": "CTERM"})

    # PASS 1 — fractions and the majority state of every site. Both assignment
    # rules start here; rule B then flips a minimal set (see
    # _net_charge_assignment).
    _pka_of, _frac, _major, _flippable = {}, {}, {}, {}
    _hh_pre, _repr_pre = 0.0, 0

    def _buildable(group, protonated):
        st = _FF_REPRESENTABLE_STATES[group][
            "protonated" if protonated else "deprotonated"]
        return st is not None and (
            group, "protonated" if protonated else "deprotonated"
        ) not in unavailable_states

    for s in sites:
        g = s["group"]
        key = (s["chain"], s["resid"], g)
        pka = float(pka_overrides.get((s["chain"], s["resid"]), _MODEL_PKA[g]))
        qp, qd = _TITRATION_CHARGE[g]
        f = _hh_fraction_protonated(pka, ph)
        maj = pka > ph
        _pka_of[key], _frac[key], _major[key] = pka, f, maj
        # A site is a usable flip candidate only if BOTH states are buildable.
        _flippable[key] = _buildable(g, maj) and _buildable(g, not maj)
        _hh_pre += f * qp + (1.0 - f) * qd
        # Baseline must be the REPRESENTABLE charge: that is what pdb2gmx will
        # build, and therefore what the flips have to close the gap on.
        q_maj = qp if maj else qd
        _repr_pre += q_maj if _buildable(g, maj) else (qd if maj else qp)
    flips, n_flipped = ({}, 0)
    if assignment == "net_charge":
        flips, n_flipped = _net_charge_assignment(
            _frac, _major, _hh_pre, _repr_pre, _flippable)

    hh_q = 0.0
    disc_q = 0
    repr_q = 0
    unrepresentable = []
    per_site = []
    for s in sites:
        g = s["group"]
        _key = (s["chain"], s["resid"], g)
        pka = _pka_of[_key]
        qp, qd = _TITRATION_CHARGE[g]
        f = _frac[_key]
        hh_q += f * qp + (1.0 - f) * qd
        protonated = flips.get(_key, _major[_key])
        q_disc = qp if protonated else qd
        disc_q += q_disc
        _which = "protonated" if protonated else "deprotonated"
        state = _FF_REPRESENTABLE_STATES[g][_which]
        if (g, _which) in unavailable_states:
            state = None
        if state is None:
            # Force field cannot build this state: it will be built in the
            # OTHER state, so the representable charge takes that instead.
            other = qp if not protonated else qd
            repr_q += other
            unrepresentable.append(
                {"chain": s["chain"], "resid": s["resid"], "group": g,
                 "pka": pka, "wanted_charge": q_disc, "built_charge": other,
                 "reason": ("amber99sb-ildn has no residue type for the "
                            f"{'deprotonated' if not protonated else 'protonated'} "
                            f"form of {g}")})
        else:
            repr_q += q_disc
        per_site.append({"chain": s["chain"], "resid": s["resid"],
                         "group": g, "pka": pka,
                         "fraction_protonated": round(f, 4),
                         "majority_state": "protonated" if protonated else "deprotonated",
                         "ff_residue": state, "discrete_charge": q_disc})

    by_group = {}
    for p in per_site:
        b = by_group.setdefault(p["group"], {"n": 0, "hh": 0.0, "discrete": 0})
        b["n"] += 1
        qp, qd = _TITRATION_CHARGE[p["group"]]
        b["hh"] += p["fraction_protonated"] * qp + (1 - p["fraction_protonated"]) * qd
        b["discrete"] += p["discrete_charge"]
    for b in by_group.values():
        b["hh"] = round(b["hh"], 3)

    return {
        "ph": ph,
        "assignment": assignment,
        "propka_used": propka_used,
        "n_sites_flipped_for_net_charge": n_flipped,
        "sites_flipped_for_net_charge": sorted(
            f"{g}{r}" for (_c, r, g) in flips),
        "majority_rule_charge": int(_repr_pre),
        "model_pka": dict(_MODEL_PKA),
        "unavailable_states": sorted(f"{g}:{s}" for g, s in unavailable_states),
        "n_disulfide_cysteines": len(disulfide_cys),
        "disulfide_cysteines": sorted(disulfide_cys),
        "n_titratable_sites": len(sites),
        "hh_continuum_charge": round(hh_q, 3),
        "discrete_charge": int(disc_q),
        "representable_charge": int(repr_q),
        "charge_residual_vs_hh": round(hh_q - repr_q, 3),
        "unrepresentable_sites": unrepresentable,
        "by_group": by_group,
        "sites": per_site,
    }


# ── Protonation State ──────────────────────────────────────────

def assign_protonation_states(pdb_path: Path, output_path: Path = None,
                               ph: float = 7.4) -> Path:
    """
    Assign protonation states at specified pH.

    PROPKA when it is importable (site-specific pKa shifts), otherwise the
    Henderson-Hasselbalch model in `titration_model` on standard pKa values.
    # BEHAVIOUR CHANGE (2026-08): the no-propka branch used to be a verbatim
    # shutil.copy2 that recorded nothing, so a pH the user set was invisible in
    # the run record AND in the physics. The model is now always computed and
    # always written to the sidecar, and it is what setup_protein_topology
    # drives pdb2gmx with.

    The OUTPUT FILE ITSELF still carries default residue names unless
    PROTONATION_APPLY_STATES is on: the same PDB is the AutoDock receptor, and
    renaming HIS->HIP there changes what prepare_receptor4 builds. The MD
    states are applied at the pdb2gmx step instead, which is MD-only.

    Reference states at pH 7.4: His pKa ~6.0 -> neutral; Asp/Glu ~3.9-4.3 ->
    charged; Lys ~10.5 -> charged; Cys ~8.5 -> neutral (SH).
    """
    pdb_path = Path(pdb_path)
    if output_path is None:
        # The name used to be hardcoded to "_pH74" whatever `ph` was, which is
        # how a pH-9.5 request produced a file called *_pH74.pdb.
        _tag = f"{float(ph):g}".replace(".", "p")
        output_path = pdb_path.with_name(pdb_path.stem + f"_pH{_tag}.pdb")

    # ALWAYS computed, propka or not: this is what makes MD_SOLVENT_PH real.
    try:
        model = titration_model(pdb_path, ph)
        logger.info(
            "pH %.2f titration model for %s: HH continuum charge %+.2f e, "
            "discrete (fixed-charge) %+d e, force-field representable %+d e "
            "(%d site(s) the force field cannot build)",
            ph, pdb_path.name, model["hh_continuum_charge"],
            model["discrete_charge"], model["representable_charge"],
            len(model["unrepresentable_sites"]))
        if abs(model["charge_residual_vs_hh"]) > 1.0:
            logger.error(
                "pH %.2f: the fixed-charge model can only reach %+d e where the "
                "Henderson-Hasselbalch expectation is %+.2f e — a residual of "
                "%+.2f e, carried by groups titrating within ~1 pH unit of this "
                "pH (and by tyrosinate, which amber99sb-ildn cannot build). This "
                "is the honest limit of a fixed-charge simulation here; it is "
                "recorded, not hidden.",
                ph, model["representable_charge"], model["hh_continuum_charge"],
                model["charge_residual_vs_hh"])
    except Exception as e:
        logger.error(f"titration model failed for {pdb_path}: {e}")
        model = {"error": str(e), "ph": ph}

    # Try PROPKA
    try:
        import propka.run as propka_run
        import propka.molecular_container
        import os

        # PROPKA writes .pka file to cwd — change to output directory
        original_cwd = os.getcwd()
        os.chdir(str(output_path.parent))
        try:
            mol = propka_run.single(str(pdb_path))
        finally:
            os.chdir(original_cwd)
        # Get pKa predictions
        protonation_changes = []
        for group in mol.conformations["AVR"].groups:
            if group.residue_type in ("HIS", "CYS", "ASP", "GLU", "LYS"):
                pka = group.pka_value
                resid = group.atom.res_num
                resname = group.residue_type

                if resname == "HIS":
                    # HIS: if pKa > pH → protonated (HIP), else neutral (HID/HIE)
                    if pka > ph:
                        protonation_changes.append(
                            (resid, resname, "HIP", f"pKa={pka:.1f}>pH"))
                    # else: default (neutral) is correct
                elif resname == "CYS":
                    # CYS: if pKa < pH → deprotonated (CYM, rare)
                    if pka < ph:
                        protonation_changes.append(
                            (resid, resname, "CYM", f"pKa={pka:.1f}<pH"))

        if protonation_changes:
            logger.info(f"PROPKA protonation changes at pH {ph}:")
            for resid, orig, new, reason in protonation_changes:
                logger.info(f"  {orig}{resid} → {new} ({reason})")

        propka_pka = pdb_path.with_suffix(".pka")
        n_written = _apply_protonation_renames(
            pdb_path, output_path, protonation_changes)
        status = ("applied" if n_written
                  else "propka_ran_but_no_states_written" if protonation_changes
                  else "no_changes_predicted")
        _write_protonation_sidecar(
            output_path, status=status, propka_ran=True,
            pka_file=str(propka_pka) if propka_pka.exists() else None,
            changes=protonation_changes, n_written=n_written, ph=ph,
            model=model, model_source="propka")
        return output_path

    except ImportError:
        logger.error(
            "propka is not installed (pip install propka), so no SITE-SPECIFIC "
            "pKa shifts are available. Falling back to the standard-pKa "
            "Henderson-Hasselbalch model in titration_model(), which IS "
            "recorded in the sidecar and IS what setup_protein_topology drives "
            "pdb2gmx with — the pH is no longer inert. The output PDB itself "
            "keeps default residue names because it is also the docking "
            "receptor; the MD states are applied at the pdb2gmx step.")
        import shutil
        shutil.copy2(str(pdb_path), str(output_path))
        _write_protonation_sidecar(
            output_path, status="model_pka_henderson_hasselbalch",
            propka_ran=False, pka_file=None, changes=[], n_written=0, ph=ph,
            model=model, model_source="standard_pka_hh",
            error="propka not importable (site-specific shifts unavailable)")
        return output_path

    except Exception as e:
        logger.error(f"PROPKA FAILED: {e}. Falling back to the standard-pKa "
                     f"Henderson-Hasselbalch model at pH {ph}, which is "
                     f"recorded in the sidecar and applied at pdb2gmx.")
        import shutil
        shutil.copy2(str(pdb_path), str(output_path))
        _write_protonation_sidecar(
            output_path, status="model_pka_henderson_hasselbalch",
            propka_ran=False, pka_file=None, changes=[], n_written=0, ph=ph,
            model=model, model_source="standard_pka_hh", error=str(e))
        return output_path


# Residue renames PROPKA can ask for, and whether the rest of the toolchain can
# actually consume them.  GROMACS' amber99sb-ildn port defines HID/HIE/HIP/CYM/
# CYX/ASH/GLH/LYN in aminoacids.rtp, and utils_structure.expected_formal_charge
# already understands them, so the names below are meaningful downstream.
_PROTONATION_RENAMES = {"HIP", "CYM", "HID", "HIE", "ASH", "GLH", "LYN"}


def _protonation_apply_enabled() -> bool:
    """PROTONATION_APPLY_STATES — read via getattr so config need not define it.

    DEFAULT False, deliberately.  Writing HIP/CYM into the PDB is the correct
    end state, and the code below does it properly, but the SAME file is the
    docking receptor: ADFR's prepare_receptor4 assigns its own types and
    charges, and no one has yet verified what it does with a HIP.  Turning this
    on changes the receptor's net charge by +1 per HIP, which
    verify_receptor_charge WILL notice — that is the intended way to validate
    it.  Until then the prediction is recorded, logged at ERROR and NOT applied,
    which is a loud omission rather than a silent wrong number.
    """
    try:
        from . import config as _cfg
        return bool(getattr(_cfg, "PROTONATION_APPLY_STATES", False))
    except Exception:
        return False


def _apply_protonation_renames(pdb_path: Path, output_path: Path,
                                changes: list) -> int:
    """Write output_path from pdb_path, renaming residues PROPKA flagged.

    Returns the number of residues actually renamed (0 when the feature is off,
    when PROPKA predicted nothing, or when nothing matched).

    THE DEFECT THIS REPLACES: all four branches of assign_protonation_states
    used to end in the same `shutil.copy2`, so the HIP/CYM assignments PROPKA
    computed were logged and then thrown away.  A missing PROPKA, a crashed
    PROPKA and a successful PROPKA produced byte-identical output.
    """
    import shutil
    wanted = {(int(resid), new) for resid, _orig, new, _why in changes
              if new in _PROTONATION_RENAMES}
    if not wanted:
        shutil.copy2(str(pdb_path), str(output_path))
        return 0

    if not _protonation_apply_enabled():
        shutil.copy2(str(pdb_path), str(output_path))
        logger.error(
            "PROPKA predicted %d protonation change(s) %s but "
            "PROTONATION_APPLY_STATES is False, so they were NOT written into "
            "%s. The structure carries DEFAULT protonation. Set "
            "PROTONATION_APPLY_STATES = True in config to apply them (and "
            "expect verify_receptor_charge to report the shifted net charge).",
            len(wanted), sorted(wanted), output_path.name)
        return 0

    by_resid = {resid: new for resid, new in wanted}
    renamed = set()
    out_lines = []
    for line in Path(pdb_path).read_text().splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
            try:
                resid = int(line[22:26])
            except ValueError:
                out_lines.append(line)
                continue
            new = by_resid.get(resid)
            if new:
                # PDB residue name is columns 18-20 (0-indexed 17:20), left-
                # justified in a 3-wide field. Every name here is exactly 3.
                line = line[:17] + f"{new:<3}" + line[20:]
                renamed.add(resid)
        out_lines.append(line)

    Path(output_path).write_text("".join(out_lines))
    missing = sorted(set(by_resid) - renamed)
    if missing:
        logger.error("PROPKA asked to rename residues %s but they were not "
                     "found in %s — those states were NOT applied.",
                     missing, pdb_path.name)
    logger.info("Applied %d protonation rename(s) into %s: %s",
                len(renamed), output_path.name,
                sorted((r, by_resid[r]) for r in renamed))
    return len(renamed)


def _write_protonation_sidecar(output_path: Path, *, status: str,
                                propka_ran: bool, pka_file, changes: list,
                                n_written: int, ph: float,
                                error: str = None, model: dict = None,
                                model_source: str = None) -> None:
    """Write <output_stem>_protonation.json next to the protonated structure.

    phase1_epitope_prep._verify_protonation reads this sidecar in preference to
    its own heuristic probe, so this is what makes "PROPKA ran and wrote states"
    distinguishable from "PROPKA was never installed" in the run record.  The
    two used to be indistinguishable: same output bytes, same absence of logs.
    """
    import json as _json
    out = Path(output_path)
    payload = {
        "status": status,
        "propka_ran": bool(propka_ran),
        "pka_file": str(pka_file) if pka_file else None,
        "ph": ph,
        "n_protonation_changes": len(changes or []),
        "n_states_written": int(n_written),
        "states_applied": bool(n_written),
        "apply_enabled": _protonation_apply_enabled(),
        "changes": [
            {"resid": r, "from": o, "to": n, "reason": w}
            for r, o, n, w in (changes or [])
        ],
        "error": error,
        # THE pH MODEL ITSELF. Without this the sidecar recorded that nothing
        # happened, and a reader could not tell whether pH 9.5 had been
        # simulated, approximated or ignored.
        "model_source": model_source,
        "titration_model": model,
        "hh_continuum_charge": (model or {}).get("hh_continuum_charge"),
        "discrete_charge": (model or {}).get("discrete_charge"),
        "representable_charge": (model or {}).get("representable_charge"),
        "charge_residual_vs_hh": (model or {}).get("charge_residual_vs_hh"),
    }
    try:
        out.with_name(out.stem + "_protonation.json").write_text(
            _json.dumps(payload, indent=1), encoding="utf-8")
    except Exception as e:
        logger.error(f"could not write protonation sidecar for {out}: {e}")


# ── Receptor Formal Charge Accounting (pH 7.4) ─────────────────
#
# BLOCKER 02. The obabel `-h` + Gasteiger route has NO pH MODEL. It adds the
# hydrogens a neutral Lewis structure wants, so every Asp/Glu comes out as a
# neutral -COOH and every Lys as a neutral -NH2. Measured on the committed
# receptors the whole ECL2 summed to |q| < 0.06 e — electrostatically inert
# protein. The missing ionic term is 0.5-3.0 kcal/mol per charged contact
# against a total library ranking spread of 1.99 / 3.48 / 2.41 kcal/mol, and
# it is TARGET-ASYMMETRIC (formal charge +3 / -3 / -2 for CD63 / CD81 / CD9),
# so it does not cancel in the own-minus-cross difference that is this
# pipeline's only deliverable.
#
# These helpers exist so that "the receptor is charged" is an ASSERTED,
# LOGGED number rather than an assumption.

# The 20 standard residues, plus the protonation-state aliases that GROMACS
# and the ADFR tools emit. Anything outside this set makes the expected charge
# INDETERMINATE (see expected_formal_charge) rather than silently wrong.
_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # protonation / tautomer aliases
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "CYS2",
    "ASH", "GLH", "LYN", "ARN", "MSE",
}

# Formal side-chain charge at pH 7.4.
#   HIS pKa ~6.0 -> NEUTRAL. HIP is the explicitly-protonated form and only
#   appears when an upstream tool decided so; then it really is +1.
#   ASH/GLH/LYN are the explicitly-neutralised forms.
#   CYM (thiolate, pKa ~8.3 shifted down) is -1; free CYS at pH 7.4 is neutral.
_SIDECHAIN_CHARGE = {
    "ARG": +1, "LYS": +1, "ASP": -1, "GLU": -1,
    "HIP": +1, "HSP": +1,
    "HIS": 0, "HID": 0, "HIE": 0, "HSD": 0, "HSE": 0,
    "ASH": 0, "GLH": 0, "LYN": 0, "ARN": 0,
    "CYM": -1, "CYX": 0, "CYS": 0, "CYS2": 0,
}

# Neutral solvent under every naming convention this project produces
# (PDB: HOH/WAT; GROMACS: SOL; CHARMM/AMBER water models: TIP3/T3P/DOD).
_SOLVENT_RESNAMES = {"HOH", "WAT", "SOL", "TIP", "TIP3", "T3P", "DOD", "H2O"}


def _iter_pdb_residues(pdb_path: Path):
    """
    Yield (chain, resseq_icode, resname, set_of_atom_names) in file order.

    Deliberately a text parse and not Bio.PDB: this runs on merged
    receptor+ligand files written by utils_autodock.merge_ligand_into_receptor,
    which are truncated at column 66 and would make a strict parser guess.
    """
    cur_key = None
    cur = None
    for line in Path(pdb_path).read_text().split("\n"):
        if line.startswith("TER") or line.startswith("END"):
            if cur is not None:
                yield cur
                cur, cur_key = None, None
            continue
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        key = (line[21], line[22:27])
        if key != cur_key:
            if cur is not None:
                yield cur
            cur_key = key
            cur = (line[21], line[22:27].strip(), line[17:20].strip(), set())
        cur[3].add(line[12:16].strip())
    if cur is not None:
        yield cur


def expected_formal_charge(pdb_path: Path) -> dict:
    """
    Formal net charge of a protein structure at pH 7.4, in units of e.

    Accounts for the two TERMINUS conventions the ADFR tools apply to a
    CHOPPED loop, which are not the same at both ends:
      * the N-terminal residue of every chain gets an -NH3+ (+1);
      * the C-terminus is only -COO- (-1) when the input carries an OXT atom.
        An ECL2 sliced out of a full-length model has no OXT, so its C-terminus
        is a neutral truncated carbonyl and contributes 0.

    Returns {"charge", "determinate", "n_pos", "n_neg", "n_termini_pos",
             "n_termini_neg", "n_residues", "nonstandard"}.
    `determinate` is False when any residue is outside _STANDARD_AA — a merged
    MMSD receptor carries docked monomers as HETATM, and their charge is not
    derivable from a residue name. Callers must NOT assert on the number in
    that case.
    """
    residues = list(_iter_pdb_residues(pdb_path))
    # Solvent is neutral and gets removed by the receptor-prep cleanup, so it
    # must not make the charge INDETERMINATE — an MD conformer straight out of
    # trjconv would then never be charge-checked at all.
    nonstandard = sorted({r[2] for r in residues
                          if r[2] not in _STANDARD_AA
                          and r[2] not in _SOLVENT_RESNAMES})

    # chain breaks: first residue of each chain segment is an N-terminus
    n_pos = n_neg = 0
    charge = 0
    seen_chains = set()
    for chain, _resseq, resname, atoms in residues:
        if resname in _SOLVENT_RESNAMES:
            continue
        q = _SIDECHAIN_CHARGE.get(resname, 0)
        charge += q
        n_pos += (q > 0)
        n_neg += (q < 0)

    n_term = n_cterm = 0
    n_breaks = 0
    prev = {}          # chain -> last residue sequence number
    for chain, resseq, resname, atoms in residues:
        if resname not in _STANDARD_AA:
            continue
        if chain not in seen_chains:
            seen_chains.add(chain)
            n_term += 1
        else:
            # A NUMBERING GAP is a chain break, and whether the tools put a
            # charged -NH3+ on the residue after it is not predictable from
            # the file. Counted, not guessed: it widens the tolerance in
            # verify_receptor_charge by 1 e per break instead of turning a
            # bookkeeping ambiguity into a false pipeline abort.
            try:
                if int(resseq.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) \
                        != prev[chain] + 1:
                    n_breaks += 1
            except (ValueError, KeyError):
                pass
        try:
            prev[chain] = int(resseq.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        except ValueError:
            pass
        if "OXT" in atoms:
            n_cterm += 1
    charge += n_term - n_cterm

    return {
        "charge": charge,
        "determinate": not nonstandard,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_termini_pos": n_term,
        "n_termini_neg": n_cterm,
        "n_chain_breaks": n_breaks,
        "n_residues": len(residues),
        "nonstandard": nonstandard,
    }


def pdbqt_net_charge(pdbqt_path: Path) -> tuple:
    """Sum of the PDBQT partial-charge column (cols 71-76). Returns (q, natoms)."""
    total = 0.0
    n = 0
    for line in Path(pdbqt_path).read_text().split("\n"):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 76:
            try:
                total += float(line[70:76])
            except ValueError:
                continue
            n += 1
    return round(total, 4), n


# Gasteiger sums land within ~0.06 e of the integer on a 1000-atom ECL2
# (measured: +4.027 / -1.964 / -0.972 against +4 / -2 / -1). 0.25 e is far
# enough away to never fire on rounding and close enough to catch a single
# missing ionisable group, which is worth 1.00 e.
RECEPTOR_CHARGE_TOL = 0.25


def verify_receptor_charge(pdb_path: Path, pdbqt_path: Path,
                           tol: float = RECEPTOR_CHARGE_TOL) -> dict:
    """
    Assert that a receptor PDBQT actually carries its pH-7.4 formal charge.

    Returns a dict with "ok" ∈ {True, False, None}. None means the expected
    charge is INDETERMINATE (non-standard residues present, e.g. an MMSD
    merged receptor) — not a pass and not a failure.
    """
    exp = expected_formal_charge(pdb_path)
    net, natoms = pdbqt_net_charge(pdbqt_path)
    # +1 e of slack per chain break: the tools' treatment of the residue after
    # a gap is not derivable from the file (see expected_formal_charge).
    eff_tol = tol + exp.get("n_chain_breaks", 0)
    out = {"expected_charge": exp["charge"], "pdbqt_net_charge": net,
           "n_atoms": natoms, "tol": eff_tol, "detail": exp}
    if not exp["determinate"]:
        out["ok"] = None
        out["message"] = (f"indeterminate: non-standard residues "
                          f"{exp['nonstandard'][:6]}")
        return out
    out["ok"] = abs(net - exp["charge"]) <= eff_tol
    out["message"] = (f"net={net:+.3f} e vs formal {exp['charge']:+d} e "
                      f"(tol {eff_tol:.2f}; {exp['n_pos']} cationic / "
                      f"{exp['n_neg']} anionic side chains, "
                      f"{exp['n_termini_pos']} N-term+, "
                      f"{exp['n_termini_neg']} C-term-, "
                      f"{exp.get('n_chain_breaks', 0)} chain break(s))")
    return out


# ── PDBQT Preparation ──────────────────────────────────────────

def _adfr_available() -> bool:
    return _ADFR_PY27.exists() and _ADFR_PREPARE_RECEPTOR.exists()


def _allow_neutral_receptor_fallback() -> bool:
    """
    Read the escape hatch from config, defaulting to REFUSE.

    Deliberately defensive: this module must keep importing on a config that
    predates the key. The DEFAULT IS CHANGED BEHAVIOUR — before this fix a
    charge-free receptor was accepted silently; now it raises.

    getattr, not `from .config import ...`, on purpose: the key does not exist
    in config yet, and test_config_regression.test_import_contract requires
    every name reached by a from-import to resolve under BOTH experiments.
    This reads the key the moment it is added and defaults to REFUSE until it
    is.
    """
    try:
        from . import config as _cfg
        return bool(getattr(_cfg, "RECEPTOR_PDBQT_ALLOW_NEUTRAL_FALLBACK",
                            False))
    except Exception:                                # noqa: BLE001
        return False


def prepare_receptor_pdbqt(pdb_path: Path, output_path: Path = None,
                           add_h: bool = True,
                           verify_charge: bool = True) -> Path:
    """
    Convert receptor PDB to rigid PDBQT for AutoDock4 **with pH-7.4 charges**.

    Strategy (CHANGED — obabel was Method 1 until 2026-08, see BLOCKER 02):
    1. ADFR prepare_receptor4.py under python2.7. This is the ONLY route here
       that applies a protonation model: Asp/Glu deprotonated, Lys/Arg
       protonated, His neutral, disulfide Cys left without HG. It reproduces
       the formal charge of all three tetraspanin ECL2 receptors.
    2. OpenBabel `-h --partialcharge gasteiger`. LOUDLY WARNED FALLBACK ONLY.
       obabel has no pH model; the receptor it writes is electrostatically
       neutral and its docking scores are missing every salt bridge.
    3. pybel, same defect as (2).

    After conversion the net charge is checked against the formal charge
    computed from the residue composition. A determinate mismatch RAISES
    unless config.RECEPTOR_PDBQT_ALLOW_NEUTRAL_FALLBACK is set — a silently
    neutral receptor is the exact failure this function exists to prevent.
    """
    pdb_path = Path(pdb_path)
    if output_path is None:
        output_path = pdb_path.with_suffix(".pdbqt")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    method = None
    errors = []

    # prepare_receptor4's DEFAULT cleanup is 'nphs_lps_waters_nonstdres', and
    # 'nonstdres' DELETES any chain made only of non-standard residues. On a
    # sequential-MMSD merged receptor that chain is the monomer docked in the
    # previous round: verified, the UNL block vanished and the output was
    # byte-identical to the bare protein (832 atoms either way). Dropping the
    # 'nonstdres' term keeps it (844 atoms). Waters and lone pairs are still
    # removed, and non-polar hydrogens are still merged, exactly as before.
    _has_nonstd = bool(expected_formal_charge(pdb_path)["nonstandard"])
    _cleanup = "nphs_lps_waters" if _has_nonstd else "nphs_lps_waters_nonstdres"

    # ── Method 1: ADFR prepare_receptor4.py (python2.7) — HAS A pH MODEL ──
    if _adfr_available():
        try:
            cmd = [str(_ADFR_PY27), str(_ADFR_PREPARE_RECEPTOR),
                   "-r", str(pdb_path), "-o", str(output_path),
                   "-U", _cleanup]
            if add_h:
                cmd.extend(["-A", "hydrogens"])
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=300)
            if result.returncode == 0 and output_path.exists() \
                    and output_path.stat().st_size > 0:
                method = "ADFR prepare_receptor4 (python2.7)"
            else:
                errors.append(f"prepare_receptor4 rc={result.returncode}: "
                              f"{(result.stderr or result.stdout)[:300]}")
        except Exception as e:                       # noqa: BLE001
            errors.append(f"prepare_receptor4 raised: {e}")
    else:
        errors.append(f"ADFR not found at {_ADFR_PREPARE_RECEPTOR} / "
                      f"{_ADFR_PY27}")

    # ── Method 2: OpenBabel — NO pH MODEL, neutral receptor ──
    if method is None:
        logger.error(
            "RECEPTOR PREP FELL BACK TO OPENBABEL for %s. obabel has NO pH "
            "model: Asp/Glu will be neutral acids and Lys/Arg neutral amines, "
            "so every salt bridge (0.5-3.0 kcal/mol) is missing from the "
            "docking score and the error is target-asymmetric. Reason(s): %s",
            pdb_path.name, "; ".join(errors) or "unknown")
        try:
            cmd = ["obabel", str(pdb_path), "-O", str(output_path),
                   "-xr",  # rigid receptor mode (no torsions)
                   "--partialcharge", "gasteiger"]
            if add_h:
                cmd.append("-h")
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=300)
            if result.returncode == 0 and output_path.exists():
                content = output_path.read_text()
                if "ATOM" in content or "HETATM" in content:
                    method = "obabel (NO pH MODEL — neutral receptor)"
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            errors.append(f"obabel: {e}")

    # ── Method 3: pybel — same defect as Method 2 ──
    if method is None:
        try:
            from openbabel import pybel
            mol = next(pybel.readfile("pdb", str(pdb_path)))
            if add_h:
                mol.addh()
            mol.write("pdbqt", str(output_path), overwrite=True)
            method = "pybel (NO pH MODEL — neutral receptor)"
        except Exception as e:                       # noqa: BLE001
            errors.append(f"pybel: {e}")

    if method is None:
        raise RuntimeError(
            f"No PDBQT converter succeeded for receptor {pdb_path}. "
            f"Docking against a PDB copied to a .pdbqt name would produce "
            f"numbers, and they would be meaningless. Reasons: "
            f"{'; '.join(errors)}")

    logger.info(f"Receptor PDBQT [{method}] → {output_path}")

    if verify_charge:
        chk = verify_receptor_charge(pdb_path, output_path)
        if chk["ok"] is None:
            logger.info(f"  charge check skipped ({chk['message']}); "
                        f"net={chk['pdbqt_net_charge']:+.3f} e")
        elif chk["ok"]:
            logger.info(f"  charge OK: {chk['message']}")
        else:
            msg = (f"RECEPTOR IS NOT CHARGED AT pH 7.4: {output_path.name} "
                   f"{chk['message']} [method: {method}]. Docking this "
                   f"receptor drops every ionic contact from the score and "
                   f"does so unequally across targets, which corrupts the "
                   f"selectivity ΔΔG.")
            if _allow_neutral_receptor_fallback():
                logger.error(msg + " CONTINUING because "
                             "RECEPTOR_PDBQT_ALLOW_NEUTRAL_FALLBACK is set.")
            else:
                raise RuntimeError(
                    msg + " Set config.RECEPTOR_PDBQT_ALLOW_NEUTRAL_FALLBACK"
                          " = True to proceed anyway (the results are then "
                          "not comparable across targets).")

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
        # Fallback 1: ETKDG
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if status != 0:
        # Fallback 2: random coordinates (for large molecules like Lipid IVA)
        params2 = AllChem.ETKDGv3()
        params2.useRandomCoords = True
        params2.maxIterations = 5000
        params2.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params2)
    if status != 0:
        raise ValueError(f"Failed to generate 3D coordinates for {name}")

    # MMFF94 optimization (fallback to UFF for non-standard atoms)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            logger.warning(f"  {name}: optimization failed, using raw coordinates")

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
    # Detect non-standard atoms (Si, B) that need proxy substitution
    nonstandard = {}
    for a in mol.GetAtoms():
        anum = a.GetAtomicNum()
        if anum == 14:   # Si → S proxy
            nonstandard.setdefault("Si", []).append(a.GetIdx())
        elif anum == 5:  # B → C proxy
            nonstandard.setdefault("B", []).append(a.GetIdx())

    # Make a copy with proxy substitutions for Gasteiger charge calculation.
    # The 3D COORDINATE of each substituted atom is recorded here and is what
    # identifies it again in the PDBQT. Element/order-based matching was wrong:
    # see _restore_atom_in_pdbqt.
    mol_for_pdbqt = mol
    proxy_xyz = {}
    if nonstandard:
        conf = mol.GetConformer()
        rw = Chem.RWMol(mol)
        for idx in nonstandard.get("Si", []):
            rw.GetAtomWithIdx(idx).SetAtomicNum(16)  # S proxy
        for idx in nonstandard.get("B", []):
            rw.GetAtomWithIdx(idx).SetAtomicNum(6)   # C proxy
        mol_for_pdbqt = rw.GetMol()
        for elem, idxs in nonstandard.items():
            proxy_xyz[elem] = [tuple(conf.GetAtomPosition(i)) for i in idxs]
        subs = ", ".join(f"{k}({len(v)})" for k, v in nonstandard.items())
        logger.info(f"  {name}: proxy substitution for PDBQT: {subs}")

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
        # CHANGED DEFAULT: this used to write a ZERO-BYTE .pdbqt and return it.
        # Every downstream cache tests `pdbqt.exists()`, so the empty file was
        # a permanent poison entry — the monomer could never be regenerated
        # and every docking against it silently produced nothing.
        pdbqt_path.unlink(missing_ok=True)
        _legacy_stamp_path(pdbqt_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"All PDBQT writers (meeko, obabel, prepare_ligand4) failed for "
            f"{name} ({smiles}). No file was written, so the monomer will be "
            f"regenerated on the next attempt rather than cached empty.")

    # Restore non-standard atom types in PDBQT, matched BY COORDINATE
    if "Si" in nonstandard:
        pdbqt_text = _restore_atom_in_pdbqt(
            pdbqt_text, proxy_types=("S", "SA"), target_type="Si",
            positions=proxy_xyz["Si"], label=name)
    if "B" in nonstandard:
        pdbqt_text = _restore_atom_in_pdbqt(
            pdbqt_text, proxy_types=("C", "A"), target_type=" B",
            positions=proxy_xyz["B"], label=name)

    pdbqt_path.write_text(pdbqt_text)
    write_smiles_stamp(pdbqt_path, name, smiles)

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

    # PDB → mol2 via obabel (use tempdir to avoid spaces in path)
    mol2_path = output_dir / f"{name}.mol2"
    try:
        import sys, shutil, tempfile
        obabel_bin = shutil.which("obabel") or "obabel"

        # Work in tempdir to avoid space-in-path issues with older obabel
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pdb = Path(tmpdir) / f"{name}.pdb"
            tmp_mol2 = Path(tmpdir) / f"{name}.mol2"
            shutil.copy2(pdb_path, tmp_pdb)
            result = subprocess.run(
                [obabel_bin, str(tmp_pdb), "-O", str(tmp_mol2)],
                capture_output=True, text=True,
            )
            if tmp_mol2.exists():
                shutil.copy2(tmp_mol2, mol2_path)
            else:
                logger.warning(f"obabel mol2 failed for {name}: "
                               f"{result.stderr[:200]}")
    except Exception as e:
        logger.warning(f"obabel not available ({e}), using RDKit .mol")
        Chem.MolToMolFile(mol, str(output_dir / f"{name}.mol"))
        mol2_path = output_dir / f"{name}.mol"

    if not mol2_path.exists():
        logger.error(f"Failed to create mol2 for {name}")

    return mol2_path


# ── BLAST Epitope Uniqueness Check (Bossi 2021) ────────────────

def check_epitope_uniqueness(sequence: str, target_name: str,
                              max_hits: int = 10) -> dict:
    """
    BLAST the epitope sequence against human proteome to verify
    it is unique to the target protein.

    Bossi 2021: "alignment of the 7-12 residues with all protein
    sequences stored in UniProtKB using BLAST software"

    Uses NCBI BLAST REST API (requires internet).
    """
    import urllib.request
    import urllib.parse
    import time as _time
    import xml.etree.ElementTree as ET

    result = {
        "sequence": sequence,
        "target": target_name,
        "length": len(sequence),
    }

    # Submit BLAST query
    params = urllib.parse.urlencode({
        "CMD": "Put",
        "PROGRAM": "blastp",
        "DATABASE": "swissprot",
        "QUERY": sequence,
        "ENTREZ_QUERY": "Homo sapiens[organism]",
        "EXPECT": "10",
        "HITLIST_SIZE": str(max_hits),
    })

    try:
        req = urllib.request.Request(
            "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi",
            data=params.encode(),
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode()

        # Extract RID
        rid = None
        for line in text.split("\n"):
            if "RID = " in line:
                rid = line.split("=")[1].strip()
                break

        if not rid:
            return {**result, "status": "error", "reason": "No RID from BLAST"}

        # Poll for results (max 2 minutes)
        logger.info(f"BLAST submitted (RID={rid}), waiting for results...")
        for _ in range(60):  # 5 minutes max wait
            _time.sleep(5)
            check_url = (f"https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?"
                         f"CMD=Get&FORMAT_TYPE=XML&RID={rid}")
            with urllib.request.urlopen(check_url, timeout=30) as resp:
                xml_text = resp.read().decode()
            if "Status=WAITING" in xml_text:
                continue
            if "Status=FAILED" in xml_text:
                return {**result, "status": "error", "reason": "BLAST failed"}

            # Parse XML results
            root = ET.fromstring(xml_text)
            hits = []
            for hit in root.iter("Hit"):
                hit_def = hit.findtext("Hit_def", "")
                hit_acc = hit.findtext("Hit_accession", "")
                for hsp in hit.iter("Hsp"):
                    identity = float(hsp.findtext("Hsp_identity", "0"))
                    align_len = float(hsp.findtext("Hsp_align-len", "1"))
                    pct_id = round(identity / align_len * 100, 1)
                    evalue = hsp.findtext("Hsp_evalue", "N/A")
                    hits.append({
                        "protein": hit_def[:80],
                        "accession": hit_acc,
                        "pct_identity": pct_id,
                        "evalue": evalue,
                    })

            # Check uniqueness: the target protein should be #1 hit
            # Other hits with >70% identity = potential cross-reactivity
            is_unique = True
            cross_reactive = []
            for h in hits:
                if target_name.lower() not in h["protein"].lower():
                    if h["pct_identity"] >= 70:
                        is_unique = False
                        cross_reactive.append(h)

            result["status"] = "PASS" if is_unique else "WARN"
            result["n_hits"] = len(hits)
            result["hits"] = hits[:5]
            result["is_unique"] = is_unique
            result["cross_reactive"] = cross_reactive
            if cross_reactive:
                logger.warning(
                    f"Epitope may cross-react with: "
                    f"{[h['protein'][:40] for h in cross_reactive]}"
                )
            else:
                logger.info(f"Epitope uniqueness: PASS ({len(hits)} hits, "
                            f"no cross-reactive proteins)")
            return result

        return {**result, "status": "timeout"}

    except Exception as e:
        logger.warning(f"BLAST check failed: {e}")
        return {**result, "status": "error", "reason": str(e)}


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


def _subset_coords(pdb_path: Path, residues=None) -> np.ndarray:
    """Atom coordinates of `residues` (a list/set of resids), or of all atoms."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", str(pdb_path))
    keep = set(residues) if residues else None
    coords = [a.get_vector().get_array() for a in structure.get_atoms()
              if keep is None or a.get_parent().get_id()[1] in keep]
    return np.array(coords)


def compute_grid_center_residues(pdb_path: Path, residues=None) -> tuple:
    """
    Geometric centre of an explicit residue subset.

    Used to centre the docking box on the POLYMER-ACCESSIBLE surface only.
    Centring on the whole ECL2 puts the box centre inside the protein and
    spends a large part of the box on surface that faces lipid on an intact
    vesicle, which is not a uniform cost across targets: measured on the
    receptors actually docked, 25/89 = 28% of the CD81 box surface is
    lipid-facing against 9/79 = 11% for CD9.
    """
    coords = _subset_coords(pdb_path, residues)
    if len(coords) == 0:
        return compute_grid_center(pdb_path)
    return tuple(round(float(c), 3) for c in coords.mean(axis=0))


def compute_grid_size_residues(pdb_path: Path, residues=None,
                               padding: float = 8.0) -> tuple:
    """Grid box that encompasses an explicit residue subset + padding."""
    from .config import AUTODOCK4_SPACING
    coords = _subset_coords(pdb_path, residues)
    if len(coords) == 0:
        return compute_grid_size(pdb_path, padding=padding)
    span = coords.max(axis=0) - coords.min(axis=0)
    npts = tuple(int(np.ceil((s + 2 * padding) / AUTODOCK4_SPACING))
                 for s in span)
    return tuple(n + (n % 2) for n in npts)


def superpose_onto_reference(mobile_pdb: Path, reference_pdb: Path,
                             output_pdb: Path = None) -> dict:
    """
    Rigid-body superpose `mobile_pdb` onto `reference_pdb` on shared CA atoms.

    NECESSARY FOR ENSEMBLE DOCKING. `gmx trjconv -pbc mol` writes frames in
    the SIMULATION BOX frame: the solute was centred in a solvation box by
    editconf and then diffused and rotated during the run, so an extracted
    frame is translated and rotated with respect to the crystal structure the
    AutoGrid box was computed from. Docking such a frame against that box
    searches empty solvent. Measured on the committed CD run, where this step
    was missing: the CD63 MD conformers sat 48-70 Å from their own grid
    centre with 0-5% of their atoms inside the box, and CD9's sat 21-35 Å
    away with 17-81% inside.

    Returns {"rmsd", "n_atoms", "translation", "output"}.
    """
    from Bio.PDB import PDBParser, PDBIO, Superimposer

    parser = PDBParser(QUIET=True)
    ref = parser.get_structure("ref", str(reference_pdb))
    mob = parser.get_structure("mob", str(mobile_pdb))

    def ca_list(struct):
        return [(r.get_id()[1], r.get_resname(), r["CA"])
                for r in struct.get_residues() if "CA" in r]

    ref_l, mob_l = ca_list(ref), ca_list(mob)

    # Residue NUMBERING is not shared between the two frames. GROMACS
    # renumbers from 1 when it builds the topology, so an ECL2 written as
    # 103-203 comes back out of trjconv as 1-101. Pair on identity where the
    # numbering agrees, then on a constant offset, then positionally — but
    # only ever when the residue NAMES corroborate the pairing, so a wrong
    # correspondence cannot slip through as a plausible-looking fit.
    ref_by_key = {(n, nm): ca for n, nm, ca in ref_l}
    shared = [((n, nm), ref_by_key[(n, nm)], ca)
              for n, nm, ca in mob_l if (n, nm) in ref_by_key]
    strategy = "resid+resname"

    if len(shared) < 3:
        ref_by_num = {n: (nm, ca) for n, nm, ca in ref_l}
        best = (0, None)
        for off in {n - m for n, _, _ in ref_l for m, _, _ in mob_l}:
            hits = sum(1 for m, nm, _ in mob_l
                       if ref_by_num.get(m + off, (None,))[0] == nm)
            if hits > best[0]:
                best = (hits, off)
        hits, off = best
        if hits >= max(3, 0.9 * min(len(ref_l), len(mob_l))):
            shared = [((m + off, nm), ref_by_num[m + off][1], ca)
                      for m, nm, ca in mob_l
                      if ref_by_num.get(m + off, (None,))[0] == nm]
            strategy = f"resid offset {off:+d}"
        elif (len(ref_l) == len(mob_l)
              and [nm for _, nm, _ in ref_l] == [nm for _, nm, _ in mob_l]):
            shared = [((rn, rnm), rca, mca)
                      for (rn, rnm, rca), (_, _, mca) in zip(ref_l, mob_l)]
            strategy = "positional (identical sequence)"

    if len(shared) < 3:
        raise ValueError(
            f"Cannot superpose {Path(mobile_pdb).name} onto "
            f"{Path(reference_pdb).name}: no consistent CA correspondence "
            f"({len(ref_l)} vs {len(mob_l)} CA residues)")

    sup = Superimposer()
    sup.set_atoms([r for _, r, _ in shared], [m for _, _, m in shared])
    sup.apply(list(mob.get_atoms()))          # move ALL atoms, not just CA

    output_pdb = Path(output_pdb) if output_pdb else Path(mobile_pdb)
    io = PDBIO()
    io.set_structure(mob)
    io.save(str(output_pdb))

    return {
        "rmsd": round(float(sup.rms), 3),
        "n_atoms": len(shared),
        "strategy": strategy,
        "translation": [round(float(v), 2) for v in sup.rotran[1]],
        "output": str(output_pdb),
    }


def fraction_atoms_in_box(pdb_path: Path, center: tuple, npts: tuple,
                          spacing: float = None) -> float:
    """
    Fraction of a structure's atoms lying inside an AutoGrid box.

    The cheap assertion that catches a misplaced receptor before 27 monomers
    are docked against empty solvent. A correctly superposed conformer of the
    structure the box was built from returns ~1.0.
    """
    from Bio.PDB import PDBParser
    if spacing is None:
        from .config import AUTODOCK4_SPACING
        spacing = AUTODOCK4_SPACING
    parser = PDBParser(QUIET=True)
    coords = np.array([a.get_coord() for a in
                       parser.get_structure("m", str(pdb_path)).get_atoms()])
    if len(coords) == 0:
        return 0.0
    c = np.asarray(center, dtype=float)
    half = np.asarray(npts, dtype=float) * spacing / 2.0
    inside = np.all((coords >= c - half) & (coords <= c + half), axis=1)
    return round(float(inside.mean()), 4)


def compute_surface_area(pdb_path: Path, residues=None) -> float:
    """
    Solvent-accessible surface area (Å²) of a structure, optionally summed
    over an explicit residue subset only.

    This is the per-target SCALE that makes a cross-target docking-score
    comparison meaningful: a bigger receptor offers more places to sit, so
    absolute binding energies are not exchangeable between targets of
    different size. Returns 0.0 if Bio.PDB's Shrake-Rupley is unavailable,
    and callers must treat 0.0 as "unknown", never as "no surface".
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.SASA import ShrakeRupley
    except Exception as e:                       # pragma: no cover
        logger.warning(f"SASA unavailable ({e}) — returning 0.0")
        return 0.0
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("mol", str(pdb_path))
        model = next(structure.get_models())
        ShrakeRupley().compute(model, level="R")
        keep = set(residues) if residues else None
        total = sum(r.sasa for r in model.get_residues()
                    if getattr(r, "sasa", None) is not None
                    and (keep is None or r.get_id()[1] in keep))
        return round(float(total), 1)
    except Exception as e:                       # pragma: no cover
        logger.warning(f"SASA computation failed ({e}) — returning 0.0")
        return 0.0


# ── Internal Helpers ───────────────────────────────────────────

def _restore_atom_in_pdbqt(pdbqt_text: str, proxy_types: tuple,
                            target_type: str, positions,
                            label: str = "", tol: float = 0.05) -> str:
    """
    Restore non-standard atom types in PDBQT after proxy substitution.

    Matched BY 3D COORDINATE, not by file order.

    The previous version relabelled the FIRST line whose type was in
    `proxy_types`, which is only correct when the substituted atom happens to
    be the first such atom in the writer's output order. It usually is not:
      MPTMS  SCCC[Si](OC)(OC)OC        -> the THIOL sulfur was typed Si and
                                          the real silicon was left as S
      AAPBA  C=CC(=O)Nc1cccc(B(O)O)c1  -> the AMIDE CARBONYL carbon was typed
                                          B and the real boron left as C
    Both were reproduced on the committed code. The magnitude today is small
    (0.055-0.143 kcal/mol against a 0.26-0.34 kcal/mol noise floor) but it is
    a chemistry-scrambling error, not a numerical one, and it grows with the
    library.

    Neither meeko, obabel nor prepare_ligand4 MOVES an atom, so the RDKit
    conformer coordinate of each substituted atom identifies its PDBQT line
    exactly. A position that matches no line, or matches a line whose current
    type is not a proxy type, RAISES — a mistyped heteroatom must never reach
    AutoGrid quietly.

    Parameters
    ----------
    proxy_types : proxy atom type strings to accept (e.g. ("S", "SA"))
    target_type : 2-char target type to restore (e.g. "Si" or " B")
    positions   : iterable of (x, y, z) for the atoms to restore
    tol         : max distance (Å) for a coordinate match; PDBQT stores 3
                  decimals, so a true match is < 0.001 Å away
    """
    positions = [tuple(float(v) for v in p) for p in positions]
    lines = pdbqt_text.split("\n")

    # index every ATOM/HETATM line by its coordinate
    atom_lines = []
    for i, line in enumerate(lines):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            try:
                xyz = (float(line[30:38]), float(line[38:46]),
                       float(line[46:54]))
            except ValueError:
                continue
            atom_lines.append((i, xyz))

    tag = f"{label}: " if label else ""
    used = set()
    for px, py, pz in positions:
        best_i, best_d = None, float("inf")
        for i, (x, y, z) in atom_lines:
            if i in used:
                continue
            d = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        if best_i is None or best_d > tol:
            raise RuntimeError(
                f"{tag}cannot locate the {target_type.strip()} atom at "
                f"({px:.3f}, {py:.3f}, {pz:.3f}) in the PDBQT "
                f"(nearest atom {best_d:.3f} Å away, tol {tol} Å). The PDBQT "
                f"writer moved or dropped atoms; the proxy substitution "
                f"cannot be undone and this ligand would dock with the wrong "
                f"element.")
        line = lines[best_i]
        if len(line) < 79:
            line = line.ljust(79)
        current = line[77:79].strip()
        if current not in proxy_types:
            raise RuntimeError(
                f"{tag}atom at ({px:.3f}, {py:.3f}, {pz:.3f}) should carry a "
                f"proxy type from {proxy_types} before restoring "
                f"'{target_type.strip()}', but the PDBQT types it '{current}'."
                f" Refusing to relabel an atom that is not the substituted "
                f"one.")
        lines[best_i] = line[:77] + target_type + line[79:]
        used.add(best_i)

    if used:
        logger.info(f"  Restored {len(used)} {target_type.strip()} atom(s) "
                    f"in PDBQT (coordinate-matched)")
    return "\n".join(lines)


# ── SMILES Stamps (cache-staleness guard) ──────────────────────
#
# The expensive artefact here is a FILE that gets copied into hundreds of work
# directories, and the pipeline's caches key on the file NAME. When a SMILES is
# corrected — nine silanes had no Si-C bond, APBA/AAPBA were para instead of
# meta, TRIM was missing a CH2 — every already-materialised MONOMER.pdbqt keeps
# the OLD molecule under the NEW name, and nothing in the file says so.
#
# The REMARK SMILES line meeko writes is NOT usable for this: for a silane it
# records the Si->S PROXY smiles, so it never equals the library entry.
# Verified on MPTMS: meeko writes "SCCCS(OC)(OC)OC", the library says
# "SCCC[Si](OC)(OC)OC".
#
# Hence a sidecar, .smiles_stamps.json, written next to every finalised PDBQT.

SMILES_STAMP_FILE = ".smiles_stamps.json"

# phase3_mmsd.py already ships a PER-FILE sidecar, "<lig>.pdbqt.smiles.json",
# holding sha256(library_smiles)[:16] under the key "smiles_sha256", and
# _get_monomer_pdbqt REJECTS any cached PDBQT that does not carry it. Both are
# written here so the two schemes agree instead of each calling the other's
# artefacts unstamped. Do not change either name without changing the other.
LEGACY_STAMP_SUFFIX = ".pdbqt.smiles.json"


def _legacy_stamp_path(pdbqt_path: Path) -> Path:
    return Path(pdbqt_path).with_suffix(LEGACY_STAMP_SUFFIX)


def _legacy_stamp_hash(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:16]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _canonical_smiles(smiles: str) -> str:
    """Canonical SMILES, or the raw string if RDKit cannot parse it."""
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is not None:
            return Chem.MolToSmiles(m)
    except Exception:                                # noqa: BLE001
        pass
    return smiles


def stamp_path_for(pdbqt_path: Path) -> Path:
    return Path(pdbqt_path).parent / SMILES_STAMP_FILE


def read_smiles_stamps(directory: Path) -> dict:
    """All stamps in a directory ({} when the sidecar is absent or corrupt)."""
    p = Path(directory) / SMILES_STAMP_FILE
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as e:                           # noqa: BLE001
        logger.warning(f"Corrupt {p}: {e} — treating as empty")
        return {}


def write_smiles_stamp(pdbqt_path: Path, name: str, smiles: str) -> Path:
    """
    Record which molecule `pdbqt_path` was built from.

    Read-modify-write through a temp file + os.replace so a crash cannot leave
    a half-written sidecar that would then be read as "no stamp".
    """
    pdbqt_path = Path(pdbqt_path)
    sidecar = stamp_path_for(pdbqt_path)
    stamps = read_smiles_stamps(pdbqt_path.parent)
    stamps[name] = {
        "smiles": smiles,
        "smiles_canonical": _canonical_smiles(smiles),
        "pdbqt": pdbqt_path.name,
        "pdbqt_sha256": _sha256(pdbqt_path) if pdbqt_path.is_file() else None,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(sidecar.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(stamps, fh, indent=1, sort_keys=True)
        os.replace(tmp, sidecar)
    except Exception:                                # noqa: BLE001
        Path(tmp).unlink(missing_ok=True)
        raise

    # compatibility sidecar for phase3_mmsd._read_stamp
    try:
        _legacy_stamp_path(pdbqt_path).write_text(json.dumps({
            "name": name,
            "smiles": smiles,
            "smiles_sha256": _legacy_stamp_hash(smiles),
        }, indent=2), encoding="utf-8")
    except Exception as e:                           # noqa: BLE001
        logger.warning(f"Could not write legacy SMILES stamp for {name}: {e}")

    return sidecar


def copy_smiles_stamp(src_pdbqt: Path, dst_pdbqt: Path, name: str = None):
    """
    Carry a stamp along when a PDBQT is copied into a work directory.

    Without this the copy is unstamped, and an unstamped copy is exactly the
    artefact whose provenance the stamp exists to record.
    """
    src_pdbqt, dst_pdbqt = Path(src_pdbqt), Path(dst_pdbqt)
    name = name or src_pdbqt.stem
    entry = read_smiles_stamps(src_pdbqt.parent).get(name)
    if entry is None:
        return None
    return write_smiles_stamp(dst_pdbqt, name, entry["smiles"])


def verify_smiles_stamp(pdbqt_path: Path, name: str, expected_smiles: str,
                        require_stamp: bool = True) -> dict:
    """
    Refuse a PDBQT that was built from a different molecule than `name` now
    names in the library.

    Raises RuntimeError on a SMILES mismatch or a content-hash mismatch.
    `require_stamp=False` downgrades "no stamp at all" to a warning, which is
    what a legacy artefact predating the stamp needs; a MISMATCH always raises.
    """
    pdbqt_path = Path(pdbqt_path)
    entry = read_smiles_stamps(pdbqt_path.parent).get(name)

    if entry is None:
        # fall back to the phase3 per-file sidecar, which stores only a hash
        legacy = _legacy_stamp_path(pdbqt_path)
        if legacy.is_file():
            try:
                data = json.loads(legacy.read_text(encoding="utf-8"))
            except Exception:                        # noqa: BLE001
                data = {}
            if data.get("smiles_sha256"):
                if data["smiles_sha256"] != _legacy_stamp_hash(expected_smiles):
                    raise RuntimeError(
                        f"STALE PDBQT for {name}: {pdbqt_path} carries legacy "
                        f"stamp {data['smiles_sha256']} for "
                        f"{data.get('smiles')!r}, but the library now says "
                        f"{expected_smiles!r}. Delete and regenerate.")
                return {"ok": True, "smiles": data.get("smiles"),
                        "source": "legacy sidecar"}

    if entry is None:
        msg = (f"No SMILES stamp for {name} beside {pdbqt_path}. The file "
               f"cannot be shown to have been built from the current library "
               f"SMILES; delete it and regenerate.")
        if require_stamp:
            raise RuntimeError(msg)
        logger.warning(msg)
        return {"ok": None, "reason": "no stamp"}

    want, got = _canonical_smiles(expected_smiles), entry["smiles_canonical"]
    if want != got:
        raise RuntimeError(
            f"STALE PDBQT for {name}: {pdbqt_path} was built from "
            f"{entry['smiles']!r} but the library now says "
            f"{expected_smiles!r}. Delete the cached file (and any copy of it "
            f"in a docking work dir) and regenerate — docking it would score "
            f"the retired molecule under the current name.")

    if entry.get("pdbqt_sha256") and pdbqt_path.is_file():
        actual = _sha256(pdbqt_path)
        if actual != entry["pdbqt_sha256"]:
            raise RuntimeError(
                f"{pdbqt_path} has changed since it was stamped for {name} "
                f"(sha256 {actual[:12]} != {entry['pdbqt_sha256'][:12]}). The "
                f"stamp no longer describes the file; regenerate it.")
    return {"ok": True, "smiles": entry["smiles"]}


