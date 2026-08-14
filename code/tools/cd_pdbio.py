"""Minimal, dependency-light PDB reader/writer for CD template preparation.

Deliberately NOT Bio.PDB: the preparation tool needs the header records
(SEQADV / REMARK 465 / REMARK 470 / SSBOND / DBREF) as much as it needs the
coordinates, it needs to keep the occupancy column free for a provenance
channel, and it needs to round-trip atoms through pdbfixer without a parser
silently reinterpreting anything.

Nothing here writes into results/. Read-only on every input.

Author: prepared-template session 2026-08-12
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── amino-acid tables ──────────────────────────────────────────────────────
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # modified residues seen in the CD entries
    "MSE": "M",   # selenomethionine
    "YCM": "C",   # S-(2-amino-2-oxoethyl)-L-cysteine (iodoacetamide-alkylated Cys)
}
AA1TO3 = {v: k for k, v in AA3TO1.items() if k not in ("MSE", "YCM")}

BACKBONE = ("N", "CA", "C", "O")

# Atoms that survive when a modified residue is reduced to its parent.
YCM_KEEP = ("N", "CA", "C", "O", "CB", "SG")


# ── provenance channel ─────────────────────────────────────────────────────
# The occupancy column of every prepared template carries provenance, so a
# downstream reader can separate measured coordinates from invented ones with
# `awk '$1=="ATOM" && substr($0,55,6)+0 < 1.0'`.
OCC_EXPERIMENTAL = 1.00   # deposited coordinates from the primary entry
OCC_DONOR        = 0.75   # deposited coordinates from a secondary (spliced) entry
OCC_REBUILT_SC   = 0.50   # side-chain atom rebuilt (truncated / mutation-reverted)
OCC_DENOVO_BB    = 0.25   # backbone built de novo (loop closure)
OCC_PREDICTED    = 0.00   # AlphaFold scaffold

OCC_LABEL = {
    OCC_EXPERIMENTAL: "experimental (primary entry)",
    OCC_DONOR:        "experimental (spliced donor entry)",
    OCC_REBUILT_SC:   "REBUILT side chain (predicted rotamer)",
    OCC_DENOVO_BB:    "DE NOVO backbone (predicted)",
    OCC_PREDICTED:    "PREDICTED (AlphaFold)",
}


@dataclass
class Atom:
    name: str
    resname: str
    resnum: int
    x: float
    y: float
    z: float
    occ: float = 1.00
    b: float = 0.00
    element: str = ""
    icode: str = " "

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def set_xyz(self, v) -> None:
        self.x, self.y, self.z = float(v[0]), float(v[1]), float(v[2])


@dataclass
class Residue:
    resnum: int
    resname: str
    atoms: "OrderedDict[str, Atom]" = field(default_factory=OrderedDict)
    # free-form provenance note, emitted into the REMARK block
    note: str = ""

    @property
    def one(self) -> str:
        return AA3TO1.get(self.resname, "X")

    def ca(self) -> Optional[np.ndarray]:
        a = self.atoms.get("CA")
        return a.xyz if a else None

    def copy(self) -> "Residue":
        r = Residue(self.resnum, self.resname, OrderedDict(), self.note)
        for k, a in self.atoms.items():
            r.atoms[k] = Atom(a.name, a.resname, a.resnum, a.x, a.y, a.z,
                              a.occ, a.b, a.element, a.icode)
        return r


class Model:
    """An ordered residue map for a single chain, plus the header facts."""

    def __init__(self, source: str = "", chain: str = "A"):
        self.source = source
        self.chain = chain
        self.res: "OrderedDict[int, Residue]" = OrderedDict()
        # header-derived facts
        self.missing: Dict[str, List[Tuple[str, int]]] = {}   # REMARK 465 by chain
        self.truncated: Dict[str, Dict[int, List[str]]] = {}  # REMARK 470 by chain
        self.seqadv: List[dict] = []
        self.ssbond: List[Tuple[str, int, str, int]] = []
        self.dbref: List[dict] = []
        self.dum_z: Optional[Tuple[float, float]] = None      # OPM membrane planes

    # -- accessors ---------------------------------------------------------
    def nums(self) -> List[int]:
        return sorted(self.res)

    def span(self) -> Tuple[int, int]:
        n = self.nums()
        return (n[0], n[-1]) if n else (0, 0)

    def gaps(self) -> List[Tuple[int, int]]:
        """Internal numbering gaps as (last_before, first_after)."""
        n = self.nums()
        return [(a, b) for a, b in zip(n, n[1:]) if b != a + 1]

    def seq(self) -> str:
        return "".join(self.res[i].one for i in self.nums())

    def subset(self, lo: int, hi: int) -> "Model":
        m = Model(self.source, self.chain)
        for i in self.nums():
            if lo <= i <= hi:
                m.res[i] = self.res[i].copy()
        m.dum_z = self.dum_z
        return m

    def coords(self, nums: Sequence[int], atom: str = "CA") -> np.ndarray:
        out = []
        for i in nums:
            a = self.res[i].atoms.get(atom)
            if a is None:
                raise KeyError(f"{self.source}:{self.chain} residue {i} has no {atom}")
            out.append(a.xyz)
        return np.asarray(out, dtype=float)

    def all_atoms(self) -> List[Atom]:
        return [a for i in self.nums() for a in self.res[i].atoms.values()]

    def transform(self, R: np.ndarray, t: np.ndarray) -> None:
        for a in self.all_atoms():
            a.set_xyz(R @ a.xyz + t)

    def set_occ(self, occ: float, nums: Optional[Sequence[int]] = None) -> None:
        for i in (self.nums() if nums is None else nums):
            for a in self.res[i].atoms.values():
                a.occ = occ

    def merge(self, other: "Model", nums: Optional[Sequence[int]] = None,
              overwrite: bool = True) -> None:
        for i in (other.nums() if nums is None else nums):
            if i not in other.res:
                continue
            if i in self.res and not overwrite:
                continue
            self.res[i] = other.res[i].copy()
        self.res = OrderedDict(sorted(self.res.items()))

    def drop(self, nums: Sequence[int]) -> None:
        for i in nums:
            self.res.pop(i, None)


# ── parsing ────────────────────────────────────────────────────────────────
_R465_RE = re.compile(r"^REMARK 465\s+(?:\d+\s+)?([A-Z]{2,3})\s+([A-Za-z0-9])\s+(-?\d+)\s*$")
_R470_HDR = re.compile(r"^REMARK 470\s+(?:M\s+)?RES\s+CSSEQI")
_R470_RE = re.compile(r"^REMARK 470\s+(?:\d+\s+)?([A-Z]{2,3})\s+([A-Za-z0-9])\s*(-?\d+)\s+(.*)$")


def read_pdb(path, chain: str, keep_hetatm: Sequence[str] = ("MSE", "YCM"),
             altloc_keep: Sequence[str] = (" ", "A")) -> Model:
    """Read one chain of a PDB file into a Model, with its header facts.

    Hydrogens are dropped: every prepared template is emitted heavy-atom only,
    and the pipeline's own protonation step (PROPKA, MD_SOLVENT_PH) owns
    protonation downstream.
    """
    path = Path(path)
    m = Model(source=path.name, chain=chain)
    in470 = False
    dum_z: List[float] = []

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            rec = line[:6]

            if rec == "REMARK":
                if line.startswith("REMARK 465"):
                    mo = _R465_RE.match(line.rstrip())
                    if mo:
                        m.missing.setdefault(mo.group(2), []).append(
                            (mo.group(1), int(mo.group(3))))
                elif line.startswith("REMARK 470"):
                    if _R470_HDR.match(line):
                        in470 = True
                        continue
                    if in470:
                        mo = _R470_RE.match(line.rstrip())
                        if mo:
                            m.truncated.setdefault(mo.group(2), {})[int(mo.group(3))] = \
                                mo.group(4).split()
                continue

            if rec == "SEQADV":
                m.seqadv.append({
                    "resname": line[12:15].strip(),
                    "chain": line[16:17],
                    "resnum": _int_or_none(line[18:22]),
                    "db_resname": line[39:42].strip(),
                    "db_resnum": _int_or_none(line[43:48]),
                    "comment": line[49:].strip(),
                })
                continue

            if rec == "SSBOND":
                m.ssbond.append((line[15:16], _int_or_none(line[17:21]),
                                 line[29:30], _int_or_none(line[31:35])))
                continue

            if rec.startswith("DBREF"):
                if rec == "DBREF ":
                    m.dbref.append({
                        "chain": line[12:13],
                        "seq_begin": _int_or_none(line[14:18]),
                        "seq_end": _int_or_none(line[20:24]),
                        "db": line[26:32].strip(),
                        "accession": line[33:41].strip(),
                        "db_begin": _int_or_none(line[55:60]),
                        "db_end": _int_or_none(line[62:67]),
                    })
                continue

            if rec not in ("ATOM  ", "HETATM"):
                continue

            resname = line[17:20].strip()

            # OPM files carry the bilayer planes as DUM pseudo-atoms.
            if resname == "DUM":
                dum_z.append(float(line[46:54]))
                continue

            if line[21:22] != chain:
                continue
            if rec == "HETATM" and resname not in keep_hetatm:
                continue

            altloc = line[16:17]
            if altloc not in altloc_keep:
                continue

            element = line[76:78].strip() or _element_from_name(line[12:16])
            if element == "H" or element == "D":
                continue

            resnum = int(line[22:26])
            name = line[12:16].strip()
            atom = Atom(
                name=name, resname=resname, resnum=resnum,
                x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                occ=float(line[54:60] or 1.0), b=float(line[60:66] or 0.0),
                element=element, icode=line[26:27],
            )
            r = m.res.get(resnum)
            if r is None:
                r = m.res[resnum] = Residue(resnum, resname)
            # First conformer wins; duplicate names (alt A already taken) ignored.
            r.atoms.setdefault(name, atom)

    if dum_z:
        m.dum_z = (min(dum_z), max(dum_z))
    m.res = OrderedDict(sorted(m.res.items()))
    return m


def _int_or_none(s: str):
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return None


def _element_from_name(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    if raw[:2].strip() and raw[0].isalpha() and raw[1].isalpha() and s[:2].upper() in (
            "SE", "FE", "ZN", "MG", "NA", "CL", "NI", "CA"):
        return s[:2].upper()
    return s[0] if s[0].isalpha() else s[1]


# ── writing ────────────────────────────────────────────────────────────────
def write_pdb(model: Model, path, remarks: Sequence[str] = (),
              chain: str = "A", ter: bool = True,
              split_at_gaps: bool = False) -> Path:
    """Write the model.

    `split_at_gaps` puts each contiguously numbered fragment in its own chain and
    ends it with TER.  Only used for the minimisation round trip: a force field
    refuses to build a residue that sits at an internal chain break with no
    terminal group, so the fragments have to be declared as separate chains.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nums = model.nums()
    frag = {}
    cid = chain
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    k = 0
    for j, i in enumerate(nums):
        if split_at_gaps and j and i != nums[j - 1] + 1:
            k += 1
            cid = alphabet[k % len(alphabet)]
        frag[i] = cid

    serial = 1
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        for line in remarks:
            line = _ascii(line)
            # REMARK 999 is the wwPDB slot for depositor free text; used here so
            # the block survives any downstream parser that filters by number.
            for chunk in _wrap(line, 68):
                fh.write(f"REMARK 999 {chunk}\n")
        prev = None
        for i in nums:
            r = model.res[i]
            c = frag[i] if split_at_gaps else chain
            if ter and prev is not None and c != frag.get(prev, c):
                pr = model.res[prev]
                fh.write(f"TER   {serial:5d}      {pr.resname:>3s} "
                         f"{frag[prev]}{pr.resnum:4d}\n")
                serial += 1
            for a in r.atoms.values():
                fh.write(_atom_line(serial, a, r, c))
                serial += 1
            prev = i
        if ter and nums:
            last = model.res[nums[-1]]
            fh.write(f"TER   {serial:5d}      {last.resname:>3s} "
                     f"{frag[nums[-1]] if split_at_gaps else chain}{last.resnum:4d}\n")
        fh.write("END\n")
    return path


_ASCII_MAP = {"—": "--", "–": "-", "‘": "'", "’": "'",
              "“": '"', "”": '"', "…": "...", "Å": "A",
              "→": "->", "±": "+/-", "°": "deg", "≤": "<=",
              "≥": ">=", "×": "x"}


def _ascii(text: str) -> str:
    """PDB is an ASCII format; a stray em-dash in a REMARK breaks naive readers."""
    for k, v in _ASCII_MAP.items():
        text = text.replace(k, v)
    return text.encode("ascii", "replace").decode("ascii")


def _wrap(text: str, width: int) -> List[str]:
    if not text:
        return [""]
    out, line = [], ""
    for word in text.split(" "):
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    return out


def _atom_line(serial: int, a: Atom, r: Residue, chain: str) -> str:
    name = a.name
    if len(name) < 4 and len(a.element) < 2:
        name = f" {name:<3s}"
    else:
        name = f"{name:<4s}"
    return (f"ATOM  {serial:5d} {name}{' '}{r.resname:>3s} {chain}"
            f"{r.resnum:4d}{a.icode}   "
            f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}"
            f"{a.occ:6.2f}{a.b:6.2f}          {a.element:>2s}\n")


# ── geometry ───────────────────────────────────────────────────────────────
def kabsch(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (R, t, rmsd) mapping mobile onto target: x' = R @ x + t."""
    mobile = np.asarray(mobile, float)
    target = np.asarray(target, float)
    if mobile.shape != target.shape or len(mobile) < 3:
        raise ValueError(f"kabsch needs >=3 matched points, got {mobile.shape} vs {target.shape}")
    cm, ct = mobile.mean(0), target.mean(0)
    P, Q = mobile - cm, target - ct
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = (V @ D @ Wt).T
    t = ct - R @ cm
    rmsd = float(np.sqrt((((R @ mobile.T).T + t - target) ** 2).sum(1).mean()))
    return R, t, rmsd


def principal_axis(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    c = pts - pts.mean(0)
    _, _, Wt = np.linalg.svd(c)
    v = Wt[0]
    # orient N->C
    return v if np.dot(pts[-1] - pts[0], v) > 0 else -v


def frame_from_tm(model: Model, tm_spans: Sequence[Tuple[int, int]],
                  signs: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Membrane frame from the TM bundle: returns (R, t, half_thickness).

    Applying x' = R @ x + t puts the bilayer normal on +z with the hydrophobic
    mid-plane at z = 0 and the extracellular side at +z.  Validated against
    5TCX (see prepare_cd_templates.py --selfcheck), where it reproduces the
    OPM-frame occluded-residue set.
    """
    axes, centres = [], []
    for (lo, hi), s in zip(tm_spans, signs):
        nums = [i for i in model.nums() if lo <= i <= hi and "CA" in model.res[i].atoms]
        if len(nums) < 5:
            raise ValueError(f"TM span {lo}-{hi} has only {len(nums)} CA")
        pts = model.coords(nums)
        axes.append(s * principal_axis(pts))
        centres.append(pts.mean(0))
    n = np.mean(axes, axis=0)
    n /= np.linalg.norm(n)
    origin = np.mean(centres, axis=0)

    # Build an orthonormal frame with n as z.
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    ex = np.cross(tmp, n); ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    R = np.vstack([ex, ey, n])
    t = -R @ origin

    # Half-thickness from the TM CA z-extent (median of the four helices' half-spans).
    halves = []
    for (lo, hi) in tm_spans:
        nums = [i for i in model.nums() if lo <= i <= hi and "CA" in model.res[i].atoms]
        z = (R @ model.coords(nums).T).T[:, 2] + t[2]
        halves.append((z.max() - z.min()) / 2.0)
    return R, t, float(np.median(halves))


def z_heights(model: Model, nums: Sequence[int]) -> Dict[int, float]:
    out = {}
    for i in nums:
        r = model.res.get(i)
        if r and "CA" in r.atoms:
            out[i] = float(r.atoms["CA"].z)
    return out
