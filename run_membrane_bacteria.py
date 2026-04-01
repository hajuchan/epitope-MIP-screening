#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bacterial Membrane Affinity Profiling
======================================
Compute monomer affinity profiles for bacterial membrane compositions.

Each bacterium's membrane is modeled as a weighted mixture of known
surface components (outer membrane proteins + lipids). Monomers are
docked to each component via AutoDock4/GPU, then weighted by
composition to produce a per-species affinity score.

Output: which monomers prefer which bacteria, enabling rational
selection of MIP compositions for species-selective detection.

Usage:
    python run_membrane_bacteria.py                    # all 4 species
    python run_membrane_bacteria.py --species E.coli   # single species
    python run_membrane_bacteria.py --quick            # fewer GA runs
    python run_membrane_bacteria.py --resume           # skip completed dockings

Reference:
    - LPS-imprinted MIP: Anal. Chem. 2024 (OMV recognition)
    - Protein A imprinting: Sens. Actuators B 2016 (S. aureus)
    - OMP-LPS fingerprints: J. Chem. Theory Comput. 2019
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Pipeline imports
sys.path.insert(0, str(Path(__file__).parent / "code"))

from pipeline.config import FUNCTIONAL_MONOMERS, N_WORKERS, AUTODOCK4_GA_RUNS
from pipeline.utils_structure import (
    download_pdb, prepare_receptor_pdbqt, smiles_to_pdbqt,
    compute_grid_center, compute_grid_size,
)
from pipeline.utils_autodock import dock_single

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "results" / "membrane_bacteria"

# ======================================================================
# Bacterial Membrane Compositions
# ======================================================================
# Weights represent approximate relative abundance on the outer surface.
# Protein PDB structures are extracellular domains / surface-exposed regions.
#
# Lipid components:
#   - LPS: Lipid IVA (PubChem CID 10329124, 1406 Da) as rigid receptor.
#     Grid is restricted to headgroup (phosphate + sugar) region only,
#     since acyl chains are buried in the membrane and inaccessible to MIP.
#     Lipid IVA is the standard form used in TLR4/MD-2 docking studies
#     (PDB 3FXI) and retains both headgroup and acyl chain structure.
#     Reference: Altintas et al., Anal. Chim. Acta 2016 (endotoxin nanoMIP).
#   - Other lipids: headgroup-only SMILES (PE, PG, LTA) since these are
#     small enough that the headgroup IS the docking-relevant portion.

BACTERIA = {
    "E.coli": {
        "gram": "negative",
        "description": "E. coli K-12 outer membrane",
        "components": {
            "OmpF": {
                "pdb": "4GCP", "chain": "A", "weight": 0.15,
                "desc": "OmpF porin (2.4A, antibiotic complex, 2013)",
            },
            "OmpA": {
                "pdb": "1BXW", "chain": "A", "weight": 0.15,
                "desc": "OmpA transmembrane domain",
            },
            "OmpC": {
                "pdb": "2J1N", "chain": "A", "weight": 0.10,
                "desc": "Outer membrane porin C",
            },
            "LPS_LipidA": {
                # Lipid IVA: PubChem CID 10329124, MW 1406 Da
                # Standard tetra-acylated Lipid A precursor used in
                # TLR4/MD-2 studies (PDB 3FXI). Grid restricted to
                # headgroup region — acyl chains are membrane-buried.
                "smiles": "CCCCCCCCCCCC(CC(=O)NC1C(C(C(OC1OP(=O)(O)O)COC2C(C(C(C(O2)CO)OP(=O)(O)O)OC(=O)CC(CCCCCCCCCCC)O)NC(=O)CC(CCCCCCCCCCC)O)O)OC(=O)CC(CCCCCCCCCCC)O)O",
                "weight": 0.35,
                "desc": "Lipid IVA (CID 10329124) — grid on headgroup only",
                "grid_headgroup_only": True,
            },
            "PE": {
                # Phosphatidylethanolamine headgroup
                "smiles": "NCCOP(O)(=O)OCC(O)CO",
                "weight": 0.15,
                "desc": "Phosphatidylethanolamine headgroup",
            },
            "PG": {
                # Phosphatidylglycerol headgroup
                "smiles": "OCC(O)COP(O)(=O)OCC(O)CO",
                "weight": 0.10,
                "desc": "Phosphatidylglycerol headgroup",
            },
        },
    },

    "P.aeruginosa": {
        "gram": "negative",
        "description": "P. aeruginosa PAO1 outer membrane",
        "components": {
            "OprF": {
                "pdb": "4RLC", "chain": "A", "weight": 0.20,
                "desc": "OprF porin (OmpF homologue)",
            },
            "OprD": {
                "pdb": "3SY7", "chain": "A", "weight": 0.10,
                "desc": "OprD specific porin (carbapenem entry)",
            },
            "LPS_LipidA": {
                "smiles": "CCCCCCCCCCCC(CC(=O)NC1C(C(C(OC1OP(=O)(O)O)COC2C(C(C(C(O2)CO)OP(=O)(O)O)OC(=O)CC(CCCCCCCCCCC)O)NC(=O)CC(CCCCCCCCCCC)O)O)OC(=O)CC(CCCCCCCCCCC)O)O",
                "weight": 0.35,
                "desc": "Lipid IVA (CID 10329124) — grid on headgroup only",
                "grid_headgroup_only": True,
            },
            "PE": {
                "smiles": "NCCOP(O)(=O)OCC(O)CO",
                "weight": 0.20,
                "desc": "Phosphatidylethanolamine headgroup",
            },
            "PG": {
                "smiles": "OCC(O)COP(O)(=O)OCC(O)CO",
                "weight": 0.15,
                "desc": "Phosphatidylglycerol headgroup",
            },
        },
    },

    "K.pneumoniae": {
        "gram": "negative",
        "description": "K. pneumoniae outer membrane",
        "components": {
            "OmpK36": {
                "pdb": "6RD3", "chain": "A", "weight": 0.15,
                "desc": "OmpK36 wild-type porin (1.92A, 2019)",
            },
            "OmpA_Kp": {
                "pdb": "1BXW", "chain": "A", "weight": 0.10,
                "desc": "OmpA (conserved across Enterobacteriaceae)",
            },
            "LPS_LipidA": {
                "smiles": "CCCCCCCCCCCC(CC(=O)NC1C(C(C(OC1OP(=O)(O)O)COC2C(C(C(C(O2)CO)OP(=O)(O)O)OC(=O)CC(CCCCCCCCCCC)O)NC(=O)CC(CCCCCCCCCCC)O)O)OC(=O)CC(CCCCCCCCCCC)O)O",
                "weight": 0.35,
                "desc": "Lipid IVA (CID 10329124) — grid on headgroup only",
                "grid_headgroup_only": True,
            },
            "CPS": {
                # K-antigen capsular polysaccharide (simplified mannose unit)
                "smiles": "OCC1OC(O)C(O)C(O)C1O",
                "weight": 0.20,
                "desc": "Capsular polysaccharide (K-antigen mannose unit)",
            },
            "PE": {
                "smiles": "NCCOP(O)(=O)OCC(O)CO",
                "weight": 0.10,
                "desc": "Phosphatidylethanolamine headgroup",
            },
            "PG": {
                "smiles": "OCC(O)COP(O)(=O)OCC(O)CO",
                "weight": 0.10,
                "desc": "Phosphatidylglycerol headgroup",
            },
        },
    },

    "S.aureus": {
        "gram": "positive",
        "description": "S. aureus cell wall surface",
        "components": {
            "ProteinA": {
                "pdb": "4WWI", "chain": "A", "weight": 0.25,
                "desc": "Protein A C-domain + IgG Fc (X-ray, 2.3A, 2015)",
            },
            "FnBPA": {
                "pdb": "2RKZ", "chain": "A", "weight": 0.15,
                "desc": "Fibronectin-binding protein A (D repeat)",
            },
            "LTA": {
                # Lipoteichoic acid headgroup (glycerol-phosphate repeating unit)
                "smiles": "OCC(O)COP(O)(=O)OCC(O)COP(O)(=O)O",
                "weight": 0.30,
                "desc": "Lipoteichoic acid (polyglycerol-phosphate chain)",
            },
            "PG": {
                "smiles": "OCC(O)COP(O)(=O)OCC(O)CO",
                "weight": 0.20,
                "desc": "Phosphatidylglycerol headgroup",
            },
            "Lys_PG": {
                # Lysyl-PG (positively charged, S.aureus-specific)
                "smiles": "NCCCCC(N)C(=O)OCC(O)COP(O)(=O)OCC(O)CO",
                "weight": 0.10,
                "desc": "Lysyl-phosphatidylglycerol (S.aureus-specific)",
            },
        },
    },
}


# ======================================================================
# Functions
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bacterial membrane affinity profiling for MIP monomers"
    )
    parser.add_argument("--species", type=str, default="all",
                        help="Species to analyze (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer GA runs for fast screening")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip completed dockings (default: on)")
    parser.add_argument("--fresh", action="store_true",
                        help="Re-run all dockings from scratch")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    return parser.parse_args()


def prepare_components(bacteria: dict, output_dir: Path) -> dict:
    """
    Download/generate receptor PDBQT for each membrane component.
    Returns {species: {component: {pdbqt, center, npts}}}
    """
    receptors = {}
    comp_dir = output_dir / "components"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for species, info in bacteria.items():
        receptors[species] = {}
        for comp_name, comp in info["components"].items():
            comp_subdir = comp_dir / species / comp_name
            comp_subdir.mkdir(parents=True, exist_ok=True)

            pdbqt_path = comp_subdir / f"{comp_name}_receptor.pdbqt"
            if pdbqt_path.exists():
                # Resume: already prepared
                center = compute_grid_center(
                    comp_subdir / f"{comp_name}.pdb"
                ) if (comp_subdir / f"{comp_name}.pdb").exists() else (0, 0, 0)
                npts = (60, 60, 60)
            elif "pdb" in comp:
                # Protein component: download PDB
                logger.info(f"  [{species}/{comp_name}] Downloading PDB {comp['pdb']}...")
                pdb_path = download_pdb(comp["pdb"], comp_subdir)
                # Rename for clarity
                named_pdb = comp_subdir / f"{comp_name}.pdb"
                pdb_path.rename(named_pdb)
                prepare_receptor_pdbqt(named_pdb, pdbqt_path)
                center = compute_grid_center(named_pdb)
                npts = compute_grid_size(named_pdb, padding=10.0)
            elif "smiles" in comp:
                # Lipid component: generate 3D from SMILES
                logger.info(f"  [{species}/{comp_name}] Generating from SMILES...")
                pdbqt_path = smiles_to_pdbqt(
                    comp["smiles"], comp_name, comp_subdir
                )
                pdb_path = comp_subdir / f"{comp_name}.pdb"

                if comp.get("grid_headgroup_only") and pdb_path.exists():
                    # Lipid IVA: restrict grid to headgroup (P, N, O-rich)
                    # region only. Acyl chains are membrane-buried.
                    center = _compute_headgroup_center(pdb_path)
                    npts = (40, 40, 40)  # tight grid around headgroup
                    logger.info(f"    Grid restricted to headgroup region")
                elif pdb_path.exists():
                    center = compute_grid_center(pdb_path)
                    npts = (40, 40, 40)
                else:
                    center = (0, 0, 0)
                    npts = (40, 40, 40)
            else:
                logger.warning(f"  [{species}/{comp_name}] No structure source!")
                continue

            receptors[species][comp_name] = {
                "pdbqt": str(pdbqt_path),
                "center": center,
                "npts": npts,
                "weight": comp["weight"],
                "desc": comp.get("desc", ""),
                "type": "protein" if "pdb" in comp else "lipid",
            }

    return receptors


def prepare_monomers(output_dir: Path) -> dict:
    """Generate PDBQT for all 24 functional monomers."""
    mon_dir = output_dir / "monomers"
    mon_dir.mkdir(parents=True, exist_ok=True)
    pdbqts = {}

    for name, info in FUNCTIONAL_MONOMERS.items():
        pdbqt = mon_dir / f"{name}.pdbqt"
        if pdbqt.exists() and pdbqt.stat().st_size > 0:
            pdbqts[name] = pdbqt
            continue
        try:
            pdbqt_path = smiles_to_pdbqt(info["smiles"], name, mon_dir)
            pdbqts[name] = pdbqt_path
        except Exception as e:
            logger.warning(f"  Monomer {name} failed: {e}")

    logger.info(f"Prepared {len(pdbqts)}/{len(FUNCTIONAL_MONOMERS)} monomers")
    return pdbqts


def dock_all(receptors: dict, monomer_pdbqts: dict,
             output_dir: Path, ga_runs: int = 50,
             resume: bool = True) -> dict:
    """
    Dock all monomers to all components across all species.
    Returns {species: {component: {monomer: BE}}}
    """
    dock_dir = output_dir / "docking"
    dock_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Collect all docking jobs
    jobs = []
    for species, comps in receptors.items():
        results[species] = {}
        for comp_name, comp_info in comps.items():
            results[species][comp_name] = {}
            for mon_name, mon_pdbqt in monomer_pdbqts.items():
                work = dock_dir / species / comp_name / mon_name
                result_file = work / "dock_result.json"

                # Resume check
                if resume and result_file.exists():
                    with open(result_file) as f:
                        saved = json.load(f)
                    results[species][comp_name][mon_name] = saved.get(
                        "mean_cluster_energy", 0.0)
                    continue

                jobs.append({
                    "species": species,
                    "component": comp_name,
                    "monomer": mon_name,
                    "receptor": Path(comp_info["pdbqt"]),
                    "ligand": mon_pdbqt,
                    "center": comp_info["center"],
                    "npts": comp_info["npts"],
                    "work_dir": work,
                })

    if not jobs:
        logger.info("All dockings already completed (resume)")
        return results

    logger.info(f"Running {len(jobs)} dockings ({len(jobs)} remaining)...")
    completed = 0

    with ProcessPoolExecutor(max_workers=min(N_WORKERS, 4)) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                _dock_one, job, ga_runs
            )
            futures[future] = job

        for future in as_completed(futures):
            job = futures[future]
            completed += 1
            try:
                be = future.result()
                results[job["species"]][job["component"]][job["monomer"]] = be
                if completed % 50 == 0 or completed == len(jobs):
                    logger.info(f"  Progress: {completed}/{len(jobs)}")
            except Exception as e:
                logger.warning(f"  {job['species']}/{job['component']}/"
                               f"{job['monomer']} failed: {e}")
                results[job["species"]][job["component"]][job["monomer"]] = 0.0

    return results


def _compute_headgroup_center(pdb_path: Path) -> tuple:
    """
    Compute grid center on headgroup atoms only (P, N, O-heavy region).
    For Lipid IVA: the phosphate + sugar headgroup, not acyl chains.
    Selects atoms with element P, N, or O (polar headgroup atoms).
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("lipid", str(pdb_path))
    headgroup_coords = []
    for atom in structure.get_atoms():
        element = atom.element.strip().upper()
        if element in ("P", "N"):
            # P and N are exclusively in headgroup
            headgroup_coords.append(atom.get_vector().get_array())
        elif element == "O":
            # O in phosphate/sugar headgroup (not ester O in acyl chains)
            headgroup_coords.append(atom.get_vector().get_array())

    if not headgroup_coords:
        return compute_grid_center(pdb_path)

    center = np.mean(headgroup_coords, axis=0)
    return tuple(round(float(c), 3) for c in center)


def _dock_one(job: dict, ga_runs: int) -> float:
    """Single docking job (runs in subprocess)."""
    work = Path(job["work_dir"])
    work.mkdir(parents=True, exist_ok=True)

    result = dock_single(
        job["receptor"], job["ligand"],
        tuple(job["center"]), tuple(job["npts"]),
        work, ga_runs=ga_runs,
    )

    be = result.get("mean_cluster_energy", 0.0)

    # Save for resume
    with open(work / "dock_result.json", "w") as f:
        json.dump({"mean_cluster_energy": be,
                    "binding_energy": result.get("binding_energy", 0.0),
                    "n_clusters": result.get("n_clusters", 0)}, f)

    return be


def compute_weighted_affinity(dock_results: dict,
                               receptors: dict) -> pd.DataFrame:
    """
    Compute weighted membrane affinity per species per monomer.
    weighted_BE[species][monomer] = sum(weight_i * BE_i)
    """
    rows = []
    for species, comps in dock_results.items():
        for monomer in FUNCTIONAL_MONOMERS:
            weighted_be = 0.0
            for comp_name, bes in comps.items():
                be = bes.get(monomer, 0.0)
                weight = receptors[species][comp_name]["weight"]
                weighted_be += weight * be
            rows.append({
                "species": species,
                "monomer": monomer,
                "weighted_BE": round(weighted_be, 3),
            })

    df = pd.DataFrame(rows)
    return df.pivot(index="monomer", columns="species", values="weighted_BE")


def compute_selectivity(affinity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute species selectivity for each monomer.
    selectivity[species][monomer] = weighted_BE[species] - mean(others)
    """
    sel = pd.DataFrame(index=affinity_df.index)
    species_list = affinity_df.columns.tolist()

    for sp in species_list:
        others = [s for s in species_list if s != sp]
        sel[sp] = affinity_df[sp] - affinity_df[others].mean(axis=1)

    return sel.round(3)


def plot_results(affinity_df: pd.DataFrame,
                 selectivity_df: pd.DataFrame,
                 output_dir: Path):
    """Generate heatmaps for affinity and selectivity."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(affinity_df) * 0.35)))

        # Affinity heatmap
        im1 = ax1.imshow(affinity_df.values, cmap="RdYlBu", aspect="auto")
        ax1.set_xticks(range(len(affinity_df.columns)))
        ax1.set_xticklabels(affinity_df.columns, rotation=45, ha="right")
        ax1.set_yticks(range(len(affinity_df.index)))
        ax1.set_yticklabels(affinity_df.index, fontsize=8)
        ax1.set_title("Weighted Membrane Affinity (kcal/mol)")
        plt.colorbar(im1, ax=ax1, shrink=0.6)

        # Annotate
        for i in range(len(affinity_df.index)):
            for j in range(len(affinity_df.columns)):
                val = affinity_df.values[i, j]
                ax1.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=6, color="black" if abs(val) < 2 else "white")

        # Selectivity heatmap
        im2 = ax2.imshow(selectivity_df.values, cmap="PiYG", aspect="auto",
                         vmin=-1.5, vmax=1.5)
        ax2.set_xticks(range(len(selectivity_df.columns)))
        ax2.set_xticklabels(selectivity_df.columns, rotation=45, ha="right")
        ax2.set_yticks(range(len(selectivity_df.index)))
        ax2.set_yticklabels(selectivity_df.index, fontsize=8)
        ax2.set_title("Species Selectivity (DDG, kcal/mol)")
        plt.colorbar(im2, ax=ax2, shrink=0.6)

        for i in range(len(selectivity_df.index)):
            for j in range(len(selectivity_df.columns)):
                val = selectivity_df.values[i, j]
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=6)

        plt.tight_layout()
        plt.savefig(output_dir / "membrane_heatmap.png", dpi=150)
        plt.close()
        logger.info(f"Heatmap saved: {output_dir / 'membrane_heatmap.png'}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")


def save_results(affinity_df: pd.DataFrame,
                 selectivity_df: pd.DataFrame,
                 dock_results: dict,
                 receptors: dict,
                 output_dir: Path):
    """Save all results as CSV and JSON."""
    # CSV tables
    affinity_df.to_csv(output_dir / "membrane_affinity.csv")
    selectivity_df.to_csv(output_dir / "membrane_selectivity.csv")

    # Component-level detail CSV
    rows = []
    for species, comps in dock_results.items():
        for comp_name, bes in comps.items():
            for monomer, be in bes.items():
                rows.append({
                    "species": species,
                    "component": comp_name,
                    "type": receptors[species].get(comp_name, {}).get("type", ""),
                    "weight": receptors[species].get(comp_name, {}).get("weight", 0),
                    "monomer": monomer,
                    "BE": be,
                    "weighted_BE": be * receptors[species].get(comp_name, {}).get("weight", 0),
                })
    pd.DataFrame(rows).to_csv(output_dir / "membrane_component_detail.csv", index=False)

    # JSON summary
    summary = {
        "affinity": affinity_df.to_dict(),
        "selectivity": selectivity_df.to_dict(),
        "bacteria": {sp: info["description"] for sp, info in BACTERIA.items()},
        "n_monomers": len(FUNCTIONAL_MONOMERS),
        "n_components": sum(len(r) for r in receptors.values()),
    }
    with open(output_dir / "membrane_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Results saved to {output_dir}")


def print_summary(affinity_df: pd.DataFrame, selectivity_df: pd.DataFrame):
    """Print key findings."""
    logger.info(f"\n{'='*60}")
    logger.info("MEMBRANE AFFINITY PROFILING RESULTS")
    logger.info(f"{'='*60}")

    for species in affinity_df.columns:
        top3 = affinity_df[species].nsmallest(3)
        logger.info(f"\n[{species}] Top 3 monomers (strongest affinity):")
        for mon, be in top3.items():
            sel = selectivity_df.loc[mon, species]
            logger.info(f"  {mon:<10} BE={be:+.2f}  selectivity={sel:+.2f}")

    # Most selective monomers per species
    logger.info(f"\n--- Most Species-Selective Monomers ---")
    for species in selectivity_df.columns:
        best = selectivity_df[species].idxmin()
        val = selectivity_df.loc[best, species]
        logger.info(f"  {species}: {best} (DDG={val:+.2f})")

    logger.info(f"\n{'='*60}")


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Membrane] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(output_dir / "membrane.log")),
        ],
    )

    logger.info("Bacterial Membrane Affinity Profiling")
    logger.info(f"Output: {output_dir}")

    # Filter species
    if args.species == "all":
        bacteria = BACTERIA
    else:
        bacteria = {k: v for k, v in BACTERIA.items()
                    if k == args.species}
        if not bacteria:
            logger.error(f"Unknown species: {args.species}. "
                         f"Available: {list(BACTERIA.keys())}")
            return

    species_names = list(bacteria.keys())
    logger.info(f"Species: {species_names}")

    ga_runs = 10 if args.quick else AUTODOCK4_GA_RUNS
    resume = not args.fresh

    t0 = time.time()

    # 1. Prepare component receptors
    logger.info("\n--- Preparing membrane components ---")
    receptors = prepare_components(bacteria, output_dir)
    n_comp = sum(len(r) for r in receptors.values())
    logger.info(f"Total components: {n_comp}")

    # 2. Prepare monomers
    logger.info("\n--- Preparing monomers ---")
    monomer_pdbqts = prepare_monomers(output_dir)

    # 3. Dock all
    logger.info(f"\n--- Docking ({n_comp} components x {len(monomer_pdbqts)} "
                f"monomers = {n_comp * len(monomer_pdbqts)} pairs, "
                f"{ga_runs} GA runs) ---")
    dock_results = dock_all(receptors, monomer_pdbqts, output_dir,
                            ga_runs=ga_runs, resume=resume)

    # 4. Compute weighted affinity
    logger.info("\n--- Computing weighted membrane affinity ---")
    affinity_df = compute_weighted_affinity(dock_results, receptors)

    # 5. Compute selectivity
    selectivity_df = compute_selectivity(affinity_df)

    # 6. Output
    save_results(affinity_df, selectivity_df, dock_results, receptors, output_dir)
    print_summary(affinity_df, selectivity_df)

    elapsed = time.time() - t0
    logger.info(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
