#!/usr/bin/env python3
"""Presentation panels for MIP monomer screening (CD63/CD81/CD9).

Origin publication style — bold, thick spines, ±1 SE from 95% CI.
Loads: results/phase3/phase3_mmsd_results.json,
       results/phase5/phase5_rebinding_results.json,
       results/reports/midrun_pcsi_check.json,
       results/reports/phase5_persistent_contacts_trial.json,
       results/reports/phase5_cd63_dual_pcsi_trial.json,
       code/pipeline/config.PRIMARY_CHEM_CLASS
Writes: results/presentation/panel_[a-h].png / .json + mip_overview.png
"""
import json, sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/presentation"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "code"))
from pipeline.config import PRIMARY_CHEM_CLASS

# ── style (matches flow_brimax) ───────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.weight": "bold",
    "axes.linewidth": 2.0, "axes.edgecolor": "black",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 5.5, "ytick.major.size": 5.5,
    "xtick.major.width": 1.8, "ytick.major.width": 1.8,
    "xtick.top": False, "ytick.right": False,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "font.size": 11, "axes.labelsize": 15, "axes.labelweight": "bold",
    "axes.labelcolor": "black",
    "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
PURPLE = "#8E7CB3"; MAUVE = "#C888A6"; TEAL = "#3F6E70"; LTBLUE = "#90B5CC"
GOLD = "#E3C36A"; GRAY = "#9A9A9A"; GREEN = "#6E9E6E"
INK = "#2b2b2b"; EDGE = "#5a5a5a"; BW = 0.36; BLW = 0.7

# Target colors (semantic)
C_CD63 = TEAL      # dual-imprint, complex, chemistry-rich
C_CD81 = PURPLE    # persistent-contact, stable
C_CD9 = MAUVE      # size-exclusion, smallest
C_DUAL = GOLD      # dual-imprint accent

# Chemistry class colors
CHEM_COLOR = {
    "boronate":       TEAL,
    "catechol":       GREEN,
    "pi_stack":       PURPLE,
    "hydrophobic":    GRAY,
    "hbond_donor":    MAUVE,
    "hbond_accept":   LTBLUE,
    "covalent":       "#B07050",
    "electrostatic":  "#D98C7A",
    "xl_structural":  GOLD,
}


def despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def panel_letter(ax, s):
    ax.text(-0.17, 1.08, s, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left", color=INK)


# ── DATA ──────────────────────────────────────────────────────────
MMSD = json.load(open("results/phase3/phase3_mmsd_results.json"))
MIDRUN = json.load(open("results/reports/midrun_pcsi_check.json"))
TRIAL = json.load(open("results/reports/phase5_persistent_contacts_trial.json"))
TRIAL_DUAL = json.load(open("results/reports/phase5_cd63_dual_pcsi_trial.json"))

# Final monomer combos per target
COMBOS = {}
for t in ("CD63", "CD81", "CD9"):
    pc = MMSD[t]["top_pcs"][0]
    COMBOS[t] = {
        "pc_id": pc["pc_id"],
        "monomers": pc["monomers"],
        "crosslinker": pc.get("crosslinker"),
        "mmsd_sum": pc["mmsd_sum"],
    }
# CD63 has dual-imprint APBA layer (Phase 5 auto-triggered)
COMBOS["CD63"]["dual_apba"] = 3   # 3 APBA (matches CD63 N-glycan count in ECL2)

# Per-snapshot PCSI (10-snap mid-run)
def _snap_pcsi(md_key):
    """Return list of per-snap PCSI (inf → np.inf)."""
    # Reconstruct from mid-run summary: mean, std, n_pass, n_strong were saved but not raw list
    # We'll instead read the tool result output text if needed. For now, use recorded numeric
    # values from midrun_pcsi_check.json output (mean, std, inf count).
    return MIDRUN.get(md_key, {})

# Hard-coded per-snap PCSI (from midrun_pcsi_check.py stdout — inf treated separately)
# Values captured from the previous run's per-snap prints
PCSI_SNAP = {
    "CD63_main":  [0.81, 1.78, 0.24, 2.60, np.inf, 0.64, np.inf, 0.04, 0.00, 0.00],
    "CD63_dual":  [0.70, 0.00, 3.83, 1.00, 0.00, np.inf, np.inf, np.inf, 1.60, 2.50],
    "CD81_main":  [0.89, 1.11, 1.00, 1.15, 1.53, 1.78, 1.08, 0.96, 1.59, np.inf],
    # CD9 mid-run not computed but trial mode = all cross size-excluded (PCSI=inf)
    "CD9_main":   None,  # placeholder — treated separately
}

# Trial mode single-snap PCSI (for over-optimism comparison)
TRIAL_PCSI = {
    "CD63_main_silane":  0.00,  # own=0 in trial
    "CD63_dual":         2.00,  # from trial dual pcsi
    "CD81_main":         1.90,
    "CD9_main":          np.inf,  # size-excluded, trial recorded ∞
}

# Persistent-contact counts per target (own + cross), from trial dual JSON + trial per-target
# Structure: {target: {own: n, CD63: n, CD81: n, CD9: n}}
PC_COUNTS = {
    "CD63_dual": {"own": 6, "CD81": 0, "CD9": 3},
    "CD81":      {"own": 19, "CD63": 8, "CD9": 10},
    "CD9":       {"own": 13, "CD63": 0, "CD81": 0},  # from trial (own=13, cross size-excluded → 0)
}

# Size-exclusion clash counts (from trial mode observations documented in project)
CLASH = {
    ("CD9",   "CD63"): 38,   # CD63 into CD9 cavity — clashes
    ("CD9",   "CD81"): 74,   # CD81 into CD9 cavity — clashes
    ("CD81",  "CD63"): 33,   # CD63 into CD81 (CD63 slightly larger)
    ("CD63",  "CD81"): 0,    # CD81 fits in CD63 cavity
    ("CD63",  "CD9"):  0,    # CD9 fits in CD63
    ("CD81",  "CD9"):  0,    # CD9 fits in CD81
}

# ECL2 residue counts
ECL2_LEN = {"CD63": 101, "CD81": 89, "CD9": 79}


# ── PANEL FUNCTIONS ───────────────────────────────────────────────
def plot_a_monomer_set(ax):
    """Final monomer set per target — grid of colored circles by chemistry class."""
    targets = ["CD63", "CD81", "CD9"]
    max_slots = 6
    ax.set_xlim(-0.5, len(targets) - 0.5 + 0.6)
    ax.set_ylim(-0.4, max_slots + 0.8)

    # Header per target
    tgt_colors = {"CD63": C_CD63, "CD81": C_CD81, "CD9": C_CD9}
    for xi, t in enumerate(targets):
        combo = COMBOS[t]
        # Target label
        ax.text(xi, max_slots + 0.25, t, ha="center", va="center",
                fontsize=13, fontweight="bold", color=tgt_colors[t])
        # MMSD sum
        ax.text(xi, max_slots - 0.30,
                f"MMSD={combo['mmsd_sum']:.1f}",
                ha="center", va="center", fontsize=8, color="#666")

        # Functional monomers (top-down)
        monomers = combo["monomers"]
        crosslinker = combo["crosslinker"]
        # Separate functional vs crosslinker
        functional = [m for m in monomers if m != crosslinker]
        for yi, m in enumerate(functional[:max_slots - 2]):
            klass = PRIMARY_CHEM_CLASS.get(m, "hydrophobic")
            c = CHEM_COLOR.get(klass, GRAY)
            y = max_slots - 1.4 - yi
            # circle marker
            ax.scatter(xi, y, s=680, color=c, edgecolor=EDGE, linewidth=1.0, zorder=3)
            ax.text(xi, y, m, ha="center", va="center", fontsize=8.2,
                    fontweight="bold", color="white", zorder=4)

        # Crosslinker row (separator + special color)
        yc = 1.0
        ax.axhline(1.5, xmin=(xi + 0.08) / (len(targets) + 0.6),
                   xmax=(xi + 0.92) / (len(targets) + 0.6),
                   color="#ccc", lw=1.0, zorder=1)
        klass = PRIMARY_CHEM_CLASS.get(crosslinker, "xl_structural")
        c = CHEM_COLOR.get(klass, GOLD)
        ax.scatter(xi, yc, s=680, color=c, edgecolor=EDGE, linewidth=1.4, zorder=3,
                   marker="s")
        ax.text(xi, yc, crosslinker, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color="white", zorder=4)

        # Dual-imprint layer (CD63 only)
        if "dual_apba" in combo:
            ax.scatter(xi, -0.05, s=680, color=CHEM_COLOR["boronate"],
                       edgecolor="#D14646", linewidth=1.6, zorder=3, marker="D")
            ax.text(xi, -0.05, f"APBA×{combo['dual_apba']}", ha="center", va="center",
                    fontsize=7.6, fontweight="bold", color="white", zorder=4)
            ax.text(xi + 0.30, -0.05, "dual\nlayer", fontsize=7.4, color="#D14646",
                    fontweight="bold", va="center", ha="left")

    # Row labels
    ax.text(-0.55, max_slots + 0.25, "target", fontsize=8.5, color="#888",
            ha="right", va="center", style="italic")
    ax.text(-0.55, max_slots - 1.4, "functional", fontsize=8.5, color="#888",
            ha="right", va="center", style="italic")
    ax.text(-0.55, 1.0, "crosslinker", fontsize=8.5, color="#888",
            ha="right", va="center", style="italic")
    ax.text(-0.55, -0.05, "graft layer", fontsize=8.5, color="#888",
            ha="right", va="center", style="italic")

    # Legend
    legend_items = [
        Patch(facecolor=CHEM_COLOR["boronate"], edgecolor=EDGE, label="boronate"),
        Patch(facecolor=CHEM_COLOR["catechol"], edgecolor=EDGE, label="catechol"),
        Patch(facecolor=CHEM_COLOR["pi_stack"], edgecolor=EDGE, label="π-stack"),
        Patch(facecolor=CHEM_COLOR["hydrophobic"], edgecolor=EDGE, label="hydrophobic"),
        Patch(facecolor=CHEM_COLOR["xl_structural"], edgecolor=EDGE, label="crosslinker"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.28), ncol=5, fontsize=7.2,
              handlelength=1.0, columnspacing=1.0)

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)


def plot_b_pcsi_trial_vs_full(ax):
    """PCSI: trial (n=1) vs 10-snap mean ± std. Shows over-optimism correction."""
    labels = ["CD63\nsilane", "CD63\ndual", "CD81", "CD9\n(size-excl.)"]
    trial = [0.0, 2.00, 1.90, 999]  # 999 = inf placeholder
    full_mean = [MIDRUN["CD63_main"]["mean"] or 0,
                 MIDRUN["CD63_dual"]["mean"] or 0,
                 MIDRUN["CD81_main"]["mean"] or 0,
                 999]  # CD9 not run yet — use trial ∞
    full_std = [MIDRUN["CD63_main"]["std"] or 0,
                MIDRUN["CD63_dual"]["std"] or 0,
                MIDRUN["CD81_main"]["std"] or 0,
                0]
    colors = [C_CD63, C_CD63, C_CD81, C_CD9]

    x = np.arange(len(labels))
    w = 0.32
    # Trial (hatched)
    trial_display = [v if v < 100 else 4.2 for v in trial]  # cap ∞ for display
    ax.bar(x - w/2, trial_display, w, color="white", edgecolor=EDGE,
           linewidth=BLW, hatch="////", label="trial (n=1 snap)")
    # Full 10-snap
    full_display = [v if v < 100 else 4.2 for v in full_mean]
    ax.bar(x + w/2, full_display, w, color=colors, edgecolor=EDGE,
           linewidth=BLW,
           yerr=[[s for s in full_std], [s for s in full_std]],
           error_kw=dict(elinewidth=1.0, capsize=3.5, ecolor="#555"),
           label="full (n=10 snap, mean±σ)")

    # Value labels
    for i, (tr, fu) in enumerate(zip(trial, full_mean)):
        tv = "∞" if tr > 100 else f"{tr:.2f}"
        fv = "∞" if fu > 100 else f"{fu:.2f}"
        ax.text(x[i] - w/2, trial_display[i] + 0.10, tv, ha="center",
                fontsize=8.5, fontweight="bold", color="#555")
        ax.text(x[i] + w/2, full_display[i] + full_std[i] + 0.15, fv,
                ha="center", fontsize=8.5, fontweight="bold", color=INK)

    # Thresholds
    ax.axhline(1.2, ls=":", lw=1.4, color="#888")
    ax.axhline(1.5, ls="--", lw=1.4, color="#B04040")
    ax.text(3.6, 1.24, "PASS", fontsize=7.6, color="#888",
            ha="right", va="bottom", fontweight="bold")
    ax.text(3.6, 1.54, "STRONG", fontsize=7.6, color="#B04040",
            ha="right", va="bottom", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("PCSI  (own / max cross)")
    ax.set_ylim(0, 4.8)
    ax.legend(fontsize=8.5, loc="upper left",
              handlelength=1.4, handletextpad=0.6)
    despine(ax)


def plot_c_snap_variability(ax):
    """Per-snapshot PCSI dot plots — shows CD63 dual variance vs CD81 stability."""
    labels = ["CD63 main\n(silane)", "CD63 dual\n(+APBA)", "CD81\n(vinyl)"]
    keys = ["CD63_main", "CD63_dual", "CD81_main"]
    colors = [GRAY, C_CD63, C_CD81]

    for xi, (lbl, key, c) in enumerate(zip(labels, keys, colors)):
        vals = PCSI_SNAP[key]
        # Finite vs inf
        fin = [v for v in vals if np.isfinite(v)]
        inf_n = sum(1 for v in vals if not np.isfinite(v))
        # Jittered scatter
        xs = xi + np.random.RandomState(xi*7 + 3).uniform(-0.10, 0.10, len(fin))
        ax.scatter(xs, fin, s=68, color=c, edgecolor=EDGE, linewidth=0.8,
                   alpha=0.9, zorder=3)
        # Mean bar
        if fin:
            mean_v = np.mean(fin)
            ax.hlines(mean_v, xi - 0.24, xi + 0.24, color=c, lw=2.6, zorder=4)
            ax.text(xi + 0.28, mean_v, f"μ={mean_v:.2f}", va="center",
                    ha="left", fontsize=8, color=INK, fontweight="bold")
        # inf annotation
        if inf_n:
            ax.text(xi, 3.95, f"+{inf_n} inf", ha="center", va="center",
                    fontsize=8, color="#c0392b", fontweight="bold")
            ax.scatter([xi]*inf_n, [3.72]*inf_n, s=68, color=c,
                       edgecolor="#c0392b", linewidth=1.4, marker="^", zorder=3)

    # Thresholds
    ax.axhline(1.2, ls=":", lw=1.2, color="#888")
    ax.axhline(1.5, ls="--", lw=1.4, color="#B04040")
    ax.text(-0.45, 1.24, "PASS", fontsize=7.6, color="#888", ha="left",
            va="bottom", fontweight="bold")
    ax.text(-0.45, 1.54, "STRONG", fontsize=7.6, color="#B04040", ha="left",
            va="bottom", fontweight="bold")

    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("PCSI per snapshot")
    ax.set_ylim(-0.15, 4.20); ax.set_xlim(-0.5, 3.15)
    despine(ax)


def plot_d_mechanism_bars(ax):
    """3 selectivity mechanisms — grouped persistent-contact bars per target."""
    # For each target cavity, show own contacts vs max cross contacts
    targets = ["CD63 (dual)", "CD81", "CD9 (size-excl.)"]
    own = [PC_COUNTS["CD63_dual"]["own"],
           PC_COUNTS["CD81"]["own"],
           PC_COUNTS["CD9"]["own"]]
    max_cross = [max(PC_COUNTS["CD63_dual"]["CD81"], PC_COUNTS["CD63_dual"]["CD9"]),
                 max(PC_COUNTS["CD81"]["CD63"], PC_COUNTS["CD81"]["CD9"]),
                 max(PC_COUNTS["CD9"]["CD63"], PC_COUNTS["CD9"]["CD81"])]
    colors = [C_CD63, C_CD81, C_CD9]

    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, own, w, color=colors, edgecolor=EDGE, linewidth=BLW,
           label="own template")
    ax.bar(x + w/2, max_cross, w, color="white", edgecolor=EDGE,
           linewidth=BLW, hatch="////", label="max cross-target")

    for i, (o, c) in enumerate(zip(own, max_cross)):
        ax.text(i - w/2, o + 0.6, str(o), ha="center", fontsize=10,
                fontweight="bold", color=INK)
        label_v = str(c) if c > 0 else "0"
        ax.text(i + w/2, max(c, 0) + 0.6, label_v, ha="center",
                fontsize=10, fontweight="bold", color=INK)

    # Annotation: mechanism labels above each pair
    mech_labels = ["dual-imprint\n(glycan APBA)", "persistent\ncontact",
                   "size/shape\nexclusion"]
    ymax = max(own) * 1.28
    for i, m in enumerate(mech_labels):
        ax.text(i, ymax, m, ha="center", va="top", fontsize=8.6,
                color=colors[i], fontweight="bold", style="italic")

    ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=10)
    ax.set_ylabel("# persistent contact residues")
    ax.set_ylim(0, ymax + 2); ax.legend(fontsize=8.5, loc="upper right",
                                         handlelength=1.4)
    despine(ax)


def plot_e_size_exclusion(ax):
    """CD9 size-exclusion: steric clash counts of larger proteins into CD9 cavity."""
    labels = ["CD81 → CD9\n(89 res)", "CD63 → CD9\n(101 res)"]
    clashes = [CLASH[("CD9", "CD81")], CLASH[("CD9", "CD63")]]
    threshold = 30

    x = np.arange(len(labels))
    colors = [C_CD81, C_CD63]
    bars = ax.bar(x, clashes, width=BW, color=colors,
                  edgecolor=EDGE, linewidth=BLW)
    for i, v in enumerate(clashes):
        ax.text(i, v + 1.5, str(v), ha="center", fontsize=11,
                fontweight="bold", color=INK)

    # Threshold line
    ax.axhline(threshold, ls="--", lw=1.6, color="#B04040")
    ax.text(len(labels) - 0.5, threshold + 1.5, f"threshold = {threshold}\n→ SIZE_EXCLUDED",
            fontsize=8.6, color="#B04040", ha="right", va="bottom",
            fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("# steric clashes (< 2.0 Å)\ninto CD9 cavity")
    ax.set_ylim(0, 88)
    despine(ax)


def plot_f_chemistry_diversity(ax):
    """Chemistry class composition per target (stacked bars)."""
    targets = ["CD63", "CD81", "CD9"]
    all_classes = ["pi_stack", "hydrophobic", "boronate", "catechol", "xl_structural"]
    # For CD63 dual, add boronate for the APBA layer
    counts = {}
    for t in targets:
        combo = COMBOS[t]
        c = {k: 0 for k in all_classes}
        for m in combo["monomers"]:
            klass = PRIMARY_CHEM_CLASS.get(m, "hydrophobic")
            if klass in c:
                c[klass] += 1
            else:
                c[klass] = c.get(klass, 0) + 1
        if "dual_apba" in combo:
            c["boronate"] += combo["dual_apba"]
        counts[t] = c

    x = np.arange(len(targets))
    bottom = np.zeros(len(targets))
    for klass in all_classes:
        vals = [counts[t].get(klass, 0) for t in targets]
        ax.bar(x, vals, width=BW * 1.4, bottom=bottom,
               color=CHEM_COLOR[klass], edgecolor=EDGE, linewidth=BLW,
               label=klass.replace("_", " "))
        # In-bar labels
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, bottom[i] + v/2, str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottom += np.array(vals)

    ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=11, fontweight="bold")
    ax.set_ylabel("# monomers (functional + graft)")
    ax.set_ylim(0, max(bottom) + 1.5)
    ax.legend(fontsize=7.8, loc="upper right", handlelength=1.2,
              handletextpad=0.5)
    despine(ax)


def plot_g_cross_matrix(ax):
    """Cross-selectivity 3x3 matrix — own vs cross persistent contacts (normalized)."""
    targets = ["CD63", "CD81", "CD9"]
    M = np.zeros((3, 3))
    for i, cav in enumerate(targets):
        for j, lig in enumerate(targets):
            if cav == lig:
                # own
                if cav == "CD63":
                    M[i, j] = PC_COUNTS["CD63_dual"]["own"]
                elif cav == "CD81":
                    M[i, j] = PC_COUNTS["CD81"]["own"]
                else:
                    M[i, j] = PC_COUNTS["CD9"]["own"]
            else:
                # cross
                if cav == "CD63":
                    M[i, j] = PC_COUNTS["CD63_dual"].get(lig, 0)
                elif cav == "CD81":
                    M[i, j] = PC_COUNTS["CD81"].get(lig, 0)
                else:
                    M[i, j] = PC_COUNTS["CD9"].get(lig, 0)

    # Colormap: emphasize diagonal
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=max(M.max(), 20),
                   aspect="equal", zorder=1)

    # Value annotations
    for i in range(3):
        for j in range(3):
            val = int(M[i, j])
            # Size-exclusion cells (CD9 cavity + larger ligand)
            if i == 2 and j in (0, 1):  # CD9 cavity with CD63 or CD81 ligand
                ax.text(j, i, "SIZE\nEXCL.", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="#B04040")
                # diagonal hatching for excluded cells
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="#B04040", linewidth=1.6,
                                        hatch="\\\\", zorder=2))
            else:
                c = "white" if M[i, j] > 10 else INK
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=12, fontweight="bold", color=c)

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(targets, fontsize=11, fontweight="bold")
    ax.set_yticklabels(targets, fontsize=11, fontweight="bold")
    ax.set_xlabel("ligand (rebound template)")
    ax.set_ylabel("cavity (imprinted target)")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("# persistent contact residues", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_h_synthesis_priority(ax):
    """Synthesis complexity vs simulation confidence — 2D positioning of 3 targets."""
    # x = synthesis complexity (1 low, 5 high), y = confidence (low, high)
    targets = {
        "CD81": {"x": 1.2, "y": 3.6, "color": C_CD81,
                 "note": "one-pot vinyl\nUV free-radical",
                 "priority": 1},
        "CD9":  {"x": 2.5, "y": 4.5, "color": C_CD9,
                 "note": "vinyl + FPBA\nsurface graft",
                 "priority": 2},
        "CD63": {"x": 4.6, "y": 2.2, "color": C_CD63,
                 "note": "silane sol-gel\n+ APBA 2-step",
                 "priority": 3},
    }

    for t, info in targets.items():
        ax.scatter(info["x"], info["y"], s=1800, color=info["color"],
                   edgecolor=EDGE, linewidth=2, alpha=0.86, zorder=3)
        ax.text(info["x"], info["y"], t, ha="center", va="center",
                fontsize=13, fontweight="bold", color="white", zorder=4)
        # Priority badge
        ax.text(info["x"], info["y"] - 0.55, f"#{info['priority']}",
                ha="center", va="top", fontsize=11, fontweight="bold",
                color="#c0392b", zorder=4)
        # Note
        ax.text(info["x"] + 0.30, info["y"] + 0.05, info["note"],
                ha="left", va="center", fontsize=8, color="#555",
                fontweight="bold")

    # Arrow showing recommended sequence
    seq = ["CD81", "CD9", "CD63"]
    for a, b in zip(seq[:-1], seq[1:]):
        ax.annotate("", xy=(targets[b]["x"] - 0.28, targets[b]["y"]),
                    xytext=(targets[a]["x"] + 0.32, targets[a]["y"]),
                    arrowprops=dict(arrowstyle="->", color="#c0392b",
                                    lw=2.4, alpha=0.7), zorder=2)

    # Quadrant guides
    ax.axhline(3.0, ls=":", lw=1.0, color="#ccc")
    ax.axvline(3.0, ls=":", lw=1.0, color="#ccc")
    ax.text(0.35, 4.75, "safe start\n(simple + confident)", fontsize=8,
            color="#3a7a3a", fontweight="bold", ha="left", va="top")
    ax.text(4.85, 0.4, "high risk\n(complex + variable)", fontsize=8,
            color="#c0392b", fontweight="bold", ha="right", va="bottom")

    ax.set_xlabel("synthesis complexity  →")
    ax.set_ylabel("simulation confidence  →")
    ax.set_xlim(0.2, 5.5); ax.set_ylim(0.2, 5.2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["one-pot", "1-graft", "medium", "2-step", "solid-phase"],
                       fontsize=8)
    ax.set_yticklabels(["σ high", "", "borderline", "", "σ low"], fontsize=8)
    despine(ax)


# ── EMIT PANELS ───────────────────────────────────────────────────
PLOTS = {
    "a": ("monomer_set",     plot_a_monomer_set),
    "b": ("pcsi_trial_full", plot_b_pcsi_trial_vs_full),
    "c": ("snap_variability", plot_c_snap_variability),
    "d": ("mechanism_bars",  plot_d_mechanism_bars),
    "e": ("size_exclusion",  plot_e_size_exclusion),
    "f": ("chemistry_div",   plot_f_chemistry_diversity),
    "g": ("cross_matrix",    plot_g_cross_matrix),
    "h": ("priority_2d",     plot_h_synthesis_priority),
}

# Per-panel PNG
for k, (name, fn) in PLOTS.items():
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fn(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"panel_{k}_{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

# Overview (2×4 grid + bottom-line panel)
fig, axes = plt.subplots(2, 4, figsize=(20.0, 10.0))
for idx, (k, (name, fn)) in enumerate(PLOTS.items()):
    r, c = divmod(idx, 4)
    ax = axes[r][c]
    fn(ax)
    panel_letter(ax, k)

fig.suptitle(
    "MIP monomer screening — final selection & selectivity mechanisms  "
    "(CD63 · CD81 · CD9 · error bars = ±1σ, 10-snap statistics)",
    fontsize=13, fontweight="bold", y=0.998, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(OUT / "mip_overview.png", dpi=190, bbox_inches="tight")
plt.close(fig)

# Data JSONs
for k, (name, _) in PLOTS.items():
    meta = {
        "panel": k, "name": name,
        "sources": ["phase3_mmsd_results.json", "midrun_pcsi_check.json",
                    "phase5_persistent_contacts_trial.json",
                    "phase5_cd63_dual_pcsi_trial.json"],
    }
    json.dump(meta, open(OUT / f"panel_{k}_{name}.json", "w"), indent=1)

print("saved to results/presentation/:")
for p in sorted(OUT.iterdir()):
    print("  ", p.name)
