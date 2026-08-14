#!/usr/bin/env python3
"""Presentation *process* panels — how the pipeline picked these monomers.

Complements make_presentation_figures.py (which shows the "why").
Slide 3 uses these to show the "how": 27-monomer library → SMD filter → Pareto →
chemistry-diversity rule → selectivity-penalty tie-breaker → 1 winner per target.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/presentation"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "code"))
from pipeline.config import PRIMARY_CHEM_CLASS

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
C_CD63 = TEAL; C_CD81 = PURPLE; C_CD9 = MAUVE
CHEM_COLOR = {
    "boronate":       TEAL,   "catechol":       GREEN,
    "pi_stack":       PURPLE, "hydrophobic":    GRAY,
    "hbond_donor":    MAUVE,  "hbond_accept":   LTBLUE,
    "covalent":       "#B07050", "electrostatic":  "#D98C7A",
    "xl_structural":  GOLD,
}


def despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def panel_letter(ax, s):
    ax.text(-0.14, 1.08, s, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left", color=INK)


P2 = json.load(open("results/phase2/phase2_smd_results.json"))
P3 = json.load(open("results/phase3/phase3_mmsd_results.json"))
TARGETS = ["CD63", "CD81", "CD9"]
TCOLOR = {"CD63": C_CD63, "CD81": C_CD81, "CD9": C_CD9}
WINNERS = {t: P3[t]["top_pcs"][0]["monomers"] for t in TARGETS}
WINNER_XL = {t: P3[t]["top_pcs"][0]["crosslinker"] for t in TARGETS}


# ─────────────────────────────────────────────────────────────
# P1 · Funnel — 27 → 12 → Pareto ~20 → chemistry-diversity → 1
# ─────────────────────────────────────────────────────────────
def plot_p1_funnel(ax):
    """Cascading funnel showing how 27 monomers → 1 winner per target."""
    stages = [
        ("Library", 27, "27 monomers\n(silane · vinyl · catechol · boronate · crosslinker)"),
        ("Phase 2 · SMD", 12, "top 12 by docking BE\n(AutoDock-GPU, 6 conformers, decoy EF > 1.5)"),
        ("Phase 3 · NSGA-II", 20, "~20 non-dominated combos\n(3-objective Pareto front)"),
        ("Chemistry diversity", 8, "Rule 1 (≥2 classes) + Rule 2 (≤2 per class)\ndrops single-class-dominant combos"),
        ("Cross-MMSD ΔΔG", 1, "selectivity penalty\ntie-breaks own vs cross-target"),
    ]
    n = len(stages)
    ymax = 5.0

    # Horizontal funnel: each stage is a stacked trapezoid
    x_positions = np.arange(n) * 2.4
    max_w = 2.2

    for i, (name, cnt, note) in enumerate(stages):
        # Width proportional to count
        w = max_w * (cnt / stages[0][1]) ** 0.5
        x = x_positions[i]
        y_top = 3.5
        y_bot = 1.5
        # Trapezoid via polygon
        top_hw = w * 1.0
        bot_hw = w * 0.85
        poly = np.array([
            [x - top_hw, y_top], [x + top_hw, y_top],
            [x + bot_hw, y_bot], [x - bot_hw, y_bot]
        ])
        col_shade = ["#e8e8e8", LTBLUE, PURPLE, MAUVE, TEAL][i]
        ax.fill(poly[:, 0], poly[:, 1], color=col_shade, edgecolor=EDGE,
                linewidth=BLW, alpha=0.90, zorder=2)

        # Count label
        ax.text(x, 2.65, f"{cnt}",
                ha="center", va="center", fontsize=20,
                fontweight="bold", color="white", zorder=3)
        ax.text(x, 2.15, "× 3 target",
                ha="center", va="center", fontsize=7.5, color="white",
                fontweight="bold", zorder=3)

        # Stage name (top)
        ax.text(x, y_top + 0.28, name, ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=col_shade if i > 0 else "#666")
        # Note (bottom) — alternating y to avoid overlap
        note_y = y_bot - 0.25 if i % 2 == 0 else y_bot - 0.85
        ax.text(x, note_y, note, ha="center", va="top",
                fontsize=7.2, color="#555", linespacing=1.35)

        # Arrow to next stage
        if i < n - 1:
            arr = FancyArrowPatch((x + top_hw + 0.05, 2.5),
                                   (x_positions[i + 1] - max_w * 0.7, 2.5),
                                   arrowstyle="-|>", color="#888", lw=2,
                                   mutation_scale=16, zorder=1)
            ax.add_patch(arr)

    # Final panel: 3 target winners emerge as icons
    yf = 0.15
    ax.text(x_positions[-1] + 1.6, 2.5, "→", fontsize=22, ha="center",
            va="center", color="#888", fontweight="bold")

    winner_x = x_positions[-1] + 2.8
    for k, t in enumerate(TARGETS):
        y = 3.8 - k * 1.3
        ax.scatter(winner_x, y, s=550, color=TCOLOR[t],
                   edgecolor=EDGE, linewidth=1.4, zorder=3)
        ax.text(winner_x, y, t, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white", zorder=4)
        combo_str = " · ".join(WINNERS[t][:3]) + "..."
        ax.text(winner_x + 0.30, y, combo_str, ha="left", va="center",
                fontsize=7.5, color="#333", fontweight="bold")

    ax.text(winner_x, 5.05, "final MIP\nrecipes", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=INK)

    ax.set_xlim(-1.2, winner_x + 3.4)
    ax.set_ylim(-1.2, 5.2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ─────────────────────────────────────────────────────────────
# P2 · Phase 2 BE heatmap — 27 monomers × 3 targets
# ─────────────────────────────────────────────────────────────
def plot_p2_be_heatmap(ax):
    """Docking BE matrix — 27 monomers × 3 targets. Filtered top-12 boxed."""
    # Assemble all monomers (union across targets)
    all_monomers = sorted({m for t in TARGETS for m in P2["be_matrix"][t]},
                          key=lambda m: -np.mean([P2["be_matrix"][t].get(m, 0) for t in TARGETS]))
    n = len(all_monomers)
    M = np.array([[P2["be_matrix"][t].get(m, 0) for t in TARGETS]
                   for m in all_monomers])
    filtered = {t: set(P2["filtered"][t]) for t in TARGETS}

    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto",
                   vmin=-7, vmax=-1, zorder=1)
    # Highlight top-12 per target
    for j, t in enumerate(TARGETS):
        for i, m in enumerate(all_monomers):
            if m in filtered[t]:
                ax.add_patch(Rectangle((j - 0.42, i - 0.42), 0.84, 0.84,
                                        fill=False, edgecolor=TCOLOR[t],
                                        linewidth=1.6, zorder=3))

    # Mark winner monomers with star
    for j, t in enumerate(TARGETS):
        for m in WINNERS[t]:
            if m in all_monomers:
                i = all_monomers.index(m)
                ax.text(j, i, "★", ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold", zorder=4)

    ax.set_yticks(range(n))
    ax.set_yticklabels(all_monomers, fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(TARGETS, fontsize=11, fontweight="bold")
    ax.set_xlabel("target")
    ax.tick_params(top=False, right=False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.09, pad=0.05)
    cbar.set_label("docking BE (kcal/mol)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Legend explaining highlights — placed OUTSIDE / above the plot
    legend_items = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor="k", markersize=10, markeredgewidth=1.6,
               label="top 12 (SMD filter)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="k", markersize=13, label="final (Phase 3)"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), fontsize=7.5,
              handletextpad=0.3, ncol=2, columnspacing=0.8)

    for sp in ax.spines.values():
        sp.set_visible(False)


# ─────────────────────────────────────────────────────────────
# P3 · Pareto scatter — affinity vs selectivity
# ─────────────────────────────────────────────────────────────
def plot_p3_pareto(ax):
    """Affinity (MMSD/monomer) vs selectivity (ΔΔG cross-target)
    Pareto front per target. Winner highlighted."""
    def _obj_xy(obj_dict_or_list):
        """Return (affinity, selectivity_score). Selectivity is 'higher = better'."""
        if isinstance(obj_dict_or_list, dict):
            aff = obj_dict_or_list.get("affinity_mmsd_per",
                obj_dict_or_list.get("affinity_mmsd_per_monomer"))
            sel = obj_dict_or_list.get("selectivity_score")
            return aff, sel
        return None, None

    for t in TARGETS:
        pareto = P3[t].get("pareto_front", [])
        pts = []
        for p in pareto:
            aff, sel = _obj_xy(p.get("objectives"))
            if aff is not None and sel is not None:
                pts.append((aff, sel))
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=54, color=TCOLOR[t], edgecolor=EDGE,
                       linewidth=0.6, alpha=0.55,
                       label=f"{t} Pareto (n={len(pts)})", zorder=2)

        # Winner — use objective_details (present in top_pcs)
        w = P3[t]["top_pcs"][0]
        od = w.get("objective_details", {})
        wx = od.get("affinity_mmsd_per_monomer")
        wy = od.get("selectivity_score")
        if wx is not None and wy is not None:
            ax.scatter(wx, wy, s=320, color=TCOLOR[t],
                       edgecolor="#c0392b", linewidth=2.2, marker="*", zorder=4)
            combo = " · ".join(w["monomers"][:3])
            # Deconflicted annotations per target
            offx = {"CD63": 0.02, "CD81": 0.02, "CD9": 0.02}[t]
            offy = {"CD63": 0.55, "CD81": 0.75, "CD9": 0.75}[t]
            ha = {"CD63": "left", "CD81": "left", "CD9": "left"}[t]
            ax.annotate(f"{t}: {combo}", xy=(wx, wy),
                        xytext=(wx + offx, wy + offy),
                        fontsize=7.4, color=TCOLOR[t], fontweight="bold",
                        ha=ha, zorder=5,
                        arrowprops=dict(arrowstyle="-", color=TCOLOR[t],
                                        lw=0.6, alpha=0.5))

    ax.set_xlabel("← better    affinity (MMSD / monomer)")
    ax.set_ylabel("selectivity score →   (higher = more selective)")
    ax.legend(fontsize=8, loc="lower left", handlelength=1.4,
              markerscale=0.8)
    despine(ax)


# ─────────────────────────────────────────────────────────────
# P4 · Selectivity ΔΔG panel — own vs cross-target BE per winner
# ─────────────────────────────────────────────────────────────
def plot_p4_selectivity_ddg(ax):
    """For each winning combo, show cross-MMSD ΔΔG = own_BE − mean(cross_BE).
    Negative means own_target has stronger binding → selectivity."""
    labels = []
    own_be = []
    cross_be_mean = []
    colors = []
    ddg = []
    for t in TARGETS:
        pc = P3[t]["top_pcs"][0]
        own = pc["mmsd_sum"]
        crosses = list(pc["cross_target_be"].values())
        cross_mean = float(np.mean(crosses)) if crosses else 0
        labels.append(t)
        own_be.append(own)
        cross_be_mean.append(cross_mean)
        colors.append(TCOLOR[t])
        ddg.append(pc.get("DDG_selectivity", own - cross_mean))

    x = np.arange(len(labels))
    w = 0.34
    ax.bar(x - w/2, own_be, w, color=colors, edgecolor=EDGE,
           linewidth=BLW, label="own target BE")
    ax.bar(x + w/2, cross_be_mean, w, color="white", edgecolor=EDGE,
           linewidth=BLW, hatch="////",
           label="mean cross-target BE")

    for i, (o, c, d) in enumerate(zip(own_be, cross_be_mean, ddg)):
        ax.text(i - w/2, o + 0.4, f"{o:.1f}", ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")
        ax.text(i + w/2, c + 0.4, f"{c:.1f}", ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")
        # ΔΔG annotation UNDER bars
        ax.text(i, min(o, c) - 1.6,
                f"ΔΔG={d:+.2f}", ha="center", va="top",
                fontsize=9, fontweight="bold",
                color=("#2b7a2b" if d < -1.0 else "#c0392b"))

    ax.axhline(0, color="#ccc", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("MMSD sum (kcal/mol)")
    ax.set_ylim(min(own_be + cross_be_mean) - 8, 1)
    ax.text(0.02, 0.03, "more negative = stronger binding",
            fontsize=7.6, color="#666", transform=ax.transAxes, style="italic")
    ax.legend(fontsize=8.5, loc="upper right", handlelength=1.4)
    despine(ax)


# ─────────────────────────────────────────────────────────────
# P5 · Chemistry-diversity rule impact
# ─────────────────────────────────────────────────────────────
def plot_p5_chem_diversity(ax):
    """For each target's Pareto front, show class composition of the top-affinity
    candidate BEFORE vs AFTER applying chemistry-diversity rule (top winning combo).
    Illustrates why the rule matters."""
    # Get all Pareto combos, find "highest affinity" one and the winner.
    labels = []
    top_aff_classes = []
    winner_classes = []
    def _aff(p):
        obj = p.get("objectives")
        if isinstance(obj, dict):
            return obj.get("affinity") or obj.get("mmsd_per_monomer", 1e9)
        if isinstance(obj, (list, tuple)) and obj:
            return obj[0]
        return 1e9

    for t in TARGETS:
        pareto = P3[t].get("pareto_front", [])
        # highest-affinity combo (min mmsd_per_monomer)
        top = min(pareto, key=_aff) if pareto else P3[t]["top_pcs"][0]
        w = P3[t]["top_pcs"][0]
        labels.append(t)
        top_aff_classes.append([PRIMARY_CHEM_CLASS.get(m, "hydrophobic")
                                 for m in top["monomers"]])
        winner_classes.append([PRIMARY_CHEM_CLASS.get(m, "hydrophobic")
                                for m in w["monomers"]])

    # For each target: stacked bar of class counts (before rule) and (winner after rule)
    all_classes = ["pi_stack", "hydrophobic", "boronate", "catechol",
                   "xl_structural", "hbond_donor", "hbond_accept"]
    x_pos = np.arange(len(TARGETS))
    w_bar = 0.35

    for i, t in enumerate(TARGETS):
        # left bar: highest-affinity ignore-diversity
        pre = top_aff_classes[i]
        # right bar: winner (with diversity rule)
        post = winner_classes[i]
        b_pre = 0
        b_post = 0
        for cls in all_classes:
            c_pre = pre.count(cls); c_post = post.count(cls)
            if c_pre:
                ax.bar(i - w_bar/2, c_pre, w_bar, bottom=b_pre,
                       color=CHEM_COLOR[cls], edgecolor=EDGE, linewidth=BLW,
                       label=cls.replace("_", " ") if i == 0 and c_pre else None)
                if c_pre >= 1:
                    ax.text(i - w_bar/2, b_pre + c_pre/2, str(c_pre),
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color="white")
                b_pre += c_pre
            if c_post:
                ax.bar(i + w_bar/2, c_post, w_bar, bottom=b_post,
                       color=CHEM_COLOR[cls], edgecolor=EDGE, linewidth=BLW)
                if c_post >= 1:
                    ax.text(i + w_bar/2, b_post + c_post/2, str(c_post),
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color="white")
                b_post += c_post

    # X labels: two labels under each target
    for i, t in enumerate(TARGETS):
        ax.text(i - w_bar/2, -0.30, "top-\naffinity",
                ha="center", va="top", fontsize=7.5, color="#c0392b")
        ax.text(i + w_bar/2, -0.30, "winner\n(diverse)",
                ha="center", va="top", fontsize=7.5, color="#2b7a2b")

    # Uniform-class warning (only if pre has 3+ same class)
    for i, t in enumerate(TARGETS):
        pre = top_aff_classes[i]
        if pre:
            max_cnt = max(pre.count(c) for c in set(pre))
            if max_cnt >= 3:
                ax.text(i - w_bar/2, len(pre) + 0.3,
                        f"{max_cnt}× same\nclass!", ha="center", va="bottom",
                        fontsize=7.4, color="#c0392b", fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(TARGETS, fontsize=11, fontweight="bold")
    ax.set_ylabel("# monomers per class")
    ax.set_ylim(-1.4, 8)
    # separate x tick labels well below "top-affinity/winner" sub-labels
    ax.tick_params(axis="x", pad=32)
    ax.legend(fontsize=7.4, loc="upper right", handlelength=1.1,
              handletextpad=0.4, ncol=1)
    despine(ax)


# ─────────────────────────────────────────────────────────────
# P6 · Phase timeline (swimlane style)
# ─────────────────────────────────────────────────────────────
def plot_p6_timeline(ax):
    """Horizontal swimlane: what each Phase does + which decision applies."""
    phases = [
        ("Phase 1", "Structure prep", "ECL2 extract\n+ 20 ns stability MD", LTBLUE),
        ("Phase 2", "Single monomer\ndocking (SMD)", "27 monomers ×\n3 targets ×\n6 conformer poses", PURPLE),
        ("Phase 3", "Multi-monomer\nNSGA-II", "Pareto front\n(affinity · selectivity · synth)\n+ ΔΔG penalty\n+ chem diversity", MAUVE),
        ("Phase 4", "Pre-polymerization\nMD (350 ns)", "25 monomers on\nECL2 surface\n+ Cα restraint", TEAL),
        ("Phase 5", "VIP rebinding\n(10 snap × 50 ns)", "own vs cross\ntemplate rebind\n→ PCSI · size-excl.\n· dual-imprint trigger", GOLD),
        ("Phase 6", "Recipe generation", "MIP + NIP protocol\nratio · initiator ·\npH · cure", GREEN),
    ]
    n = len(phases)
    x = np.arange(n)
    y = 1.2

    for i, (ph, title, desc, c) in enumerate(phases):
        # Box
        ax.add_patch(Rectangle((i - 0.42, y), 0.84, 1.6,
                                facecolor=c, edgecolor=EDGE, linewidth=BLW,
                                alpha=0.85, zorder=2))
        ax.text(i, y + 1.45, ph, ha="center", va="top", fontsize=10.5,
                fontweight="bold", color="white", zorder=3)
        ax.text(i, y + 1.15, title, ha="center", va="top", fontsize=8.6,
                fontweight="bold", color="white", zorder=3)
        ax.text(i, y + 0.75, desc, ha="center", va="top", fontsize=6.9,
                color="white", zorder=3, linespacing=1.35)

        # Arrow
        if i < n - 1:
            arr = FancyArrowPatch((i + 0.44, y + 0.8),
                                   (i + 0.56, y + 0.8),
                                   arrowstyle="-|>", color="#666",
                                   mutation_scale=14, lw=1.5, zorder=1)
            ax.add_patch(arr)

    # Below the swimlane: decision criteria at each transition
    decisions = [
        (0.5, "pLDDT>70,\nRMSD<3Å"),
        (1.5, "BE<-2.0,\ntop 12"),
        (2.5, "Pareto\n+ chem rule\n+ ΔΔG<-1.0"),
        (3.5, "Q1→Q4\nconvergence"),
        (4.5, "PCSI>1.2\nor size-excl."),
    ]
    for dx, dtext in decisions:
        ax.text(dx, 0.65, dtext, ha="center", va="top", fontsize=7,
                color="#c0392b", fontweight="bold", style="italic",
                linespacing=1.3)

    ax.text(2.5, -0.15, "─── go / no-go decisions at each hand-off ───",
            ha="center", va="center", fontsize=8, color="#c0392b",
            fontweight="bold", style="italic")

    ax.set_xlim(-0.7, n - 0.3); ax.set_ylim(-0.5, 3.2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


PLOTS = {
    "p1": ("funnel",           plot_p1_funnel),
    "p2": ("be_heatmap",       plot_p2_be_heatmap),
    "p3": ("pareto",           plot_p3_pareto),
    "p4": ("selectivity_ddg",  plot_p4_selectivity_ddg),
    "p5": ("chem_diversity_rule", plot_p5_chem_diversity),
    "p6": ("timeline",         plot_p6_timeline),
}

# Per-panel PNG
FIGSIZE = {
    "p1": (11.0, 4.6),   # wide funnel
    "p2": (5.5, 6.0),    # tall heatmap
    "p3": (5.4, 4.4),
    "p4": (5.4, 4.4),
    "p5": (5.6, 4.4),
    "p6": (13.5, 4.2),   # wide timeline
}
for k, (name, fn) in PLOTS.items():
    fig, ax = plt.subplots(figsize=FIGSIZE[k])
    fn(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"panel_{k}_{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

# Combined "selection process" overview
fig = plt.figure(figsize=(20.0, 14.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.35],
                       hspace=0.42, wspace=0.42, top=0.955, bottom=0.05,
                       left=0.045, right=0.99)
ax_p6 = fig.add_subplot(gs[0, :])
plot_p6_timeline(ax_p6); panel_letter(ax_p6, "a")
ax_p1 = fig.add_subplot(gs[1, :])
plot_p1_funnel(ax_p1); panel_letter(ax_p1, "b")
ax_p2 = fig.add_subplot(gs[2, 0])
plot_p2_be_heatmap(ax_p2); panel_letter(ax_p2, "c")
ax_p3 = fig.add_subplot(gs[2, 1])
plot_p3_pareto(ax_p3); panel_letter(ax_p3, "d")
ax_p4 = fig.add_subplot(gs[2, 2])
plot_p4_selectivity_ddg(ax_p4); panel_letter(ax_p4, "e")
ax_p5 = fig.add_subplot(gs[2, 3])
plot_p5_chem_diversity(ax_p5); panel_letter(ax_p5, "f")

fig.suptitle(
    "How the pipeline picked these monomers — selection process across 6 phases",
    fontsize=14, fontweight="bold", y=0.985, color=INK)
fig.savefig(OUT / "mip_process_overview.png", dpi=190, bbox_inches="tight")
plt.close(fig)

for k, (name, _) in PLOTS.items():
    json.dump({"panel": k, "name": name,
               "sources": ["phase2_smd_results.json", "phase3_mmsd_results.json"]},
              open(OUT / f"panel_{k}_{name}.json", "w"), indent=1)

print("saved to results/presentation/:")
for p in sorted(OUT.iterdir()):
    if p.name.startswith(("panel_p", "mip_process")):
        print("  ", p.name)
