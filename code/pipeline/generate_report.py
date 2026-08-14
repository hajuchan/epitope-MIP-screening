"""
HTML Report Generator for Epitope-MIP Screening Pipeline
=========================================================
Consolidates results from all 5 phases into a single HTML report
with tables, figures, and synthesis recommendations.
"""

import base64
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px;
       background: #f8f9fa; color: #333; max-width: 1100px;
       margin: 0 auto; padding: 20px; }
h1 { color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 10px; }
h2 { color: #2c3e50; margin-top: 30px; border-left: 4px solid #3498db;
     padding-left: 12px; }
h3 { color: #34495e; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th { background: #2c3e50; color: white; padding: 10px; text-align: left; }
td { padding: 8px 10px; border-bottom: 1px solid #ddd; }
tr:hover { background: #eaf2f8; }
.highlight { background: #d5f5e3; font-weight: bold; }
.warning { background: #fdebd0; }
.card { background: white; border-radius: 8px; padding: 20px;
        margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
img { max-width: 100%; border-radius: 4px; margin: 10px 0; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 12px;
       font-size: 0.85em; margin: 2px; }
.tag-silane { background: #d6eaf8; color: #1a5276; }
.tag-vinyl { background: #fadbd8; color: #922b21; }
.tag-crosslinker { background: #d5f5e3; color: #1e8449; }
.metric { font-size: 1.8em; font-weight: bold; color: #2980b9; }
.footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ccc;
          font-size: 0.85em; color: #7f8c8d; }
"""


def generate_report(output_dir: str) -> str:
    """Generate HTML report from pipeline results."""
    out = Path(output_dir)

    sections = [
        _build_header(),
        _build_phase1(out),
        _build_phase2(out),
        _build_phase3(out),
        _build_phase4(out),
        _build_phase5_rebinding(out),     # Phase 5: cavity rebinding
        _build_selectivity_matrix(out),
        _build_phase6_recipe(out),        # Phase 6: synthesis recipe
        _build_references(),
        _build_footer(),
    ]

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Epitope-MIP Screening Report</title>
<style>{CSS}</style>
</head><body>
{''.join(sections)}
</body></html>"""

    report_path = out / "reports" / "pipeline_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html)
    logger.info(f"Report → {report_path}")
    return str(report_path)


def _build_header() -> str:
    return f"""
    <h1>Epitope-MIP Screening Report</h1>
    <div class="card">
        <p><strong>Project:</strong> Selective Recognition of Exosome
        Tetraspanin ECL2 (CD63 / CD81 / CD9)</p>
        <p><strong>Method:</strong> AutoDock4 SMD → MMSD → GROMACS MD + MM-PBSA</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>Reference:</strong> Rajpal et al., Sci. Rep. 2024</p>
    </div>
    """


def _build_phase1(out: Path) -> str:
    data = _load_json(out / "phase1" / "phase1_results.json")
    if not data:
        return "<h2>Phase 1: Epitope Preparation</h2><p>No results.</p>"

    rows = ""
    for target, info in data.items():
        if "error" in info:
            continue
        props = info.get("properties", {})
        plddt = info.get("plddt", {})
        rows += f"""<tr>
            <td><strong>{target}</strong></td>
            <td><code>{props.get('sequence', 'N/A')}</code></td>
            <td>{props.get('length', 'N/A')}</td>
            <td>{props.get('gravy', 'N/A')}</td>
            <td>{props.get('isoelectric_point', 'N/A')}</td>
            <td>{props.get('hbond_donors', 0)} / {props.get('hbond_acceptors', 0)}</td>
            <td>{props.get('n_glycan_sites_known', 0)}</td>
            <td>{plddt.get('mean_plddt', 'N/A')}</td>
        </tr>"""

    return f"""
    <h2>Phase 1: Epitope Preparation</h2>
    <div class="card">
    <table>
        <tr><th>Target</th><th>Sequence</th><th>Length</th><th>GRAVY</th>
        <th>pI</th><th>HBD/HBA</th><th>N-Glycan</th><th>pLDDT</th></tr>
        {rows}
    </table>
    </div>
    """


def _build_phase2(out: Path) -> str:
    data = _load_json(out / "phase2" / "phase2_smd_results.json")
    if not data:
        return "<h2>Phase 2: SMD Screening</h2><p>No results.</p>"

    heatmap_html = _embed_image(out / "phase2" / "phase2_heatmap.png")

    filtered = data.get("filtered", {})
    filt_rows = ""
    for target, monomers in filtered.items():
        tags = " ".join(f'<span class="tag tag-silane">{m}</span>'
                        for m in monomers)
        filt_rows += f"<tr><td>{target}</td><td>{tags}</td></tr>"

    return f"""
    <h2>Phase 2: Single Monomer Docking (SMD)</h2>
    <div class="card">
        <h3>Binding Energy Heatmap</h3>
        {heatmap_html}
    </div>
    <div class="card">
        <h3>Filtered Monomers (BE &lt; -2.0, ΔΔG &lt; -0.5 kcal/mol)</h3>
        <table>
            <tr><th>Target</th><th>Candidates</th></tr>
            {filt_rows}
        </table>
    </div>
    """


def _build_phase3(out: Path) -> str:
    data = _load_json(out / "phase3" / "phase3_mmsd_results.json")
    if not data:
        return "<h2>Phase 3: MMSD</h2><p>No results.</p>"

    sections = ""
    for target, info in data.items():
        if not isinstance(info, dict) or "top_pcs" not in info:
            continue

        plot_html = _embed_image(
            out / "phase3" / f"phase3_{target}_bo.png"
        )

        rows = ""
        has_selectivity = any(pc.get("DDG_selectivity") is not None
                              for pc in info["top_pcs"][:8])
        for pc in info["top_pcs"][:8]:
            mono_str = " + ".join(pc["monomers"])
            cls = "highlight" if pc.get("synergy") else ""
            ddg = pc.get("DDG_selectivity")
            # REVIEW FINDING 16: a bare "—" read the same for "selectivity was
            # measured and is unremarkable" and "every cross-target MMSD
            # crashed". Phase 3 now publishes selectivity_status, so say which.
            _sel_status = pc.get("selectivity_status")
            if ddg is not None:
                ddg_str = f"{ddg:.2f}"
                if _sel_status == "measured_partial":
                    ddg_str += " <sup title='measured against fewer off-targets " \
                               "than configured'>partial</sup>"
                ddg_color = "green" if ddg < -1.0 else ("red" if ddg > 0 else "")
            elif _sel_status in ("all_cross_target_mmsd_failed",
                                 "own_mmsd_unusable"):
                ddg_str = "NOT MEASURED"
                ddg_color = "orange"
            else:
                ddg_str = "—"
                ddg_color = ""
            rows += f"""<tr class="{cls}">
                <td>{pc['pc_id']}</td>
                <td>{mono_str}</td>
                <td>{pc.get('mmsd_sum', 0):.2f}</td>
                <td>{pc.get('smd_sum', 0):.2f}</td>
                <td>{pc.get('delta_sum', 0):.2f}</td>
                <td>{'Synergy' if pc.get('synergy') else '-'}</td>
                {'<td style="color:' + ddg_color + ';">' + ddg_str + '</td>' if has_selectivity else ''}
            </tr>"""

        sel_header = "<th>ΔΔG Sel.</th>" if has_selectivity else ""
        sections += f"""
        <div class="card">
            <h3>{target}</h3>
            <p>Fixed: {info.get('fixed_monomers', [])},
               {info.get('n_combinations', info.get('n_evaluations', 0))} combinations screened,
               {info.get('high_affinity_count', 0)} high-affinity</p>
            {plot_html}
            <table>
                <tr><th>PC</th><th>Monomers</th><th>MMSD Sum</th>
                <th>SMD Sum</th><th>Delta</th><th>Synergy</th>{sel_header}</tr>
                {rows}
            </table>
        </div>
        """

    return f"<h2>Phase 3: Multi-Monomer Simultaneous Docking (MMSD)</h2>{sections}"


def _build_phase4(out: Path) -> str:
    data = _load_json(out / "phase4" / "phase4_md_results.json")
    if not data:
        return "<h2>Phase 4: Pre-polymerization MD</h2><p>No results.</p>"

    sections = ""
    for target, pcs in data.items():
        if target == "cross_reactivity" or not isinstance(pcs, dict):
            continue

        for pc_id, md in pcs.items():
            # Summary table
            rmsd = md.get('rmsd_mean_nm', 'N/A')
            rmsd_str = f"{rmsd:.3f}" if isinstance(rmsd, (int, float)) else rmsd
            rmsf = md.get('rmsf_mean_nm', 'N/A')
            rmsf_str = f"{rmsf:.3f}" if isinstance(rmsf, (int, float)) else rmsf
            hbond = md.get('hbond_mean', 'N/A')
            hbond_str = f"{hbond:.1f}" if isinstance(hbond, (int, float)) else hbond
            rg = md.get('rg_mean_nm', 'N/A')
            rg_str = f"{rg:.3f}" if isinstance(rg, (int, float)) else rg
            dg = md.get('mmpbsa', {}).get('delta_total_kcal', 'N/A')
            dg_str = f"{dg:.2f}" if isinstance(dg, (int, float)) else dg
            dssp = md.get('dssp', {})
            dssp_str = "Preserved" if dssp.get('structure_preserved') else \
                       "Not preserved" if dssp.get('structure_preserved') is False else "N/A"

            # Contact frequency
            occ = md.get('occupancy_analysis', {})
            contact_freq = occ.get('contact_freq_6A', occ.get('occupancy', {}))
            min_dist = occ.get('mean_min_dist_A', {})
            # residence_ns is the REAL time. `residence_frames` is a count of
            # ANALYSED frames (i.e. stride units), which the old report
            # displayed under a heading that read as a duration.
            residence = occ.get('residence_ns', occ.get('residence_frames', {}))
            residence_is_ns = 'residence_ns' in occ
            ratio = md.get('optimal_ratio', {})

            ebn_data = occ.get('EBN', {})
            hbnmax_data = occ.get('HBNMax', {})
            # hbond_lifetime_avg is a MEAN COUNT PER FRAME, never a lifetime.
            # hbond_lifetime_ps is the real one (mean consecutive-frame run
            # length per donor-H-acceptor triple).
            hb_life = occ.get('hbond_lifetime_ps', occ.get('hbond_lifetime_avg', {}))
            hb_life_is_ps = 'hbond_lifetime_ps' in occ

            contact_rows = ""
            for m in sorted(contact_freq.keys()):
                cf = contact_freq.get(m, 0)
                md_dist = min_dist.get(m, 'N/A')
                res = residence.get(m, 'N/A')
                r = ratio.get(m, 'N/A')
                e = ebn_data.get(m, '—')
                hbn = hbnmax_data.get(m, '—')
                hbl = hb_life.get(m, '—')
                md_dist_str = f"{md_dist:.1f}" if isinstance(md_dist, (int, float)) else md_dist
                res_str = f"{res:.1f}" if isinstance(res, (int, float)) else res
                hbl_str = f"{hbl:.1f}" if isinstance(hbl, (int, float)) else hbl
                contact_rows += f"""<tr>
                    <td>{m}</td><td>{cf:.3f}</td><td>{md_dist_str} Å</td>
                    <td>{res_str}{' ns' if residence_is_ns else ' frames'}</td>
                    <td>{e}</td><td>{hbn}</td>
                    <td>{hbl_str}{' ps' if hb_life_is_ps else ' /frame'}</td>
                    <td>{r}</td></tr>"""

            # Embed RMSD/RMSF/H-bond/Rg plots
            # Phase 4 writes <target>/<pc>/rep_<i>/md now. Prefer the
            # representative replica the pooled result names.
            md_dir = Path(md.get("md_dir") or (out / "phase4" / target / pc_id / "md"))
            rmsd_img = _embed_image(md_dir / "rmsd.xvg.png") if (md_dir / "rmsd.xvg.png").exists() \
                else _xvg_to_inline_plot(md_dir / "rmsd.xvg", "RMSD", "Time (ps)", "RMSD (nm)")
            rmsf_img = _xvg_to_inline_plot(md_dir / "rmsf.xvg", "RMSF", "Residue", "RMSF (nm)")
            hbond_img = _xvg_to_inline_plot(md_dir / "hbond.xvg", "H-bonds", "Time (ps)", "H-bonds")
            rg_img = _xvg_to_inline_plot(md_dir / "gyrate.xvg", "Radius of Gyration", "Time (ps)", "Rg (nm)")

            # A rejected leg is RED, not orange: the report is where a reader
            # decides whether Phase 4 succeeded, and the old layout showed
            # "Not converged" in orange next to a row marked OK.
            n_rep = md.get("n_replicas", "?")
            n_acc = md.get("n_replicas_accepted", "?")
            _accepted = md.get("accepted")
            acc_badge = "" if _accepted is not False else \
                ' <span style="color:#c00;">— REJECTED</span>'
            acc_banner = ""
            if _accepted is False:
                acc_banner = (
                    f'<p style="background:#fdd;border:2px solid #c00;padding:8px;'
                    f'font-weight:bold;color:#900;">LEG REJECTED by Phase 4 '
                    f'acceptance — {md.get("quality_failures")}. Any ratio or '
                    f'cavity derived from it is NOT VALIDATED.</p>')

            sections += f"""
            <div class="card">
                <h3>{target} / {pc_id}{acc_badge}</h3>
                {acc_banner}
                <table>
                    <tr><th>RMSD (nm)</th><th>RMSF (nm)</th><th>H-bonds</th>
                        <th>Rg (nm)</th><th>ΔG (kcal/mol)</th><th>DSSP</th>
                        <th>Replicas accepted</th><th>Status</th></tr>
                    <tr><td>{rmsd_str}</td><td>{rmsf_str}</td><td>{hbond_str}</td>
                        <td>{rg_str}</td><td>{dg_str}</td><td>{dssp_str}</td>
                        <td>{n_acc}/{n_rep}</td>
                        <td>{'✓' if md.get('success') else '✗'}</td></tr>
                </table>

                <h4>Stability Plots</h4>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                    {rmsd_img}{rmsf_img}{hbond_img}{rg_img}
                </div>

                <h4>Monomer-Epitope Contact Analysis (6Å cutoff)</h4>
                <p><small>Residence is real time (ns) when the analysis reported
                   it; "frames" means the run predates that fix and the number is
                   a COUNT OF ANALYSED FRAMES, i.e. stride units, not a duration.
                   Likewise "/frame" under H-bond lifetime is a mean count per
                   frame — never a lifetime — while "ps" is the real mean
                   consecutive-contact duration.</small></p>
                <table>
                    <tr><th>Monomer</th><th>Contact Freq</th><th>Mean Min Dist</th>
                        <th>Residence</th><th>EBN</th><th>HBNMax</th>
                        <th>H-bond lifetime</th><th>Ratio (EBN)</th></tr>
                    {contact_rows}
                </table>
"""
            # Monomer pair distances (cavity shape)
            md_pairs = occ.get("pair_distances_md_A", {})
            if md_pairs:
                pair_rows = ""
                for pair_key in sorted(md_pairs.keys()):
                    m_val = md_pairs[pair_key]
                    m_str = f"{m_val:.1f}" if isinstance(m_val, (int, float)) else m_val
                    # Color: green if compact (<8Å), orange if moderate, red if dispersed
                    color = "green" if isinstance(m_val, (int,float)) and m_val < 8 else \
                            ("orange" if isinstance(m_val, (int,float)) and m_val < 15 else "red")
                    pair_rows += f'<tr><td>{pair_key}</td><td style="color:{color};">{m_str} Å</td></tr>'

                mean_dist = sum(md_pairs.values()) / len(md_pairs) if md_pairs else 0
                compact_str = "Compact" if mean_dist < 10 else "Dispersed"
                compact_color = "green" if mean_dist < 10 else "red"

                sections += f"""
                <h4>Monomer Arrangement (Cavity Shape)</h4>
                <p>Mean pair distance: <span style="color:{compact_color};font-weight:bold;">
                    {mean_dist:.1f} Å — {compact_str}</span>
                    (compact = cavity forming, compare with cross-reactivity below)</p>
                <table>
                    <tr><th>Monomer Pair</th><th>Min Distance</th></tr>
                    {pair_rows}
                </table>
"""
            # Convergence check
            conv = occ.get("convergence", {})
            if conv:
                conv_ok = conv.get("converged", False)
                conv_color = "green" if conv_ok else "orange"
                conv_label = "Converged" if conv_ok else "Not converged"
                diff_pct = conv.get("window_diff_pct", {})
                diff_str = ", ".join(f"{m}={v}%" for m, v in diff_pct.items())
                sections += f"""
                <h4>MD Convergence</h4>
                <p>Status: <span style="color:{conv_color};font-weight:bold;">{conv_label}</span>
                   (window diff: {diff_str}; threshold &lt;10%)</p>
"""

            # Crosslinker proximity
            xl_prox = occ.get("crosslinker_proximity_A", {})
            xl_ok = occ.get("crosslinker_well_positioned", False)
            if xl_prox:
                xl_color = "green" if xl_ok else "red"
                xl_label = "Well-positioned" if xl_ok else "Too far"
                xl_rows = ""
                for m, dist in xl_prox.items():
                    d_color = "green" if dist < 10 else "red"
                    xl_rows += f'<tr><td>{m}</td><td style="color:{d_color};">{dist:.1f} Å</td></tr>'
                sections += f"""
                <h4>Crosslinker Proximity</h4>
                <p>Status: <span style="color:{xl_color};font-weight:bold;">{xl_label}</span>
                   (all &lt;10Å = network can form)</p>
                <table><tr><th>Functional Monomer</th><th>Distance to Crosslinker</th></tr>
                    {xl_rows}</table>
"""

            sections += "</div>"

    # Cross-reactivity & Selectivity comparison
    xr = data.get("cross_reactivity", {})
    if xr:
        # Collect self-binding contact frequencies per target
        self_contacts = {}
        for target, pcs in data.items():
            if target == "cross_reactivity" or not isinstance(pcs, dict):
                continue
            for pc_id, md in pcs.items():
                occ = md.get('occupancy_analysis', {})
                cf = occ.get('contact_freq_6A', occ.get('occupancy', {}))
                if cf:
                    self_contacts[target] = cf
                break  # top PC only

        # Build MIP-level selectivity table
        selectivity_html = ""
        for key, xr_data in xr.items():
            if not isinstance(xr_data, dict):
                continue
            parts = key.split("_PC_on_")
            if len(parts) != 2:
                continue
            source, test_target = parts

            xr_occ = xr_data.get("occupancy_analysis", {}).get("contact_freq_6A",
                     xr_data.get("occupancy_analysis", {}).get("occupancy", {}))
            self_occ = self_contacts.get(source, {})

            if not xr_occ or not self_occ:
                continue

            # MIP-level: total contact of entire monomer combination
            total_self = sum(self_occ.values())
            total_cross = sum(xr_occ.values())
            mip_sel = total_self / total_cross if total_cross > 0.01 else float('inf')
            mip_sel_str = f"{mip_sel:.1f}x" if mip_sel < 100 else ">100x"
            sel_color = "green" if mip_sel > 2 else ("orange" if mip_sel > 1 else "red")

            selectivity_html += f"""<tr>
                <td>{source}-MIP</td>
                <td>{source}</td><td>{total_self:.2f}</td>
                <td>{test_target}</td><td>{total_cross:.2f}</td>
                <td style="color:{sel_color};font-weight:bold;font-size:1.1em;">{mip_sel_str}</td>
            </tr>"""

        if selectivity_html:
            # Also collect pair distances for cavity compactness comparison
            self_pair_dists = {}
            for target_t, pcs in data.items():
                if target_t == "cross_reactivity" or not isinstance(pcs, dict):
                    continue
                for pc_id, md in pcs.items():
                    pd = md.get('occupancy_analysis', {}).get('pair_distances_md_A', {})
                    if pd:
                        self_pair_dists[target_t] = pd
                    break

            cavity_html = ""
            for key, xr_data in xr.items():
                if not isinstance(xr_data, dict):
                    continue
                parts = key.split("_PC_on_")
                if len(parts) != 2:
                    continue
                source, test_target = parts

                xr_pairs = xr_data.get("occupancy_analysis", {}).get("pair_distances_md_A", {})
                self_pairs = self_pair_dists.get(source, {})
                if not xr_pairs or not self_pairs:
                    continue

                self_mean = sum(self_pairs.values()) / len(self_pairs)
                xr_mean = sum(xr_pairs.values()) / len(xr_pairs)
                ratio = xr_mean / self_mean if self_mean > 0.1 else 0
                # >1.5 = monomers more dispersed on other target = selective
                cav_color = "green" if ratio > 1.5 else ("orange" if ratio > 1 else "red")

                cavity_html += f"""<tr>
                    <td>{source}-MIP</td>
                    <td>{source}</td><td>{self_mean:.1f} Å</td>
                    <td>{test_target}</td><td>{xr_mean:.1f} Å</td>
                    <td style="color:{cav_color};font-weight:bold;">{ratio:.1f}x</td>
                </tr>"""

            sections += f"""
            <div class="card">
                <h3>MIP Selectivity (Cross-Reactivity)</h3>

                <h4>Contact-Based Selectivity</h4>
                <p>Total contact on own target / other target. <strong>>2x = selective</strong>.</p>
                <table>
                    <tr><th>MIP</th>
                        <th>Own Target</th><th>Total Contact</th>
                        <th>Other Target</th><th>Total Contact</th>
                        <th>Selectivity</th></tr>
                    {selectivity_html}
                </table>"""

            if cavity_html:
                sections += f"""
                <h4>Cavity Compactness (Pair Distance)</h4>
                <p>Mean monomer-monomer distance on own vs other target.
                   Compact on own + dispersed on other = <strong>selective cavity</strong>.
                   Ratio >1.5x = selective.</p>
                <table>
                    <tr><th>MIP</th>
                        <th>Own Target</th><th>Mean Pair Dist</th>
                        <th>Other Target</th><th>Mean Pair Dist</th>
                        <th>Dispersal Ratio</th></tr>
                    {cavity_html}
                </table>"""

            sections += "</div>"

    return f"<h2>Phase 4: Pre-polymerization MD + Selectivity</h2>{sections}"


def _xvg_to_inline_plot(xvg_path: Path, title: str, xlabel: str, ylabel: str) -> str:
    """Convert XVG file to inline base64 plot."""
    xvg_path = Path(xvg_path)
    if not xvg_path.exists():
        return f"<p><em>{title}: data not available</em></p>"
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import io

        data = []
        for line in xvg_path.read_text().split("\n"):
            if line.startswith(("#", "@")) or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    data.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass
        if not data:
            return f"<p><em>{title}: no data</em></p>"

        arr = np.array(data)
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(arr[:, 0], arr[:, 1], linewidth=0.8, color='#2196F3')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        img_data = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;" />'
    except Exception as e:
        return f"<p><em>{title}: plot error ({e})</em></p>"


def _build_phase6_recipe(out: Path) -> str:
    data = _load_json(out / "phase6" / "phase6_recipes.json")
    if not data:
        return "<h2>Phase 5: Synthesis Recipes</h2><p>No results.</p>"

    cards = ""
    for target, recipe in data.items():
        mono_tags = ""
        for m_name, m_info in recipe.get("monomers", {}).items():
            role = m_info.get("role", "functional")
            tag_class = "tag-crosslinker" if role == "cross-linker" \
                else "tag-silane"
            mono_tags += (f'<span class="tag {tag_class}">{m_name} '
                          f'({m_info.get("molar_ratio", 1)}x)</span> ')

        notes_html = ""
        for note in recipe.get("notes", []):
            notes_html += f"<p class='warning' style='padding:8px;'>{note}</p>"

        cards += f"""
        <div class="card">
            <h3>{target} — {recipe.get('polymerization_type', 'N/A')}</h3>
            <p><strong>PC:</strong> {recipe.get('pc_id', 'N/A')}</p>
            <p><strong>Monomers:</strong> {mono_tags}</p>
            {notes_html}
        </div>
        """

    return f"<h2>Phase 6: Recommended Synthesis Recipes</h2>{cards}"


def _build_selectivity_matrix(out: Path) -> str:
    """Build ΔG selectivity matrix from Phase 4 cross-reactivity results."""
    data = _load_json(out / "phase4" / "phase4_md_results.json")
    if not data:
        return ""

    xr = data.get("cross_reactivity", {})
    if not xr:
        return ""

    # Collect self ΔG per target
    self_dg = {}
    for target, pcs in data.items():
        if target == "cross_reactivity" or not isinstance(pcs, dict):
            continue
        for pc_id, md in pcs.items():
            dg = md.get("mmpbsa", {}).get("delta_total_kcal")
            if dg is not None:
                self_dg[target] = dg
            break

    # Collect cross ΔG
    cross_dg = {}
    for key, xr_data in xr.items():
        if not isinstance(xr_data, dict):
            continue
        parts = key.split("_PC_on_")
        if len(parts) != 2:
            continue
        source, test = parts
        dg = xr_data.get("mmpbsa", {}).get("delta_total_kcal")
        cross_dg[(source, test)] = dg

    targets = sorted(self_dg.keys())
    if not targets:
        return ""

    # Build matrix
    header = "<th>MIP \\ Head</th>" + "".join(f"<th>{t}</th>" for t in targets)
    rows = ""
    for source in targets:
        cells = ""
        for test in targets:
            if source == test:
                dg = self_dg.get(source)
                style = "font-weight:bold; background:#e8f5e9;"
            else:
                dg = cross_dg.get((source, test))
                style = ""

            if dg is not None:
                color = "green" if source == test else \
                    ("red" if dg < self_dg.get(source, 0) else "gray")
                cells += f'<td style="{style}color:{color};">{dg:.2f}</td>'
            else:
                cells += "<td>N/A</td>"
        rows += f"<tr><td><strong>{source}-MIP</strong></td>{cells}</tr>"

    return f"""
    <h2>Selectivity Matrix (ΔG kcal/mol)</h2>
    <div class="card">
        <p>Diagonal = own target binding. Off-diagonal = cross-reactivity.
           <strong>Selective</strong> = diagonal is most negative in its row.</p>
        <table>
            <tr>{header}</tr>
            {rows}
        </table>
    </div>
    """


def _build_phase5_rebinding(out: Path) -> str:
    """Build Phase 6 VIP Cavity Rebinding section."""
    data = _load_json(out / "phase5" / "phase5_rebinding_results.json")
    if not data:
        return "<h2>Phase 6: VIP Cavity Rebinding</h2><p>Not yet executed.</p>"

    sections = ""
    for target, result in data.items():
        n_snap = result.get("n_snapshots", 0)
        n_rebound = result.get("n_rebound", 0)
        success_rate = result.get("success_rate", "N/A")
        own_rmsd = result.get("own_rmsd_mean")
        own_rmsd_str = f"{own_rmsd:.2f} Å" if own_rmsd else "N/A"

        rate_color = "green" if n_rebound >= 3 else ("orange" if n_rebound >= 1 else "red")

        # Per-snapshot details
        snap_rows = ""
        for i, snap in enumerate(result.get("snapshots", [])):
            if not snap.get("success"):
                snap_rows += f"<tr><td>{i+1}</td><td colspan='3'>Failed</td></tr>"
                continue
            own = snap.get("rebind_own", {})
            own_r = own.get("rmsd_mean_A")
            own_str = f"{own_r:.2f} Å" if own_r else "N/A"
            own_dg = own.get("mmpbsa_dG")
            own_dg_str = f"{own_dg:.1f}" if own_dg is not None else "—"
            # The leg's OWN verdict, not a threshold re-applied here.
            # `rebound` is contact-based (>=1 persistent residue over the last
            # 50% of the run). The hardcoded 5.0 A test that stood here was
            # wrong in BOTH basis and calibration: rmsd_mean_A now carries a
            # FIT-FREE displacement, whose scale is larger than the self-fitted
            # number 5.0 was chosen against, and the self-fitted statistic
            # ranked an ESCAPED leg (contacts 4 -> 0) as the best-bound of its set.
            rebound = {True: "✓", False: "✗"}.get(own.get("rebound"), "—")
            own_fp = own.get("fraction_persistent")
            own_fp_str = f"{own_fp:.2f}" if own_fp is not None else "—"

            other_cells = ""
            for key, val in snap.items():
                if key.startswith("rebind_") and key != "rebind_own":
                    other_t = key.replace("rebind_", "")
                    other_r = val.get("rmsd_mean_A") if isinstance(val, dict) else None
                    other_dg = val.get("mmpbsa_dG") if isinstance(val, dict) else None
                    other_str = f"{other_r:.2f}" if other_r else "—"
                    dg_str = f" (ΔG={other_dg:.1f})" if other_dg is not None else ""
                    other_cells += f" | {other_t}={other_str}{dg_str}"

            # Removal test result
            removal = snap.get("removal_test", {})
            removal = removal if isinstance(removal, dict) else {}
            if removal.get("verdict_basis") == "persistent_contacts_q1_vs_q4":
                # Show the EVIDENCE the verdict is actually computed from.
                # rmsd_start_A/rmsd_end_A are still populated but are now only
                # diagnostics; presenting them as the reasoning misattributes it.
                q = removal.get("contacts_q1_q4") or {}
                k1, k4 = q.get("k_persistent_q1", "?"), q.get("k_persistent_q4", "?")
                if removal.get("escaped") is None:
                    rm_color, rm_status = "orange", "UNDEFINED (never bound)"
                    rm_str = f"{k1}→{k4} contacts"
                else:
                    rm_color = "green" if removal["escaped"] else "red"
                    rm_status = "Removable" if removal["escaped"] else "Stuck"
                    rm_str = f"{k1}→{k4} contacts"
            elif removal.get("escaped") is not None:
                rm_color = "green" if removal["escaped"] else "red"
                rm_str = f'{removal.get("rmsd_start_A","?")}→{removal.get("rmsd_end_A","?")} Å'
                rm_status = "Removable" if removal["escaped"] else "Stuck"
            elif removal:
                # escaped is None WITH a removal record now has a specific
                # meaning: the template never made contact, so "removable" is
                # not a question the cavity earned. Not the same as "no data".
                rm_color, rm_str, rm_status = "orange", "—", "UNDEFINED (never bound)"
            else:
                rm_color, rm_str, rm_status = "gray", "—", "N/A"

            _rep = snap.get("replica")
            _rep_str = f"rep{_rep} " if _rep is not None else ""
            snap_rows += f"""<tr>
                <td>{_rep_str}#{i+1} (frame {snap.get('frame_idx', '?')})</td>
                <td>{snap.get('total_contacts', '?')}</td>
                <td style="color:{rm_color};">{rm_str} ({rm_status})</td>
                <td>{own_fp_str}</td>
                <td>{own_str}</td>
                <td>{own_dg_str}</td>
                <td>{rebound}{other_cells}</td>
            </tr>"""

        # Selectivity with SI, p-value, H-bond, contacts
        sel_data = result.get("selectivity", {})
        sel_rows = ""
        for other_t, s in sel_data.items():
            si = s.get("selectivity_index")
            label = s.get("selectivity_label", "N/A")
            p_val = s.get("p_value")
            sig = s.get("significant")
            other_rmsd_val = s.get("other_rmsd_mean")
            other_std = s.get("other_rmsd_std")

            # SI color
            if label == "selective":
                si_color = "green"
            elif label == "weak":
                si_color = "orange"
            else:
                si_color = "red"

            si_str = f"{si:.2f}" if si else "N/A"
            p_str = f"{p_val:.4f}" if p_val is not None else "—"
            sig_str = "Yes" if sig else ("No" if sig is not None else "—")
            sig_color = "green" if sig else "red"

            other_rmsd_str2 = f"{other_rmsd_val:.2f}" if other_rmsd_val else "—"
            other_std_str = f" ± {other_std:.2f}" if other_std else ""

            # H-bond and contact info
            hb_own = s.get("own_hbond_mean", "—")
            hb_other = s.get("other_hbond_mean", "—")
            ct_own = s.get("own_contact_mean", "—")
            ct_other = s.get("other_contact_mean", "—")

            f_own = s.get("own_fraction_persistent_mean")
            f_other = s.get("other_fraction_persistent_mean")
            f_own_str = f"{f_own:.3f}" if f_own is not None else "—"
            f_other_str = f"{f_other:.3f}" if f_other is not None else "—"
            dcon = s.get("contrast_D")
            d_str = f"{dcon:+.3f}" if dcon is not None else "—"
            sel_rows += f"""<tr>
                <td>{other_t}</td>
                <td>{f_own_str} / {f_other_str}</td>
                <td>{d_str}</td>
                <td>{other_rmsd_str2}{other_std_str} Å</td>
                <td style="color:{si_color};font-weight:bold;">{si_str} ({label})</td>
                <td>{p_str}</td>
                <td style="color:{sig_color};">{sig_str}</td>
                <td>{hb_own} / {hb_other}</td>
                <td>{ct_own} / {ct_other}</td>
            </tr>"""

        # Fallback: old format
        if not sel_rows:
            for key, val in result.items():
                if key.startswith("other_") and key.endswith("_rmsd_mean"):
                    other_t = key.replace("other_", "").replace("_rmsd_mean", "")
                    sel_rows += f"""<tr>
                        <td>{other_t}</td><td>—</td><td>—</td><td>{val:.2f} Å</td>
                        <td colspan="5">—</td></tr>"""

        own_std = result.get("own_rmsd_std")
        own_std_str = f" ± {own_std:.2f}" if own_std else ""
        own_hb = result.get("own_hbond_mean")
        own_ct = result.get("own_contact_mean")
        extra_metrics = ""
        if own_hb is not None:
            extra_metrics += f"<p>Own H-bonds (avg): {own_hb}</p>"
        if own_ct is not None:
            extra_metrics += f"<p>Own contacts (avg): {own_ct}</p>"

        # PROVISIONAL BANNER — these numbers came from a Phase 4 MD that failed
        # its acceptance criteria. The reader must see that before the numbers.
        prov_banner = ""
        if result.get("provisional"):
            prov_banner = (
                f'<p style="background:#fdd;border:2px solid #c00;padding:8px;'
                f'font-weight:bold;color:#900;">PROVISIONAL — NOT VALIDATED. '
                f'{result.get("provisional_reason", "")}</p>')
        _nrep = result.get("n_source_replicas")
        rep_note = ""
        if _nrep:
            rep_note = (f'<p><small>Snapshots drawn from {_nrep} independent '
                        f'Phase 4 replica(s): '
                        f'{result.get("snapshots_per_replica")}. Snapshots from '
                        f'ONE trajectory are correlated (rho1 = 0.23); across '
                        f'replicas they are not.</small></p>')
        _npart = result.get("n_snapshots_partial") or 0
        part_note = ""
        if _npart:
            part_note = (f'<p style="color:#b60;"><small>{_npart} of '
                         f'{result.get("n_snapshots")} analysed snapshots had at '
                         f'least one FAILED leg; their surviving legs are still '
                         f'counted.</small></p>')

        sections += f"""
        <div class="card">
            <h3>{target} — Cavity Rebinding</h3>
            {prov_banner}
            <p>Success rate: <span style="color:{rate_color};font-weight:bold;font-size:1.2em;">
                {success_rate}</span> (≥3/5 = reliable; "rebound" = at least one
                persistent contact, NOT an RMSD threshold)</p>
            <p>Own template fit-free RMSD (diagnostic): {own_rmsd_str}{own_std_str}</p>
            {rep_note}
            {part_note}
            {extra_metrics}

            <h4>Per-Snapshot Results</h4>
            <table>
                <tr><th>Snapshot</th><th>Contacts</th><th>Removal Test</th>
                    <th>Persistent fraction</th>
                    <th>fit-free RMSD (diagnostic)</th>
                    <th>ΔG (kcal/mol)</th><th>Rebound</th></tr>
                {snap_rows}
            </table>
"""
        if sel_rows:
            sections += f"""
            <h4>Selectivity Analysis</h4>
            <p><small><b>SI = mean f<sub>own</sub> / mean f<sub>other</sub></b>, where f is the
               PERSISTENT-CONTACT FRACTION (cutoff 6 Å, persistence &gt; 0.5, last 50% of
               the run) — the same statistic PCSI* is built on, so Phase 5 and PCSI*
               cannot disagree about what "bound" means. SI &gt; 1.5: selective,
               1.0–1.5: weak, &lt; 1.0: cross-reactive. <b>D</b> is the bounded
               contrast. SI is NO LONGER derived from the RMSD column, which is a
               fit-free diagnostic and gates nothing.</small></p>
            <table>
                <tr><th>Other Target</th><th>f persistent (own/other)</th><th>contrast D</th>
                    <th>fit-free RMSD (diagnostic)</th><th>SI (label)</th>
                    <th>p-value</th><th>Significant?</th>
                    <th>H-bonds (own/other)</th><th>Contacts (own/other)</th></tr>
                {sel_rows}
            </table>
"""
        sections += "</div>"

        # Dual-imprinting results (if triggered)
        dual = result.get("dual_imprinting")
        if dual and isinstance(dual, dict) and "selectivity" in dual:
            dual_sel = dual.get("selectivity", {})
            reason = result.get("dual_imprinting_reason", "")
            n_gly = dual.get("n_glycan_sites", 0)
            dual_own_rmsd = dual.get("own_rmsd_mean")
            dual_own_str = f"{dual_own_rmsd:.2f} Å" if dual_own_rmsd else "N/A"

            dual_rows = ""
            for ot, s in dual_sel.items():
                si = s.get("selectivity_index")
                label = s.get("selectivity_label", "?")
                si_color = "green" if label == "selective" else (
                    "orange" if label == "weak" else "red")
                si_str = f"{si:.2f}" if si else "—"
                other_rmsd = s.get("other_rmsd_mean")
                or_str = f"{other_rmsd:.2f}" if other_rmsd else "—"
                dual_rows += f"""<tr>
                    <td>{ot}</td><td>{or_str} Å</td>
                    <td style="color:{si_color};font-weight:bold;">{si_str} ({label})</td>
                </tr>"""

            sections += f"""
            <div class="card" style="border-left:4px solid #e74c3c;">
                <h3>{target} — Dual-Imprinting (APBA + Glycan)</h3>
                <p><strong>Trigger:</strong> {reason}</p>
                <p>N-glycan sites: {n_gly}, APBA added as glycan recognition layer</p>
                <p>Own RMSD: {dual_own_str}, Success: {dual.get('success_rate', 'N/A')}</p>
                <table>
                    <tr><th>Other Target</th><th>RMSD</th><th>SI (label)</th></tr>
                    {dual_rows}
                </table>
                <p><small>{dual.get('note', '')}</small></p>
            </div>"""
        elif result.get("dual_imprinting_reason"):
            sections += f"""
            <div class="card" style="border-left:4px solid #95a5a6;">
                <p><em>{result['dual_imprinting_reason']}</em></p>
            </div>"""

    return f"<h2>Phase 5: VIP Cavity Rebinding (Zink 2018)</h2>{sections}"


def _build_references() -> str:
    return """
    <h2>References</h2>
    <div class="card">
    <ol>
        <li>Rajpal S et al. Sci. Rep. 2024;14:23057 — MMSD methodology</li>
        <li>Rajpal S, Mizaikoff B. J. Mater. Chem. B 2022;10:6618 — MMSD origin</li>
        <li>Sehit E et al. ACS Sensors 2024;9:1831 — epitope MD stability</li>
        <li>Sullivan MV et al. J. Phys. Chem. B 2019;123:5432 — MM-PBSA MIP</li>
        <li>Bossi AM et al. Anal. Bioanal. Chem. 2021;413:6101 — epitope selection</li>
        <li>Canfarotta F et al. Science Advances 2021;7:eabi9884 — design principles</li>
        <li>Bie Z et al. Angew. Chem. Int. Ed. 2015;54:10211 — boronic acid MIP</li>
    </ol>
    </div>
    """


def _build_footer() -> str:
    return f"""
    <div class="footer">
        <p>Generated by Epitope-MIP Screening Pipeline v2.0</p>
        <p>Targets: CD63 / CD81 / CD9 tetraspanin ECL2</p>
        <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """


def _load_json(path: Path) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _embed_image(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        return "<p><em>Image not available</em></p>"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" />'
