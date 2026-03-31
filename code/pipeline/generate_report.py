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
        _build_phase5(out),
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
            out / "phase3" / f"phase3_{target}_comparison.png"
        )

        rows = ""
        for pc in info["top_pcs"][:8]:
            mono_str = " + ".join(pc["monomers"])
            cls = "highlight" if pc.get("synergy") else ""
            rows += f"""<tr class="{cls}">
                <td>{pc['pc_id']}</td>
                <td>{mono_str}</td>
                <td>{pc.get('mmsd_sum', 0):.2f}</td>
                <td>{pc.get('smd_sum', 0):.2f}</td>
                <td>{pc.get('delta_sum', 0):.2f}</td>
                <td>{'Synergy' if pc.get('synergy') else '-'}</td>
            </tr>"""

        sections += f"""
        <div class="card">
            <h3>{target}</h3>
            <p>Fixed: {info.get('fixed_monomers', [])},
               {info.get('n_combinations', 0)} combinations screened,
               {info.get('high_affinity_count', 0)} high-affinity</p>
            {plot_html}
            <table>
                <tr><th>PC</th><th>Monomers</th><th>MMSD Sum</th>
                <th>SMD Sum</th><th>Delta</th><th>Synergy</th></tr>
                {rows}
            </table>
        </div>
        """

    return f"<h2>Phase 3: Multi-Monomer Simultaneous Docking (MMSD)</h2>{sections}"


def _build_phase4(out: Path) -> str:
    data = _load_json(out / "phase4" / "phase4_md_results.json")
    if not data:
        return "<h2>Phase 4: MD Validation</h2><p>No results.</p>"

    rows = ""
    for target, pcs in data.items():
        if target == "cross_reactivity" or not isinstance(pcs, dict):
            continue
        for pc_id, md in pcs.items():
            rows += f"""<tr>
                <td>{target}</td>
                <td>{pc_id}</td>
                <td>{md.get('rmsd_mean_nm', 'N/A')}</td>
                <td>{md.get('hbond_mean', 'N/A')}</td>
                <td>{md.get('mmpbsa', {}).get('delta_total_kcal', 'N/A')}</td>
                <td>{'OK' if md.get('success') else 'FAIL'}</td>
            </tr>"""

    return f"""
    <h2>Phase 4: MD Validation (GROMACS + MM-PBSA)</h2>
    <div class="card">
    <table>
        <tr><th>Target</th><th>PC</th><th>RMSD (nm)</th><th>H-bonds</th>
        <th>DG (kcal/mol)</th><th>Status</th></tr>
        {rows}
    </table>
    </div>
    """


def _build_phase5(out: Path) -> str:
    data = _load_json(out / "phase5" / "phase5_recipes.json")
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

    return f"<h2>Phase 5: Recommended Synthesis Recipes</h2>{cards}"


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
