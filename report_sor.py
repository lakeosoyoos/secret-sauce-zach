"""
report_sor.py — SOR-file variant of the clean report.

Takes a folder of .sor files, runs the same classification logic (single
wavelength), and produces the clean HTML + PDF output with likelihood column.
"""
import os, sys, glob, base64, subprocess, argparse
from datetime import datetime
from itertools import combinations
from io import BytesIO
import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sor_reader324802a import parse_sor_full

from report import (  # reuse helpers — all neutral
    _BASE_CSS, _embed_logo, _find_chrome, _outlier_probability,
    html_to_pdf_bytes, _fmt_time_gap,
    _COLOR_HIGH, _COLOR_MID, _COLOR_LOW,
)

_IOR = 1.4682
_LAUNCH_SKIP_M = 500
_END_BUFFER_M  = 200


def load_sor_file(path):
    r = parse_sor_full(path, trim=False)
    if r is None:
        raise ValueError(f'unparseable: {path}')
    trace = r['trace']
    sp = r.get('exfo_sampling_period')
    if not sp or sp <= 0:
        raise ValueError(f'bad sampling period: {path}')
    dz_m = 2.998e8 * sp / (2.0 * _IOR)
    pos = np.arange(len(trace)) * dz_m
    length_m = r.get('exfo_spans_length') or (pos[-1] if len(pos) else 0.0)
    events = r.get('events') or []
    # Max splice loss from event table (firmware-reported, interior events only)
    splice_vals = [e.get('splice_loss') for e in events
                   if e.get('splice_loss') is not None
                   and not e.get('is_end')
                   and (e.get('dist_km') or 0) > 0.01]
    max_splice = max((abs(v) for v in splice_vals), default=None) if splice_vals else None
    return {
        'name':     os.path.splitext(os.path.basename(path))[0],
        'filepath': path,
        'trace':    trace.astype(np.float32),
        'pos':      pos,
        'length':   float(length_m),
        'loss':     r.get('exfo_spans_loss'),
        'max_splice_dB': max_splice,
        'timestamp': r.get('date_time'),
        'wavelength': r.get('exfo_wavelength_nm') or r.get('wavelength'),
        'events':   events,
    }


def _pair_score(a, b, interior_start, interior_end):
    pa, pb = a['pos'], b['pos']
    ta, tb = a['trace'], b['trace']
    n = min(len(ta), len(tb))
    mask = (pa[:n] > interior_start) & (pa[:n] < interior_end)
    if mask.sum() < 50:
        return None
    return float(np.std(ta[:n][mask] - tb[:n][mask]))


def _distribution_chart(scores, p_dup, stats):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))
    log_s = np.log10(np.maximum(scores, 1e-9))
    ax1.hist(log_s, bins=50, color='#4A90D9', alpha=0.75, edgecolor='white', density=True)
    x = np.linspace(log_s.min() - 0.2, log_s.max() + 0.2, 400)
    ax1.plot(x, norm.pdf(x, stats['center_log'], stats['spread_log']),
             color='#b97000', linewidth=2, label='cluster fit')
    ax1.axvline(stats['center_log'], linestyle='--', color='#b97000', alpha=0.7)
    for z_line in (-3, -5, -10):
        ax1.axvline(stats['center_log'] + z_line * stats['spread_log'],
                    linestyle=':', color='#888', alpha=0.5)
    ax1.set_xticklabels([])
    ax1.set_xlabel('match score (log scale)')
    ax1.set_ylabel('density')
    ax1.set_title('Pair match-score distribution with cluster fit', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)

    # Tier masks: high ≥ 0.9, mid 0.5–0.9, low ≤ 0.5. Colors match the tables.
    p = np.asarray(p_dup)
    m_hi = p > 0.9
    m_md = (p > 0.5) & (~m_hi)
    m_lo = ~(m_hi | m_md)
    if m_lo.any():
        ax2.scatter(log_s[m_lo], p[m_lo], s=45, alpha=0.6, color=_COLOR_LOW,
                    edgecolor='white', linewidth=0.5,
                    label=f'Non-duplicate (n={int(m_lo.sum())})')
    if m_md.any():
        ax2.scatter(log_s[m_md], p[m_md], s=120, alpha=0.95,
                    color=_COLOR_MID, edgecolor='black', linewidth=1, zorder=4,
                    label=f'Borderline 50–90% (n={int(m_md.sum())})')
    if m_hi.any():
        ax2.scatter(log_s[m_hi], p[m_hi], s=140, alpha=0.95,
                    color=_COLOR_HIGH, edgecolor='black', linewidth=1, zorder=5,
                    label=f'Duplicate ≥90% (n={int(m_hi.sum())})')
    ax2.axhline(0.9, color=_COLOR_HIGH, linestyle=':', alpha=0.4, linewidth=1)
    ax2.axhline(0.5, color=_COLOR_MID, linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xticklabels([])
    ax2.set_xlabel('match score (log scale)')
    ax2.set_ylabel('duplicate likelihood')
    ax2.set_title('Per-pair likelihood vs match score', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def build_report_sor(folder, title, out_pdf):
    paths = sorted(glob.glob(os.path.join(folder, '*.sor')))
    files = []
    for p in paths:
        try:
            files.append(load_sor_file(p))
        except Exception as e:
            print(f'  skip {os.path.basename(p)}: {e}')
    if len(files) < 2:
        raise RuntimeError(f'Not enough usable .sor files in {folder}')
    print(f'Loaded {len(files)} .sor files from {folder}')

    min_L = min(f['length'] for f in files if f['length'] > 0)
    interior_start = _LAUNCH_SKIP_M
    interior_end = min_L - _END_BUFFER_M
    if interior_end - interior_start < 100:
        interior_start = max(50.0, min_L * 0.05)
        interior_end = min_L * 0.95
    print(f'Interior window: {interior_start:.0f}–{interior_end:.0f} m  '
          f'(common span {min_L:.0f} m)')

    pairs = []
    for a, b in combinations(files, 2):
        s = _pair_score(a, b, interior_start, interior_end)
        if s is None:
            continue
        pairs.append({'a': a['name'], 'b': b['name'], 'score': s})
    if not pairs:
        raise RuntimeError('No comparable pairs after interior masking')

    scores = np.array([p['score'] for p in pairs], dtype=np.float64)
    p_dup, stats = _outlier_probability(scores)
    for p, pd, z in zip(pairs, p_dup, stats['z']):
        p['p_dup'] = float(pd)
        p['z'] = float(z)

    order = np.argsort(scores)
    n99 = int((p_dup > 0.99).sum())
    n50 = int((p_dup > 0.5).sum())
    n10 = int((p_dup > 0.1).sum())
    print(f'Likelihood >99%: {n99}   >50%: {n50}   >10%: {n10}')

    best_partner = {}
    for idx, f in enumerate(files):
        best = None
        for p in pairs:
            if f['name'] not in (p['a'], p['b']):
                continue
            if best is None or p['score'] < best['score']:
                best = p
        best_partner[f['name']] = best

    verdict_block = (f'<div class="verdict-box verdict-confirm">'
                     f'<b>{n50} duplicate pair(s) identified</b> at ≥50% likelihood; '
                     f'{n99} at ≥99% likelihood across {len(pairs)} pairs.</div>'
                     if n50 else
                     '<div class="verdict-box verdict-dispute">'
                     '<b>No duplicate pairs identified</b> at ≥50% likelihood.</div>')

    dist_chart = _distribution_chart(scores, p_dup, stats)

    file_rows = ''
    for f in sorted(files, key=lambda x: x['name']):
        bp = best_partner.get(f['name'])
        if bp is None:
            continue
        partner = bp['b'] if bp['a'] == f['name'] else bp['a']
        pd_val = bp['p_dup']
        pd_color = '#2d8f48' if pd_val > 0.9 else ('#b97000' if pd_val > 0.1 else '#888')
        verdict_cell = (f'<span class="dup">DUPLICATE of {partner}</span>'
                        if pd_val > 0.5 else
                        f'<span class="na">unique (closest: {partner})</span>')
        loss_cell = f'{f["loss"]:.3f}' if f['loss'] is not None else '—'
        file_rows += (f'<tr><td class="pair-cell">{f["name"]}</td>'
                      f'<td class="center">{f["length"]/1000:.3f}</td>'
                      f'<td class="center">{loss_cell}</td>'
                      f'<td class="center">{bp["score"]:.4f}</td>'
                      f'<td class="center" style="color:{pd_color};font-weight:600">{pd_val*100:.2f}%</td>'
                      f'<td class="center">{verdict_cell}</td></tr>')

    top_rows = ''
    for rank, k in enumerate(order[:30], 1):
        p = pairs[k]
        pd_val = p['p_dup']
        pd_color = '#2d8f48' if pd_val > 0.9 else ('#b97000' if pd_val > 0.1 else '#888')
        top_rows += (f'<tr><td class="center">{rank}</td>'
                     f'<td class="pair-cell">{p["a"]} ↔ {p["b"]}</td>'
                     f'<td class="center">{p["score"]:.4f}</td>'
                     f'<td class="center" style="color:{pd_color};font-weight:600">{pd_val*100:.2f}%</td></tr>')

    # Confirmed-duplicate detail table (p_dup > 0.5)
    file_by_name = {f['name']: f for f in files}
    dup_pairs_sorted = sorted([p for p in pairs if p['p_dup'] > 0.5],
                              key=lambda q: -q['p_dup'])
    dup_detail_rows = ''
    for p in dup_pairs_sorted:
        fa = file_by_name.get(p['a']); fb = file_by_name.get(p['b'])
        if fa is None or fb is None:
            continue
        ta, tb = fa.get('timestamp'), fb.get('timestamp')
        gap_str = _fmt_time_gap(abs(ta - tb)) if ta and tb else '—'
        a_ms, b_ms = fa.get('max_splice_dB'), fb.get('max_splice_dB')
        a_sl, b_sl = fa.get('loss'), fb.get('loss')
        ms_cell = (f'<td class="center">{abs(a_ms - b_ms)*1000:.0f}</td>'
                   if a_ms is not None and b_ms is not None
                   else '<td class="center na">—</td>')
        sl_cell = (f'<td class="center">{abs(a_sl - b_sl)*1000:.0f}</td>'
                   if a_sl is not None and b_sl is not None
                   else '<td class="center na">—</td>')
        pd_val = p['p_dup']
        pd_color = '#2d8f48' if pd_val > 0.9 else '#b97000'
        dup_detail_rows += (f'<tr><td class="pair-cell">{p["a"]} ↔ {p["b"]}</td>'
                            f'<td class="center">{gap_str}</td>'
                            f'{ms_cell}{sl_cell}'
                            f'<td class="center" style="color:{pd_color};font-weight:600">{pd_val*100:.2f}%</td></tr>')
    dup_detail_block = ''
    if dup_detail_rows:
        wl_hdr = f'{int(files[0].get("wavelength") or 0)} nm' if files else ''
        dup_detail_block = f'''
<div class="dir-banner">Confirmed duplicate pairs (≥50% likelihood) — detail ({wl_hdr})</div>
<table class="vote-table">
<tr><th style="text-align:left">Pair</th><th>Time gap</th>
  <th>max splice Δ (mdB)</th><th>span loss Δ (mdB)</th>
  <th>Duplicate likelihood</th></tr>
{dup_detail_rows}
</table>
'''

    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>{_BASE_CSS}</style></head><body>
{_embed_logo()}
<h1>{title}</h1>
<div class="subtitle">{len(files)} files &bull; {len(pairs)} pairs &bull; generated {generated}</div>

{verdict_block}

<div class="cards">
  <div class="card"><div class="card-label">Files</div><div class="card-value">{len(files)}</div></div>
  <div class="card"><div class="card-label">Pairs</div><div class="card-value">{len(pairs)}</div></div>
  <div class="card"><div class="card-label">Likelihood &gt; 99%</div>
    <div class="card-value good">{n99}</div></div>
  <div class="card"><div class="card-label">Likelihood &gt; 50%</div>
    <div class="card-value">{n50}</div></div>
  <div class="card"><div class="card-label">Likelihood &gt; 10%</div>
    <div class="card-value">{n10}</div></div>
</div>

<div class="dir-banner">Distribution — duplicates vs non-duplicates</div>
<img src="data:image/png;base64,{dist_chart}" class="chart-img" />

<div class="dir-banner">Per-file verdict</div>
<table class="vote-table">
<tr><th style="text-align:left">File</th>
    <th>Length (km)</th><th>Span loss (dB)</th>
    <th>best-match score</th><th>Duplicate likelihood</th><th>Verdict</th></tr>
{file_rows}
</table>

<div class="dir-banner">Top 30 pairs — tightest match score</div>
<table class="vote-table">
<tr><th>Rank</th><th style="text-align:left">Pair</th>
    <th>match score</th><th>Duplicate likelihood</th></tr>
{top_rows}
</table>
{dup_detail_block}
</body></html>'''

    pdf_bytes = html_to_pdf_bytes(html, base_url=folder)
    with open(out_pdf, 'wb') as fh:
        fh.write(pdf_bytes)
    print(f'PDF:  {out_pdf}')
    return out_pdf


def run_sor_bytes(folder, title):
    """Run SOR mode and return (pdf_bytes, n_files, n_pairs)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_pdf = os.path.join(td, 'report.pdf')
        build_report_sor(folder, title, tmp_pdf)
        with open(tmp_pdf, 'rb') as fh:
            pdf_bytes = fh.read()
    n_files = len(glob.glob(os.path.join(folder, '*.sor')))
    n_pairs = n_files * (n_files - 1) // 2
    return pdf_bytes, n_files, n_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sor-dir', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--out-pdf', required=True)
    args = parser.parse_args()
    build_report_sor(args.sor_dir, args.title, args.out_pdf)


if __name__ == '__main__':
    main()
