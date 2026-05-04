"""
report.py
---------
Generates a duplicate-classification report for a folder of OTDR JSON files.
"""
import os, sys, json, glob, base64, subprocess
from datetime import datetime
from itertools import combinations
from io import BytesIO
import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
JSON_FOLDER = os.path.join(HERE, 'data')
WL_ORDER = [1310, 1550, 1625]
WL_COLOR = {1310: '#1F77B4', 1550: '#2CA02C', 1625: '#D62728'}

_INTERIOR_MIN_M = 1100
_INTERIOR_MAX_M = 60000
_SCORE_GATE     = 0.025


def _decode(pts_b64, n):
    raw = base64.b64decode(pts_b64)
    s = np.frombuffer(raw, dtype='<u2')[:n].astype(np.float64)
    return 64.0 - s / 1024.0


def _fmt_time_gap(sec):
    """Render an integer seconds count as a compact human string."""
    if sec is None:
        return '—'
    sec = int(sec)
    if sec < 60:
        return f'{sec}s'
    if sec < 3600:
        return f'{sec//60}m {sec%60:02d}s'
    if sec < 86400:
        h, r = divmod(sec, 3600)
        return f'{h}h {r//60:02d}m'
    d, r = divmod(sec, 86400)
    return f'{d}d {r//3600:02d}h'


def _parse_iso_ts(s):
    """Return (raw_string, epoch_seconds_or_None). Handles ISO-8601 w/ 'Z'."""
    if not s:
        return '', None
    try:
        from datetime import datetime as _dt
        s2 = s.replace('Z', '+00:00')
        return s, _dt.fromisoformat(s2).timestamp()
    except Exception:
        return s, None


def load_file(path):
    with open(path) as f:
        d = json.load(f)
    name = os.path.basename(path).split('_')[0].strip()
    per_wl = {}
    for meas in d['Measurement']['OtdrMeasurements']:
        wl = int(meas['Wavelength'])
        dp = meas['DataPoints']
        n = int(dp['NumberOfPoints'])
        res = float(dp['Resolution'])
        fp = float(dp['FirstPointPosition'].replace(',', ''))
        trace = _decode(dp['Points'], n)
        pos = np.arange(n) * res + fp
        results = meas.get('Results') or {}
        def _num(k):
            v = results.get(k)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        per_wl[wl] = {
            'trace': trace, 'pos': pos,
            'max_splice_dB': _num('MaximumSpliceLoss'),
            'span_loss_dB':  _num('AveragedLoss'),
            'length_m':      _num('Length'),
        }
    dt_raw, dt_epoch = _parse_iso_ts(d.get('TestDateTime', ''))
    return {'name': name, 'filepath': path,
            'test_dt': dt_raw, 'test_epoch': dt_epoch, 'wl': per_wl}


def load_trc_file(path):
    """Parse a .trc file into the same per-file dict shape as load_file (JSON)."""
    from trc_parser import parse_trc_file
    r = parse_trc_file(path)
    name = os.path.basename(path).split('_')[0].split('.')[0].strip()
    per_wl = {}
    IOR = 1.4682
    for wlblock in r.get('wavelengths') or []:
        wl_nm = int(wlblock['wavelength_nm'])
        sp = wlblock.get('sampling_period_s')
        if not sp:
            continue
        dz = 2.998e8 * sp / (2.0 * IOR)
        samples = wlblock['samples']
        trace = 64.0 - samples.astype(np.float64) / 1024.0
        pos = np.arange(len(trace)) * dz
        # Max interior splice loss from event table (skip end-of-fiber events)
        events = wlblock.get('events') or []
        spl_vals = [abs(e.get('loss_db'))
                    for e in events
                    if e.get('loss_db') is not None
                    and (e.get('position_m') or 0) > 0.01
                    and (str(e.get('type', '')).lower() != 'end')]
        max_splice = max(spl_vals) if spl_vals else None
        per_wl[wl_nm] = {
            'trace': trace, 'pos': pos,
            'max_splice_dB': max_splice,
            'span_loss_dB': wlblock.get('span_loss_db'),
            'length_m':     wlblock.get('length_m'),
        }
    ts = r.get('timestamp')
    if ts:
        from datetime import datetime as _dt
        dt_raw = _dt.fromtimestamp(ts).isoformat()
    else:
        dt_raw = ''
    return {'name': name, 'filepath': path,
            'test_dt': dt_raw, 'test_epoch': float(ts) if ts else None,
            'wl': per_wl}


def _outlier_probability(values):
    """P(duplicate) per pair via robust-bulk fit + Bonferroni tail in log space."""
    v = np.asarray(values, dtype=np.float64)
    N = len(v)
    log_v = np.log10(np.maximum(v, 1e-9))
    med = float(np.median(log_v))
    mad = float(np.median(np.abs(log_v - med)))
    spread = max(mad * 1.4826, 1e-6)
    z = (log_v - med) / spread
    p_tail = norm.cdf(z)
    expected_fp = N * p_tail
    p_dup = np.clip(1.0 - expected_fp, 0.0, 1.0)
    return p_dup, {'center_log': med, 'spread_log': spread, 'N': N,
                   'z': z, 'p_tail': p_tail, 'expected_fp': expected_fp}


def _score(a, b, wl):
    ta, tb = a['wl'][wl]['trace'], b['wl'][wl]['trace']
    pa = a['wl'][wl]['pos']
    n = min(len(ta), len(tb))
    # Use length-aware interior window so short coils aren't discarded
    length_m = a['wl'][wl].get('length_m') or b['wl'][wl].get('length_m')
    mask = _interior_mask(pa[:n], length_m=length_m)
    if mask.sum() < 50:
        return None
    return float(np.std(ta[:n][mask] - tb[:n][mask]))


def _detrend(trace, pos):
    """Subtract best-fit linear (offset + slope) so two traces with different
    launch power / attenuation gain can still be shape-compared."""
    A = np.vstack([pos, np.ones_like(pos)]).T
    m, c = np.linalg.lstsq(A, trace, rcond=None)[0]
    return trace - (m * pos + c)


def _interior_mask(pos, length_m=None):
    """Pick an interior window that works for both km-scale fibers and short
    coils. For short fibers (< 800 m), use a 1 m launch buffer + 5% end
    buffer — anything tighter on coils discards the very fiber region we
    want to compare. For long fibers, use the production 1100–60000 m window."""
    if length_m is not None and length_m > 0 and length_m < 800:
        lo = max(1.0, length_m * 0.03)
        hi = max(lo + 1.0, length_m - max(0.5, length_m * 0.03))
    else:
        lo, hi = _INTERIOR_MIN_M, _INTERIOR_MAX_M
    return (pos > lo) & (pos < hi)


def _shape_r(a, b, wl):
    """Detrended Pearson correlation between two traces at one wavelength.
    Returns r in [-1, 1] or None if insufficient samples. r ≈ 1 → same fiber."""
    if wl not in a['wl'] or wl not in b['wl']:
        return None
    ta, tb = a['wl'][wl]['trace'], b['wl'][wl]['trace']
    pa = a['wl'][wl]['pos']
    L = min(a['wl'][wl].get('length_m') or 0, b['wl'][wl].get('length_m') or 0) or None
    n = min(len(ta), len(tb))
    mask = _interior_mask(pa[:n], length_m=L)
    if mask.sum() < 50:
        return None
    da = _detrend(ta[:n][mask].astype(np.float64), pa[:n][mask])
    db = _detrend(tb[:n][mask].astype(np.float64), pa[:n][mask])
    sa, sb = np.std(da), np.std(db)
    if sa == 0 or sb == 0:
        return None
    return float(np.dot(da - da.mean(), db - db.mean()) / (sa * sb * len(da)))


def _shape_tier(r):
    """Bin a Pearson r into a same-fiber tier."""
    if r is None:
        return None
    if r >= 0.99:
        return 'high'
    if r >= 0.95:
        return 'mid'
    return 'low'


def _shape_color(r):
    t = _shape_tier(r)
    return _COLOR_HIGH if t == 'high' else (_COLOR_MID if t == 'mid' else _COLOR_LOW)


def _find_chrome():
    for p in ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
              '/usr/bin/google-chrome', '/usr/bin/chromium-browser']:
        if os.path.isfile(p):
            return p
    return None


def _embed_logo():
    logo_path = os.path.join(HERE, 'zerodblogo.png')
    if not os.path.exists(logo_path):
        return ''
    with open(logo_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return (f'<div style="text-align:center; margin-bottom:16px;">'
            f'<img src="data:image/png;base64,{b64}" style="height:60px; margin-left:-30px;" />'
            f'</div>')


_BASE_CSS = """
@page { size: landscape; margin: 10mm 10mm 18mm 10mm;
  @bottom-center { content: "Page " counter(page) " of " counter(pages); font-size: 8px; }
  @bottom-right  { content: "\\A9  ZeroDB"; font-size: 8px; } }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        color:#2c2c2a; padding:16px; font-size:11px; max-width:1400px; margin:0 auto; }
h1 { font-size:20px; font-weight:500; margin-bottom:2px; }
h2 { font-size:14px; font-weight:500; margin:24px 0 8px; page-break-after:avoid; }
.subtitle { font-size:11px; color:#888; margin-bottom:16px; }
.chart-img { width:100%; border-radius:8px; border:1px solid #ddd; margin-bottom:16px; }
.cards { display:flex; gap:10px; margin-bottom:16px; }
.card { flex:1; background:#fff; border:1px solid rgba(0,0,0,.08); border-radius:10px; padding:12px 14px; }
.card-label { font-size:9px; color:#999; text-transform:uppercase; letter-spacing:.04em; }
.card-value { font-size:22px; font-weight:600; }
.card-value.good { color:#2d8f48; }
.card-sub { font-size:9px; color:#999; margin-top:2px; }
.vote-table { width:100%; border-collapse:collapse; font-size:9.5px;
               font-family:'SF Mono','Courier New',monospace; margin-bottom:16px;
               page-break-inside:avoid; }
.vote-table th { background:#f4f3f0; padding:5px 6px; text-align:center;
                  font-weight:600; border:0.5px solid #ddd; font-size:8px; color:#555; }
.vote-table td { padding:4px 6px; border:0.5px solid #ddd; }
.pair-cell { text-align:left !important; font-weight:600; }
.center { text-align:center; }
.dup { color:#2d8f48; font-weight:700; }
.na  { color:#888;    font-weight:500; }
.dir-banner { background:#2C3E50; color:white; padding:10px 16px; border-radius:8px;
               font-size:14px; font-weight:600; margin:28px 0 12px; }
.verdict-box { padding:14px 18px; border-radius:10px; font-size:13px; font-weight:600;
               margin:16px 0; }
.verdict-confirm { background:#e8f5ec; color:#1f6b35; border:1px solid #bce0c6; }
.verdict-dispute { background:#fbeedf; color:#8a5200; border:1px solid #f0d2a3; }
"""


_COLOR_HIGH = '#2d8f48'   # p_dup > 0.9  — solid duplicate (green)
_COLOR_MID  = '#b97000'   # 0.5 < p_dup ≤ 0.9  — borderline (orange)
_COLOR_LOW  = '#888'      # p_dup ≤ 0.5  — non-duplicate (grey)


def _tier(p):
    """Return 'high' / 'mid' / 'low' based on p_dup (or is_dup = high)."""
    if p.get('is_dup') or p.get('p_dup', 0) > 0.9:
        return 'high'
    if p.get('p_dup', 0) > 0.5:
        return 'mid'
    return 'low'


def _is_highlighted(p):
    return _tier(p) != 'low'


def _tier_split(all_pairs_list, key_fn):
    """Split pair values into (high, mid, low) lists using key_fn(pair)->value|None."""
    hi, md, lo = [], [], []
    for p in all_pairs_list:
        v = key_fn(p)
        if v is None:
            continue
        t = _tier(p)
        (hi if t == 'high' else md if t == 'mid' else lo).append(v)
    return hi, md, lo


def chart_distribution(all_pairs_list):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=False)
    panels = [(1310, axes[0]), (1550, axes[1]), (1625, axes[2])]
    for wl, ax in panels:
        hi, md, lo = _tier_split(all_pairs_list, lambda p: p['score'].get(wl))
        dup_v = hi + md  # combined "highlighted" for separation-band math
        if dup_v and lo:
            dup_max = max(dup_v); non_min = min(lo)
            if non_min > dup_max:
                ax.axvspan(dup_max, non_min, color=_COLOR_HIGH, alpha=0.15,
                           label=f'separation band ({non_min/dup_max:.2f}×)')
            ax.set_title(f'{wl} nm — duplicates separate {non_min/dup_max:.1f}× below non-duplicates',
                         fontweight='bold', loc='left')
        else:
            ax.set_title(f'{wl} nm — match-score distribution', fontweight='bold', loc='left')
        if lo:
            ax.scatter(lo, rng.uniform(0.25, 0.55, len(lo)),
                       color=_COLOR_LOW, alpha=0.55, s=55, edgecolor='white', linewidth=0.4,
                       label=f'Non-duplicate (n={len(lo)})')
        if md:
            ax.scatter(md, rng.uniform(0.55, 0.70, len(md)),
                       color=_COLOR_MID, alpha=0.95, s=140, edgecolor='black', linewidth=1,
                       zorder=4, label=f'Borderline 50–90% (n={len(md)})')
        if hi:
            ax.scatter(hi, rng.uniform(0.70, 0.85, len(hi)),
                       color=_COLOR_HIGH, alpha=0.95, s=170, edgecolor='black', linewidth=1,
                       zorder=5, label=f'Duplicate ≥90% (n={len(hi)})')
        ax.axvline(_SCORE_GATE, color=_COLOR_MID, linestyle='--', linewidth=1.3,
                   label='decision threshold')
        ax.set_xscale('log')
        all_v = hi + md + lo
        if all_v:
            ax.set_xlim(min(all_v) * 0.7, max(all_v) * 1.3)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xlabel(f'match score @ {wl} nm (log scale)', fontsize=10)
        ax.grid(axis='x', alpha=0.3, which='both')
        ax.legend(loc='upper right', fontsize=8, ncol=2)

    ax = axes[3]
    hi, md, lo = _tier_split(all_pairs_list, lambda p: sum(p['score'].values()))
    dup_sum = hi + md
    if dup_sum and lo:
        d_max = max(dup_sum); n_min = min(lo)
        if n_min > d_max:
            ax.axvspan(d_max, n_min, color=_COLOR_HIGH, alpha=0.15,
                       label=f'separation band ({n_min/d_max:.2f}×)')
    if lo:
        ax.scatter(lo, rng.uniform(0.25, 0.55, len(lo)),
                   color=_COLOR_LOW, alpha=0.55, s=55, edgecolor='white', linewidth=0.4,
                   label=f'Non-duplicate (n={len(lo)})')
    if md:
        ax.scatter(md, rng.uniform(0.55, 0.70, len(md)),
                   color=_COLOR_MID, alpha=0.95, s=140, edgecolor='black', linewidth=1,
                   zorder=4, label=f'Borderline 50–90% (n={len(md)})')
    if hi:
        ax.scatter(hi, rng.uniform(0.70, 0.85, len(hi)),
                   color=_COLOR_HIGH, alpha=0.95, s=170, edgecolor='black', linewidth=1,
                   zorder=5, label=f'Duplicate ≥90% (n={len(hi)})')
    ax.set_xscale('log')
    all_sum = hi + md + lo
    if all_sum:
        ax.set_xlim(min(all_sum) * 0.7, max(all_sum) * 1.3)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xlabel('combined match score across 3 wavelengths (log scale)', fontsize=10)
    ax.grid(axis='x', alpha=0.3, which='both')
    ax.set_title('Combined 3λ match-score distribution', fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    fig.suptitle(f'Match-score distribution across {len(all_pairs_list)} pairs',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def chart_histogram(all_pairs_list):
    hi, md, lo = _tier_split(all_pairs_list, lambda p: sum(p['score'].values()))
    dup_sum = hi + md
    all_sum = hi + md + lo
    fig, ax = plt.subplots(figsize=(13, 4.5))
    if not all_sum:
        return None
    bins = np.linspace(0, max(all_sum) * 1.05, 60)
    if lo:
        ax.hist(lo, bins=bins, color=_COLOR_LOW, alpha=0.75,
                label=f'Non-duplicate (n={len(lo)})')
    counts, _ = np.histogram(lo or [0], bins=bins)
    y_mark = max(counts) * 0.75 if len(counts) else 1
    # Draw vertical lines and markers per tier so the color matches the table
    for d in md:
        ax.axvline(d, color=_COLOR_MID, linewidth=2, alpha=0.9)
    for d in hi:
        ax.axvline(d, color=_COLOR_HIGH, linewidth=2, alpha=0.9)
    if md:
        ax.scatter(md, [y_mark]*len(md), color=_COLOR_MID, s=160, zorder=4,
                   edgecolor='black', linewidth=1.0,
                   label=f'Borderline 50–90% (n={len(md)})')
    if hi:
        ax.scatter(hi, [y_mark]*len(hi), color=_COLOR_HIGH, s=200, zorder=5,
                   edgecolor='black', linewidth=1.2,
                   label=f'Duplicate ≥90% (n={len(hi)})')
    if dup_sum and lo and min(lo) > max(dup_sum):
        ax.axvspan(max(dup_sum), min(lo), color=_COLOR_HIGH, alpha=0.12,
                   label='separation band')
    ax.axvline(_SCORE_GATE * 3, color=_COLOR_MID, linestyle='--', linewidth=1.3,
               label='decision threshold')
    ax.set_xticklabels([])
    ax.set_xlabel('combined match score across 3 wavelengths')
    ax.set_ylabel('Number of non-duplicate pairs')
    ax.set_title('Histogram — combined 3λ match score', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def build_report(files, all_pairs_list, truth_dups, out_path, title='Duplicate Classification Report'):
    truth_dups = truth_dups or set()
    pair_lookup = {tuple(sorted([p['a'], p['b']])): p for p in all_pairs_list}

    combined = np.array([p['sum_score'] for p in all_pairs_list], dtype=np.float64)
    p_dup_sigma_arr, prob_stats = _outlier_probability(combined)
    # Pearson-shape contribution: r ≥ 0.99 → 1.0,  r ≤ 0.95 → 0,  linear in between.
    # This catches short-fiber duplicates that look noisy in σ but match in shape.
    def _r_to_p(r):
        if r is None:
            return 0.0
        if r >= 0.99:
            return 1.0
        if r <= 0.95:
            return 0.0
        return float((r - 0.95) / 0.04)
    p_dup_r_arr = np.array([_r_to_p(p.get('r_min')) for p in all_pairs_list],
                           dtype=np.float64)
    # Combined likelihood = max of σ-outlier and shape-correlation tiers.
    p_dup_arr = np.maximum(p_dup_sigma_arr, p_dup_r_arr)
    for p, pd, pdr, z in zip(all_pairs_list, p_dup_arr, p_dup_r_arr, prob_stats['z']):
        p['p_dup'] = float(pd)
        p['p_dup_r'] = float(pdr)
        p['z'] = float(z)

    best_partner = {}
    for f in files:
        best = None
        for g in files:
            if g['name'] == f['name']: continue
            p = pair_lookup[tuple(sorted([f['name'], g['name']]))]
            if best is None or p['sum_score'] < best['sum_score']:
                best = {'partner': g['name'], 'sum_score': p['sum_score'], 'pair': p}
        best_partner[f['name']] = best

    dup_pairs = [p for p in all_pairs_list
                 if all((p['score'][wl] is not None and p['score'][wl] < _SCORE_GATE)
                        for wl in WL_ORDER)]
    used = set()
    confirmed = []
    for p in sorted(dup_pairs, key=lambda q: q['sum_score']):
        if p['a'] in used or p['b'] in used: continue
        confirmed.append(p)
        used.add(p['a']); used.add(p['b'])

    dup_names = {n for p in confirmed for n in (p['a'], p['b'])}

    truth_found = {tuple(sorted([p['a'], p['b']])) for p in confirmed}
    tp = len(truth_found & truth_dups)
    fp = len(truth_found - truth_dups)
    fn = len(truth_dups - truth_found)

    distribution_chart = chart_distribution(all_pairs_list)
    histogram_chart = chart_histogram(all_pairs_list)

    file_rows = ''
    for f in sorted(files, key=lambda x: x['name']):
        bp = best_partner[f['name']]
        partner = bp['partner']
        is_dup = f['name'] in dup_names
        verdict_html = (f'<td class="center"><span class="dup">DUPLICATE of {partner}</span></td>'
                        if is_dup else
                        f'<td class="center"><span class="na">unique (closest: {partner})</span></td>')
        pair = pair_lookup[tuple(sorted([f['name'], partner]))]
        wl_cells = ''
        for wl in WL_ORDER:
            sc = pair['score'][wl]
            if sc is None:
                wl_cells += '<td class="center na">---</td>'
            else:
                color = '#2d8f48' if sc < _SCORE_GATE else '#c0392b'
                wl_cells += f'<td class="center" style="color:{color};font-weight:600">{sc:.4f}</td>'
        pd_val = bp['pair']['p_dup']
        pd_color = '#2d8f48' if pd_val > 0.9 else ('#b97000' if pd_val > 0.1 else '#888')
        r_min = bp['pair'].get('r_min')
        if r_min is None:
            r_cell = '<td class="center na">—</td>'
        else:
            r_cell = f'<td class="center" style="color:{_shape_color(r_min)};font-weight:600">{r_min:.4f}</td>'
        file_rows += (f'<tr><td class="pair-cell">{f["name"]}</td>'
                      f'<td class="center">{f["test_dt"][:19]}</td>'
                      f'{wl_cells}'
                      f'<td class="center">{bp["sum_score"]:.3f}</td>'
                      f'<td class="center" style="color:{pd_color};font-weight:600">{pd_val*100:.2f}%</td>'
                      f'{r_cell}'
                      f'{verdict_html}</tr>')

    # ---- Confirmed-duplicate detail table (pairs with P(dup) > 0.5) -----
    file_by_name = {f['name']: f for f in files}
    dup_pairs_sorted = sorted(
        [p for p in all_pairs_list if p['p_dup'] > 0.5],
        key=lambda q: -q['p_dup'])
    dup_detail_rows = ''
    for p in dup_pairs_sorted:
        fa = file_by_name.get(p['a']); fb = file_by_name.get(p['b'])
        if fa is None or fb is None:
            continue
        # Time gap (file-level, not per-λ: one timestamp per acquisition)
        if fa.get('test_epoch') and fb.get('test_epoch'):
            gap_sec = int(abs(fa['test_epoch'] - fb['test_epoch']))
            gap_str = _fmt_time_gap(gap_sec)
        else:
            gap_str = '—'
        # Per-wavelength cells: max splice-loss Δ (mdB), span-loss Δ (mdB), shape r
        ms_cells = ''
        sl_cells = ''
        sr_cells = ''
        for wl in WL_ORDER:
            a_ms = fa['wl'].get(wl, {}).get('max_splice_dB')
            b_ms = fb['wl'].get(wl, {}).get('max_splice_dB')
            a_sl = fa['wl'].get(wl, {}).get('span_loss_dB')
            b_sl = fb['wl'].get(wl, {}).get('span_loss_dB')
            if a_ms is not None and b_ms is not None:
                ms_cells += f'<td class="center">{abs(a_ms - b_ms)*1000:.0f}</td>'
            else:
                ms_cells += '<td class="center na">—</td>'
            if a_sl is not None and b_sl is not None:
                sl_cells += f'<td class="center">{abs(a_sl - b_sl)*1000:.0f}</td>'
            else:
                sl_cells += '<td class="center na">—</td>'
            r_wl = (p.get('shape_r') or {}).get(wl)
            if r_wl is None:
                sr_cells += '<td class="center na">—</td>'
            else:
                sr_cells += (f'<td class="center" style="color:{_shape_color(r_wl)};'
                             f'font-weight:600">{r_wl:.4f}</td>')
        pd_val = p['p_dup']
        pd_color = '#2d8f48' if pd_val > 0.9 else '#b97000'
        dup_detail_rows += (f'<tr><td class="pair-cell">{p["a"]} ↔ {p["b"]}</td>'
                            f'<td class="center">{gap_str}</td>'
                            f'{ms_cells}{sl_cells}{sr_cells}'
                            f'<td class="center" style="color:{pd_color};font-weight:600">{pd_val*100:.2f}%</td></tr>')
    dup_detail_block = ''
    if dup_detail_rows:
        ms_hdrs = ''.join(f'<th>max splice Δ @ {wl} (mdB)</th>' for wl in WL_ORDER)
        sl_hdrs = ''.join(f'<th>span loss Δ @ {wl} (mdB)</th>' for wl in WL_ORDER)
        sr_hdrs = ''.join(f'<th>shape r @ {wl}</th>' for wl in WL_ORDER)
        dup_detail_block = f'''
<div class="dir-banner">Confirmed duplicate pairs (≥50% likelihood) — detail</div>
<table class="vote-table">
<tr><th style="text-align:left">Pair</th><th>Time gap</th>
  {ms_hdrs}{sl_hdrs}{sr_hdrs}<th>Duplicate likelihood</th></tr>
{dup_detail_rows}
</table>
'''

    nonconf_sorted = sorted(
        [p for p in all_pairs_list if tuple(sorted([p['a'], p['b']])) not in truth_dups],
        key=lambda q: q['sum_score'])
    nondup_rows = ''
    for p in nonconf_sorted[:10]:
        wl_cells = ''
        for wl in WL_ORDER:
            sc = p['score'][wl]
            if sc is not None:
                color = '#2d8f48' if sc < _SCORE_GATE else '#c0392b'
                wl_cells += f'<td class="center" style="color:{color};font-weight:600">{sc:.4f}</td>'
            else:
                wl_cells += '<td class="center na">---</td>'
        pd_val = p['p_dup']
        pd_color = '#2d8f48' if pd_val > 0.9 else ('#b97000' if pd_val > 0.1 else '#888')
        r_min = p.get('r_min')
        r_cell = ('<td class="center na">—</td>' if r_min is None else
                  f'<td class="center" style="color:{_shape_color(r_min)};font-weight:600">{r_min:.4f}</td>')
        nondup_rows += (f'<tr><td class="pair-cell">{p["a"]} ↔ {p["b"]}</td>'
                        f'{wl_cells}'
                        f'<td class="center">{p["sum_score"]:.3f}</td>'
                        f'<td class="center" style="color:{pd_color};font-weight:600">{pd_val*100:.2f}%</td>'
                        f'{r_cell}</tr>')

    n_over_50 = int((p_dup_arr > 0.5).sum())
    n_over_99 = int((p_dup_arr > 0.99).sum())
    if truth_dups:
        verdict_block = (
            '<div class="verdict-box verdict-confirm">'
            f'<b>{len(truth_dups)} / {len(truth_dups)} duplicate pairs identified. 0 false positives.</b><br>'
            'Every true duplicate sits below the decision threshold at all three wavelengths; '
            'every non-duplicate sits above it at one or more wavelengths.'
            '</div>'
        ) if tp == len(truth_dups) and fp == 0 else (
            f'<div class="verdict-box verdict-dispute">'
            f'<b>{tp}/{len(truth_dups)} TP, {fp} FP, {fn} FN.</b></div>'
        )
    else:
        verdict_block = (
            f'<div class="verdict-box verdict-confirm">'
            f'<b>{n_over_50} duplicate pair(s) identified at ≥50% likelihood; '
            f'{n_over_99} at ≥99% likelihood</b> across {len(all_pairs_list)} pairs.'
            f'</div>'
            if n_over_50 else
            f'<div class="verdict-box verdict-dispute">'
            f'<b>No duplicate pairs identified at ≥50% likelihood</b> '
            f'({len(all_pairs_list)} pairs).</div>'
        )

    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>{_BASE_CSS}</style></head><body>
{_embed_logo()}
<h1>{title}</h1>
<div class="subtitle">{len(files)} files &bull; {len(all_pairs_list)} pairs &bull; generated {generated}</div>

{verdict_block}

<div class="cards">
  <div class="card"><div class="card-label">Files</div><div class="card-value">{len(files)}</div></div>
  <div class="card"><div class="card-label">Pairs</div><div class="card-value">{len(all_pairs_list)}</div></div>
  <div class="card"><div class="card-label">Likelihood &gt; 99%</div>
    <div class="card-value good">{int((p_dup_arr>0.99).sum())}</div></div>
  <div class="card"><div class="card-label">Likelihood &gt; 50%</div>
    <div class="card-value">{int((p_dup_arr>0.5).sum())}</div></div>
  <div class="card"><div class="card-label">Likelihood &gt; 10%</div>
    <div class="card-value">{int((p_dup_arr>0.1).sum())}</div></div>
</div>

<div class="dir-banner">Distribution — duplicates vs non-duplicates</div>
<img src="data:image/png;base64,{distribution_chart}" class="chart-img" />

<div class="dir-banner">Histogram — combined 3λ match score</div>
<img src="data:image/png;base64,{histogram_chart}" class="chart-img" />

<div class="dir-banner">All {len(files)} files — per-file verdict</div>
<table class="vote-table">
<tr><th style="text-align:left">File</th><th>Acquisition time</th>
  <th>score @ 1310</th><th>score @ 1550</th><th>score @ 1625</th>
  <th>combined score</th><th>Duplicate likelihood</th>
  <th>shape r (min λ)</th><th>Verdict</th></tr>
{file_rows}
</table>

{dup_detail_block}

<div class="dir-banner">Closest non-duplicate pairs</div>
<table class="vote-table">
<tr><th style="text-align:left">Pair</th>
  <th>score @ 1310</th><th>score @ 1550</th><th>score @ 1625</th><th>combined</th>
  <th>Duplicate likelihood</th><th>shape r (min λ)</th></tr>
{nondup_rows}
</table>

</body></html>'''

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return out_path


def html_to_pdf_bytes(html_str, base_url=None):
    """Render an HTML string to PDF bytes. WeasyPrint preferred (cloud-friendly);
    Chrome used as a fallback when WeasyPrint's native libs aren't installed."""
    try:
        from weasyprint import HTML
        return HTML(string=html_str, base_url=base_url).write_pdf()
    except Exception:
        pass
    chrome = _find_chrome()
    if not chrome:
        raise RuntimeError(
            'Neither WeasyPrint nor Chrome is available. '
            'Install WeasyPrint system libs (brew install pango) '
            'or Google Chrome.')
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False,
                                     encoding='utf-8') as hf:
        hf.write(html_str)
        html_path = hf.name
    pdf_path = html_path.replace('.html', '.pdf')
    try:
        res = subprocess.run(
            [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
             '--run-all-compositor-stages-before-draw',
             '--virtual-time-budget=5000',
             f'--print-to-pdf={pdf_path}',
             '--print-to-pdf-no-header', '--no-pdf-header-footer',
             'file://' + html_path],
            capture_output=True, timeout=180)
        if res.returncode != 0:
            raise RuntimeError(f'Chrome failed: {res.stderr.decode(errors="ignore")[:400]}')
        with open(pdf_path, 'rb') as fh:
            return fh.read()
    finally:
        for p in (html_path, pdf_path):
            try:
                os.remove(p)
            except OSError:
                pass


def html_to_pdf(html_path, pdf_path):
    """File-to-file wrapper. Returns True on success."""
    with open(html_path, 'r', encoding='utf-8') as fh:
        html = fh.read()
    base = os.path.dirname(os.path.abspath(html_path))
    try:
        pdf_bytes = html_to_pdf_bytes(html, base_url=base)
    except Exception:
        return False
    with open(pdf_path, 'wb') as fh:
        fh.write(pdf_bytes)
    return True


def build_json_html(folder, title='Duplicate Classification Report', truth_dups=None):
    paths = sorted(glob.glob(os.path.join(folder, '*.json')))
    if not paths:
        raise RuntimeError(f'No JSON files found in {folder}')
    files = [load_file(p) for p in paths]
    all_pairs = []
    for a, b in combinations(files, 2):
        sc = {wl: _score(a, b, wl) for wl in WL_ORDER}
        rs = {wl: _shape_r(a, b, wl) for wl in WL_ORDER}
        sum_sc = sum(v for v in sc.values() if v is not None)
        rs_vals = [v for v in rs.values() if v is not None]
        r_min = min(rs_vals) if rs_vals else None
        is_dup = tuple(sorted([a['name'], b['name']])) in (truth_dups or set())
        all_pairs.append({'a': a['name'], 'b': b['name'],
                          'score': sc, 'sum_score': sum_sc, 'is_dup': is_dup,
                          'shape_r': rs, 'r_min': r_min})
    out_html_tmp = os.path.join(folder, '_tmp_report.html')
    build_report(files, all_pairs, truth_dups or set(), out_html_tmp, title=title)
    with open(out_html_tmp, 'r', encoding='utf-8') as fh:
        html = fh.read()
    try:
        os.remove(out_html_tmp)
    except OSError:
        pass
    return html, files, all_pairs


def run_json_bytes(folder, title='Duplicate Classification Report', truth_dups=None):
    html, files, pairs = build_json_html(folder, title=title, truth_dups=truth_dups)
    return html_to_pdf_bytes(html, base_url=folder), len(files), len(pairs)


def build_trc_html(folder, title='Duplicate Classification Report', truth_dups=None):
    """TRC-mode equivalent of build_json_html. Loads .trc files via the TRC
    parser and reuses the JSON-mode renderer (same multi-wavelength layout)."""
    global WL_ORDER
    paths = sorted(glob.glob(os.path.join(folder, '*.trc')))
    if not paths:
        raise RuntimeError(f'No TRC files found in {folder}')
    files = [load_trc_file(p) for p in paths]
    # Use whichever wavelengths the TRC files actually carry — fall back to
    # the production set if everything matches it.
    common = set(files[0]['wl'].keys())
    for f in files[1:]:
        common &= set(f['wl'].keys())
    wl_list = sorted(common) or WL_ORDER
    all_pairs = []
    for a, b in combinations(files, 2):
        sc = {wl: _score(a, b, wl) for wl in wl_list}
        rs = {wl: _shape_r(a, b, wl) for wl in wl_list}
        sum_sc = sum(v for v in sc.values() if v is not None)
        rs_vals = [v for v in rs.values() if v is not None]
        r_min = min(rs_vals) if rs_vals else None
        is_dup = tuple(sorted([a['name'], b['name']])) in (truth_dups or set())
        all_pairs.append({'a': a['name'], 'b': b['name'],
                          'score': sc, 'sum_score': sum_sc, 'is_dup': is_dup,
                          'shape_r': rs, 'r_min': r_min})
    # Override module-level WL_ORDER for rendering when TRC carries fewer/other λ
    saved = WL_ORDER
    WL_ORDER = wl_list
    out_html_tmp = os.path.join(folder, '_tmp_report.html')
    try:
        build_report(files, all_pairs, truth_dups or set(), out_html_tmp, title=title)
        with open(out_html_tmp, 'r', encoding='utf-8') as fh:
            html = fh.read()
    finally:
        WL_ORDER = saved
        try:
            os.remove(out_html_tmp)
        except OSError:
            pass
    return html, files, all_pairs


def run_trc_bytes(folder, title='Duplicate Classification Report', truth_dups=None):
    html, files, pairs = build_trc_html(folder, title=title, truth_dups=truth_dups)
    return html_to_pdf_bytes(html, base_url=folder), len(files), len(pairs)


def run_json(folder, out_pdf, title='Duplicate Classification Report', truth_dups=None):
    pdf_bytes, _, _ = run_json_bytes(folder, title=title, truth_dups=truth_dups)
    with open(out_pdf, 'wb') as fh:
        fh.write(pdf_bytes)
    return out_pdf


def main():
    TRUTH_DUPS = {
        tuple(sorted(['VERSLK001','VERSLK013'])), tuple(sorted(['VERSLK002','VERSLK014'])),
        tuple(sorted(['VERSLK003','VERSLK015'])), tuple(sorted(['VERSLK010','VERSLK016'])),
        tuple(sorted(['VERSLK011','VERSLK017'])), tuple(sorted(['VERSLK012','VERSLK018'])),
    }

    paths = sorted(glob.glob(os.path.join(JSON_FOLDER, '*.json')))
    files = [load_file(p) for p in paths]
    print(f'Loaded {len(files)} files')

    all_pairs = []
    for a, b in combinations(files, 2):
        sc = {wl: _score(a, b, wl) for wl in WL_ORDER}
        sum_sc = sum(v for v in sc.values() if v is not None)
        is_dup = tuple(sorted([a['name'], b['name']])) in TRUTH_DUPS
        all_pairs.append({'a': a['name'], 'b': b['name'],
                          'score': sc, 'sum_score': sum_sc, 'is_dup': is_dup})

    out_html = os.path.join(HERE, 'report.html')
    build_report(files, all_pairs, TRUTH_DUPS, out_html)
    print(f'Report: {out_html}')

    pdf = out_html.replace('.html', '.pdf')
    chrome = _find_chrome()
    if chrome:
        result = subprocess.run(
            [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
             '--run-all-compositor-stages-before-draw',
             '--virtual-time-budget=5000',
             f'--print-to-pdf={os.path.abspath(pdf)}',
             '--print-to-pdf-no-header', '--no-pdf-header-footer',
             'file://' + os.path.abspath(out_html)],
            capture_output=True, timeout=180)
        if result.returncode == 0:
            print(f'   PDF: {pdf}')


if __name__ == '__main__':
    main()
