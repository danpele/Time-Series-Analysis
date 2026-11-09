"""Export the inputs of the Chapter 11 interactive page (TSA/interactive/ch11_data.js).

No foundation model is run here. The page uses what the chapter's code already produced:
  * ch11_backtest.csv and ch11_abs_err.json (the rolling-origin backtest of the seven series: MASE, WQL parts, coverage of
    the 80% interval and the absolute error of the median at every step, for every origin and model) and ch11_numbers.json;
  * the zero-shot quantiles of Chronos-Bolt small in the three fan charts, read from the vector charts of the slides
    (charts/tsa_ch11_fan_load.pdf, tsa_ch11_fan_ip.pdf, tsa_ch11_fan_eurron.pdf): each band is a fill_between polygon and
    each median a line with one vertex per forecast step, so every point is drawn; the y axis is mapped from the tick marks.
    The quantile levels 0.1, 0.3, 0.5, 0.7 and 0.9 are drawn (the 80% and 50% bands and the median); the PDF writes
    coordinates with 6 decimals, so the numbers of the slides are reproduced to a relative 5e-6 (checked below).
Series, read with the chapter's loaders (get_series) on 4 November 2026 and cut at the last period of the slides:
  * hourly load (Ch_04 extract), HICP inflation, unemployment (Eurostat) and EUR/RON (BNR): unrevised; they reproduce the
    numbers of ch11_numbers.json exactly (the series lengths, the worked examples, the seasonal naive, ETS and random-walk
    forecasts of the fan charts, every seasonal naive row of the backtest);
  * Romanian industrial production (Eurostat sts_inpr_m): July 2026 has been revised since the slides (today's value and
    the slide value are exported); the 96 months of the token chart and the 60 months of the fan chart are read from their
    vector charts (every month is a vertex), which give the slide vintage; the months up to July 2025 (the context of the
    fan chart, used for ETS and for the MASE scale) are unrevised, as the backtest rows show;
  * Romanian real GDP (Eurostat namq_10_gdp) has been revised since the slides: the series chart of the page uses the line of
    charts/tsa_ch11_series.pdf (1995 is not drawn; every quarter 2005Q1-2026Q2 is a vertex), not today's data;
  * US retail sales (FRED RSXFSN): July and August 2026 have been revised; the series chart uses today's data (the line of
    the slide chart is simplified and lacks one month) and the page states today's and the slide's values of those months.
The checks stop the export on any drift. Nothing of the chapter is written: st.save_fig is replaced by plt.close and the
backtest is never re-run (no CSV or chart of the chapter is touched).
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_11/export_interactive.py   (reads Eurostat, FRED and the BNR; about a minute)
"""
import json
import os
import re
import sys
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
g.st.save_fig = lambda name, *a, **k: plt.close('all')     # never overwrite the charts of the slides
ROOT = os.path.join(HERE, '..', '..')
CHARTS = os.path.join(ROOT, 'charts')
OUT = os.path.join(ROOT, 'interactive', 'ch11_data.js')
N = json.load(open(os.path.join(HERE, 'ch11_numbers.json')))
DOWNLOADED = '2026-11-04'
TOL, TOLPDF = 1e-9, 5e-6
END = {k: v['end'] for k, v in N['series'].items()}        # last period of the slides


def rd(x, d=6):
    a = np.round(np.asarray(x, float), d)
    if a.ndim == 2:
        return [rd(r, d) for r in a]
    return [None if not np.isfinite(v) else float(v) for v in np.ravel(a)] if a.ndim else float(a)


def sig(x, k=7):
    """Round to k significant digits (the errors and scores of the backtest)."""
    return [float(f'{v:.{k}g}') for v in np.ravel(np.asarray(x, float))]


def close(a, b, tol, what):
    a, b = float(a), float(b)
    if not (abs(a - b) <= tol * max(1.0, abs(b)) or (np.isnan(a) and np.isnan(b))):
        raise SystemExit(f'{what}: {a} differs from the lecture value {b}')


def cmp(a, b, tol, path, skip=()):
    if isinstance(b, dict):
        for k, v in b.items():
            if k not in skip:
                cmp(a[k], v, tol, f'{path}/{k}')
    elif isinstance(b, list):
        if len(a) != len(b):
            raise SystemExit(f'{path}: length {len(a)} != {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, tol, f'{path}[{i}]')
    elif isinstance(b, (bool, str)) or b is None:
        if a != b:
            raise SystemExit(f'{path}: {a} != {b}')
    else:
        close(a, b, tol, path)


iso = lambda d: str(pd.Timestamp(d).date())   # noqa: E731
J = lambda x: json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))   # as written to the JSON  # noqa: E731


# ---------------------------------------------------------------- series, cut at the last period of the slides
_get = g.get_series
TODAY = {}


def get_cut(key):
    if key not in TODAY:
        TODAY[key] = _get(key)
    s = TODAY[key]
    return s.loc[:pd.Timestamp(END[key]) + pd.Timedelta(hours=23, minutes=59)].copy()


g.get_series = get_cut


def today_of(key):
    """Today's series, not cut."""
    get_cut(key)
    return TODAY[key]


# ---------------------------------------------------------------- vector charts of the slides
NUM = re.compile(r'-?\d+\.?\d*$')


def pdf_paths(name):
    """Paths of a matplotlib PDF in drawing order: dict(op, stroke, fill, pts), and the text labels with their positions."""
    import pymupdf
    s = pymupdf.open(os.path.join(CHARTS, name + '.pdf'))[0].read_contents().decode('latin1')
    toks = re.findall(r'\[[^\]]*\]|\S+', s)
    st_, paths, texts, cur, RG, rg, tm = [], [], [], [], None, None, None
    for t in toks:
        if NUM.match(t):
            st_.append(float(t))
            continue
        if t in ('m', 'l'):
            cur.append((st_[-2], st_[-1]))
        elif t == 'RG':
            RG = tuple(st_[-3:])
        elif t == 'rg':
            rg = tuple(st_[-3:])
        elif t == 'cm':
            tm = (st_[-2], st_[-1])
        elif t.startswith('['):
            m = re.match(r'\[ \((.*)\) \]', t)
            if m and tm:
                texts.append((tm[0], tm[1], m.group(1)))
        elif t in ('f', 'S', 'B', 'n', 'f*'):
            if cur:
                paths.append(dict(op=t, stroke=RG, fill=rg, pts=cur))
            cur = []
        st_ = [] if t not in ('m', 'l') else st_[:-2]
    return paths, texts


def ymap(paths, texts, left, labels=None):
    """Linear map of the PDF y coordinate to data for the axes whose y axis is at x = left (from its tick marks)."""
    tk = sorted(p['pts'][0][1] for p in paths if p['op'] == 'B' and len(p['pts']) == 2 and p['pts'][0][1] == p['pts'][1][1]
                and abs(p['pts'][0][0] - p['pts'][1][0] - 3.5) < 1e-6 and abs(p['pts'][0][0] - left) < 1e-4)
    lab = labels or sorted([t for t in texts if t[0] < left - 3 and NUM.match(t[2].replace('−', '-'))], key=lambda t: t[1])
    vals = [float(t[2].replace('−', '-')) if isinstance(t, tuple) else t for t in lab]
    assert len(tk) == len(vals), (len(tk), len(vals))
    b, a = np.polyfit(vals, tk, 1)
    assert np.max(np.abs(a + b * np.array(vals) - np.array(tk))) < 2e-6           # equally spaced to the printed precision
    return lambda y: (np.asarray(y, float) - a) / b


def fill_bounds(pts, n):
    """Lower and upper edges of a fill_between polygon of n points: start (x0, hi0), lo forward, then hi backward."""
    lo = pts[1:n + 1]
    xs = [x for x, _ in lo]
    hi = {}
    for x, y in [pts[0]] + pts[n + 1:]:
        hi.setdefault(x, y)
    assert len(set(xs)) == n and all(x in hi for x in xs) and np.all(np.diff(xs) > 0)
    return np.array([y for _, y in lo]), np.array([hi[x] for x in xs])


def fan_from_pdf(name, col, n):
    """The quantiles 0.1, 0.3, 0.5, 0.7, 0.9 of the model drawn in colour col (two fills and the median line of n points)."""
    P, T = pdf_paths(name)
    left = min(p['pts'][0][0] for p in P if p['op'] == 'B' and len(p['pts']) == 2 and p['pts'][0][1] == p['pts'][1][1])
    f = ymap(P, T, left)
    fills = [p for p in P if p['op'] == 'f' and p['stroke'] == col and len(p['pts']) == 2 * n + 2]
    assert len(fills) == 2, len(fills)
    lo80, hi80 = fill_bounds(fills[0]['pts'], n)
    lo50, hi50 = fill_bounds(fills[1]['pts'], n)
    med = [p for p in P if p['op'] == 'S' and p['stroke'] == col and len(p['pts']) == n]
    assert len(med) == 1
    q = np.vstack([f(lo80), f(lo50), f([y for _, y in med[0]['pts']]), f(hi50), f(hi80)]).T
    assert np.all(np.diff(q, axis=1) >= 0)
    return q, P, f


def pdf_line(P, f, col, n, k=0):
    """The k-th line of n vertices drawn in colour col."""
    ls = [p for p in P if p['op'] == 'S' and p['stroke'] == col and len(p['pts']) == n]
    return f([y for _, y in ls[k]['pts']])


DARK, BLUE = (0.1215686275, 0.1647058824, 0.2666666667), (0.1019607843, 0.2274509804, 0.431372549)


# ---------------------------------------------------------------- 1. ideas: worked examples, tokens, patching, scores
def ideas_block():
    cmp(J(g.worked_examples()), N['ex'], TOL, 'ex')
    cmp(J(g.fig_patching()), N['patching'], TOL, 'patching')
    cmp(J(g.fig_scores()), N['scores'], TOL, 'scores')
    # the 96 months of the token chart (slide vintage of industrial production), from the line of tsa_ch11_tokens.pdf
    P, T = pdf_paths('tsa_ch11_tokens')
    lefts = sorted({round(p['pts'][0][0], 6) for p in P if p['op'] == 'B' and len(p['pts']) == 2 and p['pts'][0][1] == p['pts'][1][1]})
    f = ymap(P, T, lefts[0], [60, 70, 80, 90, 100, 110])
    x = pdf_line(P, f, BLUE, 96) if any(p['stroke'] == BLUE and len(p['pts']) == 96 for p in P) else None
    if x is None:                                  # the first line of the left axes (its colour is set before the clip)
        x = f([y for _, y in [p for p in P if p['op'] == 'S' and len(p['pts']) == 96][0]['pts']])
    s = float(np.mean(np.abs(x)))
    z = x / s
    grid = np.linspace(-3, 3, N['tokens']['bins'])
    tok = [int(np.argmin(np.abs(grid - v))) for v in z]
    cmp(dict(s=s, zmin=float(z.min()), zmax=float(z.max())), {k: N['tokens'][k] for k in ('s', 'zmin', 'zmax')}, TOLPDF, 'tokens')
    if len(set(tok)) != N['tokens']['ntok']:
        raise SystemExit('tokens: number of tokens')
    today = today_of('ip').loc[N['tokens']['start']:N['tokens']['end']].values
    diff = np.flatnonzero(np.abs(today - x) > 1e-4)
    if list(diff) != [95]:
        raise SystemExit(f'tokens: industrial production differs from today in months {diff}')
    return dict(m0=N['tokens']['start'], y=rd(x, 6), rev=dict(month=N['tokens']['end'], slide=float(x[-1]), today=float(today[-1])))


# ---------------------------------------------------------------- 2. the seven series
def series_block():
    cmp(J(g.fig_series()), N['series'], TOL, 'series')
    out = {}
    for k in g.EVAL_SERIES:
        s = get_cut(k)
        if k == 'load':
            s = s.loc['2026-06-01':'2026-06-30']
            out[k] = dict(t0=str(s.index[0]), y=rd(s.values, 5))
        elif k == 'eurron':
            s = s.loc['2005':]
            out[k] = dict(d=[iso(d) for d in s.index], y=rd(s.values, 6))
        elif k != 'gdp':
            s = s.loc['2005':]
            out[k] = dict(p0=iso(s.index[0]), y=rd(s.values, 6))
    # Romanian GDP of the slides: the line of tsa_ch11_series.pdf (Forest), 86 quarters 2005Q1-2026Q2, every one a vertex
    P, T = pdf_paths('tsa_ch11_series')
    line = [p for p in P if p['op'] == 'S' and p['stroke'] == (0.1803921569, 0.4901960784, 0.1960784314) and len(p['pts']) > 50]
    assert len(line) == 1 and len(line[0]['pts']) == 86 and np.all(np.diff([x for x, _ in line[0]['pts']]) > 0)
    left = min(p['pts'][0][0] for p in P if p['op'] == 'B' and len(p['pts']) == 2 and p['pts'][0][1] == p['pts'][1][1])
    ys = [y for _, y in line[0]['pts']]
    lab = [t for t in T if t[0] < left - 3 and t[2] in ('20', '40', '60') and min(ys) - 40 < t[1] < max(ys) + 40]
    tk = [p for p in P if p['op'] == 'B' and len(p['pts']) == 2 and p['pts'][0][1] == p['pts'][1][1] and abs(p['pts'][0][0] - left) < 1e-4
          and any(abs(p['pts'][0][1] - t[1] - 4.1) < 1 for t in lab)]
    sub = [dict(op='B', pts=p['pts']) for p in tk]
    f = ymap(sub, T, left, sorted([float(t[2]) for t in lab]))
    gdp = f(ys)
    today = today_of('gdp').loc['2005':END['gdp']]
    out['gdp'] = dict(p0='2005-01-01', y=rd(gdp, 6), maxrev=float(np.max(np.abs(today.values - gdp))))
    # revised months of US retail sales (the drawn line of the slide is simplified: one month is missing)
    r = get_cut('retail')
    rv = today_of('retail')
    out['retail']['today_last'] = [float(r.loc['2026-07-01']), float(r.loc['2026-08-01'])]
    out['m'] = {k: v[1] for k, v in g.EVAL_SERIES.items()}
    return out, float(rv.loc['2026-07-01'])


def retail_slide_values():
    """Values of July and August 2026 on the slide: the last vertices of the retail line of tsa_ch11_series.pdf."""
    P, T = pdf_paths('tsa_ch11_series')
    line = [p for p in P if p['op'] == 'S' and p['stroke'] == (0.7098039216, 0.5215686275, 0.2470588235) and len(p['pts']) > 100][0]['pts']
    tk = sorted(p['pts'][0][1] for p in P if p['op'] == 'B' and len(p['pts']) == 2 and p['pts'][0][1] == p['pts'][1][1]
                and abs(p['pts'][0][0] - line[0][0] + 15.859) < 2 and p['pts'][0][1] < 120)
    b, a = np.polyfit([400, 600], tk[:2], 1)
    xt = sorted(p['pts'][0][0] for p in P if p['op'] == 'B' and len(p['pts']) == 2 and p['pts'][0][0] == p['pts'][1][0] and abs(p['pts'][0][1] - 24.8406) < 1e-3)[:6]
    days = [(pd.Timestamp(f'{y}-01-01') - pd.Timestamp('1970-01-01')).days for y in (2004, 2008, 2012, 2016, 2020, 2024)]
    bx, ax_ = np.polyfit(days, xt, 1)
    res = {}
    for x, y in line[-3:]:
        d = pd.Timestamp('1970-01-01') + pd.Timedelta(days=round((x - ax_) / bx))
        res[iso(d.to_period('M').to_timestamp() if d.day < 15 else (d + pd.Timedelta(days=20)).to_period('M').to_timestamp())] = (y - a) / b
    return [float(res['2026-07-01']), float(res['2026-08-01'])]


# ---------------------------------------------------------------- 3. the three fan charts
def fan_load_block():
    F = N['fan_load']
    s = get_cut('load')
    p = s.index.get_loc(pd.Timestamp(F['origin']))
    ctx, y, h = s.values[p - 2048:p], s.values[p:p + 48], 48
    Qn = g.snaive_q(ctx, h, 168)
    close(g.mase(y, Qn[:, 4], ctx, 168), F['mase_naive'], TOL, 'fan_load mase_naive')
    q, P, f = fan_from_pdf('tsa_ch11_fan_load', BLUE, h)
    close(f(P[[i for i, x in enumerate(P) if x['op'] == 'S' and x['stroke'] == DARK and len(x['pts']) == 48][0]]['pts'][23][1]), F['y_24'], TOLPDF, 'fan_load y_24 (chart)')
    cmp(dict(q_24=list(q[23, [0, 2, 4]]), y_24=float(y[23]), mase_bolt=g.mase(y, q[:, 2], ctx, 168),
             cov_bolt=float(np.mean((y >= q[:, 0]) & (y <= q[:, 4]))),
             pin_24={str(t): float(g.pinball(y[23], q[23, j], t)) for j, t in ((0, 0.1), (2, 0.5), (4, 0.9))}),
        {k: F[k] for k in ('q_24', 'y_24', 'mase_bolt', 'cov_bolt', 'pin_24')}, TOLPDF, 'fan_load')
    if str(s.index[p + 23]) != F['t_24']:
        raise SystemExit('fan_load t_24')
    return dict(t0=str(s.index[p - 2048]), ctx=rd(ctx, 5), y=rd(y, 5), naive=rd(Qn[:, [0, 4, 8]].T, 6), bolt=rd(q.T, 6),
                scale={str(m): float(np.mean(np.abs(ctx[m:] - ctx[:-m]))) for m in (1, 24, 168)})


def fan_ip_block():
    F = N['fan_ip']
    s = get_cut('ip')
    p = len(s) - 12
    ctx = s.values[:p]
    Qe = g.ets_q(ctx, 12, 12)
    q, P, f = fan_from_pdf('tsa_ch11_fan_ip', BLUE, 12)
    hist, y = pdf_line(P, f, DARK, 48), pdf_line(P, f, DARK, 12)
    ets_med = pdf_line(P, f, (0.1803921569, 0.4901960784, 0.1960784314), 12)
    if np.max(np.abs(hist - ctx[-48:])) > 1e-4 or np.max(np.abs(ets_med - Qe[:, 4])) > 1e-4:
        raise SystemExit('fan_ip: the chart differs from the context or from ETS')
    close(g.mase(y, Qe[:, 4], ctx, 12), F['mase_ets'], TOLPDF, 'fan_ip mase_ets')
    close(np.mean((y >= Qe[:, 0]) & (y <= Qe[:, 8])), F['cov_ets'], TOL, 'fan_ip cov_ets')
    close(g.mase(y, q[:, 2], ctx, 12), F['mase_bolt'], TOLPDF, 'fan_ip mase_bolt')
    close(np.mean((y >= q[:, 0]) & (y <= q[:, 4])), F['cov_bolt'], TOL, 'fan_ip cov_bolt')
    close(g.mase(y, g.snaive_q(ctx, 12, 12)[:, 4], ctx, 12), F['mase_naive'], TOLPDF, 'fan_ip mase_naive')
    if iso(s.index[p]) != F['origin'] or iso(s.index[-1]) != F['end']:
        raise SystemExit('fan_ip dates')
    today = today_of('ip').loc[s.index[p]:END['ip']].values
    return dict(m0=iso(s.index[p - 48]), hist=rd(hist, 6), y=rd(y, 6), bolt=rd(q.T, 6), ets=rd(Qe.T, 6), naive=rd(ctx[-12:], 6),
                scale=float(np.mean(np.abs(ctx[12:] - ctx[:-12]))), today=rd(today, 6))


def fan_eurron_block():
    F = N['fan_eurron']
    s = get_cut('eurron')
    p = len(s) - 20
    ctx, y = s.values[max(0, p - 2048):p], s.values[p:]
    k = np.arange(1, 21)
    sd = float(np.std(np.diff(ctx)))
    Qn = g.normal_q(np.repeat(ctx[-1], 20), sd * np.sqrt(k))
    q, P, f = fan_from_pdf('tsa_ch11_fan_eurron', BLUE, 20)
    cmp(dict(last=float(ctx[-1]), width_rw=float(Qn[-1, 8] - Qn[-1, 0]), mae_rw=float(np.mean(np.abs(y - ctx[-1])))),
        {k_: F[k_] for k_ in ('last', 'width_rw', 'mae_rw')}, TOL, 'fan_eurron')
    cmp(dict(med_bolt=float(q[-1, 2]), width_bolt=float(q[-1, 4] - q[-1, 0]), mae_bolt=float(np.mean(np.abs(y - q[:, 2])))),
        {k_: F[k_] for k_ in ('med_bolt', 'width_bolt', 'mae_bolt')}, TOLPDF, 'fan_eurron (chart)')
    if iso(s.index[p]) != F['origin'] or iso(s.index[-1]) != F['end']:
        raise SystemExit('fan_eurron dates')
    hist = s.iloc[p - 120:p]
    return dict(d=[iso(d) for d in s.index[p - 120:]], hist=rd(hist.values, 6), y=rd(y, 6), bolt=rd(q.T, 6), sd=sd, last=float(ctx[-1]),
                cov_rw=float(np.mean((y >= Qn[:, 0]) & (y <= Qn[:, 8]))))


# ---------------------------------------------------------------- 4. the backtest (saved results of the chapter)
def bench_block():
    df = g.load_results()
    cmp(J(g.table_results(df)), N['bench'], TOL, 'bench')
    cmp(J(g.fig_coverage(df)), N['coverage'], TOL, 'coverage')
    cmp(J(g.fig_horizon(df)), N['horizon'], TOL, 'horizon')
    cmp(J(g.fig_prepost(df)), N['prepost'], TOL, 'prepost')
    out = {}
    for k in g.EVAL_SERIES:
        d = df[df.series == k]
        org = sorted(d.origin.unique())
        o = dict(origins=[str(pd.Timestamp(x))[:16] if k == 'load' else iso(x) for x in org], models={})
        for m in g.MODELS:
            e = d[d.model == m].set_index('origin').loc[org]
            o['models'][m] = dict(mase=sig(e.mase), num=sig(e.wql_num), den=sig(e.wql_den, 9), cov=sig(e.cov80, 9),
                                  err=[sig(v, 5) for v in e.abs_err])
        out[k] = o
    return dict(series=out, models=g.MODELS, release=g.RELEASE, label={k: v[0] for k, v in g.EVAL_SERIES.items()},
                h={k: v[2] for k, v in g.EVAL_SERIES.items()}, short=g.SHORT)


def main():
    d = dict(numbers=N, downloaded=DOWNLOADED)
    print('ideas', flush=True)
    d['tokens'] = ideas_block()
    print('series', flush=True)
    d['series'], _ = series_block()
    d['series']['retail']['slide_last'] = retail_slide_values()
    print('fans', flush=True)
    d['fan_load'] = fan_load_block()
    d['fan_ip'] = fan_ip_block()
    d['fan_eurron'] = fan_eurron_block()
    print('backtest', flush=True)
    d['bench'] = bench_block()
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 11 interactive page, exported by Quantlets/Ch_11/export_interactive.py.\n')
        fh.write('window.CH11_DATA = ' + json.dumps(d, separators=(',', ':'), default=float).replace('NaN', 'null') + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')
    print('revised: ip', d['tokens']['rev'], 'retail today', d['series']['retail']['today_last'], 'slide', d['series']['retail']['slide_last'],
          'gdp max revision', d['series']['gdp']['maxrev'])


if __name__ == '__main__':
    main()
