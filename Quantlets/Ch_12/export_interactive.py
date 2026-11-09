"""Export the inputs of the Chapter 12 interactive page (TSA/interactive/ch12_data.js).

Every number of the page is recomputed in the browser from the series exported here; this script first runs the chapter's
own functions (generate_all_charts.py) and checks their results against ch12_numbers.json, then exports the series they use.
  * simulations of the chapter (seed 2026): the Normal draws of the two-cycles example (T = 120), of the white-noise
    periodograms (T = 128 and T = 2048), of the leakage example (T = 256), and the simulated AR(2) of the smoothing example
    (T = 512, simulate_arma);
  * yearly sunspot numbers 1700-2008 (statsmodels; not revised): they reproduce the numbers of the lecture exactly;
  * hourly electricity load of Romania (ENTSO-E extract of Chapter 4, Quantlets/Ch_04/ch4_ro_load_hourly.csv, put on a regular
    hourly grid by the chapter's loader), 1 January 2022 - 30 June 2026: not revised, reproduces the numbers exactly;
  * daily S&P 500 log returns 2000-2026 (data/market, EODHD; not revised): reproduce the GPH numbers exactly;
  * US real GDP (FRED GDPC1) and US industrial production and unemployment (FRED INDPRO, UNRATE), read on 4 November 2026 and
    cut at 2019 Q4 / December 2019 as in the lecture: the data of that day reproduce the numbers of the lecture exactly;
  * Romanian real GDP (Eurostat namq_10_gdp) has been revised since the slides: the page uses the slide vintage, read from
    the vector chart charts/tsa_ch12_gdp_series.pdf, in which every one of the 99 quarterly growth rates (1995 Q2 - 2019 Q4)
    is a vertex of the line (and every one of the 100 quarters of the HP cycle); the HP cycle recomputed from them matches
    the drawn cycle to 3e-7 and the numbers of the lecture to 1e-7. Today's values are exported for comparison.
The checks stop the export on any drift. Nothing of the chapter is written: st.save_fig is replaced by plt.close (no CSV or
chart of the chapter is touched).
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_12/export_interactive.py   (reads FRED and Eurostat; about half a minute)
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
OUT = os.path.join(ROOT, 'interactive', 'ch12_data.js')
N = json.load(open(os.path.join(HERE, 'ch12_numbers.json')))
DOWNLOADED = '2026-11-04'
TOL, TOLPDF = 1e-9, 1e-6


def close(a, b, tol, what):
    a, b = float(a), float(b)
    if not abs(a - b) <= tol * max(1.0, abs(b)):
        raise SystemExit(f'{what}: {a} differs from the lecture value {b}')


def cmp(a, b, tol, path):
    if isinstance(b, dict):
        for k, v in b.items():
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


J = lambda x: json.loads(json.dumps(x, default=float))   # as written to the JSON  # noqa: E731


def rd(x, d):
    return [float(v) for v in np.round(np.asarray(x, float), d)]


def sg(x, k=12):
    return [float(f'{v:.{k}g}') for v in np.ravel(np.asarray(x, float))]


def same(a, b, tol, what):
    if len(a) != len(b) or np.max(np.abs(np.asarray(a) - np.asarray(b))) > tol:
        raise SystemExit(f'{what}: the exported series differs from the one of the chapter')


# ---------------------------------------------------------------- 1. the chapter's results, recomputed with today's data
def chapter_results():
    """Every function of the chapter except the Romanian GDP parts, checked against ch12_numbers.json."""
    for name, f in [('ex', g.worked_examples), ('fourier', g.fig_fourier), ('alias', g.fig_aliasing), ('spectra', g.fig_spectra),
                    ('sun', g.fig_sunspots), ('incons', g.fig_inconsistency), ('leak', g.fig_leakage),
                    ('smooth', g.fig_smoothing), ('sunci', g.fig_sunspot_ci), ('load', g.fig_load), ('lm', g.fig_long_memory),
                    ('filt', g.fig_filters), ('coh', g.fig_coherence), ('wav', g.fig_wavelet)]:
        print(name, flush=True)
        cmp(J(f()), N[name], TOL, name)
    gd = J(g.fig_gdp_series())
    cmp(gd['US'], N['gdp']['US'], TOL, 'gdp/US')
    gs = J(g.fig_gdp_spectra())
    for k in ('US.growth', 'US.HP cycle'):
        cmp(gs[k], N['gdpspec'][k], TOL, 'gdpspec/' + k)
    return gd['RO'], {k: gs[k] for k in ('RO.growth', 'RO.HP cycle')}


# ---------------------------------------------------------------- 2. simulations of the chapter (seed 2026)
def simulations():
    rng = np.random.default_rng(g.SEED)
    e120 = rng.standard_normal(120)
    t = np.arange(1, 121)
    x = 2.0 * np.cos(2 * np.pi * t / 12) + 1.0 * np.cos(2 * np.pi * t / 4 + 1.0) + 0.7 * e120
    nu, I = g.periodogram(x)
    close(I[np.argmin(abs(nu - 1 / 12))], N['fourier']['I12'], TOL, 'fourier I12 (draws)')
    rng = np.random.default_rng(g.SEED)
    w128, w2048 = rng.standard_normal(128), rng.standard_normal(2048)
    close(g.periodogram(w2048)[1][:-1].max(), N['incons']['2048']['max'], TOL, 'incons (draws)')
    rng = np.random.default_rng(g.SEED)
    e256 = rng.standard_normal(256)
    tt = np.arange(256)
    xl = np.cos(2 * np.pi * 25.5 / 256 * tt) + 0.02 * np.cos(2 * np.pi * 0.3 * tt) + 0.002 * e256
    nu, I = g.periodogram(xl)
    close(I[np.argmin(abs(nu - 0.3))], N['leak']['raw_at2'], TOL, 'leak (draws)')
    ar2 = g.simulate_arma(g.AR2, (), 512)
    nu, I = g.periodogram(ar2)
    close(nu[np.argmax(I)], N['smooth']['raw_peak'], TOL, 'smooth (draws)')
    return dict(e120=sg(e120, 15), w128=sg(w128, 15), w2048=sg(w2048, 15), e256=sg(e256, 15), ar2=sg(ar2, 15))


# ---------------------------------------------------------------- 3. Romanian GDP of the slides, from the vector chart
NUM = re.compile(r'-?\d+\.?\d*$')


def pdf_paths(name):
    """Paths of a matplotlib PDF in drawing order: dict(op, stroke, pts)."""
    import pymupdf
    s = pymupdf.open(os.path.join(CHARTS, name + '.pdf'))[0].read_contents().decode('latin1')
    st_, paths, cur, RG = [], [], [], None
    for tk in re.findall(r'\[[^\]]*\]|\S+', s):
        if NUM.match(tk):
            st_.append(float(tk))
            continue
        if tk in ('m', 'l'):
            cur.append((st_[-2], st_[-1]))
        elif tk == 'RG':
            RG = tuple(st_[-3:])
        elif tk in ('f', 'S', 'B', 'n', 'f*'):
            if cur:
                paths.append(dict(op=tk, stroke=RG, pts=cur))
            cur = []
        st_ = [] if tk not in ('m', 'l') else st_[:-2]
    return paths


RED, FOREST = (0.8039215686, 0.0, 0.0), (0.1803921569, 0.4901960784, 0.1960784314)


def ro_gdp_slide(ro_today, ro_spec_today):
    P = pdf_paths('tsa_ch12_gdp_series')
    left = max(p['pts'][0][0] for p in P if p['op'] == 'B' and len(p['pts']) == 2 and p['pts'][0][1] == p['pts'][1][1]
               and abs(p['pts'][0][0] - p['pts'][1][0] - 3.5) < 1e-6)                     # y ticks of the right (Romania) axes
    tk = sorted(p['pts'][0][1] for p in P if p['op'] == 'B' and len(p['pts']) == 2 and p['pts'][0][1] == p['pts'][1][1]
                and abs(p['pts'][0][0] - left) < 1e-4)
    vals = [-10, -5, 0, 5, 10]                                                               # tick labels of the Romania panel
    assert len(tk) == len(vals)
    b, a = np.polyfit(vals, tk, 1)
    assert np.max(np.abs(a + b * np.array(vals) - np.array(tk))) < 2e-6
    f = lambda y: (np.asarray(y, float) - a) / b                                            # noqa: E731
    grw = [p for p in P if p['op'] == 'S' and p['stroke'] == RED and len(p['pts']) == 99]
    cyc = [p for p in P if p['op'] == 'S' and p['stroke'] == FOREST and len(p['pts']) == 100]
    assert len(grw) == 1 and len(cyc) == 1 and np.all(np.diff([x for x, _ in grw[0]['pts']]) > 0)
    gr = f([y for _, y in grw[0]['pts']])
    lev = pd.Series(np.concatenate([[0.0], np.cumsum(gr)]))                                 # 100 ln GDP up to a constant
    c = g.hp_cycle(lev).values                                                              # the HP cycle ignores the constant
    if np.max(np.abs(c - f([y for _, y in cyc[0]['pts']]))) > 1e-5:
        raise SystemExit('Romanian GDP: the HP cycle of the drawn growth rates differs from the drawn cycle')
    cmp(dict(T=len(gr), mean=float(gr.mean()), sd=float(gr.std(ddof=1)), cyc_sd=float(c.std(ddof=1))),
        {k: N['gdp']['RO'][k] for k in ('T', 'mean', 'sd', 'cyc_sd')}, TOLPDF, 'gdp/RO (chart)')
    for kind, x in (('growth', gr), ('HP cycle', c)):
        nu, I = g.periodogram(x)
        band = (nu >= 1 / 32) & (nu <= 1 / 6)
        cmp(dict(share_bc=I[band].sum() / I.sum(), share_low=I[nu < 1 / 32].sum() / I.sum(), peak_period=1 / nu[np.argmax(g.daniell(I, 2))]),
            {k: N['gdpspec']['RO.' + kind][k] for k in ('share_bc', 'share_low', 'peak_period')}, TOLPDF, 'gdpspec/RO.' + kind)
    today = g.gdp_growth()['RO']
    return dict(g=sg(gr, 10), lev=sg(lev.values, 10), maxrev=float(np.max(np.abs(today.values - gr))),
                today=dict(mean=ro_today['mean'], sd=ro_today['sd'], cyc_sd=ro_today['cyc_sd'],
                           share_bc=ro_spec_today['RO.growth']['share_bc'], hp_bc=ro_spec_today['RO.HP cycle']['share_bc'],
                           hp_low=ro_spec_today['RO.HP cycle']['share_low']))


# ---------------------------------------------------------------- 4. real series
def real_series():
    s = g.sunspots()
    same(s.values, rd(s.values, 1), 1e-9, 'sunspots')
    G = g.gdp_growth()
    us_g, us_l = G['US'], G['US_level']
    if str(us_l.index[-1].date()) != '2019-10-01' or len(us_g) != N['gdp']['US']['T']:
        raise SystemExit('US GDP: sample')
    load = g.load_hourly().loc[:N['load']['end'] + ' 23:00']
    if len(load) != N['load']['T']:
        raise SystemExit('load: length')
    lv = sg(load.values, 15)
    same(load.values, lv, 1e-9, 'load')
    r = g.log_returns('sp500')
    if len(r) != N['lm']['T']:
        raise SystemExit('S&P 500: length')
    F = g.read_fred(['INDPRO', 'UNRATE']).loc['1948-01-01':'2019-12-31'].dropna()
    ip = 100 * np.log(F['INDPRO']).diff()
    du = -F['UNRATE'].diff()
    d = pd.concat([ip, du], axis=1).dropna()
    if len(d) != N['coh']['T']:
        raise SystemExit('coherence: length')
    return dict(sun=dict(y0=int(s.index[0]), y=rd(s.values, 1)),
                us=dict(q0=str(us_g.index[0].date()), g=sg(us_g.values, 12), lev=sg(us_l.values, 12)),
                load=dict(t0=str(load.index[0]), y=lv),
                sp=dict(d0=str(r.index[0].date()), d1=str(r.index[-1].date()), r=sg(r.values, 12)),
                coh=dict(m0=str(d.index[0].date()), ip=sg(d.iloc[:, 0].values, 12), du=sg(d.iloc[:, 1].values, 12)))


def main():
    ro_today, ro_spec_today = chapter_results()
    d = dict(numbers=N, downloaded=DOWNLOADED)
    print('simulations', flush=True)
    d['sim'] = simulations()
    print('Romanian GDP of the slides', flush=True)
    d['ro'] = ro_gdp_slide(ro_today, ro_spec_today)
    print('series', flush=True)
    d.update(real_series())
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 12 interactive page, exported by Quantlets/Ch_12/export_interactive.py.\n')
        fh.write('window.CH12_DATA = ' + json.dumps(d, separators=(',', ':'), default=float) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB; Romanian GDP: largest revision of a growth rate', round(d['ro']['maxrev'], 3))


if __name__ == '__main__':
    main()
