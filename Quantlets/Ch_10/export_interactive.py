"""Export the inputs of the Chapter 10 interactive page (TSA/interactive/ch10_data.js).

Uses the functions, seeds and data loaders of generate_all_charts.py, so the page starts from the same data and numbers as
the slides. Data and vintages:
  * the Nile flow 1871-1970 (statsmodels data set) and the saved weekly prices of data/market (S&P 500, BET, Euro Stoxx 50, up
    to 18 September 2026): unrevised; they reproduce every number of ch10_numbers.json exactly (relative tolerance 1e-9);
  * FRED (GDPC1, USREC, INDPRO, PAYEMS, W875RX1, CMRMTSPL), read with the chapter's loader on 4 November 2026 and cut at the
    last period of the slides (GDP: 2026Q2; the monthly series: July 2026). US GDP growth reproduces the numbers exactly. The
    coincident indicators have been revised since the slides: the parameters of the factor model (estimated on 1967-2019)
    are unchanged, but three numbers of the latest months differ (mean of the factor in expansions, the July 2026 value,
    filtered and smoothed); they are exported with today's value and the download date, next to the value of the slide;
  * Romanian real GDP (Eurostat namq_10_gdp) has been revised since the slides (by up to 2.3 points of 100 x log GDP). The
    slide vintage is read from the vector line of charts/tsa_ch10_ro_trend.pdf, which contains all 126 quarters (1995Q1-
    2026Q2); the y axis is mapped from its five tick marks. The PDF writes coordinates with 6 decimals (about 4e-7 points of
    100 x log GDP), so the numbers of the slides (UC model, HP, Hamilton filter, real-time gaps, Romanian Markov switching)
    are reproduced to a relative 5e-6 (checked below), not to 1e-9. Today's Eurostat series is not used.
Exported, with the seeds of the chapter: the Normal draws of the simulated paths; US GDP growth (AR(2) worked example and
the Markov-switching models); the Nile; the Romanian GDP of the slides with the smoothed and filtered UC components, and the
same UC model (parameters fixed) smoothed with the data up to earlier quarters (extra runs); the weekly BET and Euro Stoxx 50
returns; the common factor for 0-6 missing months of income and sales at the ragged edge (extra runs; the lecture: 2); the
smoothed probabilities of the two maxima of Hamilton's MS-AR(4); the switching-variance and switching-mean fits for other
sample ends (extra runs; the lecture: 2019Q4 and 2026Q2); the weekly S&P 500 returns with the GARCH(1,1) volatility; the 40
simulated regime paths of the memory experiment (ACF, local Whittle d, GARCH persistence of the first 15).
No chart and no CSV of the chapter is written: st.save_fig is replaced by plt.close.
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_10/export_interactive.py   (reads FRED; a few minutes)
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
import statsmodels.api as sm            # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
g.st.save_fig = lambda name, *a, **k: plt.close('all')     # never overwrite the charts of the slides
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch10_data.js')
N = json.load(open(os.path.join(HERE, 'ch10_numbers.json')))
FRED_END = {'GDPC1': '2026-04-01', 'USREC': '2026-07-01', 'INDPRO': '2026-07-01', 'PAYEMS': '2026-07-01', 'W875RX1': '2026-07-01',
            'CMRMTSPL': '2026-07-01'}                       # last period of the slides
DOWNLOADED = '2026-11-04'
TOL, TOLPDF = 1e-9, 5e-6
# numbers of the factor model that depend on revised months of the coincident indicators: (key, digits shown on the slide)
DFM_REVISED = {'mean_exp': 1, 'last': 1, 'last_filt': 1}


def rd(x, d=6):
    a = np.round(np.asarray(x, float), d)
    return [None if not np.isfinite(v) else float(v) for v in np.ravel(a)] if a.ndim else float(a)


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
dates = lambda idx: [iso(d) for d in idx]     # noqa: E731


# ---------------------------------------------------------------- FRED, cut at the last period of the slides
_read_fred = g.read_fred


def read_fred_cut(ids, start=None, end=None):
    x = _read_fred(ids, start, end)
    if isinstance(ids, str):
        return x.loc[:FRED_END[ids]]
    return x.loc[:min(FRED_END[i] for i in ids)]


g.read_fred = read_fred_cut


# ---------------------------------------------------------------- Romanian GDP of the slides, from the vector chart
def ro_gdp_slide():
    """100 x log Romanian real GDP, 1995Q1-2026Q2, read from the line of charts/tsa_ch10_ro_trend.pdf (the slide vintage)."""
    import pymupdf
    s = pymupdf.open(os.path.join(ROOT, 'charts', 'tsa_ch10_ro_trend.pdf'))[0].read_contents().decode('latin1')
    ticks = [(float(y0), float(lab)) for x0, y0, x1, y1, lab in re.findall(
        r'(\d+\.?\d*) (-?\d+\.?\d*) m\n(\d+\.?\d*) (-?\d+\.?\d*) l\n\nB\n1 w\nq\n1 0 -0 1 [\d.\-]+ [\d.\-]+ cm\nBT\n/F1 [\d.]+ Tf\n0 0 Td\n'
        r'\[ \(([0-9]+)\) \] TJ', s) if y0 == y1]
    assert len(ticks) == 5, ticks
    t = np.array(ticks)
    b, a = np.polyfit(t[:, 1], t[:, 0], 1)
    assert np.max(np.abs(a + b * t[:, 1] - t[:, 0])) < 2e-6           # the ticks are equally spaced to the printed precision
    i = re.search(r'0\.1019607843\n0\.2274509804 0\.431372549 RG[^\n]*\n', s).end()     # the GDP line (MainBlue, first path)
    pts = np.array(re.findall(r'(-?\d+\.?\d*) (-?\d+\.?\d*) [ml]', s[i:s.index('S', i)]), float)
    assert len(pts) == 126 and np.all(np.diff(pts[:, 0]) > 0), len(pts)   # every quarter is a vertex of the drawn line
    return pd.Series((pts[:, 1] - a) / b, index=pd.date_range('1995-01-01', periods=126, freq='QS'), name='ro_gdp')


RO = ro_gdp_slide()
g.ro_gdp = lambda: RO.copy()


# ---------------------------------------------------------------- 1. state space examples and the AR(2) of US GDP growth
def ss_block():
    cmp(g.fig_ss_examples(), N['sim'], TOL, 'sim')
    rng = np.random.default_rng(g.SEED)
    eps, eta, zeta = rng.standard_normal(200), rng.standard_normal(200), rng.standard_normal(200)
    return dict(eps=rd(eps, 12), eta=rd(eta, 12), zeta=rd(zeta, 12))


def arma_block():
    cmp(g.arma_ss_check(), N['arma'], TOL, 'arma')
    x = g.us_gdp_growth().loc[g.MS_SAMPLE[0]:g.MS_SAMPLE[1]]
    return dict(d=dates(x.index), x=rd(x.values, 12))


# ---------------------------------------------------------------- 2-3. the Nile
def nile_block():
    for k, f in (('nile', g.fig_nile_filter), ('sm', g.statsmodels_local_level), ('gain', g.fig_gain_ses), ('smooth', g.fig_nile_smooth),
                 ('lik', g.fig_likelihood), ('diag', g.fig_diagnostics), ('miss', g.fig_missing), ('ex', g.worked_example)):
        cmp(f(), N[k], TOL, k)
    y = g.nile()
    return dict(y0=int(y.index[0]), y=rd(y.values, 6))


# ---------------------------------------------------------------- 4. trend and cycle (slide vintage of Romanian GDP)
RO_ENDS = ['2008-07-01'] + [f'{y}-04-01' for y in range(2009, 2027)]   # data up to these quarters (extra runs; lecture: 2026-04-01)


def ro_block():
    for k, f in (('trend', g.fig_ro_trend), ('gap', g.fig_output_gap), ('rt', g.fig_realtime), ('msro', g.fig_ms_ro)):
        cmp(f(), N[k], TOLPDF, k)
    y = RO
    r = g.uc_fit(y.values)
    p = dict(zip(r.model.param_names, r.params))
    vint = {}
    for e in RO_ENDS:
        n = int((y.index <= e).sum())
        m = sm.tsa.UnobservedComponents(y.values[:n], level='smooth trend', autoregressive=2)
        s = m.smooth(r.params)
        vint[e] = dict(n=n, cs=rd(s.autoregressive.smoothed, 6), cf=rd(s.autoregressive.filtered, 6))
    assert np.allclose(vint['2026-04-01']['cs'], np.round(r.autoregressive.smoothed, 6))
    return dict(q0=iso(y.index[0]), y=rd(y.values, 8), trend=rd(r.level.smoothed, 6), cyc=rd(r.autoregressive.smoothed, 6),
                cycf=rd(r.autoregressive.filtered, 6), par={k.replace('.', '_'): float(v) for k, v in p.items()}, ends=RO_ENDS, vint=vint)


# ---------------------------------------------------------------- 5. time-varying beta and the common factor
def tvp_block():
    cmp(g.fig_tvp_beta(), N['tvp'], TOL, 'tvp')
    r = g.tvp_data()
    return dict(d=dates(r.index), y=rd(r.iloc[:, 0].values, 10), x=rd(r.iloc[:, 1].values, 10))


def dfm_block():
    res = g.fig_dfm()
    cmp(res, N['dfm'], TOL, 'dfm', skip=DFM_REVISED)
    today = {}
    for k, dgt in DFM_REVISED.items():
        if round(res[k], dgt) != round(N['dfm'][k], dgt):
            raise SystemExit(f'dfm/{k}: {res[k]} differs from the slide value {N["dfm"][k]} at the {dgt} decimal(s) shown')
        today[k] = float(res[k])
    z = g.coincident()
    mod = sm.tsa.DynamicFactor(z.loc[:'2019'], k_factors=1, factor_order=2)
    r = mod.fit(disp=False, maxiter=2000)
    sign = 1.0 if dict(zip(mod.param_names, r.params))['loading.f1.PAYEMS'] > 0 else -1.0
    fac = {}
    for k in range(7):                                       # 0-6 missing months of income and sales (lecture: 2)
        zz = z.copy()
        if k:
            zz.iloc[-k:, [2, 3]] = np.nan
        s = sm.tsa.DynamicFactor(zz, k_factors=1, factor_order=2).smooth(r.params)
        fac[str(k)] = dict(s=rd(sign * s.factors.smoothed[0], 6), f=rd(sign * s.factors.filtered[0], 6))
    close(fac['2']['s'][-1], res['last'], 1e-6, 'dfm last (k = 2)')
    rec = g.nber('MS').reindex(z.index).fillna(0).astype(int)
    return dict(m0=iso(z.index[0]), fac=fac, rec=rec.tolist(), today=today, downloaded=DOWNLOADED)


# ---------------------------------------------------------------- 6. Markov switching
SV_ENDS = ['2019-10-01', '2020-04-01', '2020-10-01', '2021-10-01', '2023-10-01', '2026-04-01']   # extra runs; lecture: 2019Q4, 2026Q2


def msus_block():
    for k, f in (('msus', g.fig_ms_us), ('msf', g.fig_ms_filtered), ('pit', g.ms_us_pitfalls)):
        cmp(f(), N[k], TOL, k)
    gr = g.us_gdp_growth()
    xa = gr.loc[g.MS_SAMPLE[0]:]
    rec = g.nber('QS').reindex(gr.index).fillna(0).astype(int)
    # Hamilton's MS-AR(4), 1951Q2-1984Q4: the two maxima of the slide (same starting values as fig_ms_us)
    x0 = gr.loc[g.HAMILTON_SAMPLE[0]:g.HAMILTON_SAMPLE[1]]
    m0 = sm.tsa.MarkovAutoregression(x0.values, k_regimes=2, order=4, switching_ar=False)
    maxima = {}
    for lab, sp in (('A', [0.75, 0.10, -0.2, 1.3, 0.7, 0.0, 0.0, 0.0, 0.0]), ('B', [0.15, 0.05, -1.1, 1.0, 0.8, 0.3, 0.15, -0.15, -0.15])):
        r0 = m0.fit(start_params=sp, disp=False)
        low = g.regime_order(r0)[0]
        cmp(g.ms_summary(r0, x0.index[4:], low), N['msus']['ham' if lab == 'A' else 'ham_b'], TOL, f'maximum {lab}')
        maxima[lab] = rd(r0.smoothed_marginal_probabilities[:, low], 9)
    # switching variance and switching mean for other sample ends
    ends = {}
    for e in SV_ENDS:
        x = gr.loc[g.MS_SAMPLE[0]:e]
        rv = g.ms_fit(sm.tsa.MarkovRegression(x.values, k_regimes=2, switching_variance=True))
        hv = int(np.argmax([rv.params[rv.model.param_names.index(f'sigma2[{k}]')] for k in range(2)]))
        rm = g.ms_fit(sm.tsa.MarkovRegression(x.values, k_regimes=2))
        low = g.regime_order(rm)[0]
        ends[e] = dict(n=int(len(x)), sv=g.ms_summary(rv, x.index, 1 - hv), sv_p=rd(rv.smoothed_marginal_probabilities[:, hv], 9),
                       mean=g.ms_summary(rm, x.index, low), mean_p=rd(rm.smoothed_marginal_probabilities[:, low], 9))
    cmp(ends['2019-10-01']['sv'], N['pit']['sv'], TOL, 'sv 2019Q4')
    cmp(ends['2019-10-01']['mean'], N['msus']['ext'], TOL, 'mean 2019Q4')
    cmp(ends['2026-04-01']['mean'], N['pit']['all'], TOL, 'mean 2026Q2')
    return dict(q0=iso(gr.index[0]), g=rd(gr.values, 12), rec=rec.tolist(), x0=iso(xa.index[0]), ham0=iso(x0.index[4]), maxima=maxima,
                ends=SV_ENDS, fits=ends)


def vol_block():
    from arch import arch_model
    cmp(g.fig_vol_regimes(), N['vol'], TOL, 'vol')
    x = g.weekly_returns('sp500')
    ga = arch_model(x.values, mean='Constant', vol='GARCH', p=1, q=1, dist='normal').fit(disp='off')
    return dict(d=dates(x.index), r=rd(x.values, 10), garch=rd(ga.conditional_volatility * np.sqrt(52), 8),
                gpar={k: float(v) for k, v in ga.params.items()})


def mem_block(reps=40, L=100):
    """fig_regimes_memory with every simulated path kept (the same draws of the same generator, in the same order)."""
    from arch import arch_model
    cmp(g.fig_regimes_memory(), N['mem'], TOL, 'mem')
    x, r = g.ms_vol('sp500')
    names = r.model.param_names
    mu = np.array([r.params[names.index(f'const[{k}]')] for k in range(2)])
    s2 = np.array([r.params[names.index(f'sigma2[{k}]')] for k in range(2)])
    P = r.regime_transition[:, :, 0]
    rng = np.random.default_rng(g.SEED)
    n = len(x)
    acf_data = sm.tsa.acf(np.abs(x.values), nlags=L)[1:]
    sims, ds, abs_ = [], [], []
    for i in range(reps):
        z, _ = g.simulate_ms(n, mu, s2, P, rng)
        sims.append(sm.tsa.acf(np.abs(z), nlags=L)[1:])
        ds.append(g.local_whittle(np.abs(z)))
        if i < 15:
            ga = arch_model(z, mean='Constant', vol='GARCH', p=1, q=1).fit(disp='off')
            abs_.append(ga.params['alpha[1]'] + ga.params['beta[1]'])
    iid = rng.permutation(x.values)
    acf_iid = sm.tsa.acf(np.abs(iid), nlags=L)[1:]
    M = N['mem']
    close(np.mean(sims, axis=0)[0], M['acf_sim_1'], TOL, 'mem acf_sim_1')
    close(np.mean(ds), M['d_sim'], TOL, 'mem d_sim')
    close(np.mean(abs_), M['garch_ab_sim'], TOL, 'mem garch_ab_sim')
    close(g.local_whittle(np.abs(iid)), M['d_iid'], TOL, 'mem d_iid')
    return dict(acf=rd(acf_data, 8), sims=[rd(s, 8) for s in sims], d=rd(ds, 8), ab=rd(abs_, 8), iid=rd(acf_iid, 8))


def main():
    d = dict(numbers=N, downloaded=DOWNLOADED)
    for k, f in [('ss', ss_block), ('arma', arma_block), ('nile', nile_block), ('ro', ro_block), ('tvp', tvp_block), ('dfm', dfm_block),
                 ('us', msus_block), ('vol', vol_block), ('mem', mem_block)]:
        print(k, flush=True)
        d[k] = f()
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 10 interactive page, exported by Quantlets/Ch_10/export_interactive.py.\n')
        fh.write('window.CH10_DATA = ' + json.dumps(d, separators=(',', ':'), default=float).replace('NaN', 'null') + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
