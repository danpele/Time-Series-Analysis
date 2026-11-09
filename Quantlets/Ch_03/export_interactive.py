"""Export the inputs of the Chapter 3 interactive page (TSA/interactive/ch3_data.js).

Uses the functions and seeds of generate_all_charts.py, so the page starts from the same data and numbers as the slides:
the shocks of every simulated chart (drawn in the same order from the same generators), the per-sample results of the
Monte Carlo charts (t-statistics of the spurious regressions, Dickey-Fuller statistics under a unit root, rejection
rates of the level-shift experiment), and the real series of the chapter: Romanian real GDP in the version of the slides,
Romanian HICP up to August 2026 (the last month of the slides), EUR/RON (BNR), BET and S&P 500 (course market data),
US real GDP (FRED) up to 2026Q2 and the Nile flow (statsmodels). Results that need statsmodels (ARIMA fits, ADF,
Zivot-Andrews) are computed here on the grid of parameters the page offers. Every series is checked against
ch3_numbers.json before it is written. No chart is written: the chapter's save() is replaced by plt.close().
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_03/export_interactive.py   (needs internet for Eurostat and FRED)
      ... export_interactive.py --tables   recomputes only the inputs of the table blocks (key 'tab') from the existing file
"""
import json
import os
import re
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from statsmodels.tsa.arima.model import ARIMA          # noqa: E402
from statsmodels.tsa.adfvalues import mackinnoncrit   # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
g.save = lambda name, save_it=True: plt.close('all')   # never overwrite the charts of the slides
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch3_data.js')
N = json.load(open(os.path.join(HERE, 'ch3_numbers.json')))


def r(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


def close(a, b, tol, what):
    assert abs(a - b) <= tol, (what, a, b)


def dd(idx):
    """Daily dates, compactly: the first date and the gaps in days (the page rebuilds the dates)."""
    return dict(d0=idx[0].strftime('%Y-%m-%d'), gap=np.diff(idx.values).astype('timedelta64[D]').astype(int).tolist())


def qd(idx):
    return [str(p) for p in idx]


# ---------------------------------------------------------------- series of the slides
def gdp_slide_vintage():
    """100 ln of Romanian real GDP (million EUR, 2010 prices), 2000Q1-2026Q2, as on the slides. Eurostat has revised the
    series since the charts were made (the last quarter by 1.0 bn EUR), so the values are read back from the vector path
    of the level panel of charts/tsa_ch3_gdp_ident.pdf: scaled by the mean quarterly growth and the last value in
    ch3_numbers.json, and rounded to the 0.1 million EUR of Eurostat."""
    import pymupdf
    doc = pymupdf.open(os.path.join(HERE, '..', '..', 'charts', 'tsa_ch3_gdp_ident.pdf'))
    s = b''.join(doc.xref_stream(x) for x in doc[0].get_contents()).decode('latin1')
    run = re.findall(r'(-?[\d.]+ -?[\d.]+ m\n(?:-?[\d.]+ -?[\d.]+ l\n){100,})', s)[0]
    u = np.array([float(l.split()[1]) for l in run.strip().split('\n')])
    c = np.diff(u).mean() / N['gdp_id']['mean']
    y = (u - u[-1]) / c + N['gdp_fc']['last']
    y = 100 * np.log(np.round(np.exp(y / 100), 1))
    idx = pd.period_range('2000Q1', periods=len(y), freq='Q')
    s = pd.Series(y, index=idx, name='gdp')
    d = s.diff().dropna()
    close(len(s), N['gdp_id']['n'], 0, 'gdp n')
    close(d.mean(), N['gdp_id']['mean'], 1e-5, 'gdp mean')
    close(d.std(), N['gdp_id']['sd'], 1e-5, 'gdp sd')
    close(d.min(), N['gdp_id']['worst1'][1], 1e-4, 'gdp min')
    return s


GDP = gdp_slide_vintage()
g.gdp_series = lambda start=g.GDP_START: GDP                 # every GDP function of the chapter now uses the slide series
_ro_gdp = g.ro_gdp
g.ro_gdp = lambda: pd.Series(np.exp(GDP.values / 100), index=GDP.index.to_timestamp(), name='gdp')
_hicp = g.ro_hicp
g.ro_hicp = lambda: _hicp().loc[:N['infl']['last'] + '-01']   # the sample of the slides ends in August 2026
_us = g.us_gdp
g.us_gdp = lambda: _us().loc[:'2026-04-01']                  # ... and US GDP in 2026Q2


def arima(y, order, trend):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ARIMA(y, order=order, trend=trend).fit()


# ---------------------------------------------------------------- 1. trends
def four_block():
    f = g.fig_four_series()
    for k in ('gdp_growth_q', 'gdp_maxdev', 'hicp_ratio', 'fx_last', 'bet_logchg', 'sp_logchg'):
        close(f[k], N['four'][k], 1e-6 * max(1, abs(N['four'][k])), k)
    h = np.log(g.ro_hicp().loc['2000-01-01':])
    fx = g.load_close('eurron')
    bet = np.log(g.load_close('bet', start=g.START))
    sp = np.log(g.load_close('sp500', start=g.START))
    return dict(gdp_q=qd(GDP.index), hicp_m=[t.strftime('%Y-%m') for t in h.index], hicp=r(h.values, 5),
                fx=dict(dd(fx.index), v=r(fx.values, 4)), bet=dict(dd(bet.index), v=r(bet.values - bet.iloc[0], 4)),
                sp=dict(dd(sp.index), v=r(sp.values - sp.iloc[0], 4)))


def tsds_block():
    g.fig_ts_ds()
    return r(np.random.default_rng(g.SEED + 1).normal(0, 1, 200), 6)


def detrend_block(T=200, R=2000):
    """The path of fig_detrend and its Monte Carlo, with the rejection rate and median R^2 also for the first T' < T
    observations of the same random walks (every truncated row is a random walk of length T')."""
    f = g.fig_detrend()
    rng = np.random.default_rng(g.SEED + 2)
    y = np.cumsum(rng.normal(size=T))
    Y = np.cumsum(rng.normal(size=(R, T)), axis=1)
    mc = {}
    for n in range(50, T + 1, 10):
        Z, t = Y[:, :n], np.arange(n)
        tc = t - t.mean()
        b = (Z - Z.mean(axis=1, keepdims=True)) @ tc / (tc @ tc)
        u = Z - Z.mean(axis=1, keepdims=True) - b[:, None] * tc[None, :]
        ts = b / np.sqrt(np.sum(u ** 2, axis=1) / (n - 2) / (tc @ tc))
        r2 = 1 - np.sum(u ** 2, axis=1) / np.sum((Z - Z.mean(axis=1, keepdims=True)) ** 2, axis=1)
        mc[str(n)] = [round(float(np.mean(np.abs(ts) > 1.96)), 4), round(float(np.median(r2)), 4)]
    close(mc[str(T)][0], f['mc_reject'], 1e-9, 'detrend mc')
    return dict(y=r(y, 5), mc=mc)


# ---------------------------------------------------------------- 2. spurious regression
def spurious_block(Ts=(25, 50, 100, 200, 500, 1000), R=2000):
    """spurious_mc of the chapter (same generator, same order of draws), keeping the t-statistic of every sample."""
    rng = np.random.default_rng(g.SEED + 3)
    out = {}
    for T in Ts:
        x = np.cumsum(rng.normal(size=(R, T)), axis=1)
        y = np.cumsum(rng.normal(size=(R, T)), axis=1)
        o = {}
        for kind, (a, b) in {'levels': (y, x), 'differences': (np.diff(y, axis=1), np.diff(x, axis=1))}.items():
            n = a.shape[1]
            xc, yc = b - b.mean(axis=1, keepdims=True), a - a.mean(axis=1, keepdims=True)
            beta = np.sum(xc * yc, axis=1) / np.sum(xc ** 2, axis=1)
            u = yc - beta[:, None] * xc
            t = beta / np.sqrt(np.sum(u ** 2, axis=1) / (n - 2) / np.sum(xc ** 2, axis=1))
            close(float(np.mean(np.abs(t) > 1.96)), N['spur_mc'][str(T)][kind]['reject'], 1e-12, 'spur')
            o[kind] = r(t, 2)
        out[str(T)] = o
    real = g.fig_spurious_real()
    close(real['levels']['t'], N['spur_real']['levels']['t'], 1e-9, 'spur real')
    h = np.log(g.ro_hicp()).rename('hicp')
    sp = np.log(g.load_close('sp500', start=g.START)).resample('MS').last().rename('sp500')
    d = pd.concat([h, sp], axis=1).dropna().loc['2005-01-01':]
    return dict(t=out, real=dict(m=[t.strftime('%Y-%m') for t in d.index], hicp=r(100 * d['hicp'].values, 4),
                                 sp=r(100 * d['sp500'].values, 4)))


# ---------------------------------------------------------------- 3. Dickey-Fuller
def dfdist_block(T=250, R=20000, lo=-8.0, hi=5.0, w=0.005):
    """fig_df_dist: counts of tau in bins of width 0.005 (the share below any multiple of 0.005 is exact) and the
    kernel density of the slide on its grid."""
    from scipy import stats
    rng = np.random.default_rng(g.SEED + 4)
    y = g.random_walks(R, T, rng)
    xx = np.linspace(-5.5, 3.5, 400)
    edges = np.round(np.arange(lo, hi + w / 2, w), 6)
    out = dict(lo=lo, w=w, x=r(xx, 4), T=T, R=R)
    for reg in ('n', 'c', 'ct'):
        tau = g.df_tau_batch(y, reg)
        close(float(np.mean(tau < -1.645)), N['dfdist'][reg]['p_below_normal'], 1e-12, 'df')
        assert tau.max() < hi, tau.max()
        cnt, _ = np.histogram(tau, bins=edges)
        out[reg] = dict(cnt=cnt.tolist(), kde=r(stats.gaussian_kde(tau)(xx), 5), below=int(np.sum(tau < lo)))
    return out


def power_extra(R=2000, phis=(0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 1.0), Ts=(50, 1000)):
    """The power experiment of fig_adf_power (same code, same seed) for two more sample sizes."""
    rng = np.random.default_rng(g.SEED + 5)
    out = {}
    for T in Ts:
        for reg in ('c', 'ct'):
            cv = mackinnoncrit(N=1, regression=reg, nobs=T)[1]
            for phi in phis:
                y = g.random_walks(R, T + 100, rng, phi=phi)[:, 100:]
                out[f'{T}_{reg}_{phi}'] = float(np.mean(g.df_tau_batch(y, reg) < cv))
    crit = {str(T): {reg: float(mackinnoncrit(N=1, regression=reg, nobs=T)[1]) for reg in ('c', 'ct')} for T in (50, 100, 250, 500, 1000)}
    return dict(power=out, crit=crit, phis=list(phis))


# ---------------------------------------------------------------- 4. KPSS
def kpss_block(T=250):
    k = g.fig_kpss_sums()
    close(k['ar']['stat'], N['kpss']['ar']['stat'], 1e-12, 'kpss')
    rng = np.random.default_rng(g.SEED + 6)
    e = rng.normal(size=T)
    w = rng.normal(size=T)
    return dict(e=r(e, 6), w=r(w, 6))


# ---------------------------------------------------------------- 5. breaks
def breaks_sim_block(T=200, tb=100, phi=0.6, R=2000):
    """fig_breaks_sim: the first path of the noise and, for level shifts delta = 0, 0.5, ..., 10 at t = 100, the ADF and
    Zivot-Andrews results on that path and the Dickey-Fuller rejection rate on the 2000 paths."""
    f = g.fig_breaks_sim()
    rng = np.random.default_rng(g.SEED + 7)
    u = g.random_walks(R, T + 100, rng, phi=phi)[:, 101:]
    cv = mackinnoncrit(N=1, regression='c', nobs=T)[1]
    grid = {}
    for k in range(21):
        delta = k / 2
        step = delta * (np.arange(T) >= tb)
        y = u + step
        tau = g.df_tau_batch(np.column_stack([y[:, :1], y]), 'c')
        x = pd.Series(y[0], index=pd.RangeIndex(T))
        a, z = g.adf_test(x, 'c'), g.za_test(x, 'c')
        grid[f'{delta:.1f}'] = dict(rej=float(np.mean(tau < cv)), adf=round(a['stat'], 4), adf_p=round(a['p'], 4), adf_k=a['lags'],
                                    za=round(z['stat'], 4), za_p=z['p'], za_b=int(z['break']), za_k=z['lags'])
    close(grid['5.0']['rej'], f['rej_break'], 1e-12, 'brk rej')
    close(grid['5.0']['za'], f['za']['stat'], 1e-4, 'brk za')
    close(grid['0.0']['rej'], f['rej_nobreak'], 1e-12, 'brk rej0')
    return dict(u=r(u[0], 6), grid=grid, crit_za=f['za']['crit5'], crit_adf=f['adf']['crit5'])


def breaks_real_block():
    b = g.fig_breaks_real()
    close(b['eurron']['za']['stat'], N['brk_real']['eurron']['za']['stat'], 1e-9, 'za fx')
    nile = g.load_statsmodels('nile')
    fx = np.log(g.load_close('eurron')).resample('MS').last() * 100
    out = {}
    for name, x in (('nile', nile), ('eurron', fx)):
        res = {}
        for reg in ('c', 't', 'ct'):
            z = g.za_test(x, reg)
            bt = pd.Timestamp(z['break'])
            res[reg] = dict(stat=round(z['stat'], 4), p=z['p'], crit5=z['crit5'], lags=z['lags'], brk=z['break'][:7],
                            m0=round(float(x[x.index < bt].mean()), 3), m1=round(float(x[x.index >= bt].mean()), 3))
        a = b[name]['adf']
        out[name] = dict(d=[t.strftime('%Y-%m') for t in x.index], v=r(x.values, 4), za=res, adf_p=a['p'], kpss=b[name]['kpss']['stat'])
    return out


# ---------------------------------------------------------------- 6. ARIMA: Romanian GDP
def gdp_block():
    i = g.fig_gdp_ident()
    close(i['lb8']['lb'], N['gdp_id']['lb8']['lb'], 1e-4, 'lb8')
    grid = {}
    for p in range(3):
        for q in range(3):
            m = arima(GDP, (p, 1, q), 't')
            e = m.resid.iloc[1:]
            grid[f'{p}{q}'] = dict(e=r(e.values, 4), aicc=round(float(m.aicc), 3), conv=bool(m.mle_retvals.get('converged', True)),
                                   sigma=round(float(np.sqrt(m.params['sigma2'])), 4))
            close(m.aicc, N['gdp_models']['main']['grid'][f'{p}{q}']['aicc'], 1e-3, 'aicc')
    dg = g.fig_gdp_diag(order=(0, 1, 0))
    close(dg['jb'], N['gdp_diag']['jb'], 1e-3, 'jb')
    return dict(q=qd(GDP.index), y=r(GDP.values, 5), models=grid)


def gdp_fc_block(H=12, first='2015Q4'):
    """fig_gdp_forecast from every origin since 2015Q4: ARIMA(0,1,0) with drift and a linear trend with AR(1) errors,
    refitted on the data up to the origin; forecasts of 100 ln GDP and their standard errors."""
    f = g.fig_gdp_forecast(order=(0, 1, 0))
    close(f['ds']['level12'], N['gdp_fc']['ds']['level12'], 1e-6, 'fc ds')
    close(f['ts']['level12'], N['gdp_fc']['ts']['level12'], 1e-5, 'fc ts')
    out = {}
    for k in range(GDP.index.get_loc(pd.Period(first, 'Q')), len(GDP)):
        y = GDP.iloc[:k + 1]
        o = {}
        for lab, order, tr in (('ds', (0, 1, 0), 't'), ('ts', (1, 0, 0), 'ct')):
            fc = arima(y, order, tr).get_forecast(H)
            o[lab] = dict(f=r(fc.predicted_mean.values, 4), se=r(fc.se_mean.values, 4))
        out[str(y.index[-1])] = o
    last = out[str(GDP.index[-1])]
    close(np.exp(last['ts']['f'][-1] / 100) / 1000, N['gdp_fc']['ts']['level12'], 1e-4, 'fc ts last')
    return out


def overdiff_block():
    o = g.fig_overdiff_gdp()
    close(o['theta'], N['overdiff']['theta'], 1e-6, 'overdiff')
    us = g.quarterly(100 * np.log(g.us_gdp()))
    out = {}
    for name, y in (('ro', GDP), ('us', us)):
        th = {}
        for d in (2, 3):
            x = y.copy()
            for _ in range(d):
                x = x.diff()
            m = arima(x.dropna(), (0, 0, 1), 'n')
            th[str(d)] = [round(float(m.params['ma.L1']), 4), round(float(m.bse['ma.L1']), 4)]
        out[name] = th
    return dict(us_q=qd(us.index), us=r(us.values, 4), theta=out)


# ---------------------------------------------------------------- 7. practice
def inflation_block(H=24):
    f = g.fig_inflation_d()
    close(f['d0']['h24'], N['infl']['d0']['h24'], 1e-9, 'infl')
    pi = g.monthly(g.ro_inflation())
    fits = {}
    for d, tr in ((0, 'c'), (1, 'n')):
        for p in range(4):
            for q in range(3):
                m = arima(pi, (p, d, q), tr)
                fc = m.get_forecast(H)
                fits[f'{d}{p}{q}'] = dict(aicc=round(float(m.aicc), 3), conv=bool(m.mle_retvals.get('converged', True)),
                                          f=r(fc.predicted_mean.values, 4), se=r(fc.se_mean.values, 4),
                                          mu=round(float(m.params['const']), 4) if d == 0 else None)
    return dict(m=[str(p) for p in pi.index], y=r(pi.values, 4), fits=fits)


def eurron_block():
    f = g.fig_eurron_forecast()
    close(f['rmse1']['ar'], N['fx']['rmse1']['ar'], 1e-12, 'fx')
    y = 100 * np.log(g.load_close('eurron'))
    return dict(dd(y.index), v=r(y.values, 5))


def us_block(H_end='2026Q2'):
    """fig_us_gdp from the fourth quarter of every year 1990-2019: both models refitted on the data up to the origin,
    forecasts to 2026Q2."""
    u = g.fig_us_gdp()
    close(u['ts']['gap_end'], N['us']['ts']['gap_end'], 1e-6, 'us')
    y = g.quarterly(100 * np.log(g.us_gdp()))
    out = {}
    for yr in range(1990, 2020):
        cut = f'{yr}Q4'
        tr = y.loc[:cut]
        H = len(y.loc[cut:]) - 1
        o = {}
        for lab, order, trend in (('ts', (2, 0, 0), 'ct'), ('ds', (1, 1, 0), 't')):
            fc = arima(tr, order, trend).get_forecast(H)
            o[lab] = dict(f=r(fc.predicted_mean.values, 3), se=r(fc.se_mean.values, 3))
        out[cut] = o
    gap = y.iloc[-1] - out['2007Q4']['ts']['f'][-1]
    close(gap, N['us']['ts']['gap_end'], 2e-3, 'us gap')
    return dict(q=qd(y.index), y=r(y.values, 4), origins=out)


# ---------------------------------------------------------------- tables of the lecture (blocks of their own frames)
def tables_block(D):
    """Inputs of the table blocks, computed from the series already exported (so they are the series of the slides):
    the unit-root table for every deterministic case and lag criterion (ADF, PP, KPSS and the DF-GLS test of Elliott,
    Rothenberg and Stock from the arch package), and the AICc grid of ARIMA(p,1,q) with drift for Romanian GDP for every
    sample end since 2015Q4."""
    from statsmodels.tsa.stattools import adfuller
    from arch.unitroot import DFGLS
    from tsa_data import load_close
    gdp = np.asarray(D['gdp']['y'], float)
    hm = D['four']['hicp_m']
    h = 100 * np.asarray(D['four']['hicp'], float)[hm.index('2005-01'):]
    pi = np.asarray(D['infl']['y'], float)
    fx = np.asarray(D['fx']['v'], float)
    bet = 100 * np.log(load_close('bet', start=g.START)).values
    sp = 100 * np.log(load_close('sp500', start=g.START)).values
    us = np.asarray(D['overdiff']['us'], float)
    nile = np.asarray(D['brk_real']['nile']['v'], float)
    ser = [('Romania real GDP, log', gdp, 'ct'), ('Romania real GDP, growth', np.diff(gdp), 'c'), ('Romania HICP, log', h, 'ct'),
           ('Romania inflation, 12 months', pi, 'c'), ('Romania inflation, change', np.diff(pi), 'c'), ('EUR/RON, log', fx, 'c'),
           ('EUR/RON, returns', np.diff(fx), 'c'), ('BET, log price', bet, 'ct'), ('BET, returns', np.diff(bet), 'c'),
           ('S&P 500, log price', sp, 'ct'), ('S&P 500, returns', np.diff(sp), 'c'), ('US real GDP, log', us, 'ct'),
           ('US real GDP, growth', np.diff(us), 'c'), ('Nile flow', nile, 'c')]
    urt = []
    for name, y, reg0 in ser:
        e = dict(name=name, reg=reg0, n=len(y), adf={}, pp={}, kpss={}, ers={})
        for reg in ('n', 'c', 'ct'):
            for ic in ('AIC', 'BIC'):
                a = adfuller(y, regression=reg, autolag=ic)
                e['adf'][f'{reg}_{ic}'] = [round(float(a[0]), 4), float('%.4g' % a[1]), int(a[2]), int(a[3])]
            p = g.pp_test(y, reg)
            e['pp'][reg] = [round(p['stat'], 4), float('%.4g' % p['p']), p['lags']]
            if reg != 'n':
                k = g.kpss_test(y, reg)
                e['kpss'][reg] = [round(k['stat'], 4), round(k['p'], 4), k['lags']]
                for ic in ('AIC', 'BIC'):
                    z = DFGLS(y, trend=reg, method=ic.lower())
                    cv = z.critical_values
                    e['ers'][f'{reg}_{ic}'] = [round(float(z.stat), 4), float('%.4g' % z.pvalue), int(z.lags),
                                               round(cv['1%'], 4), round(cv['5%'], 4), round(cv['10%'], 4)]
        u = N['urt'][name]
        close(e['adf'][f'{reg0}_AIC'][0], u['adf']['stat'], 2e-3, 'urt adf ' + name)
        close(e['kpss'][reg0][0], u['kpss']['stat'], 2e-3, 'urt kpss ' + name)
        close(e['pp'][reg0][0], u['pp']['stat'], 2e-3, 'urt pp ' + name)
        urt.append(e)
    S = pd.Series(gdp, index=pd.PeriodIndex(D['gdp']['q'], freq='Q'))
    grid = {}
    for k in range(S.index.get_loc(pd.Period('2015Q4', 'Q')), len(S)):
        y = S.iloc[:k + 1]
        o = {}
        for p in range(3):
            for q in range(3):
                m = arima(y, (p, 1, q), 't')
                o[f'{p}{q}'] = [round(float(m.aicc), 3), round(float(m.aic), 3), round(float(m.bic), 3), int(bool(m.mle_retvals.get('converged', True))),
                                round(float(m.params['x1']), 4), round(float(m.bse['x1']), 4), round(float(np.sqrt(m.params['sigma2'])), 4), round(float(m.llf), 3)]
        grid[str(y.index[-1])] = o
    for pq, v in N['gdp_models']['main']['grid'].items():
        close(grid['2026Q2'][pq][0], v['aicc'], 1e-3, 'aicc ' + pq)
    return dict(urt=urt, aicc=grid)


def main():
    keys = ('four', 'tsds', 'detrend', 'spur_mc', 'spur_real', 'dfdist', 'power', 'kpss', 'dfhand', 'brk_sim', 'brk_real',
            'gdp_id', 'gdp_models', 'gdp_diag', 'gdp_fc', 'overdiff', 'width', 'hand110', 'infl', 'fx', 'us')
    d = dict(numbers={k: N[k] for k in keys})
    for k in ('grid0', 'grid1'):
        d['numbers']['infl'] = {kk: v for kk, v in N['infl'].items() if kk not in ('grid0', 'grid1')}
    steps = [('four', four_block), ('tsds', tsds_block), ('detrend', detrend_block), ('spur', spurious_block),
             ('dfdist', dfdist_block), ('power', power_extra), ('kpss', kpss_block), ('brk_sim', breaks_sim_block),
             ('brk_real', breaks_real_block), ('gdp', gdp_block), ('gdp_fc', gdp_fc_block), ('overdiff', overdiff_block),
             ('infl', inflation_block), ('fx', eurron_block), ('us', us_block)]
    for k, f in steps:
        print(k)
        d[k] = f()
    d['tab'] = tables_block(d)
    write(d)


def write(d):
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 3 interactive page, exported by Quantlets/Ch_03/export_interactive.py.\n')
        fh.write('window.CH3_DATA = ' + json.dumps(d, separators=(',', ':')) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    import sys
    if '--tables' in sys.argv:          # add only the inputs of the table blocks to the existing ch3_data.js (no download)
        txt = open(OUT).read()
        d = json.loads(txt[txt.index('=') + 1:].strip().rstrip(';'))
        d['tab'] = tables_block(d)
        write(d)
    else:
        main()
