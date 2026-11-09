"""Export the inputs of the Chapter 1 interactive page (TSA/interactive/ch1_data.js).

Uses the functions and seeds of generate_all_charts.py, so the page starts from the same data and numbers as the slides:
the shocks of every simulated chart (drawn in the same order from the same generators), the per-sample results of the
Monte Carlo charts (sample ACF under white noise, sizes of the Box-Pierce and Ljung-Box tests), and the real series of
the chapter (Romanian GDP and HICP from Eurostat, EUR/RON from the BNR, BET and S&P 500 from the course market data,
the Nile and the sunspots from statsmodels). Every series is checked against ch1_numbers.json before it is written.
Run:  python3 Quantlets/Ch_01/export_interactive.py   (needs internet for Eurostat and the BNR)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch1_data.js')
N = json.load(open(os.path.join(HERE, 'ch1_numbers.json')))


def r(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


def close(a, b, tol, what):
    assert abs(a - b) <= tol, (what, a, b)


def ds(idx, fmt='%Y-%m-%d'):
    return [t.strftime(fmt) for t in idx]


def dd(idx):
    """Daily dates, compactly: the first date and the gaps in days (the page rebuilds the dates)."""
    return dict(d0=idx[0].strftime('%Y-%m-%d'), gap=np.diff(idx.values).astype('timedelta64[D]').astype(int).tolist())


# ---------------------------------------------------------------- real series
def chart_path(name, k):
    """Vertical coordinates of the k-th long line path of charts/<name>.pdf (matplotlib vector output)."""
    import re
    import pymupdf
    doc = pymupdf.open(os.path.join(HERE, '..', '..', 'charts', name + '.pdf'))
    s = b''.join(doc.xref_stream(x) for x in doc[0].get_contents()).decode('latin1')
    run = re.findall(r'(-?[\d.]+ -?[\d.]+ m\n(?:-?[\d.]+ -?[\d.]+ l\n){60,})', s)[k]
    return np.array([float(l.split()[1]) for l in run.strip().split('\n')])


def gdp_slide_vintage(kind):
    """Romanian quarterly GDP (million EUR) as plotted on the slides. Eurostat has revised the series since the charts
    were made (from 2018Q1 on, by up to 1.3 bn EUR a quarter), so the values are read back from the vector path of the
    GDP panel of charts/tsa_ch1_four_series.pdf (real) or charts/tsa_ch1_boxcox.pdf (nominal), scaled by the first and
    last values in ch1_numbers.json and rounded to the 0.1 million EUR of Eurostat. The statistics of the slides
    (ACF, means, standard deviations, Guerrero's lambda) are reproduced exactly from these values."""
    if kind == 'real':
        yy, v0, v1 = chart_path('tsa_ch1_four_series', 0), N['four']['gdp_v0'], N['four']['gdp_v1']
    else:
        yy, v0, v1 = chart_path('tsa_ch1_boxcox', 0), N['boxcox']['v0'], N['boxcox']['v1']
    v = np.round(1000 * (v0 + (yy - yy[0]) / (yy[-1] - yy[0]) * (v1 - v0)), 1)
    idx = pd.date_range(N['gdp']['first'], N['gdp']['last'], freq='QS')
    assert len(v) == len(idx) == N['four']['gdp_n']
    s = pd.Series(v, index=idx, name=f'gdp_{kind}')
    if kind == 'real':
        l = np.log(s)
        d1, d4 = (100 * l.diff()).dropna(), (100 * l.diff(4)).dropna()
        close(d4.mean(), N['gdp']['mean_d4'], 1e-9, 'mean d4'); close(d1.std(), N['gdp']['sd_d1'], 1e-9, 'sd d1')
        close(d4['2020-04-01'], N['gdp']['covid_d4'], 1e-9, 'covid'); close(g.sample_acf(l, 8)[7], N['gdp']['acf8_level'], 1e-9, 'acf8')
    else:
        lam, _, cv = g.guerrero_lambda(v)
        close(lam, N['boxcox']['lambda'], 1e-12, 'lambda'); close(cv.min(), N['boxcox']['cv_lam'], 1e-12, 'cv')
    return s


def real_block():
    F = N['four']
    gdp = gdp_slide_vintage('real')
    hicp = g.ro_hicp().loc[:F['infl_last']]
    infl = (100 * np.log(hicp).diff(12)).dropna()
    close(infl.max(), F['infl_max'], 1e-6, 'infl max'); close(infl.iloc[-1], F['infl_lastv'], 1e-6, 'infl last')
    nom = gdp_slide_vintage('nominal')
    fx = g.load_close('eurron')
    assert len(fx) == F['fx_n']; close(fx.iloc[-1], F['fx_v1'], 1e-6, 'fx v1')
    bet = g.load_close('bet')
    bp = np.log(bet.loc[g.START:])
    rb = (100 * bp.diff()).dropna()
    assert len(rb) == F['bet_n']; close(rb.std(), F['bet_sd'], 1e-9, 'bet sd')
    sp = np.log(g.load_close('sp500', start=g.START))
    rs = (100 * sp.diff()).dropna()
    assert len(rs) == N['sp500']['n']; close(rs.std(), N['sp500']['sd_r'], 1e-9, 'sp sd')
    nile, sun = g.load_statsmodels('nile'), g.load_statsmodels('sunspots')
    d = dict(gdp=dict(d=ds(gdp.index, '%Y-%m'), y=r(gdp.values, 1)), nom=dict(d=ds(nom.index, '%Y-%m'), y=r(nom.values, 1)),
             hicp=dict(d=ds(hicp.index, '%Y-%m'), y=r(hicp.values, 3)),
             fx=dict(dd(fx.index), y=r(fx.values, 4)), bet=dict(dd(bp.index), p=r(bp.values, 6)),
             sp=dict(dd(sp.index), p=r(sp.values, 6)),
             nile=dict(y0=int(nile.index[0].year), y=r(nile.values, 1)), sun=dict(y0=int(sun.index[0].year), y=r(sun.values, 1)))
    return d


# ---------------------------------------------------------------- simulations, in the order of generate_all_charts.py
def ensemble(n=100, paths=40):
    rng = np.random.default_rng(g.SEED)
    E, Z, W = [], [], []
    for _ in range(paths):
        E.append(rng.normal(size=n)); Z.append(rng.normal(0, 1))   # x[0] = sd_ar * z (normal(0, s) = s * normal())
        W.append(rng.normal(size=n))
    return dict(e=[r(v, 4) for v in E], z0=r(Z, 6), w=[r(v, 4) for v in W])


def nonstationary(n=300):
    rng = np.random.default_rng(g.SEED + 1)
    return {k: r(g.simulate_arma([0.5], n=n, rng=rng), 4) for k in 'abcd'}


def counterexample(n=4000):
    rng = np.random.default_rng(g.SEED + 2)
    ev = rng.normal(size=n // 2)
    od = rng.chisquare(5, size=n // 2)
    close(stats.skew((od - 5) / np.sqrt(10)), N['counter']['skew_o'], 1e-9, 'skew')
    return dict(ev=r(ev, 4), chi=r(od, 4))


def white_noise(n=1000, burn=500):
    rng = np.random.default_rng(g.SEED + 3)
    ga, la = rng.normal(size=n), rng.laplace(0, 1 / np.sqrt(2), size=n)
    z = rng.normal(size=n + burn)
    rng2 = np.random.default_rng(g.SEED + 3); rng2.normal(size=n); rng2.laplace(0, 1 / np.sqrt(2), size=n)
    x = g.simulate_garch(n, rng=rng2)
    close(g.ljung_box(x ** 2, 10)['lb'], N['wn']['GARCH(1,1): weak WN']['lb_x2']['lb'], 1e-6, 'garch')
    return dict(ga=r(ga, 5), la=r(la, 5), z=r(z, 6))


def random_walk(n=250, paths=60, nsim=5000):
    rng = np.random.default_rng(g.SEED + 4)
    x = rng.normal(size=(nsim, n)).cumsum(axis=1)
    t = np.arange(1, n + 1)
    v = x.var(axis=0)
    close(v[-1], N['rw']['var_end'], 1e-9, 'rw var')
    c10 = [float(np.corrcoef(x[:, k], x[:, k + 10])[0, 1]) for k in range(n - 10)]
    close(c10[99], N['rw']['corr_100_110'], 1e-12, 'rw corr')
    out = (np.abs(x) > 1.96 * np.sqrt(t)).mean(axis=0)
    return dict(paths=[r(p, 3) for p in x[:paths]], var=r(v, 3), c10=r(c10, 4), out=r(out, 4), nsim=nsim)


def spot_real(n=500):
    p = np.log(g.load_close('bet')).iloc[-n - 1:]
    rr = p.diff().dropna()
    rng = np.random.default_rng(g.SEED + 5)
    pos = int(rng.integers(0, 6))
    Z = [None if i == pos else r(rng.normal(size=n), 6) for i in range(6)]
    assert 'ABCDEF'[pos] == N['spot']['real'] and ds(p.index[:1])[0] == N['spot']['first']
    return dict(pos=pos, d=ds(p.index), p=r(p.values, 6), z=Z, mu=float(rr.mean()), sd=float(rr.std()))


def ergodicity(n=2000, paths=6, burn=500):
    rng = np.random.default_rng(g.SEED + 6)
    E, Z, U = [], [], []
    for _ in range(paths):
        E.append(rng.normal(0, 1, n + burn)); Z.append(rng.normal()); U.append(rng.normal(size=n))
    return dict(e=[r(v, 4) for v in E], z=r(Z, 6), u=[r(v, 4) for v in U], burn=burn)


def bartlett(T=100, nsim=10000, nlags=20):
    rng = np.random.default_rng(g.SEED + 8)
    b = 1.96 / np.sqrt(T)
    r1, mask = np.empty(nsim), np.empty(nsim, int)
    for i in range(nsim):
        rr = g.sample_acf(rng.normal(size=T), nlags)
        r1[i] = rr[0]
        mask[i] = int(sum(1 << k for k in range(nlags) if abs(rr[k]) > b))
    close(r1.std(), N['bartlett']['sd_r1'], 1e-12, 'bartlett')
    return dict(T=T, r1=r(r1, 4), mask=mask.tolist())


def acf_models(n=500, burn=500):
    rng = np.random.default_rng(g.SEED + 7)
    return dict(wn=r(rng.normal(size=n), 5), ar=r(rng.normal(0, 1, n + burn), 5), ma=r(rng.normal(0, 1, n + burn), 5),
                rw=r(rng.normal(size=n), 5))


def acf_pacf(n=500, burn=500):
    rng = np.random.default_rng(g.SEED + 9)
    return [r(rng.normal(0, 1, n + burn), 5) for _ in range(3)]


def lb_size(nsim=5000, mmax=20, seed=g.SEED + 10):
    """Rejection rates of Box-Pierce and Ljung-Box for every m <= 20 and three levels, from the samples of fig_lb_size
    (lb_size restarts the generator for every case, so the samples of T = 50 are the same for m = 10 and m = 20)."""
    h = np.arange(1, mmax + 1)
    out = {}
    for T in (50, 100, 500):
        rng = np.random.default_rng(seed)
        R = np.array([g.sample_acf(rng.normal(size=T), mmax) for _ in range(nsim)])
        bp = T * np.cumsum(R ** 2, axis=1)
        lb = T * (T + 2) * np.cumsum(R ** 2 / (T - h), axis=1)
        o = {}
        for a in ('0.01', '0.05', '0.1'):
            c = stats.chi2.ppf(1 - float(a), h)
            o[a] = dict(bp=r((bp > c).mean(axis=0), 4), lb=r((lb > c).mean(axis=0), 4))
        out[str(T)] = o
        for m in (10, 20):
            close(o['0.05']['bp'][m - 1], N['lbsize'][f'{T}_{m}']['bp'], 1e-9, 'bp size')
            close(o['0.05']['lb'][m - 1], N['lbsize'][f'{T}_{m}']['lb'], 1e-9, 'lb size')
    return dict(nsim=nsim, rates=out)


def lb_real(mmax=30):
    R = g.real_series()
    l = np.log(gdp_slide_vintage('real'))                   # the GDP of the slides, not the revised one
    R['GDP growth, y/y'], R['GDP growth, q/q'] = (100 * l.diff(4)).dropna(), (100 * l.diff()).dropna()
    h = np.log(g.ro_hicp().loc[:N['four']['infl_last']])   # the months of the slides (Eurostat has added months since)
    R['HICP inflation, m/m'] = (100 * h.diff())['2005':].dropna()
    out = {}
    for k, x in R.items():
        x = np.asarray(x, float)
        rr = g.sample_acf(x, mmax)
        close(len(x) * (len(x) + 2) * np.sum(rr[:10] ** 2 / (len(x) - np.arange(1, 11))), N['real'][k]['m10']['lb'], 1e-6 * max(1, N['real'][k]['m10']['lb']), k)
        out[k] = dict(n=int(len(x)), r=r(rr, 6))
    return out


def main():
    d = dict(numbers={k: N[k] for k in ('four', 'ensemble', 'counter', 'wn', 'rw', 'spot', 'sp500', 'ergo', 'acfm', 'bartlett',
                                       'acfpacf', 'bet', 'lbsize', 'gdp', 'boxcox', 'overdiff', 'textbook')})
    print('real'); d['real'] = real_block()
    print('ensemble'); d['ens'] = ensemble()
    print('nonstationary'); d['ns'] = nonstationary()
    print('counterexample'); d['ce'] = counterexample()
    print('white noise'); d['wn'] = white_noise()
    print('random walk'); d['rw'] = random_walk()
    print('spot'); d['spot'] = spot_real()
    print('ergodicity'); d['ergo'] = ergodicity()
    print('acf models'); d['acfm'] = acf_models()
    print('bartlett'); d['bart'] = bartlett()
    print('acf pacf'); d['ap'] = acf_pacf()
    print('lb size'); d['lbs'] = lb_size()
    print('lb real'); d['lbr'] = lb_real()
    rng = np.random.default_rng(g.SEED + 11)
    d['od'] = r(rng.normal(size=501), 5)
    with open(OUT, 'w') as f:
        f.write('// Inputs of the Chapter 1 interactive page, exported by Quantlets/Ch_01/export_interactive.py.\n')
        f.write('window.CH1_DATA = ' + json.dumps(d, separators=(',', ':')) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
