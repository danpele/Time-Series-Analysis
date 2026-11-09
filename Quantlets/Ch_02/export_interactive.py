"""Export the inputs of the Chapter 2 interactive page (TSA/interactive/ch2_data.js).

Uses the functions and seeds of generate_all_charts.py, so the page starts from the same data and numbers as the slides:
the simulated shocks of the AR(1), MA(1) and ACF/PACF charts, the Monte Carlo estimates of the estimation, selection and
Ljung-Box experiments, Romanian GDP growth with the ARMA(p,q) fits, Romanian HICP inflation with AR(p) fits, BET and
EUR/RON daily returns and the sunspot numbers.
Run:  python3 Quantlets/Ch_02/export_interactive.py   (needs internet for Eurostat and the course market data)
      python3 Quantlets/Ch_02/export_interactive.py --extend   (adds only the inputs of the tables: GDP fits, return SEs)
"""
import json
import os

import numpy as np
import pandas as pd

import generate_all_charts as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'interactive', 'ch2_data.js')


def r(x, d=6):
    return np.round(np.asarray(x, float), d).tolist()


def qd(idx):
    return [f'{t.year}Q{(t.month - 1) // 3 + 1}' for t in idx]


def shocks_ar1(n=200):
    """The shocks of fig_ar1 (same generator and draw)."""
    return np.random.default_rng(g.SEED).normal(size=n + 300)


def shocks_ma1(n=300):
    return np.random.default_rng(g.SEED + 1).normal(size=n + 1)


def shocks_patterns(n=500, burn=500):
    """fig_patterns draws n + burn shocks per model, one model after the other, from one generator."""
    rng = np.random.default_rng(g.SEED + 2)
    return {lab: rng.normal(0, 1, n + burn) for lab in g.MODELS}


def estimators(nrep=400, T=100, phi=0.9, theta=0.5):
    """The Monte Carlo of fig_estimators, keeping every estimate."""
    rng = np.random.default_rng(g.SEED + 3)
    A = np.array([g.ar1_estimates(g.simulate_arma([phi], n=T, rng=rng)) for _ in range(nrep)])
    M = []
    for _ in range(nrep):
        x = g.simulate_arma([], [theta], n=T, rng=rng)
        M.append((g.css_ma1(x), float(g.fit_arma(x, 0, 1).params[1])))
    M = np.array(M)
    return dict(T=T, nrep=nrep, phi=phi, theta=theta, ar=dict(yw=r(A[:, 0], 5), cls=r(A[:, 1], 5), ml=r(A[:, 2], 5)),
                ma=dict(cls=r(M[:, 0], 5), ml=r(M[:, 1], 5)))


def ic_freq(nrep=500, phi=(0.5, 0.25), pmax=6):
    """The Monte Carlo of fig_ic: share of samples in which AIC / BIC choose each order."""
    rng = np.random.default_rng(g.SEED + 4)
    out = {}
    for T in [100, 1000]:
        ch = np.array([g.ar_ols_ic(g.simulate_arma(phi, n=T, rng=rng), pmax) for _ in range(nrep)])
        out[str(T)] = dict(aic=r(np.bincount(ch[:, 0], minlength=pmax + 1) / nrep, 4),
                           bic=r(np.bincount(ch[:, 1], minlength=pmax + 1) / nrep, 4))
    return dict(nrep=nrep, phi=list(phi), pmax=pmax, freq=out)


def lb_acf(nrep=1000, T=200, mmax=20):
    """The Monte Carlo of fig_lb_df, keeping the residual autocorrelations r_1..r_mmax of every sample, so that the page
    can compute the Ljung-Box statistic for any m <= mmax."""
    rng = np.random.default_rng(g.SEED + 5)
    ar, arma = [], []
    for i in range(nrep):
        x = g.simulate_arma([0.7], n=T, rng=rng)
        Z = np.c_[np.ones(T - 1), x[:-1]]
        e = x[1:] - Z @ np.linalg.lstsq(Z, x[1:], rcond=None)[0]
        ar.append(r(g.sample_acf(e, mmax), 6))
        if i < nrep // 2:
            x = g.simulate_arma([0.7], [0.4], n=T, rng=rng)
            e = g.fit_arma(x, 1, 1).resid[1:]
            arma.append(r(g.sample_acf(e, mmax), 6))
    return dict(T=T, n=T - 1, nrep=nrep, ar=ar, arma=arma)


def gdp_slide_vintage():
    """Romanian annual GDP growth as plotted on the slides. Eurostat has revised the series since the charts were made
    (8 Oct 2026; differences up to 1.3 p.p.), so the values are read back from the vector path of
    charts/tsa_ch2_gdp_ident.pdf, scaled by its zero line and its mean line (the mean is in ch2_numbers.json).
    The ARMA fits on this series reproduce ch2_gdp_models.csv to 1e-4."""
    import re
    import pymupdf
    N = json.load(open(os.path.join(HERE, 'ch2_numbers.json')))['gdp']
    doc = pymupdf.open(os.path.join(HERE, '..', '..', 'charts', 'tsa_ch2_gdp_ident.pdf'))
    s = b''.join(doc.xref_stream(x) for x in doc[0].get_contents()).decode('latin1')
    run = re.findall(r'(-?[\d.]+ -?[\d.]+ m\n(?:-?[\d.]+ -?[\d.]+ l\n){100,})', s)[0]
    yy = np.array([float(l.split()[1]) for l in run.strip().split('\n')])
    hl = [float(m.group(1)) for m in re.finditer(r'36\.903125 (-?[\d.]+) m\n652\.973125 \1 l', s)]
    ym, y0 = hl[1], hl[2]                                     # dashed mean line, then the zero line (top panel)
    v = (yy - y0) / (ym - y0) * N['mean']
    idx = pd.date_range(N['first'], N['last'], freq='QS')
    assert len(v) == len(idx) == N['T'] and abs(v.mean() - N['mean']) < 1e-5 and abs(v.min() - N['min']) < 1e-4
    return pd.Series(v, index=idx, name='gdp_yoy')


def gdp_block():
    y = gdp_slide_vintage()
    d = dict(dates=qd(y.index), y=r(y.values, 5))
    models, H = {}, 8
    for p in range(4):
        for qq in range(4):
            res = g.fit_arma(y, p, qq)
            e = np.asarray(res.resid)[max(p, qq):]
            f = res.get_forecast(H)
            models[f'{p}{qq}'] = dict(resid=r(e, 4), aic=round(float(res.aic), 3), bic=round(float(res.bic), 3),
                                      f=r(f.predicted_mean, 4), se=r(f.se_mean, 4), mu=round(float(res.params[0]), 4))
    d['models'] = models
    # forecasts from earlier origins (the two models of the slide, refitted on the data up to each origin)
    orig = {}
    for k in range(len(y) - 48, len(y) + 1):
        ys = y.iloc[:k]
        o = {}
        for p, qq in [(0, 3), (1, 0)]:
            f = g.fit_arma(ys, p, qq).get_forecast(H)
            o[f'{p}{qq}'] = dict(f=r(f.predicted_mean, 4), se=r(f.se_mean, 4))
        orig[qd(ys.index[-1:])[0]] = o
    d['origins'] = orig
    return d


def gdp_fits(y=None):
    """Every ARMA(p, q), p, q <= 3, on the slide series: log-likelihood, k, sigma^2, coefficients with their standard errors
    (statsmodels default) and the moduli of the AR and MA roots (tables of the frames „Step 2” and „Interpreting the
    estimated MA(3)”)."""
    y = gdp_slide_vintage() if y is None else y
    out = {}
    for p in range(4):
        for qq in range(4):
            res = g.fit_arma(y, p, qq)
            ph = [float(v) for n, v in zip(res.param_names, res.params) if n.startswith('ar.')]
            th = [float(v) for n, v in zip(res.param_names, res.params) if n.startswith('ma.')]
            mod = lambda c: sorted(float(abs(z)) for z in np.roots(np.r_[c[::-1], 1.0])) if c else []   # noqa: E731
            out[f'{p}{qq}'] = dict(llf=round(float(res.llf), 4), k=p + qq + 2, nobs=int(res.nobs), names=list(res.param_names),
                                   b=r(res.params, 5), se=r(res.bse, 5), ar_mod=r(mod([-v for v in ph]), 4), ma_mod=r(mod(th), 4))
    return out


def returns_se():
    """Standard errors of the AR(1) coefficient of the daily returns: the statsmodels default of the slide (outer product
    of gradients, OPG) and the observed information matrix (OIM), close to the classical sqrt((1 - phi^2)/T)."""
    out = {}
    for k in ['bet', 'eurron']:
        x = g.log_returns(k).values
        out[k] = {ct: float(g.ARIMA(x, order=(1, 0, 0), trend='c').fit(cov_type=ct).bse[1]) for ct in ('opg', 'oim')}
    return out


def inflation_block(H=24):
    last = json.load(open(os.path.join(HERE, 'ch2_numbers.json')))['infl']['last']
    y = g.ro_inflation().loc[:last]                          # the sample of the slides (Eurostat has added months since)
    d = dict(dates=[f'{t:%Y-%m}' for t in y.index], y=r(y.values, 4), fit={})
    for p in range(1, 7):
        res = g.fit_arma(y.values, p, 0)
        f = res.get_forecast(H)
        ph = res.params[1:1 + p]
        lb = g.ljung_box(res.resid[p:], 12, 12 - p)
        d['fit'][str(p)] = dict(bic=round(float(res.bic), 3), mu=round(float(res.params[0]), 4), phi=r(ph, 4), se_phi=r(res.bse[1:1 + p], 4),
                                sigma=round(float(np.sqrt(res.params[-1])), 4), f=r(f.predicted_mean, 4), se=r(f.se_mean, 4),
                                inv_max=round(float(np.max(np.abs(g.inverse_roots(ph)))), 4), lb12=round(lb['lb'], 3), lb12_p=lb['lb_p'])
    return d


def returns_block():
    out = {}
    for k in ['bet', 'eurron']:
        x = g.log_returns(k)
        m = g.fit_arma(x.values, 1, 0)
        out[k] = dict(dates=[f'{t:%Y-%m-%d}' for t in x.index], r=r(x.values, 5), mu=float(m.params[0]), phi=float(m.params[1]))
    return out


def sunspots_block():
    s = g.load_statsmodels('sunspots')
    years = [int(t.year) if hasattr(t, 'year') else int(t) for t in s.index]
    d = dict(years=years, y=r(s.values, 2), fit={})
    for p in range(1, 11):
        res = g.fit_arma(s.values, p, 0)
        fitted = s.values - res.resid
        d['fit'][str(p)] = dict(aic=round(float(res.aic), 3), bic=round(float(res.bic), 3), phi=r(res.params[1:1 + p], 4),
                                pred=r(fitted, 3), r2=round(float(1 - np.var(res.resid[p:]) / np.var(s.values[p:])), 4))
    return d


def main():
    N = json.load(open(os.path.join(HERE, 'ch2_numbers.json')))
    d = dict(numbers=dict(ar1=N['ar1'], ma1=N['ma1'], ar2=N['ar2'], est=N['est'], ic=N['ic'], lbdf=N['lbdf'], fct=N['fct'],
                          gdp=N['gdp'], gdptab={k: N['gdptab'][k] for k in ('aic_best', 'bic_best', 'bic_params', 'bic_se')},
                          gdpdiag=N['gdpdiag'], infl={k: N['infl'][k] for k in ('p', 'bic', 'mu', 'last', 'last_v', 'lb12', 'T')},
                          ret=N['ret'], sun=N['sun']))
    d['ar1_e'] = r(shocks_ar1(), 6)
    d['ma1_e'] = r(shocks_ma1(), 6)
    d['pat_e'] = {k: r(v, 6) for k, v in shocks_patterns().items()}
    d['models'] = {k: dict(phi=v[0], theta=v[1]) for k, v in g.MODELS.items()}
    print('estimators'); d['est'] = estimators()
    print('ic'); d['ic'] = ic_freq()
    print('lb'); d['lb'] = lb_acf()
    print('gdp'); d['gdp'] = gdp_block()
    print('inflation'); d['infl'] = inflation_block()
    print('returns'); d['ret'] = returns_block()
    print('sunspots'); d['sun'] = sunspots_block()
    print('gdp fits'); d['gdpfit'] = gdp_fits()
    print('returns se'); d['retse'] = returns_se()
    with open(OUT, 'w') as f:
        f.write('// Inputs of the Chapter 2 interactive page, exported by Quantlets/Ch_02/export_interactive.py.\n')
        f.write('window.CH2_DATA = ' + json.dumps(d, separators=(',', ':')) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


def extend():
    """Add only the tables' inputs (gdpfit, retse) to the existing ch2_data.js, keeping the rest of the file as it is."""
    txt = open(OUT).read()
    d = json.loads(txt[txt.index('=') + 1:txt.rindex(';')])
    d['gdpfit'] = gdp_fits()
    d['retse'] = returns_se()
    with open(OUT, 'w') as f:
        f.write(txt[:txt.index('window.CH2_DATA')] + 'window.CH2_DATA = ' + json.dumps(d, separators=(',', ':')) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    import sys
    extend() if '--extend' in sys.argv else main()
