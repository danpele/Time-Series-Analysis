"""Export the inputs of the Chapter 5 interactive page (TSA/interactive/ch5_data.js).

Uses the functions, seeds and data of generate_all_charts.py, so the page starts from the same data and numbers as the
slides: daily log returns in % of the S&P 500, the BET, the EUR/RON (BNR reference rate) and Bitcoin up to 18 September 2026
(the end date of the chapter), the GARCH(1,1)-t, GARCH(1,1)-N, AR(1)-GARCH(1,1)-t, GJR-t and EGARCH-t estimates of the
chapter, the Normal draws of the chapter's seeds (simulated paths and likelihood examples), and the out-of-sample variance
forecasts of GARCH-t and GJR-t with the parameters of every re-estimation window. The page recomputes the conditional
variances from the parameters and the first conditional variance of each fit; this is checked here against the arch package.
Every value of the slides that the page shows is checked against ch5_numbers.json before the file is written. No chart is
written: the chapter's save() is replaced by plt.close().
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_05/export_interactive.py   (BNR reference rates are read online; a few minutes)
"""
import json
import os
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from scipy import stats                 # noqa: E402
from arch import arch_model             # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
g.save = lambda name, save_it=True: plt.close('all')   # never overwrite the charts of the slides
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch5_data.js')
N = json.load(open(os.path.join(HERE, 'ch5_numbers.json')))
DEC = 6                                 # decimals of the exported returns (EUR/RON moves by thousandths of a percent)


def rd(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


def sig6(x):
    return [float(f'{v:.6g}') for v in np.asarray(x, float)]


def close(a, b, tol, what):
    if a == b:
        return
    if not abs(a - b) <= tol * max(1.0, abs(b)):
        raise SystemExit(f'{what}: {a} differs from the lecture value {b}')


def recursion(eps, om, a, b, s0):
    s = np.empty(len(eps))
    s[0] = s0
    for t in range(1, len(eps)):
        s[t] = om + a * eps[t - 1] ** 2 + b * s[t - 1]
    return s


def series_block():
    """Returns (rounded as on the page), dates as gaps in days, and the GARCH(1,1)-t fit of each market."""
    out = {}
    for k in g.ASSETS:
        r = g.returns(k)
        assert r.index[-1].date().isoformat() == N['end']
        x = np.round(r.values, DEC)
        res = g.fit(r, dist='t')
        p = g.par(res)
        v = g.sig(res).values
        s = recursion(x - p['mu'], p['omega'], p['alpha[1]'], p['beta[1]'], v[0] ** 2)
        close(float(np.max(np.abs(np.sqrt(s) / v - 1))), 0, 1e-4, f'{k}: GARCH recursion')
        sm = g.summary(res, r)
        for q in ('pers', 'hl'):
            close(sm[q], N['markets'][k][q], 1e-9, f'{k} {q}')
        days = (r.index - r.index[0]).days.values
        out[k] = dict(d0=r.index[0].date().isoformat(), gap=np.diff(days).tolist(), r=x.tolist(), ppy=sm['ppy'],
                      g=dict(mu=p['mu'], omega=p['omega'], alpha=p['alpha[1]'], beta=p['beta[1]'], nu=p['nu'], s0=float(v[0] ** 2),
                             pers=sm['pers'], hl=sm['hl'], vol_lr=sm.get('vol_lr'), vol_sample=sm['vol_sample']))
    return out


def normal_sp500():
    """Sorted standardised residuals of the GARCH(1,1)-N of the S&P 500 (QQ plot), as the arch package gives them."""
    r = g.returns('sp500')
    z = np.sort(g.fit(r, dist='normal').std_resid.dropna().values)
    close(float(np.quantile(z, 0.001)), N['qq']['q']['GARCH(1,1)-N']['q001'], 1e-9, 'GARCH-N quantile')
    return rd(z, 7)


def ar_garch_bet():
    """AR(1)-GARCH(1,1)-t of the BET (forecast intervals); the first return is held back by the AR(1) mean."""
    r = g.returns('bet')
    res = g.fit(r, mean='AR', dist='t')
    p = g.par(res)
    v = g.sig(res).values
    x = np.round(r.values, DEC)
    e = x[1:] - p['Const'] - p['bet[1]'] * x[:-1]
    s = recursion(e, p['omega'], p['alpha[1]'], p['beta[1]'], v[1] ** 2)
    close(float(np.max(np.abs(np.sqrt(s) / v[1:] - 1))), 0, 1e-4, 'AR-GARCH recursion')
    return dict(c=p['Const'], phi=p['bet[1]'], omega=p['omega'], alpha=p['alpha[1]'], beta=p['beta[1]'], nu=p['nu'], s1=float(v[1] ** 2))


def asym_block():
    """GARCH-t, GJR-t and EGARCH-t parameters of the series without scaling (news impact curves)."""
    out = {}
    for k in ('sp500', 'bet', 'btc'):
        r = g.returns(k)
        m = {}
        for lab, kw in [('garch', dict()), ('gjr', dict(o=1)), ('egarch', dict(vol='EGARCH'))]:
            m[lab] = {n: float(v) for n, v in g.fit(r, dist='t', **kw).params.items()}
        m['s2bar'] = float(r.var())
        out[k] = m
    rg = g.fit(g.returns('sp500'), o=1, dist='t')
    s2 = float(g.returns('sp500').var())
    ratio = g.news_impact(rg, np.array([-2.0]), s2)[0] / g.news_impact(rg, np.array([2.0]), s2)[0]
    close(float(ratio), N['nic']['sp500']['ratio_gjr'], 1e-9, 'GJR ratio')
    return out


def sim_block():
    """Normal draws of the chapter's seeds: the i.i.d. path (SEED), the ARCH(1) and GARCH(1,1) shocks (SEED + 1, SEED + 2,
    500 burn-in values each), the ARCH(1) likelihood samples (SEED + n) and the GARCH(1,1) likelihood sample (SEED + 7)."""
    n, burn = 1000, 500
    iid = np.random.default_rng(g.SEED).standard_normal(n)
    z1 = np.random.default_rng(g.SEED + 1).standard_normal(n + burn)
    z2 = np.random.default_rng(g.SEED + 2).standard_normal(n + burn)
    e1 = g.simulate_garch(n, 0.5, 0.5, 0.0, seed=g.SEED + 1)[0]
    close(float(stats.kurtosis(e1, fisher=False)), N['sim']['ARCH(1)']['kurt'], 1e-9, 'ARCH kurtosis')
    lik1 = {}
    for m in (100, 250, 500, 1000, 2000, 5000):
        lik1[str(m)] = rd(np.random.default_rng(g.SEED + m).standard_normal(m + burn), 5)
    z7 = np.random.default_rng(g.SEED + 7).standard_normal(5000 + burn)
    a = g.simulate_garch(2000, 0.1, 0.1, 0.8, seed=g.SEED + 7)[0]
    b = g.simulate_garch(5000, 0.1, 0.1, 0.8, seed=g.SEED + 7)[0]
    assert np.allclose(a, b[:2000])                       # the sample of n = 500 and of n = 2000 are prefixes of one path
    return dict(iid=rd(iid, 5), z1=rd(z1, 5), z2=rd(z2, 5), burn=burn, lik1=lik1, z7=rd(z7, 5))


def oos_block():
    """Out-of-sample one-day variance forecasts of GARCH-t and GJR-t (the loop of oos_forecasts), with mu and nu of the
    GARCH-t of every re-estimation window (VaR at other levels)."""
    out = {}
    for k in g.ASSETS:
        r = g.returns(k)
        sc = g.SCALE.get(k, 1)
        i0 = r.index.searchsorted(pd.Timestamp(g.OOS_START))
        gv, jv, win = [], [], []
        for s in range(i0, len(r), g.REFIT):
            e = min(s + g.REFIT, len(r))
            for lab, o, col in [('garch', 0, gv), ('gjr', 1, jv)]:
                am = arch_model(sc * r.iloc[:e], mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='t', rescale=False)
                res = am.fit(disp='off', last_obs=r.index[s], options={'maxiter': 2000})
                col.extend((am.fix(res.params).conditional_volatility.iloc[s:e] ** 2 / sc ** 2).tolist())
                if lab == 'garch':
                    win.append([int(s - i0), float(res.params['mu'] / sc), float(res.params['nu'])])
        gv, jv = np.array(gv), np.array(jv)
        rr = r.values[i0:]
        ew = g.ewma_variance(r).values[i0:]
        L = {m: g.qlike(rr ** 2, h) for m, h in [('garch', gv), ('gjr', jv), ('ewma', ew)]}
        for m in L:
            close(float(L[m].mean()), N['fe'][k]['qlike'][m], 1e-9, f'{k} QLIKE {m}')
        close(g.dm_test(L['garch'], L['ewma'])['t'], N['fe'][k]['dm_garch_ewma']['t'], 1e-9, f'{k} DM')
        out[k] = dict(i0=int(i0), garch=sig6(gv), gjr=sig6(jv), win=win)
    return out


def main():
    keys = ('sty', 'returns', 'acf_sq', 'arma', 'sim', 'lik_arch1', 'lik_garch', 'qq', 'markets', 'vol1', 'vol2', 'ewma', 'ag', 'bands',
            'nic', 'asym', 'diag', 'diag_m', 'acf_diag', 'managed', 'term', 'fe', 'fe_fig', 'var_fig', 'end')
    d = dict(numbers={k: N[k] for k in keys})
    steps = [('s', series_block), ('gn', normal_sp500), ('ag', ar_garch_bet), ('asym', asym_block), ('sim', sim_block), ('oos', oos_block)]
    for k, f in steps:
        print(k, flush=True)
        d[k] = f()
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 5 interactive page, exported by Quantlets/Ch_05/export_interactive.py.\n')
        fh.write('window.CH5_DATA = ' + json.dumps(d, separators=(',', ':'), default=str) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
