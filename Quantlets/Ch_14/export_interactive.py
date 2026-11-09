"""Export the inputs of the Chapter 14 interactive page (TSA/interactive/ch14_data.js).

The script first runs every function of the chapter (generate_all_charts.py) and checks the results against
ch14_numbers.json (it stops on any drift), then exports what the page needs:
  * the daily log returns (in %) of the chapter's own samples, on the common trading days of each sample (data/market,
    EODHD, not revised): S&P 500 and DAX since 2000, BET and DAX since 2000 and since 2005, S&P 500 and Bitcoin since
    September 2014, the five-asset panel since 2015, and the weekly returns of the panel;
  * the estimates of the chapter's code: the GARCH(1,1) fits of step 1 (arch package), the DCC parameters (a, b) of
    step 2 with their standard errors and log-likelihoods, for the full S&P 500 and DAX sample, for the five-asset
    panel, for the 2000-2014 estimation sample of the portfolio VaR and for the 2005-2014 sample of the hedge.
The browser runs the cheap recursions itself (the GARCH variance filter with the given parameters, the DCC recursion,
the rolling correlations, the Kupiec and Christoffersen tests); the exporter checks here that the same recursions,
written as in the page, give back the chapter's numbers. Nothing of the chapter is written: st.save_fig is replaced by
plt.close and the numbers JSON is only read.
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_14/export_interactive.py   (about one minute)
"""
import json
import math
import os
import sys
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
g.st.save_fig = lambda name, *a, **k: plt.close('all')     # never overwrite the charts of the slides
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch14_data.js')
N = json.load(open(os.path.join(HERE, 'ch14_numbers.json')))
TOL = 1e-9


def close(a, b, tol, what):
    a, b = float(a), float(b)
    if math.isnan(b) and math.isnan(a):
        return
    if not abs(a - b) <= tol * max(1.0, abs(b)):
        raise SystemExit(f'{what}: {a} differs from the lecture value {b}')


def cmp(a, b, tol, path):
    if isinstance(b, dict):
        if set(a) != set(b):
            raise SystemExit(f'{path}: keys {sorted(a)} != {sorted(b)}')
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


J = lambda x: json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))  # noqa: E731


def sg(x, k=15):
    return [None if not np.isfinite(v) else float(f'{v:.{k}g}') for v in np.ravel(np.asarray(x, float))]


# ---------------------------------------------------------------- 1. every result of the chapter
def chapter_results():
    for name, f in [('ex', g.worked_examples), ('roll', g.fig_rolling_corr), ('div', g.fig_diversification),
                    ('params', g.fig_param_count), ('step1', g.fig_step1), ('dcc', g.fig_dcc_sp_dax), ('zoom', g.fig_crisis_zoom),
                    ('panel', g.fig_panel_corr), ('heat', g.fig_panel_heatmap), ('var', g.fig_var_backtest), ('hedge', g.fig_hedge)]:
        print(name, flush=True)
        cmp(J(f(save_it=True) if name != 'ex' else f()), N[name], TOL, name)


# ---------------------------------------------------------------- 2. the recursions of the page, written as in ch14.js
def garch_sigma(r, p, s0):
    """sigma_t of the GARCH(1,1) filter as a loop (the page's version of garch11_variance)."""
    e = np.asarray(r, float) - p['mu']
    s2 = np.empty(len(e))
    s2[0] = s0
    for t in range(1, len(e)):
        s2[t] = p['omega'] + p['alpha'] * e[t - 1] ** 2 + p['beta'] * s2[t - 1]
    return np.sqrt(s2)


def dcc_loop(Z, a, b, Qbar):
    """R_t of the DCC recursion as a loop (the page's version of dcc_path)."""
    T, n = Z.shape
    Q = Qbar.copy()
    R = np.empty((T, n, n))
    for t in range(T):
        if t:
            z = Z[t - 1]
            Q = (1 - a - b) * Qbar + a * np.outer(z, z) + b * Q
        d = np.sqrt(np.diag(Q))
        R[t] = Q / np.outer(d, d)
    return R


def sample(R, P, fit_until=None):
    """Returns, sigma and z of one sample, with the starting variance of the chapter (step1)."""
    S, Z, s0 = {}, {}, {}
    for c in R.columns:
        x = R[c].loc[:fit_until] if fit_until else R[c]
        s0[c] = float(np.var(x.values - P[c]['mu']))
        S[c] = garch_sigma(R[c].values, P[c], s0[c])
        Z[c] = (R[c].values - P[c]['mu']) / S[c]
    return pd.DataFrame(S, index=R.index), pd.DataFrame(Z, index=R.index), s0


def check(a, b, what, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.nanmax(np.abs(a - b))
    if not m <= tol:
        raise SystemExit(f'{what}: the recursion of the page differs by {m}')


def ret_block(R):
    d0 = R.index[0]
    return dict(d0=str(d0.date()), off=[int((d - d0).days) for d in R.index], cols=list(R.columns),
                r={c: sg(R[c].values) for c in R.columns})


def par(P):
    return {k: {q: float(v) for q, v in P[k].items()} for k in P}


def fitd(f):
    return dict(a=f['a'], b=f['b'], se_a=f['se_a'], se_b=f['se_b'], ll=f['loglik'], ll0=f['ccc_loglik'], lr=f['lr'],
                qbar=sg(f['Qbar']))


def pair():
    R, P, S, Z, f, rho = g.dcc_sp_dax()
    S2, Z2, s0 = sample(R, P)
    check(S2.values, S.values, 'step 1 sigma (S&P 500, DAX)')
    check(Z2.values, Z.values, 'step 1 z (S&P 500, DAX)')
    Rt = dcc_loop(Z2.values, f['a'], f['b'], f['Qbar'])
    check(Rt[:, 0, 1], rho.values, 'DCC path (S&P 500, DAX)')
    close(g.dcc_loglik(Z2.values, f['a'], f['b'], f['Qbar']), N['dcc']['ll'], 1e-8, 'DCC log-likelihood')
    return dict(ret_block(R), P=par(P), s0=s0, fit=fitd(f))


def panel():
    R, P, S, Z, f, Rt = g.dcc_panel()
    S2, Z2, s0 = sample(R, P)
    check(Z2.values, Z.values, 'step 1 z (panel)')
    check(dcc_loop(Z2.values, f['a'], f['b'], f['Qbar']), Rt, 'DCC path (panel)')
    W = g.weekly_returns(g.PANEL)
    return dict(ret_block(R), P=par(P), s0=s0, fit=fitd(f), week=ret_block(W))


def var():
    R = g.common_returns(['sp500', 'dax'], g.PAIR_START)
    P, S, Z = g.step1(R, fit_until=g.SPLIT)
    f = g.dcc_fit(Z.loc[:g.SPLIT].values)
    close(f['a'], N['var']['a'], TOL, 'var a')
    close(f['b'], N['var']['b'], TOL, 'var b')
    S2, Z2, s0 = sample(R, P, g.SPLIT)
    check(Z2.values, Z.values, 'step 1 z (VaR)')
    return dict(P=par(P), s0=s0, fit=fitd(f), split=g.SPLIT, n_in=int((R.index <= g.SPLIT).sum()),
                cov_in=sg(np.cov(R.loc[:g.SPLIT].values.T)))


def hedge():
    R = g.common_returns(['bet', 'dax'], g.HEDGE_START)
    P, S, Z = g.step1(R, fit_until=g.SPLIT)
    f = g.dcc_fit(Z.loc[:g.SPLIT].values)
    close(f['a'], N['hedge']['a'], TOL, 'hedge a')
    S2, Z2, s0 = sample(R, P, g.SPLIT)
    check(Z2.values, Z.values, 'step 1 z (hedge)')
    return dict(ret_block(R), P=par(P), s0=s0, fit=fitd(f))


def rolling():
    out = {}
    for x, y, a in (('bet', 'dax', g.PAIR_START), ('sp500', 'btc', '2014-09-17')):
        out[f'{x}_{y}'] = ret_block(g.common_returns([x, y], a))
    return out


def main():
    chapter_results()
    d = dict(numbers=N, ex=g.EX, roll=g.ROLL, split=g.SPLIT, crises=g.CRISES, calm=g.CALM, panel_crisis=g.PANEL_CRISIS,
             names=g.NAME, panel_keys=g.PANEL, weights=list(g.WEIGHTS), alpha=g.ALPHA_VAR)
    print('pair', flush=True)
    d['pair'] = pair()
    print('panel', flush=True)
    d['panel'] = panel()
    print('var', flush=True)
    d['var'] = var()
    print('hedge', flush=True)
    d['hedge'] = hedge()
    d['rollpairs'] = rolling()
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 14 interactive page, exported by Quantlets/Ch_14/export_interactive.py.\n')
        fh.write('window.CH14_DATA = ' + json.dumps(J(d), separators=(',', ':')).replace('NaN', 'null') + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
