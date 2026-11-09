"""Export the inputs of the Chapter 13 interactive page (TSA/interactive/ch13_data.js).

The script first runs every function of the chapter (generate_all_charts.py) and checks the results against
ch13_numbers.json (it stops on any drift), then exports what the page needs:
  * daily closes of the six episodes (data/market, EODHD: S&P 500, Nasdaq 100, BET, Shanghai Composite, Bitcoin; not
    revised): the browser finds the low, the peak and the fall again and reproduces the numbers of the lecture exactly;
  * the draws of the simulations of the chapter: the random walk of the fundamental value and the shocks of the AR(1)
    paths (seed 2026), the Normal and uniform draws of the Blanchard-Watson bubble (seed 13, in the order the chapter
    draws them);
  * weekly log prices of the four PSY samples; the browser computes ADF, SADF, GSADF and BSADF itself. The Monte Carlo
    critical values (1000 random walks with drift 1/T, seed 2026) are the chapter's psy_cv, run once more here so that the
    90% and 99% quantiles of BSADF are kept as well (the 95% ones are checked against psy_cv);
  * the LPPL fits (lppl_fit, lppl_conditions of the chapter): the six ex-ante fits for windows ending 0-90 days before
    the peak (30 days is the lecture), the fits of the 29 windows of fig_windows and the fits of fig_tc_path, each with the
    p-value of the Lomb test and of the two unit-root tests of the residual (so that the page can apply other filter
    bounds); the cost landscape of fig_cost (SSR minimised over omega up to several upper limits);
  * the cached indicator series of the chapter (Quantlets/Ch_13/ch13_ci_*.csv) and the daily closes of the evaluation.
Nothing of the chapter is written: st.save_fig is replaced by plt.close, cached_ci only reads the CSV files.
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_13/export_interactive.py   (about one minute)
"""
import json
import math
import os
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
_cached = g.cached_ci
g.cached_ci = lambda name, s, start, end=None, step=g.CI_STEP, recompute=False: _cached(name, s, start, end, step, False)
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch13_data.js')
N = json.load(open(os.path.join(HERE, 'ch13_numbers.json')))
TOL = 1e-9
LAGS = [90, 75, 60, 45, 30, 15, 0]          # end of the ex-ante window: days before the peak (lecture: 30)


def close(a, b, tol, what):
    a, b = float(a), float(b)
    if math.isnan(b) and math.isnan(a):
        return
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


J = lambda x: json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))  # noqa: E731


def sg(x, k=12):
    return [None if not np.isfinite(v) else float(f'{v:.{k}g}') for v in np.ravel(np.asarray(x, float))]


def iso(idx):
    return [g.d2s(d) for d in idx]


def series(s, a=None, b=None):
    """Dates as day offsets from the first one, values unrounded."""
    x = s.loc[a:b]
    d0 = x.index[0]
    off = [(d - d0).days for d in x.index]
    return dict(d0=g.d2s(d0), off=off, p=[float(v) for v in x.values])


# ---------------------------------------------------------------- 1. every result of the chapter
def chapter_results():
    for name, f in [('ex', g.worked_examples), ('episodes', g.fig_episodes), ('growth', g.fig_growth), ('rational', g.fig_rational),
                    ('psy_ndx', g.fig_psy_ndx), ('psy', g.fig_psy_panel), ('comp', g.fig_lppl_components), ('cost', g.fig_cost),
                    ('fits', g.fig_fits), ('tcpath', g.fig_tc_path), ('windows', g.fig_windows), ('ci', g.fig_ci), ('eval', g.fig_eval)]:
        print(name, flush=True)
        cmp(J(f()), N[name], TOL, name)


# ---------------------------------------------------------------- 2. episodes
def episodes():
    out = {}
    for k, e in g.EPISODES.items():
        s, low, peak, t2 = g.episode(k)
        a = min(pd.Timestamp(e['low'][0]), low - pd.Timedelta(days=180), peak - pd.Timedelta(days=540))
        b = max(pd.Timestamp(e['peak'][1]), peak + pd.Timedelta(days=540))
        out[k] = dict(series(s, a, b), low=list(e['low']), peak=list(e['peak']), crypto=e['crypto'], label=e['label'])
    return out


# ---------------------------------------------------------------- 3. draws of the simulations
def draws():
    T = 400
    fz = np.random.default_rng(g.SEED).standard_normal(T)
    rng = np.random.default_rng(13)
    e, u = [], []
    for _ in range(1, T):
        e.append(rng.standard_normal())
        u.append(rng.random())
    e, u = np.array(e), np.array(u)
    b = np.empty(T)
    b[0] = 1.0
    for t in range(1, T):
        b[t] = 1.02 / 0.98 * b[t - 1] + 0.3 * e[t - 1] if u[t - 1] < 0.98 else 0.3 * e[t - 1]
    if np.max(np.abs(b - g.blanchard_watson(T))) > 0:
        raise SystemExit('Blanchard-Watson draws')
    ar = np.random.default_rng(g.SEED).standard_normal(200)
    P = g.ar1_paths()
    y = np.zeros(200)
    for t in range(1, 200):
        y[t] = 1.02 * y[t - 1] + ar[t]
    if np.max(np.abs(y - P[1.02])) > 0:
        raise SystemExit('AR(1) draws')
    return dict(fz=sg(fz, 17), bwe=sg(e, 17), bwu=sg(u, 17), ar=sg(ar, 17))


# ---------------------------------------------------------------- 4. PSY: weekly log prices and Monte Carlo critical values
def psy_cv_all(T, w0, R=g.PSY_REPS, seed=g.SEED, chunk=100):
    """The chapter's psy_cv, keeping the 90%, 95% and 99% quantiles of BSADF for each end point."""
    rng = np.random.default_rng(seed)
    Y = np.concatenate([np.zeros((R, 1)), np.cumsum(1.0 / T + rng.standard_normal((R, T)), axis=1)], axis=1)
    bs = np.full((R, T), np.nan)
    fw = np.full((R, T), np.nan)
    for i in range(0, R, chunk):
        C = g._cums(Y[i:i + chunk])
        for e in range(w0 - 1, T):
            stat = g._adf_end(C, e, np.arange(0, e - w0 + 2))
            bs[i:i + chunk, e] = stat.max(axis=1)
            fw[i:i + chunk, e] = stat[:, 0]
    q = lambda a, p: float(np.quantile(a, p))   # noqa: E731
    return dict(adf={p: q(fw[:, -1], p / 100) for p in (90, 95, 99)},
                sadf={p: q(np.nanmax(fw, axis=1), p / 100) for p in (90, 95, 99)},
                gsadf={p: q(np.nanmax(bs, axis=1), p / 100) for p in (90, 95, 99)},
                bq={p: np.nanquantile(bs, p / 100, axis=0) for p in (90, 95, 99)})


def psy_samples():
    out = {}
    for k, (sym, crypto, a, b) in g.PSY_SAMPLES.items():
        print('psy', k, flush=True)
        y = g.weekly(g.prices(sym, crypto)).loc[a:b]
        o = g.psy(y.values)
        cv = psy_cv_all(o['n'], o['w0'])
        ref = g.psy_cv(o['n'], o['w0'])
        for kk in ('adf', 'sadf', 'gsadf'):
            cmp(cv[kk], ref[kk], 0, f'psy {k} cv {kk}')
        if not np.array_equal(cv['bq'][95], ref['bsadf95'], equal_nan=True):
            raise SystemExit(f'psy {k}: BSADF 95% quantiles')
        ep = g.episodes_above(o['bsadf'], cv['bq'][95], y.index[1:], int(round(np.log(o['n']))))
        lec = N['psy_ndx'] if k == 'ndx' else N['psy'][k]
        cmp(J(ep), lec['episodes'], 0, f'psy {k} episodes')
        close(o['gsadf'], lec['gsadf'], TOL, f'psy {k} gsadf')
        out[k] = dict(dates=iso(y.index), y=sg(y.values, 17), cv={kk: {str(p): v for p, v in cv[kk].items()} for kk in ('adf', 'sadf', 'gsadf')},
                      bq={str(p): sg(v, 12) for p, v in cv['bq'].items()}, label=g.PSY_LABEL[k])
    return out


# ---------------------------------------------------------------- 5. LPPL fits
def fit_row(f, c):
    """Parameters of one fit, its filter conditions and the p-values of the residual tests."""
    from statsmodels.tsa.stattools import adfuller
    from arch.unitroot import PhillipsPerron
    r = {p: float(f[p]) for p in ('t1', 't2', 'tc', 'm', 'w', 'A', 'B', 'C1', 'C2', 'C', 'damping', 'osc', 'rel_err', 'rmse', 'ssr')}
    r['n'] = int(f['n'])
    e = f['_yhat'] - f['_y']
    r['lomb_p'] = g.lomb_pvalue(f)
    r['adf_p'] = float(adfuller(e, maxlag=0, autolag=None, regression='c')[1])
    r['pp_p'] = float(PhillipsPerron(e, trend='c').pvalue)
    if g.ar1_pass(f, g.LPPLS_FILTER['ar1_alpha']) != (max(r['adf_p'], r['pp_p']) < g.LPPLS_FILTER['ar1_alpha']):
        raise SystemExit('residual unit-root tests')
    if c['lomb'] is not None and c['lomb'] != (r['lomb_p'] <= g.LPPLS_FILTER['lomb_alpha']):
        raise SystemExit('Lomb test')
    r['cond'] = {k: (None if v is None else bool(v)) for k, v in c.items()}
    r['q_param'], r['q_full'] = g.qualified(c, 'param'), g.qualified(c, 'full')
    return r


def fits():
    out = {}
    for k in g.EPISODES:
        print('fits', k, flush=True)
        s, low, peak, t2 = g.episode(k)
        rows = {}
        for lag in LAGS:
            e = s.loc[:peak - pd.Timedelta(days=lag)].index[-1]
            f = g.fit_window(s, low, e)
            r = fit_row(f, g.lppl_conditions(f))
            r['end'] = g.d2s(e)
            rows[str(lag)] = r
            if lag == g.FIT_LAG_DAYS:
                lec = N['fits'][k]
                if r['end'] != lec['t2'] or g.d2s(g.todate(r['tc'])) != lec['tc']:
                    raise SystemExit(f'fits {k}: window end or tc')
                for p in ('m', 'w', 'A', 'B', 'C1', 'C2', 'C', 'damping', 'osc', 'rel_err', 'rmse'):
                    close(r[p], lec[p], TOL, f'fits {k} {p}')
                cmp(r['cond'], lec['cond'], 0, f'fits {k} cond')
        out[k] = dict(low=g.d2s(low), peak=g.d2s(peak), rows=rows)
    return out


def windows():
    out = {}
    for k in ('ndx', 'ssec'):
        print('windows', k, flush=True)
        s, low, peak, t2 = g.episode(k)
        t, y = g.yrs(s.index), np.log(s.values)
        i2 = s.index.get_loc(t2)
        rows = []
        for L in g.WINDOWS:
            f = g.lppl_fit(t[i2 - L:i2 + 1], y[i2 - L:i2 + 1])
            r = fit_row(f, g.lppl_conditions(f))
            r['L'] = L
            r['start'] = g.d2s(s.index[i2 - L])
            rows.append(r)
        d, _, _ = g.windows_at(k)
        if len(d) != len(rows) or any(g.todate(r['tc']) != tc for r, tc in zip(rows, d['tc'])) or [r['q_param'] for r in rows] != list(d['q_param']):
            raise SystemExit(f'windows {k}')
        lec = N['windows'][k]
        if sum(r['q_param'] for r in rows) != lec['n_param'] or sum(r['q_full'] for r in rows) != lec['n_full']:
            raise SystemExit(f'windows {k}: counts')
        out[k] = dict(t2=g.d2s(t2), peak=g.d2s(peak), rows=rows)
    return out


def tc_paths():
    out = {}
    for k in ('ndx', 'btc17'):
        print('tc path', k, flush=True)
        s, low, peak, _ = g.episode(k)
        ends = s.loc[peak - pd.Timedelta(days=240):peak + pd.Timedelta(days=30)].index
        ends = ends[::max(1, 7 if g.EPISODES[k]['crypto'] else 5)]
        rows = []
        for e in ends:
            f = g.fit_window(s, low, e)
            if f is None:
                continue
            rows.append(dict(fit_row(f, g.lppl_conditions(f)), end=g.d2s(e)))
        d, _ = g.tc_path(k)
        if len(d) != len(rows) or any(g.todate(r['tc']) != tc for r, tc in zip(rows, d['tc'])) or [r['q_param'] for r in rows] != list(d['q']):
            raise SystemExit(f'tc path {k}')
        out[k] = dict(low=g.d2s(low), peak=g.d2s(peak), rows=rows)
    return out


def cost():
    print('cost', flush=True)
    s, low, peak, t2 = g.episode('ndx')
    x = s.loc[low:t2]
    t, y = g.yrs(x.index), np.log(x.values)
    D = t[-1] - t[0]
    tcs = np.linspace(t[-1] + 0.002, t[-1] + D / 3, 60)
    ms = np.linspace(0.05, 0.99, 48)
    ws = np.linspace(2, 25, 47)
    TC, M, W, beta, ssr = g._grid(t, y, tcs, ms, ws)
    cube = ssr.reshape(len(tcs), len(ms), len(ws))
    keep = list(range(2, 47, 4))                                        # omega up to 3, 5, ..., 25
    S = {f'{ws[j]:g}': sg(cube[:, :, :j + 1].min(axis=2), 10) for j in keep}
    full = np.array(S['25'], float).reshape(len(tcs), len(ms))
    prof = full.min(axis=1)
    i, j = np.unravel_index(np.argmin(full), full.shape)
    lec = N['cost']
    if int(np.sum((prof[1:-1] < prof[:-2]) & (prof[1:-1] < prof[2:]))) != lec['locmin'] or g.d2s(g.todate(tcs[i])) != lec['tc_best'] or ms[j] != lec['m_best']:
        raise SystemExit('cost landscape (rounded)')
    close(prof.max() / prof.min(), lec['ssr_ratio'], 1e-8, 'cost ssr_ratio (rounded)')
    return dict(tcs=sg(tcs, 15), ms=sg(ms, 15), wmax=[float(f'{ws[j]:g}') for j in keep], S=S, peak=g.d2s(peak), t1=g.d2s(low), t2=g.d2s(t2), n=len(t))


# ---------------------------------------------------------------- 6. indicator series and evaluation
def ci_csv(name):
    c = pd.read_csv(os.path.join(HERE, f'ch13_ci_{name}.csv'), index_col=0, parse_dates=True)
    return dict(d=iso(c.index), full=sg(c['ci_full'].values), param=sg(c['ci_param'].values), tcm=sg(c['tc_median'].values))


def indicator():
    out = {}
    for k in ('ndx', 'ssec', 'bet', 'btc17'):
        ci, s, peak = g.episode_ci(k)
        out[k] = dict(ci_csv(f'ep_{k}'), price=series(s, ci.index[0], ci.index[-1]), peak=g.d2s(peak))
    return out


def evaluation():
    out = {}
    for k in ('sp500', 'btc'):
        s, d = g.eval_ci(k)
        out[k] = dict(ci_csv(f'eval_{k}'), price=series(s, d.index[0]))
    return out


def main():
    chapter_results()
    d = dict(numbers=N, crash=g.CRASH, horizon=g.HORIZON, level=g.CI_LEVEL, search=g.LPPLS_SEARCH, filter=g.LPPLS_FILTER, lags=LAGS)
    d['ep'] = episodes()
    d['sim'] = draws()
    d['psy'] = psy_samples()
    d['fits'] = fits()
    d['win'] = windows()
    d['tcp'] = tc_paths()
    d['cost'] = cost()
    d['ci'] = indicator()
    d['ev'] = evaluation()
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 13 interactive page, exported by Quantlets/Ch_13/export_interactive.py.\n')
        fh.write('window.CH13_DATA = ' + json.dumps(J(d), separators=(',', ':')).replace('NaN', 'null') + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
