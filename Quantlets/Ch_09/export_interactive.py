"""Export the inputs of the Chapter 9 interactive page (TSA/interactive/ch9_data.js).

Uses the functions, seeds and data loaders of generate_all_charts.py, so the page starts from the same data and numbers as
the slides. Data: the daily electricity load of Romania (the ENTSO-E extract of Chapter 4, Quantlets/Ch_04/ch4_ro_load_hourly.csv,
up to 30 June 2026), the saved daily prices of data/market (S&P 500, DAX, BET, up to 18 September 2026), the Romanian HICP
index (Eurostat prc_hicp_minr, read with the chapter's loader and cut at August 2026, the last month of the slides), the M4
evaluation file (ch9_m4_owa.csv) and the saved walk-forward forecasts of the chapter (ch9_load_ml.csv, ch9_inflation.csv and
Quantlets/Ch_04/ch4_load_cv.csv). The Eurostat index, downloaded on 4 November 2026 and cut at August 2026, reproduces the
numbers of the extrapolation slide exactly; the inflation forecasts are read from ch9_inflation.csv (the slide vintage).
Exported, with the seeds of the chapter: the load and its calendar flags; the draws of the bias-variance sample; the results
of the leakage experiment (per fold) for other horizons of the target (extra runs; the lecture: 21 days); the bias-variance
table for other noise levels (extra runs; the lecture: sigma = 0.3); the ridge and lasso paths for other sample ends (extra
runs; the lecture: 30 June 2025); the features of the depth-2 tree; the validation curves of the ensembles for other training
splits (extra runs); the random forests of the extrapolation slide for other training ends (extra runs); the permutation
importance for every horizon 1-14 (the lecture: 1 and 14); the LSTM of every forecast origin (training loss, forecasts, which
equal ch9_load_ml.csv); the 90% intervals of direct GB (and the conformal half-widths for other levels, extra runs); the
walk-forward volatility forecasts and the sign probabilities.
The exporter compares every recomputed number with ch9_numbers.json and stops if one of them differs (relative tolerance
1e-9 for recomputed results; 5e-5 for results computed from the saved CSV files, which hold 5 decimals).
No chart and no CSV of the chapter is written: st.save_fig is replaced by plt.close and every function is called with
save_csv=False / save=False or replicated without the writing step.
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_09/export_interactive.py   (reads Eurostat; about 30 minutes)
      CH9_CACHE=<dir> keeps the result of every part in <dir> (pickle) for a faster rerun.
"""
import json
import math
import os
import pickle
import sys
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from sklearn.ensemble import RandomForestRegressor                    # noqa: E402
from sklearn.inspection import permutation_importance                 # noqa: E402
from sklearn.linear_model import LassoCV, lasso_path                  # noqa: E402
from sklearn.model_selection import KFold, TimeSeriesSplit            # noqa: E402
from sklearn.preprocessing import StandardScaler                      # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
g.st.save_fig = lambda name, *a, **k: plt.close('all')     # never overwrite the charts of the slides
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch9_data.js')
N = json.load(open(os.path.join(HERE, 'ch9_numbers.json')))
EUROSTAT_END = '2026-08-01'          # last month of the slides
DOWNLOADED = '2026-11-04'
TOL, TOLCSV = 1e-9, 5e-5
CACHE = os.environ.get('CH9_CACHE')


def rd(x, d=6):
    a = np.round(np.asarray(x, float), d)
    return [None if not np.isfinite(v) else float(v) for v in np.ravel(a)] if a.ndim else float(a)


def close(a, b, tol, what):
    a, b = float(a), float(b)
    if not (abs(a - b) <= tol * max(1.0, abs(b)) or (np.isnan(a) and np.isnan(b))):
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
        if a != b and not (b is None and (a is None or (isinstance(a, float) and np.isnan(a)))):
            raise SystemExit(f'{path}: {a} != {b}')
    else:
        close(a, b, tol, path)


def cached(name, f):
    if CACHE:
        p = os.path.join(CACHE, name + '.pkl')
        if os.path.exists(p):
            return pickle.load(open(p, 'rb'))
        r = f()
        os.makedirs(CACHE, exist_ok=True)
        pickle.dump(r, open(p, 'wb'))
        return r
    return f()


# ---------------------------------------------------------------- the chapter's Eurostat loader, cut at the slides' last month
_read_eurostat = g.read_eurostat
g.read_eurostat = lambda dataset, key, start=None: _read_eurostat(dataset, key, start).loc[:EUROSTAT_END]

Y = g.load_daily()
iso = lambda d: str(pd.Timestamp(d).date())   # noqa: E731


# ---------------------------------------------------------------- 1. the load, its calendar, the lecture's simple results
def load_block():
    assert Y.index.equals(pd.date_range(Y.index[0], periods=len(Y), freq='D'))
    cal = g.calendar(pd.date_range(Y.index[0], Y.index[-1] + pd.Timedelta(days=14), freq='D'))
    cmp(g.fig_features(Y), N['features'], TOL, 'features')
    cmp(g.mini_table(Y), N['mini'], TOL, 'mini')
    return dict(d0=iso(Y.index[0]), v=rd(Y.values, 6), easter=cal['easter'].astype(int).tolist(), xmas=cal['xmas'].astype(int).tolist(),
                holiday=cal['holiday'].astype(int).tolist())


# ---------------------------------------------------------------- 2. validation: schemes, leakage, bias and variance
def leak_cv(X, splitter, feats=('m5', 'm21', 'm63', 'v21', 'v63'), seed=g.SEED):
    """g.cv_r2 with the sums of every fold kept: sum (y - yhat)^2 and sum (y - ybar)^2."""
    folds = []
    for tr, te in splitter.split(X):
        m = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.6, n_jobs=-1, random_state=seed)
        m.fit(X.iloc[tr][list(feats)], X['y'].iloc[tr])
        yt, yh = X['y'].iloc[te].values, m.predict(X.iloc[te][list(feats)])
        yb = X['y'].iloc[tr].mean()
        folds.append([int(len(te)), float(np.sum((yt - yh) ** 2)), float(np.sum((yt - yb) ** 2))])
    return folds


LEAK_H = (1, 5, 10, 21, 42, 63)


def leakage_block():
    """The leakage experiment of the lecture (h = 21) and extra runs with other horizons of the summed target."""
    rng = np.random.default_rng(g.SEED)
    sim = pd.Series(rng.normal(0, 1, 4000), index=pd.bdate_range('2000-01-03', periods=4000))
    sp = g.log_returns('sp500', '2000-01-01')
    out = {}
    for h in LEAK_H:
        out[str(h)] = {}
        for lab, r in (('sim', sim), ('sp500', sp)):
            X = g.leakage_frame(r, h)
            res = {'n': int(len(X))}
            for k, spl in (('kfold', KFold(5, shuffle=True, random_state=g.SEED)), ('wf', TimeSeriesSplit(5)), ('gap', TimeSeriesSplit(5, gap=h))):
                res[k] = leak_cv(X, spl)
            out[str(h)][lab] = res
        print('  leakage h', h, flush=True)
    for lab in ('sim', 'sp500'):
        L = out['21'][lab]
        assert L['n'] == N['leakage'][lab]['n']
        for k in ('kfold', 'wf', 'gap'):
            f = np.array(L[k])
            close(1 - f[:, 1].sum() / f[:, 2].sum(), N['leakage'][lab][k], TOL, f'leakage {lab} {k}')
    return dict(h=list(LEAK_H), res=out, sp=dict(d0=iso(sp.index[0]), r=rd(sp.values, 5)), sim=rd(sim.values, 6))


BV_SIGMA = tuple(np.round(np.arange(0.1, 0.61, 0.05), 2))


def bv_block():
    """bias_variance for the lecture's sigma = 0.3 and extra runs with other noise levels; the draws of the sample of the
    left panel (seed SEED + 1): x and the standard Normal draws z (y = sin(2 pi x) + sigma z)."""
    tab = {}
    for s in BV_SIGMA:
        bv = g.bias_variance(sigma=float(s))
        tab[f'{s:.2f}'] = [[bv[d][k] for k in ('bias2', 'var', 'test', 'train')] for d in range(1, 11)]
    L = N['bv']
    for d in (1, 3, 10):
        for j, k in enumerate(('bias2', 'var', 'test', 'train')):
            close(tab['0.30'][d - 1][j], L[f'd{d}'][k], TOL, f'bv d{d} {k}')
    rng = np.random.default_rng(g.SEED + 1)
    x = rng.uniform(0, 1, 80)
    z = rng.standard_normal(80)
    r2 = np.random.default_rng(g.SEED + 1)
    x2 = r2.uniform(0, 1, 80)
    assert np.array_equal(x, x2) and np.allclose(r2.normal(0, 0.3, 80), 0.3 * z, rtol=0, atol=1e-15)
    return dict(sigma=[float(s) for s in BV_SIGMA], tab=tab, x=rd(x, 15), z=rd(z, 15))


# ---------------------------------------------------------------- 3. ridge and lasso
SHRINK_ENDS = ('2023-06-30', '2023-12-31', '2024-06-30', '2024-12-31', '2025-06-30', '2025-12-31', '2026-06-29')


def shrink_full(end):
    """shrinkage_paths of the chapter, with the coefficients at the chosen penalty kept."""
    F = g.direct_frame(Y.loc[:end], 1).dropna()
    Z = StandardScaler().fit_transform(F[g.FEATS].values)
    t = F['y'].values - F['y'].mean()
    alphas_l, coefs_l, _ = lasso_path(Z, t, n_alphas=60, eps=1e-4)
    lams = np.logspace(-2, 5, 60)
    coefs_r = np.array([np.linalg.solve(Z.T @ Z + l * np.eye(Z.shape[1]), Z.T @ t) for l in lams]).T
    cv = LassoCV(cv=TimeSeriesSplit(5), n_alphas=60, eps=1e-4).fit(Z, t)
    entry = {}
    for j, f in enumerate(g.FEATS):
        nz = np.flatnonzero(np.abs(coefs_l[j]) > 1e-10)
        entry[f] = float(alphas_l[nz[0]]) if len(nz) else 0.0
    return dict(n=int(len(F)), alphas=rd(alphas_l, 9), lasso=[rd(c, 6) for c in coefs_l], lams=rd(lams, 9), ridge=[rd(c, 6) for c in coefs_r],
                alpha_cv=float(cv.alpha_), coef_cv=rd(cv.coef_, 8), entry=[entry[f] for f in g.FEATS],
                cvmse=rd(cv.mse_path_.mean(axis=1), 8), cvalphas=rd(cv.alphas_, 9))


def shrink_block():
    out = {e: shrink_full(e) for e in SHRINK_ENDS}
    S, L = g.shrinkage_paths(Y), out['2025-06-30']
    close(L['alpha_cv'], N['shrink']['alpha_cv'], TOL, 'alpha_cv')
    assert L['n'] == N['shrink']['n'] and S['first3'] == N['shrink']['first3'] and S['kept'] == N['shrink']['kept']
    order = sorted(range(30), key=lambda j: -L['entry'][j])[:3]
    assert [g.FEATS[j] for j in order] == N['shrink']['first3']
    for f, v in N['shrink']['top']:
        close(L['coef_cv'][g.FEATS.index(f)], v, 1e-7, f'coef {f}')
    return dict(feats=g.FEATS, ends=list(SHRINK_ENDS), m=out)


# ---------------------------------------------------------------- 4. trees and ensembles
def tree_block():
    """The features of the depth-2 tree (up to the last row of the load), as in fig_tree."""
    cmp(g.fig_tree(Y), N['tree'], TOL, 'tree')
    F = g.direct_frame(Y, 1).dropna()
    F = F.assign(weekend=1 - F[['mon', 'tue', 'wed', 'thu', 'fri']].sum(axis=1))
    cols = ['lag0', 'same_dow', 'weekend', 'holiday', 'xmas', 'easter']
    return dict(d0=iso(F.index[0]), cols=cols, X=[rd(F[c].values, 9) for c in cols], y=rd(F['y'].values, 9))


ENS_SPLITS = ('2023-12-31', '2024-06-30', '2024-12-31')


def ens_block():
    out = {}
    for s in ENS_SPLITS:
        E = g.ensemble_curves(Y, split=s)
        out[s] = dict(tree=rd(E['tree'], 8), rf=rd(E['rf'], 8), gb03=rd(E['gb0.3'], 8), gb005=rd(E['gb0.05'], 8), naive=E['naive'], n_tr=E['n_tr'], n_va=E['n_va'])
        print('  ensembles', s, flush=True)
    E = g.ensemble_curves(Y)
    cmp(g.fig_ensembles(E), N['ens'], TOL, 'ens')
    assert np.allclose(E['rf'], out['2024-12-31']['rf'], atol=1e-8)
    return dict(splits=list(ENS_SPLITS), m=out)


EXTRA_SPLITS = ('2018-12-01', '2019-12-01', '2020-12-01', '2021-12-01', '2022-12-01')


def extrapolation(split, seed=g.SEED):
    """fig_extrapolation without the chart: the forecasts of the two random forests."""
    p = g.hicp('RO').loc['2005-01-01':]
    lp = np.log(p)
    lv = pd.DataFrame({f'l{j}': p.shift(j) for j in range(1, 13)}).assign(y=p)
    ch = pd.DataFrame({f'l{j}': 100 * lp.diff().shift(j) for j in range(1, 13)}).assign(y=100 * lp.diff())
    lv, ch = lv.dropna(), ch.dropna()
    tr_l, tr_c = lv.loc[:split], ch.loc[:split]
    rf_l = RandomForestRegressor(300, min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(tr_l.drop(columns='y'), tr_l['y'])
    rf_c = RandomForestRegressor(300, min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(tr_c.drop(columns='y'), tr_c['y'])
    te = lv.index[lv.index > split]
    f_l = rf_l.predict(lv.loc[te].drop(columns='y'))
    f_c = p.shift(1).loc[te].values * np.exp(rf_c.predict(ch.loc[te].drop(columns='y')) / 100)
    return p, te, f_l, f_c, float(tr_l['y'].max())


def extra_block():
    cmp(g.fig_extrapolation(), N['extra'], TOL, 'extra')
    out = {}
    for s in EXTRA_SPLITS:
        p, te, f_l, f_c, mx = extrapolation(s)
        out[s[:7]] = dict(t0=iso(te[0]), fl=rd(f_l, 5), fc=rd(f_c, 5), max=mx)
    p, te, f_l, f_c, mx = extrapolation('2020-12-01')
    close(np.sqrt(np.mean((p.loc[te].values - f_l) ** 2)), N['extra']['rmse_l'], TOL, 'extra rmse_l')
    close(np.sqrt(np.mean((p.loc[te].values - f_c) ** 2)), N['extra']['rmse_c'], TOL, 'extra rmse_c')
    assert p.index.equals(pd.date_range(p.index[0], periods=len(p), freq='MS'))
    return dict(m0=iso(p.index[0]), p=rd(p.values, 4), splits=[s[:7] for s in EXTRA_SPLITS], m=out, downloaded=DOWNLOADED)


def imp_block(seed=g.SEED):
    """importance() of the chapter for every horizon h = 1..14 (the lecture: 1 and 14)."""
    out = {}
    for h in range(1, 15):
        F = g.direct_frame(Y.loc[:g.CV_FIRST], h).dropna()
        tr = F.loc[:pd.Timestamp('2024-12-31') - pd.Timedelta(days=h)]
        te = F.loc['2025-01-01':]
        m = g.make_gb(seed).fit(tr[g.FEATS], tr['y'])
        pi = permutation_importance(m, te[g.FEATS], te['y'], n_repeats=20, random_state=seed, scoring='neg_mean_absolute_error')
        out[str(h)] = rd(pi.importances_mean, 8)
    for h in ('1', '14'):
        for f, v in N['imp'][h].items():
            close(out[h][g.FEATS.index(f)], v, 1e-7, f'imp {h} {f}')
    return dict(feats=g.FEATS, m=out)


# ---------------------------------------------------------------- 5. the LSTM of every origin, the saved forecasts
def lstm_block():
    E = g.saved_load_ml()
    out = []
    for o in g.origins(Y):
        f, loss = g.forecast_lstm(Y, o)
        s = E[(E['model'] == 'LSTM') & (E['origin'] == o)].sort_values('h')['fc'].values
        if not np.allclose(np.round(f, 5), s, atol=1.1e-5):
            raise SystemExit(f'LSTM {o.date()}: the forecasts differ from ch9_load_ml.csv')
        out.append(rd(loss, 6))
        print('  lstm', o.date(), flush=True)
    close(out[0][0], N['load']['lstm_loss'][0], 1e-6, 'lstm loss 0')
    close(out[0][-1], N['load']['lstm_loss'][1], 1e-6, 'lstm loss end')
    return dict(loss=out)


def errors_block():
    """The walk-forward errors of the eleven models (ch9_load_ml.csv and Quantlets/Ch_04/ch4_load_cv.csv)."""
    E_ml, E_st = g.saved_load_ml(), g.ch4_errors()
    tab, dm, scale, byh = g.load_summary(E_ml, E_st, Y)
    for m, r in N['load']['tab'].items():
        for k, v in r.items():
            close(tab.loc[m, k], v, TOLCSV, f'load {m} {k}')
    for k, v in N['load']['dm'].items():
        for q in ('dm', 'hln'):
            close(dm[k][q], v[q], TOLCSV * 10, f'dm {k} {q}')
        close(dm[k]['p'], v['p'], TOLCSV, f'dm {k} p')
    close(scale, N['load']['scale'], TOL, 'scale')
    E = pd.concat([E_st[E_st['model'].isin(['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'Combination'])], E_ml])
    os_ = sorted(E['origin'].unique())
    models = list(tab.index)
    fc = {}
    for m in models:
        e = E[E['model'] == m].sort_values(['origin', 'h'])
        assert len(e) == 26 * 14 and (pd.DatetimeIndex(e['origin']) == pd.DatetimeIndex(np.repeat(pd.DatetimeIndex(os_), 14))).all()
        fc[m] = rd(e['fc'].values, 5)
    eo = E[E['model'] == 'DHR'].sort_values(['origin', 'h'])
    assert np.allclose(eo['actual'].values, Y.reindex(eo['date']).values, atol=1e-5)
    return dict(origins=[iso(o) for o in os_], models=models, fc=fc, n_train0=N['load']['n_train0'])


def intervals_block(level=0.9, n_cal=182, seed=g.SEED):
    """load_intervals of the chapter, with the conformal half-widths for other levels (extra runs) and the sorted calibration
    errors of the first origin (worked example)."""
    frames = {h: g.direct_frame(Y, h) for h in range(1, g.CV_H + 1)}
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
    levels = np.round(np.arange(0.5, 0.951, 0.05), 2)
    rows, Q, first = [], [], []
    for i, o in enumerate(g.origins(Y)):
        for h in range(1, g.CV_H + 1):
            F = frames[h]
            tr = F.loc[:o - pd.Timedelta(days=h)].dropna()
            x0 = F.loc[[o], g.FEATS].values
            point = g.make_gb(seed).fit(tr[g.FEATS].values, tr['y'].values).predict(x0)[0]
            ql = g.make_gb(seed, loss='quantile', quantile=lo_q).fit(tr[g.FEATS].values, tr['y'].values).predict(x0)[0]
            qh = g.make_gb(seed, loss='quantile', quantile=hi_q).fit(tr[g.FEATS].values, tr['y'].values).predict(x0)[0]
            fit, cal = tr.iloc[:-n_cal], tr.iloc[-n_cal:]
            mc = g.make_gb(seed).fit(fit[g.FEATS].values, fit['y'].values)
            res = np.sort(np.abs(cal['y'].values - mc.predict(cal[g.FEATS].values)))
            k = int(np.ceil((n_cal + 1) * level)) - 1
            pc = mc.predict(x0)[0]
            rows.append([point, ql, qh, pc, res[min(k, n_cal - 1)]])
            Q.append([res[min(int(np.ceil((n_cal + 1) * a)) - 1, n_cal - 1)] for a in levels])
            if i == 0:
                first.append(rd(res, 5))
        print('  intervals', o.date(), flush=True)
    R = np.array(rows)
    act = np.array([Y.get(o + pd.Timedelta(days=h), np.nan) for o in g.origins(Y) for h in range(1, 15)])
    for k, (a, b) in {'quantile': (R[:, 1], R[:, 2]), 'conformal': (R[:, 3] - R[:, 4], R[:, 3] + R[:, 4])}.items():
        close(np.mean((act >= a) & (act <= b)), N['int'][k]['cover'], TOL, f'int {k} cover')
        close(np.mean(b - a), N['int'][k]['width'], TOL, f'int {k} width')
    return dict(levels=[float(a) for a in levels], point=rd(R[:, 0], 6), qlo=rd(R[:, 1], 6), qhi=rd(R[:, 2], 6), cpoint=rd(R[:, 3], 6),
                q=[rd(q, 6) for q in Q], first=first, n_cal=n_cal)


# ---------------------------------------------------------------- 6. inflation (saved forecasts) and the Romanian HICP
def inflation_block():
    E = pd.read_csv(os.path.join(HERE, 'ch9_inflation.csv'), parse_dates=['origin', 'target'])
    S = g.inflation_summary(E)
    for h in ('1', '3', '6', '12'):
        L, R = N['inf']['sum'][h], S[int(h)]
        for m, v in L['rmse'].items():
            close(R['rmse'][m], v, TOLCSV, f'inf {h} rmse {m}')
        for grp in ('dm_rw', 'dm_ar'):
            for m, v in L[grp].items():
                close(R[grp][m]['p'], v['p'], TOLCSV, f'inf {h} {grp} {m}')
        close(R['dm_gl']['p'], L['dm_gl']['p'], TOLCSV, f'inf {h} dm_gl')
        assert R['n'] == L['n']
    p = g.hicp('RO')
    pi = (100 * (p / p.shift(12) - 1)).loc['2015-01-01':]
    a = E[(E['h'] == 1) & (E['model'] == 'RW')].set_index('target')['actual']
    assert np.allclose(pi.reindex(a.index).values, a.values, atol=1e-5), 'the HICP of today differs from the slide vintage'
    cmp({k: v for k, v in g.fig_inflation(E, pd.DataFrame({'RO': pi}), S).items()}, N['inf']['fig'], 1e-7, 'inf fig')
    models = ['RW', 'AR', 'Lasso', 'RF', 'GB local', 'GB global']
    out = {}
    for h in (1, 3, 6, 12):
        e = E[E['h'] == h]
        piv = e.pivot(index='origin', columns='model', values='fc')
        act = e[e['model'] == 'RW'].set_index('origin')['actual'].reindex(piv.index)
        out[str(h)] = dict(o0=iso(piv.index[0]), act=rd(act.values, 5), fc={m: rd(piv[m].values, 5) for m in models})
        assert piv.index.equals(pd.date_range(piv.index[0], periods=len(piv), freq='MS'))
    return dict(models=models, m=out, pi0=iso(pi.index[0]), pi=rd(pi.values, 5), n_countries=N['inf']['n_countries'], downloaded=DOWNLOADED)


# ---------------------------------------------------------------- 7. volatility and the sign of returns
def rv_block():
    out = {}
    for k in ('sp500', 'dax'):
        X, P = g.rv_forecasts(k)
        cmp(g.rv_metrics(X, P), N['rv'][k], TOL, f'rv {k}')
        idx = P['HAR'].index
        yrs = sorted(set(idx.year))
        out[k] = dict(d=[iso(d) for d in idx], y=rd(X.loc[idx, 'y'].values, 6), rv=rd(X.loc[idx, 'rv'].values, 6),
                      pred={m: rd(f['pred'].values, 6) for m, f in P.items() if m != 'Mean'},
                      mean=[float(P['Mean'].loc[str(yr), 'pred'].iloc[0]) for yr in yrs],
                      s2={m: [float(f.loc[str(yr), 's2'].iloc[0]) for yr in yrs] for m, f in P.items()}, years=yrs)
        print('  rv', k, flush=True)
    return out


def sign_block():
    out, S = {}, {}
    for k in ('sp500', 'bet'):
        X, prob, base = g.sign_walk_forward(k)
        S[k] = g.sign_metrics(X, prob, base)
        yrs = sorted(set(X.index.year))
        out[k] = dict(d=[iso(d) for d in X.index], y=X['y'].astype(int).tolist(), p={m: rd(v.values, 7) for m, v in prob.items()},
                      base=[int(base.loc[str(yr)].iloc[0]) for yr in yrs], years=yrs)
        print('  sign', k, flush=True)
    cmp(S, N['sign'], TOL, 'sign')
    for k in out:   # the rounded probabilities give the same accuracy and AUC
        from sklearn.metrics import roc_auc_score
        y = np.array(out[k]['y'])
        for m, p in out[k]['p'].items():
            p = np.array(p)
            close(np.mean((p >= 0.5) == y), N['sign'][k][m]['acc'], TOL, f'sign {k} {m} acc (rounded)')
            close(roc_auc_score(y, p), N['sign'][k][m]['auc'], 1e-6, f'sign {k} {m} auc (rounded)')
    return out


def m4_block():
    cmp(g.fig_m4(), N['m4'], TOL, 'm4')
    M = g.m4_table()
    return dict(method=M['method'].astype(str).tolist(), team=M['team'].tolist(), type=M['type'].tolist(), smape=rd(M['smape'], 6),
                mase=rd(M['mase'], 6), owa=rd(M['owa'], 9))


def main():
    d = dict(numbers=N, gkx=dict(models=g.GKX_MODELS, r2=g.GKX_R2, sr=g.GKX_SR), perm48=np.random.default_rng(3).permutation(48).tolist())
    for k, f in [('load', load_block), ('leak', leakage_block), ('bv', bv_block), ('shrink', shrink_block), ('tree', tree_block), ('ens', ens_block),
                 ('extra', extra_block), ('imp', imp_block), ('err', errors_block), ('lstm', lstm_block), ('int', intervals_block),
                 ('inf', inflation_block), ('rv', rv_block), ('sign', sign_block), ('m4', m4_block)]:
        print(k, flush=True)
        d[k] = cached(k, f)
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 9 interactive page, exported by Quantlets/Ch_09/export_interactive.py.\n')
        fh.write('window.CH9_DATA = ' + json.dumps(d, separators=(',', ':'), default=float).replace('NaN', 'null') + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
