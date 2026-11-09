"""Export the inputs of the Chapter 8 interactive page (TSA/interactive/ch8_data.js).

Uses the functions, seeds and data loaders of generate_all_charts.py, so the page starts from the same data and numbers as
the slides. The series are read with the chapter's own loaders (Quantlets/common/tsa_data.py) and cut at the end dates of
the slides: US CPI inflation (FRED CPIAUCSL, up to August 2026) and the US unemployment rate (FRED UNRATE, up to September
2026), the Romanian HICP (Eurostat prc_hicp_minr, up to August 2026), the BNR EUR/RON reference rate and the saved daily
prices of data/market (up to 18 September 2026); the Nile flow comes with statsmodels. FRED has no October 2025 value of CPIAUCSL and UNRATE (not published); the dates of
the US series are therefore exported as they are. Downloaded on 4 November 2026, these
series reproduce every number of ch8_numbers.json exactly; the exporter stops if one of them drifts.
Exported: the series; the Normal draws of the circulant-embedding simulations (ARFIMA paths, fractional Brownian motion);
the sample ACF of the 20 000-observation ARFIMA simulations for other d (extra runs, same seeds); the estimates of every
Monte Carlo replication (estimators, breaks and regimes); the Monte Carlo bands of the rolling estimates for other window
lengths (extra runs); the out-of-sample forecast errors by origin; the Whittle fits and AR forecasts of US inflation for
other forecast origins (extra runs); the exact-ML model tables of Romanian inflation for other sample ends (extra runs).
No chart and no CSV of the chapter is written: save() is replaced by plt.close() and memory_table() is not called.
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_08/export_interactive.py   (reads FRED, Eurostat and the BNR; a few minutes)
"""
import json
import os
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
g.save = lambda name, save_it=True: plt.close('all')   # never overwrite the charts of the slides
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch8_data.js')
N = json.load(open(os.path.join(HERE, 'ch8_numbers.json')))
FRED_END = {'CPIAUCSL': '2026-08-01', 'UNRATE': '2026-09-01'}     # last month of the slides
EUROSTAT_END = ['2026-08-01']                                      # last month of the slides (changed only for extra runs)
DOWNLOADED = '2026-11-04'
TOL = 1e-9


def rd(x, d=6):
    return np.round(np.asarray(x, float), d).tolist()


def close(a, b, tol, what):
    if not (abs(a - b) <= tol * max(1.0, abs(b)) or (np.isnan(a) and np.isnan(b))):
        raise SystemExit(f'{what}: {a} differs from the lecture value {b}')


def cmp(a, b, tol, path, skip=()):
    """Every number of the lecture's result b equals the recomputed a (relative tolerance tol)."""
    if isinstance(b, dict):
        for k, v in b.items():
            if k not in skip:
                cmp(a[k], v, tol, f'{path}/{k}', skip)
    elif isinstance(b, list):
        if len(a) != len(b):
            raise SystemExit(f'{path}: length {len(a)} != {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, tol, f'{path}[{i}]', skip)
    elif isinstance(b, (bool, str)):
        if a != b:
            raise SystemExit(f'{path}: {a} != {b}')
    elif isinstance(b, (int, float)):
        close(float(a), float(b), tol, path)


# ---------------------------------------------------------------- the chapter's loaders, cut at the slides' end dates
_read_fred, _read_eurostat = g.read_fred, g.read_eurostat


def read_fred(ids, start=None, end=None):
    return _read_fred(ids, start).loc[:FRED_END[ids]].loc[:end]


def read_eurostat(dataset, key, start=None):
    return _read_eurostat(dataset, key, start).loc[:EUROSTAT_END[0]]


g.read_fred, g.read_eurostat = read_fred, read_eurostat


def ro_raw():
    return (100 * np.log(g.ro_hicp()).diff()).dropna().loc[g.INFL_START:]


def lecture_table():
    """The numbers of memory_table(), without writing ch8_memory_table.csv."""
    S = {'Nile flow': g.nile().values, 'RO inflation, monthly SA': g.ro_inflation().values,
         'RO inflation, 12-month': g.ro_inflation12().values, 'US inflation, monthly': g.us_inflation().values,
         'US inflation, 1985-2019': g.us_inflation('1985-01-01', '2019-12-01').values,
         'US unemployment': g.us_unemployment().values, 'US unemployment, change': g.us_unemployment_change().values}
    for k in ('sp500', 'bet', 'eurron'):
        r = g.returns(k)
        S[f'{g.NAME[k]} r'] = r.values
        S[f'{g.NAME[k]} |r|'] = np.abs(r.values)
    return {k: g.all_estimates(v) for k, v in S.items()}


# ---------------------------------------------------------------- 1. every result of the lecture
def check_lecture():
    for k, f in [('ex', g.worked_examples), ('data', g.fig_memory_data), ('acf', g.fig_memory_acf), ('decay', g.fig_decay),
                 ('weights', g.fig_weights), ('ffd', g.fig_ffd), ('paths', g.fig_arfima_paths), ('fbm', g.fig_fbm),
                 ('rsdfa', g.fig_rs_dfa), ('gph', g.fig_gph), ('bw', g.fig_bandwidth), ('fit', g.fig_arfima_fit),
                 ('fcpath', g.fig_forecast_path), ('vol', g.fig_vol_acf), ('rv', g.fig_rv), ('volm', g.fig_vol_models),
                 ('nile', g.fig_nile_break), ('table', lecture_table), ('rolling', g.fig_rolling)]:
        print('check', k, flush=True)
        cmp(f(), N[k], TOL, k)


# ---------------------------------------------------------------- 2. data
def iso(d):
    return str(d)[:10]


def gaps(idx):
    return np.diff(idx.values).astype('timedelta64[D]').astype(int).tolist()


def data_block():
    y = g.nile()
    raw, im, i12 = ro_raw(), g.ro_inflation(), g.ro_inflation12()
    sa, _ = g.seasonal_adjust(raw)
    assert np.allclose(sa.values + raw.mean(), im.values, atol=1e-12) and im.index.equals(i12.index)
    us, un = g.us_inflation(), g.us_unemployment()
    rv = {k: g.monthly_rv(k) for k in ('sp500', 'bet')}
    assert rv['sp500'].index.equals(rv['bet'].index)
    for z in (raw, rv['sp500']):            # contiguous months; FRED has no October 2025 (not published), so the US months are exported
        assert z.index.equals(pd.date_range(z.index[0], periods=len(z), freq='MS'))
    p = np.log(g.load_close('sp500', '2000-01-01'))
    r = g.returns('sp500')
    assert len(r) == len(p) - 1 and np.allclose(r.values, 100 * np.diff(p.values), atol=1e-10) and r.index.equals(p.index[1:])
    bet, eur = g.returns('bet'), g.returns('eurron')
    pk = g.parkinson_vol('sp500')
    perm = {}
    for k, x in (('sp500', r.values), ('bet', bet.values)):
        pi = np.random.default_rng(g.SEED + 9).permutation(len(x))
        assert np.array_equal(np.abs(x)[pi], np.random.default_rng(g.SEED + 9).permutation(np.abs(x)))
        perm[k] = pi.tolist()
    return dict(downloaded=DOWNLOADED,
                nile=dict(y0=int(y.index[0]), v=rd(y.values, 2)),
                ro=dict(m0=iso(raw.index[0]), raw=rd(raw.values, 8), i12=rd(i12.values, 8)),
                us=dict(d=[iso(d)[:7] for d in us.index], v=rd(us.values, 8), ud=[iso(d)[:7] for d in un.index], u=rd(un.values, 2)),
                rv=dict(m0=iso(rv['sp500'].index[0]), sp500=rd(rv['sp500'].values, 8), bet=rd(rv['bet'].values, 8)),
                sp=dict(d0=iso(p.index[0]), gap=gaps(p.index), lp=rd(p.values, 9)),
                bet=dict(d0=iso(bet.index[0]), gap=gaps(bet.index), r=rd(bet.values, 7)),
                eur=dict(r=rd(eur.values, 7)),
                park=dict(d0=iso(pk.index[0]), gap=gaps(pk.index), v=rd(pk.values, 7)),
                perm=perm)


# ---------------------------------------------------------------- 3. simulations
def circ_draws(n, seed):
    """The Normal draws of circulant_gaussian for a series of length n (M = 2(n - 1)), real and imaginary parts."""
    rng = np.random.default_rng(seed)
    M = 2 * (n - 1)
    re_, im_ = rng.standard_normal((1, M)), rng.standard_normal((1, M))
    return dict(re=rd(re_[0], 7), im=rd(im_[0], 7))


def sim_block(ds=np.round(np.arange(0.05, 0.451, 0.05), 2), nlags=50):
    """Draws of the paths (n = 1000, seeds SEED + 7, SEED + 8) and of the fBm (seed SEED); sample ACF of the 20 000-observation
    simulations for d and -d (seeds SEED + 17, SEED + 18; the lecture: d = 0.4, extra runs: other d)."""
    out = dict(path=[circ_draws(1000, g.SEED + 7), circ_draws(1000, g.SEED + 8)], fbm=circ_draws(1000, g.SEED), acf={})
    for d in ds:
        a = [g.acf(g.sim_arfima(20000, s * d, np.random.default_rng(g.SEED + 17 + row))[0], nlags) for row, s in ((0, 1), (1, -1))]
        out['acf'][f'{d:.2f}'] = [rd(v, 6) for v in a]
    for row, s in ((0, 1), (1, -1)):        # the draws reproduce the paths of the slide
        x = g.sim_arfima(1000, s * 0.4, np.random.default_rng(g.SEED + 7 + row))[0]
        close(float(x.std()), N['paths'][str(s * 0.4)]['sd'], TOL, 'path sd')
    return out


def mc_block(T=500, reps=g.MC_REPS):
    """fig_mc with every estimate kept (same seed, designs and order of the draws)."""
    designs = [('white noise, d = 0', 0.0, None), ('ARFIMA(0,0.3,0), d = 0.3', 0.3, None), ('AR(1), phi = 0.6, d = 0', 0.0, 0.6)]
    names = ['GPH', 'LW', 'Whittle', 'R/S', 'DFA']
    rng = np.random.default_rng(g.SEED + 41)
    sizes = g.block_sizes(T)
    res = []
    for lab, d, phi in designs:
        if phi is None:
            X = g.sim_arfima(T, d, rng, size=reps)
        else:
            e = rng.standard_normal((reps, T + 200))
            X = np.zeros_like(e)
            for t in range(1, T + 200):
                X[:, t] = phi * X[:, t - 1] + e[:, t]
            X = X[:, 200:]
        est = {n: [] for n in names}
        for x in X:
            est['GPH'].append(g.gph(x)['d'])
            est['LW'].append(g.local_whittle(x)['d'])
            est['Whittle'].append(g.arfima_whittle(x)['d'])
            est['R/S'].append(g.hurst_rs(x, sizes) - 0.5)
            est['DFA'].append(g.hurst_dfa(x, sizes) - 0.5)
        for n, v in est.items():
            L = N['mc']['res'][lab][n]
            close(float(np.mean(v)), L['mean'], TOL, f'mc {lab} {n} mean')
            close(float(np.std(v)), L['sd'], TOL, f'mc {lab} {n} sd')
        res.append(dict(lab=lab, d=d, v={n: rd(v, 6) for n, v in est.items()}))
    return dict(T=T, reps=reps, names=names, designs=res)


def spurious_block(T=1000, reps=200):
    """fig_spurious with the GPH and local Whittle estimate of every sample kept."""
    rng = np.random.default_rng(g.SEED + 21)
    br, ms = {}, {}
    for dl in g.SHIFTS:
        X = rng.standard_normal((reps, T))
        X[:, T // 2:] += dl
        br[str(dl)] = {'gph': rd([g.gph(v)['d'] for v in X]), 'lw': rd([g.local_whittle(v)['d'] for v in X])}
    for p in g.SWITCH:
        S = np.zeros((reps, T))
        s = rng.integers(0, 2, reps)
        U = rng.random((reps, T))
        for t in range(T):
            s = np.where(U[:, t] < p, 1 - s, s)
            S[:, t] = s
        X = S + rng.standard_normal((reps, T))
        ms[str(p)] = {'gph': rd([g.gph(v)['d'] for v in X]), 'lw': rd([g.local_whittle(v)['d'] for v in X]),
                      'sw': np.abs(np.diff(S, axis=1)).sum(axis=1).astype(int).tolist()}
    for k, v in br.items():
        for e in ('gph', 'lw'):
            close(float(np.mean(v[e])), N['spurious']['break'][k][e], 1e-6, f'spurious break {k} {e}')
    for k, v in ms.items():
        for e in ('gph', 'lw'):
            close(float(np.mean(v[e])), N['spurious']['switch'][k][e], 1e-6, f'spurious switch {k} {e}')
        close(float(np.mean(v['sw'])), N['spurious']['switch'][k]['switches'], TOL, f'switches {k}')
    return dict(T=T, reps=reps, shifts=g.SHIFTS, switch=g.SWITCH, br=br, ms=ms)


def bands_block(mult=(0.5, 0.75, 1.0, 1.25, 1.5)):
    """95% Monte Carlo bands of i.i.d. series for the rolling windows: the lecture (240 months, 1000 days) and extra runs
    with other window lengths (mc_band, seed SEED + n)."""
    out = {}
    for m in mult:
        wu, wd = int(round(240 * m)), int(round(1000 * m))
        bu = g.mc_band(wu, lambda v: g.local_whittle(v)['d'])
        bh = g.mc_band(wd, lambda v: g.hurst_dfa(v, g.block_sizes(wd)))
        if m == 1.0:
            cmp(bu, N['rolling']['us']['band'], TOL, 'band us')
            cmp(bh, N['rolling']['band_h'], TOL, 'band h')
        out[str(m)] = dict(wu=wu, wd=wd, bu=[bu['q025'], bu['q975']], bh=[bh['q025'], bh['q975']])
    return out


# ---------------------------------------------------------------- 4. forecasts and model tables
def oos_errors(series, start, H=24, refit=12, seasonal=False):
    """oos_inflation of the chapter with the forecast errors of every origin kept (NaN, exported as null: a month without
    a value, October 2025 for the US)."""
    idx = series.index
    mo = np.asarray(idx.year) * 12 + np.asarray(idx.month)
    origins = [t for t in range(len(series)) if idx[t] >= pd.Timestamp(start) and mo[-1] - mo[t] >= H]
    E = {m: [] for m in ('ARFIMA', 'AR', 'RW', 'Mean')}
    est = None
    for i, t in enumerate(origins):
        w = series.iloc[:t + 1]
        if seasonal:
            sa, means = g.seasonal_adjust(w)
            y = sa.values + w.mean()
            fut = pd.date_range(idx[t], periods=H + 1, freq='MS')[1:]
            add = means.reindex(fut.month).values - w.mean()
        else:
            y, add = w.values, np.zeros(H)
        if i % refit == 0:
            est = g.arfima_whittle(y, 1, 0)
        fa, p = g.ar_forecast(y, H)
        f = {'ARFIMA': g.arfima_forecast(y, est['d'], est['phi'], [], H), 'AR': fa, 'RW': np.repeat(y[-1], H), 'Mean': np.repeat(y.mean(), H)}
        for m in E:
            E[m].append(g.oos_errors_by_month(series, t, f[m] + add, H))
    return dict(o0=iso(idx[origins[0]]), n=len(origins), e={m: rd(np.array(e), 5) for m, e in E.items()},
                rmse={m: np.sqrt(np.nanmean(np.array(e) ** 2, axis=0)).tolist() for m, e in E.items()})


def fc_block():
    ro = oos_errors(ro_raw(), '2014-01-01', seasonal=True)
    us = oos_errors(g.us_inflation(), '1990-01-01')
    for k, R in (('ro', ro), ('us', us)):
        cmp(R['rmse'], N['fc'][k]['rmse'], TOL, f'fc {k}')
        assert R['n'] == N['fc'][k]['n']
        R.pop('rmse')
    return dict(ro=ro, us=us)


FC_ORIGINS = ('2019-12', '2020-12', '2021-06', '2021-12', '2022-06', '2022-12', '2023-12', '2024-12', '2025-08', '2026-08')


def fcpath_block(H=36):
    """Whittle ARFIMA(1,d,0) and the AR(p) forecasts of US inflation for other forecast origins (extra runs; the lecture:
    August 2026)."""
    y = g.us_inflation()
    out = {}
    for o in FC_ORIGINS:
        x = y.loc[:o + '-01'].values
        e = g.arfima_whittle(x, 1, 0)
        fr, p = g.ar_forecast(x, H)
        out[o] = dict(T=int(len(x)), d=float(e['d']), phi=float(e['phi'][0]), p=int(p), ar=rd(fr, 6))
    L = out['2026-08']
    for k, v in (('d', L['d']), ('phi', L['phi']), ('p', L['p']), ('fr1', L['ar'][0]), ('fr12', L['ar'][11]), ('fr36', L['ar'][35])):
        close(v, N['fcpath'][k], 1e-6, f'fcpath {k}')
    return dict(origins=list(FC_ORIGINS), H=H, m=out)


FIT_ENDS = ('2014-12', '2016-12', '2018-12', '2020-12', '2022-12', '2024-12', '2026-08')


def fit_block():
    """model_table of Romanian monthly inflation (seasonal adjustment inside the sample) for other sample ends
    (extra runs; the lecture: August 2026)."""
    out = {}
    for e in FIT_ENDS:
        EUROSTAT_END[0] = e + '-01'
        x = g.ro_inflation().values
        tab = g.model_table(x)
        out[e] = dict(T=int(len(x)), rows=[dict(name=t['name'], d=t['d'], phi=t['phi'], theta=t['theta'], ll=t['loglik'], aic=t['aic'],
                                                bic=t['bic']) for t in tab])
    EUROSTAT_END[0] = '2026-08-01'
    for a, b in zip(out['2026-08']['rows'], N['fit']['table']):
        for k, kb in (('d', 'd'), ('ll', 'loglik'), ('aic', 'aic'), ('bic', 'bic')):
            close(a[k], b[kb], TOL, f"fit {b['name']} {k}")
    return dict(ends=list(FIT_ENDS), m=out)


def mk_block():
    """MacKinnon (2010) coefficients of the ADF critical values (one series, constant), as statsmodels mackinnoncrit."""
    import statsmodels.tsa.adfvalues as a
    return dict(c=np.asarray(a.tau_2010s['c'])[0].tolist())


def main():
    check_lecture()
    d = dict(numbers=N)
    for k, f in [('mk', mk_block), ('data', data_block), ('sim', sim_block), ('mc', mc_block), ('spurious', spurious_block),
                 ('bands', bands_block), ('fc', fc_block), ('fcpath', fcpath_block), ('fit', fit_block)]:
        print(k, flush=True)
        d[k] = f()
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 8 interactive page, exported by Quantlets/Ch_08/export_interactive.py.\n')
        fh.write('window.CH8_DATA = ' + json.dumps(d, separators=(',', ':'), default=float).replace('NaN', 'null') + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
