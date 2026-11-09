"""Export the inputs of the Chapter 4 interactive page (TSA/interactive/ch4_data.js).

Uses the functions, seeds and data of generate_all_charts.py, so the page starts from the same data and numbers as the
slides. Romanian GDP (not adjusted and SCA): Eurostat has revised both series since the charts were made (the 2024 total by
almost 4 bn EUR), so the slide versions are read back from the vector paths of charts/tsa_ch4_series.pdf and
charts/tsa_ch4_sa_nsa.pdf and every GDP function of the chapter is run on them. The monthly Eurostat series stop at the last
month of the slides (retail trade, industrial production and tourism: July 2026; HICP: August 2026). Electricity load: the
ENTSO-E extract of the chapter (ch4_ro_load_hourly.csv); the cross-validation forecasts are the per-origin results of the
lecture (ch4_load_cv.csv). Results that need statsmodels or Prophet (SARIMA fits, MSTL, Prophet components) are computed
here on the grid of parameters the page offers; every value of the slides is checked against ch4_numbers.json before the
file is written. No chart is written: the chapter's save() is replaced by plt.close().
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_04/export_interactive.py   (needs internet for Eurostat; a few minutes)
      ... export_interactive.py --extra   only the inputs of the result tables and worked examples, merged into ch4_data.js
"""
import json
import logging
import os
import re
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)
g.save = lambda name, save_it=True: plt.close('all')   # never overwrite the charts of the slides
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch4_data.js')
N = json.load(open(os.path.join(HERE, 'ch4_numbers.json')))


def r(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


def close(a, b, tol, what):
    assert abs(a - b) <= tol, (what, a, b)


def runs(chart):
    """Polylines (x, y in PDF points) of a chart of the slides with at least 21 vertices."""
    import pymupdf
    doc = pymupdf.open(os.path.join(ROOT, 'charts', chart + '.pdf'))
    s = b''.join(doc.xref_stream(x) for x in doc[0].get_contents()).decode('latin1')
    R = re.findall(r'(-?[\d.]+ -?[\d.]+ m\n(?:-?[\d.]+ -?[\d.]+ l\n){20,})', s)
    return [np.array([[float(v) for v in ln.split()[:2]] for ln in q.strip().split('\n')]) for q in R]


# ---------------------------------------------------------------- GDP in the version of the slides
def gdp_slide_vintage():
    """Real GDP, not adjusted (million EUR, 2010 prices), 2000Q1-2026Q2, read from the first panel of tsa_ch4_series.pdf:
    the vertical scale is fixed by the four quarter means and the last value in ch4_numbers.json, then rounded to the
    0.1 million EUR of Eurostat. SCA, 2010Q1-2026Q2, from the GDP panel of tsa_ch4_sa_nsa.pdf, whose NSA line fixes the
    scale."""
    u = runs('tsa_ch4_series')[0][:, 1]
    assert len(u) == N['series']['gdp_T']
    q = np.arange(len(u)) % 4
    A = [[1, u[q == k].mean()] for k in range(4)] + [[1, u[-1]]]
    b = N['series']['gdp_q'] + [N['gdpfc']['last_obs']]
    c = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    nsa = np.round(1000 * (c[0] + c[1] * u), 1)
    idx = pd.date_range('2000-01-01', periods=len(nsa), freq='QS')
    nsa = pd.Series(nsa, index=idx, name='OBS_VALUE')
    sa = runs('tsa_ch4_sa_nsa')
    n0, s0 = sa[0][:, 1], sa[1][:, 1]
    tgt = nsa.loc['2010':].values / 1000
    c2 = np.polyfit(n0, tgt, 1)
    assert np.max(np.abs(np.polyval(c2, n0) - tgt)) < 2e-3
    sca = pd.Series(np.round(1000 * np.polyval(c2, s0), 1), index=nsa.loc['2010':].index, name='OBS_VALUE')
    return nsa, sca


NSA, SCA = gdp_slide_vintage()
_read = g.read_eurostat
LAST = {g.RETAIL: '2026-07-01', g.RETAIL_SCA: '2026-07-01', g.FOOD: '2026-07-01', g.IPI: '2026-07-01', g.TOUR: '2026-07-01',
        g.HICP: '2026-08-01'}


def read_slides(dataset, key, start=None):
    k = (dataset, key)
    if k == g.GDP_NSA:
        return NSA.copy()
    if k == g.GDP_SCA:
        return SCA.copy()
    s = _read(dataset, key, start)
    return s.loc[:LAST[k]] if k in LAST else s


g.read_eurostat = read_slides


def mq(s):
    return s.index[0].strftime('%Y-%m')


# ---------------------------------------------------------------- 1-2. seasonal data, seasonal plots
def series_block():
    s = g.fig_series(False)
    for k in ('gdp_T', 'gdp_last', 'ret_last', 'tour_last', 'infl_last', 'load_T', 'tour_max_m', 'tour_min_m'):
        assert s[k] == N['series'][k], k
    close(s['gdp_q4_q1'], N['series']['gdp_q4_q1'], 1e-5, 'q4/q1')
    close(s['tour_aug_jan'], N['series']['tour_aug_jan'], 1e-9, 'tour')
    assert np.allclose(s['infl_month_mean'], N['series']['infl_month_mean'])
    sp = g.fig_seasonal_plot(False)
    for k, v in sp.items():
        close(v, N['splot'][k], 1e-9, k)
    ret = g.read_eurostat(*g.RETAIL).loc['2010':]
    ipi = g.read_eurostat(*g.IPI).loc['2010':]
    tour = g.read_eurostat(*g.TOUR).loc['2010':] / 1e6
    infl = 100 * np.log(g.read_eurostat(*g.HICP)).diff().loc['2010':]
    return dict(ret=dict(m0=mq(ret), v=r(ret, 2)), ipi=dict(m0=mq(ipi), v=r(ipi, 2)), tour=dict(m0=mq(tour), v=r(tour, 6)),
                infl=dict(m0=mq(infl), v=r(infl, 4)))


# ---------------------------------------------------------------- 4. GDP differences
def gdpdiff_block():
    d = g.fig_gdp_differencing(save_it=False)
    for k in ('d1', 'd4', 'dd'):
        assert np.allclose(d[k]['r'], N['gdpdiff'][k]['r'], atol=1.5e-4), k
        close(d[k]['sd'], N['gdpdiff'][k]['sd'], 2e-4, k)
    h = g.seasonal_tests(save_csv=False)
    for k in ('t1', 't2', 'F34'):
        close(h['hegy'][k], N['tests']['hegy'][k], 5e-3, k)
    return dict(hegy=h['hegy'], hegy_k=h['hegy_k'])


# ---------------------------------------------------------------- 6-7. airline
def airline_block():
    a = g.fig_airline(save_it=False)
    for k in ('r1', 'r12', 'r3', 'band'):
        close(a[k], N['air'][k], 1e-9, k)
    fc = g.fig_airline_forecast(save_it=False)
    for k in ('theta', 'Theta', 'mape_sarima', 'mape_snaive', 'f_last', 'lo_last', 'hi_last'):
        close(fc[k], N['airfc'][k], 1e-6, k)
    y = g.airline()
    ly = np.log(y)
    org = {}
    for yr in range(1952, 1959):
        tr = ly.loc[:f'{yr}-12-01']
        m = g.fit_sarima(tr, (0, 1, 1), (0, 1, 1, 12))
        H = len(ly) - len(tr)
        f = m.get_forecast(H)
        ci = f.conf_int(alpha=0.05)
        org[str(yr)] = dict(th=float(m.params[0]), Th=float(m.params[1]), th_se=float(m.bse[0]), Th_se=float(m.bse[1]),
                            s=float(np.sqrt(m.params[2])), f=r(np.exp(f.predicted_mean), 3), lo=r(np.exp(ci[:, 0]), 3), hi=r(np.exp(ci[:, 1]), 3))
    close(org['1958']['f'][-1], N['airfc']['f_last'], 1e-3, 'air f')
    return dict(y=r(y, 0), org=org)


# ---------------------------------------------------------------- 8-9. GDP models
def gdp_models_block():
    t = g.gdp_models(save_csv=False)
    assert t['best_aicc'] == N['gdpm']['best_aicc'] and t['best_bic'] == N['gdpm']['best_bic']
    y = 100 * g.eurostat_log(g.GDP_NSA, g.GDP_START)
    out = {}
    for row, ref in zip(t['rows'], N['gdpm']['rows']):
        assert row['model'] == ref['model']
        close(row['aicc'], ref['aicc'], 0.05, row['model'])
        close(row['lb8_p'], ref['lb8_p'], 2e-3, row['model'])
        o, so = g.parse_model(row['model'])
        res = g.fit_sarima(y, o, so + (4,))
        e = res.resid[5:]
        out[row['model']] = dict(k=row['k'], aicc=row['aicc'], bic=row['bic'], lb8_p=row['lb8_p'], sigma=row['sigma'],
                                 par={k: [v, row['se'][k]] for k, v in row['params'].items() if k != 'sigma2'}, e=r(e, 4))
    dg = g.fig_gdp_diag((0, 1, 0), (0, 1, 1), save_it=False)
    close(dg['lb8']['lb'], N['gdpdiag']['lb8']['lb'], 0.02, 'lb8')
    close(dg['jb'], N['gdpdiag']['jb'], 0.1, 'jb')
    close(dg['out_v'], N['gdpdiag']['out_v'], 0.01, 'out')
    fc = g.fig_gdp_forecast((0, 1, 0), (0, 1, 1), save_it=False)
    assert np.allclose(fc['f'], N['gdpfc']['f'], atol=2e-3), (fc['f'], N['gdpfc']['f'])
    assert np.allclose(fc['lo95'], N['gdpfc']['lo95'], atol=3e-3)
    close(fc['params']['ma.S.L4'], N['gdpfc']['params']['ma.S.L4'], 5e-4, 'Theta')
    return dict(models=out, fc=dict(Th=fc['params']['ma.S.L4'], Th_se=fc['se']['ma.S.L4'], s=fc['sigma'], f=fc['f'], lo=fc['lo95'], hi=fc['hi95']))


# ---------------------------------------------------------------- 10. NSA and SCA
def sa_block():
    s = g.fig_sa_nsa(save_it=False)
    assert np.allclose(s['fq'], N['sa']['fq'], atol=1e-4)
    assert np.allclose(s['qoq_nsa'], N['sa']['qoq_nsa'], atol=0.011) and np.allclose(s['qoq_sca'], N['sa']['qoq_sca'], atol=0.011)
    close(s['sum_nsa_2024'], N['sa']['sum_nsa_2024'], 0.01, 'nsa 2024')
    close(s['sum_sca_2024'], N['sa']['sum_sca_2024'], 0.01, 'sca 2024')
    close(s['f_dec'], N['sa']['f_dec'], 1e-9, 'f_dec')
    rr, rs = g.read_eurostat(*g.RETAIL).loc['2015':], g.read_eurostat(*g.RETAIL_SCA).loc['2015':]
    return dict(gdp=r(NSA.values, 1), sca=r(SCA.values, 1), ret=dict(m0=mq(rr), v=r(rr, 2)), ret_sca=dict(m0=mq(rs), v=r(rs, 2)))


# ---------------------------------------------------------------- 11-12. Fourier, Easter
def fourier_block():
    f = g.fig_fourier(save_it=False)
    for K in (1, 2, 3, 6):
        close(f[f'K{K}']['r2'], N['fourier'][f'K{K}']['r2'], 1e-9, K)
    x = g.eurostat_log(g.TOUR, '2012-01-01').loc[:'2019-12-01']
    z = (x - x.rolling(13, center=True).mean()).dropna()
    pat = z.groupby(z.index.month).mean()
    return dict(pat=r(pat - pat.mean(), 6))


def easter_block():
    e = g.fig_easter(save_it=False)
    for k in ('easter', 'wd', 'ao_2020_04'):
        close(e['food']['params'][k], N['easter']['food']['params'][k], 1e-4, k)
        close(e['total']['params'][k], N['easter']['total']['params'][k], 1e-4, k)
    close(e['slope'], N['easter']['slope'], 1e-6, 'slope')
    reg = {}
    for w in range(1, 21):
        a, b = g.easter_regression(g.FOOD, w=w), g.easter_regression(g.RETAIL, w=w)
        reg[str(w)] = dict(food=[a['params']['easter'], a['se']['easter'], a['params']['wd'], a['se']['wd']],
                           total=[b['params']['easter'], b['se']['easter'], b['params']['wd'], b['se']['wd']])
    close(reg['10']['food'][0], N['easter']['food']['params']['easter'], 1e-4, 'w10')
    for k in ('food', 'total'):                                  # w = 10: the fit of the slide
        reg['10'][k] = [e[k]['params']['easter'], e[k]['se']['easter'], e[k]['params']['wd'], e[k]['se']['wd']]
    y = g.eurostat_log(g.FOOD, '2009-01-01')
    eas = {str(yr): str(pd.Timestamp(g.easter(yr, g.EASTER_ORTHODOX)).date()) for yr in range(2000, 2031)}
    return dict(food=dict(m0=mq(y), v=r(y, 6)), eas=eas, reg=reg)


# ---------------------------------------------------------------- 13-15. electricity load
def load_block():
    h = g.fig_load_hourly(save_it=False)
    for k in ('r12', 'r24', 'r168', 'peak_v', 'trough_v'):
        close(h[k], N['hourly'][k], 1e-9, k)
    m = g.fig_mstl(save_it=False)
    for k in ('amp24', 'amp168', 'share24', 'share168', 'share_r'):
        close(m[k], N['mstl'][k], 1e-9, k)
    dl = g.fig_load_daily(save_it=False)
    assert np.allclose(dl['wd'], N['daily']['wd'])
    s = g.load_hourly() / 1000
    d = g.load_daily()
    from statsmodels.tsa.seasonal import MSTL
    win = {}
    for mo in range(1, 13):
        t0 = pd.Timestamp(2025, mo, 1)
        t0 = t0 + pd.Timedelta(days=(7 - t0.dayofweek) % 7)
        if mo == 2:
            assert t0 == pd.Timestamp('2025-02-03')
        x = s.loc[t0:t0 + pd.Timedelta(hours=672 - 1)]
        res = MSTL(x.values, periods=(24, 168)).fit()
        win[str(t0.date())] = dict(tr=r(res.trend, 4), s24=r(res.seasonal[:, 0], 4), s168=r(res.seasonal[:, 1], 4))
    H = g.ro_holidays(range(2022, 2027))
    return dict(h0=str(s.index[0]), hv=r(s.values, 4), d0=str(d.index[0].date()), dv=r(d.values, 5), mstl=win,
                hol={str(k.date()): v for k, v in H.items()})


# ---------------------------------------------------------------- 16. Prophet
def prophet_block():
    from prophet import Prophet
    y = g.load_daily().loc[:g.CV_FIRST]
    H = g.ro_holidays(range(2022, 2027))
    out = {}
    for K in (3, 5, 10, 20):
        for hol in (1, 0):
            p = Prophet(holidays=pd.DataFrame({'ds': H.index, 'holiday': H.values}) if hol else None, daily_seasonality=False,
                        yearly_seasonality=K)
            p.fit(pd.DataFrame({'ds': y.index, 'y': y.values}))
            c = p.predict(p.make_future_dataframe(periods=0))
            wk = c.groupby(c['ds'].dt.dayofweek)['weekly'].mean()
            yr = c[(c['ds'] >= '2024-01-01') & (c['ds'] <= '2024-12-31')]
            eff = {}
            if hol:
                eff = {h: float(c.loc[c[h] != 0, h].mean()) if (c[h] != 0).any() else 0.0 for h in p.train_holiday_names}
            cps = [str(cp.date()) for cp in p.changepoints[np.abs(np.nanmean(p.params['delta'], axis=0)) > 0.01]]
            res = y.values - c['yhat'].values
            out[f'{K}{hol}'] = dict(tr=r(c['trend'], 4), wk=r(wk, 4), yr=r(yr['yearly'], 4), hol=eff, cp=cps,
                                    mae=float(np.mean(np.abs(res))))
    ref = out['101']
    assert np.allclose(ref['wk'], N['prophet']['weekly'], atol=2e-4), ref['wk']
    close(max(ref['yr']), N['prophet']['yearly_max'], 1e-3, 'ymax')
    for k, v in N['prophet']['hol'].items():
        close(ref['hol'][k], v, 2e-3, k)
    return dict(d0=str(y.index[0].date()), cfg=out)


# ---------------------------------------------------------------- 18-20. cross-validation results of the lecture
def cv_block():
    E = pd.read_csv(os.path.join(HERE, 'ch4_load_cv.csv'), parse_dates=['origin', 'date'])
    y = g.load_daily()
    tab, dm, best, scale, byh = g.cv_summary(E, y)
    for m, row in N['cv']['tab'].items():
        close(tab.loc[m, 'MAE'], row['MAE'], 1e-5, m)
    close(scale, N['cv']['scale'], 1e-9, 'scale')
    c = g.fig_combination(E, 'DHR', 'Prophet', save_it=False)
    for k, tol in (('mae_w0', 0.01), ('mae_w1', 0.01), ('mae_half', 0.01), ('w_best', 1e-9), ('corr', 1e-4)):   # the CSV keeps 5 decimals
        close(c[k], N['comb'][k], tol, k)
    origins = sorted(E['origin'].unique())
    models = ['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'TBATS', 'Prophet', 'Combination']
    fc = {m: [E[(E['model'] == m) & (E['origin'] == o)].sort_values('h')['fc'].round(5).tolist() for o in origins] for m in models}
    lf = E[E['origin'] == pd.Timestamp(N['loadfc']['origin'])]
    for m, v in N['loadfc']['mae'].items():
        close(lf[lf['model'] == m]['err'].abs().mean(), v, 1e-4, m)
    return dict(origins=[str(pd.Timestamp(o).date()) for o in origins], models=models, fc=fc)


# ================================================================= result tables and worked examples (the EXTRA blocks of the page)
def hegy_cv_grid(ns, nrep=4000):
    """5% critical values of HEGY by simulation (generate_all_charts.hegy_critical, vectorised: the same seed, the same
    draws, the same regressions) for every number of observations n the page offers, with and without the trend."""
    out = {}
    for trend in (True, False):
        for n in ns:
            rng = np.random.default_rng(g.SEED)
            E = rng.normal(size=(nrep, n + 4))
            Y = np.zeros_like(E)
            for t in range(4, n + 4):
                Y[:, t] = Y[:, t - 4] + E[:, t]
            L = lambda x, j: np.concatenate([np.full((x.shape[0], j), np.nan), x[:, :x.shape[1] - j]], axis=1)  # noqa: E731
            y1 = Y + L(Y, 1) + L(Y, 2) + L(Y, 3)
            y2 = -(Y - L(Y, 1) + L(Y, 2) - L(Y, 3))
            y3 = -(Y - L(Y, 2))
            d4 = Y - L(Y, 4)
            m = Y.shape[1]
            q = np.arange(m) % 4
            cols = [L(y1, 1), L(y2, 1), L(y3, 2), L(y3, 1), np.ones_like(Y)] + [np.broadcast_to((q == j).astype(float), Y.shape) for j in (1, 2, 3)]
            if trend:
                cols.append(np.broadcast_to(np.arange(m, dtype=float), Y.shape))
            X = np.stack(cols, axis=2)[:, 4:, :]
            z = d4[:, 4:]
            XtX = np.einsum('rtk,rtl->rkl', X, X)
            b = np.linalg.solve(XtX, np.einsum('rtk,rt->rk', X, z)[..., None])[..., 0]
            e = z - np.einsum('rtk,rk->rt', X, b)
            s2 = (e ** 2).sum(1) / (X.shape[1] - X.shape[2])
            V = np.linalg.inv(XtX)
            t1, t2 = b[:, 0] / np.sqrt(s2 * V[:, 0, 0]), b[:, 1] / np.sqrt(s2 * V[:, 1, 1])
            bb, VV = b[:, 2:4], V[:, 2:4, 2:4]
            F = np.einsum('rk,rkl,rl->r', bb, np.linalg.inv(VV), bb) / 2 / s2
            out[f'{n}{"t" if trend else "c"}'] = [float(np.quantile(t1, 0.05)), float(np.quantile(t2, 0.05)), float(np.quantile(F, 0.95))]
    ref = N['tests']['hegy_cv']
    assert np.allclose(out['75t'], [ref['t1'], ref['t2'], ref['F34']], atol=1e-9), (out['75t'], ref)
    return out


def seasonal_tests_grid():
    """OCSB and Canova-Hansen statistics with their critical values (pmdarima) and the KPSS number of differences, logs of
    the five series of the slide, from 2005 to the end of each year 2012-2025; the end 2019 is the table of the slide."""
    import pmdarima as pm
    from pmdarima.arima.seasonality import OCSBTest, CHTest
    res = {}
    for name, key, m in [('GDP', g.GDP_NSA, 4), ('Retail trade', g.RETAIL, 12), ('Industrial production', g.IPI, 12),
                         ('Tourism nights', g.TOUR, 12), ('HICP', g.HICP, 12)]:
        x0 = g.eurostat_log(key, '2005-01-01')
        row = {}
        for yr in range(2012, 2026):
            x = x0.loc[:f'{yr}-12-31'].values
            o, c = OCSBTest(m=m), CHTest(m=m)
            so, co = float(o._compute_test_statistic(x)), float(c._sd_test(x, m))
            row[str(yr)] = dict(ocsb=[so, float(o._calc_ocsb_crit_val(m)), int(pm.arima.nsdiffs(x, m=m, test='ocsb'))],
                                ch=[co, float(c._calc_ch_crit_val(m)), int(pm.arima.nsdiffs(x, m=m, test='ch'))],
                                d=int(pm.arima.ndiffs(x, test='kpss')), n=int(len(x)))
        res[name] = row
    for ref in N['tests']['table']:
        r = res[ref['series']]['2019']
        assert (r['ocsb'][2], r['ch'][2], r['d']) == (ref['ocsb'], ref['ch'], ref['d_kpss']), (ref, r)
    return res


def airline_extra():
    """For every origin of the airline block: the last 13 residuals of the fit (ln y) and Ljung-Box Q*(24) on 22 degrees of
    freedom, for the forecast recursion and the estimates table."""
    ly = np.log(g.airline())
    out = {}
    for yr in list(range(1952, 1959)) + ['full']:
        tr = ly if yr == 'full' else ly.loc[:f'{yr}-12-01']
        m = g.fit_sarima(tr, (0, 1, 1), (0, 1, 1, 12))
        lb = g.ljung_box(m.resid[13:], 24, 22)
        out[str(yr)] = dict(e=r(m.resid[-13:], 6), lb=[lb['lb'], lb['lb_p']], th=float(m.params[0]), Th=float(m.params[1]),
                            th_se=float(m.bse[0]), Th_se=float(m.bse[1]), s=float(np.sqrt(m.params[2])), n=int(len(tr)))
    close(out['full']['lb'][1], N['airfc']['lb24']['lb_p'], 1e-9, 'air lb full')
    return out


def gdp_cv_errors():
    """Rolling-origin errors of the GDP comparison (generate_all_charts.gdp_cv, same models and origins) on the slide
    version of GDP: one row per origin, errors for h = 1..4 of each method."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = 100 * g.eurostat_log(g.GDP_NSA, g.GDP_START)
    H, E = 4, {'SARIMA': [], 'ETS': [], 'Seasonal naive': [], 'Combination': []}
    org = []
    for i in range(len(y)):
        o = y.index[i]
        if o < pd.Timestamp('2012-10-01') or i + H >= len(y):
            continue
        tr, act = y.iloc[:i + 1], y.iloc[i + 1:i + 1 + H].values
        f = {'SARIMA': g.fit_sarima(tr, (0, 1, 1), (0, 1, 1, 4)).forecast(H),
             'ETS': ExponentialSmoothing(tr.values, trend='add', damped_trend=True, seasonal='add', seasonal_periods=4).fit().forecast(H),
             'Seasonal naive': tr.values[-4:][:H] + 0.0}
        f['Combination'] = (f['SARIMA'] + f['ETS']) / 2
        for k in E:
            E[k].append(r(act - f[k], 5))
        org.append(str(o.date()))
    ref = N['gdpcv']
    for m in E:
        for h in (1, 4):
            close(np.mean(np.abs([e[h - 1] for e in E[m]])), ref['tab'][m][str(h)]['mae'], 2e-3, (m, h))
    return dict(origins=org, err=E)


def extra_blocks():
    ns = sorted({T - 4 - k for T in list(range(36, 105, 4)) + [106] for k in range(0, 5)})
    return dict(hegy_cv=hegy_cv_grid(ns), stests=seasonal_tests_grid(), air2=airline_extra(), gdpcv=gdp_cv_errors())


def main_extra():
    """Only the inputs of the result tables and worked examples, merged into the existing ch4_data.js."""
    txt = open(OUT).read()
    head, js = txt.split('window.CH4_DATA = ', 1)
    d = json.loads(js.rstrip().rstrip(';'))
    d['numbers']['gdpm'] = N['gdpm']
    d['numbers']['gdpcv'] = N['gdpcv']
    d.update(extra_blocks())
    with open(OUT, 'w') as fh:
        fh.write(head + 'window.CH4_DATA = ' + json.dumps(d, separators=(',', ':'), default=str) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


def main():
    keys =('series', 'splot', 'gdpdiff', 'tests', 'theory', 'air', 'airfc', 'gdpm', 'gdpdiag', 'gdpfc', 'sa', 'fourier', 'easter',
            'hourly', 'mstl', 'daily', 'prophet', 'tbats', 'dhr', 'loadfc', 'comb', 'cv')
    d = dict(numbers={k: N[k] for k in keys})
    d['numbers']['cv'] = {k: v for k, v in N['cv'].items() if k != 'dm'}
    d['numbers']['cv']['dm'] = {k: v for k, v in N['cv']['dm'].items() if k.endswith('|se')}
    steps = [('series', series_block), ('gdpdiff', gdpdiff_block), ('air', airline_block), ('gdp', gdp_models_block),
             ('sa', sa_block), ('fourier', fourier_block), ('easter', easter_block), ('load', load_block),
             ('prophet', prophet_block), ('cv', cv_block)]
    for k, f in steps:
        print(k, flush=True)
        d[k] = f()
    d['numbers']['gdpm'] = N['gdpm']
    d['numbers']['gdpcv'] = N['gdpcv']
    d.update(extra_blocks())
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 4 interactive page, exported by Quantlets/Ch_04/export_interactive.py.\n')
        fh.write('window.CH4_DATA = ' + json.dumps(d, separators=(',', ':'), default=str) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    import sys
    main_extra() if '--extra' in sys.argv else main()
