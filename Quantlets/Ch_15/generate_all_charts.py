"""
generate_all_charts.py -- charts and numbers of Chapter 15 (TSA): review and exam preparation
==============================================================================================
Course data (tsa_data.py), chart style (tsa_style.py), statsmodels (SARIMAX, unit-root tests), arch (GARCH).
  * Box-Jenkins end to end on one Romanian series: the monthly HICP of Romania (Eurostat prc_hicp_minr, 2015 = 100),
    January 2005 (inflation targeting) to the last month:
      - data       -- the index, monthly inflation 100 dln P and annual inflation 100 d12 ln P;
      - tests      -- ADF and KPSS on ln P, d ln P, d12 ln P and d d12 ln P (Chapters 1, 3, 4);
      - identify   -- ACF and PACF of z = d d12 ln P (Chapters 1, 2, 4);
      - estimate   -- every SARIMA(p,1,q)(P,1,Q)12 with p, q <= 2, P, Q <= 1 on the training sample (to August 2024):
                      AICc, BIC, Ljung-Box of the residuals (Chapters 2, 4);
      - check      -- residual diagnostics of the chosen model: Ljung-Box, Jarque-Bera, the largest residuals;
      - forecast   -- one-step forecasts of monthly inflation on the test sample (September 2024 onwards) against the
                      seasonal naive and naive methods (RMSE, MAE, MASE, Diebold-Mariano with the HLN correction),
                      then 12-month forecasts of annual inflation with 95% intervals from the full sample (Chapter 4);
  * exam problem 3 -- an AR(1) x SAR(1) model of z = d d12 ln P (the form of the EViews output DLOG(IPC,1,12) of
                      past exams): estimates, Ljung-Box, the last values and the one-step forecast computed by hand;
  * the course in one notebook -- short functions for the lecture notebook: SES and Holt-Winters on the HICP, a
                      GARCH(1,1)-t for the BET, a VAR(2) and Granger tests for Romanian inflation and ROBOR 3M, the local
                      Whittle d of |r| of the BET, a local level model of monthly inflation.
Output: charts/tsa_ch15_*.pdf/.png, Quantlets/Ch_15/ch15_numbers.json
Run:  python3 Quantlets/Ch_15/generate_all_charts.py
Time Series Analysis - Daniel Traian PELE
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from tsa_data import log_returns, read_eurostat                         # noqa: E402
import tsa_style as st                                                   # noqa: E402
import statsmodels.api as sm                                             # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf          # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX                   # noqa: E402

warnings.filterwarnings('ignore')
HICP = ('prc_hicp_minr', 'M.I15.TOTAL.RO')       # Romanian HICP, all items, 2015 = 100 (Eurostat)
ROBOR = ('irt_st_m', 'M.IRT_M3.RO')              # ROBOR 3M, % per year (Eurostat)
START = '2005-01-01'                              # inflation targeting in Romania since August 2005
TRAIN_END = '2024-08-01'                          # last month of the training sample
S = 12                                            # seasonal period (months)
MAXLAG = 36                                       # lags of the correlograms
IDENT = ((1, 1, 0), (0, 1, 1, S))                 # the model suggested by the correlogram of z
H = 12                                            # forecast horizon (months)


# =============================================================================
# DATA AND HELPERS
# =============================================================================
def hicp(start=START):
    """Romanian HICP (Eurostat, monthly, 2015 = 100) from `start`."""
    h = read_eurostat(*HICP).loc[start:].astype(float)
    h.index = pd.DatetimeIndex(h.index, freq='MS')
    return h.rename('hicp')


def transforms(h):
    """100 ln P and its differences: monthly inflation d ln P, annual inflation d12 ln P, z = d d12 ln P."""
    y = 100 * np.log(h)
    return pd.DataFrame({'y': y, 'm': y.diff(), 'a': y.diff(S), 'z': y.diff().diff(S)})


def save(name, save_it=True):
    if save_it:
        st.check_no_grey(plt.gcf())
        st.save_fig(name)
    else:
        plt.show()


def adf_kpss(x, reg):
    """ADF (lags by AIC) and KPSS (automatic bandwidth) with the same deterministic terms ('c' or 'ct')."""
    a = adfuller(x, regression=reg, autolag='AIC')
    k = kpss(x, regression=reg, nlags='auto')
    return {'reg': reg, 'n': int(len(x)), 'adf': float(a[0]), 'adf_p': float(a[1]), 'adf_lags': int(a[2]),
            'adf_cv1': float(a[4]['1%']), 'adf_cv5': float(a[4]['5%']), 'adf_cv10': float(a[4]['10%']),
            'kpss': float(k[0]), 'kpss_p': float(k[1]), 'kpss_cv5': float(k[3]['5%'])}


def ljung_box(e, lags, df_adj=0):
    """Ljung-Box Q(m) for each m in `lags`, with m - df_adj degrees of freedom."""
    e = np.asarray(e, float)
    T = len(e)
    r = acf(e, nlags=max(lags), fft=False)
    out = {}
    for m in lags:
        q = T * (T + 2) * np.sum(r[1:m + 1] ** 2 / (T - np.arange(1, m + 1)))
        out[str(m)] = {'q': float(q), 'df': int(m - df_adj), 'p': float(stats.chi2.sf(q, m - df_adj))}
    return out


def dm_test(e1, e2, h=1):
    """Diebold-Mariano test on squared errors, HLN correction, t(n - 1) p-value; a negative mean favours e1."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    g = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    v = (g[0] + 2 * sum(g[1:])) / n
    dm = dbar / np.sqrt(v)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * dm
    return {'dbar': float(dbar), 'dm': float(dm), 'hln': float(hln), 'p': float(2 * stats.t.sf(abs(hln), n - 1)), 'n': int(n)}


def fit_sarima(y, order, sorder):
    return SARIMAX(y, order=order, seasonal_order=sorder, trend='n').fit(disp=False)


def label(order, sorder):
    return f'({order[0]},{order[1]},{order[2]})({sorder[0]},{sorder[1]},{sorder[2]})'


# =============================================================================
# BOX-JENKINS ON THE ROMANIAN HICP
# =============================================================================
def fig_bj_data(save_it=True):
    """Step 1: plot the data -- the index and the two inflation rates."""
    h = hicp()
    d = transforms(h)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={'height_ratios': [1, 1.25]})
    ax[0].plot(h.index, h, color=st.MainBlue, label='HICP of Romania, 2015 = 100')
    ax[0].set_ylabel('index')
    ax[1].bar(d.index, d['m'], width=25, color=st.Amber, label='monthly inflation, 100 Δln P (%)')
    ax[1].plot(d.index, d['a'], color=st.IDAred, lw=1.6, label='annual inflation, 100 Δ12 ln P (%)')
    ax[1].axhline(0, color=st.DarkText, lw=0.6)
    ax[1].set_ylabel('%')
    st.fig_legend_bottom(fig, ncol=3, y=0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch15_bj_data', save_it)
    a = d['a'].dropna()
    m = d['m'].dropna()
    return {'n': int(len(h)), 'first': str(h.index[0].date()), 'last': str(h.index[-1].date()),
            'last_index': float(h.iloc[-1]), 'last_a': float(a.iloc[-1]), 'last_m': float(m.iloc[-1]),
            'max_a': float(a.max()), 'max_a_d': str(a.idxmax().date()), 'min_a': float(a.min()),
            'min_a_d': str(a.idxmin().date()), 'mean_m': float(m.mean()), 'mean_a': float(a.mean()),
            'a_2024_08': float(a.loc[TRAIN_END]), 'a_2025_07': float(a.loc['2025-07-01']),
            'a_2025_08': float(a.loc['2025-08-01']), 'm_2025_08': float(m.loc['2025-08-01']),
            'm_by_month': {int(k): float(v) for k, v in m.groupby(m.index.month).mean().items()}}


def bj_tests():
    """Step 2: unit-root (ADF) and stationarity (KPSS) tests on four transformations."""
    d = transforms(hicp())
    return {'y': adf_kpss(d['y'].dropna(), 'ct'), 'm': adf_kpss(d['m'].dropna(), 'c'),
            'a': adf_kpss(d['a'].dropna(), 'c'), 'z': adf_kpss(d['z'].dropna(), 'c')}


def fig_bj_acf(save_it=True):
    """Step 3: identification from the ACF and PACF of z = d d12 ln P."""
    z = transforms(hicp())['z'].dropna()
    r = acf(z, nlags=MAXLAG, fft=False)[1:]
    p = pacf(z, nlags=MAXLAG)[1:]
    band = 1.96 / np.sqrt(len(z))
    lags = np.arange(1, MAXLAG + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for a, v, nm in ((ax[0], r, 'ACF'), (ax[1], p, 'PACF')):
        col = [st.IDAred if k % S == 0 else st.MainBlue for k in lags]
        a.bar(lags, v, width=0.55, color=col)
        a.axhline(band, color=st.Forest, ls='--', lw=1, label='±1.96/√T')
        a.axhline(-band, color=st.Forest, ls='--', lw=1)
        a.axhline(0, color=st.DarkText, lw=0.6)
        a.set_title(f'{nm} of z = ΔΔ12 ln P')
        a.set_xlabel('lag (months); seasonal lags 12, 24, 36 in red')
        a.set_xticks([1, 6, 12, 18, 24, 30, 36])
    st.fig_legend_bottom(fig, ncol=1, y=0.06)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save('tsa_ch15_bj_acf', save_it)
    out = {'n': int(len(z)), 'band': float(band), 'r': [float(x) for x in r], 'p': [float(x) for x in p]}
    out['out_r'] = [int(k) for k, v in zip(lags, r) if abs(v) > band]
    out['out_p'] = [int(k) for k, v in zip(lags, p) if abs(v) > band]
    return out


def bj_grid(pmax=2, qmax=2):
    """Step 4: every SARIMA(p,1,q)(P,1,Q)12 with p, q <= 2 and P, Q <= 1 on the training sample."""
    y = transforms(hicp())['y'].loc[:TRAIN_END]
    rows = []
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            for P in (0, 1):
                for Q in (0, 1):
                    o, so = (p, 1, q), (P, 1, Q, S)
                    try:
                        m = fit_sarima(y, o, so)
                    except Exception:            # noqa: BLE001  (a model that fails to converge is skipped)
                        continue
                    k = p + q + P + Q
                    e = m.resid[S + 1:]
                    lb = ljung_box(e, [12, 24], df_adj=k)
                    rows.append({'model': label(o, so), 'order': list(o), 'sorder': list(so), 'k': k + 1,
                                 'loglik': float(m.llf), 'aicc': float(m.aicc), 'bic': float(m.bic),
                                 'lb12_p': lb['12']['p'], 'lb24_p': lb['24']['p'],
                                 'converged': bool(m.mle_retvals.get('converged', True))})
    rows.sort(key=lambda r: r['bic'])
    ok = [r for r in rows if r['lb24_p'] > 0.05 and r['lb12_p'] > 0.05 and r['converged']]
    ident = next(r for r in rows if r['model'] == label(*IDENT))
    return {'n_train': int(len(y)), 'train_first': str(y.index[0].date()), 'train_last': str(y.index[-1].date()),
            'n_models': len(rows), 'rows': rows, 'best_bic': rows[0]['model'],
            'best_aicc': min(rows, key=lambda r: r['aicc'])['model'], 'chosen': ok[0]['model'],
            'chosen_order': ok[0]['order'], 'chosen_sorder': ok[0]['sorder'], 'ident': ident}


def model_summary(m, k):
    """Estimates, standard errors and residual checks of a fitted SARIMAX model with k ARMA parameters."""
    e = m.resid[S + 1:]
    zs = e / np.sqrt(m.params['sigma2'])
    lb = ljung_box(e, [12, 24], df_adj=k)
    jb = stats.jarque_bera(e)
    big = zs.abs().sort_values(ascending=False).head(3)
    return {'params': {k_: float(v) for k_, v in m.params.items()}, 'se': {k_: float(v) for k_, v in m.bse.items()},
            'z': {k_: float(v) for k_, v in m.tvalues.items()}, 'p': {k_: float(v) for k_, v in m.pvalues.items()},
            'loglik': float(m.llf), 'aicc': float(m.aicc), 'bic': float(m.bic), 'lb': lb, 'jb': float(jb[0]),
            'jb_p': float(jb[1]), 'skew': float(stats.skew(e)), 'kurt': float(stats.kurtosis(e, fisher=False)),
            'big': [[str(i.date()), float(zs.loc[i])] for i in big.index], 'n': int(m.nobs),
            'ar_roots': [float(abs(r)) for r in m.arroots], 'ma_roots': [float(abs(r)) for r in m.maroots]}


def fig_bj_diag(grid=None, save_it=True):
    """Step 5: diagnostics of the identified model and of the chosen model (training sample)."""
    grid = grid or bj_grid()
    y = transforms(hicp())['y'].loc[:TRAIN_END]
    o, so = tuple(grid['chosen_order']), tuple(grid['chosen_sorder'])
    m = fit_sarima(y, o, so)
    mi = fit_sarima(y, *IDENT)
    k = sum(o) - 1 + so[0] + so[2]
    out = {'chosen': model_summary(m, k), 'ident': model_summary(mi, 2), 'model': label(o, so)}
    e = m.resid[S + 1:]
    zs = e / np.sqrt(m.params['sigma2'])
    ei = mi.resid[S + 1:]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    ax[0].plot(zs.index, zs, color=st.MainBlue, lw=0.9, label=f'standardised residuals, {label(o, so)}')
    ax[0].axhline(1.96, color=st.Forest, ls='--', lw=0.8)
    ax[0].axhline(-1.96, color=st.Forest, ls='--', lw=0.8)
    ax[0].set_title('residuals over time')
    ax[0].set_xticks(pd.to_datetime(['2008-01-01', '2012-01-01', '2016-01-01', '2020-01-01', '2024-01-01']))
    ax[0].set_xticklabels(['2008', '2012', '2016', '2020', '2024'])
    lags = np.arange(1, 25)
    band = 1.96 / np.sqrt(len(e))
    ax[1].bar(lags - 0.18, acf(ei, nlags=24, fft=False)[1:], width=0.36, color=st.Amber, label=f'{label(*IDENT)} (identified)')
    ax[1].bar(lags + 0.18, acf(e, nlags=24, fft=False)[1:], width=0.36, color=st.MainBlue, label=f'{label(o, so)} (chosen)')
    ax[1].axhline(band, color=st.Forest, ls='--', lw=1)
    ax[1].axhline(-band, color=st.Forest, ls='--', lw=1)
    ax[1].axhline(0, color=st.DarkText, lw=0.6)
    ax[1].set_title('ACF of the residuals')
    ax[1].set_xticks([1, 6, 12, 18, 24])
    sm.qqplot(zs, line='45', ax=ax[2], markersize=3, markerfacecolor=st.MainBlue, markeredgecolor=st.MainBlue)
    for ln in ax[2].get_lines()[1:]:
        ln.set_color(st.IDAred)
    ax[2].set_title('QQ plot against N(0, 1)')
    ax[2].set_xlabel('N(0, 1) quantiles')
    ax[2].set_ylabel('residual quantiles')
    st.fig_legend_bottom(fig, ncol=3, y=0.07)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save('tsa_ch15_bj_diag', save_it)
    return out


def fig_bj_forecast(grid=None, save_it=True):
    """Step 6: one-step forecasts on the test sample against benchmarks; 12-month forecasts of annual inflation."""
    grid = grid or bj_grid()
    d = transforms(hicp())
    y = d['y']
    o, so = tuple(grid['chosen_order']), tuple(grid['chosen_sorder'])
    m = fit_sarima(y.loc[:TRAIN_END], o, so)
    mf = m.apply(y)                                           # the same parameters, filtered over the whole sample
    test = y.index[y.index > TRAIN_END]
    pred = mf.get_prediction(start=test[0], end=test[-1], dynamic=False).predicted_mean
    w = d['m']
    f_sar = (pred - y.shift(1).loc[test]).rename('SARIMA')    # forecast of monthly inflation
    f_sn = w.shift(S).loc[test].rename('Seasonal naive')
    f_nv = w.shift(1).loc[test].rename('Naive')
    act = w.loc[test]
    wt = w.loc[:TRAIN_END].dropna()
    scale = np.mean(np.abs(wt.values[S:] - wt.values[:-S]))
    acc = {}
    for f in (f_sar, f_sn, f_nv):
        e = act - f
        acc[f.name] = {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
                       'mase': float(np.mean(np.abs(e)) / scale)}
    dm_sn = dm_test(act - f_sar, act - f_sn)
    dm_nv = dm_test(act - f_sar, act - f_nv)
    # 12-month forecasts of annual inflation from the full sample
    mfull = fit_sarima(y, o, so)
    fc = mfull.get_forecast(H)
    mean, se = fc.predicted_mean, fc.se_mean
    base = y.shift(S).reindex(mean.index)
    base.loc[:] = [y.loc[t - pd.DateOffset(months=S)] for t in mean.index]
    a_f = mean - base
    lo, hi = a_f - 1.96 * se, a_f + 1.96 * se
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1, 1.2]})
    ax[0].plot(act.index, act, 'o-', color=st.DarkText, ms=3.5, lw=1, label='actual')
    ax[0].plot(f_sar.index, f_sar, 's-', color=st.MainBlue, ms=3.5, lw=1.2, label=f'SARIMA{label(o, so)}')
    ax[0].plot(f_sn.index, f_sn, '^--', color=st.Amber, ms=3.5, lw=1, label='seasonal naive')
    ax[0].axhline(0, color=st.DarkText, lw=0.5)
    ax[0].set_title('one-step forecasts of monthly inflation (%)')
    ax[0].tick_params(axis='x', rotation=30)
    ah = d['a'].loc['2015-01-01':]
    ax[1].plot(ah.index, ah, color=st.IDAred, lw=1.5, label='annual inflation (%)')
    ax[1].plot(a_f.index, a_f, color=st.MainBlue, lw=2, label='forecast')
    ax[1].fill_between(a_f.index, lo, hi, color=st.MainBlue, alpha=0.2, label='95% interval')
    ax[1].axhline(2.5, color=st.Forest, ls='--', lw=1, label='BNR target 2.5%')
    ax[1].set_title(f'annual inflation: forecast to {a_f.index[-1].strftime("%B %Y")}')
    st.fig_legend_bottom(fig, ncol=4, y=0.08)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save('tsa_ch15_bj_forecast', save_it)
    big = (act - f_sar).abs().sort_values(ascending=False)
    return {'model': label(o, so), 'n_test': int(len(test)), 'test_first': str(test[0].date()),
            'test_last': str(test[-1].date()), 'acc': acc, 'scale': float(scale), 'dm_sn': dm_sn, 'dm_nv': dm_nv,
            'worst': [[str(i.date()), float(act.loc[i]), float(f_sar.loc[i])] for i in big.index[:2]],
            'origin': str(y.index[-1].date()), 'fc_first': str(a_f.index[0].date()), 'fc_last': str(a_f.index[-1].date()),
            'a1': float(a_f.iloc[0]), 'a1_lo': float(lo.iloc[0]), 'a1_hi': float(hi.iloc[0]),
            'a6': float(a_f.iloc[5]), 'a12': float(a_f.iloc[-1]), 'a12_lo': float(lo.iloc[-1]), 'a12_hi': float(hi.iloc[-1]),
            'se1': float(se.iloc[0]), 'se12': float(se.iloc[-1]), 'last_a': float(d['a'].iloc[-1]),
            'params_full': {k: float(v) for k, v in mfull.params.items()}}


# =============================================================================
# EXAM PROBLEM 3: AR(1) x SAR(1) FOR z = d d12 ln P, THE FORECAST BY HAND
# =============================================================================
def exam_p3():
    """An AR(1) x SAR(1) model without constant for z = d d12 ln P (the EViews form DLOG(IPC,1,12) with AR(1) and
    SAR(12)), estimated on the full sample; the one-step forecast of z and of annual inflation by hand."""
    d = transforms(hicp())
    z = d['z'].dropna()
    m = SARIMAX(z, order=(1, 0, 0), seasonal_order=(1, 0, 0, S), trend='n').fit(disp=False)
    phi, Phi = float(m.params['ar.L1']), float(m.params['ar.S.L12'])
    zT, z11, z12 = float(z.iloc[-1]), float(z.iloc[-12]), float(z.iloc[-13])
    hand = phi * zT + Phi * z11 - phi * Phi * z12
    sm_f = float(m.get_forecast(1).predicted_mean.iloc[0])
    aT = float(d['a'].iloc[-1])
    e = m.resid[S + 1:]
    lb = ljung_box(e, [12, 24], df_adj=2)
    t = z.index[-1]
    return {'n': int(m.nobs), 'first': str(z.index[0].date()), 'last': str(t.date()),
            'phi': phi, 'Phi': Phi, 'se_phi': float(m.bse['ar.L1']), 'se_Phi': float(m.bse['ar.S.L12']),
            'z_phi': float(m.tvalues['ar.L1']), 'z_Phi': float(m.tvalues['ar.S.L12']),
            'p_phi': float(m.pvalues['ar.L1']), 'p_Phi': float(m.pvalues['ar.S.L12']),
            'sigma2': float(m.params['sigma2']), 'loglik': float(m.llf), 'aic': float(m.aic), 'bic': float(m.bic),
            'lb': lb, 'zT': zT, 'zT11': z11, 'zT12': z12, 'dT': str(t.date()),
            'dT11': str(z.index[-12].date()), 'dT12': str(z.index[-13].date()),
            'aT': aT, 'z_next_hand': hand, 'z_next_sm': sm_f, 'a_next': aT + hand,
            'next': str((t + pd.DateOffset(months=1)).date()), 'inv_ar': [1 / abs(r) for r in m.arroots][:2]}


# =============================================================================
# THE COURSE IN ONE NOTEBOOK (lecture notebook; short functions, one per block of chapters)
# =============================================================================
def course_smoothing():
    """Chapter 0: SES and Holt-Winters for the HICP (log), trained to August 2024, tested on the following months."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = transforms(hicp())['y']
    tr, te = y.loc[:TRAIN_END], y.loc[y.index > TRAIN_END]
    ses = ExponentialSmoothing(tr, trend=None, seasonal=None).fit()
    hw = ExponentialSmoothing(tr, trend='add', seasonal='add', seasonal_periods=S).fit()
    out = {}
    for nm, m in (('SES', ses), ('Holt-Winters', hw)):
        f = m.forecast(len(te))
        out[nm] = {'rmse_level': float(np.sqrt(np.mean((te.values - f.values) ** 2))),
                   'alpha': float(m.params['smoothing_level'])}
    return out


def course_garch(name='bet'):
    """Chapter 5: GARCH(1,1) with Student-t innovations for daily returns (arch package)."""
    from arch import arch_model
    r = log_returns(name)
    m = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
    a, b = float(m.params['alpha[1]']), float(m.params['beta[1]'])
    return {'n': int(len(r)), 'alpha': a, 'beta': b, 'persistence': a + b, 'half_life': float(np.log(0.5) / np.log(a + b)),
            'nu': float(m.params['nu'])}


def course_var():
    """Chapters 6-7: VAR for monthly Romanian annual inflation and ROBOR 3M; lag order by BIC; Granger tests."""
    from statsmodels.tsa.api import VAR
    pi = transforms(read_eurostat(*HICP).loc['2004-01-01':].astype(float))['a']
    i = read_eurostat(*ROBOR)
    dd = pd.concat([pi.rename('pi'), i.rename('i')], axis=1).loc[START:].dropna()
    dd.index = pd.DatetimeIndex(dd.index, freq='MS')
    sel = VAR(dd).select_order(12)
    p = max(int(sel.bic), 1)
    r = VAR(dd).fit(p)
    g1 = r.test_causality('i', ['pi'], kind='f')
    g2 = r.test_causality('pi', ['i'], kind='f')
    return {'n': int(r.nobs), 'p': p, 'stable': bool(r.is_stable()), 'pi_to_i': {'F': float(g1.test_statistic), 'p': float(g1.pvalue)},
            'i_to_pi': {'F': float(g2.test_statistic), 'p': float(g2.pvalue)}}


def local_whittle(x, m=None):
    """Local Whittle estimate of d (Chapter 8) with bandwidth m = T^0.65."""
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    m = m or int(T ** 0.65)
    lam = 2 * np.pi * np.arange(1, m + 1) / T
    I = np.abs(np.fft.fft(x)[1:m + 1]) ** 2 / (2 * np.pi * T)
    from scipy.optimize import minimize_scalar

    def R(d):
        return np.log(np.mean(lam ** (2 * d) * I)) - 2 * d * np.mean(np.log(lam))
    return float(minimize_scalar(R, bounds=(-0.49, 0.99), method='bounded').x)


def course_memory():
    """Chapter 8: local Whittle d of the BET returns and of their absolute values."""
    r = log_returns('bet')
    return {'d_r': local_whittle(r.values), 'd_abs': local_whittle(np.abs(r.values))}


def course_state_space():
    """Chapter 10: local level model of monthly inflation (UnobservedComponents) and the equivalent SES weight."""
    w = transforms(hicp())['m'].dropna()
    m = sm.tsa.UnobservedComponents(w, 'llevel').fit(disp=False)
    s2e, s2n = float(m.params['sigma2.irregular']), float(m.params['sigma2.level'])
    q = s2n / s2e
    P = (q + np.sqrt(q ** 2 + 4 * q)) / 2
    return {'s2_eps': s2e, 's2_eta': s2n, 'q': q, 'K': float(P / (P + 1))}


if __name__ == '__main__':
    st.apply()
    N = {}
    only = sys.argv[1:]
    path = os.path.join(HERE, 'ch15_numbers.json')
    if only and os.path.exists(path):
        N = json.load(open(path))
    G = None
    for name, f in [('data', fig_bj_data), ('tests', bj_tests), ('acf', fig_bj_acf), ('grid', bj_grid),
                    ('diag', None), ('fc', None), ('p3', exam_p3)]:
        if only and name not in only:
            continue
        print(name)
        if name == 'grid':
            G = N[name] = bj_grid()
        elif name == 'diag':
            N[name] = fig_bj_diag(G or N.get('grid'))
        elif name == 'fc':
            N[name] = fig_bj_forecast(G or N.get('grid'))
        else:
            N[name] = f()
        with open(path, 'w') as fh:
            json.dump(N, fh, indent=1, default=float)
    print('written ch15_numbers.json')
